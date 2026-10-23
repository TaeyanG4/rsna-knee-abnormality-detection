"""PL production supervision v1.1.1, 2026-10-14.

Changelog 1.1.1: admit registered v1.0 and v1.1 production scratch roots.
Changelog: reuse H-B-147 owned-process supervision via its H-B-150 extension,
with measured full-shape memory admission and the 30-minute throughput gate.
No CUDA initialization; invoke inside gpu_job.py. Completed attempts are never
overwritten and partial model outputs are never removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb150_watchdog_v1 as supervisor
from tools.hb150_io_v1 import atomic_json
from tools.hb147_progress_v1 import Progress, ProgressWriter
from tools.job_runtime import file_lock

VERSION = "1.1.1"
WORK = ROOT / ".local/scratch/astra-b4/PL-PRODUCTION-v1"
PRODUCTION_ROOTS = (WORK, ROOT / ".local/scratch/astra-b4/PL-PRODUCTION-v11")
Policy = supervisor.Policy


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_memory(path, *, go=None, arm=None):
    """Validate B's pinned post-screen measurement artifact, never live outputs.

    Both full-size training and flip/TTA peak measurements are mandatory. The
    artifact is prepared after screen completion; this function does not infer
    healthy rates or safe VRAM from successful exit codes.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value.get("schema") == "pl-production-memory/v1", "memory evidence schema")
    require("4070 Ti SUPER" in value.get("device_name", ""), "memory evidence must measure the 4070 Ti SUPER")
    total = value.get("total_memory_mib", 0)
    require(isinstance(total, (int, float)) and 16000 <= total <= 16400, "measured 16 GB device required")
    require(value.get("selected_arm") in ("global", "fill"), "winning PL arm required")
    if arm is not None:
        require(value["selected_arm"] == arm, "memory evidence selected arm differs")
    require(isinstance(value.get("go_sha256"), str) and len(value["go_sha256"]) == 64, "memory GO binding absent")
    if go is not None:
        require(value["go_sha256"] == sha(go), "memory GO binding changed")
    shape = value.get("full_shape", {})
    require(shape.get("batch_size") == 8 and shape.get("k_train") == 12
            and shape.get("k_eval") == 36 and shape.get("resolution") == 224
            and shape.get("tta") is True, "full production fit and flip/TTA shape required")
    measurements = value.get("measurements", [])
    require({m.get("phase") for m in measurements} >= {"fit", "tta"}, "fit and TTA measurements required")
    for measurement in measurements:
        peak = measurement.get("peak_reserved_bytes")
        require(measurement.get("full_shape") is True and isinstance(peak, (int, float))
                and math.isfinite(peak) and 0 < peak <= 10 * 1024**3,
                "measured peak reserved exceeds 10 GiB or is not full shape")
    rates = value.get("healthy_rates", {})
    rate = rates.get("train_step")
    require(isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0,
            "measured healthy committed-update rate required")
    pins = value.get("source_pins", [])
    require(bool(pins), "measured source receipt pins required")
    for pin in pins:
        require(sha(pin["path"]) == pin["sha256"], "memory source receipt changed")
    return value


class ProductionProgress(Progress):
    """Only completed work advances deadlines; failed telemetry is nonfatal."""
    def completed(self, phase, detail=None, *, throttle=False):
        if phase == "training_step" and isinstance(detail, dict) and "update" in detail:
            detail = dict(detail, progress_kind="train_step",
                          cumulative_units=detail["update"] - detail.get("skips", 0))
        return super().completed(phase, detail, throttle=throttle)

    def _write(self, event, detail):
        if self.path is None:
            return
        atomic_json(self.path, dict(schema="hb147-progress/v1", token=self.token,
                    pid=os.getpid(), sequence=self.sequence, phase=self.phase,
                    event=event, completed_at=self.completed_at, detail=detail), critical=False)
        self.last_write = time.monotonic()


PROGRESS = ProductionProgress(os.environ.get("HB147_HEARTBEAT"), os.environ.get("HB147_TOKEN"))


def supervise(command, attempt, **kwargs):
    return supervisor.supervise(command, attempt, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--memory-evidence", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--arm", choices=("global", "fill"), required=True)
    parser.add_argument("--cpu", action="store_true", help="CPU supervisor integration tests only")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    attempt, budget = args.attempt.resolve(), args.budget.resolve()
    require(any(attempt.is_relative_to(root.resolve()) and budget.is_relative_to(root.resolve())
                for root in PRODUCTION_ROOTS),
            "attempt and budget must stay inside production scratch")
    evidence = validate_memory(args.memory_evidence, go=args.go, arm=args.arm)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    require(bool(command), "child command required")
    Policy(0, healthy_rates=evidence["healthy_rates"])
    with file_lock(budget.with_suffix(".lock"), blocking=False):
        state = json.loads(budget.read_text()) if budget.exists() else {"used_seconds": 0.}
        if state.get("active_since") is not None:
            state["used_seconds"] += max(0., time.time() - state["active_since"])
        remaining = 43200. - state["used_seconds"]
        require(remaining > 0, "shared production 12-hour cap exhausted")
        if attempt.exists():
            raise FileExistsError("partial attempt preserved; explicit new attempt name required")
        state.update(active_since=time.time(), attempt=str(attempt),
                     memory_evidence_sha256=sha(args.memory_evidence), go_sha256=sha(args.go))
        atomic_json(budget, state)
        start = time.monotonic()
        try:
            return supervise(command, attempt, total=remaining, cpu=args.cpu,
                             healthy_rates=evidence["healthy_rates"])
        finally:
            state.update(used_seconds=state["used_seconds"] + time.monotonic()-start,
                         active_since=None)
            atomic_json(budget, state)


if __name__ == "__main__":
    raise SystemExit(main())
