"""H-B-166 supervisor v1.2.2 (2026-10-15): reviewed hb155_watchdog_v1 policy, H-B-166 namespace.

Reuses hb155_watchdog_v1.Policy/supervise unchanged (15-min no-progress,
10-min WDDM stall, 30-min <25 % healthy-throughput abort, owned-tree cleanup,
immutable named attempts, NO wall-clock cap). Only the stage path and the
command allowlist differ. The child still receives HB155_HEARTBEAT/HB155_TOKEN
(inherited contract read by tools/hb166_allfold_fit_v1.py and tools/hb166_panel_v1.py).
Run inside gpu_job.py via tools/hb166_allfold_build_v1.py sequences.
"""
import argparse
import json
from pathlib import Path
import time

from tools import hb155_watchdog_v1 as base
from tools.hb150_io_v1 import atomic_json
from tools.job_runtime import file_lock

VERSION = "1.2.2"
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local/scratch/slot-b/hb166/allfold-v122"
MODULES = {"tools.hb166_allfold_fit_v1": ("train_step", "validation")}


def classify(command):
    """Return (module, mode, device) for an allowlisted python -B -m command."""
    if len(command) < 5 or command[1:3] != ["-B", "-m"] or command[3] not in MODULES:
        raise ValueError("command must be python -B -m tools.hb166_allfold_fit_v1")
    if command[4] not in ("smoke", "fit") or "--device" not in command:
        raise ValueError("command needs smoke|fit and --device")
    index = command.index("--device") + 1
    if index >= len(command):
        raise ValueError("missing device")
    return command[3], command[4], command[index]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--budget", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--healthy-rates")
    ap.add_argument("--measurement", action="store_true")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    attempt, budget = Path(args.attempt).resolve(), Path(args.budget).resolve()
    if not all(path.is_relative_to(WORK) for path in (attempt, budget)):
        ap.error("attempt and budget must stay inside the H-B-166 stage")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        module, mode, device = classify(command)
    except ValueError as exc:
        ap.error(str(exc))
    rates = json.loads(Path(args.healthy_rates).read_text()) if args.healthy_rates else None
    if args.measurement and (args.cpu or mode != "smoke" or device != "cuda"):
        ap.error("measurement is restricted to H-B-166 CUDA smokes")
    if not args.cpu and not args.measurement:
        if mode != "fit" or not isinstance(rates, dict) or not set(MODULES[module]) <= rates.keys():
            ap.error("GPU fits/panels require measured healthy rates " + "/".join(MODULES[module]))
    base.Policy(0., healthy_rates=rates)  # validate before creating an attempt
    with file_lock(budget.with_suffix(".lock"), blocking=False):
        state = json.loads(budget.read_text()) if budget.exists() else {"used_seconds": 0.}
        now = time.time()
        if state.get("active_since") is not None:
            state["used_seconds"] += max(0., now - state["active_since"])
        if attempt.exists():
            raise FileExistsError("partial attempt preserved; choose an explicitly named new attempt")
        state.update(active_since=now, attempt=str(attempt))
        atomic_json(budget, state)
        start = time.monotonic()
        try:
            return base.supervise(command, attempt, total=None, cpu=args.cpu, healthy_rates=rates)
        finally:
            state.update(used_seconds=state["used_seconds"] + time.monotonic() - start, active_since=None)
            atomic_json(budget, state)


if __name__ == "__main__":
    raise SystemExit(main())
