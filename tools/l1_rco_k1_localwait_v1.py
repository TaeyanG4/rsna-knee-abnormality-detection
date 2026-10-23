"""v1.0.0: B-only Windows f1 SUCCESS -> f3, one canonical local GPU owner.

Changelog: bind an explicit predecessor run ID; durable waiter and dispatch
ledgers; no retry, cancellation, timeout, or alteration of predecessor files.
Dry-run prints the plan only and NEVER opens f1 or scans live job directories.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools import l1_rco_k1_folds_v1 as k
from tools import l1_rco_k1_foldseq_v1 as seq

VERSION = "1.0.0"
TASK = ROOT / ".local/scratch/astra-b2/h224-device-plan-v1"


def plan(run_id=None):
    return dict(version=VERSION, owner=str(k.ROOT),
                ledger=str(TASK / "local-f3-dispatch.json"),
                waiter_ledger=str(TASK / "local-f3-waiter.json"),
                predecessor=dict(tag="H224_K1_V3G_F1", run_id=run_id,
                                 argv=k.launch_command(fold=1, gpu=0)),
                jobs=[dict(fold=3, gpu=0, tag="H224_K1_V3G_F3",
                           argv=k.launch_command(fold=3, gpu=0))],
                policy="explicit f1 run bound SUCCESS, then f3; local owner; no retry")


def fresh_target(spec):
    for name in ("ledger", "waiter_ledger"):
        k.need(not Path(spec[name]).exists(), "local sequence already attempted; no retry")
    k.need(not (Path(spec["owner"]) / "artifacts/logs/H224_K1_V3G_F3").exists(),
           "f3 job already attempted; no retry")
    k.need(not (k.BASE / "f3/fit/admission.json").exists(),
           "f3 admission consumed; no retry")


def preflight_review():
    k.configure(3, 0)
    reg = k.validate(k.read(k.WORK / "registration.json"))
    result = k.admit(reg, k.CACHE, k.REVIEW, dry_run=True)
    k.need(result["status"] == "CPU_ADMISSIBLE",
           "f3 review/cache admission blocked: " + "; ".join(result["blockers"]))


def run(spec):
    """B executes; unit tests substitute only dispatch, never train."""
    ledger = Path(spec["waiter_ledger"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with seq.runtime.file_lock(ledger.with_suffix(".lock"), blocking=False):
        fresh_target(spec)
        pred = spec["predecessor"]
        k.need(bool(pred["run_id"]), "explicit --f1-run-id required")
        preflight_review()
        logs = Path(spec["owner"]) / "artifacts/logs"
        bound = seq.runtime.read_json(logs / pred["tag"] / "job.json")
        child = pred["argv"][pred["argv"].index("--") + 1:]
        k.need(bound["schema"] == "gpu-job/v2" and bound["tag"] == pred["tag"]
               and bound["run_id"] == pred["run_id"] and bound["command"] == child
               and Path(bound["cwd"]).resolve() == Path(spec["owner"]).resolve()
               and bound["command_sha256"] == seq.runtime.command_digest(child, spec["owner"]),
               "f1 run/owner/command binding mismatch")
        state = dict(version=VERSION, state="WAITING", predecessor=bound)
        seq.runtime.atomic_json(ledger, state)
        try:
            state["predecessor_receipt"] = seq.wait_success(logs, pred, bound)
            # Recheck review and cache after the potentially long wait.
            preflight_review()
            state["state"] = "DISPATCHING"
            seq.runtime.atomic_json(ledger, state)
            state["dispatch"] = seq.run_sequence(spec)
            state["state"] = "SUCCESS"
            seq.runtime.atomic_json(ledger, state)
            return state
        except BaseException as exc:
            state.update(state="STOPPED", error_type=type(exc).__name__)
            seq.runtime.atomic_json(ledger, state)
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["launch-after-f1"])
    ap.add_argument("--f1-run-id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    spec = plan(args.f1_run_id)
    if args.dry_run:
        print(json.dumps(dict(status="PLAN_ONLY", admission_checked=False,
                              predecessor_checked=False, **spec), indent=2))
        return
    k.need(os.name == "nt", "local f3 waiter is Windows only")
    print(json.dumps(run(spec), indent=2))


if __name__ == "__main__":
    main()
