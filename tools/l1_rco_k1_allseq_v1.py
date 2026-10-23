"""v1.0.0: B-only single-4090 all-data launch after the sealed f2 succeeds.

Changelog: separate durable all-data attempt; reuse reviewed sequencing/ownership.
No waiting for a running f2, no implicit f2/f3 dispatch, no automatic retries.
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
from tools import l1_rco_k1_all_v1 as k
from tools import l1_rco_k1_folds_v1 as folds
from tools import l1_rco_k1_foldseq_v1 as seq


def plan():
    return dict(version="1.0.0", owner=str(k.BASE / "gpu0-owner"),
                ledger=str(k.ALL_BASE / "sequential-all-v1.json"),
                jobs=[dict(fold="all", gpu=0, tag=k.TAG, argv=k.launch_command())],
                policy="f2 bound SUCCESS; same GPU0 owner; no retry; final EMA only")


def preflight(spec):
    logs = Path(spec["owner"]) / "artifacts/logs"
    command = folds.command(fold=2, gpu=0)
    status = seq.runtime.job_status(logs, "H224_K1_V3G_F2", expected_command=command)
    k.need(status["state"] == "SUCCESS" and status.get("receipt_kind") == "bound",
           "f2 bound SUCCESS required before all-data launch")
    job = seq.runtime.read_json(logs / "H224_K1_V3G_F2/job.json")
    k.need(Path(job["cwd"]).resolve() == Path(spec["owner"]).resolve()
           and job["command"] == command
           and job["command_sha256"] == seq.runtime.command_digest(command, spec["owner"]),
           "f2 owner/command mismatch")
    k.need(not (k.OUT / "admission.json").exists(), "all-data admission consumed; no retry")
    seq.check_fresh(spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["launch-after-f2"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    spec = plan()
    if args.dry_run:
        try:
            preflight(spec)
            status, blockers = "PLAN_ONLY", []
        except (ValueError, OSError, KeyError) as exc:
            status, blockers = "BLOCKED", [str(exc)]
        print(json.dumps(dict(status=status, blockers=blockers, **spec), indent=2))
        return
    k.need(os.name != "nt", "all-data 4090 launch is Linux only")
    preflight(spec)
    print(json.dumps(seq.run_sequence(spec), indent=2))


if __name__ == "__main__":
    main()
