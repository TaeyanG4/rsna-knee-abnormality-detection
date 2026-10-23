"""H224 single-4090 launcher v1.1.0.

Changelog v1.1.0: fixed schedule permits ONLY f2 on Vast; historical two-fold
CLI is disabled. Add launch-single --fold 2 with unchanged ownership/receipts.
Changelog: additive f2 -> bound SUCCESS -> f3 on GPU0; no retry or resume.
The pinned v1 training/registration modules remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools import l1_rco_k1_folds_v1 as k
from tools import job_runtime as runtime

VERSION = "1.1.0"


def command():
    """Canonical fixed-schedule entrypoint; child uses the frozen command()."""
    return [sys.executable, "-B", str(Path(__file__).resolve()), "launch-single", "--fold", "2"]


def plan():
    owner = k.BASE / "gpu0-owner"
    return {
        "version": VERSION,
        "owner": str(owner),
        "ledger": str(k.BASE / "sequential-v1.json"),
        "jobs": [
            {"fold": fold, "gpu": 0, "tag": f"H224_K1_V3G_F{fold}",
             "argv": k.launch_command(fold=fold, gpu=0)}
            for fold in (2, 3)
        ],
        "policy": "f2 bound SUCCESS before f3; wait for both; no retry",
    }


def single_plan(fold=2):
    k.need(fold == 2, "fixed Vast schedule permits only fold 2")
    spec = plan()
    spec.update(ledger=str(k.BASE / "single-f2-v1.json"),
                jobs=[spec["jobs"][0]],
                policy="Vast f2 ONLY, then separate allseq after bound SUCCESS; no retry")
    return spec


def check_fresh(spec):
    """Read-only preflight. Reject even completed/failed historical attempts."""
    ledger = Path(spec["ledger"])
    k.need(not ledger.exists(), "sequence already attempted; no retry")
    logs = Path(spec["owner"]) / "artifacts/logs"
    for item in spec["jobs"]:
        directory = logs / item["tag"]
        k.need(not directory.exists(), "existing fold job directory; no retry")
        k.need(not (k.BASE / f"f{item['fold']}" / "fit/admission.json").exists(),
               "existing fold admission; no retry")
    k.need(not runtime.active_jobs(logs), "GPU0 owner occupied or unconfirmed")


def wait_success(logs, item, bound):
    """No training timeout. Unknown/dead/unbound states fail closed."""
    directory = logs / item["tag"]
    child = item["argv"][item["argv"].index("--") + 1:]
    while True:
        job = runtime.read_json(directory / "job.json")
        for key in ("schema", "run_id", "tag", "command", "cwd",
                    "command_sha256", "pid", "create_time", "started_epoch"):
            k.need(job.get(key) == bound.get(key), "job binding changed: " + key)
        status = runtime.job_status(logs, item["tag"], expected_command=child)
        if status["state"] == "running":
            time.sleep(5)
            continue
        k.need(status["state"] == "SUCCESS" and status.get("receipt_kind") == "bound",
               "fold did not finish with bound SUCCESS: " + status["state"])
        receipt = runtime.read_json(directory / "receipt.json")
        k.need(receipt["run_id"] == bound["run_id"]
               and receipt["command_sha256"] == bound["command_sha256"],
               "terminal receipt binding changed")
        # Revalidate after reading the receipt, including process identities.
        k.need(runtime.read_json(directory / "job.json") == job,
               "job changed during terminal verification")
        return receipt


def run_sequence(spec):
    """Internal orchestration, separately CPU-testable without GPU commands."""
    ledger = Path(spec["ledger"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with runtime.file_lock(ledger.with_suffix(".lock"), blocking=False):
        check_fresh(spec)
        state = {"version": VERSION, "state": "STARTED", "jobs": []}
        runtime.atomic_json(ledger, state)  # Durable before any dispatch.
        try:
            owner = Path(spec["owner"])
            owner.mkdir(parents=True, exist_ok=True)
            logs = owner / "artifacts/logs"
            env = {**os.environ, "KNEE_WORKSPACE_ROOT": str(owner),
                   "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1"}
            for item in spec["jobs"]:
                # Do not archive/retry a job created since the initial preflight.
                k.need(not (logs / item["tag"]).exists(), "fold already attempted")
                dispatched = time.time()
                subprocess.run(item["argv"], cwd=k.ROOT, env=env, check=True)
                bound = runtime.read_json(logs / item["tag"] / "job.json")
                child = item["argv"][item["argv"].index("--") + 1:]
                k.need(bound["schema"] == "gpu-job/v2"
                       and bound["tag"] == item["tag"]
                       and bound["command"] == child
                       and Path(bound["cwd"]).resolve() == owner.resolve()
                       and bound["command_sha256"] == runtime.command_digest(child, owner)
                       and bound["started_epoch"] >= dispatched,
                       "dispatch did not produce a fresh bound job")
                entry = {"fold": item["fold"], "gpu": 0, "run_id": bound["run_id"],
                         "command_sha256": bound["command_sha256"]}
                state["jobs"].append(entry)
                runtime.atomic_json(ledger, state)
                entry["receipt"] = wait_success(logs, item, bound)
                runtime.atomic_json(ledger, state)
            state["state"] = "SUCCESS"
            runtime.atomic_json(ledger, state)
            return state
        except BaseException as exc:
            state.update(state="STOPPED", error_type=type(exc).__name__)
            runtime.atomic_json(ledger, state)
            raise  # Never launch the next fold or retry on failure/interruption.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["launch-sequential", "launch-single"])
    ap.add_argument("--fold", type=int, choices=[2], default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    k.need(args.action == "launch-single",
           "launch-sequential retired: Vast f2 ONLY; use launch-single --fold 2")
    spec = single_plan(args.fold)
    if args.dry_run:
        check_fresh(spec)
        print(json.dumps({"status": "PLAN_ONLY", **spec}, indent=2))
    else:
        k.need(os.name != "nt", "single-4090 sequence is Linux only")
        print(json.dumps(run_sequence(spec), indent=2))


if __name__ == "__main__":
    main()
