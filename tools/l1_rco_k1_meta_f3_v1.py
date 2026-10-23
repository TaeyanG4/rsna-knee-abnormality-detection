"""v1.0.0: additive Vast GPU0 f3 after an explicit all-data bound SUCCESS.

Changelog: preserve the live f2/all META_EPS entrypoint and all frozen trainers.
Separate Linux f3 review, same canonical owner, durable single attempt, no retry.
This is a post-success launcher, not a waiter for a running all-data job.
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
from tools import l1_rco_k1_meta_admit_v1 as meta

k, seq = meta.folds, meta.seq
SCRIPT = Path(__file__).resolve()
VERSION = "1.0.0"


def review_path():
    return k.BASE / "f3/REVIEW-META-EPS.json"


def command(fold=3, cache=None, review=None, gpu=0, action="run"):
    k.need(fold == 3 and gpu == 0, "Vast f3 entrypoint requires fold 3 / GPU0")
    argv = meta.ORIGINAL_COMMANDS[2](3, cache, review or review_path(), 0, action)
    argv[1] = str(SCRIPT)
    return argv


def evidence_binding(cache):
    binding = meta.evidence_binding(cache)
    binding["code_pins"].update({
        str(p.relative_to(ROOT)).replace("\\", "/"): k.sha(p)
        for p in (SCRIPT, ROOT / "tools/l1_rco_k1_foldseq_v1.py",
                  ROOT / "tools/job_runtime.py", ROOT / "tools/gpu_job.py")
    })
    return binding


def admit(reg, cache, review, dry_run=True, owner_pid=None):
    try:
        proof = meta.parity.validate_receipt(cache)
        k.need(proof.get("mode") == "META_EPS", "explicit META_EPS receipt required")
        binding = evidence_binding(cache)
        k.need(k.read(review).get("meta_eps_bindings") == binding,
               "independent f3 META_EPS review binding required")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        if not dry_run:
            raise ValueError("f3 META_EPS admission blocked: " + str(exc)) from exc
        return dict(status="BLOCKED", blockers=[str(exc)], dry_run=True)
    # The frozen fold implementation is shared by folds 1, 2 and 3.
    result = meta.ORIGINAL_ADMITS[2](reg, cache, review, dry_run, owner_pid)
    result["meta_eps_bindings"] = binding
    if dry_run:
        k.write(k.WORK / "admit-dry-run.json", result)
    return result


def install():
    k.command = command
    k.admit = admit
    k.require_infer_job = lambda cache, review: meta.require_infer_job(k, cache, review)
    k.configure(3, 0)
    return k


def plan(run_id):
    install()
    return dict(
        version=VERSION, owner=str(k.BASE / "gpu0-owner"),
        ledger=str(k.BASE / "after-all-f3-v1.json"),
        predecessor=dict(tag=meta.all_data.TAG, run_id=run_id,
                         command=meta.command("all")),
        jobs=[dict(fold=3, gpu=0, tag="H224_K1_V3G_F3",
                   argv=k.launch_command(fold=3, gpu=0))],
        policy="explicit all-data bound SUCCESS; Vast GPU0; META_EPS; no retry")


def predecessor_success(spec):
    pred = spec["predecessor"]
    k.need(bool(pred["run_id"]), "explicit --all-run-id required")
    logs = Path(spec["owner"]) / "artifacts/logs"
    directory = logs / pred["tag"]
    job = seq.runtime.read_json(directory / "job.json")
    child = pred["command"]
    k.need(job.get("schema") == "gpu-job/v2" and job.get("tag") == pred["tag"]
           and job.get("run_id") == pred["run_id"] and job.get("command") == child
           and Path(job["cwd"]).resolve() == Path(spec["owner"]).resolve()
           and job.get("command_sha256") == seq.runtime.command_digest(child, spec["owner"]),
           "all-data run/owner/command binding mismatch")
    status = seq.runtime.job_status(logs, pred["tag"], expected_command=child)
    k.need(status["state"] == "SUCCESS" and status.get("receipt_kind") == "bound",
           "all-data bound SUCCESS required; no waiting or dispatch")
    receipt = seq.runtime.read_json(directory / "receipt.json")
    k.need(receipt["run_id"] == pred["run_id"]
           and receipt["command_sha256"] == job["command_sha256"]
           and seq.runtime.read_json(directory / "job.json") == job,
           "all-data terminal binding changed")
    return receipt


def preflight(spec):
    receipt = predecessor_success(spec)
    seq.check_fresh(spec)
    reg = k.validate(k.read(k.WORK / "registration.json"))
    result = admit(reg, k.CACHE, review_path(), dry_run=True)
    k.need(result["status"] == "CPU_ADMISSIBLE",
           "f3 review/cache blocked: " + "; ".join(result["blockers"]))
    k.need(predecessor_success(spec) == receipt, "all-data receipt changed during preflight")
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["register", "validate", "prepare", "cpu-smoke",
                                      "review-binding", "admit", "run", "infer",
                                      "launch-after-all"])
    ap.add_argument("--fold", type=int, choices=[3], default=3)
    ap.add_argument("--gpu", type=int, choices=[0], default=0)
    ap.add_argument("--cache", type=Path, default=k.CACHE)
    ap.add_argument("--review", type=Path)
    ap.add_argument("--all-run-id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    install()
    if a.action not in ("run", "infer", "launch-after-all"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.action == "review-binding":
        reg = k.validate(k.read(k.WORK / "registration.json"))
        proof = meta.parity.validate_receipt(a.cache)
        k.need(proof.get("mode") == "META_EPS", "explicit META_EPS receipt required")
        print(json.dumps(dict(bindings=k.review_binding(reg),
                              meta_eps_bindings=evidence_binding(a.cache)), indent=2))
    elif a.action == "launch-after-all":
        k.need(a.cache.resolve() == k.CACHE.resolve() and a.review is None,
               "sequence uses canonical cache/review paths")
        spec = plan(a.all_run_id)
        if a.dry_run:
            try:
                preflight(spec)
                status, blockers = "PLAN_ONLY", []
            except (ValueError, OSError, KeyError, TypeError) as exc:
                status, blockers = "BLOCKED", [str(exc)]
            print(json.dumps(dict(status=status, blockers=blockers, **spec), indent=2))
        else:
            k.need(os.name != "nt", "Vast f3 GPU launch is Linux only")
            preflight(spec)
            print(json.dumps(seq.run_sequence(spec), indent=2))
    else:
        k.need(a.all_run_id is None, "--all-run-id only applies to launch-after-all")
        if not any(x == "--fold" or x.startswith("--fold=") for x in sys.argv[1:]):
            sys.argv += ["--fold", "3"]
        if a.review is None:
            sys.argv += ["--review", str(review_path())]
        k.main()


if __name__ == "__main__":
    main()
