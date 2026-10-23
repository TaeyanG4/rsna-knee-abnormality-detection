"""v1.0.0: additive, evidence-bound H224 META_EPS admission and GPU0 sequence.

Changelog: preserve all frozen registrations and trainers; revalidate explicit
META_EPS evidence inside admission. Only f2/all use this opt-in entrypoint.
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
from tools import l1_rco_k1_foldcache_v2 as parity
from tools import l1_rco_k1_folds_v1 as folds
from tools import l1_rco_k1_all_v1 as all_data
from tools import l1_rco_k1_foldseq_v1 as seq
from tools import l1_rco_k1_allseq_v1 as allseq

SCRIPT = Path(__file__).resolve()
ORIGINAL_COMMANDS = {2: folds.command, "all": all_data.command}
ORIGINAL_ADMITS = {2: folds.admit, "all": all_data.admit}


def module(fold):
    folds.need(fold in (2, "all"), "META_EPS entrypoint permits only Vast f2/all")
    return folds if fold == 2 else all_data


def command(fold=2, cache=None, review=None, gpu=0, action="run"):
    module(fold)
    folds.need(gpu == 0, "META_EPS entrypoint requires GPU0")
    argv = ORIGINAL_COMMANDS[fold](fold, cache, review or review_path(fold), gpu, action)
    argv[1] = str(SCRIPT)
    return argv


def review_path(fold):
    return ((folds.BASE / "f2") if fold == 2 else
            (all_data.ALL_BASE / "all")) / "REVIEW-META-EPS.json"


def evidence_binding(cache):
    return dict(
        parity_receipt_sha256=folds.sha(Path(cache) / "H224-PARITY.json"),
        meta_reference_sha256=folds.sha(parity.META),
        local32_sha256=folds.sha(parity.old.HASHES),
        code_pins={str(p.relative_to(ROOT)).replace("\\", "/"): folds.sha(p)
                   for p in (SCRIPT, ROOT / "tools/l1_rco_k1_foldcache_v2.py")})


def admit(k, fold, reg, cache, review, dry_run=True, owner_pid=None):
    # Fail before original admission can inspect GPU state or consume admission.
    try:
        proof = parity.validate_receipt(cache)
        folds.need(proof.get("mode") == "META_EPS", "explicit META_EPS receipt required")
        binding = evidence_binding(cache)
        r = k.read(review)
        folds.need(r.get("meta_eps_bindings") == binding, "independent META_EPS review binding required")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        if not dry_run:
            raise ValueError("META_EPS admission blocked: " + str(exc)) from exc
        result = dict(status="BLOCKED", blockers=[str(exc)], dry_run=True)
        return result
    result = ORIGINAL_ADMITS[fold](reg, cache, review, dry_run, owner_pid)
    result["meta_eps_bindings"] = binding
    if dry_run:
        k.write(k.WORK / "admit-dry-run.json", result)
    return result


def require_infer_job(k, cache, review):
    """Original inference ownership checks, with exact wrapper parent argv."""
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    expected = k.run_command(cache, review)
    with file_lock(LOGS / ".gpu-claim.lock"):
        job = read_json(LOGS / k.TAG / "job.json")
        parent = psutil.Process().parent()
        k.need(parent is not None and job.get("schema") == "gpu-job/v2"
               and job.get("tag") == k.TAG and job.get("phase") == "running",
               "inference requires live B gpu_job")
        k.need(job.get("child_identity") == identity(parent.pid),
               "inference must be an owned run child")
        k.need(job.get("command") == expected
               and job.get("command_sha256") == command_digest(expected, k.owner_scope())
               and Path(job["cwd"]).resolve() == k.owner_scope(),
               "inference parent command identity")
        k.need(parent.cmdline() == expected, "inference parent must be exact META_EPS run")


def install(fold):
    k = module(fold)
    k.command = command
    k.admit = lambda reg, cache, review, dry_run=True, owner_pid=None: admit(
        k, fold, reg, cache, review, dry_run, owner_pid)
    k.require_infer_job = lambda cache, review: require_infer_job(k, cache, review)
    return k


def sequence_plan(fold):
    install(2)
    if fold == "all":
        install("all")
        return allseq.plan()
    return dict(version="1.0.0", owner=str(folds.BASE / "gpu0-owner"),
                ledger=str(folds.BASE / "single-f2-v1.json"),
                jobs=[dict(fold=2, gpu=0, tag="H224_K1_V3G_F2",
                           argv=folds.launch_command(fold=2, gpu=0))],
                policy="Vast f2 ONLY; META_EPS admission; bound SUCCESS; no retry")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["admit", "run", "infer", "review-binding",
                                      "launch-single", "launch-after-f2"])
    ap.add_argument("--fold", choices=["2", "all"], default="2")
    ap.add_argument("--gpu", type=int, choices=[0], default=0)
    ap.add_argument("--cache", type=Path, default=parity.old.CACHE)
    ap.add_argument("--review", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fold = 2 if a.fold == "2" else "all"
    k = install(fold)
    k.configure(fold, 0)
    if a.action == "review-binding":
        parity.validate_receipt(a.cache)
        print(json.dumps(dict(meta_eps_bindings=evidence_binding(a.cache)), indent=2))
    elif a.action in ("launch-single", "launch-after-f2"):
        k.need((a.action == "launch-single" and fold == 2)
               or (a.action == "launch-after-f2" and fold == "all"), "sequence/fold mismatch")
        k.need(a.cache.resolve() == parity.old.CACHE.resolve() and a.review is None,
               "sequence uses canonical cache/review paths")
        spec = sequence_plan(fold)
        preflight = allseq.preflight if fold == "all" else seq.check_fresh
        if a.dry_run:
            try:
                preflight(spec)
                status, blockers = "PLAN_ONLY", []
            except (ValueError, OSError, KeyError) as exc:
                status, blockers = "BLOCKED", [str(exc)]
            print(json.dumps(dict(status=status, blockers=blockers, **spec), indent=2))
        else:
            k.need(os.name != "nt", "META_EPS GPU launch is Linux only")
            preflight(spec)
            print(json.dumps(seq.run_sequence(spec), indent=2))
    else:
        # Same CLI, training implementation, registration and ownership gate.
        # Never inherit the frozen fold CLI's default f1 when --fold is omitted.
        if not any(x == "--fold" or x.startswith("--fold=") for x in sys.argv[1:]):
            sys.argv += ["--fold", a.fold]
        if a.review is None:
            sys.argv += ["--review", str(review_path(fold))]
        k.main()


if __name__ == "__main__":
    main()
