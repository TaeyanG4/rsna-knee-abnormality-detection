"""v1.0.0: seed0 capped64 fold0 with one change: study/target BCE weights.

Changelog: isolated producer, unchanged seed0 recipe/state/serving; predecessor-
free one-shot B launch. Exact independent review and real admission remain required.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("tools._failure_fold_base", ROOT/"tools/l1_rco_k1_folds_v1.py")
k = importlib.util.module_from_spec(spec)
saved = {n: os.environ.get(n) for n in ("K1_FOLD", "K1_GPU")}
try:
    os.environ["K1_FOLD"] = "1"
    spec.loader.exec_module(k)
    from tools import l1_rco_k1_meta_admit_v1 as meta
finally:
    for n, v in saved.items():
        if v is None: os.environ.pop(n, None)
        else: os.environ[n] = v
from tools import l1_rco_k1_failure_weights_v1 as weights

VERSION = "1.0.0"
SCRIPT = Path(__file__).resolve()
TASK = ROOT/".local/scratch/astra-b2/failure-train-v1"
TEST_ONLY = os.environ.get("K1_FAILURE_TEST") == "1"
WORK = TASK/"qualification" if TEST_ONLY else TASK
WEIGHTS = WORK/"weights.csv"
OLD_ROOT = Path("/root/work/h224-folds-v2")
OWNER = ROOT/".local/scratch/astra-b2/h224-folds-v1/gpu0-owner"
FILES = ["l1_rco_k1_failure_v1.py", "l1_rco_k1_failure_weights_v1.py",
         "l1_k1failure_trainer_v1.py", "l1_rco_k1_failure_smoke_v1.py",
         "l1_rco_k1_failure_gate_v1.py", "l1_rco_k1_meta_admit_v1.py",
         "l1_rco_k1_foldcache_v2.py", "job_runtime.py", "gpu_job.py"]
original_pins, original_admit = k.code_pins, k.admit


def expected_registration():
    old = k.read(k.PARENT)
    k.need(old["contract_id"] == "a58f1476ea4ca3c8d64fe06c80bc6e24d78fdf2f347c4de68c1eba5d9c2ea83d", "sealed seed0 parent")
    k.need(k.digest({n:v for n,v in old.items() if n != "contract_id"}) == old["contract_id"], "seed0 digest")
    for name, value in dict(old["pins"], **old["code_pins"]).items():
        k.need(k.sha(ROOT/name) == value, "source pin changed: "+name)
    k.s.verify_pins()
    k.need(len(old["train"]) == 3190 and len(old["holdout"]) == 977, "canonical roster")
    values = weights.load_weights(WEIGHTS, old["train"], old["holdout"], k.s.LABELS)
    reg = json.loads(json.dumps(old))
    reg.update(version=VERSION, tag=k.TAG, model_fold=0, variant="failure_weights", test_only=TEST_ONLY,
               parent_seed0_contract=old["contract_id"], parent_registration_sha256=k.sha(k.PARENT),
               code_pins=dict(original_pins(), **{"tools/"+n:k.sha(ROOT/"tools"/n) for n in FILES}),
               changes_vs_seed0=["per-study per-target BCE weights; batch-target mean normalized to one"],
               loss_weights=dict(sha256=k.sha(WEIGHTS), rows=len(values), default=1,
                                 normalization="per batch per target mean=1", holdout_rows=0),
               evaluator=dict(rulers=["jev_hybrid","aquila_v1","dread","consensus"],
                              rule="candidate>=seed0 on >=3/4 rulers AND arithmetic mean delta>=0.002",
                              holdout_rows=977, clean_oof=False, exposure="EXPOSED_EXPLORATORY",
                              limitation="weight-generator ancestors trained on fold0; not independent gain validation"),
               projection=dict(fit_hours_calendar_estimate=2.4, fit_hours_training_lower_bound=15950*.55/3600,
                               scope="capped64 4090 planning reference, weighted fit unmeasured; excludes inference/export"))
    # Recipe remains byte-for-byte equal as a JSON object to the seed0 parent.
    reg.pop("contract_id")
    reg["contract_id"] = k.digest(reg)
    return reg


def derive_trainer(source):
    source = k.parent.derive_trainer(source)
    k.need(source.count("tools.l1_rco_k1_v3") == 2, "seed0 identity imports")
    source = source.replace("tools.l1_rco_k1_v3", "tools.l1_rco_k1_failure_v1")
    before = "    loss = F.binary_cross_entropy_with_logits(z, target, reduction=\"none\") * weight"
    k.need(source.count(before) == 1, "seed0 BCE patch identity")
    source = source.replace(before, "    from tools.l1_rco_k1_failure_weights_v1 import normalize\n    weight = normalize(weight)\n"+before)
    before = "        return kind, t, torch.ones_like(t)"
    k.need(source.count(before) == 1, "seed0 arm weight patch identity")
    source = source.replace(before, "        from tools.l1_rco_k1_failure_v1 import training_weights\n        return kind, t, training_weights(pool, t)")
    source = "# K1 failure-weight v1.0.0; original capped64 seed0 trainer.\n" + source
    compile(source, "failure-weight-trainer", "exec")
    return source


def training_weights(pool, like):
    reg = k.read(WORK/"registration.json")
    k.need(k.sha(WEIGHTS) == reg["loss_weights"]["sha256"], "registered weights changed")
    return weights.tensor_weights(WEIGHTS, reg["train"], reg["holdout"], k.s.LABELS, pool, like)


def command(fold=0, cache=None, review=None, gpu=0, action="run"):
    k.need(fold == gpu == 0, "fold0/GPU0 only")
    return [sys.executable, str(SCRIPT), action, "--cache", str(Path(cache or k.CACHE).resolve()),
            "--review", str(Path(review or k.REVIEW).resolve())]


def launch_command(fold=0, cache=None, review=None, gpu=0):
    return [sys.executable, str(ROOT/"tools/gpu_job.py"), "--tag", k.TAG, "--"] + command(fold, cache, review, gpu)


def admit(reg, cache, review, dry_run=True, owner_pid=None):
    try:
        k.need(not TEST_ONLY and not reg["test_only"], "TEST_ONLY weights cannot be admitted")
        if os.name != "nt":
            proof = meta.parity.validate_receipt(cache)
            k.need(proof["mode"] == "META_EPS", "explicit META_EPS required")
            k.need(k.read(review).get("meta_eps_bindings") == meta.evidence_binding(cache), "independent META_EPS binding required")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if not dry_run: raise
        result = dict(status="BLOCKED", blockers=[str(exc)], dry_run=True)
        k.write(WORK/"admit-dry-run.json", result)
        return result
    return original_admit(reg, cache, review, dry_run, owner_pid)


def check_trainer_args(a): return k.check_trainer_args(a)
def check_training_ids(ids): return k.check_training_ids(ids)


def install():
    k.FOLD = k.GPU = 0
    k.WORK, k.OUT = WORK, WORK/"fit"
    k.TAG, k.REVIEW = "K1_FAILURE_WEIGHT_F0_V1", WORK/"REVIEW.json"
    k.BEST, k.TRAIN_STUDIES = "ckpt/best_E_prior_s0_f0.pt", 3190
    if os.name != "nt": k.CACHE = OLD_ROOT/".local/scratch/astra-b2/h224-folds-v1/cache"
    k.expected_registration, k.derive_trainer = expected_registration, derive_trainer
    k.command, k.launch_command, k.admit = command, launch_command, admit
    k.owner_scope = lambda: OWNER
    k.import_trainer = lambda: __import__("tools.l1_k1failure_trainer_v1", fromlist=["Net"])
    k.require_infer_job = lambda cache, review: meta.require_infer_job(k, cache, review)
    return k


install()


def idle_files(root=OLD_ROOT):
    """Known terminal receipts only; not a live process/GPU utilization claim."""
    owner = Path(root)/".local/scratch/astra-b2/h224-folds-v1/gpu0-owner/artifacts/logs"
    evidence = {}
    for tag in ("H224_K1_V3G_ALL", "H224_K1_V3G_F3", "DA052_K1_FULLBAG_QUALIFY", "DA052_K1_V3G_FULLBAG_F0"):
        path = owner/tag
        job, receipt = k.read(path/"job.json"), k.read(path/"receipt.json")
        k.need(receipt["state"] in ("SUCCESS", "FAILED", "STOPPED"), "nonterminal known predecessor")
        k.need(job["run_id"] == receipt["run_id"] and job["command_sha256"] == receipt["command_sha256"], "terminal identity mismatch")
        evidence[tag] = dict(run_id=receipt["run_id"], state=receipt["state"], receipt_sha256=k.sha(path/"receipt.json"))
    return dict(status="KNOWN_JOBS_TERMINAL", jobs=evidence, scope="file-only; live admission checks active GPU processes at launch")


def launch(cache, review, dry_run=False):
    k.need(os.name != "nt" and not TEST_ONLY, "Vast production launcher only")
    reg = k.validate(k.read(WORK/"registration.json"))
    result = admit(reg, cache, review)
    k.need(result["status"] == "CPU_ADMISSIBLE", "CPU admission blocked")
    idle = idle_files()
    k.need(not (k.OUT/"admission.json").exists(), "consumed admission; no retry")
    ledger = WORK/"launch-once.json"
    k.need(not ledger.exists(), "consumed launch; no retry")
    if dry_run: return dict(status="READY_PENDING_B_LAUNCH", idle=idle, command=launch_command(cache=cache,review=review))
    from tools.job_runtime import file_lock
    with file_lock(ledger.with_suffix(".lock"), blocking=False):
        k.need(not ledger.exists(), "consumed launch; no retry")
        k.write(ledger, dict(state="STARTED", contract_id=reg["contract_id"], idle=idle))
        env = os.environ.copy()
        env.update(KNEE_WORKSPACE_ROOT=str(OWNER), CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
        OWNER.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(launch_command(cache=cache, review=review), cwd=ROOT, env=env, check=False)
        k.write(ledger, dict(state="DISPATCHED" if p.returncode == 0 else "FAILED", returncode=p.returncode,
                             contract_id=reg["contract_id"], idle=idle))
        k.need(p.returncode == 0, "launch failed; preserve ledger, no retry")
    return dict(status="DISPATCHED", tag=k.TAG)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["register","validate","prepare","cpu-smoke","admit","review-binding","idle-files","launch","run","infer","export"])
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--cache", type=Path, default=k.CACHE)
    ap.add_argument("--review", type=Path, default=k.REVIEW)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.action not in ("run","infer","launch"): os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.action == "idle-files": result = idle_files()
    elif a.action == "launch": result = launch(a.cache, a.review, a.dry_run)
    elif a.action == "register":
        WORK.mkdir(parents=True, exist_ok=True)
        if a.weights:
            parent = k.read(k.PARENT)
            weights.load_weights(a.weights, parent["train"], parent["holdout"], k.s.LABELS)
            if WEIGHTS.exists(): k.need(k.sha(WEIGHTS) == k.sha(a.weights), "immutable weights differ")
            else: shutil.copyfile(a.weights, WEIGHTS)
        r = k.register()
        result = dict(status="PASS",contract_id=r["contract_id"],train=3190,holdout=977,test_only=TEST_ONLY)
    elif a.action == "run":
        k.need(Path(os.environ.get("KNEE_WORKSPACE_ROOT", "")).resolve() == OWNER.resolve(), "isolated owner required")
        result = k.run(a.cache, a.review)
    else:
        reg = k.validate(k.read(WORK/"registration.json"))
        if a.action == "validate": result = dict(status="PASS",train=3190,holdout=977,recipe_equal_seed0=reg["recipe"]==k.read(k.PARENT)["recipe"])
        elif a.action == "prepare": result = k.prepare(reg, a.cache)
        elif a.action == "cpu-smoke":
            from tools.l1_rco_k1_failure_smoke_v1 import cpu_smoke
            result = cpu_smoke(reg, a.cache)
        elif a.action == "admit":
            k.need(a.dry_run, "real admission only through gpu_job")
            result = admit(reg, a.cache, a.review)
        elif a.action == "review-binding":
            result = dict(bindings=k.review_binding(reg))
            if os.name != "nt": result["meta_eps_bindings"] = meta.evidence_binding(a.cache)
        elif a.action == "infer":
            k.require_infer_job(a.cache,a.review)
            result = k.inference(reg,a.cache)
        else: result = k.export(reg)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__": main()
