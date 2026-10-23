"""D-A-052 v1.0.0: isolated fullbag fold0 tool, no changes to live fold tools.

Changelog: full stride candidate bag, cap640 fail closed, encoder replay, same
seed0 recipe/roster and serving architecture. Production requires independent
META_EPS and numerical review. No retry; Vast GPU0 after bound f3 SUCCESS.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
# Load a private instance: never mutate the live fold/meta modules' globals.
spec = importlib.util.spec_from_file_location("tools._k1_fullbag_fold_base",
                                             ROOT/"tools/l1_rco_k1_folds_v1.py")
k = importlib.util.module_from_spec(spec)
_environment = {n: os.environ.get(n) for n in ("K1_FOLD", "K1_GPU")}
try:
    # Frozen folds reject fold0 at import. Import dependencies under their
    # supported default, then restore the caller's environment unchanged.
    os.environ["K1_FOLD"] = "1"
    spec.loader.exec_module(k)
    from tools import l1_rco_k1_meta_admit_v1 as meta
finally:
    for _name, _value in _environment.items():
        if _value is None:
            os.environ.pop(_name, None)
        else:
            os.environ[_name] = _value
VERSION = "1.0.0"
SCRIPT = Path(__file__).resolve()
WORK = ROOT/".local/scratch/astra-b2/k1-fullbag-v1"
CANONICAL_OWNER = k.BASE/"gpu0-owner"
CAP = 640
FILES = ["l1_rco_k1_fullbag_v1.py", "l1_rco_k1_fullbag_replay_v1.py",
         "l1_rco_k1_fullbag_check_v1.py", "l1_k1fullbag_trainer_v1.py",
         "l1_rco_k1_meta_f3_v1.py", "l1_rco_k1_meta_admit_v1.py",
         "l1_rco_k1_foldcache_v2.py", "l1_rco_k1_foldseq_v1.py",
         "job_runtime.py", "gpu_job.py"]
original_pins, original_admit = k.code_pins, k.admit
original_review_binding, original_trainer_argv = k.review_binding, k.trainer_argv


def code_pins():
    return dict(original_pins(), **{"tools/"+n: k.sha(ROOT/"tools"/n) for n in FILES})


def expected_registration():
    old = k.read(k.PARENT)
    k.need(old["contract_id"] == "a58f1476ea4ca3c8d64fe06c80bc6e24d78fdf2f347c4de68c1eba5d9c2ea83d",
           "sealed seed0 parent")
    k.need(k.digest({n:v for n,v in old.items() if n != "contract_id"}) == old["contract_id"],
           "seed0 digest")
    for name, value in dict(old["pins"], **old["code_pins"]).items():
        k.need(k.sha(ROOT/name) == value, "source pin changed: "+name)
    k.s.verify_pins()
    reg = json.loads(json.dumps(old))
    k.need(len(reg["train"]) == 3190 and len(reg["holdout"]) == 977, "canonical fold0 roster")
    reg.update(version=VERSION, tag=k.TAG, model_fold=0, variant="fullbag",
               parent_seed0_contract=old["contract_id"], parent_registration_sha256=k.sha(k.PARENT),
               code_pins=code_pins(), changes_vs_seed0=["full stride candidate bag and encoder-gradient replay"])
    reg["recipe"].update(windows=CAP, model_fold=0, only_folds="0",
                         unchanged="all seed0 settings except train bag/replay",
                         train_cap_policy="fail if >640; never subsample", encoder_microbatch=32,
                         replay_boundary="encoder + spatial_pool; one full-head BCE",
                         optimizer_steps_per_batch=1)
    reg["evaluator"] = dict(rulers=["jev_hybrid","aquila_v1","dread","consensus"],
                           rule="macro_auc >= seed0 on >=3/4 rulers; exact 977 own holdout; 12 defined targets each",
                           exposure="EXPOSED_EXPLORATORY; different GPU hosts and window exposure confound attribution")
    reg["projection"] = dict(fit_hours_training_lower_bound=15950*.55/3600,
                             fit_hours_calendar_estimate=None,
                             scope="Unmeasured fullbag: 0.55s * padded_full_windows/padded_capped64_windows * 1.25..1.5; excludes validation, I/O and checkpoints")
    reg.pop("contract_id")
    reg["contract_id"] = k.digest(reg)
    return reg


def derive_trainer(source):
    source = k.parent.derive_trainer(source)
    edits = [
        ("tools.l1_rco_k1_state_v2", "tools.l1_rco_k1_foldstate_v1"),
        ("guard_constant=a.l1_guard_constant)", "guard_constant=a.l1_guard_constant, fold=0)"),
        ("            cand = [cand[j] for j in np.sort(rng.choice(len(cand), self.n_win, replace=False))]",
         '            raise ValueError("fullbag candidate cap exceeded; no subsampling")'),
    ]
    for before, after in edits:
        k.need(source.count(before) == 1, "fullbag patch identity")
        source = source.replace(before, after)
    k.need(source.count("tools.l1_rco_k1_v3") == 2, "identity import count")
    source = source.replace("tools.l1_rco_k1_v3", "tools.l1_rco_k1_fullbag_v1")
    start = source.index('                    with torch.autocast("cuda"', source.index("for img, meta, mask, idx in l1_state.batches():"))
    end = source.index("                    if a.forward_only:", start)
    source = source[:start] + '''                    assert kind == "bce" and state_spec is None and not a.extent and not a.pair_rank
                    from tools.l1_rco_k1_fullbag_replay_v1 import training_backward
                    opt.zero_grad(set_to_none=True)
                    loss = training_backward(model, img, meta, mask, gen, a,
                                             lambda z: bce_soft(z, tgt[sel], wgt[sel]))
''' + source[end:]
    compile(source, "fullbag-trainer", "exec")
    return source


def trainer_argv(cache, reg):
    cmd = original_trainer_argv(cache, reg)
    cmd[cmd.index("--train-windows")+1] = str(CAP)
    return cmd


def command(fold=0, cache=None, review=None, gpu=0, action="run"):
    k.need(fold == gpu == 0, "fullbag is seed0/fold0 on GPU0")
    return [sys.executable, str(SCRIPT), action, "--mode", "fullbag",
            "--cache", str(Path(cache or k.CACHE).resolve()),
            "--review", str(Path(review or k.REVIEW).resolve())]


def launch_command(fold=0, cache=None, review=None, gpu=0):
    return [sys.executable, str(ROOT/"tools/gpu_job.py"), "--tag", k.TAG, "--"] + command(fold, cache, review, gpu)


def numerical_binding(device):
    path = WORK/f"numerical-{device}.json"
    r = k.read(path)
    k.need(r["status"] == "PASS" and r["device"] == device and r["actual_model"]
           and r["code_pins"] == code_pins(), "fresh numerical replay evidence required")
    if device == "cuda":
        k.need(r["amp"] and r["peak_reserved"] <= 12*2**30, "GPU numerical memory/AMP gate")
    return k.sha(path)


def review_binding(reg):
    return dict(original_review_binding(reg), numerical_cpu_sha256=numerical_binding("cpu"),
                numerical_cuda_sha256=numerical_binding("cuda"))


def meta_binding(cache):
    from tools import l1_rco_k1_meta_admit_v1 as meta
    proof = meta.parity.validate_receipt(cache)
    k.need(proof["mode"] == "META_EPS", "explicit META_EPS required")
    return dict(meta.evidence_binding(cache), fullbag_code_pins=code_pins())


def fullshape_command(run_id):
    return [sys.executable, str(ROOT/"tools/l1_rco_k1_fullbag_check_v1.py"),
            "--device", "cuda", "--full-shape", "--f3-run-id", run_id]


def fullshape_evidence(reg, review):
    memory = k.read(WORK/"fullshape-cuda.json")
    k.need(review["fullbag_memory_sha256"] == k.sha(WORK/"fullshape-cuda.json"),
           "full-shape receipt review binding")
    k.need(memory["contract_id"] == reg["contract_id"] and memory["gpu"] == "4090"
           and memory["status"] == "PASS" and memory["code_pins"] == code_pins()
           and memory["batch"] == 2 and memory["windows_per_study"] == CAP
           and memory["crop"] == 352 and memory["amp"] is True
           and memory["peak_reserved"] <= 12*2**30
           and memory["baseline_used"] + memory["peak_reserved"] + 2*2**30 <= memory["total_bytes"],
           "reviewed measured full-shape 4090 memory evidence required")
    k.need(memory["qualification_review_sha256"] == k.sha(WORK/"REVIEW-QUALIFY.json"),
           "qualification review changed")
    once = k.read(WORK/"fullshape-once.json")
    k.need(once["run_id"] == memory["run_id"]
           and once["predecessor"] == predecessor_success(plan(memory["f3_run_id"])),
           "qualification predecessor changed")
    predecessor_success(dict(owner=str(CANONICAL_OWNER), predecessor=dict(
        tag="DA052_K1_FULLBAG_QUALIFY", run_id=memory["run_id"],
        command=fullshape_command(memory["f3_run_id"]))))
    return memory


def admit(reg, cache, review, dry_run=True, owner_pid=None):
    try:
        binding = meta_binding(cache)
        k.need(k.read(review).get("meta_eps_bindings") == binding,
               "independent fullbag META_EPS review required")
        # New bag memory is not certified by the old capped64 K0.
        memory = fullshape_evidence(reg, k.read(review))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        if not dry_run:
            raise ValueError("fullbag admission blocked: "+str(exc)) from exc
        return dict(status="BLOCKED", blockers=[str(exc)], dry_run=True)
    result = original_admit(reg, cache, review, dry_run, owner_pid)
    if not dry_run:
        k.need(result["desktop_bytes"] + memory["peak_reserved"] + 2*2**30 <= result["total_bytes"],
               "current desktop + fullbag measured peak +2GiB exceeds memory gate")
    result["meta_eps_bindings"] = binding
    return result


def check_trainer_args(a):
    return k.check_trainer_args(a)


def check_training_ids(ids):
    return k.check_training_ids(ids)


def install():
    k.FOLD = k.GPU = 0
    k.WORK, k.OUT = WORK, WORK/"fit"
    k.TAG, k.REVIEW = "DA052_K1_V3G_FULLBAG_F0", WORK/"REVIEW-META-EPS.json"
    k.BEST, k.TRAIN_STUDIES = "ckpt/best_E_prior_s0_f0.pt", 3190
    k.expected_registration, k.derive_trainer = expected_registration, derive_trainer
    k.trainer_argv, k.command, k.launch_command = trainer_argv, command, launch_command
    k.admit, k.review_binding = admit, review_binding
    k.owner_scope = lambda: CANONICAL_OWNER
    k.import_trainer = lambda: __import__("tools.l1_k1fullbag_trainer_v1", fromlist=["Net"])
    from tools import l1_rco_k1_meta_admit_v1 as meta
    k.require_infer_job = lambda cache, review: meta.require_infer_job(k, cache, review)
    return k


install()


def plan(run_id):
    from tools import l1_rco_k1_meta_f3_v1 as f3
    return dict(version=VERSION, owner=str(CANONICAL_OWNER),
                ledger=str(WORK/"after-f3-once.json"),
                predecessor=dict(tag="H224_K1_V3G_F3", run_id=run_id, command=f3.command()),
                jobs=[dict(fold="fullbag", gpu=0, tag=k.TAG, argv=launch_command())],
                policy="bound f3 SUCCESS; same GPU0 owner; independent review; no retry")


def predecessor_success(spec):
    # Pure receipt verifier from the reviewed f3 launcher; messages retain its
    # predecessor wording, but all identities come from this explicit f3 spec.
    from tools import l1_rco_k1_meta_f3_v1 as f3
    return f3.predecessor_success(spec)


def preflight(spec):
    from tools import l1_rco_k1_foldseq_v1 as seq
    receipt = predecessor_success(spec)
    seq.check_fresh(spec)
    k.need(not (k.OUT/"admission.json").exists(), "fullbag consumed admission; no retry")
    reg = k.validate(k.read(WORK/"registration.json"))
    result = admit(reg, k.CACHE, k.REVIEW)
    k.need(result["status"] == "CPU_ADMISSIBLE", "fullbag admission blocked")
    k.need(predecessor_success(spec) == receipt, "f3 terminal receipt changed")
    return receipt


def run_sequence(spec):
    """Separate one-shot ledger: recheck own admission and f3 under the lock."""
    from tools import l1_rco_k1_foldseq_v1 as seq
    runtime = seq.runtime
    ledger = Path(spec["ledger"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with runtime.file_lock(ledger.with_suffix(".lock"), blocking=False):
        receipt = preflight(spec)
        state = dict(version=VERSION, state="STARTED", spec=spec,
                     predecessor_receipt=receipt, jobs=[])
        runtime.atomic_json(ledger, state)
        try:
            owner = Path(spec["owner"])
            env = {**os.environ, "KNEE_WORKSPACE_ROOT": str(owner),
                   "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1"}
            item = spec["jobs"][0]
            logs = owner/"artifacts/logs"
            k.need(not (logs/item["tag"]).exists(), "fullbag already attempted")
            started = time.time()
            subprocess.run(item["argv"], cwd=ROOT, env=env, check=True)
            bound = runtime.read_json(logs/item["tag"]/"job.json")
            child = item["argv"][item["argv"].index("--")+1:]
            k.need(bound["schema"] == "gpu-job/v2" and bound["tag"] == k.TAG
                   and bound["command"] == child and Path(bound["cwd"]).resolve() == owner.resolve()
                   and bound["command_sha256"] == runtime.command_digest(child, owner)
                   and bound["started_epoch"] >= started, "fresh fullbag job binding required")
            state["jobs"].append(dict(run_id=bound["run_id"],
                                      receipt=seq.wait_success(logs, item, bound)))
            state["state"] = "SUCCESS"
            runtime.atomic_json(ledger, state)
            return state
        except BaseException as exc:
            state.update(state="STOPPED", error_type=type(exc).__name__)
            runtime.atomic_json(ledger, state)
            raise


def require_sequence():
    ledger = k.read(WORK/"after-f3-once.json")
    k.need(ledger["state"] == "STARTED", "launch-after-f3 durable admission required")
    spec = ledger["spec"]
    k.need(spec == plan(spec["predecessor"]["run_id"]), "canonical sequence binding required")
    k.need(predecessor_success(spec) == ledger["predecessor_receipt"],
           "f3 terminal receipt changed after launch")


def compare(reg):
    from tools import l1_rco_k1_foldgate_v1 as gate
    from tools import exp156_k_sweep as rulers
    baseline = k.parent.OUT
    paths = dict(k1_csv=baseline/"holdout.csv", k1_result=baseline/"result.json")
    gate.validate_k1(paths, reg["holdout"], 0)
    # export() checks production receipts, model/CSV hashes and exact roster.
    k.export(reg)
    refs = rulers.references(reg["holdout"], root=ROOT)
    scores = {name: gate.macro_aucs(gate.predictions(path, reg["holdout"], 0), refs)
              for name, path in (("seed0", paths["k1_csv"]), ("fullbag", k.OUT/"holdout.csv"))}
    result = decide(scores["seed0"], scores["fullbag"])
    result.update(metrics=scores, cohort_sha256=k.digest(reg["holdout"]), holdout_rows=977,
                  baseline_result_sha256=k.sha(paths["k1_result"]),
                  fullbag_result_sha256=k.sha(k.OUT/"result.json"), clean_oof=False,
                  r7_complement_comparison="NOT_RUN: fullbag vs seed0 own-holdout objective test",
                  interpretation="EXPOSED_EXPLORATORY; host/window exposure confounded; independent review pending")
    k.write(WORK/"comparison.json", result)
    return result


def decide(baseline, candidate):
    import math
    from tools.l1_rco_k1_foldgate_v1 import RULERS
    k.need(set(baseline) == set(candidate) == set(RULERS)
           and all(math.isfinite(v) and 0 <= v <= 1 for d in (baseline, candidate) for v in d.values()),
           "four finite macro AUCs required")
    wins = [r for r in RULERS if candidate[r] >= baseline[r]]
    return dict(status="SCREEN_PASS_PENDING_REVIEW" if len(wins) >= 3 else "HOLD",
                rule="macro_auc >= seed0 on >=3/4 rulers, 12 defined targets on exact own holdout",
                passing_rulers=wins)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["register","prepare","validate","cpu-smoke","admit","run",
                                      "export","infer","evaluate","review-binding","launch-after-f3"])
    ap.add_argument("--mode", choices=["fullbag"], default="fullbag")
    ap.add_argument("--cache", type=Path, default=k.CACHE)
    ap.add_argument("--review", type=Path, default=k.REVIEW)
    ap.add_argument("--f3-run-id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.action not in ("run", "infer"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.action == "launch-after-f3":
        k.need(a.cache.resolve() == k.CACHE.resolve() and a.review == k.REVIEW, "canonical sequence inputs")
        spec = plan(a.f3_run_id)
        if a.dry_run:
            try:
                preflight(spec)
                result = dict(status="PLAN_ONLY", **spec)
            except (OSError, ValueError, KeyError, TypeError):
                result = dict(status="BLOCKED", reason="f3 bound SUCCESS / fresh admission evidence required", **spec)
        else:
            k.need(os.name != "nt", "Vast Linux only")
            preflight(spec)
            result = run_sequence(spec)
    elif a.action == "register":
        r = k.register()
        result = dict(status="PASS", contract_id=r["contract_id"], train=3190, holdout=977)
    else:
        reg = k.validate(k.read(WORK/"registration.json"))
        if a.action == "validate":
            result = dict(status="PASS", train=3190, holdout=977)
        elif a.action == "prepare":
            prep = k.prepare(reg, a.cache)
            result = dict(status="PASS", train=prep["train_count"], trainer_sha256=prep["trainer_sha256"])
        elif a.action == "cpu-smoke":
            from tools.l1_rco_k1_fullbag_check_v1 import cpu_smoke
            result = cpu_smoke(reg, a.cache)
        elif a.action == "admit":
            k.need(a.dry_run, "admission only through gpu_job")
            result = admit(reg, a.cache, a.review)
        elif a.action == "review-binding":
            try:
                result = dict(status="PASS", bindings=review_binding(reg),
                              meta_eps_bindings=meta_binding(a.cache),
                              fullbag_memory_sha256=k.sha(WORK/"fullshape-cuda.json"))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = dict(status="BLOCKED", blockers=[str(exc)])
        elif a.action == "run":
            k.need(os.name != "nt" and Path(os.environ.get("KNEE_WORKSPACE_ROOT","")).resolve() == CANONICAL_OWNER.resolve(),
                   "Vast canonical GPU0 owner required")
            # Direct run cannot bypass the predecessor SUCCESS requirement.
            require_sequence()
            result = k.run(a.cache, a.review)
        elif a.action == "infer":
            k.require_infer_job(a.cache, a.review)
            result = k.inference(reg, a.cache)
        elif a.action == "export":
            result = k.export(reg)
        else:
            result = compare(reg)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
