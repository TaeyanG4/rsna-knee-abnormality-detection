"""EXP174 v1.1.0: label-only EXP143 v2 wrapper; imports and preparation are CPU only.

Frozen source files are never edited. Explicit probe/train commands are for B's
gpu_queue -> gpu_job lane. Release operations only build local artifacts/plans.
Changelog 1.1.0 (2026-10-12): elapsed/projection values are telemetry; no self-imposed run deadline.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "training/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from tools import exp143_v2 as base

VERSION, SCHEMA = "1.1.0", "exp174-own0949-jev-v1"
BASE = ROOT / "artifacts/experiments/EXP174_OWN0949_JEV"
PROTOCOL = BASE / "protocol-v1/manifest.json"
PROTOCOL_DOC = BASE / "PROTOCOL.md"
BASE_PROTOCOL = ROOT / "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json"
BASE_PROVENANCE = ROOT / "artifacts/experiments/EXP143_OWN0949/S42v2/provenance.json"
BASE_SHA = "b2aaca64289249595e532b74d361175ed78b1170109eee302ec29068e0e7af11"
INIT_SHA = "e92b52ca0fc0e2a46a1f0d44d9a73b219407771c5a65a87d42e44b776b248ba2"
INITIAL_STATE_SHA = "8064b3a56c602e5e52abd3a10b77f235300dd7008f56fc4d7d47fca75e5d3c43"
LABEL_SHA = "33ea0ccaf855811a9756faa418a27da759e954758f5813aee7a35def6b9dd84d"
INIT = ROOT / "storage/weights/exp143-imagenet-init/coatnet384-imagenet.pt"
CORPUS = base.CORPUS
LABELS = ROOT / "storage/labels/derived_constellation/l2_dread75_jev25_v1.csv"
LABEL_PROVENANCE = LABELS.with_suffix(".provenance.json")
RUN_DIRS = {"local": "artifacts/experiments/EXP174_OWN0949_JEV/S42"}
PROBE_ROOT = ROOT / ".local/exp174-probes"
PROBE_OUT = PROBE_ROOT / "probe-s42"
PROBE_TAG, FULL_TAG = "EXP174_JEV_PROBE_S42", "EXP174_OWN0949_JEV_S42"
PREDECESSOR = "EXP168_OWN0949_OAI_JEV_S42"
TORCH_VERSION = "2.5.1+cu121"
RECIPE = dict(base.RECIPE, labels="l2-dread75-jev25")
CODE = list(dict.fromkeys([*base.CODE, base.PROTOCOL,
    "tools/exp168_labels.py", "tools/exp155_own0949_oai_v11.py",
    "tools/exp174_own0949_jev.py", "tools/exp174_own0949_jev_release.py",
    "tools/exp152_own0949_release.py", "tools/gpu_queue.py", "tools/gpu_job.py",
    ".local/exp174-autochain.sh",
    "artifacts/experiments/EXP174_OWN0949_JEV/PROTOCOL.md"]))
TARGETS = base.TARGETS
sha256, json_hash, read_json = base.sha256, base.json_hash, base.read_json
save_torch_new, save_torch_atomic = base.save_torch_new, base.save_torch_atomic
projection = base.projection
series_records = base.series_records


def scoped_path(path, root=None):
    path, root = Path(path).resolve(), Path(root or BASE).resolve()
    if root not in path.parents:
        raise ValueError("output must be a child of the EXP174 subtree")
    return path


def write_new_json(path, value):
    path = Path(path).resolve()
    if not any(r.resolve() in path.parents for r in (BASE, PROBE_ROOT)):
        raise ValueError("EXP174 output outside registered subtrees")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def registered_run(p, lane="local"):
    if lane != "local" or p["run_dirs"] != RUN_DIRS:
        raise ValueError("EXP174 permits only its fresh local seed42 run")
    return scoped_path(ROOT / RUN_DIRS[lane])


def execution_settings(p, world, precision, seed, checkpointing=None):
    if (world != 1 or precision != "bf16" or seed != 42
            or (checkpointing is not None and checkpointing is not False)):
        raise ValueError("EXP174 requires local BF16 seed42, checkpointing OFF")
    return base.execution_settings(p, world, precision, seed, False)


def source_bindings():
    return {name: sha256(ROOT / name) for name in CODE}


def baseline_registration(corpus=None, init_weights=None):
    if sha256(BASE_PROTOCOL) != BASE_SHA:
        raise ValueError("EXP143 v2 control manifest SHA changed")
    p = base.load_registration(BASE_PROTOCOL, corpus, init_weights)
    old = read_json(BASE_PROVENANCE)
    if (p["recipe"] != dict(base.RECIPE, labels="registered-replacement-soft")
            or p["label_file_name"] != "l2_dread75_aquila25_v1.csv"
            or p["budgets"] != dict(local=14.0, kaggle=9.0, reserve_seconds=600)
            or p["init_sha256"] != INIT_SHA or p["recipe"]["oai"] is not False
            or old["protocol_sha256"] != BASE_SHA
            or old["initial_state_sha256"] != INITIAL_STATE_SHA
            or old["torch"] != TORCH_VERSION
            or old["execution"] != base.execution_settings(p, 1, "bf16", 42, False)):
        raise ValueError("EXP143 v2 S42v2 control contract changed")
    return p


def label_table(baseline):
    """Verify existing EXP168 provenance and reconstruct exact labels; no row output."""
    from tools import exp168_labels as lb
    record = read_json(LABEL_PROVENANCE)
    ids = baseline["train_ids"]
    expected_sources = {str(lb.DREAD.relative_to(ROOT)): lb.DREAD_SHA,
                        str(lb.JEV.relative_to(ROOT)): lb.JEV_SHA}
    actual_sources = {str(path.relative_to(ROOT)): sha256(path) for path in (lb.DREAD, lb.JEV)}
    if (sha256(LABELS) != LABEL_SHA or record.get("csv_sha256") != LABEL_SHA
            or record.get("name") != "l2_dread75_jev25_v1"
            or record.get("rows") != 4347 or record.get("targets") != TARGETS
            or record.get("oai") is not False or record.get("gold58_included") is not False
            or record.get("baseline_protocol_sha256") != BASE_SHA
            or record.get("baseline_train_ids_sha256") != hashlib.sha256("\n".join(ids).encode()).hexdigest()
            or record.get("source_sha256") != expected_sources or actual_sources != expected_sources):
        raise ValueError("reused EXP168 label provenance/hash mismatch")
    labels = base.read_label_table(LABELS)
    if set(labels) != set(ids) or set(labels) & set(baseline["excluded_gold_ids"]):
        raise ValueError("JEV mix cohort differs from control")
    dread, jev = lb.read_source(lb.DREAD, ids), lb.read_source(lb.JEV, ids, binary=True)
    for uid in ids:
        expected = [float(format(float(v), ".9g")) for v in lb.mix(dread[uid], jev[uid])]
        if labels[uid] != expected:
            raise ValueError("JEV mix differs from registered float32/.9g formula")
    return {uid: labels[uid] for uid in ids}, record


# Only labels and experiment bookkeeping may differ from the frozen control.
LABEL_FIELDS = {"train_labels", "pos_weight", "label_sha256", "label_provenance",
                "label_provenance_sha256", "label_file_name"}
IDENTITY_FIELDS = {"schema", "created", "run_dirs", "source_bindings", "content_sha256"}
EXTRA_FIELDS = {"version", "base_protocol_sha256", "base_provenance_sha256",
                "expected_initial_state_sha256", "probe_contract"}


def assert_label_only_diff(p, baseline):
    if set(p) - set(baseline) != EXTRA_FIELDS or set(baseline) - set(p):
        raise ValueError("unexpected registration fields")
    expected_recipe = dict(baseline["recipe"], labels=RECIPE["labels"])
    if p["recipe"] != expected_recipe or p["recipe"] != RECIPE:
        raise ValueError("recipe changed beyond label source")
    for key in set(baseline) - LABEL_FIELDS - IDENTITY_FIELDS - {"recipe"}:
        if p[key] != baseline[key]:
            raise ValueError("label-only control contract changed: " + key)
    if p["run_dirs"] != RUN_DIRS or p["expected_initial_state_sha256"] != INITIAL_STATE_SHA:
        raise ValueError("run/initial state changed")
    if p["probe_contract"] != dict(measured_updates=100, warmup_updates=2,
                                    grad_checkpointing=False, max_seconds=1800):
        raise ValueError("probe contract changed")


def prepare(args):
    if Path(args.protocol).resolve() != PROTOCOL.resolve():
        raise ValueError("only the registered EXP174 manifest path is permitted")
    if PROTOCOL.exists() or registered_run({"run_dirs": RUN_DIRS}).exists():
        raise FileExistsError("EXP174 protocol/run already exists")
    baseline = baseline_registration(args.corpus, args.init_weights)
    labels, record = label_table(baseline)
    p = copy.deepcopy(baseline)
    p.update(schema=SCHEMA, version=VERSION, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        recipe=copy.deepcopy(RECIPE), run_dirs=RUN_DIRS.copy(), train_labels=labels,
        pos_weight=base.pos_weights(labels, baseline["train_ids"]), label_sha256=LABEL_SHA,
        label_provenance=record, label_provenance_sha256=sha256(LABEL_PROVENANCE),
        label_file_name=LABELS.name, source_bindings=source_bindings(),
        base_protocol_sha256=BASE_SHA, base_provenance_sha256=sha256(BASE_PROVENANCE),
        expected_initial_state_sha256=INITIAL_STATE_SHA,
        probe_contract=dict(measured_updates=100, warmup_updates=2,
                            grad_checkpointing=False, max_seconds=1800))
    assert_label_only_diff(p, baseline)
    p.pop("content_sha256")
    p["content_sha256"] = json_hash(p)
    write_new_json(PROTOCOL, p)
    print(json.dumps(dict(status=p["status"], studies=len(p["train_ids"]),
        label_sha256=LABEL_SHA, protocol_sha256=sha256(PROTOCOL), gpu_started=False)))


def load_registration(protocol=PROTOCOL, corpus=None, init_weights=None):
    if Path(protocol).resolve() != PROTOCOL.resolve():
        raise ValueError("EXP174 protocol path changed")
    p = read_json(protocol)
    digest = p.pop("content_sha256")
    if json_hash(p) != digest or p.get("schema") != SCHEMA or p.get("version") != VERSION:
        raise ValueError("EXP174 registration content changed")
    p["content_sha256"] = digest
    baseline = baseline_registration(corpus, init_weights)
    assert_label_only_diff(p, baseline)
    labels, record = label_table(baseline)
    if (p["source_bindings"] != source_bindings() or p["base_protocol_sha256"] != BASE_SHA
            or p["base_provenance_sha256"] != sha256(BASE_PROVENANCE)
            or p["label_sha256"] != LABEL_SHA or p["label_file_name"] != LABELS.name
            or p["label_provenance_sha256"] != sha256(LABEL_PROVENANCE)
            or p["label_provenance"] != record or p["train_labels"] != labels
            or p["pos_weight"] != base.pos_weights(labels, p["train_ids"])):
        raise ValueError("EXP174 source/input binding changed")
    registered_run(p)
    return p


def validation_path(protocol):
    return Path(protocol).with_name("validation.json")


def validate(args):
    import exp143_v2 as fit
    result = fit.tiny_validation()
    if args.fakes_only:
        result["status"] = "FAKES_VALIDATED_NO_TRAINING"
    else:
        import torch
        from tools.exp112_full_swa import state_dict_sha256
        p = load_registration(args.protocol, args.corpus, args.init_weights)
        if str(torch.__version__) != TORCH_VERSION:
            raise ValueError("local torch stack differs from S42v2")
        fit._cpu_seed(42)
        model = fit.build_model()
        fit.load_imagenet(model, args.init_weights)
        initial = state_dict_sha256(model.state_dict())
        if initial != p["expected_initial_state_sha256"]:
            raise ValueError("CPU initial state differs from S42v2")
        del model
        base.check_corpus(args.corpus, full_hash=True)
        result.update(status="VALIDATED_NO_TRAINING", protocol_sha256=sha256(args.protocol),
            full_corpus_hashes=True, real_backbone_keys_loaded_cpu=True,
            initial_state_sha256=initial, label_formula_verified=True, label_only_diff_verified=True)
        write_new_json(validation_path(args.protocol), result)
    print(json.dumps(result, indent=2))


def require_validation(protocol):
    v = read_json(validation_path(protocol))
    if (v.get("status") != "VALIDATED_NO_TRAINING" or v.get("protocol_sha256") != sha256(protocol)
            or v.get("initial_state_sha256") != INITIAL_STATE_SHA
            or not all(v.get(k) for k in ("actual_pin_memory_tuple_returns_list", "full_corpus_hashes",
                "real_backbone_keys_loaded_cpu", "label_formula_verified", "label_only_diff_verified"))):
        raise ValueError("full EXP174 CPU validation receipt missing/stale")


def _queue_command(args, action):
    return [args.python, "-m", "tools.exp174_own0949_jev", action,
        "--protocol", str(Path(args.protocol).resolve()), "--corpus", str(Path(args.corpus).resolve()),
        "--init-weights", str(Path(args.init_weights).resolve()), "--workers", str(args.workers)]


def checked_probe(report, p, protocol):
    results = report.get("results", [])
    settings = execution_settings(p, 1, "bf16", 42, False)
    if (report.get("status") != "SCRATCH_PROBE_ONLY_NO_FIT"
            or report.get("protocol_sha256") != sha256(protocol) or len(results) != 1
            or report.get("steps_per_mode") != 100 or report.get("warmup_updates_per_mode") != 2):
        raise ValueError("missing same-protocol probe report")
    r = results[0]
    if (r.get("protocol_sha256") != sha256(protocol) or r.get("mode") != "off"
            or r.get("status") != "MEASURED_SCRATCH_ONLY"
            or r.get("precision") != "bf16" or r.get("grad_checkpointing") is not False
            or r.get("execution") != settings or r.get("initial_state_sha256") != INITIAL_STATE_SHA
            or r.get("torch") != TORCH_VERSION or "4070 Ti SUPER" not in r.get("device_name", "")
            or r.get("skipped_updates") != 0 or r.get("timed_updates") != 100
            or r.get("steps_requested") != 100 or r.get("warmup_updates") != 2):
        raise ValueError("probe did not complete the fixed local 100-update contract")
    durations = r.get("update_seconds", [])
    setup = r.get("setup_seconds", -1)
    if (len(durations) != 100 or any(not math.isfinite(v) or v <= 0 for v in durations)
            or not math.isfinite(setup) or setup < 0):
        raise ValueError("invalid probe timing")
    rate = sum(durations) / len(durations)
    estimate = projection(settings, rate, setup)
    if (not math.isclose(r.get("seconds_per_update", -1), rate, rel_tol=1e-12)
            or any(r.get(k) != v for k, v in estimate.items())
            or not (r.get("admissible") is True or
                    (r.get("admissible") is False and estimate["fits_budget"] is False))):
        raise ValueError("probe timing/diagnostic projection or admissibility differs")
    return r


def queue_json(args):
    p = load_registration(args.protocol, args.corpus, args.init_weights)
    require_validation(args.protocol)
    if not 0 <= args.workers <= 12:
        raise ValueError("workers must be 0..12")
    is_probe = args.command in ("probe-queue", "probe-queue-json")
    if registered_run(p).exists():
        raise FileExistsError("fresh EXP174 run already exists")
    if is_probe:
        if PROBE_OUT.exists():
            raise FileExistsError("probe scratch already exists; no automatic retry")
        command = _queue_command(args, "probe")
        tag, after = PROBE_TAG, PREDECESSOR
    else:
        checked_probe(read_json(PROBE_OUT / "report.json"), p, args.protocol)
        command = _queue_command(args, "train") + ["--device", "cuda", "--precision", "bf16",
            "--seed", "42", "--no-grad-checkpointing"]
        tag, after = FULL_TAG, PROBE_TAG
    if args.after != after:
        raise ValueError("queue predecessor differs from the registered chain")
    out = scoped_path(args.out or PROTOCOL.with_name("probe-queue-s42.json" if is_probe else "queue-s42.json"))
    write_new_json(out, [dict(tag=tag, after=after, continue_on_failure=False, command=command)])
    print(json.dumps(dict(status="QUEUE_JSON_ONLY_NOT_LAUNCHED", out=str(out))))


def scratch_directory(path=None):
    path = Path(path or PROBE_OUT).resolve()
    if path != PROBE_OUT.resolve():
        raise ValueError("only the registered EXP174 scratch directory is permitted")
    return scoped_path(path, PROBE_ROOT)


def isolated_fit():
    """New module object; frozen fit source and shared import module stay untouched."""
    spec = importlib.util.spec_from_file_location("_exp174_frozen_fit", ROOT / "training/src/exp143_v2.py")
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    original_load = fit.load_imagenet
    def checked_initialization(model, path):
        original_load(model, path)
        from tools.exp112_full_swa import state_dict_sha256
        if state_dict_sha256(model.state_dict()) != INITIAL_STATE_SHA:
            raise ValueError("initial state mismatch: STOP before first forward")
    fit.load_imagenet = checked_initialization
    fit.scratch_directory = scratch_directory
    return fit


@contextlib.contextmanager
def guard_shim():
    """Redirect only the frozen trainer's runtime guard import, in this process."""
    import tools
    old_attr, old_module = tools.exp143_v2, sys.modules["tools.exp143_v2"]
    module = sys.modules[__name__]
    tools.exp143_v2 = module
    sys.modules["tools.exp143_v2"] = module
    try:
        yield
    finally:
        tools.exp143_v2 = old_attr
        sys.modules["tools.exp143_v2"] = old_module


def require_gpu_job(tag):
    # gpu_job records its supervising wrapper PID, not the training child PID.
    path = ROOT / "artifacts/logs" / tag / "job.json"
    # environment() is collected after the child starts; allow that known
    # launcher race without admitting a direct unsupervised training command.
    while True:
        try:
            job = read_json(path)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.25)
    if os.getppid() != job.get("pid") or job.get("tag") != tag:
        raise ValueError("explicit GPU work must run through gpu_queue/gpu_job")


def require_local_device():
    import torch
    if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or str(torch.__version__) != TORCH_VERSION
            or "4070 Ti SUPER" not in torch.cuda.get_device_name(0)):
        raise ValueError("EXP174 must run on the original local 4070 / torch stack")
    # The gpu_job launcher may itself hold a metadata-only CUDA context. B
    # owns lane exclusivity; the supervised queue, not a process-name heuristic,
    # gates ordering. The probe separately records measured free VRAM.


def probe(args):
    p = load_registration(args.protocol, args.corpus, args.init_weights)
    require_validation(args.protocol)
    require_gpu_job(PROBE_TAG)
    out = scratch_directory()
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "tools.exp174_own0949_jev", "_probe-mode",
        "--protocol", str(args.protocol), "--corpus", str(args.corpus),
        "--init-weights", str(args.init_weights), "--workers", str(args.workers)]
    env = dict(os.environ, EXP174_PROBE_PARENT=str(os.getpid()), EXP174_SUPERVISOR=str(os.getppid()))
    with (out / "off.log").open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    report = dict(status="SCRATCH_PROBE_ONLY_NO_FIT", protocol_sha256=sha256(args.protocol),
        results=[read_json(out / "off.json")], steps_per_mode=100, warmup_updates_per_mode=2)
    checked_probe(report, p, args.protocol)
    write_new_json(out / "report.json", report)
    print(json.dumps(dict(status=report["status"], projected_hours=report["results"][0]["projected_hours"])))


def run_gpu(args):
    p = load_registration(args.protocol, args.corpus, args.init_weights)
    require_validation(args.protocol)
    if not 0 <= args.workers <= 12:
        raise ValueError("workers must be 0..12")
    if args.command == "train":
        require_gpu_job(FULL_TAG)
        checked_probe(read_json(PROBE_OUT / "report.json"), p, args.protocol)
        execution_settings(p, 1, args.precision, args.seed, args.grad_checkpointing)
        args.resume = False
    else:
        job = read_json(ROOT / "artifacts/logs" / PROBE_TAG / "job.json")
        if (os.environ.get("EXP174_PROBE_PARENT") != str(os.getppid())
                or os.environ.get("EXP174_SUPERVISOR") != str(job.get("pid"))):
            raise ValueError("probe child must belong to the supervised probe")
        args.steps, args.mode, args.precision, args.out = 100, "off", "bf16", str(PROBE_OUT)
    require_local_device()
    fit = isolated_fit()
    with guard_shim():
        return (fit.train if args.command == "train" else fit.probe_mode)(args)


def predecessor_status():
    """CPU state check. No process termination; no optimistic wall-clock release."""
    logs = ROOT / "artifacts/logs" / PREDECESSOR
    full_log = ROOT / ".local/exp168-full-queue.log"
    chain_log = ROOT / ".local/exp168-autochain.log"
    texts = [f.read_text(encoding="utf-8", errors="replace") if f.is_file() else ""
             for f in (full_log, chain_log)]
    if ((logs / "FAILED").exists() or any(marker in text for text in texts
            for marker in ("plan stopped", "PROBE_NOT_SUCCESS", "QUEUE_JSON_REJECTED"))):
        raise RuntimeError("EXP168 chain failed; manual B review required")
    # The existing EXP168 shell redirects only its two queue logs, not its own
    # stdout. Its last GPU operation is this synchronous full-fit queue. Do not
    # depend on an exp168-autochain.log that that script never creates.
    if not ((logs / "SUCCESS").is_file() and "plan finished" in texts[0]
            and PREDECESSOR + ": SUCCESS" in texts[0]):
        return False
    queue = read_json(ROOT / "artifacts/experiments/EXP168_OWN0949_OAI_JEV/protocol-v1/queue-s42.json")
    if len(queue) != 1 or queue[0].get("tag") != PREDECESSOR:
        raise ValueError("EXP168 chain queue changed; manual B review required")
    summary_path = ROOT / "artifacts/experiments/EXP168_OWN0949_OAI_JEV/S42/summary.json"
    summary = read_json(summary_path)
    if (summary.get("status") != "COMPLETE_TRAINING_NO_EVALUATION" or summary.get("epochs") != 16
            or summary.get("model_sha256") != sha256(summary_path.with_name("model_best.pt"))):
        raise ValueError("EXP168 process success lacks authenticated completed fit")
    return True


def require_completed(protocol=PROTOCOL):
    p = load_registration(protocol)
    summary = read_json(registered_run(p) / "summary.json")
    identity = summary.get("identity", {})
    if (summary.get("status") != "COMPLETE_TRAINING_NO_EVALUATION"
            or summary.get("epochs") != 16 or summary.get("updates") != 4352
            or summary.get("skips") != 0 or len(summary.get("history", [])) != 16
            or [h.get("epoch") for h in summary["history"]] != list(range(1, 17))
            or identity.get("protocol_sha256") != sha256(protocol)
            or identity.get("initial_state_sha256") != INITIAL_STATE_SHA
            or identity.get("execution") != execution_settings(p, 1, "bf16", 42, False)):
        raise ValueError("EXP174 full fixed fit is incomplete or identity changed")
    return p


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    def common(p):
        p.add_argument("--protocol", default=str(PROTOCOL))
        p.add_argument("--corpus", default=str(CORPUS))
        p.add_argument("--init-weights", default=str(INIT))
    p = sub.add_parser("register", aliases=["prepare"]); common(p)
    p = sub.add_parser("validate"); common(p)
    p.add_argument("--fakes-only", action="store_true")
    for name, aliases, after in (("probe-queue", ["probe-queue-json"], PREDECESSOR),
                                 ("full-queue", ["queue-json"], PROBE_TAG)):
        p = sub.add_parser(name, aliases=aliases); common(p)
        p.add_argument("--after", default=after)
        p.add_argument("--out")
        p.add_argument("--python", default=base.DEFAULT_PYTHON)
        p.add_argument("--workers", type=int, default=12)
    for name in ("probe", "_probe-mode", "train"):
        p = sub.add_parser(name); common(p)
        p.add_argument("--workers", type=int, default=12)
        if name == "train":
            p.add_argument("--device", choices=["cuda"], default="cuda")
            p.add_argument("--precision", choices=["bf16"], default="bf16")
            p.add_argument("--seed", type=int, choices=[42], default=42)
            p.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p = sub.add_parser("average"); common(p)
    p.add_argument("--checkpoint")
    p.add_argument("--out", default=str(BASE / "average-v1"))
    sub.add_parser("predecessor-status")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ("register", "prepare"):
        return prepare(args)
    if args.command == "validate":
        return validate(args)
    if args.command in ("probe-queue", "probe-queue-json", "full-queue", "queue-json"):
        return queue_json(args)
    if args.command == "probe":
        return probe(args)
    if args.command in ("train", "_probe-mode"):
        return run_gpu(args)
    if args.command == "average":
        from tools.exp174_own0949_jev_release import average
        require_completed(args.protocol)
        print(json.dumps(average(args), indent=2))
        return
    if args.command == "predecessor-status":
        ready = predecessor_status()
        print("EXP168_COMPLETE" if ready else "EXP168_WAIT")
        return 0 if ready else 3


if __name__ == "__main__":
    sys.exit(main())
