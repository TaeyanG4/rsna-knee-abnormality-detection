"""Portable, fail-closed HC06 CLS/PATCH pair. Import has no GPU/network/IO effects.

Local: prepare --out <fresh-package-directory> (small allowlisted files only).
Linux: verify --protocol manifest.json; preflight --protocol manifest.json;
       python tools/gpu_queue.py queue.json --foreground
The coordinator must review/admit the exact manifest, copy its declared corpus
shards, and establish paid-resource ownership before run. No provisioning here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vast-hc06-pair-v1"
PARENT = "artifacts/experiments/HC06_PATCH_MIL_PROTOCOL_V1/manifest.json"
PARENT_SHA = "aa703d59a83779465398c2d149bef358913d6f6bc68fb4b26e8922898519dd41"
TEMPLATE = "training/runners/hc06_patch_mil.py"
RUNNER = "training/runners/vast_hc06.py"
WEIGHT = "storage/weights/vast_hc06/dinov2s.safetensors"
WEIGHT_SHA = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
INITIAL_SHA = "688756a1dff13683a5d55a60dd667e0bbd88b17df324d2598c5baf3122c41ad5"
TRACE_SHA = "ab6b802eb04ed6636f8067eeeaa0c6b4e2c0338c1f81c429d204b31c688d29c7"
POSITION_POLICY = "canonical518-trainable-cpu-fp32-bicubic-aa-v2"
POLICIES = {"CLS": "dinov2-normalized-cls-existing-mil-v1",
            "PATCH": "dinov2-normalized-patch256-existing-mil-v1"}
AGGREGATION_POLICY = None  # Set once from the registered process tag.
CODE_NAMES = ("raptor_train.py", "raptor_common.py", "raptor_data.py", "raptor_runtime.py")
SOURCE_FILES = ["tools/vast_hc06.py", "tools/exp084_group_loss.py",
                "tools/exp086_dinov2s_v2.py", "tools/process_resource_receipt.py",
                "tools/gpu_queue.py", "tools/gpu_job.py",
                "training/adapters/dinov2_cpu_pos_v2.py",
                "training/adapters/dinov2_patch_mil.py", TEMPLATE,
                *["training/src/" + n for n in CODE_NAMES]]
DATA_FILES = ["storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv",
              "artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/excluded.csv",
              "artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json",
              "artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/group-metadata.csv"]
CORPUS = "storage/cache/derived/RAPTOR64_336_C140_S02"
SMALL_CORPUS = [CORPUS + "/" + n for n in ("ids.npy", "masks.npy", "manifest.json")]
OUTPUT_NAMES = {"summary.json", "provenance.json", "history.json", "trace.json", "model_best.pt", "oof_best.csv"}
ENV = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                         "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE")}
ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
BUDGET = {"pilot_seconds": 180, "full_seconds": 3300, "total_seconds": 7200,
          "score_seconds": 180, "pilot_updates": 32, "full_updates": 3448, "retries": 0,
          "peak_gpu_allocated_cap_bytes": 20 * 2**30, "host_available_floor_bytes": 8 * 2**30,
          "disk_free_floor_bytes": 16 * 2**30}
PINNED = {"torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121", "timm": "1.0.19", "numpy": "2.0.2"}
QUEUE_HASHES = {"tools/gpu_queue.py": "3daeb8551d420dc5e64aa73de605422560e204f6a435e6d53f7c5b7ee355b51f",
                "tools/gpu_job.py": "71a007a12302bc5783042275bf97f886c02078e0bb6473916341c9013f6e55dc"}
QUEUE = [{"tag": "VAST_HC06_V1_PAIR", "command": ["python", "tools/vast_hc06.py", "run", "--protocol", "manifest.json", "--admission", "admission.json"]}]
FROZEN_ARGS = {"arch": "vit_small_patch14_dinov2.lvd142m", "res": 224, "k": 12, "k_eval": 24,
               "epochs": 8, "fixed_epoch": 8, "select": "oof", "bs": 8, "accum": 1,
               "grad_ckpt": True, "workers": 12, "amp": "fp16", "eval_amp": "fp16",
               "runtime_policy": "strict", "amp_init_scale": 1024.0, "seed": 42, "norm": "imagenet",
               "bb_lr": 3e-5, "head_lr": 0.001, "wd": 0.02, "limit": 0, "smoke": False,
               "pretrained": "local-generic-strict", "oof_subsample": 0, "folds": 5,
               "window_policy": "within-slot-valid", "device": "cuda", "fold": 2, "topk": 3}


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def contained(root, value):
    """Reject Windows drives/UNC paths even while validating on Linux."""
    need(isinstance(value, str) and value and "\\" not in value and ":" not in value, "relative POSIX path required")
    p = PurePosixPath(value)
    need(not p.is_absolute() and value == p.as_posix() and all(x not in ("..", ".") for x in p.parts), "unsafe relative path")
    result = (Path(root) / value).resolve()
    need(result.is_relative_to(Path(root).resolve()), "path escapes package")
    return result


def allowed_file(name):
    return name in SOURCE_FILES + DATA_FILES + SMALL_CORPUS + [WEIGHT, RUNNER, "config/workspace.toml", "queue.json"] or bool(
        re.fullmatch(re.escape(CORPUS) + r"/vols_shard(?:0[0-9]|1[01])\.npy", name))


def uid_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def helpers():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools import exp084_group_loss
    return exp084_group_loss


def read_rows(path): return helpers().read_rows(path)
def worker_init(worker): return helpers().worker_init(worker)
def install_gold_blocker(root): return helpers().install_gold_blocker(root)
def validate_membership(*args): return helpers().validate_membership(*args)
def validate_label_contract(*args): return helpers().validate_label_contract(*args)
def loss_weights(*args): return helpers().loss_weights(*args)


def render_runner(root=ROOT):
    text = (Path(root) / TEMPLATE).read_text(encoding="utf-8")
    old = "from tools import patch_mil as guard"
    need(text.count(old) == 1, "frozen runner guard changed")
    return text.replace(old, "from tools import vast_hc06 as guard")


def prepare(out, root=ROOT):
    root, out = Path(root), Path(out)
    need(not out.exists(), "fresh package path required")
    need(sha256(root / PARENT) == PARENT_SHA, "frozen parent changed")
    old = read(root / PARENT)
    # Only explicit input/source allowlists are copied; no archives, reports,
    # credentials, local config, old predictions, B runs, or environment tree.
    bindings = {**old["source_bindings"], **old["input_bindings"], **old["hc06_bindings"],
                **{"training/src/" + n: h for n, h in old["training_code_sha256"].items()}, **QUEUE_HASHES}
    for name in SOURCE_FILES + DATA_FILES + SMALL_CORPUS:
        if name != "tools/vast_hc06.py":
            need(sha256(root / name) == bindings[name], "frozen source/input changed: " + name)
    need(sha256(old["initial_asset"]["path"]) == WEIGHT_SHA, "generic asset changed")
    out.mkdir(parents=True)
    files = {}
    for name in SOURCE_FILES + DATA_FILES + SMALL_CORPUS + [WEIGHT]:
        src = Path(old["initial_asset"]["path"]) if name == WEIGHT else root / name
        dst = contained(out, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[name] = {"sha256": sha256(dst), "bytes": dst.stat().st_size, "payload": True}
    generated = {RUNNER: render_runner(root), "config/workspace.toml": 'schema_version = 1\ncompetition = "rsna-knee-abnormality-detection"\n',
                 "queue.json": json.dumps(QUEUE, indent=2) + "\n"}
    for name, content in generated.items():
        dst = contained(out, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8", newline="\n")
        files[name] = {"sha256": sha256(dst), "bytes": dst.stat().st_size, "payload": True}
    corpus = read(root / CORPUS / "manifest.json")
    for shard in corpus["shards"]:
        name = CORPUS + "/" + shard["file"]
        need(allowed_file(name), "unexpected corpus shard")
        files[name] = {"sha256": shard["sha256"], "bytes": (root / name).stat().st_size, "payload": False}
    keep = ("targets", "paths", "ordered_ids", "groups", "expected_ids", "split", "excluded_uids", "loss",
            "label_hash", "eval_label_hash", "split_hash", "corpus_manifest_sha256", "training_code_sha256", "prediction_columns")
    p = {key: copy.deepcopy(old[key]) for key in keep}
    p.update(schema=SCHEMA, status="REQUIRES_EXACT_COORDINATOR_ADMISSION", parent_sha256=PARENT_SHA,
             execution=copy.deepcopy(BUDGET), files=files, pinned_packages=PINNED,
             position_policy=POSITION_POLICY, expected_training_trace_sha256=TRACE_SHA,
             expected_backend=old["expected_runtime"]["runtime"]["backends"],
             initial_asset={**{k: v for k, v in old["initial_asset"].items() if k != "path"}, "path": WEIGHT},
             arms={}, phases=["CLS:pilot", "PATCH:pilot", "CLS:full", "PATCH:full"],
             evaluation={"primary": "fresh RTX3090 PATCH minus fresh RTX3090 CLS raw macro12 AUC",
                         "labels": "original Dread float32 then rint; fixed V888, all 12 targets",
                         "bootstrap": {"unit": "component_id", "draws": 500, "seed": 20261003},
                         "selection": "fixed epoch8 only; evaluate after both arms validated; no adoption/submission",
                         "historical_B": "local descriptive comparison only; not a causal comparator"},
             limitations=["Prospective Linux/RTX3090 pair; historical 4070 control cannot establish its effect",
                          "Head dropout consumes a different RNG stream by token count",
                          "Researcher-exposed weak V888, one seed, unknown patient links; not private/clinical evidence",
                          "Any failure ends this attempt; no automatic retry, resume, extra fit or resource expansion"])
    for arm in POLICIES:
        p["arms"][arm] = {k: copy.deepcopy(v) for k, v in old["arm"].items() if k not in ("outputs", "tag", "id")}
        p["arms"][arm].update(id=arm, tag="VAST_HC06_V1_" + arm, aggregation_policy=POLICIES[arm])
    write_new(out / "manifest.json", p)
    validate_manifest(out / "manifest.json", out, full=False, allow_missing_shards=True)
    print(json.dumps({"status": "PACKAGE_PREPARED_NOT_ADMITTED", "manifest": str(out / "manifest.json"),
                      "sha256": sha256(out / "manifest.json"), "bytes": sum(v["bytes"] for v in files.values()),
                      "copy_separately": [k for k, v in files.items() if not v["payload"]]}))


def validate_manifest(path, root=ROOT, full=False, allow_missing_shards=False):
    root = Path(root)
    p = read(path)
    need(p.get("schema") == SCHEMA and p.get("parent_sha256") == PARENT_SHA, "wrong protocol")
    need(p.get("execution") == BUDGET and p.get("pinned_packages") == PINNED, "budget/dependency policy changed")
    need(p.get("phases") == ["CLS:pilot", "PATCH:pilot", "CLS:full", "PATCH:full"], "phase order changed")
    need(p.get("position_policy") == POSITION_POLICY and p.get("expected_training_trace_sha256") == TRACE_SHA, "operation/trace changed")
    need(set(p["arms"]) == set(POLICIES), "exactly two arms required")
    need(p["arms"]["CLS"]["expected_args"] == p["arms"]["PATCH"]["expected_args"] == FROZEN_ARGS, "unmatched/frozen arm arguments")
    for key, arm in p["arms"].items():
        need(arm["id"] == key and arm["tag"] == "VAST_HC06_V1_" + key and arm["aggregation_policy"] == POLICIES[key]
             and arm["initial_state_sha256"] == INITIAL_SHA and arm["expected_attempts"] == 3448, "arm identity mismatch")
    required = set(SOURCE_FILES + DATA_FILES + SMALL_CORPUS + [WEIGHT, RUNNER, "config/workspace.toml", "queue.json"])
    required |= {CORPUS + f"/vols_shard{i:02d}.npy" for i in range(12)}
    need(set(p["files"]) == required and all(allowed_file(n) for n in p["files"]), "file allowlist differs")
    for name, entry in p["files"].items():
        target = contained(root, name)
        need(re.fullmatch("[0-9a-f]{64}", entry["sha256"]) and type(entry["bytes"]) is int and entry["bytes"] > 0, "bad file binding")
        is_shard = name.startswith(CORPUS + "/vols_shard")
        if is_shard and allow_missing_shards and not target.exists():
            continue
        need(target.is_file() and target.stat().st_size == entry["bytes"], "missing/size mismatch: " + name)
        if full or not is_shard:
            need(sha256(target) == entry["sha256"], "hash mismatch: " + name)
    need((root / RUNNER).read_text(encoding="utf-8") == render_runner(root), "generated runner differs")
    need(read(root / "queue.json") == QUEUE, "queue must call the bounded outer wrapper")
    need(all(p["files"][n]["sha256"] == h for n, h in QUEUE_HASHES.items()), "queue launcher source differs")
    for name in [*p["paths"].values(), p["initial_asset"]["path"], p["ordered_ids"]["path"], p["groups"]["path"]]:
        contained(root, name)
    need(p["initial_asset"]["path"] == WEIGHT and p["initial_asset"]["sha256"] == WEIGHT_SHA, "asset path/hash mismatch")
    ids = read(root / p["ordered_ids"]["path"])
    need(len(ids["train"]) == 3455 and len(ids["val"]) == 888 and not set(ids["train"]) & set(ids["val"]), "cohort differs")
    need(all(uid_hash(ids[k]) == p["ordered_ids"][k + "_sha256"] for k in ("train", "val")), "UID order differs")
    return p, ids


def remote_environment():
    import torch
    import timm
    import cv2
    import numpy
    import pandas
    import psutil
    import safetensors
    import torchvision
    import pydicom
    import timm.layers.pos_embed
    import timm.models.vision_transformer
    need(sys.platform == "linux" and torch.cuda.is_available() and torch.cuda.device_count() == 1, "one Linux GPU required")
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
                "numpy": numpy.__version__, "pandas": pandas.__version__, "psutil": psutil.__version__,
                "safetensors": safetensors.__version__, "cv2": cv2.__version__, "pydicom": pydicom.__version__}
    need(all(versions[k] == v for k, v in PINNED.items()), "pinned Linux package version differs")
    gpu = torch.cuda.get_device_properties(0)
    need(gpu.name == "NVIDIA GeForce RTX 3090" and list(torch.cuda.get_device_capability()) == [8, 6], "registered RTX3090 required")
    need(gpu.total_memory >= 23 * 2**30, "24GB GPU required")
    return {"versions": versions, "python": platform.python_version(), "platform": platform.platform(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": gpu.name,
            "gpu_memory_bytes": gpu.total_memory, "compute_capability": [8, 6],
            "timm_source_sha256": {"pos_embed": sha256(timm.layers.pos_embed.__file__),
                                   "vision_transformer": sha256(timm.models.vision_transformer.__file__)}}


def preflight(path):
    os.environ.update(ENV)
    p, _ = validate_manifest(path, full=True)
    import psutil
    env = remote_environment()
    need(psutil.virtual_memory().available >= BUDGET["host_available_floor_bytes"], "insufficient host RAM")
    need(shutil.disk_usage(ROOT).free >= BUDGET["disk_free_floor_bytes"], "insufficient free disk")
    need(shutil.disk_usage("/dev/shm").free >= 2 * 2**30, "frozen twelve-worker loader needs at least2GiB free /dev/shm")
    # The package's exact installed source/versions are frozen before any pilot.
    value = {"schema": SCHEMA, "status": "VERIFIED_INPUTS_ENVIRONMENT", "manifest_sha256": sha256(path),
             "environment": env, "file_stats": {n: {"bytes": (ROOT / n).stat().st_size,
                 "mtime_ns": (ROOT / n).stat().st_mtime_ns} for n in p["files"]}}
    write_new(Path(path).with_name("preflight.json"), value)
    print(json.dumps(value))


def check_preflight(path, protocol):
    value = read(Path(path).with_name("preflight.json"))
    need(value.get("status") == "VERIFIED_INPUTS_ENVIRONMENT" and value.get("manifest_sha256") == sha256(path), "preflight absent/mismatched")
    for name, expected in value["file_stats"].items():
        file = contained(ROOT, name)
        need(expected == {"bytes": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns}, "file changed since full hash: " + name)
    need(set(value["file_stats"]) == set(protocol["files"]), "incomplete preflight inventory")
    return value


def arm_for(protocol, tag):
    matches = [a for a in protocol["arms"].values() if tag in (a["tag"], a["tag"] + "_PILOT")]
    need(len(matches) == 1, "unregistered arm tag")
    return matches[0]


def validate_arguments(args, root):
    global AGGREGATION_POLICY
    p, ids = validate_manifest(args.protocol, root)
    arm = arm_for(p, args.tag)
    need(args.tag == arm["tag"] + ("_PILOT" if args.phase == "pilot" else ""), "tag/phase mismatch")
    for k, v in arm["expected_args"].items():
        need(getattr(args, k, None) == v, "frozen argument differs: " + k)
    for k, v in p["paths"].items():
        need(Path(getattr(args, k)).resolve() == contained(root, v), "frozen path differs: " + k)
    need(all(os.environ.get(k) == v for k, v in ENV.items()), "thread/offline environment differs")
    # Generated runners may only run under this outer admission's active phase.
    execution = read(Path(args.protocol).with_name("execution.json"))
    need(execution.get("status") == "RUNNING" and execution.get("manifest_sha256") == sha256(args.protocol)
         and execution.get("active_phase") == arm["id"] + ":" + args.phase, "outer run admission required")
    preflight_record = check_preflight(args.protocol, p)
    # Import runtime dependencies BEFORE the original runner's seed boundary.
    need(remote_environment() == preflight_record["environment"], "worker environment differs from preflight")
    AGGREGATION_POLICY = arm["aggregation_policy"]
    return p, ids


def build_offline_model(arch, asset):
    from tools import exp086_dinov2s_v2 as base
    from training.adapters.dinov2_patch_mil import install
    need(AGGREGATION_POLICY in POLICIES.values(), "arm not established")
    model = base.build_offline_model(arch, {**asset, "path": str(contained(ROOT, asset["path"]))})
    return install(model) if AGGREGATION_POLICY == POLICIES["PATCH"] else model


def validate_pretraining(provenance, env, protocol, arm):
    p = provenance
    expected = {"code_sha256": protocol["training_code_sha256"], "initial_state_sha256": INITIAL_SHA,
                "train_ids_sha256": protocol["ordered_ids"]["train_sha256"], "eval_ids_sha256": protocol["ordered_ids"]["val_sha256"],
                "train_ids_ordered_sha256": protocol["ordered_ids"]["train_sha256"], "eval_ids_ordered_sha256": protocol["ordered_ids"]["val_sha256"],
                "corpus_manifest_sha256": protocol["corpus_manifest_sha256"], "checkpoint_selection": "fixed-final",
                "sampling": "sha256(seed,epoch,uid,stream)-v1", "gold_access": "disabled", "loss_policy": arm["loss_policy"],
                "metric_access": "deferred_until_final_candidate", "initial_asset_sha256": WEIGHT_SHA,
                "position_policy": POSITION_POLICY, "aggregation_policy": arm["aggregation_policy"],
                "requested_workers": 12, "effective_workers": 12, "internal_cpu_threads": 1}
    need(all(p.get(k) == v for k, v in expected.items()), "pretraining provenance differs")
    for name in ("runtime", "eval_runtime"):
        runtime = p[name]
        need(runtime["requested_amp"] == runtime["effective_amp"] == "fp16" and runtime["policy"] == "strict"
             and runtime["precision_fallback_allowed"] is False and runtime["device"] == "cuda"
             and runtime["compute_capability"] == [8, 6] and runtime["backends"] == protocol["expected_backend"], "strict runtime changed")
    expected_env = read(ROOT / "preflight.json")["environment"]
    observed = remote_environment()
    need(observed == expected_env, "environment changed after preflight")
    p.update(remote_environment=observed, arm=arm["id"])


def projection(pilot):
    times = pilot.get("step_seconds", [])
    need(len(times) == 32 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in times), "32 finite pilot intervals required")
    eval_seconds, initial = pilot["eval_seconds"], pilot["initial_seconds"]
    need(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in (eval_seconds, initial)), "pilot overhead timing invalid")
    # Includes measured K24 inference on four T studies, extrapolated to V888.
    return 1.3 * sum(times[8:]) / 24 * 3448 + max(300, 1.3 * eval_seconds * 888 / 4 + initial + 30)


def finish_pilot(out, protocol, provenance, timings, updates, skips, initial_seconds, eval_seconds, peak):
    need(updates == 32 and skips == 0, "pilot did not make exactly32 successful updates")
    value = dict(schema=SCHEMA, status="PASS", phase="pilot", arm=provenance["arm"],
                 protocol_sha256=provenance["protocol_sha256"], initial_state_sha256=INITIAL_SHA,
                 initial_asset_sha256=WEIGHT_SHA, position_policy=POSITION_POLICY, aggregation_policy=AGGREGATION_POLICY,
                 updates=updates, skips=skips, step_seconds=timings, initial_seconds=initial_seconds, eval_seconds=eval_seconds,
                 peak_gpu_allocated_bytes=peak, training_ids_only=True, eval_train_studies=4, validation_rows=0,
                 metric_access="disabled", checkpoints_saved=False, predictions_saved=False, argv=[sys.executable, *sys.argv])
    value["projected_full_seconds"] = projection(value)
    write_new(Path(out) / "receipt.json", value)


def validate_completion(out, protocol, arm):
    out = Path(out)
    s, provenance, history = read(out / "summary.json"), read(out / "provenance.json"), read(out / "history.json")
    need(s["selected_epoch_1based"] == 8 and s["best_epoch"] == 7 and s["optimizer_updates"] == 3448 and s["skipped_updates"] == 0
         and s["weak_oof_macro_auc"] is None and s["metric_access"] == "deferred_until_final_candidate"
         and s["train"] == 3455 and s["val"] == 888 and s["arm"] == arm["id"], "full completion contract failed")
    need(s["provenance"] == provenance and provenance["initial_state_sha256"] == INITIAL_SHA
         and provenance["aggregation_policy"] == arm["aggregation_policy"], "provenance mismatch")
    need(len(history) == 8 and all(h["epoch"] == i and h["optimizer_updates"] == (i + 1) * 431
         and h["skipped_updates"] == 0 and h["oof_auc"] is None and h["gold_auc"] is None
         and math.isfinite(h["loss"]) for i, h in enumerate(history)), "fixed-final eight-epoch history differs")
    ids = read(ROOT / protocol["ordered_ids"]["path"])["val"]
    with (out / "oof_best.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        need(reader.fieldnames == protocol["prediction_columns"] and [r["StudyInstanceUID"] for r in rows] == ids, "prediction UID/target order differs")
    need(all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1 for r in rows for t in protocol["targets"]), "invalid prediction probabilities")


def finish_full(out, protocol, provenance, trace, elapsed):
    out = Path(out)
    arm = protocol["arms"][provenance["arm"]]
    validate_completion(out, protocol, arm)
    need(trace == TRACE_SHA, "batch trace differs from frozen scientific recipe")
    write_new(out / "trace.json", {"schema": SCHEMA, "training_trace_sha256": trace, "epochs": 8, "updates": 3448, "skips": 0})
    write_new(out / "receipt.json", {"schema": SCHEMA, "status": "PASS", "phase": "full", "arm": arm["id"],
              "protocol_sha256": provenance["protocol_sha256"], "initial_state_sha256": INITIAL_SHA, "initial_asset_sha256": WEIGHT_SHA,
              "position_policy": POSITION_POLICY, "aggregation_policy": arm["aggregation_policy"], "updates": 3448, "skips": 0,
              "validation_rows": 888, "training_trace_sha256": trace, "elapsed_seconds": elapsed, "argv": [sys.executable, *sys.argv],
              "outputs": {n: {"sha256": sha256(out / n), "bytes": (out / n).stat().st_size} for n in sorted(OUTPUT_NAMES)}})


def phase_command(protocol, path, arm, phase):
    command = [sys.executable, str(ROOT / RUNNER), "--protocol", str(Path(path).resolve()), "--phase", phase,
               "--tag", arm["tag"] + ("_PILOT" if phase == "pilot" else "")]
    for key, value in {**arm["expected_args"], **protocol["paths"]}.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value: command.append(option)
        else:
            command.extend([option, str(ROOT / value) if key in protocol["paths"] else str(value)])
    return command


def validate_result(result, p, path, arm, phase, command, out):
    need(result.get("schema") == SCHEMA and result.get("status") == "PASS" and result.get("phase") == phase
         and result.get("arm") == arm["id"] and result.get("argv") == command and result.get("protocol_sha256") == sha256(path)
         and result.get("initial_state_sha256") == INITIAL_SHA and result.get("initial_asset_sha256") == WEIGHT_SHA
         and result.get("position_policy") == POSITION_POLICY and result.get("aggregation_policy") == arm["aggregation_policy"], "phase receipt identity differs")
    need(result.get("updates") == (32 if phase == "pilot" else 3448) and result.get("skips") == 0, "unsuccessful or skipped updates")
    if phase == "pilot":
        need(result.get("training_ids_only") is True and result.get("validation_rows") == 0 and result.get("eval_train_studies") == 4
             and result.get("metric_access") == "disabled" and result.get("checkpoints_saved") is False
             and result.get("predictions_saved") is False, "pilot exposure contract differs")
        need(not any((Path(out) / n).exists() for n in ("model_best.pt", "oof_best.csv")), "pilot saved forbidden predictions/checkpoint")
        need(result.get("projected_full_seconds") == projection(result) and projection(result) <= BUDGET["full_seconds"], "pilot projects beyond cap")
        need(0 < result.get("peak_gpu_allocated_bytes", 0) <= BUDGET["peak_gpu_allocated_cap_bytes"], "pilot GPU peak exceeds cap")
    else:
        validate_completion(out, p, arm)
        need(result.get("validation_rows") == 888 and result.get("training_trace_sha256") == TRACE_SHA, "full cohort/trace differs")
        need(set(result.get("outputs", {})) == OUTPUT_NAMES, "six artifact inventory required")
        for name, value in result["outputs"].items():
            need((Path(out) / name).stat().st_size == value["bytes"] and sha256(Path(out) / name) == value["sha256"], "output inventory differs")


def execute_pair(path, admission):
    need(Path(path).resolve() == ROOT / "manifest.json", "run manifest at package root")
    p, _ = validate_manifest(path)
    a = read(admission)
    need(a.get("status") == "READY" and a.get("manifest_sha256") == sha256(path) and a.get("allow_execution") is True,
         "coordinator exact-manifest admission required")
    launch = read(ROOT / "launch.json")
    need(launch.get("status") == "RUNNING" and launch.get("manifest_sha256") == sha256(path)
         and launch.get("admission_sha256") == sha256(admission), "outer7200s supervisor admission required")
    check_preflight(path, p)
    os.environ.update(ENV)
    need(remote_environment() == read(ROOT / "preflight.json")["environment"], "preflight environment changed")
    from tools.process_resource_receipt import supervise
    from tools.exp086_dinov2s_v2 import resource_gate
    import psutil
    sentinel = Path(path).with_name("execution.json")
    state = {"schema": SCHEMA, "status": "RUNNING", "manifest_sha256": sha256(path), "admission_sha256": sha256(admission), "retries": 0}
    write_new(sentinel, state)  # Exclusive and durable before any child launch; never removed.
    started = time.monotonic()
    phases, pilots, failure = [], {}, None
    try:
        for name in p["phases"]:
            arm_name, phase = name.split(":")
            arm, limit = p["arms"][arm_name], BUDGET[phase + "_seconds"]
            out = ROOT / "artifacts/experiments" / (arm["tag"] + ("_PILOT" if phase == "pilot" else ""))
            need(not out.exists() and not out.with_name(out.name + "_RESOURCE").exists(), "fresh arm paths required; retry forbidden")
            need(psutil.virtual_memory().available >= BUDGET["host_available_floor_bytes"], "insufficient available RAM")
            need(shutil.disk_usage(ROOT).free >= BUDGET["disk_free_floor_bytes"], "insufficient free disk")
            if phase == "full":
                need(set(pilots) == set(POLICIES), "both pilots required before either full fit")
                need(sum(projection(x) for x in pilots.values()) + 360 + BUDGET["score_seconds"] <= BUDGET["total_seconds"], "pair projection exceeds total cap")
            remaining = BUDGET["total_seconds"] - (time.monotonic() - started)
            need(remaining >= limit + BUDGET["score_seconds"], "insufficient aggregate phase budget")
            state["active_phase"] = name
            sentinel.write_text(json.dumps(state, indent=2), encoding="utf-8")
            command = phase_command(p, path, arm, phase)
            resource = supervise(command, out.with_name(out.name + "_RESOURCE"), limit, ROOT)
            resource_gate(resource, command, limit)
            result = read(out / "receipt.json")
            validate_result(result, p, path, arm, phase, command, out)
            if phase == "pilot": pilots[arm_name] = result
            phases.append({"name": name, "receipt_sha256": sha256(out / "receipt.json"),
                           "resource_receipt_sha256": sha256(out.with_name(out.name + "_RESOURCE") / "receipt.json")})
        # Equality is verified independently of each arm's own receipt.
        left = read(ROOT / "artifacts/experiments" / p["arms"]["CLS"]["tag"] / "provenance.json")
        right = read(ROOT / "artifacts/experiments" / p["arms"]["PATCH"]["tag"] / "provenance.json")
        for key in ("initial_state_sha256", "train_ids_ordered_sha256", "eval_ids_ordered_sha256", "actual_pos_weight", "runtime", "eval_runtime", "remote_environment"):
            need(left[key] == right[key], "paired provenance differs: " + key)
        write_new(ROOT / "PAIR_VALIDATED.json", {"schema": SCHEMA, "manifest_sha256": sha256(path), "phases": phases,
                                                "status": "PAIR_VALIDATED_METRICS_PENDING"})
        remaining = BUDGET["total_seconds"] - (time.monotonic() - started)
        need(remaining >= BUDGET["score_seconds"], "no score budget remains")
        command = [sys.executable, str(ROOT / "tools/vast_hc06.py"), "score", "--protocol", str(Path(path).resolve())]
        resource = supervise(command, ROOT / "SCORE_RESOURCE", BUDGET["score_seconds"], ROOT)
        resource_gate(resource, command, BUDGET["score_seconds"])
        need(read(ROOT / "comparison.json")["manifest_sha256"] == sha256(path), "score binding differs")
        need(time.monotonic() - started <= BUDGET["total_seconds"], "aggregate cap exceeded")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
    record = {**state, "status": "INVALID" if failure else "VALIDATED_SUCCESS", "failure": failure,
              "valid_for_scoring": failure is None, "elapsed_seconds": time.monotonic() - started, "phases": phases}
    sentinel.write_text(json.dumps(record, indent=2), encoding="utf-8")
    write_new(ROOT / ("INVALID.json" if failure else "PAIR_COMPLETE.json"), record)
    print(json.dumps(record))
    return 1 if failure else 0


def run(path, admission):
    """Total-wall-time supervisor includes all workers, validators and scoring."""
    need(Path(path).resolve() == ROOT / "manifest.json", "run manifest at package root")
    p, _ = validate_manifest(path)
    a = read(admission)
    need(a.get("status") == "READY" and a.get("manifest_sha256") == sha256(path) and a.get("allow_execution") is True,
         "coordinator exact-manifest admission required")
    check_preflight(path, p)
    os.environ.update(ENV)
    from tools.process_resource_receipt import supervise
    from tools.exp086_dinov2s_v2 import resource_gate
    # Taking this exclusive sentinel counts as the sole launch attempt, including
    # early environment/supervisor failure. A stale sentinel is never removed.
    write_new(ROOT / "launch.json", {"status": "RUNNING", "manifest_sha256": sha256(path),
                                    "admission_sha256": sha256(admission), "retries": 0})
    command = [sys.executable, str(ROOT / "tools/vast_hc06.py"), "_pair", "--protocol", str(Path(path).resolve()),
               "--admission", str(Path(admission).resolve())]
    failure = None
    try:
        receipt = supervise(command, ROOT / "PAIR_RESOURCE", BUDGET["total_seconds"], ROOT)
        resource_gate(receipt, command, BUDGET["total_seconds"])
        result = read(ROOT / "PAIR_COMPLETE.json")
        need(result.get("status") == "VALIDATED_SUCCESS" and result.get("valid_for_scoring") is True
             and result.get("manifest_sha256") == sha256(path), "pair output not validated")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
    result = {"schema": SCHEMA, "status": "INVALID" if failure else "VALIDATED_SUCCESS", "failure": failure,
              "valid_for_scoring": failure is None, "manifest_sha256": sha256(path),
              "pair_resource_sha256": sha256(ROOT / "PAIR_RESOURCE/receipt.json") if (ROOT / "PAIR_RESOURCE/receipt.json").is_file() else None}
    write_new(ROOT / ("OUTER_INVALID.json" if failure else "VALIDATED_SUCCESS.json"), result)
    (ROOT / "launch.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 1 if failure else 0


def auc12(y, predictions):
    import numpy as np
    values = []
    for t in range(12):
        mask = np.isfinite(y[:, t])
        truth, score = y[mask, t], predictions[mask, t]
        positive, negative = int(truth.sum()), int(len(truth) - truth.sum())
        # Exact average-tie Mann-Whitney ranks, independent of sklearn/scipy.
        order = np.argsort(score, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
        stops = np.r_[starts[1:], len(score)]
        ranks = np.empty(len(score), dtype=np.float64)
        if len(score):
            ranks[order] = np.repeat((starts + stops + 1) / 2, stops - starts)
        values.append(float((ranks[truth == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)) if positive and negative else None)
    return values


def score(path):
    """CPU-only paired scoring; no epoch choice, ratio, adoption, or submission."""
    p, ids = validate_manifest(path)
    outer = read(ROOT / "PAIR_VALIDATED.json")
    need(outer["manifest_sha256"] == sha256(path) and outer["status"] == "PAIR_VALIDATED_METRICS_PENDING", "outer pair validation required")
    import numpy as np
    sys.path.insert(0, str(ROOT / "training/src"))
    from raptor_common import load_label_csv
    pred = {}
    for arm_name, arm in p["arms"].items():
        out = ROOT / "artifacts/experiments" / arm["tag"]
        validate_result(read(out / "receipt.json"), p, path, arm, "full", phase_command(p, path, arm, "full"), out)
        pred[arm_name] = np.array([[float(r[t]) for t in p["targets"]] for r in read_rows(out / "oof_best.csv")])
    labels, digest = load_label_csv(ROOT / p["paths"]["eval_labels"])
    need(digest == p["eval_label_hash"], "score label semantic hash differs")
    y = np.rint(np.stack([labels[u] for u in ids["val"]]).astype(np.float32))
    rows = {r["StudyInstanceUID"]: r for r in read_rows(ROOT / p["groups"]["path"])}
    groups = {}
    for i, uid in enumerate(ids["val"]): groups.setdefault(rows[uid]["component_id"], []).append(i)
    groups = [np.asarray(groups[k], dtype=int) for k in sorted(groups)]
    baseline, candidate = auc12(y, pred["CLS"]), auc12(y, pred["PATCH"])
    need(None not in baseline and None not in candidate, "primary macro must contain all12 targets")
    rng, deltas, missing = np.random.default_rng(20261003), [], 0
    for _ in range(500):
        ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        a, b = auc12(y[ix], pred["PATCH"][ix]), auc12(y[ix], pred["CLS"][ix])
        if None in a or None in b:
            missing += 1  # Never silently switch the macro's target set.
        else:
            deltas.append(float(np.mean(a) - np.mean(b)))
    need(deltas, "no valid fixed12 bootstrap draw")
    diagnostics = {}
    for column in p["groups"]["subgroup_columns"]:
        diagnostics[column] = {}
        for value in sorted({rows[u][column] for u in ids["val"]}):
            ix = [i for i, u in enumerate(ids["val"]) if rows[u][column] == value]
            a, b = auc12(y[ix], pred["PATCH"][ix]), auc12(y[ix], pred["CLS"][ix])
            diagnostics[column][value] = {"rows": len(ix), "PATCH": a, "CLS": b,
                "fixed12_delta": float(np.mean(a) - np.mean(b)) if None not in a and None not in b else None}
    result = {"schema": SCHEMA, "manifest_sha256": sha256(path), "status": "COMPLETE", "rows": 888, "targets": p["targets"],
              "CLS_macro12": float(np.mean(baseline)), "PATCH_macro12": float(np.mean(candidate)),
              "delta_PATCH_minus_CLS": float(np.mean(candidate) - np.mean(baseline)),
              "targets_CLS": baseline, "targets_PATCH": candidate, "target_positive_counts": y.sum(0).astype(int).tolist(),
              "paired_component_ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
              "bootstrap": {"draws": 500, "seed": 20261003, "valid_fixed12": len(deltas), "undefined_target_draws": missing},
              "groups": diagnostics, "selection": "fixed-final; no adoption decision", "limitations": p["limitations"]}
    write_new(ROOT / "comparison.json", result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--out", type=Path, required=True)
    for name in ("verify", "preflight", "run", "score", "_pair"):
        q = sub.add_parser(name); q.add_argument("--protocol", type=Path, required=True)
        if name in ("run", "_pair"): q.add_argument("--admission", type=Path, required=True)
    args = ap.parse_args()
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    if args.action == "prepare": prepare(args.out)
    elif args.action == "verify":
        validate_manifest(args.protocol, full=True)
        print(json.dumps({"status": "VERIFIED_PACKAGE", "manifest_sha256": sha256(args.protocol)}))
    elif args.action == "preflight": preflight(args.protocol)
    elif args.action == "run": return run(args.protocol, args.admission)
    elif args.action == "_pair": return execute_pair(args.protocol, args.admission)
    elif args.action == "score": score(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
