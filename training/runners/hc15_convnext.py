"""Guarded H-C-15 preparation runner; importing this file uses only the stdlib.

This is a whole-recipe adaptation of goodpjw2008's Apache-2.0 reader, with
original Dread T3455/V888, offline generic initialization and fixed-final EMA.
Reference source is retained as DATA at artifacts/research/20261007_open_baselines_v1.
Downloaded Python is never imported. The local adapter implements the model.

Execution is deliberately unavailable without a separately reviewed admission:
  python -m training.runners.hc15_convnext inspect --manifest ... --manifest-sha256 ...
  python -m training.runners.hc15_convnext train --manifest ... --manifest-sha256 ...
      --admission ... --cache-manifest ... --cache-dir ... --output ...
  python -m training.runners.hc15_convnext serve [same arguments]
      --checkpoint ... --checkpoint-sha256 ...

``serve`` writes only the frozen validation cohort, after a final checkpoint;
arbitrary test/Gold serving and raw DICOM decoding need separate reviewed work.
The reviewed outer wrapper must enforce exclusive GPU ownership and a hard wall
timeout. This module does not grant permission, reserve a GPU, or create a gate.
Checkpoints resume at COMPLETE EPOCH BOUNDARIES ONLY; an interrupted epoch is
replayed from the preceding boundary, never resumed from a partial accumulation.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import hmac
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "training/adapters/goodpjw_reader.py"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ARCH = "convnext_tiny.in12k_ft_in1k"
TRAIN_N, VAL_N, EPOCHS, EFFECTIVE_BATCH = 3455, 888, 14, 8
UPDATES_PER_EPOCH = TRAIN_N // EFFECTIVE_BATCH
TOTAL_UPDATES = UPDATES_PER_EPOCH * EPOCHS
_EXECUTION_CAPABILITY = object()
RECIPE = {
    "microbatch": 1, "accumulation": 8, "effective_batch": 8, "num_workers": 0,
    "seed": 42, "epochs": 14, "successful_updates": 6034,
    "cache_size": 384, "model_res": 256, "slots": 6,
    "train_windows_per_slot": 8, "eval_windows_per_slot": 12,
    "optimizer": "AdamW", "encoder_lr": 1e-4, "head_lr": 5e-4,
    "weight_decay": .02, "clip_grad": 3., "ema": .998,
    "loss": "plain_soft_bce", "precision": "native_cuda_bf16_only",
    "checkpoint_selection": "fixed_final_ema", "validation": "final_only",
    "shuffle": "dedicated_torch_generator_once_per_epoch_drop_last_at_effective8",
    "scheduler": "source_lambda_warmup_times_cosine_optimizer_updates",
    "resume": "complete_epoch_boundary_only", "internal_cpu_threads": 1,
    "adaptation": "microbatch1_accum8_changes_dropout_and_augmentation_rng_vs_source_batch8",
}


class ContractError(RuntimeError):
    """An admission or frozen-input contract failed; no fallback is allowed."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid_sha(uids):
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def path_in(root, relative):
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Path escapes its declared root")
    return path


def check_binding(path, binding):
    path = Path(path)
    require(path.is_file(), f"Missing bound file: {path}")
    if "bytes" in binding:
        require(path.stat().st_size == binding["bytes"], f"File size changed: {path.name}")
    require(sha256(path) == binding["sha256"], f"SHA256 changed: {path.name}")


def generated_binding(manifest, basename):
    matches = [b for b in manifest["generated"] if Path(b["path"]).name == basename]
    require(len(matches) == 1, f"Expected exactly one binding for {basename}")
    return matches[0]


def load_frozen_inputs(manifest_path, expected_sha, root=ROOT):
    """Metadata/labels only. Never opens raw labels, Gold, image caches or CUDA."""
    require(len(expected_sha) == 64, "An explicit manifest SHA256 is required")
    check_binding(manifest_path, {"sha256": expected_sha})
    manifest = read_json(manifest_path)
    require(manifest["schema"] == "hc15-open-recipe-preparation-v1", "Wrong manifest schema")
    require(manifest["source_plan"] == "H-C-15", "Wrong source plan")
    require(manifest["initialization"]["architecture"] == ARCH, "Initialization architecture changed")
    fixed = {"cache_size": 384, "mm_per_pixel": .4, "max_slices": 64,
             "model_res": 256, "slots": 6, "transformer_depth": 2,
             "train_windows_per_slot": 8, "eval_windows_per_slot": 12,
             "epochs": 14, "effective_batch": 8, "successful_updates": 6034,
             "optimizer": "AdamW", "encoder_lr": 1e-4, "head_lr": 5e-4,
             "weight_decay": .02, "ema": .998, "clip_grad": 3}
    require(all(manifest["recipe"].get(k) == v for k, v in fixed.items()), "Frozen recipe changed")
    cohort_binding = generated_binding(manifest, "cohort.json")
    cohort_path = path_in(root, cohort_binding["path"])
    check_binding(cohort_path, cohort_binding)
    cohort = read_json(cohort_path)
    train, val = cohort["train_uids"], cohort["val_uids"]
    require(len(train) == TRAIN_N and len(val) == VAL_N, "Frozen cohort counts changed")
    require(all(isinstance(u, str) and u for u in train + val), "Invalid study UID")
    require(len(set(train + val)) == TRAIN_N + VAL_N, "UID overlap or duplicate")
    require(uid_sha(train) == manifest["cohort"]["train_uid_sha256"], "Training order changed")
    require(uid_sha(val) == manifest["cohort"]["validation_uid_sha256"], "Validation order changed")
    require(cohort["targets"] == LABELS, "Target order changed")
    require(cohort["target_mask"] == "all twelve present per study; no adaptive masking", "Mask changed")
    require(manifest["cohort"]["gold_excluded"] == 58, "Gold exclusion receipt changed")
    for key, count in (("train_soft_float32", TRAIN_N), ("val_soft_float32", VAL_N),
                       ("val_binary", VAL_N)):
        rows = cohort[key]
        require(len(rows) == count and all(len(row) == 12 for row in rows), f"Bad {key} shape")
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
                    for row in rows for v in row), f"Nonfinite/out-of-range {key}")
    require(cohort["val_binary"] == [[round(v) for v in row] for row in cohort["val_soft_float32"]],
            "Validation binary labels differ from frozen float32->rint")
    return manifest, cohort


def npy_header(path):
    """Read a bounded NPY header without numpy, pixels or pickle."""
    with Path(path).open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "Invalid NPY magic")
        version = stream.read(2)
        require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Unsupported NPY version")
        width = 2 if version[0] == 1 else 4
        length = int.from_bytes(stream.read(width), "little")
        require(0 < length <= 65536, "Unbounded NPY header")
        header = ast.literal_eval(stream.read(length).decode("utf-8" if version[0] == 3 else "latin1"))
        return header, stream.tell()


def validate_cache_metadata(cache_path, cache_dir, manifest, cohort, manifest_sha,
                            adapter_sha, root=ROOT):
    """Check all series metadata and NPY headers, without loading pixel payloads.

    Files are hashed lazily on first actual use, then size/mtime checked on reuse.
    Metadata includes failed/ineligible series with n<3 and no pixel path.
    ``n`` is retained cache length (<=64), as in the source series selector;
    ``decoded_count`` separately records successful frames before retention.
    """
    cache = read_json(cache_path)
    require(cache.get("schema") == "hc15-cache-v1" and cache.get("complete") is True,
            "A complete reviewed HC15 cache is required")
    require(cache.get("source_manifest_sha256") == manifest_sha, "Cache input manifest differs")
    require(cache.get("adapter_sha256") == adapter_sha, "Cache adapter differs")
    require(cache.get("contract") == {"size": 384, "mm_per_pixel": .4, "max_slices": 64,
                                       "dtype": "uint8", "slots": 6}, "Cache geometry differs")
    binding = generated_binding(manifest, "series.csv")
    series_path = path_in(root, binding["path"])
    check_binding(series_path, binding)
    with series_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    allowed = set(cohort["train_uids"] + cohort["val_uids"])
    require(all(r["StudyInstanceUID"] in allowed for r in rows), "Noncohort series metadata")
    series = cache["series"]
    require(isinstance(series, dict), "Cache series must be keyed by series UID")
    require(len(rows) == len({r["SeriesInstanceUID"] for r in rows}), "Duplicate series metadata")
    require(set(series) == {r["SeriesInstanceUID"] for r in rows}, "Cache series membership differs")
    valid_studies, files, counts = set(), {}, {}
    for row in rows:
        uid = row["SeriesInstanceUID"]
        record = series[uid]
        n, decoded = record["n"], record["decoded_count"]
        require(type(n) is int and type(decoded) is int and 0 <= n <= 64 and decoded >= n,
                "Invalid decoded/retained count")
        counts[uid] = n
        if n < 3:
            require(record.get("ineligible") is True and record.get("reason"), "Undocumented rejected series")
            continue
        require(n == min(decoded, 64), "Retention count differs from recipe")
        file = path_in(cache_dir, record["path"])
        require(file.is_file() and file.stat().st_size == record["bytes"], "Cache file missing/size changed")
        require(len(record["sha256"]) == 64, "Cache file hash is missing")
        header, offset = npy_header(file)
        require(header == {"descr": "|u1", "fortran_order": False, "shape": (n, 384, 384)},
                "Cache NPY geometry/dtype changed")
        require(offset + n * 384 * 384 == record["bytes"], "NPY payload length changed")
        files[uid] = {**record, "absolute_path": str(file)}
        valid_studies.add(row["StudyInstanceUID"])
    require(valid_studies == allowed, "At least one frozen study has no valid decoded series")
    return rows, counts, files


def execution_admission(args, root=ROOT):
    """Fail before importing torch/adapter or touching CUDA unless every gate matches."""
    require(args.admission and Path(args.admission).is_file(),
            "Execution blocked: separate reviewed execution-admission JSON is absent")
    admission = read_json(args.admission)
    require(admission.get("schema") == "hc15-execution-admission-v1", "Wrong admission schema")
    require(admission.get("status") == "READY", "Execution admission is not READY")
    review = admission.get("review", {})
    require(review.get("status") == "READY" and review.get("independent") is True
            and review.get("author") and review.get("reviewer")
            and review.get("reviewer") != review.get("author"),
            "Independent readiness review is absent")
    # This prototype deliberately cannot authorize a full fit from a generic
    # READY file. The reviewed pilot/cache/decision protocol has not been built.
    require(False, "Execution admission is not implemented for pilot/CPU/cost/decision receipts; "
                   "a separately reviewed wrapper and evidence validator are required")
    require(args.command in admission.get("allowed_actions", []), "Action not admitted")
    require(admission.get("recipe") == RECIPE, "Reviewed microbatch/RNG/recipe adaptation differs")
    require(0 < admission.get("max_wall_seconds", 0) <= 14400, "Bounded wall-clock contract missing")
    require(Path(args.output).resolve() == Path(admission["output_dir"]).resolve(), "Output path not admitted")
    require(Path(args.cache_dir).resolve() == Path(admission["cache_dir"]).resolve(), "Cache path not admitted")
    token = os.environ.get("HC15_OUTER_WRAPPER_TOKEN", "")
    require(len(token) >= 32 and hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(),
            admission.get("outer_wrapper_token_sha256", "")), "Approved outer-wrapper token missing")
    wrapper = admission["outer_wrapper"]
    check_binding(wrapper["path"], wrapper)
    require(os.environ.get("HC15_OUTER_WRAPPER_SHA256") == wrapper["sha256"], "Outer wrapper identity differs")
    manifest, cohort = load_frozen_inputs(args.manifest, args.manifest_sha256, root)
    bindings = {"manifest": args.manifest_sha256, "runner": sha256(__file__),
                "adapter": sha256(ADAPTER), "cache_manifest": sha256(args.cache_manifest),
                "cohort": generated_binding(manifest, "cohort.json")["sha256"],
                "initialization": manifest["initialization"]["asset"]["sha256"],
                "outer_wrapper": wrapper["sha256"]}
    require(admission.get("bindings") == bindings and review.get("bindings") == bindings,
            "Admission/review hashes do not bind the exact inputs and implementation")
    checkpoint = ({"path": str(Path(args.checkpoint).resolve()), "sha256": args.checkpoint_sha256}
                  if args.checkpoint else None)
    require(admission.get("checkpoint") == checkpoint and review.get("checkpoint") == checkpoint,
            "Resume/serving checkpoint is not explicitly admitted and reviewed")
    if checkpoint:
        check_binding(checkpoint["path"], checkpoint)
    check_binding(manifest["initialization"]["asset"]["path"], manifest["initialization"]["asset"])
    for package, version in manifest["environment"]["packages"].items():
        require(importlib.metadata.version(package) == version, f"Package version differs: {package}")
    require(sys.version == manifest["environment"]["python"], "Python version differs")
    cache = validate_cache_metadata(args.cache_manifest, args.cache_dir, manifest, cohort,
                                    args.manifest_sha256, bindings["adapter"], root)
    return admission, manifest, cohort, bindings, cache


class VerifiedSeriesLoader:
    """Hash before first pixel read and reject changed files during this process."""
    def __init__(self, files):
        self.files, self.verified = files, {}

    def __call__(self, uid):
        import numpy as np
        record = self.files[uid]
        file = Path(record["absolute_path"])
        stat = file.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if uid not in self.verified:
            check_binding(file, record)
            self.verified[uid] = stamp
        require(self.verified[uid] == stamp, "Cache file changed during execution")
        volume = np.load(file, mmap_mode="r", allow_pickle=False)
        require(volume.dtype == np.uint8 and volume.shape == (record["n"], 384, 384), "Cache array changed")
        return volume


def scheduler_multiplier(step, warm=UPDATES_PER_EPOCH, total=TOTAL_UPDATES):
    """Exact source formula: linear warmup multiplied by cosine from step zero."""
    return min(1., (step + 1) / warm) * (.5 * (1 + math.cos(math.pi * min(1., step / total))))


def epoch_order(count, generator, effective_batch=EFFECTIVE_BATCH):
    import torch
    order = torch.randperm(count, generator=generator).tolist()
    usable = count // effective_batch * effective_batch
    return order, [order[start:start + effective_batch] for start in range(0, usable, effective_batch)]


def capture_rng():
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"kind": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                          if torch.cuda.is_initialized() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    saved = state["numpy"]
    np.random.set_state((saved["kind"], np.asarray(saved["keys"], dtype=np.uint32),
                         saved["position"], saved["has_gauss"], saved["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        require(torch.cuda.is_initialized(), "Cannot restore CUDA RNG before admitted device initialization")
        require(len(state["torch_cuda"]) == torch.cuda.device_count(), "CUDA device count changed on resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def save_boundary(path, model, ema, optimizer, scheduler, order_generator, *,
                  epoch_next, step, last_order, bindings, count=TRAIN_N,
                  effective_batch=EFFECTIVE_BATCH):
    import torch
    require(step == epoch_next * (count // effective_batch), "Checkpoint is not an epoch boundary")
    require(scheduler.state_dict()["last_epoch"] == step, "Scheduler update counter differs")
    require(sorted(last_order) == list(range(count)), "Checkpoint data order is not a permutation")
    path = Path(path)
    require(not path.exists(), "Refusing to overwrite a checkpoint")
    state = {"schema": "hc15-epoch-state-v1", "bindings": dict(bindings), "recipe": RECIPE,
             "epoch_next": epoch_next, "step": step, "accumulation_cursor": 0,
             "order_position": count // effective_batch * effective_batch,
             "last_epoch_order": last_order, "order_generator": order_generator.get_state(),
             "model": cpu_copy(model.state_dict()), "ema": cpu_copy(ema.state_dict()),
             "optimizer": cpu_copy(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
             "rng": capture_rng()}
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return sha256(path)


def load_boundary(path, expected_sha, bindings, *, count=TRAIN_N, effective_batch=EFFECTIVE_BATCH):
    import torch
    check_binding(path, {"sha256": expected_sha})
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(state.get("schema") == "hc15-epoch-state-v1", "Unknown checkpoint schema")
    require(state.get("bindings") == bindings and state.get("recipe") == RECIPE, "Checkpoint lineage changed")
    epoch, step = state["epoch_next"], state["step"]
    require(type(epoch) is int and 1 <= epoch <= EPOCHS and step == epoch * (count // effective_batch),
            "Checkpoint updates/epoch differ")
    require(state["accumulation_cursor"] == 0 and state["order_position"] == count // effective_batch * effective_batch,
            "Mid-epoch/partial-accumulation resume is unsupported")
    require(sorted(state["last_epoch_order"]) == list(range(count)), "Saved data order is invalid")
    require(state["scheduler"]["last_epoch"] == step, "Saved scheduler/update counters differ")
    return state


def restore_boundary(state, model, ema, optimizer, scheduler, order_generator):
    model.load_state_dict(state["model"], strict=True)
    ema.load_state_dict(state["ema"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    order_generator.set_state(state["order_generator"].cpu())
    restore_rng(state["rng"])  # Last, after all objects and their random initializations exist.
    return state["epoch_next"], state["step"]


def ema_update(ema, model, step):
    import torch
    decay = min(.998, (1 + step) / (10 + step))
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters(), strict=True):
            target.lerp_(source, 1 - decay)
        for target, source in zip(ema.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def build_offline_model(asset):
    from safetensors.torch import load_file
    from training.adapters.goodpjw_reader import KneeNet
    require(Path(asset["path"]).suffix == ".safetensors", "Only pinned generic safetensors initialization allowed")
    check_binding(asset["path"], asset)
    model = KneeNet(backbone=ARCH, pretrained=False, dim=384, depth=2, drop=.2, drop_path=.1)
    weights = load_file(asset["path"], device="cpu")
    # The pinned ImageNet checkpoint has a 1000-class classifier. Its removal is
    # exact; arbitrary missing/unexpected backbone keys must never be swallowed.
    require(weights["head.fc.weight"].shape == (1000, 768) and weights["head.fc.bias"].shape == (1000,),
            "Generic initialization classifier shape differs")
    weights.pop("head.fc.weight")
    weights.pop("head.fc.bias")
    model.enc.load_state_dict(weights, strict=True)
    return model


def configure_cuda(admission_capability=None):
    require(admission_capability is _EXECUTION_CAPABILITY,
            "CUDA initialization requires the admitted execute entrypoint")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one admitted local CUDA device required")
    require(torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported(),
            "Native BF16 is required; no FP16/FP32 fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # CUDA grid_sample backward has no deterministic implementation in the pinned
    # runtime. RNG/state continuity is tested; GPU bitwise resume is not promised.
    torch.use_deterministic_algorithms(False)
    random.seed(RECIPE["seed"])
    np.random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    torch.cuda.manual_seed_all(RECIPE["seed"])


def predict_final(model, dataset, uids, path, deadline, admission_capability=None):
    require(admission_capability is _EXECUTION_CAPABILITY,
            "GPU predictions require the admitted execute entrypoint")
    import torch
    path = Path(path)
    require(list(dataset.studies) == list(uids) and len(dataset) == len(uids), "Prediction UID alignment differs")
    require(not path.exists(), "Refusing to overwrite predictions")
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    model.eval()
    try:
        with temp.open("x", encoding="utf-8", newline="") as stream, torch.inference_mode():
            writer = csv.writer(stream)
            writer.writerow(["StudyInstanceUID", *LABELS])
            for index, uid in enumerate(uids):
                require(time.monotonic() < deadline, "Execution wall-clock budget exceeded")
                x, mask, pos, _ = dataset[index]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda(),
                                   pos.unsqueeze(0).cuda(), res=256, train=False)
                require(logits.shape == (1, 12) and bool(torch.isfinite(logits).all()), "Invalid final logits")
                probabilities = logits.float().sigmoid()
                require(bool(torch.isfinite(probabilities).all()), "Invalid final probabilities")
                writer.writerow([uid, *probabilities[0].cpu().tolist()])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return {"rows": len(uids), "target_order": LABELS, "ordered_uid_sha256": uid_sha(uids),
            "sha256": sha256(path), "output": "sigmoid_probabilities", "metrics": "deferred_to_fixed_evaluator"}


def execute(args):
    start = time.monotonic()
    admission, manifest, cohort, bindings, cache = execution_admission(args)
    # No numerical package or CUDA API is reached before the admission above.
    sys.path.insert(0, str(ROOT))  # Also supports invocation by absolute script path.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import torch
    from training.adapters.goodpjw_reader import StudyDataset, build_study_table
    configure_cuda(_EXECUTION_CAPABILITY)
    deadline = start + admission["max_wall_seconds"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "train" and not args.checkpoint:
        require(not any(output.iterdir()), "Fresh training requires an empty new output directory")
    rows, counts, files = cache
    table = build_study_table(rows, counts)
    loader = VerifiedSeriesLoader(files)
    val_ds = StudyDataset(cohort["val_uids"], table, targets=None, k=12, train=False, loader=loader)
    model = build_offline_model(manifest["initialization"]["asset"]).cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    order_generator = torch.Generator().manual_seed(RECIPE["seed"])
    if args.command == "serve":
        state = load_boundary(args.checkpoint, args.checkpoint_sha256, bindings)
        require(state["epoch_next"] == EPOCHS and state["step"] == TOTAL_UPDATES, "Only final EMA may serve")
        ema.load_state_dict(state["ema"], strict=True)
        step, resumed_step = state["step"], state["step"]
        final_checkpoint = {"path": str(Path(args.checkpoint).resolve()), "sha256": args.checkpoint_sha256}
        del state, model
        result = predict_final(ema, val_ds, cohort["val_uids"], output / "val_final.csv", deadline,
                               _EXECUTION_CAPABILITY)
    else:
        train_ds = StudyDataset(cohort["train_uids"], table,
                                targets=np.asarray(cohort["train_soft_float32"], dtype=np.float32),
                                k=8, train=True, loader=loader)
        encoder = list(model.enc.parameters())
        head = [parameter for name, parameter in model.named_parameters() if not name.startswith("enc.")]
        optimizer = torch.optim.AdamW([{"params": encoder, "lr": 1e-4}, {"params": head, "lr": 5e-4}],
                                      weight_decay=.02)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_multiplier)
        epoch_next, step = 0, 0
        final_checkpoint = None
        if args.checkpoint:
            state = load_boundary(args.checkpoint, args.checkpoint_sha256, bindings)
            epoch_next, step = restore_boundary(state, model, ema, optimizer, scheduler, order_generator)
            final_checkpoint = {"path": str(Path(args.checkpoint).resolve()), "sha256": args.checkpoint_sha256}
            del state
        resumed_step = step
        for epoch in range(epoch_next, EPOCHS):
            model.train()
            order, batches = epoch_order(TRAIN_N, order_generator)
            total_loss = 0.
            for indices in batches:
                require(time.monotonic() < deadline, "Execution wall-clock budget exceeded")
                optimizer.zero_grad(set_to_none=True)
                for index in indices:
                    x, mask, pos, target = train_ds[index]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda(),
                                       pos.unsqueeze(0).cuda(), res=256, train=True)
                    require(logits.shape == (1, 12), "Training output target shape differs")
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.float(), target.unsqueeze(0).cuda()) / EFFECTIVE_BATCH
                    require(bool(torch.isfinite(loss)), "Nonfinite loss; no skipped update or precision fallback")
                    loss.backward()
                    total_loss += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                step += 1
                ema_update(ema, model, step)
            require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()),
                    "Nonfinite final-epoch parameters")
            require(all(bool(torch.isfinite(parameter).all()) for parameter in ema.parameters()),
                    "Nonfinite final-epoch EMA parameters")
            checkpoint = output / f"state_epoch{epoch + 1:02d}.pt"
            checksum = save_boundary(checkpoint, model, ema, optimizer, scheduler, order_generator,
                                     epoch_next=epoch + 1, step=step, last_order=order, bindings=bindings)
            final_checkpoint = {"path": str(checkpoint.resolve()), "sha256": checksum}
            # No validation loader, validation labels or metric is accessed here.
            print(json.dumps({"epoch_completed": epoch + 1, "updates": step,
                              "mean_train_bce": total_loss / UPDATES_PER_EPOCH,
                              "checkpoint": checkpoint.name, "sha256": checksum}), flush=True)
        require(step == TOTAL_UPDATES, "Final successful update count differs")
        del model, optimizer, scheduler
        result = predict_final(ema, val_ds, cohort["val_uids"], output / "val_final.csv", deadline,
                               _EXECUTION_CAPABILITY)
    receipt = {"status": "RUNNER_COMPLETED_NOT_OUTER_VALIDATED", "bindings": bindings,
               "recipe": RECIPE, "predictions": result, "gold_access": "none",
               "final_checkpoint": final_checkpoint, "successful_updates": step,
               "updates_this_process": step - resumed_step, "skipped_updates": 0,
               "cuda_deterministic": False, "resume_claim": "epoch-boundary RNG/state continuity; GPU bitwise unproven",
               "seconds": time.monotonic() - start}
    with (output / "runner_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "train", "serve"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--cache-manifest")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            _, cohort = load_frozen_inputs(args.manifest, args.manifest_sha256)
            print(json.dumps({"status": "PREPARATION_ONLY_NOT_EXECUTION_READY",
                              "train": len(cohort["train_uids"]), "validation": len(cohort["val_uids"]),
                              "recipe_adaptation_requiring_review": RECIPE,
                              "gpu_imported": "torch" in sys.modules}, indent=2))
            return 0
        require(args.admission and Path(args.admission).is_file(),
                "Execution blocked: separate reviewed execution-admission JSON is absent")
        require(args.cache_manifest and args.cache_dir and args.output, "Execution paths are required")
        require(bool(args.checkpoint) == bool(args.checkpoint_sha256), "Checkpoint requires an explicit SHA256")
        require(args.command != "serve" or args.checkpoint, "Serving requires a pinned final checkpoint")
        execute(args)
        return 0
    except (ContractError, KeyError, FileNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"HC15 BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
