"""Explicit ConvNeXt spatial MIL family, v1.2.0 (2026-10-11).

Changelog: config-bound S42 F0/F1, explicit factories, fixed-final fitting,
epoch-boundary resume and ordered K24/K36/plane-flip readouts. No generated
source, guard delegation, downloads, metric selection or wall-time gates.
v1.0.1: admit the same fixed recipe for S43 folds 0-4; each fold still binds
its own roster, label weights, input hashes and recorded target environment.
v1.1.0: S42/S43 folds 0-4, with scientific control identity checked at every
data validation. Portable legacy registration is an explicit separate boundary.
v1.2.0: production member mode retains recipe/data/environment/evidence checks
without paired controls; run kind is explicit in output provenance and summary.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.arms import lifecycle as life
from training.src import raptor_common as common
from training.src.raptor_exp185 import finding_attention, spatial_tokens
from training.src.raptor_runtime import configure_runtime, runtime_snapshot

VERSION = "1.2.0"
UID = "StudyInstanceUID"
ARCH = "convnext_tiny.fb_in22k_ft_in1k"
TARGETS = list(common.LAB)
OUTPUTS = ["model_best.pt", "history.json", "provenance.json", "summary.json",
           "oof_best.csv", "oof_k36.csv", "oof_k36_flip_tta.csv", "output-inventory.json"]
WINDOWS = {"res": 224, "train_k": 12, "eval_k": 24, "extra_k": 36,
           "input": "center3", "crop": "mixed", "norm": "imagenet",
           "policy": "within-slot-valid", "augmentation": "AUG1", "jitter": .20}
SAMPLER = {"batch": 8, "eval_batch": 4, "shuffle": True, "drop_last": True,
           "generator": "separate-seeded-torch", "stream": "sha256(seed,epoch,uid,stream)-v1"}
UPDATES = {"epochs": 20, "fixed_epoch": 20, "accum": 1, "bb_lr": .0005,
           "head_lr": .001, "weight_decay": .02, "pct_start": .15, "clip_norm": 3.,
           "optimizer": "AdamW", "scheduler": "OneCycleLR", "loss": "clipped_positive_term",
           "amp_train": "fp16", "amp_eval": "fp16", "amp_init_scale": 1024.}
READOUT = {"selection": "fixed-final", "metric_access": "disabled",
           "tta": "coronal_axial_horizontal_mean_probability", "id_column": UID,
           "csv": "float32-pandas-float64"}
SOURCE_FILES = ["training/arms/mil.py", "training/arms/lifecycle.py",
                "training/arms/mil_registration.py", "tools/mil_register.py",
                "training/runners/mil_arms.py", "training/src/raptor_common.py",
                "training/src/raptor_exp185.py", "training/src/raptor_runtime.py",
                "config/workspace.toml"]


def need(condition, message):
    if not condition:
        raise life.ContractError(message)


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return life.parse_json(Path(path).read_bytes())


def write_new(path, value):
    life.create_new(path, life.canonical_bytes(value))


def uid_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # queues CUDA seeding, does not initialize CUDA


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FindingMIL(nn.Module):
    """Same construction order, RNG draws and state keys as the legacy model."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        features = backbone.num_features
        self.norm = nn.LayerNorm(features)
        self.att = nn.Sequential(nn.Linear(features, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, features))
        self.clsb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsW, std=.02)
        self.n = 12
        self.pool_policy = "finding_spatial_tokens_v1"

    def encode(self, x):
        return spatial_tokens(self, x)

    def head(self, encoded):
        return finding_attention(self, *encoded)[0]

    def forward(self, x):
        return self.head(self.encode(x))


def strict_load(model, state):
    expected = model.state_dict()
    need(set(state) == set(expected), "model state keys differ")
    for key, tensor in state.items():
        need(tensor.shape == expected[key].shape and tensor.dtype == expected[key].dtype
             and bool(torch.isfinite(tensor).all()), "model tensor shape, dtype or finiteness differs")
    model.load_state_dict(state, strict=True)


def build_model(config, root):
    """Always consume construction RNG; full initial state is an explicit input."""
    from safetensors.torch import load_file
    root = Path(root)
    model = FindingMIL(common.build_backbone(config["arch"], pretrained=False))
    need(sha(root / config["weights"]) == config["weights_sha256"], "offline weights differ")
    state = load_file(str(root / config["weights"]), device="cpu")
    for key in ("head.fc.weight", "head.fc.bias"):
        state.pop(key, None)
    strict_load(model.backbone, state)
    need(sha(root / config["initial_state"]) == config["initial_file_sha256"], "initial state file differs")
    initial = torch.load(root / config["initial_state"], map_location="cpu", weights_only=True)
    need(state_hash(initial) == config["initial_semantic_sha256"], "initial semantic identity differs")
    # The full saved state must really contain this offline backbone.
    need(all(torch.equal(initial["backbone." + k], v) for k, v in state.items()), "initial backbone differs")
    strict_load(model, initial)
    need(not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()), "unexpected BN")
    return model


def scale_windows(vol, centers):
    need(vol.shape == (64, 336, 336) and vol.dtype == np.uint8 and len(centers) in (12, 24, 36),
         "registered mixed windows required")
    out = np.empty((len(centers), 3, 224, 224), np.float32)
    for j, center in enumerate(centers):
        if center == -1:
            out[j] = 0
            continue
        center = max(1, min(int(center), 62))
        image = vol[center] if j % 2 == 0 else vol[center, 48:288, 48:288]
        plane = torch.from_numpy(image[None].astype(np.float32) / 255.)
        out[j] = torch.nn.functional.interpolate(plane[None], size=(224, 224), mode="bilinear",
                                                align_corners=False)[0].numpy()
    return out


def augment_windows(x, centers, rng, spec):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import affine
    need([tuple(s) for s in spec["slots"]] == common.SLOTS64 and len(x) == len(centers), "AUG1 layout differs")
    out = np.empty_like(x)
    height, width = x.shape[-2:]
    for j, center in enumerate(centers):
        angle, scale = rng.uniform(-10., 10.), rng.uniform(.9, 1.1)
        dx, dy = rng.uniform(-.05, .05) * width, rng.uniform(-.05, .05) * height
        gamma, flip = rng.uniform(.8, 1.25), rng.random() < .5
        if center == -1:
            out[j] = x[j]
            continue
        need(0 <= center < 64, "AUG1 center out of range")
        t = affine(torch.from_numpy(x[j]), angle=angle, translate=[dx, dy], scale=scale,
                   shear=[0., 0.], interpolation=InterpolationMode.BILINEAR, fill=0.)
        result = np.power(np.clip(t.numpy(), 0, 1), np.float32(gamma))
        if flip and 32 <= center < 64:
            result = result[..., ::-1]
        out[j] = result
    return out


class MILWindows(Dataset):
    """Explicit, spawn-safe dataset factory with a shared epoch counter."""

    def __init__(self, corpus_dir, ids, labels, k, train, seed):
        self.corpus_dir, self.ids, self.labels = Path(corpus_dir), list(ids), labels
        self.k, self.train, self.seed = k, train, seed
        self.corpus = None
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __getitem__(self, index):
        if self.corpus is None:
            self.corpus = common.Corpus(self.corpus_dir)
        uid, epoch = self.ids[index], int(self.epoch.item())
        rng = common.study_rng(self.seed, epoch, uid) if self.train else None
        centers = common.select_centers(self.corpus.mask(uid), 64, self.k, self.corpus.spec,
                                        "within-slot-valid", rng)
        x = scale_windows(self.corpus.volume(uid), centers)
        if self.train:
            intensity = common.study_rng(self.seed, epoch, uid, "intensity").random()
            x = np.clip(x * (1 + (intensity - .5) * .20), 0, 1)
            x = augment_windows(x, centers, rng, self.corpus.spec)
        mean = np.array([.485, .456, .406], np.float32).reshape(3, 1, 1)
        std = np.array([.229, .224, .225], np.float32).reshape(3, 1, 1)
        x = (x - mean) / std
        need(x.dtype == np.float32 and np.isfinite(x).all(), "invalid windows")
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid


def collate(rows):
    xs, ys, ids = zip(*rows)
    return torch.stack(xs), torch.stack(ys), list(ids)


def worker_init(_):
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def load_labels(path):
    import pandas as pd
    allowed = {UID, *TARGETS, *(t + "_soft" for t in TARGETS)}
    frame = pd.read_csv(path, usecols=lambda c: c in allowed)
    need(UID in frame and not frame[UID].isna().any() and not frame[UID].duplicated().any(), "label UID malformed")
    columns = [t if t in frame else t + "_soft" for t in TARGETS]
    need(set(columns) <= set(frame), "label targets missing")
    frame = frame.dropna(subset=columns)
    values = frame[columns].to_numpy(dtype=np.float32)
    need(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "invalid labels")
    semantic = hashlib.sha256(frame[[UID] + columns].to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    return dict(zip(frame[UID].astype(str), values)), semantic


def positive_weights(labels, ids):
    prevalence = np.clip(np.stack([labels[u] for u in ids]).mean(0), .03, .7)
    return np.clip((1 - prevalence) / prevalence, 1, 10).astype(np.float32)


def validate_membership(corpus_ids, ordered, rows, canonical, order_policy="base-canonical"):
    train, val = list(ordered["train"]), list(ordered["val"])
    split = {r[UID]: r["split"] for r in rows}
    need(len(split) == len(rows) and not set(split.values()) - {"train", "val"}, "groups malformed")
    positions = {u: i for i, u in enumerate(canonical)}
    need(len(positions) == len(canonical), "canonical order duplicated")
    need(len(set(train)) == len(train) and len(set(val)) == len(val) and not set(train) & set(val)
         and set(train) == {u for u, s in split.items() if s == "train"}
         and set(val) == {u for u, s in split.items() if s == "val"}
         and set(split) <= set(corpus_ids) and len(set(corpus_ids)) == len(corpus_ids)
         and set(split) <= set(positions), "fold membership differs")
    need(order_policy in ("base-canonical", "sorted-uid"), "unknown roster order")
    for ids in (train, val):
        if order_policy == "sorted-uid":
            need(ids == sorted(ids), "sorted roster order differs")
        else:
            sequence = [positions[u] for u in ids]
            need(all(b > a for a, b in zip(sequence, sequence[1:])), "canonical order differs")
    need(not {r["component_id"] for r in rows if r["split"] == "train"} &
         {r["component_id"] for r in rows if r["split"] == "val"}, "component crosses split")
    return train, val


def validate_recipe(spec):
    doc, config = spec.document, spec.document["config"]
    need(doc["family"] == "convnext-mil" and doc["seed"] in (42, 43)
         and doc["fold"] in range(5), "unsupported MIL member")
    for key, expected in (("windows", WINDOWS), ("sampler", SAMPLER), ("readout", READOUT),
                          ("targets", TARGETS), ("bn", {"policy": "none", "module_count": 0})):
        need(life.canonical_bytes(config[key]) == life.canonical_bytes(expected), key + " recipe differs")
    need(set(config["updates"]) == set(UPDATES) | {"loader_batches", "steps", "max_skips"}, "update fields differ")
    need(all(config["updates"][k] == v for k, v in UPDATES.items()), "update recipe differs")
    need(set(config["model"]) == {"arch", "kind", "weights", "weights_sha256", "initial_state",
                                   "initial_file_sha256", "initial_semantic_sha256"}, "model fields differ")
    need(config["model"]["arch"] == ARCH and config["model"]["kind"] == "finding_spatial_tokens_v1", "model differs")
    need(set(config["labels"]) == {"path", "name", "semantic_sha256", "pos_weight"}, "label fields differ")
    need(set(config["roots"]) == {"corpus", "manifest", "ordered", "groups", "canonical",
                                   "n_train", "n_val", "train_sha256", "val_sha256", "order_policy"}, "root fields differ")
    need(config["roots"]["order_policy"] == ("base-canonical" if doc["fold"] == 0 else "sorted-uid"),
         "registered fold order policy differs")
    need(doc["outputs"] == OUTPUTS and set(doc["inputs"]["sources"]) == set(SOURCE_FILES), "source/output inventory differs")
    # Paths used by this factory must all be pinned, not merely mentioned in prose.
    for group, paths in (("data", [config["roots"]["manifest"], config["labels"]["path"],
                                   config["model"]["weights"], config["model"]["initial_state"]]),
                         ("rosters", [config["roots"][k] for k in ("ordered", "groups", "canonical")])):
        need(set(doc["inputs"][group]) == set(paths), "factory input bindings differ")
    need(doc["inputs"]["data"][config["model"]["weights"]] == config["model"]["weights_sha256"]
         and doc["inputs"]["data"][config["model"]["initial_state"]] == config["model"]["initial_file_sha256"],
         "model input identity differs")
    return config


def validate_data(spec, root, corpus_dir):
    config = validate_recipe(spec)
    root, corpus_dir = Path(root), Path(corpus_dir)
    life.validate_inputs(spec, root)
    if spec.run_kind == "experiment":
        from training.arms.mil_registration import control_records, validate_control
        validate_control(spec, *control_records(spec, root))
    # The loaded source must match the staged source, even when launched elsewhere.
    source_root = Path(__file__).resolve().parents[2]
    for path, digest in spec.document["inputs"]["sources"].items():
        need(sha(source_root / path) == digest, "executing source differs from stage")
    manifest = read(corpus_dir / "manifest.json")
    need(sha(corpus_dir / "manifest.json") == spec.inventory[config["roots"]["manifest"]], "mounted corpus differs")
    for name in ("ids", "masks"):
        need(sha(corpus_dir / (name + ".npy")) == manifest[name + "_sha256"], "corpus index differs")
    corpus = common.Corpus(corpus_dir)
    need(corpus.depth == 64 and [tuple(s) for s in corpus.spec["slots"]] == common.SLOTS64, "corpus layout differs")
    roots = config["roots"]
    base = read(root / roots["canonical"])
    canonical = base["train"] + base["val"]
    with open(root / roots["groups"], newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train, val = validate_membership(corpus.ids.tolist(), read(root / roots["ordered"]), rows, canonical, roots["order_policy"])
    need((len(train), len(val)) == (roots["n_train"], roots["n_val"])
         and uid_hash(train) == roots["train_sha256"] and uid_hash(val) == roots["val_sha256"], "roster identity differs")
    labels, semantic = load_labels(root / config["labels"]["path"])
    need(semantic == config["labels"]["semantic_sha256"] and all(u in labels for u in train), "label identity/coverage differs")
    weights = positive_weights(labels, train)
    need(np.array_equal(weights, np.asarray(config["labels"]["pos_weight"], np.float32)), "positive weights differ")
    batches = len(train) // 8
    need(config["updates"]["loader_batches"] == batches and config["updates"]["steps"] == batches * 20
         and config["updates"]["max_skips"] == batches * 20 // 100, "update counts differ")
    return corpus, train, val, labels, weights


def verify_shards(corpus, ids=None):
    """Full run hashes every shard; engineering smoke hashes all accessed shards."""
    rows = None if ids is None else {corpus.row[u] for u in ids}
    checked = {}
    for shard in corpus.manifest["shards"]:
        if rows is not None and not any(shard["start"] <= row < shard["stop"] for row in rows):
            continue
        name = life.safe_member(shard["file"])
        path = corpus.dir / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == shard["bytes"], "shard size differs")
        need(sha(path) == shard["sha256"], "shard content differs")
        checked[name] = shard["sha256"]
    need(bool(checked), "no corpus shard verified")
    return {"scope": "all" if ids is None else "accessed", "shards": checked}


def flip_inputs(x, ids, corpus, k):
    output = x.clone()
    for row, uid in enumerate(ids):
        centers = common.select_centers(corpus.mask(uid), 64, k, corpus.spec, "within-slot-valid")
        start, selected = 0, [False] * k
        for plane, _, count in corpus.spec["slots"]:
            need(plane in ("Sagittal", "Coronal", "Axial") and count >= 3, "TTA layout differs")
            for index, center in enumerate(centers):
                if center != -1 and start + 1 <= center < start + count - 1:
                    selected[index] = plane in ("Coronal", "Axial")
            start += count
        need(start == 64, "TTA depth differs")
        output[row, selected] = torch.flip(output[row, selected], dims=[-1])
    return output


def autocast(device, amp):
    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" and amp == "fp16" else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, amp, corpus, *, tta=False):
    model.eval()
    predictions, ordered, batches = [], [], 0
    for x, _, ids in loader:
        with autocast(device, amp):
            logits = model(x.to(device))
        need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid evaluation logits")
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        if tta:
            flipped = flip_inputs(x, ids, corpus, loader.dataset.k)
            with autocast(device, amp):
                logits = model(flipped.to(device))
            need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid TTA logits")
            probabilities = (probabilities + torch.sigmoid(logits.float()).cpu().numpy()) * np.float32(.5)
        predictions.append(probabilities)
        ordered += ids
        batches += 1
    need(bool(predictions), "empty evaluation")
    return np.concatenate(predictions), ordered, batches


def optimizer_stack(model, steps, device, amp):
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": .0005}, {"params": head, "lr": .001}], weight_decay=.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[.0005, .001], total_steps=steps, pct_start=.15)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=amp == "fp16" and device == "cuda")
    return opt, sched, scaler


def train_epoch(model, loader, opt, sched, scaler, weights, device, amp, max_skips, counters):
    model.train()
    opt.zero_grad(set_to_none=True)
    total, batches = 0., 0
    lossf = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    for x, y, ids in loader:
        with autocast(device, amp):
            logits = model(x.to(device, non_blocking=True))
            need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid training logits")
            loss = lossf(logits.float(), y.to(device, non_blocking=True))
        need(bool(torch.isfinite(loss)), "nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=amp != "fp16")
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if amp == "fp16" and device == "cuda" and scaler.get_scale() < before:
            counters["skips"] += 1
            need(counters["skips"] <= max_skips, "scientific optimizer skip limit exceeded")
        else:
            sched.step()
        counters["updates"] += 1  # same attempted-update convention as legacy
        total += loss.item()
        batches += 1
    need(batches > 0, "empty training loader")
    return {"loss": total / batches, "batches": batches}


def save_epoch(path, *, epoch, model, opt, sched, scaler, generator, history, counters, identity, device):
    payload = {"schema": "mil-epoch/v1", "identity": identity, "epoch": epoch,
               "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
               "scaler": scaler.state_dict(), "generator": generator.get_state(), "history": history,
               "counters": counters, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    write_new(path.with_suffix(".receipt.json"), {"schema": "mil-epoch-receipt/v1", "sha256": sha(path),
                                                "identity": identity, "epoch": epoch})


def restore_epoch(path, *, model, opt, sched, scaler, generator, identity, device, loader=None):
    path = Path(path)
    receipt = read(path.with_suffix(".receipt.json"))
    need(receipt["sha256"] == sha(path) and receipt["identity"] == identity, "resume receipt differs")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    need(ck["schema"] == "mil-epoch/v1" and ck["identity"] == identity and ck["epoch"] == receipt["epoch"], "resume identity differs")
    need((ck["cuda_rng"] is not None) == (device == "cuda"), "resume device differs")
    strict_load(model, ck["model"])
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    scaler.load_state_dict(ck["scaler"])
    # A persistent iterator consumes a worker base_seed only on construction.
    # Recreate it before restoring the shared sampler generator; its prefetched
    # indices are discarded by the next iterator reset. Otherwise a restart
    # consumes an extra base_seed and silently changes the next epoch's shuffle.
    if loader is not None and loader.persistent_workers and loader._iterator is None:
        iter(loader)
    generator.set_state(ck["generator"])
    random.setstate(ck["python_rng"])
    np.random.set_state(ck["numpy_rng"])
    torch.set_rng_state(ck["torch_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck["epoch"] + 1, ck["history"], ck["counters"]


def observation(device):
    import cv2
    import pydicom
    import timm
    runtime = {"python": platform.python_version(), "driver": "none", "torch": str(torch.__version__),
               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": "cpu",
               "timm": timm.__version__, "numpy": np.__version__, "pydicom": pydicom.__version__, "cv2": cv2.__version__}
    if device == "cuda":
        runtime["gpu"] = torch.cuda.get_device_name(0)
        # No application timeout or projection gate. B owns liveness monitoring.
        runtime["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                                      "--format=csv,noheader"], text=True).strip().splitlines()[0]
    return {"hostname": socket.gethostname(), "runtime": runtime, "backend_policy": runtime_snapshot()}


def run(spec, stage, corpus_dir, destination, *, mode, device, workers=0, resume=None,
        note=None, evidence_root=None):
    """One explicit family execution. CUDA full fits require exact bound PASS.

    CPU smoke uses the real model and data, one training study/update and one
    validation study for each readout; it is explicitly a FP32 engineering run.
    Target smoke uses the registered batch of eight and the target FP16 policy.
    """
    need(mode in ("smoke", "full") and device in ("cpu", "cuda"), "execution mode differs")
    need(mode == "smoke" or device == "cuda", "full CPU training is unsupported")
    need(type(workers) is int and 0 <= workers <= 12, "worker count differs")
    life.validate_stage(spec, stage)
    corpus, train_ids, val_ids, labels, weights = validate_data(spec, stage, corpus_dir)
    config, doc = spec.document["config"], spec.document
    amp = "none" if device == "cpu" else "fp16"
    runtime = configure_runtime(amp, device, policy="strict")
    observed = observation(device)
    verification_hash = None
    if device == "cuda":
        life.check_target_observation(spec, observed)
    if mode == "full":
        need(note is not None and evidence_root is not None, "full fit requires evidence")
        verification_hash = life.require_execution_evidence(spec, stage, note, evidence_root, fresh_observation=observed)
    batch, epochs = 8, 20
    if mode == "smoke":
        batch, epochs = (1 if device == "cpu" else 8), 1
        train_ids, val_ids = train_ids[:batch], val_ids[:1]
    shard_evidence = verify_shards(corpus, train_ids + val_ids if mode == "smoke" else None)
    seed_all(doc["seed"])
    selected_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(12, np.float32) for u in val_ids}}
    dataset = MILWindows(corpus_dir, train_ids, selected_labels, 12, True, doc["seed"])
    generator = torch.Generator().manual_seed(doc["seed"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, worker_init_fn=worker_init, generator=generator,
                        collate_fn=collate, pin_memory=device == "cuda")
    steps = len(loader) * epochs
    need(steps > 0 and (mode == "smoke" or steps == config["updates"]["steps"]), "loader steps differ")
    model = build_model(config["model"], stage).to(device)
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, 224, 224, device=device))
        model.train()
    opt, sched, scaler = optimizer_stack(model, steps, device, amp)
    identity = {"spec_sha256": spec.sha256, "mode": mode, "device": device, "observation": observed,
                "runtime": runtime, "batch": batch, "steps": steps, "workers": workers}
    start_epoch, history, counters = 0, [], {"updates": 0, "skips": 0}
    if resume is not None:
        start_epoch, history, counters = restore_epoch(resume, model=model, opt=opt, sched=sched, scaler=scaler,
                                                       generator=generator, identity=identity, device=device, loader=loader)
        need(0 <= start_epoch <= epochs and counters["updates"] == start_epoch * len(loader), "resume epoch/update count differs")
    starting_updates, starting_skips = counters["updates"], counters["skips"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    out, work = destination / "outputs", destination / "epochs"
    out.mkdir()
    start_time = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs):
            dataset.set_epoch(epoch)
            then = time.perf_counter()
            record = train_epoch(model, loader, opt, sched, scaler, torch.from_numpy(weights), device, amp, steps // 100, counters)
            record.update(epoch=epoch, epoch_seconds=time.perf_counter() - then, **counters)
            history.append(record)
            save_epoch(work / f"epoch-{epoch:03d}.pt", epoch=epoch, model=model, opt=opt, sched=sched, scaler=scaler,
                       generator=generator, history=history, counters=counters, identity=identity, device=device)
    finally:
        # Persistent workers must stop before readouts (Windows mmap/spawn).
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    need(counters["updates"] == steps and counters["skips"] <= steps // 100, "fixed-final update count differs")
    provenance = {"tool_version": VERSION, "spec_sha256": spec.sha256, "verification_sha256": verification_hash,
                  "run_kind": spec.run_kind,
                  "identity": identity, "shard_verification": shard_evidence, "checkpoint_selection": "fixed-final",
                  "gold_access": "disabled", "metric_access": "disabled", "bn_module_count": 0,
                  "train_ids_ordered_sha256": uid_hash(train_ids), "eval_ids_ordered_sha256": uid_hash(val_ids)}
    checkpoint = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "arch": ARCH, "res": 224, "lab": TARGETS, "contract": corpus.contract, "k_eval": 24,
                  "norm": "imagenet", "window_policy": "within-slot-valid", "epoch": epochs - 1,
                  "tag": doc["arm"], "amp_train": amp, "amp_eval": amp, "fold": doc["fold"],
                  "label_set": config["labels"]["name"], "provenance": provenance, "input_policy": "center3",
                  "init": "timm-in22k", "init_weights_sha256": config["model"]["weights_sha256"],
                  "flip_tta": READOUT["tta"], "model_kind": "finding_spatial_tokens_v1"}
    with (out / "model_best.pt").open("xb") as stream:
        torch.save(checkpoint, stream)
    strict_load(model, torch.load(out / "model_best.pt", map_location="cpu", weights_only=False)["model"])
    validation_batches = 0
    for k, tta, filename in ((24, False, "oof_best.csv"), (36, False, "oof_k36.csv"), (36, True, "oof_k36_flip_tta.csv")):
        ds = MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"])
        evaluation = DataLoader(ds, batch_size=4, shuffle=False, num_workers=min(workers, 2),
                                collate_fn=collate, worker_init_fn=worker_init)
        probabilities, ids, batches = evaluate(model, evaluation, device, amp, corpus, tta=tta)
        need(ids == val_ids, "readout order differs")
        life.create_new(out / filename, life.prediction_csv_bytes(ids, probabilities, TARGETS, id_column=UID))
        validation_batches += batches
    # Smoke evidence describes this invocation, never work inherited on resume.
    results = {"real_data": True, "train_batches": sum(r["batches"] for r in history[start_epoch:]),
               "validation_batches": validation_batches, "bn_batches": 0, "bn_module_count": 0,
               "optimizer_steps": counters["updates"] - starting_updates - counters["skips"] + starting_skips,
               "checkpoint_reloaded": True, "finite_outputs": True}
    summary = {"tool_version": VERSION, "status": "ENGINEERING_SMOKE" if mode == "smoke" else "FIXED_FINAL_COMPLETE",
               "scientific_status": "UNASSESSED", "run_kind": spec.run_kind, "results": results, "observation": observed,
               "elapsed_seconds": time.perf_counter() - start_time, "train": len(train_ids), "val": len(val_ids),
               "epochs": epochs, "attempted_updates": counters["updates"], "skipped_updates": counters["skips"],
               "spec_sha256": spec.sha256, "cpu_cuda_initialized": torch.cuda.is_initialized() if device == "cpu" else None}
    need(device != "cpu" or not summary["cpu_cuda_initialized"], "CPU execution initialized CUDA")
    for name, value in (("history.json", history), ("provenance.json", provenance), ("summary.json", summary)):
        write_new(out / name, value)
    write_new(out / "output-inventory.json", {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                                              for name in OUTPUTS if name != "output-inventory.json"})
    return summary


def record_smoke(spec, destination, summary, command):
    """Executed runner output -> immutable evidence and HOLD note, never PASS."""
    destination = Path(destination)
    out = destination / "outputs"
    observed, results = summary["observation"], summary["results"]
    kind = "cpu" if summary["cpu_cuda_initialized"] is not None else "target"
    environment_hash = life.content_sha256(observed) if kind == "cpu" else spec.bindings["environment"]
    smoke = {"hostname": observed["hostname"], "device": "cpu" if kind == "cpu" else "cuda",
             "environment_sha256": environment_hash, "bindings_sha256": life.content_sha256(spec.bindings), "results": results}
    artifacts = {role: {"path": "outputs/" + name, "sha256": sha(out / name)} for role, name in
                 (("checkpoint", "model_best.pt"), ("readout", "oof_best.csv"), ("summary", "summary.json"),
                  ("k36", "oof_k36.csv"), ("tta", "oof_k36_flip_tta.csv"), ("provenance", "provenance.json"))}
    check = {"schema": life.EVIDENCE_SCHEMA, "command": command, "exit_code": 0, "artifacts": artifacts, "smoke": None}
    check_hash = life.content_sha256(check)
    write_new(destination / "evidence" / (check_hash + ".json"), check)
    log = {**check, "smoke": {**smoke, "observation": observed}}
    log_hash = life.content_sha256(log)
    write_new(destination / "evidence" / (log_hash + ".json"), log)
    note = {"schema": life.VERIFICATION_SCHEMA, "scope": spec.scope,
            "reviewer": {"name": "executed-mil-smoke-producer", "host": "Codex"}, "verdict": "HOLD",
            "blockers": ["Independent review and both CPU/target evidence required before full training"],
            "notes": ["Engineering smoke; no scientific acceptance or resource admission"], "bindings": spec.bindings,
            "checks": [{"name": "executed-family-smoke", "command": command, "exit_code": 0, "log_sha256": check_hash}],
            "smokes": {"cpu": None, "target": None}}
    note["smokes"][kind] = {**smoke, "command": command, "exit_code": 0, "log_sha256": log_hash}
    life.verify_evidence(spec, note, destination)
    write_new(destination / "verification.json", note)
    return note
