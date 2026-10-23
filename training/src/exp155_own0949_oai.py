"""EXP155 mixed RSNA/OAI trainer. Imports EXP143 v2 geometry, model and optimizer.

Importing this module is CPU passive. Only explicit train/probe_mode enter CUDA.
Changelog 1.1.0 (2026-10-12): elapsed/projection values are telemetry; no self-imposed run deadline.
"""
from __future__ import annotations

VERSION = "1.1.0"

import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "training/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import exp143_v2 as base

ARCH, CONTRACT, TRAIN_RES = base.ARCH, base.CONTRACT, base.TRAIN_RES
StudyDataset, Corpus = base.StudyDataset, base.Corpus
build_model, load_imagenet = base.build_model, base.load_imagenet
amp_context = base.amp_context
worker_init = base.worker_init
install_portable_gold_blocker = base.install_portable_gold_blocker
make_optimization = base.make_optimization
accumulation_group = base.accumulation_group
average_registered_epochs = base.average_registered_epochs
elapsed_before_resume = base.elapsed_before_resume
_cpu_seed = base._cpu_seed


def masked_study_bce(logits, labels, masks, pos_weight):
    """Mean observed-target loss per study, then mean across the batch.

    Thus each study has unit weight regardless of its count of observed targets.
    RSNA with all twelve targets exactly matches mean BCEWithLogitsLoss.
    """
    import torch
    import torch.nn.functional as F

    if logits.ndim != 2 or logits.shape != labels.shape or logits.shape != masks.shape:
        raise ValueError("logits, labels and masks must share [study,target] shape")
    if logits.shape[1] != len(pos_weight):
        raise ValueError("pos_weight target count mismatch")
    if not torch.isfinite(labels).all() or not torch.isfinite(masks).all():
        raise ValueError("nonfinite labels or masks")
    if ((masks != 0) & (masks != 1)).any() or (masks.sum(dim=1) == 0).any():
        raise ValueError("every study must have at least one binary observed-target mask")
    safe_labels = torch.where(masks.bool(), labels, torch.zeros_like(labels))
    if ((safe_labels < 0) | (safe_labels > 1)).any():
        raise ValueError("observed labels must be in [0,1]")
    weighted = F.binary_cross_entropy_with_logits(
        logits, safe_labels, pos_weight=torch.as_tensor(pos_weight, device=logits.device,
                                                        dtype=logits.dtype), reduction="none")
    return ((weighted * masks).sum(dim=1) / masks.sum(dim=1)).mean()


class MixedStudyDataset:
    """Route each registered UID to exactly one native EXP143 corpus."""
    def __init__(self, rsna_corpus, oai_corpus, ids, labels, masks,
                 rsna_ids, oai_ids, seed, augment=True):
        self.ids = list(ids)
        self.labels, self.masks = labels, masks
        rsna_ids, oai_ids = set(rsna_ids), set(oai_ids)
        if rsna_ids & oai_ids or set(self.ids) != rsna_ids | oai_ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("RSNA/OAI registration is not a disjoint partition")
        if set(labels) != set(self.ids) or set(masks) != set(self.ids):
            raise ValueError("registered label and mask coverage differs from train IDs")
        self.rsna_ids, self.oai_ids = rsna_ids, oai_ids
        self.rsna = StudyDataset(rsna_corpus, [u for u in self.ids if u in rsna_ids], labels, seed, augment)
        self.oai = StudyDataset(oai_corpus, [u for u in self.ids if u in oai_ids], labels, seed, augment)
        self.rsna_index = {u: i for i, u in enumerate(self.rsna.ids)}
        self.oai_index = {u: i for i, u in enumerate(self.oai.ids)}
        oai_columns = {5, 6, 8}  # Lateral OA, PF OA, Synovitis in EXP143 order.
        for uid in self.ids:
            target, mask = np.asarray(labels[uid]), np.asarray(masks[uid])
            if target.shape != (12,) or mask.shape != (12,) or not np.isfinite(target).all():
                raise ValueError("label/mask must have 12 finite targets")
            if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all() or not mask.any():
                raise ValueError("mask must contain at least one observed target")
            if uid in rsna_ids and not mask.all():
                raise ValueError("RSNA studies require all 12 targets")
            if uid in oai_ids and any(mask[i] for i in range(12) if i not in oai_columns):
                raise ValueError("OAI study has an observed target outside the three registered labels")
            if ((target < 0) | (target > 1))[mask.astype(bool)].any():
                raise ValueError("observed training label outside [0,1]")

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch):
        self.rsna.set_epoch(epoch)
        self.oai.set_epoch(epoch)

    def __getitem__(self, index):
        import torch
        uid = self.ids[index]
        if uid in self.rsna_ids:
            image, label, actual = self.rsna[self.rsna_index[uid]]
        else:
            image, label, actual = self.oai[self.oai_index[uid]]
        if actual != uid:
            raise ValueError("corpus UID routing mismatch")
        return image, label, torch.tensor(self.masks[uid], dtype=torch.float32), uid


def collate(batch):
    import torch
    return (torch.stack([row[0] for row in batch]),
            torch.stack([row[1] for row in batch]),
            torch.stack([row[2] for row in batch]),
            [row[3] for row in batch])


def unpack_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
        raise TypeError("expected (images, labels, masks, UID-list)")
    x, y, mask, ids = batch
    if x.ndim != 5 or y.ndim != 2 or y.shape != mask.shape or x.shape[0] != y.shape[0] or len(ids) != len(x):
        raise ValueError("invalid mixed study batch shape")
    return x, y, mask, ids


def mixed_dataset(args, p, seed, augment=True):
    return MixedStudyDataset(args.corpus, args.oai_corpus, p["train_ids"],
        p["train_labels"], p["train_masks"], p["rsna_train_ids"],
        p["oai_train_ids"], seed, augment)


def tiny_validation():
    """CPU fake geometry and real autograd checks for RSNA/OAI mask semantics."""
    import torch
    import torch.nn.functional as F

    result = base.tiny_validation()
    logits = torch.zeros(2, 12, requires_grad=True)
    labels = torch.zeros(2, 12)
    labels[0, 6] = 1.
    labels[1, [5, 8]] = torch.tensor([1., 1.])
    mask = torch.zeros_like(labels)
    mask[0, 6] = 1.
    mask[1, [5, 8]] = 1.
    weights = torch.ones(12)
    masked = masked_study_bce(logits, labels, mask, weights)
    expected = (F.binary_cross_entropy_with_logits(logits[0, 6], labels[0, 6])
                + F.binary_cross_entropy_with_logits(logits[1, [5, 8]], labels[1, [5, 8]])) / 2
    torch.testing.assert_close(masked, expected)
    gradient = torch.autograd.grad(masked, logits)[0]
    if not (gradient[mask == 0] == 0).all() or not (gradient[mask == 1] != 0).all():
        raise AssertionError("masked loss gradient leaks into unknown targets")
    rsna_logits = torch.zeros(2, 12, requires_grad=True)
    rsna_labels = torch.full((2, 12), .25)
    torch.testing.assert_close(masked_study_bce(rsna_logits, rsna_labels,
        torch.ones_like(rsna_labels), weights),
        F.binary_cross_entropy_with_logits(rsna_logits, rsna_labels, pos_weight=weights))
    return dict(result, masked_loss_verified=True, oai_unknown_gradient_zero=True,
                rsna_bce_equivalent=True)


def scratch_directory(path=None):
    """Probe outputs stay under a dedicated scratch root."""
    root = (ROOT / ".local/exp155-probes").resolve()
    out = Path(path).resolve() if path else root / str(time.time_ns())
    try:
        relative = out.relative_to(root)
    except ValueError:
        raise ValueError("probe output must be under .local/exp155-probes/") from None
    if not relative.parts:
        raise ValueError("probe requires a new scratch subdirectory")
    return out


def runtime_details(device):
    """Run-local reproducibility receipt; called only after explicit CUDA setup."""
    import importlib.metadata as metadata
    import torch

    versions = {}
    for package in ("torch", "timm", "numpy", "pydicom", "pylibjpeg", "opencv-python"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    gpu = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    return dict(python=sys.version, packages=versions, cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), nvidia_driver=driver,
                gpu=dict(name=gpu.name, total_memory_bytes=gpu.total_memory,
                         compute_capability=list(torch.cuda.get_device_capability(device))),
                precision="bf16", cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
def train(args):
    """Registered one-GPU 4070 recipe; only explicit train initializes CUDA."""
    invocation_start = time.monotonic()
    from tools import exp155_own0949_oai as guard
    from tools.exp113_radr50_local import install_gold_blocker
    from tools.exp112_full_swa import recompute_bn, state_dict_sha256
    p = guard.load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
    guard.require_validation(args.protocol)
    install_gold_blocker(ROOT)
    install_portable_gold_blocker()
    # No labels file is opened here: only the registered non-Gold training subset.
    if args.device != "cuda":
        raise ValueError("production training requires CUDA; use validate for CPU fakes")
    if args.grad_checkpointing is not False:
        raise ValueError("EXP155 registers gradient checkpointing off only")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 1 or rank != 0 or local_rank != 0:
        raise ValueError("EXP155 registers one local GPU only")
    settings = guard.execution_settings(p, world, args.precision, args.seed, args.grad_checkpointing)
    recipe = p["recipe"]
    if recipe["epochs"] != 16 or recipe["swa_1based"] != [14, 11, 12] or settings["batch"] != 1 or settings["accumulation"] != 16:
        raise ValueError("EXP155 fixed training recipe differs")
    if not 0 <= args.workers <= 12:
        raise ValueError("workers must be in 0..12")
    out = guard.registered_run(p, settings["lane"])
    # Only rank0 creates/refuses the directory; a late-starting rank must not
    # mistake the directory just created by rank0 for a previous experiment.
    if rank == 0 and not args.resume and out.exists():
        raise FileExistsError("training output exists; use explicit --resume")
    if (out / "summary.json").exists():
        raise FileExistsError("complete experiment is immutable")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=args.resume)
    torch.cuda.set_device(local_rank)
    if world == 2:
        dist.init_process_group("nccl")
        dist.barrier()
    device = f"cuda:{local_rank}"
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("native bf16 unsupported; T4 requires fp16")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    _cpu_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_model(TRAIN_RES)
    load_imagenet(model, args.init_weights)
    initial_hash = state_dict_sha256(model.state_dict())
    model.backbone.set_grad_checkpointing(settings["grad_checkpointing"])
    model.to(device)
    raw_model = model
    if world == 2:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    ids, labels = p["train_ids"], p["train_labels"]
    ds = mixed_dataset(args, p, args.seed)
    sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(ds, batch_size=settings["batch"], sampler=sampler, drop_last=True,
                        num_workers=args.workers, pin_memory=True, collate_fn=collate,
                        persistent_workers=args.workers > 0, worker_init_fn=worker_init)
    accum = settings["accumulation"]
    updates_per_epoch = math.ceil(len(loader) / accum)
    total_steps = recipe["epochs"] * updates_per_epoch
    if total_steps != settings["total_steps"]:
        raise ValueError("loader length differs from registered update schedule")
    optimizer, scheduler, scaler = make_optimization(raw_model, recipe, total_steps, args.precision)
    loss_weights = torch.tensor(p["pos_weight"], device=device)
    start_epoch, updates, skips, previous_seconds = 0, 0, 0, 0.
    history = []
    run_identity = dict(protocol_sha256=guard.sha256(args.protocol), initial_state_sha256=initial_hash,
                        seed=args.seed, precision=args.precision, world_size=world,
                        contract=CONTRACT, train_n=len(ids), rsna_train_n=len(p["rsna_train_ids"]),
                        oai_train_n=len(p["oai_train_ids"]), torch=torch.__version__,
                        run_dir=p["run_dirs"][settings["lane"]], execution=settings)
    if args.resume:
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != run_identity:
            raise ValueError("resume environment/registration/recipe changed")
        raw_model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, updates, skips = saved["epoch"], saved["updates"], saved["skips"]
        previous_seconds, history = elapsed_before_resume(out, saved["seconds"]), saved["history"]
    elif rank == 0:
        guard.write_new_json(out / "provenance.json", dict(run_identity, oai=True,
            validation="none", gold_access="blocked", rsna_train_n=len(p["rsna_train_ids"]), oai_train_n=len(p["oai_train_ids"]), checkpoint_selection="fixed-SWA",
            swa_epochs_1based=recipe["swa_1based"], runtime=runtime_details(device), **settings))
        # Keep the initial restart point before any training update.
        guard.save_torch_atomic(out / "resume.pt", dict(
            model={k: v.detach().cpu() for k, v in raw_model.state_dict().items()}, epoch=0,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            updates=0, skips=0, seconds=time.monotonic() - invocation_start,
            history=[], identity=run_identity))
    start = invocation_start
    rate_start = time.monotonic()
    limit = settings["max_hours"] * 3600
    reserve = settings["reserve_seconds"]
    wall = lambda: previous_seconds + time.monotonic() - start
    this_updates = 0
    projected_once = False
    def record_timing():
        nonlocal projected_once
        elapsed = wall()
        if world == 2:
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor.item())
        if this_updates >= 100 and not projected_once:
            projected_once = True
            seconds_per = (time.monotonic() - rate_start) / 100
            if world == 2:
                rate_tensor = torch.tensor(seconds_per, dtype=torch.float64, device=device)
                dist.all_reduce(rate_tensor, op=dist.ReduceOp.MAX)
                seconds_per = float(rate_tensor.item())
            projected = elapsed + (total_steps - updates - skips) * seconds_per + reserve
            if rank == 0:
                guard.write_new_json(out / f"timing-epoch{start_epoch:02d}-{time.time_ns()}.json",
                    dict(seconds_per_update=seconds_per, projected_seconds=projected, limit_seconds=limit))
    try:
        record_timing()
        for epoch in range(start_epoch, recipe["epochs"]):
            # Epoch-local seeds make epoch-boundary resume independent of earlier RNG consumption.
            _cpu_seed(args.seed * 1000 + epoch * world + rank)
            torch.cuda.manual_seed_all(args.seed * 1000 + epoch * world + rank)
            ds.set_epoch(epoch)
            sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum, count = 0., 0
            ep_start = time.monotonic()
            for bi, batch in enumerate(loader):
                x, y, mask, _ = unpack_batch(batch)
                final, group_size = accumulation_group(bi, len(loader), accum)
                sync = contextlib.nullcontext() if final or world == 1 else model.no_sync()
                with sync:
                    with amp_context(device, args.precision):
                        logits = model(x.to(device, non_blocking=True)).float()
                        loss = masked_study_bce(logits, y.to(device, non_blocking=True), mask.to(device, non_blocking=True), loss_weights)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite loss")
                        scaled_loss = loss / group_size
                    scaler.scale(scaled_loss).backward()
                loss_sum += float(loss.detach()); count += 1
                if final:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), recipe["clip"], error_if_nonfinite=args.precision != "fp16")
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = scaler.get_scale() < old_scale
                    if skipped:
                        skips += 1
                    else:
                        scheduler.step(); updates += 1
                    if skips > total_steps // 100:
                        raise FloatingPointError("FP16 skipped-update gate exceeded")
                    optimizer.zero_grad(set_to_none=True)
                    this_updates += 1
                    record_timing()
            if rank == 0:
                epoch1 = epoch + 1
                state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
                guard.save_torch_new(out / f"epoch_{epoch1:02d}.pt", dict(model=state, epoch=epoch1))
                history.append(dict(epoch=epoch1, loss=loss_sum / count,
                                    seconds=time.monotonic() - ep_start, updates=updates, skips=skips))
                guard.save_torch_atomic(out / "resume.pt", dict(model=state, epoch=epoch1,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                    updates=updates, skips=skips, seconds=wall(), history=history, identity=run_identity))
                print(json.dumps(history[-1]), flush=True)
            if world == 2:
                dist.barrier()
        # Rank0 only: no distributed BN pass, no padded rows, no DDP collectives in inference.
        if rank == 0:
            raw_model.load_state_dict(average_registered_epochs(out, recipe), strict=True)
            raw_model.backbone.set_grad_checkpointing(False)
            bn_ds = mixed_dataset(args, p, args.seed, augment=False)
            bn_ds.set_epoch(recipe["epochs"])
            bn_loader = DataLoader(bn_ds, batch_size=recipe["bn_batch"], shuffle=False, drop_last=False,
                                   num_workers=args.workers, collate_fn=collate, pin_memory=True,
                                   worker_init_fn=worker_init)
            class BNLoader:
                def __iter__(self):
                    for b in bn_loader:
                        # Rank0-only BN: NEVER use the collective training timing observation here.
                        x, y, _mask, uids = unpack_batch(b)
                        yield x, y, uids
            bn = recompute_bn(raw_model, BNLoader(), device, amp_context(device, args.precision))
            state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
            payload = dict(model=state, arch=ARCH, res=384, inference_res=320, lab=guard.TARGETS,
                contract=CONTRACT, k_eval=94, norm="imagenet", window_policy="legacy",
                epoch=recipe["epochs"] - 1, tag=out.name, amp_train=args.precision, amp_eval="fp16", fold=-1,
                label_set=recipe["labels"], provenance=run_identity, seed=args.seed,
                checkpoint_kind="fixed-swa-bn", swa_epochs_1based=recipe["swa_1based"], oai=True)
            guard.save_torch_new(out / "model_best.pt", payload)
            guard.write_new_json(out / "summary.json", dict(status="COMPLETE_TRAINING_NO_EVALUATION",
                identity=run_identity, epochs=recipe["epochs"], updates=updates, skips=skips, history=history,
                seconds=wall(), bn=bn, model_sha256=guard.sha256(out / "model_best.pt"),
                swa_member_sha256={str(e): guard.sha256(out / f"epoch_{e:02d}.pt") for e in recipe["swa_1based"]}))
        if world == 2:
            dist.barrier()
    except BaseException as exc:
        if rank == 0:
            # Durable failure marker is distinct from COMPLETE and has no metric.
            guard.write_new_json(out / f"stop-{time.time_ns()}.json", dict(
                status="INCOMPLETE_NO_VALIDATION_SUCCESS", error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=wall(), updates=updates, skips=skips,
                resume="resume.pt contains the last completed epoch, or initial state"))
        raise
    finally:
        if world == 2:
            dist.destroy_process_group()



def probe(args):
    """One isolated CUDA process; no fit/checkpoint/validation writes."""
    import subprocess
    from tools import exp155_own0949_oai as guard
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("probe measures one local GPU; two-T4 fit has its own admission gate")
    p = guard.load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
    guard.require_validation(args.protocol)
    settings = guard.execution_settings(p, 1, args.precision, p["recipe"]["seed_local"], False)
    if not 1 <= args.steps <= settings["updates_per_epoch"] - 3 or not 0 <= args.workers <= 12:
        raise ValueError("probe steps must fit one full epoch after two warmups; workers 0..12")
    out = scratch_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ("off",):
        command = [sys.executable, str(ROOT / "tools/exp155_own0949_oai.py"), "_probe-mode",
                   "--protocol", str(Path(args.protocol).resolve()),
                   "--corpus", str(Path(args.corpus).resolve()),
                   "--oai-corpus", str(Path(args.oai_corpus).resolve()),
                   "--init-weights", str(Path(args.init_weights).resolve()),
                   "--steps", str(args.steps), "--workers", str(args.workers),
                   "--precision", args.precision, "--out", str(out), "--mode", mode]
        with (out / (mode + ".log")).open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result_file = out / (mode + ".json")
        if completed.returncode != 0 or not result_file.is_file():
            raise RuntimeError(f"{mode} probe failed unexpectedly; inspect {out / (mode + '.log')}")
        result = guard.read_json(result_file)
        if result.get("protocol_sha256") != guard.sha256(args.protocol):
            raise ValueError("probe child registration differs")
        results.append(result)
    report = dict(status="SCRATCH_PROBE_ONLY_NO_FIT", protocol_sha256=guard.sha256(args.protocol),
                  results=results, steps_per_mode=args.steps, warmup_updates_per_mode=2, grad_checkpointing=False,
                  caution="Short sample, not full-run/BN timing; full-fit 100-update gate still applies")
    guard.write_new_json(out / "report.json", report)
    print(json.dumps(dict(report, scratch_directory=str(out)), indent=2), flush=True)
    if not all(r.get("admissible") is True and r.get("mode") == "off" for r in results):
        raise RuntimeError("EXP155 probe inadmissible; queued fit must not start")


def probe_mode(args):
    """Internal subprocess entry; called only by explicit probe. Never saves weights."""
    invocation_start = time.monotonic()
    from tools import exp155_own0949_oai as guard
    from tools.exp113_radr50_local import install_gold_blocker
    from tools.exp112_full_swa import state_dict_sha256
    out = scratch_directory(args.out)
    if args.mode != "off" or not out.is_dir() or (out / (args.mode + ".json")).exists():
        raise ValueError("internal probe requires an existing new scratch directory")
    p = guard.load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
    guard.require_validation(args.protocol)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("single-GPU probe only")
    seed = p["recipe"]["seed_local"]
    settings = guard.execution_settings(p, 1, args.precision, seed, False)
    if not 1 <= args.steps <= settings["updates_per_epoch"] - 3 or not 0 <= args.workers <= 12:
        raise ValueError("invalid probe steps/workers")
    install_gold_blocker(ROOT)
    install_portable_gold_blocker()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, DistributedSampler
    torch.cuda.set_device(0)
    device = "cuda:0"
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("native bf16 unsupported")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    result = dict(mode=args.mode, grad_checkpointing=settings["grad_checkpointing"],
                  protocol_sha256=guard.sha256(args.protocol), precision=args.precision,
                  execution=settings, device_name=torch.cuda.get_device_name(0),
                  device_total_bytes=total, free_bytes_before=free, steps_requested=args.steps,
                  torch=torch.__version__, warmup_updates=2, skipped_updates=0,
                  timed_updates=0, status="INCOMPLETE", admissible=False)
    try:
        _cpu_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_model(TRAIN_RES)
        load_imagenet(model, args.init_weights)
        result["initial_state_sha256"] = state_dict_sha256(model.state_dict())
        model.backbone.set_grad_checkpointing(settings["grad_checkpointing"])
        model.to(device).train()
        ds = mixed_dataset(args, p, seed)
        ds.set_epoch(0)
        sampler = DistributedSampler(ds, 1, 0, shuffle=True, seed=seed, drop_last=True)
        sampler.set_epoch(0)
        loader = DataLoader(ds, batch_size=settings["batch"], sampler=sampler, drop_last=True,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate,
                            persistent_workers=args.workers > 0, worker_init_fn=worker_init)
        optimizer, scheduler, scaler = make_optimization(model, p["recipe"], settings["total_steps"], args.precision)
        loss_weights = torch.tensor(p["pos_weight"], device=device)
        _cpu_seed(seed * 1000)
        torch.cuda.manual_seed_all(seed * 1000)
        iterator = iter(loader)
        result["setup_seconds"] = time.monotonic() - invocation_start
        durations = []
        for update in range(2 + args.steps):
            torch.cuda.synchronize()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(settings["accumulation"]):
                x, y, mask, _uids = unpack_batch(next(iterator))
                with amp_context(device, args.precision):
                    logits = model(x.to(device, non_blocking=True)).float()
                    loss = masked_study_bce(logits, y.to(device, non_blocking=True), mask.to(device, non_blocking=True), loss_weights)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite probe loss")
                    scaled_loss = loss / settings["accumulation"]
                scaler.scale(scaled_loss).backward()
                # Same per-microbatch CPU sync as the full fit's loss accumulator.
                float(loss.detach())
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), p["recipe"]["clip"],
                                     error_if_nonfinite=args.precision != "fp16")
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:
                result["skipped_updates"] += 1
            else:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if update >= 2:
                durations.append(time.monotonic() - started)
                result["timed_updates"] = len(durations)
        seconds_per = sum(durations) / len(durations)
        estimate = guard.projection(settings, seconds_per, result["setup_seconds"])
        result.update(status="MEASURED_SCRATCH_ONLY", seconds_per_update=seconds_per,
                      update_seconds=durations, **estimate)
        result["admissible"] = result["skipped_updates"] == 0
    except torch.cuda.OutOfMemoryError:
        result.update(status="CUDA_OOM_SCRATCH_ONLY", admissible=False)
    finally:
        # A fresh child for the next mode guarantees OOM/allocator/BN isolation.
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      elapsed_seconds=time.monotonic() - invocation_start)
        guard.write_new_json(out / (args.mode + ".json"), result)
    print(json.dumps(result, indent=2), flush=True)

