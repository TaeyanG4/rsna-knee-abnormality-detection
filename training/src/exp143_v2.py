"""EXP143 v2 data/model/training and separate scratch probe; CPU-passive imports.

The Raptor head/SWA are imported from existing implementations. New geometry and
inference mirror are unchanged from v1. Recipe and run paths come from v2 protocol.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "training/src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Preserve the byte-bound v1 implementations of geometry/model/normalization.
# v2 owns only registration-driven optimization, scheduling, serving guards and probes.
from exp143_own0949 import (
    ARCH, CONTRACT, SPEC, TRAIN_RES, EVAL_RES, IMG, DEPTH, SLOT_SIZES, K_TRAIN, K_EVAL,
    Corpus, StudyDataset, unpack_batch, mirror_triplets, windows, collate, worker_init,
    install_portable_gold_blocker, build_model, load_imagenet, amp_context, infer_volume,
    tiny_validation, _cpu_seed,
)

def make_optimization(model, recipe, total_steps, precision):
    """One optimizer construction for probe and full fit; parameters are registered."""
    import torch
    rates = [recipe["bb_lr"], recipe["head_lr"]]
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=rates[0]),
        dict(params=[v for k, v in model.named_parameters() if not k.startswith("backbone.")], lr=rates[1])
    ], weight_decay=recipe["wd"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=rates,
                                                   total_steps=total_steps, pct_start=recipe["pct_start"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024.)
    return optimizer, scheduler, scaler


def elapsed_before_resume(out, saved_seconds):
    """Charge failed mid-epoch work too; never refund a stopped attempt's wall time."""
    seconds = float(saved_seconds)
    for file in Path(out).glob("stop-*.json"):
        receipt = json.loads(file.read_text(encoding="utf-8"))
        seconds = max(seconds, float(receipt["elapsed_seconds"]))
    return seconds


def accumulation_group(index, microbatches, accumulation):
    final = (index + 1) % accumulation == 0 or index + 1 == microbatches
    count = min(accumulation, microbatches - (index // accumulation) * accumulation)
    return final, count


def average_registered_epochs(out, recipe):
    import torch
    from tools.exp112_full_swa import swa_average
    states = [torch.load(Path(out) / f"epoch_{e:02d}.pt", map_location="cpu", weights_only=True)["model"]
              for e in recipe["swa_1based"]]
    return swa_average(states)


def train(args):
    """Same recipe on one4070 or torchrun2xT4; only explicit train initializes CUDA."""
    invocation_start = time.monotonic()
    from tools import exp143_v2 as guard
    from tools.exp113_radr50_local import install_gold_blocker
    from tools.exp112_full_swa import recompute_bn, state_dict_sha256
    p = guard.load_registration(args.protocol, args.corpus, args.init_weights)
    guard.require_validation(args.protocol)
    install_gold_blocker(ROOT)
    install_portable_gold_blocker()
    # No labels file is opened here: only the registered non-Gold training subset.
    if args.device != "cuda":
        raise ValueError("production training requires CUDA; use validate for CPU fakes")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    settings = guard.execution_settings(p, world, args.precision, args.seed, args.grad_checkpointing)
    recipe = p["recipe"]
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
    ds = StudyDataset(args.corpus, ids, labels, args.seed)
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
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p["pos_weight"], device=device))
    start_epoch, updates, skips, previous_seconds = 0, 0, 0, 0.
    history = []
    run_identity = dict(protocol_sha256=guard.sha256(args.protocol), initial_state_sha256=initial_hash,
                        seed=args.seed, precision=args.precision, world_size=world,
                        contract=CONTRACT, train_n=len(ids), torch=torch.__version__,
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
        guard.write_new_json(out / "provenance.json", dict(run_identity, oai=False,
            validation="none", gold_access="blocked", checkpoint_selection="fixed-SWA",
            swa_epochs_1based=recipe["swa_1based"], **settings))
        # A projected-budget abort in the first epoch still leaves a restart point.
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
    def check_budget():
        nonlocal projected_once
        elapsed = wall()
        if world == 2:
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor.item())
        if elapsed > limit - reserve:
            raise TimeoutError("budget reserve reached; incomplete fit, NOT validation success")
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
            if projected > limit:
                raise TimeoutError("100-update projection exceeds registered wall budget")
    try:
        check_budget()
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
                x, y, _ = unpack_batch(batch)
                final, group_size = accumulation_group(bi, len(loader), accum)
                sync = contextlib.nullcontext() if final or world == 1 else model.no_sync()
                with sync:
                    with amp_context(device, args.precision):
                        logits = model(x.to(device, non_blocking=True)).float()
                        loss = loss_fn(logits, y.to(device, non_blocking=True))
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
                    check_budget()
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
            bn_ds = StudyDataset(args.corpus, ids, labels, args.seed, augment=False)
            bn_ds.set_epoch(recipe["epochs"])
            bn_loader = DataLoader(bn_ds, batch_size=recipe["bn_batch"], shuffle=False, drop_last=False,
                                   num_workers=args.workers, collate_fn=collate, pin_memory=True,
                                   worker_init_fn=worker_init)
            class BudgetLoader:
                def __iter__(self):
                    for b in bn_loader:
                        # Rank0-only BN: NEVER use the collective training guard here.
                        if wall() > limit:
                            raise TimeoutError("wall budget reached during SWA BN replay")
                        yield unpack_batch(b)
            bn = recompute_bn(raw_model, BudgetLoader(), device, amp_context(device, args.precision))
            if wall() > limit:
                raise TimeoutError("wall budget reached after SWA BN replay")
            state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
            payload = dict(model=state, arch=ARCH, res=384, inference_res=320, lab=guard.TARGETS,
                contract=CONTRACT, k_eval=94, norm="imagenet", window_policy="legacy",
                epoch=recipe["epochs"] - 1, tag=out.name, amp_train=args.precision, amp_eval="fp16", fold=-1,
                label_set=recipe["labels"], provenance=run_identity, seed=args.seed,
                checkpoint_kind="fixed-swa-bn", swa_epochs_1based=recipe["swa_1based"], oai=False)
            guard.save_torch_new(out / "model_best.pt", payload)
            if wall() > limit:
                raise TimeoutError("wall budget reached while saving final checkpoint")
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


def scratch_directory(path=None):
    """Resolve without writing; probe outputs cannot touch any experiment run."""
    root = (ROOT / ".local/exp143-v2-probes").resolve()
    out = Path(path).resolve() if path else root / str(time.time_ns())
    try:
        relative = out.relative_to(root)
    except ValueError:
        raise ValueError("probe output must be under .local/exp143-v2-probes/") from None
    if not relative.parts:
        raise ValueError("probe requires a new scratch subdirectory")
    return out


def probe(args):
    """Two isolated sequential CUDA processes; no fit/checkpoint/validation writes."""
    import subprocess
    from tools import exp143_v2 as guard
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("probe measures one local GPU; two-T4 fit has its own admission gate")
    p = guard.load_registration(args.protocol, args.corpus, args.init_weights)
    guard.require_validation(args.protocol)
    settings = guard.execution_settings(p, 1, args.precision, p["recipe"]["seed_local"])
    if not 1 <= args.steps <= settings["updates_per_epoch"] - 3 or not 0 <= args.workers <= 12:
        raise ValueError("probe steps must fit one full epoch after two warmups; workers 0..12")
    out = scratch_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ("on", "off"):
        command = [sys.executable, str(ROOT / "tools/exp143_v2.py"), "_probe-mode",
                   "--protocol", str(Path(args.protocol).resolve()),
                   "--corpus", str(Path(args.corpus).resolve()),
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
                  results=results, steps_per_mode=args.steps, warmup_updates_per_mode=2,
                  caution="Short sample, not full-run/BN timing; full-fit 100-update gate still applies")
    guard.write_new_json(out / "report.json", report)
    print(json.dumps(dict(report, scratch_directory=str(out)), indent=2), flush=True)


def probe_mode(args):
    """Internal subprocess entry; called only by explicit probe. Never saves weights."""
    invocation_start = time.monotonic()
    from tools import exp143_v2 as guard
    from tools.exp113_radr50_local import install_gold_blocker
    from tools.exp112_full_swa import state_dict_sha256
    out = scratch_directory(args.out)
    if not out.is_dir() or (out / (args.mode + ".json")).exists():
        raise ValueError("internal probe requires an existing new scratch directory")
    p = guard.load_registration(args.protocol, args.corpus, args.init_weights)
    guard.require_validation(args.protocol)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("single-GPU probe only")
    seed = p["recipe"]["seed_local"]
    settings = guard.execution_settings(p, 1, args.precision, seed, args.mode == "on")
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
        ds = StudyDataset(args.corpus, p["train_ids"], p["train_labels"], seed)
        ds.set_epoch(0)
        sampler = DistributedSampler(ds, 1, 0, shuffle=True, seed=seed, drop_last=True)
        sampler.set_epoch(0)
        loader = DataLoader(ds, batch_size=settings["batch"], sampler=sampler, drop_last=True,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate,
                            persistent_workers=args.workers > 0, worker_init_fn=worker_init)
        optimizer, scheduler, scaler = make_optimization(model, p["recipe"], settings["total_steps"], args.precision)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p["pos_weight"], device=device))
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
                x, y, _uids = unpack_batch(next(iterator))
                with amp_context(device, args.precision):
                    logits = model(x.to(device, non_blocking=True)).float()
                    loss = loss_fn(logits, y.to(device, non_blocking=True))
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
        result["admissible"] = estimate["fits_budget"] and result["skipped_updates"] == 0
    except torch.cuda.OutOfMemoryError:
        result.update(status="CUDA_OOM_SCRATCH_ONLY", admissible=False)
    finally:
        # A fresh child for the next mode guarantees OOM/allocator/BN isolation.
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      elapsed_seconds=time.monotonic() - invocation_start)
        guard.write_new_json(out / (args.mode + ".json"), result)
    print(json.dumps(result, indent=2), flush=True)
