"""EXP143 data/model/training; imports are CPU passive. No downloads or cached init.

The Raptor head/SWA are imported from existing implementations. New geometry and
inference mirror follow the scored public source cited in EXP143/PROTOCOL.md.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "training/src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

IMG = TRAIN_RES = 384
EVAL_RES = 320
DEPTH = 96
SLOT_SIZES = [26, 22, 18, 12, 18]
K_TRAIN, K_EVAL = 12, 94
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
CONTRACT = "exp143_raptor96_384_c140_s02"
SPEC = dict(slots=[["Sagittal", 1, 26], ["Sagittal", 0, 22], ["Coronal", 1, 18],
                   ["Coronal", 0, 12], ["Axial", -1, 18]],
            depth=96, img=384, crop_mm=140.0, span=[.02, .98])
MEAN = np.asarray([.485, .456, .406], np.float32).reshape(1, 3, 1, 1)
STD = np.asarray([.229, .224, .225], np.float32).reshape(1, 3, 1, 1)


def unpack_batch(batch):
    """PyTorch pin_memory converts an ordinary batch tuple to a list."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise TypeError("expected (images, labels, UID-list), tuple OR list")
    x, y, ids = batch
    if x.ndim != 5 or y.ndim != 2 or x.shape[0] != y.shape[0] or len(ids) != x.shape[0]:
        raise ValueError("invalid study batch shape")
    return x, y, ids


def mirror_triplets(tri, centers):
    if sum(SLOT_SIZES) != DEPTH:
        raise ValueError("slot sizes do not sum to depth")
    slots = np.searchsorted(np.cumsum(SLOT_SIZES), np.asarray(centers), side="right")
    out = tri.copy()
    sag = slots < 2
    out[sag] = tri[sag, ::-1]
    out[~sag] = tri[~sag, :, :, ::-1]
    return out


def windows(vol, mask, train=False, seed=42, epoch=0, uid="x", mirror=False):
    from raptor_common import eval_centers, train_centers, study_rng
    if vol.shape != (DEPTH, IMG, IMG) or vol.dtype != np.uint8 or mask.shape != (DEPTH,):
        raise ValueError("EXP143 requires native uint8 96x384x384, not an old corpus")
    if not mask.any():
        raise ValueError("empty study cannot enter EXP143")
    if train and mirror:
        raise ValueError("mirror is inference only")
    centers = (train_centers(mask, DEPTH, K_TRAIN, study_rng(seed, epoch, uid)) if train
               else eval_centers(mask, DEPTH, K_EVAL))
    cs = np.asarray(centers)
    tri = np.asarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])
    if mirror:
        tri = mirror_triplets(tri, centers)
    res = TRAIN_RES if train else EVAL_RES
    low = (IMG - res) // 2
    x = tri[:, :, low:low + res, low:low + res].astype(np.float32) / 255.
    if train:
        x = np.clip(x * (1 + (study_rng(seed, epoch, uid, "intensity").random() - .5) * .20), 0, 1)
    return np.ascontiguousarray((x - MEAN) / STD)


class Corpus:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if (self.manifest.get("contract") != CONTRACT or self.manifest.get("spec") != SPEC
                or not self.manifest.get("finished")):
            raise ValueError("incomplete or incompatible EXP143 corpus")
        self.ids = self.manifest["ids"]
        self.rows = {u: i for i, u in enumerate(self.ids)}
        if len(self.rows) != len(self.ids):
            raise ValueError("duplicate corpus UIDs")

    def get(self, uid):
        row = self.rows[uid]
        volume = np.load(self.directory / f"study_{row:05d}.npy", mmap_mode="r", allow_pickle=False)
        receipt = json.loads((self.directory / f"study_{row:05d}.json").read_text(encoding="utf-8"))
        if receipt["uid"] != uid:
            raise ValueError("receipt UID mismatch")
        return volume, np.asarray(receipt["mask"], np.uint8)


class StudyDataset:
    def __init__(self, corpus, ids, labels, seed, augment=True):
        import torch
        self.path, self.ids, self.labels = str(corpus), list(ids), labels
        self.seed, self.augment = seed, augment
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.corpus = None

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __getitem__(self, i):
        import torch
        from raptor_common import train_centers, study_rng
        if self.corpus is None:
            self.corpus = Corpus(self.path)
        uid, ep = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.get(uid)
        if self.augment:
            x = windows(vol, mask, True, self.seed, ep, uid)
        else:  # BN recalibration: K12 deterministic draw, full384, no intensity jitter.
            cs = np.asarray(train_centers(mask, DEPTH, K_TRAIN, study_rng(self.seed, ep, uid)))
            x = np.asarray(vol[np.stack([cs - 1, cs, cs + 1], 1)], np.float32) / 255.
            x = np.ascontiguousarray((x - MEAN) / STD)
        return torch.from_numpy(x), torch.tensor(self.labels[uid], dtype=torch.float32), uid


def collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]


def worker_init(_worker_id):
    from tools.exp113_radr50_local import install_gold_blocker
    import torch
    install_gold_blocker(ROOT)
    install_portable_gold_blocker()
    torch.set_num_threads(1)


def install_portable_gold_blocker():
    """Also block mounted competition train.csv on Kaggle, not just local ROOT."""
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.name.lower() == "train.csv" or "gold" in {s.lower() for s in path.parts}:
                raise PermissionError("Gold/official train.csv forbidden during EXP143 fit")
    sys.addaudithook(audit)


def build_model(res=384, backbone_factory=None):
    """Reuse exact Raptor head; explicit resolution for timm's RMLP buffers."""
    from raptor_common import build_model as original_model
    if backbone_factory is None:
        import timm
        def backbone_factory():
            return timm.create_model(ARCH, pretrained=False, num_classes=0,
                                     in_chans=3, global_pool="avg", img_size=res)
    # Import/wrap the head without constructing a second real backbone.
    # original_model resolves build_backbone in its module at invocation time.
    import raptor_common
    previous = raptor_common.build_backbone
    try:
        raptor_common.build_backbone = lambda arch, pretrained: backbone_factory()
        model = original_model(ARCH, pretrained=False, drop=.2)
    finally:
        raptor_common.build_backbone = previous
    return model


def load_imagenet(model, path):
    """Explicit ImageNet-only timm backbone state; no load of knee head weights."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or "model" in state or any(k.startswith("backbone.") for k in state):
        raise ValueError("require plain ImageNet backbone state, not a Raptor checkpoint")
    state = dict(state)
    for key in ("head.fc.weight", "head.fc.bias", "head.weight", "head.bias"):
        state.pop(key, None)
    model.backbone.load_state_dict(state, strict=True)


def amp_context(device, precision):
    import torch
    if str(device) == "cpu":
        if precision != "none":
            raise ValueError("CPU requires precision none")
        return contextlib.nullcontext()
    if precision not in ("bf16", "fp16"):
        raise ValueError("GPU precision must be fp16 or bf16")
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def infer_volume(model, vol, mask, device="cpu", precision="none", chunk=16):
    import torch
    if any(m.training for m in model.modules()):
        raise ValueError("inference requires eval mode")
    probs = []
    with torch.inference_mode():
        for mirror in (False, True):
            # Keep the full windows on CPU; only chunk16 enters GPU memory.
            x = torch.from_numpy(windows(vol, mask, mirror=mirror))
            with amp_context(device, precision):
                feats = torch.cat([model.backbone(z.to(device)) for z in x.split(chunk)], 0)
                logits = model.head(feats.unsqueeze(0)).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite prediction; no fallback")
            probs.append(logits.sigmoid()[0].cpu().numpy())
    return .5 * (probs[0] + probs[1])


def tiny_validation():
    """CPU-only forward and actual pin_memory tuple/list conversion; no CUDA query."""
    import torch
    from torch import nn
    from torch.utils.data._utils.pin_memory import pin_memory
    class FakeBackbone(nn.Module):
        num_features = 4
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)
        def forward(self, x):
            return self.conv(x).mean((-1, -2))
    class FakePinnable:
        def pin_memory(self):
            return self
    converted = pin_memory((FakePinnable(), FakePinnable(), ["fake"]))
    if not isinstance(converted, list):
        raise AssertionError("expected torch pin_memory's tuple-to-list behavior")
    # torch.manual_seed may call CUDA seed hooks; only the CPU generator is seeded.
    torch.random.default_generator.manual_seed(42)
    model = build_model(backbone_factory=FakeBackbone).eval()
    batch = (torch.zeros(2, 3, 3, 8, 8), torch.zeros(2, 12), ["a", "b"])
    with torch.no_grad():
        for b in (batch, list(batch)):
            x, _, _ = unpack_batch(b)
            out = model(x)
            if out.shape != (2, 12) or not torch.isfinite(out).all():
                raise AssertionError("fake CPU forward failed")
    return dict(fake_forward=[2, 12], batch_tuple=True, batch_list=True,
                actual_pin_memory_tuple_returns_list=True, cuda_calls=0)


def _cpu_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def train(args):
    """Same recipe on one4070 or torchrun2xT4; only explicit train initializes CUDA."""
    invocation_start = time.monotonic()
    from tools import exp143_own0949 as guard
    from tools.exp113_radr50_local import install_gold_blocker
    from tools.exp112_full_swa import swa_average, recompute_bn, state_dict_sha256
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
    if world not in (1, 2) or (world == 2 and args.precision != "fp16"):
        raise ValueError("registered: one local GPU or two fp16 T4 GPUs")
    if (world, args.seed) not in ((1, 42), (2, 43)):
        raise ValueError("seed42 local pilot; seed43 two-GPU Kaggle trajectory only")
    cap = 9. if world == 1 else 11.
    if not 0 < args.max_hours <= cap or not 0 <= args.workers <= 12:
        raise ValueError("wall/worker budget exceeds registration")
    out = Path(args.out)
    if not args.resume and out.exists():
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
    model.backbone.set_grad_checkpointing(True)
    model.to(device)
    raw_model = model
    if world == 2:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    ids, labels = p["train_ids"], p["train_labels"]
    ds = StudyDataset(args.corpus, ids, labels, args.seed)
    sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(ds, batch_size=2, sampler=sampler, drop_last=True,
                        num_workers=args.workers, pin_memory=True, collate_fn=collate,
                        persistent_workers=args.workers > 0, worker_init_fn=worker_init)
    accum = 4 // world
    updates_per_epoch = math.ceil(len(loader) / accum)
    total_steps = 16 * updates_per_epoch
    optimizer = torch.optim.AdamW([
        dict(params=raw_model.backbone.parameters(), lr=3e-5),
        dict(params=[v for k, v in raw_model.named_parameters() if not k.startswith("backbone.")], lr=1e-3)
    ], weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[3e-5, 1e-3],
                                                   total_steps=total_steps, pct_start=.15)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16", init_scale=1024.)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p["pos_weight"], device=device))
    start_epoch, updates, skips, previous_seconds = 0, 0, 0, 0.
    history = []
    run_identity = dict(protocol_sha256=guard.sha256(args.protocol), initial_state_sha256=initial_hash,
                        seed=args.seed, precision=args.precision, world_size=world,
                        contract=CONTRACT, train_n=len(ids), torch=torch.__version__)
    if args.resume:
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != run_identity:
            raise ValueError("resume environment/registration/recipe changed")
        raw_model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, updates, skips = saved["epoch"], saved["updates"], saved["skips"]
        previous_seconds, history = saved["seconds"], saved["history"]
    elif rank == 0:
        guard.write_new_json(out / "provenance.json", dict(run_identity, oai=False,
            validation="none", gold_access="blocked", checkpoint_selection="fixed13,15,16",
            max_hours=args.max_hours, batch=2, accumulation=accum))
        # A projected-budget abort in the first epoch still leaves a restart point.
        guard.save_torch_atomic(out / "resume.pt", dict(
            model={k: v.detach().cpu() for k, v in raw_model.state_dict().items()}, epoch=0,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            updates=0, skips=0, seconds=time.monotonic() - invocation_start,
            history=[], identity=run_identity))
    start = invocation_start
    rate_start = time.monotonic()
    limit = args.max_hours * 3600
    wall = lambda: previous_seconds + time.monotonic() - start
    this_updates = 0
    def check_budget():
        elapsed = wall()
        if elapsed > limit - 600:
            raise TimeoutError("budget reserve reached; incomplete fit, NOT validation success")
        if this_updates == 100:
            seconds_per = (time.monotonic() - rate_start) / 100
            projected = elapsed + (total_steps - updates - skips) * seconds_per + 600
            if rank == 0:
                guard.write_new_json(out / f"timing-epoch{start_epoch:02d}.json",
                    dict(seconds_per_update=seconds_per, projected_seconds=projected, limit_seconds=limit))
            if projected > limit:
                raise TimeoutError("100-update projection exceeds registered wall budget")
    try:
        for epoch in range(start_epoch, 16):
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
                final = (bi + 1) % accum == 0 or bi + 1 == len(loader)
                group_size = min(accum, len(loader) - (bi // accum) * accum)
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
                    nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=args.precision != "fp16")
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
            members = [torch.load(out / f"epoch_{e:02d}.pt", map_location="cpu", weights_only=True)["model"]
                       for e in (13, 15, 16)]
            raw_model.load_state_dict(swa_average(members), strict=True)
            del members
            raw_model.backbone.set_grad_checkpointing(False)
            bn_ds = StudyDataset(args.corpus, ids, labels, args.seed, augment=False)
            bn_ds.set_epoch(16)
            bn_loader = DataLoader(bn_ds, batch_size=2, shuffle=False, drop_last=False,
                                   num_workers=args.workers, collate_fn=collate, pin_memory=True,
                                   worker_init_fn=worker_init)
            class BudgetLoader:
                def __iter__(self):
                    for b in bn_loader:
                        check_budget()
                        yield unpack_batch(b)
            bn = recompute_bn(raw_model, BudgetLoader(), device, amp_context(device, args.precision))
            check_budget()
            state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
            payload = dict(model=state, arch=ARCH, res=384, inference_res=320, lab=guard.TARGETS,
                contract=CONTRACT, k_eval=94, norm="imagenet", window_policy="legacy",
                epoch=15, tag=out.name, amp_train=args.precision, amp_eval="fp16", fold=-1,
                label_set="jev-hybrid-soft", provenance=run_identity, seed=args.seed,
                checkpoint_kind="fixed-swa-bn", swa_epochs_1based=[13, 15, 16], oai=False)
            guard.save_torch_new(out / "model_best.pt", payload)
            guard.write_new_json(out / "summary.json", dict(status="COMPLETE_TRAINING_NO_EVALUATION",
                identity=run_identity, epochs=16, updates=updates, skips=skips, history=history,
                seconds=wall(), bn=bn, model_sha256=guard.sha256(out / "model_best.pt"),
                swa_member_sha256={str(e): guard.sha256(out / f"epoch_{e:02d}.pt") for e in (13, 15, 16)}))
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
