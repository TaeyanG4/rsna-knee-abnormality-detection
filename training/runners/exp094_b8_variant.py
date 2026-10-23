"""EXP094: pre-registered single-variable variants of the G1 B8 recipe (one arm per run).

Arms (tools/exp094_b8_variant.py registers them):
  seed43   — identical recipe, training seed 43            (H-B-05 seed noise floor)
  strat    — slot-stratified training window sampling     (H-B-08)
  geoaug   — one affine bundle shared across a study's K windows (H-B-04)
  slotdrop — zero one whole slot per study with p=0.2      (H-B-11)
Everything else equals EXP084_B_G1 / EXP090 control: CoAtNet 224, G1 T3455/V888, within-slot-valid windows,
clipped pos_weight, fixed-final epoch 8, 3448 exact updates, BF16 autocast, Gold blocker, no metric access.
Evaluation windows (K24, evenly spaced) are never changed, so every arm is scored exactly like B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "storage" / "cache" / "derived"
EXPERIMENTS = ROOT / "artifacts" / "experiments"
DEFAULT_LABELS = ROOT / "storage" / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv"


class VariantStudyWindows:
    """Wraps the unchanged StudyWindows; applies exactly one pre-registered training-time variation."""

    def __init__(self, baseline, variant, slot_dropout=0.0, aug=None):
        self.baseline, self.variant, self.slot_dropout, self.aug = baseline, variant, slot_dropout, aug
        self.ids = baseline.ids
        self.train = baseline.train

    def __len__(self):
        return len(self.baseline)

    def set_epoch(self, epoch):
        self.baseline.set_epoch(epoch)

    def _slot_tables(self, corpus):
        sizes = [int(s[2]) for s in corpus.spec["slots"]]
        bounds, start = [], 0
        for size in sizes:
            bounds.append((start, start + size)); start += size
        return bounds

    def _stratified_centers(self, mask, corpus, k, rng):
        from raptor_common import select_centers
        bounds = self._slot_tables(corpus)
        per_slot = []
        for lo, hi in bounds:
            cs = [c for c in range(lo + 1, hi - 1) if np.all(mask[c - 1:c + 2] > 0)]
            if cs:
                per_slot.append(cs)
        if not per_slot:
            return select_centers(mask, corpus.depth, k, corpus.spec, "within-slot-valid", rng)
        n = len(per_slot); base, rem = divmod(k, n)
        extra = list(range(n)); rng.shuffle(extra)
        counts = [base + (1 if i in extra[:rem] else 0) for i in range(n)]
        out = []
        for cs, cnt in zip(per_slot, counts):
            pool = cs * (cnt // len(cs) + 1); rng.shuffle(pool); out.extend(pool[:cnt])
        rng.shuffle(out)
        return out

    def __getitem__(self, index):
        import torch
        from raptor_common import Corpus, study_rng, select_centers, windows_from
        from raptor_data import IMAGENET_MEAN, IMAGENET_STD
        b = self.baseline
        if not b.train or self.variant == "seed":
            return b[index]
        if b.corpus is None:
            b.corpus = Corpus(b.corpus_dir)
        uid, epoch = b.ids[index], int(b.epoch.item())
        vol, mask = b.corpus.volume(uid), b.corpus.mask(uid)
        rng = study_rng(b.seed, epoch, uid)
        if self.variant == "strat":
            centers = self._stratified_centers(mask, b.corpus, b.k, rng)
        else:
            centers = select_centers(mask, b.corpus.depth, b.k, b.corpus.spec, b.window_policy, rng)
        x = windows_from(vol, centers, b.res)
        intensity = study_rng(b.seed, epoch, uid, "intensity").random()
        x = np.clip(x * (1 + (intensity - .5) * b.jitter), 0, 1)
        if self.variant == "geoaug":
            import torch.nn.functional as F
            g = study_rng(b.seed, epoch, uid, "geoaug")
            tx, ty = (g.random() - .5) * 2 * .06, (g.random() - .5) * 2 * .06
            s = .9 + g.random() * .2
            th = (g.random() - .5) * 2 * math.radians(8)
            flip = -1.0 if g.random() < .5 else 1.0
            A = torch.tensor([[math.cos(th) * s * flip, -math.sin(th) * s, tx], [math.sin(th) * s * flip, math.cos(th) * s, ty]], dtype=torch.float32)
            t = torch.from_numpy(x)
            grid = F.affine_grid(A.unsqueeze(0).expand(t.shape[0], 2, 3), t.shape, align_corners=False)
            x = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False).numpy()
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        if self.variant == "slotdrop" and self.slot_dropout > 0:
            g = study_rng(b.seed, epoch, uid, "slotdrop")
            if g.random() < self.slot_dropout:
                bounds = self._slot_tables(b.corpus)
                present = sorted({next(i for i, (lo, hi) in enumerate(bounds) if lo <= c < hi) for c in centers if c != -1})
                if present:
                    drop = present[int(g.random() * len(present))]
                    lo, hi = bounds[drop]
                    zero = ((0 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
                    for j, c in enumerate(centers):
                        if c != -1 and lo <= c < hi:
                            x[j] = zero
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite input for {uid}")
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(b.labels[uid]), uid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", choices=["seed", "strat", "geoaug", "slotdrop"], required=True)
    ap.add_argument("--slot-dropout", type=float, default=0.0)
    ap.add_argument("--corpus", default=str(CACHE / "RAPTOR64_336_C140_S02"))
    ap.add_argument("--arch", default="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k-eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--bb-lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--amp-init-scale", type=float, default=1024.0)
    ap.add_argument("--norm", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--eval-labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--window-policy", choices=["within-slot-valid"], default="within-slot-valid")
    ap.add_argument("--runtime-policy", choices=["strict", "throughput"], default="strict")
    ap.add_argument("--eval-amp", choices=["fp16", "bf16", "none"], default="fp16")
    ap.add_argument("--fixed-epoch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", default="timm")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools import exp094_b8_variant as guard
    from tools import exp087_training as trace_tools
    protocol, ordered, arm = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, build_model, environment, load_label_csv, seed_all, sha256_text, write_predictions, sha256_file)
    from raptor_data import StudyWindows, collate
    from raptor_runtime import configure_runtime
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    runtime = configure_runtime(a.amp, dev, policy=a.runtime_policy)
    eval_runtime = configure_runtime(a.eval_amp, dev, policy=a.runtime_policy)
    guard.worker_init(0)
    run_environment = environment()
    seed_all(a.seed)
    out = EXPERIMENTS / a.tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); log_file.write(msg + "\n"); log_file.flush()

    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    eval_labels, eval_label_hash = load_label_csv(Path(a.eval_labels))
    group_rows = guard.read_rows(ROOT / protocol["groups"]["path"])
    train_ids, oof_ids = guard.validate_membership(corpus.ids.astype(str).tolist(), ordered, group_rows)
    if label_hash != protocol["label_hash"] or eval_label_hash != protocol["eval_label_hash"]:
        raise SystemExit("Label semantic hash mismatch")
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}
    log(f"{a.tag} | arm {a.arm} variant {a.variant} seed {a.seed} slot_dropout {a.slot_dropout} | train {len(train_ids)} | val {len(oof_ids)} | Gold and metrics disabled")
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pos_weight_np = np.clip((1 - prev) / prev, 1, 10).astype(np.float32)
    if not np.array_equal(pos_weight_np, np.asarray(protocol["loss"]["baseline_pos_weight"], dtype=np.float32)):
        raise SystemExit("T-only loss vector mismatch")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)
    tds = VariantStudyWindows(StudyWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy), a.variant, a.slot_dropout)
    shuffle_generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=dev == "cuda",
                    persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)
    eval_bs = max(1, a.bs // 2)

    def loader(ids):
        return DataLoader(StudyWindows(corpus.dir, ids, all_labels, a.res, a.k_eval, False, a.norm, a.seed, window_policy=a.window_policy), batch_size=eval_bs,
                          shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False, worker_init_fn=guard.worker_init)

    model = build_model(a.arch, pretrained=a.pretrained == "timm").to(dev)
    initial_hash = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        initial_hash.update(key.encode()); initial_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    provenance = {"code_sha256": {p: sha256_file(ROOT / "training" / "src" / p) for p in guard.CODE_NAMES}, "initial_state_sha256": initial_hash.hexdigest(),
                  "runtime": runtime, "eval_runtime": eval_runtime, "train_ids_ordered_sha256": guard.uid_hash(train_ids), "eval_ids_ordered_sha256": guard.uid_hash(oof_ids),
                  "eval_label_hash": eval_label_hash, "label_hash": label_hash, "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"),
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": "fixed-final", "gold_access": "disabled", "loss_policy": "clipped_positive_term",
                  "actual_pos_weight": pos_weight_np.tolist(), "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)),
                  "protocol_sha256": sha256_file(Path(a.protocol)), "arm": a.arm, "variant": a.variant, "seed": a.seed, "slot_dropout": a.slot_dropout,
                  "training_mode": a.mode, "scheduler_total_steps": 3448, "metric_access": "deferred_single_fixed_evaluation", "requested_workers": a.workers}
    provenance["initial_state_matches_B"] = provenance["initial_state_sha256"] == protocol["initial_state_sha256"]
    if a.seed == 42 and not provenance["initial_state_matches_B"] and not __import__("os").environ.get("KNEE_REMOTE_RELAX_ENV"):
        raise SystemExit("generic initial state differs from EXP084 B")
    provenance["environment_matches_baseline"] = all(run_environment.get(k) == protocol["environment"][k] for k in guard.ENV_KEYS)
    provenance["environment"] = {k: run_environment.get(k) for k in guard.ENV_KEYS}
    if not provenance["environment_matches_baseline"] and not __import__("os").environ.get("KNEE_REMOTE_RELAX_ENV"):
        raise SystemExit("runtime environment differs from the registered baseline environment (set KNEE_REMOTE_RELAX_ENV=1 on a remote GPU to record instead)")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, a.res, a.res, device=dev)); model.train()
    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(True); log("gradient checkpointing on")
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(math.ceil(len(tl) / a.accum) * a.epochs, 1)
    if len(tl) != 431 or steps != 3448 or eval_bs != 4:
        raise RuntimeError("B8 batch/update/evaluation contract differs")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    scaler = torch.amp.GradScaler("cuda", init_scale=a.amp_init_scale, enabled=a.amp == "fp16" and dev == "cuda")
    skipped = 0
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]

    @torch.no_grad()
    def evaluate(ids):
        model.eval(); P, U = [], []
        for x, y, uids in loader(ids):
            with torch.autocast(dev, dtype=eval_dtype or torch.float32, enabled=eval_dtype is not None and dev == "cuda"):
                logits = model(x.to(dev))
            if logits.shape != (len(uids), len(LAB)) or not torch.isfinite(logits).all():
                raise RuntimeError(f"invalid evaluation logits: {uids}")
            P.append(torch.sigmoid(logits.float()).cpu().numpy()); U += uids
        return np.concatenate(P), U

    history, best, optimizer_updates, pilot_times = [], None, 0, []
    t_start = time.time(); whole_update_start = time.monotonic()
    for ep in range(a.epochs):
        tds.set_epoch(ep); model.train(); tot, t0 = 0.0, time.time(); opt.zero_grad(set_to_none=True)
        for i, (x, y, batch_uids) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None and dev == "cuda"):
                logits = model(x)
                if logits.shape != (len(batch_uids), 12) or not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite or invalid training logits")
                loss = lossf(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at epoch {ep} step {i}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=a.amp != "fp16")
            scale_before = scaler.get_scale()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if a.amp == "fp16" and scaler.get_scale() < scale_before:
                skipped += 1
                if skipped > 34:
                    raise RuntimeError(f"FP16 optimizer skips exceed 1% ({skipped})")
            else:
                sched.step()
            optimizer_updates += 1; tot += loss.item()
            if a.mode == "pilot":
                torch.cuda.synchronize(); pilot_times.append(time.monotonic() - whole_update_start)
                if optimizer_updates == 32:
                    times = pilot_times[-24:]; projection = 1.3 * sum(times) / 24 * 3448 + 300
                    trace_tools.stop_loader(tl)
                    (out / "pilot-receipt.json").write_text(json.dumps(dict(status="PILOT_COMPLETE", arm=a.arm, updates=32, last24_mean_seconds=sum(times) / 24,
                                                                           projected_full_seconds=projection, peak_cuda_bytes=torch.cuda.max_memory_allocated(), all_finite=True), indent=1), encoding="utf-8")
                    log(f"pilot complete: projected full {projection:.0f}s"); log_file.close(); return
            whole_update_start = time.monotonic()
            if (i + 1) % 200 == 0:
                rate = (time.time() - t0) / (i + 1)
                log(f"  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f} | {rate:.2f}s/step, epoch ETA {rate * (len(tl) - i - 1) / 60:.0f} min")
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "epoch_seconds": round(time.time() - t0), "peak_gpu_gib": round(peak, 2), "optimizer_updates": optimizer_updates}
        history.append(rec); log(f"ep{ep} loss {rec['loss']:.4f} | fixed-final | {rec['epoch_seconds']}s | peak {peak:.1f} GiB")
        if ep == a.epochs - 1:
            best = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB, "contract": corpus.contract,
                    "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy, "epoch": ep, "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp, "provenance": provenance}
            torch.save(best, out / "model_best.pt")
        json.dump(history, open(out / "history.json", "w"), indent=1)
    cleanup = trace_tools.stop_loader(tl)
    if optimizer_updates != 3448:
        raise RuntimeError("full exact update gate failed before inference")
    provenance["fp16_skipped_updates"] = skipped
    model.load_state_dict(best["model"])
    P, U = evaluate(oof_ids)
    if U != oof_ids:
        raise RuntimeError("Final prediction UID order changed")
    write_predictions(out / "oof_best.csv", U, P)
    summary = {"contract": corpus.contract, "backbone": a.arch, "img": a.res, "max_epochs": a.epochs, "train": len(train_ids), "val": len(oof_ids), "gold": 0, "rows": len(oof_ids),
               "weak_oof_macro_auc": None, "selected_checkpoint": "best", "gold_access": "disabled", "loss_policy": "clipped_positive_term", "provenance": provenance,
               "selected_epoch_1based": best["epoch"] + 1, "optimizer_updates": optimizer_updates, "skipped_updates": skipped, "args": vars(a), "history": history,
               "environment": environment(), "worker_cleanup": cleanup, "total_seconds": round(time.time() - t_start), "metric_access": "deferred_single_fixed_evaluation", "arm": a.arm,
               "caveat": "exposed weak G1 development set; one seed; not D01/private evidence"}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv"]
    (out / "output-inventory.json").write_text(json.dumps({n: dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size) for n in names}, indent=1), encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
