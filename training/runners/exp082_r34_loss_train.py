"""EXP082 fixed RN20/ones entrypoint; generated explicit diff, no Gold access."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "storage" / "cache" / "derived"
EXPERIMENTS = ROOT / "artifacts" / "experiments"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
DEFAULT_LABELS = ROOT / "storage" / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv"
DEFAULT_FOLDS = EXPERIMENTS / "EXP026_GROUPED_FROZEN_PROBE" / "folds.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--corpus", default=str(CACHE / "RAPTOR64_336_C140_S02"))
    ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k-eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = bs*accum)")
    ap.add_argument("--bb-lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--amp-init-scale", type=float, default=1024., help="initial FP16 loss scale; skips and effective updates are recorded")
    ap.add_argument("--norm", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--eval-labels", default=str(DEFAULT_LABELS), help="fixed reference labels, independent of training labels")
    ap.add_argument("--exclude-ids", help="CSV with StudyInstanceUID for a preregistered common quality quarantine")
    ap.add_argument("--window-policy", choices=["legacy", "global-valid", "within-slot-valid"], default="legacy")
    ap.add_argument("--runtime-policy", choices=["strict", "throughput"], default="strict")
    ap.add_argument("--eval-amp", choices=["fp16", "bf16", "none"], default="fp16")
    ap.add_argument("--fixed-epoch", type=int, help="1-based final epoch, must equal --epochs; no OOF checkpoint/SWA selection")
    ap.add_argument("--fold-file", default=str(DEFAULT_FOLDS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0, help="held-out OOF fold; -1 trains on everything (no OOF)")
    ap.add_argument("--select", default="oof", choices=["oof", "gold"], help="epoch selection metric")
    ap.add_argument("--oof-subsample", type=int, default=0, help="evaluate only N OOF studies per epoch (0 = all)")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pretrained", default="timm", help="'timm' or a checkpoint path to start from")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="cpu is for wiring checks only")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools import exp082_r34_loss as guard
    protocol, ordered = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, build_model, environment, load_folds,
                               load_label_csv, macro_auc, seed_all, sha256_text,
                               write_predictions, sha256_file)
    from raptor_data import StudyWindows, collate
    from raptor_runtime import configure_runtime
    if a.smoke:
        a.epochs, a.bs, a.accum, a.k, a.k_eval, a.limit, a.workers, a.oof_subsample, a.topk = 1, 2, 1, 4, 6, (a.limit or 24), 0, 16, 2
    if a.select != "oof":
        raise SystemExit("Gold58 checkpoint selection is prohibited; use --select oof")
    if a.fixed_epoch is not None and a.fixed_epoch != a.epochs:
        raise SystemExit("--fixed-epoch must equal --epochs (1-based); choose it before running")
    if a.fold < 0 and a.fixed_epoch is None:
        raise SystemExit("Full-data training requires --fixed-epoch; Gold fallback is prohibited")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    runtime = configure_runtime(a.amp, dev, policy=a.runtime_policy)
    eval_runtime = configure_runtime(a.eval_amp, dev, policy=a.runtime_policy)
    run_environment = environment()  # Audit imports happen before the original seed boundary.
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
    folds, split_hash = load_folds(Path(a.fold_file))
    excluded = set(protocol["excluded_uids"])
    train_ids, oof_ids = guard.reconstruct_ids(corpus.ids, folds, excluded, protocol["expected_ids"])
    guard.validate_ordered_ids(train_ids, oof_ids, ordered)
    if any(u not in labels for u in train_ids) or any(u not in eval_labels for u in oof_ids):
        raise SystemExit("Frozen train/eval label coverage changed")
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: eval_labels[u] for u in oof_ids}}
    guard.validate_label_contract(label_hash, eval_label_hash, split_hash, protocol)
    log(f"{a.tag} | corpus {corpus.contract} ({len(corpus)}) | train {len(train_ids)} | oof {len(oof_ids)} (fold {a.fold}/{a.folds}) | Gold disabled")
    log(f"arch {a.arch} res {a.res} k {a.k}/{a.k_eval} bs {a.bs}x{a.accum} epochs {a.epochs} amp {a.amp} select {a.select} | {dev}")
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    baseline_pos_weight = np.clip((1 - prev) / prev, 1, 10).astype(np.float32)
    guard.validate_loss_vectors(baseline_pos_weight, protocol)
    pos_weight = torch.ones(len(LAB), dtype=torch.float32, device=dev)

    tds = StudyWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
    shuffle_generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate,
                    pin_memory=dev == "cuda", persistent_workers=a.workers > 0, generator=shuffle_generator)
    eval_bs = max(1, a.bs // 2)

    def loader(ids):
        return DataLoader(StudyWindows(corpus.dir, ids, all_labels, a.res, a.k_eval, False, a.norm, a.seed, window_policy=a.window_policy), batch_size=eval_bs,
                          shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False)

    rng = random.Random(a.seed)
    oof_epoch_ids = rng.sample(oof_ids, a.oof_subsample) if a.oof_subsample and len(oof_ids) > a.oof_subsample else oof_ids
    use_timm = a.pretrained == "timm"
    model = build_model(a.arch, pretrained=use_timm).to(dev)
    if not use_timm:
        ck = torch.load(a.pretrained, map_location="cpu", weights_only=False)
        log(f"init from {a.pretrained}: {model.load_state_dict(ck['model'], strict=False)}")
    source_files = [ROOT / "training" / "src" / name for name in guard.CODE_NAMES]
    initial_hash = __import__("hashlib").sha256()
    for key, value in sorted(model.state_dict().items()):
        initial_hash.update(key.encode()); initial_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    provenance = {"code_sha256": {p.name: sha256_file(p) for p in source_files},
                  "initial_state_sha256": initial_hash.hexdigest(), "runtime": runtime, "eval_runtime": eval_runtime,
                  "train_ids_sha256": sha256_text("\n".join(sorted(train_ids))),
                  "eval_ids_sha256": sha256_text("\n".join(sorted(oof_ids))),
                  "excluded_ids": sorted(excluded), "eval_label_hash": eval_label_hash,
                  "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"),
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": "fixed-final" if a.fixed_epoch else "development-oof"}
    provenance.update(adapter_sha256=sha256_file(Path(__file__)),
                      contract_helper_sha256=sha256_file(Path(guard.__file__)),
                      protocol_sha256=sha256_file(Path(a.protocol)),
                      train_ids_ordered_sha256=guard.uid_hash(train_ids),
                      eval_ids_ordered_sha256=guard.uid_hash(oof_ids),
                      gold_access="disabled", loss_policy="ones_positive_term",
                      baseline_pos_weight=baseline_pos_weight.tolist(),
                      actual_pos_weight=pos_weight.detach().cpu().tolist())
    guard.validate_pretraining(provenance, run_environment, protocol)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    # Preflight: some backbones (CoAtNet rmlp *_384 / *_224) only accept their native size.
    try:
        with torch.no_grad():
            model.eval()(torch.zeros(1, 1, 3, a.res, a.res, device=dev))
        model.train()
    except RuntimeError as exc:
        raise SystemExit(f"{a.arch} does not accept {a.res}px input ({str(exc)[:120]}). Use an arch built for {a.res}px, "
                         f"e.g. coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k for 224.") from None
    if a.grad_ckpt:
        try:
            model.backbone.set_grad_checkpointing(True); log("gradient checkpointing on")
        except Exception as exc:
            log(f"grad ckpt unsupported: {exc}")
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    if not len(tl):
        raise SystemExit("Not enough studies for one batch")
    steps = max(__import__("math").ceil(len(tl) / a.accum) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]
    scaler = torch.amp.GradScaler("cuda", init_scale=a.amp_init_scale, enabled=a.amp == "fp16" and dev == "cuda")

    @torch.no_grad()
    def evaluate(ids):
        model.eval(); P, Y, U = [], [], []
        for x, y, uids in loader(ids):
            with torch.autocast(dev, dtype=eval_dtype or torch.float32, enabled=eval_dtype is not None and dev == "cuda"):
                logits = model(x.to(dev))
            if logits.shape != (len(uids), len(LAB)) or not torch.isfinite(logits).all():
                raise RuntimeError(f"invalid evaluation logits: {uids}")
            o = torch.sigmoid(logits.float())
            if not torch.isfinite(o).all():
                raise RuntimeError(f"nonfinite evaluation output: {uids}")
            P.append(o.cpu().numpy()); Y.append(y.numpy()); U += uids
        P, Y = np.concatenate(P), np.concatenate(Y)
        macro, per = macro_auc(Y, P)
        return macro, per, P, U

    def state(epoch, extra):
        return {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB,
                "contract": corpus.contract, "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy,
                "epoch": epoch, "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp, "provenance": provenance, **extra}

    history, topk, best = [], [], None
    optimizer_updates, skipped_updates, consecutive_skips = 0, 0, 0
    t_start = time.time()
    for ep in range(a.epochs):
        tds.set_epoch(ep)
        model.train(); tot, t0 = 0.0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None and dev == "cuda"):
                group_size = min(a.accum, len(tl) - (i // a.accum) * a.accum)
                loss = lossf(model(x).float(), y) / group_size
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at epoch {ep} step {i}")
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0 or i + 1 == len(tl):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scale_before = scaler.get_scale()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before:
                    sched.step()
                    optimizer_updates += 1; consecutive_skips = 0
                else:
                    skipped_updates += 1; consecutive_skips += 1
                    log(f"FP16 overflow: optimizer/scheduler step skipped ({skipped_updates} total)")
                    if consecutive_skips >= 10:
                        raise RuntimeError("Ten consecutive optimizer skips; inspect FP16 stability")
            tot += loss.item() * group_size
            if (i + 1) % 200 == 0:
                rate = (time.time() - t0) / (i + 1)
                log(f"  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f} | {rate:.2f}s/step, epoch ETA {rate * (len(tl) - i - 1) / 60:.0f} min")
        train_s = time.time() - t0
        oof_auc, oof_per, _, _ = evaluate(oof_epoch_ids) if oof_epoch_ids and a.fixed_epoch is None else (None, {}, None, None)
        gold_auc = None  # Compatibility placeholder; Gold is never loaded or evaluated.
        score = float(ep) if a.fixed_epoch is not None else oof_auc
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "oof_auc": oof_auc, "gold_auc": gold_auc, "selection": score,
               "train_seconds": round(train_s), "epoch_seconds": round(time.time() - t0), "peak_gpu_gib": round(peak, 2),
               "optimizer_updates": optimizer_updates, "skipped_updates": skipped_updates}
        history.append(rec)
        log(f"ep{ep} loss {rec['loss']:.4f} | OOF {oof_auc} | checkpoint {'fixed-final' if a.fixed_epoch else 'development-oof'} | {rec['epoch_seconds']}s | peak {peak:.1f} GiB")
        if best is None or score > best["selection"]:
            best = state(ep, {"selection": score, "oof_auc": oof_auc, "gold_auc": gold_auc, "oof_per_target": oof_per})
            torch.save(best, out / "model_best.pt")
        if a.fixed_epoch is None:
            topk.append(state(ep, {"selection": score})); topk.sort(key=lambda s: -s["selection"]); del topk[max(1, a.topk):]
        json.dump(history, open(out / "history.json", "w"), indent=1)

    results = {}
    for name, st in [("best", best)] + ([("swa", None)] if len(topk) > 1 else []):
        if name == "swa":
            avg = {}
            for key, v0 in topk[0]["model"].items():
                avg[key] = (sum(s["model"][key].double() for s in topk) / len(topk)).to(v0.dtype) if v0.is_floating_point() else v0.clone()
            model.load_state_dict(avg)
            st = state([s["epoch"] for s in topk], {"swa_over": [s["epoch"] for s in topk]})
        else:
            model.load_state_dict(st["model"])
        r = {}
        if oof_ids:
            m, per, P, U = evaluate(oof_ids); write_predictions(out / f"oof_{name}.csv", U, P)
            r.update(oof_auc=m, oof_per_target=per, oof_rows=len(U))
        st.update(r); torch.save(st, out / f"model_{name}.pt"); results[name] = r
        log(f"{name}: OOF {r.get('oof_auc', float('nan')):.4f} | Gold disabled")
    chosen = "swa" if "swa" in results and results["swa"].get("oof_auc", -1) > results["best"].get("oof_auc", -1) else "best"
    summary = {"contract": corpus.contract, "backbone": a.arch, "img": a.res, "fold": a.fold, "folds": a.folds, "max_epochs": a.epochs,
               "best_epoch": best["epoch"], "train": len(train_ids), "val": len(oof_ids), "gold": 0, "rows": len(oof_ids),
               "weak_oof_macro_auc": results[chosen].get("oof_auc"), "selected_checkpoint": chosen,
               "gold_access": "disabled", "loss_policy": "ones_positive_term",
               "oof_per_target": results[chosen].get("oof_per_target"), "results": results, "no_gold_epoch_selection": True,
               "eval_label_hash": eval_label_hash, "window_policy": a.window_policy, "provenance": provenance,
               "no_oof_epoch_selection": a.fixed_epoch is not None, "selected_epoch_1based": best["epoch"] + 1,
               "optimizer_updates": optimizer_updates, "skipped_updates": skipped_updates,
               "args": {**vars(a), "weak_labels": str(Path(a.labels).relative_to(ROOT)) if Path(a.labels).is_relative_to(ROOT) else a.labels},
               "label_hash": label_hash, "split_hash": split_hash, "hash_scheme": "lf-v2", "study_set_hash": sha256_text("\n".join(sorted(train_ids + oof_ids))),
               "corpus_manifest": corpus.manifest.get("ids_sha256"), "history": history, "environment": environment(),
               "total_seconds": round(time.time() - t_start), "caveat": "exposed F2 weak development; known F0/F1 duplicate both in train; not independent/private evidence"}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log(f"done in {summary['total_seconds']}s -> {out}")


if __name__ == "__main__":
    main()
