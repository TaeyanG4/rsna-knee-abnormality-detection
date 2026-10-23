"""Train a Raptor-style model (2.5D windows + per-finding attention pooling) on a local corpus.

Differences from the public recipe, all deliberate:
  * grouped K-fold with a held-out OOF fold; the epoch is selected on OOF weak-label AUC,
    never on Gold58 (which is reported as a diagnostic only);
  * every output is a contract-tagged artifact: model_best.pt / model_swa.pt carry the
    corpus contract, OOF and gold predictions are CSVs the app's 예측 비교 can read, and
    summary.json is what the dashboard importer expects;
  * precision policy is explicit (bf16 locally; Kaggle T4 inference is fp16, so parity is
    checked separately with raptor_infer.py --amp fp16).

  python training/src/raptor_train.py --tag EXP060_RAPTOR_F0 --fold 0 --res 224 --epochs 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raptor_common import (CACHE, EXPERIMENTS, LAB, ROOT, Corpus, build_model, environment, eval_centers,  # noqa: E402
                           gold_labels, load_folds, load_label_csv, macro_auc, seed_all, sha256_text,
                           train_centers, windows_from, write_predictions)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
DEFAULT_LABELS = ROOT / "storage" / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv"
DEFAULT_FOLDS = EXPERIMENTS / "EXP026_GROUPED_FROZEN_PROBE" / "folds.csv"


class StudyWindows:
    def __init__(self, corpus_dir, ids, labels, res, k, train, norm, seed, jitter=0.20):
        self.corpus_dir, self.ids, self.labels, self.res, self.k = corpus_dir, list(ids), labels, res, k
        self.train, self.norm, self.seed, self.jitter = train, norm, seed, jitter
        self.corpus = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid = self.ids[i]
        vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        if self.train:
            rng = random.Random(hash((self.seed, uid, time.time_ns())))
            centers = train_centers(mask, self.corpus.depth, self.k, rng)
        else:
            centers = eval_centers(mask, self.corpus.depth, self.k)
        x = windows_from(vol, centers, self.res)
        if self.train and self.jitter:
            x = np.clip(x * (1.0 + (random.random() - 0.5) * self.jitter), 0, 1)
        if self.norm == "imagenet":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid


def collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--norm", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
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
    if a.smoke:
        a.epochs, a.bs, a.accum, a.k, a.k_eval, a.limit, a.workers, a.oof_subsample, a.topk = 1, 2, 1, 4, 6, (a.limit or 24), 0, 16, 2
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    out = EXPERIMENTS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); log_file.write(msg + "\n"); log_file.flush()

    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    gold = gold_labels()
    folds, split_hash = load_folds(Path(a.fold_file))
    gold_ids = [u for u in corpus.ids if u in gold]
    # fold -1 marks quarantined studies (report duplicates of Gold58); they never train or score.
    pool = [u for u in corpus.ids if u in labels and u not in gold and folds.get(u, -1) >= 0]
    if a.fold >= 0:
        oof_ids = [u for u in pool if folds[u] == a.fold]
        train_ids = [u for u in pool if folds[u] != a.fold]
    else:
        oof_ids, train_ids = [], pool
    if a.limit:
        train_ids = train_ids[:a.limit]
    all_labels = {**{u: labels[u] for u in train_ids + oof_ids}, **{u: gold[u] for u in gold_ids}}
    log(f"{a.tag} | corpus {corpus.contract} ({len(corpus)}) | train {len(train_ids)} | oof {len(oof_ids)} (fold {a.fold}/{a.folds}) | gold {len(gold_ids)}")
    log(f"arch {a.arch} res {a.res} k {a.k}/{a.k_eval} bs {a.bs}x{a.accum} epochs {a.epochs} amp {a.amp} select {a.select} | {dev}")
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pos_weight = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)

    tds = StudyWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate,
                    pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
    eval_bs = max(1, a.bs // 2)

    def loader(ids):
        return DataLoader(StudyWindows(corpus.dir, ids, all_labels, a.res, a.k_eval, False, a.norm, a.seed), batch_size=eval_bs,
                          shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False)

    rng = random.Random(a.seed)
    oof_epoch_ids = rng.sample(oof_ids, a.oof_subsample) if a.oof_subsample and len(oof_ids) > a.oof_subsample else oof_ids
    use_timm = a.pretrained == "timm"
    model = build_model(a.arch, pretrained=use_timm).to(dev)
    if not use_timm:
        ck = torch.load(a.pretrained, map_location="cpu", weights_only=False)
        log(f"init from {a.pretrained}: {model.load_state_dict(ck['model'], strict=False)}")
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
    steps = max(len(tl) // a.accum * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp == "fp16" and dev == "cuda")

    @torch.no_grad()
    def evaluate(ids):
        model.eval(); P, Y, U = [], [], []
        for x, y, uids in loader(ids):
            with torch.autocast(dev, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None and dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy()); U += uids
        P, Y = np.concatenate(P), np.concatenate(Y)
        macro, per = macro_auc(Y, P)
        return macro, per, P, U

    def state(epoch, extra):
        return {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB,
                "contract": corpus.contract, "k_eval": a.k_eval, "norm": a.norm, "epoch": epoch, "tag": a.tag, "amp_train": a.amp, **extra}

    history, topk, best = [], [], None
    t_start = time.time()
    for ep in range(a.epochs):
        model.train(); tot, n_steps, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None and dev == "cuda"):
                loss = lossf(model(x).float(), y) / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if n_steps < steps - 1:
                    sched.step()
                n_steps += 1
            tot += loss.item() * a.accum
            if (i + 1) % 200 == 0:
                rate = (time.time() - t0) / (i + 1)
                log(f"  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f} | {rate:.2f}s/step, epoch ETA {rate * (len(tl) - i - 1) / 60:.0f} min")
        train_s = time.time() - t0
        oof_auc, oof_per, _, _ = evaluate(oof_epoch_ids) if oof_epoch_ids else (float("nan"), {}, None, None)
        gold_auc, gold_per, _, _ = evaluate(gold_ids) if gold_ids else (float("nan"), {}, None, None)
        score = oof_auc if a.select == "oof" and oof_epoch_ids else gold_auc
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "oof_auc": oof_auc, "gold_auc": gold_auc, "selection": score,
               "train_seconds": round(train_s), "epoch_seconds": round(time.time() - t0), "peak_gpu_gib": round(peak, 2)}
        history.append(rec)
        log(f"ep{ep} loss {rec['loss']:.4f} | OOF {oof_auc:.4f} (n={len(oof_epoch_ids)}) | gold {gold_auc:.4f} | {rec['epoch_seconds']}s | peak {peak:.1f} GiB")
        if best is None or score > best["selection"]:
            best = state(ep, {"selection": score, "oof_auc": oof_auc, "gold_auc": gold_auc, "oof_per_target": oof_per, "gold_per_target": gold_per})
            torch.save(best, out / "model_best.pt")
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
        if gold_ids:
            m, per, P, U = evaluate(gold_ids); write_predictions(out / f"gold_{name}.csv", U, P)
            r.update(gold_auc=m, gold_per_target=per)
        st.update(r); torch.save(st, out / f"model_{name}.pt"); results[name] = r
        log(f"{name}: OOF {r.get('oof_auc', float('nan')):.4f} | gold {r.get('gold_auc', float('nan')):.4f}")
    chosen = "swa" if "swa" in results and results["swa"].get("oof_auc", -1) > results["best"].get("oof_auc", -1) else "best"
    summary = {"contract": corpus.contract, "backbone": a.arch, "img": a.res, "fold": a.fold, "folds": a.folds, "max_epochs": a.epochs,
               "best_epoch": best["epoch"], "train": len(train_ids), "val": len(oof_ids), "gold": len(gold_ids), "rows": len(oof_ids),
               "weak_oof_macro_auc": results[chosen].get("oof_auc"), "selected_checkpoint": chosen,
               "gold_once_after_weak_selected_epoch": results[chosen].get("gold_auc"), "gold_per_target": results[chosen].get("gold_per_target"),
               "oof_per_target": results[chosen].get("oof_per_target"), "results": results, "no_gold_epoch_selection": a.select == "oof",
               "args": {**vars(a), "weak_labels": str(Path(a.labels).relative_to(ROOT)) if Path(a.labels).is_relative_to(ROOT) else a.labels},
               "label_hash": label_hash, "split_hash": split_hash, "hash_scheme": "lf-v2", "study_set_hash": sha256_text("\n".join(sorted(train_ids + oof_ids))),
               "corpus_manifest": corpus.manifest.get("ids_sha256"), "history": history, "environment": environment(),
               "total_seconds": round(time.time() - t_start), "caveat": "weak-label OOF; Gold58 is an exposed diagnostic set"}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log(f"done in {summary['total_seconds']}s -> {out}")


if __name__ == "__main__":
    main()
