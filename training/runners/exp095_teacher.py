"""EXP095 teacher: ResNet34 fit on two inner folds of T3455, predictions for the held-out inner fold only.

Never reads V888 or Gold. Fixed-final epoch 20, BF16 autocast, within-slot-valid windows, clipped pos_weight from the
training subset. Output: artifacts/experiments/<tag>/{oof_fold.csv, model_best.pt, summary.json, provenance.json, history.json}.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--fold-index", type=int, required=True)
    ap.add_argument("--corpus", default=str(CACHE / "RAPTOR64_336_C140_S02")); ap.add_argument("--arch", required=True); ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--k-eval", type=int, default=24); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--fixed-epoch", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--accum", type=int, default=1); ap.add_argument("--bb-lr", type=float, default=5e-4); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "none"]); ap.add_argument("--eval-amp", choices=["fp16", "bf16", "none"], default="fp16")
    ap.add_argument("--norm", default="imagenet"); ap.add_argument("--labels", required=True); ap.add_argument("--eval-labels", required=True)
    ap.add_argument("--window-policy", choices=["within-slot-valid"], default="within-slot-valid"); ap.add_argument("--runtime-policy", default="strict")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--pretrained", default="timm"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools import exp095_teacher_mix as guard
    from tools import exp087_training as trace_tools
    protocol, ordered, inner = guard.validate_common(a, ROOT, "teacher")
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import LAB, Corpus, build_model, environment, load_label_csv, seed_all, sha256_text, write_predictions, sha256_file
    from raptor_data import StudyWindows, collate
    from raptor_runtime import configure_runtime
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    dev = a.device
    runtime = configure_runtime(a.amp, dev, policy=a.runtime_policy); eval_runtime = configure_runtime(a.eval_amp, dev, policy=a.runtime_policy)
    guard.worker_init(0); run_environment = environment(); seed_all(a.seed)
    out = EXPERIMENTS / a.tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); log_file.write(msg + "\n"); log_file.flush()

    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    if label_hash != protocol["label_hash"]:
        raise SystemExit("teacher must train on the original Dread labels")
    group_rows = guard.read_rows(ROOT / protocol["groups"]["path"])
    all_train, oof_ids_outer = guard.validate_membership(corpus.ids.astype(str).tolist(), ordered, group_rows)
    train_ids = [u for u in all_train if inner[u] != a.fold_index]
    held_ids = [u for u in all_train if inner[u] == a.fold_index]
    if not train_ids or not held_ids or set(train_ids) & set(held_ids) or set(train_ids) | set(held_ids) != set(all_train):
        raise SystemExit("inner split invalid")
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in held_ids}}
    log(f"{a.tag} | inner fold {a.fold_index} | train {len(train_ids)} | held-out {len(held_ids)} | V888 untouched | Gold disabled")
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pos_weight_np = np.clip((1 - prev) / prev, 1, 10).astype(np.float32)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)
    tds = StudyWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=True,
                    persistent_workers=a.workers > 0, generator=torch.Generator().manual_seed(a.seed), worker_init_fn=guard.worker_init)
    eval_bs = max(1, a.bs // 2)

    def loader(ids):
        return DataLoader(StudyWindows(corpus.dir, ids, all_labels, a.res, a.k_eval, False, a.norm, a.seed, window_policy=a.window_policy), batch_size=eval_bs,
                          shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False, worker_init_fn=guard.worker_init)

    model = build_model(a.arch, pretrained=a.pretrained == "timm").to(dev)
    ih = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        ih.update(key.encode()); ih.update(value.detach().cpu().contiguous().numpy().tobytes())
    provenance = {"code_sha256": {p: sha256_file(ROOT / "training" / "src" / p) for p in guard.CODE_NAMES}, "initial_state_sha256": ih.hexdigest(), "runtime": runtime, "eval_runtime": eval_runtime,
                  "train_ids_ordered_sha256": guard.uid_hash(train_ids), "held_ids_ordered_sha256": guard.uid_hash(held_ids), "label_hash": label_hash, "inner_fold": a.fold_index,
                  "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"), "checkpoint_selection": "fixed-final", "gold_access": "disabled", "v888_access": "none",
                  "actual_pos_weight": pos_weight_np.tolist(), "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)), "protocol_sha256": sha256_file(Path(a.protocol))}
    provenance["environment_matches_baseline"] = all(run_environment.get(k) == protocol["environment"][k] for k in guard.ENV_KEYS)
    provenance["environment"] = {k: run_environment.get(k) for k in guard.ENV_KEYS}
    if not provenance["environment_matches_baseline"] and not __import__("os").environ.get("KNEE_REMOTE_RELAX_ENV"):
        raise SystemExit("runtime environment differs from the registered baseline environment (set KNEE_REMOTE_RELAX_ENV=1 on a remote GPU to record instead)")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]; bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(math.ceil(len(tl) / a.accum) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = {"bf16": torch.bfloat16, "none": None}[a.amp]; eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]
    history, updates, t_start = [], 0, time.time()
    for ep in range(a.epochs):
        tds.set_epoch(ep); model.train(); tot, t0 = 0.0, time.time(); opt.zero_grad(set_to_none=True)
        for i, (x, y, uids) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(x)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite training logits")
                loss = lossf(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); updates += 1; tot += loss.item()
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "epoch_seconds": round(time.time() - t0), "optimizer_updates": updates}
        history.append(rec); log(f"ep{ep} loss {rec['loss']:.4f} | {rec['epoch_seconds']}s")
        json.dump(history, open(out / "history.json", "w"), indent=1)
    cleanup = trace_tools.stop_loader(tl)
    if updates != steps:
        raise RuntimeError("update count differs from schedule")
    torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB, "contract": corpus.contract,
                "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy, "epoch": a.epochs - 1, "tag": a.tag, "provenance": provenance}, out / "model_best.pt")
    model.eval(); P, U = [], []
    with torch.no_grad():
        for x, y, uids in loader(held_ids):
            with torch.autocast(dev, dtype=eval_dtype or torch.float32, enabled=eval_dtype is not None):
                logits = model(x.to(dev))
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite held-out logits")
            P.append(torch.sigmoid(logits.float()).cpu().numpy()); U += uids
    if U != held_ids:
        raise RuntimeError("held-out order changed")
    write_predictions(out / "oof_fold.csv", U, np.concatenate(P))
    summary = {"backbone": a.arch, "inner_fold": a.fold_index, "train": len(train_ids), "held_out": len(held_ids), "epochs": a.epochs, "optimizer_updates": updates, "scheduled_steps": steps,
               "provenance": provenance, "history": history, "worker_cleanup": cleanup, "environment": environment(), "total_seconds": round(time.time() - t_start), "metric_access": "none (teacher)"}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
