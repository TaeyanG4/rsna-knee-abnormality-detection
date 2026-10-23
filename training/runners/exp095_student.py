"""EXP095 student: the unchanged G1 B8 recipe trained on the mixed labels (T only), evaluated on V888 with fixed Dread labels.

Identical to the EXP094 'seed' arm (seed 42) except the training label file, whose SHA256 must match mix-receipt.json.
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
    ap.add_argument("--protocol", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--corpus", default=str(CACHE / "RAPTOR64_336_C140_S02")); ap.add_argument("--arch", required=True); ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--k-eval", type=int, default=24); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--fixed-epoch", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--accum", type=int, default=1); ap.add_argument("--bb-lr", type=float, default=3e-5); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "none"]); ap.add_argument("--eval-amp", choices=["fp16", "bf16", "none"], default="fp16")
    ap.add_argument("--norm", default="imagenet"); ap.add_argument("--labels", required=True); ap.add_argument("--eval-labels", required=True)
    ap.add_argument("--window-policy", choices=["within-slot-valid"], default="within-slot-valid"); ap.add_argument("--runtime-policy", default="strict")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--pretrained", default="timm"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools import exp095_teacher_mix as guard
    from tools import exp087_training as trace_tools
    protocol, ordered, inner = guard.validate_common(a, ROOT, "student")
    receipt = guard.read(ROOT / guard.PROTOCOL / "mix-receipt.json")
    if Path(a.labels).resolve() != (ROOT / receipt["path"]).resolve() or guard.sha256(Path(a.labels)) != receipt["sha256"]:
        raise SystemExit("student training labels must be the receipted mixed_labels.csv")
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import LAB, Corpus, build_model, environment, load_label_csv, seed_all, write_predictions, sha256_file
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
    labels, label_hash = load_label_csv(Path(a.labels)); eval_labels, eval_label_hash = load_label_csv(Path(a.eval_labels))
    if eval_label_hash != protocol["eval_label_hash"]:
        raise SystemExit("fixed evaluation labels changed")
    group_rows = guard.read_rows(ROOT / protocol["groups"]["path"])
    train_ids, oof_ids = guard.validate_membership(corpus.ids.astype(str).tolist(), ordered, group_rows)
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}
    log(f"{a.tag} | student on mixed labels | train {len(train_ids)} | val {len(oof_ids)} | Gold and metrics disabled")
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
                  "train_ids_ordered_sha256": guard.uid_hash(train_ids), "eval_ids_ordered_sha256": guard.uid_hash(oof_ids), "training_label_hash": label_hash, "training_labels_sha256": receipt["sha256"],
                  "eval_label_hash": eval_label_hash, "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"), "checkpoint_selection": "fixed-final", "gold_access": "disabled",
                  "actual_pos_weight": pos_weight_np.tolist(), "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)), "protocol_sha256": sha256_file(Path(a.protocol)),
                  "scheduler_total_steps": 3448, "metric_access": "deferred_single_fixed_evaluation"}
    if provenance["initial_state_sha256"] != protocol["student_initial_state_sha256"]:
        raise SystemExit("generic initial state differs from EXP084 B")
    provenance["environment_matches_baseline"] = all(run_environment.get(k) == protocol["environment"][k] for k in guard.ENV_KEYS)
    provenance["environment"] = {k: run_environment.get(k) for k in guard.ENV_KEYS}
    if not provenance["environment_matches_baseline"] and not __import__("os").environ.get("KNEE_REMOTE_RELAX_ENV"):
        raise SystemExit("runtime environment differs from the registered baseline environment (set KNEE_REMOTE_RELAX_ENV=1 on a remote GPU to record instead)")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]; bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(math.ceil(len(tl) / a.accum) * a.epochs, 1)
    if len(tl) != 431 or steps != 3448 or eval_bs != 4:
        raise RuntimeError("B8 batch/update/evaluation contract differs")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = {"bf16": torch.bfloat16, "none": None}[a.amp]; eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]
    history, updates, best, t_start = [], 0, None, time.time()
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
            if (i + 1) % 200 == 0:
                log(f"  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f}")
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "epoch_seconds": round(time.time() - t0), "optimizer_updates": updates}
        history.append(rec); log(f"ep{ep} loss {rec['loss']:.4f} | fixed-final | {rec['epoch_seconds']}s")
        if ep == a.epochs - 1:
            best = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB, "contract": corpus.contract, "k_eval": a.k_eval,
                    "norm": a.norm, "window_policy": a.window_policy, "epoch": ep, "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp, "provenance": provenance}
            torch.save(best, out / "model_best.pt")
        json.dump(history, open(out / "history.json", "w"), indent=1)
    cleanup = trace_tools.stop_loader(tl)
    if updates != 3448:
        raise RuntimeError("full exact update gate failed before inference")
    model.load_state_dict(best["model"]); model.eval(); P, U = [], []
    with torch.no_grad():
        for x, y, uids in loader(oof_ids):
            with torch.autocast(dev, dtype=eval_dtype or torch.float32, enabled=eval_dtype is not None):
                logits = model(x.to(dev))
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite evaluation logits")
            P.append(torch.sigmoid(logits.float()).cpu().numpy()); U += uids
    if U != oof_ids:
        raise RuntimeError("Final prediction UID order changed")
    write_predictions(out / "oof_best.csv", U, np.concatenate(P))
    summary = {"backbone": a.arch, "train": len(train_ids), "val": len(oof_ids), "epochs": a.epochs, "optimizer_updates": updates, "skipped_updates": 0, "provenance": provenance, "history": history,
               "worker_cleanup": cleanup, "environment": environment(), "total_seconds": round(time.time() - t_start), "metric_access": "deferred_single_fixed_evaluation", "arm": "student",
               "caveat": "exposed weak G1 development set; one seed; not D01/private evidence"}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
