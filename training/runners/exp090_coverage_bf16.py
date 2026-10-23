"""EXP090: EXP087 coverage retry under BF16 with a matched BF16 control (H-B-01).

Derived from training/runners/exp087_coverage_b8.py (frozen, unchanged). Differences:
  * --amp defaults to bf16 (no GradScaler path); fp16 is refused.
  * --arm candidate|control: control trains the identical B8 recipe with the coverage
    adapter disabled (enabled=False), so numerics repair and coverage are separated.
  * Guard hashes bind to the EXP090 protocol (tools/exp090_coverage_bf16.py), not EXP087.
Everything else (G1 split, labels, seed, fixed-final epoch 8, 3448 exact updates, Gold
blocker, no metric access) is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "storage" / "cache" / "derived"
EXPERIMENTS = ROOT / "artifacts" / "experiments"
DEFAULT_LABELS = ROOT / "storage" / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--arm", choices=["candidate", "control"], required=True)
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--tag", required=True)
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
    ap.add_argument("--amp", default="bf16", choices=["bf16", "none"])
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
    from tools import exp090_coverage_bf16 as guard
    from tools import exp087_training as trace_tools
    protocol, ordered, arm = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, build_model, environment, load_label_csv, seed_all, sha256_text,
                               write_predictions, sha256_file)
    from raptor_data import StudyWindows, collate
    from exp087_coverage_data import CoverageStore, CoverageStudyWindows
    from raptor_runtime import configure_runtime
    if a.fixed_epoch != a.epochs:
        raise SystemExit("--fixed-epoch must equal --epochs")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    runtime = configure_runtime(a.amp, dev, policy=a.runtime_policy)
    eval_runtime = configure_runtime(a.eval_amp, dev, policy=a.runtime_policy)
    guard.worker_init(0)
    run_environment = environment()
    coverage_store = CoverageStore(ROOT / protocol["coverage_manifest"], protocol["coverage_manifest_sha256"])
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
    if any(u not in labels for u in train_ids) or any(u not in eval_labels for u in oof_ids):
        raise SystemExit("Frozen label coverage changed")
    if label_hash != protocol["label_hash"] or eval_label_hash != protocol["eval_label_hash"]:
        raise SystemExit("Label semantic hash mismatch")
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}
    enabled = a.arm == "candidate"
    log(f"{a.tag} | arm {a.arm} coverage {'on' if enabled else 'off'} | train {len(train_ids)} | val {len(oof_ids)} | Gold and metrics disabled")
    log(f"arch {a.arch} res {a.res} k {a.k}/{a.k_eval} bs {a.bs}x{a.accum} epochs {a.epochs} amp {a.amp} | {dev}")
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pos_weight_np = np.clip((1 - prev) / prev, 1, 10).astype(np.float32)
    if not np.array_equal(pos_weight_np, np.asarray(protocol["loss"]["baseline_pos_weight"], dtype=np.float32)):
        raise SystemExit("T-only loss vector mismatch")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)

    tds = CoverageStudyWindows(StudyWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy),
                               coverage_store, enabled=enabled)
    shuffle_generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate,
                    pin_memory=dev == "cuda", persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)
    eval_bs = max(1, a.bs // 2)

    def loader(ids):
        return DataLoader(CoverageStudyWindows(StudyWindows(corpus.dir, ids, all_labels, a.res, a.k_eval, False, a.norm, a.seed, window_policy=a.window_policy),
                                               coverage_store, enabled=enabled), batch_size=eval_bs, shuffle=False, num_workers=min(a.workers, 2),
                          collate_fn=collate, persistent_workers=False, worker_init_fn=guard.worker_init)

    model = build_model(a.arch, pretrained=a.pretrained == "timm").to(dev)
    initial_hash = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        initial_hash.update(key.encode()); initial_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    source_files = [ROOT / "training" / "src" / name for name in guard.CODE_NAMES]
    provenance = {"code_sha256": {p.name: sha256_file(p) for p in source_files},
                  "initial_state_sha256": initial_hash.hexdigest(), "runtime": runtime, "eval_runtime": eval_runtime,
                  "train_ids_sha256": sha256_text("\n".join(sorted(train_ids))), "eval_ids_sha256": sha256_text("\n".join(sorted(oof_ids))),
                  "train_ids_ordered_sha256": guard.uid_hash(train_ids), "eval_ids_ordered_sha256": guard.uid_hash(oof_ids),
                  "eval_label_hash": eval_label_hash, "label_hash": label_hash,
                  "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"),
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": "fixed-final", "gold_access": "disabled",
                  "loss_policy": "clipped_positive_term", "actual_pos_weight": pos_weight_np.tolist(),
                  "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)),
                  "protocol_sha256": sha256_file(Path(a.protocol)), "arm": a.arm, "coverage_enabled": enabled,
                  "coverage_manifest_sha256": protocol["coverage_manifest_sha256"], "training_mode": a.mode,
                  "scheduler_total_steps": 3448, "metric_access": "deferred_single_fixed_evaluation",
                  "requested_workers": a.workers, "internal_cpu_threads": 1}
    if provenance["initial_state_sha256"] != protocol["initial_state_sha256"]:
        raise SystemExit("generic initial state differs from EXP084 B")
    if any(run_environment.get(k) != protocol["environment"][k] for k in guard.ENV_KEYS):
        raise SystemExit("runtime environment differs from the registered baseline environment")
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
    amp_dtype = {"bf16": torch.bfloat16, "none": None}[a.amp]
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]

    @torch.no_grad()
    def evaluate(ids):
        model.eval(); P, U = [], []
        for x, y, uids in loader(ids):
            with torch.autocast(dev, dtype=eval_dtype or torch.float32, enabled=eval_dtype is not None and dev == "cuda"):
                logits = model(x.to(dev))
            if logits.shape != (len(uids), len(LAB)) or not torch.isfinite(logits).all():
                raise RuntimeError(f"invalid evaluation logits: {uids}")
            o = torch.sigmoid(logits.float())
            P.append(o.cpu().numpy()); U += uids
        return np.concatenate(P), U

    history, best = [], None
    optimizer_updates = 0
    trace = trace_tools.new_trace()
    pilot_times = []
    t_start = time.time()
    whole_update_start = time.monotonic()
    for ep in range(a.epochs):
        tds.set_epoch(ep)
        model.train(); tot, t0 = 0.0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, (x, y, batch_uids) in enumerate(tl):
            if enabled:
                trace_tools.trace_batch(trace, batch_uids, ep, tds, x)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None and dev == "cuda"):
                logits = model(x)
                if logits.shape != (len(batch_uids), 12) or not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite or invalid training logits")
                loss = lossf(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at epoch {ep} step {i}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            optimizer_updates += 1
            tot += loss.item()
            trace["updates"] = optimizer_updates; trace["scheduler_steps"] = sched.last_epoch; trace["epochs"] = ep + 1
            if a.mode == "pilot":
                torch.cuda.synchronize()
                pilot_times.append(time.monotonic() - whole_update_start)
                if optimizer_updates == 32:
                    times = pilot_times[-24:]
                    projection = 1.3 * sum(times) / 24 * 3448 + 300
                    result = {k: v for k, v in trace.items() if k != "_uid_hasher"}
                    result["delivered_uid_sha256"] = trace["_uid_hasher"].hexdigest()
                    result["worker_cleanup"] = trace_tools.stop_loader(tl)
                    (out / "input-trace.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
                    receipt = dict(status="PILOT_COMPLETE", arm=a.arm, updates=32, skips=0, scheduler_horizon=3448,
                                   protocol_sha256=provenance["protocol_sha256"], last24_mean_seconds=sum(times) / 24,
                                   projected_full_seconds=projection, projection_pass=projection <= 3600,
                                   peak_cuda_bytes=torch.cuda.max_memory_allocated(), all_finite=True,
                                   coverage_checks=result.get("actual_replacement_tensor_checks", 0) if enabled else None)
                    (out / "pilot-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
                    log(f"pilot complete: projected full {projection:.0f}s pass={projection <= 3600}")
                    log_file.close()
                    return
            whole_update_start = time.monotonic()
            if (i + 1) % 200 == 0:
                rate = (time.time() - t0) / (i + 1)
                log(f"  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f} | {rate:.2f}s/step, epoch ETA {rate * (len(tl) - i - 1) / 60:.0f} min")
        train_s = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "train_seconds": round(train_s), "epoch_seconds": round(time.time() - t0),
               "peak_gpu_gib": round(peak, 2), "optimizer_updates": optimizer_updates}
        history.append(rec)
        log(f"ep{ep} loss {rec['loss']:.4f} | fixed-final | {rec['epoch_seconds']}s | peak {peak:.1f} GiB")
        if ep == a.epochs - 1:
            best = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB,
                    "contract": corpus.contract, "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy, "epoch": ep,
                    "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp, "provenance": provenance}
            torch.save(best, out / "model_best.pt")
        json.dump(history, open(out / "history.json", "w"), indent=1)

    result = {k: v for k, v in trace.items() if k != "_uid_hasher"}
    result["delivered_uid_sha256"] = trace["_uid_hasher"].hexdigest()
    result["worker_cleanup"] = trace_tools.stop_loader(tl)
    (out / "input-trace.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if optimizer_updates != 3448:
        raise RuntimeError("full exact update gate failed before inference")
    model.load_state_dict(best["model"])
    P, U = evaluate(oof_ids)
    if U != oof_ids:
        raise RuntimeError("Final prediction UID order changed")
    write_predictions(out / "oof_best.csv", U, P)
    summary = {"contract": corpus.contract, "backbone": a.arch, "img": a.res, "max_epochs": a.epochs, "best_epoch": best["epoch"],
               "train": len(train_ids), "val": len(oof_ids), "gold": 0, "rows": len(oof_ids), "weak_oof_macro_auc": None,
               "selected_checkpoint": "best", "gold_access": "disabled", "loss_policy": "clipped_positive_term",
               "no_gold_epoch_selection": True, "eval_label_hash": eval_label_hash, "window_policy": a.window_policy,
               "provenance": provenance, "selected_epoch_1based": best["epoch"] + 1, "optimizer_updates": optimizer_updates,
               "skipped_updates": 0, "args": vars(a), "label_hash": label_hash, "history": history, "environment": environment(),
               "total_seconds": round(time.time() - t_start), "metric_access": "deferred_single_fixed_evaluation", "arm": a.arm,
               "caveat": "exposed weak G1 development set; one seed; not D01/private evidence"}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", "input-trace.json"]
    (out / "output-inventory.json").write_text(json.dumps({n: dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size) for n in names}, indent=1), encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}")
    log_file.close()


if __name__ == "__main__":
    main()
