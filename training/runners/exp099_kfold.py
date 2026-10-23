"""EXP099: one fold of the 5-fold B8 seed42 recipe (tools/exp099_kfold.py registers folds and label sets).

Identical to EXP094 V4 arm seed42 (CoAtNet 224, within-slot-valid windows, clipped pos_weight, fixed-final epoch 8, FP16 GradScaler
init 1024, strict runtime, Gold blocker, no metric access) except that the cohort is fold K of EXP099_KFOLD_PROTOCOL_V1 and the
training labels are a registered label set. Every G1-specific constant of the EXP094 runner (431 batches, 3448 updates, scheduler
total, pos_weight, 1% FP16 skip gate) is replaced by the fold's frozen manifest values, recomputed here and required to match.
After training: oof_best.csv (K24 evaluation windows, as EXP094) and oof_k36.csv (same final model, 36 windows; D-B-012).
--validate-only checks protocol, label registration, membership, pos_weight, loader contract and the seed-42 initial state on CPU,
then exits without creating any output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--label-set", required=True)
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--validate-only", action="store_true")
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
    from tools import exp099_kfold as guard
    from tools import exp087_training as trace_tools
    protocol, fold, ordered, registration, registration_file = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, build_model, environment, load_label_csv, seed_all, write_predictions, sha256_file)
    from raptor_data import StudyWindows, collate
    from raptor_runtime import configure_runtime
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    # ---- data/label/membership contract (pure CPU, before any runtime or CUDA call)
    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    eval_labels, eval_label_hash = load_label_csv(Path(a.eval_labels))
    if label_hash != registration["semantic_sha256"] or eval_label_hash != protocol["label_sets"]["dread"]["semantic_sha256"]:
        raise SystemExit("Label semantic hash mismatch")
    if sha256_file(corpus.dir / "manifest.json") != protocol["corpus_manifest_sha256"]:
        raise SystemExit("Corpus manifest changed")
    group_rows = guard.read_rows(ROOT / fold["groups"]["path"])
    canonical = guard.canonical_order(guard.read(ROOT / protocol["paths"]["base_ordered_ids"]))
    train_ids, oof_ids = guard.validate_fold_membership(corpus.ids.astype(str).tolist(), ordered, group_rows, canonical)
    if len(train_ids) != fold["n_train"] or len(oof_ids) != fold["n_val"]:
        raise SystemExit("Fold size mismatch")
    if any(u not in labels for u in train_ids):
        raise SystemExit("Label set does not cover the fold training ids")
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}
    computed_pos_weight = guard.pos_weight(labels, train_ids)
    registered_pos_weight = np.asarray(registration["pos_weight"][str(a.fold)], dtype=np.float32)
    if registration["pos_weight_source"] == "labels" and not np.array_equal(computed_pos_weight, registered_pos_weight):
        raise SystemExit("pos_weight differs from the label-set registration")
    if a.label_set == "dread" and not np.array_equal(registered_pos_weight, np.asarray(fold["pos_weight"], dtype=np.float32)):
        raise SystemExit("pos_weight differs from the manifest fold vector")
    pos_weight_np = registered_pos_weight
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    tds = StudyWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
    shuffle_generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=dev == "cuda",
                    persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)
    eval_bs = max(1, a.bs // 2)
    steps = max(math.ceil(len(tl) / a.accum) * a.epochs, 1)
    if len(tl) != fold["len_train_loader"] or steps != fold["steps"] or steps != fold["scheduler_total_steps"] or eval_bs != fold["eval_batch"]:
        raise RuntimeError("Fold batch/update/evaluation contract differs from the manifest")
    max_skips = fold["max_fp16_skips"]
    if max_skips != steps // 100:
        raise RuntimeError("FP16 skip gate differs from 1% of the fold's steps")
    k_extra = int(protocol["extra_eval"]["k_eval"])

    def initial_state(model):
        h = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    if a.validate_only:
        # CPU only: no runtime configuration, no .to(dev), no CUDA query (hiding the GPU via CUDA_VISIBLE_DEVICES crashes cv2 on this host).
        guard.worker_init(0)
        seed_all(a.seed)
        init = initial_state(build_model(a.arch, pretrained=a.pretrained == "timm"))
        if torch.cuda.is_initialized():
            raise SystemExit("--validate-only touched CUDA")
        if a.seed == 42 and init != protocol["initial_state_sha256"]:
            raise SystemExit("generic initial state differs from EXP084 B")
        print(json.dumps(dict(status="VALIDATED_NO_TRAINING", tag=a.tag, fold=a.fold, label_set=a.label_set, train=len(train_ids), val=len(oof_ids),
                              len_train_loader=len(tl), steps=steps, max_fp16_skips=max_skips, eval_batch=eval_bs, k_eval=a.k_eval, k_eval_extra=k_extra,
                              pos_weight=[round(float(x), 6) for x in pos_weight_np], pos_weight_source=registration["pos_weight_source"],
                              label_semantic_sha256=label_hash, initial_state_sha256=init, initial_state_matches_B=init == protocol["initial_state_sha256"],
                              output_exists=(EXPERIMENTS / a.tag).exists()), indent=1))
        return

    # ---- runtime and run directory (same order as EXP094: runtime -> worker threads -> environment -> seed)
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

    log(f"{a.tag} | fold {a.fold} label set {a.label_set} seed {a.seed} | train {len(train_ids)} | val {len(oof_ids)} | steps {steps} | Gold and metrics disabled")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)

    def loader(ids, k_eval):
        return DataLoader(StudyWindows(corpus.dir, ids, all_labels, a.res, k_eval, False, a.norm, a.seed, window_policy=a.window_policy), batch_size=eval_bs,
                          shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False, worker_init_fn=guard.worker_init)

    model = build_model(a.arch, pretrained=a.pretrained == "timm").to(dev)
    provenance = {"code_sha256": {p: sha256_file(ROOT / "training" / "src" / p) for p in guard.CODE_NAMES}, "initial_state_sha256": initial_state(model),
                  "runtime": runtime, "eval_runtime": eval_runtime, "train_ids_ordered_sha256": guard.uid_hash(train_ids), "eval_ids_ordered_sha256": guard.uid_hash(oof_ids),
                  "eval_label_hash": eval_label_hash, "label_hash": label_hash, "label_semantic_sha256": label_hash, "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"),
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": "fixed-final", "gold_access": "disabled", "loss_policy": "clipped_positive_term",
                  "actual_pos_weight": pos_weight_np.tolist(), "computed_pos_weight": computed_pos_weight.tolist(), "pos_weight_source": registration["pos_weight_source"],
                  "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)),
                  "protocol_sha256": sha256_file(Path(a.protocol)), "fold": a.fold, "label_set": a.label_set, "label_file": registration["path"],
                  "label_file_sha256": registration["sha256"], "label_registration_sha256": sha256_file(registration_file), "variant": "seed", "seed": a.seed,
                  "training_mode": a.mode, "len_train_loader": len(tl), "scheduler_total_steps": steps, "max_fp16_skips": max_skips,
                  "k_eval": a.k_eval, "k_eval_extra": k_extra, "metric_access": "deferred_single_fixed_evaluation", "requested_workers": a.workers}
    provenance["initial_state_matches_B"] = provenance["initial_state_sha256"] == protocol["initial_state_sha256"]
    if a.seed == 42 and not provenance["initial_state_matches_B"] and not os.environ.get("KNEE_REMOTE_RELAX_ENV"):
        raise SystemExit("generic initial state differs from EXP084 B")
    provenance["environment_matches_baseline"] = all(run_environment.get(k) == protocol["environment"][k] for k in guard.ENV_KEYS)
    provenance["environment"] = {k: run_environment.get(k) for k in guard.ENV_KEYS}
    if not provenance["environment_matches_baseline"] and not os.environ.get("KNEE_REMOTE_RELAX_ENV"):
        raise SystemExit("runtime environment differs from the registered baseline environment (set KNEE_REMOTE_RELAX_ENV=1 on a remote GPU to record instead)")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, a.res, a.res, device=dev)); model.train()
    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(True); log("gradient checkpointing on")
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    scaler = torch.amp.GradScaler("cuda", init_scale=a.amp_init_scale, enabled=a.amp == "fp16" and dev == "cuda")
    skipped = 0
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]

    @torch.no_grad()
    def evaluate(ids, k_eval):
        model.eval(); P, U = [], []
        for x, y, uids in loader(ids, k_eval):
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
                if skipped > max_skips:
                    raise RuntimeError(f"FP16 optimizer skips exceed 1% of {steps} ({skipped})")
            else:
                sched.step()
            optimizer_updates += 1; tot += loss.item()
            if a.mode == "pilot":
                torch.cuda.synchronize(); pilot_times.append(time.monotonic() - whole_update_start)
                if optimizer_updates == 32:
                    times = pilot_times[-24:]; projection = 1.3 * sum(times) / 24 * steps + 300
                    trace_tools.stop_loader(tl)
                    (out / "pilot-receipt.json").write_text(json.dumps(dict(status="PILOT_COMPLETE", fold=a.fold, label_set=a.label_set, updates=32,
                                                                           last24_mean_seconds=sum(times) / 24, projected_full_seconds=projection, steps=steps,
                                                                           peak_cuda_bytes=torch.cuda.max_memory_allocated(), all_finite=True), indent=1), encoding="utf-8")
                    log(f"pilot complete: projected full {projection:.0f}s (training only; K24+K36 evaluation extra)"); log_file.close(); return
            whole_update_start = time.monotonic()
            if (i + 1) % 200 == 0:
                rate = (time.time() - t0) / (i + 1)
                log(f"  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f} | {rate:.2f}s/step, epoch ETA {rate * (len(tl) - i - 1) / 60:.0f} min")
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "epoch_seconds": round(time.time() - t0), "peak_gpu_gib": round(peak, 2), "optimizer_updates": optimizer_updates}
        history.append(rec); log(f"ep{ep} loss {rec['loss']:.4f} | fixed-final | {rec['epoch_seconds']}s | peak {peak:.1f} GiB")
        if ep == a.epochs - 1:
            best = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB, "contract": corpus.contract,
                    "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy, "epoch": ep, "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp,
                    "fold": a.fold, "label_set": a.label_set, "provenance": provenance}
            torch.save(best, out / "model_best.pt")
        json.dump(history, open(out / "history.json", "w"), indent=1)
    cleanup = trace_tools.stop_loader(tl)
    if optimizer_updates != steps:
        raise RuntimeError("full exact update gate failed before inference")
    provenance["fp16_skipped_updates"] = skipped
    model.load_state_dict(best["model"])
    t_eval = time.time()
    P, U = evaluate(oof_ids, a.k_eval)
    if U != oof_ids:
        raise RuntimeError("Final prediction UID order changed")
    write_predictions(out / "oof_best.csv", U, P)
    log(f"K{a.k_eval} OOF written ({len(U)} rows, {time.time() - t_eval:.0f}s)")
    t_eval = time.time()
    P36, U36 = evaluate(oof_ids, k_extra)
    if U36 != oof_ids:
        raise RuntimeError("K36 prediction UID order changed")
    write_predictions(out / protocol["extra_eval"]["file"], U36, P36)
    log(f"K{k_extra} OOF written ({len(U36)} rows, {time.time() - t_eval:.0f}s)")
    summary = {"contract": corpus.contract, "backbone": a.arch, "img": a.res, "max_epochs": a.epochs, "train": len(train_ids), "val": len(oof_ids), "gold": 0, "rows": len(oof_ids),
               "weak_oof_macro_auc": None, "selected_checkpoint": "best", "gold_access": "disabled", "loss_policy": "clipped_positive_term", "provenance": provenance,
               "selected_epoch_1based": best["epoch"] + 1, "optimizer_updates": optimizer_updates, "skipped_updates": skipped, "args": vars(a), "history": history,
               "environment": environment(), "worker_cleanup": cleanup, "total_seconds": round(time.time() - t_start), "metric_access": "deferred_single_fixed_evaluation",
               "fold": a.fold, "label_set": a.label_set, "oof_files": {"k24": "oof_best.csv", f"k{k_extra}": protocol["extra_eval"]["file"]},
               "caveat": "weak-label fold OOF for teacher pseudo-labels; one seed; not D01/private evidence"}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", protocol["extra_eval"]["file"]]
    (out / "output-inventory.json").write_text(json.dumps({n: dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size) for n in names}, indent=1), encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
