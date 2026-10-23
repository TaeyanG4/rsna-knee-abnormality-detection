"""EXP087 unchanged-recipe numerical prefix; additive telemetry, no recovery."""
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
    ap.add_argument("--mode", choices=["diagnostic"], required=True)
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
    from tools import exp087_numerical_diagnostic_v1 as guard
    protocol, ordered = guard.validate_arguments(a, ROOT)
    arm = guard.arm_for(protocol, a.tag)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, build_model, environment,
                               load_label_csv, seed_all, sha256_text,
                               write_predictions, sha256_file)
    from raptor_data import StudyWindows, collate
    from exp087_coverage_data import CoverageStore, CoverageStudyWindows
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
    guard.worker_init(0)
    run_environment = environment()  # Imports happen before the original seed boundary.
    coverage_store = CoverageStore(ROOT / protocol["coverage_manifest"], protocol["coverage_manifest_sha256"])
    seed_all(a.seed)
    out = EXPERIMENTS / a.tag
    allowed_supervisor = {"supervisor.stdout.log", "supervisor.stderr.log", "supervisor-job.json", "phase-authorization.json"}
    if out.exists() and any(p.name not in allowed_supervisor for p in out.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); log_file.write(msg + "\n"); log_file.flush()

    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    eval_labels, eval_label_hash = load_label_csv(Path(a.eval_labels))
    split_hash = guard.sha256(Path(a.fold_file))
    excluded = set(protocol["excluded_uids"])
    group_rows = guard.read_rows(ROOT / protocol["groups"]["path"])
    train_ids, oof_ids = guard.validate_membership(corpus.ids.astype(str).tolist(), ordered, group_rows)
    if any(u not in labels for u in train_ids) or any(u not in eval_labels for u in oof_ids):
        raise SystemExit("Frozen label coverage changed")
    # V labels are not used by training/inference; scoring belongs to the later evaluator.
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}
    guard.validate_label_contract(label_hash, eval_label_hash, split_hash, protocol)
    log(f"{a.tag} | train {len(train_ids)} | val {len(oof_ids)} | Gold and metrics disabled")
    log(f"arch {a.arch} res {a.res} k {a.k}/{a.k_eval} bs {a.bs}x{a.accum} epochs {a.epochs} amp {a.amp} select {a.select} | {dev}")
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    baseline_pos_weight = np.clip((1 - prev) / prev, 1, 10).astype(np.float32)
    pos_weight = torch.tensor(guard.loss_weights(baseline_pos_weight, protocol, arm), dtype=torch.float32, device=dev)

    tds = StudyWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
    tds = CoverageStudyWindows(tds, coverage_store)
    shuffle_generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate,
                    pin_memory=dev == "cuda", persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)
    eval_bs = max(1, a.bs // 2)

    def loader(ids):
        return DataLoader(CoverageStudyWindows(StudyWindows(corpus.dir, ids, all_labels, a.res, a.k_eval, False, a.norm, a.seed, window_policy=a.window_policy), coverage_store), batch_size=eval_bs,
                          shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False, worker_init_fn=guard.worker_init)

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
                      gold_access="disabled", loss_policy=arm["loss_policy"],
                      baseline_pos_weight=baseline_pos_weight.tolist(),
                      actual_pos_weight=pos_weight.detach().cpu().tolist())
    provenance.update(metric_access="deferred_single_fixed_evaluation", requested_workers=a.workers, effective_workers=a.workers, internal_cpu_threads=1)
    provenance.update(coverage_manifest_sha256=protocol["coverage_manifest_sha256"],
        coverage_adapter_sha256=sha256_file(ROOT / protocol["coverage_adapter"]),
        coverage_policy_sha256=sha256_file(ROOT / protocol["coverage_helper"]),
        stage_acceptance_bindings=protocol["acceptance_bindings"],training_mode=a.mode,scheduler_total_steps=3448)
    guard.validate_pretraining(provenance, run_environment, protocol, arm)
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
            raise RuntimeError(f"required baseline gradient checkpointing unavailable: {exc}") from exc
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    if not len(tl):
        raise SystemExit("Not enough studies for one batch")
    steps = max(__import__("math").ceil(len(tl) / a.accum) * a.epochs, 1)
    if len(tl) != 431 or steps != 3448 or eval_bs != 4:
        raise RuntimeError("B8 batch/update/evaluation contract differs")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]
    scaler = torch.amp.GradScaler("cuda", init_scale=a.amp_init_scale, enabled=a.amp == "fp16" and dev == "cuda")

    optimizer_updates, skipped_updates, consecutive_skips = 0, 0, 0
    trace = guard.new_trace()
    diagnostic_state = {}
    attempt = 0
    caught = None
    cleaned = False
    try:
        for ep in range(a.epochs):
            tds.set_epoch(ep)
            model.train(); tot, t0 = 0.0, time.time()
            opt.zero_grad(set_to_none=True)
            for i, (x, y, batch_uids) in enumerate(tl):
                attempt += 1
                guard.begin_attempt(out, diagnostic_state, attempt, optimizer_updates, ep, i, batch_uids,
                    bool(torch.isfinite(x).all()), scaler.get_scale())
                if not diagnostic_state["input_finite"]:
                    raise RuntimeError("nonfinite diagnostic training input; forward/update forbidden")
                guard.trace_batch(trace, batch_uids, ep, tds, x)
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast(dev, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None and dev == "cuda"):
                    group_size = min(a.accum, len(tl) - (i // a.accum) * a.accum)
                    logits = model(x)
                    logits_finite = bool(torch.isfinite(logits).all())
                    guard.current(out, diagnostic_state, "after_forward", logits_finite=logits_finite,
                        logits_shape=list(logits.shape))
                    if logits.shape != (len(batch_uids), 12) or not logits_finite:
                        raise RuntimeError("nonfinite or invalid training logits")
                    loss = lossf(logits.float(), y) / group_size
                loss_finite = bool(torch.isfinite(loss))
                guard.current(out, diagnostic_state, "before_backward", loss_finite=loss_finite,
                    loss=guard.safe_number(loss.item()))
                if not loss_finite:
                    raise RuntimeError(f"nonfinite training loss at epoch {ep} step {i}")
                scaler.scale(loss).backward()
                if (i + 1) % a.accum == 0 or i + 1 == len(tl):
                    guard.current(out, diagnostic_state, "before_unscale")
                    scaler.unscale_(opt)
                    guard.current(out, diagnostic_state, "before_clip")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
                    scale_before = scaler.get_scale()
                    guard.assert_no_found_inf(scaler, opt)
                    guard.current(out, diagnostic_state, "before_optimizer_step")
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    if scaler.get_scale() >= scale_before:
                        sched.step()
                        optimizer_updates += 1; consecutive_skips = 0
                    else:
                        skipped_updates += 1; consecutive_skips += 1
                        log(f"FP16 overflow: optimizer/scheduler step skipped ({skipped_updates} total)")
                        raise RuntimeError("Any optimizer skip violates exact B8 update contract; no retry")
                tot += loss.item() * group_size
                trace["updates"] = optimizer_updates; trace["scheduler_steps"] = sched.last_epoch
                trace["epochs"] = ep + 1
                guard.completed_attempt(out, diagnostic_state, optimizer_updates, sched.last_epoch, scaler.get_scale())
                if attempt == guard.MAX_ATTEMPTS:
                    guard.finish_diagnostic(out, diagnostic_state, trace, tl, guard.sha256(Path(a.protocol)))
                    cleaned = True
                    log("bounded200 prefix complete; no failure observed; not training completion")
                    return
        raise RuntimeError("unexpected diagnostic loop exhaustion")
    except BaseException as exc:
        caught = exc
        guard.capture_failure(out, diagnostic_state, exc, model, opt, scaler)
        raise
    finally:
        try:
            if not cleaned:
                cleanup = guard.stop_loader(tl)
                guard.durable_json(out / "worker-cleanup.json", cleanup)
        except BaseException as cleanup_exc:
            print("diagnostic worker cleanup failed: " + repr(cleanup_exc), file=sys.stderr, flush=True)
            try:
                guard.durable_json(out / "worker-cleanup-failure.json", {"clean": False, "error": repr(cleanup_exc)})
            except BaseException as receipt_exc:
                print("cleanup failure receipt error: " + repr(receipt_exc), file=sys.stderr, flush=True)
            if caught is None:
                raise
        finally:
            log_file.close()


if __name__ == "__main__":
    main()
