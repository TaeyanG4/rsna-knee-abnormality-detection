"""EXP107: one fold of the RadImageNet ResNet50 5-fold MIL family (tools/exp107_kfold_radr50.py registers the protocol; H-B-24, Kaggle T4).

A copy of training/runners/exp101_kfold_arch.py (the EXP101 ResNet34 RN_clip runner) whose training loop, loss, optimizer, scheduler, FP16
skip gate, evaluation and output files are unchanged (tests/test_exp107_kfold_radr50.py compares the loop text). The only differences, all
marked `EXP107:` below:
  * model: timm resnet50 inside the same RaptorClassifier, backbone loaded strictly from the RadImageNet checkpoint --init-weights (sha256
    registered in the protocol; training/src/raptor_radimagenet.py) instead of timm ImageNet weights (--pretrained radimagenet);
  * input: --input-policy center3 = the centre cache plane repeated as the three channels (same within-slot-valid window centres and
    sampling streams as EXP101), --norm radimagenet = 2x - 1 (the anchor's uint8 / 127.5 - 1);
  * initial state: the registered seed-42 hash is enforced locally; under KNEE_REMOTE_RELAX_ENV only head keys may differ - a backbone key
    that differs from the registered RadImageNet bytes always stops the run.

Modes: full (registered per-fold contract) and smoke (the first N training / M validation studies of a fold, a few updates, CPU allowed,
written only to an --output-root outside artifacts/). --validate-only checks protocol, membership, pos_weight, loader contract, the weight
file and the seed-42 initial state on CPU, then exits without creating output.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--mode", choices=["full", "smoke"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--output-root", default=None, help="parent of the <tag> run directory (default artifacts/experiments)")
    ap.add_argument("--smoke-train", type=int, default=24)
    ap.add_argument("--smoke-val", type=int, default=4)
    ap.add_argument("--smoke-epochs", type=int, default=1)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--eval-labels", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--k-eval", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--fixed-epoch", type=int, required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--accum", type=int, required=True)
    ap.add_argument("--bb-lr", type=float, required=True)
    ap.add_argument("--head-lr", type=float, required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], required=True)
    ap.add_argument("--amp-init-scale", type=float, required=True)
    ap.add_argument("--eval-amp", choices=["fp16", "bf16", "none"], required=True)
    ap.add_argument("--norm", choices=["radimagenet"], required=True)  # EXP107: 2x - 1
    ap.add_argument("--input-policy", choices=["center3"], required=True)  # EXP107: centre plane x3
    ap.add_argument("--window-policy", choices=["within-slot-valid"], required=True)
    ap.add_argument("--runtime-policy", choices=["strict", "throughput"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pretrained", choices=["radimagenet"], required=True)  # EXP107: RadImageNet checkpoint
    ap.add_argument("--init-weights", required=True, help="RadImageNet ResNet50.pt (sha256 checked against the protocol)")  # EXP107
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from tools import exp107_kfold_radr50 as guard  # EXP107: own guard
    from tools import exp087_training as trace_tools
    protocol, fold, ordered, out_root = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, environment, load_label_csv, seed_all, write_predictions, sha256_file)
    from raptor_data import collate
    from raptor_radimagenet import CenterSliceWindows, build_radimagenet_model  # EXP107
    from raptor_runtime import configure_runtime
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    smoke = a.mode == "smoke"
    relaxed = bool(os.environ.get("KNEE_REMOTE_RELAX_ENV"))
    dev = a.device
    weights_sha = protocol["init_weights"]["sha256"]  # EXP107
    # ---- data/label/membership contract (pure CPU, before any runtime or CUDA call)
    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    eval_labels, eval_label_hash = load_label_csv(Path(a.eval_labels))
    if label_hash != protocol["label_set"]["semantic_sha256"] or eval_label_hash != protocol["label_set"]["semantic_sha256"]:
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
    computed_pos_weight = guard.pos_weight(labels, train_ids)
    pos_weight_np = np.asarray(fold["pos_weight"], dtype=np.float32)
    if not np.array_equal(computed_pos_weight, pos_weight_np):
        raise SystemExit("pos_weight differs from the manifest fold vector")
    full_contract = guard.fold_contract(len(train_ids), protocol["common_args"])
    if any(full_contract[k] != fold[k] for k in full_contract):
        raise RuntimeError("Fold batch/update/evaluation contract differs from the manifest")
    if smoke:  # the registered recipe and pos_weight, on a few studies of this fold
        train_ids, oof_ids = train_ids[:a.smoke_train], oof_ids[:a.smoke_val]
    epochs = a.smoke_epochs if smoke else a.epochs
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}
    tds = CenterSliceWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy,
                             input_policy=a.input_policy)  # EXP107
    shuffle_generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=dev == "cuda",
                    persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)
    eval_bs = max(1, a.bs // 2)
    steps = max(math.ceil(len(tl) / a.accum) * epochs, 1)
    if not smoke and (len(tl) != fold["len_train_loader"] or steps != fold["steps"] or steps != fold["scheduler_total_steps"] or eval_bs != fold["eval_batch"]):
        raise RuntimeError("Fold batch/update/evaluation contract differs from the manifest")
    max_skips = steps // 100 if smoke else fold["max_fp16_skips"]
    if not smoke and max_skips != steps // 100:
        raise RuntimeError("FP16 skip gate differs from 1% of the fold's steps")
    k_extra = int(protocol["extra_eval"]["k_eval"])
    registered_keys = guard.read(ROOT / protocol["initial_state"]["keys_file"])["keys"]

    def init_record(model):
        combined, keys = guard.state_hashes(model)
        mismatched = guard.mismatched_keys(keys, registered_keys)
        backbone_mismatched = [k for k in mismatched if k.startswith("backbone.")]  # EXP107: the checkpoint fixes every backbone byte
        if backbone_mismatched:
            raise SystemExit(f"RadImageNet backbone differs from the registered state ({len(backbone_mismatched)} keys: {backbone_mismatched[:5]})")
        return combined, combined == protocol["initial_state_sha256"], mismatched

    def make_model():  # EXP107: RadImageNet backbone, strict load, file sha256 checked
        return build_radimagenet_model(a.arch, Path(a.init_weights), weights_sha)

    if a.validate_only:
        # CPU only: no runtime configuration, no .to(dev), no CUDA query (hiding the GPU via CUDA_VISIBLE_DEVICES crashes cv2 on this host).
        guard.worker_init(0)
        seed_all(a.seed)
        built, load_report = make_model()
        init, matches, mismatched = init_record(built)
        if torch.cuda.is_initialized():
            raise SystemExit("--validate-only touched CUDA")
        if a.seed == protocol["seed"] and not matches and not relaxed:
            raise SystemExit(f"generic initial state differs from the registered {a.arch} seed-{a.seed} state ({len(mismatched)} keys)")
        print(json.dumps(dict(status="VALIDATED_NO_TRAINING", tag=a.tag, fold=a.fold, mode=a.mode, arch=a.arch, train=len(train_ids), val=len(oof_ids),
                              len_train_loader=len(tl), steps=steps, epochs=epochs, max_fp16_skips=max_skips, eval_batch=eval_bs, k_eval=a.k_eval,
                              k_eval_extra=k_extra, pos_weight=[round(float(x), 6) for x in pos_weight_np], label_semantic_sha256=label_hash,
                              initial_state_sha256=init, initial_state_matches_registered=matches, initial_state_mismatched_keys=mismatched[:20],
                              init_weights=load_report, input_policy=a.input_policy, norm=a.norm,
                              output_root=str(out_root), output_exists=(out_root / a.tag).exists(), remote_relaxed=relaxed), indent=1))
        return

    # ---- runtime and run directory (same order as EXP094/EXP099: runtime -> worker threads -> environment -> seed)
    runtime = configure_runtime(a.amp, dev, policy=a.runtime_policy)
    eval_runtime = configure_runtime(a.eval_amp, dev, policy=a.runtime_policy)
    guard.worker_init(0)

    def environment_snapshot():
        if dev == "cuda":
            return environment()
        cuda_available = torch.cuda.is_available  # CPU smoke: record versions without creating a CUDA context on a busy local GPU
        torch.cuda.is_available = lambda: False
        try:
            return environment()
        finally:
            torch.cuda.is_available = cuda_available

    run_environment = environment_snapshot()
    seed_all(a.seed)
    out = out_root / a.tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); log_file.write(msg + "\n"); log_file.flush()

    log(f"{a.tag} | {a.arch} RadImageNet | {a.input_policy} {a.norm} | fold {a.fold} seed {a.seed} mode {a.mode} | train {len(train_ids)} | val {len(oof_ids)} | "
        f"steps {steps} | Gold and metrics disabled")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)

    def loader(ids, k_eval):
        return DataLoader(CenterSliceWindows(corpus.dir, ids, all_labels, a.res, k_eval, False, a.norm, a.seed, window_policy=a.window_policy,
                                             input_policy=a.input_policy), batch_size=eval_bs,  # EXP107
                          shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False, worker_init_fn=guard.worker_init)

    built, load_report = make_model()
    model = built.to(dev)
    init, init_matches, init_mismatched = init_record(model)
    provenance = {"code_sha256": {p: sha256_file(ROOT / "training" / "src" / p) for p in guard.CODE_NAMES}, "initial_state_sha256": init,
                  "initial_state_matches_registered": init_matches, "initial_state_mismatched_keys": init_mismatched,
                  "runtime": runtime, "eval_runtime": eval_runtime, "train_ids_ordered_sha256": guard.uid_hash(train_ids), "eval_ids_ordered_sha256": guard.uid_hash(oof_ids),
                  "eval_label_hash": eval_label_hash, "label_hash": label_hash, "label_semantic_sha256": label_hash, "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"),
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": "fixed-final", "gold_access": "disabled", "loss_policy": "clipped_positive_term",
                  "actual_pos_weight": pos_weight_np.tolist(), "computed_pos_weight": computed_pos_weight.tolist(),
                  "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)),
                  "protocol_sha256": sha256_file(Path(a.protocol)), "fold": a.fold, "label_set": guard.LABEL_SET, "label_file": protocol["label_set"]["path"],
                  "label_file_sha256": protocol["label_set"]["sha256"], "arch": a.arch, "seed": a.seed, "training_mode": a.mode,
                  "len_train_loader": len(tl), "scheduler_total_steps": steps, "max_fp16_skips": max_skips, "k_eval": a.k_eval, "k_eval_extra": k_extra,
                  "metric_access": "deferred_single_fixed_evaluation", "requested_workers": a.workers, "output_root": str(out_root), "remote_relaxed": relaxed,
                  "init_weights": load_report, "radimagenet_module_sha256": sha256_file(ROOT / "training" / "src" / "raptor_radimagenet.py"),  # EXP107
                  "input_policy": a.input_policy, "norm": a.norm}
    if smoke:
        provenance["smoke"] = dict(train=len(train_ids), val=len(oof_ids), epochs=epochs, note="engineering smoke only; not a result")
    if a.seed == protocol["seed"] and not init_matches and not relaxed:
        raise SystemExit(f"generic initial state differs from the registered {a.arch} seed-{a.seed} state ({len(init_mismatched)} keys)")
    provenance["environment_matches_baseline"] = all(run_environment.get(k) == protocol["environment"][k] for k in guard.ENV_KEYS)
    provenance["environment"] = {k: run_environment.get(k) for k in guard.ENV_KEYS}
    if not provenance["environment_matches_baseline"] and not relaxed and not smoke:
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
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]
    scaler = torch.amp.GradScaler("cuda", init_scale=a.amp_init_scale, enabled=a.amp == "fp16" and dev == "cuda")
    skipped = 0

    def autocast(dtype):
        return torch.autocast(dev, dtype=dtype) if dtype is not None and dev == "cuda" else contextlib.nullcontext()

    @torch.no_grad()
    def evaluate(ids, k_eval):
        model.eval(); P, U = [], []
        for x, y, uids in loader(ids, k_eval):
            with autocast(eval_dtype):
                logits = model(x.to(dev))
            if logits.shape != (len(uids), len(LAB)) or not torch.isfinite(logits).all():
                raise RuntimeError(f"invalid evaluation logits: {uids}")
            P.append(torch.sigmoid(logits.float()).cpu().numpy()); U += uids
        return np.concatenate(P), U

    history, best, optimizer_updates = [], None, 0
    t_start = time.time()
    for ep in range(epochs):
        tds.set_epoch(ep); model.train(); tot, t0 = 0.0, time.time(); opt.zero_grad(set_to_none=True)
        for i, (x, y, batch_uids) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with autocast(amp_dtype):
                logits = model(x)
                if logits.shape != (len(batch_uids), 12) or not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite or invalid training logits")
                loss = lossf(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at epoch {ep} step {i}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=a.amp != "fp16")
            scale_before = scaler.get_scale()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if a.amp == "fp16" and dev == "cuda" and scaler.get_scale() < scale_before:
                skipped += 1
                if skipped > max_skips:
                    raise RuntimeError(f"FP16 optimizer skips exceed 1% of {steps} ({skipped})")
            else:
                sched.step()
            optimizer_updates += 1; tot += loss.item()
            if (i + 1) % 200 == 0 or smoke:
                rate = (time.time() - t0) / (i + 1)
                log(f"  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f} | {rate:.2f}s/step, epoch ETA {rate * (len(tl) - i - 1) / 60:.0f} min")
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "epoch_seconds": round(time.time() - t0), "peak_gpu_gib": round(peak, 2),
               "optimizer_updates": optimizer_updates, "fp16_skips": skipped}
        history.append(rec); log(f"ep{ep} loss {rec['loss']:.4f} | fixed-final | {rec['epoch_seconds']}s | peak {peak:.1f} GiB | skips {skipped}")
        if ep == epochs - 1:
            best = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB, "contract": corpus.contract,
                    "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy, "epoch": ep, "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp,
                    "fold": a.fold, "label_set": guard.LABEL_SET, "provenance": provenance}
            best.update(input_policy=a.input_policy, init="radimagenet", init_weights_sha256=weights_sha)  # EXP107: serving must rebuild center3 + 2x-1
            torch.save(best, out / "model_best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    cleanup = trace_tools.stop_loader(tl)
    if optimizer_updates != steps:
        raise RuntimeError("full exact update gate failed before inference")
    provenance["fp16_skipped_updates"] = skipped
    if dev == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU run initialised CUDA")
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
    provenance["cuda_initialized"] = bool(torch.cuda.is_initialized())
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    summary = {"contract": corpus.contract, "backbone": a.arch, "img": a.res, "max_epochs": epochs, "train": len(train_ids), "val": len(oof_ids), "gold": 0, "rows": len(oof_ids),
               "weak_oof_macro_auc": None, "selected_checkpoint": "fixed-final", "gold_access": "disabled", "loss_policy": "clipped_positive_term", "provenance": provenance,
               "selected_epoch_1based": best["epoch"] + 1, "optimizer_updates": optimizer_updates, "skipped_updates": skipped, "args": vars(a), "history": history,
               "environment": environment_snapshot(), "worker_cleanup": cleanup, "total_seconds": round(time.time() - t_start), "metric_access": "deferred_single_fixed_evaluation",
               "fold": a.fold, "label_set": guard.LABEL_SET, "mode": a.mode, "oof_files": {"k24": "oof_best.csv", f"k{k_extra}": protocol["extra_eval"]["file"]},
               "caveat": "weak-label fold OOF for ensembling / teacher pseudo-labels; one seed; remote runs are independent trajectories"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", protocol["extra_eval"]["file"]]
    (out / "output-inventory.json").write_text(json.dumps({n: dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size) for n in names}, indent=1), encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
