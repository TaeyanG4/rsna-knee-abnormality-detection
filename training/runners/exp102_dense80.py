"""EXP102 (H-B-22): B8 seed-42 G1 fold-0 fit on the 80-slice cache with depth-scaled windows (tools/exp102_dense80.py registers it).

The training loop is the EXP099 runner's (itself the EXP094 V4 seed42 loop: CoAtNet-2 224, within-slot-valid windows, clipped pos_weight,
fixed-final epoch 8, FP16 GradScaler init 1024, strict runtime, Gold blocker, no metric access). Changed: corpus RAPTOR80_336_C140_S02,
K train 16, K eval 31. After training the same final model writes oof_best.csv (K31), oof_k24.csv (K24) and oof_allpos.csv (every
within-slot-valid centre once, batch 1).

Modes: full (registered contract, artifacts/experiments/<tag>), smoke (first N train / M val studies, CPU allowed, --output-root outside
artifacts/), baseline-eval (inference only: the EXP094L_B8_SEED42 checkpoint on the 64-slice cache at K24 - checked against its stored
oof_best.csv - and at all positions) and baseline-eval-smoke. --validate-only checks protocol, bindings, caches, membership, pos_weight,
loader contract and the seed-42 initial state (or the baseline checkpoint) on CPU, then exits without creating output.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--mode", choices=["full", "smoke", "baseline-eval", "baseline-eval-smoke"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--output-root", default=None, help="parent of the <tag> run directory (default artifacts/experiments)")
    ap.add_argument("--smoke-train", type=int, default=24)
    ap.add_argument("--smoke-val", type=int, default=3)
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
    ap.add_argument("--norm", choices=["imagenet", "none"], required=True)
    ap.add_argument("--window-policy", choices=["within-slot-valid"], required=True)
    ap.add_argument("--runtime-policy", choices=["strict", "throughput"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pretrained", choices=["timm", "none"], required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from tools import exp102_dense80 as guard
    from tools import exp087_training as trace_tools
    protocol, ordered, out_root = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    import raptor_contracts_ext  # noqa: F401  (registers raptor80_336_c140_s02; Corpus itself reads the slot table from the cache manifest)
    from raptor_common import (LAB, Corpus, build_model, environment, load_label_csv, seed_all, write_predictions, sha256_file)
    from raptor_data import StudyWindows, collate
    from raptor_runtime import configure_runtime
    from raptor_windows_ext import AllValidWindows
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    baseline = guard.is_baseline(a.mode)
    smoke = guard.is_smoke(a.mode)
    dev = a.device
    split = protocol["split"]
    # ---- data/label/membership contract (pure CPU, before any runtime or CUDA call)
    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    eval_labels, eval_label_hash = load_label_csv(Path(a.eval_labels))
    if label_hash != protocol["label_set"]["semantic_sha256"] or eval_label_hash != protocol["label_set"]["semantic_sha256"]:
        raise SystemExit("Label semantic hash mismatch")
    expected_corpus = protocol["base_corpus"] if baseline else protocol["corpus"]
    if sha256_file(corpus.dir / "manifest.json") != expected_corpus["manifest_sha256"] or corpus.contract != expected_corpus["contract"]:
        raise SystemExit("Corpus manifest changed")
    if not baseline and (corpus.depth != protocol["corpus"]["depth"] or [list(s) for s in corpus.spec["slots"]] != protocol["corpus"]["slots"]):
        raise SystemExit("Corpus slot layout differs from the registered 80-slot contract")
    group_rows = guard.read_rows(ROOT / split["groups"]["path"])
    canonical = guard.canonical_order(ordered)
    train_ids, oof_ids = guard.validate_fold_membership(corpus.ids.astype(str).tolist(), ordered, group_rows, canonical)
    if len(train_ids) != split["n_train"] or len(oof_ids) != split["n_val"]:
        raise SystemExit("Split size mismatch")
    if any(u not in labels for u in train_ids):
        raise SystemExit("Label set does not cover the training ids")
    computed_pos_weight = guard.pos_weight(labels, train_ids)
    pos_weight_np = np.asarray(split["pos_weight"], dtype=np.float32)
    if not np.array_equal(computed_pos_weight, pos_weight_np):
        raise SystemExit("pos_weight differs from the registered vector")
    full_contract = guard.run_contract(len(train_ids), protocol["common_args"])
    if any(full_contract[k] != split[k] for k in full_contract):
        raise RuntimeError("Batch/update/evaluation contract differs from the manifest")
    if smoke:
        train_ids, oof_ids = train_ids[:a.smoke_train], oof_ids[:a.smoke_val]
    epochs = a.smoke_epochs if smoke else a.epochs
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}
    eval_bs = max(1, a.bs // 2)
    evals = protocol["baseline_eval"]["evals"] if baseline else [dict(name=f"k{a.k_eval}", k_eval=a.k_eval, file="oof_best.csv", batch="eval_batch"),
                                                                 *protocol["extra_evals"]]
    if not baseline:
        tds = StudyWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
        shuffle_generator = torch.Generator().manual_seed(a.seed)
        tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=dev == "cuda",
                        persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)
        steps = max(math.ceil(len(tl) / a.accum) * epochs, 1)
        if not smoke and (len(tl) != split["len_train_loader"] or steps != split["steps"] or steps != split["scheduler_total_steps"] or eval_bs != split["eval_batch"]):
            raise RuntimeError("Batch/update/evaluation contract differs from the manifest")
        max_skips = steps // 100 if smoke else split["max_fp16_skips"]
        if not smoke and max_skips != steps // 100:
            raise RuntimeError("FP16 skip gate differs from 1% of the steps")
    else:
        tl, steps, max_skips = None, 0, 0
    checkpoint = ROOT / protocol["baseline_eval"]["checkpoint"]

    def load_baseline():
        if sha256_file(checkpoint) != protocol["baseline_eval"]["checkpoint_sha256"]:
            raise SystemExit("baseline checkpoint differs from the registered sha256")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        meta = {k: v for k, v in ck.items() if k != "model"}
        want = dict(arch=a.arch, res=a.res, contract=protocol["base_corpus"]["contract"], norm=a.norm, window_policy=a.window_policy, epoch=a.epochs - 1)
        if any(meta.get(k) != v for k, v in want.items()):
            raise SystemExit("baseline checkpoint metadata differ: " + str({k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}))
        m = build_model(meta["arch"], pretrained=False)
        m.load_state_dict(ck["model"], strict=True)
        return m, meta

    if a.validate_only:
        # CPU only: no runtime configuration, no .to(dev), no CUDA query (hiding the GPU via CUDA_VISIBLE_DEVICES crashes cv2 on this host).
        guard.worker_init(0)
        seed_all(a.seed)
        report = dict(status="VALIDATED_NO_TRAINING", tag=a.tag, mode=a.mode, contract=corpus.contract, depth=corpus.depth, train=len(train_ids),
                      val=len(oof_ids), eval_batch=eval_bs, evals=[(e["name"], e["k_eval"], e["file"]) for e in evals],
                      pos_weight=[round(float(x), 6) for x in pos_weight_np], label_semantic_sha256=label_hash,
                      output_root=str(out_root), output_exists=(out_root / a.tag).exists())
        if baseline:
            _, meta = load_baseline()
            report.update(checkpoint_sha256_matches=True, checkpoint_tag=meta.get("tag"), checkpoint_contract=meta.get("contract"))
        else:
            init = guard.state_hash(build_model(a.arch, pretrained=a.pretrained == "timm"))
            report.update(len_train_loader=len(tl), steps=steps, epochs=epochs, max_fp16_skips=max_skips, k_train=a.k, k_eval=a.k_eval,
                          initial_state_sha256=init, initial_state_matches_registered=init == protocol["initial_state_sha256"])
            if not report["initial_state_matches_registered"]:
                raise SystemExit("generic initial state differs from the registered B state")
        if torch.cuda.is_initialized():
            raise SystemExit("--validate-only touched CUDA")
        print(json.dumps(report, indent=1))
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

    log(f"{a.tag} | mode {a.mode} | corpus {corpus.contract} depth {corpus.depth} | K {a.k}/{a.k_eval} | train {len(train_ids)} | val {len(oof_ids)} | "
        f"steps {steps} | Gold and metrics disabled")
    if baseline:
        model, ck_meta = load_baseline()
        model = model.to(dev)
        init = None
    else:
        model = build_model(a.arch, pretrained=a.pretrained == "timm").to(dev)
        init = guard.state_hash(model)
    provenance = {"code_sha256": {p: sha256_file(ROOT / "training" / "src" / p) for p in guard.CODE_NAMES},
                  "ext_sha256": {p: sha256_file(ROOT / p) for p in guard.EXT_FILES}, "initial_state_sha256": init,
                  "initial_state_matches_registered": None if baseline else init == protocol["initial_state_sha256"],
                  "runtime": runtime, "eval_runtime": eval_runtime, "train_ids_ordered_sha256": guard.uid_hash(train_ids), "eval_ids_ordered_sha256": guard.uid_hash(oof_ids),
                  "eval_label_hash": eval_label_hash, "label_hash": label_hash, "label_semantic_sha256": label_hash,
                  "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"), "corpus_contract": corpus.contract, "corpus_depth": corpus.depth,
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": "fixed-final", "gold_access": "disabled", "loss_policy": "clipped_positive_term",
                  "actual_pos_weight": pos_weight_np.tolist(), "computed_pos_weight": computed_pos_weight.tolist(),
                  "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)),
                  "protocol_sha256": sha256_file(Path(a.protocol)), "fold": a.fold, "label_set": "dread", "label_file": protocol["label_set"]["path"],
                  "label_file_sha256": protocol["label_set"]["sha256"], "arch": a.arch, "seed": a.seed, "training_mode": a.mode, "k_train": a.k, "k_eval": a.k_eval,
                  "len_train_loader": len(tl) if tl is not None else None, "scheduler_total_steps": steps, "max_fp16_skips": max_skips,
                  "evals": evals, "metric_access": "deferred_single_fixed_evaluation", "requested_workers": a.workers, "output_root": str(out_root)}
    if baseline:
        provenance["baseline_checkpoint"] = dict(path=protocol["baseline_eval"]["checkpoint"], sha256=protocol["baseline_eval"]["checkpoint_sha256"],
                                                 tag=ck_meta.get("tag"), contract=ck_meta.get("contract"))
    if smoke:
        provenance["smoke"] = dict(train=len(train_ids), val=len(oof_ids), epochs=epochs, note="engineering smoke only; not a result")
    if not baseline and not provenance["initial_state_matches_registered"]:
        raise SystemExit("generic initial state differs from the registered B state")
    provenance["environment_matches_baseline"] = all(run_environment.get(k) == protocol["environment"][k] for k in guard.ENV_KEYS)
    provenance["environment"] = {k: run_environment.get(k) for k in guard.ENV_KEYS}
    if not provenance["environment_matches_baseline"] and not smoke:
        raise SystemExit("runtime environment differs from the registered baseline environment")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]

    def autocast(dtype):
        return torch.autocast(dev, dtype=dtype) if dtype is not None and dev == "cuda" else contextlib.nullcontext()

    def eval_loader(ids, spec):
        if spec["k_eval"] == "all":
            ds, bs = AllValidWindows(corpus.dir, ids, all_labels, a.res, a.norm, a.seed, window_policy=a.window_policy), 1
        else:
            ds = StudyWindows(corpus.dir, ids, all_labels, a.res, int(spec["k_eval"]), False, a.norm, a.seed, window_policy=a.window_policy)
            bs = eval_bs if spec["batch"] == "eval_batch" else int(spec["batch"])
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False,
                          worker_init_fn=guard.worker_init)

    @torch.no_grad()
    def evaluate(ids, spec):
        model.eval(); P, U, K = [], [], []
        for x, y, uids in eval_loader(ids, spec):
            with autocast(eval_dtype):
                logits = model(x.to(dev))
            if logits.shape != (len(uids), len(LAB)) or not torch.isfinite(logits).all():
                raise RuntimeError(f"invalid evaluation logits: {uids}")
            P.append(torch.sigmoid(logits.float()).cpu().numpy()); U += uids; K += [int(x.shape[1])] * len(uids)
        return np.concatenate(P), U, K

    history, optimizer_updates, skipped, cleanup = [], 0, 0, None
    t_start = time.time()
    if not baseline:
        with torch.no_grad():
            model.eval()(torch.zeros(1, 1, 3, a.res, a.res, device=dev)); model.train()
        if a.grad_ckpt:
            model.backbone.set_grad_checkpointing(True); log("gradient checkpointing on")
        head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
        opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_np, dtype=torch.float32, device=dev))
        scaler = torch.amp.GradScaler("cuda", init_scale=a.amp_init_scale, enabled=a.amp == "fp16" and dev == "cuda")
        best = None
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
                best = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB,
                        "contract": corpus.contract, "k_train": a.k, "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy, "epoch": ep,
                        "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp, "fold": a.fold, "label_set": "dread", "provenance": provenance}
                torch.save(best, out / "model_best.pt")
            (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        cleanup = trace_tools.stop_loader(tl)
        if optimizer_updates != steps:
            raise RuntimeError("full exact update gate failed before inference")
        provenance["fp16_skipped_updates"] = skipped
        model.load_state_dict(best["model"])
    window_counts, files, reproduction = {}, [], None
    for spec in evals:
        t_eval = time.time()
        P, U, K = evaluate(oof_ids, spec)
        if U != oof_ids:
            raise RuntimeError(f"{spec['name']}: prediction UID order changed")
        write_predictions(out / spec["file"], U, P); files.append(spec["file"])
        window_counts[spec["name"]] = dict(mean=round(float(np.mean(K)), 3), min=int(min(K)), max=int(max(K)), total=int(sum(K)),
                                           seconds=round(time.time() - t_eval, 1))
        log(f"{spec['name']} (K {spec['k_eval']}) predictions written to {spec['file']} ({len(U)} rows, mean K {np.mean(K):.1f}, {time.time() - t_eval:.0f}s)")
        if baseline and spec["name"] == "k24":
            import pandas as pd
            ref = pd.read_csv(ROOT / protocol["baseline_eval"]["reference_oof"], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
            diff = float(np.abs(ref.loc[U, LAB].to_numpy(dtype=np.float64) - P.astype(np.float64)).max())
            reproduction = dict(reference=protocol["baseline_eval"]["reference_oof"], max_abs_diff=diff, rows=len(U),
                                tolerance=protocol["baseline_eval"]["reproduction_tolerance"],
                                reproduces=diff <= protocol["baseline_eval"]["reproduction_tolerance"])
            log(f"K24 reproduction vs {reproduction['reference']}: max |diff| {diff:.3g} ({'OK' if reproduction['reproduces'] else 'DIFFERS'})")
    if dev == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU run initialised CUDA")
    provenance["cuda_initialized"] = bool(torch.cuda.is_initialized())
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    summary = {"contract": corpus.contract, "depth": corpus.depth, "backbone": a.arch, "img": a.res, "max_epochs": epochs, "train": len(train_ids),
               "val": len(oof_ids), "gold": 0, "rows": len(oof_ids), "weak_oof_macro_auc": None, "selected_checkpoint": "fixed-final",
               "gold_access": "disabled", "loss_policy": "clipped_positive_term", "provenance": provenance,
               "selected_epoch_1based": None if baseline else epochs, "optimizer_updates": optimizer_updates, "skipped_updates": skipped,
               "args": vars(a), "history": history, "environment": environment_snapshot(), "worker_cleanup": cleanup,
               "total_seconds": round(time.time() - t_start), "metric_access": "deferred_single_fixed_evaluation", "fold": a.fold, "label_set": "dread",
               "mode": a.mode, "oof_files": {e["name"]: e["file"] for e in evals}, "window_counts": window_counts, "reproduction": reproduction,
               "caveat": "weak-label G1 V888 predictions; one seed; compare only with tools/g1_compare.py against the same-machine B8 baseline"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    names = ["summary.json", "provenance.json"] + ([] if baseline else ["history.json", "model_best.pt"]) + files
    (out / "output-inventory.json").write_text(json.dumps({n: dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size) for n in names}, indent=1),
                                               encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
