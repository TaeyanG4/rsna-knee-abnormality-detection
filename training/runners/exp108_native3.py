"""EXP108 (H-B-25): B8 seed-42 G1 fold-0 fit whose window channels are native-adjacent DICOM slices (tools/exp108_native3.py registers it).

The training loop is the EXP102 runner's without its baseline modes (itself the EXP099 / EXP094 V4 seed42 loop: CoAtNet-2 224,
within-slot-valid windows, K12 train / K24 eval, clipped pos_weight, fixed-final epoch 8, FP16 GradScaler init 1024, strict runtime,
Gold blocker, no metric access). Changed: the dataset is raptor_native3_data.NativeTripletWindows on RAPTOR64N3_336_C140_S02, i.e.
each window is [native c-1, c, native c+1] instead of [cache c-1, c, cache c+1]; the centres, mask, jitter and centre plane are the
baseline's. After training the same final model writes oof_best.csv (K24) and oof_k36.csv (K36).

Modes: full (registered contract, artifacts/experiments/<tag>) and smoke (first N train / M val studies, CPU allowed, --output-root
outside artifacts/). --validate-only checks protocol, bindings, cache, membership, pos_weight, loader contract, the seed-42 initial
state and an input check (centre channel identical to the baseline loader on 4 training studies at epoch 0) on CPU, then exits.
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
    ap.add_argument("--mode", choices=["full", "smoke"], required=True)
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


def input_check(corpus_dir, base_dir, ids, labels, a, n=4):
    """Epoch-0 training inputs of the first n studies: the centre channel must equal the baseline loader's; count changed outer channels."""
    from raptor_data import StudyWindows
    from raptor_native3_data import NativeTripletWindows
    ids = list(ids[:n])
    nat = NativeTripletWindows(corpus_dir, ids, labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
    ref = StudyWindows(base_dir, ids, labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
    windows = centre_equal = outer_changed = 0
    for i in range(len(ids)):
        x, _, u1 = nat[i]
        y, _, u2 = ref[i]
        if u1 != u2 or tuple(x.shape) != tuple(y.shape):
            raise RuntimeError("native3 / baseline loaders disagree on uid or shape")
        x, y = x.numpy(), y.numpy()
        windows += x.shape[0]
        centre_equal += int(sum(np.array_equal(x[j, 1], y[j, 1]) for j in range(x.shape[0])))
        outer_changed += int(sum(not (np.array_equal(x[j, 0], y[j, 0]) and np.array_equal(x[j, 2], y[j, 2])) for j in range(x.shape[0])))
    report = dict(studies=len(ids), windows=windows, centre_channel_equal=centre_equal, outer_channels_changed=outer_changed)
    if centre_equal != windows:
        raise RuntimeError(f"centre channel differs from the baseline loader: {report}")
    return report


def main(argv=None):
    a = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from tools import exp108_native3 as guard
    from tools import exp087_training as trace_tools
    protocol, ordered, out_root = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, build_model, environment, load_label_csv, seed_all, write_predictions, sha256_file)
    from raptor_data import collate
    from raptor_native3 import is_native3_spec
    from raptor_native3_data import NativeTripletWindows
    from raptor_runtime import configure_runtime
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    smoke = guard.is_smoke(a.mode)
    dev = a.device
    split = protocol["split"]
    # ---- data/label/membership contract (pure CPU, before any runtime or CUDA call)
    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    eval_labels, eval_label_hash = load_label_csv(Path(a.eval_labels))
    if label_hash != protocol["label_set"]["semantic_sha256"] or eval_label_hash != protocol["label_set"]["semantic_sha256"]:
        raise SystemExit("Label semantic hash mismatch")
    if sha256_file(corpus.dir / "manifest.json") != protocol["corpus"]["manifest_sha256"] or corpus.contract != protocol["corpus"]["contract"]:
        raise SystemExit("Corpus manifest changed")
    if (not is_native3_spec(corpus.spec) or corpus.depth != protocol["corpus"]["depth"]
            or [list(s) for s in corpus.spec["slots"]] != protocol["corpus"]["slots"]):
        raise SystemExit("Corpus is not the registered native3 64-slot cache")
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
    evals = [dict(name=f"k{a.k_eval}", k_eval=a.k_eval, file="oof_best.csv", batch="eval_batch"), *protocol["extra_evals"]]
    tds = NativeTripletWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
    shuffle_generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=dev == "cuda",
                    persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)
    steps = max(math.ceil(len(tl) / a.accum) * epochs, 1)
    if not smoke and (len(tl) != split["len_train_loader"] or steps != split["steps"] or steps != split["scheduler_total_steps"] or eval_bs != split["eval_batch"]):
        raise RuntimeError("Batch/update/evaluation contract differs from the manifest")
    max_skips = steps // 100 if smoke else split["max_fp16_skips"]
    if not smoke and max_skips != steps // 100:
        raise RuntimeError("FP16 skip gate differs from 1% of the steps")
    base_dir = guard.relpath(ROOT, protocol["paths"]["base_corpus"])
    guard.worker_init(0)
    check = input_check(corpus.dir, base_dir, train_ids, all_labels, a)  # CPU, before any seeding; uses no global RNG

    if a.validate_only:
        # CPU only: no runtime configuration, no .to(dev), no CUDA query (hiding the GPU via CUDA_VISIBLE_DEVICES crashes cv2 on this host).
        seed_all(a.seed)
        report = dict(status="VALIDATED_NO_TRAINING", tag=a.tag, mode=a.mode, contract=corpus.contract, depth=corpus.depth, planes=corpus.spec["planes"],
                      train=len(train_ids), val=len(oof_ids), eval_batch=eval_bs, evals=[(e["name"], e["k_eval"], e["file"]) for e in evals],
                      pos_weight=[round(float(x), 6) for x in pos_weight_np], label_semantic_sha256=label_hash, input_check=check,
                      output_root=str(out_root), output_exists=(out_root / a.tag).exists())
        init = guard.state_hash(build_model(a.arch, pretrained=a.pretrained == "timm"))
        report.update(len_train_loader=len(tl), steps=steps, epochs=epochs, max_fp16_skips=max_skips, k_train=a.k, k_eval=a.k_eval,
                      initial_state_sha256=init, initial_state_matches_registered=init == protocol["initial_state_sha256"])
        if not report["initial_state_matches_registered"]:
            raise SystemExit("generic initial state differs from the registered B state")
        if torch.cuda.is_initialized():
            raise SystemExit("--validate-only touched CUDA")
        print(json.dumps(report, indent=1))
        return

    # ---- runtime and run directory (same order as EXP094/EXP099/EXP102: runtime -> worker threads -> environment -> seed)
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

    log(f"{a.tag} | mode {a.mode} | corpus {corpus.contract} depth {corpus.depth} x {corpus.spec['planes']} planes | K {a.k}/{a.k_eval} | "
        f"train {len(train_ids)} | val {len(oof_ids)} | steps {steps} | Gold and metrics disabled")
    model = build_model(a.arch, pretrained=a.pretrained == "timm").to(dev)
    init = guard.state_hash(model)
    provenance = {"code_sha256": {p: sha256_file(ROOT / "training" / "src" / p) for p in guard.CODE_NAMES},
                  "ext_sha256": {p: sha256_file(ROOT / p) for p in guard.EXT_FILES}, "initial_state_sha256": init,
                  "initial_state_matches_registered": init == protocol["initial_state_sha256"],
                  "runtime": runtime, "eval_runtime": eval_runtime, "train_ids_ordered_sha256": guard.uid_hash(train_ids), "eval_ids_ordered_sha256": guard.uid_hash(oof_ids),
                  "eval_label_hash": eval_label_hash, "label_hash": label_hash, "label_semantic_sha256": label_hash,
                  "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"), "corpus_contract": corpus.contract, "corpus_depth": corpus.depth,
                  "window_channels": "native3: [native c-1, c, native c+1] (raptor_native3_data.NativeTripletWindows)",
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": "fixed-final", "gold_access": "disabled", "loss_policy": "clipped_positive_term",
                  "actual_pos_weight": pos_weight_np.tolist(), "computed_pos_weight": computed_pos_weight.tolist(),
                  "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)),
                  "protocol_sha256": sha256_file(Path(a.protocol)), "fold": a.fold, "label_set": "dread", "label_file": protocol["label_set"]["path"],
                  "label_file_sha256": protocol["label_set"]["sha256"], "arch": a.arch, "seed": a.seed, "training_mode": a.mode, "k_train": a.k, "k_eval": a.k_eval,
                  "len_train_loader": len(tl), "scheduler_total_steps": steps, "max_fp16_skips": max_skips,
                  "evals": evals, "metric_access": "deferred_single_fixed_evaluation", "requested_workers": a.workers, "output_root": str(out_root)}
    provenance["input_check"] = check
    if smoke:
        provenance["smoke"] = dict(train=len(train_ids), val=len(oof_ids), epochs=epochs, note="engineering smoke only; not a result")
    if not provenance["initial_state_matches_registered"]:
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
        ds = NativeTripletWindows(corpus.dir, ids, all_labels, a.res, int(spec["k_eval"]), False, a.norm, a.seed, window_policy=a.window_policy)
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

    history, optimizer_updates, skipped = [], 0, 0
    t_start = time.time()
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
                    "window_channels": "native3", "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp, "fold": a.fold, "label_set": "dread",
                    "provenance": provenance}
            torch.save(best, out / "model_best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    cleanup = trace_tools.stop_loader(tl)
    if optimizer_updates != steps:
        raise RuntimeError("full exact update gate failed before inference")
    provenance["fp16_skipped_updates"] = skipped
    model.load_state_dict(best["model"])
    window_counts, files = {}, []
    for spec in evals:
        t_eval = time.time()
        P, U, K = evaluate(oof_ids, spec)
        if U != oof_ids:
            raise RuntimeError(f"{spec['name']}: prediction UID order changed")
        write_predictions(out / spec["file"], U, P); files.append(spec["file"])
        window_counts[spec["name"]] = dict(mean=round(float(np.mean(K)), 3), min=int(min(K)), max=int(max(K)), total=int(sum(K)),
                                           seconds=round(time.time() - t_eval, 1))
        log(f"{spec['name']} (K {spec['k_eval']}) predictions written to {spec['file']} ({len(U)} rows, {time.time() - t_eval:.0f}s)")
    if dev == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU run initialised CUDA")
    provenance["cuda_initialized"] = bool(torch.cuda.is_initialized())
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    summary = {"contract": corpus.contract, "depth": corpus.depth, "planes": corpus.spec["planes"], "backbone": a.arch, "img": a.res, "max_epochs": epochs,
               "train": len(train_ids), "val": len(oof_ids), "gold": 0, "rows": len(oof_ids), "weak_oof_macro_auc": None, "selected_checkpoint": "fixed-final",
               "gold_access": "disabled", "loss_policy": "clipped_positive_term", "provenance": provenance, "input_check": provenance["input_check"],
               "selected_epoch_1based": epochs, "optimizer_updates": optimizer_updates, "skipped_updates": skipped,
               "args": vars(a), "history": history, "environment": environment_snapshot(), "worker_cleanup": cleanup,
               "total_seconds": round(time.time() - t_start), "metric_access": "deferred_single_fixed_evaluation", "fold": a.fold, "label_set": "dread",
               "mode": a.mode, "oof_files": {e["name"]: e["file"] for e in evals}, "window_counts": window_counts,
               "caveat": "weak-label G1 V888 predictions; one seed; compare only with tools/g1_compare.py against the same-machine B8 baseline"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    names = ["summary.json", "provenance.json", "history.json", "model_best.pt"] + files
    (out / "output-inventory.json").write_text(json.dumps({n: dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size) for n in names}, indent=1),
                                               encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
