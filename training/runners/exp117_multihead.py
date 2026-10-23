"""EXP117 (H-B-47) v1.0.0: one fold of the label-source-head RadR50 (tools/exp117_multihead.py registers EXP117_MULTIHEAD_PROTOCOL_V1).

A copy of training/runners/exp107_kfold_radr50.py (the runner EXP113 executes unchanged on the local GPU): data, windows, sampling, optimizer,
OneCycle scheduler, gradient clip, FP16 skip gate, fixed-final checkpoint, K24 + K36 evaluation and the run-file layout are unchanged.
Every difference is marked `EXP117:`:
  * labels: the four registered head sources (dread, relabel_v1, steven_v4, pilkwang_v2) are loaded and hash-checked; a training study's
    target is the head-major 48-vector (NaN where a source has no row); --labels / --eval-labels stay Dread (the dread head's file);
  * model: the EXP113 seed-42 RaptorClassifier (state enforced equal to EXP113's initial_state_sha256) wrapped by
    tools/exp117_multihead.build_multihead: shared trunk, the head copied once per source (head 0 = dread = EXP113's head);
  * loss: sum over heads of the masked weighted BCE with each source's own per-fold pos_weight (tools/exp117_multihead.masked_multihead_bce);
  * outputs: every head's probabilities (oof_head_<name>.csv / _k36.csv) and the served per-target head rank mean (oof_best.csv / oof_k36.csv);
  * history: per-head training loss and the pre-clip gradient norm per epoch (diagnostics);
  * OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 are supplied when absent, before torch is imported (as the EXP113 entry).

Modes: full (registered per-fold contract) and smoke (first N training / M validation studies, CPU allowed, --output-root outside
artifacts/). --validate-only checks protocol, membership, every head's pos_weight, the loader contract, the weight file and both initial
states on CPU, then exits without creating output.
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
REGISTERED_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1'}  # EXP117


def supply_registered_env(environ=None):  # EXP117: as training/runners/exp113_radr50_local.py
    environ = os.environ if environ is None else environ
    supplied = [k for k in REGISTERED_ENV if k not in environ]
    for k in supplied:
        environ[k] = REGISTERED_ENV[k]
    return supplied


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
    ap.add_argument("--norm", choices=["radimagenet"], required=True)
    ap.add_argument("--input-policy", choices=["center3"], required=True)
    ap.add_argument("--window-policy", choices=["within-slot-valid"], required=True)
    ap.add_argument("--runtime-policy", choices=["strict", "throughput"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pretrained", choices=["radimagenet"], required=True)
    ap.add_argument("--init-weights", required=True, help="RadImageNet ResNet50.pt (sha256 checked against the protocol)")
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    return ap


def main(argv=None):
    supply_registered_env()  # EXP117: before torch is imported
    a = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from tools import exp117_multihead as guard  # EXP117: own guard
    from tools import exp087_training as trace_tools
    protocol, fold, ordered, out_root = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, environment, load_label_csv, seed_all, write_predictions, sha256_file)
    from raptor_data import collate
    from raptor_radimagenet import CenterSliceWindows, build_radimagenet_model
    from raptor_runtime import configure_runtime
    import torch
    from torch.utils.data import DataLoader

    smoke = a.mode == "smoke"
    relaxed = bool(os.environ.get("KNEE_REMOTE_RELAX_ENV"))
    dev = a.device
    weights_sha = protocol["init_weights"]["sha256"]
    heads = [h["name"] for h in protocol["heads"]]  # EXP117
    # ---- data/label/membership contract (pure CPU, before any runtime or CUDA call)
    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    eval_labels, eval_label_hash = load_label_csv(Path(a.eval_labels))
    if label_hash != protocol["label_set"]["semantic_sha256"] or eval_label_hash != protocol["label_set"]["semantic_sha256"]:
        raise SystemExit("Label semantic hash mismatch")
    head_labels = {}  # EXP117: the four sources, each hash-checked against the registration
    for h in protocol["heads"]:
        path = ROOT / h["path"]
        if sha256_file(path) != h["sha256"]:
            raise SystemExit(f"head label file changed: {h['name']}")
        head_labels[h["name"]], semantic = load_label_csv(path)
        if semantic != h["semantic_sha256"]:
            raise SystemExit(f"head label semantic hash mismatch: {h['name']}")
    if protocol["heads"][0]["semantic_sha256"] != label_hash:
        raise SystemExit("the dread head is not the --labels file")
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
    head_pw = {n: guard.head_pos_weight(head_labels[n], train_ids) for n in heads}  # EXP117: each source's own rule
    for n in heads:
        if not np.array_equal(head_pw[n], np.asarray(fold["head_pos_weight"][n], np.float32)):
            raise SystemExit(f"{n} head pos_weight differs from the manifest fold vector")
    if not np.array_equal(head_pw["dread"], pos_weight_np):
        raise SystemExit("dread head pos_weight differs from the EXP113 vector")
    full_contract = guard.fold_contract(len(train_ids), protocol["common_args"])
    if any(full_contract[k] != fold[k] for k in full_contract):
        raise RuntimeError("Fold batch/update/evaluation contract differs from the manifest")
    if smoke:  # the registered recipe and pos_weight, on a few studies of this fold
        train_ids, oof_ids = train_ids[:a.smoke_train], oof_ids[:a.smoke_val]
    epochs = a.smoke_epochs if smoke else a.epochs
    stacked = guard.stack_sources(head_labels, heads, train_ids)  # EXP117: head-major 48-vectors, NaN = source absent
    absent = {n: int(sum(1 for u in train_ids if u not in head_labels[n])) for n in heads}
    all_labels = {**stacked, **{u: np.zeros(len(LAB) * len(heads), np.float32) for u in oof_ids}}
    tds = CenterSliceWindows(corpus.dir, train_ids, all_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy,
                             input_policy=a.input_policy)
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
    multihead_registered = protocol["multihead"]["multihead_initial_state_sha256"]  # EXP117

    def init_record(model):
        combined, keys = guard.state_hashes(model)
        mismatched = guard.mismatched_keys(keys, registered_keys)
        backbone_mismatched = [k for k in mismatched if k.startswith("backbone.")]
        if backbone_mismatched:
            raise SystemExit(f"RadImageNet backbone differs from the registered state ({len(backbone_mismatched)} keys: {backbone_mismatched[:5]})")
        return combined, combined == protocol["initial_state_sha256"], mismatched

    def make_model():
        return build_radimagenet_model(a.arch, Path(a.init_weights), weights_sha)

    if a.validate_only:
        # CPU only: no runtime configuration, no .to(dev), no CUDA query (hiding the GPU via CUDA_VISIBLE_DEVICES crashes cv2 on this host).
        guard.worker_init(0)
        seed_all(a.seed)
        built, load_report = make_model()
        init, matches, mismatched = init_record(built)
        multi = guard.build_multihead(built, len(heads))  # EXP117
        multi_init, _ = guard.state_hashes(multi)
        if torch.cuda.is_initialized():
            raise SystemExit("--validate-only touched CUDA")
        if a.seed == protocol["seed"] and not matches and not relaxed:
            raise SystemExit(f"generic initial state differs from the registered {a.arch} seed-{a.seed} state ({len(mismatched)} keys)")
        if a.seed == protocol["seed"] and multi_init != multihead_registered and not relaxed:
            raise SystemExit("multi-head initial state differs from the registration")
        print(json.dumps(dict(status="VALIDATED_NO_TRAINING", tag=a.tag, fold=a.fold, mode=a.mode, arch=a.arch, train=len(train_ids), val=len(oof_ids),
                              len_train_loader=len(tl), steps=steps, epochs=epochs, max_fp16_skips=max_skips, eval_batch=eval_bs, k_eval=a.k_eval,
                              k_eval_extra=k_extra, pos_weight=[round(float(x), 6) for x in pos_weight_np], label_semantic_sha256=label_hash,
                              heads=heads, head_pos_weight={n: [round(float(x), 6) for x in v] for n, v in head_pw.items()}, absent_rows=absent,
                              initial_state_sha256=init, initial_state_matches_registered=matches, initial_state_mismatched_keys=mismatched[:20],
                              multihead_initial_state_sha256=multi_init, multihead_initial_state_matches_registered=multi_init == multihead_registered,
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

    log(f"{a.tag} | {a.arch} RadImageNet x{len(heads)} heads {heads} | {a.input_policy} {a.norm} | fold {a.fold} seed {a.seed} mode {a.mode} | "
        f"train {len(train_ids)} | val {len(oof_ids)} | steps {steps} | Gold and metrics disabled")
    pos_weight = torch.tensor(np.stack([head_pw[n] for n in heads]), dtype=torch.float32, device=dev)  # EXP117: (heads, 12)

    def loader(ids, k_eval):
        return DataLoader(CenterSliceWindows(corpus.dir, ids, all_labels, a.res, k_eval, False, a.norm, a.seed, window_policy=a.window_policy,
                                             input_policy=a.input_policy), batch_size=eval_bs,
                          shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False, worker_init_fn=guard.worker_init)

    built, load_report = make_model()
    init, init_matches, init_mismatched = init_record(built)  # EXP113's trunk + head state (enforced below)
    model = guard.build_multihead(built, len(heads)).to(dev)  # EXP117
    multi_init, _ = guard.state_hashes(model)
    provenance = {"code_sha256": {p: sha256_file(ROOT / "training" / "src" / p) for p in guard.CODE_NAMES}, "initial_state_sha256": init,
                  "initial_state_matches_registered": init_matches, "initial_state_mismatched_keys": init_mismatched,
                  "multihead_initial_state_sha256": multi_init, "multihead_initial_state_matches_registered": multi_init == multihead_registered,
                  "runtime": runtime, "eval_runtime": eval_runtime, "train_ids_ordered_sha256": guard.uid_hash(train_ids), "eval_ids_ordered_sha256": guard.uid_hash(oof_ids),
                  "eval_label_hash": eval_label_hash, "label_hash": label_hash, "label_semantic_sha256": label_hash, "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"),
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": "fixed-final", "gold_access": "disabled", "loss_policy": "clipped_positive_term",
                  "actual_pos_weight": pos_weight_np.tolist(), "computed_pos_weight": computed_pos_weight.tolist(),
                  "heads": heads, "head_pos_weight": {n: head_pw[n].tolist() for n in heads}, "head_label_files": {h["name"]: h["sha256"] for h in protocol["heads"]},
                  "absent_training_rows": absent, "loss": protocol["multihead"]["loss"], "served_output": protocol["multihead"]["serving"],  # EXP117
                  "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)),
                  "protocol_sha256": sha256_file(Path(a.protocol)), "fold": a.fold, "label_set": guard.LABEL_SET, "label_file": protocol["label_set"]["path"],
                  "label_file_sha256": protocol["label_set"]["sha256"], "arch": a.arch, "seed": a.seed, "training_mode": a.mode,
                  "len_train_loader": len(tl), "scheduler_total_steps": steps, "max_fp16_skips": max_skips, "k_eval": a.k_eval, "k_eval_extra": k_extra,
                  "metric_access": "deferred_single_fixed_evaluation", "requested_workers": a.workers, "output_root": str(out_root), "remote_relaxed": relaxed,
                  "init_weights": load_report, "radimagenet_module_sha256": sha256_file(ROOT / "training" / "src" / "raptor_radimagenet.py"),
                  "input_policy": a.input_policy, "norm": a.norm}
    if smoke:
        provenance["smoke"] = dict(train=len(train_ids), val=len(oof_ids), epochs=epochs, note="engineering smoke only; not a result")
    if a.seed == protocol["seed"] and not init_matches and not relaxed:
        raise SystemExit(f"generic initial state differs from the registered {a.arch} seed-{a.seed} state ({len(init_mismatched)} keys)")
    if a.seed == protocol["seed"] and multi_init != multihead_registered and not relaxed:  # EXP117
        raise SystemExit("multi-head initial state differs from the registration")
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
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]
    scaler = torch.amp.GradScaler("cuda", init_scale=a.amp_init_scale, enabled=a.amp == "fp16" and dev == "cuda")
    skipped = 0

    def autocast(dtype):
        return torch.autocast(dev, dtype=dtype) if dtype is not None and dev == "cuda" else contextlib.nullcontext()

    @torch.no_grad()
    def evaluate(ids, k_eval):  # EXP117: (n, heads, 12) head probabilities
        model.eval(); P, U = [], []
        for x, y, uids in loader(ids, k_eval):
            with autocast(eval_dtype):
                logits = model(x.to(dev))
            if logits.shape != (len(uids), len(heads), len(LAB)) or not torch.isfinite(logits).all():
                raise RuntimeError(f"invalid evaluation logits: {uids}")
            P.append(torch.sigmoid(logits.float()).cpu().numpy()); U += uids
        return np.concatenate(P), U

    def write_outputs(P, U, served_name, suffix):  # EXP117: every head + the served head rank mean
        for h, n in enumerate(heads):
            write_predictions(out / f"oof_head_{n}{suffix}.csv", U, P[:, h])
        write_predictions(out / served_name, U, guard.rank_mean(P))

    history, best, optimizer_updates = [], None, 0
    t_start = time.time()
    for ep in range(epochs):
        tds.set_epoch(ep); model.train(); tot, t0 = 0.0, time.time(); opt.zero_grad(set_to_none=True)
        head_tot, norms, clipped = np.zeros(len(heads)), [], 0  # EXP117: diagnostics
        for i, (x, y, batch_uids) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with autocast(amp_dtype):
                logits = model(x)
                if logits.shape != (len(batch_uids), len(heads), 12) or not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite or invalid training logits")
                loss, head_losses = guard.masked_multihead_bce(logits.float(), y, pos_weight)  # EXP117
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at epoch {ep} step {i}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=a.amp != "fp16")
            norm_value = float(total_norm)  # EXP117: diagnostic only
            if math.isfinite(norm_value):
                norms.append(norm_value); clipped += norm_value > 3.0
            scale_before = scaler.get_scale()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if a.amp == "fp16" and dev == "cuda" and scaler.get_scale() < scale_before:
                skipped += 1
                if skipped > max_skips:
                    raise RuntimeError(f"FP16 optimizer skips exceed 1% of {steps} ({skipped})")
            else:
                sched.step()
            optimizer_updates += 1; tot += loss.item(); head_tot += head_losses.detach().float().cpu().numpy()
            if (i + 1) % 200 == 0 or smoke:
                rate = (time.time() - t0) / (i + 1)
                log(f"  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f} | {rate:.2f}s/step, epoch ETA {rate * (len(tl) - i - 1) / 60:.0f} min")
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        rec = {"epoch": ep, "loss": tot / max(1, len(tl)), "epoch_seconds": round(time.time() - t0), "peak_gpu_gib": round(peak, 2),
               "optimizer_updates": optimizer_updates, "fp16_skips": skipped,
               "head_loss": {n: float(head_tot[h] / max(1, len(tl))) for h, n in enumerate(heads)},  # EXP117
               "grad_norm_preclip_mean": float(np.mean(norms)) if norms else None, "clipped_fraction": clipped / max(1, len(norms))}
        history.append(rec); log(f"ep{ep} loss {rec['loss']:.4f} | fixed-final | {rec['epoch_seconds']}s | peak {peak:.1f} GiB | skips {skipped} | "
                                 f"clipped {rec['clipped_fraction']:.2f}")
        if ep == epochs - 1:
            best = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res, "lab": LAB, "contract": corpus.contract,
                    "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy, "epoch": ep, "tag": a.tag, "amp_train": a.amp, "amp_eval": a.eval_amp,
                    "fold": a.fold, "label_set": guard.LABEL_SET, "provenance": provenance}
            best.update(input_policy=a.input_policy, init="radimagenet", init_weights_sha256=weights_sha)
            best.update(model_class="exp117 MultiHeadRaptor (tools/exp117_multihead.build_multihead)", heads=heads,
                        serving=protocol["multihead"]["serving"])  # EXP117: serving must rebuild the multi-head model and rank-mean the heads
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
    write_outputs(P, U, "oof_best.csv", "")  # EXP117
    log(f"K{a.k_eval} OOF written ({len(U)} rows, {len(heads)} heads + rank mean, {time.time() - t_eval:.0f}s)")
    t_eval = time.time()
    P36, U36 = evaluate(oof_ids, k_extra)
    if U36 != oof_ids:
        raise RuntimeError("K36 prediction UID order changed")
    write_outputs(P36, U36, protocol["extra_eval"]["file"], f"_k{k_extra}")  # EXP117
    log(f"K{k_extra} OOF written ({len(U36)} rows, {time.time() - t_eval:.0f}s)")
    provenance["cuda_initialized"] = bool(torch.cuda.is_initialized())
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    summary = {"contract": corpus.contract, "backbone": a.arch, "img": a.res, "max_epochs": epochs, "train": len(train_ids), "val": len(oof_ids), "gold": 0, "rows": len(oof_ids),
               "weak_oof_macro_auc": None, "selected_checkpoint": "fixed-final", "gold_access": "disabled", "loss_policy": "clipped_positive_term", "provenance": provenance,
               "selected_epoch_1based": best["epoch"] + 1, "optimizer_updates": optimizer_updates, "skipped_updates": skipped, "args": vars(a), "history": history,
               "environment": environment_snapshot(), "worker_cleanup": cleanup, "total_seconds": round(time.time() - t_start), "metric_access": "deferred_single_fixed_evaluation",
               "fold": a.fold, "label_set": guard.LABEL_SET, "mode": a.mode, "heads": heads,
               "oof_files": {"k24": "oof_best.csv", f"k{k_extra}": protocol["extra_eval"]["file"],
                             "heads": {n: [f"oof_head_{n}.csv", f"oof_head_{n}_k{k_extra}.csv"] for n in heads}},
               "caveat": "weak-label fold OOF; served = per-target head rank mean within the fold's validation set; one seed"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", protocol["extra_eval"]["file"]]
    names += [f"oof_head_{n}{s}.csv" for n in heads for s in ("", f"_k{k_extra}")]  # EXP117
    (out / "output-inventory.json").write_text(json.dumps({n: dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size) for n in names}, indent=1), encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
