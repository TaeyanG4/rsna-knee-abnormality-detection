"""EXP112 V2 (H-B-33): the B8 CoAtNet recipe on all 4,343 G1 studies, 16 epochs, per-epoch checkpoints, fixed SWA of 1-based epochs 13/15/16.

V2 of training/runners/exp112_full_swa.py (V1 stays byte-identical: its sha256 is bound by EXP112_FULL_SWA_PROTOCOL_V1 and queued in
LOCAL_GPU_CHAIN_V14). The guard is tools/exp112_full_swa_v2.py (EXP112_FULL_SWA_PROTOCOL_V2: 16 epochs, OneCycle over all 8672 updates, SWA
members 1-based 13, 15, 16 = the public v5 0-based swa_over [14, 12, 15], D-A-050 + review M1). The training loop, per-epoch checkpoints,
SWA mean and BatchNorm pass are the V1 code unchanged (= EXP099 / EXP101 loop = EXP094 V4 arm seed42: AdamW, OneCycleLR pct_start 0.15,
grad clip 3.0, FP16 autocast + GradScaler init 1024 with the 1% skip gate, strict runtime, gradient checkpointing, sha256(seed, epoch, uid)
windows, seeded batch order; no validation, no evaluation, no metric, no checkpoint selection, Gold blocked by the audit hook).
Only change against the V1 runner (review m1): model_best.pt records every epoch number in one stated convention: epoch and swa_over are
0-based loop indices (the public / raptor_train / EXP099 convention), next to epoch_1based, swa_epochs_1based and epoch_numbering; swa.json
carries swa_over_0based and swa_epochs_1based.
Modes: full (registered contract) and smoke (the first N registered studies, a few epochs, CPU allowed, --output-root outside artifacts/,
own SWA epochs). --validate-only checks protocol, training list, pos_weight, loader contract and the seed-42 initial state on CPU and exits.
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
    ap.add_argument("--mode", choices=["full", "smoke"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--output-root", default=None, help="parent of the <tag> run directory (default artifacts/experiments)")
    ap.add_argument("--swa-epochs", required=True, help="registered 1-based SWA member epochs, e.g. 12,14,15")
    ap.add_argument("--smoke-train", type=int, default=8)
    ap.add_argument("--smoke-epochs", type=int, default=3)
    ap.add_argument("--smoke-swa-epochs", default="1,2,3")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--labels", required=True)
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
    from tools import exp112_full_swa_v2 as guard
    from tools import exp087_training as trace_tools
    protocol, registered_ids, out_root, plan = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import LAB, Corpus, environment, load_label_csv, seed_all, sha256_file
    from raptor_data import StudyWindows, collate
    from raptor_runtime import configure_runtime
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    smoke = plan["smoke"]
    epochs, swa_epochs = plan["epochs"], plan["swa_epochs"]
    dev = a.device
    # ---- data/label/training-set contract (pure CPU, before any runtime or CUDA call)
    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    if label_hash != protocol["label_set"]["semantic_sha256"]:
        raise SystemExit("Label semantic hash mismatch")
    if sha256_file(corpus.dir / "manifest.json") != protocol["corpus_manifest_sha256"]:
        raise SystemExit("Corpus manifest changed")
    train_ids = guard.validate_training_set(corpus.ids.astype(str).tolist(), registered_ids, protocol)
    if any(u not in labels for u in train_ids):
        raise SystemExit("Label set does not cover the training ids")
    computed_pos_weight = guard.pos_weight(labels, train_ids)
    pos_weight_np = np.asarray(protocol["pos_weight"], dtype=np.float32)
    if not np.array_equal(computed_pos_weight, pos_weight_np):
        raise SystemExit("pos_weight differs from the registered full-set vector")
    full_contract = guard.full_contract(len(train_ids), protocol["common_args"])
    if full_contract != protocol["contract"]:
        raise RuntimeError("batch/update contract differs from the manifest")
    if smoke:  # the registered recipe and pos_weight, on the first studies of the registered list
        train_ids = train_ids[:a.smoke_train]
    train_labels = {u: labels[u] for u in train_ids}
    tds = StudyWindows(corpus.dir, train_ids, train_labels, a.res, a.k, True, a.norm, a.seed, window_policy=a.window_policy)
    shuffle_generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=dev == "cuda",
                    persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)
    steps = max(math.ceil(len(tl) / a.accum) * epochs, 1)
    if not smoke and (len(tl) != full_contract["len_train_loader"] or steps != full_contract["steps"] or steps != full_contract["scheduler_total_steps"]):
        raise RuntimeError("batch/update contract differs from the manifest")
    max_skips = steps // 100 if smoke else full_contract["max_fp16_skips"]
    if not smoke and max_skips != steps // 100:
        raise RuntimeError("FP16 skip gate differs from 1% of the run's steps")
    registered_keys = guard.read(Path(a.protocol).resolve().parent / protocol["initial_state"]["keys_file"])["keys"]

    def init_record(model):
        combined, keys = guard.state_hashes(model)
        return combined, combined == protocol["initial_state_sha256"], guard.mismatched_keys(keys, registered_keys)

    def new_model():
        return guard.build_run_model(protocol, a.arch, pretrained=a.pretrained == "timm")

    if a.validate_only:
        # CPU only: no runtime configuration, no .to(dev), no CUDA query (hiding the GPU via CUDA_VISIBLE_DEVICES crashes cv2 on this host).
        guard.worker_init(0)
        seed_all(a.seed)
        init, matches, mismatched = init_record(new_model())
        if torch.cuda.is_initialized():
            raise SystemExit("--validate-only touched CUDA")
        if not matches:
            raise SystemExit(f"initial state differs from the registered {a.arch} seed-{a.seed} state ({len(mismatched)} keys)")
        print(json.dumps(dict(status="VALIDATED_NO_TRAINING", tag=a.tag, mode=a.mode, arch=a.arch, train=len(train_ids), validation=0,
                              len_train_loader=len(tl), steps=steps, epochs=epochs, swa_epochs_1based=list(swa_epochs), max_fp16_skips=max_skips,
                              pos_weight=[round(float(x), 6) for x in pos_weight_np], label_semantic_sha256=label_hash,
                              train_ids_ordered_sha256=guard.uid_hash(train_ids), initial_state_sha256=init, initial_state_matches_registered=matches,
                              output_root=str(out_root), output_exists=(out_root / a.tag).exists()), indent=1))
        return

    # ---- runtime and run directory (same order as EXP094/EXP099/EXP101: runtime -> worker threads -> environment -> seed)
    runtime = configure_runtime(a.amp, dev, policy=a.runtime_policy)
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
    ckpt_dir = out / guard.EPOCH_DIR
    ckpt_dir.mkdir(exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); log_file.write(msg + "\n"); log_file.flush()

    log(f"{a.tag} | {a.arch} | seed {a.seed} mode {a.mode} | train {len(train_ids)} (all G1, no validation) | epochs {epochs} | steps {steps} | "
        f"SWA epochs {list(swa_epochs)} (1-based) | Gold, evaluation and metrics disabled")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)
    model = new_model().to(dev)
    init, init_matches, init_mismatched = init_record(model)
    provenance = {"code_sha256": {p: sha256_file(ROOT / "training" / "src" / p) for p in guard.CODE_NAMES}, "initial_state_sha256": init,
                  "initial_state_matches_registered": init_matches, "initial_state_mismatched_keys": init_mismatched, "runtime": runtime,
                  "train_ids_ordered_sha256": guard.uid_hash(train_ids), "n_train": len(train_ids), "validation": "none", "eval_ids_ordered_sha256": None,
                  "label_hash": label_hash, "label_semantic_sha256": label_hash, "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"),
                  "sampling": "sha256(seed,epoch,uid,stream)-v1", "checkpoint_selection": protocol["recipe"]["checkpoint_selection"],
                  "gold_access": "disabled", "metric_access": "none", "loss_policy": "clipped_positive_term",
                  "actual_pos_weight": pos_weight_np.tolist(), "computed_pos_weight": computed_pos_weight.tolist(),
                  "runner_sha256": sha256_file(Path(__file__)), "guard_sha256": sha256_file(Path(guard.__file__)),
                  "protocol_sha256": sha256_file(Path(a.protocol)), "registration": protocol["registration"], "recipe_sha256": protocol["recipe_sha256"],
                  "fold": None, "training_set": "all G1 (union of the EXP099 fold training lists)", "label_set": guard.LABEL_SET,
                  "label_file": protocol["label_set"]["path"], "label_file_sha256": protocol["label_set"]["sha256"], "arch": a.arch, "seed": a.seed,
                  "training_mode": a.mode, "epochs": epochs, "swa_epochs_1based": list(swa_epochs), "epoch_convention": guard.SWA["convention"],
                  "len_train_loader": len(tl), "scheduler_total_steps": steps, "max_fp16_skips": max_skips, "k_eval": a.k_eval,
                  "requested_workers": a.workers, "output_root": str(out_root), "remote_relaxed": False}
    if smoke:
        provenance["smoke"] = dict(train=len(train_ids), epochs=epochs, swa_epochs_1based=list(swa_epochs), model=protocol["initial_state"]["model"],
                                   note="engineering smoke only; not a result")
    if not init_matches:
        raise SystemExit(f"initial state differs from the registered {a.arch} seed-{a.seed} state ({len(init_mismatched)} keys)")
    provenance["environment_matches_baseline"] = all(run_environment.get(k) == protocol["environment"][k] for k in guard.ENV_KEYS)
    provenance["environment"] = {k: run_environment.get(k) for k in guard.ENV_KEYS}
    if not provenance["environment_matches_baseline"] and not smoke:
        raise SystemExit("runtime environment differs from the registered baseline environment")
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

    def autocast(dtype):
        return torch.autocast(dev, dtype=dtype) if dtype is not None and dev == "cuda" else contextlib.nullcontext()

    meta = dict(arch=a.arch, res=a.res, lab=LAB, contract=corpus.contract, k_eval=a.k_eval, norm=a.norm, window_policy=a.window_policy, tag=a.tag,
                amp_train=a.amp, amp_eval=a.eval_amp, fold=None, label_set=guard.LABEL_SET, provenance=provenance)

    def save(payload, path):
        torch.save(payload, path)
        return dict(sha256=sha256_file(path), bytes=path.stat().st_size)

    history, epoch_inventory, optimizer_updates = [], {}, 0
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
        train_seconds = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        # ---- per-epoch checkpoint (1-based file name = epoch number; no selection, no evaluation)
        e1 = ep + 1
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        state_sha = guard.state_dict_sha256(state)
        epoch_ck = guard.release_payload(state, meta, ep, "epoch", epoch_1based=e1, optimizer_updates=optimizer_updates, fp16_skips=skipped,
                                         state_sha256=state_sha)
        rec_file = save(epoch_ck, ckpt_dir / guard.epoch_file(e1))
        epoch_inventory[guard.epoch_file(e1)] = dict(rec_file, epoch_1based=e1, loop_index=ep, state_sha256=state_sha, optimizer_updates=optimizer_updates,
                                                     fp16_skips=skipped, loss=tot / max(1, len(tl)))
        (ckpt_dir / guard.INVENTORY).write_text(json.dumps(epoch_inventory, indent=1), encoding="utf-8")
        if e1 == epochs:
            final_files = save(guard.release_payload(state, meta, ep, "final", epoch_1based=e1, optimizer_updates=optimizer_updates, fp16_skips=skipped,
                                                     state_sha256=state_sha), out / "model_final.pt")
        del state, epoch_ck
        rec = {"epoch": ep, "epoch_1based": e1, "loss": tot / max(1, len(tl)), "epoch_seconds": round(train_seconds),
               "checkpoint_seconds": round(time.time() - t0 - train_seconds, 1), "peak_gpu_gib": round(peak, 2), "optimizer_updates": optimizer_updates,
               "fp16_skips": skipped, "state_sha256": state_sha}
        history.append(rec)
        log(f"ep{ep} (epoch {e1}) loss {rec['loss']:.4f} | {rec['epoch_seconds']}s | peak {peak:.1f} GiB | skips {skipped} | saved {guard.epoch_file(e1)}")
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    if optimizer_updates != steps:
        raise RuntimeError("full exact update gate failed")
    provenance["fp16_skipped_updates"] = skipped

    # ---- SWA: arithmetic mean of the registered epochs, re-read from disk and checked against the recorded state hashes
    t_swa = time.time()
    members, states = [], []
    for e1 in swa_epochs:
        name = guard.epoch_file(e1)
        rec = epoch_inventory[name]
        if sha256_file(ckpt_dir / name) != rec["sha256"]:
            raise RuntimeError(f"{name} changed on disk")
        ck = torch.load(ckpt_dir / name, map_location="cpu", weights_only=False)
        if ck["epoch_1based"] != e1 or guard.state_dict_sha256(ck["model"]) != rec["state_sha256"]:
            raise RuntimeError(f"{name} does not hold the epoch-{e1} state")
        states.append(ck["model"]); members.append(dict(epoch_1based=e1, loop_index=e1 - 1, file=f"{guard.EPOCH_DIR}/{name}", sha256=rec["sha256"],
                                                        state_sha256=rec["state_sha256"], optimizer_updates=rec["optimizer_updates"]))
        del ck
    swa_state = guard.swa_average(states)
    del states
    swa_mean_sha = guard.state_dict_sha256(swa_state)
    model.load_state_dict(swa_state, strict=True)
    log(f"SWA mean of epochs {list(swa_epochs)} (1-based) loaded ({time.time() - t_swa:.0f}s); recomputing BatchNorm statistics")
    # ---- BatchNorm statistics: one more pass of the training loader, train mode, no gradient, training autocast
    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(False)
    if not smoke and epochs != guard.BN_DRAW_EPOCH:
        raise RuntimeError('BatchNorm window draw index differs from the registration')
    tds.set_epoch(epochs)  # 0-based draw index = epochs: a window draw no training epoch used
    bn_record = guard.recompute_bn(model, tl, dev, autocast_ctx=autocast(amp_dtype))
    bn_record.update(window_draw_epoch_0based=int(tds.epoch.item()), amp=a.amp if dev == "cuda" else "none", device=dev,
                     grad_checkpointing="off during the pass")
    if not smoke and (bn_record["batches"] != full_contract["len_train_loader"] or bn_record["bn_modules"] != protocol["initial_state"]["bn_modules"]):
        raise RuntimeError(f"BatchNorm pass differs from the registration: {bn_record}")
    swa_bn_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bn_buffers = {n + "." + b for n, _ in guard.batchnorm_modules(model) for b in ("running_mean", "running_var", "num_batches_tracked")}
    changed = sorted(k for k in swa_bn_state if not torch.equal(swa_bn_state[k], swa_state[k]))
    if set(changed) - bn_buffers:
        raise RuntimeError("the BatchNorm pass changed non-BatchNorm tensors: " + ", ".join(sorted(set(changed) - bn_buffers)[:5]))
    cleanup = trace_tools.stop_loader(tl)
    swa_record = dict(epochs_1based=list(swa_epochs), swa_over_0based=[e - 1 for e in swa_epochs], epoch_numbering=guard.EPOCH_NUMBERING,
                      convention=guard.SWA["convention"], rule=guard.SWA["rule"], selection=guard.SWA["selection"], members=members,
                      mean_state_sha256=swa_mean_sha, bn_recompute=bn_record, bn_changed_tensors=len(changed),
                      final_state_sha256=guard.state_dict_sha256(swa_bn_state), seconds=round(time.time() - t_swa))
    if dev == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU run initialised CUDA")
    provenance["cuda_initialized"] = bool(torch.cuda.is_initialized())
    # epoch numbering (review m1): epoch / swa_over are 0-based loop indices like every other checkpoint and the public swa_over;
    # the 1-based numbers sit next to them under explicit names
    best = guard.release_payload(swa_bn_state, meta, epochs - 1, "swa", epoch_1based=epochs, swa_over=[e - 1 for e in swa_epochs],
                                 swa_epochs_1based=list(swa_epochs), epoch_numbering=guard.EPOCH_NUMBERING, swa=swa_record,
                                 optimizer_updates=optimizer_updates, fp16_skips=skipped, state_sha256=swa_record["final_state_sha256"])
    guard.release_check(best, LAB)
    best_files = save(best, out / "model_best.pt")
    del best, swa_state
    log(f"model_best.pt = SWA{list(swa_epochs)} + BatchNorm recompute ({bn_record['batches']} batches, {bn_record['studies']} studies, "
        f"{bn_record['bn_modules']} BN modules, {bn_record['seconds']}s)")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    (out / "swa.json").write_text(json.dumps(swa_record, indent=1), encoding="utf-8")
    summary = {"contract": corpus.contract, "backbone": a.arch, "img": a.res, "max_epochs": epochs, "train": len(train_ids), "val": 0, "gold": 0, "rows": 0,
               "validation": "none", "selected_checkpoint": "none: model_best.pt = fixed SWA of 1-based epochs "
               + ",".join(map(str, swa_epochs)) + " + BatchNorm recompute", "gold_access": "disabled", "metric_access": "none",
               "loss_policy": "clipped_positive_term", "provenance": provenance, "swa": swa_record, "final_epoch_1based": epochs,
               "optimizer_updates": optimizer_updates, "skipped_updates": skipped, "args": vars(a), "history": history,
               "environment": environment_snapshot(), "worker_cleanup": cleanup, "total_seconds": round(time.time() - t_start), "mode": a.mode,
               "label_set": guard.LABEL_SET, "files": {"model_best": "model_best.pt", "model_final": "model_final.pt",
                                                       "epoch_checkpoints": f"{guard.EPOCH_DIR}/{guard.INVENTORY}"},
               "caveat": "trained on every labelled G1 study: no local validation score exists; judged on public only (H-B-33); one seed"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    inventory = {"model_best.pt": best_files, "model_final.pt": final_files}
    for n in ("summary.json", "provenance.json", "history.json", "swa.json", f"{guard.EPOCH_DIR}/{guard.INVENTORY}"):
        inventory[n] = dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size)
    (out / guard.INVENTORY).write_text(json.dumps(inventory, indent=1), encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()


if __name__ == "__main__":
    main()
