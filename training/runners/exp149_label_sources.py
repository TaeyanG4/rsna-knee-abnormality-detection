"""EXP149 AUG1 RadR50 nested teachers and label-source students.

The run registration and label provenance are owned by tools.exp149_label_sources.
No metric or Gold58 data is accessed during training. Full runs write a fixed
final checkpoint plus K24 and K36 predictions. Validation exits before CUDA.
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

ROOT = Path(__file__).resolve().parents[2]
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}


def supply_registered_env(environ=None):
    """Set worker limits before importing torch and the image stack."""
    environ = os.environ if environ is None else environ
    for key, value in THREAD_ENV.items():
        environ[key] = value


def move_batch(batch, device):
    """pin_memory can turn a collated tuple into a list on CUDA."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise TypeError("expected a three-element tuple or list batch")
    x, y, uids = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), uids


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--run", required=True, help="registered EXP149 run tag")
    ap.add_argument("--tag", default=None, help="optional alias, must match --run")
    ap.add_argument("--fold", type=int, default=-1, help="outer fold, -1 for a nested teacher")
    ap.add_argument("--variant", choices=["AUG1"], default="AUG1")
    ap.add_argument("--mode", choices=["full", "smoke"], required=True)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--output-root")
    ap.add_argument("--smoke-train", type=int, default=8)
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
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    return ap


def main(argv=None):
    supply_registered_env()
    import numpy as np
    a = build_parser().parse_args(argv)
    if a.tag is not None and a.tag != a.run:
        raise ValueError("--tag must equal --run")
    if a.validate_only and a.device != "cpu":
        raise ValueError("validation must use CPU")
    sys.path.insert(0, str(ROOT))
    from tools import exp149_label_sources as guard
    from tools import exp117_multihead as mh
    from tools import exp087_training as trace_tools
    from tools import exp134_aug1_kfold as aug
    if a.device == "cpu":
        aug.forbid_cuda_init()
    protocol, spec, ordered, out_root = guard.validate_arguments(a, ROOT)
    aug.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, environment, seed_all,
                               write_predictions, sha256_file)
    from raptor_data import collate
    from raptor_radimagenet import build_radimagenet_model
    from raptor_exp127 import loader_factory, ordered_equivalence
    from raptor_runtime import configure_runtime
    import torch
    from torch.utils.data import DataLoader

    smoke, dev = a.mode == "smoke", a.device
    heads = tuple(spec.get("heads", ()))
    multi = len(heads) > 0
    if multi and (heads[:4] != tuple(mh.HEAD_NAMES) or heads[4:] != ("aquila_v1",)):
        raise ValueError("H1 requires four EXP138 heads and aquila_v1 fifth")
    serving_rule = mh.SERVING_RULE.replace("four heads", "five heads") if multi else "single-head probabilities"
    train_ids, val_ids = list(ordered["train"]), list(ordered["val"])
    if (len(train_ids), len(val_ids)) != (spec["n_train"], spec["n_val"]):
        raise ValueError("registered study counts differ")
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids):
        raise ValueError("train/eval study membership invalid")
    corpus = Corpus(Path(a.corpus))
    if not set(train_ids + val_ids).issubset(set(corpus.ids.astype(str))):
        raise ValueError("registered study absent from corpus")
    labels, semantic = aug.load_label_csv(Path(a.labels))
    eval_labels, eval_semantic = aug.load_label_csv(Path(a.eval_labels))
    if semantic != spec["label_semantic_sha256"] or sha256_file(Path(a.labels)) != spec["label_sha256"]:
        raise ValueError("training label file changed")
    if eval_semantic != protocol["label_set"]["semantic_sha256"]:
        raise ValueError("evaluation label file changed")
    if any(uid not in labels for uid in train_ids):
        raise ValueError("training labels do not cover every registered study")
    head_labels = {}
    if multi:
        for h in spec["head_labels"]:
            path = ROOT / h["path"]
            if sha256_file(path) != h["sha256"]:
                raise ValueError("head label file changed: " + h["name"])
            head_labels[h["name"]], digest = aug.load_label_csv(path)
            if digest != h["semantic_sha256"]:
                raise ValueError("head label semantic hash changed: " + h["name"])
        if tuple(head_labels) != heads or any(uid not in head_labels["aquila_v1"] for uid in train_ids):
            raise ValueError("H1 head order or aquila_v1 coverage differs")
        actual_pw = {h: mh.head_pos_weight(head_labels[h], train_ids) for h in heads}
        for h in heads:
            if not np.array_equal(actual_pw[h], np.asarray(spec["head_pos_weight"][h], np.float32)):
                raise ValueError("H1 head pos_weight differs: " + h)
    pos_weight = aug.pos_weight(labels, train_ids)
    if not np.array_equal(pos_weight, np.asarray(spec["pos_weight"], np.float32)):
        raise ValueError("training pos_weight differs from protocol")
    if multi:
        all_labels = {**mh.stack_sources(head_labels, heads, train_ids),
                      **{uid: np.zeros(len(LAB) * len(heads), np.float32) for uid in val_ids}}
    else:
        all_labels = {**{uid: labels[uid] for uid in train_ids},
                      **{uid: np.zeros(len(LAB), np.float32) for uid in val_ids}}

    if smoke:
        if a.smoke_train < a.bs or a.smoke_val < 1 or a.smoke_epochs != 1:
            raise ValueError("smoke requires one epoch and at least one training batch")
        train_ids, val_ids = train_ids[:a.smoke_train], val_ids[:a.smoke_val]
    epochs = a.smoke_epochs if smoke else a.epochs
    windows = loader_factory("AUG1")
    train_ds = windows(corpus.dir, train_ids, all_labels, a.res, a.k, True,
                       a.norm, a.seed, window_policy=a.window_policy,
                       input_policy=a.input_policy)
    train_loader = DataLoader(train_ds, batch_size=a.bs, shuffle=True,
                              num_workers=a.workers, drop_last=True, collate_fn=collate,
                              pin_memory=dev == "cuda", persistent_workers=a.workers > 0,
                              generator=torch.Generator().manual_seed(a.seed),
                              worker_init_fn=aug.worker_init)
    eval_bs = max(1, a.bs // 2)
    if smoke and (len(train_loader) != 1 or len(val_ids) > eval_bs):
        raise ValueError("GPU smoke must be exactly one training and one evaluation batch")
    steps = max(math.ceil(len(train_loader) / a.accum) * epochs, 1)
    if not smoke and (len(train_loader), steps, eval_bs) != (
            spec["len_train_loader"], spec["steps"], spec["eval_batch"]):
        raise ValueError("registered loader/update/eval contract differs")
    max_skips = steps // 100 if smoke else spec["max_fp16_skips"]
    if not smoke and (spec["scheduler_total_steps"] != steps or max_skips != steps // 100):
        raise ValueError("scheduler or FP16 skip gate differs")
    k_extra = int(protocol["extra_eval"]["k_eval"])
    weights_sha = protocol["init_weights"]["sha256"]

    def make_model():
        base, report = build_radimagenet_model(a.arch, Path(a.init_weights), weights_sha)
        base_initial, _ = aug.state_hashes(base)
        if base_initial != protocol["initial_state_sha256"]:
            raise ValueError("initial RadR50 state differs from EXP134 registration")
        model = mh.build_multihead(base, len(heads)) if multi else base
        full_initial, _ = aug.state_hashes(model)
        if full_initial != spec["initial_state_sha256"]:
            raise ValueError("initial EXP149 model state differs from registration")
        if multi:
            # EXP117 clones the registered base head; no new initialization draw.
            for head_model in model.heads:
                if (not torch.equal(head_model.clsW, base.clsW)
                        or not torch.equal(head_model.clsb, base.clsb)
                        or any(not torch.equal(value, base.norm.state_dict()[key])
                               for key, value in head_model.norm.state_dict().items())
                        or any(not torch.equal(value, base.att.state_dict()[key])
                               for key, value in head_model.att.state_dict().items())):
                    raise ValueError("H1 head did not clone the registered base head")
        return model, report, base_initial, full_initial

    if a.validate_only:
        aug.worker_init(0)
        seed_all(a.seed)
        equivalence = ordered_equivalence("AUG1", corpus.dir, train_ids, val_ids,
                                          all_labels, a.seed, a.res)
        seed_all(a.seed)
        with torch.device("cpu"):
            model, load_report, initial, full_initial = make_model()
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU validation initialized CUDA")
        print(json.dumps({"status": "VALIDATED_NO_TRAINING", "run": a.run,
                          "train": len(train_ids), "val": len(val_ids), "steps": steps,
                          "epochs": epochs, "heads": list(heads),
                          "label_semantic_sha256": semantic,
                          "initial_state_sha256": initial,
                          "full_model_initial_state_sha256": full_initial,
                          "ordered_equivalence": equivalence, "init_weights": load_report,
                          "output_exists": (out_root / a.run).exists()}, indent=1))
        return

    runtime = configure_runtime(a.amp, dev, policy=a.runtime_policy)
    eval_runtime = configure_runtime(a.eval_amp, dev, policy=a.runtime_policy)
    aug.worker_init(0)
    seed_all(a.seed)
    model, load_report, initial, full_initial = make_model()
    model = model.to(dev)
    env = environment()
    environment_matches = all(env.get(k) == protocol["environment"][k] for k in aug.ENV_KEYS)
    if not smoke and not environment_matches:
        raise ValueError("runtime environment differs from registered EXP134 baseline")
    out = out_root / a.run
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refusing to overwrite existing run: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    log(f"{a.run}: {len(train_ids)} train, {len(val_ids)} held out, {steps} updates, {len(heads) or 1} head(s)")
    provenance = {"run": a.run, "fold": a.fold, "arch": a.arch,
                  "training_mode": a.mode, "label_set": spec["label_set"],
                  "train_folds": spec["train_folds"],
                  "eval_folds": spec["eval_folds"], "seed": a.seed,
                  "train_ids_ordered_sha256": aug.uid_hash(train_ids),
                  "eval_ids_ordered_sha256": aug.uid_hash(val_ids),
                  "label_file": spec["labels"], "label_file_sha256": spec["label_sha256"],
                  "label_semantic_sha256": semantic,
                  "eval_label_hash": eval_semantic,
                  "head_label_files": {h["name"]: h["sha256"] for h in spec.get("head_labels", [])},
                  "initial_state_sha256": initial, "initial_state_matches_registered": True,
                  "full_model_initial_state_sha256": full_initial,
                  "init_weights": load_report,
                  "protocol_sha256": sha256_file(Path(a.protocol)),
                  "runner_sha256": sha256_file(Path(__file__)),
                  "guard_sha256": sha256_file(Path(guard.__file__)),
                  "runtime": runtime, "eval_runtime": eval_runtime,
                  "optimizer": "AdamW", "scheduler": "OneCycleLR", "checkpoint_selection": "fixed-final",
                  "gold_access": "disabled", "epochs": epochs, "steps": steps,
                  "k_train": a.k, "k_eval": a.k_eval, "k_eval_extra": k_extra,
                  "input_policy": a.input_policy, "norm": a.norm,
                  "len_train_loader": len(train_loader), "scheduler_total_steps": steps,
                  "max_fp16_skips": max_skips, "actual_pos_weight": pos_weight.tolist(),
                  "heads": list(heads),
                  "head_pos_weight": {h: actual_pw[h].tolist() for h in heads} if multi else {},
                  "loss": mh.LOSS_RULE if multi else "BCEWithLogitsLoss",
                  "serving": serving_rule,
                  "environment": {k: env.get(k) for k in aug.ENV_KEYS},
                  "environment_matches_baseline": environment_matches,
                  "remote_relaxed": False}
    if smoke:
        provenance["smoke"] = {"train": len(train_ids), "val": len(val_ids),
                               "epochs": epochs, "engineering_only": True}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, a.res, a.res, device=dev))
        model.train()
    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr},
                                   {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                   max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    pw = torch.tensor(np.stack([actual_pw[h] for h in heads]) if multi else pos_weight,
                      dtype=torch.float32, device=dev)
    single_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pw) if not multi else None
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    eval_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.eval_amp]
    scaler = torch.amp.GradScaler("cuda", init_scale=a.amp_init_scale,
                                  enabled=a.amp == "fp16" and dev == "cuda")

    def autocast(dtype):
        return torch.autocast(dev, dtype=dtype) if dtype is not None and dev == "cuda" else contextlib.nullcontext()

    def eval_loader(ids, k_eval):
        dataset = windows(corpus.dir, ids, all_labels, a.res, k_eval, False,
                          a.norm, a.seed, window_policy=a.window_policy,
                          input_policy=a.input_policy)
        return DataLoader(dataset, batch_size=eval_bs, shuffle=False,
                          num_workers=min(a.workers, 2), collate_fn=collate,
                          persistent_workers=False, worker_init_fn=aug.worker_init)

    @torch.no_grad()
    def evaluate(ids, k_eval):
        model.eval()
        outputs, seen = [], []
        for batch in eval_loader(ids, k_eval):
            x, _, uids = move_batch(batch, dev)
            with autocast(eval_dtype):
                logits = model(x)
            expected = (len(uids), len(heads), len(LAB)) if multi else (len(uids), len(LAB))
            if logits.shape != expected or not torch.isfinite(logits).all():
                raise RuntimeError("invalid evaluation logits")
            outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
            seen.extend(uids)
        if seen != ids:
            raise RuntimeError("evaluation UID order changed")
        return np.concatenate(outputs)

    history, skipped, updates = [], 0, 0
    started = time.time()
    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, epoch_start = 0.0, time.time()
        head_tot = np.zeros(len(heads) or 1)
        for step, batch in enumerate(train_loader):
            x, y, uids = move_batch(batch, dev)
            with autocast(amp_dtype):
                logits = model(x)
                expected = (len(uids), len(heads), len(LAB)) if multi else (len(uids), len(LAB))
                if logits.shape != expected or not torch.isfinite(logits).all():
                    raise RuntimeError("invalid training logits")
                if multi:
                    loss, each = mh.masked_multihead_bce(logits.float(), y, pw)
                else:
                    loss = single_loss(logits.float(), y)
                    each = loss.reshape(1)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0,
                                           error_if_nonfinite=a.amp != "fp16")
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if a.amp == "fp16" and dev == "cuda" and scaler.get_scale() < before:
                skipped += 1
                if skipped > max_skips:
                    raise RuntimeError("FP16 skip gate failed")
            else:
                scheduler.step()
            updates += 1
            total += float(loss)
            head_tot += each.detach().float().cpu().numpy()
            if (step + 1) % 200 == 0 or smoke:
                log(f"epoch {epoch} step {step + 1}/{len(train_loader)} loss {total / (step + 1):.4f}")
        record = {"epoch": epoch, "loss": total / len(train_loader),
                  "epoch_seconds": round(time.time() - epoch_start),
                  "optimizer_updates": updates, "fp16_skips": skipped,
                  "head_loss": {name: float(head_tot[j] / len(train_loader))
                                for j, name in enumerate(heads or ("single",))}}
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    cleanup = trace_tools.stop_loader(train_loader)
    if updates != steps:
        raise RuntimeError("optimizer update count differs from registration")
    provenance["fp16_skipped_updates"] = skipped
    checkpoint = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "arch": a.arch, "res": a.res, "lab": LAB, "contract": corpus.contract,
                  "k_eval": a.k_eval, "norm": a.norm, "window_policy": a.window_policy,
                  "input_policy": a.input_policy, "epoch": epochs - 1, "tag": a.run,
                  "amp_train": a.amp, "amp_eval": a.eval_amp, "fold": a.fold,
                  "label_set": spec["label_set"], "provenance": provenance,
                  "init": "radimagenet", "init_weights_sha256": weights_sha}
    if multi:
        checkpoint.update(model_class="exp117 MultiHeadRaptor (tools/exp117_multihead.build_multihead)",
                          heads=list(heads), serving=serving_rule)
    torch.save(checkpoint, out / "model_best.pt")
    model.load_state_dict(checkpoint["model"])
    output_files = []
    evaluation_specs = ((k_extra, protocol["extra_eval"]["file"], f"_k{k_extra}"),) if smoke else (
        (a.k_eval, "oof_best.csv", ""),
        (k_extra, protocol["extra_eval"]["file"], f"_k{k_extra}"))
    for k_eval, filename, suffix in evaluation_specs:
        probs = evaluate(val_ids, k_eval)
        if multi:
            for j, name in enumerate(heads):
                head_file = f"oof_head_{name}{suffix}.csv"
                write_predictions(out / head_file, val_ids, probs[:, j])
                output_files.append(head_file)
            probs = mh.rank_mean(probs)
        write_predictions(out / filename, val_ids, probs)
        output_files.append(filename)
    provenance["cuda_initialized"] = bool(torch.cuda.is_initialized())
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    summary = {"run": a.run, "fold": a.fold, "label_set": spec["label_set"],
               "train": len(train_ids), "val": len(val_ids),
               "rows": len(val_ids), "gold": 0, "selected_checkpoint": "fixed-final",
               "selected_epoch_1based": epochs, "optimizer_updates": updates,
               "skipped_updates": skipped, "args": vars(a), "history": history,
               "environment": env, "worker_cleanup": cleanup,
               "total_seconds": round(time.time() - started), "heads": list(heads),
               "weak_oof_macro_auc": None, "gold_access": "disabled",
               "mode": a.mode, "oof_files": ({f"k{k_extra}": protocol["extra_eval"]["file"]} if smoke else
                   {"k24": "oof_best.csv", f"k{k_extra}": protocol["extra_eval"]["file"]}),
               "provenance": provenance}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", *output_files]
    (out / "output-inventory.json").write_text(json.dumps({
        name: {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size}
        for name in names}, indent=1), encoding="utf-8")
    log(f"done in {summary['total_seconds']}s -> {out}")
    log_file.close()


if __name__ == "__main__":
    main()
