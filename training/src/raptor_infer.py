"""Run a Raptor checkpoint on a local corpus and write prediction CSVs.

Used for (1) public-checkpoint input checks on our corpus, (2) precision checks
and same-checkpoint device comparisons (a local FP16 run alone is not a T4 test), and
(3) OOF/test predictions of our own checkpoints. Output CSVs follow the app's
prediction format (StudyInstanceUID + 12 finding columns), so 예측 비교 can read them.

  python training/src/raptor_infer.py --arm maxspan-v5 --subset gold --amp fp16
  python training/src/raptor_infer.py --ckpt artifacts/experiments/X/model.pt --subset oof --fold 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raptor_common import (CACHE, COMP, EXPERIMENTS, PUBLIC_ARMS, ROOT, UID, WEIGHTS, Corpus,  # noqa: E402
                           environment, gold_labels, load_checkpoint, load_folds, macro_auc,
                           sha256_file, write_predictions)
from raptor_runtime import checkpoint_norm, configure_runtime, predict  # noqa: E402


def subset_ids(corpus, subset, fold, fold_file):
    import pandas as pd
    if subset == "gold":
        ids = list(gold_labels())
    elif subset == "test":
        ids = list(pd.read_csv(COMP / "test.csv", dtype={UID: str})[UID])
    elif subset == "all":
        ids = list(corpus.ids)
    elif subset == "oof":
        folds, _ = load_folds(Path(fold_file))
        gold = set(gold_labels())
        ids = [u for u, f in folds.items() if f == fold and u not in gold]
    else:
        raise ValueError(subset)
    missing = [u for u in ids if u not in corpus]
    if missing:
        raise ValueError(f"corpus misses {len(missing)} requested studies: {missing[:3]}")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--arm", choices=sorted(PUBLIC_ARMS), help="public checkpoint under storage/weights")
    src.add_argument("--ckpt", help="our checkpoint (.pt with arch/res/contract)")
    ap.add_argument("--corpus", default=None, help="corpus directory (default from the arm/checkpoint contract)")
    ap.add_argument("--subset", default="gold", choices=["gold", "test", "oof", "all"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fold-file", default=str(EXPERIMENTS / "EXP026_GROUPED_FROZEN_PROBE" / "folds.csv"))
    ap.add_argument("--k-eval", type=int, default=None)
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"], help="use fp16 for the T4 comparison; CPU requires none")
    ap.add_argument("--runtime-policy", default="strict", choices=["strict", "throughput"])
    ap.add_argument("--batch", type=int, default=1, help="study batch size; keep equal on devices being compared")
    ap.add_argument("--out", default=None, help="output CSV (default artifacts/predictions/raptor/<name>_<subset>_<amp>.csv)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="cpu lets a check run while training holds the GPU")
    a = ap.parse_args()
    import torch
    if a.batch < 1 or a.limit < 0 or (a.k_eval is not None and a.k_eval < 1):
        raise SystemExit("batch/k-eval must be positive; limit must be nonnegative")
    runtime = configure_runtime(a.amp, a.device, a.runtime_policy)
    device = runtime["device"]

    if a.arm:
        arm = PUBLIC_ARMS[a.arm]
        ckpt_path = WEIGHTS / arm["file"]; contract = arm["contract"]; name = a.arm
        k_eval = a.k_eval or arm["k_eval"]
    else:
        ckpt_path = Path(a.ckpt); name = ckpt_path.parent.name + "_" + ckpt_path.stem
        meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        contract = meta.get("contract"); k_eval = a.k_eval or int(meta.get("k_eval", 24)); del meta
        if not contract:
            raise SystemExit("checkpoint lacks a corpus contract")
    corpus = Corpus(Path(a.corpus) if a.corpus else CACHE / contract.upper())
    if corpus.contract != contract:
        raise SystemExit(f"corpus contract {corpus.contract} != checkpoint contract {contract}")
    checkpoint_hash = sha256_file(ckpt_path)
    model, meta = load_checkpoint(ckpt_path)
    res = int(meta.get("res", 384))
    norm, norm_source = checkpoint_norm(meta)
    window_policy = meta.get("window_policy", "legacy")
    ids = subset_ids(corpus, a.subset, a.fold, a.fold_file)
    if a.limit:
        ids = ids[:a.limit]
    out = Path(a.out) if a.out else ROOT / "artifacts" / "predictions" / "raptor" / f"{name}_{a.subset}_{a.amp}.csv"
    if out.exists() or out.with_suffix(".json").exists():
        raise SystemExit(f"output already exists; choose a new --out to preserve the previous run: {out}")
    source_hashes = {filename: sha256_file(Path(__file__).parent / filename)
                     for filename in ("raptor_infer.py", "raptor_common.py", "raptor_runtime.py")}
    print(f"{name} | contract {contract} | res {res} | k_eval {k_eval} | norm {norm} ({norm_source}) | windows {window_policy} | amp {a.amp} | batch {a.batch} | {a.subset} {len(ids)} studies | {device}", flush=True)
    probs, timing = predict(model, corpus, ids, k_eval, res, a.amp, device, norm=norm, batch=a.batch,
                            window_policy=window_policy, log=lambda m: print(m, flush=True))
    write_predictions(out, ids, probs)
    report = {"name": name, "checkpoint": str(ckpt_path.relative_to(ROOT)) if ckpt_path.is_relative_to(ROOT) else str(ckpt_path),
              "checkpoint_meta": {k: (v if isinstance(v, (int, float, str, list)) else str(v)) for k, v in meta.items() if k != "aucs"},
              "contract": contract, "corpus": str(corpus.dir), "subset": a.subset, "fold": a.fold if a.subset == "oof" else None,
              "checkpoint_sha256": checkpoint_hash, "corpus_manifest_sha256": sha256_file(corpus.dir / "manifest.json"),
              "source_sha256": source_hashes,
              "studies": len(ids), "k_eval": k_eval, "res": res, "norm": norm, "norm_source": norm_source,
              "window_policy": window_policy, "amp": a.amp, "requested_amp": a.amp, "effective_amp": runtime["effective_amp"],
              "runtime": runtime, "batch_size": a.batch, "timing": timing, "environment": environment(), "csv": str(out)}
    if a.subset == "gold":
        gold = gold_labels(); y = np.stack([gold[u] for u in ids])
        macro, per = macro_auc(y, probs)
        report.update(gold_macro_auc=macro, gold_per_target=per, published_gold_auc=meta.get("gold_auc"))
        print(f"Gold58 macro AUC {macro:.4f} (checkpoint header says {meta.get('gold_auc')})", flush=True)
        for k, v in per.items():
            print(f"   {k:18s} {v:.3f}")
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
