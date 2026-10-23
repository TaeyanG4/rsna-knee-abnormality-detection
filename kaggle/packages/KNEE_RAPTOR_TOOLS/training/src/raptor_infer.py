"""Run a Raptor checkpoint on a local corpus and write prediction CSVs.

Used for (1) parity: public checkpoints on our corpus must reproduce their published
Gold58 numbers, (2) precision checks: fp16 (Kaggle T4) vs bf16/fp32 (local), and
(3) OOF/test predictions of our own checkpoints. Output CSVs follow the app's
prediction format (StudyInstanceUID + 12 finding columns), so 예측 비교 can read them.

  python training/src/raptor_infer.py --arm maxspan-v5 --subset gold --amp fp16
  python training/src/raptor_infer.py --ckpt artifacts/experiments/X/model.pt --subset oof --fold 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raptor_common import (CACHE, COMP, CONTRACTS, EXPERIMENTS, LAB, PUBLIC_ARMS, ROOT, UID, WEIGHTS, Corpus,  # noqa: E402
                           environment, eval_centers, gold_labels, load_checkpoint, load_folds, macro_auc,
                           windows_from, write_predictions)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def subset_ids(corpus, subset, fold, fold_file):
    import pandas as pd
    if subset == "gold":
        return [u for u in gold_labels() if u in corpus]
    if subset == "test":
        return [u for u in pd.read_csv(COMP / "test.csv", dtype={UID: str})[UID] if u in corpus]
    if subset == "all":
        return list(corpus.ids)
    if subset == "oof":
        folds, _ = load_folds(Path(fold_file))
        gold = set(gold_labels())
        return [u for u in corpus.ids if folds.get(u, -1) == fold and u not in gold]
    raise ValueError(subset)


def predict(model, corpus, ids, k_eval, res, amp, device, norm="imagenet", batch=1, log=print):
    import torch
    model.eval().to(device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[amp]
    probs = np.zeros((len(ids), len(LAB)), np.float32)
    t0 = time.time(); fallbacks = 0
    for i, uid in enumerate(ids):
        vol, mask = corpus.volume(uid), corpus.mask(uid)
        x = windows_from(vol, eval_centers(mask, corpus.depth, k_eval), res)
        if norm == "imagenet":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        xt = torch.from_numpy(x)[None].to(device)
        with torch.no_grad():
            try:
                if dtype is None:
                    out = model(xt)
                else:
                    with torch.autocast(device, dtype=dtype):
                        out = model(xt)
            except RuntimeError:
                fallbacks += 1; torch.cuda.empty_cache()
                out = model(xt)
            probs[i] = torch.sigmoid(out.float())[0].cpu().numpy()
        if (i + 1) % 50 == 0 or i + 1 == len(ids):
            log(f"  {i + 1}/{len(ids)} studies, {time.time() - t0:.0f}s")
    return probs, {"seconds": round(time.time() - t0, 1), "per_study_s": round((time.time() - t0) / max(1, len(ids)), 3), "fp32_fallbacks": fallbacks}


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
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"], help="fp16 = Kaggle T4 path")
    ap.add_argument("--out", default=None, help="output CSV (default artifacts/predictions/raptor/<name>_<subset>_<amp>.csv)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="cpu lets a check run while training holds the GPU")
    a = ap.parse_args()
    import torch
    torch.backends.cudnn.benchmark = True
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if device == "cpu" and a.amp == "fp16":
        a.amp = "none"  # CPU has no fp16 autocast path worth using; fp32 is the reference

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
    model, meta = load_checkpoint(ckpt_path)
    res = int(meta.get("res", 384))
    ids = subset_ids(corpus, a.subset, a.fold, a.fold_file)
    if a.limit:
        ids = ids[:a.limit]
    print(f"{name} | contract {contract} | res {res} | k_eval {k_eval} | amp {a.amp} | {a.subset} {len(ids)} studies | {device}", flush=True)
    probs, timing = predict(model, corpus, ids, k_eval, res, a.amp, device, log=lambda m: print(m, flush=True))
    out = Path(a.out) if a.out else ROOT / "artifacts" / "predictions" / "raptor" / f"{name}_{a.subset}_{a.amp}.csv"
    write_predictions(out, ids, probs)
    report = {"name": name, "checkpoint": str(ckpt_path.relative_to(ROOT)) if ckpt_path.is_relative_to(ROOT) else str(ckpt_path),
              "checkpoint_meta": {k: (v if isinstance(v, (int, float, str, list)) else str(v)) for k, v in meta.items() if k != "aucs"},
              "contract": contract, "corpus": str(corpus.dir), "subset": a.subset, "fold": a.fold if a.subset == "oof" else None,
              "studies": len(ids), "k_eval": k_eval, "res": res, "amp": a.amp, "timing": timing, "environment": environment(), "csv": str(out)}
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
