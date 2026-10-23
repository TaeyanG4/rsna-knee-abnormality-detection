"""Build a mixed training-label CSV from teacher labels and out-of-fold image predictions.

The discussion evidence (thread 735304) is that mixing properly out-of-fold predictions into
report-derived labels moved single models to 0.950, while in-fold predictions just parrot
the labels. This tool therefore refuses OOF files whose study set is not a clean cover of the
training studies, and records the recipe + hashes so a model trained on the result can be
traced.

Modes:
  --mode blend     label = (1-alpha) * teacher + alpha * oof                   (all cells)
  --mode silence   like blend, but only where the teacher is 'silent' (teacher value <= --silent-max),
                   i.e. fill likely false negatives; confident teacher cells stay
  --mode target    blend only for --targets (e.g. "Synovitis" "Lateral OA" "PF OA")

  python training/src/label_mix.py --oof artifacts/predictions/EXP061_oof_best.csv --alpha 0.5 --mode blend --out storage/labels/derived_mix/mix_a50.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raptor_common import LAB, ROOT, UID, gold_labels, load_label_csv, macro_auc, sha256_text  # noqa: E402

DEFAULT_TEACHER = ROOT / "storage" / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv"


def mix(teacher: pd.DataFrame, oof: pd.DataFrame, alpha: float, mode: str, targets, silent_max: float):
    t = teacher.set_index(UID)[LAB].astype(float)
    o = oof.set_index(UID)[LAB].astype(float)
    common = t.index.intersection(o.index)
    out = t.copy()
    cols = targets if mode == "target" else LAB
    a = alpha
    tt, oo = t.loc[common, cols], o.loc[common, cols]
    if mode == "silence":
        mask = tt <= silent_max
        mixed = tt.where(~mask, (1 - a) * tt + a * oo)
    else:
        mixed = (1 - a) * tt + a * oo
    out.loc[common, cols] = mixed.clip(0, 1)
    return out.reset_index(), len(common)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    ap.add_argument("--oof", required=True, help="stitched OOF CSV (raptor_report.py output) covering the training studies")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mode", default="blend", choices=["blend", "silence", "target"])
    ap.add_argument("--targets", nargs="*", default=["Synovitis", "Lateral OA", "PF OA"])
    ap.add_argument("--silent-max", type=float, default=0.1, help="teacher values at or below this count as 'not addressed'")
    ap.add_argument("--min-cover", type=float, default=0.95, help="minimum fraction of teacher studies the OOF must cover")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    teacher = pd.read_csv(a.teacher, dtype={UID: str})
    oof = pd.read_csv(a.oof, dtype={UID: str})
    gold = gold_labels()
    if oof[UID].duplicated().any():
        raise SystemExit("OOF has duplicate studies; stitch folds without overlap first")
    leak = [u for u in oof[UID] if u in gold]
    if leak:
        raise SystemExit(f"OOF contains {len(leak)} Gold58 studies; gold must never enter training labels")
    cover = oof[UID].isin(teacher[UID]).sum() / len(teacher)
    if cover < a.min_cover:
        raise SystemExit(f"OOF covers only {cover:.1%} of teacher studies (< {a.min_cover:.0%}); run all folds first")
    mixed, n = mix(teacher, oof, a.alpha, a.mode, a.targets, a.silent_max)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(out, index=False)
    # Diagnostics on Gold58 overlap are impossible by construction (gold excluded); report label movement instead.
    delta = (mixed.set_index(UID)[LAB] - teacher.set_index(UID)[LAB]).abs()
    recipe = {"teacher": str(Path(a.teacher)), "teacher_hash": load_label_csv(Path(a.teacher))[1], "oof": str(Path(a.oof)),
              "oof_hash": sha256_text(oof.to_csv(index=False)), "alpha": a.alpha, "mode": a.mode, "targets": a.targets if a.mode == "target" else LAB,
              "silent_max": a.silent_max, "studies_mixed": int(n), "mean_abs_change": {k: round(float(v), 4) for k, v in delta.mean().items()},
              "cells_changed": int((delta > 1e-9).sum().sum()), "output_hash": load_label_csv(out)[1]}
    out.with_suffix(".json").write_text(json.dumps(recipe, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in recipe.items() if k not in ("mean_abs_change",)}, indent=1))
    print("mean |change| per target:", recipe["mean_abs_change"])


if __name__ == "__main__":
    main()
