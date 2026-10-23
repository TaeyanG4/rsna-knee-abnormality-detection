from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = workspace_root()
UID = "StudyInstanceUID"
SELECTED = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]


def load_pred(folder: str) -> pd.DataFrame:
    p = ROOT / folder / "weak_val_pred.csv"
    df = pd.read_csv(p, dtype={UID: str})
    if df[UID].duplicated().any():
        raise RuntimeError(f"duplicate UID in {p}")
    return df.set_index(UID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dirs", nargs="+", required=True)
    ap.add_argument("--specialist-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()
    if len(args.base_dirs) != len(args.specialist_dirs):
        raise ValueError("base/specialist fold counts differ")

    teacher = pd.read_csv(
        ROOT / "labels" / "flight0234__rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        dtype={UID: str},
    ).set_index(UID)

    parts = []
    for fold, (bd, sd) in enumerate(zip(args.base_dirs, args.specialist_dirs)):
        b = load_pred(bd)
        s = load_pred(sd)
        if set(b.index) != set(s.index):
            raise RuntimeError(f"fold {fold} UID mismatch")
        s = s.loc[b.index]
        x = pd.DataFrame(index=b.index)
        x["fold"] = fold
        for t in SELECTED:
            x[t + "__base"] = b[t]
            x[t + "__spec"] = s[t]
        parts.append(x)
    pred = pd.concat(parts, axis=0)
    if pred.index.duplicated().any():
        raise RuntimeError("duplicate OOF UID across folds")

    rows = []
    prepared = {}
    for t in SELECTED:
        y_soft = teacher.loc[pred.index, t].to_numpy(float)
        conf = teacher.loc[pred.index, t + "__conf"].to_numpy(float)
        use = ((y_soft >= 0.7) | (y_soft <= 0.3)) & (conf >= 0.6)
        y = (y_soft[use] >= 0.7).astype(np.uint8)
        b = pred.loc[pred.index, t + "__base"].to_numpy(float)[use]
        s = pred.loc[pred.index, t + "__spec"].to_numpy(float)[use]
        blend = 0.5 * b + 0.5 * s
        prepared[t] = {"use": use, "y": y, "base": b, "spec": s, "blend": blend}
        ba = float(roc_auc_score(y, b))
        sa = float(roc_auc_score(y, s))
        aa = float(roc_auc_score(y, blend))
        rows.append(
            {
                "target": t,
                "n": int(use.sum()),
                "positive": int(y.sum()),
                "base_auc": ba,
                "specialist_auc": sa,
                "blend50_auc": aa,
                "delta_blend_vs_base": aa - ba,
                "pearson": float(np.corrcoef(b, s)[0, 1]),
            }
        )

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pooled_oof_selected_targets.csv", index=False)

    # Paired study bootstrap: resample studies once per replicate, then score all
    # selected targets on the same sampled study indices. This keeps the
    # candidate-vs-base comparison paired and avoids using Gold58 for uncertainty.
    rng = np.random.default_rng(args.seed)
    n_study = len(pred)
    boot = []
    for _ in range(max(0, args.bootstrap)):
        sampled = rng.integers(0, n_study, size=n_study)
        deltas = []
        for t in SELECTED:
            d = prepared[t]
            use_sampled = d["use"][sampled]
            pos = sampled[use_sampled]
            if len(pos) < 2:
                continue
            # Map full-study positions into this target's compact high-confidence arrays.
            compact_index = np.full(n_study, -1, dtype=np.int32)
            compact_index[np.flatnonzero(d["use"])] = np.arange(len(d["y"]), dtype=np.int32)
            ci = compact_index[pos]
            yb = d["y"][ci]
            if np.unique(yb).size != 2:
                continue
            base_auc = roc_auc_score(yb, d["base"][ci])
            blend_auc = roc_auc_score(yb, d["blend"][ci])
            deltas.append(float(blend_auc - base_auc))
        if len(deltas) == len(SELECTED):
            boot.append(float(np.mean(deltas)))
    boot_arr = np.asarray(boot, dtype=float)
    if len(boot_arr):
        bootstrap = {
            "requested": int(args.bootstrap),
            "valid": int(len(boot_arr)),
            "seed": int(args.seed),
            "mean_delta": float(boot_arr.mean()),
            "std_delta": float(boot_arr.std(ddof=1)) if len(boot_arr) > 1 else 0.0,
            "ci95_low": float(np.quantile(boot_arr, 0.025)),
            "ci95_high": float(np.quantile(boot_arr, 0.975)),
            "fraction_positive": float((boot_arr > 0).mean()),
            "unit": "study",
        }
    else:
        bootstrap = {"requested": int(args.bootstrap), "valid": 0, "seed": int(args.seed)}
    summary = {
        "contract": "exp056_specialist_pooled_oof_audit_v2",
        "folds": len(parts),
        "rows": int(len(pred)),
        "selected_targets": SELECTED,
        "base_macro_auc": float(df.base_auc.mean()),
        "specialist_macro_auc": float(df.specialist_auc.mean()),
        "blend50_macro_auc": float(df.blend50_auc.mean()),
        "delta_blend_vs_base": float(df.blend50_auc.mean() - df.base_auc.mean()),
        "targets_improved_by_blend": int((df.delta_blend_vs_base > 0).sum()),
        "paired_study_bootstrap_delta": bootstrap,
        "selection_surface": "pooled weak-label OOF only; no Gold58 tuning",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
