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
TARGETS = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True).to_numpy(float)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialist-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fixed-alpha", type=float, default=0.10,
                    help="Specialist rank weight in the pre-specified bridge diagnostic")
    ap.add_argument("--alpha-grid", default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.3")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    teacher = pd.read_csv(
        ROOT / "labels" / "flight0234__rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        dtype={UID: str},
    ).set_index(UID)
    folds = pd.read_csv(
        ROOT / "experiments" / "EXP026_GROUPED_FROZEN_PROBE" / "folds.csv",
        dtype={UID: str},
    ).set_index(UID)
    raptor = pd.read_csv(
        ROOT / "experiments" / "EXP030_RAPTOR_CANONICAL_HEAD_PSEUDOOOF" / "noncanonical_weak_oof.csv",
        dtype={UID: str},
    ).set_index(UID)

    pieces = []
    for sd in args.specialist_dirs:
        p = pd.read_csv(ROOT / sd / "weak_val_pred.csv", dtype={UID: str}).set_index(UID)
        p = p.loc[~p.index.duplicated(keep="first")]
        p["fold"] = folds.loc[p.index, "fold"].to_numpy(int)
        pieces.append(p[[*TARGETS, "fold"]])
    spec = pd.concat(pieces, axis=0)
    if spec.index.duplicated().any():
        raise RuntimeError("specialist OOF UIDs overlap across supplied fold directories")
    common = spec.index.intersection(raptor.index)
    spec = spec.loc[common]
    raptor = raptor.loc[common]

    rows = []
    prepared = {}
    for target in TARGETS:
        ys = teacher.loc[common, target].to_numpy(float)
        conf = teacher.loc[common, target + "__conf"].to_numpy(float)
        use = ((ys >= 0.7) | (ys <= 0.3)) & (conf >= 0.6)
        y = (ys[use] >= 0.7).astype(np.uint8)
        r = raptor[target].to_numpy(float)[use]
        s = spec[target].to_numpy(float)[use]
        f = spec["fold"].to_numpy(int)[use]
        rr, sr = rank01(r), rank01(s)
        fixed = (1.0 - args.fixed_alpha) * rr + args.fixed_alpha * sr
        compact_index = np.full(len(common), -1, dtype=np.int32)
        compact_index[np.flatnonzero(use)] = np.arange(len(y), dtype=np.int32)
        prepared[target] = {
            "use": use, "compact_index": compact_index,
            "y": y, "r": rr, "s": sr, "fold": f,
        }
        rows.append({
            "target": target,
            "n": int(len(y)),
            "positive": int(y.sum()),
            "raptor_auc": auc(y, rr),
            "specialist_auc": auc(y, sr),
            "fixed_blend_auc": auc(y, fixed),
            "fixed_delta_vs_raptor": auc(y, fixed) - auc(y, rr),
            "pearson_rank": float(np.corrcoef(rr, sr)[0, 1]),
        })

    frame = pd.DataFrame(rows)
    alphas = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    fold_ids = sorted(set(spec["fold"].astype(int)))
    nested_rows = []
    for held in fold_ids:
        train_scores = []
        for alpha in alphas:
            vals = []
            for target in TARGETS:
                d = prepared[target]
                tr = d["fold"] != held
                if np.unique(d["y"][tr]).size == 2:
                    vals.append(auc(d["y"][tr], (1 - alpha) * d["r"][tr] + alpha * d["s"][tr]))
            train_scores.append((float(np.mean(vals)), alpha))
        # Smallest alpha wins ties, a conservative complexity rule.
        best_train, alpha = max(train_scores, key=lambda z: (z[0], -z[1]))
        held_base, held_blend = [], []
        for target in TARGETS:
            d = prepared[target]
            va = d["fold"] == held
            if np.unique(d["y"][va]).size != 2:
                continue
            held_base.append(auc(d["y"][va], d["r"][va]))
            held_blend.append(auc(d["y"][va], (1 - alpha) * d["r"][va] + alpha * d["s"][va]))
        nested_rows.append({
            "held_fold": int(held),
            "selected_alpha": alpha,
            "train_macro": best_train,
            "held_raptor_macro": float(np.mean(held_base)),
            "held_blend_macro": float(np.mean(held_blend)),
            "held_delta": float(np.mean(held_blend) - np.mean(held_base)),
        })

    nested = pd.DataFrame(nested_rows)

    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(max(0, args.bootstrap)):
        sampled = rng.integers(0, len(common), size=len(common))
        deltas = []
        for target in TARGETS:
            d = prepared[target]
            full_pos = sampled[d["use"][sampled]]
            ci = d["compact_index"][full_pos]
            yb = d["y"][ci]
            if len(ci) < 2 or np.unique(yb).size != 2:
                continue
            rb = d["r"][ci]
            sb = d["s"][ci]
            base_auc = auc(yb, rb)
            blend_auc = auc(yb, (1 - args.fixed_alpha) * rb + args.fixed_alpha * sb)
            deltas.append(blend_auc - base_auc)
        if len(deltas) == len(TARGETS):
            boot.append(float(np.mean(deltas)))
    boot_arr = np.asarray(boot, dtype=float)
    if len(boot_arr):
        boot_summary = {
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
        boot_summary = {"requested": int(args.bootstrap), "valid": 0, "seed": int(args.seed)}
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "fixed_bridge_per_target.csv", index=False)
    nested.to_csv(out / "nested_alpha_by_fold.csv", index=False)
    summary = {
        "contract": "exp059_raptor_specialist_bridge_v2",
        "folds": fold_ids,
        "studies": int(len(common)),
        "targets": TARGETS,
        "fixed_specialist_rank_alpha": float(args.fixed_alpha),
        "fixed_raptor_macro": float(frame.raptor_auc.mean()),
        "fixed_blend_macro": float(frame.fixed_blend_auc.mean()),
        "fixed_delta_vs_raptor": float(frame.fixed_delta_vs_raptor.mean()),
        "fixed_targets_improved": int((frame.fixed_delta_vs_raptor > 0).sum()),
        "fixed_paired_study_bootstrap_delta": boot_summary,
        "nested_mean_held_delta": float(nested.held_delta.mean()),
        "nested_selected_alphas": nested.selected_alpha.tolist(),
        "caveat": "Raptor head pseudo-OOF is optimistic/leaky because its backbone saw full weak train; use only as a bridge-direction diagnostic, not absolute validation.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nFIXED")
    print(frame.to_string(index=False))
    print("\nNESTED")
    print(nested.to_string(index=False))


if __name__ == "__main__":
    main()
