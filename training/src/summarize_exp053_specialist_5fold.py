from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = workspace_root()
UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
SELECTED = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]
BASE_DIRS = [
    ROOT / "experiments" / "EXP043B_CONVNEXT_S_REG_FOLD0",
    *[ROOT / "experiments" / f"EXP044_CONVNEXT_S_FOLD{i}" for i in range(1, 5)],
]
SPEC_DIRS = [
    ROOT / "experiments" / "EXP051_CONVNEXT_S_WEAK4_FOLD0",
    *[ROOT / "experiments" / f"EXP053_CONVNEXT_S_WEAK4_FOLD{i}" for i in range(1, 5)],
]
OUT = ROOT / "experiments" / "EXP053_SPECIALIST_5FOLD"


def read_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={UID: str})
    if df[UID].duplicated().any():
        raise RuntimeError(f"duplicate UID: {path}")
    return df.set_index(UID)


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True).to_numpy(float)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def weak_eval(y_soft: np.ndarray, conf: np.ndarray, pred: np.ndarray) -> float:
    use = ((y_soft >= 0.7) | (y_soft <= 0.3)) & (conf >= 0.6)
    y = (y_soft[use] >= 0.7).astype(np.uint8)
    return auc(y, pred[use])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in SPEC_DIRS + BASE_DIRS if not (p / "weak_val_pred.csv").is_file()]
    if missing:
        raise FileNotFoundError("missing completed fold predictions: " + ", ".join(missing))

    weak = pd.read_csv(
        ROOT / "labels" / "flight0234__rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        dtype={UID: str},
    ).set_index(UID)
    official = pd.read_csv(ROOT / "data" / "train.csv", dtype={UID: str}).set_index(UID)

    fold_rows = []
    fused_oof_parts = []
    for fold, (bd, sd) in enumerate(zip(BASE_DIRS, SPEC_DIRS)):
        b = read_pred(bd / "weak_val_pred.csv")
        s = read_pred(sd / "weak_val_pred.csv")
        if set(b.index) != set(s.index):
            raise RuntimeError(f"fold {fold}: validation UID sets differ")
        s = s.loc[b.index]
        fused = b.copy()
        for t in SELECTED:
            y = weak.loc[b.index, t].to_numpy(float)
            c = weak.loc[b.index, t + "__conf"].to_numpy(float)
            bp = b[t].to_numpy(float)
            sp = s[t].to_numpy(float)
            pp = 0.5 * bp + 0.5 * sp
            rp = 0.5 * rank01(bp) + 0.5 * rank01(sp)
            fold_rows.append({
                "fold": fold,
                "target": t,
                "base_auc": weak_eval(y, c, bp),
                "specialist_auc": weak_eval(y, c, sp),
                "prob_blend50_auc": weak_eval(y, c, pp),
                "rank_blend50_auc": weak_eval(y, c, rp),
                "pearson": float(np.corrcoef(bp, sp)[0, 1]),
                "spearman": float(np.corrcoef(rank01(bp), rank01(sp))[0, 1]),
            })
            fused[t] = pp
        fused_oof_parts.append(fused.reset_index())

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "per_fold_selected_targets.csv", index=False)
    fold_macro = fold_df.groupby("fold")[["base_auc", "specialist_auc", "prob_blend50_auc", "rank_blend50_auc"]].mean().reset_index()
    fold_macro.to_csv(OUT / "per_fold_selected_macro.csv", index=False)

    fused_oof = pd.concat(fused_oof_parts, ignore_index=True).set_index(UID)
    if fused_oof.index.duplicated().any():
        raise RuntimeError("OOF contains duplicate UIDs")

    base_oof_parts = [read_pred(p / "weak_val_pred.csv").reset_index() for p in BASE_DIRS]
    spec_oof_parts = [read_pred(p / "weak_val_pred.csv").reset_index() for p in SPEC_DIRS]
    base_oof = pd.concat(base_oof_parts, ignore_index=True).set_index(UID).loc[fused_oof.index]
    spec_oof = pd.concat(spec_oof_parts, ignore_index=True).set_index(UID).loc[fused_oof.index]

    pooled_rows = []
    full_base_aucs = []
    full_fused_aucs = []
    for t in TARGETS:
        y = weak.loc[fused_oof.index, t].to_numpy(float)
        c = weak.loc[fused_oof.index, t + "__conf"].to_numpy(float)
        ba = weak_eval(y, c, base_oof[t].to_numpy(float))
        fa = weak_eval(y, c, fused_oof[t].to_numpy(float))
        full_base_aucs.append(ba)
        full_fused_aucs.append(fa)
        row = {"target": t, "base_auc": ba, "fused_auc": fa, "delta": fa - ba}
        if t in SELECTED:
            sa = weak_eval(y, c, spec_oof[t].to_numpy(float))
            row["specialist_auc"] = sa
        pooled_rows.append(row)
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(OUT / "pooled_oof_per_target.csv", index=False)
    fused_oof.reset_index().to_csv(OUT / "fused_oof_pred.csv", index=False)

    # Gold58 is diagnostic only: average fold predictions first, then score once.
    gold_uids = read_pred(BASE_DIRS[0] / "gold_pred.csv").index
    bg = np.mean(np.stack([read_pred(p / "gold_pred.csv").loc[gold_uids, TARGETS].to_numpy(float) for p in BASE_DIRS]), axis=0)
    sg = np.mean(np.stack([read_pred(p / "gold_pred.csv").loc[gold_uids, TARGETS].to_numpy(float) for p in SPEC_DIRS]), axis=0)
    fg = bg.copy()
    for t in SELECTED:
        j = TARGETS.index(t)
        fg[:, j] = 0.5 * bg[:, j] + 0.5 * sg[:, j]
    gy = official.loc[gold_uids, TARGETS].to_numpy(np.uint8)
    gold_rows = []
    for j, t in enumerate(TARGETS):
        row = {
            "target": t,
            "base_auc": auc(gy[:, j], bg[:, j]),
            "fused_auc": auc(gy[:, j], fg[:, j]),
        }
        row["delta"] = row["fused_auc"] - row["base_auc"]
        if t in SELECTED:
            row["specialist_auc"] = auc(gy[:, j], sg[:, j])
        gold_rows.append(row)
    gold_df = pd.DataFrame(gold_rows)
    gold_df.to_csv(OUT / "gold_ensemble_diagnostic_per_target.csv", index=False)

    selected_fold_base = float(fold_macro.base_auc.mean())
    selected_fold_fused = float(fold_macro.prob_blend50_auc.mean())
    selected_pooled = pooled_df[pooled_df.target.isin(SELECTED)]
    summary = {
        "contract": "exp053_specialist_5fold_fixed_blend_v1",
        "selected_targets": SELECTED,
        "blend": "selected targets only: 0.5 full ConvNeXt-S + 0.5 target specialist; all other targets unchanged",
        "selection_surface": "5-fold weak OOF using pre-specified fixed 50/50 blend",
        "fold_mean_selected": {
            "base_macro_auc": selected_fold_base,
            "prob_blend50_macro_auc": selected_fold_fused,
            "delta": selected_fold_fused - selected_fold_base,
            "fold_deltas": (fold_macro.prob_blend50_auc - fold_macro.base_auc).tolist(),
        },
        "pooled_oof": {
            "base_full12_macro_auc": float(np.mean(full_base_aucs)),
            "fused_full12_macro_auc": float(np.mean(full_fused_aucs)),
            "delta_full12": float(np.mean(full_fused_aucs) - np.mean(full_base_aucs)),
            "base_selected4_macro_auc": float(selected_pooled.base_auc.mean()),
            "fused_selected4_macro_auc": float(selected_pooled.fused_auc.mean()),
            "delta_selected4": float(selected_pooled.fused_auc.mean() - selected_pooled.base_auc.mean()),
        },
        "gold58_diagnostic_only": {
            "base_full12_macro_auc": float(gold_df.base_auc.mean()),
            "fused_full12_macro_auc": float(gold_df.fused_auc.mean()),
            "delta_full12": float(gold_df.fused_auc.mean() - gold_df.base_auc.mean()),
            "warning": "Do not tune blend weights or target set on Gold58.",
        },
        "promotion_gate": {
            "rule": "promote if all five fold deltas are positive and pooled full12 OOF delta >= 0.002",
            "pass": bool(((fold_macro.prob_blend50_auc - fold_macro.base_auc) > 0).all() and (np.mean(full_fused_aucs) - np.mean(full_base_aucs) >= 0.002)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nFOLD MACRO")
    print(fold_macro.to_string(index=False))
    print("\nPOOLED OOF")
    print(pooled_df.to_string(index=False))
    print("\nGOLD DIAGNOSTIC")
    print(gold_df.to_string(index=False))


if __name__ == "__main__":
    main()
