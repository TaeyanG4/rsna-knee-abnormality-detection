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
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
SELECTED = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]

BASE_DIR = ROOT / "experiments" / "EXP043B_CONVNEXT_S_REG_FOLD0"
SPEC_DIR = ROOT / "experiments" / "EXP051_CONVNEXT_S_WEAK4_FOLD0"
OUT_DIR = ROOT / "experiments" / "EXP052_SPECIALIST_FOLD0_AUDIT"


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True).to_numpy(np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float(np.corrcoef(rank01(a), rank01(b))[0, 1])
    return pearson, spearman


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def aligned_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={UID: str})
    if df[UID].duplicated().any():
        raise RuntimeError(f"duplicate UID in {path}")
    return df.set_index(UID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(BASE_DIR.relative_to(ROOT)))
    ap.add_argument("--specialist-dir", default=str(SPEC_DIR.relative_to(ROOT)))
    ap.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = ap.parse_args()
    base_dir = ROOT / args.base_dir
    spec_dir = ROOT / args.specialist_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weak = pd.read_csv(
        ROOT / "labels" / "flight0234__rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        dtype={UID: str},
    ).set_index(UID)
    official = pd.read_csv(ROOT / "data" / "train.csv", dtype={UID: str}).set_index(UID)

    base_w = aligned_pred(base_dir / "weak_val_pred.csv")
    spec_w = aligned_pred(spec_dir / "weak_val_pred.csv")
    if set(base_w.index) != set(spec_w.index):
        raise RuntimeError("weak validation UID sets differ")
    uids_w = base_w.index
    spec_w = spec_w.loc[uids_w]

    base_g = aligned_pred(base_dir / "gold_pred.csv")
    spec_g = aligned_pred(spec_dir / "gold_pred.csv")
    if set(base_g.index) != set(spec_g.index):
        raise RuntimeError("gold UID sets differ")
    uids_g = base_g.index
    spec_g = spec_g.loc[uids_g]

    weak_rows: list[dict] = []
    gold_rows: list[dict] = []
    for target in SELECTED:
        y_soft = weak.loc[uids_w, target].to_numpy(np.float64)
        conf = weak.loc[uids_w, target + "__conf"].to_numpy(np.float64)
        use = ((y_soft >= 0.7) | (y_soft <= 0.3)) & (conf >= 0.6)
        y = (y_soft[use] >= 0.7).astype(np.uint8)
        bp = base_w.loc[uids_w, target].to_numpy(np.float64)[use]
        sp = spec_w.loc[uids_w, target].to_numpy(np.float64)[use]
        pc, sc = corr(bp, sp)
        weak_rows.append(
            {
                "target": target,
                "n": int(use.sum()),
                "positive": int(y.sum()),
                "base_auc": auc(y, bp),
                "specialist_auc": auc(y, sp),
                "delta_specialist_vs_base": auc(y, sp) - auc(y, bp),
                "prob_blend50_auc": auc(y, 0.5 * bp + 0.5 * sp),
                "rank_blend50_auc": auc(y, 0.5 * rank01(bp) + 0.5 * rank01(sp)),
                "pearson": pc,
                "spearman": sc,
            }
        )

        gy = official.loc[uids_g, target].to_numpy(np.uint8)
        bg = base_g.loc[uids_g, target].to_numpy(np.float64)
        sg = spec_g.loc[uids_g, target].to_numpy(np.float64)
        gpc, gsc = corr(bg, sg)
        gold_rows.append(
            {
                "target": target,
                "n": int(len(gy)),
                "positive": int(gy.sum()),
                "base_auc": auc(gy, bg),
                "specialist_auc": auc(gy, sg),
                "delta_specialist_vs_base": auc(gy, sg) - auc(gy, bg),
                "prob_blend50_auc": auc(gy, 0.5 * bg + 0.5 * sg),
                "rank_blend50_auc": auc(gy, 0.5 * rank01(bg) + 0.5 * rank01(sg)),
                "pearson": gpc,
                "spearman": gsc,
            }
        )

    wdf = pd.DataFrame(weak_rows)
    gdf = pd.DataFrame(gold_rows)
    wdf.to_csv(out_dir / "weak_selected_targets.csv", index=False)
    gdf.to_csv(out_dir / "gold_selected_targets_diagnostic.csv", index=False)

    summary = {
        "contract": "exp052_specialist_fold_audit_v2",
        "base": str(base_dir.relative_to(ROOT)),
        "specialist": str(spec_dir.relative_to(ROOT)),
        "selected_targets": SELECTED,
        "selection_surface": "fold-local weak high-confidence report labels only",
        "weak": {
            "base_macro_auc": float(wdf.base_auc.mean()),
            "specialist_macro_auc": float(wdf.specialist_auc.mean()),
            "delta": float(wdf.specialist_auc.mean() - wdf.base_auc.mean()),
            "prob_blend50_macro_auc": float(wdf.prob_blend50_auc.mean()),
            "rank_blend50_macro_auc": float(wdf.rank_blend50_auc.mean()),
            "targets_improved": int((wdf.delta_specialist_vs_base > 0).sum()),
        },
        "gold_diagnostic_once": {
            "base_macro_auc": float(gdf.base_auc.mean()),
            "specialist_macro_auc": float(gdf.specialist_auc.mean()),
            "delta": float(gdf.specialist_auc.mean() - gdf.base_auc.mean()),
            "prob_blend50_macro_auc": float(gdf.prob_blend50_auc.mean()),
            "rank_blend50_macro_auc": float(gdf.rank_blend50_auc.mean()),
            "targets_improved": int((gdf.delta_specialist_vs_base > 0).sum()),
            "warning": "Gold58 is diagnostic only and must not be used to tune weights or targets.",
        },
        "standalone_specialist_gate": {
            "rule": "standalone specialist passes only if weak selected-target macro improves >=0.002 and at least 3/4 targets improve",
            "pass": bool(
                (wdf.specialist_auc.mean() - wdf.base_auc.mean() >= 0.002)
                and ((wdf.delta_specialist_vs_base > 0).sum() >= 3)
            ),
        },
        "fixed_blend_diversity_gate": {
            "rule": "pre-specified 50/50 probability blend passes if weak selected-target macro improves >=0.002 and at least 3/4 targets beat base",
            "delta": float(wdf.prob_blend50_auc.mean() - wdf.base_auc.mean()),
            "targets_improved": int((wdf.prob_blend50_auc > wdf.base_auc).sum()),
            "pass": bool(
                (wdf.prob_blend50_auc.mean() - wdf.base_auc.mean() >= 0.002)
                and ((wdf.prob_blend50_auc > wdf.base_auc).sum() >= 3)
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nWEAK FOLD")
    print(wdf.to_string(index=False))
    print("\nGOLD DIAGNOSTIC")
    print(gdf.to_string(index=False))


if __name__ == "__main__":
    main()
