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


def auc_if_binary(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")


def main() -> None:
    out = ROOT / "experiments" / "EXP054_LABEL_CONTRACT_AUDIT"
    out.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(
        ROOT / "experiments" / "EXP026_GROUPED_FROZEN_PROBE" / "folds.csv",
        dtype={UID: str},
    )
    eligible = folds[(folds.fold >= 0) & (~folds.is_gold) & (~folds.quarantined_gold_report_duplicate)]
    uids = eligible[UID].astype(str).tolist()

    flight = pd.read_csv(
        ROOT / "labels" / "flight0234__rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        dtype={UID: str},
    ).set_index(UID).loc[uids]
    dread = pd.read_csv(
        ROOT / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv",
        dtype={UID: str},
    ).set_index(UID).loc[uids]
    oof = pd.read_csv(
        ROOT / "experiments" / "EXP044_CONVNEXT_S_5FOLD" / "oof_pred.csv",
        dtype={UID: str},
    ).set_index(UID).loc[uids]

    rows = []
    for t in TARGETS:
        f = flight[t].to_numpy(float)
        fc = flight[t + "__conf"].to_numpy(float)
        d = dread[t].to_numpy(float)
        p = oof[t].to_numpy(float)

        f_use = ((f >= 0.7) | (f <= 0.3)) & (fc >= 0.6)
        d_use = (d >= 0.7) | (d <= 0.3)
        both = f_use & d_use
        fy = (f[f_use] >= 0.7).astype(np.uint8)
        dy = (d[d_use] >= 0.7).astype(np.uint8)
        same = ((f[both] >= 0.7) == (d[both] >= 0.7)) if both.any() else np.array([], bool)

        rows.append(
            {
                "target": t,
                "flight_highconf_n": int(f_use.sum()),
                "flight_highconf_positive_rate": float(fy.mean()) if len(fy) else float("nan"),
                "dread_extreme_n": int(d_use.sum()),
                "dread_extreme_positive_rate": float(dy.mean()) if len(dy) else float("nan"),
                "both_extreme_n": int(both.sum()),
                "both_extreme_hard_agreement": float(same.mean()) if len(same) else float("nan"),
                "non_gold_pearson": float(np.corrcoef(f, d)[0, 1]),
                "non_gold_spearman": float(pd.Series(f).corr(pd.Series(d), method="spearman")),
                "mean_flight": float(f.mean()),
                "mean_dread": float(d.mean()),
                "mean_dread_minus_flight": float((d - f).mean()),
                "dread_lower_by_0p2_fraction": float((d <= f - 0.2).mean()),
                "dread_higher_by_0p2_fraction": float((d >= f + 0.2).mean()),
                "exp044_oof_auc_vs_flight": auc_if_binary(fy, p[f_use]),
                "exp044_oof_auc_vs_dread": auc_if_binary(dy, p[d_use]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_target.csv", index=False)
    summary = {
        "contract": "exp054_label_contract_audit_v1",
        "eligible_non_gold": len(uids),
        "purpose": "Compare two train-only report-derived supervision contracts without using Gold58 for label-source selection.",
        "flight": "flight0234/report_labels_v4hybrid.csv",
        "dread": "dreaddevelopment/labels_llm_soft.csv",
        "macro_exp044_oof_auc_vs_flight": float(df.exp044_oof_auc_vs_flight.mean()),
        "macro_exp044_oof_auc_vs_dread": float(df.exp044_oof_auc_vs_dread.mean()),
        "mean_hard_agreement_on_joint_extremes": float(df.both_extreme_hard_agreement.mean()),
        "mean_spearman": float(df.non_gold_spearman.mean()),
        "note": "OOF-vs-label AUC measures supervision alignment, not true image-label accuracy. Dread has no Gold rows by construction.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
