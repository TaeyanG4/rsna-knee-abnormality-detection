from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = workspace_root()
UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
SWITCH = ["ACL", "MCL", "Effusion", "Baker's", "Fracture"]


def main() -> None:
    out = ROOT / "labels" / "derived_EXP055"
    out.mkdir(parents=True, exist_ok=True)
    flight_path = ROOT / "labels" / "flight0234__rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv"
    dread_path = ROOT / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv"
    train_path = ROOT / "data" / "train.csv"

    flight = pd.read_csv(flight_path, dtype={UID: str}).set_index(UID)
    dread = pd.read_csv(dread_path, dtype={UID: str}).set_index(UID)
    official = pd.read_csv(train_path, dtype={UID: str}).set_index(UID)
    hybrid = flight.copy()

    common = hybrid.index.intersection(dread.index)
    for t in SWITCH:
        p = dread.loc[common, t].astype(float)
        hybrid.loc[common, t] = p
        # Dread's probabilities encode uncertainty directly. Use distance from 0.5
        # only as an auxiliary sample weight, while preserving the soft target.
        hybrid.loc[common, t + "__conf"] = np.clip(2.0 * np.abs(p - 0.5), 0.05, 1.0)

    # Gold58 is never used by the fold trainer, but keeping Flight values for the
    # missing Dread rows lets the file remain row-complete and makes provenance explicit.
    gold = official[TARGETS].notna().all(axis=1)
    gold_uids = set(official.index[gold])
    out_path = out / "flight_dread_severity_hybrid_v1.csv"
    hybrid.reset_index().to_csv(out_path, index=False)

    stats = []
    for t in SWITCH:
        f = flight.loc[common, t].to_numpy(float)
        d = dread.loc[common, t].to_numpy(float)
        stats.append({
            "target": t,
            "common_rows": int(len(common)),
            "flight_mean": float(f.mean()),
            "dread_mean": float(d.mean()),
            "mean_delta_dread_minus_flight": float((d - f).mean()),
            "spearman": float(pd.Series(f).corr(pd.Series(d), method="spearman")),
        })
    pd.DataFrame(stats).to_csv(out / "switched_target_stats.csv", index=False)

    receipt = {
        "contract": "exp055_flight_dread_severity_hybrid_v1",
        "rows": int(len(hybrid)),
        "dread_rows": int(len(dread)),
        "common_rows": int(len(common)),
        "gold_rows": int(len(gold_uids)),
        "switched_targets": SWITCH,
        "switch_rule": "Use Dread soft target on rows present in Dread; confidence = clip(2*abs(p-0.5), 0.05, 1.0). Other targets remain Flight v4hybrid.",
        "gold_handling": "Dread excludes Gold58. Hybrid retains Flight values on those rows only for row completeness; training folds exclude Gold58 and quarantined report duplicates.",
        "selection_rationale": "Chosen before student training from host severity definitions plus train-only EXP054 label-contract audit; no Gold58 target-specific student result used.",
        "flight_source": str(flight_path.relative_to(ROOT)),
        "dread_source": str(dread_path.relative_to(ROOT)),
        "output": str(out_path.relative_to(ROOT)),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
