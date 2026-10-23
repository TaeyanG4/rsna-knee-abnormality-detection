"""Fixed, low-DOF Gold58 ensemble screen for public image components.

Weights are predeclared coarse candidates. This is a diagnostic because some
public components have Gold58 selection exposure; it must not be treated as an
unbiased model-selection benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from label_audit import ROOT, TARGETS, UID, auc_vector
from public_prediction_audit import paired_delta


def load_csv(path: Path, ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, dtype={UID: str})
    if UID not in frame or frame[UID].duplicated().any():
        raise ValueError(f"bad UID column: {path}")
    frame = frame.set_index(UID).loc[ids, TARGETS]
    out = frame.to_numpy(float)
    if out.shape != (58, 12) or not np.isfinite(out).all():
        raise ValueError(f"bad predictions: {path}")
    return out


def pct_rank(p: np.ndarray) -> np.ndarray:
    return rankdata(p, method="average", axis=0) / len(p)


def main() -> None:
    gold = pd.read_csv(ROOT / "gold" / "gold_labels.csv", dtype={UID: str}).set_index(UID)
    ids = gold.index.tolist(); y = gold[TARGETS].to_numpy(float)
    raptor = load_csv(ROOT / "predictions" / "EXP006A_kaggle_v1" / "exp006a_original.csv", ids)
    coat = load_csv(ROOT / "predictions" / "CoAt_top3_probability_gold58_public.csv", ids)
    dino = load_csv(ROOT / "predictions" / "Pilkwang_public_OOF_gold58_public.csv", ids)

    rr, cr, dr = map(pct_rank, (raptor, coat, dino))
    candidates = {
        "raptor_v5": raptor,
        "coat_top3_prob": coat,
        "dino_public_oof": dino,
        "prob_r75_c25": .75 * raptor + .25 * coat,
        "prob_r50_c50": .50 * raptor + .50 * coat,
        "rank_r75_c25": .75 * rr + .25 * cr,
        "rank_r50_c50": .50 * rr + .50 * cr,
        "rank_r75_c12.5_d12.5": .75 * rr + .125 * cr + .125 * dr,
    }
    rows = []
    summary = {"experiment": "ENSEMBLE_SCREEN_001", "n_gold": 58,
               "baseline": "raptor_v5", "selection_warning":
               "CoAt checkpoints were selected on Gold58; deltas are conditional diagnostics."}
    for name, pred in candidates.items():
        auc = auc_vector(y, pred)
        delta = None if name == "raptor_v5" else paired_delta(y, raptor, pred, repeats=3000)
        rows.append({"model": name, "macro_auc": float(auc.mean()),
                     **{t: float(auc[j]) for j, t in enumerate(TARGETS)}})
        summary[name] = {"macro_auc": float(auc.mean()), "paired_delta_vs_raptor": delta}

    corr = []
    for other_name, other in [("coat_top3_prob", coat), ("dino_public_oof", dino)]:
        for j, target in enumerate(TARGETS):
            corr.append({"pair": f"raptor_v5__{other_name}", "target": target,
                         "pearson": float(np.corrcoef(raptor[:, j], other[:, j])[0, 1]),
                         "spearman": float(spearmanr(raptor[:, j], other[:, j]).statistic)})
    pd.DataFrame(rows).sort_values("macro_auc", ascending=False).to_csv(
        ROOT / "experiments" / "gold_ensemble_screen.csv", index=False)
    pd.DataFrame(corr).to_csv(ROOT / "experiments" / "gold_ensemble_component_correlation.csv", index=False)
    (ROOT / "reports" / "gold_ensemble_screen.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["model", "macro_auc"]].sort_values("macro_auc", ascending=False).to_string(index=False))
    for name, item in summary.items():
        if isinstance(item, dict) and item.get("paired_delta_vs_raptor"):
            d = item["paired_delta_vs_raptor"]
            print(name, "delta", d["observed_delta"], "CI", d["ci95_low"], d["ci95_high"],
                  "p+", d["bootstrap_fraction_delta_positive"])


if __name__ == "__main__":
    main()
