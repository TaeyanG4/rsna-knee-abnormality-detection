"""Bootstrap stability audit for non-gold-overwritten public report teachers.

This is a diagnostic, not an image-model validation score. It uses a
predeclared conservative switch rule to avoid turning Gold58 into a large
per-target hyperparameter search.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from label_audit import ROOT, TARGETS, UID, load_teacher

REPEATS = 5000
SEED = 20260912
MIN_DELTA = 0.02
MIN_WIN_FREQ = 0.60
MIN_POSITIVE_FRAC = 0.90


def auc1(y: np.ndarray, p: np.ndarray) -> float:
    pos = float(y.sum()); neg = float(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(p, method="average")
    return float(((r * y).sum() - pos * (pos + 1.0) / 2.0) / (pos * neg))


def main() -> None:
    board = pd.read_csv(ROOT / "experiments" / "label_leaderboard.csv")
    board = board[(board["eligible_for_teacher_comparison"] == True) &
                  (board["all_gold_exact_copy_flag"] == False)].copy()  # noqa: E712
    # Deduplicate byte-identical prediction tables before counting wins.
    board = board.drop_duplicates("prediction_sha256", keep="first")
    gold = pd.read_csv(ROOT / "gold" / "gold_labels.csv", dtype={UID: str}).set_index(UID)

    loaded: dict[str, pd.DataFrame] = {}
    common = set(gold.index)
    for _, row in board.iterrows():
        source = str(row["source"])
        df, _ = load_teacher(ROOT / str(row["file"]))
        df = df.set_index(UID)[TARGETS]
        loaded[source] = df
        common &= set(df.dropna().index)
    ids = sorted(common)
    yall = gold.loc[ids, TARGETS].to_numpy(float)
    sources = list(loaded)
    preds = np.stack([loaded[s].loc[ids, TARGETS].to_numpy(float) for s in sources])
    flight_candidates = [i for i, s in enumerate(sources) if s.startswith("flight0234__")]
    if len(flight_candidates) != 1:
        raise RuntimeError(f"Expected one Flight control after dedup, found {flight_candidates}")
    flight_i = flight_candidates[0]

    rng = np.random.default_rng(SEED)
    rows = []
    switches = {}
    for j, target in enumerate(TARGETS):
        y = yall[:, j]
        observed = np.array([auc1(y, preds[i, :, j]) for i in range(len(sources))])
        win = np.zeros(len(sources), np.int64)
        deltas = [[] for _ in sources]
        valid = 0
        for _ in range(REPEATS):
            ix = rng.integers(0, len(ids), len(ids))
            yy = y[ix]
            if yy.min() == yy.max():
                continue
            scores = np.array([auc1(yy, preds[i, ix, j]) for i in range(len(sources))])
            mx = np.nanmax(scores)
            winners = np.flatnonzero(np.isclose(scores, mx, atol=1e-12, rtol=0))
            for w in winners:
                win[w] += 1 / len(winners)
            for i in range(len(sources)):
                deltas[i].append(scores[i] - scores[flight_i])
            valid += 1

        for i, source in enumerate(sources):
            d = np.asarray(deltas[i], float)
            rows.append({
                "target": target, "source": source, "n_common_gold": len(ids),
                "observed_auc": observed[i], "observed_delta_vs_flight": observed[i] - observed[flight_i],
                "bootstrap_valid": valid, "bootstrap_win_frequency": float(win[i] / max(valid, 1)),
                "delta_mean": float(d.mean()), "delta_ci95_low": float(np.quantile(d, .025)),
                "delta_ci95_high": float(np.quantile(d, .975)),
                "delta_positive_fraction": float((d > 0).mean()),
            })
        candidates = []
        for i, source in enumerate(sources):
            d = np.asarray(deltas[i], float)
            if (observed[i] - observed[flight_i] >= MIN_DELTA and
                    win[i] / max(valid, 1) >= MIN_WIN_FREQ and
                    (d > 0).mean() >= MIN_POSITIVE_FRAC):
                candidates.append((observed[i], source))
        switches[target] = max(candidates)[1] if candidates else sources[flight_i]

    out = pd.DataFrame(rows).sort_values(["target", "observed_auc"], ascending=[True, False])
    out.to_csv(ROOT / "experiments" / "teacher_selection_stability.csv", index=False)

    candidate = np.empty_like(yall, dtype=float)
    for j, target in enumerate(TARGETS):
        candidate[:, j] = loaded[switches[target]].loc[ids, target].to_numpy(float)
    per = [auc1(yall[:, j], candidate[:, j]) for j in range(len(TARGETS))]
    flight = preds[flight_i]
    flight_per = [auc1(yall[:, j], flight[:, j]) for j in range(len(TARGETS))]
    summary = {
        "experiment": "LABEL_STABILITY_001",
        "common_gold": len(ids), "repeats": REPEATS, "seed": SEED,
        "switch_rule": {"min_observed_delta": MIN_DELTA, "min_bootstrap_win_frequency": MIN_WIN_FREQ,
                        "min_delta_positive_fraction": MIN_POSITIVE_FRAC},
        "flight_source": sources[flight_i], "flight_macro_auc_common": float(np.mean(flight_per)),
        "candidate_macro_auc_common": float(np.mean(per)),
        "switches": switches, "candidate_per_target_auc": dict(zip(TARGETS, map(float, per))),
        "warning": "Gold-selected diagnostic; do not treat candidate macro as unbiased validation."
    }
    (ROOT / "reports" / "teacher_selection_stability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
