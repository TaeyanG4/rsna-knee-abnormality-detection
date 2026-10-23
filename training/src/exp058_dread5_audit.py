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
TARGETS = ["ACL", "MCL", "Effusion", "Baker's", "Fracture"]


def auc(y, p):
    return float(roc_auc_score(y, p))


def evaluate(df_base, df_spec, labels, mode: str):
    rows = []
    for t in TARGETS:
        ysoft = labels.loc[df_base.index, t].to_numpy(float)
        if mode == "flight":
            conf = labels.loc[df_base.index, t + "__conf"].to_numpy(float)
            use = ((ysoft <= 0.3) | (ysoft >= 0.7)) & (conf >= 0.6)
            y = (ysoft[use] >= 0.7).astype(np.uint8)
        elif mode == "dread":
            use = (ysoft <= 0.25) | (ysoft >= 0.75)
            y = (ysoft[use] >= 0.75).astype(np.uint8)
        else:
            raise ValueError(mode)
        b = df_base[t].to_numpy(float)[use]
        s = df_spec[t].to_numpy(float)[use]
        blend = 0.5 * b + 0.5 * s
        rows.append({
            "target": t,
            "n": int(use.sum()),
            "positive": int(y.sum()),
            "base_auc": auc(y, b),
            "specialist_auc": auc(y, s),
            "blend50_auc": auc(y, blend),
            "delta_spec_vs_base": auc(y, s) - auc(y, b),
            "delta_blend_vs_base": auc(y, blend) - auc(y, b),
            "pearson": float(np.corrcoef(b, s)[0, 1]),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--specialist-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    base = pd.read_csv(ROOT / args.base_dir / "weak_val_pred.csv", dtype={UID: str}).set_index(UID)
    spec = pd.read_csv(ROOT / args.specialist_dir / "weak_val_pred.csv", dtype={UID: str}).set_index(UID)
    if set(base.index) != set(spec.index):
        raise RuntimeError("weak UID mismatch")
    spec = spec.loc[base.index]
    flight = pd.read_csv(
        ROOT / "labels" / "flight0234__rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        dtype={UID: str},
    ).set_index(UID)
    dread = pd.read_csv(
        ROOT / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv",
        dtype={UID: str},
    ).set_index(UID)

    # Dread is train-only and excludes Gold58, but covers this fold's eligible UIDs.
    fdf = evaluate(base, spec, flight, "flight")
    ddf = evaluate(base, spec, dread, "dread")

    bg = pd.read_csv(ROOT / args.base_dir / "gold_pred.csv", dtype={UID: str}).set_index(UID)
    sg = pd.read_csv(ROOT / args.specialist_dir / "gold_pred.csv", dtype={UID: str}).set_index(UID).loc[bg.index]
    off = pd.read_csv(ROOT / "data" / "train.csv", dtype={UID: str}).set_index(UID)
    grows = []
    for t in TARGETS:
        y = off.loc[bg.index, t].to_numpy(np.uint8)
        b, s = bg[t].to_numpy(float), sg[t].to_numpy(float)
        grows.append({
            "target": t,
            "base_auc": auc(y, b),
            "specialist_auc": auc(y, s),
            "blend50_auc": auc(y, 0.5*b + 0.5*s),
        })
    gdf = pd.DataFrame(grows)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fdf.to_csv(out / "flight_surface.csv", index=False)
    ddf.to_csv(out / "dread_surface.csv", index=False)
    gdf.to_csv(out / "gold_diagnostic.csv", index=False)
    summary = {
        "contract": "exp058_dread5_fold_audit_v1",
        "targets": TARGETS,
        "base": args.base_dir,
        "specialist": args.specialist_dir,
        "flight": {
            "base": float(fdf.base_auc.mean()),
            "specialist": float(fdf.specialist_auc.mean()),
            "blend50": float(fdf.blend50_auc.mean()),
            "blend_delta": float((fdf.blend50_auc - fdf.base_auc).mean()),
        },
        "dread": {
            "base": float(ddf.base_auc.mean()),
            "specialist": float(ddf.specialist_auc.mean()),
            "blend50": float(ddf.blend50_auc.mean()),
            "blend_delta": float((ddf.blend50_auc - ddf.base_auc).mean()),
        },
        "gold_diagnostic": {
            "base": float(gdf.base_auc.mean()),
            "specialist": float(gdf.specialist_auc.mean()),
            "blend50": float(gdf.blend50_auc.mean()),
            "warning": "Gold58 is diagnostic only and must not be used for target/weight tuning.",
        },
        "selection_note": "Targets were predeclared from non-Gold EXP054 severity/alignment evidence before this student was trained.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nFLIGHT\n", fdf.to_string(index=False))
    print("\nDREAD\n", ddf.to_string(index=False))
    print("\nGOLD DIAGNOSTIC\n", gdf.to_string(index=False))


if __name__ == "__main__":
    main()
