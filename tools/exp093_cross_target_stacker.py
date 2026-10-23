"""EXP093 (H-B-09): cross-target 12→12 second-stage map fit on old-fold B OOF, applied to EXP084_B_G1.

CPU only. Fit set: stitched development OOF of the B models EXP078_INPUT_B_F0, EXP078_INPUT_B_F1, EXP080_INPUT_B_F2
(old report-group folds), with every G1 V888 study and the two excluded studies removed. Model: for each finding t,
ridge-regularised logistic regression on the 12 input logits (plus bias), L2 strength fixed in advance (lambda=1.0 on
standardised inputs), fitted by Newton iterations. Applied to EXP084_B_G1 V888 logits. Reports macro/12 AUC vs raw B
with component bootstrap (500, seed 20261003). Writes artifacts/experiments/EXP093_CROSS_TARGET_STACKER/result.json.
"""
from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FIT = ['artifacts/experiments/EXP078_INPUT_B_F0/oof_best.csv', 'artifacts/experiments/EXP078_INPUT_B_F1/oof_best.csv', 'artifacts/experiments/EXP080_INPUT_B_F2/oof_best.csv']
APPLY = 'artifacts/experiments/EXP084_B_G1/oof_best.csv'
LABELS = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
META = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/group-metadata.csv'
EXCLUDED = 'artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/excluded.csv'
OUT = ROOT / 'artifacts/experiments/EXP093_CROSS_TARGET_STACKER'
LAMBDA, SEED, REPS = 1.0, 20261003, 500


def avg_rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); sx = x[o]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def auc(y, s):
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return np.nan
    r = avg_rank(s); return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fit_logistic(X, y, lam, iters=50):
    n, d = X.shape; Xb = np.hstack([X, np.ones((n, 1))]); w = np.zeros(d + 1); reg = np.eye(d + 1) * lam; reg[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w)); g = Xb.T @ (p - y) + reg @ w; H = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    lab = pd.read_csv(ROOT / LABELS).set_index('StudyInstanceUID')
    meta = pd.read_csv(ROOT / META).set_index('StudyInstanceUID')
    apply_df = pd.read_csv(ROOT / APPLY).set_index('StudyInstanceUID')[T]; v_uids = apply_df.index.tolist()
    excluded = set(pd.read_csv(ROOT / EXCLUDED)['StudyInstanceUID'].astype(str))
    fit_df = pd.concat([pd.read_csv(ROOT / f).set_index('StudyInstanceUID')[T] for f in FIT])
    fit_df = fit_df[~fit_df.index.duplicated(keep='first')]
    keep = [u for u in fit_df.index if u not in set(v_uids) and u not in excluded]
    fit_df = fit_df.loc[keep]
    Xf = logit(fit_df.to_numpy()); Yf = np.rint(lab.loc[keep, T].to_numpy().astype(np.float32)).astype(int)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    W = np.stack([fit_logistic((Xf - mu) / sd, Yf[:, t], LAMBDA) for t in range(12)])
    Xv = (logit(apply_df.to_numpy()) - mu) / sd; Yv = np.rint(lab.loc[v_uids, T].to_numpy().astype(np.float32)).astype(int)
    Sraw = apply_df.to_numpy(); Sst = np.hstack([Xv, np.ones((len(Xv), 1))]) @ W.T
    comp = meta.loc[v_uids, 'component_id'].to_numpy()
    a_raw = np.array([auc(Yv[:, t], Sraw[:, t]) for t in range(12)]); a_st = np.array([auc(Yv[:, t], Sst[:, t]) for t in range(12)])
    rng = np.random.default_rng(SEED); cu = np.unique(comp); idx = {c: np.where(comp == c)[0] for c in cu}; d = []
    for _ in range(REPS):
        ii = np.concatenate([idx[c] for c in rng.choice(cu, size=len(cu), replace=True)])
        v = np.mean([auc(Yv[ii, t], Sst[ii, t]) - auc(Yv[ii, t], Sraw[ii, t]) for t in range(12)])
        if np.isfinite(v):
            d.append(v)
    result = dict(schema='exp093-cross-target-stacker-v1', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_item='H-B-09',
                  fit=dict(files=FIT, rows=len(keep), overlap_with_V888_removed=int(sum(u in set(v_uids) for u in pd.concat([pd.read_csv(ROOT / f) for f in FIT])['StudyInstanceUID'])), excluded_removed=len(excluded)),
                  model='per-finding ridge logistic on 12 standardised input logits + bias, lambda=%s, Newton' % LAMBDA,
                  apply=dict(file=APPLY, rows=len(v_uids), components=int(len(cu))), bootstrap=dict(reps=REPS, seed=SEED, valid=len(d)),
                  macro12=dict(raw_B=float(a_raw.mean()), stacked=float(a_st.mean()), delta=float(a_st.mean() - a_raw.mean())),
                  ci=dict(low=float(np.percentile(d, 2.5)), high=float(np.percentile(d, 97.5))),
                  targets={T[t]: dict(raw=float(a_raw[t]), stacked=float(a_st[t]), delta=float(a_st[t] - a_raw[t])) for t in range(12)},
                  weights={T[t]: {T[j]: float(W[t, j]) for j in range(12)} for t in range(12)},
                  caveats=['old-fold B checkpoints were development-OOF selected and differ from the G1 B model', 'fit labels are the same weak Dread labels', 'V888 used only for evaluation'])
    (OUT / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(fit_rows=len(keep), macro12=result['macro12'], ci=result['ci'], targets={k: round(v['delta'], 5) for k, v in result['targets'].items()}), indent=1))


if __name__ == '__main__':
    main()
