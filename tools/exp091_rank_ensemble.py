"""EXP091 (H-B-03): pre-registered equal-weight rank ensembles of existing own G1 predictions.

CPU only. Reads stored V888 predictions (no model, no Gold), the fixed Dread labels (float32 → np.rint)
and the G1 component ids, and writes artifacts/experiments/EXP091_RANK_ENSEMBLE_REPORT/{result.json,result.md}.

Pre-registered blends (no weight tuning):
  E1 = mean average-tie rank(B, R384)
  E2 = mean rank(B, R384, EXP085 treatment)
  E3 = 0.9*E1 + 0.1*rank(RN clip)
Contrasts: each blend minus B, each blend minus R384; component bootstrap 500 draws, seed 20261003.
"""
from __future__ import annotations
import json
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
MEMBERS = {
    'B': 'artifacts/experiments/EXP084_B_G1/oof_best.csv',
    'R384': 'artifacts/experiments/EXP088_RAPTOR384_G1_V4/oof_best.csv',
    'E085T': 'artifacts/experiments/EXP085_RELATIVE_POSITION_G1/treatment/oof_final.csv',
    'RNclip': 'artifacts/experiments/EXP084_RN_CLIP_G1/oof_best.csv',
}
LABELS = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
META = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/group-metadata.csv'
OUT = ROOT / 'artifacts/experiments/EXP091_RANK_ENSEMBLE_REPORT'
SEED, REPS = 20261003, 500


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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


def rank_cols(S):
    return np.stack([avg_rank(S[:, k]) / len(S) for k in range(12)], 1)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    lab = pd.read_csv(ROOT / LABELS).set_index('StudyInstanceUID')
    meta = pd.read_csv(ROOT / META).set_index('StudyInstanceUID')
    preds = {k: pd.read_csv(ROOT / v).set_index('StudyInstanceUID')[T] for k, v in MEMBERS.items()}
    uids = preds['B'].index.tolist()
    if len(uids) != 888 or any(list(p.index) != uids for p in preds.values()):
        for k, p in preds.items():
            preds[k] = p.loc[uids]
    Y = np.rint(lab.loc[uids, T].to_numpy().astype(np.float32)).astype(int)
    comp = meta.loc[uids, 'component_id'].to_numpy()
    S = {k: p.loc[uids, T].to_numpy() for k, p in preds.items()}
    R = {k: rank_cols(v) for k, v in S.items()}
    blends = {'E1': (R['B'] + R['R384']) / 2, 'E2': (R['B'] + R['R384'] + R['E085T']) / 3}
    blends['E3'] = 0.9 * blends['E1'] + 0.1 * R['RNclip']
    scores = {**S, **blends}
    aucs = {k: np.array([auc(Y[:, t], v[:, t]) for t in range(12)]) for k, v in scores.items()}
    rng = np.random.default_rng(SEED); cu = np.unique(comp); idx = {c: np.where(comp == c)[0] for c in cu}
    draws = [np.concatenate([idx[c] for c in rng.choice(cu, size=len(cu), replace=True)]) for _ in range(REPS)]

    def ci(a, b):
        d = []
        for ii in draws:
            v = np.mean([auc(Y[ii, t], scores[a][ii, t]) - auc(Y[ii, t], scores[b][ii, t]) for t in range(12)])
            if np.isfinite(v):
                d.append(v)
        return dict(valid=len(d), low=float(np.percentile(d, 2.5)), high=float(np.percentile(d, 97.5)))
    contrasts = {}
    for b in ('E1', 'E2', 'E3'):
        for base in ('B', 'R384'):
            key = f'{b}_minus_{base}'
            contrasts[key] = dict(macro12=float(aucs[b].mean() - aucs[base].mean()),
                                  targets={T[t]: float(aucs[b][t] - aucs[base][t]) for t in range(12)}, ci=ci(b, base))
    spearman = {}
    names = list(S)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            spearman[f'{a}~{b}'] = float(np.mean([np.corrcoef(R[a][:, t], R[b][:, t])[0, 1] for t in range(12)]))
    result = dict(schema='exp091-rank-ensemble-v1', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_item='H-B-03', rows=len(uids),
                  members={k: dict(path=v, sha256=sha256(ROOT / v)) for k, v in MEMBERS.items()}, labels=dict(path=LABELS, sha256=sha256(ROOT / LABELS), rule='float32 then numpy.rint'),
                  groups=dict(path=META, sha256=sha256(ROOT / META), components=int(len(cu))), bootstrap=dict(reps=REPS, seed=SEED, unit='component_id'),
                  pre_registered=['E1=mean rank(B,R384)', 'E2=mean rank(B,R384,E085T)', 'E3=0.9*E1+0.1*rank(RNclip)', 'no weight tuning'],
                  macro12={k: float(v.mean()) for k, v in aucs.items()}, targets={k: {T[t]: float(v[t]) for t in range(12)} for k, v in aucs.items()},
                  contrasts=contrasts, mean_target_spearman=spearman,
                  caveats=['same exposed weak G1 V888, one seed per member', 'E085T is warm-started from B (correlated errors)',
                           'CIs conditional on fixed predictions; no seed/label/private uncertainty', 'no .943/public/private claim'])
    (OUT / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    lines = ['# EXP091 rank ensembles of existing G1 predictions', '', f'V888 rows, {len(cu)} components, bootstrap {REPS}/seed {SEED}.', '',
             '| score | macro12 |', '|---|---:|'] + [f'| {k} | {v:.6f} |' for k, v in result['macro12'].items()] + ['', '| contrast | macro12 Δ | CI low | CI high |', '|---|---:|---:|---:|']
    lines += [f"| {k} | {v['macro12']:+.6f} | {v['ci']['low']:+.6f} | {v['ci']['high']:+.6f} |" for k, v in contrasts.items()]
    lines += ['', '| pair | mean Spearman |', '|---|---:|'] + [f'| {k} | {v:.4f} |' for k, v in spearman.items()] + ['']
    (OUT / 'result.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(macro12=result['macro12'], contrasts={k: (v['macro12'], v['ci']['low'], v['ci']['high']) for k, v in contrasts.items()}), indent=1))


if __name__ == '__main__':
    main()
