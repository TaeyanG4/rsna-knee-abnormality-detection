"""EXP092 (H-B-10): scanner/field-strength intercept-shift diagnostic on stored G1 predictions.

CPU only, diagnostic. For each grouping (field, scanner_group, manufacturer, cached_mask) and each model
(B, R384) it cross-fits a per-group additive logit offset per finding with 5-fold component-grouped CV inside
V888 (offset estimated on the training folds as the difference between the group mean logit of positives/negatives
balance: mean logit of the group minus overall mean logit, within label class, i.e. a per-group centering of
scores), applies it to the held-out fold, and reports the pooled macro change. This uses V888 labels, so it is an
upper-bound diagnostic of a calibration/domain-shift mechanism, never a method to adopt.
Writes artifacts/experiments/EXP092_GROUP_OFFSET_DIAG/result.json.
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
MODELS = {'B': 'artifacts/experiments/EXP084_B_G1/oof_best.csv', 'R384': 'artifacts/experiments/EXP088_RAPTOR384_G1_V4/oof_best.csv'}
LABELS = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
META = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/group-metadata.csv'
GROUPINGS = ['field', 'scanner_group', 'manufacturer', 'cached_mask']
OUT = ROOT / 'artifacts/experiments/EXP092_GROUP_OFFSET_DIAG'
SEED = 20261003


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


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    lab = pd.read_csv(ROOT / LABELS).set_index('StudyInstanceUID')
    meta = pd.read_csv(ROOT / META).set_index('StudyInstanceUID')
    B = pd.read_csv(ROOT / MODELS['B']).set_index('StudyInstanceUID')[T]
    uids = B.index.tolist()
    Y = np.rint(lab.loc[uids, T].to_numpy().astype(np.float32)).astype(int)
    comp = meta.loc[uids, 'component_id'].to_numpy()
    rng = np.random.default_rng(SEED); cu = np.unique(comp); rng.shuffle(cu)
    fold_of_comp = {c: i % 5 for i, c in enumerate(cu)}
    fold = np.array([fold_of_comp[c] for c in comp])
    result = dict(schema='exp092-group-offset-diag-v1', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_item='H-B-10', rows=len(uids),
                  method='5-fold component-grouped CV inside V888; per-group, per-finding logit offset = (group mean logit − overall mean logit) estimated on training folds using scores only (label-free centering) and, separately, a label-aware offset = mean over classes of (group class-mean − overall class-mean); applied to held-out fold; pooled macro recomputed',
                  seed=SEED, models={}, caveat='diagnostic upper bound using V888; label-aware variant uses V labels; never a method to adopt')
    for name, path in MODELS.items():
        S = pd.read_csv(ROOT / path).set_index('StudyInstanceUID').loc[uids, T].to_numpy()
        L = np.log(np.clip(S, 1e-6, 1 - 1e-6) / (1 - np.clip(S, 1e-6, 1 - 1e-6)))
        base = np.array([auc(Y[:, t], L[:, t]) for t in range(12)])
        entry = dict(macro12=float(base.mean()), groupings={})
        for g in GROUPINGS:
            gv = meta.loc[uids, g].astype(str).to_numpy()
            for variant in ('label_free', 'label_aware'):
                adj = L.copy()
                for f in range(5):
                    tr, te = fold != f, fold == f
                    for t in range(12):
                        overall = L[tr, t].mean() if variant == 'label_free' else np.mean([L[tr & (Y[:, t] == c), t].mean() for c in (0, 1) if (tr & (Y[:, t] == c)).any()])
                        for grp in np.unique(gv[te]):
                            m = tr & (gv == grp)
                            if m.sum() < 20:
                                continue
                            if variant == 'label_free':
                                off = L[m, t].mean() - overall
                            else:
                                parts = [L[m & (Y[:, t] == c), t].mean() for c in (0, 1) if (m & (Y[:, t] == c)).sum() >= 5]
                                if not parts:
                                    continue
                                off = np.mean(parts) - overall
                            adj[te & (gv == grp), t] -= off
                a = np.array([auc(Y[:, t], adj[:, t]) for t in range(12)])
                entry['groupings'][f'{g}/{variant}'] = dict(macro12=float(a.mean()), delta=float(a.mean() - base.mean()), targets={T[t]: float(a[t] - base[t]) for t in range(12)},
                                                            groups={str(k): int(v) for k, v in zip(*np.unique(gv, return_counts=True))})
        result['models'][name] = entry
    (OUT / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({m: {k: round(v['delta'], 6) for k, v in e['groupings'].items()} for m, e in result['models'].items()}, indent=1))


if __name__ == '__main__':
    main()
