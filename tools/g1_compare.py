"""Fixed G1 V888 comparison of stored predictions (CPU only, no model access).

  python tools/g1_compare.py --out artifacts/experiments/EXP090_COVERAGE_BF16_REPORT \
      --candidate cand=artifacts/experiments/EXP090_COV_BF16_CAND/oof_best.csv \
      --candidate ctrl=artifacts/experiments/EXP090_COV_BF16_CTRL/oof_best.csv \
      --contrast cand-ctrl --contrast cand-B --contrast ctrl-B

B (EXP084_B_G1) is always loaded as reference `B`. Labels: Dread soft → float32 → np.rint. Bootstrap: 873 G1 components,
500 draws, seed 20261003, conditional on fixed predictions. Writes result.json and result.md into --out (must not exist).

Quarantine (EXP116 v1.1.1, D-B-024): every prediction file (B and each --candidate) is checked BEFORE anything is read or --out is
created; a file whose directory or any parent directory below the workspace root carries DIAGNOSTIC_ONLY.json, or a result.json /
manifest.json with decision_eligible false, is refused (exit with the reason). Leaky predictions are never a ruler.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.diagnostic_quarantine import require_decision_eligible  # noqa: E402
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
REF = {'B': 'artifacts/experiments/EXP084_B_G1/oof_best.csv'}
LABELS = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
META = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/group-metadata.csv'
GROUPINGS = ['field', 'cached_mask']


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True); ap.add_argument('--candidate', action='append', default=[]); ap.add_argument('--contrast', action='append', default=[])
    ap.add_argument('--reps', type=int, default=500); ap.add_argument('--seed', type=int, default=20261003)
    a = ap.parse_args()
    files = dict(REF, **{c.split('=', 1)[0]: c.split('=', 1)[1] for c in a.candidate})
    for v in files.values():  # before any read and before --out exists
        require_decision_eligible(ROOT / v, 'tools/g1_compare.py', ROOT)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=False)
    lab = pd.read_csv(ROOT / LABELS).set_index('StudyInstanceUID'); meta = pd.read_csv(ROOT / META).set_index('StudyInstanceUID')
    preds = {k: pd.read_csv(ROOT / v).set_index('StudyInstanceUID')[T] for k, v in files.items()}
    uids = preds['B'].index.tolist()
    Y = np.rint(lab.loc[uids, T].to_numpy().astype(np.float32)).astype(int); comp = meta.loc[uids, 'component_id'].to_numpy()
    S = {k: p.loc[uids, T].to_numpy() for k, p in preds.items()}
    aucs = {k: np.array([auc(Y[:, t], v[:, t]) for t in range(12)]) for k, v in S.items()}
    rng = np.random.default_rng(a.seed); cu = np.unique(comp); idx = {c: np.where(comp == c)[0] for c in cu}
    draws = [np.concatenate([idx[c] for c in rng.choice(cu, size=len(cu), replace=True)]) for _ in range(a.reps)]
    contrasts = {}
    for c in a.contrast:
        x, y = c.split('-', 1)
        d = [np.mean([auc(Y[ii, t], S[x][ii, t]) - auc(Y[ii, t], S[y][ii, t]) for t in range(12)]) for ii in draws]
        d = [v for v in d if np.isfinite(v)]
        tci = {}
        for t in range(12):
            dt = [auc(Y[ii, t], S[x][ii, t]) - auc(Y[ii, t], S[y][ii, t]) for ii in draws]; dt = [v for v in dt if np.isfinite(v)]
            tci[T[t]] = [float(np.percentile(dt, 2.5)), float(np.percentile(dt, 97.5))]
        groups = {}
        for g in GROUPINGS:
            gv = meta.loc[uids, g].astype(str).to_numpy()
            for grp in np.unique(gv):
                m = gv == grp
                if m.sum() < 30:
                    continue
                vals = [auc(Y[m, t], S[x][m, t]) - auc(Y[m, t], S[y][m, t]) for t in range(12)]
                groups[f'{g}/{grp}'] = dict(n=int(m.sum()), delta=float(np.nanmean(vals)) if not any(np.isnan(vals)) else None)
        contrasts[c] = dict(macro12=float(aucs[x].mean() - aucs[y].mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], valid=len(d),
                            targets={T[t]: dict(delta=float(aucs[x][t] - aucs[y][t]), ci=tci[T[t]]) for t in range(12)}, groups_exploratory=groups)
    result = dict(schema='g1-compare-v1', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), rows=len(uids), components=int(len(cu)),
                  files={k: dict(path=v, sha256=hashlib.sha256((ROOT / v).read_bytes()).hexdigest()) for k, v in files.items()},
                  labels=dict(path=LABELS, rule='float32 then numpy.rint'), bootstrap=dict(reps=a.reps, seed=a.seed, unit='component_id'),
                  macro12={k: float(v.mean()) for k, v in aucs.items()}, targets={k: {T[t]: float(v[t]) for t in range(12)} for k, v in aucs.items()}, contrasts=contrasts,
                  caveats=['exposed weak G1 development set; one seed per model', 'CIs conditional on fixed predictions', 'no .943/public/private claim'])
    (out / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    lines = ['# G1 V888 comparison', '', '| score | macro12 |', '|---|---:|'] + [f'| {k} | {v:.6f} |' for k, v in result['macro12'].items()] + ['', '| contrast | Δ macro12 | CI |', '|---|---:|---|']
    lines += [f"| {k} | {v['macro12']:+.6f} | [{v['ci'][0]:+.6f}, {v['ci'][1]:+.6f}] |" for k, v in contrasts.items()] + ['']
    for k, v in contrasts.items():
        lines += [f'## {k} per target', '', '| target | Δ | CI |', '|---|---:|---|'] + [f"| {t} | {d['delta']:+.6f} | [{d['ci'][0]:+.5f}, {d['ci'][1]:+.5f}] |" for t, d in v['targets'].items()] + ['']
    (out / 'result.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(macro12=result['macro12'], contrasts={k: (round(v['macro12'], 6), [round(x, 6) for x in v['ci']]) for k, v in contrasts.items()}), indent=1))


if __name__ == '__main__':
    main()
