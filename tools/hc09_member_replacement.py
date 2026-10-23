"""H-C-09: one CPU-only, preregistered replacement using stored predictions.

Usage: python tools/hc09_member_replacement.py --manifest PATH --review PATH --out NEW_DIR
Exits nonzero on contract failure. Never loads a model or edits existing artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.exp091_rank_ensemble import T, auc, rank_cols


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load_frame(path):
    frame = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    require(frame.StudyInstanceUID.notna().all(), f'missing UID: {path}')
    require(frame.StudyInstanceUID.is_unique, f'duplicate UID: {path}')
    return frame.set_index('StudyInstanceUID')


def target_aucs(y, prediction):
    return np.array([auc(y[:, k], prediction[:, k]) for k in range(12)])


def finite_summary(samples):
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    return {'valid': len(samples), 'low': float(np.percentile(samples, 2.5)) if len(samples) else None,
            'high': float(np.percentile(samples, 97.5)) if len(samples) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--review', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    started = time.monotonic()
    m = json.loads(args.manifest.read_text(encoding='utf-8'))
    review = json.loads(args.review.read_text(encoding='utf-8'))
    manifest_hash = digest(args.manifest)
    require(m['schema'] == 'hc09-fixed-member-replacement-v1', 'unknown protocol')
    require(review['status'] == 'READY' and review['manifest_sha256'] == manifest_hash,
            'review not READY for this manifest')
    require(m['targets'] == T and m['rows'] == 888 and m['components'] == 873, 'wrong cohort/targets')
    require(m['seed'] == 20261003 and m['reps'] == 500, 'wrong bootstrap')
    require(m['primary'] == 'mean_rank(T,R384)-mean_rank(B,R384)', 'wrong contrast')
    require(m['budget_seconds'] == 1200, 'wrong CPU budget')
    for name, spec in m['inputs'].items():
        require(digest(ROOT / spec['path']) == spec['sha256'], f'input drift: {name}')
    require(m['inputs']['evaluator']['path'] == str(Path(__file__).relative_to(ROOT)).replace('\\', '/'),
            'wrong evaluator binding')
    prior = json.loads((ROOT / m['inputs']['reference']['path']).read_text(encoding='utf-8'))
    frames = {k: load_frame(ROOT / m['inputs'][k]['path']) for k in ('B', 'R384', 'T')}
    uids = frames['B'].index.tolist()
    require(len(uids) == 888, 'not V888')
    for name, frame in frames.items():
        require(set(frame.index) == set(uids) and list(frame.columns) == T, f'prediction contract: {name}')
    lab = load_frame(ROOT / m['inputs']['labels']['path'])
    meta = load_frame(ROOT / m['inputs']['groups']['path'])
    ysoft = lab.loc[uids, T].to_numpy(dtype=np.float32)
    require(np.isfinite(ysoft).all() and ((ysoft >= 0) & (ysoft <= 1)).all(), 'invalid labels')
    y = np.rint(ysoft).astype(int)
    require(((y.sum(0) > 0) & (y.sum(0) < 888)).all(), 'missing target class')
    cohort = meta.loc[uids]
    require(cohort.component_id.notna().all(), 'missing component')
    require(cohort['split'].nunique() == 1 and str(cohort['split'].iloc[0]).lower() in ('val', 'v', 'valid', 'validation'),
            'not validation split')
    comp = cohort.component_id.to_numpy()
    unique = np.unique(comp)
    require(len(unique) == 873, 'wrong component count')
    scores = {k: f.loc[uids, T].to_numpy(dtype=np.float64) for k, f in frames.items()}
    ranks = {}
    for k, values in scores.items():
        require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), f'bad predictions: {k}')
        ranks[k] = rank_cols(values)
        require(np.array_equal(ranks[k], rankdata(values, axis=0, method='average') / 888), 'rank implementation disagreement')
    scores['E1'] = (ranks['B'] + ranks['R384']) / 2
    scores['E2'] = (ranks['B'] + ranks['R384'] + ranks['T']) / 3
    scores['replacement'] = (ranks['T'] + ranks['R384']) / 2
    point = {k: target_aucs(y, s) for k, s in scores.items()}
    for key, values in point.items():
        independent = roc_auc_score(y, scores[key], average=None)
        require(np.allclose(values, independent, atol=1e-12, rtol=0), f'AUC implementation disagreement: {key}')
        if key != 'replacement':
            refkey = 'E085T' if key == 'T' else key
            expected = np.array([prior['targets'][refkey][t] for t in T])
            require(np.allclose(values, expected, atol=1e-12, rtol=0), f'prior target-score drift: {key}')
            require(abs(values.mean() - prior['macro12'][refkey]) < 1e-12, f'prior macro drift: {key}')
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'STARTED.json', {'manifest_sha256': manifest_hash, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    rng = np.random.default_rng(m['seed'])
    indices = {c: np.flatnonzero(comp == c) for c in unique}
    boot = []
    for _ in range(m['reps']):
        require(time.monotonic() - started < m['budget_seconds'], 'CPU budget exhausted')
        ii = np.concatenate([indices[c] for c in rng.choice(unique, size=len(unique), replace=True)])
        # Rank the source predictions once on V888; resample the FIXED ensemble scores.
        boot.append(target_aucs(y[ii], scores['replacement'][ii]) - target_aucs(y[ii], scores['E1'][ii]))
    boot = np.asarray(boot)
    delta = point['replacement'] - point['E1']
    ci = finite_summary(boot.mean(1))  # A draw is macro-valid only when all twelve targets are valid.
    require(ci['valid'] >= 475, 'fewer than 95% valid macro draws')
    fields = {}
    for group, frame in cohort.groupby('field', dropna=False):
        ii = np.flatnonzero(cohort.index.isin(frame.index))
        a = target_aucs(y[ii], scores['E1'][ii]); b = target_aucs(y[ii], scores['replacement'][ii])
        fields[str(group)] = {'rows': len(ii), 'macro_delta': float((b-a).mean()) if np.isfinite(b-a).all() else None,
                             'targets': {t: {'positive': int(y[ii, k].sum()), 'negative': int(len(ii)-y[ii, k].sum()),
                                             'delta': float(b[k]-a[k]) if np.isfinite(b[k]-a[k]) else None} for k, t in enumerate(T)}}
    classification = ('development_candidate' if delta.mean() > 0 and ci['low'] > 0 else
                      'positive_uncertain' if delta.mean() > 0 else 'nonpositive_on_this_dev_contrast')
    result = {'schema': m['schema'], 'plan_item': 'H-C-09', 'observed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'manifest_sha256': manifest_hash, 'review_sha256': digest(args.review), 'rows': len(uids), 'components': len(unique),
              'uid_order_sha256': hashlib.sha256(('\n'.join(uids)+'\n').encode()).hexdigest(),
              'macro12': {k: float(v.mean()) for k, v in point.items()},
              'primary': {'delta': float(delta.mean()), 'ci': ci, 'classification': classification},
              'replacement_minus_E2_descriptive': float(point['replacement'].mean()-point['E2'].mean()),
              'targets': {t: {'positive': int(y[:, k].sum()), 'negative': int(888-y[:, k].sum()),
                              'auc': {name: float(values[k]) for name, values in point.items()},
                              'delta': float(delta[k]), 'ci': finite_summary(boot[:, k])} for k, t in enumerate(T)},
              'field_diagnostics': fields, 'bootstrap': {'reps': 500, 'seed': m['seed'], 'unit': 'component_id', 'ranks_recomputed': False},
              'checks': {'input_hashes': 'PASS', 'prior_all_target_and_macro_scores': 'PASS',
                         'scipy_rank_parity': 'PASS', 'sklearn_auc_parity': 'PASS'},
              'caveats': m['caveats'], 'elapsed_seconds': time.monotonic()-started, 'numerical_threads': 1}
    for spec in m['inputs'].values():
        require(digest(ROOT / spec['path']) == spec['sha256'], 'input changed during evaluation')
    require(time.monotonic()-started < m['budget_seconds'], 'CPU budget exhausted')
    write_json(args.out / 'result.json', result)
    write_json(args.out / 'VALIDATED_SUCCESS.json', {'status': 'VALIDATED_SUCCESS', 'manifest_sha256': manifest_hash,
               'result_sha256': digest(args.out / 'result.json'), 'checks': result['checks'], 'primary_valid_draws': ci['valid']})
    print(json.dumps({'macro12': result['macro12'], 'primary': result['primary'], 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
