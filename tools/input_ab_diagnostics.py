"""Descriptive EXP078 target/group supplement; never changes the fixed A/B gate.

Requires the completed primary report and all four validated fixed-epoch fits.
No training, Gold lookup, threshold fitting or leaderboard input is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_ab_report as primary
from oof_error_audit import weighted_auc

UID, LAB, ROOT = primary.UID, primary.LAB, primary.ROOT


def target_pair(y, a, b):
    """Weak labels use the same float32 ties-to-even rounding as macro_auc."""
    binary = np.asarray(y).round().astype(int)
    rows = []
    for j, target in enumerate(LAB):
        pos = int(binary[:, j].sum()); neg = len(binary)-pos
        aa = bb = delta = None
        if pos and neg:
            weight = np.ones(len(binary))
            aa = weighted_auc(binary[:, j], a[:, j], weight)
            bb = weighted_auc(binary[:, j], b[:, j], weight)
            delta = bb-aa
        rows.append(dict(target=target, P=pos, N=neg, auc_a=aa, auc_b=bb,
                         delta=delta, macro_contribution=None if delta is None else delta/12))
    return rows


def target_intervals(folds, reps=500, seed=20261003):
    """Paired report-group bootstrap within each fold; mean-fold estimand."""
    rng = np.random.default_rng(seed)
    coded = []
    for y, a, b, groups in folds:
        unique, codes = np.unique(groups, return_inverse=True)
        coded.append((y.round().astype(int), a, b, codes, len(unique)))
    draws = np.empty((reps, len(LAB)))
    for k in range(reps):
        ds = []
        for y, a, b, codes, n in coded:
            w = np.bincount(rng.integers(0, n, n), minlength=n)[codes]
            ds.append([weighted_auc(y[:, j], b[:, j], w)-weighted_auc(y[:, j], a[:, j], w)
                       for j in range(len(LAB))])
        draws[k] = np.mean(ds, axis=0)
    result = {}
    for j, target in enumerate(LAB):
        good = draws[:, j][np.isfinite(draws[:, j])]
        result[target] = dict(valid_reps=len(good), requested_reps=reps,
                              low=float(np.quantile(good, .025)) if len(good) else None,
                              high=float(np.quantile(good, .975)) if len(good) else None)
    return result


def build(manifest_path, primary_path, out, reps=500):
    manifest_path, primary_path, out = map(Path, (manifest_path, primary_path, out))
    if out.exists():
        raise FileExistsError(out)
    if reps < 100:
        raise ValueError('At least100 draws required')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    report = json.loads(primary_path.read_text(encoding='utf-8'))
    if report['manifest_sha256'] != primary.sha256_file(manifest_path):
        raise ValueError('Primary report uses a different protocol')
    if sorted(r['fold'] for r in report['folds']) != sorted(manifest['folds']):
        raise ValueError('Incomplete primary report')
    labels = primary.validate_sources(manifest)
    split_path = ROOT/manifest['fold_file']
    split = pd.read_csv(split_path, dtype={UID: str}).set_index(UID)
    metadata_path = ROOT/'storage/labels/derived_meta/study_meta.csv'
    meta = pd.read_csv(metadata_path, dtype={UID: str}).set_index(UID)
    corpus = ROOT/manifest['corpus']
    corpus_ids = np.load(corpus/'ids.npy').astype(str)
    mask = np.load(corpus/'masks.npy', mmap_mode='r')
    if len(corpus_ids) != len(mask) or len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError('Invalid corpus mask IDs')
    mask_groups = pd.Series(np.where((mask == 0).any(axis=1), 'some_zero_mask', 'all_nonzero'), index=corpus_ids)
    if not meta.index.is_unique or not split.index.is_unique:
        raise ValueError('Nonunique metadata or split')
    fold_data = []; fold_rows = []; group_rows = []; hashes = []
    for fold in manifest['folds']:
        runs = {r['arm']: r for r in manifest['runs'] if r['fold'] == fold}
        if set(runs) != {'A', 'B'}:
            raise ValueError('Expected both arms')
        ids = sorted(set(split.index[split.fold == fold])-set(manifest['excluded_uids']))
        values = {}
        for arm in ('A', 'B'):
            folder = ROOT/'artifacts/experiments'/runs[arm]['tag']
            summary_path = folder/'summary.json'; prediction_path = folder/'oof_best.csv'
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            primary.validate_summary(summary, manifest, runs[arm])
            frame = pd.read_csv(prediction_path, dtype={UID: str}).set_index(UID)
            if not frame.index.is_unique or set(frame.index) != set(ids) or list(frame.columns) != LAB:
                raise ValueError('Study/target mismatch')
            values[arm] = frame.loc[ids, LAB].to_numpy()
            if not np.isfinite(values[arm]).all() or np.any((values[arm] < 0) | (values[arm] > 1)):
                raise ValueError('Invalid probabilities')
            for path in (summary_path, prediction_path):
                hashes.append({'path': 'workspace://'+path.relative_to(ROOT).as_posix(), 'sha256': primary.sha256_file(path)})
        y = np.stack([labels[u] for u in ids]); a, b = values['A'], values['B']
        g = split.loc[ids, 'report_group_sha256'].to_numpy()
        if pd.isna(g).any():
            raise ValueError('Missing report group')
        rows = target_pair(y, a, b)
        observed = next(r for r in report['folds'] if r['fold'] == fold)
        if abs(np.mean([r['delta'] for r in rows])-observed['delta']) > 1e-10:
            raise ValueError('Predictions differ from primary result')
        fold_rows.append({'fold': fold, 'studies': len(ids), 'targets': rows})
        fold_data.append((y, a, b, g))
        group_meta = meta.reindex(ids)[['scanner_group', 'field', 'manufacturer']].fillna('unknown')
        group_meta['cached_mask'] = mask_groups.reindex(ids)
        if group_meta.cached_mask.isna().any():
            raise ValueError('Missing mask group')
        for field in group_meta.columns:
            vals = group_meta[field].astype(str).to_numpy()
            for group in sorted(set(vals)):
                take = vals == group; n = int(take.sum())
                targets = target_pair(y[take], a[take], b[take])
                group_rows.append({'fold': fold, 'field': field, 'group': group, 'studies': n,
                    'small_group': n < 40, 'targets': targets,
                    'macro12_delta': float(np.mean([r['delta'] for r in targets]))
                        if all(r['delta'] is not None for r in targets) else None})
    intervals = target_intervals(fold_data, reps=reps)
    result = {'observed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Descriptive fixed-four-fit EXP078 supplement; does not alter primary decision gate',
        'manifest_sha256': primary.sha256_file(manifest_path), 'primary_report_sha256': primary.sha256_file(primary_path),
        'primary_gate_unchanged': report['applied_gate'], 'folds': fold_rows,
        'mean_fold_target_delta': {t: float(np.mean([f['targets'][j]['delta'] for f in fold_rows])) for j,t in enumerate(LAB)},
        'mean_fold_target_delta_ci': intervals, 'groups': group_rows,
        'caveats': ['Exposed weak-label development folds, not clinical/private validation.',
            'Conditional report-group intervals ignore seed/label/selection/distribution uncertainty; no multiplicity correction.',
            'Scanner groups are observational; masks mean nonzero pixels, not acquisition or clinical availability.',
            'Small or one-class groups remain visible with counts; no partial-target mean is reported as macro12.',
            'No subgroup routing or LB tuning follows automatically from these descriptive differences.'],
        'input_hashes': hashes}
    for path in (metadata_path, corpus/'ids.npy', corpus/'masks.npy', Path(__file__)):
        result['input_hashes'].append({'path':'workspace://'+path.relative_to(ROOT).as_posix(),'sha256':primary.sha256_file(path)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return {'folds':len(fold_rows),'group_slices':len(group_rows),'out':str(out)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--primary', required=True)
    p.add_argument('--out', required=True); p.add_argument('--bootstrap', type=int, default=500)
    args = p.parse_args()
    print(json.dumps(build(args.manifest, args.primary, args.out, args.bootstrap)))
