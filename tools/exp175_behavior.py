"""EXP175 v1.2.0 CPU-only, descriptive image-domain/behavior associations.

Only explicit numerical prediction, roster, component and label files are read.
No training modules, reports, test labels, CUDA, or network are used.

Changelog v1.2.0: legacy deadline arguments are ignored; finish all registered
behavior calculations while retaining genuine missing-data results.
Changelog v1.1.1: release-version alignment with audit provenance/timing fixes;
behavior calculations and registered-screen semantics are unchanged.
Changelog v1.1.0: compare prediction receipts with original metric-bound
cache hashes; require audit/verification provenance; expose paired valid draws.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

VERSION = '1.2.0'
UID = 'StudyInstanceUID'
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
REFERENCES = {
    'jev': 'storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv',
    'dread': 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv',
}
COMPONENTS = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
NATIVE = {
    'MRNET': [('abnormal', TARGETS), ('acl', ('ACL',)), ('any_meniscus', ('Medial Meniscus', 'Lateral Meniscus'))],
    'KNEEMRI': [('acl_any', ('ACL',)), ('acl_complete', ('ACL',))],
    'FASTMRI_PLUS': [('acl_any', ('ACL',)), ('mcl_any', ('MCL',)),
                     ('any_meniscus', ('Medial Meniscus', 'Lateral Meniscus')),
                     ('effusion', ('Effusion',))],
}
SOURCE_ALIASES = {'FASTMRI_SC': 'FASTMRI_PLUS', 'SKMTEA': 'SKM_TEA'}


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _record(path, root, provenance):
    path = Path(path)
    provenance[str(path.relative_to(root)).replace('\\', '/')] = _sha(path)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _table(path, key, columns, *, probabilities=False, binary=False):
    """Read allowlisted numeric columns only, including when other fields exist."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream))
    selected = [c for c in columns if c in header]
    if key not in header or (probabilities and selected != list(columns)):
        raise ValueError('Required numerical columns missing: ' + str(path))
    frame = pd.read_csv(path, usecols=[key, *selected], dtype={key: str})
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError('Duplicate/empty study keys: ' + str(path))
    frame = frame.set_index(key)
    for column in selected:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    frame = frame.reindex(columns=columns)
    finite = np.isfinite(frame.to_numpy(dtype=float))
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('Infinite numerical values: ' + str(path))
    if probabilities and (not finite.all() or ((values < 0) | (values > 1)).any()):
        raise ValueError('Invalid probability values: ' + str(path))
    if binary and not np.isin(values[finite], (0., 1.)).all():
        raise ValueError('Nonbinary source labels: ' + str(path))
    return frame


def _rosters(root, provenance):
    rosters, union = [], set()
    for fold in range(5):
        path = root / f'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{fold}/ordered-ids.json'
        _record(path, root, provenance)
        value = json.loads(path.read_text(encoding='utf-8'))
        train, val = list(map(str, value['train'])), list(map(str, value['val']))
        if len(set(val)) != len(val) or len(set(train)) != len(train) or set(train) & set(val) or union & set(val):
            raise ValueError('Fold duplicate, overlap, or train/validation leakage')
        union.update(val)
        rosters.append((train, val))
    for train, val in rosters:
        if set(train) != union - set(val):
            raise ValueError('Fold training complement differs from pooled validation roster')
    return rosters


def _model_paths(root, name, fold):
    base = root / 'artifacts/experiments'
    if name == 'L2_ConvNeXt_T':
        return [base / 'EXP170_CONVNEXT_T_L2/outputs/results' /
                f'EXP170_CONVNEXT_T_L2_S42_F{fold}/oof_k36_flip_tta.csv']
    return [base / (f'EXP149_L2_S42_F{fold}' if fold < 2 else f'EXP157_L2_S42_F{fold}') / 'oof_k36.csv',
            base / f'EXP157_L2_S43_F{fold}/oof_k36.csv']


def _load_model(root, name, rosters, provenance):
    parts = []
    for fold, (train, val) in enumerate(rosters):
        seed_parts = []
        for path in _model_paths(root, name, fold):
            _record(path, root, provenance)
            inventory = path.parent / 'output-inventory.json'
            if inventory.is_file():
                _record(inventory, root, provenance)
                entry = json.loads(inventory.read_text(encoding='utf-8')).get(path.name)
                if not entry or entry.get('bytes') != path.stat().st_size or entry.get('sha256') != _sha(path):
                    raise ValueError('Prediction output inventory differs: ' + str(path))
            run_provenance = path.parent / 'provenance.json'
            if run_provenance.is_file():
                _record(run_provenance, root, provenance)
                saved = json.loads(run_provenance.read_text(encoding='utf-8'))
                if 'fold' in saved and int(saved['fold']) != fold:
                    raise ValueError('Run provenance fold differs: ' + str(path))
                for key, roster in (('train_ids_ordered_sha256', train), ('eval_ids_ordered_sha256', val)):
                    # EXP170 F1-F4 ships sorted copies of the same registered EXP099 sets.
                    accepted = {hashlib.sha256('\n'.join(order).encode('utf-8')).hexdigest()
                                for order in (roster, sorted(roster))}
                    if key in saved and saved[key] not in accepted:
                        raise ValueError('Run provenance roster hash differs: ' + str(path))
            frame = _table(path, UID, TARGETS, probabilities=True)
            if set(frame.index) != set(val) or len(frame) != len(val):
                raise ValueError('OOF study set does not equal held-out fold: ' + str(path))
            value = frame.loc[val].to_numpy(dtype=float)
            seed_parts.append(rankdata(value, axis=0, method='average') / len(val))
        parts.append(np.mean(seed_parts, axis=0))
    return np.concatenate(parts)


def _components(root, ids, provenance):
    path = root / COMPONENTS
    _record(path, root, provenance)
    by_id = {}
    for component in json.loads(path.read_text(encoding='utf-8')):
        for uid in component['members']:
            if uid in by_id:
                raise ValueError('Study occurs in more than one validation component')
            by_id[uid] = component['component_id']
    if any(uid not in by_id for uid in ids):
        raise ValueError('Validation component coverage incomplete; no singleton fallback')
    names = sorted({by_id[uid] for uid in ids})
    lookup = {name: i for i, name in enumerate(names)}
    return np.asarray([lookup[by_id[uid]] for uid in ids]), len(names)


def auc(y, p, weights=None):
    """Tie-correct weighted Mann-Whitney AUC; undefined classes return NaN."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[ok], p[ok], w[ok]
    if not np.isin(y, (0, 1)).all():
        raise ValueError('AUC labels must be binary')
    _, bins = np.unique(p, return_inverse=True)
    pos = np.bincount(bins, weights=w * y)
    neg = np.bincount(bins, weights=w * (1-y), minlength=len(pos))
    P, N = pos.sum(), neg.sum()
    if not P or not N:
        return float('nan')
    return float((pos * (np.cumsum(neg) - neg / 2)).sum() / (P * N))


def _ci(values, requested):
    values = np.asarray(values, float)
    good = values[np.isfinite(values)]
    if requested < 20 or len(good) < max(20, int(np.ceil(.8 * requested))):
        return None
    return list(np.quantile(good, [.025, .975]))


def train_terciles(values, splits):
    """Frozen train-only quantiles; ties stay together and empty bins are honest."""
    values = np.asarray(values, float)
    train = values[(np.asarray(splits) == 'train') & np.isfinite(values)]
    if len(train) < 3:
        return None, np.full(len(values), 'missing', object)
    cuts = np.quantile(train, [1/3, 2/3])
    bins = np.full(len(values), 'missing', object)
    ok = np.isfinite(values)
    bins[ok] = np.asarray(['low', 'mid', 'high'], object)[np.searchsorted(cuts, values[ok], side='left')]
    return list(cuts), bins


def _strata(frame, metric):
    if metric in ('noise_hh_flat', 'laplacian_var'):
        cuts, bins = train_terciles(pd.to_numeric(frame[metric], errors='coerce'), frame['split'])
        return cuts, bins, ['low', 'mid', 'high']
    vals = frame[metric].fillna('UNKNOWN').astype(str).str.strip().replace({'': 'UNKNOWN', 'nan': 'UNKNOWN', 'None': 'UNKNOWN'})
    # Preserve mixed/multi-series metadata as its own category rather than inventing a scanner.
    return None, vals.to_numpy(), sorted(vals.unique())


def _score_axis(Y, P, selections, component_index, n_components, bootstraps, seed, deadline):
    """Joint global component draws preserve covariance across all strata."""
    point, support = {}, []
    for name, sel in selections.items():
        value = np.asarray([auc(Y[sel, j], P[sel, j]) for j in range(len(TARGETS))])
        point[name] = value
        # A test-only scanner (or collapsed empty tercile) must not erase the
        # training strata's macro support. Its own AUC remains undefined.
        if np.any(sel):
            support.append(np.isfinite(value))
    common = np.logical_and.reduce(support) if support else np.zeros(12, bool)
    draws = {name: np.full((bootstraps, 12), np.nan) for name in selections}
    rng = np.random.default_rng(seed)
    for b in range(bootstraps):
        counts = np.bincount(rng.integers(0, n_components, n_components), minlength=n_components)
        w = counts[component_index]
        for name, sel in selections.items():
            for j in np.flatnonzero(np.isfinite(point[name])):
                draws[name][b, j] = auc(Y[sel, j], P[sel, j], w[sel])
    result = {}
    for name, sel in selections.items():
        per_target = {}
        for j, target in enumerate(TARGETS):
            labels = Y[sel, j]
            labels = labels[np.isfinite(labels)]
            p = int((labels == 1).sum()); n = int((labels == 0).sum())
            per_target[target] = {'n': p+n, 'positive': p, 'negative': n,
                                  'auc': point[name][j], 'low_count': min(p, n) < 10,
                                  'ci95': _ci(draws[name][:, j], bootstraps),
                                  'valid_draws': int(np.isfinite(draws[name][:, j]).sum())}
        macro_draws = np.mean(draws[name][:, common], axis=1) if common.any() else np.full(bootstraps, np.nan)
        result[name] = {'studies': int(sel.sum()), 'components': int(len(np.unique(component_index[sel]))),
                        'targets': per_target,
                        'macro_all12': float(np.mean(point[name])) if np.isfinite(point[name]).all() else None,
                        'macro_fixed_support': float(np.mean(point[name][common])) if common.any() and np.isfinite(point[name][common]).all() else None,
                        'macro_ci95': _ci(macro_draws, bootstraps),
                        'valid_macro_draws': int(np.isfinite(macro_draws).sum())}
    contrasts = {}
    if 'low' in result and 'high' in result:
        delta = point['high'][common] - point['low'][common]
        ds = np.mean(draws['high'][:, common] - draws['low'][:, common], axis=1) if common.any() else []
        contrasts['high_minus_low'] = {'macro_delta': float(np.mean(delta)) if common.any() and np.isfinite(delta).all() else None,
                                        'ci95': _ci(ds, bootstraps),
                                        'valid_draws': int(np.isfinite(ds).sum()),
                                        'meaning': 'association; high sharpness means sharper, high noise means noisier'}
    return {'fixed_support_targets': [t for j, t in enumerate(TARGETS) if common[j]],
            'strata': result, 'contrasts': contrasts}


def _competition(root, metrics, result, bootstraps, seed, deadline):
    frame = metrics[(metrics.source == 'COMPETITION') & (metrics.domain == 'production')].copy()
    if frame.empty:
        result['missing'].append('competition production metrics absent')
        return
    provenance = result['input_sha256']
    try:
        rosters = _rosters(root, provenance)
        ids = [u for _, val in rosters for u in val]
        component_index, n_components = _components(root, ids, provenance)
    except (FileNotFoundError, ValueError) as exc:
        result['missing'].append('Competition OOF held: ' + str(exc))
        return
    result['oof_validation'] = {'folds': list(range(5)), 'studies': len(ids), 'components': n_components,
                                'held_out_sets_exact': True, 'train_is_validation_complement': True,
                                'run_validation': 'available output inventories and fold/train/eval provenance hashes checked; absent run provenance is not independently reconstructed',
                                'semantics': 'held-out predictions against weak JEV/Dread references, not adjudicated ground truth'}
    models, references = {}, {}
    for name in ('L2_ConvNeXt_T', 'L2_RadR50'):
        try:
            models[name] = _load_model(root, name, rosters, provenance)
        except (FileNotFoundError, ValueError) as exc:
            result['missing'].append(name + ' pooled OOF held: ' + str(exc))
    for name, relative in REFERENCES.items():
        try:
            path = root / relative
            _record(path, root, provenance)
            table = _table(path, UID, TARGETS, binary=name == 'jev')
            labels = table.reindex(ids).to_numpy(float)
            result['joins'].append({'scope': 'competition_reference', 'reference': name,
                                    'pooled_oof_studies': len(ids), 'label_row_present': int(sum(u in table.index for u in ids)),
                                    'label_row_missing': int(sum(u not in table.index for u in ids)),
                                    'finite_labels_per_target': {t: int(np.isfinite(labels[:, j]).sum()) for j, t in enumerate(TARGETS)}})
            if not np.isfinite(labels).any():
                raise ValueError('No finite labels on pooled held-out studies')
            if not np.isfinite(labels).all():
                result['missing'].append(name + ': incomplete held-out reference labels; per-target support reported')
            if name == 'dread':
                if ((labels[np.isfinite(labels)] < 0) | (labels[np.isfinite(labels)] > 1)).any():
                    raise ValueError('Dread labels outside [0,1]')
                labels = np.rint(labels.astype(np.float32)).astype(float)
            references[name] = labels
        except (FileNotFoundError, ValueError) as exc:
            result['missing'].append(name + ' reference held: ' + str(exc))
    outputs = result['competition']
    for group, subset in frame.groupby('group', sort=True):
        if subset.uid.duplicated().any():
            raise ValueError('Duplicate competition production uid/group metrics')
        aligned = subset.set_index('uid').reindex(ids)
        split_match = aligned['split'].eq('train').to_numpy()
        for metric in ('noise_hh_flat', 'laplacian_var', 'vendor', 'field_strength', 'study_vendor', 'study_field_strength'):
            if metric not in subset:
                continue
            cuts, bins, names = _strata(subset, metric)
            bin_by_id = dict(zip(subset.uid.astype(str), bins))
            pooled_bins = np.asarray([bin_by_id.get(u, 'missing') for u in ids])
            selections = {name: (pooled_bins == name) & split_match for name in names if name != 'missing'}
            mix = {split: {name: int(((bins == name) & (subset.split.to_numpy() == split)).sum())
                           for name in names} for split in ('train', 'test', 'unknown')}
            for model, P in models.items():
                for reference, Y in references.items():
                    scored = _score_axis(Y, P, selections, component_index, n_components, bootstraps, seed, deadline)
                    outputs.append({'group': group, 'metric': metric, 'model': model, 'reference': reference,
                                    'metadata_scope': 'supplemental study acquisition summary; not selected-series metadata' if metric.startswith('study_') else 'selected-series / production image group',
                                    'train_quantile_cuts': cuts, 'mix': mix,
                                    'metric_missing_or_unsampled_oof': int((pooled_bins == 'missing').sum()), **scored})
    result['competition'] = outputs
    result['prediction_policy'] = 'within-fold target percentile ranks; RadR50 S42/S43 rank mean; ConvNeXt flip-TTA; concatenate five held-out folds'


def _association(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or len(np.unique(x[ok])) < 2 or len(np.unique(y[ok])) < 2:
        return {'n': int(ok.sum()), 'spearman': None}
    return {'n': int(ok.sum()), 'spearman': float(spearmanr(x[ok], y[ok]).statistic)}


def _external(root, metrics, result, out, deadline):
    rows, associations = [], result['external']
    production = metrics[(metrics.source != 'COMPETITION') & (metrics.domain == 'production')]
    for actual, source_frame in production.groupby('source', sort=True):
        source = SOURCE_ALIASES.get(actual, actual)
        pred_path = root / f'artifacts/experiments/EXP137_EXT_QA/{source}/predictions_full.csv'
        if not pred_path.is_file():
            result['missing'].append(f'{actual}: saved full zero-shot predictions absent')
            continue
        try:
            _record(pred_path, root, result['input_sha256'])
            pred = _table(pred_path, 'study_key', TARGETS, probabilities=True)
            mapped_path = root / f'storage/labels/external/{source}_12.csv'
            _record(mapped_path, root, result['input_sha256'])
            mapped = _table(mapped_path, 'study_key', TARGETS, binary=True)
            receipt_path = pred_path.parent / 'zeroshot_full.json'
            _record(receipt_path, root, result['input_sha256'])
            saved = json.loads(receipt_path.read_text(encoding='utf-8'))
            for key, path in (('predictions_sha256', pred_path), ('mapped_labels_sha256', mapped_path)):
                if saved.get(key) != _sha(path):
                    raise ValueError('Saved zero-shot receipt hash differs: ' + key)
            cache_name = f'storage/cache/derived/EXT_{actual}_RAPTOR64_336_C140/manifest.json'
            hashes = {k.replace('\\', '/'): v for k, v in result['metrics_provenance']['input_hashes'].items()}
            bound_hash = hashes.get(cache_name)
            if not bound_hash or saved.get('cache_manifest_sha256') != bound_hash:
                raise ValueError('Saved zero-shot production cache differs from original image metrics')
            result['joins'].append({'scope': 'external_source', 'source': actual, 'prediction_rows': len(pred),
                                    'mapped_label_rows': len(mapped),
                                    'prediction_label_row_present': int(sum(u in mapped.index for u in pred.index)),
                                    'prediction_label_row_missing': int(sum(u not in mapped.index for u in pred.index))})
            comparisons = [('mapped:' + t, mapped[t], pred[t]) for t in TARGETS]
            if source in NATIVE:
                native_path = root / f'storage/labels/external/{source}_native.csv'
                _record(native_path, root, result['input_sha256'])
                native = _table(native_path, 'study_key', [n for n, _ in NATIVE[source]], binary=True)
                if saved and saved.get('native_labels_sha256') != _sha(native_path):
                    raise ValueError('Saved zero-shot native-label hash differs')
                comparisons.extend(('native:' + n, native[n], pred[list(cols)].max(axis=1)) for n, cols in NATIVE[source])
        except (FileNotFoundError, ValueError) as exc:
            result['missing'].append(f'{actual}: zero-shot linkage held: {exc}')
            continue
        for group, frame in source_frame.groupby('group', sort=True):
            if frame.uid.duplicated().any():
                raise ValueError('Duplicate external production uid/group metrics')
            frame = frame.set_index('uid')
            for comparison, labels, prediction in comparisons:
                ids = frame.index.intersection(prediction.index).intersection(labels.index)
                y = labels.loc[ids].to_numpy(float); p = prediction.loc[ids].to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(p)
                result['joins'].append({'scope': 'external_comparison', 'source': actual, 'group': group,
                                        'comparison': comparison, 'metric_studies': len(frame),
                                        'prediction_row_present': int(sum(u in prediction.index for u in frame.index)),
                                        'label_row_present': int(sum(u in labels.index for u in frame.index)),
                                        'row_intersection': len(ids), 'finite_target_labels': int(ok.sum())})
                ids, y, p = ids[ok], y[ok], p[ok]
                if not len(ids):
                    continue
                losses = {'brier': (p-y)**2,
                          'log_loss': -(y*np.log(np.clip(p, 1e-6, 1-1e-6))+(1-y)*np.log(np.clip(1-p, 1e-6, 1-1e-6))),
                          'correct_05': ((p >= .5) == y).astype(float)}
                positives = int(y.sum()); negatives = int(len(y)-y.sum())
                for metric in ('noise_hh_flat', 'laplacian_var'):
                    if metric not in frame:
                        continue
                    x = pd.to_numeric(frame.loc[ids, metric], errors='coerce').to_numpy(float)
                    for loss, values in losses.items():
                        a = {'source': actual, 'group': group, 'comparison': comparison, 'metric': metric,
                             'outcome': loss, 'positive': positives, 'negative': negatives,
                             'mean_outcome': float(values.mean()), 'unmatched_metric_studies': int(len(frame)-len(ids)),
                             **_association(x, values)}
                        # Class-conditioned correlations expose prevalence confounding, but cannot remove case-mix effects.
                        a['within_class'] = {str(c): _association(x[y == c], values[y == c]) for c in (0, 1)}
                        associations.append(a)
                for i, uid in enumerate(ids):
                    rows.append({'source': actual, 'group': group, 'uid': uid, 'comparison': comparison,
                                 'brier': losses['brier'][i], 'log_loss': losses['log_loss'][i],
                                 'correct_05': losses['correct_05'][i]})
    if rows:
        pd.DataFrame(rows).to_csv(out / 'zeroshot_study_errors.csv', index=False)
    result['external'] = associations
    result['external_interpretation'] = ('Frozen zero-shot scores include rank blending; Brier/log loss and threshold 0.5 correctness are descriptive, '
        'not calibrated clinical accuracy. ACL severity is collapsed in mapped labels; KneeMRI any/complete native labels are separate. '
        'Native max-meniscus is evaluated as any-meniscus. Label definitions, prevalence, anatomy and missing-series selection confound associations. '
        'No external source is adjudicated competition ground truth; no new training or external test labels are loaded.')


def run_behavior(root: Path, metrics: pd.DataFrame, out: Path, bootstraps=200, seed=175, deadline=None,
                 *, metrics_provenance: dict, verification: dict) -> dict:
    """Write the complete descriptive readout; legacy ``deadline`` is ignored."""
    root, out = Path(root), Path(out)
    required = {'source', 'split', 'uid', 'domain', 'group'}
    if not required.issubset(metrics.columns):
        raise ValueError('Behavior metrics missing columns: ' + ', '.join(sorted(required-set(metrics.columns))))
    if bootstraps < 0:
        raise ValueError('bootstraps must be nonnegative')
    if not isinstance(metrics_provenance.get('input_hashes'), dict):
        raise ValueError('Original image metrics provenance is required')
    if not all(verification.get(k) for k in ('path', 'sha256', 'verified_tool_version', 'source_hashes')):
        raise ValueError('Verification provenance is required')
    out.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'exp175-behavior-v1', 'version': VERSION, 'status': 'COMPLETE',
              'metrics_provenance': metrics_provenance, 'verification': verification,
              'input_sha256': {}, 'missing': [], 'joins': [], 'competition': [], 'external': [],
              'bootstrap': {'unit': 'validation component', 'draws': int(bootstraps), 'seed': int(seed),
                            'ci': 'percentile 95%; fixed target support across compared strata; at least 80% valid draws and 20 draws required'},
              'cautions': ['Observational associations do not identify causality or imply an augmentation/denoising gain.',
                           'Noise/sharpness and scanner mix are measured on sampled production arrays; unsampled held-out studies are omitted.',
                           'Train-defined tercile cuts are shared with test; test labels are never loaded.',
                           'Sparse categories have null AUC/CI; multiplicity and weak-label scanner dependence preclude confirmatory claims.']}
    _competition(root, metrics, result, int(bootstraps), int(seed), deadline)
    _external(root, metrics, result, out, deadline)
    if result['status'] == 'COMPLETE' and result['missing']:
        result['status'] = 'COMPLETE_WITH_GAPS'
    result = _clean(result)
    (out / 'behavior.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result
