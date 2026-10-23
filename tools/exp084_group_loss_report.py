"""Prepare then evaluate the three fixed EXP084 fits once, on CPU only.

Import is passive. No training modules, models, Gold, or prior predictions are
loaded. Full-validation ranks are formed once before any subgroup/bootstrap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ARMS = ('RN_clip', 'RN_ones', 'B')
CONTRASTS = {
    'primary_blend_ones_minus_clip': ('blend_clip', 'blend_ones'),
    'secondary_rn_ones_minus_clip': ('RN_clip', 'RN_ones'),
    'secondary_blend_clip_minus_B': ('B', 'blend_clip'),
    'secondary_blend_ones_minus_B': ('B', 'blend_ones'),
}
SCHEMA = 'exp084-group-loss-report-v1'
EVIDENCE = 'docs/handoffs/evidence/2026-10/2026-10-04-exp084-report-tool'
LIMITS = [
    'Weak report-derived labels and prior researcher exposure; this regrouping is not an untouched clinical holdout.',
    'Known report/duplicate components are not proven patient IDs; unknown patient and near-image duplicates remain.',
    'External Dread label-generator ancestry and calibration are not independently certified.',
    'Bootstrap is conditional on these fixed labels, scores, trained models and one seed; it excludes label, training-seed, researcher-selection and private-distribution uncertainty.',
    'Scanner and missing-pixel groups are exploratory; no subgroup CI or multiple-comparison correction is claimed.',
    'Cached zero slices are an image-input diagnostic, not a clinical negative label or proof of acquisition absence.',
    'No input-A arm: this comparison cannot establish resolution of historical B-minus-A Fracture loss.',
    'No exact D01, public 0.943, private-LB or clinical improvement claim; no automatic adoption or ratio/epoch/target search.',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root, relative):
    value = str(relative)
    if value.startswith('workspace://'):
        value = value[len('workspace://'):]
    p = Path(value)
    if p.is_absolute() or p.drive or '..' in p.parts:
        raise ValueError('Require contained relative workspace path')
    target = Path(root) / p
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes workspace')
    return target


def relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def binary_labels(soft):
    values = np.asarray(soft, dtype=np.float32)
    if (values.ndim != 2 or values.shape[1] != 12 or not len(values)
            or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError('Expected finite n by12 soft labels in [0,1]')
    return np.rint(values).astype(np.int8)


def checked_binary(y):
    y = np.asarray(y)
    if (y.ndim != 2 or y.shape[1] != 12 or not len(y)
            or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all()):
        raise ValueError('Expected binary n by12 labels')
    return y


def scores_for(b, clip, ones):
    arrays = [np.asarray(x, dtype=np.float64) for x in (b, clip, ones)]
    if (arrays[0].ndim != 2 or arrays[0].shape[1] != 12 or not len(arrays[0])
            or any(x.shape != arrays[0].shape or not np.isfinite(x).all()
                   or ((x < 0) | (x > 1)).any() for x in arrays)):
        raise ValueError('Expected matching finite n by12 probabilities in [0,1]')
    rb, rc, ro = [pd.DataFrame(x).rank(method='average', pct=True).to_numpy() for x in arrays]
    return {'B': rb, 'RN_clip': arrays[1], 'RN_ones': arrays[2],
            'blend_clip': .9 * rb + .1 * rc, 'blend_ones': .9 * rb + .1 * ro}


def weighted_auc(y, prediction, weights):
    """Tie-aware weighted AUC, equivalent to integer-multiplicity repeated rows.

    Pure algorithm retained from oof_error_audit, without importing its training
    label adapters or historical experiment constants.
    """
    y, p, w = [np.asarray(x) for x in (y, prediction, weights)]
    if (y.ndim != 1 or not len(y) or p.shape != y.shape or w.shape != y.shape
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or not np.isfinite(w).all() or (w < 0).any()):
        raise ValueError('Invalid weighted AUC inputs')
    order = np.argsort(p, kind='stable')
    y, p, w = y[order], p[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    pos, neg = np.add.reduceat(w * y, starts), np.add.reduceat(w * (1 - y), starts)
    denominator = pos.sum() * neg.sum()
    if denominator == 0:
        return np.nan
    return float(np.sum(pos * (np.cumsum(neg) - neg + .5 * neg)) / denominator)


def target_rows(y, scores, weights=None):
    y = checked_binary(y)
    if set(scores) != {'B', 'RN_clip', 'RN_ones', 'blend_clip', 'blend_ones'}:
        raise ValueError('Require exactly all fixed score arms')
    if any(v.shape != y.shape or not np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Invalid fixed score shape/values')
    w = np.ones(len(y), dtype=np.int64) if weights is None else np.asarray(weights)
    rows = []
    for j, target in enumerate(TARGETS):
        p, n = int(y[:, j].sum()), int(len(y) - y[:, j].sum())
        auc = {name: weighted_auc(y[:, j], values[:, j], w) for name, values in scores.items()}
        auc = {name: value if np.isfinite(value) else None for name, value in auc.items()}
        delta = {name: None if auc[a] is None else auc[b] - auc[a]
                 for name, (a, b) in CONTRASTS.items()}
        rows.append(dict(target=target, P=p, N=n, auc=auc, delta=delta,
                         delta_over_12={key: None if v is None else v / 12 for key, v in delta.items()}))
    return rows


def aggregate(rows):
    complete = len(rows) == 12 and all(r['delta'][next(iter(CONTRASTS))] is not None for r in rows)
    result = {'complete12': complete, 'auc': {}, 'contrasts': {}}
    for arm in ('B', 'RN_clip', 'RN_ones', 'blend_clip', 'blend_ones'):
        result['auc'][arm] = float(np.mean([r['auc'][arm] for r in rows])) if complete else None
    for name in CONTRASTS:
        result['contrasts'][name] = {
            'macro12': float(np.mean([r['delta'][name] for r in rows])) if complete else None,
            'Fracture': next(r['delta'][name] for r in rows if r['target'] == 'Fracture'),
            'PFOA': next(r['delta'][name] for r in rows if r['target'] == 'PF OA'),
            'MCL': next(r['delta'][name] for r in rows if r['target'] == 'MCL'),
        }
    return result


def component_weights(codes, draw, count):
    return np.bincount(draw, minlength=count)[codes]


def paired_bootstrap(y, scores, components, reps=500, seed=20261003):
    y = checked_binary(y)
    components = np.asarray(components, dtype=str)
    groups, codes = np.unique(components, return_inverse=True)
    if len(codes) != len(y) or not len(groups) or reps < 1 or (components == '').any():
        raise ValueError('Invalid component bootstrap inputs')
    rng = np.random.default_rng(seed)
    draws_hash = hashlib.sha256()
    invalid_targets = np.zeros(12, dtype=np.int64)
    values, invalid = [], 0
    for _ in range(reps):
        draw = rng.integers(0, len(groups), len(groups))
        draws_hash.update(draw.astype('<i8').tobytes())
        weights = component_weights(codes, draw, len(groups))
        bad = ((weights[:, None] * y).sum(0) == 0) | ((weights[:, None] * (1 - y)).sum(0) == 0)
        if bad.any():
            invalid += 1
            invalid_targets += bad
            continue
        rows = target_rows(y, scores, weights)
        values.append([[r['delta'][name] for r in rows] for name in CONTRASTS])
    array = np.asarray(values)
    def ci(v):
        return np.quantile(v, [.025, .975]).tolist() if len(v) else None
    intervals = {}
    for k, name in enumerate(CONTRASTS):
        intervals[name] = dict(macro12=ci(array[:, k].mean(1)) if len(values) else None,
                              targets={t: ci(array[:, k, j]) if len(values) else None
                                       for j, t in enumerate(TARGETS)})
    return dict(attempted=reps, valid=len(values), invalid=invalid,
                invalid_targets=dict(zip(TARGETS, map(int, invalid_targets))),
                component_count=len(groups), seed=seed, draws_sha256=draws_hash.hexdigest(),
                same_draw_all_arms_and_targets=True, ranks_recomputed=False, intervals=intervals,
                scope='Conditional fixed-score component bootstrap; every interval uses the same complete12-target valid draws.')


def verify_bindings(root, bindings):
    for path, expected in bindings.items():
        if sha(contained(root, path)) != expected:
            raise ValueError('Frozen artifact hash mismatch: ' + path)


def validate_protocol(m):
    if (m.get('schema') != 'exp084-group-r34-loss-v1' or m['targets'] != TARGETS
            or m['arm_order'] != list(ARMS) or set(m['arms']) != set(ARMS)
            or m['execution']['new_fits'] != 3 or m['execution']['all_three_before_metrics'] is not True):
        raise ValueError('Require the exact registered three-fit EXP084 contract')
    ev = m['evaluation']
    if (ev['primary'] != 'macro12(blend_ones)-macro12(blend_clip)' or ev['weights'] != [.9, .1]
            or ev['rank'] != 'average tie rank/N on full V; no final/subgroup/bootstrap rerank'
            or ev['bootstrap_reps'] != 500 or ev['bootstrap_seed'] != 20261003
            or ev['bootstrap_unit'] != 'component_id' or ev['no_ratio_epoch_target_search'] is not True
            or m['split']['salt'] != 'RN_GROUP_SCREEN_V1' or m['split']['no_redraw'] is not True):
        raise ValueError('Changed fixed evaluation/split contract')
    if (m['paths']['labels'] != m['paths']['eval_labels'] or m['label_hash'] != m['eval_label_hash']
            or m['labels']['path'] != m['paths']['labels']
            or m['labels']['semantic_sha256'] != m['label_hash']
            or m['groups']['path'] != m['paths']['fold_file']
            or m['groups']['sha256'] != m['split_hash']
            or m['groups']['component_column'] != 'component_id'
            or m['groups']['subgroup_columns'] != ['scanner_group', 'field', 'manufacturer', 'cached_mask']):
        raise ValueError('Changed label/group reference contract')
    n_train, n_val = m['expected_ids']['train'], m['expected_ids']['val']
    if n_train < 8 or n_val < 2:
        raise ValueError('Empty/insufficient frozen cohort')
    counts = m['reference_counts']
    if ([r['target'] for r in counts] != TARGETS
            or any(type(r[k]) is not int or r[k] < 0 for r in counts for k in ('P', 'N'))
            or any(r['P'] + r['N'] != n_val for r in counts)
            or m['macro12_valid'] != all(r['P'] > 0 and r['N'] > 0 for r in counts)):
        raise ValueError('Invalid predeclared P/N counts')
    clip = np.asarray(m['loss']['baseline_pos_weight'], dtype=np.float32)
    if clip.shape != (12,) or not np.isfinite(clip).all() or ((clip < 1) | (clip > 10)).any():
        raise ValueError('Invalid T-only loss vector')
    for arm in ARMS:
        spec = m['arms'][arm]
        args = spec['expected_args']
        epochs = 8 if arm == 'B' else 20
        if (spec['id'] != arm or args['epochs'] != epochs or args['fixed_epoch'] != epochs
                or args['bs'] != 8 or args['accum'] != 1 or args['seed'] != 42
                or args['amp'] != 'fp16' or args['eval_amp'] != 'fp16'
                or args['runtime_policy'] != 'strict' or args['window_policy'] != 'within-slot-valid'
                or spec['expected_attempts'] != epochs * (n_train // 8)):
            raise ValueError('Changed fixed arm budget/runtime')
        wanted = np.ones(12, dtype=np.float32) if arm == 'RN_ones' else clip
        policy = 'ones_positive_term' if arm == 'RN_ones' else 'clipped_positive_term'
        if spec['loss_policy'] != policy or not np.array_equal(np.asarray(spec['actual_pos_weight'], dtype=np.float32), wanted):
            raise ValueError('Changed arm loss intervention')
    a, b = m['arms']['RN_clip'], m['arms']['RN_ones']
    if a['expected_args'] != b['expected_args'] or a['initial_state_sha256'] != b['initial_state_sha256']:
        raise ValueError('RN recipes or initial states differ beyond registered loss')


def cohort_metadata(root, m):
    ordered = read(contained(root, m['ordered_ids']['path']))
    if set(ordered) != {'train', 'val'}:
        raise ValueError('Expected explicit train/val ordered IDs')
    for part in ('train', 'val'):
        ids = ordered[part]
        if (not ids or not all(isinstance(x, str) and x for x in ids) or ids != sorted(set(ids))
                or len(ids) != m['expected_ids'][part]
                or uid_hash(ids) != m['ordered_ids'][part + '_sha256']):
            raise ValueError('Frozen ordered UID mismatch')
    if set(ordered['train']) & set(ordered['val']):
        raise ValueError('Study crosses train/validation')
    groups = pd.read_csv(contained(root, m['groups']['path']), dtype=str, keep_default_na=False)
    required = [UID, 'component_id', 'report_group_sha256', 'split', *m['groups']['subgroup_columns']]
    if (list(groups.columns) != required or not groups[UID].is_unique or (groups[UID] == '').any()
            or set(groups[UID]) != set(ordered['train'] + ordered['val'])
            or set(ordered['train'] + ordered['val']) & set(m['excluded_uids'])):
        raise ValueError('Eligible group coverage/column/exclusion mismatch')
    groups = groups.set_index(UID)
    for part in ('train', 'val'):
        selected = groups.loc[ordered[part]]
        if set(selected['split']) != {part} or selected['component_id'].nunique() != m['expected_ids'][part + '_components']:
            raise ValueError('Component count/split mismatch')
    for component, frame in groups.groupby('component_id', sort=True):
        if not re.fullmatch('[0-9a-f]{64}', component):
            raise ValueError('Malformed fixed component ID')
        value = int.from_bytes(hashlib.sha256(('RN_GROUP_SCREEN_V1:' + component).encode()).digest(), 'big')
        expected = 'val' if value % 5 == 2 else 'train'
        if set(frame['split']) != {expected}:
            raise ValueError('Component crosses split or fixed salt assignment changed')
    if (groups.groupby('report_group_sha256')['split'].nunique() > 1).any():
        raise ValueError('Report group crosses train/validation')
    return ordered, groups


def prepare(protocol, protocol_sha256, root=ROOT):
    """Freeze metadata/source/output contracts only; never read label or prediction values."""
    root = Path(root)
    protocol_path = contained(root, protocol)
    if sha(protocol_path) != protocol_sha256:
        raise ValueError('Protocol hash mismatch')
    m = read(protocol_path)
    validate_protocol(m)
    bindings = {relative(root, protocol_path): protocol_sha256, **m['source_bindings']}
    bindings.update({'training/src/' + k: v for k, v in m['training_code_sha256'].items()})
    for key in ('labels', 'ordered_ids', 'groups'):
        ref = m[key]
        if ref['path'] in bindings and bindings[ref['path']] != ref['sha256']:
            raise ValueError('Conflicting registered hash for ' + key)
        bindings[ref['path']] = ref['sha256']
    queue_path = protocol_path.parent / 'queue.json'
    bindings[relative(root, queue_path)] = sha(queue_path)
    for path in ('tools/exp084_group_loss_report.py', 'tests/test_exp084_group_loss_report.py'):
        observed = sha(contained(root, path))
        if path in bindings and observed != bindings[path]:
            raise ValueError('Evaluator changed since experiment registration')
        bindings[path] = observed
    verify_bindings(root, bindings)
    ordered, _ = cohort_metadata(root, m)
    queue = read(queue_path)
    if [item['tag'] for item in queue] != [m['arms'][a]['tag'] for a in ARMS]:
        raise ValueError('Queue arm ordering differs from registration')
    complete = [contained(root, m['arms'][a]['logs']).joinpath('SUCCESS').exists() for a in ARMS]
    if all(complete):
        raise ValueError('Evaluation contract must be frozen before all three fits complete')
    output = EVIDENCE + '/evaluation-v1'
    if contained(root, output).exists():
        raise FileExistsError('Frozen evaluation output already exists')
    paths = {a: dict(outputs=m['arms'][a]['outputs'], logs=m['arms'][a]['logs']) for a in ARMS}
    for spec in paths.values():
        for value in [*spec['outputs'].values(), spec['logs']]:
            contained(root, value)
    registration = dict(schema=SCHEMA, prepared_at=datetime.now(timezone.utc).isoformat(),
        protocol=relative(root, protocol_path), protocol_sha256=protocol_sha256,
        queue=relative(root, queue_path), source_bindings=bindings, paths=paths, output=output,
        targets=TARGETS, arms=list(ARMS), contrasts=CONTRASTS,
        bootstrap=dict(reps=500, seed=20261003, unit='component_id', same_draw_all_arms_and_targets=True,
                       invalid='Any single-class target invalidates whole draw; all CIs use complete12 draws'),
        reference_counts=m['reference_counts'], expected_ids=m['expected_ids'],
        ordered_id_sha256={k: uid_hash(v) for k, v in ordered.items()},
        rank='average tie rank/N on full V; no final/subgroup/bootstrap rerank',
        limits=LIMITS, scope='Prepared without label values, predictions, Gold, GPU or model access.')
    destination = contained(root, EVIDENCE)
    destination.mkdir(parents=True, exist_ok=True)
    write_new(destination / 'registration.json', registration)
    return registration


def valid_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_execution(summary, provenance, m, arm, protocol_sha256):
    spec = m['arms'][arm]
    epochs = spec['expected_args']['epochs']
    if summary['provenance'] != provenance:
        raise ValueError('Summary/provenance mismatch')
    expected = dict(code_sha256=m['training_code_sha256'], initial_state_sha256=spec['initial_state_sha256'],
        train_ids_sha256=m['ordered_ids']['train_sha256'], eval_ids_sha256=m['ordered_ids']['val_sha256'],
        train_ids_ordered_sha256=m['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=m['ordered_ids']['val_sha256'],
        corpus_manifest_sha256=m['corpus_manifest_sha256'], checkpoint_selection='fixed-final',
        sampling='sha256(seed,epoch,uid,stream)-v1', gold_access='disabled', loss_policy=spec['loss_policy'],
        actual_pos_weight=spec['actual_pos_weight'], baseline_pos_weight=m['loss']['baseline_pos_weight'],
        metric_access='deferred_until_all_three_complete', protocol_sha256=protocol_sha256,
        adapter_sha256=m['source_bindings']['training/runners/exp084_group_loss.py'],
        contract_helper_sha256=m['source_bindings']['tools/exp084_group_loss.py'])
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual provenance differs from frozen source/UID/init/loss/runtime contract')
    for key in ('runtime', 'eval_runtime'):
        if provenance[key] != m['expected_runtime'][key]:
            raise ValueError('Strict runtime mismatch')
    for key in ('torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'pydicom', 'cv2'):
        if summary['environment'].get(key) != m['environment'][key]:
            raise ValueError('Runtime environment differs')
    if any(summary['args'].get(k) != v for k, v in spec['expected_args'].items()):
        raise ValueError('Actual fixed arguments differ')
    required = dict(gold=0, gold_access='disabled', metric_access='deferred_until_all_three_complete',
        loss_policy=spec['loss_policy'], arm=arm, selected_checkpoint='best', best_epoch=epochs - 1,
        selected_epoch_1based=epochs, max_epochs=epochs, train=m['expected_ids']['train'],
        val=m['expected_ids']['val'], rows=m['expected_ids']['val'], no_oof_epoch_selection=True,
        weak_oof_macro_auc=None, oof_per_target=None, label_hash=m['label_hash'],
        eval_label_hash=m['eval_label_hash'], split_hash=m['split_hash'], expected_attempts=spec['expected_attempts'])
    if any(summary.get(k) != v for k, v in required.items()):
        raise ValueError('Completed arm metadata/exposure/fixed-epoch mismatch')
    updates, skips = summary['optimizer_updates'], summary['skipped_updates']
    if (not valid_count(updates) or not valid_count(skips) or updates <= 0
            or updates + skips != spec['expected_attempts']):
        raise ValueError('Invalid effective update/attempt counts')
    fraction = skips / (updates + skips)
    if fraction >= .01 or summary['optimizer_skip_fraction'] != fraction:
        raise ValueError('Optimizer skip fraction requires review before scoring')
    history = summary['history']
    if len(history) != epochs:
        raise ValueError('Incomplete fixed-epoch history')
    old_u = old_s = 0
    per_epoch = m['expected_ids']['train'] // 8
    for ep, h in enumerate(history):
        u, s = h['optimizer_updates'], h['skipped_updates']
        if (h['epoch'] != ep or not valid_count(u) or not valid_count(s) or u < old_u or s < old_s
                or u + s != (ep + 1) * per_epoch or not np.isfinite(h['loss'])
                or h.get('oof_auc') is not None or h.get('gold_auc') is not None
                or h['selection'] != float(ep)):
            raise ValueError('Invalid epoch history, intermediate metrics or realized budget')
        old_u, old_s = u, s
    if (old_u, old_s) != (updates, skips):
        raise ValueError('Final update history mismatch')
    if summary['results'] != {'best': {'oof_rows': m['expected_ids']['val'], 'metric_access': 'deferred_until_all_three_complete'}}:
        raise ValueError('Unexpected intermediate/selected metric results')
    return dict(optimizer_updates=updates, skipped_updates=skips, attempted_updates=updates + skips,
                skip_fraction=fraction, epochs=epochs, total_seconds=summary['total_seconds'],
                fixed_final=True, gold_access='disabled', metrics_deferred=True)


def require_completed(root, reg, m):
    """Check every arm's completion markers before reading summaries or predictions."""
    required = {}
    for arm in ARMS:
        spec = m['arms'][arm]
        if reg['paths'][arm] != dict(outputs=spec['outputs'], logs=spec['logs']):
            raise ValueError('Frozen output paths changed')
        jobdir = contained(root, spec['logs'])
        paths = [jobdir / 'SUCCESS', jobdir / 'job.json', *[contained(root, x) for x in spec['outputs'].values()]]
        if (jobdir / 'FAILED').exists() or any(not p.is_file() for p in paths):
            raise ValueError('All three completed SUCCESS and output artifacts required before scoring')
        if list(contained(root, spec['outputs']['summary']).parent.glob('gold_*.csv')):
            raise ValueError('Unexpected Gold output')
        if (contained(root, spec['outputs']['summary']).parent / 'model_swa.pt').exists():
            raise ValueError('Unexpected SWA output')
        required[arm] = paths
    queue = read(contained(root, reg['queue']))
    execution, observed = {}, {}
    for arm, item in zip(ARMS, queue):
        spec = m['arms'][arm]
        paths = required[arm]
        if re.fullmatch(r'exit 0 at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s*', paths[0].read_text()) is None:
            raise ValueError('Invalid successful exit marker')
        job = read(paths[1])
        if job['tag'] != spec['tag'] or item['tag'] != spec['tag'] or job['command'] != item['command']:
            raise ValueError('Actual job command differs from frozen queue')
        summary = read(contained(root, spec['outputs']['summary']))
        provenance = read(contained(root, spec['outputs']['provenance']))
        if read(contained(root, spec['outputs']['history'])) != summary['history']:
            raise ValueError('Separate history differs from summary')
        execution[arm] = validate_execution(summary, provenance, m, arm, reg['protocol_sha256'])
        for path in paths:
            observed[relative(root, path)] = sha(path)
    execution['paired_RN_equal_effective_updates'] = execution['RN_clip']['optimizer_updates'] == execution['RN_ones']['optimizer_updates']
    return execution, observed


def load_reference(root, m, ids):
    """Only run after all-three completion; preserve original float32/rint reference."""
    frame = pd.read_csv(contained(root, m['labels']['path']), dtype={UID: str})
    if frame[UID].isna().any() or not frame[UID].is_unique:
        raise ValueError('Invalid label UID coverage')
    cols = [t if t in frame.columns else t + '_soft' for t in TARGETS]
    if not set(cols) <= set(frame.columns):
        raise ValueError('Missing fixed reference target columns')
    canonical = frame.dropna(subset=cols)
    semantic = hashlib.sha256(canonical[[UID, *cols]].to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    if semantic != m['label_hash'] or not set(ids) <= set(canonical[UID]):
        raise ValueError('Fixed reference semantic hash/coverage mismatch')
    y = binary_labels(canonical.set_index(UID).loc[ids, cols].to_numpy(dtype=np.float32))
    counts = [{'target': t, 'P': int(y[:, j].sum()), 'N': int(len(y) - y[:, j].sum())} for j, t in enumerate(TARGETS)]
    if counts != m['reference_counts']:
        raise ValueError('Prepared reference P/N changed')
    return y


def predictions(path, ids):
    frame = pd.read_csv(path, dtype={UID: str})
    if list(frame.columns) != [UID, *TARGETS] or frame[UID].tolist() != ids or not frame[UID].is_unique:
        raise ValueError('Prediction UID/order/target mismatch')
    values = frame[TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid final predictions')
    return values


def run(registration, registration_sha256, root=ROOT):
    root = Path(root)
    registration_path = contained(root, registration)
    if sha(registration_path) != registration_sha256:
        raise ValueError('Analysis registration hash mismatch')
    reg = read(registration_path)
    if (reg['schema'] != SCHEMA or reg['targets'] != TARGETS or reg['arms'] != list(ARMS)
            or reg['contrasts'] != {k: list(v) for k, v in CONTRASTS.items()}
            or reg['bootstrap']['reps'] != 500 or reg['bootstrap']['seed'] != 20261003
            or reg['output'] != EVIDENCE + '/evaluation-v1'):
        raise ValueError('Wrong frozen analysis contract')
    out = contained(root, reg['output'])
    if out.exists():
        raise FileExistsError('One-shot output already exists; no overwrite/repeated evaluation')
    verify_bindings(root, reg['source_bindings'])
    m = read(contained(root, reg['protocol']))
    validate_protocol(m)
    ordered, groups = cohort_metadata(root, m)
    execution, output_hashes = require_completed(root, reg, m)
    if reg['reference_counts'] != m['reference_counts'] or reg['expected_ids'] != m['expected_ids']:
        raise ValueError('Registered count metadata differs')
    out.mkdir(parents=True, exist_ok=False)  # exclusive one-shot claim before outcome access
    try:
        write_new(out / 'analysis-plan.json', dict(registration_sha256=registration_sha256,
            output_hashes=output_hashes, execution=execution, contrasts=CONTRASTS, limits=LIMITS))
        ids = ordered['val']
        y = load_reference(root, m, ids)
        p = {a: predictions(contained(root, m['arms'][a]['outputs']['predictions']), ids) for a in ARMS}
        scores = scores_for(p['B'], p['RN_clip'], p['RN_ones'])
        rows = target_rows(y, scores)
        totals = aggregate(rows)
        ci = paired_bootstrap(y, scores, groups.loc[ids, 'component_id'].to_numpy())
        diagnostics = []
        selected = groups.loc[ids]
        for field in m['groups']['subgroup_columns']:
            vals = selected[field].replace('', 'unknown').to_numpy(dtype=str)
            for value in sorted(set(vals)):
                take = vals == value
                group_rows = target_rows(y[take], {k: v[take] for k, v in scores.items()})
                diagnostics.append(dict(field=field, group=value, studies=int(take.sum()),
                                        targets=group_rows, aggregate=aggregate(group_rows), ci=None))
        verify_bindings(root, {**reg['source_bindings'], **output_hashes})
        if sha(registration_path) != registration_sha256:
            raise ValueError('Registration changed during evaluation')
        report = dict(schema=SCHEMA, evaluated_at=datetime.now(timezone.utc).isoformat(),
            registration_sha256=registration_sha256, protocol_sha256=reg['protocol_sha256'],
            source_bindings=reg['source_bindings'], output_hashes=output_hashes,
            evaluation_environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
            studies=len(ids), targets=rows, aggregate=totals, conditional_intervals=ci,
            groups=diagnostics, execution=execution, limits=LIMITS,
            decision='Root review required; no automatic adoption, ratio/epoch search, submission or follow-up training.')
        write_new(out / 'report.json', report)
        pd.json_normalize(rows).to_csv(out / 'targets.csv', index=False, mode='x')
        write_new(out / 'SUCCESS.json', dict(report_sha256=sha(out / 'report.json'), registration_sha256=registration_sha256))
        return dict(output=reg['output'], complete12=totals['complete12'], aggregate=totals,
                    attempted=ci['attempted'], valid=ci['valid'], invalid=ci['invalid'])
    except Exception as error:
        write_new(out / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
            rule='Preserve failed one-shot output; root must explicitly review any new registration/version.'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--protocol', required=True)
    prep.add_argument('--protocol-sha256', required=True)
    evaluate = sub.add_parser('run')
    evaluate.add_argument('--registration', required=True)
    evaluate.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.protocol, args.protocol_sha256)
        print(json.dumps({'registration': EVIDENCE + '/registration.json',
                          'sha256': sha(ROOT / EVIDENCE / 'registration.json'), 'scope': result['scope']}, indent=2))
    else:
        print(json.dumps(run(args.registration, args.registration_sha256), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
