"""Passive CPU evaluation of the three completed, frozen EXP080 F2 fits.

This module has its own EXP080 lineage/completion path. Only pure validation and
metric adapters are shared with earlier reports. It never loads Gold or models.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_ab_report as primary
from exp079_complement_report import fixed_blend, validate_run
from oof_error_audit import weighted_auc

ROOT, UID, LAB = primary.ROOT, primary.UID, primary.LAB
SCHEMA = 'exp080-f2-report-v1'
CONTRASTS = {'input_B_minus_A': ('A', 'B'),
             'complement_blend_minus_B': ('B', 'blend'),
             'descriptive_blend_minus_A': ('A', 'blend')}
ENV_KEYS = ('torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'pydicom', 'cv2')
CODE_NAMES = ('raptor_train.py', 'raptor_common.py', 'raptor_data.py', 'raptor_runtime.py')
TAGS = ('EXP080_INPUT_A_F2', 'EXP080_INPUT_B_F2', 'EXP080_R34_B_F2')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def uri(path, root=ROOT):
    return 'workspace://' + Path(path).absolute().relative_to(Path(root).absolute()).as_posix()


def unuri(value, root=ROOT):
    prefix = 'workspace://'
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError('Expected workspace URI')
    relative = Path(value[len(prefix):])
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('Expected contained workspace URI')
    return Path(root) / relative


def uid_digest(ids, final_newline=False):
    text = '\n'.join(sorted(ids)) + ('\n' if final_newline else '')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_protocol(manifest):
    if (manifest['schema'] != 'exp080-f2-confirm-v1' or manifest['folds'] != [2]
            or manifest['seed'] != 42 or manifest['targets'] != LAB):
        raise ValueError('Expected frozen EXP080 F2 / 12-target protocol')
    if (manifest['eval_labels'] != manifest['labels'] or manifest['eval_label_hash'] != manifest['label_hash']
            or set(manifest['training_code_sha256']) != set(CODE_NAMES)):
        raise ValueError('Changed reference labels or training source contract')
    expected = manifest['expected_ids']
    if set(expected) != {'2'} or expected['2']['train'] != 3475 or expected['2']['val'] != 868:
        raise ValueError('Expected exact3475 training /868 validation studies')
    signatures = [(r['tag'], r['arm'], r['fold'], r['policy'], r['fixed_epoch']) for r in manifest['runs']]
    if signatures != [(TAGS[0], 'A', 2, 'global-valid', 8),
                      (TAGS[1], 'B', 2, 'within-slot-valid', 8),
                      (TAGS[2], 'RN_B', 2, 'within-slot-valid', 20)]:
        raise ValueError('Require exactly registered A/B/RN_B fits in order')
    if set(manifest['expected_args_by_arm']) != {'A', 'B', 'RN_B'}:
        raise ValueError('Missing or extra arm recipe')
    common = dict(res=224, k=12, k_eval=24, select='oof', bs=8, accum=1, amp='fp16',
                  eval_amp='fp16', runtime_policy='strict', amp_init_scale=1024,
                  seed=42, norm='imagenet', head_lr=.001, wd=.02, limit=0, smoke=False,
                  pretrained='timm', oof_subsample=0, folds=5)
    for run in manifest['runs']:
        rn = run['arm'] == 'RN_B'
        recipe = {**common, 'arch': 'resnet34.a1_in1k' if rn else 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k',
                  'epochs': run['fixed_epoch'], 'fixed_epoch': run['fixed_epoch'],
                  'window_policy': run['policy'], 'bb_lr': .0005 if rn else 3e-5,
                  'grad_ckpt': not rn, 'workers': 4 if rn else 6}
        if rn:
            recipe['device'] = 'cuda'
        if manifest['expected_args_by_arm'][run['arm']] != recipe:
            raise ValueError('Changed frozen architecture recipe: ' + run['arm'])
        initial = manifest['pretrained']['ResNet34' if rn else 'CoAtNet']['initial_full_model_sha256']
        if run['initial_state_sha256'] != initial or run['command_changes'] != ['tag', 'fold']:
            raise ValueError('Changed initialization / copied-command contract')
    ev = manifest['evaluation']
    if (ev['requires_all_three_completed_fits'] is not True or set(ev['contrasts']) != set(CONTRASTS)
            or ev['fixed_global_rank_weights'] != {'baseline_B': .9, 'candidate_RN_B': .1}
            or ev['ties'] != 'average rank/n within fold and target; no final re-rank, subgroup re-rank or bootstrap re-rank'
            or ev['ratio_epoch_target_search'] is not False or ev['major_drop_review'] != -.01
            or ev['groups'] != ['scanner_group', 'field', 'manufacturer', 'cached_mask']
            or ev['conditional_ci']['reps'] != 500 or ev['conditional_ci']['seed'] != 20261003):
        raise ValueError('Changed fixed evaluation contract')
    stop = manifest['stop_review']
    if (stop['optimizer_skip_fraction'] != .01 or stop['consecutive_skips_review'] != 10
            or stop['epoch_metric_early_stopping'] is not False):
        raise ValueError('Changed numeric / epoch-selection guard')


def validate_launch(manifest, manifest_hash, launch, queue_hash, decision_hash, metadata_hash):
    expected = {'manifest_sha256': manifest_hash, 'queue_sha256': queue_hash,
                'exp079_decision_sha256': decision_hash, 'metadata_preflight_sha256': metadata_hash,
                'training_source_sha256': manifest['training_code_sha256'],
                'expected_ids': manifest['expected_ids']['2'], 'exit_code': 0,
                'active_training_or_queue_before_launch': []}
    if any(launch.get(k) != v for k, v in expected.items()):
        raise ValueError('Launch receipt does not bind frozen decision / metadata / sources')
    if (not isinstance(launch.get('queue_pid'), int) or launch['queue_pid'] <= 0
            or launch['env_overrides'].get('HF_HUB_OFFLINE') != '1'
            or len(launch['command']) != 3
            or launch['command'][1].replace('\\', '/') != 'tools/gpu_queue.py'):
        raise ValueError('Invalid launch receipt')


def validate_completion(manifest, queue, root=ROOT):
    """Check ALL artifacts exist before reading any new summary or predictions."""
    if len(queue) != 3 or [q['tag'] for q in queue] != list(TAGS):
        raise ValueError('Queue must contain exactly frozen three fits')
    files = []
    for i, (run, job) in enumerate(zip(manifest['runs'], queue)):
        if job.get('continue_on_failure', False) or job.get('after') != (TAGS[i-1] if i else None):
            raise ValueError('Queue changed dependency / failure behavior')
        log = Path(root) / 'artifacts/logs' / run['tag']
        exp = Path(root) / 'artifacts/experiments' / run['tag']
        paths = [log/'SUCCESS', log/'job.json', exp/'summary.json', exp/'oof_best.csv']
        if (log/'FAILED').exists() or any(not p.is_file() for p in paths):
            raise ValueError('All three completed SUCCESS / summary / OOF fits required before scoring: '+run['tag'])
        files.append(paths)
    # All3 completion was established above; only job receipts are opened here.
    for job, (success, receipt, _, _) in zip(queue, files):
        if re.fullmatch(r'exit 0 at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s*', success.read_text()) is None:
            raise ValueError('Invalid successful exit marker')
        recorded = read(receipt)
        if recorded.get('tag') != job['tag'] or recorded.get('command') != job['command']:
            raise ValueError('Job receipt command differs from frozen queue')
    return files


def protocol_for_run(manifest, run):
    protocol = deepcopy(manifest)
    protocol.update(epochs=run['fixed_epoch'], initial_state_sha256=run['initial_state_sha256'],
                    expected_args=deepcopy(manifest['expected_args_by_arm'][run['arm']]))
    return protocol


def predictions(path, ids):
    frame = pd.read_csv(path, dtype={UID: str})
    if (list(frame.columns) != [UID, *LAB] or frame[UID].isna().any()
            or not frame[UID].is_unique or set(frame[UID]) != set(ids)):
        raise ValueError('Exact F2 UID /12-target order mismatch')
    values = frame.set_index(UID).loc[ids, LAB].to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid prediction probabilities')
    return values


def rows_from_scores(y, scores):
    """Consume already frozen full-fold scores, including for every subgroup."""
    binary = np.rint(np.asarray(y, dtype=np.float32)).astype(int)
    rows = []
    for j, target in enumerate(LAB):
        pos = int(binary[:, j].sum()); neg = len(binary)-pos
        aucs = {name: weighted_auc(binary[:, j], value[:, j], np.ones(len(binary)))
                if pos and neg else None for name, value in scores.items()}
        delta = {name: aucs[b]-aucs[a] if pos and neg else None for name, (a, b) in CONTRASTS.items()}
        rows.append({'target': target, 'P': pos, 'N': neg, 'auc': aucs, 'delta': delta,
                     'macro_contribution': {k: None if v is None else v/12 for k, v in delta.items()}})
    return rows


def paired_rows(y, a, b, rn):
    if np.asarray(y).shape != a.shape or a.shape != b.shape or a.shape != rn.shape:
        raise ValueError('Labels and all three predictions must have the same shape')
    if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError('Invalid A probabilities')
    rank_b, blend = fixed_blend(b, rn)
    scores = {'A': a, 'B': rank_b, 'RN_B': rn, 'blend': blend}
    return rows_from_scores(y, scores), scores


def aggregate(rows):
    def mean_all(values):
        return None if any(v is None for v in values) else float(np.mean(values))
    return {name: {'macro12': mean_all([r['delta'][name] for r in rows]),
                   'PFOA': next(r['delta'][name] for r in rows if r['target'] == 'PF OA'),
                   'other11': mean_all([r['delta'][name] for r in rows if r['target'] != 'PF OA'])}
            for name in CONTRASTS}


def conditional_intervals(y, scores, groups, reps=500, seed=20261003):
    """One paired report-group draw supplies all3 contrasts; never re-rank."""
    if not isinstance(reps, int) or reps < 1 or len(groups) != len(y) or not len(y):
        raise ValueError('Invalid group bootstrap inputs')
    binary = np.rint(np.asarray(y, dtype=np.float32)).astype(int)
    unique, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = {k: np.empty((reps, len(LAB))) for k in CONTRASTS}
    for i in range(reps):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[codes]
        aucs = {name: np.array([weighted_auc(binary[:, j], scores[name][:, j], weights)
                               for j in range(len(LAB))]) for name in ('A', 'B', 'blend')}
        for name, (a, b) in CONTRASTS.items():
            draws[name][i] = aucs[b]-aucs[a]
    def interval(values):
        good = values[np.isfinite(values)]
        return {'requested_reps': reps, 'valid_reps': len(good),
                'low': float(np.quantile(good, .025)) if len(good) else None,
                'high': float(np.quantile(good, .975)) if len(good) else None}
    pfoa = LAB.index('PF OA')
    result = {'unit': 'report_group_within_fold', 'groups': len(unique), 'seed': seed,
              'scope': 'Conditional on fixed full-fold scores/ranks and exposed weak labels; no reranking, seed/label/prior-selection/private uncertainty or multiplicity correction.',
              'contrasts': {}}
    for name, d in draws.items():
        result['contrasts'][name] = {'macro12': interval(d.mean(axis=1)),
            'PFOA': interval(d[:, pfoa]), 'other11': interval(np.delete(d, pfoa, axis=1).mean(axis=1)),
            'targets': {target: interval(d[:, j]) for j, target in enumerate(LAB)}}
    return result


def validate_cohort(manifest, metadata, split, labels, corpus_ids):
    if (not split.index.is_unique or split.index.isna().any() or split.report_group_sha256.isna().any()
            or split.groupby('report_group_sha256').fold.nunique().max() > 1
            or len(set(corpus_ids)) != len(corpus_ids)):
        raise ValueError('Duplicate IDs / missing or cross-fold report groups')
    eligible = split.fold.ge(0)
    for column in ('is_gold', 'quarantined_gold_report_duplicate'):
        flags = split[column].astype(str).str.lower()
        if not flags.isin(['true', 'false']).all() or (eligible & flags.eq('true')).any():
            raise ValueError('Invalid held-out/quarantine flags in frozen split')
    pool = set(split.index[eligible]) & set(corpus_ids) & set(labels)
    pool -= set(manifest['excluded_uids'])
    val_ids = sorted(u for u in pool if split.at[u, 'fold'] == 2)
    train_ids = sorted(pool-set(val_ids))
    expected = manifest['expected_ids']['2']
    actual = {'train': len(train_ids), 'val': len(val_ids), 'train_ids_sha256': uid_digest(train_ids),
              'eval_ids_sha256': uid_digest(val_ids)}
    if actual != expected:
        raise ValueError('Exact train3475 / val868 population or UID digest differs')
    c, r = metadata['corpus_and_split'], metadata['report_groups']
    if (c['f2_train'] != len(train_ids) or c['f2_validation'] != len(val_ids)
            or c['f2_train_uid_hash_sorted_newline'] != uid_digest(train_ids, True)
            or c['f2_validation_uid_hash_sorted_newline'] != uid_digest(val_ids, True)
            or 'with final newline' not in c['uid_hash_rule']):
        raise ValueError('Metadata cohort binding or final-newline serialization mismatch')
    tg, vg = set(split.loc[train_ids, 'report_group_sha256']), set(split.loc[val_ids, 'report_group_sha256'])
    if (tg & vg or len(vg) != 840 or len(tg) != r['f2_train_groups']
            or len(vg) != r['f2_validation_groups'] or len(pool) != c['eligible_after_common_exclusion']):
        raise ValueError('Cohort/report-group count or separation differs')
    if (metadata['status'] != 'METADATA_CHECKS_COMPLETE_NOT_PATIENT_OR_IMAGE_INDEPENDENCE_CERTIFICATION'
            or not metadata['checks_passed'] or any(v is not True for v in metadata['checks_passed'].values())
            or metadata['conclusion']['detected_cross_f2_series_or_report_group_reuse'] is not False
            or any(c[k] != 0 for k in ('f2_train_validation_uid_overlap', 'f2_gold_uid_overlap',
                                      'f2_quarantined_uid_overlap', 'f2_common_exclusion_overlap'))
            or metadata['series_metadata']['f2_train_validation_shared_series_uid_count'] != 0
            or metadata['selected_series']['f2_shared_selected_series_sets'] != 0
            or metadata['selected_series']['f2_any_shared_selected_series_uids'] != []
            or any(r[k] != [] for k in ('groups_spanning_folds', 'f2_train_validation_shared_groups',
                                       'f2_train_gold_shared_groups', 'f2_validation_gold_shared_groups'))):
        raise ValueError('Metadata preflight does not pass frozen separation checks')
    return train_ids, val_ids


def load_inputs(manifest_path, manifest_sha256, launch_path, launch_sha256, decision_path, metadata_path, root=ROOT):
    root = Path(root); hashes = {}
    def bind(path, expected=None):
        path = Path(path)
        key = uri(path, root)
        # Gold is represented solely by the bound preflight receipt. Never open it.
        if 'gold' in [part.lower() for part in path.parts]:
            raise ValueError('Direct Gold access is not allowed')
        digest = primary.sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError('Frozen source changed: '+key)
        hashes[key] = digest
        return digest
    bind(manifest_path, manifest_sha256); manifest = read(manifest_path)
    validate_protocol(manifest)
    bind(launch_path, launch_sha256); launch = read(launch_path)
    queue_path = unuri(manifest['queue']['path'], root)
    queue_hash = bind(queue_path, manifest['queue']['sha256']); queue = read(queue_path)
    decision_hash = bind(decision_path, launch['exp079_decision_sha256'])
    metadata_hash = bind(metadata_path, launch['metadata_preflight_sha256'])
    validate_launch(manifest, manifest_sha256, launch, queue_hash, decision_hash, metadata_hash)
    if Path(launch['command'][2]).absolute() != queue_path.absolute():
        raise ValueError('Launch queue path differs')
    # No EXP080 summary/prediction, including its hash, is read until ALL3 pass.
    completed_files = validate_completion(manifest, queue, root)
    for ref in manifest['parents'].values():
        bind(unuri(ref['path'], root), ref['sha256'])
    for path, digest in manifest['frozen_parent_result_sources'].items():
        bind(unuri(path, root), digest)
    for run, job in zip(manifest['runs'], queue):
        ref = run['copied_from']; path = unuri(ref['path'], root); bind(path, ref['sha256'])
        source = [r for r in read(path) if r['tag'] == ref['tag']]
        if len(source) != 1:
            raise ValueError('Missing copied parent command')
        command = list(source[0]['command'])
        for flag, value in (('--tag', run['tag']), ('--fold', '2')):
            if command.count(flag) != 1:
                raise ValueError('Ambiguous source command')
            command[command.index(flag)+1] = value
        if job['command'] != command:
            raise ValueError('Queue changed more than tag/fold from parent')
    decision = read(decision_path); metadata = read(metadata_path)
    if (decision['manifest_sha256'] != manifest['parents']['exp079_manifest']['sha256']
            or decision['report_sha256'] != manifest['parents']['exp079_report']['sha256']
            or decision['report'] != manifest['parents']['exp079_report']['path']
            or decision['status'] != 'PROMISING_DEVELOPMENT_COMPLEMENT_RETAINED_NOT_DEPLOYED'):
        raise ValueError('Decision lineage differs from frozen parents')
    for name, ref in metadata['sources'].items():
        if name != 'gold':
            bind(unuri(ref['path'], root), ref['sha256'])
    for path, digest in manifest['source_file_sha256'].items():
        bind(root/path, digest)
    for path, digest in manifest['metadata_source_hashes'].items():
        bind(unuri(path, root), digest)
    bind(root/manifest['corpus']/'manifest.json', manifest['corpus_manifest_sha256'])
    for name, digest in manifest['training_code_sha256'].items():
        bind(root/'training/src'/name, digest)
    labels = primary.validate_sources(manifest, root=root)
    split = pd.read_csv(root/manifest['fold_file'], dtype={UID: str}).set_index(UID)
    corpus = root/manifest['corpus']
    corpus_ids = np.load(corpus/'ids.npy', allow_pickle=False).astype(str)
    masks = np.load(corpus/'masks.npy', mmap_mode='r', allow_pickle=False)
    _, ids = validate_cohort(manifest, metadata, split, labels, corpus_ids)
    if masks.ndim != 2 or len(masks) != len(corpus_ids):
        raise ValueError('Corpus mask shape differs')
    meta = pd.read_csv(root/'storage/labels/derived_meta/study_meta.csv', dtype={UID: str}).set_index(UID)
    if not meta.index.is_unique or meta.index.isna().any() or not set(ids) <= set(meta.index):
        raise ValueError('Missing/duplicate study metadata')
    groups = meta.loc[ids, ['scanner_group', 'field', 'manufacturer']].fillna('unknown')
    mask_groups = pd.Series(np.where((masks == 0).any(axis=1), 'some_zero_mask', 'all_nonzero'), index=corpus_ids)
    groups['cached_mask'] = mask_groups.loc[ids]
    values, execution, receipts, environments = {}, [], [], []
    for run, paths in zip(manifest['runs'], completed_files):
        for path in paths:
            bind(path)
        summary = read(paths[2])
        protocol = protocol_for_run(manifest, run)
        fraction = validate_run(summary, protocol, run, manifest['stop_review']['optimizer_skip_fraction'])
        receipts.extend(summary['provenance'][key] for key in ('runtime', 'eval_runtime'))
        environment = {k: summary['environment'][k] for k in ENV_KEYS}
        if any(v is None or v == '' for v in environment.values()):
            raise ValueError('Incomplete local environment receipt')
        environments.append(environment)
        if not np.isfinite(summary['total_seconds']) or summary['total_seconds'] <= 0:
            raise ValueError('Invalid elapsed training time')
        execution.append({'tag': run['tag'], 'arm': run['arm'], 'fixed_epoch': run['fixed_epoch'],
            'total_seconds': summary['total_seconds'], 'optimizer_updates': summary['optimizer_updates'],
            'skipped_updates': summary['skipped_updates'], 'skip_fraction': fraction,
            'consecutive_skip_evidence': 'Completed frozen training source raises at10 consecutive skips; maximum observed count is not separately recorded.'})
        values[run['arm']] = predictions(paths[3], ids)
    if any(r != receipts[0] for r in receipts[1:]) or any(e != environments[0] for e in environments[1:]):
        raise ValueError('Uncontrolled runtime/environment difference across three fits')
    y = np.stack([labels[u] for u in ids]).astype(np.float32)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Invalid reference labels')
    for path, digest in hashes.items():
        if primary.sha256_file(unuri(path, root)) != digest:
            raise ValueError('Input changed during validation')
    data = {'ids': ids, 'y': y, 'values': values, 'groups': groups,
            'report_groups': split.loc[ids, 'report_group_sha256'].to_numpy()}
    return manifest, data, hashes, execution, environments[0]


def build(manifest_path, manifest_sha256, launch_path, launch_sha256, decision_path, metadata_path, out, root=ROOT):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest, data, hashes, execution, environment = load_inputs(
        manifest_path, manifest_sha256, launch_path, launch_sha256, decision_path, metadata_path, root)
    out.mkdir(parents=True, exist_ok=False)
    tool_hashes = {name: primary.sha256_file(Path(__file__).parent/name) for name in
                  ('exp080_f2_report.py', 'exp079_complement_report.py', 'input_ab_report.py',
                   'input_ab_diagnostics.py', 'oof_error_audit.py')}
    plan = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': manifest_sha256, 'launch_receipt_sha256': launch_sha256,
            'source_hashes': hashes, 'evaluator_sha256': tool_hashes,
            'evaluation': manifest['evaluation'], 'scope': 'F2 only; all3 completed before scoring; no Gold reads or selection.'}
    write_new(out/'analysis_plan.json', plan)
    y, v = data['y'], data['values']
    rows, scores = paired_rows(y, v['A'], v['B'], v['RN_B'])
    summaries = aggregate(rows)
    if any(s['macro12'] is None for s in summaries.values()):
        raise ValueError('Full F2 does not provide all12 target AUCs')
    intervals = conditional_intervals(y, scores, data['report_groups'])
    group_rows = []
    for field in data['groups'].columns:
        vals = data['groups'][field].astype(str).to_numpy()
        for group in sorted(set(vals)):
            take = vals == group
            targets = rows_from_scores(y[take], {k: p[take] for k, p in scores.items()})
            group_rows.append({'field': field, 'group': group, 'studies': int(take.sum()),
                               'small_group': int(take.sum()) < 40, 'targets': targets,
                               'contrasts': aggregate(targets)})
    fracture = next(r for r in rows if r['target'] == 'Fracture')
    result = {'schema': SCHEMA, 'observed_at': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': manifest_sha256, 'launch_receipt_sha256': launch_sha256,
        'analysis_plan_sha256': primary.sha256_file(out/'analysis_plan.json'),
        'source_hashes': hashes, 'evaluator_sha256': tool_hashes, 'fold': 2, 'studies': len(y),
        'targets': rows, 'contrasts': summaries, 'groups': group_rows, 'conditional_intervals': intervals,
        'fracture': {**fracture, 'conditional_ci': {k: intervals['contrasts'][k]['targets']['Fracture'] for k in CONTRASTS},
                     'prior_risk': 'Prior F1 input-B and same-blend losses versusA remain in the frozen parent decision; F2 does not replace them.'},
        'review': {'directions': {k: s['macro12'] > 0 for k, s in summaries.items()},
                   'complement_PFOA_positive': summaries['complement_blend_minus_B']['PFOA'] > 0,
                   'major_target_drops': [{'contrast': k, 'target': r['target'], 'delta': r['delta'][k]}
                       for r in rows for k in CONTRASTS if r['delta'][k] < manifest['evaluation']['major_drop_review']],
                   'rule': 'No new .002 cutoff; no automatic acceptance/expansion. Review all3 contrasts, all targets/P/N, groups, Fracture, conditional CI and cost. Prior gates stay unchanged.'},
        'execution': execution, 'local_environment': environment,
        'pretrained_provenance': manifest['pretrained'],
        'initialization_check': 'All3 saved full-state digests match registered initialization; upstream weight files are not loaded by this CPU evaluator.',
        'parents': manifest['parents'], 'decision': 'Review only; no routing, ratio/epoch tuning, training or submission follows.',
        'caveats': ['F2 is exposed report-label development evidence, not an untouched clinical/private holdout.',
                    'Report/series metadata do not establish patient or image independence; Gold preflight evidence is receipt-bound and Gold files are never read.',
                    'Paired intervals condition on fixed predictions/full-fold ranks and exclude seed, labels, prior selection and private-distribution uncertainty; no multiplicity correction.',
                    'Groups are descriptive; one-class targets stay null and cannot produce a partial-target macro12.',
                    'The RN20epoch versus CoAt8epoch comparison is a practical recipe comparison, not architecture-only causality.',
                    'Local strict FP16/backend receipt agreement does not establish fresh Kaggle T4 full-stack parity or hidden-test runtime.',
                    'Final training summaries may contain Gold diagnostics; this evaluator ignores them and never evaluates/selects Gold.']}
    # Source mutation during bootstrap or group computation must also fail closed.
    for path, digest in hashes.items():
        if primary.sha256_file(unuri(path, root)) != digest:
            raise ValueError('Input changed during scoring')
    validate_completion(manifest, read(unuri(manifest['queue']['path'], root)), root)
    write_new(out/'report.json', result)
    pd.json_normalize(rows).to_csv(out/'targets.csv', index=False)
    return {'fold': 2, 'studies': len(y), 'contrasts': summaries, 'review': result['review']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'manifest-sha256', 'launch-receipt', 'launch-sha256', 'decision', 'metadata-preflight', 'out'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.manifest_sha256, args.launch_receipt, args.launch_sha256,
                           args.decision, args.metadata_preflight, args.out), ensure_ascii=False))
