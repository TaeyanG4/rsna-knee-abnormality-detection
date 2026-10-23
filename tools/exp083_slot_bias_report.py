"""Register then evaluate EXP083 once; import is passive and never reads scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = 'docs/handoffs/evidence/2026-10/2026-10-03-exp083-report-tool'
GROUP_DIR = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1'
META = 'storage/labels/derived_meta/study_meta.csv'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ARMS = ('control', 'treatment', 'frozen_b8')
LIMITS = [
    'Exposed weak F2 development labels; one training seed and inherited EXP080 B8/researcher/Gold ancestry.',
    'No Gold file access does not erase prior researcher or backbone/model-selection exposure.',
    'Component bootstrap is conditional on these trained models, labels and selected F2 studies; not seed/label/selection/private uncertainty.',
    'Report plus one known duplicate components are not proven patient IDs; unknown patient/near-duplicate links remain.',
    'Scanner/mask subgroups are exploratory observational diagnostics, not held-out scanner tests or clinical false-positive rates.',
    'Mask means nonzero cached pixels, not proven clinical or acquisition availability.',
    'Frozen B8 here is the same-cache secondary reference only; historical saved B OOF equivalence has no passed replay gate.',
    'No private LB, clinical validation, anchor0.943, D01, rank/blend tuning or automatic adoption claim.',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def contained(root, relative):
    p = Path(relative)
    if p.is_absolute() or p.drive or '..' in p.parts:
        raise ValueError('Require relative workspace path')
    result = root / p
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes workspace')
    return result


def bind(root, relative, expected, observed):
    value = sha(contained(root, relative))
    if value != expected:
        raise ValueError('Hash mismatch: ' + relative)
    observed[relative] = value


def prepare(protocol, protocol_sha256, root=ROOT, job_dir='artifacts/logs/EXP083_SLOT_BIAS_V2_FULL'):
    """Hash existing contracts and metadata only; no prediction/label-value reads."""
    root = Path(root)
    contained(root, job_dir)
    if sha(contained(root, protocol)) != protocol_sha256:
        raise ValueError('EXP083 registration differs from the frozen CPU preparation')
    m = read(contained(root, protocol))
    if (m.get('schema') != 'exp083-slot-bias-f2-v2-native-fp32' or m['cache']['dtype'] != 'float32'
            or m['targets'] != TARGETS or m['ordered_ids']['train'] != 3475 or m['ordered_ids']['eval'] != 868):
        raise ValueError('Wrong target or study contract')
    group = read(root / GROUP_DIR / 'contract.json')
    bindings = {protocol: protocol_sha256, **m['source_bindings']}
    bindings.update({'training/src/' + k: v for k, v in m['training_code_sha256'].items()})
    for name, item in group['outputs'].items():
        bindings[GROUP_DIR + '/' + name] = item['sha256']
    bindings[META] = sha(root / META)
    for p in ('tools/exp083_slot_bias_report.py', 'tests/test_exp083_slot_bias_report.py'):
        bindings[p] = sha(root / p)
    verified = {}
    for p, expected in bindings.items():
        bind(root, p, expected, verified)
    out = root / EVIDENCE
    out.mkdir(parents=True, exist_ok=True)
    registration = dict(schema='exp083-report-registration-v1', observed_at=datetime.now(timezone.utc).isoformat(),
        protocol=protocol, protocol_sha256=protocol_sha256, source_bindings=verified,
        run_dir=str(Path(protocol).parent / 'run-v1').replace('\\', '/'),
        pilot_receipt=str(Path(protocol).parent / 'pilot-v1/receipt.json').replace('\\', '/'),
        root_full_decision=str(Path(protocol).parent / 'root-full-decision.json').replace('\\', '/'),
        runner='training/runners/exp083_slot_bias_v2.py', job_dir=job_dir, output=EVIDENCE + '/evaluation-v1',
        targets=TARGETS, arms=list(ARMS), primary='macro12(treatment)-macro12(retrained_control)',
        secondary=['macro12(control)-macro12(common_cache_frozen_B8)', 'macro12(treatment)-macro12(common_cache_frozen_B8)'],
        bootstrap=dict(reps=500, seed=20261003, unit='fixed connected component',
            invalid='Any target without weighted positives or negatives invalidates the complete macro draw',
            same_draw_all_arms_and_targets=True, interval='percentile2.5/97.5 of valid complete12-target draws'),
        reference_binary_rule='same canonical soft CSV; float32 then numpy.rint',
        csv_npy_rule='Exact ordered UID/columns and bit-equal float32 values',
        skip_rule='Any optimizer skip blocks scoring pending a separately recorded root review; no automatic override',
        historical_B_replay_gate='NOT_PASSED_NO_HISTORICAL_COMPARISON', limits=LIMITS,
        scope='Prepared before new scores; prepare reads hashes/contracts only. Execution is separate.')
    write_new(out / 'registration.json', registration)
    return registration


def validate_counts(receipt, histories, scales, traces):
    """Fail before predictions/labels on incomplete or unequal realized budgets."""
    if receipt.get('requires_skip_review') is not False or receipt.get('equal_realized_update_budget') is not True:
        raise ValueError('Any skip requires root review before scoring')
    if set(receipt.get('arms', {})) != {'control', 'treatment'}:
        raise ValueError('Require exactly two retrained heads')
    for arm in ('control', 'treatment'):
        r = receipt['arms'][arm]
        if (r.get('attempts'), r.get('updates'), r.get('skips'), r.get('last_batch_size')) != (1100, 1100, 0, 19):
            raise ValueError('Wrong fixed attempt/update/skip budget')
        if r.get('any_skips_require_review') is not False or r.get('separate_optimizer_scheduler_scaler') is not True:
            raise ValueError('Incomplete independent arm execution receipt')
        history, scale, trace = histories[arm], scales[arm], traces[arm]
        if len(history) != 20 or len(scale) != 1100 or len(trace) != 1100:
            raise ValueError('Incomplete fixed training history')
        for ep, h in enumerate(history):
            if (h['epoch'], h['attempts'], h['updates'], h['skips']) != (ep, (ep + 1) * 55, (ep + 1) * 55, 0):
                raise ValueError('Epoch/update history changed')
            if h['F2_metric'] is not None or h['gold_metric'] is not None:
                raise ValueError('Unexpected intermediate scoring')
        previous = 1024.
        for i, s in enumerate(scale):
            if s['attempt'] != i + 1 or s['before'] != previous or s['after'] < s['before'] or s['after'] not in (s['before'], s['before'] * 2):
                raise ValueError('Invalid no-skip scale trajectory')
            previous = s['after']
        for i, t in enumerate(trace):
            if (t['epoch'], t['batch']) != divmod(i, 55):
                raise ValueError('Wrong paired batch trace coordinates')
    if traces['control'] != traces['treatment'] or receipt.get('paired_rng_trace_exact') is not True:
        raise ValueError('Paired batch/dropout trace differs')


def validate_completed(root, registration, m):
    import math
    root = Path(root); run = contained(root, registration['run_dir'])
    job = contained(root, registration['job_dir'])
    if not (job / 'SUCCESS').is_file() or (job / 'FAILED').exists() or (run / 'failure.json').exists():
        raise ValueError('Require completed successful full job with no failure marker')
    job_info = read(job / 'job.json'); command = job_info['command']
    if len(command) < 2 or (root / command[1]).resolve() != (root / registration['runner']).resolve():
        raise ValueError('Completed job used a different runner entrypoint')
    expected_args = {'--phase': 'full', '--protocol': registration['protocol'], '--out': registration['run_dir'],
                     '--pilot-receipt': registration['pilot_receipt'], '--root-decision': registration['root_full_decision']}
    for flag, value in expected_args.items():
        if command.count(flag) != 1:
            raise ValueError('Job command is not this full phase')
        actual = command[command.index(flag) + 1]
        if flag == '--phase':
            equal = actual == value
        else:
            equal = (root / actual).resolve() == (root / value).resolve()
        if not equal:
            raise ValueError('Job command differs: ' + flag)
    r = read(run / 'receipt.json')
    flags = ['no_F2_score_evaluated', 'initial_logits_exact', 'initial_train_logits_exact',
             'native_cache_roundtrip_exact', 'native_cache_logits_exact', 'disjoint_initial_head_storage',
             'extraction_split_loaders', 'paired_rng_trace_exact']
    if (r.get('schema'), r.get('status'), r.get('phase'), r.get('protocol_sha256'), r.get('gold_access')) != (
            'exp083-phase-v2-native-fp32', 'PASS', 'full', registration['protocol_sha256'], 'disabled') or any(r.get(k) is not True for k in flags):
        raise ValueError('Incomplete or different full phase receipt')
    pilot = read(contained(root, registration['pilot_receipt']))
    decision = read(contained(root, registration['root_full_decision']))
    required = dict(decision='RUN_ONE_FULL_PHASE', protocol_sha256=registration['protocol_sha256'],
                    pilot_receipt_sha256=sha(contained(root, registration['pilot_receipt'])),
                    prior_failure_sha256=m['phases']['prior_failure_sha256'], spent_previous_seconds=m['phases']['spent_previous_seconds'],
                    max_total_seconds=2700, automatic_retries=0)
    if any(decision.get(k) != v for k, v in required.items()) or pilot.get('status') != 'PASS' or pilot.get('phase') != 'pilot' or pilot.get('protocol_sha256') != registration['protocol_sha256']:
        raise ValueError('Missing matching pilot/root full authority')
    spent = m['phases']['spent_previous_seconds']
    pilot_flags = ['initial_logits_exact', 'initial_train_logits_exact', 'native_cache_roundtrip_exact',
                   'native_cache_logits_exact', 'extraction_split_loaders', 'disjoint_initial_head_storage']
    if (not math.isfinite(spent) or spent <= 0 or r['spent_previous_seconds'] != spent or pilot['spent_previous_seconds'] != spent
            or r['prior_failure_sha256'] != m['phases']['prior_failure_sha256'] or pilot['prior_failure_sha256'] != m['phases']['prior_failure_sha256']
            or pilot.get('observed_native_feature_dtypes') != ['torch.float32'] or pilot.get('feature_dtype') != 'float32'
            or any(pilot.get(k) is not True for k in pilot_flags) or pilot.get('pilot_ids') != m['phases']['pilot_ids']
            or pilot.get('encoder_before_sha256') != m['encoder_state_sha256'] or pilot.get('encoder_after_sha256') != m['encoder_state_sha256']
            or not math.isfinite(pilot['elapsed_seconds']) or not 0 < pilot['elapsed_seconds'] <= 300):
        raise ValueError('Wrong preserved-failure or successful FP32 pilot receipt')
    remaining = 2700 - spent - pilot['elapsed_seconds']
    if not math.isfinite(r['elapsed_seconds']) or not 0 < r['elapsed_seconds'] <= remaining or r['budget_seconds'] != remaining:
        raise ValueError('Full phase exceeded fixed wall budget')
    if (r['feature_shape'] != m['cache']['shape'] or r['feature_dtype'] != 'float32'
            or r.get('observed_native_feature_dtypes') != ['torch.float32']
            or r['selected_ids_sha256'] != m['ordered_ids']['combined_sha256']
            or r['encoder_before_sha256'] != m['encoder_state_sha256'] or r['encoder_after_sha256'] != m['encoder_state_sha256']
            or r['runtime'] != m['expected_provenance']['eval_runtime']
            or any(r['environment'].get(k) != v for k, v in m['environment'].items())):
        raise ValueError('Feature/encoder/runtime/environment contract differs')
    hashes = {}
    names = {'features.npy': 'features_sha256', 'extraction-batches.json': 'extraction_batches_sha256',
             'frozen-b8-eval-predictions.npy': 'frozen_b8_common_cache_predictions_sha256'}
    for name, key in names.items():
        if sha(run / name) != r[key]:
            raise ValueError('Phase artifact hash mismatch: ' + name)
        hashes[name] = r[key]
    histories, scales, traces = {}, {}, {}
    for arm in ('control', 'treatment'):
        ar = r['arms'][arm]
        if ar['initial_shared_head_sha256'] != m['saved_head_sha256'] or ar['initial_slot_bias'] != 'allzero' or ar['feature_sha256'] != r['features_sha256']:
            raise ValueError('Arm changed common starting state/cache')
        for suffix, key in [('predictions.npy', 'predictions_sha256'), ('history.json', 'history_sha256'),
                            ('rng-trace.json', 'rng_trace_sha256'), ('scale-history.json', 'scale_history_sha256')]:
            name = arm + '-' + suffix
            if sha(run / name) != ar[key]:
                raise ValueError('Arm artifact hash mismatch: ' + name)
            hashes[name] = ar[key]
        histories[arm] = read(run / (arm + '-history.json'))
        scales[arm] = read(run / (arm + '-scale-history.json'))
        traces[arm] = read(run / (arm + '-rng-trace.json'))
    validate_counts(r, histories, scales, traces)
    if any(not math.isfinite(h['loss']) for hs in histories.values() for h in hs):
        raise ValueError('Nonfinite training loss')
    # Small head checkpoints only; no encoder execution or CUDA context.
    import torch
    from tools.exp083_slot_bias_v2 import state_hash
    if torch.cuda.is_initialized():
        raise ValueError('Evaluator must stay on CPU')
    for arm in ('control', 'treatment'):
        p = run / (arm + '-final.pt'); ck = torch.load(p, map_location='cpu', weights_only=True)
        if (ck['arm'], ck['epoch'], ck['protocol_sha256']) != (arm, 19, registration['protocol_sha256']) or state_hash(ck['model']) != r['arms'][arm]['final_state_sha256']:
            raise ValueError('Wrong fixed final head checkpoint')
        if arm == 'control' and bool(ck['model']['slot_bias'].any()):
            raise ValueError('Control bias changed from zero')
        hashes[p.name] = sha(p)
    for p in (run / 'receipt.json', job / 'job.json', job / 'SUCCESS', contained(root, registration['pilot_receipt']), contained(root, registration['root_full_decision'])):
        hashes[str(p.relative_to(root)).replace('\\', '/')] = sha(p)
    return r, hashes


def validate_prediction(values, ids, targets, csv_path=None):
    import numpy as np
    import pandas as pd
    if values.shape != (len(ids), 12) or values.dtype != np.float32 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Expected finite float32 probabilities n by12')
    if csv_path:
        df = pd.read_csv(csv_path, dtype={UID: str})
        if list(df.columns) != [UID, *targets] or df[UID].tolist() != ids or not df[UID].is_unique:
            raise ValueError('Prediction CSV exact UID/order/target mismatch')
        if not np.array_equal(df[targets].to_numpy(dtype=np.float32), values):
            raise ValueError('CSV and NPY probabilities differ')


def target_rows(y, predictions, targets=TARGETS, weights=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rows = []
    for j, target in enumerate(targets):
        p, n = int((y[:, j] == 1).sum()), int((y[:, j] == 0).sum())
        valid = p > 0 and n > 0
        if weights is not None:
            valid = (weights * y[:, j]).sum() > 0 and (weights * (1 - y[:, j])).sum() > 0
        auc = {arm: float(roc_auc_score(y[:, j], v[:, j], sample_weight=weights)) if valid else None for arm, v in predictions.items()}
        row = dict(target=target, P=p, N=n, auc=auc)
        row['primary_delta'] = auc['treatment'] - auc['control'] if valid else None
        row['secondary_control_minus_frozen'] = auc['control'] - auc['frozen_b8'] if valid else None
        row['secondary_treatment_minus_frozen'] = auc['treatment'] - auc['frozen_b8'] if valid else None
        rows.append(row)
    return rows


def macro(rows):
    import numpy as np
    valid = len(rows) == 12 and all(r['primary_delta'] is not None for r in rows)
    keys = ('primary_delta', 'secondary_control_minus_frozen', 'secondary_treatment_minus_frozen')
    return {**{k: float(np.mean([r[k] for r in rows])) if valid else None for k in keys},
            'auc': {a: float(np.mean([r['auc'][a] for r in rows])) if valid else None for a in ARMS},
            'complete12': valid}


def paired_bootstrap(y, predictions, components, reps=500, seed=20261003):
    """Single component draw shared by all arms and targets; never reduce macro."""
    import numpy as np
    groups, inverse = np.unique(np.asarray(components, dtype=str), return_inverse=True)
    if len(inverse) != len(y) or not len(groups) or not 0 < reps:
        raise ValueError('Invalid component bootstrap inputs')
    rng = np.random.default_rng(seed); draws_hash = hashlib.sha256()
    values, invalid, invalid_targets = [], 0, np.zeros(12, dtype=np.int64)
    for _ in range(reps):
        draw = rng.integers(0, len(groups), len(groups))
        draws_hash.update(draw.astype('<i8').tobytes())
        weights = np.bincount(draw, minlength=len(groups))[inverse]
        bad = ((weights[:, None] * y).sum(0) == 0) | ((weights[:, None] * (1 - y)).sum(0) == 0)
        if bad.any():
            invalid += 1; invalid_targets += bad; continue
        rows = target_rows(y, predictions, weights=weights)
        values.append([[r[k] for r in rows] for k in ('primary_delta', 'secondary_control_minus_frozen', 'secondary_treatment_minus_frozen')])
    def interval(v):
        return np.quantile(v, [.025, .975]).tolist() if len(v) else None
    arr = np.asarray(values)
    intervals = {}
    for k, name in enumerate(('primary_delta', 'secondary_control_minus_frozen', 'secondary_treatment_minus_frozen')):
        intervals[name] = dict(macro12=interval(arr[:, k].mean(1)) if len(arr) else None,
                              targets={t: interval(arr[:, k, j]) if len(arr) else None for j, t in enumerate(TARGETS)})
    return dict(attempted=reps, valid=len(values), invalid=invalid, invalid_targets=dict(zip(TARGETS, map(int, invalid_targets))),
                component_count=len(groups), seed=seed, draws_sha256=draws_hash.hexdigest(), intervals=intervals,
                scope='Conditional fixed-model component bootstrap, complete12-target valid draws only')


def run(registration_path, registration_sha256, root=ROOT):
    import numpy as np
    import pandas as pd
    root = Path(root)
    if sha(registration_path) != registration_sha256:
        raise ValueError('Analysis registration hash mismatch')
    reg = read(registration_path)
    if reg['schema'] != 'exp083-report-registration-v1' or reg['targets'] != TARGETS or reg['bootstrap']['reps'] != 500 or reg['bootstrap']['seed'] != 20261003:
        raise ValueError('Wrong frozen EXP083 analysis registration')
    out = contained(root, reg['output'])
    if out.exists():
        raise FileExistsError('One registered evaluation output already exists')
    observed = {}
    for relative, expected in reg['source_bindings'].items():
        bind(root, relative, expected, observed)
    m = read(contained(root, reg['protocol']))
    if sha(contained(root, reg['protocol'])) != reg['protocol_sha256']:
        raise ValueError('Actual protocol SHA differs from registration')
    if any(reg['source_bindings'].get(p) != v for p,v in m['source_bindings'].items()):
        raise ValueError('Analysis registration omitted or changed required source bindings')
    required = {'training/src/' + p:v for p,v in m['training_code_sha256'].items()}
    group_contract = read(root / GROUP_DIR / 'contract.json')
    required.update({GROUP_DIR + '/' + p:v['sha256'] for p,v in group_contract['outputs'].items()})
    required.update({p:sha(root / p) for p in ('tools/exp083_slot_bias_report.py','tests/test_exp083_slot_bias_report.py',META)})
    if any(reg['source_bindings'].get(p) != v for p,v in required.items()):
        raise ValueError('Analysis registration omitted training/group/reporter/metadata binding')
    for p, item in group_contract['outputs'].items():
        if (root / GROUP_DIR / p).stat().st_size != item['bytes']:
            raise ValueError('Component artifact byte size differs')
    from tools.exp082_r34_loss import install_gold_blocker
    install_gold_blocker(root)
    receipt, output_hashes = validate_completed(root, reg, m)
    protocol_dir = contained(root, reg['protocol']).parent
    ids = read(protocol_dir / 'ordered-ids.json')
    for part in ('train', 'eval'):
        if len(ids[part]) != m['ordered_ids'][part] or uid_hash(ids[part]) != m['ordered_ids'][part + '_sha256'] or len(set(ids[part])) != len(ids[part]):
            raise ValueError('Frozen ordered IDs differ')
    if set(ids['train']) & set(ids['eval']):
        raise ValueError('Train/eval ID overlap')
    batches = read(contained(root, reg['run_dir']) / 'extraction-batches.json')
    # Explicit split order, independent of JSON key ordering.
    expected_batches = [dict(split=part, size=len(ids[part][i:i + 4]), uids_sha256=uid_hash(ids[part][i:i + 4])) for part in ('train','eval') for i in range(0,len(ids[part]),4)]
    if batches != expected_batches:
        raise ValueError('Original split batch4 extraction boundaries differ')
    groups = pd.read_csv(root / GROUP_DIR / 'groups.csv', dtype={UID:str, 'component_id':str})
    if not groups[UID].is_unique or groups[UID].isna().any() or groups['component_id'].isna().any() or set(groups[UID]) != set(ids['train'] + ids['eval']):
        raise ValueError('Component membership/UID contract differs')
    groups = groups.set_index(UID)
    if set(groups.loc[ids['train'], 'component_id']) & set(groups.loc[ids['eval'], 'component_id']):
        raise ValueError('Known component crosses F2 train/eval')
    component_ids = groups.loc[ids['eval'], 'component_id'].to_numpy()
    frame = pd.read_csv(root / m['labels'], dtype={UID:str})
    if frame[UID].isna().any() or not frame[UID].is_unique:
        raise ValueError('Invalid label UID')
    frame = frame.dropna(subset=TARGETS)
    canonical = hashlib.sha256(frame[[UID,*TARGETS]].to_csv(index=False,lineterminator='\n').encode()).hexdigest()
    if canonical != m['label_hash'] or canonical != m['eval_label_hash']:
        raise ValueError('Canonical label contract differs')
    soft = frame.set_index(UID).loc[ids['eval'], TARGETS].to_numpy(dtype=np.float32)
    if not np.isfinite(soft).all() or ((soft < 0) | (soft > 1)).any():
        raise ValueError('Invalid soft reference labels')
    y = np.rint(soft).astype(np.int8)
    if any(len(np.unique(y[:,j])) != 2 for j in range(12)):
        raise ValueError('Full F2 must have positives and negatives for all12 targets')
    run_dir = contained(root, reg['run_dir']); predictions = {}
    for arm in ARMS:
        filename = 'frozen-b8-eval-predictions.npy' if arm == 'frozen_b8' else arm + '-predictions.npy'
        v = np.load(run_dir / filename, allow_pickle=False)
        csv = None if arm == 'frozen_b8' else run_dir / (arm + '-predictions.csv')
        validate_prediction(v, ids['eval'], TARGETS, csv); predictions[arm] = v
        if csv: output_hashes[csv.name] = sha(csv)
    metadata = pd.read_csv(root / META, dtype={UID:str})
    if not metadata[UID].is_unique or metadata[UID].isna().any():
        raise ValueError('Invalid scanner metadata UID')
    diagnostic = metadata.set_index(UID).reindex(ids['eval'])[['scanner_group','field','manufacturer']].fillna('unknown').astype(str)
    corpus_ids = np.load(root / m['corpus'] / 'ids.npy', allow_pickle=False).astype(str)
    masks = np.load(root / m['corpus'] / 'masks.npy', mmap_mode='r', allow_pickle=False)
    if len(set(corpus_ids)) != len(corpus_ids) or masks.shape != (len(corpus_ids), 64):
        raise ValueError('Invalid corpus masks/UIDs')
    mask_index = {u:i for i,u in enumerate(corpus_ids)}
    mask = masks[[mask_index[u] for u in ids['eval']]]
    diagnostic['cached_mask'] = np.where((mask == 0).any(1), 'some_zero_mask', 'all_nonzero')
    # Reserve the single output before any scoring; failed attempts remain visible.
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'analysis-plan.json', dict(registration_sha256=sha(registration_path), protocol_sha256=reg['protocol_sha256'], source_hashes=observed,
        output_hashes=output_hashes, primary=reg['primary'], secondary=reg['secondary'], bootstrap=reg['bootstrap'], limits=LIMITS))
    rows = target_rows(y, predictions)
    subgroups = []
    for field in diagnostic.columns:
        for value in sorted(diagnostic[field].unique()):
            take = diagnostic[field].to_numpy() == value
            sr = target_rows(y[take], {a:p[take] for a,p in predictions.items()})
            subgroups.append(dict(field=field, group=value, studies=int(take.sum()), small_group=int(take.sum()) < 40,
                                  exploratory=True, targets=sr, macro12=macro(sr)))
    report = dict(schema='exp083-slot-bias-report-v1', observed_at=datetime.now(timezone.utc).isoformat(),
        status='FIXED_EXPOSED_WEAK_F2_EVALUATED_NO_ADOPTION', registration_sha256=sha(registration_path),
        protocol_sha256=reg['protocol_sha256'], studies=len(y), eval_ids_sha256=uid_hash(ids['eval']),
        component_ids_sha256=uid_hash(component_ids.tolist()), reference_binary_rule=reg['reference_binary_rule'],
        targets=rows, macro12=macro(rows), fracture=rows[TARGETS.index('Fracture')],
        bootstrap=paired_bootstrap(y, predictions, component_ids), groups=subgroups, limits=LIMITS,
        historical_B_replay_gate=reg['historical_B_replay_gate'], execution=dict(arms=receipt['arms'], no_interim_F2_metrics=True))
    for relative, expected in observed.items():
        bind(root, relative, expected, {})
    for filename, expected in output_hashes.items():
        p = root / filename if '/' in filename else run_dir / filename
        if sha(p) != expected:
            raise ValueError('Run artifact changed during evaluation')
    if sha(registration_path) != registration_sha256:
        raise ValueError('Analysis registration changed during evaluation')
    write_new(out / 'report.json', report)
    write_new(out / 'SUCCESS.json', dict(report_sha256=sha(out / 'report.json'), registration_sha256=sha(registration_path)))
    return dict(status=report['status'], report=str(out / 'report.json'))


def main():
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', metavar='REGISTRATION')
    parser.add_argument('--protocol')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--registration-sha256')
    parser.add_argument('--job-dir', default='artifacts/logs/EXP083_SLOT_BIAS_V2_FULL')
    args = parser.parse_args()
    if args.prepare and (not args.protocol or not args.protocol_sha256):
        parser.error('--prepare requires --protocol and --protocol-sha256')
    if args.run and not args.registration_sha256:
        parser.error('--run requires --registration-sha256')
    result = prepare(args.protocol,args.protocol_sha256,job_dir=args.job_dir) if args.prepare else run(args.run,args.registration_sha256)
    print(json.dumps({k:v for k,v in result.items() if k in ('schema','status','report','protocol_sha256')}, indent=2))


if __name__ == '__main__':
    main()
