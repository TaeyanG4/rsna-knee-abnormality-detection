"""CPU-only fixed EXP087 coverage evaluator; no training/acquisition imports."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'exp087-coverage-report-v1'
EVIDENCE = 'docs/handoffs/evidence/2026-10/2026-10-04-exp087-report-tool-v1'
BASE_PATH = 'tools/exp084_group_loss_report.py'
BASE_SHA = '9eaec5593ad42c44dc345836fa22ec01431f5ab8f1bd790ad5c62b75a721f0fb'
PRIOR_REPORT = 'tools/exp086_dinov2s_report_v2.py'
PRIOR_REPORT_SHA = '5172d63306b7566d438fe75c79fb5fe3d1c8c62344700bc1aa5b6bd41dec359f'
PRIOR_REG = 'docs/handoffs/evidence/2026-10/2026-10-04-exp086-report-tool-v2/registration.json'
SCORES = ('B_original', 'candidate')
CONTRASTS = {'primary_candidate_minus_B': ('B_original', 'candidate')}
POLICY = 'coronal-fluid-second-series-preserve-baseline-centers-v1'
STAGE_SHA = '3f1541aeb190ba49164e7bc3d067d2b4c1bc0142b87f44acd0ca9a4b1e562371'
INITIAL_SHA = 'e4ea3ae625eacde8e3ab563b6c663d715600a0cea039e1c9ee9d201367d5b38e'
PRODUCER_SCHEMA = 'exp087-coverage-b8-single-v1'
HELPER = 'tools/exp087_training.py'
RUNNER = 'training/runners/exp087_coverage_b8.py'
TAG = 'EXP087_COVERAGE_B8_G1'
METRIC_ACCESS = 'deferred_single_fixed_evaluation'
PRIOR_REG_SHA = '30a2ed6768dd704b8868373327393c0cf52f8a7367f79a9e73116c680a03e331'
STAGE_ACCEPTANCE = 'docs/handoffs/evidence/2026-10/2026-10-04-exp087-coverage-scale-v1/root-stage-acceptance.json'
STAGE_ACCEPTANCE_SHA = '2c790e31feb4eee674b0fc54454fa150eedf8e0b272a75b71e6b985b53a26397'
STAGE_DIRECTORY = '.local/exp087-coverage/stage-v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

need(sha(ROOT / BASE_PATH) == BASE_SHA, 'Frozen passive statistics changed')
_spec = importlib.util.spec_from_file_location('exp087_base_report', ROOT / BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
TARGETS, UID = base.TARGETS, base.UID
read, contained, relative, write_new = base.read, base.contained, base.relative, base.write_new
LIMITS = base.LIMITS + [
    'The same exposed weak development V has been repeatedly observed; one fixed seed and fit.',
    'Generic external pretraining and Dread label ancestry are not independently certified.',
    'Input policy changes training and evaluation. Eligible63 versus other825 is a predeclared diagnostic, not an independent subgroup efficacy test.',
    'The315 staged inputs and scheduled replacements do not establish clinical new information; training drop_last and actual tensor differences are separate.',
    'No original-B bitwise training trace equality or candidate T4 serving parity claim.',
    'Small positive uncertain evidence and Baker/Fracture losses remain visible; no blend/routing/search.',
]


def positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def hexhash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def scores_for(candidate, original):
    c, b = [np.asarray(value, dtype=np.float64) for value in (candidate, original)]
    need(c.ndim == 2 and c.shape[1] == 12 and len(c) > 0 and c.shape == b.shape
         and all(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all() for v in (c, b)),
         'Matching finite n by12 raw probabilities required')
    return dict(B_original=b, candidate=c)


def target_rows(y, scores, weights=None):
    y = base.checked_binary(y)
    need(set(scores) == set(SCORES)
         and all(v.shape == y.shape and np.isfinite(v).all() for v in scores.values()),
         'Wrong raw score arms or shapes')
    weights = np.ones(len(y), dtype=np.int64) if weights is None else np.asarray(weights)
    rows = []
    for j, target in enumerate(TARGETS):
        auc = {name: base.weighted_auc(y[:, j], scores[name][:, j], weights) for name in SCORES}
        auc = {name: float(value) if np.isfinite(value) else None for name, value in auc.items()}
        delta = {name: None if auc[a] is None or auc[b] is None else auc[b] - auc[a]
                 for name, (a, b) in CONTRASTS.items()}
        rows.append(dict(target=target, P=int(y[:, j].sum()), N=int(len(y) - y[:, j].sum()),
                         auc=auc, delta=delta,
                         delta_over_12={k: None if v is None else v / 12 for k, v in delta.items()}))
    return rows


def aggregate(rows):
    need([row['target'] for row in rows] == TARGETS, 'Exact12 target rows required')
    complete = all(all(value is not None for value in row['auc'].values()) for row in rows)
    return dict(complete12=complete,
                auc={arm: float(np.mean([row['auc'][arm] for row in rows])) if complete else None for arm in SCORES},
                contrasts={key: {'macro12': float(np.mean([row['delta'][key] for row in rows])) if complete else None,
                                 **{target: next(row['delta'][key] for row in rows if row['target'] == target)
                                    for target in ["Baker's", 'Fracture', 'PF OA', 'MCL']}}
                           for key in CONTRASTS})


def paired_bootstrap(y, scores, components, reps=500, seed=20261004):
    y = base.checked_binary(y)
    components = np.asarray(components, dtype=str)
    groups, codes = np.unique(components, return_inverse=True)
    need(len(codes) == len(y) and len(groups) > 0 and not (components == '').any()
         and type(reps) is int and reps > 0, 'Invalid component bootstrap')
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    invalid_targets = np.zeros(12, np.int64)
    values, invalid = [], 0
    for _ in range(reps):
        draw = rng.integers(0, len(groups), len(groups))
        digest.update(draw.astype('<i8').tobytes())
        weights = base.component_weights(codes, draw, len(groups))
        bad = ((weights[:, None] * y).sum(0) == 0) | ((weights[:, None] * (1 - y)).sum(0) == 0)
        if bad.any():
            invalid += 1
            invalid_targets += bad
            continue
        rows = target_rows(y, scores, weights)
        values.append([[row['delta'][name] for row in rows] for name in CONTRASTS])
    array = np.asarray(values)
    intervals = {key: dict(macro12=np.quantile(array[:, i].mean(1), [.025, .975]).tolist() if values else None,
                          targets={target: np.quantile(array[:, i, j], [.025, .975]).tolist() if values else None
                                   for j, target in enumerate(TARGETS)})
                 for i, key in enumerate(CONTRASTS)}
    return dict(attempted=reps, valid=len(values), invalid=invalid,
                invalid_targets=dict(zip(TARGETS, map(int, invalid_targets))), component_count=len(groups),
                seed=seed, draws_sha256=digest.hexdigest(), same_draw_all_arms_and_targets=True,
                ranks_applied=False, intervals=intervals,
                scope='Conditional fixed raw-score component bootstrap; all CIs share complete12 valid draws.')


def inference_contract(p):
    a = p['arm']['expected_args']
    return {**{key: a[key] for key in ['res', 'k_eval', 'norm', 'window_policy', 'eval_amp', 'runtime_policy']},
            'corpus': p['paths']['corpus'], 'source': p['training_code_sha256'],
            'batch': max(1, a['bs'] // 2), 'workers': min(a['workers'], 2),
            'internal_cpu_threads': 1, 'grad_mode': 'no_grad', 'output': 'float32 sigmoid then ordered CSV'}


def resource_gate(receipt, expected_argv, max_seconds, source_sha):
    need(type(max_seconds) is int and max_seconds > 0 and hexhash(source_sha), 'Invalid resource contract')
    need(all(type(receipt.get(key)) is int and receipt[key] == 0
             for key in ['child_exit_code', 'wrapper_exit_code', 'child_exit_code_before_cleanup',
                         'retries', 'failed_sample_attempts', 'access_failure_count'])
         and type(receipt.get('sample_count')) is int and receipt['sample_count'] > 0
         and type(receipt.get('sampled_tree_rss_max_bytes')) is int
         and receipt['sampled_tree_rss_max_bytes'] > 0
         and type(receipt.get('timeout_seconds')) is int, 'Resource counter types or values invalid')
    need(receipt.get('schema') == 'process-resource-receipt-v1'
         and receipt.get('supervisor_sha256') == source_sha and receipt.get('argv') == expected_argv
         and receipt.get('status') == 'COMPLETE' and receipt.get('failure') is None
         and receipt.get('child_exit_code') == receipt.get('wrapper_exit_code') == receipt.get('child_exit_code_before_cleanup') == 0,
         'Supervised execution/command failed')
    need(positive_number(receipt.get('elapsed_seconds')) and receipt['elapsed_seconds'] < max_seconds
         and receipt.get('timeout_seconds') == max_seconds and receipt.get('retries') == 0,
         'Phase exceeded frozen runtime budget')
    need(all(receipt.get('cleanup', {}).get(key) == [] for key in ['actions', 'errors', 'survivors']),
         'Natural worker cleanup not established')
    pid = receipt.get('child_identity', {}).get('pid')
    need(type(pid) is int and pid > 0 and pid in {row.get('pid') for row in receipt.get('observed_identities', [])},
         'Owned worker PID missing')
    need(receipt.get('sample_count', 0) > 0 and receipt.get('sampled_tree_rss_max_bytes', 0) > 0
         and receipt.get('failed_sample_attempts') == receipt.get('access_failure_count') == 0,
         'Resource telemetry incomplete')
    return receipt['elapsed_seconds']


def command_value(argv, flag):
    need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), 'Missing/duplicate command flag: ' + flag)
    return argv[argv.index(flag) + 1]


def job_gate(root, spec):
    job = contained(root, spec['logs'])
    need((job / 'SUCCESS').is_file() and (job / 'job.json').is_file() and not (job / 'FAILED').exists(),
         'Registered job did not complete successfully')
    need(re.fullmatch(r'exit 0 at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s*',
                      (job / 'SUCCESS').read_text(encoding='utf-8')) is not None, 'Invalid SUCCESS marker')
    record = read(job / 'job.json')
    need(record['tag'] == spec['tag'] and record['command'] == spec['command'], 'Actual job differs from frozen command')
    return [job / 'SUCCESS', job / 'job.json']


def function_sha(path, name):
    text = Path(path).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256(''.join(text.splitlines(True)[node.lineno - 1:node.end_lineno]).encode()).hexdigest()


def arithmetic_binding(root):
    need(sha(contained(root, PRIOR_REPORT)) == PRIOR_REPORT_SHA, 'Prior statistical source changed')
    trees = [ast.parse(contained(root, path).read_text(encoding='utf-8'))
             for path in [PRIOR_REPORT, 'tools/exp087_coverage_report.py']]
    result = {}
    for name in ['scores_for', 'target_rows', 'aggregate', 'paired_bootstrap']:
        nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                 for tree in trees]
        need(ast.dump(nodes[0]) == ast.dump(nodes[1]), 'Frozen arithmetic changed: ' + name)
        result[name] = 'AST_IDENTICAL_TO_HASH_BOUND_EXP086_V2'
    return result


def inherited_ledgers(root):
    """Read a completed metadata ledger; never reopen its outcome/label/weight files."""
    path = contained(root, PRIOR_REG)
    need(sha(path) == PRIOR_REG_SHA, 'Completed original-B ledger changed')
    old = read(path)
    need(old['reference']['tag'] == 'EXP084_B_G1' and old['expected_ids'] ==
         dict(train=3455, val=888, train_components=3323, val_components=873), 'Wrong baseline ledger')
    return old


def coverage_groups(ids, manifest):
    """Fixed metadata eligibility, independent of scores and observed effect."""
    need(len(ids) == 888 and len(set(ids)) == 888, 'Exact888 V membership required')
    membership = manifest.get('membership', {})
    need(len(membership.get('val', [])) == 888 and set(ids) == set(membership['val']),
         'Validation diagnostic includes unregistered studies')
    entries = manifest.get('entries', [])
    need(len(entries) == 315 and len({r['uid'] for r in entries}) == 315,
         'Exact unique315 staged entries required')
    train = {r['uid'] for r in entries if r['split'] == 'train'}
    val = {r['uid'] for r in entries if r['split'] == 'val'}
    need(len(train) == 252 and len(val) == 63 and not train & val
         and val <= set(ids) and not train & set(ids), 'Frozen eligible split changed')
    need(all(r.get('reason') is None for r in entries), 'Accepted no-fallback315 contract changed')
    eligible = np.asarray([uid in val for uid in ids], dtype=bool)
    need(int(eligible.sum()) == 63 and int((~eligible).sum()) == 825, 'Eligible diagnostic counts changed')
    return {'eligible63': eligible, 'other825': ~eligible}


def group_rows(y, scores, groups, ids, manifest, fields):
    diagnostics = []
    selected = groups.loc[ids]
    for field in fields:
        values = selected[field].replace('', 'unknown').to_numpy(dtype=str)
        for value in sorted(set(values)):
            take = values == value
            rows = target_rows(y[take], {key: values_[take] for key, values_ in scores.items()})
            diagnostics.append(dict(field=field, group=value, studies=int(take.sum()),
                                    targets=rows, aggregate=aggregate(rows), ci=None))
    for name, take in coverage_groups(ids, manifest).items():
        rows = target_rows(y[take], {key: value[take] for key, value in scores.items()})
        diagnostics.append(dict(field='preregistered_coverage_eligibility', group=name,
                                studies=int(take.sum()), targets=rows, aggregate=aggregate(rows), ci=None,
                                scope='Descriptive fixed subgroup; no separate CI, test, routing or selection.'))
    return diagnostics


def stage_gate(root, *, read_manifest=True):
    """Accept the completed stage without decoding or reopening saved pixel arrays."""
    acceptance_path = contained(root, STAGE_ACCEPTANCE)
    need(sha(acceptance_path) == STAGE_ACCEPTANCE_SHA, 'Root stage acceptance changed')
    a = read(acceptance_path)
    expected = dict(schema='exp087-stage-root-acceptance-v1', status='PASS_INPUT_STAGE_ONLY',
                    entries=315, T=252, V=63, independent_new_arrays=307, reused_pilot_arrays=8,
                    manifest_sha256=STAGE_SHA, supervisor_status='PASS', child_exit=0,
                    cleanup_errors=0, fallbacks=0, training_authorization=False)
    need(all(a.get(key) == value for key, value in expected.items()), 'Stage acceptance scope differs')
    bound = {STAGE_ACCEPTANCE: STAGE_ACCEPTANCE_SHA}
    for filename, key in [('manifest.json', 'manifest_sha256'), ('receipt.json', 'receipt_sha256'),
                          ('supervisor.json', 'supervisor_sha256'), ('job.json', 'job_sha256')]:
        bound[STAGE_DIRECTORY + '/' + filename] = a[key]
    for name, value in a['evidence_sha256'].items():
        bound[relative(root, acceptance_path.parent / name)] = value
    base.verify_bindings(root, bound)
    s = read(contained(root, STAGE_DIRECTORY + '/supervisor.json'))
    need(s['status'] == 'PASS' and s['returncode'] == 0 and s['killed'] is None and s['error'] is None
         and s['cleanup_killed_pids'] == s['cleanup_errors'] == []
         and positive_number(s['wall_seconds']) and s['wall_seconds'] < 900
         and 0 < s['peak_process_tree_rss'] <= 8 * 1024**3, 'Stage supervisor did not succeed cleanly')
    need(not contained(root, STAGE_DIRECTORY + '/FAIL.json').exists(), 'Stage failure marker present')
    r = read(contained(root, STAGE_DIRECTORY + '/receipt.json'))
    need(r['status'] == 'CPU_STAGING_COMPLETE' and r['manifest_sha256'] == STAGE_SHA
         and len(r['entries']) == 315 and r['reused8'] == 8 and r['new_planned307'] == 307,
         'Stage receipt incomplete')
    independent = read(acceptance_path.parent / 'result-independent-review.json')
    need(independent['status'] == 'PASS_INPUT_STAGE_ACCEPTANCE_ONLY', 'Independent stage acceptance missing')
    if not read_manifest:
        return bound, None
    manifest = read(contained(root, STAGE_DIRECTORY + '/manifest.json'))
    need(manifest['status'] == 'STAGED_INPUTS_ONLY_NO_TRAINING' and manifest['policy'] == POLICY
         and manifest['fixture_sha256'] == a['fixture_sha256'], 'Stage manifest policy differs')
    coverage_groups(manifest['membership']['val'], manifest)
    return bound, manifest


def protocol_view(raw, parent):
    """Use the frozen producer's actual two-phase schema; parent supplies unchanged reference metadata."""
    need(raw.get('schema') == PRODUCER_SCHEMA and set(raw.get('arms', {})) == {'pilot', 'full'},
         'Wrong single-candidate two-phase protocol')
    shared = ['paths', 'ordered_ids', 'groups', 'expected_ids', 'excluded_uids', 'loss',
              'corpus_manifest_sha256', 'label_hash', 'eval_label_hash', 'split_hash',
              'training_code_sha256', 'expected_runtime', 'environment', 'labels', 'initial_assets']
    need(all(raw.get(key) == parent[key] for key in shared), 'Producer changed originalB metadata/recipe')
    need(raw['coverage_manifest_sha256'] == STAGE_SHA and raw['metric_access'] == METRIC_ACCESS
         and raw['baseline_comparator'] == 'EXP084_B_G1', 'Coverage/reference/metric contract changed')
    original = parent['arms']['B']
    evaluation = dict(primary='macro12(candidate)-macro12(EXP084_B_G1)', bootstrap_unit='component_id',
                      bootstrap_reps=500, bootstrap_seed=20261004, no_rank_or_blend=True)
    need(raw.get('evaluation') == evaluation and raw.get('reference_outputs') == original['outputs'],
         'Producer primary/bootstrap/reference outputs changed')
    for mode, updates in [('pilot', 32), ('full', 3448)]:
        arm = raw['arms'][mode]
        need(arm['tag'] == TAG + ('_PILOT' if mode == 'pilot' else '')
             and arm['expected_args'] == {**original['expected_args'], 'mode': mode}
             and arm['expected_attempts'] == updates and arm['initial_state_sha256'] == INITIAL_SHA,
             'Fixed phase recipe/initialization changed')
        need(arm['loss_policy'] == original['loss_policy'] and arm['actual_pos_weight'] == original['actual_pos_weight'],
             'Pilot/full loss intervention differs from originalB')
    need(raw['training_budgets'] == dict(pilot_seconds=180, full_seconds=3600,
         total_GPU_seconds=3780, rss_bytes=24 * 1024**3, retry=0), 'Frozen training budget differs')
    need(raw['windows_gate']['seconds'] == 180 and raw['windows_gate']['workers'] == 12
         and raw['windows_gate']['internal'] == 1 and raw['windows_gate']['rss_bytes'] == 24 * 1024**3,
         'Frozen Windows gate contract differs')
    return {**parent, **raw, 'arm': raw['arms']['full'], 'reference': original, 'evaluation': evaluation}


def validate_protocol(p, parent):
    need(p.get('schema') == PRODUCER_SCHEMA, 'Wrong fixed coverage producer schema')
    need(p['targets'] == TARGETS and p['arm']['tag'] == TAG, 'Wrong coverage arm/target order')
    for key in ['paths', 'ordered_ids', 'groups', 'expected_ids', 'split', 'excluded_uids', 'loss',
                'reference_counts', 'macro12_valid', 'corpus_manifest_sha256', 'label_hash',
                'eval_label_hash', 'split_hash', 'training_code_sha256', 'expected_runtime', 'environment']:
        need(p[key] == parent[key], 'Original-B fixed contract differs: ' + key)
    original = parent['arms']['B']
    need(p['arm']['expected_args'] == {**original['expected_args'], 'mode': 'full'},
         'Learning recipe changed beyond input coverage')
    for key in ['initial_state_sha256', 'expected_attempts', 'loss_policy', 'actual_pos_weight']:
        need(p['arm'][key] == original[key], 'Original B ancestry/loss/update contract differs: ' + key)
    need(p['arm']['initial_state_sha256'] == INITIAL_SHA and p['arm']['expected_attempts'] == 3448,
         'Original generic initialization or3448 updates differs')
    need(p['reference']['tag'] == 'EXP084_B_G1' and p['reference']['outputs'] == original['outputs'],
         'Completed original B reference changed')
    need(p['expected_ids'] == dict(train=3455, val=888, train_components=3323, val_components=873),
         'Exact G1 cohort required')
    infer = inference_contract(p)
    need(infer == inference_contract({**parent, 'arm': original}), 'Shared inference recipe changed')
    ev = p['evaluation']
    need(ev['primary'] == 'macro12(candidate)-macro12(EXP084_B_G1)'
         and ev['bootstrap_unit'] == 'component_id' and ev['bootstrap_reps'] == 500
         and ev['bootstrap_seed'] == 20261004 and ev['no_rank_or_blend'] is True,
         'Raw primary/bootstrap preregistration changed')
    return infer


def validate_candidate(summary, provenance, history, p, protocol_sha):
    arm = p['arm']; args = arm['expected_args']; n = p['expected_ids']
    need(summary.get('provenance') == provenance and summary.get('history') == history,
         'Summary/provenance/history disagree')
    expected = dict(code_sha256=p['training_code_sha256'], initial_state_sha256=INITIAL_SHA,
                    train_ids_sha256=p['ordered_ids']['train_sha256'], eval_ids_sha256=p['ordered_ids']['val_sha256'],
                    train_ids_ordered_sha256=p['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=p['ordered_ids']['val_sha256'],
                    corpus_manifest_sha256=p['corpus_manifest_sha256'], checkpoint_selection='fixed-final',
                    sampling='sha256(seed,epoch,uid,stream)-v1', gold_access='disabled',
                    loss_policy=arm['loss_policy'], actual_pos_weight=arm['actual_pos_weight'],
                    baseline_pos_weight=p['loss']['baseline_pos_weight'], metric_access=METRIC_ACCESS,
                    protocol_sha256=protocol_sha, adapter_sha256=p['source_bindings'][RUNNER],
                    contract_helper_sha256=p['source_bindings'][HELPER], coverage_manifest_sha256=STAGE_SHA,
                    requested_workers=12, effective_workers=12, internal_cpu_threads=1,
                    scheduler_total_steps=3448)
    need(all(provenance.get(key) == value for key, value in expected.items()),
         'Candidate source/ancestor/recipe/input identity mismatch')
    need(all(provenance.get(key) == p['expected_runtime'][key] for key in ['runtime', 'eval_runtime']),
         'Strict train/eval FP16 runtime differs')
    need(all(summary['environment'].get(key) == p['environment'][key]
             for key in ['torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'pydicom', 'cv2']),
         'Runtime environment changed')
    need(all(summary['args'].get(key) == value for key, value in args.items()), 'Actual training arguments changed')
    required = dict(gold=0, gold_access='disabled', no_gold_epoch_selection=True, metric_access=METRIC_ACCESS, arm='full',
                    selected_checkpoint='best', best_epoch=7, selected_epoch_1based=8, max_epochs=8,
                    train=n['train'], val=n['val'], rows=n['val'], no_oof_epoch_selection=True,
                    weak_oof_macro_auc=None, oof_per_target=None, label_hash=p['label_hash'],
                    eval_label_hash=p['eval_label_hash'], split_hash=p['split_hash'],
                    expected_attempts=3448, optimizer_updates=3448, skipped_updates=0,
                    optimizer_skip_fraction=0, backbone=args['arch'], img=224,
                    contract='raptor64_336_c140_s02', window_policy='within-slot-valid')
    need(all(summary.get(key) == value for key, value in required.items()), 'Fixed8epoch completion differs')
    need(type(summary['optimizer_updates']) is int and type(summary['skipped_updates']) is int
         and positive_number(summary['total_seconds']), 'Completion counter/time types invalid')
    need(len(history) == 8, 'Eight completed epochs required')
    for epoch, row in enumerate(history):
        need(type(row.get('epoch')) is int and row['epoch'] == epoch
             and type(row.get('optimizer_updates')) is int and row['optimizer_updates'] == (epoch + 1) * 431
             and type(row.get('skipped_updates')) is int and row['skipped_updates'] == 0
             and positive_number(row.get('loss')) and row.get('oof_auc') is None and row.get('gold_auc') is None
             and row.get('selection') == float(epoch), 'Epoch/update/metric-selection drift')
    need(summary['results'] == {'best': {'oof_rows': 888, 'metric_access': METRIC_ACCESS}},
         'Unexpected intermediate/alternate results')
    return dict(updates=3448, skips=0, epochs=8, fixed_final=True, initial_state_sha256=INITIAL_SHA,
                comparison='Single train-and-eval input-coverage treatment versus completed original B8',
                metrics_deferred=True, gold_access='disabled', reported_training_seconds=summary['total_seconds'])


def exact_worker_command(p, mode, protocol_path, root, python):
    if mode == 'windows':
        return [python, str(contained(root, 'tools/exp087_windows_gate.py')), '--protocol',
                str(protocol_path), '--output', str(contained(root, p['windows_gate']['output']))]
    arm = p['arms'][mode]
    command = [python, str(contained(root, RUNNER)), '--protocol', str(protocol_path), '--tag', arm['tag']]
    for key, value in {**arm['expected_args'], **{k: str(contained(root, v)) for k, v in p['paths'].items()}}.items():
        if isinstance(value, bool):
            if value: command.append('--' + key.replace('_', '-'))
        else:
            command += ['--' + key.replace('_', '-'), str(value)]
    return command


def parse_plan(plan, p, protocol_path, protocol_sha, root):
    need(plan['protocol_sha256'] == protocol_sha and set(plan['plans']) == {'windows', 'pilot', 'full'},
         'Exact three-phase execution plan required')
    jobs = {}
    for mode in ['windows', 'pilot', 'full']:
        item = plan['plans'][mode]; argv = item['argv']
        need(item['mode'] == mode and item['status'] == 'ROOT_EXECUTION_DECISION_REQUIRED'
             and Path(item['cwd']).resolve() == Path(root).resolve(), 'Plan mode/cwd differs')
        if mode == 'windows':
            command = argv
        else:
            tag = TAG + ('_PILOT' if mode == 'pilot' else '')
            need(argv[:5] == [argv[0], str(contained(root, 'tools/gpu_job.py')), '--tag', tag, '--'],
                 'GPU job wrapper/tag differs')
            command = argv[5:]
        need(command[:3] == [argv[0], str(contained(root, HELPER)), 'run']
             and command_value(command, '--protocol') == str(protocol_path)
             and command_value(command, '--protocol-sha256') == protocol_sha
             and command_value(command, '--mode') == mode and '--execute' in command,
             'Plan command does not invoke exact protocol/phase')
        # Additional root authorization options are frozen by the actual plan; no CLI is inferred.
        output = p['windows_gate']['output'] if mode == 'windows' else 'artifacts/experiments/' + tag
        jobs[mode] = dict(mode=mode, command=command, plan_argv=argv, output=output,
                          worker_command=exact_worker_command(p, mode, protocol_path, root, argv[0]))
        if mode != 'windows': jobs[mode].update(tag=tag, logs='artifacts/logs/' + tag)
    return jobs


def supervisor_gate(s, job, spec, protocol_sha, cap, rss_cap):
    need(s.get('status') == 'PASS' and type(s.get('returncode')) is int and s['returncode'] == 0
         and s.get('error') is None and s.get('killed') is None
         and s.get('cleanup_errors') == s.get('cleanup_killed_pids') == []
         and s.get('protocol_sha256') == protocol_sha, 'Clean phase supervisor completion required')
    need(positive_number(s.get('wall_seconds')) and s['wall_seconds'] < cap
         and type(s.get('peak_tree_RSS')) is int and 0 < s['peak_tree_RSS'] <= rss_cap,
         'Phase resource budget exceeded or invalid')
    need(job.get('argv') == spec['worker_command'] and job.get('protocol_sha256') == protocol_sha
         and type(job.get('pid')) is int and job['pid'] > 0
         and type(job.get('retry')) is int and job['retry'] == 0
         and job.get('seconds_cap') == cap and job.get('rss_cap') == rss_cap,
         'Phase command/resource ownership differs')
    return s['wall_seconds']


def trace_gate(trace, mode):
    updates, epochs = (32, 1) if mode == 'pilot' else (3448, 8)
    need(trace.get('schema') == 'exp087-delivered-input-trace-v1'
         and type(trace.get('updates')) is int and trace['updates'] == updates
         and type(trace.get('scheduler_steps')) is int and trace['scheduler_steps'] == updates
         and type(trace.get('epochs')) is int and trace['epochs'] == epochs
         and hexhash(trace.get('delivered_uid_sha256')), 'Delivered trace/update horizon differs')
    for name in ['eligible_deliveries', 'scheduled_replacements', 'actual_replacement_tensor_checks']:
        need(type(trace.get(name)) is int and trace[name] > 0, 'No verified treatment delivery: ' + name)
    need(trace['eligible_deliveries'] <= updates * 8
         and trace['scheduled_replacements'] <= (1742 if mode == 'full' else updates * 8 * 6)
         and trace['actual_replacement_tensor_checks'] <= trace['eligible_deliveries'],
         'Impossible delivered treatment counters')
    clean = trace.get('worker_cleanup', {})
    need(clean.get('clean') is True and len(clean.get('pids', [])) == len(clean.get('exitcodes', [])) == 12
         and len(set(clean['pids'])) == 12 and all(type(v) is int and v > 0 for v in clean['pids'])
         and all(type(v) is int and v == 0 for v in clean['exitcodes']), 'Persistent worker cleanup not established')
    return trace


def pilot_gate(receipt, trace):
    trace_gate(trace, 'pilot')
    expected = dict(status='PILOT_COMPLETE_NO_FULL_AUTHORIZATION', updates=32, skips=0,
                    scheduler_horizon=3448, projection_pass=True, no_checkpoint_saved=True,
                    no_validation_inference=True, no_metric_access=True)
    need(all(receipt.get(k) == v for k, v in expected.items()), 'T-only pilot completion differs')
    times = trace.get('pilot_whole_update_seconds', [])
    need(len(times) == 32 and all(positive_number(v) for v in times), 'Whole pilot update timings missing')
    projection = 1.3 * sum(times[8:]) / 24 * 3448 + 300
    need(positive_number(receipt.get('projected_full_seconds'))
         and math.isclose(receipt['projected_full_seconds'], projection, rel_tol=0, abs_tol=1e-7)
         and projection <= 3600, 'Pilot throughput projection fails')
    need(type(receipt.get('peak_cuda_bytes')) is int and receipt['peak_cuda_bytes'] > 0,
         'Pilot GPU memory telemetry missing')
    return projection


def windows_receipt_gate(receipt, p, protocol_sha):
    expected = dict(status='PASS_WINDOWS_INPUT_INTEGRATION', schema='exp087-windows-input-gate-v1',
                    protocol_sha256=protocol_sha, coverage_manifest_sha256=STAGE_SHA,
                    population=3455, production_batches=431, workers=12, internal=1, batch=8, K=12,
                    epochs=[0, 1], modes=['disabled', 'forced', 'enabled'],
                    production_baseline_corpusNone_at_spawn=True, multiprocessing_context='spawn',
                    parent_comparator_is_separate=True, model_loads=0, GPU=0)
    need(all(receipt.get(k) == v for k, v in expected.items()), 'Actual Windows integration contract differs')
    outcomes = receipt.get('results', [])
    need([(r['mode'], r['epoch']) for r in outcomes] ==
         [(mode, ep) for mode in ['disabled', 'forced', 'enabled'] for ep in [0, 1]],
         'Missing Windows no-op/treatment/epoch cases')
    need(all([r['uid'] for r in row['rows']] == p['windows_gate']['uids']
             and all(hexhash(r.get('input_sha256')) and hexhash(r.get('baseline_sha256'))
                     for r in row['rows']) for row in outcomes), 'Windows selected delivery/hash evidence missing')
    cleanup = receipt.get('worker_cleanup', [])
    need(len(cleanup) == 3 and all(row.get('clean') is True and len(row.get('pids', [])) == 12
         and len(row.get('exitcodes', [])) == 12 and all(type(v) is int and v == 0 for v in row['exitcodes'])
         for row in cleanup), 'Windows workers not all naturally closed')
    changes = receipt.get('replacement_checks', [])
    need(len(changes) == 48, 'Windows48 individual tensor checks missing')
    active = p['windows_gate']['cases']['eligible_active']
    zero = p['windows_gate']['cases']['eligible_zero_epoch0']
    missing = p['windows_gate']['cases']['missing']
    need(any(r['mode'] == 'enabled' and r['epoch'] == 0 and r['uid'] == active and r['changed'] for r in changes)
         and not any(r['changed'] for r in changes if r['mode'] != 'enabled' or r['uid'] == missing
                     or (r['uid'] == zero and r['epoch'] == 0)), 'Windows treatment/no-op check failed')
    need(all(set(r['changed']) <= {int(k) for k in r['replacement_map']} for r in changes),
         'Windows unchanged positions not preserved')


def require_completed(root, reg, p):
    """Validate phase chain and immutable final outputs before any outcome parsing."""
    hashes = {}; elapsed = {}; phases = {}
    for mode in ['windows', 'pilot', 'full']:
        spec = reg['jobs'][mode]; out = contained(root, spec['output'])
        paths = {name: out / name for name in ['resource-supervisor.json', 'supervisor-job.json', 'phase-authorization.json']}
        need(all(path.is_file() for path in paths.values())
             and not any((out / name).exists() for name in ['FAIL.json', 'failure.json', 'FAILED']),
             'Phase not completed or failure marker present: ' + mode)
        authorization = read(paths['phase-authorization.json'])
        auth_sha = sha(paths['phase-authorization.json'])
        decision_path = Path(authorization['root_decision_path'])
        need(decision_path.resolve().is_relative_to(Path(root).resolve()), 'Root decision outside workspace')
        decision = read(decision_path)
        need(sha(decision_path) == authorization['root_decision_sha256']
             and authorization['decision'] == decision
             and authorization['protocol_sha256'] == reg['protocol_sha256']
             and authorization['execution_plans_sha256'] == reg['execution_plan_sha256']
             and decision.get('status') == 'AUTHORIZE_EXACT_PHASE' and decision.get('phase') == mode
             and decision.get('protocol_sha256') == reg['protocol_sha256'], 'Phase root authorization mismatch')
        need(Path(command_value(spec['command'], '--decision')).resolve() == decision_path.resolve(),
             'Executed decision path differs from frozen plan')
        previous = {}
        if mode in ('pilot', 'full'):
            gate_dir = contained(root, reg['jobs']['windows']['output'])
            previous.update({relative(root, gate_dir / name): sha(gate_dir / name)
                             for name in ['receipt.json', 'resource-supervisor.json']})
        if mode == 'full':
            pilot_dir = contained(root, reg['jobs']['pilot']['output'])
            previous.update({relative(root, pilot_dir / name): sha(pilot_dir / name)
                             for name in ['pilot-receipt.json', 'resource-supervisor.json', 'provenance.json', 'input-trace.json']})
        need(decision.get('prior_artifact_sha256', {}) == previous
             and authorization['prior_artifact_sha256'] == previous, 'Previous phase artifacts not accepted exactly')
        s, job = read(paths['resource-supervisor.json']), read(paths['supervisor-job.json'])
        need(all(value.get('phase') == mode and value.get('phase_authorization_sha256') == auth_sha
                 for value in [s, job]), 'Supervisor belongs to different phase/authorization')
        start, end = datetime.fromisoformat(job['started_at']), datetime.fromisoformat(s['finished_at'])
        registered = datetime.fromisoformat(reg['prepared_at'])
        need(start.tzinfo is not None and end.tzinfo is not None and registered.tzinfo is not None
             and registered.replace(microsecond=0) <= start <= end,
             'Invalid phase execution chronology')
        cap = 180 if mode != 'full' else 3600
        elapsed[mode] = supervisor_gate(s, job, spec, reg['protocol_sha256'], cap, 24 * 1024**3)
        if mode != 'windows':
            for path in job_gate(root, spec): hashes[relative(root, path)] = sha(path)
        for path in [*paths.values(), decision_path]: hashes[relative(root, path)] = sha(path)
        phases[mode] = out
    need(elapsed['pilot'] + elapsed['full'] <= 3780, 'Combined GPU phase budget exceeded')
    gate_receipt = read(phases['windows'] / 'receipt.json')
    windows_receipt_gate(gate_receipt, p, reg['protocol_sha256'])
    hashes[relative(root, phases['windows'] / 'receipt.json')] = sha(phases['windows'] / 'receipt.json')
    pilot = read(phases['pilot'] / 'pilot-receipt.json')
    pilot_trace = read(phases['pilot'] / 'input-trace.json')
    need(pilot.get('input_trace_sha256') == sha(phases['pilot'] / 'input-trace.json')
         and pilot.get('provenance_sha256') == sha(phases['pilot'] / 'provenance.json'),
         'Pilot trace/provenance hash inventory differs')
    projection = pilot_gate(pilot, pilot_trace)
    pilot_prov = read(phases['pilot'] / 'provenance.json')
    need(pilot_prov['initial_state_sha256'] == INITIAL_SHA and pilot_prov['protocol_sha256'] == reg['protocol_sha256']
         and pilot_prov['training_mode'] == 'pilot' and pilot_prov['scheduler_total_steps'] == 3448
         and pilot_prov['coverage_manifest_sha256'] == STAGE_SHA
         and pilot_prov['runtime'] == p['expected_runtime']['runtime'], 'Pilot initialization/input/runtime differs')
    need(not list(phases['pilot'].glob('model*.pt')) and not list(phases['pilot'].glob('oof*.csv')),
         'Pilot saved forbidden weights/predictions')
    for name in ['pilot-receipt.json', 'provenance.json', 'input-trace.json']:
        hashes[relative(root, phases['pilot'] / name)] = sha(phases['pilot'] / name)
    out = phases['full']
    need(not list(out.glob('gold*')) and not (out / 'model_swa.pt').exists(), 'Unexpected Gold/SWA artifacts')
    inventory = read(out / 'output-inventory.json')
    names = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'input-trace.json'}
    need(inventory.get('schema') == 'exp087-training-output-inventory-v1' and set(inventory['files']) == names,
         'Full artifact inventory incomplete')
    for name, record in inventory['files'].items():
        path = out / name
        need(path.is_file() and type(record['bytes']) is int and path.stat().st_size == record['bytes']
             and sha(path) == record['sha256'], 'Full artifact hash/size mismatch: ' + name)
        hashes[relative(root, path)] = record['sha256']
    hashes[relative(root, out / 'output-inventory.json')] = sha(out / 'output-inventory.json')
    trace = trace_gate(read(out / 'input-trace.json'), 'full')
    provenance = read(out / 'provenance.json')
    need(provenance['training_mode'] == 'full'
         and provenance['coverage_adapter_sha256'] == p['source_bindings'][p['coverage_adapter']]
         and provenance['coverage_policy_sha256'] == p['source_bindings'][p['coverage_helper']]
         and provenance['stage_acceptance_bindings'] == p['acceptance_bindings'], 'Full coverage lineage differs')
    same_provenance = ['code_sha256', 'initial_state_sha256', 'runtime', 'eval_runtime', 'train_ids_sha256',
                      'eval_ids_sha256', 'train_ids_ordered_sha256', 'eval_ids_ordered_sha256',
                      'corpus_manifest_sha256', 'checkpoint_selection', 'sampling', 'gold_access',
                      'loss_policy', 'actual_pos_weight', 'baseline_pos_weight', 'metric_access',
                      'protocol_sha256', 'adapter_sha256', 'contract_helper_sha256',
                      'coverage_manifest_sha256', 'coverage_adapter_sha256', 'coverage_policy_sha256',
                      'stage_acceptance_bindings', 'scheduler_total_steps', 'requested_workers',
                      'effective_workers', 'internal_cpu_threads']
    need(all(key in pilot_prov and pilot_prov[key] == provenance.get(key) for key in same_provenance),
         'Pilot and clean full initial/input/runtime provenance differ')
    execution = validate_candidate(read(out / 'summary.json'), provenance, read(out / 'history.json'), p, reg['protocol_sha256'])
    execution.update(phase_seconds=elapsed, pilot_conservative_projection_seconds=projection,
                     delivered_uid_sha256=trace['delivered_uid_sha256'], eligible_deliveries=trace['eligible_deliveries'],
                     scheduled_replacements=trace['scheduled_replacements'], actual_tensor_checks=trace['actual_replacement_tensor_checks'],
                     trace_scope='Delivered candidate trace; no historical original-B trace equality claim')
    return execution, hashes


def prepare(protocol, protocol_sha256, execution_plan, root=ROOT):
    """Freeze source/metadata only; real label, prediction and weight bytes stay unopened."""
    root = Path(root); protocol_path = contained(root, protocol)
    need(sha(protocol_path) == protocol_sha256, 'Producer protocol hash differs')
    raw = read(protocol_path)
    parent_path = contained(root, 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json')
    parent = read(parent_path); base.validate_protocol(parent)
    p = protocol_view(raw, parent); infer = validate_protocol(p, parent)
    old = inherited_ledgers(root)
    need(old['source_bindings'].get(relative(root, parent_path)) == sha(parent_path),
         'Original parent protocol differs from completed ledger')
    need(old['reference'] == parent['arms']['B'] and old['input_bindings'] ==
         {key: raw['source_bindings'].get(key) for key in old['input_bindings']},
         'Inherited original data/reference hash ledger differs')
    plan_path = contained(root, execution_plan)
    jobs = parse_plan(read(plan_path), p, protocol_path, protocol_sha256, root)
    need(all(not contained(root, spec['output']).exists() for spec in jobs.values()),
         'Evaluation preregistration must precede new phase outputs')
    stage_bindings, _ = stage_gate(root, read_manifest=False)
    arithmetic = arithmetic_binding(root)
    sources = {key: value for key, value in raw['source_bindings'].items()
               if key.endswith('.py') or key.startswith('docs/')}
    sources.update(stage_bindings)
    sources.update({relative(root, protocol_path): protocol_sha256,
                    relative(root, parent_path): sha(parent_path),
                    relative(root, plan_path): sha(plan_path), PRIOR_REG: PRIOR_REG_SHA,
                    BASE_PATH: BASE_SHA, PRIOR_REPORT: PRIOR_REPORT_SHA})
    for key in ['tools/exp087_coverage_report.py', 'tests/test_exp087_coverage_report.py']:
        sources[key] = sha(contained(root, key))
    base.verify_bindings(root, sources)
    output = EVIDENCE + '/evaluation-v1'
    need(not contained(root, output).exists(), 'Fixed one-shot evaluation output exists')
    reg = dict(schema=SCHEMA, prepared_at=datetime.now(timezone.utc).isoformat(),
               protocol=relative(root, protocol_path), protocol_sha256=protocol_sha256,
               parent_protocol=relative(root, parent_path), parent_protocol_sha256=sha(parent_path),
               execution_plan=relative(root, plan_path), execution_plan_sha256=sha(plan_path),
               source_bindings=sources, producer_bindings=raw['source_bindings'],
               input_bindings=old['input_bindings'], reference_hashes=old['reference_hashes'],
               reference=old['reference'], inference_contract=infer, jobs=jobs,
               targets=TARGETS, scores=list(SCORES), contrasts=CONTRASTS,
               bootstrap=dict(reps=500, seed=20261004, unit='component_id', shared_complete12_draws=True),
               coverage_groups=dict(source=STAGE_DIRECTORY + '/manifest.json', sha256=STAGE_SHA,
                                    rule='split=val staged metadata eligibility before outcome access',
                                    eligible=63, other=825, subgroup_ci=False),
               reference_counts=parent['reference_counts'], expected_ids=parent['expected_ids'],
               output=output, limits=LIMITS, unchanged_arithmetic=arithmetic,
               inherited_hash_scope='Original label, prediction and weight hashes reused from the completed hash-bound ledger; none reopened during preparation.',
               status='CPU_REGISTERED_PENDING_ROOT_SOURCE_REVIEW_AND_EXECUTIONS')
    destination = contained(root, EVIDENCE); destination.mkdir(parents=True, exist_ok=True)
    write_new(destination / 'registration.json', reg)
    return reg


def run(registration, registration_sha256, root=ROOT):
    root = Path(root); registration_path = contained(root, registration)
    need(sha(registration_path) == registration_sha256, 'Registration hash mismatch')
    reg = read(registration_path)
    need(reg.get('schema') == SCHEMA and reg['targets'] == TARGETS and reg['scores'] == list(SCORES)
         and reg['contrasts'] == {k: list(v) for k, v in CONTRASTS.items()}
         and reg['bootstrap'] == dict(reps=500, seed=20261004, unit='component_id', shared_complete12_draws=True)
         and reg['output'] == EVIDENCE + '/evaluation-v1'
         and reg['coverage_groups'] == dict(source=STAGE_DIRECTORY + '/manifest.json', sha256=STAGE_SHA,
             rule='split=val staged metadata eligibility before outcome access', eligible=63, other=825, subgroup_ci=False),
         'Frozen analysis contract changed')
    out = contained(root, reg['output'])
    need(not out.exists(), 'One-shot evaluation path already exists')
    base.verify_bindings(root, reg['source_bindings'])
    parent = read(contained(root, reg['parent_protocol']))
    raw = read(contained(root, reg['protocol']))
    p = protocol_view(raw, parent); infer = validate_protocol(p, parent)
    need(infer == reg['inference_contract'] and parent['reference_counts'] == reg['reference_counts']
         and parent['expected_ids'] == reg['expected_ids'], 'Registered inference/reference metadata changed')
    execution, hashes = require_completed(root, reg, p)
    stage_bindings, manifest = stage_gate(root)
    hashes.update(stage_bindings)
    ordered, groups = base.cohort_metadata(root, parent)
    need(manifest['membership'] == ordered, 'Stage versus evaluated cohort ordering differs')
    ref = reg['reference']
    need(not (contained(root, ref['logs']) / 'FAILED').exists(), 'Original B failure marker present')
    base.validate_execution(read(contained(root, ref['outputs']['summary'])),
                            read(contained(root, ref['outputs']['provenance'])), parent, 'B', reg['parent_protocol_sha256'])
    # Atomic claim precedes all label/prediction parsing, statistics and derived outputs.
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_new(out / 'analysis-plan.json', dict(registration_sha256=registration_sha256,
                  output_hashes=hashes, execution=execution, contrasts=CONTRASTS, limits=LIMITS,
                  preregistered_groups=reg['coverage_groups']))
        bindings = {**reg['producer_bindings'], **reg['input_bindings'], **reg['reference_hashes']}
        base.verify_bindings(root, bindings)
        ids = ordered['val']
        y = base.load_reference(root, parent, ids)
        candidate = base.predictions(contained(root, p['arm']['outputs']['predictions']), ids)
        original = base.predictions(contained(root, ref['outputs']['predictions']), ids)
        scores = scores_for(candidate, original)
        rows = target_rows(y, scores); totals = aggregate(rows)
        ci = paired_bootstrap(y, scores, groups.loc[ids, 'component_id'].to_numpy())
        diagnostics = group_rows(y, scores, groups, ids, manifest, parent['groups']['subgroup_columns'])
        base.verify_bindings(root, {**reg['source_bindings'], **bindings, **hashes})
        need(sha(registration_path) == registration_sha256, 'Registration changed during evaluation')
        report = dict(schema=SCHEMA, evaluated_at=datetime.now(timezone.utc).isoformat(),
                      registration_sha256=registration_sha256, protocol_sha256=reg['protocol_sha256'],
                      source_bindings=reg['source_bindings'], output_hashes=hashes,
                      reference_hashes=reg['reference_hashes'], reference_inference_contract=infer,
                      coverage_manifest_sha256=STAGE_SHA, studies=len(ids), targets=rows, aggregate=totals,
                      conditional_intervals=ci, groups=diagnostics, execution=execution,
                      evaluation_environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                      limits=LIMITS, decision='Root review only; no automatic adoption, blending, routing, follow-up or submission.')
        write_new(out / 'report.json', report)
        pd.json_normalize(rows).to_csv(out / 'targets.csv', index=False, mode='x')
        write_new(out / 'SUCCESS.json', dict(report_sha256=sha(out / 'report.json'), registration_sha256=registration_sha256))
        return dict(output=reg['output'], aggregate=totals, valid_bootstrap_draws=ci['valid'], invalid=ci['invalid'])
    except Exception as exc:
        write_new(out / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                  rule='Preserve claimed one-shot failure; root must review any new registration/version.'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for name in ['protocol', 'protocol-sha256', 'execution-plan']: prep.add_argument('--' + name, required=True)
    evaluate = sub.add_parser('run')
    for name in ['registration', 'registration-sha256']: evaluate.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = prepare(args.protocol, args.protocol_sha256, args.execution_plan) if args.action == 'prepare' else run(args.registration, args.registration_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
