"""Frozen CPU-only EXP086 final evaluator; no training module imports.

The original EXP084 evaluator remains unchanged. Only its hash-bound passive
metadata and statistical primitives are reused for a new single-fit contract.
"""
from __future__ import annotations

import argparse
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
SCHEMA = 'exp086-dinov2s-report-v1'
EVIDENCE = 'docs/handoffs/evidence/2026-10/2026-10-04-exp086-report-tool'
BASE_PATH = 'tools/exp084_group_loss_report.py'
BASE_SHA = '9eaec5593ad42c44dc345836fa22ec01431f5ab8f1bd790ad5c62b75a721f0fb'
SCORES = ('B_original', 'candidate')
CONTRASTS = {'primary_candidate_minus_B': ('B_original', 'candidate')}
PRODUCER_SCHEMA = 'exp086-dinov2s-v1'
HELPER = 'tools/exp086_dinov2s.py'
RUNNER = 'training/runners/exp086_dinov2s.py'
ARCH = 'vit_small_patch14_dinov2.lvd142m'
TAG = 'EXP086_DINOV2S_G1'
WEIGHT_SHA = '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
ORIGINAL_ANALYSIS = 'docs/handoffs/evidence/2026-10/2026-10-04-exp084-report-tool/evaluation-v1/analysis-plan.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


need(sha(ROOT / BASE_PATH) == BASE_SHA, 'Frozen passive statistical adapter changed')
_spec = importlib.util.spec_from_file_location('exp086_base_report', ROOT / BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
TARGETS, UID = base.TARGETS, base.UID
read, contained, relative, write_new = base.read, base.contained, base.relative, base.write_new
LIMITS = base.LIMITS + [
    'The same weak development V was already observed in EXP084/085; one seed and one fixed fit.',
    'The contrast compares encoder-family transfer packages, including feature dimension and generic pretraining; it does not isolate pretraining alone or match parameter count/FLOPs.',
    'Generic external pretraining corpus patient overlap is not independently certified. No competition-adapted EXP042 or public teacher weights are permitted as initialization.',
    'Baker and Fracture losses, subgroup heterogeneity, EXP082 positive uncertain evidence and small RN complement must remain visible; no blend or routing search is performed here.',
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


def validate_protocol(p, parent):
    need(p['schema'] == PRODUCER_SCHEMA and p['targets'] == TARGETS and p['arm']['id'] == 'DINO'
         and p['arm']['tag'] == TAG, 'Wrong single-fit EXP086 contract')
    for key in ['paths', 'ordered_ids', 'groups', 'expected_ids', 'split', 'excluded_uids', 'loss',
                'reference_counts', 'macro12_valid', 'corpus_manifest_sha256', 'label_hash',
                'eval_label_hash', 'split_hash', 'training_code_sha256', 'expected_runtime', 'environment']:
        need(p[key] == parent[key], 'Original-B data/runtime contract differs: ' + key)
    original = parent['arms']['B']
    wanted = {**original['expected_args'], 'arch': ARCH, 'pretrained': 'local-generic-strict'}
    need(p['arm']['expected_args'] == wanted, 'Recipe changed beyond registered encoder and initial asset')
    need(p['arm']['loss_policy'] == original['loss_policy']
         and p['arm']['actual_pos_weight'] == original['actual_pos_weight']
         and p['arm']['expected_attempts'] == 3448 and hexhash(p['arm']['initial_state_sha256']),
         'Loss/update/initial-state contract differs')
    asset = p['initial_asset']
    need(asset['sha256'] == WEIGHT_SHA and asset['repo'] == 'timm/' + ARCH
         and asset['revision'] == '4610ca143709d58a633b6397a74412c2c3842454'
         and asset['bytes'] == 88240510, 'Wrong generic initialization asset')
    need(p['outputs'] == p['arm']['outputs']
         and set(p['outputs']) == {'summary', 'provenance', 'checkpoint', 'predictions', 'history', 'trace', 'receipt'},
         'Final output schema differs')
    expected_execution = dict(pilot_seconds=180, full_seconds=5400, total_seconds=5580,
                              pilot_updates=32, warmup_updates=8, pilot_train_eval_rows=4,
                              full_epochs=8, full_updates=3448, retries=0, all_skips_fail=True,
                              workers=12, internal_threads=1, eval_workers=2, same_training_schedule_in_pilot=True)
    need(all(p['execution'].get(key) == value for key, value in expected_execution.items()), 'Fixed cost/schedule gates differ')
    ev = p['evaluation']
    need(ev['primary'] == 'macro12(candidate)-macro12(EXP084_B_G1)'
         and ev['bootstrap_unit'] == 'component_id' and ev['bootstrap_reps'] == 500
         and ev['bootstrap_seed'] == 20261004 and ev['no_rank_or_blend'] is True,
         'Fixed raw primary/bootstrap contract differs')
    need(p['reference']['tag'] == original['tag'] and p['reference']['outputs'] == original['outputs'],
         'Original B reference output paths differ')
    infer = inference_contract(p)
    need(infer == inference_contract({**parent, 'arm': original}), 'Original B inference contract differs')
    return infer


def resource_gate(receipt, expected_argv, max_seconds, source_sha):
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


def pilot_gate(pilot, elapsed, p, protocol_sha):
    expected = dict(schema=PRODUCER_SCHEMA, status='PASS', phase='pilot', protocol_sha256=protocol_sha,
                    initial_state_sha256=p['arm']['initial_state_sha256'], initial_asset_sha256=WEIGHT_SHA,
                    updates=32, skips=0, training_ids_only=True, eval_train_studies=4, validation_rows=0,
                    metric_access='disabled', checkpoints_saved=False, predictions_saved=False, warmup_steps=8)
    need(all(pilot.get(key) == value for key, value in expected.items()), 'T-only pilot completion differs')
    times = pilot.get('step_seconds', [])
    need(len(times) == 32 and all(positive_number(value) for value in times), 'Incomplete measured pilot throughput')
    steady = sum(times[8:]) / 24
    projection = 1.3 * (steady * 3448) + 300
    need(positive_number(pilot.get('steady_update_seconds')) and positive_number(pilot.get('projected_full_seconds'))
         and math.isclose(pilot['steady_update_seconds'], steady, rel_tol=0, abs_tol=1e-12)
         and math.isclose(pilot['projected_full_seconds'], projection, rel_tol=0, abs_tol=1e-9)
         and projection <= 5400 and 0 < elapsed < 180 and projection + elapsed <= 5580,
         'Pilot throughput or conservative full projection failed')
    return projection


def worker_command(p, protocol, phase, root, python):
    argv = [python, str(contained(root, RUNNER)), '--protocol', str(contained(root, protocol)),
            '--phase', phase, '--tag', TAG + ('_PILOT' if phase == 'pilot' else '')]
    for key, value in {**p['arm']['expected_args'], **p['paths']}.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value: argv.append(flag)
        else:
            argv += [flag, str(contained(root, value)) if key in p['paths'] else str(value)]
    return argv


def command_value(argv, flag):
    need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), 'Missing/duplicate command flag: ' + flag)
    return argv[argv.index(flag) + 1]


def parse_plan(plan, p, protocol, protocol_sha, root):
    need(plan['protocol_sha256'] == protocol_sha, 'Execution-plan protocol differs')
    result = {}
    for phase in ['pilot', 'full']:
        argv = plan['phase_commands'][phase]['argv']
        need(argv.count('--') == 1 and len(argv) > argv.index('--') + 2, 'Invalid GPU-job command plan')
        tag = command_value(argv[:argv.index('--')], '--tag')
        command = argv[argv.index('--') + 1:]
        need(Path(argv[1]).resolve() == contained(root, 'tools/gpu_job.py').resolve()
             and command[0] == argv[0] and Path(command[1]).resolve() == contained(root, HELPER).resolve()
             and command[2] == 'run' and command_value(command, '--phase') == phase
             and Path(command_value(command, '--protocol')).resolve() == contained(root, protocol).resolve(),
             'Planned job does not invoke the registered phase')
        decision = relative(root, command_value(command, '--root-decision'))
        pilot = 'artifacts/experiments/' + TAG + '_PILOT'
        expected = [argv[0], str(contained(root, HELPER)), 'run', '--protocol', str(contained(root, protocol)),
                    '--phase', phase, '--root-decision', str(contained(root, decision))]
        if phase == 'full': expected += ['--pilot-dir', str(contained(root, pilot))]
        need(command == expected, 'Unexpected full/pilot options or changed command order')
        result[phase] = dict(tag=tag, logs='artifacts/logs/' + tag, command=command,
                             root_decision=decision, worker_command=worker_command(p, protocol, phase, root, argv[0]))
    return result


def validate_candidate(summary, provenance, history, p, protocol_sha):
    arm = p['arm']; args = arm['expected_args']; n = p['expected_ids']
    need(summary.get('provenance') == provenance and summary.get('history') == history, 'Summary provenance/history mismatch')
    expected = dict(code_sha256=p['training_code_sha256'], initial_state_sha256=arm['initial_state_sha256'],
                    train_ids_sha256=p['ordered_ids']['train_sha256'], eval_ids_sha256=p['ordered_ids']['val_sha256'],
                    train_ids_ordered_sha256=p['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=p['ordered_ids']['val_sha256'],
                    corpus_manifest_sha256=p['corpus_manifest_sha256'], checkpoint_selection='fixed-final',
                    sampling='sha256(seed,epoch,uid,stream)-v1', gold_access='disabled', loss_policy=arm['loss_policy'],
                    actual_pos_weight=arm['actual_pos_weight'], baseline_pos_weight=p['loss']['baseline_pos_weight'],
                    metric_access='deferred_until_final_candidate', protocol_sha256=protocol_sha,
                    adapter_sha256=p['source_bindings'][RUNNER], contract_helper_sha256=p['source_bindings'][HELPER],
                    phase='full', initial_asset_sha256=WEIGHT_SHA, requested_workers=12, effective_workers=12, internal_cpu_threads=1)
    need(all(provenance.get(key) == value for key, value in expected.items()), 'Candidate source/ancestry/input/runtime mismatch')
    need(all(provenance.get(key) == p['expected_runtime'][key] for key in ['runtime', 'eval_runtime']), 'Strict FP16 runtime differs')
    need(all(summary['environment'].get(key) == p['environment'][key]
             for key in ['torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'pydicom', 'cv2']), 'Runtime environment changed')
    need(all(summary['args'].get(key) == value for key, value in args.items())
         and summary['args'].get('phase') == 'full', 'Actual fixed training arguments changed')
    required = dict(gold=0, gold_access='disabled', metric_access='deferred_until_final_candidate', arm='DINO',
                    selected_checkpoint='best', best_epoch=7, selected_epoch_1based=8, max_epochs=8,
                    train=n['train'], val=n['val'], rows=n['val'], no_oof_epoch_selection=True,
                    weak_oof_macro_auc=None, oof_per_target=None, label_hash=p['label_hash'], eval_label_hash=p['eval_label_hash'],
                    split_hash=p['split_hash'], expected_attempts=3448, optimizer_updates=3448,
                    skipped_updates=0, optimizer_skip_fraction=0, backbone=ARCH, img=224)
    need(all(summary.get(key) == value for key, value in required.items()), 'Fixed-final/skip/metric completion failed')
    need(type(summary['optimizer_updates']) is int and type(summary['skipped_updates']) is int, 'Invalid update counter types')
    need(len(history) == 8, 'Incomplete final history')
    for ep, row in enumerate(history):
        need(row['epoch'] == ep and row['optimizer_updates'] == (ep + 1) * (n['train'] // 8)
             and row['skipped_updates'] == 0 and math.isfinite(row['loss'])
             and row.get('oof_auc') is None and row.get('gold_auc') is None and row['selection'] == float(ep),
             'Intermediate validation/Gold metric or update-budget drift')
    need(summary['results'] == {'best': {'oof_rows': n['val'], 'metric_access': 'deferred_until_final_candidate'}},
         'Unexpected selected/intermediate result')
    return dict(updates=3448, skips=0, epochs=8, fixed_final=True, initial_state_sha256=arm['initial_state_sha256'],
                comparison='Single generic encoder transfer package versus frozen original B8', gold_access='disabled',
                metrics_deferred=True, reported_training_seconds=summary['total_seconds'])


def job_gate(root, spec):
    job = contained(root, spec['logs'])
    need((job / 'SUCCESS').is_file() and (job / 'job.json').is_file() and not (job / 'FAILED').exists(),
         'Registered job did not complete successfully')
    need(re.fullmatch(r'exit 0 at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s*',
                      (job / 'SUCCESS').read_text(encoding='utf-8')) is not None, 'Invalid SUCCESS marker')
    record = read(job / 'job.json')
    need(record['tag'] == spec['tag'] and record['command'] == spec['command'], 'Actual job differs from frozen command')
    return [job / 'SUCCESS', job / 'job.json']


def require_completed(root, reg, p):
    """Completion, cleanup and artifact hashes are checked before outcome parsing."""
    out = contained(root, reg['phase_output'])
    pilot_out = contained(root, reg['pilot_output'])
    required = [contained(root, path) for path in p['outputs'].values()]
    required += [pilot_out / 'receipt.json', pilot_out / 'provenance.json',
                 contained(root, reg['pilot_resource']), contained(root, reg['full_resource'])]
    need(all(path.is_file() for path in required) and not (out / 'failure.json').exists()
         and not (pilot_out / 'failure.json').exists(), 'Complete final and pilot artifacts required')
    need(not list(out.glob('gold*')) and not (out / 'model_swa.pt').exists(), 'Unexpected Gold or alternate checkpoint')
    need(not list(pilot_out.glob('model*.pt')) and not list(pilot_out.glob('oof*.csv')),
         'Pilot must not retain checkpoint or predictions')
    for phase in ['pilot', 'full']:
        required += job_gate(root, reg['jobs'][phase])
        decision_path = contained(root, reg['jobs'][phase]['root_decision'])
        required.append(decision_path)
        decision = read(decision_path)
        need(decision.get('protocol_sha256') == reg['protocol_sha256'] and decision.get('allow_phase') == phase
             and decision.get('retries') == 0, 'Root phase decision missing or changed')
        if phase == 'full':
            need(decision.get('pilot_receipt_sha256') == sha(pilot_out / 'receipt.json'), 'Root full decision not bound to actual pilot')
    source_sha = p['source_bindings']['tools/process_resource_receipt.py']
    pilot_resource = read(contained(root, reg['pilot_resource']))
    full_resource = read(contained(root, reg['full_resource']))
    pilot_seconds = resource_gate(pilot_resource, reg['jobs']['pilot']['worker_command'], 180, source_sha)
    full_seconds = resource_gate(full_resource, reg['jobs']['full']['worker_command'], 5400, source_sha)
    need(pilot_seconds + full_seconds <= 5580, 'Combined active phase budget exceeded')
    pilot = read(pilot_out / 'receipt.json')
    projection = pilot_gate(pilot, pilot_seconds, p, reg['protocol_sha256'])
    need(pilot.get('argv') == reg['jobs']['pilot']['worker_command'], 'Pilot receipt command differs')
    pilot_prov = read(pilot_out / 'provenance.json')
    need(pilot_prov.get('phase') == 'pilot' and pilot_prov.get('protocol_sha256') == reg['protocol_sha256']
         and pilot_prov.get('initial_state_sha256') == p['arm']['initial_state_sha256']
         and pilot_prov.get('initial_asset_sha256') == WEIGHT_SHA
         and pilot_prov.get('gold_access') == 'disabled'
         and pilot_prov.get('runtime') == p['expected_runtime']['runtime'], 'Pilot provenance differs')
    receipt = read(out / 'receipt.json')
    wanted = dict(schema=PRODUCER_SCHEMA, status='PASS', phase='full', protocol_sha256=reg['protocol_sha256'],
                  initial_state_sha256=p['arm']['initial_state_sha256'], initial_asset_sha256=WEIGHT_SHA,
                  updates=3448, skips=0, metric_access='deferred_until_final_candidate', validation_rows=888,
                  argv=reg['jobs']['full']['worker_command'])
    need(all(receipt.get(key) == value for key, value in wanted.items())
         and positive_number(receipt.get('elapsed_seconds')) and receipt['elapsed_seconds'] < 5400
         and hexhash(receipt.get('training_trace_sha256')), 'Whole final receipt incomplete')
    artifacts = {Path(path).name: path for key, path in p['outputs'].items() if key != 'receipt'}
    need(set(receipt['outputs']) == set(artifacts), 'Final receipt inventory differs')
    need(all(sha(contained(root, artifacts[name])) == value for name, value in receipt['outputs'].items()),
         'Final artifact hash mismatch')
    trace = read(contained(root, p['outputs']['trace']))
    need(trace.get('schema') == PRODUCER_SCHEMA and trace.get('training_trace_sha256') == receipt['training_trace_sha256']
         and trace.get('seed') == 42 and trace.get('epochs') == 8 and trace.get('updates') == 3448
         and trace.get('train_ids_sha256') == p['ordered_ids']['train_sha256']
         and trace.get('val_ids_sha256') == p['ordered_ids']['val_sha256'], 'Training trace/source cohort binding differs')
    summary = read(contained(root, p['outputs']['summary']))
    provenance = read(contained(root, p['outputs']['provenance']))
    history = read(contained(root, p['outputs']['history']))
    execution = validate_candidate(summary, provenance, history, p, reg['protocol_sha256'])
    execution.update(pilot_supervised_seconds=pilot_seconds, full_supervised_seconds=full_seconds,
                     pilot_conservative_projection_seconds=projection, training_trace_sha256=receipt['training_trace_sha256'],
                     trace_scope='Producer-bound training order digest; no original-B trace equality claim')
    return execution, {relative(root, path): sha(path) for path in required}


def prepare(protocol, protocol_sha256, execution_plan, root=ROOT):
    root = Path(root)
    protocol_path = contained(root, protocol)
    need(sha(protocol_path) == protocol_sha256, 'Protocol SHA mismatch')
    p = read(protocol_path)
    parent_path = contained(root, p['parent_protocol']['path'])
    need(sha(parent_path) == p['parent_protocol']['sha256'], 'Original-B parent protocol changed')
    parent = read(parent_path)
    base.validate_protocol(parent)
    infer = validate_protocol(p, parent)
    ordered, _ = base.cohort_metadata(root, parent)
    probe = read(contained(root, p['cpu_probe']['path']))
    need(sha(contained(root, p['cpu_probe']['path'])) == p['cpu_probe']['sha256']
         and probe['schema'] == PRODUCER_SCHEMA and probe['status'] == 'CPU_INITIAL_FORWARD_FP32_GRADIENT_PASS'
         and probe['asset'] == p['initial_asset'] and probe['initial_state_sha256'] == p['arm']['initial_state_sha256']
         and probe['native_forward_exact'] is True and probe['head_forward_exact'] is True
         and probe['checkpointing_supported'] is True and probe['native_position_shape'] == [1, 1370, 384]
         and probe['runtime_input'] == 224 and probe['optimizer_steps'] == 0 and probe['GPU'] is False
         and all(v['finite'] and v['nonzero'] for v in probe['FP32_gradients'].values()), 'CPU initial/gradient probe gate failed')
    thread_path = 'docs/handoffs/evidence/2026-10/2026-10-04-exp086-dinov2s/initial-thread1-binding.json'
    thread = read(contained(root, thread_path))
    need(p['source_bindings'].get(thread_path) == sha(contained(root, thread_path))
         and thread.get('status') == 'PASS_SAME_INITIAL_STATE_THREADS1_AND12' and thread.get('threads') == 1
         and thread.get('initial_state_sha256') == p['arm']['initial_state_sha256']
         and thread.get('primary_probe_sha256') == p['cpu_probe']['sha256']
         and thread.get('constructor_sha256') == probe['model_constructor_sha256'], 'Actual CPU-thread initial binding differs')
    plan_path = contained(root, execution_plan)
    jobs = parse_plan(read(plan_path), p, relative(root, protocol_path), protocol_sha256, root)
    phase_output = 'artifacts/experiments/' + TAG
    pilot_output = phase_output + '_PILOT'
    need(relative(root, contained(root, p['outputs']['summary']).parent) == phase_output
         and not (contained(root, phase_output) / 'summary.json').exists()
         and not (contained(root, phase_output) / 'receipt.json').exists(), 'Freeze before final candidate outcomes exist')
    original_analysis = read(contained(root, ORIGINAL_ANALYSIS))
    reference = parent['arms']['B']
    reference_hashes = {path: original_analysis['output_hashes'][path] for path in reference['outputs'].values()}
    for name in ['SUCCESS', 'job.json']:
        path = reference['logs'] + '/' + name
        reference_hashes[path] = original_analysis['output_hashes'][path]
    need({key: reference_hashes[path] for key, path in reference['outputs'].items()} == p['reference']['hashes'],
         'Original B bytes differ from previously evaluated frozen reference')
    base.validate_execution(read(contained(root, reference['outputs']['summary'])),
                            read(contained(root, reference['outputs']['provenance'])), parent, 'B', p['parent_protocol']['sha256'])
    sources = {relative(root, protocol_path): protocol_sha256, **p['source_bindings'], BASE_PATH: BASE_SHA,
               ORIGINAL_ANALYSIS: sha(contained(root, ORIGINAL_ANALYSIS)), relative(root, plan_path): sha(plan_path),
               **{'training/src/' + key: value for key, value in p['training_code_sha256'].items()}}
    for name in ['tools/exp086_dinov2s_report.py', 'tests/test_exp086_dinov2s_report.py']:
        sources[name] = sha(contained(root, name))
    base.verify_bindings(root, sources)
    base.verify_bindings(root, p['input_bindings'])
    need(sha(Path(p['initial_asset']['path'])) == WEIGHT_SHA, 'Initial asset bytes changed')
    output = EVIDENCE + '/evaluation-v1'
    need(not contained(root, output).exists(), 'One-shot evaluation path already exists')
    reg = dict(schema=SCHEMA, prepared_at=datetime.now(timezone.utc).isoformat(), protocol=relative(root, protocol_path),
               protocol_sha256=protocol_sha256, source_bindings=sources, input_bindings=p['input_bindings'],
               initial_asset=p['initial_asset'], reference_hashes=reference_hashes, reference=reference,
               inference_contract=infer, phase_output=phase_output, pilot_output=pilot_output,
               full_resource=phase_output + '_RESOURCE/receipt.json', pilot_resource=pilot_output + '_RESOURCE/receipt.json',
               jobs=jobs, targets=TARGETS, scores=list(SCORES), contrasts=CONTRASTS,
               bootstrap=dict(reps=500, seed=20261004, unit='component_id', shared_complete12_draws=True),
               reference_counts=p['reference_counts'], expected_ids=p['expected_ids'], output=output, limits=LIMITS,
               scope='Metadata/byte-hash freeze only; no candidate/reference prediction or label values, model loading, GPU, Gold or evaluation.')
    destination = contained(root, EVIDENCE)
    destination.mkdir(parents=True, exist_ok=True)
    write_new(destination / 'registration.json', reg)
    return reg


def run(registration, registration_sha256, root=ROOT):
    root = Path(root)
    registration_path = contained(root, registration)
    need(sha(registration_path) == registration_sha256, 'Registration SHA mismatch')
    reg = read(registration_path)
    need(reg['schema'] == SCHEMA and reg['targets'] == TARGETS and reg['scores'] == list(SCORES)
         and reg['contrasts'] == {k: list(v) for k, v in CONTRASTS.items()}
         and reg['bootstrap'] == dict(reps=500, seed=20261004, unit='component_id', shared_complete12_draws=True)
         and reg['output'] == EVIDENCE + '/evaluation-v1', 'Analysis contract changed')
    out = contained(root, reg['output'])
    need(not out.exists(), 'One-shot output already exists')
    base.verify_bindings(root, reg['source_bindings'])
    p = read(contained(root, reg['protocol']))
    parent = read(contained(root, p['parent_protocol']['path']))
    infer = validate_protocol(p, parent)
    need(infer == reg['inference_contract'] and reg['reference_counts'] == p['reference_counts']
         and reg['expected_ids'] == p['expected_ids'], 'Registration/reference metadata changed')
    ordered, groups = base.cohort_metadata(root, parent)
    execution, hashes = require_completed(root, reg, p)
    base.verify_bindings(root, {**reg['input_bindings'], **reg['reference_hashes']})
    need(sha(Path(reg['initial_asset']['path'])) == WEIGHT_SHA, 'Generic initial asset changed')
    ref = reg['reference']
    need(not (contained(root, ref['logs']) / 'FAILED').exists(), 'Original B failed marker present')
    base.validate_execution(read(contained(root, ref['outputs']['summary'])),
                            read(contained(root, ref['outputs']['provenance'])), parent, 'B', p['parent_protocol']['sha256'])
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_new(out / 'analysis-plan.json', dict(registration_sha256=registration_sha256, output_hashes=hashes,
                                                 execution=execution, contrasts=CONTRASTS, limits=LIMITS))
        ids = ordered['val']
        y = base.load_reference(root, parent, ids)
        candidate = base.predictions(contained(root, p['outputs']['predictions']), ids)
        original = base.predictions(contained(root, ref['outputs']['predictions']), ids)
        scores = scores_for(candidate, original)
        rows = target_rows(y, scores)
        totals = aggregate(rows)
        ci = paired_bootstrap(y, scores, groups.loc[ids, 'component_id'].to_numpy())
        diagnostics = []
        selected = groups.loc[ids]
        for field in p['groups']['subgroup_columns']:
            values = selected[field].replace('', 'unknown').to_numpy(dtype=str)
            for value in sorted(set(values)):
                take = values == value
                rr = target_rows(y[take], {key: v[take] for key, v in scores.items()})
                diagnostics.append(dict(field=field, group=value, studies=int(take.sum()), targets=rr, aggregate=aggregate(rr), ci=None))
        base.verify_bindings(root, {**reg['source_bindings'], **reg['input_bindings'], **reg['reference_hashes'], **hashes})
        need(sha(registration_path) == registration_sha256, 'Registration changed during evaluation')
        report = dict(schema=SCHEMA, evaluated_at=datetime.now(timezone.utc).isoformat(), registration_sha256=registration_sha256,
                      protocol_sha256=reg['protocol_sha256'], source_bindings=reg['source_bindings'], output_hashes=hashes,
                      reference_hashes=reg['reference_hashes'], reference_inference_contract=infer, studies=len(ids),
                      targets=rows, aggregate=totals, conditional_intervals=ci, groups=diagnostics, execution=execution,
                      evaluation_environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                      limits=LIMITS, decision='Root review required; no automatic adoption, routing, blending, follow-up training or submission.')
        write_new(out / 'report.json', report)
        pd.json_normalize(rows).to_csv(out / 'targets.csv', index=False, mode='x')
        write_new(out / 'SUCCESS.json', dict(report_sha256=sha(out / 'report.json'), registration_sha256=registration_sha256))
        return dict(output=reg['output'], aggregate=totals, valid_bootstrap_draws=ci['valid'], invalid=ci['invalid'])
    except Exception as exc:
        write_new(out / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                                           rule='Preserve failed one-shot output; root reviews any new registration/version.'))
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
