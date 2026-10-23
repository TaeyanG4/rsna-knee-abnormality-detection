"""Synthetic EXP086 evaluator checks; no model, real labels or predictions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('exp086_report', ROOT / 'tools/exp086_dinov2s_report.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def sample():
    y = np.tile(np.array([0, 1, 0, 1, 0, 1], np.int8)[:, None], (1, 12))
    good = .1 + .8 * y
    return y, good


def test_raw_single_contrast_keeps_losses_and_contributions_visible(monkeypatch):
    y, original = sample()
    candidate = original.copy()
    candidate[:, r.TARGETS.index("Baker's")] = 1 - original[:, 0]
    candidate[:, r.TARGETS.index('Fracture')] = .5
    monkeypatch.setattr(r.pd.DataFrame, 'rank', lambda *a, **k: pytest.fail('No rank or blend permitted'))
    scores = r.scores_for(candidate, original)
    rows = r.target_rows(y, scores)
    a = r.aggregate(rows)
    assert set(scores) == {'candidate', 'B_original'}
    np.testing.assert_array_equal(scores['candidate'], candidate)
    assert a['contrasts']['primary_candidate_minus_B']["Baker's"] == -1
    assert a['contrasts']['primary_candidate_minus_B']['Fracture'] == -.5
    assert a['contrasts']['primary_candidate_minus_B']['macro12'] == -.125
    assert sum(row['delta_over_12']['primary_candidate_minus_B'] for row in rows) == -.125


def test_component_shared_draws_are_conditional_and_deterministic():
    y, prediction = sample()
    scores = r.scores_for(prediction, prediction)
    first = r.paired_bootstrap(y, scores, ['a', 'a', 'b', 'b', 'c', 'c'], reps=20)
    second = r.paired_bootstrap(y, scores, ['a', 'a', 'b', 'b', 'c', 'c'], reps=20)
    assert first == second and first['valid'] == 20
    assert first['component_count'] == 3 and first['ranks_applied'] is False
    assert first['intervals']['primary_candidate_minus_B']['macro12'] == [0., 0.]


def test_missing_one_class_prevents_renormalizing_macro_to11():
    y, prediction = sample()
    y[:, 0] = 0
    scores = r.scores_for(prediction, prediction)
    aggregate = r.aggregate(r.target_rows(y, scores))
    result = r.paired_bootstrap(y, scores, ['a', 'a', 'b', 'b', 'c', 'c'], reps=10)
    assert aggregate['complete12'] is False
    assert aggregate['contrasts']['primary_candidate_minus_B']['macro12'] is None
    assert result['valid'] == 0 and result['invalid'] == 10
    assert result['intervals']['primary_candidate_minus_B']['macro12'] is None


@pytest.mark.parametrize('bad', ['nan', 'range', 'shape'])
def test_raw_prediction_contract_invalid(bad):
    _, original = sample()
    candidate = original.copy()
    if bad == 'nan': candidate[0, 0] = np.nan
    if bad == 'range': candidate[0, 0] = -0.1
    if bad == 'shape': candidate = candidate[:, :11]
    with pytest.raises(ValueError): r.scores_for(candidate, original)


def test_passive_import_has_no_torch_or_training_modules():
    code = "import runpy,sys;runpy.run_path('tools/exp086_dinov2s_report.py',run_name='passive');assert 'torch' not in sys.modules;assert 'raptor_train' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def metadata():
    runtime = {'requested_amp': 'fp16', 'effective_amp': 'fp16', 'policy': 'strict', 'device': 'cuda'}
    env = {key: 'same' for key in ['torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'pydicom', 'cv2']}
    p = {'expected_ids': {'train': 3455, 'val': 888},
         'arm': {'expected_args': {'res': 224, 'bs': 8}, 'initial_state_sha256': 'a' * 64,
                 'loss_policy': 'clipped_positive_term', 'actual_pos_weight': [2.] * 12},
         'training_code_sha256': {'raptor_train.py': 'b' * 64},
         'ordered_ids': {'train_sha256': 'c' * 64, 'val_sha256': 'd' * 64},
         'corpus_manifest_sha256': 'e' * 64, 'loss': {'baseline_pos_weight': [2.] * 12},
         'source_bindings': {r.RUNNER: '1' * 64, r.HELPER: '2' * 64, 'tools/process_resource_receipt.py': '3' * 64},
         'expected_runtime': {'runtime': runtime, 'eval_runtime': deepcopy(runtime)},
         'environment': env, 'label_hash': '4' * 64, 'eval_label_hash': '4' * 64, 'split_hash': '5' * 64}
    prov = dict(code_sha256=p['training_code_sha256'], initial_state_sha256='a' * 64,
                train_ids_sha256='c' * 64, eval_ids_sha256='d' * 64,
                train_ids_ordered_sha256='c' * 64, eval_ids_ordered_sha256='d' * 64,
                corpus_manifest_sha256='e' * 64, checkpoint_selection='fixed-final',
                sampling='sha256(seed,epoch,uid,stream)-v1', gold_access='disabled',
                loss_policy='clipped_positive_term', actual_pos_weight=[2.] * 12, baseline_pos_weight=[2.] * 12,
                metric_access='deferred_until_final_candidate', protocol_sha256='6' * 64,
                adapter_sha256='1' * 64, contract_helper_sha256='2' * 64, phase='full',
                initial_asset_sha256=r.WEIGHT_SHA, requested_workers=12, effective_workers=12, internal_cpu_threads=1,
                runtime=runtime, eval_runtime=deepcopy(runtime))
    history = [dict(epoch=ep, optimizer_updates=(ep + 1) * 431, skipped_updates=0, loss=.5,
                    oof_auc=None, gold_auc=None, selection=float(ep)) for ep in range(8)]
    summary = dict(provenance=prov, history=history, environment=env, args={'res': 224, 'bs': 8, 'phase': 'full'},
                   gold=0, gold_access='disabled', metric_access='deferred_until_final_candidate', arm='DINO',
                   selected_checkpoint='best', best_epoch=7, selected_epoch_1based=8, max_epochs=8,
                   train=3455, val=888, rows=888, no_oof_epoch_selection=True, weak_oof_macro_auc=None,
                   oof_per_target=None, label_hash='4' * 64, eval_label_hash='4' * 64, split_hash='5' * 64,
                   expected_attempts=3448, optimizer_updates=3448, skipped_updates=0, optimizer_skip_fraction=0,
                   backbone=r.ARCH, img=224, total_seconds=3000.,
                   results={'best': {'oof_rows': 888, 'metric_access': 'deferred_until_final_candidate'}})
    return p, summary, prov, history


@pytest.mark.parametrize('bad', ['skip', 'epoch', 'ancestor', 'uid', 'gold', 'precision', 'history_metric', 'argument'])
def test_candidate_rejects_contract_drift(bad):
    p, summary, prov, history = metadata()
    if bad == 'skip': summary['skipped_updates'] = 1
    if bad == 'epoch': summary['selected_epoch_1based'] = 7
    if bad == 'ancestor': prov['initial_asset_sha256'] = 'f' * 64
    if bad == 'uid': prov['eval_ids_ordered_sha256'] = 'f' * 64
    if bad == 'gold': prov['gold_access'] = 'enabled'
    if bad == 'precision': prov['runtime'] = {**prov['runtime'], 'effective_amp': 'bf16'}
    if bad == 'history_metric': history[2]['oof_auc'] = .9
    if bad == 'argument': summary['args']['res'] = 336
    with pytest.raises(ValueError): r.validate_candidate(summary, prov, history, p, '6' * 64)


def test_candidate_accepts_only_registered_final_fit():
    p, summary, prov, history = metadata()
    result = r.validate_candidate(summary, prov, history, p, '6' * 64)
    assert result['updates'] == 3448 and result['skips'] == 0 and result['epochs'] == 8


def resource(argv=None, full=False):
    return dict(schema='process-resource-receipt-v1', supervisor_sha256='3' * 64,
                argv=argv or ['python', 'registered'], status='COMPLETE', failure=None,
                child_exit_code=0, wrapper_exit_code=0, child_exit_code_before_cleanup=0,
                elapsed_seconds=3000. if full else 100., timeout_seconds=5400 if full else 180, retries=0,
                cleanup=dict(actions=[], errors=[], survivors=[]), child_identity={'pid': 123},
                observed_identities=[{'pid': 123}], sample_count=100, sampled_tree_rss_max_bytes=100000,
                failed_sample_attempts=0, access_failure_count=0)


@pytest.mark.parametrize('bad', ['exit', 'cleanup', 'nan', 'timeout', 'missing_identity', 'telemetry', 'command'])
def test_resource_failures_do_not_become_completion(bad):
    value = resource()
    if bad == 'exit': value['child_exit_code'] = 1
    if bad == 'cleanup': value['cleanup']['survivors'] = [{'pid': 456}]
    if bad == 'nan': value['elapsed_seconds'] = float('nan')
    if bad == 'timeout': value['timeout_seconds'] = 181
    if bad == 'missing_identity': value['observed_identities'] = []
    if bad == 'telemetry': value['access_failure_count'] = 1
    if bad == 'command': value['argv'] = ['python', 'other']
    with pytest.raises(ValueError): r.resource_gate(value, ['python', 'registered'], 180, '3' * 64)


def pilot():
    times = [.7] * 32
    steady = sum(times[8:]) / 24
    return dict(schema=r.PRODUCER_SCHEMA, status='PASS', phase='pilot', protocol_sha256='6' * 64,
                initial_state_sha256='a' * 64, initial_asset_sha256=r.WEIGHT_SHA,
                updates=32, skips=0, training_ids_only=True, eval_train_studies=4, validation_rows=0,
                metric_access='disabled', checkpoints_saved=False, predictions_saved=False, warmup_steps=8,
                step_seconds=times, steady_update_seconds=steady, projected_full_seconds=1.3 * (steady * 3448) + 300)


@pytest.mark.parametrize('bad', ['v_rows', 'skips', 'timing_nan', 'projection', 'speed_budget', 'checkpoint'])
def test_pilot_scope_and_cost_gate(bad):
    p, *_ = metadata(); value = pilot()
    if bad == 'v_rows': value['validation_rows'] = 4
    if bad == 'skips': value['skips'] = 1
    if bad == 'timing_nan': value['step_seconds'][12] = float('nan')
    if bad == 'projection': value['projected_full_seconds'] -= 100
    if bad == 'speed_budget':
        value['step_seconds'] = [2.] * 32; value['steady_update_seconds'] = 2.
        value['projected_full_seconds'] = 1.3 * (2. * 3448) + 300
    if bad == 'checkpoint': value['checkpoints_saved'] = True
    with pytest.raises(ValueError): r.pilot_gate(value, 100., p, '6' * 64)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def completed(tmp_path):
    p, summary, prov, history = metadata()
    p['outputs'] = {key: 'full/' + name for key, name in [('summary', 'summary.json'), ('provenance', 'provenance.json'),
                    ('checkpoint', 'model_best.pt'), ('predictions', 'oof_best.csv'), ('history', 'history.json'), ('trace', 'trace.json'), ('receipt', 'receipt.json')]}
    p['arm']['outputs'] = p['outputs']
    reg = dict(protocol_sha256='6' * 64, phase_output='full', pilot_output='pilot',
               pilot_resource='pilot_RESOURCE/receipt.json', full_resource='full_RESOURCE/receipt.json', jobs={})
    for key, value in [('summary', summary), ('provenance', prov), ('history', history)]: dump(tmp_path / p['outputs'][key], value)
    dump(tmp_path / p['outputs']['trace'], dict(schema=r.PRODUCER_SCHEMA, training_trace_sha256='7' * 64,
        seed=42, epochs=8, updates=3448, train_ids_sha256='c' * 64, val_ids_sha256='d' * 64))
    for key in ['checkpoint', 'predictions']:
        (tmp_path / p['outputs'][key]).write_bytes(b'synthetic bytes: not real model or outcome values')
    for phase in ['pilot', 'full']:
        reg['jobs'][phase] = dict(tag=phase, logs='logs/' + phase, root_decision=phase + '-decision.json',
                                  command=['python', 'wrapper', phase], worker_command=['python', 'worker', phase])
        dump(tmp_path / ('logs/' + phase + '/job.json'), {'tag': phase, 'command': reg['jobs'][phase]['command']})
        (tmp_path / ('logs/' + phase + '/SUCCESS')).write_text('exit 0 at 2026-10-04T06:00:00')
        dump(tmp_path / (phase + '_RESOURCE/receipt.json'), resource(reg['jobs'][phase]['worker_command'], full=phase == 'full'))
    value = pilot(); value['argv'] = reg['jobs']['pilot']['worker_command']; dump(tmp_path / 'pilot/receipt.json', value)
    pilot_prov = deepcopy(prov); pilot_prov['phase'] = 'pilot'; dump(tmp_path / 'pilot/provenance.json', pilot_prov)
    for phase in ['pilot', 'full']:
        decision = dict(protocol_sha256='6' * 64, allow_phase=phase, retries=0)
        if phase == 'full': decision['pilot_receipt_sha256'] = r.sha(tmp_path / 'pilot/receipt.json')
        dump(tmp_path / (phase + '-decision.json'), decision)
    value = dict(schema=r.PRODUCER_SCHEMA, status='PASS', phase='full', protocol_sha256='6' * 64,
                  initial_state_sha256='a' * 64, initial_asset_sha256=r.WEIGHT_SHA, updates=3448, skips=0,
                  metric_access='deferred_until_final_candidate', validation_rows=888, argv=reg['jobs']['full']['worker_command'],
                  elapsed_seconds=2990., training_trace_sha256='7' * 64,
                  outputs={Path(path).name: r.sha(tmp_path / path) for key, path in p['outputs'].items() if key != 'receipt'})
    dump(tmp_path / 'full/receipt.json', value)
    return tmp_path, reg, p


def test_completion_only_hashes_artifacts_without_parsing_outcomes(completed, monkeypatch):
    root, reg, p = completed
    monkeypatch.setattr(r.base, 'predictions', lambda *a: pytest.fail('Premature outcome access'))
    monkeypatch.setattr(r.base, 'load_reference', lambda *a: pytest.fail('Premature label access'))
    execution, hashes = r.require_completed(root, reg, p)
    assert execution['updates'] == 3448 and execution['skips'] == 0
    assert p['outputs']['checkpoint'] in hashes and execution['pilot_supervised_seconds'] == 100.


@pytest.mark.parametrize('bad', ['missing', 'failed_marker', 'stale_artifact', 'wrong_worker', 'decision', 'pilot_saved_model'])
def test_completion_mismatch_blocks_before_score_parsing(completed, monkeypatch, bad):
    root, reg, p = completed
    monkeypatch.setattr(r.base, 'predictions', lambda *a: pytest.fail('Premature outcome access'))
    monkeypatch.setattr(r.base, 'load_reference', lambda *a: pytest.fail('Premature label access'))
    if bad == 'missing': (root / p['outputs']['summary']).unlink()
    if bad == 'failed_marker': (root / 'logs/full/FAILED').write_text('exit 1')
    if bad == 'stale_artifact': (root / p['outputs']['predictions']).write_text('changed')
    if bad == 'wrong_worker':
        path = root / reg['full_resource']; value = r.read(path); value['argv'] = ['other']; dump(path, value)
    if bad == 'decision':
        path = root / 'full-decision.json'; value = r.read(path); value['pilot_receipt_sha256'] = 'f' * 64; dump(path, value)
    if bad == 'pilot_saved_model': (root / 'pilot/model_best.pt').write_text('not allowed')
    with pytest.raises(ValueError): r.require_completed(root, reg, p)


def plan_fixture(root):
    p = {'arm': {'expected_args': {'res': 224, 'grad_ckpt': True, 'smoke': False}},
         'paths': {'corpus': 'input/cache'}}
    protocol = 'protocol/manifest.json'
    plan = {'protocol_sha256': '6' * 64, 'phase_commands': {}}
    for phase in ['pilot', 'full']:
        command = [sys.executable, str(root / r.HELPER), 'run', '--protocol', str(root / protocol),
                   '--phase', phase, '--root-decision', str(root / (phase + '-decision.json'))]
        if phase == 'full': command += ['--pilot-dir', str(root / ('artifacts/experiments/' + r.TAG + '_PILOT'))]
        plan['phase_commands'][phase] = {'argv': [sys.executable, str(root / 'tools/gpu_job.py'), '--tag', phase, '--', *command]}
    return p, protocol, plan


def test_execution_plan_freezes_both_wrapper_and_worker_arguments(tmp_path):
    p, protocol, plan = plan_fixture(tmp_path)
    result = r.parse_plan(plan, p, protocol, '6' * 64, tmp_path)
    assert result['full']['command'] == plan['phase_commands']['full']['argv'][5:]
    worker = result['full']['worker_command']
    assert worker[worker.index('--tag') + 1] == r.TAG
    assert '--grad-ckpt' in worker and '--smoke' not in worker
    assert worker[worker.index('--corpus') + 1] == str(tmp_path / 'input/cache')


@pytest.mark.parametrize('bad', ['extra_arg', 'duplicate_phase', 'unregistered_runner', 'outside_decision'])
def test_command_plan_rejects_hidden_or_escaped_mutations(tmp_path, bad):
    p, protocol, plan = plan_fixture(tmp_path)
    argv = plan['phase_commands']['full']['argv']
    if bad == 'extra_arg': argv += ['--batch', '4']
    if bad == 'duplicate_phase': argv += ['--phase', 'pilot']
    if bad == 'unregistered_runner': argv[6] = str(tmp_path / 'other.py')
    if bad == 'outside_decision': argv[argv.index('--root-decision') + 1] = str(tmp_path.parent / 'outside.json')
    with pytest.raises(ValueError): r.parse_plan(plan, p, protocol, '6' * 64, tmp_path)
