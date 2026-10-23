"""Focused V2 evaluator checks using synthetic receipts and source only."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'tools/exp086_dinov2s_report_v2.py'
spec = importlib.util.spec_from_file_location('exp086_report_v2_test', REPORT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

PROTOCOL_SHA = '6' * 64
INITIAL_SHA = 'a' * 64
SUPERVISOR_SHA = '3' * 64
COMMAND = ['python', 'synthetic_v2_worker', '--phase', 'full']


def pilot_receipt(projected_seconds=5299.0):
    step_seconds = (projected_seconds - 300) / (1.3 * 3448)
    timings = [step_seconds] * 32
    steady = sum(timings[8:]) / 24
    return dict(
        schema='exp086-dinov2s-v2', status='PASS', phase='pilot',
        position_policy=r.POSITION_POLICY, v1_elapsed_seconds=83.235,
        protocol_sha256=PROTOCOL_SHA, initial_state_sha256=INITIAL_SHA,
        initial_asset_sha256=r.WEIGHT_SHA, updates=32, skips=0,
        training_ids_only=True, eval_train_studies=4, validation_rows=0,
        metric_access='disabled', checkpoints_saved=False, predictions_saved=False,
        warmup_steps=8, step_seconds=timings, steady_update_seconds=steady,
        projected_full_seconds=1.3 * (steady * 3448) + 300,
    )


def full_resource():
    return dict(
        schema='process-resource-receipt-v1', supervisor_sha256=SUPERVISOR_SHA,
        argv=COMMAND.copy(), status='COMPLETE', failure=None,
        child_exit_code=0, wrapper_exit_code=0, child_exit_code_before_cleanup=0,
        elapsed_seconds=5299.0, timeout_seconds=5300, retries=0,
        cleanup=dict(actions=[], errors=[], survivors=[]),
        child_identity=dict(pid=123, create_time=100.0),
        observed_identities=[dict(pid=123, create_time=100.0)],
        sample_count=100, sampled_tree_rss_max_bytes=100000,
        failed_sample_attempts=0, access_failure_count=0,
    )


def test_projection_below_v2_cap_passes_with_prior_failure_cost():
    result = r.pilot_gate(pilot_receipt(5299), 179.0,
                          {'arm': {'initial_state_sha256': INITIAL_SHA}}, PROTOCOL_SHA)
    assert result == pytest.approx(5299.0)
    assert 83.235 + result + 179.0 < 5580


def test_projection_above_v2_cap_fails_even_when_v1_would_allow_it():
    with pytest.raises(ValueError, match='Pilot throughput'):
        r.pilot_gate(pilot_receipt(5301), 100.0,
                     {'arm': {'initial_state_sha256': INITIAL_SHA}}, PROTOCOL_SHA)


@pytest.mark.parametrize(('field', 'value'), [
    ('schema', 'exp086-dinov2s-v1'),
    ('protocol_sha256', '7' * 64),
    ('initial_state_sha256', 'b' * 64),
    ('initial_asset_sha256', 'c' * 64),
    ('status', 'FAILED'),
])
def test_pilot_rejects_failed_or_mismatched_registered_identity(field, value):
    receipt = pilot_receipt()
    receipt[field] = value
    with pytest.raises(ValueError, match='T-only pilot completion'):
        r.pilot_gate(receipt, 100.0,
                     {'arm': {'initial_state_sha256': INITIAL_SHA}}, PROTOCOL_SHA)


def test_natural_full_completion_under_5300_passes():
    assert r.resource_gate(full_resource(), COMMAND, 5300, SUPERVISOR_SHA) == 5299.0


@pytest.mark.parametrize(('field', 'value'), [
    ('timeout_seconds', 5400),
    ('elapsed_seconds', 5300.0),
    ('status', 'FAILED'),
    ('failure', {'type': 'TimeoutExpired', 'message': 'synthetic'}),
    ('child_exit_code', 1),
    ('supervisor_sha256', '4' * 64),
])
def test_full_rejects_old_budget_failure_or_unbound_supervisor(field, value):
    receipt = full_resource()
    receipt[field] = value
    with pytest.raises(ValueError):
        r.resource_gate(receipt, COMMAND, 5300, SUPERVISOR_SHA)


@pytest.mark.parametrize('function_name', [
    'scores_for', 'target_rows', 'aggregate', 'paired_bootstrap',
])
def test_statistical_function_ast_is_unchanged_from_frozen_v1(function_name):
    def function_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        matches = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == function_name]
        assert len(matches) == 1
        return ast.dump(matches[0], include_attributes=False)

    assert function_ast(REPORT) == function_ast(ROOT / 'tools/exp086_dinov2s_report.py')


def test_fresh_import_is_passive_without_training_or_model_modules():
    script = textwrap.dedent('''
        import importlib.abc
        import importlib.util
        from pathlib import Path
        import sys

        report = Path(sys.argv[1]).resolve()
        root = report.parents[1]
        sys.dont_write_bytecode = True
        forbidden = ('torch', 'timm', 'training', 'raptor_', 'exp086_dinov2s')

        class RejectTrainingImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(forbidden):
                    raise AssertionError('Non-passive module import: ' + fullname)
                return None

        def audit(event, args):
            if event != 'open' or not isinstance(args[0], (str, bytes)):
                return
            path = Path(args[0]).resolve()
            if (path.is_relative_to(root / 'artifacts')
                    or path.is_relative_to(root / 'storage')
                    or path.suffix.lower() in ('.safetensors', '.pt', '.npy', '.npz', '.csv')):
                raise AssertionError('Non-passive data access: ' + str(path))

        sys.meta_path.insert(0, RejectTrainingImport())
        sys.addaudithook(audit)
        spec = importlib.util.spec_from_file_location('passive_report_v2', report)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert not any(name.startswith(forbidden) for name in sys.modules)
        assert module.SCHEMA == 'exp086-dinov2s-report-v2'
        print('PASSIVE_IMPORT_PASS')
    ''')
    result = subprocess.run([sys.executable, '-B', '-c', script, str(REPORT)],
                            cwd=REPORT.parent, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'PASSIVE_IMPORT_PASS'


def test_combined_gate_preserves_original_failure_time_and_new_caps():
    assert r.combined_budget_gate(83.235, 179.0, 5299.0) == pytest.approx(5561.235)
    for prior, pilot, full in [(0., 100., 5000.), (84., 100., 5000.), (83.235, 180., 5000.), (83.235, 100., 5300.)]:
        with pytest.raises(ValueError):
            r.combined_budget_gate(prior, pilot, full)


@pytest.mark.parametrize(('field', 'value'), [('position_policy', 'native-cuda'), ('v1_elapsed_seconds', 0.)])
def test_pilot_cannot_erase_operation_or_previous_failure(field, value):
    receipt = pilot_receipt()
    receipt[field] = value
    with pytest.raises(ValueError):
        r.pilot_gate(receipt, 100., {'arm': {'initial_state_sha256': INITIAL_SHA}}, PROTOCOL_SHA)
