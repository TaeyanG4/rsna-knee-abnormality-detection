"""EXP087 evaluator contracts using synthetic values and passive source only.

No prior metrics, bootstrap calls, real cohorts, pixel arrays, or model assets.
"""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'tools/exp087_coverage_report.py'
spec = importlib.util.spec_from_file_location('exp087_report_test', REPORT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

ARITHMETIC = ('scores_for', 'target_rows', 'aggregate', 'paired_bootstrap')
SUPERVISOR_SHA = '3' * 64
COMMAND = ['python', 'synthetic_worker', '--phase', 'full']


@pytest.mark.parametrize('function_name', ARITHMETIC)
def test_statistical_ast_matches_hash_bound_exp086_v2(function_name):
    def function_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == function_name]
        assert len(nodes) == 1
        return ast.dump(nodes[0], include_attributes=False)

    assert function_ast(REPORT) == function_ast(ROOT / r.PRIOR_REPORT)


def test_arithmetic_binding_checks_all_four_functions_without_evaluation():
    assert r.arithmetic_binding(ROOT) == {
        name: 'AST_IDENTICAL_TO_HASH_BOUND_EXP086_V2' for name in ARITHMETIC
    }


def synthetic_scores():
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 12))
    original = np.tile(np.array([.1, .2, .8, .9])[:, None], (1, 12))
    candidate = 1 - original
    return y, candidate, original


def test_raw_probabilities_and_negative_candidate_minus_B_direction():
    y, candidate, original = synthetic_scores()
    scores = r.scores_for(candidate, original)
    np.testing.assert_array_equal(scores['candidate'], candidate)
    np.testing.assert_array_equal(scores['B_original'], original)
    rows = r.target_rows(y, scores)
    contrast = 'primary_candidate_minus_B'
    for row in rows:
        assert row['P'] == row['N'] == 2
        assert row['auc'] == {'B_original': 1., 'candidate': 0.}
        assert row['delta'][contrast] == -1.
        assert row['delta_over_12'][contrast] == pytest.approx(-1 / 12)
    result = r.aggregate(rows)
    assert result['complete12'] is True
    assert result['auc'] == {'B_original': 1., 'candidate': 0.}
    assert result['contrasts'][contrast]['macro12'] == -1.


@pytest.mark.parametrize('arm', ['candidate', 'original'])
@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -.001, 1.001])
def test_non_probabilities_are_rejected(arm, bad_value):
    _, candidate, original = synthetic_scores()
    arrays = {'candidate': candidate, 'original': original}
    arrays[arm][0, 0] = bad_value
    with pytest.raises(ValueError, match='probabilities'):
        r.scores_for(arrays['candidate'], arrays['original'])


@pytest.mark.parametrize('shape', [(0, 12), (4, 11), (4,), (4, 12, 1)])
def test_probability_shape_contract(shape):
    with pytest.raises(ValueError):
        r.scores_for(np.zeros(shape), np.zeros(shape))


def test_single_class_subgroup_keeps_macro12_none():
    y, candidate, original = synthetic_scores()
    y[:, r.TARGETS.index("Baker's")] = 0
    rows = r.target_rows(y, r.scores_for(candidate, original))
    result = r.aggregate(rows)
    baker = next(row for row in rows if row['target'] == "Baker's")
    contrast = 'primary_candidate_minus_B'
    assert baker['P'] == 0 and baker['N'] == 4
    assert baker['auc'] == {'B_original': None, 'candidate': None}
    assert baker['delta'][contrast] is None
    assert result['complete12'] is False
    assert result['auc'] == {'B_original': None, 'candidate': None}
    assert result['contrasts'][contrast]['macro12'] is None
    assert result['contrasts'][contrast]["Baker's"] is None
    assert result['contrasts'][contrast]['Fracture'] == -1.


def synthetic_manifest():
    ids = [f'synthetic-val-{i:04}' for i in range(888)]
    train = [f'synthetic-train-{i:04}' for i in range(3455)]
    eligible = ids[::14][:63]
    entries = [dict(uid=uid, split='train', reason=None) for uid in train[:252]]
    entries += [dict(uid=uid, split='val', reason=None) for uid in eligible]
    for entry in entries:
        entry['array_path'] = 'synthetic-pixels-must-not-be-opened.npy'
    return ids, dict(
        status='STAGED_INPUTS_ONLY_NO_TRAINING', policy=r.POLICY,
        fixture_sha256='e' * 64, entries=entries,
        membership=dict(train=train, val=ids.copy()),
    )


def test_coverage_groups_are_disjoint_complete_and_follow_id_order():
    ids, manifest = synthetic_manifest()
    ids.reverse()
    groups = r.coverage_groups(ids, manifest)
    eligible_ids = {entry['uid'] for entry in manifest['entries'] if entry['split'] == 'val'}
    assert set(groups) == {'eligible63', 'other825'}
    assert groups['eligible63'].dtype == np.bool_
    assert groups['eligible63'].sum() == 63
    assert groups['other825'].sum() == 825
    np.testing.assert_array_equal(groups['eligible63'], [uid in eligible_ids for uid in ids])
    assert not (groups['eligible63'] & groups['other825']).any()
    assert (groups['eligible63'] | groups['other825']).all()


@pytest.mark.parametrize('case', [
    'short_ids', 'duplicate_ids', 'short_entries', 'duplicate_entries',
    'eligible_outside_val', 'train_overlaps_val', 'wrong_split',
    'fallback', 'empty_fallback', 'wrong_noneligible_membership',
])
def test_coverage_rejects_changed_membership_duplicates_or_fallback(case):
    ids, manifest = synthetic_manifest()
    entries = manifest['entries']
    if case == 'short_ids':
        ids.pop()
    elif case == 'duplicate_ids':
        ids[-1] = ids[0]
    elif case == 'short_entries':
        entries.pop()
    elif case == 'duplicate_entries':
        entries[-1] = deepcopy(entries[0])
    elif case == 'eligible_outside_val':
        entries[-1]['uid'] = 'synthetic-outside-val'
    elif case == 'train_overlaps_val':
        entries[0]['uid'] = ids[1]
    elif case == 'wrong_split':
        entries[0]['split'] = 'test'
    elif case == 'fallback':
        entries[0]['reason'] = 'no_valid_alt_triplet'
    elif case == 'empty_fallback':
        entries[0]['reason'] = ''
    else:
        # Counts and eligible63 remain correct: exact V membership must still hold.
        ids[1] = 'synthetic-unregistered-noneligible'
    with pytest.raises(ValueError):
        r.coverage_groups(ids, manifest)


def synthetic_resource():
    return dict(
        schema='process-resource-receipt-v1', supervisor_sha256=SUPERVISOR_SHA,
        argv=COMMAND.copy(), status='COMPLETE', failure=None,
        child_exit_code=0, wrapper_exit_code=0, child_exit_code_before_cleanup=0,
        elapsed_seconds=12.5, timeout_seconds=180, retries=0,
        cleanup=dict(actions=[], errors=[], survivors=[]),
        child_identity=dict(pid=123, create_time=100.),
        observed_identities=[dict(pid=123, create_time=100.)],
        sample_count=10, sampled_tree_rss_max_bytes=100000,
        failed_sample_attempts=0, access_failure_count=0,
    )


def test_natural_resource_completion_passes():
    assert r.resource_gate(synthetic_resource(), COMMAND, 180, SUPERVISOR_SHA) == 12.5


@pytest.mark.parametrize('field', [
    'child_exit_code', 'wrapper_exit_code', 'child_exit_code_before_cleanup',
    'retries', 'failed_sample_attempts', 'access_failure_count',
])
@pytest.mark.parametrize('bad_value', [False, 0.0, -1, 1])
def test_resource_zero_counters_require_integer_zero(field, bad_value):
    receipt = synthetic_resource()
    receipt[field] = bad_value
    with pytest.raises(ValueError):
        r.resource_gate(receipt, COMMAND, 180, SUPERVISOR_SHA)


@pytest.mark.parametrize('field', ['sample_count', 'sampled_tree_rss_max_bytes'])
@pytest.mark.parametrize('bad_value', [True, 0, -1, 1.5, float('inf'), float('nan')])
def test_resource_positive_counters_require_finite_positive_integers(field, bad_value):
    receipt = synthetic_resource()
    receipt[field] = bad_value
    with pytest.raises(ValueError):
        r.resource_gate(receipt, COMMAND, 180, SUPERVISOR_SHA)


@pytest.mark.parametrize('bad_value', [True, 0, -1, 180., float('inf'), float('nan')])
def test_resource_elapsed_is_finite_positive_and_strictly_under_budget(bad_value):
    receipt = synthetic_resource()
    receipt['elapsed_seconds'] = bad_value
    with pytest.raises(ValueError):
        r.resource_gate(receipt, COMMAND, 180, SUPERVISOR_SHA)


@pytest.mark.parametrize(('field', 'bad_value'), [
    ('status', 'FAILED'), ('failure', {'error': 'synthetic'}),
    ('supervisor_sha256', '4' * 64), ('argv', ['different-worker']),
    ('timeout_seconds', 181), ('observed_identities', []),
    ('cleanup', dict(actions=['terminate'], errors=[], survivors=[])),
    ('cleanup', dict(actions=[], errors=['synthetic'], survivors=[])),
    ('cleanup', dict(actions=[], errors=[], survivors=[123])),
])
def test_resource_rejects_unbound_failed_or_forced_cleanup(field, bad_value):
    receipt = synthetic_resource()
    receipt[field] = bad_value
    with pytest.raises(ValueError):
        r.resource_gate(receipt, COMMAND, 180, SUPERVISOR_SHA)


@pytest.fixture
def synthetic_stage(tmp_path, monkeypatch):
    def save(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    _, manifest = synthetic_manifest()
    manifest_path = save(r.STAGE_DIRECTORY + '/manifest.json', manifest)
    manifest_sha = r.sha(manifest_path)
    monkeypatch.setattr(r, 'STAGE_SHA', manifest_sha)
    receipt_path = save(r.STAGE_DIRECTORY + '/receipt.json', dict(
        status='CPU_STAGING_COMPLETE', manifest_sha256=manifest_sha,
        entries=manifest['entries'], reused8=8, new_planned307=307,
    ))
    supervisor_path = save(r.STAGE_DIRECTORY + '/supervisor.json', dict(
        status='PASS', returncode=0, killed=None, error=None,
        cleanup_killed_pids=[], cleanup_errors=[], wall_seconds=12.,
        peak_process_tree_rss=100000,
    ))
    job_path = save(r.STAGE_DIRECTORY + '/job.json', {'synthetic': True})
    acceptance_parent = Path(r.STAGE_ACCEPTANCE).parent
    independent_path = save(acceptance_parent / 'result-independent-review.json',
                            {'status': 'PASS_INPUT_STAGE_ACCEPTANCE_ONLY'})
    acceptance_path = save(r.STAGE_ACCEPTANCE, dict(
        schema='exp087-stage-root-acceptance-v1', status='PASS_INPUT_STAGE_ONLY',
        entries=315, T=252, V=63, independent_new_arrays=307, reused_pilot_arrays=8,
        manifest_sha256=manifest_sha, supervisor_status='PASS', child_exit=0,
        cleanup_errors=0, fallbacks=0, training_authorization=False,
        receipt_sha256=r.sha(receipt_path), supervisor_sha256=r.sha(supervisor_path),
        job_sha256=r.sha(job_path), fixture_sha256=manifest['fixture_sha256'],
        evidence_sha256={independent_path.name: r.sha(independent_path)},
    ))
    monkeypatch.setattr(r, 'STAGE_ACCEPTANCE_SHA', r.sha(acceptance_path))
    return tmp_path, manifest, manifest_path


def test_stage_accepts_synthetic_metadata_without_pixel_access(synthetic_stage, monkeypatch):
    root, manifest, _ = synthetic_stage

    def reject_pixels(*args, **kwargs):
        pytest.fail('Stage report must not load pixel arrays')

    monkeypatch.setattr(np, 'load', reject_pixels)
    bindings, result = r.stage_gate(root)
    assert result == manifest
    assert bindings[r.STAGE_ACCEPTANCE] == r.STAGE_ACCEPTANCE_SHA
    assert bindings[r.STAGE_DIRECTORY + '/manifest.json'] == r.STAGE_SHA


def test_stage_metadata_only_mode_does_not_parse_manifest(synthetic_stage, monkeypatch):
    root, _, manifest_path = synthetic_stage
    original_read = r.read

    def guarded_read(path):
        assert Path(path) != manifest_path, 'Manifest parsed in metadata-only mode'
        return original_read(path)

    monkeypatch.setattr(r, 'read', guarded_read)
    bindings, result = r.stage_gate(root, read_manifest=False)
    assert bindings and result is None


def test_stage_rejects_changed_bound_metadata(synthetic_stage):
    root, _, manifest_path = synthetic_stage
    manifest_path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        r.stage_gate(root)


def test_stage_rejects_failure_marker_even_with_valid_hashes(synthetic_stage):
    root, _, _ = synthetic_stage
    (root / r.STAGE_DIRECTORY / 'FAIL.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='failure marker'):
        r.stage_gate(root)


def test_fresh_import_is_passive_without_training_data_or_process_launch():
    script = textwrap.dedent('''
        import importlib.abc
        import importlib.util
        from pathlib import Path
        import sys

        report = Path(sys.argv[1]).resolve()
        root = report.parents[1]
        sys.dont_write_bytecode = True
        forbidden = ('torch', 'timm', 'training', 'raptor_', 'exp087_training',
                     'exp087_series_coverage', 'exp087_coverage_stage_v1', 'kaggle')

        class RejectActiveImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(forbidden):
                    raise AssertionError('Non-passive import: ' + fullname)
                return None

        def audit(event, args):
            if event in ('subprocess.Popen', 'os.system', 'socket.connect'):
                raise AssertionError('Non-passive action: ' + event)
            if event != 'open' or not isinstance(args[0], str):
                return
            path = Path(args[0]).resolve()
            if (any(path.is_relative_to(root / name)
                    for name in ('artifacts', 'storage', '.local'))
                    or path.suffix.lower() in
                    ('.safetensors', '.pt', '.npy', '.npz', '.csv', '.sqlite', '.db')):
                raise AssertionError('Non-passive data access: ' + str(path))

        sys.meta_path.insert(0, RejectActiveImport())
        sys.addaudithook(audit)
        spec = importlib.util.spec_from_file_location('passive_coverage_report', report)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert not any(name.startswith(forbidden) for name in sys.modules)
        assert module.SCHEMA == 'exp087-coverage-report-v1'
        print('PASSIVE_IMPORT_PASS')
    ''')
    result = subprocess.run([sys.executable, '-B', '-c', script, str(REPORT)],
                            cwd=REPORT.parent, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'PASSIVE_IMPORT_PASS'


def synthetic_supervisor():
    protocol_sha = '7' * 64
    cap, rss_cap = 180, 24 * 1024**3
    supervisor = dict(
        status='PASS', returncode=0, error=None, killed=None,
        cleanup_errors=[], cleanup_killed_pids=[], protocol_sha256=protocol_sha,
        wall_seconds=12.5, peak_tree_RSS=100000,
    )
    job = dict(argv=COMMAND.copy(), protocol_sha256=protocol_sha,
               pid=123, retry=0, seconds_cap=cap, rss_cap=rss_cap)
    return supervisor, job, {'worker_command': COMMAND.copy()}, protocol_sha, cap, rss_cap


def test_exp087_custom_supervisor_accepts_clean_bound_completion():
    assert r.supervisor_gate(*synthetic_supervisor()) == 12.5


@pytest.mark.parametrize(('owner', 'field', 'bad_value'), [
    ('supervisor', 'status', 'FAILED'), ('supervisor', 'returncode', False),
    ('supervisor', 'returncode', 1), ('supervisor', 'error', 'synthetic'),
    ('supervisor', 'killed', 'timeout'), ('supervisor', 'cleanup_errors', ['synthetic']),
    ('supervisor', 'cleanup_killed_pids', [123]),
    ('supervisor', 'protocol_sha256', '8' * 64),
    ('supervisor', 'wall_seconds', True), ('supervisor', 'wall_seconds', 180.),
    ('supervisor', 'wall_seconds', float('nan')),
    ('supervisor', 'peak_tree_RSS', True), ('supervisor', 'peak_tree_RSS', 0),
    ('supervisor', 'peak_tree_RSS', float('inf')),
    ('supervisor', 'peak_tree_RSS', 24 * 1024**3 + 1),
    ('job', 'argv', ['different-worker']), ('job', 'protocol_sha256', '8' * 64),
    ('job', 'pid', True), ('job', 'pid', 0), ('job', 'retry', False),
    ('job', 'retry', 1), ('job', 'seconds_cap', 181), ('job', 'rss_cap', 1),
])
def test_exp087_custom_supervisor_rejects_invalid_execution(owner, field, bad_value):
    args = synthetic_supervisor()
    args[0 if owner == 'supervisor' else 1][field] = bad_value
    with pytest.raises(ValueError):
        r.supervisor_gate(*args)


def synthetic_trace(mode='full'):
    updates, epochs = (32, 1) if mode == 'pilot' else (3448, 8)
    return dict(
        schema='exp087-delivered-input-trace-v1', updates=updates,
        scheduler_steps=updates, epochs=epochs, delivered_uid_sha256='9' * 64,
        eligible_deliveries=20, scheduled_replacements=80,
        actual_replacement_tensor_checks=12,
        worker_cleanup=dict(clean=True, pids=list(range(100, 112)), exitcodes=[0] * 12),
        pilot_whole_update_seconds=[.5] * 32,
    )


@pytest.mark.parametrize('mode', ['pilot', 'full'])
def test_delivered_trace_accepts_matching_update_horizon_and_clean_workers(mode):
    trace = synthetic_trace(mode)
    assert r.trace_gate(trace, mode) == trace


@pytest.mark.parametrize(('field', 'bad_value'), [
    ('schema', 'different-trace'), ('updates', 3447), ('updates', 3448.),
    ('scheduler_steps', 3447), ('epochs', 7), ('epochs', 8.),
    ('delivered_uid_sha256', 'not-a-hash'), ('eligible_deliveries', 0),
    ('eligible_deliveries', True), ('eligible_deliveries', 3448 * 8 + 1),
    ('scheduled_replacements', 0), ('scheduled_replacements', 1743),
    ('actual_replacement_tensor_checks', 0), ('actual_replacement_tensor_checks', 21),
])
def test_delivered_trace_rejects_missing_or_impossible_treatment(field, bad_value):
    trace = synthetic_trace()
    trace[field] = bad_value
    with pytest.raises(ValueError):
        r.trace_gate(trace, 'full')


@pytest.mark.parametrize('case', ['unclean', 'eleven', 'duplicate_pid', 'bool_pid', 'failed_exit', 'bool_exit'])
def test_delivered_trace_requires_all_twelve_distinct_workers_to_exit_cleanly(case):
    trace = synthetic_trace()
    workers = trace['worker_cleanup']
    if case == 'unclean':
        workers['clean'] = False
    elif case == 'eleven':
        workers['pids'].pop()
        workers['exitcodes'].pop()
    elif case == 'duplicate_pid':
        workers['pids'][-1] = workers['pids'][0]
    elif case == 'bool_pid':
        workers['pids'][0] = True
    elif case == 'failed_exit':
        workers['exitcodes'][0] = 1
    else:
        workers['exitcodes'][0] = False
    with pytest.raises(ValueError):
        r.trace_gate(trace, 'full')


def synthetic_pilot():
    trace = synthetic_trace('pilot')
    projection = 1.3 * sum(trace['pilot_whole_update_seconds'][8:]) / 24 * 3448 + 300
    receipt = dict(
        status='PILOT_COMPLETE_NO_FULL_AUTHORIZATION', updates=32, skips=0,
        scheduler_horizon=3448, projection_pass=True, no_checkpoint_saved=True,
        no_validation_inference=True, no_metric_access=True,
        projected_full_seconds=projection, peak_cuda_bytes=100000,
    )
    return receipt, trace


def test_T_only_pilot_recomputes_conservative_projection_from_last_24_updates():
    receipt, trace = synthetic_pilot()
    trace['pilot_whole_update_seconds'][:8] = [99.] * 8
    assert r.pilot_gate(receipt, trace) == pytest.approx(2541.2)


@pytest.mark.parametrize(('field', 'bad_value'), [
    ('status', 'COMPLETE'), ('updates', 31), ('skips', 1), ('scheduler_horizon', 32),
    ('projection_pass', False), ('no_checkpoint_saved', False),
    ('no_validation_inference', False), ('no_metric_access', False),
    ('projected_full_seconds', 1.), ('peak_cuda_bytes', True), ('peak_cuda_bytes', 0),
])
def test_pilot_rejects_changed_horizon_validation_access_or_fabricated_projection(field, bad_value):
    receipt, trace = synthetic_pilot()
    receipt[field] = bad_value
    with pytest.raises(ValueError):
        r.pilot_gate(receipt, trace)


@pytest.mark.parametrize('case', ['short', 'zero', 'nan', 'over_budget'])
def test_pilot_rejects_missing_nonfinite_or_over_budget_whole_update_times(case):
    receipt, trace = synthetic_pilot()
    times = trace['pilot_whole_update_seconds']
    if case == 'short':
        times.pop()
    elif case == 'zero':
        times[8] = 0.
    elif case == 'nan':
        times[8] = float('nan')
    else:
        times[:] = [1.] * 32
        receipt['projected_full_seconds'] = 1.3 * 3448 + 300
    with pytest.raises(ValueError):
        r.pilot_gate(receipt, trace)


def test_run_source_orders_completion_before_atomic_claim_before_outcome_parsing():
    tree = ast.parse(REPORT.read_text(encoding='utf-8'))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    completion = next(node.lineno for node in calls
                      if isinstance(node.func, ast.Name) and node.func.id == 'require_completed')
    claims = [node for node in calls if isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name)
              and node.func.value.id == 'out' and node.func.attr == 'mkdir']
    assert len(claims) == 1
    claim = claims[0]
    assert any(keyword.arg == 'exist_ok' and isinstance(keyword.value, ast.Constant)
               and keyword.value.value is False for keyword in claim.keywords)
    outcomes = [node.lineno for node in calls
                if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in ('load_reference', 'predictions'))
                or (isinstance(node.func, ast.Name)
                    and node.func.id in ('scores_for', 'target_rows', 'aggregate', 'paired_bootstrap'))]
    assert len(outcomes) >= 7
    assert completion < claim.lineno < min(outcomes)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    registration_sha = 'a' * 64
    ordered = {'train': ['synthetic-train'], 'val': ['synthetic-val']}
    parent = {'reference_counts': {}, 'expected_ids': {}}
    protocol = {'arm': {'outputs': {'predictions': 'synthetic-candidate.csv'}}}
    reg = dict(
        schema=r.SCHEMA, targets=r.TARGETS, scores=list(r.SCORES),
        contrasts={key: list(value) for key, value in r.CONTRASTS.items()},
        bootstrap=dict(reps=500, seed=20261004, unit='component_id', shared_complete12_draws=True),
        output=r.EVIDENCE + '/evaluation-v1',
        coverage_groups=dict(source=r.STAGE_DIRECTORY + '/manifest.json', sha256=r.STAGE_SHA,
                             rule='split=val staged metadata eligibility before outcome access',
                             eligible=63, other=825, subgroup_ci=False),
        source_bindings={}, producer_bindings={}, input_bindings={}, reference_hashes={},
        parent_protocol='synthetic-parent.json', protocol='synthetic-protocol.json',
        protocol_sha256='b' * 64, parent_protocol_sha256='c' * 64,
        inference_contract={}, reference_counts={}, expected_ids={},
        reference=dict(logs='synthetic-reference-logs', outputs=dict(
            summary='synthetic-summary.json', provenance='synthetic-provenance.json',
            predictions='synthetic-original.csv')),
    )
    out = tmp_path / reg['output']
    reads = {
        'synthetic-registration.json': reg, reg['parent_protocol']: parent,
        reg['protocol']: {}, 'synthetic-summary.json': {}, 'synthetic-provenance.json': {},
    }
    calls = {key: 0 for key in ['completion', 'stage', 'labels', 'predictions', 'statistics', 'bootstrap']}

    def metadata_read(path):
        return reads[Path(path).relative_to(tmp_path).as_posix()]

    def complete(*args):
        calls['completion'] += 1
        return {}, {}

    def stage(*args):
        calls['stage'] += 1
        return {}, {'membership': ordered}

    def reject_labels(*args):
        calls['labels'] += 1
        assert out.is_dir(), 'Outcome parser reached before atomic claim'
        raise RuntimeError('SYNTHETIC_LABEL_PARSE_FAILURE')

    def forbidden(name):
        def reject(*args, **kwargs):
            calls[name] += 1
            pytest.fail('Forbidden outcome/statistics call: ' + name)
        return reject

    monkeypatch.setattr(r, 'sha', lambda path: registration_sha)
    monkeypatch.setattr(r, 'read', metadata_read)
    monkeypatch.setattr(r, 'protocol_view', lambda raw, parent: protocol)
    monkeypatch.setattr(r, 'validate_protocol', lambda protocol, parent: {})
    monkeypatch.setattr(r, 'require_completed', complete)
    monkeypatch.setattr(r, 'stage_gate', stage)
    monkeypatch.setattr(r.base, 'verify_bindings', lambda *args: None)
    monkeypatch.setattr(r.base, 'cohort_metadata', lambda *args: (ordered, None))
    monkeypatch.setattr(r.base, 'validate_execution', lambda *args: None)
    monkeypatch.setattr(r.base, 'load_reference', reject_labels)
    monkeypatch.setattr(r.base, 'predictions', forbidden('predictions'))
    for name in ['scores_for', 'target_rows', 'aggregate', 'group_rows']:
        monkeypatch.setattr(r, name, forbidden('statistics'))
    monkeypatch.setattr(r, 'paired_bootstrap', forbidden('bootstrap'))
    return tmp_path, registration_sha, out, calls


def test_incomplete_phase_rejects_before_claim_or_any_outcome_loader(synthetic_run, monkeypatch):
    root, digest, out, calls = synthetic_run

    def incomplete(*args):
        calls['completion'] += 1
        raise ValueError('SYNTHETIC_PHASE_INCOMPLETE')

    monkeypatch.setattr(r, 'require_completed', incomplete)
    with pytest.raises(ValueError, match='SYNTHETIC_PHASE_INCOMPLETE'):
        r.run('synthetic-registration.json', digest, root=root)
    assert not out.exists()
    assert calls == dict(completion=1, stage=0, labels=0, predictions=0, statistics=0, bootstrap=0)


def test_existing_one_shot_path_rejects_before_completion_or_outcome_access(synthetic_run):
    root, digest, out, calls = synthetic_run
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match='One-shot'):
        r.run('synthetic-registration.json', digest, root=root)
    assert all(value == 0 for value in calls.values())


def test_failure_after_claim_is_recorded_and_second_attempt_stays_locked(synthetic_run):
    root, digest, out, calls = synthetic_run
    with pytest.raises(RuntimeError, match='SYNTHETIC_LABEL_PARSE_FAILURE'):
        r.run('synthetic-registration.json', digest, root=root)
    assert (out / 'analysis-plan.json').is_file()
    failure = json.loads((out / 'failure.json').read_text(encoding='utf-8'))
    assert failure['error_type'] == 'RuntimeError'
    assert failure['error'] == 'SYNTHETIC_LABEL_PARSE_FAILURE'
    assert not (out / 'SUCCESS.json').exists()
    before_retry = calls.copy()
    with pytest.raises(ValueError, match='One-shot'):
        r.run('synthetic-registration.json', digest, root=root)
    assert calls == before_retry
    assert calls == dict(completion=1, stage=1, labels=1, predictions=0, statistics=0, bootstrap=0)
