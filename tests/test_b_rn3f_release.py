import ast
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release', ROOT / 'tools/b_rn3f_release.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def table(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def competition(tmp_path, n=3):
    p = tmp_path / 'competition'; p.mkdir(); (p / 'test_series').mkdir()
    ids = ['uid' + str(i) for i in range(n)]
    table(p / 'test.csv', [r.UID], [[u] for u in reversed(ids)])
    table(p / 'sample_submission.csv', [r.UID, *r.TARGETS], [[u, *([.5]*12)] for u in ids])
    table(p / 'test_series.csv', [r.UID, 'SeriesInstanceUID', 'Anatomical_Plane'], [[ids[0], 'series', 'Sagittal']])
    return p, ids


@pytest.mark.parametrize('n', [1, 2, 3, 100, 131])
def test_dynamic_test_only_contract_uses_official_sample_order(tmp_path, n):
    p, ids = competition(tmp_path, n)
    c = r.test_contract(p)
    assert c['ids'] == ids and c['test_ids'] == list(reversed(ids))
    assert not (p / 'train.csv').exists() and not (p / 'train_series').exists()
    assert set(c['metadata_sha256']) == {'test.csv', 'test_series.csv', 'sample_submission.csv'}


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'extra', 'schema', 'width', 'unsafe', 'empty', 'duplicate_header'])
def test_invalid_dynamic_inputs_fail(tmp_path, bad):
    p, ids = competition(tmp_path)
    header, rows = r.csv_table(p / 'sample_submission.csv')
    if bad == 'duplicate': rows[1][0] = rows[0][0]
    if bad == 'missing': rows.pop()
    if bad == 'extra': rows.append(['extra', *([.5]*12)])
    if bad == 'schema': header[1], header[2] = header[2], header[1]
    if bad == 'width': rows[0].pop()
    if bad == 'unsafe': rows[0][0] = '../escape'
    if bad == 'empty': rows = []
    if bad == 'duplicate_header': header[1] = header[2]
    table(p / 'sample_submission.csv', header, rows)
    with pytest.raises(ValueError): r.test_contract(p)


def test_fixed_time_screen_boundary_and_distinct_reserves():
    threshold = math.floor((30600 - 300) / (1.3 * 877.354403408 / 100))
    assert r.operational_plan(threshold)['projected_seconds'] <= 30600
    with pytest.raises(ValueError): r.operational_plan(threshold + 1)
    assert [r.operational_plan(n)['internal_seconds'] for n in [1, 2, 3, 4]] == [600, 600, 600, 32100]
    assert r.TIME_POLICY['platform_seconds'] - r.TIME_POLICY['screen_max_seconds'] == 1800
    assert r.TIME_POLICY['platform_seconds'] - r.TIME_POLICY['full_internal_seconds'] == 300
    for n in [True, 0, -1, 3.0]:
        with pytest.raises(ValueError): r.operational_plan(n)


def resource():
    snapshot = {'errors': {}, 'location': {'version': 2}, 'values': {'memory.max': 30000, 'memory.peak': 3000,
        'memory.current': 2000, 'memory.events': {'max': 0, 'oom': 0, 'oom_kill': 0}}}
    return {'schema': 'process-resource-receipt-v1', 'supervisor_sha256': r.RESOURCE_SHA, 'status': 'COMPLETE',
            'child_exit_code': 0, 'wrapper_exit_code': 0, 'child_exit_code_before_cleanup': 0, 'failure': None,
            'argv': ['worker'], 'child_identity': {'pid': 99}, 'observed_identities': [{'pid': 99}],
            'sample_count': 10, 'sampled_tree_rss_max_bytes': 2000, 'failed_sample_attempts': 0,
            'access_failure_count': 0, 'access_failures': [], 'cleanup': {'actions': [], 'errors': [], 'survivors': []},
            'retries': 0, 'timeout_seconds': 570.0, 'elapsed_seconds': 20.0,
            'platform': 'linux', 'cgroup_telemetry_complete': True,
            'cgroup_start': copy.deepcopy(snapshot), 'cgroup_end': copy.deepcopy(snapshot)}


@pytest.mark.parametrize('bad', [None, 'status', 'child', 'identity', 'argv', 'cleanup', 'cleanup_missing',
                               'survivors', 'telemetry', 'access', 'late_exit0', 'oom', 'memory', 'helper'])
def test_resource_rejects_unsafe_complete_receipt(bad):
    v = resource()
    if bad == 'status': v['status'] = 'FAILED'
    if bad == 'child': v['child_exit_code'] = 1
    if bad == 'identity': v['observed_identities'] = []
    if bad == 'argv': v['argv'] = ['other']
    if bad == 'cleanup': v['cleanup']['actions'] = [{'pid': 99}]
    if bad == 'cleanup_missing': del v['cleanup']
    if bad == 'survivors': v['cleanup']['survivors'] = [99]
    if bad == 'telemetry': v['cgroup_telemetry_complete'] = False
    if bad == 'access': v['access_failure_count'] = 1
    if bad == 'late_exit0': v['elapsed_seconds'] = 570.0
    if bad == 'oom': v['cgroup_end']['values']['memory.events']['oom'] = 1
    if bad == 'memory': v['cgroup_end']['values']['memory.peak'] = 40000
    if bad == 'helper': v['supervisor_sha256'] = 'wrong'
    if bad:
        with pytest.raises(ValueError): r.validate_resource(v, ['worker'], 99, 570.0)
    else: r.validate_resource(v, ['worker'], 99, 570.0)


@pytest.mark.parametrize('n', [1, 3, 100, 131])
def test_frozen_full_cohort_aggregate_ties_does_not_shard_rank(n):
    p = r.load(ROOT / 'tools/b_rn3f_serving.py', 'test_frozen_rank', r.PRODUCER_SHA)
    ids = ['u' + str(i) for i in range(n)]
    raw = np.random.default_rng(7).integers(0, 5, (6, n, 12)).astype(np.float32) / 4
    actual = p.aggregate(raw, ids, ids, p.POLICY['member_order'])
    ranked = np.empty(raw.shape, np.float64)
    for m in range(6):
        for i in range(n):
            ranked[m, i] = ((raw[m] < raw[m, i]).sum(0) + ((raw[m] == raw[m, i]).sum(0) + 1) / 2) / n
    expected = np.stack([.9 * ranked[2*f] + .1 * ranked[2*f+1] for f in range(3)]).mean(0)
    assert np.array_equal(actual, expected)
    with pytest.raises(ValueError): p.aggregate(raw[:5], ids, ids, p.POLICY['member_order'][:5])


def publication_setup(tmp_path, monkeypatch):
    work = tmp_path / 'work'; (work / 'result').mkdir(parents=True); (work / 'resource').mkdir()
    (work / 'result/predictions.csv').write_text('test-final-csv\n', encoding='utf-8')
    r.write_new(work / 'launch.json', {})
    r.write_new(work / 'result/receipt.json', {'worker': 99})
    r.write_new(work / 'resource/receipt.json', resource())
    r.write_new(work / 'codec.json', {'codec': 'fixture'})
    r.write_new(work / 'parent-receipt.json', {'schema': r.SCHEMA, 'status': 'PARENT_COMPLETE_RESOURCE_CLEAN',
                'worker_exit_code': 0, 'supervisor_exit_code': 0, 'worker_sha256': r.sha(work / 'result/receipt.json'),
                'resource_sha256': r.sha(work / 'resource/receipt.json'), 'launch_sha256': r.sha(work / 'launch.json')})
    context = {'worker_argv': ['worker'], 'worker_timeout_seconds': 570.0, 'competition': 'test-only',
               'test_contract': {'ids': ['x'], 'studies': 1}, 'operational_plan': {'internal_seconds': 600},
               'deadline_monotonic': time.monotonic() + 600, 'started_monotonic': time.monotonic() - 30}
    receipt = {'runtime': {'pid': 99}, 'output_sha256': {'predictions.csv': r.sha(work / 'result/predictions.csv')}}
    monkeypatch.setattr(r, 'checked_outputs', lambda *a: receipt)
    monkeypatch.setattr(r, 'test_contract', lambda *a: context['test_contract'])
    monkeypatch.setattr(r, 'producer', lambda *a: SimpleNamespace(validate_frame=lambda *a: None))
    return work, context, tmp_path / 'submission.csv'


@pytest.mark.parametrize('bad', [None, 'parent', 'late_elapsed', 'late_clock', 'late_rename', 'existing', 'rename', 'resource'])
def test_atomic_final_after_all_gates_only(tmp_path, monkeypatch, bad):
    work, context, out = publication_setup(tmp_path, monkeypatch)
    if bad == 'parent': (work / 'result/receipt.json').write_text('changed', encoding='utf-8')
    if bad == 'late_clock': context['deadline_monotonic'] = time.monotonic() - 1
    if bad == 'existing': out.write_text('keep original', encoding='utf-8')
    if bad == 'resource':
        def failed(*a): raise ValueError('cleanup failed')
        monkeypatch.setattr(r, 'validate_resource', failed)
    if bad == 'rename':
        def failed(*a): raise OSError('rename failed')
        monkeypatch.setattr(r.os, 'replace', failed)
    if bad == 'late_rename':
        replace = r.os.replace
        def late(src, dst):
            replace(src, dst); context['deadline_monotonic'] = time.monotonic() - 1
        monkeypatch.setattr(r.os, 'replace', late)
    if bad:
        with pytest.raises((ValueError, OSError)):
            r.publish_final(work, tmp_path, context, 'manifest', out, 600.0 if bad == 'late_elapsed' else 30.0)
        if bad == 'existing': assert out.read_text() == 'keep original'
        else: assert not out.exists()
    else:
        result = r.publish_final(work, tmp_path, context, 'manifest', out, 30.)
        assert result['status'] == 'PASS_TEST_COHORT_ARTIFACT_ONLY'
        assert result['submission_sha256'] == r.sha(out) and not out.with_name('submission.csv.pending').exists()
        assert 30 <= result['elapsed_seconds'] <= result['post_ack_elapsed_seconds'] < 600
        assert r.read(work / 'result.json')['elapsed_seconds'] == result['elapsed_seconds']
        assert result['post_ack_deadline_pass']


def test_codec_receipt_requires_actual_default_pixel_and_lossless_pair():
    v = {'schema': 'B_RN3F_FINAL_CODEC_V1', 'status': 'PASS', 'pydicom_version': '3.0.2', 'gdcm_version': '3.2.6',
         'decoders': [{'uid': uid, 'available': True} for uid in r.SYNTAXES],
         'fixtures': [{'file': n, 'decode': 'SUCCESS', 'default_equals_fixed_reference': True,
                       'file_sha256': r.FIXTURE_HASHES[n], 'matches_uncompressed_fixture': True}
                      for n in list(r.FIXTURE_HASHES)[:3]]}
    r.validate_codec(v)
    for key in ['default_equals_fixed_reference', 'matches_uncompressed_fixture']:
        bad = copy.deepcopy(v); bad['fixtures'][1][key] = False
        with pytest.raises(ValueError): r.validate_codec(bad)


def test_build_safe_ast_and_no_runtime_model_import_at_module_top():
    source = (ROOT / 'tools/b_rn3f_release.py').read_text(encoding='utf-8')
    parsed = ast.parse(source)
    imports = [n for n in parsed.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(('torch' in ast.unparse(n) or 'raptor_' in ast.unparse(n)) for n in imports)
    assert 'setattr(p,' not in source and 'p.resolve_ids =' not in source


def test_mount_discovery_never_requires_train_data(tmp_path, monkeypatch):
    comp, _ = competition(tmp_path)
    native = Path
    def path(value):
        return comp if str(value).startswith('/kaggle/input/competitions/') else native(tmp_path / 'absent')
    monkeypatch.setattr(r, 'Path', path)
    assert r.find_test_competition() == comp


def test_failed_supervisor_saves_failure_without_final(tmp_path, monkeypatch):
    comp, _ = competition(tmp_path)
    work, out = tmp_path / 'work', tmp_path / 'submission.csv'
    monkeypatch.setattr(r, 'check_bundle', lambda *a: {})
    monkeypatch.setattr(r, 'producer', lambda *a: SimpleNamespace(validate_assets=lambda *a: {},
                        cpu_allocation=lambda *a: {'selected_threads': 4}))
    monkeypatch.setattr(r.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    args = SimpleNamespace(bundle=tmp_path, assets=tmp_path, work=work, competition=comp,
                           out=out, manifest_sha256='manifest', opencv_wheel='wheel', started_monotonic=None)
    result = r.run(args)
    assert result['status'] == 'FAILED' and not out.exists()
    assert r.read(work / 'result.json')['status'] == 'FAILED' and (work / 'evidence.zip').is_file()


@pytest.mark.parametrize('status,expected_exists', [('PASS_TEST_COHORT_ARTIFACT_ONLY', False), ('FAILED', True)])
def test_failed_acknowledgement_removes_only_new_success_artifact(tmp_path, monkeypatch, status, expected_exists):
    out = tmp_path / 'submission.csv'; out.write_bytes(b'owned result')
    def failed(*a): raise OSError('receipt filesystem failure')
    monkeypatch.setattr(r, 'write_new', failed)
    with pytest.raises(OSError): r.finish_record(tmp_path, {'status': status, 'submission_sha256': r.sha(out)}, out)
    assert out.exists() is expected_exists


@pytest.mark.parametrize('failure', ['stat_after_rename', 'receipt_sha_after_rename', 'late_metadata', 'late_ack', 'ack_write'])
def test_post_rename_metadata_and_ack_failures_rollback_final(tmp_path, monkeypatch, failure):
    work, context, out = publication_setup(tmp_path, monkeypatch)
    original_stat, original_sha, original_write, original_replace = Path.stat, r.sha, r.write_new, r.os.replace
    exposed = False
    def replace(src, dst):
        nonlocal exposed
        original_replace(src, dst)
        if Path(dst) == out: exposed = True
        if failure == 'late_ack' and Path(dst) == work / 'result.json':
            context['deadline_monotonic'] = time.monotonic() - 1
    monkeypatch.setattr(r.os, 'replace', replace)
    if failure == 'stat_after_rename':
        def stat(path, *a, **k):
            nonlocal exposed
            if path == out and exposed:
                exposed = False
                raise OSError('post-rename stat failure')
            return original_stat(path, *a, **k)
        monkeypatch.setattr(Path, 'stat', stat)
    if failure in ['receipt_sha_after_rename', 'late_metadata']:
        def sha(path):
            if Path(path) == work / 'result/receipt.json' and exposed:
                if failure == 'receipt_sha_after_rename': raise OSError('post-rename receipt hash failure')
                context['deadline_monotonic'] = time.monotonic() - 1
            return original_sha(path)
        monkeypatch.setattr(r, 'sha', sha)
    if failure == 'ack_write':
        def write(path, value):
            if Path(path) == work / 'result.json.pending': raise OSError('final acknowledgement write failure')
            return original_write(path, value)
        monkeypatch.setattr(r, 'write_new', write)
    with pytest.raises((ValueError, OSError)):
        r.publish_final(work, tmp_path, context, 'manifest', out, 30.)
    assert not out.exists() and not (work / 'result.json').exists()


def test_external_changed_final_preserved_when_ack_fails(tmp_path, monkeypatch):
    work, context, out = publication_setup(tmp_path, monkeypatch)
    original_write = r.write_new
    def write(path, value):
        if Path(path) == work / 'result.json.pending':
            out.write_bytes(b'external replacement')
            raise OSError('receipt filesystem failure after external change')
        return original_write(path, value)
    monkeypatch.setattr(r, 'write_new', write)
    with pytest.raises(OSError): r.publish_final(work, tmp_path, context, 'manifest', out, 30.)
    assert out.read_bytes() == b'external replacement'
