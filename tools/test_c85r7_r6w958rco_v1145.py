"""Synthetic-only K3 formula/publication tests; no models, GPU or network."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from tools import c85r7_rco_overlay_v1 as o
from tools import c85r7_r6w958rco_v1145 as b


def payload(ids, values):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow([o.UID] + o.LABELS)
    writer.writerows([uid, *[format(float(x), '.9g') for x in row]] for uid, row in zip(ids, values))
    return stream.getvalue().encode()


def oracle(values):
    values = np.asarray(values, dtype=np.float64)
    return np.column_stack([((values[:, j, None] > values[None, :, j]).sum(1) +
        ((values[:, j, None] == values[None, :, j]).sum(1) + 1) / 2) / len(values) for j in range(12)])


@pytest.mark.parametrize('n', [1, 2, 3, 32, 976])
@pytest.mark.parametrize('ties', [False, True])
def test_full_roster_formula_independent_oracle(n, ties):
    rng = np.random.default_rng(193)
    a, c = rng.random((n, 12)).astype(np.float32), rng.random((n, 12)).astype(np.float32)
    if ties:
        a = np.round(a * 4) / 4
        c = np.round(c * 3) / 3
        a[:, 0] = .5
    ids = [str(i) for i in range(n)]
    order = rng.permutation(n)
    got = o.blend(payload(ids, a), payload([ids[i] for i in order], c[order]), ids)
    _, actual = o.table(got, ids)
    expected = oracle(.90 * oracle(a) + .10 * oracle(c)).astype(np.float32)
    assert np.array_equal(actual.astype(np.float32), expected)


def test_float32_serialization_boundary_not_float64_near_ties():
    ids = ['a', 'b', 'c']
    a = np.tile([.5, .5 + 1e-9, .5 + 2e-9], (12, 1)).T
    c = np.tile([.9, .2, .7], (12, 1)).T
    _, result = o.table(o.blend(payload(ids, a), payload(ids, c), ids))
    expected = oracle(.9 * oracle(a.astype(np.float32)) + .1 * oracle(c.astype(np.float32))).astype(np.float32)
    assert np.array_equal(result.astype(np.float32), expected)


def test_outer_rank_and_whole_roster_are_material():
    ids = ['a', 'b', 'c']
    a = np.full((3, 12), .5)
    c = np.tile([.1, .5, .9], (12, 1)).T
    _, got = o.table(o.blend(payload(ids, a), payload(ids, c), ids))
    assert np.array_equal(got[:, 0].astype(np.float32), np.asarray([1/3, 2/3, 1], np.float32))
    assert not np.allclose(got, .9 * oracle(a) + .1 * oracle(c))


@pytest.mark.parametrize('kind', ['missing', 'extra', 'duplicate', 'blank', 'target', 'width', 'nan', 'inf', 'negative', 'above_one'])
def test_table_rejects_invalid_shape_or_probabilities(kind):
    ids = ['a', 'b', 'c']
    value = payload(ids, np.full((3, 12), .5))
    rows = list(csv.reader(io.StringIO(value.decode())))
    if kind == 'missing': rows.pop()
    if kind == 'extra': rows.append(['d'] + ['.5'] * 12)
    if kind == 'duplicate': rows[2][0] = 'a'
    if kind == 'blank': rows[2][0] = ''
    if kind == 'target': rows[0][1:3] = rows[0][2:0:-1]
    if kind == 'width': rows[2].pop()
    if kind in ('nan', 'inf', 'negative', 'above_one'):
        rows[2][1] = {'nan': 'nan', 'inf': 'inf', 'negative': '-.1', 'above_one': '1.1'}[kind]
    stream = io.StringIO(newline=''); csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError): o.table(stream.getvalue().encode(), ids)


@pytest.fixture
def flow(tmp_path, monkeypatch):
    output, runtime, assets, data = [tmp_path / x for x in ('output', 'runtime', 'assets', 'data')]
    for folder in (output, runtime, assets, data): folder.mkdir()
    ids = ['a', 'b', 'c']
    a = payload(ids, np.full((3, 12), .5))
    c = payload(ids, np.tile([.9, .3, .7], (12, 1)).T)
    (data / 'sample_submission.csv').write_bytes(a)
    (assets / 'source').mkdir(); (assets / 'wheels').mkdir()
    worker = assets / 'source/l1_rco_k2_worker_v101.py'; worker.write_bytes(b'# fixture worker, never executed\n')
    wheel = assets / 'wheels/test.whl'; wheel.write_bytes(b'fixture wheel, never installed')
    manifest = dict(contract={'model_sha256': o.MODEL_SHA}, wheels={'test.whl': o.sha(wheel)})
    (assets / 'asset-manifest.json').write_bytes(o.encoded(manifest))
    monkeypatch.setattr(o, 'MANIFEST_SHA', o.sha(assets / 'asset-manifest.json'))
    monkeypatch.setattr(o, 'WORKER_SHA', o.sha(worker))
    rco = dict(status='COMPLETE', version='1.0.1-rcof0', kernel_id='taeyangg4/knee-l1-rcof0-v1',
        contract=manifest['contract'], manifest_sha256=o.MANIFEST_SHA, template_sha256=o.sha(data / 'sample_submission.csv'),
        output_sha256=hashlib.sha256(c).hexdigest(), study_count=3, prediction_count=3, fast_path=False,
        placeholder=False, fallbacks=dict(prediction=0, anchor=2), codecs=dict(status='ACTIVE'),
        studies=[dict(index=i, uid_sha256=hashlib.sha256(uid.encode()).hexdigest(), error=None, prediction_fallback=0)
                 for i, uid in enumerate(ids)])
    state = dict(ids=ids, a=a, c=c, output=output, runtime=runtime, assets=assets, data=data,
                 manifest=manifest, rco=rco, calls=[], fail=None)
    def runner(command, **kwargs):
        state['calls'].append(command)
        if 'parent-run.py' in command[-1]:
            if state['fail'] == 'parent': raise RuntimeError('fixture parent failure')
            (output / 'submission.csv').write_bytes(a)
            parent = dict(status='COMPLETE_C85R7', version=o.PARENT_VERSION,
                submission_sha256=hashlib.sha256(a).hexdigest(), studies=3, findings=12,
                fallbacks=dict(a172_prediction=0, r_prediction=0, s_r7_prediction=0, r224_prediction=0),
                codecs=dict(status='ACTIVE', s_r7_gate='ACTIVE'))
            if state['fail'] == 'parent_counter': parent['fallbacks']['r224_prediction'] = 1
            if state['fail'] == 'parent_hash': parent['submission_sha256'] = '0'*64
            (output / 'c85r7-receipt.json').write_bytes(o.encoded(parent))
            (output / 'c85r7-run/codecs').mkdir(parents=True)
            hook = output / 'hook'; hook.mkdir()
            (output / 'c85r7-run/codecs/install.json').write_bytes(o.encoded(dict(status='ACTIVE', hook_dir=str(hook))))
        elif 'pip' in command:
            assert (output / 'k3-rco-run/parent-submission.csv').read_bytes() == a
            if state['fail'] == 'pip': raise RuntimeError('fixture install failure')
        else:
            assert command[-2:] == ['--device', 'cuda:0']
            assert 'packages' in kwargs['env']['PYTHONPATH'] and 'hook' in kwargs['env']['PYTHONPATH']
            assert kwargs['env']['KNEE_CODEC_TELEMETRY'] == str(output / 'k3-rco-run/codec-telemetry')
            assert 'c85r7codec-runtime' not in kwargs['env']['PYTHONPATH']
            assert (output / 'submission.csv').read_bytes() == a
            if state['fail'] == 'worker': raise RuntimeError('fixture worker failure')
            job = output / 'k3-rco-run'
            if state['fail'] != 'missing_csv': (job / 'rco.csv').write_bytes(state['c'])
            (job / 'worker-receipt.json').write_bytes(o.encoded(state['rco']))
    state['runner'] = runner
    return state


def execute(f):
    return o.run(f['runtime'], f['output'], runner=f['runner'], assets=f['assets'], data=f['data'])


def test_actual_flow_publishes_distinct_runtime_receipt(flow):
    result = execute(flow)
    assert result['status'] == 'COMPLETE_K3' and result['parent_children_released']
    assert result['rco_fallbacks']['anchor'] == 2
    assert len(flow['calls']) == 3
    assert (flow['output'] / 'submission.csv').read_bytes() == o.blend(flow['a'], flow['c'], flow['ids'])
    assert (flow['output'] / 'k3-rco-run/parent-submission.csv').read_bytes() == flow['a']
    assert o.read(flow['output'] / 'c85r7-receipt.json')['version'] == o.PARENT_VERSION
    assert result['submission_sha256'] == o.sha(flow['output'] / 'submission.csv')
    assert result['qualified_for_submission'] is False
    # Actual generic receipt checks, no gate or network substitution.
    from tools import kaggle_submit_once as sender
    sender._prediction_evidence(result)


@pytest.mark.parametrize('failure', ['parent', 'parent_hash', 'parent_counter', 'pip', 'worker', 'missing_csv'])
def test_failures_preserve_parent_and_never_complete(flow, failure):
    flow['fail'] = failure
    if failure in ('parent', 'parent_hash'):
        with pytest.raises(ValueError if failure == 'parent_hash' else RuntimeError): execute(flow)
        assert o.read(flow['output'] / 'k3-receipt.json')['status'] == 'FAILED_PARENT'
    else:
        result = execute(flow)
        assert result['status'] == 'DEGRADED_PARENT_ONLY'
        assert (flow['output'] / 'submission.csv').read_bytes() == flow['a']
        assert result['executed_members'] == ['H189'] and result['qualified_for_submission'] is False


@pytest.mark.parametrize('field,value', [('status', 'FAILED'), ('version', 'old'), ('contract', {}),
    ('manifest_sha256', 'wrong'), ('template_sha256', 'wrong'), ('output_sha256', 'wrong'),
    ('study_count', 2), ('prediction_count', 2), ('fast_path', True), ('placeholder', True),
    ('fallbacks', {'prediction': 1}), ('codecs', {'status': 'ABSENT'}), ('studies', [])])
def test_rco_evidence_failures_disable_entire_overlay(flow, field, value):
    flow['rco'][field] = value
    result = execute(flow)
    assert result['status'] == 'DEGRADED_PARENT_ONLY'
    assert (flow['output'] / 'submission.csv').read_bytes() == flow['a']


def test_one_study_failure_disables_all(flow):
    flow['rco']['studies'][1]['error'] = 'failed'
    assert execute(flow)['status'] == 'DEGRADED_PARENT_ONLY'
    assert (flow['output'] / 'submission.csv').read_bytes() == flow['a']


@pytest.mark.parametrize('failure', ['duplicate', 'nonfinite', 'alignment', 'partial'])
def test_rco_csv_failures_at_actual_join(flow, failure):
    rows = list(csv.reader(io.StringIO(flow['c'].decode())))
    if failure == 'duplicate': rows[2][0] = 'a'
    if failure == 'nonfinite': rows[2][3] = 'nan'
    if failure == 'alignment': rows[2][0] = 'unknown'
    if failure == 'partial': rows.pop()
    stream = io.StringIO(newline=''); csv.writer(stream).writerows(rows)
    flow['c'] = stream.getvalue().encode()
    flow['rco']['output_sha256'] = hashlib.sha256(flow['c']).hexdigest()
    assert execute(flow)['status'] == 'DEGRADED_PARENT_ONLY'
    assert (flow['output'] / 'submission.csv').read_bytes() == flow['a']


@pytest.mark.parametrize('boundary', ['csv', 'receipt'])
def test_publication_failure_restores_exact_parent(flow, monkeypatch, boundary):
    original = o.publish
    failed = []
    def injected(path, payload):
        is_target = Path(path).name == ('submission.csv' if boundary == 'csv' else 'k3-receipt.json')
        if is_target and not failed:
            failed.append(True)
            if boundary == 'csv': original(path, payload)  # fail after replacement too
            raise OSError('injected publication failure')
        return original(path, payload)
    monkeypatch.setattr(o, 'publish', injected)
    assert execute(flow)['status'] == 'DEGRADED_PARENT_ONLY'
    assert (flow['output'] / 'submission.csv').read_bytes() == flow['a']
    assert o.read(flow['output'] / 'k3-receipt.json')['status'] == 'DEGRADED_PARENT_ONLY'


def test_stale_job_rejected_without_launch(flow):
    (flow['output'] / 'k3-rco-run').mkdir()
    with pytest.raises(ValueError, match='stale K3'): execute(flow)
    assert flow['calls'] == []


def test_frozen_parent_entire_archive_and_notebook_delta():
    parent, raw, pins, script = b.parent_material()
    assert len(pins) == 31 and hashlib.sha256(raw).hexdigest() == b.PARENT_BLOB_SHA
    assert script == b'FAST = False\n' + ''.join(parent['cells'][2]['source']).encode()
    notebook, metadata, inventory = b.package()
    assert inventory['parent_members'] == pins
    old = json.loads((b.PARENT / 'kernel-metadata.json').read_bytes())
    assert metadata['dataset_sources'] == old['dataset_sources'] + ['taeyangg4/knee-l1-rcof0-v1-assets']
    assert {k for k in metadata if old[k] != metadata[k]} == {'id', 'title', 'dataset_sources'}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), '<test notebook>', 'exec')


def test_child_exit_nonzero_is_not_success():
    with pytest.raises(ValueError, match='child failed'):
        o.run_child([sys.executable, '-B', '-c', 'raise SystemExit(4)'])


def test_child_success_cpu_only():
    o.run_child([sys.executable, '-B', '-c', 'pass'])


def test_owned_grandchild_stopped_before_next_phase(tmp_path):
    import psutil
    marker = tmp_path / 'owned-child.json'
    child = "import os,time,json,sys;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()}));time.sleep(30)"
    parent = ("import subprocess,sys,time;from pathlib import Path;"
              "p=subprocess.Popen([sys.executable,'-B','-c',sys.argv[1],sys.argv[2]]);"
              "\nwhile not Path(sys.argv[2]).exists(): time.sleep(.01)\n")
    o.run_child([sys.executable, '-B', '-c', parent, child, str(marker)])
    pid = json.loads(marker.read_text())['pid']
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
