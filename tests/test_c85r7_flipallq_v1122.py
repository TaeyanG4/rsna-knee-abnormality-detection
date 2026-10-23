"""One strict flipallq candidate; no model/GPU execution in unit tests."""
import ast
import copy
from unittest.mock import patch
import pytest
from tools import c85r7_flipallq_v1122 as b
from tools import c85r7_own25q3_cpu_fixtures as fixtures


def test_full_scope_and_exact_references():
    assert b.validate()['identical_to_flipall'] == 29
    nb, meta = b.payload()
    assert nb['cells'][0] == b.flip.read(b.fast.OUT / 'kernel/release.ipynb')['cells'][0]
    assert meta['id'] == b.SLUG and meta['title'] == 'knee-lb-flipallq-v1'
    files, blob = b.scope(nb, meta)
    parent = b.flip.parent()[1]
    assert sum(files[k] == parent[k] for k in files) == 26
    assert files['exp169_r7_port.py'] == b.flip.replace(parent['exp169_r7_port.py'], b.flip.FLAG_OLD, b.flip.FLAG_NEW)
    for name, data in files.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))


@pytest.mark.parametrize('mutation', ['cell0', 'runtime', 'markdown', 'extra_cell', 'metadata'])
def test_reject_unregistered_changes(mutation):
    nb, meta = b.payload()
    if mutation == 'metadata':
        meta['enable_internet'] = True
    elif mutation == 'extra_cell':
        nb['cells'].append(copy.deepcopy(nb['cells'][0]))
    else:
        nb['cells'][{'cell0':0, 'markdown':1, 'runtime':2}[mutation]]['source'].append('\n# drift\n')
    with pytest.raises(ValueError, match='forbidden'):
        b.scope(nb, meta)


@pytest.mark.parametrize('name,kwargs', fixtures.case_specs(), ids=[n for n,_ in fixtures.case_specs()])
def test_actual_candidate_fast_paths(tmp_path, name, kwargs):
    with patch.object(fixtures, 'b', b.fixture_builder()):
        assert fixtures.run_case(tmp_path, **kwargs)['status'] == 'PASS'


@pytest.mark.parametrize('state', ['missing', 'stale', 'failed'])
def test_admit_fails_closed(tmp_path, monkeypatch, state):
    monkeypatch.setattr(b, 'SCRATCH', tmp_path)
    if state != 'missing':
        (tmp_path / 'smoke.json').write_bytes(b.gate.encoded(dict(status='FAIL' if state == 'failed' else 'PASS', notebook_sha256='stale')))
    with pytest.raises((ValueError, FileNotFoundError)):
        b.admit()


def test_admit_does_not_waive_review():
    assert b.admit() == dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False)


@pytest.mark.parametrize('status,seconds,expected', [
    ('COMPLETE', 8.3*3600, True), ('COMPLETE', 9*3600, True),
    ('COMPLETE', 9*3600+1, False), ('TIMEOUT', 8*3600, False),
    ('ERROR', 8*3600, False), ('RUNNING', 1, False),
    ('COMPLETE', None, False), ('COMPLETE', float('nan'), False),
    ('COMPLETE', float('inf'), False), ('COMPLETE', -1, False),
    ('COMPLETE', True, False)])
def test_registered_runtime_decision(status, seconds, expected):
    assert b.runtime_admissible(status, seconds) is expected
    reading = b.registration()['public_reading']
    assert reading['purpose'] == 'runtime evidence only'
    assert reading['score_is_gain_test'] is False


def test_registration_tampering_rejected(monkeypatch):
    original = b.flip.read
    def altered(path):
        value = original(path)
        if path == b.OUT / 'registration.json':
            value['public_reading']['score_is_gain_test'] = True
        return value
    monkeypatch.setattr(b.flip, 'read', altered)
    with pytest.raises(ValueError, match='registration drift'):
        b.validate()


def test_real_pixel_candidate_receipt():
    report = b.flip.read(b.SCRATCH / 'flip-cpu-smoke.json')
    files, _ = b.scope(*b.payload())
    assert report['status'] == 'PASS' and report['device'] == 'cpu'
    assert report['studies'] == 3 and report['decoded_dicoms'] == 9
    assert len(report['folds']) == 4
    assert all(f['plain_mcl_exact'] and f['state_mcl_exact'] and f['non_mcl_changed'] > 0 for f in report['folds'])
    worker = next(k for k in files if k.endswith('/compact_infer.py'))
    assert report['worker_sha256'] == b.gate.sha(files[worker])
