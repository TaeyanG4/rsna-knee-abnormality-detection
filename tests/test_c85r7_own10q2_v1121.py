"""own10q2: exact candidate scope, both branches, and fail-closed admission."""
import copy
from unittest.mock import patch
import pytest
from tools import c85r7_own10q2_v1121 as b
from tools import c85r7_own25q3_cpu_fixtures as fixtures


def test_frozen_scope():
    assert b.validate()['identical_embedded'] == 29
    nb, meta = b.payload()
    assert nb['cells'][0] == b.own.read(b.fast.OUT / 'kernel/release.ipynb')['cells'][0]
    assert meta['title'] == 'knee-lb-own10q2-v1'


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
