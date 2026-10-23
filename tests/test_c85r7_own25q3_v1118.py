"""Regression tests for the binding cell-zero fallback failure; CPU only."""
import copy
import os
from unittest.mock import patch

import pytest
from tools import c85r7_own25q3_v1118 as b
from tools.c85r7_own25q3_cpu_fixtures import case_specs, run_case


@pytest.mark.parametrize('name,kwargs', case_specs(), ids=[name for name, _ in case_specs()])
def test_all_review_and_write_failure_cases(tmp_path, name, kwargs):
    result = run_case(tmp_path, **kwargs)
    assert result['status'] == 'PASS' and result['error'] is None


@pytest.mark.parametrize('marker', ['0', 'False', '1', ' '])
def test_rerun_marker_never_touches_filesystem(marker):
    source = b.source()
    with patch.dict(os.environ, {'KAGGLE_IS_COMPETITION_RERUN': marker}), \
         patch('pathlib.Path', side_effect=AssertionError('filesystem touched')):
        ns = {}
        exec(source, ns)
    assert ns['FAST'] is False


def test_frozen_scope_and_pins():
    assert b.validate()['runtime_files_identical'] == 29
    nb, meta = b.payload()
    parent, parent_meta = b.parent()
    assert nb['cells'][1:] == parent['cells'][1:]
    assert meta['id'] == 'taeyangg4/knee-lb-own25q3-v1'
    assert meta['title'] == 'knee-lb-own25q3-v1'
    assert {k:v for k,v in meta.items() if k not in ['id','title']} == {
        k:v for k,v in parent_meta.items() if k not in ['id','title']}


@pytest.mark.parametrize('mutation', ['cell0', 'runtime', 'markdown', 'extra_cell', 'metadata'])
def test_scope_rejects_unregistered_changes(mutation):
    nb, meta = b.payload()
    if mutation == 'metadata':
        meta['enable_gpu'] = False
    elif mutation == 'extra_cell':
        nb['cells'].append(copy.deepcopy(nb['cells'][0]))
    else:
        nb['cells'][{'cell0':0, 'runtime':2, 'markdown':1}[mutation]]['source'].append('\n# unauthorized\n')
    with pytest.raises(ValueError):
        b.scope(nb, meta)


def test_partial_parent_rejected(tmp_path, monkeypatch):
    (tmp_path / 'build-receipt.json').write_text('{}')
    monkeypatch.setattr(b, 'BASE', tmp_path)
    with pytest.raises(ValueError):
        b.parent()


@pytest.mark.parametrize('state', ['missing', 'stale', 'failed'])
def test_admit_fails_closed(tmp_path, monkeypatch, state):
    monkeypatch.setattr(b, 'SCRATCH', tmp_path)
    if state != 'missing':
        (tmp_path / 'smoke.json').write_bytes(b.old.encoded(dict(status='FAIL' if state == 'failed' else 'PASS', notebook_sha256='stale')))
    with pytest.raises((FileNotFoundError, ValueError)):
        b.admit()
