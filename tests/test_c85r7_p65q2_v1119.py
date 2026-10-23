"""Authenticated visible-only fast path; all tests CPU-only."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
from tools import c85r7_p65q2_v1119 as b


from tools.c85r7_own25q3_cpu_fixtures import case_specs

@pytest.mark.parametrize('name,kwargs', case_specs(), ids=[name for name, _ in case_specs()])
def test_fixed_detection_and_write_failures(tmp_path, name, kwargs):
    result = b.run_case(tmp_path, **kwargs)
    assert result['status'] == 'PASS' and result['error'] is None


def test_frozen_scope():
    assert b.validate()['runtime_files_identical'] == 29
    nb, meta = b.payload()
    before, before_meta = b.parent()
    assert nb['cells'][1] == before['cells'][0]
    assert b.unguard(''.join(nb['cells'][2]['source'])) == ''.join(before['cells'][1]['source'])
    assert nb['cells'][0] == b.donor()['cells'][0]
    assert {k:v for k,v in meta.items() if k not in ['id', 'title']} == {
        k:v for k,v in before_meta.items() if k not in ['id', 'title']}
    assert meta['title'] == 'knee-lb-p65q2-v1'


@pytest.mark.parametrize('mutation', ['cell0', 'runtime', 'markdown', 'extra_cell', 'metadata'])
def test_scope_rejects_unregistered_changes(mutation):
    nb, meta = b.payload()
    if mutation == 'metadata': meta['enable_gpu'] = False
    elif mutation == 'extra_cell': nb['cells'].append(copy.deepcopy(nb['cells'][0]))
    else:
        index = {'cell0': 0, 'runtime': 2, 'markdown': 1}[mutation]
        nb['cells'][index]['source'].append('\n# unauthorized\n')
    with pytest.raises(ValueError): b.scope(nb, meta)


def test_partial_parent_rejected(tmp_path, monkeypatch):
    (tmp_path / 'build-receipt.json').write_text('{}')
    monkeypatch.setattr(b, 'BASE', tmp_path)
    with pytest.raises(ValueError): b.parent()


def test_wrong_visible_identity_rejected(monkeypatch):
    monkeypatch.setattr(b.reference, 'VISIBLE_IDS', ['wrong'])
    with pytest.raises(ValueError): b.reference.authenticate_visible()


@pytest.mark.parametrize('state', ['missing', 'stale', 'failed'])
def test_admit_fails_closed(tmp_path, monkeypatch, state):
    monkeypatch.setattr(b, 'SCRATCH', tmp_path)
    if state != 'missing':
        report = dict(status='FAIL' if state == 'failed' else 'PASS', notebook_sha256='stale')
        (tmp_path / 'smoke.json').write_bytes(b.old.encoded(report))
    with pytest.raises((FileNotFoundError, ValueError)): b.admit()


def test_exact_runtime_and_guards():
    nb, meta = b.payload()
    files, blob = b.scope(nb, meta)
    original, _ = b.parent()
    expected_files, expected_blob, _ = b.old.unpack(original)
    assert files == expected_files and blob == expected_blob and len(files) == 29
    donor_guard = ''.join(b.donor()['cells'][2]['source'])
    assert b.guard(b.unguard(donor_guard)) == donor_guard
    assert b.cell_zero_diff(nb).endswith(b'EMPTY (0 changed bytes).\n')


@pytest.mark.parametrize('target', ['receipt', 'registration', 'full-diff', 'cell0-diff'])
def test_evidence_tamper_rejected(monkeypatch, target):
    real_read = b.old.read
    real_bytes = Path.read_bytes
    if target in ('receipt', 'registration'):
        path = b.OUT / ('build-receipt.json' if target == 'receipt' else 'registration.json')
        def altered(p):
            value = real_read(p)
            if p == path: value['qualified_for_submission'] = True
            return value
        monkeypatch.setattr(b.old, 'read', altered)
    else:
        path = b.OUT / ('full-diff-vs-p65.txt' if target == 'full-diff' else 'cell0-diff-vs-own25q3.txt')
        monkeypatch.setattr(Path, 'read_bytes', lambda p: b'changed' if p == path else real_bytes(p))
    with pytest.raises(ValueError): b.validate()


def test_real_visible_arithmetic():
    report = b.replay()
    assert report['status'] == 'PASS'
    assert report['independent_pandas_exact']
    assert report['baseline_reproduced_byte_for_byte']
    assert report['candidate_csv_sha256'] == '03c4f7df001a606b98ff7b25ba21857508d30fb769a4480713d2a0dc215ae6a3'


def test_admission_cannot_launch(monkeypatch):
    assert b.admit()['qualified_for_submission'] is False
    assert b.admit()['status'] == 'HOLD_INDEPENDENT_REVIEW'
    monkeypatch.setattr(b.sys, 'argv', ['builder', 'admit'])
    with pytest.raises(ValueError, match='dry-run only'): b.main()
