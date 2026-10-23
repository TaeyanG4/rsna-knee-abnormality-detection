"""Authenticated visible-only fast path; all tests CPU-only."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
from tools import c85r7_p65q_v1116 as b


@pytest.mark.parametrize('marker', [None, '', '0', 'False', '1'])
@pytest.mark.parametrize('ids', [[], b.VISIBLE_IDS[:2], b.VISIBLE_IDS,
                               ['different-a', 'different-b', 'different-c'],
                               b.VISIBLE_IDS + ['fourth']])
def test_marker_cohort_matrix(tmp_path, marker, ids):
    result = b.run_case(tmp_path, ids=ids, manifest_ids=b.VISIBLE_IDS, marker=marker)
    expected = not marker and ids == b.VISIBLE_IDS
    assert result['FAST'] == expected
    assert result['placeholder_written'] == expected
    assert result['original_first_statement'] == (not expected)
    assert result['error'] is None


@pytest.mark.parametrize('ids', [[], b.VISIBLE_IDS[:2], ['a', 'b', 'c'],
                               b.VISIBLE_IDS + ['fourth'],
                               [b.VISIBLE_IDS[0]] * 3])
def test_manifest_must_be_exact(tmp_path, ids):
    result = b.run_case(tmp_path, ids=b.VISIBLE_IDS, manifest_ids=ids)
    assert not result['FAST'] and not result['placeholder_written']
    assert result['original_first_statement']


@pytest.mark.parametrize('raw', ['wrong_header\nfoo\n',
                               'StudyInstanceUID,StudyInstanceUID\nfoo,foo\n',
                               'StudyInstanceUID\n"unclosed'])
def test_malformed_manifest_no_placeholder(tmp_path, raw):
    result = b.run_case(tmp_path, ids=b.VISIBLE_IDS, manifest_ids=[], raw_manifest=raw)
    assert not result['FAST'] and not result['placeholder_written']


@pytest.mark.parametrize('layout', ['flat', 'nested'])
def test_manifest_order_irrelevant_and_real_mounts(tmp_path, layout):
    result = b.run_case(tmp_path, ids=list(reversed(b.VISIBLE_IDS)),
                        manifest_ids=list(reversed(b.VISIBLE_IDS)), layout=layout)
    assert result['FAST']


@pytest.mark.parametrize('marker', ['0', 'False', '1'])
def test_nonempty_marker_never_touches_filesystem(marker):
    source = b.source()
    with patch.dict(b.os.environ, {'KAGGLE_IS_COMPETITION_RERUN': marker}), \
         patch('pathlib.Path', side_effect=AssertionError('filesystem touched')):
        ns = {}
        exec(source, ns)
        assert ns['FAST'] is False


@pytest.mark.parametrize('malformation', ['absent', 'ambiguous', 'missing-series', 'missing-manifest'])
def test_malformed_mount_no_placeholder(tmp_path, monkeypatch, malformation):
    comp = tmp_path / 'rsna-knee-abnormality-detection'
    if malformation != 'absent':
        comp.mkdir()
        if malformation != 'missing-manifest':
            (comp / 'test.csv').write_text('StudyInstanceUID\n')
        if malformation == 'ambiguous':
            other = tmp_path / 'competitions' / comp.name
            other.mkdir(parents=True)
            (other / 'test.csv').write_text('StudyInstanceUID\n')
    monkeypatch.delenv('KAGGLE_IS_COMPETITION_RERUN', raising=False)
    monkeypatch.delenv('C85_COMPETITION', raising=False)
    if malformation == 'missing-manifest': monkeypatch.setenv('C85_COMPETITION', str(comp))
    source = b.source()
    output = tmp_path / 'submission.csv'
    def mapped(value):
        if str(value) == '/kaggle/input': return tmp_path
        if str(value) == '/kaggle/working/submission.csv': return output
        return Path(value)
    ns = {}
    with patch('pathlib.Path', mapped), pytest.raises((ValueError, OSError)):
        exec(source, ns)
    assert ns['FAST'] is False and not output.exists()


def test_frozen_scope():
    assert b.validate()['runtime_files_identical'] == 29
    nb, meta = b.payload()
    before, before_meta = b.parent()
    assert nb['cells'][1] == before['cells'][0]
    assert b.unguard(''.join(nb['cells'][2]['source'])) == ''.join(before['cells'][1]['source'])
    assert nb['cells'][0] == b.donor()['cells'][0]
    assert {k:v for k,v in meta.items() if k not in ['id', 'title']} == {
        k:v for k,v in before_meta.items() if k not in ['id', 'title']}
    assert meta['title'] == 'knee-lb-p65q-v1'


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
    monkeypatch.setattr(b.harness, 'VISIBLE_IDS', ['wrong'])
    with pytest.raises(ValueError): b.harness.authenticate_visible()


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
        path = b.OUT / ('full-diff-vs-p65.txt' if target == 'full-diff' else 'cell0-diff-vs-own25q2.txt')
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
