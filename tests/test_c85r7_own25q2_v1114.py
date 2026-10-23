"""Authenticated visible-only fast path; all tests CPU-only."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
from tools import c85r7_own25q2_v1114 as b


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
    assert nb['cells'][1:] == before['cells'][1:]
    assert {k:v for k,v in meta.items() if k not in ['id', 'title']} == {
        k:v for k,v in before_meta.items() if k not in ['id', 'title']}
    assert meta['title'] == 'knee-lb-own25q2-v1'


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
    monkeypatch.setattr(b, 'VISIBLE_IDS', ['wrong'])
    with pytest.raises(ValueError): b.authenticate_visible()


@pytest.mark.parametrize('state', ['missing', 'stale', 'failed'])
def test_admit_fails_closed(tmp_path, monkeypatch, state):
    monkeypatch.setattr(b, 'SCRATCH', tmp_path)
    if state != 'missing':
        report = dict(status='FAIL' if state == 'failed' else 'PASS', notebook_sha256='stale')
        (tmp_path / 'smoke.json').write_bytes(b.old.encoded(report))
    with pytest.raises((FileNotFoundError, ValueError)): b.admit()
