"""Notebook-only fallback scope and bypass tests; CPU only."""
import ast
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
from tools import c85r7_own25q_v1111 as b


def test_frozen_stage_scope():
    assert b.validate()['embedded_files_identical'] == 29
    nb, _ = b.payload()
    base, _ = b.parent()
    original = ast.parse(''.join(base['cells'][1]['source']))
    guarded = ast.parse(''.join(nb['cells'][2]['source']))
    assert ast.dump(ast.Module(body=guarded.body[0].body, type_ignores=[])) == ast.dump(original)


@pytest.mark.parametrize('mutation', ['blob', 'guard', 'fast', 'markdown', 'metadata', 'extra_cell'])
def test_scope_fails_closed(mutation):
    nb, meta = b.payload()
    if mutation == 'metadata': meta['enable_gpu'] = False
    elif mutation == 'extra_cell': nb['cells'].append(copy.deepcopy(nb['cells'][0]))
    elif mutation == 'markdown': nb['cells'][1]['source'].append('changed')
    else:
        i = 0 if mutation == 'fast' else 2
        src = ''.join(nb['cells'][i]['source'])
        if mutation == 'guard': src = src.replace('if not FAST:', 'if FAST:', 1)
        elif mutation == 'fast': src = src.replace('<= 3', '<= 4', 1)
        else: src = src.replace('BLOB = ', 'BLOB = "corruption" + ', 1)
        nb['cells'][i]['source'] = src.splitlines(True)
    with pytest.raises(ValueError): b.scope(nb, meta)


@pytest.mark.parametrize('value', ['1', '0', 'true'])
def test_rerun_does_not_resolve_or_access_input(value):
    with patch.dict(b.os.environ, {'KAGGLE_IS_COMPETITION_RERUN': value}), patch('pathlib.Path', side_effect=AssertionError('input touched')):
        ns = {}
        exec(b.FAST_SOURCE, ns)
        assert ns['FAST'] is False


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4, 5])
def test_exact_count_predicate_and_csv(tmp_path, monkeypatch, count):
    comp = tmp_path / 'competition'
    series = comp / 'test_series'
    series.mkdir(parents=True)
    ids = [f'uid-{i}' for i in range(count)]
    for uid in ids: (series / uid).mkdir()
    (series / 'ignore-file').write_text('x')
    monkeypatch.delenv('KAGGLE_IS_COMPETITION_RERUN', raising=False)
    monkeypatch.setenv('C85_COMPETITION', str(comp))
    output = tmp_path / 'submission.csv'
    def path(value):
        return output if str(value) == '/kaggle/working/submission.csv' else Path(value)
    with patch('pathlib.Path', path):
        ns = {}
        exec(b.FAST_SOURCE, ns)
    assert ns['FAST'] == (count <= 3)
    assert output.exists() == (count <= 3)
    if output.exists():
        with output.open(newline='') as f: rows = list(b.csv.reader(f))
        assert rows[0] == b.HEADER
        assert rows[1:] == [[uid] + ['0.5'] * 12 for uid in sorted(ids)]


def test_ambiguous_mount_rejected(tmp_path, monkeypatch):
    for part in ['', 'competitions']:
        comp = tmp_path / part / 'rsna-knee-abnormality-detection'
        comp.mkdir(parents=True)
        (comp / 'test.csv').write_text('StudyInstanceUID\n')
    monkeypatch.delenv('KAGGLE_IS_COMPETITION_RERUN', raising=False)
    monkeypatch.delenv('C85_COMPETITION', raising=False)
    def path(value): return tmp_path if str(value) == '/kaggle/input' else Path(value)
    with patch('pathlib.Path', path), pytest.raises(ValueError, match='absent or ambiguous'):
        exec(b.FAST_SOURCE, {})


def test_admission_missing_smoke_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'SCRATCH', tmp_path)
    with pytest.raises(FileNotFoundError): b.admit()
