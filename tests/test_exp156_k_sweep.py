"""CPU-only contract checks; no checkpoint load, CUDA, dataset or network access."""
import ast
import json
from pathlib import Path
import zipfile

import pytest

from tools import exp156_k_sweep as e


def test_notebook_header_and_two_worker_contract():
    cell = e.notebook('abc')['cells'][1]['source']
    assert cell.startswith("STARTED = __import__('time').monotonic()\n")
    assert cell.index("CUBLAS_WORKSPACE_CONFIG") < cell.index('import hashlib')
    assert "os.environ['CUDA_VISIBLE_DEVICES'] = ''" in cell
    assert "subprocess.run(cmd, env=env" in cell
    mount = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == 'mount')
    roots = next(n.value for n in mount.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'roots' for t in n.targets))
    assert isinstance(roots, ast.List) and len(roots.elts) == 2
    assert [ast.unparse(p) for p in roots.elts] == [
        "pathlib.Path('/kaggle/input') / slug",
        "pathlib.Path('/kaggle/input/datasets') / ('taeyangg4' if slug != 'resnet-50-radimagenet-marwan' else 'marwanmath') / slug",
    ]
    assert 'rglob(' not in cell
    assert "len(names) == len(set(names))" in cell
    assert e.SEEDS == (42,) and e.K_VALUES == (36, 62)
    assert e.WORKER_FOLDS == ((0, 2, 4), (1, 3))
    assert sorted(f for folds in e.WORKER_FOLDS for f in folds) == list(range(5))


def test_archive_inventory_rejects_unsafe_and_duplicates(tmp_path):
    archive = tmp_path / 'data.zip.bin'
    e.pack(archive, {'safe/a.txt': b'a', 'safe/b.txt': b'b'})
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ['safe/a.txt', 'safe/b.txt']
        assert len(z.namelist()) == len(set(z.namelist()))
    with pytest.raises(ValueError):
        e.pack(tmp_path / 'bad.zip.bin', {'../escape': b'x'})
    with pytest.raises(ValueError):
        e.pack(tmp_path / 'bad2.zip.bin', {'C:\\abs': b'x'})


def test_csv_roundtrip_and_reject_wrong_order(tmp_path):
    np = pytest.importorskip('numpy')
    ids = ['study-a', 'study-b']
    values = np.array([[.1]*12, [.9]*12])
    path = tmp_path / 'oof.csv'
    e.write_csv(path, ids, values)
    assert np.allclose(e.csv_values(path, ids), values)
    with pytest.raises(ValueError, match='order'):
        e.csv_values(path, ids[::-1])


def test_collect_refuses_partial_session(tmp_path):
    source = tmp_path / 'download'
    source.mkdir()
    (source / 'exp156-status.json').write_text(json.dumps({'state': 'INCOMPLETE'}))
    with pytest.raises(ValueError, match='incomplete'):
        e.collect(source, tmp_path / 'collected', tmp_path / 'stage')
    assert not (tmp_path / 'collected').exists()


def test_serving_time_gate_requires_both_releases(tmp_path):
    assert not e.timing_gate(None, 62)['pass_']
    p = tmp_path / 'timing.json'
    timing = {'schema': 'exp156-serving-time-v1', 'units': 'seconds',
              'cohort': 'full competition test', 'releases': {
                  'B3': {'36': 28000, '62': 31000, 'reserve': 500,
                         'source': 'b3 timing receipt'},
                  'H1': {'36': 31000, '62': 31800, 'reserve': 600,
                         'source': 'h1 timing receipt'}}}
    p.write_text(json.dumps(timing))
    assert e.timing_gate(p, 62)['pass_']  # H1 exactly meets the 9-hour cap.
    timing['releases']['H1']['62'] = 31801
    p.write_text(json.dumps(timing))
    assert not e.timing_gate(p, 62)['pass_']
    del timing['releases']['H1']
    p.write_text(json.dumps(timing))
    with pytest.raises(ValueError, match='evidence differs'):
        e.timing_gate(p, 62)


def test_packaged_serving_functions_match_frozen_sources():
    body = e.components_source(e.ROOT).decode()
    assert 'def select_centers(' in body
    assert 'def center3_windows_from(' in body
    assert 'def plane_ranges(' in body
    assert 'def flip_mask(' in body
    compile(body, 'exp156_components.py', 'exec')
