"""CPU fake-only release contract checks; never import torch or use network."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import exp157_l2_release as m

def test_notebook_checks_mounts_before_materialization_and_cuda(monkeypatch):
    adapter = SimpleNamespace(a_notebook=lambda data: {'cells': [{'source': ''}, {'source': 'import exp133_oai_b3 as e\nstart = time.monotonic()\nmaterialize()\n'}]})
    monkeypatch.setattr(m.importlib.util, 'spec_from_file_location', lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda x: None)))
    monkeypatch.setattr(m.importlib.util, 'module_from_spec', lambda x: adapter)
    data = {'checkpoints': [{'path': 'checkpoints/rad42_f0.pt', 'bytes': 4}]}
    code = m.notebook(data)['cells'][-1]['source']
    ast.parse(code)
    assert code.startswith('NOTEBOOK_STARTED = ')
    assert code.index('CUDA_VISIBLE_DEVICES') < code.index('import exp133_oai_b3')
    assert code.index('CUBLAS_WORKSPACE_CONFIG') < code.index('import exp133_oai_b3')
    assert code.index('_p.stat().st_size') < code.index('materialize()')
    assert "start = NOTEBOOK_STARTED" in code
    assert '_exp157_input_dir' in code and 'rglob' not in code
    assert code.count('/kaggle/input/datasets/taeyangg4/knee-exp157-l2-aug1-assets') == 1
    assert code.count('/kaggle/input/competitions/rsna-knee-abnormality-detection') == 1

def test_manifest_requires_distinct_complete_ten_models(monkeypatch):
    monkeypatch.setattr(m, 'checkpoint_entry', lambda root, seed, fold: {'seed': seed, 'fold': fold, 'sha256': f'{seed}-{fold}'})
    data = m.manifest(Path('fake'))
    assert len(data['checkpoints']) == 10
    assert data['dataset'] == m.DATASET
    monkeypatch.setattr(m, 'checkpoint_entry', lambda *a: {'sha256': 'duplicate'})
    with pytest.raises(ValueError, match='distinct'):
        m.manifest(Path('fake'))

def test_stage_rejects_tampered_checkpoint_before_copy(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'upload-stage').mkdir()
    source = tmp_path / 'fake.pt'
    source.write_bytes(b'tiny')
    m.base.save(out / 'upload-manifest.json', {'files': [{'source': str(source), 'destination': 'checkpoints/fake.pt', 'bytes': 4, 'sha256': 'wrong'}]})
    with pytest.raises(ValueError, match='Source hash'):
        m.base.stage(out)
    assert not (out / 'upload-stage/checkpoints/fake.pt').exists()
