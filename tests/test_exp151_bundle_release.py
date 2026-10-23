"""EXP151 offline contract tests. No CUDA initialization, DICOM read, or network."""
import ast
import inspect
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import exp151_bundle_release as m


def test_head_view_model_reduction_uses_full_cohort_ranks():
    scope = {}
    exec(m.components_source(), scope)
    a = np.zeros((10, 3, 4, 12), np.float32)
    b = np.zeros_like(a)
    # The first head ranks forward; all other heads rank backward. This
    # distinguishes rank-then-mean from averaging raw probabilities first.
    for i in range(10):
        a[i, :, 0, :] = np.array([0.1, 0.2, 0.9])[:, None]  # B: test fixture needed a column axis
        a[i, :, 1:, :] = np.array([0.9, 0.2, 0.1])[:, None, None]
        b[i] = a[i]
    result = m.reduce_heads(a, b, scope['rank_mean'], scope['fold_ranks'])
    assert result.shape == (3, 12)
    assert result[:, 0] == pytest.approx([1, 2 / 3, 1 / 3])
    # A mirrored view gets ranked separately before the view mean and
    # the served score is ranked again exactly as EXP138's fold_ranks.
    b = b[:, ::-1].copy()
    result2 = m.reduce_heads(a, b, scope['rank_mean'], scope['fold_ranks'])
    assert result2[0, 0] != result[0, 0]


def test_flip_functions_are_exact_exp138_delegate_and_keep_sagittal():
    scope = {}
    exec(m.components_source(), scope)
    ranges = scope['plane_ranges']({'slots': [['Sagittal', 'a', 4],
                                              ['Coronal', 'b', 4], ['Axial', 'c', 4]],
                                     'depth': 12})
    assert scope['flip_mask']([1, 5, 9, -1], ranges) == [False, True, True, False]
    with pytest.raises(ValueError, match='plane identification'):
        scope['flip_mask']([4], ranges)


def test_checkpoint_inventory_and_provenance_gate(tmp_path):
    tag = 'EXP138_BUNDLE_AUG1_MH_S42_F0'
    run = tmp_path / 'artifacts/experiments' / tag
    run.mkdir(parents=True)
    protocol = tmp_path / 'artifacts/experiments/EXP138_BUNDLE_AUG1_MH/s42/manifest.json'
    m.save(protocol, {'seed': 42})
    proto_sha = m.sha(protocol)
    (run / 'model_best.pt').write_bytes(b'tiny model')
    m.save(run / 'summary.json', {'mode': 'full', 'fold': 0,
           'selected_checkpoint': 'fixed-final', 'worker_cleanup': {'clean': True},
           'total_seconds': 1, 'heads': list(m.HEADS),
           'provenance': {'protocol_sha256': proto_sha}})
    m.save(run / 'provenance.json', {'seed': 42, 'fold': 0,
           'heads': list(m.HEADS), 'exp127_variant': 'AUG1', 'arch': 'resnet50',
           'norm': 'radimagenet', 'input_policy': 'center3', 'k_eval_extra': 36,
           'protocol_sha256': proto_sha})
    m.save(run / 'output-inventory.json', {name: {'sha256': m.sha(run / name),
           'bytes': (run / name).stat().st_size}
           for name in ('model_best.pt', 'summary.json', 'provenance.json')})
    assert m.checkpoint_entry(tmp_path, 42, 0)['sha256'] == m.sha(run / 'model_best.pt')
    (run / 'model_best.pt').write_bytes(b'othermodel')
    with pytest.raises(ValueError, match='SHA'):
        m.checkpoint_entry(tmp_path, 42, 0)


def test_zip_bin_integrity_and_notebook_contract(tmp_path):
    archive = tmp_path / 'code.zip.bin'
    digest = m.pack_archive(archive, {'code/a.py': b'print(1)\n'})
    assert digest == m.sha(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.read('code/a.py') == b'print(1)\n'
    with pytest.raises(ValueError, match='Unsafe archive'):
        m.pack_archive(tmp_path / 'bad.zip.bin', {'../evil': b'x'})
    nb = m.notebook()
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(cell['source'], f'exp151 cell {i}', 'exec')
    code = nb['cells'][-1]['source']
    assert 'code.zip.bin' in code and 'checkpoints.zip.bin' in code
    assert "'/kaggle/working'" in code


def test_notebook_sets_one_visible_gpu_before_any_other_code():
    code = m.notebook('a' * 64)['cells'][-1]['source']
    assert code.splitlines()[0] == "NOTEBOOK_STARTED = __import__('time').monotonic()"
    tree = ast.parse(code)
    assert isinstance(tree.body[0], ast.Assign)
    assert [ast.unparse(node.targets[0]) for node in tree.body[2:4]] == [
        "os.environ['CUDA_VISIBLE_DEVICES']", "os.environ['CUBLAS_WORKSPACE_CONFIG']"]
    assert [ast.literal_eval(node.value) for node in tree.body[2:4]] == ['0', ':4096:8']
    assert code.index("CUBLAS_WORKSPACE_CONFIG") < code.index('import torch, timm')
    assert code.count('knee-exp151-bundle-aug1-assets') == 1
    assert '.rglob(' not in code
    assert code.index("_roots = [") < code.index("checkpoints.zip.bin")
    assert code.index("torch.cuda.device_count()") < code.index("checkpoints.zip.bin")
    assert 'notebook_started=NOTEBOOK_STARTED' in code
    assert inspect.signature(m.serve).parameters['notebook_started'].default is None
    assert 'started = time.monotonic() if notebook_started is None else notebook_started' in inspect.getsource(m.serve)
    assert "CUBLAS_WORKSPACE_CONFIG=':4096:8'" in inspect.getsource(m.public3)
    compile(code, 'EXP151 notebook', 'exec')


def test_projection_and_push_plan_are_offline(tmp_path):
    p = m.projection()
    assert p['margin_seconds'] > 0 and p['studies'] == 1300
    stage = tmp_path / 'release'
    m.save(stage / 'dataset/release-manifest.json', {'schema': 'fake'})
    commands = m.push_plan(stage)['commands']
    assert commands[1] == 'kaggle datasets create -p .'
    assert commands[4] == 'kaggle kernels push -p .'
