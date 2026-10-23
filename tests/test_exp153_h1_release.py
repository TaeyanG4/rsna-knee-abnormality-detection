"""EXP153 release contract tests: tiny files and CPU arrays only."""
import json
import inspect
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import exp153_h1_release as m


def _fake_run(root, seed, fold):
    tag = m.tag_for(seed, fold)
    run = root / 'artifacts/experiments' / tag
    run.mkdir(parents=True)
    if tag.startswith('EXP149'):
        protocol = root / m.EXP149_MANIFEST
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text(json.dumps({'runs': {tag: {
            'initial_state_sha256': m.INITIAL_SHA, 'heads': list(m.HEADS)}}}), encoding='utf-8')
        protocol_path = None
    else:
        protocol_path = 'artifacts/experiments/EXP153_H1_PROTOCOL/manifest.json'
        protocol = root / protocol_path
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text('{"schema":"tiny-exp153"}', encoding='utf-8')
    proto_sha = m.base.sha(protocol)
    prov = {'run': tag, 'seed': seed, 'fold': fold, 'heads': list(m.HEADS),
            'label_set': 'h1', 'arch': 'resnet50', 'norm': 'radimagenet',
            'input_policy': 'center3', 'k_eval_extra': 36, 'epochs': 20,
            'train_folds': [k for k in range(5) if k != fold], 'eval_folds': [fold],
            'checkpoint_selection': 'fixed-final', 'steps': 10,
            'gold_access': 'disabled', 'full_model_initial_state_sha256': m.INITIAL_SHA,
            'fp16_skipped_updates': 0, 'protocol_sha256': proto_sha}
    if protocol_path:
        prov['protocol_path'] = protocol_path
    summary = {'run': tag, 'fold': fold, 'mode': 'full', 'label_set': 'h1',
               'heads': list(m.HEADS), 'selected_checkpoint': 'fixed-final',
               'selected_epoch_1based': 20, 'optimizer_updates': 10,
               'skipped_updates': 0, 'gold': 0,
               'gold_access': 'disabled', 'worker_cleanup': {'clean': True},
               'total_seconds': 1, 'provenance': {'protocol_sha256': proto_sha}}
    for name, obj in (('summary.json', summary), ('provenance.json', prov),
                      ('history.json', [{'epoch': i} for i in range(20)])):
        (run / name).write_text(json.dumps(obj), encoding='utf-8')
    names = {'oof_best.csv', 'oof_k36.csv'} | {
        f'oof_head_{h}{suffix}.csv' for h in m.HEADS for suffix in ('', '_k36')}
    for name in names:
        (run / name).write_bytes(b'tiny predictions')
    (run / 'model_best.pt').write_bytes(b'tiny model')
    inventory = {p.name: {'sha256': m.base.sha(p), 'bytes': p.stat().st_size}
                 for p in run.iterdir() if p.name != 'output-inventory.json'}
    (run / 'output-inventory.json').write_text(json.dumps(inventory), encoding='utf-8')
    return run


@pytest.mark.parametrize('seed,fold', [(42, 0), (42, 2), (43, 4)])
def test_checkpoint_gate_accepts_complete_exp149_and_exp153_runs(tmp_path, seed, fold):
    run = _fake_run(tmp_path, seed, fold)
    entry = m.checkpoint_entry(tmp_path, seed, fold)
    assert entry['tag'] == run.name
    assert entry['path'] == f'checkpoints/s{seed}_f{fold}.pt'
    assert entry['sha256'] == m.base.sha(run / 'model_best.pt')


def test_checkpoint_gate_rejects_skips_missing_head_and_changed_model(tmp_path):
    run = _fake_run(tmp_path, 43, 2)
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    summary['skipped_updates'] = 1
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA'):
        m.checkpoint_entry(tmp_path, 43, 2)
    _repair_inventory(run)
    with pytest.raises(ValueError, match='completion'):
        m.checkpoint_entry(tmp_path, 43, 2)
    summary['skipped_updates'] = 0
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    _repair_inventory(run)
    (run / 'oof_head_aquila_v1_k36.csv').unlink()
    with pytest.raises(ValueError, match='missing'):
        m.checkpoint_entry(tmp_path, 43, 2)
    (run / 'oof_head_aquila_v1_k36.csv').write_bytes(b'tiny predictions')
    (run / 'model_best.pt').write_bytes(b'different model')
    with pytest.raises(ValueError, match='bytes'):
        m.checkpoint_entry(tmp_path, 43, 2)


def _repair_inventory(run):
    inv_path = run / 'output-inventory.json'
    inv = json.loads(inv_path.read_text(encoding='utf-8'))
    for name in ('summary.json',):
        path = run / name
        inv[name] = {'sha256': m.base.sha(path), 'bytes': path.stat().st_size}
    inv_path.write_text(json.dumps(inv), encoding='utf-8')


def test_five_head_full_cohort_reduction_and_plane_flip():
    scope = {}
    exec(m.base.components_source(), scope)
    a = np.zeros((10, 3, 5, 12), np.float32)
    a[:, :, 0, :] = np.array([0.1, 0.2, 0.9])[None, :, None]
    a[:, :, 1:, :] = np.array([0.9, 0.2, 0.1])[None, :, None, None]
    result = m.reduce_heads(a, a.copy(), scope['rank_mean'], scope['fold_ranks'])
    assert result.shape == (3, 12)
    assert result[:, 0] == pytest.approx([1, 2 / 3, 1 / 3])
    with pytest.raises(ValueError, match='shape'):
        m.reduce_heads(a[:, :, :4], a[:, :, :4], scope['rank_mean'], scope['fold_ranks'])
    ranges = scope['plane_ranges']({'slots': [['Sagittal', 'a', 4],
                                              ['Coronal', 'b', 4], ['Axial', 'c', 4]],
                                     'depth': 12})
    assert scope['flip_mask']([1, 5, 9, -1], ranges) == [False, True, True, False]


def test_notebook_routes_to_five_head_serving_and_archive_stays_offline(tmp_path):
    nb = m.notebook('a' * 64)
    cell = nb['cells'][-1]['source']
    assert cell.splitlines()[0] == "NOTEBOOK_STARTED = __import__('time').monotonic()"
    compile(cell, 'EXP153 notebook', 'exec')
    assert 'knee-exp153-h1-aquila-head-assets' in cell
    assert cell.count('knee-exp153-h1-aquila-head-assets') == 1
    assert '.rglob(' not in cell
    assert cell.index('CUBLAS_WORKSPACE_CONFIG') < cell.index('import torch, timm')
    assert cell.index('torch.cuda.device_count()') < cell.index('checkpoints.zip.bin')
    assert 'notebook_started=NOTEBOOK_STARTED' in cell
    assert inspect.signature(m.serve).parameters['notebook_started'].default is None
    assert 'started = time.monotonic() if notebook_started is None else notebook_started' in inspect.getsource(m.serve)
    assert "CUBLAS_WORKSPACE_CONFIG=':4096:8'" in inspect.getsource(m.public3)
    assert 'import exp153_h1_release as release' in cell
    assert 'import exp151_bundle_release as release' not in cell
    assert 'code.zip.bin' in cell and 'checkpoints.zip.bin' in cell
    archive = tmp_path / 'code.zip.bin'
    m.base.pack_archive(archive, {'code/a.py': b'print(1)\n'})
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ['code/a.py']
    with pytest.raises(ValueError, match='Unsafe archive'):
        m.base.pack_archive(tmp_path / 'bad.zip.bin', {'../escape': b'x'})


def test_push_plan_is_only_a_plan(tmp_path):
    m.base.save(tmp_path / 'dataset/release-manifest.json', {'schema': m.VERSION_SCHEMA})
    plan = m.push_plan(tmp_path)
    assert plan['kernel'] == 'taeyangg4/knee-exp153-h1-aquila-head'
    assert plan['commands'][1] == 'kaggle datasets create -p .'
    assert plan['commands'][4] == 'kaggle kernels push -p .'
