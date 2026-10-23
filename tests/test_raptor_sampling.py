"""Contract tests for reproducible input interventions; no trainer import."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/src'))
from raptor_common import select_centers, study_rng, windows_from, load_label_csv, LAB, UID
from raptor_data import StudyWindows


def tiny_corpus(root):
    root.mkdir()
    vol = np.arange(2 * 12 * 4 * 4, dtype=np.uint8).reshape(2, 12, 4, 4)
    np.save(root / 'vol.npy', vol)
    np.save(root / 'ids.npy', np.array(['u0', 'u1']))
    np.save(root / 'masks.npy', np.ones((2, 12), np.uint8))
    (root / 'manifest.json').write_text(json.dumps({'contract': 'tiny',
        'spec': {'depth': 12, 'slots': [['A', 0, 6], ['B', 1, 6]]},
        'shards': [{'start': 0, 'stop': 2, 'file': 'vol.npy'}]}))
    return root


def test_boundary_is_only_intervention_when_masks_are_fixed():
    mask = np.ones(12, np.uint8); mask[3] = 0
    spec = {'slots': [('a', 0, 6), ('b', 0, 6)]}
    a = select_centers(mask, 12, 40, spec, 'global-valid')
    b = select_centers(mask, 12, 40, spec, 'within-slot-valid')
    assert {5, 6}.issubset(a) and not ({5, 6} & set(b))
    assert all(np.all(mask[c-1:c+2]) for c in a + b)
    assert select_centers(np.zeros(12), 12, 4, spec, 'within-slot-valid') == [-1] * 4
    assert not windows_from(np.ones((12, 4, 4), np.uint8), [-1], 4).any()


def test_stable_sampling_and_jitter_streams(tmp_path):
    root = tiny_corpus(tmp_path / 'corpus')
    labels = {'u0': np.zeros(12, np.float32), 'u1': np.ones(12, np.float32)}
    ds = StudyWindows(root, ['u0', 'u1'], labels, 4, 3, True, 'none', 42)
    first = ds[0][0].clone()
    assert (first == ds[0][0]).all()
    ds.set_epoch(1); second = ds[0][0].clone()
    assert not (first == second).all()
    ds.set_epoch(0); assert (first == ds[0][0]).all()
    a = study_rng(42, 0, 'u0', 'intensity').random()
    _ = [study_rng(42, 0, 'u0').random() for _ in range(50)]
    assert a == study_rng(42, 0, 'u0', 'intensity').random()


def test_persistent_workers_receive_epoch(tmp_path):
    import torch
    root = tiny_corpus(tmp_path / 'corpus')
    labels = {u: np.zeros(12, np.float32) for u in ['u0', 'u1']}
    ds = StudyWindows(root, ['u0', 'u1'], labels, 4, 3, True, 'none', 42)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2, persistent_workers=True)
    for ep in (0, 1, 0):
        ds.set_epoch(ep)
        actual = next(iter(loader))[0]
        expected = torch.stack([ds[i][0] for i in range(2)])
        assert torch.equal(actual, expected)
    del loader


def test_label_duplicates_are_rejected(tmp_path):
    import pandas as pd
    p = tmp_path / 'labels.csv'
    pd.DataFrame([{UID: 'u', **{t: .5 for t in LAB}}] * 2).to_csv(p, index=False)
    with pytest.raises(ValueError, match='duplicate'):
        load_label_csv(p)
