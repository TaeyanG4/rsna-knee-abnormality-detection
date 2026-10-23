"""CPU-only plane-aware flip tests; all tensors stay on CPU."""
import numpy as np
import pytest
import torch

from tools import exp124_flip_tta as ex


SPEC = {'depth': 64, 'slots': [['Sagittal', 1, 18], ['Sagittal', 0, 14],
                               ['Coronal', 1, 12], ['Coronal', 0, 8], ['Axial', -1, 12]]}


def test_plane_ranges_match_corpus_slot_contract():
    assert ex.plane_ranges(SPEC) == [dict(plane='Sagittal', start=0, stop=18),
                                     dict(plane='Sagittal', start=18, stop=32),
                                     dict(plane='Coronal', start=32, stop=44),
                                     dict(plane='Coronal', start=44, stop=52),
                                     dict(plane='Axial', start=52, stop=64)]
    assert ex.flip_mask([2, 20, 35, 46, 55, -1], ex.plane_ranges(SPEC)) == [False, False, True, True, True, False]


def test_ambiguous_plane_and_cross_slot_window_fail_closed():
    with pytest.raises(ValueError, match='plane identification failed'):
        ex.plane_ranges({'depth': 64, 'slots': [['Sagittal', 1, 64]]})
    with pytest.raises(ValueError, match='plane identification failed'):
        ex.plane_ranges({'depth': 64, 'slots': [['Sagittal', 1, 18], ['Unknown', 1, 46]]})
    with pytest.raises(ValueError, match='plane identification failed'):
        ex.flip_mask([32], ex.plane_ranges(SPEC))


def test_mirror_only_coronal_axial_windows_cpu():
    before = torch.cuda.is_initialized()
    x = torch.arange(4 * 1 * 2 * 3).reshape(4, 1, 2, 3).float()
    y = ex.mirror_windows(x, [False, True, True, False])
    assert torch.equal(y[0], x[0]) and torch.equal(y[3], x[3])
    assert torch.equal(y[1], torch.flip(x[1], dims=(-1,)))
    assert torch.equal(y[2], torch.flip(x[2], dims=(-1,)))
    assert torch.equal(x, torch.arange(24).reshape(4, 1, 2, 3).float())
    assert torch.cuda.is_initialized() == before


def test_tiny_cpu_pair_inference_matches_manual_average():
    class Mean(torch.nn.Module):
        def forward(self, x):
            v = x.mean(dim=(1, 2, 3, 4))
            return v[:, None].expand(-1, 12)

    x = torch.arange(24, dtype=torch.float32).reshape(2, 1, 1, 3, 4) / 24
    loader = [(x, torch.flip(x, dims=(-1,)), torch.zeros(2, 12), ['a', 'b'])]
    plain, flip, ids = ex.infer_pair(Mean(), loader, 'cpu', 'exp113')
    assert ids == ['a', 'b'] and plain.shape == (2, 12)
    assert np.allclose((plain + flip) / 2, plain)


def test_queue_has_two_independent_jobs():
    entries = ex.queue_entries('PREVIOUS')
    assert [e['tag'] for e in entries] == ['EXP124_FLIP_TTA_EXP113', 'EXP124_FLIP_TTA_EXP117']
    assert entries[0]['after'] == 'PREVIOUS'
    assert entries[1]['after'] == entries[0]['tag']
    assert all(e['continue_on_failure'] for e in entries)
