"""EXP129 tiny CPU-only loader and head proofs. Never initialises CUDA."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/src'))
from raptor_common import SLOTS64, Corpus, select_centers, study_rng
from raptor_radimagenet import CenterSliceWindows
from raptor_exp129 import (VARIANTS, PositionSliceWindows, adapter_state_hashes,
    base_state_hashes, build_variant, collate, head_equivalence, initial_adapter_state,
    loader_factory, model_forward, move_input, ordered_equivalence, slot_positions)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def make_fail():  # one distinct stub per attribute: torch._dynamo rejects one object registered under several rules
        def fail(*args, **kwargs):
            raise AssertionError('EXP129 tests must never initialize CUDA')
        return fail
    for name in ('_lazy_init', 'init', 'get_rng_state', 'manual_seed_all'):
        monkeypatch.setattr(torch.cuda, name, make_fail())
    with torch.device('cpu'):
        yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def corpus(tmp_path):
    ids = [f'synthetic-{i}' for i in range(10)]
    spec = dict(slots=SLOTS64, depth=64)
    volumes = np.broadcast_to(np.arange(64, dtype=np.uint8)[None, :, None, None],
                               (10, 64, 8, 8)).copy()
    masks = np.ones((10, 64), np.uint8)
    masks[0, 17:20] = 0
    masks[1, 30:35] = 0
    masks[2] = 0
    np.save(tmp_path / 'ids.npy', np.asarray(ids))
    np.save(tmp_path / 'masks.npy', masks)
    np.save(tmp_path / 'volumes.npy', volumes)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(contract='synthetic', spec=spec,
        shards=[dict(start=0, stop=10, file='volumes.npy')])), encoding='utf-8')
    labels = {u: np.asarray([(i + j) % 2 for j in range(12)], np.float32) for i, u in enumerate(ids)}
    return tmp_path, ids, labels


class FakeRaptor(nn.Module):
    """Control's exact head, with a tiny CPU backbone in place of timm."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 2048)
        self.norm = nn.LayerNorm(2048)
        self.att = nn.Sequential(nn.Linear(2048, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.randn(12, 2048) * .02)
        self.clsb = nn.Parameter(torch.zeros(12))
        self.n = 12

    def encode(self, x):
        return self.backbone(x.mean(dim=(-2, -1)))

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        return self.head(self.encode(x))


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('seed', (42, 43))
def test_ordered_disabled_and_enabled_images(variant, seed, corpus):
    path, ids, labels = corpus
    report = ordered_equivalence(variant, path, ids[:8], ids[8:], labels, seed, res=8)
    assert report['batches'] == 4
    assert report['exact_inputs_labels_and_order'] and report['enabled_images_equal_control']
    assert loader_factory(variant, False) is CenterSliceWindows
    json.dumps(report)


def test_position_slot_edges_empty_and_permutation():
    spec = dict(slots=SLOTS64)
    indices = [0, 17, 18, 31, 32, 43, 44, 51, 52, 63, -1]
    expected = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], np.float32)
    positions = slot_positions(indices, spec, 64)
    np.testing.assert_array_equal(positions[:, 0], expected)
    perm = [9, 2, 6, 0, 10, 5, 8, 1, 3, 7, 4]
    np.testing.assert_array_equal(slot_positions([indices[i] for i in perm], spec, 64), positions[perm])
    assert slot_positions([8, 24, 40], spec, 64)[0, 0] == np.float32(8 / 17)
    with pytest.raises(ValueError, match='depth'):
        slot_positions([0], spec, 63)
    with pytest.raises(ValueError, match='index'):
        slot_positions([64], spec, 64)
    with pytest.raises(ValueError, match='index'):
        slot_positions([-2], spec, 64)


@pytest.mark.parametrize('train,k', ((True, 12), (False, 24), (False, 36)))
def test_positions_keep_sampled_image_association(train, k, corpus):
    path, ids, labels = corpus
    ds = PositionSliceWindows(path, ids, labels, 8, k, train, 'radimagenet', 42,
                              jitter=0, window_policy='within-slot-valid')
    c = Corpus(path)
    for epoch in (0, 1):
        ds.set_epoch(epoch)
        for index in (0, 1, 2):
            (x, p), y, uid = ds[index]
            rng = study_rng(42, epoch, uid) if train else None
            centers = select_centers(c.mask(uid), c.depth, k, c.spec, 'within-slot-valid', rng)
            np.testing.assert_array_equal(p.numpy(), slot_positions(centers, c.spec, c.depth))
            decoded = ((x[:, 0, 0, 0] + 1) * 127.5).round().to(torch.int64)
            assert decoded.tolist() == [max(i, 0) for i in centers]
            assert torch.equal(y, torch.from_numpy(labels[uid]))
            if index == 2:
                assert torch.count_nonzero(p) == 0 and bool((x == -1).all())
    batch = collate([ds[0], ds[1]])
    moved = move_input(batch[0], 'cpu')
    assert isinstance(moved, tuple) and moved[0].shape == (2, k, 3, 8, 8)
    assert moved[1].shape == (2, k, 1)


@pytest.mark.parametrize('variant', ('POS', 'CTX'))
@pytest.mark.parametrize('seed', (42, 43))
def test_adapters_preserve_control_state_and_rng_and_hash_registration(variant, seed):
    base = FakeRaptor()
    original = base_state_hashes(base)
    rng = torch.get_rng_state().clone()
    active = build_variant(base, variant, seed)
    assert torch.equal(rng, torch.get_rng_state())
    assert base_state_hashes(active) == original
    assert adapter_state_hashes(active) == initial_adapter_state(variant, seed)
    assert torch.equal(rng, torch.get_rng_state())
    another = build_variant(FakeRaptor(), variant, seed)
    assert adapter_state_hashes(active) == adapter_state_hashes(another)
    if variant == 'CTX':
        different = initial_adapter_state(variant, seed + 1)
        assert adapter_state_hashes(active) != different
        layer0 = active.exp129_context.layers[0].linear1.weight
        layer1 = active.exp129_context.layers[1].linear1.weight
        assert not torch.equal(layer0, layer1)
    else:
        assert torch.count_nonzero(active.exp129_pos_projection.weight) == 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_head_proof_training_and_eval_and_constant_position(variant):
    model = FakeRaptor()
    rng = torch.get_rng_state().clone()
    state = base_state_hashes(model)
    report = head_equivalence(model, variant, 42)
    assert torch.equal(rng, torch.get_rng_state())
    assert base_state_hashes(model) == state
    assert len(report['checks']) == 4
    assert all(check['exact_outputs'] and check['exact_rng_state'] for check in report['checks'])
    if variant == 'POS':
        assert report['constant_position_max_abs'] < 1e-5
    json.dumps(report)


@pytest.mark.parametrize('variant', VARIANTS)
def test_disabled_full_forward_gradients_optimizer_step_identical(variant):
    base = FakeRaptor().train()
    other = copy.deepcopy(base)
    assert build_variant(other, variant, 42, enabled=False) is other
    assert base_state_hashes(base) == base_state_hashes(other)
    assert not adapter_state_hashes(other)[1]
    x = torch.randn(2, 4, 3, 4, 4)
    state = torch.get_rng_state()
    outputs = []
    for model in (base, other):
        torch.set_rng_state(state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        logits = model_forward(model, move_input(x, 'cpu'))
        logits.square().sum().backward()
        outputs.append(logits.detach())
        optimizer.step()
    assert torch.equal(*outputs)
    assert base_state_hashes(base) == base_state_hashes(other)


@pytest.mark.parametrize('variant', ('POS', 'CTX'))
def test_zero_adapter_full_forward_and_learning_signal(variant):
    base = FakeRaptor().train()
    other = build_variant(copy.deepcopy(base), variant, 42)
    x = torch.randn(2, 4, 3, 4, 4)
    p = torch.rand(2, 4, 1)
    rng = torch.get_rng_state()
    left = base(x)
    torch.set_rng_state(rng)
    right = model_forward(other, (x, p) if variant == 'POS' else x)
    assert torch.equal(left, right)
    right.square().sum().backward()
    grad = (other.exp129_pos_projection.weight.grad if variant == 'POS'
            else other.exp129_context.expand.weight.grad)
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    if variant == 'POS':
        with pytest.raises(ValueError, match='expects'):
            other(x)
        with pytest.raises(ValueError, match='per window'):
            other((x, p[:, :-1]))
