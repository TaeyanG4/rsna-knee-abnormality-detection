"""Tiny CPU fakes: AUG1 equivalence includes RNG, augmentation and AdamW steps."""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/src'))
import raptor_exp127_aug1 as aug
import raptor_exp139 as v139
from raptor_common import SLOTS64, select_centers, study_rng
from raptor_exp129_late import LateTracker


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def make_fail(name):
        def fail(*args, **kwargs):
            raise AssertionError('CUDA forbidden in EXP139 tests: ' + name)
        return fail
    for name in ('_lazy_init', 'init', 'get_rng_state', 'set_rng_state', 'manual_seed_all'):
        monkeypatch.setattr(torch.cuda, name, make_fail(name))
    with torch.device('cpu'):
        yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def corpus(tmp_path):
    ids = [f'fake-{i}' for i in range(10)]
    # Slice-dependent, spatially asymmetric images reveal flips/affine/ordering.
    vol = (np.arange(10 * 64 * 8 * 8).reshape(10, 64, 8, 8) * 7 % 251).astype(np.uint8)
    masks = np.ones((10, 64), np.uint8)
    masks[0, 17:21] = 0
    masks[1, 30:35] = 0
    masks[2] = 0
    np.save(tmp_path / 'ids.npy', np.asarray(ids))
    np.save(tmp_path / 'masks.npy', masks)
    np.save(tmp_path / 'volumes.npy', vol)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(contract='synthetic',
        spec=dict(slots=SLOTS64, depth=64), shards=[dict(start=0, stop=10, file='volumes.npy')])))
    labels = {u: np.asarray([(i + j) % 2 for j in range(12)], np.float32) for i, u in enumerate(ids)}
    return tmp_path, ids, labels


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


class FakeRaptor(nn.Module):
    """Original MIL arithmetic with a tiny three-channel linear backbone."""
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


@pytest.mark.parametrize('variant', v139.VARIANTS)
@pytest.mark.parametrize('seed', (42, 43))
def test_ordered_aug1_inputs(variant, seed, corpus):
    path, ids, labels = corpus
    result = v139.ordered_equivalence(variant, path, ids[:8], ids[8:], labels, seed, res=8)
    assert result['batches'] == 4 and result['enabled_images_equal_control']
    assert v139.loader_factory(variant, False) is aug.AugmentedSliceWindows


@pytest.mark.parametrize('variant', v139.VARIANTS)
@pytest.mark.parametrize('seed', (42, 43))
def test_augmentation_draw_stream_and_positions(variant, seed, corpus, monkeypatch):
    path, ids, labels = corpus
    traces = []
    original = aug.augment_windows

    def observed(x, centers, rng, spec):
        before = rng.getstate()
        out = original(x, centers, rng, spec)
        after = rng.getstate()
        replay = random.Random()
        replay.setstate(before)
        for _ in range(6 * len(centers)):
            replay.random()
        assert replay.getstate() == after  # includes empty + sagittal windows
        traces.append((tuple(centers), before, after))
        return out

    monkeypatch.setattr(aug, 'augment_windows', observed)
    args = (path, ids, labels, 8, 12, True, 'radimagenet', seed)
    kw = dict(window_policy='within-slot-valid')
    datasets = [aug.AugmentedSliceWindows(*args, **kw),
                v139.loader_factory(variant, False)(*args, **kw),
                v139.loader_factory(variant)(*args, **kw)]
    for epoch in (0, 1, 19):
        for index in (0, 1, 2):
            batches = []
            traces.clear()
            before = rng_state()
            for ds in datasets:
                ds.set_epoch(epoch)
                batches.append(ds[index])
                assert_tree_equal(before, rng_state())
            assert len(traces) == 3 and traces[0] == traces[1] == traces[2]
            a, b, c = batches
            images = c[0][0] if variant == 'POS' else c[0]
            assert torch.equal(a[0], b[0]) and torch.equal(a[0], images)
            assert torch.equal(a[1], b[1]) and torch.equal(a[1], c[1])
            assert a[2] == b[2] == c[2]
            if variant == 'POS':
                # Uses the centres ACTUALLY observed by augmentation, in order.
                expected = v139.slot_positions(traces[2][0], datasets[2].corpus.spec, 64)
                np.testing.assert_array_equal(c[0][1].numpy(), expected)


def test_horizontal_flip_preserves_position_and_sagittal_image(monkeypatch):
    from torchvision.transforms import functional as tf
    monkeypatch.setattr(tf, 'affine', lambda x, **kwargs: x)
    class FlipOnly:
        def __init__(self):
            self.n = 0
        def uniform(self, lo, hi):
            self.n += 1
            return (0., 1., 0., 0., 1.)[(self.n - 1) % 5]
        def random(self):
            return 0.  # always flip eligible plane slots
    centers = [43, 17, -1, 52, 0, 32]  # deliberately out of slot order
    x = np.arange(len(centers) * 3 * 8 * 8, dtype=np.float32).reshape(-1, 3, 8, 8) / 2000
    positions = v139.slot_positions(centers, dict(slots=SLOTS64), 64)
    y = aug.augment_windows(x, centers, FlipOnly(), dict(slots=SLOTS64))
    for j, center in enumerate(centers):
        np.testing.assert_array_equal(y[j], x[j, ..., ::-1] if center >= 32 else x[j])
    np.testing.assert_array_equal(positions, v139.slot_positions(centers, dict(slots=SLOTS64), 64))


@pytest.mark.parametrize('variant', v139.VARIANTS)
@pytest.mark.parametrize('seed', (42, 43))
def test_disabled_outputs_rng_optimizer_step_bitwise(variant, seed, corpus):
    path, ids, labels = corpus
    v139.seed_for_device(seed, 'cpu')
    base = FakeRaptor().train()
    off = copy.deepcopy(base)
    before = rng_state()
    assert v139.build_variant(off, variant, seed, enabled=False) is off
    assert_tree_equal(before, rng_state())
    assert_tree_equal(base.state_dict(), off.state_dict())
    args = (path, ids, labels, 8, 12, True, 'radimagenet', seed)
    kw = dict(window_policy='within-slot-valid')
    ds = [aug.AugmentedSliceWindows(*args, **kw), v139.loader_factory(variant, False)(*args, **kw)]
    optimizers, schedulers = [], []
    for model in (base, off):
        head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
        bb = [p for n, p in model.named_parameters() if n.startswith('backbone.')]
        opt = torch.optim.AdamW([dict(params=bb, lr=1e-4), dict(params=head, lr=1e-3)], weight_decay=.01)
        optimizers.append(opt)
        schedulers.append(torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=20, pct_start=.15))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.linspace(1., 3., 12))
    for epoch in (0, 1, 19):
        batches = []
        for dataset in ds:
            dataset.set_epoch(epoch)
            batches.append(v139.collate([dataset[0], dataset[1]]))
        assert_tree_equal(batches[0], batches[1])
        before = rng_state()
        records = []
        for i, model in enumerate((base, off)):
            restore_rng(before)
            x, y, _ = batches[i]
            opt, sched = optimizers[i], schedulers[i]
            opt.zero_grad(set_to_none=True)
            logits = model(v139.move_input(x, 'cpu'))
            loss = lossf(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            records.append((logits.detach(), loss.detach(), gradients, copy.deepcopy(model.state_dict()),
                            copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict()), rng_state()))
        assert_tree_equal(records[0], records[1])


@pytest.mark.parametrize('variant', v139.VARIANTS)
@pytest.mark.parametrize('seed', (42, 43))
def test_imported_head_initial_and_disabled_equivalence(variant, seed):
    v139.seed_for_device(seed, 'cpu')
    result = v139.head_equivalence(FakeRaptor(), variant, seed)
    assert all(c['exact_outputs'] and c['exact_original_gradients'] and c['exact_rng_state'] for c in result['checks'])


def test_late_observer_leaves_aug1_trajectory_bitwise(corpus, tmp_path):
    path, ids, labels = corpus
    v139.seed_for_device(42, 'cpu')
    models = [FakeRaptor().train()]
    models.append(copy.deepcopy(models[0]))
    opts = [torch.optim.AdamW(m.parameters(), lr=.001) for m in models]
    tracker = LateTracker(models[1], epochs=3, decay=.998)
    ds = aug.AugmentedSliceWindows(path, ids, labels, 8, 12, True, 'radimagenet', 42, window_policy='within-slot-valid')
    for epoch in range(3):
        ds.set_epoch(epoch)
        x, y, _ = v139.collate([ds[0], ds[1]])
        before = rng_state()
        records = []
        for i, model in enumerate(models):
            restore_rng(before)
            opts[i].zero_grad(set_to_none=True)
            logits = model(x)
            nn.functional.binary_cross_entropy_with_logits(logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            opts[i].step()
            if i == 1:
                tracker.update(model, successful=True)
                tracker.end_epoch(model, epoch + 1, tmp_path / 'late', {})
            records.append((logits.detach(), copy.deepcopy(model.state_dict()), copy.deepcopy(opts[i].state_dict()), rng_state()))
        assert_tree_equal(records[0], records[1])
    assert set(tracker.final_states()) == {'swa', 'ema'}
    assert tracker.metadata(expected_updates=3)['successful_updates'] == 3
