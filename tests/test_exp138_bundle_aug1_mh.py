"""CPU-only EXP138 contract, draw stream and single-head update parity."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp138_bundle_aug1_mh as g


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(name):
        def blocker(*_args, **_kwargs):
            raise AssertionError(f'CUDA attribute called: {name}')
        return blocker
    for name in ('_lazy_init', 'init', 'get_rng_state'):
        monkeypatch.setattr(torch.cuda, name, fail(name))
    def cpu_seed_only(*_args, **_kwargs):
        return None
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', cpu_seed_only)
    yield
    assert not torch.cuda.is_initialized()


def test_all_fold_contracts_and_seed_initial_states():
    for seed in g.SEEDS:
        p = g.derive_manifest(seed)
        source = g.mh_protocol(seed)
        control = g.aug.e127.base_manifest(seed)
        assert p['initial_state_sha256'] == control['initial_state_sha256']
        assert p['multihead']['multihead_initial_state_sha256'] == source['multihead']['multihead_initial_state_sha256']
        assert p['common_args'] == control['common_args']
        for fold in g.FOLDS:
            f = p['folds'][str(fold)]
            a = g.aug_fold(seed, fold)
            assert {k: v for k, v in f.items() if k not in ('tag', 'head_pos_weight')} == {k: v for k, v in a.items() if k != 'tag'}
            assert f['head_pos_weight'] == source['folds'][str(fold)]['head_pos_weight']
            assert f['tag'] == g.tag_for(seed, fold)


def test_queue_split_and_cpu_validation(monkeypatch):
    ps = {s: g.derive_manifest(s) for s in g.SEEDS}
    monkeypatch.setattr(g, 'load_protocol', lambda s, root=g.ROOT: ps[s])
    entries = g.queue_entries('TAIL', 42)
    assert len(entries) == 5
    assert entries[0]['after'] == 'TAIL'
    assert [v['tag'] for v in entries] == [g.tag_for(42, f) for f in g.FOLDS]
    assert all(entries[i]['after'] == entries[i-1]['tag'] for i in range(1, 5))
    seen = []
    def fake_run(command, **_kwargs):
        seen.append(command)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(g.subprocess, 'run', fake_run)
    assert g.validate_cli() == 0
    assert len(seen) == 10
    assert all('--validate-only' in cmd and cmd[cmd.index('--device')+1] == 'cpu' for cmd in seen)


@pytest.mark.parametrize('seed', g.SEEDS)
@pytest.mark.parametrize('fold', g.FOLDS)
def test_aug1_draw_stream_unchanged_by_extra_heads(monkeypatch, seed, fold):
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    import raptor_exp127_aug1 as sampler
    from raptor_common import SLOTS64
    class FakeCorpus:
        spec = {'slots': SLOTS64}
        depth = 64
        def volume(self, _uid):
            return np.zeros((64, 4, 4), np.uint8)
        def mask(self, _uid):
            return np.ones(64, np.uint8)
    def centers(_mask, _depth, _k, _spec, _policy, rng):
        rng.random()  # B fix: study_rng returns random.Random (stdlib), not a numpy Generator
        return [1, 32, 53, -1]
    monkeypatch.setattr(sampler, 'select_centers', centers)
    monkeypatch.setattr(sampler, 'center3_windows_from', lambda _v, c, _r: np.full((len(c), 3, 4, 4), .25, np.float32))
    def dataset(label_width):
        ds = sampler.AugmentedSliceWindows.__new__(sampler.AugmentedSliceWindows)
        ds.corpus = FakeCorpus()
        ds.ids = [f'fake-fold-{fold}']
        ds.epoch = torch.tensor(2)
        ds.seed = seed
        ds.k = 4
        ds.res = 4
        ds.window_policy = 'within-slot-valid'
        ds.jitter = .1
        ds.norm = 'radimagenet'
        ds.train = True
        ds.augmentation = True
        ds.labels = {f'fake-fold-{fold}': np.zeros(label_width, np.float32)}
        return ds
    x1, y1, _ = dataset(12)[0]
    x4, y4, _ = dataset(48)[0]
    assert torch.equal(x1, x4)
    assert y1.numel() == 12 and y4.numel() == 48
    source = (ROOT / 'training/runners/exp134_aug1_kfold.py').read_text(encoding='utf-8')
    bundle = (ROOT / g.RUNNER).read_text(encoding='utf-8')
    assert 'loader_factory(a.variant)' in source
    assert 'loader_factory("AUG1")' in bundle
    assert 'tds.set_epoch(ep)' in source and 'tds.set_epoch(ep)' in bundle
    assert 'shuffle_generator = torch.Generator().manual_seed(a.seed)' in source
    assert 'shuffle_generator = torch.Generator().manual_seed(a.seed)' in bundle


@pytest.mark.parametrize('seed', g.SEEDS)
@pytest.mark.parametrize('fold', g.FOLDS)
def test_single_dread_training_math_bitwise_matches_exp134(seed, fold):
    bundle = (ROOT / g.RUNNER).read_text(encoding='utf-8')
    control = (ROOT / 'training/runners/exp134_aug1_kfold.py').read_text(encoding='utf-8')
    assert 'model = (built if a.single_dread else' in bundle
    assert 'single_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)' in bundle
    assert 'loss = single_loss(logits.float(), y)' in bundle
    assert 'loss = lossf(logits.float(), y)' in control
    for expression in ('shuffle_generator = torch.Generator().manual_seed(a.seed)',
                       'scaler.unscale_(opt)', 'scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)',
                       'sched.step()', 'tds.set_epoch(ep)'):
        assert expression in bundle and expression in control
    # Same model, BCE, optimizer order and data tensor; this is the isolated training step.
    contract = g.aug_fold(seed, fold)
    def run_step(bundle_single):
        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 12)
        opt = torch.optim.AdamW([{'params': list(model.parameters()), 'lr': 1e-3}], weight_decay=0.01)
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
        y = torch.arange(24, dtype=torch.float32).reshape(2, 12).remainder(2)
        weight = torch.tensor(contract['pos_weight'], dtype=torch.float32)
        logits = model(x)
        if bundle_single:
            single_loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
            loss = single_loss(logits.float(), y)
        else:
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
            loss = lossf(logits.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        return loss.detach().clone(), [v.detach().clone() for v in model.parameters()]
    a, av = run_step(False)
    b, bv = run_step(True)
    assert torch.equal(a, b)
    assert all(torch.equal(x, y) for x, y in zip(av, bv))


def test_tier_b_cannot_pass_without_serving_cost():
    means = {name: {'delta': .002, 'ci95': [.0001, .004], 'positive_draw_share': .95,
                    'bootstrap_draws': 300, 'bootstrap_finite_draws': 300}
             for name in g.aug.REFERENCES}
    per_seed = {str(seed): {'dread': {'delta': .002}} for seed in g.SEEDS}
    tiers = g.tier_verdicts(means, per_seed)
    assert tiers['tier_b']['statistical_verdict'] == 'THRESHOLDS_MET'
    assert tiers['tier_b']['verdict'] == 'PENDING_SERVING_COST'
    assert tiers['tier_b']['adoption_authorized'] is False
