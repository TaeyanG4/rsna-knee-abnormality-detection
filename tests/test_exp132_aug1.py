"""CPU-only EXP132 on AUG1 integration checks."""
import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'src'))
from tools import exp132_aug1 as g
import raptor_exp132 as old
import raptor_exp132_aug1 as new


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


def test_protocol_keeps_external_contract_and_aug1_control(monkeypatch):
    def fake_external(variant, seed, root=g.ROOT):
        base = g.aug.e127.base_manifest(seed, root)
        return {'variant': variant, 'seed': seed, 'external': {variant: {'corpus': 'fake-cache', 'label_names': ['fake']}},
                'common_args': base['common_args'], 'code_sha256': {'existing': 'frozen'},
                'screen': {'rule': 'F0'}, 'fold_contract': base['folds']['0']}
    monkeypatch.setattr(g.ext, 'make_protocol', fake_external)
    for variant in g.VARIANTS:
        for seed in g.SEEDS:
            p = g.make_protocol(variant, seed)
            source = fake_external(variant, seed)
            control = g.aug.e127.load_protocol('AUG1', seed)
            assert p['external'] == source['external']
            assert p['external_batches_per_competition_step'] == .25
            assert p['auxiliary_bce_weight'] == .5
            assert p['base_tag'] == control['folds']['0']['tag']
            assert p['fold_contract'] == control['folds']['0']
            assert p['common_args'] == control['common_args']
            assert p['serving'] == 'competition model only; auxiliary heads dropped'


def test_external_window_and_25pct_schedule_unchanged():
    source = (ROOT / g.RUNNER).read_text(encoding='utf-8')
    assert 'from raptor_exp132 import (aux_due, competition_backward, make_aux_loaders,' in source
    assert 'from raptor_radimagenet import CenterSliceWindows  # external batches retain EXP132' in source
    assert 'CenterSliceWindows = loader_factory("AUG1")' in source
    for n in range(1, 17):
        assert new.aux_due(n) == old.aux_due(n) == (n % 4 == 0)
    assert source.count('auxiliary_backward(model, aux_heads[source]') == 1
    assert 'weight=.5' in source


def test_competition_step_bitwise_equal_when_external_disabled():
    class Scaler:
        def scale(self, loss):
            return loss
    def run(backward):
        torch.manual_seed(42)
        model = torch.nn.Linear(4, 12)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
        y = torch.arange(24, dtype=torch.float32).reshape(2, 12).remainder(2)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(12))
        loss = backward(model, x, y, lossf, Scaler(), 2, contextlib.nullcontext())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
        opt.step()
        return loss, [p.detach().clone() for p in model.parameters()]
    def aug1_direct(model, x, y, lossf, scaler, batch_size, autocast_context):
        with autocast_context:
            logits = model(x)
            assert logits.shape == (batch_size, 12)
            loss = lossf(logits.float(), y)
        scaler.scale(loss).backward()
        return loss
    a, av = run(aug1_direct)
    b, bv = run(old.competition_backward)
    assert a == b
    assert all(torch.equal(x, y) for x, y in zip(av, bv))


def test_queue_is_one_seed_and_does_not_launch(monkeypatch):
    monkeypatch.setattr(g, 'load_protocol', lambda variant, seed, root=g.ROOT: {'tag': g.tag_for(variant, seed)})
    entry = g.queue_entries('KMRI', 42, 'TAIL')
    assert len(entry) == 1 and entry[0]['tag'] == g.tag_for('KMRI', 42)
    assert entry[0]['after'] == 'TAIL'
    assert entry[0]['command'][1].endswith(g.RUNNER.replace('/', '\\')) or entry[0]['command'][1].endswith(g.RUNNER)
