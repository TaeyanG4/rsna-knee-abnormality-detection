"""CPU-only tests for EXP142's bounded MRNet rescue intervention."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'src'))
sys.path.insert(0, str(ROOT / 'training' / 'runners'))
from tools import exp142_mrn_rescue as guard
import exp142_mrn_rescue as runner
from raptor_common import SLOTS64, study_rng


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def forbidden(*_args, **_kwargs):
        raise AssertionError('CUDA was initialized by EXP142 test')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(torch.cuda, 'init', forbidden)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *_args, **_kwargs: None)
    yield
    assert not torch.cuda.is_initialized()


def test_external_bn_does_not_update_stats_and_restores_competition_path():
    torch.manual_seed(42)
    base = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 1))
    control, candidate = copy.deepcopy(base), copy.deepcopy(base)
    x1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    x2 = torch.tensor([[7., 8., 9.], [2., 3., 4.]])
    external = torch.tensor([[50., 60., 70.], [80., 90., 100.]])
    bn = candidate[1]
    assert torch.equal(control(x1), candidate(x1))
    before = (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
    with runner.external_bn_eval(candidate):
        assert candidate.training and not bn.training
        candidate(external).sum().backward()
        assert bn.weight.grad is not None and bn.bias.grad is not None
        assert torch.count_nonzero(bn.weight.grad) > 0
        assert all(torch.equal(a, b) for a, b in zip(
            (bn.running_mean, bn.running_var, bn.num_batches_tracked), before))
    assert candidate.training and bn.training
    assert torch.equal(control(x2), candidate(x2))
    control_bn = control[1]
    assert all(torch.equal(a, b) for a, b in zip(
        (bn.running_mean, bn.running_var, bn.num_batches_tracked),
        (control_bn.running_mean, control_bn.running_var, control_bn.num_batches_tracked)))


def test_external_aug1_keeps_competition_draw_stream_bitwise_identical():
    from raptor_exp127_aug1 import augment_windows
    spec = {'slots': SLOTS64}
    centers = [0, 44, 52]
    x = np.full((3, 3, 5, 5), .5, dtype=np.float32)
    competition_before = augment_windows(x, centers, study_rng(42, 2, 'COMPETITION_UID'), spec)
    control = study_rng(42, 2, 'COMPETITION_UID')
    first = [control.random() for _ in range(7)]
    expected_next = [control.random() for _ in range(7)]
    candidate = study_rng(42, 2, 'COMPETITION_UID')
    assert first == [candidate.random() for _ in range(7)]  # B fix: study_rng is stdlib random.Random
    changed = runner.external_aug1(x, centers, 42, 2, 'MRN_fake', spec)
    competition_after = augment_windows(x, centers, study_rng(42, 2, 'COMPETITION_UID'), spec)
    assert np.array_equal(competition_before, competition_after)
    assert changed.shape == x.shape and np.isfinite(changed).all()
    assert expected_next == [candidate.random() for _ in range(7)]
    assert np.array_equal(changed, runner.external_aug1(x, centers, 42, 2, 'MRN_fake', spec))
    assert not np.array_equal(x, changed)


def test_pin_memory_tuple_to_list_survives_move_to_device(monkeypatch):
    from torch.utils.data._utils.pin_memory import pin_memory
    from torch.utils.data import DataLoader
    from raptor_data import collate
    monkeypatch.setattr(torch.Tensor, 'pin_memory', lambda self, device=None: self)
    dataset = [(torch.ones(3), torch.zeros(1), 'a'),
               (torch.ones(3), torch.zeros(1), 'b')]
    original = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate, num_workers=0)))
    assert isinstance(original, tuple)
    pinned = pin_memory(original)
    assert isinstance(pinned, list)
    x, y, uids = runner.move_batch(pinned, 'cpu')
    assert torch.equal(x, original[0]) and torch.equal(y, original[1]) and uids == original[2]
    x2, y2, _ = runner.move_batch(original, 'cpu')
    assert torch.equal(x, x2) and torch.equal(y, y2)


def test_schedule_and_single_arm_queue(monkeypatch):
    from raptor_exp132 import aux_due
    assert [n for n in range(1, 13) if aux_due(n)] == [4, 8, 12]
    assert guard.RATIO == .25 and guard.LOSS_WEIGHT == .2
    assert guard.tag_for(42) == 'EXP142_MRN_RESCUE_S42_F0'
    with pytest.raises(ValueError):
        guard.tag_for(44)
    monkeypatch.setattr(guard, 'runner_argv', lambda seed, root=guard.ROOT, workers=12: ['python', 'runner', str(seed)])
    entry = guard.queue_entries(42, 'TAIL')
    assert len(entry) == 1 and entry[0]['tag'] == guard.tag_for(42)
    assert entry[0]['after'] == 'TAIL'


def test_registered_screen_rule_boundaries():
    def metric(a, b):
        return {'seed_mean': {'dread': {'delta': (a + b) / 2}},
                'per_seed': {'42': {'dread': {'delta': a}}, '43': {'dread': {'delta': b}}}}
    assert guard.screen_decision(metric(-.001, .002))[0] == 'ADMIT_FIVE_FOLD_DISCUSSION'
    assert guard.screen_decision(metric(-.0011, .003))[0] == 'CLOSE_AUXILIARY_HEAD_CONFIGURATION'
    assert guard.screen_decision(metric(0, 0))[0] == 'CLOSE_AUXILIARY_HEAD_CONFIGURATION'


def test_protocol_contract_is_one_mrn_arm(monkeypatch):
    source = {'variant': 'MRN', 'seed': 42, 'code_sha256': {}, 'fold_contract': {'steps': 10},
              'common_args': {'epochs': 1}, 'external': {'MRN': {'label_names': ['a', 'b', 'c']}}}
    monkeypatch.setattr(guard.old, 'make_protocol', lambda variant, seed, root=guard.ROOT: source)
    monkeypatch.setattr(guard.aug, 'sha256', lambda path: 'digest')
    p = guard.make_protocol(42)
    assert p['external'] == source['external'] and p['auxiliary_bce_weight'] == .2
    assert p['external_batches_per_competition_step'] == .25
    assert p['tag'] == guard.tag_for(42) and 'BatchNorm eval' in p['external_batchnorm']
    assert 'EXP127 AUG1' in p['external_augmentation']
