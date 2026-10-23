"""CPU-only LATE tests; no data/cache/network reads and no CUDA initialization.

OMP_NUM_THREADS=2 PYTHONUTF8=1 Python312 -m pytest tests/test_exp129_late.py
--basetemp .local/pytest-exp129-late -p no:cacheprovider
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/src'))
from raptor_exp129_late import LateTracker, average_states, state_sha256


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def forbidden(*args, **kwargs):
        raise AssertionError('LATE tests must never initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(torch.cuda, 'init', forbidden)
    with torch.device('cpu'):
        yield
    assert not torch.cuda.is_initialized()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.bn = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(.3)
        self.register_buffer('bool_buffer', torch.tensor(True))

    def forward(self, x):
        return self.dropout(self.bn(self.linear(x)))


def assert_states_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key
        assert left[key].dtype == right[key].dtype


def complete(tracker, model, out):
    for epoch in range(1, tracker.epochs + 1):
        tracker.end_epoch(model, epoch, out, {'tag': 'synthetic', 'arch': 'tiny'})


def test_numeric_ema_skips_bn_buffers_and_no_mutation(tmp_path):
    model = TinyModel()
    start = copy.deepcopy(model.state_dict())
    tracker = LateTracker(model, epochs=3, decay=.998)
    with torch.no_grad():
        model.linear.weight.fill_(4)
        model.bn.running_mean.fill_(3)
        model.bn.running_var.fill_(5)
        model.bn.num_batches_tracked.fill_(7)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    tracker.update(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert_states_equal(before, model.state_dict())
    expected = start['linear.weight'] * .998 + torch.full_like(model.linear.weight, 4) * .002
    assert torch.allclose(tracker.ema['linear.weight'], expected, rtol=1e-7, atol=1e-7)
    assert torch.equal(tracker.ema['bn.running_mean'], model.bn.running_mean)
    assert torch.equal(tracker.ema['bn.running_var'], model.bn.running_var)
    assert tracker.ema['bn.num_batches_tracked'].item() == 7
    prior = copy.deepcopy(tracker.ema)
    with torch.no_grad():
        model.linear.weight.fill_(8)
        model.bn.running_mean.fill_(9)
        model.bn.num_batches_tracked.fill_(10)
    tracker.update(model, successful=False)
    assert_states_equal(prior, tracker.ema)
    tracker.update(model)
    assert torch.allclose(tracker.ema['linear.weight'], expected * .998 + 8 * .002)
    assert tracker.ema['bn.running_mean'].tolist() == [9., 9., 9.]
    assert tracker.ema['bn.num_batches_tracked'].item() == 10
    complete(tracker, model, tmp_path)
    report = tracker.metadata(expected_updates=2)
    assert report['attempted_updates'] == 3 and report['skipped_updates'] == 1
    assert report['successful_updates'] == 2
    assert 'no recalibration' in report['batchnorm_policy']
    assert report['scope'] == 'engineering_smoke_only'
    with pytest.raises(ValueError, match='expected count'):
        tracker.metadata(expected_updates=3)


def test_fp32_average_and_final_integer_bn_statistics():
    states = [{
        'weight': torch.tensor([value], dtype=torch.float16),
        'bn.running_mean': torch.tensor([value + 2], dtype=torch.float32),
        'bn.running_var': torch.tensor([value + 4], dtype=torch.float32),
        'bn.num_batches_tracked': torch.tensor(epoch, dtype=torch.int64),
        'enabled': torch.tensor(epoch % 2 == 0),
    } for epoch, value in enumerate([1., 2., 6.], 1)]
    original = copy.deepcopy(states)
    result = average_states(states)
    assert result['weight'].dtype == torch.float16 and result['weight'].item() == 3
    assert result['bn.running_mean'].item() == 5
    assert result['bn.running_var'].item() == 7
    assert result['bn.num_batches_tracked'].item() == 3
    assert not result['enabled'].item()
    for state, previous in zip(states, original, strict=True):
        assert_states_equal(state, previous)
    result['weight'].fill_(20)
    assert states[-1]['weight'].item() == 6


@pytest.mark.parametrize('change,match', [
    ('missing_key', 'keys'), ('shape', 'shape/dtype'), ('dtype', 'shape/dtype'),
    ('nonfinite', 'nonfinite'), ('overflow', 'nonfinite'), ('complex', 'complex'),
])
def test_average_rejects_invalid_state(change, match):
    states = [{'w': torch.tensor([1.])} for _ in range(3)]
    if change == 'missing_key':
        states[0]['extra'] = torch.tensor(1)
    elif change == 'shape':
        states[0]['w'] = torch.tensor([1., 2.])
    elif change == 'dtype':
        states[0]['w'] = states[0]['w'].double()
    elif change == 'nonfinite':
        states[1]['w'][0] = float('nan')
    elif change == 'overflow':
        states = [{'w': torch.tensor([3e38])} for _ in range(3)]
    else:
        states = [{'w': torch.tensor([1j])} for _ in range(3)]
    with pytest.raises(ValueError, match=match):
        average_states(states)


def test_checkpoint_roundtrip_and_exclusive_writes(tmp_path):
    model = TinyModel()
    tracker = LateTracker(model, epochs=20)
    meta = {'tag': 'test', 'epoch': -1, 'lab': ['synthetic'], 'provenance': {'seed': 42}}
    unchanged_meta = copy.deepcopy(meta)
    for ep in range(1, 21):
        with torch.no_grad():
            model.linear.weight.fill_(ep)
            model.bn.running_mean.fill_(ep * 2)
            model.bn.num_batches_tracked.fill_(ep)
        tracker.update(model)
        tracker.end_epoch(model, ep, tmp_path, meta)
    assert meta == unchanged_meta
    assert sorted(p.name for p in tmp_path.iterdir()) == ['epoch_18.pt', 'epoch_19.pt', 'epoch_20.pt']
    expected = tracker.final_states()
    assert expected['swa']['linear.weight'].flatten().tolist() == [19.] * 9
    assert expected['swa']['bn.running_mean'].tolist() == [38.] * 3
    assert expected['swa']['bn.num_batches_tracked'].item() == 20
    assert expected['ema']['bn.running_mean'].tolist() == [40.] * 3
    report = tracker.save_final(tmp_path, meta, expected_updates=20)
    for ep in (18, 19, 20):
        payload = torch.load(tmp_path / f'epoch_{ep}.pt', weights_only=True)
        assert payload['epoch'] == ep - 1
        assert payload['tag'] == 'test'
        assert payload['late']['state_sha256'] == state_sha256(payload['model'])
        assert torch.equal(payload['model']['linear.weight'], torch.full((3, 3), float(ep)))
    for arm in ('swa', 'ema'):
        checkpoint = torch.load(tmp_path / f'model_{arm}.pt', weights_only=True)
        assert_states_equal(expected[arm], checkpoint['model'])
        assert checkpoint['late']['state_sha256'] == report['final_state_sha256'][arm]
        rebuilt = TinyModel()
        rebuilt.load_state_dict(checkpoint['model'], strict=True)
        assert_states_equal(rebuilt.state_dict(), checkpoint['model'])
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        tracker.save_final(tmp_path, meta, expected_updates=20)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    assert before == after
    expected['ema']['linear.weight'].zero_()
    assert state_sha256(tracker.final_states()['ema']) == report['final_state_sha256']['ema']


def test_ordered_training_trajectory_with_and_without_tracker(tmp_path):
    """Identical shuffle, dropout, BN, SGD and scheduler despite checkpoint tracking."""
    def run(tracked):
        torch.manual_seed(42)
        model = TinyModel().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
        tracker = LateTracker(model, epochs=3) if tracked else None
        x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
        history = []
        for ep in range(1, 4):
            order = torch.randperm(8)
            for indices in order.reshape(2, 4):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x[indices])
                loss = prediction.square().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if tracker:
                    tracker.update(model)
                history.append((order.clone(), loss.detach().clone(), copy.deepcopy(model.state_dict()),
                                torch.get_rng_state().clone()))
            if tracker:
                tracker.end_epoch(model, ep, tmp_path, {'tag': 'synthetic'})
        if tracker:
            tracker.save_final(tmp_path, {'tag': 'synthetic'}, expected_updates=6)
        return history, copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict()), torch.get_rng_state().clone()
    control, control_opt, control_sched, control_rng = run(False)
    variant, variant_opt, variant_sched, variant_rng = run(True)
    for left, right in zip(control, variant, strict=True):
        assert torch.equal(left[0], right[0])
        assert torch.equal(left[1], right[1])
        assert_states_equal(left[2], right[2])
        assert torch.equal(left[3], right[3])
    assert control_sched == variant_sched
    assert control_opt['param_groups'] == variant_opt['param_groups']
    for key in control_opt['state']:
        assert_states_equal(control_opt['state'][key], variant_opt['state'][key])
    assert torch.equal(control_rng, variant_rng)


def test_invalid_or_missing_epochs_and_changed_state(tmp_path):
    model = TinyModel()
    for epochs in (0, 2, True, 3.1):
        with pytest.raises(ValueError):
            LateTracker(model, epochs=epochs)
    for decay in (-1, 0, 1, float('nan')):
        with pytest.raises(ValueError):
            LateTracker(model, decay=decay)
    tracker = LateTracker(model, epochs=3)
    with pytest.raises(ValueError, match='three epochs'):
        tracker.final_states()
    with pytest.raises(ValueError, match='consecutive order'):
        tracker.end_epoch(model, 2, tmp_path, {})
    tracker.end_epoch(model, 1, tmp_path, {})
    with pytest.raises(ValueError, match='consecutive order'):
        tracker.end_epoch(model, 1, tmp_path, {})
    with pytest.raises(ValueError, match='explicit bool'):
        tracker.update(model, successful=1)
    changed = TinyModel().double()
    with pytest.raises(ValueError, match='shape/dtype'):
        tracker.update(changed)
    tracker.end_epoch(model, 2, tmp_path, {})
    tracker.end_epoch(model, 3, tmp_path, {})
    with pytest.raises(ValueError, match='after final epoch'):
        tracker.update(model)
    assert tracker.metadata(expected_updates=0)['successful_updates'] == 0


def test_hash_matches_registered_raw_bytes_and_bfloat16_support():
    state = {'z': torch.tensor(1, dtype=torch.int64), 'a': torch.tensor([1., 2.])}
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    assert state_sha256(state) == digest.hexdigest()
    assert len(state_sha256({'b': torch.tensor([1.], dtype=torch.bfloat16)})) == 64
