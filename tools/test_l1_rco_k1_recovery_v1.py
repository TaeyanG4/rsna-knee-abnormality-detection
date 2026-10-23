"""Bounded CPU replay parity and watchdog tests; no actual training/checkpoints."""
from types import SimpleNamespace
import random

import pytest
import torch

from tools import l1_rco_k1_recovery_v1 as r
from tools.l1_rco_k1_state_v1 import TrainerState, ThroughputGuard
from tools.l1_rco_runtime_v120 import stale_reason


class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 36

    def __getitem__(self, i):
        return i, torch.rand(3)


class Monitor:
    def __init__(self, out):
        self.out, self.phase_name, self.now = out, 'train', 0
        self.phases = []

    def emit(self):
        self.now += 200  # replay legitimately exceeds15 min in the test clock
        r.write(self.out / 'status.json', dict(time=self.now, phase=self.phase_name, steps=0, windows=0))

    def phase(self, name):
        self.phases.append(name)
        self.phase_name = name
        self.emit()


def make_state(path, skip):
    gen = torch.Generator().manual_seed(431)
    loader = torch.utils.data.DataLoader(Dataset(), batch_size=2, shuffle=True, generator=gen, num_workers=0)
    torch.manual_seed(111)
    saved = torch.get_rng_state().clone()
    return SimpleNamespace(dl=loader, epoch_loader=gen.get_state().clone(), batch=skip,
        monitor=Monitor(path), step_total=9100, saved_rng=saved,
        restore_rng=lambda: torch.set_rng_state(saved))


@pytest.mark.parametrize('skip', [0, 1, 8, 17])
def test_exact_remaining_data_shuffle_and_rng_match_original(tmp_path, skip):
    a = make_state(tmp_path / 'original', skip)
    original = [(i.clone(), x.clone()) for i, x in TrainerState.batches(a)]
    original_rng = torch.get_rng_state().clone()
    b = make_state(tmp_path / 'recovered', skip)
    recovered = [(i.clone(), x.clone()) for i, x in r.replay_batches(b)]
    assert len(original) == len(recovered)
    assert all(torch.equal(i, j) and torch.equal(x, y) for (i, x), (j, y) in zip(original, recovered))
    assert torch.equal(original_rng, torch.get_rng_state())
    assert torch.equal(a.dl.generator.get_state(), b.dl.generator.get_state())
    assert b.step_total == 9100 and b.batch == skip and b.saved_rng is None
    assert b.monitor.phases == (['resume-loader', 'train'] if skip else [])
    assert not torch.cuda.is_initialized()


def test_replay_progress_is_real_and_does_not_disable_stall_watchdog(tmp_path, monkeypatch):
    state = make_state(tmp_path, 8)
    observed = []
    original_write = r.write
    def capture(path, value):
        original_write(path, value)
        if 'resume_batches_completed' in value:
            observed.append(dict(value))
    monkeypatch.setattr(r, 'write', capture)
    list(r.replay_batches(state))
    assert [s['resume_batches_completed'] for s in observed] == list(range(1, 9))
    assert all(s['phase'] == 'resume-loader' and s['steps'] == 0 for s in observed)
    guard = ThroughputGuard(0)
    for status in observed:
        assert stale_reason(status, status['time'] + 14) is None
        guard.observe(status, status['time'])
    assert stale_reason(observed[-1], observed[-1]['time'] + 900) == '15-minute zero-progress watchdog'
    assert not guard.history


def test_loader_exception_propagates_without_train_heartbeat(tmp_path):
    state = make_state(tmp_path, 40)  # exceeds the actual loader length
    with pytest.raises(RuntimeError, match='generator raised StopIteration'):
        list(r.replay_batches(state))
    assert state.monitor.phases == ['resume-loader']
