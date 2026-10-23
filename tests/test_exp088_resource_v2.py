"""New resource decisions and atomic snapshots; no corpus/model/GPU accesses."""
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import pytest

from tools.exp088_resource_v2 import atomic_write, limit_reason, memory_sample, supervise


def test_shared_rss_does_not_trip_private_gate():
    sample = dict(tree_rss_bytes=150 * 2**30, tree_private_commit_bytes=6 * 2**30,
                  host_available_bytes=30 * 2**30)
    assert limit_reason(sample, 1, 20, 24 * 2**30, 8 * 2**30) is None


@pytest.mark.parametrize('sample,elapsed,reason', [
    (dict(tree_private_commit_bytes=11, host_available_bytes=20), 1, 'private_commit_limit'),
    (dict(tree_private_commit_bytes=1, host_available_bytes=4), 1, 'host_available_floor'),
    (dict(tree_private_commit_bytes=1, host_available_bytes=20), 30, 'time_limit'),
])
def test_independent_limits(sample, elapsed, reason):
    assert limit_reason(sample, elapsed, 20, 10, 5) == reason


def test_atomic_failure_preserves_previous(tmp_path):
    out = tmp_path / 'latest.pt'
    out.write_bytes(b'previous complete snapshot')
    def fail(value, stream):
        stream.write(b'partial')
        raise OSError('synthetic disk error')
    with pytest.raises(OSError):
        atomic_write(out, {}, fail)
    assert out.read_bytes() == b'previous complete snapshot'
    assert not out.with_name(out.name + '.tmp').exists()


def test_atomic_replacement_roundtrip(tmp_path):
    out = tmp_path / 'latest.pt'
    for epoch in (1, 2):
        atomic_write(out, dict(next_epoch=epoch, optimizer_updates=432 * epoch), pickle.dump)
    assert pickle.loads(out.read_bytes()) == dict(next_epoch=2, optimizer_updates=864)


def test_missing_private_counter_fails():
    class P:
        def create_time(self): return 1
        def memory_info(self): return SimpleNamespace(rss=999)
    with pytest.raises(RuntimeError, match='counter unavailable'):
        memory_sample([((1, 1), P())], SimpleNamespace(NoSuchProcess=ProcessLookupError))


@pytest.mark.skipif(os.name != 'nt', reason='Windows metric contract')
@pytest.mark.parametrize('private_cap,available_floor,expected', [
    (24 * 2**30, 1, 'PASS'), (1, 1, 'FAIL'), (24 * 2**30, 2**60, 'FAIL')])
def test_owned_child_lifecycle(tmp_path, private_cap, available_floor, expected):
    import json
    env = {**os.environ, 'EXP087_PHASE': 'synthetic', 'EXP087_PHASE_AUTHORIZATION_SHA256': 'cpu-only'}
    command = [sys.executable, '-c', 'import time; time.sleep(.4)']
    if expected == 'FAIL':
        with pytest.raises(RuntimeError, match='no retry'):
            supervise(command, tmp_path, 20, private_cap, available_floor, env, 'synthetic')
    else:
        supervise(command, tmp_path, 20, private_cap, available_floor, env, 'synthetic')
    receipt = json.loads((tmp_path / 'resource-supervisor.json').read_text())
    assert receipt['status'] == expected
    assert not receipt['cleanup_survivors'] and not receipt['cleanup_errors']
    assert receipt['peak_tree_private_commit_bytes'] > 0


def test_torch_snapshot_preserves_rng_and_tensors(tmp_path):
    import torch
    from tools.exp088_resource_v2 import save_recovery
    before = torch.get_rng_state().clone()
    payload = dict(model={'test_tensor': torch.arange(3)}, scheduler={'last_epoch': 432},
                   torch_rng=before, next_epoch=1, no_auto_resume=True)
    save_recovery(tmp_path, torch, payload)
    actual = torch.load(tmp_path / 'recovery_latest.pt', weights_only=True)
    assert torch.equal(actual['model']['test_tensor'], payload['model']['test_tensor'])
    assert torch.equal(actual['torch_rng'], before)
    assert torch.equal(torch.get_rng_state(), before)

