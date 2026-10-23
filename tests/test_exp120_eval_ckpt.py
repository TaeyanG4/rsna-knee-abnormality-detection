"""CPU-only tests for the EXP120 checkpoint rescue."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from tools import exp120_eval_ckpt as ex


def test_checkpoint_contract():
    p = {'common_args': dict(arch='densenet121', res=224, norm='imagenet', window_policy='within-slot-valid',
                             input_policy='center3', k_eval=24, epochs=20),
         'folds': {'0': {'ordered_ids': {'val_sha256': 'ids'}}}}
    c = dict(tag='EXP120_DENSENET_LOCAL_F0', fold=0, arch='densenet121', res=224, norm='imagenet',
             window_policy='within-slot-valid', input_policy='center3', k_eval=24, epoch=19,
             provenance=dict(protocol_sha256='proto', eval_ids_ordered_sha256='ids'))
    old = ex.ex.sha256
    ex.ex.sha256 = lambda _path: 'proto'
    try:
        ex.checkpoint_contract(c, p)
        with pytest.raises(ValueError, match='input contract'):
            ex.checkpoint_contract({**c, 'norm': 'radimagenet'}, p)
    finally:
        ex.ex.sha256 = old


def test_fp32_tiny_cpu_evaluation_never_uses_cuda():
    class Tiny(torch.nn.Module):
        def forward(self, x):
            return x.reshape(len(x), -1)[:, :12]

    x = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 24
    loader = [(x, torch.zeros(2, 12), ['a', 'b'])]
    before = torch.cuda.is_initialized()
    p, ids = ex.evaluate_model(Tiny(), loader, 'cpu')
    assert ids == ['a', 'b']
    assert np.allclose(p, torch.sigmoid(x).numpy())
    assert torch.cuda.is_initialized() == before


def test_first_nonfinite_module_hook_on_cpu():
    class Overflow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Identity()
            self.bad = torch.nn.Identity()

        def forward(self, x):
            x = self.first(x)
            return self.bad(x * float('inf'))

    before = torch.cuda.is_initialized()
    result = ex.first_nonfinite_module(Overflow(), torch.ones(1, 12), device='cpu')
    assert result['first_nonfinite']['module'] == 'bad'
    assert result['output_finite'] is False
    assert torch.cuda.is_initialized() == before


def test_checked_predictions_aligns_tiny_uid_subset(tmp_path):
    path = tmp_path / 'oof.csv'
    df = pd.DataFrame(np.full((2, 12), .5), columns=ex.TARGETS)
    df.insert(0, ex.UID, ['b', 'a'])
    df.to_csv(path, index=False)
    arr = ex.checked_predictions(path, ['a', 'b'])
    assert arr.shape == (2, 12)
    with pytest.raises(ValueError, match='mismatch'):
        ex.checked_predictions(path, ['a', 'c'])


def test_queue_entry_is_independent_rescue():
    q = ex.queue_entries('PREVIOUS')
    assert len(q) == 1 and q[0]['after'] == 'PREVIOUS'
    assert q[0]['continue_on_failure'] is True
    assert q[0]['command'][-3:] == ['evaluate', '--device', 'cuda']
