# Compatibility tests v2.0.0: retired launch guards, archived scientific parity, collected readouts.
"""Synthetic CPU checks for EXP185 spatial pooling and frozen fold-0 contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from training.src.raptor_exp185 import (
    collapsed_logits, finding_attention, install_spatial_pooling, spatial_tokens,
    valid_windows,
)
from tools import exp185_finding_mil as exp
from tools import exp170_convnext_t_l2 as base


class TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.head = nn.Module()
        self.head.norm = nn.Identity()

    def forward_features(self, x):
        return self.conv(nn.functional.adaptive_avg_pool2d(x, (2, 2)))

    def forward_head(self, x):
        return self.head.norm(x).mean((2, 3))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class TinyControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.norm = nn.LayerNorm(4)
        self.att = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Dropout(0), nn.Linear(6, 12))
        self.clsW = nn.Parameter(torch.randn(12, 4) * .02)
        self.clsb = nn.Parameter(torch.zeros(12))

    def encode(self, x):
        b, k = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).reshape(b, k, -1)

    def head(self, feats):
        h = self.norm(feats)
        weights = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', weights, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        return self.head(self.encode(x))


def sample():
    torch.manual_seed(42)
    x = torch.randn(2, 3, 3, 8, 8)
    background = (-torch.tensor((.485, .456, .406)) /
                  torch.tensor((.229, .224, .225))).view(3, 1, 1)
    x[0, 1] = background
    return x


def test_spatial_shapes_and_empty_window_mask():
    x = sample()
    model = TinyControl()
    tokens, mask = spatial_tokens(model, x)
    assert tokens.shape == (2, 12, 4)
    assert mask.shape == (2, 12)
    assert not mask[0, 4:8].any()
    assert mask[0, :4].all() and mask[0, 8:].all()
    assert torch.isfinite(install_spatial_pooling(model)(x)).all()
    assert model(x).shape == (2, 12)


def test_finding_attention_normalizes_independently_and_masks_padding():
    x = sample()
    model = TinyControl()
    tokens, mask = spatial_tokens(model, x)
    logits, weights = finding_attention(model, tokens, mask)
    assert logits.shape == (2, 12) and weights.shape == (2, 12, 12)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, 12))
    assert torch.count_nonzero(weights[0, 4:8]) == 0
    assert not torch.allclose(weights[0, :, 0], weights[0, :, 1])


def test_all_empty_study_is_finite_and_deterministic():
    x = sample()
    background = (-torch.tensor((.485, .456, .406)) /
                  torch.tensor((.229, .224, .225))).view(1, 1, 3, 1, 1)
    x[:] = background
    mask = valid_windows(x)
    assert mask[:, 0].all() and not mask[:, 1:].any()
    model = install_spatial_pooling(TinyControl())
    logits, weights = finding_attention(model, *spatial_tokens(model, x))
    assert torch.isfinite(logits).all()
    assert (weights[:, 4:] == 0).all()


def test_collapsed_grid_exact_control_and_identical_initial_state():
    x = sample()
    control = TinyControl().eval()
    before = {k: v.clone() for k, v in control.state_dict().items()}
    with torch.no_grad():
        reference = control(x)
        collapsed = collapsed_logits(control, x)
    torch.testing.assert_close(collapsed, reference, rtol=0, atol=0)
    install_spatial_pooling(control)
    assert before.keys() == control.state_dict().keys()
    for key, tensor in control.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(collapsed_logits(control, x), reference, rtol=0, atol=0)


def test_real_convnext_collapsed_grid_matches_control_normalization():
    from training.src.raptor_common import build_model

    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=False).eval()
    x = torch.randn(1, 2, 3, 64, 64)
    with torch.no_grad():
        baseline = model(x)
        collapsed = collapsed_logits(model, x)
        torch.testing.assert_close(collapsed, baseline, rtol=0, atol=0)
        install_spatial_pooling(model)
        assert spatial_tokens(model, x)[0].shape == (1, 8, model.backbone.num_features)
        assert model(x).shape == baseline.shape


def test_fold0_data_label_seed_and_historical_contract():
    overlay = exp.ROOT / '.local/exp185-overlay'
    manifest = exp.validate(overlay)
    for seed in exp.SEEDS:
        source = exp.read(exp.ROOT / exp.BASE_OVERLAY / f'protocols/M{seed}.json')
        control = exp.control_receipt(seed)
        candidate = exp.read(overlay / manifest['arms'][str(seed)]['protocol'])
        assert source['seed'] == control['summary']['args']['seed'] == seed
        assert candidate['paths'] == source['paths']
        assert candidate['label_set'] == source['label_set']
        assert source['label_set']['sha256'] == exp.sha(exp.ROOT / base.LABEL) == base.LABEL_SHA
        assert candidate['initial_state_sha256'] == source['initial_state_sha256'] == control['provenance']['initial_state_sha256']
        assert candidate['common_args'] == source['common_args']
        assert all(control['summary']['args'][key] == value for key, value in source['common_args'].items())
        assert candidate['folds']['0']['ordered_ids'] == source['folds']['0']['ordered_ids']
        assert source['folds']['0']['steps'] == 8620


@pytest.mark.parametrize('entry,args', [('prepare', (None, None)), ('runner_source', ()),
    ('runner_args', (42,)), ('admit', ()), ('run', (42,)), ('vast_stage', (None,))])
def test_fresh_execution_retired_without_writes(tmp_path, entry, args):
    with pytest.raises(ValueError, match='retired'):
        getattr(exp, entry)(*args)
    assert not list(tmp_path.iterdir())
    assert '__getattr__' not in vars(exp)


def test_historical_package_keeps_seed43_and_original_source_bytes():
    overlay = exp.ROOT / '.local/exp185-overlay'
    m = exp.validate(overlay)
    assert m['version'] == exp.HISTORICAL_VERSION
    assert m['files'][exp.TOOL]['sha256'] == exp.sha(overlay / exp.TOOL)
    assert m['files'][exp.TOOL]['sha256'] != exp.sha(exp.ROOT / exp.TOOL)
    assert m['files'][base.SOURCE_WEIGHT]['origin'] == 'exp179'
    assert m['files'][exp.MODEL]['origin'] == 'overlay'
    assert set(m['arms']) == {'42', '43'}


def test_review_requires_astra_and_final_exact_pass(tmp_path):
    hashes = '\n'.join(exp.sha(exp.ROOT / name) for name in
                       (exp.TOOL, exp.MODEL, exp.PROTOCOL))
    note = tmp_path / 'review.md'
    note.write_text(f'Reviewer: builder\n{hashes}\nVerdict: PASS-WITH-NOTES\n', encoding='utf-8')
    with pytest.raises(ValueError, match='independent astra review'):
        exp.check_verification(note)
    note.write_text(f'Reviewer: builder\n{hashes}\nVerdict: PASS\n', encoding='utf-8')
    with pytest.raises(ValueError, match='independent astra review'):
        exp.check_verification(note)
    note.write_text(f'Reviewer: Codex / astra\n{hashes}\nVerdict: PASS-WITH-NOTES\n',
                    encoding='utf-8')
    with pytest.raises(ValueError, match='final exact PASS'):
        exp.check_verification(note)
    note.write_text(f'Reviewer: builder\nReviewer: Codex / astra\n{hashes}\nVerdict: PASS\n',
                    encoding='utf-8')
    with pytest.raises(ValueError, match='independent astra review'):
        exp.check_verification(note)
    note.write_text(f'Reviewer: Codex / astra\n{hashes}\nVerdict: PASS\nNote: pending\n',
                    encoding='utf-8')
    with pytest.raises(ValueError, match='final exact PASS'):
        exp.check_verification(note)
    note.write_text(f'Reviewer: Codex / astra\n{hashes}\nVerdict: PASS\n', encoding='utf-8')
    assert exp.check_verification(note) == exp.sha(note)


def test_weak_primary_and_guard_on_synthetic_arrays():
    import numpy as np
    from tools.exp162_dino_folds_kaggle import TARGETS

    class ToyEvaluator:
        def auc_targets(self, y, p, mask, minimum):
            return (p[0] - p[1]) / 10

        def macro(self, v):
            return float(np.mean(v))

    n = 6
    control = np.full((n, 12), .5)
    candidate = control.copy()
    weak = ('Lateral Meniscus', 'PF OA', 'Lateral OA', 'Synovitis')
    for name in weak:
        candidate[0, TARGETS.index(name)] += .1
    refs = {name: (np.zeros((n, 12)), np.ones((n, 12), bool), 1)
            for name in ('jev', 'consensus', 'dread', 'aquila')}
    draws = [np.arange(n)] * 3
    result = exp.weak_target_readout(candidate, control, refs, draws, ToyEvaluator())
    assert result['primary_weak4_mean_delta'] > 0
    assert all(result['harm_guard'].values())
    assert result['numeric_gate'] == 'ADVANCE'
    assert result['status'] == 'REVIEW_IMAGE_READ'
    other = next(t for t in TARGETS if t not in weak)
    candidate[1, TARGETS.index(other)] += .2
    result = exp.weak_target_readout(candidate, control, refs, draws, ToyEvaluator())
    assert not result['harm_guard']['other_targets_ge_minus_0_010']
    assert result['status'] == 'HOLD_REVIEW'
