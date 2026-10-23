"""CPU algebra/guard tests; no learned-model or GPU parity claim."""
import copy

import pytest
import torch
from torch import nn

from tools.tta_feature_reuse import mirror_pair_logits

torch.set_num_threads(1)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.att = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Dropout(.2), nn.Linear(3, 2))
        self.cls = nn.Parameter(torch.randn(2, 4))

    def forward(self, x):
        x = self.norm(x)
        att = self.att(x).softmax(dim=1)
        return (torch.einsum('bkn,bkf->bnf', att, x) * self.cls).sum(-1)


class Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Head()
        self.seen = []

    def encode(self, x):
        self.seen.append(x.detach().clone())
        # Location-dependent features detect wrong flip, scatter or bag order.
        return torch.stack((x[:, :, 0, 0, 0], x[:, :, 1, -1, -1],
                            x[:, :, 2, 1, 0], x.mean(dim=(2, 3, 4))), dim=-1)

    def forward(self, x):
        return self.head(self.encode(x))


class Multi(Single):
    def __init__(self):
        super().__init__()
        del self.head
        self.heads = nn.ModuleList([Head(), Head(), Head(), Head()])

    def forward(self, x):
        f = self.encode(x)
        return torch.stack([head(f) for head in self.heads], dim=1)


@pytest.mark.parametrize('kind', [Single, Multi])
@pytest.mark.parametrize('mask_kind', ['none', 'all', 'mixed'])
def test_full_bag_two_view_equivalence(kind, mask_kind):
    torch.manual_seed(42)
    model = kind().eval()
    baseline = copy.deepcopy(model)
    x = torch.randn(3, 5, 3, 4, 4)
    # Retain duplicate selected centres and empty sentinel; masks differ by UID.
    x[0, 2] = x[0, 1]
    x[1, 0] = 0
    flags = {'none': torch.zeros(3, 5, dtype=torch.bool),
             'all': torch.ones(3, 5, dtype=torch.bool),
             'mixed': torch.tensor([[False, True, True, False, True],
                                    [False, False, True, True, False],
                                    [True, False, False, False, True]])}[mask_kind]
    xm = x.clone()
    xm[flags] = torch.flip(x[flags], dims=(-1,))
    with torch.inference_mode():
        expected = baseline(x), baseline(xm)
    a, b, counts = mirror_pair_logits(model, x, flags, input_policy='center3')
    torch.testing.assert_close(a, expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b, expected[1], atol=1e-6, rtol=1e-6)
    assert counts['encoded_windows'] == 15 + int(flags.sum())
    assert counts['reused_windows'] == int((~flags).sum())
    assert counts['naive_windows'] == 30
    assert torch.equal(model.seen[0], x)
    assert len(model.seen) == 1 + bool(flags.any())
    if flags.any():
        assert torch.equal(model.seen[1][0], xm[flags])
    assert not a.requires_grad and not b.requires_grad


@pytest.mark.parametrize('error', ['training', 'child_training', 'batchnorm', 'policy', 'integer_flags', 'flag_shape', 'input_shape', 'nan'])
def test_reject_invalid_reuse(error):
    model = Single().eval()
    x = torch.randn(2, 4, 3, 4, 4)
    flags = torch.zeros(2, 4, dtype=torch.bool)
    policy = 'center3'
    if error == 'training': model.train()
    if error == 'child_training': model.head.att.train()
    if error == 'batchnorm': model.extra = nn.BatchNorm1d(4, track_running_stats=False).eval()
    if error == 'policy': policy = 'adjacent3'
    if error == 'integer_flags': flags = flags.int()
    if error == 'flag_shape': flags = flags[0]
    if error == 'input_shape': x = x[0]
    if error == 'nan': x[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        mirror_pair_logits(model, x, flags, input_policy=policy)


@pytest.mark.parametrize('multihead', [False, True])
def test_actual_raptor_resnet50_interfaces_without_checkpoint(multihead):
    from training.src.raptor_common import build_model
    from tools.exp117_multihead import build_multihead
    torch.manual_seed(43)
    model = build_model('resnet50', pretrained=False)
    if multihead:
        model = build_multihead(model, 4)
    model.eval()
    x = torch.randn(2, 4, 3, 32, 32)
    x[1, 0] = 0
    flags = torch.tensor([[False, True, False, True], [False, False, True, True]])
    xm = x.clone()
    xm[flags] = torch.flip(x[flags], dims=(-1,))
    with torch.inference_mode():
        expected_a, expected_b = model(x), model(xm)
    a, b, _ = mirror_pair_logits(model, x, flags, input_policy='center3')
    torch.testing.assert_close(a, expected_a, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b, expected_b, atol=1e-6, rtol=1e-6)
