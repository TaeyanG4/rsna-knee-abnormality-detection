"""Independent policy coverage and actual AdamW behavior for K1 v3-G."""
import copy
import os
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pytest
import torch
from torch import nn
from tools import l1_rco_k1_optim_v3 as op


class ConvNeXtBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(3))
        self.conv = nn.Conv2d(3, 3, 1)
        self.norm = nn.LayerNorm([3, 2])  # multidimensional norm cannot rely on ndim<=1


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ConvNeXtBlock()
        self.sp_head_norm = nn.LayerNorm(3)
        self.proj = nn.Linear(3, 3)
        self.slot = nn.Embedding(4, 3)
        self.sp_q = nn.Parameter(torch.ones(2, 3))
        self.q_win = nn.Parameter(torch.ones(2, 3))
        self.cls_w = nn.Parameter(torch.ones(2, 3))
        self.bias = nn.Parameter(torch.ones(3))
        self.scalar = nn.Parameter(torch.ones(()))
        self.vector = nn.Parameter(torch.ones(3))
        self.out_norm = nn.LayerNorm([3, 2])
        self.frozen = nn.Parameter(torch.ones(2, 3), requires_grad=False)


EXPECTED = {
    'pretrained_decay': {'enc.conv.weight'},
    'pretrained_no_decay': {'enc.gamma', 'enc.conv.bias', 'enc.norm.weight', 'enc.norm.bias',
                            'sp_head_norm.weight', 'sp_head_norm.bias'},
    'head_decay': {'proj.weight', 'slot.weight', 'sp_q', 'q_win', 'cls_w'},
    'head_no_decay': {'proj.bias', 'bias', 'scalar', 'vector', 'out_norm.weight', 'out_norm.bias'},
}


def test_explicit_membership_identity_and_full_manifest():
    model = Fixture(); groups, manifest = op.parameter_groups(model)
    assert [g['group_name'] for g in groups] == list(EXPECTED)
    assert {g: {r['name'] for r in manifest['parameters'] if r['group']==g} for g in EXPECTED} == EXPECTED
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == 18
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert manifest['frozen_parameters'] == ['frozen']
    for row in manifest['parameters']:
        param = dict(model.named_parameters())[row['name']]
        assert row['shape'] == list(param.shape) and row['numel'] == param.numel()
    assert [g['lr'] for g in groups] == [1e-4, 1e-4, 3e-4, 3e-4]
    assert [g['weight_decay'] for g in groups] == [.02, 0, .02, 0]
    reasons = {r['name']: r['reasons'] for r in manifest['parameters']}
    assert 'normalization' in reasons['enc.norm.weight']
    assert 'LayerScale' in reasons['enc.gamma'] and 'bias' in reasons['bias']
    assert op.manifest_digest(manifest) == op.manifest_digest(copy.deepcopy(manifest))


@pytest.mark.parametrize('kw', [{'lr':5e-5}, {'head_lr':1e-3}, {'weight_decay':.01}, {'llrd':.9}])
def test_rejects_unregistered_recipe(kw):
    with pytest.raises(ValueError, match='fixed v2'): op.parameter_groups(Fixture(), **kw)


def test_rejects_trainable_alias_and_unknown_norm_or_layerscale():
    model=Fixture(); model.alias=model.proj.weight
    with pytest.raises(ValueError, match='duplicate'): op.parameter_groups(model)
    class UnknownNorm(nn.Module):
        def __init__(self):
            super().__init__(); self.weight=nn.Parameter(torch.ones(2, 3))
    model=Fixture(); model.mystery=UnknownNorm()
    with pytest.raises(ValueError, match='unclassified normalization'): op.parameter_groups(model)
    class UnknownLayerScale(nn.Module):
        def __init__(self):
            super().__init__(); self.weight=nn.Parameter(torch.ones(3))
    model=Fixture(); model.mystery=UnknownLayerScale()
    with pytest.raises(ValueError, match='unclassified LayerScale'): op.parameter_groups(model)
    model.mystery.gamma=model.mystery.weight; del model.mystery.weight
    with pytest.raises(ValueError, match='unclassified LayerScale'): op.parameter_groups(model)
    model=Fixture(); model.sp_head_norm=nn.Linear(3,3)
    with pytest.raises(ValueError, match='relocated pretrained norm'): op.parameter_groups(model)


def test_recognized_timm_layerscale():
    from timm.layers import LayerScale
    model=Fixture(); model.scale=LayerScale(3)
    _,manifest=op.parameter_groups(model)
    row=next(r for r in manifest['parameters'] if r['name']=='scale.gamma')
    assert row['group']=='head_no_decay' and row['weight_decay']==0. and 'LayerScale' in row['reasons']


def test_real_adamw_step_lr_and_weight_decay():
    model=Fixture().double()
    with torch.no_grad():
        for p in model.parameters(): p.fill_(1)
    groups,_=op.parameter_groups(model); opt=torch.optim.AdamW(groups)
    # A zero gradient isolates decoupled decay (None would skip the parameter).
    for p in model.parameters():
        if p.requires_grad: p.grad=torch.zeros_like(p)
    opt.step()
    for group,names in EXPECTED.items():
        lr=1e-4 if group.startswith('pretrained') else 3e-4
        decay=0. if group.endswith('no_decay') else .02
        for name in names:
            assert torch.allclose(dict(model.named_parameters())[name], torch.tensor(1-lr*decay,dtype=torch.float64),atol=1e-14,rtol=0)
    # First moment update with fresh state has the same normalized gradient for
    # relocated norm and head bias; their update ratio must be 1:3.
    model=Fixture().double(); groups,_=op.parameter_groups(model); opt=torch.optim.AdamW(groups)
    a=model.sp_head_norm.bias; b=model.bias
    before_a=a.detach().clone(); before_b=b.detach().clone()
    a.grad=torch.ones_like(a); b.grad=torch.ones_like(b); opt.step()
    assert torch.allclose((before_b-b),3*(before_a-a),atol=1e-14,rtol=0)


def test_manifest_resume_rejects_changed_shape(tmp_path):
    args=SimpleNamespace(lr=1e-4,head_lr=3e-4,wd=.02,llrd=1.,out=tmp_path)
    model=Fixture(); op.training_groups(model,args); op.training_groups(model,args)
    model.slot=nn.Embedding(5,3)
    with pytest.raises(ValueError,match='manifest differs'): op.training_groups(model,args)


def test_four_group_state_roundtrip_and_old_two_group_rejected():
    model=Fixture(); groups,_=op.parameter_groups(model); opt=torch.optim.AdamW(groups)
    for p in model.parameters():
        if p.requires_grad: p.grad=torch.ones_like(p)
    opt.step(); state=opt.state_dict()
    other=Fixture(); groups,_=op.parameter_groups(other); restored=torch.optim.AdamW(groups)
    restored.load_state_dict(state)
    assert [g['group_name'] for g in restored.param_groups] == list(EXPECTED)
    assert len(restored.state)==18 and all(v['step'].item()==1 for v in restored.state.values())
    old=torch.optim.AdamW([{'params':list(other.enc.parameters())},
                          {'params':[p for n,p in other.named_parameters() if not n.startswith('enc.')]}])
    with pytest.raises(ValueError,match='different number'): restored.load_state_dict(old.state_dict())
