"""No optimizer, gradients, CUDA, network or project data mutation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from tools import hb112_bn_only_v1 as h


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


class Tiny(nn.Module):
    def __init__(self, mutate=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout(.25)
        self.register_buffer('geometry', torch.ones(2), persistent=False)
        self.mutate = mutate

    def forward(self, x):
        if self.mutate == 'buffer': self.geometry.add_(1)
        if self.mutate == 'weight': self.bn.weight.add_(1)
        if self.mutate == 'attribute': self.dropout.p = .5
        return self.dropout(self.bn(x.flatten(0, 1)))


def batches():
    return [(torch.arange(32, dtype=torch.float32).reshape(2,1,2,2,4),
             torch.zeros(2,12), torch.ones(2,12), ['train-a','train-b']),
            (torch.ones(1,1,2,2,4), torch.zeros(1,12), torch.ones(1,12), ['train-c'])]


def test_exact_no_gradient_replay_and_shared_rng():
    a, b = Tiny(), Tiny()
    b.load_state_dict(a.state_dict())
    torch.random.default_generator.manual_seed(8)
    ra = h.replay(a, batches(), 'cpu')
    torch.random.default_generator.manual_seed(997)
    rb = h.replay(b, batches(), 'cpu')
    assert ra['bn']['studies'] == 3
    assert ra['bn']['num_batches_tracked'] == [2]
    assert ra['bn']['studies_sha256'] == h.bn_uid_hash(['train-a','train-b','train-c'])
    assert ra['cpu_rng_before'] == rb['cpu_rng_before']
    assert ra['cpu_rng_after'] != ra['cpu_rng_before']  # train-mode dropout consumes RNG
    assert h.tensor_hash(a.state_dict().items()) == h.tensor_hash(b.state_dict().items())
    assert ra['non_bn_before'] == ra['non_bn_after']
    assert all(p.grad is None and not p.requires_grad for p in a.parameters())


@pytest.mark.parametrize('mutation', ['buffer','weight','attribute'])
def test_reject_any_non_bn_mutation(mutation):
    with pytest.raises(ValueError, match='changed'):
        h.replay(Tiny(mutation), batches(), 'cpu')


def test_nonpersistent_and_bn_affine_are_protected():
    model = Tiny()
    before = h.non_bn(model)
    model.bn.running_mean.add_(1)
    assert h.non_bn(model) == before
    model.geometry.add_(1)
    assert h.non_bn(model) != before
    assert 'geometry' not in model.state_dict()


def test_tensor_hash_frames_shapes_and_dtype():
    x = torch.ones(4, dtype=torch.float32)
    assert h.tensor_hash([('x',x)]) != h.tensor_hash([('x',x.reshape(2,2))])
    assert h.tensor_hash([('x',x)]) != h.tensor_hash([('x',x.view(torch.int32))])


def test_no_preexisting_gradients():
    model = Tiny()
    model.bn.weight.grad = torch.ones_like(model.bn.weight)
    with pytest.raises(ValueError, match='preexisting gradient'):
        h.replay(model, batches(), 'cpu')


def test_order_and_no_holdout_in_selected_cohort():
    p = dict(rsna_train_ids=['r2','r1'], oai_train_ids=['o1'],
             train_ids=['r2','r1','o1'], val_ids=['holdout'],
             train_labels={u:[0]*12 for u in ['r2','r1','o1']},
             train_masks={u:[1]*12 for u in ['r2','r1','o1']})
    selected = h.selection(p, ['r2','o1'])
    assert selected['train_ids'] == ['r2','o1']
    assert selected['rsna_train_ids'] == ['r2']
    assert selected['oai_train_ids'] == ['o1']
    assert p['train_ids'] == ['r2','r1','o1']
    assert list(selected['train_labels']) == list(selected['train_masks']) == ['r2','o1']
    for invalid in (['holdout'], ['r1','r1'], ['unknown']):
        with pytest.raises(ValueError, match='invalid BN training selection'):
            h.selection(p, invalid)


def review_fixture():
    reg = dict(builder='Codex sol-4')
    evidence = {'cpu.json':'e'*64}
    review = dict(verdict='PASS', reviewer='independent reviewer',
                  registration_sha256=h.life.content_sha256(reg), evidence=evidence,
                  approve_new_matched_policy=True, b_gpu_authorized=True)
    return reg, review, evidence


def test_review_and_b_gate_pass_only_bound_evidence():
    reg, review, evidence = review_fixture()
    assert h.admit(reg, review, evidence)['launch_authorized']
    assert not h.admit(reg, {}, evidence)['launch_authorized']
    assert not h.admit(reg, review, {'cpu.json':'f'*64})['launch_authorized']


def test_production_gate_precedes_cuda_and_model_loading(monkeypatch):
    reg, _, evidence = review_fixture()
    monkeypatch.setattr(h, 'validate', lambda r: {})
    monkeypatch.setattr(h, 'evidence_files', lambda: evidence)
    def forbidden(*args, **kwargs):
        raise AssertionError('CUDA/model access before admission')
    monkeypatch.setattr(h, 'observation', forbidden)
    monkeypatch.setattr(h.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='Independent reviewer PASS'):
        h.execute(reg, smoke=False)


@pytest.mark.parametrize('field,value', [('verdict','HOLD'),('reviewer','Codex sol-4'),
    ('reviewer',None),('registration_sha256','wrong'),('approve_new_matched_policy',False),
    ('b_gpu_authorized',False)])
def test_review_gate_refuses(field, value):
    reg, review, evidence = review_fixture()
    review[field] = value
    assert not h.admit(reg, review, evidence)['launch_authorized']


def test_immutable_registration(tmp_path):
    path = tmp_path/'registration.json'
    h.put(path, {'fixed':1})
    h.put(path, {'fixed':1})
    with pytest.raises(ValueError, match='immutable'):
        h.put(path, {'fixed':2})


def test_scratch_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match='outside fixed lane'):
        h.put(tmp_path/'oops.json', {}, scratch=True)


def test_swa_recovery_rejects_changed_member_before_load(monkeypatch):
    monkeypatch.setattr(h, 'sha', lambda p:'wrong')
    with pytest.raises(ValueError, match='SWA member changed'):
        h.recover()


def test_all_views_all_rulers_and_baseline_sensitivity(tmp_path):
    ids = [f'train-{i}' for i in range(40)]
    y = np.tile((np.arange(40)%2)[:,None],(1,12)).astype(float)
    refs = {r:(y, np.ones_like(y,dtype=bool),1) for r in h.ml.RULERS}
    ctx = h.ml.Context(ids, refs, [{'component_id':u,'members':[u]} for u in ids])
    base = np.random.default_rng(7).uniform(.1,.9,(40,12))
    predictions = {name:{v:base.copy() for v in h.VIEWS} for name in ('rsna','mixed','control','original')}
    predictions['rsna']['plain'] = 1-base
    for name in predictions:
        for view in h.VIEWS:
            path = tmp_path/f'{name}-{view}.csv'
            path.write_bytes(h.life.prediction_csv_bytes(ids,predictions[name][view],h.e.control.LAB))
            predictions[name][view] = h.life.load_prediction_csv(path,ids,h.e.control.LAB)
    result = h.compare(ctx,predictions)
    assert len(result) == 12
    for view in h.VIEWS:
        s = result['matched-mixed-minus-original-EXP190F/'+view]
        assert s['max_abs_probability_delta'] == 0
        assert set(s['multi_lens']['rulers']) == set(h.ml.RULERS)
    assert result['rsna-minus-mixed/plain']['max_abs_probability_delta'] > 0
    assert result['rsna-minus-mixed/mirror']['max_abs_probability_delta'] == 0
    # Sensitivity is measurable, not assumed identical.
    predictions['original']['mean'] = 1-base
    assert h.compare(ctx,predictions)['matched-mixed-minus-original-EXP190F/mean']['max_abs_probability_delta'] > 0


def test_prediction_exact_roster_no_intersection(tmp_path):
    path = tmp_path/'prediction.csv'
    path.write_bytes(h.life.prediction_csv_bytes(['a','b'],np.ones((2,12))*.5,h.e.control.LAB))
    with pytest.raises(ValueError, match='roster'):
        h.life.load_prediction_csv(path,['a','c'],h.e.control.LAB)


def test_policy_fixed_no_training_or_original_reproduction_claim():
    assert h.POLICY['workers'] <= 2 and h.POLICY['threads'] <= 2
    assert h.POLICY['gradients'] is False
    assert [n for n,_ in h.SWA] == [14,11,12]
    assert len(h.CONTRASTS) == 4 and h.POLICY['seed'] == 43112
