"""Synthetic loss/parity/readout tests; metadata-only registered-spec checks."""
import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from training.arms import weak_rank_v1 as rank
from training.arms import mil_screen_v1 as run
from tools import hb122_build_v1 as build, hb122_screen_v1 as screen

torch.set_num_threads(2)


def batch(n=8):
    y = torch.zeros(n, 12)
    for t, nuisance in rank.MAPPING:
        y[:, t] = torch.tensor(([0., 1.] * 4)[:n])
        y[:, nuisance] = torch.tensor(([0., 0., 1., 1.] * 2)[:n])
    return y, [str(i) for i in range(n)]


def test_formula_gap_weight_and_equal_strata():
    y, groups = batch()
    y[:, 3] = torch.tensor([0., .5, .1, .9, .2, .8, .3, 1.])
    z = torch.arange(96, dtype=torch.float32).reshape(8, 12) / 40
    actual, coverage = rank.ranking_loss(z, y, groups)
    target_terms = []
    for t, nuisance in rank.MAPPING:
        all_pairs, lo, hi = [], [], []
        for i in range(8):
            for j in range(8):
                gap = float(y[i, t] - y[j, t])
                if gap < .5 or groups[i] == groups[j]:
                    continue
                term = gap * math.log1p(math.exp(-float(z[i, t] - z[j, t])))
                all_pairs.append(term)
                if y[i, nuisance] < .5 and y[j, nuisance] < .5: lo.append(term)
                if y[i, nuisance] >= .5 and y[j, nuisance] >= .5: hi.append(term)
        target_terms.append(.5*np.mean(all_pairs)+.25*np.mean(lo or all_pairs)+.25*np.mean(hi or all_pairs))
        assert coverage[str(t)]['eligible'] == len(all_pairs)
    assert actual.item() == pytest.approx(np.mean(target_terms), rel=1e-6)


@pytest.mark.parametrize('extreme', [0., 1e4, -1e4])
def test_finite_gradients_and_nonweak_isolation(extreme):
    y, groups = batch()
    z = (torch.arange(96).reshape(8,12).float()*extreme).requires_grad_()
    loss, cov = rank.ranking_loss(z, y, groups)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert (z.grad[:, [0,1,2,4,7,9,10,11]] == 0).all()
    assert all(v['eligible'] and v['low'] and v['high'] for v in cov.values())


def test_pair_exclusion_gap_boundary_and_nuisance_boundary():
    y, groups = batch(4)
    y[:, 3] = torch.tensor([1., .5, .49, 0.])
    y[:, 2] = torch.tensor([.5, .5, .499, .499])
    groups = ['a','a','b','c']
    sets, cov = rank.pair_sets(y, groups)
    pairs = lambda p: set(zip(p[0].tolist(), p[1].tolist()))
    assert pairs(sets[3][0]) == {(0,2),(0,3),(1,3)}
    assert not pairs(sets[3][1]) and not pairs(sets[3][2])
    assert cov['3']['fallback_strata'] == 2


def test_fixed_four_denominator_gap_not_normalized_and_fallback():
    y = torch.zeros(2,12); y[0,3] = .5
    z = torch.zeros_like(y, requires_grad=True)
    loss, cov = rank.ranking_loss(z, y, ['a','b'])
    assert loss.item() == pytest.approx(.5*math.log(2)/4)
    assert cov['3'] == dict(eligible=1,low=1,high=0,fallback_strata=1)
    loss.backward()
    assert z.grad[0,3].item() == pytest.approx(-.5*.5/4)


@pytest.mark.parametrize('n', [1,2,8])
def test_empty_pairs_differentiable_zero(n):
    y, _ = batch(n)
    z = torch.ones_like(y, requires_grad=True)
    loss, cov = rank.ranking_loss(z, y, ['one']*n)
    loss.backward()
    assert loss.item() == 0 and (z.grad == 0).all()
    assert all(v['eligible'] == 0 for v in cov.values())


def test_per_batch_coverage_bound_and_no_rng_consumption():
    y, groups = batch()
    states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    _, cov = rank.ranking_loss(torch.zeros_like(y), y, groups)
    assert all(0 < c['eligible'] <= 28 and c['low'] and c['high'] for c in cov.values())
    assert torch.equal(states[0], torch.get_rng_state()) and states[1] == random.getstate()
    np.testing.assert_array_equal(states[2][1], np.random.get_state()[1])


@pytest.mark.parametrize('mode', ['nan','infinite','range','shape','batch9','component','logit'])
def test_invalid_batches_fail_closed(mode):
    y, groups = batch()
    z = torch.zeros_like(y)
    if mode == 'nan': y[0,0] = float('nan')
    if mode == 'infinite': y[0,0] = float('inf')
    if mode == 'range': y[0,0] = 1.01
    if mode == 'shape': y = y[:,:11]; z = z[:,:11]
    if mode == 'batch9': y = torch.zeros(9,12); z = y.clone(); groups = ['a']*9
    if mode == 'component': groups[0] = ''
    if mode == 'logit': z[0,0] = float('nan')
    with pytest.raises(ValueError): rank.ranking_loss(z,y,groups)


def test_disabled_objective_bit_exact():
    y, groups = batch()
    z = torch.linspace(-2,2,96).reshape(8,12).requires_grad_()
    w = torch.arange(1,13).float()
    expected = F.binary_cross_entropy_with_logits(z.float(), y, pos_weight=w)
    actual, cov = rank.objective(z,y,w,groups,enabled=False)
    assert torch.equal(expected, actual) and cov == {}
    assert torch.equal(torch.autograd.grad(expected,z,retain_graph=True)[0], torch.autograd.grad(actual,z)[0])


def trained(which):
    run.seed_all(42)
    model = torch.nn.Sequential(torch.nn.Linear(4,24),torch.nn.Dropout(.3),torch.nn.Linear(24,12))
    opt, sched, scaler = run.optimizer_stack(model,3,'cpu','none')
    y, ids = batch()
    x = torch.arange(32).reshape(8,4).float()/10
    loader = [(x,y,ids)]*3
    count = dict(updates=0,skips=0)
    args = (model,loader,opt,sched,scaler,torch.ones(12),'cpu','none',0,count)
    extra = {} if which == 'old' else dict(components={u:u for u in ids},enabled=which=='rank')
    result = (run.deployed.train_epoch if which == 'old' else run.train_epoch)(*args,**extra)
    return model.state_dict(), opt.state_dict(), sched.state_dict(), count, result, torch.get_rng_state(), np.random.get_state(), random.getstate()


def equal_tree(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys() == b.keys()
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b): equal_tree(x,y)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    else: assert a == b


def test_control_step_optimizer_scheduler_rng_exact_deployed_equality():
    equal_tree(trained('old'),trained('control'))


def test_arm_keeps_rng_and_update_exposure():
    c,a = trained('control'),trained('rank')
    equal_tree(c[5:],a[5:]); equal_tree(c[2:4],a[2:4])
    assert any(not torch.equal(c[0][k],a[0][k]) for k in c[0])
    assert a[4]['batches'] == 3
    assert all(v['batches_with_pairs']==3 for v in a[4]['pair_coverage'].values())


@pytest.mark.parametrize('seed',[42,43])
def test_registered_config_source_initialization_exact_deployed_parity(seed):
    original = run.life.ArmSpec((build.ROOT/f'training/configs/hb105-small-s{seed}-f0-v1.json').read_bytes()).document
    control, arm = build.spec_for('control',seed), build.spec_for('rank',seed)
    for spec in (control,arm):
        run.validate_recipe(spec)
        run.life.validate_inputs(spec,build.ROOT)
        for p,h in original['inputs']['sources'].items(): assert run.sha(build.ROOT/p) == h
    assert control.document['config'] == original['config']
    changed = copy.deepcopy(arm.document['config'])
    assert changed['updates'].pop('weak_rank_v1') == rank.CONTRACT
    assert changed == original['config']
    assert control.document['target_environment'] == arm.document['target_environment']
    assert control.document['inputs']['sources'] == arm.document['inputs']['sources']


def receipt_fixture():
    ids=['a','b']; recipe={'key':'control-s42'}
    doc={'sources':{'a':'1'*64},'panel_pins':{'b':'2'*64}}
    r=dict(version=screen.VERSION,recipe=recipe,sources=doc['sources'],panel_pins=doc['panel_pins'],
           recipe_file_sha256=run.life.content_sha256(doc),predictions_sha256='3'*64,rows=2,
           ids_sha256=run.uid_hash(ids),engineering_smoke=False,full_panel=True,device='cuda',volume_files_verified=2)
    return r,recipe,doc,ids,'3'*64


@pytest.mark.parametrize('field,value', [('rows',1),('device','cpu'),('full_panel',False),('engineering_smoke',True),
    ('predictions_sha256','4'*64),('recipe',{'key':'rank-s42'}),('recipe_file_sha256','4'*64),('volume_files_verified',0)])
def test_panel_partial_tamper_swapped_member_fail_closed(field,value):
    args = receipt_fixture()
    screen.authenticate_panel_receipt(*args)
    args[0][field]=value
    with pytest.raises(run.life.ContractError): screen.authenticate_panel_receipt(*args)


def test_prediction_csv_shape_roster_and_probabilities(tmp_path):
    p=tmp_path/'predictions.csv'; ids=['a','b']; a=np.full((2,12),.3,np.float32)
    p.write_bytes(run.life.prediction_csv_bytes(ids,a,run.TARGETS,id_column=run.UID))
    np.testing.assert_array_equal(screen.load_predictions(p,ids),a)
    with pytest.raises(run.life.ContractError): screen.load_predictions(p,list(reversed(ids)))
    p.write_text(p.read_text().replace('0.3','nan'))
    with pytest.raises(run.life.ContractError): screen.load_predictions(p,ids)


def test_fixed_bundle_only_replaces_fold0_before_pooled_ranks():
    raw={f:{k:np.full((3,12),f/5,np.float32) for k in ('S42','S43','R42')} for f in range(5)}
    before=copy.deepcopy(raw); calls=[]
    arm={s:np.full((3,12),.9,np.float32) for s in ('42','43')}
    control={s:np.full((3,12),.1,np.float32) for s in ('42','43')}
    def variants(value):
        calls.append(value)
        return {'add':screen.ml.rank(np.concatenate([value[f]['S42'] for f in range(5)]))}
    a,c=screen.fixed_bundle(raw,arm,control,variants)
    assert a[0,0]>c[0,0] and a.shape==(3,12)
    for got in calls:
        assert got[0]['R42'] is raw[0]['R42']
        for f in range(1,5):
            for key in raw[f]: np.testing.assert_array_equal(got[f][key],before[f][key])
    for f in raw:
        for key in raw[f]: np.testing.assert_array_equal(raw[f][key],before[f][key])


def test_seven_lenses_mapped9_macro_paired_ci_matches_reference():
    ids=[str(i) for i in range(40)]
    y=np.tile(np.arange(40)%2,(12,1)).T.astype(float)
    rng=np.random.default_rng(91)
    a={s:rng.random(y.shape).astype(np.float32) for s in ('42','43')}
    c={s:rng.random(y.shape).astype(np.float32) for s in ('42','43')}
    components=[dict(component_id=u,members=[u]) for u in ids]
    refs={r:(y,np.ones_like(y,bool),1) for r in screen.ml.RULERS}
    ctx=screen.ml.Context(ids,refs,components)
    panel_y=y.copy(); panel_y[:,[1,10,11]]=np.nan
    item=screen.image_item(ids,a,c,panel_y)
    bundle=dict(context=ctx,arm=a['42'],control=c['42'],recipe={})
    result=screen.evaluate(ctx,a,c,bundle,item)
    assert set(result['lenses'])=={f'L{i}' for i in range(1,8)}
    assert result['l4_rule']['target_threshold']==.003
    ix=list(screen.PANEL_TARGETS)
    d=screen.ml.auc_draws(item['arm'],item['labels'],item['mask'],ctx.w)-screen.ml.auc_draws(item['control'],item['labels'],item['mask'],ctx.w)
    expected=screen.ml.stat(d[:,ix].mean(1),eligible=False,reason=screen.ANCESTRY)
    entry=result['lenses']['L7']['entries']['oai_moaks']
    for k,v in expected.items(): assert entry['overall_macro'][k]==v
    assert len(entry['overall_macro']['targets'])==9 and len(entry['targets'])==12
    assert not result['lenses']['L7']['eligible'] and not result['lenses']['L6']['final_proxy']['eligible']
    assert result['release_scope']=='none'


@pytest.mark.parametrize('bad',[.4,float('inf'),-1])
def test_invalid_panel_labels_rejected(bad):
    y=np.zeros((2,12)); y[0,0]=bad; p={s:np.zeros_like(y) for s in ('42','43')}
    with pytest.raises(run.life.ContractError): screen.image_item(['a','b'],p,p,y)
