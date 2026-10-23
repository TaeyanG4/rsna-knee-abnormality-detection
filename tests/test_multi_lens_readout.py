"""CPU regression tests for the multi-lens dispatch contract."""
import json
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from tools import multi_lens_readout as m


def delta(value=0): return np.full((301,4,12),value,dtype=float)


def context(n=40):
    ids=[str(i) for i in range(n)]
    y=np.tile(np.arange(n)%2,(12,1)).T
    refs={r:(y,np.ones_like(y,dtype=bool),1) for r in m.RULERS}
    components=[{'component_id':str(i),'members':ids[i:i+2]} for i in range(0,n,2)]
    return m.Context(ids,refs,components),y,components


@pytest.mark.parametrize('lens,index,gain',[('L1',m.WEAK,.001),('L2',tuple(range(12)),.001)])
def test_exact_threshold_and_negative_ci(lens,index,gain):
    d=delta();d[:,:2,index]=gain;d[1:151]-=.05
    ls,g,_=m.lenses(d)
    assert ls[lens]['PASS'] and ls[lens]['ci95'][0]<0
    assert m.verdict(ls,g)['verdict']=='ADVANCE'


@pytest.mark.parametrize('r,t,loss',[(0,0,-.023),(1,0,-.023),(2,None,-.00301)])
def test_guard_overrides_lenses(r,t,loss):
    d=delta(.002)
    if t is None:d[:,r,:]=loss
    else:d[:,r,t]=loss
    ls,g,_=m.lenses(d)
    assert ls['L1']['PASS']
    assert not g['PASS'] and m.verdict(ls,g)['verdict']=='HOLD'


def test_other_target_guard_excludes_weak():
    d=delta(.004);d[:,:2,8]=-.012
    _,g,_=m.lenses(d)
    assert g['PASS']


def test_l3_no_other_ruler_below_floor_self_reference():
    d=delta();d[:,3]=.002
    ls,g,_=m.lenses(d,True)
    assert ls['L3']['PASS'] and ls['L3']['rulers']['aquila']['self_referential']
    d[:,2]=-.0011
    assert not m.lenses(d)[0]['L3']['PASS']


def test_l4_requires_three_positive_rulers_and_target_scope():
    d=delta();d[:,:3,0]=.0035
    ls,g,_=m.lenses(d)
    v=m.verdict(ls,g)
    assert v['release_scope']==['ACL'] and v['decision_label']=='PER-TARGET ADVANCE (ACL)'
    d[:,2,0]=0
    assert not m.lenses(d)[0]['L4']['PASS']


def test_missing_target_fails_closed_not_nanmean():
    d=delta(.01);d[:,:,11]=np.nan
    ls,g,_=m.lenses(d)
    assert not g['PASS'] and ls['L2']['value'] is None


def test_weighted_bootstrap_matches_expansion_with_ties_and_masks():
    ctx,y,comp=context();rng=np.random.default_rng(42)
    p=rng.integers(0,5,y.shape)/5;mask=rng.random(y.shape)>.15
    scores=m.auc_draws(p,y,mask,ctx.w)
    for b in (0,1,18,300):
        ix=np.repeat(np.arange(len(y)),ctx.w[b].astype(int))
        for t in range(12):
            idx=ix[mask[ix,t]];yy=y[idx,t];n=yy.sum();nn=len(yy)-n
            expected=(rankdata(p[idx,t])[yy==1].sum()-n*(n+1)/2)/(n*nn)
            assert scores[b,t]==pytest.approx(expected)
    np.testing.assert_array_equal(ctx.w,m.weights(ctx.ids,comp))


def test_component_membership_strict():
    with pytest.raises(ValueError,match='missing'):m.weights(['x'],[])
    with pytest.raises(ValueError,match='duplicate'):m.weights(['x'],[{'component_id':'a','members':['x','x']}])


def test_seed_averaging_is_prediction_averaging():
    ctx,y,_=context();rng=np.random.default_rng(5)
    c={str(s):rng.random(y.shape) for s in (42,43)}
    a={str(s):np.clip(c[str(s)] + y*.2,0,1) for s in (42,43)}
    result=m.evaluate(ctx,a,c)
    expected=ctx.delta(np.mean(list(a.values()),0),np.mean(list(c.values()),0))[0,:2].mean()
    assert result['lenses']['L5']['value']==pytest.approx(expected)
    assert result['lenses']['L5']['PASS']
    assert result['lenses']['L5']['sign_agreement']=={'L1':True,'L2':True}
    assert expected!=pytest.approx(result['lenses']['L2']['value'])


def test_seed_sign_disagreement_blocks_l5():
    ctx,y,_=context();c={'a':np.full(y.shape,.5),'b':np.full(y.shape,.5)}
    a={'a':y*.8+.1,'b':1-y*.2}
    assert not m.evaluate(ctx,a,c)['lenses']['L5']['PASS']


def test_gold_cannot_select_oai_requires_eligibility():
    ctx,y,comp=context();p=np.full(y.shape,.5)
    item=dict(ids=ctx.ids,arm=y,control=p,labels=y,mask=np.ones_like(y,bool),components=comp)
    result=m.evaluate(ctx,{'a':p},{'a':p},image_reads={'gold58':item})
    assert result['verdict']=='HOLD'
    assert result['lenses']['L7']['entries']['gold58']['PASS']
    assert not result['lenses']['L7']['PASS']
    result=m.evaluate(ctx,{'a':p},{'a':p},image_reads={'oai_moaks':dict(item,eligible=True)})
    assert result['verdict']=='ADVANCE'
    assert m.evaluate(ctx,{'a':p},{'a':p},image_reads={'oai_moaks':item})['verdict']=='HOLD'


def test_bundle_and_final_proxy_threshold():
    ctx,y,_=context();p=np.full(y.shape,.5)
    bundle=dict(context=ctx,arm=p,control=p,anchor_arm=y,anchor_control=p,recipe={})
    out=m.evaluate(ctx,{'a':p},{'a':p},bundle=bundle)
    assert out['verdict']=='ADVANCE' and out['lenses']['L6']['final_proxy']['PASS']


def csv(path,ids,p):
    df=pd.DataFrame(p,columns=m.TARGETS);df.insert(0,m.UID,ids);df.to_csv(path,index=False)


def test_loader_rosters_hashes_folds_and_blends(tmp_path):
    ids=['b','a','c'];p=np.full((3,12),.4);file=tmp_path/'oof.csv';csv(file,ids,p)
    ri,a,pr=m.load_oof({'path':str(file),'sha256':m.sha(file)})
    assert ri==sorted(ids) and pr[0]['rows']==3
    with pytest.raises(ValueError,match='hash'):m.load_oof({'path':str(file),'sha256':'wrong'})
    with pytest.raises(ValueError,match='duplicate'):m.load_oof([file,file])
    expr={'blend':[{'files':str(file),'weight':.5},{'files':str(file),'weight':.5}]}
    np.testing.assert_array_equal(m.load_oof(expr)[1],m.rank(a))
    csv(file,ids,np.full((3,12),np.nan))
    with pytest.raises(ValueError,match='probabilities'):m.load_oof(file)


def test_csv_roundtrip_preserves_nearly_tied_scores(tmp_path):
    # Decimal parser rounding can collapse distinct scores and change rank blends.
    value = .30661037892268384
    p = np.tile([value, np.nextafter(value, 1.0)], (12, 1)).T
    path = tmp_path/'close-scores.csv'; csv(path, ['a','b'], p)
    np.testing.assert_array_equal(m.load_oof(path)[1], p)


def test_paired_roster_no_silent_intersection(tmp_path):
    a=tmp_path/'a.csv';c=tmp_path/'c.csv'
    csv(a,['1','2'],np.zeros((2,12)));csv(c,['1','3'],np.zeros((2,12)))
    with pytest.raises(ValueError,match='rosters differ'):
        m.run_spec({'seeds':{'42':{'arm':str(a),'control':str(c)}}})


def test_json_no_nan_and_no_label_rows(tmp_path):
    ctx,y,_=context();out=m.evaluate(ctx,{'a':y},{'a':y})
    file=tmp_path/'result.json';m.write(file,out);text=file.read_text()
    assert 'NaN' not in text and 'StudyInstanceUID' not in text
    assert json.loads(text)['rows']==40


def test_fold_ranking_not_repeated_and_explicit_pooled_rank(tmp_path):
    folds=[]; expected=[]
    for f, values in enumerate(([.1,.3,.3],[.2,.8])):
        p=np.tile(values,(12,1)).T; file=tmp_path/f'f{f}.csv'
        csv(file,[f'{f}-{i}' for i in range(len(p))],p); folds.append(str(file));expected.append(m.rank(p))
    expected=np.concatenate(expected)
    expr={'blend':[{'files':folds,'weight':1}]}
    np.testing.assert_array_equal(m.load_oof(expr,fold_rank=True)[1],expected)
    np.testing.assert_array_equal(m.load_oof(dict(expr,rank='pooled'),fold_rank=True)[1],m.rank(expected))
    pooled=tmp_path/'pooled.csv';ids,p,_=m.load_oof(folds);csv(pooled,ids,p)
    wrapper={'files':str(pooled),'rank_folds_like':folds}
    np.testing.assert_array_equal(m.load_oof(wrapper)[1],expected)
    with pytest.raises(ValueError,match='partition'):
        m.load_oof(dict(wrapper,rank_folds_like=folds[:1]))


@pytest.mark.parametrize('family',['r50','coatnet'])
def test_spec_family_replacement_anchor_image_and_seed_paths(tmp_path,monkeypatch,family):
    ctx,y,components=context();rng=np.random.default_rng(55)
    files={}
    for key in ('a','c','a2','c2','r50','dino','convnext','anchor'):
        file=tmp_path/(key+'.csv');csv(file,ctx.ids,rng.random(y.shape));files[key]=str(file)
    labels=tmp_path/'labels.csv';csv(labels,ctx.ids,y)
    cp=tmp_path/'components.json';cp.write_text(json.dumps(components))
    monkeypatch.setattr(m,'make_context',lambda ids,root:ctx)
    spec={'family':family,'seeds':{'42':{'arm':files['a'],'control':files['c']},'43':{'arm':files['a2'],'control':files['c2']}},
          'partners':{k:files[k] for k in ('r50','dino','convnext')},'anchor':files['anchor'],
          'image_reads':{'oai_moaks':{'arm':files['a'],'control':files['c'],'labels':str(labels),'components':str(cp),'eligible':True}}}
    out=m.run_spec(spec)
    assert out['lenses']['L6']['recipe']['fixed_families']==(['dino','convnext'] if family=='r50' else ['r50','dino','convnext'])
    assert ('final_proxy' in out['lenses']['L6'])==(family=='r50')
    assert out['lenses']['L7']['entries']['oai_moaks']['eligible']
    assert out['lenses']['L5']['value'] is not None
    spec['image_reads']={'gold58':dict(spec['image_reads']['oai_moaks'],components='gold58_singletons')}
    assert not m.run_spec(spec)['lenses']['L7']['eligible']


def test_optional_missing_partner_retains_primary_readout(tmp_path,monkeypatch):
    ctx,y,_=context();file=tmp_path/'oof.csv';csv(file,ctx.ids,y)
    monkeypatch.setattr(m,'make_context',lambda ids,root:ctx)
    spec={'seeds':{'42':{'arm':str(file),'control':str(file)}},'family':'r50','partners':{'dino':str(tmp_path/'absent.csv')}}
    result=m.run_spec(spec)
    assert result['lenses']['L1']['value']==0
    assert 'Missing optional partner' in result['lenses']['L6']['reason']
    spec['partners']['dino']={'path':str(file),'sha256':'invalid'}
    with pytest.raises(ValueError,match='hash'):m.run_spec(spec)


def test_old_gate_extraction_and_plain_dedup(tmp_path):
    from tools.multi_lens_retro import old,add_plain,Inventory,held_verdict
    assert not held_verdict('tier_a: THRESHOLDS_MET; tier_b: PENDING_SERVING_COST')
    assert held_verdict('tier_a: THRESHOLDS_NOT_MET')
    assert held_verdict('REVIEW_WEAK_AND_IMAGE_EVIDENCE')
    assert old({'tiers':{'tier_a':{'verdict':'THRESHOLDS_NOT_MET'}}})=='tier_a: THRESHOLDS_NOT_MET'
    assert old({'pass':False})=='HOLD'
    for name in ('a','c'):
        csv(tmp_path/(name+'.csv'),['x'],np.full((1,12),.5))
        csv(tmp_path/(name+'_flip_tta.csv'),['x'],np.full((1,12),.6))
    inv=Inventory(tmp_path,[])
    def spec(mode):
        return {'name':'EXP170/'+mode,'old_verdict':'HOLD','seeds':{'42':{s:inv.get(str(tmp_path/(n+('_flip_tta' if mode=='primary' else '')+'.csv'))) for s,n in [('arm','a'),('control','c')]}}}
    assert len(add_plain([spec('primary'),spec('plain')],inv))==2
    assert len(add_plain([spec('primary')],inv))==2


def test_gold_fixed_support_does_not_drop_targets_per_draw():
    ctx,y,comp=context();y=y.copy();y[:,11]=0
    p=np.full(y.shape,.5);item=dict(ids=ctx.ids,arm=y,control=p,labels=y,mask=np.ones_like(y,bool),components=comp)
    result=m.evaluate(ctx,{'a':p},{'a':p},image_reads={'gold58':item})
    gold=result['lenses']['L7']['entries']['gold58']
    assert gold['value']==.5 and len(gold['fixed_targets'])==11
    assert gold['targets']['Fracture']['value'] is None
