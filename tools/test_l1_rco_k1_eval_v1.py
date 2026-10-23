"""CPU evaluator regression tests: exact masks, rank ties, gate, completeness."""
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
import pytest
from tools import l1_rco_k1_eval_v1 as e


def fixture():
    rng = np.random.default_rng(41)
    y = rng.integers(0, 2, (200, 12))
    a = y * .5 + rng.uniform(0, .5, y.shape)
    b = y * .5 + rng.uniform(0, .5, y.shape)
    refs = {k: (y.copy(), np.ones(y.shape, bool), 1) for k in ('jev_hybrid', 'aquila_v1', 'dread', 'consensus')}
    return a, b, refs


def test_gate_accepts_three_rulers_and_rejects_two():
    a, b, refs = fixture()
    assert e.metrics(a, b, refs)['decision'] == 'KEEP'
    # Use a direct controlled four-ruler configuration, including consensus masks.
    a, b, refs = fixture()
    bad_y = 1-refs['dread'][0]
    a = b.copy(); a[:, 0] = 1-b[:, 0]
    # Three rulers reward candidate ordering for target0, one rewards baseline.
    for key in ('jev_hybrid', 'aquila_v1', 'dread'):
        refs[key][0][:, 0] = bad_y[:, 0]
    r = e.metrics(a, b, refs)
    assert r['rulers_passed'] == 3 and r['decision'] == 'KEEP'
    refs['dread'][0][:, 0] = 1-bad_y[:, 0]
    r = e.metrics(a, b, refs)
    assert r['rulers_passed'] == 2 and r['decision'] == 'DROP'


def test_spearman_duplicate_predictions_drop_and_undefined_hold():
    a, b, refs = fixture()
    r = e.metrics(b, b, refs)
    assert r['rulers_passed'] == 4 and r['spearman_mean'] == pytest.approx(1)
    assert r['decision'] == 'DROP'
    a[:, 0] = .5
    assert e.metrics(a, b, refs)['decision'] == 'HOLD'
    refs['consensus'][1][:, 1] = False
    assert e.metrics(b, b, refs)['decision'] == 'HOLD'


def test_auc_average_ties_and_mask():
    assert e.auc([0, 1, 0, 1], [0, 0, 1, 1]) == .5
    a, b, refs = fixture()
    refs['consensus'][1][::2] = False
    r = e.metrics(a, b, refs)
    row = next(x for x in r['per_target'] if x['ruler'] == 'consensus')
    assert row['n'] == 100
    assert row['rco_auc'] == e.auc(refs['consensus'][0][1::2, 0], a[1::2, 0])


def test_predictions_reject_duplicates_and_wrong_fold(tmp_path):
    p = tmp_path/'pred.csv'
    df = pd.DataFrame({**{k: [.2, .8] for k in e.TARGETS}, 'StudyInstanceUID': ['a', 'b'],
                       'status': ['ok', 'ok'], 'model_fold': [0, 1], 'fold_k': [0, 0]})
    df.to_csv(p, index=False)
    ids, a = e.predictions(p, ['a', 'b'], teacher=True)
    assert ids == ['a'] and a.shape == (1, 12)
    pd.concat([df, df.iloc[:1]]).to_csv(p, index=False)
    with pytest.raises(ValueError, match='Duplicate'): e.predictions(p, ['a', 'b'], teacher=True)


def test_registration_pin_mutation_rejected(monkeypatch):
    spec = dict(rule=e.RULE, version=e.VERSION, pins={'x': 'old'})
    monkeypatch.setattr(e, 'sha', lambda _: 'new')
    with pytest.raises(ValueError, match='changed'): e.evaluate(dict(evaluator=spec))


def test_nonfinite_or_wrong_rulers_rejected():
    a, b, refs = fixture()
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'): e.metrics(a, b, refs)
    del refs['dread']
    with pytest.raises(ValueError, match='four rulers'): e.metrics(b, b, refs)


def test_full_evaluate976_primary_all977_output_and_candidate_receipt(tmp_path, monkeypatch):
    ids = sorted([str(i) for i in range(5000) if int(hashlib.sha1(str(i).encode()).hexdigest()[:8],16)%4 == 0][:977])
    primary = ids[:-1]
    rng = np.random.default_rng(123)
    y = rng.integers(0,2,(977,12))
    refs = {k:(y[:-1],np.ones_like(y[:-1],dtype=bool),1) for k in ('jev_hybrid','aquila_v1','dread','consensus')}
    teacher = tmp_path/'teacher.csv'
    def write_pred(p, teacher=False):
        df = pd.DataFrame(y*.5+rng.uniform(0,.5,y.shape),columns=e.TARGETS)
        df['StudyInstanceUID'] = ids; df['status']='ok'; df['model_fold']=0; df['fold_k']=0
        if teacher: df.loc[len(df)-1,'status']='dropped series'
        df.to_csv(p,index=False)
    write_pred(teacher, teacher=True)
    spec = dict(rule=e.RULE,version=e.VERSION,pins={},teacher=str(teacher),exposure='TEST',
                primary=dict(ids=primary,id_list_sha256=e.ids_digest(primary),q12b_equal=True,missing=e.excluded_metadata(teacher,[ids[-1]])))
    monkeypatch.setattr(e,'evaluator_registration',lambda _:spec)
    def references(requested):
        assert requested == primary
        return refs
    monkeypatch.setattr(e.rulers,'references',references)
    monkeypatch.setattr(e,'WORK',tmp_path)
    fit=tmp_path/'fit'; fit.mkdir(); write_pred(fit/'holdout.csv')
    receipt = dict(contract_id='test',holdout_sha256=e.sha(fit/'holdout.csv'))
    (fit/'result.json').write_text(json.dumps(receipt))
    reg=dict(evaluator=spec,holdout=ids,contract_id='test')
    result=e.evaluate(reg)
    assert result['decision']=='KEEP' and len(result['per_target'])==48 and result['n']==976 and result['candidate_valid']==977
    assert result['excluded_from_primary'][0]['id_sha256']==hashlib.sha256(ids[-1].encode()).hexdigest()
    assert result['excluded_from_primary'][0]['status']=='dropped series'
    assert 'unknown' in result['excluded_from_primary'][0]['root_cause']
    assert (tmp_path/'evaluate/per-target.csv').is_file()
    pred=fit/'holdout.csv'; df=pd.read_csv(pred,dtype={'StudyInstanceUID':str})
    # The excluded output remains mandatory but cannot affect any scientific statistic.
    df.loc[len(df)-1,list(e.TARGETS)] = 1-df.loc[len(df)-1,list(e.TARGETS)]
    df.to_csv(pred,index=False); receipt['holdout_sha256']=e.sha(pred)
    (fit/'result.json').write_text(json.dumps(receipt))
    again=e.evaluate(reg)
    assert again['rulers']==result['rulers'] and again['spearman_mean']==result['spearman_mean']
    df.iloc[:-1].to_csv(pred,index=False); receipt['holdout_sha256']=e.sha(pred)
    (fit/'result.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='exact holdout roster'): e.evaluate(reg)
    receipt['holdout_sha256']='wrong'; (fit/'result.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='receipt/hash'): e.evaluate(reg)


def test_auc_delete_one_matches_bruteforce_ties_and_undefined():
    y=np.array([0,1,0,1,0,1]); values=np.array([0,0,.5,.5,1,1])
    exact=np.array([e.auc(np.delete(y,i),np.delete(values,i)) for i in range(len(y))])
    np.testing.assert_allclose(e.auc_delete_one(y,values),exact,atol=1e-15)
    assert np.isnan(e.auc_delete_one([0,0,1],[.1,.3,.5])[-1])
    assert np.isnan(e.auc_delete_one([1,1],[.1,.2])).all()


def test_paired_jackknife_matches_bruteforce_masks_and_target_covariance():
    rng=np.random.default_rng(19); n=18
    y=rng.integers(0,2,(n,12)); y[:4]=np.array([0,1,0,1])[:,None]
    a=rng.integers(0,5,(n,12))/4; b=rng.integers(0,5,(n,12))/4
    mask=rng.random((n,12))>.2; mask[:4]=True
    refs={k:(y,mask,1) for k in ('jev_hybrid','aquila_v1','dread','consensus')}
    actual=e.metrics(a,b,refs)['rulers']['consensus']
    leaves=[]
    for omitted in range(n):
        deltas=[]
        for j in range(12):
            m=mask[:,j].copy(); m[omitted]=False
            deltas.append(e.auc(y[m,j],a[m,j])-e.auc(y[m,j],b[m,j]))
        leaves.append(np.mean(deltas))
    expected=e.normal_summary(actual['delta'],leaves)
    assert actual['jackknife_se']==pytest.approx(expected['jackknife_se'],abs=1e-14)
    assert actual['ci95']==pytest.approx(expected['ci95'],abs=1e-14)
    # Identical targets have perfect covariance: macro SE equals one target's SE.
    yy=np.tile(y[:,0,None],(1,12)); aa=np.tile(a[:,0,None],(1,12)); bb=np.tile(b[:,0,None],(1,12))
    rr={k:(yy,np.ones_like(yy,bool),1) for k in refs}
    one=e.normal_summary(e.auc(yy[:,0],aa[:,0])-e.auc(yy[:,0],bb[:,0]),
                         e.auc_delete_one(yy[:,0],aa[:,0])-e.auc_delete_one(yy[:,0],bb[:,0]))
    assert e.metrics(aa,bb,rr)['rulers']['dread']['jackknife_se']==pytest.approx(one['jackknife_se'],abs=1e-14)


def test_normal_probability_direction_crossing_zero_and_zero_undefined_se():
    from statistics import NormalDist
    r=e.normal_summary(.001,[-.04,.03,.02,-.01])
    assert r['direction']=='positive' and r['ci95'][0]<0<r['ci95'][1]
    assert r['p_delta_positive_normal']==pytest.approx(NormalDist().cdf(.001/r['jackknife_se']))
    for delta,expected in ((.1,1),(-.1,0),(0,.5)):
        r=e.normal_summary(delta,[.2,.2,.2]); assert r['p_delta_positive_normal']==expected and r['jackknife_se']==0
    r=e.normal_summary(.01,[0,float('nan')])
    assert r['direction']=='positive' and r['p_delta_positive_normal'] is None and r['uncertainty_status']=='UNDEFINED'


def test_real_primary_exact_roster_hash_and_q12b_mutations_refused(tmp_path,monkeypatch):
    from tools import l1_rco_k1_v1 as k
    reg=k.read(k.WORK/'registration.json'); actual=e.cohort_registration(reg['holdout'])
    assert actual['n']==976 and actual['q12b_equal'] and actual['id_list_sha256']=='fb04d36607a6fb9130357449dadb4f7c573183fdc033a1549975bd328a21942a'
    q=json.loads(e.Q12B_REG.read_text(encoding='utf-8')); q['ids'][0]=q['ids'][1]
    changed=tmp_path/'q12b.json'; changed.write_text(json.dumps(q)); monkeypatch.setattr(e,'Q12B_REG',changed)
    with pytest.raises(ValueError,match='exact Q12b976'): e.cohort_registration(reg['holdout'])


@pytest.mark.parametrize('field',['ids','id_list_sha256'])
def test_evaluate_primary_roster_hash_mutation_refused(monkeypatch,field):
    import copy
    from tools import l1_rco_k1_v1 as k
    reg=copy.deepcopy(k.read(k.WORK/'registration.json'))
    if field=='ids': reg['evaluator']['primary']['ids'][0]=reg['holdout'][-1]
    else: reg['evaluator']['primary']['id_list_sha256']='0'*64
    with pytest.raises(ValueError,match='identity/pins differ'): e.evaluate(reg,dry_run=True)
