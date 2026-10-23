"""Q28 paired ablation statistics, immutable reuse and admission regressions."""
import copy
from pathlib import Path
import numpy as np
import pytest
from tools import oai_flip_readout_v1 as r


def fixture(n=96):
    rng=np.random.default_rng(28); y=np.full((n,12),np.nan)
    for j in r.predecessor.MAPPED:
        y[:,j]=rng.permutation(np.arange(n)%2).astype(float)
        y[rng.choice(n,n//10,replace=False),j]=np.nan
    return y,{a:rng.choice(np.linspace(0,1,9),(n,12)) for a in r.ALL_ARMS}


@pytest.mark.parametrize('n',[96,461,557])
def test_bootstrap_matches_expanded_sklearn(n):
    from sklearn.metrics import roc_auc_score
    y,v=fixture(n); eligible=r.label_contract(y)['eligible']; arrays=list(v.values())
    boot,w,_=r.bootstrap(y,arrays,eligible,draws=4)
    for b,weights in enumerate(w):
        ix=np.repeat(np.arange(n),weights.astype(int))
        for k,a in enumerate(arrays):
            for q,j in enumerate(eligible):
                z=ix[np.isfinite(y[ix,j])]
                assert boot[k,b,q]==pytest.approx(roc_auc_score(y[z,j],a[z,j]))


def test_shared_draws_interaction_sign_and_family_interval():
    y,_=fixture(); good=np.nan_to_num(y,nan=.5)
    v=dict(R6flip=good,R6plain=1-good,R7flip=1-good,R7plain=good)
    x=r.metrics(y,v,r.label_contract(y),draws=10)['comparisons']
    assert x['R6flip-minus-R6plain']['macro7']['delta']==1
    assert x['R7flip-minus-R7plain']['macro7']['delta']==-1
    assert x['R6-effect-minus-R7-effect']['macro7']['delta']==2
    assert x['R6flip-minus-R6plain']['macro7']['ci97_5_family_two']==[1,1]
    assert x['R7flip-minus-R7plain']['point_floors']=='FAIL'
    assert r.RULES['macro_floor']==-.001 and r.RULES['weak_four_floor']==-.002


def test_optional_targets_do_not_change_primary_draws():
    y,v=fixture(30); y[:,[0,4]]=np.nan
    a=r.metrics(y,v,r.label_contract(y))
    y[:10,0]=[0]*5+[1]*5; y[10:20,4]=[0]*5+[1]*5
    b=r.metrics(y,v,r.label_contract(y))
    assert a['valid_draws']==b['valid_draws']==2000
    for key in a['comparisons']:
        for name in ('macro7','weak_four','point_floors'): assert a['comparisons'][key][name]==b['comparisons'][key][name]


@pytest.mark.parametrize('missing',[2,8])
def test_missing_fixed_target_incomplete(missing):
    y,v=fixture(); y[:,missing]=np.nan
    for x in r.metrics(y,v,r.label_contract(y),draws=3)['comparisons'].values():
        assert x['macro7'] is None
        assert x['point_floors'] in ('INCOMPLETE','NOT_APPLICABLE_INTERACTION')


@pytest.mark.parametrize('damage',['nan','inf','range','shape','label','unmapped','arm'])
def test_invalid_inputs_fail(damage):
    y,v=fixture()
    if damage=='nan': v['R6flip'][0,0]=np.nan
    if damage=='inf': v['R6flip'][0,0]=np.inf
    if damage=='range': v['R6flip'][0,0]=2
    if damage=='shape': v['R6flip']=v['R6flip'][:-1]
    if damage=='label': y[0,0]=2
    if damage=='unmapped': y[0,1]=0
    if damage=='arm': del v['R6flip']
    with pytest.raises(ValueError): r.metrics(y,v,r.label_contract(y),draws=3)


def test_private_module_does_not_mutate_predecessor():
    assert r.predecessor.ARMS==('R7flip','R6flip')
    assert r.predecessor.WORK!=r.WORK and r.e.WORK==r.WORK
    assert r.e.infer is r.infer and r.predecessor.infer is not r.infer
    assert r.predecessor.check_measurement is not r.check_measurement
    assert r.queue_plan()[0]['after']==r.predecessor.TAG
    argv=r.queue_plan()[0]['command']
    assert Path(argv[0]).is_absolute() and Path(argv[2]).is_absolute()
    assert argv[-1]=='run'


def test_bound_independent_review():
    reg={'binding_sha256':'b'}
    v=dict(verdict='PASS',reviewer='independent',builder=r.BUILDER,tag=r.TAG,execution_authority='B',binding_sha256='b',validation_sha256='v')
    r.e.check_review(v,reg,'v')
    for k,x in [('reviewer',r.BUILDER),('verdict','HOLD'),('binding_sha256','wrong'),('validation_sha256','wrong')]:
        with pytest.raises(ValueError): r.e.check_review(dict(v,**{k:x}),reg,'v')


def test_cache_does_not_open_running_output(monkeypatch):
    from tools import job_runtime
    monkeypatch.setattr(job_runtime,'job_status',lambda *a,**k:dict(state='running'))
    monkeypatch.setattr(r.io,'read_json',lambda _:pytest.fail('running output opened'))
    with pytest.raises(ValueError,match='incomplete'): r.authenticate_cache(r.predecessor)


def test_cache_hashes_before_prediction_read(monkeypatch):
    from tools import job_runtime
    monkeypatch.setattr(job_runtime,'job_status',lambda *a,**k:dict(state='SUCCESS'))
    def read(p):
        if p.name=='registration.json': return dict(binding_sha256='b')
        if p.name=='measurement.json': return dict(state='COMPLETE_PENDING_RESULT_REVIEW',binding_sha256='b',attempt_sha256='bad',review_sha256='h')
        pytest.fail('unsealed predictions accessed')
    monkeypatch.setattr(r.io,'read_json',read); monkeypatch.setattr(r,'sha',lambda _:'h')
    with pytest.raises(ValueError,match='seal'): r.authenticate_cache(r.predecessor)


def pairing():
    ids=['a','b']; b=dict(ids=ids,pilot_ids=['a'],confirmation_ids=['b'])
    def arm(ids,flip):
        return dict(ids=ids,runtime=dict(flip_tta=flip,workers=2),torch='same',desktop={'uuid':'gpu'},
                    predictions=np.full((len(ids),12),.5).tolist(),rows=[dict(study=u,windows=10,batch_windows=192) for u in ids])
    plain={'arms':{a:arm(ids,False) for a in r.ARMS}}
    p={'arms':{a.replace('plain','flip'):arm(['a'],True) for a in r.ARMS}}
    c={'arms':{a.replace('plain','flip'):arm(['b'],True) for a in r.ARMS}}
    return plain,p,c,{'binding':b}


@pytest.mark.parametrize('damage',['uuid','torch','runtime','order','windows','flip','overlap'])
def test_cached_pair_identity_rejects_drift(damage):
    plain,p,c,reg=pairing(); assert set(r.check_paired_cache(plain,p,c,reg))==set(r.ALL_ARMS)
    a=c['arms']['R6flip']
    if damage=='uuid': a['desktop']['uuid']='other'
    if damage=='torch': a['torch']='other'
    if damage=='runtime': a['runtime']['workers']=1
    if damage=='order': a['ids']=['wrong']
    if damage=='windows': a['rows'][0]['windows']=11
    if damage=='flip': a['runtime']['flip_tta']=False
    if damage=='overlap': reg['binding']['confirmation_ids']=['a']
    with pytest.raises(ValueError): r.check_paired_cache(plain,p,c,reg)


def test_plain_flag_gate_does_not_mutate_receipt(monkeypatch):
    v=dict(runtime=dict(flip_tta=False)); original=copy.deepcopy(v)
    monkeypatch.setattr(r.predecessor,'check_measurement',lambda value,*a: value['runtime']['flip_tta'] is True)
    r.check_measurement(v,{},'R6plain',{},{}); assert v==original
    with pytest.raises(ValueError): r.check_measurement(dict(runtime=dict(flip_tta=True)),{},'R6plain',{}, {})


def test_no_unreviewed_launch_or_worker(monkeypatch):
    monkeypatch.delenv('HB164_CONTROLLER_PID',raising=False)
    with pytest.raises(ValueError,match='supervised'): r.e.worker()
    monkeypatch.setattr(r.e,'review_ready',lambda:(_ for _ in ()).throw(ValueError('missing review')))
    with pytest.raises(ValueError,match='review'): r.e.launch_check()


@pytest.mark.parametrize('drift',[False,True])
def test_cpu_oracle_metadata_exists_before_inference(tmp_path,monkeypatch,drift):
    monkeypatch.setattr(r,'WORK',tmp_path)
    monkeypatch.setattr(r.g.c,'cpu_only',lambda:None)
    monkeypatch.setattr(r.e,'verify_prepared',lambda:{'binding':{'smoke_id':'study'}})
    data=tmp_path/'data'; data.mkdir(); source=data/'test_series.csv'
    source.write_text('StudyInstanceUID,SeriesInstanceUID\nstudy,series\n')
    if drift: (data/'train_series.csv').write_text('wrong')
    def check_alias(*args):
        assert (data/'train_series.csv').read_bytes()==source.read_bytes()
        raise RuntimeError('metadata-ready')
    monkeypatch.setattr(r,'infer',check_alias)
    with pytest.raises(ValueError if drift else RuntimeError,match='oracle metadata differs' if drift else 'metadata-ready'):
        r.smoke()
