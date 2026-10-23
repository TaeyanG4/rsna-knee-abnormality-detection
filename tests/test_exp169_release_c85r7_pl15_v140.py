import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tools import exp169_release_c85r7_pl15_v140 as m
from tools import vnext_r7_pl_train_v110 as train
from pl_readout_fixture_v111 import make_readout


@pytest.fixture(scope='module')
def stage(tmp_path_factory):
    path=tmp_path_factory.mktemp('c85r7-pl15-stage')
    assert m.register(path)['status']=='REGISTERED_LOCAL_ONLY'
    assert m.prepare(path)['placeholder_members']==10
    return path


@pytest.fixture
def scores():
    rng=np.random.default_rng(143)
    ids=[f'study-{i}' for i in range(17)]
    raw={k:np.round(rng.uniform(.05,.95,(17,12)),1) for k in ('R','S_R7','EXP143v2','EXP155','Q_A5q')}
    return ids,raw,np.round(rng.uniform(.05,.95,(17,12)),1)


def test_sealed_stage_and_placeholders_hold(stage):
    assert m.validate(stage)['status']=='PASS_PREPARED'
    for v in m.VARIANTS:
        assert m.admit(stage,variant=v)['status']=='HOLD'
    for arm in ('global','fill'):
        assert m.pl.registration(stage/'student'/arm)['members']==m.MIL
        with pytest.raises((TypeError,ValueError)):
            m.bind(stage,stage/'student'/arm/'bindings-template.json',stage/f'invalid-{arm}.json')
        assert not (stage/f'invalid-{arm}.json').exists()


@pytest.mark.parametrize('variant',['a','b'])
def test_independent_nested_rank_formula(scores,variant):
    ids,raw,student=scores
    rank=lambda a:pd.DataFrame(a).rank(method='average',pct=True).to_numpy()
    values=dict(raw,PL_STUDENT=student)
    own=sum(w*rank(values[k]) for k,w in m.VARIANTS[variant].items())/sum(m.VARIANTS[variant].values())
    expected=.85*rank(.5*rank(raw['R'])+.5*rank(raw['S_R7']))+.15*rank(own)
    np.testing.assert_array_equal(m.blend(ids,raw,student,variant),expected)


@pytest.mark.parametrize('variant',['a','b'])
@pytest.mark.parametrize('failed',[None,'R','S_R7','EXP143v2','EXP155','Q_A5q','all'])
def test_zero_student_csv_byte_parity(scores,variant,failed):
    ids,raw,_=scores
    failures={k:{ids[0]:'failure',ids[3]:'failure'} for k in raw if failed in (k,'all')}
    expected=m.base.blend(ids,*raw.values(),failures=failures)
    result=m.blend(ids,raw,None,variant,failures,pl_weight=0,diagnostic=True)
    assert m.base.p.csv_bytes(ids,result)==m.base.p.csv_bytes(ids,expected)


@pytest.mark.parametrize('weight',[0,.4,11,13])
def test_production_weight_override_blocked(scores,weight):
    ids,raw,student=scores
    with pytest.raises(ValueError,match='sealed'):
        m.blend(ids,raw,student,pl_weight=weight)


def test_replace_does_not_consume_q(scores):
    ids,raw,student=scores
    expected=m.blend(ids,raw,student,'b')
    raw.pop('Q_A5q')
    np.testing.assert_array_equal(m.blend(ids,raw,student,'b'),expected)


@pytest.mark.parametrize('variant',['a','b'])
def test_failure_insurance(scores,variant):
    ids,raw,student=scores
    with pytest.raises(ValueError,match='absent'):m.blend(ids,raw,None,variant)
    failed={k:dict.fromkeys(ids,'injected') for k in ['R','S_R7',*m.VARIANTS[variant]]}
    np.testing.assert_array_equal(m.blend(ids,raw,student,variant,failed),np.full((len(ids),12),.5))
    # Failed PL rows are excluded from ranks/denominators, not neutral scores.
    f={'PL_STUDENT':dict.fromkeys(ids,'injected')}
    if variant=='a':
        np.testing.assert_array_equal(m.blend(ids,raw,student,variant,f),m.base.blend(ids,*raw.values()))


def test_runtime_budget():
    result=m.runtime_projection(10)
    assert result['projected_compute_hours']==pytest.approx(8.311111111)
    assert result['max_student_seconds_per_study_with_reserve']==pytest.approx(10.98461538)
    assert not result['scenarios'][-1]['fits_compute_budget']


def test_real_prediction_cpu_replay(tmp_path):
    result=m.replay(tmp_path/'replay')
    assert result['studies']==3
    assert set(result['zero_pl_csv_sha256'].values())=={result['base_sha256']}


def make_binding(stage,work,arm='global',ids=None):
    """Real readout + real exporter, synthetic numeric values and tiny CPU model."""
    import torch
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    reg=m.pl.registration(Path(stage)/'student'/arm)
    go=make_readout(work/'readout',arm,ids=ids)
    teacher=work/'teacher.json'
    m.dump(teacher,dict(producer='H-B-145',producer_version='1.1.2',producer_exposure='EXPOSED',
        status='PASS',role='heldout',output_kind='pre_stacker',model_fold='int(sha1(uid)[:8],16)%4',
        complete=True,failures=0,clean_oof=False,exposure='EXPOSED_EXPLORATORY',
        producer_identity='SYNTHETIC_CPU_FIXTURE',csv_sha256='a'*64))
    entries=[]
    decision=train.scientific_go.validate(train.asset(go),arm)
    for member in reg['members']:
        target=dict(arm=arm,seed=member['seed'],fold=member['fold'],engineering_only=False,
            teacher_sha256=m.pl.sha(teacher),target_sha256='b'*64,train_ids_sha256='c'*64,
            l2_pos_weight_sha256='d'*64,registration_sha256=m.digest(reg),silence_mask_sha256='f'*64)
        spec=dict(member=member,spec_sha256=m.digest(member),source_pins=[],target_bundle={'fixture':True})
        admission=dict(status='ADMITTED',go=train.asset(go),decision=decision,spec_sha256=spec['spec_sha256'])
        folder=work/member['id']
        if not (folder/'training-receipt.json').exists():
            train.export(spec,target,torch.nn.Linear(3,12),folder,smoke=False,counters={'updates':40},
                         runtime={'fixture':'CPU_ONLY_NOT_PRODUCTION'},go=admission)
        entries.append(dict(member,**train.asset(folder/'model_best.pt'),
                            training_receipt=train.asset(folder/'training-receipt.json')))
    binding=dict(registration_sha256=m.digest(reg),arm=arm,go=train.asset(go),teacher=train.asset(teacher),members=entries)
    m.dump(work/'inputs.json',binding)
    return work/'inputs.json'


@pytest.mark.parametrize('arm',['global','fill'])
def test_adapter_export_binding_and_native_dispatch(stage,tmp_path,arm):
    binding=make_binding(stage,tmp_path/arm,arm)
    assert m.bind(stage,binding,tmp_path/'bound.json')['status']=='BOUND'
    assert m.admit(stage,tmp_path/'bound.json')['status']=='HOLD'
    ids=['one','two']
    job=m.native_job(stage,tmp_path/'bound.json',ids,[],{'competition':str(tmp_path)},tmp_path/'worker','test',device='cpu')
    from tools import vnext_r7_pl_worker_v1 as worker
    from tools import hb116_vnext_release as native
    original=native.c.Journal
    def fake_worker(path,device):
        assert device=='cpu'
        actual=m.read(path)
        assert all(e['kind']=='MIL' for e in actual['entries'])
        journal=native.c.Journal()
        for uid in actual['ids']:
            for entry in actual['entries']:journal.record(uid,entry['id'],np.full(12,.75))
    worker.native(job,worker=fake_worker)
    assert native.c.Journal is original
    values,failures,count,reason=m.student_predictions(tmp_path/'worker/own',ids,job['branches'][0]['identity'])
    assert np.all(values==.75) and not failures and count==0 and reason is None
    b=m.read(binding);b['arm']='fill' if arm=='global' else 'global'
    m.dump(tmp_path/'wrong.json',b)
    with pytest.raises(ValueError):m.bind(stage,tmp_path/'wrong.json',tmp_path/'bad.json')
    # A genuine journal identity mismatch is a hard authentication failure.
    changed=copy.deepcopy(job['branches'][0]['identity']);changed['run_id']='other'
    with pytest.raises(ValueError,match='identity authentication'):
        m.student_predictions(tmp_path/'worker/own',ids,changed)


@pytest.mark.parametrize('file',['variant-a.json','student/global/training-plan.json','student/fill/bindings-template.json'])
def test_stage_tampering_rejected(stage,tmp_path,file):
    import shutil
    target=tmp_path/'stage';shutil.copytree(stage,target)
    m.dump(target/file,{'tampered':True})
    with pytest.raises(ValueError):m.validate(target)


def qualification(stage,binding,variant,path):
    q=dict(registration_sha256=m.digest(m.registration(stage)),binding_sha256=m.pl.sha(binding),
        variant=variant,device='2xT4',upper_seconds=30000,publication_reserve_seconds=1200,total_seconds=31200,
        fixture='SYNTHETIC CPU interface evidence, not real target qualification')
    q.update({k:True for k in ('complete_ordered_csv','finite','zero_healthy_fallbacks','descendants_exited',
        'gpu_memory_released','fault_n32_pass','native_parity','codecs_best_effort')})
    m.dump(path,q)
    return train.asset(path)


def publication_smoke(stage,work,binding):
    from tools.exp169_c85r7ins_checks_v122 import snapshot
    work=Path(work)
    replay=m.replay(work/'replay')
    pdir=work/'replay/base/replay-input/pr7'
    ids,raw,_=snapshot(m.base.c85.A5Q,pdir)
    job=m.native_job(stage,binding,ids,[],{'competition':str(work)},work/'native','real-recorded-cpu',device='cpu')
    from tools import vnext_r7_pl_worker_v1 as worker
    from tools import hb116_vnext_release as native
    def recorded_worker(path,device):
        assert device=='cpu'
        task=m.read(path)
        journal=native.c.Journal()
        for i,uid in enumerate(ids):
            for entry in task['entries']:journal.record(uid,entry['id'],raw['EXP155'][i])
    worker.native(job,worker=recorded_worker)
    outputs={}
    for v in ('a','b'):
        q=qualification(stage,binding,v,work/f'qualification-{v}.json')
        output=work/f'published-{v}'
        # The fixed smoke location is idempotent; never reuse as a qualification.
        if output.exists():
            result=m.read(output/'receipt.json')
            assert result['csv_sha256']==m.pl.sha(output/'submission.csv')
        else:
            result=m.publish(stage,binding,q,v,m.base.c85.A5Q,pdir,job['branches'],'real-recorded-cpu',output)
        student=np.mean(np.stack([raw['EXP155']]*10),axis=0,dtype=np.float32)
        assert (output/'submission.csv').read_bytes()==m.base.p.csv_bytes(ids,m.blend(ids,raw,student,v))
        outputs[v]=dict(status=result['status'],csv_sha256=result['csv_sha256'])
    return dict(replay=replay,publish=outputs,scope='real recorded vectors, synthetic readout/export/qualification; no inference')


def test_authenticated_final_publication(stage,tmp_path):
    binding=make_binding(stage,tmp_path/'binding')
    smoke=publication_smoke(stage,tmp_path/'smoke',binding)
    assert smoke['replay']['studies']==3 and len(smoke['publish'])==2
    q=qualification(stage,binding,'a',tmp_path/'q.json')
    with pytest.raises(ValueError,match='fresh publication'):
        m.publish(stage,binding,q,'a',None,None,[],None,tmp_path/'smoke/published-a')
    with pytest.raises(ValueError,match='HOLD'):
        m.publish(stage,binding,None,'a',None,None,[],None,tmp_path/'missing-q')
    assert not (tmp_path/'missing-q').exists()
    doc=m.read(q['path']);doc['variant']='b';m.dump(q['path'],doc)
    with pytest.raises(ValueError,match='qualification identity'):
        m.admit(stage,binding,train.asset(q['path']),'a')


def test_placeholders_cannot_publish(stage,tmp_path):
    with pytest.raises(ValueError,match='HOLD'):
        m.publish(stage,None,None,'a',None,None,[],None,tmp_path/'publish')
    assert not (tmp_path/'publish').exists()
