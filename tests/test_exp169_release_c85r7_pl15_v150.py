import copy
from pathlib import Path
import numpy as np
import pytest
from tools import exp169_release_c85r7_pl15_v150 as m
from tools import c85r7_v150_runner as runner
from tools import c85r7_v150_safety as safety
from test_exp169_release_c85r7_pl15_v140 import make_binding, scores


@pytest.fixture(scope='module')
def stage(tmp_path_factory):
    out=tmp_path_factory.mktemp('v150')
    m.register(out);m.prepare(out)
    return out


def test_stage_seals_and_hold(stage,tmp_path):
    assert m.validate(stage)['status']=='PASS_PREPARED'
    assert m.blend is m.predecessor.blend
    for v in m.VARIANTS:
        assert m.admit(stage,variant=v)['status']=='HOLD'
    for mode in ('visible','n32'):
        assert runner.plan(stage,mode)['status']=='HOLD'
    for arm in ('global','fill'):
        with pytest.raises((TypeError,ValueError)):
            m.bind(stage,stage/'student'/arm/'bindings-template.json',tmp_path/'bad.json')
        assert not (tmp_path/'bad.json').exists()


@pytest.mark.parametrize('variant',['a','b'])
@pytest.mark.parametrize('failed',[None,'R','S_R7','EXP143v2','EXP155','Q_A5q','PL_STUDENT','all'])
def test_identical_formulas_and_zero_pl(scores,variant,failed):
    ids,raw,student=scores
    failures={k:{ids[1]:'injected'} for k in [*raw,'PL_STUDENT'] if failed in (k,'all')}
    csv=m.base.p.csv_bytes
    assert csv(ids,m.blend(ids,raw,student,variant,failures))==csv(ids,m.predecessor.blend(ids,raw,student,variant,failures))
    assert csv(ids,m.blend(ids,raw,None,variant,failures,pl_weight=0,diagnostic=True))==csv(ids,m.base.blend(ids,*raw.values(),failures=failures))


def test_real_recorded_flip_boundary(tmp_path):
    result=m.replay(tmp_path/'replay')
    assert result['zero_pl_v130_bytes'] and result['flip_off_v140_bytes']
    assert result['studies']==3


def test_frozen_runtime(stage,tmp_path):
    module,rt,bootstrap=runner.load_runtime(stage,tmp_path)
    expected=m.flip_build.runtime_source(m.flip_build.frozen_files(),'FLIP_F0F2')
    assert (rt/(m.flip_build.MODULE+'.py')).read_bytes()==expected
    assert module.FLIP_VARIANT=='FLIP_F0F2'
    assert (bootstrap/'sitecustomize.py').exists()
    for name,data in m.flip_build.frozen_files().items():
        assert (rt/name).read_bytes()==data


def test_runtime_boundary():
    q=m.runtime_projection()
    assert q['before_pl_hours']==pytest.approx(6.7931827444)
    cap=q['max_student_seconds_per_study']
    assert m.runtime_projection(cap)['projected_compute_hours']==pytest.approx(7.5)
    assert m.runtime_projection(cap)['status']=='WITHIN_CAP'
    assert m.runtime_projection(cap+1e-6)['status']=='OVER_CAP_HOLD'
    assert m.runtime_projection(cap)['projected_with_reserve_hours']==pytest.approx(7.5+1/3)
    for value in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):m.runtime_projection(value)


def go():
    return dict(variant='FLIP_F0F2',macro_delta=.001,ci_lower=.00001,defined_fraction=.95,target_deltas=[-.005]*12)


@pytest.mark.parametrize('key,value',[('variant','FLIP_ALL'),('macro_delta',.00099),('ci_lower',0),
    ('defined_fraction',.949),('target_deltas',[-.00501]*12),('target_deltas',[0]*11),('ci_lower',float('nan'))])
def test_registered_go_rejects(key,value):
    d=go();assert m.decision_gate(d)
    d[key]=value
    with pytest.raises(ValueError):m.decision_gate(d)


def review(stage,work):
    pins=[]
    for kind in ('suite','real_cpu_smoke'):
        path=work/(kind+'.json')
        evidence=work/(kind+'-evidence.json');m.dump(evidence,dict(fixture=True))
        m.dump(path,dict(kind=kind,status='PASS',registration_sha256=m.digest(m.registration(stage)),fixture=True,
                         sources=[runner.asset(evidence)]))
        pins.append(runner.asset(path))
    path=work/'review.json'
    m.dump(path,dict(status='PASS',independent=True,reviewer='SYNTHETIC_TEST_ONLY',
        registration_sha256=m.digest(m.registration(stage)),smoke_receipts=pins))
    return runner.asset(path)


def qualification(stage,binding,work):
    common=dict(registration_sha256=m.digest(m.registration(stage)),binding_sha256=m.pl.sha(binding),
        flip_policy=m.flip.policy('FLIP_F0F2'),variants=['a','b'],image_sha256='1'*64)
    q=dict(common,device='2xT4',n32_count=1,student_seconds_per_study=1.5,upper_seconds=26700,
        publication_reserve_seconds=1200,total_seconds=27900)
    q['health']=health(stage,binding,work)
    for kind,n in [('visible',3),('n32',32)]:
        e=dict(common,mode=kind,status='PASS',studies=n,fault_n32_pass=True)
        e.update({k:True for k in ('complete_ordered_csv','finite','zero_healthy_fallbacks','descendants_exited',
            'gpu_memory_released','native_parity','codecs_best_effort')})
        path=work/(kind+'.json');m.dump(path,e);q[kind]=runner.asset(path)
    path=work/'q.json';m.dump(path,q)
    return runner.asset(path)


def health(stage,binding,work):
    evidence=work/'health-raw.json';m.dump(evidence,dict(synthetic=True))
    path=work/'health.json'
    m.dump(path,dict(status='PASS',registration_sha256=m.digest(m.registration(stage)),
        binding_sha256=m.pl.sha(binding),image_sha256='1'*64,device='2xT4',full_shape=True,
        flip_policy=m.flip.policy('FLIP_F0F2'),peak_reserved_mib=[9000,9000],
        healthy_commits_per_second={k:1. for k in ('own','Q_A5q','R','S_R7','PL-0','PL-1')},
        sources=[runner.asset(evidence)]))
    return runner.asset(path)


def test_shared_qualification_and_review_gate(stage,tmp_path):
    binding=make_binding(stage,tmp_path/'binding')
    r=review(stage,tmp_path);q=qualification(stage,binding,tmp_path)
    path=tmp_path/'go.json';m.dump(path,go());g=runner.asset(path)
    for v in m.VARIANTS:
        assert m.admit(stage,binding,q,v,r,g)['status']=='ADMITTED'
        assert m.admit(stage,binding,q,v,None,g)['status']=='HOLD'
    original=m.read(q['path'])
    for key,value in [('n32_count',2),('flip_policy',{}),('student_seconds_per_study',2.01),
                      ('upper_seconds',27001),('publication_reserve_seconds',0),('variants',['a'])]:
        d=dict(original);d[key]=value;m.dump(q['path'],d)
        with pytest.raises(ValueError):m.admit(stage,binding,runner.asset(q['path']),'a',r,g)
    m.dump(q['path'],original)
    v=m.read(original['visible']['path']);v['image_sha256']='2'*64
    m.dump(original['visible']['path'],v)
    with pytest.raises(ValueError):m.admit(stage,binding,q,'b',r,g)
    smoke=m.read(r['path'])['smoke_receipts'][0]
    m.dump(smoke['path'],dict(status='PASS'))
    with pytest.raises(ValueError):m.review_gate(stage,r)


def publication_smoke(stage,work,arm='global'):
    from tools.exp169_c85r7ins_checks_v122 import snapshot
    from tools import vnext_r7_pl_worker_v1 as worker
    from tools import hb116_vnext_release as native
    work=Path(work)
    binding=make_binding(stage,work/'binding',arm)
    m.bind(stage,binding,work/'bound.json')
    m.replay(work/'replay')
    pdir=work/'replay/base/replay-input/pr7'
    ids,raw,failures=snapshot(m.base.c85.A5Q,pdir)
    receipt=work/'base-receipt.json'
    m.dump(receipt,dict(flip_policy=m.flip.policy('FLIP_F0F2'),snapshot_sha256=m.snapshot_digest(ids,raw,failures)))
    job=m.native_job(stage,binding,ids,[],{'competition':str(work)},work/'native','recorded-smoke',device='cpu')
    def recorded(path,device):
        assert device=='cpu'
        task=m.read(path);journal=native.c.Journal()
        for i,uid in enumerate(ids):
            for member in task['entries']:journal.record(uid,member['id'],raw['EXP155'][i])
    with safety.metadata_io(m.pl.metadata):worker.native(job,worker=recorded)
    expected=np.mean(np.stack([raw['EXP155']]*10),axis=0,dtype=np.float32)
    outputs={}
    for v in m.VARIANTS:
        output=work/('variant-'+v)
        result=m.compose(stage,binding,v,m.base.c85.A5Q,pdir,job['branches'],'recorded-smoke',output,base_receipt=runner.asset(receipt))
        assert (output/'diagnostic.csv').read_bytes()==m.base.p.csv_bytes(ids,m.blend(ids,raw,expected,v,failures))
        outputs[v]=result['csv_sha256']
    return dict(status='PASS',arm=arm,studies=len(ids),outputs=outputs,
        scope='real recorded predictions through actual v1.1.1 adapter; synthetic CPU checkpoints; no PL inference')


@pytest.mark.parametrize('arm',['global','fill'])
def test_combined_cpu_compositor(stage,tmp_path,arm):
    assert publication_smoke(stage,tmp_path/arm,arm)['status']=='PASS'


def test_old_partial_baseline_rejected(stage,tmp_path):
    binding=make_binding(stage,tmp_path/'binding')
    for receipt in ({},{'flip_policy':m.flip.policy('FLIP_ALL')}):
        path=tmp_path/'baseline.json';m.dump(path,receipt)
        with pytest.raises(ValueError,match='FLIP_F0F2 baseline'):
            m.compose(stage,binding,'a',None,None,[],'x',tmp_path/'bad',base_receipt=runner.asset(path))
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='HOLD'):
        m.compose(stage,binding,'a',None,None,[],'x',tmp_path/'bad',base_receipt=None,diagnostic=False)


def test_retry_and_watchdog(tmp_path,monkeypatch):
    calls=[]
    def transient():
        calls.append(1)
        if len(calls)<3:raise PermissionError('busy')
        return 'ok'
    assert safety.retry(transient,attempts=3,pause=0)=='ok'
    assert len(calls)==3
    with pytest.raises(PermissionError):safety.retry(lambda:(_ for _ in ()).throw(PermissionError()),attempts=2,pause=0)
    def broken(*args,**kwargs):raise PermissionError('persistent')
    monkeypatch.setattr(safety,'_replace',broken)
    monkeypatch.setattr(safety.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):safety.atomic_write(tmp_path/'journal.json',b'{}')
    assert not (tmp_path/'journal.json').exists()
    assert safety.stall_reason(900)=='no_progress_15min'
    assert safety.stall_reason(600,[(100,80,15500,16376)])=='wddm_stall_10min'
    assert safety.stall_reason(599,[(100,80,15500,16376)]) is None


@pytest.mark.parametrize('mode,n',[('visible',3),('n32',32)])
def test_shared_runner_cpu_orchestration(stage,tmp_path,monkeypatch,mode,n):
    """Exercise both orchestrator paths; replace only hardware/inference boundaries."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import subprocess
    from tools import exp169_c85r7ins_checks_v122 as checks
    from tools import release_cleanup_checks_v121 as cleanup
    from tools import vnext_r7_pl_worker_v1 as worker
    from tools import hb116_vnext_release as native
    binding=make_binding(stage,tmp_path/'binding')
    b=m.read(binding);b['runtime']={'competition':str(tmp_path/'competition')};m.dump(binding,b)
    r=review(stage,tmp_path/'review')
    ids=[f'cpu-orchestration-{i:02d}' for i in range(n)]
    raw={k:np.random.default_rng(i).random((n,12)).astype(np.float32)
         for i,k in enumerate(('R','S_R7','EXP143v2','EXP155','Q_A5q'))}
    failures={k:{} for k in raw}
    monkeypatch.setattr(checks,'snapshot',lambda *args:(ids,raw,failures))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'Tesla T4\nTesla T4\n')
    monkeypatch.setattr(m.pl,'runtime_preflight',lambda *a:None)
    monkeypatch.setattr(safety,'watchdog',lambda *a:nullcontext())
    monkeypatch.delenv(m.base.insurance.START_ENV,raising=False)
    monkeypatch.delenv(m.base.insurance.BUDGET_ENV,raising=False)
    for key in ('C85_COMPETITION',m.base.STRESS_ENV,m.base.insurance.FAULT_ENV):
        monkeypatch.setenv(key,'')
    module,rt,bootstrap=runner.load_runtime(stage,tmp_path/'loaded')
    calls=[]
    def cohort(path):
        path.mkdir(parents=True,exist_ok=True)
        (path/'test_series.csv').write_text(m.pl.UID+'\n'+'\n'.join(ids)+'\n')
    cohort(Path(b['runtime']['competition']))
    def baseline(runtime,output,**kwargs):
        calls.append('base')
        m.dump(output/module.RECEIPT,dict(flip_policy=m.flip.policy('FLIP_F0F2')))
    monkeypatch.setattr(module,'run_release',baseline)
    monkeypatch.setattr(module.old,'codec_receipt',lambda *a:{'status':'INACTIVE_TEST'})
    monkeypatch.setattr(module.codecs,'child_env',lambda env,*a,**kw:env)
    monkeypatch.setattr(runner,'load_runtime',lambda *a:(module,rt,bootstrap))
    def stress(runtime,output):
        calls.append('n32')
        cohort(output/'insurance-stress/competition')
        baseline(runtime,output/'insurance-stress/clean')
        return dict(status='MOCK_BASE_FAULT_BOUNDARY')
    monkeypatch.setattr(checks,'target_stress',stress)
    class Audit:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def verify(self,*a):return dict(cpu_fixture=True)
    monkeypatch.setattr(cleanup,'CleanupAudit',Audit)
    class Supervisor(Audit):
        def __init__(self,*a):pass
        def start(self,args,**kwargs):
            calls.append('PL')
            job=m.read(args[-1]);job['device']='cpu'
            def inference(path,device):
                assert device=='cpu'
                task=m.read(path);journal=native.c.Journal()
                for uid in task['ids']:
                    for entry in task['entries']:journal.record(uid,entry['id'],raw['EXP155'][ids.index(uid)])
            worker.native(job,worker=inference)
            return SimpleNamespace(returncode=0,reason='CPU_FIXTURE',cleanup_error=None)
        def wait(self,child):return child
    monkeypatch.setattr(module.insurance,'Supervisor',Supervisor)
    result=runner.run(stage,mode,binding,r,tmp_path/'output','1'*64,health(stage,binding,tmp_path/'health'))
    assert result['status']=='AUDIT_REQUIRED' and result['variants']==['a','b']
    assert result['studies']==n and not result['qualified_for_submission']
    assert calls.count('base')==1 and calls.count('PL')==2 and calls.count('n32')==(mode=='n32')
    assert len(result['outputs'])==2 and all(x['pl_failed_studies']==0 for x in result['outputs'].values())


def test_review_pins_underlying_smoke_evidence(stage,tmp_path):
    r=review(stage,tmp_path)
    assert m.review_gate(stage,r)['status']=='PASS'
    receipt=m.read(m.read(r['path'])['smoke_receipts'][0]['path'])
    m.dump(receipt['sources'][0]['path'],dict(changed=True))
    with pytest.raises(ValueError):m.review_gate(stage,r)


def test_watchdog_cpu_owned_process_cleanup(tmp_path,monkeypatch):
    import sys,time
    from tools import release_deadline_v1 as d
    monkeypatch.setattr(safety,'gpu_samples',lambda:[])
    monkeypatch.setattr(safety,'stall_reason',lambda age,samples:'cpu-test-stall' if age>=.1 else None)
    with safety.watchdog(d,tmp_path/'progress',tmp_path/'empty-bootstrap'):
        with d.Supervisor(d.Deadline(time.monotonic(),10),poll_seconds=.02) as supervisor:
            child=supervisor.run([sys.executable,'-c','import time; time.sleep(5)'],label='CPU-test')
    assert child.reason=='watchdog:cpu-test-stall'
    assert child.cleanup_error is None and child.process.poll() is not None


def test_throughput_collapse_and_recovery():
    guard=safety.ThroughputGuard(1.,0)
    for minute in range(1,30):assert guard.update(minute*60,minute*6) is None
    assert guard.update(1800,180)=='throughput_below_25pct_30min'
    guard=safety.ThroughputGuard(1.,0)
    assert guard.update(1740,174) is None
    assert guard.update(1800,234) is None  # recovered minute resets sustained-low timer
    assert guard.update(1860,240) is None
    for bad in (0,-1,float('nan')):
        with pytest.raises(ValueError):safety.ThroughputGuard(bad,0)


def test_health_gate_fail_closed(stage,tmp_path):
    binding=make_binding(stage,tmp_path/'binding')
    h=health(stage,binding,tmp_path)
    original=m.read(h['path'])
    assert m.health_gate(stage,binding,'1'*64,h)
    for key,value in [('full_shape',False),('peak_reserved_mib',[10241,9000]),
                      ('healthy_commits_per_second',{}),('flip_policy',{}),('sources',[])]:
        m.dump(h['path'],dict(original,**{key:value}))
        with pytest.raises(ValueError):m.health_gate(stage,binding,'1'*64,runner.asset(h['path']))
    with pytest.raises(ValueError):m.health_gate(stage,binding,'1'*64,None)


def test_commit_counters_cpu(tmp_path,monkeypatch):
    marker=tmp_path/'progress.json';monkeypatch.setenv('C85R7_V150_PROGRESS',str(marker))
    with safety.metadata_io(m.pl.metadata):
        m.pl.metadata.atomic_write(tmp_path/'journal-rows/000.json',b'{}')
        m.pl.metadata.atomic_write(tmp_path/'heartbeat.json',b'{}')
    assert safety.committed_records(marker)==1
