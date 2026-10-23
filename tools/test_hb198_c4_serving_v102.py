"""Cheap CPU-only C4 integration fixtures. No actual CUDA or real admission."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import pytest
from tools import hb198_c4_gold58_v102 as m

spec=importlib.util.spec_from_file_location('c4_gate_fixture',m.ROOT/'tests/test_hb198_c4_fix1.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)


def put(root,path,value):
    return support.write(root,str(path),value)


def approval(root,phase,subject,claim=None,changes=None,final='VERDICT: PASS'):
    work=root/m.REL
    names={'BUILD':'BUILD-REVIEW.md','MEMORY_LAUNCH':'MEMORY-LAUNCH-REVIEW.md','MEMORY_RESULT':'MEMORY-RESULT-REVIEW.md'}
    body=m.attestation(phase,m.sha(work/'registration.json'),subject,claim,'/root/test_c4_reviewer')
    body.update(changes or {})
    report=work/names[phase]
    text='# TEST-ONLY COMPLETE NATIVE PARSER FIXTURE; NOT REAL COMPLETION\nReviewer-Lane: native-astra-r1\n'
    text+=m.PREFIX+json.dumps(body,sort_keys=True)+'\n'+final+'\n'
    put(root,report.relative_to(root),text.encode())
    prompt=report.with_suffix('.prompt');prompt.write_text('TEST ONLY, NO PRODUCTION APPROVAL')
    completion=dict(schema=m.gate.native.SCHEMA,host=m.gate.native.HOST,model=m.gate.native.MODEL,
        agent_id=body['reviewer'],lane='native-astra-r1',prompt=str(prompt),report=str(report),
        terminal_status='completed',observed_at='2026-10-18T02:00:00+00:00',prompt_sha256=m.sha(prompt),
        report_sha256=m.sha(report),completion_message='TEST-ONLY synthetic native archive')
    cp=report.with_suffix('.completion.json');put(root,cp.relative_to(root),completion)
    m.gate.native.record(cp,'TEST ONLY, NEVER ACTUAL COMPLETION',root)
    return report


class BuildFixture(support.Fixture):
    """Construct only the real BUILD phase needed by this worker's first gate."""
    def __init__(self,root):
        self.constructing=True
        super().__init__(root)
        self.constructing=False
    def build(self,phase):
        if not self.constructing or phase=='BUILD_PASS':super().build(phase)
    def prepare_launch(self):pass
    def prepare_gold(self):pass
    def prepare_submission(self):pass


@pytest.fixture(scope='module')
def foundation(tmp_path_factory):
    tmp_path=tmp_path_factory.mktemp('c4-serving-native-fixture')
    f=BuildFixture(tmp_path)
    for p in m.SOURCES:put(tmp_path,p,(m.ROOT/p).read_bytes())
    reg=m.register(tmp_path,f.frozen,f.receipts['BUILD_PASS']);m.prepare(tmp_path,f.frozen)
    work=tmp_path/m.REL
    approval(tmp_path,'BUILD',m.sha(work/'registration.json'))
    c=dict(schema='h198-serving-resource/v1',owner='B',state='CLAIMED',candidate=m.gate.b.KEY,
        tag=m.TAGS['memory'],exclusive=True,physical_gpu=0,logical_device='cuda:0',
        device_name='NVIDIA GeForce RTX 4070 Ti SUPER',reservation='TEST-ONLY-RESOURCE',shared_evidence='TEST ONLY')
    put(tmp_path,m.REL/'memory-claim.json',c)
    approval(tmp_path,'MEMORY_LAUNCH',m.sha(work/'prepare.json'),m.sha(work/'memory-claim.json'))
    baseline={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    return f,reg,work,baseline


@pytest.fixture
def ready(foundation):
    original,reg,work,baseline=foundation
    f=copy.deepcopy(original)
    try:yield f,copy.deepcopy(reg),work
    finally:
        original.counter=max(original.counter,f.counter)
        history=f.root/'TEST-ONLY-HISTORY'/uuid.uuid4().hex
        for path,body in baseline.items():
            if not path.exists() or path.read_bytes()!=body:
                if path.exists():
                    dest=history/path.relative_to(f.root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
                path.write_bytes(body)
        # Preserve generated measurement/output fixtures outside the next case's
        # fixed paths. This never touches any production file or GPU output.
        for path in list(work.iterdir()):
            if path not in baseline and (path.is_dir() or path.is_file()):
                dest=history/path.relative_to(f.root);dest.parent.mkdir(parents=True,exist_ok=True);path.rename(dest)


def test_complete_real_schema_memory_positive_and_no_cycle(ready):
    f,reg,w=ready
    value=m.admit_memory(True,f.root,f.frozen)
    assert value['state']=='READY_FOR_MEMORY_ONLY' and value['fixture_only'] is True
    assert not (w/'memory-admission.json').exists() and not (w/'memory-run').exists()
    assert not value['qualified_for_submission']
    assert m.SELF in value['consumed_sha256']
    assert not any('REVIEW' in x for x in reg['source_sha256'])
    assert 'review' not in m.read(w/'memory-plan.json')
    with pytest.raises((ValueError,OSError)):m.gate.verify_receipt(f.receipts['BUILD_PASS'],phase='BUILD_PASS')


@pytest.mark.parametrize('name',['BUILD-REVIEW.md','MEMORY-LAUNCH-REVIEW.md','memory-claim.json','prepare.json','memory-plan.json','cache-integration.json','memory-queue.json','cache-queue.json'])
def test_missing_exact_phase_artifact_blocks(ready,name):
    f,_,w=ready;p=w/name;p.rename(p.with_name(p.name+'.TEST-ONLY-MISSING'))
    with pytest.raises((ValueError,OSError,KeyError)):m.authorize_memory(f.root,f.frozen)


@pytest.mark.parametrize('target',['report','receipt','completion','prompt','archive_report','archive_receipt','archive_completion','archive_prompt','archive_log'])
def test_complete_native_chain_missing_rejects(ready,target):
    f,_,w=ready;r=w/'MEMORY-LAUNCH-REVIEW.md';side=Path(str(r)+'.receipt.json');v=m.read(side)
    arc=f.root/'.local/astra-improve/lanes/runs'/v['lane']/v['run_id'];c=Path(v['completion_source']);e=m.read(c)
    paths=dict(report=r,receipt=side,completion=c,prompt=Path(e['prompt']),archive_report=arc/'report.md',
        archive_receipt=arc/'receipt.json',archive_completion=arc/'completion.json',archive_prompt=arc/'prompt.txt',archive_log=arc/'run.log')
    p=paths[target];p.rename(p.with_name(p.name+'.TEST-ONLY-MISSING'))
    with pytest.raises((ValueError,OSError,KeyError)):m.authorize_memory(f.root,f.frozen)


@pytest.mark.parametrize('changes,final',[
    ({'reviewer':m.BUILDER},'VERDICT: PASS'),({'phase':'BUILD'},'VERDICT: PASS'),
    ({'candidate':'K2'},'VERDICT: PASS'),({'kernel_sha256':'0'*64},'VERDICT: PASS'),
    ({'subject_sha256':'0'*64},'VERDICT: PASS'),({'registration_sha256':'0'*64},'VERDICT: PASS'),
    ({'resource_claim_sha256':'0'*64},'VERDICT: PASS'),({},'VERDICT: FAIL'),({},'VERDICT: PASS\nVERDICT: PASS'),
    ({},' VERDICT: FAIL\nVERDICT: PASS')])
def test_authentic_success_never_replaces_exact_independent_pass(ready,changes,final):
    f,_,w=ready
    for suffix in ('','.receipt.json'):
        p=Path(str(w/'MEMORY-LAUNCH-REVIEW.md')+suffix);p.rename(p.with_name(p.name+'.OLD-TEST-'+uuid.uuid4().hex))
    approval(f.root,'MEMORY_LAUNCH',m.sha(w/'prepare.json'),m.sha(w/'memory-claim.json'),changes,final)
    with pytest.raises((ValueError,KeyError)):m.authorize_memory(f.root,f.frozen)


@pytest.mark.parametrize('name',[m.SELF,m.TEST,'tools/native_codex_dispatch_record_v1.py','tools/job_runtime.py'])
def test_source_late_mutation_rejects(ready,name):
    f,_,_=ready;p=f.root/name;p.write_bytes(p.read_bytes()+b'\n# TEST-ONLY drift')
    with pytest.raises(ValueError):m.authorize_memory(f.root,f.frozen)


def test_last_recheck_catches_concurrent_claim_mutation(ready,monkeypatch):
    f,_,w=ready;original=m.gate.Inputs.recheck;changed=[]
    def mutate(self):
        if any(p.endswith('memory-claim.json') for p in self.consumed) and not changed:
            p=w/'memory-claim.json';p.write_bytes(p.read_bytes()+b'\n');changed.append(True)
        return original(self)
    monkeypatch.setattr(m.gate.Inputs,'recheck',mutate)
    with pytest.raises(ValueError,match='stale evidence'):m.authorize_memory(f.root,f.frozen)
    assert changed


def measured_fixture(f,reg,w):
    m.admit_memory(False,f.root,f.frozen)
    run=w/'memory-run';run.mkdir();raw=[];desktop=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',desktop_bytes=2**30,total_bytes=16376*2**20)
    for i,e in enumerate(f.original['models']):
        row=dict(schema='h198-memory-model/v1',fold=i,model_sha256=e['sha256'],representative_index=0,views=2,
            input_shape=[36,3,224,224],plan_sha256=reg['memory_plan']['sha256'],source_sha256=reg['source_sha256'][m.SELF],
            peak_reserved_bytes=8*m.GIB,peak_allocated_bytes=7*m.GIB,synchronized=True,model_deleted=True,seconds=1.,
            registration_sha256=m.sha(w/'registration.json'),desktop=desktop,
            environment=dict(os='win32',physical_gpu=0,logical_device='cuda:0',gpu='TEST ONLY 4070 Ti SUPER',
                python='TEST ONLY',torch='TEST ONLY',cuda='TEST ONLY',cudnn='TEST ONLY',timm='TEST ONLY',
                precision='original FP16 forward; FP32 sigmoid and two-probability mean'))
        raw.append(put(f.root,m.REL/f'memory-run/model-{i}.json',row))
    common=dict(source_sha256=reg['source_sha256'][m.SELF],registration_sha256=m.sha(w/'registration.json'),
        plan_sha256=reg['memory_plan']['sha256'],card_sha256=reg['cache_card']['sha256'],
        model_sha256=[e['sha256'] for e in f.original['models']],peak_reserved_bytes=[8*m.GIB]*5,raw_receipts=raw)
    worker=put(f.root,m.REL/'memory-run/worker.json',dict(common,state='MEASUREMENT_COMPLETE',mode='memory',
        failures=0,workload=5,models=5,studies=1,npz_sha256=None,kernel_sha256=m.KERNEL,
        admission_sha256=m.sha(w/'memory-admission.json')))
    process=put(f.root,m.REL/'memory-run/process.json',dict(child_exit_code=0,owned_cleanup_complete=True,survivors=[],source_sha256=common['source_sha256']))
    monitor=put(f.root,m.REL/'memory-run/monitor.json',dict(status='PASS',owned_cleanup_complete=True,child_exit_code=0,worker_sha256=worker['sha256'],process=process))
    value=dict(common,schema='h198-memory-evidence/v1',status='PASS_MEASURED',kernel_sha256=m.KERNEL,
        renderer_sha256=f.original['renderer_sha256'],K=36,tta='plane-aware-coronal-axial-flip',actual_full_shape_both_views=True,
        physical_gpu=0,worker=worker,monitor=monitor,raw_log=put(f.root,m.REL/'memory-run/worker.log',b'TEST-ONLY MEASUREMENT\n'),
        admission=m.ref(w/'memory-admission.json',f.root),
        launch=put(f.root,m.REL/'memory-run/ticket.json',dict(mode='memory',parent=dict(pid=123,create_time=1000.),
            admission=m.ref(w/'memory-admission.json',f.root),registration_sha256=m.sha(w/'registration.json'))),
        desktop=put(f.root,m.REL/'memory-run/desktop.json',desktop),
        bound_job=put(f.root,m.REL/'memory-run/bound-job.json',dict(schema='gpu-job/v2',tag=m.TAGS['memory'],phase='running',
            command=m.command('memory',f.root),command_sha256=m.jr.command_digest(m.command('memory',f.root),f.root),
            cwd=str(f.root),child_identity=dict(pid=123,create_time=1000.))),**desktop)
    put(f.root,m.REL/'memory-run/measurement.json',value)
    approval(f.root,'MEMORY_RESULT',m.sha(run/'measurement.json'))
    return value


def cache_fixture(f,reg,w):
    measured_fixture(f,reg,w)
    f.sources={**f.sources,**reg['source_sha256']}
    oldtest=m.read(f.root/f.refs['tests']['path']);oldtest['source_sha256']=f.sources
    f.refs['tests']=put(f.root,'synthetic/new-serving-tests.json',oldtest)
    f.refs['launch_card']=reg['cache_card'];f.refs['memory']=m.ref(w/'memory-run/measurement.json',f.root)
    c=m.read(w/'memory-claim.json');c['tag']=m.TAGS['cache']
    f.refs['resource_claim']=put(f.root,m.REL/'cache-claim.json',c)
    f.build('CACHE_LAUNCH_PASS')
    put(f.root,m.REL/'CACHE-ADMISSION.json',m.read(f.root/f.receipts['CACHE_LAUNCH_PASS']['path']))


def test_actual_gate_schema_cache_positive_after_independent_memory(ready):
    f,reg,w=ready;cache_fixture(f,reg,w)
    value=m.cache_admission(f.root,f.frozen)
    assert value['phase']=='CACHE_LAUNCH_PASS' and value['fixture_only'] and not value['qualified_for_submission']
    assert m.SELF in value['source_sha256']


@pytest.mark.parametrize('kind',['memory_peak','raw_identity','memory_review','cache_review','queue','worker','claim','wrong_phase'])
def test_cache_boundaries_actual_gate_and_measurement_negatives(ready,kind):
    f,reg,w=ready;cache_fixture(f,reg,w)
    if kind=='memory_peak':p=w/'memory-run/measurement.json';v=m.read(p);v['peak_reserved_bytes'][0]=13*m.GIB;put(f.root,p.relative_to(f.root),v)
    elif kind=='raw_identity':p=w/'memory-run/model-0.json';v=m.read(p);v['views']=1;put(f.root,p.relative_to(f.root),v)
    elif kind=='memory_review':p=w/'MEMORY-RESULT-REVIEW.md';p.write_text('VERDICT: PASS')
    elif kind=='cache_review':p=f.root/m.read(f.root/f.reviews['CACHE_LAUNCH_PASS']['path'])['report']['path'];p.write_text('VERDICT: FAIL')
    elif kind=='wrong_phase':put(f.root,m.REL/'CACHE-ADMISSION.json',m.read(f.root/f.receipts['BUILD_PASS']['path']))
    else:
        p={'queue':w/'cache-queue.json','worker':f.root/m.SELF,'claim':w/'cache-claim.json'}[kind];p.write_bytes(p.read_bytes()+b'\nTEST-ONLY-CHANGED')
    with pytest.raises((ValueError,OSError,KeyError)):m.cache_admission(f.root,f.frozen)


@pytest.mark.parametrize('peak,used,total,ok',[(12*m.GIB,1*m.GIB,16*m.GIB,True),(12*m.GIB+1,0,16*m.GIB,False),
    (11*m.GIB,4*m.GIB,16*m.GIB,False),(float('nan'),0,16*m.GIB,False),(0,1*m.GIB,16*m.GIB,True)])
def test_first_measurement_memory_not_circular(peak,used,total,ok):
    d=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',desktop_bytes=used,total_bytes=total)
    if ok:m.memory_ok(d,[peak])
    else:
        with pytest.raises(ValueError):m.memory_ok(d,[peak])


def test_model_load_diagnostic_failure_nonfatal(monkeypatch,tmp_path):
    calls=[]
    def fail(*a,**k):calls.append(a);raise PermissionError('TEST ONLY model-load status lock')
    monkeypatch.setattr(m.io,'atomic_json',fail)
    assert m.diagnostic(tmp_path/'phase.json',dict(phase='model_load',sequence=1)) is False
    assert calls and not (tmp_path/'phase.json').exists()


def test_npz_failure_never_publishes_success(tmp_path,monkeypatch):
    monkeypatch.setattr(m.io,'replace',lambda *a: (_ for _ in ()).throw(PermissionError('TEST ONLY')))
    with pytest.raises(PermissionError):m.atomic_npz(tmp_path/'c4-members.npz',values=np.ones((5,58,12),np.float32))
    assert not (tmp_path/'c4-members.npz').exists() and (tmp_path/'c4-members.tmp').exists()


@pytest.mark.parametrize('kind',['child_exit','cleanup','worker_missing','monitor_publication'])
def test_controller_completion_failure_is_not_success(tmp_path,monkeypatch,kind):
    root=tmp_path;work=root/m.REL;run=work/'cache-run';run.mkdir(parents=True)
    put(root,m.REL/'registration.json',dict(TEST_ONLY=True));put(root,m.REL/'CACHE-ADMISSION.json',dict(TEST_ONLY=True))
    process=dict(child_exit_code=1 if kind=='child_exit' else 0,owned_cleanup_complete=kind!='cleanup',
        survivors=[dict(TEST_ONLY=True)] if kind=='cleanup' else [],seconds=1.)
    reg=dict(source_sha256={m.SELF:'1'*64},cache_card=dict(sha256='2'*64))
    legacy=dict(models=[dict(sha256=str(i)*64) for i in range(5)],corpus=dict(ids=[str(i) for i in range(58)],targets=[str(i) for i in range(12)]))
    desktop=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',desktop_bytes=2**30,total_bytes=16*m.GIB)
    if kind=='monitor_publication':
        m.atomic_npz(run/'c4-members.npz',ids=np.asarray(legacy['corpus']['ids']),targets=np.asarray(legacy['corpus']['targets']),folds=np.arange(5),values=np.full((5,58,12),.5,np.float32))
        result=dict(mode='cache',state='INFERENCE_COMPLETE',workload=290,failures=0,models=5,studies=58,
            roster_sha256=m.jr.digest(legacy['corpus']['ids']),targets=legacy['corpus']['targets'],
            source_sha256='1'*64,card_sha256='2'*64,registration_sha256=m.sha(work/'registration.json'),
            admission_sha256=m.sha(work/'CACHE-ADMISSION.json'),kernel_sha256=m.KERNEL,
            model_sha256=[e['sha256'] for e in legacy['models']],study_seconds=[1.]*290,peak_reserved_bytes=[8*m.GIB]*5,
            npz_sha256=m.sha(run/'c4-members.npz'),environment=dict(physical_gpu=0,logical_device='cuda:0',gpu='4070 Ti SUPER',
                precision='original FP16 forward; FP32 sigmoid and two-probability mean'))
        put(root,m.REL/'cache-run/worker.json',result);put(root,m.REL/'cache-run/process.json',process)
        real=m.io.atomic_json
        def fail_monitor(path,value):
            if Path(path).name=='monitor.json':raise PermissionError('TEST ONLY monitor publication')
            return real(path,value)
        monkeypatch.setattr(m.io,'atomic_json',fail_monitor)
    with pytest.raises((ValueError,OSError)):m.finish('cache',run,process,reg,legacy,desktop,root)
    assert not (run/'monitor.json').exists()


def test_required_worker_and_journal_writes_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(m.io,'atomic_json',lambda *a,**k: (_ for _ in ()).throw(PermissionError('TEST ONLY persistent write')))
    with pytest.raises(PermissionError):m.fresh(tmp_path/'worker.json',dict(state='INFERENCE_COMPLETE'))
    with pytest.raises(m.io.JournalCommitError):m.io.journal(tmp_path/'journal.json',dict(identity='TEST ONLY'),tmp_path)
    assert not (tmp_path/'worker.json').exists()


@pytest.mark.parametrize('bad',['shape','dtype','nan','missing','duplicate'])
def test_full_output_contract(bad):
    ids=[str(i) for i in range(58)];v=np.ones((5,58,12),np.float32)
    if bad=='shape':v=v[:4]
    elif bad=='dtype':v=v.astype(np.float64)
    elif bad=='nan':v[0,0,0]=np.nan
    elif bad=='missing':ids=ids[:-1]
    else:ids[1]=ids[0]
    with pytest.raises(ValueError):m.finite_cache(v,ids,list(range(12)))


@pytest.mark.parametrize('bad',['pid','start','command','tag','parent'])
def test_pid_start_command_ownership(bad):
    own=m.jr.identity();parent=m.jr.identity(os.getppid());cmd=m.command('memory');rid='TESTONLY'
    item=dict(schema='gpu-job/v2',tag=m.TAGS['memory'],phase='running',command=cmd,command_sha256=m.jr.command_digest(cmd,m.ROOT),
        cwd=str(m.ROOT),child_identity=copy.deepcopy(own),pid=parent['pid'],create_time=parent['create_time'],run_id=rid)
    args=[str(m.ROOT/'tools/gpu_job.py'),'--_run-job',m.TAGS['memory'],rid]
    assert m.check_job(item,'memory',own,parent,cmd,cmd,args)==item
    if bad=='pid':item['child_identity']['pid']+=1
    elif bad=='start':item['child_identity']['create_time']-=1
    elif bad=='command':item['command']=['wrong']
    elif bad=='tag':item['tag']='oldtag'
    else:args[-1]='wrong'
    with pytest.raises(ValueError):m.check_job(item,'memory',own,parent,cmd,cmd,args)


def test_optional_telemetry_loss_and_real_stall_progress():
    p=m.Progress(0,dict(pid=1,create_time=1000),'token')
    p.check(899,None,None,1899)
    with pytest.raises(ValueError,match='900s'):p.check(900,None,None,1900)
    p=m.Progress(0,dict(pid=1,create_time=1000),'token')
    with pytest.raises(ValueError,match='WDDM'):
        p.check(600,None,dict(util=99,power=40,desktop_bytes=15*m.GIB,total_bytes=16*m.GIB),1600)
    p=m.Progress(0,dict(pid=1,create_time=1000),'token')
    s=dict(pid=1,token='token',time=1001,sequence=1,units=0,phase='model_load')
    p.check(1,s,None,1001);s['time']=1800;p.check(800,s,None,1800)
    assert p.last==1


@pytest.mark.parametrize('bad',[dict(pid=2),dict(token='wrong'),dict(units=581),dict(phase='touch'),dict(time=999999)])
def test_forged_progress_cannot_extend_watchdog(bad):
    p=m.Progress(0,dict(pid=1,create_time=1000),'token')
    s=dict(pid=1,token='token',time=1001,sequence=1,units=0,phase='model_load');s.update(bad)
    with pytest.raises(ValueError):p.check(1,s,None,1001)


def test_owned_cleanup_before_process_publication_failure(tmp_path,monkeypatch):
    run=tmp_path/'owned';run.mkdir();seen=[];real=m.terminate_owned
    def clean(root,owned):
        seen.append(m.jr.identity(root.pid));return real(root,owned)
    monkeypatch.setattr(m,'terminate_owned',clean)
    monkeypatch.setattr(m,'fresh',lambda *a,**k: (_ for _ in ()).throw(PermissionError('TEST ONLY receipt failure')))
    def sampler():raise RuntimeError('TEST ONLY interrupt')
    with pytest.raises(PermissionError):m.supervise([sys.executable,'-B','-c','import time;time.sleep(30)'],run,
        dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),'token',sample=sampler,poll=.01)
    assert seen and m.jr.liveness(seen[0])=='dead'


def test_no_fixture_cli_or_environment_admission_bypass(tmp_path):
    with pytest.raises(SystemExit):m.main(['measure','--root',str(tmp_path)])
    with pytest.raises(ValueError):m.fixture_scope(tmp_path,None)
    with pytest.raises(ValueError):m.run('memory')
    cp=subprocess.run([sys.executable,'-B','-c',"from tools import hb198_c4_gold58_v102; import sys; assert 'torch' not in sys.modules"],
        cwd=m.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
