"""CPU-only external wrapper fixtures; no CUDA, model or production admission."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pytest
from tools import l1_rco_k2_local_v103 as m
from tools import native_codex_dispatch_record_v1 as native
from tools.test_l1_rco_k2_v101 import production_schema


def hb(pid=7, at=1000., phase='render', done=0, windows=0):
    return dict(pid=pid, time=at, phase=phase, completed=done, windows=windows)


def policy():
    return m.Progress(0., dict(pid=7, create_time=1000.))


def test_phase_progress_without_windows_and_touches_do_not_reset():
    p = policy()
    assert p.update(0, hb(), None, 1000) is None
    assert p.update(400, hb(at=1400, phase='classifier'), None, 1400) is None
    assert p.last == 400 and p.windows == 0
    assert p.update(800, hb(at=1800, phase='classifier'), None, 1800) is None
    assert p.last == 400
    assert 'zero-progress' in p.update(1300, hb(at=2300, phase='classifier'), None, 2300)


@pytest.mark.parametrize('bad',[
    {'pid':999}, {'time':999999}, {'time':998}, {'completed':-1}, {'windows':-1},
    {'windows':float('nan')}, {'phase':'file-written'}, {'phase':'complete','completed':2}])
def test_forged_progress_rejected(bad):
    p=policy(); x=hb(); x.update(bad)
    with pytest.raises((ValueError,KeyError)):p.update(1,x,None,1001)


def test_regressing_or_fake_window_progress_rejected():
    p=policy();p.update(0,hb(phase='classifier'),None,1000)
    with pytest.raises(ValueError):p.update(1,hb(at=1001,phase='render'),None,1001)
    p=policy();p.update(0,hb(),None,1000)
    with pytest.raises(ValueError):p.update(1,hb(at=1001,windows=32),None,1001)


def test_missing_status_unavailable_telemetry_and_wddm_combination():
    p=policy()
    assert p.update(899,None,None,1899) is None
    assert 'zero-progress' in p.update(900,None,None,1900)
    sample=dict(util=95,power=89,desktop_bytes=15*m.GIB,total_bytes=16*m.GIB)
    p=policy()
    assert p.update(599,None,sample,1599) is None
    assert 'WDDM' in p.update(600,None,sample,1600)
    p=policy();p.update(590,hb(at=1590),None,1590)
    assert p.update(600,hb(at=1600),sample,1600) is None
    p=policy()
    assert p.update(600,None,dict(sample,power=100),1600) is None


def test_throughput_real_active_phase_clock_and_no_touch_reset():
    p=policy();p.update(0,hb(phase='classifier'),None,1000)
    p.update(120,hb(at=1120,phase='classifier-chunk',windows=1200),None,1120)
    assert p.healthy == 10
    reason=None
    for t in range(180,1981,60):
        reason=p.update(t,hb(at=1000+t,phase='classifier-chunk',windows=1200+(t-120)),None,1000+t)
    assert reason == 'throughput_below_25pct_30min'
    # Advancing to the next render pauses comparable time, not the accumulated slow clock.
    active=p.active
    p.update(1982,hb(at=2982,phase='render',done=1,windows=p.windows),None,2982)
    p.update(2200,hb(at=3200,phase='classifier',done=1,windows=p.windows),None,3200)
    assert p.active == active+2
    assert p.guard.low_since is not None


@pytest.mark.parametrize('peak,before,total,ok',[
    (12*m.GIB,2*m.GIB,16*m.GIB,True), (12*m.GIB+1,0,16*m.GIB,False),
    (11*m.GIB,4*m.GIB,16*m.GIB,False), (float('nan'),0,16*m.GIB,False),
    (0,15*m.GIB,16*m.GIB,False)])
def test_memory_real_measurements(peak,before,total,ok):
    assert m.memory_ok(peak,before,total) is ok


def test_bounded_telemetry_reuses_actual_parser_and_python_filter(monkeypatch):
    seen=[]
    def fake(command,**kwargs):
        seen.append(kwargs)
        out=('0, NVIDIA GeForce RTX 4070 Ti SUPER, 16376, 2048, 5, 30\n'
             if '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw' in command
             else '1, C:\\Windows\\explorer.exe\n2, chrome.exe\n3, python.exe\n')
        return subprocess.CompletedProcess(command,0,out,'')
    monkeypatch.setattr(m.subprocess,'run',fake)
    sample=m.telemetry();assert sample['total_bytes']==16376*2**20
    assert m.bounded_legacy(m.legacy.active_ml_processes)()==['3, python.exe']
    assert all(x['timeout']==3 for x in seen)
    if os.name=='nt':assert all(x['creationflags']==subprocess.CREATE_NO_WINDOW for x in seen)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k: (_ for _ in ()).throw(subprocess.TimeoutExpired('fixture',3)))
    assert m.telemetry() is None


def native_archive(root,lane,report,content,agent_id,observed='2026-10-18T13:00:00+09:00'):
    """Complete production-schema native chain in a TEST-ONLY isolated root."""
    path=root/report;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
    prompt=path.with_name(path.name+'.prompt.md')
    prompt.write_text('TEST-ONLY native fixture; no actual task, GPU or production admission.\n')
    completion=path.with_name(path.name+'.completion.json')
    value=dict(schema=native.SCHEMA,host=native.HOST,model=native.MODEL,agent_id=agent_id,
        lane=lane,prompt=str(prompt),report=str(path),terminal_status='completed',observed_at=observed,
        prompt_sha256=m.sha(prompt),report_sha256=m.sha(path),completion_message='TEST-ONLY observed fixture')
    completion.write_bytes(native.wire(value))
    receipt=native.record(completion,'TEST-ONLY isolated completion, never production',root)
    native.verify(receipt,root)
    return path


@pytest.fixture
def local_schema(production_schema,monkeypatch):
    f=production_schema; root=f['root']; work=root/m.REL
    for name in m.SOURCES:
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((m.ROOT/name).read_bytes())
    monkeypatch.setattr(m,'REG_SHA',m.sha(f['reg']))
    monkeypatch.setattr(m,'ASSET_SHA',m.sha(root/m.FIX1/'assets-v1/asset-manifest.json'))
    driver=root/'nvdxgdmal64.dll';driver.write_bytes(b'CPU FIXTURE, NOT A DLL')
    m.register(driver,m.sha(driver),root)
    m.prepare(work/'registration.json',root)
    (root/m.FIX1/'build-gate.json').write_bytes((root/'build-gate.json').read_bytes())
    registration_line='K2-Local-Registration-SHA256: '+m.sha(work/'registration.json')
    native_archive(root,'native-astra-b1',(m.REL/'BUILD-REPORT.md').as_posix(),
        'Kernel-SHA256: '+m.jr.read_json(work/'registration.json')['kernel_sha256']+'\n'+registration_line+
        '\nTEST-ONLY FIXTURE\n','/root/test_k2_builder')
    native_archive(root,'native-astra-r1',(m.REL/'BUILD-REVIEW.md').as_posix(),
        registration_line+'\nReviewer-Lane: native-astra-r1\nCandidate-ID: '+m.CANDIDATE+
        '\nLocal-Wrapper-Build: PASS\nVERDICT: PASS\n','/root/test_k2_reviewer')
    resource=dict(schema='k2-local-resource/v1',recorded_by='B',candidate_id=m.CANDIDATE,tag=m.TAG,
        state='CLAIMED',exclusive=True,physical_index=0,logical_device='cuda:0',
        gpu_name='NVIDIA GeForce RTX 4070 Ti SUPER',reservation='FIXTURE-ONLY',shared_claim_evidence='TEST FIXTURE')
    m.fresh_json(work/'resource-claim.json',resource)
    native_archive(root,'native-astra-r1',(m.REL/'LAUNCH-REVIEW.md').as_posix(),
        '\n'.join([registration_line,'Reviewer-Lane: native-astra-r1','Candidate-ID: '+m.CANDIDATE,
        'K2-Local-Prepare-SHA256: '+m.sha(work/'prepare.json'),
        'K2-Build-Gate-SHA256: '+m.sha(root/m.FIX1/'build-gate.json'),
        'Resource-Claim-SHA256: '+m.sha(work/'resource-claim.json'),'Local-CUDA-Launch: PASS','VERDICT: PASS'])+'\n',
        '/root/test_k2_reviewer')
    return root,work


def test_actual_register_prepare_gate_archives_and_admission(local_schema):
    root,work=local_schema
    result=m.admit(work/'prepare.json',True,root)
    assert result['status']=='READY_FOR_GPU_JOB' and result['fixture_only']
    assert not (work/'admission.json').exists()
    m.admit(work/'prepare.json',False,root)
    assert m.replay_admission(work/'admission.json',root)['runtime_checks_pending']
    with pytest.raises((ValueError,OSError)):
        m.replay_admission(work/'admission.json')  # Production root rejects fixture.
    assert not result['qualified_for_submission']


def test_admission_streams_weight_pins_without_oversize_read(local_schema,monkeypatch):
    root,work=local_schema
    real=m.base.Inputs.read
    def no_weights(self,path):
        assert Path(path).suffix!='.pt','weight must use streamed digest, never bounded document read'
        return real(self,path)
    monkeypatch.setattr(m.base.Inputs,'read',no_weights)
    assert m.authorize(work/'prepare.json',root)['status']=='READY_FOR_GPU_JOB'


@pytest.mark.parametrize('mutation',['source','build','review','claim','prepare','driver','self-review'])
def test_wrong_or_absent_admissions_fail_closed(local_schema,mutation):
    root,work=local_schema
    if mutation=='source':(root/m.SELF).write_text('modified source')
    elif mutation=='build':(root/m.FIX1/'build-gate.json').write_text('{}')
    elif mutation=='review':(work/'BUILD-REVIEW.md').write_text('VERDICT: PASS')
    elif mutation=='claim':
        p=work/'resource-claim.json';r=m.jr.read_json(p);r['exclusive']=False;m.io.atomic_json(p,r)
    elif mutation=='prepare':
        p=work/'prepare.json';r=m.jr.read_json(p);r['queue'][0]['command'].append('--bypass');m.io.atomic_json(p,r)
    elif mutation=='driver':(root/'nvdxgdmal64.dll').write_bytes(b'wrong')
    else:
        p=work/'BUILD-REVIEW.md.receipt.json';r=m.jr.read_json(p);r['lane']='fixture-local-builder';m.io.atomic_json(p,r)
    result=m.admit(work/'prepare.json',True,root)
    assert result['status']=='BLOCKED' and not (work/'admission.json').exists()


@pytest.mark.parametrize('bad',['pid','start','command','tag','resource'])
def test_job_identity_and_resource_fail_closed(bad):
    own=m.jr.identity();parent=m.jr.identity(os.getppid());command=['fixture-command']
    record=dict(schema='gpu-job/v2',tag=m.TAG,phase='running',command=command,
        command_sha256=m.jr.command_digest(command,m.ROOT),cwd=str(m.ROOT),child_identity=own,
        pid=parent['pid'],create_time=parent['create_time'])
    assert m.check_job(record,own,parent,command)==record
    if bad=='resource':
        with pytest.raises((ValueError,KeyError)):m.check_resource({'schema':'fixture'})
        return
    r=copy.deepcopy(record)
    if bad=='pid':r['child_identity']['pid']+=1
    elif bad=='start':r['child_identity']['create_time']-=100
    elif bad=='command':r['command']=['wrong']
    else:r['tag']='another'
    with pytest.raises(ValueError):m.check_job(r,own,parent,command)


CHILD = r'''
import json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,os.getcwd())
from tools.hb150_io_v1 import atomic_json
p=Path(sys.argv[1]);mode=sys.argv[2];s=p/'scratch';s.mkdir()
def put(phase='render',done=0,windows=0):
 v=dict(pid=os.getpid()+(1 if mode=='forged' else 0),time=time.time(),phase=phase,completed=done,windows=windows)
 atomic_json(s/'status.json',v,critical=False)
if mode=='tree':
 subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
if mode!='missing':put()
if mode=='normal':
 time.sleep(.1);put('classifier');time.sleep(.1);put('classifier-chunk',windows=32)
 time.sleep(.1);put('complete',done=3,windows=32);time.sleep(.1)
else:
 while True:
  if mode in ('stuck','wddm'):put()
  time.sleep(.03)
'''


class FastClock:
    def __init__(self,attempt):self.value=0.;self.calls=0;self.attempt=attempt
    def now(self):return self.value
    def sleep(self,_):
        time.sleep(.02);self.calls+=1
        # Allow native child startup, then advance watchdog time without a real wait.
        if self.calls>10:self.value+=300


@pytest.mark.parametrize('mode',['normal','missing','stuck','forged','wddm','tree'])
def test_real_owned_tiny_child_supervision(tmp_path,mode):
    attempt=tmp_path/'attempt';attempt.mkdir();script=tmp_path/'child.py';script.write_text(CHILD)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1');env.pop('S6PLUS_PIN_DLL',None)
    kwargs=dict(sample=lambda:None,poll=.02)
    if mode!='normal':
        clock=FastClock(attempt);kwargs.update(clock=clock.now,sleep=clock.sleep)
    if mode=='wddm':kwargs['sample']=lambda:dict(util=99,power=40,desktop_bytes=15*m.GIB,total_bytes=16*m.GIB)
    result=m._supervise([sys.executable,'-B',str(script),str(attempt),mode],attempt,env,**kwargs)
    if mode=='normal':assert result['exit_code']==0 and result['reason']=='completed'
    elif mode=='forged':assert result['exit_code']==125 and result['reason']=='supervisor_ValueError'
    elif mode=='wddm':assert result['exit_code']==124 and 'WDDM' in result['reason']
    else:assert result['exit_code']==124 and 'zero-progress' in result['reason']
    assert m.jr.liveness(result['child'])=='dead' and not result['survivors']
    for identity in result['owned_descendants']:assert m.jr.liveness(identity)=='dead'
    if mode=='tree':assert result['owned_descendants']
    assert (attempt/'process-receipt.json').exists() and (attempt/'worker.log').exists()


def test_persistent_diagnostic_failure_cannot_prevent_cleanup(tmp_path,monkeypatch):
    attempt=tmp_path/'attempt';attempt.mkdir();script=tmp_path/'child.py';script.write_text(CHILD)
    clock=FastClock(attempt)
    monkeypatch.setattr(m.io,'atomic_json',lambda *a,**k: (_ for _ in ()).throw(PermissionError('fixture')))
    result=m._supervise([sys.executable,'-B',str(script),str(attempt),'tree'],attempt,
        dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),sample=lambda:None,clock=clock.now,sleep=clock.sleep)
    assert result['exit_code']==124 and not result['diagnostics_complete'] and not result['survivors']
    assert m.jr.liveness(result['child'])=='dead'
    assert result['owned_descendants']
    assert all(m.jr.liveness(i)=='dead' for i in result['owned_descendants'])


def test_sampler_exception_finally_cleans_owned_child(tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir();script=tmp_path/'child.py';script.write_text(CHILD)
    result=m._supervise([sys.executable,'-B',str(script),str(attempt),'missing'],attempt,
        dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),sample=lambda: (_ for _ in ()).throw(RuntimeError('fixture')))
    assert result['exit_code']==125 and m.jr.liveness(result['child'])=='dead' and not result['survivors']


def test_no_cli_fixture_bypass_and_child_requires_actual_ticket(tmp_path):
    with pytest.raises(ValueError):m.child(tmp_path/'ticket.json')
    with pytest.raises(ValueError):m.execute(tmp_path/'admission.json')
    with pytest.raises(SystemExit):m.main(['execute','--admission',str(tmp_path/'x'),'--cpu'])
    source=(m.ROOT/m.SELF).read_text()
    assert 'CREATE_NO_WINDOW' in source and 'runpy.run_path(sys.argv[0]' in source


def test_collect_refuses_live_or_unconfirmed_job(monkeypatch):
    monkeypatch.setattr(m.jr,'job_status',lambda *a,**k:dict(state='running'))
    with pytest.raises(ValueError,match='completed bound'):m.collect()


@pytest.mark.parametrize('mutation',[None,'memory','codec','timing','output','environment'])
def test_real_result_validation_and_complete_timing(production_schema,monkeypatch,mutation):
    f=production_schema;root=f['root'];attempt=root/'visible'
    monkeypatch.setattr(m,'ROOT',root)
    monkeypatch.setattr(m,'ASSET_SHA',m.sha(root/m.FIX1/'assets-v1/asset-manifest.json'))
    p=attempt/'worker-receipt.json';r=m.jr.read_json(p)
    name='NVIDIA GeForce RTX 4070 Ti SUPER'
    r['environment'].update(platform='Windows fixture',gpu=name)
    r['memory']['device']=name
    if mutation=='memory':r['peak_reserved_bytes']=13*m.GIB
    if mutation=='codec':r['codecs']['status']='MISSING'
    if mutation=='timing':del r['timing']['codec_header_seconds']
    if mutation=='environment':r['environment']['autocast']=True
    if mutation=='output':r['output_sha256']='0'*64
    m.io.atomic_json(p,r)
    process=dict(exit_code=0,survivors=[],full_wall_seconds=30.)
    before=dict(desktop_bytes=2*m.GIB,total_bytes=16*m.GIB)
    if mutation:
        with pytest.raises((ValueError,KeyError)):m.validate_result(attempt,{},process,before)
    else:
        result=m.validate_result(attempt,{},process,before)
        assert result['local_all_in_timing']['total_seconds']==30
        assert result['status']=='LOCAL_COMPLETE_PENDING_INDEPENDENT_REVIEW'
        assert not result['qualified_for_submission']


def test_external_module_import_is_cpu_only():
    cp=subprocess.run([sys.executable,'-B','-c',
        "from tools import l1_rco_k2_local_v103; import sys; assert 'torch' not in sys.modules"],
        cwd=m.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr


def chain(root,work,name='BUILD-REVIEW.md'):
    sidecar=work/(name+'.receipt.json');receipt=native.parse(sidecar.read_bytes())
    arc=root/'.local/astra-improve/lanes/runs'/receipt['lane']/receipt['run_id']
    completion=Path(receipt['completion_source']);evidence=native.parse(completion.read_bytes())
    return dict(sidecar=sidecar,receipt=receipt,archive=arc,completion=completion,evidence=evidence,
                prompt=Path(evidence['prompt']),report=Path(evidence['report']))


def republish_fixture(c,changes=None,receipt_changes=None,log_changes=None):
    """Coherently mutate TEST-ONLY archives so tests reach the intended gate."""
    e=c['evidence'];r=c['receipt'];e.update(changes or {});r.update(receipt_changes or {})
    e['prompt_sha256']=m.sha(c['prompt']);e['report_sha256']=m.sha(c['report'])
    for key in ('lane','agent_id','host','model'):r[key]=e[key]
    r['completed']=e['observed_at'];r['prompt_sha256']=e['prompt_sha256'];r['report_sha256']=e['report_sha256']
    raw=native.wire(e);c['completion'].write_bytes(raw);(c['archive']/'completion.json').write_bytes(raw)
    r['completion_evidence_sha256']=native.digest(raw)
    (c['archive']/'prompt.txt').write_bytes(c['prompt'].read_bytes())
    (c['archive']/'report.md').write_bytes(c['report'].read_bytes())
    log=native.parse((c['archive']/'run.log').read_bytes())
    for key in ('lane','agent_id','host','model','note'):log[key]=r[key]
    log.update(log_changes or {});(c['archive']/'run.log').write_bytes(native.wire(log))
    r['log_sha256']=m.sha(c['archive']/'run.log')
    payload=native.wire(r);c['sidecar'].write_bytes(payload);(c['archive']/'receipt.json').write_bytes(payload)


def test_complete_native_chain_pinned_and_shared_reviewer_allowed(local_schema):
    root,work=local_schema
    result=m.authorize(work/'prepare.json',root)
    assert result['builder_agent_id']=='/root/test_k2_builder'
    assert result['reviewer_agent_id']==result['launch_reviewer_agent_id']=='/root/test_k2_reviewer'
    assert result['reviewer_lane']==result['launch_reviewer_lane']=='native-astra-r1'
    for filename in ('BUILD-REPORT.md','BUILD-REVIEW.md','LAUNCH-REVIEW.md'):
        c=chain(root,work,filename)
        native.verify(c['sidecar'],root)
        expected=[c[k] for k in ('sidecar','completion','prompt','report')]
        expected += [c['archive']/n for n in ('receipt.json','completion.json','prompt.txt','report.md','run.log')]
        assert all(p.relative_to(root).as_posix() in result['input_sha256'] for p in expected)
    assert m.NATIVE in result['input_sha256']


@pytest.mark.parametrize('filename',['BUILD-REPORT.md','BUILD-REVIEW.md','LAUNCH-REVIEW.md'])
@pytest.mark.parametrize('target',['completion','archive_completion','prompt','archive_prompt','report','archive_report','log','receipt','archive_receipt'])
def test_missing_or_changed_native_chain_never_admits(local_schema,filename,target):
    root,work=local_schema;c=chain(root,work,filename)
    paths={**{k:c[k] for k in ('completion','prompt','report')},'receipt':c['sidecar'],
           **{'archive_'+k:c['archive']/n for k,n in [('completion','completion.json'),('prompt','prompt.txt'),('report','report.md'),('receipt','receipt.json')]},
           'log':c['archive']/'run.log'}
    p=paths[target]
    # A moved original is missing; no fixture or user file is deleted.
    p.rename(p.with_name(p.name+'.missing-test-only'))
    result=m.admit(work/'prepare.json',True,root)
    assert result['status']=='BLOCKED' and not (work/'admission.json').exists()


@pytest.mark.parametrize('target',['completion','archive_completion','prompt','archive_prompt','report','archive_report','log','receipt','archive_receipt'])
def test_late_native_mutation_rejects_sealed_admission(local_schema,target):
    root,work=local_schema
    assert m.admit(work/'prepare.json',False,root)['status']=='READY_FOR_GPU_JOB'
    c=chain(root,work)
    paths={**{k:c[k] for k in ('completion','prompt','report')},'receipt':c['sidecar'],
           **{'archive_'+k:c['archive']/n for k,n in [('completion','completion.json'),('prompt','prompt.txt'),('report','report.md'),('receipt','receipt.json')]},
           'log':c['archive']/'run.log'}
    p=paths[target];p.write_bytes(p.read_bytes()+b'\nchanged TEST-ONLY')
    with pytest.raises(ValueError,match='admission input mutation'):m.replay_admission(work/'admission.json',root)


@pytest.mark.parametrize('filename',['BUILD-REVIEW.md','LAUNCH-REVIEW.md'])
def test_same_actual_agent_under_distinct_lane_rejects(local_schema,filename):
    root,work=local_schema;c=chain(root,work,filename)
    republish_fixture(c,{'agent_id':'/root/test_k2_builder'})
    assert native.verify(c['sidecar'],root)['agent_id']=='/root/test_k2_builder'
    result=m.admit(work/'prepare.json',True,root)
    assert result['status']=='BLOCKED' and 'agent/lane' in result['error']


@pytest.mark.parametrize('mutation',['host','model','agent','lane','time_naive','time_future','time_recorded_before','time_review_before','time_launch_before','terminal','log_identity'])
def test_native_identity_and_chronology_rejects(local_schema,mutation):
    root,work=local_schema;c=chain(root,work,'LAUNCH-REVIEW.md' if mutation=='time_launch_before' else 'BUILD-REVIEW.md')
    if mutation=='host':republish_fixture(c,{'host':'Claude Code'})
    elif mutation=='model':republish_fixture(c,{'model':'opus'})
    elif mutation=='agent':republish_fixture(c,{'agent_id':'invented'})
    elif mutation=='lane':
        c['receipt']['lane']='native-astra-r99'
        c['sidecar'].write_bytes(native.wire(c['receipt']))
        (c['archive']/'receipt.json').write_bytes(c['sidecar'].read_bytes())
    elif mutation=='time_naive':republish_fixture(c,{'observed_at':'2026-10-18T13:00:00'})
    elif mutation=='time_future':republish_fixture(c,{'observed_at':'2099-01-01T00:00:00+00:00'}, {'recorded_at':'2099-01-01T00:00:01+00:00'})
    elif mutation=='time_recorded_before':republish_fixture(c,receipt_changes={'recorded_at':'2026-10-18T01:00:00+00:00'})
    elif mutation=='time_review_before':republish_fixture(c,{'observed_at':'2026-10-18T12:59:59+09:00'})
    elif mutation=='time_launch_before':
        bc=chain(root,work,'BUILD-REVIEW.md');republish_fixture(bc,{'observed_at':'2026-10-18T13:00:01+09:00'})
    elif mutation=='terminal':republish_fixture(c,{'terminal_status':'running'})
    else:republish_fixture(c,log_changes={'agent_id':'/root/another_agent'})
    assert m.admit(work/'prepare.json',True,root)['status']=='BLOCKED'


@pytest.mark.parametrize('mutation',['build-fail','launch-fail','wrong-phase','wrong-reg','wrong-candidate','duplicate-pass','swap-fail-receipt','wrong-kernel'])
def test_authentic_native_success_never_substitutes_for_exact_pass(local_schema,mutation):
    root,work=local_schema
    name='LAUNCH-REVIEW.md' if mutation=='launch-fail' else 'BUILD-REPORT.md' if mutation=='wrong-kernel' else 'BUILD-REVIEW.md'
    c=chain(root,work,name);body=c['report'].read_text()
    if mutation=='swap-fail-receipt':
        other=native_archive(root,'native-astra-r1',(m.REL/'OTHER-FAIL.md').as_posix(),
            body.replace('VERDICT: PASS','VERDICT: FAIL'),'/root/test_k2_reviewer')
        c['sidecar'].write_bytes(Path(str(other)+'.receipt.json').read_bytes())
    else:
        if mutation in ('build-fail','launch-fail'):body=body.replace('VERDICT: PASS','VERDICT: FAIL')
        elif mutation=='wrong-phase':body=body.replace('Local-Wrapper-Build: PASS','Local-CUDA-Launch: PASS')
        elif mutation=='wrong-reg':body=body.replace(m.sha(work/'registration.json'),'0'*64)
        elif mutation=='wrong-candidate':body=body.replace('Candidate-ID: '+m.CANDIDATE,'Candidate-ID: OTHER')
        elif mutation=='wrong-kernel':body=body.replace('Kernel-SHA256: ','Kernel-SHA256: '+('0'*64))
        else:body+='VERDICT: PASS\n'
        c['report'].write_text(body);republish_fixture(c)
        assert native.verify(c['sidecar'],root)['state']=='SUCCESS'
    assert m.admit(work/'prepare.json',True,root)['status']=='BLOCKED'


def test_native_source_checked_before_execution(local_schema):
    root,work=local_schema
    (root/m.NATIVE).write_text("raise RuntimeError('UNVERIFIED SOURCE EXECUTED')\n")
    result=m.admit(work/'prepare.json',True,root)
    assert result['status']=='BLOCKED' and 'source/asset changed' in result['error']
    # Even direct bounded resolver cannot execute an unpinned helper.
    expected=m.jr.read_json(work/'registration.json')['input_sha256'][m.NATIVE]
    with pytest.raises(ValueError,match='source changed before use'):
        m.native_dispatch(work/'BUILD-REPORT.md.receipt.json',m.base.Inputs(root),expected)


def test_mutation_during_native_snapshot_caught_by_late_recheck(local_schema,monkeypatch):
    root,work=local_schema;c=chain(root,work,'BUILD-REPORT.md');original=m.base.Inputs.read;changed=[]
    def read_then_change(self,path):
        result=original(self,path)
        if Path(path)==c['archive']/'run.log' and not changed:
            c['completion'].write_bytes(c['completion'].read_bytes()+b'\nTEST-ONLY LATE CHANGE');changed.append(True)
        return result
    monkeypatch.setattr(m.base.Inputs,'read',read_then_change)
    result=m.admit(work/'prepare.json',True,root)
    assert changed and result['status']=='BLOCKED' and 'changed during gate' in result['error']


def test_old_released_tag_claim_cannot_be_reused(local_schema):
    root,work=local_schema;p=work/'resource-claim.json';value=m.jr.read_json(p)
    value['tag']='K2_RCOF0_FIX1_LOCAL_WRAPPED_1';m.io.atomic_json(p,value)
    result=m.admit(work/'prepare.json',True,root)
    assert result['status']=='BLOCKED' and 'resource claim' in result['error']


def test_production_actual_archived_fail_remains_fail():
    report=m.ROOT/'.local/scratch/slot-b/k2-release/fix1/local-wrapper/BUILD-REVIEW.md'
    assert m.sha(report)=='a83bdf3191efe2c03ddb7f8ca976f234ad93c0fc00423b0ab76345fccd9234cd'
    record=native.verify(Path(str(report)+'.receipt.json'),m.ROOT)
    assert record['agent_id']=='/root/k1_result_review' and record['state']=='SUCCESS'
    with pytest.raises(ValueError,match='VERDICT: PASS'):m.base.verdict(report.read_bytes())
