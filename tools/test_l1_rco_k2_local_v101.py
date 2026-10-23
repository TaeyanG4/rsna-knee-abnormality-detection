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
from tools import l1_rco_k2_local_v101 as m
from tools.test_l1_rco_k2_v101 import production_schema, archive


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
    archive(root,'fixture-local-builder','6'*32,(m.REL/'BUILD-REPORT.md').as_posix(),registration_line+'\nFIXTURE ONLY\n')
    archive(root,'fixture-local-reviewer','7'*32,(m.REL/'BUILD-REVIEW.md').as_posix(),
        registration_line+'\nReviewer-Lane: fixture-local-reviewer\nCandidate-ID: '+m.CANDIDATE+
        '\nLocal-Wrapper-Build: PASS\nVERDICT: PASS\n')
    resource=dict(schema='k2-local-resource/v1',recorded_by='B',candidate_id=m.CANDIDATE,tag=m.TAG,
        state='CLAIMED',exclusive=True,physical_index=0,logical_device='cuda:0',
        gpu_name='NVIDIA GeForce RTX 4070 Ti SUPER',reservation='FIXTURE-ONLY',shared_claim_evidence='TEST FIXTURE')
    m.fresh_json(work/'resource-claim.json',resource)
    archive(root,'fixture-local-launcher','8'*32,(m.REL/'LAUNCH-REVIEW.md').as_posix(),
        '\n'.join([registration_line,'Reviewer-Lane: fixture-local-launcher','Candidate-ID: '+m.CANDIDATE,
        'K2-Local-Prepare-SHA256: '+m.sha(work/'prepare.json'),
        'K2-Build-Gate-SHA256: '+m.sha(root/m.FIX1/'build-gate.json'),
        'Resource-Claim-SHA256: '+m.sha(work/'resource-claim.json'),'Local-CUDA-Launch: PASS','VERDICT: PASS'])+'\n')
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
        "from tools import l1_rco_k2_local_v101; import sys; assert 'torch' not in sys.modules"],
        cwd=m.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
