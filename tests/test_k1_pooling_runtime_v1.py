"""v1.0.1: POOL-R1 deadline/telemetry/cleanup CPU regressions.

Changelog: reproduce the reviewer's 5409.05-second false SUCCESS and exercise
bounded telemetry, termination reserve, source pins and independent repair review.
"""
import json
import io
import subprocess
import sys
import time
import pytest
from tools import k1_pooling_v1 as p
from tools import l1_rco_runtime_v120 as safety


def test_direct_run_cannot_bypass_review(monkeypatch):
    monkeypatch.setattr(p,'admit',lambda:dict(status='BLOCKED'))
    monkeypatch.setattr(p,'owner_check',lambda:pytest.fail('ownership before review'))
    with pytest.raises(ValueError,match='admission'):p.run()


@pytest.mark.parametrize('fault',['missing','self','repair_self','repair_missing','bindings','report','builder'])
def test_review_fails_closed(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(p,'OUT',tmp_path); monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'review_bindings',lambda:{'smoke':'bound'})
    if fault!='missing':
        (tmp_path/'review.md').write_text('Independent review evidence')
        r=dict(verdict='PASS',reviewer='astra-5',builder='astra-b1',repair_builder='astra-b2',bindings={'smoke':'bound'},report='review.md',report_sha256=p.sha(tmp_path/'review.md'))
        if fault=='self':r['reviewer']='astra-b1'
        if fault=='repair_self':r['reviewer']='astra-b2'
        if fault=='repair_missing':r.pop('repair_builder')
        if fault=='bindings':r['bindings']={}
        if fault=='report':r['report_sha256']='bad'
        if fault=='builder':r['builder']='other'
        (tmp_path/'REVIEW.json').write_text(json.dumps(r))
    with pytest.raises((OSError,ValueError)):p.require_review()


def test_worker_rejects_unowned_process(monkeypatch):
    monkeypatch.setattr(p,'read',lambda _:dict(supervisor={'pid':0,'create_time':0}))
    with pytest.raises(ValueError,match='owned worker'):p.worker()


def test_owner_command_binding(monkeypatch):
    monkeypatch.setattr(p,'read',lambda _:dict(command=['wrong']))
    with pytest.raises(ValueError,match='command identity'):p.owner_check()


@pytest.mark.parametrize('busy,used,blocked',[(False,False,False),(True,False,True),(False,True,True)])
def test_admit_busy_and_no_retry(tmp_path,monkeypatch,busy,used,blocked):
    monkeypatch.setattr(p,'OUT',tmp_path)
    monkeypatch.setattr(p,'validate',lambda:None)
    monkeypatch.setattr(p,'sha',lambda _: 'pin')
    monkeypatch.setattr(p,'sources',lambda:{})
    monkeypatch.setattr(p,'read',lambda _:dict(status='PASS',cuda_initialized=False,preregistration_sha256='pin',source_pins={}))
    monkeypatch.setattr(p,'require_review',lambda:None)
    monkeypatch.setattr(p,'review_bindings',lambda:{})
    monkeypatch.setattr(p,'busy_jobs',lambda:[dict(tag='K1S1_STRIDE1_PAIRED_V1',state='running')] if busy else [])
    if used:(tmp_path/'gpu-admission.json').write_text('{}')
    assert (p.admit()['status']=='BLOCKED') is blocked


def test_real_desktop_parser_and_resource_limits(monkeypatch):
    monkeypatch.setattr(safety.subprocess,'check_output',lambda *a,**kw:'123, C:\\Windows\\explorer.exe\n456, C:\\Python312\\python.exe\n789, C:\\Chrome\\chrome.exe\n')
    assert len(safety.active_ml_processes())==1 and not safety.active_ml_processes(456)
    gib=2**30
    assert safety.memory_ok(12*gib,1*gib,16*gib)
    assert not safety.memory_ok(12*gib+1,1*gib,16*gib)
    assert not safety.memory_ok(12*gib,3*gib,16*gib)
    assert safety.stale_reason({'time':0},900)
    assert safety.stale_reason({'time':0},600,dict(util=99,power=80,desktop_bytes=15*gib,total_bytes=16*gib))


SNAP = dict(util=0, desktop_bytes=0, total_bytes=16*2**30, power=100)


class Clock:
    now = 0.
    def monotonic(self): return self.now
    def time(self): return 1800000000.+self.now
    def sleep(self, n): self.now += n


class FakePath:
    def __init__(self, name=''): self.name=name
    def __truediv__(self, name): return FakePath(name)
    def exists(self): return self.name=='gpu-status.json'
    def open(self, *a, **kw): return io.StringIO()


def mock_run(monkeypatch, *, spawn=5379., complete=5405., delay=30.,
             honors_timeout=False, poll_delay=0., cleanup='normal', returncode=0):
    """Reviewer's actual run() harness, with explicit timeout/cleanup variants."""
    c=Clock(); writes=[]; timeouts=[]; waits=[]
    class Child:
        terminated=False; killed=False; polls=0
        def poll(self):
            self.polls+=1
            if self.polls==1: c.now+=poll_delay
            return returncode if self.terminated or c.now>=complete else None
        def terminate(self):
            if cleanup=='terminate_error': raise OSError('terminate failed')
            if cleanup=='normal': self.terminated=True
        def wait(self, timeout):
            waits.append(timeout)
            if not self.terminated:
                c.now+=timeout
                raise subprocess.TimeoutExpired('owned CPU mock',timeout)
            return 0
        def kill(self):
            self.killed=True
            if cleanup=='kill_error': raise OSError('kill failed')
            if cleanup!='both_waits_timeout': self.terminated=True
    child=Child(); child.returncode=returncode
    def popen(*a, **kw): c.now=spawn; return child
    def snapshot(timeout=p.TELEMETRY_SECONDS):
        timeouts.append(timeout)
        if len(timeouts)==2:
            if honors_timeout and delay>=timeout:
                c.now+=timeout
                raise subprocess.TimeoutExpired('nvidia-smi',timeout)
            c.now+=delay
        return SNAP.copy()
    monkeypatch.setattr(p,'OUT',FakePath())
    monkeypatch.setattr(p,'admit',lambda:dict(status='CPU_ADMIT_PASS'))
    monkeypatch.setattr(p,'owner_check',lambda:dict(pid=123,run_id='fake'))
    monkeypatch.setattr(p.jobs,'identity',lambda:dict(pid=456))
    monkeypatch.setattr(p,'sha',lambda _:'mock')
    monkeypatch.setattr(p,'read',lambda _:dict(time=c.time(),done=900,phase='s1'))
    monkeypatch.setattr(p,'write',lambda path,v:writes.append(v))
    monkeypatch.setattr(p.subprocess,'Popen',popen)
    monkeypatch.setattr(p,'gpu_snapshot',snapshot)
    monkeypatch.setattr(p,'active_ml_processes',lambda _:[])
    monkeypatch.setattr(p.time,'time',c.time)
    monkeypatch.setattr(p.time,'monotonic',c.monotonic)
    monkeypatch.setattr(p.time,'sleep',c.sleep)
    return writes,child,timeouts,waits


def test_reviewer_5409_second_false_success_now_failed(monkeypatch):
    writes,child,timeouts,_=mock_run(monkeypatch)
    with pytest.raises(ValueError,match='watchdog'): p.run()
    # Original implementation slept another .05 and reported SUCCESS at 5409.05.
    assert writes[-1]['status']=='FAILED' and writes[-1]['seconds']==5409.
    assert 'budget overrun' in writes[-1]['error']
    assert timeouts==[5.,1.] and not child.terminated


def test_stalled_telemetry_timeout_stops_worker_inside_budget(monkeypatch):
    writes,child,timeouts,_=mock_run(monkeypatch,honors_timeout=True)
    with pytest.raises(ValueError,match='TimeoutExpired'): p.run()
    assert writes[-1]['status']=='FAILED' and writes[-1]['seconds']==5380.
    assert child.terminated and timeouts==[5.,1.]


@pytest.mark.parametrize('spawn,poll_delay',[(5380.,0.),(5409.05,0.),(5379.,30.)])
def test_completed_child_cannot_skip_post_spawn_or_poll_deadline(monkeypatch,spawn,poll_delay):
    writes,_,_,_=mock_run(monkeypatch,spawn=spawn,complete=1.,poll_delay=poll_delay)
    with pytest.raises(ValueError,match='watchdog'): p.run()
    assert writes[-1]['status']=='FAILED'


@pytest.mark.parametrize('cleanup',['normal','grace_timeout','terminate_error','kill_error','both_waits_timeout'])
def test_cleanup_bounded_and_always_writes_failure(monkeypatch,cleanup):
    writes,child,_,waits=mock_run(monkeypatch,spawn=5380.,complete=float('inf'),cleanup=cleanup)
    with pytest.raises(ValueError): p.run()
    assert writes[-1]['status']=='FAILED' and writes[-1]['seconds']<=5400.
    assert len(waits)<=2 and all(0<=t<=10 for t in waits)
    if cleanup!='normal': assert child.killed
    if cleanup in ('kill_error','both_waits_timeout'): assert 'cleanup:' in writes[-1]['error']


@pytest.mark.parametrize('spawn,status',[(0.,'SUCCESS'),(5379.999,'SUCCESS'),(5380.,'FAILED')])
def test_completion_cutoff_boundary(monkeypatch,spawn,status):
    writes,_,_,_=mock_run(monkeypatch,spawn=spawn,complete=0.)
    if status=='FAILED':
        with pytest.raises(ValueError): p.run()
    else: p.run()
    assert writes[-1]['status']==status


def test_worker_nonzero_exit_failed(monkeypatch):
    writes,_,_,_=mock_run(monkeypatch,spawn=1.,complete=0.,returncode=3)
    with pytest.raises(ValueError,match='worker failed'): p.run()
    assert writes[-1]['status']=='FAILED'


@pytest.mark.parametrize('which',['gpu_snapshot','active_ml_processes'])
def test_telemetry_subprocess_timeout_with_real_cpu_child(monkeypatch,which):
    check_output=subprocess.check_output
    observed=[]
    def cpu_only(argv, **kw):
        observed.append(kw['timeout'])
        return check_output([sys.executable,'-c','import time; time.sleep(30)'],**kw)
    monkeypatch.setattr(p.subprocess,'check_output',cpu_only)
    with pytest.raises(subprocess.TimeoutExpired): getattr(p,which)(timeout=.05)
    assert observed==[.05]


def test_local_telemetry_parser_preserves_desktop_filter(monkeypatch):
    calls=[]
    def output(argv, **kw):
        calls.append(kw['timeout'])
        if '--query-compute-apps=pid,process_name' in argv:
            return '123, explorer.exe\n456, python.exe\n789, chrome.exe\n'
        return '0, NVIDIA GeForce RTX 4070 Ti SUPER, 16376, 1000, 0, 45\n'
    monkeypatch.setattr(p.subprocess,'check_output',output)
    assert p.gpu_snapshot()['total_bytes']==16376*2**20
    assert len(p.active_ml_processes())==1 and not p.active_ml_processes(456)
    assert calls==[5.,5.,5.]


@pytest.mark.parametrize('which',['gpu_snapshot','active_ml_processes'])
@pytest.mark.parametrize('timeout',[0.,-1.,float('inf'),float('nan'),6.])
def test_unbounded_telemetry_timeout_rejected(which,timeout):
    with pytest.raises(ValueError,match='bounded'): getattr(p,which)(timeout=timeout)


def test_real_cpu_worker_is_terminated_and_failure_persisted(tmp_path,monkeypatch):
    popen=subprocess.Popen; children=[]
    def cpu_only(*args,**kw):
        child=popen([sys.executable,'-c','import time; time.sleep(30)'])
        children.append(child)
        return child
    monkeypatch.setattr(p,'OUT',tmp_path)
    monkeypatch.setattr(p,'admit',lambda:dict(status='CPU_ADMIT_PASS'))
    monkeypatch.setattr(p,'owner_check',lambda:dict(pid=123,run_id='real-cpu-only'))
    monkeypatch.setattr(p,'sha',lambda _:'mock')
    monkeypatch.setattr(p,'gpu_snapshot',lambda **kw:SNAP.copy())
    monkeypatch.setattr(p,'active_ml_processes',lambda _:[])
    monkeypatch.setattr(p.subprocess,'Popen',cpu_only)
    monkeypatch.setattr(p,'STOP_SECONDS',.1)
    monkeypatch.setattr(p,'BUDGET_SECONDS',2.)
    with pytest.raises(ValueError,match='watchdog'): p.run()
    terminal=json.loads((tmp_path/'gpu-terminal.json').read_text())
    assert terminal['status']=='FAILED' and terminal['seconds']<2.
    assert len(children)==1 and children[0].poll() is not None


def test_repair_registration_and_original_science_are_bound():
    assert p.validate()['status']=='PASS'
    assert p.sha(p.OUT/'PREREGISTRATION.json')==p.PREREG_SHA256


@pytest.mark.parametrize('fault',['pin','extra','science','version'])
def test_repair_manifest_drift_rejected(monkeypatch,fault):
    read=p.read
    def changed(path):
        value=read(path)
        if path.name=='EXECUTION-REPAIR.json':
            if fault=='pin':value['sources']['tools/k1_pooling_v1.py']['after']='bad'
            if fault=='extra':value['sources']['unregistered.py']={'before':'a','after':'b'}
            if fault=='version':value['version']='1.0.0'
        return value
    monkeypatch.setattr(p,'read',changed)
    if fault=='science':monkeypatch.setitem(p.RULE,'mean_macro_delta_floor',.001)
    with pytest.raises(ValueError,match='drift|identity'): p.validate()
