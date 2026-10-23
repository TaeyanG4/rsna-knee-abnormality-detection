"""v1.0.2: dead legacy hand markers unblock launches; modern validation unchanged."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading

import pytest
from tools import job_runtime as rt
from tools import gpu_job

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != 'nt', reason='Windows sharing semantics')
def test_atomic_publication_waits_for_open_metadata_reader(tmp_path):
    target = tmp_path/'state.json'
    rt.atomic_json(target, {'phase':'running'})
    reader = target.open('rb')
    assert rt.sharing_locked(target)
    entered = threading.Event()
    errors = []
    def publish():
        entered.set()
        try: rt.atomic_json(target, {'phase':'terminal'})
        except Exception as exc: errors.append(exc)
    worker = threading.Thread(target=publish)
    worker.start(); entered.wait()
    assert rt.read_json(target)['phase']=='running'
    reader.close(); worker.join()
    assert not errors and rt.read_json(target)['phase']=='terminal'


def test_atomic_publication_does_not_retry_real_permission_denial(tmp_path, monkeypatch):
    def denied(*args): raise PermissionError('denied')
    monkeypatch.setattr(rt.os, 'replace', denied)
    monkeypatch.setattr(rt, 'sharing_locked', lambda path: False)
    with pytest.raises(PermissionError): rt.atomic_json(tmp_path/'state.json', {})


@pytest.mark.skipif(os.name != 'nt', reason='Windows replacement semantics')
def test_atomic_publication_retries_reader_released_before_probe(tmp_path, monkeypatch):
    target=tmp_path/'state.json'
    rt.atomic_json(target, {'phase':'running'})
    replace=os.replace
    attempts=[]
    def transient(source,destination):
        attempts.append(1)
        if len(attempts)==1:
            error=PermissionError('reader closed before probe')
            error.winerror=5
            raise error
        return replace(source,destination)
    monkeypatch.setattr(rt.os,'replace',transient)
    assert not rt.sharing_locked(target)
    rt.atomic_json(target, {'phase':'terminal'})
    assert len(attempts)==2 and rt.read_json(target)['phase']=='terminal'


@pytest.mark.skipif(os.name != 'nt', reason='Windows readonly semantics')
def test_atomic_publication_refuses_readonly_destination(tmp_path):
    import stat
    source=tmp_path/'new';source.write_text('new')
    target=tmp_path/'old';target.write_text('old')
    target.chmod(stat.S_IREAD)
    try:
        assert not rt.replacement_permitted(source,target)
        with pytest.raises(PermissionError): rt.replace_file(source,target)
        assert target.read_text()=='old'
    finally:
        target.chmod(stat.S_IWRITE)


def write_job(tmp_path, *, modern=False, **kw):
    job = dict(tag="TEST", command=["python", "unused"], started="2026-10-12T00:00:00",
               pid=2147483647, cwd=str(tmp_path))
    if modern:
        job.update(schema="gpu-job/v2", create_time=1, run_id="fixture", phase="running", child_identity=None)
        job['command_sha256'] = rt.command_digest(job['command'], tmp_path)
    job.update(kw)
    rt.atomic_json(tmp_path/'TEST/job.json', job)
    return job


def test_pid_reuse_and_missing_identity_fail_closed():
    me = rt.identity()
    assert rt.liveness(me) == 'alive'
    assert rt.liveness(dict(me,create_time=me['create_time']-10)) == 'dead'
    assert rt.liveness({'pid':me['pid']}) == 'unknown'
    assert rt.liveness({'pid':2147483647,'create_time':1}) == 'dead'


@pytest.mark.parametrize('phase,expected',[('running','stale'),('spawning','unknown'),('dispatching','unknown')])
def test_crash_windows(tmp_path,phase,expected):
    write_job(tmp_path, modern=True, phase=phase)
    assert rt.job_status(tmp_path,'TEST')['state'] == expected


def test_orphan_child_and_legacy_live(tmp_path):
    write_job(tmp_path,modern=True,child_identity=rt.identity())
    assert rt.job_status(tmp_path,'TEST')['state'] == 'orphaned'
    write_job(tmp_path,pid=os.getpid())
    assert rt.job_status(tmp_path,'TEST')['state'] == 'running'
    assert rt.active_jobs(tmp_path)


def test_markers_receipts_and_command_binding(tmp_path):
    write_job(tmp_path)
    rt.atomic_text(tmp_path/'TEST/SUCCESS','exit 0 at 2026-10-12T00:00:00')
    assert rt.job_status(tmp_path,'TEST')['receipt_kind'] == 'legacy-marker'
    assert rt.job_status(tmp_path,'TEST',expected_command=['other'])['state'] == 'unknown'
    job = write_job(tmp_path,modern=True,started_epoch=time.time())
    assert rt.job_status(tmp_path,'TEST')['state'] == 'unknown'
    gpu_job.finish(tmp_path/'TEST',job,0)
    assert rt.job_status(tmp_path,'TEST')['state'] == 'SUCCESS'
    job['child_identity'] = rt.identity()
    rt.atomic_json(tmp_path/'TEST/job.json',job)
    gpu_job.finish(tmp_path/'TEST',job,0)
    assert rt.job_status(tmp_path,'TEST')['state'] == 'unknown'


@pytest.mark.parametrize('markers,expected', [
    ({'FAILED':'stopped by B'}, 'FAILED'),
    ({'SUCCESS':'finished manually'}, 'SUCCESS'),
    ({'SUCCESS':'old success', 'FAILED':'stopped by B'}, 'FAILED'),
])
def test_dead_legacy_hand_markers_are_terminal(tmp_path, markers, expected):
    write_job(tmp_path)
    for name, content in markers.items():
        (tmp_path/'TEST'/name).write_text(content, encoding='utf-8')
    assert rt.job_status(tmp_path,'TEST')['state'] == expected
    assert not rt.active_jobs(tmp_path)
    write_job(tmp_path, modern=True)
    assert rt.job_status(tmp_path,'TEST')['state'] == 'unknown'
    assert rt.active_jobs(tmp_path)


@pytest.mark.parametrize('name',['../bad','x/y','x\\y','',None])
def test_path_escape_rejected(name):
    with pytest.raises(ValueError): rt.tag_name(name)


def test_list_is_metadata_only_and_tail_bounded(tmp_path,monkeypatch,capsys):
    write_job(tmp_path)
    log=tmp_path/'TEST/run.log'; log.write_bytes(b'x'*200000+b'\nlast\n')
    monkeypatch.setattr(gpu_job,'LOGS',tmp_path)
    def no_tail(*a,**k): pytest.fail('list read a log')
    monkeypatch.setattr(rt,'bounded_tail',no_tail)
    assert gpu_job.main(['--list']) == 0
    assert 'stale' in capsys.readouterr().out
    monkeypatch.undo()
    assert rt.bounded_tail(log,limit=16)[-1] == 'last'


@pytest.fixture
def cpu(tmp_path):
    (tmp_path/'config').mkdir(); (tmp_path/'config/workspace.toml').write_text('')
    env=dict(os.environ,KNEE_WORKSPACE_ROOT=str(tmp_path),CUDA_VISIBLE_DEVICES='',
             PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def command(tag,*args,wait=True):
        return [sys.executable,'-B',str(ROOT/'tools/gpu_job.py'),'--tag',tag,*(['--wait'] if wait else []),'--',*args]
    def run(tag,*args,wait=True):
        return subprocess.run(command(tag,*args,wait=wait),env=env,capture_output=True,text=True,encoding='utf-8')
    return tmp_path,env,command,run


def test_real_cpu_marker_cli_rerun_and_literal_delimiter(cpu):
    root,env,cmd,run=cpu
    result=run('CPU',sys.executable,'-c','import sys; print(sys.argv[1:])','--','retained')
    assert result.returncode == 0,result.stderr+result.stdout
    folder=root/'artifacts/logs/CPU'
    assert (folder/'run.log').read_text().strip() == "['--', 'retained']"
    assert re.fullmatch(r'exit 0 at \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d',(folder/'SUCCESS').read_text())
    assert rt.job_status(root/'artifacts/logs','CPU')['receipt_kind']=='bound'
    result=run('CPU',sys.executable,'-c','raise SystemExit(7)')
    assert result.returncode == 7
    assert rt.job_status(root/'artifacts/logs','CPU')['state']=='FAILED'
    assert len(list((folder/'attempts').glob('*/SUCCESS'))) == 1
    result=run('BAD',str(root/'nonexistent-executable'))
    assert result.returncode != 0
    assert rt.job_status(root/'artifacts/logs','BAD')['state']=='FAILED'


def test_real_concurrent_claim_and_wait(cpu):
    root,env,cmd,run=cpu
    script=root/'child.py'
    script.write_text("from pathlib import Path\nimport time\nwhile not Path('release').exists(): time.sleep(.02)\n")
    processes=[subprocess.Popen(cmd(tag,sys.executable,str(script)),env=env,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE) for tag in ('ONE','TWO')]
    try:
        # One launcher owns the reservation; the competing launcher must terminate.
        while all(p.poll() is None for p in processes): time.sleep(.02)
        exited=[p for p in processes if p.poll() is not None]
        assert len(exited)==1 and exited[0].returncode==2
        statuses=[rt.job_status(root/'artifacts/logs',tag) for tag in ('ONE','TWO')]
        assert sum(s['state']=='running' for s in statuses)==1
    finally:
        (root/'release').write_text('go')
        for p in processes: p.communicate()
    assert sorted(p.returncode for p in processes)==[0,2]


def test_status_import_is_lightweight():
    cp=subprocess.run([sys.executable,'-B','-c',
        "from tools import gpu_job, gpu_queue; import sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
        cwd=ROOT,capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
