"""Dispatcher v1.2.0: UTF-8/status compatibility and audited dead-tree release."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import psutil

from tools import dispatch_runtime as dr
from tools import job_runtime as rt

ROOT = Path(__file__).resolve().parents[1]
FAKE = r'''
import csv,hashlib,json,os,sys,time,subprocess,psutil
from pathlib import Path
args=sys.argv[1:]
if '-o' not in args: raise ValueError(repr(args))
out=Path(args[args.index('-o')+1])
(out.parent/'fake-args.json').write_text(json.dumps(args))
print(json.dumps({'type':'thread.started','thread_id':'11111111-2222-3333-4444-555555555555'}),flush=True)
mode=os.environ.get('FAKE_MODE','normal')
text='fake completion\r\n'
if os.environ.get('FAKE_DATA'):
    source=Path(os.environ['FAKE_DATA'])
    with source.open(newline='') as f: rows=list(csv.reader(f))
    assert len(rows)==889 and len(rows[0])==13
    text=json.dumps({'rows':len(rows)-1,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
if mode=='early': out.write_bytes(text.encode())
(out.parent/'fake-identity.json').write_text(json.dumps({'pid':os.getpid(),'create_time':psutil.Process().create_time()}))
if os.environ.get('FAKE_ORPHAN'):
    # Middle exits immediately, so the surviving leaf cannot be recovered by
    # traversing only live parent PIDs. Windows job membership must catch it.
    leaf="import json,os,psutil,time; from pathlib import Path; Path(os.environ['FAKE_ORPHAN']).write_text(json.dumps({'pid':os.getpid(),'create_time':psutil.Process().create_time()})); gate=Path(os.environ['FAKE_GATE']); exec('while not gate.exists(): time.sleep(.05)')"
    middle='import subprocess,sys; subprocess.Popen([sys.executable,"-B","-c",'+repr(leaf)+'])'
    subprocess.run([sys.executable,'-B','-c',middle],check=True)
(out.parent/'fake-started').write_text('started')
gate=os.environ.get('FAKE_GATE')
while gate and not Path(gate).exists(): time.sleep(.05)
if mode!='no_report': out.write_bytes(text.encode())
raise SystemExit(7 if mode=='fail' else 0)
'''


@pytest.fixture
def sandbox(tmp_path):
    root=tmp_path/'workspace with spaces'; root.mkdir()
    bin_dir=root/'bin'; bin_dir.mkdir()
    fake=bin_dir/'fake.py'; fake.write_text(FAKE,encoding='utf-8')
    wrapper=bin_dir/'codex'; wrapper.write_text('#!/usr/bin/env bash\nexec "$FAKE_PY" -B "$FAKE_SCRIPT" "$@"\n',encoding='utf-8',newline='\n')
    wrapper.chmod(0o755)
    (root/'prompt.txt').write_text('Engineering CPU fake dispatcher fixture.\n',encoding='utf-8')
    env=dict(os.environ,KNEE_WORKSPACE_ROOT=str(root),KNEE_BASH=dr.bash(),FAKE_PY=sys.executable,
             FAKE_SCRIPT=str(fake),PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('FORCE_MODEL','SIMPLE','EFFORT','FAKE_MODE','FAKE_GATE','FAKE_DATA','FAKE_ORPHAN',
                'DISPATCH_NO_OVERFLOW','DISPATCH_MAX_OVERFLOW','DISPATCH_KEEP_OVERFLOW'): env.pop(key,None)
    return root,env


def cli(box,*args,wait=False):
    root,env=box
    name='wait-out.sh' if wait else 'dispatch.sh'
    return subprocess.run([dr.bash(),str(ROOT/'.local/astra-improve'/name),*map(str,args)],cwd=ROOT,
                          env=env,capture_output=True,text=True,encoding='utf-8')


def terminal(box,lane='sol-2'):
    runtime=dr.Lanes(box[0])
    while True:
        state=rt.read_json(runtime.file(lane,'state.json'))
        if state['phase']=='terminal': return state,rt.read_json(Path(state['run'])/'receipt.json')
        if state.get('worker') and rt.liveness(state['worker'])=='dead':
            state=rt.read_json(runtime.file(lane,'state.json'))
            if state['phase']=='terminal': return state,rt.read_json(Path(state['run'])/'receipt.json')
            pytest.fail('worker died before terminal publication: '+(Path(state['run'])/'worker.log').read_text())
        time.sleep(.05)


def test_success_cli_lock_pid_and_waiter_deferred_until_terminal(sandbox):
    root,env=sandbox; env.update(FAKE_MODE='early',FAKE_GATE=str(root/'release'))
    result=cli(sandbox,'sol-2','prompt.txt','report.md','nosubagents')
    assert result.returncode==0,result.stdout+result.stderr
    assert 'NOTE: sub-agents hard-disabled for this job' in result.stdout
    assert 'dispatched report.md on lane astra-b2 [DATA/TRAINING implementer' in result.stdout
    assert 'DEPRECATED lane sol-2; use astra-b2' in result.stderr
    runtime=dr.Lanes(root); state=rt.read_json(runtime.file('sol-2','state.json'))
    while not (Path(state['run'])/'fake-started').exists():
        current=rt.read_json(Path(state['run'])/'job.json')
        assert current['phase']!='terminal', (Path(state['run'])/'run.log').read_text()
        time.sleep(.05)
    pid=runtime.read('sol-2','lock')
    assert pid==str(state['bash_pid'])
    assert subprocess.run([dr.bash(),'-c','kill -0 "$1"','probe',pid]).returncode==0
    assert not (root/'report.md').exists()
    waiter=subprocess.Popen([dr.bash(),str(ROOT/'.local/astra-improve/wait-out.sh'),'report.md','sol-2'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert waiter.poll() is None
    assert cli(sandbox,'close','sol-2').returncode==3
    (root/'release').write_text('go')
    out,err=waiter.communicate()
    assert waiter.returncode==0,(out,err)
    assert out.startswith(b'DONE ')
    state,receipt=terminal(sandbox)
    assert receipt['state']=='SUCCESS' and runtime.read('sol-2','n')=='1'
    assert not runtime.file('sol-2','lock').exists()
    args=rt.read_json(Path(state['run'])/'fake-args.json')
    assert '--disable' in args and 'multi_agent' in args
    assert cli(sandbox,'lanes').returncode==0


@pytest.mark.parametrize('mode',['fail','no_report'])
def test_failed_jobs_do_not_increment_or_accept_old_report(sandbox,mode):
    root,env=sandbox
    assert cli(sandbox,'sol-2','prompt.txt','report.md').returncode==0
    terminal(sandbox)
    old=(root/'report.md').read_bytes()
    env['FAKE_MODE']=mode
    assert cli(sandbox,'sol-2','prompt.txt','report.md').returncode==0
    state,receipt=terminal(sandbox)
    assert receipt['state']=='FAILED' and dr.Lanes(root).read('sol-2','n')=='1'
    assert (root/'report.md').read_bytes()==old
    assert (Path(state['run'])/'previous/report.md').read_bytes()==old
    assert cli(sandbox,'report.md','sol-2',wait=True).returncode==1


def test_concurrent_dispatch_single_atomic_winner(sandbox):
    root,env=sandbox; env['FAKE_GATE']=str(root/'release')
    env['DISPATCH_NO_OVERFLOW']='1'
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(cli,sandbox,'sol-2','prompt.txt',f'report-{i}.md') for i in range(2)]
        results=[f.result() for f in futures]
    assert sorted(r.returncode for r in results)==[0,3]
    assert sum('BUSY lane astra-b2' in r.stdout for r in results)==1
    (root/'release').write_text('go'); terminal(sandbox)
    assert dr.Lanes(root).read('sol-2','n')=='1'


def test_same_output_cannot_be_claimed_by_different_lanes(sandbox):
    root,env=sandbox; env['FAKE_GATE']=str(root/'release')
    assert cli(sandbox,'sol-2','prompt.txt','report.md').returncode==0
    assert cli(sandbox,'astra-2','prompt.txt','report.md').returncode==3
    assert cli(sandbox,'astra-2','prompt.txt','report').returncode==3
    (root/'release').write_text('go'); terminal(sandbox)


def test_resume_rotation_and_model_change_preserve_history(sandbox):
    root,env=sandbox
    for i in range(4):
        result=cli(sandbox,'sol-2','prompt.txt','report.md'); assert result.returncode==0
        state,_=terminal(sandbox)
        args=rt.read_json(Path(state['run'])/'fake-args.json')
        assert ('resume' in args)==(i in (1,2))
    assert 'closed after 3 jobs' in result.stdout
    assert dr.Lanes(root).read('sol-2','n')=='1'
    env.update(SIMPLE='1',EFFORT='xhigh')
    result=cli(sandbox,'sol-2','prompt.txt','report.md')
    assert 'was open on another model' in result.stdout
    state,_=terminal(sandbox)
    args=rt.read_json(Path(state['run'])/'fake-args.json')
    assert 'gpt-6-sol' in args and 'model_reasoning_effort=xhigh' in args
    assert cli(sandbox,'reset','sol-2').returncode==0
    assert not dr.Lanes(root).file('sol-2','id').exists()
    assert len(list(dr.Lanes(root).directory.glob('history/*.json')))>=3


@pytest.mark.parametrize('args,lane,model',[
    (['workspace-write','prompt.txt','report.md'],'sol-1','gpt-6-astra'),
    (['read-only','prompt.txt','report.md','sol'],'sol-3','gpt-6-astra'),
    (['read-only','prompt.txt','report.md','astra'],'astra-1','gpt-6-astra'),
    (['extra','prompt.txt','report.md','sol','nosubagents'],'extra','gpt-6-sol'),
])
def test_legacy_and_extra_arguments(sandbox,args,lane,model):
    assert cli(sandbox,*args).returncode==0
    state,_=terminal(sandbox,lane)
    assert state['model']==model
    if lane=='extra':
        assert dr.Lanes(sandbox[0]).read(lane,'model')=='sol'
        assert cli(sandbox,'extra','prompt.txt','other.md','astra').returncode==2


@pytest.mark.parametrize('tamper',['report','receipt','pointer','run_receipt','prompt','log'])
def test_waiter_rejects_changed_evidence(sandbox,tamper):
    root,_=sandbox
    assert cli(sandbox,'sol-2','prompt.txt','report.md').returncode==0
    state,_=terminal(sandbox); run=Path(state['run'])
    target={'report':root/'report.md','receipt':root/'report.md.receipt.json',
            'pointer':root/'report.md.dispatch.json','run_receipt':run/'receipt.json',
            'prompt':run/'prompt.txt','log':run/'run.log'}[tamper]
    target.write_text('tampered',encoding='utf-8')
    assert cli(sandbox,'report.md','sol-2',wait=True).returncode==1


def test_stale_reservation_and_legacy_report_fail_closed(sandbox):
    root,env=sandbox; runtime=dr.Lanes(root)
    env['DISPATCH_NO_OVERFLOW']='1'
    rt.atomic_json(runtime.file('sol-2','state.json'),dict(phase='reserved'))
    assert cli(sandbox,'sol-2','prompt.txt','report.md').returncode==3
    (root/'legacy.md').write_text('not terminal evidence')
    assert cli(sandbox,'legacy.md',wait=True).returncode==1
    assert cli(sandbox,'close','sol-2').returncode==3


def test_real_numeric_data_smoke_through_shell_dispatch_and_wait(sandbox):
    root,env=sandbox
    source=ROOT/'artifacts/experiments/EXP185_FINDING_MIL/outputs/S42/results/EXP185_FINDING_MIL_F0_S42/oof_k36_flip_tta.csv'
    before=dr.sha(source); env['FAKE_DATA']=str(source)
    assert cli(sandbox,'sol-2','prompt.txt','report.md','nosubagents').returncode==0
    terminal(sandbox)
    result=cli(sandbox,'report.md','sol-2',wait=True)
    assert result.returncode==0
    assert rt.read_json(root/'report.md')=={'rows':888,'sha256':before}
    assert dr.sha(source)==before


@pytest.mark.skipif(os.name != 'nt', reason='Windows kernel job-object release')
def test_release_checks_orphaned_descendant_and_records_failure(sandbox):
    root,env=sandbox
    env.update(FAKE_GATE=str(root/'release-gate'),FAKE_ORPHAN=str(root/'orphan.json'))
    runtime=dr.Lanes(root)
    assert cli(sandbox,'sol-2','prompt.txt','report.md').returncode==0
    while not (root/'orphan.json').exists(): time.sleep(.05)
    job=rt.read_json(runtime.file('sol-2','state.json')); run=Path(job['run'])
    before=runtime.file('sol-2','state.json').read_bytes()
    assert cli(sandbox,'release','sol-2').returncode==3
    assert runtime.file('sol-2','state.json').read_bytes()==before
    identities=[job['worker'],job['child'],rt.read_json(run/'fake-identity.json')]
    for identity in identities:
        if rt.liveness(identity)=='alive': psutil.Process(identity['pid']).terminate()
    while any(rt.liveness(identity)!='dead' for identity in identities): time.sleep(.05)
    leaf=rt.read_json(root/'orphan.json')
    assert rt.liveness(leaf)=='alive'
    assert cli(sandbox,'release','sol-2').returncode==3
    assert runtime.file('sol-2','state.json').read_bytes()==before
    (root/'release-gate').write_text('finish orphan')
    while rt.liveness(leaf)!='dead': time.sleep(.05)
    result=cli(sandbox,'release','sol-2')
    assert result.returncode==0,result.stdout+result.stderr
    released=rt.read_json(runtime.file('sol-2','state.json'))
    audit=rt.read_json(run/'release.json')
    assert released['phase']=='terminal' and released['release']['sha256']==dr.sha(run/'release.json')
    assert audit['tree']=='dead' and audit['success'] is False and audit['previous_job']==job
    assert runtime.read('sol-2','n','0')=='0' and not (root/'report.md').exists()
    assert cli(sandbox,'report.md','sol-2',wait=True).returncode==1
    assert cli(sandbox,'release','sol-2').returncode==0
    del env['FAKE_GATE'],env['FAKE_ORPHAN']
    assert cli(sandbox,'sol-2','prompt.txt','report.md').returncode==0
    terminal(sandbox)
    assert runtime.read('sol-2','n')=='1'


def test_release_rejects_missing_or_unknown_process_tree(sandbox,monkeypatch):
    runtime=dr.Lanes(sandbox[0])
    assert cli(sandbox,'release','sol-2').returncode==3
    assert cli(sandbox,'release','sol-2','unexpected').returncode==2
    monkeypatch.setattr(dr,'contained_tree_state',lambda *args:'unknown')
    run_id='f'*32; run=runtime.directory/'runs/sol-2'/run_id
    job=dict(run_id=run_id,lane='sol-2',phase='running',run=str(run),owner=rt.identity(),worker=rt.identity())
    rt.atomic_json(run/'job.json',job);rt.atomic_json(runtime.file('sol-2','state.json'),job)
    assert runtime.release('sol-2')==3
    assert rt.read_json(runtime.file('sol-2','state.json'))==job
    assert not (run/'release.json').exists()


def status_cli(box):
    root, env = box
    return subprocess.run([dr.bash(), str(ROOT/'.local/astra-improve/astra_status.sh'), str(root)],
                          cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')


def test_cp949_lanes_and_status_show_reserved_busy_without_session(sandbox):
    root, env = sandbox
    env.update(PYTHONUTF8='0', PYTHONIOENCODING='cp949')
    runtime = dr.Lanes(root)
    rt.atomic_json(runtime.file('sol-2','state.json'), dict(phase='reserved'))
    result = cli(sandbox,'lanes')
    assert result.returncode == 0, result.stderr
    assert 'BUSY pid reserved' in result.stdout and 'temporary — close' in result.stdout
    result = status_cli(sandbox)
    assert result.returncode == 0, result.stderr
    assert 'astra-b2: BUSY pid reserved' in result.stdout


@pytest.mark.parametrize('mode,expected', [('normal','DONE'), ('fail','ERROR terminal FAILED')])
def test_status_uses_terminal_receipts_and_real_predictions(sandbox, mode, expected):
    root, env = sandbox
    env.update(FAKE_MODE=mode, PYTHONUTF8='0', PYTHONIOENCODING='cp949',
        FAKE_DATA=str(ROOT/'artifacts/experiments/EXP185_FINDING_MIL/outputs/S42/results/EXP185_FINDING_MIL_F0_S42/oof_k36_flip_tta.csv'))
    assert cli(sandbox,'sol-2','prompt.txt','report.md').returncode == 0
    terminal(sandbox)
    result = status_cli(sandbox)
    assert result.returncode == 0, result.stderr
    assert 'report: '+expected in result.stdout and 'astra-b2: '+expected in result.stdout
    if mode == 'normal':
        (root/'report.md').write_text('changed')
        assert 'ERROR report hash differs' in status_cli(sandbox).stdout


def test_status_preserves_legacy_error_token_and_no_output_lines(sandbox):
    root, _ = sandbox
    (root/'old.log').write_text('{"type":"turn.failed","message":"example failure","input_tokens":2000000}\n')
    (root/'ended.log').write_text('{"type":"turn.completed"}\n')
    result = status_cli(sandbox)
    assert result.returncode == 0
    assert 'old: ERROR example failure in_tokens=2.00M' in result.stdout
    assert 'ended: ENDED-NO-OUTPUT' in result.stdout
