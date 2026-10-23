"""H-B-156 L2 v3.0.0: seed43 runtime gates and isolated transfer tests."""
import copy
import json
from pathlib import Path
import pytest
from tools import hb156_vast_v3 as b
from tools.hb156_package_v3 import PACKAGE


@pytest.fixture
def package(monkeypatch):
    monkeypatch.setattr(b,'ROOT',PACKAGE)
    monkeypatch.setattr(b,'STAGE',PACKAGE/'stage-convnext-l2-s43-v1')
    monkeypatch.setattr(b,'PROTOCOL',b.STAGE/'manifest.json')
    monkeypatch.setattr(b,'INIT',PACKAGE/'weights/convnext-small384-imagenet.pt')
    return b.load_registration(b.PROTOCOL)


def test_registered_recipe_and_trainer(package):
    b.verify_trainer()
    assert package['recipe']['seed_local']==43
    assert package['clean_oof'] is False and package['exposure']=='EXPOSED'
    assert len(package['train_ids'])==4347
    assert len(package['cache_files'])==8812
    s=b.execution_settings(package,1,'bf16',43,False)
    assert s['batch']==1 and s['accumulation']==16 and s['total_steps']==4352


@pytest.mark.parametrize('args',[(2,'bf16',43,False),(1,'fp16',43,False),(1,'bf16',44,False),(1,'bf16',43,True)])
def test_execution_mutation(package,args):
    with pytest.raises(ValueError): b.execution_settings(package,*args)


def test_recipe_and_cache_mutation(package,monkeypatch,tmp_path):
    mutated=copy.deepcopy(package);mutated['recipe']['batch_local']=2
    with pytest.raises(ValueError): b.execution_settings(mutated,1,'bf16',43,False)
    monkeypatch.setattr(b,'CORPUS',tmp_path)
    (tmp_path/'cache.npy').write_bytes(b'abc')
    p={'cache_files':{'cache.npy':3}}
    assert b.verify_cache(p)==dict(files=1,bytes=3)
    (tmp_path/'cache.npy').write_bytes(b'ab')
    with pytest.raises(ValueError,match='upload incomplete'): b.verify_cache(p)
    with pytest.raises(ValueError): b.verify_cache({'cache_files':{'missing':2}})


def smoke():
    return dict(schema='hb151-smoke/v3',protocol_sha256=b.sha256(b.PROTOCOL),
        device_name='NVIDIA GeForce RTX 4090',total_memory_mib=24564,
        desktop_before_cuda=dict(used_mib=100,total_mib=24564),
        shape=dict(batch=1,accumulation=16,k=12,res=384,bn_batch=2,precision='bf16',grad_checkpointing=False),
        measurements={p:dict(peak_reserved_bytes=11*1024**3,seconds_per_unit=2.) for p in ('train','bn')},
        elapsed_seconds=100,train_updates=2,bn_batches=2)


@pytest.mark.parametrize('fault',['device','shape','time','memory','partial','stale','nan'])
def test_smoke_fail_closed(package,fault):
    s=smoke();b.check_smoke(s)
    if fault=='device': s['device_name']='RTX 3090'
    if fault=='shape': s['shape']['batch']=2
    if fault=='time': s['elapsed_seconds']=176
    if fault=='memory': s['measurements']['bn']['peak_reserved_bytes']=13*1024**3
    if fault=='partial': s['train_updates']=1
    if fault=='stale': s['protocol_sha256']='stale'
    if fault=='nan': s['measurements']['train']['seconds_per_unit']=float('nan')
    with pytest.raises(ValueError): b.check_smoke(s)


def review_fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'PROTOCOL',tmp_path/'manifest.json')
    monkeypatch.setattr(b,'load_registration',lambda *a,**k:{})
    monkeypatch.setattr(b,'require_validation',lambda *a:None)
    monkeypatch.setattr(b,'runtime_snapshot',lambda:{'gpu':'RTX 4090'})
    b.atomic_json(b.PROTOCOL,{'test':True})
    b.atomic_json(tmp_path/'runtime-effective.json',b.runtime_snapshot())
    s=smoke();s['runtime_sha256']=b.sha256(tmp_path/'runtime-effective.json')
    b.atomic_json(tmp_path/'cuda-smoke.json',s)
    b.atomic_json(tmp_path/'smoke-attempt/receipt.json',dict(exit_code=0,reason='completed',survivors=[]))
    b.atomic_json(tmp_path/'cpu-validation.json',{'status':'PASS_CPU'})
    b.atomic_json(tmp_path/'runtime-diff.json',{'differences':'reviewed'})
    review=dict(verdict='PASS',reviewer='independent-fixture',independent=True,
        host_differences_accepted=True,probability_semantics='native-probability-space accepted by owner')
    for key,path in {'protocol':b.PROTOCOL,'smoke':tmp_path/'cuda-smoke.json',
        'smoke_receipt':tmp_path/'smoke-attempt/receipt.json','cpu_validation':tmp_path/'cpu-validation.json',
        'runtime':tmp_path/'runtime-effective.json','runtime_diff':tmp_path/'runtime-diff.json'}.items():
        review[key+'_sha256']=b.sha256(path)
    return review


@pytest.mark.parametrize('fault',['none','missing','stale','builder','host','semantics','runtime','receipt'])
def test_independent_review_bound(monkeypatch,tmp_path,fault):
    review=review_fixture(monkeypatch,tmp_path)
    if fault=='stale': review['smoke_sha256']='stale'
    if fault=='builder': review['reviewer']='astra-b5'
    if fault=='host': review['host_differences_accepted']=False
    if fault=='semantics': review.pop('probability_semantics')
    if fault=='runtime': monkeypatch.setattr(b,'runtime_snapshot',lambda:{'gpu':'changed'})
    if fault=='receipt': b.atomic_json(tmp_path/'smoke-attempt/receipt.json',dict(exit_code=124,reason='total_cap',survivors=[]))
    if fault!='missing': b.atomic_json(tmp_path/'independent-review.json',review)
    if fault=='none': assert b.require_fit()['train_updates']==2
    else:
        with pytest.raises((ValueError,FileNotFoundError)): b.require_fit()


def test_direct_child_bypass(monkeypatch,tmp_path):
    monkeypatch.setattr(b.sys,'platform','linux')
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'admission_snapshot',lambda:'')
    b.atomic_json(tmp_path/'supervisor.json',dict(mode='fit',pid=-1,created=0))
    with pytest.raises(ValueError,match='ancestry'): b.require_gpu_job('fit',child=True)
    monkeypatch.setattr(b,'admission_snapshot',lambda:'99999999, python3\n')
    with pytest.raises(ValueError,match='another Python'): b.require_gpu_job('smoke')


def test_budget_and_watchdog_parameters(monkeypatch,tmp_path):
    from tools import hb150_watchdog_v1 as wd
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'RUN',tmp_path/'S43')
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**k:None)
    monkeypatch.setattr(b,'load_registration',lambda *a,**k:{})
    monkeypatch.setattr(b,'require_validation',lambda *a:None)
    monkeypatch.setattr(b,'require_fit',lambda:dict(measurements={'train':{'seconds_per_unit':4},'bn':{'seconds_per_unit':2}}))
    calls=[]
    monkeypatch.setattr(wd,'supervise',lambda cmd,attempt,**kw:calls.append((cmd,attempt,kw)) or 124)
    assert b.supervise('smoke')==124 and calls[-1][2]['total']==165
    assert b.supervise('fit')==124
    assert calls[-1][2]['healthy_rates']==dict(train_step=.25,bn_replay=.5)
    assert b.read_json(tmp_path/'budget.json')['running'] is False
    b.atomic_json(tmp_path/'budget.json',dict(attempt=1,running=True,charged_seconds=0))
    with pytest.raises(ValueError,match='reconcile'): b.supervise('fit',True)


def test_portable_export_recovery(monkeypatch,tmp_path):
    import torch
    monkeypatch.setattr(b,'RUN',tmp_path/'S43')
    monkeypatch.setattr(b,'EXPORT',tmp_path/'export/exp143-v2-e14-e11-e12-bn.pt')
    monkeypatch.setattr(b,'PROTOCOL',tmp_path/'manifest.json')
    monkeypatch.setattr(b,'load_registration',lambda *a,**k:{})
    b.atomic_json(b.PROTOCOL,{'seed':43})
    source=b.RUN/'model_best.pt'
    b.save_torch_atomic(source,{'model':{'weight':torch.tensor([1.])},'seed':43})
    b.atomic_json(b.RUN/'summary.json',dict(status='COMPLETE_TRAINING_NO_EVALUATION',model_sha256=b.sha256(source)))
    replace=b.safe_replace
    monkeypatch.setattr(b,'safe_replace',lambda *a,**k:False)
    assert b.export_checkpoint(source)['status']=='EXPORT_PENDING'
    monkeypatch.setattr(b,'safe_replace',replace)
    assert b.main(['export'])==0
    assert b.sha256(b.EXPORT)==b.sha256(source)
    assert torch.load(b.EXPORT,weights_only=True)['seed']==43
    assert b.read_json(b.EXPORT.with_suffix('.json'))['protocol_sha256']==b.sha256(b.PROTOCOL)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint drift'): b.recover_export()


@pytest.mark.parametrize('mode,extra,valid',[('smoke',[],True),('fit',[],True),('fit',['--resume'],True),
                                         ('smoke',['--resume'],False),('bad',[],False)])
def test_bash_launcher_mocked(tmp_path,mode,extra,valid):
    import os,shutil,subprocess
    bash=shutil.which('bash') if os.name!='nt' else 'C:/Program Files/Git/usr/bin/bash.exe'
    script=Path(__file__).resolve().parents[1]/'tools/hb156_run_v3.sh'
    shutil.copyfile(script,tmp_path/'run.sh')
    (tmp_path/'stage-convnext-l2-s43-v1').mkdir()
    bindir=tmp_path/'bin';bindir.mkdir()
    (bindir/'python').write_text('#!/usr/bin/env bash\nprintf "MOCK_PY %s\\n" "$*"\n',newline='\n')
    (bindir/'nohup').write_text('#!/usr/bin/env bash\nprintf "MOCK_NOHUP\\n"\nexec "$@"\n',newline='\n')
    for f in bindir.iterdir(): f.chmod(0o755)
    # wait joins only our immediate mocked child; no sleeps or real GPU/process launches.
    wrapper='export PATH="$PWD/bin:$PATH"; source ./run.sh "$@"; wait'
    result=subprocess.run([bash,'-c',wrapper,'fixture',mode,*extra],cwd=tmp_path,text=True,capture_output=True,timeout=10)
    assert (result.returncode==0)==valid,result.stderr
    if valid:
        assert 'MOCK_PY -B -m tools.hb156_vast_v1 verify' in result.stdout
        assert 'STARTED supervisor_pid=' in result.stdout
        logs=list((tmp_path/'stage-convnext-l2-s43-v1').glob('launch-*.log'))
        assert len(logs)==1
        assert 'MOCK_NOHUP' in logs[0].read_text()
        assert f'tools.hb156_vast_v1 {mode}' in logs[0].read_text()
        if extra: assert '--resume' in logs[0].read_text()


@pytest.mark.parametrize('action,fault',[('Copy','none'),('Copy','hash'),('Copy','transport'),('Copy','cache'),
                                        ('Copy','upload_exists'),('Copy','destination_exists'),
                                        ('Pull','none'),('Pull','hash'),('Pull','provenance')])
def test_transfer_offline(tmp_path,action,fault):
    import hashlib,shutil,subprocess
    shell=shutil.which('powershell') or shutil.which('pwsh')
    assert shell,'PowerShell needed to qualify the transfer card'
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'tools').mkdir()
    shutil.copyfile(root/'tools/hb156_transfer_v3.ps1',tmp_path/'tools/transfer.ps1')
    work=tmp_path/'.local/scratch/hb156-l2-s43/H-B-156-L2-S43';work.mkdir(parents=True)
    manifest=work/'package/stage-convnext-l2-s43-v1/manifest.json';manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b'{"seed":43}')
    payload=b'offline transfer fixture'
    digest=hashlib.sha256(payload).hexdigest()
    (work/'hb156-convnext-l2-s43-v1.tar').write_bytes(payload)
    pin='0'*64 if fault=='hash' else digest
    (work/'hb156-convnext-l2-s43-v1.tar.sha256').write_text(pin+'  hb156-convnext-l2-s43-v1.tar\n')
    fixture=tmp_path/'fixture';fixture.mkdir()
    stem='exp143-v2-e14-e11-e12-bn'
    (fixture/(stem+'.pt')).write_bytes(payload)
    (fixture/(stem+'.sha256')).write_text(pin+'  '+stem+'.pt\n')
    (fixture/(stem+'.json')).write_text(json.dumps(dict(sha256=digest,clean_oof=False,exposure='EXPOSED',
        protocol_sha256='stale' if fault=='provenance' else hashlib.sha256(manifest.read_bytes()).hexdigest())))
    harness=tmp_path/'harness.ps1'
    harness.write_text('''$ErrorActionPreference = 'Stop'
function scp {
    Add-Content -LiteralPath "$PSScriptRoot/calls.txt" -Value "scp $args"
    if ($env:TRANSFER_FAULT -eq 'transport') { $global:LASTEXITCODE=1; return }
    if ($env:TRANSFER_ACTION -eq 'Pull') {
        $name=($args[-2] -split '/')[-1]
        Copy-Item -LiteralPath "$PSScriptRoot/fixture/$name" -Destination $args[-1]
    }
    $global:LASTEXITCODE=0
}
function ssh {
    if ($env:TRANSFER_FAULT -eq 'cache') { $global:LASTEXITCODE=1; return }
    Add-Content -LiteralPath "$PSScriptRoot/calls.txt" -Value "ssh $args"
    if ($env:TRANSFER_FAULT -eq 'upload_exists' -and $args[-1] -eq 'mkdir /workspace/hb156-l2-s43-upload-v1') { $global:LASTEXITCODE=1; return }
    if ($env:TRANSFER_FAULT -eq 'destination_exists' -and $args[-1] -like 'test ! -e /workspace/hb156-convnext-l2-s43-v1*') { $global:LASTEXITCODE=1; return }
    $global:LASTEXITCODE=0
}
& "$PSScriptRoot/tools/transfer.ps1" $env:TRANSFER_ACTION
''')
    import os
    result=subprocess.run([shell,'-NoProfile','-NonInteractive','-File',str(harness)],
        env=dict(os.environ,TRANSFER_ACTION=action,TRANSFER_FAULT=fault),text=True,capture_output=True,timeout=20)
    assert (result.returncode==0)==(fault=='none'),result.stdout+result.stderr
    if fault in ('upload_exists','destination_exists'):
        assert 'scp ' not in (tmp_path/'calls.txt').read_text()
    if fault=='none':
        calls=(tmp_path/'calls.txt').read_text()
        assert 'scp <remote-host>' in calls
        if action=='Copy':
            assert 'sha256sum -c SHA256SUMS' in calls
            assert 'build-registration.json' in calls and 'manifest.json' in calls
            assert 'scp ' not in calls.split('scp ')[0]
            assert '/workspace/hb151-seed44-v1' not in calls
            assert calls.index('ssh ') < calls.index('scp ')
            assert 'root@<remote-host>:/workspace/hb156-l2-s43-upload-v1/' in calls
            assert 'root@<remote-host>:/workspace/\n' not in calls
            assert calls.index('mkdir /workspace/hb156-l2-s43-upload-v1') < calls.index('scp ')
            assert 'cd /workspace/hb156-l2-s43-upload-v1 && sha256sum -c' in calls
            assert 'mkdir /workspace/hb156-convnext-l2-s43-v1 && tar -xf' in calls
            assert '-C /workspace/hb156-convnext-l2-s43-v1 --strip-components=1' in calls
        else: assert 'VERIFIED '+digest in result.stdout and 'ssh ' not in calls


def test_transfer_shell_local_filesystem(tmp_path):
    """Execute only the changed shell commands in a local sandbox, never SSH."""
    import hashlib, io, os, shutil, subprocess, tarfile
    bash=shutil.which('bash') if os.name!='nt' else 'C:/Program Files/Git/usr/bin/bash.exe'
    root=Path(__file__).resolve().parents[1]
    script=(root/'tools/hb156_transfer_v3.ps1').read_text()
    commands=[line.split("$remote '",1)[1].rsplit("'",1)[0]
              for line in script.splitlines() if '& ssh ' in line]
    assert len(commands)==3
    workspace=tmp_path/'workspace';workspace.mkdir()
    upload=workspace/'hb156-l2-s43-upload-v1'
    deploy=workspace/'hb156-convnext-l2-s43-v1'
    posix=subprocess.run([bash,'-c','pwd'],cwd=tmp_path,text=True,capture_output=True,check=True).stdout.strip()
    def execute(command):
        return subprocess.run([bash,'-c',command.replace('/workspace',posix+'/workspace')],
                              cwd=tmp_path,text=True,encoding='utf-8',errors='replace',capture_output=True,timeout=10)
    assert execute(commands[1]).returncode==0
    assert execute(commands[1]).returncode!=0  # existing upload refused
    payload=b'local CPU transfer smoke\n'
    digest=hashlib.sha256(payload).hexdigest()
    archive=upload/'hb156-convnext-l2-s43-v1.tar'
    with tarfile.open(archive,'w') as tar:
        for name,data in [('payload.txt',payload),('SHA256SUMS',(digest+'  payload.txt\n').encode())]:
            info=tarfile.TarInfo('hb156-convnext-l2-s43-v1/'+name);info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    pin=archive.with_suffix('.tar.sha256')
    pin.write_text('0'*64+'  '+archive.name+'\n',newline='\n')
    assert execute(commands[2]).returncode!=0 and not deploy.exists()
    pin.write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n',newline='\n')
    deploy.mkdir()
    assert execute(commands[2]).returncode!=0 and not list(deploy.iterdir())
    deploy.rmdir()  # test-owned empty fixture only
    result=execute(commands[2])
    assert result.returncode==0,result.stdout+result.stderr
    assert (deploy/'payload.txt').read_bytes()==payload
    assert sorted(p.name for p in workspace.iterdir())==['hb156-convnext-l2-s43-v1','hb156-l2-s43-upload-v1']
    assert not (deploy/'hb156-convnext-l2-s43-v1').exists()


@pytest.mark.parametrize('action',['launch-check','smoke','fit'])
def test_shared_lock_blocks_before_any_work(monkeypatch,tmp_path,action):
    from tools import job_runtime
    from tools import hb156_admission_v1 as a
    lock=tmp_path/'gpu.lock'
    original=job_runtime.file_lock
    def mapped(path,*,blocking):
        assert path.as_posix()=='/workspace/.hb151-gpu.lock' and blocking is False
        return original(lock,blocking=False)
    monkeypatch.setattr(job_runtime,'file_lock',mapped)
    monkeypatch.setattr(a,'require_idle',lambda *x:pytest.fail('passed held lock'))
    with original(lock,blocking=False):
        with pytest.raises(BlockingIOError): b.main([action])


@pytest.mark.parametrize('action',['launch-check','smoke','fit'])
def test_process_fallback_blocks_before_ready(monkeypatch,tmp_path,action):
    from contextlib import nullcontext
    from tools import job_runtime
    from tools import hb156_admission_v1 as a
    monkeypatch.setattr(job_runtime,'file_lock',lambda *x,**k:nullcontext())
    monkeypatch.setattr(a,'hb151_processes',lambda allowed:[43])
    monkeypatch.setattr(b,'ready',lambda:pytest.fail('busy job reached ready'))
    monkeypatch.setattr(b,'desktop_memory_snapshot',lambda:pytest.fail('blocker must reject first'))
    with pytest.raises(ValueError,match='another HB151'): b.main([action])


def test_process_detection(monkeypatch):
    from types import SimpleNamespace
    from tools import hb156_admission_v1 as a
    def proc(pid,name,args): return SimpleNamespace(pid=pid,info=dict(name=name,cmdline=args))
    rows=[proc(10,'python',['python','-m','tools.hb151_vast43_v1','fit']),
          proc(11,'python3',['python','-m','tools.hb151_vast43_v1','_fit']),
          proc(12,'python',['python','-m','tools.hb156_vast_v1','smoke']),
          proc(13,'python',['python','-m','tools.unrelated','fit']),
          proc(14,'bash',['bash','hb151.py','fit']),
          proc(15,'python',['python','-m','tools.hb151_vast43_v1','verify']),
          proc(17,'python',['python','-m','tools.hb156_vast_v1','fit']),
          proc(18,'python',['python','-m','tools.hb156_vast_v1','_fit']),
          proc(19,'python',['python','-m','tools.hb156_vast_v1','verify'])]
    monkeypatch.setattr(a.psutil,'process_iter',lambda *x:rows)
    assert a.hb151_processes()==[10,11,12,17,18]
    assert a.hb151_processes([10,12])==[11,17,18]
    rows.append(proc(16,'python',None))
    with pytest.raises(ValueError,match='cannot inspect'): a.hb151_processes()


@pytest.mark.parametrize('free,passes',[(14335,False),(14336,True),(24564,True)])
def test_fresh_memory_gate(monkeypatch,free,passes):
    from tools import hb156_admission_v1 as a
    monkeypatch.setattr(a,'hb151_processes',lambda allowed:[])
    snapshot=lambda:dict(free_mib=free)
    if passes: assert a.require_idle(snapshot)['free_mib']==free
    else:
        with pytest.raises(ValueError,match='14 GiB'): a.require_idle(snapshot)


def test_checkpoint_disk_threshold(package,monkeypatch):
    from types import SimpleNamespace
    reserve=24*b.INIT.stat().st_size+2*1024**3
    monkeypatch.setattr(b,'load_registration',lambda *a:package)
    monkeypatch.setattr(b,'verify_cache',lambda p:{})
    monkeypatch.setattr(b.shutil,'disk_usage',lambda path:SimpleNamespace(free=reserve-1))
    with pytest.raises(ValueError,match='disk reserve'): b.ready()
    monkeypatch.setattr(b.shutil,'disk_usage',lambda path:SimpleNamespace(free=reserve))
    assert b.ready()['disk_free']==reserve


def test_launch_check_returns_without_launch(monkeypatch):
    from contextlib import nullcontext
    from tools import job_runtime
    from tools import hb156_admission_v1 as a
    monkeypatch.setattr(job_runtime,'file_lock',lambda *x,**k:nullcontext())
    monkeypatch.setattr(a,'hb151_processes',lambda allowed:[])
    monkeypatch.setattr(b,'desktop_memory_snapshot',lambda:dict(free_mib=24564))
    monkeypatch.setattr(b,'ready',lambda:pytest.fail('launch-check reached ready'))
    monkeypatch.setattr(b,'supervise',lambda *x:pytest.fail('launch-check launched GPU'))
    assert b.main(['launch-check'])==0


@pytest.mark.parametrize('mode',['smoke','fit'])
def test_bash_busy_no_background_launch(tmp_path,mode):
    import os,shutil,subprocess
    bash=shutil.which('bash') if os.name!='nt' else 'C:/Program Files/Git/usr/bin/bash.exe'
    shutil.copyfile(Path(__file__).resolve().parents[1]/'tools/hb156_run_v3.sh',tmp_path/'run.sh')
    script='python() { echo "BLOCKED $*"; return 23; }; nohup() { echo UNEXPECTED; return 99; }; export -f python nohup; source ./run.sh "$1"'
    result=subprocess.run([bash,'-c',script,'fixture',mode],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode==23
    assert 'launch-check' in result.stdout and 'STARTED' not in result.stdout and 'UNEXPECTED' not in result.stdout
    assert not (tmp_path/'stage-convnext-l2-s43-v1').exists()


def test_supervisor_rechecks_idle_after_launcher(monkeypatch):
    from tools import hb156_admission_v1 as a
    monkeypatch.setattr(b.sys,'platform','linux')
    monkeypatch.setattr(b,'admission_snapshot',lambda:'')  # container can hide compute PIDs
    monkeypatch.setattr(a,'hb151_processes',lambda allowed:[43])
    with pytest.raises(ValueError,match='another HB151'): b.require_gpu_job('fit')








def test_registered_decision_and_arch_cannot_drift(package,monkeypatch):
    original=b.read_json
    altered=copy.deepcopy(package)
    altered['decision']['dead_max']=0.950
    content=copy.deepcopy(altered)
    content.pop('content_sha256')
    altered['content_sha256']=b.parent.json_hash(content)
    monkeypatch.setattr(b,'read_json',lambda path:copy.deepcopy(altered) if Path(path)==b.PROTOCOL else original(path))
    with pytest.raises(ValueError,match='decision drift'):
        b.load_registration(b.PROTOCOL)
    altered=copy.deepcopy(package)
    altered['recipe']['arch']='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
    content=copy.deepcopy(altered)
    content.pop('content_sha256')
    altered['content_sha256']=b.parent.json_hash(content)
    with pytest.raises(ValueError,match='recipe delta'):
        b.load_registration(b.PROTOCOL)


@pytest.mark.parametrize('score,expected',[(.950,'alive'),(.951,'alive'),(.946,'dead'),
    (.945,'dead'),(.947,'seed43'),(.949999,'seed43')])
def test_candidate_decision_blocks_below_alive_floor(package,score,expected):
    assert b.public_decision(score)==expected
    if score < .950:
        assert b.public_decision(score)!='alive'


@pytest.mark.parametrize('score',[float('nan'),float('inf'),-.001,1.001])
def test_candidate_decision_rejects_invalid_score(package,score):
    with pytest.raises(ValueError,match='invalid AUC'):
        b.public_decision(score)
