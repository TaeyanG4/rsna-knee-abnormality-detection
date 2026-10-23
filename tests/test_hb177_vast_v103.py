"""Vast transport, environment and admission regressions (CPU only)."""
import copy
import io
import json
import tarfile
from pathlib import Path
import pytest
from tools import hb177_vast_v103 as v
from tools import hb177_build_v103 as b
from tools import hb177_build_v1 as old


def test_recipe_and_preservation():
    assert b.VERSION=='1.0.3' and b.RECIPE==old.RECIPE
    pins=v.read(b.BASE/'prior-preservation.json')
    assert len(pins)>=13
    assert all(v.sha(b.ROOT/p)==h for p,h in pins.items())


def smoke():
    return dict(device_name='NVIDIA GeForce RTX 4090',total_mib=49140,desktop_used_mib=100,
        train_updates=3,train_shape=[1,12,3,384,384],accumulation=16,serving_views=2,serving_k=94,
        precision='bf16',grad_checkpointing=True,train_peak_reserved_bytes=12*1024**3,
        serve_peak_reserved_bytes=11*1024**3,seconds_per_update=5.)


def test_memory_identity_and_headroom():
    b.check_memory(smoke())
    for change in ({'total_mib':float('nan')},{'total_mib':-1},
                   {'train_peak_reserved_bytes':12*1024**3+1},{'desktop_used_mib':40000},
                   {'accumulation':8},{'train_updates':2},{'serving_k':12},
                   {'seconds_per_update':float('nan')}):
        with pytest.raises(ValueError): b.check_memory(dict(smoke(),**change))


@pytest.mark.parametrize('total',[16376,24564,49140])
def test_device_agnostic_memory_boundaries(total):
    # Exact 12 GiB peak and exactly 2 GiB remaining headroom are admissible.
    s=dict(smoke(),device_name='any measured device',total_mib=total,
           desktop_used_mib=total-12288-2048)
    b.check_memory(s)
    with pytest.raises(ValueError,match='memory gate'):
        b.check_memory(dict(s,desktop_used_mib=s['desktop_used_mib']+1))
    with pytest.raises(ValueError,match='unmeasured'):
        b.check_memory(dict(s,desktop_used_mib=float('nan')))


def test_exact_frozen_kernel_sources():
    import zipfile
    from tools import hb177_export_v103 as e
    contract=b.read(e.STAGE/e.FILES[2])
    assert contract['archive_sha256']==e.sha(e.ARCHIVE)
    with zipfile.ZipFile(e.ARCHIVE) as archive:
        for name,info in contract['source'].items():
            source=archive.read(info['member']).decode('utf-8').splitlines()
            expected='\n\n'.join('\n'.join(source[start-1:end]) for start,end in info['segments'])+'\n'
            assert (e.STAGE/name).read_text(encoding='utf-8')==expected


@pytest.mark.parametrize('host,seed',[('local',42),('vast',43)])
def test_ema_export_through_actual_frozen_loader(host,seed):
    import torch
    from tools import hb177_export_v103 as e
    b.cpu_guard()
    ema=b.EMA(b.make_model())
    ema.updates=5440
    payload=e.export(ema,20,b.PROTOCOL,'new-fit',b.RECIPE,b.VERSION,seed=seed,host=host)
    assert payload['training_provenance']==dict(checkpoint_kind='final-epoch-EMA',
        swa_epochs_1based=[],protocol_sha256=b.sha(b.PROTOCOL),run_dir='new-fit',
        epoch=20,seed=seed,host=host,ema_updates=5440,recipe_version='1.0.3')
    assert payload['seed'] == payload['provenance']['seed'] == 42
    assert 'seed' in payload['legacy_loader_compatibility']['fields']
    assert not payload['fixture'] and payload['artifact_role']=='trained-final'
    assert 'NOT training history' in payload['legacy_loader_compatibility']['meaning']
    path=b.WORK/'test-export.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,path)
    model=e.load_with_kernel(path)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in ema.module.state_dict().items())
    rt,_,contract=e.kernel_namespace()
    for changes in ({'checkpoint_kind':'final-epoch-EMA'},{'swa_epochs_1based':[]},
                    {'provenance':dict(protocol_sha256='wrong',seed=42,run_dir='new-fit')}):
        with pytest.raises(ValueError,match='mismatch'):
            rt.own_meta(dict(payload,**changes),contract['protocol_sha256'],
                        run_dir=contract['hb151_export']['run_dir'],seed=42)
    payload['model'].pop(next(iter(payload['model'])))
    torch.save(payload,path)
    with pytest.raises(ValueError,match='keys/shapes/dtypes mismatch'):
        e.load_with_kernel(path)
    assert e.export(ema,0,b.PROTOCOL,'cpu',b.RECIPE,b.VERSION,seed=seed,host=host)['fixture']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mode',['smoke','fit'])
def test_supervise_disables_wall_and_throughput_abort(monkeypatch,tmp_path,mode):
    from tools import hb150_watchdog_v1 as w
    monkeypatch.setattr(b,'require_admission',lambda *a:None)
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**k:None)
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'desktop_memory_snapshot',lambda:dict(used_mib=100,total_mib=24564,device_name='4090'))
    monkeypatch.setattr(b,'read',lambda p:smoke())
    calls=[]
    monkeypatch.setattr(w,'supervise',lambda *a,**k:calls.append((a,k)) or 0)
    assert b.supervise(mode)==0
    args,kwargs=calls[0]
    assert kwargs['total'] is None and kwargs['healthy_rates'] is None
    assert kwargs['idle']==900 and kwargs['stall']==600
    assert args[0][-1]=='_'+mode


def test_idle_stall_only_for_slow_progress_and_long_wall_time():
    from tools.hb150_watchdog_v1 import Policy
    p=Policy(0,total=None,healthy_rates=None,idle=900,stall=600)
    for now in range(0,100000,500):
        beat=dict(sequence=now+1,completed_at=now,phase='training_step',event='completed',
                  detail=dict(progress_kind='train_step',cumulative_units=now//500+1,epoch=now))
        assert p.check(now,beat) is None
    assert p.ratio is None
    assert p.check(now+900)=='no_heartbeat_15m'
    p=Policy(0,total=None,healthy_rates=None,idle=900,stall=600)
    stalled=dict(util=100,power=70,used_mib=24000,total_mib=24564)
    assert p.check(0,telemetry=stalled) is None
    assert p.check(600,telemetry=stalled)=='wddm_no_heartbeat_10m'


def env():
    return dict(python='3.12.6',platform='Windows',cuda='12.1',cudnn=90100,
                packages={'torch':'2.5.1+cu121','torchvision':'0.20.1','timm':'1.0.19'},
                imports={},settings={'cudnn_benchmark':False})


def test_environment_allowed_and_blocked_differences():
    a=env(); z=copy.deepcopy(a)
    z.update(python='3.11.10',platform='Linux'); z['packages']['torchvision']='0.20.1+cu121'
    differences,mismatch=v.environment_diff(a,z)
    assert not mismatch and set(differences)=={'python','platform','packages'}
    z['packages']['timm']=None
    assert 'timm' in v.environment_diff(a,z)[1]
    z['settings']['cudnn_benchmark']=True
    assert 'settings' in v.environment_diff(a,z)[1]
    z['python']='3.10.0'
    with pytest.raises(ValueError): v.environment_diff(a,z)


@pytest.mark.parametrize('name',['../x','/x','C:/x','a\\x','a\nx'])
def test_unsafe_paths(name):
    with pytest.raises(ValueError): v.safe_name(name)


def test_manifest_tampering_and_duplicates(tmp_path):
    p=tmp_path/'a'; p.write_bytes(b'payload')
    item=dict(path='a',bytes=7,sha256=v.sha(p)); m={'files':[item]}
    assert v.verify_manifest(tmp_path,m)['files']==1
    with pytest.raises(ValueError,match='duplicate'): v.verify_manifest(tmp_path,{'files':[item,item]})
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'): v.verify_manifest(tmp_path,m)


def test_results_roundtrip_and_reject_unlisted(tmp_path):
    payload=b'fixture'; h=__import__('hashlib').sha256(payload).hexdigest()
    archive=tmp_path/'out.tar'
    def make(extra=False):
        manifest=json.dumps({'files':[dict(path='fit/result.pt',bytes=len(payload),sha256=h)]}).encode()
        with tarfile.open(archive,'w') as tar:
            for name,data in [('fit/result.pt',payload),('results.json',manifest)]+([('extra',b'x')] if extra else []):
                info=tarfile.TarInfo(name); info.size=len(data); tar.addfile(info,io.BytesIO(data))
    make()
    assert v.unpack_results(archive,v.sha(archive),tmp_path/'returned')['bytes']==len(payload)
    with pytest.raises(ValueError,match='SHA256'): v.unpack_results(archive,'bad',tmp_path/'returned')
    make(True)
    with pytest.raises(ValueError,match='unmanifested'): v.unpack_results(archive,v.sha(archive),tmp_path/'returned')


def test_cross_host_rejection_before_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'registration',lambda: ({'local_workspace':str(tmp_path.resolve())},{}))
    monkeypatch.setattr(b,'require_cpu',lambda:None)
    monkeypatch.setattr(b,'ROOT',tmp_path)
    b.select_host('vast')
    calls=[]
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**k:calls.append(True))
    with pytest.raises(ValueError,match='Vast-only'): b.supervise('fit')
    assert not calls
    binding={'protocol_sha256':'x','environment_sha256':'y'}
    review=dict(verdict='PASS',independent=True,reviewer='other',scope='pre-execution',**binding)
    b.check_review(review,binding,'pre-execution')
    with pytest.raises(ValueError,match='stale'): b.check_review(review,dict(binding,environment_sha256='z'),'pre-execution')


def test_remote_commands_and_no_launch_during_cpu():
    for action in ('remote-install','remote-cpu','launch-smoke','launch-fit','status'):
        cmd=v.remote_command(action)
        assert cmd[:len(v.SSH)]==v.SSH and 'StrictHostKeyChecking=yes' in cmd
        assert v.REMOTE in cmd[-1]
    cpu=v.remote_command('remote-cpu')[-1]
    assert 'gpu_queue' not in cpu and 'admit-dry-run' in cpu
    assert 'fit-queue.json' in v.remote_command('launch-fit')[-1]
    assert '--foreground' not in v.remote_command('launch-fit')[-1]
    assert 'pip install -r' in v.remote_command('remote-install')[-1]


def test_running_results_are_not_read(monkeypatch,tmp_path):
    from tools import job_runtime
    monkeypatch.setattr(v,'STAGE',tmp_path)
    monkeypatch.setattr(job_runtime,'job_status',lambda *a,**k:{'state':'running'})
    with pytest.raises(ValueError,match='completed successful'): v.result_pack('fit')


def test_secret_scan_suppresses_match(tmp_path):
    v.scan(Path(v.__file__))
    p=tmp_path/'source.py'; p.write_text('token='+('gh'+'p_')+'a'*25)
    with pytest.raises(ValueError,match='value suppressed') as exc: v.scan(p)
    assert 'a'*25 not in str(exc.value)


def test_send_sealed_archive_and_exact_host(monkeypatch,tmp_path):
    monkeypatch.setattr(v,'ROOT',tmp_path); monkeypatch.setattr(v,'STAGE',tmp_path)
    payload=tmp_path/'code.py'; payload.write_bytes(b'pass\n')
    v.write(tmp_path/'transfer-inventory.json',{'files':[dict(path='code.py',bytes=5,sha256=v.sha(payload))]})
    archive=tmp_path/'transfer.tar'; archive.write_bytes(b'fixture')
    (tmp_path/'transfer.tar.sha256').write_text(v.sha(archive))
    calls=[]; monkeypatch.setattr(v,'run',lambda args:calls.append(args))
    v.send()
    assert len(calls)==3 and 'test ! -e' in calls[0][-1]
    assert calls[1][0]=='scp' and v.HOST in calls[1][-1]
    assert 'sha256sum --quiet -c' in calls[2][-1]
    archive.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='archive drift'): v.send()
    assert len(calls)==3


def test_environment_receipt_live_drift(monkeypatch,tmp_path):
    monkeypatch.setattr(v,'STAGE',tmp_path)
    v.write(tmp_path/'local-environment.json',{'reference':True})
    actual=env(); actual['packages']={name:'1.0' for name in v.PACKAGES}
    v.write(tmp_path/'remote-environment.json',dict(status='PASS_ENV_CPU',actual=actual,
                                                 local_sha256=v.sha(tmp_path/'local-environment.json')))
    monkeypatch.setattr(v.platform,'python_version',lambda:actual['python'])
    monkeypatch.setattr(v.platform,'system',lambda:'Linux')
    monkeypatch.setattr(v.metadata,'version',lambda name:'1.0')
    v.require_environment()
    monkeypatch.setattr(v.metadata,'version',lambda name:'changed' if name=='timm' else '1.0')
    with pytest.raises(ValueError,match='package drift'): v.require_environment()


def test_approval_transfer_missing_fails_before_network(monkeypatch,tmp_path):
    monkeypatch.setattr(v,'STAGE',tmp_path)
    calls=[]; monkeypatch.setattr(v,'run',lambda args:calls.append(args))
    with pytest.raises(ValueError,match='missing independent'): v.send_approval()
    assert not calls


@pytest.mark.parametrize('host,seed',[('local',42),('vast',43)])
def test_host_review_and_fit_b_authorization(monkeypatch,tmp_path,host,seed):
    from types import SimpleNamespace
    b.select_host(host)
    monkeypatch.setattr(b,'os',SimpleNamespace(name='posix' if host=='vast' else 'nt'))
    monkeypatch.setattr(b,'BASE',tmp_path)
    monkeypatch.setattr(b,'ROOT',tmp_path); monkeypatch.setattr(b,'REMOTE_ROOT',tmp_path)
    monkeypatch.setattr(b,'STAGE',tmp_path); monkeypatch.setattr(b,'PROTOCOL',tmp_path/'registration.json')
    monkeypatch.setattr(b,'registration',lambda:({'local_workspace':str(tmp_path.resolve())},{})); monkeypatch.setattr(b,'require_cpu',lambda:None)
    monkeypatch.setattr(v,'require_environment',lambda:None)
    monkeypatch.setattr(v,'require_local_environment',lambda:None)
    for name in ('registration.json','cpu-validation.json','remote-environment.json','local-environment.json','cuda-runtime.json'):
        v.write(tmp_path/name,{'fixture':True})
    with pytest.raises(FileNotFoundError): b.require_admission()
    pre=dict(verdict='PASS',independent=True,reviewer='other',scope='pre-execution',**b.review_bindings())
    v.write(tmp_path/'pre-review.json',pre)
    auth=dict(authorized_by='B',protocol_sha256=v.sha(b.PROTOCOL),pre_review_sha256=v.sha(tmp_path/'pre-review.json'),allow_smoke=True)
    v.write(tmp_path/'b-authorization.json',auth); b.require_admission()
    v.write(tmp_path/'cuda-smoke.json',dict(smoke(),protocol_sha256=v.sha(b.PROTOCOL),host=host,seed=seed))
    v.write(tmp_path/'smoke-attempt/receipt.json',dict(exit_code=0,reason='completed',survivors=[]))
    review=dict(verdict='PASS',independent=True,reviewer='other',scope='smoke-result-and-fit',**b.review_bindings(True))
    v.write(tmp_path/'fit-review.json',review)
    with pytest.raises(ValueError,match='B fit authorization'): b.require_admission(True)
    auth.update(allow_fit=True,fit_review_sha256=v.sha(tmp_path/'fit-review.json'))
    v.write(tmp_path/'b-authorization.json',auth); b.require_admission(True)
    v.write(tmp_path/'cuda-runtime.json',{'drift':True})
    with pytest.raises(ValueError,match='stale review'): b.require_admission(True)


@pytest.fixture(autouse=True)
def isolated_host_state(monkeypatch):
    for name in ('HOST','SEED','TAG','STAGE','WORK','RUN'):
        monkeypatch.setattr(b,name,getattr(b,name))
    b.select_host('local')


@pytest.mark.parametrize('host,seed,tag',[('local',42,'HB177_FIT_LOCAL'),('vast',43,'HB177_FIT')])
def test_seed_consumers_and_host_queue(monkeypatch,tmp_path,host,seed,tag):
    import torch
    from exp143_own0949 import StudyDataset
    b.select_host(host)
    base={'train_ids':['fixture']*4347,'train_labels':{'fixture':[.5]*12}}
    ds,sampler,loader=b.make_loader(base,pin_memory=False)
    assert ds.seed == sampler.seed == loader.generator.initial_seed() == seed
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'registration',lambda:None)
    monkeypatch.setattr(b,'require_cpu',lambda:None)
    b.prepare()
    for mode in ('smoke','fit'):
        queue=b.read(tmp_path/(mode+'-queue.json'))
        assert queue[0]['tag'] == tag+('_SMOKE' if mode=='smoke' else '')
        assert queue[0]['command'][-2:] == ['tools.hb177_build_v103',mode]
    assert not torch.cuda.is_initialized()


def test_seeds_registered_before_fit_and_source_bound():
    registration,_=b.registration()
    assert registration['execution_hosts'] == b.EXECUTION_HOSTS
    assert {x['seed'] for x in registration['execution_hosts'].values()} == {42,43}
    assert registration['recipe'] == old.RECIPE


def test_model_and_epoch_rng_dispatch(monkeypatch):
    import ast
    import inspect
    import torch
    import exp143_own0949 as data
    seen=[]
    monkeypatch.setattr(data,'_cpu_seed',lambda seed:seen.append(seed))
    monkeypatch.setattr(data,'install_portable_gold_blocker',lambda:None)
    monkeypatch.setattr(b,'make_model',lambda init:torch.nn.Linear(1,1))
    monkeypatch.setattr(b,'optimization',lambda model:None)
    for host in ('local','vast'):
        b.select_host(host)
        b.setup('cpu',{'pos_weight':[1.]})
    assert seen == [42,43]
    fit_ast=ast.parse(inspect.getsource(b.fit))
    calls=[n for n in ast.walk(fit_ast) if isinstance(n,ast.Call)]
    epoch_calls=[n for n in calls if ast.unparse(n.func) in ('_cpu_seed','torch.cuda.manual_seed_all')
                 and ast.unparse(n.args[0]) == 'SEED * 1000 + epoch']
    assert len(epoch_calls)==2
    assert any(ast.unparse(n.func)=='torch.cuda.manual_seed_all' and ast.unparse(n.args[0])=='SEED' for n in calls)
    smoke_ast=ast.parse(inspect.getsource(b.cuda_smoke))
    assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.manual_seed_all'
               and ast.unparse(n.args[0])=='SEED' for n in ast.walk(smoke_ast))


def test_local_environment_live_drift(monkeypatch,tmp_path):
    monkeypatch.setattr(v,'STAGE',tmp_path)
    actual=env(); actual['packages']={name:'1.0' for name in v.PACKAGES}
    v.write(tmp_path/'local-environment.json',actual)
    monkeypatch.setattr(v.platform,'python_version',lambda:actual['python'])
    monkeypatch.setattr(v.platform,'system',lambda:'Windows')
    monkeypatch.setattr(v.metadata,'version',lambda name:'1.0')
    v.require_local_environment()
    monkeypatch.setattr(v.metadata,'version',lambda name:'changed' if name=='timm' else '1.0')
    with pytest.raises(ValueError,match='package drift'): v.require_local_environment()
    monkeypatch.setattr(v.metadata,'version',lambda name:'1.0')
    monkeypatch.setattr(v.platform,'system',lambda:'Linux')
    with pytest.raises(ValueError,match='Python/platform drift'): v.require_local_environment()


def test_local_and_vast_evidence_separate():
    b.select_host('local'); local=(b.STAGE,b.RUN,b.TAG)
    b.select_host('vast'); remote=(b.STAGE,b.RUN,b.TAG)
    assert local[0] == remote[0]/'local'
    assert all(x!=y for x,y in zip(local,remote))
    assert b.PROTOCOL == b.BASE/'registration.json'


def test_cpu_receipt_wrong_seed_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'STAGE',tmp_path)
    b.atomic_json(tmp_path/'cpu-validation.json',dict(status='PASS_CPU',protocol_sha256=b.sha(b.PROTOCOL),host='local',seed=43))
    with pytest.raises(ValueError,match='stale/missing'): b.require_cpu()
