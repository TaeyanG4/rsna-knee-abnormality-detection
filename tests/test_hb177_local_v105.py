"""Local seed44 registration, serving and fail-closed gates; CPU only."""
from pathlib import Path
import pytest
from tools import hb177_local_v105 as v
from tools import hb177_build_v105 as b

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
    from tools import hb177_export_v105 as e
    contract=b.read(e.STAGE/e.FILES[2])
    assert contract['archive_sha256']==e.sha(e.ARCHIVE)
    with zipfile.ZipFile(e.ARCHIVE) as archive:
        for name,info in contract['source'].items():
            source=archive.read(info['member']).decode('utf-8').splitlines()
            expected='\n\n'.join('\n'.join(source[start-1:end]) for start,end in info['segments'])+'\n'
            assert (e.STAGE/name).read_text(encoding='utf-8')==expected


@pytest.mark.parametrize('host,seed',[('local',44)])
def test_ema_export_through_actual_frozen_loader(host,seed,tmp_path,monkeypatch):
    monkeypatch.setattr(b,'WORK',tmp_path)
    import torch
    from tools import hb177_export_v105 as e
    b.cpu_guard()
    ema=b.EMA(b.make_model())
    ema.updates=5440
    payload=e.export(ema,20,b.PROTOCOL,'new-fit',b.RECIPE,b.VERSION,seed=seed,host=host)
    assert payload['training_provenance']==dict(checkpoint_kind='final-epoch-EMA',
        swa_epochs_1based=[],protocol_sha256=b.sha(b.PROTOCOL),run_dir='new-fit',
        epoch=20,seed=seed,host=host,ema_updates=5440,recipe_version='1.0.5')
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


@pytest.mark.parametrize('host,seed',[('local',44)])
def test_host_review_and_fit_b_authorization(monkeypatch,tmp_path,host,seed):
    from types import SimpleNamespace
    b.select_host(host)
    monkeypatch.setattr(b,'os',SimpleNamespace(name='posix' if host=='vast' else 'nt'))
    monkeypatch.setattr(b,'BASE',tmp_path)
    monkeypatch.setattr(b,'ROOT',tmp_path)
    monkeypatch.setattr(b,'STAGE',tmp_path); monkeypatch.setattr(b,'PROTOCOL',tmp_path/'registration.json')
    monkeypatch.setattr(b,'registration',lambda:({'local_workspace':str(tmp_path.resolve())},{})); monkeypatch.setattr(b,'require_cpu',lambda:None)
    monkeypatch.setattr(v,'require_local_environment',lambda:None)
    for name in ('registration.json','cpu-validation.json','remote-environment.json','local-environment.json','cuda-runtime.json'):
        b.atomic_json(tmp_path/name,{'fixture':True})
    with pytest.raises(FileNotFoundError): b.require_admission()
    pre=dict(verdict='PASS',independent=True,reviewer='other',scope='pre-execution',**b.review_bindings())
    b.atomic_json(tmp_path/'pre-review.json',pre)
    auth=dict(authorized_by='B',protocol_sha256=v.sha(b.PROTOCOL),pre_review_sha256=v.sha(tmp_path/'pre-review.json'),allow_smoke=True)
    b.atomic_json(tmp_path/'b-authorization.json',auth); b.require_admission()
    b.atomic_json(tmp_path/'cuda-smoke.json',dict(smoke(),protocol_sha256=v.sha(b.PROTOCOL),host=host,seed=seed))
    b.atomic_json(tmp_path/'smoke-attempt/receipt.json',dict(exit_code=0,reason='completed',survivors=[]))
    review=dict(verdict='PASS',independent=True,reviewer='other',scope='smoke-result-and-fit',**b.review_bindings(True))
    b.atomic_json(tmp_path/'fit-review.json',review)
    with pytest.raises(ValueError,match='B fit authorization'): b.require_admission(True)
    auth.update(allow_fit=True,fit_review_sha256=v.sha(tmp_path/'fit-review.json'))
    b.atomic_json(tmp_path/'b-authorization.json',auth); b.require_admission(True)
    b.atomic_json(tmp_path/'cuda-runtime.json',{'drift':True})
    with pytest.raises(ValueError,match='stale review'): b.require_admission(True)


@pytest.fixture(autouse=True)
def isolated_host_state(monkeypatch):
    for name in ('HOST','SEED','TAG','STAGE','WORK','RUN'):
        monkeypatch.setattr(b,name,getattr(b,name))
    b.select_host('local')


@pytest.mark.parametrize('host,seed,tag',[('local',44,'HB177_FIT_LOCAL_S44')])
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
        assert queue[0]['command'][-2:] == ['tools.hb177_build_v105',mode]
    assert not torch.cuda.is_initialized()


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
    for host in ('local',):
        b.select_host(host)
        b.setup('cpu',{'pos_weight':[1.]})
    assert seen == [44]
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
    b.atomic_json(tmp_path/'local-environment.json',actual)
    monkeypatch.setattr(v.platform,'python_version',lambda:actual['python'])
    monkeypatch.setattr(v.platform,'system',lambda:'Windows')
    monkeypatch.setattr(v.metadata,'version',lambda name:'1.0')
    v.require_local_environment()
    monkeypatch.setattr(v.metadata,'version',lambda name:'changed' if name=='timm' else '1.0')
    with pytest.raises(ValueError,match='package drift'): v.require_local_environment()
    monkeypatch.setattr(v.metadata,'version',lambda name:'1.0')
    monkeypatch.setattr(v.platform,'system',lambda:'Linux')
    with pytest.raises(ValueError,match='Python/platform drift'): v.require_local_environment()


def test_cpu_receipt_wrong_seed_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'STAGE',tmp_path)
    b.atomic_json(tmp_path/'cpu-validation.json',dict(status='PASS_CPU',protocol_sha256=b.sha(b.PROTOCOL),host='local',seed=43))
    with pytest.raises(ValueError,match='stale/missing'): b.require_cpu()


def test_registration_identity_and_parent_preservation():
    p,_=b.registration()
    assert b.VERSION=='1.0.5' and b.SEED==44 and b.TAG=='HB177_FIT_LOCAL_S44'
    assert p['execution_hosts']=={'local':dict(seed=44,tag='HB177_FIT_LOCAL_S44',platform='Windows',device='RTX 4070 Ti SUPER')}
    keys=('recipe','parent_manifest_sha256','init_sha256','corpus_manifest_sha256','label_sha256','train_n','train_ids_sha256','labels_sha256','pos_weight_sha256','serving_files','total_updates','updates_per_epoch')
    for version in ('1.0.3','1.0.4'):
        prior=b.read(b.BASE.parent/('v'+version)/'registration.json')
        assert {k:p[k] for k in keys}=={k:prior[k] for k in keys}
        for name in ('kernel-loader.py','kernel-model.py','kernel-contract.json'):
            assert (b.BASE/name).read_bytes()==(b.BASE.parent/('v'+version)/name).read_bytes()
    assert all(b.sha(b.ROOT/path)==pin for path,pin in b.read(b.BASE/'prior-preservation.json').items())
    assert p['decision']['alive_at_least']==.950


def test_remote_host_refused_and_local_stage():
    with pytest.raises(ValueError,match='local-only'): b.select_host('vast')
    assert b.STAGE==b.BASE/'local' and b.RUN==b.STAGE/'fit'


def test_live_used_memory_blocks_launch_before_child(monkeypatch,tmp_path):
    from tools import hb150_watchdog_v1 as w
    monkeypatch.setattr(b,'require_admission',lambda *a:None)
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**kw:None)
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'read',lambda path:dict(smoke(),total_mib=16376,desktop_used_mib=100))
    # Historical smoke passes; another workload occupies memory at actual launch.
    monkeypatch.setattr(b,'desktop_memory_snapshot',lambda:dict(used_mib=2041,total_mib=16376,device_name='RTX 4070 Ti SUPER'))
    calls=[]
    monkeypatch.setattr(w,'supervise',lambda *a,**kw:calls.append(a) or 0)
    with pytest.raises(ValueError,match='memory gate'): b.supervise('fit')
    assert not calls
    monkeypatch.setattr(b,'desktop_memory_snapshot',lambda:dict(used_mib=2040,total_mib=16376,device_name='RTX 4070 Ti SUPER'))
    assert b.supervise('fit')==0 and len(calls)==1


def test_scientific_function_identity():
    import ast
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    names=('parent','cache_inventory','corpus_check','save_torch','cpu_guard','setup','real_batches','make_loader','require_cpu','check_memory','check_review')
    current=functions(Path(b.__file__))
    for version in ('103','104'):
        prior=functions(b.ROOT/f'tools/hb177_build_v{version}.py')
        assert all(current[n]==prior[n] for n in names)
    e=b.ROOT/'tools/hb177_export_v105.py'
    assert e.read_text()==(b.ROOT/'tools/hb177_export_v104.py').read_text().replace('v104','v105').replace('1.0.4','1.0.5')


def test_review_bypass_before_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'registration',lambda:({'local_workspace':str(b.ROOT.resolve())},{}))
    monkeypatch.setattr(b,'require_cpu',lambda:None)
    monkeypatch.setattr(v,'require_local_environment',lambda:None)
    calls=[]
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**kw:calls.append(True))
    with pytest.raises(FileNotFoundError): b.supervise('fit')
    assert not calls
