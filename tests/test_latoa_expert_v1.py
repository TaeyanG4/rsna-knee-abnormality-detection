"""Candidate-specific formula, frozen recipe, provenance and admission regressions."""
import ast
import copy
from pathlib import Path
import numpy as np
import pytest
from tools import latoa_expert_v1 as b
from tools import latoa_expert_eval_v1 as e


def test_route_only_lateral_ties():
    a=np.random.default_rng(12).random((20,12))
    crop=np.round(np.random.default_rng(13).random((20,12)),1)
    out=e.route(a,crop)
    assert out[:,e.OTHER].tobytes()==a[:,e.OTHER].tobytes()
    assert np.array_equal(out[:,e.LAT],(.5*e.ranks(a)+.5*e.ranks(crop))[:,e.LAT])
    assert not np.array_equal(out[:,e.LAT],a[:,e.LAT])


def test_nonfinite_route_rejected():
    a=np.ones((4,12)); a[0,0]=np.nan
    with pytest.raises(ValueError): e.route(a,a)


@pytest.mark.parametrize('deltas,identical,expected', [
    ([.006]*4,True,True), ([.01,.01,.01,-.001],True,True),
    ([.005]*4,True,False), ([.02,.02,-.001,-.001],True,False),
    ([.02]*4,False,False), ([float('nan')]*4,True,False)])
def test_registered_gate(deltas,identical,expected):
    m={'rulers':{k:{'delta':d} for k,d in zip(b.RULERS,deltas)},'other_11_identical':identical}
    assert e.go_rule(m) is expected
    m['rulers'].pop(b.RULERS[0]); assert not e.go_rule(m)


def test_metrics_real_shape_logic():
    a=np.arange(72,dtype=float).reshape(6,12)/72
    y=np.tile([0,0,0,1,1,1],(12,1)).T
    refs={k:(y,np.ones_like(y,dtype=bool),1) for k in b.RULERS}
    m=e.metrics(a,e.route(a,a),refs)
    assert m['mean_delta']==0 and m['nonnegative_rulers']==4 and m['other_11_identical']
    refs.pop(b.RULERS[0])
    with pytest.raises(ValueError): e.metrics(a,a,refs)


def test_exact_population_and_recipe():
    train,val,groups=b.population()
    assert (len(train),len(val))==(3190,977)
    assert not set(train)&set(val)
    assert not {groups[u] for u in train}&{groups[u] for u in val}
    parent=b.read(b.PARENT)
    for seed in b.SEEDS:
        c=b.validate(b.config_path(seed))
        p=copy.deepcopy(parent['recipe']); q=copy.deepcopy(c['recipe'])
        for name in ('roots','updates'): p.pop(name); q.pop(name)
        assert p==q
        assert c['weights']==parent['weights'] and c['resolution']==parent['resolution']==224
        assert c['optimizer_steps']==7960 and c['gold']==[] and c['fold']==0
    other=b.read(b.PARENT.with_name('crop-s43.json'))
    assert other['recipe']==parent['recipe'] and other['weights']==parent['weights']


def test_fit_scientific_functions_unchanged():
    def funcs(p):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=funcs(b.ROOT/'tools/hb165_fit_v1.py'); new=funcs(b.ROOT/'tools/latoa_expert_fit_v1.py')
    for name in ('TinyModel','save_epoch','export_prediction','scientific_targets'):
        assert old[name]==new[name]
    # Whole fit function differs only in namespaces, identity strings, and documented memory gates.
    before=(b.ROOT/'tools/hb165_fit_v1.py').read_text()
    after=(b.ROOT/'tools/latoa_expert_fit_v1.py').read_text()
    def fit_source(s):
        return ast.get_source_segment(s,next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='fit'))
    expected=fit_source(before).replace('tools.hb165_build_v1','tools.latoa_expert_v1').replace('H-B-165','LATOA_EXPERT_V1').replace('fold3','fold0').replace('FOLD3','FOLD0')
    expected=expected.replace('10*(1<<30)','12*(1<<30)').replace('10 GiB','12 GiB')
    expected=expected.replace('        peak =','        memory_guard()\n        peak =')
    expected=expected.replace('        h.need(counters["updates"] == steps and (not smoke or counters["skips"] == 0), "incomplete fixed-final fit")',
        '        from tools.latoa_expert_v1 import exact_optimizer_steps\n        actual_optimizer_steps = exact_optimizer_steps(counters, steps)')
    expected=expected.replace('optimizer_steps=counters["updates"]-counters["skips"],',
        'optimizer_steps=actual_optimizer_steps, attempted_steps=counters["updates"],')
    expected=expected.replace('attempted_steps=counters["updates"], validation_shapes=',
        'attempted_steps=counters["updates"],\n            validation_shapes=')
    expected=expected.replace('dict(status="FAILED",error=type(exc).__name__,',
        'dict(status="FAILED",error=type(exc).__name__, counters=counters,\n            optimizer_steps=counters["updates"]-counters["skips"],')
    assert fit_source(after)==expected


def test_partial_baseline_fail_closed(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'BASELINE',tmp_path)
    b.atomic_json(tmp_path/'pilot-receipt.json',{'status':'RUNNING'})
    with pytest.raises(ValueError): e.load_baseline(['unused']*977)


@pytest.mark.parametrize('change',[{'seed':44},{'fold':3},{'epochs':19},{'engineering_only':True},{'smoke':True},{'prepare_sha256':'drift'}])
def test_crop_provenance_rejects_wrong_training(change):
    identity=dict(experiment='LATOA_EXPERT_V1 OA crop expert fold0 confirmation',seed=46,fold=0,epochs=20,
        smoke=False,engineering_only=False,config_sha256='c',prepare_sha256='p',val_sha256='v',clean_oof=False)
    e.check_crop_identity(identity,46,'c','p','v')
    identity.update(change)
    with pytest.raises(ValueError): e.check_crop_identity(identity,46,'c','p','v')


def test_review_bypass_blocks(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'OUT',tmp_path); monkeypatch.setattr(b,'prepared',lambda:None)
    monkeypatch.setattr(b,'sha',lambda p:'pin')
    b.atomic_json(tmp_path/'cpu-smoke.json',dict(status='PASS',prepared_sha256='pin',cuda_initialized=False))
    with pytest.raises(FileNotFoundError): b.require_review()
    b.atomic_json(tmp_path/'PRE-REVIEW-v2.json',dict(verdict='PASS',independent=True,builder='astra-b5',reviewer='astra-b5'))
    with pytest.raises(ValueError): b.require_review()


def test_fit_review_mandatory(monkeypatch):
    monkeypatch.setattr(b,'validate',lambda p:None)
    calls=[]
    def review(fit=False):
        calls.append(fit)
        if fit: raise ValueError('missing result-bound fit review')
    monkeypatch.setattr(b,'require_review',review)
    with pytest.raises(ValueError): b.require_fit('x')
    assert calls==[False,True]


def test_unowned_launch_blocked(monkeypatch):
    monkeypatch.setattr(b,'read',lambda p:dict(command=['wrong'],command_sha256='',cwd=str(b.ROOT)))
    with pytest.raises(ValueError): b.owner_check('smoke')


def test_memory_guard(monkeypatch):
    from tools import latoa_expert_fit_v1 as f
    monkeypatch.setattr(f.h.torch.cuda,'is_initialized',lambda:True)
    monkeypatch.setenv('LATOA_DESKTOP_BYTES',str(2<<30)); monkeypatch.setenv('LATOA_TOTAL_BYTES',str(16<<30))
    monkeypatch.setattr(f.h.torch.cuda,'max_memory_reserved',lambda:12<<30)
    f.memory_guard()
    monkeypatch.setattr(f.h.torch.cuda,'max_memory_reserved',lambda:13<<30)
    with pytest.raises(ValueError): f.memory_guard()


def test_queue_plans():
    from tools.gpu_queue import validate_plan
    for mode in ('smoke','fit'):
        p=b.read(b.queue_path(mode))
        validate_plan(p)
        assert p==[dict(tag=b.tag(mode),command=b.command(mode))]


def test_runtime_wrong_host_and_missing_owner(monkeypatch):
    monkeypatch.setattr(b,'validate',lambda p:None)
    monkeypatch.setattr(b,'require_review',lambda:None)
    with pytest.raises(ValueError,match='runtime'): b.require_runtime('x',{'gpu':'wrong'})
    snapshot=dict(gpu=b.GPU,torch='2.5.1+cu121',cuda='12.1',libs=b.old.PINS,
        matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,
        deterministic_algorithms=True,flash_sdp=True,mem_efficient_sdp=True,math_sdp=True)
    monkeypatch.delenv('LATOA_OWNER_PID',raising=False)
    with pytest.raises(ValueError,match='supervisor'): b.require_runtime('x',snapshot)


def test_baseline_wrong_checkpoint_and_stride(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'BASELINE',tmp_path); monkeypatch.setattr(b,'sha',lambda p:'pin')
    b.atomic_json(tmp_path/'REVIEW.json',{'verdict':'PASS'})
    receipt=dict(status='COMPLETE',preregistration_sha256='pin',review_sha256='pin',
        outputs={'v2_s2':dict(csv_sha256='pin',checkpoint_sha256='wrong',eval_stride=2)})
    b.atomic_json(tmp_path/'pilot-receipt.json',receipt)
    with pytest.raises(ValueError,match='checkpoint'): e.load_baseline(['x']*977)
    receipt['outputs']['v2_s2'].update(checkpoint_sha256=e.CHECKPOINT_PINS['v2'],eval_stride=1)
    b.atomic_json(tmp_path/'pilot-receipt.json',receipt)
    with pytest.raises(ValueError,match='stride'): e.load_baseline(['x']*977)


def test_new_fit_blocks_science_before_outputs(monkeypatch,tmp_path):
    from tools import latoa_expert_fit_v1 as f
    monkeypatch.setattr(f,'read_config',lambda p:{})
    with pytest.raises(ValueError,match='CUDA'): f.fit('c',tmp_path/'out',device='cpu',smoke=False)
    def blocked(*args): raise ValueError('independent review missing')
    monkeypatch.setattr(b,'require_fit',blocked)
    with pytest.raises(ValueError,match='independent'): f.fit('c',tmp_path/'out',device='cuda',smoke=False)
    assert not (tmp_path/'out').exists()


def test_authenticated_crop_checkpoint_csv_roundtrip(monkeypatch,tmp_path):
    from training.arms.mil_screen_hooks_v3 import torch
    monkeypatch.setattr(b,'OUT',tmp_path)
    b.atomic_json(b.config_path(46),{'seed':46})
    b.atomic_json(tmp_path/'prepared.json',{'prepared':True})
    ids=[f'synthetic-{i:04d}' for i in range(977)]
    identity=dict(experiment='LATOA_EXPERT_V1 OA crop expert fold0 confirmation',seed=46,fold=0,epochs=20,
        smoke=False,engineering_only=False,config_sha256=b.sha(b.config_path(46)),
        prepare_sha256=b.sha(tmp_path/'prepared.json'),val_sha256=b.uid_hash(ids),clean_oof=False)
    dest=tmp_path/'fits/crop-s46'; dest.mkdir(parents=True)
    torch.save(dict(epoch=19,identity=identity,model={'test':torch.zeros(1)}),dest/'final.pt')
    values=np.random.default_rng(46).random((977,12))
    e.rulers.write_csv(dest/'val.csv',ids,values)
    pin=b.sha(dest/'final.pt')
    b.atomic_json(dest/'summary.json',dict(passed=True,full_shape=True,tiny_model=False,
        engineering_only=False,steps=7960,counters={'updates':7960,'skips':0},
        optimizer_steps=7960,attempted_steps=7960,identity=identity,checkpoint_sha256=pin))
    watch=tmp_path/'watch-fit-s46/receipt.json'
    b.atomic_json(watch,dict(status='PASS',exit_code=0,cleanup_confirmed=True))
    receipt=dict(checkpoint_sha256=pin,csv_sha256=b.sha(dest/'val.csv'),identity=identity,
        k=36,tta=True,engineering_only=False,rows=977,uid_sha256=b.uid_hash(ids))
    b.atomic_json(dest/'val.receipt.json',receipt)
    loaded,_=e.load_crop(46,ids)
    np.testing.assert_allclose(loaded,values,atol=1e-9)
    b.atomic_json(tmp_path/b.PREREG,
        dict(schema='latoa-expert/v2',version='1.0.3',seeds=[46],rule=b.RULE))
    single,receipts=e.load_experts(ids)
    np.testing.assert_array_equal(single,loaded)
    assert set(receipts)=={'46'}
    summary=b.read(dest/'summary.json')
    for claimed in (7881,7960):
        bad=copy.deepcopy(summary); bad['counters']['skips']=79; bad['optimizer_steps']=claimed
        b.atomic_json(dest/'summary.json',bad)
        with pytest.raises(ValueError,match='real optimizer'): e.load_crop(46,ids)
    b.atomic_json(dest/'summary.json',summary)
    b.atomic_json(watch,dict(status='FAILED',exit_code=125,cleanup_confirmed=True))
    with pytest.raises(ValueError,match='watchdog'): e.load_crop(46,ids)
    b.atomic_json(watch,dict(status='PASS',exit_code=0,cleanup_confirmed=True))
    receipt['tta']=False; b.atomic_json(dest/'val.receipt.json',receipt)
    with pytest.raises(ValueError,match='receipt'): e.load_crop(46,ids)


@pytest.mark.parametrize('counters',[
    {'updates':7960,'skips':79},{'updates':7959,'skips':0},{'updates':7961,'skips':1},
    {'updates':7960},{'updates':True,'skips':0},{'updates':7960,'skips':-1},
    {'updates':7960,'skips':7961}])
def test_exact_real_optimizer_count_rejects(counters):
    with pytest.raises(ValueError): b.exact_optimizer_steps(counters,7960)


def test_exact_real_optimizer_count_accepts():
    assert b.exact_optimizer_steps({'updates':7960,'skips':0},7960)==7960


def test_run_watchdog_failure_records_failed_and_preserves_fit_failure(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from tools import l1_rco_runtime_v120 as safety
    from tools import latoa_expert_watchdog_v1 as watchdog
    monkeypatch.setattr(b,'OUT',tmp_path)
    monkeypatch.setattr(b,'require_review',lambda:None)
    monkeypatch.setattr(b,'owner_check',lambda mode:{'pid':123})
    monkeypatch.setattr(b.shutil,'disk_usage',lambda p:SimpleNamespace(free=46*(1<<30)))
    monkeypatch.setattr(safety,'gpu_snapshot',lambda:dict(util=0,desktop_bytes=0,total_bytes=16*(1<<30)))
    monkeypatch.setattr(safety,'active_ml_processes',lambda pid:[])
    failure=dict(status='FAILED',optimizer_steps=7881,counters=dict(updates=7960,skips=79))
    def fail(*args,**kwargs):
        b.atomic_json(tmp_path/'gpu-smoke/failure.json',failure)
        return 125
    monkeypatch.setattr(watchdog,'supervise',fail)
    with pytest.raises(ValueError,match='supervised crop job failed'): b.run('smoke')
    result=b.read(tmp_path/'gpu-smoke/failure.json')
    assert result['status']=='FAILED' and result['fit_failure']==failure
    assert not (tmp_path/'gpu-smoke.json').exists()


@pytest.mark.parametrize('util,active_ml,allowed',[
    (40,False,True),(60,False,False),(70,False,False),
    (0,True,False),(40,True,False),(70,True,False)])
def test_local_utilization_admission_v102(monkeypatch,tmp_path,util,active_ml,allowed):
    from types import SimpleNamespace
    from tools import l1_rco_runtime_v120 as safety
    from tools import latoa_expert_watchdog_v1 as watchdog
    monkeypatch.setattr(b,'OUT',tmp_path)
    monkeypatch.setattr(b,'require_review',lambda:None)
    monkeypatch.setattr(b,'owner_check',lambda mode:{'pid':123})
    monkeypatch.setattr(b.shutil,'disk_usage',lambda p:SimpleNamespace(free=46*(1<<30)))
    monkeypatch.setattr(safety,'gpu_snapshot',lambda:dict(util=util,desktop_bytes=0,total_bytes=16*(1<<30)))
    monkeypatch.setattr(safety,'active_ml_processes',lambda pid:[1234] if active_ml else [])
    reached=[]
    def stop_before_launch(*args,**kwargs):
        reached.append(True)
        raise RuntimeError('CPU test: admitted, no GPU launch')
    monkeypatch.setattr(watchdog,'supervise',stop_before_launch)
    with pytest.raises(RuntimeError if allowed else ValueError,
                       match='CPU test: admitted' if allowed else 'local GPU busy'):
        b.run('smoke')
    assert bool(reached)==allowed
    receipt=b.read(tmp_path/'local-admission-v102-smoke.json')
    assert receipt['util']==util and receipt['util_limit_exclusive']==60
    assert receipt['active_ml']==active_ml
    assert receipt['status']==('PASS' if allowed else 'FAILED')


def test_v2_single_expert_file_happy_path(monkeypatch,tmp_path):
    reg=dict(schema='latoa-expert/v2',version='1.0.3',seeds=[46],rule=b.RULE)
    path=tmp_path/'single.json'; b.atomic_json(path,reg)
    values=np.random.default_rng(46).random((977,12)); ids=list(range(977)); calls=[]
    def load(seed, roster):
        calls.append((seed,roster)); return values,{'authenticated':True}
    monkeypatch.setattr(e,'load_crop',load)
    actual,receipts=e.load_experts(ids,path)
    assert actual is values and receipts=={'46':{'authenticated':True}}
    assert calls==[(46,ids)]
    baseline=np.random.default_rng(12).random((977,12))
    routed=e.route(baseline,actual)
    np.testing.assert_array_equal(routed[:,e.LAT],
        .5*e.ranks(baseline)[:,e.LAT]+.5*e.ranks(values)[:,e.LAT])
    assert routed[:,e.OTHER].tobytes()==baseline[:,e.OTHER].tobytes()


@pytest.mark.parametrize('seeds',[[46,47],[47],[],[46,46],[46.0]])
def test_v2_refuses_pair_and_wrong_seed_files(monkeypatch,tmp_path,seeds):
    path=tmp_path/'experts.json'
    b.atomic_json(path,dict(schema='latoa-expert/v2',version='1.0.3',seeds=seeds,rule=b.RULE))
    monkeypatch.setattr(e,'load_crop',lambda *a:pytest.fail('invalid registration reached expert IO'))
    with pytest.raises(ValueError,match='exactly one seed-46'): e.load_experts([],path)


def test_v2_refuses_original_pair_file():
    with pytest.raises(ValueError,match='exactly one seed-46'):
        e.load_experts([],b.OUT/'PREREGISTRATION.json')


def test_v2_refuses_wrong_seed_before_io():
    with pytest.raises(ValueError,match='unregistered expert seed'): e.load_crop(47,[])
    with pytest.raises(ValueError,match='unregistered seed'): b.config(47)


def test_v2_preserves_originals_and_smoke_recipe():
    assert b.SEEDS==(46,)
    reg=b.read(b.OUT/b.PREREG)
    assert reg==b.registration() and reg['seeds']==[46]
    assert reg['optimizer_steps_per_seed']==7960
    from datetime import datetime
    assert datetime.fromisoformat(reg['registered_at']).utcoffset().total_seconds()==9*3600
    for name in ('PREREGISTRATION.json','fit-queue.json','gpu-smoke.json','PRE-REVIEW.json'):
        assert b.sha(b.OUT/name)==b.sha(b.BACKUP/name)
    old=b.read(b.OUT/'fit-queue.json')
    assert old[0]['command'][-1]=='run-fit'
    assert b.command('fit')[-1]=='run-fit-s46' and b.tag('fit')!=old[0]['tag']
    b.check_smoke_carryover()
