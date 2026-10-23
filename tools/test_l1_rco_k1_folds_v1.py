"""H224 fold isolation, scientific delta, export integrity and offline portability."""
import copy, csv, json, os
from pathlib import Path
import numpy as np
import pytest
os.environ['CUDA_VISIBLE_DEVICES']=''
from tools import l1_rco_k1_folds_v1 as k
from tools import l1_rco_k1_foldcache_v1 as cache
from tools import l1_rco_k1_foldpack_v1 as pack

@pytest.fixture(autouse=True)
def reset_fold():
    k.configure(1)
    yield
    k.configure(1)

@pytest.mark.parametrize('fold,train,hold,steps',[(1,3073,1094,15360),(2,3100,1067,15500),(3,3138,1029,15690)])
def test_scientific_registration(fold,train,hold,steps):
    k.configure(fold);r=k.expected_registration();p=k.read(k.PARENT)
    assert len(r['train'])==train and len(r['holdout'])==hold
    assert set(r['train'])|set(r['holdout'])==set(p['train']+p['holdout'])
    assert not set(r['train'])&set(r['holdout'])
    assert all(k.s.fold(u)==fold for u in r['holdout'])
    changed={'model_fold','only_folds','steps_per_epoch','total_steps','warm_steps','bias','unchanged'}
    assert {n:v for n,v in r['recipe'].items() if n not in changed}=={n:v for n,v in p['recipe'].items() if n not in changed}
    assert r['recipe']['total_steps']==steps
    assert r['pins']==p['pins'] and r['optimizer_policy']==p['optimizer_policy']
    assert r['contract_id']==k.digest({n:v for n,v in r.items() if n!='contract_id'})

@pytest.mark.parametrize('fold',[1,2,3])
def test_exact_derivative_delta(fold):
    k.configure(fold)
    source=(k.ROOT/k.s.SOURCE/'compact_train.py').read_text(encoding='utf-8')
    original=k.parent.derive_trainer(source);actual=k.derive_trainer(source)
    restored=actual.replace('a.only_folds == "'+str(fold)+'"','a.only_folds == "0"')
    restored=restored.replace('all(pool_fold != '+str(fold)+')','all(pool_fold != 0)')
    restored=restored.replace('tools.l1_rco_k1_foldstate_v1','tools.l1_rco_k1_state_v2')
    restored=restored.replace('guard_constant=a.l1_guard_constant, fold='+str(fold)+')','guard_constant=a.l1_guard_constant)')
    restored=restored.replace('tools.l1_rco_k1_folds_v1','tools.l1_rco_k1_v3')
    assert restored==original

def test_fold_partition_and_parity_sample():
    r=k.read(k.PARENT);sample=cache.selected_ids(r)
    assert len(sample)==len(set(sample))==32
    assert [sum(k.s.fold(u)==f for u in sample) for f in range(4)]==[8]*4
    for field in ('train','holdout'):r[field]=list(reversed(r[field]))
    assert sample==cache.selected_ids(r)

def test_command_is_canonical_and_no_launch(monkeypatch):
    monkeypatch.setattr(k.subprocess,'run',lambda *a,**kw:pytest.fail('must not launch'))
    for fold,gpu in ((1,0),(2,0),(3,1)):
        cmd=k.command(fold,gpu=gpu);launch=k.launch_command(fold,gpu=gpu)
        assert launch[launch.index('--')+1:]==cmd
        assert cmd[cmd.index('--fold')+1]==str(fold)
        assert cmd[cmd.index('--gpu')+1]==str(gpu)
        assert 'f'+str(fold) in cmd[-1]


def test_linux_pair_routes_distinct_owners_without_launch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(k, 'BASE', tmp_path)
    monkeypatch.setattr(k, 'os', SimpleNamespace(name='posix', environ={}))
    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(k.subprocess, 'run', fake_run)
    assert k.launch_pair() == {'dispatched': [0, 0]}
    assert len(calls) == 2
    for (argv, kwargs), fold, gpu in zip(calls, (2, 3), (0, 1)):
        assert argv == k.launch_command(fold, gpu=gpu)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert kwargs['env']['KNEE_WORKSPACE_ROOT'] == str(tmp_path / ('gpu%d-owner' % gpu))
        assert kwargs['cwd'] == k.ROOT and kwargs['check']

def write_predictions(path,ids,fold=1,value=.5):
    with path.open('w',newline='',encoding='utf-8') as stream:
        w=csv.writer(stream);w.writerow([k.s.UID]+k.s.LABELS+['status','model_fold'])
        w.writerows([[u]+[value]*12+['ok',fold] for u in ids])

@pytest.mark.parametrize('change',['duplicate','nan','range','fold'])
def test_predictions_fail_closed(tmp_path,change):
    path=tmp_path/'p.csv';reg={'holdout':['a','b']}
    write_predictions(path,['a','a'] if change=='duplicate' else ['a','b'],
                      fold=2 if change=='fold' else 1,value=float('nan') if change=='nan' else 2 if change=='range' else .5)
    with pytest.raises(ValueError):k.prediction_scores(reg,path)

def test_completed_export_and_step_count(tmp_path,monkeypatch):
    monkeypatch.setattr(k,'WORK',tmp_path);monkeypatch.setattr(k,'OUT',tmp_path/'fit')
    out=k.OUT;(out/'models').mkdir(parents=True)
    model=out/'models/E_prior_s0_f1.pt';model.write_bytes(b'model')
    r={'contract_id':'test','holdout':['a','b'],'recipe':{'epochs':10,'total_steps':15360},'collapse_guard':{'constant_loss':1.}}
    rec=dict(status='COMPLETE',epochs=10,model_fold=1,train_studies=3073,optimizer_steps=15360,contract_id='test',model_sha256=k.sha(model))
    k.write(out/'fit-complete.json',rec);k.write(out/k.EPOCHS,dict(committed_epochs=10,log=[.4]*10))
    write_predictions(out/'holdout.csv',r['holdout'])
    k.write(out/'result.json',dict(status='COMPLETE',contract_id='test',model_fold=1,model_sha256=k.sha(model),holdout_rows=2,
                                holdout_sha256=k.sha(out/'holdout.csv'),cpu_test=False))
    for p in (out/'runtime.json',tmp_path/'registration.json',tmp_path/'prepare.json'):k.write(p,{})
    assert k.export(r)['status']=='PASS'
    assert len(k.read(tmp_path/'EXPORT.json')['sha256'])==7
    rec['optimizer_steps']-=1;k.write(out/'fit-complete.json',rec)
    with pytest.raises(ValueError):k.export(r)

def test_review_and_tamper_block(monkeypatch,tmp_path):
    monkeypatch.setattr(k,'WORK',tmp_path)
    monkeypatch.setattr(k,'validate',lambda r:r)
    monkeypatch.setattr(k,'complete_cache',lambda r,c:dict(studies=4167))
    result=k.admit({},tmp_path,tmp_path/'absent-review')
    assert result['status']=='BLOCKED' and result['dry_run']

def test_registration_tamper(monkeypatch):
    monkeypatch.setattr(k,'expected_registration',lambda:dict(contract_id='a'))
    with pytest.raises(ValueError):k.validate(dict(contract_id='b'))

def test_hash_shape_and_content():
    a=np.arange(24,dtype=np.uint8).reshape(2,3,4)
    assert cache.array_hash(a)==cache.array_hash(a.copy())
    assert cache.array_hash(a)!=cache.array_hash(a.reshape(2,6,2))
    b=a.copy();b[0,0,0]=99
    assert cache.array_hash(a)!=cache.array_hash(b)

def test_parity_rejects_payload_change(tmp_path,monkeypatch):
    r={'contract_id':'r','pins':{},'train':[],'holdout':[]};ref=tmp_path/'ref.json'
    k.write(ref,dict(contract_id='r',ids=['a','b'],renderer_pins={},studies={'a':1,'b':2}))
    monkeypatch.setattr(cache.f,'REG',ref);monkeypatch.setattr(cache.f,'validate',lambda _:r)
    monkeypatch.setattr(cache,'selected_ids',lambda _:['a','b'])
    monkeypatch.setattr(cache,'study_hashes',lambda *a:{'a':1,'b':3})
    with pytest.raises(ValueError,match='parity mismatch'):cache.parity(tmp_path,ref,True)

def test_package_inventory_rejects_drift(tmp_path):
    (tmp_path/'data').write_bytes(b'fixed');k.write(tmp_path/'INVENTORY.json',dict(files=[dict(path='data',bytes=5,sha256=k.sha(tmp_path/'data'))]))
    assert pack.verify(tmp_path)['files']==1
    (tmp_path/'data').write_bytes(b'other')
    with pytest.raises(ValueError,match='drift'):pack.verify(tmp_path)

def test_fold_state_paths_and_identity(tmp_path):
    import torch
    from types import SimpleNamespace
    from tools.l1_rco_k1_foldstate_v1 import TrainerState,Monitor
    for fold in (1,2,3):
        out=tmp_path/str(fold);model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters())
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        gen=torch.Generator().manual_seed(42);dl=SimpleNamespace(dataset=[1,2],generator=gen)
        state=TrainerState(out,model,opt,sched,None,gen,dl,['a','b'],1,Monitor(out,cpu=True),fold=fold,resume=False)
        assert state.path.name=='E_prior_s0_f%d.pt'%fold
        assert state.best_path=='ckpt/best_E_prior_s0_f%d.pt'%fold
        state.save(0,0,[])
        assert state.path.exists()
