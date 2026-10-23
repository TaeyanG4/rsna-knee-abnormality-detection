"""Fold1 confirmation: metadata-only lineage and synthetic decision tests."""
from copy import deepcopy
import json
import numpy as np
import pytest
from tools import hb147_confirm_build_v1 as b
from tools import hb147_confirm_readout_v1 as r


def test_recipe_changes_only_fold_roster_and_derived_updates():
    for res in (224,320):
        for seed in (42,43):
            c=b.validate(b.config_path(res,seed))
            p=b.read(b.PILOT/(b.member(res,seed)+'.json'))
            old,new=deepcopy(p['config']),deepcopy(c['recipe'])
            for k in ('ordered','groups','n_train','n_val','train_sha256','val_sha256','order_policy'):
                old['roots'].pop(k,None); new['roots'].pop(k,None)
            for k in ('loader_batches','steps','max_skips'):
                old['updates'].pop(k); new['updates'].pop(k)
            assert old==new
            assert c['fold']==1 and p['fold']==0 and c['workers']==2
            assert (len(c['train']),len(c['val']),c['optimizer_steps'])==(3479,864,8680)
            assert 'non-PL L2' in c['label_mode']


def test_gate_same_three_rulers():
    mean=np.full((4,12),.001); seeds=np.stack([mean,mean])
    assert r.gate(seeds,mean)['GO']
    mean[3]=0
    assert r.gate(seeds,mean)['GO']
    mean[2]=0
    assert not r.gate(seeds,mean)['GO']
    mean[:]=.002
    mean[0,r.lens._base.WEAK]=0
    mean[1,r.lens._base.OTHER]=-.001
    assert not r.gate(seeds,mean)['GO']


@pytest.mark.parametrize('target,bad',[(8,-.009),(3,-.005001)])
@pytest.mark.parametrize('where',['seed','mean'])
def test_pilot_harm_and_any_cell_block_go(target,bad,where):
    mean=np.full((4,12),.02); seeds=np.stack([mean,mean])
    array=seeds[0] if where=='seed' else mean
    array[0,target]=bad
    assert not r.gate(seeds,mean)['GO']
    array[0,target]=-.005
    assert r.gate(seeds,mean)['GO']


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_invalid_and_engineering_never_go(bad):
    mean=np.full((4,12),.02); seeds=np.stack([mean,mean])
    assert not r.gate(seeds,mean,engineering=True)['GO']
    assert not r.gate(seeds[:1],mean)['GO']
    seeds[0,0,0]=bad
    assert not r.gate(seeds,mean)['GO']


def test_auc_backend_and_descriptive_gold():
    ids=[f'synthetic-{i}' for i in range(8)]
    y=np.tile((np.arange(8)%2)[:,None],(1,12)).astype(float)
    components=[{'component_id':u,'members':[u]} for u in ids]
    refs={n:(y,np.ones_like(y,dtype=bool),1) for n in r.lens.RULERS}
    ctx=r.lens._base.Context(ids,refs,components)
    good,flat=.2+.6*y,np.full(y.shape,.5)
    images={'gold58':dict(ids=ids,labels=y,mask=np.isfinite(y),components=components,
        arm=good,control=flat,eligible=True)}
    out=r.evaluate(ctx,{'42':good,'43':good},{'42':flat,'43':flat},images)
    assert out['registered_go']['GO']
    assert not out['lenses']['L7']['entries']['gold58']['eligible']
    assert images['gold58']['eligible']


def test_independent_review_missing_and_stale_binding_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'validate',lambda p:{})
    with pytest.raises(ValueError,match='independent review'):
        b.require_fit('synthetic',None)
    path=tmp_path/'review.json'
    path.write_text(json.dumps(dict(schema='hb147-confirm-review/v1',verdict='PASS',
        independent=True,reviewer='test',bindings={'stale':'hash'})))
    monkeypatch.setattr(b,'review_bindings',lambda:{'current':'hash'})
    with pytest.raises(ValueError,match='bindings'):
        b.require_fit('synthetic',path)


def test_python_only_admission():
    desktop='1, C:\\Windows\\explorer.exe\n2, chrome.exe\n3, NVIDIA Overlay.exe'
    assert b.python_processes(desktop)==[]
    assert b.python_processes(desktop+'\n4, C:\\Python312\\python.exe')[0]['pid']=='4'


@pytest.mark.parametrize('fault',['memory','shape','gpu','config','cpu_cuda','steps'])
def test_smoke_binding_rejects_memory_shape_and_identity_faults(monkeypatch,fault):
    monkeypatch.setattr(b,'prepare',lambda:None)
    monkeypatch.setattr(b.Path,'is_file',lambda self:True)
    monkeypatch.setattr(b,'sha',lambda p:'current')
    def smoke(path):
        device='cpu' if 'cpu' in path.parts else 'cuda'
        s=dict(passed=True,smoke=True,engineering_only=True,config_sha256='current',device=device,
            optimizer_steps=1 if device=='cpu' else 2,cpu_cuda_initialized=False,
            gpu_name=b.GPU,full_shape=True,peak_reserved_bytes=8*(1<<30),batch=8,val_batch=4,
            validation_shapes=['k24','k36','k36flip'],healthy_rates=dict(train_step=1,validation=1))
        if fault=='memory': s['peak_reserved_bytes']=11*(1<<30)
        elif fault=='shape': s['validation_shapes']=['k24']
        elif fault=='gpu': s['gpu_name']='NVIDIA T4'
        elif fault=='config': s['config_sha256']='stale'
        elif fault=='cpu_cuda': s['cpu_cuda_initialized']=True
        else: s['optimizer_steps']=0
        return s
    monkeypatch.setattr(b,'read',smoke)
    with pytest.raises(ValueError): b.review_bindings()


def test_old_or_partial_receipts_rejected(monkeypatch):
    c=b.read(b.config_path(224,42))
    identity=dict(experiment='H-B-147 fold1 confirmation',fold=1,arm=c['arm'],seed=42,
        resolution=224,config_sha256='config',registration_sha256='config',
        train_sha256=b.uid_hash(c['train']),val_sha256=b.uid_hash(c['val']),workers=2,
        epochs=20,smoke=False,engineering_only=False,gpu_name=b.GPU,weights=c['weights'],
        model=c['recipe']['model'],labels_sha256=c['inputs'][c['label_path']],prepare_sha256='prepared',
        runtime=dict(policy='strict',precision_fallback_allowed=False,device='cuda'))
    summary=dict(identity=identity,passed=True,smoke=False,device='cuda',steps=8680,
        counters=dict(updates=8680,skips=0),checkpoint_sha256='checkpoint')
    receipt=dict(identity=identity,k=36,tta=True,engineering_only=False,rows=864,
        uid_sha256=b.uid_hash(c['val']),csv_sha256='csv',checkpoint_sha256='checkpoint')
    monkeypatch.setattr(b,'sha',lambda p:'prepared')
    args=dict(config_hash='config',csv_hash='csv',checkpoint_hash='checkpoint')
    assert r.validate_records(c,summary,receipt,**args)['fold']==1
    identity['fold']=0
    with pytest.raises(ValueError,match='old/foreign'):
        r.validate_records(c,summary,receipt,**args)
    identity['fold']=1; summary['steps']-=1
    with pytest.raises(ValueError,match='partial'):
        r.validate_records(c,summary,receipt,**args)
