"""Changed scientific path and fail-closed execution gates; CPU only."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import patch_mil as g
from training.adapters import dinov2_patch_mil as adapter


def fake_model():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_prefix_tokens=1;self.num_features=384;self.global_pool='token';self.dynamic_img_size=True
            self.patch_embed=SimpleNamespace(patch_size=(14,14))
            self.fc_norm=torch.nn.Identity();self.head=torch.nn.Identity();self.head_drop=torch.nn.Dropout(0)
            self.tokens=torch.nn.Parameter(torch.arange(257*384,dtype=torch.float32).reshape(1,257,384))
        def forward_features(self,x):return self.tokens.expand(x.shape[0],-1,-1)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.backbone=Backbone()
        def encode(self,x):return None
    return Model()


def test_prefix_excluded_and_patch_gradient_retained_without_rng_or_class_mutation():
    model=fake_model();other=fake_model();original=type(model).encode
    param=model.backbone.tokens;rng=torch.get_rng_state().clone()
    adapter.install(model)
    assert model.backbone.tokens is param and torch.equal(rng,torch.get_rng_state())
    assert type(model).encode is original and other.encode(torch.zeros(1)) is None
    features=model.encode(torch.zeros(2,3,3,224,224))
    assert features.shape==(2,768,384)
    assert torch.equal(features[0,:256],param[0,1:])
    features.sum().backward()
    assert torch.equal(param.grad[:,0],torch.zeros(1,384)) and torch.all(param.grad[:,1:]==6)
    with pytest.raises(ValueError):adapter.install(model)


@pytest.mark.parametrize('bad',['prefix','norm','head_dropout','resolution'])
def test_rejects_second_changes_to_backbone_policy(bad):
    model=fake_model()
    if bad=='prefix':model.backbone.num_prefix_tokens=2
    if bad=='norm':model.backbone.fc_norm=torch.nn.LayerNorm(384)
    if bad=='head_dropout':model.backbone.head_drop.p=.1
    if bad=='resolution':
        adapter.install(model)
        with pytest.raises(ValueError):model.encode(torch.zeros(1,1,3,112,112))
    else:
        with pytest.raises(ValueError):adapter.install(model)


def pilot():
    times=[.2]*32;steady=sum(times[8:])/24
    return dict(schema=g.SCHEMA,status='PASS',phase='pilot',updates=32,skips=0,training_ids_only=True,
        validation_rows=0,eval_train_studies=4,aggregation_policy=g.AGGREGATION_POLICY,position_policy=g.POSITION_POLICY,
        step_seconds=times,steady_update_seconds=steady,projected_full_seconds=1.3*(steady*3448)+300,
        peak_gpu_allocated_bytes=2**30)


@pytest.mark.parametrize('bad',[None,'fabricated_projection','slow','skip','val','memory','elapsed'])
def test_pilot_cost_and_numerical_gate(bad):
    p=pilot();elapsed=100
    if bad=='fabricated_projection':p['projected_full_seconds']=100
    if bad=='slow':
        p['step_seconds']=[1.0]*32;p['steady_update_seconds']=1;p['projected_full_seconds']=1.3*3448+300
    if bad=='skip':p['skips']=1
    if bad=='val':p['validation_rows']=1
    if bad=='memory':p['peak_gpu_allocated_bytes']=13*2**30
    if bad=='elapsed':elapsed=181
    if bad:
        with pytest.raises(ValueError):g.pilot_cost_gate(p,elapsed)
    else:g.pilot_cost_gate(p,elapsed)


def test_full_result_requires_inventory_trace_and_identity(tmp_path,monkeypatch):
    path=tmp_path/'manifest.json';path.write_text('{}')
    p={'arm':{'initial_state_sha256':'initial'},'expected_cls_training_trace_sha256':'trace'}
    r=dict(schema=g.SCHEMA,status='PASS',phase='full',protocol_sha256=g.sha256(path),argv=['runner'],
           aggregation_policy=g.AGGREGATION_POLICY,position_policy=g.POSITION_POLICY,
           initial_asset_sha256=g.WEIGHT_SHA,initial_state_sha256='initial',updates=3448,skips=0,
           validation_rows=888,training_trace_sha256='trace',outputs={})
    monkeypatch.setattr(g.base,'validate_completion',lambda *a:None)
    for n in ['summary.json','provenance.json','history.json','trace.json','model_best.pt','oof_best.csv']:
        (tmp_path/n).write_bytes(b'bound')
        r['outputs'][n]={'bytes':5,'sha256':g.sha256(tmp_path/n)}
    g.validate_result(r,p,path,'full',['runner'],tmp_path)
    for key,value in [('outputs',{}),('training_trace_sha256','wrong'),('skips',1),('aggregation_policy','cls')]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):g.validate_result(bad,p,path,'full',['runner'],tmp_path)
    (tmp_path/'model_best.pt').write_bytes(b'other')
    with pytest.raises(ValueError):g.validate_result(r,p,path,'full',['runner'],tmp_path)


@pytest.mark.parametrize('failure',[None,'pilot','resource'])
def test_outer_never_runs_full_after_invalid_pilot_and_never_retries(tmp_path,monkeypatch,failure):
    import psutil
    from tools import process_resource_receipt
    path=tmp_path/'manifest.json';path.write_text('{}')
    monkeypatch.setattr(g,'ROOT',tmp_path)
    monkeypatch.setattr(g,'validate_protocol',lambda *a,**k:({},{}))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=16*2**30))
    monkeypatch.setattr(g,'phase_command',lambda p,path,phase:[phase])
    monkeypatch.setattr(g,'validate_result',lambda *a:None)
    monkeypatch.setattr(g.base,'resource_gate',lambda *a:None)
    calls=[]
    def supervise(command,out,limit,root):
        calls.append(command[0]);out.mkdir(parents=True)
        if failure=='resource':raise TimeoutError('owned child stopped by supervisor')
        phase=command[0];run_dir=out.with_name(out.name.removesuffix('_RESOURCE'));run_dir.mkdir()
        data=pilot() if phase=='pilot' else {'status':'PASS'}
        if failure=='pilot':data['skips']=1
        (run_dir/'receipt.json').write_text(json.dumps(data))
        (out/'receipt.json').write_text('{}')
        return {'elapsed_seconds':1}
    monkeypatch.setattr(process_resource_receipt,'supervise',supervise)
    assert g.run(path)==(1 if failure else 0)
    assert calls==(['pilot'] if failure else ['pilot','full'])
    with pytest.raises(FileExistsError):g.run(path)
    assert calls==(['pilot'] if failure else ['pilot','full'])
