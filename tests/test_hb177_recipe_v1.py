"""H-B-177 recipe and admission regressions; all CPU, no label output."""
import copy
import math
import os
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training/src'))
from hb177_recipe_v1 import RECIPE, layer_id, lr_factor, set_lr, optimization, EMA, microbatch_step, public_decision
from tools import hb177_build_v1 as b


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Module()
        self.backbone.stem=torch.nn.Linear(3,3)
        self.head=torch.nn.Sequential(torch.nn.LayerNorm(3),torch.nn.Linear(3,12))
    def forward(self,x):
        return self.head(self.backbone.stem(x))


def test_official_small_layer_map():
    assert layer_id('stem.0.weight')==0
    assert layer_id('stages.0.blocks.2.gamma')==1
    assert layer_id('stages.1.downsample.0.weight')==2
    assert layer_id('stages.2.downsample.0.weight')==3
    assert [layer_id(f'stages.2.blocks.{i}.gamma') for i in range(27)]==[3+i//3 for i in range(27)]
    assert layer_id('stages.3.blocks.2.gamma')==12
    assert layer_id('head.norm.weight')==13
    with pytest.raises(ValueError): layer_id('unknown.weight')


def test_groups_cover_exactly_once_and_actual_lr():
    model=Tiny(); opt=optimization(model)
    params=[p for g in opt.param_groups for p in g['params']]
    assert len(params)==len(set(map(id,params)))==len(list(model.parameters()))
    for g in opt.param_groups:
        for name,p in zip(g['names'],g['params']):
            assert g['weight_decay']==(0 if p.ndim<=1 else .02)
            expected=.00025*.8**13 if name.startswith('backbone.') else .001
            assert g['lr']==pytest.approx(expected)
    set_lr(opt,0,5440,544)
    assert all(g['lr']==pytest.approx(g['peak_lr']*.01) for g in opt.param_groups)


def test_schedule_and_decisions():
    assert lr_factor(0,5440,544)==.01
    assert lr_factor(544,5440,544)==1
    assert lr_factor(5439,5440,544)==pytest.approx(.01)
    assert lr_factor(543,5440,544)<1 and lr_factor(545,5440,544)<1
    with pytest.raises(ValueError): lr_factor(5440,5440,544)
    assert public_decision(.950)=='ALIVE'
    assert public_decision(.946)=='DEAD'
    assert public_decision(.948)=='HOLD'
    for score in (float('nan'),float('inf'),-1,1.1):
        with pytest.raises(ValueError): public_decision(score)


def test_accumulation_partial_and_ema_resume(tmp_path):
    torch.manual_seed(42)
    a=Tiny(); z=copy.deepcopy(a)
    oa,oz=optimization(a),optimization(z)
    ea,ez=EMA(a),EMA(z)
    x=torch.randn(3,3); y=torch.rand(3,12)
    criterion=torch.nn.BCEWithLogitsLoss()
    microbatch_step(a,oa,ea,[(x[i:i+1],y[i:i+1],[]) for i in range(3)],criterion,'cpu','none')
    microbatch_step(z,oz,ez,[(x,y,[])],criterion,'cpu','none')
    for p,q in zip(a.parameters(),z.parameters()): torch.testing.assert_close(p,q)
    # Actual serialization, optimizer/EMA continuation must reproduce uninterrupted run.
    b.save_torch(tmp_path/'resume.pt',dict(model=a.state_dict(),opt=oa.state_dict(),ema=ea.state_dict()))
    loaded=torch.load(tmp_path/'resume.pt',weights_only=True)
    resumed=Tiny(); resumed.load_state_dict(loaded['model']); ro=optimization(resumed); ro.load_state_dict(loaded['opt'])
    re=EMA(resumed); re.load_state_dict(loaded['ema'])
    for model,opt,ema in ((a,oa,ea),(resumed,ro,re)):
        microbatch_step(model,opt,ema,[(x,y,[])],criterion,'cpu','none')
    for p,q in zip(ea.module.parameters(),re.module.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert re.updates==ea.updates==2


def smoke():
    return dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_mib=16376,desktop_used_mib=1200,
        train_updates=3,train_shape=[1,12,3,384,384],accumulation=16,serving_views=2,serving_k=94,
        precision='bf16',grad_checkpointing=True,train_peak_reserved_bytes=11*1024**3,
        serve_peak_reserved_bytes=11*1024**3,seconds_per_update=5.)


def test_memory_both_caps_and_shape():
    b.check_memory(smoke())
    for mutation in (dict(train_peak_reserved_bytes=12.01*1024**3),dict(desktop_used_mib=4000),
                     dict(serving_views=1),dict(accumulation=8),dict(train_updates=0),
                     dict(grad_checkpointing=False),dict(seconds_per_update=float('nan')),
                     dict(device_name='4090')):
        with pytest.raises(ValueError): b.check_memory(dict(smoke(),**mutation))


def test_review_identity_binding_and_bypass(tmp_path,monkeypatch):
    bindings={'protocol_sha256':'a','cpu_sha256':'b'}
    review=dict(verdict='PASS',independent=True,reviewer='different-lane',scope='pre-execution',**bindings)
    b.check_review(review,bindings,'pre-execution')
    for mutation in (dict(reviewer=b.BUILDER),dict(independent=False),dict(verdict='FAIL'),
                     dict(protocol_sha256='stale'),dict(scope='other')):
        with pytest.raises(ValueError): b.check_review(dict(review,**mutation),bindings,'pre-execution')
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'registration',lambda:None)
    monkeypatch.setattr(b,'require_cpu',lambda:None)
    with pytest.raises(FileNotFoundError): b.require_admission()
    called=[]
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**k:called.append('GPU'))
    with pytest.raises(FileNotFoundError): b.supervise('fit')
    assert not called


def test_desktop_parser_and_watchdog_no_wall_cap():
    text='123, C:\\Windows\\explorer.exe\n456, C:\\Python312\\python.exe\n789, C:\\chrome.exe\n'
    assert b.python_blockers(text)==[456]
    assert b.python_blockers(text,[456])==[]
    from tools.hb150_watchdog_v1 import Policy
    p=Policy(0,total=None)
    assert p.total is None


def test_cpu_guard_does_not_initialize_cuda():
    b.cpu_guard()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()


def test_b_authorization_and_result_receipt_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'PROTOCOL',tmp_path/'registration.json')
    monkeypatch.setattr(b,'registration',lambda:None)
    monkeypatch.setattr(b,'require_cpu',lambda:None)
    b.atomic_json(b.PROTOCOL,{'test':True})
    b.atomic_json(tmp_path/'cpu-validation.json',{'test':True})
    pre=dict(verdict='PASS',independent=True,reviewer='independent-test-lane',
             scope='pre-execution',**b.review_bindings())
    b.atomic_json(tmp_path/'pre-review.json',pre)
    authorization=dict(authorized_by='B',protocol_sha256=b.sha(b.PROTOCOL),
        pre_review_sha256=b.sha(tmp_path/'pre-review.json'),allow_smoke=True)
    b.atomic_json(tmp_path/'b-authorization.json',authorization)
    b.require_admission()
    for change in ({'allow_smoke':False},{'authorized_by':'builder'},{'pre_review_sha256':'old'}):
        b.atomic_json(tmp_path/'b-authorization.json',dict(authorization,**change))
        with pytest.raises(ValueError): b.require_admission()
    b.atomic_json(tmp_path/'cuda-smoke.json',dict(smoke(),protocol_sha256=b.sha(b.PROTOCOL)))
    receipt=tmp_path/'smoke-attempt/receipt.json'
    b.atomic_json(receipt,dict(exit_code=0,reason='completed',survivors=[]))
    review=dict(verdict='PASS',independent=True,reviewer='independent-test-lane',
                scope='smoke-result-and-fit',**b.review_bindings(True))
    b.atomic_json(tmp_path/'fit-review.json',review)
    authorization.update(allow_fit=True,fit_review_sha256=b.sha(tmp_path/'fit-review.json'))
    b.atomic_json(tmp_path/'b-authorization.json',authorization)
    b.require_admission(True)
    b.atomic_json(receipt,dict(exit_code=0,reason='completed',survivors=[123]))
    with pytest.raises(ValueError,match='supervision'): b.require_admission(True)


def test_registration_source_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'ROOT',tmp_path)
    monkeypatch.setattr(b,'PROTOCOL',tmp_path/'registration.json')
    monkeypatch.setattr(b,'PARENT_MANIFEST',tmp_path/'parent.json')
    monkeypatch.setattr(b,'SOURCES',['trainer.py'])
    monkeypatch.setattr(b,'serving_inventory',lambda:{'frozen':'hash'})
    base=dict(train_ids=['synthetic'],train_labels={'synthetic':[.5]*12},pos_weight=[1]*12)
    monkeypatch.setattr(b,'parent',lambda:base)
    b.atomic_json(b.PARENT_MANIFEST,{'synthetic':True})
    (tmp_path/'trainer.py').write_text('version=1',encoding='utf-8')
    b.atomic_json(b.PROTOCOL,dict(schema='hb177-registration/v1',recipe=RECIPE,
        source_pins={'trainer.py':b.sha(tmp_path/'trainer.py')},serving_files={'frozen':'hash'},
        parent_manifest_sha256=b.sha(b.PARENT_MANIFEST),train_ids_sha256=b.digest(base['train_ids']),
        labels_sha256=b.digest(base['train_labels']),pos_weight_sha256=b.digest(base['pos_weight'])))
    b.registration()
    (tmp_path/'trainer.py').write_text('version=2',encoding='utf-8')
    with pytest.raises(ValueError,match='source drift'): b.registration()
