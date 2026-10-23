import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from tools import vnext_r7_pl_train_v110 as t
from training.arms import vnext_r7_pl_coat_v110 as coat
from pl_readout_fixture_v111 import make_readout


def seal(d):
    d['spec_sha256']=t.p.digest({k:v for k,v in d.items() if k!='spec_sha256'})
    return d


def export_admission(tmp_path,spec,target):
    path=make_readout(tmp_path/'export-readout',target['arm'])
    pin=t.asset(path)
    return dict(status='ADMITTED',spec_sha256=spec['spec_sha256'],go=pin,
                decision=t.scientific_go.validate(pin,target['arm']))


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    member=dict(id='PL-S-s42-fall',kind='MIL',arch=t.p.MIL_ARCH,seed=42,fold='all')
    reg=dict(arm='global',members=[member])
    target=dict(train_ids=[str(i) for i in range(16)],val_ids=[],weights=[2.]*12,arm='global',
        seed=42,fold='all',mode='all-data',engineering_only=False,teacher_sha256='a'*64,target_sha256='b'*64,
        train_ids_sha256='c'*64,l2_pos_weight_sha256='d'*64,registration_sha256=t.p.digest(reg))
    labels={u:np.full(12,.5,np.float32) for u in target['train_ids']}
    pin=tmp_path/'target-spec.json';t.p.dump(pin,{'synthetic':True})
    spec=seal(dict(schema='pl-production-trainer-v1.1.1',version=t.VERSION,workers=0,
        stage=str(tmp_path),registration_sha256=t.p.digest(reg),member=member,
        target_bundle=t.asset(pin),inputs=[],source_pins=[],recipe=t.mil_recipe(16),corpus='unused'))
    path=tmp_path/'training-spec.json';t.p.dump(path,spec)
    monkeypatch.setattr(t.p,'registration',lambda _:reg)
    monkeypatch.setattr(t,'target_loader',lambda _:(target,labels))
    return path,spec,target,labels


def test_complete_mil_recipe_copied_with_only_roster_changes():
    actual=t.mil_recipe(40)
    base=t.p.read(t.ROOT/t.CONTROL)['config']
    for key in ('windows','sampler','backend_policy','model','bn','readout','targets'):
        assert actual[key]==base[key]
    assert actual['updates']['amp_init_scale']==1024.
    assert actual['updates']['steps']==100
    assert actual['updates']['max_skips']==1


def test_validated_all_data_has_no_evaluation(prepared):
    path,spec,target,_=prepared
    assert t.validate(path)[1]['val_ids']==[]
    assert t.admit(path)['status']=='HOLD'


@pytest.mark.parametrize('kind',['weights','windows','backend_policy'])
def test_recipe_tamper_rejected_even_resealed(prepared,kind):
    path,spec,_,_=prepared
    if kind=='weights':spec['recipe']['updates']['bb_lr']=.9
    elif kind=='windows':spec['recipe']['windows']['train_k']=2
    else:spec['recipe']['backend_policy']['cudnn_benchmark']=True
    t.p.dump(path,seal(spec))
    with pytest.raises(ValueError,match='scientific recipe'):t.validate(path)


def test_validation_target_leakage_rejected(prepared):
    path,_,target,labels=prepared
    target['val_ids']=['validation'];labels['validation']=np.zeros(12,np.float32)
    with pytest.raises(ValueError,match='training-only'):t.validate(path)


def test_wrong_member_seed_rejected(prepared):
    path,_,target,_=prepared;target['seed']=43
    with pytest.raises(ValueError,match='member targets'):t.validate(path)


@pytest.mark.parametrize('change',[{'status':'HOLD'},{'winning_arm':'fill'},{'clean_oof':True},{'exposure':'CLEAN'}])
def test_nonmatching_go_rejected(prepared,tmp_path,change):
    path,*_=prepared
    go=dict(status='GO',winning_arm='global',clean_oof=False,exposure='EXPOSED_EXPLORATORY');go.update(change)
    f=tmp_path/'go.json';t.p.dump(f,go)
    with pytest.raises(ValueError,match='scientific GO'):t.admit(path,f)


def test_positive_admission(prepared,tmp_path):
    path,*_=prepared
    f=make_readout(tmp_path/'readout')
    assert t.admit(path,f)['status']=='ADMITTED'


@pytest.mark.parametrize('arm',['global','fill'])
def test_actual_readout_to_admission_export_binding(prepared,tmp_path,monkeypatch,arm):
    path,spec,target,_=prepared
    reg=t.p.registration('unused')
    reg['arm']=arm
    target.update(arm=arm,silence_mask_sha256='f'*64,registration_sha256=t.p.digest(reg))
    spec['registration_sha256']=t.p.digest(reg)
    t.p.dump(path,seal(spec))
    teacher=tmp_path/'teacher.json'
    t.p.dump(teacher,dict(producer='H-B-145',producer_version='1.1.2',producer_exposure='EXPOSED',
        status='PASS',role='heldout',output_kind='pre_stacker',model_fold='int(sha1(uid)[:8],16)%4',
        complete=True,failures=0,clean_oof=False,exposure='EXPOSED_EXPLORATORY',
        producer_identity='synthetic',csv_sha256='a'*64))
    target['teacher_sha256']=t.p.sha(teacher)
    readout=make_readout(tmp_path/'screen',arm)
    admission=t.admit(path,readout)
    exported=t.export(spec,target,torch.nn.Linear(3,12),tmp_path/'export',smoke=False,
                      counters={'updates':40},runtime={'fixture':'CPU'},go=admission)
    entry=dict(spec['member'],**t.asset(exported['checkpoint']),
               training_receipt=t.asset(tmp_path/'export/training-receipt.json'))
    bound=dict(registration_sha256=t.p.digest(reg),arm=arm,members=[entry],
               teacher=t.asset(teacher),go=t.asset(readout))
    inputs=tmp_path/'inputs.json';t.p.dump(inputs,bound)
    assert t.p.bind('unused',inputs,tmp_path/'bound.json')['status']=='BOUND'
    assert t.p.validate_bindings(reg,t.p.read(tmp_path/'bound.json'))==[entry]
    # Another valid source may select the same arm, but cannot authorize an
    # already trained checkpoint whose receipt names a different source.
    doc=t.p.read(readout);doc['notes'].append('other readout identity')
    other=tmp_path/'other.json';t.p.dump(other,doc)
    bound['go']=t.asset(other)
    with pytest.raises(ValueError,match='source differs'):t.p.validate_bindings(reg,bound)


@pytest.mark.parametrize('arm',['global','fill'])
@pytest.mark.parametrize('kind',['MIL','CoAt'])
def test_engineering_bundle_cannot_be_admitted(prepared,tmp_path,monkeypatch,arm,kind):
    path,spec,target,labels=prepared
    target.update(engineering_only=True,arm=arm)
    spec['member']['kind']=kind
    monkeypatch.setattr(t,'validate',lambda _:(spec,target,labels))
    f=make_readout(tmp_path/'readout',arm)
    with pytest.raises(ValueError,match='engineering targets'):t.admit(path,f)


def test_optimizer_shared_with_screen_and_scaler1024(monkeypatch):
    from training.arms import mil_screen_hooks_v1 as h
    from tools import hb146_screen_v111 as screen
    assert h.optimizer_stack is screen.h.optimizer_stack
    assert h.train_epoch_with is screen.h.train_epoch_with
    model=torch.nn.Module();model.backbone=torch.nn.Linear(3,4);model.head=torch.nn.Linear(4,12)
    seen={}
    original=torch.amp.GradScaler
    def scaler(*args,**kwargs):
        seen.update(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(torch.amp,'GradScaler',scaler)
    opt,sched,_=h.optimizer_stack(model,100,'cpu','none')
    assert seen['init_scale']==1024.
    assert opt.defaults['weight_decay']==.02
    assert [g['max_lr'] for g in opt.param_groups]==[.0005,.001]


def test_full_cpu_forbidden_before_validation(monkeypatch):
    monkeypatch.setattr(t,'validate',lambda _:pytest.fail('must reject before file access'))
    with pytest.raises(ValueError,match='requires CUDA'):t.fit('unused','unused',device='cpu',smoke=False)


def test_screen_dataset_hook_is_same_targets_and_train_only():
    from training.arms.hb146_pl_v1 import Arm
    from tools.hb146_screen_v111 import Arm as ScreenArm
    class Base:
        k=12;ids=['study']
        def __getitem__(self,index):return torch.ones(2),torch.zeros(12),'study'
        def __len__(self):return 1
        def set_epoch(self,epoch):pass
    labels={'study':np.linspace(0,1,12,dtype=np.float32)}
    a=Arm(labels); b=ScreenArm(labels);base=Base()
    assert type(a) is type(b)
    assert torch.equal(a.dataset(base,train=True)[0][1],b.dataset(base,train=True)[0][1])
    assert a.dataset(base,train=False) is base


@pytest.mark.parametrize('smoke',[True,False])
def test_export_metadata_serving_and_receipt(prepared,tmp_path,smoke):
    _,spec,target,_=prepared
    model=torch.nn.Linear(3,12)
    out=tmp_path/'export'
    go=export_admission(tmp_path,spec,target)
    result=t.export(spec,target,model,out,smoke=smoke,counters={'updates':40,'skips':0},runtime={},go=go)
    ck=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    receipt=t.p.read(out/'training-receipt.json')
    from tools.vnext_r7_pl_worker_v1 import checkpoint_metadata
    checkpoint_metadata(ck,spec['member'],receipt)
    assert receipt['status']==('ENGINEERING_SMOKE' if smoke else 'COMPLETE')
    assert ck['pl_provenance']==t.provenance(spec,target)
    if not smoke:
        from tools.hb105_serving_v101 import meta
        from tools import exp169_release_v6 as serving
        meta(ck,dict(spec['member'],spec_sha256=spec['spec_sha256']),serving)
    with pytest.raises(FileExistsError):
        t.export(spec,target,model,out,smoke=smoke,counters={},runtime={},go=go)


def test_fill_export_binds_mask(prepared,tmp_path):
    _,spec,target,_=prepared
    target.update(arm='fill',silence_mask_sha256='f'*64)
    result=t.export(spec,target,torch.nn.Linear(3,12),tmp_path/'fill',smoke=False,counters={},runtime={},
                    go=export_admission(tmp_path,spec,target))
    ck=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['pl_provenance']['silence_mask_sha256']=='f'*64


@pytest.mark.parametrize('arm',['global','fill'])
@pytest.mark.parametrize('kind',['MIL','CoAt'])
def test_engineering_cannot_export_production(prepared,tmp_path,arm,kind):
    _,spec,target,_=prepared
    target.update(engineering_only=True,arm=arm)
    spec['member']['kind']=kind
    with pytest.raises(ValueError,match='engineering targets'):
        t.export(spec,target,None,tmp_path/'blocked',smoke=False,counters={},runtime={},
                 go=export_admission(tmp_path,spec,target))
    assert not (tmp_path/'blocked').exists()


@pytest.mark.parametrize('kind',['absent','wrong-spec','changed-decision'])
def test_export_requires_live_matching_admission(prepared,tmp_path,kind):
    _,spec,target,_=prepared
    go=export_admission(tmp_path,spec,target)
    if kind=='absent':go=None
    elif kind=='wrong-spec':go['spec_sha256']='0'*64
    else:
        t.p.dump(go['go']['path'],{'status':'HOLD'})
    with pytest.raises(ValueError):
        t.export(spec,target,None,tmp_path/'blocked',smoke=False,counters={},runtime={},go=go)
    assert not (tmp_path/'blocked').exists()


def test_coat_hook_reuses_frozen_recipe_and_restores(tmp_path):
    from tools import exp143_v2 as guard
    output=t.ROOT/'.local/scratch/astra-b1/pl-prod-prep/train/test-coat'
    init=tmp_path/'init';init.write_bytes(b'init')
    spec=dict(recipe=copy.deepcopy(guard.RECIPE),corpus=str(tmp_path),inputs=[{},t.asset(init)])
    target=dict(fold='all',train_ids=['a','b'],val_ids=[],weights=[2.]*12)
    labels={u:np.full(12,.3,np.float32) for u in target['train_ids']}
    old=(guard.load_registration,guard.require_validation,guard.registered_run)
    calls=[]
    with coat.target_hook(spec,target,labels,output,validate=lambda:calls.append('validated')) as doc:
        assert guard.load_registration('unused',tmp_path,init) is doc
        assert doc['train_labels'] is labels
        assert guard.registered_run(doc)==output
        settings=guard.execution_settings(doc,1,'fp16',42,True)
        assert settings['accumulation']==16
        assert doc['recipe']['swa_1based']==[14,11,12]
        assert settings['grad_checkpointing'] is True
    assert calls==['validated']
    assert (guard.load_registration,guard.require_validation,guard.registered_run)==old


def test_coat_hook_forbids_validation_bn_members(tmp_path):
    from tools.exp143_v2 import RECIPE
    with pytest.raises(ValueError,match='training-only'):
        coat.protocol({'recipe':RECIPE},{'fold':'all','train_ids':['a'],'val_ids':['b']},
                      {'a':np.zeros(12)},tmp_path)
