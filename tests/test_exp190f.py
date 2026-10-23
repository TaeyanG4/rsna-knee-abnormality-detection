"""EXP190F v1.0.0: real contract, masking, sampling, update/resume parity."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from training.arms import exp190f as c, exp190f_data as data, exp190f_fit as fit
from training.arms import coatnet_fit as control_fit, lifecycle as life
from tools import exp190f_readout as screen

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def spec():
    return c.register(ROOT,controls=c.read(ROOT/c.BASE/'controls.json'),arm='EXP190F-test',attempt='v1')


def test_actual_paired_recipe_and_exclusions(spec):
    p=c.validate_inputs(spec,ROOT)
    reference=c.read(ROOT/c.CONTROL/'provenance.json')['scientific']
    assert c.projection(spec.document).document['config']==reference
    assert spec.run_kind=='experiment'
    assert (len(p['train_ids']),len(p['rsna_train_ids']),len(p['oai_train_ids']))==(4959,3455,1504)
    assert not {c.exclusions.participant(u) for u in p['oai_train_ids']} & set(p['oai_exclusion']['participants'])
    original=c.control.population(c.projection(spec.document),ROOT)
    assert {u:p['train_labels'][u] for u in p['rsna_train_ids']}==original['train_labels']
    assert p['pos_weight']==original['pos_weight']
    assert spec.document['config']['recipe']['swa_1based']==[14,11,12]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('change',['seed','fold','variant','checkpointing','recipe','bn','control','cache','supervision','source'])
def test_mutations_rejected(spec,change):
    d=spec.document
    if change=='seed': d['seed']=42
    if change=='fold': d['fold']=1
    if change=='variant': d['config']['variant']='crop'
    if change=='checkpointing': d['config']['checkpointing']=True
    if change=='recipe': d['config']['recipe']['swa_1based']=[11,12,14]
    if change=='bn': d['config']['bn']['count']=5630
    if change=='control': d['inputs']['controls'][c.CONTROL+'summary.json']='0'*64
    if change=='cache': d['inputs']['data'][c.OAI+'/manifest.json']='0'*64
    if change=='supervision': d['config']['supervision']['sampling']['rsna']=2408
    if change=='source': d['inputs']['sources'].pop('training/arms/exp190f_data.py')
    with pytest.raises(ValueError): c.validate_inputs(c.Spec.from_document(d),ROOT)


@pytest.mark.parametrize('change',['leak','mask','unobserved','count','panel'])
def test_cohort_semantics_rejected(spec,monkeypatch,change):
    cohort=c.read(ROOT/c.COHORT)
    uid=cohort['ids'][0]
    if change=='leak':
        old=uid; uid='OAI_'+cohort['exclusion']['participants'][0]+'_L_V00'
        cohort['ids'][0]=uid
        for key in ('labels','masks'): cohort[key][uid]=cohort[key].pop(old)
    if change=='mask': cohort['masks'][uid][0]=1
    if change=='unobserved': cohort['labels'][uid][0]=1
    if change=='count': cohort['ids'].pop()
    if change=='panel': cohort['exclusion']['participants'][0]='0000000'
    original=c.read
    monkeypatch.setattr(c,'read',lambda path: cohort if Path(path)==ROOT/c.COHORT else original(path))
    with pytest.raises(ValueError): c.population(spec,ROOT)


def test_fixed_sampling_and_unaugmented_bn(spec):
    p=c.population(spec,ROOT)
    ds=data.Dataset(ROOT/c.control.c.CORPUS,ROOT/c.OAI,p,43)
    bn=data.Dataset(ROOT/c.control.c.CORPUS,ROOT/c.OAI,p,43,augment=False)
    selected=[]
    for epoch in range(16):
        indexes=ds.selection(epoch)
        assert len(indexes)==len(set(indexes))==3455
        assert sum(ds.ids[i] in ds.rsna_ids for i in indexes)==2407
        assert sum(ds.ids[i] in ds.oai_ids for i in indexes)==1048
        assert bn.selection(epoch)==tuple(range(4959))
        selected.append(indexes)
    assert len(set(selected))==16


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone=nn.Sequential(nn.Flatten(),nn.Linear(12,12))
        self.head=nn.Linear(12,12)
    def forward(self,x): return self.head(self.backbone(x))


def stack(initial):
    model=Tiny();model.load_state_dict(initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,total_steps=12)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    return model,optimizer,scheduler,scaler


def test_all_rsna_loop_matches_control_and_tail():
    torch.manual_seed(43); initial=Tiny().state_dict()
    mixed=stack(initial); control=stack(initial)
    batches=[(torch.randn(1,1,3,2,2),torch.rand(1,12),[str(i)]) for i in range(17)]
    mb=[(x,y,torch.ones_like(y),uids) for x,y,uids in batches]
    a=fit.train_epoch(*mixed[:1],mb,*mixed[1:],torch.ones(12),{'clip':3},'cpu','none')
    b=control_fit.train_epoch(*control[:1],batches,*control[1:],torch.ones(12),{'clip':3},'cpu','none')
    assert a['updates']==b['updates']==2 and a['skips']==b['skips']==0
    for k,v in mixed[0].state_dict().items(): torch.testing.assert_close(v,control[0].state_dict()[k],rtol=1e-5,atol=1e-7)


def test_unknown_gradient_and_denominator():
    logits=torch.zeros(2,12,requires_grad=True); labels=torch.zeros_like(logits)
    mask=torch.zeros_like(logits);mask[:,[5,6,8]]=1
    loss=data.mixed.masked_study_bce(logits,labels,mask,torch.ones(12))
    torch.testing.assert_close(loss,torch.tensor(np.log(2)/4,dtype=torch.float32))
    loss.backward()
    assert (logits.grad[mask==0]==0).all() and (logits.grad[mask==1]!=0).all()


def test_header_adapter_preserves_every_probability_byte(tmp_path):
    ids=['synthetic-study']; raw=life.prediction_csv_bytes(ids,np.full((1,12),.123456789),c.control.LAB)
    source=tmp_path/'source.csv';source.write_bytes(raw)
    target=tmp_path/'target.csv'; screen.convert(source,target,ids)
    assert target.read_bytes().split(b'\n',1)[1]==raw.split(b'\n',1)[1]
    life.load_prediction_csv(target,ids,c.control.LAB,id_column='StudyInstanceUID')
    with pytest.raises(ValueError): screen.convert(source,tmp_path/'bad.csv',['different'])


def test_full_cpu_and_missing_controls_rejected(spec,tmp_path):
    with pytest.raises(ValueError,match='genuine paired'):
        c.register(ROOT,controls=None,arm='bad',attempt='v1')
    with pytest.raises(ValueError,match='full CPU'):
        fit.run(spec,'unused','unused',tmp_path/'bad',oai_corpus='unused',mode='full',device='cpu')


def test_screen_rejects_smoke(spec,tmp_path):
    c.write(tmp_path/'summary.json',dict(status='ENGINEERING_SMOKE_COMPLETE'))
    c.write(tmp_path/'provenance.json',{})
    with pytest.raises(ValueError,match='full arm output'):
        screen.prepare(spec,ROOT,tmp_path/'screen',tmp_path)


def test_full_readout_adapter_contract_fixture(spec,tmp_path):
    # Synthetic qualification metadata only: exercise the production adapter
    # branch with actual CONTROL CSV bytes; never a scientific arm result.
    import shutil
    outputs=tmp_path/'fixture-only'; outputs.mkdir()
    c.write(outputs/'summary.json',dict(status='COMPLETE_TRAINING_AND_READOUT',
        mode='full',updates=3456,skips=0))
    c.write(outputs/'provenance.json',dict(spec_sha256=spec.sha256,
        scientific=spec.document['config'],verification_sha256='f'*64))
    for mode in ('mean','plain','mirror'):
        shutil.copyfile(ROOT/c.CONTROL/f'predictions-{mode}.csv',outputs/f'predictions-{mode}.csv')
    result=screen.prepare(spec,ROOT,tmp_path/'screen',outputs)
    assert result==dict(status='READOUT_SPEC_PREPARED',scope='PAIRED_FOLD0_S43',rows=888)
    for mode in ('mean','plain','mirror'):
        output=c.read(tmp_path/f'screen/spec-{mode}.json')
        assert output['family']=='coatnet' and output['label_change']
        assert set(output['seeds'])=={'43'} and 'image_reads' not in output
        assert (tmp_path/f'screen/arm-{mode}.csv').read_bytes().split(b'\n',1)[1]==(outputs/f'predictions-{mode}.csv').read_bytes().split(b'\n',1)[1]
