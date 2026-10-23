"""R+ fixed-budget continuation contracts and execution semantics (CPU only)."""
from pathlib import Path
import pytest
import torch
from torch import nn
from training.arms import rplus as c, rplus_fit as fit, rplus_data as data
from training.arms import coatnet_fit as parent_fit
from tools import rplus_readout as screen

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module',params=['C-L2','C-OAI'])
def spec(request):
    return c.register(ROOT,controls=c.read(ROOT/c.BASE/'controls.json'),arm=request.param,attempt='test-v1')


def test_parent_budget_population(spec):
    p=c.validate_inputs(spec,ROOT);d=spec.document;cfg=d['config']
    assert cfg['epochs']==3 and cfg['total_updates']==648 and cfg['recipe']['swa_1based']==[1,2,3]
    assert cfg['continuation']['optimizer_state']=='fresh' and cfg['continuation']['scheduler']=='constant'
    assert (len(p['rsna_train_ids']),len(p['val_ids']))==(3455,888)
    assert set(p['train_ids']).isdisjoint(p['val_ids']+p['excluded_gold_ids'])
    ds=data.Dataset(ROOT/c.control.c.CORPUS,ROOT/c.OAI,p,43)
    bn=data.Dataset(ROOT/c.control.c.CORPUS,ROOT/c.OAI,p,43,augment=False)
    assert len(ds)==3455 and (len(ds)+15)//16*3==648
    assert len(bn)==cfg['bn']['count']
    if d['arm']=='C-L2':
        assert not p['oai_train_ids'] and all(m==[1]*12 for m in p['train_masks'].values())
    else:
        assert len(p['oai_train_ids'])==1504
        for epoch in (16,17,18):
            selection=ds.selection(epoch)
            assert sum(ds.ids[i] in ds.rsna_ids for i in selection)==2407
            assert sum(ds.ids[i] in ds.oai_ids for i in selection)==1048
        assert not {c.o.exclusions.participant(u) for u in p['oai_train_ids']} & set(p['oai_exclusion']['participants'])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('field',['epoch','lr','swa','parent','source','arm','seed','sampling'])
def test_mutations_rejected(spec,field):
    d=spec.document
    if field=='epoch': d['config']['epochs']=4
    if field=='lr': d['config']['recipe']['bb_lr']=1e-4
    if field=='swa': d['config']['recipe']['swa_1based']=[3,2,1]
    if field=='parent': d['preset']['parent_sha256']='0'*64
    if field=='source': d['inputs']['sources']['unexpected.py']='0'*64
    if field=='arm': d['arm']='unregistered'
    if field=='seed': d['seed']=42
    if field=='sampling': d['config']['supervision']['sampling']['per_epoch']=4000
    with pytest.raises((ValueError,KeyError)): c.Spec.from_document(d)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Sequential(nn.Flatten(),nn.Linear(12,12));self.head=nn.Linear(12,12)
    def forward(self,x): return self.head(self.backbone(x))


def test_constant_lr_tail_and_masked_gradient(spec):
    torch.manual_seed(43);model=Tiny();recipe=spec.document['config']['recipe']
    opt,sch,scaler=fit.make_optimization(model,recipe,648,'none')
    assert not opt.state
    mask=torch.ones(1,12)
    if spec.document['arm']=='C-OAI': mask[:,[0,1,2,3,4,7,9,10,11]]=0
    batches=[(torch.randn(1,1,3,2,2),torch.rand(1,12),mask,['synthetic']) for _ in range(17)]
    result=fit.train_epoch(model,batches,opt,sch,scaler,torch.ones(12),recipe,'cpu','none')
    assert result['updates']==2 and result['skips']==0
    assert [g['lr'] for g in opt.param_groups]==[4.01e-6,1.95e-4]
    logits=torch.zeros(1,12,requires_grad=True)
    data.mixed.masked_study_bce(logits,torch.ones(1,12),mask,torch.ones(12)).backward()
    assert (logits.grad[mask==0]==0).all()


def test_l2_training_matches_parent_loss():
    torch.manual_seed(43);a=Tiny();b=Tiny();b.load_state_dict(a.state_dict())
    recipe=dict(bb_lr=4.01e-6,head_lr=1.95e-4,wd=.0588,clip=3)
    oa=fit.make_optimization(a,recipe,2,'none');ob=fit.make_optimization(b,recipe,2,'none')
    rows=[(torch.randn(1,1,3,2,2),torch.rand(1,12),['synthetic']) for _ in range(17)]
    fit.train_epoch(a,[(x,y,torch.ones_like(y),u) for x,y,u in rows],*oa,torch.ones(12),recipe,'cpu','none')
    parent_fit.train_epoch(b,rows,*ob,torch.ones(12),recipe,'cpu','none')
    for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],rtol=1e-5,atol=1e-7)


def test_smoke_not_scientific_and_full_cpu_rejected(spec,tmp_path):
    c.write(tmp_path/'summary.json',dict(status='ENGINEERING_SMOKE_COMPLETE'))
    c.write(tmp_path/'provenance.json',{})
    with pytest.raises(ValueError,match='full continuation'): screen.full(spec,tmp_path)
    with pytest.raises(ValueError,match='full CPU'):
        fit.run(spec,'unused','unused',tmp_path/'full',oai_corpus='unused',mode='full',device='cpu')


def test_parent_metadata_and_state_rejected(spec,monkeypatch):
    monkeypatch.setattr(fit.c,'sha256',lambda _:spec.document['preset']['parent_sha256'])
    monkeypatch.setattr(fit.torch,'load',lambda *a,**k:dict(fold=1,seed=43,checkpoint_kind='fixed-swa-bn'))
    with pytest.raises(ValueError,match='parent metadata'): fit.load_parent(Tiny(),spec,ROOT)


def test_paired_readout_real_csv_fixture(spec,tmp_path):
    # Real parent predictions with synthetic continuation metadata: plumbing only.
    import shutil
    def fixture(s,path):
        path.mkdir()
        c.write(path/'summary.json',dict(status='COMPLETE_TRAINING_AND_READOUT',mode='full',updates=648,skips=0))
        c.write(path/'provenance.json',dict(spec_sha256=s.sha256,scientific=s.document['config'],verification_sha256='f'*64))
        for mode in ('mean','plain','mirror'):
            shutil.copyfile(ROOT/c.o.CONTROL/f'predictions-{mode}.csv',path/f'predictions-{mode}.csv')
    arm=tmp_path/'arm';fixture(spec,arm)
    control=None;control_out=None
    if spec.document['arm']=='C-OAI':
        with pytest.raises(ValueError,match='C-L2 control'): screen.prepare(spec,ROOT,tmp_path/'bad',arm)
        control=c.register(ROOT,controls=c.read(ROOT/c.BASE/'controls.json'),arm='C-L2',attempt='test-v1')
        control_out=tmp_path/'control';fixture(control,control_out)
    result=screen.prepare(spec,ROOT,tmp_path/'screen',arm,control,control_out)
    assert result['rows']==888
    for mode in ('mean','plain','mirror'):
        assert (tmp_path/f'screen/arm-{mode}.csv').read_bytes().split(b'\n',1)[1]==(arm/f'predictions-{mode}.csv').read_bytes().split(b'\n',1)[1]
