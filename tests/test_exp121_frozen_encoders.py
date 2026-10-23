import numpy as np
import torch
from tools import exp121_frozen_encoders as e

def test_cpu_head_never_calls_cuda(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('CUDA forbidden in CPU test')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',forbidden)
    torch.random.default_generator.manual_seed(42)
    model=e.make_head(8).to(e.device_context('cpu'))
    x=torch.randn(2,36,8,device='cpu'); out=model(x)
    assert out.shape==(2,12)
    out.sum().backward()
    assert model.weight.grad is not None
    assert torch.allclose(model.eval()(x),model(x.flip(1)),atol=1e-6)

def test_normalizations():
    x=torch.zeros(1,3,2,2,device='cpu')
    assert torch.all(e.normalize(x,'medsiglip')==-1)
    for name,cfg in e.ENCODERS.items():
        y=e.normalize(x,name)
        assert np.allclose(y[:, :, 0, 0].numpy(),-np.array(cfg['mean'])/np.array(cfg['std']))

def test_queue_is_extraction_only_and_continues():
    jobs=e.queue_entries('CURRENT_TAIL')
    assert len(jobs)==3 and jobs[0]['after']=='CURRENT_TAIL'
    assert all(j['continue_on_failure'] for j in jobs)
    assert all('extract' in j['command'] and j['command'][-1]=='cuda' for j in jobs)
    assert jobs[1]['after']==jobs[0]['tag']

def test_component_bootstrap_and_consensus_mask():
    rng=np.random.default_rng(0); n=24
    y=np.tile(np.arange(n)[:,None]%2,(1,12)).astype(float)
    consensus=y.copy(); consensus[:2]=np.nan
    s=y+rng.normal(0,.1,(n,12))
    result=e.comparison(list(map(str,range(n))),s,s,s,y,consensus,np.arange(n)//2,reps=10)
    assert result['unit']=='component_id'
    assert result['results']['dread']['delta']==0
    assert result['results']['consensus']['valid_draws']==10

def test_mri_teacher_mapping_and_fail_closed():
    import pytest
    target={'pos_embed':torch.zeros(1,2,2,3),
            'blocks.1.mlp.lin1.weight':torch.zeros(4,3),'neck.0.weight':torch.zeros(2,3,1,1)}
    teacher={'backbone.pos_embed':torch.ones(1,10,3),
             'backbone.blocks.0.1.mlp.fc1.weight':torch.ones(4,3),
             'backbone.cls_token':torch.zeros(1,1,3)}
    mapped,report=e.map_mri_teacher(teacher,target)
    assert mapped['pos_embed'].shape==(1,2,2,3)
    assert torch.all(mapped['pos_embed']==1)
    assert report['initialized_unused']==['neck.0.weight']
    assert 'backbone.cls_token' in report['unmapped']
    teacher.pop('backbone.blocks.0.1.mlp.fc1.weight')
    with pytest.raises(ValueError,match='unmapped MRI trunk keys'):
        e.map_mri_teacher(teacher,target)
