import numpy as np
import pytest
from tools import hb155_readout_v1 as r

def test_all_four_rulers_and_seed_sign_required():
    a=np.full((2,4,12),.003); m=np.full((4,12),.003)
    assert r.gate(a,m)['GO']
    m[3]=.001999; assert not r.gate(a,m)['GO']
    m[:]=.003; a[1,2]=0; assert not r.gate(a,m)['GO']

def test_harm_floor_and_weak_four():
    a=np.full((2,4,12),.01); m=np.full((4,12),.01)
    a[0,0,0]=-.003001; assert not r.gate(a,m)['GO']
    a[:]=.01; m[0,0]=-.003001; assert not r.gate(a,m)['GO']
    m[:]=.01; a[1,:,r.lens._base.WEAK]=-.0001
    assert not r.gate(a,m)['GO']

def test_finite_shapes_engineering():
    a=np.full((2,4,12),.003); m=np.full((4,12),.003)
    assert not r.gate(a,m,engineering=True)['GO']
    a[0,0,0]=np.nan; assert r.gate(a,m)['status']=='INVALID_EVIDENCE'
    assert not r.gate(np.zeros((1,4,12)),m)['GO']

def records():
    c=dict(fold=2,arm='crop',seed=42,train=['train'],val=['val'],weights=[1]*12,
        recipe={'model':{'name':'model'}},corpus_contract='crop',label_path='labels',inputs={'labels':'L'},optimizer_steps=20,
        exclusions_sha256='E',side_unknown_policy='metadata-only')
    i=dict(experiment='H-B-155 compartment crop paired pilot',fold=2,arm='crop',seed=42,
        resolution=224,config_sha256='C',registration_sha256='C',train_sha256=r.b.uid_hash(c['train']),
        val_sha256=r.b.uid_hash(c['val']),workers=2,epochs=20,smoke=False,engineering_only=False,
        gpu_name=r.b.GPU,weights=c['weights'],model=c['recipe']['model'],batch=8,microbatch=2,accumulation=4,
        corpus_contract='crop',prepare_sha256='P',runtime_snapshot_sha256='R',labels_sha256='L',
        exclusions_sha256='E',side_unknown_policy='metadata-only',
        runtime=dict(policy='strict',precision_fallback_allowed=False,device='cuda'))
    s=dict(identity=i,passed=True,smoke=False,device='cuda',steps=20,counters=dict(updates=20,skips=0),checkpoint_sha256='K')
    p=dict(identity=i,k=36,tta=True,engineering_only=False,rows=1,uid_sha256=r.b.uid_hash(c['val']),csv_sha256='V',checkpoint_sha256='K')
    kw=dict(config_hash='C',csv_hash='V',checkpoint_hash='K',prepare_hash='P',runtime_hash='R')
    return c,s,p,kw

def test_valid_records_and_old_baseline_rejected():
    c,s,p,kw=records(); r.validate_records(c,s,p,**kw)
    s['identity']['experiment']='old baseline'
    with pytest.raises(ValueError,match='foreign'): r.validate_records(c,s,p,**kw)

@pytest.mark.parametrize('field,value',[('steps',19),('passed',False),('smoke',True)])
def test_partial_receipts_rejected(field,value):
    c,s,p,kw=records(); s[field]=value
    with pytest.raises(ValueError,match='partial'): r.validate_records(c,s,p,**kw)

def test_k36_flip_required():
    c,s,p,kw=records(); p['tta']=False
    with pytest.raises(ValueError,match='K36'): r.validate_records(c,s,p,**kw)
