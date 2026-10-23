"""H-B-106 v1.0.0 CPU allocation, RNG isolation and real-cache parity tests."""
import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from training.arms import lifecycle as life, mil, mil_coverage as cov
from tools import exp180_k12_coverage as audit

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02'


def slots():
    result=[]; start=0
    for i,(plane,_,size) in enumerate(mil.common.SLOTS64):
        result.append(dict(group=cov.GROUPS[i],start=start,stop=start+size,
                           plane=plane,series=str(i),fallback=False))
        start+=size
    return result


@pytest.mark.parametrize('seed',[42,43])
@pytest.mark.parametrize('variant',['all','fallback','absent','invalid','sparse','empty'])
def test_coverage_population_and_rng(seed,variant):
    s=slots(); mask=np.ones(64,dtype=np.uint8)
    if variant=='fallback': s[3]['fallback']=True
    if variant=='absent': s[1]['series']=None; mask[18:32]=0
    if variant=='invalid': mask[32:44:2]=0
    if variant in ('sparse','empty'):
        mask[:]=0
        if variant=='sparse': mask[0:4]=1
    for epoch in range(20):
        a=mil.common.study_rng(seed,epoch,'fixture'); b=mil.common.study_rng(seed,epoch,'fixture')
        mil.common.select_centers(mask,64,12,{'slots':mil.common.SLOTS64},'within-slot-valid',a)
        centers=cov.coverage_centres(mask,s,12,b)
        assert len(centers)==12 and a.getstate()==b.getstate()
        assert len(set(c for c in centers if c>=0))==sum(c>=0 for c in centers)
        event,_,omitted=audit.audit_bag(centers,mask,s,audit.availability(mask,s))
        assert not omitted and not event['invalid_windows'] and not event['boundary_windows']
        pool,_=cov.populations(mask,s)
        assert set(c for c in centers if c>=0)<=set(pool)
        assert centers.count(-1)==max(0,12-len(pool))
    _,g=cov.populations(mask,s)
    if variant=='fallback': assert s[3]['group'] not in dict(g)


def test_bad_geometry_and_insufficient_k_rejected():
    s=slots(); mask=np.ones(64,dtype=np.uint8)
    with pytest.raises(ValueError): cov.coverage_centres(mask,s,4,random.Random(42))
    s[1]['group']='sagittal_fluid'
    with pytest.raises(ValueError): cov.populations(mask,s)


@pytest.fixture(scope='module')
def real():
    spec=life.ArmSpec((ROOT/'training/configs/mil-coverage-s42-f0-v1.json').read_bytes())
    return spec,cov.validate_data(spec,ROOT,CORPUS)


@pytest.mark.parametrize('seed',[42,43])
def test_actual_exp185_pair_and_only_sampler_change(seed):
    spec=life.ArmSpec((ROOT/f'training/configs/mil-coverage-s{seed}-f0-v1.json').read_bytes())
    base=cov.base_spec(spec)
    original=mil.read(ROOT/f'training/configs/mil-s{seed}-f0-v3.json')
    assert base.document['config']==original['config']
    cov.validate_data(spec,ROOT,CORPUS)
    assert all('EXP179_MULTISCALE' not in p for p in spec.document['inputs']['controls'])


@pytest.mark.parametrize('section,key,value',[('windows','crop','detail'),('updates','epochs',19),
    ('sampler','fallback_credit',True),('sampler','augmentation_rng','coverage'),('model','arch','convnext_small')])
def test_recipe_mutation_rejected(real,section,key,value):
    doc=real[0].document; doc['config'][section][key]=value
    with pytest.raises(ValueError): cov.base_spec(life.ArmSpec.from_document(doc))


@pytest.mark.parametrize('seed',[42,43])
@pytest.mark.parametrize('k',[24,36])
def test_real_evaluation_unchanged(real,seed,k):
    _,(_,_,val,_,_)=real
    labels={val[0]:np.zeros(12,np.float32)}
    base=mil.MILWindows(CORPUS,val[:1],labels,k,False,seed)
    changed=cov.CoverageWindows(CORPUS,val[:1],labels,k,False,seed,metadata=ROOT/cov.METADATA)
    a,b=base[0],changed[0]
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]


@pytest.mark.parametrize('seed',[42,43])
def test_real_training_matches_manual_transform_and_epoch(real,seed):
    _,(corpus,train,_,labels,_)=real
    uid=train[0]
    ds=cov.CoverageWindows(CORPUS,[uid],labels,12,True,seed,metadata=ROOT/cov.METADATA)
    rows=mil.read(ROOT/cov.METADATA)['studies']; row=rows[uid]
    values=[]
    for epoch in (0,1):
        ds.set_epoch(epoch)
        rng=mil.common.study_rng(seed,epoch,uid)
        centers=cov.coverage_centres(corpus.mask(uid),row['slots'],12,rng)
        x=mil.scale_windows(corpus.volume(uid),centers)
        x=np.clip(x*(1+(mil.common.study_rng(seed,epoch,uid,'intensity').random()-.5)*.20),0,1)
        x=mil.augment_windows(x,centers,rng,corpus.spec)
        x=(x-np.array([.485,.456,.406],np.float32).reshape(3,1,1))/np.array([.229,.224,.225],np.float32).reshape(3,1,1)
        actual=ds[0][0]
        assert torch.equal(actual,torch.from_numpy(x))
        assert torch.equal(actual,ds[0][0])
        values.append(actual)
    assert not torch.equal(*values)


def test_real_spawn_worker_uses_metadata_and_epoch(real):
    _,(_,train,_,labels,_)=real
    ids=train[:2]
    ds=cov.CoverageWindows(CORPUS,ids,labels,12,True,42,metadata=ROOT/cov.METADATA)
    from torch.utils.data import DataLoader
    loader=DataLoader(ds,batch_size=1,num_workers=1,persistent_workers=True,collate_fn=mil.collate)
    try:
        for epoch in (0,1):
            ds.set_epoch(epoch)
            first=next(iter(loader))[0][0]
            assert torch.equal(first,ds[0][0])
    finally:
        loader._iterator._shutdown_workers()
