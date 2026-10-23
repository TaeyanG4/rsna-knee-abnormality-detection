"""R+ v1.1 labels-only amendment: real populations and exact fitter parity."""
from pathlib import Path
import ast
import copy
import shutil
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DistributedSampler
from training.arms import rplus_base as c, rplus as old, rplus_data as data
from training.arms import rplus_base_fit as fit
from tools import rplus_base_readout as screen

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def spec():
    return c.register(ROOT,controls=c.read(ROOT/c.BASE/'controls.json'),arm='C-BASE',attempt='local4070-s43-v1')


def test_same_pinned_s6_baseline():
    # Inspect declarations without importing the concurrently built S6 runner.
    source=ast.parse((ROOT/'tools/s6plus.py').read_text(encoding='utf-8'))
    nodes={n.targets[0].id:n.value for n in source.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('DEFAULT','PINS')}
    default={ast.literal_eval(k):v for k,v in zip(nodes['DEFAULT'].keys,nodes['DEFAULT'].values)}
    inputs={ast.literal_eval(k):v for k,v in zip(default['inputs'].keys,default['inputs'].values)}
    assert ast.literal_eval(inputs['dread'])==c.LABELS
    assert ast.literal_eval(default['arms'])['S-B']==dict(labels='dread',distill=False)
    assert ast.literal_eval(nodes['PINS'])['dread']==c.LABEL_SHA
    exp099=c.read(ROOT/'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/label-sets/dread.json')
    assert (exp099['path'],exp099['sha256'])==(c.LABELS,c.LABEL_SHA)
    assert c.life.file_sha256(ROOT/c.LABELS)==c.LABEL_SHA


def test_labels_only_population_and_config(spec):
    l2=old.Spec.from_document(spec.document['l2_basis'])
    a=c.validate_inputs(spec,ROOT);b=old.population(l2,ROOT)
    assert len(a['train_ids'])==3455 and len(a['val_ids'])==888
    assert not a['oai_train_ids'] and set(a['train_ids']).isdisjoint(a['val_ids']+a['excluded_gold_ids'])
    assert a['train_labels']!=b['train_labels']
    assert {k:v for k,v in a.items() if k!='train_labels'}=={k:v for k,v in b.items() if k!='train_labels'}
    cfg=copy.deepcopy(spec.document['config']); prior=l2.document['config']
    cfg['recipe']['labels']=prior['recipe']['labels'];cfg['supervision']['kind']=prior['supervision']['kind']
    assert cfg==prior
    assert cfg['epochs']==3 and cfg['total_updates']==648 and cfg['recipe']['swa_1based']==[1,2,3]
    assert cfg['continuation']['optimizer_state']=='fresh' and cfg['continuation']['scheduler']=='constant'
    assert cfg['bn']['count']==3455
    existing=old.Spec((ROOT/'.local/rplus-v1/C-L2/spec.json').read_bytes())
    screen.require_pair(spec,existing)


def test_real_images_order_masks_and_bn_match(spec):
    a=c.population(spec,ROOT);b=old.population(old.Spec.from_document(spec.document['l2_basis']),ROOT)
    for augment in (True,False):
        datasets=[data.Dataset(ROOT/c.control.c.CORPUS,ROOT/c.OAI,p,43,augment=augment) for p in (a,b)]
        for epoch in ((16,17,18) if augment else (19,)):
            for ds in datasets: ds.set_epoch(epoch)
            samplers=[DistributedSampler(ds,1,0,shuffle=True,seed=43,drop_last=True) for ds in datasets]
            for sampler in samplers: sampler.set_epoch(epoch)
            assert list(samplers[0])==list(samplers[1])
            samples=[ds[0] for ds in datasets]
            assert torch.equal(samples[0][0],samples[1][0])
            assert torch.equal(samples[0][2],samples[1][2]) and samples[0][3]==samples[1][3]
    assert not torch.cuda.is_initialized()


def test_fitter_exact_v1_logic():
    old_source=(ROOT/'training/arms/rplus_fit.py').read_text()
    new_source=(ROOT/'training/arms/rplus_base_fit.py').read_text()
    normalized=new_source.replace('v1.1.0','v1.0.0').replace("VERSION = '1.1.0'","VERSION = '1.0.0'").replace('rplus_base as contract','rplus as contract')
    assert ast.dump(ast.parse(normalized))==ast.dump(ast.parse(old_source))


@pytest.mark.parametrize('change',['epochs','weights','seed','order','parent','labels','source','arm','controls'])
def test_contract_mutations_rejected(spec,change):
    d=spec.document
    if change=='epochs': d['config']['epochs']=4
    if change=='weights': d['config']['recipe']['wd']=0
    if change=='seed': d['seed']=42
    if change=='order': d['config']['continuation']['epoch_offset']=0
    if change=='parent': d['preset']['parent_sha256']='0'*64
    if change=='labels': d['inputs']['data'][c.LABELS]='0'*64
    if change=='source': d['inputs']['sources']['training/arms/rplus.py']='0'*64
    if change=='arm': d['arm']='C-L2'
    if change=='controls': d['inputs']['controls']={}
    with pytest.raises(ValueError): c.Spec.from_document(d)


@pytest.mark.parametrize('bad',['missing','duplicate','nan','out_of_range'])
def test_bad_labels_rejected(tmp_path,bad):
    targets=[str(i) for i in range(12)]
    frame=pd.DataFrame(np.zeros((2,12)),columns=targets);frame['StudyInstanceUID']=['a','b']
    if bad=='missing': frame=frame.iloc[:1]
    if bad=='duplicate': frame['StudyInstanceUID']=['a','a']
    if bad=='nan': frame.loc[0,'0']=np.nan
    if bad=='out_of_range': frame.loc[0,'0']=1.1
    path=tmp_path/'synthetic.csv';frame.to_csv(path,index=False)
    with pytest.raises(ValueError): c.load_labels(path,['a','b'],targets)


def test_label_file_drift_rejected(spec,monkeypatch):
    monkeypatch.setattr(old,'population',lambda *args: {})
    monkeypatch.setattr(c.life,'file_sha256',lambda *args:'0'*64)
    with pytest.raises(ValueError,match='baseline label bytes'): c.population(spec,ROOT)


def test_full_cpu_and_smoke_science_rejected(spec,tmp_path):
    with pytest.raises(ValueError,match='full CPU'):
        fit.run(spec,'unused','unused',tmp_path/'full',oai_corpus='unused',mode='full',device='cpu')
    c.write(tmp_path/'summary.json',dict(status='ENGINEERING_SMOKE_COMPLETE'))
    c.write(tmp_path/'provenance.json',{})
    with pytest.raises(ValueError,match='full continuation'): screen.full(spec,tmp_path)


def test_pairing_and_real_csv_direction(spec,tmp_path):
    l2=old.Spec.from_document(spec.document['l2_basis'])
    with pytest.raises(ValueError,match='embedded continuation'): screen.require_pair(spec,None)
    other=old.register(ROOT,controls=c.read(ROOT/c.BASE/'controls.json'),arm='C-OAI',attempt='local4070-s43-v1')
    with pytest.raises(ValueError,match='embedded continuation'): screen.require_pair(spec,other)
    # Parent prediction CSVs with explicit fixture metadata test the adapter,
    # not the scientific outcome of unrun continuations.
    for s,name in ((spec,'baseline'),(l2,'l2')):
        path=tmp_path/name;path.mkdir()
        c.write(path/'summary.json',dict(status='COMPLETE_TRAINING_AND_READOUT',mode='full',updates=648,skips=0))
        c.write(path/'provenance.json',dict(spec_sha256=s.sha256,scientific=s.document['config'],verification_sha256='f'*64))
        for mode in ('mean','plain','mirror'):
            shutil.copyfile(ROOT/c.o.CONTROL/f'predictions-{mode}.csv',path/f'predictions-{mode}.csv')
            if name=='l2':
                csv=path/f'predictions-{mode}.csv'
                frame=pd.read_csv(csv,dtype={'StudyInstanceUID':str})
                frame.iloc[:,1:]=1-frame.iloc[:,1:]
                frame.to_csv(csv,index=False)
    result=screen.prepare(spec,ROOT,tmp_path/'screen',tmp_path/'baseline',l2,tmp_path/'l2')
    assert result['rows']==888
    for sub,name in (('', 'C-L2-minus-C-BASE'),('base-minus-parent','C-BASE-minus-P')):
        for mode in ('mean','plain','mirror'):
            d=c.read(tmp_path/'screen'/sub/f'spec-{mode}.json')
            assert d['name']==name+'-'+mode and d['label_change'] is True
            converted=pd.read_csv(tmp_path/'screen'/sub/f'arm-{mode}.csv')
            expected=pd.read_csv(tmp_path/('l2' if not sub else 'baseline')/f'predictions-{mode}.csv')
            np.testing.assert_allclose(converted.iloc[:,1:],expected.iloc[:,1:],rtol=1e-6,atol=1e-7)


def test_frozen_v1_bytes_unchanged():
    manifest=ROOT/old.BASE/'SHA256SUMS.txt'
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1)
        if name.startswith((old.BASE+'/', 'training/arms/rplus', 'training/runners/rplus', 'tools/rplus', 'tests/test_rplus')) or name.endswith(('/C-L2/spec.json','/C-OAI/spec.json')):
            assert c.life.file_sha256(ROOT/name)==digest, name
    for name in ('C-L2.json','C-OAI.json','controls.json'):
        assert (ROOT/c.BASE/name).read_bytes()==(ROOT/old.BASE/name).read_bytes()
