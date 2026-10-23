"""R+ 1.2.1 fold exclusion, fixed recipe, pair identity and OOF gates."""
import ast
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from training.arms import rplus_f1 as c, rplus_f1_fit as fit, rplus_fit as old_fit
from training.arms import coatnet_fold1 as parent, coatnet_fold1_fit as parent_fit
from training.arms import lifecycle as life
from tools import rplus_f1_readout as screen

ROOT=c.ROOT
SCRATCH=Path('C:/kaggle-offload/sol-1/rplus-f1-v121')


@pytest.fixture(scope='module')
def pair():
    return [c.register(SCRATCH/'engineering-root',arm=a,attempt='test') for a in ('C-L2-f1','C-BASE-f1')]


def test_parent_real_population_and_control_recipe():
    s=parent.Spec((ROOT/c.BASE/'parent-spec.json').read_bytes())
    p=parent.validate_inputs(s,ROOT)
    ordered=c.read(ROOT/'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold1/ordered-ids.json')
    assert set(p['train_ids'])==set(ordered['train']) and p['val_ids']==ordered['val']
    assert (len(p['train_ids']),len(p['val_ids']))==(3479,864)
    assert not set(p['train_ids'])&set(p['val_ids']+p['excluded_gold_ids'])
    control=c.read(ROOT/'.local/refactor/qual-4070-control-fold0-s43-8594f460d5d24d09a9f0f3f6796ab082/spec.json')['config']
    for key in ('recipe','factory','seed','checkpointing','precision','eval_precision','backend_policy','initial_state_sha256'):
        assert s.document['config'][key]==control[key]
    assert s.document['config']['total_updates']==3488
    assert s.document['config']['bn']==dict(policy='all-training-once',count=3479,epoch=16)


def test_pair_only_labels_and_budget(pair):
    screen.require_pair(*pair)
    a,b=[c.validate_inputs(s,SCRATCH/'engineering-root') for s in pair]
    assert a['train_labels']!=b['train_labels']
    assert {k:v for k,v in a.items() if k!='train_labels'}=={k:v for k,v in b.items() if k!='train_labels'}
    assert len(a['train_ids'])==3455 and len(a['val_ids'])==864
    assert not set(a['train_ids'])&set(a['val_ids']+a['excluded_gold_ids'])
    assert not a['oai_train_ids'] and set(map(tuple,a['train_masks'].values()))=={(1,)*12}
    cfg=pair[0].document['config']
    assert cfg['total_updates']==648 and cfg['updates_per_epoch']==216 and cfg['epochs']==3
    assert cfg['recipe']['swa_1based']==[1,2,3] and cfg['bn']['epoch']==19
    assert cfg['recipe']['bb_lr']==4.01e-6 and cfg['recipe']['head_lr']==1.95e-4
    assert cfg['continuation']['optimizer_state']=='fresh' and cfg['continuation']['scheduler']=='constant'


def test_label_independent_selection(pair):
    p=parent.population(parent.Spec.from_document(pair[0].document['parent_basis']),SCRATCH/'engineering-root')
    ids=c.continuation_ids(p)
    p['train_labels']={}
    assert c.continuation_ids(p)==ids
    p['train_ids']=p['train_ids'][::-1]
    assert set(c.continuation_ids(p))==set(ids)


@pytest.mark.parametrize('key',['seed','fold','engineering','budget','parent','source','swa','weights'])
def test_contract_drift_rejected(pair,key):
    d=pair[0].document
    if key=='seed':d['seed']=42
    if key=='fold':d['fold']=0
    if key=='engineering':d['engineering']=False
    if key=='budget':d['config']['total_updates']=654
    if key=='parent':d['config']['continuation']['checkpoint_sha256']='0'*64
    if key=='source':d['inputs']['sources'].pop(next(iter(d['inputs']['sources'])))
    if key=='swa':d['config']['recipe']['swa_1based']=[3,2,1]
    if key=='weights':d['config']['pos_weight'][0]+=1
    with pytest.raises(ValueError):c.validate_inputs(c.Spec.from_document(d),SCRATCH/'engineering-root')


def test_numerical_core_exact_fold0_ast():
    def functions(module):
        return {n.name:ast.dump(n) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    old,new=functions(old_fit),functions(fit)
    assert new['make_optimization']==old['make_optimization']
    assert new['train_epoch']==old['train_epoch']
    source=Path(fit.__file__).read_text()
    assert 'dataset.set_epoch(16+epoch); sampler.set_epoch(16+epoch)' in source
    assert "bn_ds.set_epoch(config['bn']['epoch'])" in source
    assert "p['val_ids'] if doc['fold'] == 1" in source
    assert "p['val_ids'] if doc['fold'] == 1" in Path(parent_fit.__file__).read_text()


def test_engineering_never_science_or_training(pair,tmp_path):
    for s in pair:
        with pytest.raises(ValueError,match='engineering parent'):
            screen.full(s,tmp_path)
        with pytest.raises(ValueError,match='engineering parent'):
            fit.run(s,'unused','unused',tmp_path/'out',mode='full',device='cuda')
    assert not torch.cuda.is_initialized()


def test_parent_wrong_fold_rejected(pair,monkeypatch):
    cfg=pair[0].document['config']
    monkeypatch.setattr(fit.c,'sha256',lambda _:cfg['continuation']['checkpoint_sha256'])
    monkeypatch.setattr(torch,'load',lambda *a,**k:dict(fold=0,seed=43,checkpoint_kind='fixed-swa-bn'))
    with pytest.raises(ValueError,match='parent metadata'):fit.load_parent(None,pair[0],ROOT)


def test_pair_drift_rejected(pair):
    d=pair[1].document;d['config']['recipe']['bb_lr']*=2
    with pytest.raises(ValueError,match='non-label pair drift'):screen.require_pair(pair[0],c.Spec.from_document(d))


def test_pooled_real_rosters_and_direction(tmp_path):
    ids0,_=screen.fold0(ROOT)
    ids1=c.read(ROOT/parent.FOLD)['val_ids']
    paths=[]
    for fold,ids in enumerate((ids0,ids1)):
        arms={}
        for index,key in enumerate(('P','L2','BASE')):
            directory=tmp_path/f'f{fold}-{key}';directory.mkdir()
            for view in ('plain','mirror','mean'):
                life.create_new(directory/f'predictions-{view}.csv',life.prediction_csv_bytes(ids,np.full((len(ids),12),.1*(index+1)),list(parent.LAB)))
            arms[key]=directory
        paths.append(arms)
    result=screen.build_screens(tmp_path/'screen',ids0,ids1,*paths)
    assert result['screens']==18 and result['rows']==1752
    from tools.multi_lens_readout import load_oof
    for scope in ('fold1','pooled'):
        d=c.read(tmp_path/f'screen/{scope}/L2-minus-BASE/spec-mean.json')
        assert d['fold_rank']==(scope=='pooled')
        ids,a,_=load_oof(d['seeds']['43']['arm'],ROOT)
        _,b,_=load_oof(d['seeds']['43']['control'],ROOT)
        assert len(ids)==(864 if scope=='fold1' else 1752)
        assert np.allclose(a,.2) and np.allclose(b,.3)
    with pytest.raises(ValueError,match='disjoint'):
        screen.build_screens(tmp_path/'invalid',ids0,ids0[:864],*paths)


def test_full_output_gate_with_explicit_fixture(pair,tmp_path):
    d=pair[0].document;d['engineering']=False
    s=c.Spec.from_document(d) # Schema fixture only, not admitted/validated.
    c.write(tmp_path/'summary.json',dict(status='COMPLETE_TRAINING_AND_READOUT',mode='full',updates=648,skips=0,
        bn=dict(studies=3455,unique_studies=3455),results=dict(validation_batches=864)))
    c.write(tmp_path/'provenance.json',dict(spec_sha256=s.sha256,scientific=d['config'],verification_sha256='f'*64))
    assert screen.full(s,tmp_path)==tmp_path


def test_original_artifact_bytes_preserved():
    life._verify_files(ROOT,c.read(ROOT/c.BASE/'frozen-before.json'))


def test_historical_sources_restored_and_fold1_runner_bound(pair):
    proof=c.read(ROOT/c.BASE/'RESTORATION.json')
    for item in proof['restored']:
        assert life.file_sha256(ROOT/item['path'])==item['restored_sha256']
        assert (ROOT/item['path']).read_bytes()==(ROOT/'.local/backups/rplus-v1.2.0'/item['path']).read_bytes()
    from training.arms import coatnet_lifecycle as historical
    for item in proof['registrations']:
        spec=historical.Spec((ROOT/item['spec']).read_bytes())
        historical.validate_inputs(spec,ROOT)
        life._verify_files(ROOT,spec.document['inputs']['sources'])
    for spec in pair:
        assert 'training/runners/coatnet_fold1.py' in spec.document['inputs']['sources']
        assert 'training/runners/coatnet_fold1.py' in spec.document['parent_basis']['inputs']['sources']


def test_real_images_sampler_and_bn_pair_identity(pair):
    from training.arms import rplus_data
    from torch.utils.data import DistributedSampler
    populations=[c.population(s,SCRATCH/'engineering-root') for s in pair]
    for augment,epochs in ((True,(16,17,18)),(False,(19,))):
        datasets=[rplus_data.Dataset(ROOT/parent.c.CORPUS,None,p,43,augment=augment) for p in populations]
        for epoch in epochs:
            samplers=[DistributedSampler(ds,1,0,shuffle=True,seed=43,drop_last=True) for ds in datasets]
            for ds,sampler in zip(datasets,samplers):
                ds.set_epoch(epoch);sampler.set_epoch(epoch)
            assert list(samplers[0])==list(samplers[1])
            left,right=[ds[0] for ds in datasets]
            assert torch.equal(left[0],right[0]) and torch.equal(left[2],right[2]) and left[3]==right[3]
    assert not torch.cuda.is_initialized()


def test_real_smoke_parent_rejected_as_full():
    spec=parent.Spec((ROOT/c.BASE/'parent-spec.json').read_bytes())
    with pytest.raises(ValueError,match='completed full fold1 parent'):
        c.check_parent(spec,SCRATCH/'engineering-root',False)


def test_full_parent_gate_explicit_fixture(monkeypatch):
    """Real roster/CSV validation with declared fixture full-run metadata."""
    spec=parent.Spec((ROOT/c.BASE/'parent-spec.json').read_bytes())
    original_read=c.read
    summary=original_read(SCRATCH/'engineering-root/parent/outputs/summary.json')
    summary.update(status='COMPLETE_TRAINING_AND_READOUT',mode='full',updates=3488,skips=0,
        bn=dict(studies=3479,unique_studies=3479),observation=original_read(ROOT/spec.document['target_source']))
    provenance=original_read(SCRATCH/'engineering-root/parent/outputs/provenance.json')
    provenance['verification_sha256']='f'*64
    def read_fixture(path):
        if Path(path).name=='summary.json':return summary
        if Path(path).name=='provenance.json':return provenance
        return original_read(path)
    monkeypatch.setattr(c,'read',read_fixture)
    loader=life.load_prediction_csv
    monkeypatch.setattr(life,'load_prediction_csv',lambda path,*args:loader(
        SCRATCH/'screens-self-prediction-fixture'/Path(path).name,*args))
    p,_=c.check_parent(spec,SCRATCH/'engineering-root',False)
    assert len(p['val_ids'])==864
    summary['updates']=3456
    with pytest.raises(ValueError,match='completed full fold1 parent'):
        c.check_parent(spec,SCRATCH/'engineering-root',False)
