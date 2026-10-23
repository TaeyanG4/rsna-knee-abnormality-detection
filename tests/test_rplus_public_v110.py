"""v1.1.0: retain the original contract/gate regression suite for seed44."""
from pathlib import Path
import copy
import json
import pytest
import torch
from torch import nn
from training.arms import rplus_public_v110 as c, rplus_public_fit_v110 as fit, lifecycle as life
from training.runners import rplus_public_v110 as cli


@pytest.fixture(scope='module')
def spec():
    return c.register(c.ROOT)


def test_full_roster_weights_no_holdout(spec):
    p=c.validate_inputs(spec,c.ROOT);cfg=spec.document['config']
    assert len(p['train_ids'])==4347
    assert not set(p['train_ids'])&set(p['excluded_gold_ids'])
    assert cfg['val_count']==0 and cfg['pos_weight']==p['pos_weight']
    assert cfg['total_updates']==816 and cfg['updates_per_epoch']==272
    assert cfg['recipe']['swa_1based']==[1,2,3]
    assert spec.run_kind=='member' and spec.document['inputs']['controls']=={}
    assert c.PARENT not in spec.inventory  # hash-pinned mount, no heavy staging copy
    assert spec.document['config']['scope']==c.SCOPE
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('field',['seed','arm','epoch','lr','swa','parent','source','scope','weights','bn','sampling','holdout','checkpointing','cap','augmentation'])
def test_frozen_mutations(spec,field):
    d=spec.document;cfg=d['config']
    if field=='seed': d['seed']=42
    if field=='arm': d['arm']='BASE'
    if field=='epoch': cfg['continuation']['data_epochs']=[0,1,2]
    if field=='lr': cfg['recipe']['head_lr']=.00195
    if field=='swa': cfg['recipe']['swa_1based']=[3,2,1]
    if field=='parent': cfg['continuation']['checkpoint_sha256']='0'*64
    if field=='source': d['inputs']['sources']['unregistered.py']='0'*64
    if field=='scope': cfg['scope']='HONEST_OOF'
    if field=='weights': cfg['pos_weight'][0]+=1
    if field=='bn': cfg['bn']['epoch']=20
    if field=='sampling': cfg['supervision']['oai_added']=1
    if field=='holdout': cfg['val_count']=888
    if field=='checkpointing': cfg['checkpointing']=True
    if field=='cap': cfg['wall_seconds']=100
    if field=='augmentation': cfg['factory']['augmentation']='aug1'
    with pytest.raises((ValueError,KeyError)):
        mutated=c.Spec.from_document(d);c.validate_inputs(mutated,c.ROOT)


def test_full_roster_weight_recomputation(spec):
    p=c.read(c.ROOT/c.FULL)
    original=c.scientific(p)
    p['pos_weight']=[999]*12
    assert c.scientific(p)==original # never import cached/fold weights


def test_no_cuda_before_independent_review(spec,tmp_path,monkeypatch):
    monkeypatch.setattr(fit,'observation',lambda *_:pytest.fail('CUDA observation before review'))
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:pytest.fail('CUDA probe before review'))
    for mode in ('smoke','full'):
        with pytest.raises(ValueError,match='pre-CUDA'):
            fit.run(spec,'unused','unused',tmp_path/'no-output',parent='unused',mode=mode,device='cuda')
    with pytest.raises(ValueError,match='full CPU'):
        fit.run(spec,'unused','unused',tmp_path/'no-output',parent='unused',mode='full',device='cpu')
    assert not (tmp_path/'no-output').exists() and not torch.cuda.is_initialized()


def review_fixture(spec,tmp_path,monkeypatch):
    report=tmp_path/'independent.md';report.write_text('Synthetic unit fixture only; never CUDA authority.')
    note=tmp_path/'verification.json'
    c.write(note,dict(smokes=dict(cpu=dict(exit_code=0,
        results={**dict.fromkeys(life.SMOKE_COUNTS,1),**dict.fromkeys(life.SMOKE_FLAGS,True)}))))
    monkeypatch.setattr(life,'verify_evidence',lambda *args:None)
    return dict(schema='rplus-public-pre-cuda/v1',spec_sha256=spec.sha256,scope=c.SCOPE,
        reviewer=dict(name='independent-unit-fixture',host='fixture'),independent=True,verdict='PASS',blockers=[],
        report=dict(path=report.name,sha256=life.file_sha256(report)),
        cpu_note=dict(path=note.name,sha256=life.file_sha256(note)),h112_precedence='completed-before-H113')


def test_review_validates_hashes_and_precedence(spec,tmp_path,monkeypatch):
    review=review_fixture(spec,tmp_path,monkeypatch)
    assert c.require_pre_cuda_review(spec,review,tmp_path)==life.content_sha256(review)
    for change in ('spec','self','hold','precedence','cpu','report'):
        bad=copy.deepcopy(review)
        if change=='spec': bad['spec_sha256']='0'*64
        if change=='self': bad['independent']=False
        if change=='hold': bad['verdict']='HOLD'
        if change=='precedence': bad['h112_precedence']='unknown'
        if change=='cpu': bad['cpu_note']['sha256']='0'*64
        if change=='report': bad['report']['path']='../outside.md'
        with pytest.raises(ValueError): c.require_pre_cuda_review(spec,bad,tmp_path)


def test_review_rejects_incomplete_cpu_smoke(spec,tmp_path,monkeypatch):
    review=review_fixture(spec,tmp_path,monkeypatch)
    note=c.read(tmp_path/'verification.json')
    note['smokes']['cpu']['results']['checkpoint_reloaded']=False
    (tmp_path/'verification.json').write_bytes(life.canonical_bytes(note))
    review['cpu_note']['sha256']=life.file_sha256(tmp_path/'verification.json')
    with pytest.raises(ValueError,match='real CPU'):
        c.require_pre_cuda_review(spec,review,tmp_path)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Linear(12,8);self.head=nn.Linear(8,12)
    def forward(self,x):return self.head(self.backbone(x.flatten(1)))


def test_tail_optimizer_matches_explicit_group_average(spec):
    torch.manual_seed(43);a=Tiny();b=copy.deepcopy(a)
    recipe=spec.document['config']['recipe']
    opts=[fit.make_optimization(m,recipe,816,'none') for m in (a,b)]
    assert not opts[0][0].state
    rows=[(torch.randn(1,1,3,2,2),torch.rand(1,12),torch.ones(1,12),['synthetic']) for _ in range(19)]
    result=fit.train_epoch(a,rows,*opts[0],torch.ones(12),recipe,'cpu','none')
    opt,sch,_=opts[1]
    for group in (rows[:16],rows[16:]):
        opt.zero_grad(set_to_none=True)
        torch.stack([nn.functional.binary_cross_entropy_with_logits(b(x),y) for x,y,_,_ in group]).mean().backward()
        nn.utils.clip_grad_norm_(b.parameters(),3.,error_if_nonfinite=True)
        opt.step();sch.step()
    assert result['updates']==2 and result['batches']==19 and result['skips']==0
    assert [g['lr'] for g in opts[0][0].param_groups]==[4.01e-6,1.95e-4]
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=1e-5,atol=1e-7)


def test_parent_hash_and_metadata_fail_closed(tmp_path,monkeypatch):
    path=tmp_path/'not-parent';path.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='parent bytes'):fit.load_parent(Tiny(),path)
    monkeypatch.setattr(fit.c,'sha256',lambda _:c.PUBLIC_SHA)
    monkeypatch.setattr(torch,'load',lambda *a,**k:dict(arch='wrong',res=384))
    with pytest.raises(ValueError,match='metadata'):fit.load_parent(Tiny(),path)


def test_serial_failure_cannot_publish_success(spec,tmp_path,monkeypatch):
    monkeypatch.setattr(life,'verify_evidence',lambda *args:'a'*64)
    (tmp_path/'outputs').mkdir()
    c.write(tmp_path/'outputs/summary.json',dict(status='FAILED'))
    with pytest.raises(ValueError,match='not complete'):cli.pack(spec,tmp_path,{},tmp_path)
    assert not (tmp_path/'serial-receipt.json').exists()
    assert not (tmp_path/'artifacts.tar').exists()


def test_cuda_observe_unavailable():
    with pytest.raises(SystemExit):cli.parser().parse_args(['observe','--device','cuda','--out','unused'])


def test_seed_only_and_frozen_v100(spec):
    from training.arms import rplus_public as old
    previous = old.Spec((c.ROOT/c.PREVIOUS).read_bytes())
    assert life.file_sha256(c.ROOT/c.PREVIOUS) == c.PREVIOUS_SHA
    old.validate_inputs(previous, c.ROOT)
    assert dict(spec.document['config'], seed=43) == previous.document['config']
    assert spec.document['seed'] == spec.document['config']['seed'] == 44
    assert spec.document['attempt'] == 'local4070-s44-v110'
    assert c.BASE != old.BASE
    assert set(old.SOURCES) <= set(c.SOURCES)
    for key in ('data', 'rosters', 'controls'):
        assert spec.document['inputs'][key] == previous.document['inputs'][key]


@pytest.mark.parametrize('change', ['seed43', 'attempt', 'previous'])
def test_seed44_identity_cannot_drift(spec, change):
    d=spec.document
    if change == 'seed43': d['seed']=d['config']['seed']=43
    elif change == 'attempt': d['attempt']='local4070-s43'
    else: d['inputs']['evidence'][c.PREVIOUS]='0'*64
    with pytest.raises(ValueError): c.Spec.from_document(d)


def test_fit_arithmetic_identical_to_v100():
    import ast
    old = (c.ROOT/'training/arms/rplus_public_fit.py').read_text()
    new = (c.ROOT/'training/arms/rplus_public_fit_v110.py').read_text()
    new = new.replace('rplus_public_v110 as contract', 'rplus_public as contract')
    def normalized(source):
        tree=ast.parse(source)
        tree.body=tree.body[1:]  # module documentation differs
        tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='VERSION' for t in n.targets))]
        return ast.dump(tree, include_attributes=False)
    assert normalized(old)==normalized(new)


def test_cpu_launcher_cuda_tripwire():
    assert cli.os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')
    for callback in (torch.cuda.init, torch.cuda._lazy_init):
        with pytest.raises(RuntimeError, match='CUDA forbidden'): callback()
    assert not torch.cuda.is_initialized()
