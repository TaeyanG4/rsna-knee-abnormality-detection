"""K1 CPU fail-closed admission, data isolation and durable-state regression tests."""
import ast
import copy
import csv
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pytest
from tools import l1_rco_k1_v1 as k
from tools.l1_rco_k1_state_v1 import Monitor, TrainerState, ThroughputGuard


def _assert_k0_evidence_mutation_rejected(tmp_path, monkeypatch, path, key, message):
    reg = k.read(k.WORK / 'registration.json')
    k.validate(reg)  # Control: current authenticated inputs remain valid.
    actual_sha = k.sha
    assert actual_sha(path) == reg[key]
    monkeypatch.setattr(k, 'sha', lambda p: '0' * 64 if Path(p) == path else actual_sha(p))
    assert k.sha(path) != reg[key]
    with pytest.raises(ValueError, match=message):
        k.validate(reg)
    # The same mismatch must block admission before review/VRAM evidence use.
    monkeypatch.setattr(k, 'WORK', tmp_path)
    monkeypatch.setattr(k, 'complete_cache', lambda r, c: dict(studies=4167))
    result = k.admit(reg, tmp_path, tmp_path / 'review.json')
    assert result['status'] == 'BLOCKED' and result['blockers'] == [message]
    assert actual_sha(path) == reg[key]  # Fault injection never altered the receipt.


def test_registered_k0_result_mutation_rejected_before_admission(tmp_path, monkeypatch):
    _assert_k0_evidence_mutation_rejected(tmp_path, monkeypatch, k.K0_RESULT,
                                        'K0_result_sha256', 'registered K0 result changed')


def test_registered_k0_review_mutation_rejected_before_admission(tmp_path, monkeypatch):
    _assert_k0_evidence_mutation_rejected(tmp_path, monkeypatch, k.K0_REVIEW,
                                        'K0_review_sha256', 'registered K0 review changed')


@pytest.mark.parametrize('evidence',['admission.json','holdout.csv','ckpt/E_prior_s0_f0.pt'])
@pytest.mark.parametrize('registration_present',[True,False])
def test_post_candidate_admission_or_output_registration_change_refused(tmp_path,monkeypatch,evidence,registration_present):
    old=k.read(k.WORK/'registration.json'); old['version']='old-version'
    monkeypatch.setattr(k,'WORK',tmp_path); monkeypatch.setattr(k,'OUT',tmp_path/'fit')
    (k.OUT/evidence).parent.mkdir(parents=True,exist_ok=True)
    (k.OUT/evidence).write_text('existing production evidence')
    if registration_present: k.write(tmp_path/'registration.json',old)
    with pytest.raises(ValueError,match='immutable|cannot change'): k.register()


def test_derivative_preserves_model_data_loss_and_single_fold_math():
    source=(k.ROOT/k.s.SOURCE/'compact_train.py').read_text(encoding='utf-8')
    new=k.derive_trainer(source)
    before,after=ast.parse(source),ast.parse(new)
    for name in ('Net','Studies','collate','bce_soft','lr_loss','augment'):
        nodes=lambda tree:[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
        assert ast.dump(nodes(before)[0])==ast.dump(nodes(after)[0])
    assert 'assert not gold and all(pool_fold != 0)' in new
    assert 'check_training_ids(pool)' in new
    assert 'bias0 = torch.logit(col(pool, "prior", float("nan")).mean(0)' in new
    assert 'assert a.folds == 4 and a.only_folds == "0" and a.holdout_fold is None' in new
    assert 'l1_state.finish(model)\n            return' in new
    assert 'z_pool[torch.from_numpy(ho)' not in new[new.index('    def run(arm, seed):'):]
    with pytest.raises(ValueError): k.derive_trainer(source.replace('                ds.epoch = ep','                ds.epoch = -1'))


def test_production_targets_contain_only_train_and_no_bias_holdout():
    full=k.read(k.ROOT/k.s.REG)
    reg=dict(train=sorted(full['train']),holdout=sorted(full['val']))
    with (k.ROOT/k.s.TARGETS).open(newline='',encoding='utf-8') as stream:
        rows=k.training_rows(reg,csv.DictReader(stream))
    assert len(rows)==3190 and {r[k.s.UID] for r in rows}==set(reg['train'])
    assert not {r[k.s.UID] for r in rows}&set(reg['holdout'])
    # Poison every numeric holdout field. They must never be parsed or emitted.
    poison=[{k.s.UID:u,**{label:'NOT_A_NUMBER' for label in k.s.LABELS}} for u in reg['holdout']]
    with (k.ROOT/k.s.TARGETS).open(newline='',encoding='utf-8') as stream:
        retained=[r for r in csv.DictReader(stream) if r[k.s.UID] in set(reg['train'])]
    assert k.training_rows(reg,retained+poison)==rows
    bad=copy.deepcopy(reg); bad['train'][0]=reg['holdout'][0]
    with pytest.raises(ValueError): k.training_rows(bad,retained+poison)


def test_training_exact_roster_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(k,'WORK',tmp_path)
    reg=dict(train=['a','b'],holdout=['c']); k.write(tmp_path/'registration.json',reg)
    k.check_training_ids(['b','a'])
    for ids in (['a'],['a','b','c'],['a','b','b']):
        with pytest.raises(ValueError): k.check_training_ids(ids)


def test_incomplete_cache_refuses_before_manifest_read(tmp_path):
    k.write(tmp_path/'status.json',dict(phase='assemble',done=4167,total=4167))
    with pytest.raises(ValueError,match='incomplete'): k.complete_cache({},tmp_path)
    k.write(tmp_path/'status.json',dict(phase='complete',done=4166,total=4167))
    with pytest.raises(ValueError,match='incomplete'): k.complete_cache({},tmp_path)


def test_complete_cache_receipt_and_manifest_checks(tmp_path,monkeypatch):
    ids=[str(i) for i in range(4167)]; reg=dict(train=ids[:3190],holdout=ids[3190:],fullcache_contract_id='sealed')
    index=dict(contract_id='sealed',order=ids,cursor=4167,errors={},
               done={u:[i,1,1] for i,u in enumerate(ids)},
               blocks={u:[dict(views=[[0,i]])] for i,u in enumerate(ids)})
    k.write(tmp_path/'status.json',dict(phase='complete',done=4167,total=4167))
    k.write(tmp_path/'index.json',index)
    k.write(tmp_path/'contract.json',dict(contract_id='sealed',ids=ids,size=384,views='v4'))
    for name in ('px.u8.npy','meta.f32.npy'): (tmp_path/name).write_bytes(b'unit fixture; mmap metadata mocked')
    hashes={name:k.sha(tmp_path/name) for name in ('px.u8.npy','meta.f32.npy','index.json')}
    receipt=dict(version='1.1.0',contract_id='sealed',studies=4167,ids=ids,hashes=hashes,cuda_initialized=False,
                 bytes=sum((tmp_path/name).stat().st_size for name in hashes))
    k.write(tmp_path/'receipt.json',receipt)
    def fake_load(path,**kwargs):
        return SimpleNamespace(dtype=np.uint8,shape=(4167,384,384)) if Path(path).name=='px.u8.npy' else SimpleNamespace(dtype=np.float32,shape=(4167,12))
    monkeypatch.setattr(np,'load',fake_load)
    assert k.complete_cache(reg,tmp_path)['studies']==4167
    receipt['ids']=ids[:-1]+[ids[0]]; k.write(tmp_path/'receipt.json',receipt)
    with pytest.raises(ValueError,match='receipt roster'): k.complete_cache(reg,tmp_path)
    receipt['ids']=ids; k.write(tmp_path/'receipt.json',receipt)
    index['done'][ids[-1]][0]=0; k.write(tmp_path/'index.json',index)
    with pytest.raises(ValueError,match='manifest hash'): k.complete_cache(reg,tmp_path)
    receipt['hashes']['index.json']=k.sha(tmp_path/'index.json'); receipt['bytes']=sum((tmp_path/name).stat().st_size for name in hashes)
    k.write(tmp_path/'receipt.json',receipt)
    with pytest.raises(ValueError,match='contiguous'): k.complete_cache(reg,tmp_path)


def test_review_admission_cannot_self_approve(tmp_path,monkeypatch):
    monkeypatch.setattr(k,'WORK',tmp_path)
    monkeypatch.setattr(k,'validate',lambda r:r)
    monkeypatch.setattr(k,'review_binding',lambda r:dict(contract_id='c'))
    monkeypatch.setattr(k,'complete_cache',lambda r,c:dict(studies=4167))
    k.write(tmp_path/'prepare.json',dict(argv=k.trainer_argv(tmp_path),cache=str(tmp_path.resolve())))
    review=tmp_path/'review.json'
    k.write(review,dict(verdict='PASS',builder='same',reviewer='same',bindings=dict(contract_id='c')))
    result=k.admit({},tmp_path,review)
    assert result['status']=='BLOCKED' and 'independent' in result['blockers'][0]
    with pytest.raises(ValueError,match='independent'): k.admit({},tmp_path,review,dry_run=False)


def test_checkpoint_contract_identity_rejects_different_recipe(tmp_path):
    import torch
    model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters()); sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
    ds=torch.utils.data.TensorDataset(torch.zeros(2,2)); dl=torch.utils.data.DataLoader(ds,batch_size=1,generator=torch.Generator().manual_seed(3))
    monitor=Monitor(tmp_path,cpu=True); monitor.admission=dict(contract_id='recipe-a')
    state=TrainerState(tmp_path,model,opt,sched,None,torch.Generator(),dl,['a','b'],2,monitor)
    state.epoch_loader=dl.generator.get_state(); state.save(1,0,[])
    monitor.admission=dict(contract_id='recipe-b')
    with pytest.raises(ValueError,match='training identity'):
        TrainerState(tmp_path,model,opt,sched,None,torch.Generator(),dl,['a','b'],2,monitor)


def test_epoch_boundary_restores_loader_generator(tmp_path):
    import torch
    model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters()); sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
    ds=torch.utils.data.TensorDataset(torch.zeros(2,2)); ds.epoch=0
    dl=torch.utils.data.DataLoader(ds,batch_size=1,generator=torch.Generator().manual_seed(3))
    monitor=Monitor(tmp_path,cpu=True)
    state=TrainerState(tmp_path,model,opt,sched,None,torch.Generator(),dl,['a','b'],2,monitor)
    list(dl); expected=dl.generator.get_state().clone(); state.save(1,0,[])
    dl.generator.manual_seed(3)
    loaded=TrainerState(tmp_path,model,opt,sched,None,torch.Generator(),dl,['a','b'],2,monitor)
    loaded.begin_epoch(1,ds)
    assert torch.equal(loaded.epoch_loader,expected)


def test_checkpoint_heartbeats_do_not_hide_throughput_collapse():
    guard=ThroughputGuard(0)
    for t in range(0,241,15): guard.observe(dict(phase='train',windows=t*100),t)
    with pytest.raises(ValueError,match='throughput collapse'):
        for t in range(255,2400,15):
            guard.observe(dict(phase='checkpoint',windows=24000),t-1)
            guard.observe(dict(phase='train',windows=24000+(t-240)*5),t)


def test_completed_model_must_match_contract_and_hash(tmp_path):
    (tmp_path/'models').mkdir(); (tmp_path/'models/E_prior_s0_f0.pt').write_bytes(b'model fixture')
    rec=dict(status='COMPLETE',epochs=16,model_fold=0,train_studies=3190,contract_id='sealed',model_sha256=k.sha(tmp_path/'models/E_prior_s0_f0.pt'))
    k.write(tmp_path/'fit-complete.json',rec)
    assert k.completed_fit(dict(contract_id='sealed'),tmp_path)==rec
    with pytest.raises(ValueError): k.completed_fit(dict(contract_id='other'),tmp_path)
    (tmp_path/'models/E_prior_s0_f0.pt').write_bytes(b'changed')
    with pytest.raises(ValueError): k.completed_fit(dict(contract_id='sealed'),tmp_path)


def test_absolute_production_commands_and_no_cache_rewrite():
    cache=k.s.WORK/'cpu-cache-reviewed'; argv=k.trainer_argv(cache)
    assert argv[argv.index('--px')+1]==str(cache)
    assert argv[argv.index('--out')+1]==str(k.OUT)
    assert argv[argv.index('--only-folds')+1]=='0' and '--resume' in argv
    for path in (argv[0],argv[1],*[argv[argv.index(flag)+1] for flag in ('--px','--out','--targets','--init')]): assert Path(path).is_absolute()
    command=k.run_command()
    for path in (command[0],command[1],command[command.index('--cache')+1],command[command.index('--review')+1]): assert Path(path).is_absolute()
