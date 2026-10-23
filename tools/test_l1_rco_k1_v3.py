"""K1 v3-G lineage, narrow diff, actual ConvNeXt coverage and fail-closed binding."""
import ast
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pytest
import torch
from tools import l1_rco_k1_v2 as parent
from tools import l1_rco_k1_v3 as k
from tools import l1_rco_k1_optim_v3 as op
from tools import l1_rco_k1_state_v2 as st


def reg(): return k.read(k.WORK/'registration.json')


def test_registered_parent_and_unchanged_recipe_data_evaluator():
    r=k.validate(reg()); p=parent.read(parent.WORK/'registration.json')
    for name in ('train','holdout','roster','recipe','evaluator','collapse_guard','best_checkpoint','pins','projection'):
        assert r[name]==p[name]
    assert r['optimizer_policy']==op.POLICY and r['contract_id']!=p['contract_id']
    assert k.OUT!=parent.OUT and k.QUEUE!=parent.QUEUE and k.TAG!=parent.TAG
    assert r['exposure']=='EXPOSED_EXPLORATORY' and r['clean_oof'] is False
    assert k.sha(k.WORK/'targets.csv')==k.sha(parent.WORK/'targets.csv')==r['parent_v2']['targets_sha256']


def test_optimizer_is_only_computational_trainer_diff():
    original=(k.ROOT/k.s.SOURCE/'compact_train.py').read_text(encoding='utf-8')
    base=parent.derive_trainer(original); actual=k.derive_trainer(original)
    assert k.sha(parent.WORK/'compact_train_l1.py')=='34dae104527a2e8c27ed328c5e2bca4039d28e7e596ab530b32de63a95848d8e'
    assert base==(parent.WORK/'compact_train_l1.py').read_text(encoding='utf-8')
    start=base.index('            enc = [p for n, p in model.named_parameters()')
    end=base.index('            ema = None\n',start)
    expected=base[:start]+('            from tools.l1_rco_k1_optim_v3 import training_groups\n'
        '            groups = training_groups(model, a)\n'
        '            opt = torch.optim.AdamW(groups, weight_decay=a.wd)\n')+base[end:]
    for name in ('check_training_ids','check_trainer_args'):
        expected=expected.replace('from tools.l1_rco_k1_v2 import '+name,'from tools.l1_rco_k1_v3 import '+name)
    assert actual.split('\n',1)[1]==expected
    assert actual==(k.WORK/'compact_train_l1.py').read_text(encoding='utf-8')
    assert (k.WORK/'compact_train_l1.py').read_bytes()==actual.replace('\n','\r\n').encode('utf-8')


def test_all_other_runtime_functions_and_spawn_shim_are_unchanged():
    def functions(text): return {n.name: ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=Path(parent.__file__).read_text(encoding='utf-8')
    new=Path(k.__file__).read_text(encoding='utf-8').replace('k1v3-grouping','k1v2').replace('k1v3_trainer','k1v2_trainer').replace('l1_rco_k1_v3.py','l1_rco_k1_v2.py').replace('l1_rco_k1_smoke_v3','l1_rco_k1_smoke_v2')
    a,b=functions(old),functions(new)
    changed={'code_pins','register','validate','derive_trainer','check_trainer_args','prepare'}
    assert {n for n in a if a[n]!=b[n]}==changed
    assert set(b)-set(a)=={'expected_registration'}
    old=(k.ROOT/'tools/l1_k1v2_trainer.py').read_text(encoding='utf-8')
    new=(k.ROOT/'tools/l1_k1v3_trainer.py').read_text(encoding='utf-8').split('\n',1)[1]
    assert new.replace('l1_k1v3_trainer','l1_k1v2_trainer').replace('k1v3-grouping','k1v2').replace('l1_rco_k1_v3','l1_rco_k1_v2')==old


def test_train_argv_is_parent_except_candidate_paths_and_fixed_flags_rejected():
    r=reg(); a=k.trainer_argv(k.CACHE,r); b=parent.trainer_argv(k.CACHE,r)
    replacements={str(k.WORK/'compact_train_l1.py'):str(parent.WORK/'compact_train_l1.py'),
                  str(k.WORK/'targets.csv'):str(parent.WORK/'targets.csv'),str(k.OUT):str(parent.OUT)}
    assert [replacements.get(x,x) for x in a]==b
    recipe=r['recipe']; args=SimpleNamespace(**{n:recipe[n] for n in ('lr','head_lr','epochs','warm_frac','batch','crop')},
        train_windows=64,l1_guard_constant=r['collapse_guard']['constant_loss'],llrd=1.,wd=.02,ema=.999)
    k.check_trainer_args(args)
    for name,value in [('lr',5e-5),('head_lr',1e-3),('epochs',16),('warm_frac',.2),('llrd',.9),('wd',0.),('ema',0.)]:
        bad=copy.copy(args); setattr(bad,name,value)
        with pytest.raises(ValueError): k.check_trainer_args(bad)


def test_policy_and_recipe_tamper_rejected(monkeypatch):
    original=reg(); monkeypatch.setattr(k,'expected_registration',lambda:original)
    for field in ('optimizer_policy','recipe'):
        bad=copy.deepcopy(original); bad[field]['lr']=5e-5
        bad['contract_id']=k.digest({n:v for n,v in bad.items() if n!='contract_id'})
        with pytest.raises(ValueError): k.validate(bad)


def test_actual_convnext360_group_membership_without_forward():
    torch.set_num_threads(2)
    ct=k.import_trainer()
    model=ct.Net('convnext_small','',crop=352,in_chans=3,spatial=True,extent=False)
    groups,manifest=op.parameter_groups(model)
    rows={r['name']:r for r in manifest['parameters']}
    assert len(rows)==360 and len(model.state_dict())==362 and all(g['params'] for g in groups)
    assert len({id(p) for g in groups for p in g['params']})==360
    assert len([n for n in rows if n.endswith('.gamma')])==36
    assert rows['sp_head_norm.weight']['lr']==1e-4 and rows['sp_head_norm.bias']['lr']==1e-4
    for name in ('sp_q','q_win','cls_w','slot.weight','view.weight','geom.weight','sp_key.weight','sp_pos.weight'):
        assert rows[name]['lr']==3e-4 and rows[name]['weight_decay']==.02
    for name in ('bias','proj.1.weight','out_norm.weight'):
        assert rows[name]['lr']==3e-4 and rows[name]['weight_decay']==0
    # Independent structural oracle: enumerate concrete owning module objects,
    # not the candidate helper's prefix/reason logic.
    modules=dict(model.named_modules())
    for name,p in model.named_parameters():
        owner,_,leaf=name.rpartition('.')
        expected_lr=1e-4 if owner=='sp_head_norm' or owner.split('.')[0]=='enc' else 3e-4
        decay=.02
        if isinstance(modules[owner],torch.nn.LayerNorm) or leaf in ('bias','gamma') or p.ndim<=1: decay=0.
        assert (rows[name]['lr'],rows[name]['weight_decay'])==(expected_lr,decay),name
    k.write(k.WORK/'parameter-coverage.json',manifest)
    assert not torch.cuda.is_initialized()


def test_missing_or_self_review_blocks_cpu_admission(tmp_path,monkeypatch):
    r=reg(); original_work=k.WORK
    monkeypatch.setattr(k,'WORK',tmp_path)
    monkeypatch.setattr(k,'validate',lambda r:r)
    monkeypatch.setattr(k,'trainer_argv',lambda cache,reg:['same'])
    monkeypatch.setattr(k,'complete_cache',lambda r,cache:{'status':'COMPLETE'})
    monkeypatch.setattr(k,'review_binding',lambda r:{'contract_id':r['contract_id']})
    k.write(tmp_path/'prepare.json',{'argv':['same'],'cache':str(k.CACHE.resolve())})
    review=tmp_path/'review.json'
    for payload in (None,dict(verdict='PASS',builder='same',reviewer='same',bindings={'contract_id':r['contract_id']})):
        if payload is not None:k.write(review,payload)
        answer=k.admit(r,k.CACHE,review,dry_run=True)
        assert answer['status']=='BLOCKED' and answer['blockers']


def test_old_candidate_contract_rejected_before_state_restore(tmp_path):
    # Real TrainerState save/load with identical architecture/group shape; the
    # admission contract alone must still reject a checkpoint from another fit.
    def state(contract):
        model=torch.nn.Linear(2,2); opt=torch.optim.AdamW(model.parameters())
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        monitor=st.Monitor(tmp_path,cpu=True); monitor.admission={'contract_id':contract}
        dl=torch.utils.data.DataLoader(torch.ones(2,2),batch_size=1,generator=torch.Generator().manual_seed(1))
        return st.TrainerState(tmp_path,model,opt,sched,None,torch.Generator().manual_seed(2),dl,['a','b'],10,monitor)
    one=state('v2-contract'); one.begin_epoch(0,SimpleNamespace()); one.save(0,0,[])
    with pytest.raises(ValueError,match='checkpoint training identity'): state('v3-contract')
