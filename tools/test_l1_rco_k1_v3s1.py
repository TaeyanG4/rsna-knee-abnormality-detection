"""Prospective seed replicate: exact recipe/data parity and fail-closed admission."""
import ast
import copy
from types import SimpleNamespace
import pytest
from tools import l1_rco_k1_v3 as old
from tools import l1_rco_k1_v3s1 as k

def reg(): return k.read(k.WORK/'registration.json')

def test_exact_same_recipe_and_cohorts_except_seed():
    r=k.validate(reg());p=old.read(old.WORK/'registration.json')
    for name in ('train','holdout','roster','evaluator','collapse_guard','best_checkpoint','pins','projection','optimizer_policy'):
        assert r[name]==p[name]
    recipe=copy.deepcopy(r['recipe']);assert recipe.pop('seed_base')==1
    assert p['recipe'].get('seed_base',0)==0
    assert recipe==p['recipe']
    assert r['seed_replicate']['parent_contract']==p['contract_id']
    assert k.sha(k.WORK/'targets.csv')==old.sha(old.WORK/'targets.csv')
    assert k.WORK!=old.WORK and k.TAG!=old.TAG and k.QUEUE!=old.QUEUE

def test_generated_trainer_only_seed_assertion_identity_change():
    a=(old.WORK/'compact_train_l1.py').read_text(encoding='utf-8')
    b=(k.WORK/'compact_train_l1.py').read_text(encoding='utf-8')
    expected=a.replace('tools.l1_rco_k1_v3 import','tools.l1_rco_k1_v3s1 import').replace('a.seed_base == 0','a.seed_base == 1')
    assert b==expected
    for token in ('1000 + 17 * seed','5000 + 17 * seed + k','9000 + 17 * seed + k',
                  'range(a.seed_base, a.seed_base + a.seeds)'):
        assert token in b

def test_argv_uses_only_new_paths_and_one_seed_override():
    r=reg();a=k.trainer_argv(k.CACHE,r);b=old.trainer_argv(old.CACHE,old.read(old.WORK/'registration.json'))
    assert a.count('--seed-base')==1 and a[a.index('--seed-base')+1]=='1'
    start=a.index('--seed-base');del a[start:start+2]
    repl={str(k.WORK/'compact_train_l1.py'):str(old.WORK/'compact_train_l1.py'),
          str(k.WORK/'targets.csv'):str(old.WORK/'targets.csv'),str(k.OUT):str(old.OUT)}
    assert [repl.get(x,x) for x in a]==b

def test_seed_and_schedule_drift_rejected():
    r=reg();recipe=r['recipe']
    a=SimpleNamespace(**{n:recipe[n] for n in ('seed_base','lr','head_lr','epochs','warm_frac','batch','crop')},
         train_windows=64,l1_guard_constant=r['collapse_guard']['constant_loss'],llrd=1.,wd=.02,ema=.999)
    k.check_trainer_args(a)
    for name,value in [('seed_base',0),('epochs',11),('lr',5e-5),('ema',0),('warm_frac',.2)]:
        b=copy.copy(a);setattr(b,name,value)
        with pytest.raises(ValueError):k.check_trainer_args(b)

def test_guard_and_process_functions_identical():
    def funcs(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    from pathlib import Path
    a,b=funcs(Path(old.__file__)),funcs(Path(k.__file__))
    for name in ('collapse_reason','constant_loss','require_no_candidate_evidence','completed_fit',
                 'epoch_collapse','stop_child','supervise','final_training_log','evaluate'):
        assert a[name]==b[name],name

def test_review_missing_tampered_or_falsely_independent_fails(tmp_path,monkeypatch):
    r=reg();monkeypatch.setattr(k,'WORK',tmp_path);monkeypatch.setattr(k,'validate',lambda x:x)
    monkeypatch.setattr(k,'trainer_argv',lambda *args:['fixed'])
    monkeypatch.setattr(k,'complete_cache',lambda *args:{'status':'COMPLETE'})
    monkeypatch.setattr(k,'review_binding',lambda *args:{'contract_id':r['contract_id']})
    k.write(tmp_path/'prepare.json',dict(argv=['fixed'],cache=str(k.CACHE.resolve())))
    path=tmp_path/'review.json'
    for payload in (None,dict(schema='independent',verdict='PASS'),
                    dict(schema='b-direct-review/v1',verdict='PASS_B_DIRECT',reviewer='B/Codex',
                         bindings={'contract_id':'wrong'},authority='Taeyang2026-10-19-solo-continuous-GPU')):
        if payload:k.write(path,payload)
        assert k.admit(r,k.CACHE,path)['status']=='BLOCKED'

def test_real_cpu_smoke_covers_seed1_resume_and_guard():
    s=k.read(k.WORK/'cpu-smoke.json')
    assert s['contract_id']==reg()['contract_id'] and s['status']=='PASS' and not s['cuda_initialized']
    assert s['mid_epoch_exact_resume'] and s['epoch_boundary_exact_resume'] and s['full_net_backward_finite']
    assert s['guard_stop']['no_model_export'] and s['holdout_training_artifact_rows']==0
