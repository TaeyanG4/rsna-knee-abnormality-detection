"""Synthetic CPU tests only; no original trainer import or model/data access."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import os
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import exp087_training as guard


def read_source(path):return (ROOT/path).read_text(encoding='utf-8')


def test_new_runner_and_windows_gate_have_no_top_level_model_or_data_execution():
    for path in ['training/runners/exp087_coverage_b8.py','tools/exp087_windows_gate.py']:
        tree=ast.parse(read_source(path))
        for stmt in tree.body:
            if isinstance(stmt,(ast.Import,ast.ImportFrom)):
                rendered=ast.unparse(stmt)
                assert 'raptor_train' not in rendered and 'build_model' not in rendered and 'torch' not in rendered
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='exec' for n in ast.walk(tree))


def test_runner_fixed_recipe_and_init_are_unchanged_except_declared_wrapper():
    old=read_source('training/runners/exp084_group_loss.py');new=read_source('training/runners/exp087_coverage_b8.py')
    exact=['model = build_model(a.arch, pretrained=use_timm).to(dev)',
        'torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)',
        'total_steps=steps, pct_start=0.15','shuffle_generator = torch.Generator().manual_seed(a.seed)',
        'shuffle=True, num_workers=a.workers, drop_last=True','pin_memory=dev == "cuda", persistent_workers=a.workers > 0',
        'model.eval()(torch.zeros(1, 1, 3, a.res, a.res, device=dev))','o = torch.sigmoid(logits.float())']
    assert all(x in old and x in new for x in exact)
    assert new.index('coverage_store = CoverageStore')<new.index('seed_all(a.seed)')
    assert 'tds = CoverageStudyWindows(tds, coverage_store)' in new
    assert 'DataLoader(CoverageStudyWindows(StudyWindows' in new
    assert 'if len(tl) != 431 or steps != 3448 or eval_bs != 4' in new
    assert 'Any optimizer skip violates exact B8 update contract' in new
    assert 'deferred_until_all_three_complete' not in new
    assert 'macro_auc(' not in new and 'gold_labels(' not in new


def test_pilot_uses_full_horizon_and_exits_before_checkpoint_or_eval():
    s=read_source('training/runners/exp087_coverage_b8.py')
    assert s.index('total_steps=steps')<s.index('if optimizer_updates == 32:')<s.index('torch.save(best')
    assert 'guard.pilot_finish(out, trace, tl, optimizer_updates, skipped_updates, torch)' in s
    assert 'a.epochs = 1' not in s and 'load_state_dict(pilot' not in s
    assert 'if optimizer_updates != 3448 or skipped_updates != 0' in s


def test_windows_gate_retains_full_membership_and_explicit_spawn():
    s=read_source('tools/exp087_windows_gate.py')
    assert 'ids=ordered[\'train\']' in s and 'labels={uid:np.zeros(12,np.float32) for uid in ids}' in s
    tree=ast.parse(s)
    loaders=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='DataLoader']
    assert len(loaders)==2 and all(any(k.arg=='multiprocessing_context' and isinstance(k.value,ast.Constant) and k.value.value=='spawn' for k in n.keywords) for n in loaders)
    assert "sampler=p['windows_gate']['indices']" in s and 'if len(production)!=431' in s
    assert 'dataset.baseline.corpus is not None' in s and "parent=make('enabled')" in s
    assert 'for epoch in (0,1)' in s and "('disabled','forced','enabled')" in s
    assert 'finally:cleanups.append(guard.stop_loader(loader))' in s
    assert 'train=False' not in s and 'build_model(' not in s


def test_exact_completion_refuses_even_one_skip(monkeypatch,tmp_path):
    prov={'initial_state_sha256':'generic'}
    s=dict(optimizer_updates=3448,skipped_updates=0,selected_epoch_1based=8,train=3455,val=888,
        weak_oof_macro_auc=None,metric_access=guard.METRICS,provenance=prov)
    t=dict(updates=3448,scheduler_steps=3448,worker_cleanup={'clean':True},actual_replacement_tensor_checks=1)
    monkeypatch.setattr(guard,'read',lambda p:s if Path(p).name=='summary.json' else t)
    for name in ('model_best.pt','oof_best.csv','provenance.json','history.json'):(tmp_path/name).write_bytes(b'synthetic')
    guard.validate_completion(tmp_path,{},dict(initial_state_sha256='generic'))
    s['optimizer_updates']=3447;s['skipped_updates']=1
    with pytest.raises(ValueError,match='exact-update'):guard.validate_completion(tmp_path,{},dict(initial_state_sha256='generic'))
    s.update(optimizer_updates=3448,skipped_updates=0);t['worker_cleanup']['clean']=False
    with pytest.raises(ValueError):guard.validate_completion(tmp_path,{},dict(initial_state_sha256='generic'))


def test_pilot_projection_uses_last24_and_does_not_write_checkpoint(monkeypatch,tmp_path):
    record=dict(pilot_whole_update_seconds=[100]*8+[.70]*24,updates=32,scheduler_steps=32)
    monkeypatch.setattr(guard,'finish_trace',lambda *args:record)
    (tmp_path/'input-trace.json').write_text('synthetic');(tmp_path/'provenance.json').write_text('{"protocol_sha256":"synthetic"}')
    torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:123))
    trace={'actual_replacement_tensor_checks':1}
    guard.pilot_finish(tmp_path,trace,None,32,0,torch)
    r=json.loads((tmp_path/'pilot-receipt.json').read_text())
    assert r['projection_pass'] and r['projected_full_seconds']==pytest.approx(1.3*.70*3448+300)
    assert r['scheduler_horizon']==3448 and r['no_checkpoint_saved'] and not (tmp_path/'model_best.pt').exists()
    with pytest.raises(ValueError):guard.pilot_finish(tmp_path,trace,None,31,0,torch)
    with pytest.raises(ValueError):guard.pilot_finish(tmp_path,trace,None,32,1,torch)


def test_prior_phase_cannot_be_accepted_from_child_receipt_alone(monkeypatch,tmp_path):
    monkeypatch.setattr(guard,'ROOT',tmp_path)
    gate=tmp_path/'gate';gate.mkdir()
    (gate/'receipt.json').write_text(json.dumps({'status':'PASS_WINDOWS_INPUT_INTEGRATION'}))
    (gate/'resource-supervisor.json').write_text(json.dumps({'status':'FAIL'}))
    with pytest.raises(ValueError,match='Windows gate'):guard.require_previous({'windows_gate':{'output':'gate'}},'pilot','synthetic')


def test_argument_builder_keeps_pilot_epochs8_pretrained_timm_and_trainbatch8():
    p=dict(arms={'pilot':dict(tag='pilot',expected_args=dict(epochs=8,bs=8,k=12,k_eval=24,accum=1,pretrained='timm',grad_ckpt=True,smoke=False,mode='pilot'))},paths={})
    argv=guard.runner_argv(p,'pilot',Path('registered.json'))
    assert argv[argv.index('--epochs')+1]=='8' and argv[argv.index('--bs')+1]=='8'
    assert argv[argv.index('--pretrained')+1]=='timm' and '--smoke' not in argv


def test_loader_cleanup_refuses_live_or_failed_worker():
    class Worker:
        pid=8;exitcode=0
        def join(self,timeout):pass
        def is_alive(self):return False
    w=Worker();it=SimpleNamespace(_workers=[w],_shutdown_workers=lambda:None)
    assert guard.stop_loader(SimpleNamespace(_iterator=it))['clean']
    w.exitcode=1
    with pytest.raises(RuntimeError,match='cleanly'):guard.stop_loader(SimpleNamespace(_iterator=it))


@pytest.mark.parametrize('failure',['job_write','monitor',None])
def test_supervisor_error_cleanup_or_final_elapsed_race(monkeypatch,tmp_path,failure):
    import psutil
    elapsed=2. if failure else 180.
    ticks=iter([0.,elapsed]);monkeypatch.setattr(guard.time,'monotonic',lambda:next(ticks))
    fake=SimpleNamespace(pid=123,returncode=None,running=True)
    class P:
        def __init__(self,pid):self.pid=pid;self.running=True
        def children(self,recursive=True):return [desc] if self.pid==123 else []
        def is_running(self):return fake.running if self.pid==123 else self.running
        def kill(self):
            self.running=False
            if self.pid==123:fake.running=False;fake.returncode=-9
        def memory_info(self):
            if failure=='monitor':raise OSError('synthetic monitor')
            return SimpleNamespace(rss=1)
    child=P(123);desc=P(124);objects={123:child,124:desc,os.getpid():P(os.getpid())}
    if not failure:child.children=lambda recursive=True:[]
    monkeypatch.setattr(psutil,'Process',lambda pid:objects[pid]);monkeypatch.setattr(psutil,'wait_procs',lambda *a,**k:([],[]))
    def poll():
        if not failure:fake.returncode=0;fake.running=False
        return fake.returncode
    fake.poll=poll;fake.kill=child.kill;fake.wait=lambda timeout:fake.returncode
    monkeypatch.setattr(guard.subprocess,'Popen',lambda *a,**k:fake)
    original=guard.dump
    if failure=='job_write':
        def fail_job(path,value):
            if Path(path).name=='supervisor-job.json':raise OSError('synthetic job write')
            original(path,value)
        monkeypatch.setattr(guard,'dump',fail_job)
    with pytest.raises(RuntimeError,match='supervised run failed'):
        guard.supervise(['not-run'],tmp_path,180,1000,dict(EXP087_PHASE='pilot',EXP087_PHASE_AUTHORIZATION_SHA256='auth'),'proto')
    r=json.loads((tmp_path/'resource-supervisor.json').read_text());assert r['status']=='FAIL'
    if failure:assert not fake.running and not desc.running and set(r['cleanup_killed_pids'])=={123,124}
    else:assert r['returncode']==0 and r['wall_seconds']==180
