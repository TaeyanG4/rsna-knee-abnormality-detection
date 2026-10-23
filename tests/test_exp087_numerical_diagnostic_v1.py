"""Focused synthetic telemetry tests: CPU tensors only, no model/data import."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import exp087_numerical_diagnostic_v1 as d


def test_finite_elements_reduction_overflow_is_distinguished_without_writeback():
    g=torch.tensor([2e20,-2e20],dtype=torch.float32);before=g.clone();rng=torch.random.get_rng_state().clone()
    row=d.gradient_statistics([('synthetic',SimpleNamespace(grad=g))])
    assert row['finite']==2 and row['nonfinite']==0 and row['native_total_l2']=='+Infinity'
    assert row['all_elements_l2_fp64']==pytest.approx(2**.5*float(g[0]))
    assert row['writeback']==0 and row['fp64_on_failure_only']
    assert torch.equal(g,before) and torch.equal(rng,torch.random.get_rng_state())


def test_nonfinite_elements_counts_and_valid_json():
    g=torch.tensor([3.,4.,float('inf'),-float('inf'),float('nan')]);before=g.view(torch.int32).clone()
    r=d.gradient_statistics([('synthetic',SimpleNamespace(grad=g)),('missing',SimpleNamespace(grad=None))])
    assert r['count']==5 and r['finite']==2 and r['nonfinite']==3 and r['all_elements_l2_fp64'] is None
    assert r['finite_elements_l2_fp64']==5 and r['parameters_without_grad']==1
    item=r['parameters'][0]
    assert (item['nan'],item['positive_inf'],item['negative_inf'],item['max_finite_abs'])==(1,1,1,4.)
    assert torch.equal(g.view(torch.int32),before)
    json.dumps(r,allow_nan=False)


def test_zero_grad_and_mixed_fp32_fp64_are_aggregated():
    r=d.gradient_statistics([('a',SimpleNamespace(grad=torch.zeros(2))),('b',SimpleNamespace(grad=torch.tensor([5.,12.],dtype=torch.float64)))])
    assert r['all_elements_l2_fp64']==13 and r['count']==4 and r['nonfinite']==0
    assert {x['dtype'] for x in r['parameters']}=={'torch.float32','torch.float64'}


def test_durable_current_and_last_completed_hash_are_available_before_next_step(tmp_path):
    state={};d.begin_attempt(tmp_path,state,1,0,0,0,['u2','u1'],True,1024.)
    r=json.loads((tmp_path/'current-attempt.json').read_text())
    assert r['ordered_batch_uid_sha256']==hashlib.sha256(b'u2\nu1').hexdigest()
    assert r['stage']=='before_forward' and r['last_completed'] is None and r['logits_finite'] is None
    d.current(tmp_path,state,'before_backward',logits_finite=True,loss_finite=True,loss=.25)
    d.completed_attempt(tmp_path,state,1,1,1024.)
    d.begin_attempt(tmp_path,state,2,1,0,1,['u3','u4'],True,1024.)
    r=json.loads((tmp_path/'current-attempt.json').read_text())
    assert r['attempt_1based']==2 and r['last_completed']['optimizer_updates']==1
    assert len((tmp_path/'attempts.jsonl').read_text().splitlines())==1
    assert not list(tmp_path.glob('*.tmp'))


def test_cap_fails_before_current_or_journal_growth(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'OUTPUT_CAP',d.RESERVE+8)
    with pytest.raises(RuntimeError,match='output cap'):d.durable_json(tmp_path/'a.json',{'large':'abcdefghijk'})
    with pytest.raises(RuntimeError,match='output cap'):d.journal(tmp_path,{'large':'abcdefghijk'})
    assert not list(tmp_path.iterdir())


def scaler_fixture(found):
    opt=object();state={id(opt):{'found_inf_per_device':{'cpu':torch.tensor(found)}}}
    return opt,SimpleNamespace(_per_optimizer_states=state,get_scale=lambda:1024.)


def test_foundinf_refuses_step_without_backoff_or_state_mutation():
    opt,scaler=scaler_fixture(1.)
    before=scaler._per_optimizer_states[id(opt)]['found_inf_per_device']['cpu'].clone()
    with pytest.raises(RuntimeError,match='no optimizer/scaler update'):d.assert_no_found_inf(scaler,opt)
    assert torch.equal(before,scaler._per_optimizer_states[id(opt)]['found_inf_per_device']['cpu'])
    opt2,scaler2=scaler_fixture(0.);d.assert_no_found_inf(scaler2,opt2)


def test_capture_preserves_nonfinite_clip_exception_and_gradient_bits(tmp_path):
    g=torch.tensor([float('inf'),1.]);before=g.clone();opt,scaler=scaler_fixture(1.)
    model=SimpleNamespace(named_parameters=lambda:[('synthetic',SimpleNamespace(grad=g))])
    original=RuntimeError('synthetic original nonfinite norm')
    with pytest.raises(RuntimeError,match='synthetic original nonfinite norm') as got:
        try:raise original
        except BaseException as exc:
            d.capture_failure(tmp_path,{'stage':'before_clip','attempt_1based':39},exc,model,opt,scaler)
            raise
    assert got.value is original and torch.equal(g,before)
    r=json.loads((tmp_path/'numerical-failure.json').read_text())
    assert r['status']=='CHILD_FAILURE_OBSERVED' and r['scale']==1024. and r['found_inf']=={'cpu':1.}
    assert r['gradients']['nonfinite']==1 and not r['optimizer_step_after_failure']
    assert not r['scheduler_step_after_failure'] and not r['scaler_update_after_failure']


def test_capture_failure_does_not_mask_original_when_telemetry_itself_fails(monkeypatch,tmp_path):
    opt,scaler=scaler_fixture(0.);original=ValueError('original')
    def fail(*a,**k):raise OSError('synthetic telemetry failure')
    monkeypatch.setattr(d,'gradient_statistics',fail);monkeypatch.setattr(d,'durable_json',fail)
    with pytest.raises(ValueError) as got:
        try:raise original
        except BaseException as exc:
            d.capture_failure(tmp_path,{},exc,SimpleNamespace(named_parameters=lambda:[]),opt,scaler)
            raise
    assert got.value is original


def test_normal_prefix_is_not_reported_as_training_completion(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'finish_trace',lambda *a:{'scheduler_steps':200})
    d.finish_diagnostic(tmp_path,dict(attempt_1based=200,optimizer_updates_after=200),{},None,'proto')
    r=json.loads((tmp_path/'diagnostic-receipt.json').read_text())
    assert r['status']=='BOUNDED_PREFIX_COMPLETE_NO_FAILURE_OBSERVED' and not r['training_complete']
    assert r['scheduler_horizon']==3448 and r['no_validation_inference'] and r['no_checkpoint']


def test_runner_is_passive_and_keeps_original_recipe_prefix_and_failure_barrier():
    s=(ROOT/d.RUNNER).read_text();old=(ROOT/'training/runners/exp087_coverage_b8.py').read_text();tree=ast.parse(s)
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and 'raptor_train' in ast.unparse(n) for n in tree.body)
    prefix=s[s.index('    runtime = configure_runtime'):s.index('    optimizer_updates, skipped_updates')]
    oldprefix=old[old.index('    runtime = configure_runtime'):old.index('    @torch.no_grad()')]
    assert prefix==oldprefix
    assert 'torch.save(' not in s and 'def evaluate(' not in s and 'write_predictions(out' not in s
    assert s.index('guard.begin_attempt(')<s.index('logits = model(x)')<s.index('scaler.scale(loss).backward()')
    assert s.index('error_if_nonfinite=True')<s.index('guard.assert_no_found_inf')<s.index('scaler.step(opt); scaler.update()')
    assert 'guard.capture_failure(out, diagnostic_state, exc, model, opt, scaler)\n        raise' in s
    assert 'guard.stop_loader(tl)' in s and 'attempt == guard.MAX_ATTEMPTS' in s
    assert 'total_steps=steps, pct_start=0.15' in s and 'steps != 3448' in s


def test_supervisor_keeps_child_failure_distinct_and_cleans_up(tmp_path):
    command=[sys.executable,'-c','import sys; print("synthetic child failure"); sys.exit(1)']
    budget=dict(seconds=8,rss_bytes=1024**3,output_bytes=d.OUTPUT_CAP)
    with pytest.raises(RuntimeError,match='failure preserved'):
        d.supervise(command,tmp_path,{'budget':budget},{**__import__('os').environ,'EXP087_DIAGNOSTIC_PHASE_SHA256':'synthetic'},'proto')
    r=json.loads((tmp_path/'resource-supervisor.json').read_text())
    assert r['status']=='FAIL' and r['returncode']==1 and r['killed'] is None
    assert r['cleanup_errors']==[] and r['cleanup_killed_pids']==[] and not r['training_complete']
