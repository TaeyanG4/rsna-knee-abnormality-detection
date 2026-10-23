"""H-B-162 v1.0.1 mathematical, protocol and fail-closed tests.

Changelog 1.0.1: aggregate-only breach, immutable historical bindings, complete
prediction authentication, and evaluation-only CLI scope.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tools import hb162_sam_f0_v1 as m


def toy():
    torch.manual_seed(133)
    return torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.BatchNorm1d(5),
        torch.nn.Tanh(),torch.nn.Dropout(.3),torch.nn.Linear(5,2)).double()


def batches(n=3):
    g = torch.Generator().manual_seed(87)
    return [(torch.randn(4,3,generator=g,dtype=torch.float64),
             torch.randn(4,2,generator=g,dtype=torch.float32)) for _ in range(n)]


@pytest.mark.parametrize('count',[1,3,15,16])
@pytest.mark.parametrize('arm',['ADAMW','SAM'])
def test_step_matches_independent_reference(count,arm):
    actual = toy(); expected = copy.deepcopy(actual)
    data = batches(count)
    oa = torch.optim.AdamW(actual.parameters(),lr=.002,weight_decay=.0588)
    oe = torch.optim.AdamW(expected.parameters(),lr=.002,weight_decay=.0588)
    fn = torch.nn.MSELoss()
    for step in range(2):
        torch.manual_seed(100+step)
        start_rng = torch.get_rng_state()
        for x,y in data:
            (fn(expected(x).float(),y)/count).backward()
        after = torch.get_rng_state()
        original = copy.deepcopy(expected.state_dict())
        if arm == 'SAM':
            weights = [p for p in expected.parameters() if p.ndim >= 2]
            norm = sum(p.grad.float().square().sum() for p in weights).sqrt().item()
            with torch.no_grad():
                for p in weights:
                    p += .05*p.grad/(norm+1e-12)
            oe.zero_grad(set_to_none=True)
            torch.set_rng_state(start_rng)
            for x,y in data:
                (fn(expected(x).float(),y)/count).backward()
            expected.load_state_dict(original)
            torch.set_rng_state(after)
        torch.nn.utils.clip_grad_norm_(expected.parameters(),3.,error_if_nonfinite=True)
        oe.step(); oe.zero_grad(set_to_none=True)
        torch.set_rng_state(start_rng)
        m.effective_step(actual,oa,data,fn,arm)
        assert torch.equal(torch.get_rng_state(),after)
        for name,p in actual.state_dict().items():
            torch.testing.assert_close(p,expected.state_dict()[name],rtol=2e-7,atol=1e-9)
        assert actual[1].num_batches_tracked == count*(step+1)


def test_second_pass_same_dropout_rng_and_samples():
    model = toy(); seen = []
    hook = model[3].register_forward_hook(lambda mod,inp,out: seen.append((out == 0).clone()))
    data = batches(3)
    m.effective_step(model,torch.optim.AdamW(model.parameters()),data,torch.nn.MSELoss(),'SAM')
    hook.remove()
    assert len(seen) == 6
    assert all(torch.equal(a,b) for a,b in zip(seen[:3],seen[3:]))


def test_exception_restores_unperturbed_weights_and_bn():
    model = toy(); original = copy.deepcopy(model.state_dict()); count = 0
    def fail(phase,detail):
        nonlocal count
        if phase == 'gradient':
            count += 1
            if count == 2:
                raise RuntimeError('second pass failed')
    with pytest.raises(RuntimeError,match='second pass'):
        m.effective_step(model,torch.optim.AdamW(model.parameters()),batches(1),torch.nn.MSELoss(),'SAM',progress=fail)
    for name,p in model.named_parameters():
        assert torch.equal(p,original[name])
    assert model[1].num_batches_tracked == 1


def test_bn_stats_equal_one_pass():
    sam = toy(); adam = copy.deepcopy(sam)
    for model,arm in ((sam,'SAM'),(adam,'ADAMW')):
        torch.manual_seed(42)
        m.effective_step(model,torch.optim.AdamW(model.parameters()),batches(3),torch.nn.MSELoss(),arm)
    for name in ('running_mean','running_var','num_batches_tracked'):
        assert torch.equal(getattr(sam[1],name),getattr(adam[1],name))


def pair(delta=.0025):
    return dict(references={r:dict(delta=delta,per_target_delta={t:delta for t in m.history.TARGETS})
                            for r in m.serving.RULERS})


def inputs():
    return ({b:pair() for b in ('ADAMW','PARENT')},
            {str(s):{b:pair(.001) for b in ('ADAMW','PARENT')} for s in (42,43)})


def test_gate_go_and_exact_boundary():
    a,b = inputs()
    for ref in a.values():
        ref['references']['dread']['delta'] = -.001
        ref['references']['dread']['per_target_delta']['ACL'] = -.003
    assert m.scientific_gate(a,b)['status'] == 'GO'


def test_three_rulers_must_beat_both_controls():
    a,b = inputs()
    a['PARENT']['references']['jev']['delta'] = .001
    a['ADAMW']['references']['aquila']['delta'] = .001
    result = m.scientific_gate(a,b)
    assert result['wins'] == {'ADAMW':3,'PARENT':3}
    assert result['joint_wins'] == 2 and result['status'] == 'DEAD'


@pytest.mark.parametrize('defect',['floor','seed_floor','zero_seed','negative_seed','two_wins','nan','missing_target','missing_ruler','missing_control','missing_seed'])
def test_gate_rejects(defect):
    a,b = inputs()
    if defect == 'floor': a['PARENT']['references']['jev']['per_target_delta']['ACL'] = -.00301
    if defect == 'seed_floor': b['43']['ADAMW']['references']['aquila']['per_target_delta']['ACL'] = -.00301
    if defect in ('zero_seed','negative_seed'):
        b['43']['PARENT'] = pair(0 if defect == 'zero_seed' else -.001)
    if defect == 'two_wins':
        for r in ('jev','dread'): a['ADAMW']['references'][r]['delta'] = .00199
    if defect == 'nan': a['ADAMW']['references']['jev']['delta'] = float('nan')
    if defect == 'missing_target': del a['PARENT']['references']['jev']['per_target_delta']['ACL']
    if defect == 'missing_ruler': del a['PARENT']['references']['jev']
    if defect == 'missing_control': del a['PARENT']
    if defect == 'missing_seed': del b['43']
    assert m.scientific_gate(a,b)['status'] == 'DEAD'


def test_probability_mean_precedes_auc(monkeypatch):
    import tools.exp170_convnext_t_l2 as e
    calls = []
    def fake(a,b,*args):
        calls.append((a.copy(),b.copy())); return pair()
    monkeypatch.setattr(e,'paired_jev_consensus',fake)
    arms = {a+str(s):np.full((4,12),v) for a,s,v in [('SAM',42,.2),('SAM',43,.8),('ADAMW',42,.1),('ADAMW',43,.3)]}
    m.comparisons(arms,np.full((4,12),.4),{},[])
    np.testing.assert_allclose(calls[0][0],.5)
    np.testing.assert_allclose(calls[0][1],.2)
    assert len(calls) == 6


def test_review_bypass_fails():
    with pytest.raises(ValueError,match='independent'):
        m.check_review({})
    with pytest.raises(ValueError,match='independent'):
        m.check_review(dict(verdict='PASS',independent=True,reviewer=m.BUILDER,scope=m.TAG))


def test_direct_worker_bypass_fails(monkeypatch):
    monkeypatch.delenv('HB162_CONTROLLER',raising=False)
    with pytest.raises(ValueError,match='controller'):
        m.require_controller('full')


def test_bad_arm_and_oversize_rejected():
    model = toy()
    for arm,data in [('other',batches()),('SAM',batches(17)),('SAM',[])]:
        with pytest.raises(ValueError,match='effective'):
            m.effective_step(model,torch.optim.AdamW(model.parameters()),data,torch.nn.MSELoss(),arm)


def test_policy_no_wall_cap_and_parent_precision():
    assert m.POLICY['max_wall_hours'] is None
    assert m.POLICY['train_precision'] == 'bf16'
    assert m.POLICY['accumulation'] == 16
    assert m.GATE == 'seed-mean macro >= +0.002 on >= 3/4 rulers against BOTH the AdamW continuation and the untouched parent, no ruler-target delta < −0.003, positive four-ruler mean in both continuation seeds; otherwise DEAD.'


@pytest.mark.parametrize('elapsed,expected',[(599,False),(600,True),(899,True),(900,True)])
def test_idle_watchdog(elapsed,expected):
    assert bool(m.serving.guard_reason(elapsed,0,(15000,100,70),16376)) == expected


def test_healthy_progress_has_no_wall_cap():
    assert m.serving.guard_reason(100000,99999,(9000,98,180),16376) is None


def test_registered_rejects_changed_identity(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'WORK',tmp_path)
    (tmp_path/'registration.json').write_text(json.dumps(dict(identity='old')))
    monkeypatch.setattr(m,'registration',lambda:dict(identity='new'))
    with pytest.raises(ValueError,match='stale'):
        m.registered()


@pytest.mark.parametrize('fails',[False,True])
def test_serving_restores_training_tf32(monkeypatch,fails):
    before = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        def fake(*args):
            assert torch.backends.cuda.matmul.allow_tf32
            if fails:
                raise RuntimeError('inference failure')
            return 'ok'
        monkeypatch.setattr(m,'_panel',fake)
        if fails:
            with pytest.raises(RuntimeError,match='inference failure'):
                m.panel(None,[], 'cuda:0')
        else:
            assert m.panel(None,[], 'cuda:0') == 'ok'
        assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before


@pytest.mark.parametrize('defect',[None,'roster','peaks','artifacts','steps','bn','model'])
def test_completed_requires_whole_bound_run(monkeypatch,tmp_path,defect):
    from tools import c85r7_flipall_timing as timing
    monkeypatch.setattr(m,'WORK',tmp_path)
    monkeypatch.setattr(m,'binding',lambda:{'test':'bound'})
    monkeypatch.setattr(timing,'memory_admission',lambda *args:None)
    folder = tmp_path/'gpu/calibration'; folder.mkdir(parents=True)
    artifacts = {}
    for name in ['parent.json','runtime.json']+[f'{a}42/{n}' for a in m.POLICY['arms'] for n in ('model_best.pt','predictions.json')]:
        path = folder/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('test')
        artifacts[name] = m.sha(path)
    records = [dict(arm=a,seed=42,studies=2,optimizer_steps=2,bn=dict(studies=2,unique_studies=2),
                    epoch_seconds=[1.,1.],model_sha256=artifacts[f'{a}42/model_best.pt']) for a in m.POLICY['arms']]
    output = dict(binding=m.binding(),phase='calibration',peaks=[1]*5,records=records,artifacts=artifacts)
    if defect == 'roster': records.pop()
    if defect == 'peaks': output['peaks'].pop()
    if defect == 'artifacts': del artifacts['parent.json']
    if defect == 'steps': records[0]['optimizer_steps'] = 1
    if defect == 'bn': records[0]['bn']['unique_studies'] = 1
    if defect == 'model': records[0]['model_sha256'] = 'stale'
    (folder/'output.json').write_text(json.dumps(output))
    receipt = dict(binding=m.binding(),exit_code=0,cleanup_errors=[],phase='calibration',
                   desktop={},peak_device_mib=1,output_sha256=m.sha(folder/'output.json'))
    (folder/'controller.json').write_text(json.dumps(receipt))
    (folder/'status.json').write_text(json.dumps(dict(state='COMPLETE',error=None)))
    if defect:
        with pytest.raises(ValueError,match='completed'):
            m.completed('calibration')
    else:
        assert m.completed('calibration') == output


def retained_run(monkeypatch, tmp_path):
    """Complete synthetic receipt graph; numeric telemetry mirrors the retained run."""
    monkeypatch.setattr(m, 'WORK', tmp_path)
    monkeypatch.setattr(m, 'binding', lambda: {'fixture': 'bound'})
    m.io.atomic_json(tmp_path/'registration.json', {'fixture': 'registration'})
    monkeypatch.setattr(m, 'LEGACY_REGISTRATION_SHA', m.sha(tmp_path/'registration.json'))
    for phase in ('calibration', 'full'):
        folder = tmp_path/'gpu'/phase
        folder.mkdir(parents=True)
        seeds = [42] if phase == 'calibration' else [42, 43]
        roster = [(a, s) for s in seeds for a in m.POLICY['arms']]
        artifacts = {}
        for name in ['parent.json', 'runtime.json']+[f'{a}{s}/{n}' for a,s in roster
                                                    for n in ('model_best.pt', 'predictions.json')]:
            path = folder/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture')
            artifacts[name] = m.sha(path)
        studies, steps, epochs = (2, 2, 2) if phase == 'calibration' else (3455, 648, 3)
        records = [dict(arm=a, seed=s, studies=studies, optimizer_steps=steps,
                        bn=dict(studies=studies,unique_studies=studies),epoch_seconds=[1.]*epochs,
                        model_sha256=artifacts[f'{a}{s}/model_best.pt']) for a,s in roster]
        output = dict(binding=m.binding(), phase=phase, peaks=[11700*2**20]*(2*len(roster)+1),
                      records=records, artifacts=artifacts)
        m.io.atomic_json(folder/'output.json', output)
        breach = phase == 'full'
        receipt = dict(binding=m.binding(), phase=phase, exit_code=1 if breach else 0,
            error=m.MEMORY_ERROR if breach else None, cleanup_errors=[],
            desktop=dict(device_name=m.POLICY['gpu'],total_mib=16376.,used_mib=1138. if breach else 1145.),
            peak_device_mib=13686. if breach else 13129.,
            output_sha256=None if breach else m.sha(folder/'output.json'))
        m.io.atomic_json(folder/'controller.json', receipt)
        m.io.atomic_json(folder/'status.json', dict(state='FAILED' if breach else 'COMPLETE',error=receipt['error']))
        (folder/'run.log').write_text('{"status": "WORKER_COMPLETE"}\n')
    monkeypatch.setattr(m, 'LEGACY_FULL_OUTPUT_SHA', m.sha(tmp_path/'gpu/full/output.json'))
    return tmp_path/'gpu/full'


def test_memory_breach_requires_explicit_flag_and_keeps_strict_admission(monkeypatch,tmp_path):
    retained_run(monkeypatch,tmp_path)
    with pytest.raises(ValueError, match='completed controller'):
        m.completed('full')
    assert len(m.completed('full', True)['records']) == 4
    # Calibration never receives the exception, even when explicitly requested.
    m.io.atomic_json(tmp_path/'gpu/calibration/status.json', dict(state='FAILED',error=m.MEMORY_ERROR))
    with pytest.raises(ValueError, match='completed controller'):
        m.completed('calibration', True)


@pytest.mark.parametrize('defect', [
    'oom','substring','cleanup','missing_cleanup','exit','status','status_error','binding',
    'calibration','manifest','worker','artifact','peak_reserved','aggregate_not_breached',
    'headroom','registration','roster','accounting'])
def test_breach_rejects_any_other_failure(monkeypatch,tmp_path,defect):
    folder = retained_run(monkeypatch,tmp_path)
    receipt = m.io.read_json(folder/'controller.json')
    output = m.io.read_json(folder/'output.json')
    if defect == 'oom': receipt['error'] = 'RuntimeError: CUDA out of memory'
    if defect == 'substring': receipt['error'] += '; another error'
    if defect == 'cleanup': receipt['cleanup_errors'] = ['owned child still live']
    if defect == 'missing_cleanup': del receipt['cleanup_errors']
    if defect == 'exit': receipt['exit_code'] = 2
    if defect == 'status': m.io.atomic_json(folder/'status.json',dict(state='RUNNING',error=None))
    if defect == 'status_error': m.io.atomic_json(folder/'status.json',dict(state='FAILED',error='other'))
    if defect == 'binding': receipt['binding'] = {}
    if defect == 'calibration': m.io.atomic_json(tmp_path/'gpu/calibration/status.json',dict(state='FAILED',error='bad'))
    if defect == 'manifest': (folder/'output.json').write_text('{}')
    if defect == 'worker': (folder/'run.log').write_text('{"status": "WORKER_FAILED"}\n')
    if defect == 'artifact': (folder/'SAM43/predictions.json').write_text('changed')
    if defect == 'peak_reserved': output['peaks'][0] = 12300*2**20
    if defect == 'aggregate_not_breached': receipt['peak_device_mib'] = 13000
    if defect == 'headroom': receipt['peak_device_mib'] = 15000
    if defect == 'registration': (tmp_path/'registration.json').write_text('{}')
    if defect == 'roster': output['records'].pop()
    if defect == 'accounting': output['records'][0]['optimizer_steps'] = 647
    if defect in ('peak_reserved','roster','accounting'):
        m.io.atomic_json(folder/'output.json', output)
        monkeypatch.setattr(m,'LEGACY_FULL_OUTPUT_SHA',m.sha(folder/'output.json'))
    m.io.atomic_json(folder/'controller.json',receipt)
    with pytest.raises(ValueError):
        m.completed('full',True)


@pytest.mark.parametrize('target',['SAM42','SAM43','ADAMW42','ADAMW43','parent'])
@pytest.mark.parametrize('defect',['order','duplicate','shape','nan','infinity','range','hash'])
def test_every_prediction_is_authenticated_before_comparisons(monkeypatch,tmp_path,target,defect):
    monkeypatch.setattr(m,'WORK',tmp_path)
    monkeypatch.setattr(m,'evaluation_admission',lambda flag:None)
    ids = [f's{i}' for i in range(888)]
    monkeypatch.setattr(m,'authenticate',lambda:(None,ids,{},[],{}))
    output = dict(artifacts={})
    monkeypatch.setattr(m,'completed',lambda *args:output)
    folder = tmp_path/'gpu/full'
    for key in ('SAM42','SAM43','ADAMW42','ADAMW43','parent'):
        name = 'parent.json' if key == 'parent' else key+'/predictions.json'
        row = dict(ids=ids.copy(),probabilities=np.full((888,12),.5).tolist())
        if key == target:
            if defect == 'order': row['ids'] = list(reversed(ids))
            if defect == 'duplicate': row['ids'][-1] = row['ids'][0]
            if defect == 'shape': row['probabilities'].pop()
            if defect == 'nan': row['probabilities'][0][0] = float('nan')
            if defect == 'infinity': row['probabilities'][0][0] = float('inf')
            if defect == 'range': row['probabilities'][0][0] = 1.01
        path = folder/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(row))
        output['artifacts'][name] = 'bad' if key == target and defect == 'hash' else m.sha(path)
    with pytest.raises(ValueError,match='prediction UID|invalid probabilities'):
        m.evaluation_inputs(True)


@pytest.mark.parametrize('state',['running','orphaned','unknown','stale','absent'])
def test_memory_evaluation_never_reads_active_or_unknown_job(monkeypatch,state):
    from tools import job_runtime
    monkeypatch.setattr(job_runtime,'job_status',lambda *a:{'state':state})
    monkeypatch.setattr(m.io,'read_json',lambda *a:pytest.fail('read run before terminal check'))
    with pytest.raises(ValueError,match='stopped terminal'):
        m.evaluation_admission(True)


@pytest.mark.parametrize('command',['register','validate','prepare','cpu-smoke','admit','run','_worker'])
def test_memory_flag_never_weakens_launch_commands(command):
    with pytest.raises(SystemExit) as exc:
        m.main([command,'--accept-memory-breach'])
    assert exc.value.code == 2


def test_historical_backup_and_only_registered_source_substitutions(monkeypatch,tmp_path):
    from tools import job_runtime
    monkeypatch.setattr(job_runtime,'job_status',lambda *a:{'state':'FAILED'})
    monkeypatch.setattr(m,'WORK',tmp_path)
    monkeypatch.setattr(m,'LEGACY_BACKUP',tmp_path/'backup')
    sources = {}
    for name in m.LEGACY_SOURCES:
        path = m.LEGACY_BACKUP/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('original')
        sources[name] = m.sha(path)
    monkeypatch.setattr(m,'LEGACY_SOURCES',sources)
    saved = dict(version='1.0.0',sources=dict(sources, dependency='unchanged'),policy={'gate':'fixed'})
    m.io.atomic_json(tmp_path/'registration.json',saved)
    monkeypatch.setattr(m,'LEGACY_REGISTRATION_SHA',m.sha(tmp_path/'registration.json'))
    current = dict(version='1.0.1',sources={**sources,'dependency':'unchanged'},policy={'gate':'fixed'})
    monkeypatch.setattr(m,'registration',lambda:copy.deepcopy(current))
    m.io.atomic_json(tmp_path/'cpu-smoke.json',dict(status='CPU_SMOKE_PASS',
        registration_sha256=m.LEGACY_REGISTRATION_SHA,studies=2,steps_per_arm=2))
    m.io.atomic_json(tmp_path/'review.json',{})
    reviewed=[]; monkeypatch.setattr(m,'check_review',lambda v:reviewed.append(v))
    m.evaluation_admission(True)
    assert reviewed == [{}]
    current['sources']['dependency'] = 'changed'
    with pytest.raises(ValueError,match='dependencies/policy/identity'):
        m.evaluation_admission(True)
    current['sources']['dependency'] = 'unchanged'
    (m.LEGACY_BACKUP/next(iter(sources))).write_text('changed')
    with pytest.raises(ValueError,match='backup changed'):
        m.evaluation_admission(True)
