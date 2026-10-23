"""R6 diagnostic: exact serving contracts, fixed ranks, paired statistics and gates."""
import ast
import copy
import os
import types
import numpy as np
import pytest
from tools import r6_gold58_precheck_v1 as r


def synthetic():
    rng=np.random.default_rng(231)
    y=np.column_stack([rng.permutation(np.arange(58)%2) for _ in range(12)])
    vals={n:rng.choice(np.linspace(0,1,13),(58,12)) for n in ('R6flip','R7flip','replacement','HB167','stack')}
    return y,vals


def test_exact_actual_release_contracts_and_worker_arguments(tmp_path):
    files,ports=r.contracts()
    assert ports['flip']==r.d.stage_files(r.STAGES[0])['exp169_r7_port.py']
    assert ports['plain'].replace(b'    if manifest.get("flip_tta"):\n        args += ["--flip-tta"]\n',b'    args += ["--flip-tta"]\n')==ports['flip']
    manifest=r.io.read_json(r.BUNDLE/'manifest.json')
    outputs=[]
    for raw in [ports['plain'],ports['flip'],r.d.stage_files(r.STAGES[1])['exp169_r7_port.py']]:
        tree=ast.parse(raw); fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='worker_args')
        from pathlib import Path
        ns=dict(Path=Path,require=r.need)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<worker-args>','exec'),ns)
        outputs.append(ns['worker_args'](manifest,'bundle','anatomy'))
    assert outputs[1]==outputs[2] and outputs[0]==[x for x in outputs[1] if x!='--flip-tta']
    assert r.sha(r.BUNDLE/'manifest.json')==r.MANIFEST_SHA


def test_cpu_envelope_and_gpu_instrumentation(tmp_path):
    from tools import exp169_r7_port as port
    runtime=r.cpu_diagnostic_source(port,r.ROOT/'tools/exp169_r7_src',tmp_path)
    worker=(runtime/'compact_infer.py').read_text()
    assert '"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")' in worker
    assert 'names, visible = ["CPU"], [""]' in (runtime/'probe_dual.py').read_text()
    # Execute the injected envelope itself: CPU disables compile, GPU preserves it.
    start=worker.index('    if a.device == "cpu" and os.environ.get(')
    end=worker.index('    dev = torch.device(a.device)', start)
    import textwrap
    envelope=compile(textwrap.dedent(worker[start:end]),'<envelope>','exec')
    for device,expected in [('cpu',False),('cuda:0',True)]:
        args=types.SimpleNamespace(device=device,compile=True)
        ns=dict(a=args,os=os)
        exec(envelope,ns)
        assert args.compile is expected
        assert ns['cpu_compile_requested']==(True if device=='cpu' else None)
    compile(worker,'<cpu>','exec')
    gpu=port.diagnostic_source(r.ROOT/'tools/exp169_r7_src',tmp_path/'gpu')
    assert 'cpu_compile_requested' not in (gpu/'compact_infer.py').read_text()
    compile(r.d.instrument((gpu/'compact_infer.py').read_text()),'<gpu>','exec')


def test_cpu_controller_env(monkeypatch):
    seen=[]
    port=types.SimpleNamespace(diagnostic_source=lambda *a:None,s6=types.SimpleNamespace(_run_controller=lambda args,env:seen.append(env)))
    monkeypatch.setattr(r,'port_module',lambda _:port)
    with r.execution_port('plain','cpu') as p: p.s6._run_controller([],{'CUDA_VISIBLE_DEVICES':'-1'})
    assert seen==[{'CUDA_VISIBLE_DEVICES':''}]
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    r.c.cpu_only()


def test_bootstrap_ties_shared_draws_against_sklearn():
    from sklearn.metrics import roc_auc_score
    y,v=synthetic(); rules=dict(r.RULES,draws=15)
    actual,weights,attempts=r.bootstrap_auc(y,list(v.values()),rules)
    for k,value in enumerate(v.values()):
        for b,w in enumerate(weights):
            idx=np.repeat(np.arange(58),w.astype(int))
            assert np.allclose(actual[k,b],[roc_auc_score(y[idx,j],value[idx,j]) for j in range(12)])
    again=r.bootstrap_auc(y,list(v.values()),rules)
    assert np.array_equal(actual,again[0]) and np.array_equal(weights,again[1])


def test_degenerate_draw_rejection_and_cap():
    y,v=synthetic(); y[:]=0; y[0,:]=1
    _,w,n=r.bootstrap_auc(y,list(v.values()),dict(r.RULES,draws=20))
    assert n>len(w)
    with pytest.raises(ValueError,match='degeneracy'):
        r.bootstrap_auc(y,list(v.values()),dict(r.RULES,draws=20,max_attempts=1))


@pytest.mark.parametrize('damage',['nan','inf','shape','bounds','label','single'])
def test_bad_metrics(damage):
    y,v=synthetic()
    if damage=='nan': v['stack'][0,0]=np.nan
    if damage=='inf': v['stack'][0,0]=np.inf
    if damage=='shape': v['stack']=v['stack'][:-1]
    if damage=='bounds': v['stack'][0,0]=1.1
    if damage=='label': y[0,0]=2
    if damage=='single': y[:,0]=0
    with pytest.raises(ValueError): r.paired_metrics(y,v)


def test_hold_even_for_gain_and_harm(monkeypatch):
    y,_=synthetic(); monkeypatch.setitem(r.RULES,'draws',12)
    for sign in (0,1):
        v={n:y.copy() for n in ('R6flip','R7flip','replacement','HB167','stack')}
        v['replacement']=sign*y+(1-sign)*(1-y)
        stats=r.paired_metrics(y,v)
        assert stats['decision']==r.HOLD and stats['qualified_for_submission'] is False
        assert stats['comparisons']['replacement-minus-HB167']['operational']==('PASS' if sign else 'FAIL')
    assert r.metric(0,np.zeros(12))['statistical']=='UNRESOLVED'
    assert r.metric(1,np.ones(12))['statistical']=='GAIN'
    assert r.metric(-1,-np.ones(12))['statistical']=='HARM'


def test_fixed_stack_is_not_reranked_and_parent_oracle():
    rng=np.random.default_rng(38); ids=[str(i) for i in range(58)]
    common={n:rng.random((58,12)) for n in r.d.COMMON}
    masks={n:np.zeros(58,dtype=bool) for n in common}
    a,b=rng.random((2,58,12)); v=r.outputs(ids,common,masks,a,b)
    rr=.65*r.rank(common['R'])+.35*r.rank(common['R224'])
    own=r.rank((17*r.rank(common['EXP143v2'])+17*r.rank(common['EXP155'])+12*r.rank(common['Q_A5q']))/46)
    slot=.5*r.rank(a)+.5*r.rank(b)
    assert np.array_equal(v['stack'],.85*r.rank(.5*rr+.5*slot)+.15*own)
    assert not np.array_equal(v['stack'],.85*r.rank(.5*rr+.5*r.rank(slot))+.15*own)
    for damage in ('mask','dtype','sentinel','duplicate'):
        co=copy.deepcopy(common); ma=copy.deepcopy(masks); ii=ids[:]
        if damage=='mask': ma['R'][0]=True
        if damage=='dtype': ma['R']=ma['R'].astype(int)
        if damage=='sentinel': co['R'][0,:]=.5
        if damage=='duplicate': ii[0]=ii[1]
        with pytest.raises(ValueError): r.outputs(ii,co,ma,a,b)


def review_fixture():
    reg=dict(binding_sha256='b',binding=dict(differences=['x']))
    review=dict(verdict='PASS',builder=r.BUILDER,reviewer='independent',execution_authority='B',tag=r.TAG,
                binding_sha256='b',validation_sha256='v',approved_differences=['x'])
    return reg,review


@pytest.mark.parametrize('field,bad',[('reviewer',r.BUILDER),('reviewer',''),('verdict','FAIL'),('execution_authority','builder'),
    ('tag','old'),('binding_sha256','old'),('validation_sha256','old'),('approved_differences',[])])
def test_review_bypass(field,bad):
    reg,review=review_fixture(); r.check_review(review,reg,'v'); review[field]=bad
    with pytest.raises(ValueError): r.check_review(review,reg,'v')


def test_unreviewed_run_and_unsupervised_worker_never_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'WORK',tmp_path)
    monkeypatch.setattr(r,'lane',lambda:pytest.fail('GPU reached'))
    with pytest.raises(OSError): r.run()
    monkeypatch.delenv('HB164_CONTROLLER_PID',raising=False)
    with pytest.raises(ValueError,match='supervised'): r.worker()
    assert not list(tmp_path.glob('*'))


def test_launch_check_writes_no_admission(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'WORK',tmp_path)
    monkeypatch.setattr(r,'review_ready',lambda:dict(binding_sha256='b'))
    assert r.launch_check()['state'].endswith('NO_ADMISSION')
    assert not list(tmp_path.glob('*'))
