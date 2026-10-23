"""S6-all: safe identity, serving contract, registered contrasts and review gates."""
import ast
import copy
import os
import textwrap
import types
import numpy as np
import pytest
from tools import s6all_gold58_precheck_v1 as s


def fixture():
    rng=np.random.default_rng(136)
    ids=[str(i) for i in range(58)]
    common={n:rng.choice(np.linspace(0,1,17),(58,12)) for n in s.d.COMMON}
    failed={n:np.zeros(58,dtype=bool) for n in common}
    r7,model,r6=rng.random((3,58,12))
    labels=np.column_stack([rng.permutation(np.arange(58)%2) for _ in range(12)])
    return ids,common,failed,r7,model,r6,labels


def test_real_asset_safe_identity_and_pinned_source():
    identity=s.bundle_identity(load=True)
    assert identity['sha256']==s.MODEL_SHA and identity['bytes']==200185003
    assert identity['state_keys']==362
    s.s6.verify_source()
    source=s.s6.SOURCE/'compact_infer.py'
    tree=ast.parse(source.read_text())
    loads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='load']
    assert loads and all(any(k.arg=='weights_only' and isinstance(k.value,ast.Constant)
                           and k.value.value is True for k in n.keywords) for n in loads)


def test_safe_load_failure_has_no_unsafe_retry(monkeypatch):
    import torch
    seen=[]
    def broken(*args,**kwargs):
        seen.append(kwargs); raise TypeError('unsupported')
    monkeypatch.setattr(torch,'load',broken)
    with pytest.raises(ValueError,match='failed closed; no unsafe retry'):
        s.bundle_identity(load=True)
    assert len(seen)==1 and seen[0]['weights_only'] is True and seen[0]['map_location']=='cpu'


@pytest.mark.parametrize('value',[{}, {'x':1}, {'x':np.zeros(1)}])
def test_non_tensor_checkpoint_rejected(monkeypatch,value):
    import torch
    monkeypatch.setattr(torch,'load',lambda *a,**k:value)
    with pytest.raises(ValueError,match='tensor state dictionary'):
        s.bundle_identity(load=True)


def test_362_tensor_keys_without_required_spatial_schema_rejected(monkeypatch):
    import torch
    fake={str(i):torch.zeros(1) for i in range(362)}
    monkeypatch.setattr(torch,'load',lambda *a,**k:fake)
    with pytest.raises(ValueError,match='architecture mismatch'): s.bundle_identity(load=True)


def test_adapter_change_scope_and_exact_recipe():
    raw=s.adapter_bytes(); compile(raw,'<adapter>','exec')
    assert b'"--size", "320", "--crop", "288", "--chunk", "192"' in raw
    assert b'"--bundle", str(bundle_dir), "--arm", "E_prior"' in raw
    assert b'"--state-spec", str(Path(bundle_dir)' not in raw
    assert b'"--side-model", str(Path(bundle_dir)' not in raw
    assert b'env["CUDA_VISIBLE_DEVICES"] = ""' in raw
    original=s.Path(s.s6.__file__).read_bytes()
    lines=list(__import__('difflib').unified_diff(original.decode().splitlines(),raw.decode().splitlines()))
    assert len([x for x in lines if x.startswith('+') and not x.startswith('+++')])==3
    args=s.worker_args()
    assert args[args.index('--eval-stride')+1]=='1'
    assert all(a not in args for a in ('--state-spec','--side-model','--flip-tta','--compile'))


def test_runtime_cpu_envelope_and_gpu_observation_compile(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'WORK',tmp_path)
    runtime=s.diagnostic(s.s6.SOURCE,tmp_path/'cpu','cpu')
    worker=(runtime/'compact_infer.py').read_text()
    assert 'names, visible = ["CPU"], [""]' in (runtime/'probe_dual.py').read_text()
    start=worker.index('    if a.device == "cpu" and os.environ.get(')
    end=worker.index('    dev = torch.device(a.device)',start)
    envelope=compile(textwrap.dedent(worker[start:end]),'<envelope>','exec')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    exec(envelope,dict(a=types.SimpleNamespace(device='cpu'),os=os))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(RuntimeError,match='present-empty'):
        exec(envelope,dict(a=types.SimpleNamespace(device='cpu'),os=os))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    compile(worker,'<cpu-worker>','exec')
    gpu=s.diagnostic(s.s6.SOURCE,tmp_path/'gpu','cuda')
    code=(gpu/'compact_infer.py').read_text(); compile(code,'<gpu-worker>','exec')
    assert 'hb153.commit(' in code and 'torch.cuda.reset_peak_memory_stats()' in code
    assert 'wall_started=mark' in code


def test_three_formulas_independent_rank_oracle_no_mixed_rerank():
    from scipy.stats import rankdata
    ids,common,failed,r7,model,r6,_=fixture()
    rank=lambda x:rankdata(x,axis=0,method='average')/58
    values=s.outputs(ids,common,failed,r7,model,r6)
    a=.65*rank(common['R'])+.35*rank(common['R224'])
    own=rank((17*rank(common['EXP143v2'])+17*rank(common['EXP155'])+12*rank(common['Q_A5q']))/46)
    final=lambda slot:.85*rank(.5*a+.5*slot)+.15*own
    assert np.array_equal(values['S6all'],model) and np.array_equal(values['R7flip'],r7)
    for base,baseline,mixed in [(r7,'HB167','S6all_HB167'),(r6,'HB174','S6all_HB174')]:
        assert np.array_equal(values[baseline],final(rank(base)))
        slot=.60*rank(base)+.40*rank(model)
        assert np.array_equal(values[mixed],final(slot))
        assert not np.array_equal(values[mixed],final(rank(slot)))


@pytest.mark.parametrize('damage',['mask','dtype','sentinel','duplicate','nan','inf','shape','bounds'])
def test_bad_inputs_rejected(damage):
    ids,common,failed,r7,model,r6,_=fixture()
    if damage=='mask': failed['R'][0]=True
    if damage=='dtype': failed['R']=failed['R'].astype(int)
    if damage=='sentinel': common['R'][0,:]=.5
    if damage=='duplicate': ids[0]=ids[1]
    if damage=='nan': model[0,0]=np.nan
    if damage=='inf': model[0,0]=np.inf
    if damage=='shape': model=model[:-1]
    if damage=='bounds': model[0,0]=1.1
    with pytest.raises(ValueError): s.outputs(ids,common,failed,r7,model,r6)


def test_each_floor_independently_enforced():
    assert s.floors(np.zeros(12))
    macro=np.full(12,-.0011); assert not s.floors(macro)
    weak=np.full(12,.01); weak[s.RULES['weak_four']]=-.0021
    assert weak.mean()>-.001 and not s.floors(weak)
    one=np.full(12,.01); one[0]=-.0101
    assert one.mean()>-.001 and one[s.RULES['weak_four']].mean()>-.002 and not s.floors(one)
    one[0]=-.01; assert s.floors(one)


def test_per_target_bootstrap_and_missing_r6_hold(monkeypatch):
    from sklearn.metrics import roc_auc_score
    ids,common,failed,r7,model,r6,y=fixture()
    monkeypatch.setitem(s.RULES,'draws',19)
    # Repeated scores explicitly exercise half-credit AUC ties.
    model=np.round(model,1); r7=np.round(r7,1)
    values=s.outputs(ids,common,failed,r7,model,None)
    stats=s.paired_metrics(y,values)
    names=list(values); _,weights,_=s.r.bootstrap_auc(y,list(values.values()),s.RULES)
    for a,b in s.PAIRS[:2]:
        result=stats['comparisons'][a+'-minus-'+b]
        assert set(result['per_target'])==set(s.d.gold.LABELS)
        direct=[]
        for w in weights:
            index=np.repeat(np.arange(58),w.astype(int))
            direct.append([roc_auc_score(y[index,j],values[a][index,j])-roc_auc_score(y[index,j],values[b][index,j]) for j in range(12)])
        direct=np.asarray(direct)
        for j,target in enumerate(s.d.gold.LABELS):
            expected=np.quantile(direct[:,j],[.025,.975],method='linear')
            assert np.allclose(result['per_target'][target]['ci95'],expected)
        assert np.allclose(result['macro12']['ci95'],np.quantile(direct.mean(1),[.025,.975]))
    assert stats['comparisons']['S6all_HB174-minus-HB174']['operational']=='HOLD'
    assert stats['bootstrap']['valid_draws']==19
    assert stats['decision']==s.HOLD and stats['qualified_for_submission'] is False


def test_identity_control_zero_intervals(monkeypatch):
    ids,common,failed,r7,_,_,y=fixture(); monkeypatch.setitem(s.RULES,'draws',12)
    stats=s.paired_metrics(y,s.outputs(ids,common,failed,r7,r7,r7))
    for value in stats['comparisons'].values():
        assert value['operational']=='PASS'
        assert value['macro12']['ci95']==value['weak_four']['ci95']==[0.,0.]
        assert all(v['ci95']==[0.,0.] for v in value['per_target'].values())


@pytest.mark.parametrize('field,bad',[('reviewer',s.BUILDER),('reviewer',''),('verdict','FAIL'),('execution_authority','builder'),
    ('tag','old'),('binding_sha256','old'),('validation_sha256','old'),('approved_differences',[])])
def test_gpu_review_bypass(field,bad):
    reg=dict(binding_sha256='b')
    review=dict(verdict='PASS',builder=s.BUILDER,reviewer='independent',execution_authority='B',tag=s.TAG,
        binding_sha256='b',validation_sha256='v',approved_differences=s.DIFFERENCES)
    s.check_review(review,reg,'v'); review[field]=bad
    with pytest.raises(ValueError): s.check_review(review,reg,'v')


@pytest.mark.parametrize('field,bad',[('reviewer',s.BUILDER),('verdict','FAIL'),('binding_sha256','old'),('scope','GPU')])
def test_cpu_prereview_bypass(monkeypatch,field,bad):
    review=dict(verdict='PASS',builder=s.BUILDER,reviewer='independent',binding_sha256='b',scope='CPU_IMPLEMENTATION_TESTS_PARITY')
    monkeypatch.setattr(s.io,'read_json',lambda _:review)
    s.prereview(dict(binding_sha256='b')); review[field]=bad
    with pytest.raises(ValueError): s.prereview(dict(binding_sha256='b'))


def test_unreviewed_run_and_unsupervised_worker_never_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'WORK',tmp_path)
    monkeypatch.setattr(s,'lane',lambda _:pytest.fail('GPU reached'))
    with pytest.raises(OSError): s.run()
    monkeypatch.delenv('HB164_CONTROLLER_PID',raising=False)
    with pytest.raises(ValueError,match='supervised'): s.worker()
    assert not list(tmp_path.glob('*'))


def test_launch_check_no_admission_and_cache_unavailable(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'WORK',tmp_path)
    monkeypatch.setattr(s,'review_ready',lambda:dict(binding_sha256='b'))
    assert s.launch_check()['state'].endswith('NO_ADMISSION')
    assert not list(tmp_path.glob('*'))
    assert s.cached_r6(dict(binding=dict(r6_available=False))) is None


def test_registration_and_cpu_evidence_drift_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'WORK',tmp_path)
    b=dict(builder=s.BUILDER,rules=s.RULES,comparisons=[list(p) for p in s.PAIRS],decision=s.HOLD,
        command=s.launch_command(),differences=s.DIFFERENCES,inputs={},adapter_sha256=s.d.pack.digest(s.adapter_bytes()))
    reg=dict(state='REGISTERED',binding=b,binding_sha256=s.digest(b))
    s.io.journal(tmp_path/'registration.json',reg,tmp_path)
    assert s.current()==reg
    reg['binding_sha256']='wrong'; s.io.journal(tmp_path/'registration.json',reg,tmp_path)
    with pytest.raises(ValueError,match='policy drift'): s.current()
    reg['binding_sha256']=s.digest(b)
    monkeypatch.setattr(s,'current',lambda:reg)
    monkeypatch.setattr(s,'prereview',lambda _:None)
    valid=dict(state='CPU_VALIDATED',binding_sha256=reg['binding_sha256'],evidence={'prepare':'wrong'})
    s.io.journal(tmp_path/'prepare.json',{},tmp_path); s.io.journal(tmp_path/'validate.json',valid,tmp_path)
    with pytest.raises(ValueError,match='CPU evidence drift'): s.review_ready()


def test_inherited_supervisor_command_matches_registered_command(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'WORK',tmp_path)
    for attr in ('WORK','TAG','__file__','monitor'):
        monkeypatch.setattr(s.t,attr,getattr(s.t,attr))
    s.configure()
    assert s.t.launch_command()==s.launch_command()
    assert s.t.WORK==tmp_path/'gpu' and s.t.TAG==s.TAG


def test_copied_adapter_verifier_uses_trusted_source_default(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'WORK',tmp_path)
    s.export_adapter(); port=s.port_module('cpu')
    assert port.verify_code is s.s6.verify_code
    actual=port.verify_code(s.d.CODE,s.d.ANATOMY)
    assert set(s.s6.DIRECT_CODE_MODULES)<=set(actual['closure_sha256'])
