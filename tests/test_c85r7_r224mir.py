"""H-B-170 strict patch and fail-closed real-output postcheck regressions."""
import copy
import json
import math
from pathlib import Path
import pytest
from tools import c85r7_r224mirq_v1126 as b
from tools import c85r7_r224mir_timing as h
from tools import c85r7_r224mir_cpu as cpu


def test_exact_scope_and_parent_preservation():
    nb,meta,files,blob=b.payload()
    original=b.parent()[1]
    assert len(files)==31
    assert sorted(k for k in files if files[k]!=original[k])==sorted([b.SOURCE,b.WRAPPER])
    assert cpu.function_bytes(files[b.SOURCE],'_mirror_tri')==cpu.function_bytes(files['public-serving.py'],'_mirror_tri')
    for name in ('eval_windows','infer_probs','load_model','build_study'):
        assert cpu.function_bytes(files[b.SOURCE],name)==cpu.function_bytes(original[b.SOURCE],name)
    assert b.scope(nb,meta)[0]==files
    nb['cells'][0]['source'].append('\n# forbidden\n')
    with pytest.raises(ValueError,match='forbidden'): b.scope(nb,meta)


def test_shape_crop_and_consumer_patch():
    files=b.payload()[2]
    raw=files[b.SOURCE].decode()
    assert '(94,3,224,224)' in raw
    assert 'mirror=False' in raw and 'mirror=True)' in raw
    assert 'arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)' in raw
    assert raw.index('tri = _mirror_tri(tri, cs, D)',raw.index('def gpu_windows')) < raw.index('key = (str(dev), norm)',raw.index('def gpu_windows'))
    assert b.decision(.949,.948,25000,1000)
    assert not b.decision(.949,.9479,25000,1000)
    assert not b.decision(.949,.950,32000,401)
    with pytest.raises(ValueError): b.decision(.949,float('nan'),25000,1000)


@pytest.fixture
def passes():
    real=h.t.io.read_json(h.LEGACY/'worker-status.json')
    ids=[r['study'] for r in real['rows']]
    result=[]
    for mode in h.ORDER:
        value=copy.deepcopy(real); value.update(identity=h.TAG,mode=mode)
        for row in value['rows']: row['inference_seconds']=row['seconds']
        result.append(value)
    return result,ids


def test_real_output_summary_holds_without_receipt(passes):
    values,ids=passes
    out=h.summarize(values,ids,dict(conservative_plain_seconds=100,r7_hidden_seconds=3000,assumption='test'))
    assert out['runtime_pass'] is None
    assert out['runtime_limit_seconds']==32400 and out['failed_study_fallbacks']==0
    assert all(abs(v-1)<1e-12 for v in out['per_target_spearman'])
    with pytest.raises(ValueError,match='actual independently'): h.summarize(values,ids,dict(conservative_plain_seconds=100,r7_hidden_seconds=3000,assumption='test'),{'hidden_seconds':100})


@pytest.mark.parametrize('mutate',[
    lambda p:p[0].update(identity='other'),
    lambda p:p[0].update(seconds=float('nan')),
    lambda p:p[0].update(seconds=.001),
    lambda p:p.reverse(),
    lambda p:p[0]['rows'][0].update(probabilities=[.5]*11),
    lambda p:p[0]['rows'][0].update(probabilities=[float('nan')]*12),
    lambda p:p[0]['rows'][0].update(study=p[0]['rows'][1]['study']),
    lambda p:p[0]['rows'][0].update(failed_study_fallback=True),
    lambda p:p[0]['rows'][0].update(peak_reserved_bytes=13*2**30),
    lambda p:p[0]['rows'][0].update(inference_seconds=-1),
    lambda p:p[0]['rows'][0].update(windows=93),
    lambda p:p[0]['rows'][0].update(fp32_retry=not p[0]['rows'][0]['fp32_retry']),
])
def test_corrupt_real_outputs_block(passes,mutate):
    values,ids=passes; mutate(values)
    with pytest.raises(ValueError): h.summarize(values,ids,dict(conservative_plain_seconds=100,r7_hidden_seconds=3000,assumption='test'))


def test_pair_smoke_required():
    with pytest.raises(ValueError,match='full pair'): h.validate_pair({}, {})


def test_pair_smoke_reproduces_previous_tensor_lifetime(monkeypatch):
    import weakref
    from types import SimpleNamespace
    import numpy as np
    refs=[]; alive=[]
    class Window:
        shape=(94,3,224,224)
    def windows(*args,**kwargs):
        alive.append(sum(r() is not None for r in refs))
        out=Window(); refs.append(weakref.ref(out)); return out
    cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:100,max_memory_reserved=lambda:200)
    ns=dict(torch=SimpleNamespace(cuda=cuda),build_study=lambda *a:(None,None),
        gpu_windows=windows,infer_pair_probs=lambda *a:np.full(12,.5))
    monkeypatch.setattr(h.t,'validate_volume',lambda *a:None)
    monkeypatch.setattr(h.t.shared,'memory_admission',lambda *a:None)
    pair=h.pair_smoke(ns,None,{'studies':['first','second']},None,None,{})
    assert alive==[0,1,2,2]  # Both prior windows live during next RHS calls.
    assert pair['consecutive_studies']==2 and all(r() is None for r in refs)
    for mutation in ({'consecutive_studies':1},{'retain_until_replacement':False},{'peak_reserved_bytes':201}):
        with pytest.raises(ValueError): h.validate_pair(dict(pair,**mutation),{})


def test_self_declared_parent_receipt_cannot_promote_runtime(passes):
    values,ids=passes
    fake=dict(item='H-B-167',submission_id=57224580,measurement='actual_hidden',
        hidden_seconds=100,source_receipt_sha256='a'*64,independent_review='PASS')
    with pytest.raises(ValueError,match='no self-declared'):
        h.summarize(values,ids,dict(conservative_plain_seconds=100,r7_hidden_seconds=3000,assumption='test'),fake)


def test_complete_postchecks_replayed_from_real_measurement(monkeypatch,tmp_path):
    monkeypatch.setattr(h,'WORK',tmp_path)
    value=h.replay()
    assert value['state']=='CPU_POSTCHECKS_REPLAY_ONLY'
    assert value['result']['state']=='MEASURED_REQUIRES_RESULT_REVIEW'
    assert value['result']['result']['runtime_pass'] is None
    assert 'Pair flags are schema adapters only' in value['limitation']


@pytest.mark.parametrize('mutation',['pair','aggregate','headroom','telemetry'])
def test_complete_postchecks_reject_invalid_replay(monkeypatch,tmp_path,mutation):
    monkeypatch.setattr(h,'WORK',tmp_path)
    original=h.finalize
    def damaged(value,telemetry,desktop,binding,calibration):
        if mutation=='pair': value['pair']['retain_until_replacement']=False
        if mutation=='aggregate': value['rows']=value['rows'][:-1]
        if mutation=='headroom': desktop=dict(desktop,used_mib=12000)
        if mutation=='telemetry': telemetry=dict(telemetry,peak_gpu_mib=16000)
        return original(value,telemetry,desktop,binding,calibration)
    monkeypatch.setattr(h,'finalize',damaged)
    with pytest.raises(ValueError): h.replay()


def test_review_cannot_be_self_or_unbound(monkeypatch,tmp_path):
    monkeypatch.setattr(h,'WORK',tmp_path)
    (tmp_path/'validate.json').write_text('{}')
    current={'kernel':'x'}
    good=dict(verdict='PASS',builder='hb170-build',reviewer='independent',execution_authority='B',binding_sha256=h.t.digest(current),validate_sha256=h.t.sha(tmp_path/'validate.json'))
    h.check_review(good,current)
    for change in ({'reviewer':'hb170-build'},{'binding_sha256':'x'},{'execution_authority':'builder'},{'verdict':'FAIL'}):
        with pytest.raises(ValueError): h.check_review(dict(good,**change),current)


def test_gpu_no_direct_worker_bypass(monkeypatch):
    monkeypatch.delenv('HB164_CONTROLLER_PID',raising=False)
    with pytest.raises(ValueError,match='supervised'): h.worker()


def test_real_windows_desktop_parser():
    import subprocess
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5)
    found=h.t.shared.gpu_python_pids(result.stdout,set())
    assert isinstance(found,list)
    assert all('python' in str(x).lower() or isinstance(x,int) for x in found)
