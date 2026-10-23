"""CPU-only refusal, runtime and post-run evidence tests for H-B-175."""
import copy
import json
import numpy as np
import pytest
from tools import c85r7_r76_timing as t
from tools import c85r7_r76_cpu_smoke as smoke


def test_decision_boundaries():
    assert t.decision(.956,9)['keep']
    assert not t.decision(.955999,8)['keep']
    assert not t.decision(.957,9.00001)['keep']
    assert not t.decision(.957,8)['qualified_for_submission']
    for score,hours in [(float('nan'),8),(.957,float('inf')),(.957,6)]:
        with pytest.raises(ValueError): t.decision(score,hours)


def test_projection():
    assert t.project(360,10)['projected_hours']==7.73
    assert t.project(1080,10)['margin_hours']==pytest.approx(-.73)
    with pytest.raises(ValueError): t.project(0,1)


def test_progress_and_wddm():
    t.progress_guard(899,0,[],1)
    with pytest.raises(ValueError): t.progress_guard(900,0,[],1)
    with pytest.raises(ValueError): t.progress_guard(1800,1799,[(0,0),(1799,1)],1)
    t.progress_guard(1800,1799,[(0,0),(1799,8)],1)
    with pytest.raises(ValueError): t.wddm_guard(600,0,[14000,100,80],16376)
    t.wddm_guard(599,0,[14000,100,80],16376)
    t.wddm_guard(601,0,[14000,100,100],16376)


@pytest.mark.parametrize('arm',['flip-r7','flip-r6'])
def test_full_runtime(arm):
    folder=t.b.R6_LOCAL if arm=='flip-r6' else t.ROOT/'storage/external/heliosli-r7cnxfsss-v1'
    m=json.loads((folder/'manifest.json').read_text())
    r=dict(models=m['models'],models2=[],flip_tta=True,workers=2,
           **{k:m[k] for k in ('size','crop','views','slices')})
    t.runtime_check({'runtime':r},arm)
    for k,v in [('flip_tta',False),('workers',1),('crop',224),('models',[])]:
        bad=dict(r); bad[k]=v
        with pytest.raises(ValueError): t.runtime_check({'runtime':bad},arm)


def test_review_gate_and_stale_binding(tmp_path,monkeypatch):
    with pytest.raises(ValueError): t.review(None)
    report=tmp_path/'report.md'; report.write_text('Independent scope evidence')
    path=tmp_path/'review.json'
    bound={'kernel':'exact'}
    monkeypatch.setattr(t,'bindings',lambda:bound)
    good=dict(verdict='PASS',reviewer='independent-lane',bindings=bound,
              scope='HB175_N32_LOCAL_MEASUREMENT_ONLY',report_path=str(report),report_sha256=t.sha(report))
    path.write_text(json.dumps(good)); assert t.review(path)==good
    for key,value in [('reviewer','astra-b5'),('verdict','PENDING'),('bindings',{}),('report_sha256','bad'),('scope','other')]:
        bad=dict(good); bad[key]=value; path.write_text(json.dumps(bad))
        with pytest.raises(ValueError): t.review(path)


def test_run_refuses_before_gpu(monkeypatch):
    monkeypatch.setattr(t,'lane_admission',lambda:pytest.fail('GPU reached'))
    with pytest.raises(ValueError): t.run(None)


@pytest.mark.parametrize('changed',['r6','r7'])
def test_bindings_reject_manifest_drift(monkeypatch,changed):
    monkeypatch.setattr(t.b,'validate',lambda:{'kernel_sha256':'test'})
    def digest(path):
        is_r6='heliosli-r6-' in str(path)
        if (changed=='r6')==is_r6: return 'changed'
        return t.b.R6_SHA if is_r6 else t.R7_PIN
    monkeypatch.setattr(t,'sha',digest)
    with pytest.raises(ValueError,match='manifest changed'): t.bindings()


@pytest.mark.parametrize('failure',['none','fallback','flip','ids'])
def test_real_output_slot_replay(tmp_path,monkeypatch,failure):
    candidate=smoke.runtime()
    monkeypatch.setattr(smoke,'runtime',lambda:candidate)
    ids=[str(i) for i in range(32)]
    (tmp_path/'ids.json').write_text(json.dumps(ids))
    monkeypatch.setattr(t,'WORK',tmp_path)
    monkeypatch.setattr(t,'bindings',lambda:{'kernel':'test'})
    def auth(directory,member,**kwargs):
        rng=np.random.default_rng(7 if member=='S_R7' else 6)
        values=rng.integers(0,20,(32,12))/20
        rec={'failures':{member:{'0':'failed'} if failure=='fallback' else {}},
             'r7_identity':{'worker_args':[] if failure=='flip' else ['--flip-tta']}}
        return ids[::-1] if failure=='ids' else ids,None,values,rec
    monkeypatch.setattr(candidate,'authenticated_pr7',auth)
    if failure=='none': assert t.real_output_replay()['status']=='PASS_REAL_OUTPUT_REPLAY'
    else:
        with pytest.raises(ValueError): t.real_output_replay()
