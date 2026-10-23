"""Offline HB177 regressions. Synthetic approvals stay in fixed pytest scratch."""
import copy
import csv
import json
import os
import sys
from pathlib import Path
import types
from unittest.mock import patch

import numpy as np
import pytest

from tools import hb177_postfit_v1 as b
from tools import hb177_postfit_smoke_v1 as smoke
from tools import hb177_gold58_v1 as gold
from tools import hb177_release_admission_v1 as admission
from tools import hb177_n32_v1 as n32
from tools import strict_variant_gate as base
from tools import strict_variant_gate_hb177 as gate
from tools import kaggle_submit_once as submit
from test_strict_variant_gate import dispatched, put


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    import torch
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda: pytest.fail('CUDA forbidden'))


def test_real_identity_and_current_registration():
    assert b.validate()['kernel_sha256']=='d11f2e5700a20067d5a54b55ba5803cdde485f3d3238c71876b5b9fe04564c5a'
    assert gold.validate()['policy']['weak_indices']==[3,5,6,8]
    assert b.read(b.OUT/'kill-registration.json')['decision']==b.DECISION


def test_extraction_reuse_and_gpu_separation(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'SCRATCH',tmp_path)
    a=smoke.extract(b.OUT,b.DATASET,'HB177')
    before={p.name:b.sha(p) for p in a['_assets'].iterdir()}
    again=smoke.extract(b.OUT,b.DATASET,'HB177')
    assert before=={p.name:b.sha(p) for p in again['_assets'].iterdir()}
    gpu=smoke.extract(b.OUT,b.DATASET,'HB177','cuda:0')
    assert gpu['_assets']!=a['_assets']
    (a['_assets']/'foreign.txt').write_text('foreign')
    with pytest.raises(ValueError,match='inventory'): smoke.extract(b.OUT,b.DATASET,'HB177')


def runtime():
    return dict(kind='KAGGLE_HIDDEN_RUNTIME',kernel='taeyangg4/knee-hb156-cnxs-l2-v1',
        kernel_version=1,submission_ref='57198732',source='SYNTHETIC TEST ONLY',hidden_seconds=1000)


@pytest.mark.parametrize('field,value',[('hidden_seconds',float('nan')),('hidden_seconds',True),
    ('hidden_seconds',-1),('kernel_version',True),('kernel','wrong'),('source',''),('submission_ref','wrong')])
def test_runtime_invalid(field,value):
    rec=runtime(); rec[field]=value
    with pytest.raises(ValueError): admission.runtime_fields(rec)


@pytest.mark.parametrize('reviewer',[None,'','hb177-postfit'])
def test_fit_reviewer_rejects(tmp_path,reviewer):
    report=tmp_path/'fit.md'; report.write_text('Verdict: PASS\n'+b.CHECKPOINT_SHA)
    with pytest.raises(ValueError): admission.fit_bind(report,reviewer)


def test_fit_verdict_and_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'OUT',tmp_path)
    report=tmp_path/'fit.md'; report.write_text('Verdict: NOT PASS\n'+b.CHECKPOINT_SHA)
    with pytest.raises(ValueError): admission.fit_bind(report,'independent')
    report.write_text('Verdict: PASS\n'+b.CHECKPOINT_SHA)
    assert admission.fit_bind(report,'independent')['status']=='FIT_REVIEW_BOUND'
    assert b.read(tmp_path/'fit-result-review-binding.json')['sha256']==b.sha(report)


@pytest.fixture
def bound(tmp_path,monkeypatch):
    original=base.dispatch
    def isolated(path,inputs):
        scope=base.Inputs(tmp_path)
        result=original(base.local_path(path,inputs.root),scope)
        inputs.data.update(scope.data)
        return result
    monkeypatch.setattr(base,'dispatch',isolated)
    source=tmp_path/'source'
    # Immutable bytes hardlinked; tests only mutate new evidence, never sources.
    for folder in ('kernel','dataset'):
        for p in (b.OUT/folder).iterdir():
            dest=source/folder/p.name; dest.parent.mkdir(parents=True,exist_ok=True); dest.hardlink_to(p)
    for p in b.OUT.iterdir():
        if p.is_file() and (p.suffix=='.json' or p.name=='full-diff-vs-parent.txt'):
            dest=source/p.name; dest.write_bytes(p.read_bytes())
    report=tmp_path/'fit.md'; report.write_text('Verdict: PASS\n'+b.CHECKPOINT_SHA)
    put(source/'fit-result-review-binding.json',dict(verdict='PASS',reviewer='independent-fit',
        report=report.relative_to(b.ROOT).as_posix(),sha256=b.sha(report),checkpoint_sha256=b.CHECKPOINT_SHA))
    ev=tmp_path/'runtime.json'; put(ev,runtime())
    cpu=b.read(source/'cpu-smoke.json'); ratio=cpu['serving_ratio']
    put(source/'runtime-binding.json',dict(version=admission.VERSION,evidence=ev.relative_to(b.ROOT).as_posix(),
        evidence_sha256=b.sha(ev),cpu_smoke_sha256=b.sha(source/'cpu-smoke.json'),
        hidden_seconds=1000,serving_ratio=ratio,projected_seconds=1000*ratio))
    g,ns=gate.context()
    info=gate.kernel(source,g.Inputs(b.ROOT),g)
    receipt=dispatched(tmp_path,'hb177-test-builder',tmp_path/'builder.md',
        ('Kernel-SHA256: '+g.sha(g.encoded(info[2]['kernel_files']))+'\n').encode())
    output=tmp_path/'bound'; review=tmp_path/'review.md'
    result=ns['bind'](source,b.PARENT,output,receipt,review,True,b.ROOT,g)
    assert result['status']=='REVIEW_REQUIRED'
    allow=b.read(output/'allowlist.json')
    dispatched(tmp_path,'hb177-test-reviewer',review,(
        'Reviewer-Lane: hb177-test-reviewer\nSHA256SUMS: '+allow['stage_sha256sums']+
        '\nAllowlist-SHA256: '+b.sha(output/'allowlist.json')+'\nVERDICT: PASS\n').encode())
    return source,dict(stage=output,parent=output/'parent',allow_path=output/'allowlist.json',review=review,root=b.ROOT)


def test_bound_gate_consumer_and_mutations(bound,monkeypatch):
    source,args=bound
    original=base.kernel
    g,ns=gate.context(admission=True)
    result=ns['evaluate'](**args,g=g)
    assert result['status']=='PASS',result['errors']
    assert base.kernel is original
    gatepath=submit.ROOT/'.local/state/gates'/ (b.SLUG.split('/')[1]+'.json')
    read=submit._json_evidence
    monkeypatch.setattr(submit,'_json_evidence',lambda p,h: result if p==gatepath else read(p,h))
    assert submit._strict_gate(b.SLUG,{'gate_sha256':'fixture'},{str(gatepath):'fixture'})==result
    for filename,field,value in [('fit-result-review-binding.json','verdict','FAIL'),
        ('runtime-binding.json','projected_seconds',40000),('kill-registration.json','clean_oof',True),
        ('cpu-smoke.json','status','FAIL')]:
        p=source/filename; before=p.read_bytes(); rec=b.read(p); rec[field]=value; put(p,rec)
        rejected=ns['evaluate'](**args,g=g)
        assert rejected['status']=='FAIL',(filename,rejected)
        p.write_bytes(before)


def test_missing_release_admission_fails():
    if not (b.OUT/'fit-result-review-binding.json').exists():
        with pytest.raises((OSError,ValueError)): admission.check(b.OUT,base.Inputs(b.ROOT),base)


def test_gate_pin_cli_consumes_mode(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['strict_variant_gate_hb177.py','pin','--stage',str(b.OUT)])
    assert gate.main()==0
    assert json.loads(capsys.readouterr().out)['status']=='PINNED_NOT_ADMITTED'


def test_paired_bootstrap_alignment_and_invalid():
    y=np.tile(np.arange(58)[:,None]%2,(1,12))
    a=y*.8+.1; arms={'HB177':a,'HB156':a.copy(),'R7':1-a}
    result=gold.paired(y,arms,draws=20)
    assert result==gold.paired(y,arms,draws=20)
    assert result['paired']['HB156']['delta']==[0,0]
    assert result['paired']['R7']['delta']==[1,1]
    for value in (np.nan,1.1,-.1):
        bad=a.copy(); bad[0,0]=value
        with pytest.raises(ValueError): gold.paired(y,{**arms,'HB177':bad},draws=2)
    with pytest.raises(ValueError): gold.paired(y,{k:v for k,v in arms.items() if k!='R7'},draws=2)


def test_n32_exact_candidate_and_selector(tmp_path):
    before=b.sha(b.OUT/'kernel/release.ipynb'); nb,meta=n32.payload()
    assert meta['id']==b.SLUG+'-stress-n32' and 'FAST=False' in nb['cells'][0]['source']
    assert 'find_test_root = lambda: str(_stress_root)' in nb['cells'][5]['source']
    assert b.sha(b.OUT/'kernel/release.ipynb')==before
    ns={}; exec(n32.selector(),ns)
    root=tmp_path/'corpus'; root.mkdir(); rows=[]
    for i in range(32):
        uid=str(i); p=root/'train_series'/uid/'series'; p.mkdir(parents=True)
        (p/'slice.dcm').write_bytes(b'SYNTHETIC_FILENAME_ONLY')
        rows.append([uid,'series','Sagittal','True'])
    ns['table'](root/'train_series.csv',ns['COLS'],rows)
    selected=ns['prepare'](root,tmp_path/'selected',n=32)
    assert len(set(selected))==32


@pytest.mark.parametrize('memory_failure',['','synthetic memory failure'])
def test_gold_supervisor_cleanup_and_finalize(tmp_path,monkeypatch,memory_failure):
    from tools import hb149_monitor_v125 as monitor,c85r7_r224_timing as guards,c85r7_flipall_timing as memory,job_runtime
    import psutil
    monkeypatch.setattr(gold,'WORK',tmp_path)
    monkeypatch.setattr(gold,'admit',lambda *a,**k:None)
    parent=types.SimpleNamespace(pid=123)
    process=types.SimpleNamespace(pid=456,parents=lambda:[parent],children=lambda **k:[],
        memory_info=lambda:types.SimpleNamespace(private=1024,rss=1024))
    monkeypatch.setattr(psutil,'Process',lambda *a:process)
    monkeypatch.setattr(job_runtime,'job_status',lambda *a,**k:dict(state='running',pid=123))
    monkeypatch.setattr(gold.subprocess,'run',lambda *a,**k:types.SimpleNamespace(stdout=''))
    monkeypatch.setattr(memory,'gpu_python_pids',lambda *a:[])
    monkeypatch.setattr(memory,'desktop_snapshot',lambda:dict(test=True))
    calls=[]
    class Telemetry:
        def __init__(self,*a):pass
        def start(self):calls.append('start')
        def close(self):calls.append('close')
        def snapshot(self,**k):return dict(complete=True,memory_failure=memory_failure,latest_gpu={})
    monkeypatch.setattr(guards,'_TimingTelemetry',Telemetry)
    polls=iter([None,0])
    child=types.SimpleNamespace(pid=456,returncode=0,poll=lambda:next(polls,0))
    monkeypatch.setattr(gold.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(monitor,'_WindowsJob',lambda p:'owner')
    monkeypatch.setattr(monitor,'_cleanup',lambda *a:calls.append('cleanup') or [])
    monkeypatch.setattr(guards,'watchdog',lambda *a:calls.append('watchdog'))
    monkeypatch.setattr(gold.time,'sleep',lambda *a:None)
    monkeypatch.setattr(gold,'finalize',lambda t:dict(status='MOCK_COMPLETE'))
    if memory_failure:
        with pytest.raises(ValueError,match='synthetic memory failure'):gold.run()
        assert b.read(tmp_path/'gpu/failure.json')['status']=='FAILED'
        assert calls==['start','cleanup','close']
    else:
        assert gold.run()['status']=='MOCK_COMPLETE'
        assert calls==['start','watchdog','cleanup','close']
    with pytest.raises(ValueError,match='immutable'):gold.run()


def test_gold_and_visible_offline_binding(tmp_path,monkeypatch):
    report=tmp_path/'review.md'; report.write_text('Verdict: PASS\n')
    monkeypatch.setattr(gold,'validate',lambda:None)
    monkeypatch.setattr(gold,'review_binding',lambda:dict(test='SYNTHETIC'))
    monkeypatch.setattr(b,'OUT',tmp_path)
    assert admission.gold_bind(report,'independent')['status']=='GOLD_REVIEW_BOUND'
    assert b.read(tmp_path/'gold-independent-review.json')['report_sha256']==b.sha(report)
    with pytest.raises(ValueError):admission.visible_bind(report,tmp_path,0)
    monkeypatch.setattr(admission,'ROOT',tmp_path)
    bound=tmp_path/'bound'; receipt=tmp_path/'visible'/'receipt.json'
    put(bound/'registration.json',dict(kernel_id=b.SLUG))
    put(receipt,dict(test='SYNTHETIC'))
    gate_path=tmp_path/'.local/state/gates'/(b.SLUG.split('/')[1]+'.json')
    put(gate_path,dict(stage=str(bound)))
    calls=[]
    monkeypatch.setattr(submit,'_strict_gate',lambda *args:calls.append(args))
    assert admission.visible_bind(receipt,bound,7)['status']=='VISIBLE_TRANSPORT_BOUND'
    value=b.read(receipt.parent/'submission-binding.json')
    assert value['receipt_sha256']==b.sha(receipt) and value['kernel_version']==7
    assert value['registration_sha256']==b.sha(bound/'registration.json') and len(calls)==1
    with pytest.raises(ValueError,match='immutable'):admission.visible_bind(receipt,bound,8)


def test_real_parent_fast_output_receipt_and_rejections(tmp_path):
    parent=b.ROOT/'.local/scratch/slot-b/knee-hb156-cnxs-l2-v1-visible'
    log=parent/'knee-hb156-cnxs-l2-v1.log'
    csv_path=tmp_path/'submission.csv'; csv_path.write_bytes((parent/'submission.csv').read_bytes())
    status=tmp_path/'wait.log'
    status.write_text('pushed; real ref = '+b.SLUG+' (url)\n'+b.SLUG+" after 1 min: has status 'complete'\n")
    receipt=tmp_path/'visible.json'
    result=admission.visible_receipt(receipt,log,csv_path,status,1)
    assert result['rows']==3 and result['inference_performed'] is False
    assert b.read(receipt)['fallbacks']==dict(placeholder_errors=0)
    rows=list(csv.reader(csv_path.read_text().splitlines()))
    rows[1][1]='nan'
    with csv_path.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError,match='finite 0.5'):
        admission.visible_receipt(receipt,log,csv_path,status,1)
    csv_path.write_bytes((parent/'submission.csv').read_bytes())
    status.write_text("other/kernel after 1 min: complete\n")
    with pytest.raises(ValueError,match='exact pushed slug'):
        admission.visible_receipt(receipt,log,csv_path,status,1)
