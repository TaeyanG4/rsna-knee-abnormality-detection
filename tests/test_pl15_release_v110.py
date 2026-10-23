"""Offline PL transport, fail-closed admission and exact insurance tests."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import pl15_transport_v110 as t, pl15_runtime_v110 as r
from tools import vnext_r7_pl_v1 as pl


def evidence(count,shards,mode=None):
    timings=[]
    for i in range(shards):
        ids=[str(j) for j in range(i,count,shards)]
        timings.append(dict(study_seconds={u:1. for u in ids},setup_seconds=1.,wall_seconds=10.,
            load_failures={},telemetry_errors=[],peak_reserved_bytes=1000,device='cuda:0',gpu_name='Tesla T4'))
    out=dict(status='PASS',manifest_sha256='seal',k=36,flip=True,full_shape=True,folds=5,clean_oof=False,
        studies=count,timings=timings,parities=[dict(input_studies=len(x['study_seconds']),inference_checks=2,frozen_tta_checks=1,max_abs_error=0.) for x in timings],
        exits=[dict(code=0,cleanup_error=None)]*shards,kind='local_cuda',device='cuda:0',production_provenance=True,diagnostic_only=False)
    if mode:
        out.update(mode=mode,variant='a',image_sha256=t.IMAGE,publication={'status':'COMPLETE_PL'},
            cleanup=dict(gpu_memory_released=True,descendants_remaining=0),
            base_fault_suite={'status':'PASS_DIAGNOSTIC_ONLY'},
            pl_faults=[dict(fault=f,studies=32,complete_csv=True,base_byte_identical=True,
                cleanup=dict(gpu_memory_released=True,descendants_remaining=0)) for f in ('kill','timeout','corrupt')])
    return out


def qualification():
    return dict(status='PASS',manifest_sha256='seal',local_cuda=evidence(3,1),visible=evidence(3,2,'visible'),
        n32=evidence(32,2,'n32'),healthy_rates={x:1. for x in ('own','Q_A5q','R','S_R7','PL-0','PL-1')})


def test_complete_evidence_admits_only_a():
    assert r.admit({'diagnostic_only':False},'seal',qualification())['variant']=='a'


@pytest.mark.parametrize('case',range(16))
def test_partial_or_fabricated_qualification_rejected(case):
    q=qualification()
    if case==0:q['local_cuda']['timings']=[]
    if case==1:q['n32']['parities']=[]
    if case==2:q['local_cuda']['exits']=[]
    if case==3:q['local_cuda']['device']='cpu'
    if case==4:q['visible']['studies']=1
    if case==5:q['n32']['timings'][0]['study_seconds']={'only-one':1.}
    if case==6:q['n32']['timings'][0]['peak_reserved_bytes']=11*1024**3
    if case==7:q['n32']['exits'][0]={'code':1,'cleanup_error':None}
    if case==8:q['local_cuda']['diagnostic_only']=True
    if case==9:q['n32']['variant']='b'
    if case==10:q['n32']['timings'][0]['study_seconds']['0']=100.
    if case==11:q['n32']['pl_faults'].pop()
    if case==12:q['visible']['manifest_sha256']='old'
    if case==13:q['visible']['timings'][0]['telemetry_errors']=['failed']
    if case==14:q['healthy_rates'].pop('R')
    if case==15:q['local_cuda']['parities'][0]['input_studies']=1
    with pytest.raises(ValueError):r.admit({'diagnostic_only':False},'seal',q)


@pytest.mark.parametrize('name',['../x','H:/model.pt',r'x\y','/absolute'])
def test_transport_path_escape_rejected(tmp_path,name):
    with pytest.raises(ValueError):t.safe_path(tmp_path,name)


def test_manifest_tamper_rejected_before_deserialization(tmp_path):
    t.write(tmp_path/'manifest.json',b'not json')
    with pytest.raises(ValueError,match='pinned'):t.verify(tmp_path,'old')


def journals(tmp_path,broken=False):
    ids=['one','two','three'];members={'PL-S-s42-f'+str(i):str(i) for i in range(5)}
    identity=dict(version=pl.VERSION,registration='s',run_id='run',branch='own',members=members)
    journal=pl.Journal(tmp_path/'own',ids,identity)
    for i,u in enumerate(ids):
        for m in members:
            journal.record(u,m,None if broken else np.full(12,.2+i*.2),error='broken' if broken else None)
    journal.finish()
    return ids,[dict(ids=ids,path=tmp_path/'own',identity=identity)]


@pytest.mark.parametrize('variant',['a','b'])
@pytest.mark.parametrize('fault',['journal','exit','export','missing'])
def test_failed_pl_preserves_exact_base_csv(tmp_path,monkeypatch,variant,fault):
    ids,shards=journals(tmp_path,broken=fault=='journal')
    baseline=b'StudyInstanceUID,target\none,0.12\ntwo,0.33\nthree,0.9\n'
    raw={k:np.full((3,12),.4) for k in ('R','S_R7','EXP143v2','EXP155','Q_A5q')}
    destination=tmp_path/'submission.csv'
    exits=[dict(code=1 if fault=='exit' else 0,cleanup_error=None)]
    if fault=='missing':shards[0]['path']=tmp_path/'absent'
    if fault=='export':
        original=r.t.write
        def write(path,data,*args,**kw):
            if Path(path)==destination and data!=baseline:raise PermissionError('busy optional output')
            return original(path,data,*args,**kw)
        monkeypatch.setattr(r.t,'write',write)
    out=r.publish(baseline,ids,raw,{},shards,destination,variant,exits)
    assert out['fallback'] and destination.read_bytes()==baseline


def test_real_journals_both_variants_match_frozen_compositor(tmp_path):
    from tools import exp169_release_c85r7_pl15_v140 as c
    ids,shards=journals(tmp_path)
    raw={k:np.asarray([[.9]*12,[.1]*12,[.4]*12],np.float32) for k in ('R','S_R7','EXP143v2','EXP155','Q_A5q')}
    for variant in ('a','b'):
        dest=tmp_path/(variant+'.csv')
        assert r.publish(b'base',ids,raw,{},shards,dest,variant)['status']=='COMPLETE_PL'
        assert dest.read_bytes()==c.base.p.csv_bytes(ids,c.blend(ids,raw,r.collect(ids,shards),variant,{}))


def test_deadline_inherited_by_actual_child(tmp_path):
    clock=r.d.Deadline.start(30,env={})
    output=tmp_path/'clock.json'
    command=[sys.executable,'-c',
        'import json,os,sys;json.dump({k:v for k,v in os.environ.items() if k.startswith("KNEE_RELEASE_")},open(sys.argv[1],"w"))',str(output)]
    with r.d.Supervisor(clock) as supervisor:
        child=supervisor.run(command,env=clock.environment({}),label='cpu-test')
    assert child.returncode==0 and child.cleanup_error is None
    assert json.loads(output.read_text())[r.d.START_ENV]==repr(clock.started)


def test_review_rejects_missing_receipts(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'validate',lambda _:dict(sources={}))
    t.dump(tmp_path/'registration.json',{})
    t.dump(tmp_path/'receipt.json',dict(status='HOLD',kind='suite'))
    with pytest.raises(ValueError):t.review(tmp_path,tmp_path/'receipt.json',None,'reviewer',tmp_path/'review.json')
    assert not (tmp_path/'review.json').exists()


@pytest.mark.parametrize('fault',['missing','source','receipt','hold','none'])
def test_independent_review_is_source_and_receipt_bound(tmp_path,monkeypatch,fault):
    sources={n:'source-seal' for n in t.OWN_SOURCES}
    monkeypatch.setattr(t,'validate',lambda _:dict(sources=sources))
    t.dump(tmp_path/'registration.json',{})
    receipts={}
    for kind in ('suite','real_cpu_smoke'):
        path=tmp_path/(kind+'.json')
        t.dump(path,dict(status='PASS',kind=kind,sources=sources,registration_sha256=t.sha(tmp_path/'registration.json')))
        receipts[kind]=t.sha(path)
    report=dict(status='PASS',scope='independent_source_review_only',reviewer='external',sources=sources,receipts=receipts)
    if fault=='source':report['sources']={}
    if fault=='receipt':report['receipts']={}
    if fault=='hold':report['status']='HOLD'
    artifact=tmp_path/'independent.json';t.dump(artifact,report)
    args=(tmp_path,tmp_path/'suite.json',tmp_path/'real_cpu_smoke.json','external',tmp_path/'review.json',None if fault=='missing' else artifact)
    if fault=='none':assert t.review(*args)['independent']
    else:
        with pytest.raises(ValueError):t.review(*args)
        assert not (tmp_path/'review.json').exists()


@pytest.mark.parametrize('mode',['n32','submit'])
def test_b_cannot_be_promoted(tmp_path,mode):
    with pytest.raises(ValueError,match='visible-only'):t.notebook(tmp_path,tmp_path/'kernel',mode,'b')


def test_kernel_bootstrap_deadline_and_private_metadata(tmp_path,monkeypatch):
    t.write(tmp_path/'manifest.json',b'{}')
    monkeypatch.setattr(t,'verify',lambda *a,**k:dict(files=[dict(path='runtime.blob',sha256='archive')]))
    t.notebook(tmp_path,tmp_path/'kernel','visible')
    code=''.join(t.read(tmp_path/'kernel/release.ipynb')['cells'][0]['source'])
    compile(code,'notebook','exec')
    assert code.index('time.monotonic()')<code.index('import hashlib')
    assert str(t.ROOT) not in code
    meta=t.read(tmp_path/'kernel/kernel-metadata.json')
    assert meta['is_private'] is True and meta['enable_internet'] is False and t.SLUG in meta['dataset_sources']


def test_native_job_remaps_every_runtime_path(tmp_path):
    members=[dict(id='PL-S-s42-f'+str(i),path='checkpoints/f'+str(i)+'.pt',sha256=str(i),training_receipt={'path':'receipts/f'+str(i)+'.json'}) for i in range(5)]
    manifest=dict(members=members,science={'registration_sha256':'s'},diagnostic_only=False)
    j=r.job(tmp_path,manifest,tmp_path/'code',tmp_path/'input',['one'],[],tmp_path/'out','r')
    assert all(Path(e['path']).is_relative_to(tmp_path) and Path(e['training_receipt']['path']).is_relative_to(tmp_path) for e in j['entries'])


def test_projection_rejects_absent_and_slow_evidence():
    with pytest.raises(ValueError):r.projection([])
    assert r.projection([dict(setup_seconds=5,study_seconds={'x':20})])['status']=='HOLD'


def test_frozen_r7_source_instrumentation_and_actual_counter(tmp_path,monkeypatch):
    import zipfile
    from tools import c85r7_v150_safety as safety
    with zipfile.ZipFile(t.PARENT/'base-v122-runtime.zip') as z:
        names=[n for n in z.namelist() if n=='exp169_r7_src/compact_infer.py']
        assert len(names)==1
        code=z.read(names[0]).decode()
    patched=r.compact_progress(code)
    assert patched.count("_pl15_progress('S_R7:csv-row', committed=True)")==1
    marker=tmp_path/'progress.json';monkeypatch.setenv('C85R7_V150_PROGRESS',str(marker))
    source='def write(wr,fh):\n    wr.writerow([1])\n    fh.flush()\n'
    import csv,io
    ns={};exec(r.compact_progress(source),ns)
    stream=io.StringIO();ns['write'](csv.writer(stream),stream)
    assert safety.committed_records(marker)==1 and stream.getvalue().strip()=='1'
