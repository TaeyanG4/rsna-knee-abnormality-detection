"""Independent arithmetic, real helper ABI, offline package and closed admission."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

from tools import hb116_vnext_release as h


@pytest.fixture
def reg():
    return h.read(h.REG/'consults/astra3-registration-draft.json')


def reference(reg, raw, candidate, rx, s43):
    """Pandas ranks + explicit study survivor loops, independent from release helpers."""
    n=len(next(iter(raw.values())))
    def get(k): return raw.get(k,np.full((n,12),np.nan))
    def mean(arr,weights=None,fp32=False):
        result=np.full((n,12),np.nan,dtype=np.float32 if fp32 else np.float64)
        for i in range(n):
            keep=[j for j,a in enumerate(arr) if np.isfinite(a[i]).all()]
            if keep:
                w=np.asarray([weights[j] if weights else 1 for j in keep],dtype=result.dtype)
                result[i]=sum(np.asarray(arr[j][i],dtype=result.dtype)*w[k] for k,j in enumerate(keep))/sum(w)
        return result
    def rank(a):
        ok=np.isfinite(a).all(1)
        out=np.full((n,12),np.nan)
        out[ok]=pd.DataFrame(a[ok]).rank(method='average',pct=True).to_numpy()
        return out
    family={}
    for f in ('R50','C4','S'):
        es=[e for e in reg['Q']['members'] if e['family']==f]
        if f=='R50':
            a=mean([mean([get(e['id']) for e in es if e['fold']==fold],fp32=True) for fold in range(5)],fp32=True)
        else: a=mean([get(e['id']) for e in es],fp32=True)
        family[f]=rank(a)
    q=mean(list(family.values()))
    a=mean([get('EXP143v2')]+([get('S43')] if s43 else []))
    o=mean([rank(a),rank(get('EXP155')),rank(q)],[17,17,12] if candidate=='A' else [11,11,18])
    rxv=get('R') if rx=='R' else mean([get('R2-s43'),get('R2-s44')],fp32=True)
    p=mean([rank(rxv),rank(get('S6'))])
    out=mean([rank(p),rank(o)],[.7,.3])
    out[~np.isfinite(out).all(1)]=.5
    return out.astype(np.float32)


@pytest.mark.parametrize('candidate',['A','B'])
@pytest.mark.parametrize('rx',['R','R2'])
@pytest.mark.parametrize('s43',[False,True])
@pytest.mark.parametrize('failures',[False,True])
def test_independent_formula_survivors_ties(reg,candidate,rx,s43,failures):
    rng=np.random.default_rng(116)
    raw={k:rng.integers(0,8,(13,12)).astype(np.float32)/8 for k in h.roster(reg,rx,s43)}
    if failures:
        for i,(k,a) in enumerate(raw.items()):
            a[i%5]=np.nan  # member/seed/fold survivors
            if k.startswith('S-'): a[5]=np.nan  # whole MIL-S family
            if k not in ('S6','R','R2-s43','R2-s44'): a[6]=np.nan  # O branch
            else: a[7]=np.nan  # P branch
            a[8]=np.nan  # both branches
    got,_=h.blend(reg,raw,candidate,rx,s43)
    np.testing.assert_array_equal(got,reference(reg,raw,candidate,rx,s43))
    if failures: np.testing.assert_array_equal(got[8],np.full(12,.5,np.float32))


@pytest.mark.parametrize('fault',['member','family','branch','nan','range','extra','empty'])
def test_fail_closed_qualification(reg,fault):
    raw={k:np.full((3,12),.2,np.float32) for k in h.roster(reg)}
    assert h.qualify(reg,raw)
    if fault=='member': raw.pop(next(iter(raw)))
    if fault=='family': raw={k:v for k,v in raw.items() if not k.startswith('S-')}
    if fault=='branch': raw={k:v for k,v in raw.items() if k not in ('R','S6')}
    if fault=='nan': raw['R'][0,0]=np.nan
    if fault=='range': raw['R'][0,0]=2
    if fault=='extra': raw['R2-s43']=raw['R']
    if fault=='empty': raw={k:v[:0] for k,v in raw.items()}
    with pytest.raises(ValueError): h.qualify(reg,raw)


def test_real_helpers_and_sealed_bindings(reg):
    from tools import exp155_own0949_oai_release_v12 as own
    from tools import exp152_own0949_release as public
    from tools import hb105_serving_v101 as hb
    from tools import exp169_release_v6 as v6
    assert callable(own.checked_model) and callable(own._shared.input_record)
    assert callable(public.checked_model) and callable(public.input_record)
    assert callable(hb.load_model) and callable(v6.anchor_reader)
    entries=h.bind_q(reg,h.ROOT/h.Q_AUTH)
    assert len(entries)==25
    assert all(e['spec_sha256'] for e in entries if e['family']=='S')
    h.pin(h.REG/'REGISTRATION.md',{'sha256':h.REG_SHA})
    h.pin(h.REG/'consults/astra3-registration-draft.json',{'sha256':h.DRAFT_SHA})
    assert reg['OWN']['A_members'][0]['sha256']=='deeb0ab4d578a9af9df8288a739eaf1df11322b24373aeaafa9bc7ef2b04a093'


def test_mil_preprocessing_tta_matches_frozen_training():
    import torch
    from training.arms import mil
    sys.path.insert(0,str(h.LEGACY))
    from raptor_common import select_centers, CONTRACTS
    # Use the actual 64-slot corpus specification; no study/label data.
    spec=dict(CONTRACTS['raptor64_336_c140_s02'],depth=64)
    mask=np.ones(64,bool)
    centers=select_centers(mask,64,36,spec,'within-slot-valid')
    vol=np.random.default_rng(2).integers(0,255,(64,336,336),dtype=np.uint8)
    x=h.q_inputs(vol,centers,'S')
    expected=(mil.scale_windows(vol,centers)-np.asarray([.485,.456,.406],np.float32).reshape(3,1,1))/np.asarray([.229,.224,.225],np.float32).reshape(3,1,1)
    np.testing.assert_array_equal(x,expected)
    from types import SimpleNamespace
    corpus=SimpleNamespace(spec=spec,mask=lambda uid:mask)
    actual=x.copy(); flags=np.asarray(h.c.a172.base.plane_flags(centers,spec),bool)
    actual[flags]=actual[flags,:,:,::-1]
    np.testing.assert_array_equal(actual,mil.flip_inputs(torch.from_numpy(x)[None],['test'],corpus,36)[0].numpy())


def test_package_imports_without_project_fallback(tmp_path):
    files=h.source_payloads()
    blob,pins=h.package(files)
    with zipfile.ZipFile(io.BytesIO(blob)) as z: z.extractall(tmp_path)
    script="""import sys
sys.path.insert(0,sys.argv[1])
from tools import hb116_vnext_release as h
sys.path.insert(0,str(h.ROOT/'legacy'))
from tools import hb105_serving_v101 as hb
from tools import exp155_own0949_oai_release_v12 as own
from raptor_dicom import DicomCorpus, DirectoryDicomSource
from raptor_common import build_model
np,torch,mil,serving=hb.imports()
assert callable(own._shared.input_record)
assert 'site-packages' not in str(h.__file__)
print('STANDALONE_IMPORT_PASS')
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',script,str(tmp_path)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'STANDALONE_IMPORT_PASS' in result.stdout


def test_timing_truncated_journal_and_s6_anchor(tmp_path):
    folder=tmp_path/'S/worker0'; folder.mkdir(parents=True)
    h.dump(folder/'timing.json',dict(study_seconds={'a':2,'b':3,'c':4},setup_seconds=1,gpu_name='Tesla T4'))
    (folder/'predictions.jsonl').write_text('{"member":"S-test","seconds":2}\n{"member":')
    h.dump(tmp_path/'public/p956-run/s6/controller/submission_receipt.json',{'workers':[
        dict(status='PASS',stats={},per_study=[['a',3,1,2,3,4],['b',3,2,3,4,5],['c',3,3,4,5,6]])]})
    result=h.timings(tmp_path,{'S':{'seconds':10},'S6':{'seconds':45}},3)
    assert result['members']['S-test']['count']==1
    assert result['families']['S6']['study_seconds']==[10,14,18]
    assert result['families']['S']['upper95_study_seconds']>3


def target_fixture(tmp_path,reg,diagnostic=False):
    n=32 if diagnostic else 3
    folder=tmp_path/('stress' if diagnostic else 'visible'); folder.mkdir()
    ids=[str(i) for i in range(n)]
    raw={k:np.random.default_rng(i).random((n,12),dtype=np.float32) for i,k in enumerate(h.roster(reg))}
    h.save_raw(folder/'hb116-member-raw.npz',ids,raw)
    cfg=dict(version=h.VERSION,candidate='A',constants=[17,17,12],rx='R',s43=True,stress=diagnostic)
    value,_=h.blend(reg,raw)
    pred=folder/('diagnostic-hb116.csv' if diagnostic else 'submission.csv')
    pred.write_bytes(h.c.p.csv_bytes(ids,value))
    fam={k:dict(study_seconds=[1.]*n,upper95_study_seconds=1.,setup_upper_seconds=2.) for k in ('NATIVE','S6')}
    fam['NATIVE']['memory']=[dict(peak_allocated_bytes=4e9,peak_reserved_bytes=5e9,total_device_bytes=16e9)]*2
    rec=dict(status='COMPLETE_HB116',version=h.VERSION,diagnostic=diagnostic,registration_sha256=h.REG_SHA,draft_sha256=h.DRAFT_SHA,
        runtime_sha256=h.sha(Path(h.__file__)),native_runtime_sha256=h.sha(h.ROOT/'tools/hb116_native_runtime.py'),device='2xT4',gpu_names=['Tesla T4']*2,image=h.c.p.IMAGE,
        studies=n,ids=ids,normal_visible_qualified=True,missing={k:0 for k in raw},phases={k:{'exit_codes':[0] if k=='S6' else [0,0]} for k in fam},
        s6_telemetry={'worker_exit_codes':{'0':0,'1':0}},config=cfg,candidate='A',rx='R',s43=True,
        config_sha256=h.hashlib.sha256((json.dumps(cfg,sort_keys=True)+'\n').encode()).hexdigest(),
        raw_sha256=h.sha(folder/'hb116-member-raw.npz'),prediction_sha256=h.sha(pred),fixed_seconds=2.,
        timing={'families':fam,'errors':[]})
    rec['projection']=h.projection(rec)
    h.dump(folder/'hb116-receipt.json',rec)
    return folder,rec,raw


def test_target_runtime_parity_gate(tmp_path,reg):
    v,_,_=target_fixture(tmp_path,reg)
    s,_,_=target_fixture(tmp_path,reg,True)
    result=h.admission_gates(v,s,v/'hb116-member-raw.npz')
    assert result['status']=='TECHNICAL_GATES_PASS'
    assert result['visible_projection']['projected_seconds']<28800


@pytest.mark.parametrize('fault',['cpu','cohort','child','missing','source','timing','raw','csv','config','empty_phases','empty_exits','empty_s6','native_source','memory'])
def test_target_admission_fails_closed(tmp_path,reg,fault):
    v,rec,raw=target_fixture(tmp_path,reg)
    if fault=='cpu': rec['device']='cpu'
    if fault=='cohort': rec['studies']=2
    if fault=='child': rec['phases']['NATIVE']['exit_codes']=[1]
    if fault=='missing': rec['missing'][next(iter(raw))]=1
    if fault=='source': rec['runtime_sha256']='0'*64
    if fault=='timing': rec['timing']['families'].pop('NATIVE')
    if fault=='raw': (v/'hb116-member-raw.npz').write_bytes(b'bad')
    if fault=='csv': (v/'submission.csv').write_bytes(b'bad')
    if fault=='config': rec['config']['constants']=[11,11,18]
    if fault=='empty_phases': rec['phases']={}
    if fault=='empty_exits': rec['phases']['NATIVE']['exit_codes']=[]
    if fault=='native_source': rec['native_runtime_sha256']='0'*64
    if fault=='memory': rec['timing']['families']['NATIVE']['memory'][0]['peak_reserved_bytes']=16e9
    if fault=='empty_s6': rec['s6_telemetry']['worker_exit_codes']={}
    h.dump(v/'hb116-receipt.json',rec)
    with pytest.raises(ValueError): h.target_receipt(v,False,reg)


def test_stress_never_accepted_with_submission(tmp_path,reg):
    folder,_,_=target_fixture(tmp_path,reg,True)
    (folder/'submission.csv').write_text('forbidden')
    with pytest.raises(ValueError,match='stress submission'): h.target_receipt(folder,True,reg)


@pytest.mark.parametrize('fault',['runtime','parity','unmatched'])
def test_admission_over_limit_or_unmatched(tmp_path,reg,fault):
    v,_,raw=target_fixture(tmp_path,reg)
    s,rec,_=target_fixture(tmp_path,reg,True)
    local=tmp_path/'local.npz'
    h.save_raw(local,[str(i) for i in range(3)],raw)
    if fault=='runtime':
        rec['fixed_seconds']=28801.
        rec['projection']=h.projection(rec)
        h.dump(s/'hb116-receipt.json',rec)
    elif fault=='parity':
        raw[next(iter(raw))]=raw[next(iter(raw))]+.003
        h.save_raw(local,[str(i) for i in range(3)],raw)
    else:
        # A valid, differently configured N32 receipt must still be rejected.
        rec['gpu_names']=['Tesla T4 alternate']*2
        h.dump(s/'hb116-receipt.json',rec)
    with pytest.raises(ValueError): h.admission_gates(v,s,local)


def test_smoke_reset_rejects_outside_fixed_scratch(tmp_path):
    with pytest.raises(ValueError,match='unsafe smoke output'): h.reset_smoke(tmp_path)


def test_cpu_s6_never_installs_packages(tmp_path,monkeypatch):
    monkeypatch.setattr(h.c.s6,'install_wheels',lambda *a:pytest.fail('CPU package install'))
    seen=[]
    monkeypatch.setattr(h.c.s6,'infer_studies',lambda *a,**k:seen.append(k))
    job=tmp_path/'job.json'
    h.dump(job,dict(ids=['x'],competition='test',bundle='bundle',code='code',anatomy='anatomy',output='output',source='source',device='cpu'))
    h.s6_worker(job)
    assert seen[0]['device']=='cpu' and seen[0]['workers']==0


def test_validate_rejects_external_embedded_config_drift(tmp_path):
    folder=tmp_path/'kernels/a'; folder.mkdir(parents=True)
    h.dump(tmp_path/'registration.json',{'draft_sha256':h.DRAFT_SHA})
    h.dump(tmp_path/'build.json',{'kernels':['kernels/a']})
    h.dump(tmp_path/'datasets/knee-hb116-vnext-weights-v1/manifest.json',{'files':[]})
    h.dump(folder/'kernel-metadata.json',h.c.metadata())
    cfg=dict(version=h.VERSION,candidate='A',constants=[17,17,12],rx='R',s43=True,stress=False)
    blob,pins=h.package({'tools/hb116_vnext_release.py':Path(h.__file__).read_bytes(),
                        'tools/hb116_native_runtime.py':(h.ROOT/'tools/hb116_native_runtime.py').read_bytes(),
                        'build-config.json':json.dumps(cfg).encode()})
    h.dump(folder/'release.ipynb',h.notebook(blob,pins,cfg))
    h.dump(folder/'source-pins.json',pins)
    h.dump(folder/'build-config.json',dict(cfg,candidate='B',constants=[11,11,18]))
    with pytest.raises(ValueError,match='embedded config differs'): h.validate(tmp_path)


@pytest.mark.parametrize('fault',[None,'raw','identity'])
def test_legacy_cpu_archive_requires_identical_pinned_journal(tmp_path,fault):
    ids=['one','two']; member='S-s42-f0'; digest='a'*64
    identity=dict(version=h.VERSION,members={member:digest})
    journal=h.c.Journal(tmp_path/'S/worker0',ids,identity)
    values=np.full((2,12),.25,dtype=np.float32)
    for uid,value in zip(ids,values): journal.record(uid,member,value,seconds=1.)
    if fault=='raw': values[0,0]=.75
    path=tmp_path/'raw.npz'; np.savez(path,ids=ids,**{member:values})
    registry={member:'b'*64 if fault=='identity' else digest}
    with np.load(path,allow_pickle=False) as data:
        if fault:
            with pytest.raises(ValueError): h.legacy_raw_journals(tmp_path,ids,data,['S'],registry)
        else: h.legacy_raw_journals(tmp_path,ids,data,['S'],registry)


def controller_fixture(tmp_path,reg):
    comp=tmp_path/'competition'; comp.mkdir()
    ids=['one','two','three']
    (comp/'sample_submission.csv').write_bytes(h.c.p.csv_bytes(ids,np.full((3,12),.5)))
    (comp/'test_series.csv').write_text(h.UID+'\n'+'\n'.join(ids)+'\n')
    entries=[]
    for family in ('R50','C4'):
        e=dict(next(e for e in reg['Q']['members'] if e['family']==family))
        path=tmp_path/(e['id']+'.pt'); path.write_bytes(b'test-only-model')
        e.update(resolved=str(path),bytes=path.stat().st_size,sha256=h.sha(path))
        entries.append(e)
    output=tmp_path/'output'; work=output/'hb116-run'; work.mkdir(parents=True)
    runtime=tmp_path/'runtime'
    h.dump(runtime/'build-config.json',dict(version=h.VERSION,candidate='A',constants=[17,17,12],rx='R',s43=True,stress=False))
    return comp,entries,output,work,runtime


def child_stub(command, calls, fault=None, corrupt=False):
    job=h.read(command[command.index('--job')+1]); calls.append(job['family'])
    first=job['family']=='R50'
    if first and fault=='launch': raise OSError('injected launch failure')
    identity=dict(job['identity'])
    if corrupt: identity['version']='wrong'
    journal=h.c.Journal(Path(job['output']),job['ids'],identity)
    # A dying child completed only one study. Keep that prediction.
    for i,uid in enumerate(job['ids'][:1] if first and fault else job['ids']):
        for e in job['entries']: journal.record(uid,e['id'],np.full(12,(i+1)/4),seconds=1.)
    class Child:
        killed=False
        def wait(self,timeout):
            if first and not self.killed:
                if fault=='timeout': raise subprocess.TimeoutExpired(command,timeout)
                if fault=='wait': raise OSError('injected wait failure')
                if fault=='nonzero': return 7
            return 0
        def kill(self): self.killed=True
    return Child()


@pytest.mark.parametrize('phase',['native','S6'])
@pytest.mark.parametrize('fault',['launch','timeout','wait','nonzero'])
def test_operational_failure_keeps_survivors_and_csv(tmp_path,reg,monkeypatch,phase,fault):
    comp,entries,output,work,runtime=controller_fixture(tmp_path,reg)
    calls=[]
    monkeypatch.setattr(h.subprocess,'Popen',lambda cmd,**kw:child_stub(cmd,calls,fault if phase=='native' else None))
    ids,raw,phases=h.run_phases(reg,entries,{},comp,work,selected=['R50','C4'])
    assert calls==['R50','C4']
    assert phases['C4']['exit_codes']==[0]
    assert np.isfinite(raw[entries[1]['id']]).all()
    if phase=='native':
        assert phases['R50']['exit_codes']!=[0]
        assert phases['R50']['failures'][0]['error']
        assert np.isnan(raw[entries[0]['id']][-1]).all()
        if fault!='launch': assert np.isfinite(raw[entries[0]['id']][0]).all()
    def s6_run(cmd,**kw):
        if phase=='native' or fault in ('launch','wait'): raise OSError('injected S6 launch failure')
        if fault=='timeout': raise subprocess.TimeoutExpired(cmd,kw['timeout'])
        return subprocess.CompletedProcess(cmd,9)
    monkeypatch.setattr(h.subprocess,'run',s6_run)
    raw['S6'],info=h.run_s6(dict(ids=ids,output=str(work/'public/p956-run/s6')),work,phases)
    assert phases['S6']['exit_codes']!=[0] and phases['S6']['failures'][0]['error']
    assert h.read(work/'phases.json')==phases
    rec=h.finish_run(runtime,output,reg,ids,raw,phases,info,0.,time.monotonic(),device='cpu')
    got_ids,values=h.c.v5q._read_csv(output/'submission.csv')
    assert got_ids==ids and np.isfinite(values).all()
    assert (output/'submission.csv').read_bytes()==h.c.p.csv_bytes(ids,reference(reg,raw,'A','R',True))
    assert rec['status']=='COMPLETE_HB116' and not rec['normal_visible_qualified']
    assert h.read(output/'hb116-receipt.json')['phases']==phases


@pytest.mark.parametrize('fault',['pin','journal','s6_identity'])
def test_operational_recovery_does_not_hide_identity_failure(tmp_path,reg,monkeypatch,fault):
    comp,entries,output,work,runtime=controller_fixture(tmp_path,reg)
    calls=[]
    monkeypatch.setattr(h.subprocess,'Popen',lambda cmd,**kw:child_stub(cmd,calls,corrupt=fault=='journal'))
    if fault=='pin': entries[0]['sha256']='0'*64
    if fault in ('pin','journal'):
        with pytest.raises(ValueError,match='identity'):
            h.run_phases(reg,entries,{},comp,work)
        if fault=='pin': assert calls==[]
    else:
        monkeypatch.setattr(h.subprocess,'run',lambda *a,**kw:(_ for _ in ()).throw(OSError('launch failed')))
        h.dump(work/'public/p956-run/s6/provenance.json',{'bundle_id':'wrong'})
        with pytest.raises(ValueError,match='S6 identity'):
            h.run_s6(dict(ids=['one'],output=str(work/'public/p956-run/s6')),work,{})
    assert not (output/'submission.csv').exists()


def test_runtime_recomputes_samples_and_rejects_review_reproduction(tmp_path,reg):
    v,rec,_=target_fixture(tmp_path,reg)
    for f in rec['timing']['families'].values(): f['study_seconds']=[100.]*3
    # Even a projection recached using the old flawed formula cannot qualify.
    rec['projection'].update(fixed_seconds=614.,marginal_seconds=3.,projected_seconds=4514.)
    h.dump(v/'hb116-receipt.json',rec)
    with pytest.raises(ValueError,match='inconsistent timing summary'): h.target_receipt(v,False,reg)
    for f in rec['timing']['families'].values(): f['upper95_study_seconds']=100.
    assert h.projection(rec)['projected_seconds']==130206.


@pytest.mark.parametrize('field',['sample','setup','fixed','upper','mean'])
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_projection_invalid_timing_fails_closed(tmp_path,reg,field,bad):
    _,rec,_=target_fixture(tmp_path,reg)
    family=rec['timing']['families']['NATIVE']
    if field=='sample': family['study_seconds'][0]=bad
    if field=='setup': family['setup_upper_seconds']=bad
    if field=='fixed': rec['fixed_seconds']=bad
    if field=='upper': family['upper95_study_seconds']=bad
    if field=='mean': family['service_seconds_per_study']=bad
    with pytest.raises(ValueError): h.projection(rec)


def test_projection_nonconstant_samples_independent_reference(tmp_path,reg):
    import math
    _,rec,_=target_fixture(tmp_path,reg)
    # Known one-sided t(.95, df=2) = 2.919985580355516, sample std = 1.
    upper=2.+2.919985580355516/math.sqrt(3)
    for f in rec['timing']['families'].values():
        f.update(study_seconds=[1.,2.,3.],upper95_study_seconds=upper,service_seconds_per_study=2.)
    result=h.projection(rec)
    assert result['projected_seconds']==pytest.approx(12.+1300*upper,abs=1e-8)


def test_s_to_t_is_explicitly_on_demand(monkeypatch):
    monkeypatch.setattr(h,'validate',lambda out:None)
    result=h.admit()
    assert result['s_to_t']['prebuilt'] is False
    assert result['s_to_t']['status']=='ON_DEMAND_ONLY'
    assert 'after cutting S43 and R2' in result['s_to_t']['trigger']


@pytest.mark.parametrize('missing',[False,True])
def test_shared_q_inputs_are_frozen_exact(missing):
    from tools.hb116_native_runtime import shared_q_inputs
    sys.path.insert(0,str(h.LEGACY))
    from raptor_common import CONTRACTS,select_centers
    spec=dict(CONTRACTS['raptor64_336_c140_s02'],depth=64)
    vol=np.random.default_rng(13).integers(0,255,(64,336,336),dtype=np.uint8)
    mask=np.ones(64,bool)
    if missing: mask[20:45]=False; vol[20:45]=0
    centers=select_centers(mask,64,36,spec,'within-slot-valid')
    actual=shared_q_inputs(vol,centers)
    for family in ('R50','C4','S'):
        np.testing.assert_array_equal(actual[family],h.q_inputs(vol,centers,family))
    assert actual['C4'] is actual['S']


@pytest.mark.parametrize('signed,mono,bits,scale',[(False,False,16,False),(True,False,12,True),(False,True,12,True)])
def test_shared_pixel_exact_decode_and_restoration(signed,mono,bits,scale):
    import pydicom
    from pydicom.dataset import FileDataset,FileMetaDataset
    from tools.hb116_native_runtime import SharedPixels
    from tools import exp169_release_v6 as v6
    sys.path.insert(0,str(h.LEGACY))
    import raptor_dicom
    meta=FileMetaDataset()
    meta.TransferSyntaxUID=pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID='1.2.3.4'
    ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.Rows=8; ds.Columns=9; ds.SamplesPerPixel=1
    ds.BitsAllocated=16; ds.BitsStored=bits; ds.HighBit=bits-1; ds.PixelRepresentation=int(signed)
    ds.PhotometricInterpretation='MONOCHROME1' if mono else 'MONOCHROME2'
    if scale: ds.RescaleSlope='1.2345'; ds.RescaleIntercept='-127.8'
    values=np.arange(72,dtype=np.int16).reshape(8,9)-30
    ds.PixelData=values.tobytes()
    buf=io.BytesIO(); ds.save_as(buf,enforce_file_format=True); raw=buf.getvalue()
    class Source:
        def read(self,member): return raw
    source=Source(); original=raptor_dicom._decode
    expected,digest=original(source,'x')
    ns=h.load_module('hb116_test_pixels',h.ROOT/'artifacts/releases/EXP155_S42v1_1_stage_v12/assets/public-serving.py').__dict__
    ns.update(FR_HDR=1,FR_PX=1)
    reader=v6.anchor_reader(ns)
    cache=SharedPixels(reader)
    with pytest.raises(RuntimeError,match='injected'):
        with cache:
            actual,got=raptor_dicom._decode(source,'x')
            np.testing.assert_array_equal(actual,expected); assert got==digest
            anchor,_=cache.ctx.px(raw,None)
            anchor=anchor.to_float() if hasattr(anchor,'to_float') else anchor
            np.testing.assert_array_equal(anchor,expected)
            assert cache.misses==1 and cache.hits==1
            assert cache.anchor_decode(b'broken',None)==(None,None)
            raise RuntimeError('injected')
    assert raptor_dicom._decode is original and cache.ctx.px is cache.decoder
    assert not cache.cache


def test_prefetch_is_bounded_and_ordered():
    from tools.hb116_native_runtime import prefetched
    from threading import Event
    second=Event(); release=Event(); calls=[]
    def prepare(uid):
        calls.append(uid)
        if uid==1: second.set(); assert release.wait(5)
        return uid*2
    stream=prefetched(range(4),prepare)
    assert next(stream)[:2]==(0,0)
    assert second.wait(5); assert calls==[0,1]
    release.set()
    assert [(uid,value) for uid,value,_ in stream]==[(1,2),(2,4),(3,6)]


@pytest.mark.parametrize('fault',['load','infer','q_prepare','anchor_prepare',
                                'cache_enter','cache_decode','r50_normalize','c4_normalize'])
def test_native_worker_isolates_member_and_input_failures(tmp_path,monkeypatch,fault,reg):
    from tools import hb116_native_runtime as native
    from tools import exp169_release_v6 as v6
    import torch
    torch.set_num_threads(1)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    sys.path.insert(0,str(h.LEGACY))
    import raptor_dicom,raptor_common
    # Operational worker path with tiny stand-ins; actual readers/models are
    # separately exercised by the visible-study native smoke.
    class Corpus:
        spec={}
        @classmethod
        def from_records(cls,*a): return cls()
        def volume(self,uid):
            if fault=='q_prepare' and uid=='one': raise ValueError('injected Q')
            return np.full((64,2,2),1 if uid=='one' else 2,np.uint8)
        def mask(self,uid): return np.ones(64,bool)
    class Cache:
        seconds=0.; misses=1; hits=1
        failed=False
        attempts=0; exits=0
        def __init__(self,reader): pass
        def __enter__(self):
            Cache.attempts+=1
            if Cache.attempts==1:
                if fault=='cache_enter': raise OSError('injected cache setup')
                if fault=='cache_decode': self.failed=True
            return self
        def __exit__(self,*a): Cache.exits+=1
    def reader(uid,*a):
        if fault=='anchor_prepare' and uid=='one': raise ValueError('injected anchor')
        return np.ones((96,2,2),np.uint8),np.ones(96,bool)
    def model(entry,device):
        if fault=='load' and entry['id']=='q0': raise ValueError('injected load')
        m=torch.nn.Linear(1,1); m.tag=entry['id']; return m
    anchor=model({'id':'anchor'},'cpu')
    ns={'gpu_windows':lambda *a,**k:torch.zeros(2,3,2,2)}
    monkeypatch.setattr(h,'configure',lambda:None)
    monkeypatch.setattr(h,'q_model',model)
    monkeypatch.setattr(h,'anchor_setup',lambda *a:(ns,{'anchor':anchor},reader))
    monkeypatch.setattr(v6,'anchor_reader',lambda ns:reader)
    monkeypatch.setattr(native,'SharedPixels',Cache)
    monkeypatch.setattr(raptor_dicom,'DicomCorpus',Corpus)
    monkeypatch.setattr(raptor_common,'select_centers',lambda *a:[0,1])
    monkeypatch.setattr(h.c.a172.base,'plane_flags',lambda *a:[False,True])
    if fault.endswith('normalize'):
        import raptor_radimagenet,raptor_runtime
        monkeypatch.setattr(raptor_radimagenet,'center3_windows_from',
            lambda vol,centers,size:np.full((len(centers),3,2,2),vol.flat[0] if vol.size else 1,np.float32))
        def normalize(value,kind):
            if value.flat[0]==1 and kind==('radimagenet' if fault=='r50_normalize' else 'imagenet'):
                raise ValueError('injected '+kind+' normalization')
            return value
        monkeypatch.setattr(raptor_radimagenet,'normalise',normalize)
        monkeypatch.setattr(raptor_runtime,'normalize_input',normalize)
    else:
        monkeypatch.setattr(native,'shared_q_inputs',lambda *a:{k:np.zeros((2,3,2,2),np.float32) for k in ('R50','C4','S')})
    def infer(m,*a,**kw):
        if fault=='infer' and m.tag=='q0': raise ValueError('injected inference')
        return np.full(12,.25,np.float32)
    monkeypatch.setattr(v6,'infer',infer)
    entries=[dict(id='q0',family='R50'),dict(id='q1',family='S'),
             dict(id='q2',family='C4'),dict(id='anchor')]
    identity={'version':h.VERSION,'members':{e['id']:'a'*64 for e in entries}}
    out=tmp_path/'worker'; job=tmp_path/'job.json'
    h.dump(job,dict(ids=['one','two'],rows=[],images=str(tmp_path),legacy=str(h.LEGACY),
                   output=str(out),entries=entries,identity=identity))
    try: native.worker(job,'cpu')
    finally: torch.set_num_threads(1)
    values,errors=h.c.read_journal(out,['one','two'],[e['id'] for e in entries],identity)
    if fault.startswith('cache'):
        assert all(np.isfinite(a).all() for a in values.values())
        assert all(not e for e in errors.values())
        assert Cache.attempts==Cache.exits==2
    else:
        bad='anchor' if fault=='anchor_prepare' else ('q2' if fault=='c4_normalize' else 'q0')
        assert errors[bad]
        healthy='anchor' if fault=='q_prepare' else ('q0' if fault=='c4_normalize' else 'q1')
        assert np.isfinite(values[healthy]).all()
        if fault.endswith(('prepare','normalize')): assert np.isfinite(values[bad][1]).all()
        if fault=='r50_normalize':
            assert np.isfinite(values['q2']).all() and not errors['q2']
            assert 'one' in errors['q0'] and 'two' not in errors['q0']
    timing=h.read(out/'timing.json')
    assert len(timing['details'])==2 and timing['prefetch_depth']==1
    if fault.startswith('cache'):
        assert timing['details']['one']['shared_cache_error']
        assert 'shared_cache_error' not in timing['details']['two']
    # Parent serving discards neutral journal placeholders for failed members.
    for member,failures in errors.items():
        for i,uid in enumerate(['one','two']):
            if uid in failures: values[member][i]=np.nan
    raw={'EXP143v2':values['anchor']}
    for key,family in [('q0','R50'),('q1','S'),('q2','C4')]:
        raw[next(e['id'] for e in reg['Q']['members'] if e['family']==family)]=values[key]
    runtime=tmp_path/'runtime'
    h.dump(runtime/'build-config.json',dict(version=h.VERSION,candidate='A',rx='R',s43=True,stress=False))
    output=tmp_path/'export'
    h.finish_run(runtime,output,reg,['one','two'],raw,{}, {},0.,time.monotonic(),device='cpu')
    ids,csv_values=h.c.v5q._read_csv(output/'submission.csv')
    assert ids==['one','two'] and np.isfinite(csv_values).all()
    assert (output/'submission.csv').read_bytes()==h.c.p.csv_bytes(ids,reference(reg,raw,'A','R',True))


def test_cache_retry_failure_stays_study_local(monkeypatch):
    from tools import hb116_native_runtime as native
    class Broken:
        def __init__(self,reader): raise OSError('cache constructor')
    monkeypatch.setattr(native,'SharedPixels',Broken)
    def prepare(uid,independent=False):
        assert independent
        if uid=='one': raise ValueError('independent decode also failed')
        return {'errors':{},'timing':{},'prediction':'healthy'}
    result=list(native.prefetched(['one','two'],lambda uid:native.prepare_study(uid,None,prepare)))
    assert set(result[0][1]['errors'])=={'Q','ANCHOR'}
    assert result[1][1]['prediction']=='healthy'


@pytest.mark.parametrize('fault', ['normal', 'reset_peak_memory_stats', 'synchronize',
    'get_device_name', 'max_memory_allocated', 'max_memory_reserved',
    'get_device_properties', 'monotonic', 'timing.json', 'all_telemetry',
    'print', 'resident_model_bytes'])
def test_kaggle_native_launch_telemetry_cannot_drop_predictions(tmp_path, reg, monkeypatch, fault):
    """Drive the REAL two-shard launcher, but forbid every hardware CUDA call.

    Physical mask 1 still has just logical device 0. The CUDA stand-ins reject
    invalid indices and reset-before-model-initialization, matching the crash.
    """
    import os
    from types import SimpleNamespace
    from tools import hb116_native_runtime as native
    from tools import exp169_release_v6 as v6
    import torch
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)
    sys.path.insert(0, str(h.LEGACY))
    import raptor_dicom
    comp, entries, output, work, runtime = controller_fixture(tmp_path, reg)
    (comp/'test_series').mkdir()
    for entry in (next(e for e in reg['Q']['members'] if e['family']=='S'), dict(id='EXP143v2')):
        path=tmp_path/(entry['id']+'.pt'); path.write_bytes(b'test-only-model')
        entries.append(dict(entry, resolved=str(path), bytes=path.stat().st_size, sha256=h.sha(path)))
    state={'initialized': False}; launches=[]; calls=[]
    def check_device(device):
        resolved=torch.device(device)
        if resolved.type!='cuda' or resolved.index!=0:
            raise RuntimeError('Invalid device argument')
        assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0','1')
    def model(entry, device):
        check_device(device)
        state['initialized']=True
        m=torch.nn.Linear(1,1)
        if fault=='resident_model_bytes':
            m.buffers=lambda: (_ for _ in ()).throw(RuntimeError('injected resident_model_bytes'))
        return m
    def cuda_call(name):
        def call(device):
            check_device(device)
            calls.append((os.environ['CUDA_VISIBLE_DEVICES'], name, device))
            if name=='reset_peak_memory_stats':
                assert state['initialized'], 'reset before model CUDA initialization'
            if fault in (name, 'all_telemetry'):
                raise RuntimeError('injected '+name)
            return {'get_device_name':'Tesla T4',
                    'get_device_properties':SimpleNamespace(total_memory=16*1024**3)}.get(name, 128)
        return call
    names=('reset_peak_memory_stats','synchronize','get_device_name',
           'max_memory_allocated','max_memory_reserved','get_device_properties')
    for name in names:
        monkeypatch.setattr(torch.cuda,name,cuda_call(name))
        with pytest.raises(RuntimeError, match='Invalid device argument'):
            getattr(torch.cuda,name)('cuda:1')
    def forbid_cuda(*args, **kwargs):
        raise AssertionError('Unexpected hardware CUDA path in CPU-only test')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbid_cuda)
    monkeypatch.setattr(torch.cuda,'set_device',forbid_cuda)
    monkeypatch.setattr(h,'configure',lambda:None)
    monkeypatch.setattr(h,'q_model',model)
    monkeypatch.setattr(h,'anchor_setup',lambda job,part,device,failed:
        ({'gpu_windows':lambda *a,**k:torch.full((2,3,2,2),.25)},
         {e['id']:model(e,device) for e in part},None))
    monkeypatch.setattr(v6,'anchor_reader',lambda ns:None)
    monkeypatch.setattr(raptor_dicom.DicomCorpus,'from_records',lambda *args:None)
    def prepare(uid,*args):
        return dict(errors={},timing={},q={k:np.full((2,3,2,2),.25,np.float32)
                    for k in ('R50','C4','S')}, flags=np.array([False,True]),
                    anchor=(np.ones((96,2,2),np.uint8),np.ones(96,bool)))
    monkeypatch.setattr(native,'prepare_study',prepare)
    def to_cpu(tensor,device,*args,**kwargs):
        check_device(device)
        return tensor
    monkeypatch.setattr(torch.Tensor,'to',to_cpu)
    def infer(model,tensor,device,**kwargs):
        check_device(device)
        return np.full(12,tensor.mean().item(),np.float32)
    monkeypatch.setattr(v6,'infer',infer)
    def raising(): raise RuntimeError('injected monotonic')
    if fault in ('monotonic','all_telemetry'):
        monkeypatch.setattr(native,'time',SimpleNamespace(monotonic=raising))
    if fault in ('print','all_telemetry'):
        monkeypatch.setattr(native,'print',lambda *a,**k:(_ for _ in ()).throw(OSError('injected print')),raising=False)
    original_dump=h.dump
    def dump(path,value):
        if (fault in ('timing.json','all_telemetry') and Path(path).name=='timing.json') or (
                fault=='all_telemetry' and Path(path).name=='telemetry-errors.json'):
            raise OSError('injected telemetry write')
        return original_dump(path,value)
    monkeypatch.setattr(h,'dump',dump)
    def launch(command,env,**kwargs):
        device=command[command.index('--device')+1]
        launches.append((env['CUDA_VISIBLE_DEVICES'],device))
        assert device=='cuda:0'
        state['initialized']=False
        with monkeypatch.context() as child:
            child.setenv('CUDA_VISIBLE_DEVICES',env['CUDA_VISIBLE_DEVICES'])
            native.worker(command[command.index('--job')+1],device)
        return SimpleNamespace(wait=lambda timeout:0)
    monkeypatch.setattr(h,'subprocess',SimpleNamespace(Popen=launch,
        SubprocessError=subprocess.SubprocessError, STDOUT=subprocess.STDOUT))
    ids,raw,phases=h.run_phases(reg,entries,{'legacy':str(h.LEGACY)},comp,work,device='cuda')
    assert launches==[('0','cuda:0'),('1','cuda:0')]
    assert phases['NATIVE']['exit_codes']==[0,0]
    assert {mask for mask,_,_ in calls}=={'0','1'}
    for values in raw.values(): np.testing.assert_array_equal(values,np.full((3,12),.25,np.float32))
    if fault!='all_telemetry':
        for shard in range(2):
            folder=work/'NATIVE'/f'worker{shard}'
            errors=h.read(folder/'telemetry-errors.json') if fault!='normal' else h.read(folder/'timing.json')['telemetry_errors']
            assert any(e['operation']==fault for e in errors) if fault!='normal' else not errors
    rec=h.finish_run(runtime,output,reg,ids,raw,phases,{},0.,time.monotonic(),device='cpu-mocked-cuda')
    assert (output/'submission.csv').read_bytes()==h.c.p.csv_bytes(ids,reference(reg,raw,'A','R',True))
    assert not any(rec['missing'].values())
    if fault not in ('normal','all_telemetry'):
        assert rec['timing']['errors'] and 'timing telemetry incomplete' in rec['timing_error']


def test_native_clock_failure_does_not_break_prefetch_or_decode(monkeypatch):
    from types import SimpleNamespace
    from tools import hb116_native_runtime as native
    def clock(): raise RuntimeError('broken clock')
    monkeypatch.setattr(native,'time',SimpleNamespace(monotonic=clock))
    telemetry=native.Telemetry()
    result=list(native.prefetched(['one','two'],lambda uid:uid,telemetry))
    assert result==[('one','one',0.),('two','two',0.)]
    assert telemetry.snapshot()[0]['count']==4


@pytest.mark.parametrize('device,expected',[('cpu','cpu'),('cuda','cuda:0'),('cuda:0','cuda:0')])
def test_worker_device_uses_process_local_index(monkeypatch,device,expected):
    from tools.hb116_native_runtime import worker_device
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    assert worker_device(device)==expected
