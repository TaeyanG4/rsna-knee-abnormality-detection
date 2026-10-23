"""HB127: independent math, process boundaries, recovery, package and closed admission."""
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
import pytest
from tools import hb127_vnextr7_release as m
from tools import hb127_vnextr7_verify as v
h=m.h


@pytest.fixture
def reg(): return h.read(h.REG/'consults/astra3-registration-draft.json')


def sample(reg,n=13):
    rng=np.random.default_rng(127)
    return {k:rng.integers(0,8,(n,12)).astype(np.float32)/8 for k in m.roster(reg)}


@pytest.mark.parametrize('candidate',['A','B'])
@pytest.mark.parametrize('failed',[False,True])
def test_reference_and_v112_standin(reg,candidate,failed):
    raw=sample(reg)
    if failed:
        for i,(k,a) in enumerate(raw.items()):
            a[i%5]=np.nan
            if k.startswith('S-'): a[5]=np.nan
            a[6 if k not in ('S_R7','R2-s43','R2-s44') else 7]=np.nan
            a[8]=np.nan
    actual,_=m.blend(reg,raw,candidate)
    expected,_=v.reference(reg,raw,candidate)
    old,_=h.blend(reg,{('S6' if k=='S_R7' else k):a for k,a in raw.items()},candidate,'R2',True)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual,old)
    if failed: np.testing.assert_array_equal(actual[8],np.full(12,.5,np.float32))
    with pytest.raises(ValueError,match='ambiguous'): m.blend(reg,dict(raw,S6=raw['S_R7']))


@pytest.mark.parametrize('candidate,stress',[('A',False),('B',False),('A',True)])
@pytest.mark.parametrize('fault',['none','timing','codec','p_csv'])
def test_export_precedes_telemetry_and_never_qualifies(tmp_path,reg,monkeypatch,candidate,stress,fault):
    raw=sample(reg,3)
    runtime=tmp_path/'runtime'
    h.dump(runtime/'build-config.json',m.config(candidate,stress))
    def fail(*a,**k): raise OSError('injected telemetry')
    monkeypatch.setattr(m,'timing',fail)
    if fault=='codec': monkeypatch.setattr(m.hook_integration,'codec_summary',fail)
    if fault=='p_csv':
        original=Path.write_bytes
        def write(path,data):
            if path.name=='p-branch-r7.csv': return fail()
            return original(path,data)
        monkeypatch.setattr(Path,'write_bytes',write)
    rec=m.finish(runtime,tmp_path/'out',reg,['a','b','c'],raw,{}, {},0.,time.monotonic(),candidate=candidate,stress=stress,device='cpu')
    assert not rec['qualified_for_submission']
    dest=tmp_path/'out'/('diagnostic-hb127.csv' if stress else 'submission.csv')
    assert dest.read_bytes()==h.c.p.csv_bytes(['a','b','c'],v.reference(reg,raw,candidate)[0])
    if stress: assert not (tmp_path/'out/submission.csv').exists()
    if fault=='codec': assert rec['codecs']['status']=='TELEMETRY_ERROR'
    if fault=='p_csv': assert rec['p_branch_write']['status']=='TELEMETRY_ERROR'


def recovery_fixture(work):
    identity=dict(manifest_sha256=m.MANIFEST_SHA,manifest_pinned=True,bundle_id=m.r7.BUNDLE_ID,
        dataset_slug=m.r7.DATASET_SLUG,file_sha256={'model':'a'*64})
    h.dump(work/'public/pr7-run/r7-identity.json',identity)
    return identity


@pytest.mark.parametrize('fault',['exit','oserror','timeout','partial','identity'])
def test_r7_operational_failures_recover_but_identity_is_hard(tmp_path,monkeypatch,fault):
    identity=recovery_fixture(tmp_path)
    job=dict(ids=['a','b'],codec={'status':'INACTIVE'})
    def run(*a,**k):
        if fault=='oserror': raise OSError('spawn failed')
        if fault=='timeout': raise subprocess.TimeoutExpired('child',1)
        if fault in ('partial','identity'):
            root=tmp_path/'public/pr7-run/s_r7'
            prov=dict(identity,source_sha256=m.r7.SOURCE_HASHES,anchor_weights_sha256=h.c.s6.ANATOMY_SHA256)
            if fault=='identity': prov['manifest_sha256']='0'*64
            h.dump(root/'provenance.json',prov)
            h.dump(root/'controller/worker0/ids.json',['a','b'])
            (root/'controller/worker0/predictions.csv').write_bytes(h.c.p.csv_bytes(['a'],np.full((1,12),.2,np.float32)))
        return type('Exit',(),{'returncode':7})()
    monkeypatch.setattr(m.subprocess,'run',run)
    phases={}
    if fault=='identity':
        with pytest.raises(ValueError,match='identity'): m.run_r7(job,tmp_path,phases)
    else:
        values,info=m.run_r7(job,tmp_path,phases)
        assert np.isnan(values[1]).all()
        assert np.isfinite(values[0]).all() if fault=='partial' else np.isnan(values[0]).all()
        assert phases['S_R7']['exit_codes'][0]!=0


def test_bad_expected_pin_fatal_without_predictions(tmp_path,monkeypatch):
    identity=recovery_fixture(tmp_path)
    identity['manifest_sha256']='0'*64
    h.dump(tmp_path/'public/pr7-run/r7-identity.json',identity)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:type('Exit',(),{'returncode':1})())
    with pytest.raises(ValueError,match='pinned preflight'):
        m.run_r7(dict(ids=['a'],codec={}),tmp_path,{})


@pytest.mark.parametrize('active',[False,True])
def test_nested_children_inherit_hook_and_restore(tmp_path,active):
    before=dict(os.environ)
    codec={'status':'INACTIVE'}
    if active: codec=v.stub_codec(tmp_path,tmp_path/'packages')
    out=tmp_path/'child.json'
    child="import json,os,subprocess,sys; c='import json,os;print(json.dumps(dict(os.environ)))'; r=subprocess.check_output([sys.executable,'-c',c],text=True);open(sys.argv[1],'w').write(r)"
    with m.child_environment(codec,'NATIVE_O_R2',tmp_path):
        subprocess.run([sys.executable,'-B','-c',child,str(out)],check=True)
    env=h.read(out)
    assert env.get('CUDA_VISIBLE_DEVICES','')==''
    if active:
        assert codec['hook_dir'] in env['PYTHONPATH']
        assert env[m.codecs.BRANCH_ENV]=='NATIVE_O_R2'
        assert len(m.codecs.read_telemetry(tmp_path/'c85r7-run/codecs/telemetry'))>0
    else: assert env.get('PYTHONPATH')==before.get('PYTHONPATH')
    assert dict(os.environ)==before


@pytest.mark.parametrize('fault',['install','gate','strip','none'])
def test_shared_hook_errors_do_not_veto_r7(tmp_path,monkeypatch,fault):
    def fail(*a,**k): raise RuntimeError('injected diagnostic')
    if fault=='install':
        monkeypatch.setattr(m.hook_integration,'install_codecs',fail)
        assert m.install_codecs(tmp_path,tmp_path)['status']=='TELEMETRY_ERROR'
    if fault in ('gate','strip'): monkeypatch.setattr(m.hook_integration,'gate_s_r7',fail)
    if fault=='strip': monkeypatch.setattr(m.codecs,'strip_env',fail)
    called=[]
    monkeypatch.setattr(m.r7,'infer_studies',lambda *a,**k:called.append(k))
    job=dict(device='cpu',work=str(tmp_path),codec={},code=str(tmp_path),ids=['a'],competition=str(tmp_path),
        bundle=str(tmp_path),anatomy=str(tmp_path/'anatomy'),output=str(tmp_path/'out'),source=str(m.r7.SOURCE))
    h.dump(tmp_path/'job.json',job)
    m.r7_worker(tmp_path/'job.json')
    assert called[0]['expected_manifest_sha256']==m.MANIFEST_SHA and called[0]['device']=='cpu'


def test_bundle_sources_and_isolated_package(tmp_path):
    files=m.source_payloads()
    assert files['tools/hb126_codecs.py']==Path(m.codecs.__file__).read_bytes()
    assert files['tools/exp169_release_c85r7codec.py']==Path(m.hook_integration.__file__).read_bytes()
    for name in ('hb116_vnext_release.py','hb116_native_runtime.py'):
        assert h.sha(h.ROOT/'tools'/name)==h.read(m.BASE/'kernels/r2/a/source-pins.json')['tools/'+name]
    blob,_=h.package(files)
    with zipfile.ZipFile(io.BytesIO(blob)) as z: z.extractall(tmp_path)
    env=dict(os.environ,PYTHONPATH=str(tmp_path),CUDA_VISIBLE_DEVICES='')
    proc=subprocess.run([sys.executable,'-B','-c','from tools import hb127_vnextr7_release as m; assert m.VERSION=="1.2.0"'],
        cwd=tmp_path,env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr


def test_prepare_validate_and_closed_admit(tmp_path,monkeypatch):
    h.dump(tmp_path/'registration.json',{'r7':{'manifest_sha256':m.MANIFEST_SHA},
        'source_pins':{k:hashlib.sha256(v).hexdigest() for k,v in m.source_payloads().items()}})
    assert m.prepare(tmp_path)['kernels']==3
    assert m.validate(tmp_path)['kernels']==3
    for key in ('a','b','stress-n32'):
        meta=h.read(tmp_path/'kernels'/key/'kernel-metadata.json')
        assert meta['id']=='taeyangg4/knee-hb127-vnextr7-'+key
        assert meta['enable_internet'] is False and meta['is_private'] is True
        assert m.codecs.DATASET_SLUG in meta['dataset_sources']
    h.dump(tmp_path/'validation/replay.json',dict(status='PASS',standin_exact=True,independent_exact=True))
    assert m.admit(tmp_path)['status']=='HOLD_NOT_QUALIFIED'
    path=tmp_path/'kernels/b/build-config.json'
    cfg=h.read(path);cfg['rx']='R';h.dump(path,cfg)
    with pytest.raises(ValueError,match='config'): m.validate(tmp_path)


@pytest.mark.parametrize('candidate,stress',[('A',False),('B',False),('A',True)])
def test_serve_dispatch_preserves_native_and_r2(tmp_path,monkeypatch,reg,candidate,stress):
    import torch
    from tools import exp169_c85_stress
    runtime=tmp_path/'runtime'; out=tmp_path/'out'
    h.dump(runtime/'build-config.json',m.config(candidate,stress))
    (runtime/'sealed-registration.json').write_bytes((h.REG/'consults/astra3-registration-draft.json').read_bytes())
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'MOCK T4 CPU')
    monkeypatch.setattr(h.c.v5q,'_mount',lambda *a:tmp_path/'comp')
    monkeypatch.setattr(exp169_c85_stress,'prepare',lambda *a,**k:None)
    monkeypatch.setattr(h,'remote_assets',lambda r,rt,w,rx,s43:([],{}) if rx=='R2' and s43 else (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(m.r7,'verify_bundle',lambda *a:({},{}))
    monkeypatch.setattr(m.r7,'code_record',lambda *a:None)
    monkeypatch.setattr(h.c.s6,'prepare_code',lambda *a,**k:(tmp_path/'code',tmp_path/'anatomy'))
    monkeypatch.setattr(m,'install_codecs',lambda *a:{'status':'INACTIVE'})
    raw=sample(reg,3)
    def native(*a,**k): return ['a','b','c'],{key:val for key,val in raw.items() if key!='S_R7'},{}
    monkeypatch.setattr(h,'run_phases',native)
    monkeypatch.setattr(m,'run_r7',lambda *a,**k:(raw['S_R7'],{}))
    result=m.serve(runtime,out,candidate,stress)
    assert result['rx']=='R2' and result['diagnostic']==stress and not result['qualified_for_submission']


def test_real_recorded_replay(tmp_path):
    assert v.replay(tmp_path)['standin_exact']


def test_r7_timing_keeps_p_branch_and_frozen_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'timings', lambda *a: dict(members={}, families={}, branches={}, units='seconds/study', errors=[]))
    monkeypatch.setattr(h.c.s6, '_worker_receipts', lambda rec: [dict(per_study=[
        ['a', 'ok', 1., 2., 3., 4.], ['b', 'ok', 1., 2., 3., 4.], ['c', 'ok', 1., 2., 3., 4.]])])
    h.dump(tmp_path/'public/pr7-run/s_r7/controller/submission_receipt.json', {})
    result = m.timing(tmp_path, {'S_R7': {'seconds': 35.}}, 3)
    assert result['branches']['P'] == 35./3
    assert result['members']['S_R7'] == dict(seconds=30., seconds_per_study=10., count=3)
    assert result['families']['S6']['setup_upper_seconds'] == 5.
    result['families']['NATIVE'] = dict(study_seconds=[8., 8., 8.], setup_upper_seconds=2.,
        service_seconds_per_study=8., upper95_study_seconds=8.)
    projection = h.projection(dict(timing=result, studies=3, fixed_seconds=1.))
    assert projection['limit_seconds'] == 28800
    assert projection['projected_seconds'] == 26.+1300*9.
