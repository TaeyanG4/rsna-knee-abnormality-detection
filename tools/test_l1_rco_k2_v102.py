"""K2 focused CPU tests. Every forged object is confined to pytest tmp_path.

No fixture emits a gate beneath the real project, invokes a CLI endpoint,
uses production ledgers, or claims qualification of real GPU paths.
"""
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types

os.environ['CUDA_VISIBLE_DEVICES']='-1'

import numpy as np
import pytest

from tools import l1_rco_k2_worker_v101 as w
from tools import l1_rco_k2_v102 as b
from tools import l1_rco_k2_admission_v102 as a
from tools import strict_variant_gate as base
from tools import strict_variant_gate_k2_v102 as g
from tools import kaggle_submit_once_k2_v102 as ks


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args,**kwargs):raise AssertionError('test forbids real subprocess/network')
    monkeypatch.setattr(subprocess,'run',denied)
    monkeypatch.setattr(socket.socket,'connect',denied)


def test_fixed_contract_and_ast():
    assert w.CONTRACT['models']==1 and w.CONTRACT['workers']==0
    assert w.CONTRACT['precision']=='float32' and w.CONTRACT['crop']==[16,368]
    assert w.CONTRACT['eval_stride']==3 and w.CONTRACT['phys_mm']==3.5
    assert set(b.ast_identity())=={'Net','Studies','collate','window_rows','covering_centers','phys_step'}


def test_offline_denies_connection():
    with w.offline(),pytest.raises(RuntimeError,match='network denied'):
        socket.create_connection(('example.com',443))


def test_raw_source_never_reads_train_labels(monkeypatch):
    calls=[]
    class Original:
        current='private-fixture-uid'
        def __init__(self):self.split='train'
        def read(self,name):calls.append(name);return b'metadata'
    fake=types.SimpleNamespace(OneStudySource=Original)
    monkeypatch.setitem(sys.modules,'probe_infer',fake)
    w.install_label_free_source();reader=fake.OneStudySource()
    text=reader.read('train.csv').decode()
    assert text.splitlines()==[w.UID,'private-fixture-uid'] and not calls
    assert reader.read('train_series.csv')==b'metadata'
    with pytest.raises(ValueError):reader.read('reports.txt')


@pytest.mark.parametrize('change',['contract','extra-file','hash','missing-file'])
def test_asset_inventory_failures(tmp_path,change):
    (tmp_path/'source').mkdir();p=tmp_path/'source/x.py';p.write_text('x=1')
    manifest=dict(contract=w.CONTRACT,version=w.VERSION,files={'source/x.py':w.sha(p)})
    if change=='contract':manifest['contract']=dict(w.CONTRACT,eval_stride=1)
    if change=='hash':manifest['files']['source/x.py']='0'*64
    if change=='missing-file':manifest['files']['source/absent.py']='0'*64
    if change=='extra-file':(tmp_path/'extra.pt').write_bytes(b'extra model')
    w.write(tmp_path/'asset-manifest.json',manifest)
    with pytest.raises(ValueError):w.assets_check(tmp_path,w.sha(tmp_path/'asset-manifest.json'))


@pytest.mark.parametrize('kind,delta,expected',[
    ('same_runtime',1e-7,'PASS'),('same_runtime',1e-3,'FAIL'),
    ('cross_runtime',1e-5,'PASS'),('cross_runtime',1e-2,'FAIL'),
    ('k1_bf16',.0005,'PASS'),('k1_bf16',.0011,'FAIL')])
def test_preregistered_parity(kind,delta,expected):
    p=np.full((32,12),.5);q=p.copy()+delta
    assert b.parity(p,q,kind)['status']==expected


def test_bf16_target_and_max_floors():
    p=np.zeros((32,12));q=p.copy();q[:,1]=.0021
    assert b.parity(p,q,'k1_bf16')['status']=='FAIL'
    q=p.copy();q[0,0]=.010001
    assert b.parity(p,q,'k1_bf16')['status']=='FAIL'


@pytest.mark.parametrize('bad',[np.zeros((0,12)),np.zeros((32,11)),np.full((32,12),np.nan)])
def test_parity_rejects_bad_shape_nan(bad):
    with pytest.raises(ValueError):b.parity(bad,bad,'same_runtime')


def test_pair_rank_disagreement():
    p=np.zeros((2,12));p[1]=1;q=p[::-1]
    assert b.parity(p,q,'same_runtime')['rank_disagreements']==[1]*12


@pytest.mark.parametrize('per,expected',[(20,'PASS'),(25,'FAIL')])
def test_runtime_full_study_projection(per,expected):
    def timing(n,fixed):return dict(timing_schema=b.TIMING_POLICY['schema'],studies=n,fixed_seconds=fixed,
        linear_seconds=per*n,journal_seconds=0,total_seconds=fixed+per*n)
    r=a.runtime_projection(timing(3,20),timing(32,30))
    assert r['projected_seconds']==30+1300*per and r['status']==expected


def test_d1_preprocessing_scales_and_legacy_schema_refused():
    def t(n):return dict(timing_schema=b.TIMING_POLICY['schema'],studies=n,fixed_seconds=0,
        linear_seconds=28*n,journal_seconds=0,total_seconds=28*n)
    assert a.runtime_projection(t(3),t(32))['projected_seconds']==36400
    assert a.runtime_projection(t(3),t(32))['status']=='FAIL'
    with pytest.raises((ValueError,KeyError)):
        a.runtime_projection(dict(setup_seconds=12,mean_e2e=24),dict(setup_seconds=128,mean_e2e=24))


def test_cumulative_journal_scales_by_total_prefix_rows():
    def t(n):return dict(timing_schema=b.TIMING_POLICY['schema'],studies=n,fixed_seconds=1,
        linear_seconds=n,journal_seconds=n*(n+1)*.001,total_seconds=1+n+n*(n+1)*.001)
    r=a.runtime_projection(t(3),t(32))
    assert r['projected_journal_seconds']==pytest.approx(1300*1301*.001)


def test_actual_phase_reconciliation_and_d1_counterexample():
    def measured(n):
        worker=dict(study_count=n,studies=[dict(end_to_end_seconds=24.) for _ in range(n)],total_seconds=28.*n,
            timing=dict(schema='k2-timing/v1.0.1',studies=n,journal_rows=n*(n+1)//2,fixed_seconds=0.,
                codec_header_seconds=4.*n,cohort_setup_seconds=0.,study_body_seconds=24.*n,
                cumulative_journal_seconds=0.,export_seconds=0.,residual_seconds=0.))
        boot=dict(schema='k2-bootstrap/v1',timing_policy=b.TIMING_POLICY['schema'],fixed_startup_seconds=0.,total_seconds=28.*n)
        actual=a.timing_components(worker,boot,28.*n)
        worker['timing']['codec_header_seconds']-=1
        with pytest.raises(ValueError,match='reconciliation'):a.timing_components(worker,boot,28.*n)
        return actual
    r=a.runtime_projection(measured(3),measured(32))
    assert r['projected_seconds']==36400 and r['status']=='FAIL'


@pytest.mark.parametrize('mode',['zero-progress','successful-write','persistent-write-failure','throughput-collapse'])
def test_d2_watchdog_unconditional_exit(monkeypatch,tmp_path,mode):
    d=w.Watchdog(tmp_path,tmp_path/'unused.json',False)
    class Stop:
        def wait(self,_):return False
    d.stop=Stop();d.last=0;now=901.;writes=[];exits=[]
    if mode=='throughput-collapse':
        now=1801.;d.last=now;d.healthy=10.;d.samples=[(0.,0),(1800.,1800)]
    def write(*args):
        writes.append(args)
        if mode=='persistent-write-failure':raise PermissionError('persistent simulated sharing violation')
    monkeypatch.setattr(w.time,'monotonic',lambda:now)
    monkeypatch.setattr(w,'write',write);monkeypatch.setattr(w.os,'_exit',lambda code:exits.append(code))
    d.watch()
    assert exits==[70] and len(writes)==1
    assert ('throughput' if mode=='throughput-collapse' else 'no progress') in writes[0][1]['reason']


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),True])
def test_runtime_invalid_measurements(value):
    with pytest.raises(ValueError):a.positive(value)


@pytest.mark.parametrize('mutation',['missing','duplicate','order','blank'])
def test_template_fail_closed(tmp_path,mutation):
    headers=[w.UID]+w.LABELS;ids=['a','b']
    if mutation=='missing':headers=headers[:-1]
    if mutation=='order':headers[1],headers[2]=headers[2],headers[1]
    if mutation=='duplicate':ids=['a','a']
    if mutation=='blank':ids=['a','']
    p=tmp_path/'template.csv'
    with p.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(headers)
        for uid in ids:writer.writerow([uid]+['']*12)
    with pytest.raises(ValueError):w.read_template(p)


def item():
    import torch
    return dict(uid='a',error=None,images=torch.zeros((1,3,384,384),dtype=torch.uint8),
        meta=torch.zeros((1,6),dtype=torch.float32),anchor=dict(method='slab_prior'),side=None)


@pytest.mark.parametrize('change',['empty','uid','error','metadata','nonfinite','dtype','side','anchor'])
def test_renderer_failure_never_becomes_prediction(change):
    import torch
    v=item()
    if change=='empty':v['images']=v['images'][:0]
    if change=='uid':v['uid']='other'
    if change=='error':v['error']='corrupt series'
    if change=='metadata':v['meta']=torch.zeros((1,7))
    if change=='nonfinite':v['meta'][0,0]=float('nan')
    if change=='dtype':v['images']=v['images'].float()
    if change=='side':v['side']={}
    if change=='anchor':v['anchor']=None
    with pytest.raises(ValueError):w.validate_item(v,'a')


def test_anchor_fallback_is_accepted_without_prediction_fallback():
    x,m=w.validate_item(item(),'a')
    assert len(x)==1 and m.shape==(1,6)


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_logits(bad):
    import torch
    class Model:
        def __call__(self,*args,**kwargs):return torch.full((1,12),bad)
    x,m=w.validate_item(item(),'a')
    with pytest.raises(ValueError,match='logits'):w.predict(Model(),x,m,'cpu')


def test_predict_exact_crop_mask_chunk():
    import torch
    class Model:
        def __call__(self,x,m,mask,chunk):
            assert x.shape==(1,1,3,352,352) and m.shape==(1,1,6)
            assert mask.dtype==torch.bool and mask.all() and chunk==32
            assert not torch.is_autocast_enabled()
            return torch.zeros((1,12),dtype=torch.float32)
    x,m=w.validate_item(item(),'a')
    assert np.array_equal(w.predict(Model(),x,m,'cpu'),np.full(12,.5))


def test_checkpoint_hash_checked_before_load(tmp_path,monkeypatch):
    import torch
    p=tmp_path/'wrong.pt';p.write_bytes(b'wrong')
    monkeypatch.setattr(torch,'load',lambda *x,**y:pytest.fail('must reject before unpickle'))
    with pytest.raises(ValueError,match='SHA256'):w.load_model(None,p,'cpu')


def test_windows_replace_bounded_retry(tmp_path,monkeypatch):
    import os
    p=tmp_path/'source';q=tmp_path/'dest';p.write_text('a');real=os.replace;calls=[]
    def busy(src,dst):
        calls.append(1)
        if len(calls)<3:raise PermissionError('sharing violation')
        real(src,dst)
    monkeypatch.setattr(w.os,'replace',busy);monkeypatch.setattr(w.time,'sleep',lambda _:None)
    w.replace(p,q);assert len(calls)==3 and q.read_text()=='a'


@pytest.mark.parametrize('phase',['BUILD_READY',None,'READY','FINAL_READY'])
def test_final_gate_phase_and_qualified(tmp_path,phase):
    path=tmp_path/'gate.json'
    w.write(path,dict(schema='k2-gate/v1.0.2',provenance_contract=b.PROVENANCE_CONTRACT,status='PASS',errors=[],gate_version=g.VERSION,
        phase=phase,kernel_id=w.SLUG,qualified_for_submission=False,fixture_only=False))
    with pytest.raises(ValueError,match='phase'):g.verify_receipt(path,base.Inputs(tmp_path),'FINAL_READY')


def archive(root,lane,agent,report,content):
    from tools import native_codex_dispatch_record_v1 as native
    p=root/report;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
    prompt=Path(str(p)+'.prompt');prompt.write_text('ISOLATED TEST FIXTURE ONLY; no real completion')
    completion=Path(str(p)+'.completion.json')
    w.write(completion,dict(schema=native.SCHEMA,host=native.HOST,model=native.MODEL,agent_id=agent,lane=lane,
        prompt=str(prompt),report=str(p),terminal_status='completed',observed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        prompt_sha256=w.sha(prompt),report_sha256=w.sha(p),completion_message='ISOLATED SYNTHETIC TEST ONLY'))
    native.record(completion,'ISOLATED TEST ONLY, not B production evidence',root)
    return p


@pytest.fixture
def production_schema(tmp_path,monkeypatch):
    realroot=b.ROOT;root=tmp_path.resolve()
    stage=root/'artifacts/experiments/L1_RCO_K2/v1.0.2-rcof0';work=root/'.local/scratch/slot-b/k2-release/reseal-v102'
    oldstage=root/'artifacts/experiments/L1_RCO_K2/v1.0.1-rcof0';assetwork=root/'.local/scratch/slot-b/k2-release/fix1'
    assets=assetwork/'assets-v1';assets.mkdir(parents=True)
    for name,value in [('ROOT',root),('STAGE',stage),('WORK',work),('ASSET_WORK',assetwork),('ASSETS',assets),('OLD_STAGE',oldstage)]:
        monkeypatch.setattr(b,name,value)
    def put(path,value):
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(value if isinstance(value,bytes) else w.encoded(value));return p
    def ref(p):return dict(path=p.relative_to(root).as_posix(),sha256=w.sha(p))
    def table(path,ids,k1=False):
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',newline='') as stream:
            cw=csv.writer(stream);cw.writerow([w.UID]+w.LABELS+(['status','model_fold'] if k1 else []))
            cw.writerows([[uid]+[.5]*12+(['ok','0'] if k1 else []) for uid in ids])
        return p
    oldsource={}
    for name in set(b.NEW+b.FROZEN):
        put(name,(realroot/name).read_bytes())
        old=name.replace('_v102.py','_v101.py').replace('-v102.py','-v101.py')
        p=put(old,(realroot/old).read_bytes());oldsource[old]=w.sha(p)
    canonical=put('tools/kaggle_submit_once.py',(realroot/'tools/kaggle_submit_once.py').read_bytes());oldsource['tools/kaggle_submit_once.py']=w.sha(canonical)
    model=put('fixture/model.bin',b'NONMODEL TEST FIXTURE');anatomy=put('fixture/anatomy.bin',b'NONANATOMY TEST FIXTURE')
    monkeypatch.setattr(w,'MODEL_SHA',w.sha(model));monkeypatch.setattr(w,'ANATOMY_SHA',w.sha(anatomy))
    monkeypatch.setattr(w,'CONTRACT',dict(w.CONTRACT,model_sha256=w.MODEL_SHA))
    ids=['fixture'+str(i) for i in range(32)];visible=['visible'+str(i) for i in range(3)]
    saved=table('.local/scratch/l1-rco/k1v2/fit/holdout.csv',ids,True)
    monkeypatch.setattr(b,'PINS',{saved.relative_to(root).as_posix():w.sha(saved)})
    oldsource.update(b.PINS);origins={};asset_files={}
    for original,name in [(model,'weights/E_prior_s0_f0.pt'),(anatomy,'weights/anatomy.pt'),(root/'tools/l1_rco_k2_worker_v101.py','source/l1_rco_k2_worker_v101.py')]:
        p=put((assets/name).relative_to(root),original.read_bytes());asset_files[name]=w.sha(p)
        origins[name]=dict(source=original.relative_to(root).as_posix(),sha256=w.sha(p),relation='byte-identical')
    origin=put((assetwork/'asset-origins.json').relative_to(root),origins);oldsource[origin.relative_to(root).as_posix()]=w.sha(origin)
    am=put((assets/'asset-manifest.json').relative_to(root),dict(contract=w.CONTRACT,version=w.VERSION,
        dataset=b.DATASET,private=True,wheels=b.s6.WHEEL_HASHES,deployment_versions=b.s6.WHEEL_VERSIONS,files=asset_files))
    monkeypatch.setattr(b,'ASSET_SHA',w.sha(am))
    put((assets/'dataset-metadata.json').relative_to(root),dict(id=b.DATASET,title=b.DATASET.split('/')[1]))
    parent=dict(schema='k2-registration/v1',candidate_id=b.CANDIDATE,kernel_id=w.SLUG,version=w.VERSION,contract=w.CONTRACT,
        qualified_for_submission=False,assets=assets.relative_to(root).as_posix(),asset_manifest_sha256=w.sha(am),
        visible_ids=visible,n32_ids=ids,smoke_holdout_ids=ids[:2],n32_roster_sha256=w.digest(ids),cache='fixture-cache',
        parity={},ast_identity={},runtime=dict(b.TIMING_POLICY,population_source='.local/scratch/slot-b/k2-release/POPULATION-SOURCE.json',
        population_affirmation_required=True,workload_rule='complete studies and cumulative journals; all residual scaled'))
    parentpath=put((oldstage/'registration.json').relative_to(root),parent)
    sm=put((oldstage/'source-manifest.json').relative_to(root),oldsource)
    oldnb=put((oldstage/'kernel/release.ipynb').relative_to(root),b.notebook(w.sha(am)))
    oldmeta=put((oldstage/'kernel/kernel-metadata.json').relative_to(root),dict(id=w.SLUG,title=w.SLUG.split('/')[1],code_file='release.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[b.DATASET,b.CODEC_DATASET],competition_sources=['rsna-knee-abnormality-detection'],kernel_sources=[]))
    for name,p in [('OLD_REG_SHA',parentpath),('OLD_MANIFEST_SHA',sm),('NOTEBOOK_SHA',oldnb),('METADATA_SHA',oldmeta)]:monkeypatch.setattr(b,name,w.sha(p))
    b.register();regpath=stage/'registration.json';reg=w.read(regpath);nb=stage/'kernel/release.ipynb'
    material=g.snapshot(stage,base.Inputs(root));sm=stage/'source-manifest.json'
    implementation_builder=archive(root,'native-astra-b1','/root/fixture_builder','implementation-build.md',
        'Kernel-SHA256: '+material['kernel_sha256']+'\nK2-Implementation-Sources-SHA256: '+w.sha(sm)+'\nTEST ONLY\n')
    implementation_review=archive(root,'native-astra-r1','/root/fixture_reviewer','implementation-review.md',
        'Reviewer-Lane: native-astra-r1\nK2-Implementation-Sources-SHA256: '+w.sha(sm)+'\nCandidate-ID: '+b.CANDIDATE+
        '\nReview-Phase: IMPLEMENTATION\nKernel-ID: '+w.SLUG+'\nVERDICT: PASS\n')
    prepared=root/'cpu-prepare.json';g.prepare(str(implementation_builder)+'.receipt.json',implementation_review,prepared,root)
    cpu_admission=root/'cpu-admission.json';g.admit(prepared,cpu_admission,root=root)
    env=dict(device='cpu',cuda_initialized=False,tf32=False,autocast=False);tensors=[dict(images_sha256='a'*64,metadata_sha256='b'*64,windows=2)]*2
    parts={};partpaths={}
    for part in b.CPU_PARTS:
        directory=work/'cpu-v1'/part;directory.mkdir(parents=True)
        now=dt.datetime.now(dt.timezone.utc)
        r=dict(schema='k2-cpu-smoke/v1.0.2',part=part,status='PASS',environment=env,cuda_visible_devices='-1',
            registration_sha256=w.sha(regpath),manifest_sha256=w.sha(am),builder_source_sha256=w.sha(root/'tools/l1_rco_k2_v102.py'),
            worker_sha256=b.WORKER_SHA,runtime_version=w.VERSION,provenance_contract=b.PROVENANCE_CONTRACT,qualified_for_submission=False,
            driver_sha256=b.CPU_DLL_SHA,driver_path=b.CPU_DLL_PATH,
            run_id=hashlib.md5(part.encode()).hexdigest(),pid=123,started_at=now.isoformat(),completed_at=(now+dt.timedelta(microseconds=1)).isoformat(),
            output_directory=directory.relative_to(root).as_posix(),cpu_admission=ref(cpu_admission))
        if part=='model':r['initialization_plus_deploy_exact']=True
        if part=='spawn':r.update(workers=[0,3],studies=2,exact=True)
        if part.startswith('raw-'):
            cohort=visible if part=='raw-visible' else ids[:2];p=table((directory/'submission.csv').relative_to(root),cohort)
            studies=[dict(index=i,uid_sha256=hashlib.sha256(uid.encode()).hexdigest(),windows=2,error=None,prediction_fallback=0,
                images_sha256='a'*64,metadata_sha256='b'*64) for i,uid in enumerate(cohort)]
            wr=put((directory/'worker-receipt.json').relative_to(root),dict(schema='k2-worker/v1',status='COMPLETE',version=w.VERSION,
                kernel_id=w.SLUG,contract=w.CONTRACT,manifest_sha256=w.sha(am),roster_sha256=w.digest(cohort),study_count=len(cohort),
                prediction_count=len(cohort),output_sha256=w.sha(p),fallbacks=dict(prediction=0),fast_path=False,placeholder=False,environment=env,studies=studies))
            r.update(worker_receipt_sha256=w.sha(wr),prediction_sha256=w.sha(p))
        if part=='cached-reference':
            p=table((directory/'reference.csv').relative_to(root),ids[:2]);r.update(reference_sha256=w.sha(p),
                k1_bf16=b.parity(np.full((2,12),.5,dtype=np.float32),np.full((2,12),.5,dtype=np.float32),'k1_bf16'),studies=2,tensors=tensors)
        result=put((directory/'result.json').relative_to(root),r);parts[part]=ref(result);partpaths[part]=result
        put((directory/'execution-attestation.json').relative_to(root),dict(schema='k2-cpu-execution/v1.0.2',recorded_by='B',fixture_only=False,
            part=part,run_id=r['run_id'],pid=r['pid'],exit_code=0,result=ref(result),admission=ref(cpu_admission),observed_terminal_status='completed',
            observed_at=dt.datetime.now(dt.timezone.utc).isoformat()))
    cpu=work/'CPU-COLLECTION.json';b.collect_cpu([root/p['path'] for p in parts.values()],cpu)
    builder=archive(root,'native-astra-b1','/root/fixture_builder','build.md',
        'Kernel-SHA256: '+material['kernel_sha256']+'\nK2-Registration-SHA256: '+w.sha(regpath)+'\nTEST ONLY\n')
    bp=root/'build-pin.json';g.pin(stage,cpu,str(builder)+'.receipt.json',bp,root)
    review=archive(root,'native-astra-r1','/root/fixture_reviewer','build-review.md',
        f'Kernel-ID: {w.SLUG}\nReviewer-Lane: native-astra-r1\nK2-Pin-SHA256: {w.sha(bp)}\nCandidate-ID: {b.CANDIDATE}\nReview-Phase: BUILD_READY\nVERDICT: PASS\n')
    bg=root/'build-gate.json';g.bind(bp,review,bg,root)
    assert g.verify_receipt(bg,base.Inputs(root),'BUILD_READY')['phase']=='BUILD_READY'
    transports={};workers={};transport_paths={};aux={}
    for kind,cohort,version in [('visible',visible,3),('n32',ids,2)]:
        directory=root/kind;directory.mkdir()
        output=table(kind+'/submission.csv',cohort);n=len(cohort)
        studies=[dict(index=i,uid_sha256=hashlib.sha256(u.encode()).hexdigest(),windows=2,error=None,
            prediction_fallback=0,end_to_end_seconds=1.,images_sha256='a'*64,metadata_sha256='b'*64) for i,u in enumerate(cohort)]
        timing=dict(schema='k2-timing/v1.0.1',fixed_seconds=2.,codec_header_seconds=n*.1,cohort_setup_seconds=.1,
            study_body_seconds=float(n),cumulative_journal_seconds=n*(n+1)*.0001,export_seconds=.1,
            residual_seconds=.1,studies=n,journal_rows=n*(n+1)//2)
        total=sum(v for k,v in timing.items() if k.endswith('_seconds'))
        memory=dict(used_before_bytes=2**30,total_bytes=16*2**30,source='measured before inference',device='NVIDIA T4')
        wr=dict(schema='k2-worker/v1',status='COMPLETE',version=w.VERSION,kernel_id=w.SLUG,contract=w.CONTRACT,
            manifest_sha256=w.sha(am),roster_sha256=w.digest(cohort),study_count=n,prediction_count=n,output_sha256=w.sha(output),
            fast_path=False,placeholder=False,fallbacks=dict(prediction=0),environment=dict(device='cuda:0',cuda_initialized=True,
                tf32=False,autocast=False,platform='Linux fixture',gpu='NVIDIA T4',versions=b.s6.WHEEL_VERSIONS,python_abi='cpython-313'),
            codecs=dict(status='ACTIVE',python_abi='cpython-313',syntaxes={'fixture':dict(decoded=True)}),studies=studies,
            total_seconds=total,timing=timing,peak_reserved_bytes=2**30,memory=memory)
        wp=put(kind+'/worker-receipt.json',wr);workers[kind]=wp
        boot=put(kind+'/bootstrap.json',dict(schema='k2-bootstrap/v1',total_seconds=total+2,
            fixed_startup_seconds=2.,manifest_sha256=w.sha(am),timing_policy=b.TIMING_POLICY['schema']))
        source=put(kind+'/remote.ipynb',b.notebook(w.sha(am)) if kind=='visible' else b.n32_notebook(reg))
        start=dt.datetime(2026,10,18,tzinfo=dt.timezone.utc)
        tr=dict(schema='k2-transport/v1',kind=kind,recorded_by='B',kernel=w.SLUG,kernel_version=version,fixture_only=False,
            status='COMPLETE',prepush=False,started_at=start.isoformat(),completed_at=(start+dt.timedelta(seconds=total+2.1)).isoformat(),
            remote_run_id='fixture-'+kind,remote_metadata=ref(put(kind+'/remote-metadata.json',dict(id=w.SLUG,versionNumber=version))),
            worker_receipt=ref(wp),submission=ref(output),bootstrap_timing=ref(boot),worker_log=ref(put(kind+'/worker.log',b'K2 real inference complete')),
            remote_notebook=ref(source),memory=memory)
        codec=put(kind+'/codec.json',dict(schema='k2-codec-qualification/v1',status='PASS',dataset=b.CODEC_DATASET,
            python_abi='cpython-313',compatible=True,asset_version=1,wheel_sha256={'fixture.whl':'1'*64}))
        tr['codec_asset_receipt']=ref(codec)
        if kind=='n32':
            hd=put(kind+'/full-diff.txt',b.harness_diff(reg).encode())
            hb=put(kind+'/harness.json',dict(schema='k2-n32-harness/v1',worker_sha256=w.sha(root/'tools/l1_rco_k2_worker_v101.py'),
                release_notebook_sha256=w.sha(nb),remote_notebook_sha256=w.sha(source),roster_sha256=w.digest(ids),
                allowed_changes=['train split','registered32 template','separate output directory'],complete_diff=ref(hd)))
            tr['harness_binding']=ref(hb);aux['harness']=hb;aux['n32_source']=source
        tp=put(kind+'/transport.json',tr);transports[kind]=ref(tp);transport_paths[kind]=tp
    reference=table('reference.csv',ids)
    pr=put('reference.json',dict(schema='k2-fp32-reference/v1',status='COMPLETE',model_sha256=w.MODEL_SHA,
        roster_sha256=w.digest(ids),output_sha256=w.sha(reference),contract=w.CONTRACT,source_sha256=w.sha(root/'tools/l1_rco_k2_v102.py'),
        runtime_version=w.VERSION,worker_sha256=b.WORKER_SHA,manifest_sha256=w.sha(am),environment=env,cuda_visible_devices='-1',
        registration_sha256=w.sha(regpath),tensors=[dict(images_sha256='a'*64,metadata_sha256='b'*64,windows=2)]*32))
    geometry=put('geometry.json',dict(schema='k2-geometry-parity/v1',status='PASS',roster_sha256=w.digest(ids),
        images_exact=True,metadata_exact=True,window_order_exact=True,production_route='cuda:0',model_sha256=w.MODEL_SHA,
        comparison=ref(put('geometry-comparison.json',dict(fixture=True)))))
    ts=root/'transport-source-manifest.json';a.create_transport_sources(bg,3,ts,root)
    bundle=put('bundle.json',dict(schema='k2-final-evidence/v1.0.2',transport_sources=ref(ts),fixture_only=False,candidate_id=b.CANDIDATE,build_gate=ref(bg),
        population_affirmation=dict(population=1300,recorded_by='B',applicable=True,basis=ref(put('basis.md',b'FIXTURE 1300'))),
        **transports,fp32_reference=ref(reference),fp32_reference_receipt=ref(pr),geometry_parity=ref(geometry)))
    return dict(root=root,stage=stage,reg=regpath,bundle=bundle,workers=workers,transports=transport_paths,
        aux=aux,reference=pr,ref=ref,put=put,cpu=cpu,parts=partpaths,bg=bg,bp=bp,builder=builder,build_review=review,transport_sources=ts,canonical=canonical,table=table)


def final_fixture(f,suffix=''):
    root=f['root'];fp=root/('final-pin'+suffix+'.json');a.pin(f['bundle'],fp,root)
    review=archive(root,'native-astra-r2','/root/fixture_final_reviewer','final-review'+suffix+'.md',
        f'Kernel-ID: {w.SLUG}\nReviewer-Lane: native-astra-r2\nK2-Final-Pin-SHA256: {w.sha(fp)}\nKernel-Version: 3\n'
        f'Candidate-ID: {b.CANDIDATE}\nReview-Phase: FINAL_READY\nDuplicate-Check: PASS\nDescription-Check: PASS\nMinimum-Submission-Review: PASS\nVERDICT: PASS\n')
    gatepath=root/'.local/state/gates/knee-l1-rcof0-v1.json'
    a.bind(fp,review,gatepath,root)
    observed=g.verify_receipt(gatepath,base.Inputs(root),'FINAL_READY')
    assert observed['qualified_for_submission'] is True
    receipt=f['workers']['visible']
    f['put']('visible/submission-binding.json',dict(schema_version='1.0.0',receipt_sha256=w.sha(receipt),kernel=w.SLUG,kernel_version=3,
        phase='FINAL_READY',qualified_for_submission=True,gate_sha256=w.sha(gatepath),registration=f['reg'].relative_to(root).as_posix(),registration_sha256=w.sha(f['reg'])))
    return receipt,review,gatepath


def qualify(f,receipt,review):return ks._qualify(b.CANDIDATE,w.SLUG,3,receipt,review,f['root'])


def test_complete_actual_schema_build_final_offline(production_schema):
    f=production_schema;receipt,review,_=final_fixture(f)
    first=qualify(f,receipt,review);second=qualify(f,receipt,review)
    assert first==second and first['phase']=='FINAL_READY' and first['kernel_version']=='3'
    assert str(f['canonical']) in first['input_sha256']
    build_inputs=base.Inputs(f['root']);g.verify_receipt(f['bg'],build_inputs,'BUILD_READY')
    assert 'tools/kaggle_submit_once.py' not in build_inputs.hashes()
    assert set(ks.EXECUTABLE_SOURCES)<=set(build_inputs.hashes())
    assert not any(x.name in ('ledger.jsonl','lock') for x in f['root'].iterdir())
    with pytest.raises((ValueError,OSError,KeyError)):
        ks.qualify(b.CANDIDATE,w.SLUG,3,receipt,review)


def test_two_canonical_versions_keep_build_but_require_fresh_final(production_schema):
    f=production_schema;receipt,review,gatepath=final_fixture(f)
    preserved={str(p):w.sha(p) for p in [f['bg'],f['bp'],f['cpu'],f['reg'],f['bundle'],f['transport_sources'],*f['parts'].values(),*f['transports'].values()]}
    f['canonical'].write_bytes(f['canonical'].read_bytes()+b'\n# isolated transport-only evolution\n')
    assert g.verify_receipt(f['bg'],base.Inputs(f['root']),'BUILD_READY')['phase']=='BUILD_READY'
    with pytest.raises(ValueError):qualify(f,receipt,review)
    ts=f['root']/'transport-source-manifest-v2.json';a.create_transport_sources(f['bg'],3,ts,f['root'])
    bundle=w.read(f['bundle']);bundle['transport_sources']=f['ref'](ts)
    new=f['put']('bundle-v2.json',bundle);f=dict(f,bundle=new)
    gatepath.rename(gatepath.with_name('old-final-preserved.json'))
    r,v,_=final_fixture(f,'-v2')
    assert qualify(f,r,v)['qualified_for_submission'] is True
    assert all(w.sha(Path(p))==h for p,h in preserved.items())


@pytest.mark.parametrize('name',['tools/l1_rco_k2_worker_v101.py','tools/l1_rco_k2_v102.py','tools/strict_variant_gate_k2_v102.py',
    'tools/strict_variant_gate_hb151.py','tools/native_codex_dispatch_record_v1.py','.local/run-scripts/b-preflight-k2-v102.py'])
def test_build_pinned_mutation_fails(production_schema,name):
    f=production_schema;p=f['root']/name;p.write_bytes(p.read_bytes()+b'\n# drift\n')
    with pytest.raises(ValueError):g.verify_receipt(f['bg'],base.Inputs(f['root']),'BUILD_READY')


@pytest.mark.parametrize('target',['worker','asset','registration'])
def test_asset_registration_mutation_fails(production_schema,target):
    f=production_schema
    p=f['reg'] if target=='registration' else b.ASSETS/('weights/E_prior_s0_f0.pt' if target=='asset' else 'source/l1_rco_k2_worker_v101.py')
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):g.verify_receipt(f['bg'],base.Inputs(f['root']),'BUILD_READY')


@pytest.mark.parametrize('sub_id,kernel',[(None,w.SLUG),('',w.SLUG),('bad id',w.SLUG),('other',w.SLUG),('K2-RCOF0-V1','u/other')])
def test_reserved_identity_both_directions(sub_id,kernel):
    with pytest.raises(ValueError):ks.qualify(sub_id,kernel,3,'missing','missing')


@pytest.mark.parametrize('schema',['k2-cpu-reuse/v1.0.1','k2-cpu-collection/v1'])
def test_old_cpu_schema_refused(production_schema,schema):
    f=production_schema;r=w.read(f['cpu']);r['schema']=schema
    with pytest.raises(ValueError):g.cpu_collection(r,base.Inputs(f['root']),f['stage'])


@pytest.mark.parametrize('defect',['missing','asset','builder','registration','cuda','run-id','attestation','runtime','raw-contract','raw-roster','raw-order','fallback','nan','target-order','tensors','same-runtime'])
def test_fresh_cpu_semantics_not_just_claimed_pass(production_schema,defect):
    f=production_schema;rec=w.read(f['cpu']);p=f['parts']['raw-holdout'];part=w.read(p);parent=p.parent
    if defect=='missing':del rec['parts']['model']
    elif defect=='same-runtime':rec['same_runtime']['max_abs']=.1
    else:
        if defect in ('asset','builder','registration'):
            part[{'asset':'manifest_sha256','builder':'builder_source_sha256','registration':'registration_sha256'}[defect]]='0'*64
        elif defect=='cuda':part['environment']['cuda_initialized']=True
        elif defect=='run-id':part['run_id']=w.read(f['parts']['model'])['run_id']
        elif defect=='attestation':
            att=w.read(parent/'execution-attestation.json');att['recorded_by']='builder';w.write(parent/'execution-attestation.json',att)
        elif defect in ('runtime','raw-contract','raw-roster','raw-order','fallback','tensors'):
            wr=w.read(parent/'worker-receipt.json')
            if defect=='runtime':wr['version']='1.0.0-rcof0'
            if defect=='raw-contract':wr['contract']=dict(w.CONTRACT,eval_stride=1)
            if defect=='raw-roster':wr['roster_sha256']='0'*64
            if defect=='raw-order':wr['studies'].reverse()
            if defect=='fallback':wr['fallbacks']['prediction']=1
            if defect=='tensors':wr['studies'][0]['images_sha256']='0'*64
            w.write(parent/'worker-receipt.json',wr);part['worker_receipt_sha256']=w.sha(parent/'worker-receipt.json')
        elif defect in ('nan','target-order'):
            out=parent/'submission.csv';lines=out.read_text().splitlines()
            if defect=='nan':lines[1]=lines[1].replace('0.5','nan',1)
            else:
                header=lines[0].split(',');header[1],header[2]=header[2],header[1];lines[0]=','.join(header)
            out.write_text('\n'.join(lines)+'\n');part['prediction_sha256']=w.sha(out)
            wr=w.read(parent/'worker-receipt.json');wr['output_sha256']=w.sha(out);w.write(parent/'worker-receipt.json',wr)
            part['worker_receipt_sha256']=w.sha(parent/'worker-receipt.json')
        w.write(p,part);rec['parts']['raw-holdout']=f['ref'](p)
        att=w.read(parent/'execution-attestation.json');att['result']=f['ref'](p);w.write(parent/'execution-attestation.json',att)
    with pytest.raises((ValueError,KeyError)):g.cpu_collection(rec,base.Inputs(f['root']),f['stage'])


@pytest.mark.parametrize('role',['builder','reviewer'])
@pytest.mark.parametrize('member',['completion-source','archive-completion','prompt-source','report-source','receipt','prompt','report','log'])
def test_complete_actual_native_archive_required(production_schema,role,member):
    f=production_schema;p=f['builder'] if role=='builder' else f['build_review'];r=w.read(str(p)+'.receipt.json')
    arc=f['root']/'.local/astra-improve/lanes/runs'/r['lane']/r['run_id'];c=w.read(r['completion_source'])
    paths={'completion-source':Path(r['completion_source']),'archive-completion':arc/'completion.json','prompt-source':Path(c['prompt']),
        'report-source':Path(c['report']),'receipt':arc/'receipt.json','prompt':arc/'prompt.txt','report':arc/'report.md','log':arc/'run.log'}
    paths[member].unlink()
    with pytest.raises((ValueError,OSError)):g.verify_receipt(f['bg'],base.Inputs(f['root']),'BUILD_READY')


@pytest.mark.parametrize('defect',['same-agent','phase','candidate','pin','fail'])
def test_real_native_success_not_independent_pass(production_schema,defect):
    f=production_schema;pin=w.read(f['bp']);agent='/root/fixture_builder' if defect=='same-agent' else '/root/fixture_new_reviewer'
    text=f'Kernel-ID: {w.SLUG}\nReviewer-Lane: native-astra-r3\nK2-Pin-SHA256: {w.sha(f["bp"])}\nCandidate-ID: {b.CANDIDATE}\nReview-Phase: BUILD_READY\nVERDICT: PASS\n'
    if defect=='phase':text=text.replace('Review-Phase: BUILD_READY','Review-Phase: FINAL_READY')
    if defect=='candidate':text=text.replace(b.CANDIDATE,'OTHER')
    if defect=='pin':text=text.replace(w.sha(f['bp']),'0'*64)
    if defect=='fail':text=text.replace('VERDICT: PASS','VERDICT: FAIL')
    p=archive(f['root'],'native-astra-r3',agent,'bad-review.md',text)
    with pytest.raises(ValueError):g.independent_review(p,f['bp'],pin['builder_native'],base.Inputs(f['root']))


@pytest.mark.parametrize('defect',['missing','drop-source','canonical-drift','wrong-version','wrong-build','fixture'])
def test_final_transport_manifest_mandatory(production_schema,defect):
    f=production_schema;bundle=w.read(f['bundle']);ts=w.read(f['transport_sources'])
    if defect=='missing':del bundle['transport_sources']
    else:
        if defect=='drop-source':del ts['source_sha256']['tools/strict_variant_gate_hb151.py']
        if defect=='canonical-drift':ts['source_sha256']['tools/kaggle_submit_once.py']='0'*64
        if defect=='wrong-version':ts['kernel_version']=2
        if defect=='wrong-build':ts['build_gate']['sha256']='0'*64
        if defect=='fixture':ts['fixture_only']=True
        w.write(f['transport_sources'],ts);bundle['transport_sources']=f['ref'](f['transport_sources'])
    w.write(f['bundle'],bundle)
    with pytest.raises((ValueError,KeyError)):a.pin(f['bundle'],f['root']/'invalid-final.json',f['root'])


def test_dependency_authenticated_before_import_and_after_cache(production_schema,monkeypatch):
    f=production_schema;r,v,_=final_fixture(f);qualify(f,r,v)
    p=f['root']/'tools/strict_variant_gate_hb151.py';p.write_bytes(p.read_bytes()+b'\n# mutation\n')
    monkeypatch.setattr(ks.importlib,'import_module',lambda *args:pytest.fail('must refuse before importing any validator'))
    with pytest.raises(ValueError,match='before import'):qualify(f,r,v)


@pytest.mark.parametrize('target',['canonical','review','dependency'])
def test_late_requalification_refuses_mutation(production_schema,target):
    f=production_schema;r,v,_=final_fixture(f);first=qualify(f,r,v)
    p=f['canonical'] if target=='canonical' else v if target=='review' else f['root']/'tools/l1_rco_k2_admission_v102.py'
    p.write_bytes(p.read_bytes()+b'\n# changed after initial qualification\n')
    with pytest.raises((ValueError,KeyError)):qualify(f,r,v)
    assert first['qualified_for_submission'] is True


def test_no_legacy_snapshot_or_live_canonical_in_build(production_schema,monkeypatch):
    from tools import strict_variant_gate_k2_v1 as old
    monkeypatch.setattr(old,'snapshot',lambda *args:pytest.fail('historical live snapshot forbidden'))
    f=production_schema;f['canonical'].unlink()
    inputs=base.Inputs(f['root']);g.verify_receipt(f['bg'],inputs,'BUILD_READY')
    assert 'tools/kaggle_submit_once.py' not in inputs.hashes()


def test_adapter_no_sender_or_cli_and_exact_api():
    import ast,inspect
    tree=ast.parse(Path(ks.__file__).read_bytes());names={x.name for x in tree.body if isinstance(x,ast.FunctionDef)}
    assert not names&{'main','submit_once','_submit_locked','send','context'}
    assert list(inspect.signature(ks.qualify).parameters)==['sub_id','kernel','version','receipt','review']
    calls={ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)}
    assert not any(x.endswith(('.main','.submit_once','._submit_locked')) for x in calls)


def reseal_stage(f):
    source=f['stage']/'source-manifest.json';reg=w.read(f['reg']);reg['source_manifest_sha256']=w.sha(source);w.write(f['reg'],reg)
    entries={p.relative_to(f['stage']).as_posix():w.sha(p) for p in f['stage'].rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'}
    (f['stage']/'SHA256SUMS.txt').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(entries.items())))


@pytest.mark.parametrize('defect',['missing-hb151','canonical-in-build','expanded-final','old-schema','stage-version','runtime-version','worker-version','asset-version','fixture-only'])
def test_registration_partition_and_versions_fail_even_coherently_rehashed(production_schema,defect):
    f=production_schema;source=w.read(f['stage']/'source-manifest.json');reg=w.read(f['reg'])
    if defect=='missing-hb151':del source['tools/strict_variant_gate_hb151.py']
    elif defect=='canonical-in-build':source['tools/kaggle_submit_once.py']=w.sha(f['canonical'])
    elif defect=='expanded-final':reg['final_only_sources'].append('tools/strict_variant_gate_hb151.py')
    elif defect=='old-schema':reg['schema']='k2-registration/v1'
    elif defect=='fixture-only':reg['fixture_only']=True
    else:reg[defect.replace('-','_') if defect!='runtime-version' else 'version']='BAD'
    w.write(f['stage']/'source-manifest.json',source);w.write(f['reg'],reg);reseal_stage(f)
    with pytest.raises((ValueError,KeyError)):g.snapshot(f['stage'],base.Inputs(f['root']))


@pytest.mark.parametrize('field',['registration_sha256','source_sha256','runtime_version','manifest_sha256','worker_sha256','roster_sha256'])
def test_full32_reference_exact_new_identity(production_schema,field):
    f=production_schema;r=w.read(f['reference']);r[field]='wrong';w.write(f['reference'],r)
    bundle=w.read(f['bundle']);bundle['fp32_reference_receipt']=f['ref'](f['reference']);w.write(f['bundle'],bundle)
    with pytest.raises((ValueError,KeyError)):a.pin(f['bundle'],f['root']/'bad-full32.json',f['root'])


@pytest.mark.parametrize('defect',['wrong-version','wrong-review','missing-binding','missing-gate'])
def test_nonfast_adapter_requires_exact_final_binding(production_schema,defect):
    f=production_schema;r,v,gatepath=final_fixture(f)
    if defect=='wrong-version':
        with pytest.raises(ValueError):ks._qualify(b.CANDIDATE,w.SLUG,2,r,v,f['root'])
        return
    if defect=='wrong-review':
        with pytest.raises(ValueError):ks._qualify(b.CANDIDATE,w.SLUG,3,r,f['build_review'],f['root'])
        return
    if defect=='missing-binding':(r.parent/'submission-binding.json').unlink()
    if defect=='missing-gate':gatepath.unlink()
    with pytest.raises((ValueError,OSError,KeyError)):qualify(f,r,v)


@pytest.mark.parametrize('mutation',['naive-time','future-time','bad-log-identity'])
def test_native_chronology_and_log_identity(production_schema,mutation):
    f=production_schema;p=f['build_review'];side=Path(str(p)+'.receipt.json');r=w.read(side)
    arc=f['root']/'.local/astra-improve/lanes/runs'/r['lane']/r['run_id']
    if mutation=='naive-time':r['recorded_at']='2026-10-18T00:00:00'
    elif mutation=='future-time':r['recorded_at']=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(days=1)).isoformat()
    else:
        log=w.read(arc/'run.log');log['agent_id']='/root/wrong_agent';w.write(arc/'run.log',log);r['log_sha256']=w.sha(arc/'run.log')
    w.write(side,r);w.write(arc/'receipt.json',r)
    with pytest.raises(ValueError):g.native_dispatch(side,base.Inputs(f['root']),b.NATIVE_SHA)


def test_native_source_guard_precedes_execution(production_schema):
    f=production_schema;p=f['root']/b.NATIVE;p.write_text('raise AssertionError("unverified helper executed")\n')
    with pytest.raises(ValueError,match='before use'):
        g.native_dispatch(str(f['builder'])+'.receipt.json',base.Inputs(f['root']),b.NATIVE_SHA)


def test_cpu_model_work_cannot_start_without_actual_implementation_admission(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'configure',lambda *args:pytest.fail('must block before torch/configure/model work'))
    with pytest.raises((ValueError,OSError,TypeError)):
        b.cpu_smoke('model',tmp_path/'model',admission=tmp_path/'missing.json')


def test_native_dispatch_unchanged_reviewed_source_and_no_old_reuse_branch():
    import ast
    root=Path(__file__).resolve().parents[1]
    def fn(p,name):
        text=p.read_text();n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.get_source_segment(text,n)
    assert fn(root/'tools/strict_variant_gate_k2_v102.py','native_dispatch')==fn(root/'tools/l1_rco_k2_local_v103.py','native_dispatch')
    text=(root/'tools/strict_variant_gate_k2_v102.py').read_text()
    assert 'old_gate.snapshot' not in text and 'base.dispatch(' not in text and 'reused_cpu_check' not in text
    for name in ('prediction_table','parity','ast_identity','extract_holdout','notebook','n32_notebook','harness_diff'):
        assert fn(root/'tools/l1_rco_k2_v101.py',name)==fn(root/'tools/l1_rco_k2_v102.py',name)
