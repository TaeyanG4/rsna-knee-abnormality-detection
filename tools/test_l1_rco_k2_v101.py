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
from tools import l1_rco_k2_v101 as b
from tools import l1_rco_k2_admission_v101 as a
from tools import strict_variant_gate as base
from tools import strict_variant_gate_k2_v101 as g
from tools import kaggle_submit_once_k2_v101 as ks


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
    w.write(path,dict(schema='k2-gate/v1',status='PASS',errors=[],gate_version=g.VERSION,
        phase=phase,kernel_id=w.SLUG,qualified_for_submission=False,fixture_only=False))
    with pytest.raises(ValueError,match='phase'):g.verify_receipt(path,base.Inputs(tmp_path),'FINAL_READY')


def archive(root,lane,run,report,content):
    path=root/report;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
    arc=root/'.local/astra-improve/lanes/runs'/lane/run;arc.mkdir(parents=True)
    (arc/'report.md').write_bytes(path.read_bytes());(arc/'prompt.txt').write_text('TEST FIXTURE ONLY')
    receipt=dict(schema='dispatch-terminal/v1',state='SUCCESS',exit_code=0,lane=lane,run_id=run,
        out=report,report_sha256=w.sha(path),prompt_sha256=w.sha(arc/'prompt.txt'))
    w.write(arc/'receipt.json',receipt);w.write(str(path)+'.receipt.json',receipt)
    return path


@pytest.mark.parametrize('defect',['same-lane','wrong-pin','wrong-lane','no-archive','forged-pass'])
def test_independent_review_binding(tmp_path,defect):
    pin=tmp_path/'pin.json';pin.write_text('{}');lane='builder' if defect=='same-lane' else 'reviewer'
    text=f'{w.SLUG}\nReviewer-Lane: {lane}\nK2-Pin-SHA256: {w.sha(pin)}\nVERDICT: PASS\n'
    if defect=='wrong-pin':text=text.replace(w.sha(pin),'0'*64)
    if defect=='wrong-lane':text=text.replace('Reviewer-Lane: reviewer','Reviewer-Lane: other')
    if defect=='forged-pass':text=text.replace('VERDICT: PASS','VERDICT: FAIL')
    if defect=='no-archive':
        path=tmp_path/'review.md';path.write_text(text)
    else:path=archive(tmp_path,lane,'a'*32,'review.md',text)
    with pytest.raises((ValueError,OSError)):g.independent_review(path,pin,'builder',base.Inputs(tmp_path))


def test_private_namespaces_preserve_old_policies():
    from tools import kaggle_submit_once_v139 as old
    original=(base.kernel,base.source_binding,old.qualify,old._strict_gate,old.submit_once)
    x=g.context();y=ks.context()
    assert (base.kernel,base.source_binding,old.qualify,old._strict_gate,old.submit_once)==original
    assert x.source_binding is not base.source_binding and y.qualify is not old.qualify
    assert y.LEDGER==old.LEDGER and y.LOCK==old.LOCK


def test_nonfast_cannot_omit_final_gate(tmp_path):
    m=ks.context(tmp_path);p=tmp_path/'receipt.json'
    w.write(p,dict(status='COMPLETE',phase='FINAL_READY',qualified_for_submission=True))
    with pytest.raises((ValueError,OSError,KeyError)):m.qualify(w.SLUG,1,p,tmp_path/'review.md')


def test_build_receipt_cannot_submit(tmp_path):
    m=ks.context(tmp_path);p=tmp_path/'receipt.json'
    w.write(p,dict(status='COMPLETE',phase='BUILD_READY',qualified_for_submission=False))
    with pytest.raises(ValueError,match='FINAL_READY'):m.qualify(w.SLUG,1,p,tmp_path/'review.md')


def ordinary_evidence(root):
    reg=root/'reg.json';w.write(reg,dict(kernel_id='u/k',version='1.0.0'))
    receipt=root/'visible.json';w.write(receipt,dict(status='COMPLETE',version='1.0.0',kernel='u/k',kernel_version=2,
        registration=str(reg),registration_sha256=w.sha(reg),fallbacks=dict(prediction=0),codecs=dict(status='ACTIVE')))
    review=root/'review.md';review.write_text('u/k\nVERDICT: PASS\n')
    return receipt,review


class FakeCli:
    def __init__(self,ledger,fail=False):self.rows=[];self.sent=0;self.ledger=ledger;self.fail=fail
    def __call__(self,*args):
        if args[1]=='submissions':return 0,'\n'.join(['ref,fileName,date,description,status,publicScore,privateScore',*self.rows])
        assert json.loads(self.ledger.read_text().splitlines()[-1])['outcome']=='IN_FLIGHT'
        self.sent+=1
        if not self.fail:
            description=args[args.index('-m')+1]
            self.rows.append(f'1,submission.csv,2026-10-18 00:01:00,"{description}",SubmissionStatus.PENDING,,')
        return (1,'uncertain') if self.fail else (0,'accepted')


def execute(m,root,cli,receipt,review):
    return m.submit_once('test-id','u/k',2,'test',cli=cli,poll_tries=1,sleep=lambda _:None,
        now=dt.datetime(2026,10,18,tzinfo=dt.timezone.utc),ledger=root/'ledger.jsonl',lock=root/'lock',receipt=receipt,review=review)


def test_unchanged_non_k2_delegate_durable_one_send(tmp_path):
    m=ks.context(tmp_path);r,v=ordinary_evidence(tmp_path);cli=FakeCli(tmp_path/'ledger.jsonl')
    code,result=execute(m,tmp_path,cli,r,v)
    assert code==0 and result['outcome']=='SUBMITTED' and cli.sent==1
    code,_=execute(m,tmp_path,cli,r,v);assert code==2 and cli.sent==1


def test_uncertain_send_never_retries_and_inflight_blocks(tmp_path):
    m=ks.context(tmp_path);r,v=ordinary_evidence(tmp_path);cli=FakeCli(tmp_path/'ledger.jsonl',True)
    assert execute(m,tmp_path,cli,r,v)[0]==3 and cli.sent==1
    assert execute(m,tmp_path,cli,r,v)[0]==2 and cli.sent==1


def test_lock_contention(tmp_path):
    m=ks.context(tmp_path);r,v=ordinary_evidence(tmp_path);(tmp_path/'lock').write_text('held')
    cli=FakeCli(tmp_path/'ledger.jsonl');assert execute(m,tmp_path,cli,r,v)[0]==2 and cli.sent==0


def test_mutation_between_initial_and_presend(tmp_path):
    m=ks.context(tmp_path);r,v=ordinary_evidence(tmp_path);cli=FakeCli(tmp_path/'ledger.jsonl')
    def mutated(*args):
        if args[1]=='submissions':v.write_text('u/k\nVERDICT: FAIL\n')
        return cli(*args)
    assert execute(m,tmp_path,mutated,r,v)[0]==2 and cli.sent==0


@pytest.mark.parametrize('changes',[
    dict(kernel='u/wrong'),dict(kernel_version=0),dict(prepush=True),dict(fixture_only=True),
    dict(status='FAST_COMPLETE'),dict(kind='n32'),dict(recorded_by='builder')])
def test_transport_rejects_wrong_identity_before_files(tmp_path,changes):
    r=dict(schema='k2-transport/v1',kind='visible',recorded_by='B',kernel=w.SLUG,kernel_version=1,
        fixture_only=False,status='COMPLETE',prepush=False)
    r.update(changes);p=tmp_path/'transport.json';w.write(p,r)
    with pytest.raises(ValueError):a.transport(dict(path=str(p),sha256=w.sha(p)),base.Inputs(tmp_path),{},'visible')


def test_full_source_notebook_has_no_fast_or_constant_route():
    source=''.join(b.notebook('a'*64)['cells'][0]['source'])
    assert 'FAST_PATH_VISIBLE' not in source and 'len(' not in source
    assert '--device' in source and 'cuda:0' in source and '--no-index' in source


def test_missing_codec_fails_before_publication(tmp_path,monkeypatch):
    # Genuine decoder API failure is separate from a prediction fallback.
    import pydicom
    class Header:
        StudyInstanceUID='a'
        class file_meta:TransferSyntaxUID='1.2.840.10008.1.2.4.90'
    p=tmp_path/'test_series/a/series/x.dcm';p.parent.mkdir(parents=True);p.write_bytes(b'test-fixture')
    monkeypatch.setattr(pydicom,'dcmread',lambda *x,**y:Header())
    monkeypatch.setattr(pydicom.pixels,'get_decoder',lambda _:type('Decoder',(),{'is_available':False})())
    with pytest.raises(ValueError,match='codec'):w.codec_evidence(tmp_path,'test',['a'])


@pytest.fixture
def production_schema(tmp_path,monkeypatch):
    """Real pin/bind/replay logic, synthetic files in an isolated root only.

    Only file-input constants are redirected; no gate, receipt checker or
    computation is mocked. A fixture root/pins/archive is not project evidence.
    """
    realroot=b.ROOT;root=tmp_path.resolve()
    stage=root/'artifacts/experiments/L1_RCO_K2/v1.0.1-rcof0';work=root/'.local/scratch/slot-b/k2-release/fix1'
    assets=work/'assets-v1';stage.mkdir(parents=True);assets.mkdir(parents=True)
    monkeypatch.setattr(b,'ROOT',root);monkeypatch.setattr(b,'STAGE',stage);monkeypatch.setattr(b,'WORK',work)
    monkeypatch.setattr(b,'ASSETS',assets)
    def put(path,value):
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(value if isinstance(value,bytes) else w.encoded(value));return p
    def ref(p):return dict(path=p.relative_to(root).as_posix(),sha256=w.sha(p))
    def table(path,ids,k1=False):
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',newline='') as f:
            cw=csv.writer(f);cw.writerow([w.UID]+w.LABELS+(['status','model_fold'] if k1 else []))
            cw.writerows([[u]+[.5]*12+(['ok','0'] if k1 else []) for u in ids])
        return p
    source={}
    for name in b.NEW+b.FROZEN:
        p=put(name,(realroot/name).read_bytes());source[name]=w.sha(p)
    model=put('fixture/model.bin',b'NONMODEL TEST FIXTURE');anatomy=put('fixture/anatomy.bin',b'NONANATOMY TEST FIXTURE')
    monkeypatch.setattr(w,'MODEL_SHA',w.sha(model));monkeypatch.setattr(w,'ANATOMY_SHA',w.sha(anatomy))
    monkeypatch.setattr(w,'CONTRACT',dict(w.CONTRACT,model_sha256=w.MODEL_SHA))
    ids=['fixture'+str(i) for i in range(32)];visible=['visible'+str(i) for i in range(3)]
    saved=table('.local/scratch/l1-rco/k1v2/fit/holdout.csv',ids,True)
    monkeypatch.setattr(b,'PINS',{saved.relative_to(root).as_posix():w.sha(saved)})
    source.update(b.PINS)
    origins={};asset_files={}
    for original,name in [(model,'weights/E_prior_s0_f0.pt'),(anatomy,'weights/anatomy.pt'),
        (root/'tools/l1_rco_k2_worker_v101.py','source/l1_rco_k2_worker_v101.py')]:
        p=put((assets/name).relative_to(root),original.read_bytes());asset_files[name]=w.sha(p)
        origins[name]=dict(source=original.relative_to(root).as_posix(),sha256=w.sha(p),relation='byte-identical')
    put((work/'asset-origins.json').relative_to(root),origins)
    am=put((assets/'asset-manifest.json').relative_to(root),dict(contract=w.CONTRACT,version=w.VERSION,
        dataset=b.DATASET,private=True,wheels=b.s6.WHEEL_HASHES,deployment_versions=b.s6.WHEEL_VERSIONS,files=asset_files))
    put((assets/'dataset-metadata.json').relative_to(root),dict(id=b.DATASET,title=b.DATASET.split('/')[1]))
    sm=put((stage/'source-manifest.json').relative_to(root),source)
    reg=dict(schema='k2-registration/v1',candidate_id=b.CANDIDATE,kernel_id=w.SLUG,version=w.VERSION,
        contract=w.CONTRACT,qualified_for_submission=False,assets=assets.relative_to(root).as_posix(),
        asset_manifest_sha256=w.sha(am),source_manifest_sha256=w.sha(sm),visible_ids=visible,n32_ids=ids,
        smoke_holdout_ids=ids[:2],n32_roster_sha256=w.digest(ids),runtime=dict(b.TIMING_POLICY,
            population_source='.local/scratch/slot-b/k2-release/POPULATION-SOURCE.json',population_affirmation_required=True,
            workload_rule='complete studies and cumulative journals; all residual scaled'))
    regpath=put((stage/'registration.json').relative_to(root),reg)
    nb=put((stage/'kernel/release.ipynb').relative_to(root),b.notebook(w.sha(am)))
    meta=put((stage/'kernel/kernel-metadata.json').relative_to(root),dict(id=w.SLUG,title=w.SLUG.split('/')[1],
        code_file='release.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=False,dataset_sources=[b.DATASET,b.CODEC_DATASET],competition_sources=['rsna-knee-abnormality-detection'],kernel_sources=[]))
    diff=put((stage/'full-inventory-diff.txt').relative_to(root),b'TEST FIXTURE INPUT INVENTORY\n')
    put((stage/'embedded-file-manifest.json').relative_to(root),dict(new_sources={n:source[n] for n in b.NEW},
        inherited_assets=origins,removed=[],changed_inherited=[],notebook_sha256=w.sha(nb),metadata_sha256=w.sha(meta),
        complete_diff_sha256=w.sha(diff)))
    seal={p.relative_to(stage).as_posix():w.sha(p) for p in stage.rglob('*') if p.is_file()}
    put((stage/'SHA256SUMS.txt').relative_to(root),''.join(h+'  '+n+'\n' for n,h in sorted(seal.items())).encode())
    env=dict(device='cpu',cuda_initialized=False);tensors=[dict(images_sha256='a'*64,metadata_sha256='b'*64,windows=2)]*2
    parts={}
    for part in ('model','spawn','raw-visible','raw-holdout','cached-reference'):
        directory=root/'cpu'/part;directory.mkdir(parents=True)
        r=dict(part=part,status='PASS',environment=env,cuda_visible_devices='-1',registration_sha256=w.sha(regpath))
        if part=='model':r['initialization_plus_deploy_exact']=True
        if part=='spawn':r.update(workers=[0,3],studies=2,exact=True)
        if part.startswith('raw-'):
            cohort=visible if part=='raw-visible' else ids[:2];p=table((directory/'submission.csv').relative_to(root),cohort)
            wr=put((directory/'worker-receipt.json').relative_to(root),dict(status='COMPLETE',fallbacks=dict(prediction=0),
                study_count=len(cohort),output_sha256=w.sha(p),studies=tensors if len(cohort)==2 else [tensors[0]]*3))
            r.update(worker_receipt_sha256=w.sha(wr),prediction_sha256=w.sha(p))
        if part=='cached-reference':
            p=table((directory/'reference.csv').relative_to(root),ids[:2]);r.update(reference_sha256=w.sha(p),
                k1_bf16=dict(status='PASS'),studies=2,tensors=tensors)
        parts[part]=ref(put((directory/'result.json').relative_to(root),r))
    cpu=put('cpu/collection.json',dict(schema='k2-cpu-collection/v1',status='PASS',qualified_for_submission=False,
        raw_cached_images_exact=True,same_runtime=b.parity(np.full((2,12),.5),np.full((2,12),.5),'same_runtime'),
        registration_sha256=w.sha(regpath),parts=parts))
    material=g.snapshot(stage,base.Inputs(root));builder=archive(root,'fixture-builder','1'*32,'build.md',
        'Kernel-SHA256: '+material['kernel_sha256']+'\nFIXTURE ONLY\n')
    bp=root/'build-pin.json';g.pin(stage,cpu,str(builder)+'.receipt.json',bp,root)
    review=archive(root,'fixture-reviewer','2'*32,'build-review.md',
        f'{w.SLUG}\nReviewer-Lane: fixture-reviewer\nK2-Pin-SHA256: {w.sha(bp)}\nVERDICT: PASS\n')
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
        roster_sha256=w.digest(ids),output_sha256=w.sha(reference),contract=w.CONTRACT,source_sha256=w.sha(root/'tools/l1_rco_k2_v101.py'),
        registration_sha256=w.sha(regpath),tensors=[dict(images_sha256='a'*64,metadata_sha256='b'*64,windows=2)]*32))
    geometry=put('geometry.json',dict(schema='k2-geometry-parity/v1',status='PASS',roster_sha256=w.digest(ids),
        images_exact=True,metadata_exact=True,window_order_exact=True,production_route='cuda:0',model_sha256=w.MODEL_SHA,
        comparison=ref(put('geometry-comparison.json',dict(fixture=True)))))
    bundle=put('bundle.json',dict(schema='k2-final-evidence/v1',fixture_only=False,candidate_id=b.CANDIDATE,build_gate=ref(bg),
        population_affirmation=dict(population=1300,recorded_by='B',applicable=True,basis=ref(put('basis.md',b'FIXTURE 1300'))),
        **transports,fp32_reference=ref(reference),fp32_reference_receipt=ref(pr),geometry_parity=ref(geometry)))
    return dict(root=root,stage=stage,reg=regpath,bundle=bundle,workers=workers,transports=transport_paths,
        aux=aux,reference=pr,ref=ref,put=put)


def final_fixture(f):
    root=f['root'];fp=root/'final-pin.json';a.pin(f['bundle'],fp,root)
    review=archive(root,'fixture-final-reviewer','3'*32,'final-review.md',
        f'{w.SLUG}\nReviewer-Lane: fixture-final-reviewer\nK2-Final-Pin-SHA256: {w.sha(fp)}\nKernel-Version: 3\n'
        f'Candidate-ID: {b.CANDIDATE}\nDuplicate-Check: PASS\nDescription-Check: PASS\nMinimum-Submission-Review: PASS\nVERDICT: PASS\n')
    gatepath=root/'.local/state/gates'/f'{w.SLUG.split("/")[1]}.json'
    a.bind(fp,review,gatepath,root)
    observed=g.verify_receipt(gatepath,base.Inputs(root),'FINAL_READY')
    assert observed['qualified_for_submission'] is True
    receipt=f['workers']['visible']
    f['put']('visible/submission-binding.json',dict(schema_version='1.0.0',receipt_sha256=w.sha(receipt),
        kernel=w.SLUG,kernel_version=3,phase='FINAL_READY',qualified_for_submission=True,gate_sha256=w.sha(gatepath),
        registration=f['reg'].relative_to(root).as_posix(),registration_sha256=w.sha(f['reg'])))
    return ks.context(root),receipt,review


def test_complete_k2_production_schema_pin_bind_final_delegate(production_schema):
    f=production_schema;m,receipt,review=final_fixture(f);root=f['root'];cli=FakeCli(root/'ledger.jsonl')
    assert m.qualify(w.SLUG,3,receipt,review)['phase']=='FINAL_READY'
    code,r=m.submit_once(b.CANDIDATE,w.SLUG,3,'synthetic fixed formula',cli=cli,poll_tries=1,sleep=lambda _:None,
        now=dt.datetime(2026,10,18,tzinfo=dt.timezone.utc),ledger=root/'ledger.jsonl',lock=root/'lock',receipt=receipt,review=review)
    assert code==0 and r['outcome']=='SUBMITTED' and cli.sent==1
    # Production verification has actual root/pins and rejects this isolated fixture.
    with pytest.raises((ValueError,OSError)):
        g.verify_receipt(root/'.local/state/gates'/f'{w.SLUG.split("/")[1]}.json',base.Inputs(g.ROOT),'FINAL_READY')


@pytest.mark.parametrize('defect',['n32-source','missing-timing','wrong-parity','codec-abi','stale-version'])
def test_late_transport_failure_through_actual_final_pin(production_schema,defect):
    f=production_schema;root=f['root'];ref=f['ref'];bundle=w.read(f['bundle'])
    kind='n32' if defect=='n32-source' else 'visible';tp=f['transports'][kind];tr=w.read(tp)
    if defect=='n32-source':
        p=f['aux']['n32_source'];nb=w.read(p);nb['cells'][0]['source'].append('raise RuntimeError("changed predictor")\n');w.write(p,nb)
        tr['remote_notebook']=ref(p);hb=f['aux']['harness'];h=w.read(hb);h['remote_notebook_sha256']=w.sha(p);w.write(hb,h);tr['harness_binding']=ref(hb)
    elif defect in ('missing-timing','stale-version'):
        p=f['workers'][kind];wr=w.read(p)
        if defect=='missing-timing':del wr['timing']['codec_header_seconds']
        else:wr['version']='1.0.0-rcof0'
        w.write(p,wr);tr['worker_receipt']=ref(p)
    elif defect=='codec-abi':
        p=root/tr['codec_asset_receipt']['path'];c=w.read(p);c['python_abi']='cpython-312';w.write(p,c);tr['codec_asset_receipt']=ref(p)
    elif defect=='wrong-parity':
        p=f['reference'];r=w.read(p);r['contract']=dict(w.CONTRACT,crop=[0,352]);w.write(p,r);bundle['fp32_reference_receipt']=ref(p)
    w.write(tp,tr);bundle[kind]=ref(tp);w.write(f['bundle'],bundle)
    with pytest.raises((ValueError,KeyError)):a.pin(f['bundle'],root/'must-not-pin.json',root)


def test_stale_final_review_and_presend_mutation(production_schema):
    f=production_schema;m,receipt,review=final_fixture(f);root=f['root'];cli=FakeCli(root/'ledger.jsonl')
    assert m.qualify(w.SLUG,3,receipt,review)['phase']=='FINAL_READY'
    def mutate(*args):
        if args[1]=='submissions':review.write_text(review.read_text().replace('VERDICT: PASS','VERDICT: FAIL'))
        return cli(*args)
    code,_=m.submit_once(b.CANDIDATE,w.SLUG,3,'fixture',cli=mutate,poll_tries=1,sleep=lambda _:None,
        now=dt.datetime(2026,10,18,tzinfo=dt.timezone.utc),ledger=root/'ledger.jsonl',lock=root/'lock',receipt=receipt,review=review)
    assert code==2 and cli.sent==0
