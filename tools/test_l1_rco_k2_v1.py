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

from tools import l1_rco_k2_worker_v1 as w
from tools import l1_rco_k2_v1 as b
from tools import l1_rco_k2_admission_v1 as a
from tools import strict_variant_gate as base
from tools import strict_variant_gate_k2_v1 as g
from tools import kaggle_submit_once_k2_v1 as ks


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
    r=a.runtime_projection(dict(setup_seconds=20,mean_e2e=per),dict(setup_seconds=30,mean_e2e=per))
    assert r['projected_seconds']==30+1300*per and r['status']==expected


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
