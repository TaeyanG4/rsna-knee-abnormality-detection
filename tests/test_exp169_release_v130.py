"""v1.3.0 CPU fault injection on real DICOM windows and saved Public3 members."""
import ast
import csv
import copy
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import pytest
import torch
# Import before legacy fixtures snapshot sys.modules: TorchVision operators
# cannot be registered twice after those fixtures unload newly imported modules.
import timm
from tools import exp169_release_v6 as v

SAVED = v.ROOT / '.local/kout-v6-v121-A6q/v6-run'

@pytest.fixture
def members():
    rc = v.read(SAVED/'receipt.json')
    assert v.sha(SAVED/'members.npz') == rc['members_sha256']
    with np.load(SAVED/'members.npz') as z:
        return {k:z[k].copy() for k in z.files}

def blend(z):
    return v.release_values(v.candidate_specs(), 'A6q', z['ids'].tolist(),
        {k:z[k] for k in ('s42','s43','exp155','public')},
        {k:z[k] for k in ('l2','convnext','mil')},s6=z['s6'])

def test_real_healthy_replay_all_six_byte_identical(members,tmp_path):
    results, replacements = blend(members)
    assert replacements == []
    _, v5 = v.maths()
    for name, a in results.items():
        v5._write_csv(tmp_path/(name+'.csv'),members['ids'].tolist(),a)
        assert (tmp_path/(name+'.csv')).read_bytes() == (SAVED/(name+'.csv')).read_bytes()
        assert v.sha(tmp_path/(name+'.csv')) == v.read(SAVED/'receipt.json')['candidates'][name]['sha256']

@pytest.mark.parametrize('key',['s42','s43','exp155','public','s6','l2','convnext','mil'])
def test_real_member_nonfinite_replacement_and_receipt(members,key):
    a=members[key]
    if a.ndim == 3: a[0,0,0]=np.nan
    else: a[0,0]=np.inf
    values, records=blend(members)
    member='own' if key in ('s42','s43') else key
    rows=[r for r in records if r['member']==member]
    assert rows and all(np.isfinite(x).all() for x in values.values())
    for r in rows:
        assert r['UID']==members['ids'][0] and 'Traceback' in r['traceback']
        assert r['exception_type']=='FloatingPointError'
        donor='convnext' if key=='mil' else 'mil'
        assert r['donor_weights']=={donor:1.}

def test_simultaneous_failures_renormalise_original_weights_without_circular_donors(members):
    for k in ('l2','convnext','mil'): members[k][:,0]=np.nan
    values, records=blend(members)
    a=next(r for r in records if r['candidate']=='A6q')
    assert a['donor_weights']==dict(own=.25,exp155=.25,public=.25,s6=.25)
    b=next(r for r in records if r['candidate']=='B6')
    assert b['donor_weights']=={'own':1.}
    for r in records:
        assert not (set(r['donor_weights']) & {'l2','convnext','mil'})
    assert all(np.isfinite(x).all() for x in values.values())

def test_no_remaining_member_cannot_invent_prediction(members):
    for k,a in members.items():
        if k!='ids':
            if a.ndim==3:a[:,0]=np.nan
            else:a[0]=np.nan
    with pytest.raises(ValueError,match='no remaining member'):blend(members)

@pytest.fixture(scope='module')
def real_data():
    torch.set_num_threads(1)
    v.maths()
    partner=v.ROOT/v.B172/'assets'
    sys.path.insert(0,str(partner/'code'))
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_common import select_centers
    competition=v.ROOT/'.local/relv6-smoke/competition'
    with (competition/'test_series.csv').open() as stream: rows=list(csv.DictReader(stream))
    uid=rows[0][v.UID]
    corpus=DicomCorpus.from_records([uid],rows,DirectoryDicomSource(competition/'test_series'))
    volume,mask=corpus.volume(uid),corpus.mask(uid)
    centers=select_centers(mask,64,36,corpus.spec,'within-slot-valid')
    assert volume.any() and mask.sum()>=3
    _,windows=v.a4_policy()
    inputs=windows(volume,centers)
    ns=v.load_module('v130_cpu_anchor',v.ROOT/v.B172/'own-anchor/assets/public-serving.py').__dict__
    av,am=v.anchor_reader(ns)(uid,{uid:[r for r in rows if r[v.UID]==uid]},str(competition/'test_series'))
    return dict(uid=uid,rows=rows,corpus=corpus,inputs=inputs,centers=centers,
                volume=volume,mask=mask,ns=ns,av=av,am=am,images=str(competition/'test_series'))

class Model(torch.nn.Module):
    def __init__(self,mode='healthy'):super().__init__();self.mode=mode;self.calls=0
    def forward(self,x):
        self.calls+=1
        if self.mode=='exception':raise RuntimeError('injected model exception')
        return torch.full((1,12),float('nan') if self.mode=='nonfinite' else .3)

def fail(*a,**kw):raise OSError('injected study preparation failure')

@pytest.mark.parametrize('mode',['reader','windows','exception','nonfinite'])
def test_anchor_real_dicom_failure_paths(real_data,monkeypatch,mode):
    d=real_data; models={'s42':Model(mode),'s43':Model()}
    ns=dict(d['ns'])
    if mode=='windows':ns['gpu_windows']=fail
    monkeypatch.setattr(v,'anchor_models',lambda *a:(ns,models))
    monkeypatch.setattr(v,'anchor_reader',lambda ns:fail if mode=='reader' else lambda *a:(d['av'],d['am']))
    job=dict(ids=[d['uid']],rows=d['rows'],images=d['images'],paths={},candidate='B6')
    values=v.anchor_worker(job,'cpu')
    assert np.isnan(values['s42']).all() and job['failures']
    assert all(f['UID']==d['uid'] and 'Traceback' in f['traceback'] for f in job['failures'])
    if mode in ('exception','nonfinite'):
        assert np.isfinite(values['s43']).all() and models['s42'].calls==1 and models['s43'].calls==2

@pytest.mark.parametrize('mode',['reader','a4_windows','mil_windows','r50_exception','c4_nonfinite','mil_exception','mil_nonfinite'])
def test_partner_real_dicom_failure_paths(real_data,monkeypatch,mode):
    import raptor_dicom
    d=real_data
    class Corpus:
        spec=d['corpus'].spec
        def volume(self,uid):return fail() if mode=='reader' else d['volume']
        def mask(self,uid):return d['mask']
    monkeypatch.setattr(raptor_dicom.DicomCorpus,'from_records',lambda *a:Corpus())
    manifest=v.read(v.ROOT/'.local/kout-a4-v2/exp169-partner/release-manifest.json')
    monkeypatch.setattr(v,'read',lambda p:manifest)
    entries=v.q7_entries(manifest)
    models=[Model('exception' if mode=='r50_exception' and i==0 else 'nonfinite' if mode=='c4_nonfinite' and i==10 else 'healthy') for i in range(15)]
    monkeypatch.setattr(v,'a4_policy',lambda:(lambda *a:models, fail if mode=='a4_windows' else lambda *a:d['inputs']))
    monkeypatch.setattr(v,'load_mil',lambda *a:Model('exception' if mode=='mil_exception' else 'nonfinite' if mode=='mil_nonfinite' else 'healthy'))
    if mode=='mil_windows':monkeypatch.setattr(v,'mil_windows',fail)
    job=dict(ids=[d['uid']],rows=d['rows'],images=d['images'],paths=dict(partner='unused',mil='unused'),mil=[dict(name=str(i)) for i in range(10)])
    values=v.partner_worker(job,'cpu')
    assert job['failures'] and all('Traceback' in f['traceback'] for f in job['failures'])
    failed='l2' if mode in ('reader','a4_windows','r50_exception') else 'convnext' if mode=='c4_nonfinite' else 'mil'
    assert np.isnan(values[failed][0]).all()
    if mode not in ('reader','a4_windows'):
        assert np.isfinite(values['l2'][1:]).all()

def test_real_r50_checkpoint_cpu_forward_and_faults(real_data):
    d=real_data;v2,_=v.maths();partner=v.ROOT/v.B172/'assets'
    entry=next(e for e in v.read(partner/'release-manifest.json')['entries'] if e['family']=='l2')
    model=v2._load_models(partner,[entry],'cpu')[0]
    job={}
    good=v.study_prediction(job,d['uid'],'l2/s42/f0',lambda:v.infer(model,d['inputs']['l2'],'cpu'))
    assert np.isfinite(good).all() and not job.get('failures')
    for mode in ('exception','nonfinite'):
        def hook(module,args,output):
            if mode=='exception':raise RuntimeError('real checkpoint output hook failure')
            return torch.full_like(output,float('nan'))
        handle=model.register_forward_hook(hook)
        try: result=v.study_prediction(job,d['uid'],'l2/s42/f0',lambda:v.infer(model,d['inputs']['l2'],'cpu'))
        finally:handle.remove()
        assert np.isnan(result).all() and 'Traceback' in job['failures'][-1]['traceback']

@pytest.mark.parametrize('mode',['shared','models','models2','healthy'])
def test_s6_adapted_score_one_attempt_on_real_windows(real_data,tmp_path,mode):
    from tools import exp189_s6_port as s6
    original=s6._diagnostic_source
    with v.s6_one_attempt(s6):folder=s6._diagnostic_source(s6.SOURCE,tmp_path)
    assert s6._diagnostic_source is original
    code=(folder/'compact_infer.py').read_text()
    compile(code,str(folder/'compact_infer.py'),'exec')
    tree=ast.parse(code)
    score=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='score')
    calls=[]
    def run(m,*args):
        calls.append(m)
        return torch.full((1,12),float('nan') if mode=='models' or (mode=='models2' and m=='second') else .3)
    def shared(*args):calls.append('shared');return torch.full((2,12),float('nan'))
    import types
    ns=dict(torch=torch,a=types.SimpleNamespace(crop=224),dev=torch.device('cpu'),off=0,
            shared_trunk=mode=='shared',run_shared=shared,run_model=run,amp='fp32',models=['first'],models2=['second'] if mode=='models2' else [],W2=.5)
    exec(compile(ast.Module(body=[score],type_ignores=[]),'<s6-real-window-score>','exec'),ns)
    x=torch.from_numpy(real_data['inputs']['l2'])
    if mode=='healthy':assert torch.isfinite(ns['score'](x,torch.zeros((len(x),1)))[0]).all()
    else:
        with pytest.raises(FloatingPointError):ns['score'](x,torch.zeros((len(x),1)))
    assert len(calls)==(2 if mode=='models2' else 1)

@pytest.mark.parametrize('reason',['decode','nonfinite','startup'])
def test_s6_failure_receipt_discards_controller_mean(members,reason):
    try:
        if reason=='decode':raise OSError('DICOM failure')
        if reason=='nonfinite':raise FloatingPointError('nonfinite score')
        raise RuntimeError('worker startup failure')
    except Exception:tb=traceback.format_exc()
    uid=members['ids'][0];failures=[]
    values=v.s6_failed_rows(members['ids'].tolist(),members['s6'],dict(fallback_studies=1,fallback_reasons={uid:tb if reason!='startup' else 'worker missing output'}),failures)
    assert np.isnan(values[0]).all() and np.array_equal(values[1:],members['s6'][1:])
    assert failures[0]['UID']==uid and failures[0]['member']=='s6' and 'Traceback' in failures[0]['traceback']

def test_shard_failure_receipts_preserved(tmp_path,monkeypatch):
    class Process:
        returncode=0
        def poll(self):return 0
    def start(args,**kwargs):
        jp=Path(args[args.index('--job')+1]);op=Path(args[args.index('--out')+1]);job=v.read(jp)
        with op.open('wb') as stream:np.savez(stream,ids=job['ids'],s42=np.zeros((len(job['ids']),12)))
        v.write_new(str(op)+'.json',dict(sha256=v.sha(op),failures=[dict(UID=job['ids'][0],member='s42',exception_type='OSError',traceback='original traceback')]))
        return Process()
    monkeypatch.setattr(v.subprocess,'Popen',start)
    job=dict(ids=['a','b','c']);v.run_shards(job,tmp_path,'anchor')
    assert [f['UID'] for f in job['failures']]==['a','b']
