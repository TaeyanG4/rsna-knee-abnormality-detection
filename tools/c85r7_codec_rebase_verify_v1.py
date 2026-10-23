"""CPU replay v1.0.0: authenticated real visible predictions; no labels or CUDA."""
from pathlib import Path
import copy
import io
import json
import shutil
import sys
import types
import numpy as np
import pandas as pd
from tools import c85r7_codec_rebase_v1 as m
from tools import c85r7_lb_r2_adapter_v1 as adapter
h,b=m.h,m.b

def reset(path):
    path=Path(path).resolve()
    b.need(path.is_relative_to(m.SCRATCH.resolve()) and path!=m.SCRATCH.resolve(),'scratch boundary')
    if path.exists(): shutil.rmtree(path)
    path.mkdir(parents=True)
    return path

def module(tag, source=None):
    name, original=m.runtime_source(tag)
    path=m.SCRATCH/'modules'/(name+'.py'); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(source or original,encoding='utf-8')
    mod=types.ModuleType(name);mod.__file__=str(path)
    sys.modules[name]=mod
    exec(compile(source or original,mod.__file__,'exec'),mod.__dict__)
    mod.ROOT=m.ROOT
    return mod

def recorded():
    native=m.ROOT/'.local/scratch/sol-1/hb116-runtime/cpu-NATIVE'
    rec=h.read(native/'receipt.json')
    b.need(rec['status']=='PASS' and rec['phases']['NATIVE']['exit_codes']==[0],'native completed receipt')
    h.pin(native/'raw.npz',{'sha256':rec['raw_sha256']})
    job=h.read(native/'NATIVE/worker0/job.json')
    with np.load(native/'raw.npz',allow_pickle=False) as z:
        ids=z['ids'].tolist(); b.need(z['targets'].tolist()==h.LABELS and ids==job['ids'],'native schema')
        raw={k:z[k].copy() for k in z.files if k not in ('ids','targets')}
    j,failed=h.c.read_journal(native/'NATIVE/worker0',ids,list(raw),job['identity'])
    b.need(all(not failed[k] and np.array_equal(raw[k],j[k].astype(np.float32)) for k in raw),'native journal authentication')
    r7dir=m.ROOT/'.local/scratch/sol-1/hb127/r7-smoke'
    r7receipt=m.ROOT/'artifacts/experiments/HB116_VNEXT/v1.2.0/release/validation/r7-smoke.json'
    rec7=h.read(r7receipt);b.need(rec7['status']=='PASS' and rec7['studies']==3,'completed R7 CPU receipt')
    h.pin(r7dir/'combined-raw.npz',{'sha256':rec7['combined_raw_sha256']})
    with np.load(r7dir/'combined-raw.npz',allow_pickle=False) as z:
        b.need(z['ids'].tolist()==ids and z['targets'].tolist()==h.LABELS,'R7 schema')
        raw['S_R7']=z['S_R7'].copy()
    rids,_,r7,info=b.authenticated_pr7(r7dir/'public')
    b.need(rids==ids and not info['failures']['S_R7'],'genuine R7 identity')
    np.testing.assert_array_equal(raw['S_R7'],r7.astype(np.float32))
    b.p.check_pin(b.p.REPLAY/'p956-receipt.json',b.c85.P956_REPLAY_SHA)
    pids,r,_,rinfo=b.c85.authenticated_p956(b.p.REPLAY)
    b.need(set(pids)==set(ids) and not rinfo['failures']['R'],'original R authentication')
    raw['R']=r[[pids.index(u) for u in ids]]
    b.p.check_pin(b.c85.A5Q/'exp169-executed-candidate.json',b.c85.A5Q_RECEIPT_SHA)
    aids,x,y,q,a=b.c85.authenticated_a172(b.c85.A5Q)
    b.need(set(aids)==set(ids) and all(not f for f in a['failures'].values()),'original O authentication')
    order=[aids.index(u) for u in ids]
    original=(x[order],y[order],q[order])
    reg=adapter.registration()
    for key,sha in adapter.identity(reg)['members'].items():
        b.need(job['identity']['members'][key]==sha,'R2 checkpoint identity')
    return reg,ids,raw,original,a,info,dict(native_raw_sha256=h.sha(native/'raw.npz'),r7_raw_sha256=rec7['combined_raw_sha256'],original_r_receipt_sha256=b.c85.P956_REPLAY_SHA,original_o_receipt_sha256=b.c85.A5Q_RECEIPT_SHA)

def rank(x): return pd.DataFrame(x).rank(method='average',pct=True).to_numpy()

def reference(tag,reg,raw,own):
    r=raw['R']
    if tag=='r2': r=np.mean([raw['R2-s43'],raw['R2-s44']],axis=0,dtype=np.float32)
    public=(rank(r)+rank(raw['S_R7']))/2
    if tag=='fluidR': public[:,[7,9]]=rank(r)[:,[7,9]]
    o=(17*rank(own[0])+17*rank(own[1])+12*rank(own[2]))/46
    w=.25 if tag=='own25' else .15
    return (1-w)*rank(public)+w*rank(o)

def r2_journal(path,reg,ids,raw):
    journal=h.c.Journal(path/'NATIVE/worker0',ids,adapter.identity(reg))
    for k in adapter.identity(reg)['members']:
        for uid,values in zip(ids,raw[k]): journal.record(uid,k,values)

def replay(tag):
    reg,ids,raw,own,a,info,pins=recorded()
    work=reset(m.SCRATCH/('replay-'+tag)); runtime=module(tag)
    r=raw['R']
    if tag=='r2':
        r2_journal(work/'input/pr7-run/r2',reg,ids,raw)
        r,failed,ident=adapter.recover(work/'input/pr7-run/r2',ids,reg)
        b.need(not failed and ident['clean_oof'] is False,'R2 survivor replay')
        np.testing.assert_array_equal(r,np.mean([raw['R2-s43'],raw['R2-s44']],axis=0,dtype=np.float32))
    got=runtime.blend(ids,r,raw['S_R7'],*own)
    neutral_module=module(tag)
    exec(m.math_source(.15,False),neutral_module.__dict__)
    neutral=neutral_module.blend(ids,raw['R'],raw['S_R7'],*own)
    parent=b.blend(ids,raw['R'],raw['S_R7'],*own)
    runtime=module(tag)
    info=copy.deepcopy(info);info['failures']['R']={}; info['r_prediction_fallbacks']=0
    original=runtime.c85.authenticated_a172
    runtime.c85.authenticated_a172=lambda _: (ids,*own,copy.deepcopy(a))
    if tag=='r2':
        runtime._original_authenticated_pr7=lambda _: (ids,raw['R'],raw['S_R7'],copy.deepcopy(info))
    else:
        runtime.authenticated_pr7=lambda _: (ids,r,raw['S_R7'],copy.deepcopy(info))
    try:
        receipt,published=runtime.collect(b.c85.A5Q,work/'input',work/'export')
    finally:
        runtime.c85.authenticated_a172=original
    b.need((work/'export/submission.csv').read_bytes()==published,'collector publication')
    expected=reference(tag,reg,raw,own)
    payload=b.p.csv_bytes(ids,got)
    b.need(payload==b.p.csv_bytes(ids,expected),'independent formula CSV bytes')
    b.need(published==payload,'runtime published CSV bytes')
    b.need(b.p.csv_bytes(ids,neutral)==b.p.csv_bytes(ids,parent),'parent no-op CSV bytes')
    table=pd.read_csv(io.BytesIO(payload))
    b.need(table.shape==(3,13) and table.columns.tolist()[1:]==h.LABELS,'CSV schema')
    b.need(table.iloc[:,0].tolist()==ids,'original visible order')
    values=table.iloc[:,1:].to_numpy()
    b.need(np.isfinite(values).all() and (values>=0).all() and (values<=1).all(),'CSV probabilities')
    b.need(receipt['formula']==m.SPECS[tag]['formula'] and receipt['clean_oof'] is False,'formula/exposure receipt')
    if tag=='r2': b.need(receipt['pr7']['r2_identity']['members']==adapter.identity(reg)['members'],'collector R2 identity')
    result=dict(status='PASS_CPU_REPLAY',tag=tag,studies=3,targets=12,ordered=True,independent_formula_csv_bytes=True,
        parent_noop_csv_bytes=True,published_csv_bytes=True,csv_sha256=m.digest(payload),input_pins=pins,
        scope='Real authenticated visible predictions; CPU serving arithmetic, journal recovery and publication; FLIP_ALL uses plain recorded R7 predictions, scoring transform tested separately; no CUDA')
    m.dump(work/'result.json',result)
    return result
