"""H-B-166 serving parity v1.0.0; real pixels and predictions, no labels/GPU."""
from __future__ import annotations
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import hb166_release_v1 as b, hb166_serving_v1 as s

def extract(ids, destination):
    from training.src import raptor_corpus as raw
    arc,index,_=raw.archive_paths()
    with sqlite3.connect(f'file:{index}?mode=ro',uri=True) as con, raw.IndexedZip(arc) as z:
        for uid in ids:
            rows=con.execute('SELECT path,series,info FROM files WHERE split=? AND uid=? ORDER BY path',('train',uid)).fetchall()
            s.need(bool(rows),'missing real DICOM study')
            for name,series,info in rows:
                target=destination/'train_images'/uid/series/Path(name).name
                s.need(target.resolve().is_relative_to(destination.resolve()),'unsafe archive path')
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(raw.zip_info(name,info)) as f: data=f.read()
                if not target.exists() or target.read_bytes()!=data: target.write_bytes(data)
    return raw.series_records()

def replay(preprocess_only=False):
    import numpy as np
    import pandas as pd
    import torch
    from training.src import raptor_common as common
    from training.arms import hb166_multiview_v1 as mv, mil
    s.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU isolation missing')
    reg=s.read(b.WORK/'registration.json'); s.need(reg==b.registered(),'register first')
    s.need(s.read(b.FIT/'heartbeat.json')['status']=='complete','running fit output forbidden')
    s.need(s.sha(b.FIT/'val.receipt.json')==reg['fit_receipt_sha256'],'fit receipt drift')
    receipt=s.read(b.FIT/'val.receipt.json')
    s.need(s.sha(b.FIT/'val.csv')==reg['fit_csv_sha256']==receipt['csv_sha256'],'fit output drift')
    cfg=b.config(); ids=reg['cohort']; paths=[ROOT/cfg[x] for x in ('corpus','crop_corpus')]
    for p,h in zip(paths,reg['cache_manifest_sha256']): s.need(s.sha(p/'manifest.json')==h,'cache drift')
    full,crop=[common.Corpus(p) for p in paths]
    for corpus in (full,crop):
        for name in ('ids','masks'):
            if name+'_sha256' in corpus.manifest:
                s.need(s.sha(corpus.dir/(name+'.npy'))==corpus.manifest[name+'_sha256'],'cache roster drift')
    src=b.source_payload(); (b.WORK/'hb166-sources.json').write_bytes(src)
    r=s.recipe(b.WORK/'hb166-sources.json',hashlib.sha256(src).hexdigest())
    data=b.WORK/'real-data'; rows=extract(ids,data); s.configure_transport(r,data)
    # Actual hidden directory convention, linked to the same preserved DICOMs.
    for uid in ids:
        for original in (data/'train_images'/uid).glob('*/*.dcm'):
            target=data/'test_series'/original.relative_to(data/'train_images')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): os.link(original,target)
            s.need(s.sha(original)==s.sha(target),'real test transport drift')
    results=[]; served=[]
    labels={u:np.zeros(12,np.float32) for u in ids}
    offline=mv.make_dataset(*paths,ids,labels,36,False,42,224)
    offline.full.corpus,offline.crop.corpus=full,crop
    for i,uid in enumerate(ids):
        f,c,provenance=s.preprocess(r,uid,rows[('train',uid)],'train')
        tf,tc,_=s.preprocess(r,uid,rows[('train',uid)],'test')
        for train,test in ((f,tf),(c,tc)):
            s.need(np.array_equal(train.volume(uid),test.volume(uid)) and
                   np.array_equal(train.mask(uid),test.mask(uid)),'real test_series parity failed')
        for a,z in ((f,full),(c,crop)):
            s.need(np.array_equal(a.volume(uid),z.volume(uid)),'raw uint8 parity failed')
            s.need(np.array_equal(a.mask(uid),z.mask(uid)),'raw mask parity failed')
        x,_,_,views=s.dataset(r,[uid],f,c)[0]; ox,_,_,ov=offline[i]
        s.need(torch.equal(x,ox) and torch.equal(views,ov),'normalized windows parity failed')
        results.append(dict(full_sha256=hashlib.sha256(f.volume(uid).tobytes()).hexdigest(),
            crop_sha256=hashlib.sha256(c.volume(uid).tobytes()).hexdigest(),
            window_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),shape=list(x.shape),uint8_exact=True,masks_exact=True,windows_exact=True,test_series_exact=True))
        served.append((f,c))
        print(json.dumps(dict(study_index=i,raw_and_windows='PASS')),flush=True)
    result=dict(version='1.0.0',registration_sha256=s.sha(b.WORK/'registration.json'),studies=results,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],exposure='EXPOSED',clean_oof=False)
    if not preprocess_only:
        started=time.monotonic(); entry=b.manifest()['roster'][0]
        sm=s.model(r,b.FIT/'final.pt',entry,'cpu')
        sp=np.concatenate([s.predict(r,sm,[u],f,c,'cpu') for u,(f,c) in zip(ids,served)])
        del sm
        print(json.dumps(dict(served_cpu='complete',seconds=time.monotonic()-started)),flush=True)
        om=mv.attach_views(mil.FindingMIL(common.build_backbone('convnext_small.fb_in22k_ft_in1k',pretrained=False)))
        state=torch.load(b.FIT/'final.pt',map_location='cpu',weights_only=True)
        s.need(state['identity']==receipt['identity'],'fit/checkpoint identity differs')
        mil.strict_load(om,state['model']); om=om.to(memory_format=torch.channels_last).eval()
        loader=torch.utils.data.DataLoader(offline,batch_size=1,shuffle=False,num_workers=0,collate_fn=mv.collate)
        op,order,_=mv.evaluate(mv.ARM,om,loader,'cpu','fp32',full,crop,s.Progress(),tta=True)
        s.need(order==ids,'offline ID order differs')
        saved=pd.read_csv(b.FIT/'val.csv')
        s.need(saved.columns.tolist()==[common.UID,*common.LAB] and not saved[common.UID].duplicated().any(),'saved prediction schema')
        fp=saved.set_index(common.UID).loc[ids,common.LAB].to_numpy()
        exact=float(np.max(abs(sp-op))); portability=float(np.max(abs(sp-fp)))
        result.update(cpu_maxabs=exact,saved_fp16_maxabs=portability,cpu_atol=reg['cpu_atol'],saved_fp16_atol=reg['saved_fp16_atol'],seconds=time.monotonic()-started)
        np.savez(b.WORK/'cpu-predictions.npz',ids=np.asarray(ids),targets=np.asarray(common.LAB),served=sp,offline=op,saved=fp)
        result['predictions_sha256']=s.sha(b.WORK/'cpu-predictions.npz')
        result['status']='PASS' if exact<=reg['cpu_atol'] and portability<=reg['saved_fp16_atol'] else 'FAIL'
    else: result['status']='PREPROCESS_PASS'
    output=b.WORK/('preprocess-replay.json' if preprocess_only else 'cpu-replay.json')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    s.need(result['status']!='FAIL','registered prediction tolerance failed')
    print(json.dumps({k:v for k,v in result.items() if k!='studies'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--preprocess-only',action='store_true'); a=p.parse_args(); replay(a.preprocess_only)
