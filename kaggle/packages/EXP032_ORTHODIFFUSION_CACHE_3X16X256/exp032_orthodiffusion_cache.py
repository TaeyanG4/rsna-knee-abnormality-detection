from __future__ import annotations
import os, json, time, hashlib, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pydicom
import cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut

ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
OUT = Path('/kaggle/working')
IMG = 256
DEPTH = 16
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 1))
NSHARD = 8
SLOTS = [('SAG','Sagittal'),('COR','Coronal'),('AX','Axial')]
HEADER_TAGS = ['ImagePositionPatient','ImageOrientationPatient','InstanceNumber','PixelSpacing','Rows','Columns','PhotometricInterpretation','RescaleSlope','RescaleIntercept']

def count_dcm(folder: Path) -> int:
    try: return sum(1 for _ in folder.glob('*.dcm'))
    except Exception: return 0

def pick_series(rows, plane, study_dir):
    cand = [r for r in rows if str(r.get('Anatomical_Plane')) == plane]
    if not cand: return None, 'missing'
    fluid = [r for r in cand if int(r.get('Fluid_Sensitive', r.get('Fat_Suppression',0)) or 0) == 1]
    pool = fluid if fluid else cand
    scored = [(count_dcm(study_dir/str(r['SeriesInstanceUID'])), str(r['SeriesInstanceUID']), r) for r in pool]
    scored.sort(key=lambda z:(-z[0],z[1]))
    return (scored[0][2], 'fluid' if fluid else 'plane_fallback') if scored else (None,'missing')

def ordered_files(folder: Path):
    paths = list(folder.glob('*.dcm'))
    if not paths: return [], 'empty'
    recs=[]; ref=None
    for p in paths:
        try:
            d=pydicom.dcmread(str(p),stop_before_pixels=True,specific_tags=HEADER_TAGS)
            iop=np.asarray(getattr(d,'ImageOrientationPatient',[]),float); ipp=np.asarray(getattr(d,'ImagePositionPatient',[]),float)
            inst=float(getattr(d,'InstanceNumber',0) or 0)
            normal=None
            if iop.shape==(6,):
                normal=np.cross(iop[:3],iop[3:]); nn=float(np.linalg.norm(normal))
                if nn>0:
                    normal=normal/nn
                    if ref is None: ref=normal.copy()
            recs.append((p,ipp,inst))
        except Exception: recs.append((p,None,0.0))
    if ref is not None:
        g=[]
        for p,ipp,inst in recs:
            if isinstance(ipp,np.ndarray) and ipp.shape==(3,) and np.isfinite(ipp).all(): g.append((float(np.dot(ipp,ref)),p))
        if len(g)==len(recs):
            g.sort(key=lambda z:z[0]); return [p for _,p in g], 'geometry'
    f=sorted([(inst,p) for p,ipp,inst in recs],key=lambda z:(z[0],str(z[1])))
    return [p for _,p in f], 'instance_fallback'

def read_resize(path: Path):
    d=pydicom.dcmread(str(path)); a=apply_modality_lut(d.pixel_array,d).astype(np.float32)
    if a.ndim!=2: raise ValueError(f'2D expected, got {a.shape}')
    if str(getattr(d,'PhotometricInterpretation',''))=='MONOCHROME1': a=a.max()-a
    if a.shape!=(IMG,IMG): a=cv2.resize(a,(IMG,IMG),interpolation=cv2.INTER_AREA if max(a.shape)>=IMG else cv2.INTER_LINEAR)
    return a

def build_study(index,uid,rows):
    sd=ROOT/'train_series'/uid
    out=np.zeros((3,DEPTH,IMG,IMG),np.uint8); mask=np.zeros(3,np.uint8)
    st={'index':index,'StudyInstanceUID':uid,'geometry_slots':0,'fallback_order_slots':0,'plane_fallback_slots':0,'missing_slots':0,'short_slots':0,'decode_failed_slots':0}
    for si,(name,plane) in enumerate(SLOTS):
        rec,sel=pick_series(rows,plane,sd)
        st[f'{name}_selection']=sel
        if rec is None: st['missing_slots']+=1; continue
        if sel=='plane_fallback': st['plane_fallback_slots']+=1
        suid=str(rec['SeriesInstanceUID']); st[f'{name}_series_uid']=suid
        files,method=ordered_files(sd/suid)
        if method=='geometry': st['geometry_slots']+=1
        else: st['fallback_order_slots']+=1
        n=len(files); st[f'{name}_n_slices']=n
        if n<DEPTH: st['short_slots']+=1; continue
        start=(n-DEPTH)//2; chosen=files[start:start+DEPTH]
        try:
            vol=np.stack([read_resize(p) for p in chosen],axis=0)
        except Exception:
            st['decode_failed_slots']+=1; continue
        lo=float(np.min(vol)); hi=float(np.max(vol)); scale=max(hi-lo,1e-6)
        out[si]=np.rint(np.clip((vol-lo)/scale,0,1)*255).astype(np.uint8)
        mask[si]=1
    return index,out,mask,st

def sha256(path,chunk=32<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def main():
    t0=time.time(); tr=pd.read_csv(ROOT/'train.csv',dtype={'StudyInstanceUID':str}); se=pd.read_csv(ROOT/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    uids=tr.StudyInstanceUID.astype(str).tolist(); groups={u:g.to_dict('records') for u,g in se.groupby('StudyInstanceUID')}; n=len(uids)
    bounds=np.linspace(0,n,NSHARD+1,dtype=int); maps=[]; shard_meta=[]
    for k in range(NSHARD):
        a,b=int(bounds[k]),int(bounds[k+1]); fn=f'orthodiffusion_cache_3x16x256_shard{k:02d}.npy'; p=OUT/fn
        mm=np.lib.format.open_memmap(p,mode='w+',dtype=np.uint8,shape=(b-a,3,DEPTH,IMG,IMG)); maps.append((a,b,mm,p)); shard_meta.append({'shard':k,'start':a,'stop':b,'file':fn})
    masks=np.lib.format.open_memmap(OUT/'orthodiffusion_cache_masks.npy',mode='w+',dtype=np.uint8,shape=(n,3))
    pd.DataFrame({'index':np.arange(n),'StudyInstanceUID':uids}).to_csv(OUT/'orthodiffusion_cache_uids.csv',index=False)
    stats=[]; fails=[]
    print('expected GiB', n*3*DEPTH*IMG*IMG/1024**3, 'workers',WORKERS,flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        fut={ex.submit(build_study,i,u,groups.get(u,[])):(i,u) for i,u in enumerate(uids)}
        done=0
        for f in as_completed(fut):
            i,u=fut[f]
            try:
                idx,arr,msk,st=f.result();
                for a,b,mm,p in maps:
                    if a<=idx<b: mm[idx-a]=arr; break
                masks[idx]=msk; stats.append(st)
            except Exception as e: fails.append({'index':i,'StudyInstanceUID':u,'error':f'{type(e).__name__}: {e}','trace':traceback.format_exc()[-1200:]})
            done+=1
            if done%100==0 or done==n:
                el=time.time()-t0; print(f'{done}/{n} {el/60:.1f}m eta={(el/max(done,1))*(n-done)/60:.1f}m failures={len(fails)}',flush=True)
    for a,b,mm,p in maps: mm.flush()
    masks.flush(); del masks
    pd.DataFrame(stats).to_csv(OUT/'orthodiffusion_cache_stats.csv',index=False); pd.DataFrame(fails).to_csv(OUT/'orthodiffusion_cache_failures.csv',index=False)
    sm=pd.DataFrame(stats)
    manifest={'contract':'orthodiffusion_native_center16_3plane_fluid_256_v1','source_commit':'99d15186bb8face1728263023d8414d9405c8f7e','studies':n,'shape_total':[n,3,DEPTH,IMG,IMG],'dtype':'uint8','slots':[x[0] for x in SLOTS],'series_selection':'longest Fluid_Sensitive series in plane; longest plane fallback only when no fluid-sensitive series exists','ordering':'IOP/IPP normal projection; InstanceNumber fallback only if geometry incomplete','depth':'contiguous center 16; series with <16 slices marked invalid','spatial':'whole FOV resized to 256x256, no physical crop','normalization':'after center16+resize, exact volume min-max mapped to uint8; model-time uint8/255*2-1 approximates official min-max [-1,1]','shards':shard_meta,'mask_file':'orthodiffusion_cache_masks.npy','uid_file':'orthodiffusion_cache_uids.csv','failure_count':len(fails),'geometry_slots':int(sm.geometry_slots.sum()) if len(sm) else 0,'fallback_order_slots':int(sm.fallback_order_slots.sum()) if len(sm) else 0,'plane_fallback_slots':int(sm.plane_fallback_slots.sum()) if len(sm) else 0,'missing_slots':int(sm.missing_slots.sum()) if len(sm) else 0,'short_slots':int(sm.short_slots.sum()) if len(sm) else 0,'decode_failed_slots':int(sm.decode_failed_slots.sum()) if len(sm) else 0,'elapsed_seconds':time.time()-t0}
    for r in shard_meta:
        p=OUT/r['file']; r['bytes']=p.stat().st_size; print('hash',r['file'],flush=True); r['sha256']=sha256(p)
    mp=OUT/'orthodiffusion_cache_masks.npy'; manifest['mask_sha256']=sha256(mp); (OUT/'orthodiffusion_cache_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__': main()
