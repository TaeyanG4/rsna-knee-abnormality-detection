from __future__ import annotations
import os,json,time,hashlib,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,pydicom,cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut
ROOT=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'); OUT=Path('/kaggle/working')
IMG=288;DEPTH=9;CROP_MM=140.0;WORKERS=max(2,min(8,(os.cpu_count() or 4)-1));NSHARD=8
SLOTS=[('AX_FS','Axial',1),('SAG_FS','Sagittal',1),('COR_FS','Coronal',1),('SAG_STRUCT','Sagittal',0)]
TAGS=['ImagePositionPatient','ImageOrientationPatient','InstanceNumber','PixelSpacing','PhotometricInterpretation','RescaleSlope','RescaleIntercept']
def count(folder):
 try:return sum(1 for _ in folder.glob('*.dcm'))
 except:return 0
def pick(rows,plane,fluid,sd):
 p=[r for r in rows if str(r.get('Anatomical_Plane'))==plane and int(r.get('Fluid_Sensitive',r.get('Fat_Suppression',0)) or 0)==fluid]
 if not p:return None
 z=[(count(sd/str(r['SeriesInstanceUID'])),str(r['SeriesInstanceUID']),r) for r in p];z.sort(key=lambda x:(-x[0],x[1]));return z[0][2] if z else None
def order(folder):
 fs=list(folder.glob('*.dcm'));recs=[];ref=None
 for p in fs:
  try:
   d=pydicom.dcmread(str(p),stop_before_pixels=True,specific_tags=TAGS);iop=np.asarray(getattr(d,'ImageOrientationPatient',[]),float);ipp=np.asarray(getattr(d,'ImagePositionPatient',[]),float);inst=float(getattr(d,'InstanceNumber',0) or 0)
   if iop.shape==(6,):
    n=np.cross(iop[:3],iop[3:]);nn=np.linalg.norm(n)
    if nn>0 and ref is None:ref=n/nn
   recs.append((p,ipp,inst))
  except:recs.append((p,None,0.0))
 if ref is not None:
  g=[(float(np.dot(ipp,ref)),p) for p,ipp,inst in recs if isinstance(ipp,np.ndarray) and ipp.shape==(3,) and np.isfinite(ipp).all()]
  if len(g)==len(recs):g.sort(key=lambda x:x[0]);return [p for _,p in g],'geometry'
 f=sorted([(inst,p) for p,ipp,inst in recs],key=lambda x:(x[0],str(x[1])));return [p for _,p in f],'instance_fallback'
def read_crop(p):
 d=pydicom.dcmread(str(p));a=apply_modality_lut(d.pixel_array,d).astype(np.float32)
 if str(getattr(d,'PhotometricInterpretation',''))=='MONOCHROME1':a=a.max()-a
 ps=np.asarray(getattr(d,'PixelSpacing',[]),float);sp=float(ps[0]) if ps.shape==(2,) and np.all(ps>0) else CROP_MM/min(a.shape)
 side=min(a.shape[0],a.shape[1],max(1,int(round(CROP_MM/max(sp,1e-4)))));y=(a.shape[0]-side)//2;x=(a.shape[1]-side)//2;a=a[y:y+side,x:x+side]
 return cv2.resize(a,(IMG,IMG),interpolation=cv2.INTER_AREA if side>=IMG else cv2.INTER_LINEAR)
def build(i,u,rows):
 sd=ROOT/'train_series'/u;out=np.zeros((4,DEPTH,IMG,IMG),np.uint8);mask=np.zeros((4,DEPTH),np.uint8);st={'index':i,'StudyInstanceUID':u,'geometry_slots':0,'fallback_slots':0,'missing_slots':0,'short_slots':0,'decode_failures':0}
 for si,(name,plane,fluid) in enumerate(SLOTS):
  r=pick(rows,plane,fluid,sd)
  if r is None:st['missing_slots']+=1;continue
  su=str(r['SeriesInstanceUID']);fs,m=order(sd/su);st['geometry_slots' if m=='geometry' else 'fallback_slots']+=1;n=len(fs)
  if n<DEPTH:st['short_slots']+=1;continue
  start=(n-DEPTH)//2;chosen=fs[start:start+DEPTH];raw=[]
  try:raw=[read_crop(p) for p in chosen]
  except Exception:st['decode_failures']+=1;continue
  v=np.stack(raw);sample=v[:,::4,::4].reshape(-1);lo,hi=np.percentile(sample,[2,98]);scale=max(float(hi-lo),1e-6);out[si]=np.rint(np.clip((v-lo)/scale,0,1)*255).astype(np.uint8);mask[si]=1
 return i,out,mask,st
def sha(p):
 h=hashlib.sha256();f=open(p,'rb')
 for b in iter(lambda:f.read(32<<20),b''):h.update(b)
 f.close();return h.hexdigest()
def main():
 t0=time.time();tr=pd.read_csv(ROOT/'train.csv',dtype={'StudyInstanceUID':str});se=pd.read_csv(ROOT/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str});uids=tr.StudyInstanceUID.astype(str).tolist();groups={u:g.to_dict('records') for u,g in se.groupby('StudyInstanceUID')};n=len(uids);bounds=np.linspace(0,n,NSHARD+1,dtype=int);maps=[];meta=[]
 for k in range(NSHARD):
  a,b=map(int,(bounds[k],bounds[k+1]));fn=f'adjacent_cache_4x9x288_shard{k:02d}.npy';p=OUT/fn;mm=np.lib.format.open_memmap(p,mode='w+',dtype=np.uint8,shape=(b-a,4,DEPTH,IMG,IMG));maps.append((a,b,mm,p));meta.append({'shard':k,'start':a,'stop':b,'file':fn})
 masks=np.lib.format.open_memmap(OUT/'adjacent_cache_masks.npy',mode='w+',dtype=np.uint8,shape=(n,4,DEPTH));pd.DataFrame({'index':range(n),'StudyInstanceUID':uids}).to_csv(OUT/'adjacent_cache_uids.csv',index=False);stats=[];fails=[];print('expected GiB',n*4*DEPTH*IMG*IMG/1024**3,'workers',WORKERS,flush=True)
 with ProcessPoolExecutor(max_workers=WORKERS) as ex:
  fut={ex.submit(build,i,u,groups.get(u,[])):(i,u) for i,u in enumerate(uids)};done=0
  for f in as_completed(fut):
   i,u=fut[f]
   try:
    idx,arr,msk,st=f.result();
    for a,b,mm,p in maps:
     if a<=idx<b:mm[idx-a]=arr;break
    masks[idx]=msk;stats.append(st)
   except Exception as e:fails.append({'index':i,'StudyInstanceUID':u,'error':f'{type(e).__name__}: {e}','trace':traceback.format_exc()[-1000:]})
   done+=1
   if done%100==0 or done==n:
    el=time.time()-t0;print(f'{done}/{n} {el/60:.1f}m eta={(el/max(done,1))*(n-done)/60:.1f}m failures={len(fails)}',flush=True)
 for a,b,mm,p in maps:mm.flush()
 masks.flush();del masks;pd.DataFrame(stats).to_csv(OUT/'adjacent_cache_stats.csv',index=False);pd.DataFrame(fails).to_csv(OUT/'adjacent_cache_failures.csv',index=False);sm=pd.DataFrame(stats)
 man={'contract':'adjacent_center9_4slot_288_crop140_v1','studies':n,'shape_total':[n,4,DEPTH,IMG,IMG],'dtype':'uint8','slots':[{'name':n,'plane':p,'fluid':f} for n,p,f in SLOTS],'sampling':'contiguous center 9 after geometry ordering','crop_mm':CROP_MM,'image_size':IMG,'normalization':'per series center9 pooled subsample percentile2-98 to uint8','ordering':'IOP/IPP normal projection; InstanceNumber fallback only if incomplete','series_selection':'longest exact plane+Fluid_Sensitive slot, no cross-fluid fallback','shards':meta,'mask_file':'adjacent_cache_masks.npy','uid_file':'adjacent_cache_uids.csv','failure_count':len(fails),'geometry_slots':int(sm.geometry_slots.sum()) if len(sm) else 0,'fallback_slots':int(sm.fallback_slots.sum()) if len(sm) else 0,'missing_slots':int(sm.missing_slots.sum()) if len(sm) else 0,'short_slots':int(sm.short_slots.sum()) if len(sm) else 0,'decode_failures':int(sm.decode_failures.sum()) if len(sm) else 0,'elapsed_seconds':time.time()-t0}
 for r in meta:p=OUT/r['file'];r['bytes']=p.stat().st_size;print('hash',r['file'],flush=True);r['sha256']=sha(p)
 mp=OUT/'adjacent_cache_masks.npy';man['mask_sha256']=sha(mp);(OUT/'adjacent_cache_manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps(man,indent=2),flush=True)
if __name__=='__main__':main()
