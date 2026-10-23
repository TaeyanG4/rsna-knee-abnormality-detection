from __future__ import annotations
import os,json,time
from pathlib import Path
import numpy as np,pandas as pd,pydicom,cv2,torch,torch.nn as nn,torch.nn.functional as F,timm
from pydicom.pixel_data_handlers.util import apply_modality_lut
ROOT=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'); OUT=Path('/kaggle/working')
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']; UID='StudyInstanceUID'
IMG=288;DEPTH=9;CROP_MM=140.0;CENTERS=(1,4,7);NSLOTS=4
SLOTS=[('AX_FS','Axial',1),('SAG_FS','Sagittal',1),('COR_FS','Coronal',1),('SAG_STRUCT','Sagittal',0)]
TAGS=['ImagePositionPatient','ImageOrientationPatient','InstanceNumber','PixelSpacing','PhotometricInterpretation','RescaleSlope','RescaleIntercept']
class MIL(nn.Module):
 def __init__(self):
  super().__init__();self.backbone=timm.create_model('convnext_small',pretrained=False,num_classes=0,global_pool='avg');d=int(self.backbone.num_features);self.norm=nn.LayerNorm(d);self.slot=nn.Embedding(NSLOTS,d);self.att=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,len(TARGETS)));self.cls=nn.Parameter(torch.empty(len(TARGETS),d));nn.init.trunc_normal_(self.cls,std=.02);self.bias=nn.Parameter(torch.zeros(len(TARGETS)))
 def forward(self,x,valid,slot):
  b,k=x.shape[:2];f=self.backbone(x.flatten(0,1)).view(b,k,-1);h=self.norm(f+self.slot(slot));a=self.att(h).masked_fill(~valid[...,None],-1e4);a=torch.softmax(a,1);p=torch.einsum('bkt,bkd->btd',a,h);return (p*self.cls[None]).sum(-1)+self.bias
def find_weights():
 base=Path('/kaggle/input'); hits=[]
 dsroot=base/'datasets'
 if dsroot.is_dir(): hits.extend(dsroot.rglob('fold0_best.pt'))
 for child in base.iterdir():
  if child.name in ('competitions','datasets','models'): continue
  if child.is_dir(): hits.extend(child.rglob('fold0_best.pt'))
 hits=list(dict.fromkeys(hits))
 if len(hits)!=1:
  tops=[str(p) for p in base.iterdir()]
  raise RuntimeError(f'fold0 dataset-only hits={hits}; tops={tops}')
 d=hits[0].parent; out=[d/f'fold{i}_best.pt' for i in range(5)]; miss=[str(p) for p in out if not p.is_file()]
 if miss: raise FileNotFoundError(f'missing sibling weight files: {miss}')
 print('weight_dir',d,flush=True); return out
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
 ps=np.asarray(getattr(d,'PixelSpacing',[]),float);sp=float(ps[0]) if ps.shape==(2,) and np.all(ps>0) else CROP_MM/min(a.shape);side=min(a.shape[0],a.shape[1],max(1,int(round(CROP_MM/max(sp,1e-4)))));y=(a.shape[0]-side)//2;x=(a.shape[1]-side)//2;a=a[y:y+side,x:x+side];return cv2.resize(a,(IMG,IMG),interpolation=cv2.INTER_AREA if side>=IMG else cv2.INTER_LINEAR)
def build(uid,rows):
 sd=ROOT/'test_series'/uid;out=np.zeros((NSLOTS,DEPTH,IMG,IMG),np.uint8);mask=np.zeros((NSLOTS,DEPTH),np.uint8);st={'geometry_slots':0,'fallback_slots':0,'missing_slots':0,'short_slots':0,'decode_failures':0}
 for si,(name,plane,fluid) in enumerate(SLOTS):
  r=pick(rows,plane,fluid,sd)
  if r is None:st['missing_slots']+=1;continue
  fs,m=order(sd/str(r['SeriesInstanceUID']));st['geometry_slots' if m=='geometry' else 'fallback_slots']+=1
  if len(fs)<DEPTH:st['short_slots']+=1;continue
  start=(len(fs)-DEPTH)//2
  try:v=np.stack([read_crop(p) for p in fs[start:start+DEPTH]])
  except:st['decode_failures']+=1;continue
  sample=v[:,::4,::4].reshape(-1);lo,hi=np.percentile(sample,[2,98]);scale=max(float(hi-lo),1e-6);out[si]=np.rint(np.clip((v-lo)/scale,0,1)*255).astype(np.uint8);mask[si]=1
 inst=[];valid=[];slots=[]
 for s in range(NSLOTS):
  for c in CENTERS:inst.append(out[s,c-1:c+2]);valid.append(int(mask[s,c-1:c+2].min()>0));slots.append(s)
 return np.stack(inst),np.asarray(valid,bool),np.asarray(slots,np.int64),st
def main():
 t0=time.time();torch.set_num_threads(max(1,os.cpu_count() or 4));torch.set_num_interop_threads(1);test=pd.read_csv(ROOT/'test.csv',dtype={UID:str});series=pd.read_csv(ROOT/'test_series.csv',dtype={UID:str,'SeriesInstanceUID':str});groups={u:g.to_dict('records') for u,g in series.groupby(UID)};uids=test[UID].astype(str).tolist();wfiles=find_weights();models=[]
 for i,p in enumerate(wfiles):
  ck=torch.load(p,map_location='cpu',weights_only=False);m=MIL().eval();m.load_state_dict(ck['model'],strict=True);m=m.to(memory_format=torch.channels_last);models.append(m);print('loaded fold',i,'epoch',ck.get('epoch'),flush=True)
 mean=torch.tensor([.485,.456,.406]).view(1,1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,1,3,1,1);preds=[];stats=[]
 with torch.inference_mode():
  for ix,u in enumerate(uids):
   arr,val,slot,st=build(u,groups.get(u,[]));x=torch.from_numpy(arr).unsqueeze(0).float().div_(255);x=(x-mean)/std;v=torch.from_numpy(val).unsqueeze(0);s=torch.from_numpy(slot).unsqueeze(0);fp=[]
   for m in models:fp.append(torch.sigmoid(m(x,v,s)).numpy()[0])
   preds.append(np.mean(np.stack(fp),axis=0));st[UID]=u;stats.append(st)
   if (ix+1)%25==0 or ix+1==len(uids):print(f'{ix+1}/{len(uids)} elapsed={(time.time()-t0)/60:.1f}m',flush=True)
 sub=pd.DataFrame(np.asarray(preds),columns=TARGETS);sub.insert(0,UID,uids);sub.to_csv(OUT/'submission.csv',index=False);pd.DataFrame(stats).to_csv(OUT/'exp045_runtime_stats.csv',index=False);rec={'contract':'exp045_convnext_s_adjacent_288_5fold_cpu_v4','rows':len(sub),'elapsed_seconds':time.time()-t0,'threads':torch.get_num_threads(),'preprocess':'EXP034 exact','fold_merge':'mean sigmoid probabilities'};(OUT/'exp045_receipt.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2),flush=True)
if __name__=='__main__':main()