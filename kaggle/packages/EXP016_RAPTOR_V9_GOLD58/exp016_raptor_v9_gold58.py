import os, glob, time, gc, json, hashlib
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm, pydicom, cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
IMG=336; CROP_MM=140.0; K_EVAL=62
SLOTS=[('Sagittal',1,18),('Sagittal',0,14),('Coronal',1,12),('Coronal',0,8),('Axial',-1,12)]
MAXS=sum(x[2] for x in SLOTS)
MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
ARCH='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
WEIGHT='raptor_ft_coatnet_v9_full.pt'
ROOT='/kaggle/input/competitions/rsna-knee-abnormality-detection'

class RaptorClassifier(nn.Module):
    def __init__(self,bb,n=12,drop=.2):
        super().__init__(); d=bb.num_features; self.backbone=bb; self.norm=nn.LayerNorm(d)
        self.att=nn.Sequential(nn.Linear(d,256),nn.Tanh(),nn.Dropout(drop),nn.Linear(256,n))
        self.clsW=nn.Parameter(torch.zeros(n,d)); self.clsb=nn.Parameter(torch.zeros(n))
    def encode(self,x):
        B,K=x.shape[:2]; return self.backbone(x.flatten(0,1)).view(B,K,-1)
    def head(self,f):
        h=self.norm(f); a=torch.softmax(self.att(h),dim=1); p=torch.einsum('bkn,bkf->bnf',a,h)
        return (p*self.clsW).sum(-1)+self.clsb
    def forward(self,x): return self.head(self.encode(x))

def find_weight():
    c=['/kaggle/input/datasets/dreaddevelopment/raptor-knee-finespacing/'+WEIGHT,
       '/kaggle/input/raptor-knee-finespacing/'+WEIGHT]
    h=[p for p in c if os.path.isfile(p)]
    if len(h)!=1: raise RuntimeError(f'weight hits={h}')
    return h[0]

def load_model(dev):
    p=find_weight(); ck=torch.load(p,map_location='cpu',weights_only=False); arch=ck.get('arch',ARCH); res=int(ck.get('res',384))
    bb=timm.create_model(arch,pretrained=False,num_classes=0,in_chans=3,global_pool='avg')
    m=RaptorClassifier(bb); m.load_state_dict(ck['model'],strict=True); m.eval().to(dev); del ck; gc.collect(); return m,res,p

def order_and_meta(sdir):
    rec=[]; pss=[]
    for f in glob.glob(sdir+'/*.dcm'):
        try:
            h=pydicom.dcmread(f,stop_before_pixels=True); iop=getattr(h,'ImageOrientationPatient',None); ipp=getattr(h,'ImagePositionPatient',None)
            if iop is not None and ipp is not None and len(iop)==6:
                r=np.array(iop[:3],float); c=np.array(iop[3:],float); pos=float(np.dot(np.array(ipp,float),np.cross(r,c)))
            else: pos=float(getattr(h,'InstanceNumber',0) or 0)
            ps=getattr(h,'PixelSpacing',None); ps=float(ps[0]) if ps is not None else .5; pss.append(ps); rec.append((pos,f,ps))
        except Exception: rec.append((0.,f,.5))
    rec.sort(key=lambda x:x[0]); med=float(np.median(pss)) if pss else .5; return [(f,p) for _,f,p in rec],med

def read_px(f):
    d=pydicom.dcmread(f); a=apply_modality_lut(d.pixel_array,d).astype(np.float32)
    if str(getattr(d,'PhotometricInterpretation',''))=='MONOCHROME1': a=a.max()-a
    return a

def crop(a,ps):
    h,w=a.shape; cpx=min(int(round(CROP_MM/max(ps,1e-3))),min(h,w)); y=(h-cpx)//2; x=(w-cpx)//2
    return cv2.resize(a[y:y+cpx,x:x+cpx],(IMG,IMG),interpolation=cv2.INTER_AREA)

def pick(rows,plane,fluid,used):
    c=[r for r in rows if r['Anatomical_Plane']==plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0,1):
        p=[r for r in c if int(r.get('Fluid_Sensitive',0) or 0)==fluid]
        if p:return p[0]
    return c[0] if c else None

def build(uid,series,root):
    vol=np.zeros((MAXS,IMG,IMG),np.uint8); idx=0; used=set(); rows=series.get(uid,[])
    for plane,fluid,k in SLOTS:
        r=pick(rows,plane,fluid,used)
        if r is None: idx+=k; continue
        used.add(r['SeriesInstanceUID']); files,med=order_and_meta(f"{root}/{uid}/{r['SeriesInstanceUID']}")
        if not files: idx+=k; continue
        n=len(files); lo,hi=int(n*.06),int(n*.94)-1; hi=max(hi,lo); picks=np.linspace(lo,hi,k).round().astype(int) if n>1 else [0]*k
        arr=[]; pss=[]
        for q in picks:
            fp,ps=files[min(q,n-1)]
            try: arr.append(read_px(fp)); pss.append(ps)
            except Exception: arr.append(None); pss.append(med)
        valid=[a for a in arr if a is not None]
        loq,hiq=np.percentile(np.concatenate([a.ravel() for a in valid]),[2.,98.]) if valid else (0.,1.)
        for a,ps in zip(arr,pss):
            if a is None: idx+=1; continue
            aw=np.clip((a-loq)/(hiq-loq+1e-6),0,1); vol[idx]=(crop(aw,ps if ps>0 else med)*255).astype(np.uint8); idx+=1
    mask=(vol.reshape(MAXS,-1).sum(1)>0).astype(np.uint8); return vol,mask

def windows(vol,mask,res):
    valid=np.where(mask>0)[0]; lo,hi=int(valid.min()),int(valid.max()); cs=[c for c in range(lo+1,hi) if c-1>=lo and c+1<=hi]
    ix=np.linspace(0,len(cs)-1,K_EVAL).round().astype(int); cs=[cs[i] for i in ix]
    out=np.empty((len(cs),3,res,res),np.float32)
    for j,c in enumerate(cs):
        t=torch.from_numpy(np.stack([vol[c-1],vol[c],vol[c+1]],0).astype(np.float32)/255.)
        if t.shape[-1]!=res:t=F.interpolate(t[None],size=(res,res),mode='bilinear',align_corners=False)[0]
        out[j]=t.numpy()
    x=torch.from_numpy(out); return (x-MEAN)/STD

@torch.no_grad()
def infer(m,x,dev):
    x=x.unsqueeze(0).to(dev)
    try:
        with torch.autocast('cuda',dtype=torch.float16): return torch.sigmoid(m(x).float())[0].cpu().numpy()
    except RuntimeError:
        torch.cuda.empty_cache(); return torch.sigmoid(m(x).float())[0].cpu().numpy()

def aucs(y,p): return np.array([roc_auc_score(y[:,j],p[:,j]) for j in range(len(LAB))])

def main():
    t0=time.time();
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    dev='cuda:0'; gpu=torch.cuda.get_device_name(0); print('GPU',gpu,'count',torch.cuda.device_count(),flush=True)
    if 'P100' in gpu: raise RuntimeError('P100 unsupported by current torch; require T4 or newer')
    tr=pd.read_csv(ROOT+'/train.csv',dtype={'StudyInstanceUID':str}); g=tr.loc[tr[LAB].notna().all(1),['StudyInstanceUID']+LAB].copy(); assert len(g)==58
    ids=g.StudyInstanceUID.tolist(); y=g[LAB].to_numpy(float)
    ts=pd.read_csv(ROOT+'/train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str}); ts=ts[ts.StudyInstanceUID.isin(ids)]
    ser={k:v.to_dict('records') for k,v in ts.groupby('StudyInstanceUID')}; sroot=ROOT+'/train_series'
    m,res,wp=load_model(dev); pred=np.full((58,12),np.nan,np.float32); fail=[]
    for i,u in enumerate(ids):
        try:
            v,mask=build(u,ser,sroot); x=windows(v,mask,res); pred[i]=infer(m,x,dev); del v,mask,x
        except Exception as e: fail.append({'uid':u,'error':f'{type(e).__name__}: {e}'})
        if (i+1)%10==0 or i==57: print('v9',i+1,'/58 elapsed',round(time.time()-t0,1),flush=True)
    if fail or not np.isfinite(pred).all(): raise RuntimeError(f'fail={fail[:3]} nonfinite={int((~np.isfinite(pred)).sum())}')
    a=aucs(y,pred); out=pd.DataFrame(pred,columns=LAB); out.insert(0,'StudyInstanceUID',ids); out.to_csv('/kaggle/working/exp016_v9_gold58.csv',index=False)
    # Compare with released CoAt Gold58 predictions.
    cands=['/kaggle/input/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3/gold58_top3_predictions.npz','/kaggle/input/rsna-knee-coat-resgated-ep10-top3/gold58_top3_predictions.npz']
    ch=[p for p in cands if os.path.isfile(p)]; comp={}
    if len(ch)==1:
        z=np.load(ch[0],allow_pickle=False); cu=[str(x) for x in z['study_uids'].tolist()]; mp={u:i for i,u in enumerate(cu)}; cp=z['probability_mean'][[mp[u] for u in ids]].astype(float)
        ar=pd.DataFrame(pred,columns=LAB).rank(method='average',pct=True).to_numpy(); cr=pd.DataFrame(cp,columns=LAB).rank(method='average',pct=True).to_numpy(); hy=.6*ar+.4*cr
        ha=aucs(y,hy); comp={'coat_macro':float(aucs(y,cp).mean()),'v9_coat_60_40_macro':float(ha.mean()),'v9_coat_delta':float(ha.mean()-a.mean())}
    per=pd.DataFrame({'target':LAB,'v9_auc':a}); per.to_csv('/kaggle/working/exp016_per_target.csv',index=False)
    resu={'experiment':'EXP016_RAPTOR_V9_GOLD58','gpu':gpu,'weight':WEIGHT,'weight_path':wp,'config':{'img':IMG,'crop_mm':CROP_MM,'slices':MAXS,'span':[.06,.94],'k_eval':K_EVAL},'macro_auc':float(a.mean()),'per_target':dict(zip(LAB,map(float,a))),'elapsed_seconds':float(time.time()-t0),**comp,'warning':'Gold58 diagnostic only; public v9 release and some comparison checkpoints may be gold-selected.'}
    open('/kaggle/working/exp016_results.json','w').write(json.dumps(resu,indent=2)); print(json.dumps(resu,indent=2),flush=True)
main()