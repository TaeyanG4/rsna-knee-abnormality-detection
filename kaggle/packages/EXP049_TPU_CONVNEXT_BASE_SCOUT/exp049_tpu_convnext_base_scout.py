from __future__ import annotations
import os
os.environ.setdefault('PJRT_DEVICE','TPU')
os.environ.setdefault('XLA_USE_BF16','1')
import json,time,random
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
import timm
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
UID='StudyInstanceUID'; NSLOTS=5; NCENTERS=5; IMG=256
OUT=Path('/kaggle/working')
def find_input(name):
 b=Path('/kaggle/input'); hits=list(b.glob(f'*/{name}'))+list(b.glob(f'*/*/{name}'))
 if len(hits)!=1: raise RuntimeError(f'{name} hits={hits}')
 return hits[0]
class Shards:
 def __init__(self):
  mp=find_input('multicenter_manifest.json'); self.root=mp.parent; self.m=json.loads(mp.read_text()); n=int(self.m['studies']); self.lookup=np.empty(n,np.int16); self.parts=[]
  for si,r in enumerate(self.m['shards']):
   a,b=int(r['start']),int(r['stop']); x=np.load(self.root/r['file'],mmap_mode='r'); self.parts.append((a,b,x)); self.lookup[a:b]=si
  self.mask=np.load(self.root/self.m['mask_file'],mmap_mode='r'); self.uids=pd.read_csv(self.root/self.m['uid_file'],dtype={UID:str})[UID].astype(str).tolist()
  assert self.m['contract']=='multicenter_5slot_5x3_256_crop140_v1',self.m['contract']
 def get(self,i):
  si=int(self.lookup[i]); a,b,x=self.parts[si]; return np.asarray(x[i-a],dtype=np.uint8),np.asarray(self.mask[i],dtype=np.uint8)
class DS(Dataset):
 def __init__(self,cache,idx,y,conf): self.cache=cache; self.idx=np.asarray(idx); self.y=y; self.conf=conf
 def __len__(self): return len(self.idx)
 def __getitem__(self,k):
  i=int(self.idx[k]); v,m=self.cache.get(i); inst=[]; valid=[]; slots=[]
  for s in range(NSLOTS):
   for c in range(NCENTERS):
    a=c*3; inst.append(v[s,a:a+3]); valid.append(int(m[s,a:a+3].min()>0)); slots.append(s)
  return torch.from_numpy(np.stack(inst)),torch.tensor(valid,dtype=torch.bool),torch.tensor(slots,dtype=torch.long),torch.from_numpy(self.y[i]),torch.from_numpy(self.conf[i])
class MIL(nn.Module):
 def __init__(self,pretrained_state):
  super().__init__(); self.backbone=timm.create_model('convnext_base',pretrained=False,num_classes=0,global_pool='avg'); st=torch.load(pretrained_state,map_location='cpu',weights_only=True); self.backbone.load_state_dict(st,strict=True); d=int(self.backbone.num_features); self.norm=nn.LayerNorm(d); self.slot=nn.Embedding(NSLOTS,d); self.att=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,len(TARGETS))); self.cls=nn.Parameter(torch.empty(len(TARGETS),d)); nn.init.trunc_normal_(self.cls,std=.02); self.bias=nn.Parameter(torch.zeros(len(TARGETS)))
 def forward(self,x,valid,slot):
  b,k=x.shape[:2]; f=self.backbone(x.flatten(0,1)).view(b,k,-1); h=self.norm(f+self.slot(slot)); a=self.att(h).masked_fill(~valid[...,None],-1e4); a=torch.softmax(a,1); p=torch.einsum('bkt,bkd->btd',a,h); return (p*self.cls[None]).sum(-1)+self.bias
def weak_auc(y,p,conf):
 vals=[]; per={}
 for j,t in enumerate(TARGETS):
  use=((y[:,j]>=.7)|(y[:,j]<=.3))&(conf[:,j]>=.6); yy=(y[use,j]>=.7).astype(np.uint8); a=float(roc_auc_score(yy,p[use,j])) if np.unique(yy).size==2 else float('nan'); vals.append(a); per[t]={'auc':a,'n':int(use.sum()),'pos':int(yy.sum())}
 return float(np.nanmean(vals)),per
def rank_world():
 try: return int(xm.get_ordinal()),int(xm.xrt_world_size())
 except Exception:
  import torch_xla.runtime as xr; return int(xr.global_ordinal()),int(xr.world_size())
def worker(index):
 rank,world=rank_world(); seed=20260914+rank; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); device=xm.xla_device(); cache=Shards(); ids=cache.uids
 fp=find_input('folds.csv'); folds=pd.read_csv(fp,dtype={UID:str}).set_index(UID).loc[ids]
 lp=find_input('report_labels_v4hybrid.csv'); weak=pd.read_csv(lp,dtype={UID:str}).set_index(UID).loc[ids]
 y=weak[TARGETS].to_numpy(np.float32); conf=weak[[t+'__conf' for t in TARGETS]].to_numpy(np.float32); f=folds.fold.to_numpy(int); gold=folds.is_gold.to_numpy(bool); quar=folds.quarantined_gold_report_duplicate.to_numpy(bool); eligible=(f>=0)&~gold&~quar; tr=np.where(eligible&(f!=0))[0]; va=np.where(eligible&(f==0))[0]
 ds=DS(cache,tr,y,conf); sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,drop_last=True,seed=20260914); dl=DataLoader(ds,batch_size=1,sampler=sampler,num_workers=0,drop_last=True)
 pretrained=find_input('convnext_base_timm_pretrained_state.pt'); model=MIL(pretrained).to(device); bb=list(model.backbone.parameters()); bbid={id(p) for p in bb}; head=[p for p in model.parameters() if id(p) not in bbid]; opt=torch.optim.AdamW([{'params':bb,'lr':5e-6},{'params':head,'lr':1e-4}],weight_decay=.05); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=8)
 mean=torch.tensor([.485,.456,.406],device=device).view(1,1,3,1,1); std=torch.tensor([.229,.224,.225],device=device).view(1,1,3,1,1); hist=[]; best=-1.; best_ep=0
 if rank==0: print(json.dumps({'rank':rank,'world':world,'device':str(device),'train':len(tr),'val':len(va),'cache':cache.m['contract']}),flush=True)
 for ep in range(1,9):
  sampler.set_epoch(ep); model.train(); losses=[]; t0=time.time(); xdl=pl.MpDeviceLoader(dl,device)
  for x,v,s,yt,ct in xdl:
   x=x.float().div_(255); gain=.95+.10*torch.rand((x.shape[0],1,1,1,1),device=device); shift=-.03+.06*torch.rand((x.shape[0],1,1,1,1),device=device); x=torch.clamp(x*gain+shift,0,1); x=(x-mean)/std
   z=model(x,v,s); w=(.1+.9*ct.square())*(ct>=.5); loss=(F.binary_cross_entropy_with_logits(z,yt,reduction='none')*w).sum()/w.sum().clamp_min(1); loss.backward(); xm.optimizer_step(opt,barrier=False); opt.zero_grad(set_to_none=True); losses.append(float(loss.detach().cpu()))
  sch.step(); xm.rendezvous(f'train_ep_{ep}')
  if rank==0:
   model.eval(); pp=[]; yy=[]; cc=[]; vdl=DataLoader(DS(cache,va,y,conf),batch_size=1,shuffle=False,num_workers=0)
   with torch.no_grad():
    for x,v,s,yt,ct in pl.MpDeviceLoader(vdl,device):
     x=(x.float().div_(255)-mean)/std; p=torch.sigmoid(model(x,v,s)).float().cpu().numpy(); pp.append(p); yy.append(yt.cpu().numpy()); cc.append(ct.cpu().numpy())
   vp=np.concatenate(pp); vy=np.concatenate(yy); vc=np.concatenate(cc); vm,vper=weak_auc(vy,vp,vc); row={'epoch':ep,'weak_auc':vm,'train_loss_rank0':float(np.mean(losses)),'seconds':time.time()-t0}; hist.append(row); print(json.dumps(row),flush=True); pd.DataFrame(hist).to_csv(OUT/'exp049_history.csv',index=False)
   if vm>best:
    best=vm; best_ep=ep; xm.save({'model':model.state_dict(),'epoch':ep,'weak_auc':vm,'contract':'exp049_multicenter_convnext_base_tpu_fold0_v1'},OUT/'exp049_best.pt',master_only=True); pd.DataFrame(vp,columns=TARGETS).assign(StudyInstanceUID=np.array(ids)[va]).to_csv(OUT/'exp049_best_val_pred.csv',index=False); (OUT/'exp049_best_per_target.json').write_text(json.dumps(vper,indent=2))
  xm.rendezvous(f'val_ep_{ep}')
 if rank==0:
  rec={'contract':'exp049_multicenter_convnext_base_tpu_fold0_v1','world_size':world,'epochs':8,'best_epoch':best_ep,'best_weak_auc':best,'train':len(tr),'val':len(va),'input_contract':cache.m['contract'],'gold_used':False}; (OUT/'exp049_receipt.json').write_text(json.dumps(rec,indent=2)); print(json.dumps(rec,indent=2),flush=True)
if __name__=='__main__':
 xmp.spawn(worker,nprocs=None,start_method='fork')