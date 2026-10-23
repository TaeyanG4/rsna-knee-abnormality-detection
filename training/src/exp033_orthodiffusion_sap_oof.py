from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse, hashlib, json, math, unicodedata, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
ROOT=workspace_root()
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
PLANES=['sagittal','coronal','axial']

def report_key(x): return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',str(x)).lower().split()).encode()).hexdigest()
class SAP(nn.Module):
 def __init__(self,c=256,h=16):
  super().__init__();self.c=c;self.h=h;self.q=nn.Linear(c,c);self.k=nn.Linear(c,c);self.v=nn.Linear(c,c);self.o=nn.Linear(c,c)
 def forward(self,f):
  B,C,D,H,W=f.shape;N=D*H*W;x=f.view(B,C,N).permute(0,2,1);d=C//self.h
  q=self.q(x).view(B,N,self.h,d).transpose(1,2);k=self.k(x).view(B,N,self.h,d).transpose(1,2);v=self.v(x).view(B,N,self.h,d).transpose(1,2)
  a=torch.softmax(torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d),-1);ctx=torch.matmul(a,v).transpose(1,2).contiguous().view(B,N,C);p=self.o(ctx.mean(1));g=f.mean((2,3,4));return torch.cat([g,p],1)
class Fusion(nn.Module):
 def __init__(self):
  super().__init__();self.sap=nn.ModuleList([SAP() for _ in range(3)]);self.norm=nn.ModuleList([nn.LayerNorm(512) for _ in range(3)]);self.fc=nn.Linear(512*3+3,12)
 def forward(self,xs,m):
  z=[]
  for p in range(3):
   v=self.norm[p](self.sap[p](xs[p]));v=v*m[:,p:p+1];z.append(v)
  return self.fc(torch.cat(z+[m],1))
def auc(y,p):
 a=[]
 for k in range(12): a.append(float(roc_auc_score(y[:,k],p[:,k])))
 return a,float(np.mean(a))
def weak_metric(y,p,conf):
 a=[]
 for k in range(12):
  use=((y[:,k]>=.7)|(y[:,k]<=.3))&(conf[:,k]>=.6); yy=(y[use,k]>=.7).astype(int); a.append(float(roc_auc_score(yy,p[use,k])) if len(np.unique(yy))==2 else np.nan)
 return float(np.nanmean(a)),a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--cache-dir',required=True);ap.add_argument('--out',default=str(ROOT/'experiments'/'EXP033_ORTHODIFF_SAP_OOF'));ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--lr',type=float,default=5e-4);ap.add_argument('--seed',type=int,default=20260913);args=ap.parse_args()
 feat=workspace_path(args.features);cache=workspace_path(args.cache_dir);out=workspace_path(args.out);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(args.seed);np.random.seed(args.seed)
 X=[np.load(feat/f'{p}_features_f16.npy',mmap_mode='r') for p in PLANES];M=np.stack([np.load(feat/f'{p}_valid.npy') for p in PLANES],1).astype(np.float32);uid_path=next((cache/n for n in ('orthodiffusion_cache_uids.csv','orthodiff_uids.csv') if (cache/n).is_file()),None); assert uid_path is not None; uids=pd.read_csv(uid_path).StudyInstanceUID.astype(str).tolist()
 teacher=pd.read_csv(ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv').set_index('StudyInstanceUID').loc[uids];tr=pd.read_csv(ROOT/'data'/'train.csv').set_index('StudyInstanceUID').loc[uids]
 y=teacher[TARGETS].to_numpy(np.float32);conf=teacher[[t+'__conf' for t in TARGETS]].to_numpy(np.float32);gold=tr[TARGETS].notna().all(axis=1).to_numpy();gy=tr.loc[np.array(uids)[gold],TARGETS].to_numpy(np.float32)
 groups=tr.Report.map(report_key).to_numpy();gset=set(groups[gold]); quarantine=np.array([g in gset for g in groups])&~gold;eligible=~gold&~quarantine
 # Deterministic report-group greedy folds identical in spirit to EXP026.
 rng=np.random.default_rng(args.seed);sizes=pd.Series(groups[eligible]).value_counts();order=list(sizes.index);rng.shuffle(order);order.sort(key=lambda g:-sizes[g]);loads=np.zeros(5,int);mp={}
 for g in order: f=int(np.argmin(loads));mp[g]=f;loads[f]+=int(sizes[g])
 folds=np.array([mp[g] if ok else -1 for g,ok in zip(groups,eligible)]);pd.DataFrame({'StudyInstanceUID':uids,'fold':folds,'is_gold':gold,'quarantine':quarantine}).to_csv(out/'folds.csv',index=False)
 oof=np.full((len(uids),12),np.nan,np.float32);gpred=[];fold_receipt=[];dev='cuda'
 def batch_data(ii):
  xs=[torch.from_numpy(np.asarray(x[ii],dtype=np.float32)).to(dev,non_blocking=True) for x in X];m=torch.from_numpy(M[ii]).to(dev);return xs,m
 for fold in range(5):
  torch.manual_seed(args.seed+fold);np.random.seed(args.seed+fold);model=Fusion().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=5e-4);tri=np.where(eligible&(folds!=fold))[0];vi=np.where(eligible&(folds==fold))[0];rg=np.random.default_rng(args.seed+fold);t0=time.time()
  for ep in range(args.epochs):
   model.train();ordr=rg.permutation(tri);ls=[]
   for a in range(0,len(ordr),args.batch):
    ii=ordr[a:a+args.batch];xs,m=batch_data(ii);yy=torch.from_numpy(y[ii]).to(dev);ww=torch.from_numpy(conf[ii]).to(dev);opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda',dtype=torch.float16): logit=model(xs,m);loss=(F.binary_cross_entropy_with_logits(logit,yy,reduction='none')*(.25+.75*ww)).mean()
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ls.append(float(loss.detach()))
   print('fold',fold,'epoch',ep+1,'loss',round(float(np.mean(ls)),5),flush=True)
  def pred(ii):
   model.eval();oo=[]
   with torch.no_grad():
    for a in range(0,len(ii),args.batch):
     jj=ii[a:a+args.batch];xs,m=batch_data(jj)
     with torch.amp.autocast('cuda',dtype=torch.float16): oo.append(torch.sigmoid(model(xs,m)).float().cpu().numpy())
   return np.concatenate(oo)
  pv=pred(vi);oof[vi]=pv;pg=pred(np.where(gold)[0]);gpred.append(pg);wm,_=weak_metric(y[vi],pv,conf[vi]);fold_receipt.append({'fold':fold,'n_train':len(tri),'n_val':len(vi),'weak_highconf_auc':wm,'seconds':time.time()-t0});print(fold_receipt[-1],flush=True)
  del model,opt;torch.cuda.empty_cache()
 gp=np.mean(gpred,0);ga,gm=auc(gy,gp);wm,wa=weak_metric(y[eligible],oof[eligible],conf[eligible]);receipt={'contract':'OrthoDiffusion official LP-Opt frozen features + official-style SAP simple concat; fixed 10 epochs; no Gold selection','eligible':int(eligible.sum()),'quarantine':int(quarantine.sum()),'fold_sizes':loads.tolist(),'weak_oof_highconf_macro':wm,'weak_oof_per_target':dict(zip(TARGETS,wa)),'gold_macro_once_after_5fold':gm,'gold_per_target':dict(zip(TARGETS,ga)),'folds':fold_receipt}
 pd.DataFrame(oof[eligible],columns=TARGETS).assign(StudyInstanceUID=np.array(uids)[eligible]).to_csv(out/'weak_oof.csv',index=False);pd.DataFrame(gp,columns=TARGETS).assign(StudyInstanceUID=np.array(uids)[gold]).to_csv(out/'gold_pred.csv',index=False);(out/'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()
