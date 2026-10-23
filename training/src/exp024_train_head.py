try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
ROOT=workspace_root()
FEATROOT=ROOT/'experiments'/'exp024_dinov2_features'
CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'

class TargetMIL(nn.Module):
    def __init__(self, in_dim=384, dim=256, drop=0.2):
        super().__init__()
        self.norm=nn.LayerNorm(in_dim)
        self.proj=nn.Sequential(nn.Linear(in_dim,dim),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(dim))
        self.slot=nn.Parameter(torch.randn(6,dim)*0.02)
        self.pos=nn.Parameter(torch.randn(3,dim)*0.02)
        self.query=nn.Parameter(torch.randn(12,dim)*0.02)
        self.head_w=nn.Parameter(torch.randn(12,dim)*0.02)
        self.head_b=nn.Parameter(torch.zeros(12))
        self.drop=nn.Dropout(drop)
    def forward(self,x,mask):
        # x B,6,3,384; mask B,6,3
        z=self.proj(self.norm(x))
        z=z+self.slot[None,:,None,:]+self.pos[None,None,:,:]
        tok=z.reshape(z.shape[0],18,-1); m=mask.reshape(mask.shape[0],18)
        score=torch.einsum('bnd,td->btn',tok,self.query)/math.sqrt(tok.shape[-1])
        score=score.masked_fill(m[:,None,:]<=0,-1e4)
        att=torch.softmax(score,dim=-1)
        pooled=torch.einsum('btn,bnd->btd',att,tok)
        pooled=self.drop(pooled)
        return (pooled*self.head_w[None,:,:]).sum(-1)+self.head_b[None,:]

def aucs(y,p):
    aa=[]
    for k in range(12):
        try: aa.append(float(roc_auc_score(y[:,k],p[:,k])))
        except Exception: aa.append(float('nan'))
    return aa,float(np.nanmean(aa))

def pair_loss(logit,y,w,margin=0.15):
    losses=[]
    for k in range(12):
        pos=(y[:,k]>=0.70)&(w[:,k]>=0.6); neg=(y[:,k]<=0.30)&(w[:,k]>=0.6)
        if pos.sum()==0 or neg.sum()==0: continue
        d=logit[pos,k][:,None]-logit[neg,k][None,:]
        losses.append(F.softplus(-d).mean())
    return torch.stack(losses).mean() if losses else logit.sum()*0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--lr',type=float,default=8e-4); ap.add_argument('--rank',type=float,default=0.0); ap.add_argument('--batch',type=int,default=256); args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    feat=np.load(FEATROOT/'features.npy',mmap_mode='r').astype(np.float32)
    valid=np.load(FEATROOT/'valid.npy').astype(np.float32)
    uids=pd.read_csv(CACHE/'train_cache_uids.csv'); ids=uids.StudyInstanceUID.astype(str).tolist()
    teacher=pd.read_csv(ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv').set_index('StudyInstanceUID')
    meta=pd.read_csv(ROOT/'data'/'train.csv').set_index('StudyInstanceUID')
    y=np.zeros((len(ids),12),np.float32); w=np.zeros_like(y); gold=np.zeros(len(ids),bool); gy=np.full_like(y,np.nan)
    for i,uid in enumerate(ids):
        r=teacher.loc[uid]; y[i]=r[TARGETS].to_numpy(np.float32); w[i]=r[[t+'__conf' for t in TARGETS]].to_numpy(np.float32)
        vv=meta.loc[uid,TARGETS].to_numpy(np.float32)
        if np.isfinite(vv).all(): gold[i]=True; gy[i]=vv
    tr=np.where(~gold)[0]; va=np.where(gold)[0]
    dev='cuda'; X=torch.from_numpy(feat).to(dev); M=torch.from_numpy(valid).to(dev); Y=torch.from_numpy(y).to(dev); W=torch.from_numpy(w).to(dev)
    model=TargetMIL().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)
    best=-1; history=[]
    out=ROOT/'experiments'/f'exp024_head_rank{args.rank:g}_seed{args.seed}'; out.mkdir(parents=True,exist_ok=True)
    for ep in range(args.epochs):
        model.train(); perm=np.random.permutation(tr); ls=[]
        for st in range(0,len(perm),args.batch):
            ii=torch.as_tensor(perm[st:st+args.batch],device=dev)
            opt.zero_grad(set_to_none=True)
            logit=model(X[ii],M[ii]); per=F.binary_cross_entropy_with_logits(logit,Y[ii],reduction='none'); bce=(per*(0.25+0.75*W[ii])).mean()
            rk=pair_loss(logit,Y[ii],W[ii]) if args.rank>0 else logit.sum()*0
            loss=bce+args.rank*rk; loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); ls.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): pred=torch.sigmoid(model(X[va],M[va])).cpu().numpy()
        aa,macro=aucs(gy[va],pred); row={'epoch':ep+1,'loss':float(np.mean(ls)),'gold_macro_auc':macro}; row.update({f'auc_{TARGETS[k]}':aa[k] for k in range(12)}); history.append(row)
        if (ep+1)%5==0 or ep<3: print(json.dumps(row,ensure_ascii=False),flush=True)
        if macro>best: best=macro; torch.save({'model':model.state_dict(),'epoch':ep+1,'macro':macro,'rank':args.rank,'seed':args.seed},out/'best.pt')
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    bi=int(np.nanargmax([r['gold_macro_auc'] for r in history])); summary={'best':history[bi],'last':history[-1],'rank':args.rank,'seed':args.seed}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)); print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
