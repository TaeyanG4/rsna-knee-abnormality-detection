try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

class ShardCache:
    def __init__(self, root):
        self.root=Path(root)
        mani=json.loads((self.root/'train_cache_shards_manifest.json').read_text())
        self.shards=[]
        for s in mani['shards']:
            arr=np.load(self.root/s['file'], mmap_mode='r')
            self.shards.append((s['start'],s['stop'],arr))
    def get(self, idx):
        for a,b,x in self.shards:
            if a <= idx < b:
                return x[idx-a]
        raise IndexError(idx)

class KneeDataset(Dataset):
    def __init__(self, cache, indices, y, w, masks, train=True):
        self.cache=cache; self.indices=np.asarray(indices); self.y=y; self.w=w; self.masks=masks; self.train=train
    def __len__(self): return len(self.indices)
    def __getitem__(self, j):
        i=int(self.indices[j])
        x=np.array(self.cache.get(i), copy=True) # 6,9,224,224 uint8
        m=self.masks[i].astype(np.float32)
        if self.train:
            # cheap intensity jitter shared per study
            scale=np.random.uniform(0.9,1.1); shift=np.random.uniform(-8,8)
            x=np.clip(x.astype(np.float32)*scale+shift,0,255).astype(np.uint8)
            if np.random.rand()<0.5:
                x=x[:,:,:,::-1].copy()
        x=torch.from_numpy(x).float().div_(255.0)
        return x, torch.from_numpy(m), torch.from_numpy(self.y[i]), torch.from_numpy(self.w[i]), i

class ConvEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        def block(a,b,s=1):
            return nn.Sequential(nn.Conv2d(a,b,3,s,1,bias=False),nn.BatchNorm2d(b),nn.SiLU(inplace=True),
                                 nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.SiLU(inplace=True))
        self.net=nn.Sequential(
            nn.Conv2d(1,24,5,2,2,bias=False),nn.BatchNorm2d(24),nn.SiLU(inplace=True),
            block(24,40,2), block(40,64,2), block(64,96,2), block(96,dim,2), nn.AdaptiveAvgPool2d(1))
    def forward(self,x): return self.net(x).flatten(1)

class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__(); self.enc=ConvEncoder(dim)
        self.slot_emb=nn.Parameter(torch.randn(6,dim)*0.02)
        self.slice_emb=nn.Parameter(torch.randn(9,dim)*0.02)
        self.attn=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,1))
        self.head=nn.Sequential(nn.LayerNorm(6*dim*2),nn.Linear(6*dim*2,512),nn.SiLU(),nn.Dropout(0.25),nn.Linear(512,12))
    def forward(self,x,mask):
        B=x.shape[0]
        # downsample for speed before shared encoder
        z=x.reshape(B*54,1,224,224)
        z=F.interpolate(z,size=(128,128),mode='bilinear',align_corners=False)
        f=self.enc(z).reshape(B,6,9,-1)
        f=f+self.slot_emb[None,:,None,:]+self.slice_emb[None,None,:,:]
        score=self.attn(f).squeeze(-1)
        score=score.masked_fill(mask<=0,-1e4)
        a=torch.softmax(score,dim=2)
        mean=(f*a[...,None]).sum(2)
        masked=f.masked_fill(mask[...,None]<=0,-1e4)
        mx=masked.max(2).values
        mx=torch.where(torch.isfinite(mx),mx,torch.zeros_like(mx))
        return self.head(torch.cat([mean,mx],dim=-1).flatten(1))

def aucs(y,p):
    out=[]
    for k in range(12):
        try: out.append(float(roc_auc_score(y[:,k],p[:,k])))
        except: out.append(float('nan'))
    return out,float(np.nanmean(out))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=4); ap.add_argument('--batch',type=int,default=3); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    root=workspace_root()
    cache_root=root/'kaggle_notebooks'/'own_exp022_shard_cache'
    uids=pd.read_csv(cache_root/'train_cache_uids.csv')
    teacher=pd.read_csv(root/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv').set_index('StudyInstanceUID')
    trainmeta=pd.read_csv(root/'data'/'train.csv').set_index('StudyInstanceUID')
    ids=uids.StudyInstanceUID.astype(str).tolist()
    y=np.zeros((len(ids),12),np.float32); w=np.zeros_like(y)
    gold=np.zeros(len(ids),bool); goldy=np.full_like(y,np.nan)
    for i,uid in enumerate(ids):
        r=teacher.loc[uid]; y[i]=r[TARGETS].to_numpy(np.float32); w[i]=r[[t+'__conf' for t in TARGETS]].to_numpy(np.float32)
        tr=trainmeta.loc[uid]
        vals=tr[TARGETS].to_numpy(dtype=np.float32)
        if np.isfinite(vals).all(): gold[i]=True; goldy[i]=vals
    train_idx=np.where(~gold)[0]; val_idx=np.where(gold)[0]
    masks=np.load(cache_root/'train_cache_masks.npy')
    cache=ShardCache(cache_root)
    ds=KneeDataset(cache,train_idx,y,w,masks,True); dv=KneeDataset(cache,val_idx,y,w,masks,False)
    dl=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    vl=DataLoader(dv,batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    dev='cuda'; model=Model().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    outdir=root/'experiments'/'exp023_image_baseline'; outdir.mkdir(parents=True,exist_ok=True)
    history=[]; best=-1
    for ep in range(args.epochs):
        model.train(); losses=[]; t0=time.time()
        for step,(x,m,yy,ww,ii) in enumerate(dl):
            x=x.to(dev,non_blocking=True); m=m.to(dev,non_blocking=True); yy=yy.to(dev,non_blocking=True); ww=ww.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',dtype=torch.float16):
                logit=model(x,m)
                per=F.binary_cross_entropy_with_logits(logit,yy,reduction='none')
                loss=(per*(0.25+0.75*ww)).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update(); losses.append(float(loss))
            if step%100==0: print(f'ep{ep+1} step{step}/{len(dl)} loss={np.mean(losses[-50:]):.4f}',flush=True)
        model.eval(); pp=[]; jj=[]
        with torch.no_grad():
            for x,m,yy,ww,ii in vl:
                with torch.amp.autocast('cuda',dtype=torch.float16): pred=torch.sigmoid(model(x.to(dev),m.to(dev)))
                pp.append(pred.float().cpu().numpy()); jj.extend(ii.numpy().tolist())
        p=np.concatenate(pp); gy=goldy[np.asarray(jj)]
        per_auc,macro=aucs(gy,p)
        row={'epoch':ep+1,'train_loss':float(np.mean(losses)),'gold_macro_auc':macro,'seconds':time.time()-t0}
        row.update({f'auc_{TARGETS[k]}':per_auc[k] for k in range(12)})
        history.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
        pd.DataFrame(history).to_csv(outdir/'history.csv',index=False)
        pred_df=pd.DataFrame(p,columns=TARGETS); pred_df.insert(0,'StudyInstanceUID',[ids[i] for i in jj]); pred_df.to_csv(outdir/f'gold_pred_ep{ep+1}.csv',index=False)
        if macro>best:
            best=macro; torch.save({'model':model.state_dict(),'epoch':ep+1,'macro':macro,'targets':TARGETS},outdir/'best.pt')
    (outdir/'summary.json').write_text(json.dumps({'best_gold_macro_auc':best,'history':history},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
