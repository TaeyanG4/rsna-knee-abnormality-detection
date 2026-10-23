from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse, json, math, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
UID='StudyInstanceUID'
SLOTS=6
NS=9
INST_PER_SLOT=NS-2
NINST=SLOTS*INST_PER_SLOT
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def greedy_multilabel_folds(y:np.ndarray, nfold:int, seed:int):
    # Approximate multilabel stratification from weak-label probabilities.
    hard=(y>=0.5).astype(np.int32)
    rarity=(hard/np.maximum(hard.sum(0,keepdims=True),1)).sum(1)
    rng=np.random.default_rng(seed)
    order=np.lexsort((rng.random(len(y)),-rarity))
    fold=np.full(len(y),-1,np.int16)
    counts=np.zeros((nfold,y.shape[1]),np.float64)
    sizes=np.zeros(nfold,np.int32)
    target=hard.sum(0)/nfold
    for i in order:
        scores=[]
        for f in range(nfold):
            after=counts[f]+hard[i]
            imbalance=np.square((after-target)/np.maximum(target,1)).mean()
            size_pen=((sizes[f]+1)-len(y)/nfold)**2/(len(y)/nfold)**2
            scores.append(imbalance+0.05*size_pen)
        f=int(np.argmin(scores)); fold[i]=f; counts[f]+=hard[i]; sizes[f]+=1
    return fold


class CacheDataset(Dataset):
    def __init__(self, cache_path, mask_path, rows, labels, train=False):
        self.cache=np.load(cache_path,mmap_mode='r')
        self.masks=np.load(mask_path,mmap_mode='r')
        self.rows=np.asarray(rows,np.int64)
        self.labels=np.asarray(labels,np.float32)
        self.train=bool(train)
    def __len__(self): return len(self.rows)
    def __getitem__(self,k):
        i=int(self.rows[k])
        v=np.asarray(self.cache[i],dtype=np.uint8) # 6,9,H,W
        m=np.asarray(self.masks[i],dtype=np.uint8)
        # 7 triplets per slot -> 42 RGB instances
        inst=[]; valid=[]; slot=[]
        for s in range(SLOTS):
            for c in range(1,NS-1):
                tri=v[s,c-1:c+2].astype(np.float32)/255.0
                if self.train:
                    if random.random()<0.5: tri=tri[:,:,::-1].copy()
                    if random.random()<0.25:
                        gamma=random.uniform(0.85,1.15); tri=np.clip(tri,0,1)**gamma
                inst.append(tri)
                valid.append(int(m[s,c-1] and m[s,c] and m[s,c+1]))
                slot.append(s)
        x=torch.from_numpy(np.stack(inst))
        x=(x-MEAN)/STD
        return x, torch.tensor(valid,dtype=torch.bool), torch.tensor(slot,dtype=torch.long), torch.from_numpy(self.labels[k])


class TargetAttentionMIL(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, drop=0.2):
        super().__init__()
        self.backbone=timm.create_model(backbone,pretrained=pretrained,num_classes=0,global_pool='avg')
        d=int(self.backbone.num_features)
        self.norm=nn.LayerNorm(d)
        self.slot=nn.Embedding(SLOTS,d)
        self.att=nn.Sequential(nn.Linear(d,256),nn.Tanh(),nn.Dropout(drop),nn.Linear(256,len(TARGETS)))
        self.cls=nn.Parameter(torch.empty(len(TARGETS),d)); nn.init.trunc_normal_(self.cls,std=.02)
        self.bias=nn.Parameter(torch.zeros(len(TARGETS)))
    def forward(self,x,valid,slot):
        b,k=x.shape[:2]
        f=self.backbone(x.flatten(0,1)).view(b,k,-1)
        h=self.norm(f+self.slot(slot))
        a=self.att(h)
        a=a.masked_fill(~valid.unsqueeze(-1),-1e4)
        a=torch.softmax(a,dim=1)
        pooled=torch.einsum('bkt,bkd->btd',a,h)
        return (pooled*self.cls.unsqueeze(0)).sum(-1)+self.bias


def pairwise_auc_loss(logits,y,margin=0.1,max_pairs=256):
    losses=[]
    for j in range(y.shape[1]):
        pos=torch.where(y[:,j]>=0.8)[0]; neg=torch.where(y[:,j]<=0.2)[0]
        if len(pos)==0 or len(neg)==0: continue
        if len(pos)*len(neg)>max_pairs:
            pos=pos[torch.randperm(len(pos),device=pos.device)[:min(len(pos),16)]]
            neg=neg[torch.randperm(len(neg),device=neg.device)[:min(len(neg),16)]]
        d=logits[pos,j,None]-logits[neg,j][None,:]
        losses.append(F.softplus(margin-d).mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0


def macro_auc(y,p):
    vals=[]
    for j in range(y.shape[1]):
        if len(np.unique(y[:,j]))<2: vals.append(np.nan)
        else: vals.append(roc_auc_score(y[:,j],p[:,j]))
    return float(np.nanmean(vals)),vals


@torch.no_grad()
def predict(model,loader,device):
    model.eval(); out=[]; yy=[]
    for x,v,s,y in loader:
        x=x.to(device,non_blocking=True); v=v.to(device); s=s.to(device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            z=model(x,v,s)
        out.append(torch.sigmoid(z.float()).cpu().numpy()); yy.append(y.numpy())
    return np.concatenate(yy),np.concatenate(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache-dir',required=True)
    ap.add_argument('--root',default='.')
    ap.add_argument('--backbone',default='resnet18')
    ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--wd',type=float,default=1e-4)
    ap.add_argument('--rank-lambda',type=float,default=0.15)
    ap.add_argument('--seed',type=int,default=20260912)
    ap.add_argument('--pretrained',action='store_true')
    ap.add_argument('--out',default='experiments/compact_mil')
    args=ap.parse_args()
    seed_all(args.seed)
    root=workspace_root(args.root); cdir=workspace_path(args.cache_dir); out=workspace_path(args.out); out.mkdir(parents=True,exist_ok=True)
    uidmap=pd.read_csv(cdir/'train_cache_uids.csv',dtype={UID:str})
    weak=pd.read_csv(root/'labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv',dtype={UID:str}).set_index(UID)
    official=pd.read_csv(root/'data/train.csv',dtype={UID:str}).set_index(UID)
    ids=uidmap[UID].tolist()
    weak_y=weak.reindex(ids)[TARGETS].to_numpy(np.float32)
    gold_mask=official.reindex(ids)[TARGETS].notna().all(1).to_numpy()
    train_pool=np.where(~gold_mask & np.isfinite(weak_y).all(1))[0]
    gold_rows=np.where(gold_mask)[0]
    folds=greedy_multilabel_folds(weak_y[train_pool],args.folds,args.seed)
    tr=train_pool[folds!=args.fold]; va=train_pool[folds==args.fold]
    gold_y=official.reindex(ids).iloc[gold_rows][TARGETS].to_numpy(np.float32)
    ds_tr=CacheDataset(cdir/'train_cache_6x9x224.npy',cdir/'train_cache_masks.npy',tr,weak_y[tr],True)
    ds_va=CacheDataset(cdir/'train_cache_6x9x224.npy',cdir/'train_cache_masks.npy',va,weak_y[va],False)
    ds_g=CacheDataset(cdir/'train_cache_6x9x224.npy',cdir/'train_cache_masks.npy',gold_rows,gold_y,False)
    dl_tr=DataLoader(ds_tr,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    dl_va=DataLoader(ds_va,batch_size=max(args.batch,4),shuffle=False,num_workers=args.workers,pin_memory=True)
    dl_g=DataLoader(ds_g,batch_size=max(args.batch,4),shuffle=False,num_workers=args.workers,pin_memory=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TargetAttentionMIL(args.backbone,args.pretrained).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    history=[]
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]; t0=time.time()
        for x,v,s,y in dl_tr:
            x=x.to(dev,non_blocking=True); v=v.to(dev); s=s.to(dev); y=y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=dev.type=='cuda'):
                z=model(x,v,s)
                bce=F.binary_cross_entropy_with_logits(z,y)
                rank=pairwise_auc_loss(z,y)
                loss=bce+args.rank_lambda*rank
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu()))
        vy,vp=predict(model,dl_va,dev); gy,gp=predict(model,dl_g,dev)
        weak_auc,_=macro_auc((vy>=.5).astype(np.float32),vp)
        gold_auc,gper=macro_auc(gy,gp)
        row={'epoch':ep,'loss':float(np.mean(losses)),'weak_val_auc':weak_auc,'gold58_auc_diagnostic':gold_auc,'elapsed_s':time.time()-t0}
        history.append(row); print(json.dumps(row),flush=True)
        torch.save({'model':model.state_dict(),'args':vars(args),'epoch':ep,'gold_auc_diagnostic':gold_auc},out/f'fold{args.fold}_e{ep:02d}.pt')
        pd.DataFrame(history).to_csv(out/f'fold{args.fold}_history.csv',index=False)
        pd.DataFrame({'target':TARGETS,'auc':gper}).to_csv(out/f'fold{args.fold}_e{ep:02d}_gold_per_target.csv',index=False)
    print('DONE',out)

if __name__=='__main__': main()
