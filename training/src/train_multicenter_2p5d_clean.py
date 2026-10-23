from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse, json, random, time
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

def parse_target_mask(raw:str):
    if not raw.strip(): return list(range(len(TARGETS)))
    names=[x.strip() for x in raw.split(',') if x.strip()]
    if all(x.isdigit() for x in names): return [int(x) for x in names if 0 <= int(x) < len(TARGETS)]
    unknown=[x for x in names if x not in TARGETS]
    if unknown: raise ValueError(f'Unknown target(s): {unknown}; valid targets={TARGETS}')
    return [TARGETS.index(x) for x in names]

class Shards:
    def __init__(self,root:Path,manifest_name:str):
        self.root=root
        self.m=json.loads((root/manifest_name).read_text())
        shape=list(map(int,self.m['shape_total']))
        if len(shape)!=5 or shape[2] % 3 != 0: raise ValueError(f'expected [N,S,D,H,W] with D divisible by 3, got {shape}')
        self.slots=shape[1]; self.depth=shape[2]; self.triplet_centers=tuple(range(1,self.depth,3)); self.cache_img=shape[-1]
        self.parts=[]
        self.lookup=np.empty(int(self.m['studies']),np.int16)
        for si,r in enumerate(self.m['shards']):
            a,b=int(r['start']),int(r['stop']); x=np.load(root/r['file'],mmap_mode='r')
            self.parts.append((a,b,x)); self.lookup[a:b]=si
        self.mask=np.load(root/self.m['mask_file'],mmap_mode='r')
        self.uids=pd.read_csv(root/self.m['uid_file'],dtype={UID:str})[UID].astype(str).tolist()
    def get(self,i:int):
        si=int(self.lookup[i]);a,b,x=self.parts[si];return np.asarray(x[i-a],dtype=np.uint8),np.asarray(self.mask[i],dtype=np.uint8)

class DS(Dataset):
    def __init__(self,cache,idx,y,conf): self.cache=cache;self.idx=np.asarray(idx);self.y=y;self.conf=conf
    def __len__(self): return len(self.idx)
    def __getitem__(self,k):
        i=int(self.idx[k]);v,m=self.cache.get(i);inst=[];valid=[];slots=[]
        for s in range(self.cache.slots):
            for c in self.cache.triplet_centers:
                inst.append(v[s,c-1:c+2])
                valid.append(int(m[s,c-1:c+2].min()>0));slots.append(s)
        return (torch.from_numpy(np.stack(inst)),torch.tensor(valid,dtype=torch.bool),torch.tensor(slots,dtype=torch.long),
                torch.from_numpy(self.y[i]),torch.from_numpy(self.conf[i]),torch.tensor(i,dtype=torch.long))

class MIL(nn.Module):
    def __init__(self,backbone:str,img:int,nslots:int,pretrained=True,drop=.2):
        super().__init__()
        is_vit=('vit_' in backbone or 'dinov2' in backbone)
        kwargs=dict(pretrained=pretrained,num_classes=0)
        if is_vit: kwargs['img_size']=img
        else: kwargs['global_pool']='avg'
        self.backbone=timm.create_model(backbone,**kwargs)
        d=int(self.backbone.num_features); self.norm=nn.LayerNorm(d);self.slot=nn.Embedding(nslots,d)
        self.att=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,len(TARGETS)))
        self.cls=nn.Parameter(torch.empty(len(TARGETS),d));nn.init.trunc_normal_(self.cls,std=.02);self.bias=nn.Parameter(torch.zeros(len(TARGETS)))
    def forward(self,x,valid,slot):
        b,k=x.shape[:2];f=self.backbone(x.flatten(0,1)).view(b,k,-1);h=self.norm(f+self.slot(slot));a=self.att(h).masked_fill(~valid[...,None],-1e4);a=torch.softmax(a,1)
        p=torch.einsum('bkt,bkd->btd',a,h);return (p*self.cls[None]).sum(-1)+self.bias

def weak_auc(y,p,conf,target_indices=None):
    selected=set(range(len(TARGETS)) if target_indices is None else target_indices);vals=[]; per={}
    for j,t in enumerate(TARGETS):
        use=((y[:,j]>=.7)|(y[:,j]<=.3))&(conf[:,j]>=.6); yy=(y[use,j]>=.7).astype(np.uint8)
        a=float(roc_auc_score(yy,p[use,j])) if np.unique(yy).size==2 else float('nan');per[t]={'auc':a,'n':int(use.sum()),'pos':int(yy.sum())}
        if j in selected: vals.append(a)
    return float(np.nanmean(vals)),per

def gold_auc(y,p,target_indices=None):
    selected=set(range(len(TARGETS)) if target_indices is None else target_indices);vals=[];per={}
    for j,t in enumerate(TARGETS):
        a=float(roc_auc_score(y[:,j],p[:,j]));per[t]=a
        if j in selected: vals.append(a)
    return float(np.mean(vals)),per

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-dir',required=True);ap.add_argument('--manifest-name',default='multicenter_manifest.json');ap.add_argument('--backbone',default='convnext_small');ap.add_argument('--img',type=int,default=224);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--batch',type=int,default=2);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--head-lr',type=float,default=2e-4);ap.add_argument('--backbone-lr',type=float,default=2e-5);ap.add_argument('--wd',type=float,default=.02);ap.add_argument('--min-conf',type=float,default=0.0);ap.add_argument('--seed',type=int,default=20260913);ap.add_argument('--out',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--save-every',type=int,default=5);ap.add_argument('--augment',action='store_true');ap.add_argument('--patience',type=int,default=0);ap.add_argument('--min-delta',type=float,default=0.0);ap.add_argument('--weak-labels',default='labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv');ap.add_argument('--confidence-mode',choices=['auto','columns','ones','distance'],default='auto',help='auto uses <target>__conf columns when present, otherwise unit confidence; distance uses 2*abs(p-0.5)');ap.add_argument('--target-mask',default='',help='Comma-separated target names; empty means all targets');args=ap.parse_args();ti=parse_target_mask(args.target_mask)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.backends.cuda.matmul.allow_tf32=True
    R=workspace_root();out=workspace_path(args.out);out.mkdir(parents=True,exist_ok=True);cache=Shards(workspace_path(args.cache_dir),args.manifest_name);ids=cache.uids
    folds=pd.read_csv(R/'experiments'/'EXP026_GROUPED_FROZEN_PROBE'/'folds.csv',dtype={UID:str}).set_index(UID).loc[ids]
    f=folds.fold.to_numpy(int);gold=folds.is_gold.to_numpy(bool);quar=folds.quarantined_gold_report_duplicate.to_numpy(bool)
    weak_raw=pd.read_csv(R/args.weak_labels,dtype={UID:str}).set_index(UID)
    weak=weak_raw.reindex(ids);y=weak[TARGETS].to_numpy(np.float32)
    conf_cols=[t+'__conf' for t in TARGETS]; has_conf=all(c in weak.columns for c in conf_cols)
    mode=args.confidence_mode
    if mode=='auto': mode='columns' if has_conf else 'ones'
    if mode=='columns':
        if not has_conf: raise ValueError(f'confidence columns missing in {args.weak_labels}')
        conf=weak[conf_cols].to_numpy(np.float32)
    elif mode=='distance': conf=(2.0*np.abs(y-0.5)).astype(np.float32)
    else: conf=np.ones_like(y,dtype=np.float32)
    missing=np.isnan(y).any(axis=1); y=np.nan_to_num(y,nan=0.5); conf=np.nan_to_num(conf,nan=0.0); conf[missing]=0.0
    off=pd.read_csv(R/'data'/'train.csv',dtype={UID:str}).set_index(UID).loc[ids];gy=off.loc[np.array(ids)[gold],TARGETS].to_numpy(np.float32)
    eligible=(f>=0)&~gold&~quar;tr=np.where(eligible&(f!=args.fold))[0];va=np.where(eligible&(f==args.fold))[0];gi=np.where(gold)[0]
    ds_tr=DS(cache,tr,y,conf);ds_va=DS(cache,va,y,conf);ds_g=DS(cache,gi,np.nan_to_num(off[TARGETS].to_numpy(np.float32),nan=.5),np.ones_like(conf))
    dl_tr=DataLoader(ds_tr,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0);dl_va=DataLoader(ds_va,batch_size=max(2,args.batch),shuffle=False,num_workers=0,pin_memory=True);dl_g=DataLoader(ds_g,batch_size=max(2,args.batch),shuffle=False,num_workers=0,pin_memory=True)
    dev='cuda';model=MIL(args.backbone,args.img,cache.slots,True).cuda();bb=list(model.backbone.parameters());bbid={id(p) for p in bb};head=[p for p in model.parameters() if id(p) not in bbid]
    opt=torch.optim.AdamW([{'params':bb,'lr':args.backbone_lr},{'params':head,'lr':args.head_lr}],weight_decay=args.wd);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs);scaler=torch.amp.GradScaler('cuda')
    mean=torch.tensor([.485,.456,.406],device=dev).view(1,1,3,1,1);std=torch.tensor([.229,.224,.225],device=dev).view(1,1,3,1,1)
    def augment_x(x):
        if not args.augment:
            return x
        b,k=x.shape[:2]; base=int(x.shape[-1]); lo=max(32,int(round(base*.94))); out=[]
        for bi in range(b):
            side=int(torch.randint(lo,base+1,(1,),device=x.device).item())
            maxoff=base-side
            y0=int(torch.randint(0,maxoff+1,(1,),device=x.device).item()) if maxoff else 0
            x0=int(torch.randint(0,maxoff+1,(1,),device=x.device).item()) if maxoff else 0
            z=x[bi,:,:,y0:y0+side,x0:x0+side]
            z=F.interpolate(z,size=(base,base),mode='bilinear',align_corners=False)
            gamma=float(torch.empty(1,device=x.device).uniform_(0.88,1.12).item())
            gain=float(torch.empty(1,device=x.device).uniform_(0.93,1.07).item())
            z=torch.clamp((z.clamp(0,1).pow(gamma))*gain,0,1)
            if torch.rand(1,device=x.device).item()<0.35:
                z=torch.clamp(z+torch.randn_like(z)*0.008,0,1)
            out.append(z)
        return torch.stack(out,0)

    @torch.no_grad()
    def pred(dl):
        model.eval();pp=[];yy=[];cc=[];ii=[]
        for x,v,s,yt,ct,it in dl:
            x=x.cuda(non_blocking=True).float().div_(255);b,k=x.shape[:2];x=F.interpolate(x.flatten(0,1),size=(args.img,args.img),mode='bilinear',align_corners=False).view(b,k,3,args.img,args.img);x=(x-mean)/std;v=v.cuda();s=s.cuda()
            with torch.amp.autocast('cuda',dtype=torch.float16): z=model(x,v,s)
            pp.append(torch.sigmoid(z).float().cpu().numpy());yy.append(yt.numpy());cc.append(ct.numpy());ii.append(it.numpy())
        return np.concatenate(ii),np.concatenate(yy),np.concatenate(cc),np.concatenate(pp)
    hist=[]; start_ep=1; last=out/'last.pt'; best_metric=-1.0; best_epoch=0; stale=0
    if args.resume and last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sch']); scaler.load_state_dict(ck['scaler']); start_ep=int(ck['epoch'])+1
        hp=out/'history.csv'
        if hp.exists(): hist=pd.read_csv(hp).to_dict('records');
        if hist: best_metric=max(float(r['weak_highconf_auc']) for r in hist); best_epoch=int(max(hist,key=lambda r:float(r['weak_highconf_auc']))['epoch']); stale=max(0,(start_ep-1)-best_epoch)
        print(json.dumps({'resume_from_epoch':start_ep-1,'out':str(out)}),flush=True)
    for ep in range(start_ep,args.epochs+1):
        model.train();losses=[];t0=time.time()
        for x,v,s,yt,ct,it in dl_tr:
            x=x.cuda(non_blocking=True).float().div_(255);x=augment_x(x);b,k=x.shape[:2];x=F.interpolate(x.flatten(0,1),size=(args.img,args.img),mode='bilinear',align_corners=False).view(b,k,3,args.img,args.img);x=(x-mean)/std;v=v.cuda();s=s.cuda();yt=yt.cuda();ct=ct.cuda();opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',dtype=torch.float16):
                z=model(x,v,s);w=(.1+.9*ct.square())*(ct>=args.min_conf);loss=(F.binary_cross_entropy_with_logits(z[:,ti],yt[:,ti],reduction='none')*w[:,ti]).sum()/w[:,ti].sum().clamp_min(1)
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        sch.step();_,vy,vc,vp=pred(dl_va);vm,vper=weak_auc(vy,vp,vc,ti);row={'epoch':ep,'loss':float(np.mean(losses)),'weak_highconf_auc':vm,'seconds':time.time()-t0};hist.append(row);print(json.dumps(row),flush=True)
        improved=vm > best_metric + args.min_delta
        if improved:
            best_metric=float(vm); best_epoch=int(ep); stale=0
            torch.save({'model':model.state_dict(),'args':vars(args),'epoch':ep,'weak_highconf_auc':vm},out/'best.pt')
        else:
            stale+=1
        state={'model':model.state_dict(),'opt':opt.state_dict(),'sch':sch.state_dict(),'scaler':scaler.state_dict(),'args':vars(args),'epoch':ep,'best_metric':best_metric,'best_epoch':best_epoch,'stale':stale}; tmp=out/'last.tmp.pt';torch.save(state,tmp);tmp.replace(last);
        if ep%args.save_every==0 or ep==args.epochs: torch.save({'model':model.state_dict(),'args':vars(args),'epoch':ep},out/f'fold{args.fold}_e{ep:02d}.pt')
        pd.DataFrame(hist).to_csv(out/'history.csv',index=False)
        if args.patience>0 and stale>=args.patience:
            print(json.dumps({'early_stop_epoch':ep,'best_epoch':best_epoch,'best_metric':best_metric}),flush=True); break
    bp=out/'best.pt'
    if bp.exists():
        bck=torch.load(bp,map_location='cpu',weights_only=False);model.load_state_dict(bck['model']);best_epoch=int(bck['epoch']);best_metric=float(bck.get('weak_highconf_auc',best_metric))
    vi,vy,vc,vp=pred(dl_va);svm,svper=weak_auc(vy,vp,vc,ti);pd.DataFrame(vp,columns=TARGETS).assign(StudyInstanceUID=np.array(ids)[vi]).to_csv(out/'weak_val_pred.csv',index=False)
    _,_,_,gp=pred(dl_g);gm,gper=gold_auc(gy,gp,ti);gall,_=gold_auc(gy,gp);pd.DataFrame(gp,columns=TARGETS).assign(StudyInstanceUID=np.array(ids)[gi]).to_csv(out/'gold_pred.csv',index=False)
    receipt={'contract':'multicenter_dynamic_2p5d_clean_fold_v1','args':vars(args),'cache_contract':cache.m.get('contract'),'cache_slots':cache.slots,'cache_depth':cache.depth,'instances_per_study':cache.slots*len(cache.triplet_centers),'backbone':args.backbone,'img':args.img,'fold':args.fold,'max_epochs':args.epochs,'best_epoch':best_epoch,'train':len(tr),'val':len(va),'gold':len(gi),'weak_label_rows':int(len(weak_raw)),'weak_label_missing_cache_rows':int(missing.sum()),'resolved_confidence_mode':mode,'selected_target_indices':ti,'selected_targets':[TARGETS[j] for j in ti],'best_weak_highconf_auc':best_metric,'selected_checkpoint_weak_highconf_auc':svm,'weak_highconf_per_target':svper,'gold_selected_targets_macro_auc_diagnostic_only':gm,'gold_all12_macro_auc_diagnostic_only':gall,'gold_per_target':gper,'no_gold_epoch_selection':True,'augment':bool(args.augment),'patience':args.patience}
    (out/'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__': main()














