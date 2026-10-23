from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import gc, hashlib, json, math, os, random, sys, time, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

PLANE = "CORONAL"
FOLD = 1
EPOCHS = 5
BATCH = 4
HEAD_LR = 5e-4
BACKBONE_LR = HEAD_LR * 0.05
SEED = 20260913
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Effusion"]
ALL_TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
SETTINGS = {
    "AXIAL": {"slot": "AX", "weight": "axial_model.pt", "timestep": 200, "block": "mid_0"},
    "CORONAL": {"slot": "COR", "weight": "coronal_model.pt", "timestep": 50, "block": "mid_2"},
    "SAGITTAL": {"slot": "SAG", "weight": "sagittal_model.pt", "timestep": 150, "block": "mid_0"},
}
EXPECTED_WEIGHT_SHA = {
    "AXIAL": "f7daa0406c6312dacebf1e56ecfdf14b1f3c98c59b24f2fa75d1899490d31a3e",
    "CORONAL": "dcce97a061af1a6df1f902cc0d5a7f604dc9bd0f52f3b3714be39b288dc541e6",
    "SAGITTAL": "3afc323c58b6446c1ea6dc4f09133d744e5a1609dba7662ac9a15ca688e34951",
}
EXPECTED_TRAINABLE = 13_702_144
ROOT_LOCAL=workspace_root()
OUT=ROOT_LOCAL/'experiments'/'EXP039B_ORTHODIFF_FT_CORONAL_LOCAL'
OUT.mkdir(parents=True,exist_ok=True)

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

def log(*x): print(f"[{PLANE}]", *x, flush=True)
def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16<<20),b''): h.update(b)
    return h.hexdigest()
def report_key(x): return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',str(x)).lower().split()).encode()).hexdigest()
def find_one(name:str):
    if name == 'report_labels_v4hybrid.csv': return ROOT_LOCAL/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/name
    p=ROOT_LOCAL/'external'/'research_20260913'/'OrthoDiffusion'/'orthodiffusion'/name
    if not p.is_file(): raise FileNotFoundError(p)
    return p
def find_comp_root():
    return ROOT_LOCAL/'data'

def find_ortho_code():
    return ROOT_LOCAL/'external'/'research_20260913'/'OrthoDiffusion'/'code_99d15186'/'OrthoDiffusion-99d15186bb8face1728263023d8414d9405c8f7e'

class Shards:
    def __init__(self):
        self.root=ROOT_LOCAL/'cache'/'EXP032_orthodiff'
        mp=self.root/'orthodiffusion_cache_manifest.json'; self.m=json.loads(mp.read_text())
        self.slots=list(self.m['slots']); self.slot=self.slots.index(SETTINGS[PLANE]['slot'])
        self.shards=[]
        for r in self.m['shards']:
            p=self.root/r['file'];
            if p.stat().st_size!=int(r['bytes']): raise RuntimeError(f'bad shard size {p}')
            self.shards.append((int(r['start']),int(r['stop']),np.load(p,mmap_mode='r')))
        self.n=int(self.m['studies'])
        self.uids=pd.read_csv(self.root/self.m['uid_file']).StudyInstanceUID.astype(str).tolist()
        self.mask=np.load(self.root/self.m['mask_file']).astype(np.uint8)[:,self.slot]
        self.lookup=np.empty(self.n,np.int16)
        for si,(a,b,_) in enumerate(self.shards): self.lookup[a:b]=si
    def batch(self, idx):
        out=np.empty((len(idx),16,256,256),np.uint8)
        for q,i in enumerate(idx):
            si=int(self.lookup[int(i)]); a,b,x=self.shards[si]; out[q]=x[int(i)-a,self.slot]
        return out

class SelfAttnPooling3D(nn.Module):
    def __init__(self,c=256,h=16):
        super().__init__(); self.h=h; self.q=nn.Linear(c,c); self.k=nn.Linear(c,c); self.v=nn.Linear(c,c); self.o=nn.Linear(c,c)
    def forward(self,f):
        B,C,D,H,W=f.shape; N=D*H*W; x=f.view(B,C,N).permute(0,2,1); d=C//self.h
        q=self.q(x).view(B,N,self.h,d).transpose(1,2); k=self.k(x).view(B,N,self.h,d).transpose(1,2); v=self.v(x).view(B,N,self.h,d).transpose(1,2)
        a=torch.softmax(torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d),-1); ctx=torch.matmul(a,v).transpose(1,2).contiguous().view(B,N,C)
        return torch.cat([f.mean((2,3,4)),self.o(ctx.mean(1))],1)
class PlaneClassifier(nn.Module):
    def __init__(self): super().__init__(); self.attn_pool=SelfAttnPooling3D(); self.fc=nn.Linear(512,len(TARGETS))
    def forward(self,f): return self.fc(self.attn_pool(f))

def aucs(y,p):
    vals=[]
    for j,t in enumerate(TARGETS): vals.append(float(roc_auc_score(y[:,j],p[:,j])))
    return dict(zip(TARGETS,vals)),float(np.mean(vals))
def weak_aucs(y,p,conf):
    d={}; vals=[]
    for j,t in enumerate(TARGETS):
        use=((y[:,j]>=.7)|(y[:,j]<=.3))&(conf[:,j]>=.6); yy=(y[use,j]>=.7).astype(int)
        a=float(roc_auc_score(yy,p[use,j])) if np.unique(yy).size==2 else float('nan'); d[t]={'auc':a,'n':int(use.sum()),'pos':int(yy.sum())}; vals.append(a)
    return d,float(np.nanmean(vals))

def main():
    t0=time.time(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cache=Shards(); comp=find_comp_root(); code=find_ortho_code(); sys.path.insert(0,str(code))
    from diffusion_model.unet import create_model
    from diffusion_model.trainer import GaussianDiffusion
    from diffusion_model.modules import AttentionBlock
    AttentionBlock.forward = lambda self, value: self._forward(value)
    # public official checkpoint
    wp=find_one(SETTINGS[PLANE]['weight']); actual=sha256(wp)
    if actual!=EXPECTED_WEIGHT_SHA[PLANE]: raise RuntimeError((PLANE,actual))
    labels=find_one('report_labels_v4hybrid.csv')
    train=pd.read_csv(comp/'train.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[cache.uids]
    teacher=pd.read_csv(labels,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[cache.uids]
    y=teacher[TARGETS].to_numpy(np.float32); conf=teacher[[t+'__conf' for t in TARGETS]].to_numpy(np.float32)
    if not np.isfinite(y).all() or not np.isfinite(conf).all(): raise RuntimeError('teacher nonfinite')
    gold=train[ALL_TARGETS].notna().all(axis=1).to_numpy(); gy=train.loc[np.array(cache.uids)[gold],TARGETS].to_numpy(np.float32)
    groups=train.Report.map(report_key).to_numpy(); gold_groups=set(groups[gold]); quarantine=np.array([g in gold_groups for g in groups])&~gold
    eligible=~gold&~quarantine&(cache.mask>0)
    # exact deterministic greedy group split used in prior diagnostics
    rng=np.random.default_rng(SEED); sizes=pd.Series(groups[eligible]).value_counts(); order=list(sizes.index); rng.shuffle(order); order.sort(key=lambda g:-sizes[g]); loads=np.zeros(5,int); mp={}
    for g in order: f=int(np.argmin(loads)); mp[g]=f; loads[f]+=int(sizes[g])
    folds=np.array([mp.get(g,-1) if ok else -1 for g,ok in zip(groups,eligible)],int)
    tri=np.where(eligible&(folds!=FOLD))[0]; vi=np.where(eligible&(folds==FOLD))[0]; gi=np.where(gold&(cache.mask>0))[0]
    log('cache',cache.n,'slot',cache.slot,'train',len(tri),'val',len(vi),'gold_valid',len(gi),'quarantine',int(quarantine.sum()),'fold_loads',loads.tolist())
    # model
    den=create_model(256,64,1,in_channels=1,out_channels=1,use_checkpoint=False).cuda()
    diff=GaussianDiffusion(den,image_size=256,depth_size=16,timesteps=1000,loss_type='l2').cuda()
    ck=torch.load(wp,map_location='cpu',weights_only=True); sd={}
    for k,v in ck['ema'].items(): sd[k.replace('denoise_fn.module.','denoise_fn.') if k.startswith('denoise_fn.module.') else k]=v
    st=diff.load_state_dict(sd,strict=False)
    if st.missing_keys or st.unexpected_keys: raise RuntimeError({'missing':st.missing_keys[:10],'unexpected':st.unexpected_keys[:10]})
    for p in diff.parameters(): p.requires_grad=False
    named=dict(diff.named_parameters()); names=sorted(n for n in named if 'input' in n.lower() or 'middle' in n.lower())
    for n in names: named[n].requires_grad=True
    ntrain=sum(named[n].numel() for n in names)
    if ntrain!=EXPECTED_TRAINABLE: raise RuntimeError(('trainable params',ntrain,names[:10]))
    head=PlaneClassifier().cuda()
    opt=torch.optim.Adam([{'params':head.parameters(),'lr':HEAD_LR},{'params':[named[n] for n in names],'lr':BACKBONE_LR}])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    scaler=torch.cuda.amp.GradScaler(enabled=True)
    setting=SETTINGS[PLANE]
    # fixed five epochs; validation is not used for model/epoch selection
    for ep in range(EPOCHS):
        diff.train(); head.train(); erng=np.random.default_rng(SEED+ep); order=erng.permutation(tri); losses=[]; torch.cuda.reset_peak_memory_stats()
        for a in range(0,len(order),BATCH):
            ii=order[a:a+BATCH]; x=cache.batch(ii).astype(np.float32)/127.5-1.; xt=torch.from_numpy(x[:,None]).cuda(non_blocking=True); yy=torch.from_numpy(y[ii]).cuda(); ww=torch.from_numpy(conf[ii]).cuda(); tt=torch.full((len(ii),),setting['timestep'],dtype=torch.long,device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                feat=diff.get_feature(xt,tt,name=setting['block'],use_amp=True); logit=head(feat); loss=(F.binary_cross_entropy_with_logits(logit,yy,reduction='none')*(.25+.75*ww)).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_([p for p in list(head.parameters())+[named[n] for n in names] if p.grad is not None],5.0); scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        sch.step(); log('epoch',ep+1,'loss',round(float(np.mean(losses)),6),'lr_head',opt.param_groups[0]['lr'],'peak_gib',round(torch.cuda.max_memory_allocated()/2**30,2))
    def predict(indices, return_embed=False):
        diff.eval(); head.eval(); ps=[]; es=[]
        with torch.no_grad():
            for bi,a in enumerate(range(0,len(indices),BATCH)):
                ii=indices[a:a+BATCH]; torch.manual_seed(SEED+100000+int(a)); torch.cuda.manual_seed_all(SEED+100000+int(a))
                x=cache.batch(ii).astype(np.float32)/127.5-1.; xt=torch.from_numpy(x[:,None]).cuda(non_blocking=True); tt=torch.full((len(ii),),setting['timestep'],dtype=torch.long,device='cuda')
                with torch.cuda.amp.autocast(enabled=True): feat=diff.get_feature(xt,tt,name=setting['block'],use_amp=True); emb=head.attn_pool(feat); logit=head.fc(emb)
                ps.append(torch.sigmoid(logit).float().cpu().numpy());
                if return_embed: es.append(emb.float().cpu().numpy())
        return np.concatenate(ps), (np.concatenate(es) if return_embed else None)
    pv,_=predict(vi); wpt,wm=weak_aucs(y[vi],pv,conf[vi]); pg,_=predict(gi); gpt,gm=aucs(gy[cache.mask[gold]>0],pg)
    log('VAL weak macro',wm,wpt); log('GOLD once macro',gm,gpt)
    # save deterministic embeddings for all valid studies for fusion training
    all_idx=np.arange(cache.n); pa,emb=predict(all_idx,True); emb[cache.mask==0]=0; pa[cache.mask==0]=.5
    np.save(OUT/f'orthodiff_ft_{PLANE.lower()}_embeddings_f16.npy',emb.astype(np.float16)); np.save(OUT/f'orthodiff_ft_{PLANE.lower()}_probs_f16.npy',pa.astype(np.float16))
    pd.DataFrame({'StudyInstanceUID':cache.uids,'fold':folds,'eligible':eligible,'gold':gold,'quarantine':quarantine,'plane_valid':cache.mask}).to_csv(OUT/f'orthodiff_ft_{PLANE.lower()}_split.csv',index=False)
    payload={'plane':PLANE,'targets':TARGETS,'settings':{'timestep':setting['timestep'],'block':setting['block']},'base_checkpoint_sha256':actual,'trainable_parameter_count':ntrain,'trainable_names':names,'backbone_trainable_state':{n:named[n].detach().cpu() for n in names},'head_state_dict':head.state_dict()}
    torch.save(payload,OUT/f'orthodiff_ft_{PLANE.lower()}_state.pt')
    receipt={'contract':'EXP039_orthodiff_partial_ft_fold1_v1','plane':PLANE,'fold':FOLD,'epochs':EPOCHS,'batch':BATCH,'head_lr':HEAD_LR,'backbone_lr':BACKBONE_LR,'seed':SEED,'train':len(tri),'val':len(vi),'gold_valid':len(gi),'quarantine':int(quarantine.sum()),'fold_loads':loads.tolist(),'weak_val_macro':wm,'weak_val_per_target':wpt,'gold_once_macro':gm,'gold_once_per_target':gpt,'base_checkpoint_sha256':actual,'trainable_parameter_count':ntrain,'elapsed_seconds':time.time()-t0,'embedding_shape':list(emb.shape)}
    (OUT/f'orthodiff_ft_{PLANE.lower()}_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8'); print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__': main()
