try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, timm
from sklearn.metrics import roc_auc_score
ROOT=workspace_root()
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
WEIGHT=ROOT/'weights'/'dreaddevelopment__raptor-knee-maxspan'/'raptor_ft_coatnet_v5_full_swa.pt'
CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'
OLD=ROOT/'experiments'/'exp025_raptor_features'
NEW=ROOT/'experiments'/'EXP028_RAPTOR_CANONICAL_FEATURE_HEAD'
OUT=ROOT/'experiments'/'EXP028_CANONICAL_PAIRED_AUDIT'; OUT.mkdir(parents=True,exist_ok=True)

def build_backbone(arch):
    hybrid=arch.startswith(('maxvit','maxxvit','coatnet','coat_','convnext')); is_vit=(not hybrid) and any(k in arch for k in ('vit','deit','dinov2','eva','beit'))
    kw=dict(pretrained=False,num_classes=0,in_chans=3); kw.update(global_pool='token',dynamic_img_size=True) if is_vit else kw.update(global_pool='avg'); return timm.create_model(arch,**kw)
class Raptor(nn.Module):
    def __init__(self,bb,Fdim,n=12):
        super().__init__(); self.backbone=bb; self.norm=nn.LayerNorm(Fdim); self.att=nn.Sequential(nn.Linear(Fdim,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,n)); self.clsW=nn.Parameter(torch.zeros(n,Fdim)); self.clsb=nn.Parameter(torch.zeros(n))
    def masked_head(self,f,mask):
        h=self.norm(f); s=self.att(h).masked_fill(mask[...,None]<=0,-1e4); a=torch.softmax(s,dim=1); pooled=torch.einsum('bkn,bkf->bnf',a,h); return (pooled*self.clsW).sum(-1)+self.clsb

def aucs(y,p): return np.array([roc_auc_score(y[:,k],p[:,k]) for k in range(12)],float)

def main():
    u=pd.read_csv(CACHE/'train_cache_uids.csv'); ids=u.StudyInstanceUID.astype(str).tolist(); train=pd.read_csv(ROOT/'data'/'train.csv').set_index('StudyInstanceUID')
    gold=[]; yy=[]
    for i,uid in enumerate(ids):
        v=train.loc[uid,TARGETS].to_numpy(np.float32)
        if np.isfinite(v).all(): gold.append(i); yy.append(v)
    gold=np.asarray(gold); y=np.stack(yy)
    ck=torch.load(WEIGHT,map_location='cpu',weights_only=False); bb=build_backbone(ck.get('arch','coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k')); model=Raptor(bb,bb.num_features); model.load_state_dict(ck['model'],strict=True); model.eval().cuda()
    def pred(root):
        f=np.load(root/'features_f16.npy',mmap_mode='r'); m=np.load(root/'window_mask.npy'); out=[]
        with torch.no_grad():
            for s in range(0,len(gold),16):
                ii=gold[s:s+16]; ft=torch.from_numpy(np.asarray(f[ii],dtype=np.float32)).cuda(); mt=torch.from_numpy(m[ii].astype(np.float32)).cuda()
                with torch.amp.autocast('cuda',dtype=torch.float16): out.append(torch.sigmoid(model.masked_head(ft,mt)).float().cpu().numpy())
        return np.concatenate(out)
    po=pred(OLD); pn=pred(NEW); ao=aucs(y,po); an=aucs(y,pn); delta=an-ao
    # paired study bootstrap; resamples with unusable single-class targets are skipped target-wise.
    rng=np.random.default_rng(2801); B=10000; ds=[]
    n=len(y)
    for _ in range(B):
        idx=rng.integers(0,n,n); d=[]
        for k in range(12):
            yy=y[idx,k]
            if np.unique(yy).size<2: continue
            d.append(roc_auc_score(yy,pn[idx,k])-roc_auc_score(yy,po[idx,k]))
        if d: ds.append(float(np.mean(d)))
    ds=np.asarray(ds)
    summary={'contract':'paired_same_checkpoint_head_orientation_only','n_gold':int(n),'old_macro':float(ao.mean()),'new_macro':float(an.mean()),'delta_macro':float(delta.mean()),'bootstrap_n':int(len(ds)),'bootstrap_mean':float(ds.mean()),'bootstrap_ci95':[float(np.quantile(ds,.025)),float(np.quantile(ds,.975))],'bootstrap_positive_probability':float((ds>0).mean()),'per_target':{TARGETS[k]:{'old':float(ao[k]),'new':float(an[k]),'delta':float(delta[k])} for k in range(12)}}
    pd.DataFrame(po,columns=TARGETS).assign(StudyInstanceUID=[ids[i] for i in gold]).to_csv(OUT/'noncanonical_checkpoint_pred.csv',index=False)
    pd.DataFrame(pn,columns=TARGETS).assign(StudyInstanceUID=[ids[i] for i in gold]).to_csv(OUT/'canonical_checkpoint_pred.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()