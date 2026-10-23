try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
ROOT=workspace_root()
CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'
WEIGHT=ROOT/'weights'/'dreaddevelopment__raptor-knee-maxspan'/'raptor_ft_coatnet_v5_full_swa.pt'

def build_backbone(arch):
    hybrid=arch.startswith(('maxvit','maxxvit','coatnet','coat_','convnext'))
    is_vit=(not hybrid) and any(k in arch for k in ('vit','deit','dinov2','eva','beit'))
    kw=dict(pretrained=False,num_classes=0,in_chans=3)
    kw.update(global_pool='token',dynamic_img_size=True) if is_vit else kw.update(global_pool='avg')
    return timm.create_model(arch,**kw)
class RaptorClassifier(nn.Module):
    def __init__(self,backbone,F_dim=768,n=12,drop=.2):
        super().__init__(); self.backbone=backbone; self.norm=nn.LayerNorm(F_dim)
        self.att=nn.Sequential(nn.Linear(F_dim,256),nn.Tanh(),nn.Dropout(drop),nn.Linear(256,n))
        self.clsW=nn.Parameter(torch.zeros(n,F_dim)); self.clsb=nn.Parameter(torch.zeros(n)); self.n=n
    def encode(self,x):
        B,K=x.shape[:2]; f=self.backbone(x.flatten(0,1)); return f.view(B,K,-1)
    def head(self,feats):
        h=self.norm(feats); a=torch.softmax(self.att(h),dim=1); pooled=torch.einsum('bkn,bkf->bnf',a,h); return (pooled*self.clsW).sum(-1)+self.clsb
    def forward(self,x): return self.head(self.encode(x))
class Shards:
    def __init__(self):
        m=json.loads((CACHE/'train_cache_shards_manifest.json').read_text()); self.s=[]
        for r in m['shards']: self.s.append((r['start'],r['stop'],np.load(CACHE/r['file'],mmap_mode='r')))
    def get(self,i):
        for a,b,x in self.s:
            if a<=i<b:return x[i-a]
        raise IndexError

def make_windows(x,mask):
    # x: 6,9,H,W ; 7 triplets per available slot
    ws=[]
    for s in range(6):
        if mask[s].sum()==0: continue
        for c in range(1,8):
            idx=[max(0,c-1),c,min(8,c+1)]
            im=x[s,idx].astype(np.float32)/255.0
            ws.append(im)
    if not ws: ws=[np.zeros((3,224,224),np.float32)]
    return np.stack(ws)

def main():
    u=pd.read_csv(CACHE/'train_cache_uids.csv'); train=pd.read_csv(ROOT/'data'/'train.csv').set_index('StudyInstanceUID'); masks=np.load(CACHE/'train_cache_masks.npy'); sh=Shards()
    gold=[]; yy=[]
    for i,uid in enumerate(u.StudyInstanceUID.astype(str)):
        vals=train.loc[uid,TARGETS].to_numpy(np.float32)
        if np.isfinite(vals).all(): gold.append(i); yy.append(vals)
    yy=np.stack(yy)
    ck=torch.load(WEIGHT,map_location='cpu',weights_only=False); arch=ck.get('arch','coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'); bb=build_backbone(arch); model=RaptorClassifier(bb,F_dim=bb.num_features); model.load_state_dict(ck['model'],strict=True); model.eval().cuda()
    preds=[]; t0=time.time()
    with torch.no_grad():
        for j,i in enumerate(gold):
            w=make_windows(np.array(sh.get(i),copy=False),masks[i])
            xt=torch.from_numpy(w).cuda()
            chunks=[]
            for k in range(0,len(xt),8):
                z=F.interpolate(xt[k:k+8],size=(384,384),mode='bilinear',align_corners=False)
                with torch.amp.autocast('cuda',dtype=torch.float16): chunks.append(model.backbone(z).float().cpu())
            feat=torch.cat(chunks)[None].cuda()
            with torch.amp.autocast('cuda',dtype=torch.float16): p=torch.sigmoid(model.head(feat)).float().cpu().numpy()[0]
            preds.append(p)
            if (j+1)%10==0: print(j+1,'/',len(gold),flush=True)
    p=np.stack(preds); auc=[]
    for k,t in enumerate(TARGETS): auc.append(float(roc_auc_score(yy[:,k],p[:,k])))
    out={'macro':float(np.mean(auc)),'seconds':time.time()-t0,'per_target':dict(zip(TARGETS,auc))}
    od=ROOT/'experiments'/'exp024_raptor_v5_compact_cache_gold58'; od.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(p,columns=TARGETS).assign(StudyInstanceUID=[u.iloc[i].StudyInstanceUID for i in gold]).to_csv(od/'pred.csv',index=False)
    (od/'result.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

