try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
R=workspace_root()
T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
CACHE=R/'kaggle_notebooks'/'own_exp022_shard_cache'; SPLIT=R/'experiments'/'EXP026_GROUPED_FROZEN_PROBE'/'folds.csv'; WEIGHT=R/'weights'/'dreaddevelopment__raptor-knee-maxspan'/'raptor_ft_coatnet_v5_full_swa.pt'; FEAT=R/'experiments'/'exp025_raptor_features'; OUT=R/'experiments'/'EXP037_RAPTOR_ORTHO_FIXED_BLEND'; OUT.mkdir(parents=True,exist_ok=True)
class Head(nn.Module):
 def __init__(self,d,n=12):
  super().__init__(); self.norm=nn.LayerNorm(d); self.att=nn.Sequential(nn.Linear(d,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,n)); self.clsW=nn.Parameter(torch.zeros(n,d)); self.clsb=nn.Parameter(torch.zeros(n))
 def forward(self,f,m):
  h=self.norm(f); s=self.att(h).masked_fill(m[...,None]<=0,-1e4); a=torch.softmax(s,1); pool=torch.einsum('bkn,bkf->bnf',a,h); return (pool*self.clsW).sum(-1)+self.clsb
ids=pd.read_csv(CACHE/'train_cache_uids.csv').StudyInstanceUID.astype(str).tolist(); ft=pd.read_csv(SPLIT).set_index('StudyInstanceUID').loc[ids]; folds=ft.fold.to_numpy(int); eligible=folds>=0; gold=ft.is_gold.to_numpy(bool); gi=np.where(gold)[0]
teacher=pd.read_csv(R/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv').set_index('StudyInstanceUID').loc[ids]; y=teacher[T].to_numpy(np.float32); conf=teacher[[t+'__conf' for t in T]].to_numpy(np.float32)
tr=pd.read_csv(R/'data'/'train.csv').set_index('StudyInstanceUID').loc[ids]; gy=tr.loc[np.array(ids)[gold],T].to_numpy(float)
feat=np.load(FEAT/'features_f16.npy',mmap_mode='r'); mask=np.load(FEAT/'window_mask.npy').astype(np.float32)
ck=torch.load(WEIGHT,map_location='cpu',weights_only=False); st=ck['model']; d=int(st['norm.weight'].numel()); init={k:v for k,v in st.items() if not k.startswith('backbone.')}; gps=[]; SEED=20260913
for f in range(5):
 torch.manual_seed(SEED+f); rng=np.random.default_rng(SEED+f); model=Head(d).cuda(); model.load_state_dict(init,strict=True); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3); a=np.where(eligible&(folds!=f))[0]
 for ep in range(10):
  order=rng.permutation(a); model.train()
  for s in range(0,len(order),64):
   ii=order[s:s+64]; x=torch.from_numpy(np.asarray(feat[ii],np.float32)).cuda(); m=torch.from_numpy(mask[ii]).cuda(); yy=torch.from_numpy(y[ii]).cuda(); ww=torch.from_numpy(conf[ii]).cuda(); opt.zero_grad(set_to_none=True); log=model(x,m); loss=(F.binary_cross_entropy_with_logits(log,yy,reduction='none')*(.25+.75*ww)).mean(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
 model.eval(); oo=[]
 with torch.no_grad():
  for s in range(0,len(gi),64):
   jj=gi[s:s+64]; x=torch.from_numpy(np.asarray(feat[jj],np.float32)).cuda(); m=torch.from_numpy(mask[jj]).cuda(); oo.append(torch.sigmoid(model(x,m)).cpu().numpy())
 gps.append(np.concatenate(oo)); print('fold',f,'done',flush=True); del model,opt; torch.cuda.empty_cache()
rp=np.mean(gps,axis=0); gold_ids=np.array(ids)[gold]; pd.DataFrame(rp,columns=T).assign(StudyInstanceUID=gold_ids).to_csv(OUT/'raptor_gold_pred.csv',index=False)
op=pd.read_csv(R/'experiments'/'EXP033_ORTHODIFF_SAP_OOF'/'gold_pred.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[gold_ids,T].to_numpy(float)
def rank(x): return np.stack([pd.Series(x[:,j]).rank(method='average',pct=True).to_numpy() for j in range(x.shape[1])],1)
rr,oo=rank(rp),rank(op); blend=.9*rr+.1*oo
ra=[roc_auc_score(gy[:,j],rp[:,j]) for j in range(12)]; oa=[roc_auc_score(gy[:,j],op[:,j]) for j in range(12)]; ba=[roc_auc_score(gy[:,j],blend[:,j]) for j in range(12)]
res=pd.DataFrame({'target':T,'raptor':ra,'ortho':oa,'blend_0p10':ba}); res['delta']=res.blend_0p10-res.raptor
print(res.to_string(index=False,float_format=lambda x:f'{x:.6f}')); print('macro',np.mean(ra),np.mean(oa),np.mean(ba),'delta',np.mean(ba)-np.mean(ra)); res.to_csv(OUT/'gold_fixed_blend.csv',index=False)
