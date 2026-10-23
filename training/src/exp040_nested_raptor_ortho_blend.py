try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
R=workspace_root()
T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
out=R/'experiments'/'EXP040_NESTED_RAPTOR_ORTHO_BLEND'; out.mkdir(parents=True,exist_ok=True)
r=pd.read_csv(R/'experiments'/'EXP030_RAPTOR_CANONICAL_HEAD_PSEUDOOOF'/'noncanonical_weak_oof.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
o=pd.read_csv(R/'experiments'/'EXP033_ORTHODIFF_SAP_OOF'/'weak_oof.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
f=pd.read_csv(R/'experiments'/'EXP026_GROUPED_FROZEN_PROBE'/'folds.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
te=pd.read_csv(R/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
ids=r.index.intersection(o.index).intersection(f.index).intersection(te.index)
r=r.loc[ids,T].to_numpy(float); o=o.loc[ids,T].to_numpy(float); folds=f.loc[ids,'fold'].to_numpy(int); y=te.loc[ids,T].to_numpy(float); conf=te.loc[ids,[t+'__conf' for t in T]].to_numpy(float)
def ranks(x):
 return np.stack([pd.Series(x[:,j]).rank(method='average',pct=True).to_numpy() for j in range(x.shape[1])],1)
rr,oo=ranks(r),ranks(o)
def auc_target(pred, idx, j):
 use=idx & (((y[:,j]>=.7)|(y[:,j]<=.3)) & (conf[:,j]>=.6))
 yy=(y[use,j]>=.7).astype(int)
 return float(roc_auc_score(yy,pred[use,j])) if np.unique(yy).size==2 else np.nan
alphas=np.array([0,.025,.05,.075,.10,.15,.20,.30],float)
rows=[]; pred_global=np.full_like(rr,np.nan); pred_target=np.full_like(rr,np.nan)
for fold in range(5):
 tr=(folds>=0)&(folds!=fold); va=folds==fold
 # global alpha selected on training macro, ties favor smaller alpha
 train_scores=[]
 for a in alphas:
  p=(1-a)*rr+a*oo
  vals=[auc_target(p,tr,j) for j in range(len(T))]
  train_scores.append(float(np.nanmean(vals)))
 bestg=int(np.flatnonzero(np.isclose(train_scores,np.nanmax(train_scores),rtol=0,atol=1e-12))[0]); ag=float(alphas[bestg])
 pg=(1-ag)*rr+a*oo if False else (1-ag)*rr+ag*oo
 pred_global[va]=pg[va]
 # target alpha selected independently on train, tie favor zero/smaller
 ats=[]
 for j in range(len(T)):
  sc=[auc_target((1-a)*rr+a*oo,tr,j) for a in alphas]
  m=np.nanmax(sc); k=int(np.flatnonzero(np.isclose(sc,m,rtol=0,atol=1e-12))[0]); ats.append(float(alphas[k]))
 for j,a in enumerate(ats): pred_target[va,j]=((1-a)*rr[:,j]+a*oo[:,j])[va]
 rows.append({'fold':fold,'global_alpha':ag,'train_macro_at_alpha':train_scores[bestg],'target_alphas':dict(zip(T,ats))})
# evaluate OOF across folds
def eval_pred(p):
 vals=[]; detail={}
 for j,t in enumerate(T):
  a=auc_target(p,folds>=0,j); vals.append(a); detail[t]=a
 return float(np.nanmean(vals)),detail
base_macro,base=eval_pred(rr); g_macro,g=eval_pred(pred_global); t_macro,tg=eval_pred(pred_target)
fixed={}
for a in [0.025,0.05,0.075,0.10,0.15,0.20]: fixed[str(a)]=eval_pred((1-a)*rr+a*oo)[0]
# same fold map sanity vs Ortho's recorded folds where available
of=pd.read_csv(R/'experiments'/'EXP033_ORTHODIFF_SAP_OOF'/'folds.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
common=pd.Index(ids).intersection(of.index); match=float((f.loc[common,'fold'].to_numpy(int)==of.loc[common,'fold'].to_numpy(int)).mean())
res={'n':len(ids),'fold_match_fraction_exp026_vs_exp033':match,'alphas':alphas.tolist(),'base_macro':base_macro,'nested_global_macro':g_macro,'nested_global_delta':g_macro-base_macro,'nested_target_macro':t_macro,'nested_target_delta':t_macro-base_macro,'fixed_macro':fixed,'base_per_target':base,'nested_global_per_target':g,'nested_target_per_target':tg,'fold_choices':rows,'caveat':'Raptor backbone saw full weak train; absolute pseudo-OOF is leaky. Interpret only paired blend direction.'}
(out/'result.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
pd.DataFrame({'target':T,'base':[base[t] for t in T],'nested_global':[g[t] for t in T],'nested_target':[tg[t] for t in T]}).assign(delta_global=lambda d:d.nested_global-d.base,delta_target=lambda d:d.nested_target-d.base).to_csv(out/'per_target.csv',index=False)
print(json.dumps(res,indent=2))

