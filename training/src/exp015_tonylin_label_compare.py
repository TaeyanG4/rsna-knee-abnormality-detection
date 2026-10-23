try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
ROOT=workspace_root()
UID='StudyInstanceUID'
T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
g=pd.read_csv(ROOT/'gold/gold_labels.csv',dtype={UID:str}).set_index(UID)
f=pd.read_csv(ROOT/'labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv',dtype={UID:str}).set_index(UID)[T]
td=pd.read_csv(ROOT/'labels/fresh_tonylin/refined_soft_labels.csv',dtype={UID:str}).set_index(UID)
t=td[[x+'_soft' for x in T]].copy(); t.columns=T
ids=g.index; y=g[T].to_numpy(float); pf=f.loc[ids].to_numpy(float); pt=t.loc[ids].to_numpy(float)
rows=[]
for j,name in enumerate(T):
 a0=roc_auc_score(y[:,j],pf[:,j]); a1=roc_auc_score(y[:,j],pt[:,j])
 ng=[x for x in t.index if x not in set(ids)]
 sp=float(spearmanr(f.loc[ng,name],t.loc[ng,name]).statistic)
 pr=float(pearsonr(f.loc[ng,name],t.loc[ng,name]).statistic)
 rows.append({'target':name,'flight_auc':a0,'tonylin_auc':a1,'delta':a1-a0,'non_gold_spearman':sp,'non_gold_pearson':pr,'tonylin_unique_non_gold':int(t.loc[ng,name].nunique())})
out=pd.DataFrame(rows)
# paired study bootstrap
rng=np.random.default_rng(20260912); ds=[]; n=len(ids)
for _ in range(5000):
 ix=rng.integers(0,n,n)
 if any(np.unique(y[ix,j]).size<2 for j in range(len(T))): continue
 a0=np.mean([roc_auc_score(y[ix,j],pf[ix,j]) for j in range(len(T))])
 a1=np.mean([roc_auc_score(y[ix,j],pt[ix,j]) for j in range(len(T))])
 ds.append(a1-a0)
d=np.asarray(ds)
res={'experiment':'EXP015_TONYLIN_VS_FLIGHT_LABEL','flight_macro':float(out.flight_auc.mean()),'tonylin_macro':float(out.tonylin_auc.mean()),'observed_delta':float(out.delta.mean()),'bootstrap_valid':int(len(d)),'bootstrap_mean_delta':float(d.mean()),'bootstrap_std':float(d.std()),'ci95_low':float(np.quantile(d,.025)),'ci95_high':float(np.quantile(d,.975)),'fraction_positive':float((d>0).mean()),'gold_exact_soft_rows':int(np.all(pt==y,axis=1).sum()),'tonylin_is_gold_rows':int(td['is_gold'].astype(bool).sum()),'selection_warning':'Tonylin flags all 58 gold rows and provenance/calibration method is not documented; treat as gold-exposed candidate, not independent teacher validation.'}
out.to_csv(ROOT/'experiments/exp015_tonylin_vs_flight.csv',index=False)
(ROOT/'reports/EXP015_tonylin_vs_flight.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2)); print(out.to_string(index=False))
