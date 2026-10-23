try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
root=workspace_root()
targets=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
g=pd.read_csv(root/'gold/gold_labels.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
r=pd.read_csv(root/'predictions/EXP006A_kaggle_v1/exp006a_original.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').reindex(g.index)
c=pd.read_csv(root/'predictions/CoAt_top3_probability_gold58_public.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').reindex(g.index)
rr=r[targets].rank(method='average',pct=True)
cr=c[targets].rank(method='average',pct=True)
h=0.6*rr+0.4*cr
rows=[]
for t in targets:
    ar=roc_auc_score(g[t],r[t]); ac=roc_auc_score(g[t],c[t]); ah=roc_auc_score(g[t],h[t])
    rows.append({'target':t,'auc_raptor_v5':ar,'auc_coat_top3':ac,'auc_hybrid_60_40':ah,'delta_hybrid_vs_raptor':ah-ar})
out=pd.DataFrame(rows)
out.to_csv(root/'experiments/exp013_raptor_v5_coat_inner_blend.csv',index=False)
print(out.to_string(index=False))
print('macro_raptor',out.auc_raptor_v5.mean())
print('macro_coat',out.auc_coat_top3.mean())
print('macro_hybrid',out.auc_hybrid_60_40.mean())
print('delta',out.auc_hybrid_60_40.mean()-out.auc_raptor_v5.mean())
# paired bootstrap study resampling
rng=np.random.default_rng(20260912); deltas=[]
y=g[targets].to_numpy(); a=r[targets].to_numpy(); b=h[targets].to_numpy(); n=len(g)
for _ in range(3000):
    ix=rng.integers(0,n,n); vals=[]
    for j in range(len(targets)):
        yy=y[ix,j]
        if np.unique(yy).size<2: continue
        vals.append(roc_auc_score(yy,b[ix,j])-roc_auc_score(yy,a[ix,j]))
    if len(vals)==len(targets): deltas.append(np.mean(vals))
d=np.asarray(deltas)
print('bootstrap',len(d),float(d.mean()),float(d.std()),float(np.quantile(d,.025)),float(np.quantile(d,.975)),float((d>0).mean()))
