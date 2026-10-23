try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=workspace_root()
UID='StudyInstanceUID'
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
fold_dirs=[ROOT/'experiments'/'EXP043B_CONVNEXT_S_REG_FOLD0']+[ROOT/'experiments'/f'EXP044_CONVNEXT_S_FOLD{i}' for i in range(1,5)]
for d in fold_dirs:
    for f in ('summary.json','weak_val_pred.csv','gold_pred.csv'):
        p=d/f
        if not p.is_file(): raise FileNotFoundError(p)
folds=pd.read_csv(ROOT/'experiments'/'EXP026_GROUPED_FROZEN_PROBE'/'folds.csv',dtype={UID:str})
weak=pd.read_csv(ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv',dtype={UID:str}).set_index(UID)
parts=[]; summaries=[]; gold_parts=[]
for i,d in enumerate(fold_dirs):
    s=json.loads((d/'summary.json').read_text(encoding='utf-8')); summaries.append(s)
    p=pd.read_csv(d/'weak_val_pred.csv',dtype={UID:str}); p['fold']=i; parts.append(p)
    g=pd.read_csv(d/'gold_pred.csv',dtype={UID:str}).set_index(UID)[TARGETS]; gold_parts.append(g)
oof=pd.concat(parts,ignore_index=True)
if oof[UID].duplicated().any(): raise RuntimeError('duplicate OOF UID')
elig=folds[(folds.fold>=0)&(~folds.is_gold)&(~folds.quarantined_gold_report_duplicate)]
if set(oof[UID]) != set(elig[UID]): raise RuntimeError(f'OOF coverage mismatch pred={len(oof)} expected={len(elig)}')
oof=oof.set_index(UID).loc[elig[UID]].reset_index()
per=[]
for t in TARGETS:
    y=weak.loc[oof[UID],t].to_numpy(float); c=weak.loc[oof[UID],t+'__conf'].to_numpy(float); p=oof[t].to_numpy(float)
    use=((y>=.7)|(y<=.3))&(c>=.6); yy=(y[use]>=.7).astype(np.uint8)
    auc=float(roc_auc_score(yy,p[use]))
    per.append({'target':t,'auc':auc,'n':int(use.sum()),'pos':int(yy.sum())})
macro=float(np.mean([r['auc'] for r in per]))
# Gold ensemble diagnostic only; never used to select epochs.
off=pd.read_csv(ROOT/'data'/'train.csv',dtype={UID:str}).set_index(UID)
gold_uids=folds.loc[folds.is_gold,UID].astype(str).tolist()
G=np.mean(np.stack([g.loc[gold_uids,TARGETS].to_numpy(float) for g in gold_parts],axis=0),axis=0)
gy=off.loc[gold_uids,TARGETS].to_numpy(float)
gper=[]
for j,t in enumerate(TARGETS): gper.append({'target':t,'auc':float(roc_auc_score(gy[:,j],G[:,j]))})
gmacro=float(np.mean([r['auc'] for r in gper]))
out=ROOT/'experiments'/'EXP044_CONVNEXT_S_5FOLD'; out.mkdir(parents=True,exist_ok=True)
oof.to_csv(out/'oof_pred.csv',index=False)
pd.DataFrame(per).to_csv(out/'weak_oof_per_target.csv',index=False)
pd.DataFrame(G,columns=TARGETS).assign(**{UID:gold_uids}).to_csv(out/'gold_ensemble_pred.csv',index=False)
pd.DataFrame(gper).to_csv(out/'gold_per_target.csv',index=False)
receipt={'contract':'exp044_convnext_s_adjacent_288_5fold_v1','fold_best_epochs':[int(s['best_epoch']) for s in summaries],'fold_best_weak_auc':[float(s['best_weak_highconf_auc']) for s in summaries],'weak_oof_macro_auc':macro,'weak_oof_rows':len(oof),'gold_5fold_mean_macro_auc_diagnostic_only':gmacro,'gold_not_used_for_model_selection':True}
(out/'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2),flush=True)
