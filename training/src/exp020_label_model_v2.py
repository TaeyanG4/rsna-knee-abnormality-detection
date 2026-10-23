from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT=workspace_root()
import sys
sys.path.insert(0,str(ROOT/'src'))
from label_audit import load_teacher, TARGETS, UID

AUD=json.loads((ROOT/'reports'/'label_audit_receipt.json').read_text(encoding='utf-8'))['teachers']
# Independent-comparison eligible, full Gold58 coverage, no exact binary copy.
rows=[r for r in AUD if r.get('eligible_for_teacher_comparison') and r.get('full58_comparable') and not r.get('all_gold_exact_copy_flag')]
# Deduplicate identical prediction payloads (e.g. same labels under two filenames).
uniq={}
for r in sorted(rows,key=lambda x:(-(x.get('macro_auc') or -1),x['source'])):
    uniq.setdefault(r['prediction_sha256'],r)
rows=list(uniq.values())
# Keep strongest 8 unique sources to control variance on 58 studies.
rows=sorted(rows,key=lambda x:x.get('macro_auc') or -1,reverse=True)[:8]
print('sources')
for r in rows: print(r['source'],r.get('macro_auc'))

G=pd.read_csv(ROOT/'gold'/'gold_labels.csv',dtype={UID:str}).set_index(UID)
all_train=pd.read_csv(ROOT/'data'/'train.csv',dtype={UID:str}) if (ROOT/'data'/'train.csv').exists() else pd.read_csv(ROOT/'gold'/'gold_labels.csv',dtype={UID:str})[[UID]]
# use official train.csv from acquired public metadata if present
if len(all_train)<4407:
    for cand in [ROOT/'data'/'raw'/'train.csv',ROOT/'competition'/'train.csv',ROOT/'metadata'/'train.csv']:
        if cand.exists():
            all_train=pd.read_csv(cand,dtype={UID:str}); break
loaded=[]
for r in rows:
    df,_=load_teacher(ROOT/r['file'])
    loaded.append((r['source'],df.set_index(UID)[TARGETS]))

ids=G.index.astype(str).tolist()
Y=G[TARGETS].to_numpy(float)
X=np.stack([df.loc[ids].to_numpy(float) for _,df in loaded],axis=2) # N,T,S
if not np.isfinite(X).all(): raise RuntimeError('nonfinite gold teacher features')

Cs=[0.01,0.03,0.1,0.3,1.0,3.0]
oof=np.zeros_like(Y,dtype=float)
chosen=[]
for j,t in enumerate(TARGETS):
    y=Y[:,j].astype(int)
    x=X[:,j,:]
    per_c=[]
    for C in Cs:
        pred=np.zeros(len(y),float)
        # fixed 5-fold internal OOF for full-data C selection diagnostic
        sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260912+j)
        for tr,va in sk.split(x,y):
            m=make_pipeline(StandardScaler(),LogisticRegression(C=C,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced'))
            m.fit(x[tr],y[tr]); pred[va]=m.predict_proba(x[va])[:,1]
        per_c.append(roc_auc_score(y,pred))
    Cbest=Cs[int(np.argmax(per_c))]
    chosen.append(Cbest)
    # strict outer leave-one-out; C is deliberately fixed from full-gold CV to avoid 58x nested instability.
    # We report this as semi-nested, not untouched validation.
    for i in range(len(y)):
        tr=np.arange(len(y))!=i
        m=make_pipeline(StandardScaler(),LogisticRegression(C=Cbest,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced'))
        m.fit(x[tr],y[tr]); oof[i,j]=m.predict_proba(x[i:i+1])[:,1][0]

aucs=[roc_auc_score(Y[:,j],oof[:,j]) for j in range(len(TARGETS))]
macro=float(np.mean(aucs))
# teacher baselines on same 58
base=[]
for k,(name,_) in enumerate(loaded):
    av=[roc_auc_score(Y[:,j],X[:,j,k]) for j in range(len(TARGETS))]
    base.append({'source':name,'macro_auc':float(np.mean(av)),**dict(zip(TARGETS,map(float,av)))})

# fit final per-target models on all Gold and export 4349/available non-gold labels from union of teacher indices
common=set(loaded[0][1].index)
for _,df in loaded[1:]: common &= set(df.index)
common_ids=sorted(common)
features=np.stack([df.loc[common_ids].to_numpy(float) for _,df in loaded],axis=2)
final=np.zeros((len(common_ids),len(TARGETS)),float)
coefs={}
for j,t in enumerate(TARGETS):
    m=make_pipeline(StandardScaler(),LogisticRegression(C=chosen[j],penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced'))
    m.fit(X[:,j,:],Y[:,j].astype(int)); final[:,j]=m.predict_proba(features[:,j,:])[:,1]
    lr=m.named_steps['logisticregression']; coefs[t]={'C':chosen[j],'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0])}

out=pd.DataFrame(final,columns=TARGETS); out.insert(0,UID,common_ids)
# never overwrite official Gold in student weak targets
out=out[~out[UID].isin(set(ids))].reset_index(drop=True)
out_path=ROOT/'labels'/'derived_EXP020'/'label_model_v2_soft.csv'; out_path.parent.mkdir(exist_ok=True)
out.to_csv(out_path,index=False)
pd.DataFrame({UID:ids,**{TARGETS[j]:oof[:,j] for j in range(len(TARGETS))}}).to_csv(ROOT/'experiments'/'exp020_oof_gold.csv',index=False)
pd.DataFrame(base).to_csv(ROOT/'experiments'/'exp020_teacher_baselines.csv',index=False)
receipt={'experiment':'EXP020_LABEL_MODEL_V2','evaluation':'semi-nested Gold58 OOF: target-specific L2 logistic stack; C selected by 5-fold Gold CV then leave-one-out prediction','sources':[{'source':n,'sha':r['prediction_sha256']} for (n,_),r in zip(loaded,rows)],'gold_oof_macro_auc':macro,'gold_oof_per_target':dict(zip(TARGETS,map(float,aucs))),'best_single_macro':max(x['macro_auc'] for x in base),'best_single_source':max(base,key=lambda x:x['macro_auc'])['source'],'chosen_C':dict(zip(TARGETS,chosen)),'coefficients':coefs,'export_rows':len(out),'export_sha256':hashlib.sha256(out_path.read_bytes()).hexdigest(),'warnings':['Teacher sources may themselves have prior Gold exposure; OOF stacking does not undo source-selection leakage.','This evaluates report-teacher label quality, not MRI model AUC.','Use as training supervision candidate only; final value must be established by image-model CV/LB.']}
(ROOT/'reports'/'EXP020_label_model_v2.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='coefficients'},indent=2))
print(pd.DataFrame({'target':TARGETS,'oof_auc':aucs,'best_teacher_auc':[max(b[t] for b in base) for t in TARGETS]}).to_string(index=False))
