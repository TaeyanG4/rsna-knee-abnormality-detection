try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
from pathlib import Path
import re, unicodedata, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT=workspace_root()
UID='StudyInstanceUID'
TARGET='Synovitis'
AUX=['Effusion','Medial OA','Lateral OA','PF OA',"Baker's"]
train=pd.read_csv(ROOT/'data'/'train.csv',dtype={UID:str})
flight=pd.read_csv(ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv',dtype={UID:str})
gold=pd.read_csv(ROOT/'gold'/'gold_labels.csv',dtype={UID:str})[[UID,TARGET]]

terms=['synovitis','synovial','sinovit','sinovyal','sinovite','sinovitis','synovit']
def norm(s):
    s='' if not isinstance(s,str) else s
    s=unicodedata.normalize('NFKD',s.lower())
    s=''.join(ch for ch in s if not unicodedata.combining(ch))
    return s

def mention(s):
    z=norm(s)
    return int(any(t in z for t in terms))

base=train[[UID,'Report']].merge(flight[[UID,TARGET]+AUX],on=UID,how='inner',validate='one_to_one')
base['mentioned']=base['Report'].map(mention).astype(int)
g=gold.merge(base,on=UID,how='left',validate='one_to_one')
y=g[TARGET+'_x'].to_numpy(int) if TARGET+'_x' in g.columns else g[TARGET].to_numpy(int)
# merged column handling
if TARGET+'_y' in g.columns:
    syn=g[TARGET+'_y'].to_numpy(float)
else:
    syn=g[TARGET].to_numpy(float)
X0=np.column_stack([syn, g[AUX].to_numpy(float), g['mentioned'].to_numpy(float)])
# add interactions: silent * auxiliaries and silent * base synovitis
silent=1-g['mentioned'].to_numpy(float)
X=np.column_stack([X0, silent[:,None]*g[AUX].to_numpy(float), silent*syn])

# LOO OOF with fixed strong regularization; feature count deliberately small
Cs=[0.01,0.03,0.1,0.3]
rows=[]
for C in Cs:
    p=np.zeros(len(g),float)
    for i in range(len(g)):
        tr=np.arange(len(g))!=i
        m=LogisticRegression(C=C,max_iter=2000,class_weight=None)
        m.fit(X[tr],y[tr])
        p[i]=m.predict_proba(X[i:i+1])[:,1][0]
    auc=roc_auc_score(y,p)
    rows.append((C,auc,p))
base_auc=roc_auc_score(y,syn)
best=max(rows,key=lambda x:x[1])
print('gold58 mention rate',g['mentioned'].mean(),'count',int(g['mentioned'].sum()))
print('flight synovitis auc',base_auc)
for C,auc,p in rows: print('C',C,'loo_auc',auc,'delta',auc-base_auc)
print('best',best[0],best[1])
# final model for non-gold export only if useful
res={'experiment':'EXP021_SYN_MISSINGNESS_CORRECTION','gold58_mentions':int(g['mentioned'].sum()),'gold58_mention_rate':float(g['mentioned'].mean()),'flight_auc':float(base_auc),'candidates':[{'C':float(C),'loo_auc':float(auc),'delta':float(auc-base_auc)} for C,auc,_ in rows]}
(ROOT/'reports'/'EXP021_synovitis_missingness.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
