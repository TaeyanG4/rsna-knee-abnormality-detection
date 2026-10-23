try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from pathlib import Path
root=workspace_root()
targets=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
g=pd.read_csv(root/'gold/gold_labels.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
files={
 'Pilkwang_DINO':root/'predictions/Pilkwang_public_OOF_gold58_public.csv',
 'CoAt_top3':root/'predictions/CoAt_top3_probability_gold58_public.csv',
 'Raptor_v5':root/'predictions/EXP006A_kaggle_v1/exp006a_original.csv',
}
p={k:pd.read_csv(v,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').reindex(g.index) for k,v in files.items()}
rows=[]
for t in targets:
    r={'target':t,'positives':int(g[t].sum())}
    for k in p:
        r[f'auc_{k}']=roc_auc_score(g[t],p[k][t])
    keys=list(p)
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            a,b=keys[i],keys[j]
            r[f'spearman_{a}__{b}']=spearmanr(p[a][t],p[b][t]).statistic
    rows.append(r)
out=pd.DataFrame(rows)
out.to_csv(root/'experiments/exp012_three_family_diversity.csv',index=False)
print(out.to_string(index=False))
print('\nMACRO AUC')
for k in p:
    print(k, np.mean([roc_auc_score(g[t],p[k][t]) for t in targets]))
