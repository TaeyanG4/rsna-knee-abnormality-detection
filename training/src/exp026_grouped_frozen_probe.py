"""Reproducible CPU diagnostics: no training on gold, no gold checkpoint selection."""
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
from pathlib import Path
import hashlib,json,unicodedata,time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=workspace_root()
OUT=ROOT/'experiments'/'EXP026_GROUPED_FROZEN_PROBE'; OUT.mkdir(exist_ok=True)
T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
UID='StudyInstanceUID'; SEED=20260913; ALPHA=100.0; NF=5

def key(text):
    t=' '.join(unicodedata.normalize('NFKC',str(text)).lower().split())
    return hashlib.sha256(t.encode('utf-8')).hexdigest()

def metrics(y,p,conf=None):
    out=[]
    for j,t in enumerate(T):
        use=np.ones(len(y),bool) if conf is None else (((y[:,j]>=.7)|(y[:,j]<=.3))&(conf[:,j]>=.6))
        yy=y[use,j] if conf is None else (y[use,j]>=.7).astype(int)
        a=float(roc_auc_score(yy,p[use,j])) if len(np.unique(yy))==2 else None
        pp=np.clip(p[:,j],1e-6,1-1e-6)
        out.append({'target':t,'auc':a,'auc_n':int(use.sum()),'auc_positive_n':int((yy==1).sum()),'soft_mse':float(np.mean((p[:,j]-y[:,j])**2)),'soft_bce_clipped':float(np.mean(-y[:,j]*np.log(pp)-(1-y[:,j])*np.log1p(-pp)))})
    return {'macro_auc':float(np.mean([r['auc'] for r in out if r['auc'] is not None])),'defined_targets':sum(r['auc'] is not None for r in out),'per_target':out}

def main():
    t0=time.time(); cache=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'
    ids=pd.read_csv(cache/'train_cache_uids.csv',dtype={UID:str})[UID].tolist()
    tr=pd.read_csv(ROOT/'data'/'train.csv',dtype={UID:str}).set_index(UID).loc[ids]
    teacher=pd.read_csv(ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv',dtype={UID:str}).set_index(UID).loc[ids]
    y=teacher[T].to_numpy(np.float32); conf=teacher[[t+'__conf' for t in T]].to_numpy(np.float32)
    gold=tr[T].notna().all(axis=1).to_numpy(); groups=tr.Report.map(key).to_numpy(); gset=set(groups[gold])
    quarantine=np.array([g in gset for g in groups])&~gold; eligible=~gold&~quarantine
    rg=np.random.default_rng(SEED); sizes=pd.Series(groups[eligible]).value_counts(); order=list(sizes.index); rg.shuffle(order); order.sort(key=lambda g:-sizes[g])
    loads=np.zeros(NF,int); mapping={}
    for g in order:
        f=int(np.argmin(loads)); mapping[g]=f; loads[f]+=int(sizes[g])
    folds=np.array([mapping[g] if ok else -1 for g,ok in zip(groups,eligible)])
    table=pd.DataFrame({UID:ids,'index':range(len(ids)),'report_group_sha256':groups,'is_gold':gold,'quarantined_gold_report_duplicate':quarantine,'fold':folds})
    table.to_csv(OUT/'folds.csv',index=False)
    config={'seed':SEED,'folds':NF,'head':'Ridge alpha=100 fixed; per-fold train-only StandardScaler','gold_used_for_selection':False,'raw_reports_used_as_prediction_input':False,'split':'normalized report hash grouped; gold-report duplicate non-gold rows excluded','eligible':int(eligible.sum()),'quarantine':int(quarantine.sum()),'fold_sizes':loads.tolist(),'limits':['Report equality is not verified patient identity.','Weak AUC is high-confidence teacher surrogate, not gold AUC.','External frozen feature provenance is required; full-train supervised Raptor features are excluded.','Sparse 9-slice cache limits representation; this is not an upper bound on DINO.']}
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    froot=ROOT/'experiments'/'exp024_dinov2_features'; ft=np.load(froot/'features.npy').astype(np.float32); mask=np.load(froot/'valid.npy').astype(bool)
    assert ft.shape[:3]==mask.shape and len(ft)==len(ids) and np.isfinite(ft).all()
    ft=np.where(mask[...,None],ft,0); pooled=ft.sum(2)/np.maximum(mask.sum(2)[...,None],1)
    features={'dinov2_s_external_frozen':pooled.reshape(len(ids),-1),'sequence_mask_only_negative_control':mask.any(2).astype(np.float32)}
    result={}
    with threadpool_limits(limits=6):
        for name,X in features.items():
            pred=np.full_like(y,np.nan); gold_predictions=[]
            for f in range(NF):
                a=np.where(eligible&(folds!=f))[0]; b=np.where(eligible&(folds==f))[0]; gi=np.where(gold)[0]
                assert not (set(groups[a])&set(groups[b])) and not (set(groups[a])&gset)
                scaler=StandardScaler().fit(X[a]); model=Ridge(alpha=ALPHA,solver='cholesky').fit(scaler.transform(X[a]),y[a])
                pred[b]=model.predict(scaler.transform(X[b])); gold_predictions.append(model.predict(scaler.transform(X[gi])))
                print(name,'fold',f,'train',len(a),'validation',len(b),flush=True)
            gp=np.mean(gold_predictions,axis=0); gy=tr.loc[np.array(ids)[gold],T].to_numpy(float)
            result[name]={'weak_oof':metrics(y[eligible],pred[eligible],conf[eligible]),'gold_once_after_fit':metrics(gy,gp),'gold_selection':False}
            pd.DataFrame(pred[eligible],columns=T).assign(StudyInstanceUID=np.array(ids)[eligible]).to_csv(OUT/(name+'_weak_oof.csv'),index=False)
            pd.DataFrame(gp,columns=T).assign(StudyInstanceUID=np.array(ids)[gold]).to_csv(OUT/(name+'_gold.csv'),index=False)
    result['runtime_sec']=time.time()-t0
    (OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'config':config,'result':{k:{'weak_surrogate_auc':v['weak_oof']['macro_auc'],'gold_auc':v['gold_once_after_fit']['macro_auc']} for k,v in result.items() if isinstance(v,dict)},'seconds':result['runtime_sec']},indent=2),flush=True)
if __name__=='__main__':main()
