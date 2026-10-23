try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
from pathlib import Path
import json, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
ROOT=workspace_root()
T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'; SPLIT=ROOT/'experiments'/'EXP026_GROUPED_FROZEN_PROBE'/'folds.csv'; WEIGHT=ROOT/'weights'/'dreaddevelopment__raptor-knee-maxspan'/'raptor_ft_coatnet_v5_full_swa.pt'; OUT=ROOT/'experiments'/'EXP030_RAPTOR_CANONICAL_HEAD_PSEUDOOOF'; OUT.mkdir(parents=True,exist_ok=True)
VAR={'noncanonical':ROOT/'experiments'/'exp025_raptor_features','canonical':ROOT/'experiments'/'EXP028_RAPTOR_CANONICAL_FEATURE_HEAD'}; SEED=20260913
class Head(nn.Module):
    def __init__(self,Fdim,n=12):
        super().__init__();self.norm=nn.LayerNorm(Fdim);self.att=nn.Sequential(nn.Linear(Fdim,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,n));self.clsW=nn.Parameter(torch.zeros(n,Fdim));self.clsb=nn.Parameter(torch.zeros(n))
    def forward(self,f,m):
        h=self.norm(f);s=self.att(h).masked_fill(m[...,None]<=0,-1e4);a=torch.softmax(s,1);pool=torch.einsum('bkn,bkf->bnf',a,h);return (pool*self.clsW).sum(-1)+self.clsb
def metrics(y,p,conf):
    out=[]
    for j,t in enumerate(T):
        use=((y[:,j]>=.7)|(y[:,j]<=.3))&(conf[:,j]>=.6); yy=(y[use,j]>=.7).astype(int); a=float(roc_auc_score(yy,p[use,j])) if np.unique(yy).size==2 else None
        out.append({'target':t,'auc':a,'n':int(use.sum()),'positive_n':int(yy.sum())})
    return {'macro_auc':float(np.mean([r['auc'] for r in out if r['auc'] is not None])),'per_target':out}
def main():
    ids=pd.read_csv(CACHE/'train_cache_uids.csv').StudyInstanceUID.astype(str).tolist(); foldtab=pd.read_csv(SPLIT).set_index('StudyInstanceUID').loc[ids]; folds=foldtab.fold.to_numpy(int); eligible=folds>=0; gold=foldtab.is_gold.to_numpy(bool)
    teacher=pd.read_csv(ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv').set_index('StudyInstanceUID').loc[ids]; y=teacher[T].to_numpy(np.float32); conf=teacher[[t+'__conf' for t in T]].to_numpy(np.float32)
    trn=pd.read_csv(ROOT/'data'/'train.csv').set_index('StudyInstanceUID').loc[ids]; gy=trn.loc[np.array(ids)[gold],T].to_numpy(float); gi=np.where(gold)[0]
    ck=torch.load(WEIGHT,map_location='cpu',weights_only=False); st=ck['model']; Fdim=int(st['norm.weight'].numel()); init={k:v for k,v in st.items() if not k.startswith('backbone.')}; assert set(init)==set(Head(Fdim).state_dict())
    results={}; saved={}
    for name,root in VAR.items():
        feat=np.load(root/'features_f16.npy',mmap_mode='r'); mask=np.load(root/'window_mask.npy').astype(np.float32); oof=np.full((len(ids),12),np.nan,np.float32); gps=[]
        for f in range(5):
            torch.manual_seed(SEED+f); rng=np.random.default_rng(SEED+f); model=Head(Fdim).cuda(); model.load_state_dict(init,strict=True); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3); a=np.where(eligible&(folds!=f))[0]; b=np.where(eligible&(folds==f))[0]
            for ep in range(10):
                order=rng.permutation(a);model.train()
                for s in range(0,len(order),64):
                    ii=order[s:s+64];x=torch.from_numpy(np.asarray(feat[ii],np.float32)).cuda();m=torch.from_numpy(mask[ii]).cuda();yy=torch.from_numpy(y[ii]).cuda();ww=torch.from_numpy(conf[ii]).cuda();opt.zero_grad(set_to_none=True);log=model(x,m);loss=(F.binary_cross_entropy_with_logits(log,yy,reduction='none')*(.25+.75*ww)).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            def pred(ii):
                model.eval();oo=[]
                with torch.no_grad():
                    for s in range(0,len(ii),64):
                        jj=ii[s:s+64];x=torch.from_numpy(np.asarray(feat[jj],np.float32)).cuda();m=torch.from_numpy(mask[jj]).cuda();oo.append(torch.sigmoid(model(x,m)).cpu().numpy())
                return np.concatenate(oo)
            oof[b]=pred(b);gps.append(pred(gi));print(name,'fold',f,'done',len(a),len(b),flush=True)
            del model,opt;torch.cuda.empty_cache()
        met=metrics(y[eligible],oof[eligible],conf[eligible]); gp=np.mean(gps,axis=0); ga=np.array([roc_auc_score(gy[:,j],gp[:,j]) for j in range(12)]); results[name]={'weak_head_pseudo_oof':met,'gold_once_after_fit_macro':float(ga.mean()),'gold_per_target':dict(zip(T,map(float,ga)))};saved[name]=oof
        pd.DataFrame(oof[eligible],columns=T).assign(StudyInstanceUID=np.array(ids)[eligible]).to_csv(OUT/f'{name}_weak_oof.csv',index=False)
    # paired target deltas and study bootstrap on high-confidence weak surrogate
    per={}; old=results['noncanonical']['weak_head_pseudo_oof']['per_target']; new=results['canonical']['weak_head_pseudo_oof']['per_target']
    for j,t in enumerate(T):per[t]={'old':old[j]['auc'],'new':new[j]['auc'],'delta':new[j]['auc']-old[j]['auc'],'n':new[j]['n']}
    # target-macro bootstrap resampling eligible studies
    idx=np.where(eligible)[0];rng=np.random.default_rng(3001);ds=[]
    for _ in range(3000):
        ss=rng.choice(idx,len(idx),replace=True);dd=[]
        for j in range(12):
            use=((y[ss,j]>=.7)|(y[ss,j]<=.3))&(conf[ss,j]>=.6);yy=(y[ss[use],j]>=.7).astype(int)
            if np.unique(yy).size<2:continue
            dd.append(roc_auc_score(yy,saved['canonical'][ss[use],j])-roc_auc_score(yy,saved['noncanonical'][ss[use],j]))
        if dd:ds.append(np.mean(dd))
    ds=np.asarray(ds);results['paired']={'delta_macro':results['canonical']['weak_head_pseudo_oof']['macro_auc']-results['noncanonical']['weak_head_pseudo_oof']['macro_auc'],'per_target':per,'bootstrap_mean':float(ds.mean()),'bootstrap_ci95':[float(np.quantile(ds,.025)),float(np.quantile(ds,.975))],'positive_probability':float((ds>0).mean()),'note':'Head-only pseudo-OOF; backbone was trained on full weak train, so absolute weak OOF is leaky. Use paired orientation delta only.'}
    (OUT/'result.json').write_text(json.dumps(results,indent=2));print(json.dumps(results['paired'],indent=2),flush=True)
if __name__=='__main__':main()