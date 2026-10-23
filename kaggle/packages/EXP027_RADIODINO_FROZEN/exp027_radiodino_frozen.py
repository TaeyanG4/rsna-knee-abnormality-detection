"""Train-only feature extraction + fixed grouped linear probe; NOT a submission.
Only a pinned public safetensors file is fetched. No report/image is transmitted.
"""
from pathlib import Path
import hashlib,json,time,unicodedata,os
import numpy as np
import pandas as pd
import requests
import torch
import timm
from safetensors.torch import load_file
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']; UID='StudyInstanceUID'
REV='773d28d873455153495d39179fc42ea816aa1212'; HASH='6f88c66e9155a5d223d8d08a78aebc294e76deede181c4eb8b1895a70af49a01'
OUT=Path('/kaggle/working'); CENTERS=[2,4,6]; SEED=20260913; NF=5; ALPHA=100.0

def unique_file(pattern):
    candidates=list(Path('/kaggle/input').rglob(pattern))
    if len(candidates)!=1: raise RuntimeError(f'Expected one {pattern}, found {candidates}')
    return candidates[0]

def hashfile(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def report_key(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',str(text)).lower().split()).encode()).hexdigest()

def per_target(y,p,conf=None):
    rows=[]
    for j,t in enumerate(T):
        use=np.ones(len(y),bool) if conf is None else (((y[:,j]>=.7)|(y[:,j]<=.3))&(conf[:,j]>=.6))
        yy=y[use,j] if conf is None else (y[use,j]>=.7).astype(int)
        auc=float(roc_auc_score(yy,p[use,j])) if len(np.unique(yy))==2 else None
        rows.append({'target':t,'auc':auc,'n':int(use.sum()),'positive_n':int((yy==1).sum())})
    return {'macro_auc':float(np.mean([r['auc'] for r in rows if r['auc'] is not None])),'defined_targets':sum(r['auc'] is not None for r in rows),'per_target':rows}

def main():
    started=time.time(); OUT.mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('GPU requested but unavailable')
    probe=torch.ones(2,device='cuda'); assert float((probe+1).sum())==4
    torch.manual_seed(SEED); torch.set_num_threads(4)
    cache=unique_file('train_cache_shards_manifest.json').parent
    manifest=json.loads((cache/'train_cache_shards_manifest.json').read_text())
    ids=pd.read_csv(cache/'train_cache_uids.csv',dtype={UID:str})[UID].tolist(); n=len(ids)
    masks=np.load(cache/'train_cache_masks.npy'); assert masks.shape==(n,6,9)
    trainfile=unique_file('train.csv'); train=pd.read_csv(trainfile,dtype={UID:str}).set_index(UID).loc[ids]
    teacherfile=unique_file('report_labels_v4hybrid.csv'); teacher=pd.read_csv(teacherfile,dtype={UID:str}).set_index(UID).loc[ids]
    weight=OUT/'radiodino_s16.safetensors'
    url=f'https://huggingface.co/Snarcy/RadioDino-s16/resolve/{REV}/model.safetensors'
    with requests.get(url,stream=True,timeout=(20,90)) as r:
        r.raise_for_status()
        with weight.open('wb') as f:
            for b in r.iter_content(2**20):
                if b:f.write(b)
    assert weight.stat().st_size==86676336 and hashfile(weight)==HASH
    model=timm.create_model('vit_small_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict(load_file(str(weight)),strict=True); model.eval().cuda()
    features=np.lib.format.open_memmap(OUT/'features.npy',mode='w+',dtype=np.float16,shape=(n,6,3,384)); features[:]=0
    valid=np.zeros((n,6,3),np.uint8)
    mean=torch.tensor([.485,.456,.406],device='cuda').reshape(1,3,1,1); std=torch.tensor([.229,.224,.225],device='cuda').reshape(1,3,1,1)
    imgs=[]; keys=[]
    def flush():
        if not imgs:return
        x=torch.from_numpy(np.stack(imgs)).cuda().float()/255; x=(x-mean)/std
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            z=model(x).float().cpu().numpy()
        assert np.isfinite(z).all()
        for k,v in zip(keys,z):features[k]=v; valid[k]=1
        imgs.clear(); keys.clear()
    for entry in manifest['shards']:
        sh=np.load(cache/entry['file'],mmap_mode='r')
        for j in range(len(sh)):
            i=entry['start']+j
            for s in range(6):
                for ci,c in enumerate(CENTERS):
                    if masks[i,s,c-1:c+2].min()==0:continue
                    imgs.append(np.asarray(sh[j,s,c-1:c+2])); keys.append((i,s,ci))
                    if len(imgs)==64:flush()
            if (i+1)%100==0:
                flush(); features.flush(); print(json.dumps({'stage':'extract','studies':i+1,'n':n,'seconds':time.time()-started}),flush=True)
        del sh
    flush(); features.flush(); np.save(OUT/'valid.npy',valid)
    # Fixed five-fold report-group split identical to EXP026. No gold selection.
    gold=train[T].notna().all(axis=1).to_numpy(); groups=train.Report.map(report_key).to_numpy(); gset=set(groups[gold])
    quarantine=np.array([g in gset for g in groups])&~gold; eligible=~gold&~quarantine
    rng=np.random.default_rng(SEED); sizes=pd.Series(groups[eligible]).value_counts(); order=list(sizes.index); rng.shuffle(order); order.sort(key=lambda g:-sizes[g])
    loads=np.zeros(NF,int); mapping={}
    for g in order:
        f=int(loads.argmin()); mapping[g]=f; loads[f]+=int(sizes[g])
    folds=np.array([mapping[g] if e else -1 for g,e in zip(groups,eligible)])
    pd.DataFrame({UID:ids,'fold':folds,'gold':gold,'quarantine':quarantine,'report_group':groups}).to_csv(OUT/'folds.csv',index=False)
    ft=np.asarray(features,dtype=np.float32); m=valid.astype(bool); ft=np.where(m[...,None],ft,0)
    X=(ft.sum(2)/np.maximum(m.sum(2)[...,None],1)).reshape(n,-1)
    y=teacher[T].to_numpy(np.float32); conf=teacher[[t+'__conf' for t in T]].to_numpy(np.float32)
    oof=np.full_like(y,np.nan); gp=[]
    with threadpool_limits(limits=4):
        for f in range(NF):
            a=np.where(eligible&(folds!=f))[0]; b=np.where(eligible&(folds==f))[0]; gi=np.where(gold)[0]
            assert not set(groups[a])&set(groups[b]) and not set(groups[a])&gset
            scaler=StandardScaler().fit(X[a]); head=Ridge(alpha=ALPHA,solver='cholesky').fit(scaler.transform(X[a]),y[a])
            oof[b]=head.predict(scaler.transform(X[b])); gp.append(head.predict(scaler.transform(X[gi])))
    pred=np.mean(gp,axis=0)
    pd.DataFrame(pred,columns=T).assign(StudyInstanceUID=np.array(ids)[gold]).to_csv(OUT/'gold_predictions.csv',index=False)
    pd.DataFrame(oof[eligible],columns=T).assign(StudyInstanceUID=np.array(ids)[eligible]).to_csv(OUT/'weak_oof.csv',index=False)
    receipt={'experiment':'EXP027_RADIODINO_FROZEN','not_submission':True,'model':'RadioDino-s16','revision':REV,'weights_sha256':HASH,'declared_license':'CC-BY-4.0; upstream/competition release audit still required','source_cache_manifest_sha256':hashfile(cache/'train_cache_shards_manifest.json'),'label_sha256':hashfile(teacherfile),'split_seed':SEED,'fold_sizes':loads.tolist(),'gold_used_for_selection':False,'gold_evaluation_count':1,'sampling':'cache-adjacent triplets, not native adjacency; identical to EXP024 DINO feature inputs','weak_oof':per_target(y[eligible],oof[eligible],conf[eligible]),'gold':per_target(train[T].to_numpy(float)[gold],pred),'elapsed_seconds':time.time()-started,'gpu':torch.cuda.get_device_name(),'peak_vram_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,'timm':timm.__version__,'limitations':['Frozen pooled linear probe only; not an estimate of fine-tuned upper bound.','Different natural and medical backbone families; not isolated pretraining ablation.','Report-group isolation is not verified patient/site isolation.','Report teachers have imperfect provenance and weak OOF is only a surrogate.']}
    (OUT/'result.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()
