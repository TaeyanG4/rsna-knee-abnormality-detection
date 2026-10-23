"""Training-only, hash-bound public CoAtNet branch reproduction.

This does not reproduce the complete public ensemble and is not a submission.
Public image preparation and model arithmetic are unchanged. Our orchestration
records every preparation error and never silently substitutes 0.5 predictions.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd

TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
UID='StudyInstanceUID'
OUT=Path('/kaggle/working')
MANIFEST='coat_resgated_ep10_top3_manifest.json'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()

def inputs():
    candidates=list(Path('/kaggle/input').glob('**/'+MANIFEST))
    if len(candidates)!=1:raise RuntimeError('Expected exactly one public CoAt package')
    package=candidates[0].parent
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    root=next((p for p in roots if (p/'train.csv').exists()),None)
    if root is None:raise FileNotFoundError('Official training data absent')
    gold=pd.read_csv(root/'train.csv',usecols=[UID]+TARGETS,dtype={UID:str})
    gold=gold.loc[gold[TARGETS].notna().all(1)].sort_values(UID).reset_index(drop=True)
    if len(gold)!=58 or gold[UID].duplicated().any():raise ValueError('Gold58 contract changed')
    return root,package,gold

def validate_package(package):
    manifest=json.loads((package/MANIFEST).read_text())
    bound=manifest['files']
    expected={
       'coat_resgated_gold_e04.pt':'cefaf007415b5e45ce2a720d5e211a183a4e90f160710ba1f72b1ce7c46239a2',
       'coat_resgated_gold_e06.pt':'56eca1f61651362ecfae8aecdd9327125713b95653344b8ad62ef8cff17d08f4',
       'coat_resgated_gold_e08.pt':'8d3159102046c127dc0d084fb69c9304ffcbd4b98d1333f9811d4d613fea1b12',
       'coatnet_resgated_ep10_top3_inference.py':'b11e58f8d7cabe9e264ac70b01e811dc6a846aa01248ace1f0e6536d0d4094d9',
       'coatnet_residual_gated_k32.py':'83735f535235bfd575767b9ac9b12ca0a77a6b11adf14343ca5c2466257f3374',
       'coatnet_q2l_bagmatch.py':'a3303a724e95cf3d9f13b67ef52529adf854b0cbe580b896ef42382cc740d932',
       'cnx_dicom_geometry.py':'1e6e46ca40828aec8e68d93dea3676c68ee4803a10b349ad092861006e8301d4',
       'raptor_light_inference.py':'3ce34aba32b44509bda7fe37a9b9b62c68028ea9a4d1364a5b4db3d97640b318',
       'raptor_light224.py':'d8f533df336f097910cd8c9ed7ae2248783554c9c797b661ee89406b5041cde5',
       'raptor_k6_surgical.py':'cc525db3bb67f0fde9db8dcfbbd33ff5bf5200d0c3c92251b368101314ae647f',
       'smart_crop224.py':'31b66d1a687fb9de3d01c10f31baf8b728903b6ed9e9effedec6fc8e77c4e424',
       'triplet_physical_crop.py':'b84826fede9fa4195774e208e829ad60f189cfee90dce8a2c9528349b18717cf'}
    for name,digest in expected.items():
        if bound.get(name)!=digest or sha(package/name)!=digest:raise RuntimeError('Pinned artifact drift: '+name)
    return manifest

def worker(start,stop):
    started=time.monotonic();root,package,gold=inputs();validate_package(package)
    sys.path.insert(0,str(package))
    import cv2
    import torch
    import coatnet_resgated_ep10_top3_inference as public
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True
    manifest,_=public.validate_artifact_root(package,verify_hashes=True)
    gold=gold.iloc[start:stop].copy();uids=gold[UID].tolist()
    table=pd.read_csv(root/'train_series.csv',dtype={UID:str,'SeriesInstanceUID':str})
    records={uid:frame.to_dict('records') for uid,frame in table.loc[table[UID].isin(uids)].groupby(UID,sort=False)}
    tasks=[(i,uid,records[uid],root/'train_series') for i,uid in enumerate(uids)]
    device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats()
    models=[public.load_checkpoint_model(package,i,device,manifest=manifest)[0] for i in range(3)]
    raw=np.full((3,len(uids),12),np.nan,dtype=np.float32);windows=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0,len(tasks),2):
            prepared=list(pool.map(public._prepare_bag_task,tasks[offset:offset+2]))
            bags=[bag for _,bag in prepared]
            for local,bag in prepared:
                windows.append({'uid':uids[local],'windows':len(bag.slots),'slot_counts':np.bincount(bag.slots,minlength=7)[1:].tolist()})
                if bag.errors:errors.append({'uid':uids[local],'errors':list(bag.errors)})
            values=public.infer_top3_microbatch(models,bags,device,backbone_micro_images=8)
            for pos,(local,bag) in enumerate(prepared):raw[:,local,:]=values[:,pos,:]
            print(f'gold shard {start}:{stop} completed {min(offset+2,len(tasks))}/{len(tasks)} in {time.monotonic()-started:.1f}s',flush=True)
    if not np.isfinite(raw).all() or ((raw<0)|(raw>1)).any():raise ValueError('Invalid predictions')
    np.savez_compressed(OUT/f'gold_shard_{start}_{stop}.npz',uids=np.asarray(uids),raw=raw)
    receipt={'start':start,'stop':stop,'studies':len(uids),'gpu':torch.cuda.get_device_name(0),
        'runtime_sec':time.monotonic()-started,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'opencv':cv2.__version__,
        'packages':{name:importlib.metadata.version(name) for name in ['torch','timm','numpy','pandas','pydicom']},
        'windows':windows,'preparation_errors':errors,'fallback_predictions':0,
        'scheduler':'two CPU threads, two studies per batch; unchanged public prepare/infer functions',
        'public_source_sha256':sha(package/'coatnet_resgated_ep10_top3_inference.py')}
    (OUT/f'gold_shard_{start}_{stop}.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=2,type=int)
    args=parser.parse_args()
    if args.worker is not None:return worker(*args.worker)
    started=time.monotonic();root,package,gold=inputs();manifest=validate_package(package)
    import torch
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    count=torch.cuda.device_count()
    if count<1:raise RuntimeError('Kaggle accelerator unavailable')
    nworkers=min(count,2);bounds=np.linspace(0,58,nworkers+1,dtype=int);processes=[]
    for i in range(nworkers):
        a,b=int(bounds[i]),int(bounds[i+1]);env=dict(os.environ)
        env['CUDA_VISIBLE_DEVICES']=str(i);env['PYTHONUNBUFFERED']='1'
        logfile=(OUT/f'gold_worker_{i}.log').open('w')
        proc=subprocess.Popen([sys.executable,__file__,'--worker',str(a),str(b)],env=env,stdout=logfile,stderr=subprocess.STDOUT)
        processes.append((a,b,proc,logfile))
    failure=[]
    for a,b,proc,logfile in processes:
        try:rc=proc.wait(timeout=max(1,1700-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            proc.terminate();proc.wait(timeout=20);rc=-1
        logfile.close()
        if rc:failure.append({'start':a,'stop':b,'returncode':rc})
    if failure:
        (OUT/'component_failure.json').write_text(json.dumps(failure,indent=2));raise RuntimeError(str(failure))
    raw=np.empty((3,58,12),np.float32);shards=[]
    for a,b,_,_ in processes:
        with np.load(OUT/f'gold_shard_{a}_{b}.npz',allow_pickle=False) as z:
            if z['uids'].tolist()!=gold[UID].iloc[a:b].tolist():raise ValueError('Shard UID mismatch')
            raw[:,a:b,:]=z['raw']
        shards.append(json.loads((OUT/f'gold_shard_{a}_{b}.json').read_text()))
    with np.load(package/'gold58_top3_predictions.npz',allow_pickle=False) as z:
        order=z['study_uids'].astype(str).tolist();indices=[order.index(x) for x in gold[UID]]
        reference=z['raw_probabilities'][:,indices,:]
        np.testing.assert_array_equal(z['truth'][indices],gold[TARGETS].to_numpy())
    ranks=rankdata(raw.astype(np.float64),axis=1,method='average')/58
    rank_mean=ranks.mean(0);prob_mean=raw.astype(np.float64).mean(0);y=gold[TARGETS].to_numpy()
    per=roc_auc_score(y,rank_mean,average=None);delta=np.abs(raw.astype(float)-reference.astype(float))
    error_count=sum(len(s['preparation_errors']) for s in shards)
    status='EXECUTED_TRAIN_GOLD_COMPONENT_ONLY'
    if error_count:status='EXECUTED_WITH_PREPARATION_ERRORS_NOT_ACCEPTED'
    receipt={'status':status,'utc':datetime.now(timezone.utc).isoformat(),'study_count':58,
        'full_public_pipeline_reproduced':False,'competition_submission':False,
        'gold_selection_exposed':True,'runtime_sec':time.monotonic()-started,
        'rank_macro_auc':float(per.mean()),'probability_macro_auc':float(roc_auc_score(y,prob_mean,average='macro')),
        'per_target_rank_auc':dict(zip(TARGETS,per.tolist())),
        'raw_probability_delta_saved_bf16':{'max':float(delta.max()),'mean':float(delta.mean())},
        'rank_delta_saved_bf16':float(per.mean()-manifest['selection']['gold58_rank_ensemble_auc']),
        'preparation_error_studies':error_count,'shards':shards,
        'limitations':['Gold-selected checkpoints; not independent validation','Our scheduler differs, public model and pixels unchanged','58 studies is not hidden-test-size simulation','No complete public ensemble or Public LB reproduction']}
    np.savez_compressed(OUT/'coat_gold58_reproduced.npz',uids=gold[UID].to_numpy(dtype=str),raw=raw,rank_mean=rank_mean,prob_mean=prob_mean,truth=y)
    pd.DataFrame(rank_mean,columns=TARGETS,index=gold[UID]).to_csv(OUT/'coat_gold58_reproduced.csv')
    (OUT/'coat_gold58_reproduction_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
