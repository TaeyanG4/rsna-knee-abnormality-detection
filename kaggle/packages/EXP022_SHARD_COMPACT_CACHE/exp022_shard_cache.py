from __future__ import annotations
import hashlib, json, shutil, time
from pathlib import Path
import numpy as np

OUT = Path('/kaggle/working')
N_SHARDS = 8

def sha256(path: Path, chunk=32<<20):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def find_one(name: str) -> Path:
    hits=[p for p in Path('/kaggle/input').rglob(name) if p.is_file()]
    if len(hits)!=1:
        raise RuntimeError(f'expected one {name}, found {hits}')
    return hits[0]

def main():
    t0=time.time()
    src=find_one('train_cache_6x9x224.npy')
    uid_src=find_one('train_cache_uids.csv')
    mask_src=find_one('train_cache_masks.npy')
    manifest_src=find_one('train_cache_manifest.json')
    arr=np.load(src, mmap_mode='r')
    n=int(arr.shape[0])
    edges=np.linspace(0,n,N_SHARDS+1).round().astype(int)
    manifest={'contract':'rsna_knee_train_cache_6slot_9slice_224_shards_v1','source':str(src),'shape':list(arr.shape),'dtype':str(arr.dtype),'shards':[]}
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        path=OUT/f'train_cache_shard_{i:02d}_{a:04d}_{b:04d}.npy'
        chunk=np.asarray(arr[a:b])
        np.save(path,chunk,allow_pickle=False)
        rec={'index':i,'start':int(a),'stop':int(b),'shape':list(chunk.shape),'bytes':path.stat().st_size,'sha256':sha256(path),'file':path.name}
        manifest['shards'].append(rec)
        print(json.dumps(rec), flush=True)
        del chunk
    shutil.copy2(uid_src,OUT/'train_cache_uids.csv')
    shutil.copy2(mask_src,OUT/'train_cache_masks.npy')
    shutil.copy2(manifest_src,OUT/'source_train_cache_manifest.json')
    manifest['elapsed_seconds']=time.time()-t0
    (OUT/'train_cache_shards_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
