"""H224 cache portability v1.0.0. Reuse fullcache v1.1.0 renderer byte-for-byte.
Changelog: spawn-safe scratch relocation and per-study payload/block parity.
CPU only; B supplies the original competition ZIP at the documented path.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
from tools import l1_rco_fullcache_v110 as f
BASE=ROOT/'.local/scratch/astra-b2/h224-folds-v1'
WORK=BASE/'render';CACHE=BASE/'cache';HASHES=BASE/'local32.json'
_original_init=f.init_worker

def init_worker(reg,ids,run_slot):
    f.WORK=WORK;f.OUT=CACHE
    _original_init(reg,ids,run_slot)

def render(out=CACHE,limit=None,workers=6):
    f.WORK=WORK;f.OUT=CACHE;f.init_worker=init_worker
    WORK.mkdir(parents=True,exist_ok=True)
    reg=f.validate(f.read(f.REG));ids=reg['train']+reg['holdout']
    if limit:
        f.need(limit==2,'only two-study CPU smoke limit is allowed')
        ids=selected_ids(reg)[:2]
        f.need(Path(out).resolve()==(WORK/'smoke-cache').resolve(),'smoke output must stay in fixed scratch')
    return f.build(reg,Path(out),ids,workers=workers)

def selected_ids(reg):
    ids=reg['train']+reg['holdout']
    # Eight studies per hash fold; independent fixed salted ordering.
    return [u for fold in range(4) for u in sorted((u for u in ids if f.s.fold(u)==fold),
            key=lambda u:hashlib.sha256(('H224-parity-v1:'+u).encode()).hexdigest())[:8]]

def array_hash(array):
    h=hashlib.sha256()
    # Include dtype and shape and hash bounded slices, avoiding whole-study copies.
    h.update(json.dumps([array.dtype.str,list(array.shape)],separators=(',',':')).encode())
    for row in array:h.update(row.tobytes(order='C'))
    return h.hexdigest()

def study_hashes(cache,ids):
    import numpy as np
    index=f.read(Path(cache)/'index.json')
    px=np.load(Path(cache)/'px.u8.npy',mmap_mode='r');meta=np.load(Path(cache)/'meta.f32.npy',mmap_mode='r')
    result={}
    for uid in ids:
        offset,n,_=index['done'][uid];blocks=json.loads(json.dumps(index['blocks'][uid]))
        for b in blocks:b['views']=[[v,i-offset] for v,i in b['views']]
        result[uid]=dict(px=array_hash(px[offset:offset+n]),meta=array_hash(meta[offset:offset+n]),blocks=f.digest(blocks),rows=n)
    return result

def export_hashes(cache,output=HASHES):
    reg=f.validate(f.read(f.REG));ids=selected_ids(reg)
    from tools.l1_rco_k1_v1 import complete_cache
    complete_cache(dict(train=reg['train'],holdout=reg['holdout'],fullcache_contract_id=reg['contract_id']),cache)
    result=dict(version='1.0.0',contract_id=reg['contract_id'],ids=ids,studies=study_hashes(cache,ids),
                source_receipt_sha256=f.sha(Path(cache)/'receipt.json'),renderer_pins=reg['pins'])
    f.write(output,result);return dict(status='PASS',studies=32,sha256=f.sha(output))

def parity(cache,reference=HASHES,smoke=False):
    ref=f.read(reference);reg=f.validate(f.read(f.REG));ids=selected_ids(reg)
    f.need(ref['contract_id']==reg['contract_id'] and ref['ids']==ids and ref['renderer_pins']==reg['pins'],'parity reference identity')
    chosen=ids[:2] if smoke else ids
    actual=study_hashes(cache,chosen)
    f.need(actual=={u:ref['studies'][u] for u in chosen},'cache payload/meta/blocks parity mismatch; HOLD')
    result=dict(status='PASS',studies=len(chosen),reference_sha256=f.sha(reference),
                cache_receipt_sha256=f.sha(Path(cache)/'receipt.json'),contract_id=reg['contract_id'],smoke=smoke)
    f.write(Path(cache)/'H224-PARITY.json',result);return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['render','export-hashes','parity'])
    ap.add_argument('--cache',type=Path,default=CACHE);ap.add_argument('--reference',type=Path,default=HASHES)
    ap.add_argument('--workers',type=int,default=6);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.action=='render':r=render(WORK/'smoke-cache' if a.smoke else a.cache,2 if a.smoke else None,a.workers)
    elif a.action=='export-hashes':r=export_hashes(a.cache,a.reference)
    else:r=parity(a.cache,a.reference,a.smoke)
    # Never print study identities, metadata or labels.
    print(json.dumps({k:v for k,v in r.items() if k in ('status','studies','sha256','version','contract_id','cuda_initialized','elapsed_seconds','smoke')}))
if __name__=='__main__':main()
