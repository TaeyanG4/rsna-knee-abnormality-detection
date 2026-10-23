"""H-B-155 cache orchestration v2.1.1: bounded chunks, resume and fail-closed QA.

Changelog: one to four CPU workers, single-threaded decoders, immutable identity,
atomic per-chunk commits, deterministic 24-study montages, registered QA limits.
2.1.1: cap crop width to actual FOV after v2.1.0 small-FOV padding QA HOLD.
2.1.0: crop geometry 1.2.0, unchanged numeric gates, registered 64-study QA;
engineering mode cannot waive geometry failures. New versioned output only.
No GPU, network, labels or radiology report access.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import time

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import numpy as np
from tools import hb155_crop_v1 as c
from tools.hb150_io_v1 import atomic_json, safe_replace

VERSION='2.1.1'
DEFAULT_OUT=c.QA_WORK/'crop-cache-v211'
SAMPLE_OUT=c.QA_WORK/'geometry-qa64-v211'
SOURCE_SHA='f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
QA_THRESHOLDS=dict(empty_studies=0,decode_failures=0,laterality_conflicts=0,
    geometry_failures=0,max_empty_slice_fraction=.05,max_padded_slice_fraction=.01,
    max_padding_fraction=.25,max_clamped_slice_fraction=.10,
    max_unknown_laterality_fraction=1.0,random_overlay_studies=24,overlay_seed=155)
_RAW=None
_SPEC=None

def need(ok,message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def register():
    result=dict(schema='hb155-geometry-registration/v1',version=VERSION,
        thresholds=QA_THRESHOLDS,geometry=c.GEOMETRY_POLICY,
        sample='first 64 source positions; includes all original 24 QA cases',
        sample_size=64,source_manifest_sha256=SOURCE_SHA,
        geometry_sha256=c.sha(c.__file__),builder_sha256=c.sha(__file__),
        exposure='EXPOSED',clean_oof=False)
    path=c.QA_WORK/'geometry-registration-v211.json'
    if path.exists(): need(read(path)==result,'geometry registration changed')
    else: atomic_json(path,result)
    return result

def atomic_npy(path,array):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('wb') as f: np.save(f,array,allow_pickle=False)
    safe_replace(tmp,path)

def init_worker(zip_path,index,spec):
    global _RAW,_SPEC
    import cv2
    cv2.setNumThreads(1)
    from training.src import raptor_corpus as raw
    raw._init_worker(zip_path,index,spec)
    _RAW,_SPEC=raw,spec

def montage(directory,destination):
    import cv2
    tiles=[]
    for path in sorted(directory.glob('*.png')):
        im=cv2.imread(str(path)); need(im is not None,'overlay decode failed')
        tile=cv2.resize(im,(320,320)); cv2.putText(tile,path.stem,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1)
        tiles.append(tile)
    if not tiles: return False
    while len(tiles)%3: tiles.append(np.zeros((320,320,3),np.uint8))
    im=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
    ok,data=cv2.imencode('.png',im); need(ok,'overlay encode failed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.partial'); tmp.write_bytes(data.tobytes()); safe_replace(tmp,destination)
    return True

def run_chunk(args):
    out,index,start,tasks,selected,identity=args; out=Path(out); then=time.perf_counter()
    name=f'vols_shard{index:02d}.npy'; tmp=out/(name+'.partial')
    arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(tasks),64,448,448))
    masks=[]; records=[]; overlays={}
    for row,task in enumerate(tasks):
        position=start+row
        overlay_dir=out/'overlay-parts'/str(position) if position in selected else None
        volume,mask,record=c.build_study(_RAW,task,_SPEC,overlay_dir)
        arr[row]=volume; masks.append(mask); records.append(dict(uid=task[1],position=position,**record))
        if overlay_dir:
            dest=out/'overlays'/f'study-{position:04d}.png'
            if montage(overlay_dir,dest): overlays[dest.relative_to(out).as_posix()]=c.sha(dest)
    arr.flush(); del arr; safe_replace(tmp,out/name)
    qa=f'qa-shard{index:02d}.json'; maskname=f'masks-shard{index:02d}.npy'
    atomic_json(out/qa,records); atomic_npy(out/maskname,np.asarray(masks,dtype=np.uint8))
    receipt=dict(index=index,start=start,stop=start+len(tasks),file=name,done=True,
        bytes=(out/name).stat().st_size,sha256=c.sha(out/name),identity=identity,
        qa_file=qa,qa_sha256=c.sha(out/qa),mask_file=maskname,mask_sha256=c.sha(out/maskname),
        overlays=overlays,seconds=time.perf_counter()-then)
    atomic_json(out/f'chunk-{index:03d}.json',receipt)
    return receipt

def verify_chunk(out,receipt,identity,hash_pixels=True):
    need(receipt['identity']==identity,'chunk identity drift')
    path=out/receipt['file']
    need(path.stat().st_size==receipt['bytes'],'chunk truncated')
    if hash_pixels: need(c.sha(path)==receipt['sha256'],'chunk pixels changed')
    for key in ('qa','mask'):
        need(c.sha(out/receipt[key+'_file'])==receipt[key+'_sha256'],'chunk '+key+' changed')
    for name,digest in receipt['overlays'].items(): need(c.sha(out/name)==digest,'overlay changed')

def limits(counts,studies,valid_slices,max_padding):
    total=studies*64
    failures=counts
    decode=sum(failures.get(k,0) for k in ('header_decode_error','pixel_decode_error'))
    geometry=sum(failures.get(k,0) for k in ('missing_geometry','oblique_geometry','missing_orthogonal_prior','anchor_outside_field','invalid_geometry','singular_geometry','empty_series'))
    known={'header_decode_error','pixel_decode_error','missing_geometry','oblique_geometry','missing_orthogonal_prior','anchor_outside_field','invalid_geometry','singular_geometry','empty_series','missing_series','unknown_laterality','laterality_conflict','padded_crop','clamped_stack_target'}
    metrics=dict(decode_failures=decode,laterality_conflicts=failures.get('laterality_conflict',0),
        geometry_failures=geometry,empty_slice_fraction=(total-valid_slices)/max(total,1),
        padded_slice_fraction=failures.get('padded_crop',0)/max(total,1),padding_fraction=max_padding,
        clamped_slice_fraction=failures.get('clamped_stack_target',0)/max(total,1),
        unknown_laterality_fraction=failures.get('unknown_laterality',0)/max(studies,1))
    violations=['unrecognized_failure:'+k for k in failures if k not in known]
    for key,value in metrics.items():
        threshold=QA_THRESHOLDS.get(key,QA_THRESHOLDS.get('max_'+key))
        if value>threshold: violations.append(key)
    return metrics,violations

def qa(out,hash_pixels=True):
    out=Path(out); m=read(out/'manifest.json'); identity=m['identity']; ids=np.load(out/'ids.npy',allow_pickle=False)
    need(m['finished'] is True,'cache incomplete')
    need(identity.get('version')==VERSION and identity.get('geometry_sha256')==c.sha(c.__file__)
         and identity.get('builder_sha256')==c.sha(__file__),'QA implementation identity drift')
    need(identity.get('registration')==register(),'QA registration differs')
    need(c.sha(out/'ids.npy')==m['ids_sha256'],'ids changed')
    need(c.sha(out/'masks.npy')==m['masks_sha256'],'masks changed')
    masks=np.load(out/'masks.npy',allow_pickle=False); end=0; counts={}; overlays={}; max_padding=0.; bad_studies=0
    need(m['shape_total']==[len(ids),64,448,448] and masks.shape==(len(ids),64),'shape mismatch')
    need(identity['thresholds']==QA_THRESHOLDS,'QA thresholds changed')
    for index,s in enumerate(m['shards']):
        need(s['start']==end and s['stop']>end,'shard sequence invalid'); end=s['stop']
        verify_chunk(out,s,identity,hash_pixels)
        need(s==read(out/f'chunk-{index:03d}.json'),'chunk commit differs')
        np.testing.assert_array_equal(masks[s['start']:s['stop']],np.load(out/s['mask_file'],allow_pickle=False))
        records=read(out/s['qa_file']); need(len(records)==s['stop']-s['start'],'QA roster incomplete')
        for offset,r in enumerate(records):
            need(r['uid']==str(ids[s['start']+offset]),'QA roster changed')
            bad_studies+=bool(r['failures'])
            for key,value in r['failures'].items(): counts[key]=counts.get(key,0)+value
            for slot in r['slots']:
                for crop in slot['crops']: max_padding=max(max_padding,crop['padding_fraction'])
        overlays.update(s['overlays'])
    need(end==len(ids),'incomplete shards')
    empty=int((masks.sum(1)==0).sum()); valid=int(masks.sum())
    metrics,violations=limits(counts,len(ids),valid,max_padding)
    if empty: violations.append('empty_studies')
    selected=identity['overlay_positions']
    expected={f'overlays/study-{i:04d}.png' for i in selected}
    if set(overlays)!=expected: violations.append('missing_random_overlays')
    full=len(ids)==4410
    geometry_pass=not violations
    if not full: violations.append('engineering_subset_not_full_cache')
    result=dict(version=VERSION,studies=len(ids),empty_studies=empty,valid_slices=valid,
        studies_with_failures=bad_studies,failure_counts=counts,metrics=metrics,thresholds=QA_THRESHOLDS,
        qa_pass=not violations,geometry_qa_pass=geometry_pass,
        sample_qa_pass=geometry_pass and len(ids)>=64,
        violations=violations,overlay_sha256=overlays,overlay_count=len(overlays),
        overlay_selection='random_seeded',overlay_positions=selected,exposure='EXPOSED',clean_oof=False,
        bytes=sum(s['bytes'] for s in m['shards']),manifest_sha256=c.sha(out/'manifest.json'),
        pixel_hash_verification='all_chunks' if hash_pixels else 'build_or_resume_verified',
        full_cache=full,geometry=c.GEOMETRY_POLICY,
        scientific_admission='HOLD_FULL_CACHE_AND_INDEPENDENT_REVIEW')
    atomic_json(out/'qa-summary.json',result)
    return result

def build(limit=0,workers=4,chunk_size=16,max_chunks=0,out=None):
    need(1<=workers<=4,'workers must be 1..4'); need(1<=chunk_size<=100,'chunk size must be 1..100')
    need(limit>=0 and max_chunks>=0,'nonnegative limits required')
    out=Path(out) if out else (SAMPLE_OUT if limit else DEFAULT_OUT)
    need(out.resolve().is_relative_to(c.QA_WORK.resolve()),'output outside fixed task scratch')
    registration=register()
    from training.src import raptor_corpus as raw
    need(c.sha(c.SOURCE/'manifest.json')==SOURCE_SHA,'source identity drift')
    source=read(c.SOURCE/'manifest.json')
    for name in ('ids','masks'): need(c.sha(c.SOURCE/(name+'.npy'))==source[name+'_sha256'],'source index drift')
    ids=np.load(c.SOURCE/'ids.npy',allow_pickle=False)
    if limit: ids=ids[:limit]
    spec=dict(source['spec']); spec.update(img=448,crop_mm=100.,compartment_offset_mm=18.,
        span_mm={'Sagittal':[-12,12],'Coronal':[-24,24],'Axial':[-24,24]},
        localization=c.GEOMETRY_POLICY)
    selected=sorted(random.Random(QA_THRESHOLDS['overlay_seed']).sample(range(len(ids)),min(24,len(ids))))
    identity=dict(version=VERSION,source_manifest_sha256=SOURCE_SHA,source_picks_sha256=c.sha(c.SOURCE/'picks.jsonl'),
        builder_sha256=c.sha(__file__),geometry_sha256=c.sha(c.__file__),raw_builder_sha256=c.sha(raw.__file__),
        ids_sha256=hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest(),studies=len(ids),chunk_size=chunk_size,
        spec=spec,thresholds=QA_THRESHOLDS,registration=registration,
        overlay_positions=selected,exposure='EXPOSED',clean_oof=False)
    out.mkdir(parents=True,exist_ok=True); ip=out/'build-identity.json'
    if ip.exists(): need(read(ip)==identity,'resume identity drift; use a new versioned cache')
    else:
        need(not (out/'manifest.json').exists(),'legacy cache cannot be overwritten')
        atomic_json(ip,identity)
    completed={}; pending=[]
    for index,start in enumerate(range(0,len(ids),chunk_size)):
        rp=out/f'chunk-{index:03d}.json'; stop=min(start+chunk_size,len(ids))
        if rp.exists():
            r=read(rp); need(r['start']==start and r['stop']==stop and r['index']==index,'chunk roster drift')
            verify_chunk(out,r,identity); completed[index]=r
        else: pending.append((index,start,stop))
    if not pending and (out/'manifest.json').exists() and read(out/'manifest.json')['finished']:
        result=qa(out,hash_pixels=False); return dict(status='RESUMED_COMPLETE',new_chunks=0,**result)
    selected_pending=pending[:max_chunks] if max_chunks else pending
    required=sum((stop-start)*64*448*448 for _,start,stop in selected_pending)+2*(1<<30)
    need(shutil.disk_usage(out).free>=required,'insufficient disk for selected chunks + 2 GiB reserve')
    atomic_json(out/'manifest.json',dict(contract=c.CONTRACT,finished=False,identity=identity))
    picks={}
    with (c.SOURCE/'picks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            record=json.loads(line); picks[record['uid']]=record['split']
    recs=raw.series_records(); zip_path,index_path,_=raw.archive_paths(); jobs=[]
    for index,start,stop in selected_pending:
        tasks=[(picks[str(uid)],str(uid),recs[(picks[str(uid)],str(uid))]) for uid in ids[start:stop]]
        jobs.append((str(out),index,start,tasks,selected,identity))
    then=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker,
                             initargs=(str(zip_path),str(index_path),spec)) as pool:
        for result in pool.map(run_chunk,jobs,chunksize=1):
            completed[result['index']]=result
            print(json.dumps(dict(committed_chunk=result['index'],studies=result['stop']-result['start'],seconds=round(result['seconds'],2))),flush=True)
    if len(completed)*chunk_size<len(ids):
        return dict(status='CHUNKED_PARTIAL',new_chunks=len(jobs),committed_studies=sum(s['stop']-s['start'] for s in completed.values()),seconds=time.perf_counter()-then)
    shards=[completed[i] for i in sorted(completed)]
    masks=np.concatenate([np.load(out/s['mask_file'],allow_pickle=False) for s in shards])
    atomic_npy(out/'ids.npy',ids); atomic_npy(out/'masks.npy',masks)
    manifest=dict(contract=c.CONTRACT,spec=spec,studies=len(ids),shape_total=[len(ids),64,448,448],
        shards=shards,finished=True,ids_sha256=c.sha(out/'ids.npy'),masks_sha256=c.sha(out/'masks.npy'),
        identity=identity,source_manifest_sha256=SOURCE_SHA,source_picks_sha256=identity['source_picks_sha256'],
        archive=source['archive'],version=VERSION,exposure='EXPOSED',clean_oof=False,
        seconds_this_run=time.perf_counter()-then)
    atomic_json(out/'manifest.json',manifest)
    result=qa(out,hash_pixels=False)
    return dict(status=('BUILT_QA_PASS' if result['qa_pass'] else 'BUILT_QA_HOLD') if len(ids)==4410 else 'ENGINEERING_ONLY',new_chunks=len(jobs),**result)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('command',choices=('register','build','qa'))
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--chunk-size',type=int,default=16); ap.add_argument('--max-chunks',type=int,default=0)
    ap.add_argument('--out',type=Path); ap.add_argument('--engineering',action='store_true')
    a=ap.parse_args()
    if a.command=='register': result=register()
    elif a.command=='build': result=build(a.limit,a.workers,a.chunk_size,a.max_chunks,a.out)
    else: result=qa(a.out or DEFAULT_OUT)
    print(json.dumps(result))
    if a.command=='qa' and not result['qa_pass'] and not (a.engineering and result['sample_qa_pass']): return 2
    return 0

if __name__=='__main__': mp.freeze_support(); raise SystemExit(main())
