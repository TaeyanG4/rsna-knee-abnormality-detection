"""H-B-155 laterality v1.0.0 / cache v2.2.0, CPU-only affected-row migration.

Changelog 2026-10-15: independent DICOM fields + pinned R7 native side inputs;
registered conservative +/-2 logits, medial/lateral order, explicit unknown
channel, immutable parent and resumable audit/chunk commits. No pilot launch.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
from tools import hb155_crop_v1 as crop
from tools import hb155_cache_v2 as cache
from tools.hb150_io_v1 import atomic_json, safe_replace

VERSION='1.0.0'
CACHE_VERSION='2.2.0'
CONTRACT='hb155_compartment64_448_side_v2'
WORK=crop.QA_WORK
PARENT=WORK/'crop-cache-v211'
OUT=WORK/'crop-cache-v220'
PARENT_SHA='903c7eea7006ef34d37dcca419225708d1f02a7684c2e6bcb148429c5df6cac1'
WEIGHTS=crop.ROOT/'storage/external/heliosli-r7cnxfsss-v1/side_cls.pt'
WEIGHTS_SHA='ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a'
GEOM=crop.ROOT/'.local/exp189-cpu-v120/input/cloud_code/training/dicom_geometry.py'
GEOM_SHA='12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1'
SIDE_SHA='bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566'
THRESHOLDS=dict(cache.QA_THRESHOLDS,max_unknown_laterality_fraction=.02)
POLICY=dict(tag_fields=['ImageLaterality','Laterality','BodyPartExamined','SeriesDescription'],
    tags='exact L/R/LEFT/RIGHT; descriptions standalone LEFT/RIGHT/LT/RT or L/R adjacent to KNEE',
    classifier='all available coronal/axial series; exact R7 native uint16 inputs; L basis; mean logit',
    classifier_threshold=2.,resolution='consistent tags first; confident classifier resolves all conflicts',
    inconclusive='abs(logit)<2 or no decodable input; consistent tag stands, otherwise side_unknown',
    order='known: medial then lateral in each nonaxial slot; unknown: low then high patient LPS X',
    channel='side_unknown.npy uint8 [N,64,1,1]; broadcast channel 1, intensity channel 0',
    missing_series='zero pixels/masks; descriptive availability, not a failure',
    exposure='EXPOSED',clean_oof=False)
_RAW=_GEOM=_NET=_SC=None

def need(ok,message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def registration():
    need(crop.sha(PARENT/'manifest.json')==PARENT_SHA,'parent manifest drift')
    from training.src import raptor_corpus as raw
    parent=read(PARENT/'manifest.json'); zip_path,index,arc=raw.archive_paths()
    picks_sha=crop.sha(crop.SOURCE/'picks.jsonl'); raw_sha=crop.sha(raw.__file__)
    need(picks_sha==parent['identity']['source_picks_sha256'],'source picks drift')
    need(raw_sha==parent['identity']['raw_builder_sha256'],'raw builder drift')
    need(zip_path.stat().st_size==parent['archive']['zip_bytes'],'source archive size drift')
    need(arc['directory_sha256']==parent['archive']['directory_sha256'],'archive directory drift')
    inputs=dict(picks_sha256=picks_sha,raw_builder_sha256=raw_sha,
        series_sha256={name:crop.sha(raw.COMP/name) for name in ('train_series.csv','test_series.csv')},
        archive=dict(path=str(zip_path.resolve()),bytes=zip_path.stat().st_size,mtime_ns=zip_path.stat().st_mtime_ns,
                     directory_sha256=arc['directory_sha256'],integrity='file stat + pinned indexed member CRC checked on every read'),
        index=dict(path=str(index.resolve()),sha256=crop.sha(index)))
    for path,digest in ((WEIGHTS,WEIGHTS_SHA),(GEOM,GEOM_SHA),
                        (crop.ROOT/'tools/exp169_r7_src/side_common.py',SIDE_SHA)):
        need(crop.sha(path)==digest,'R7 input identity drift')
    return dict(version=VERSION,cache_version=CACHE_VERSION,parent_sha256=PARENT_SHA,
        tool_sha256=crop.sha(__file__),crop_sha256=crop.sha(crop.__file__),
        cache_sha256=crop.sha(cache.__file__),geometry_sha256=GEOM_SHA,
        weights_sha256=WEIGHTS_SHA,side_common_sha256=SIDE_SHA,
        inputs=inputs,policy=POLICY,thresholds=THRESHOLDS)

def register():
    result=registration(); path=WORK/'laterality-registration-v220.json'
    if path.exists(): need(read(path)==result,'laterality registration drift')
    else: atomic_json(path,result)
    return result

def tag_sides(ds):
    """Parse only allowlisted laterality fields; never store free-form strings."""
    values=set()
    for field in POLICY['tag_fields']:
        text=str(getattr(ds,field,'')).strip().upper()
        if text in ('L','LEFT'): values.add('L')
        elif text in ('R','RIGHT'): values.add('R')
        if field in ('BodyPartExamined','SeriesDescription'):
            tokens=re.findall(r'[A-Z]+',text)
            for i,token in enumerate(tokens):
                if token in ('LEFT','LT'): values.add('L')
                if token in ('RIGHT','RT'): values.add('R')
                if token in ('L','R') and 'KNEE' in tokens[max(0,i-1):i+2]: values.add(token)
    return values

def resolve(tags,score):
    tags=set(tags)
    need(tags<=set(('L','R')),'invalid side tag')
    model=None
    if score is not None:
        need(np.isfinite(score),'nonfinite side score')
        model='L' if score>=2 else 'R' if score<=-2 else None
    tag=next(iter(tags)) if len(tags)==1 else None
    conflict=len(tags)>1 or (tag is not None and model is not None and tag!=model)
    side=model if conflict else tag or model
    return dict(side=side or 'side_unknown',tag_side=tag,classifier_side=model,
        classifier_score=score,tag_values=sorted(tags),conflict=conflict,
        conflict_resolved=bool(conflict and model),
        source='classifier_conflict' if conflict and model else 'dicom' if tag else 'classifier' if model else 'unknown',
        agreement=None if tag is None or model is None else tag==model)

def load_geometry():
    need(crop.sha(GEOM)==GEOM_SHA,'geometry identity drift')
    name='_hb155_pinned_r7_geometry'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,GEOM)
        module=importlib.util.module_from_spec(spec); sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]

def init_worker(spec):
    global _RAW,_GEOM,_NET,_SC
    import torch
    import cv2
    from training.src import raptor_corpus as raw
    from tools.exp169_r7_src import side_common
    torch.set_num_threads(1); cv2.setNumThreads(1)
    z,i,_=raw.archive_paths(); raw._init_worker(str(z),str(i),spec)
    _RAW,_GEOM,_SC=raw,load_geometry(),side_common
    need(crop.sha(WEIGHTS)==WEIGHTS_SHA,'weights identity drift')
    _NET=side_common.build_side_net().cpu().eval()
    _NET.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)

def series_inputs(datasets,plane):
    """Exact R7 native_cache quantization and side_inputs on L canonical basis."""
    geom=_GEOM; headers=[]
    for path,ds in datasets.items():
        # Side evidence was retained separately. Geometry must not reject a
        # contradictory tag before the image classifier can resolve it.
        # Dataset.copy is shallow: deepcopy before changing tags.
        normalized=copy.deepcopy(ds)
        normalized.Laterality=''; normalized.ImageLaterality=''
        headers.append(geom.header_from_dataset(normalized,path))
    order=geom.order_series(Path('.'),plane,provided_records=headers)
    images=[geom.canonicalize_payload(geom.pixels_from_dataset(datasets[p],path=p),plane).image for p in order.files]
    volume=np.stack(images).astype(np.float32)
    lo=float(volume.min()); hi=float(volume.max())
    encoded=np.rint((volume-lo)/max(hi-lo,1e-6)*65535).clip(0,65535).astype(np.uint16)
    return _SC.side_inputs(encoded)

def audit_study(task):
    split,uid,rows=task; tags=set(); inputs=[]; diagnostics=[]
    by={}
    for p,s,i in _RAW._WORKER['con'].execute('SELECT path,series,info FROM files WHERE split=? AND uid=? ORDER BY path',(split,uid)):
        by.setdefault(s,[]).append((p,i))
    for row in rows:
        sid=row['SeriesInstanceUID']; plane=row['Anatomical_Plane']; datasets={}; failed=False
        for path,info in by.get(sid,[]):
            try:
                ds=_RAW._WORKER['pydicom'].dcmread(io.BytesIO(_RAW._member_bytes(path,info)),stop_before_pixels=plane not in ('Coronal','Axial'))
                tags.update(tag_sides(ds))
                if plane in ('Coronal','Axial'): datasets[path]=ds
            except Exception as exc:
                failed=True
                diagnostics.append(dict(series=sid,plane=plane,error=type(exc).__name__))
        if datasets and not failed:
            try: inputs.append(series_inputs(datasets,plane))
            except Exception as exc: diagnostics.append(dict(series=sid,plane=plane,error=type(exc).__name__))
    score=None
    if inputs:
        import torch
        xb=torch.from_numpy(np.concatenate(inputs)).float().div_(255.)[:,None]
        with torch.inference_mode(): score=float(_NET((xb-.45)/.25).float().mean())
    return dict(uid=uid,**resolve(tags,score),classifier_images=sum(len(x) for x in inputs),diagnostics=diagnostics)

def audit_chunk(job):
    index,tasks,identity=job
    rows=[audit_study(task) for task in tasks]
    path=OUT/'side-audit'/f'chunk-{index:03d}.json'
    atomic_json(path,dict(identity=identity,rows=rows,rows_sha256=rows_hash(rows)))
    return index,len(rows)


def rows_hash(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()


def checked_audit(path,identity,wanted):
    item=read(path)
    need(item['identity']==identity,'audit identity drift')
    need([r['uid'] for r in item['rows']]==wanted,'audit roster drift')
    need(item['rows_sha256']==rows_hash(item['rows']),'audit rows changed')
    for row in item['rows']:
        expected=resolve(row['tag_values'],row['classifier_score'])
        need(all(row[k]==v for k,v in expected.items()),'audit side decision drift')
    return item

def audit(workers=4,limit_chunks=0):
    need(1<=workers<=4,'workers must be 1..4')
    need(limit_chunks>=0,'negative chunk limit')
    identity=register(); OUT.mkdir(parents=True,exist_ok=True)
    from training.src import raptor_corpus as raw
    parent=read(PARENT/'manifest.json'); ids=np.load(PARENT/'ids.npy',allow_pickle=False)
    need(crop.sha(PARENT/'ids.npy')==parent['ids_sha256'],'parent ids drift')
    records=raw.series_records()
    splits={r['uid']:r['split'] for r in map(json.loads,(crop.SOURCE/'picks.jsonl').read_text(encoding='utf-8').splitlines())}
    jobs=[]; done=0
    for s in parent['shards']:
        index=s['index']; path=OUT/'side-audit'/f'chunk-{index:03d}.json'
        wanted=list(map(str,ids[s['start']:s['stop']]))
        if path.exists():
            checked_audit(path,identity,wanted); done+=len(wanted)
        else:
            tasks=[(splits[u],u,records[(splits[u],u)]) for u in wanted]
            jobs.append((index,tasks,identity))
    if limit_chunks: jobs=jobs[:limit_chunks]
    then=time.perf_counter()
    atomic_json(OUT/'status.json',dict(phase='side_audit',completed=done,total=len(ids)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(parent['spec'],)) as pool:
        for index,count in pool.map(audit_chunk,jobs,chunksize=1):
            done+=count
            atomic_json(OUT/'status.json',dict(phase='side_audit',completed=done,total=len(ids),last_chunk=index,seconds=time.perf_counter()-then))
    result=dict(phase='audit_complete' if done==len(ids) else 'audit_partial',completed=done,total=len(ids),seconds=time.perf_counter()-then)
    atomic_json(OUT/'status.json',result); return result

def permutation(spec,side):
    result=[]; start=0
    for plane,fluid,count in spec['slots']:
        indices=list(range(start,start+count))
        if side=='R' and plane!='Axial': indices=indices[count//2:]+indices[:count//2]
        result.extend(indices); start+=count
    need(start==64,'slot depth differs'); return np.array(result)

def updated_record(old,side,spec):
    record=copy.deepcopy(old); record['laterality']=side['side']; record['side_resolution']=side
    record['missing_series']=record['failures'].pop('missing_series',0)
    record['failures'].pop('unknown_laterality',None); record['failures'].pop('laterality_conflict',None)
    if side['side']=='side_unknown': record['failures']['unknown_laterality']=1
    if side['conflict'] and not side['conflict_resolved']: record['failures']['laterality_conflict']=1
    for slot,(plane,_,count) in zip(record['slots'],spec['slots']):
        if plane=='Axial': continue
        # Existing provenance has no output index; compartment group membership
        # is nevertheless fixed by low/high-X names or original known side.
        groups=[[],[]]
        for entry in slot['crops']:
            old_name=entry['compartment']
            comp=0 if old_name==crop.compartment_name(plane,0,old['laterality']) else 1
            entry['compartment']=crop.compartment_name(plane,comp,side['side'])
            entry['patient_x_group']='low' if comp==0 else 'high'
            groups[comp].append(entry)
        slot['crops']=sum(groups[::-1] if side['side']=='R' else groups,[])
    return record

def rebuild():
    identity=register(); parent=read(PARENT/'manifest.json')
    ids=np.load(PARENT/'ids.npy',allow_pickle=False); parent_masks=np.load(PARENT/'masks.npy',allow_pickle=False)
    need(crop.sha(PARENT/'masks.npy')==parent['masks_sha256'],'parent masks drift')
    import shutil
    pending=[s for s in parent['shards'] if not (OUT/f"chunk-{s['index']:03d}.json").exists()]
    need(shutil.disk_usage(WORK).free>=sum(s['bytes'] for s in pending)+2*(1<<30),'insufficient delta disk space')
    from training.src import raptor_corpus as raw
    recs=raw.series_records(); splits={r['uid']:r['split'] for r in map(json.loads,(crop.SOURCE/'picks.jsonl').read_text(encoding='utf-8').splitlines())}
    z,i,_=raw.archive_paths(); raw._init_worker(str(z),str(i),parent['spec'])
    shards=[]; all_masks=[]; flags=[]; then=time.perf_counter()
    try:
        for s in parent['shards']:
            index=s['index']; path=OUT/f'chunk-{index:03d}.json'
            wanted=list(map(str,ids[s['start']:s['stop']]))
            audit_record=checked_audit(OUT/'side-audit'/f'chunk-{index:03d}.json',identity,wanted)
            sides=audit_record['rows']
            audit_hash=crop.sha(OUT/'side-audit'/f'chunk-{index:03d}.json')
            if path.exists():
                receipt=read(path); cache.verify_chunk(OUT,receipt,identity)
                need(receipt['side_audit_sha256']==audit_hash,'side audit changed after commit')
            else:
                cache.verify_chunk(PARENT,s,parent['identity'])
                originals=read(PARENT/s['qa_file']); volume=np.load(PARENT/s['file'],mmap_mode='r',allow_pickle=False)
                masks=parent_masks[s['start']:s['stop']].copy(); records=[]; changed=[]; decoded=[]
                name=s['file']; tmp=OUT/(name+'.partial')
                arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=volume.shape)
                for row,(old,side) in enumerate(zip(originals,sides)):
                    position=s['start']+row; uid=side['uid']
                    need(old['uid']==uid,'parent QA roster drift')
                    # Only an existing geometric failure can trigger raw crop regeneration.
                    regenerate=bool(set(old['failures'])-{'unknown_laterality','laterality_conflict','missing_series'})
                    if regenerate:
                        split=splits[uid]
                        pixels,mask,old=crop.build_study(raw,(split,uid,recs[(split,uid)]),parent['spec'])
                        old=dict(uid=uid,position=position,**old); decoded.append(position)
                    else: pixels,mask=volume[row],masks[row]
                    perm=permutation(parent['spec'],side['side'])
                    arr[row]=pixels[perm]; masks[row]=mask[perm]
                    # Prove untouched rows byte-identical and reordered rows exact permutations.
                    need(np.array_equal(arr[row],pixels[perm]),'delta pixel identity differs')
                    if regenerate or side['side']=='R': changed.append(position)
                    records.append(updated_record(old,side,parent['spec']))
                arr.flush(); del arr,volume; safe_replace(tmp,OUT/name)
                qa_name=s['qa_file']; mask_name=s['mask_file']
                atomic_json(OUT/qa_name,records); cache.atomic_npy(OUT/mask_name,masks)
                overlay_map={}
                for overlay,digest in s['overlays'].items():
                    dest=OUT/overlay; dest.parent.mkdir(parents=True,exist_ok=True)
                    need(crop.sha(PARENT/overlay)==digest,'parent overlay drift')
                    shutil.copyfile(PARENT/overlay,dest)
                    overlay_map[overlay]=crop.sha(dest)
                interpretation=dict(note='Parent FOV overlays retained verbatim. Old text is historical; use these current side decisions and slot permutations. No crop location changed for sampled studies.',
                    rows=[dict(position=s['start']+j,side=r['side'],permutation=permutation(parent['spec'],r['side']).tolist()) for j,r in enumerate(sides)])
                interp_name=f'overlay-side-map-{index:03d}.json'
                atomic_json(OUT/interp_name,interpretation)
                receipt=dict(index=index,start=s['start'],stop=s['stop'],file=name,done=True,
                    bytes=(OUT/name).stat().st_size,sha256=crop.sha(OUT/name),identity=identity,
                    qa_file=qa_name,qa_sha256=crop.sha(OUT/qa_name),mask_file=mask_name,mask_sha256=crop.sha(OUT/mask_name),
                    overlays=overlay_map,overlay_interpretation=interp_name,overlay_interpretation_sha256=crop.sha(OUT/interp_name),side_audit_sha256=audit_hash,affected_pixel_positions=changed,
                    regenerated_positions=decoded,parent_chunk_sha256=s['sha256'])
                atomic_json(path,receipt)
            shards.append(receipt); all_masks.append(np.load(OUT/receipt['mask_file'],allow_pickle=False))
            flags.extend([int(r['side']=='side_unknown') for r in sides])
            atomic_json(OUT/'status.json',dict(phase='delta_rebuild',completed=s['stop'],total=len(ids),seconds=time.perf_counter()-then))
    finally: raw._WORKER['zip'].close(); raw._WORKER['con'].close()
    cache.atomic_npy(OUT/'ids.npy',ids); cache.atomic_npy(OUT/'masks.npy',np.concatenate(all_masks))
    cache.atomic_npy(OUT/'side_unknown.npy',np.broadcast_to(np.array(flags,np.uint8)[:,None,None,None],(len(ids),64,1,1)).copy())
    manifest=dict(contract=CONTRACT,version=CACHE_VERSION,finished=True,identity=identity,spec=parent['spec'],
        studies=len(ids),shape_total=[len(ids),64,448,448],shards=shards,
        ids_sha256=crop.sha(OUT/'ids.npy'),masks_sha256=crop.sha(OUT/'masks.npy'),side_unknown_sha256=crop.sha(OUT/'side_unknown.npy'),
        exposure='EXPOSED',clean_oof=False,channel_contract=POLICY['channel'],
        parent_manifest_sha256=PARENT_SHA,scientific_admission='HOLD_INDEPENDENT_REVIEW_NO_PILOT')
    atomic_json(OUT/'manifest.json',manifest)
    result=dict(phase='rebuild_complete',studies=len(ids),affected_pixel_studies=sum(len(s['affected_pixel_positions']) for s in shards),
        regenerated_studies=sum(len(s['regenerated_positions']) for s in shards),seconds=time.perf_counter()-then)
    atomic_json(OUT/'status.json',result); return result

def limits(counts,studies,valid_slices,max_padding,empty_studies):
    metrics,violations=cache.limits(counts,studies,valid_slices,max_padding)
    if metrics['unknown_laterality_fraction']>THRESHOLDS['max_unknown_laterality_fraction']:
        violations.append('unknown_laterality_fraction')
    if empty_studies>THRESHOLDS['empty_studies']: violations.append('empty_studies')
    return metrics,sorted(set(violations))


def overlays_valid(overlays,parent):
    expected={name:digest for shard in parent['shards'] for name,digest in shard['overlays'].items()}
    names={f'overlays/study-{i:04d}.png' for i in parent['identity']['overlay_positions']}
    return overlays==expected and set(overlays)==names and len(overlays)==THRESHOLDS['random_overlay_studies']


def qa():
    identity=register(); m=read(OUT/'manifest.json')
    need(m['finished'] and m['identity']==identity and m['contract']==CONTRACT,'cache identity drift')
    for name in ('ids','masks','side_unknown'):
        need(crop.sha(OUT/(name+'.npy'))==m[name+'_sha256'],'cache '+name+' drift')
    ids=np.load(OUT/'ids.npy',allow_pickle=False); masks=np.load(OUT/'masks.npy',allow_pickle=False)
    flags=np.load(OUT/'side_unknown.npy',allow_pickle=False)
    need(len(ids)==4410 and len(set(ids))==4410 and masks.shape==(4410,64) and flags.shape==(4410,64,1,1),'full roster/shape differs')
    need(np.isin(masks,[0,1]).all() and np.isin(flags,[0,1]).all(),'nonbinary masks/flags')
    need(crop.sha(OUT/'ids.npy')==crop.sha(PARENT/'ids.npy'),'parent roster differs')
    counts={}; sides=[]; max_padding=0.; missing=0; end=0; overlays={}
    parent=read(PARENT/'manifest.json')
    for index,s in enumerate(m['shards']):
        need(s['start']==end and s['stop']>end,'invalid chunk coverage'); end=s['stop']
        need(read(OUT/f'chunk-{index:03d}.json')==s,'chunk commit differs')
        cache.verify_chunk(OUT,s,identity)
        need(crop.sha(OUT/s['overlay_interpretation'])==s['overlay_interpretation_sha256'],'overlay interpretation drift')
        interpretations=read(OUT/s['overlay_interpretation'])['rows']
        overlays.update(s['overlays'])
        audit_path=OUT/'side-audit'/f'chunk-{index:03d}.json'
        need(crop.sha(audit_path)==s['side_audit_sha256'],'audit evidence drift')
        audit_record=checked_audit(audit_path,identity,list(map(str,ids[s['start']:s['stop']])))
        records=read(OUT/s['qa_file']); need(len(records)==s['stop']-s['start'],'QA count differs')
        np.testing.assert_array_equal(masks[s['start']:s['stop']],np.load(OUT/s['mask_file'],allow_pickle=False))
        pixels=np.load(OUT/s['file'],mmap_mode='r',allow_pickle=False)
        original=np.load(PARENT/s['file'],mmap_mode='r',allow_pickle=False)
        need(pixels.dtype==np.uint8 and pixels.shape==(len(records),64,448,448),'pixel shape differs')
        need(crop.sha(PARENT/s['file'])==s['parent_chunk_sha256'],'parent pixel drift')
        for offset,r in enumerate(records):
            pos=s['start']+offset; side=r['side_resolution']
            need(interpretations[offset]==dict(position=pos,side=side['side'],permutation=permutation(m['spec'],side['side']).tolist()),'overlay interpretation differs')
            if pos in parent['identity']['overlay_positions']:
                need(pos not in s['regenerated_positions'],'regenerated overlay needs fresh review')
            need(r['uid']==str(ids[pos]) and side==audit_record['rows'][offset],'QA roster/side differs')
            need(np.all(flags[pos]==int(side['side']=='side_unknown')),'side flag differs')
            expected=resolve(side['tag_values'],side['classifier_score'])
            need(all(side[k]==v for k,v in expected.items()),'side decision differs from registered rule')
            actual_mask=(pixels[offset].reshape(64,-1).sum(1)>0).astype(np.uint8)
            np.testing.assert_array_equal(actual_mask,masks[pos])
            if pos not in s['regenerated_positions']:
                np.testing.assert_array_equal(pixels[offset],original[offset][permutation(m['spec'],side['side'])])
            sides.append(side); missing+=r['missing_series']
            for k,v in r['failures'].items(): counts[k]=counts.get(k,0)+v
            slot_start=0
            for slot,(_,_,count) in zip(r['slots'],m['spec']['slots']):
                if slot['series'] is None:
                    need(not masks[pos,slot_start:slot_start+count].any(),'missing series is not masked')
                slot_start+=count
                for entry in slot['crops']: max_padding=max(max_padding,entry['padding_fraction'])
        del pixels,original
        atomic_json(OUT/'status.json',dict(phase='full_qa',completed=end,total=len(ids)))
    need(end==len(ids),'incomplete chunk coverage')
    empty=int((masks.sum(1)==0).sum())
    metrics,violations=limits(counts,len(ids),int(masks.sum()),max_padding,empty)
    if not overlays_valid(overlays,parent):
        violations.append('missing_random_overlays')
    both=[r for r in sides if r['agreement'] is not None]; conflicts=[r for r in sides if r['conflict']]
    result=dict(version=CACHE_VERSION,studies=len(ids),qa_pass=not violations,verdict='PASS' if not violations else 'HOLD',
        failure_counts=counts,metrics=metrics,thresholds=THRESHOLDS,violations=sorted(set(violations)),
        unknown_studies=sum(r['side']=='side_unknown' for r in sides),empty_studies=empty,missing_series_masked=missing,
        tags_classifier_both=len(both),agreement_count=sum(r['agreement'] for r in both),
        agreement_fraction=sum(r['agreement'] for r in both)/len(both) if both else None,
        conflicts=len(conflicts),conflicts_resolved=sum(r['conflict_resolved'] for r in conflicts),
        contradictory_tag_studies=sum(len(r['tag_values'])>1 for r in sides),
        classifier_missing=sum(r['classifier_score'] is None for r in sides),
        classifier_inconclusive=sum(r['classifier_score'] is not None and r['classifier_side'] is None for r in sides),
        affected_pixel_studies=sum(len(s['affected_pixel_positions']) for s in m['shards']),
        regenerated_studies=sum(len(s['regenerated_positions']) for s in m['shards']),
        overlay_sha256=overlays,overlay_count=len(overlays),overlay_selection='inherited registered seed 155; original FOV + authenticated side/permutation interpretation',
        manifest_sha256=crop.sha(OUT/'manifest.json'),exposure='EXPOSED',clean_oof=False,
        scientific_admission='HOLD_INDEPENDENT_REVIEW_NO_PILOT')
    atomic_json(OUT/'qa-summary.json',result)
    atomic_json(OUT/'status.json',dict(phase='qa_complete',completed=end,total=len(ids),verdict=result['verdict']))
    return result

def channels(volume,unknown):
    """Materialize [64,2,H,W]: uint8 intensity then binary side-unknown channel.

    Downstream training must explicitly adopt the v2 contract. The legacy
    three-channel MIL adapter rejects it; no model recipe is silently changed.
    """
    need(volume.ndim==3 and volume.shape[0]==64,'volume shape differs')
    need(unknown in (0,1),'unknown channel must be binary')
    return np.stack((volume,np.full_like(volume,unknown)),axis=1)

def load_channels(uid,directory=OUT):
    """Read the authenticated explicit two-channel cache representation."""
    from training.src.raptor_common import Corpus
    directory=Path(directory); m=read(directory/'manifest.json')
    need(m['contract']==CONTRACT and m['finished'],'laterality cache contract differs')
    need(crop.sha(directory/'side_unknown.npy')==m['side_unknown_sha256'],'unknown channel drift')
    corpus=Corpus(directory); matches=np.flatnonzero(corpus.ids==uid)
    need(len(matches)==1,'UID absent or duplicated')
    flags=np.load(directory/'side_unknown.npy',allow_pickle=False)
    row=flags[int(matches[0])]
    need(row.shape==(64,1,1) and np.all(row==row[0]) and row[0,0,0] in (0,1),'unknown channel malformed')
    return channels(corpus.volume(uid),int(row[0,0,0])),corpus.mask(uid)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('register','audit','rebuild','qa'))
    ap.add_argument('--workers',type=int,default=4); ap.add_argument('--limit-chunks',type=int,default=0)
    a=ap.parse_args()
    result=register() if a.command=='register' else audit(a.workers,a.limit_chunks) if a.command=='audit' else rebuild() if a.command=='rebuild' else qa()
    print(json.dumps(result))
    return 2 if a.command=='qa' and not result['qa_pass'] else 0

if __name__=='__main__': mp.freeze_support(); raise SystemExit(main())
