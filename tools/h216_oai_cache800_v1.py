"""H216 v1.0.0: missing-only fixed800 OAI native/geometry/actual-v4 producer.

Registration and one predeclared CPU native/geometry smoke are separate from
bulk execution. Bulk requires the actual pilot result, bound review, measured
cost/capacity decision and exclusive gpu_job. No retry, deletion or new roster.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.l1_rco_stage0_v1 import need,read,write,sha,digest,PYTHON

VERSION='1.0.0';TAG='H216_OAI_CACHE800_V1';GIB=2**30
OUT=ROOT/'.local/scratch/slot-b/h216-oai-cache800-v1'
CACHE=Path('G:/temp_data/rsna-h216-oai-v4-v1')
INPUT=ROOT/'.local/scratch/slot-b/h216-oai-pair-inputs-v1'
PILOT=ROOT/'.local/scratch/slot-b/h216-oai-v4pilot8-v1'
CODE=ROOT/'.local/exp189-cpu-v120/input/cloud_code'
SOURCE=ROOT/'tools/exp189_s6_src'
_NATIVE=None


def key(uid):return hashlib.sha256(uid.encode()).hexdigest()[:24]


def safe_destination(path,anchor):
    """No reparse component, parent escape, root itself or user-data removal."""
    p=Path(path).absolute();base=Path(anchor).resolve()
    need(p.resolve().is_relative_to(base) and p.resolve()!=base,'DESTINATION_ESCAPE')
    for part in (p,*p.parents):
        if part.exists():
            st=part.lstat();need(not part.is_symlink() and not (getattr(st,'st_file_attributes',0)&1024),'REPARSE_DESTINATION')
    return p


def split_required(binding,pilot_ids):
    ids=binding['schedules']['oai_required_ids'];reuse=[u for u in ids if u in set(pilot_ids)]
    missing=[u for u in ids if u not in set(pilot_ids)]
    need(len(ids)==len(set(ids))==800 and len(reuse)==3 and len(missing)==797,'FIXED800_PARTITION')
    need(sorted(reuse)==binding['pilot_reusable_ids'],'REUSE_IDENTITY')
    return ids,reuse,missing


def storage_bound(rows,headers):
    native=0;render=0;frames=0
    for r in rows:
        h=headers[r['group_key']];n=r['frame_count']
        need(n==h['frame_count'] and h['rows']>0 and h['columns']>0,'HEADER_DIMENSIONS')
        native+=n*int(h['rows'])*int(h['columns'])*2
        render+=n*(4 if r['Anatomical_Plane']!='Coronal' else 3)*(384**2+48)
        frames+=n
    # Raw sizing is a conservative planning allowance, not a measured DICOM size.
    # At execution each archive's selected-member bytes are measured before copy.
    raw_allowance=4*native+frames*2**20
    return dict(native_bytes=native,v4_unslabbed_bytes=render,raw_planning_allowance=raw_allowance,
                assembly_duplication_bytes=render,metadata_allowance=GIB,
                required_free_bytes=native+2*render+raw_allowance+101*GIB,
                actual_raw_member_preflight_required=True)


def register():
    need(not OUT.exists(),'OUTPUT_EXISTS')
    from tools.h216_oai_pair_runtime_v1 import inputs
    from tools.extcache import oai_cache_v2
    from tools.h216_oai_native8_v101 import audited_sides
    binding=inputs();pr=read(PILOT/'REGISTRATION.json')
    ids,reuse,missing=split_required(binding,[c['uid'] for c in pr['cases']])
    by={r['study_key']:r['series'] for r in binding['selected_oai']}
    audit=oai_cache_v2.load_audit();groups={};archives={};headers={}
    wanted={r['group_key'] for rs in by.values() for r in rs}
    for gkey in wanted:
        g=audit.groups[gkey];need(g['eligible'] and g['contrast_verified'],'UNQUALIFIED_AUDIT_GROUP')
        groups[gkey]=g;archives[g['series_key']]=audit.archives[g['series_key']]
    with (audit.directory/'routing.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['group_key'] in wanted:headers[r['group_key']]={k:r[k] for k in ('rows','columns','frame_count')}
    need(set(headers)==wanted,'HEADER_BINDING_MISSING')
    for uid,rs in by.items():
        for row in rs:
            g=groups[row['group_key']]
            need(g['study_key']==uid and g['series_key']==row['SeriesInstanceUID'] and g['frame_count']==row['frame_count'],'GROUP_IDENTITY')
    for ar in archives.values():
        st=Path(ar['path']).stat();need(st.st_size==ar['size'] and st.st_mtime_ns==ar['mtime_ns'],'ARCHIVE_STAMP_DRIFT')
    paths=[Path(__file__),ROOT/'tests/test_h216_oai_cache800_v1.py',ROOT/'docs/research/h216-oai-cache800-v1.md',
           ROOT/'tools/h216_oai_native8_v101.py',ROOT/'tools/h216_oai_geometry8_v1.py',
           ROOT/'tools/h216_oai_v4pilot8_v1.py',ROOT/'tools/h216_oai_pair_runtime_v1.py',
           ROOT/'tools/hb144_oai_public_v1.py',ROOT/'tools/l1_rco_cache_v1.py',ROOT/'tools/l1_rco_runtime_v120.py',
           ROOT/'tools/l1_rco_stage0_v1.py',ROOT/'tools/job_runtime.py',
           INPUT/'REGISTRATION.private.json',INPUT/'RESULT.json',PILOT/'REGISTRATION.json']
    paths += [SOURCE/n for n in ('probe_infer.py','px_cache.py','compact_train.py','knee_anchor_bank_v2.py')]
    paths += list((CODE/'training').rglob('*.py'))
    paths += [audit.directory/n for n in ('summary.json','routing.jsonl','archives.jsonl','slot-selections.jsonl')]
    reg=dict(version=VERSION,input_contract=binding['contract_id'],ids=ids,reuse_ids=reuse,missing_ids=missing,
             rows=by,groups=groups,archives=archives,headers=headers,side_by_uid=audited_sides(by),
             source_pins={str(p):sha(p) for p in paths},cache=str(CACHE),pilot_registration_sha256=sha(PILOT/'REGISTRATION.json'),
             storage=storage_bound([r for rs in by.values() for r in rs],headers),
             smoke_uid=missing[0],bulk_admitted=False,raw_source='existing approved local archives only')
    reg['contract_id']=digest(reg);OUT.mkdir();write(OUT/'REGISTRATION.private.json',reg)
    return dict(status='REGISTERED_NOT_ADMITTED',studies=800,reuse_pilot=3,missing=797,
                contract_id=reg['contract_id'],required_free_gib=reg['storage']['required_free_bytes']/GIB,
                cuda=False,models_loaded=0,bulk_allocation=False)


def registration():
    r=read(OUT/'REGISTRATION.private.json')
    need(r['contract_id']==digest({k:v for k,v in r.items() if k!='contract_id'}),'REGISTRATION_DIGEST')
    for p,h in r['source_pins'].items():need(sha(p)==h,'SOURCE_DRIFT')
    need(r['cache']==str(CACHE),'CACHE_PATH_DRIFT')
    return r


def archive_member_bytes(archive,group):
    st=Path(archive['path']).stat()
    need(st.st_size==archive['size'] and st.st_mtime_ns==archive['mtime_ns'],'ARCHIVE_STAMP_DRIFT')
    wanted=set(group['members']);need(len(wanted)==group['frame_count'],'FRAME_ROSTER')
    found={}
    with tarfile.open(archive['path'],'r|gz') as tf:
        for member in tf:
            name=PurePosixPath(member.name.replace('\\','/')).as_posix()
            if name in wanted:
                need(member.isfile() and name not in found and member.size>0,'ARCHIVE_MEMBER_INVALID')
                found[name]=member.size
    need(set(found)==wanted,'ARCHIVE_MEMBERS_MISSING')
    return sum(found.values())


def cpu_case(reg,uid,dest,progress):
    """New per-study composition of reviewed native/geometry functions only."""
    global _NATIVE
    need(uid in reg['missing_ids'],'CPU_CASE_NOT_MISSING');need(not dest.exists(),'CASE_ALREADY_EXISTS')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ISOLATION')
    from types import SimpleNamespace
    from tools import h216_oai_native8_v101 as native
    from tools import h216_oai_geometry8_v1 as geometry
    from tools import hb144_oai_public_v1 as extraction
    import numpy as np
    import torch
    import pydicom
    import cv2
    torch.set_num_threads(1);cv2.setNumThreads(1);need(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
    old_load=torch.load
    def forbidden(*a,**kw):raise ValueError('CPU_MODEL_LOAD_FORBIDDEN')
    torch.load=forbidden
    started=time.perf_counter();dest.mkdir(parents=True);data=dest/'data';data.mkdir()
    rows=reg['rows'][uid];raw_bytes=0;extractions=[]
    try:
        with (data/'test_series.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with (data/'test.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['StudyInstanceUID']);w.writerow([uid])
        audit=SimpleNamespace(archives=reg['archives'])
        for row in rows:
            group=reg['groups'][row['group_key']];ar=reg['archives'][row['SeriesInstanceUID']]
            size=archive_member_bytes(ar,group)
            need(shutil.disk_usage(dest).free>size+100*GIB,'RAW_COPY_DISK_RESERVE')
            progress('raw-member-check')
            directory=data/'test_series'/uid/row['SeriesInstanceUID']
            rec=extraction.extract_group(audit,group,directory);raw_bytes+=size;extractions.append(rec)
            geometry.frame_binding(directory,group['members'],rec);progress('extract')
        # All helper imports resolve before installing the frozen training package.
        sys.path[:0]=[str(SOURCE),str(CODE)]
        import probe_infer
        if _NATIVE is None:_NATIVE=probe_infer.install_native(True)
        module=_NATIVE;original_side=module.geom.decide_study_side
        module.geom.decide_study_side=native.side_resolver(original_side,reg['side_by_uid'])
        probe_infer.OneStudySource.current=uid
        try:
            with contextlib.redirect_stdout(io.StringIO()):module.prepare(data,dest/'native',split='test',study_ids=[uid],source_directory=True)
        finally:module.geom.decide_study_side=original_side
        sp=dest/'native/studies'/(key(uid)+'.json');study=read(sp);native.validate_native(study,rows)
        pins={str(sp):sha(sp)};native_bytes=0
        for s in study['series']:
            directory=data/'test_series'/uid/s['SeriesInstanceUID']
            need(len(set(s['files']))==len(s['files']) and {directory/n for n in s['files']}==set(directory.glob('*.dcm')),'NATIVE_RAW_ROSTER')
            a=np.load(s['path'],mmap_mode='r',allow_pickle=False)
            need(a.dtype==np.uint16 and tuple(a.shape)==tuple(s['native_shape']) and a.ndim==3 and a.size>0,'NATIVE_VOLUME_LAYOUT')
            need(len(a)==len(s['positions'])==len(s['files']) and np.isfinite(s['positions']).all(),'NATIVE_POSITIONS')
            need(np.isfinite(s['spacing_mm']).all() and np.all(np.asarray(s['spacing_mm'])>0),'NATIVE_SPACING')
            pins[s['path']]=sha(s['path']);native_bytes+=Path(s['path']).stat().st_size;del a
        progress('native')
        mp=dest/'native-manifest.private.json';write(mp,dict(config=dict(archive_etag='h216-verified-directory:'+digest(extractions)),studies=[study]))
        with contextlib.redirect_stdout(io.StringIO()):geometry.bank.build(data,mp,dest/'geometry','test',True)
        frames=0;records=[];max_error=0.
        for s in study['series']:
            gp=dest/'geometry'/(key(uid+'/'+s['SeriesInstanceUID'])+'.json');rec=read(gp)
            need(rec['StudyInstanceUID']==uid and rec['SeriesInstanceUID']==s['SeriesInstanceUID'],'GEOMETRY_IDENTITY')
            hs=[pydicom.dcmread(data/'test_series'/uid/s['SeriesInstanceUID']/n,stop_before_pixels=True,
                specific_tags=['ImageOrientationPatient','ImagePositionPatient','PixelSpacing','Rows','Columns','Laterality','ImageLaterality']) for n in s['files']]
            max_error=max(max_error,geometry.validate_record(rec,s,hs));frames+=len(hs);records.append(rec);pins[str(gp)]=sha(gp)
            progress('geometry')
        side,evidence=geometry.bank.resolve_anatomy_laterality([dict(raw_explicit_sides=r['raw_explicit_sides']) for r in records])
        need(all(r['explicit_side']==side and r['laterality_status']==evidence['status'] and
                 r['study_explicit_sides']==evidence['explicit_sides'] for r in records),'SIDE_PROMOTION')
        pins[str(mp)]=sha(mp);pins[str(dest/'geometry/manifest.json')]=sha(dest/'geometry/manifest.json')
        need(frames==sum(r['frame_count'] for r in rows),'NATIVE_FINAL_FRAME_COUNT')
        result=dict(status='NATIVE_GEOMETRY_PREPARED',contract_id=reg['contract_id'],uid=uid,study=str(sp),
                    geometry=str(dest/'geometry'),series=len(records),frames=frames,raw_bytes=raw_bytes,
                    native_bytes=native_bytes,seconds=time.perf_counter()-started,max_mapping_error_mm=max_error,
                    geometry_errors=0,explicit_side_status=evidence['status'],outputs=pins,extractions=extractions,
                    cuda=False,models_loaded=0,actual_roi=False)
        write(dest/'RESULT.private.json',result);progress('native-case-complete');return result
    finally:torch.load=old_load


def cpu_smoke():
    reg=registration();need(not (OUT/'CPU-SMOKE.json').exists(),'SMOKE_EXISTS')
    # A single new case, permanently reusable by the eventual bulk worker.
    uid=reg['smoke_uid'];estimated=storage_bound(reg['rows'][uid],reg['headers'])['required_free_bytes']
    need(shutil.disk_usage(OUT).free>estimated,'SMOKE_DISK_RESERVE')
    rec=cpu_case(reg,uid,OUT/'cpu-smoke',lambda phase:None)
    r={k:rec[k] for k in ('status','contract_id','series','frames','raw_bytes','native_bytes','seconds','max_mapping_error_mm','cuda','models_loaded','actual_roi')}
    r.update(scope='ONE_NEW_PREDECLARED_NATIVE_GEOMETRY_CASE',receipt_sha256=sha(OUT/'cpu-smoke/RESULT.private.json'),training_ready=False)
    write(OUT/'CPU-SMOKE.json',r);return r


def pilot_ready(reg):
    r=read(PILOT/'RESULT.json');check=read(PILOT/'B-RESULT-CHECK.json')
    need(r['status']=='V4_PILOT_PREPARED' and r['studies']==8 and r['actual_roi'] is True and r['parity']=='PASS','ACTUAL_PILOT_REQUIRED')
    need(r['contract_id']==read(PILOT/'REGISTRATION.json')['contract_id'] and
         sha(PILOT/'REGISTRATION.json')==reg['pilot_registration_sha256'],'PILOT_IDENTITY')
    need(check['verdict']=='PASS_B_DIRECT' and check['result_sha256']==sha(PILOT/'RESULT.json'),'PILOT_RESULT_CHECK_REQUIRED')
    need(all(not a['readiness_hold'] for a in r['anchors']),'PILOT_FIELD_HOLD')
    for uid in reg['reuse_ids']:
        for rel in ('shards/'+key(uid)+'/receipt.json','shards/'+key(uid)+'/px.npy',
                    'shards/'+key(uid)+'/meta.npy','native/studies/'+key(uid)+'.json','anchors/'+key(uid)+'.json'):
            need(rel in check['reusable_files'] and sha(PILOT/rel)==check['reusable_files'][rel],'PILOT_REUSE_BINDING')
    return r


def readiness():
    reg=registration();blockers=[];pilot=None
    try:pilot=pilot_ready(reg)
    except (OSError,KeyError,ValueError):blockers.append('ACTUAL_PILOT_RESULT_AND_B_CHECK_MISSING')
    need((OUT/'CPU-SMOKE.json').exists(),'CPU_SMOKE_REQUIRED')
    smoke=read(OUT/'CPU-SMOKE.json');need(smoke['receipt_sha256']==sha(OUT/'cpu-smoke/RESULT.private.json'),'SMOKE_DRIFT')
    r=dict(status='BLOCKED' if blockers else 'PILOT_READY_CAPACITY_REVIEW_PENDING',blockers=blockers,
           contract_id=reg['contract_id'],physical_gpu_claim=False,bulk_allocation=False,
           native_remaining_after_smoke=796,v4_new_studies=797,pilot_reuse=3,
           storage=reg['storage'],pilot_result_sha256=sha(PILOT/'RESULT.json') if pilot else None,
           smoke_sha256=sha(OUT/'CPU-SMOKE.json'),cost_scope='pilot all-up and actual native smoke; assembly/hash cost and tails must be budgeted')
    write(OUT/'READINESS.json',r);return {k:v for k,v in r.items() if k not in ('storage',)}


def admit_cpu():
    reg=registration();pilot=pilot_ready(reg);review=read(OUT/'B-PRECHECK.json')
    need(review['verdict']=='PASS_B_DIRECT' and review['reviewer']=='B/Codex' and
         review['authority']=='Taeyang2026-10-19-solo-continuous-GPU','B_PRECHECK_REQUIRED')
    need(review['registration_sha256']==sha(OUT/'REGISTRATION.private.json') and
         review['report_sha256']==sha(OUT/'B-PRECHECK.md') and
         review['pilot_result_sha256']==sha(PILOT/'RESULT.json') and
         review['smoke_sha256']==sha(OUT/'CPU-SMOKE.json'),'REVIEW_BINDING')
    cost=read(OUT/'COST-DECISION.json')
    need(sha(OUT/'COST-DECISION.json')==review['cost_sha256'] and cost['decision']=='PROCEED' and
         cost['pilot_result_sha256']==sha(PILOT/'RESULT.json') and cost['contract_id']==reg['contract_id'] and
         cost['includes_native_geometry_anchor_render_assembly_hash_pair_infer_review'] is True,'MEASURED_COST_DECISION_REQUIRED')
    return reg,pilot


def command():return [PYTHON,str(Path(__file__).resolve()),'run']


def require_job(worker=False):
    import psutil
    from tools.job_runtime import LOGS,file_lock,identity,read_json,command_digest
    with file_lock(LOGS/'.gpu-claim.lock'):
        parent=psutil.Process().parent();need(parent is not None,'MISSING_PARENT')
        owner=parent.parent() if worker else parent;need(owner is not None,'MISSING_OWNER')
        j=read_json(LOGS/TAG/'job.json')
        need(j.get('schema')=='gpu-job/v2' and j.get('tag')==TAG and j.get('phase')=='running','GPU_JOB_REQUIRED')
        need(j.get('child_identity')==(identity(parent.pid) if worker else identity()) and
             {k:j.get(k) for k in ('pid','create_time')}==identity(owner.pid),'GPU_JOB_IDENTITY')
        need(j.get('command')==command() and j.get('command_sha256')==command_digest(command(),ROOT) and Path(j['cwd']).resolve()==ROOT,'GPU_JOB_COMMAND')
        need(owner.cmdline()[-3:]==['--_run-job',TAG,j['run_id']] and any(Path(x).name=='gpu_job.py' for x in owner.cmdline()),'GPU_JOB_PARENT')
        if worker:need(parent.cmdline()[1:]==command()[1:],'WORKER_PARENT_COMMAND')
        return owner.pid


def progress_writer(gpu=False):
    count=0
    def progress(phase='copy'):
        nonlocal count
        count+=1;peak=0
        if gpu:
            import torch
            from tools.l1_rco_runtime_v120 import memory_ok
            a=read(OUT/'ADMISSION.json');peak=torch.cuda.max_memory_reserved()
            need(memory_ok(peak,a['desktop_bytes'],a['total_bytes']),'MEMORY_GATE')
        try:write(OUT/'status.json',dict(phase=phase,time=time.time(),steps=count,windows=count,peak_reserved=peak))
        except OSError:pass  # persistent missing real progress is supervised externally
    return progress


def check_case(rec):
    need(rec['status']=='NATIVE_GEOMETRY_PREPARED' and rec['geometry_errors']==0,'NATIVE_CASE_INCOMPLETE')
    for p,h in rec['outputs'].items():need(sha(p)==h,'NATIVE_CASE_DRIFT')


def native_worker():
    require_job(True);reg,_=admit_cpu();progress=progress_writer();case_map={}
    smoke=read(OUT/'cpu-smoke/RESULT.private.json');check_case(smoke)
    need(smoke['uid']==reg['smoke_uid'] and smoke['contract_id']==reg['contract_id'],'SMOKE_CASE_IDENTITY')
    for i,uid in enumerate(reg['missing_ids'],1):
        rec=smoke if uid==reg['smoke_uid'] else cpu_case(reg,uid,CACHE/'cases'/key(uid),progress)
        case_map[uid]=dict(receipt=str((OUT/'cpu-smoke' if uid==reg['smoke_uid'] else CACHE/'cases'/key(uid))/'RESULT.private.json'),study=rec['study'],geometry=rec['geometry'])
        progress('native-case-committed');print(json.dumps(dict(native_completed=i,total=797,reused_smoke=uid==reg['smoke_uid'])),flush=True)
    write(CACHE/'NATIVE-COMPLETE.private.json',dict(status='NATIVE797_COMPLETE',contract_id=reg['contract_id'],cases=case_map))


def copy_checked(source,target,expected,progress):
    """Stream copy with hash; no hardlink alias or source mutation."""
    need(not target.exists(),'COPY_TARGET_EXISTS');target.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    with source.open('rb') as a,target.open('xb') as b:
        while data:=a.read(8*2**20):b.write(data);h.update(data);progress()
        b.flush();os.fsync(b.fileno())
    need(h.hexdigest()==expected,'COPIED_SHARD_HASH')


def stream_hash(path,progress):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(32*2**20):h.update(b);progress('checksum')
    return h.hexdigest()


def v4_worker():
    require_job(True);reg,pilot_result=admit_cpu()
    # CPU-only adapters deliberately never imported here (they disable CUDA).
    from tools import h216_oai_v4pilot8_v1 as pilot
    torch,pxmod,ct,anchors=pilot.modules();need(torch.cuda.is_available(),'CUDA_UNAVAILABLE')
    torch.cuda.reset_peak_memory_stats();progress=progress_writer(True);started=time.perf_counter()
    need(sha(pilot.WEIGHT)==pilot.WEIGHT_SHA,'ANATOMY_WEIGHT_DRIFT');model=anchors.load_model(pilot.WEIGHT,'cuda')
    calls=0
    def after_forward(*args):
        nonlocal calls
        torch.cuda.synchronize();calls+=1;progress('anchor-forward')
    hook=model.register_forward_hook(after_forward);bank=anchors.AnchorBank(model,'cuda')
    manifest=read(CACHE/'NATIVE-COMPLETE.private.json')
    need(manifest['status']=='NATIVE797_COMPLETE' and manifest['contract_id']==reg['contract_id'] and
         set(manifest['cases'])==set(reg['missing_ids']),'NATIVE_MANIFEST')
    recs=[];audits=[]
    for uid in reg['ids']:
        before=calls;t=time.perf_counter()
        if uid in reg['reuse_ids']:
            src=PILOT/'shards'/key(uid);rec=read(src/'receipt.json')
            need(rec['uid']==uid,'REUSE_SHARD_IDENTITY')
            for name,h in rec['hashes'].items():copy_checked(src/name,CACHE/'shards'/key(uid)/name,h,progress)
            study=read(PILOT/'native/studies'/(key(uid)+'.json'));anchor=read(PILOT/'anchors'/(key(uid)+'.json'))
            audit=pilot.anchor_check(anchor,study);audit.update(reused=True,model_calls=0,seconds=0.)
            write(CACHE/'shards'/key(uid)/'receipt.json',rec)
        else:
            case=manifest['cases'][uid];native_rec=read(case['receipt']);check_case(native_rec)
            study=read(case['study']);anchor=bank.record(uid,study,geo_dir=case['geometry'])
            audit=pilot.anchor_check(anchor,study);audit.update(reused=False,model_calls=calls-before,seconds=time.perf_counter()-t)
            write(CACHE/'anchors'/(key(uid)+'.json'),anchor)
            need(not audit['readiness_hold'],'OUT_OF_FIELD_READINESS_HOLD')
            rec=pilot.render_one(study,anchor,CACHE,pxmod,ct,progress)
        need(not audit['readiness_hold'],'REUSED_FIELD_HOLD')
        if audit['no_sagittal']:need(audit['slab_prior'] and audit['model_calls']==0,'NO_SAGITTAL_CONTRACT')
        recs.append(rec);audits.append(audit);write(CACHE/'audits'/(key(uid)+'.json'),audit)
        progress('render-case-committed');print(json.dumps(dict(v4_completed=len(recs),total=800,reused=uid in reg['reuse_ids'])),flush=True)
    hook.remove();del bank,model;torch.cuda.empty_cache()
    pilot.assemble(CACHE,recs,pxmod,progress);pilot.loader_parity(CACHE,recs,ct,progress)
    files={};hashes={}
    for name in ('px.u8.npy','meta.f32.npy','index.json'):
        hashes[name]=stream_hash(CACHE/name,progress);st=(CACHE/name).stat();files[name]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    r=dict(status='QUALIFIED_ACTUAL_V4_CACHE',version=VERSION,contract_id=reg['contract_id'],
        input_contract=reg['input_contract'],ids=reg['ids'],views='v4',size=384,actual_anchor=True,
        loader_parity='PASS',geometry_errors=0,model_errors=0,out_of_field_count=0,
        anchor_model_studies=sum(not a['slab_prior'] for a in audits),explicit_slab_prior_studies=sum(a['slab_prior'] for a in audits),
        hashes=hashes,files=files,actual_model_calls=calls,reused_pilot_studies=3,
        newly_rendered_studies=797,seconds=time.perf_counter()-started,peak_reserved=torch.cuda.max_memory_reserved(),
        pilot_result_sha256=sha(PILOT/'RESULT.json'),producer_registration_sha256=sha(OUT/'REGISTRATION.private.json'))
    from tools.h216_oai_pair_runtime_v1 import cache_receipt_check
    binding=read(INPUT/'REGISTRATION.private.json');cache_receipt_check(r,read(CACHE/'index.json'),binding)
    write(CACHE/'receipt.json',r);progress('complete')


def run():
    owner=require_job();reg,pilot=admit_cpu()
    need(not (OUT/'ADMISSION.json').exists(),'ATTEMPT_ALREADY_ADMITTED')
    safe_destination(CACHE,Path('G:/temp_data'));need(not CACHE.exists(),'CACHE_DESTINATION_EXISTS')
    from tools.l1_rco_runtime_v120 import active_ml_processes,gpu_snapshot,stale_reason
    need(not active_ml_processes(owner),'ANOTHER_GPU_PROCESS');snap=gpu_snapshot()
    need(snap['desktop_bytes']+2*GIB<snap['total_bytes'],'GPU_HEADROOM')
    need(shutil.disk_usage(CACHE.parent).free>reg['storage']['required_free_bytes'],'CAPACITY_RESERVE')
    write(OUT/'ADMISSION.json',dict(snap,contract_id=reg['contract_id'],time=time.time(),
          actual_free_bytes=shutil.disk_usage(CACHE.parent).free))
    CACHE.mkdir();write(CACHE/'contract.json',dict(contract_id=reg['contract_id'],input_contract=reg['input_contract']))
    child=None;started=time.perf_counter()
    try:
        for phase in ('native-worker','v4-worker'):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if phase=='native-worker' else '0',
                HF_HUB_OFFLINE='1',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            status=dict(phase='launch-'+phase,time=time.time(),steps=0,windows=0);write(OUT/'status.json',status)
            with (OUT/(phase+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),phase],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    time.sleep(15)
                    try:status=read(OUT/'status.json')
                    except (OSError,ValueError):pass
                    need(not stale_reason(status,time.time()),'ZERO_PROGRESS_WATCHDOG')
                need(child.returncode==0,'PRODUCER_WORKER_FAILED')
        receipt=read(CACHE/'receipt.json');need(receipt['status']=='QUALIFIED_ACTUAL_V4_CACHE','NO_CACHE_COMMIT')
        write(OUT/'RESULT.json',dict(status='CACHE800_COMPLETE_PENDING_B_CHECK',contract_id=reg['contract_id'],
              cache_receipt_sha256=sha(CACHE/'receipt.json'),seconds=time.perf_counter()-started,training_ready=False))
    finally:
        if child is not None and child.poll() is None:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=False)
            else:child.terminate()
            child.wait(timeout=30)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['register','cpu-smoke','readiness','admit','run','native-worker','v4-worker']);a=ap.parse_args()
    if a.action not in ('run','v4-worker'):os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        if a.action=='admit':admit_cpu();r=dict(status='CPU_ADMISSIBLE',physical_gpu_claim=False)
        else:r={'register':register,'cpu-smoke':cpu_smoke,'readiness':readiness,'run':run,
                'native-worker':native_worker,'v4-worker':v4_worker}[a.action]()
        if r is not None:print(json.dumps(r))
    except Exception as exc:
        code=str(exc) if isinstance(exc,ValueError) and str(exc).replace('_','').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED'
        r=dict(status='FAILED_OR_BLOCKED',error_type=type(exc).__name__,code=code,partial_preserved=True)
        if OUT.exists() and a.action in ('cpu-smoke','run','native-worker','v4-worker'):
            dest=OUT/(a.action+'-FAILURE.json')
            if not dest.exists():write(dest,r)
        print(json.dumps(r));raise SystemExit(1)


if __name__=='__main__':main()
