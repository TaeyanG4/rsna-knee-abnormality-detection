"""H216 native8 v1.0.1; explicit audited OAI side adapter; reuse four completed cases. Original v1 is preserved."""
from __future__ import annotations
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'.local/scratch/slot-b/h216-oai-inputs-v1'
PREVIOUS=ROOT/'.local/scratch/slot-b/h216-oai-native8-v1'
OUT=ROOT/'.local/scratch/slot-b/h216-oai-native8-v101'
CODE=ROOT/'.local/exp189-cpu-v120/input/cloud_code'
SOURCE=ROOT/'tools/exp189_s6_src'


def need(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,obj):
    with Path(path).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)


def check(bindings):
    for p,h in bindings.items():need(sha(p)==h,'INPUT_OR_SOURCE_DRIFT')


def selected():
    result=read(INPUT/'RESULT.json')
    check(read(INPUT/'REGISTRATION.json')['sources'])
    check({str(INPUT/n):h for n,h in result['outputs'].items()})
    pilot=read(INPUT/'pilot8.private.json')['study_ids']
    need(len(pilot)==len(set(pilot))==8,'PILOT_ROSTER')
    cohort={r['study_key']:r for r in read(INPUT/'cohort.private.json')['studies']}
    need(set(pilot)<=set(cohort),'PILOT_NOT_ELIGIBLE')
    return pilot,{uid:cohort[uid]['series'] for uid in pilot}


def validate_native(study,expected):
    need(not study['errors'] and study['series'],'NATIVE_STUDY_ERROR')
    got={s['SeriesInstanceUID']:s for s in study['series']}
    need(len(got)==len(study['series']) and set(got)=={s['SeriesInstanceUID'] for s in expected},'SERIES_IDENTITY')
    for row in expected:
        s=got[row['SeriesInstanceUID']]
        need(not s['errors'],'NATIVE_SERIES_ERROR')
        need(s['plane']==row['Anatomical_Plane'] and s['slot']==row['slot'] and
             s['fat_suppression']==row['Fat_Suppression'] and s['fluid_sensitive']==row['Fluid_Sensitive'],
             'NATIVE_METADATA_DRIFT')
        need(len(s['positions'])==len(s['files'])==row['frame_count'],'NATIVE_FRAME_COUNT')
        need(s['side']==row['StudyInstanceUID'].split('_')[2],'NATIVE_SIDE')
    return got


def side_resolver(original, side_by_uid):
    def resolve(study, tag_values, sagittal_x):
        tags=tuple(tag_values); sagittal=tuple(sagittal_x)
        need(study in side_by_uid,'UNBOUND_OAI_SIDE')
        expected=side_by_uid[study]
        clean={v for v in tags if v in ('L','R')}
        need(not clean or clean=={expected},'DICOM_AUDIT_SIDE_CONFLICT')
        old=original(study,tags,sagittal)
        return type(old)(expected,'authenticated_OAI_side_join',old.tag_values,
                         old.sagittal_ipp_x_median)
    return resolve


def audited_sides(rows):
    from tools import h216_oai_inputs_v1 as inputs
    wanted={r['group_key']:uid for uid,rs in rows.items() for r in rs}; found=set(); sides={}
    with (inputs.AUDIT/'routing.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['group_key'] not in wanted:continue
            uid=wanted[r['group_key']];side=uid.split('_')[2]
            need(r['study_key']==uid and r['route_eligible'] and not r['routing_issues']
                 and not r['issues'],'SIDE_ROUTE_NOT_ADMITTED')
            need(r['laterality_basis'] in ('DICOM+image03+selection+MOAKS',
                 'image03+selection+MOAKS_DICOM_absent'),'UNVERIFIED_SIDE_JOIN')
            need(r['resolved_laterality'] in ('',side) and
                 set(r['header_description_sides'])<= {side},'AUDIT_SIDE_CONFLICT')
            sides[uid]=side;found.add(r['group_key'])
    need(found==set(wanted),'MISSING_SIDE_EVIDENCE')
    return sides


def register():
    need(not OUT.exists(),'OUTPUT_EXISTS')
    ids,rows=selected()
    from tools import oai_k1_preview_v1 as old
    for name,h in old.SOURCE_PINS.items():need(sha(SOURCE/name)==h,'PINNED_K1_DRIFT')
    paths=[Path(__file__),ROOT/'tools/hb144_oai_public_v1.py',ROOT/'tools/oai_k1_preview_v1.py',
           ROOT/'docs/research/h216-oai-native8-v101.md',ROOT/'tests/test_h216_oai_native8_v101.py',
           SOURCE/'probe_infer.py']
    paths += [CODE/'training'/n for n in ['native_cache.py','dicom_geometry.py','common.py','archive_cache.py']]
    paths += [INPUT/n for n in ['REGISTRATION.json','RESULT.json','cohort.private.json','pilot8.private.json']]
    from tools.extcache import oai_cache_v2
    a=oai_cache_v2.load_audit()
    sides=audited_sides(rows)
    prior=read(PREVIOUS/'REGISTRATION.json');check(prior['sources'])
    reuse={}
    for i,uid in enumerate(ids[:4],1):
        rp=PREVIOUS/(f'INPUT{i:02d}-RECEIPT.private.json');receipt=read(rp)
        sp=PREVIOUS/'native-private/studies'/(hashlib.sha256(uid.encode()).hexdigest()[:24]+'.json')
        need(sha(sp)==receipt['study_json_sha256'],'PREVIOUS_STUDY_DRIFT')
        study=read(sp);validate_native(study,rows[uid]);paths.extend([rp,sp])
        for s in study['series']:
            need(sha(s['path'])==receipt['volumes'][s['SeriesInstanceUID']]['sha256'],'PREVIOUS_VOLUME_DRIFT')
            paths.append(Path(s['path']))
        reuse[uid]=dict(receipt=str(rp),study=str(sp))
    paths.extend([PREVIOUS/'REGISTRATION.json',PREVIOUS/'FAILURE.json'])
    ar={r['SeriesInstanceUID']:a.archives[r['SeriesInstanceUID']] for rs in rows.values() for r in rs}
    frames=sum(r['frame_count'] for rs in rows.values() for r in rs)
    need(frames==1705 and len(ar)==31,'PINNED_PILOT_COUNTS')
    need(shutil.disk_usage(ROOT).free>100*2**30+frames*4*2**20,'CAPACITY_RESERVE')
    for record in ar.values():
        st=Path(record['path']).stat()
        need(st.st_size==record['size'] and st.st_mtime_ns==record['mtime_ns'],'ARCHIVE_DRIFT')
    OUT.mkdir(parents=True)
    write(OUT/'REGISTRATION.json',dict(version='1.0.1',ids=ids,rows=rows,archives=ar,
          sources={str(p):sha(p) for p in paths},side_by_uid=sides,reuse=reuse,
          cuda=False,models=False,full_v4_ready=False))


def run():
    reg=read(OUT/'REGISTRATION.json');check(reg['sources']);selected()
    need(not (OUT/'ATTEMPT.json').exists(),'ATTEMPT_EXISTS')
    write(OUT/'ATTEMPT.json',dict(version='1.0.1',started_unix=time.time()))
    from tools import hb144_oai_public_v1 as extraction
    from tools.extcache import oai_cache_v2
    a=oai_cache_v2.load_audit();data=OUT/'data';data.mkdir()
    rows=[r for rs in reg['rows'].values() for r in rs]
    with (data/'test_series.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (data/'test.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['StudyInstanceUID']);w.writerows([[u] for u in reg['ids']])
    # Install frozen native module only after metadata-only project helpers have loaded.
    sys.path.insert(0,str(CODE));sys.path.insert(0,str(SOURCE))
    import torch,cv2,numpy as np
    torch.set_num_threads(1);cv2.setNumThreads(1)
    need(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
    def forbidden(*a,**k):raise RuntimeError('MODEL_LOAD_FORBIDDEN')
    torch.load=forbidden
    import probe_infer
    native=probe_infer.install_native(True)
    native.geom.decide_study_side=side_resolver(native.geom.decide_study_side,reg['side_by_uid'])
    started=time.perf_counter();counts=[];out=OUT/'native-private'
    for i,uid in enumerate(reg['ids'],1):
        t=time.perf_counter();extract_receipts=[]
        if uid in reg['reuse']:
            old=reg['reuse'][uid];study=read(old['study']);validate_native(study,reg['rows'][uid])
            rp=OUT/(f'INPUT{i:02d}-RECEIPT.private.json');write(rp,dict(reused=True,source=old))
            oldr=read(old['receipt'])
            counts.append(dict(token=f'INPUT{i:02d}',series=len(study['series']),
                frames=sum(s['native_shape'][0] for s in study['series']),seconds=time.perf_counter()-t,
                volume_bytes=sum(v['bytes'] for v in oldr['volumes'].values()),reused=True))
            print(json.dumps(dict(completed=i,total=8,reused=True)),flush=True)
            continue
        case_data=PREVIOUS/'data' if i==5 else data
        for row in reg['rows'][uid]:
            g=a.groups[row['group_key']]
            extract_receipts.append(extraction.extract_group(a,g,case_data/'test_series'/uid/row['SeriesInstanceUID']))
        probe_infer.OneStudySource.current=uid
        with contextlib.redirect_stdout(io.StringIO()):
            native.prepare(case_data,out,split='test',study_ids=[uid],source_directory=True)
        study_path=out/'studies'/(hashlib.sha256(uid.encode()).hexdigest()[:24]+'.json')
        study=read(study_path);got=validate_native(study,reg['rows'][uid])
        volumes={};frames=0
        for sid,s in got.items():
            vol=np.load(s['path'],mmap_mode='r',allow_pickle=False)
            need(vol.dtype==np.uint16 and vol.ndim==3 and list(vol.shape)==s['native_shape'] and vol.size>0,'VOLUME_LAYOUT')
            need(vol.shape[0]==len(s['positions']) and np.isfinite(s['positions']).all(),'POSITIONS')
            need(np.isfinite(s['spacing_mm']).all() and np.all(np.asarray(s['spacing_mm'])>0),'SPACING')
            volumes[sid]=dict(sha256=sha(s['path']),bytes=Path(s['path']).stat().st_size,shape=list(vol.shape))
            frames+=len(vol);del vol
        token=f'INPUT{i:02d}'
        write(OUT/(token+'-RECEIPT.private.json'),dict(study_key=uid,extractions=extract_receipts,
              study_json_sha256=sha(study_path),volumes=volumes))
        counts.append(dict(token=token,series=len(got),frames=frames,seconds=time.perf_counter()-t,
                           volume_bytes=sum(v['bytes'] for v in volumes.values())))
        print(json.dumps(dict(completed=i,total=8,series=len(got),frames=frames)),flush=True)
    need(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
    check(reg['sources']);selected()
    result=dict(status='NATIVE8_PREPARED',version='1.0.1',studies=8,series=sum(c['series'] for c in counts),
         frames=sum(c['frames'] for c in counts),counts=counts,seconds=time.perf_counter()-started,
         volume_bytes=sum(c['volume_bytes'] for c in counts),reused_studies=4,newly_decoded_studies=4,
         cuda_initialized=False,model_calls=0,
         geometry_errors=0,full_v4_ready=False,training_ready=False,
         remaining='actual anatomy anchors/full-v4 renderer and paired training implementation',
         receipt_hashes={p.name:sha(p) for p in sorted(OUT.glob('INPUT*-RECEIPT.private.json'))})
    write(OUT/'RESULT.json',result)
    return {k:v for k,v in result.items() if k not in ('counts','receipt_hashes')}


if __name__=='__main__':
    try:
        if sys.argv[1:]==['register']:register();print('REGISTERED_NATIVE8_CPU_ONLY')
        elif sys.argv[1:]==['run']:print(json.dumps(run()))
        else:raise ValueError('ACTION')
    except Exception as e:
        print(json.dumps(dict(status='FAILED',error_type=type(e).__name__,code=str(e) if
              type(e) is ValueError and str(e).replace('_','').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED')))
        sys.exit(1)
