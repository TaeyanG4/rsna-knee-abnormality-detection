"""v1.0.0: fixed16 raw/header/cache engineering audit; no diagnoses or GPU.

Read-only completed inputs. Raw images/identities remain in private scratch;
only aggregate technical findings may be published. No full-cohort cap census.
"""
from __future__ import annotations
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pydicom
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.local/scratch/slot-b/k1-input-audit16-v1'
CACHE = Path('G:/temp_data/rsna-knee/l1rco-cache-full')
RAW = ROOT/'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip'
SOURCE = ROOT/'tools/exp189_s6_src/px_cache.py'
TRAINER = ROOT/'.local/scratch/l1-rco/k1v3-grouping/compact_train_l1.py'
GEOMETRY = ROOT/'.local/exp189-cpu-v120/input/cloud_code/training/dicom_geometry.py'
TAGS = ['SOPInstanceUID','InstanceNumber','ImagePositionPatient','ImageOrientationPatient',
        'PixelSpacing','Rows','Columns','Laterality','ImageLaterality','PhotometricInterpretation',
        'RescaleSlope','RescaleIntercept','BurnedInAnnotation']
PIXEL_TAGS = TAGS+['PixelData','BitsAllocated','BitsStored','HighBit','PixelRepresentation',
                  'SamplesPerPixel','PlanarConfiguration','NumberOfFrames']


def need(ok, code):
    if not ok: raise ValueError(code)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write(name, x):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,indent=2,allow_nan=False)


def slab_groups(n, spacing):
    groups=[[i] for i in range(n)]
    if 0 < spacing < 1.75:
        g=max(1,int(round(3.5/spacing)))
        n_new=n//g+(1 if n%g>=max(1,g//2) else 0)
        if n_new>=2: groups=[list(range(j*g,min(n,(j+1)*g))) for j in range(n_new)]
    return groups


def slab_source_oracle(n,spacing):
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='StudySlices')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='build')
    node=next(x for x in ast.walk(method) if isinstance(x,ast.If) and ast.unparse(x.test)=='0 < spacing < THIN_MM')
    ns=dict(np=np,spacing=spacing,THIN_MM=1.75,SLAB_MM=3.5,
            vol=np.arange(n,dtype=np.uint16)[:,None,None],pos=np.arange(n,dtype=float)*spacing,
            order=np.arange(n),order_rep=np.arange(n))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<bound-slab>', 'exec'),ns)
    return len(ns['pos']), ns['order_rep'].tolist()


def window_functions():
    tree=ast.parse(TRAINER.read_text(encoding='utf-8'))
    names={'covering_centers','phys_step','window_rows'}
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    need({x.name for x in nodes}==names,'WINDOW_SOURCE')
    ns={'math':math};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<bound-window>', 'exec'),ns)
    return ns


def pixel_check(x):
    need(x.shape==(384,384) and x.dtype==np.uint8,'PIXEL_LAYOUT')
    c=x[16:368,16:368]
    nz=np.count_nonzero(x)
    return dict(constant_full=bool(x.min()==x.max()),constant_crop=bool(c.min()==c.max()),
                all_zero_full=not bool(nz),all_zero_crop=not bool(np.count_nonzero(c)),
                nonzero_fraction=float(nz/x.size),
                nonzero_removed_fraction=float((nz-np.count_nonzero(c))/nz) if nz else 0.)


def tile(array,title,box=False):
    if array is None:
        im=Image.new('RGB',(192,192),'#333333')
    else:
        a=np.asarray(array)
        if a.dtype!=np.uint8:
            lo,hi=np.percentile(a,[1,99]);a=np.rint(np.clip((a-lo)/max(hi-lo,1e-8),0,1)*255).astype(np.uint8)
        im=Image.fromarray(a).convert('RGB').resize((192,192))
    canvas=Image.new('RGB',(192,215),'white');canvas.paste(im,(0,23));d=ImageDraw.Draw(canvas)
    d.text((3,4),title,fill='black')
    if box:d.rectangle((8,31,183,206),outline='red',width=2)
    return canvas


def run():
    started=time.perf_counter();need(not (OUT/'RESULT.json').exists(),'RESULT_EXISTS')
    registration=read(OUT/'REGISTRATION.json')
    need(sha(OUT/'selection.private.json')==registration['selection_sha256'],'SELECTION_HASH')
    selection=read(OUT/'selection.private.json');need(len(selection)==16,'SAMPLE_SIZE')
    reg=read(ROOT/'.local/scratch/l1-rco/fullcache/registration.json')
    cache_receipt=read(CACHE/'receipt.json');index=read(CACHE/'index.json')
    need(sha(CACHE/'index.json')==cache_receipt['hashes']['index.json'],'INDEX_HASH')
    need(reg['contract_id']==index['contract_id']==cache_receipt['contract_id'],'CACHE_CONTRACT')
    need(sha(SOURCE)==reg['pins']['tools/exp189_s6_src/px_cache.py'],'SLAB_SOURCE_HASH')
    need(sha(TRAINER)=='0d5fbae473eb1dc451871be0d610689d6baff8e6ac258acb091e66343ad0b791','TRAINER_HASH')
    spec=importlib.util.spec_from_file_location('audit_geometry',GEOMETRY)
    geom=importlib.util.module_from_spec(spec);sys.modules[spec.name]=geom;spec.loader.exec_module(geom)
    geom.cv2.setNumThreads(1)
    windows=window_functions()
    meta=np.load(CACHE/'meta.f32.npy',mmap_mode='r',allow_pickle=False)
    pixels=np.load(CACHE/'px.u8.npy',mmap_mode='r',allow_pickle=False)
    stats_before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in [CACHE/'meta.f32.npy',CACHE/'px.u8.npy']}
    wanted={x['study_uid'] for x in selection};cases=[];sheets=[];pixel_bytes=0;codecs=Counter();header_n=0
    archive_start=RAW.stat()
    with zipfile.ZipFile(RAW) as z:
        members={u:[] for u in wanted}
        for ent in z.infolist():
            p=ent.filename.split('/')
            if len(p)==4 and p[0]=='train_series' and p[1] in wanted and p[-1].endswith('.dcm'):
                members[p[1]].append(ent)
        series_bytes=z.read('train_series.csv')
        need(hashlib.sha256(series_bytes).hexdigest()==reg['series_sha256'],'SERIES_METADATA_HASH')
        records_by={u:[] for u in wanted}
        for row in csv.DictReader(io.StringIO(series_bytes.decode('utf-8'))):
            if row['StudyInstanceUID'] in wanted:
                records_by[row['StudyInstanceUID']].append({k:row[k] for k in ['SeriesInstanceUID','Anatomical_Plane']})
        for number,sel in enumerate(selection):
            uid=sel['study_uid'];token=sel['audit_token'];entries=members[uid]
            identity=sorted([[i.filename,i.CRC,i.file_size,i.compress_size] for i in entries])
            need(digest(identity)==reg['raw'][uid]['digest'],'RAW_IDENTITY')
            blocks=index['blocks'][uid];offset,total,block_count=index['done'][uid]
            shard=read(CACHE/'shards'/hashlib.sha256(uid.encode()).hexdigest()[:24]/'receipt.json')
            need(shard['contract_id']==reg['contract_id'] and shard['rows']==total and len(blocks)==block_count,'SHARD_IDENTITY')
            names={};orders={};tags=set();sag=[];header_errors=[];burned=set()
            raw_series={row['SeriesInstanceUID']:row['Anatomical_Plane'] for row in records_by[uid]}
            for sid,plane in raw_series.items():
                records=[]
                for ent in entries:
                    if ent.filename.split('/')[2]!=sid:continue
                    with z.open(ent) as stream:ds=pydicom.dcmread(stream,stop_before_pixels=True,specific_tags=TAGS)
                    header_n+=1;codecs[str(ds.file_meta.TransferSyntaxUID)]+=1
                    if str(getattr(ds,'BurnedInAnnotation','')).upper()=='YES':burned.add(sid)
                    records.append(geom.header_from_dataset(ds,ent.filename))
                    names[ent.filename]=ent
                if plane not in geom.CARDINAL_NORMAL:continue
                try:
                    order=geom.order_series(Path('private_series'),plane,provided_records=records)
                    orders[sid]=order;tags.update(order.laterality_values)
                    if plane=='Sagittal' and order.ipp_x_median is not None:sag.append(order.ipp_x_median)
                except Exception as exc:header_errors.append(type(exc).__name__)
            need(not header_errors,'RAW_GEOMETRY_FAILURE')
            side=geom.decide_study_side('private_study',tags,sag)
            cached_sids={b['series'] for b in blocks};missing=set(raw_series)-cached_sids
            need(cached_sids<=set(raw_series),'UNMAPPED_CACHE_SERIES')
            need(len(missing)==shard['removed'],'UNEXPLAINED_SERIES_REMOVAL')
            case=dict(token=token,raw_series=len(raw_series),cached_series=len(blocks),removed_duplicate_series=len(missing),
                      headers=len(entries),anchor_method=shard['anchor_method'],geometry=shard['geometry'],
                      side_source=side.source,series=[],window_count=0,boundary_errors=0,diagnostic_labels='NOT_READ')
            candidate_visual=None
            for bi,b in enumerate(blocks):
                sid=b['series'];plane=b['plane'];order=orders[sid]
                pos=np.array(order.positions_mm);native_order=np.arange(len(pos))
                if plane=='Sagittal' and side.side=='R':pos=-pos
                native_order=np.argsort(pos,kind='stable');pos=pos[native_order]
                spacing=float(np.median(np.diff(pos))) if len(pos)>1 else 0.
                groups=slab_groups(len(pos),spacing)
                need(len(groups)==b['n_slices'],'RAW_CACHE_SLICE_COUNT')
                used=sum(map(len,groups));expected_pos=np.array([pos[gr].mean() for gr in groups])
                expected_rel=2*(expected_pos-expected_pos[0])/max(expected_pos[-1]-expected_pos[0],1e-6)-1
                srec=dict(block=bi,plane=plane,raw_slices=len(pos),stored_slabs=len(groups),
                          native_spacing_mm=spacing,tail_slices_not_averaged=len(pos)-used,views=[])
                expected_views={0,2,3} if plane=='Coronal' else {0,1,4,5}
                need({v for v,_ in b['views']}==expected_views,'VIEW_SET')
                saved_images={}
                for v,start in b['views']:
                    end=start+b['n_slices'];need(offset<=start<end<=offset+total,'VIEW_BOUNDS')
                    mm=np.array(meta[start:end],copy=True)
                    need(np.isfinite(mm).all() and np.array_equal(mm[:,9],np.arange(len(mm))) and
                         np.all(mm[:,10]==len(mm)) and np.all(mm[:,3]==v) and np.all(mm[:,0]==bi),'METADATA_ORDER')
                    need(np.allclose(mm[:,4],expected_rel,atol=2e-6,rtol=0),'PHYSICAL_POSITION_MAPPING')
                    arr=np.array(pixels[start+len(mm)//2],copy=True);pixel_bytes+=arr.nbytes
                    need(pixel_bytes<=128*2**20,'PIXEL_READ_BUDGET')
                    srec['views'].append(dict(view=v,**pixel_check(arr)));saved_images[v]=arr
                srec['source_oracle']=slab_source_oracle(len(pos),spacing)[0]==len(groups)
                need(srec['source_oracle'],'SLAB_ORACLE')
                case['series'].append(srec)
                if candidate_visual is None or (plane=='Sagittal' and candidate_visual[0]!='Sagittal'):
                    candidate_visual=(plane,sid,order,native_order,groups,saved_images)
            as_blocks=[dict(b,views=[(0,v,st) for v,st in b['views']]) for b in blocks]
            wr=windows['window_rows'](as_blocks,lambda f,r:meta[r],'eval',3,3,3,3.5)
            intervals=[(st,st+b['n_slices']) for b in blocks for _,st in b['views']]
            for row in wr:
                matches=[(lo,hi) for lo,hi in intervals if lo<=row[1]<hi]
                need(len(matches)==1 and all(matches[0][0]<=v<matches[0][1] for v in row[2:]),'CROSS_SERIES_WINDOW')
            case['window_count']=len(wr)
            plane,sid,order,permutation,groups,ims=candidate_visual
            raw_image=None
            if sid not in burned:
                gr=groups[len(groups)//2];native_index=int(permutation[gr[len(gr)//2]])
                with z.open(names[order.files[native_index]]) as stream:
                    ds=pydicom.dcmread(stream,specific_tags=PIXEL_TAGS)
                payload=geom.canonicalize_payload(geom.pixels_from_dataset(ds,path='private_frame'),plane)
                raw_image=payload.image
                if geom.flip_for_plane(plane,side.side):raw_image=raw_image[:,::-1]
            case['visual_scope']=dict(plane=plane,one_representative_plane_only=True,raw_suppressed_burned_tag=sid in burned,
                                      exact_pixel_parity=False,raw='middle native member of middle cached slab; display normalization differs')
            panel=Image.new('RGB',(4*192,240),'white');ImageDraw.Draw(panel).text((4,3),token+' / '+plane+' / engineering QA only',fill='black')
            v_local=2 if plane=='Coronal' else 1;v_other=3 if plane=='Coronal' else 4
            show=ims if sid not in burned else {}
            for j,(arr,title,box) in enumerate([(raw_image,'Raw canonical',False),(show.get(0),'Saved full + crop box',True),
                                                  (show.get(v_local),'Saved local',True),(show.get(v_other),'Saved second local',True)]):
                panel.paste(tile(arr,title,box),(j*192,25))
            sheets.append(panel);cases.append(case)
            write(f'cases/{token}.json',case)
            print(json.dumps(dict(status='AUDITING',completed=number+1,total=16)),flush=True)
    for k in range(4):
        page=Image.new('RGB',(768,960),'white')
        for j,im in enumerate(sheets[k*4:k*4+4]):page.paste(im,(0,j*240))
        folder=OUT/'private-images';folder.mkdir(exist_ok=True);page.save(folder/f'sheet-{k+1}.png')
    stats_after={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in [CACHE/'meta.f32.npy',CACHE/'px.u8.npy']}
    need(stats_before==stats_after,'CACHE_CHANGED')
    need((archive_start.st_size,archive_start.st_mtime_ns)==(RAW.stat().st_size,RAW.stat().st_mtime_ns),'ARCHIVE_CHANGED')
    series=[s for c in cases for s in c['series']];views=[v for s in series for v in s['views']]
    result=dict(version='1.0.0',status='COMPLETE_ENGINEERING_AUDIT_PENDING_VISUAL_REVIEW',studies=16,
                raw_headers_read=header_n,raw_series=sum(c['raw_series'] for c in cases),cached_series=len(series),
                duplicate_series_removed=sum(c['removed_duplicate_series'] for c in cases),views_middle_frame_checked=len(views),
                eval_windows_boundary_checked=sum(c['window_count'] for c in cases),pixel_bytes_read=pixel_bytes,
                constant_full_views=sum(v['constant_full'] for v in views),constant_crops=sum(v['constant_crop'] for v in views),
                all_zero_full_views=sum(v['all_zero_full'] for v in views),all_zero_crops=sum(v['all_zero_crop'] for v in views),
                tail_omission_series=sum(s['tail_slices_not_averaged']>0 for s in series),
                tail_omission_native_slices=sum(s['tail_slices_not_averaged'] for s in series),
                native_slices_compared=sum(s['raw_slices'] for s in series),
                nonzero_removed_fraction_range=[min(v['nonzero_removed_fraction'] for v in views),max(v['nonzero_removed_fraction'] for v in views)],
                anchor_methods=dict(Counter(c['anchor_method'] for c in cases)),side_sources=dict(Counter(c['side_source'] for c in cases)),
                codecs=dict(codecs),known_identity_metadata_window_failures=0,actual_diagnoses=0,new_labels=0,
                model_or_gpu_calls=0,seconds=time.perf_counter()-started,
                bindings={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [SOURCE,TRAINER,GEOMETRY,Path(__file__),OUT/'REGISTRATION.json',OUT/'selection.private.json']},
                cache_receipt_sha256=sha(CACHE/'receipt.json'),cache_index_sha256=cache_receipt['hashes']['index.json'],
                limits=['Sparse engineering QA, not full-study reading or diagnosis.','No confirmed shared model errors or score effect.',
                        'Raw/cached visual normalization and slab averaging differ; no pixel-byte parity claim.',
                        'Clinical laterality/lesion preservation not adjudicated.','No historical training-bag replay or general coverage benefit claim.',
                        'B direct audit, not independent review.'])
    write('RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['bindings','limits','codecs']}),flush=True)
    return result


if __name__=='__main__':
    try: run()
    except Exception as e:
        safe=str(e) if isinstance(e,ValueError) and str(e).replace('_','').isupper() and len(str(e))<80 else 'DETAIL_SUPPRESSED'
        failure=dict(status='FAILED',type=type(e).__name__,code=safe,identifiers_suppressed=True)
        if not (OUT/'FAILURE.json').exists():write('FAILURE.json',failure)
        print(json.dumps(failure));sys.exit(1)
