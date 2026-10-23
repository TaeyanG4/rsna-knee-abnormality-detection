"""Training-only DICOM header, decoding and physical crop audit.

This is not a competition submission. It intentionally never reads test paths
or reports and never uses model weights. Runs in an Internet-OFF CPU notebook.
"""
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

START=time.monotonic()
OUT=Path('/kaggle/working')
ROOTS=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
ROOT=next((p for p in ROOTS if (p/'train.csv').exists()),None)
if ROOT is None:raise FileNotFoundError('Official training metadata not attached')
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
train=pd.read_csv(ROOT/'train.csv',usecols=['StudyInstanceUID']+TARGETS,dtype={'StudyInstanceUID':str})
series=pd.read_csv(ROOT/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
gold=train.loc[train[TARGETS].notna().all(1),'StudyInstanceUID'].sort_values().tolist()
weak=train.loc[train[TARGETS].isna().all(1),'StudyInstanceUID'].sample(n=16,random_state=20260912).tolist()
selected=gold+weak
subset=series.loc[series.StudyInstanceUID.isin(selected)].copy()
TAGS=['ImagePositionPatient','ImageOrientationPatient','InstanceNumber','Rows','Columns','PixelSpacing','SliceThickness','SpacingBetweenSlices','SeriesDescription','ScanningSequence','SequenceName','EchoTime','RepetitionTime','Manufacturer','ManufacturerModelName','MagneticFieldStrength']

def header_series(rec):
    folder=ROOT/'train_series'/rec['StudyInstanceUID']/rec['SeriesInstanceUID']
    paths=sorted(folder.glob('*.dcm'))
    result={**rec,'n_files':len(paths),'header_failures':0,'position_missing':0,'orientation_missing':0,'spacing_missing':0}
    slices=[];normals=[];first=None
    for path in paths:
        try:
            ds=pydicom.dcmread(path,stop_before_pixels=True,specific_tags=TAGS)
            if first is None:first=ds
            iop=np.asarray(getattr(ds,'ImageOrientationPatient',[]),dtype=float)
            ipp=np.asarray(getattr(ds,'ImagePositionPatient',[]),dtype=float)
            normal=np.cross(iop[:3],iop[3:]) if iop.shape==(6,) else None
            if normal is not None and np.linalg.norm(normal)>0:normal=normal/np.linalg.norm(normal)
            else:normal=None;result['orientation_missing']+=1
            if ipp.shape!=(3,) or not np.isfinite(ipp).all():ipp=None;result['position_missing']+=1
            spacing=np.asarray(getattr(ds,'PixelSpacing',[]),dtype=float)
            if spacing.shape!=(2,) or np.any(spacing<=0):result['spacing_missing']+=1
            projection=float(np.dot(ipp,normal)) if ipp is not None and normal is not None else None
            slices.append({'path':str(path),'filename':path.name,'position':projection,'ipp':ipp,'normal':normal,'instance':getattr(ds,'InstanceNumber',None)})
            if normal is not None:normals.append(normal)
        except Exception as exc:
            result['header_failures']+=1
    if first is not None:
        for tag in ['Manufacturer','ManufacturerModelName','MagneticFieldStrength','SeriesDescription','SliceThickness','SpacingBetweenSlices','Rows','Columns']:
            value=getattr(first,tag,None);result[tag]=str(value) if value is not None else None
        spacing=np.asarray(getattr(first,'PixelSpacing',[]),dtype=float)
        result['anisotropic_inplane']=bool(len(spacing)==2 and not np.isclose(spacing[0],spacing[1],rtol=1e-4))
        result['PixelSpacing']=spacing.tolist()
        result['transfer_syntax']=str(getattr(first.file_meta,'TransferSyntaxUID','missing'))
    geometric=[s for s in slices if s['position'] is not None]
    if len(geometric)==len(paths) and len(geometric)>1:
        reference=next(s['normal'] for s in geometric if s['normal'] is not None)
        result['orientation_min_abs_dot']=float(min(abs(np.dot(n,reference)) for n in normals))
        # Use one series normal, not an independently changing normal per slice.
        for s in geometric:s['position']=float(np.dot(s['ipp'],reference))
        ordered=sorted(geometric,key=lambda s:s['position'])
        xyz=np.asarray([s['position'] for s in ordered]);gaps=np.diff(xyz)
        result['median_projected_spacing_mm']=float(np.median(np.abs(gaps)))
        result['duplicate_positions']=int(np.isclose(gaps,0,atol=1e-4).sum())
        name_order=[s['filename'] for s in slices]
        geometry_order=[s['filename'] for s in ordered]
        result['filename_matches_geometry']=name_order==geometry_order
        result['filename_matches_geometry_reversed']=name_order==geometry_order[::-1]
        numbered=[s for s in slices if s['instance'] is not None]
        result['instance_matches_geometry']=len(numbered)==len(slices) and [s['filename'] for s in sorted(numbered,key=lambda s:float(s['instance']))]==geometry_order
        result['instance_matches_geometry_reversed']=len(numbered)==len(slices) and [s['filename'] for s in sorted(numbered,key=lambda s:float(s['instance']))]==geometry_order[::-1]
        result['geometry_sort_available']=True
    else:ordered=[];result['geometry_sort_available']=False
    return result,slices,ordered

with ThreadPoolExecutor(max_workers=8) as pool:
    audited=list(pool.map(header_series,subset.to_dict('records')))
table=pd.DataFrame([r for r,_,_ in audited]);table.to_csv(OUT/'train_series_geometry_audit.csv',index=False)
decode_errors=[];decode_rows=[]

def render(path,crop_mm=None):
    ds=pydicom.dcmread(path)
    arr=ds.pixel_array.astype(np.float32)
    if arr.ndim!=2:raise ValueError(f'Expected single frame, got {arr.shape}')
    arr=arr*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
    if crop_mm is not None:
        spacing=np.asarray(ds.PixelSpacing,dtype=float)
        if spacing.shape!=(2,) or np.any(spacing<=0):raise ValueError('Invalid crop spacing')
        nh,nw=np.maximum(1,np.round(float(crop_mm)/spacing).astype(int))
        h,w=arr.shape;cy,cx=h//2,w//2
        arr=arr[max(0,cy-nh//2):min(h,cy+(nh+1)//2),max(0,cx-nw//2):min(w,cx+(nw+1)//2)]
    lo,hi=np.percentile(arr,[1,99]);arr=np.clip((arr-lo)/max(float(hi-lo),1e-6),0,1)
    if str(getattr(ds,'PhotometricInterpretation',''))=='MONOCHROME1':arr=1-arr
    return Image.fromarray(np.uint8(arr*255)).resize((192,192))

montage_studies=gold[:4]+weak[:2]
made=[]
for uid in montage_studies:
    candidates=[x for x in audited if x[0]['StudyInstanceUID']==uid and x[0]['Fat_Suppression']==1 and x[2]]
    planes=[]
    for plane in ['Sagittal','Coronal','Axial']:
        choice=next((x for x in candidates if x[0]['Anatomical_Plane']==plane),None)
        if choice is not None:planes.append(choice)
    if not planes:continue
    canvas=Image.new('RGB',(6*192,len(planes)*220),(0,0,0));draw=ImageDraw.Draw(canvas)
    for row,(rec,filename_order,geometry_order) in enumerate(planes):
        pick=np.linspace(.25*(len(geometry_order)-1),.75*(len(geometry_order)-1),3).round().astype(int)
        draw.text((3,row*220),rec['Anatomical_Plane']+' | filename/no crop (left); physical/140mm (right)',fill='white')
        for col in range(6):
            entry=(filename_order if col<3 else geometry_order)[int(pick[col%3])]
            try:
                img=render(entry['path'],None if col<3 else 140)
                canvas.paste(img,(col*192,row*220+24))
                decode_rows.append({'study':uid,'series':rec['SeriesInstanceUID'],'file':entry['filename'],'transfer_syntax':rec.get('transfer_syntax'),'crop_mm':None if col<3 else 140,'success':True})
            except Exception as exc:
                error={'study':uid,'series':rec['SeriesInstanceUID'],'file':entry['filename'],'error':str(exc)[:300]}
                decode_errors.append(error);draw.text((col*192+4,row*220+80),'DECODE FAILED',fill='white')
    filename=f'train_montage_{len(made)+1:02d}.png';canvas.save(OUT/filename);made.append({'file':filename,'study':uid,'gold':uid in gold})
pd.DataFrame(decode_rows).to_csv(OUT/'decode_successes.csv',index=False)
summary={'created_utc':datetime.now(timezone.utc).isoformat(),'train_only':True,'gold_studies':len(gold),'weak_studies':len(weak),'series':len(table),
         'header_failures':int(table.header_failures.sum()),'geometrically_sortable_series':int(table.geometry_sort_available.sum()),
         'filename_matches_geometry':int(table.filename_matches_geometry.fillna(False).sum()),
         'filename_matches_geometry_reversed':int(table.filename_matches_geometry_reversed.fillna(False).sum()),
         'anisotropic_series':int(table.anisotropic_inplane.fillna(False).sum()),
         'transfer_syntax_counts':table.transfer_syntax.value_counts().to_dict(),'manufacturer_counts':table.Manufacturer.fillna('missing').value_counts().to_dict(),
         'decode_successes':len(decode_rows),'decode_failures':decode_errors,'montages':made,
         'elapsed_seconds':time.monotonic()-START,'pydicom_version':pydicom.__version__,
         'limitations':['Training subset only; not hidden-test-size runtime simulation','Left and right panels change both sorting and crop; diagnostic, not controlled performance ablation','Decode checks do not establish support for syntax absent from this subset','No clinical diagnosis from these images','No model inference or competition submission']}
(OUT/'geometry_audit_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
