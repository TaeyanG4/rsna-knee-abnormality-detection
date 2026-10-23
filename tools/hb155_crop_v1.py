"""H-B-155 compartment cache v1.2.2 (2026-10-15).

Changelog 1.2.2: tolerate <1e-7 pixel affine border roundoff, no geometric relaxation.
1.2.1: cap physical square width to smaller series FOV; never synthesize borders.
1.2.0: deterministic series-FOV fallback, affine oblique support,
bounded stack windows and translated crop boxes; absent planes stay masked.
1.1.0: distinguish absent from contradictory laterality.

Geometry-only R7 slab prior: no segmentation, labels, GPU or network. Native
LPS grids retain patient geometry. Unknown laterality is explicitly unnamed
low/high-LPS-X compartments, never silently called medial/lateral.
"""
from __future__ import annotations
import argparse, hashlib, io, json, time, shutil
from pathlib import Path
import numpy as np
from tools.hb150_io_v1 import atomic_json, safe_replace
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.local/scratch/p65q-v1116/H-B-155'
SOURCE=ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02'
CONTRACT='hb155_compartment64_448_v1'
VERSION='1.2.2'
QA_WORK=ROOT/'.local/scratch/slot-b/hb155'
GEOMETRY_POLICY=dict(version=VERSION,centre='R7 slab prior; own FOV if out of volume or oblique',
    orientation='unit orthogonal IOP, expected dominant normal axis, parallel slices',
    compartments='LPS X +/-18 mm; translated inside each image',
    stack='translate full window inside slab; shrink only if slab is shorter',
    padding='translate box inside FOV; width=min(100 mm, row FOV, column FOV); no synthetic pixels',
    missing_plane='zero pixels and mask; never substitute another plane',
    laterality='unknown uses low/high LPS X; contradictory metadata remains a QA failure')

def compartment_name(plane, compartment, side):
    if plane=='Axial': return 'axial_center'
    if side not in ('L','R'): return 'low_lps_x' if compartment==0 else 'high_lps_x'
    return 'medial' if (compartment==0)==(side=='L') else 'lateral'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def project(aff, point):
    """R7 nearest-plane LPS projection, with explicit singular geometry rejection."""
    aff=np.asarray(aff,float); point=np.asarray(point,float)
    if aff.ndim!=3 or aff.shape[1:]!=(3,3) or not np.isfinite(aff).all():
        raise ValueError('invalid_geometry')
    n=np.cross(aff[0,:,0],aff[0,:,1]); norm=np.linalg.norm(n)
    if norm<1e-8: raise ValueError('singular_geometry')
    d=(aff[:,:,2]-point)@(n/norm); z=int(np.argmin(abs(d)))
    cr=np.linalg.lstsq(aff[z,:,:2],point-aff[z,:,2],rcond=None)[0]
    return z,float(cr[0]),float(cr[1]),float(abs(d[z]))

def outside_slab(aff,point):
    n=np.cross(aff[0,:,0],aff[0,:,1]); n/=np.linalg.norm(n)
    positions=np.sort(aff[:,:,2]@n); gaps=np.diff(positions)
    tolerance=float(np.median(gaps[gaps>1e-5]))/2 if np.any(gaps>1e-5) else 0.
    return bool(point@n<positions[0]-tolerance or point@n>positions[-1]+tolerance)

def slab_prior(series):
    """Exact R7 slab-prior axis policy, only accepted orthogonal geometry."""
    per={}
    for s in series:
        if s.get('oblique',False): continue
        a=s['aff']; h,w=s['shape']; z=len(a)//2
        per.setdefault(s['plane'],[]).append(a[z]@np.array([(w-1)/2,(h-1)/2,1.]))
    if not per.get('Sagittal') or not per.get('Coronal'): raise ValueError('missing_orthogonal_prior')
    return np.array([np.median([c[axis] for p in planes for c in per[p]])
                     for axis,planes in enumerate((('Sagittal',),('Coronal',),('Sagittal','Coronal')))])


def series_center(series, prior):
    """Never project a different acquisition's out-of-volume prior onto an edge."""
    a=series['aff']; h,w=series['shape']; reason=None
    if prior is None: reason='missing_orthogonal_prior'
    elif series.get('oblique',False): reason='oblique_series'
    else:
        _,x,y,_=project(a,prior)
        if not (0<=x<w and 0<=y<h): reason='prior_outside_fov'
        elif outside_slab(a,prior): reason='prior_outside_slab'
    if reason:
        centers=np.einsum('nij,j->ni',a,np.array([(w-1)/2,(h-1)/2,1.]))
        return np.median(centers,axis=0),reason
    return np.asarray(prior,float).copy(),'slab_prior'


def stack_points(aff, point, span, count):
    """Shift then shrink the requested window, instead of duplicating edge slices."""
    normal=np.cross(aff[0,:,0],aff[0,:,1]); normal/=np.linalg.norm(normal)
    positions=aff[:,:,2]@normal; lo,hi=float(positions.min()),float(positions.max())
    radius=min(float(span),(hi-lo)/2)
    middle=float(np.clip(point@normal,lo+radius,hi-radius))
    coords=np.linspace(middle-radius,middle+radius,count) if count>1 else np.array([middle])
    requested=point@normal+np.linspace(-span,span,count)
    points=[point+normal*(x-point@normal) for x in coords]
    return points,dict(window_shift_mm=middle-float(point@normal),span_mm=2*radius,
        requested_outside_slab=int(((requested<lo)|(requested>hi)).sum()))

def crop(image, aff, point, width_mm=100., size=448):
    import cv2
    _,c,r,_=project(aff[None],point)
    sy=np.linalg.norm(aff[:,1]); sx=np.linalg.norm(aff[:,0])
    h,w=image.shape
    width_mm=min(width_mm,h*sy,w*sx)
    hh=min(h,max(1,round(width_mm/sy))); ww=min(w,max(1,round(width_mm/sx)))
    y0=round(r-hh/2); x0=round(c-ww/2)
    # Reprojection of a clamped oblique affine may produce -1e-14 at a border.
    # Tolerate numerical roundoff only, never an anatomical out-of-field point.
    if not (-1e-7<=r<h and -1e-7<=c<w): raise ValueError('anchor_outside_field')
    # Preserve a square physical FOV, capped to real coverage. Translate the
    # box inside the image instead of creating black/replicated borders.
    y0=int(np.clip(y0,min(0,h-hh),max(0,h-hh)))
    x0=int(np.clip(x0,min(0,w-ww),max(0,w-ww)))
    ya,yb=max(0,y0),min(h,y0+hh); xa,xb=max(0,x0),min(w,x0+ww)
    buf=image[ya:yb,xa:xb]
    fraction=1-(yb-ya)*(xb-xa)/(hh*ww)
    return cv2.resize(buf,(size,size),interpolation=cv2.INTER_AREA),[x0,y0,ww,hh],fraction

def read_series(raw, members, plane, sid):
    records=[]; sides=set()
    expected={'Sagittal':0,'Coronal':1,'Axial':2}[plane]
    for path,info in members:
        d=raw._WORKER['pydicom'].dcmread(io.BytesIO(raw._member_bytes(path,info)),stop_before_pixels=True)
        iop=np.asarray(getattr(d,'ImageOrientationPatient',[]),float)
        ipp=np.asarray(getattr(d,'ImagePositionPatient',[]),float)
        ps=np.asarray(getattr(d,'PixelSpacing',[]),float)
        if iop.shape!=(6,) or ipp.shape!=(3,) or ps.shape!=(2,) or not np.isfinite(np.r_[iop,ipp,ps]).all() or min(ps)<=0:
            raise ValueError('missing_geometry')
        u,v=iop[:3],iop[3:]; normal=np.cross(u,v)
        if (abs(np.linalg.norm(u)-1)>.01 or abs(np.linalg.norm(v)-1)>.01 or
                abs(np.dot(u,v))>.01): raise ValueError('invalid_geometry')
        if int(np.argmax(abs(normal)))!=expected: raise ValueError('oblique_geometry')
        aff=np.column_stack((u*ps[1],v*ps[0],ipp))
        side=str(getattr(d,'ImageLaterality',getattr(d,'Laterality',''))).upper()
        if side in ('L','R'): sides.add(side)
        records.append((float(ipp@normal),path,info,aff,(int(d.Rows),int(d.Columns))))
    if not records: raise ValueError('empty_series')
    records.sort(key=lambda r:(r[0],r[1]))
    a=np.stack([r[3] for r in records])
    normals=np.cross(a[:,:,0],a[:,:,1]); normals/=np.linalg.norm(normals,axis=1)[:,None]
    if np.min(normals@normals[0])<.999 or any(r[4]!=records[0][4] for r in records):
        raise ValueError('invalid_geometry')
    return dict(plane=plane,sid=sid,records=records,aff=a,shape=records[0][4],sides=sides,
        oblique=bool(abs(normals[0,expected])<.85))

def build_study(raw, task, spec, overlay_dir=None):
    split,uid,rows=task; failures={}; selected=[]; used=set()
    def fail(reason): failures[reason]=failures.get(reason,0)+1
    members=raw._WORKER['con'].execute('SELECT path,series,info FROM files WHERE split=? AND uid=? ORDER BY path',(split,uid)).fetchall()
    by={}
    for p,s,i in members: by.setdefault(s,[]).append((p,i))
    for plane,fluid,k in spec['slots']:
        r=raw.pick_series(rows,plane,fluid,used)
        if r is None: selected.append(None); fail('missing_series'); continue
        sid=r['SeriesInstanceUID']; used.add(sid)
        try: selected.append(read_series(raw,by.get(sid,[]),plane,sid))
        except Exception as exc: selected.append(None); fail(str(exc) if isinstance(exc,ValueError) else 'header_decode_error')
    vol=np.zeros((64,448,448),np.uint8); provenance=[]
    try: center=slab_prior([s for s in selected if s])
    except ValueError:
        center=None  # Each available series still has an authenticated own FOV.
    sides=set().union(*(s['sides'] for s in selected if s)); side=next(iter(sides)) if len(sides)==1 else None
    if side is None: fail('laterality_conflict' if len(sides)>1 else 'unknown_laterality')
    offset=0
    for slot,(plane,fluid,k) in enumerate(spec['slots']):
        s=selected[slot]; entries=[]; localization=None; windows=[]
        if s:
            local_center,localization=series_center(s,center)
            points=[]
            for comp in ((0,1) if plane!='Axial' else (0,)):
                pt=local_center.copy()
                if plane!='Axial': pt[0]+=(-18 if comp==0 else 18)
                half=k//2 if plane!='Axial' else k
                span=12 if plane=='Sagittal' else 24
                targets,window=stack_points(s['aff'],pt,span,half); windows.append(window)
                for target in targets:
                    z,x,y,distance=project(s['aff'],target)
                    h,w=s['shape']
                    bounded=s['aff'][z]@np.array([np.clip(x,0,w-1),np.clip(y,0,h-1),1.])
                    clipped=outside_slab(s['aff'],target)
                    if clipped: fail('clamped_stack_target')
                    points.append((z,bounded,comp,distance,clipped))
            decoded={}
            for z,pt,comp,_,_ in points:
                if z not in decoded:
                    try: decoded[z]=raw._read_pixels(s['records'][z][1],s['records'][z][2])
                    except Exception: decoded[z]=None; fail('pixel_decode_error')
            valid=[a for a in decoded.values() if a is not None]
            lo,hi=np.percentile(np.concatenate([a.ravel() for a in valid]),[2,98]) if valid else (0.,1.)
            for j,(z,pt,comp,distance,clipped) in enumerate(points):
                a=decoded[z]
                name=compartment_name(plane,comp,side)
                if a is None: continue
                try:
                    normalized=np.clip((a-lo)/(hi-lo+1e-6),0,1)
                    rendered,box,pad=crop(normalized,s['aff'][z],pt)
                    vol[offset+j]=(rendered*255).astype(np.uint8)
                    if pad>0: fail('padded_crop')
                    entries.append(dict(slice=z,compartment=name,box=box,padding_fraction=pad,
                        effective_fov_mm=[box[2]*float(np.linalg.norm(s['aff'][z,:,0])),
                                          box[3]*float(np.linalg.norm(s['aff'][z,:,1]))],
                        nearest_plane_distance_mm=distance,clamped_stack_target=clipped))
                    if overlay_dir and (j==0 or (plane!='Axial' and j==k//2)):
                        import cv2
                        overlay_dir.mkdir(parents=True,exist_ok=True)
                        im=cv2.cvtColor((normalized*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                        x,y,w,h=box; cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                        ok,data=cv2.imencode('.png',im)
                        if ok: (overlay_dir/f'{slot}-{plane}-{name}.png').write_bytes(data.tobytes())
                except ValueError as exc: fail(str(exc))
        provenance.append(dict(plane=plane,series=None if s is None else s['sid'],crops=entries,
            localization=localization,windows=windows,
            center=None if s is None else local_center.tolist())); offset+=k
    return vol,(vol.reshape(64,-1).sum(1)>0).astype(np.uint8),dict(failures=failures,slots=provenance,laterality=side or 'unknown',anchor=None if center is None else center.tolist(),method='R7_slab_prior_with_bounded_series_FOV_fallback')

def build(limit=3, out=None):
    from training.src import raptor_corpus as raw
    out=Path(out) if out else WORK/('crop-smoke' if limit else 'crop-cache')
    if not out.resolve().is_relative_to(WORK.resolve()): raise ValueError('output outside fixed scratch')
    source=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    if sha(SOURCE/'manifest.json')!='f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e': raise ValueError('source identity drift')
    for name in ('ids','masks'):
        if sha(SOURCE/(name+'.npy'))!=source[name+'_sha256']: raise ValueError('source roster identity drift')
    ids=np.load(SOURCE/'ids.npy',allow_pickle=False)
    picks={}
    with (SOURCE/'picks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            d=json.loads(line); picks[d['uid']]=d['split']
    if limit: ids=ids[:limit]
    out.mkdir(parents=True,exist_ok=True)
    required=len(ids)*64*448*448+2*(1<<30)
    # Conservative; never start a full corpus write on a nearly full volume.
    if shutil.disk_usage(out).free<required: raise ValueError('insufficient_crop_build_disk')
    spec=dict(source['spec']); spec.update(img=448,crop_mm=100.,compartment_offset_mm=18.,
        span_mm={'Sagittal':[-12,12],'Coronal':[-24,24],'Axial':[-24,24]},
        localization='R7 slab prior, geometry-only; no tissue confirmation')
    atomic_json(out/'manifest.json',dict(contract=CONTRACT,finished=False,version=VERSION))
    zip_path,index,_=raw.archive_paths(); raw._init_worker(str(zip_path),str(index),spec)
    recs=raw.series_records(); masks=np.zeros((len(ids),64),np.uint8); failures={}; shards=[]; failed_studies=0; started=time.perf_counter()
    try:
        for begin in range(0,len(ids),100):
            stop=min(begin+100,len(ids)); name=f'vols_shard{begin//100:02d}.npy'; tmp=out/(name+'.partial')
            arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(stop-begin,64,448,448))
            records=[]
            for row,uid in enumerate(ids[begin:stop]):
                split=picks[str(uid)]
                v,m,p=build_study(raw,(split,str(uid),recs[(split,str(uid))]),spec,out/'overlays'/str(begin+row) if begin+row<3 else None)
                arr[row]=v; masks[begin+row]=m; records.append(dict(uid=str(uid),**p))
                failed_studies+=bool(p['failures'])
                for reason,count in p['failures'].items(): failures[reason]=failures.get(reason,0)+count
            arr.flush(); del arr; safe_replace(tmp,out/name)
            atomic_json(out/f'qa-shard{begin//100:02d}.json',records)
            shards.append(dict(start=begin,stop=stop,file=name,done=True,bytes=(out/name).stat().st_size,sha256=sha(out/name)))
    finally: raw._WORKER['zip'].close(); raw._WORKER['con'].close()
    np.save(out/'ids.npy',ids); np.save(out/'masks.npy',masks)
    manifest=dict(contract=CONTRACT,spec=spec,studies=len(ids),shape_total=[len(ids),64,448,448],shards=shards,finished=True,
        ids_sha256=sha(out/'ids.npy'),masks_sha256=sha(out/'masks.npy'),source_manifest_sha256=sha(SOURCE/'manifest.json'),source_picks_sha256=sha(SOURCE/'picks.jsonl'),archive=source['archive'],builder_sha256=sha(__file__),raw_builder_sha256=sha(raw.__file__),version=VERSION,
        localization='R7 slab_prior; geometry-only; segmentation weights not used',failure_counts=failures,
        scientific_admission='HOLD_GEOMETRY_AND_INDEPENDENT_REVIEW',seconds=time.perf_counter()-started)
    atomic_json(out/'manifest.json',manifest)
    overlays={str(p.relative_to(out)):sha(p) for p in (out/'overlays').rglob('*.png') if any(k in p.name for k in ('medial','lateral','low_lps_x','high_lps_x','axial_center'))}
    result=dict(studies=len(ids),studies_with_failures=failed_studies,valid_slices=int(masks.sum()),empty_studies=int((masks.sum(1)==0).sum()),failure_counts=failures,overlay_sha256=overlays,seconds=manifest['seconds'],bytes=sum(s['bytes'] for s in shards),output=str(out.relative_to(ROOT)))
    atomic_json(out/'qa-summary.json',result); return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=3); ap.add_argument('--out'); a=ap.parse_args()
    if a.limit<0: ap.error('nonnegative limit required')
    print(json.dumps(build(a.limit,a.out)))
