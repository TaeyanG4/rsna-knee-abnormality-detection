"""v1.0.1: private OAI eight-case K1 full-field preparation, no model or labels.

Full-field component only: not a v4 ROI training cache. Existing K1 numerical
source is reused unmodified. Images, mappings and arrays stay in local scratch.
v1.0.1 supplies the frozen ID-only test.csv virtually; original extraction has
only test_series.csv. First registration failed before writing or pixel reads.
"""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.local/scratch/slot-b/oai-k1-preview8-v1'
OLD = ROOT/'.local/scratch/astra-oai-extension/r6-oai'
DATA = OLD/'data'
RSNA = Path('G:/temp_data/rsna-knee/l1rco-cache-full')
CODE = ROOT/'.local/exp189-cpu-v120/input/cloud_code'
SOURCE = ROOT/'tools/exp189_s6_src'
SOURCE_PINS = {
    'px_cache.py':'b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632',
    'probe_infer.py':'a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980'}

def need(ok, code):
    if not ok: raise ValueError(code)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(p, data):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,allow_nan=False)

def select(ids):
    need(len(ids)==len(set(ids)) and len(ids)>=8,'ROSTER')
    return sorted(ids,key=lambda u:hashlib.sha256(('OAI-K1-PREVIEW-v1:'+u).encode()).hexdigest())[:8]

def positions(n):
    need(n>0,'EMPTY_SERIES')
    return sorted(set(round((n-1)*q) for q in (.1,.3,.5,.7,.9)))

def source_class(parent,ids):
    roster=('StudyInstanceUID\n'+'\n'.join(ids)+'\n').encode('utf-8')
    class Source(parent):
        def read(self,item):
            name=item.filename if hasattr(item,'filename') else item
            if name=='test.csv':return roster
            return super().read(item)
    return Source

def field_only(fn):
    def views(*a,**kw):
        result=[v for v in fn(*a,**kw) if v[0]==0]
        need(len(result)==1,'FULL_FIELD_VIEW')
        return result
    return views

def setup():
    sys.path.insert(0,str(CODE));sys.path.insert(0,str(SOURCE))
    import torch,cv2
    torch.set_num_threads(1);cv2.setNumThreads(1)
    need(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
    def forbidden(*a,**kw):raise RuntimeError('CHECKPOINT_LOAD_FORBIDDEN')
    torch.load=forbidden
    import probe_infer,px_cache
    return torch,probe_infer,px_cache

def display(x):
    import numpy as np
    x=np.asarray(x,dtype=np.float32);lo,hi=np.percentile(x,[1,99])
    return np.rint(np.clip((x-lo)/max(float(hi-lo),1e-6),0,1)*255).astype(np.uint8)

def sheet(items,path):
    from PIL import Image,ImageDraw
    need(bool(items),'EMPTY_SHEET')
    w=384;h=420;cols=min(4,len(items));rows=(len(items)+cols-1)//cols
    canvas=Image.new('RGB',(cols*w,rows*h),(22,22,22));draw=ImageDraw.Draw(canvas)
    for i,(title,array) in enumerate(items):
        x=(i%cols)*w;y=(i//cols)*h
        im=Image.fromarray(array).convert('RGB');im.thumbnail((384,384))
        canvas.paste(im,(x+(384-im.width)//2,y+30+(384-im.height)//2))
        draw.text((x+6,y+6),title,fill='white')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def register():
    need(not OUT.exists(),'OUTPUT_EXISTS')
    for name,h in SOURCE_PINS.items():need(sha(SOURCE/name)==h,'K1_SOURCE_DRIFT')
    reg=read(OLD/'registration.json');ids=select(reg['binding']['ids'])
    selected=read(ROOT/'.local/scratch/slot-b/k1-input-audit16-v1/selection.private.json')[:8]
    paths={str(SOURCE/n):h for n,h in SOURCE_PINS.items()}
    paths.update({str(p):sha(p) for p in [Path(__file__),CODE/'training/native_cache.py',
        CODE/'training/dicom_geometry.py',CODE/'training/input_provenance.py',
        CODE/'training/intensity_mapping.py',CODE/'training/native_data.py',
        OLD/'registration.json',OLD/'prepare.json',DATA/'test_series.csv']})
    OUT.mkdir(parents=True)
    write(OUT/'selection.private.json',dict(oai=ids,rsna=selected))
    write(OUT/'REGISTRATION.json',dict(version='1.0.1',oai_n=8,rsna_n=8,
        selection_sha256=sha(OUT/'selection.private.json'),sources=paths,
        scope='full-field384 + actual352 center crop; no v4 local ROI or training admission',
        cuda=False,checkpoints=False,added_denoising=False,clinical_labels=False))

def run():
    import numpy as np
    from PIL import Image
    need(not (OUT/'ATTEMPT.json').exists(),'ATTEMPT_ALREADY_EXISTS')
    r=read(OUT/'REGISTRATION.json')
    for p,h in r['sources'].items():need(sha(p)==h,'SOURCE_OR_INPUT_DRIFT')
    need(sha(OUT/'selection.private.json')==r['selection_sha256'],'SELECTION_DRIFT')
    write(OUT/'ATTEMPT.json',dict(started=time.time(),version='1.0.1'))
    selected=read(OUT/'selection.private.json');prep=read(OLD/'prepare.json')
    reg=read(OLD/'registration.json');torch,pi,px=setup()
    native=pi.install_native(True);px.series_views=field_only(px.series_views)
    native.DirectorySource=source_class(pi.OneStudySource,selected['oai'])
    counts=[];started=time.perf_counter();native_out=OUT/'native-private'
    for i,uid in enumerate(selected['oai'],1):
        token=f'OAI{i:02d}'
        files=list((DATA/'test_series'/uid).glob('*/*.dcm'))
        need(bool(files),'DICOM_MISSING')
        for p in files:
            rel=p.relative_to(DATA).as_posix()
            need(sha(p)==prep['data'].get(rel),'DICOM_HASH')
        pi.OneStudySource.current=uid
        with contextlib.redirect_stdout(io.StringIO()):
            native.prepare(DATA,native_out,split='test',study_ids=[uid],source_directory=True)
        key=hashlib.sha256(uid.encode()).hexdigest()[:24]
        st=read(native_out/'studies'/f'{key}.json')
        need(not st['errors'] and all(not s['errors'] for s in st['series']),'NATIVE_DECODE_OR_GEOMETRY')
        expected=sorted(x['SeriesInstanceUID'] for x in reg['binding']['routes'][uid])
        need(sorted(s['SeriesInstanceUID'] for s in st['series'])==expected,'NATIVE_SERIES_IDENTITY')
        ds=px.StudySlices(str(native_out),[uid],384,'v4',None);item=ds.build(uid)
        need(not item['error'] and not item['skipped_series'],'RENDER_ERROR')
        need(sorted(b['series'] for b in item['blocks'])==expected,'RENDER_SERIES_IDENTITY')
        a=item['px'].numpy();meta=item['meta'].numpy()
        need(a.dtype==np.uint8 and a.shape[1:]==(384,384) and np.isfinite(meta).all(),'ARRAY_LAYOUT')
        base=OUT/'oai-processed'/token;base.mkdir(parents=True)
        np.save(base/'fullfield384.npy',a,allow_pickle=False);np.save(base/'meta.npy',meta,allow_pickle=False)
        write(base/'blocks.private.json',item['blocks'])
        raw_tiles=[];proc_tiles=[];tails=0
        for j,b in enumerate(item['blocks'],1):
            s=next(s for s in st['series'] if s['SeriesInstanceUID']==b['series'])
            vol=np.load(s['path'],mmap_mode='r',allow_pickle=False)
            pos=np.asarray(s['positions'],float);spacing=abs(float(np.median(np.diff(np.sort(pos))))) if len(pos)>1 else 0
            group=max(1,round(3.5/spacing)) if 0<spacing<1.75 else 1
            nnew=len(vol)//group+(1 if len(vol)%group>=max(1,group//2) else 0)
            discarded=max(0,len(vol)-nnew*group) if nnew>=2 and group>1 else 0;tails+=discarded
            n=b['n_slices'];offset=b['views'][0][1]
            stack=a[offset:offset+n];need(np.ptp(stack)>0 and np.count_nonzero(stack)>0,'BLANK_SERIES')
            for k in positions(n):
                Image.fromarray(stack[k]).save(base/f'S{j:02d}-{b["plane"]}-{k:03d}-384.png')
                Image.fromarray(stack[k,16:368,16:368]).save(base/f'S{j:02d}-{b["plane"]}-{k:03d}-352.png')
            raw_tiles.append((f'{token} S{j:02d} {b["plane"]} native',display(vol[len(vol)//2])))
            proc_tiles.append((f'{token} S{j:02d} {b["plane"]} K1 full',stack[n//2]))
        sheet(raw_tiles,OUT/'oai-native-preview'/f'{token}.png')
        sheet(proc_tiles,OUT/'oai-processed'/f'{token}.png')
        row=dict(token=token,series=len(item['blocks']),dicoms=len(files),rows=len(a),slab_tail_omissions=tails,
                 fullfield_sha256=sha(base/'fullfield384.npy'),metadata_sha256=sha(base/'meta.npy'))
        counts.append(row);write(base/'RECEIPT.json',row)
        print(json.dumps(dict(phase='oai',completed=i,total=8,series=row['series'],rows=row['rows'])),flush=True)
    idx=read(RSNA/'index.json');arr=np.load(RSNA/'px.u8.npy',mmap_mode='r',allow_pickle=False)
    for i,row in enumerate(selected['rsna'],1):
        uid=row['study_uid'];tiles=[]
        for j,b in enumerate(idx['blocks'][uid],1):
            off=next(v[1] for v in b['views'] if v[0]==0);mid=arr[off+b['n_slices']//2]
            tiles.append((f'RSNA{i:02d} S{j:02d} {b["plane"]} K1 full',mid))
        sheet(tiles,OUT/'rsna-existing-preview'/f'RSNA{i:02d}.png')
    need(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
    result=dict(version='1.0.1',status='COMPLETE_FULLFIELD_PREVIEW_ONLY',oai_n=8,rsna_n=8,
        series=sum(x['series'] for x in counts),dicoms=sum(x['dicoms'] for x in counts),
        rows=sum(x['rows'] for x in counts),slab_tail_omissions=sum(x['slab_tail_omissions'] for x in counts),
        elapsed_seconds=time.perf_counter()-started,full_v4_ready=False,training_ready=False,
        checkpoint_loads=0,cuda_initialized=False,added_denoising=False,cases=counts)
    write(OUT/'RESULT.json',result)
    (OUT/'README.txt').write_text('PRIVATE MRI engineering previews\n\n'
        'rsna-existing-preview: copied display of existing K1 full-field inputs.\n'
        'oai-native-preview: orientation-corrected native OAI, per-frame display window; not raw DICOM replacement.\n'
        'oai-processed: exact K1 full-field384 component, with352 centre crops and arrays.\n'
        'native-private: original K1 native decoder outputs with private metadata.\n\n'
        'Eight OAI studies only. No extra denoising, diagnoses or new labels.\n'
        'Full v4 anatomical ROI views are NOT prepared; NOT training-ready.\n'
        'Slab averaging and display normalization can differ between previews.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)

if __name__=='__main__':
    try:
        if sys.argv[1:] == ['register']:register();print('REGISTERED_8_NO_PIXEL_READS')
        elif sys.argv[1:] == ['run']:run()
        else:raise ValueError('ACTION')
    except Exception as exc:
        if OUT.exists() and not (OUT/'FAILURE.json').exists():write(OUT/'FAILURE.json',dict(type=type(exc).__name__,code=str(exc) if str(exc).isupper() else 'PRIVATE_EXCEPTION'))
        print(json.dumps(dict(status='FAILED',type=type(exc).__name__,code=str(exc) if str(exc).isupper() else 'PRIVATE_EXCEPTION')))
        raise SystemExit(1)
