"""HB177 post-fit v1.0.0: exact guarded loader and two-real-study CPU parity."""
from __future__ import annotations
import ast
import builtins
import copy
import gc
import os
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch
from tools import hb177_postfit_v1 as b


def extract(stage, dataset, arm, device='cpu'):
    """Execute only the hash/extract prelude, never the public serving main."""
    area=b.SCRATCH/('loader' if device=='cpu' else 'loader-gpu')/arm
    area.mkdir(parents=True,exist_ok=True)
    assets=area/'exp152-assets'
    # A rerun may reuse only our complete, hash-identical extraction. Never
    # remove files or accept a partial/foreign directory.
    reused=assets.exists()
    if reused:
        pins=b.read(stage/'build-receipt.json')['files']
        b.need(not assets.is_symlink() and {p.name for p in assets.iterdir()}==set(pins),'foreign extraction inventory')
        for name,pin in pins.items():
            p=assets/name
            b.need(p.is_file() and not p.is_symlink() and b.sha(p)==pin,'stale extraction: '+name)
    mkdir=Path.mkdir
    def checked_mkdir(path,*args,**kwargs):
        if reused and path==assets: return None
        return mkdir(path,*args,**kwargs)
    tree=ast.parse(''.join(b.read(stage/'kernel/release.ipynb')['cells'][5]['source']))
    body=tree.body[0].body
    stop=next(i for i,n in enumerate(body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
              and isinstance(n.value.func,ast.Name) and n.value.func.id=='exec')
    tree.body[0].body=body[:stop]
    original=builtins.__import__
    def imported(name,*args,**kwargs):
        if name==b.RUNTIME.removesuffix('.py'):
            mod=types.ModuleType('_hb177_cpu_'+arm)
            mod.__file__=str(area/'exp152-assets'/b.RUNTIME)
            exec(compile(Path(mod.__file__).read_bytes(),mod.__file__,'exec'),mod.__dict__)
            return mod
        return original(name,*args,**kwargs)
    def mapped(value,*parts):
        if str(value) in ('/kaggle/input/'+dataset.split('/')[1],'/kaggle/input/datasets/'+dataset): return stage/'dataset'
        if str(value)=='/kaggle/working/exp152-assets': return area/'exp152-assets'
        if str(value)=='/kaggle/working': return area
        return Path(value,*parts)
    ns={'FAST':False}; paths=sys.path[:]
    try:
        with patch.object(Path,'mkdir',checked_mkdir),patch('pathlib.Path',mapped),patch('builtins.__import__',imported):
            exec(compile(tree,'<exact-staged-cell5-prelude>','exec'),ns)
    finally: sys.path[:]=paths
    b.need('main' not in ns and not (area/'submission.csv').exists(),'serving body executed')
    return ns


def namespace(prelude):
    import numpy as np
    import torch
    import timm
    from tools.forensics_1015_v1 import definitions
    source=(prelude['_assets']/'public-serving.py').read_text(encoding='utf-8')
    snippets=definitions(source)
    ns=dict(np=np,torch=torch,nn=torch.nn,F=torch.nn.functional,timm=timm,gc=gc,INPUT_RES=320,
        IMG=384,NORM='imagenet',SLOT_SIZES=[26,22,18,12,18],SAGITTAL_SLOTS=(0,1),_WLUT={},
        _MEAN=torch.tensor([.485,.456,.406]).view(3,1,1),_STD=torch.tensor([.229,.224,.225]).view(3,1,1),
        _E152=prelude['_e152'],_E152_FILES=prelude['_files'],_E152_MANIFEST=copy.deepcopy(prelude['_manifest']))
    exec(compile('\n'.join(snippets.values()),'<exact-sealed-serving-definitions>','exec'),ns)
    return ns


def load(arm,device='cpu'):
    stage,dataset=(b.OUT,b.DATASET) if arm=='HB177' else (b.PARENT,'taeyangg4/knee-hb156-cnxs-l2-weights-v1')
    pre=extract(stage,dataset,arm,device); ns=namespace(pre)
    model,res=ns['load_model'](pre['_files'][b.NAME],b.ARCH,384,device)
    b.need(res==320 and not model.training,'model serving contract')
    return ns,model,pre


def infer(ns,model,vol,mask,device='cpu',reference=None):
    import torch
    rows=[]; values=[]
    with torch.inference_mode():
        for mirror in (False,True):
            x=ns['gpu_windows'](vol,mask,94,320,device,mirror=mirror)
            if device=='cpu':
                torch.testing.assert_close(x,ns['eval_windows'](vol,mask,94,320,mirror=mirror),atol=0,rtol=0)
                tick=time.perf_counter(); logits=model(x[None]).float(); elapsed=time.perf_counter()-tick
                values.append(logits.sigmoid()[0].numpy())
                error=None
                if reference is not None:
                    feats=torch.cat([reference.backbone(c) for c in x.split(16)]).view(1,94,-1)
                    expected=reference.head(feats)
                    torch.testing.assert_close(logits,expected,atol=1e-6,rtol=0)
                    error=float((logits-expected).abs().max())
                rows.append(dict(mirror=mirror,input_sha256=b.digest(x.numpy().tobytes()),shape=list(x.shape),max_abs=error,seconds=elapsed))
            else:
                values.append(ns['infer_probs'](model,x,device))
            del x
    import numpy as np
    result=.5*(np.asarray(values[0])+np.asarray(values[1]))
    b.need(result.shape==(12,) and np.isfinite(result).all() and ((result>=0)&(result<=1)).all(),'invalid probabilities')
    return result,rows


def run():
    b.cpu_only(); b.validate()
    from tools import hb177_gold58_v1 as gold
    reg=gold.validate()  # Registration precedes reading any arm predictions.
    from tools.hb158_scale_contrast_v1 import volume
    sys.path.insert(0,str(b.ROOT/'training/src'))
    from training.src.hb156_model_v1 import build_model
    import torch
    torch.set_num_threads(2)
    from tools import hb151_killtest_v102_fixtures as fixtures
    shim=types.SimpleNamespace(ROOT=b.ROOT,OUT=b.OUT,SCRATCH=b.SCRATCH,read=b.read)
    with patch.object(fixtures,'b',shim): fast=fixtures.run_smoke(b.SCRATCH/'fast-cases')
    reports={}; predictions={}
    for arm in ('HB177','HB156'):
        ns,model,pre=load(arm)
        reference=build_model(backbone=b.ARCH).eval()
        reference.load_state_dict(pre['_e152'].safe_load(pre['_files'][b.NAME])['model'],strict=True)
        reports[arm]=[]; predictions[arm]=[]
        for uid in reg['ids'][:2]:
            vol,mask,pins=volume(uid)
            value,rows=infer(ns,model,vol,mask,reference=reference)
            reports[arm].append(dict(study_sha256=b.digest(uid.encode()),pins=pins,windows=rows))
            predictions[arm].append(value.tolist())
        del model,reference,ns; gc.collect()
    need_empty=os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'CUDA_VISIBLE_DEVICES' in os.environ
    b.need(need_empty and not torch.cuda.is_initialized(),'CPU isolation')
    seconds={arm:sum(x['seconds'] for row in values for x in row['windows']) for arm,values in reports.items()}
    report=dict(version=b.VERSION,status='PASS',placeholder=False,studies=2,real_data=reports,
        fast_path=fast,build_receipt_sha256=b.sha(b.OUT/'build-receipt.json'),
        registration_sha256=b.sha(gold.REG),smoke_source_sha256=b.sha(__file__),
        cuda_visible_devices='',cuda_initialized=False,serving_seconds=seconds,
        serving_ratio=seconds['HB177']/seconds['HB156'],
        limitation='CPU fp32 cached real native volumes; CUDA, raw DICOM decode and hidden runtime not qualified.',
        qualified_for_submission=False)
    b.write(b.OUT/'cpu-smoke.json',b.encoded(report))
    b.write(b.SCRATCH/'cpu-predictions.json',b.encoded(predictions))
    return {k:v for k,v in report.items() if k not in ('real_data','fast_path')}
