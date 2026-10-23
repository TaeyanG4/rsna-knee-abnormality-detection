"""H-B-175 CPU smoke v1.0.0: real pixels, both four-fold bundles, frozen flip/stackers.

Reduced windows are not full-study predictions or timing qualification.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r224r76fq_v1131 as b
from tools.c85r7_flipall_cpu_smoke import node
from tools.hb145_output_io import atomic_json


def runtime():
    """Import the actual archive, including fallback imports, outside tools namespace."""
    directory = b.SCRATCH/'candidate-runtime'
    with patch.dict(sys.modules, {'tools': None}), patch.object(sys, 'path', [str(directory), *sys.path]):
        return importlib.import_module('exp169_release_c85r7codec')


def run():
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import pydicom
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    val = b.validate()
    mod = runtime()
    source_dir = b.SCRATCH/'candidate-runtime/exp169_r7_src'
    source = (source_dir/'compact_infer.py').read_text(encoding='utf-8')
    spec = importlib.util.spec_from_file_location('hb175_trainer', source_dir/'compact_train.py')
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    data = ROOT/'.local/exp189-cpu-v120/input/visible_data/test_series'
    ids = sorted(p.name for p in data.iterdir() if p.is_dir())
    b.old.need(len(ids) == 3, 'expected real3')
    images, pins = [], {}
    for uid in ids:
        paths = sorted((data/uid).rglob('*.dcm'))
        mid = len(paths)//2
        pixels = []
        for path in paths[mid-1:mid+2]:
            pixels.append(pydicom.dcmread(path).pixel_array.astype('float32'))
            pins[str(path.relative_to(ROOT))] = b.old.digest(path.read_bytes())
        arr = np.stack(pixels)
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip((arr-lo)/max(hi-lo, 1)*255, 0, 255)
        img = F.interpolate(torch.from_numpy(arr)[None], size=(384,384), mode='bilinear', align_corners=False)[0]
        # Three views of this study's real window; preserve study separation.
        images.append(img[:,16:368,16:368].to(torch.uint8)[None].repeat(3,1,1,1)[None])
    meta = torch.zeros((1,3,9)); meta[0,:,7] = torch.tensor([0,1,2])
    meta[0,:,1] = torch.tensor([0,2,3]); meta[0,:,2] = .75; meta[0,:,6] = 1
    mask = torch.ones((1,3), dtype=torch.bool)
    bundles, predictions, started = {}, {}, time.perf_counter()
    for label, folder, bid, slug, pin in [
        ('R7', ROOT/'storage/external/heliosli-r7cnxfsss-v1', mod.r7.BUNDLE_ID, mod.r7.DATASET_SLUG,
         'c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6'),
        ('R6', b.R6_LOCAL, b.R6_ID, b.R6_PRIVATE, b.R6_SHA)]:
        manifest, identity = mod.r7.verify_bundle(folder, pin, source_dir, bundle_id=bid, dataset_slug=slug)
        assert '--flip-tta' in mod.r7.worker_args(manifest, folder, 'anatomy.pt')
        state = b.old.read(folder/'state_spec.json')
        ns = dict(torch=torch, np=np, math=math, json=json, hashlib=hashlib, Path=Path,
                  LABELS=ct.LABELS, MIRROR_PAIRS=(("Medial Meniscus","Lateral Meniscus"),("Medial OA","Lateral OA")))
        for name in ('mirror_perm','mirror_windows','state_layout','state_readout','score_v4c',
                     'fluid_stacker_load','fluid_stack','state_stacker_load','apply_stacker','apply_state_stacker'):
            exec(compile(ast.Module(body=[node(source,name)],type_ignores=[]), '<frozen-worker>', 'exec'), ns)
        # Frozen loader uses the worker's declared D2 volume whitelist.
        tree = ast.parse(source)
        for item in tree.body:
            if isinstance(item, ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('FLUID_') for t in item.targets):
                exec(compile(ast.Module(body=[item],type_ignores=[]), '<fluid-constants>', 'exec'), ns)
        bins = [(k,x) for k,v in state['binaries'].items() for x in v]
        perm,bp = ns['mirror_perm'](ct.LABELS,state,bins)
        layout = ns['state_layout'](state,ct.LABELS)
        mcl = ct.LABELS.index('MCL')
        cols = [i for c in layout['columns'] if c['class']=='MCL'
                for i in (range(c['start'],c['stop']) if 'categories' in c else [c['column']])]
        ns.update(FLIP=dict(perm=perm,bin_perm=bp,mcl=mcl), MCL_STATE_COLS=cols, EXTRA={})
        fold_z, fold_state, checks = [[] for _ in ids], [[] for _ in ids], []
        calls=[]
        def one_pass(model,img,mt,valid):
            value=model(img,mt,valid,chunk=32,return_aux=True); calls.append(value); return value
        ns['one_pass']=one_pass
        with torch.inference_mode():
            for index,name in enumerate(manifest['models']):
                sd=torch.load(folder/name,map_location='cpu',weights_only=True)
                model=ct.Net(manifest['backbone'],None,crop=352,state_spec=state,in_chans=3,
                             spatial='sp_q' in sd,extent='ext_u' in sd,syn_decomp='syn_beta' in sd)
                model.load_state_dict(sd,strict=True); model.eval().requires_grad_(False)
                for j,img in enumerate(images):
                    calls.clear()
                    actual=ns['score_v4c']([model],img,meta,mask)[0]
                    (z,sl,bl),(zm,slm,blm)=calls
                    expected=(z[0]+zm[0,perm])/2; expected[mcl]=z[0,mcl]
                    assert torch.equal(actual,expected) and torch.isfinite(actual).all()
                    plain=ns['state_readout'](model,sl,bl)[0]
                    mirror=ns['state_readout'](model,slm[:,perm],blm[:,bp])[0]
                    st=(plain+mirror)/2; st[cols]=plain[cols]
                    assert torch.equal(ns['EXTRA']['group1']['state'][0],st)
                    fold_z[j].append(actual.clone()); fold_state[j].append(st.clone())
                checks.append(dict(fold=index,studies=3,flip_and_mcl_exact=True))
                print(json.dumps(dict(bundle=label,fold=index,status='PASS')),flush=True)
                del model,sd
        ns.update(STK=ns['fluid_stacker_load'](folder/manifest['fluid_stacker'],1),
                  SST=ns['state_stacker_load'](folder/manifest['state_stacker'],ct.LABELS,layout,
                                              hashlib.md5((folder/'state_spec.json').read_bytes()).hexdigest()),
                  FL=dict(studies=0,with_volume={},missing={},status={},fluid_s=0.,stack_fallback={},first=[]),
                  SS=dict(studies=0,stack_fallback={},first=[]),
                  swr=csv.writer(io.StringIO()),sfh=io.StringIO(),fwr=csv.writer(io.StringIO()),ffh=io.StringIO())
        ns.update(models=[None]*4, models2=[])
        ns['FL']['with_volume']={k:0 for k in ns['STK']['classes']}
        ns['FL']['missing']={k:0 for k in ns['STK']['classes']}
        rows=[]
        for j,uid in enumerate(ids):
            zs=torch.stack(fold_z[j]); value=torch.sigmoid(zs.mean(0).double()).numpy()
            ns['EXTRA']={'group1':{'state':fold_state[j]}}
            # Missing fluid is an existing supported worker path; no synthetic volume claimed.
            ns['apply_stacker'](uid,{},zs,value)
            ns['apply_state_stacker'](uid,{},zs,value)
            assert np.isfinite(value).all() and not ns['SS']['stack_fallback'] and not ns['FL']['stack_fallback']
            rows.append(value)
        predictions[label]=np.asarray(rows)
        bundles[label]=dict(identity=identity,folds=checks,fluid_path='missing volume',state_stacker=True)
    # Replay real visible R/own outputs with explicitly reduced-window R6/R7 results.
    from tools import exp169_release_c85r7codec as historical
    staged = historical.stage_replay_input(b.SCRATCH/'real-replay-pr7')
    rid,r,_,_=historical.authenticated_pr7(staged)
    oid,e143,e155,q,_=historical.c85.authenticated_a172(historical.c85.A5Q)
    assert set(rid)==set(oid)==set(ids)
    r=r[[rid.index(u) for u in ids]]
    e143,e155,q=[a[[oid.index(u) for u in ids]] for a in (e143,e155,q)]
    # R224 unchanged: use R384 only as explicit arithmetic surrogate, not inference evidence.
    result=mod.r224.blend(ids,r,r,predictions['R7'],e143,e155,q,r6=predictions['R6'])
    expected=mod.r224.reference(r,r,predictions['R7'],e143,e155,q,r6=predictions['R6'])
    np.testing.assert_array_equal(result,expected)
    payload=mod.p.csv_bytes(ids,result)
    (b.SCRATCH/'reduced-window-replay.csv').write_bytes(payload)
    assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==''
    report=dict(status='PASS',kernel_sha256=val['kernel_sha256'],device='cpu',studies=3,decoded_dicoms=9,
                bundles=bundles,input_sha256=pins,seconds=time.perf_counter()-started,
                replay_sha256=b.old.digest(payload),real_output_replay='reduced-window R7/R6 plus authenticated historical R/own; R224 arithmetic surrogate=R384',
                scope='All eight actual models, frozen flip/MCL/state and stackers. Reduced windows, missing fluid path; not full anatomical rendering, full-study output, CUDA or timing qualification.',
                qualified_for_submission=False,exposure='EXPOSED',clean_oof=False)
    atomic_json(b.SCRATCH/'smoke.json',report)
    return {k:v for k,v in report.items() if k not in ('bundles','input_sha256')}


if __name__=='__main__': print(json.dumps(run(),indent=2))
