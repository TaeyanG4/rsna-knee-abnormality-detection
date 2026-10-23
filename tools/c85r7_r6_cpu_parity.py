"""H-B-174 CPU parity v1.0.0: two full real studies, all four R6 folds.

Public notebook is parsed, never executed. Only its literal SOURCES and bounded
WORKER_ARGS statements are extracted. CPU-only diagnostic adaptation uses the
parent's reviewed device adapter; compile is disabled equally in both CPU runs.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import c85r7_r224r6fq_v1130 as b
from tools import c85r7_r224flip_diagnostics as d
from tools import hb145_output_io as io


def public_args(manifest,bundle,anatomy):
    tree=ast.parse(''.join(b.old.read(b.PUBLIC_NOTEBOOK)['cells'][6]['source']))
    start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and
               any(isinstance(t,ast.Name) and t.id=='WORKER_ARGS' for t in n.targets))
    nodes=[]
    for n in tree.body[start:]:
        if isinstance(n,ast.Expr):
            break
        b.need(isinstance(n,(ast.Assign,ast.AugAssign,ast.If)),'public arg block scope')
        nodes.append(n)
        if isinstance(n,ast.If) and ast.unparse(n.test)=="manifest.get('views') in ('v3k', 'v4')":
            break
    # Physical anatomy filename differs locally; the public code still selects
    # exactly anatomy.pt and checks the original digest.
    class CodeRoot:
        def __truediv__(self,name):
            b.need(name=='anatomy.pt','unexpected code reference')
            return Path(anatomy)
    source=b.WORK/'public-source'
    for k,v in b.public_sources().items(): b.old.write(source/k,v)
    ns=dict(manifest=manifest,BUNDLE=Path(bundle),SRC=source,FLIP_TTA=True,
            SOURCE_SHA={k:b.sha(v) for k,v in b.public_sources().items()},hashlib=hashlib,CODE_SRC=CodeRoot())
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<public9576-worker-args-only>','exec'),ns)
    return ns['WORKER_ARGS']


def inputs():
    data=ROOT/'.local/exp189-cpu-v120/input/visible_data/test_series'
    paths=sorted(p for p in data.iterdir() if p.is_dir())[:2]
    b.need(len(paths)==2,'two real studies required')
    pins={str(p.relative_to(ROOT)):b.sha(p.read_bytes()) for study in paths for p in sorted(study.rglob('*.dcm'))}
    b.need(len(pins)>2,'full study DICOM inventory required')
    return paths,dict(studies=[p.name for p in paths],dicom_sha256=pins,
        series_sha256=b.sha((data.parent/'test_series.csv').read_bytes()),
        public_notebook_sha256=b.sha(b.PUBLIC_NOTEBOOK.read_bytes()),
        kernel_sha256=b.validate()['kernel_sha256'],tool_sha256=b.sha(Path(__file__).read_bytes()))


def run_arm(arm):
    b.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA_VISIBLE_DEVICES must be present and empty')
    import torch
    b.need(not torch.cuda.is_initialized(),'CPU only')
    paths,binding=inputs()
    runtime=b.WORK/'runtime'
    sys.path[:0]=[str(runtime),str(d.gold.RUNTIME/'deps')]
    os.environ['PYTHONPATH']=os.pathsep.join([str(runtime),str(d.gold.RUNTIME/'deps'),str(ROOT)])
    port=b.port_module()
    source=runtime/'exp169_r7_src'
    if arm=='public':
        source=b.WORK/'public-source'
        for k,v in b.public_sources().items(): b.old.write(source/k,v)
        port.worker_args=public_args
    m,_=port.verify_bundle(b.BUNDLE,source_dir=source)
    serving=b.port_module().worker_args(m,b.BUNDLE,d.ANATOMY)
    b.need(serving==public_args(m,b.BUNDLE,d.ANATOMY),'public/serving args differ')
    out=b.WORK/'cpu-parity'/arm
    # Only this tool's generated arm output may be replaced on an explicit rerun.
    b.need(out.resolve().is_relative_to((b.WORK/'cpu-parity').resolve()),'scratch boundary')
    if out.exists(): shutil.rmtree(out)
    frozen=port.diagnostic_source
    def diagnostic(*args,**kwargs):
        path=frozen(*args,**kwargs)
        worker=path/'compact_infer.py'
        raw=worker.read_text(encoding='utf-8')
        raw=port.s6._replace_once(raw,'    if a.compile:\n','    if a.compile and dev.type == "cuda":\n','CPU eager only')
        worker.write_text(raw,encoding='utf-8')
        return path
    port.diagnostic_source=diagnostic
    frozen_run=port.s6._run_controller
    def cpu_controller(args,env):
        env=dict(env,CUDA_VISIBLE_DEVICES='')
        return frozen_run(args,env)
    port.s6._run_controller=cpu_controller
    started=time.perf_counter()
    pred=port.infer_studies(paths,bundle_dir=b.BUNDLE,code_dir=d.CODE,anatomy_weights=d.ANATOMY,
        output_dir=out,workers=2,amp='auto',device='cpu',max_fallback_fraction=0,source_dir=source)
    b.need(not torch.cuda.is_initialized(),'CPU initialized CUDA')
    receipt=io.read_json(out/'controller/submission_receipt.json')
    record=dict(state='COMPLETE_CPU',arm=arm,binding=binding,seconds=time.perf_counter()-started,
        prediction_sha256=b.sha((out/'predictions.csv').read_bytes()),rows=len(pred),
        full_studies=True,folds=4,flip=True,compile='CPU eager; CUDA qualification deferred',
        controller_receipt=receipt,exposure='EXPOSED',clean_oof=False)
    io.atomic_json(out/'parity-receipt.json',record)
    print(json.dumps(dict(state=record['state'],arm=arm,rows=len(pred),seconds=record['seconds'])))


def evaluate():
    import numpy as np
    paths,binding=inputs()
    values=[]
    for arm in ('serving','public'):
        out=b.WORK/'cpu-parity'/arm
        rec=io.read_json(out/'parity-receipt.json')
        b.need(rec['binding']==binding and rec['state']=='COMPLETE_CPU','stale/incomplete CPU evidence')
        b.need(rec['prediction_sha256']==b.sha((out/'predictions.csv').read_bytes()),'prediction drift')
        rows=b.port_module().s6.read_predictions(out/'predictions.csv',[p.name for p in paths])
        values.append(np.asarray([[r[t] for t in d.gold.LABELS] for r in rows]))
    a,c=values
    b.need(a.shape==c.shape==(2,12) and np.isfinite(a).all() and np.isfinite(c).all(),'bad parity output')
    delta=float(np.max(np.abs(a-c)))
    b.need(delta<=1e-8,'R6 public/serving CPU parity failure')
    result=dict(state='PASS_REAL_FULL_STUDY_CPU_PARITY',studies=2,folds=4,max_abs_delta=delta,
        all_worker_sources_byte_identical=True,args_exact=True,binding=binding,exposure='EXPOSED',clean_oof=False,
        limitation='Both CPU runs eager fp32 through the same parent CPU device adaptation. No CUDA/compile/Linux qualification.')
    io.atomic_json(b.WORK/'cpu-parity/result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='binding'}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['serving','public','evaluate']); a=p.parse_args()
    evaluate() if a.action=='evaluate' else run_arm(a.action)
