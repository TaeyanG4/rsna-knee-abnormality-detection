"""v1.0.0: preregistered exact977 paired inference, no training or uploads.

Changelog: isolated pilot; authenticated inputs/review/owner/output receipts,
measured memory gate, owned subprocess watchdog, aggregates-only evaluation.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from tools import k1n_stage_v11 as b
from tools import job_runtime as jobs
from tools.l1_rco_stage0_v1 import write, sha, need

ROOT=b.ROOT
OUT=ROOT/'.local/scratch/astra-b2/k1s1stride1-v1'
CACHE=Path('F:/rsna-knee-cache/l1rco-cache-full')
PYTHON='C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
TAG='K1S1_STRIDE1_PAIRED_V1'
MODULE='tools.k1n_stride1_pilot_v1'
COMMAND=[PYTHON,'-m',MODULE,'run']
ORDER=(('v2',2),('s0',2),('s1',2),('s1',1))
CHECKPOINTS={
    'v2':('.local/scratch/l1-rco/k1v2/fit/models/E_prior_s0_f0.pt','2c3c033e25605b5fb8bcc08e0030bed081c726e10fc56d8da62524ac08d9638c'),
    's0':('.local/scratch/l1-rco/k1v3-grouping/fit/models/E_prior_s0_f0.pt','6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5'),
    's1':('.local/scratch/astra-b2/h210-recovery-v1/attempt/fit/models/E_prior_s0_f0.pt','29fc2b2bbaa4e4e8ad341bba17e47c5110388db697458a8cbb9ff8d83fa9399b')}
TRAINER=ROOT/'tools/exp189_s6_src/compact_train.py'
TRAINER_SHA='14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'
RULE=dict(inference_n=977, evaluation_n=977, rulers_required=3, ruler_count=4,
    macro_delta_floor=0., every_target_delta_floor=-.005, targets_per_ruler=12,
    missing='FAIL; no roster shrink, ruler or target omission', gpu_seconds_limit=7200,
    trio='rank(mean(rank(v2_s2), rank(s0_s2), rank(s1_sN))); average ties',
    exposure='EXPOSED_EXPLORATORY',clean_oof=False)


def read(path): return jobs.read_json(path)
def digest(value): return jobs.digest(value)


def cohort():
    from tools import l1_rco_k1_foldgate_v1 as gate
    from tools import h216_oai_pair_runtime_v101 as old
    ids=gate.rosters()[0]
    need(len(ids)==977 and ids==old.inputs()['holdout'],'exact977 foldgate roster')
    return ids


def sources():
    from tools import exp156_k_sweep as rulers
    paths=[Path(__file__),TRAINER,ROOT/'tools/l1_rco_k1_foldgate_v1.py',
        ROOT/'tools/h216_oai_pair_runtime_v101.py',ROOT/'tools/h216_pair_eval_v1.py',
        ROOT/'tools/l1_rco_k1_eval_v1.py',ROOT/'tools/l1_rco_runtime_v120.py',
        ROOT/'tools/exp156_k_sweep.py',ROOT/'tools/exp115_vlm_eval.py',
        ROOT/'tools/job_runtime.py',ROOT/'tools/gpu_job.py',
        *[ROOT/p for p in sorted(set((rulers.JEV,rulers.AQUILA,rulers.DREAD,*rulers.CONSENSUS)))]]
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}


def inputs():
    from tools import l1_rco_k1_v1 as k
    reg=read(ROOT/'.local/scratch/l1-rco/k1v3-grouping/registration.json')
    k.complete_cache(reg,CACHE)
    original=Path('G:/temp_data/rsna-knee/l1rco-cache-full')
    pins={n:sha(CACHE/n) for n in ('receipt.json','index.json','meta.f32.npy')}
    need(all(sha(original/n)==h for n,h in pins.items()),'cache replica metadata identity')
    need(sha(TRAINER)==TRAINER_SHA,'trainer pin')
    for path,pin in CHECKPOINTS.values(): need(sha(ROOT/path)==pin,'checkpoint pin')
    return dict(cache=str(CACHE),cache_pins=pins,checkpoint_pins=CHECKPOINTS,
                id_list_sha256=digest(cohort()),n=977,sources=sources())


def register():
    value=dict(schema='k1s1-stride1-pilot/v1',version='1.0.0',rule=RULE,
        inputs=inputs(),order=ORDER,command=COMMAND,job_tag=TAG,
        parent=str(b.PARENT.relative_to(ROOT)),candidate=str(b.stage().relative_to(ROOT)),
        preprocessing=dict(size=384,crop=[16,368],slices=3,phys_mm=3.5,
                           encoder_chunk=32,fp16=True,tf32=False,extent=False))
    path=OUT/'PREREGISTRATION.json'
    if path.exists(): need(read(path)==json.loads(json.dumps(value)),'immutable preregistration drift')
    else: b.util.write_once(path,b.util.encoded(value))
    return dict(status='PASS',preregistration_sha256=sha(path),n=977)


def validate():
    reg=read(OUT/'PREREGISTRATION.json')
    need(reg['inputs']==json.loads(json.dumps(inputs())) and reg['rule']==RULE,'registration drift')
    need(reg['command']==COMMAND and reg['order']==[list(x) for x in ORDER],'command/order drift')
    b.run('validate')
    smoke=read(OUT/'cpu-smoke/comparison.json')
    need(smoke['status']=='PASS' and smoke['tensors_bit_exact'] and not smoke['cuda_initialized'],'real DICOM CPU smoke')
    for p,h in smoke['sources'].items(): need(sha(ROOT/p)==h,'smoke source binding')
    return reg


def import_trainer():
    if os.name=='nt':
        os.environ['S6PLUS_PIN_DLL']=r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
        from tools import s6plus
    os.environ['HF_HUB_OFFLINE']='1'
    import importlib.util
    spec=importlib.util.spec_from_file_location('stride1_frozen_trainer',TRAINER)
    ct=importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
    return ct


def dataset(ct,ids,stride):
    return ct.Studies(str(CACHE),ids,64,'eval',0,eval_stride=stride,slices=3,phys_mm=3.5)


def window_counts(ct,ds):
    import numpy as np
    meta=np.load(CACHE/'meta.f32.npy',mmap_mode='r')
    return [len(ct.window_rows(ds.blocks[u],lambda f,r:meta[r],'eval',3,ds.eval_stride,phys_mm=3.5)) for u in ds.uids]


def prepare():
    validate(); os.environ['CUDA_VISIBLE_DEVICES']='-1'
    ct=import_trainer(); import torch
    from tools import k1sr_runtime_v11 as runtime
    from tools import l1_rco_k2_worker_v102 as w
    torch.set_num_threads(4); ids=cohort(); results={}
    # Same frozen checkpoint and exact cached loader as pilot; two windows only
    # for CPU forward, full largest-study windows are counted and materialized.
    with w.offline():
        net=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
        net.load_deploy_state_dict(torch.load(ROOT/CHECKPOINTS['s1'][0],map_location='cpu',weights_only=True))
        net.eval().requires_grad_(False)
        for stride in (2,1):
            ds=dataset(ct,ids,stride); counts=window_counts(ct,ds); index=max(range(977),key=counts.__getitem__)
            img,meta,i=ds[index]
            need(i==index and len(img)==counts[index] and img.dtype==torch.uint8,'cached full-shape loader')
            value=runtime.predict_prepared(net,runtime.transfer_inputs(img[:2],meta[:2],'cpu'),fp16=False)
            results[str(stride)]=dict(studies=977,total_windows=sum(counts),max_windows=max(counts),
                max_index=index,full_largest_loader=True,cpu_forward_windows=2,finite=bool(__import__('numpy').isfinite(value).all()))
    need(not torch.cuda.is_initialized(),'CPU only')
    out=dict(status='PASS',preregistration_sha256=sha(OUT/'PREREGISTRATION.json'),
             strides=results,checkpoint_sha256=CHECKPOINTS['s1'][1],cuda_initialized=False)
    write(OUT/'pilot-cpu-smoke.json',out)
    return out


def review_bindings():
    paths=[OUT/'PREREGISTRATION.json',OUT/'cpu-smoke/comparison.json',OUT/'pilot-cpu-smoke.json',
           b.stage()/'registration.json',b.stage(True)/'registration.json',
           ROOT/'tests/test_k1n_v11.py',ROOT/'tests/test_k1n_stride1_pilot_v1.py']
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}


def require_review():
    r=read(OUT/'REVIEW.json')
    need(r.get('verdict')=='PASS' and r.get('reviewer') not in (None,'','astra-b2'),'independent review PASS required')
    need(r.get('bindings')==review_bindings(),'review smoke/stage binding')
    report=(ROOT/r['report']).resolve()
    need(report.is_relative_to(ROOT.resolve()) and sha(report)==r['report_sha256'],'review report pin')
    return r


def admit():
    validate(); smoke=read(OUT/'pilot-cpu-smoke.json')
    need(smoke['status']=='PASS' and smoke['preregistration_sha256']==sha(OUT/'PREREGISTRATION.json'),'pilot CPU smoke binding')
    bindings=review_bindings()
    try: require_review()
    except (OSError,KeyError,ValueError) as e:
        return dict(status='BLOCKED',reason='independent review missing or invalid: '+type(e).__name__,
                    required_review_bindings=bindings,gpu_started=False)
    return dict(status='CPU_ADMIT_PASS',gpu_started=False,live_idle_and_full_shape_memory_gate='required at run')


def go_rule(metrics):
    names={'jev_hybrid','aquila_v1','dread','consensus'}
    rows=metrics.get('per_target',[]); macros=metrics.get('rulers',{})
    from tools.l1_rco_k1_eval_v1 import TARGETS
    if set(macros)!=names or len(rows)!=48 or {(r['ruler'],r['target']) for r in rows}!={(n,t) for n in names for t in TARGETS}: return False
    values=[v['delta'] for v in macros.values()]+[r['delta'] for r in rows]
    return all(math.isfinite(v) for v in values) and sum(v['delta']>=0 for v in macros.values())>=3 and min(r['delta'] for r in rows)>=-.005


def owner_check():
    import psutil
    job=read(jobs.LOGS/TAG/'job.json')
    need(job['command']==COMMAND and job['command_sha256']==jobs.command_digest(COMMAND,ROOT) and
         Path(job['cwd']).resolve()==ROOT.resolve(),'gpu_job command identity')
    wrapper=psutil.Process(job['pid'])
    need(jobs.liveness(job)=='alive' and wrapper.cmdline()[-3:]==['--_run-job',TAG,job['run_id']],'gpu_job wrapper identity')
    need(os.getppid()==job['pid'],'gpu_job parent ownership')
    # gpu_job publishes child identity just after Popen; bounded startup race.
    for _ in range(50):
        job=read(jobs.LOGS/TAG/'job.json')
        if job.get('child_identity')==jobs.identity(): break
        time.sleep(.1)
    need(job.get('child_identity')==jobs.identity() and job['phase']=='running','gpu_job child ownership')
    return job


def run():
    need(admit()['status']=='CPU_ADMIT_PASS','independent review required before GPU')
    job=owner_check()
    from tools import l1_rco_runtime_v120 as safety
    snap=safety.gpu_snapshot()
    need(not safety.active_ml_processes(job['pid']) and snap['util']<20,'GPU must be idle')
    need(not (OUT/'pilot-receipt.json').exists() and not (OUT/'preds').exists(),'no replay/overwrite of pilot')
    write(OUT/'gpu-admission.json',dict(snapshot=snap,job_run_id=job['run_id'],supervisor=jobs.identity(),
        review_sha256=sha(OUT/'REVIEW.json'),preregistration_sha256=sha(OUT/'PREREGISTRATION.json')))
    started=time.time(); child=None; history=[]; error=None
    try:
        child=subprocess.Popen([PYTHON,'-m',MODULE,'worker'],cwd=ROOT,
            env={**os.environ,'K1S1_OWNER_PID':str(os.getpid())},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        while child.poll() is None:
            now=time.time(); need(now-started<7200,'two GPU-hour budget exhausted')
            status=read(OUT/'pilot-status.json') if (OUT/'pilot-status.json').exists() else dict(time=started,done=0,phase='init')
            telemetry=safety.gpu_snapshot(); reason=safety.stale_reason(status,now,telemetry)
            need(not reason,reason or 'watchdog')
            history.append((now,status['done'],status['phase'])); history=[x for x in history if now-x[0]<=1830]
            phase=[x for x in history if x[2]==status['phase']]
            if len(phase)>1 and now-phase[0][0]>=1800 and status.get('healthy_rate',0)>0:
                need((status['done']-phase[0][1])/(now-phase[0][0])>=.25*status['healthy_rate'],'30-minute throughput collapse')
            time.sleep(5)
        need(child.returncode==0,'pilot worker failed')
        evaluate()
    except Exception as e:
        error=type(e).__name__+': '+str(e); raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        write(OUT/'pilot-terminal.json',dict(status='FAILED' if error else 'SUCCESS',error=error,
            seconds=time.time()-started,run_id=job['run_id'],qualified_for_submission=False))


def worker():
    import psutil
    admission=read(OUT/'gpu-admission.json'); parent=admission['supervisor']
    need(os.getppid()==parent['pid']==int(os.environ.get('K1S1_OWNER_PID','0')) and jobs.liveness(parent)=='alive','owned worker only')
    argv=psutil.Process(parent['pid']).cmdline()
    need(Path(argv[0]).resolve()==Path(COMMAND[0]).resolve() and argv[1:]==COMMAND[1:],'supervisor command identity')
    need(admit()['status']=='CPU_ADMIT_PASS' and sha(OUT/'REVIEW.json')==admission['review_sha256'],'worker review recheck')
    from tools import l1_rco_runtime_v120 as safety
    from tools import l1_rco_k2_worker_v102 as w
    from tools import k1sr_runtime_v11 as runtime
    from tools import h216_oai_pair_runtime_v101 as prior
    ct=import_trainer(); import torch; import numpy as np
    torch.set_num_threads(4); ids=cohort(); output={}; net=None; loaded=None
    (OUT/'preds').mkdir()
    def heartbeat(phase,done,healthy=0.):
        with contextlib.suppress(PermissionError): write(OUT/'pilot-status.json',dict(time=time.time(),phase=phase,done=done,healthy_rate=healthy))
    def memory():
        peak=torch.cuda.max_memory_reserved(); snap=admission['snapshot']
        need(safety.memory_ok(peak,snap['desktop_bytes'],snap['total_bytes']),'measured 12 GiB + 2 GiB headroom gate')
        return peak
    with w.offline(),runtime.precision():
        for name,stride in ORDER:
            arm=f'{name}_s{stride}'; heartbeat('load_'+arm,0)
            if loaded!=name:
                net=None; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                net=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
                net.load_deploy_state_dict(torch.load(ROOT/CHECKPOINTS[name][0],map_location='cpu',weights_only=True))
                net=net.to('cuda').eval().requires_grad_(False); loaded=name
            ds=dataset(ct,ids,stride); counts=window_counts(ct,ds); largest=max(range(977),key=counts.__getitem__)
            def predict(index):
                img,meta,i=ds[index]; need(i==index and len(img)==counts[index],'inference roster/windows')
                return runtime.predict_prepared(net,runtime.transfer_inputs(img,meta,'cuda'),fp16=True)
            heartbeat('memory_'+arm,0); predict(largest); peak=memory()
            values=np.empty((977,12),np.float32); start=time.time(); healthy=0.
            for i in range(977):
                values[i]=predict(i); peak=max(peak,memory())
                if i==49: healthy=50/(time.time()-start)
                heartbeat(arm,i+1,healthy)
            dest=OUT/'preds'/f'{arm}.csv'; prior.save_predictions(dest,ids,values)
            output[arm]=dict(csv_sha256=sha(dest),seconds=time.time()-start,windows=sum(counts),
                max_windows=max(counts),peak_reserved=peak,checkpoint_sha256=CHECKPOINTS[name][1],eval_stride=stride)
            write(OUT/'pilot-progress.json',dict(completed=output))
    write(OUT/'pilot-receipt.json',dict(status='COMPLETE',preregistration_sha256=sha(OUT/'PREREGISTRATION.json'),
        review_sha256=sha(OUT/'REVIEW.json'),run_id=admission['job_run_id'],outputs=output,rule=RULE,clean_oof=False))


def evaluate():
    validate(); require_review()
    from tools import h216_pair_eval_v1 as loader
    from tools import l1_rco_k1_eval_v1 as evaluator
    from tools import exp156_k_sweep as rulers
    from tools.c85r7_rco_overlay_v1 import ranks
    import numpy as np
    receipt=read(OUT/'pilot-receipt.json'); ids=cohort()
    need(receipt['status']=='COMPLETE' and receipt['preregistration_sha256']==sha(OUT/'PREREGISTRATION.json') and
         receipt['review_sha256']==sha(OUT/'REVIEW.json'),'paired output receipt identity')
    preds={}
    need(set(receipt['outputs'])=={f'{m}_s{s}' for m,s in ORDER},'all four authenticated arms')
    for name,stride in ORDER:
        arm=f'{name}_s{stride}'; p=OUT/'preds'/f'{arm}.csv'; r=receipt['outputs'][arm]
        need(r['csv_sha256']==sha(p) and r['checkpoint_sha256']==CHECKPOINTS[name][1] and r['eval_stride']==stride,'arm output identity')
        preds[arm]=loader.load(p,ids)
    trio=lambda s:ranks(np.mean([ranks(preds['v2_s2']),ranks(preds['s0_s2']),ranks(preds[f's1_s{s}'])],axis=0))
    metrics=evaluator.metrics(trio(1),trio(2),rulers.references(ids,root=ROOT))
    result=dict(status='GO' if go_rule(metrics) else 'NO_GO',rule=RULE,metrics=metrics,
        pilot_receipt_sha256=sha(OUT/'pilot-receipt.json'),qualified_for_submission=False)
    write(OUT/'PILOT-RESULT.json',result)
    return dict(status=result['status'],n=977,rulers=metrics['rulers'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','validate','prepare','admit-dry-run','run','worker','evaluate'))
    command=parser.parse_args().command
    action={'register':register,'validate':lambda:(validate() and dict(status='PASS',n=977)),
        'prepare':prepare,'admit-dry-run':admit,'run':run,'worker':worker,'evaluate':evaluate}[command]
    print(json.dumps(action(),indent=2,allow_nan=False))
