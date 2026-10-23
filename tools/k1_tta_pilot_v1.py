"""v1.0.0: paired K1 crop TTA pilot, exact977, no training or uploads.
Changelog 2026-10-22: one trained crop (y=8,x=16), authenticated completed
stride2 controls, probability-pair then unchanged rank trio, user GO rule,
utilization <60%, no total time limit. Original pilot remains untouched.

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
OUT=ROOT/'.local/scratch/astra-b1/k1-tta-v1'
CACHE=Path('F:/rsna-knee-cache/l1rco-cache-full')
PYTHON='C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
TAG='K1_CROP_TTA_PAIRED_V1'
MODULE='tools.k1_tta_pilot_v1'
COMMAND=[PYTHON,'-m',MODULE,'run']
ORDER=(('v2',2),('s0',2),('s1',2))
CHECKPOINTS={
    'v2':('.local/scratch/l1-rco/k1v2/fit/models/E_prior_s0_f0.pt','2c3c033e25605b5fb8bcc08e0030bed081c726e10fc56d8da62524ac08d9638c'),
    's0':('.local/scratch/l1-rco/k1v3-grouping/fit/models/E_prior_s0_f0.pt','6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5'),
    's1':('.local/scratch/astra-b2/h210-recovery-v1/attempt/fit/models/E_prior_s0_f0.pt','29fc2b2bbaa4e4e8ad341bba17e47c5110388db697458a8cbb9ff8d83fa9399b')}
TRAINER=ROOT/'tools/exp189_s6_src/compact_train.py'
TRAINER_SHA='14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'
VERSION='1.0.0'
CONTROL=ROOT/'.local/scratch/astra-b2/k1s1stride1-v1'
CONTROL_RECEIPT_SHA='64adb96f2b97778ee5a1d52ca8cd04443a629c3439024846f102d1c689c71f7c'
VIEW=dict(kind='trained_random_crop_fixed_offset', y=8, x=16, size=352,
          original_y=16, original_x=16, stride=2, flips=False)
RULE=dict(inference_n=977,evaluation_n=977,ruler_count=4,rulers_required=3,
    macro_mean_delta_floor=.003,nonnegative_ruler_floor=0.,target_mean_delta_floor=-.005,
    targets_per_ruler=12,missing='FAIL; no roster shrink, ruler or target omission',
    gpu_seconds_limit=None,trio='rank(mean(rank((p_original+p_tta)/2) per model)); average ties',
    exposure='EXPOSED_EXPLORATORY',clean_oof=False)
TRAINING_FILES=[
    '.local/scratch/l1-rco/k1v2/prepare.json',
    '.local/scratch/l1-rco/k1v2/compact_train_l1.py',
    '.local/scratch/l1-rco/k1v3-grouping/prepare.json',
    '.local/scratch/l1-rco/k1v3-grouping/compact_train_l1.py',
    '.local/scratch/l1-rco/k1v3-seed1/prepare.json',
    '.local/scratch/l1-rco/k1v3-seed1/compact_train_l1.py',
    '.local/scratch/astra-b2/h210-recovery-v1/RECOVERY.json']



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
        ROOT/'tools/k1n_stride1_pilot_v1.py',ROOT/'tools/k1sr_runtime_v11.py',
        ROOT/'tools/gpu_queue.py',ROOT/'tools/c85r7_rco_overlay_v1.py',
        ROOT/'tools/l1_rco_k2_worker_v102.py',ROOT/'tools/k1n_config_v11.py',
        ROOT/'tools/l1_rco_k1_v1.py',ROOT/'tools/l1_rco_stage0_v1.py',
        *[ROOT/p for p in TRAINING_FILES],
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


def controls():
    from tools import h216_pair_eval_v1 as loader
    need(sha(CONTROL/'pilot-receipt.json')==CONTROL_RECEIPT_SHA,'control receipt pin')
    receipt=read(CONTROL/'pilot-receipt.json')
    reg=read(CONTROL/'PREREGISTRATION.json')
    need(receipt['status']=='COMPLETE','incomplete control receipt')
    need(receipt['preregistration_sha256']==sha(CONTROL/'PREREGISTRATION.json'),'control registration pin')
    need(reg['inputs']['id_list_sha256']==digest(cohort()) and reg['inputs']['n']==977,'control cohort')
    need(reg['preprocessing']==dict(size=384,crop=[16,368],slices=3,phys_mm=3.5,
        encoder_chunk=32,fp16=True,tf32=False,extent=False),'control preprocessing')
    values={}
    for name,stride in ORDER:
        arm=f'{name}_s2'; p=CONTROL/'preds'/f'{arm}.csv'; r=receipt['outputs'][arm]
        need(r['eval_stride']==2 and r['checkpoint_sha256']==CHECKPOINTS[name][1]
             and r['csv_sha256']==sha(p),'control prediction/checkpoint pin')
        values[name]=loader.load(p,cohort())
    return values


def training_evidence():
    evidence={}
    for name,folder in [('v2','k1v2'),('s0','k1v3-grouping'),('s1','k1v3-seed1')]:
        directory=ROOT/'.local/scratch/l1-rco'/folder
        prep=read(directory/'prepare.json'); argv=prep['argv']
        need('--aug' not in argv and argv[argv.index('--crop')+1]=='352','only trained crop allowed')
        text=(directory/'compact_train_l1.py').read_text(encoding='utf-8')
        need('default="crop", choices=["crop", "v2"]' in text,'crop default drift')
        need(sha(directory/'compact_train_l1.py')==prep['trainer_sha256'],'training source pin')
        evidence[name]=dict(augmentation='crop',crop=352,offset_support=[0,32],
            config=str((directory/'prepare.json').relative_to(ROOT)),
            config_crop_line=29,default_aug_line=623 if name=='v2' else 624,
            crop_code_lines=[940,946] if name=='v2' else [941,947],
            training_call_line=1151 if name=='v2' else 1132,
            trainer_sha256=prep['trainer_sha256'])
    recovery=read(ROOT/TRAINING_FILES[-1])
    original=list(read(ROOT/'.local/scratch/l1-rco/k1v3-seed1/prepare.json')['argv'])
    original[original.index('--out')+1]=recovery['out']
    need(recovery['trainer_argv']==original,'seed1 recovery recipe; output-only change')
    return evidence


def registration():
    controls()
    return dict(schema='k1-crop-tta-pilot/v1',version=VERSION,rule=RULE,view=VIEW,
        inputs=inputs(),training=training_evidence(),order=ORDER,command=COMMAND,job_tag=TAG,
        control_receipt_sha256=CONTROL_RECEIPT_SHA,
        view_reason='One fixed trained crop: shift top-left by (-8,0) from center. '
            'Positive axis directions, channel/slice order and metadata unchanged in every plane; '
            'no reflection or medial/lateral swap. Small translation chosen without candidate results.',
        preprocessing=dict(size=384,crop=352,slices=3,phys_mm=3.5,encoder_chunk=32,
                           fp16=True,tf32=False,extent=False),
        safety=dict(util_below=60,peak_gib_max=12,headroom_gib=2,total_time_limit=None))


def assert_fresh():
    for name in ('preds','pilot-receipt.json','gpu-admission.json','pilot-terminal.json',
                 'PILOT-RESULT.json','pilot-status.json','pilot-progress.json'):
        need(not (OUT/name).exists(),'existing destination: '+name+'; never delete/replay')


def register():
    assert_fresh()
    path=OUT/'PREREGISTRATION.json'
    need(not path.exists(),'preregistration already exists; use validate')
    b.util.write_once(path,b.util.encoded(registration()))
    return dict(status='PASS',preregistration_sha256=sha(path),n=977)


def validate():
    reg=read(OUT/'PREREGISTRATION.json')
    need(reg==json.loads(json.dumps(registration())),'immutable registration drift')
    return reg


def transfer_view(img,meta,device,tta=True):
    import torch
    need(img.dtype==torch.uint8 and img.ndim==4 and tuple(img.shape[-2:])==(384,384),'384 uint8 input required')
    y,x=(VIEW['y'],VIEW['x']) if tta else (16,16)
    return (img[None,...,y:y+352,x:x+352].to(device),meta[None].to(device),
            torch.ones((1,len(img)),dtype=torch.bool,device=device))


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
    need(not (OUT/'pilot-cpu-smoke.json').exists(),'existing CPU smoke destination')
    validate(); os.environ['CUDA_VISIBLE_DEVICES']='-1'
    ct=import_trainer(); import torch; import numpy as np
    from tools import k1sr_runtime_v11 as runtime
    from tools import l1_rco_k2_worker_v102 as w
    torch.set_num_threads(4); ids=cohort(); ds=dataset(ct,ids,2)
    counts=window_counts(ct,ds); index=min(range(977),key=counts.__getitem__)
    img,meta,i=ds[index]; need(i==index and len(img)==counts[index],'real-study loader')
    original=transfer_view(img,meta,'cpu',False); tta=transfer_view(img,meta,'cpu',True)
    old=runtime.transfer_inputs(img,meta,'cpu')
    need(all(torch.equal(a,b) for a,b in zip(original,old)),'original preprocessing parity')
    need(torch.equal(tta[0],img[None,...,8:360,16:368]) and
         torch.equal(original[1],tta[1]) and torch.equal(original[2],tta[2]),'crop axes/metadata parity')
    results={}
    with w.offline():
        for name,stride in ORDER:
            net=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
            net.load_deploy_state_dict(torch.load(ROOT/CHECKPOINTS[name][0],map_location='cpu',weights_only=True))
            net.eval().requires_grad_(False)
            a=runtime.predict_prepared(net,original,fp16=False)
            z=runtime.predict_prepared(net,tta,fp16=False)
            pair=(a+z)/2
            need(a.shape==z.shape==(12,) and np.isfinite(pair).all() and ((pair>=0)&(pair<=1)).all(),'CPU finite probability')
            results[name]=dict(checkpoint_sha256=CHECKPOINTS[name][1],original_finite=True,
                tta_finite=True,paired_finite=True,full_study_windows=len(img))
            print(json.dumps(dict(cpu_smoke_model=name,status='PASS',windows=len(img))),flush=True)
            del net
    need(not torch.cuda.is_initialized(),'CPU only')
    # Real authenticated full-cohort controls exercise the evaluator plumbing;
    # identity pairing is an engineering smoke, never candidate evidence.
    from tools import l1_rco_k1_eval_v1 as evaluator
    from tools import exp156_k_sweep as rulers
    control_values=controls()
    identity,control=paired_trios(control_values,control_values)
    need(np.array_equal(identity,control),'identity probability pairing')
    replay=evaluator.metrics(identity,control,rulers.references(ids,root=ROOT))
    need(replay['n']==977 and len(replay['per_target'])==48 and
         all(v['delta']==0 for v in replay['rulers'].values()) and
         all(r['delta']==0 for r in replay['per_target']) and not go_rule(replay),'control identity evaluator replay')
    out=dict(status='PASS',preregistration_sha256=sha(OUT/'PREREGISTRATION.json'),
        studies=1,cohort_n=977,total_windows=sum(counts),max_windows=max(counts),
        study_id_sha256=digest(ids[index]),models=results,original_bit_exact=True,
        tta_crop_bit_exact=True,cuda_initialized=False,
        control_identity_replay=dict(n=977,rulers=4,target_cells=48,all_delta_zero=True,go=False))
    write(OUT/'pilot-cpu-smoke.json',out)
    return out


def review_bindings():
    paths=[OUT/'PREREGISTRATION.json',OUT/'pilot-cpu-smoke.json',
           ROOT/'tests/test_k1_tta_pilot_v1.py']
    return {str(p.relative_to(ROOT)).replace(chr(92),'/'):sha(p) for p in paths}


def require_review():
    r=read(OUT/'REVIEW.json')
    need(r.get('verdict')=='PASS' and r.get('reviewer') not in (None,'','astra-b1'),'independent review PASS required')
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
    if not all(math.isfinite(v) for v in values): return False
    return (math.fsum(v['delta'] for v in macros.values())/4>=.003
        and sum(v['delta']>=0 for v in macros.values())>=3
        and all(math.fsum(r['delta'] for r in rows if r['target']==t)/4>=-.005 for t in TARGETS))


def paired_trios(original,tta):
    import numpy as np
    from tools.c85r7_rco_overlay_v1 import ranks
    names=[m for m,_ in ORDER]
    for data in (original,tta):
        need(set(data)==set(names),'exact trio members')
        for v in data.values():
            need(v.shape==(977,12) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all(),'probability shape/range')
    control=ranks(np.mean([ranks(original[m]) for m in names],axis=0))
    candidate=ranks(np.mean([ranks((original[m]+tta[m])/2) for m in names],axis=0))
    return candidate,control


def idle_snapshot(owner_pid):
    from tools import l1_rco_runtime_v120 as safety
    snap=safety.gpu_snapshot()
    # WDDM desktop C+G apps are allowed; unknown compute programs fail closed.
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    import csv
    desktop={'explorer.exe','chrome.exe','msedge.exe','claude.exe','codex.exe',
        'chatgpt.exe','dwm.exe','nvidia overlay.exe','nvidia share.exe',
        'textinputhost.exe','startmenuexperiencehost.exe','shellexperiencehost.exe',
        'applicationframehost.exe','searchhost.exe','searchapp.exe','lockapp.exe','systemsettings.exe',
        'docker desktop.exe','windowsterminal.exe','msedgewebview2.exe',
        'codex-computer-use-swift.exe','discord.exe','phoneexperiencehost.exe','osu!.exe'}
    blocked=[]
    for row in csv.reader(raw.splitlines()):
        if not row or row[0].strip() in ('','No running processes found'): continue
        need(len(row)==2 and row[0].strip().isdigit(),'unknown compute process row')
        if int(row[0].strip())==owner_pid: continue
        name=Path(row[1].strip()).name.lower()
        if name=='[insufficient permissions]':
            import psutil
            try: name=psutil.Process(int(row[0].strip())).name().lower()
            except psutil.Error: pass  # fail closed if identity remains unavailable
        if name not in desktop: blocked.append(row[0].strip())
    need(not blocked,'another ML/CUDA or unknown compute process is active: '+','.join(blocked))
    need(snap['util']<60,'utilization must be <60%')
    return snap


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
    assert_fresh()
    snap=idle_snapshot(job['pid'])
    write(OUT/'gpu-admission.json',dict(snapshot=snap,job_run_id=job['run_id'],supervisor=jobs.identity(),
        review_sha256=sha(OUT/'REVIEW.json'),preregistration_sha256=sha(OUT/'PREREGISTRATION.json')))
    started=time.time(); child=None; history=[]; error=None
    try:
        child=subprocess.Popen([PYTHON,'-m',MODULE,'worker'],cwd=ROOT,
            env={**os.environ,'K1TTA_OWNER_PID':str(os.getpid())},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        while child.poll() is None:
            now=time.time()  # No total time limit; progress/throughput watchdogs remain.
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
    need(os.getppid()==parent['pid']==int(os.environ.get('K1TTA_OWNER_PID','0')) and jobs.liveness(parent)=='alive','owned worker only')
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
            arm=f'{name}_tta_s{stride}'; heartbeat('load_'+arm,0)
            if loaded!=name:
                net=None; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                net=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
                net.load_deploy_state_dict(torch.load(ROOT/CHECKPOINTS[name][0],map_location='cpu',weights_only=True))
                net=net.to('cuda').eval().requires_grad_(False); loaded=name
            ds=dataset(ct,ids,stride); counts=window_counts(ct,ds); largest=max(range(977),key=counts.__getitem__)
            def predict(index):
                img,meta,i=ds[index]; need(i==index and len(img)==counts[index],'inference roster/windows')
                return runtime.predict_prepared(net,transfer_view(img,meta,'cuda',True),fp16=True)
            heartbeat('memory_'+arm,0); predict(largest); peak=memory()
            values=np.empty((977,12),np.float32); start=time.time(); healthy=0.
            for i in range(977):
                values[i]=predict(i); peak=max(peak,memory())
                if i==49: healthy=50/(time.time()-start)
                heartbeat(arm,i+1,healthy)
            dest=OUT/'preds'/f'{arm}.csv'; prior.save_predictions(dest,ids,values)
            output[arm]=dict(csv_sha256=sha(dest),seconds=time.time()-start,windows=sum(counts),
                max_windows=max(counts),peak_reserved=peak,checkpoint_sha256=CHECKPOINTS[name][1],eval_stride=stride,view=VIEW)
            write(OUT/'pilot-progress.json',dict(completed=output))
    write(OUT/'pilot-receipt.json',dict(status='COMPLETE',preregistration_sha256=sha(OUT/'PREREGISTRATION.json'),
        review_sha256=sha(OUT/'REVIEW.json'),run_id=admission['job_run_id'],outputs=output,rule=RULE,view=VIEW,control_receipt_sha256=CONTROL_RECEIPT_SHA,clean_oof=False))


def evaluate():
    need(not (OUT/'PILOT-RESULT.json').exists(),'existing result destination')
    validate(); require_review()
    from tools import h216_pair_eval_v1 as loader
    from tools import l1_rco_k1_eval_v1 as evaluator
    from tools import exp156_k_sweep as rulers
    from tools.c85r7_rco_overlay_v1 import ranks
    import numpy as np
    receipt=read(OUT/'pilot-receipt.json'); ids=cohort()
    need(receipt['status']=='COMPLETE' and receipt['preregistration_sha256']==sha(OUT/'PREREGISTRATION.json') and
         receipt['review_sha256']==sha(OUT/'REVIEW.json'),'paired output receipt identity')
    need(receipt['view']==VIEW and receipt['control_receipt_sha256']==CONTROL_RECEIPT_SHA,'view/control identity')
    preds={}
    need(set(receipt['outputs'])=={f'{m}_tta_s2' for m,s in ORDER},'all three authenticated TTA arms')
    for name,stride in ORDER:
        arm=f'{name}_tta_s2'; p=OUT/'preds'/f'{arm}.csv'; r=receipt['outputs'][arm]
        need(r['csv_sha256']==sha(p) and r['checkpoint_sha256']==CHECKPOINTS[name][1]
             and r['eval_stride']==2 and r['view']==VIEW,'arm output identity')
        preds[name]=loader.load(p,ids)
    candidate,control=paired_trios(controls(),preds)
    metrics=evaluator.metrics(candidate,control,rulers.references(ids,root=ROOT))
    # The reused evaluator's historical decision is NOT this pilot's decision.
    metrics['legacy_evaluator_decision']=metrics.pop('decision')
    metrics['pilot_macro_mean_delta']=math.fsum(v['delta'] for v in metrics['rulers'].values())/4
    metrics['pilot_target_mean_delta']={t:math.fsum(r['delta'] for r in metrics['per_target'] if r['target']==t)/4 for t in evaluator.TARGETS}
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
