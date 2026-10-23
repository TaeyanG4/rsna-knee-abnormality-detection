"""K1 pooling v1.0.1: paired inference only; B owns the reviewed GPU launch.

Changelog 1.0.1: POOL-R1 bounded telemetry, deadline checks and fail-closed cleanup;
immutable scientific registration retained with separately reviewed repair pins.
Changelog: new isolated experiment, singleton trained-head window logits,
fixed target operators, exact977 four-ruler gate, no serving modification.
"""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from tools import k1n_stride1_pilot_v1 as base
from tools import job_runtime as jobs
from tools.l1_rco_stage0_v1 import write, sha, need

VERSION = '1.0.1'
ROOT = base.ROOT
OUT = ROOT / '.local/scratch/astra-b1/k1-pooling-v1'
REPAIR_OUT = ROOT / '.local/scratch/astra-b2/k1-pooling-r1'
PREREG_SHA256 = '272313464f508766ae53421cf9928165b184bddf3854dea590bec38f8e3b85e7'
REPAIR_PATHS = {'tools/k1_pooling_v1.py', 'tests/test_k1_pooling_runtime_v1.py'}
TELEMETRY_SECONDS = 5.
STOP_SECONDS = 5380.
BUDGET_SECONDS = 5400.
PYTHON = base.PYTHON
CACHE = base.CACHE
CHECKPOINTS = base.CHECKPOINTS
ORDER = ('v2', 's0', 's1')
MEMBERS = dict(v2='k1_v2', s0='k1_v3g_seed0', s1='k1_v3g_seed1')
TAG = 'K1_TARGET_POOLING_V1'
MODULE = 'tools.k1_pooling_v1'
COMMAND = [PYTHON, '-m', MODULE, 'run']
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
FOCAL = (0, 1, 2, 3, 4, 5, 6, 10, 11)
EXTENT = (7, 8, 9)
RULERS = ('jev_hybrid', 'aquila_v1', 'dread', 'consensus')
RULE = dict(inference_n=977, evaluation_n=977, rulers_required=3,
            ruler_count=4, macro_delta_floor=0., mean_macro_delta_floor=.002,
            every_target_delta_floor=-.005, targets_per_ruler=12,
            missing='HOLD; no row/ruler/target omission; original target availability masks only',
            gpu_seconds_limit=5400, eval_stride=2, encoder_chunk=32,
            focal=[TARGETS[i] for i in FOCAL], extent=[TARGETS[i] for i in EXTENT],
            focal_formula='0.5*trained_pooled + 0.5*mean(top-min(3,n) singleton-window logits)',
            extent_formula='0.5*trained_pooled + 0.5*mean(all singleton-window logits)',
            candidate_count=1, pooling_operators=2, search=False,
            window_definition='same trained spatial pool, metadata, projection, out_norm and classifier; singleton window head',
            spatial_position_logits='not defined by trained head; spatial attention unchanged',
            trio='rank(mean(rank(sigmoid_fp32(member_logits)))); equal three members, average ties',
            diagnostics_only=['FOCAL-only mean AUC delta', 'EXTENT-only mean AUC delta'],
            precision='GPU autocast FP16; TF32 off; original pooled retained; aggregation CPU FP32',
            reconstruction_fp32_atol=1e-6, reconstruction_fp16_atol=1e-5,
            singleton_fp32_atol=1e-5, singleton_fp16_atol=.005,
            exposure='EXPOSED_EXPLORATORY', clean_oof=False,
            no_retry=True, training=False, fallback=False)

read = jobs.read_json
cohort = base.cohort


def sources():
    pins = base.sources()
    paths = [Path(__file__), ROOT/'tools/k1_pooling_cpu_v1.py',
             ROOT/'tools/k1sr_runtime_v11.py', ROOT/'tools/k1_shared_render_v1.py',
             ROOT/'tools/k1_shared_render_v3.py', ROOT/'tools/k1n_config_v11.py',
             ROOT/'tools/l1_rco_k2_worker_v102.py', ROOT/'tools/c85r7_rco_overlay_v1.py',
             ROOT/'tools/l1_rco_k1_v1.py', ROOT/'tools/l1_rco_stage0_v1.py',
             ROOT/'tools/s6plus.py', ROOT/'tests/test_k1_pooling_v1.py',
             ROOT/'tests/test_k1_pooling_runtime_v1.py']
    pins.update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    return pins


def inputs():
    result = base.inputs()
    result['sources'] = sources()
    return result


def registration():
    return dict(schema='k1-target-pooling/v1', version=VERSION, rule=RULE,
                inputs=inputs(), members=MEMBERS, order=list(ORDER), command=COMMAND,
                job_tag=TAG, preprocessing=dict(size=384, crop=[16,368], slices=3,
                    phys_mm=3.5, train_cap_unchanged=64, eval_full_bag=True,
                    extent_head=False, state_head=False),
                cpu_smoke=dict(studies=3, selection='three smallest full stride2 bags; no labels',
                    all_members=True, full_bags=True, device='cpu', fp16=False),
                interpretation='Training fold0 excluded, but repeatedly consulted rulers/holdout are exploratory. No P25 gain inference.')


def register():
    path = OUT/'PREREGISTRATION.json'
    # This repair applies only to the existing, immutable v1.0.0 registration.
    validate()
    return dict(status='PASS', n=977, preregistration_sha256=sha(path))


def validate():
    need(sha(OUT/'PREREGISTRATION.json') == PREREG_SHA256, 'immutable preregistration drift')
    original = read(OUT/'PREREGISTRATION.json')
    current = json.loads(json.dumps(registration()))
    repair = read(REPAIR_OUT/'EXECUTION-REPAIR.json')
    need(repair['version'] == VERSION and repair['preregistration_sha256'] == PREREG_SHA256
         and set(repair['sources']) == REPAIR_PATHS, 'execution repair identity')
    for path, pin in repair['sources'].items():
        need(pin['before'] == original['inputs']['sources'][path] and
             pin['after'] == current['inputs']['sources'][path], 'execution repair source drift')
        current['inputs']['sources'][path] = pin['before']
    current['version'] = original['version']
    need(original == current, 'preregistration/input drift')
    return dict(status='PASS', n=977, cuda_initialized=False)


def reconstruct(net, tok, meta, mask):
    """Exact trained pooled path plus singleton logits; no spatial approximation."""
    import torch
    need(net.spatial and not net.extent and not net.has_state and not net.training,
         'requires eval spatial head without extent/state')
    need(mask.ndim == 2 and bool(mask.any(1).all()), 'empty bag')
    f = net.proj(tok.float())
    emb = net.slot(meta[...,0].long().clamp(0,9)) + net.view(meta[...,1].long().clamp(0,5)) + net.geom(meta[...,2:6])
    t = f.new_zeros(tuple(mask.shape) + tuple(f.shape[1:]))
    t[mask] = f
    t = net.drop(t + emb[:,:,None,:])
    score = torch.einsum('bwch,ch->bwc', torch.tanh(net.win_proj(t)), net.q_win).float()
    attention = score.masked_fill(~mask[:,:,None], float('-inf')).softmax(1).to(t.dtype)
    study = net.out_norm(torch.einsum('bwc,bwch->bch', attention,t))
    pooled = (study*net.cls_w).sum(-1).float() + net.bias
    window = (net.out_norm(t)*net.cls_w).sum(-1).float() + net.bias
    return pooled, window[mask]


def extract(net, prepared, fp16=False, prove_singletons=False):
    """One unchanged forward; capture only head tokens, not CNN feature maps."""
    import torch
    import numpy as np
    x, meta, mask = prepared
    need(mask.shape[0] == 1 and bool(mask.all()), 'one complete unpadded study')
    need(not fp16 or x.device.type == 'cuda', 'FP16 requires CUDA')
    captured = []
    handle = net.proj.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    try:
        with torch.inference_mode(), torch.autocast(x.device.type, dtype=torch.float16, enabled=fp16):
            trained = net(x,meta,mask,chunk=32)
    finally:
        handle.remove()
    need(len(captured) == 1, 'exactly one trained head invocation')
    with torch.inference_mode(), torch.autocast(x.device.type, dtype=torch.float16, enabled=fp16):
        rebuilt, windows = reconstruct(net,captured[0],meta,mask)
        err = float((rebuilt-trained).abs().max())
        single_error = 0.
        if prove_singletons:
            for i in sorted(set((0,len(windows)//2,len(windows)-1))):
                one = net.head(captured[0][i:i+1],meta[:,i:i+1],mask[:,i:i+1])
                single_error = max(single_error,float((one[0]-windows[i]).abs().max()))
    tolerance = RULE['reconstruction_fp16_atol' if fp16 else 'reconstruction_fp32_atol']
    singleton_tolerance = RULE['singleton_fp16_atol' if fp16 else 'singleton_fp32_atol']
    need(err <= tolerance and single_error <= singleton_tolerance, 'trained/singleton head parity failed')
    pooled, windows = trained.float().cpu().numpy()[0], windows.float().cpu().numpy()
    need(pooled.shape == (12,) and windows.shape == (int(mask.sum()),12) and
         np.isfinite(pooled).all() and np.isfinite(windows).all(), 'invalid extracted logits')
    return pooled, windows, dict(pooled_max_abs=err,singleton_max_abs=single_error)


def candidate(pooled, windows):
    import numpy as np
    pooled, windows = np.asarray(pooled,np.float32),np.asarray(windows,np.float32)
    need(pooled.shape==(12,) and windows.ndim==2 and windows.shape[1]==12 and len(windows)>0 and
         np.isfinite(pooled).all() and np.isfinite(windows).all(), 'invalid window logits')
    k = min(3,len(windows))
    means = windows.mean(0,dtype=np.float32)
    means[list(FOCAL)] = np.sort(windows[:,FOCAL],axis=0)[-k:].mean(0,dtype=np.float32)
    return np.float32(.5)*pooled + np.float32(.5)*means


def metrics(candidate_trio, baseline_trio, refs):
    """Aggregates only; no inherited KEEP/976/spearman selection rule."""
    import numpy as np
    from tools.l1_rco_k1_eval_v1 import auc
    a,b = np.asarray(candidate_trio),np.asarray(baseline_trio)
    need(a.shape == b.shape == (977,12) and np.isfinite(a).all() and np.isfinite(b).all(), 'exact977 finite predictions')
    need(set(refs)==set(RULERS), 'four rulers required')
    table, macros, diagnostics = [],{},{}
    for name in RULERS:
        y,mask,_ = refs[name]
        need(y.shape==mask.shape==a.shape and mask.dtype==bool and np.isfinite(y).all() and np.isin(y,[0,1]).all(), 'ruler shape/values')
        aa,bb = [],[]
        for j,target in enumerate(TARGETS):
            m = mask[:,j]
            ca,ba = float(auc(y[m,j],a[m,j])),float(auc(y[m,j],b[m,j]))
            need(math.isfinite(ca) and math.isfinite(ba), 'undefined target AUC; HOLD')
            aa.append(ca); bb.append(ba)
            table.append(dict(ruler=name,target=target,n=int(m.sum()),candidate_auc=ca,baseline_auc=ba,delta=ca-ba))
        macros[name] = dict(candidate=float(np.mean(aa)),baseline=float(np.mean(bb)),delta=float(np.mean(aa)-np.mean(bb)))
        diagnostics[name] = {t:float(np.mean(np.asarray(aa)[list(ix)]-np.asarray(bb)[list(ix)])) for t,ix in [('FOCAL',FOCAL),('EXTENT',EXTENT)]}
    return dict(n=977,rulers=macros,per_target=table,diagnostics=diagnostics)


def go_rule(m):
    rows,macros = m.get('per_target',[]),m.get('rulers',{})
    if set(macros)!=set(RULERS) or len(rows)!=48 or {(r['ruler'],r['target']) for r in rows}!={(n,t) for n in RULERS for t in TARGETS}:
        return False
    values = [v['delta'] for v in macros.values()] + [r['delta'] for r in rows]
    return (all(math.isfinite(v) for v in values) and sum(v['delta']>=0 for v in macros.values())>=3
            and sum(v['delta'] for v in macros.values())/4>=.002 and min(r['delta'] for r in rows)>=-.005)


def trio(logits):
    import numpy as np
    import torch
    from tools.c85r7_rco_overlay_v1 import ranks
    need(len(logits)==3, 'equal trio required')
    # Exactly the original serving FP32 torch sigmoid (including saturated ties).
    return ranks(np.mean([ranks(torch.from_numpy(np.asarray(x,np.float32)).sigmoid().numpy()) for x in logits],axis=0))


def load_dump(path, pin, ids, counts):
    import numpy as np
    need(sha(path)==pin, 'dump hash mismatch')
    with np.load(path,allow_pickle=False) as f:
        need(set(f.files)=={'ids','offsets','pooled','windows'}, 'dump schema')
        uid,off,pool,win = (f[k] for k in ('ids','offsets','pooled','windows'))
    need(uid.tolist()==ids and len(ids)==977, 'dump roster mismatch')
    expected = np.r_[0,np.cumsum(counts,dtype=np.int64)]
    need(off.dtype==np.int64 and np.array_equal(off,expected) and all(n>0 for n in counts), 'ragged offsets/counts mismatch')
    need(pool.shape==(977,12) and win.shape==(int(off[-1]),12) and pool.dtype==win.dtype==np.float32 and
         np.isfinite(pool).all() and np.isfinite(win).all(), 'invalid dump logits')
    return pool,np.stack([candidate(pool[i],win[off[i]:off[i+1]]) for i in range(977)])


def review_bindings():
    paths = [OUT/'PREREGISTRATION.json', OUT/'cpu-smoke.json', OUT/'tests.json',
             OUT/'BUILD-RESULTS.md', OUT/'LAUNCH.md',
             REPAIR_OUT/'EXECUTION-REPAIR.json', REPAIR_OUT/'tests.json',
             REPAIR_OUT/'cpu-smoke.json']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def require_review():
    r = read(OUT/'REVIEW.json')
    need(r.get('verdict')=='PASS' and r.get('reviewer') not in (None,'','astra-b1','astra-b2') and
         r.get('builder')=='astra-b1' and r.get('repair_builder')=='astra-b2', 'independent review PASS required')
    need(r.get('bindings')==review_bindings(), 'review artifact drift')
    report = (ROOT/r['report']).resolve()
    need(report.is_relative_to(ROOT.resolve()) and sha(report)==r['report_sha256'], 'review report pin')
    return r


def busy_jobs():
    return [dict(tag=x['tag'],state=x['state']) for x in jobs.active_jobs(jobs.LOGS,exclude=TAG)]


def admit():
    validate()
    smoke = read(OUT/'cpu-smoke.json')
    need(smoke['status']=='PASS' and not smoke['cuda_initialized'] and
         smoke['preregistration_sha256']==sha(OUT/'PREREGISTRATION.json'), 'CPU smoke identity')
    need(read(OUT/'tests.json')['status']=='PASS', 'relevant tests required')
    need(read(REPAIR_OUT/'tests.json')['status']=='PASS', 'repair tests required')
    repair_smoke = read(REPAIR_OUT/'cpu-smoke.json')
    need(repair_smoke['status']=='PASS' and not repair_smoke['cuda_initialized'] and
         repair_smoke['source_pins']==sources(), 'repair CPU smoke identity')
    blockers = []
    try: require_review()
    except (OSError,KeyError,ValueError) as exc:
        blockers.append('Independent review missing/invalid: '+type(exc).__name__)
    busy = busy_jobs()
    if busy: blockers.append('GPU lane busy or unresolved; no waiting, stopping, or stealing')
    if (OUT/'gpu-admission.json').exists(): blockers.append('Attempt already consumed; no retry')
    return dict(status='BLOCKED' if blockers else 'CPU_ADMIT_PASS',blockers=blockers,busy=busy,
                gpu_started=False,required_review_bindings=review_bindings(),
                measured_vram_gate='pending B launch on idle GPU')


def owner_check():
    import psutil
    job = read(jobs.LOGS/TAG/'job.json')
    need(job['command']==COMMAND and job['command_sha256']==jobs.command_digest(COMMAND,ROOT) and
         Path(job['cwd']).resolve()==ROOT.resolve(), 'gpu_job command identity')
    wrapper = psutil.Process(job['pid'])
    need(jobs.liveness(job)=='alive' and wrapper.cmdline()[-3:]==['--_run-job',TAG,job['run_id']], 'gpu_job wrapper identity')
    need(os.getppid()==job['pid'], 'gpu_job parent ownership')
    for _ in range(50):
        job = read(jobs.LOGS/TAG/'job.json')
        if job.get('child_identity')==jobs.identity(): break
        time.sleep(.1)
    need(job.get('child_identity')==jobs.identity() and job['phase']=='running', 'gpu_job child ownership')
    return job


def gpu_snapshot(timeout=TELEMETRY_SECONDS):
    """Same local-card parser; isolated timeout repair, no shared runtime edit."""
    need(0 < timeout <= TELEMETRY_SECONDS, 'bounded telemetry timeout required')
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw',
                                   '--format=csv,noheader,nounits'], text=True, timeout=timeout)
    rows = [x.strip().split(', ') for x in out.splitlines() if x.strip()]
    need(len(rows) == 1 and '4070 Ti SUPER' in rows[0][1], 'requires exclusive local RTX 4070 Ti SUPER')
    r = rows[0]
    return dict(index=int(r[0]), name=r[1], total_bytes=int(r[2])*2**20,
                desktop_bytes=int(r[3])*2**20, util=float(r[4]), power=float(r[5]))


def active_ml_processes(owner_pid=None, timeout=TELEMETRY_SECONDS):
    need(0 < timeout <= TELEMETRY_SECONDS, 'bounded telemetry timeout required')
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                   '--format=csv,noheader'], text=True, timeout=timeout)
    return [line for line in out.splitlines() if 'python' in line.lower() and line.split(',')[0].strip() != str(owner_pid)]


def run():
    need(admit()['status']=='CPU_ADMIT_PASS', 'review/idle/unused admission required before GPU')
    job = owner_check()
    from tools import l1_rco_runtime_v120 as safety
    snap = gpu_snapshot()
    need(not active_ml_processes(job['pid']) and snap['util']<20, 'GPU must be idle')
    admission = dict(snapshot=snap,job_run_id=job['run_id'],supervisor=jobs.identity(),
        review_sha256=sha(OUT/'REVIEW.json'),preregistration_sha256=sha(OUT/'PREREGISTRATION.json'),started=time.time())
    with (OUT/'gpu-admission.json').open('x',encoding='utf-8') as f: json.dump(admission,f,indent=2)
    started = time.monotonic(); child = None; error = None; history = []
    def remaining():
        left = STOP_SECONDS-(time.monotonic()-started)
        need(left > 0, '1.5 GPU-hour watchdog cutoff exhausted; no retry')
        return left
    try:
        child = subprocess.Popen([PYTHON,'-m',MODULE,'worker'],cwd=ROOT,
            env={**os.environ,'K1_POOL_OWNER_PID':str(os.getpid()),'CUDA_VISIBLE_DEVICES':'0'},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        while True:
            # Check even if Popen/poll or the last telemetry crossed the deadline.
            remaining()
            done = child.poll() is not None
            remaining()
            if done: break
            status = read(OUT/'gpu-status.json') if (OUT/'gpu-status.json').exists() else dict(time=admission['started'],done=0,phase='init')
            telemetry = gpu_snapshot(timeout=min(TELEMETRY_SECONDS,remaining()))
            remaining()
            now = time.time()
            reason = safety.stale_reason(status,now,telemetry); need(not reason,reason or 'watchdog')
            history.append((now,status['done'],status['phase'])); history = [x for x in history if now-x[0]<=1830]
            phase = [x for x in history if x[2]==status['phase']]
            if len(phase)>1 and now-phase[0][0]>=1800 and status.get('healthy_rate',0)>0:
                need((status['done']-phase[0][1])/(now-phase[0][0])>=.25*status['healthy_rate'], '30-minute throughput collapse')
            time.sleep(min(5,remaining()))
        need(child.returncode==0, 'inference worker failed; preserve partial files; no retry')
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
    finally:
        # Never let a cleanup exception suppress the terminal FAILED receipt.
        try:
            if child is not None and child.poll() is None:
                error = error or 'Worker still running at supervisor exit'
                try:
                    child.terminate()
                    child.wait(timeout=max(0., min(10.,BUDGET_SECONDS-(time.monotonic()-started))))
                except BaseException:
                    child.kill()
                    child.wait(timeout=max(0., min(10.,BUDGET_SECONDS-(time.monotonic()-started))))
        except BaseException as exc:
            error = (error or 'Cleanup failed')+'; cleanup: '+type(exc).__name__+': '+str(exc)
        elapsed = time.monotonic()-started
        if elapsed >= STOP_SECONDS:
            error = error or '1.5 GPU-hour watchdog cutoff exhausted; no retry'
        if elapsed > BUDGET_SECONDS:
            error = (error or 'Watchdog failed')+'; 5400-second budget overrun'
        write(OUT/'gpu-terminal.json',dict(status='FAILED' if error else 'SUCCESS',error=error,
            seconds=elapsed,run_id=job['run_id'],qualified_for_submission=False))
    need(not error, error or 'watchdog failure')


def worker():
    import psutil
    admission = read(OUT/'gpu-admission.json'); parent = admission['supervisor']
    need(os.getppid()==parent['pid']==int(os.environ.get('K1_POOL_OWNER_PID','0')) and jobs.liveness(parent)=='alive', 'owned worker only')
    argv = psutil.Process(parent['pid']).cmdline()
    need(Path(argv[0]).resolve()==Path(COMMAND[0]).resolve() and argv[1:]==COMMAND[1:], 'supervisor command identity')
    validate(); require_review()
    need(sha(OUT/'REVIEW.json')==admission['review_sha256'] and sha(OUT/'PREREGISTRATION.json')==admission['preregistration_sha256'], 'worker admission identity')
    from tools import l1_rco_runtime_v120 as safety
    from tools import l1_rco_k2_worker_v102 as offline
    from tools import k1sr_runtime_v11 as runtime
    ct = base.import_trainer()
    import torch
    import numpy as np
    torch.set_num_threads(4); ids = cohort(); output = {}; net = None
    dest = OUT/'dumps'; dest.mkdir()
    def heartbeat(phase,done,healthy=0.):
        with contextlib.suppress(PermissionError):
            write(OUT/'gpu-status.json',dict(time=time.time(),phase=phase,done=done,healthy_rate=healthy))
    def memory():
        peak = torch.cuda.max_memory_reserved(); snap = admission['snapshot']
        need(safety.memory_ok(peak,snap['desktop_bytes'],snap['total_bytes']), '12 GiB peak and measured 2 GiB headroom gate')
        return peak
    with offline.offline(),runtime.precision():
        ds = base.dataset(ct,ids,2); counts = base.window_counts(ct,ds)
        largest = max(range(977),key=counts.__getitem__)
        for name in ORDER:
            heartbeat('load_'+name,0)
            net = None; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            net = ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
            net.load_state_dict(torch.load(ROOT/CHECKPOINTS[name][0],map_location='cpu',weights_only=True),strict=True)
            net = net.to('cuda').eval().requires_grad_(False)
            def predict(i):
                img,meta,index = ds[i]
                need(index==i and len(img)==counts[i], 'study/window identity')
                return extract(net,runtime.transfer_inputs(img,meta,'cuda'),fp16=True)
            heartbeat('memory_'+name,0); predict(largest); peak = memory()
            offsets = np.r_[0,np.cumsum(counts,dtype=np.int64)]
            # Memmaps preserve successfully written partial logits on watchdog abort.
            win = np.lib.format.open_memmap(dest/(name+'-partial-windows.npy'),mode='w+',dtype=np.float32,shape=(sum(counts),12))
            pooled = np.lib.format.open_memmap(dest/(name+'-partial-pooled.npy'),mode='w+',dtype=np.float32,shape=(977,12))
            start = time.monotonic(); healthy = 0.; max_error = 0.
            for i in range(977):
                p,w,proof = predict(i); peak=max(peak,memory())
                pooled[i] = p; win[offsets[i]:offsets[i+1]] = w
                max_error = max(max_error,proof['pooled_max_abs'])
                if i==49: healthy=50/(time.monotonic()-start)
                heartbeat(name,i+1,healthy)
            pooled.flush(); win.flush()
            path = dest/(name+'.npz')
            heartbeat('export_'+name,977)
            np.savez(path,ids=np.asarray(ids),offsets=offsets,pooled=pooled,windows=win)
            output[name] = dict(sha256=sha(path),checkpoint_sha256=CHECKPOINTS[name][1],
                seconds=time.monotonic()-start,total_windows=sum(counts),max_windows=max(counts),
                peak_reserved=peak,pooled_max_abs=max_error,counts=counts)
            write(OUT/'gpu-progress.json',dict(completed=output))
        write(OUT/'gpu-receipt.json',dict(status='COMPLETE',outputs=output,
            preregistration_sha256=sha(OUT/'PREREGISTRATION.json'),review_sha256=sha(OUT/'REVIEW.json'),
            run_id=admission['job_run_id'],clean_oof=False,rule=RULE))


def evaluate():
    validate(); require_review()
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    base.import_trainer()  # pin Windows DLL before torch; no CUDA context
    from tools import exp156_k_sweep as rulers
    receipt = read(OUT/'gpu-receipt.json'); admission = read(OUT/'gpu-admission.json'); terminal = read(OUT/'gpu-terminal.json')
    need(receipt['status']=='COMPLETE' and receipt['rule']==RULE and set(receipt['outputs'])==set(ORDER) and
         receipt['preregistration_sha256']==sha(OUT/'PREREGISTRATION.json') and receipt['review_sha256']==sha(OUT/'REVIEW.json') and
         receipt['run_id']==admission['job_run_id']==terminal['run_id'] and terminal['status']=='SUCCESS' and terminal['seconds']<=5400,
         'complete authenticated successful bounded attempt required')
    ids = cohort(); baseline=[]; changed=[]
    counts = read(OUT/'cpu-smoke.json')['counts']
    for name in ORDER:
        item = receipt['outputs'][name]
        need(item['checkpoint_sha256']==CHECKPOINTS[name][1] and item['counts']==counts and
             item['pooled_max_abs']<=RULE['reconstruction_fp16_atol'], 'member/count/parity receipt mismatch')
        p,c = load_dump(OUT/'dumps'/(name+'.npz'),item['sha256'],ids,counts)
        baseline.append(p); changed.append(c)
    m = metrics(trio(changed),trio(baseline),rulers.references(ids,root=ROOT))
    result = dict(status='GO' if go_rule(m) else 'NO_GO',rule=RULE,metrics=m,
        mean_macro_delta=sum(v['delta'] for v in m['rulers'].values())/4,
        receipt_sha256=sha(OUT/'gpu-receipt.json'),qualified_for_submission=False,
        next='Independent result review and B decision; no serving build or automatic adoption')
    write(OUT/'RESULT.json',result)
    return dict(status=result['status'],mean_macro_delta=result['mean_macro_delta'],rulers=m['rulers'],diagnostics=m['diagnostics'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','validate','prepare','admit-dry-run','run','worker','evaluate'))
    command=parser.parse_args().command
    if command=='prepare':
        from tools.k1_pooling_cpu_v1 import prepare
        result=prepare()
    else:
        result={'register':register,'validate':validate,'admit-dry-run':admit,'run':run,'worker':worker,'evaluate':evaluate}[command]()
    print(json.dumps(result,indent=2,allow_nan=False))
