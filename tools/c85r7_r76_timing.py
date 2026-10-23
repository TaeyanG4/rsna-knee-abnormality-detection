"""H-B-175 local timing v1.0.0. B-only supervised 4070 N32; never embedded.

Reuses HB153 v2.1.3 supervisor, preparation and measurement instrumentation.
Adds exact candidate bindings, sequential R7/R6 arms, mandatory real-output
replay, 6.73h baseline / 9h ceiling and stricter progress/throughput guards.
No GPU operation occurs in register, validate, prepare --dry-run or admit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import c85r7_r224r76fq_v1131 as b
from tools import c85r7_flipall_timing as shared
from tools import c85r7_r224_timing as calibration
from tools import hb145_output_io as io

VERSION='1.0.0'
TAG='HB175_N32'
WORK=b.SCRATCH/'timing'
ORDER=('flip-r7','flip-r6')
BASELINE_HOURS=6.73
LIMIT_HOURS=9.
R7_PIN='c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6'
need=b.old.need


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def configure():
    shared.WORK=WORK
    shared.__file__=__file__
    shared.CALIBRATING=False


def transfer():
    value=calibration.transfer_calibration()
    value['assumption']='Same frozen R7/R6 architecture and renderer. Transfer authenticated R7 plain N32-to-hidden scale to measured R6 flip elapsed seconds. Hidden study-count assumption and local 4070-to-Linux/T4 scheduling remain unqualified; no speedup credit.'
    return value


def bindings():
    val=b.validate()
    need(sha(b.R6_LOCAL/'manifest.json')==b.R6_SHA,'R6 manifest changed')
    need(sha(ROOT/'storage/external/heliosli-r7cnxfsss-v1/manifest.json')==R7_PIN,'R7 manifest changed')
    smoke=io.read_json(b.SCRATCH/'smoke.json')
    need(smoke['status']=='PASS' and smoke['kernel_sha256']==val['kernel_sha256'],'CPU smoke missing/stale')
    names=['c85r7_r76_timing.py','c85r7_r76_cpu_smoke.py','c85r7_r224r76fq_v1131.py',
           'c85r7_flipall_timing.py','c85r7_flipall_measure.py','c85r7_r224_timing.py',
           'hb149_monitor_v125.py','hb145_output_io.py']
    return dict(kernel_sha256=val['kernel_sha256'],tools={n:sha(ROOT/'tools'/n) for n in names},
                smoke_sha256=sha(b.SCRATCH/'smoke.json'),r6_manifest_sha256=b.R6_SHA,
                r7_manifest_sha256=R7_PIN,
                parent_receipt_sha256=b.PARENT_RECEIPT_SHA,calibration=transfer(),
                cohort_sha256=calibration.digest(shared.cohort()[0]))


def decision(score,projected_hours):
    need(all(math.isfinite(v) for v in (score,projected_hours)) and 0<=score<=1 and projected_hours>=BASELINE_HOURS,'invalid decision inputs')
    return dict(keep=score>=.956 and projected_hours<=LIMIT_HOURS,
                score_floor=.956,runtime_ok=projected_hours<=LIMIT_HOURS,
                runtime_margin_hours=LIMIT_HOURS-projected_hours,
                qualified_for_submission=False)


def project(r6_seconds,scale):
    need(all(math.isfinite(x) and x>0 for x in (r6_seconds,scale)),'positive measured timing required')
    increment=r6_seconds*scale/3600
    total=BASELINE_HOURS+increment
    return dict(baseline_hours=BASELINE_HOURS,r6_increment_hours=increment,projected_hours=total,
                margin_hours=LIMIT_HOURS-total,hard_limit_hours=LIMIT_HOURS,
                runtime_ok=total<=LIMIT_HOURS,qualified_for_submission=False)


def review(path):
    need(path is not None and Path(path).is_file(),'independent timing review required')
    value=io.read_json(path)
    need(value.get('verdict')=='PASS' and value.get('reviewer') not in (None,'','astra-b5','builder'),
         'independent PASS required')
    need(value.get('bindings')==bindings(),'review bindings stale')
    report=ROOT/value['report_path']
    need(sha(report)==value['report_sha256'],'independent report hash mismatch')
    need(value.get('scope')=='HB175_N32_LOCAL_MEASUREMENT_ONLY','wrong review scope')
    return value


def register():
    value=dict(identity=TAG,version=VERSION,state='HOLD_INDEPENDENT_REVIEW',bindings=bindings(),
               order=ORDER,baseline_hours=BASELINE_HOURS,hard_limit_hours=LIMIT_HOURS,
               measured_increment=False,available_increment_hours=2.27,
               review_instructions='Independent reviewer creates review.json: verdict PASS, reviewer (not astra-b5), scope HB175_N32_LOCAL_MEASUREMENT_ONLY, report_path, report_sha256, bindings copied exactly. No placeholder verdict belongs in the review report.',
               initial_memory='Same-shape R7 historical full-window measurements; R6 measurement is calibration only until its actual <=12GiB and desktop+peak+2GiB guards pass.',
               exposure='EXPOSED',clean_oof=False,qualified_for_submission=False)
    io.atomic_json(WORK/'registration.json',value)
    return {k:v for k,v in value.items() if k not in ('bindings',)}


def validate():
    expected=bindings()
    reg=io.read_json(WORK/'registration.json')
    need(reg['bindings']==expected,'timing registration stale')
    return dict(status='PASS_CPU_TIMING_IDENTITY',kernel_sha256=expected['kernel_sha256'])


def lane_admission():
    import psutil
    parents=psutil.Process().parents()
    from tools.job_runtime import LOGS,job_status
    job=job_status(LOGS,TAG,tail=False)
    need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in parents) and
         job['state']=='running' and job['pid'] in {p.pid for p in parents},'HB175_N32 gpu_job lock required')
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                         capture_output=True,text=True,check=True,timeout=5)
    need(not shared.gpu_python_pids(query.stdout,{os.getpid(),*[p.pid for p in parents]}),'foreign Python GPU process')
    desktop=shared.desktop_snapshot()
    shared.live_memory_guard(desktop['used_mib'],desktop)
    paired=io.read_json(calibration.R7_PAIRED)
    peaks=[row['peak_reserved_bytes'] for arm in paired['per_mode'].values() for row in arm['rows']]
    shared.memory_admission(desktop,peaks)
    return desktop['uuid']


def progress_guard(now,last_progress,history,healthy):
    need(now-last_progress<900,'no progress for 15 minutes')
    if len(history)>1:
        eligible=[p for p in history if now-p[0]>=1800]
        if eligible:
            old=eligible[-1]
            rate=(history[-1][1]-old[1])*60/(now-old[0])
            need(rate>=.25*healthy,'30-minute throughput below 25% of measured healthy rate')


def wddm_guard(now,last_progress,gpu,total_mib):
    memory,util,power=gpu
    need(not (now-last_progress>=600 and util>=95 and power<90 and memory>=total_mib-3072),
         'WDDM near-limit low-power stall for 10 minutes')


def runtime_check(value,arm):
    folder=b.R6_LOCAL if arm=='flip-r6' else ROOT/'storage/external/heliosli-r7cnxfsss-v1'
    manifest=io.read_json(folder/'manifest.json')
    expected=dict(models=manifest['models'],models2=[],flip_tta=True,workers=2,
                  **{k:manifest[k] for k in ('size','crop','views','slices')})
    need(value.get('runtime')==expected and len(expected['models'])==4,'wrong full-shape runtime')


def supervise(arm,healthy,started):
    original=shared.idle_guard
    sample=shared.monitor._gpu_sample
    latest=[]
    desktop=shared.desktop_snapshot()
    def sampled(env):
        value=sample(env)
        latest[:]=value
        return value
    history=[(time.perf_counter(),0)]
    def guarded(now,last_progress,last_activity):
        if arm!='prepare':
            # Only this named status journal may be read while the child runs.
            try:
                status=io.read_json(WORK/(arm+'-measure-status.json'))
                count=status['done']
                if not history: history.append((now,count))
                elif count!=history[-1][1]: history.append((now,count))
            except (OSError,ValueError,KeyError): pass
        progress_guard(now,last_progress,history,healthy)
        if latest: wddm_guard(now,last_progress,latest,desktop['total_mib'])
        original(now,last_progress,last_activity)
    shared.idle_guard=guarded
    shared.monitor._gpu_sample=sampled
    try: return shared.supervise(arm,healthy,started)
    finally:
        shared.idle_guard=original
        shared.monitor._gpu_sample=sample


def arm_run(arm):
    from tools.c85r7_r76_cpu_smoke import runtime
    mod=runtime()
    source=b.SCRATCH/'candidate-runtime/exp169_r7_src'
    deps=ROOT/'.local/scratch/claude-hb128/runtime/deps'
    os.environ['PYTHONPATH']=os.pathsep.join([str(source.parent),str(deps),str(ROOT)])
    os.environ['HB153_MEASURE_STATUS']=str(WORK/(arm+'-measure-status.json'))
    frozen=mod.r7.diagnostic_source
    def instrument(*a,**kw):
        directory=frozen(*a,**kw)
        path=directory/'compact_infer.py'
        io.retry(path.write_text,shared.measure.instrument(path.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
        return directory
    mod.r7.diagnostic_source=instrument
    r6=arm=='flip-r6'
    bundle=b.R6_LOCAL if r6 else ROOT/'storage/external/heliosli-r7cnxfsss-v1'
    kwargs=dict(bundle_id=b.R6_ID,dataset_slug=b.R6_PRIVATE,expected_manifest_sha256=b.R6_SHA) if r6 else dict(expected_manifest_sha256=R7_PIN)
    ids=io.read_json(WORK/'cohort.json')['studies']
    mod.r7.infer_studies([WORK/'data/train_series'/uid for uid in ids],bundle_dir=bundle,
        code_dir=ROOT/'.local/exp189-cpu-v120/input/cloud_code',
        anatomy_weights=ROOT/'.local/exp189-cpu-v120/input/anatomy-fast.pt',
        output_dir=WORK/'pr7-run'/('s_r6' if r6 else 's_r7'),source_dir=source,
        workers=2,amp='auto',device='cuda',max_fallback_fraction=1.,**kwargs)


def real_output_replay():
    """Read only after both owned GPU children stopped; no artificial predictions."""
    import numpy as np
    import pandas as pd
    from tools.c85r7_r76_cpu_smoke import runtime
    mod=runtime()
    ids=io.read_json(WORK/'ids.json')
    arrays=[]
    files={}
    for member,bid,slug,pin in [('S_R7',mod.r7.BUNDLE_ID,mod.r7.DATASET_SLUG,R7_PIN),
                               ('S_R6',b.R6_ID,b.R6_PRIVATE,b.R6_SHA)]:
        uids,_,values,rec=mod.authenticated_pr7(WORK,member=member,bundle_id=bid,dataset_slug=slug,manifest_pin=pin)
        need(uids==ids and len(ids)==32 and not rec['failures'][member],'incomplete REAL N32 predictions')
        need(rec['r7_identity']['worker_args'] and '--flip-tta' in rec['r7_identity']['worker_args'],'flip not attested')
        arrays.append(values)
        directory=WORK/'pr7-run'/member.lower()
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.csv'): files[str(path.relative_to(WORK))]=sha(path)
    slot,failed=mod.r224.s_slot(*arrays,np.zeros(32,bool),np.zeros(32,bool))
    expected=.5*pd.DataFrame(arrays[0]).rank(pct=True).to_numpy()+.5*pd.DataFrame(arrays[1]).rank(pct=True).to_numpy()
    need(not failed.any() and np.array_equal(slot,expected),'REAL R7/R6 slot replay differs')
    result=dict(identity=TAG,status='PASS_REAL_OUTPUT_REPLAY',studies=32,files=files,
                slot_sha256=hashlib.sha256(slot.tobytes()).hexdigest(),bindings=bindings(),
                scope='Actual full-study R7/R6 slot only; not a full R/R224/Own inference replay',
                exposure='EXPOSED',clean_oof=False,qualified_for_submission=False)
    io.journal(WORK/'real-output-replay.json',result,WORK)
    return result


def run(review_path):
    reviewed=review(review_path)
    device=lane_admission()
    need(not (WORK/'attempt.json').exists(),'preserve previous attempt; B must archive explicitly')
    started=time.time()
    state=dict(identity=TAG,state='RUNNING',controller_pid=os.getpid(),attempt_started=started,
               bindings=bindings(),review_path=str(Path(review_path).resolve()),review_sha256=sha(review_path),
               device_uuid=device,completed={},exposure='EXPOSED',clean_oof=False)
    io.journal(WORK/'attempt.json',state,WORK)
    pair=io.read_json(calibration.R7_PAIRED)
    healthy=min(32*60/a['seconds'] for a in pair['per_mode'].values())
    try:
        supervise('prepare',healthy,started)
        ids=io.read_json(WORK/'cohort.json')['studies']
        need(ids==shared.cohort()[0] and len(ids)==32,'N32 cohort changed')
        io.atomic_json(WORK/'ids.json',ids)
        from tools.c85r7_r76_cpu_smoke import runtime
        mod=runtime()
        for arm in ORDER:
            need(lane_admission()==device,'device changed')
            r6=arm=='flip-r6'
            bundle=b.R6_LOCAL if r6 else ROOT/'storage/external/heliosli-r7cnxfsss-v1'
            kw=dict(bundle_id=b.R6_ID,dataset_slug=b.R6_PRIVATE,expected_manifest_sha256=b.R6_SHA) if r6 else dict(expected_manifest_sha256=R7_PIN)
            _,identity=mod.r7.verify_bundle(bundle,source_dir=b.SCRATCH/'candidate-runtime/exp169_r7_src',**kw)
            io.atomic_json(WORK/'pr7-run'/('r6-identity.json' if r6 else 'r7-identity.json'),identity)
            need(not (WORK/(arm+'-measure-status.json')).exists() and
                 not (WORK/'pr7-run'/('s_r6' if r6 else 's_r7')).exists(),'stale arm output')
            elapsed=supervise(arm,healthy,started)
            measured=shared.measurements(arm,ids)
            runtime_check(measured,arm)
            timing=io.read_json(WORK/(arm+'-timing.json'))
            need(not timing['error'] and timing['exit_code']==0 and not timing['cleanup'],'child not cleanly stopped')
            shared.memory_admission(timing['desktop_before_cuda'],[r['peak_reserved_bytes'] for r in measured['rows']],timing['peak_gpu_mib']*2**20)
            state['completed'][arm]=dict(measured,seconds=elapsed,desktop_before_cuda=timing['desktop_before_cuda'],peak_total_vram_bytes=timing['peak_gpu_mib']*2**20)
            io.journal(WORK/'attempt.json',state,WORK)
        shared.paired_memory_admission(state['completed'])
        real_output_replay()
        projection=project(state['completed']['flip-r6']['seconds'],transfer()['scale'])
        result=dict(identity=TAG,status='PASS_LOCAL_RUNTIME_ONLY' if projection['runtime_ok'] else 'DO_NOT_SUBMIT',
                    **projection,bindings=bindings(),per_mode=state['completed'],
                    replay_sha256=sha(WORK/'real-output-replay.json'),review_sha256=sha(review_path),
                    exposure='EXPOSED',clean_oof=False)
        io.journal(WORK/'result.json',result,WORK)
        state['state']=result['status']
        return {k:v for k,v in result.items() if k not in ('bindings','per_mode')}
    except BaseException as exc:
        state.update(state='ABORTED_DO_NOT_SUBMIT',error=str(exc),partial_attempt_preserved=True)
        raise
    finally: io.journal(WORK/'attempt.json',state,WORK)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','validate','prepare','admit','run','replay','_prepare','_arm'])
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--review',type=Path)
    ap.add_argument('--arm',choices=['prepare',*ORDER])
    ap.add_argument('--attempt-started',type=float)
    args=ap.parse_args()
    configure()
    if args.command in ('register','validate','prepare','admit'):
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.command=='register': result=register()
    elif args.command=='validate': result=validate()
    elif args.command=='prepare':
        need(args.dry_run,'CPU prepare requires --dry-run')
        validate(); shared.prepare_dry_run(); return 0
    elif args.command=='admit':
        need(args.dry_run,'measurement admission is dry-run only; run rechecks review')
        validate(); review(args.review)
        result=dict(status='PASS_LOCAL_MEASUREMENT_REVIEW_ONLY',qualified_for_submission=False)
    elif args.command in ('_arm','_prepare'):
        state=io.read_json(WORK/'attempt.json')
        need(state['state']=='RUNNING' and state['identity']==TAG and
             state['attempt_started']==args.attempt_started and state['controller_pid']==os.getppid() and
             state['bindings']==bindings(),'authenticated supervisor required')
        need(sha(state['review_path'])==state['review_sha256'],'review changed during attempt')
        review(Path(state['review_path']))
        if args.command=='_prepare': shared.prepare_data()
        else:
            need(args.arm in ORDER,'registered arm required'); arm_run(args.arm)
        return 0
    elif args.command=='replay':
        state=io.read_json(WORK/'attempt.json')
        need(state['state'] in ('PASS_LOCAL_RUNTIME_ONLY','DO_NOT_SUBMIT'),'stopped complete attempt required')
        result=real_output_replay(); result={k:v for k,v in result.items() if k not in ('files','bindings')}
    else: result=run(args.review)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print('HOLD / DO_NOT_SUBMIT: '+str(exc),file=sys.stderr)
        raise SystemExit(2)
