"""H-B-170 N32 anatomical mirror ABBA harness 1.0.0.

Changelog: reuse H-B-164 supervised watchdog/cleanup and sealed cohort; actual
pair peak smoke, four sequential passes, real-output CPU validator replay.
Diagnostic only. Never embedded. B alone launches via gpu_job HB170_N32.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import c85r7_r224_timing as t
from tools import c85r7_r224mir_cpu as cpu
from tools import c85r7_r224mirq_v1126 as b

VERSION='1.0.0'
TAG='HB170_N32'
WORK=b.SCRATCH/'timing'
LEGACY=t.WORK
ORDER=('plain-1','mirror-1','mirror-2','plain-2')
HELPERS=('c85r7_r224mir_timing.py','c85r7_r224mir_cpu.py','c85r7_r224mirq_v1126.py',
 'c85r7_r224_timing.py','c85r7_flipall_timing.py','hb145_output_io.py','hb149_monitor_v125.py',
 'gpu_job.py','gpu_queue.py','job_runtime.py','c85r7_flipall_v1115.py',
 'c85r7_flipall_measure.py','c85r7_flipall_baseline.py','c85r7_r224q_v1123.py',
 'c85r7_flipallq_v1122.py','strict_variant_gate.py','strict_variant_gate_r224_v143.py',
 'strict_variant_gate_r224_v142.py','strict_variant_gate_standalone.py',
 'strict_variant_gate_hb151.py','c85r7_r224_cpu_smoke.py')


def stamp(**kw):
    return dict(identity=TAG,version=VERSION,exposure='EXPOSED',clean_oof=False,**kw)


def finite_positive(x):
    return type(x) in (int,float) and math.isfinite(x) and x>0


def validate_pass(value,ids):
    t.need(value.get('identity')==TAG and value.get('mode') in ORDER,'pass identity')
    adapted=dict(value,identity='HB164_R224_N32')
    saved=t.TAG
    try:
        t.TAG='HB164_R224_N32'
        t.validate_measurement(adapted,ids)
    finally: t.TAG=saved
    t.need(finite_positive(value.get('seconds')),'positive pass time required')
    t.need(value['seconds']>=sum(r['seconds'] for r in value['rows'])-1e-6,'pass time shorter than rows')
    for row in value['rows']:
        t.need(finite_positive(row.get('inference_seconds')) and row['inference_seconds']<=row['seconds'], 'invalid inference time')


def summarize(passes,ids,calibration,parent_receipt=None):
    import numpy as np
    from scipy.stats import spearmanr
    t.need([p['mode'] for p in passes]==list(ORDER),'ABBA order drift')
    for p in passes: validate_pass(p,ids)
    plain=np.mean([p['rows'] and [r['probabilities'] for r in p['rows']] for p in (passes[0],passes[3])],axis=0)
    mirror=np.mean([[r['probabilities'] for r in p['rows']] for p in passes[1:3]],axis=0)
    correlations=[]
    for col in range(12):
        val=float(spearmanr(plain[:,col],mirror[:,col]).statistic)
        t.need(math.isfinite(val),'undefined per-target Spearman')
        correlations.append(val)
    increment=max(p['seconds'] for p in passes[1:3])
    local,hidden=calibration['conservative_plain_seconds'],calibration['r7_hidden_seconds']
    t.need(all(finite_positive(x) for x in (increment,local,hidden)),'projection inputs')
    projected=increment*hidden/local
    total=None
    t.need(parent_receipt is None,
        'actual independently reviewed H-B-167 hidden receipt integration is not implemented; HOLD, no self-declared mapping accepted')
    return dict(per_target_spearman=correlations,failed_study_fallbacks=0,
        plain_seconds=[passes[i]['seconds'] for i in (0,3)], mirror_seconds=[p['seconds'] for p in passes[1:3]],
        conservative_increment_seconds=increment,
        inference_only_increment_seconds=max(sum(r['inference_seconds'] for r in p['rows']) for p in passes[1:3]),
        projected_hidden_increment_seconds=projected,projected_hidden_seconds=total,
        runtime_limit_seconds=32400,runtime_pass=None if total is None else total<=32400,
        runtime_state='HOLD_HB167_ACTUAL_HIDDEN_RECEIPT' if total is None else 'REQUIRES_RESULT_REVIEW',
        limitation=calibration['assumption'])


def validate_pair(pair,desktop):
    t.need(pair.get('shape')==[94,3,224,224] and pair.get('simultaneous_windows') is True
        and pair.get('actual_infer_pair_probs') is True and pair.get('failed_study_fallback') is False
        and pair.get('consecutive_studies')==2 and pair.get('retain_until_replacement') is True,
        'actual full pair steady-state smoke required')
    samples=pair['samples']
    t.need(len(samples)==2 and len({s['study'] for s in samples})==2,'two distinct pair studies required')
    for sample in samples:
        t.need(len(sample['probabilities'])==12 and all(math.isfinite(x) and 0<=x<=1 for x in sample['probabilities']), 'pair probabilities')
        t.need(0<sample['peak_allocated_bytes']<=sample['peak_reserved_bytes'],'pair memory measurement')
        t.shared.memory_admission(desktop,[sample['peak_reserved_bytes']])
    t.need(pair['peak_reserved_bytes']==max(s['peak_reserved_bytes'] for s in samples)
        and pair['peak_allocated_bytes']==max(s['peak_allocated_bytes'] for s in samples),'pair peak mismatch')
    t.shared.memory_admission(desktop,[pair['peak_reserved_bytes']])


def pair_smoke(ns,model,prepared,grouped,reader,desktop):
    # Keep both previous windows alive until each assignment completes, exactly
    # as the serving consumer does across consecutive studies. No early del.
    torch=ns['torch']; samples=[]
    for uid in prepared['studies'][:2]:
        torch.cuda.reset_peak_memory_stats()
        vol,mask=ns['build_study'](uid,grouped,str(LEGACY/'data/train_series'),reader)
        t.validate_volume(vol,mask)
        xw=ns['gpu_windows'](vol,mask,94,224,'cuda:0')
        xm=ns['gpu_windows'](vol,mask,94,224,'cuda:0',mirror=True)
        probs=ns['infer_pair_probs'](model,xw,xm,'cuda:0'); torch.cuda.synchronize()
        t.need(list(xw.shape)==[94,3,224,224] and list(xm.shape)==[94,3,224,224],'pair shape mismatch')
        samples.append(dict(study=uid,probabilities=probs.tolist(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        t.shared.memory_admission(desktop,[samples[-1]['peak_reserved_bytes']])
    pair=dict(shape=[94,3,224,224],simultaneous_windows=True,actual_infer_pair_probs=True,
        failed_study_fallback=False,consecutive_studies=2,retain_until_replacement=True,samples=samples,
        peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in samples),
        peak_reserved_bytes=max(s['peak_reserved_bytes'] for s in samples))
    validate_pair(pair,desktop)
    return pair


def replay():
    real=t.io.read_json(LEGACY/'worker-status.json')
    measured=t.io.read_json(LEGACY/'measurement.json')
    ids=t.shared.cohort()[0]
    t.validate_measurement(real,ids)
    t.need(real['binding_sha256']==measured['binding_sha256'] and measured['identity']=='HB164_R224_N32',
        'real measurement/output identity mismatch')
    passes=[]
    for mode in ORDER:
        arm=copy.deepcopy(real)
        arm.update(identity=TAG,mode=mode)
        for row in arm['rows']: row['inference_seconds']=row['seconds']
        passes.append(arm)
    samples=[{k:r[k] for k in ('study','probabilities','peak_allocated_bytes','peak_reserved_bytes')} for r in real['rows'][:2]]
    pair=dict(shape=[94,3,224,224],simultaneous_windows=True,actual_infer_pair_probs=True,
        failed_study_fallback=False,consecutive_studies=2,retain_until_replacement=True,samples=samples,
        peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in samples),
        peak_reserved_bytes=max(s['peak_reserved_bytes'] for s in samples))
    adapted=stamp(state='COMPLETE',done=128,passes=passes,pair=pair,
        rows=[r for p in passes for r in p['rows']],binding_sha256=real['binding_sha256'])
    result=finalize(adapted,measured['telemetry'],measured['desktop_before_cuda'],
        real['binding_sha256'],t.transfer_calibration())
    value=stamp(state='CPU_POSTCHECKS_REPLAY_ONLY',source_sha256=t.sha(LEGACY/'worker-status.json'),
        measurement_sha256=t.sha(LEGACY/'measurement.json'),tool_sha256=t.sha(__file__),result=result,
        limitation='Preserved REAL plain N32 output schema-adapted four times; first two plain rows adapted to pair schema, with real desktop/telemetry. Pair flags are schema adapters only. inference_seconds conservatively uses row wall time. No mirror, ABBA, pair CUDA or performance evidence.')
    t.io.journal(WORK/'postchecks-replay.json',value,WORK)
    return value


def prepare(dry_run=False):
    inv=t.inventory()
    legacy=t.io.read_json(LEGACY/'prepare.json')
    t.need(legacy['inventory_sha256']==t.digest(inv) and legacy['studies']==inv['studies'], 'legacy prepared cohort drift')
    t.need(set(legacy['files'])=={x['target'] for x in inv['members']}, 'prepared file set')
    for name,h in legacy['files'].items(): t.need(t.sha(LEGACY/'data'/name)==h,'prepared data drift')
    probe=t.cpu_decode_probe(inv)
    value=stamp(state='CPU_PREPARED',dry_run=dry_run,inventory_sha256=t.digest(inv),
        prepared_sha256=t.sha(LEGACY/'prepare.json'),tool_sha256=t.sha(__file__),probe=probe,
        data_reused_read_only=True)
    t.io.journal(WORK/'prepare.json',value,WORK)
    return value


def binding():
    validated=b.validate()
    pre=t.io.read_json(WORK/'prepare.json')
    t.need(pre['state']=='CPU_PREPARED' and pre['tool_sha256']==t.sha(__file__),'fresh preparation required')
    t.need(pre['prepared_sha256']==t.sha(LEGACY/'prepare.json') and pre['inventory_sha256']==t.digest(t.inventory()), 'preparation drift')
    legacy=t.io.read_json(LEGACY/'prepare.json')
    for name,h in legacy['files'].items(): t.need(t.sha(LEGACY/'data'/name)==h,'data drift')
    smoke=t.io.read_json(b.SCRATCH/'cpu-mirror.json')
    t.need(smoke['status']=='PASS_CPU_REAL_MIRROR' and smoke['kernel_sha256']==validated['kernel_sha256']
        and smoke['tool_sha256']==t.sha(cpu.__file__), 'fresh CPU smoke required')
    rep=t.io.read_json(WORK/'postchecks-replay.json')
    t.need(rep['state']=='CPU_POSTCHECKS_REPLAY_ONLY' and rep['tool_sha256']==t.sha(__file__)
        and rep['source_sha256']==t.sha(LEGACY/'worker-status.json')
        and rep['measurement_sha256']==t.sha(LEGACY/'measurement.json'),'real postchecks replay required')
    t.need(t.sha(t.CHECKPOINT)==t.CHECKPOINT_SHA,'checkpoint drift')
    return dict(kernel_sha256=validated['kernel_sha256'],checkpoint_sha256=t.CHECKPOINT_SHA,
        spec_sha256=t.sha(b.SCRATCH/'SPEC.md'),cpu_sha256=t.sha(b.SCRATCH/'cpu-mirror.json'),
        timing_addendum_sha256=t.sha(b.SCRATCH/'TIMING-ADDENDUM.md'),
        replay_sha256=t.sha(WORK/'postchecks-replay.json'),prepare_sha256=t.sha(WORK/'prepare.json'),
        helpers={n:t.sha(ROOT/'tools'/n) for n in HELPERS},
        tests=t.sha(ROOT/'tests/test_c85r7_r224mir.py'),calibration=t.transfer_calibration())


def validate():
    value=stamp(state='CPU_VALIDATED',binding=binding())
    value['binding_sha256']=t.digest(value['binding'])
    t.io.journal(WORK/'validate.json',value,WORK)
    return value


def check_review(review,current):
    t.need(review.get('verdict')=='PASS' and review.get('builder')=='hb170-build'
        and bool(review.get('reviewer')) and review['reviewer']!=review['builder']
        and review.get('execution_authority')=='B'
        and review.get('binding_sha256')==t.digest(current)
        and review.get('validate_sha256')==t.sha(WORK/'validate.json'), 'independent bound PASS required')


def admit(review,dry_run=False):
    current=binding()
    t.need(t.io.read_json(WORK/'validate.json')['binding']==current,'fresh validation required')
    t.need(review is not None,'independent review required even for dry-run')
    check_review(t.io.read_json(review),current)
    value=stamp(state='DRY_RUN_PASS' if dry_run else 'ADMITTED',binding=current,binding_sha256=t.digest(current),
        review=str(Path(review).resolve()),review_sha256=t.sha(review))
    t.io.journal(WORK/('admit-dry-run.json' if dry_run else 'admission.json'),value,WORK)
    return value


def admitted():
    saved=t.io.read_json(WORK/'admission.json')
    current=binding()
    t.need(saved['state']=='ADMITTED' and saved['binding']==current and saved['binding_sha256']==t.digest(current),'admission drift')
    t.need(t.sha(saved['review'])==saved['review_sha256'],'review drift')
    check_review(t.io.read_json(saved['review']),current)
    return saved


def install_supervisor():
    # Reuse the reviewed watchdog/process-tree cleanup verbatim, with this task's
    # paths and entrypoint. Does not change a packaged file or legacy source file.
    t.WORK=WORK; t.TAG=TAG; t.VERSION=VERSION; t.__file__=__file__


def worker():
    t.need(os.environ.get('HB164_CONTROLLER_PID')==str(os.getppid()),'supervised worker only')
    admission=admitted()
    install_supervisor()
    desktop=t.lane_admission()
    prepared=t.io.read_json(LEGACY/'prepare.json')
    ns=cpu.namespace(); torch,np=ns['torch'],ns['np']
    t.need(torch.cuda.is_available() and torch.cuda.device_count()==1 and '4070 Ti SUPER' in torch.cuda.get_device_name(0),'one 4070 required')
    torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True
    runtime=t.runtime_snapshot(ns)
    arm=ns['ARMS'][0]
    model,res=ns['load_model'](str(t.CHECKPOINT),arm['arch'],arm['res'],'cuda:0')
    t.need(res==224,'resolution changed')
    reader=ns['_make_reader'](); grouped=t.group_series(prepared['series'])
    rows=[]; passes=[]; pair=None; start=time.perf_counter()
    def publish(state):
        t.io.journal(WORK/'worker-status.json',stamp(state=state,done=len(rows),rows=rows,passes=passes,pair=pair,
            runtime=runtime,binding_sha256=admission['binding_sha256'],seconds=time.perf_counter()-start),WORK)
    publish('MODEL_LOADED')
    pair=pair_smoke(ns,model,prepared,grouped,reader,desktop)
    publish('PAIR_SMOKE_COMPLETE')
    for mode in ORDER:
        begun=time.perf_counter(); armrows=[]
        for uid in prepared['studies']:
            mark=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            vol,mask=ns['build_study'](uid,grouped,str(LEGACY/'data/train_series'),reader)
            t.validate_volume(vol,mask)
            windows=ns['gpu_windows'](vol,mask,94,224,'cuda:0',mirror=mode.startswith('mirror'))
            torch.cuda.synchronize(); inf=time.perf_counter()
            probs,precision=t.observed_inference(ns,model,windows); torch.cuda.synchronize()
            inference=time.perf_counter()-inf
            allocated,reserved=torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
            t.shared.memory_admission(desktop,[reserved])
            t.need(probs.shape==(12,) and np.isfinite(probs).all(),'invalid probabilities')
            row=dict(study=uid,probabilities=probs.tolist(),windows=94,resolution=224,failed_study_fallback=False,
                forward_precision=precision,fp32_retry='float32' in precision,seconds=time.perf_counter()-mark,
                inference_seconds=inference,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved)
            armrows.append(row); rows.append(row); publish('INFERENCE')
            del windows,vol,mask,probs
        passes.append(stamp(state='COMPLETE',done=32,mode=mode,rows=armrows,seconds=time.perf_counter()-begun))
    publish('COMPLETE')
    return 0


def finalize(value,telemetry,desktop,binding_sha256,calibration):
    t.need(value['state']=='COMPLETE' and value['done']==128 and value['binding_sha256']==binding_sha256,'complete bound ABBA required')
    t.need(value['rows']==[r for p in value['passes'] for r in p['rows']],'aggregate rows differ from ABBA passes')
    t.need([s['study'] for s in value['pair']['samples']]==t.shared.cohort()[0][:2],'pair cohort/order drift')
    validate_pair(value['pair'],desktop)
    t.shared.memory_admission(desktop,[r['peak_reserved_bytes'] for r in value['rows']]+[value['pair']['peak_reserved_bytes']],telemetry['peak_gpu_mib']*2**20)
    result=summarize(value['passes'],t.shared.cohort()[0],calibration)
    return dict(state='MEASURED_REQUIRES_RESULT_REVIEW' if telemetry['complete'] else 'HOLD_TELEMETRY',result=result)


def run():
    admission=admitted(); install_supervisor(); desktop=t.lane_admission()
    t.need(not (WORK/'worker-status.json').exists(),'existing attempt: preserve and ask B')
    value,telemetry=t.supervise(desktop)
    result=finalize(value,telemetry,desktop,admission['binding_sha256'],admission['binding']['calibration'])
    receipt=stamp(**result,telemetry=telemetry,runtime=value['runtime'],binding_sha256=admission['binding_sha256'],
        desktop_before_cuda=desktop,raw_output_sha256=t.sha(WORK/'worker-status.json'))
    t.io.journal(WORK/'measurement.json',receipt,WORK)
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','replay','validate','admit','run','_worker'])
    ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--review',type=Path)
    a=ap.parse_args()
    try:
        if a.command in ('run','_worker'):
            t.need(not a.dry_run,'GPU run has no dry-run bypass')
            result=run() if a.command=='run' else worker()
        else:
            os.environ['CUDA_VISIBLE_DEVICES']='-1'
            result={'prepare':lambda:prepare(a.dry_run),'replay':replay,'validate':validate,'admit':lambda:admit(a.review,a.dry_run)}[a.command]()
        print(json.dumps({k:v for k,v in result.items() if k in ('identity','version','state')} if isinstance(result,dict) else result))
        return 0
    except (OSError,ValueError,KeyError,ImportError) as exc:
        print(json.dumps(stamp(state='HOLD',reason=str(exc))))
        return 2


if __name__=='__main__': raise SystemExit(main())
