"""HB177 Gold58 v1.0.0: preregistered exposed standalone paired diagnostic.

Changelog: serial HB177/HB156 exact serving, authenticated cached R7 standalone,
study-paired bootstrap. GPU execution requires B's gpu_job and independent review.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import hb177_postfit_v1 as b
from tools import hb150_io_v1 as io
VERSION='1.0.0'
REG=b.OUT/'gold-registration.json'
WORK=b.SCRATCH/'gold'
TAG='HB177_GOLD58_S42'
BUILDER='hb177-postfit'
GPU_PYTHON='C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
R7_SOURCE=ROOT/'.local/scratch/astra-b1/hb167-gold-bind/gold/v1.0.5'
R7_PINS=R7_SOURCE.parent/'v1.0.6/preserved-pins.json'
R7_SHA='a3debbd5a122a0763756fcbacf0c6e85e0f60d11ffefab776d10805c4309dcd6'
WEAK=(3,5,6,8)
POLICY=dict(draws=2000,seed=17742,unit='same study indices for all three standalone arms',
    invalid='reject entire resample if any target lacks either class',max_attempts=200000,
    interval='percentile 95%, numpy linear quantile',weak_indices=list(WEAK),
    scope='EXPOSED diagnostic; clean_oof=false; no release decision or label-based tuning')
SOURCES=['hb177_gold58_v1.py','hb177_postfit_v1.py','hb177_postfit_smoke_v1.py',
    'hb113_gold58.py','hb158_scale_contrast_v1.py','forensics_1015_v1.py','c85r7_r224flip_diagnostics.py',
    'hb149_monitor_v125.py','c85r7_r224_timing.py','c85r7_flipall_timing.py','hb150_io_v1.py']


def register():
    b.validate()
    from tools import hb113_gold58 as gold
    from tools.hb158_scale_contrast_v1 import CORPUS
    ids,_=gold.gold()
    b.need(b.sha(R7_PINS)==R7_SHA,'R7 historical seal drift')
    reg=dict(version=VERSION,ids=ids,targets=list(gold.LABELS),policy=POLICY,
        exposure='EXPOSED',clean_oof=False,checkpoint_sha256=b.CHECKPOINT_SHA,
        parent_checkpoint_sha256=b.read(b.PARENT/'build-receipt.json')['files'][b.NAME],
        kernel_sha256=b.validate()['kernel_sha256'],
        files={str(p.relative_to(ROOT)):b.sha(p) for p in [gold.GOLD,gold.SERIES,R7_PINS,CORPUS/'manifest.json']},
        sources={n:b.sha(ROOT/'tools'/n) for n in SOURCES},
        serving='HB177/HB156 exact kernel definitions; native cache K94 320 plain+anatomical mirror probability mean; R7 sealed plain-1 standalone',
        comparisons=['HB177 minus HB156 standalone','HB177 minus R7 standalone'],
        metrics=['macro AUC','weak-four AUC'],fit_result_review='REQUIRED before GPU',builder=BUILDER)
    if REG.exists(): b.need(b.read(REG)==reg,'immutable Gold registration drift')
    else: b.write(REG,b.encoded(reg))
    return dict(status='REGISTERED',studies=58,registration_sha256=b.sha(REG))


def validate():
    reg=b.read(REG)
    b.need(reg['version']==VERSION and reg['policy']==POLICY and reg['builder']==BUILDER,'Gold policy drift')
    for name,pin in reg['files'].items(): b.need(b.sha(ROOT/name)==pin,'Gold input drift')
    for name,pin in reg['sources'].items(): b.need(b.sha(ROOT/'tools'/name)==pin,'Gold source drift: '+name)
    from tools import hb113_gold58 as gold
    ids,_=gold.gold()
    b.need(reg['ids']==ids and reg['targets']==list(gold.LABELS),'Gold roster/target drift')
    b.need(reg['kernel_sha256']==b.validate()['kernel_sha256'],'Gold kernel drift')
    return reg


def r7_values(reg):
    import numpy as np
    from tools import c85r7_r224flip_diagnostics as d
    old,_,_,read=d.preserved_context(R7_SOURCE,R7_PINS,R7_SHA)
    value,warnings,_=d.preserved_arm(R7_SOURCE,'plain-1',old,read)
    ids=old['binding']['ids']
    b.need(len(ids)==len(set(ids))==58 and set(ids)==set(reg['ids']),'R7 exact same58 required')
    values=np.asarray(value['predictions'],float)
    values=values[[ids.index(uid) for uid in reg['ids']]]
    probabilities(values,58)
    return values,warnings


def probabilities(x,n):
    import numpy as np
    b.need(x.shape==(n,12) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all(),'invalid probabilities')


def paired(y,arms,draws=2000,seed=17742):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y); n=len(y)
    b.need(y.shape==(n,12) and np.isin(y,[0,1]).all(),'binary complete12 labels required')
    b.need(set(arms)=={'HB177','HB156','R7'},'standalone comparison inventory')
    for x in arms.values(): probabilities(x,n)
    def metric(index):
        return {name:roc_auc_score(y[index],x[index],average=None) for name,x in arms.items()}
    point=metric(np.arange(n)); rng=np.random.default_rng(seed)
    samples={name:[] for name in ('HB156','R7')}; accepted=attempts=0
    while accepted<draws:
        attempts+=1; b.need(attempts<=POLICY['max_attempts'],'bootstrap too few valid draws')
        index=rng.integers(0,n,n)
        counts=y[index].sum(0)
        if np.any((counts==0)|(counts==n)): continue
        values=metric(index)
        for name in samples:
            delta=values['HB177']-values[name]
            samples[name].append([float(delta.mean()),float(delta[list(WEAK)].mean())])
        accepted+=1
    return dict(arms={name:dict(macro=float(v.mean()),weak_four=float(v[list(WEAK)].mean())) for name,v in point.items()},
        paired={name:dict(delta=[float((point['HB177']-point[name]).mean()),float((point['HB177']-point[name])[list(WEAK)].mean())],
            ci95=np.quantile(rows,[.025,.975],axis=0,method='linear').T.tolist()) for name,rows in samples.items()},
        accepted=accepted,attempts=attempts,seed=seed,metric_order=['macro','weak_four'],exposure='EXPOSED',clean_oof=False)


def review_binding():
    return dict(registration_sha256=b.sha(REG),cpu_smoke_sha256=b.sha(b.OUT/'cpu-smoke.json'),tool_sha256=b.sha(__file__))


def fit_review():
    fit=b.read(b.OUT/'fit-result-review-binding.json')
    report=(ROOT/fit['report']).resolve()
    b.need(report.is_relative_to(ROOT) and not any(p.lower() in
        {'.ssh','.kaggle','.codex','api-keys','relabel'} for p in report.parts),'unsafe fit review path')
    b.need(fit['verdict']=='PASS' and fit['reviewer'] not in ('',BUILDER) and
        fit['checkpoint_sha256']==b.CHECKPOINT_SHA and b.sha(report)==fit['sha256'],
        'independent checkpoint-bound fit result PASS required')
    text=report.read_text(encoding='utf-8')
    b.need(re.search(r'(?m)^Verdict: PASS\s*$',text) and b.CHECKPOINT_SHA in text,
        'fit report must state exact checkpoint and Verdict: PASS')
    return fit


def admit(dry=False):
    validate()
    value=b.read(b.OUT/'gold-independent-review.json')
    b.need(value['verdict']=='PASS' and value['builder']==BUILDER and value['reviewer']!=BUILDER
           and value['binding']==review_binding(),'independent current Gold review required')
    report=(ROOT/value['report_path']).resolve()
    b.need(report.is_relative_to(ROOT) and not any(p.lower() in
        {'.ssh','.kaggle','.codex','api-keys','relabel'} for p in report.parts),'unsafe Gold review path')
    b.need(b.sha(report)==value['report_sha256'],'Gold review report drift')
    b.need(re.search(r'(?m)^Verdict: PASS\s*$',report.read_text(encoding='utf-8')),'Gold report must state Verdict: PASS')
    fit_review()
    return dict(status='ADMITTED_DRY_RUN' if dry else 'ADMITTED',binding=review_binding())


def launch_command():
    return [GPU_PYTHON,'-B',str(Path(__file__).resolve()),'run']


def worker():
    reg=validate(); admit()
    import psutil
    ancestors=psutil.Process().parents()
    b.need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in ancestors) and
           any(str(Path(__file__).resolve()).lower() in ' '.join(p.cmdline()).lower() and 'run' in p.cmdline() for p in ancestors),
           'supervised gpu_job worker required')
    from tools import hb177_postfit_smoke_v1 as serving
    from tools.hb158_scale_contrast_v1 import volume
    from tools import c85r7_flipall_timing as memory
    import numpy as np
    import torch
    torch.set_num_threads(2)
    device='cuda:0'; out=WORK/'gpu'; arrays={}; measurements=[]; sequence=0; started=time.monotonic()
    committed=0
    def progress(phase):
        nonlocal sequence
        sequence+=1
        io.atomic_json(out/'status.json',dict(sequence=sequence,phase=phase,committed=committed,elapsed=time.monotonic()-started),critical=False)
    desktop=b.read(out/'desktop.json')
    for arm in ('HB177','HB156'):
        progress(arm+':load'); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ns,model,_=serving.load(arm,device); values=[]
        for i,uid in enumerate(reg['ids']):
            progress(arm+':decode'); vol,mask,_=volume(uid)
            progress(arm+':infer')
            # Use the exact two-view public entry point on GPU, including chunking/precision.
            x=ns['gpu_windows'](vol,mask,94,320,device)
            m=ns['gpu_windows'](vol,mask,94,320,device,mirror=True)
            value=ns['infer_pair_probs'](model,x,m,device)
            values.append(value); probabilities(np.asarray(values),i+1)
            torch.cuda.synchronize(); peak=torch.cuda.max_memory_reserved()
            memory.memory_admission(desktop,[peak])
            measurements.append(dict(arm=arm,index=i,peak_reserved_bytes=peak))
            io.atomic_json(out/(arm+'-partial.json'),dict(ids=reg['ids'][:i+1],values=np.asarray(values).tolist()))
            committed+=1
            progress(arm+':committed'); del x,m,vol,mask
        arrays[arm]=np.asarray(values).tolist(); del model,ns; gc.collect()
    io.atomic_json(out/'worker.json',dict(state='INFERENCE_COMPLETE',registration_sha256=b.sha(REG),
        ids=reg['ids'],targets=reg['targets'],predictions=arrays,memory=measurements,seconds=time.monotonic()-started))
    progress('INFERENCE_COMPLETE')


def finalize(telemetry):
    reg=validate(); out=WORK/'gpu'; w=b.read(out/'worker.json')
    b.need(w['state']=='INFERENCE_COMPLETE' and w['registration_sha256']==b.sha(REG) and
        w['ids']==reg['ids'] and w['targets']==reg['targets'],'stale/incomplete Gold worker')
    from tools.c85r7_flipall_timing import memory_admission
    b.need(telemetry['complete'],'complete GPU telemetry required')
    memory_admission(b.read(out/'desktop.json'),[m['peak_reserved_bytes'] for m in w['memory']],telemetry['peak_gpu_mib']*2**20)
    import numpy as np
    from tools.hb113_gold58 import gold
    ids,y=gold(); b.need(ids==reg['ids'],'label alignment')
    rv,warnings=r7_values(reg)
    result=paired(y,{**{n:np.asarray(v) for n,v in w['predictions'].items()},'R7':rv})
    result.update(status='COMPLETE_EXPOSED_DIAGNOSTIC',registration_sha256=b.sha(REG),worker_sha256=b.sha(out/'worker.json'),
        r7_seal_sha256=R7_SHA,r7_cleanup_warnings=warnings,telemetry=telemetry)
    io.atomic_json(out/'result.json',result)
    return dict(status=result['status'],result=str(out/'result.json'))


def run():
    admit()
    from tools import hb149_monitor_v125 as monitor
    from tools import c85r7_r224_timing as guards
    from tools import c85r7_flipall_timing as memory
    from tools.job_runtime import job_status,LOGS
    import psutil
    out=WORK/'gpu'; parents=psutil.Process().parents()
    job=job_status(LOGS,TAG,expected_command=launch_command(),tail=False)
    b.need(job['state']=='running' and job['pid'] in {p.pid for p in parents},'exclusive exact-command gpu_job required')
    b.need(not out.exists(),'GPU output immutable; no implicit retry')
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5)
    b.need(not memory.gpu_python_pids(query.stdout,{os.getpid(),*[p.pid for p in parents]}),'other Python GPU owner')
    desktop=memory.desktop_snapshot(); io.atomic_json(out/'desktop.json',desktop)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    telemetry=guards._TimingTelemetry(env,.1,desktop); child=None; owner=None; tracked={}
    last=time.monotonic(); sequence=None; rate=None; baseline=last
    try:
        telemetry.start()
        with (out/'worker.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([*launch_command()[:-1],'_worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            owner=monitor._WindowsJob(child)
            while child.poll() is None:
                try: status=b.read(out/'status.json')
                except (OSError,ValueError): status={}
                if status.get('sequence')!=sequence: last=time.monotonic(); sequence=status.get('sequence')
                snapshot=telemetry.snapshot(); b.need(not snapshot['memory_failure'],snapshot['memory_failure'])
                guards.watchdog(time.monotonic(),last,snapshot['latest_gpu'],desktop)
                now=time.monotonic(); completed=status.get('committed',0)
                if rate is None and completed>=2:
                    rate=monitor.RateGuard(completed*60/(now-baseline),now)
                    rate.history.clear(); rate.history.append((now,completed))
                if rate is not None:
                    reason=rate.update(now,completed)
                    b.need(reason is None,reason)
                try:
                    process=psutil.Process(child.pid)
                    owned=[process]+process.children(recursive=True)
                    for proc in owned: tracked[proc.pid]=proc
                    b.need(sum(getattr(p.memory_info(),'private',p.memory_info().rss) for p in owned)<monitor.RAM_LIMIT_BYTES,
                           'owned process RAM guard')
                except psutil.NoSuchProcess:
                    b.need(child.poll() is not None,'owned process vanished while worker running')
                time.sleep(1)
            b.need(child.returncode==0,'Gold worker failed; inspect completed log')
    except BaseException as exc:
        io.atomic_json(out/'failure.json',dict(status='FAILED',error=str(exc))); raise
    finally:
        try:
            if child is not None: b.need(not monitor._cleanup(child,tracked,owner,psutil),'owned cleanup incomplete')
        finally: telemetry.close()
    return finalize(telemetry.snapshot(final=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','validate','admit','review-binding','r7-check','run','_worker'])
    p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
    if a.command not in ('run','_worker'): b.cpu_only()
    if a.command=='register': result=register()
    elif a.command=='validate': validate(); result=dict(status='PASS_CPU_GOLD_VALIDATION')
    elif a.command=='admit': result=admit(a.dry_run)
    elif a.command=='review-binding': result=review_binding()
    elif a.command=='r7-check':
        reg=validate(); values,warnings=r7_values(reg)
        result=dict(status='PASS_R7_STANDALONE_IDENTITY',studies=len(values),warnings=warnings,scores_read=False)
    elif a.command=='run': result=run()
    else: worker(); return
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
