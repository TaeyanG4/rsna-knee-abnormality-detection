"""H-B-157 R-KD v1.0.2. Public-R on H151 soft targets.

Changelog v1.0.2: unlimited fit duration with activity-aware idle protection;
versioned gpu_job logs and authenticated v1.0.1 CUDA measurement carryover.

Changelog v1.0.1 build 2: versioned real-stage registration and evidence paths;
preserve v1.0.0 history and require fresh independent review before GPU smoke.
Changelog v1.0.1: measured 12 GiB peaks and 2 GiB desktop headroom gate;
recipe unchanged (H-B-151 v3.0.2 / BUILDER-CHECKLIST v1.0.6 item 3).
v1.0.0: fixed six-epoch recipe, fail-closed admission, measured smoke,
epoch-boundary resume, supervised GPU execution. No account operations.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'.local/scratch/hb157-rkd/H-B-157/v1.0.2'
RUN = WORK/'fit'
PROTOCOL = WORK/'protocol.json'
TAG = 'HB157_RKD_V102'
SMOKE_TAG = 'HB157_RKD_SMOKE_V102'
PRIOR = WORK.parent/'v1.0.1'
PRIOR_PROTOCOL_SHA = '51dd754e919046edf537893280309d3703ac76c1ae56bb0a413af04827091c49'
SUPERVISION_CHANGES = {'tools/hb157_rkd_v1.py', 'tools/hb150_watchdog_v1.py'}
MODULE = 'tools.hb157_rkd_v1'
LABELS = '.local/scratch/astra-b4/H-B-151/v3/soft-labels.csv'
LABEL_SHA = 'dbd0e671778fa418f9ffa776a6bcd134e804a70280cb02e101def04b31fb72fb'
PARENT = '.local/rplus-v1/public-R-download.bin'
PARENT_SHA = '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
SPEC = 'artifacts/experiments/H_B_109_RPLUS_PUBLIC/v1.1.0/member-spec.json'
ROSTER = 'artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json'
CORPUS = 'storage/cache/derived/EXP143_RAPTOR96_384_C140_S02'
EXTRA = ['tools/hb157_rkd_v1.py','training/arms/hb157_rkd_fit_v1.py',
         'tools/hb157_rkd_killtest_v1.py','tools/hb157_rkd_fixtures_v1.py','tools/hb151_build_v3.py',
         'tools/hb150_io_v1.py','tools/hb150_watchdog_v1.py',
         'tools/hb147_progress_v1.py','tools/pl_production_watchdog_v1.py',
         'tools/gpu_job.py','tools/job_runtime.py']
from tools.hb150_io_v1 import atomic_json

def need(ok, message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def scientific():
    cfg = copy.deepcopy(read(ROOT/SPEC)['config'])
    cfg.update(epochs=6,total_updates=1632,scope='EXPOSED',clean_oof=False,
               decision='H-B-157 owner 2026-10-15; standalone alive>=0.952 dead<=0.949',
               adoption='STANDALONE_KILL_TEST_BEFORE_R_SLOT_SWAP')
    cfg['recipe'].update(epochs=6,swa_1based=[4,5,6],labels='H-B-151 soft probabilities')
    cfg['continuation']['data_epochs'] = list(range(16,22))
    cfg['bn']['epoch'] = 22
    cfg['supervision']['kind'] = 'H-B-151 soft probabilities; frozen L2 pos_weight'
    cfg['class_weight_recipe'] = 'frozen authenticated R+ v1.1.0 L2 weights; never recomputed from soft labels'
    return cfg

def population():
    from tools.hb151_build_v3 import probability_rows
    p = read(ROOT/ROSTER)
    rows, _ = probability_rows(ROOT/LABELS)
    ids = p['train_ids']
    need(len(ids)==len(set(ids))==4347 and set(rows)==set(ids),'exact 4347-study soft roster required')
    need(len(set(p['excluded_gold_ids']))==58 and not set(ids)&set(p['excluded_gold_ids']),'Gold58 overlap')
    need(p['pos_weight']==scientific()['pos_weight'],'frozen L2 weights drift')
    return ids, rows

def registration():
    prior=read(ROOT/SPEC)
    pins={**prior['inputs']['sources'],ROSTER:prior['inputs']['rosters'][ROSTER],
          CORPUS+'/manifest.json':prior['inputs']['data'][CORPUS+'/manifest.json'],
          LABELS:LABEL_SHA,PARENT:PARENT_SHA}
    for path, pin in pins.items(): need(sha(ROOT/path)==pin,'authenticated input differs: '+path)
    pins.update({p:sha(ROOT/p) for p in [SPEC,*EXTRA]})
    ids,_=population()
    p = dict(schema='hb157-rkd/v1',version='1.0.2',exposure='EXPOSED',clean_oof=False,
                config=scientific(),pins=pins,train_roster_sha256=digest(ids),
                decisions=dict(alive=0.952,dead=0.949,otherwise='INCONCLUSIVE',swap_keep=0.957),
                run_dir=RUN.relative_to(ROOT).as_posix(),workers=4,
                resume='epoch boundary; discard uncommitted epoch; fixed per-epoch seed and sampler',
                gpu='4070 Ti SUPER only',smoke_cap_seconds=165,fit_cap_seconds=None,
                watchdog=dict(idle_seconds=900,require_gpu_inactive=True,stall_seconds=600,
                              collapse_seconds=1800,collapse_fraction=.25),
                gpu_job_tags=dict(fit=TAG,smoke=SMOKE_TAG))
    p['cuda_reuse'] = reuse_descriptor(p)
    return p

def reuse_descriptor(current):
    """Authenticate historical evidence without rewriting its protocol identity."""
    need(sha(PRIOR/'protocol.json')==PRIOR_PROTOCOL_SHA,'prior protocol identity')
    old=read(PRIOR/'protocol.json')
    metadata={'version','pins','run_dir','fit_cap_seconds','watchdog','gpu_job_tags','cuda_reuse'}
    need({k:v for k,v in old.items() if k not in metadata}==
         {k:v for k,v in current.items() if k not in metadata},'reuse scientific metadata drift')
    need(old['pins'].keys()==current['pins'].keys(),'reuse source roster drift')
    need(all(old['pins'][k]==v for k,v in current['pins'].items() if k not in SUPERVISION_CHANGES),
         'reuse training/input pin drift')
    files=['protocol.json','cpu-smoke.json','pre-gpu-review.json','cuda-smoke.json',
           'smoke-attempt/receipt.json','smoke-attempt/command.json',
           'smoke-attempt/identity.json','smoke-attempt/heartbeat.json','cuda-run/model_best.pt']
    check_review(read(PRIOR/'pre-gpu-review.json'),
                 dict(protocol_sha256=PRIOR/'protocol.json',cpu_smoke_sha256=PRIOR/'cpu-smoke.json'))
    s=read(PRIOR/'cuda-smoke.json');check_smoke(s,protocol_sha256=PRIOR_PROTOCOL_SHA)
    need(s['model_sha256']==sha(PRIOR/'cuda-run/model_best.pt'),'reuse model drift')
    need(s['identity']['protocol_sha256']==PRIOR_PROTOCOL_SHA and
         s['identity']['device']=='cuda' and s['identity']['mode']=='smoke','reuse CUDA lineage')
    receipt=read(PRIOR/'smoke-attempt/receipt.json')
    need(receipt['exit_code']==0 and receipt['reason']=='completed' and not receipt['survivors'],
         'reuse supervisor failed')
    command=read(PRIOR/'smoke-attempt/command.json')
    identity=read(PRIOR/'smoke-attempt/identity.json')
    heartbeat=read(PRIOR/'smoke-attempt/heartbeat.json')
    need(command['token']==heartbeat['token'] and identity['pid']==heartbeat['pid'] and
         command['command'][1:]==['-B','-m',MODULE,'_smoke'],'reuse supervisor lineage')
    return dict(schema='hb157-cuda-reuse/v1',source_protocol_sha256=PRIOR_PROTOCOL_SHA,
                allowed_changed_pins=sorted(SUPERVISION_CHANGES),
                files={(PRIOR/f).relative_to(ROOT).as_posix():sha(PRIOR/f) for f in files})

def register():
    p=registration()
    if PROTOCOL.exists(): need(read(PROTOCOL)==p,'registration changed; new version required')
    else: atomic_json(PROTOCOL,p)
    return dict(status='REGISTERED',protocol_sha256=sha(PROTOCOL),studies=4347,clean_oof=False)

def validate():
    need(read(PROTOCOL)==registration(),'registration drift')
    return dict(status='VALIDATED',studies=4347,epochs=6,updates=1632,exposure='EXPOSED',clean_oof=False)

def require_cpu():
    s=read(WORK/'cpu-smoke.json')
    need(s.get('status')=='PASS' and s.get('protocol_sha256')==sha(PROTOCOL) and
         s.get('cuda_initialized') is False and s.get('real_data') is True and
         all(s.get(k) is True for k in ['backward','resume_roundtrip','swa','bn','strict_reload','serving_pair']),
         'CPU evidence missing/stale/incomplete')

def check_smoke(s, *, protocol_sha256=None):
    need(s.get('protocol_sha256')==(protocol_sha256 or sha(PROTOCOL)) and s.get('status')=='PASS','GPU smoke identity')
    need('4070 Ti SUPER' in s.get('device_name','') and 16000<=s.get('total_memory_mib',0)<=16400,'real 4070 measurement required')
    need(s.get('shape')==dict(batch=1,k=12,res=384,accumulation=16,bn_batch=2,k_eval=94,crop=320,mirror=True,precision='bf16',checkpointing=False),'full-shape smoke required')
    need(0<s.get('elapsed_seconds',0)<=165 and s.get('updates')==1 and s.get('bn_batches')==1,'smoke cap/completion')
    desktop=s['desktop_before_cuda']; total=min(desktop['total_mib'],s['total_memory_mib'])*1024**2
    need(math.isfinite(desktop['total_mib']) and 16000<=desktop['total_mib']<=16400 and
         math.isfinite(desktop['used_mib']) and 0<=desktop['used_mib']<desktop['total_mib'],'desktop measurement')
    peaks=[]
    for phase in ['train','bn','serving']:
        m=s['measurements'][phase]; peak=m['peak_reserved_bytes']; rate=m['seconds_per_unit']
        need(math.isfinite(peak) and 0<peak<=12*1024**3 and math.isfinite(rate) and rate>0,'peak/rate gate')
        peaks.append(peak)
    need(desktop['used_mib']*1024**2+max(peaks)+2*1024**3<=total,'desktop + peak + 2 GiB exceeds card')

def check_review(review, bindings):
    need(review.get('verdict')=='PASS' and review.get('independent') is True and
         isinstance(review.get('reviewer'),str) and review['reviewer'].strip() not in ('','hb157-rkd','builder'),
         'recorded independent review required')
    for key,path in bindings.items(): need(review.get(key)==sha(path),'stale/missing review binding: '+key)

def require_pre_review():
    require_cpu()
    check_review(read(WORK/'pre-gpu-review.json'),dict(protocol_sha256=PROTOCOL,cpu_smoke_sha256=WORK/'cpu-smoke.json'))

def require_fit():
    validate(); require_pre_review()
    # registration/validate authenticates the original measurements and full
    # unchanged training/input pin set; reviews bind the new reuse descriptor.
    s=read(PRIOR/'cuda-smoke.json'); check_smoke(s,protocol_sha256=PRIOR_PROTOCOL_SHA)
    r=read(PRIOR/'smoke-attempt/receipt.json')
    need(r['exit_code']==0 and r['reason']=='completed' and not r['survivors'],'smoke supervisor failed')
    check_review(read(WORK/'fit-review.json'),dict(protocol_sha256=PROTOCOL,cpu_smoke_sha256=WORK/'cpu-smoke.json',
        cuda_smoke_sha256=PRIOR/'cuda-smoke.json',supervisor_receipt_sha256=PRIOR/'smoke-attempt/receipt.json',
        source_protocol_sha256=PRIOR/'protocol.json'))
    return s

def require_gpu_job(mode, child=False):
    import psutil
    from tools.job_runtime import file_lock,read_json,command_digest
    from tools.hb151_build_v3 import startup_publication,python_blockers,admission_snapshot
    def check():
        tag=TAG if mode=='fit' else SMOKE_TAG
        with file_lock(ROOT/'artifacts/logs/.gpu-claim.lock',blocking=False):
            job=read_json(ROOT/'artifacts/logs'/tag/'job.json')
            ancestors=[psutil.Process(),*psutil.Process().parents()]
            ids={(p.pid,p.create_time()) for p in ancestors}; direct=job.get('child_identity') or {}
            need((job['pid'],job['create_time']) in ids and (direct.get('pid'),direct.get('create_time')) in ids,'live gpu_job ancestry required')
            expected=[sys.executable.replace('/','\\'),'-B','-m',MODULE,mode]
            need(job['tag']==tag and job['phase']=='running' and job['command'] in ([expected,expected+['--resume']] if mode=='fit' else [expected]) and
                 job['command_sha256']==command_digest(job['command'],ROOT),'gpu_job command mismatch')
            need(not python_blockers(admission_snapshot(),[p.pid for p in ancestors]),'other Python ML compute active')
            if child:
                attempt=WORK/('smoke-attempt' if mode=='smoke' else 'fit-attempt-'+str(read(WORK/'budget.json')['attempt']))
                identity=read(attempt/'identity.json'); command=read(attempt/'command.json')
                need(identity['pid']==os.getpid() and identity['create_time']==psutil.Process().create_time(),'watchdog child identity')
                need(command['token']==os.environ.get('HB147_TOKEN') and Path(os.environ.get('HB147_HEARTBEAT','')).resolve()==(attempt/'heartbeat.json').resolve(),'watchdog token binding')
    startup_publication(check)

def supervise(mode,resume=False):
    from tools.hb150_watchdog_v1 import supervise as watch
    from tools.hb151_build_v3 import desktop_memory_snapshot
    require_gpu_job(mode); validate(); require_pre_review()
    command=[sys.executable.replace('/','\\'),'-B','-m',MODULE,'_'+mode]
    if resume: command.append('--resume')
    log_dir=(ROOT/'artifacts/logs'/(TAG if mode=='fit' else SMOKE_TAG)).relative_to(ROOT).as_posix()
    if mode=='smoke': return watch(command,WORK/'smoke-attempt',total=165,poll=1,gpu_job_log_dir=log_dir)
    smoke=require_fit()
    desktop=desktop_memory_snapshot()
    check_smoke(dict(smoke,desktop_before_cuda=desktop),protocol_sha256=PRIOR_PROTOCOL_SHA)
    previous=read(WORK/'budget.json') if (WORK/'budget.json').exists() else dict(attempt=0,charged_seconds=0,running=False)
    need(not previous['running'],'supervisor loss: B must reconcile prior attempt before resume')
    need(not RUN.exists() or resume,'existing fit requires --resume')
    budget=dict(attempt=previous['attempt']+1,charged_seconds=previous['charged_seconds'],running=True)
    atomic_json(WORK/'budget.json',budget); start=time.monotonic()
    try:
        return watch(command,WORK/('fit-attempt-'+str(budget['attempt'])),total=None,
                     idle_requires_gpu_inactive=True,gpu_job_log_dir=log_dir,
                     healthy_rates={'train_step':1/smoke['measurements']['train']['seconds_per_unit'],
                                    'bn_replay':1/smoke['measurements']['bn']['seconds_per_unit']})
    finally:
        budget.update(running=False,charged_seconds=previous['charged_seconds']+time.monotonic()-start)
        atomic_json(WORK/'budget.json',budget)

def admit():
    from tools.hb151_build_v3 import admission_snapshot,python_blockers,desktop_memory_snapshot
    validate(); require_cpu()
    snapshot=admission_snapshot(); blockers=python_blockers(snapshot)
    pending=[]
    try: require_fit()
    except (OSError,ValueError,KeyError) as exc: pending.append(str(exc))
    result=dict(status='DRY_RUN_HOLD' if blockers or pending else 'DRY_RUN_READY',fit_launched=False,
                python_blockers=blockers,pending=pending,desktop_memory=desktop_memory_snapshot(),cuda_executed=False)
    atomic_json(WORK/'admit-dry-run.json',result)
    return result

def launch(mode,resume=False):
    # B invokes this explicitly; no wait/queue/retry and no caller-selected interpreter.
    executable=sys.executable.replace('/','\\')
    tag=TAG if mode=='fit' else SMOKE_TAG
    command=[executable,'-B',str(ROOT/'tools/gpu_job.py'),'--tag',tag,'--',executable,'-B','-m',MODULE,mode]
    if resume: command.append('--resume')
    return subprocess.call(command,cwd=ROOT)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['register','validate','cpu-smoke','prepare','admit-dry-run','smoke','fit','_smoke','_fit','launch-smoke','launch-fit'])
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action.startswith('launch-'): return launch(args.action[7:],args.resume)
    if args.action in ('fit','smoke'): return supervise(args.action,args.resume)
    if args.action in ('_fit','_smoke'):
        mode=args.action[1:];require_gpu_job(mode,True);validate();require_pre_review()
        if mode=='fit': require_fit()
        from training.arms.hb157_rkd_fit_v1 import run
        run('cuda',mode,args.resume);return 0
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1')
    if args.action=='register': result=register()
    elif args.action=='validate': result=validate()
    elif args.action=='admit-dry-run': result=admit()
    elif args.action=='prepare':
        from tools.hb157_rkd_killtest_v1 import prepare
        validate();result=prepare()
    else:
        validate()
        from training.arms.hb157_rkd_fit_v1 import run
        result=run('cpu','smoke')
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__': raise SystemExit(main())
