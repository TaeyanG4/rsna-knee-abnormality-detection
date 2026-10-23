"""H-B-147 fold1 confirmation v1.0.0, 2026-10-14.

Changelog: new immutable non-PL L2 fold1 stage, fresh 224 controls, strict
confirmation gate, smoke-bound review and sequential supervised dispatch.
No pilot source, receipt or coordination file is modified.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tools.hb150_io_v1 import atomic_json

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.local/scratch/astra-b3/H-B-147/fold1-confirm-v1'
PILOT = ROOT / '.local/scratch/astra-b3/H-B-147/watchdog-v2/configs'
CORPUS = '.local/scratch/astra-b3/H-B-147/native448'
FOLD = 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold1'
GPU = 'NVIDIA GeForce RTX 4070 Ti SUPER'
VERSION = '1.0.0'
GATE = dict(macro=.001, weak4=.001, same_rulers=3, synovitis_floor=-.005,
            all_cell_floor=-.005, cell_scope='each seed and seed-mean predictions',
            aggregate='AUC of arithmetic seed-mean predictions', gold58='descriptive')
SOURCES = ['tools/hb147_confirm_'+s+'_v1.py' for s in ('build','fit','readout','watchdog')]
SOURCES += ['tools/hb150_io_v1.py', 'tools/hb147_progress_v1.py',
 'tools/gpu_job.py','tools/job_runtime.py','training/arms/native_resolution_v2.py',
 'training/arms/mil_screen_hooks_v3.py','training/arms/mil_screen_hooks_v1.py',
 'training/arms/mil_screen_v1.py','training/arms/mil_small.py','training/arms/mil.py',
 'training/arms/lifecycle.py','training/src/raptor_common.py','training/src/raptor_runtime.py',
 'training/src/raptor_exp185.py','training/src/raptor_corpus.py',
 'tools/hb147_native_cache_v1.py','tools/multi_lens_readout_l4_v2.py','tools/multi_lens_readout.py',
 'training/arms/mil_registration.py','training/arms/weak_rank_v1.py',
 'tools/exp156_k_sweep.py','tools/exp125_radr50_k16_local.py']

def need(ok, why):
    if not ok: raise ValueError(why)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()
def uid_hash(ids): return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def immutable(p, value):
    if Path(p).exists(): need(read(p)==value, 'immutable binding changed: '+str(p))
    else: atomic_json(p,value)
def member(res,seed):
    need(res in (224,320) and seed in (42,43),'unregistered member')
    return f'native{res}-s{seed}'
def config_path(res,seed): return WORK/'configs'/(member(res,seed)+'.json')

def roster():
    d=read(ROOT/FOLD/'ordered-ids.json')
    with (ROOT/FOLD/'group-metadata.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    a,b=d['train'],d['val']
    need((len(a),len(b))==(3479,864) and len(set(a))==len(a) and len(set(b))==len(b)
         and not set(a)&set(b),'fold1 roster differs')
    need(len(rows)==len(a)+len(b),'group roster size differs')
    for split, ids in [('train',a),('val',b)]:
        need({r['StudyInstanceUID'] for r in rows if r['split']==split}==set(ids),'group membership differs')
    need(not {r['component_id'] for r in rows if r['split']=='train'} &
         {r['component_id'] for r in rows if r['split']=='val'},'component crosses fold')
    canonical=read(ROOT/'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json')
    positions={u:i for i,u in enumerate(canonical['train']+canonical['val'])}
    need(all(all(positions[y]>positions[x] for x,y in zip(ids,ids[1:])) for ids in (a,b)), 'canonical order differs')
    return a,b

def expected(res,seed):
    pilot=PILOT/(member(res,seed)+'.json'); p=read(pilot); recipe=copy.deepcopy(p['config'])
    need(p['fold']==0 and p['seed']==seed and recipe['windows']['res']==res,'pilot was not fold0')
    train,val=roster(); r=recipe['roots']
    r.update(ordered=FOLD+'/ordered-ids.json',groups=FOLD+'/group-metadata.csv',
             n_train=len(train),n_val=len(val),train_sha256=uid_hash(train),val_sha256=uid_hash(val),order_policy='base-canonical')
    recipe['updates'].update(loader_batches=len(train)//8,steps=len(train)//8*20,max_skips=len(train)//8*20//100)
    paths=set(SOURCES)|{rel(pilot),rel(WORK/'PROTOCOL.md'),rel(WORK/'launch.ps1'),
          FOLD+'/ordered-ids.json',FOLD+'/group-metadata.csv',CORPUS+'/ids.npy',CORPUS+'/masks.npy'}
    paths.update(p['inputs']['data'])
    paths.add(recipe['roots']['canonical'])
    inputs={path:sha(ROOT/path) for path in sorted(paths)}
    # Byte hashes only for L2; no label rows are parsed by registration/smokes.
    need(all(inputs[path]==digest for path,digest in p['inputs']['data'].items()),'pilot input changed')
    manifest=read(ROOT/CORPUS/'manifest.json')
    need(manifest['finished'] and manifest['shape_total']==[4410,64,448,448]
         and manifest['contract']=='hb147_raptor64_448_c140_s02_v1','native cache differs')
    for name in ('ids','masks'): need(inputs[CORPUS+'/'+name+'.npy']==manifest[name+'_sha256'],'native index changed')
    return dict(schema='hb147-confirm/v1',version=VERSION,experiment='H-B-147-fold1-confirm',
       arm=f'native{res}',seed=seed,fold=1,pilot=rel(pilot),recipe=recipe,train=train,val=val,gold=[],
       weights=recipe['labels']['pos_weight'],label_path=recipe['labels']['path'],
       label_semantic_sha256=recipe['labels']['semantic_sha256'],inputs=inputs,
       corpus=CORPUS,resolution=res,epochs=20,batch=8,microbatch=2,train_k=12,workers=2,
       optimizer_steps=len(train)//8*20,selection='fixed_epoch20',gpu_name=GPU,gate=GATE,
       label_mode='non-PL L2; frozen pilot positive weights',clean_oof=False,
       exposure='L2 report-derived; no clean-ground-truth claim',review_required=True)

def register():
    for res in (224,320):
        for seed in (42,43): immutable(config_path(res,seed),expected(res,seed))
    return dict(status='REGISTERED',fold=1,train=3479,val=864,members=4,steps=8680)

def validate(path):
    c=read(path); need(c==expected(c['resolution'],c['seed']),'config or source binding differs')
    return c

def prepare():
    bindings={}
    for res in (224,320):
        for seed in (42,43):
            p=config_path(res,seed); validate(p); bindings[rel(p)]=sha(p)
    immutable(WORK/'prepared.json',dict(schema='hb147-confirm-prepared/v1',fold=1,bindings=bindings))
    return dict(status='PREPARED',members=4)

def review_bindings():
    prepare(); paths=[WORK/'prepared.json']
    for res in (224,320):
        for seed in (42,43):
            cp=config_path(res,seed); paths.append(cp)
            for device in ('cpu','cuda'):
                path=WORK/device/member(res,seed)/'summary.json'
                need(path.is_file(),device+' smoke missing')
                s=read(path)
                need(s.get('passed') is True and s.get('smoke') is True and s.get('engineering_only') is True
                     and s.get('config_sha256')==sha(cp) and s.get('device')==device,'smoke binding differs')
                need(s['optimizer_steps']==(1 if device=='cpu' else 2),'smoke updates differ')
                if device=='cpu': need(s.get('cpu_cuda_initialized') is False,'CPU initialized CUDA')
                else:
                    need(s.get('gpu_name')==GPU and s.get('full_shape') is True and
                         0<s.get('peak_reserved_bytes',0)<=10*(1<<30),'full-shape <=10GiB 4070 required')
                    need(s['batch']==8 and s['val_batch']==4 and s['validation_shapes']==['k24','k36','k36flip'],
                         'full train/TTA shape differs')
                    need(all(s['healthy_rates'].get(k,0)>0 for k in ('train_step','validation')),'healthy rates absent')
                paths.append(path)
    return {rel(p):sha(p) for p in paths}

def require_fit(path,review):
    c=validate(path)
    need(review is not None and Path(review).is_file(),'independent review artifact required')
    r=read(review)
    need(r.get('schema')=='hb147-confirm-review/v1' and r.get('verdict')=='PASS' and
         r.get('independent') is True and bool(r.get('reviewer')),'independent PASS required')
    need(r.get('bindings')==review_bindings(),'review/smoke bindings differ')
    return c

def admit(review=None):
    try:
        for res in (224,320):
            for seed in (42,43): require_fit(config_path(res,seed),review)
        return dict(status='ADMITTED')
    except (OSError,KeyError,ValueError) as exc: return dict(status='HOLD',reason=str(exc))

def python_processes(output):
    """nvidia-smi query-compute-apps includes desktop C+G; only Python blocks."""
    rows=list(csv.reader(output.splitlines())); found=[]
    for row in rows:
        if len(row)<2: continue
        name=row[1].strip().replace('\\','/').split('/')[-1].lower()
        if name in ('python','python.exe','pythonw.exe','python3','python3.exe'):
            found.append(dict(pid=row[0].strip(),name=name))
    return found

def admission_probe():
    s=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                     capture_output=True,text=True,timeout=5,check=True)
    blocked=python_processes(s.stdout)
    return dict(status='HOLD' if blocked else 'PYTHON_PROCESS_CHECK_CLEAR',python_processes=blocked,
                note='read-only telemetry; gpu_job atomic lock is authoritative')

def sequence(mode,review=None,attempt='v1'):
    need(mode in ('smoke','fit') and attempt.replace('-','').isalnum(),'invalid sequence')
    if mode=='fit': need(admit(review)['status']=='ADMITTED','admission HOLD')
    for seed in (42,43):
        for res in (224,320):
            name=member(res,seed); out=WORK/('cuda' if mode=='smoke' else 'fits')/name
            cmd=[sys.executable,'-B','-m','tools.hb147_confirm_watchdog_v1','--attempt',
                 str(WORK/'attempts'/f'{mode}-{name}-{attempt}'),'--budget',str(WORK/'gpu-budget.json')]
            if mode=='smoke': cmd+=['--measurement']
            else:
                rates=WORK/'cuda'/name/'healthy-rates.json'
                immutable(rates,read(WORK/'cuda'/name/'summary.json')['healthy_rates'])
                cmd+=['--healthy-rates',str(rates)]
            cmd+=['--',sys.executable,'-B','-m','tools.hb147_confirm_fit_v1',mode,
                  '--config',str(config_path(res,seed)),'--out',str(out),'--device','cuda']
            if mode=='fit': cmd+=['--review',str(review)]
            subprocess.run(cmd,cwd=ROOT,check=True)
    return dict(status='FOUR_'+mode.upper()+'_COMPLETE')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('register','validate','prepare','admit','admission-probe','review-bindings','smoke-sequence','fit-sequence'))
    ap.add_argument('--review',type=Path); ap.add_argument('--attempt',default='v1'); a=ap.parse_args()
    if a.command=='register': result=register()
    elif a.command in ('validate','prepare'): result=prepare()
    elif a.command=='admit': result=admit(a.review)
    elif a.command=='admission-probe': result=admission_probe()
    elif a.command=='review-bindings': result=review_bindings()
    else: result=sequence(a.command.split('-')[0],a.review,a.attempt)
    print(json.dumps(result))

if __name__=='__main__': main()
