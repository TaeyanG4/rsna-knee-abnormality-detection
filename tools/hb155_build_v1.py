"""H-B-155 v1.2.0: explicit exposed-pair decision and full-cache QA admission.

Changelog 1.2.0: bind registered corrupt-source exclusions and side-v2 metadata; new receipts.
Changelog 1.1.0: B's recorded fold2 decision; isolated v1.1.0 receipts; mandatory
full-cache thresholds and B's 24-overlay review. Existing receipts stay immutable.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tools.hb150_io_v1 import atomic_json

ROOT=Path(__file__).resolve().parents[1]
BASE_WORK=ROOT/'.local/scratch/p65q-v1116/H-B-155'
WORK=ROOT/'.local/scratch/slot-b/hb155/pilot-v120'
SMOKE_CACHE=BASE_WORK/'crop-smoke-v2'
VERSION='1.2.0'
DECISION_ID='B-20261015-H-B-155'
COORDINATOR_DECISION=dict(schema='hb155-coordinator-decision/v1',decision_id=DECISION_ID,
    coordinator='B',recorded_at='2026-10-15 01:5x KST',source='explicit user-provided B coordinator decisions',
    fold=2,exposure='EXPOSED',clean_oof=False,selection_exposure='identical for both paired arms',
    estimand='paired crop-minus-control delta on exposed fold 2',
    geometry='R7 slab-prior knee centre accepted subject to full-cache QA and B overlay review',
    capacity='second RTX 4090 with >=250 GB disk; H-B-151 remains on current 4090',
    remaining_gates='full cache QA, independent overlay review, runtime parity and independent smoke-bound fit review')
PILOT=ROOT/'.local/scratch/astra-b3/H-B-147/watchdog-v2/configs'
CONTROL='.local/scratch/astra-b3/H-B-147/native448'
CROP='.local/scratch/slot-b/hb155/crop-cache-v220'
FOLD='.local/scratch/slot-b/hb155/pilot-v120/fold2'
GPU='NVIDIA GeForce RTX 4090'
GATE=dict(macro=.002,rulers='all four',both_seeds_positive=True,weak4=0.,all_cell_floor=-.003,
          cell_scope='every seed and arithmetic seed-mean predictions, every ruler',
          aggregate='AUC of arithmetic seed-mean predictions',gold58='descriptive only')
SOURCES=['tools/hb155_'+s+'_v1.py' for s in ('build','fit','watchdog','crop','readout','transfer')]
SOURCES+=['tools/hb155_tar_transfer_v1.py','tools/hb155_cache_v2.py','tools/hb155_control_v1.py','tools/hb155_exclusions_v1.py']
SOURCES+=['training/arms/hb155_crops_v1.py','tools/parity_runtime.py','config/workspace.toml',
 'tools/hb150_io_v1.py','tools/hb147_progress_v1.py','tools/gpu_job.py','tools/job_runtime.py',
 'training/arms/native_resolution_v2.py','training/arms/mil_screen_hooks_v3.py',
 'training/arms/mil_screen_hooks_v1.py','training/arms/mil_screen_v1.py','training/arms/mil_small.py',
 'training/arms/mil.py','training/arms/lifecycle.py','training/src/raptor_common.py',
 'training/src/raptor_runtime.py','training/src/raptor_exp185.py','training/src/raptor_corpus.py',
 'tools/multi_lens_readout_l4_v2.py','tools/multi_lens_readout.py','training/arms/mil_registration.py',
 'training/arms/weak_rank_v1.py','tools/exp156_k_sweep.py','tools/exp125_radr50_k16_local.py']

def need(ok,why):
    if not ok: raise ValueError(why)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()
def uid_hash(ids): return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def immutable(p,value):
    if Path(p).exists(): need(read(p)==value,'immutable binding changed: '+str(p))
    else: atomic_json(p,value)
def member(arm,seed):
    need(arm in ('control','crop') and seed in (42,43),'unregistered member')
    return f'{arm}-s{seed}'
def config_path(arm,seed,engineering=False):
    return WORK/('cpu-configs' if engineering else 'configs')/(member(arm,seed)+'.json')

def decision_binding(record=None):
    """Record only the exact authorized decision; never infer approval from a PASS."""
    path=WORK/'coordinator-decision.json'
    if record is not None:
        need(record==DECISION_ID,'unknown coordinator decision')
        immutable(path,COORDINATOR_DECISION)
    need(path.is_file(),'explicit coordinator decision missing; register with --coordinator-decision '+DECISION_ID)
    need(read(path)==COORDINATOR_DECISION,'coordinator decision differs from recorded authorization')
    return {rel(path):sha(path)}

def source_closure():
    """Explicit local Python dependency closure, including legacy bare imports."""
    found=set(SOURCES); todo=[p for p in found if p.endswith('.py')]
    while todo:
        path=todo.pop(); tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import): names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    package=path.split('/')[:-node.level]
                    base='.'.join(package+([base] if base else []))
                names=[base]+[base+'.'+a.name for a in node.names if a.name!='*']
            for name in names:
                stem=name.replace('.','/')
                for prefix in ('','training/src/','tools/'):
                    for suffix in ('.py','/__init__.py'):
                        candidate=prefix+stem+suffix
                        if candidate not in found and (ROOT/candidate).is_file():
                            found.add(candidate); todo.append(candidate)
    return sorted(found)

def roster():
    from tools.hb155_exclusions_v1 import verify_fold
    a,b=verify_fold()
    need(len(a)+len(b)==4343 and len(set(a+b))==4343,'fold2 roster differs')
    with (ROOT/FOLD/'group-metadata.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    need(len(rows)==4343,'group count differs')
    for split,ids in [('train',a),('val',b)]:
        need({r['StudyInstanceUID'] for r in rows if r['split']==split}==set(ids),'group membership differs')
    need(not {r['component_id'] for r in rows if r['split']=='train'} &
         {r['component_id'] for r in rows if r['split']=='val'},'component crosses fold')
    canonical=read(ROOT/'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json')
    positions={u:i for i,u in enumerate(canonical['train']+canonical['val'])}
    need(all(all(positions[y]>positions[x] for x,y in zip(ids,ids[1:])) for ids in (a,b)),'noncanonical order')
    return a,b

def expected(arm,seed,engineering=False):
    from tools.hb155_exclusions_v1 import CONTRACT, RECORD, OUT, HB145_RECORD, load
    import numpy as np
    pilot=PILOT/f'native224-s{seed}.json'; p=read(pilot); recipe=copy.deepcopy(p['config'])
    train,val=roster(); corpus=CONTROL if arm=='control' else CROP
    if engineering:
        train,val=train[:2],val[:1]
    contract='hb147_raptor64_448_c140_s02_v1' if arm=='control' else CONTRACT
    recipe['roots'].update(ordered=FOLD+'/ordered-ids.json',groups=FOLD+'/group-metadata.csv',
        corpus=corpus,manifest=corpus+'/manifest.json',n_train=len(train),n_val=len(val),
        train_sha256=uid_hash(train),val_sha256=uid_hash(val),order_policy='base-canonical')
    recipe['windows']['native_source']=contract
    steps=len(train)//8*20
    recipe['updates'].update(loader_batches=len(train)//8,steps=steps,max_skips=steps//100)
    # Allowlist proof: only population/cache roots, update count and source contract.
    proof=copy.deepcopy(recipe); proof['roots']=p['config']['roots']; proof['windows']['native_source']=p['config']['windows']['native_source']
    for key in ('loader_batches','steps','max_skips'): proof['updates'][key]=p['config']['updates'][key]
    need(proof==p['config'],'unregistered scientific recipe change')
    paths=set(source_closure())|set(p['inputs']['data'])|{rel(pilot),FOLD+'/ordered-ids.json',FOLD+'/group-metadata.csv',recipe['roots']['canonical']}
    inputs={path:sha(ROOT/path) for path in sorted(paths)}
    need(all(inputs[path]==digest for path,digest in p['inputs']['data'].items()),'4070 input changed')
    exclusion=load()
    inputs.update(exclusion['bindings'])
    inputs.update({rel(p):sha(p) for p in (RECORD,OUT/'fold-binding.json',HB145_RECORD)})
    inputs.update(decision_binding())
    QA_THRESHOLDS=read(ROOT/CROP/'qa-summary.json')['thresholds']
    return dict(schema='hb155-config/v1',version=VERSION,experiment='H-B-155',arm=arm,seed=seed,fold=2,
        side_unknown_policy='authenticated cache flag retained as metadata; model keeps three adjacent intensity channels; unknown low/high LPS X order, no anatomical claim',
        exclusions_sha256=sha(RECORD),recipe=recipe,train=train,val=val,gold=[],weights=recipe['labels']['pos_weight'],
        label_path=recipe['labels']['path'],label_semantic_sha256=recipe['labels']['semantic_sha256'],inputs=inputs,
        corpus=corpus,corpus_contract=contract,resolution=224,epochs=20,batch=8,microbatch=2,accumulation=4,
        train_k=12,workers=2,optimizer_steps=steps,selection='fixed_epoch20',gpu_name=GPU,gate=GATE,
        engineering_only=engineering,holdout_status='EXPOSED_PAIRED_FOLD2_AUTHORIZED',
        coordinator_decision=COORDINATOR_DECISION,qa_thresholds=QA_THRESHOLDS,
        estimand=COORDINATOR_DECISION['estimand'],selection_exposure='identical for both paired arms',
        clean_oof=False,exposure='EXPOSED',exposure_details='L2 report-derived; R7 geometry-only prior; selection-exposed fold',review_required=True)

def register(coordinator_decision=None):
    decision_binding(coordinator_decision)
    for engineering in (False,True):
        for arm in ('control','crop'):
            for seed in (42,43): immutable(config_path(arm,seed,engineering),expected(arm,seed,engineering))
    return dict(status='REGISTERED_EXPOSED',members=4,engineering_members=4,fold=2,
        exposure='EXPOSED',clean_oof=False,estimand=COORDINATOR_DECISION['estimand'],
        reason='Recorded B decision; full-cache QA, runtime and independent smoke-bound review still required.')
def validate(path):
    c=read(path); need(c==expected(c['arm'],c['seed'],c['engineering_only']),'config/source binding differs'); return c

def cache_binding(c):
    import numpy as np
    directory=ROOT/c['corpus']; m=read(directory/'manifest.json')
    need(m.get('finished') is True and m['contract']==c['corpus_contract'],'cache incomplete/foreign')
    names=('ids','masks','side_unknown') if c.get('arm')=='crop' or m['contract']=='hb155_compartment64_448_side_v2' else ('ids','masks')
    for name in names: need(sha(directory/(name+'.npy'))==m[name+'_sha256'],'cache index changed')
    ids=np.load(directory/'ids.npy',allow_pickle=False).astype(str).tolist()
    need(len(set(ids))==len(ids) and set(c['train']+c['val'])<=set(ids),'cache roster missing')
    need(m['shape_total']==[len(ids),64,448,448],'cache shape differs')
    end=0
    for s in m['shards']:
        need(s['done'] and s['start']==end and s['stop']>end and Path(s['file']).name==s['file'],'invalid shard roster')
        need((directory/s['file']).stat().st_size==s['bytes'],'cache shard missing/truncated')
        end=s['stop']
    need(end==len(ids),'incomplete shard roster')
    return {rel(directory/'manifest.json'):sha(directory/'manifest.json')}

def prepare(engineering=False):
    bindings={}; missing=[]
    for arm in ('control','crop'):
        for seed in (42,43):
            path=config_path(arm,seed,engineering); c=validate(path); bindings[rel(path)]=sha(path)
            try: bindings.update(cache_binding(c))
            except (OSError,ValueError) as exc: missing.append(type(exc).__name__+': '+str(exc))
    if missing: return dict(status='HOLD',reasons=sorted(set(missing)))
    immutable(WORK/('cpu-prepared.json' if engineering else 'prepared.json'),dict(schema='hb155-prepared/v1',bindings=bindings))
    return dict(status='PREPARED_ENGINEERING' if engineering else 'PREPARED',members=4)

def runtime_bindings():
    from tools.parity_runtime import diff_runtime
    paths=[WORK/'runtime'/name for name in ('4090.json','4070.json','kaggle.json','vs4070.json','vskaggle.json')]
    target,a,k,da,dk=[read(p) for p in paths]
    need(da==diff_runtime(a,target) and dk==diff_runtime(k,target),'runtime diff stale')
    need(a.get('gpu')=='NVIDIA GeForce RTX 4070 Ti SUPER' and 'T4' in k.get('gpu',''),'baseline GPU identity absent')
    need(target.get('gpu')==GPU and target.get('capability')==[8,9],'target is not 4090 sm89')
    need(target.get('torch')=='2.5.1+cu121' and target.get('cuda')=='12.1','torch/CUDA differs')
    for name,value in dict(timm='1.0.19',pydicom='3.0.2',pylibjpeg='2.1.0',numpy='2.0.2',pandas='2.2.2',sklearn='1.9.0').items():
        need(target.get('libs',{}).get(name)==value,'pin differs: '+name)
    for key,value in dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,
            cudnn_deterministic=True,deterministic_algorithms=True,flash_sdp=True,mem_efficient_sdp=True,math_sdp=True).items():
        need(target.get(key)==value,'runtime policy differs: '+key)
    return {rel(p):sha(p) for p in paths}

def require_runtime(config,snapshot,review=None):
    validate(config)
    r=read(WORK/'runtime-review.json'); bindings=runtime_bindings()
    need(r.get('schema')=='hb155-runtime-review/v1' and r.get('verdict')=='PASS' and r.get('independent') is True
         and bool(r.get('reviewer')) and r.get('bindings')==bindings,'independent runtime review absent/stale')
    from tools.parity_runtime import diff_runtime
    need(snapshot.get('gpu')==GPU and snapshot.get('capability')==[8,9] and
         not diff_runtime(read(WORK/'runtime/4090.json'),snapshot),'runtime changed after review')
    return r

def crop_review_bindings():
    from tools.hb155_exclusions_v1 import QA, RECORD, OUT, load, verify_fold
    load(); verify_fold();q=read(QA)
    need(q.get('studies')==4408 and q.get('qa_pass') is True and q.get('verdict')=='PASS' and
         q.get('excluded_studies')==2 and q.get('clean_oof') is False and q.get('exposure')=='EXPOSED' and
         q.get('overlay_count')==24 and
         q.get('thresholds')==read(ROOT/CROP/'qa-summary.json')['thresholds'] and
         q.get('exclusions_sha256')==sha(RECORD) and q.get('fold_binding_sha256')==sha(OUT/'fold-binding.json') and
         q.get('pixel_hash_verification')=='all_chunks_both_arms' and
         q.get('control_manifest_sha256')==sha(ROOT/CONTROL/'manifest.json') and
         q.get('manifest_sha256')==sha(ROOT/CROP/'manifest.json'), 'exclusion-bound full-cache crop QA absent/stale')
    overlays=q.get('overlay_sha256',{});need(len(overlays)==24,'24 bound overlays required')
    for name,d in overlays.items():
        p=(ROOT/CROP/name).resolve()
        need(p.is_relative_to((ROOT/CROP).resolve()) and sha(p)==d,'bound overlay absent/stale')
    bindings={rel(p):sha(p) for p in (QA,RECORD,OUT/'fold-binding.json',ROOT/CROP/'manifest.json',ROOT/CONTROL/'manifest.json')}
    r=read(WORK/'crop-review.json')
    need(r.get('schema')=='hb155-crop-review/v1' and r.get('verdict')=='PASS' and r.get('independent') is True
         and r.get('reviewer')=='B' and r.get('overlays_reviewed')==24 and r.get('bindings')==bindings,
         'independent B review of 24 overlays and side interpretation absent/stale')
    return {**bindings,rel(WORK/'crop-review.json'):sha(WORK/'crop-review.json')}

def review_bindings():
    need(prepare()['status']=='PREPARED','full paired caches missing')
    paths=[WORK/'prepared.json',WORK/'runtime-review.json']; bindings=runtime_bindings()
    bindings.update(decision_binding()); bindings.update(crop_review_bindings())
    for arm in ('control','crop'):
        for seed in (42,43):
            for device in ('cpu','cuda'):
                cp=config_path(arm,seed,device=='cpu'); path=WORK/device/member(arm,seed)/'summary.json'; s=read(path)
                need(s.get('passed') is True and s.get('smoke') is True and s.get('engineering_only') is True and
                     s.get('config_sha256')==sha(cp) and s.get('device')==device,'smoke identity differs')
                need(s['optimizer_steps']==(1 if device=='cpu' else 2),'smoke steps differ')
                if device=='cpu': need(s.get('cpu_cuda_initialized') is False,'CPU used CUDA')
                else:
                    need(s.get('gpu_name')==GPU and s.get('full_shape') is True and 0<s.get('peak_reserved_bytes',0)<=10*(1<<30),'full shape VRAM gate')
                    need(s['batch']==8 and s['val_batch']==4 and s['validation_shapes']==['k24','k36','k36flip'],'smoke shapes differ')
                    need(all(s['healthy_rates'].get(k,0)>0 for k in ('train_step','validation')),'rates missing')
                paths.extend([cp,path])
    bindings.update({rel(p):sha(p) for p in paths}); return bindings

def require_fit(path,review=None):
    c=validate(path)
    need(not c['engineering_only'],'engineering config cannot fit')
    bindings=decision_binding()
    need(c.get('holdout_status')=='EXPOSED_PAIRED_FOLD2_AUTHORIZED' and c.get('fold')==2 and
         c.get('exposure')=='EXPOSED' and c.get('clean_oof') is False and
         c.get('coordinator_decision')==COORDINATOR_DECISION and
         c.get('estimand')==COORDINATOR_DECISION['estimand'] and
         c.get('selection_exposure')=='identical for both paired arms' and
         all(c.get('inputs',{}).get(p)==digest for p,digest in bindings.items()),
         'explicit registered EXPOSED paired-fold decision absent or changed; D-B-157 exposure cannot be waived')
    need(review is not None,'independent review required'); r=read(review)
    need(r.get('schema')=='hb155-review/v1' and r.get('verdict')=='PASS' and r.get('independent') is True and
         bool(r.get('reviewer')) and r.get('bindings')==review_bindings(),'independent smoke-bound review required')
    return c
def admit(review=None):
    try:
        for arm in ('control','crop'):
            for seed in (42,43): require_fit(config_path(arm,seed),review)
        return dict(status='ADMITTED',exposure='EXPOSED',clean_oof=False,estimand=COORDINATOR_DECISION['estimand'])
    except (ValueError,OSError,KeyError) as exc: return dict(status='HOLD',reason=str(exc))

def sequence(mode,review=None,attempt='v1'):
    need(mode in ('smoke','fit') and attempt.replace('-','').isalnum(),'invalid sequence')
    from tools.hb155_transfer_v1 import disk_gate
    disk_gate(WORK,additional_bytes=100*(1<<30))
    if mode=='fit': need(admit(review)['status']=='ADMITTED','admission HOLD')
    for seed in (42,43):
        for arm in ('control','crop'):
            name=member(arm,seed); out=WORK/('cuda' if mode=='smoke' else 'fits')/name
            cmd=[sys.executable,'-B','-m','tools.hb155_watchdog_v1','--attempt',str(WORK/'attempts'/f'{mode}-{name}-{attempt}'),
                 '--budget',str(WORK/'gpu-budget.json')]
            if mode=='smoke': cmd+=['--measurement']
            else:
                rates=WORK/'cuda'/name/'healthy-rates.json'; immutable(rates,read(WORK/'cuda'/name/'summary.json')['healthy_rates'])
                cmd+=['--healthy-rates',str(rates)]
            cmd+=['--',sys.executable,'-B','-m','tools.hb155_fit_v1',mode,'--config',str(config_path(arm,seed)),
                  '--out',str(out),'--device','cuda']
            if review: cmd+=['--review',str(review)]
            subprocess.run(cmd,cwd=ROOT,check=True)
    return dict(status='FOUR_'+mode.upper()+'_COMPLETE')

def snapshot_runtime(out,device='cpu'):
    if device=='cpu':
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
    from tools import hb155_fit_v1 as f
    f.h.torch.set_num_threads(2)
    f.h.configure_runtime('none' if device=='cpu' else 'fp16',device,policy='strict')
    snapshot=f.runtime_info(); atomic_json(out,snapshot)
    return dict(status='SNAPSHOT',device=device,path=str(out))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('register','validate','prepare','admit','runtime-bindings','review-bindings','smoke-sequence','fit-sequence','snapshot'))
    ap.add_argument('--engineering',action='store_true'); ap.add_argument('--review',type=Path); ap.add_argument('--attempt',default='v1')
    ap.add_argument('--coordinator-decision',choices=(DECISION_ID,))
    ap.add_argument('--out',type=Path); ap.add_argument('--device',choices=('cpu','cuda'),default='cpu'); a=ap.parse_args()
    if a.command=='register': result=register(a.coordinator_decision)
    elif a.command=='validate':
        for arm in ('control','crop'):
            for seed in (42,43): validate(config_path(arm,seed,a.engineering))
        result=dict(status='VALID',members=4)
    elif a.command=='prepare': result=prepare(a.engineering)
    elif a.command=='admit': result=admit(a.review)
    elif a.command=='runtime-bindings': result=runtime_bindings()
    elif a.command=='review-bindings': result=review_bindings()
    elif a.command=='snapshot':
        need(a.out is not None,'snapshot needs --out'); result=snapshot_runtime(a.out,a.device)
    else: result=sequence(a.command.split('-')[0],a.review,a.attempt)
    print(json.dumps(result))
    return 2 if result.get('status')=='HOLD' else 0
if __name__=='__main__': raise SystemExit(main())
