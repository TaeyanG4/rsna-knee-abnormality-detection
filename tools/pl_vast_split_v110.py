"""PL mixed-device transport 1.1.1. Production-only parallel loader wiring.

Changelog 1.1.1: configurable vCPU-sized production workers, propagated through
the watchdog, with current loader/train source pins required by split review.

Changelog 1.1.0: export locally validated materialized targets, retain original
scientific identities, authenticate portable paths, and bind mixed GPU exports.
No network or implicit GPU use. Production preparation is B-only after GO.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
import math
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
VERSION = '1.1.1'
SCHEMA = 'pl-vast-split/v1.1.1'
REVIEW_SOURCES = ('tools/pl_vast_split_v110.py', 'tools/pl_production_loader_v1.py',
                  'tools/pl_production_train_v1.py')

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    from tools.hb150_io_v1 import _retry
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with _retry(lambda:path.open('x',encoding='utf-8')) as f: json.dump(value,f,indent=2,sort_keys=True)

def safe(name):
    p=PurePosixPath(name)
    need(name and '\\' not in name and ':' not in name and not p.is_absolute()
         and '..' not in p.parts and not any(x in {'.ssh','.kaggle','.codex','api-keys','relabel'} for x in p.parts), 'unsafe transport path')
    return p.as_posix()

def pin(path): return dict(sha256=sha(path),bytes=Path(path).stat().st_size)
def checked(root,name,expected):
    path=Path(root)/safe(name)
    need(path.resolve().is_relative_to(Path(root).resolve()),'transport escape')
    need(path.is_file() and pin(path)==expected,'transport hash/size mismatch: '+name)
    return path

def split(plan):
    members=plan['members']
    need(len(members)==5 and [m['member']['fold'] for m in members]==list(range(5)), 'exact ordered fivefold roster required')
    need(all(m['member']['seed']==42 and m['member']['kind']=='MIL' for m in members),'seed42 MIL only')
    return dict(version=VERSION,local_folds=[0,1,2],remote_folds=[3,4],
                local_gpu='NVIDIA GeForce RTX 4070 Ti SUPER',remote_gpu='NVIDIA GeForce RTX 3090',
                fallback_folds=[3,4],resume_across_devices=False,
                exposure='EXPOSED_EXPLORATORY',clean_oof=False,serving_parity='T4 qualification required')

def prepare(plan_path, output, review):
    from tools import pl_production_launch_v1 as launch
    from tools import pl_production_train_v1 as train
    need(review is not None and Path(review).is_file(),'independent split review absent')
    plan=read(plan_path); assignment=split(plan)
    for key in ('go','memory','review','teacher'): launch.p.pin(plan[key])
    launch.review_gate(plan['review']['path']); launch.selected(plan['go']['path'])
    output=Path(output); need(not (output/'bundle.json').exists(),'bundle already sealed')
    output.mkdir(parents=True,exist_ok=True)
    files={}
    def add(source,name):
        name=safe(name); dest=output/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest); files[name]=pin(dest); return name
    # Code/config only. No data directories, original label tables, or credentials.
    for folder in ('tools','training','legacy'):
        for source in sorted((ROOT/folder).rglob('*.py')):
            if '__pycache__' not in source.parts: add(source,source.relative_to(ROOT).as_posix())
    add(ROOT/train.base.CONTROL,train.base.CONTROL)
    r=read(review)
    need(r.get('status')=='PASS' and r.get('independent') is True and r.get('reviewer'),'independent split review required')
    need(set(r.get('sources',{}))>=set(REVIEW_SOURCES),'split source review absent')
    for name,digest in r['sources'].items():need(files[safe(name)]['sha256']==digest,'reviewed source drift')
    portable=copy.deepcopy(r)
    for key in ('cpu_suite','real_cpu_smoke'):
        a=r[key];need(pin(a['path'])==dict(sha256=a['sha256'],bytes=a['bytes']),'review evidence drift')
        evidence=read(a['path'])
        need(evidence.get('status')=='PASS','passing CPU evidence required')
        need(all(evidence.get('sources',{}).get(name)==r['sources'][name] for name in REVIEW_SOURCES),
             'current-source CPU evidence required')
        name=add(a['path'],'metadata/'+key+'.json');portable[key]=dict(path=name,**files[name])
    write(output/'metadata/split-review.json',portable);files['metadata/split-review.json']=pin(output/'metadata/split-review.json')
    add(plan['go']['path'],'metadata/go.json')
    members=[]; retained={}
    for entry in plan['members']:
        with train.adapters():
            spec,target,_=train.base.validate(entry['spec'])
            admission=train.base.admit(entry['spec'],plan['go']['path'])
        need(admission['status']=='ADMITTED','scientific GO absent')
        fold=entry['member']['fold']
        members.append(dict(member=entry['member'],spec_sha256=spec['spec_sha256'],
                            local_spec=entry['spec'],local_output=entry['output']))
        source_projection(spec,output)
        if fold not in assignment['remote_folds']: continue
        prefix=f'metadata/fold{fold}'
        members[-1].update(spec=add(entry['spec'],prefix+'/training-spec.json'),
            target=add(spec['target_bundle']['path'],prefix+'/target-spec.json'),
            payload=add(target['target_path'],prefix+'/targets.npz'),admission=admission)
        corpus=Path(spec['corpus']); manifest=read(corpus/'manifest.json')
        need(manifest.get('finished') is True,'incomplete retained corpus')
        for name in ('manifest.json','ids.npy','masks.npy'):
            retained['corpus-hardlinks/'+name]=pin(corpus/name)
        for shard in manifest['shards']:
            need(shard.get('done') is True,'unfinished corpus shard')
            retained['corpus-hardlinks/'+safe(shard['file'])]=dict(sha256=shard['sha256'],bytes=shard['bytes'])
        for key in ('weights','initial_state'):
            name=safe(spec['recipe']['model'][key]); retained[name]=pin(ROOT/name)
    bundle=dict(schema=SCHEMA,version=VERSION,assignment=assignment,arm=plan['arm'],
        original_plan=launch.asset(plan_path),teacher=plan['teacher'],go=plan['go'],
        exposure='EXPOSED_EXPLORATORY',clean_oof=False,members=members,files=files,retained=retained)
    write(output/'bundle.json',bundle)
    with zipfile.ZipFile(output/'transfer.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*files,'bundle.json']): z.write(output/name,name)
    return dict(status='PREPARED',bundle_sha256=sha(output/'bundle.json'),transfer_sha256=sha(output/'transfer.zip'),
                transfer_bytes=(output/'transfer.zip').stat().st_size,**assignment)

def verify(bundle_path, expected, retained_root=None):
    bundle_path=Path(bundle_path)
    need(sha(bundle_path)==expected,'bundle trust anchor mismatch')
    b=read(bundle_path); need(b['schema']==SCHEMA,'bundle schema')
    need(b['assignment']==split(dict(members=b['members'])),'split drift')
    need(b['exposure']=='EXPOSED_EXPLORATORY' and b['clean_oof'] is False,'exposure drift')
    for name,asset in b['files'].items(): checked(bundle_path.parent,name,asset)
    if retained_root is not None:
        for name,asset in b['retained'].items(): checked(retained_root,name,asset)
    return b

def load_fold(bundle_path,expected,retained_root,fold):
    from tools import vnext_r7_pl_v1 as p
    from tools import pl_production_targets_v1 as targets
    from tools import vnext_r7_pl_train_v110 as base
    b=verify(bundle_path,expected,retained_root)
    need(fold in (3,4),'remote folds must be 3 or 4')
    e=b['members'][fold]; root=Path(bundle_path).parent
    spec=read(root/e['spec']); t=read(root/e['target'])
    source_projection(spec,root)
    need(spec['spec_sha256']==e['spec_sha256']==p.digest({k:v for k,v in spec.items() if k!='spec_sha256'}),'spec identity drift')
    need(t['spec_sha256']==p.digest({k:v for k,v in t.items() if k!='spec_sha256'}),'target identity drift')
    need(sha(root/e['target'])==spec['target_bundle']['sha256'],'target spec pin')
    need(spec['member']==e['member'] and t['fold']==fold and t['seed']==42 and t['arm']==b['arm'],'member/target drift')
    need(t['engineering_only'] is False and t['teacher_sha256']==b['teacher']['sha256'],'teacher/engineering drift')
    need(t['registration_sha256']==spec['registration_sha256'],'registration drift')
    need(spec['recipe']==base.mil_recipe(len(t['train_ids'])),'recipe drift')
    need(sha(root/e['payload'])==t['target_sha256'],'materialized payload drift')
    np=targets.pl.np
    with np.load(root/e['payload'],allow_pickle=False) as d:
        ids,values,mask,weights=d['ids'].tolist(),d['targets'],d['mask'],d['weights']
    need(ids==t['train_ids'] and len(ids)==len(set(ids)) and not set(ids)&set(t['val_ids']),'target roster drift')
    need(targets.a3.ids_hash(ids)==t['train_ids_sha256'] and targets.a3.ids_hash(t['val_ids'])==t['val_ids_sha256'],'roster hash drift')
    need(t['scale_fit_ids']==[u for u in ids if u not in t['l2_only_ids']]
        and targets.a3.ids_hash(t['scale_fit_ids'])==t['scale_fit_ids_sha256']
        and set(t['l2_only_ids'])<=set(targets.exclusions.REGISTERED_UIDS),'exclusion drift')
    need(values.dtype==np.float32 and values.shape==(len(ids),12) and np.isfinite(values).all()
        and ((values>=0)&(values<=1)).all(),'invalid target tensor')
    need(mask.dtype==bool and mask.shape==values.shape and targets.a3.array_hash(mask)==t['silence_mask_sha256']
        and weights.dtype==np.float32 and weights.shape==(12,) and weights.tolist()==t['weights']==t['pos_weight']
        and targets.a3.array_hash(weights)==t['l2_pos_weight_sha256'],'mask/weights drift')
    return b,e,spec,t,dict(zip(ids,values))

def source_projection(spec,root):
    for a in spec['source_pins']:
        parts=a['path'].replace('\\','/').split('/')
        starts=[i for i,x in enumerate(parts) if x in ('tools','training','legacy')]
        need(bool(starts),'unmapped source pin')
        name='/'.join(parts[starts[0]:])
        checked(root,name,dict(sha256=a['sha256'],bytes=a['bytes']))

@contextmanager
def go_mapping(bundle_path,b):
    # Map precisely one already authenticated GO asset; all other pin checks stay intact.
    from tools import vnext_r7_pl_v1 as p
    old=p.pin
    def mapped(asset):
        if asset==b['go']:
            path=Path(bundle_path).parent/'metadata/go.json'
            need(sha(path)==asset['sha256'],'GO copy drift'); return path
        return old(asset)
    p.pin=mapped
    try: yield
    finally: p.pin=old

def fit(bundle_path,expected,retained_root,fold,output,device='cpu',smoke=True,workers=None):
    from tools import pl_production_train_v1 as train
    from tools import vnext_r7_pl_go_v111 as go
    need(smoke or device=='cuda','full fit requires CUDA')
    b,e,original,target,labels=load_fold(bundle_path,expected,retained_root,fold)
    spec=copy.deepcopy(original); spec['corpus']=str(Path(retained_root)/'corpus-hardlinks')
    for key in ('weights','initial_state'):
        spec['recipe']['model'][key]=str(Path(retained_root)/safe(original['recipe']['model'][key]))
    device_name='CPU'
    if device=='cuda':
        import torch
        device_name=torch.cuda.get_device_name(0)
        need('RTX 3090' in device_name and torch.cuda.device_count()==1,'single RTX 3090 required')
    with go_mapping(bundle_path,b):
        go.admission(e['admission'],b['arm'],original['spec_sha256'])
        # Only runtime filesystem locators change; export retains original spec hash and pins.
        result=train.mil_fit(spec,target,labels,output,device=device,smoke=smoke,
                             go=None if smoke else e['admission'],workers=workers,loader_profile='vast')
    write(Path(output)/'device.json',dict(schema=SCHEMA,device_name=device_name,fold=fold,
        bundle_sha256=expected,spec_sha256=original['spec_sha256'],checkpoint_sha256=result['checkpoint_sha256'],
        receipt_sha256=sha(Path(output)/'training-receipt.json'),engineering_only=smoke,
        runtime_paths=dict(corpus=spec['corpus'],model=spec['recipe']['model'])))
    return result

def pack_results(bundle_path,expected,output):
    b=verify(bundle_path,expected); root=Path(bundle_path).parent; files={}
    for fold in (3,4):
        if not (root/f'fits/fold{fold}/training-receipt.json').exists(): continue
        rec=read(root/f'fits/fold{fold}/training-receipt.json')
        need(rec.get('status')=='COMPLETE' and rec.get('engineering_only') is False,'incomplete remote fold cannot be packed')
        for name in ('model_best.pt','training-receipt.json','device.json'):
            rel=f'fits/fold{fold}/{name}'; files[rel]=pin(root/rel)
    need(files,'no completed remote folds'); result=dict(schema=SCHEMA,bundle_sha256=expected,files=files)
    write(root/'results.json',result)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_STORED) as z:
        for name in [*files,'results.json']: z.write(root/name,name)
    return dict(status='PACKED',sha256=sha(output),bytes=Path(output).stat().st_size)

def verify_results(bundle_path,expected,results):
    b=verify(bundle_path,expected); results=Path(results); r=read(results/'results.json')
    need(r['schema']==SCHEMA and r['bundle_sha256']==expected,'result bundle identity')
    folds=[f for f in (3,4) if f'fits/fold{f}/training-receipt.json' in r['files']]
    need(folds,'empty remote results')
    wanted={f'fits/fold{f}/{n}' for f in folds for n in ('model_best.pt','training-receipt.json','device.json')}
    need(set(r['files'])==wanted,'result roster')
    for name,a in r['files'].items(): checked(results,name,a)
    for fold in folds:
        directory=results/f'fits/fold{fold}'; rec=read(directory/'training-receipt.json'); dev=read(directory/'device.json')
        need(rec['status']=='COMPLETE' and rec['engineering_only'] is False and rec['fold']==fold
             and rec['seed']==42 and rec['spec_sha256']==b['members'][fold]['spec_sha256']
             and rec['teacher_sha256']==b['teacher']['sha256'],'result scientific identity')
        need(dev['schema']==SCHEMA and dev['fold']==fold and dev['bundle_sha256']==expected
             and dev['engineering_only'] is False and 'RTX 3090' in dev['device_name']
             and dev['spec_sha256']==rec['spec_sha256'] and dev['receipt_sha256']==sha(directory/'training-receipt.json')
             and dev['checkpoint_sha256']==rec['checkpoint_sha256']==sha(directory/'model_best.pt'),'GPU/checkpoint receipt drift')
    return dict(status='VERIFIED',folds=folds,bundle_sha256=expected)

def bind(plan_path,bundle_path,expected,results,output):
    from tools import pl_production_launch_v1 as launch
    b=verify(bundle_path,expected); plan=read(plan_path); split(plan)
    need(launch.asset(plan_path)==b['original_plan'],'original plan changed')
    remote_folds=verify_results(bundle_path,expected,results)['folds'] if results is not None else []
    for key in ('go','memory','review','teacher'): launch.p.pin(plan[key])
    launch.review_gate(plan['review']['path']);launch.selected(plan['go']['path'])
    entries=[]
    for entry in plan['members']:
        e=copy.deepcopy(entry); fold=e['member']['fold']; remote=fold in remote_folds
        if remote: e['output']=str(Path(results)/f'fits/fold{fold}')
        need(launch.completed(e,plan),'incomplete fold')
        directory=Path(e['output']); rec=read(directory/'training-receipt.json')
        gpu=read(directory/'device.json')['device_name'] if remote else 'NVIDIA GeForce RTX 4070 Ti SUPER'
        # Local provenance derives from pinned device-specific memory evidence used by original launcher.
        if not remote:
            memory=read(plan['memory']['path']); need('4070 Ti SUPER' in memory['device_name'],'local GPU evidence')
            gpu=memory['device_name']
        item=dict(e['member'],**launch.asset(directory/'model_best.pt'),training_receipt=launch.asset(directory/'training-receipt.json'),training_gpu=gpu)
        if remote: item['device_receipt']=launch.asset(directory/'device.json')
        entries.append(item)
    stage=Path(plan['stage']); reg=launch.p.registration(stage/'student'/plan['arm'])
    binding=launch.p.read(stage/'student'/plan['arm']/'bindings-template.json')
    binding.update(registration_sha256=launch.p.digest(reg),arm=plan['arm'],go=plan['go'],teacher=plan['teacher'],members=entries)
    inputs=Path(output).with_name('split-binding-inputs.json'); write(inputs,binding)
    result=launch.release.bind_completed(stage,inputs,Path(output))
    write(Path(output).with_suffix('.gpu-provenance.json'),dict(schema=SCHEMA,binding=launch.asset(output),
          folds=[dict(fold=e['fold'],gpu=e['training_gpu'],checkpoint_sha256=e['sha256']) for e in entries],serving_parity='T4 qualification required'))
    return result

def memory_gate(path,expected):
    m=read(path)
    need(m.get('schema')==SCHEMA and m.get('bundle_sha256')==expected and m.get('status')=='PASS'
         and 'RTX 3090' in m.get('device_name',''),'3090 measured memory evidence required')
    need(m.get('shape')==dict(batch=8,k_train=12,k_eval=36,resolution=224,tta=True),'full shape required')
    need(set(m['peak_reserved_bytes'])=={'fit','tta'} and all(math.isfinite(v) and 0<v<=10*1024**3 for v in m['peak_reserved_bytes'].values()),'memory peak exceeds 10 GiB')
    need(math.isfinite(m['healthy_rates']['train_step']) and m['healthy_rates']['train_step']>0,'measured committed rate required')
    return m

def measure(bundle_path,expected,retained_root,output):
    """B-only full-shape optimizer/TTA probe; no checkpoint or scientific output."""
    from tools import pl_production_train_v1 as train
    from training.arms.hb146_pl_v1 import Arm
    from training.arms import mil_screen_hooks_v1 as h
    b,e,spec,target,labels=load_fold(bundle_path,expected,retained_root,3)
    need(h.torch.cuda.is_available() and h.torch.cuda.device_count()==1 and 'RTX 3090' in h.torch.cuda.get_device_name(0),'3090 required')
    h.configure_runtime('fp16','cuda',policy='strict');h.seed_all(42)
    model_cfg=copy.deepcopy(spec['recipe']['model'])
    for key in ('weights','initial_state'):model_cfg[key]=str(Path(retained_root)/safe(model_cfg[key]))
    model=h.build_model(model_cfg,ROOT).to('cuda'); arm=Arm(labels)
    corpus=Path(retained_root)/'corpus-hardlinks'; ids=target['train_ids'][:24]
    zeros={u:h.np.zeros(12,h.np.float32) for u in ids}
    ds=arm.dataset(h.MILWindows(corpus,ids,zeros,12,True,42),train=True)
    loader=h.DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,collate_fn=arm.collate_fn)
    opt,sched,scaler=h.optimizer_stack(model,spec['recipe']['updates']['steps'],'cuda','fp16')
    counters=train.Counters(updates=0,skips=0); h.torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    h.train_epoch_with(arm,model,loader,opt,sched,scaler,h.torch.tensor(target['weights']),'cuda','fp16',0,counters)
    h.torch.cuda.synchronize();rate=counters['updates']/(time.monotonic()-start)
    fit_peak=h.torch.cuda.max_memory_reserved(); h.torch.cuda.reset_peak_memory_stats()
    evalds=h.MILWindows(corpus,ids[:4],zeros,36,False,42)
    ev=h.DataLoader(evalds,batch_size=4,num_workers=0,collate_fn=h.collate)
    model.eval()
    with h.torch.inference_mode():
        for x,_,uids in ev:
            with h.autocast('cuda','fp16'):
                a=model(x.to('cuda')).sigmoid(); z=h.flip_inputs(x,uids,h.common.Corpus(corpus),36)
                pred=(a+model(z.to('cuda')).sigmoid())*.5
            need(bool(h.torch.isfinite(pred).all()),'nonfinite K36 flip smoke')
    h.torch.cuda.synchronize();tta_peak=h.torch.cuda.max_memory_reserved()
    m=dict(schema=SCHEMA,status='PASS',bundle_sha256=expected,device_name=h.torch.cuda.get_device_name(0),
           shape=dict(batch=8,k_train=12,k_eval=36,resolution=224,tta=True),
           peak_reserved_bytes=dict(fit=fit_peak,tta=tta_peak),healthy_rates=dict(train_step=rate),
           committed_updates=counters['updates'],engineering_only=True)
    need(max(fit_peak,tta_peak)<=10*1024**3,'full-shape peak exceeds 10 GiB')
    write(output,m);return m

def review_gate(path,bundle_path,expected):
    b=verify(bundle_path,expected)
    path=Path(path or Path(bundle_path).parent/'metadata/split-review.json')
    need(path.resolve()==(Path(bundle_path).parent/'metadata/split-review.json').resolve(),'review must be bundled')
    r=read(path)
    need(r.get('status')=='PASS' and r.get('independent') is True and r.get('reviewer'),'independent split review required')
    need(set(r.get('sources',{}))>=set(REVIEW_SOURCES),'split source review absent')
    for name,digest in r['sources'].items():need(b['files'][safe(name)]['sha256']==digest,'reviewed source drift')
    need(r.get('cpu_suite') and r.get('real_cpu_smoke'),'split review must bind CPU evidence')
    for k in ('cpu_suite','real_cpu_smoke'):
        a=r[k];checked(Path(bundle_path).parent,a['path'],dict(sha256=a['sha256'],bytes=a['bytes']))
        evidence=read(Path(bundle_path).parent/a['path'])
        need(evidence.get('status')=='PASS','passing CPU evidence required')
        need(all(evidence.get('sources',{}).get(name)==r['sources'][name] for name in REVIEW_SOURCES),
             'current-source CPU evidence required')
    return r

def record_review(reviewer,suite,smoke,output):
    need(bool(reviewer),'reviewer required')
    sources={name:sha(ROOT/name) for name in REVIEW_SOURCES}
    for path,kind in ((suite,'suite'),(smoke,'real_cpu_smoke')):
        need(path is not None and Path(path).is_file(),'CPU evidence absent')
        e=read(path)
        need(e.get('status')=='PASS' and e.get('kind')==kind and
             all(e.get('sources',{}).get(k)==v for k,v in sources.items()),'passing current-source CPU evidence required')
    r=dict(status='PASS',independent=True,reviewer=reviewer,sources=sources,
           cpu_suite=dict(path=str(Path(suite).resolve()),**pin(suite)),
           real_cpu_smoke=dict(path=str(Path(smoke).resolve()),**pin(smoke)))
    write(output,r);return dict(status='REVIEW_RECORDED')

def supervised(args):
    from tools.hb150_watchdog_v1 import supervise
    from tools.job_runtime import file_lock
    need(args.device=='cuda','supervised remote job requires cuda')
    need(args.deadline is not None and 0<args.deadline-time.time()<=14400,'original four-hour deadline required')
    verify(args.bundle,args.expected)
    review_gate(args.review,args.bundle,args.expected)
    rates=None
    if not args.smoke and args.command!='measure':rates=memory_gate(args.memory_evidence,args.expected)['healthy_rates']
    argv=[sys.executable,'-B','-m','tools.pl_vast_split_v110',args.command,'--bundle',str(args.bundle),
          '--expected',args.expected,'--retained-root',str(args.retained_root),'--output',str(args.output),'--device','cuda','--child']
    if args.fold is not None:argv+=['--fold',str(args.fold)]
    if args.workers is not None:argv+=['--workers',str(args.workers)]
    if args.smoke:argv+=['--smoke']
    attempt=Path(args.output).with_name(Path(args.output).name+'-watchdog')
    with file_lock(Path(args.bundle).parent/'remote-gpu.lock',blocking=False):
        code=supervise(argv,attempt,total=min(args.deadline-time.time(),900 if args.smoke or args.command=='measure' else 7200),healthy_rates=rates)
    need(code==0,'remote watchdog/child failed: '+str(code))
    return dict(status='PASS',attempt=str(attempt))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','verify','fit','measure','pack-results','verify-results','bind','split','record-review'])
    for name in ('plan','output','bundle','retained-root','results'):ap.add_argument('--'+name,type=Path)
    ap.add_argument('--expected');ap.add_argument('--fold',type=int);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--deadline',type=float);ap.add_argument('--memory-evidence',type=Path);ap.add_argument('--review',type=Path);ap.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--suite',type=Path);ap.add_argument('--smoke-evidence',type=Path);ap.add_argument('--reviewer')
    ap.add_argument('--workers',type=int,help='production fit workers; default min(6,max(0,vCPUs-2)); 0 disables multiprocessing')
    a=ap.parse_args()
    if a.command=='record-review':r=record_review(a.reviewer,a.suite,a.smoke_evidence,a.output)
    elif a.command=='prepare': r=prepare(a.plan,a.output,a.review)
    elif a.command=='split': r=split(read(a.plan))
    elif a.command=='verify':
        b=verify(a.bundle,a.expected,a.retained_root);r=dict(status='VERIFIED',files=len(b['files']),retained=len(b['retained']))
    elif a.command in ('fit','measure'):
        need(a.command!='measure' or a.device=='cuda','measure requires explicit CUDA and watchdog')
        if not a.child and a.device=='cuda':r=supervised(a)
        else:
            if a.child:
                import os
                need(os.environ.get('HB147_TOKEN') and os.environ.get('HB147_HEARTBEAT'),'owned watchdog child required')
            r=measure(a.bundle,a.expected,a.retained_root,a.output) if a.command=='measure' else fit(a.bundle,a.expected,a.retained_root,a.fold,a.output,a.device,a.smoke,workers=a.workers)
    elif a.command=='pack-results':r=pack_results(a.bundle,a.expected,a.output)
    elif a.command=='verify-results':r=verify_results(a.bundle,a.expected,a.results)
    else:r=bind(a.plan,a.bundle,a.expected,a.results,a.output)
    print(json.dumps(r,sort_keys=True))

if __name__=='__main__':main()
