"""H-B-147 L6/L7 completion v1.0.0 (2026-10-15).

Changelog: authenticated Q7S replacement replay and one supervised panel job.
No training/download/submission. CPU commands never initialize CUDA.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'training/src'))
from tools import lens_completion_1015_v1 as L
from tools import hb145_output_io as io
need, sha, rel = L.need, L.sha, L.rel
read = io.read_json
VERSION = '1.0.0'
WORK = ROOT/'.local/scratch/hb147-lenses/H-B-147'
BASE = ROOT/'.local/scratch/astra-b3/H-B-147'
TAG = 'LENSES_HB147'
PREDECESSOR = 'FORENSICS_LENSES_1015'
PYTHON = L.PYTHON
MEMBERS = [(f,r,s) for f in (0,1) for r in (224,320) for s in (42,43)]
SOURCES = ('tools/hb147_lenses_v1.py','tools/hb147_oai_native448_v1.py',
    'training/arms/native_resolution_v2.py','training/arms/mil_screen_hooks_v3.py',
    'training/arms/mil_screen_hooks_v1.py','tools/lens_completion_1015_v1.py',
    'tools/hb105_aligned_oof_v1.py','tools/multi_lens_readout_l4_v2.py',
    'tools/multi_lens_readout.py','tools/hb147_confirm_build_v1.py',
    'tools/hb147_confirm_readout_v1.py','tools/hb145_output_io.py',
    'tools/hb149_monitor_v125.py','tools/c85r7_flipall_timing.py',
    'tools/gpu_queue.py','tools/gpu_job.py','tools/job_runtime.py',
    'training/arms/mil.py','training/arms/mil_small.py','training/arms/mil_screen_v1.py',
    'training/src/raptor_common.py','training/src/raptor_corpus.py','training/src/raptor_runtime.py',
    'tools/oai_moaks_panel.py','tools/exp130_gold58_compare.py',
    'tools/extcache/common.py','tools/extcache/oai_cache_v2.py',
    'tests/test_hb147_lenses_v1.py','tests/test_hb147_oai_native448_v1.py')
POLICY = dict(primary='native320 vs unchanged HB105 Q7S replace (R50,C4,S)',
    sensitivity='native320 vs native224 matched replacement',
    ranks='one fold replaced at a time; unchanged other four folds; pooled five-fold ranks',
    macro_threshold=.0005,minimum_rulers=3,two_fold_target_ruler_floor=-.003,
    exposure='EXPOSED',clean_oof=False,draws=300,seed=20261003,
    gold58='EXPOSED descriptive; prior label development/selection',
    oai='independent image panel, native448 input; no OAI training in HB147 lineage',
    k=36,tta=True,batch=4,microbatch=2,workers=2,amp='fp16 strict',
    idle_seconds=900,no_wall_timeout=True)


def key(f,r,s): return f'f{f}-native{r}-s{s}'


def member(f,r,s):
    """Reject partial/foreign outputs before reading predictions or checkpoints."""
    from tools import hb147_confirm_build_v1 as b
    from tools import hb147_confirm_readout_v1 as cr
    from training.arms import lifecycle as life
    name=f'native{r}-s{s}'
    if f == 0:
        cp=BASE/'watchdog-v2/configs'/(name+'.json'); c=read(cp)
        attempt=BASE/'watchdog-v2'/name/'attempts/fit-v1'
        terminal=read(attempt/'receipt.json')
        need(terminal.get('reason')=='completed' and terminal.get('exit_code')==0
             and terminal.get('survivors')==[], 'fold0 fit not terminal')
        d=attempt/'run/outputs'; inv=read(d/'output-inventory.json')
        for n in ('summary.json','provenance.json','model_best.pt','oof_k36_flip_tta.csv'):
            need(sha(d/n)==inv[n]['sha256'] and (d/n).stat().st_size==inv[n]['bytes'], 'fold0 inventory differs: '+n)
        summary=read(d/'summary.json'); provenance=read(d/'provenance.json')
        spec=life.ArmSpec(cp.read_bytes())
        need(c['fold']==0 and c['seed']==s and c['arm']==f'HB147_NATIVE{r}_S{s}_F0'
             and c['config']['windows']['res']==r, 'fold0 config identity')
        need(summary['status']=='FIXED_FINAL_COMPLETE' and summary['engineering_targets'] is False
             and summary['attempted_updates']==8620 and summary['skipped_updates']==0
             and summary['spec_sha256']==spec.sha256==provenance['spec_sha256'], 'fold0 partial/foreign fit')
        for p,pin in c['inputs']['rosters'].items(): need(sha(ROOT/p)==pin,'fold0 roster drift')
        roster=read(ROOT/c['config']['roots']['ordered']); ids=roster['val']; train=roster['train']
        need(len(ids)==888 and L.uid_hash(ids)==provenance['eval_ids_ordered_sha256'], 'fold0 eval identity')
        need(L.uid_hash(train)==provenance['train_ids_ordered_sha256'] and
             provenance['checkpoint_selection']=='fixed-final' and provenance['resolution']==r and
             provenance['native_source']=='hb147_raptor64_448_c140_s02_v1','fold0 serving ancestry')
        model=c['config']['model']; corpus=c['config']['roots']['corpus']
        checkpoint=d/'model_best.pt'; prediction=d/'oof_k36_flip_tta.csv'
        evidence=[cp,attempt/'receipt.json',d/'output-inventory.json',d/'summary.json',d/'provenance.json']
    else:
        cp=b.config_path(r,s); c=b.require_fit(cp,b.WORK/'review.json'); d=b.WORK/'fits'/name
        need(read(d/'heartbeat.json').get('status')=='complete','fold1 incomplete; outputs not opened')
        checkpoint=d/'final.pt'; prediction=d/'val.csv'
        cr.validate_records(c,read(d/'summary.json'),read(d/'val.receipt.json'),config_hash=sha(cp),
                            csv_hash=sha(prediction),checkpoint_hash=sha(checkpoint))
        ids=c['val']; train=c['train']; model=c['recipe']['model']; corpus=c['corpus']
        evidence=[cp,d/'heartbeat.json',d/'summary.json',d/'val.receipt.json',b.WORK/'review.json',b.WORK/'prepared.json']
    from tools.multi_lens_readout import TARGETS
    configured_targets = c['config']['targets'] if f == 0 else c['recipe']['targets']
    need(list(configured_targets) == list(TARGETS), 'checkpoint target order differs')
    canonical=read(ROOT/'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json')
    need(set(train)<=set(canonical['train']+canonical['val']) and not set(train)&set(ids),
         'training must be RSNA only and heldout disjoint')
    need(model['initial_file_sha256']=='bf08e71a6c568009ff1e78e0ee1fb4f7e501883ce84000996480aa2c9e53a348'
         and model['weights_sha256']=='2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9',
         'OAI-free ImageNet initialization lineage differs')
    return dict(key=key(f,r,s),fold=f,resolution=r,seed=s,ids=ids,targets=list(TARGETS),model=model,corpus=corpus,
        oai_ancestry=dict(eligible=True,initial='frozen ImageNet Small initialization; no OAI fit',
                         training='RSNA canonical roster only',train_sha256=L.uid_hash(train)),
        checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint),prediction=rel(prediction),
        prediction_sha256=sha(prediction),evidence={rel(p):sha(p) for p in evidence})


def members(): return {key(*t):member(*t) for t in MEMBERS}


def predictions(m):
    import numpy as np
    import pandas as pd
    need(sha(ROOT/m['prediction'])==m['prediction_sha256'],'CSV changed')
    frame=pd.read_csv(ROOT/m['prediction'])
    need(frame.StudyInstanceUID.astype(str).tolist()==m['ids'],'CSV roster/order mismatch')
    p=frame[m['targets']].to_numpy(dtype=float)
    need(p.shape==(len(m['ids']),12) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'invalid predictions')
    return p


def registration():
    gold,_=L.gold_panel(); oai,_=L.oai_panel()
    return dict(version=VERSION,policy=POLICY,members=members(),
        spec_sha256=sha(WORK/'SPEC.md'),sources={p:sha(ROOT/p) for p in SOURCES},
        panels={n:dict(ids=ids,ids_sha256=L.uid_hash(ids)) for n,ids in [('gold58',gold),('oai557',oai)]},
        native_manifest_sha256=sha(BASE/'native448/manifest.json'),
        oai_registration_sha256=sha(ROOT/'artifacts/experiments/OAI_MOAKS_PANEL_V1/registration.json'))


def registered():
    saved=read(WORK/'registration.json')
    need(saved==registration(),'registration stale; register, smoke and review again')
    return saved


def q_guard(points):
    import numpy as np
    d=np.asarray(points,dtype=float)
    need(d.shape==(2,4,12) and np.isfinite(d).all(),'two folds x four rulers x twelve finite deltas required')
    macro=d.mean(2); qualifies=(macro>=.0005).sum(1)
    floor=d.mean(0)
    return dict(fold_macro=macro.tolist(),qualifying_rulers=qualifies.tolist(),
        two_fold_target_ruler_mean=floor.tolist(),minimum_two_fold_cell=float(floor.min()),
        harm_guard_pass=bool((floor>=-.003).all()),
        fold_pass=[bool(n>=3 and (floor>=-.003).all()) for n in qualifies],
        PASS=bool((qualifies>=3).all() and (floor>=-.003).all()))


def l6():
    import numpy as np
    from tools import multi_lens_readout_l4_v2 as ml
    reg=registered()
    review_check(read(WORK/'pre-review.json'),TAG+'_L6')
    # All model/checkpoint authentication precedes the aligned reader's CPU tripwires.
    ms=reg['members']; folds={}; primary=[]; sensitivity=[]
    from tools import hb105_aligned_oof_v1 as aligned
    raw,rosters,auth=aligned.authenticate()
    original=aligned.variants(raw)['replace']
    for f in (0,1):
        ids=ms[key(f,320,42)]['ids']
        need(ids==rosters[f]['val'],'HB105 exact fold roster/order required')
        arms={str(s):predictions(ms[key(f,320,s)]) for s in (42,43)}
        controls={str(s):predictions(ms[key(f,224,s)]) for s in (42,43)}
        ctx=ml._base.make_context(ids,ROOT)
        qa,qc,qb=q_replacements(aligned,raw,rosters,f,ids,arms,controls,original)
        pd=ml._tensor(ctx.delta(qa,qb)); sd=ml._tensor(ctx.delta(qa,qc))
        primary.append(pd[0]); sensitivity.append(sd[0])
        result=L.evaluate(ctx,arms,controls,reasons={'L7':'PENDING_GPU: queued Gold58 and OAI557; not ineligible'})
        result['lenses']['L7'].update(status='PENDING_GPU',eligible=None)
        def stats(d):
            return {name:ml._base.stat(d[:,i,:].mean(1),.0005) for i,name in enumerate(ml.RULERS)}
        result['q_primary']=dict(macro=stats(pd),points=pd[0].tolist(),
            targets={name:{target:ml._base.stat(pd[:,i,j]) for j,target in enumerate(ml.TARGETS)}
                     for i,name in enumerate(ml.RULERS)})
        result['q_matched_sensitivity']=dict(macro=stats(sd),points=sd[0].tolist())
        result['ruler_bindings']=ctx.bindings
        folds[str(f)]=result
    gate=q_guard(primary); secondary=q_guard(sensitivity)
    for f in (0,1):
        folds[str(f)]['lenses']['L6']=dict(eligible=True,PASS=gate['fold_pass'][f],
            value=folds[str(f)]['q_primary']['macro'],rule=POLICY,
            common_harm_guard=gate['harm_guard_pass'],matched_sensitivity=secondary['fold_pass'][f])
        evidence_only(folds[str(f)])
    out=dict(status='RESULT_REVIEW_REQUIRED',version=VERSION,registration_sha256=sha(WORK/'registration.json'),
        pre_review_sha256=sha(WORK/'pre-review.json'),
        exposure='EXPOSED',clean_oof=False,scope='HB105 aligned surrogate; one fold substituted per replay',
        folds=folds,primary_guard=gate,matched_guard=secondary,aligned_auth=auth,
        historical_gates_unchanged=True)
    io.atomic_json(WORK/'l6.json',out)
    return dict(status=out['status'],primary={k:gate[k] for k in
        ('fold_macro','qualifying_rulers','minimum_two_fold_cell','harm_guard_pass','fold_pass','PASS')},
        matched={k:secondary[k] for k in ('fold_macro','minimum_two_fold_cell','fold_pass')})


def q_replacements(aligned,raw,rosters,fold,ids,arms,controls,original):
    import numpy as np
    need(ids==rosters[fold]['val'],'exact HB105 fold order required')
    offset=sum(len(rosters[j]['val']) for j in range(fold)); qs=[]
    for pair in (arms,controls):
        need(set(pair)=={'42','43'},'paired S42/S43 required')
        changed={j:dict(v) for j,v in raw.items()}
        for s in ('42','43'):
            p=np.asarray(pair[s],np.float32)
            need(p.shape==(len(ids),12) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),
                 'replacement predictions invalid')
            changed[fold]['S'+s]=p
        qs.append(aligned.variants(changed)['replace'][offset:offset+len(ids)])
    return *qs,original[offset:offset+len(ids)]


def evidence_only(result):
    # Lens evidence does not replace historical experiment or release decisions.
    for field in ('verdict','decision_label','release_scope'):
        result.pop(field,None)
    result['passing_lenses']=[k for k,v in result['lenses'].items()
                             if v.get('PASS') and v.get('eligible',True)]
    result['decision_use']='Independent result review required; historical gates unchanged'


def task_plan(phase):
    need(phase in ('cpu','calibration','full'),'unknown phase')
    return [dict(key=key(*m),panel=p,n=(1 if phase=='cpu' else 4 if phase=='calibration' else n))
        for m in MEMBERS for p,n in [('gold58',58),('oai557',557)]]


def task_name(t): return t['key']+'-'+t['panel']+'.json'


def worker(phase,device,folder):
    import numpy as np
    import torch
    from training.arms import mil_screen_hooks_v3 as h
    from training.arms import native_resolution_v2 as native
    from tools import hb147_oai_native448_v1 as oai
    reg=registered(); tasks=task_plan(phase); folder=Path(folder)
    need(read(folder/'plan.json')['tasks']==tasks,'worker task plan drift')
    if device!='cpu': require_controller(phase)
    torch.set_num_threads(2)
    amp='none' if device=='cpu' else 'fp16'
    runtime=h.configure_runtime(amp,device,policy='strict')
    status=L.Status(folder,sum(t['n'] for t in tasks)); h.PROGRESS=status
    oids=reg['panels']['oai557']['ids']
    progress=lambda: status.tick('oai-verified-file')
    oc=oai.prepare(oids[:1],progress=progress) if phase=='cpu' else oai.prepare(progress=progress)
    oai.verify(oc,oids[:1] if phase=='cpu' else None,progress=progress)
    for name,m in reg['members'].items():
        need(sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'],'checkpoint changed')
        model=h.build_model(m['model'],ROOT).to(memory_format=torch.channels_last).to(device)
        payload=torch.load(ROOT/m['checkpoint'],map_location='cpu',weights_only=False)
        h.strict_load(model,payload['model']); del payload; model.eval(); status.tick('model')
        for t in [t for t in tasks if t['key']==name]:
            ids=reg['panels'][t['panel']]['ids'][:t['n']]
            corpus_path=ROOT/m['corpus'] if t['panel']=='gold58' else oc
            corpus=h.common.Corpus(corpus_path)
            need(corpus.contract==native.CONTRACT and corpus.spec['img']==448,'native448 input contract')
            h.verify_shards(corpus,ids); status.tick('verify')
            ds=native.NativeWindows(h.MILWindows(corpus_path,ids,{u:np.zeros(12,np.float32) for u in ids},36,False,m['seed']),m['resolution'])
            loader=h.DataLoader(ds,batch_size=1 if device=='cpu' else 4,shuffle=False,num_workers=2,
                worker_init_fn=h.worker_init,collate_fn=native.NativeArm().collate_fn)
            if device!='cpu': torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            values,ordered,_=h.evaluate_with(native.NativeArm(),model,loader,device,amp,corpus,tta=True)
            need(ordered==ids and values.shape==(len(ids),12) and np.isfinite(values).all()
                 and ((values>=0)&(values<=1)).all(),'panel output contract')
            io.journal(folder/'journal'/task_name(t),dict(identity=dict(key=name,panel=t['panel'],phase=phase),key=name,panel=t['panel'],ids=ids,
                probabilities=values.tolist(),checkpoint_sha256=m['checkpoint_sha256'],
                registration_sha256=sha(WORK/'registration.json'),corpus_manifest_sha256=sha(corpus_path/'manifest.json'),
                seconds=time.perf_counter()-start,runtime=runtime,k=36,tta=True,resolution=m['resolution'],
                batch=1 if device=='cpu' else 4,workers=2,device=device,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device!='cpu' else 0),folder)
            status.tick('journal')
        del model
        if device!='cpu': torch.cuda.empty_cache()
    need(device!='cpu' or not torch.cuda.is_initialized(),'CPU initialized CUDA')


def cpu_smoke():
    reg=registered(); folder=WORK/'cpu'
    io.atomic_json(folder/'plan.json',dict(tasks=task_plan('cpu')))
    worker('cpu','cpu',folder)
    paths=[folder/'journal'/task_name(t) for t in task_plan('cpu')]
    for t in task_plan('cpu'):
        validate_journal(read(folder/'journal'/task_name(t)),t,'cpu',reg)
    result=dict(status='PASS',registration_sha256=sha(WORK/'registration.json'),cuda_validated=False,
        studies_per_panel=1,checkpoints=8,journals={p.name:sha(p) for p in paths})
    io.atomic_json(WORK/'cpu-smoke.json',result); return result


def review_check(r,scope):
    need(r.get('verdict')=='PASS' and r.get('independent') is True and
         r.get('reviewer') not in (None,'','root','builder') and r.get('scope')==scope and
         r.get('registration_sha256')==sha(WORK/'registration.json'),'independent bound review required')
    evidence=ROOT/r.get('evidence_path','')
    need(ROOT in evidence.resolve().parents and evidence.is_file() and
         sha(evidence)==r.get('evidence_sha256'),'review evidence missing/stale')


def admit(dry=False):
    registered(); s=read(WORK/'cpu-smoke.json')
    need(s.get('status')=='PASS' and s.get('registration_sha256')==sha(WORK/'registration.json')
         and set(s['journals'])=={task_name(t) for t in task_plan('cpu')},'CPU smoke stale/incomplete')
    for n,pin in s['journals'].items(): need(sha(WORK/'cpu/journal'/n)==pin,'CPU journal changed')
    if dry and not (WORK/'review.json').exists(): return dict(status='HOLD_REVIEW_REQUIRED',cuda_started=False)
    r=read(WORK/'review.json'); review_check(r,TAG)
    need(r.get('cpu_smoke_sha256')==sha(WORK/'cpu-smoke.json'),'review smoke binding differs')
    return dict(status='READY_FOR_B',memory='PENDING_FULL_SHAPE_MEASUREMENT',cuda_started=False)


def binding(): return {n:sha(WORK/n) for n in ('registration.json','cpu-smoke.json','review.json')}


def require_gpu_job():
    import psutil
    from tools.hb158_scale_contrast_v1 import no_other_ml
    from tools.job_runtime import job_status
    ancestors={p.pid:p for p in psutil.Process().parents()}
    job=read(ROOT/'artifacts/logs'/TAG/'job.json')
    need(job.get('tag')==TAG and job.get('pid') in ancestors and
         ancestors[job['pid']].create_time()==job.get('create_time') and
         job.get('phase') in ('spawning','running'),'live LENSES_HB147 gpu_job required')
    need(job_status(ROOT/'artifacts/logs',PREDECESSOR)['state']=='SUCCESS','today 4070 predecessor not successful')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'0'),'physical GPU0 required')
    no_other_ml(set(ancestors)|{os.getpid()})


def require_controller(phase):
    import psutil
    parent=psutil.Process().parent(); ticket=json.loads(os.environ.get('HB147_LENSES_CONTROLLER','{}'))
    need(parent is not None and ticket.get('pid')==parent.pid and ticket.get('created')==parent.create_time()
         and ticket.get('phase')==phase and ticket.get('binding')==binding(),'direct worker bypass rejected')
    argv=parent.cmdline()
    need(len(argv)==3 and Path(argv[1]).resolve()==Path(__file__).resolve() and argv[2]=='run','wrong controller')
    if phase=='full': completed('calibration')


def idle_reason(now,progress,sample=None,total_mib=None):
    if now-progress>=900: return 'no committed progress for 900 seconds'
    if sample is not None and total_mib is not None and sample[1]>=95 and sample[2]<90 and sample[0]>=total_mib-3072 and now-progress>=600:
        return 'WDDM signature and no progress for 600 seconds'
    return None


def completed(phase):
    from tools.c85r7_flipall_timing import memory_admission
    folder=WORK/'gpu'/phase; r=read(folder/'controller.json')
    expected={task_name(t) for t in task_plan(phase)}
    need(r.get('exit_code')==0 and r.get('cleanup_errors')==[] and r.get('binding')==binding() and r.get('phase')==phase
         and set(r.get('journals',{}))==expected,'incomplete/stale GPU phase')
    peaks=[]
    for n,pin in r['journals'].items():
        need(sha(folder/'journal'/n)==pin,'GPU journal changed')
        row=read(folder/'journal'/n)
        t=next(t for t in task_plan(phase) if task_name(t)==n)
        validate_journal(row,t,phase,read(WORK/'registration.json'))
        peaks.append(row['peak_reserved_bytes'])
    memory_admission(r['desktop'],peaks)
    return r


def validate_journal(row,t,phase,reg):
    import numpy as np
    m=reg['members'][t['key']]; ids=reg['panels'][t['panel']]['ids'][:t['n']]
    need(row.get('identity')==dict(key=t['key'],panel=t['panel'],phase=phase) and
         row.get('key')==t['key'] and row.get('panel')==t['panel'] and row.get('ids')==ids and
         row.get('checkpoint_sha256')==m['checkpoint_sha256'] and row.get('resolution')==m['resolution'] and
         row.get('registration_sha256')==sha(WORK/'registration.json'),'journal model/roster identity')
    need(row.get('k')==36 and row.get('tta') is True and row.get('workers')==2 and
         row.get('batch')==(1 if phase=='cpu' else 4) and row.get('device')==('cpu' if phase=='cpu' else 'cuda'),
         'journal serving contract')
    runtime=row.get('runtime',{})
    need(runtime.get('policy')=='strict' and runtime.get('precision_fallback_allowed') is False,
         'journal runtime policy')
    values=np.asarray(row.get('probabilities'),float)
    need(values.shape==(len(ids),12) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),
         'journal predictions invalid')


def supervise(phase,desktop):
    import psutil
    from tools import hb149_monitor_v125 as mon
    from tools.c85r7_flipall_timing import memory_admission,live_memory_guard
    folder=WORK/'gpu'/phase; folder.mkdir(parents=True,exist_ok=True)
    need(not (folder/'controller.json').exists(),'prior GPU attempt retained; B must review before retry')
    io.atomic_json(folder/'plan.json',dict(tasks=task_plan(phase)))
    env=os.environ.copy(); env['HB147_LENSES_CONTROLLER']=json.dumps(dict(pid=os.getpid(),
        created=psutil.Process().create_time(),phase=phase,binding=binding()))
    child=job=None; tracked={}; code=1; error=None; hashes={}; cleanup=[]
    previous=-1; progress=time.monotonic(); started=progress
    try:
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_worker','--phase',phase],
                stdout=log,stderr=subprocess.STDOUT,env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
            proc=psutil.Process(child.pid); tracked[proc.pid]=proc; job=mon._WindowsJob(child)
            while child.poll() is None:
                now=time.monotonic(); status=io.read_receipt(folder/'status.json')
                if status.get('completed',-1)>previous: previous=status['completed']; progress=now
                sample=None
                try: sample=mon._gpu_sample(env)
                except (OSError,ValueError,subprocess.SubprocessError): pass
                if sample is not None: live_memory_guard(sample[0],desktop)
                reason=idle_reason(now,progress,sample,desktop['total_mib']); need(reason is None,reason)
                try:
                    for p in proc.children(recursive=True): tracked[p.pid]=p
                except psutil.NoSuchProcess: pass
                time.sleep(1)
            code=child.wait(); need(code==0,'worker failed; see retained run.log')
            paths=[folder/'journal'/task_name(t) for t in task_plan(phase)]
            memory_admission(desktop,[read(p)['peak_reserved_bytes'] for p in paths])
            hashes={p.name:sha(p) for p in paths}
    except BaseException as exc: code=1; error=f'{type(exc).__name__}: {exc}'
    finally:
        cleanup=mon._cleanup(child,tracked,job,psutil)
        io.receipt(folder/'controller.json',dict(exit_code=code,error=error,cleanup_errors=cleanup,
            binding=binding(),desktop=desktop,journals=hashes,phase=phase,idle_only=True,
            no_wall_cap=True,elapsed_seconds=time.monotonic()-started))
    need(code==0 and not cleanup,error or 'cleanup failure')


def prepare():
    from tools.gpu_queue import validate_plan
    registered()
    plan=[dict(tag=TAG,command=[PYTHON,str(Path(__file__).resolve()),'run'],after=PREDECESSOR,continue_on_failure=False)]
    validate_plan(plan); io.atomic_json(WORK/'queue.json',plan)
    return dict(status='QUEUE_VALID',jobs=1,after=PREDECESSOR,cuda_started=False)


def panels_readout():
    import numpy as np
    from tools import multi_lens_readout_l4_v2 as ml
    reg=registered(); completed('calibration'); completed('full')
    output=read(WORK/'l6.json')
    need(output['registration_sha256']==sha(WORK/'registration.json'),'L6 stale')
    for f in (0,1):
        images={}
        for panel,loader in [('gold58',L.gold_panel),('oai557',L.oai_panel)]:
            ids,y=loader(); tables={}
            for r in (224,320):
                tables[r]={}
                for s in (42,43):
                    m=reg['members'][key(f,r,s)]
                    row=read(WORK/'gpu/full/journal'/task_name(dict(key=m['key'],panel=panel)))
                    need(row['ids']==ids and row['checkpoint_sha256']==m['checkpoint_sha256'] and
                         row['registration_sha256']==sha(WORK/'registration.json'),'L7 identity mismatch')
                    tables[r][str(s)]=np.asarray(row['probabilities'])
            name='gold58' if panel=='gold58' else 'oai_moaks'
            images[name]=L.image_item(ids,tables[320],tables[224],y,panel=='oai557',
                'Gold58 EXPOSED study singletons' if panel=='gold58' else 'OAI one knee per participant; singleton bootstrap')
            images[name]['reason']=POLICY['gold58'] if panel=='gold58' else POLICY['oai']
        ms=reg['members']; ctx=ml._base.make_context(ms[key(f,320,42)]['ids'],ROOT)
        result=L.evaluate(ctx,{str(s):predictions(ms[key(f,320,s)]) for s in (42,43)},
            {str(s):predictions(ms[key(f,224,s)]) for s in (42,43)},images=images)
        output['folds'][str(f)]['lenses']['L7']=result['lenses']['L7']
        evidence_only(output['folds'][str(f)])
    output['gpu_binding']=binding(); output['status']='RESULT_REVIEW_REQUIRED'
    io.atomic_json(WORK/'seven-lens.json',output)
    return dict(status=output['status'],panels=2,checkpoints=8)


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['register','validate','l6','cpu-smoke','prepare','admit','run','_worker','readout'])
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--phase',choices=['calibration','full'])
    a=p.parse_args()
    if a.command=='register': io.atomic_json(WORK/'registration.json',registration()); result={'status':'REGISTERED'}
    elif a.command=='validate': registered(); result={'status':'VALIDATED'}
    elif a.command=='l6': result=l6()
    elif a.command=='cpu-smoke': result=cpu_smoke()
    elif a.command=='prepare': result=prepare()
    elif a.command=='admit': result=admit(a.dry_run)
    elif a.command=='readout': result=panels_readout()
    elif a.command=='_worker': worker(a.phase,'cuda',WORK/'gpu'/a.phase); result={'status':'PHASE_COMPLETE'}
    else:
        from tools.c85r7_flipall_timing import desktop_snapshot
        from tools import hb147_oai_native448_v1 as oai
        admit(); require_gpu_job(); oai.prepare(); desktop=desktop_snapshot()
        for phase in ('calibration','full'): supervise(phase,desktop); completed(phase)
        result=panels_readout()
    print(json.dumps(result))


if __name__=='__main__': main()
