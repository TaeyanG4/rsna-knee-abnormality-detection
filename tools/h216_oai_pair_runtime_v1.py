"""H216 fixed pair outer runner v1.0.0. Preparation is not admission.

Two sequential arms, original immutable seed0 each; exact977 label-free inference.
No automatic retry/resume. Missing qualified800 cache is a hard launch blocker.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.l1_rco_stage0_v1 import need,read,write,sha,digest,replace,LABELS,PYTHON

VERSION='1.0.0';TAG='H216_OAI_PAIR_V1'
OUT=ROOT/'.local/scratch/slot-b/h216-oai-pair-runtime-v1'
INPUT=ROOT/'.local/scratch/slot-b/h216-oai-pair-inputs-v1'
K1=ROOT/'.local/scratch/l1-rco/k1v3-grouping'
RSNA_CACHE=Path('G:/temp_data/rsna-knee/l1rco-cache-full')
OAI_CACHE=Path('G:/temp_data/rsna-h216-oai-v4-v1')
ARMS=('rsna_control','oai_native')


def source_paths():
    names=['h216_oai_pair_runtime_v1.py','h216_oai_pair_core_v101.py','h216_oai_pair_core_v1.py',
           'h216_oai_pair_inputs_v1.py','l1_rco_k1_optim_v3.py','l1_rco_k1_state_v1.py',
           'l1_rco_runtime_v120.py','l1_rco_k1_v1.py','job_runtime.py']
    return [ROOT/'tools'/n for n in names]+[ROOT/'tools/exp189_s6_src/compact_train.py',
            ROOT/'tests/test_h216_oai_pair_runtime_v1.py',ROOT/'tests/test_h216_oai_pair_core_v101.py',
            ROOT/'docs/research/h216-oai-pair-runtime-v1.md',INPUT/'RESULT.json',INPUT/'REGISTRATION.private.json',
            ROOT/'.local/scratch/slot-b/h216-oai-pair-core-v1/PARAMETER-MANIFESTS.json']


def inputs():
    result=read(INPUT/'RESULT.json');reg=read(INPUT/'REGISTRATION.private.json')
    need(result['status']=='PRODUCTION_INPUTS_BOUND_NOT_MATERIALIZED' and result['contract_id']==reg['contract_id'],'INPUT_BINDING')
    need(reg['contract_id']==digest({k:v for k,v in reg.items() if k!='contract_id'}),'INPUT_DIGEST')
    for n,h in result['output_hashes'].items():need(sha(INPUT/n)==h,'INPUT_OUTPUT_DRIFT')
    for p,h in reg['sources'].items():need(sha(p)==h,'INPUT_SOURCE_DRIFT')
    return reg


def cache_receipt_check(receipt,index,binding):
    """Producer commits actual full hashes/parity first; no hypothetical receipt."""
    required=binding['schedules']['oai_required_ids'];wanted=set(required)
    need(receipt['status']=='QUALIFIED_ACTUAL_V4_CACHE' and receipt['input_contract']==binding['contract_id'],'OAI_CACHE_INCOMPLETE')
    need(receipt['ids']==required and len(required)==len(wanted)==800,'OAI_CACHE_ROSTER')
    need(index['order']==required and set(index['blocks'])==set(index['done'])==wanted and not index['errors'],'OAI_CACHE_INDEX')
    need(receipt['actual_anchor'] is True and receipt['loader_parity']=='PASS' and receipt['views']=='v4' and
         receipt['size']==384 and receipt['geometry_errors']==0 and receipt['model_errors']==0 and
         receipt['out_of_field_count']==0,'OAI_CACHE_QUALIFICATION')
    need(receipt['anchor_model_studies']+receipt['explicit_slab_prior_studies']==800,'ANCHOR_ACCOUNTING')
    need(set(receipt['hashes'])==set(receipt['files'])=={'index.json','px.u8.npy','meta.f32.npy'},'OAI_CACHE_FILES')
    need(all(len(h)==64 for h in receipt['hashes'].values()),'CACHE_HASH_FORMAT')
    cursor=0
    for uid in required:
        offset,count,nblocks=index['done'][uid];blocks=index['blocks'][uid]
        need(offset==cursor and count>0 and nblocks==len(blocks)>0,'CACHE_PARTITION')
        need(all(offset<=start and start+b['n_slices']<=offset+count for b in blocks for _,start in b['views']),'CACHE_VIEW_BOUNDS')
        cursor+=count
    need(cursor==index['cursor'],'CACHE_CURSOR')


def qualified_cache(binding):
    import numpy as np
    receipt=read(OAI_CACHE/'receipt.json');index=read(OAI_CACHE/'index.json')
    cache_receipt_check(receipt,index,binding)
    need(sha(OAI_CACHE/'index.json')==receipt['hashes']['index.json'],'OAI_INDEX_HASH')
    for n,stamp in receipt['files'].items():
        st=(OAI_CACHE/n).stat()
        need(st.st_size==stamp['bytes'] and st.st_mtime_ns==stamp['mtime_ns'],'OAI_COMMITTED_FILE_CHANGED')
    px=np.load(OAI_CACHE/'px.u8.npy',mmap_mode='r');meta=np.load(OAI_CACHE/'meta.f32.npy',mmap_mode='r')
    need(px.dtype==np.uint8 and px.shape==(index['cursor'],384,384) and
         meta.dtype==np.float32 and meta.shape==(len(px),12),'OAI_ARRAY_LAYOUT')
    # Reviewed original immutable RSNA4167 cache check: header/index/commit only.
    from tools.l1_rco_k1_v1 import complete_cache
    complete_cache(read(K1/'registration.json'),RSNA_CACHE)
    return dict(oai_receipt_sha256=sha(OAI_CACHE/'receipt.json'),rsna_receipt_sha256=sha(RSNA_CACHE/'receipt.json'))


def prepare():
    binding=inputs();blockers=[];cache=None
    try:cache=qualified_cache(binding)
    except (FileNotFoundError,KeyError,ValueError):blockers.append('QUALIFIED_ACTUAL800_CACHE_MISSING_OR_INVALID')
    paths=source_paths();pins={str(p):sha(p) for p in paths}
    result=dict(version=VERSION,status='BLOCKED' if blockers else 'INPUTS_READY_REVIEW_PENDING',
        blockers=blockers,input_contract=binding['contract_id'],sources=pins,cache_binding=cache,
        physical_gpu_claim=False,checkpoint_reads=0,cuda=False,training_ready=False,
        arms=list(ARMS),actual_oai_fit=False)
    OUT.mkdir(parents=True,exist_ok=True)
    need(not (OUT/'REGISTRATION.json').exists(),'REGISTERED_PREPARATION_IMMUTABLE')
    write(OUT/'PREPARATION.json',result)
    return {k:v for k,v in result.items() if k not in ('sources','cache_binding')}


def cpu_inference_smoke():
    """New outer inference path only; bounded existing cache/random model."""
    import numpy as np
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU_ISOLATION')
    need(not (OUT/'CPU-INFERENCE.json').exists(),'CPU_SMOKE_EXISTS')
    binding=inputs();sys.path.insert(0,str(ROOT/'tools/exp189_s6_src'));os.environ['HF_HUB_OFFLINE']='1'
    import compact_train as ct
    torch.set_num_threads(1)
    def forbidden(*args,**kwargs):raise ValueError('DISEASE_CHECKPOINT_READ_FORBIDDEN')
    old_load=torch.load;torch.load=forbidden
    try:
        ds=ct.Studies(str(RSNA_CACHE),binding['rsna_train'][:1],64,'eval',0,eval_stride=3,slices=3,phys_mm=3.5)
        x,meta,i=ds[0];x=x[:2];meta=meta[:2]
        class TwoWindows:
            def __len__(self):return 1
            def __getitem__(self,i):return x,meta,i
        torch.manual_seed(21600);model=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False).eval()
        events=[];values=infer_values(model,TwoWindows(),ct,'cpu',events.append)
        img,m,mask,_=ct.collate([(x,meta,0)])
        with torch.no_grad():expected=model(img[...,16:368,16:368],m,mask,chunk=32).float().sigmoid().numpy()
        need(np.array_equal(expected,values),'CPU_INFERENCE_PARITY')
        need(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
        result=dict(status='PASS_CPU_INFERENCE_MECHANICS',runtime_sha256=sha(Path(__file__)),
                    input_contract=binding['contract_id'],actual_rsna_windows=len(x),rows=1,targets=12,
                    parent_formula_max_abs=float(np.abs(expected-values).max()),random_weights=True,
                    disease_checkpoint_reads=0,cuda=False,fit_steps=0,actual_oai_roi=False,training_ready=False)
        OUT.mkdir(exist_ok=True);write(OUT/'CPU-INFERENCE.json',result);return result
    finally:torch.load=old_load


def register():
    from tools.h216_oai_pair_core_v101 import BASE_SHA,RECIPE
    need(not (OUT/'REGISTRATION.json').exists(),'ALREADY_REGISTERED')
    binding=inputs();cache=qualified_cache(binding)
    preparation=read(OUT/'PREPARATION.json')
    need(preparation['status']=='INPUTS_READY_REVIEW_PENDING' and preparation['cache_binding']==cache,'PREPARATION_NOT_READY')
    pins={str(p):sha(p) for p in source_paths()}
    need(pins==preparation['sources'],'PREPARATION_SOURCE_DRIFT')
    reg=dict(version=VERSION,input_contract=binding['contract_id'],base_sha256=BASE_SHA,recipe=RECIPE,
             sources=pins,cache_binding=cache,arms=list(ARMS),holdout_count=977,
             preparation_sha256=sha(OUT/'PREPARATION.json'),clean_validation=False)
    reg['contract_id']=digest(reg);write(OUT/'REGISTRATION.json',reg)
    return dict(status='REGISTERED_REVIEW_PENDING',contract_id=reg['contract_id'],physical_gpu_claim=False)


def admit_cpu():
    reg=read(OUT/'REGISTRATION.json');binding=inputs()
    need(reg['contract_id']==digest({k:v for k,v in reg.items() if k!='contract_id'}),'REGISTRATION_DIGEST')
    need(reg['sources']=={str(p):sha(p) for p in source_paths()},'RUNTIME_SOURCE_DRIFT')
    need(reg['cache_binding']==qualified_cache(binding) and reg['input_contract']==binding['contract_id'],'CACHE_BINDING_DRIFT')
    review=read(OUT/'B-PRECHECK.json')
    need(review['verdict']=='PASS_B_DIRECT' and review['reviewer']=='B/Codex' and
         review['authority']=='Taeyang2026-10-19-solo-continuous-GPU' and
         review['registration_sha256']==sha(OUT/'REGISTRATION.json') and
         review['report_sha256']==sha(OUT/'B-PRECHECK.md'),'BOUND_PRECHECK_REQUIRED')
    return reg,binding


def command():return [PYTHON,str(Path(__file__).resolve()),'run']


def verify_job(job,current,owner,owner_argv):
    from tools.job_runtime import command_digest
    need(job.get('schema')=='gpu-job/v2' and job.get('tag')==TAG and job.get('phase')=='running','GPU_JOB_REQUIRED')
    need(job.get('child_identity')==current and {k:job.get(k) for k in ('pid','create_time')}==owner,'GPU_JOB_IDENTITY')
    need(job.get('command')==command() and job.get('command_sha256')==command_digest(command(),ROOT) and
         Path(job['cwd']).resolve()==ROOT,'GPU_JOB_COMMAND')
    need(owner_argv[-3:]==['--_run-job',TAG,job['run_id']] and any(Path(x).name=='gpu_job.py' for x in owner_argv),'GPU_JOB_PARENT')


def require_job(worker=False):
    import psutil
    from tools.job_runtime import LOGS,file_lock,identity,read_json
    with file_lock(LOGS/'.gpu-claim.lock'):
        parent=psutil.Process().parent();need(parent is not None,'MISSING_PARENT')
        owner=parent.parent() if worker else parent;need(owner is not None,'MISSING_OWNER')
        job=read_json(LOGS/TAG/'job.json')
        verify_job(job,identity(parent.pid) if worker else identity(),identity(owner.pid),owner.cmdline())
        if worker:need(parent.cmdline()[1:]==command()[1:],'WORKER_PARENT_COMMAND')
        return owner.pid


def infer_values(model,ds,ct,device,progress):
    import numpy as np
    import torch
    model.eval();values=np.empty((len(ds),12),np.float32);seen=set()
    with torch.no_grad():
        for i in range(len(ds)):
            img,meta,mask,idx=ct.collate([ds[i]])
            need(idx.tolist()==[i] and i not in seen,'INFERENCE_ORDER')
            need(img.shape[-2:]==(384,384),'INFERENCE_CROP')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                z=model(img[...,16:368,16:368].to(device),meta.to(device),mask.to(device),chunk=32)
            need(tuple(z.shape)==(1,12) and bool(torch.isfinite(z).all()),'INFERENCE_LOGITS')
            values[i]=z.float().sigmoid().cpu().numpy()[0];seen.add(i)
            progress(dict(phase='eval',eval_batches=i+1,windows=int(mask.sum())))
    need(len(seen)==len(ds) and np.isfinite(values).all(),'INFERENCE_COMPLETE')
    return values


def save_predictions(path,ids,values):
    import numpy as np
    need(len(ids)==len(set(ids)) and values.shape==(len(ids),12) and np.isfinite(values).all(),'PREDICTION_ROSTER')
    with path.with_suffix('.tmp').open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['StudyInstanceUID']+LABELS+['status','model_fold'])
        for uid,row in zip(ids,values):writer.writerow([uid]+row.tolist()+['ok',0])
    need(not path.exists(),'PREDICTIONS_EXIST');replace(path.with_suffix('.tmp'),path)


def worker(arm):
    require_job(True);reg,binding=admit_cpu();need(arm in ARMS,'ARM')
    import numpy as np
    import torch
    from tools import h216_oai_pair_core_v101 as core
    from tools.h216_oai_pair_inputs_v1 import ScheduledStudies
    from tools.l1_rco_k1_state_v1 import atomic_torch_save
    from tools.l1_rco_runtime_v120 import memory_ok
    os.environ['HF_HUB_OFFLINE']='1';sys.path.insert(0,str(ROOT/'tools/exp189_s6_src'))
    import compact_train as ct
    torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();dev=torch.device('cuda')
    output=OUT/arm;output.mkdir(exist_ok=False)
    need(sha(K1/'fit/models/E_prior_s0_f0.pt')==reg['base_sha256'],'BASE_CHECKPOINT_DRIFT')
    base=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
    base.load_deploy_state_dict(torch.load(K1/'fit/models/E_prior_s0_f0.pt',map_location='cpu',weights_only=False))
    base=base.to(dev)
    rs=ct.Studies(str(RSNA_CACHE),binding['rsna_train'],64,'train',22601,train_sampling='cover',cover_stride=3,slices=3,slice_jitter=.25,phys_mm=3.5)
    oa=ScheduledStudies(ct,OAI_CACHE,binding['oai_logical_roster'],binding['schedules']['oai_required_ids']) if arm=='oai_native' else None
    rt=torch.from_numpy(np.load(INPUT/'rsna-targets.private.npy'));ot=torch.from_numpy(np.load(INPUT/'native-targets.private.npy')) if oa is not None else None
    admission=read(OUT/'ADMISSION.json');started=time.perf_counter();windows=0;last_phase=None;phase_windows=0
    def progress(s):
        nonlocal windows,last_phase,phase_windows
        torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()
        need(memory_ok(peak,admission['desktop_bytes'],admission['total_bytes']),'MEMORY_GATE')
        windows+=s.get('windows',0)
        if s['phase']!=last_phase:phase_windows=0;last_phase=s['phase']
        phase_windows+=s.get('windows',0)
        write(OUT/'status.json',dict(s,phase='train' if s['phase'] in ('adapt','recovery') else s['phase'],
              subphase=s['phase'],arm=arm,time=time.time(),windows=phase_windows,total_windows=windows,
              peak_reserved=peak,elapsed_seconds=time.perf_counter()-started))
    manifests=read(ROOT/'.local/scratch/slot-b/h216-oai-pair-core-v1/PARAMETER-MANIFESTS.json')
    core.v1.reset_rng(21600)
    temp=core.v1.TemporaryAdapter(base,120 if arm=='oai_native' else 12)
    core.v1.configure_trainable(temp,True);_,manifest=core.v1.parameter_groups(temp)
    need(manifest==manifests[arm],'ADAPT_PARAMETER_MANIFEST');del temp
    core.v1.configure_trainable(base,False);_,manifest=core.v1.parameter_groups(base)
    need(manifest==manifests['recovery'],'RECOVERY_PARAMETER_MANIFEST')
    def checkpoint(state):
        atomic_torch_save(state,output/'resume.pt')
        write(output/'resume.json',dict(phase=state['phase'],step=state['step'],identity=state['identity'],
              contract_id=reg['contract_id'],sha256=sha(output/'resume.pt'),time=time.time()))
    try:
        fit=core.pair_arm(base,rs,oa,rt,ot,ct,arm=arm,device=dev,progress=progress,checkpoint=checkpoint,
                         input_contract=binding['contract_id'],pause_requested=lambda:(OUT/'PAUSE-REQUEST').exists())
    except core.PausedAtCommit:
        write(output/'PAUSED.json',dict(status='PAUSED_AFTER_COMMIT',contract_id=reg['contract_id'],
              checkpoint=read(output/'resume.json'),automatic_resume=False))
        raise SystemExit(3)
    atomic_torch_save(base.deploy_state_dict(),output/'model.pt')
    write(output/'fit-complete.json',dict(fit,contract_id=reg['contract_id'],base_sha256=reg['base_sha256'],
          model_sha256=sha(output/'model.pt'),elapsed_seconds=time.perf_counter()-started))
    del rs,oa,rt,ot
    ds=ct.Studies(str(RSNA_CACHE),binding['holdout'],64,'eval',0,eval_stride=3,slices=3,phys_mm=3.5)
    values=infer_values(base,ds,ct,dev,progress);save_predictions(output/'holdout.csv',binding['holdout'],values)
    write(output/'RESULT.json',dict(status='COMPLETE',arm=arm,contract_id=reg['contract_id'],holdout_rows=977,
          finite=True,model_sha256=sha(output/'model.pt'),holdout_sha256=sha(output/'holdout.csv'),
          elapsed_seconds=time.perf_counter()-started,peak_reserved=torch.cuda.max_memory_reserved(),
          efficacy_evaluated=False,clean_validation=False,exposure='EXPOSED_EXPLORATORY'))


def run():
    owner=require_job();reg,_=admit_cpu()
    need(not (OUT/'ADMISSION.json').exists(),'ATTEMPT_ALREADY_ADMITTED')
    from tools.l1_rco_runtime_v120 import active_ml_processes,gpu_snapshot,stale_reason
    from tools.l1_rco_k1_state_v1 import ThroughputGuard
    need(not active_ml_processes(owner),'ANOTHER_GPU_PROCESS');snap=gpu_snapshot()
    need(snap['desktop_bytes']+2*2**30<snap['total_bytes'],'GPU_HEADROOM')
    need(shutil.disk_usage(OUT).free>104*2**30,'DISK_RESERVE')
    need(not (OUT/'PAUSE-REQUEST').exists(),'PAUSE_REQUEST_PRESENT')
    write(OUT/'ADMISSION.json',dict(snap,contract_id=reg['contract_id'],time=time.time()))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',PYTHONPATH=str(ROOT))
    child=None
    try:
        for arm in ARMS:
            status=dict(phase='launch',subphase='launch',arm=arm,time=time.time(),windows=0)
            write(OUT/'status.json',status);guard=ThroughputGuard(time.time());phase_key=None
            with (OUT/(arm+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'worker','--arm',arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    time.sleep(15)
                    try:status=read(OUT/'status.json')
                    except (OSError,ValueError):pass
                    need(not stale_reason(status,time.time()),'ZERO_PROGRESS_WATCHDOG')
                    key=(status.get('arm'),status.get('subphase'))
                    if key!=phase_key:guard=ThroughputGuard(time.time());phase_key=key
                    guard.observe(status,time.time())
                if child.returncode==3:
                    need(read(OUT/arm/'PAUSED.json')['status']=='PAUSED_AFTER_COMMIT','PAUSE_RECEIPT_MISSING')
                    write(OUT/'PAUSED.json',dict(status='PAUSED_AFTER_COMMIT',arm=arm,contract_id=reg['contract_id'],
                          automatic_resume=False))
                    raise SystemExit(3)
                need(child.returncode==0,'ARM_WORKER_FAILED')
            need(read(OUT/arm/'RESULT.json')['status']=='COMPLETE','ARM_INCOMPLETE')
        write(OUT/'RESULT.json',dict(status='PAIR_COMPLETE_PENDING_FIXED_EVALUATION',contract_id=reg['contract_id'],
              arms={a:sha(OUT/a/'RESULT.json') for a in ARMS},efficacy_evaluated=False))
    finally:
        if child is not None and child.poll() is None:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=False)
            else:child.terminate()
            child.wait(timeout=30)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','cpu-smoke','register','admit','run','worker']);ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    if a.action not in ('run','worker'):os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        if a.action=='prepare':result=prepare()
        elif a.action=='cpu-smoke':result=cpu_inference_smoke()
        elif a.action=='register':result=register()
        elif a.action=='admit':admit_cpu();result=dict(status='CPU_ADMISSIBLE',physical_gpu_claim=False)
        elif a.action=='worker':worker(a.arm);result=dict(status='ARM_COMPLETE')
        else:run();result=dict(status='PAIR_COMPLETE_PENDING_FIXED_EVALUATION')
        print(json.dumps(result))
    except Exception as exc:
        code=str(exc) if isinstance(exc,ValueError) and str(exc).replace('_','').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED'
        result=dict(status='FAILED_OR_BLOCKED',type=type(exc).__name__,code=code,partial_preserved=True)
        if a.action in ('run','worker') and (OUT/'ADMISSION.json').exists():
            dest=OUT/(a.action+'-'+str(a.arm)+'-FAILURE.json')
            if not dest.exists():write(dest,result)
        print(json.dumps(result));raise SystemExit(1)


if __name__=='__main__':main()
