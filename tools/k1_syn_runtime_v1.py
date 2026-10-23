"""Frozen Syn feature extraction runtime v1.0.3.

Changelog v1.0.3: local WDDM N/A memory contexts allowed; Python/ML names
and foreign positive CUDA allocations refused. Vast/utilization/VRAM unchanged.
Changelog v1.0.2: local <60 utilization, verified desktop-only compute allowance,
exact V1B SUCCESS dependency, and immutable scientific registration overlay.
Changelog v1.0.1: pre-result amendment adds reviewed environment and explicit
B ownership transfer gates for the new Vast sibling; parent stays read-only.
Changelog: CPU preparation; owner/review gates; resumable hash-bound rows;
portable read-only cache; local-built Vast package; no GPU launch on import.
"""
import contextlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from tools import k1_syn_head_v1 as h

TAG='K1_SYN_HEAD_V1'
VAST_ROOT=Path('/root/work/k1-syn-head-v102')
VAST_CACHE=Path('/root/work/h224-folds-v2/.local/scratch/astra-b2/h224-folds-v1/cache')
VAST_PREVIOUS_OWNER=Path('/root/work/h224-failure-v1/.local/scratch/astra-b2/h224-folds-v1/gpu0-owner')
VAST_PREVIOUS_TAG='K1_FAILURE_WEIGHT_F0_V1B'
VAST_PREVIOUS_RUN='0a238057f13c4d41afe9762ba33fdab3'
VAST_PREVIOUS_COMMAND_SHA='e364ad5635cd7bd6b84b6b1b7f0a1aa67ce43fea0f7947501c215f232309c496'
LOCAL_CACHE=Path('F:/rsna-knee-cache/l1rco-cache-full')


def config(host):
    return dict(host=host,cache=str(LOCAL_CACHE) if host=='local' else VAST_CACHE.as_posix(),
                device='RTX 4070 Ti SUPER' if host=='local' else 'RTX 4090',
                owner=str(h.ROOT) if host=='local' else VAST_ROOT.as_posix())


def command(host): return [sys.executable,'-B','-m','tools.k1_syn_head_v1','run','--host',host]


def cpu_environment():
    import platform
    import torch
    import timm
    import numpy
    import pydicom
    return dict(python=platform.python_version(),platform=sys.platform,torch=torch.__version__,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),timm=timm.__version__,
                numpy=numpy.__version__,pydicom=pydicom.__version__)


def predecessor_success(launch=False):
    # CPU preparation authenticates files only. Liveness queries belong exclusively
    # to the launch gate, and never probe an unrelated job or scan another root.
    import re
    from tools import job_runtime as jobs
    directory=VAST_PREVIOUS_OWNER/'artifacts/logs'/VAST_PREVIOUS_TAG
    job=h.read(directory/'job.json'); receipt=h.read(directory/'receipt.json')
    h.need(job.get('schema')=='gpu-job/v2' and job.get('tag')==VAST_PREVIOUS_TAG and
           job.get('run_id')==VAST_PREVIOUS_RUN and job.get('cwd')==VAST_PREVIOUS_OWNER.as_posix() and
           job.get('command_sha256')==VAST_PREVIOUS_COMMAND_SHA==jobs.command_digest(job['command'],job['cwd']),
           'exact V1B predecessor identity')
    h.need(not (directory/'FAILED').exists() and
           re.fullmatch(r'exit 0 at \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', (directory/'SUCCESS').read_text().strip()) and
           receipt.get('schema')=='gpu-job-receipt/v1' and receipt.get('state')=='SUCCESS' and
           receipt.get('exit_code')==0 and receipt.get('tag')==VAST_PREVIOUS_TAG and
           receipt.get('run_id')==VAST_PREVIOUS_RUN and receipt.get('command_sha256')==VAST_PREVIOUS_COMMAND_SHA and
           receipt.get('wrapper_identity')=={'pid':job['pid'],'create_time':job['create_time']} and
           job.get('child_identity') and receipt.get('child_identity')==job['child_identity'],
           'authenticated V1B SUCCESS required')
    if launch:
        state=jobs.job_status(directory.parent,VAST_PREVIOUS_TAG,tail=False)
        h.need(state['state']=='SUCCESS' and state.get('receipt_kind')=='bound', 'V1B live/unconfirmed predecessor')
    return dict(run_id=VAST_PREVIOUS_RUN,receipt_sha256=h.sha(directory/'receipt.json'))


def vast_transfer(launch=False):
    # A new directory is not a shared lock. B reserves all launch routes until
    # terminal. Only status reads are performed in the existing owner tree.
    transfer=h.read(h.OUT/'OWNER-TRANSFER.json')
    h.need(transfer.get('owner')=='B' and transfer.get('exclusive_until_terminal') is True and
           transfer.get('registration_sha256')==h.sha(h.OUT/'PREREGISTRATION.json') and
           transfer.get('destination')==VAST_ROOT.as_posix() and
           transfer.get('previous_owner')==VAST_PREVIOUS_OWNER.as_posix() and
           0 < transfer.get('admit_before_epoch',0)-time.time() <= 86400, 'B exclusive owner transfer required')
    predecessor_success(launch=launch)
    return transfer


def import_trainer():
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    if os.name=='nt':
        os.environ['S6PLUS_PIN_DLL']=r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
        from tools import s6plus
    spec=importlib.util.spec_from_file_location('syn_frozen_trainer',h.ROOT/h.TRAINER)
    ct=importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
    return ct


@contextlib.contextmanager
def precision():
    import torch
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    try: yield
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark=old


def model(ct,device):
    import torch
    # init=None is essential: the complete deploy checkpoint supplies every tensor.
    net=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
    net.load_deploy_state_dict(torch.load(h.ROOT/h.CHECKPOINT,map_location='cpu',weights_only=True))
    return net.to(device).eval().requires_grad_(False)


def dataset(ct,cache,ids,split):
    return ct.Studies(str(cache),ids,64,'train' if split=='train' else 'eval',0,
        eval_stride=2,train_sampling='cover',cover_stride=3,slices=3,phys_mm=3.5,slice_jitter=.25 if split=='train' else 0.)


def forward(net,ct,item,device):
    import numpy as np
    import torch
    captured=[]
    hook=net.out_norm.register_forward_hook(lambda module,args,out:captured.append(out[:,h.SYN,:].detach().float().cpu()))
    try:
        img,meta,mask,_=ct.collate([item])
        with torch.inference_mode(),precision(),torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):
            z=net(img[...,16:368,16:368].to(device),meta.to(device),mask.to(device),chunk=32)
        h.need(len(captured)==1 and z.shape==(1,12),'trained head interface')
        f=captured[0].numpy()[0]; pred=z.float().sigmoid().cpu().numpy()[0]
        initial=np.r_[net.cls_w[h.SYN].detach().float().cpu().numpy(),net.bias[h.SYN].detach().float().cpu().numpy()]
        error=abs(float(f@initial[:384]+initial[384])-float(z[0,h.SYN]))
        h.need(error<=1e-4 and np.isfinite(f).all() and np.isfinite(pred).all(),'frozen head reconstruction')
        return f,pred,initial,error
    finally: hook.remove()


def cache_proof(cache,reg):
    import numpy as np
    cache=Path(cache); parent=h.read(h.ROOT/h.PARENT/'registration.json')
    receipt=h.read(cache/'receipt.json'); index=h.read(cache/'index.json'); contract=h.read(cache/'contract.json'); status=h.read(cache/'status.json')
    ids=reg['train']+reg['holdout']
    h.need(status['phase']=='complete' and status['done']==status['total']==4167,'committed full cache required')
    h.need(receipt['contract_id']==index['contract_id']==contract['contract_id']==parent['fullcache_contract_id'],'cache contract')
    h.need(receipt['studies']==len(set(receipt['ids']))==4167 and set(receipt['ids'])==set(ids),'cache roster')
    h.need(index['order']==receipt['ids']==contract['ids'] and not index['errors'] and set(index['blocks'])==set(ids)==set(index['done']),'cache manifest')
    h.need(h.sha(cache/'index.json')==receipt['hashes']['index.json'] and contract['size']==384 and contract['views']=='v4','cache index/recipe')
    px=np.load(cache/'px.u8.npy',mmap_mode='r'); meta=np.load(cache/'meta.f32.npy',mmap_mode='r')
    h.need(px.dtype==np.uint8 and px.shape==(index['cursor'],384,384) and meta.dtype==np.float32 and meta.shape==(index['cursor'],12),'cache layout')
    h.need(sum((cache/n).stat().st_size for n in receipt['hashes'])==receipt['bytes'],'committed cache sizes')
    return dict(contract_id=receipt['contract_id'],receipt_sha256=h.sha(cache/'receipt.json'),
                index_sha256=h.sha(cache/'index.json'),meta_sha256=h.sha(cache/'meta.f32.npy'),
                payload_sha256=receipt['hashes'],studies=4167,
                verification='reviewed immutable full-payload commit; current index, metadata, layout and sizes')


def prepare(host):
    h.need(host=='local' or h.ROOT==VAST_ROOT,'Vast preparation requires new sibling deployment')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    reg=h.validate(host=='vast'); ct=import_trainer(); import torch; import numpy as np
    torch.set_num_threads(4); proof=cache_proof(config(host)['cache'],reg)
    # Full shape loading and deterministic cap64 sampling are checked on real data;
    # CPU encoder replay is two windows per study, not a claimed GPU/fullbag benchmark.
    cpu=h.ROOT/'.local/scratch/l1-rco/stage0/cpu-cache-reviewed' if host=='local' else Path(config(host)['cache'])
    available=set(h.read(cpu/'index.json')['blocks']); used={s:[u for u in reg[s] if u in available][:2 if s=='train' else 1] for s in ('train','holdout')}
    h.need(len(used['train'])==2 and len(used['holdout'])==1,'real smoke train2/holdout1')
    net=model(ct,'cpu'); arrays={}; shapes={}; worst=0.
    with precision():
        for split in used:
            ds=dataset(ct,cpu,used[split],split); features=[]; preds=[]; shapes[split]=[]
            for i in range(len(ds)):
                img,meta,idx=ds[i]; shapes[split].append(len(img))
                h.need(idx==i and img.dtype==torch.uint8 and img.shape[-2:]==(384,384),'real loader layout')
                if split=='train': h.need(len(img)<=64 and torch.equal(img,ds[i][0]),'deterministic capped train bag')
                f,p,initial,err=forward(net,ct,(img[:2],meta[:2],idx),'cpu'); worst=max(worst,err); features.append(f); preds.append(p)
            arrays[split]=(np.array(features),np.array(preds))
    # Three bins are synthetic in this diagnostic only; the encoder features are
    # real train-only rows. Production priors and all3190 are never fitted here.
    x=np.tile(arrays['train'][0],(3,1)); fitted=h.fit_head(x,np.tile([.2,.8],3),np.repeat([.1,.5,.9],2),initial)
    candidate=h.replace_syn(arrays['holdout'][1],arrays['holdout'][0],fitted['theta'])
    h.need(np.array_equal(candidate[:,np.arange(12)!=h.SYN],arrays['holdout'][1][:,np.arange(12)!=h.SYN]),'Syn-only replacement')
    diagnostic=h.OUT/'cpu-smoke/roundtrip.npz'
    h.npz(diagnostic,features=arrays['holdout'][0],predictions=candidate)
    with np.load(diagnostic,allow_pickle=False) as data: h.need(np.array_equal(data['predictions'],candidate),'atomic feature roundtrip')
    y,eff=h.training_targets(reg); _,_,bins=h.soft_weights(y,eff)
    h.need(not torch.cuda.is_initialized(),'CPU only')
    result=dict(status='PASS',registration_sha256=h.sha(h.OUT/'PREREGISTRATION.json'),cache=proof,
        host=host,environment=cpu_environment(),real_train_studies=2,real_holdout_studies=1,loaded_windows=shapes,forward_windows_per_study=2,
        max_logit_reconstruction_error=worst,other_11_bit_identical=True,diagnostic_optimizer_converged=fitted['converged'],
        train_strata=bins,cuda_initialized=False,scientific_fit_performed=False)
    h.write(h.OUT/('CPU-SMOKE.json' if host=='local' else 'VAST-CPU-SMOKE.json'),result)
    return result


def review_bindings():
    paths=['PREREGISTRATION.json','OPERATIONAL-PINS-v102.json','CPU-SMOKE.json','TEST-RESULTS.json']
    if (h.OUT/'VAST-CPU-SMOKE.json').exists(): paths.append('VAST-CPU-SMOKE.json')
    return {p:h.sha(h.OUT/p) for p in paths}


def require_review():
    r=h.read(h.OUT/'REVIEW.json')
    h.need(r.get('verdict')=='PASS' and r.get('reviewer') not in (None,'','astra-b4'),'independent review required')
    h.need(r.get('bindings')==review_bindings(),'review evidence binding')
    h.need(r.get('approved_hosts') and set(r['approved_hosts'])<=set(['local','vast']),'review hosts')
    report=(h.ROOT/r['report']).resolve()
    h.need(report.is_relative_to(h.ROOT) and h.sha(report)==r['report_sha256'],'review report identity')
    return r


def admit_dry_run(host,launch=False):
    reg=h.validate(host=='vast'); smoke=h.read(h.OUT/'CPU-SMOKE.json')
    h.need(smoke['status']=='PASS' and not smoke['cuda_initialized'] and smoke['registration_sha256']==h.sha(h.OUT/'PREREGISTRATION.json'),'smoke binding')
    blockers=[]
    try:
        r=require_review(); h.need(host in r['approved_hosts'],'host not reviewed')
        h.need(r.get('approved_environments',{}).get(host)==cpu_environment(), 'host runtime must match independent reviewed environment')
    except (OSError,KeyError,ValueError) as e: blockers.append('independent review: '+type(e).__name__)
    if host=='vast':
        if h.ROOT!=VAST_ROOT: blockers.append('Vast package not deployed/CPU-prepared; host cache/environment unverified')
        else:
            v=h.read(h.OUT/'VAST-CPU-SMOKE.json'); h.need(v['status']=='PASS' and v['cache']==smoke['cache'] and v['registration_sha256']==smoke['registration_sha256'],'Vast cache/CPU parity')
            try: vast_transfer(launch=launch)
            except (OSError,KeyError,ValueError) as e: blockers.append('B exclusive Vast transfer: '+type(e).__name__)
    return dict(status='BLOCKED' if blockers else 'CPU_ADMIT_PASS',host=host,blockers=blockers,
                required_review_bindings=review_bindings(),gpu_started=False)


def owner_check(host):
    import psutil
    from tools import job_runtime as jobs
    h.need(jobs.ROOT==Path(config(host)['owner']).resolve(),'canonical host owner root')
    job=jobs.read_json(jobs.LOGS/TAG/'job.json'); cmd=command(host)
    h.need(job['command']==cmd and job['command_sha256']==jobs.command_digest(cmd,h.ROOT) and Path(job['cwd']).resolve()==h.ROOT,'owner command/cwd')
    h.need(os.getppid()==job['pid'] and jobs.liveness(job)=='alive' and psutil.Process(job['pid']).cmdline()[-3:]==['--_run-job',TAG,job['run_id']],'gpu_job parent required')
    for _ in range(50):
        job=jobs.read_json(jobs.LOGS/TAG/'job.json')
        if job.get('child_identity')==jobs.identity(): break
        time.sleep(.1)
    h.need(job.get('child_identity')==jobs.identity() and job['phase']=='running','gpu_job child identity')
    return job


def snapshot(host):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
    lines=raw.strip().splitlines(); h.need(len(lines)==1,'single physical GPU required')
    name,total,used,util,power=[v.strip() for v in lines[0].split(',')]
    h.need(config(host)['device'] in name,'registered host GPU mismatch')
    return dict(name=name,total_bytes=int(total)*(1<<20),desktop_bytes=int(used)*(1<<20),util=float(util),power=float(power))


def desktop_process(host,row):
    import ntpath
    import re
    import psutil
    parts=row.split(',')
    if host!='local' or len(parts)!=3: return False
    pid,name,memory=(v.strip() for v in parts)
    if not pid.isdecimal() or int(pid)<=0 or not name: return False
    names=[name]
    # WDDM may hide the name from nvidia-smi. Resolve it when accessible;
    # AccessDenied alone must not reject an allocation-free desktop context.
    try: names.append(psutil.Process(int(pid)).name())
    except psutil.Error: pass
    ml_names=('python','pypy','train','torch','tensorflow','triton','cuda',
              'jupyter','ipykernel','onnx','vllm','ollama','llama','accelerate','deepspeed')
    if any(token in ntpath.basename(n).lower() for n in names for token in ml_names): return False
    if memory=='[N/A]': return True
    amount=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:MiB)?',memory)
    if amount is None: return False
    return float(amount[1])==0 or int(pid)==os.getpid()


def idle(host):
    snap=snapshot(host)
    fields='pid,process_name,used_memory' if host=='local' else 'pid,process_name'
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps='+fields,'--format=csv,noheader,nounits'],text=True)
    h.need(all(desktop_process(host,row) for row in rows.splitlines() if row.strip()) and
           0<=snap['util']<(60 if host=='local' else 20),'GPU busy; no waiting or interference')
    return snap


def stall(status,now,telemetry,history):
    lag=now-status['time']
    if lag>=900: return '15-minute no progress'
    if lag>=600 and telemetry['util']>=95 and telemetry['power']<90: return 'WDDM stall signature'
    same=[x for x in history if x['phase']==status['phase'] and now-x['time']<=1830]
    if same and now-same[0]['time']>=1800 and status.get('healthy_rate',0)>0:
        if (status['done']-same[0]['done'])/(now-same[0]['time']) < .25*status['healthy_rate']: return '30-minute throughput collapse'
    return None


def run(host):
    from tools import job_runtime as jobs
    h.need(admit_dry_run(host,launch=True)['status']=='CPU_ADMIT_PASS','review/preparation required')
    job=owner_check(host); h.need(not (h.OUT/'features/receipt.json').exists(),'completed extraction cannot be replayed')
    proof=cache_proof(config(host)['cache'],h.validate(host=='vast'))
    h.need(proof==h.read(h.OUT/'CPU-SMOKE.json')['cache'],'cache identity before GPU')
    snap=idle(host); start=time.time(); child=None; error=None
    h.write(h.OUT/'ADMISSION.json',dict(host=host,snapshot=snap,supervisor=jobs.identity(),job_run_id=job['run_id'],
        registration_sha256=h.sha(h.OUT/'PREREGISTRATION.json'),review_sha256=h.sha(h.OUT/'REVIEW.json')))
    try:
        child=subprocess.Popen([sys.executable,'-B','-m','tools.k1_syn_head_v1','worker','--host',host],cwd=h.ROOT,
            env={**os.environ,'K1_SYN_PARENT':str(os.getpid())},creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        history=[]
        while child.poll() is None:
            status=h.read(h.OUT/'STATUS.json') if (h.OUT/'STATUS.json').exists() else dict(time=start,done=0,phase='startup')
            # Ignore a previous attempt's heartbeat until this worker publishes.
            if status.get('run_id')!=job['run_id']: status=dict(time=start,done=0,phase='startup')
            now=time.time(); telemetry=snapshot(host); reason=stall(status,now,telemetry,history)
            h.need(not reason,reason or 'watchdog'); history.append(dict(status,time=now)); history=[x for x in history if now-x['time']<=1830]
            time.sleep(5)
        h.need(child.returncode==0,'owned extraction worker failed')
    except Exception as exc: error=type(exc).__name__+': '+str(exc); raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        h.write(h.OUT/'TERMINAL.json',dict(status='FAILED' if error else 'SUCCESS',error=error,run_id=job['run_id'],seconds=time.time()-start))
    return dict(status='FEATURES_COMPLETE',scientific_fit_performed=False)


def row_read(path,signature,uid):
    import numpy as np
    meta=h.read(path.with_suffix('.json'))
    h.need(meta['signature']==signature and meta['uid_digest']==h.digest(uid) and meta['sha256']==h.sha(path),'resume row identity/corruption')
    with np.load(path,allow_pickle=False) as data: row={k:data[k] for k in data.files}
    h.need(row['features'].shape==(384,) and row['predictions'].shape==(12,) and row['initial'].shape==(385,) and all(np.isfinite(v).all() for v in row.values()),'resume layout')
    return row


def worker(host):
    import psutil
    from tools import job_runtime as jobs
    a=h.read(h.OUT/'ADMISSION.json'); parent=a['supervisor']
    h.need(os.getppid()==parent['pid']==int(os.environ.get('K1_SYN_PARENT','0')) and jobs.liveness(parent)=='alive','owned worker only')
    h.need(psutil.Process(parent['pid']).cmdline()[1:]==command(host)[1:],'supervisor command')
    h.need(admit_dry_run(host)['status']=='CPU_ADMIT_PASS' and h.sha(h.OUT/'REVIEW.json')==a['review_sha256'],'worker review identity')
    reg=h.validate(host=='vast'); ct=import_trainer(); import torch; import numpy as np; import timm
    torch.set_num_threads(4)
    runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),timm=timm.__version__,
        numpy=np.__version__,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        tf32=False,cudnn_benchmark=False,autocast='float16',host=host)
    signature=h.digest(dict(registration=a['registration_sha256'],runtime=runtime,checkpoint=h.CHECKPOINT_SHA))
    h.write(h.OUT/'RUNTIME.json',runtime); net=model(ct,'cuda'); outputs={}; peak=0; worst=0.
    def heartbeat(phase,done,healthy=0.):
        with contextlib.suppress(PermissionError): h.write(h.OUT/'STATUS.json',dict(time=time.time(),phase=phase,done=done,healthy_rate=healthy,run_id=a['job_run_id']))
    def memory():
        p=torch.cuda.max_memory_reserved(); s=a['snapshot']
        h.need(p<=12*(1<<30) and s['desktop_bytes']+p+2*(1<<30)<=s['total_bytes'],'12GiB peak + measured2GiB headroom')
        return p
    for split in ('train','holdout'):
        ds=dataset(ct,config(host)['cache'],reg[split],split); heartbeat('memory_'+split,0)
        # Largest actual window count, without reading the image payload twice.
        meta=np.load(Path(config(host)['cache'])/'meta.f32.npy',mmap_mode='r')
        counts=[len(ct.window_rows(ds.blocks[u],lambda f,r:meta[r],'eval',3,2,phys_mm=3.5)) for u in ds.uids]
        largest=max(range(len(ds)),key=counts.__getitem__)
        forward(net,ct,ds[largest],'cuda'); peak=max(peak,memory())
        rows=[]; start=time.time(); healthy=0.; computed=0
        for i,uid in enumerate(reg[split]):
            path=h.OUT/'features'/split/f'{i:04d}.npz'; rowmeta=path.with_suffix('.json')
            if rowmeta.exists(): row=row_read(path,signature,uid)
            else:
                f,p,initial,err=forward(net,ct,ds[i],'cuda'); worst=max(worst,err); peak=max(peak,memory()); computed+=1
                row=dict(features=f,predictions=p,initial=initial); h.npz(path,**row)
                h.write(rowmeta,dict(signature=signature,uid_digest=h.digest(uid),sha256=h.sha(path)))
            rows.append(row)
            if computed==50 and healthy==0: healthy=50/(time.time()-start)
            heartbeat(split,i+1,healthy)
        h.need(all(np.array_equal(r['initial'],rows[0]['initial']) for r in rows),'frozen trained head drift')
        path=h.OUT/'features'/f'{split}.npz'
        h.npz(path,ids=np.array(reg[split]),features=np.stack([r['features'] for r in rows]),
              predictions=np.stack([r['predictions'] for r in rows]),initial=rows[0]['initial'])
        outputs[split]=h.sha(path)
    h.write(h.OUT/'features/receipt.json',dict(status='COMPLETE',registration_sha256=a['registration_sha256'],
        review_sha256=a['review_sha256'],runtime=runtime,signature=signature,outputs=outputs,
        checkpoint_sha256=h.CHECKPOINT_SHA,peak_reserved=peak,max_logit_reconstruction_error=worst,
        run_id=a['job_run_id'],clean_oof=False))
    return dict(status='COMPLETE')


def package(host):
    import tarfile
    h.need(os.name=='nt' and sys.version_info[:2]==(3,12),'package locally with Python3.12 only')
    h.validate(); h.need(host=='vast','package target is Vast')
    names=h.source_names()+[h.CHECKPOINT,h.PARENT+'/registration.json',h.PARENT+'/prepare.json',h.PARENT+'/targets.csv']
    names += ['tools/__init__.py'] if (h.ROOT/'tools/__init__.py').exists() else []
    names += [str((h.OUT/n).relative_to(h.ROOT)).replace('\\','/') for n in ('PREREGISTRATION.json','OPERATIONAL-PINS-v102.json','CPU-SMOKE.json','TEST-RESULTS.json','LAUNCH.md')]
    # Review is a later independent artifact. Repack after it exists, without changing code.
    if (h.OUT/'REVIEW.json').exists():
        r=require_review(); names += [str((h.OUT/'REVIEW.json').relative_to(h.ROOT)).replace('\\','/'),r['report']]
    manifest={n:h.sha(h.ROOT/n) for n in sorted(set(names))}; h.write(h.OUT/'PACKAGE-MANIFEST.json',dict(files=manifest,destination=VAST_ROOT.as_posix(),built_python=sys.version,review_included=(h.OUT/'REVIEW.json').exists()))
    dest=h.OUT/'vast-package.tar'; temp=dest.with_suffix('.tmp')
    with tarfile.open(temp,'w') as tar:
        for name in manifest: tar.add(h.ROOT/name,arcname=name,recursive=False)
        tar.add(h.OUT/'PACKAGE-MANIFEST.json',arcname=str((h.OUT/'PACKAGE-MANIFEST.json').relative_to(h.ROOT)).replace('\\','/'))
    h.replace(temp,dest)
    with tarfile.open(dest) as tar:
        for name,pin in manifest.items(): h.need(__import__('hashlib').sha256(tar.extractfile(name).read()).hexdigest()==pin,'package roundtrip')
    result=dict(status='PACKAGED',files=len(manifest),bytes=dest.stat().st_size,sha256=h.sha(dest),review_included=(h.OUT/'REVIEW.json').exists(),deployed=False)
    h.write(h.OUT/'PACKAGE-RESULT.json',result); return result
