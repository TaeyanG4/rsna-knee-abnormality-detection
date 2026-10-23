"""H-B-224 v1.0.0: additive K1 v3-G seed0 hash-fold derivatives.
Changelog: fold-specific roster/state/exports; portable supervised ownership.
The scientific delta is the held-out hash fold. No GPU work on import.
"""
from __future__ import annotations
import argparse, csv, difflib, hashlib, importlib, json, math, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools import l1_rco_stage0_v1 as s
from tools import l1_rco_k1_v3 as parent
from tools import l1_rco_k1_v1 as v1
from tools import l1_rco_k1_optim_v3 as optim
need, read, write, sha, digest = s.need, s.read, s.write, s.sha, s.digest
VERSION = '1.0.0'
BASE = ROOT / '.local/scratch/astra-b2/h224-folds-v1'
PARENT = ROOT / '.local/scratch/l1-rco/k1v3-grouping/registration.json'
FULL_REG = parent.FULL_REG
CACHE = parent.CACHE if os.name == 'nt' else BASE / 'cache'
PYTHON = sys.executable
GUARD, EPOCHS, COLLAPSE = parent.GUARD, parent.EPOCHS, parent.COLLAPSE
FOLD = int(os.environ.get('K1_FOLD', '1'))
GPU = int(os.environ.get('K1_GPU', '0'))

def configure(fold, gpu=0):
    global FOLD, GPU, WORK, OUT, TAG, REVIEW, BEST, TRAIN_STUDIES
    need(fold in (1,2,3) and gpu >= 0, 'fold must be 1,2,3; GPU nonnegative')
    FOLD, GPU = fold, gpu
    os.environ['K1_FOLD'] = str(fold); os.environ['K1_GPU'] = str(gpu)
    WORK = BASE / ('f%d' % fold); OUT = WORK / 'fit'
    TAG = 'H224_K1_V3G_F%d' % fold; REVIEW = WORK / 'REVIEW.json'
    BEST = 'ckpt/best_E_prior_s0_f%d.pt' % fold
    TRAIN_STUDIES = (0,3073,3100,3138)[fold]
configure(FOLD, GPU)

def code_pins():
    names = ['l1_rco_k1_folds_v1.py','l1_rco_k1_foldstate_v1.py',
             'l1_k1fold_trainer_v1.py','l1_rco_k1_foldsmoke_v1.py',
             'l1_rco_k1_foldcache_v1.py','l1_rco_k1_foldpack_v1.py']
    return dict(read(PARENT)['code_pins'], **{'tools/'+n:sha(ROOT/'tools'/n) for n in names})

def training_rows(reg):
    with (ROOT/s.TARGETS).open(newline='', encoding='utf-8') as stream:
        return s.target_rows(dict(train=reg['train'],holdout=[]), csv.DictReader(stream))

def expected_registration():
    old = read(PARENT)
    need(old['contract_id'] == 'a58f1476ea4ca3c8d64fe06c80bc6e24d78fdf2f347c4de68c1eba5d9c2ea83d', 'sealed seed0 parent')
    need(digest({k:v for k,v in old.items() if k!='contract_id'}) == old['contract_id'], 'parent digest')
    for name,value in dict(old['pins'], **old['code_pins']).items():
        need(sha(ROOT/name)==value, 'source pin changed: '+name)
    s.verify_pins()
    reg = json.loads(json.dumps(old)); ids=sorted(old['train']+old['holdout'])
    reg['train']=[u for u in ids if s.fold(u)!=FOLD]
    reg['holdout']=[u for u in ids if s.fold(u)==FOLD]
    need(len(ids)==len(set(ids))==4167 and len(reg['train'])==TRAIN_STUDIES,'fold roster')
    full=read(ROOT/s.REG)
    need(not set(ids)&set(full['gold']), 'Gold overlap')
    need(not {full['components'][u] for u in reg['train']} &
         {full['components'][u] for u in reg['holdout']}, 'connected component leakage')
    reg.update(version=VERSION,tag=TAG,code_pins=code_pins(),parent_seed0_contract=old['contract_id'],
               parent_registration_sha256=sha(PARENT),changes_vs_seed0=['held-out hash fold'], model_fold=FOLD)
    recipe=reg['recipe']; steps=len(reg['train'])//recipe['batch']
    recipe.update(model_fold=FOLD,only_folds=str(FOLD),steps_per_epoch=steps,total_steps=10*steps,warm_steps=steps,
                  bias='logit(mean(training-only)).clamp(.001,.999)',unchanged='seed0 recipe; original fold-dependent RNG formulas')
    reg['roster']['train']=digest(reg['train']);reg['roster']['holdout']=digest(reg['holdout'])
    constant=parent.constant_loss(training_rows(reg))
    reg['collapse_guard'].update(constant_loss=constant,threshold=constant-.002,
                                constant_definition='mean BCE of per-target training-only prior means')
    reg['best_checkpoint']['path']='fit/'+BEST
    reg['evaluator']={'rulers':['jev_hybrid','aquila_v1','dread','consensus'],
                      'cohort':'own hash-fold holdout; all targets; aggregate only',
                      'exposure':'EXPOSED_EXPLORATORY; cross-fold/cross-host comparisons are not paired'}
    reg['projection']={'fit_hours_calendar_estimate':steps*10*3.1/3600,
                       'fit_hours_training_lower_bound':steps*10*3.1/3600,
                       'scope':'4070 provisional 3.1 s/step; Linux 4090 unmeasured; excludes rendering/inference'}
    reg.pop('contract_id');reg['contract_id']=digest(reg)
    return reg

def validate(reg):
    need(reg==expected_registration(),'fold registration identity changed')
    return reg

def register():
    reg=expected_registration(); WORK.mkdir(parents=True,exist_ok=True)
    if (WORK/'registration.json').exists():
        need(read(WORK/'registration.json')==reg,'registration immutable; version a changed attempt')
    else: write(WORK/'registration.json',reg)
    return reg

def derive_trainer(source):
    source=parent.derive_trainer(source)
    changes=[('a.only_folds == "0"','a.only_folds == "'+str(FOLD)+'"'),
             ('all(pool_fold != 0)','all(pool_fold != '+str(FOLD)+')'),
             ('tools.l1_rco_k1_state_v2','tools.l1_rco_k1_foldstate_v1'),
             ('guard_constant=a.l1_guard_constant)','guard_constant=a.l1_guard_constant, fold='+str(FOLD)+')')]
    for old,new in changes:
        need(source.count(old)==1,'trainer patch identity: '+old);source=source.replace(old,new)
    need(source.count('tools.l1_rco_k1_v3')==2,'trainer identity imports')
    source=source.replace('tools.l1_rco_k1_v3','tools.l1_rco_k1_folds_v1')
    compile(source,'fold-trainer','exec');return source

def trainer_argv(cache,reg):
    cmd=s.trainer_argv(cache);cmd[0]=PYTHON;cmd[1]=str(WORK/'compact_train_l1.py')
    values={'--targets':WORK/'targets.csv','--out':OUT,'--only-folds':FOLD,'--epochs':10,
            '--lr':reg['recipe']['lr'],'--head-lr':reg['recipe']['head_lr'],'--ckpt-every':1}
    for flag,value in values.items():cmd[cmd.index(flag)+1]=str(value)
    return cmd+['--resume','--warm-frac','0.1','--l1-guard-constant',repr(reg['collapse_guard']['constant_loss'])]

def prepare(reg,cache):
    validate(reg);need(not (OUT/'admission.json').exists(),'admitted fit immutable')
    rows=training_rows(reg)
    with (WORK/'targets.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    source=(ROOT/s.SOURCE/'compact_train.py').read_text(encoding='utf-8')
    derived=derive_trainer(source)
    (WORK/'compact_train_l1.py').write_text(derived,encoding='utf-8',newline='\n')
    (WORK/'trainer.diff').write_text(''.join(difflib.unified_diff(parent.derive_trainer(source).splitlines(True),
              derived.splitlines(True),fromfile='seed0-fold0',tofile='seed0-fold%d'%FOLD)),encoding='utf-8')
    result=dict(contract_id=reg['contract_id'],trainer_sha256=sha(WORK/'compact_train_l1.py'),
                targets_sha256=sha(WORK/'targets.csv'),train_count=len(rows),holdout_training_rows=0,
                argv=trainer_argv(cache,reg),cache=str(Path(cache).resolve()))
    write(WORK/'prepare.json',result);return result

def check_training_ids(ids):
    reg=read(WORK/'registration.json')
    need(sorted(ids)==reg['train'] and not set(ids)&set(reg['holdout']),'training roster')

def check_trainer_args(a):
    reg=read(WORK/'registration.json')
    # Compare all supplied production flags, including parser defaults via the prepared CLI.
    need(sys.argv[1:]==trainer_argv(Path(a.px),reg)[2:] or
         sys.argv[1:]==trainer_argv(Path(a.px),reg)[2:]+['--check-targets'], 'production argv changed')

def command(fold=1,cache=None,review=None,gpu=0,action='run'):
    work=BASE/('f%d'%fold)
    return [PYTHON,str(Path(__file__).resolve()),action,'--fold',str(fold),'--gpu',str(gpu),
            '--cache',str(Path(cache or CACHE).resolve()),'--review',str(Path(review or work/'REVIEW.json').resolve())]

def launch_command(fold=1,cache=None,review=None,gpu=0):
    return [PYTHON,str(ROOT/'tools/gpu_job.py'),'--tag','H224_K1_V3G_F%d'%fold,'--']+command(fold,cache,review,gpu)

def run_command(cache=None,review=None):return command(FOLD,cache,review,GPU)

def import_trainer():
    return importlib.import_module('tools.l1_k1fold_trainer_v1')

complete_cache=v1.complete_cache
collapse_reason=parent.collapse_reason


def review_binding(reg):
    prep = read(WORK / 'prepare.json'); smoke = read(WORK / 'cpu-smoke.json')
    need(prep['contract_id'] == reg['contract_id'] and smoke['contract_id'] == reg['contract_id'], 'CPU receipt contract')
    need(prep['trainer_sha256'] == sha(WORK / 'compact_train_l1.py') and prep['targets_sha256'] == sha(WORK / 'targets.csv'), 'prepared artifact changed')
    need(smoke['status'] == 'PASS' and not smoke['cuda_initialized'], 'real CPU smoke required')
    return dict(contract_id=reg['contract_id'], registration_sha256=sha(WORK / 'registration.json'),
                primary_cohort_id_list_sha256=digest(reg['holdout']), prepare_sha256=sha(WORK / 'prepare.json'),
                cpu_smoke_sha256=sha(WORK / 'cpu-smoke.json'), fullcache_registration_sha256=sha(FULL_REG))


def admit(reg, cache, review, dry_run=True, owner_pid=None):
    blockers = []; binding = cache_proof = None
    try:
        validate(reg); prep = read(WORK / 'prepare.json')
        need(prep['argv'] == trainer_argv(cache, reg) and prep['cache'] == str(Path(cache).resolve()), 'prepared cache/argv changed')
        binding = review_binding(reg)
        r = read(review)
        need(r['verdict'] == 'PASS' and r['builder'] != r['reviewer'] and r['bindings'] == binding, 'independent exact-artifact review required')
        need(sha(ROOT / r['report']) == r['report_sha256'], 'review report changed')
    except (ValueError, KeyError, OSError) as exc: blockers.append(str(exc))
    try:
        cache_proof = complete_cache(reg, cache)
        if os.name!='nt':
            proof=read(Path(cache)/'H224-PARITY.json')
            need(proof['status']=='PASS' and proof['studies']==32 and not proof['smoke'] and
                 proof['cache_receipt_sha256']==sha(Path(cache)/'receipt.json') and
                 proof['reference_sha256']==sha(BASE/'local32.json'), 'Linux exact cache parity required')
    except (ValueError, KeyError, OSError) as exc: blockers.append(str(exc))
    result = dict(status='BLOCKED' if blockers else 'CPU_ADMISSIBLE', blockers=blockers, bindings=binding, cache=cache_proof, dry_run=dry_run)
    if dry_run:
        write(WORK / 'admit-dry-run.json', result); return result
    need(not blockers, '; '.join(blockers))
    need(not (OUT/'admission.json').exists(), 'consumed admission; no automatic retry')
    from tools.l1_rco_runtime_v120 import memory_ok
    need(not active_ml_processes(owner_pid), 'another Python ML GPU process exists')
    snapshot = gpu_snapshot(); k0 = read(parent.K0_RESULT)
    need(memory_ok(k0['peak_reserved'], snapshot['desktop_bytes'], snapshot['total_bytes']), 'K0 measured peak + current desktop +2GiB exceeds memory gate')
    return dict(snapshot, contract_id=reg['contract_id'], review_binding=binding, cache=cache_proof)


def require_gpu_job(cache, review):
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    command = run_command(cache, review)
    with file_lock(LOGS / '.gpu-claim.lock'):
        job = read_json(LOGS / TAG / 'job.json'); parent = psutil.Process().parent()
        need(parent is not None and job.get('schema') == 'gpu-job/v2' and job.get('tag') == TAG and job.get('phase') == 'running', 'B must use gpu_queue/gpu_job')
        need(job.get('child_identity') == identity() and {k:job.get(k) for k in ('pid','create_time')} == identity(parent.pid), 'gpu_job process identity')
        scope=owner_scope()
        need(job.get('command') == command and job.get('command_sha256') == command_digest(command, scope) and Path(job['cwd']).resolve() == scope, 'gpu_job command identity')
        need(parent.cmdline()[-3:] == ['--_run-job', TAG, job['run_id']] and any(Path(x).name == 'gpu_job.py' for x in parent.cmdline()), 'gpu_job parent identity')
        return parent.pid


def require_infer_job(cache, review):
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    command = run_command(cache, review)
    with file_lock(LOGS / '.gpu-claim.lock'):
        job = read_json(LOGS / TAG / 'job.json'); parent = psutil.Process().parent()
        need(parent is not None and job.get('schema') == 'gpu-job/v2' and job.get('tag') == TAG and job.get('phase') == 'running', 'inference requires live B gpu_job')
        need(job.get('child_identity') == identity(parent.pid), 'inference must be an owned run child')
        need(job.get('command') == command and job.get('command_sha256') == command_digest(command, owner_scope()), 'inference parent command identity')
        need(Path(parent.cmdline()[1]).resolve() == ROOT / 'tools/l1_rco_k1_folds_v1.py' and parent.cmdline()[2] == 'run', 'inference parent must be K1 v2 run')


def completed_fit(reg, out=None):
    out=OUT if out is None else out
    receipt = read(Path(out) / 'fit-complete.json')
    need(receipt['status'] == 'COMPLETE' and receipt['epochs'] == reg['recipe']['epochs'] and receipt['model_fold'] == FOLD and
         receipt['train_studies'] == TRAIN_STUDIES and receipt['contract_id'] == reg['contract_id'] and
         receipt['optimizer_steps'] == reg['recipe']['total_steps'] and
         receipt['model_sha256'] == sha(Path(out) / ('models/E_prior_s0_f%d.pt' % FOLD)), 'completed fit identity')
    return receipt


def inference(reg, cache, out=None, cpu=False, ids=None, limit_windows=None):
    import numpy as np
    import torch
    from tools.l1_rco_k1_state_v1 import Monitor
    ct = import_trainer(); out = Path(OUT if out is None else out)
    need(cpu or ids is None, 'production inference roster is the registered fold')
    ids = list(reg['holdout'] if ids is None else ids)
    need(ids and len(ids) == len(set(ids)) and set(ids) <= set(reg['holdout']), 'holdout inference roster')
    dev = torch.device('cpu' if cpu else 'cuda')
    if cpu: need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU isolation')
    else: completed_fit(reg, out)
    model = ct.Net('convnext_small', str(ROOT / s.INIT), crop=352, in_chans=3, spatial=True, extent=False).to(dev)
    model.load_deploy_state_dict(torch.load(out / ('models/E_prior_s0_f%d.pt' % FOLD), map_location='cpu', weights_only=False))
    model.eval(); monitor = Monitor(out, cpu=cpu); monitor.phase('eval')
    ds = ct.Studies(str(cache), ids, 64, 'eval', 0, eval_stride=3, slices=3, phys_mm=3.5)
    # Same stage-0 serving windows, center crop, bf16 and encoder chunk32. A batch
    # contains exactly one study; no labels, fitter state or target file loaded.
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0 if cpu else 3, collate_fn=ct.collate)
    values = np.empty((len(ids),12), np.float32)
    with torch.no_grad():
        for img, meta, mask, index in dl:
            if limit_windows is not None: img, meta, mask = img[:,:limit_windows], meta[:,:limit_windows], mask[:,:limit_windows]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not cpu):
                logits = model(img[...,16:368,16:368].to(dev), meta.to(dev), mask.to(dev), chunk=32)
            monitor.evaluation(logits); values[index.numpy()] = logits.float().sigmoid().cpu().numpy()
    need(values.shape == (len(ids),12) and np.isfinite(values).all(), 'holdout prediction shape/finite')
    path = out / 'holdout.csv'
    with path.with_suffix('.tmp').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow([s.UID]+s.LABELS+['status','model_fold'])
        for uid,row in zip(ids,values): writer.writerow([uid]+row.tolist()+['ok',FOLD])
    s.replace(path.with_suffix('.tmp'), path); monitor.phase('complete')
    result = dict(version=VERSION,status='COMPLETE',contract_id=reg['contract_id'],holdout_rows=len(ids),
                  holdout_sha256=sha(path),model_sha256=sha(out / ('models/E_prior_s0_f%d.pt' % FOLD)),
                  model_fold=FOLD, eval_chunk=32, clean_oof=False, exposure='EXPOSED_EXPLORATORY',
                  cuda_initialized=torch.cuda.is_initialized(), cpu_test=cpu,
                  decision='PENDING independent result review; CPU evaluate applies registered K1 rule')
    write(out / 'result.json', result)
    return result


class CollapseStop(RuntimeError):
    """Registered COLLAPSE outcome: run FAILED, checkpoints kept, no inference."""
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('COLLAPSE at epoch %s: %s' % (receipt.get('epoch'), '; '.join(receipt.get('criteria', []))))


def collapse_stop(out, reason, source):
    """Write collapse.json (if the trainer did not) and failure.json FAILED/COLLAPSE."""
    out = Path(out)
    if not (out / COLLAPSE).exists():
        write(out / COLLAPSE, dict(reason, version=VERSION, status='FAILED', source=source))
    receipt = read(out / COLLAPSE)
    ckpt = out / ('ckpt/E_prior_s0_f%d.pt' % FOLD)
    write(out / 'failure.json', dict(status='FAILED', reason='COLLAPSE', source=source, time=time.time(), partial_preserved=True,
                                     error='COLLAPSE: ' + '; '.join(receipt.get('criteria', [])), epoch=receipt.get('epoch'),
                                     loss=receipt.get('loss'), constant_loss=receipt.get('constant_loss'),
                                     checkpoint=str(ckpt), checkpoint_kept=ckpt.exists(),
                                     best_checkpoint=str(out / BEST) if (out / BEST).exists() else None))
    try: write(out / 'status.json', dict(phase='failed-collapse', time=time.time(), windows=0))
    except OSError: pass
    return CollapseStop(receipt)


def epoch_collapse(out, constant):
    try: sidecar = read(Path(out) / EPOCHS)
    except (OSError, ValueError): return None  # absent or mid-replace; next poll
    return collapse_reason(sidecar['log'], constant)


def stop_child(child):
    if child is not None and child.poll() is None:
        if os.name=='nt': subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=False)
        else:
            import psutil
            owned=psutil.Process(child.pid).children(recursive=True)
            for process in reversed(owned):
                try: process.terminate()
                except psutil.NoSuchProcess: pass
            child.terminate()
            _,alive=psutil.wait_procs(owned,timeout=10)
            for process in alive:
                try: process.kill()
                except psutil.NoSuchProcess: pass
        child.wait(timeout=30)


def supervise(name, command, env, out, constant, poll=15., grace=GUARD['backstop_grace_s'], gpu=True):
    """v1 child loop (stale/WDDM/throughput watchdogs) plus the registered collapse guard."""
    from tools.l1_rco_runtime_v120 import stale_reason
    from tools.l1_rco_k1_state_v1 import ThroughputGuard
    out = Path(out); child = None; detected = None
    write(out / 'status.json', dict(phase='launch-'+name,time=time.time(),windows=0))
    try:
        with (out / (name+'.log')).open('a',encoding='utf-8') as log:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status = read(out / 'status.json'); throughput = ThroughputGuard(time.time())
            while child.poll() is None:
                time.sleep(poll)
                try: status=read(out / 'status.json')
                except (OSError,json.JSONDecodeError): pass
                telemetry=None
                if gpu:
                    try: telemetry=gpu_snapshot()
                    except (OSError,subprocess.SubprocessError,ValueError): telemetry=None
                reason=stale_reason(status,time.time(),telemetry); need(not reason,reason or '')
                throughput.observe(status,time.time())
                if name == 'train':
                    collapse = epoch_collapse(out, constant)
                    if collapse and child.poll() is None:
                        detected = detected or time.time()
                        if time.time() - detected >= grace:
                            stop_child(child)  # trainer ignored its own guard: backstop, checkpoints untouched
                            raise collapse_stop(out, collapse, 'launcher backstop (trainer alive %.0f s after collapsed epoch log)' % (time.time() - detected))
            if name == 'train' and child.returncode == GUARD['exit_code']:
                need((out / COLLAPSE).exists(), 'collapse exit without collapse receipt')
                raise collapse_stop(out, read(out / COLLAPSE), 'trainer epoch-end guard')
            need(child.returncode==0,name+' child failed; partial checkpoint retained')
    except CollapseStop:
        raise
    except Exception as exc:
        write(out/'failure.json',dict(status='FAILED',reason='ERROR',error=str(exc),time=time.time(),partial_preserved=True)); raise
    finally:
        stop_child(child)


def final_training_log(reg, out=None):
    """After a train exit 0: the committed log must be complete and never collapsed."""
    out=OUT if out is None else out
    sidecar = read(Path(out) / EPOCHS); constant = reg['collapse_guard']['constant_loss']
    reason = collapse_reason(sidecar['log'], constant)
    if reason: raise collapse_stop(out, reason, 'launcher post-fit log check')
    need(sidecar['committed_epochs'] == len(sidecar['log']) == reg['recipe']['epochs'], 'completed epoch log')
    return sidecar


def run(cache, review):
    owner = require_gpu_job(cache, review); reg = read(WORK / 'registration.json')
    if (OUT / COLLAPSE).exists():
        raise CollapseStop(read(OUT / COLLAPSE))  # FAILED run is never relaunched or resumed
    admission = admit(reg, cache, review, False, owner)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'admission.json').exists():
        prior = read(OUT / 'admission.json')
        need(prior['contract_id'] == admission['contract_id'] and prior['review_binding'] == admission['review_binding'] and prior['cache'] == admission['cache'], 'resume admission identity')
    write(OUT / 'admission.json', admission)
    os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
    write(OUT / 'runtime.json', runtime_record())
    print(json.dumps(dict(estimated_fit_hours_lower_bound=reg['projection']['fit_hours_training_lower_bound'],
                          estimated_fit_hours_calendar=reg['projection']['fit_hours_calendar_estimate'], scope=reg['projection']['scope'])), flush=True)
    commands = [trainer_argv(cache, reg), command(FOLD,cache,review,GPU,'infer')]
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(GPU); env['K1_FOLD']=str(FOLD); env['K1_GPU']=str(GPU); env['PYTHONDONTWRITEBYTECODE']='1'; env['CT_PREDICT_CHUNK'] = '32'
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT / s.CODE)])
    for name,child_command in zip(('train','infer'),commands):
        if name == 'train' and (OUT / 'fit-complete.json').exists():
            completed_fit(reg); final_training_log(reg)
            continue
        supervise(name, child_command, env, OUT, reg['collapse_guard']['constant_loss'])
        if name == 'train': final_training_log(reg)
    export(reg)
    return read(OUT/'result.json')

def gpu_snapshot():
    args=['nvidia-smi','-i',str(GPU),'--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits']
    rows=[x.strip().split(', ') for x in subprocess.check_output(args,text=True).splitlines() if x.strip()]
    need(len(rows)==1,'exactly one selected GPU required');r=rows[0]
    need(('4070 Ti SUPER' if os.name=='nt' else '4090') in r[1],'wrong host GPU type')
    return dict(index=int(r[0]),name=r[1],total_bytes=int(r[2])*2**20,desktop_bytes=int(r[3])*2**20,util=float(r[4]),power=float(r[5]))

def active_ml_processes(owner_pid=None):
    args=['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']
    if os.name!='nt':args+=['-i',str(GPU)]
    return [x for x in subprocess.check_output(args,text=True).splitlines() if x.strip() and (os.name!='nt' or 'python' in x.lower()) and x.split(',')[0].strip()!=str(owner_pid)]

def runtime_record():
    import importlib.metadata as md
    import platform
    import torch
    packages={n:md.version(n) for n in ['torch','torchvision','timm','numpy','scipy','pandas','pydicom','pillow','safetensors']}
    previous=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    from tools.parity_runtime import runtime_info
    if previous is None: os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
    else: os.environ['CUBLAS_WORKSPACE_CONFIG']=previous
    return dict(python=platform.python_version(),os=platform.platform(),packages=packages,snapshot=runtime_info(),
                torch_cuda_build=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized(),
                recipe_cudnn_benchmark=True,precision='bf16',configure_parity_applied=False,
                scope='launcher snapshot; trainer enables benchmark; cross-platform evidence unpaired')

def prediction_scores(reg,path):
    import numpy as np
    ids=reg['holdout']
    with Path(path).open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    need(len(rows)==len(ids) and {r[s.UID] for r in rows}==set(ids),'exact holdout roster required')
    by={r[s.UID]:r for r in rows}
    need(all(r['status']=='ok' and int(r['model_fold'])==FOLD for r in rows),'prediction status/fold')
    scores=np.array([[float(by[u][label]) for label in s.LABELS] for u in ids])
    need(np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(),'prediction probabilities')
    return scores

def evaluate(reg,predictions=None,dry_run=False):
    import numpy as np
    from tools import exp156_k_sweep as rulers
    from tools.l1_rco_k1_eval_v1 import auc
    ids=reg['holdout'];refs=rulers.references(ids,root=ROOT)
    if dry_run:
        result=dict(status='PASS',dry_run=True,fold=FOLD,holdout_rows=len(ids),rulers=list(refs),
                    coverage={name:int(mask.sum()) for name,(_,mask,_) in refs.items()})
    else:
        path=Path(predictions or OUT/'holdout.csv')
        scores=prediction_scores(reg,path)
        metrics={}
        for name,(target,mask,_) in refs.items():
            values=[float(auc(target[mask[:,c],c],scores[mask[:,c],c])) for c in range(12)]
            need(np.isfinite(values).all(),'undefined target AUC; HOLD, never omit a target')
            metrics[name]=dict(macro_auc=float(np.mean(values)),targets=12,cells=int(mask.sum()))
        result=dict(status='DESCRIPTIVE_ONLY',fold=FOLD,holdout_rows=len(ids),metrics=metrics,
                    predictions_sha256=sha(path),contract_id=reg['contract_id'],clean_oof=False,
                    comparison='same four definitions as fold0; different holdouts/hosts are not paired evidence')
    write(WORK/('evaluation-dry-run.json' if dry_run else 'evaluation.json'),result);return result

def export(reg):
    completed_fit(reg);final_training_log(reg);result=read(OUT/'result.json')
    need(result['contract_id']==reg['contract_id'] and result['holdout_rows']==len(reg['holdout'])
         and result['holdout_sha256']==sha(OUT/'holdout.csv') and not result['cpu_test']
         and result['status']=='COMPLETE' and result['model_fold']==FOLD
         and result['model_sha256']==sha(OUT/('models/E_prior_s0_f%d.pt'%FOLD)), 'production result required')
    prediction_scores(reg,OUT/'holdout.csv')
    files=[OUT/('models/E_prior_s0_f%d.pt'%FOLD),OUT/'holdout.csv',OUT/'result.json',OUT/'fit-complete.json',
           OUT/'runtime.json',WORK/'registration.json',WORK/'prepare.json']
    manifest={str(p.relative_to(WORK)).replace('\\','/'):sha(p) for p in files}
    write(WORK/'EXPORT.json',dict(version=VERSION,fold=FOLD,contract_id=reg['contract_id'],sha256=manifest))
    (WORK/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(manifest.items())),encoding='utf-8',newline='\n')
    return dict(status='PASS',fold=FOLD,files=len(files),manifest_sha256=sha(WORK/'EXPORT.json'))

def owner_scope():
    return ROOT if os.name=='nt' else BASE/('gpu%d-owner'%GPU)

def launch_pair():
    need(os.name!='nt','dual-GPU launch is Linux only')
    results=[]
    for fold,gpu in ((2,0),(3,1)):
        env=os.environ.copy();env['KNEE_WORKSPACE_ROOT']=str(BASE/('gpu%d-owner'%gpu))
        Path(env['KNEE_WORKSPACE_ROOT']).mkdir(parents=True,exist_ok=True)
        env['CUDA_VISIBLE_DEVICES']=str(gpu)
        env['PYTHONDONTWRITEBYTECODE']='1'
        results.append(subprocess.run(launch_command(fold,gpu=gpu),cwd=ROOT,env=env,check=True).returncode)
    return dict(dispatched=results)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','validate','prepare','admit','cpu-smoke','run','infer','evaluate','export','launch-pair'])
    ap.add_argument('--fold',type=int,choices=[1,2,3],default=1);ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--cache',type=Path,default=CACHE);ap.add_argument('--review',type=Path)
    ap.add_argument('--predictions',type=Path);ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    configure(a.fold,a.gpu);review=a.review or REVIEW
    if a.action not in ('run','infer','launch-pair'):os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.action=='launch-pair':result=launch_pair()
    elif a.action=='register':
        r=register();result=dict(status='PASS',contract_id=r['contract_id'],train=len(r['train']),holdout=len(r['holdout']))
    elif a.action=='run':
        if os.name!='nt':need(Path(os.environ.get('KNEE_WORKSPACE_ROOT','')).resolve()==(BASE/('gpu%d-owner'%GPU)).resolve(),'Linux GPU owner scope required')
        result=run(a.cache,review)
    else:
        reg=validate(read(WORK/'registration.json'))
        if a.action=='validate':result=dict(status='PASS',fold=FOLD,train=len(reg['train']),holdout=len(reg['holdout']))
        elif a.action=='prepare':result=prepare(reg,a.cache)
        elif a.action=='admit':
            need(a.dry_run,'admission only through gpu_job');result=admit(reg,a.cache,review)
        elif a.action=='cpu-smoke':
            from tools.l1_rco_k1_foldsmoke_v1 import cpu_smoke
            result=cpu_smoke(reg,a.cache)
        elif a.action=='infer':require_infer_job(a.cache,review);result=inference(reg,a.cache)
        elif a.action=='evaluate':result=evaluate(reg,a.predictions,a.dry_run)
        else:result=export(reg)
    print(json.dumps(result,allow_nan=False))

if __name__=='__main__':main()
