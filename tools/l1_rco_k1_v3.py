"""K1 v3-G 3.0.0: grouping-only experiment from the completed K1 v2 KEEP.

Changelog: pretrained sp_head_norm uses encoder LR; normalization affine, bias,
LayerScale and scalar/1D parameters have zero weight decay. Four fixed groups.
V2 LR 1e-4/head 3e-4, llrd 1, warmup 1 epoch, 10 epochs, clip 5, EMA .999,
initialization/data/split/loss/guard/state/serving/launch/watchdogs are unchanged.
Fresh distinct work/tag/registration; never resume or modify the v2 candidate.
The grouping package is unmeasured, not a guaranteed correction or score gain.
"""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import l1_rco_stage0_v1 as s
from tools import l1_rco_k1_v1 as v1
from tools import l1_rco_k1_v2 as parent
from tools import l1_rco_k1_optim_v3 as optim

VERSION = '3.0.0'
WORK = ROOT / '.local/scratch/l1-rco/k1v3-grouping'
OUT = WORK / 'fit'
CACHE = Path('G:/temp_data/rsna-knee/l1rco-cache-full')
FULL_REG = ROOT / '.local/scratch/l1-rco/fullcache/registration.json'
QUEUE = ROOT / '.local/scratch/slot-b/l1rco/k1v3-grouping-gpu-queue.json'
K0_RESULT = ROOT / '.local/scratch/l1-rco/k0-v2/k0/result.json'
K0_REVIEW = ROOT / '.local/scratch/slot-b/l1rco/k0v2-fullcache-review-out.md'
REVIEW = WORK / 'K1V3-REVIEW.json'
V1_REG = v1.WORK / 'registration.json'
V1_DIAG = ROOT / '.local/scratch/slot-b/k1-diag/verdict.json'
V1_OBSERVED_WPS = 40.925715706088475  # v1 fit/status.json at stop: 1788800 windows / 43708.46 train s (I/O-bound full cache)
PYTHON = s.PYTHON
TAG = 'L1_RCO_K1_F0_V3G'
TRAIN_STUDIES, BATCH = 3190, 2
STEPS_PER_EPOCH = TRAIN_STUDIES // BATCH  # trainer DataLoader drop_last=True
CHANGES = dict(lr=(3e-4, 1e-4), head_lr=(1e-3, 3e-4), epochs=(16, 10), warm_frac=(.05, .1))
GUARD = dict(min_epoch=2, margin=.002, rise=.015, exit_code=4, backstop_grace_s=300)
BEST = 'ckpt/best_E_prior_s0_f0.pt'
EPOCHS = 'ckpt/epochs.json'
COLLAPSE = 'collapse.json'
need, read, write, sha, digest = s.need, s.read, s.write, s.sha, s.digest
training_rows, complete_cache = v1.training_rows, v1.complete_cache  # reviewed v1 code, unchanged


def code_pins():
    names = ['l1_rco_k1_v3.py', 'l1_rco_k1_optim_v3.py', 'l1_rco_k1_smoke_v3.py',
             'l1_k1v3_trainer.py', 'l1_k1v2_trainer.py']
    return dict(parent.code_pins(), **{'tools/' + n: sha(ROOT / 'tools' / n) for n in names})


def expected_registration():
    """Clone sealed v2 without recomputing/printing training-label rows."""
    old = parent.validate(read(parent.WORK / 'registration.json'))
    prep = read(parent.WORK / 'prepare.json')
    need(prep['contract_id'] == old['contract_id'], 'parent prepare contract')
    need(sha(parent.WORK / 'compact_train_l1.py') == prep['trainer_sha256'] ==
         '34dae104527a2e8c27ed328c5e2bca4039d28e7e596ab530b32de63a95848d8e', 'audited v2 trainer changed')
    need(sha(parent.WORK / 'targets.csv') == prep['targets_sha256'], 'parent target bytes changed')
    reg = json.loads(json.dumps(old))
    reg.update(version=VERSION, tag=TAG, code_pins=code_pins(), optimizer_policy=optim.POLICY,
               parent_v2=dict(tag=parent.TAG, contract_id=old['contract_id'],
                   registration_sha256=sha(parent.WORK / 'registration.json'),
                   prepare_sha256=sha(parent.WORK / 'prepare.json'),
                   trainer_sha256=prep['trainer_sha256'], targets_sha256=prep['targets_sha256'],
                   finding='D-B-220: completed v2 KEEP; v3 grouping is an unmeasured package hypothesis'),
               changes_vs_v2=['pretrained sp_head_norm uses encoder LR',
                             'normalization/bias/LayerScale/scalar/1D have weight_decay=0'])
    reg.pop('contract_id')
    reg['contract_id'] = digest(reg)
    return reg


def collapse_reason(log, constant_loss, margin=GUARD['margin'], rise=GUARD['rise'], min_epoch=GUARD['min_epoch']):
    """Registered K1 v2 collapse guard over committed 0-based per-epoch training losses.

    COLLAPSE at the first epoch e with (e >= min_epoch and loss_e >= constant - margin)
    or (e >= 1 and loss_e - min(loss_0..loss_{e-1}) > rise). Comparisons are rounded to
    1e-9 so 5-decimal logged values at an exact boundary do not trip on float error.
    """
    need(isinstance(constant_loss, float) and math.isfinite(constant_loss) and constant_loss > 0, 'collapse guard constant loss')
    best = None
    for epoch, loss in enumerate(log):
        loss = float(loss); need(math.isfinite(loss), 'nonfinite committed epoch loss')
        criteria = []
        if epoch >= min_epoch and round(loss - (constant_loss - margin), 9) >= 0:
            criteria.append('loss %.5f >= constant-prediction loss %.5f - %.3f' % (loss, constant_loss, margin))
        if best is not None and round(loss - best, 9) > rise:
            criteria.append('loss %.5f rose %.5f > %.3f above best previous epoch %.5f' % (loss, loss - best, rise, best))
        if criteria:
            return dict(reason='COLLAPSE', epoch=epoch, loss=loss, best_previous=best,
                        constant_loss=constant_loss, criteria=criteria)
        best = loss if best is None else min(best, loss)
    return None


def constant_loss(rows):
    """Expected epoch loss of the best constant predictor on the training targets.

    E_prior uses bce_soft with all-one weights, so every batch averages rows per target and
    the epoch mean equals mean_c mean_i BCE(mean_c, t_ic) exactly (float64 here).
    """
    values = []
    for label in s.LABELS:
        t = [float(r[label + '|prior']) for r in rows]
        need(t and all(0 <= x <= 1 for x in t), 'training prior targets in [0,1]')
        p = sum(t) / len(t); need(0 < p < 1, 'nondegenerate training prior mean')
        values.append(sum(-(x * math.log(p) + (1 - x) * math.log(1 - p)) for x in t) / len(t))
    return sum(values) / len(values)


def register():
    reg = expected_registration()
    if (WORK / 'registration.json').exists():
        prior = read(WORK / 'registration.json')
        need(not (OUT / 'admission.json').exists() or prior == reg, 'admitted registration is immutable')
        if prior != reg: require_no_candidate_evidence()
    else:
        require_no_candidate_evidence()
    write(WORK / 'registration.json', reg)
    return reg


def require_no_candidate_evidence():
    names = ('admission.json', 'result.json', 'fit-complete.json', 'holdout.csv', 'train.log',
             'checkpoint.pt', 'ckpt/E_prior_s0_f0.pt', BEST, 'ckpt/best.json', EPOCHS, COLLAPSE,
             'failure.json', 'status.json', 'models/E_prior_s0_f0.pt')
    need(not any((OUT / name).exists() for name in names), 'registration cannot change after candidate admission or output')


def validate(reg):
    need(reg == expected_registration(), 'K1 v3-G registration/policy/source/parent identity changed')
    return reg


def derive_trainer(source):
    """Byte-preserve the v2 derivative outside its optimizer block and identity imports."""
    source = parent.derive_trainer(source)
    start = source.index('            enc = [p for n, p in model.named_parameters()')
    end = source.index('            ema = None\n', start)
    old = source[start:end]
    need(old.count('opt = torch.optim.AdamW(') == 1 and 'weight_decay=a.wd)' in old,
         'parent optimizer block identity')
    source = source[:start] + (
        '            from tools.l1_rco_k1_optim_v3 import training_groups\n'
        '            groups = training_groups(model, a)\n'
        '            opt = torch.optim.AdamW(groups, weight_decay=a.wd)\n') + source[end:]
    for name in ('check_trainer_args', 'check_training_ids'):
        before = 'from tools.l1_rco_k1_v2 import ' + name
        need(source.count(before) == 1, 'parent identity import ' + name)
        source = source.replace(before, 'from tools.l1_rco_k1_v3 import ' + name)
    source = '# K1 v3-G 3.0.0: grouping-only derivative of the audited v2 trainer.\n' + source
    compile(source, 'compact_train_k1v3.py', 'exec')
    return source


def check_training_ids(ids):
    reg = read(WORK / 'registration.json')
    need(sorted(ids) == reg['train'] and not set(ids) & set(reg['holdout']), 'trainer must see only3190 sealed training studies')


def check_trainer_args(a):
    """Fixed v2 recipe plus explicitly sealed optimizer-policy arguments."""
    reg = read(WORK / 'registration.json'); recipe = reg['recipe']
    need(a.lr == recipe['lr'] and a.head_lr == recipe['head_lr'] and a.epochs == recipe['epochs'] and
         a.warm_frac == recipe['warm_frac'] and a.batch == recipe['batch'] and a.train_windows == recipe['windows'] and
         a.crop == recipe['crop'] and a.l1_guard_constant == reg['collapse_guard']['constant_loss'] and
         a.llrd == 1. and a.wd == .02 and a.ema == .999,
         'trainer arguments differ from the registered K1 v3-G recipe')


def trainer_argv(cache, reg):
    recipe = reg['recipe']
    cmd = s.trainer_argv(cache)
    cmd[1] = str(WORK / 'compact_train_l1.py')
    for flag, value in (('--targets', str(WORK / 'targets.csv')), ('--out', str(OUT)), ('--ckpt-every', '1'),
                        ('--epochs', str(recipe['epochs'])), ('--lr', repr(recipe['lr'])), ('--head-lr', repr(recipe['head_lr']))):
        need(cmd.count(flag) == 1, 'trainer argv flag ' + flag)
        cmd[cmd.index(flag)+1] = value
    return cmd + ['--resume', '--warm-frac', repr(recipe['warm_frac']),
                  '--l1-guard-constant', repr(reg['collapse_guard']['constant_loss'])]


def run_command(cache=CACHE, review=REVIEW):
    return [PYTHON, str(ROOT / 'tools/l1_rco_k1_v3.py'), 'run', '--cache', str(Path(cache).resolve()), '--review', str(Path(review).resolve())]


def prepare(reg, cache):
    validate(reg); WORK.mkdir(parents=True, exist_ok=True)
    need(not (OUT / 'admission.json').exists(), 'cannot re-prepare an admitted fit')
    from tools.l1_rco_k1_state_v2 import copy_file_atomic
    path = WORK / 'targets.csv'
    copy_file_atomic(parent.WORK / 'targets.csv', path)
    need(sha(path) == reg['parent_v2']['targets_sha256'], 'training target byte identity')
    original = (ROOT / s.SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    base = parent.derive_trainer(original); derivative = derive_trainer(original)
    need(base == (parent.WORK / 'compact_train_l1.py').read_text(encoding='utf-8'),
         'generated v2 parent differs from audited prepared trainer')
    # The sealed Windows parent uses CRLF; preserve that byte convention too.
    (WORK / 'compact_train_l1.py').write_text(derivative, encoding='utf-8', newline='\r\n')
    (WORK / 'trainer-v2-to-v3.diff').write_text(''.join(difflib.unified_diff(
        base.splitlines(True), derivative.splitlines(True), fromfile='K1-v2.0.0', tofile='K1-v3-G.3.0.0')),
        encoding='utf-8', newline='\n')
    receipt = dict(version=VERSION, contract_id=reg['contract_id'], targets_sha256=sha(path),
                   trainer_sha256=sha(WORK / 'compact_train_l1.py'), train_count=TRAIN_STUDIES,
                   parent_derivative_sha256=reg['parent_v2']['trainer_sha256'],
                   holdout_rows_in_training_artifacts=0, bias='training-only3190 (byte-identical v2 targets)',
                   constant_loss=reg['collapse_guard']['constant_loss'], warm_steps=reg['recipe']['warm_steps'],
                   optimizer_policy=optim.POLICY, argv=trainer_argv(cache, reg), cache=str(Path(cache).resolve()))
    write(WORK / 'prepare.json', receipt)
    plan = [dict(tag=TAG, command=run_command(cache))]
    from tools.gpu_queue import validate_plan
    validate_plan(plan); write(QUEUE, plan)
    return receipt


def review_binding(reg):
    prep = read(WORK / 'prepare.json'); smoke = read(WORK / 'cpu-smoke.json')
    need(prep['contract_id'] == reg['contract_id'] and smoke['contract_id'] == reg['contract_id'], 'CPU receipt contract')
    need(prep['trainer_sha256'] == sha(WORK / 'compact_train_l1.py') and prep['targets_sha256'] == sha(WORK / 'targets.csv'), 'prepared artifact changed')
    need(smoke['status'] == 'PASS' and not smoke['cuda_initialized'], 'real CPU smoke required')
    return dict(contract_id=reg['contract_id'], registration_sha256=sha(WORK / 'registration.json'),
                primary_cohort_id_list_sha256=reg['evaluator']['primary']['id_list_sha256'], prepare_sha256=sha(WORK / 'prepare.json'),
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
    except (ValueError, KeyError, OSError) as exc: blockers.append(str(exc))
    result = dict(status='BLOCKED' if blockers else 'CPU_ADMISSIBLE', blockers=blockers, bindings=binding, cache=cache_proof, dry_run=dry_run)
    if dry_run:
        write(WORK / 'admit-dry-run.json', result); return result
    need(not blockers, '; '.join(blockers))
    from tools.l1_rco_runtime_v120 import active_ml_processes, gpu_snapshot, memory_ok
    need(not active_ml_processes(owner_pid), 'another Python ML GPU process exists')
    snapshot = gpu_snapshot(); k0 = read(K0_RESULT)
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
        need(job.get('command') == command and job.get('command_sha256') == command_digest(command, ROOT) and Path(job['cwd']).resolve() == ROOT, 'gpu_job command identity')
        need(parent.cmdline()[-3:] == ['--_run-job', TAG, job['run_id']] and any(Path(x).name == 'gpu_job.py' for x in parent.cmdline()), 'gpu_job parent identity')
        return parent.pid


def import_trainer():
    spec = importlib.util.spec_from_file_location('l1_k1v3_trainer', WORK / 'compact_train_l1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def require_infer_job(cache, review):
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    command = run_command(cache, review)
    with file_lock(LOGS / '.gpu-claim.lock'):
        job = read_json(LOGS / TAG / 'job.json'); parent = psutil.Process().parent()
        need(parent is not None and job.get('schema') == 'gpu-job/v2' and job.get('tag') == TAG and job.get('phase') == 'running', 'inference requires live B gpu_job')
        need(job.get('child_identity') == identity(parent.pid), 'inference must be an owned run child')
        need(job.get('command') == command and job.get('command_sha256') == command_digest(command, ROOT), 'inference parent command identity')
        need(Path(parent.cmdline()[1]).resolve() == ROOT / 'tools/l1_rco_k1_v3.py' and parent.cmdline()[2] == 'run', 'inference parent must be K1 v2 run')


def completed_fit(reg, out=OUT):
    receipt = read(Path(out) / 'fit-complete.json')
    need(receipt['status'] == 'COMPLETE' and receipt['epochs'] == reg['recipe']['epochs'] and receipt['model_fold'] == 0 and
         receipt['train_studies'] == TRAIN_STUDIES and receipt['contract_id'] == reg['contract_id'] and
         receipt['model_sha256'] == sha(Path(out) / 'models/E_prior_s0_f0.pt'), 'completed fit identity')
    return receipt


def inference(reg, cache, out=OUT, cpu=False, ids=None, limit_windows=None):
    import numpy as np
    import torch
    from tools.l1_rco_k1_state_v1 import Monitor
    ct = import_trainer(); out = Path(out)
    need(cpu or ids is None, 'production inference roster is fixed977')
    ids = list(reg['holdout'] if ids is None else ids)
    need(ids and len(ids) == len(set(ids)) and set(ids) <= set(reg['holdout']), 'holdout inference roster')
    dev = torch.device('cpu' if cpu else 'cuda')
    if cpu: need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU isolation')
    else: completed_fit(reg, out)
    model = ct.Net('convnext_small', str(ROOT / s.INIT), crop=352, in_chans=3, spatial=True, extent=False).to(dev)
    model.load_deploy_state_dict(torch.load(out / 'models/E_prior_s0_f0.pt', map_location='cpu', weights_only=False))
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
        for uid,row in zip(ids,values): writer.writerow([uid]+row.tolist()+['ok',0])
    s.replace(path.with_suffix('.tmp'), path); monitor.phase('complete')
    result = dict(version=VERSION,status='COMPLETE',contract_id=reg['contract_id'],holdout_rows=len(ids),
                  holdout_sha256=sha(path),model_sha256=sha(out / 'models/E_prior_s0_f0.pt'),
                  model_fold=0, eval_chunk=32, clean_oof=False, exposure='EXPOSED_EXPLORATORY',
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
    ckpt = out / 'ckpt/E_prior_s0_f0.pt'
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
        else: child.terminate()
        child.wait(timeout=30)


def supervise(name, command, env, out, constant, poll=15., grace=GUARD['backstop_grace_s'], gpu=True):
    """v1 child loop (stale/WDDM/throughput watchdogs) plus the registered collapse guard."""
    from tools.l1_rco_runtime_v120 import stale_reason, gpu_snapshot
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


def final_training_log(reg, out=OUT):
    """After a train exit 0: the committed log must be complete and never collapsed."""
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
    print(json.dumps(dict(estimated_fit_hours_lower_bound=reg['projection']['fit_hours_training_lower_bound'],
                          estimated_fit_hours_calendar=reg['projection']['fit_hours_calendar_estimate'], scope=reg['projection']['scope'])), flush=True)
    commands = [trainer_argv(cache, reg), [PYTHON,str(ROOT / 'tools/l1_rco_k1_v3.py'),'infer','--cache',str(cache),'--review',str(review)]]
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '0'; env['CT_PREDICT_CHUNK'] = '32'
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT / s.CODE)])
    for name,command in zip(('train','infer'),commands):
        if name == 'train' and (OUT / 'fit-complete.json').exists():
            completed_fit(reg); final_training_log(reg)
            continue
        supervise(name, command, env, OUT, reg['collapse_guard']['constant_loss'])
        if name == 'train': final_training_log(reg)
    return read(OUT/'result.json')


def evaluate(reg, dry_run=False):
    """Unchanged evaluator v1.1.0 (registered pins/rule), pointed at the v2 work dir."""
    from tools import l1_rco_k1_eval_v1 as ev
    previous = ev.WORK; ev.WORK = WORK
    try: return ev.evaluate(reg, dry_run=dry_run)
    finally: ev.WORK = previous


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','validate','prepare','admit','cpu-smoke','run','infer','evaluate'])
    ap.add_argument('--cache',type=Path,default=CACHE); ap.add_argument('--review',type=Path,default=REVIEW)
    ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    if a.action not in ('run','infer'): os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        if a.action=='register':
            reg=register(); result=dict(status='PASS',action=a.action,contract_id=reg['contract_id'],projection=reg['projection'],
                                        constant_loss=reg['collapse_guard']['constant_loss'],warm_steps=reg['recipe']['warm_steps'])
        elif a.action=='run': result=run(a.cache,a.review)
        else:
            reg=validate(read(WORK/'registration.json'))
            if a.action=='validate': result=dict(status='PASS',action=a.action,train=3190,holdout=977,contract_id=reg['contract_id'])
            elif a.action=='prepare': result=prepare(reg,a.cache)
            elif a.action=='admit':
                need(a.dry_run,'real admission only through B gpu_queue'); result=admit(reg,a.cache,a.review)
            elif a.action=='infer':
                require_infer_job(a.cache,a.review); result=inference(reg,a.cache)
            elif a.action=='evaluate':
                full=evaluate(reg,dry_run=a.dry_run)
                result={k:full.get(k) for k in ('decision','final_decision_eligible','cohort_n','inference_n','r7_valid','dry_run','contract_id')}
            else:
                from tools.l1_rco_k1_smoke_v3 import cpu_smoke
                result=cpu_smoke(reg,a.cache)
    except CollapseStop as exc:
        print(json.dumps(dict(status='FAILED',reason='COLLAPSE',detail=str(exc)),allow_nan=False),flush=True)
        raise SystemExit(GUARD['exit_code'])
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
