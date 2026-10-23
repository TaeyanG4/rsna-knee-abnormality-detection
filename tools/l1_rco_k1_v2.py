"""H-B-182 K1 v2.0.0 (tag L1_RCO_K1_F0_V2): rerun of the collapsed K1 v1 fold-0 RCO fit.

Changelog v2.0.0 (registered before any v2 output; K1 v1 collapsed at epoch 3, D-B-214):
encoder LR 3e-4 -> 1e-4 and head LR 1e-3 -> 3e-4 (trainer --lr/--head-lr); warm-up 5 %
(0.8 epoch) -> --warm-frac 0.1 = exactly 1 epoch (1595 of 15950 optimizer steps; the
derivative's --warm-frac default .05 keeps the parent formula); epochs 16 -> 10; registered
collapse guard (collapse_reason below: the trainer stops itself after the committed epoch
checkpoint, the launcher marks FAILED reason COLLAPSE, keeps every checkpoint, runs no
inference and backstops a trainer that does not stop); best-loss epoch checkpoint copy.
New work/output dir, queue and tag. Unchanged: cache, fold-0 roster/holdout977, train-only
targets and bias, evaluator v1.1.0 and its exact976 KEEP rule, backbone/init/crop/batch/
windows/EMA/wd/bf16 autocast, checkpoint cadence, watchdogs and VRAM gate. The v1 fit dir
.local/scratch/l1-rco/k1/fit is never written (evidence). Base: K1 v1.0.2 (l1_rco_k1_v1.py),
whose trainer patches are reused unchanged by derive_trainer.
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

VERSION = '2.0.0'
WORK = ROOT / '.local/scratch/l1-rco/k1v2'
OUT = WORK / 'fit'
CACHE = Path('G:/temp_data/rsna-knee/l1rco-cache-full')
FULL_REG = ROOT / '.local/scratch/l1-rco/fullcache/registration.json'
QUEUE = ROOT / '.local/scratch/slot-b/l1rco/k1v2-gpu-queue.json'
K0_RESULT = ROOT / '.local/scratch/l1-rco/k0-v2/k0/result.json'
K0_REVIEW = ROOT / '.local/scratch/slot-b/l1rco/k0v2-fullcache-review-out.md'
REVIEW = WORK / 'K1V2-REVIEW.json'
V1_REG = v1.WORK / 'registration.json'
V1_DIAG = ROOT / '.local/scratch/slot-b/k1-diag/verdict.json'
V1_OBSERVED_WPS = 40.925715706088475  # v1 fit/status.json at stop: 1788800 windows / 43708.46 train s (I/O-bound full cache)
PYTHON = s.PYTHON
TAG = 'L1_RCO_K1_F0_V2'
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
    names = ['l1_rco_k1_v2.py', 'l1_rco_k1_state_v2.py', 'l1_rco_k1_smoke_v2.py',
             'l1_rco_k1_v1.py', 'l1_rco_k1_state_v1.py', 'l1_rco_k1_eval_v1.py',
             'l1_rco_stage0_v1.py', 'l1_rco_stage0_v120.py', 'l1_rco_runtime_v120.py']
    return {'tools/' + n: sha(ROOT / 'tools' / n) for n in names}


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
    from tools.l1_rco_k1_eval_v1 import evaluator_registration
    s.verify_pins()
    full = read(ROOT / s.REG); roster = s.validate_roster(full)
    cache_reg = read(FULL_REG)
    need(cache_reg['roster'] == roster and set(cache_reg['train']) == set(full['train']) and
         set(cache_reg['holdout']) == set(full['val']), 'fullcache roster mismatch')
    k0 = read(K0_RESULT)
    need(k0['K0_numeric_gate'] == 'PASS', 'K0 numeric gate must pass')
    from tools.l1_rco_runtime_v120 import go_rule
    need(go_rule(k0), 'K0 measured numeric gate reproduction failed')
    train, holdout = sorted(full['train']), sorted(full['val'])
    need(len(train) == TRAIN_STUDIES, 'K1 v2 training roster size')
    with (ROOT / s.TARGETS).open(newline='', encoding='utf-8') as stream:
        rows = training_rows(dict(train=train, holdout=holdout), csv.DictReader(stream))
    constant = constant_loss(rows)
    epochs, warm_frac = CHANGES['epochs'][1], CHANGES['warm_frac'][1]
    total = STEPS_PER_EPOCH * epochs; warm = max(1, int(warm_frac * total))
    need(warm >= STEPS_PER_EPOCH, 'registered warm-up must be >= 1 epoch')
    old = read(V1_REG)
    reg = dict(version=VERSION, tag=TAG, train=train, holdout=holdout,
               roster=roster, fullcache_contract_id=cache_reg['contract_id'],
               fullcache_registration_sha256=sha(FULL_REG), pins=s.PINS, code_pins=code_pins(),
               evaluator=evaluator_registration(holdout), exposure='EXPOSED_EXPLORATORY', clean_oof=False,
               recipe=dict(model_fold=0, folds=4, only_folds='0', holdout_fold=None, seeds=1,
                           epochs=epochs, batch=BATCH, windows=64, crop=352, eval_stride=3, eval_chunk=32,
                           bias='logit(mean(train3190_only)).clamp(.001,.999)', checkpoint_steps=100,
                           lr=CHANGES['lr'][1], head_lr=CHANGES['head_lr'][1], warm_frac=warm_frac,
                           steps_per_epoch=STEPS_PER_EPOCH, total_steps=total, warm_steps=warm,
                           warm_epochs=warm / STEPS_PER_EPOCH, schedule='linear warm-up then cosine to 0 (parent LambdaLR)',
                           precision='bf16 autocast, unchanged from v1 (no fp32-head change registered)',
                           unchanged='convnext_small init, E_prior targets, crop352, batch2, 64 windows, 3 slices, cover sampling, '
                                     'spatial attention, EMA .999, wd .02, clip 5, grad-ckpt, seed0/fold0, cache, roster'),
               changes_vs_v1={name: dict(v1=a, v2=b) for name, (a, b) in CHANGES.items()},
               collapse_guard=dict(GUARD, constant_loss=constant, threshold=constant - GUARD['margin'],
                                   epoch_index='0-based trainer epoch (printed "epoch N")',
                                   rule='COLLAPSE at first epoch e with (e >= 2 and loss_e >= constant_loss - 0.002) or '
                                        '(e >= 1 and loss_e - min(previous epoch losses) > 0.015); committed 5-decimal epoch log',
                                   constant_definition='mean over 12 targets of mean BCE(mean(prior), prior) over the 3190 training rows; '
                                                       'exact expected epoch loss of the best constant predictor (all-one E_prior weights)',
                                   action='trainer exits 4 after the committed epoch-end checkpoint and best copy and writes fit/collapse.json; '
                                          'launcher writes fit/failure.json status FAILED reason COLLAPSE, keeps all checkpoints, '
                                          'runs no inference, refuses relaunch; backstop kill if the trainer is still alive 300 s after '
                                          'fit/ckpt/epochs.json shows the collapse'),
               best_checkpoint=dict(path='fit/' + BEST, meta='fit/ckpt/best.json',
                                    rule='byte copy of the committed epoch-end rolling checkpoint whenever that epoch has the strictly '
                                         'lowest committed epoch loss (first minimum); full state incl. EMA; resume repairs a missed copy'),
               parent=dict(tag=v1.TAG, version=old['version'], contract_id=old['contract_id'], registration_sha256=sha(V1_REG),
                           outcome='COLLAPSED at epoch 3 (D-B-214): loss 0.5050 -> 0.5303 = constant-prediction loss; fit dir kept as evidence',
                           diagnostic=str(V1_DIAG.relative_to(ROOT)).replace('\\', '/'), diagnostic_sha256=sha(V1_DIAG)),
               K0_result_sha256=sha(K0_RESULT), K0_review_sha256=sha(K0_REVIEW),
               projection=dict(fit_hours_training_lower_bound=TRAIN_STUDIES*64*epochs/k0['windows_per_s']/3600,
                               measured_windows_per_s=k0['windows_per_s'],
                               v1_observed_windows_per_s=V1_OBSERVED_WPS,
                               fit_hours_calendar_estimate=TRAIN_STUDIES*64*epochs/V1_OBSERVED_WPS/3600,
                               scope='lower bound = K0 loader-inclusive rate; calendar estimate = v1 measured full-cache rate '
                                     '(I/O-bound, ~1.39 h/epoch); checkpoints and final holdout inference extra; not a deadline or timeout'))
    reg['contract_id'] = digest(reg)
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
    need(reg['version'] == VERSION and reg['tag'] == TAG and reg['contract_id'] == digest({k:v for k,v in reg.items() if k != 'contract_id'}), 'K1 v2 registration digest')
    need(reg['code_pins'] == code_pins() and reg['pins'] == s.PINS, 'K1 v2 source changed; re-register before admission')
    s.verify_pins()
    full = read(ROOT / s.REG)
    need(reg['roster'] == s.validate_roster(full) and reg['train'] == sorted(full['train']) and reg['holdout'] == sorted(full['val']), 'K1 sealed full roster')
    need(sha(FULL_REG) == reg['fullcache_registration_sha256'], 'fullcache registration changed')
    need(sha(K0_RESULT) == reg['K0_result_sha256'], 'registered K0 result changed')
    need(sha(K0_REVIEW) == reg['K0_review_sha256'], 'registered K0 review changed')
    from tools.l1_rco_k1_eval_v1 import evaluator_registration
    need(reg['evaluator'] == evaluator_registration(reg['holdout']), 'evaluator registration changed')
    recipe = reg['recipe']
    need(all(recipe[name] == new for name, (_, new) in CHANGES.items()) and recipe['batch'] == BATCH and
         recipe['warm_steps'] == max(1, int(recipe['warm_frac'] * recipe['total_steps'])) >= recipe['steps_per_epoch'] == STEPS_PER_EPOCH,
         'K1 v2 registered recipe')
    need(reg['collapse_guard']['constant_loss'] > 0 and all(reg['collapse_guard'][k] == v for k, v in GUARD.items()), 'K1 v2 collapse guard registration')
    return reg


def derive_trainer(source):
    """K1 v1.0.2 derivative (reused unchanged) plus exactly the registered v2 patches."""
    source = v1.derive_trainer(source)
    def patch(before, after):
        nonlocal source
        need(source.count(before) == 1, 'K1 v2 exact trainer patch occurrence: ' + before[:90])
        source = source.replace(before, after)
    patch('    ap.add_argument("--l1-stop-steps", type=int, default=0)\n', '''    ap.add_argument("--l1-stop-steps", type=int, default=0)
    ap.add_argument("--l1-guard-constant", type=float, default=None)
    ap.add_argument("--warm-frac", type=float, default=.05, help="K1 v2: linear warm-up fraction of all optimizer steps (default .05 = parent)")
''')
    patch('    from tools.l1_rco_k1_state_v1 import Monitor, TrainerState\n', '    from tools.l1_rco_k1_state_v2 import Monitor, TrainerState\n')
    patch('    else:\n        assert not a.l1_stop_steps\n', '''    else:
        assert not a.l1_stop_steps and a.l1_guard_constant is not None
        from tools.l1_rco_k1_v2 import check_trainer_args
        check_trainer_args(a)
''')
    patch('        from tools.l1_rco_k1_v1 import check_training_ids\n', '        from tools.l1_rco_k1_v2 import check_training_ids\n')
    patch('            warm = max(1, int(.05 * total))\n', '''            warm = max(1, int(a.warm_frac * total))
            print("warm-up steps %d of %d (%.2f epochs, warm-frac %r)" % (warm, total, warm / max(1, len(dl)), a.warm_frac), flush=True)
            assert a.l1_cpu_test or warm >= len(dl)  # K1 v2 registered warm-up >= 1 epoch
''')
    patch('                                    resume=a.resume, every=100, stop_steps=a.l1_stop_steps)\n', '''                                    resume=a.resume, every=100, stop_steps=a.l1_stop_steps,
                                    guard_constant=a.l1_guard_constant)
''')
    source = ('# K1 v2.0.0 derivative generated by tools/l1_rco_k1_v2.py: 14767-parent + K1 v1.0.2 patches + --warm-frac\n'
              '# (default .05 = parent), registered LR/epoch/guard arguments checked, collapse guard + best-epoch copy (state v2).\n') + source
    compile(source, 'compact_train_k1v2.py', 'exec')
    return source


def check_training_ids(ids):
    reg = read(WORK / 'registration.json')
    need(sorted(ids) == reg['train'] and not set(ids) & set(reg['holdout']), 'trainer must see only3190 sealed training studies')


def check_trainer_args(a):
    """Production trainer process: registered v2 recipe arguments, nothing inferred."""
    reg = read(WORK / 'registration.json'); recipe = reg['recipe']
    need(a.lr == recipe['lr'] and a.head_lr == recipe['head_lr'] and a.epochs == recipe['epochs'] and
         a.warm_frac == recipe['warm_frac'] and a.batch == recipe['batch'] and a.train_windows == recipe['windows'] and
         a.crop == recipe['crop'] and a.l1_guard_constant == reg['collapse_guard']['constant_loss'],
         'trainer arguments differ from the registered K1 v2 recipe')


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
    return [PYTHON, str(ROOT / 'tools/l1_rco_k1_v2.py'), 'run', '--cache', str(Path(cache).resolve()), '--review', str(Path(review).resolve())]


def prepare(reg, cache):
    validate(reg); WORK.mkdir(parents=True, exist_ok=True)
    need(not (OUT / 'admission.json').exists(), 'cannot re-prepare an admitted fit')
    with (ROOT / s.TARGETS).open(newline='', encoding='utf-8') as stream:
        rows = training_rows(reg, csv.DictReader(stream))
    need(constant_loss(rows) == reg['collapse_guard']['constant_loss'], 'registered constant-prediction loss differs from prepared targets')
    path = WORK / 'targets.csv'
    with path.with_suffix('.tmp').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    s.replace(path.with_suffix('.tmp'), path)
    original = (ROOT / s.SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    base = v1.derive_trainer(original); derivative = derive_trainer(original)
    (WORK / 'compact_train_l1.py').write_text(derivative, encoding='utf-8')
    (WORK / 'trainer.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), derivative.splitlines(True), fromfile='14767-parent', tofile='K1-v2.0.0')), encoding='utf-8')
    (WORK / 'trainer-v1-to-v2.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True), derivative.splitlines(True), fromfile='K1-v1.0.2-derivative', tofile='K1-v2.0.0')), encoding='utf-8')
    receipt = dict(version=VERSION, contract_id=reg['contract_id'], targets_sha256=sha(path),
                   trainer_sha256=sha(WORK / 'compact_train_l1.py'), train_count=len(rows),
                   v1_derivative_sha256=hashlib.sha256(base.encode('utf-8')).hexdigest(),
                   holdout_rows_in_training_artifacts=0, bias='training-only3190',
                   constant_loss=reg['collapse_guard']['constant_loss'], warm_steps=reg['recipe']['warm_steps'],
                   argv=trainer_argv(cache, reg), cache=str(Path(cache).resolve()))
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
    spec = importlib.util.spec_from_file_location('l1_k1v2_trainer', WORK / 'compact_train_l1.py')
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
        need(Path(parent.cmdline()[1]).resolve() == ROOT / 'tools/l1_rco_k1_v2.py' and parent.cmdline()[2] == 'run', 'inference parent must be K1 v2 run')


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
    commands = [trainer_argv(cache, reg), [PYTHON,str(ROOT / 'tools/l1_rco_k1_v2.py'),'infer','--cache',str(cache),'--review',str(review)]]
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
                from tools.l1_rco_k1_smoke_v2 import cpu_smoke
                result=cpu_smoke(reg,a.cache)
    except CollapseStop as exc:
        print(json.dumps(dict(status='FAILED',reason='COLLAPSE',detail=str(exc)),allow_nan=False),flush=True)
        raise SystemExit(GUARD['exit_code'])
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
