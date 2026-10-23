"""H210 same-contract recovery v1.0.0; additive, original artifacts read-only.

Changelog: copy the verified 10400-step pause into a fresh attempt; reuse seed1
trainer/admission/supervision/export/inference and genuine seed0 loader telemetry.
Only --out changes in the original trainer argv. No automatic retry or GPU probe
in CPU actions. B supplies the independent recovery review and launches gpu_job.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import l1_rco_k1_v3s1 as k
from tools import l1_rco_k1_recovery_v1 as replay

VERSION = '1.0.0'
TAG = 'L1_RCO_K1_F0_V3G_S1_RECOVERY_V1'
WORK = ROOT / '.local/scratch/astra-b2/h210-recovery-v1'
OUT = WORK / 'attempt/fit'
MANIFEST = WORK / 'RECOVERY.json'
REVIEW = WORK / 'REVIEW.json'
PAUSE = ROOT / '.local/scratch/slot-b/h210-pause-roi-20261019'
BACKUP = PAUSE / 'E_prior_s0_f0.pause.pt'
CHECKPOINT = 'ckpt/E_prior_s0_f0.pt'
CONTRACT = '15cc54198844a7b503b0d9da87e7d7c678f74072217e5d0c051c08302e3c3fdc'
RUN_ID = 'fad1c286d91c4e108a72e0522bdbca7a'
PAUSE_SHA = '6f7a289ebe797813e48e09f3ce3a8efb858ddc4092394fcfb2672f1fbe040a09'
STATE_ID = '7acc6e536cc22ea3d3b4c14f1afdfcfd368cd6bb6890f2583e9aa8bbf21599db'
STEP, EPOCH, BATCH, TOTAL = 10400, 6, 830, 15950
need, read, write, sha = k.need, k.read, k.write, k.sha


def command(action='run'):
    return [k.PYTHON, str(Path(__file__).resolve()), action]


def launch_command():
    return [k.PYTHON, str(ROOT / 'tools/gpu_job.py'), '--tag', TAG, '--', *command()]


def trainer_argv(reg):
    argv = k.trainer_argv(k.CACHE, reg)
    need(argv.count('--out') == 1, 'one output flag required')
    argv[argv.index('--out') + 1] = str(OUT)
    return argv


def original():
    """Validate the sealed recipe, pause evidence and terminal owner without writes."""
    from tools.job_runtime import LOGS, command_digest, liveness
    reg = k.validate(read(k.WORK / 'registration.json'))
    need(reg['contract_id'] == CONTRACT and reg['recipe']['seed_base'] == 1,
         'not the original seed1 contract')
    need(len(reg['train']) == 3190 and len(reg['holdout']) == 977, 'sealed cohorts')
    verified, stopped = read(PAUSE / 'PAUSE-VERIFIED.json'), read(PAUSE / 'STOPPED.json')
    ck = verified['checkpoint']
    need(stopped['state'] == 'USER_PAUSED_FOR_ROI' and stopped['trainer_stopped']
         and stopped['checkpoint'] == ck, 'intentional verified pause required')
    need((ck['step_total'], ck['epoch'], ck['batch'], ck['identity'], ck['sha256']) ==
         (STEP, EPOCH, BATCH, STATE_ID, PAUSE_SHA), 'pause position/hash identity')
    need(Path(ck['path']).resolve() == (k.OUT / CHECKPOINT).resolve()
         and Path(ck['backup']).resolve() == BACKUP.resolve(), 'pause paths')
    need(sha(BACKUP) == sha(k.OUT / CHECKPOINT) == PAUSE_SHA, 'paused checkpoint/backup changed')
    job, receipt = read(LOGS / k.TAG / 'job.json'), read(LOGS / k.TAG / 'receipt.json')
    need(job['run_id'] == receipt['run_id'] == RUN_ID and receipt['state'] == 'FAILED'
         and receipt['tag'] == job['tag'] == k.TAG
         and receipt['command_sha256'] == job['command_sha256'] == command_digest(k.run_command(), ROOT),
         'original terminal run binding')
    need(liveness(job) == 'dead' and liveness(job.get('child_identity')) == 'dead',
         'original process still alive')
    need(not (k.OUT / k.COLLAPSE).exists(), 'collapsed run cannot recover')
    admission = read(k.OUT / 'admission.json')
    need(admission['contract_id'] == CONTRACT and admission['review_binding'] == k.review_binding(reg),
         'original admission changed')
    best = read(k.OUT / 'ckpt/best.json')
    need(best['sha256'] == sha(k.OUT / k.BEST) and best['step_total'] <= STEP, 'best checkpoint changed')
    return reg


def expected_manifest(reg):
    from tools.job_runtime import LOGS
    paths = [Path(__file__), Path(replay.__file__), PAUSE / 'PAUSE-VERIFIED.json', PAUSE / 'STOPPED.json',
             k.WORK / 'registration.json', k.WORK / 'prepare.json', k.WORK / 'compact_train_l1.py',
             k.REVIEW, k.OUT / 'admission.json', k.OUT / 'parameter-groups.json',
             k.OUT / 'ckpt/best.json', LOGS / k.TAG / 'job.json', LOGS / k.TAG / 'receipt.json']
    return dict(version=VERSION, tag=TAG, contract_id=CONTRACT, original_run_id=RUN_ID,
                backup_sha256=PAUSE_SHA, restored_step=STEP, total_steps=TOTAL,
                replay_uncommitted_steps=33, command=command(), trainer_argv=trainer_argv(reg),
                out=str(OUT), pins={str(p.relative_to(ROOT)): sha(p) for p in paths})


def register():
    reg = original(); expected = expected_manifest(reg)
    if MANIFEST.exists():
        need(read(MANIFEST) == expected, 'registered recovery is immutable; use a new version')
    else:
        need(not OUT.exists(), 'unregistered recovery destination already exists')
        write(MANIFEST, expected)
    return dict(status='PASS', contract_id=CONTRACT, manifest_sha256=sha(MANIFEST))


def validate():
    reg = original()
    need(read(MANIFEST) == expected_manifest(reg), 'recovery source/evidence/command changed')
    return reg


def prepare():
    """Idempotent byte copies only before launch; never replace progressed state."""
    validate()
    need(not (OUT / 'admission.json').exists(), 'attempt already admitted; cannot prepare again')
    from tools.l1_rco_k1_state_v2 import copy_file_atomic
    for src, relative in [(BACKUP, CHECKPOINT), (k.OUT / k.BEST, k.BEST),
                          (k.OUT / 'ckpt/best.json', 'ckpt/best.json'),
                          (k.OUT / 'parameter-groups.json', 'parameter-groups.json')]:
        dst = OUT / relative
        if dst.exists(): need(sha(dst) == sha(src), 'prepared output changed; do not overwrite')
        else: need(copy_file_atomic(src, dst) == sha(src), 'copy hash mismatch')
    result = dict(status='PASS', manifest_sha256=sha(MANIFEST), checkpoint_sha256=sha(OUT / CHECKPOINT))
    write(WORK / 'prepare.json', result)
    return result


def review_binding():
    return dict(manifest_sha256=sha(MANIFEST), cpu_check_sha256=sha(WORK / 'cpu-check.json'),
                prepare_sha256=sha(WORK / 'prepare.json'))


def require_review():
    r = read(REVIEW)
    need(r['verdict'] == 'PASS' and r['builder'] == 'astra-b2'
         and r['reviewer'] not in ('', 'astra-b2') and r['bindings'] == review_binding(),
         'independent recovery review required')
    report = (ROOT / r['report']).resolve()
    need(report.is_relative_to(ROOT) and sha(report) == r['report_sha256'], 'recovery review report changed')


def admit(dry_run=True, owner_pid=None):
    reg = validate()
    # Reuse the entire original admission, redirecting its sole dry-run receipt.
    prior_write = k.write
    if dry_run:
        def local_write(path, value):
            need(Path(path) == k.WORK / 'admit-dry-run.json', 'unexpected original write')
            write(WORK / 'original-admit-dry-run.json', value)
        k.write = local_write
    try:
        admission = k.admit(reg, k.CACHE, k.REVIEW, dry_run, owner_pid)
    finally:
        k.write = prior_write
    if dry_run:
        need(admission['status'] == 'CPU_ADMISSIBLE', 'original admission blocked: ' + str(admission['blockers']))
    else:
        old = read(k.OUT / 'admission.json')
        need(all(old[n] == admission[n] for n in ('contract_id', 'review_binding', 'cache')), 'same admission identity')
    need(read(WORK / 'prepare.json')['manifest_sha256'] == sha(MANIFEST), 'prepare binding')
    cpu = read(WORK / 'cpu-check.json')
    need(cpu['status'] == 'PASS' and cpu['manifest_sha256'] == sha(MANIFEST)
         and not cpu['cuda_initialized'], 'CPU restoration check required')
    if dry_run:
        try: require_review(); review_status = 'PASS'
        except (OSError, ValueError, KeyError): review_status = 'PENDING'
        result = dict(status='CPU_ADMISSIBLE', recovery_review=review_status,
                      launch_ready=review_status == 'PASS', contract_id=CONTRACT)
        write(WORK / 'admit-dry-run.json', result)
        return result
    require_review()
    return admission


def owner(child=False):
    """Same seed0 recovery ownership checks, under a distinct attempt tag."""
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    with file_lock(LOGS / '.gpu-claim.lock'):
        job = read_json(LOGS / TAG / 'job.json'); parent = psutil.Process().parent()
        need(parent is not None and job.get('schema') == 'gpu-job/v2' and job.get('tag') == TAG
             and job.get('phase') == 'running', 'recovery requires actual gpu_job')
        need(job.get('command') == command() and job.get('command_sha256') == command_digest(command(), ROOT)
             and Path(job['cwd']).resolve() == ROOT, 'recovery command binding')
        if child:
            need(job.get('child_identity') == identity(parent.pid) and parent.cmdline() == command(), 'owned recovery child')
            wrapper = parent.parent()
        else:
            need(job.get('child_identity') == identity(), 'recovery run PID identity'); wrapper = parent
        need(wrapper is not None and {n: job.get(n) for n in ('pid', 'create_time')} == identity(wrapper.pid), 'wrapper identity')
        need(wrapper.cmdline()[-3:] == ['--_run-job', TAG, job['run_id']]
             and any(Path(x).name == 'gpu_job.py' for x in wrapper.cmdline()), 'wrapper command')
        return wrapper.pid


@contextlib.contextmanager
def replay_adapter():
    from tools.l1_rco_k1_state_v2 import TrainerState
    previous = TrainerState.batches
    TrainerState.batches = replay.replay_batches
    try: yield
    finally: TrainerState.batches = previous


def run():
    owner_pid = owner()
    need(not (OUT / 'admission.json').exists(), 'one recovery attempt only; no automatic retry')
    need(sha(OUT / CHECKPOINT) == PAUSE_SHA, 'initial recovery checkpoint changed')
    admission = admit(False, owner_pid)
    write(OUT / 'admission.json', admission)
    reg = validate()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CT_PREDICT_CHUNK='32',
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / k.s.CODE)]))
    for action in ('train', 'infer'):
        k.supervise(action, command(action), env, OUT, reg['collapse_guard']['constant_loss'])
        if action == 'train':
            k.final_training_log(reg, OUT)
            need(k.completed_fit(reg, OUT)['optimizer_steps'] == TOTAL, 'final optimizer step count')
    return read(OUT / 'result.json')


def cpu_check():
    """Restore the real checkpoint into the real CPU Net/AdamW/LambdaLR/EMA.

    Check exact state before a bounded real-cache backward/optimizer step. Replay
    all 830 sampler batches cheaply; full GPU image decoding remains unchanged.
    CPU smoke uses two windows/study, not a production throughput/parity claim.
    CUDA RNG bytes are compared but cannot be installed without CUDA.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import csv
    import gc
    import math
    import torch
    from types import SimpleNamespace
    from tools.l1_rco_k1_state_v2 import Monitor, atomic_torch_save
    from tools.l1_rco_k1_smoke_v3s1 import equal_state
    reg = validate(); torch.set_num_threads(2)
    need(not torch.cuda.is_initialized(), 'CPU isolation')
    need(sha(OUT / CHECKPOINT) == PAUSE_SHA, 'prepared checkpoint differs from backup')
    smoke = WORK / 'cpu'; smoke.mkdir(parents=True, exist_ok=True)
    ct = k.import_trainer()
    model = ct.Net('convnext_small', str(ROOT / k.s.INIT), crop=352, in_chans=3, spatial=True, extent=False)
    model.to(memory_format=torch.channels_last)
    groups, _ = k.optim.parameter_groups(model)
    opt = torch.optim.AdamW(groups, weight_decay=.02)
    warm = reg['recipe']['warm_steps']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
        .5 * (1 + math.cos(math.pi * (s - warm) / max(1, TOTAL - warm))))
    ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    gen = torch.Generator().manual_seed(5017)
    dl = torch.utils.data.DataLoader(range(3190), batch_size=2, shuffle=True, drop_last=True,
                                    generator=torch.Generator().manual_seed(9017), num_workers=0)
    monitor = Monitor(smoke, cpu=True)
    monitor.admission = read(k.OUT / 'admission.json')
    # TrainerState must load the prepared copy but all monitor/sidecar writes stay in CPU scratch.
    # V1 constructor performs the exact restoration; V2-only sidecars are exercised by tests.
    from tools.l1_rco_k1_state_v1 import TrainerState as RestoreState
    state = RestoreState(OUT, model, opt, sched, ema, gen, dl, reg['train'], 10, monitor)
    backup = torch.load(BACKUP, map_location='cpu', weights_only=False)
    need((state.identity, state.start_ep, state.batch, state.step_total) == (STATE_ID, EPOCH, BATCH, STEP), 'restored counters')
    for name, actual in dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ema=ema,
                             epoch_loader=dl.generator.get_state(), rng=state.saved_rng).items():
        need(equal_state(actual, backup[name]), 'restored ' + name + ' differs from backup')
    need(torch.equal(gen.get_state(), backup['rng']['gen']) and
         torch.equal(torch.get_rng_state(), backup['rng']['cpu']), 'restored CPU/augmentation RNG')
    need(state.log == backup['log'] and state.tot == backup['tot'] and state.n == backup['n'], 'partial epoch totals')
    del backup; gc.collect()
    state.begin_epoch(EPOCH, SimpleNamespace())
    reference = torch.utils.data.DataLoader(range(3190), batch_size=2, shuffle=True, drop_last=True,
        generator=torch.Generator().set_state(state.epoch_loader), num_workers=0)
    expected_iterator = iter(reference)
    for _ in range(BATCH): next(expected_iterator)
    expected_indices = next(expected_iterator).tolist()
    iterator = replay.replay_batches(state)
    indices = next(iterator).tolist()
    need(indices == expected_indices and state.step_total == STEP and monitor.steps == 0,
         'loader replay changed sampler/optimizer position')
    # Use the full original index list so per-index deterministic data RNG is unchanged.
    ds = ct.Studies(str(k.CACHE), sorted(reg['train']), 2, 'train', 9017,
                    train_sampling='cover', slices=3, phys_mm=3.5, slice_jitter=.25)
    ds.epoch = EPOCH
    img, meta, mask, _ = ct.collate([ds[i] for i in indices])
    with (k.WORK / 'targets.csv').open(encoding='utf-8', newline='') as stream:
        rows = {row[k.s.UID]: row for row in csv.DictReader(stream)}
    target = torch.tensor([[float(rows[ds.uids[i]][label + '|prior']) for label in k.s.LABELS] for i in indices])
    model.train(); model.enc.set_grad_checkpointing(True)
    logits = model(img[..., 16:368, 16:368], meta, mask)
    loss = ct.bce_soft(logits, target, torch.ones_like(target)); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    need(bool(torch.isfinite(loss)) and bool(torch.isfinite(norm)) and norm > 0, 'CPU backward failed')
    opt.step(); sched.step()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ema: ema[name].mul_(.999).add_(p, alpha=.001)
    need(sched.last_epoch == STEP + 1 and all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'CPU optimizer failed')
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ema: p.copy_(ema[name])
    atomic_torch_save(model.deploy_state_dict(), smoke / 'models/E_prior_s0_f0.pt')
    del state, iterator, model, opt, sched, ema, logits, loss, img, ds; gc.collect()
    served = k.inference(reg, k.CACHE, out=smoke, cpu=True, ids=reg['holdout'][:1], limit_windows=2)
    need(served['holdout_rows'] == 1 and not torch.cuda.is_initialized(), 'CPU export/inference failed')
    result = dict(status='PASS', version=VERSION, manifest_sha256=sha(MANIFEST), contract_id=CONTRACT,
                  checkpoint_sha256=PAUSE_SHA, restored_step=STEP, epoch=EPOCH, batch=BATCH,
                  full_state_matches_backup=True, cuda_rng_bytes_preserved=True, cuda_rng_installed=False,
                  sampler_replayed_batches=BATCH, uncommitted_steps_to_retrain=33,
                  real_cpu_optimizer_step=STEP + 1, real_train_studies=2, windows_per_study=2,
                  finite_backward=True, ema_export=True, real_holdout_rows=1, cuda_initialized=False,
                  scope='CPU restoration + bounded real-cache training/export/inference; no GPU qualification')
    write(WORK / 'cpu-check.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'validate', 'prepare', 'cpu-check', 'admit', 'run', 'train', 'infer'])
    action = parser.parse_args().action
    if action not in ('run', 'train', 'infer'): os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if action in ('train', 'infer'):
        owner(child=True); reg = validate(); require_review()
        if action == 'train':
            need(sha(OUT / CHECKPOINT) == PAUSE_SHA, 'train must start at verified pause')
            with replay_adapter():
                argv = trainer_argv(reg); sys.argv = argv[1:]
                runpy.run_path(argv[1], run_name='__main__')
            result = dict(status='TRAIN_COMPLETE')
        else: result = k.inference(reg, k.CACHE, out=OUT)
    elif action == 'validate':
        validate(); result = dict(status='PASS', contract_id=CONTRACT)
    else:
        result = {'register': register, 'prepare': prepare, 'cpu-check': cpu_check,
                  'admit': admit, 'run': run}[action]()
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
