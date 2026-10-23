"""K1 v3 recovery v1.0.0: report genuine loader replay progress after reboot.

Original trainer/state/source/registration remain immutable. This explicit
entrypoint uses the same gpu_job claim, source/cache/review/memory admission,
trainer arguments, full-state restore, watchdogs, export and inference. The
trainer child installs only the documented TrainerState.batches telemetry
adapter before running the original trainer. No fabricated heartbeat thread.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import l1_rco_k1_v3 as v3

VERSION = '1.0.0'
WORK = ROOT / '.local/scratch/slot-b/research-resume-1018/recovery-loader-v1'
MANIFEST = WORK / 'RECOVERY.json'
need, read, write, sha = v3.need, v3.read, v3.write, v3.sha


def command(action='run'):
    return [v3.PYTHON, str(Path(__file__).resolve()), action]


def replay_batches(self):
    """Identical iterator consumption/RNG restore; only actual batch progress emits."""
    self.dl.generator.set_state(self.epoch_loader)
    iterator = iter(self.dl)
    if self.batch:
        self.monitor.phase('resume-loader')
        for index in range(self.batch):
            next(iterator)  # same decode and shuffle replay; never optimizer.step
            self.monitor.emit()
            status = read(self.monitor.out / 'status.json')
            status.update(resume_batches_completed=index + 1, resume_batches_total=self.batch,
                          restored_optimizer_steps=self.step_total)
            write(self.monitor.out / 'status.json', status)
    self.restore_rng()
    self.saved_rng = None
    if self.batch:
        # The normal fit timer now correctly excludes loader replay, like checkpoints.
        self.monitor.phase('train')
    yield from iterator


def validate():
    record = read(MANIFEST)
    need(record['version'] == VERSION and record['tag'] == v3.TAG, 'recovery identity')
    need(record['source_sha256'] == sha(__file__), 'recovery source changed')
    reg = v3.validate(read(v3.WORK / 'registration.json'))
    need(record['contract_id'] == reg['contract_id'] and record['command'] == command(), 'same training contract/command')
    need(record['trainer_sha256'] == sha(v3.WORK / 'compact_train_l1.py'), 'original trainer changed')
    need(record['review_sha256'] == sha(v3.REVIEW), 'original review changed')
    need(record['original_argv'] == v3.trainer_argv(v3.CACHE, reg), 'trainer args changed')
    need(not (v3.OUT / v3.COLLAPSE).exists(), 'collapsed experiment must not resume')
    return reg


def owner(child=False):
    """Exact same gpu_job lock/process/command checks, with explicit recovery command."""
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    with file_lock(LOGS / '.gpu-claim.lock'):
        job = read_json(LOGS / v3.TAG / 'job.json')
        parent = psutil.Process().parent()
        need(parent is not None and job.get('schema') == 'gpu-job/v2' and job.get('tag') == v3.TAG
             and job.get('phase') == 'running', 'recovery requires actual gpu_job')
        need(job.get('command') == command() and job.get('command_sha256') == command_digest(command(), ROOT)
             and Path(job['cwd']).resolve() == ROOT, 'recovery command binding')
        if child:
            need(job.get('child_identity') == identity(parent.pid) and parent.cmdline() == command(), 'owned recovery child')
            wrapper = parent.parent()
        else:
            need(job.get('child_identity') == identity(), 'recovery run PID identity')
            wrapper = parent
        need(wrapper is not None and {k: job.get(k) for k in ('pid', 'create_time')} == identity(wrapper.pid), 'wrapper identity')
        need(wrapper.cmdline()[-3:] == ['--_run-job', v3.TAG, job['run_id']]
             and any(Path(x).name == 'gpu_job.py' for x in wrapper.cmdline()), 'wrapper command')
        return wrapper.pid


def run():
    owner_pid = owner()
    reg = validate()
    # The reviewed original admission remains binding; no gate is bypassed.
    admission = v3.admit(reg, v3.CACHE, v3.REVIEW, False, owner_pid)
    prior = read(v3.OUT / 'admission.json')
    need(all(prior[k] == admission[k] for k in ('contract_id', 'review_binding', 'cache')), 'resume admission identity')
    write(WORK / 'actual-admission.json', admission)
    write(v3.OUT / 'admission.json', admission)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['CT_PREDICT_CHUNK'] = '32'
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / v3.s.CODE)])
    for action in ('train', 'infer'):
        if action == 'train' and (v3.OUT / 'fit-complete.json').exists():
            v3.completed_fit(reg)
            v3.final_training_log(reg)
            continue
        v3.supervise(action, command(action), env, v3.OUT, reg['collapse_guard']['constant_loss'])
        if action == 'train':
            v3.final_training_log(reg)
    return read(v3.OUT / 'result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'train', 'infer', 'validate'])
    action = parser.parse_args().action
    if action == 'validate':
        reg = validate()
        print(json.dumps(dict(status='PASS_STATIC_B_DIRECT', contract_id=reg['contract_id'])))
    elif action == 'run':
        print(json.dumps(run(), allow_nan=False))
    else:
        owner(child=True)
        reg = validate()
        if action == 'train':
            from tools import l1_rco_k1_state_v1 as state
            state.TrainerState.batches = replay_batches
            argv = v3.trainer_argv(v3.CACHE, reg)
            sys.argv = argv[1:]
            runpy.run_path(argv[1], run_name='__main__')
        else:
            v3.inference(reg, v3.CACHE)


if __name__ == '__main__':
    main()
