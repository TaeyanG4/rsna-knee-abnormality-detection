"""EXP095 (H-B-02): nested, leak-free teacher-label mix on G1 — ResNet34 inner 3-fold OOF → CoAt B8 student.

  python tools/exp095_teacher_mix.py prepare            # inner component-grouped 3-fold split inside T3455, teacher/student arms
  python tools/exp095_teacher_mix.py launch --job teacher0|teacher1|teacher2|student
  python tools/exp095_teacher_mix.py mix                # after the 3 teacher fits: 0.5*Dread + 0.5*inner-OOF for T only → mixed_labels.csv (+ hash receipt)
  python tools/exp095_teacher_mix.py status --job ...

Nothing touches V888: teachers are trained and predicted only inside T3455 (component-grouped), the student trains on the
mixed T labels and is evaluated on V888 with the unchanged fixed Dread labels (eval-labels). Uniform 0.5 mix pre-registered.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.exp084_group_loss import (worker_init, read_rows, validate_membership, install_gold_blocker,  # noqa: E402,F401
                                     uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV)

SCHEMA = 'exp095-teacher-mix-v1'
PROTOCOL = 'artifacts/experiments/EXP095_TEACHER_MIX_PROTOCOL_V2'
BASE = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
TEACHER_RUNNER = 'training/runners/exp095_teacher.py'
STUDENT_RUNNER = 'training/runners/exp095_student.py'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
TEACHER_ARGS = dict(arch='resnet34.a1_in1k', res=224, k=12, k_eval=24, epochs=20, fixed_epoch=20, bs=8, accum=1, grad_ckpt=False, workers=12, amp='bf16', eval_amp='fp16',
                    runtime_policy='strict', seed=42, norm='imagenet', bb_lr=0.0005, head_lr=0.001, wd=0.02, pretrained='timm', window_policy='within-slot-valid', device='cuda')
STUDENT_ARGS = dict(arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', res=224, k=12, k_eval=24, epochs=8, fixed_epoch=8, bs=8, accum=1, grad_ckpt=True, workers=12, amp='bf16', eval_amp='fp16',
                    runtime_policy='strict', seed=42, norm='imagenet', bb_lr=3e-05, head_lr=0.001, wd=0.02, pretrained='timm', window_policy='within-slot-valid', device='cuda')
JOBS = {'teacher0': ('teacher', 0, 'EXP095_TEACHER_F0'), 'teacher1': ('teacher', 1, 'EXP095_TEACHER_F1'), 'teacher2': ('teacher', 2, 'EXP095_TEACHER_F2'), 'student': ('student', None, 'EXP095_STUDENT_B8')}
CODE = [TEACHER_RUNNER, STUDENT_RUNNER, 'tools/exp095_teacher_mix.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py']
SALT = 'EXP095_INNER_3FOLD_V1'


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def prepare():
    destination = ROOT / PROTOCOL; destination.mkdir(parents=True, exist_ok=False)
    baseline = read(ROOT / BASE); b = baseline['arms']['B']
    ordered = read(ROOT / baseline['ordered_ids']['path'])
    rows = read_rows(ROOT / baseline['groups']['path']); comp = {r['StudyInstanceUID']: r['component_id'] for r in rows}
    inner = {}
    for u in ordered['train']:
        c = comp[u]
        inner[u] = int.from_bytes(hashlib.sha256((SALT + ':' + c).encode()).digest(), 'big') % 3
    counts = [sum(1 for v in inner.values() if v == f) for f in range(3)]
    p = {k: baseline[k] for k in ('paths', 'groups', 'labels', 'label_hash', 'eval_label_hash', 'split_hash', 'ordered_ids', 'expected_ids', 'excluded_uids', 'loss',
                                  'corpus_manifest_sha256', 'environment', 'training_code_sha256')}
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-02',
             source_bindings={s: sha256(ROOT / s) for s in CODE + [BASE, baseline['paths']['labels'], baseline['ordered_ids']['path'], baseline['groups']['path']]},
             inner_split=dict(salt=SALT, rule='sha256(salt:component_id) mod 3 over T3455 components', counts=counts, assignment_sha256=hashlib.sha256(json.dumps(inner, sort_keys=True).encode()).hexdigest()),
             teacher_args=TEACHER_ARGS, student_args=STUDENT_ARGS, student_initial_state_sha256=b['initial_state_sha256'],
             jobs={k: dict(kind=v[0], fold=v[1], tag=v[2]) for k, v in JOBS.items()},
             mix=dict(rule='0.5*Dread_soft + 0.5*teacher_inner_OOF_probability for every T3455 study; V888 rows copied unchanged (never used for training; eval uses eval-labels)', output=f'{PROTOCOL}/mixed_labels.csv'),
             training_budgets=dict(teacher_seconds=1500, student_seconds=4500, retry=0),
             evaluation=dict(primary='macro12(student)-macro12(EXP084_B_G1)', reference='artifacts/experiments/EXP084_B_G1/oof_best.csv', bf16_control='artifacts/experiments/EXP090_COV_BF16_CTRL/oof_best.csv',
                             seed_floor='artifacts/experiments/EXP094_B8_SEED43/oof_best.csv', bootstrap_unit='component_id', bootstrap_reps=500, bootstrap_seed=20261003),
             validation='teachers never see V888 or their own held-out inner fold; student evaluated on fixed Dread V888; one seed; no private/.943 claim')
    dump(destination / 'manifest.json', p)
    dump(destination / 'inner-assignment.json', inner)
    print(json.dumps(dict(status='REGISTERED_NO_EXECUTION', protocol_sha256=sha256(destination / 'manifest.json'), inner_counts=counts)))


def validate_common(args, root, kind):
    p = read(Path(args.protocol))
    if p['schema'] != SCHEMA:
        raise ValueError('wrong protocol')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(Path(root) / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
    for k, v in p[kind + '_args'].items():
        if getattr(args, k, None) != v:
            raise ValueError('frozen argument changed: ' + k)
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    ordered = read(Path(root) / p['ordered_ids']['path'])
    for k in ('train', 'val'):
        if uid_hash(ordered[k]) != p['ordered_ids'][k + '_sha256']:
            raise ValueError('ordered G1 changed')
    inner = read(Path(root) / PROTOCOL / 'inner-assignment.json')
    if hashlib.sha256(json.dumps(inner, sort_keys=True).encode()).hexdigest() != p['inner_split']['assignment_sha256']:
        raise ValueError('inner assignment changed')
    return p, ordered, inner


def mix():
    p = read(ROOT / PROTOCOL / 'manifest.json'); ordered = read(ROOT / p['ordered_ids']['path']); inner = read(ROOT / PROTOCOL / 'inner-assignment.json')
    lab = pd.read_csv(ROOT / p['paths']['labels']).set_index('StudyInstanceUID')
    oof = {}
    for f in range(3):
        d = ROOT / 'artifacts/experiments' / JOBS[f'teacher{f}'][2]
        s = read(d / 'summary.json')
        if s.get('optimizer_updates', 0) <= 0 or not (d / 'oof_fold.csv').exists():
            raise ValueError('teacher fold incomplete: ' + str(f))
        df = pd.read_csv(d / 'oof_fold.csv').set_index('StudyInstanceUID')[T]
        expected = sorted(u for u in ordered['train'] if inner[u] == f)
        if sorted(df.index) != expected:
            raise ValueError('teacher fold coverage mismatch: ' + str(f))
        oof.update({u: df.loc[u].to_numpy(np.float32) for u in df.index})
    out = lab.copy()
    for u in ordered['train']:
        out.loc[u, T] = 0.5 * lab.loc[u, T].to_numpy(np.float32) + 0.5 * oof[u]
    if not np.isfinite(out[T].to_numpy()).all():
        raise ValueError('nonfinite mixed labels')
    path = ROOT / p['mix']['output']
    out.reset_index().to_csv(path, index=False, float_format='%.6f')
    receipt = dict(status='MIXED_LABELS_WRITTEN', rule=p['mix']['rule'], path=p['mix']['output'], sha256=sha256(path), rows=len(out), train_rows_mixed=len(ordered['train']),
                   teacher_oof_sha256={f: sha256(ROOT / 'artifacts/experiments' / JOBS[f'teacher{f}'][2] / 'oof_fold.csv') for f in range(3)}, written_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    dump(ROOT / PROTOCOL / 'mix-receipt.json', receipt)
    print(json.dumps(receipt, indent=1))


def runner_argv(p, job):
    kind, fold, tag = JOBS[job]
    runner = TEACHER_RUNNER if kind == 'teacher' else STUDENT_RUNNER
    argv = [sys.executable, str(ROOT / runner), '--protocol', str(ROOT / PROTOCOL / 'manifest.json'), '--tag', tag]
    if kind == 'teacher':
        argv += ['--fold-index', str(fold)]
    for k, v in {**p[kind + '_args'], **{k: str(ROOT / v) for k, v in p['paths'].items() if k in ('corpus', 'eval_labels')}}.items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--labels', str(ROOT / (p['paths']['labels'] if kind == 'teacher' else p['mix']['output']))]
    return argv


def launch(job):
    p = read(ROOT / PROTOCOL / 'manifest.json'); tag = JOBS[job][2]
    if JOBS[job][0] == 'student' and not (ROOT / PROTOCOL / 'mix-receipt.json').exists():
        raise SystemExit('run mix first')
    out = ROOT / 'artifacts/experiments' / tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit('output exists: ' + str(out))
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(ROOT)}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    argv = [sys.executable, str(ROOT / 'tools/gpu_job.py'), '--tag', tag, '--', *runner_argv(p, job)]
    receipt = dict(status='LAUNCHED', job=job, tag=tag, argv=argv, protocol_sha256=sha256(ROOT / PROTOCOL / 'manifest.json'), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), retry=0)
    r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    receipt.update(launcher_returncode=r.returncode, launcher_stdout=r.stdout[-2000:], launcher_stderr=r.stderr[-2000:])
    dump(ROOT / PROTOCOL / f'launch-{tag}.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'tag', 'started_at', 'launcher_returncode', 'launcher_stdout', 'launcher_stderr')}, ensure_ascii=False, indent=1))


def status(job):
    tag = JOBS[job][2]
    r = subprocess.run([sys.executable, str(ROOT / 'tools/gpu_job.py'), '--status', tag], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare'); sub.add_parser('mix')
    for name in ('launch', 'status'):
        s = sub.add_parser(name); s.add_argument('--job', choices=list(JOBS), required=True)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare()
    elif a.command == 'mix':
        mix()
    elif a.command == 'launch':
        launch(a.job)
    else:
        status(a.job)


if __name__ == '__main__':
    main()
