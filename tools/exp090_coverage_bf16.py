"""EXP090 (H-B-01) protocol and launcher: EXP087 coverage retry under BF16 with a BF16 control.

  python tools/exp090_coverage_bf16.py prepare
  python tools/exp090_coverage_bf16.py launch --arm candidate --mode pilot
  python tools/exp090_coverage_bf16.py status --arm candidate --mode pilot

prepare registers the frozen inputs (EXP084 G1 split, labels, coverage stage manifest, code hashes)
once in artifacts/experiments/EXP090_COVERAGE_BF16_PROTOCOL_V1/manifest.json. launch runs the
runner under tools/gpu_job.py with the registered thread/offline environment. Four tags:
EXP090_COV_BF16_{CAND,CTRL}[_PILOT]. No retries, no automatic full after pilot.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.exp084_group_loss import (worker_init, read_rows, validate_membership, install_gold_blocker,  # noqa: E402,F401
                                     uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV)

SCHEMA = 'exp090-coverage-bf16-v1'
PROTOCOL = 'artifacts/experiments/EXP090_COVERAGE_BF16_PROTOCOL_V1'
BASE = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
COVERAGE_MANIFEST = '.local/exp087-coverage/stage-v1/manifest.json'
COVERAGE_MANIFEST_SHA = '3f1541aeb190ba49164e7bc3d067d2b4c1bc0142b87f44acd0ca9a4b1e562371'
RUNNER = 'training/runners/exp090_coverage_bf16.py'
TAGS = {('candidate', 'pilot'): 'EXP090_COV_BF16_CAND_PILOT', ('candidate', 'full'): 'EXP090_COV_BF16_CAND',
        ('control', 'pilot'): 'EXP090_COV_BF16_CTRL_PILOT', ('control', 'full'): 'EXP090_COV_BF16_CTRL'}
EXPECTED_ARGS = dict(arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', res=224, k=12, k_eval=24, epochs=8, fixed_epoch=8, bs=8, accum=1,
                     grad_ckpt=True, workers=12, amp='bf16', eval_amp='fp16', runtime_policy='strict', seed=42, norm='imagenet',
                     bb_lr=3e-05, head_lr=0.001, wd=0.02, pretrained='timm', window_policy='within-slot-valid', device='cuda')
CODE = [RUNNER, 'tools/exp090_coverage_bf16.py', 'training/src/exp087_coverage_data.py', 'tools/exp087_series_coverage.py',
        'tools/exp087_training.py', 'tools/exp084_group_loss.py']


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def prepare():
    destination = ROOT / PROTOCOL; destination.mkdir(parents=True, exist_ok=False)
    baseline = read(ROOT / BASE); b = baseline['arms']['B']
    if sha256(ROOT / COVERAGE_MANIFEST) != COVERAGE_MANIFEST_SHA:
        raise ValueError('coverage stage manifest changed')
    p = {k: baseline[k] for k in ('paths', 'groups', 'labels', 'label_hash', 'eval_label_hash', 'split_hash', 'ordered_ids', 'expected_ids',
                                  'excluded_uids', 'loss', 'corpus_manifest_sha256', 'environment', 'training_code_sha256')}
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-01 (from H-A-04)',
             source_bindings={s: sha256(ROOT / s) for s in CODE + [BASE, COVERAGE_MANIFEST, baseline['paths']['labels'], baseline['ordered_ids']['path'], baseline['groups']['path']]},
             coverage_manifest=COVERAGE_MANIFEST, coverage_manifest_sha256=COVERAGE_MANIFEST_SHA,
             initial_state_sha256=b['initial_state_sha256'], expected_args=EXPECTED_ARGS,
             arms={arm: dict(id=arm, coverage_enabled=arm == 'candidate', tags={m: TAGS[(arm, m)] for m in ('pilot', 'full')}) for arm in ('candidate', 'control')},
             training_budgets=dict(pilot_seconds=300, full_seconds=4200, retry=0),
             projection='1.3*mean(last24 whole updates)*3448+300 <= 3600 at pilot; full launched only by a recorded decision',
             evaluation=dict(primary='macro12(candidate)-macro12(control)', secondary=['candidate-B(EXP084_B_G1)', 'control-B(EXP084_B_G1)'],
                             bootstrap_unit='component_id', bootstrap_reps=500, bootstrap_seed=20261003, no_rank_or_blend=True),
             reference_outputs=b['outputs'],
             validation='same 888 exposed weak G1 development studies / 873 components; one seed; no private/.943 claim',
             difference_from_exp087='numerics: bf16 autocast, no GradScaler, no loss scale; control arm with coverage adapter disabled; otherwise identical B8 recipe and inputs')
    dump(destination / 'manifest.json', p)
    print(json.dumps(dict(status='REGISTERED_NO_EXECUTION', protocol_sha256=sha256(destination / 'manifest.json'), output=PROTOCOL)))


def validate_arguments(args, root):
    p = read(Path(args.protocol))
    if p['schema'] != SCHEMA:
        raise ValueError('wrong protocol')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(Path(root) / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
    for k, v in p['expected_args'].items():
        if getattr(args, k, None) != v:
            raise ValueError('frozen argument changed: ' + k)
    if TAGS[(args.arm, args.mode)] != args.tag:
        raise ValueError('tag/arm/mode mismatch')
    for k, v in p['paths'].items():
        if hasattr(args, k) and Path(getattr(args, k)).resolve() != relpath(root, v).resolve():
            raise ValueError('input path differs: ' + k)
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    ordered = read(Path(root) / p['ordered_ids']['path'])
    for k in ('train', 'val'):
        if uid_hash(ordered[k]) != p['ordered_ids'][k + '_sha256']:
            raise ValueError('ordered G1 changed')
    return p, ordered, p['arms'][args.arm]


def runner_argv(p, arm, mode):
    argv = [sys.executable, str(ROOT / RUNNER), '--protocol', str(ROOT / PROTOCOL / 'manifest.json'), '--arm', arm, '--mode', mode, '--tag', TAGS[(arm, mode)]]
    for k, v in {**p['expected_args'], **{k: str(ROOT / v) for k, v in p['paths'].items() if k in ('corpus', 'labels', 'eval_labels')}}.items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    return argv


def launch(arm, mode):
    p = read(ROOT / PROTOCOL / 'manifest.json'); tag = TAGS[(arm, mode)]
    out = ROOT / 'artifacts/experiments' / tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit('output exists: ' + str(out))
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(ROOT)}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    argv = [sys.executable, str(ROOT / 'tools/gpu_job.py'), '--tag', tag, '--', *runner_argv(p, arm, mode)]
    receipt = dict(status='LAUNCHED', arm=arm, mode=mode, tag=tag, argv=argv, protocol_sha256=sha256(ROOT / PROTOCOL / 'manifest.json'),
                   started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), budget_seconds=p['training_budgets'][mode + '_seconds'], retry=0)
    r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    receipt.update(launcher_returncode=r.returncode, launcher_stdout=r.stdout[-2000:], launcher_stderr=r.stderr[-2000:])
    dump(ROOT / PROTOCOL / f'launch-{tag}.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=1))


def status(arm, mode):
    tag = TAGS[(arm, mode)]
    r = subprocess.run([sys.executable, str(ROOT / 'tools/gpu_job.py'), '--status', tag], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout)
    for name in ('pilot-receipt.json', 'summary.json'):
        f = ROOT / 'artifacts/experiments' / tag / name
        if f.exists():
            print(name, f.read_text(encoding='utf-8')[:1500])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    for name in ('launch', 'status'):
        s = sub.add_parser(name); s.add_argument('--arm', choices=['candidate', 'control'], required=True); s.add_argument('--mode', choices=['pilot', 'full'], required=True)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare()
    elif a.command == 'launch':
        launch(a.arm, a.mode)
    else:
        status(a.arm, a.mode)


if __name__ == '__main__':
    main()
