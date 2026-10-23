"""EXP094 protocol and launcher for pre-registered B8 single-variable variants (H-B-04/05/08/11).

  python tools/exp094_b8_variant.py prepare
  python tools/exp094_b8_variant.py launch --arm seed43 --mode full
  python tools/exp094_b8_variant.py status --arm seed43 --mode full

Arms: seed43 (variant seed, seed 43), strat (slot-stratified training sampling), geoaug (affine bundle), slotdrop (p=0.2).
Tags: EXP094_B8_<ARM>[_PILOT]. Each full run ~45–65 GPU-min; no retries, nothing automatic after a pilot.
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

SCHEMA = 'exp094-b8-variant-v4-fp16-seeds'
PROTOCOL = 'artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4'
BASE = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
RUNNER = 'training/runners/exp094_b8_variant.py'
ARMS = {
    'seed42': dict(variant='seed', seed=42, slot_dropout=0.0, plan_item='H-B-17'),
    'seed43': dict(variant='seed', seed=43, slot_dropout=0.0, plan_item='H-B-17'),
    'seed44': dict(variant='seed', seed=44, slot_dropout=0.0, plan_item='H-B-17'),
    'strat': dict(variant='strat', seed=42, slot_dropout=0.0, plan_item='H-B-08'),
    'geoaug': dict(variant='geoaug', seed=42, slot_dropout=0.0, plan_item='H-B-04'),
    'slotdrop': dict(variant='slotdrop', seed=42, slot_dropout=0.2, plan_item='H-B-11'),
}
COMMON = dict(arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', res=224, k=12, k_eval=24, epochs=8, fixed_epoch=8, bs=8, accum=1, grad_ckpt=True, workers=12,
              amp='fp16', amp_init_scale=1024.0, eval_amp='fp16', runtime_policy='strict', norm='imagenet', bb_lr=3e-05, head_lr=0.001, wd=0.02, pretrained='timm',
              window_policy='within-slot-valid', device='cuda')
CODE = [RUNNER, 'tools/exp094_b8_variant.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py']


def tag_for(arm, mode):
    return f'EXP094L_B8_{arm.upper()}' + ('_PILOT' if mode == 'pilot' else '')


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def prepare():
    destination = ROOT / PROTOCOL; destination.mkdir(parents=True, exist_ok=False)
    baseline = read(ROOT / BASE); b = baseline['arms']['B']
    p = {k: baseline[k] for k in ('paths', 'groups', 'labels', 'label_hash', 'eval_label_hash', 'split_hash', 'ordered_ids', 'expected_ids', 'excluded_uids', 'loss',
                                  'corpus_manifest_sha256', 'environment', 'training_code_sha256')}
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION',
             source_bindings={s: sha256(ROOT / s) for s in CODE + [BASE, baseline['paths']['labels'], baseline['ordered_ids']['path'], baseline['groups']['path']]},
             initial_state_sha256=b['initial_state_sha256'], common_args=COMMON,
             arms={arm: dict(id=arm, **spec, tags={m: tag_for(arm, m) for m in ('pilot', 'full')}) for arm, spec in ARMS.items()},
             training_budgets=dict(pilot_seconds=600, full_seconds=21600, retry=0), precision_note='remote T4 has no bf16: FP16 + GradScaler(init 1024) exactly like EXP084_B_G1; skips recorded, >1% fails; workers 4 for 2-4 vCPU remote hosts',
             evaluation=dict(primary='macro12(arm)-macro12(EXP084_B_G1)', reference='artifacts/experiments/EXP084_B_G1/oof_best.csv', bf16_control='artifacts/experiments/EXP090_COV_BF16_CTRL/oof_best.csv when available',
                             bootstrap_unit='component_id', bootstrap_reps=500, bootstrap_seed=20261003, no_rank_or_blend=True),
             validation='same 888 exposed weak G1 development studies / 873 components; one seed per arm; eval windows unchanged; no private/.943 claim',
             note='seed43 arm: the pretrained backbone is identical, but the randomly initialised head depends on the seed, so the initial-state hash is recorded (initial_state_matches_B) and only enforced for seed 42; the seed also changes window sampling and shuffling.')
    dump(destination / 'manifest.json', p)
    print(json.dumps(dict(status='REGISTERED_NO_EXECUTION', protocol_sha256=sha256(destination / 'manifest.json'), output=PROTOCOL)))


def validate_arguments(args, root):
    p = read(Path(args.protocol))
    if p['schema'] != SCHEMA or args.arm not in p['arms']:
        raise ValueError('wrong protocol or unregistered arm')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(Path(root) / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
    for k, v in p['common_args'].items():
        if getattr(args, k, None) != v:
            raise ValueError('frozen argument changed: ' + k)
    arm = p['arms'][args.arm]
    for k in ('variant', 'seed', 'slot_dropout'):
        if getattr(args, k) != arm[k]:
            raise ValueError('arm argument changed: ' + k)
    if arm['tags'][args.mode] != args.tag:
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
    return p, ordered, arm


def runner_argv(p, arm, mode):
    spec = p['arms'][arm]
    argv = [sys.executable, str(ROOT / RUNNER), '--protocol', str(ROOT / PROTOCOL / 'manifest.json'), '--arm', arm, '--mode', mode, '--tag', spec['tags'][mode],
            '--variant', spec['variant'], '--seed', str(spec['seed']), '--slot-dropout', str(spec['slot_dropout'])]
    for k, v in {**p['common_args'], **{k: str(ROOT / v) for k, v in p['paths'].items() if k in ('corpus', 'labels', 'eval_labels')}}.items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    return argv


def launch(arm, mode):
    p = read(ROOT / PROTOCOL / 'manifest.json'); tag = p['arms'][arm]['tags'][mode]
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
    print(json.dumps({k: receipt[k] for k in ('status', 'tag', 'started_at', 'launcher_returncode', 'launcher_stdout', 'launcher_stderr')}, ensure_ascii=False, indent=1))


def status(arm, mode):
    tag = tag_for(arm, mode)
    r = subprocess.run([sys.executable, str(ROOT / 'tools/gpu_job.py'), '--status', tag], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:])
    for name in ('pilot-receipt.json', 'summary.json'):
        f = ROOT / 'artifacts/experiments' / tag / name
        if f.exists():
            print(name, f.read_text(encoding='utf-8')[:800])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    for name in ('launch', 'status'):
        s = sub.add_parser(name); s.add_argument('--arm', choices=list(ARMS), required=True); s.add_argument('--mode', choices=['pilot', 'full'], required=True)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare()
    elif a.command == 'launch':
        launch(a.arm, a.mode)
    else:
        status(a.arm, a.mode)


if __name__ == '__main__':
    main()
