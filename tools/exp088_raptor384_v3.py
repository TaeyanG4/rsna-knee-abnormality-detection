"""EXP088 metadata preparation and bounded 384px G1 training.

prepare reads small existing metadata and hashes assets; it does not import a
trainer, load a model, score predictions, or redraw the split. run requires an
exact phase decision and uses Windows private-commit/host-RAM gates. No retries.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.exp084_group_loss import (read, sha256, uid_hash, relpath, worker_init,
    read_rows, validate_membership, validate_label_contract, loss_weights,
    install_gold_blocker, CODE_NAMES, ENV_KEYS, THREAD_ENV)
from tools.exp087_training import stop_loader
from tools.exp088_resource_v2 import supervise, save_recovery

SCHEMA = 'exp088-raptor384-g1-resource-v2'
BASE = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
PROTOCOL = 'artifacts/experiments/EXP088_RAPTOR384_G1_PROTOCOL_V6'
PILOT_PROTOCOL = 'artifacts/experiments/EXP088_RAPTOR384_G1_PROTOCOL_V3/manifest.json'
ENTRY = 'training/runners/exp088_raptor384_v3.py'
TAGS = {'pilot': 'EXP088_RAPTOR384_G1_PILOT_V2', 'full': 'EXP088_RAPTOR384_G1_V3'}
METRICS = 'deferred_single_fixed_evaluation'
HF_ROOT = Path('C:/Users/Taeyang/.cache/huggingface/hub/models--timm--coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k')
HF_REVISION = '2c31d82376cc244c907c064332941e6c65be81f1'
GPU_PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'


def dump(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare():
    destination = ROOT / PROTOCOL
    if destination.exists():
        raise ValueError('Existing registration; never overwrite')
    prior_path = ROOT / 'artifacts/experiments/EXP088_RAPTOR384_G1_PROTOCOL_V5/manifest.json'
    p = copy.deepcopy(read(prior_path))
    p.update(schema=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(),
             status='REGISTERED_PENDING_REVIEW')
    prior_tag = p['arms']['full']['tag']
    arm = p['arms']['full']
    arm['tag'] = TAGS['full']
    arm['logs'] = f"artifacts/logs/{TAGS['full']}"
    arm['outputs'] = {k: v.replace(prior_tag, TAGS['full']) for k, v in arm['outputs'].items()}
    p['budget'].pop('full_tree_rss_bytes', None)
    p['budget'].update(full_tree_private_commit_bytes=24 * 2**30,
                       host_available_bytes_floor=8 * 2**30,
                       sampled_interval_seconds=.1)
    old_runner = ROOT / 'training/runners/exp088_raptor384_v2.py'
    p['resource_revision'] = dict(prior_protocol_path=str(prior_path.relative_to(ROOT)).replace('\\','/'),
        prior_protocol_sha256=sha256(prior_path), prior_tag=prior_tag,
        failed_supervisor_sha256=sha256(ROOT / f'artifacts/experiments/{prior_tag}/resource-supervisor.json'),
        model_recipe_unchanged=True, no_pilot_replay=True, no_resume=True,
        original_RSS_failure_sha256=p['resource_revision']['failed_supervisor_sha256'],
        startup_fix='allow only owned resource-latest and atomic metadata tmp startup files',
        snapshot='atomic latest epoch state; fixed-final selection unchanged; future resume requires review',
        old_runner_sha256=sha256(old_runner))
    for path in ('tools/exp088_raptor384_v3.py', ENTRY, 'tools/exp088_resource_v2.py',
                 'tools/process_resource_receipt.py', p['resource_revision']['prior_protocol_path']):
        p['source_bindings'][path] = sha256(ROOT / path)
    destination.mkdir(parents=True)
    dump(destination / 'manifest.json', p)
    digest = sha256(destination / 'manifest.json')
    env = {**THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    cmd = [GPU_PYTHON, str(ROOT / 'tools/exp088_raptor384_v3.py'), 'run',
           '--protocol', str(destination / 'manifest.json'), '--protocol-sha256', digest,
           '--mode', 'full', '--decision', str(destination / 'root-full-decision.json')]
    dump(destination / 'execution-plans.json', {'full': dict(cwd=str(ROOT), environment=env,
        argv=[GPU_PYTHON, str(ROOT / 'tools/gpu_job.py'), '--tag', TAGS['full'], '--', *cmd])})
    print(json.dumps(dict(status=p['status'], manifest_sha256=digest, no_model_execution=True)))


def validate_bindings(p):
    if p.get('schema') != SCHEMA:
        raise ValueError('Wrong EXP088 protocol')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(ROOT, path)) != expected:
            raise ValueError('Frozen binding changed: ' + path)
    for path, expected in p['generic_assets'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('Frozen generic asset changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(ROOT / 'training/src' / name) != expected:
            raise ValueError('Original source changed: ' + name)


def arm_for(p, tag):
    arms = [a for a in p['arms'].values() if a['tag'] == tag]
    if len(arms) != 1:
        raise ValueError('Unregistered tag')
    return arms[0]


def validate_arguments(a, root):
    p = read(a.protocol)
    validate_bindings(p)
    arm = arm_for(p, a.tag)
    if os.environ.get('EXP088_AUTHORIZED_PROTOCOL_SHA256') != sha256(a.protocol):
        raise ValueError('Runner must be launched by exact approved supervisor')
    for key, value in arm['expected_args'].items():
        if getattr(a, key, None) != value:
            raise ValueError('Frozen argument changed: ' + key)
    for key, value in p['paths'].items():
        if Path(getattr(a, key)).resolve() != relpath(root, value).resolve():
            raise ValueError('Frozen input path changed: ' + key)
    for key, value in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(key) != value:
            raise ValueError('Frozen environment changed: ' + key)
    ordered = read(ROOT / p['ordered_ids']['path'])
    if any(uid_hash(ordered[k]) != p['ordered_ids'][k + '_sha256'] for k in ('train', 'val')):
        raise ValueError('Ordered cohort changed')
    return p, ordered


def validate_pretraining(v, env, p, arm):
    expected = dict(code_sha256=p['training_code_sha256'],
        train_ids_sha256=p['ordered_ids']['train_sha256'], eval_ids_sha256=p['ordered_ids']['val_sha256'],
        train_ids_ordered_sha256=p['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=p['ordered_ids']['val_sha256'],
        corpus_manifest_sha256=p['corpus_manifest_sha256'], checkpoint_selection='fixed-final',
        sampling='sha256(seed,epoch,uid,stream)-v1', gold_access='disabled',
        loss_policy='clipped_positive_term', metric_access=METRICS)
    if any(v.get(k) != value for k, value in expected.items()):
        raise ValueError('Pretraining contract changed')
    if any(v[k] != p['expected_runtime'][k] for k in ('runtime', 'eval_runtime')):
        raise ValueError('Frozen strict runtime changed')
    if any(env.get(k) != p['environment'][k] for k in ENV_KEYS):
        raise ValueError('Environment changed')
    if arm['id'] == 'full':
        pilot = read(ROOT / 'artifacts/experiments' / TAGS['pilot'] / 'provenance.json')
        if v['initial_state_sha256'] != pilot['initial_state_sha256']:
            raise ValueError('Full did not restart identical generic initialization')


def validate_completion(out, p, arm):
    s = read(Path(out) / ('pilot-receipt.json' if arm['id'] == 'pilot' else 'summary.json'))
    if s['optimizer_updates'] != arm['expected_attempts'] or s['skipped_updates'] != 0:
        raise ValueError('Update count/skip contract failed')
    if s['scheduler_horizon'] != 5184 or s['scheduler_steps'] != s['optimizer_updates']:
        raise ValueError('Scheduler contract failed')
    if not s['worker_cleanup']['clean'] or s['gold_access'] != 'disabled':
        raise ValueError('Cleanup/Gold contract failed')
    if arm['id'] == 'pilot':
        if len(s['whole_update_seconds']) != 32 or s['projected_full_seconds'] > p['budget']['full_seconds']:
            raise ValueError('Pilot budget projection failed')
        if s['validation_forwards'] != 0 or s['checkpoint_writes'] != 0 or not s['fp16_t_batch_finite']:
            raise ValueError('Pilot training-only/serving contract failed')
        if any((Path(out) / n).exists() for n in ('model_best.pt', 'oof_best.csv')):
            raise ValueError('Pilot evaluated or saved checkpoint')
    elif (s['selected_epoch_1based'] != 12 or s['train'] != 3455 or s['val'] != 888
          or s['weak_oof_macro_auc'] is not None or len(s['history']) != 12
          or s['metric_access'] != METRICS):
        raise ValueError('Full completion contract failed')
    for name in (('provenance.json',) if arm['id'] == 'pilot' else
                 ('model_best.pt', 'oof_best.csv', 'provenance.json', 'history.json')):
        if not (Path(out) / name).is_file():
            raise ValueError('Missing final artifact: ' + name)


def run(a):
    if sha256(a.protocol) != a.protocol_sha256:
        raise ValueError('Exact protocol SHA required')
    p = read(a.protocol)
    validate_bindings(p)
    decision = read(a.decision)
    if (decision.get('status') != 'AUTHORIZE_EXACT_PHASE' or decision.get('phase') != a.mode
            or decision.get('protocol_sha256') != a.protocol_sha256):
        raise ValueError('Root phase authorization absent')
    if a.mode == 'full':
        if decision.get('failed_supervisor_sha256') != p['resource_revision']['failed_supervisor_sha256']:
            raise ValueError('New version must explicitly bind preserved full failure')
        pilot = ROOT / 'artifacts/experiments' / TAGS['pilot']
        if not (ROOT / 'artifacts/logs' / TAGS['pilot'] / 'SUCCESS').is_file():
            raise ValueError('Pilot parent did not succeed')
        supervisor = read(pilot / 'resource-supervisor.json')
        if supervisor['status'] != 'PASS' or supervisor['protocol_sha256'] != p['pilot_reuse']['protocol_sha256']:
            raise ValueError('Pilot supervisor did not pass the bound equivalent model protocol')
        validate_completion(pilot, p, p['arms']['pilot'])
        expected = {n: sha256(pilot / n) for n in ('pilot-receipt.json', 'resource-supervisor.json', 'provenance.json')}
        if decision.get('pilot_bindings') != expected:
            raise ValueError('Root pilot acceptance changed')
    output = ROOT / 'artifacts/experiments' / TAGS[a.mode]
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'phase-authorization.json', decision)
    argv = [sys.executable, str(ROOT / ENTRY), '--protocol', str(Path(a.protocol).resolve()), '--tag', TAGS[a.mode]]
    for key, value in {**p['arms'][a.mode]['expected_args'], **p['paths']}.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(relpath(ROOT, value)) if key in p['paths'] else str(value)]
    # EXP087 supervisor is reused unchanged; these two environment keys only
    # supply generic phase/authorization fields in its resource receipt.
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'EXP088_AUTHORIZED_PROTOCOL_SHA256': a.protocol_sha256,
           'EXP087_PHASE': a.mode, 'EXP087_PHASE_AUTHORIZATION_SHA256': sha256(output / 'phase-authorization.json')}
    if a.mode == 'pilot':
        raise ValueError('Completed pilot is reused; this protocol only executes full')
    supervise(argv, output, p['budget']['full_seconds'], p['budget']['full_tree_private_commit_bytes'],
              p['budget']['host_available_bytes_floor'], env, a.protocol_sha256)
    validate_completion(output, p, p['arms'][a.mode])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    r = sub.add_parser('run')
    r.add_argument('--protocol', required=True)
    r.add_argument('--protocol-sha256', required=True)
    r.add_argument('--mode', choices=['pilot', 'full'], required=True)
    r.add_argument('--decision', required=True)
    a = ap.parse_args()
    prepare() if a.command == 'prepare' else run(a)


if __name__ == '__main__':
    main()
