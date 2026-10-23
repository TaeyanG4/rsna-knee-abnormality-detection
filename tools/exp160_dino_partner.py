"""EXP160 v1.0.0: bounded two-arm DINO package and registered CPU screen.

Preparation and planning never start compute. Remote execution requires B's
exact-manifest admission and explicit detached gpu_queue launch.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # script and -m entry points behave alike
from tools import exp158_dino_strong as old

SCHEMA = 'exp160-dino-partner-v1'
VERSION = '1.0.1'  # B 2026-10-09: re-export the exp158 guard names the runner uses
SOURCE = ROOT / '.local/exp158-dino-strong-v1/manifest.json'
SOURCE_SHA = 'f1d7cbf34869e5c304a693c06a4fba4d0ec26b1175a2c8c292f1e7e8c86ef699'
PROTOCOL = 'artifacts/experiments/EXP160_DINO_PARTNER/PROTOCOL.md'
RUNNER = 'training/runners/exp160_dino_partner.py'
ADAPTER_POS = 'training/adapters/exp160_base_cpu_pos.py'
ADAPTER_PATCH = 'training/adapters/exp160_base_patch_mil.py'
WEIGHT_BASE = 'storage/weights/exp160/dinov2b.safetensors'
ARMS = ('S43', 'BASE')
TAGS = {'S43': 'EXP160_DINO_S43', 'BASE': 'EXP160_DINO_BASE'}
BUDGET = {'pilot_seconds': 300, 'full_seconds': 10000, 'total_seconds': 10800,
          'pilot_updates': 32, 'full_updates': 8620, 'retries': 0,
          'peak_gpu_allocated_cap_bytes': 20 * 2**30}
NEW = ('tools/exp160_dino_partner.py', 'tools/exp158_dino_strong.py',
       RUNNER, ADAPTER_POS, ADAPTER_PATCH,
       'registered/EXP160_PROTOCOL.md', 'queue.json')
BASE_REPO = 'timm/vit_base_patch14_dinov2.lvd142m'
JEV, AQUILA, RAD = old.JEV, old.AQUILA, old.RAD
PARTNERS = {'RadR50': RAD,
            'H1': 'artifacts/experiments/EXP149_H1_S42_F0/oof_k36.csv',
            'L2': 'artifacts/experiments/EXP149_L2_S42_F0/oof_k36.csv'}

need, sha, read, write_new, within = old.need, old.sha, old.read, old.write_new, old.within


def runner_source():
    source = (ROOT / '.local/exp158-dino-strong-v1/training/runners/exp158_dino_strong.py').read_text(encoding='utf-8')
    before = 'from tools import exp158_dino_strong as guard'
    need(source.count(before) == 1, 'EXP158 runner changed')
    return source.replace(before, 'from tools import exp160_dino_partner as guard')


def adapter_source(name):
    original = 'dinov2_cpu_pos_v2.py' if name == ADAPTER_POS else 'dinov2_patch_mil.py'
    source = (ROOT / 'training/adapters' / original).read_text(encoding='utf-8')
    need(source.count('384') >= 1, 'DINO adapter template changed')
    return source.replace('384', '768').replace('DINOv2-S', 'DINOv2-B').replace('DINO-S', 'DINO-B')


def queue(arm):
    return [{'tag': TAGS[arm], 'command': ['python', 'tools/exp160_dino_partner.py',
            'run', '--protocol', 'manifest.json']}]


def prepare(out):
    out = Path(out)
    need(not out.exists(), 'new overlay directory required')
    need(sha(SOURCE) == SOURCE_SHA, 'EXP158 package manifest changed')
    need(sha(ROOT / 'tools/exp158_dino_strong.py') == read(SOURCE)['files']['tools/exp158_dino_strong.py']['sha256'],
         'frozen EXP158 source changed')
    parent = read(SOURCE)
    need(parent['schema'] == old.SCHEMA and parent['ordered_ids']['val_sha256'] ==
         '56ae86a4c180cc46d28e2a192485d133d58f4d05907b67fa18076f199f507dc4',
         'EXP158 split changed')
    out.mkdir(parents=True)
    for arm in ARMS:
        directory = out / arm
        directory.mkdir()
        p = copy.deepcopy(parent)
        p.update(schema=SCHEMA, version=VERSION, exp160_arm=arm, status='PREPARED_NOT_LAUNCHED',
                 source_manifest_sha256=sha(SOURCE), protocol_sha256=sha(ROOT / PROTOCOL),
                 execution=copy.deepcopy(BUDGET), expected_updates=8620,
                 evaluation={'draws': 300, 'seed': 20261003, 'primary_weight': .1})
        tag = TAGS[arm]
        p['arm']['tag'] = tag
        p['arm']['id'] = arm
        p['expected_args']['seed'] = 43 if arm == 'S43' else 42
        if arm == 'BASE':
            p['expected_args']['arch'] = 'vit_base_patch14_dinov2.lvd142m'
            p['initial_asset'] = {'path': WEIGHT_BASE, 'repo': BASE_REPO,
                                  'filename': 'model.safetensors', 'sha256': None,
                                  'bytes': None, 'revision': None}
            p['full_initial_state'] = None
            p['files'].pop(old.read(SOURCE)['full_initial_state']['path'])
            p['files'].pop(old.read(SOURCE)['initial_asset']['path'])
            p['files'][WEIGHT_BASE] = {'sha256': None, 'bytes': None, 'payload': False}
        p['arm']['expected_args'] = copy.deepcopy(p['expected_args'])
        p['arm']['initial_state_sha256'] = None  # seeded head differs; pilot binds full fit
        p['files'].pop('queue.json')
        p['files'].pop('registered/EXP158_PROTOCOL.md')
        for name in NEW:
            target = within(directory, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in ('tools/exp160_dino_partner.py', 'tools/exp158_dino_strong.py'):
                shutil.copyfile(ROOT / name, target)
            elif name == RUNNER:
                target.write_text(runner_source(), encoding='utf-8', newline='\n')
            elif name in (ADAPTER_POS, ADAPTER_PATCH):
                target.write_text(adapter_source(name), encoding='utf-8', newline='\n')
            elif name == 'registered/EXP160_PROTOCOL.md':
                shutil.copyfile(ROOT / PROTOCOL, target)
            else:
                write_new(target, queue(arm))
            p['files'][name] = {'sha256': sha(target), 'bytes': target.stat().st_size,
                                'payload': True}
        p['arm']['expected_attempts'] = 8620
        write_new(directory / 'manifest.json', p)
    print(json.dumps({'status': 'PREPARED_NOT_LAUNCHED', 'out': str(out),
                      'arms': list(ARMS), 'protocol_sha256': sha(ROOT / PROTOCOL)}))


def validate_manifest(path, package=None, full=False):
    path = Path(path)
    root = Path(package or path.parent)
    p = read(path)
    arm = p.get('exp160_arm')
    need(p.get('schema') == SCHEMA and arm in ARMS and p.get('version') == VERSION,
         'wrong EXP160 manifest')
    need(p['protocol_sha256'] == p['files']['registered/EXP160_PROTOCOL.md']['sha256']
         and p['execution'] == BUDGET and p['expected_updates'] == 8620, 'protocol/budget differs')
    need(p['source_manifest_sha256'] == SOURCE_SHA, 'source manifest binding differs')
    args = p['expected_args']
    need(p['arm']['expected_args'] == args and p['arm']['tag'] == TAGS[arm]
         and args['seed'] == (43 if arm == 'S43' else 42)
         and args['arch'] == ('vit_small_patch14_dinov2.lvd142m' if arm == 'S43'
                             else 'vit_base_patch14_dinov2.lvd142m')
         and args['epochs'] == args['fixed_epoch'] == 20 and args['k'] == 12
         and args['k_eval'] == 36 and args['res'] == 224 and args['bs'] == 8
         and args['accum'] == 1, 'fixed recipe differs')
    need(p['expected_ids']['train'] == 3455 and p['expected_ids']['val'] == 888
         and p['ordered_ids']['val_sha256'] ==
         '56ae86a4c180cc46d28e2a192485d133d58f4d05907b67fa18076f199f507dc4',
         'fixed split differs')
    need(read(root / 'queue.json') == queue(arm), 'queue changed')
    for name, entry in p['files'].items():
        target = within(root, name)
        if name == WEIGHT_BASE and entry['sha256'] is None:
            need(arm == 'BASE' and not full, 'base weight unbound on remote')
            continue
        if name in NEW or full:
            need(target.is_file() and not target.is_symlink()
                 and target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'],
                 'file differs: ' + name)
    if full:
        ids = read(root / p['ordered_ids']['path'])
        need(len(ids['train']) == 3455 and len(ids['val']) == 888
             and not set(ids['train']) & set(ids['val']), 'split differs')
        need(arm != 'BASE' or (p['initial_asset']['sha256'] == p['files'][WEIGHT_BASE]['sha256']
             and len(p['initial_asset']['revision']) == 40), 'base asset provenance missing')
    return p


def _download_base(destination):
    """Called only by explicit remote stage; no cache/token file is read by this code."""
    from huggingface_hub import hf_hub_download
    asset = Path(hf_hub_download(repo_id=BASE_REPO, filename='model.safetensors', token=False))
    revision = asset.parent.name
    need(len(revision) == 40 and all(c in '0123456789abcdef' for c in revision),
         'resolved Hub revision must be a commit')
    shutil.copyfile(asset, destination)
    return revision


def vast_stage(overlay, source, destination):
    need(sys.platform == 'linux', 'vast-stage is Linux-only')
    overlay, source, destination = map(Path, (overlay, source, destination))
    need(not destination.exists(), 'fresh remote destination required')
    need(source.resolve().parent == destination.resolve().parent,
         'hardlinks require sibling roots on same disk')
    source_manifest = read(source / 'manifest.json')
    p = validate_manifest(overlay / 'manifest.json', overlay)
    need(sha(source / 'manifest.json') == p['source_manifest_sha256'],
         'remote EXP158 package differs')
    destination.mkdir(mode=0o700)
    for name, entry in p['files'].items():
        target = within(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in NEW:
            src = within(overlay, name)
            need(not src.is_symlink(), 'overlay symlink forbidden')
            shutil.copyfile(src, target)
        elif name == WEIGHT_BASE:
            revision = _download_base(target)
            p['initial_asset'].update(revision=revision, sha256=sha(target), bytes=target.stat().st_size)
            entry.update(sha256=sha(target), bytes=target.stat().st_size)
        else:
            src = within(source, name)
            need(name in source_manifest['files'] and src.is_file() and not src.is_symlink()
                 and src.stat().st_size == entry['bytes'], 'source file missing: ' + name)
            os.link(src, target)
            need(src.stat().st_ino == target.stat().st_ino, 'hardlink failed: ' + name)
        need(target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'],
             'staged file differs: ' + name)
    write_new(destination / 'manifest.json', p)
    validate_manifest(destination / 'manifest.json', destination, full=True)
    write_new(destination / 'stage-receipt.json', {'status': 'STAGED_VERIFIED_NO_GPU',
              'manifest_sha256': sha(destination / 'manifest.json'), 'arm': p['exp160_arm'],
              'initial_asset': p['initial_asset']})
    print(json.dumps({'status': 'STAGED_VERIFIED_NO_GPU', 'destination': destination.as_posix(),
                      'manifest_sha256': sha(destination / 'manifest.json'),
                      'initial_asset_sha256': p['initial_asset']['sha256']}))


# Guard methods called by the generated runner. Keep the frozen EXP158 runner and
# training sources as package inputs; only the wrapper and BASE adapters are new.
CODE_NAMES = old.CODE_NAMES
AGGREGATION_POLICY = old.AGGREGATION_POLICY
POSITION_POLICY = old.POSITION_POLICY
WEIGHT_SHA = (read(ROOT / 'manifest.json')['initial_asset']['sha256']
              if (ROOT / 'manifest.json').is_file() else old.WEIGHT_SHA)


def arm_for(protocol, tag):
    need(tag in (TAGS[protocol['exp160_arm']], TAGS[protocol['exp160_arm']] + '_PILOT'),
         'unregistered arm tag')
    return protocol['arm']


def validate_arguments(args, root):
    root = Path(root)
    need(Path(args.protocol).resolve() == root / 'manifest.json', 'wrong manifest path')
    p = validate_manifest(args.protocol, root)
    need(args.tag == TAGS[p['exp160_arm']] + ('_PILOT' if args.phase == 'pilot' else ''),
         'phase/tag mismatch')
    for key, value in p['expected_args'].items():
        need(getattr(args, key, None) == value, 'argument differs: ' + key)
    for key, value in p['paths'].items():
        need(Path(getattr(args, key)).resolve() == within(root, value), 'path differs: ' + key)
    base = old.parent_helper()
    need(all(os.environ.get(k) == v for k, v in base.ENV.items()), 'runtime environment differs')
    state = read(root / 'execution.json')
    need(state.get('status') == 'RUNNING' and state.get('active_phase') == args.phase
         and state.get('manifest_sha256') == sha(args.protocol), 'outer admission absent')
    need(read(root / 'preflight.json')['environment'] == base.remote_environment(),
         'remote environment differs')
    return p, read(root / p['ordered_ids']['path'])


def build_offline_model(arch, asset):
    p = read(ROOT / 'manifest.json')
    if p['exp160_arm'] == 'S43':
        return old.build_offline_model(arch, asset)
    from safetensors.torch import load_file
    from training.src.raptor_common import build_model
    from training.adapters.exp160_base_cpu_pos import install as install_pos
    from training.adapters.exp160_base_patch_mil import install as install_patch
    need(os.environ.get('HF_HUB_OFFLINE') == '1' and arch == p['expected_args']['arch'],
         'offline BASE load required')
    path = ROOT / asset['path']
    need(sha(path) == asset['sha256'], 'BASE weights changed')
    model = build_model(arch, pretrained=False)
    state = load_file(str(path), device='cpu')
    need(set(state) == set(model.backbone.state_dict()), 'BASE backbone keys differ')
    model.backbone.load_state_dict(state, strict=True)
    need(model.backbone.num_features == 768 and tuple(model.backbone.pos_embed.shape) == (1, 1370, 768),
         'BASE position/head width differs')
    install_pos(model.backbone)
    return install_patch(model)


read_rows = old.read_rows
worker_init = old.worker_init
install_gold_blocker = old.install_gold_blocker
validate_membership = old.validate_membership
validate_label_contract = old.validate_label_contract
loss_weights = old.loss_weights
uid_hash = old.uid_hash
sha256 = old.sha256
CODE_NAMES = old.CODE_NAMES
POSITION_POLICY = old.POSITION_POLICY
AGGREGATION_POLICY = old.AGGREGATION_POLICY


def __getattr__(name):
    # The runner records guard.WEIGHT_SHA as the initial asset; it is per arm (BASE has its own weights).
    if name == 'WEIGHT_SHA':
        return read(ROOT / 'manifest.json')['initial_asset']['sha256']
    raise AttributeError(name)


def validate_pretraining(provenance, env, protocol, arm):
    need(provenance['aug1_sha256'] == protocol['files']['training/src/raptor_exp127_aug1.py']['sha256']
         and provenance['aug1_triplet_sha256'] == protocol['files']['training/src/exp158_aug1_triplet.py']['sha256'],
         'AUG1 binding differs')
    need(provenance['initial_asset_sha256'] == protocol['initial_asset']['sha256']
         and provenance['aggregation_policy'] == AGGREGATION_POLICY
         and provenance['train_ids_ordered_sha256'] == protocol['ordered_ids']['train_sha256']
         and provenance['eval_ids_ordered_sha256'] == protocol['ordered_ids']['val_sha256']
         and provenance['gold_access'] == 'disabled', 'pretraining provenance differs')
    need(provenance['runtime']['device'] == 'cuda' and provenance['runtime']['effective_amp'] == 'fp16'
         and provenance['runtime']['policy'] == 'strict', 'runtime differs')
    provenance.update(remote_environment=old.parent_helper().remote_environment(),
                      arm=protocol['exp160_arm'],
                      initial_asset_repo=protocol['initial_asset']['repo'],
                      initial_asset_revision=protocol['initial_asset']['revision'],
                      initial_asset_file_sha256=protocol['initial_asset']['sha256'])


def finish_pilot(out, protocol, provenance, timings, updates, skips, initial_seconds, eval_seconds, peak):
    need(updates == 32 and skips == 0 and len(timings) == 32, 'pilot updates differ')
    projection = 1.3 * sum(timings[8:]) / 24 * 8620 + max(450, 1.3 * eval_seconds * 888 / 4 + initial_seconds + 60)
    write_new(Path(out) / 'receipt.json', {'schema': SCHEMA, 'status': 'PASS', 'phase': 'pilot',
        'protocol_sha256': provenance['protocol_sha256'],
        'initial_state_sha256': provenance['initial_state_sha256'], 'updates': updates, 'skips': skips,
        'step_seconds': timings, 'initial_seconds': initial_seconds, 'eval_seconds': eval_seconds,
        'peak_gpu_allocated_bytes': peak, 'projected_full_seconds': projection,
        'validation_rows': 0, 'argv': [sys.executable, *sys.argv]})


def check_prefix_trace(trace):
    write_new(ROOT / 'artifacts/experiments' / TAGS[read(ROOT / 'manifest.json')['exp160_arm']] / 'trace_prefix.json',
              {'schema': SCHEMA, 'epochs': 8, 'training_trace_sha256': trace})


def validate_completion(out, protocol):
    out = Path(out)
    summary, provenance, history = (read(out / n) for n in ('summary.json', 'provenance.json', 'history.json'))
    need(summary['selected_epoch_1based'] == 20 and summary['optimizer_updates'] == 8620
         and summary['skipped_updates'] == 0 and summary['train'] == 3455 and summary['val'] == 888
         and summary['arm'] == protocol['exp160_arm'] and summary['provenance'] == provenance,
         'full completion differs')
    need(provenance['initial_asset_file_sha256'] == protocol['initial_asset']['sha256']
         and provenance['initial_asset_revision'] == protocol['initial_asset']['revision'],
         'asset provenance differs')
    need(len(history) == 20 and all(h['epoch'] == i and h['optimizer_updates'] == (i + 1) * 431
         and h['skipped_updates'] == 0 and h['oof_auc'] is None and h['gold_auc'] is None
         and math.isfinite(h['loss']) for i, h in enumerate(history)), 'history differs')
    ids = read(ROOT / protocol['ordered_ids']['path'])['val']
    with (out / 'oof_best.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        need(reader.fieldnames == protocol['prediction_columns']
             and [r['StudyInstanceUID'] for r in rows] == ids, 'prediction order differs')
    need(all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1
             for r in rows for t in protocol['targets']), 'invalid probability')
    pilot = read(ROOT / 'artifacts/experiments' / (TAGS[protocol['exp160_arm']] + '_PILOT') / 'receipt.json')
    need(provenance['initial_state_sha256'] == pilot['initial_state_sha256'],
         'pilot/full initial state differs')


def finish_full(out, protocol, provenance, trace, elapsed):
    out = Path(out)
    validate_completion(out, protocol)
    write_new(out / 'trace.json', {'schema': SCHEMA, 'training_trace_sha256': trace,
                                  'epochs': 20, 'updates': 8620, 'skips': 0})
    write_new(out / 'receipt.json', {'schema': SCHEMA, 'status': 'PASS', 'phase': 'full',
        'protocol_sha256': provenance['protocol_sha256'],
        'initial_state_sha256': provenance['initial_state_sha256'], 'updates': 8620, 'skips': 0,
        'validation_rows': 888, 'training_trace_sha256': trace, 'elapsed_seconds': elapsed,
        'argv': [sys.executable, *sys.argv],
        'outputs': {n: {'sha256': sha(out / n), 'bytes': (out / n).stat().st_size} for n in old.OUTPUTS}})


def phase_command(protocol, path, phase):
    command = [sys.executable, str(ROOT / RUNNER), '--protocol', str(Path(path).resolve()),
               '--phase', phase, '--tag', TAGS[protocol['exp160_arm']] + ('_PILOT' if phase == 'pilot' else '')]
    for key, value in {**protocol['expected_args'], **protocol['paths']}.items():
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command.extend((option, str(ROOT / value) if key in protocol['paths'] else str(value)))
    return command


def preflight(path):
    need(sys.platform == 'linux', 'preflight is Linux-only')
    p = validate_manifest(path, ROOT, full=True)
    base = old.parent_helper()
    os.environ.update(base.ENV)
    env = base.remote_environment()
    need(shutil.disk_usage(ROOT).free >= 4 * 2**30 and shutil.disk_usage('/dev/shm').free >= 2 * 2**30,
         'insufficient space')
    write_new(ROOT / 'preflight.json', {'status': 'VERIFIED_INPUTS_ENVIRONMENT',
        'manifest_sha256': sha(path), 'environment': env,
        'file_stats': {n: {'bytes': (ROOT / n).stat().st_size, 'mtime_ns': (ROOT / n).stat().st_mtime_ns}
                       for n in p['files']}})
    print(json.dumps({'status': 'VERIFIED_INPUTS_ENVIRONMENT', 'arm': p['exp160_arm']}))


def run(path):
    need(sys.platform == 'linux' and Path(path).resolve() == ROOT / 'manifest.json',
         'run requires Linux package root')
    p = validate_manifest(path, ROOT)
    adm, pre = read(ROOT / 'admission.json'), read(ROOT / 'preflight.json')
    need(adm.get('status') == 'READY' and adm.get('allow_execution') is True
         and adm.get('manifest_sha256') == sha(path)
         and 0 < adm.get('max_gpu_seconds', 0) <= BUDGET['total_seconds'],
         'B exact-manifest admission missing')
    need(pre['manifest_sha256'] == sha(path) and all(pre['file_stats'][n] ==
         {'bytes': (ROOT / n).stat().st_size, 'mtime_ns': (ROOT / n).stat().st_mtime_ns}
         for n in p['files']), 'package changed after preflight')
    from tools.process_resource_receipt import supervise
    from tools.exp086_dinov2s_v2 import resource_gate
    base = old.parent_helper()
    os.environ.update(base.ENV)
    need(base.remote_environment() == pre['environment'], 'environment changed')
    state = {'status': 'RUNNING', 'manifest_sha256': sha(path), 'retries': 0}
    write_new(ROOT / 'execution.json', state)
    phases, started, failure = [], time.monotonic(), None
    try:
        for phase in ('pilot', 'full'):
            limit = BUDGET[phase + '_seconds']
            out = ROOT / 'artifacts/experiments' / (TAGS[p['exp160_arm']] + ('_PILOT' if phase == 'pilot' else ''))
            need(not out.exists() and not out.with_name(out.name + '_RESOURCE').exists(),
                 'fresh output required')
            need(time.monotonic() - started + limit <= BUDGET['total_seconds'], 'total cap exhausted')
            state['active_phase'] = phase
            (ROOT / 'execution.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
            command = phase_command(p, path, phase)
            resource = supervise(command, out.with_name(out.name + '_RESOURCE'), limit, ROOT)
            resource_gate(resource, command, limit)
            receipt = read(out / 'receipt.json')
            need(receipt['status'] == 'PASS' and receipt['phase'] == phase
                 and receipt['protocol_sha256'] == sha(path) and receipt['argv'] == command
                 and receipt['skips'] == 0, 'phase receipt differs')
            if phase == 'pilot':
                need(receipt['updates'] == 32 and receipt['validation_rows'] == 0
                     and receipt['projected_full_seconds'] <= BUDGET['full_seconds']
                     and receipt['peak_gpu_allocated_bytes'] <= BUDGET['peak_gpu_allocated_cap_bytes'],
                     'pilot gate failed')
            else:
                validate_completion(out, p)
                need(receipt['updates'] == 8620 and receipt['validation_rows'] == 888
                     and set(receipt['outputs']) == set(old.OUTPUTS), 'full receipt differs')
                for n, entry in receipt['outputs'].items():
                    need(sha(out / n) == entry['sha256'] and (out / n).stat().st_size == entry['bytes'],
                         'output differs: ' + n)
            phases.append({'phase': phase, 'receipt_sha256': sha(out / 'receipt.json')})
        need(time.monotonic() - started <= BUDGET['total_seconds'], 'total cap exceeded')
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    state.update(status='INVALID' if failure else 'VALIDATED_SUCCESS', failure=failure,
                 elapsed_seconds=time.monotonic() - started, phases=phases)
    (ROOT / 'execution.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    write_new(ROOT / ('INVALID.json' if failure else 'VALIDATED_SUCCESS.json'), state)
    print(json.dumps(state))
    return 1 if failure else 0


def vast_plan(destination):
    import shlex
    name = str(destination)
    need(name.startswith('/workspace/') and '\\' not in name and ':' not in name,
         'absolute Linux /workspace path required')
    for arm in ARMS:
        d = shlex.quote(name.rstrip('/') + '-' + arm)
        print('cd ' + d)
        print('python tools/exp160_dino_partner.py validate --package . --full')
        print('python tools/exp160_dino_partner.py preflight --protocol manifest.json')
        print("python -c 'import json,hashlib; p=\"manifest.json\"; h=hashlib.sha256(open(p,\"rb\").read()).hexdigest(); json.dump(dict(status=\"READY\",allow_execution=True,manifest_sha256=h,max_gpu_seconds=10800),open(\"admission.json\",\"x\"))'  # B only, after resource claim")
        print('setsid nohup python tools/gpu_queue.py queue.json --foreground > exp160-queue.log 2>&1 < /dev/null & echo $! > exp160-queue.pid')


def collect(export, out):
    export, out = Path(export), Path(out)
    need(not out.exists(), 'fresh collection directory required')
    p = read(export / 'manifest.json')
    need(p['schema'] == SCHEMA and p['exp160_arm'] in ARMS, 'wrong export manifest')
    final = read(export / 'VALIDATED_SUCCESS.json')
    need(final['status'] == 'VALIDATED_SUCCESS' and final['manifest_sha256'] == sha(export / 'manifest.json')
         and [x['phase'] for x in final['phases']] == ['pilot', 'full'], 'remote result not validated')
    tag = TAGS[p['exp160_arm']]
    pilot = export / 'artifacts/experiments' / (tag + '_PILOT')
    full = export / 'artifacts/experiments' / tag
    receipt = read(full / 'receipt.json')
    need(final['phases'][0]['receipt_sha256'] == sha(pilot / 'receipt.json')
         and final['phases'][1]['receipt_sha256'] == sha(full / 'receipt.json')
         and receipt['status'] == 'PASS' and receipt['updates'] == 8620
         and receipt['skips'] == 0 and receipt['protocol_sha256'] == sha(export / 'manifest.json')
         and set(receipt['outputs']) == set(old.OUTPUTS), 'receipt chain differs')
    for n, entry in receipt['outputs'].items():
        need((full / n).stat().st_size == entry['bytes'] and sha(full / n) == entry['sha256'],
             'export output differs: ' + n)
    out.mkdir(parents=True)
    for n in (*old.OUTPUTS, 'receipt.json'):
        shutil.copyfile(full / n, out / n)
    for n in ('manifest.json', 'VALIDATED_SUCCESS.json'):
        shutil.copyfile(export / n, out / n)
    write_new(out / 'collect-receipt.json', {'status': 'COLLECTED_VALIDATED',
        'arm': p['exp160_arm'], 'manifest_sha256': sha(out / 'manifest.json'),
        'prediction_sha256': sha(out / 'oof_best.csv'),
        'checkpoint_sha256': sha(out / 'model_best.pt')})
    print(json.dumps({'status': 'COLLECTED_VALIDATED', 'arm': p['exp160_arm'], 'out': str(out)}))


def _require_collected(path, arm):
    c = read(path / 'collect-receipt.json')
    need(c['status'] == 'COLLECTED_VALIDATED' and c['arm'] == arm
         and c['prediction_sha256'] == sha(path / 'oof_best.csv'), 'collection changed')


def _metrics(y, pred, mask, e):
    import numpy as np
    a = e.auc_targets(y, pred, mask, 1)
    need(np.isfinite(a).all(), 'undefined target AUC')
    return a


def _blend(candidate, partner, refs, indices, weights, e):
    import numpy as np
    cr, pr = e.pct_rank(candidate), e.pct_rank(partner)
    out = {}
    for weight in weights:
        blended = (1 - weight) * pr + weight * cr
        details, draws_by_ref = {}, {}
        for name, (y, mask) in refs.items():
            a, b = _metrics(y, blended, mask, e), _metrics(y, partner, mask, e)
            draws = []
            for ix in indices:
                ai, bi = _metrics(y[ix], blended[ix], mask[ix], e), _metrics(y[ix], partner[ix], mask[ix], e)
                draws.append(float(ai.mean() - bi.mean()))
            draws_by_ref[name] = np.asarray(draws)
            details[name] = {'candidate': float(a.mean()), 'baseline': float(b.mean()),
                             'delta': float(a.mean() - b.mean()),
                             'ci95': np.percentile(draws, [2.5, 97.5]).tolist(),
                             'per_target_delta': (a - b).tolist()}
        mean_draws = (draws_by_ref['jev'] + draws_by_ref['aquila']) / 2
        out[str(weight)] = {'refs': details,
            'mean_delta': (details['jev']['delta'] + details['aquila']['delta']) / 2,
            'mean_ci95': np.percentile(mean_draws, [2.5, 97.5]).tolist()}
    return out


def screen_gate(s43_primary, base_gain, base_primary):
    """The two decisions registered in PROTOCOL.md, without score tuning."""
    return (s43_primary['mean_delta'] > 0 and s43_primary['mean_ci95'][0] > 0,
            base_gain >= .005 and base_primary['mean_ci95'][0] > 0)


def screen(s43, base, out):
    import numpy as np
    from tools import exp115_vlm_eval as e
    s43, base, out = Path(s43), Path(base), Path(out)
    need(not out.exists(), 'fresh screen output required')
    _require_collected(s43, 'S43')
    _require_collected(base, 'BASE')
    ids = read(ROOT / 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json')['val']
    need(len(ids) == 888, 'V888 differs')
    small = e.load_pred(s43 / 'oof_best.csv', ids)
    big = e.load_pred(base / 'oof_best.csv', ids)
    exp158 = e.load_pred(ROOT / '.local/exp158-collected-v1/oof_best.csv', ids)
    partners = {name: e.load_pred(ROOT / path, ids) for name, path in PARTNERS.items()}
    refs = {}
    for name, path in (('jev', JEV), ('aquila', AQUILA)):
        y = e.binarise(e.load_soft(ROOT / path, ids), 'ge0.5')
        need(np.isfinite(y).all(), 'missing reference labels: ' + name)
        refs[name] = (y, np.ones(y.shape, dtype=bool))
    indices = e.bootstrap_index(e.component_groups(ids, ROOT), 300, 20261003)
    standalone = {label: {name: float(_metrics(y, pred, mask, e).mean())
                          for name, (y, mask) in refs.items()}
                  for label, pred in [('S43', small), ('BASE', big), ('EXP158', exp158),
                                      ('RadR50', partners['RadR50'])]}
    s43_blends = {n: _blend(small, p, refs, indices, (.1,), e) for n, p in partners.items()}
    seed_average = (small + exp158) / 2
    average_blends = {n: _blend(seed_average, p, refs, indices, (.1,), e)
                      for n, p in partners.items()}
    base_blends = {n: _blend(big, p, refs, indices, (.1, .2), e)
                   for n, p in partners.items()}
    primary = s43_blends['RadR50']['0.1']
    base_gain = (sum(standalone['BASE'].values()) - sum(standalone['EXP158'].values())) / 2
    base_primary = base_blends['RadR50']['0.1']
    s43_pass, base_pass = screen_gate(primary, base_gain, base_primary)
    base_standalone_deltas = {
        baseline: {name: standalone['BASE'][name] - standalone[baseline][name]
                   for name in ('jev', 'aquila')}
        for baseline in ('EXP158', 'RadR50')}
    result = {'schema': SCHEMA, 'version': VERSION,
        'status': {'S43': 'PASS' if s43_pass else 'FAIL',
                   'BASE': 'ADOPTION_WORTHY' if base_pass else 'NOT_ADOPTION_WORTHY'},
        'standalone': standalone, 'base_standalone_deltas': base_standalone_deltas,
        'base_standalone_gain_vs_EXP158': base_gain,
        's43_blends': s43_blends, 'seed_average_blends': average_blends,
        'base_blends': base_blends, 'bootstrap': {'unit': 'component_id', 'draws': 300, 'seed': 20261003},
        'inputs_sha256': {'S43': sha(s43 / 'oof_best.csv'), 'BASE': sha(base / 'oof_best.csv'),
                         'EXP158': sha(ROOT / '.local/exp158-collected-v1/oof_best.csv'),
                         **{n: sha(ROOT / path) for n, path in PARTNERS.items()}},
        'reference_sha256': {'jev': sha(ROOT / JEV), 'aquila': sha(ROOT / AQUILA)}}
    out.mkdir(parents=True)
    write_new(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 's43_primary': primary['mean_delta'],
                      'base_gain': base_gain, 'base_primary': base_primary['mean_delta']}))


def five_fold_plan(winner):
    need(winner in ARMS, 'winning arm must be S43 or BASE')
    c_ids = read(ROOT / 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json')
    ours = ROOT / 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1'
    fold0 = read(ours / 'fold0/ordered-ids.json')
    need(c_ids['train'] == fold0['train'] and c_ids['val'] == fold0['val'],
         'C fold=2 does not map to OUR fold0')
    folds = []
    seen = set(fold0['val'])
    for f in range(1, 5):
        ids = read(ours / f'fold{f}/ordered-ids.json')
        need(not seen.intersection(ids['val']), 'fold overlap')
        seen.update(ids['val'])
        refs = []
        for seed in (42, 43):
            family = 'EXP149' if seed == 42 and f <= 1 else 'EXP153'
            ref = ROOT / f'artifacts/experiments/{family}_H1_S{seed}_F{f}/oof_k36.csv'
            need(ref.is_file(), 'H1 reference absent: ' + str(ref))
            with ref.open(newline='', encoding='utf-8') as stream:
                uids = [r['StudyInstanceUID'] for r in csv.DictReader(stream)]
            need(uids == ids['val'], 'H1 reference fold membership differs')
            refs.append(str(ref.relative_to(ROOT)).replace('\\', '/'))
        folds.append({'our_fold': f, 'train': len(ids['train']), 'val': len(ids['val']),
                      'ordered_ids': str((ours / f'fold{f}/ordered-ids.json').relative_to(ROOT)).replace('\\', '/'),
                      'h1_k36_seed42_43': refs, 'job': f'{winner} seed {43 if winner == "S43" else 42} fold {f}'})
    print(json.dumps({'status': 'PLAN_ONLY_NO_RUN', 'winner': winner,
                      'mapping': 'C runner fold=2 ordered IDs == OUR fold0 ordered IDs',
                      'remaining_folds': folds,
                      'next_test': 'pooled 5-fold OOF vs EXP153 H1 seed42/43'}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--out', required=True)
    a = sub.add_parser('validate'); a.add_argument('--package', required=True); a.add_argument('--full', action='store_true')
    a = sub.add_parser('vast-stage'); a.add_argument('--overlay', required=True)
    a.add_argument('--source', required=True); a.add_argument('--destination', required=True)
    a = sub.add_parser('vast-plan'); a.add_argument('--destination', default='/workspace/rsna-exp160-dino-partner-v1')
    a = sub.add_parser('preflight'); a.add_argument('--protocol', required=True)
    a = sub.add_parser('run'); a.add_argument('--protocol', required=True)
    a = sub.add_parser('collect'); a.add_argument('--export', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('screen'); a.add_argument('--s43', required=True); a.add_argument('--base', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('five-fold-plan'); a.add_argument('--winner', choices=ARMS, required=True)
    args = ap.parse_args()
    if args.action == 'prepare': prepare(args.out)
    elif args.action == 'validate':
        p = validate_manifest(Path(args.package) / 'manifest.json', args.package, args.full)
        print(json.dumps({'status': 'VALIDATED', 'arm': p['exp160_arm'], 'full': args.full}))
    elif args.action == 'vast-stage': vast_stage(args.overlay, args.source, args.destination)
    elif args.action == 'vast-plan': vast_plan(args.destination)
    elif args.action == 'preflight': preflight(args.protocol)
    elif args.action == 'run': return run(args.protocol)
    elif args.action == 'collect': collect(args.export, args.out)
    elif args.action == 'screen': screen(args.s43, args.base, args.out)
    elif args.action == 'five-fold-plan': five_fold_plan(args.winner)
    return 0


if __name__ == '__main__':
    sys.exit(main())
