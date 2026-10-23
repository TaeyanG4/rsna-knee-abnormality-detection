"""EXP158 v1.0.0: prepare, verify, stage, plan, collect and CPU-screen DINO PATCH.

No command contacts Vast or starts a GPU. B owns instance admission and launch.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / '.local/hc06-illinois-pair-v2/manifest.json'
PARENT_SHA = '4e21fcc33ee7f27468f0e183710110326a2f9a0225037d87ac34395f62f311af'
TAG = 'EXP158_DINO_PATCH_STRONG'
SCHEMA = 'exp158-dino-strong-v1'
INITIAL_SHA = '688756a1dff13683a5d55a60dd667e0bbd88b17df324d2598c5baf3122c41ad5'
TRACE8_SHA = 'ab6b802eb04ed6636f8067eeeaa0c6b4e2c0338c1f81c429d204b31c688d29c7'
FULL_STATE_SHA = '4ea6767e85f8090e16d4238593d8d16809735002de1cb32255a6a7bf3d03f563'
AGGREGATION_POLICY = 'dinov2-normalized-patch256-existing-mil-v1'
POSITION_POLICY = 'canonical518-trainable-cpu-fp32-bicubic-aa-v2'
WEIGHT_SHA = '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
RUNNER = 'training/runners/exp158_dino_strong.py'
PROTOCOL = 'artifacts/experiments/EXP158_DINO_PATCH_STRONG/PROTOCOL.md'
PROTOCOL_BIND = 'registered/EXP158_PROTOCOL.md'
ADD = ('tools/exp158_dino_strong.py', RUNNER, 'training/src/exp158_aug1_triplet.py',
       'training/src/raptor_exp127_aug1.py', 'training/src/raptor_radimagenet.py', PROTOCOL_BIND)
OUTPUTS = ('summary.json', 'provenance.json', 'history.json', 'trace.json', 'trace_prefix.json',
           'model_best.pt', 'oof_best.csv')
BUDGET = {'pilot_seconds': 300, 'full_seconds': 10000, 'total_seconds': 10800,
          'pilot_updates': 32, 'full_updates': 8620, 'retries': 0,
          'peak_gpu_allocated_cap_bytes': 20 * 2**30}
QUEUE = [{'tag': TAG, 'command': ['python', 'tools/exp158_dino_strong.py', 'run',
                                  '--protocol', 'manifest.json']}]
JEV = 'storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv'
AQUILA = 'storage/labels/derived_constellation/aquila_v1.csv'
RAD = 'artifacts/experiments/EXP127_AUG1_S42_F0/oof_k36.csv'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


sha256 = sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def within(root, name):
    need(isinstance(name, str) and name and '\\' not in name and ':' not in name,
         'POSIX relative path required')
    p = PurePosixPath(name)
    need(not p.is_absolute() and p.as_posix() == name and all(x not in ('.', '..') for x in p.parts),
         'unsafe package path')
    target = (Path(root) / name).resolve()
    need(target.is_relative_to(Path(root).resolve()), 'path leaves package')
    return target


def runner_source():
    text = (ROOT / 'training/runners/hc06_patch_mil.py').read_text(encoding='utf-8')
    edits = (
        ('from tools import patch_mil as guard', 'from tools import exp158_dino_strong as guard'),
        ('from raptor_data import StudyWindows, collate',
         'from raptor_data import StudyWindows, collate\n    from exp158_aug1_triplet import Aug1TripletWindows'),
        ('tds = StudyWindows(corpus.dir, train_ids,',
         'tds = Aug1TripletWindows(corpus.dir, train_ids,'),
        ('    guard.validate_pretraining(provenance, run_environment, protocol, arm)',
         '    provenance["aug1_sha256"] = sha256_file(ROOT / "training/src/raptor_exp127_aug1.py")\n'
         '    provenance["aug1_triplet_sha256"] = sha256_file(ROOT / "training/src/exp158_aug1_triplet.py")\n'
         '    guard.validate_pretraining(provenance, run_environment, protocol, arm)'),
        ('        train_s = time.time() - t0',
         '        if ep == 7:\n            guard.check_prefix_trace(trace.hexdigest())\n'
         '        train_s = time.time() - t0'),
    )
    for old, new in edits:
        need(text.count(old) == 1, 'runner template changed: ' + old[:40])
        text = text.replace(old, new)
    return text


def prepare(out):
    out = Path(out)
    need(not out.exists(), 'new overlay directory required')
    need(sha(PARENT) == PARENT_SHA, 'H-C-06 V2 manifest changed')
    parent = read(PARENT)
    args = copy.deepcopy(parent['arms']['PATCH']['expected_args'])
    args.update(epochs=20, fixed_epoch=20, k_eval=36)
    p = {k: copy.deepcopy(v) for k, v in parent.items() if k not in
         ('arms', 'phases', 'evaluation', 'limitations', 'execution', 'files', 'attempt_ancestry')}
    p.update(schema=SCHEMA, status='PREPARED_NOT_LAUNCHED', parent_manifest_sha256=PARENT_SHA,
             expected_args=args, execution=BUDGET, files=copy.deepcopy(parent['files']),
             aggregation_policy=AGGREGATION_POLICY, expected_updates=8620,
             protocol_sha256=sha(ROOT / PROTOCOL),
             augmentation='imported-AUG1-on-adjacent-triplet-training-only',
             evaluation={'gate1': {'jev_min': .8611, 'aquila_min': .8940},
                         'gate2_weights': [.1, .2], 'bootstrap_draws': 300, 'bootstrap_seed': 20261003})
    p['arm'] = copy.deepcopy(parent['arms']['PATCH'])
    p['arm'].update(id='PATCH', tag=TAG, expected_args=args, expected_attempts=8620)
    p['files'].pop('queue.json')
    out.mkdir(parents=True)
    for name in ADD:
        dest = within(out, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name == RUNNER:
            dest.write_text(runner_source(), encoding='utf-8', newline='\n')
        else:
            shutil.copyfile(ROOT / (PROTOCOL if name == PROTOCOL_BIND else name), dest)
        p['files'][name] = {'sha256': sha(dest), 'bytes': dest.stat().st_size, 'payload': True}
    write_new(out / 'queue.json', QUEUE)
    p['files']['queue.json'] = {'sha256': sha(out / 'queue.json'),
                                'bytes': (out / 'queue.json').stat().st_size, 'payload': True}
    write_new(out / 'manifest.json', p)
    print(json.dumps({'status': p['status'], 'overlay': str(out),
                      'manifest_sha256': sha(out / 'manifest.json'), 'overlay_files': [*ADD, 'queue.json']}))


def validate_manifest(path, package=None, full=False):
    path = Path(path)
    root = Path(package or path.parent)
    p = read(path)
    need(p.get('schema') == SCHEMA and p.get('parent_manifest_sha256') == PARENT_SHA,
         'wrong EXP158 manifest')
    need(p.get('protocol_sha256') == p['files'][PROTOCOL_BIND]['sha256'], 'registered protocol binding differs')
    need(p.get('execution') == BUDGET and p.get('expected_updates') == 8620,
         'budget/updates changed')
    args = p['expected_args']
    need(args['epochs'] == args['fixed_epoch'] == 20 and args['k'] == 12 and args['k_eval'] == 36
         and args['seed'] == 42 and args['fold'] == 2 and args['res'] == 224,
         'recipe changed')
    need(p['arm']['expected_args'] == args and p['arm']['initial_state_sha256'] == INITIAL_SHA
         and p['arm']['aggregation_policy'] == AGGREGATION_POLICY
         and p['arm']['expected_attempts'] == 8620 and p['arm']['tag'] == TAG,
         'PATCH arm contract differs')
    need(p['expected_ids']['train'] == 3455 and p['expected_ids']['val'] == 888,
         'study counts changed')
    need(p['full_initial_state']['sha256'] == FULL_STATE_SHA
         and p['full_initial_state']['initial_state_sha256'] == INITIAL_SHA,
         'initial state changed')
    need(p['aggregation_policy'] == AGGREGATION_POLICY, 'aggregation changed')
    need(read(root / 'queue.json') == QUEUE, 'queue changed')
    for name, entry in p['files'].items():
        target = within(root, name)
        if name in ADD or name == 'queue.json' or full:
            need(target.is_file() and target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'],
                 'file differs: ' + name)
    if full:
        ids = read(root / p['ordered_ids']['path'])
        need(len(ids['train']) == 3455 and len(ids['val']) == 888
             and not set(ids['train']) & set(ids['val']), 'split differs')
    return p


def vast_stage(overlay, source, destination):
    need(sys.platform == 'linux', 'vast-stage is Linux-only')
    overlay, source, destination = map(Path, (overlay, source, destination))
    need(not destination.exists(), 'fresh remote destination required')
    p = validate_manifest(overlay / 'manifest.json', overlay)
    parent = read(source / 'manifest.json')
    need(sha(source / 'manifest.json') == PARENT_SHA, 'remote parent manifest differs')
    need(set(parent['files']) <= set(p['files']), 'remote parent inventory differs')
    need(source.resolve().parent == destination.resolve().parent, 'hardlinks require sibling roots on same disk')
    destination.mkdir(mode=0o700)
    for name, entry in p['files'].items():
        dst = within(destination, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if name in ADD or name == 'queue.json':
            overlay_file = within(overlay, name)
            need(not overlay_file.is_symlink(), 'overlay symlink forbidden: ' + name)
            shutil.copyfile(overlay_file, dst)
        else:
            src = within(source, name)
            need(src.is_file() and not src.is_symlink() and src.stat().st_size == entry['bytes'],
                 'parent file absent/symlink/size: ' + name)
            os.link(src, dst)
            need(dst.stat().st_ino == src.stat().st_ino, 'hardlink did not preserve inode')
        need(dst.stat().st_size == entry['bytes'], 'staged size differs: ' + name)
    shutil.copyfile(overlay / 'manifest.json', destination / 'manifest.json')
    validate_manifest(destination / 'manifest.json', destination, full=True)
    write_new(destination / 'stage-receipt.json', {'status': 'STAGED_VERIFIED_NO_GPU',
             'manifest_sha256': sha(destination / 'manifest.json'), 'parent_manifest_sha256': PARENT_SHA,
             'destination': destination.as_posix(), 'files': len(p['files'])})
    print(json.dumps({'status': 'STAGED_VERIFIED_NO_GPU', 'destination': destination.as_posix(),
                      'manifest_sha256': sha(destination / 'manifest.json')}))


# The following guard methods are called by the generated, otherwise frozen HC06 runner.
def parent_helper():
    from tools import vast_hc06
    return vast_hc06


def arm_for(protocol, tag):
    need(tag in (TAG, TAG + '_PILOT'), 'unregistered EXP158 tag')
    return protocol['arm']


def validate_arguments(args, root):
    root = Path(root)
    need(Path(args.protocol).resolve() == root / 'manifest.json', 'wrong manifest path')
    p = validate_manifest(args.protocol, root)
    arm = arm_for(p, args.tag)
    need(args.tag == TAG + ('_PILOT' if args.phase == 'pilot' else ''), 'phase/tag mismatch')
    for key, value in p['expected_args'].items():
        need(getattr(args, key, None) == value, 'argument differs: ' + key)
    for key, value in p['paths'].items():
        need(Path(getattr(args, key)).resolve() == within(root, value), 'path differs: ' + key)
    base = parent_helper()
    need(all(os.environ.get(k) == v for k, v in base.ENV.items()), 'runtime environment differs')
    state = read(root / 'execution.json')
    need(state.get('status') == 'RUNNING' and state.get('active_phase') == args.phase
         and state.get('manifest_sha256') == sha(args.protocol), 'outer run admission absent')
    pre = read(root / 'preflight.json')
    need(pre.get('manifest_sha256') == sha(args.protocol), 'preflight differs')
    need(base.remote_environment() == pre['environment'], 'remote environment differs')
    base.AGGREGATION_POLICY = AGGREGATION_POLICY
    return p, read(root / p['ordered_ids']['path'])


def build_offline_model(arch, asset):
    base = parent_helper()
    base.AGGREGATION_POLICY = AGGREGATION_POLICY
    return base.build_offline_model(arch, asset)


def read_rows(path):
    return parent_helper().read_rows(path)


def worker_init(worker):
    return parent_helper().worker_init(worker)


def install_gold_blocker(root):
    return parent_helper().install_gold_blocker(root)


def validate_membership(*args):
    return parent_helper().validate_membership(*args)


def validate_label_contract(*args):
    return parent_helper().validate_label_contract(*args)


def loss_weights(*args):
    return parent_helper().loss_weights(*args)


def uid_hash(ids):
    return parent_helper().uid_hash(ids)


CODE_NAMES = ('raptor_train.py', 'raptor_common.py', 'raptor_data.py', 'raptor_runtime.py')


def validate_pretraining(provenance, env, protocol, arm):
    base = parent_helper()
    need(provenance.get('aug1_sha256') == protocol['files']['training/src/raptor_exp127_aug1.py']['sha256']
         and provenance.get('aug1_triplet_sha256') == protocol['files']['training/src/exp158_aug1_triplet.py']['sha256'],
         'AUG1 source provenance differs')
    base.AGGREGATION_POLICY = AGGREGATION_POLICY
    base.validate_pretraining(provenance, env, protocol, arm)
    need(provenance['initial_state_sha256'] == INITIAL_SHA, 'initial state differs')


def finish_pilot(out, protocol, provenance, timings, updates, skips, initial_seconds, eval_seconds, peak):
    need(updates == 32 and skips == 0 and len(timings) == 32, 'pilot update count differs')
    steady = sum(timings[8:]) / 24
    projected = 1.3 * steady * 8620 + max(450, 1.3 * eval_seconds * 888 / 4 + initial_seconds + 60)
    write_new(Path(out) / 'receipt.json', {
        'schema': SCHEMA, 'status': 'PASS', 'phase': 'pilot',
        'protocol_sha256': provenance['protocol_sha256'], 'initial_state_sha256': INITIAL_SHA,
        'augmentation_sha256': provenance['aug1_sha256'], 'updates': updates, 'skips': skips,
        'step_seconds': timings, 'initial_seconds': initial_seconds, 'eval_seconds': eval_seconds,
        'peak_gpu_allocated_bytes': peak, 'projected_full_seconds': projected,
        'training_ids_only': True, 'validation_rows': 0, 'eval_train_studies': 4,
        'checkpoints_saved': False, 'predictions_saved': False,
        'argv': [sys.executable, *sys.argv]})


def validate_completion(out, protocol):
    out = Path(out)
    s, provenance, history = (read(out / name) for name in ('summary.json', 'provenance.json', 'history.json'))
    need(s['selected_epoch_1based'] == 20 and s['best_epoch'] == 19
         and s['optimizer_updates'] == 8620 and s['skipped_updates'] == 0
         and s['train'] == 3455 and s['val'] == 888 and s['arm'] == 'PATCH'
         and s['weak_oof_macro_auc'] is None, 'full completion contract differs')
    need(s['provenance'] == provenance and provenance['initial_state_sha256'] == INITIAL_SHA
         and provenance['aggregation_policy'] == AGGREGATION_POLICY,
         'full provenance differs')
    need(provenance['aug1_sha256'] == protocol['files']['training/src/raptor_exp127_aug1.py']['sha256']
         and provenance['aug1_triplet_sha256'] == protocol['files']['training/src/exp158_aug1_triplet.py']['sha256'],
         'full augmentation provenance differs')
    need(len(history) == 20 and all(h['epoch'] == i and h['optimizer_updates'] == (i + 1) * 431
         and h['skipped_updates'] == 0 and h['oof_auc'] is None and h['gold_auc'] is None
         and math.isfinite(h['loss']) for i, h in enumerate(history)), 'fixed final history differs')
    prefix = read(out / 'trace_prefix.json')
    need(prefix == {'schema': SCHEMA, 'epochs': 8, 'training_trace_sha256': TRACE8_SHA},
         '8-epoch training order differs from validated PATCH')
    ids = read(ROOT / protocol['ordered_ids']['path'])['val']
    with (out / 'oof_best.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        need(reader.fieldnames == protocol['prediction_columns']
             and [r['StudyInstanceUID'] for r in rows] == ids, 'prediction order differs')
    need(all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1
             for r in rows for t in protocol['targets']), 'invalid prediction probability')


def finish_full(out, protocol, provenance, trace, elapsed):
    out = Path(out)
    validate_completion(out, protocol)
    need(isinstance(trace, str) and len(trace) == 64, 'missing training trace')
    write_new(out / 'trace.json', {'schema': SCHEMA, 'training_trace_sha256': trace,
                                  'epochs': 20, 'updates': 8620, 'skips': 0})
    write_new(out / 'receipt.json', {'schema': SCHEMA, 'status': 'PASS', 'phase': 'full',
              'protocol_sha256': provenance['protocol_sha256'], 'initial_state_sha256': INITIAL_SHA,
              'augmentation_sha256': provenance['aug1_sha256'], 'updates': 8620, 'skips': 0,
              'validation_rows': 888, 'training_trace_sha256': trace, 'elapsed_seconds': elapsed,
              'argv': [sys.executable, *sys.argv],
              'outputs': {n: {'sha256': sha(out / n), 'bytes': (out / n).stat().st_size}
                          for n in OUTPUTS}})


def check_prefix_trace(trace):
    need(trace == TRACE8_SHA, 'first eight epochs differ from validated PATCH UID order')
    write_new(ROOT / 'artifacts/experiments' / TAG / 'trace_prefix.json',
              {'schema': SCHEMA, 'epochs': 8, 'training_trace_sha256': trace})


def phase_command(protocol, path, phase):
    command = [sys.executable, str(ROOT / RUNNER), '--protocol', str(Path(path).resolve()),
               '--phase', phase, '--tag', TAG + ('_PILOT' if phase == 'pilot' else '')]
    for key, value in {**protocol['expected_args'], **protocol['paths']}.items():
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command.extend((option, str(ROOT / value) if key in protocol['paths'] else str(value)))
    return command


def preflight(path):
    need(sys.platform == 'linux', 'remote preflight is Linux-only')
    p = validate_manifest(path, ROOT, full=True)
    base = parent_helper()
    os.environ.update(base.ENV)
    env = base.remote_environment()
    need(shutil.disk_usage(ROOT).free >= 4 * 2**30, 'insufficient output space')
    need(shutil.disk_usage('/dev/shm').free >= 2 * 2**30, 'insufficient shared memory')
    write_new(ROOT / 'preflight.json', {'status': 'VERIFIED_INPUTS_ENVIRONMENT',
              'manifest_sha256': sha(path), 'environment': env,
              'file_stats': {name: {'bytes': (ROOT / name).stat().st_size,
                                   'mtime_ns': (ROOT / name).stat().st_mtime_ns}
                             for name in p['files']}})
    print(json.dumps({'status': 'VERIFIED_INPUTS_ENVIRONMENT', 'manifest_sha256': sha(path)}))


def run(path):
    need(sys.platform == 'linux', 'remote run is Linux-only')
    path = Path(path)
    need(path.resolve() == ROOT / 'manifest.json', 'run only at package root')
    p = validate_manifest(path, ROOT)
    adm = read(ROOT / 'admission.json')
    need(adm.get('status') == 'READY' and adm.get('allow_execution') is True
         and adm.get('manifest_sha256') == sha(path) and 0 < adm.get('max_gpu_seconds', 0) <= 10800,
         'B exact-manifest admission missing')
    pre = read(ROOT / 'preflight.json')
    need(pre.get('status') == 'VERIFIED_INPUTS_ENVIRONMENT'
         and pre.get('manifest_sha256') == sha(path), 'preflight missing')
    need(all(pre['file_stats'][name] == {'bytes': (ROOT / name).stat().st_size,
              'mtime_ns': (ROOT / name).stat().st_mtime_ns} for name in p['files']),
         'package changed after preflight')
    from tools.process_resource_receipt import supervise
    from tools.exp086_dinov2s_v2 import resource_gate
    base = parent_helper()
    os.environ.update(base.ENV)
    need(base.remote_environment() == pre['environment'], 'environment changed after preflight')
    state = {'status': 'RUNNING', 'manifest_sha256': sha(path), 'retries': 0}
    write_new(ROOT / 'execution.json', state)
    started = time.monotonic()
    phases = []
    failure = None
    try:
        for phase in ('pilot', 'full'):
            limit = BUDGET[phase + '_seconds']
            out = ROOT / 'artifacts/experiments' / (TAG + ('_PILOT' if phase == 'pilot' else ''))
            need(not out.exists() and not out.with_name(out.name + '_RESOURCE').exists(),
                 'fresh phase outputs required; no retry')
            need(time.monotonic() - started + limit <= BUDGET['total_seconds'], 'remaining time cap')
            state['active_phase'] = phase
            (ROOT / 'execution.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
            command = phase_command(p, path, phase)
            resource = supervise(command, out.with_name(out.name + '_RESOURCE'), limit, ROOT)
            resource_gate(resource, command, limit)
            receipt = read(out / 'receipt.json')
            need(receipt.get('schema') == SCHEMA and receipt.get('status') == 'PASS'
                 and receipt.get('phase') == phase and receipt.get('argv') == command
                 and receipt.get('protocol_sha256') == sha(path) and receipt.get('skips') == 0,
                 'phase receipt differs')
            if phase == 'pilot':
                need(receipt['updates'] == 32 and receipt['validation_rows'] == 0
                     and receipt['projected_full_seconds'] <= BUDGET['full_seconds']
                     and receipt['peak_gpu_allocated_bytes'] <= BUDGET['peak_gpu_allocated_cap_bytes'],
                     'pilot gate failed')
            else:
                validate_completion(out, p)
                need(receipt['updates'] == 8620 and receipt['validation_rows'] == 888
                     and set(receipt['outputs']) == set(OUTPUTS), 'full receipt differs')
                for name, item in receipt['outputs'].items():
                    need(sha(out / name) == item['sha256'] and (out / name).stat().st_size == item['bytes'],
                         'full output differs: ' + name)
            phases.append({'phase': phase, 'receipt_sha256': sha(out / 'receipt.json')})
        need(time.monotonic() - started <= BUDGET['total_seconds'], 'aggregate cap exceeded')
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
    q = shlex.quote(name)
    print('cd ' + q)
    print('python tools/exp158_dino_strong.py validate --package . --full')
    print('python tools/exp158_dino_strong.py preflight --protocol manifest.json')
    print("python -c 'import json; p=json.load(open(\"manifest.json\")); "
          "json.dump(dict(status=\"READY\",allow_execution=True,manifest_sha256=__import__(\"hashlib\").sha256(open(\"manifest.json\",\"rb\").read()).hexdigest(),max_gpu_seconds=10800),open(\"admission.json\",\"x\"))'  # B only, after resource claim")
    print('setsid nohup python tools/gpu_queue.py queue.json --foreground '
          '> exp158-queue.log 2>&1 < /dev/null & echo $! > exp158-queue.pid')
    print('cat exp158-queue.pid  # then inspect queue, GPU job and validation markers')


def collect(export, out):
    export, out = Path(export), Path(out)
    need(not out.exists(), 'fresh local collection path required')
    manifest = read(export / 'manifest.json')
    need(manifest['schema'] == SCHEMA and manifest['parent_manifest_sha256'] == PARENT_SHA,
         'export manifest differs')
    final = read(export / 'VALIDATED_SUCCESS.json')
    need(final.get('status') == 'VALIDATED_SUCCESS' and final.get('manifest_sha256') == sha(export / 'manifest.json')
         and len(final.get('phases', [])) == 2, 'remote output not validated')
    full = export / 'artifacts/experiments' / TAG
    pilot = export / 'artifacts/experiments' / (TAG + '_PILOT')
    receipt = read(full / 'receipt.json')
    need([phase.get('phase') for phase in final['phases']] == ['pilot', 'full']
         and final['phases'][0]['receipt_sha256'] == sha(pilot / 'receipt.json')
         and final['phases'][1]['receipt_sha256'] == sha(full / 'receipt.json'),
         'remote phase receipt chain differs')
    need(receipt['status'] == 'PASS' and receipt['updates'] == 8620 and receipt['skips'] == 0
         and receipt['initial_state_sha256'] == INITIAL_SHA
         and receipt['protocol_sha256'] == sha(export / 'manifest.json'), 'full receipt differs')
    need(set(receipt['outputs']) == set(OUTPUTS), 'incomplete output inventory')
    for name, entry in receipt['outputs'].items():
        need((full / name).stat().st_size == entry['bytes'] and sha(full / name) == entry['sha256'],
             'export output differs: ' + name)
    validate_completion(full, manifest)
    out.mkdir(parents=True)
    for name in (*OUTPUTS, 'receipt.json'):
        shutil.copyfile(full / name, out / name)
    for name in ('manifest.json', 'VALIDATED_SUCCESS.json'):
        shutil.copyfile(export / name, out / name)
    write_new(out / 'collect-receipt.json', {'status': 'COLLECTED_VALIDATED',
              'manifest_sha256': sha(out / 'manifest.json'),
              'prediction_sha256': sha(out / 'oof_best.csv'),
              'checkpoint_sha256': sha(out / 'model_best.pt')})
    print(json.dumps({'status': 'COLLECTED_VALIDATED', 'out': str(out)}))


def screen(collected, out):
    """CPU-only gates. All evaluation references stay on the local machine."""
    import numpy as np
    from tools import exp115_vlm_eval as e
    collected, out = Path(collected), Path(out)
    need(not out.exists(), 'fresh screen output required')
    c = read(collected / 'collect-receipt.json')
    need(c['status'] == 'COLLECTED_VALIDATED' and c['prediction_sha256'] == sha(collected / 'oof_best.csv'),
         'collection changed')
    ids = read(ROOT / 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json')['val']
    need(len(ids) == 888, 'V888 differs')
    candidate = e.load_pred(collected / 'oof_best.csv', ids)
    baseline = e.load_pred(ROOT / RAD, ids)
    refs = {}
    for name, path in (('jev', JEV), ('aquila', AQUILA)):
        values = e.load_soft(ROOT / path, ids)
        y = e.binarise(values, 'ge0.5')
        need(np.isfinite(y).all(), 'missing reference labels: ' + name)
        refs[name] = (y, np.ones(y.shape, dtype=bool), 1)
    aucs = {name: e.auc_targets(y, candidate, mask, minimum)
            for name, (y, mask, minimum) in refs.items()}
    need(all(np.isfinite(a).all() for a in aucs.values()), 'undefined reference target AUC')
    scores = {name: e.macro(a) for name, a in aucs.items()}
    need(all(math.isfinite(value) for value in scores.values()), 'undefined 12-target macro')
    passed = scores['jev'] >= .8611 and scores['aquila'] >= .8940
    result = {'schema': SCHEMA, 'status': 'STRENGTH_PASS' if passed else 'STRENGTH_FAIL',
              'standalone': scores, 'threshold': {'jev': .8611, 'aquila': .8940},
              'candidate_sha256': sha(collected / 'oof_best.csv'), 'baseline_sha256': sha(ROOT / RAD),
              'reference_sha256': {'jev': sha(ROOT / JEV), 'aquila': sha(ROOT / AQUILA)},
              'bootstrap': {'unit': 'component_id', 'draws': 300, 'seed': 20261003},
              'blends': None}
    if passed:
        ranks = e.pct_rank(baseline), e.pct_rank(candidate)
        indices = e.bootstrap_index(e.component_groups(ids, ROOT), 300, 20261003)
        result['blends'] = {}
        for weight in (.1, .2):
            blend = (1 - weight) * ranks[0] + weight * ranks[1]
            details = {}
            draw_deltas = {}
            for name, (y, mask, minimum) in refs.items():
                a = e.auc_targets(y, blend, mask, minimum)
                b = e.auc_targets(y, baseline, mask, minimum)
                need(np.isfinite(a).all() and np.isfinite(b).all(), 'undefined blend target AUC')
                deltas = []
                for ix in indices:
                    ai = e.auc_targets(y[ix], blend[ix], mask[ix], minimum)
                    bi = e.auc_targets(y[ix], baseline[ix], mask[ix], minimum)
                    need(np.isfinite(ai).all() and np.isfinite(bi).all(), 'undefined bootstrap target AUC')
                    deltas.append(float(ai.mean() - bi.mean()))
                draw_deltas[name] = np.asarray(deltas)
                details[name] = {'candidate': float(a.mean()), 'baseline': float(b.mean()),
                                 'delta': float(a.mean() - b.mean()),
                                 'ci95': np.percentile(deltas, [2.5, 97.5]).tolist(),
                                 'per_target_delta': (a - b).tolist()}
            draws = (draw_deltas['jev'] + draw_deltas['aquila']) / 2
            delta = (details['jev']['delta'] + details['aquila']['delta']) / 2
            result['blends'][str(weight)] = {'refs': details, 'mean_delta': delta,
                'mean_ci95': np.percentile(draws, [2.5, 97.5]).tolist(), 'pass': delta > 0}
        result['status'] = ('BLEND_PASS' if any(v['pass'] for v in result['blends'].values())
                            else 'BLEND_FAIL')
    out.mkdir(parents=True)
    write_new(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'standalone': scores,
                      'blend_mean_deltas': None if result['blends'] is None else
                      {k: v['mean_delta'] for k, v in result['blends'].items()}}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--out', required=True)
    a = sub.add_parser('validate'); a.add_argument('--package', required=True); a.add_argument('--full', action='store_true')
    a = sub.add_parser('vast-stage'); a.add_argument('--overlay', required=True)
    a.add_argument('--source', required=True); a.add_argument('--destination', required=True)
    a = sub.add_parser('vast-plan'); a.add_argument('--destination', default='/workspace/rsna-exp158-dino-strong-v1')
    a = sub.add_parser('preflight'); a.add_argument('--protocol', required=True)
    a = sub.add_parser('run'); a.add_argument('--protocol', required=True)
    a = sub.add_parser('collect'); a.add_argument('--export', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('screen'); a.add_argument('--collected', required=True); a.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.action == 'prepare': prepare(args.out)
    elif args.action == 'validate':
        p = Path(args.package)
        validate_manifest(p / 'manifest.json', p, full=args.full)
        print(json.dumps({'status': 'VALIDATED', 'full': args.full,
                          'manifest_sha256': sha(p / 'manifest.json')}))
    elif args.action == 'vast-stage': vast_stage(args.overlay, args.source, args.destination)
    elif args.action == 'vast-plan': vast_plan(args.destination)
    elif args.action == 'preflight': preflight(args.protocol)
    elif args.action == 'run': return run(args.protocol)
    elif args.action == 'collect': collect(args.export, args.out)
    elif args.action == 'screen': screen(args.collected, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
