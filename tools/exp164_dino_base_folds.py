"""EXP164 v1.0.0: staged, serial Vast DINOv2-BASE folds and CPU OOF screen.

prepare/validate/vast-plan/collect/screen never start GPU work. B owns admission
and the detached queue launch. Run as ``python -m tools.exp164_dino_base_folds``.
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
    sys.path.insert(0, str(ROOT))
from tools import exp158_dino_strong as old
from tools import exp160_dino_partner as base
from tools import exp162_dino_folds_kaggle as folds

SCHEMA = 'exp164-dino-base-folds-v1'
VERSION = '1.0.0'
EXP = 'artifacts/experiments/EXP164_DINO_BASE_FOLDS'
PROTOCOL = EXP + '/PROTOCOL.md'
SOURCE = '.local/exp160-collected-v1/BASE/manifest.json'
SOURCE_RECEIPT = '.local/exp160-collected-v1/BASE/collect-receipt.json'
RUNNER = 'training/runners/exp164_dino_base_fold.py'
SELF = 'tools/exp164_dino_base_folds.py'
EXTRA = ('tools/exp162_dino_folds_kaggle.py', 'tools/exp082_r34_loss.py')
WEIGHT = base.WEIGHT_BASE
COUNTS = folds.FOLD_COUNTS
BUDGET = {'pilot_seconds': 300, 'full_seconds': 10000, 'total_seconds': 10800,
          'pilot_updates': 32, 'retries': 0, 'peak_gpu_allocated_cap_bytes': 20 * 2**30}
TARGETS, UID = folds.TARGETS, folds.UID
_active_protocol = None
need, sha, read, write_new, within = old.need, old.sha, old.read, old.write_new, old.within
sha256 = old.sha256
CODE_NAMES = old.CODE_NAMES
POSITION_POLICY = old.POSITION_POLICY
AGGREGATION_POLICY = old.AGGREGATION_POLICY


def __getattr__(name):
    if name == 'WEIGHT_SHA':
        p = ROOT / 'manifest.json'
        return read(p)['initial_asset']['sha256'] if p.is_file() else read(ROOT / SOURCE)['initial_asset']['sha256']
    raise AttributeError(name)


def tag(fold):
    need(fold in range(1, 5), 'EXP164 trains OUR folds 1-4 only')
    return f'EXP164_DINO_BASE_S42_F{fold}'


def roster_path(fold):
    return f'artifacts/experiments/EXP162_DINO_FOLDS/rosters/fold{fold}-sorted.json'


def roster(root=ROOT):
    result = folds.roster(root)
    for fold in range(1, 5):
        ids = result[fold][1]
        path = Path(root) / roster_path(fold)
        expected = (json.dumps({'train': sorted(ids['train']), 'val': sorted(ids['val'])}, indent=1) + '\n').encode()
        need(path.is_file() and path.read_bytes() == expected, 'EXP162 sorted roster differs')
    return result


def runner_source():
    original = (ROOT / '.local/exp160-overlay-v1_1/BASE' / base.RUNNER).read_text(encoding='utf-8')
    before = 'from tools import exp160_dino_partner as guard'
    need(original.count(before) == 1, 'EXP160 BASE runner import changed')
    return original.replace(before, 'from tools import exp164_dino_base_folds as guard')


def queue():
    return [{'tag': tag(f), 'command': ['python', '-m', 'tools.exp164_dino_base_folds',
             'run', '--fold', str(f)]} for f in range(1, 5)]


def _entry(path):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size, 'payload': True}


def prepare(out, root=ROOT):
    out, root = Path(out), Path(root)
    need(not out.exists(), 'fresh EXP164 overlay required')
    source, receipt = read(root / SOURCE), read(root / SOURCE_RECEIPT)
    need(source['schema'] == base.SCHEMA and source['exp160_arm'] == 'BASE'
         and source['version'] == base.VERSION and receipt['status'] == 'COLLECTED_VALIDATED'
         and receipt['manifest_sha256'] == sha(root / SOURCE), 'collected BASE source differs')
    need((root / PROTOCOL).is_file(), 'registered protocol missing')
    need(source['initial_asset']['sha256'] == source['files'][WEIGHT]['sha256']
         and source['initial_asset']['bytes'] == source['files'][WEIGHT]['bytes'],
         'BASE weight binding differs')
    need(sha(root / '.local/exp160-overlay-v1_1/BASE' / base.RUNNER) ==
         source['files'][base.RUNNER]['sha256'] and
         sha(root / 'tools/exp160_dino_partner.py') ==
         source['files']['tools/exp160_dino_partner.py']['sha256'],
         'frozen EXP160 runner/guard changed')
    rosters = roster(root)
    need(source['ordered_ids']['val_sha256'] == folds.uid_hash(rosters[0][1]['val'])
         and source['ordered_ids']['train_sha256'] == folds.uid_hash(rosters[0][1]['train']),
         'EXP160 fold 0 mapping differs')
    out.mkdir(parents=True)
    p = copy.deepcopy(source)
    p.update(schema=SCHEMA, version=VERSION, status='PREPARED_NOT_LAUNCHED',
             source_manifest_sha256=sha(root / SOURCE), protocol_sha256=sha(root / PROTOCOL),
             execution=BUDGET, fold_protocols={}, serial_folds=[1, 2, 3, 4])
    p['files'].pop('queue.json')
    payload = {SELF: root / SELF, **{n: root / n for n in EXTRA}, PROTOCOL: root / PROTOCOL}
    for f in range(1, 5):
        payload[roster_path(f)] = root / roster_path(f)
    for name, src in payload.items():
        dest = within(out, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        p['files'][name] = _entry(dest)
    runner = within(out, RUNNER)
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(runner_source(), encoding='utf-8', newline='\n')
    p['files'][RUNNER] = _entry(runner)
    for f in range(1, 5):
        train, val = sorted(rosters[f][1]['train']), sorted(rosters[f][1]['val'])
        q = copy.deepcopy(source)
        args = q['expected_args']
        args['fold'] = f
        updates = (len(train) // args['bs']) * 20
        q.update(schema=SCHEMA, version=VERSION, fold=f, tag=tag(f),
                 protocol_sha256=p['protocol_sha256'], execution=BUDGET,
                 expected_updates=updates, source_manifest_sha256=p['source_manifest_sha256'],
                 expected_ids={'train': len(train), 'val': len(val)},
                 ordered_ids={'path': roster_path(f), 'sha256': sha(root / roster_path(f)),
                              'train_sha256': folds.uid_hash(train), 'val_sha256': folds.uid_hash(val)},
                 loss={'rule': 'fold-local T float32 soft mean clipped[.03,.7]; (1-prev)/prev clipped[1,10]'},
                 files=copy.deepcopy(p['files']))
        q['arm'].update(tag=tag(f), expected_args=copy.deepcopy(args), expected_attempts=updates,
                        actual_pos_weight=None)
        name = f'protocols/fold{f}.json'
        dest = within(out, name)
        write_new(dest, q)
        p['files'][name] = _entry(dest)
        p['fold_protocols'][str(f)] = {'path': name, 'sha256': sha(dest), 'updates': updates,
                                       'train': len(train), 'val': len(val)}
    write_new(out / 'queue.json', queue())
    p['files']['queue.json'] = _entry(out / 'queue.json')
    p['overlay_files'] = sorted(set(payload) | {RUNNER, 'queue.json'} |
                                {f'protocols/fold{f}.json' for f in range(1, 5)})
    write_new(out / 'manifest.json', p)
    return {'status': p['status'], 'overlay': str(out), 'manifest_sha256': sha(out / 'manifest.json'),
            'fold_protocols': p['fold_protocols']}


def validate_manifest(path, package=None, full=False):
    path = Path(path)
    root = Path(package or path.parent)
    p = read(path)
    need(p['schema'] == SCHEMA and p['version'] == VERSION and p['serial_folds'] == [1, 2, 3, 4]
         and p['execution'] == BUDGET and p['protocol_sha256'] == p['files'][PROTOCOL]['sha256']
         and len(p['source_manifest_sha256']) == 64,
         'EXP164 manifest contract differs')
    if (ROOT / SOURCE_RECEIPT).is_file():
        need(p['source_manifest_sha256'] == read(ROOT / SOURCE_RECEIPT)['manifest_sha256'],
             'collected source binding differs')
    need(read(root / 'queue.json') == queue(), 'serial queue differs')
    need(p['initial_asset']['sha256'] == p['files'][WEIGHT]['sha256']
         and p['initial_asset']['bytes'] == p['files'][WEIGHT]['bytes'], 'BASE asset differs')
    for name, entry in p['files'].items():
        target = within(root, name)
        if name in p['overlay_files'] or full:
            need(target.is_file() and not target.is_symlink() and target.stat().st_size == entry['bytes']
                 and sha(target) == entry['sha256'], 'package file differs: ' + name)
    rosters = None if full or not (ROOT / folds.PARENT).is_file() else roster(ROOT)
    for f in range(1, 5):
        binding = p['fold_protocols'][str(f)]
        q = read(root / binding['path'])
        ids = read(root / q['ordered_ids']['path'])
        updates = ((4343 - COUNTS[f]) // 8) * 20
        need(binding['sha256'] == sha(root / binding['path']) and q['schema'] == SCHEMA
             and q['fold'] == f and q['tag'] == tag(f) and q['expected_updates'] == updates
             and q['arm']['expected_attempts'] == updates and q['arm']['expected_args'] == q['expected_args']
             and q['expected_args']['fold'] == f
             and q['expected_args']['seed'] == 42 and q['expected_args']['epochs'] == q['expected_args']['fixed_epoch'] == 20
             and q['expected_args']['k'] == 12 and q['expected_args']['k_eval'] == 36
             and q['expected_args']['amp'] == q['expected_args']['eval_amp'] == 'fp16'
             and q['expected_args']['bs'] == 8 and q['expected_args']['accum'] == 1
             and q['expected_args']['arch'] == 'vit_base_patch14_dinov2.lvd142m'
             and q['arm']['actual_pos_weight'] is None and q['initial_asset'] == p['initial_asset']
             and q['ordered_ids']['sha256'] == sha(root / q['ordered_ids']['path'])
             and ids['train'] == sorted(ids['train']) and ids['val'] == sorted(ids['val'])
             and len(ids['train']) == 4343-COUNTS[f] and len(ids['val']) == COUNTS[f]
             and not set(ids['train']) & set(ids['val'])
             and q['ordered_ids']['train_sha256'] == folds.uid_hash(ids['train'])
             and q['ordered_ids']['val_sha256'] == folds.uid_hash(ids['val']),
             'fold protocol/roster differs: ' + str(f))
        if rosters is not None:
            need(set(ids['train']) == set(rosters[f][1]['train']) and
                 set(ids['val']) == set(rosters[f][1]['val']), 'local fold membership differs')
    return p


def vast_stage(overlay, source, destination):
    need(sys.platform == 'linux', 'vast-stage is Linux-only')
    overlay, source, destination = map(Path, (overlay, source, destination))
    need(not destination.exists() and source.resolve().parent == destination.resolve().parent,
         'fresh sibling destination required for hardlinks')
    p = validate_manifest(overlay / 'manifest.json', overlay)
    need(sha(source / 'manifest.json') == p['source_manifest_sha256'], 'remote BASE package differs')
    old_files = read(source / 'manifest.json')['files']
    destination.mkdir(mode=0o700)
    for name, entry in p['files'].items():
        target = within(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in old_files and old_files[name] == entry:
            src = within(source, name)
            need(src.is_file() and not src.is_symlink(), 'remote source missing: ' + name)
            os.link(src, target)
            need(src.stat().st_ino == target.stat().st_ino, 'hardlink failed: ' + name)
        else:
            src = within(overlay, name)
            need(src.is_file() and not src.is_symlink(), 'overlay payload missing: ' + name)
            shutil.copyfile(src, target)
        need(target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'],
             'staged file differs: ' + name)
    write_new(destination / 'manifest.json', p)
    validate_manifest(destination / 'manifest.json', destination, full=True)
    write_new(destination / 'stage-receipt.json', {'status': 'STAGED_VERIFIED_NO_GPU',
              'manifest_sha256': sha(destination / 'manifest.json'), 'source_manifest_sha256': p['source_manifest_sha256']})
    return {'status': 'STAGED_VERIFIED_NO_GPU', 'destination': destination.as_posix(),
            'manifest_sha256': sha(destination / 'manifest.json')}


# Frozen runner guard interface. Keep CUDA imports inside the explicit remote run.
read_rows = old.read_rows
worker_init = old.worker_init
install_gold_blocker = old.install_gold_blocker
validate_membership = old.validate_membership
uid_hash = old.uid_hash


def validate_label_contract(label_hash, eval_hash, split_hash, protocol):
    need(label_hash == eval_hash == protocol['label_hash'] and split_hash == protocol['split_hash'],
         'label/split semantic hashes differ')


def loss_weights(baseline, protocol, arm):
    import numpy as np
    need(baseline.dtype == np.float32 and baseline.shape == (12,) and np.isfinite(baseline).all()
         and ((baseline >= 1) & (baseline <= 10)).all()
         and arm['loss_policy'] == 'clipped_positive_term'
         and arm['actual_pos_weight'] is None, 'fold-local T loss weights invalid')
    return baseline


def arm_for(protocol, name):
    need(name in (tag(protocol['fold']), tag(protocol['fold']) + '_PILOT'), 'wrong fold tag')
    return protocol['arm']


def validate_arguments(args, root):
    global _active_protocol
    root = Path(root)
    q = read(args.protocol)
    f = q['fold']
    need(q['schema'] == SCHEMA and f in range(1, 5) and
         Path(args.protocol).resolve() == (root / f'protocols/fold{f}.json').resolve() and
         args.tag == tag(f) + ('_PILOT' if args.phase == 'pilot' else ''), 'wrong fold protocol')
    for key, value in q['expected_args'].items():
        need(getattr(args, key, None) == value, 'argument differs: ' + key)
    for key, value in q['paths'].items():
        need(Path(getattr(args, key)).resolve() == within(root, value), 'input path differs: ' + key)
    helper = old.parent_helper()
    helper.AGGREGATION_POLICY = AGGREGATION_POLICY
    need(all(os.environ.get(k) == v for k, v in helper.ENV.items()), 'runtime environment differs')
    state = read(root / f'execution-fold{f}.json')
    need(state['status'] == 'RUNNING' and state['active_phase'] == args.phase and
         state['protocol_sha256'] == sha(args.protocol), 'outer admission absent')
    need(read(root / 'preflight.json')['manifest_sha256'] == sha(root / 'manifest.json')
         and helper.remote_environment() == read(root / 'preflight.json')['environment'],
         'remote environment differs')
    _active_protocol = q
    return q, read(root / q['ordered_ids']['path'])


def build_offline_model(arch, asset):
    need(_active_protocol is not None, 'fold admission required before model build')
    return base.build_offline_model(arch, asset)


def validate_pretraining(provenance, env, protocol, arm):
    base.validate_pretraining(provenance, env, protocol, arm)
    need(provenance['protocol_sha256'] == sha(ROOT / f"protocols/fold{protocol['fold']}.json")
         and provenance['initial_asset_sha256'] == __getattr__('WEIGHT_SHA')
         and provenance['train_ids_ordered_sha256'] == protocol['ordered_ids']['train_sha256']
         and provenance['eval_ids_ordered_sha256'] == protocol['ordered_ids']['val_sha256']
         and provenance['checkpoint_selection'] == 'fixed-final' and
         provenance['corpus_manifest_sha256'] == protocol['corpus_manifest_sha256'],
         'fold BASE provenance differs')
    provenance.update(exp164_fold=protocol['fold'], gpu_family='RTX3090')


def finish_pilot(out, protocol, provenance, timings, updates, skips, initial_seconds, eval_seconds, peak):
    need(updates == 32 and skips == 0 and len(timings) == 32, 'pilot updates differ')
    projected = 1.3 * sum(timings[8:]) / 24 * protocol['expected_updates'] + max(
        450, 1.3 * eval_seconds * protocol['expected_ids']['val'] / 4 + initial_seconds + 60)
    write_new(Path(out) / 'receipt.json', {'schema': SCHEMA, 'status': 'PASS', 'phase': 'pilot',
        'fold': protocol['fold'], 'protocol_sha256': provenance['protocol_sha256'],
        'initial_state_sha256': provenance['initial_state_sha256'], 'updates': updates, 'skips': skips,
        'step_seconds': timings, 'initial_seconds': initial_seconds, 'eval_seconds': eval_seconds,
        'peak_gpu_allocated_bytes': peak, 'projected_full_seconds': projected,
        'validation_rows': 0, 'argv': [sys.executable, *sys.argv]})


def check_prefix_trace(trace):
    need(_active_protocol is not None and len(trace) == 64, 'fold trace missing')
    write_new(ROOT / 'artifacts/experiments' / tag(_active_protocol['fold']) / 'trace_prefix.json',
              {'schema': SCHEMA, 'epochs': 8, 'training_trace_sha256': trace})


def finish_full(out, protocol, provenance, trace, elapsed):
    out = Path(out)
    f, updates = protocol['fold'], protocol['expected_updates']
    summary, history = read(out / 'summary.json'), read(out / 'history.json')
    need(summary['fold'] == f and summary['train'] == 4343-COUNTS[f] and summary['val'] == COUNTS[f]
         and summary['selected_epoch_1based'] == 20 and summary['optimizer_updates'] == updates
         and summary['skipped_updates'] == 0 and summary['gold_access'] == 'disabled'
         and summary['weak_oof_macro_auc'] is None and summary['provenance'] == provenance
         and len(history) == 20 and all(h['epoch'] == i and
         h['optimizer_updates'] == (i+1)*(updates//20) and h['skipped_updates'] == 0
         and h['oof_auc'] is None and h['gold_auc'] is None and math.isfinite(h['loss'])
         for i, h in enumerate(history)), 'incomplete fixed-final fold')
    ids = read(ROOT / protocol['ordered_ids']['path'])['val']
    folds._prediction_values(out / 'oof_best.csv', ids)
    pilot = read(ROOT / 'artifacts/experiments' / (tag(f) + '_PILOT') / 'receipt.json')
    need(provenance['initial_state_sha256'] == pilot['initial_state_sha256'] and len(trace) == 64
         and read(out / 'trace_prefix.json')['training_trace_sha256'], 'pilot/full state or trace differs')
    write_new(out / 'trace.json', {'schema': SCHEMA, 'fold': f, 'epochs': 20,
              'updates': updates, 'training_trace_sha256': trace})
    write_new(out / 'receipt.json', {'schema': SCHEMA, 'status': 'PASS', 'phase': 'full',
        'fold': f, 'protocol_sha256': sha(ROOT / f'protocols/fold{f}.json'),
        'initial_state_sha256': provenance['initial_state_sha256'], 'updates': updates,
        'skips': 0, 'validation_rows': COUNTS[f], 'elapsed_seconds': elapsed,
        'argv': [sys.executable, *sys.argv],
        'outputs': {n: _entry(out / n) for n in old.OUTPUTS}})


def phase_command(protocol, phase):
    f = protocol['fold']
    command = [sys.executable, str(ROOT / RUNNER), '--protocol',
               str(ROOT / f'protocols/fold{f}.json'), '--phase', phase,
               '--tag', tag(f) + ('_PILOT' if phase == 'pilot' else '')]
    for key, value in {**protocol['expected_args'], **protocol['paths']}.items():
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command.extend((option, str(ROOT / value) if key in protocol['paths'] else str(value)))
    return command


def preflight():
    need(sys.platform == 'linux', 'preflight is Linux-only')
    p = validate_manifest(ROOT / 'manifest.json', ROOT, full=True)
    helper = old.parent_helper()
    os.environ.update(helper.ENV)
    need(shutil.disk_usage(ROOT).free >= 4 * 2**30 and
         shutil.disk_usage('/dev/shm').free >= 2 * 2**30, 'insufficient free space')
    write_new(ROOT / 'preflight.json', {'status': 'VERIFIED_INPUTS_ENVIRONMENT',
        'manifest_sha256': sha(ROOT / 'manifest.json'), 'environment': helper.remote_environment(),
        'file_stats': {n: {'bytes': (ROOT / n).stat().st_size,
                          'mtime_ns': (ROOT / n).stat().st_mtime_ns} for n in p['files']}})
    return {'status': 'VERIFIED_INPUTS_ENVIRONMENT', 'manifest_sha256': sha(ROOT / 'manifest.json')}


def run(fold):
    need(sys.platform == 'linux' and fold in range(1, 5), 'remote fold run requires Linux')
    p = validate_manifest(ROOT / 'manifest.json', ROOT, full=False)
    q = read(ROOT / p['fold_protocols'][str(fold)]['path'])
    admission = read(ROOT / 'admission.json')
    pre = read(ROOT / 'preflight.json')
    need(admission == {'status': 'READY', 'allow_execution': True,
                       'manifest_sha256': sha(ROOT / 'manifest.json'),
                       'serial_folds': [1, 2, 3, 4], 'max_gpu_seconds_per_fold': 10800},
         'B exact-manifest serial admission missing')
    need(pre['manifest_sha256'] == sha(ROOT / 'manifest.json') and
         all(pre['file_stats'][n] == {'bytes': (ROOT / n).stat().st_size,
                                      'mtime_ns': (ROOT / n).stat().st_mtime_ns}
             for n in p['files']), 'package changed since preflight')
    from tools.process_resource_receipt import supervise
    from tools.exp086_dinov2s_v2 import resource_gate
    helper = old.parent_helper()
    os.environ.update(helper.ENV)
    need(helper.remote_environment() == pre['environment'], 'environment changed')
    state = {'status': 'RUNNING', 'fold': fold, 'manifest_sha256': sha(ROOT / 'manifest.json'),
             'protocol_sha256': sha(ROOT / p['fold_protocols'][str(fold)]['path']), 'retries': 0}
    write_new(ROOT / f'execution-fold{fold}.json', state)
    phases, started, failure = [], time.monotonic(), None
    try:
        for phase in ('pilot', 'full'):
            limit = BUDGET[phase + '_seconds']
            out = ROOT / 'artifacts/experiments' / (tag(fold) + ('_PILOT' if phase == 'pilot' else ''))
            need(not out.exists() and not out.with_name(out.name + '_RESOURCE').exists(),
                 'fresh fold output required')
            need(time.monotonic() - started + limit <= BUDGET['total_seconds'],
                 'single-fold cap exhausted')
            state['active_phase'] = phase
            (ROOT / f'execution-fold{fold}.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
            command = phase_command(q, phase)
            resource = supervise(command, out.with_name(out.name + '_RESOURCE'), limit, ROOT)
            resource_gate(resource, command, limit)
            receipt = read(out / 'receipt.json')
            need(receipt['schema'] == SCHEMA and receipt['status'] == 'PASS' and
                 receipt['phase'] == phase and receipt['fold'] == fold and
                 receipt['protocol_sha256'] == state['protocol_sha256'] and
                 receipt['argv'] == command and receipt['skips'] == 0,
                 'phase receipt differs')
            if phase == 'pilot':
                need(receipt['updates'] == 32 and receipt['validation_rows'] == 0 and
                     receipt['projected_full_seconds'] <= BUDGET['full_seconds'] and
                     receipt['peak_gpu_allocated_bytes'] <= BUDGET['peak_gpu_allocated_cap_bytes'],
                     'single-job pilot projection or memory gate failed')
            else:
                need(receipt['updates'] == q['expected_updates'] and
                     receipt['validation_rows'] == COUNTS[fold] and
                     set(receipt['outputs']) == set(old.OUTPUTS), 'full receipt differs')
                for name, entry in receipt['outputs'].items():
                    need(_entry(out / name) == entry, 'full output differs: ' + name)
            phases.append({'phase': phase, 'receipt_sha256': sha(out / 'receipt.json')})
        need(time.monotonic() - started <= BUDGET['total_seconds'], 'fold wall cap exceeded')
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    state.update(status='INVALID' if failure else 'VALIDATED_SUCCESS', failure=failure,
                 elapsed_seconds=time.monotonic()-started, phases=phases)
    (ROOT / f'execution-fold{fold}.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    write_new(ROOT / (f'INVALID-fold{fold}.json' if failure else f'VALIDATED_SUCCESS-fold{fold}.json'), state)
    print(json.dumps(state))
    return 1 if failure else 0


def vast_plan(destination='/workspace/rsna-exp164-dino-base-folds-v1'):
    import shlex
    d = str(destination)
    need(d.startswith('/workspace/') and '\\' not in d and ':' not in d and '..' not in Path(d).parts,
         'absolute Linux /workspace destination required')
    d = shlex.quote(d)
    return '\n'.join([
        f'cd {d}',
        'python -m tools.exp164_dino_base_folds validate --package . --full',
        'python -m tools.exp164_dino_base_folds preflight',
        "python -c 'import hashlib,json; p=\"manifest.json\"; h=hashlib.sha256(open(p,\"rb\").read()).hexdigest(); json.dump(dict(status=\"READY\",allow_execution=True,manifest_sha256=h,serial_folds=[1,2,3,4],max_gpu_seconds_per_fold=10800),open(\"admission.json\",\"x\"))'",
        'setsid nohup python tools/gpu_queue.py queue.json --foreground > exp164-queue.log 2>&1 < /dev/null & echo $! > exp164-queue.pid',
    ])


def _verify_fold(run_dir, fold, protocol, manifest_sha):
    out = Path(run_dir)
    receipt, summary, provenance, history = (read(out / n) for n in
                                               ('receipt.json', 'summary.json', 'provenance.json', 'history.json'))
    updates = protocol['expected_updates']
    need(receipt['schema'] == SCHEMA and receipt['status'] == 'PASS' and
         receipt['phase'] == 'full' and receipt['fold'] == fold and
         receipt['protocol_sha256'] == manifest_sha and receipt['updates'] == updates and
         receipt['skips'] == 0 and receipt['validation_rows'] == COUNTS[fold] and
         set(receipt['outputs']) == set(old.OUTPUTS), 'fold receipt differs')
    need(summary['fold'] == fold and summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == updates and summary['skipped_updates'] == 0 and
         summary['gold_access'] == 'disabled' and summary['weak_oof_macro_auc'] is None and
         summary['train'] == 4343-COUNTS[fold] and summary['val'] == COUNTS[fold] and
         summary['provenance'] == provenance and
         len(history) == 20 and all(h['epoch'] == i and
         h['optimizer_updates'] == (i+1)*(updates//20) and h['skipped_updates'] == 0
         and h['oof_auc'] is None and h['gold_auc'] is None
         for i, h in enumerate(history)), 'fold completion differs')
    need(provenance['gpu_family'] == 'RTX3090' and
         provenance['initial_asset_file_sha256'] == protocol['initial_asset']['sha256'] and
         provenance['initial_asset_revision'] == protocol['initial_asset']['revision'] and
         provenance['aggregation_policy'] == AGGREGATION_POLICY and
         provenance['runtime']['effective_amp'] == 'fp16' and
         provenance['train_ids_ordered_sha256'] == protocol['ordered_ids']['train_sha256'] and
         provenance['eval_ids_ordered_sha256'] == protocol['ordered_ids']['val_sha256'],
         'fold BASE provenance differs')
    for name, entry in receipt['outputs'].items():
        need(_entry(out / name) == entry, 'fold output hash differs: ' + name)
    ids = read(ROOT / protocol['ordered_ids']['path'])['val']
    folds._prediction_values(out / 'oof_best.csv', ids)
    return receipt


def collect(export, overlay, out):
    export, overlay, out = map(Path, (export, overlay, out))
    need(not out.exists(), 'fresh collection directory required')
    p = validate_manifest(overlay / 'manifest.json', overlay)
    need(sha(export / 'manifest.json') == sha(overlay / 'manifest.json'), 'remote manifest differs')
    out.mkdir(parents=True)
    completed, missing = [], {}
    for f in range(1, 5):
        final_path = export / f'VALIDATED_SUCCESS-fold{f}.json'
        if not final_path.is_file():
            missing[str(f)] = 'NO_VALIDATED_SUCCESS'
            continue
        final = read(final_path)
        q = read(overlay / p['fold_protocols'][str(f)]['path'])
        full = export / 'artifacts/experiments' / tag(f)
        pilot = export / 'artifacts/experiments' / (tag(f) + '_PILOT')
        need(final['status'] == 'VALIDATED_SUCCESS' and final['fold'] == f and
             final['manifest_sha256'] == sha(export / 'manifest.json') and
             [x['phase'] for x in final['phases']] == ['pilot', 'full'] and
             final['phases'][0]['receipt_sha256'] == sha(pilot / 'receipt.json') and
             final['phases'][1]['receipt_sha256'] == sha(full / 'receipt.json'),
             'remote phase chain differs')
        pilot_receipt = read(pilot / 'receipt.json')
        receipt = _verify_fold(full, f, q, p['fold_protocols'][str(f)]['sha256'])
        need(pilot_receipt['status'] == 'PASS' and pilot_receipt['phase'] == 'pilot' and
             pilot_receipt['protocol_sha256'] == receipt['protocol_sha256'] and
             pilot_receipt['initial_state_sha256'] == receipt['initial_state_sha256'] and
             pilot_receipt['projected_full_seconds'] <= BUDGET['full_seconds'],
             'pilot/full provenance or projection differs')
        dest = out / tag(f)
        dest.mkdir()
        for name in (*receipt['outputs'], 'receipt.json'):
            shutil.copyfile(full / name, dest / name)
        shutil.copyfile(pilot / 'receipt.json', out / f'pilot-receipt-fold{f}.json')
        shutil.copyfile(final_path, out / final_path.name)
        completed.append(f)
    shutil.copyfile(export / 'manifest.json', out / 'manifest.json')
    result = {'schema': SCHEMA, 'status': 'VERIFIED' if len(completed) == 4 else 'INCOMPLETE',
              'completed_folds': completed, 'incomplete_folds': missing,
              'manifest_sha256': sha(overlay / 'manifest.json')}
    write_new(out / 'collection.json', result)
    return result


def screen_gate(pairings):
    primary = pairings['h1']['0.1']
    return bool(primary['primary_mean_delta'] > 0 and primary['primary_mean_ci95'][0] > 0)


def rank_per_fold(parts, evaluator):
    import numpy as np
    need(set(parts) == set(range(5)), 'five fold predictions required for ranks')
    return np.concatenate([evaluator.pct_rank(parts[f]) for f in range(5)])


def screen(outputs=ROOT / EXP / 'outputs', destination=ROOT / EXP / 'screen', root=ROOT):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp156_k_sweep as e156
    outputs, destination, root = map(Path, (outputs, destination, root))
    need(not destination.exists(), 'fresh screen output required')
    collection = read(outputs / 'collection.json')
    need(collection['schema'] == SCHEMA, 'wrong EXP164 collection')
    if collection['status'] != 'VERIFIED':
        result = {'schema': SCHEMA, 'verdict': 'INCOMPLETE', **collection}
        write_new(destination / 'result.json', result)
        return result
    rosters = roster(root)
    ids_by_fold = {f: sorted(rosters[f][1]['val']) if f else rosters[0][1]['val'] for f in range(5)}
    all_ids = [u for f in range(5) for u in ids_by_fold[f]]
    need(len(all_ids) == len(set(all_ids)) == 4343, 'pooled held-out IDs differ')
    baseline = root / '.local/exp160-collected-v1/BASE'
    receipt = read(baseline / 'collect-receipt.json')
    original = read(baseline / 'manifest.json')
    old_summary = read(baseline / 'summary.json')
    old_provenance = read(baseline / 'provenance.json')
    old_history = read(baseline / 'history.json')
    need(receipt['status'] == 'COLLECTED_VALIDATED' and receipt['arm'] == 'BASE' and
         receipt['manifest_sha256'] == sha(baseline / 'manifest.json') and
         receipt['prediction_sha256'] == sha(baseline / 'oof_best.csv') and
         original['initial_asset']['sha256'] == old_provenance['initial_asset_file_sha256'] and
         old_summary['selected_epoch_1based'] == 20 and old_summary['optimizer_updates'] == 8620 and
         old_summary['skipped_updates'] == 0 and len(old_history) == 20,
         'EXP160 BASE fold 0 changed')
    prediction = {0: folds._prediction_values(baseline / 'oof_best.csv', ids_by_fold[0])}
    hashes = {'dino_f0': sha(baseline / 'oof_best.csv')}
    for f in range(1, 5):
        path = outputs / tag(f) / 'oof_best.csv'
        prediction[f] = folds._prediction_values(path, ids_by_fold[f])
        hashes[f'dino_f{f}'] = sha(path)
    dino = np.concatenate([prediction[f] for f in range(5)])
    partners = {}
    for name in ('radr50', 'h1'):
        parts = {}
        for f in range(5):
            experiment = (f'EXP127_AUG1_S42_F0' if f == 0 else f'EXP134_AUG1_S42_F{f}') if name == 'radr50' else (
                         f'EXP149_H1_S42_F{f}' if f < 2 else f'EXP153_H1_S42_F{f}')
            path = root / 'artifacts/experiments' / experiment / 'oof_k36.csv'
            inv = read(path.parent / 'output-inventory.json')['oof_k36.csv']
            need(path.stat().st_size == inv['bytes'] and sha(path) == inv['sha256'],
                 'partner OOF inventory differs: ' + experiment)
            parts[f] = folds._prediction_values(path, ids_by_fold[f])
            hashes[f'{name}_f{f}'] = sha(path)
        partners[name] = parts
    refs_raw = e156.references(all_ids, root)
    refs = {'jev': refs_raw['jev_hybrid'], 'aquila': refs_raw['aquila_v1'],
            'dread': refs_raw['dread'], 'consensus': refs_raw['consensus']}
    standalone = {'pooled': {n: folds._score(dino, r, ev) for n, r in refs.items()}, 'per_fold': {}}
    offset = 0
    for f in range(5):
        sl = slice(offset, offset + len(ids_by_fold[f]))
        standalone['per_fold'][str(f)] = {n: folds._score(prediction[f], (r[0][sl], r[1][sl], r[2]), ev)
                                          for n, r in refs.items()}
        offset = sl.stop
    destination.mkdir(parents=True)
    with (destination / 'pooled_oof.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([UID, *TARGETS])
        for u, row in zip(all_ids, dino):
            writer.writerow([u, *(format(float(v), '.9g') for v in row)])
    draws = ev.bootstrap_index(ev.component_groups(all_ids, root), 300, 20261003)
    pairings = {}
    for name, parts in partners.items():
        ranked_dino = rank_per_fold(prediction, ev)
        ranked_partner = rank_per_fold(parts, ev)
        pairings[name] = {str(w): folds._paired((1-w)*ranked_partner + w*ranked_dino,
                                                ranked_partner, refs, draws, ev) for w in (.1, .2)}
    primary = pairings['h1']['0.1']
    result = {'schema': SCHEMA, 'version': VERSION, 'verdict': 'PASS' if screen_gate(pairings) else 'FAIL',
              'rows': len(all_ids), 'seed': 42, 'folds': list(range(5)),
              'bootstrap': {'draws': 300, 'seed': 20261003, 'unit': 'validation component'},
              'standalone': standalone, 'partners': pairings, 'input_sha256': hashes,
              'pooled_oof_sha256': sha(destination / 'pooled_oof.csv'),
              'reference_sha256': {n: sha(root / path) for n, path in
                  (('jev', folds.JEV), ('aquila', folds.AQUILA), ('dread', folds.DREAD),
                   *[(f'consensus_{i}', x) for i, x in enumerate(folds.CONSENSUS)])}}
    write_new(destination / 'result.json', result)
    return {'verdict': result['verdict'], 'rows': len(all_ids),
            'h1_primary_mean_delta': primary['primary_mean_delta'],
            'h1_primary_mean_ci95': primary['primary_mean_ci95'],
            'result': str(destination / 'result.json')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('validate'); s.add_argument('--package', type=Path, required=True)
    s.add_argument('--full', action='store_true')
    s = sub.add_parser('vast-stage'); s.add_argument('--overlay', type=Path, required=True)
    s.add_argument('--source', type=Path, required=True); s.add_argument('--destination', type=Path, required=True)
    s = sub.add_parser('vast-plan'); s.add_argument('--destination', default='/workspace/rsna-exp164-dino-base-folds-v1')
    sub.add_parser('preflight')
    s = sub.add_parser('run'); s.add_argument('--fold', type=int, required=True)
    s = sub.add_parser('collect'); s.add_argument('--export', type=Path, required=True)
    s.add_argument('--overlay', type=Path, required=True); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('screen'); s.add_argument('--outputs', type=Path, default=ROOT / EXP / 'outputs')
    s.add_argument('--destination', type=Path, default=ROOT / EXP / 'screen')
    a = ap.parse_args(argv)
    if a.action == 'prepare': result = prepare(a.out)
    elif a.action == 'validate': result = {'status': 'VALIDATED', 'full': a.full,
                                           'folds': validate_manifest(a.package / 'manifest.json', a.package, a.full)['serial_folds']}
    elif a.action == 'vast-stage': result = vast_stage(a.overlay, a.source, a.destination)
    elif a.action == 'vast-plan': result = vast_plan(a.destination)
    elif a.action == 'preflight': result = preflight()
    elif a.action == 'run': return run(a.fold)
    elif a.action == 'collect': result = collect(a.export, a.overlay, a.out)
    else: result = screen(a.outputs, a.destination)
    print(result if isinstance(result, str) else json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
