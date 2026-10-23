"""EXP172 v1.1.0: L2-label DINOv2-BASE, serial Vast folds and CPU screen.

prepare/validate/vast-plan/collect/screen never start GPU work. B owns admission
and the detached queue launch. Run as ``python -m tools.exp172_dino_base_l2``.

Changelog:
1.0.2 - Report standalone L2 RadR50 and L2 ConvNeXt-T scores for every
        reference from the existing ranked bundle inputs. Training, ranking,
        bootstrap and admission are unchanged; retain v1.0.0/v1.0.1 manifests.
1.0.1 - Compare Linux receipt paths with PurePosixPath on every host. Keep
        frozen v1.0.0 manifests collectable with their original hash bindings;
        training and the registered screen are unchanged.
1.0.0 - Register the L2 DINOv2-BASE five-fold run and replacement-bundle screen.
Changelog 1.1.0 (2026-10-12): elapsed/projection values are telemetry; no self-imposed run deadline.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp158_dino_strong as old
from tools import exp160_dino_partner as base
from tools import exp162_dino_folds_kaggle as folds

SCHEMA = 'exp172-dino-base-l2-v1'
VERSION = '1.1.0'
# Collect frozen packages using their original version and hash bindings.
MANIFEST_VERSIONS = ('1.0.0', '1.0.1', '1.0.2', VERSION)
EXP = 'artifacts/experiments/EXP172_DINO_BASE_L2'
PROTOCOL = EXP + '/PROTOCOL.md'
SOURCE = '.local/exp160-collected-v1/BASE/manifest.json'
SOURCE_RECEIPT = '.local/exp160-collected-v1/BASE/collect-receipt.json'
RUNNER = 'training/runners/exp172_dino_base_l2.py'
SELF = 'tools/exp172_dino_base_l2.py'
REMOTE_SOURCE = '/workspace/rsna-exp160-dino-partner-v1_1b-BASE'
REMOTE_ROOT = '/workspace/rsna-exp172-dino-base-l2-v1'
LABEL = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv'
LABEL_PROVENANCE = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.provenance.json'
LABEL_SHA = '79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9'
CONTROL_INIT_SHA = 'fecc3ea80e15f851266a1a7228435ce70ff6b1ca0a90ee72b69938b253ed9de5'
CONTROL_HOSTNAME = '44a123d2ba2a'
FOLD0_ROSTER = 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold0/ordered-ids.json'
EXTRA = ('tools/process_resource_observer.py', 'tools/process_resource_receipt.py', 'tools/exp162_dino_folds_kaggle.py', 'tools/exp082_r34_loss.py',
         'tools/exp164_dino_base_folds.py', 'training/src/raptor_radimagenet.py',
         'config/workspace.toml', LABEL, LABEL_PROVENANCE)
WEIGHT = base.WEIGHT_BASE
COUNTS = folds.FOLD_COUNTS
BUDGET = {'pilot_seconds': 300, 'full_seconds': 6000, 'total_seconds': 6300,
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
        return read(p if p.is_file() else ROOT / SOURCE)['initial_asset']['sha256']
    raise AttributeError(name)


def tag(fold):
    need(fold in range(5), 'EXP172 trains OUR folds 0-4 only')
    return f'EXP172_DINO_BASE_L2_S42_F{fold}'


def roster_path(fold):
    tag(fold)
    return (FOLD0_ROSTER if fold == 0 else
            f'artifacts/experiments/EXP162_DINO_FOLDS/rosters/fold{fold}-sorted.json')


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
    result = original.replace(before, 'from tools import exp172_dino_base_l2 as guard')
    compile(result, RUNNER, 'exec')
    return result


def guard_attributes(source):
    names = {node.attr for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
             and node.value.id == 'guard'}
    # Avoid hasattr(module, ...) here: a dynamic __getattr__ can import an unstaged
    # dependency, as happened in EXP167/170. The BASE dynamic asset is explicit.
    missing = names - set(globals()) - {'WEIGHT_SHA'}
    need(not missing, 'runner guard attributes missing: ' + ', '.join(sorted(missing)))
    return sorted(names)


def queue():
    return [{'tag': tag(f), 'command': ['python', '-m', 'tools.exp172_dino_base_l2',
             'run', '--fold', str(f)], 'continue_on_failure': False} for f in range(5)]


def _entry(path):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size, 'payload': True}


def prepare(out, root=ROOT):
    out, root = Path(out), Path(root)
    need(not out.exists(), 'fresh EXP172 overlay required')
    source, receipt = read(root / SOURCE), read(root / SOURCE_RECEIPT)
    need(source['schema'] == base.SCHEMA and source['exp160_arm'] == 'BASE'
         and source['version'] == base.VERSION and receipt['status'] == 'COLLECTED_VALIDATED'
         and receipt['manifest_sha256'] == sha(root / SOURCE), 'collected BASE source differs')
    need((root / PROTOCOL).is_file(), 'registered protocol missing')
    provenance = read(root / LABEL_PROVENANCE)
    need(sha(root / LABEL) == LABEL_SHA and provenance['csv_sha256'] == LABEL_SHA
         and provenance['gold58_included'] is False and provenance['rows'] == 4347,
         'L2 file/provenance differs')
    from training.src.raptor_common import load_label_csv
    _, label_hash = load_label_csv(root / LABEL)
    need(source['initial_asset']['sha256'] == source['files'][WEIGHT]['sha256']
         and source['initial_asset']['bytes'] == source['files'][WEIGHT]['bytes'],
         'BASE weight binding differs')
    need(read(root / '.local/exp160-collected-v1/BASE/provenance.json')
         ['initial_state_sha256'] == CONTROL_INIT_SHA,
         'BASE control initial state differs')
    need(read(root / '.local/exp160-collected-v1/BASE/summary.json')
         ['environment']['hostname'] == CONTROL_HOSTNAME,
         'BASE control host differs')
    need(sha(root / '.local/exp160-overlay-v1_1/BASE' / base.RUNNER) ==
         source['files'][base.RUNNER]['sha256'] and
         sha(root / 'tools/exp160_dino_partner.py') ==
         source['files']['tools/exp160_dino_partner.py']['sha256'],
         'frozen EXP160 runner/guard changed')
    rosters = roster(root)
    need(source['ordered_ids']['val_sha256'] == folds.uid_hash(rosters[0][1]['val'])
         and source['ordered_ids']['train_sha256'] == folds.uid_hash(rosters[0][1]['train']),
         'EXP160 fold 0 mapping differs')
    fold0 = read(root / FOLD0_ROSTER)
    need(set(fold0['train']) == set(rosters[0][1]['train']) and
         set(fold0['val']) == set(rosters[0][1]['val']), 'fold 0 EXP127 roster differs')
    out.mkdir(parents=True)
    p = copy.deepcopy(source)
    p.update(schema=SCHEMA, version=VERSION, status='PREPARED_NOT_LAUNCHED',
             source_manifest_sha256=sha(root / SOURCE), protocol_sha256=sha(root / PROTOCOL),
             execution=BUDGET, fold_protocols={}, serial_folds=list(range(5)),
             source_root=REMOTE_SOURCE, remote_root=REMOTE_ROOT,
             label_file_sha256=LABEL_SHA, label_semantic_sha256=label_hash,
             source_initial_state_sha256=CONTROL_INIT_SHA)
    p['files'].pop('queue.json')
    payload = {SELF: root / SELF, **{n: root / n for n in EXTRA},
               PROTOCOL: root / PROTOCOL, FOLD0_ROSTER: root / FOLD0_ROSTER}
    # A copied overlay must import and validate in place before stage. Pull
    # every frozen Python dependency from the exact EXP160 source overlay.
    source_overlay = root / '.local/exp160-overlay-v1_1/BASE'
    for name, entry in source['files'].items():
        if name.startswith('tools/') and name.endswith('.py'):
            candidates = (source_overlay / name, root / name,
                          root / '.local/hc06-illinois-pair-v2' / Path(name).name)
            src = next((x for x in candidates if x.is_file() and
                        sha(x) == entry['sha256']), None)
            need(src is not None, 'frozen transitive Python source differs: ' + name)
            payload.setdefault(name, src)
    for f in range(1, 5):
        payload[roster_path(f)] = root / roster_path(f)
    for name, src in payload.items():
        dest = within(out, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        p['files'][name] = _entry(dest)
    runner = within(out, RUNNER)
    runner.parent.mkdir(parents=True, exist_ok=True)
    derived = runner_source()
    guard_attributes(derived)
    runner.write_text(derived, encoding='utf-8', newline='\n')
    p['files'][RUNNER] = _entry(runner)
    for f in range(5):
        ids = read(out / roster_path(f))
        train, val = ids['train'], ids['val']
        need(f == 0 or (train == sorted(train) and val == sorted(val)),
             'EXP162 sorted roster differs')
        q = copy.deepcopy(source)
        args = q['expected_args']
        # The frozen BASE runner's fold=2 is historical; its ordered IDs are
        # our EXP099 fold 0. Preserve it for a clean paired control.
        args['fold'] = 2 if f == 0 else f
        q['paths']['labels'] = q['paths']['eval_labels'] = LABEL
        q['label_hash'] = q['eval_label_hash'] = label_hash
        updates = (len(train) // args['bs']) * 20
        q.update(schema=SCHEMA, version=VERSION, fold=f, tag=tag(f),
                 protocol_sha256=p['protocol_sha256'], execution=BUDGET,
                 expected_updates=updates, source_manifest_sha256=p['source_manifest_sha256'],
                 expected_ids={'train': len(train), 'val': len(val)},
                 ordered_ids={'path': roster_path(f), 'sha256': sha(out / roster_path(f)),
                              'train_sha256': folds.uid_hash(train), 'val_sha256': folds.uid_hash(val)},
                 loss={'rule': 'fold-local T float32 L2 soft mean clipped[.03,.7]; (1-prev)/prev clipped[1,10]',
                       'baseline_pos_weight': None},
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
                                {f'protocols/fold{f}.json' for f in range(5)})
    write_new(out / 'manifest.json', p)
    return {'status': p['status'], 'overlay': str(out), 'manifest_sha256': sha(out / 'manifest.json'),
            'fold_protocols': p['fold_protocols']}


def validate_manifest(path, package=None, full=False):
    path = Path(path)
    root = Path(package or path.parent)
    if full:
        need(str(root.resolve()) == REMOTE_ROOT, 'wrong staged remote root')
    p = read(path)
    need(p['schema'] == SCHEMA and p['version'] in MANIFEST_VERSIONS and p['serial_folds'] == list(range(5))
         and p['execution'] == BUDGET and p['protocol_sha256'] == p['files'][PROTOCOL]['sha256']
         and len(p['source_manifest_sha256']) == 64 and p['source_root'] == REMOTE_SOURCE
         and p['remote_root'] == REMOTE_ROOT and p['label_file_sha256'] == LABEL_SHA
         and p['source_initial_state_sha256'] == CONTROL_INIT_SHA
         and len(p['label_semantic_sha256']) == 64,
         'EXP172 manifest contract differs')
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
    need(sha(root / LABEL) == LABEL_SHA and
         p['files'][LABEL]['sha256'] == LABEL_SHA and
         p['files']['config/workspace.toml']['bytes'] > 0 and
         all(name in p['overlay_files'] for name in
             (SELF, RUNNER, 'tools/exp160_dino_partner.py',
              'tools/exp158_dino_strong.py', 'tools/vast_hc06.py',
              'training/src/raptor_radimagenet.py')),
         'L2 label/workspace package differs')
    guard_attributes((root / RUNNER).read_text(encoding='utf-8'))
    rosters = None if full or not (ROOT / folds.PARENT).is_file() else roster(ROOT)
    for f in range(5):
        binding = p['fold_protocols'][str(f)]
        q = read(root / binding['path'])
        ids = read(root / q['ordered_ids']['path'])
        updates = ((4343 - COUNTS[f]) // 8) * 20
        need(binding['sha256'] == sha(root / binding['path']) and q['schema'] == SCHEMA
             and q['fold'] == f and q['tag'] == tag(f) and q['expected_updates'] == updates
             and q['arm']['expected_attempts'] == updates and q['arm']['expected_args'] == q['expected_args']
             and q['expected_args']['fold'] == (2 if f == 0 else f)
             and q['expected_args']['seed'] == 42 and q['expected_args']['epochs'] == q['expected_args']['fixed_epoch'] == 20
             and q['expected_args']['k'] == 12 and q['expected_args']['k_eval'] == 36
             and q['expected_args']['amp'] == q['expected_args']['eval_amp'] == 'fp16'
             and q['expected_args']['bs'] == 8 and q['expected_args']['accum'] == 1
             and q['expected_args']['arch'] == 'vit_base_patch14_dinov2.lvd142m'
             and q['paths']['labels'] == q['paths']['eval_labels'] == LABEL
             and q['label_hash'] == q['eval_label_hash'] == p['label_semantic_sha256']
             and q['arm']['actual_pos_weight'] is None and q['initial_asset'] == p['initial_asset']
             and q['ordered_ids']['sha256'] == sha(root / q['ordered_ids']['path'])
             and (f == 0 or (ids['train'] == sorted(ids['train']) and ids['val'] == sorted(ids['val'])))
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
    need(str(source.resolve()) == REMOTE_SOURCE and str(destination.resolve()) == REMOTE_ROOT
         and not destination.exists() and source.resolve().parent == destination.resolve().parent,
         'exact fresh sibling Vast roots required for hardlinks')
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
    need(q['schema'] == SCHEMA and f in range(5) and
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
         and provenance['initial_state_sha256'] == CONTROL_INIT_SHA
         and provenance['initial_asset_sha256'] == __getattr__('WEIGHT_SHA')
         and provenance['train_ids_ordered_sha256'] == protocol['ordered_ids']['train_sha256']
         and provenance['eval_ids_ordered_sha256'] == protocol['ordered_ids']['val_sha256']
         and provenance['checkpoint_selection'] == 'fixed-final' and
         provenance['corpus_manifest_sha256'] == protocol['corpus_manifest_sha256'],
         'fold BASE provenance differs')
    provenance.update(exp172_fold=protocol['fold'], gpu_family='RTX3090',
                      label_file_sha256=LABEL_SHA,
                      label_semantic_sha256=protocol['label_hash'])


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
    need(summary['fold'] == (2 if f == 0 else f) and
         summary['train'] == 4343-COUNTS[f] and summary['val'] == COUNTS[f]
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
    need(os.uname().nodename == CONTROL_HOSTNAME,
         'EXP160 BASE control was not run on this Vast host')
    need(shutil.disk_usage(ROOT).free >= 4 * 2**30 and
         shutil.disk_usage('/dev/shm').free >= 2 * 2**30, 'insufficient free space')
    write_new(ROOT / 'preflight.json', {'status': 'VERIFIED_INPUTS_ENVIRONMENT',
        'manifest_sha256': sha(ROOT / 'manifest.json'), 'environment': helper.remote_environment(),
        'file_stats': {n: {'bytes': (ROOT / n).stat().st_size,
                          'mtime_ns': (ROOT / n).stat().st_mtime_ns} for n in p['files']}})
    return {'status': 'VERIFIED_INPUTS_ENVIRONMENT', 'manifest_sha256': sha(ROOT / 'manifest.json')}


def run(fold):
    need(sys.platform == 'linux' and str(ROOT.resolve()) == REMOTE_ROOT and fold in range(5),
         'remote fold run requires exact Vast root')
    p = validate_manifest(ROOT / 'manifest.json', ROOT, full=False)
    q = read(ROOT / p['fold_protocols'][str(fold)]['path'])
    admission = read(ROOT / 'admission.json')
    pre = read(ROOT / 'preflight.json')
    need(admission == {'status': 'READY', 'allow_execution': True,
                       'manifest_sha256': sha(ROOT / 'manifest.json'),
                       'serial_folds': list(range(5)), 'max_gpu_seconds_per_fold': 6300},
         'B exact-manifest serial admission missing')
    need(pre['manifest_sha256'] == sha(ROOT / 'manifest.json') and
         all(pre['file_stats'][n] == {'bytes': (ROOT / n).stat().st_size,
                                      'mtime_ns': (ROOT / n).stat().st_mtime_ns}
             for n in p['files']), 'package changed since preflight')
    from tools.process_resource_observer import supervise, resource_gate
    helper = old.parent_helper()
    os.environ.update(helper.ENV)
    need(helper.remote_environment() == pre['environment'], 'environment changed')
    state = {'status': 'RUNNING', 'fold': fold, 'manifest_sha256': sha(ROOT / 'manifest.json'),
             'protocol_sha256': sha(ROOT / p['fold_protocols'][str(fold)]['path']), 'retries': 0}
    write_new(ROOT / f'execution-fold{fold}.json', state)
    phases, started, failure = [], time.monotonic(), None
    try:
        for phase in ('pilot', 'full'):
            out = ROOT / 'artifacts/experiments' / (tag(fold) + ('_PILOT' if phase == 'pilot' else ''))
            need(not out.exists() and not out.with_name(out.name + '_RESOURCE').exists(),
                 'fresh fold output required')
            state['active_phase'] = phase
            (ROOT / f'execution-fold{fold}.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
            command = phase_command(q, phase)
            resource = supervise(command, out.with_name(out.name + '_RESOURCE'), ROOT)
            resource_gate(resource, command)
            receipt = read(out / 'receipt.json')
            need(receipt['schema'] == SCHEMA and receipt['status'] == 'PASS' and
                 receipt['phase'] == phase and receipt['fold'] == fold and
                 receipt['protocol_sha256'] == state['protocol_sha256'] and
                 receipt['argv'] == command and receipt['skips'] == 0,
                 'phase receipt differs')
            if phase == 'pilot':
                need(receipt['updates'] == 32 and receipt['validation_rows'] == 0 and
                     math.isfinite(receipt['projected_full_seconds']) and receipt['projected_full_seconds'] > 0 and
                     receipt['peak_gpu_allocated_bytes'] <= BUDGET['peak_gpu_allocated_cap_bytes'],
                     'invalid pilot measurement or memory gate failed')
            else:
                need(receipt['updates'] == q['expected_updates'] and
                     receipt['validation_rows'] == COUNTS[fold] and
                     set(receipt['outputs']) == set(old.OUTPUTS), 'full receipt differs')
                for name, entry in receipt['outputs'].items():
                    need(_entry(out / name) == entry, 'full output differs: ' + name)
            phases.append({'phase': phase, 'receipt_sha256': sha(out / 'receipt.json')})
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    state.update(status='INVALID' if failure else 'VALIDATED_SUCCESS', failure=failure,
                 elapsed_seconds=time.monotonic()-started, phases=phases)
    (ROOT / f'execution-fold{fold}.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    write_new(ROOT / (f'INVALID-fold{fold}.json' if failure else f'VALIDATED_SUCCESS-fold{fold}.json'), state)
    if not failure:
        try:
            state['export'] = export(fold, Path('/workspace') / f'exp172-export-fold{fold}.tar')
            (ROOT / f'execution-fold{fold}.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        except BaseException as exc:
            print(json.dumps({'fold': fold, 'export_error': f'{type(exc).__name__}: {exc}'}))
            return 1
    print(json.dumps(state))
    return 1 if failure else 0


def vast_plan(overlay='/workspace/exp172-overlay'):
    import shlex
    o = str(overlay)
    need(o.startswith('/workspace/') and '\\' not in o and ':' not in o and
         '..' not in PurePosixPath(o).parts, 'absolute Linux /workspace overlay required')
    o = shlex.quote(o)
    return '\n'.join([
        '# B checks the 3090 is idle and claims Vast compute before launch.',
        f'python {o}/{SELF} validate --package {o}',
        f'python {o}/{SELF} vast-stage --overlay {o}',
        f'cd {REMOTE_ROOT}',
        f'export KNEE_WORKSPACE_ROOT={REMOTE_ROOT} HF_HUB_OFFLINE=1 PYTHONUTF8=1',
        f'python -m tools.exp172_dino_base_l2 validate --package . --full',
        'python -m tools.exp172_dino_base_l2 preflight',
        "python -c 'import hashlib,json; p=\"manifest.json\"; h=hashlib.sha256(open(p,\"rb\").read()).hexdigest(); json.dump(dict(status=\"READY\",allow_execution=True,manifest_sha256=h,serial_folds=[0,1,2,3,4],max_gpu_seconds_per_fold=6300),open(\"admission.json\",\"x\"))'",
        'setsid nohup python tools/gpu_queue.py queue.json --foreground > exp172-queue.log 2>&1 < /dev/null & echo $! > exp172-queue.pid',
        '# Each successful fold writes /workspace/exp172-export-fold{fold}.tar automatically.',
    ])


def _verify_fold(run_dir, fold, protocol, manifest_sha, package=ROOT):
    out = Path(run_dir)
    receipt, summary, provenance, history = (read(out / n) for n in
                                               ('receipt.json', 'summary.json', 'provenance.json', 'history.json'))
    updates = protocol['expected_updates']
    need(receipt['schema'] == SCHEMA and receipt['status'] == 'PASS' and
         receipt['phase'] == 'full' and receipt['fold'] == fold and
         receipt['protocol_sha256'] == manifest_sha and receipt['updates'] == updates and
         receipt['skips'] == 0 and receipt['validation_rows'] == COUNTS[fold] and
         set(receipt['outputs']) == set(old.OUTPUTS), 'fold receipt differs')
    need(summary['fold'] == (2 if fold == 0 else fold) and
         summary['args']['seed'] == 42 and summary['args']['tag'] == tag(fold) and
         summary['args']['labels'] == str(PurePosixPath(REMOTE_ROOT) / LABEL) and
         summary['args']['eval_labels'] == str(PurePosixPath(REMOTE_ROOT) / LABEL) and
         summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == updates and summary['skipped_updates'] == 0 and
         summary['gold_access'] == 'disabled' and summary['weak_oof_macro_auc'] is None and
         summary['train'] == 4343-COUNTS[fold] and summary['val'] == COUNTS[fold] and
         summary['provenance'] == provenance and
         len(history) == 20 and all(h['epoch'] == i and
         h['optimizer_updates'] == (i+1)*(updates//20) and h['skipped_updates'] == 0
         and h['oof_auc'] is None and h['gold_auc'] is None
         for i, h in enumerate(history)), 'fold completion differs')
    need(provenance['gpu_family'] == 'RTX3090' and
         provenance['contract_helper_sha256'] == protocol['files'][SELF]['sha256'] and
         provenance['initial_asset_file_sha256'] == protocol['initial_asset']['sha256'] and
         provenance['initial_asset_revision'] == protocol['initial_asset']['revision'] and
         provenance['aggregation_policy'] == AGGREGATION_POLICY and
         provenance['label_file_sha256'] == LABEL_SHA and
         provenance['label_semantic_sha256'] == protocol['label_hash'] and
         provenance['runtime']['effective_amp'] == 'fp16' and
         provenance['train_ids_ordered_sha256'] == protocol['ordered_ids']['train_sha256'] and
         provenance['eval_ids_ordered_sha256'] == protocol['ordered_ids']['val_sha256'],
         'fold BASE provenance differs')
    for name, entry in receipt['outputs'].items():
        need(_entry(out / name) == entry, 'fold output hash differs: ' + name)
    ids = read(Path(package) / protocol['ordered_ids']['path'])['val']
    folds._prediction_values(out / 'oof_best.csv', ids)
    return receipt


def export(fold, out, root=ROOT):
    """Export one completed fold; `run` calls this before the next queue item."""
    root, out = Path(root), Path(out)
    need(sys.platform == 'linux' and str(root.resolve()) == REMOTE_ROOT and
         fold in range(5) and not out.exists(), 'fresh exact-root fold export required')
    p = validate_manifest(root / 'manifest.json', root, full=False)
    final = read(root / f'VALIDATED_SUCCESS-fold{fold}.json')
    q = read(root / p['fold_protocols'][str(fold)]['path'])
    full = root / 'artifacts/experiments' / tag(fold)
    pilot = root / 'artifacts/experiments' / (tag(fold) + '_PILOT')
    receipt = _verify_fold(full, fold, q, p['fold_protocols'][str(fold)]['sha256'], root)
    need(final['status'] == 'VALIDATED_SUCCESS' and
         final['manifest_sha256'] == sha(root / 'manifest.json') and
         [x['phase'] for x in final['phases']] == ['pilot', 'full'] and
         final['phases'][0]['receipt_sha256'] == sha(pilot / 'receipt.json') and
         final['phases'][1]['receipt_sha256'] == sha(full / 'receipt.json') and
         read(pilot / 'receipt.json')['initial_state_sha256'] == receipt['initial_state_sha256'],
         'fold receipt chain differs')
    with tarfile.open(out, 'x') as tar:
        for item in (root / 'manifest.json', root / f'VALIDATED_SUCCESS-fold{fold}.json',
                     pilot / 'receipt.json', full / 'receipt.json',
                     *(full / name for name in receipt['outputs'])):
            tar.add(item, arcname=item.relative_to(root).as_posix())
    return {'fold': fold, 'path': out.as_posix(), 'sha256': sha(out), 'bytes': out.stat().st_size}


def collect(export, overlay, out):
    export, overlay, out = map(Path, (export, overlay, out))
    need(not out.exists(), 'fresh collection directory required')
    p = validate_manifest(overlay / 'manifest.json', overlay)
    need(export.is_dir(), 'per-fold export directory required')
    out.mkdir(parents=True)
    completed, missing, archives = [], {}, {}
    try:
        for f in range(5):
            archive = export / f'exp172-export-fold{f}.tar'
            if not archive.is_file():
                missing[str(f)] = 'NO_FOLD_EXPORT'
                continue
            full_prefix = f'artifacts/experiments/{tag(f)}'
            pilot_prefix = f'artifacts/experiments/{tag(f)}_PILOT'
            expected = {'manifest.json', f'VALIDATED_SUCCESS-fold{f}.json',
                        f'{pilot_prefix}/receipt.json', f'{full_prefix}/receipt.json'}
            expected.update(f'{full_prefix}/{name}' for name in old.OUTPUTS)
            with tarfile.open(archive, 'r') as tar:
                members = tar.getmembers()
                need(len(members) == len(expected) and
                     {m.name for m in members} == expected and
                     all(m.isfile() and not m.issym() for m in members),
                     'unsafe/incomplete fold export')
                for member in members:
                    target = out / member.name
                    if target.exists():
                        need(member.name == 'manifest.json' and
                             sha(target) == sha(overlay / 'manifest.json'),
                             'duplicate fold export member differs')
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open('xb') as dest:
                        shutil.copyfileobj(src, dest)
            need(sha(out / 'manifest.json') == sha(overlay / 'manifest.json'),
                 'remote manifest differs')
            final = read(out / f'VALIDATED_SUCCESS-fold{f}.json')
            q = read(overlay / p['fold_protocols'][str(f)]['path'])
            full = out / full_prefix
            pilot = out / pilot_prefix
            need(final['status'] == 'VALIDATED_SUCCESS' and final['fold'] == f and
                 final['manifest_sha256'] == sha(out / 'manifest.json') and
                 [x['phase'] for x in final['phases']] == ['pilot', 'full'] and
                 final['phases'][0]['receipt_sha256'] == sha(pilot / 'receipt.json') and
                 final['phases'][1]['receipt_sha256'] == sha(full / 'receipt.json'),
                 'remote phase chain differs')
            pilot_receipt = read(pilot / 'receipt.json')
            receipt = _verify_fold(full, f, q, p['fold_protocols'][str(f)]['sha256'], overlay)
            need(pilot_receipt['status'] == 'PASS' and pilot_receipt['phase'] == 'pilot' and
                 pilot_receipt['protocol_sha256'] == receipt['protocol_sha256'] and
                 pilot_receipt['initial_state_sha256'] == receipt['initial_state_sha256'] and
                 math.isfinite(pilot_receipt['projected_full_seconds']) and pilot_receipt['projected_full_seconds'] > 0,
                 'pilot/full provenance or measurement differs')
            completed.append(f)
            archives[str(f)] = sha(archive)
        result = {'schema': SCHEMA, 'status': 'VERIFIED' if len(completed) == 5 else 'INCOMPLETE',
                  'completed_folds': completed, 'incomplete_folds': missing,
                  'manifest_sha256': sha(overlay / 'manifest.json'),
                  'export_sha256': archives, 'overlay': str(overlay.resolve())}
        write_new(out / 'collection.json', result)
        return result
    except BaseException:
        shutil.rmtree(out)
        raise


def paired_jev_consensus(candidate, baseline, refs, draws, evaluator):
    from tools.exp170_convnext_t_l2 import paired_jev_consensus as helper
    return helper(candidate, baseline, refs, draws, evaluator)


def admission(pair):
    from tools.exp170_convnext_t_l2 import admission as helper
    return helper(pair)


def screen(outputs=ROOT / EXP / 'outputs', destination=ROOT / EXP / 'screen', root=ROOT):
    """Registered paired DINO and replacement-bundle screen; no model execution."""
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp149_label_sources as e149
    from tools import exp156_k_sweep as e156
    from tools import exp169_bundle_screen as bundle_helper
    from tools import exp170_convnext_t_l2 as e170

    outputs, destination, root = map(Path, (outputs, destination, root))
    need(not destination.exists(), 'fresh EXP172 screen directory required')
    collection = read(outputs / 'collection.json')
    need(collection['schema'] == SCHEMA, 'wrong EXP172 collection')
    if collection['status'] != 'VERIFIED':
        result = {'schema': SCHEMA, 'verdict': 'INCOMPLETE',
                  'completed_folds': collection['completed_folds']}
        write_new(destination / 'result.json', result)
        return result
    need(collection['completed_folds'] == list(range(5)) and
         collection['manifest_sha256'] == sha(outputs / 'manifest.json'),
         'EXP172 five-fold collection differs')
    overlay = Path(collection['overlay'])
    p = validate_manifest(overlay / 'manifest.json', overlay)
    canonical, fold_of = e149.fold_rosters(e149._load_manifest('student', root), root)
    ids_by_fold = {f: [uid for uid in canonical if fold_of[uid] == f] for f in range(5)}
    ids = [uid for f in range(5) for uid in ids_by_fold[f]]
    need(len(ids) == len(set(ids)) == 4343, 'pooled EXP099 held-out coverage differs')
    refs_raw = e156.references(ids, root)
    refs = {name: refs_raw[key] for name, key in
            (('jev', 'jev_hybrid'), ('aquila', 'aquila_v1'),
             ('dread', 'dread'), ('consensus', 'consensus'))}
    sections = {name: [] for name in ('l2', 'dino_l2', 'dino_dread', 'convnext')}
    files, hashes, per_fold = {}, {}, {}
    offset = 0
    for f in range(5):
        q = read(overlay / p['fold_protocols'][str(f)]['path'])
        run = outputs / 'artifacts/experiments' / tag(f)
        _verify_fold(run, f, q, p['fold_protocols'][str(f)]['sha256'], overlay)
        control = bundle_helper.paths(root, f)
        conv = (root / 'artifacts/experiments/EXP170_CONVNEXT_T_L2/outputs/results' /
                f'EXP170_CONVNEXT_T_L2_S42_F{f}/oof_k36_flip_tta.csv')
        local = {'l2_s42': control['l2_s42'], 'l2_s43': control['l2_s43'],
                 'dino_l2': run / 'oof_best.csv', 'dino_dread': control['dino'],
                 'convnext': conv}
        fid = ids_by_fold[f]
        values = {}
        for name, path in local.items():
            bundle_helper._verify_inventory(path)
            files[f'{name}_f{f}'] = path.relative_to(root).as_posix()
            hashes[f'{name}_f{f}'] = sha(path)
            values[name] = bundle_helper.load_oof(path, fid)
        sections['l2'].append((ev.pct_rank(values['l2_s42']) +
                               ev.pct_rank(values['l2_s43'])) / 2.0)
        for name in ('dino_l2', 'dino_dread', 'convnext'):
            sections[name].append(ev.pct_rank(values[name]))
        sl = slice(offset, offset + len(fid))
        fold_refs = {name: (r[0][sl], r[1][sl], r[2]) for name, r in refs.items()}
        per_fold[str(f)] = {}
        for name, ref in fold_refs.items():
            a = folds._score(values['dino_l2'], ref, ev)['macro']
            b = folds._score(values['dino_dread'], ref, ev)['macro']
            per_fold[str(f)][name] = {'l2': a, 'dread': b,
                                      'delta': a-b if a is not None and b is not None else None}
        offset = sl.stop
    pooled = {name: np.concatenate(parts) for name, parts in sections.items()}
    prior = read(root / 'artifacts/experiments/EXP170_CONVNEXT_T_L2/screen/result.json')
    need(prior['status']['status'] == 'ADMIT', 'D-B-098 control was not admitted')
    for f in range(5):
        for current, old_name in (('l2_s42', 'l2_s42'), ('l2_s43', 'l2_s43'),
                                  ('dino_dread', 'dino'), ('convnext', 'convnext')):
            need(hashes[f'{current}_f{f}'] == prior['input_sha256'][f'{old_name}_f{f}'],
                 'D-B-098 control OOF differs: ' + old_name + ' fold ' + str(f))
    draws = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    direct = paired_jev_consensus(pooled['dino_l2'], pooled['dino_dread'],
                                  refs, draws, ev)
    old_bundle, old_ranked = bundle_helper.bundle_of({
        'l2': pooled['l2'], 'dino': pooled['dino_dread'],
        'convnext': pooled['convnext']})
    new_bundle, new_ranked = bundle_helper.bundle_of({
        'l2': pooled['l2'], 'dino': pooled['dino_l2'],
        'convnext': pooled['convnext']})
    pair = paired_jev_consensus(new_bundle, old_bundle, refs, draws, ev)
    result = {'schema': SCHEMA, 'version': VERSION, 'status': admission(pair),
              'studies': len(ids), 'folds': list(range(5)), 'seed': 42,
              'protocol_sha256': sha(root / PROTOCOL),
              'bootstrap': {'unit': 'validation component', 'draws': 300,
                            'seed': 20261003},
              'gate': 'new bundle minus D-B-098 bundle: mean(JEV,consensus)>0, CI95 lower>0, Dread>=-0.003',
              'aquila_policy': 'self-referenced to L2; report only',
              'oof_policy': {'l2_s42': 'plain oof_k36.csv', 'l2_s43': 'plain oof_k36.csv',
                             'dino_l2': 'plain oof_best.csv',
                             'dino_dread': 'plain oof_best.csv',
                             'convnext': 'flip TTA oof_k36_flip_tta.csv'},
              'aggregation': 'within-fold target ranks; L2 seed mean; pooled global family ranks; equal thirds; global bundle rerank',
              'input_files': files, 'input_sha256': hashes,
              'direct_l2_minus_dread': direct, 'direct_per_fold': per_fold,
              'new_bundle_minus_d_b_098': pair,
              'new_bundle_minus_l2': paired_jev_consensus(
                  new_bundle, new_ranked['l2'], refs, draws, ev),
              'standalone': {name: {r: folds._score(value, ref, ev) for r, ref in refs.items()}
                             for name, value in {'dino_l2': new_ranked['dino'],
                                                 'dino_dread': old_ranked['dino'],
                                                 'l2': new_ranked['l2'],
                                                 'convnext': new_ranked['convnext'],
                                                 'old_bundle': old_bundle,
                                                 'new_bundle': new_bundle}.items()},
              'reference_sha256': {name: sha(root / path) for name, path in
                   (('jev', e156.JEV), ('aquila', e156.AQUILA), ('dread', e156.DREAD),
                    *[(f'consensus_{i}', x) for i, x in enumerate(e156.CONSENSUS)])}}
    write_new(destination / 'result.json', result)
    return {'status': result['status']['status'], 'result': str(destination / 'result.json'),
            'mean_delta': pair['primary_mean_delta'], 'ci95': pair['primary_mean_ci95'],
            'dread_delta': pair['references']['dread']['delta']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('validate'); s.add_argument('--package', type=Path, required=True)
    s.add_argument('--full', action='store_true')
    s = sub.add_parser('vast-stage'); s.add_argument('--overlay', type=Path, required=True)
    s.add_argument('--source', type=Path, default=Path(REMOTE_SOURCE))
    s.add_argument('--destination', type=Path, default=Path(REMOTE_ROOT))
    s = sub.add_parser('vast-plan'); s.add_argument('--overlay', default='/workspace/exp172-overlay')
    sub.add_parser('preflight')
    s = sub.add_parser('run'); s.add_argument('--fold', type=int, required=True)
    s = sub.add_parser('export'); s.add_argument('--fold', type=int, required=True)
    s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('collect'); s.add_argument('--export', type=Path, required=True)
    s.add_argument('--overlay', type=Path, required=True); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('screen'); s.add_argument('--outputs', type=Path, default=ROOT / EXP / 'outputs')
    s.add_argument('--destination', type=Path, default=ROOT / EXP / 'screen')
    a = ap.parse_args(argv)
    if a.action == 'prepare': result = prepare(a.out)
    elif a.action == 'validate': result = {'status': 'VALIDATED', 'full': a.full,
                                           'folds': validate_manifest(a.package / 'manifest.json', a.package, a.full)['serial_folds']}
    elif a.action == 'vast-stage': result = vast_stage(a.overlay, a.source, a.destination)
    elif a.action == 'vast-plan': result = vast_plan(a.overlay)
    elif a.action == 'preflight': result = preflight()
    elif a.action == 'run': return run(a.fold)
    elif a.action == 'export': result = export(a.fold, a.out)
    elif a.action == 'collect': result = collect(a.export, a.overlay, a.out)
    else: result = screen(a.outputs, a.destination)
    print(result if isinstance(result, str) else json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
