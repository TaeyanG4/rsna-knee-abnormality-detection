"""EXP183: serial S42 mixed-scale confirmation on folds 1-4; CPU packaging/screen.

No command implicitly starts a GPU run. ``run`` is admitted only on the exact
staged Vast root, after an independently reviewed manifest is authorized.
Changelog 1.1.0 (2026-10-12): elapsed/projection values are telemetry; no self-imposed run deadline.
"""
from __future__ import annotations

import argparse
import copy
from functools import partial
import math
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import exp179_multiscale as prior
from tools import exp170_convnext_t_l2 as base

VERSION = '1.1.0'
SCHEMA = 'exp183-multiscale-5fold-v1'
EXP = 'artifacts/experiments/EXP183_MULTISCALE_5FOLD'
TOOL = 'tools/exp183_multiscale_5fold.py'
PROTOCOL = EXP + '/PROTOCOL.md'
RUNNER = 'training/runners/exp183_multiscale_5fold.py'
VERIFICATION = 'verification.md'
SOURCE_ROOT = '/workspace/exp179'
REMOTE_ROOT = '/workspace/exp183'
REMOTE_RESULTS = '/workspace/exp183-results'
FOLDS = (1, 2, 3, 4)
ARM_CAP = 4050
TOTAL_CAP = 16200
PILOT_CAP = 300
BASE_OVERLAY = '.local/exp179-overlay'
CONTROL_OVERLAY = '.local/exp170-overlay-v1'

need, read, sha, write, safe = prior.need, prior.read, prior.sha, prior.write, prior.safe


def __getattr__(name):
    return getattr(prior, name)


def tag(fold):
    need(fold in FOLDS, 'EXP183 fold must be 1-4')
    return f'EXP183_M42_F{fold}'


def fold_from_tag(value):
    found = [f for f in FOLDS if tag(f) == value]
    need(len(found) == 1, 'unknown EXP183 tag')
    return found[0]


def dataset_factory(value):
    from raptor_exp179 import ScaleSliceWindows
    fold_from_tag(value)
    return partial(ScaleSliceWindows, crop_policy='mixed')


def evaluate_flip_tta(model, corpus, ids, labels, a, dev, eval_dtype, autocast):
    from tools import exp166_convnext_t as old
    from training.src import raptor_radimagenet as module
    original = module.CenterSliceWindows
    module.CenterSliceWindows = dataset_factory(a.tag)
    try:
        return old.evaluate_flip_tta(model, corpus, ids, labels, a, dev, eval_dtype, autocast)
    finally:
        module.CenterSliceWindows = original


def stamp_outputs(out, protocol):
    prior.stamp_outputs(out, protocol)
    for name in ('summary.json', 'provenance.json'):
        path = Path(out) / name
        value = read(path)
        value['exp183'] = protocol['exp183']
        write(path, value)


def runner_source(root=ROOT):
    source = (Path(root) / BASE_OVERLAY / prior.RUNNER).read_text(encoding='utf-8')
    before = 'from tools import exp179_multiscale as guard'
    need(source.count(before) == 1, 'EXP179 runner anchor drift')
    source = source.replace(before, 'from tools import exp183_multiscale_5fold as guard')
    compile(source, RUNNER, 'exec')
    return source


def queue():
    return [{'tag': tag(f), 'command': ['python', TOOL, 'run', '--fold', str(f)],
             'continue_on_failure': False} for f in FOLDS]


def check_verification(path, root=ROOT):
    note = Path(path).read_text(encoding='utf-8')
    need('Verdict: PASS' in note and 'Verdict: FAIL' not in note,
         'independent verification PASS note required')
    for name in (TOOL, PROTOCOL, prior.TOOL, prior.LOADER):
        need(sha(Path(root) / name) in note, 'verification lacks exact hash: ' + name)
    return sha(path)


def controls(root=ROOT):
    root = Path(root)
    overlay = root / CONTROL_OVERLAY
    collection = read(root / 'artifacts/experiments/EXP170_CONVNEXT_T_L2/outputs/collection.json')
    need(collection['status'] == 'VERIFIED' and
         collection['manifest_sha256'] == sha(overlay / 'manifest.json'),
         'EXP170 collection/manifest differs')
    result = {}
    for fold in FOLDS:
        p = root / 'artifacts/experiments/EXP170_CONVNEXT_T_L2/outputs/results' / base.tag(fold)
        hashes = base.verify_run(p, fold, overlay)
        receipt = read(root / 'artifacts/experiments/EXP170_CONVNEXT_T_L2/outputs' /
                       f'validated-fold{fold}.json')
        need(receipt['status'] == 'VALIDATED_SUCCESS' and
             receipt['output_sha256'] == hashes, 'EXP170 control receipt differs')
        result[str(fold)] = {'path': str(p.relative_to(root)), 'hashes': hashes,
                             'environment': read(p / 'provenance.json')['environment'],
                             'hostname': read(p / 'summary.json')['environment']['hostname'],
                             'args': read(p / 'summary.json')['args'],
                             'initial_state_sha256': read(p / 'provenance.json')['initial_state_sha256']}
    need(len({json.dumps(v['environment'], sort_keys=True) for v in result.values()}) == 1 and
         len({v['hostname'] for v in result.values()}) == 1,
         'EXP170 controls did not share an environment')
    first = prior.control_receipt(root)
    need(all(v['environment'] == first['provenance']['environment'] and
             v['hostname'] == first['summary']['environment']['hostname']
             for v in result.values()), 'EXP170 folds differ from EXP179 fold-0 3090')
    return result


def prepare(out, verification, root=ROOT):
    root, out = Path(root), Path(out)
    need(not out.exists(), 'fresh overlay required')
    verification_sha = check_verification(verification, root)
    old = prior.validate(root / BASE_OVERLAY)
    base.validate(root / CONTROL_OVERLAY)
    control = controls(root)
    source = runner_source(root)
    out.mkdir(parents=True)
    m = {'schema': SCHEMA, 'version': VERSION, 'status': 'PREPARED_NOT_LAUNCHED',
         'source_root': SOURCE_ROOT, 'remote_root': REMOTE_ROOT,
         'source_manifest_sha256': sha(root / BASE_OVERLAY / 'manifest.json'),
         'source_bundle_sha256': old['source_bundle_sha256'],
         'protocol_sha256': sha(root / PROTOCOL), 'verification_sha256': verification_sha,
         'verification_note': str(Path(verification).resolve()), 'controls': control,
         'caps_seconds': {'arm': ARM_CAP, 'total': TOTAL_CAP, 'pilot': PILOT_CAP},
         'files': {name: {**entry, 'origin': 'exp179'} for name, entry in old['files'].items()},
         'folds': {}}
    payload = {TOOL: root / TOOL, PROTOCOL: root / PROTOCOL,
               VERIFICATION: Path(verification)}
    for fold in FOLDS:
        q0 = read(root / CONTROL_OVERLAY / base.protocol_path(fold))
        info = q0['folds'][str(fold)]
        for name in (base.roster_path(fold), info['groups']['path']):
            payload[name] = root / CONTROL_OVERLAY / name
    for name, path in payload.items():
        safe(name)
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        m['files'][name] = base._entry(dest, 'overlay')
    runner = out / RUNNER
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(source, encoding='utf-8', newline='\n')
    m['files'][RUNNER] = base._entry(runner, 'overlay')
    for fold in FOLDS:
        q = copy.deepcopy(read(root / CONTROL_OVERLAY / base.protocol_path(fold)))
        info = q['folds'][str(fold)]
        q.update(schema=SCHEMA, protocol_sha256=m['protocol_sha256'],
                 runner_sha256=sha(runner), tool_sha256=sha(out / TOOL))
        q['folds'][str(fold)]['tag'] = tag(fold)
        q['tags']['prefix'] = tag(fold)
        q['exp179'] = {'version': prior.VERSION, 'arm': 'M42', 'crop_policy': 'mixed',
             'verification_note': m['verification_note'], 'verification_sha256': verification_sha,
             'loader_sha256': sha(root / prior.LOADER), 'tool_sha256': sha(root / prior.TOOL),
             'nominal_mm': [140, 100], 'detail_slice': [48, 288],
             'allocation': 'even=context,odd=detail', 'shared_frozen_initial_state': True}
        q['exp183'] = {'version': VERSION, 'fold': fold, 'control': base.tag(fold),
                       'protocol_sha256': m['protocol_sha256']}
        name = f'protocols/fold{fold}.json'
        write(out / name, q)
        m['files'][name] = base._entry(out / name, 'overlay')
        m['folds'][str(fold)] = {'tag': tag(fold), 'protocol': name,
             'sha256': sha(out / name), 'roster': base.roster_path(fold),
             'train': info['n_train'], 'val': info['n_val'], 'steps': info['steps']}
    write(out / 'queue.json', queue())
    m['files']['queue.json'] = base._entry(out / 'queue.json', 'overlay')
    write(out / 'manifest.json', m)
    validate(out)
    return {'status': m['status'], 'overlay': str(out), 'manifest_sha256': sha(out / 'manifest.json'),
            'new_arms': [tag(f) for f in FOLDS], 'max_gpu_hours': TOTAL_CAP / 3600}


def validate(package, full=False):
    package = Path(package)
    m = read(package / 'manifest.json')
    need(m['schema'] == SCHEMA and m['version'] in ('1.0.0', '1.1.0', VERSION) and
         m['source_root'] == SOURCE_ROOT and m['remote_root'] == REMOTE_ROOT and
         set(m['folds']) == {str(f) for f in FOLDS} and
         m['caps_seconds'] == {'arm': ARM_CAP, 'total': TOTAL_CAP, 'pilot': PILOT_CAP}, 'EXP183 manifest contract differs')
    need(read(package / 'queue.json') == queue(), 'serial queue differs')
    note = (package / VERIFICATION).read_text(encoding='utf-8')
    need('Verdict: PASS' in note and 'Verdict: FAIL' not in note and
         all(m['files'][name]['sha256'] in note for name in
             (TOOL, PROTOCOL, prior.TOOL, prior.LOADER)) and
         sha(package / VERIFICATION) == m['verification_sha256'] and
         sha(package / PROTOCOL) == m['protocol_sha256'], 'verification/protocol differs')
    required = {TOOL, PROTOCOL, RUNNER, VERIFICATION, prior.TOOL, prior.LOADER,
                'queue.json', base.LABEL, base.SOURCE_WEIGHT, 'config/workspace.toml'}
    need(required <= set(m['files']) and
         m['files'][TOOL]['sha256'] == sha(package / TOOL) and
         m['files'][base.SOURCE_WEIGHT]['sha256'] == base.WEIGHT_SHA and
         m['files'][base.LABEL]['sha256'] == base.LABEL_SHA,
         'missing or changed runtime dependency')
    for name, entry in m['files'].items():
        safe(name)
        need(entry['origin'] in ('exp179', 'overlay'), 'unknown source origin')
        if full or entry['origin'] == 'overlay':
            path = package / name
            need(path.is_file() and not path.is_symlink() and
                 path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                 'package file differs: ' + name)
    for fold in FOLDS:
        record = m['folds'][str(fold)]
        q = read(package / record['protocol'])
        info = q['folds'][str(fold)]
        ids = read(package / record['roster'])
        need(sha(package / record['protocol']) == record['sha256'] and
              q['schema'] == SCHEMA and q['fold'] == fold and q['seed'] == 42 and
              q['exp179']['arm'] == 'M42' and q['exp179']['crop_policy'] == 'mixed' and
              q['exp179']['verification_sha256'] == m['verification_sha256'] and
              q['exp179']['loader_sha256'] == m['files'][prior.LOADER]['sha256'] and
              q['exp179']['tool_sha256'] == m['files'][prior.TOOL]['sha256'] and
              q['exp183'] == {'version': m['version'], 'fold': fold,
                             'control': base.tag(fold),
                             'protocol_sha256': m['protocol_sha256']} and
              q['runner_sha256'] == m['files'][RUNNER]['sha256'] and
              q['tool_sha256'] == m['files'][TOOL]['sha256'] and
              q['initial_state_sha256'] == m['controls'][str(fold)]['initial_state_sha256'] and
              info['tag'] == tag(fold) and info['ordered_ids']['path'] == record['roster'] and
              info['ordered_ids']['sha256'] == sha(package / record['roster']) and
              info['ordered_ids']['train_sha256'] == base.uid_hash(ids['train']) and
              info['ordered_ids']['val_sha256'] == base.uid_hash(ids['val']) and
              len(ids['train']) == record['train'] and len(ids['val']) == record['val'] and
              not set(ids['train']) & set(ids['val']) and
              info['steps'] == record['steps'] and
              all(m['controls'][str(fold)]['args'].get(k) == v
                  for k, v in q['common_args'].items()) and
              q['paths']['labels'] == q['paths']['eval_labels'] == base.LABEL and
              q['label_set']['sha256'] == base.LABEL_SHA and
              q['extra_eval']['k_eval'] == 36 and
              q['common_args']['epochs'] == q['common_args']['fixed_epoch'] == 20,
             'fold contract differs: ' + str(fold))
    if full:
        need(str(package.resolve()) == REMOTE_ROOT and
             sha(package / 'source-manifest.json') == m['source_manifest_sha256'],
             'wrong remote root/source manifest')
    return m


def vast_stage(overlay, source=SOURCE_ROOT, destination=REMOTE_ROOT):
    need(sys.platform.startswith('linux'), 'Vast staging is Linux-only')
    overlay, source, destination = map(Path, (overlay, source, destination))
    need(str(source.resolve()) == SOURCE_ROOT and str(destination.resolve()) == REMOTE_ROOT and
         not destination.exists(), 'fresh exact remote root required')
    m = validate(overlay)
    need(sha(source / 'manifest.json') == m['source_manifest_sha256'], 'EXP179 source drift')
    destination.mkdir(mode=0o700)
    for name, entry in m['files'].items():
        origin = (source if entry['origin'] == 'exp179' else overlay) / name
        need(origin.is_file() and not origin.is_symlink() and sha(origin) == entry['sha256'],
             'remote source drift: ' + name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry['origin'] == 'exp179':
            os.link(origin, target)
        else:
            shutil.copyfile(origin, target)
    shutil.copyfile(overlay / 'manifest.json', destination / 'manifest.json')
    shutil.copyfile(source / 'manifest.json', destination / 'source-manifest.json')
    validate(destination, full=True)
    write(destination / 'stage-receipt.json', {'status': 'STAGED_VERIFIED_NO_GPU',
          'manifest_sha256': sha(destination / 'manifest.json')})
    return {'status': 'STAGED_VERIFIED_NO_GPU', 'destination': REMOTE_ROOT}


def validate_arguments(a, root):
    root = Path(root)
    m = validate(root)
    fold = fold_from_tag(a.tag)
    q = read(root / m['folds'][str(fold)]['protocol'])
    need(a.seed == 42 and a.fold == fold and a.variant == 'AUG1' and
         a.mode == 'full' and not a.validate_only and a.device == 'cuda' and
         Path(a.protocol).resolve() == (root / m['folds'][str(fold)]['protocol']).resolve() and
         all(getattr(a, k) == v for k, v in q['common_args'].items()), 'runtime args changed')
    for key in ('corpus', 'labels', 'eval_labels'):
        need(Path(getattr(a, key)).resolve() == (root / q['paths'][key]).resolve(),
             'input path changed')
    need(0 <= a.workers <= 12 and os.environ.get('HF_HUB_OFFLINE') == '1' and
         all(os.environ.get(k) == '1' for k in base.THREADS) and
         Path(a.init_weights).resolve() == (root / base.SOURCE_WEIGHT).resolve() and
         Path(a.output_root).resolve() == Path(REMOTE_RESULTS), 'environment/output changed')
    info = q['folds'][str(fold)]
    return q, info, read(root / info['ordered_ids']['path']), Path(REMOTE_RESULTS)


def runner_args(fold, pilot=False, root=ROOT):
    from tools import exp166_convnext_t as source
    root = Path(root)
    args = source.runner_args(root, REMOTE_RESULTS, workers=8, pilot=pilot)
    args[1] = str(root / RUNNER)
    for flag, value in {'--protocol': root / f'protocols/fold{fold}.json',
                        '--tag': tag(fold), '--fold': fold, '--seed': 42,
                        '--labels': root / base.LABEL, '--eval-labels': root / base.LABEL}.items():
        args[args.index(flag) + 1] = str(value)
    return args


def verify_run(run_dir, fold, package=ROOT):
    from tools.exp162_dino_folds_kaggle import _prediction_values
    package, out = Path(package), Path(run_dir)
    m = validate(package)
    q = read(package / m['folds'][str(fold)]['protocol'])
    info = q['folds'][str(fold)]
    need(out.name == tag(fold), 'wrong output tag')
    inv = read(out / 'output-inventory.json')
    need(set(inv) == base.REQUIRED_OUTPUTS, 'incomplete output inventory')
    for name, entry in inv.items():
        safe(name)
        path = out / name
        need(path.is_file() and not path.is_symlink() and
             path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
             'output inventory differs: ' + name)
    s, p, history = (read(out / n) for n in ('summary.json', 'provenance.json', 'history.json'))
    need(s['fold'] == fold and s['train'] == info['n_train'] and
         s['val'] == s['rows'] == info['n_val'] and
         s['selected_epoch_1based'] == 20 and s['optimizer_updates'] == info['steps'] and
         s['skipped_updates'] <= info['max_fp16_skips'] and
         [row['epoch'] for row in history] == list(range(20)) and
         s['gold_access'] == 'disabled' and s['weak_oof_macro_auc'] is None and
         s['label_set'] == base.LABEL_SET and
         s['exp179'] == p['exp179'] == q['exp179'] and
         s['exp183'] == p['exp183'] == q['exp183'], 'incomplete fixed-final arm')
    control = m['controls'][str(fold)]
    need(p['environment'] == control['environment'] and
         s['environment']['hostname'] == control['hostname'] and
         p['initial_state_sha256'] == q['initial_state_sha256'] and
         p['init_weights']['weights_sha256'] == base.WEIGHT_SHA and
         p['runner_sha256'] == m['files'][RUNNER]['sha256'] and
         p['guard_sha256'] == m['files'][TOOL]['sha256'] and
         p['protocol_sha256'] == m['folds'][str(fold)]['sha256'] and
         p['train_ids_ordered_sha256'] == info['ordered_ids']['train_sha256'] and
         p['eval_ids_ordered_sha256'] == info['ordered_ids']['val_sha256'] and
         p['actual_pos_weight'] == info['pos_weight'] and
         p['label_file_sha256'] == base.LABEL_SHA and
         p['label_semantic_sha256'] == q['label_set']['semantic_sha256'] and
         p['flip_tta'] == 'coronal_axial_horizontal_mean_probability',
         'run provenance/control mismatch')
    args = s['args']
    need(args['seed'] == 42 and args['fold'] == fold and args['tag'] == tag(fold) and
         args['mode'] == 'full' and all(args.get(k) == v for k, v in q['common_args'].items()) and
         args['labels'] == args['eval_labels'] == f'{REMOTE_ROOT}/{base.LABEL}',
         'run arguments mismatch')
    ids = read(package / base.roster_path(fold))['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        _prediction_values(out / name, ids)
    return {name: inv[name]['sha256'] for name in inv}


def _allowed_launcher_pids(fold):
    pid = os.getppid()
    allowed = set()
    for _ in range(4):
        node = Path('/proc') / str(pid)
        try:
            args = (node / 'cmdline').read_bytes().decode().split('\0')
            cwd = (node / 'cwd').resolve()
            if (str(cwd) == REMOTE_ROOT and
                any(Path(a).as_posix() in ('tools/gpu_job.py', REMOTE_ROOT + '/tools/gpu_job.py')
                    for a in args) and '--tag' in args and '--wait' in args and
                args[args.index('--tag') + 1] == tag(fold)):
                allowed.add(pid)
            pid = int(next(x for x in (node / 'status').read_text().splitlines()
                           if x.startswith('PPid:')).split()[1])
            if pid <= 1:
                break
        except (OSError, ValueError, StopIteration, IndexError):
            break
    return allowed


def run(fold):
    need(sys.platform.startswith('linux') and str(ROOT.resolve()) == REMOTE_ROOT,
         'run requires exact Vast root')
    import fcntl
    with (ROOT / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(fold)


def _run_locked(fold):
    m = validate(ROOT, full=True)
    need(fold in FOLDS, 'unknown fold')
    for earlier in FOLDS[:FOLDS.index(fold)]:
        receipt = read(ROOT / f'validated-fold{earlier}.json')
        need(receipt['status'] == 'VALIDATED_SUCCESS' and
             receipt['manifest_sha256'] == sha(ROOT / 'manifest.json') and
             receipt['output_sha256'] == verify_run(Path(REMOTE_RESULTS) / tag(earlier), earlier),
             'prior arm missing or unvalidated')
    admission = read(ROOT / 'admission.json')
    need(admission == {'status': 'READY', 'manifest_sha256': sha(ROOT / 'manifest.json'),
                       'verification_sha256': m['verification_sha256'],
                       'max_gpu_seconds': TOTAL_CAP, 'folds': list(FOLDS)},
         'exact reviewed admission missing')
    started_files = [ROOT / f'started-fold{f}.json' for f in FOLDS]
    need(not started_files[FOLDS.index(fold)].exists(), 'fresh fold attempt required')
    gpu_line = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
    process_text = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
    gpu = prior.check_idle_gpu(gpu_line, process_text, _allowed_launcher_pids(fold))
    import torch
    import timm
    import numpy
    import cv2
    import pydicom
    versions = dict(torch=torch.__version__, timm=timm.__version__, numpy=numpy.__version__,
                    cv2=cv2.__version__, pydicom=pydicom.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    expected = m['controls'][str(fold)]['environment']
    need(expected['gpu'] == gpu and os.uname().nodename == m['controls'][str(fold)]['hostname'] and
         all(versions[k] == v for k, v in expected.items() if k in versions),
         'control GPU/software differs')
    base._corpus(ROOT)
    out = Path(REMOTE_RESULTS) / tag(fold)
    need(not out.exists(), 'no automatic retry/resume')
    reservation = ROOT / f'started-fold{fold}.json'
    with reservation.open('x', encoding='utf-8') as stream:
        json.dump({'fold': fold, 'reserved_seconds': ARM_CAP,
                   'manifest_sha256': sha(ROOT / 'manifest.json')}, stream)
    env = dict(os.environ, KNEE_WORKSPACE_ROOT=REMOTE_ROOT, KNEE_REMOTE_RELAX_ENV='1',
               HF_HUB_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUTF8='1',
               PYTHONDONTWRITEBYTECODE='1', EXP166_REMOTE='vast')
    env.update({k: '1' for k in base.THREADS})
    started = time.monotonic()
    subprocess.run(runner_args(fold, True), cwd=ROOT, env=env, check=True)
    pilot = read(out / 'pilot.json')
    need(math.isfinite(pilot['epoch_seconds']) and pilot['epoch_seconds'] > 0 and
         math.isfinite(pilot['peak_gpu_gib']) and 0 <= pilot['peak_gpu_gib'] < 20,
         'invalid pilot timing or memory')
    subprocess.run(runner_args(fold), cwd=ROOT, env=env, check=True)
    elapsed = time.monotonic() - started
    hashes = verify_run(out, fold)
    receipt = {'status': 'VALIDATED_SUCCESS', 'fold': fold, 'version': VERSION,
               'manifest_sha256': sha(ROOT / 'manifest.json'), 'output_sha256': hashes,
               'elapsed_seconds': elapsed, 'verification_sha256': m['verification_sha256']}
    write(ROOT / f'validated-fold{fold}.json', receipt)
    export(fold, f'/workspace/exp183-export-M42-fold{fold}.tar')
    return receipt


def export(fold, out, root=ROOT):
    root, out = Path(root), Path(out)
    need(str(root.resolve()) == REMOTE_ROOT and not out.exists(),
         'fresh exact-root export required')
    validate(root, full=True)
    hashes = verify_run(Path(REMOTE_RESULTS) / tag(fold), fold, root)
    receipt = read(root / f'validated-fold{fold}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and
         receipt['output_sha256'] == hashes and
         receipt['manifest_sha256'] == sha(root / 'manifest.json'), 'receipt mismatch')
    with tarfile.open(out, 'x') as archive:
        archive.add(root / 'manifest.json', arcname='manifest.json')
        archive.add(root / f'validated-fold{fold}.json', arcname=f'validated-fold{fold}.json')
        for name in sorted(base.REQUIRED_OUTPUTS | {'output-inventory.json'}):
            archive.add(Path(REMOTE_RESULTS) / tag(fold) / name,
                        arcname=f'results/{tag(fold)}/{name}')
    return {'status': 'EXPORTED_VALIDATED', 'sha256': sha(out), 'path': str(out)}


def collect(archive, overlay, out, fold):
    archive, overlay, out = map(Path, (archive, overlay, out))
    m = validate(overlay)
    need(not out.exists(), 'fresh collection required')
    expected = {'manifest.json', f'validated-fold{fold}.json'} | {
        f'results/{tag(fold)}/{n}' for n in base.REQUIRED_OUTPUTS | {'output-inventory.json'}}
    with tarfile.open(archive, 'r') as source:
        members = source.getmembers()
        need(len(members) == len(expected) and {x.name for x in members} == expected and
             all(x.isfile() and safe(x.name) == x.name for x in members),
             'unsafe/incomplete archive')
        out.mkdir(parents=True)
        for member in members:
            dest = out / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    need(sha(out / 'manifest.json') == sha(overlay / 'manifest.json'),
         'exported manifest differs')
    hashes = verify_run(out / 'results' / tag(fold), fold, overlay)
    receipt = read(out / f'validated-fold{fold}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and receipt['fold'] == fold and
         receipt['output_sha256'] == hashes and
         receipt['manifest_sha256'] == sha(overlay / 'manifest.json') and
         math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] > 0,
         'invalid exported success')
    write(out / 'collection.json', {'status': 'VERIFIED', 'fold': fold,
         'manifest_sha256': sha(overlay / 'manifest.json'), 'archive_sha256': sha(archive),
         'verification_sha256': m['verification_sha256']})
    return {'status': 'VERIFIED', 'fold': fold, 'path': str(out)}


def score_arrays(fold_values, refs, draws, evaluator):
    """Pure pooled screen; fold vectors must already be validated and ordered."""
    import numpy as np
    from tools import exp179_screen as old_screen
    from tools import exp169_bundle_screen as bundle
    from tools.exp162_dino_folds_kaggle import TARGETS
    need(set(fold_values) == set(range(5)), 'five folds required')
    direct_new, direct_old = [], []
    ranked = {'l2': [], 'dino': [], 'mixed': [], 'control': []}
    per_fold, per_fold_target = {}, {}
    offset = 0
    for fold in range(5):
        row = fold_values[fold]
        need(set(row) == {'mixed', 'control', 'rad42', 'rad43', 'dino'},
             'wrong fixed family/arm inputs')
        shapes = {np.asarray(v).shape for v in row.values()}
        need(len(shapes) == 1 and next(iter(shapes))[1] == 12, 'fold shape differs')
        length = len(row['mixed'])
        radial = (evaluator.pct_rank(row['rad42']) +
                  evaluator.pct_rank(row['rad43'])) / 2.0
        ranked['l2'].append(radial)
        ranked['dino'].append(evaluator.pct_rank(row['dino']))
        ranked['mixed'].append(evaluator.pct_rank(row['mixed']))
        ranked['control'].append(evaluator.pct_rank(row['control']))
        direct_new.append(row['mixed'])
        direct_old.append(row['control'])
        segment = slice(offset, offset + length)
        per_fold[str(fold)] = {}
        per_fold_target[str(fold)] = {}
        for name, (y, mask, minimum) in refs.items():
            a_target = evaluator.auc_targets(y[segment], row['mixed'], mask[segment], minimum)
            b_target = evaluator.auc_targets(y[segment], row['control'], mask[segment], minimum)
            a, b = evaluator.macro(a_target), evaluator.macro(b_target)
            per_fold[str(fold)][name] = float(a-b) if np.isfinite(a-b) else None
            per_fold_target[str(fold)][name] = {
                target: float(delta) if np.isfinite(delta) else None
                for target, delta in zip(TARGETS, a_target-b_target)}
        offset += length
    need(offset == next(iter(refs.values()))[0].shape[0], 'pooled reference length differs')
    candidate, control = np.concatenate(direct_new), np.concatenate(direct_old)
    pooled = {name: np.concatenate(parts) for name, parts in ranked.items()}
    fixed = {'l2': pooled['l2'], 'dino': pooled['dino']}
    new_bundle, _ = bundle.bundle_of({**fixed, 'convnext': pooled['mixed']})
    old_bundle, _ = bundle.bundle_of({**fixed, 'convnext': pooled['control']})
    direct = base.paired_jev_consensus(candidate, control, refs, draws, evaluator)
    replacement = base.paired_jev_consensus(new_bundle, old_bundle, refs, draws, evaluator)
    gate = old_screen.admission(direct, replacement)
    advisory = {'direct': old_screen._ruler_flags(direct, None),
                'bundle_replacement': old_screen._ruler_flags(replacement, None)}
    return {'schema': SCHEMA + '-screen', 'version': VERSION, 'folds': list(range(5)),
            'studies': offset, 'bootstrap': {'draws': len(draws), 'seed': 20261003,
                                           'unit': 'validation component'},
            'direct': direct, 'bundle_replacement': replacement,
            'per_fold_direct': per_fold, 'per_fold_target_delta': per_fold_target,
            'registered_gate': gate,
            'ruler_advisory': advisory,
            'status': 'REVIEW' if any(v['status'] == 'REVIEW' for v in advisory.values())
                       else gate['status']}


def screen(outputs, overlay, out, root=ROOT):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp149_label_sources as e149
    from tools import exp156_k_sweep as e156
    from tools import exp169_bundle_screen as bundle
    from tools import exp172_dino_base_l2 as dino_tool
    root, outputs, overlay, out = map(Path, (root, outputs, overlay, out))
    started = time.monotonic()
    m = validate(overlay)
    need(not out.exists(), 'fresh screen output required')
    control = controls(root)
    need(control == m['controls'], 'EXP170 controls changed since preparation')
    dino_outputs = root / dino_tool.EXP / 'outputs'
    dc = read(dino_outputs / 'collection.json')
    need(dc['status'] == 'VERIFIED' and dc['completed_folds'] == list(range(5)) and
         dc['manifest_sha256'] == sha(dino_outputs / 'manifest.json'),
         'EXP172 L2 DINO collection differs')
    dino_overlay = Path(dc['overlay'])
    dm = dino_tool.validate_manifest(dino_overlay / 'manifest.json', dino_overlay)
    canonical, fold_of = e149.fold_rosters(e149._load_manifest('student', root), root)
    ids_by_fold = {fold: [uid for uid in canonical if fold_of[uid] == fold]
                   for fold in range(5)}
    ids = [uid for fold in range(5) for uid in ids_by_fold[fold]]
    need(len(ids) == len(set(ids)) == 4343, 'pooled held-out roster differs')
    raw = e156.references(ids, root)
    refs = {name: raw[key] for name, key in
            (('jev', 'jev_hybrid'), ('aquila', 'aquila_v1'),
             ('dread', 'dread'), ('consensus', 'consensus'))}
    draws = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    inputs, values = {}, {}
    old_overlay = root / BASE_OVERLAY
    prior.validate(old_overlay)
    fold0 = root / 'artifacts/experiments/EXP179_MULTISCALE/outputs/M42'
    c0 = read(fold0 / 'collection.json')
    need(c0['status'] == 'VERIFIED' and c0['manifest_sha256'] == sha(old_overlay / 'manifest.json'),
         'EXP179 fold-0 collection differs')
    prior.verify_run(fold0 / 'results' / prior.tag('M42'), 'M42', old_overlay)
    prior.control_receipt(root)
    old_screen_path = root / 'artifacts/experiments/EXP179_MULTISCALE/screen/result.json'
    old_screen = read(old_screen_path)
    need(old_screen['manifest_sha256'] == sha(old_overlay / 'manifest.json') and
         '43' in old_screen['per_seed'], 'EXP179 S43 diagnostic differs')
    for fold in range(5):
        fid = ids_by_fold[fold]
        if fold:
            need(set(fid) == set(read(overlay / base.roster_path(fold))['val']),
                 'fold roster membership differs')
            collection_dir = outputs / f'fold{fold}'
            receipt = read(collection_dir / 'collection.json')
            need(receipt['status'] == 'VERIFIED' and receipt['fold'] == fold and
                 receipt['manifest_sha256'] == sha(overlay / 'manifest.json'),
                 'EXP183 collection differs')
            mixed_run = collection_dir / 'results' / tag(fold)
            verify_run(mixed_run, fold, overlay)
            control_run = root / control[str(fold)]['path']
        else:
            mixed_run = fold0 / 'results' / prior.tag('M42')
            control_run = root / prior.C42
        dino_run = dino_outputs / 'artifacts/experiments' / dino_tool.tag(fold)
        dq = dm['fold_protocols'][str(fold)]
        dino_tool._verify_fold(dino_run, fold, read(dino_overlay / dq['path']),
                               dq['sha256'], dino_overlay)
        fixed = bundle.paths(root, fold)
        paths = {'mixed': mixed_run / 'oof_k36_flip_tta.csv',
                 'control': control_run / 'oof_k36_flip_tta.csv',
                 'rad42': fixed['l2_s42'], 'rad43': fixed['l2_s43'],
                 'dino': dino_run / 'oof_best.csv'}
        for name, path in paths.items():
            bundle._verify_inventory(path)
            inputs[f'{name}_f{fold}'] = {'path': str(path), 'sha256': sha(path)}
        values[fold] = {name: bundle.load_oof(path, fid) for name, path in paths.items()}
    result = score_arrays(values, refs, draws, ev)
    result.update(input_files=inputs, manifest_sha256=sha(overlay / 'manifest.json'),
                  verification_sha256=m['verification_sha256'],
                  ordered_ids_sha256=base.uid_hash(ids),
                  components_sha256=sha(root / ev.COMPONENTS),
                  reference_sha256={name: sha(root / path) for name, path in
                      (('jev', e156.JEV), ('aquila', e156.AQUILA),
                       ('dread', e156.DREAD),
                       *[(f'consensus_{i}', path) for i, path in enumerate(e156.CONSENSUS)])},
                  mask_coverage={name: mask.sum(axis=0).astype(int).tolist()
                                 for name, (_, mask, _) in refs.items()},
                  elapsed_seconds=time.monotonic() - started,
                  fold0_seed43_diagnostic={'source_sha256': sha(old_screen_path),
                      'direct': old_screen['per_seed']['43']['direct'],
                      'bundle_replacement': old_screen['per_seed']['43']['bundle_replacement'],
                      'pooled_weight': 0})
    need(math.isfinite(result['elapsed_seconds']) and result['elapsed_seconds'] > 0, 'invalid screen timing')
    write(out / 'result.json', result)
    return {'status': result['status'], 'registered_gate': result['registered_gate'],
            'path': str(out / 'result.json')}


def vast_plan(overlay='/workspace/exp183-overlay'):
    import shlex
    need(overlay.startswith('/workspace/') and ':' not in overlay and '\\' not in overlay and
         '..' not in PurePosixPath(overlay).parts, 'absolute /workspace overlay required')
    q = shlex.quote(overlay)
    return '\n'.join([
        '# On Windows Git Bash: export MSYS_NO_PATHCONV=1 before scp/ssh.',
        '# B: inspect independent PASS note, exact manifest and live idle GPU; claim Active compute.',
        f'PYTHONPATH={SOURCE_ROOT} python {q}/{TOOL} vast-stage --overlay {q}',
        f'cd {REMOTE_ROOT}',
        f'python {TOOL} validate --package . --full',
        'python -c \'import json,hashlib; m=json.load(open("manifest.json")); h=hashlib.file_digest(open("manifest.json","rb"),"sha256").hexdigest(); json.dump(dict(status="READY",manifest_sha256=h,verification_sha256=m["verification_sha256"],max_gpu_seconds=16200,folds=[1,2,3,4]),open("admission.json","x"))\'',
        'setsid nohup python tools/gpu_queue.py queue.json --foreground > exp183-queue.log 2>&1 < /dev/null & echo $! > exp183-queue.pid',
        '# Each validated arm exports /workspace/exp183-export-M42-fold{1,2,3,4}.tar.',
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--out', required=True); a.add_argument('--verification', required=True)
    a = sub.add_parser('validate'); a.add_argument('--package', required=True); a.add_argument('--full', action='store_true')
    a = sub.add_parser('vast-stage'); a.add_argument('--overlay', required=True)
    a = sub.add_parser('vast-plan'); a.add_argument('--overlay', default='/workspace/exp183-overlay')
    a = sub.add_parser('run'); a.add_argument('--fold', type=int, choices=FOLDS, required=True)
    a = sub.add_parser('export'); a.add_argument('--fold', type=int, choices=FOLDS, required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('collect'); a.add_argument('--fold', type=int, choices=FOLDS, required=True)
    a.add_argument('--archive', required=True); a.add_argument('--overlay', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('screen'); a.add_argument('--outputs', default=EXP + '/outputs')
    a.add_argument('--overlay', required=True); a.add_argument('--out', default=EXP + '/screen')
    args = parser.parse_args(argv)
    if args.command == 'prepare': result = prepare(args.out, args.verification)
    elif args.command == 'validate':
        m = validate(args.package, args.full)
        result = {'status': 'VERIFIED_PACKAGE', 'version': VERSION,
                  'verification_sha256': m['verification_sha256']}
    elif args.command == 'vast-stage': result = vast_stage(args.overlay)
    elif args.command == 'vast-plan': result = vast_plan(args.overlay)
    elif args.command == 'run': result = run(args.fold)
    elif args.command == 'export': result = export(args.fold, args.out)
    elif args.command == 'collect': result = collect(args.archive, args.overlay, args.out, args.fold)
    else: result = screen(args.outputs, args.overlay, args.out)
    print(result if isinstance(result, str) else json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
