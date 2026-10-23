"""EXP167 v1.0.0: four serial Vast ConvNeXt-T folds and a fixed CPU screen.

Only ``run`` enters the GPU. B stages, admits and launches the queue explicitly.
The overlay's ``vast-stage`` entry works before the older modules are hardlinked.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = '1.0.0'
SCHEMA = 'exp167-convnext-t-folds-v1'
EXP = 'artifacts/experiments/EXP167_CONVNEXT_T_FOLDS'
PROTOCOL = EXP + '/PROTOCOL.md'
TOOL = 'tools/exp167_convnext_t_folds.py'
RUNNER = 'training/runners/exp167_convnext_t_fold.py'
SOURCE_ROOT = '/workspace/exp166'
REMOTE_ROOT = '/workspace/exp167'
REMOTE_RESULTS = '/workspace/exp167-results'
SOURCE_ARCHIVE = '.local/exp166-vast-stage-v1/package/exp166-code.zip.bin'
SOURCE_RECEIPT = 'exp166-bundle.json'
SOURCE_MANIFEST = 'artifacts/experiments/EXP166_CONVNEXT_T/manifest.json'
SOURCE_WEIGHT = 'storage/weights/exp166/convnext_tiny_fb_in22k_ft_in1k.safetensors'
WEIGHT_SHA = '77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a'
FOLD_SOURCE = 'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1/s42/manifest.json'
PILOT_CAP = 300
FULL_CAP = 2100
FOLD_CAP = 2500
FOLD0_SECONDS = 1438
REQUIRED_OUTPUTS = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                    'oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'}
REQUIRED_MODULES = {'tools/exp166_convnext_t.py', 'tools/exp127_radr50_single.py',
                    'tools/exp150_consensus_weight.py', 'tools/exp124_flip_tta.py',
                    'tools/exp162_dino_folds_kaggle.py', 'tools/exp115_vlm_eval.py',
                    'tools/exp156_k_sweep.py', 'tools/gpu_queue.py', 'tools/gpu_job.py',
                    'tools/colab/exp159_remote.py', 'training/src/raptor_common.py',
                    'training/src/raptor_data.py', 'training/src/raptor_exp127.py',
                    'training/src/raptor_exp127_aug1.py', 'training/src/raptor_runtime.py',
                    'training/src/raptor_radimagenet.py'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def safe(name):
    p = PurePosixPath(name)
    need(isinstance(name, str) and name and name == p.as_posix() and not p.is_absolute()
         and all(x not in ('.', '..') for x in p.parts) and ':' not in name and '\\' not in name,
         'unsafe bundle path: ' + str(name))
    return name


def tag(fold):
    need(fold in range(1, 5), 'EXP167 trains OUR folds 1-4 only')
    return f'EXP167_CONVNEXT_T_S42_F{fold}'


def roster_path(fold):
    return f'artifacts/experiments/EXP162_DINO_FOLDS/rosters/fold{fold}-sorted.json'


def protocol_path(fold):
    return f'protocols/fold{fold}.json'


def runner_source(root=ROOT):
    from tools import exp166_convnext_t as source
    old = source.runner_source((Path(root) / source.BASE_RUNNER).read_text(encoding='utf-8'))
    anchor = 'from tools import exp166_convnext_t as guard'
    need(old.count(anchor) == 1, 'EXP166 runner guard changed')
    new = old.replace(anchor, 'from tools import exp167_convnext_t_folds as guard')
    compile(new, RUNNER, 'exec')
    return new


def guard_attributes(runner):
    tree = ast.parse(runner)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == 'guard'}
    need(all(hasattr(sys.modules[__name__], n) for n in names),
         'runner guard attributes missing: ' + ', '.join(sorted(n for n in names if not hasattr(sys.modules[__name__], n))))
    return sorted(names)


def __getattr__(name):
    # The frozen EXP166 runner calls these unchanged helpers through its guard.
    from tools import exp166_convnext_t as source
    if name in ('validate_arguments', 'validate_fold_membership'):
        raise AttributeError(name)
    return getattr(source, name)


def validate_fold_membership(corpus_ids, ordered, rows, canonical):
    from tools import exp166_convnext_t as source
    # EXP084 requires the supplied order to be monotonic in its canonical list.
    # EXP162's copies sort both lists, so the canonical list must be sorted too.
    return source.validate_fold_membership(corpus_ids, ordered, rows, sorted(canonical))


def queue():
    return [{'tag': tag(f), 'command': ['python', TOOL, 'run', '--fold', str(f)],
             'continue_on_failure': False} for f in range(1, 5)]


def _entry(path, origin):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size, 'origin': origin}


def _source_files(root):
    root = Path(root)
    stage = read(root / '.local/exp166-vast-stage-v1/stage-receipt.json')
    archive = root / SOURCE_ARCHIVE
    need(stage['surface'] == 'vast' and sha(archive) == stage['archive_sha256']
         and stage['manifest_sha256'] == sha(root / SOURCE_MANIFEST),
         'EXP166 Vast source stage differs')
    with zipfile.ZipFile(archive) as z:
        receipt = json.loads(z.read(SOURCE_RECEIPT))
    need(receipt['schema'] == 'exp166-package-v1' and receipt['surface'] == 'vast'
         and receipt['manifest_sha256'] == stage['manifest_sha256']
         and receipt['files'][SOURCE_WEIGHT]['sha256'] == WEIGHT_SHA,
         'EXP166 source bundle differs')
    return receipt['files'], hashlib.sha256(json.dumps(receipt, indent=2).encode()).hexdigest()


def prepare(out, root=ROOT):
    """Make a slim overlay; old EXP166 files are SHA-checked remote hardlinks."""
    from tools import exp166_convnext_t as source
    from tools import exp164_dino_base_folds as base_folds
    from tools import exp162_dino_folds_kaggle as folds
    root, out = Path(root), Path(out)
    need(not out.exists(), 'fresh EXP167 overlay required')
    registered, _ = source.registration(root)
    files, receipt_sha = _source_files(root)
    derived = runner_source(root)
    guard_attributes(derived)
    need(SOURCE_MANIFEST in files and SOURCE_WEIGHT in files and
         files[SOURCE_MANIFEST]['sha256'] == sha(root / SOURCE_MANIFEST),
         'source manifest or weight missing')
    reference = read(root / FOLD_SOURCE)
    rosters = base_folds.roster(root)  # checks byte-identical sorted EXP162 copies
    source_labels = source.old.load_label_csv(root / registered['label_set']['path'])[0]
    payload = {TOOL: root / TOOL, PROTOCOL: root / PROTOCOL,
               'config/workspace.toml': root / 'config/workspace.toml'}
    for f in range(1, 5):
        # B fix 2026-10-10: exp164.roster() returns the ORIGINAL EXP099 rosters after checking the sorted copies;
        # the sorted copy that is shipped is the one to check.
        ids = json.loads((root / roster_path(f)).read_text(encoding='utf-8'))
        need(ids['train'] == sorted(ids['train']) and ids['val'] == sorted(ids['val']),
             'sorted roster differs')
        payload[roster_path(f)] = root / roster_path(f)
        group = reference['folds'][str(f)]['groups']['path']
        payload[group] = root / group
    out.mkdir(parents=True)
    manifest = {'schema': SCHEMA, 'version': VERSION, 'status': 'PREPARED_NOT_LAUNCHED',
                'source_root': SOURCE_ROOT, 'remote_root': REMOTE_ROOT,
                'source_bundle_sha256': receipt_sha,
                'source_manifest_sha256': sha(root / SOURCE_MANIFEST),
                'source_weight_sha256': files[SOURCE_WEIGHT]['sha256'],
                'protocol_sha256': sha(root / PROTOCOL), 'serial_folds': [1, 2, 3, 4],
                'caps_seconds': {'pilot': PILOT_CAP, 'full': FULL_CAP, 'fold': FOLD_CAP,
                                 'measured_fold0': FOLD0_SECONDS},
                'files': {n: {**e, 'origin': 'exp166'} for n, e in files.items()},
                'folds': {}}
    for name, source_file in payload.items():
        safe(name)
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        manifest['files'][name] = _entry(target, 'overlay')
    target = out / RUNNER
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(derived, encoding='utf-8', newline='\n')
    manifest['files'][RUNNER] = _entry(target, 'overlay')
    for f in range(1, 5):
        info = copy.deepcopy(reference['folds'][str(f)])
        ids = read(out / roster_path(f))
        original_ids = read(root / info['ordered_ids']['path'])
        need(set(ids['train']) == set(original_ids['train']) and
             set(ids['val']) == set(original_ids['val']) and
             len(ids['train']) + len(ids['val']) == 4343,
             'EXP134 fold membership differs')
        vector = source.old.pos_weight(source_labels, ids['train'])
        original_vector = info['pos_weight']
        info['pos_weight'] = [float(x) for x in vector]
        info['pos_weight_equals_exp099'] = info['pos_weight'] == original_vector
        info['tag'] = tag(f)
        info['ordered_ids'] = {'path': roster_path(f), 'sha256': sha(out / roster_path(f)),
                               'train_sha256': source.old.uid_hash(ids['train']),
                               'val_sha256': source.old.uid_hash(ids['val'])}
        need(sha(out / info['groups']['path']) == info['groups']['sha256'],
             'fold group metadata differs')
        q = copy.deepcopy(registered)
        q.update(schema=SCHEMA, status='PREPARED_NOT_LAUNCHED', source_manifest_sha256=manifest['source_manifest_sha256'],
                 protocol_sha256=manifest['protocol_sha256'], runner_sha256=sha(out / RUNNER),
                 tool_sha256=sha(out / TOOL), folds={str(f): info},
                 tags={'prefix': tag(f)}, fold=f)
        path = protocol_path(f)
        write(out / path, q)
        manifest['files'][path] = _entry(out / path, 'overlay')
        manifest['folds'][str(f)] = {'tag': tag(f), 'protocol': path,
                                     'protocol_sha256': sha(out / path),
                                     'roster': roster_path(f), 'train': len(ids['train']),
                                     'val': len(ids['val']), 'steps': info['steps']}
    write(out / 'queue.json', queue())
    manifest['files']['queue.json'] = _entry(out / 'queue.json', 'overlay')
    write(out / 'manifest.json', manifest)
    validate(out, full=False)
    return {'status': manifest['status'], 'overlay': out.as_posix(),
            'manifest_sha256': sha(out / 'manifest.json'), 'folds': manifest['folds'],
            'overlay_bytes': sum(e['bytes'] for e in manifest['files'].values() if e['origin'] == 'overlay')}


def validate(package, full=False):
    package = Path(package)
    m = read(package / 'manifest.json')
    need(m['schema'] == SCHEMA and m['version'] == VERSION and m['serial_folds'] == [1, 2, 3, 4]
         and m['source_root'] == SOURCE_ROOT and m['remote_root'] == REMOTE_ROOT
         and m['caps_seconds'] == {'pilot': PILOT_CAP, 'full': FULL_CAP,
                                   'fold': FOLD_CAP, 'measured_fold0': FOLD0_SECONDS}
         and FULL_CAP >= math.ceil(1.3 * FOLD0_SECONDS), 'EXP167 package contract differs')
    need(read(package / 'queue.json') == queue(), 'serial queue differs')
    need(m['files']['config/workspace.toml']['origin'] == 'overlay' and
         m['files'][SOURCE_WEIGHT]['origin'] == 'exp166' and
         m['source_weight_sha256'] == m['files'][SOURCE_WEIGHT]['sha256'] and
         m['source_weight_sha256'] == WEIGHT_SHA, 'workspace/remote weight contract differs')
    need(m['protocol_sha256'] == m['files'][PROTOCOL]['sha256'] and
         m['source_manifest_sha256'] == m['files'][SOURCE_MANIFEST]['sha256'],
         'source/protocol binding differs')
    need(REQUIRED_MODULES.issubset(m['files']),
         'transitive runner/queue imports missing: ' + ', '.join(sorted(REQUIRED_MODULES - set(m['files']))))
    for name, entry in m['files'].items():
        safe(name)
        need(entry['origin'] in ('overlay', 'exp166'), 'unknown package source')
        if full or entry['origin'] == 'overlay':
            path = package / name
            need(path.is_file() and not path.is_symlink() and path.stat().st_size == entry['bytes']
                 and sha(path) == entry['sha256'], 'package file differs: ' + name)
    for f in range(1, 5):
        record = m['folds'][str(f)]
        q = read(package / record['protocol'])
        ids = read(package / record['roster'])
        info = q['folds'][str(f)]
        need(record['protocol_sha256'] == sha(package / record['protocol']) and
             q['schema'] == SCHEMA and q['fold'] == f and q['seed'] == 42 and
             q['protocol_sha256'] == m['protocol_sha256'] and
             q['source_manifest_sha256'] == m['source_manifest_sha256'] and
             q['runner_sha256'] == m['files'][RUNNER]['sha256'] and
             q['tool_sha256'] == m['files'][TOOL]['sha256'] and
             q['common_args']['arch'] == 'convnext_tiny.fb_in22k_ft_in1k' and
             q['common_args']['k'] == 12 and q['extra_eval']['k_eval'] == 36 and
             q['common_args']['epochs'] == q['common_args']['fixed_epoch'] == 20 and
             q['common_args']['amp'] == q['common_args']['eval_amp'] == 'fp16' and
             info['tag'] == tag(f) and info['ordered_ids']['path'] == record['roster'] and
             info['ordered_ids']['sha256'] == sha(package / record['roster']) and
             info['ordered_ids']['train_sha256'] == uid_hash(ids['train']) and
             info['ordered_ids']['val_sha256'] == uid_hash(ids['val']) and
             ids['train'] == sorted(ids['train']) and ids['val'] == sorted(ids['val']) and
             len(ids['train']) == record['train'] and len(ids['val']) == record['val'] and
             len(ids['train']) + len(ids['val']) == 4343 and
             not set(ids['train']) & set(ids['val']) and
             info['steps'] == record['steps'] == (len(ids['train']) // 8) * 20 and
             len(info['pos_weight']) == 12 and all(1 <= x <= 10 for x in info['pos_weight']),
             'fold contract differs: ' + str(f))
    if full:
        need(str(package.resolve()) == REMOTE_ROOT, 'wrong staged remote root')
        need(sha(package / SOURCE_RECEIPT) == m['source_bundle_sha256'], 'EXP166 receipt differs')
        guard_attributes((package / RUNNER).read_text(encoding='utf-8'))
    return m


def vast_stage(overlay, source=SOURCE_ROOT, destination=REMOTE_ROOT):
    need(sys.platform.startswith('linux'), 'vast-stage is Linux-only')
    overlay, source, destination = map(Path, (overlay, source, destination))
    need(str(source.resolve()) == SOURCE_ROOT and str(destination.resolve()) == REMOTE_ROOT
         and not destination.exists(), 'exact fresh Vast roots required')
    m = validate(overlay)
    need(sha(source / SOURCE_RECEIPT) == m['source_bundle_sha256'] and
         sha(source / SOURCE_MANIFEST) == m['source_manifest_sha256'],
         'remote EXP166 source differs')
    destination.mkdir(mode=0o700)
    for name, entry in m['files'].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        origin = source if entry['origin'] == 'exp166' else overlay
        old = origin / name
        need(old.is_file() and not old.is_symlink() and old.stat().st_size == entry['bytes']
             and sha(old) == entry['sha256'], 'stage source differs: ' + name)
        if entry['origin'] == 'exp166':
            os.link(old, target)
            need(old.stat().st_ino == target.stat().st_ino, 'hardlink failed: ' + name)
        else:
            shutil.copyfile(old, target)
    shutil.copyfile(overlay / 'manifest.json', destination / 'manifest.json')
    os.link(source / SOURCE_RECEIPT, destination / SOURCE_RECEIPT)
    validate(destination, full=True)
    write(destination / 'stage-receipt.json', {'status': 'STAGED_VERIFIED_NO_GPU',
          'manifest_sha256': sha(destination / 'manifest.json'), 'source_root': SOURCE_ROOT})
    return {'status': 'STAGED_VERIFIED_NO_GPU', 'destination': REMOTE_ROOT,
            'manifest_sha256': sha(destination / 'manifest.json')}


def validate_arguments(a, root):
    root = Path(root)
    m = validate(root, full=False)
    f = a.fold
    need(f in range(1, 5) and Path(a.protocol).resolve() == (root / protocol_path(f)).resolve()
         and a.variant == 'AUG1' and a.seed == 42 and a.tag == tag(f) and a.mode == 'full'
         and not a.validate_only, 'wrong EXP167 fold/seed/tag/mode')
    q = read(root / protocol_path(f))
    for key, value in q['common_args'].items():
        need(getattr(a, key) == value, 'frozen argument changed: ' + key)
    need(0 <= a.workers <= 12 and a.device == 'cuda' and
         os.environ.get('HF_HUB_OFFLINE') == '1' and
         all(os.environ.get(k) == '1' for k in __getattr__('THREADS')),
         'runtime environment differs')
    for key in ('corpus', 'labels', 'eval_labels'):
        need(Path(getattr(a, key)).resolve() == (root / q['paths'][key]).resolve(),
             'input path differs: ' + key)
    need(Path(a.init_weights).resolve() == (root / SOURCE_WEIGHT).resolve() and
         Path(a.output_root).resolve() == Path(REMOTE_RESULTS),
         'remote input/output path differs')
    info = q['folds'][str(f)]
    return q, info, read(root / info['ordered_ids']['path']), Path(REMOTE_RESULTS)


def _corpus(root):
    from tools import exp150_consensus_weight as e150
    from tools import exp166_convnext_t as source
    root = Path(root)
    p = read(root / SOURCE_MANIFEST)
    target = root / p['paths']['corpus']
    if target.is_dir():
        return target
    old = Path(SOURCE_ROOT) / p['paths']['corpus']
    need(old.is_dir(), 'EXP166 remote corpus missing')
    linked = root / 'corpus-hardlinks'
    need(not linked.exists() and not target.exists(), 'fresh corpus hardlinks required')
    linked.mkdir()
    for item in old.iterdir():
        if item.is_file():
            os.link(item, linked / item.name)
    e150.verify_corpus(linked, root / e150.CORPUS_LOCAL,
                       read(root / e150.MANIFEST)['corpus_file_list'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(linked, target_is_directory=True)
    return target


def runner_args(fold, root=ROOT, pilot=False):
    from tools import exp166_convnext_t as source
    root = Path(root)
    args = source.runner_args(root, REMOTE_RESULTS, workers=8, pilot=pilot)
    def set_flag(flag, value):
        args[args.index(flag)+1] = str(value)
    args[1] = str(root / RUNNER)
    set_flag('--protocol', root / protocol_path(fold))
    set_flag('--fold', fold)
    set_flag('--tag', tag(fold))
    return args


def verify_run(run, fold, package=ROOT):
    from tools import exp162_dino_folds_kaggle as folds
    package, run = Path(package), Path(run)
    m = validate(package, full=False)
    q = read(package / protocol_path(fold))
    info = q['folds'][str(fold)]
    need(run.name == tag(fold), 'wrong run tag')
    inv = read(run / 'output-inventory.json')
    need(set(inv) == REQUIRED_OUTPUTS, 'incomplete EXP167 output inventory')
    for name, record in inv.items():
        need(safe(name) == name and (run / name).is_file() and
             (run / name).stat().st_size == record['bytes'] and
             sha(run / name) == record['sha256'], 'output inventory differs: ' + name)
    summary, prov, history = (read(run / name) for name in
                              ('summary.json', 'provenance.json', 'history.json'))
    expected_steps = info['steps']
    need(summary['fold'] == fold and summary['train'] == info['n_train'] and
         summary['val'] == info['n_val'] and summary['rows'] == info['n_val'] and
         summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == expected_steps and
         summary['skipped_updates'] <= info['max_fp16_skips'] and
         len(history) == 20 and [h['epoch'] for h in history] == list(range(20)) and
         summary['gold_access'] == 'disabled' and summary['weak_oof_macro_auc'] is None and
         '3090' in summary['environment']['gpu'] and
         '3090' in prov['environment']['gpu'],
         'incomplete fixed-final EXP167 fold')
    args = summary['args']
    need(all(args.get(k) == v for k, v in q['common_args'].items()) and
         args['seed'] == 42 and args['fold'] == fold and args['tag'] == tag(fold) and
         args['mode'] == 'full' and
         prov['initial_state_sha256'] == q['initial_state_sha256'] and
         prov['init_weights']['weights_sha256'] == m['source_weight_sha256'] and
         prov['runner_sha256'] == m['files'][RUNNER]['sha256'] and
         prov['guard_sha256'] == m['files'][TOOL]['sha256'] and
         prov['protocol_sha256'] == m['folds'][str(fold)]['protocol_sha256'] and
         prov['train_ids_ordered_sha256'] == info['ordered_ids']['train_sha256'] and
         prov['eval_ids_ordered_sha256'] == info['ordered_ids']['val_sha256'] and
         prov['actual_pos_weight'] == info['pos_weight'] and
         prov['flip_tta'] == 'coronal_axial_horizontal_mean_probability',
         'fold recipe/provenance differs')
    ids = read(package / info['ordered_ids']['path'])['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        folds._prediction_values(run / name, ids)
    return {name: inv[name]['sha256'] for name in REQUIRED_OUTPUTS}


def run(fold):
    need(sys.platform.startswith('linux') and str(ROOT.resolve()) == REMOTE_ROOT,
         'EXP167 run requires exact Vast root')
    m = validate(ROOT, full=True)
    need(fold in range(1, 5), 'wrong fold')
    admission = read(ROOT / 'admission.json')
    need(admission == {'status': 'READY', 'allow_execution': True,
                       'manifest_sha256': sha(ROOT / 'manifest.json'),
                       'serial_folds': [1, 2, 3, 4], 'max_gpu_seconds_per_fold': FOLD_CAP},
         'B exact-manifest admission missing')
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=30).stdout
    need('3090' in smi.splitlines()[0], 'Vast RTX 3090 absent')
    fold0 = read(Path('/workspace/exp166-results/EXP166_CONVNEXT_T_S42_F0/summary.json'))
    need(fold0['environment']['gpu'] == smi.splitlines()[0].strip() and
         fold0['environment']['hostname'] == os.uname().nodename,
         'EXP166 fold 0 was not run on this Vast instance')
    _corpus(ROOT)
    output = Path(REMOTE_RESULTS)
    output.mkdir(exist_ok=True)
    dest = output / tag(fold)
    if (ROOT / f'validated-fold{fold}.json').is_file():
        verify_run(dest, fold, ROOT)
        return {'status': 'ALREADY_VALIDATED', 'fold': fold}
    env = dict(os.environ, KNEE_WORKSPACE_ROOT=REMOTE_ROOT,
               KNEE_REMOTE_RELAX_ENV='1', HF_HUB_OFFLINE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUTF8='1',
               PYTHONDONTWRITEBYTECODE='1', EXP166_REMOTE='vast')
    for key in __getattr__('THREADS'):
        env[key] = '1'
    started = time.monotonic()
    if (dest / 'epoch-state.pt').is_file() and not (dest / 'pilot.json').is_file():
        first = read(dest / 'history.json')[0]
        need(first['epoch'] == 0, 'checkpoint lacks pilot epoch')
        write(dest / 'pilot.json', {'epoch_seconds': first['epoch_seconds'],
              'peak_gpu_gib': first['peak_gpu_gib'],
              'projected_fit_seconds': first['epoch_seconds'] * 20})
    if not (dest / 'pilot.json').is_file():
        subprocess.run(runner_args(fold, ROOT, pilot=True), cwd=ROOT, env=env,
                       check=True, timeout=PILOT_CAP)
    pilot = read(dest / 'pilot.json')
    need(0 < pilot['epoch_seconds'] <= PILOT_CAP and pilot['peak_gpu_gib'] < 20 and
         1.3 * (19 * pilot['epoch_seconds'] + 150) <= FULL_CAP,
         'pilot timing/memory/full projection fails cap')
    remaining = FOLD_CAP - (time.monotonic() - started)
    need(remaining > 0, 'fold cap expired after pilot')
    write(ROOT / f'execution-fold{fold}.json', {'pilot': pilot,
          'full_cap_seconds': FULL_CAP, 'fold_cap_seconds': FOLD_CAP,
          'manifest_sha256': sha(ROOT / 'manifest.json')})
    if not (dest / 'summary.json').is_file():
        subprocess.run(runner_args(fold, ROOT), cwd=ROOT, env=env, check=True,
                       timeout=min(FULL_CAP, remaining))
    need(time.monotonic() - started <= FOLD_CAP, 'fold wall cap exceeded')
    hashes = verify_run(dest, fold, ROOT)
    receipt = {'schema': SCHEMA, 'status': 'VALIDATED_SUCCESS', 'fold': fold,
               'tag': tag(fold), 'manifest_sha256': sha(ROOT / 'manifest.json'),
               'protocol_sha256': m['folds'][str(fold)]['protocol_sha256'],
               'output_inventory_sha256': sha(dest / 'output-inventory.json'),
               'output_sha256': hashes, 'pilot': pilot,
               'elapsed_seconds': time.monotonic() - started,
               'gpu': smi.splitlines()[0]}
    write(ROOT / f'validated-fold{fold}.json', receipt)
    return {'status': receipt['status'], 'fold': fold,
            'elapsed_seconds': receipt['elapsed_seconds']}


def vast_plan(overlay='/workspace/exp167-overlay'):
    import shlex
    need(str(overlay).startswith('/workspace/') and '\\' not in str(overlay)
         and '..' not in PurePosixPath(str(overlay)).parts, 'absolute Vast overlay required')
    o = shlex.quote(str(overlay))
    return '\n'.join([
        '# B checks the 3090 is idle and claims Vast compute before launch.',
        f'python {o}/{TOOL} vast-stage --overlay {o}',
        f'cd {REMOTE_ROOT}',
        f'python {TOOL} validate --package . --full',
        "python -c 'import hashlib,json; h=hashlib.sha256(open(\"manifest.json\",\"rb\").hexdigest(); json.dump(dict(status=\"READY\",allow_execution=True,manifest_sha256=h,serial_folds=[1,2,3,4],max_gpu_seconds_per_fold=2500),open(\"admission.json\",\"x\"))'",
        'export KNEE_WORKSPACE_ROOT=/workspace/exp167 HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1',
        'setsid nohup python tools/gpu_queue.py queue.json --foreground > exp167-queue.log 2>&1 < /dev/null & echo $! > exp167-queue.pid',
        '# After all four gpu_job SUCCESS markers exist, export:',
        f'python {TOOL} export --out /workspace/exp167-export.tar',
    ])


def export(out, root=ROOT):
    """CPU-only validated artifact tar; excludes resumable optimizer checkpoints."""
    root, out = Path(root), Path(out)
    need(str(root.resolve()) == REMOTE_ROOT and not out.exists(), 'Vast fresh export required')
    m = validate(root, full=True)
    with tarfile.open(out, 'x') as tar:
        tar.add(root / 'manifest.json', arcname='manifest.json')
        for f in range(1, 5):
            receipt_path = root / f'validated-fold{f}.json'
            receipt = read(receipt_path)
            run_path = Path(REMOTE_RESULTS) / tag(f)
            hashes = verify_run(run_path, f, root)
            need(receipt['status'] == 'VALIDATED_SUCCESS' and
                 receipt['manifest_sha256'] == sha(root / 'manifest.json') and
                 receipt['protocol_sha256'] == m['folds'][str(f)]['protocol_sha256'] and
                 receipt['output_inventory_sha256'] == sha(run_path / 'output-inventory.json') and
                 receipt['output_sha256'] == hashes, 'fold receipt differs')
            tar.add(receipt_path, arcname=f'validated-fold{f}.json')
            for name in sorted(REQUIRED_OUTPUTS | {'output-inventory.json'}):
                tar.add(run_path / name, arcname=f'results/{tag(f)}/{name}')
    return {'status': 'EXPORTED_VALIDATED', 'path': out.as_posix(),
            'sha256': sha(out), 'bytes': out.stat().st_size}


def collect(export_file, overlay, out, root=ROOT):
    export_file, overlay, out, root = map(Path, (export_file, overlay, out, root))
    need(not out.exists(), 'fresh collection directory required')
    m = validate(overlay)
    expected = {'manifest.json', *(f'validated-fold{f}.json' for f in range(1, 5))}
    expected.update(f'results/{tag(f)}/{name}' for f in range(1, 5)
                    for name in REQUIRED_OUTPUTS | {'output-inventory.json'})
    out.mkdir(parents=True)
    try:
        with tarfile.open(export_file, 'r') as tar:
            members = tar.getmembers()
            need(len(members) == len(expected) and {x.name for x in members} == expected
                 and all(x.isfile() and safe(x.name) == x.name for x in members),
                 'unsafe/incomplete EXP167 export')
            for member in members:
                dest = out / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        need(sha(out / 'manifest.json') == sha(overlay / 'manifest.json'),
             'export manifest differs')
        completed = []
        for f in range(1, 5):
            receipt = read(out / f'validated-fold{f}.json')
            run_dir = out / 'results' / tag(f)
            hashes = verify_run(run_dir, f, overlay)
            need(receipt['status'] == 'VALIDATED_SUCCESS' and receipt['fold'] == f and
                 receipt['manifest_sha256'] == sha(out / 'manifest.json') and
                 receipt['protocol_sha256'] == m['folds'][str(f)]['protocol_sha256'] and
                 receipt['output_inventory_sha256'] == sha(run_dir / 'output-inventory.json') and
                 receipt['output_sha256'] == hashes,
                 'collected fold receipt differs')
            completed.append(f)
        write(out / 'collection.json', {'schema': SCHEMA, 'status': 'VERIFIED',
              'completed_folds': completed, 'manifest_sha256': sha(out / 'manifest.json'),
              'export_sha256': sha(export_file), 'overlay': str(overlay.resolve())})
        return {'status': 'VERIFIED', 'folds': completed, 'path': out.as_posix()}
    except BaseException:
        shutil.rmtree(out)
        raise


def rank_per_fold(parts, evaluator):
    import numpy as np
    need(set(parts) == set(range(5)), 'five folds required for ranking')
    return np.concatenate([evaluator.pct_rank(parts[f]) for f in range(5)])


def screen_gate(pair):
    return bool(pair['primary_mean_delta'] > 0 and pair['primary_mean_ci95'][0] > 0)


def screen(outputs=ROOT / EXP / 'outputs', destination=ROOT / EXP / 'screen', root=ROOT):
    import numpy as np
    from tools import exp162_dino_folds_kaggle as folds
    from tools import exp164_dino_base_folds as dino_folds
    from tools import exp166_convnext_t as source
    from tools import exp115_vlm_eval as ev
    from tools import exp156_k_sweep as e156
    outputs, destination, root = map(Path, (outputs, destination, root))
    need(not destination.exists(), 'fresh screen directory required')
    collection = read(outputs / 'collection.json')
    need(collection['schema'] == SCHEMA and collection['status'] == 'VERIFIED' and
         collection['completed_folds'] == [1, 2, 3, 4] and
         collection['manifest_sha256'] == sha(outputs / 'manifest.json'),
         'EXP167 collection incomplete')
    dino_outputs = root / 'artifacts/experiments/EXP164_DINO_BASE_FOLDS/outputs'
    need(read(dino_outputs / 'collection.json')['status'] == 'VERIFIED',
         'EXP164 DINO-base collection incomplete')
    rosters = dino_folds.roster(root)
    ids_by_fold = {f: (rosters[0][1]['val'] if f == 0 else sorted(rosters[f][1]['val']))
                   for f in range(5)}
    all_ids = [u for f in range(5) for u in ids_by_fold[f]]
    need(len(all_ids) == len(set(all_ids)) == 4343, 'OOF five-fold coverage differs')
    fold0 = root / source.RESULT_DIR / 'vast' / source.TAG
    source.verify_run(fold0, root)
    fold0_summary = read(fold0 / 'summary.json')
    parts = {name: {} for name in ('convnext', 'convnext_plain', 'h1', 'radr50', 'dino')}
    paths = {name: {} for name in parts}
    for f in range(5):
        run = fold0 if f == 0 else outputs / 'results' / tag(f)
        if f:
            verify_run(run, f, _screen_overlay(outputs))
            recorded = read(run / 'summary.json')
            need(recorded['environment']['gpu'] == fold0_summary['environment']['gpu'] and
                 recorded['environment']['hostname'] == fold0_summary['environment']['hostname'],
                 'EXP167 fold hardware/host differs from EXP166 fold 0')
        for name, file in (('convnext', 'oof_k36_flip_tta.csv'), ('convnext_plain', 'oof_k36.csv')):
            paths[name][f] = run / file
        paths['h1'][f] = root / 'artifacts/experiments' / (
            f'EXP149_H1_S42_F{f}' if f < 2 else f'EXP153_H1_S42_F{f}') / 'oof_k36.csv'
        paths['radr50'][f] = root / 'artifacts/experiments' / (
            'EXP127_AUG1_S42_F0' if f == 0 else f'EXP134_AUG1_S42_F{f}') / 'oof_k36.csv'
        paths['dino'][f] = (root / '.local/exp160-collected-v1/BASE/oof_best.csv' if f == 0
                            else dino_outputs / dino_folds.tag(f) / 'oof_best.csv')
        for name in paths:
            path = paths[name][f]
            if name in ('h1', 'radr50'):
                inventory = read(path.parent / 'output-inventory.json')
                need(path.name in inventory and path.stat().st_size == inventory[path.name]['bytes']
                     and sha(path) == inventory[path.name]['sha256'],
                     'partner OOF inventory differs: ' + path.as_posix())
            if name == 'dino' and f:
                receipt = read(path.parent / 'receipt.json')
                need(receipt['status'] == 'PASS' and
                     receipt['outputs'][path.name]['sha256'] == sha(path),
                     'DINO-base receipt differs: ' + path.as_posix())
            if name == 'dino' and f == 0:
                receipt = read(path.parent / 'collect-receipt.json')
                need(receipt['status'] == 'COLLECTED_VALIDATED' and
                     receipt['prediction_sha256'] == sha(path),
                     'DINO-base fold 0 receipt differs')
            parts[name][f] = folds._prediction_values(path, ids_by_fold[f])
    refs_raw = e156.references(all_ids, root)
    refs = {'jev': refs_raw['jev_hybrid'], 'aquila': refs_raw['aquila_v1'],
            'dread': refs_raw['dread'], 'consensus': refs_raw['consensus']}
    standalone = {}
    for name in parts:
        pooled = np.concatenate([parts[name][f] for f in range(5)])
        per_fold, offset = {}, 0
        for f in range(5):
            sl = slice(offset, offset + len(ids_by_fold[f]))
            per_fold[str(f)] = {key: folds._score(parts[name][f],
                  (ref[0][sl], ref[1][sl], ref[2]), ev) for key, ref in refs.items()}
            offset = sl.stop
        standalone[name] = {'pooled': {key: folds._score(pooled, ref, ev)
                                       for key, ref in refs.items()}, 'per_fold': per_fold}
    ranks = {name: rank_per_fold(p, ev) for name, p in parts.items()}
    draws = ev.bootstrap_index(ev.component_groups(all_ids, root), 300, 20261003)
    h1 = ranks['h1']
    pair = {'h1_convnext_w01': folds._paired(.9*h1 + .1*ranks['convnext'], h1, refs, draws, ev),
            'radr50_convnext_w01': folds._paired(.9*ranks['radr50'] + .1*ranks['convnext'],
                                                ranks['radr50'], refs, draws, ev),
            'h1_dino_convnext_080_010_010': folds._paired(.8*h1 + .1*ranks['dino'] +
                                                         .1*ranks['convnext'], h1, refs, draws, ev),
            'h1_dino_convnext_070_015_015': folds._paired(.7*h1 + .15*ranks['dino'] +
                                                         .15*ranks['convnext'], h1, refs, draws, ev)}
    destination.mkdir(parents=True)
    with (destination / 'pooled_convnext_flip_tta.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([folds.UID, *folds.TARGETS])
        for uid, values in zip(all_ids, np.concatenate([parts['convnext'][f] for f in range(5)])):
            writer.writerow([uid, *(format(float(v), '.9g') for v in values)])
    result = {'schema': SCHEMA, 'version': VERSION,
              'verdict': 'PASS' if screen_gate(pair['h1_convnext_w01']) else 'FAIL',
              'rule': 'H1 + 0.1 ConvNeXt: mean(JEV,aquila) delta >0 and CI95 lower >0',
              'rows': len(all_ids), 'seed': 42, 'folds': list(range(5)),
              'bootstrap': {'unit': 'validation component', 'draws': 300, 'seed': 20261003},
              'oof_policy': {'convnext': 'oof_k36_flip_tta.csv',
                             'convnext_plain': 'oof_k36.csv', 'h1': 'oof_k36.csv',
                             'radr50': 'oof_k36.csv', 'dino': 'oof_best.csv'},
              'oof_files': {name: {str(f): path.as_posix() for f, path in entries.items()}
                            for name, entries in paths.items()},
              'input_sha256': {name: {str(f): sha(path) for f, path in entries.items()}
                               for name, entries in paths.items()},
              'reference_sha256': {name: sha(root / path) for name, path in
                                   (('jev', e156.JEV), ('aquila', e156.AQUILA),
                                    ('dread', e156.DREAD),
                                    *[(f'consensus_{i}', p) for i, p in enumerate(e156.CONSENSUS)])},
              'standalone': standalone, 'blends': pair,
              'pooled_oof_sha256': sha(destination / 'pooled_convnext_flip_tta.csv')}
    write(destination / 'result.json', result)
    return {'verdict': result['verdict'], 'rows': len(all_ids),
            'h1_convnext_w01_mean_delta': pair['h1_convnext_w01']['primary_mean_delta'],
            'h1_convnext_w01_ci95': pair['h1_convnext_w01']['primary_mean_ci95'],
            'result': (destination / 'result.json').as_posix()}


def _screen_overlay(outputs):
    # The local overlay is an input to collect; retain its path in collection.
    overlay = read(outputs / 'collection.json').get('overlay')
    need(overlay and Path(overlay).is_dir(), 'EXP167 local overlay unavailable for screen')
    return Path(overlay)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('validate'); s.add_argument('--package', type=Path, required=True)
    s.add_argument('--full', action='store_true')
    s = sub.add_parser('vast-stage'); s.add_argument('--overlay', type=Path, required=True)
    s.add_argument('--source', type=Path, default=SOURCE_ROOT)
    s.add_argument('--destination', type=Path, default=REMOTE_ROOT)
    s = sub.add_parser('vast-plan'); s.add_argument('--overlay', default='/workspace/exp167-overlay')
    s = sub.add_parser('run'); s.add_argument('--fold', type=int, required=True)
    s = sub.add_parser('export'); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('collect'); s.add_argument('--export', type=Path, required=True)
    s.add_argument('--overlay', type=Path, required=True); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('screen'); s.add_argument('--outputs', type=Path, default=ROOT / EXP / 'outputs')
    s.add_argument('--destination', type=Path, default=ROOT / EXP / 'screen')
    a = ap.parse_args(argv)
    if a.action == 'prepare': result = prepare(a.out)
    elif a.action == 'validate': result = {'status': 'VALIDATED', 'full': a.full,
                                          'folds': validate(a.package, a.full)['serial_folds']}
    elif a.action == 'vast-stage': result = vast_stage(a.overlay, a.source, a.destination)
    elif a.action == 'vast-plan': result = vast_plan(a.overlay)
    elif a.action == 'run': result = run(a.fold)
    elif a.action == 'export': result = export(a.out)
    elif a.action == 'collect': result = collect(a.export, a.overlay, a.out)
    else: result = screen(a.outputs, a.destination)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
