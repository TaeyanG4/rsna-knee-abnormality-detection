"""EXP170 v1.1.0: five serial Vast L2 ConvNeXt-T folds and a fixed CPU screen.

Only ``run`` enters the GPU. B stages, admits and launches the queue explicitly.
The overlay's ``vast-stage`` entry works before the older modules are hardlinked.
Changelog 1.1.0 (2026-10-12): elapsed/projection values are telemetry; no self-imposed run deadline.
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

VERSION = '1.1.0'
SCHEMA = 'exp170-convnext-t-l2-v1'
EXP = 'artifacts/experiments/EXP170_CONVNEXT_T_L2'
PROTOCOL = EXP + '/PROTOCOL.md'
TOOL = 'tools/exp170_convnext_t_l2.py'
RUNNER = 'training/runners/exp170_convnext_t_l2.py'
SOURCE_ROOT = '/workspace/exp166'
REMOTE_ROOT = '/workspace/exp170'
REMOTE_RESULTS = '/workspace/exp170-results'
SOURCE_ARCHIVE = '.local/exp166-vast-stage-v1/package/exp166-code.zip.bin'
SOURCE_RECEIPT = 'exp166-bundle.json'
SOURCE_MANIFEST = 'artifacts/experiments/EXP166_CONVNEXT_T/manifest.json'
SOURCE_WEIGHT = 'storage/weights/exp166/convnext_tiny_fb_in22k_ft_in1k.safetensors'
WEIGHT_SHA = '77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a'
FOLD_SOURCE = 'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1/s42/manifest.json'
LABEL = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv'
LABEL_PROVENANCE = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.provenance.json'
LABEL_SHA = '79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9'
FOLD0_ROSTER = 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold0/ordered-ids.json'
PILOT_CAP = 300
FULL_CAP = 2100
FOLD_CAP = 2500
MEASURED_MAX_SECONDS = 1534
REQUIRED_OUTPUTS = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                    'oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'}
REQUIRED_MODULES = {'tools/exp166_convnext_t.py', 'tools/exp127_radr50_single.py',
                    'tools/exp150_consensus_weight.py', 'tools/exp124_flip_tta.py',
                    'tools/exp087_training.py',
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
    need(fold in range(5), 'EXP170 trains folds 0-4 only')
    return f'EXP170_CONVNEXT_T_L2_S42_F{fold}'


def roster_path(fold):
    if fold == 0:
        return FOLD0_ROSTER
    return f'artifacts/experiments/EXP162_DINO_FOLDS/rosters/fold{fold}-sorted.json'


def protocol_path(fold):
    return f'protocols/fold{fold}.json'


def runner_source(root=ROOT):
    from tools import exp166_convnext_t as source
    old = source.runner_source((Path(root) / source.BASE_RUNNER).read_text(encoding='utf-8'))
    anchor = 'from tools import exp166_convnext_t as guard'
    need(old.count(anchor) == 1, 'EXP166 runner guard changed')
    new = old.replace(anchor, 'from tools import exp170_convnext_t_l2 as guard')
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
    if name in ('validate_arguments', 'validate_fold_membership', 'LABEL_SET'):
        raise AttributeError(name)
    return getattr(source, name)


LABEL_SET = 'l2_dread75_aquila25_v1'


def validate_fold_membership(corpus_ids, ordered, rows, canonical):
    from tools import exp166_convnext_t as source
    # EXP084 requires the supplied order to be monotonic in its canonical list.
    # EXP162's copies sort both lists, so the canonical list must be sorted too.
    return source.validate_fold_membership(corpus_ids, ordered, rows, sorted(canonical))


def queue():
    return [{'tag': tag(f), 'command': ['python', TOOL, 'run', '--fold', str(f)],
             'continue_on_failure': False} for f in range(5)]


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
    """Register L2 protocols in a slim overlay; EXP166 files remain hardlinks."""
    from tools import exp166_convnext_t as source
    from tools import exp164_dino_base_folds as base_folds
    from tools import exp162_dino_folds_kaggle as folds
    root, out = Path(root), Path(out)
    need(not out.exists(), 'fresh EXP170 overlay required')
    registered, _ = source.registration(root)
    files, receipt_sha = _source_files(root)
    derived = runner_source(root)
    guard_attributes(derived)
    provenance = read(root / LABEL_PROVENANCE)
    need(provenance['csv_sha256'] == LABEL_SHA and sha(root / LABEL) == LABEL_SHA
         and provenance['gold58_included'] is False and provenance['rows'] == 4347,
         'L2 provenance/label file differs')
    labels, semantic_sha = source.old.load_label_csv(root / LABEL)
    need(SOURCE_MANIFEST in files and SOURCE_WEIGHT in files and
         files[SOURCE_MANIFEST]['sha256'] == sha(root / SOURCE_MANIFEST),
         'source manifest or weight missing')
    reference = read(root / FOLD_SOURCE)
    rosters = base_folds.roster(root)  # checks byte-identical sorted EXP162 copies
    need(sha(root / FOLD0_ROSTER) ==
         registered['folds']['0']['ordered_ids']['sha256'],
         'EXP166 fold-0 ordered roster differs')
    payload = {TOOL: root / TOOL, PROTOCOL: root / PROTOCOL,
               'config/workspace.toml': root / 'config/workspace.toml',
               LABEL: root / LABEL, LABEL_PROVENANCE: root / LABEL_PROVENANCE,
               FOLD0_ROSTER: root / FOLD0_ROSTER}
    payload[registered['folds']['0']['groups']['path']] = root / registered['folds']['0']['groups']['path']
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
                'source_initial_state_sha256': registered['initial_state_sha256'],
                'source_weight_sha256': files[SOURCE_WEIGHT]['sha256'],
                'protocol_sha256': sha(root / PROTOCOL), 'serial_folds': [0, 1, 2, 3, 4],
                'caps_seconds': {'pilot': PILOT_CAP, 'full': FULL_CAP, 'fold': FOLD_CAP,
                                 'measured_max': MEASURED_MAX_SECONDS},
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
    for f in range(5):
        info = copy.deepcopy(registered['folds']['0'] if f == 0 else reference['folds'][str(f)])
        ids = read(out / roster_path(f))
        original_ids = read(root / info['ordered_ids']['path'])
        need(set(ids['train']) == set(original_ids['train']) and
             set(ids['val']) == set(original_ids['val']) and
             len(ids['train']) + len(ids['val']) == 4343,
             'EXP134 fold membership differs')
        vector = source.old.pos_weight(labels, ids['train'])
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
        q['paths']['labels'] = LABEL
        q['paths']['eval_labels'] = LABEL
        q['label_set'] = {'name': LABEL_SET, 'path': LABEL, 'sha256': LABEL_SHA,
                          'semantic_sha256': semantic_sha}
        q['recipe']['changes'] = list(registered['recipe']['changes']) + [
            'EXP170 label substitution only: Dread soft to L2 0.75 Dread + 0.25 aquila; fold-local clipped positive weights recomputed']
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
    need(m['schema'] == SCHEMA and m['version'] in ('1.0.0', '1.1.0', VERSION) and m['serial_folds'] == [0, 1, 2, 3, 4]
         and m['source_root'] == SOURCE_ROOT and m['remote_root'] == REMOTE_ROOT
         and m['caps_seconds'] == {'pilot': PILOT_CAP, 'full': FULL_CAP,
                                   'fold': FOLD_CAP, 'measured_max': MEASURED_MAX_SECONDS}, 'EXP170 package contract differs')
    need(read(package / 'queue.json') == queue(), 'serial queue differs')
    need(m['files']['config/workspace.toml']['origin'] == 'overlay' and
         m['files'][LABEL]['origin'] == 'overlay' and
         m['files'][LABEL]['sha256'] == LABEL_SHA and
         m['files'][LABEL_PROVENANCE]['origin'] == 'overlay' and
         m['files'][SOURCE_WEIGHT]['origin'] == 'exp166' and
         m['source_weight_sha256'] == m['files'][SOURCE_WEIGHT]['sha256'] and
         m['source_weight_sha256'] == WEIGHT_SHA, 'workspace/remote weight contract differs')
    need(m['protocol_sha256'] == m['files'][PROTOCOL]['sha256'] and
         m['source_manifest_sha256'] == m['files'][SOURCE_MANIFEST]['sha256'],
         'source/protocol binding differs')
    required = REQUIRED_MODULES | {TOOL, RUNNER}
    need(required.issubset(m['files']),
         'transitive runner/queue imports missing: ' + ', '.join(sorted(required - set(m['files']))))
    for name, entry in m['files'].items():
        safe(name)
        need(entry['origin'] in ('overlay', 'exp166'), 'unknown package source')
        if full or entry['origin'] == 'overlay':
            path = package / name
            need(path.is_file() and not path.is_symlink() and path.stat().st_size == entry['bytes']
                 and sha(path) == entry['sha256'], 'package file differs: ' + name)
    for f in range(5):
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
             q['paths']['labels'] == q['paths']['eval_labels'] == LABEL and
             q['label_set']['name'] == LABEL_SET and
             q['label_set']['sha256'] == LABEL_SHA and
             q['initial_state_sha256'] == m['source_initial_state_sha256'] and
             info['tag'] == tag(f) and info['ordered_ids']['path'] == record['roster'] and
             info['ordered_ids']['sha256'] == sha(package / record['roster']) and
             info['ordered_ids']['train_sha256'] == uid_hash(ids['train']) and
             info['ordered_ids']['val_sha256'] == uid_hash(ids['val']) and
             (f == 0 or (ids['train'] == sorted(ids['train']) and ids['val'] == sorted(ids['val']))) and
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
    need(f in range(5) and Path(a.protocol).resolve() == (root / protocol_path(f)).resolve()
         and a.variant == 'AUG1' and a.seed == 42 and a.tag == tag(f) and a.mode == 'full'
         and not a.validate_only, 'wrong EXP170 fold/seed/tag/mode')
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
    set_flag('--labels', root / LABEL)
    set_flag('--eval-labels', root / LABEL)
    return args


def verify_run(run, fold, package=ROOT):
    from tools import exp162_dino_folds_kaggle as folds
    package, run = Path(package), Path(run)
    m = validate(package, full=False)
    q = read(package / protocol_path(fold))
    info = q['folds'][str(fold)]
    need(run.name == tag(fold), 'wrong run tag')
    inv = read(run / 'output-inventory.json')
    need(set(inv) == REQUIRED_OUTPUTS, 'incomplete EXP170 output inventory')
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
         summary['label_set'] == LABEL_SET and
         summary['gold_access'] == 'disabled' and summary['weak_oof_macro_auc'] is None and
         '3090' in summary['environment']['gpu'] and
         '3090' in prov['environment']['gpu'],
         'incomplete fixed-final EXP170 fold')
    args = summary['args']
    need(all(args.get(k) == v for k, v in q['common_args'].items()) and
         args['seed'] == 42 and args['fold'] == fold and args['tag'] == tag(fold) and
         args['mode'] == 'full' and
         args['labels'] == f'{REMOTE_ROOT}/{LABEL}' and
         args['eval_labels'] == f'{REMOTE_ROOT}/{LABEL}' and
         prov['initial_state_sha256'] == q['initial_state_sha256'] and
         prov['init_weights']['weights_sha256'] == m['source_weight_sha256'] and
         prov['runner_sha256'] == m['files'][RUNNER]['sha256'] and
         prov['guard_sha256'] == m['files'][TOOL]['sha256'] and
         prov['protocol_sha256'] == m['folds'][str(fold)]['protocol_sha256'] and
         prov['train_ids_ordered_sha256'] == info['ordered_ids']['train_sha256'] and
         prov['eval_ids_ordered_sha256'] == info['ordered_ids']['val_sha256'] and
         prov['actual_pos_weight'] == info['pos_weight'] and
         prov['flip_tta'] == 'coronal_axial_horizontal_mean_probability' and
         prov['label_file_sha256'] == LABEL_SHA and
         prov['label_semantic_sha256'] == q['label_set']['semantic_sha256'] and
         prov['label_set'] == LABEL_SET,
         'fold recipe/provenance differs')
    ids = read(package / info['ordered_ids']['path'])['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        folds._prediction_values(run / name, ids)
    return {name: inv[name]['sha256'] for name in REQUIRED_OUTPUTS}


def run(fold):
    need(sys.platform.startswith('linux') and str(ROOT.resolve()) == REMOTE_ROOT,
         'EXP170 run requires exact Vast root')
    m = validate(ROOT, full=True)
    need(fold in range(5), 'wrong fold')
    admission = read(ROOT / 'admission.json')
    need(admission == {'status': 'READY', 'allow_execution': True,
                       'manifest_sha256': sha(ROOT / 'manifest.json'),
                       'serial_folds': [0, 1, 2, 3, 4], 'max_gpu_seconds_per_fold': FOLD_CAP},
         'B exact-manifest admission missing')
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout
    need('3090' in smi.splitlines()[0], 'Vast RTX 3090 absent')
    fold0 = read(Path('/workspace/exp166-results/EXP166_CONVNEXT_T_S42_F0/summary.json'))
    need(fold0['environment']['gpu'] == smi.splitlines()[0].strip() and
         fold0['environment']['hostname'] == os.uname().nodename,
         'EXP166 control fold 0 was not run on this Vast instance')
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
                       check=True)
    pilot = read(dest / 'pilot.json')
    need(math.isfinite(pilot['epoch_seconds']) and pilot['epoch_seconds'] > 0 and
         math.isfinite(pilot['peak_gpu_gib']) and 0 <= pilot['peak_gpu_gib'] < 20,
         'invalid pilot timing or memory')
    write(ROOT / f'execution-fold{fold}.json', {'pilot': pilot,
          'full_cap_seconds': FULL_CAP, 'fold_cap_seconds': FOLD_CAP,
          'manifest_sha256': sha(ROOT / 'manifest.json')})
    if not (dest / 'summary.json').is_file():
        subprocess.run(runner_args(fold, ROOT), cwd=ROOT, env=env, check=True)
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


def vast_plan(overlay='/workspace/exp170-overlay'):
    import shlex
    need(str(overlay).startswith('/workspace/') and '\\' not in str(overlay)
         and '..' not in PurePosixPath(str(overlay)).parts, 'absolute Vast overlay required')
    o = shlex.quote(str(overlay))
    return '\n'.join([
        '# B checks the 3090 is idle and claims Vast compute before launch.',
        f'python {o}/{TOOL} vast-stage --overlay {o}',
        f'cd {REMOTE_ROOT}',
        f'python {TOOL} validate --package . --full',
        "python -c 'import hashlib,json; h=hashlib.file_digest(open(\"manifest.json\",\"rb\"),\"sha256\").hexdigest(); json.dump(dict(status=\"READY\",allow_execution=True,manifest_sha256=h,serial_folds=[0,1,2,3,4],max_gpu_seconds_per_fold=2500),open(\"admission.json\",\"x\"))'",
        'export KNEE_WORKSPACE_ROOT=/workspace/exp170 HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1',
        'setsid nohup python tools/gpu_queue.py queue.json --foreground > exp170-queue.log 2>&1 < /dev/null & echo $! > exp170-queue.pid',
        '# After all five gpu_job SUCCESS markers exist, export:',
        f'python {TOOL} export --out /workspace/exp170-export.tar',
    ])


def export(out, root=ROOT):
    """CPU-only validated artifact tar; excludes resumable optimizer checkpoints."""
    root, out = Path(root), Path(out)
    need(str(root.resolve()) == REMOTE_ROOT and not out.exists(), 'Vast fresh export required')
    m = validate(root, full=True)
    with tarfile.open(out, 'x') as tar:
        tar.add(root / 'manifest.json', arcname='manifest.json')
        for f in range(5):
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
    expected = {'manifest.json', *(f'validated-fold{f}.json' for f in range(5))}
    expected.update(f'results/{tag(f)}/{name}' for f in range(5)
                    for name in REQUIRED_OUTPUTS | {'output-inventory.json'})
    out.mkdir(parents=True)
    try:
        with tarfile.open(export_file, 'r') as tar:
            members = tar.getmembers()
            need(len(members) == len(expected) and {x.name for x in members} == expected
                 and all(x.isfile() and safe(x.name) == x.name for x in members),
                 'unsafe/incomplete EXP170 export')
            for member in members:
                dest = out / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        need(sha(out / 'manifest.json') == sha(overlay / 'manifest.json'),
             'export manifest differs')
        completed = []
        for f in range(5):
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


def _screen_overlay(outputs):
    # The local overlay is an input to collect; retain its path in collection.
    overlay = read(outputs / 'collection.json').get('overlay')
    need(overlay and Path(overlay).is_dir(), 'EXP170 local overlay unavailable for screen')
    return Path(overlay)


def paired_jev_consensus(candidate, baseline, refs, draws, evaluator):
    """Paired component bootstrap with the prospective JEV/consensus mean."""
    import numpy as np
    from tools.exp162_dino_folds_kaggle import TARGETS

    details, primary_draws = {}, {}
    for name, (y, mask, minimum) in refs.items():
        ca = evaluator.auc_targets(y, candidate, mask, minimum)
        ba = evaluator.auc_targets(y, baseline, mask, minimum)
        values = np.asarray([
            evaluator.macro(evaluator.auc_targets(y[ix], candidate[ix], mask[ix], minimum)) -
            evaluator.macro(evaluator.auc_targets(y[ix], baseline[ix], mask[ix], minimum))
            for ix in draws], dtype=np.float64)
        c, b = evaluator.macro(ca), evaluator.macro(ba)
        if name in ('jev', 'consensus'):
            need(np.isfinite(c) and np.isfinite(b) and np.isfinite(values).all(),
                 'primary reference AUC/bootstrap undefined: ' + name)
            primary_draws[name] = values
        finite = values[np.isfinite(values)]
        details[name] = {'candidate': float(c) if np.isfinite(c) else None,
                         'baseline': float(b) if np.isfinite(b) else None,
                         'delta': float(c-b) if np.isfinite(c-b) else None,
                         'ci95': np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
                         'per_target_delta': {t: float(a-z) if np.isfinite(a-z) else None
                                              for t, a, z in zip(TARGETS, ca, ba)}}
    mean_draws = (primary_draws['jev'] + primary_draws['consensus']) / 2.0
    point = (details['jev']['delta'] + details['consensus']['delta']) / 2.0
    return {'references': details, 'primary_references': ['jev', 'consensus'],
            'primary_mean_delta': point,
            'primary_mean_ci95': np.percentile(mean_draws, [2.5, 97.5]).tolist()}


def admission(pair):
    dread = pair['references']['dread']['delta']
    return {'status': 'ADMIT' if pair['primary_mean_delta'] > 0 and
            pair['primary_mean_ci95'][0] > 0 and dread is not None and dread >= -.003
            else 'HOLD', 'mean_positive': pair['primary_mean_delta'] > 0,
            'mean_ci_lower_positive': pair['primary_mean_ci95'][0] > 0,
            'dread_floor': dread is not None and dread >= -.003}


def screen(outputs=ROOT / EXP / 'outputs', destination=ROOT / EXP / 'screen', root=ROOT):
    """CPU-only paired L2/Dread and fixed equal-thirds bundle screens."""
    import numpy as np
    from tools import exp149_label_sources as e149
    from tools import exp156_k_sweep as e156
    from tools import exp162_dino_folds_kaggle as folds
    from tools import exp167_convnext_t_folds as control
    from tools import exp169_bundle_screen as bundle_helper
    from tools import exp115_vlm_eval as ev
    from tools import exp166_convnext_t as source

    outputs, destination, root = map(Path, (outputs, destination, root))
    need(not destination.exists(), 'fresh screen directory required')
    collection = read(outputs / 'collection.json')
    need(collection['schema'] == SCHEMA and collection['status'] == 'VERIFIED' and
         collection['completed_folds'] == list(range(5)) and
         collection['manifest_sha256'] == sha(outputs / 'manifest.json'),
         'EXP170 five-fold collection incomplete')
    bundle_helper.readiness(root)
    overlay = _screen_overlay(outputs)
    manifest = e149._load_manifest('student', root)
    canonical, fold_of = e149.fold_rosters(manifest, root)
    ids_by_fold = {f: [uid for uid in canonical if fold_of[uid] == f] for f in range(5)}
    ids = [uid for f in range(5) for uid in ids_by_fold[f]]
    need(len(ids) == len(set(ids)) == 4343, 'pooled held-out coverage differs')
    controls = read(root / source.RESULT_DIR / 'vast' / source.TAG / 'summary.json')
    source.verify_run(root / source.RESULT_DIR / 'vast' / source.TAG, root)
    control_outputs = root / control.EXP / 'outputs'
    control_overlay = control._screen_overlay(control_outputs)
    sections = {n: [] for n in ('l2', 'dino', 'convnext', 'control_flip',
                                'convnext_plain', 'control_plain')}
    files, hashes, per_fold = {}, {}, {}
    refs_raw = e156.references(ids, root)
    refs = {name: refs_raw[key] for name, key in
            (('jev', 'jev_hybrid'), ('aquila', 'aquila_v1'),
             ('dread', 'dread'), ('consensus', 'consensus'))}
    offset = 0
    for f in range(5):
        run = outputs / 'results' / tag(f)
        verify_run(run, f, overlay)
        if f:
            control.verify_run(bundle_helper.paths(root, f)['convnext'].parent,
                               f, control_overlay)
        summary = read(run / 'summary.json')
        need(summary['environment']['gpu'] == controls['environment']['gpu'] and
             summary['environment']['hostname'] == controls['environment']['hostname'],
             'EXP170 hardware/host differs from EXP166/167 control')
        p = bundle_helper.paths(root, f)
        local = {'l2_s42': p['l2_s42'], 'l2_s43': p['l2_s43'], 'dino': p['dino'],
                 'control_flip': p['convnext'],
                 'control_plain': p['convnext'].with_name('oof_k36.csv'),
                 'convnext': run / 'oof_k36_flip_tta.csv',
                 'convnext_plain': run / 'oof_k36.csv'}
        fid = ids_by_fold[f]
        values = {}
        for name, path in local.items():
            bundle_helper._verify_inventory(path)
            files[f'{name}_f{f}'] = str(path.relative_to(root))
            hashes[f'{name}_f{f}'] = sha(path)
            values[name] = bundle_helper.load_oof(path, fid)
        sections['l2'].append((ev.pct_rank(values['l2_s42']) +
                               ev.pct_rank(values['l2_s43'])) / 2.0)
        for name in ('dino', 'convnext', 'control_flip', 'convnext_plain', 'control_plain'):
            sections[name].append(ev.pct_rank(values[name]))
        sl = slice(offset, offset + len(fid))
        fold_refs = {name: (ref[0][sl], ref[1][sl], ref[2]) for name, ref in refs.items()}
        per_fold[str(f)] = {}
        for policy, current, old in (('flip_tta', 'convnext', 'control_flip'),
                                     ('plain', 'convnext_plain', 'control_plain')):
            scores = {name: (folds._score(values[current], ref, ev)['macro'],
                             folds._score(values[old], ref, ev)['macro'])
                      for name, ref in fold_refs.items()}
            per_fold[str(f)][policy] = {name: {
                'l2': current_score, 'dread': control_score,
                'delta': (current_score - control_score if current_score is not None
                          and control_score is not None else None)}
                for name, (current_score, control_score) in scores.items()}
        offset = sl.stop
    pooled = {name: np.concatenate(parts) for name, parts in sections.items()}
    draws = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    direct = {'flip_tta': paired_jev_consensus(pooled['convnext'],
               pooled['control_flip'], refs, draws, ev),
              'plain': paired_jev_consensus(pooled['convnext_plain'],
               pooled['control_plain'], refs, draws, ev)}
    bundle, ranked = bundle_helper.bundle_of({name: pooled[name] for name in
                                               bundle_helper.FAMILIES})
    bundle_pair = paired_jev_consensus(bundle, ranked['l2'], refs, draws, ev)
    result = {'schema': SCHEMA, 'version': VERSION,
              'protocol_sha256': sha(root / PROTOCOL),
              'status': admission(bundle_pair), 'studies': len(ids),
              'folds': list(range(5)), 'seed': 42,
              'bootstrap': {'unit': 'validation component', 'draws': 300,
                            'seed': 20261003},
              'gate': 'mean(JEV,consensus) delta >0, paired CI95 lower >0, Dread delta >= -0.003',
              'aquila_policy': 'self-referenced to L2; report only',
              'oof_policy': {'l2_s42': 'plain oof_k36.csv',
                             'l2_s43': 'plain oof_k36.csv',
                             'dino': 'plain oof_best.csv',
                             'convnext': 'flip-TTA oof_k36_flip_tta.csv',
                             'convnext_plain': 'plain oof_k36.csv, direct pair only',
                             'control_flip': 'flip-TTA oof_k36_flip_tta.csv',
                             'control_plain': 'plain oof_k36.csv'},
              'aggregation': 'within-fold target ranks; L2 seed mean; pooled global family ranks; equal thirds; global bundle rerank',
              'input_files': files, 'input_sha256': hashes,
              'direct_l2_minus_dread': direct, 'direct_per_fold': per_fold,
              'bundle_minus_l2': bundle_pair,
              'standalone': {name: {ref: folds._score(values, r, ev) for ref, r in refs.items()}
                             for name, values in {**ranked, 'bundle': bundle}.items()},
              'reference_sha256': {k: sha(root / v) for k, v in
                                   (('jev', e156.JEV), ('aquila', e156.AQUILA),
                                    ('dread', e156.DREAD),
                                    *[(f'consensus_{i}', p) for i, p in enumerate(e156.CONSENSUS)])}}
    destination.mkdir(parents=True)
    write(destination / 'result.json', result)
    return {'status': result['status']['status'], 'result': str(destination / 'result.json'),
            'mean_delta': bundle_pair['primary_mean_delta'],
            'ci95': bundle_pair['primary_mean_ci95'],
            'dread_delta': bundle_pair['references']['dread']['delta']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('validate'); s.add_argument('--package', type=Path, required=True)
    s.add_argument('--full', action='store_true')
    s = sub.add_parser('vast-stage'); s.add_argument('--overlay', type=Path, required=True)
    s.add_argument('--source', type=Path, default=SOURCE_ROOT)
    s.add_argument('--destination', type=Path, default=REMOTE_ROOT)
    s = sub.add_parser('vast-plan'); s.add_argument('--overlay', default='/workspace/exp170-overlay')
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
