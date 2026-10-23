"""EXP153 H1 continuation v1.0.0. All commands here are CPU/offline plans.

prepare, validate, vast-stage, vast-plan, local-queue, collect, verify.
The generated queue is launched separately by B after a compute claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = '1.0.0'
EXP = 'artifacts/experiments/EXP153_H1_AQUILA'
MANIFEST = EXP + '/manifest.json'
INITIAL = EXP + '/initial-state.pt'
SOURCE = 'artifacts/experiments/EXP149_LABEL_SOURCES/students/manifest.json'
RUNNER = 'training/runners/exp153_h1.py'
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
CORPUS_SHA = 'f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
INITIAL_TENSOR_SHA = 'a56516a289178dce17d2fe736147ddbbcb52a571cb25e6c8b59b4342a3d3e636'
HEADS = ('dread', 'relabel_v1', 'steven_v4', 'pilkwang_v2', 'aquila_v1')
NEW = tuple((seed, fold) for seed in (42, 43) for fold in range(5) if seed == 43 or fold >= 2)
ALL = tuple((seed, fold) for seed in (42, 43) for fold in range(5))
REMOTE = '/workspace/exp153-h1'
REMOTE_CORPUS = '/workspace/rsna-hc06-illinois-pair-v2/storage/cache/derived/RAPTOR64_336_C140_S02'
KNOWN_HOSTS = '.local/hc06-illinois-stable-20261009-v10/known_hosts-54941539-v1'
SSH_KEY = 'C:/Users/Taeyang/.ssh/id_ed25519'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(obj, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def tag(seed, fold):
    return f'EXP153_H1_S{seed}_F{fold}'


def source_tag(fold):
    return f'EXP149_H1_S42_F{fold}'


def source_registration(root=ROOT):
    from tools import exp149_label_sources as e149
    root = Path(root)
    p = e149._load_manifest('student', root)
    need(p['runs'][source_tag(0)]['initial_state_sha256'] == INITIAL_TENSOR_SHA,
         'EXP149 five-head tensor registration differs')
    need(tuple(p['runs'][source_tag(0)]['heads']) == HEADS, 'EXP149 head order differs')
    need(p['multihead']['loss'] == 'sum of five equally weighted masked per-head BCE terms',
         'EXP149 loss rule differs')
    need(p['multihead']['serving'] == 'per-target equal mean of five head average-ranks / roster size',
         'EXP149 rank rule differs')
    need(sha256(root / CORPUS / 'manifest.json') == CORPUS_SHA, 'Local corpus manifest differs')
    return p


def build_initial_file(root, source):
    """Only prepare may draw the original seed-42 CPU state. Runs load this file."""
    import torch
    from tools import exp149_label_sources as e149
    sys.path.insert(0, str(root / 'training/src'))
    from raptor_common import seed_all
    from raptor_radimagenet import build_radimagenet_model
    from tools import exp134_aug1_kfold as aug
    from tools import exp117_multihead as mh
    aug.forbid_cuda_init()
    aug.worker_init(0)
    seed_all(42)
    with torch.device('cpu'):
        base, _ = build_radimagenet_model(source['common_args']['arch'],
                                         aug.weights_path(root), source['init_weights']['sha256'])
        base_hash, _ = aug.state_hashes(base)
        need(base_hash == source['initial_state_sha256'], 'EXP149 base initial tensor hash differs')
        model = mh.build_multihead(base, 5)
    full_hash, _ = aug.state_hashes(model)
    need(full_hash == INITIAL_TENSOR_SHA, 'EXP149 full initial tensor hash differs')
    need(not torch.cuda.is_initialized(), 'CPU prepare initialized CUDA')
    path = root / INITIAL
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), 'Initial state file already exists')
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)
    return sha256(path)


def prepare(root=ROOT):
    from tools import exp149_label_sources as e149
    from tools import exp130_gold58_compare as gold
    root = Path(root)
    need(not (root / MANIFEST).exists() and not (root / INITIAL).exists(),
         'EXP153 registration already exists; no overwrite')
    source = source_registration(root)
    base = e149.base_protocol(root)
    canonical, fold_of = e149.fold_rosters(base, root)
    gold_ids, _ = gold.official_gold(root)
    need(not set(canonical) & set(gold_ids), 'Gold58 intersects EXP099 roster')
    h0 = source['runs'][source_tag(0)]
    head_recs = h0['head_labels']
    need(tuple(x['name'] for x in head_recs) == HEADS, 'Head label order differs')
    dread, dread_rec = e149.source_record(h0['labels'], root)
    for head in head_recs:
        _, rec = e149.source_record(head['path'], root)
        need(rec == {k: head[k] for k in ('path', 'sha256', 'semantic_sha256')},
             'Head label source changed: ' + head['name'])
    runs = {}
    for seed, fold in NEW:
        ordered = e149.ordered_for(canonical, fold_of, [k for k in range(5) if k != fold], [fold])
        spec = e149._spec(tag(seed, fold), 'student', 'H1', [k for k in range(5) if k != fold],
                          [fold], ordered, dread, dread_rec, base, heads=HEADS,
                          head_labels=head_recs, root=root)
        spec['initial_state_sha256'] = INITIAL_TENSOR_SHA
        spec['seed'] = seed
        runs[spec['tag']] = spec
    for fold in (0, 1):
        need(source['runs'][source_tag(fold)]['ordered_ids'] ==
             e149.ordered_for(canonical, fold_of, [k for k in range(5) if k != fold], [fold]),
             'Existing EXP149 fold roster differs')
        old = root / 'artifacts/experiments' / source_tag(fold)
        summary, provenance, inventory = (read(old / name) for name in
                                          ('summary.json', 'provenance.json', 'output-inventory.json'))
        need(summary['selected_epoch_1based'] == 20 and summary['skipped_updates'] == 0 and
             summary['gold'] == 0 and provenance['full_model_initial_state_sha256'] ==
             INITIAL_TENSOR_SHA and provenance['protocol_sha256'] == sha256(root / SOURCE) and
             inventory['model_best.pt']['sha256'] == sha256(old / 'model_best.pt'),
             'Existing EXP149 H1 checkpoint is not a clean registered fold')
    initial_sha = build_initial_file(root, source)
    p = copy.deepcopy(source)
    p.update(schema='exp153-h1-advance-v1', version=VERSION, status='REGISTERED_NO_EXECUTION',
             experiment='EXP153_H1_AQUILA', seed=42, run_kind='student', runs=runs)
    p['exp153'] = {'source_manifest_sha256': sha256(root / SOURCE),
                   'initial_state_file': INITIAL, 'initial_state_file_sha256': initial_sha,
                   'full_model_initial_state_sha256': INITIAL_TENSOR_SHA,
                   'corpus_manifest_sha256': CORPUS_SHA,
                   'prior_tags': [source_tag(0), source_tag(1)],
                   'new_tags': list(runs), 'all_seeds': [42, 43],
                   'independent_trajectories': True, 'fixed_final_epochs': 20,
                   'loss': source['multihead']['loss'], 'serving': source['multihead']['serving']}
    p['source_bindings'][SOURCE] = sha256(root / SOURCE)
    p['source_bindings'][RUNNER] = sha256(root / RUNNER)
    p['source_bindings']['tools/exp153_h1_advance.py'] = sha256(Path(__file__))
    write_new(root / MANIFEST, p)
    return {'status': p['status'], 'manifest': MANIFEST, 'new_runs': list(runs),
            'initial_state_file_sha256': initial_sha,
            'full_model_initial_state_sha256': INITIAL_TENSOR_SHA}


def registration(root=ROOT):
    root = Path(root)
    p = read(root / MANIFEST)
    need(p['schema'] == 'exp153-h1-advance-v1' and p['version'] == VERSION and
         tuple(p['runs']) == tuple(tag(s, f) for s, f in NEW), 'EXP153 manifest differs')
    need(p['exp153']['source_manifest_sha256'] == sha256(root / SOURCE),
         'EXP149 source manifest differs')
    need(p['exp153']['initial_state_file_sha256'] == sha256(root / INITIAL),
         'Initial state file differs')
    need(p['exp153']['full_model_initial_state_sha256'] == INITIAL_TENSOR_SHA,
         'Full model tensor registration differs')
    for name, digest in p['source_bindings'].items():
        need(sha256(root / name) == digest, 'Source binding differs: ' + name)
    return p


def validate_arguments(a, root=ROOT):
    """EXP153 runner guard; no CUDA or run output."""
    from tools import exp149_label_sources as e149
    root = Path(root)
    p = registration(root)
    need(a.run in p['runs'] and a.tag == a.run and Path(a.protocol).resolve() == (root / MANIFEST).resolve(),
         'Run tag/protocol differs')
    s = p['runs'][a.run]
    need((a.seed, a.fold) in NEW and a.run == tag(a.seed, a.fold), 'Seed/fold differs')
    need(a.variant == 'AUG1' and a.mode == 'full' and not os.environ.get('KNEE_REMOTE_RELAX_ENV'),
         'EXP153 requires full AUG1 without relaxation')
    need(a.surface in ('local', 'vast') and a.device == ('cpu' if a.validate_only else 'cuda'),
         'Wrong device/surface')
    need(0 <= a.workers <= 12, 'Workers outside 0..12')
    for key, value in p['common_args'].items():
        if key == 'device' and a.validate_only:
            continue
        need(getattr(a, key) == value, 'Recipe differs: ' + key)
    need(Path(a.initial_state).resolve() == (root / INITIAL).resolve(), 'Initial state path differs')
    need(Path(a.corpus).resolve() == (root / CORPUS).resolve(), 'Corpus path differs')
    need(Path(a.labels).resolve() == (root / s['labels']).resolve() and
         Path(a.eval_labels).resolve() == (root / p['paths']['eval_labels']).resolve(),
         'Label path differs')
    need(Path(a.init_weights).resolve() == e149.aug.weights_path(root).resolve(),
         'RadImageNet source path differs')
    if a.surface == 'local':
        need(sha256(a.init_weights) == p['init_weights']['sha256'], 'Local RadImageNet weights differ')
    need(sha256(root / CORPUS / 'manifest.json') == CORPUS_SHA, 'Corpus manifest differs')
    canonical, fold_of = e149.fold_rosters(p, root)
    ordered = e149.ordered_for(canonical, fold_of, s['train_folds'], s['eval_folds'])
    need(ordered == s['ordered_ids'] and e149.uid_hash(ordered['train']) == s['train_ids_sha256']
         and e149.uid_hash(ordered['val']) == s['eval_ids_sha256'], 'EXP099 ordered roster differs')
    need(tuple(s['heads']) == HEADS and s['initial_state_sha256'] == INITIAL_TENSOR_SHA,
         'H1 head/initial hash differs')
    for head in s['head_labels']:
        need(sha256(root / head['path']) == head['sha256'], 'Head label file differs')
    need(sha256(root / s['labels']) == s['label_sha256'], 'Training label file differs')
    out = root / 'artifacts/experiments'
    need(a.output_root is None or Path(a.output_root).resolve() == out.resolve(),
         'Full output root differs')
    need(not (out / a.run).exists(), 'Run output already exists')
    return p, s, ordered, out


def runner_argv(p, run, root=ROOT, workers=12, *, validate_only=False,
                surface='local', execution_root=None):
    from tools import exp149_label_sources as e149
    root = Path(root)
    s = p['runs'][run]
    ref = source_registration(root)
    argv = e149.runner_argv(ref, source_tag(0), root, workers)
    argv[1] = str(root / RUNNER)
    for flag, value in (('--protocol', root / MANIFEST), ('--run', run), ('--tag', run),
                        ('--fold', s['fold']), ('--seed', s['seed'])):
        argv = e149.aug._set_flag(argv, flag, str(value))
    argv += ['--initial-state', str(root / INITIAL), '--surface', surface]
    if execution_root is not None:
        # Rewrite only path arguments whose prefix is the registered local root.
        # No result file is produced on the local machine by this operation.
        local_prefix = str(root.resolve())
        remote_prefix = str(Path(execution_root))
        argv = [remote_prefix + x[len(local_prefix):].replace('\\', '/') if
                isinstance(x, str) and x.startswith(local_prefix) else x for x in argv]
    if validate_only:
        argv = e149.aug._set_flag(argv, '--device', 'cpu') + ['--validate-only']
    return argv


def validate(root=ROOT):
    p = registration(root)
    from tools import exp149_label_sources as e149
    for run, spec in p['runs'].items():
        need(tuple(spec['heads']) == HEADS and spec['initial_state_sha256'] == INITIAL_TENSOR_SHA,
             'Run registration differs: ' + run)
        need(spec['seed'] in (42, 43) and spec['fold'] in range(5), 'Seed/fold outside registration')
        need(spec['n_train'] + spec['n_val'] == 4343 and spec['steps'] == spec['scheduler_total_steps'],
             'EXP099 roster/steps differ: ' + run)
        need(spec['ordered_ids']['train'] == e149.ordered_for(*e149.fold_rosters(p, root),
              spec['train_folds'], spec['eval_folds'])['train'], 'Fold train roster differs')
        argv = runner_argv(p, run, root, workers=0, validate_only=True)
        argv[0] = sys.executable
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUTF8='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
        result = subprocess.run(argv, cwd=root, env=env, check=False)
        need(result.returncode == 0, 'CPU runner validation failed: ' + run)
    return {'status': 'VALIDATED_NO_TRAINING', 'runs': len(p['runs']),
            'full_model_initial_state_sha256': INITIAL_TENSOR_SHA}


def safe_member(name):
    pure = Path(name.replace('\\', '/'))
    need(name == pure.as_posix() and not pure.is_absolute() and '..' not in pure.parts
         and ':' not in name and name != '', 'Unsafe archive name: ' + name)


def package_files(root=ROOT):
    root = Path(root)
    p = registration(root)
    names = {MANIFEST, INITIAL, SOURCE, RUNNER, 'tools/exp153_h1_advance.py',
             'tools/__init__.py',
             p['paths']['base_ordered_ids'], p['paths']['base_groups'],
             p['paths']['folds'], p['paths']['eval_labels'],
             p['fold_protocol']['path'], p['fold_protocol']['label_registration']['path'],
             'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1/s42/manifest.json'}
    names |= set(p['source_bindings'])
    names |= {r['path'] for f in p['folds'].values() for r in (f['ordered_ids'], f['groups'])}
    names |= {x['path'] for s in p['runs'].values() for x in s['head_labels']}
    names |= {s['labels'] for s in p['runs'].values()}
    names |= {x.relative_to(root).as_posix() for directory in ('tools', 'training/src')
              for x in (root / directory).glob('*.py')}
    files = {}
    for name in sorted(names):
        safe_member(name)
        file = root / name
        need(file.is_file() and not file.is_symlink(), 'Missing or symlinked package input: ' + name)
        files[name] = {'sha256': sha256(file), 'bytes': file.stat().st_size}
    need(files[INITIAL]['sha256'] == p['exp153']['initial_state_file_sha256'],
         'Initial state package binding differs')
    return files


def verify_corpus(directory):
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink(), 'Corpus path is missing or symlinked')
    manifest = directory / 'manifest.json'
    need(not manifest.is_symlink() and sha256(manifest) == CORPUS_SHA, 'Corpus manifest SHA differs')
    m = read(manifest)
    need(m['contract'] == 'raptor64_336_c140_s02' and m['finished'], 'Corpus incomplete')
    items = [('ids.npy', m['ids_sha256']), ('masks.npy', m['masks_sha256'])]
    for shard in m['shards']:
        need(shard['done'] and Path(shard['file']).name == shard['file'], 'Unsafe/incomplete shard')
        items.append((shard['file'], shard['sha256']))
    for name, digest in items:
        file = directory / name
        need(file.is_file() and not file.is_symlink() and sha256(file) == digest,
             'Corpus member SHA differs: ' + name)
    return {'manifest_sha256': CORPUS_SHA, 'members_hashed': len(items)}


def remote_link():
    """B invokes this only on Vast after unpacking; no symlink escapes root."""
    need(sys.platform.startswith('linux') and ROOT.resolve() == Path(REMOTE).resolve(),
         'Wrong Vast root/platform')
    source = Path(REMOTE_CORPUS)
    need(source.resolve() == source and not source.is_symlink(), 'Remote corpus root differs')
    before = verify_corpus(source)
    target = ROOT / CORPUS
    need(not target.exists() and not target.is_symlink(), 'Remote target already exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    need(target.parent.resolve().is_relative_to(ROOT.resolve()), 'Corpus target escapes remote root')
    target.mkdir()
    subprocess.run(['cp', '-al', str(source) + '/.', str(target)], check=True)
    after = verify_corpus(target)
    need(after == before, 'Hardlinked corpus differs')
    return {'status': 'HARDLINKED_HASH_VERIFIED', **after, 'source': str(source), 'target': str(target)}


def stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    need(not out.exists(), 'Use a new Vast stage directory')
    files = package_files(root)
    out.mkdir(parents=True)
    archive = out / 'exp153-code.zip.bin'
    marker = b'# EXP153 isolated workspace; no credentials.\n'
    receipt = {'schema': 'exp153-vast-package-v1', 'version': VERSION,
               'manifest_sha256': sha256(root / MANIFEST),
               'corpus_manifest_sha256': CORPUS_SHA, 'files': files,
               'marker_sha256': hashlib.sha256(marker).hexdigest(), 'no_execution': True}
    rb = (json.dumps(receipt, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name in sorted(files):
            z.write(root / name, name)
        z.writestr('config/workspace.toml', marker)
        z.writestr('exp153-bundle.json', rb)
    (out / 'exp153-bundle.json').write_bytes(rb)
    write_new(out / 'stage-receipt.json', {'schema': 'exp153-vast-stage-v1',
              'archive_sha256': sha256(archive),
              'bundle_sha256': hashlib.sha256(rb).hexdigest(),
              'manifest_sha256': receipt['manifest_sha256'], 'tags': list(registration(root)['runs'])})
    remote_root = Path(REMOTE)
    queue = []
    for run in registration(root)['runs']:
        entry = {'tag': run,
                 'command': ['python', RUNNER, *runner_argv(registration(root), run, root,
                    workers=12, surface='vast', execution_root=remote_root)[2:]],
                 'continue_on_failure': False}
        # runner_argv above binds to the unpacked root; no local corpus is copied.
        if queue:
            entry['after'] = queue[-1]['tag']
        queue.append(entry)
    write_new(out / 'vast-queue.json', queue)
    verify_stage(out, root)
    return {'status': 'STAGED_NO_EXECUTION', 'archive': str(archive), 'runs': len(queue),
            'archive_sha256': sha256(archive)}


def verify_stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    receipt = read(out / 'stage-receipt.json')
    need(receipt['schema'] == 'exp153-vast-stage-v1' and
         sha256(out / 'exp153-code.zip.bin') == receipt['archive_sha256'] and
         sha256(out / 'exp153-bundle.json') == receipt['bundle_sha256'], 'Stage receipt differs')
    bundle = read(out / 'exp153-bundle.json')
    with zipfile.ZipFile(out / 'exp153-code.zip.bin') as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == set(bundle['files']) |
             {'config/workspace.toml', 'exp153-bundle.json'}, 'Archive has missing/duplicate entries')
        for name, rec in bundle['files'].items():
            data = z.read(name)
            need(len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256']
                 and sha256(root / name) == rec['sha256'], 'Archive entry differs: ' + name)
        need(hashlib.sha256(z.read('config/workspace.toml')).hexdigest() ==
             bundle['marker_sha256'], 'Workspace marker differs')
    q = read(out / 'vast-queue.json')
    need([x['tag'] for x in q] == list(registration(root)['runs']) and
         all(x['continue_on_failure'] is False for x in q), 'Vast queue differs')
    return receipt


def vast_plan(out, host='<remote-host>', port=32358, root=ROOT):
    verify_stage(out, root)
    need(host == '<remote-host>' and port == 32358, 'Endpoint must be re-reviewed by B')
    out = Path(out)
    ssh = f'-i {SSH_KEY} -o UserKnownHostsFile={KNOWN_HOSTS} -o StrictHostKeyChecking=yes -p {port}'
    scp = f'-i {SSH_KEY} -o UserKnownHostsFile={KNOWN_HOSTS} -o StrictHostKeyChecking=yes -P {port}'
    remote = f'root@{host}'
    archive = out / 'exp153-code.zip.bin'
    return '\n'.join([
        '# B: verify live Vast instance/endpoint, cost, GPU idle and Active compute claim first.',
        f'scp {scp} {archive.as_posix()} {remote}:/workspace/exp153-code.zip.bin',
        f'ssh {ssh} {remote} "test ! -e {REMOTE} && mkdir {REMOTE} && python -m zipfile -e /workspace/exp153-code.zip.bin {REMOTE}"',
        f'ssh {ssh} {remote} "cd {REMOTE} && python tools/exp153_h1_advance.py remote-link"',
        f'scp {scp} {(out / "vast-queue.json").as_posix()} {remote}:{REMOTE}/vast-queue.json',
        f'ssh {ssh} {remote} "cd {REMOTE} && KNEE_WORKSPACE_ROOT={REMOTE} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 setsid nohup python tools/gpu_queue.py vast-queue.json --foreground > exp153-queue.log 2>&1 < /dev/null &"',
        f'New-Item -ItemType Directory -Path {(out / "download").as_posix()} -ErrorAction Stop',
        f'scp {scp} -r {remote}:{REMOTE}/artifacts/experiments/EXP153_H1_S* {(out / "download").as_posix()}/',
        f'& $gpuPython tools/exp153_h1_advance.py collect --source {(out / "download").as_posix()}',
        '& $gpuPython tools/exp153_h1_advance.py verify',
    ])


def local_queue(out, after='none', workers=12, root=ROOT):
    root, out = Path(root), Path(out)
    p = registration(root)
    need(0 <= workers <= 12 and not out.exists(), 'Unsafe local queue target/workers')
    previous = None if after == 'none' else after
    entries = []
    for run in p['runs']:
        cmd = runner_argv(p, run, root, workers, surface='local')
        entry = {'tag': run, 'command': cmd, 'continue_on_failure': False}
        if previous:
            entry['after'] = previous
        entries.append(entry)
        previous = run
    write_new(out, entries)
    return {'status': 'QUEUED_NO_EXECUTION', 'path': str(out), 'runs': len(entries)}


def verify_run(run_dir, seed, fold, root=ROOT):
    """CPU-only inventory, provenance, roster and zero-skip gate for one model."""
    import csv
    from tools import exp149_label_sources as e149
    root, run_dir = Path(root), Path(run_dir)
    if seed == 42 and fold in (0, 1):
        protocol = source_registration(root)
        name = source_tag(fold)
        spec = protocol['runs'][name]
        protocol_file = root / SOURCE
    else:
        protocol = registration(root)
        name = tag(seed, fold)
        spec = protocol['runs'][name]
        protocol_file = root / MANIFEST
    need(run_dir.name == name, 'Wrong run directory name')
    inv = read(run_dir / 'output-inventory.json')
    required = ({'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                 'oof_best.csv', 'oof_k36.csv'} |
                {f'oof_head_{head}{suffix}.csv' for head in HEADS for suffix in ('', '_k36')})
    need(set(inv) == required, 'Incomplete H1 output inventory: ' + name)
    for filename, item in inv.items():
        need(Path(filename).name == filename and
             (run_dir / filename).stat().st_size == item['bytes'] and
             sha256(run_dir / filename) == item['sha256'],
             'Run inventory hash/size differs: ' + name + '/' + filename)
    summary, prov, history = (read(run_dir / x) for x in
                              ('summary.json', 'provenance.json', 'history.json'))
    need(summary['provenance'] == prov and summary['history'] == history,
         'Summary provenance/history differs: ' + name)
    need(summary['run'] == name and summary['fold'] == fold and
         summary['selected_checkpoint'] == 'fixed-final' and
         summary['selected_epoch_1based'] == 20 and summary['optimizer_updates'] == spec['steps']
         and summary['skipped_updates'] == 0 and summary['gold'] == 0 and
         summary['gold_access'] == 'disabled' and summary['worker_cleanup']['clean'] is True,
         'Full completion/zero skips differs: ' + name)
    need([record['epoch'] for record in history] == list(range(20)) and
         history[-1]['optimizer_updates'] == spec['steps'] and
         all(record['fp16_skips'] == 0 for record in history),
         'Epoch history/zero skips differs: ' + name)
    need(prov['seed'] == seed and prov['fold'] == fold and
         prov['training_mode'] == 'full' and prov['gold_access'] == 'disabled' and
         prov['full_model_initial_state_sha256'] == INITIAL_TENSOR_SHA and
         prov['fp16_skipped_updates'] == 0 and tuple(prov['heads']) == HEADS and
         prov['head_pos_weight'] == spec['head_pos_weight'] and
         prov['head_label_files'] == {h['name']: h['sha256'] for h in spec['head_labels']},
         'H1 initialization/head provenance differs: ' + name)
    need(prov['train_folds'] == spec['train_folds'] and
         prov['eval_folds'] == spec['eval_folds'] and
         prov['train_ids_ordered_sha256'] == spec['train_ids_sha256'] and
         prov['eval_ids_ordered_sha256'] == spec['eval_ids_sha256'] and
         prov['protocol_sha256'] == sha256(protocol_file) and
         prov['initial_state_sha256'] == protocol['initial_state_sha256'],
         'EXP099/protocol provenance differs: ' + name)
    args = summary['args']
    need(args['run'] == name and args['tag'] == name and args['seed'] == seed and
         args['fold'] == fold and args['mode'] == 'full' and args['variant'] == 'AUG1'
         and all(args.get(k) == v for k, v in protocol['common_args'].items()),
         'Recorded recipe differs: ' + name)
    if seed != 42 or fold >= 2:
        need(prov['protocol_path'] == MANIFEST and
             prov['initial_state_file_sha256'] == protocol['exp153']['initial_state_file_sha256']
             and prov['exp153_surface'] in ('local', 'vast') and
             args['surface'] == prov['exp153_surface'],
             'EXP153 initial file/surface provenance differs: ' + name)
        if prov['exp153_surface'] == 'vast':
            env = prov['environment']
            need('3090' in env.get('gpu', '') and env.get('torch') == '2.5.1+cu121'
                 and env.get('timm') == '1.0.19' and env.get('numpy') == '2.0.2',
                 'Vast platform provenance differs: ' + name)
    ids = spec['ordered_ids']['val']
    for filename in ('oof_best.csv', 'oof_k36.csv'):
        with (run_dir / filename).open(newline='', encoding='utf-8') as stream:
            rows = csv.DictReader(stream)
            need(rows.fieldnames == [e149.UID, *e149.T], 'OOF column order differs: ' + name)
            seen = [row[e149.UID] for row in rows]
        need(seen == ids, 'OOF row order differs: ' + name)
    return {'tag': name, 'checkpoint_sha256': inv['model_best.pt']['sha256'],
            'surface': prov.get('exp153_surface', 'local-EXP149'), 'zero_skips': True}


def verify(root=ROOT):
    root = Path(root)
    registration(root)
    checks = [verify_run(root / 'artifacts/experiments' /
              (source_tag(fold) if seed == 42 and fold in (0, 1) else tag(seed, fold)),
              seed, fold, root) for seed, fold in ALL]
    need(len({x['checkpoint_sha256'] for x in checks}) == 10,
         'Submission ensemble checkpoints are not distinct')
    return {'status': 'VERIFIED_SUBMISSION_INPUTS', 'runs': 10, 'checks': checks}


def collect(source, root=ROOT):
    root, source = Path(root), Path(source)
    registration(root)
    need(source.is_dir() and not source.is_symlink(), 'Missing collected source directory')
    need({x.name for x in source.iterdir()} == {tag(s, f) for s, f in NEW},
         'Collected source must contain exactly eight new run directories')
    for seed, fold in NEW:
        verify_run(source / tag(seed, fold), seed, fold, root)
        need(not (root / 'artifacts/experiments' / tag(seed, fold)).exists(),
             'Refusing to overwrite a local run')
    for seed, fold in NEW:
        shutil.copytree(source / tag(seed, fold), root / 'artifacts/experiments' / tag(seed, fold))
    return verify(root)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for cmd in ('prepare', 'validate', 'verify', 'remote-link'):
        sub.add_parser(cmd)
    vs = sub.add_parser('vast-stage')
    vs.add_argument('--out', required=True, type=Path)
    vp = sub.add_parser('vast-plan')
    vp.add_argument('--stage', required=True, type=Path)
    vp.add_argument('--host', default='<remote-host>')
    vp.add_argument('--port', type=int, default=32358)
    lq = sub.add_parser('local-queue')
    lq.add_argument('--out', required=True, type=Path)
    lq.add_argument('--after', default='none')
    lq.add_argument('--workers', type=int, default=12)
    co = sub.add_parser('collect')
    co.add_argument('--source', required=True, type=Path)
    a = ap.parse_args(argv)
    if a.command == 'prepare': value = prepare()
    elif a.command == 'validate': value = validate()
    elif a.command == 'vast-stage': value = stage(a.out)
    elif a.command == 'vast-plan': value = vast_plan(a.stage, a.host, a.port)
    elif a.command == 'local-queue': value = local_queue(a.out, a.after, a.workers)
    elif a.command == 'collect': value = collect(a.source)
    elif a.command == 'verify': value = verify()
    else: value = remote_link()
    print(value if isinstance(value, str) else json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
