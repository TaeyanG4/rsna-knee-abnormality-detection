"""EXP141: offline, private, two-T4 execution of the frozen EXP139 CTX F0 arms.

stage/verify-stage/import-outputs/screen are CPU-only. Only launch/worker execute
training, and both refuse a non-Kaggle host. No command uploads or submits.
The original EXP139 source and manifests are copied byte-for-byte, never edited.
An explicit runtime adapter changes path/environment validation only; the actual
loader, model, optimizer, schedule, loss, update and evaluation code is EXP139.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOL = 'tools/exp141_kaggle_ctx.py'
RUNNER = 'training/runners/exp139_aug1_variants.py'
GUARD = 'tools/exp139_aug1_variants.py'
REGISTRY = 'artifacts/experiments/EXP139_AUG1_VARIANTS/CTX/protocol-v1/manifest.json'
MANIFESTS = {s: str(PurePosixPath(REGISTRY).parent / f's{s}/manifest.json') for s in (42, 43)}
REGISTERED_SHA = {
    42: '8d72b88f73861fbba383d391c466d08002198cdee1dce855c7ce958d13ac7e78',
    43: '458400da64ad5e714c4ffd071017984fc18e8620c346085e84d16c5f8430cf19',
}
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
KERNEL = 'taeyangg4/exp141-aug1-ctx-f0-two-seeds'  # B 2026-10-08: Kaggle slugs a NEW kernel from its title, not the metadata id
CODE_DATASET = 'taeyangg4/knee-exp141-code'
CORPUS_DATASET = 'taeyangg4/knee-raptor-corpus-336'
WEIGHTS_DATASET = 'marwanmath/resnet-50-radimagenet-marwan'
WEIGHTS_SHA = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
DEFAULT_STAGE = ROOT / '.local/exp141-kaggle-ctx'
SCHEMA = 'exp141-kaggle-ctx-v1'
RECEIPT = 'exp141-bundle.json'
SEEDS = (42, 43)
WORKERS = 2  # four total loader workers on the shared four-vCPU host
MAX_WALL_SECONDS = 200 * 60  # B: user granted ≈3–4 extra Kaggle GPU h (2026-10-08 20:0x); quota observed at 1× wall time for 2×T4 sessions today
ABSOLUTE_WALL_SECONDS = 10 * 3600 + 45 * 60
THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
REQUIRED_OUTPUTS = {'model_best.pt', 'oof_best.csv', 'oof_k36.csv', 'summary.json',
                    'provenance.json', 'history.json', 'exp141-run.json'}
TOOLS = ('exp084_group_loss', 'exp087_training', 'exp099_kfold', 'exp101_kfold_arch',
         'exp107_kfold_radr50', 'exp113_radr50_local', 'exp118_radr50_local_s43',
         'exp125_radr50_k16_local', 'exp127_radr50_single', 'exp129_radr50_single',
         'exp139_aug1_variants', 'exp141_kaggle_ctx')
CAVEAT = ('kaggle_t4: independent remote trajectories; same registered recipe, IDs, labels, '
          'pos_weight and steps, different Python/torch/torchvision/timm/GPU. No numerical or '
          'bit parity with local AUG1 controls. Local-control deltas/bootstrap are descriptive '
          'and confounded by environment; they cannot pass a paired Tier A/B adoption gate.')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def safe_relative(name):
    p = PurePosixPath(name)
    need(bool(name) and not p.is_absolute() and '\\' not in name and ':' not in name
         and all(x not in ('..', '.') for x in p.parts) and str(p) == name, 'Unsafe path: ' + name)
    return p


def tag(seed):
    need(seed in SEEDS, 'Only seeds 42 and 43 are registered')
    return f'EXP139_AUG1_CTX_S{seed}_F0'


def source_contract(root):
    """Read only named registrations; no torch import and no model construction."""
    root = Path(root)
    reg = read(root / REGISTRY)
    need(reg['schema'] == 'exp139-aug1-variants-v1' and reg['variant'] == 'CTX', 'Wrong CTX registry')
    protocols, bindings = {}, {}
    for seed in SEEDS:
        path = root / MANIFESTS[seed]
        need(sha(path) == REGISTERED_SHA[seed] == reg['seeds'][str(seed)]['sha256'], 'Registered manifest hash mismatch')
        need(reg['seeds'][str(seed)]['path'] == MANIFESTS[seed], 'Registry path mismatch')
        p = read(path)
        need(p['seed'] == seed and p['variant'] == 'CTX' and list(p['folds']) == ['0'], 'Wrong seed/variant/fold')
        need(p['folds']['0']['tag'] == tag(seed), 'Wrong output tag')
        for name, digest in p['source_bindings'].items():
            safe_relative(name)
            need(name not in bindings or bindings[name] == digest, 'Seed binding disagreement: ' + name)
            bindings[name] = digest
        protocols[seed] = p
    a, b = protocols.values()
    for key in ('common_args', 'paths', 'label_set', 'variant_config', 'augmentation_config'):
        need(a[key] == b[key], 'Seed contract disagreement: ' + key)
    for key in ('ordered_ids', 'groups', 'pos_weight', 'n_train', 'n_val', 'steps'):
        need(a['folds']['0'][key] == b['folds']['0'][key], 'Seed fold disagreement: ' + key)
    return protocols, bindings


def stage_files(root, protocols, bindings):
    root = Path(root)
    names = {TOOL, REGISTRY, *MANIFESTS.values(), 'tests/test_exp141_kaggle_ctx.py', 'tools/__init__.py'}
    names.update(f'tools/{n}.py' for n in TOOLS)
    # Exact source trees, no pycache, credentials, runtime state or data discovery.
    for folder in ('training/src', 'training/runners'):
        names.update(p.relative_to(root).as_posix() for p in (root / folder).glob('*.py'))
    names.update(n for n in bindings if n.endswith(('.py', '.md')))
    expected = dict(bindings)
    for p in protocols.values():
        f = p['folds']['0']
        names.update((p['label_set']['path'], p['paths']['base_ordered_ids'], p['paths']['base_groups'],
                      p['paths']['folds'], p['fold_protocol']['path'], p['initial_state']['keys_file'],
                      p['derived_from']['path'], f['ordered_ids']['path'], f['groups']['path']))
        expected.update({p['label_set']['path']: p['label_set']['sha256'],
                         p['fold_protocol']['path']: p['fold_protocol']['sha256'],
                         p['paths']['folds']: p['fold_protocol']['folds_csv_sha256'],
                         p['derived_from']['path']: p['derived_from']['sha256'],
                         **{f[k]['path']: f[k]['sha256'] for k in ('ordered_ids', 'groups')},
                         **{'training/src/' + k: v for k, v in p['training_code_sha256'].items()}})
    # The sole label file contains just UID + 12 targets. Read HEADER only before copying.
    label = protocols[42]['label_set']['path']
    with (root / label).open(encoding='utf-8', newline='') as f:
        need(next(csv.reader(f)) == [UID, *TARGETS], 'Refusing label file with extra/report columns')
    files = {}
    for name in sorted(names):
        safe_relative(name)
        need(not (root / name).is_symlink(), 'No source symlinks')
        digest = sha(root / name)
        need(name not in expected or expected[name] == digest, 'Frozen source/input hash mismatch: ' + name)
        files[name] = dict(sha256=digest, bytes=(root / name).stat().st_size)
    return files


def notebook_source(archive_sha, receipt_sha):
    return f'''# EXP141: private training only. No pip installs, credentials, network or submissions.
import hashlib, json, os, pathlib, subprocess, sys, time, zipfile
STARTED = time.time()
INPUT = pathlib.Path('/kaggle/input')
WORK = pathlib.Path('/kaggle/working')
hits = list(INPUT.rglob('exp141-code.zip.bin'))  # B 2026-10-08: Kaggle auto-extracts uploaded .zip files; the archive is uploaded as .zip.bin
assert len(hits) == 1, 'Expected exactly one EXP141 code archive'
assert hashlib.sha256(hits[0].read_bytes()).hexdigest() == {archive_sha!r}, 'Code archive SHA mismatch'
ws = pathlib.Path('/tmp/exp141-workspace')
assert not ws.exists(), 'Refusing existing workspace'
with zipfile.ZipFile(hits[0]) as z:
    names = z.namelist()
    assert len(set(names)) == len(names), 'Duplicate zip entries'
    for name in names:
        p = pathlib.PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
    z.extractall(ws)
env = {{**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'KNEE_WORKSPACE_ROOT': str(ws),
       'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}}  # B 2026-10-09: strict runtime needs it before the worker's CUDA init (v2 failed)
for k in {tuple(THREADS)!r}:
    env[k] = '1'
cmd = [sys.executable, str(ws / {TOOL!r}), 'launch', '--receipt-sha', {receipt_sha!r},
       '--started', str(STARTED)]
try:
    # Inner supervisor stops children at MAX_WALL_SECONDS (200 min), gives 30 s grace then kills their process groups.
    result = subprocess.run(cmd, env=env, cwd=ws, timeout=max(1, {MAX_WALL_SECONDS + 60} - (time.time() - STARTED)))
except subprocess.TimeoutExpired:
    raise RuntimeError('EXP141 supervisor exceeded its quota deadline; outputs are incomplete')
assert result.returncode == 0, 'EXP141 failed or stopped at its quota guard; inspect exp141-status.json and seed logs'
status = json.loads((WORK / 'exp141-status.json').read_text())
assert status['state'] == 'COMPLETE', status
print(json.dumps(status, indent=2))
'''


def stage(root=ROOT, out=DEFAULT_STAGE):
    root, out = Path(root), Path(out)
    need(not out.exists(), 'Refusing existing stage; verify it, or use a NEW --out directory')
    protocols, bindings = source_contract(root)
    files = stage_files(root, protocols, bindings)
    p = protocols[42]
    cm = root / p['paths']['corpus'] / 'manifest.json'
    need(sha(cm) == p['corpus_manifest_sha256'], 'Corpus manifest mismatch')
    corpus = read(cm)
    need(corpus['finished'] and corpus['contract'] == 'raptor64_336_c140_s02', 'Incomplete/wrong corpus')
    receipt = dict(schema=SCHEMA, provenance_flag='kaggle_t4', files=files,
        protocols={str(s): dict(path=MANIFESTS[s], sha256=REGISTERED_SHA[s]) for s in SEEDS},
        registry_sha256=sha(root / REGISTRY), docker_image=DOCKER,
        execution=dict(seeds=list(SEEDS), fold=0, workers_per_seed=WORKERS,
                       cuda_visible_devices={'42': '0', '43': '1'}, max_wall_seconds=MAX_WALL_SECONDS,
                       absolute_wall_seconds=ABSOLUTE_WALL_SECONDS, quota_multiplier_conservative=2,
                       quota_at_dispatch='21:37 / 30 h', quota_cap_hours=25),
        external_assets=dict(corpus_dataset=CORPUS_DATASET, corpus_manifest_sha256=sha(cm),
            corpus_manifest=corpus, weights_dataset=WEIGHTS_DATASET, weights_sha256=WEIGHTS_SHA),
        omitted_screen_only_bindings={n: v for n, v in bindings.items() if n not in files},
        note='Omitted bindings are frozen local controls/other-label screen assets. They are not used by training; '
             'the unchanged EXP139 screen rechecks them locally. All execution inputs and code are bundled or hash-checked mounts.',
        caveat=CAVEAT)
    out.mkdir(parents=True)
    code = out / 'code-dataset'
    code.mkdir()
    marker = b'# EXP141 isolated workspace marker; no credentials or application settings.\n'
    receipt['files']['config/workspace.toml'] = dict(sha256=hashlib.sha256(marker).hexdigest(), bytes=len(marker))
    receipt_bytes = (json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    archive = code / 'exp141-code.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        payloads = {**{n: (root / n).read_bytes() for n in files if n != 'config/workspace.toml'},
                    'config/workspace.toml': marker, RECEIPT: receipt_bytes}
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    write(code / 'dataset-metadata.json', dict(id=CODE_DATASET, title='Knee EXP141 code - private research', isPrivate=True,
        licenses=[{'name': 'other'}], description='Private exact EXP139 source/registrations and target-only Dread labels. '
        'Original third-party licences and provenance apply; no grant of new rights. See EXP141 README.'))
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                            language_info=dict(name='python')),
              cells=[dict(cell_type='code', id='exp141', metadata={}, execution_count=None, outputs=[],
                          source=notebook_source(sha(archive), receipt_sha).splitlines(keepends=True))])
    for cell in nb['cells']:
        compile(''.join(cell['source']), '<exp141-notebook>', 'exec')
    write(out / 'release.ipynb', nb)
    write(out / 'kernel-metadata.json', dict(id=KERNEL, title='EXP141 AUG1 CTX F0 two seeds', code_file='release.ipynb',
        language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
        machine_shape='NvidiaTeslaT4', docker_image=DOCKER, dataset_sources=[CORPUS_DATASET, WEIGHTS_DATASET, CODE_DATASET],
        competition_sources=['rsna-knee-abnormality-detection'], kernel_sources=[], model_sources=[]))
    (out / RECEIPT).write_bytes(receipt_bytes)
    write(out / 'stage-receipt.json', dict(schema=SCHEMA, archive_sha256=sha(archive), receipt_sha256=receipt_sha,
          source_files=len(files), source_bytes=sum(x['bytes'] for x in files.values()),
          manifest_sha256={str(s): REGISTERED_SHA[s] for s in SEEDS}, no_execution=True))
    (out / 'README.md').write_text(readme(out), encoding='utf-8')
    sum_paths = ['release.ipynb', 'kernel-metadata.json', RECEIPT, 'stage-receipt.json', 'README.md',
                 'code-dataset/dataset-metadata.json', 'code-dataset/exp141-code.zip']
    (out / 'SHA256SUMS.txt').write_text(''.join(f'{sha(out / n)}  {n}\n' for n in sum_paths), encoding='utf-8')
    verify_stage(out, root)
    return read(out / 'stage-receipt.json')


def verify_stage(out=DEFAULT_STAGE, root=ROOT):
    out, root = Path(out), Path(root)
    sr = read(out / 'stage-receipt.json')
    need(sha(out / RECEIPT) == sr['receipt_sha256'], 'Stage receipt mismatch')
    archive = out / 'code-dataset/exp141-code.zip'
    need(sha(archive) == sr['archive_sha256'], 'Stage archive mismatch')
    receipt = read(out / RECEIPT)
    metadata = read(out / 'code-dataset/dataset-metadata.json')
    need(metadata.get('isPrivate') is True and metadata.get('id') == CODE_DATASET, 'Code dataset privacy/ID mismatch')
    with zipfile.ZipFile(archive) as z:
        need(set(z.namelist()) == {*receipt['files'], RECEIPT}, 'Archive file set mismatch')
        need(hashlib.sha256(z.read(RECEIPT)).hexdigest() == sr['receipt_sha256'], 'Embedded receipt mismatch')
        for name, rec in receipt['files'].items():
            safe_relative(name)
            data = z.read(name)
            need(len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256'], 'Archive file mismatch: ' + name)
            if name != 'config/workspace.toml':
                need(sha(root / name) == rec['sha256'], 'Local source changed since staging: ' + name)
    meta = read(out / 'kernel-metadata.json')
    need(meta['is_private'] is True and meta['enable_internet'] is False and meta['enable_gpu'] is True,
         'Private/offline/GPU metadata mismatch')
    need(meta['docker_image'] == DOCKER and meta.get('machine_shape', 'NvidiaTeslaT4') == 'NvidiaTeslaT4', 'Image/GPU mismatch')
    need(meta['dataset_sources'] == [CORPUS_DATASET, WEIGHTS_DATASET, CODE_DATASET], 'Dataset sources mismatch')
    nb = read(out / 'release.ipynb')
    need(len(nb['cells']) == 1 and ''.join(nb['cells'][0]['source']) == notebook_source(sr['archive_sha256'], sr['receipt_sha256']), 'Notebook drift')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<exp141-notebook>', 'exec')
    return sr


def verify_bundle(root, expected_sha):
    root = Path(root)
    need(sha(root / RECEIPT) == expected_sha, 'Bundle receipt hash mismatch')
    receipt = read(root / RECEIPT)
    need(receipt['schema'] == SCHEMA and receipt['provenance_flag'] == 'kaggle_t4', 'Wrong bundle')
    for name, rec in receipt['files'].items():
        safe_relative(name)
        path = root / name
        need(not path.is_symlink() and path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256'], 'Bundle file hash mismatch: ' + name)
    source_contract(root)
    return receipt


def verify_corpus(directory, receipt, full=True):
    """Full shard hashes once in supervisor; worker repeats all small input hashes."""
    directory = Path(directory)
    assets = receipt['external_assets']
    need(sha(directory / 'manifest.json') == assets['corpus_manifest_sha256'], 'Mounted corpus manifest mismatch')
    manifest = read(directory / 'manifest.json')
    need(manifest == assets['corpus_manifest'], 'Corpus manifest content mismatch')
    checks = {'ids.npy': manifest['ids_sha256'], 'masks.npy': manifest['masks_sha256']}
    for shard in manifest['shards']:
        name = shard['file']
        need(Path(name).name == name and '/' not in name and '\\' not in name, 'Unsafe shard name')
        need(shard['done'] and (directory / name).stat().st_size == shard['bytes'], 'Incomplete corpus shard')
        if full:
            checks[name] = shard['sha256']
    for name, digest in checks.items():
        need(sha(directory / name) == digest, 'Corpus hash mismatch: ' + name)
    return dict(corpus_manifest_sha256=assets['corpus_manifest_sha256'], full_shards_verified=full, hashes=checks)


def one_mount(input_root, filename, digest=None):
    hits = list(Path(input_root).rglob(filename))
    if digest:
        hits = [p for p in hits if sha(p) == digest]
    need(len(hits) == 1, 'Expected one matching mount: ' + filename)
    return hits[0]


def child_environment(seed, root, environ=None):
    need(seed in SEEDS, 'Unregistered seed')
    env = dict(os.environ if environ is None else environ)
    for key in ('KNEE_RADIMAGENET_R50', 'PYTHONPATH'):
        env.pop(key, None)
    env.update(THREADS)
    env.update(CUDA_VISIBLE_DEVICES=str(SEEDS.index(seed)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        KNEE_REMOTE_RELAX_ENV='1', KNEE_WORKSPACE_ROOT=str(root))
    return env


def runner_args(p, root, output, weights):
    root = Path(root)
    argv = ['--variant', 'CTX', '--protocol', str(root / MANIFESTS[p['seed']]), '--fold', '0',
            '--mode', 'full', '--tag', tag(p['seed']), '--seed', str(p['seed']), '--workers', str(WORKERS),
            '--output-root', str(output), '--init-weights', str(weights)]
    for key, val in p['common_args'].items():
        if isinstance(val, bool):
            if val:
                argv.append('--' + key.replace('_', '-'))
        else:
            argv += ['--' + key.replace('_', '-'), str(val)]
    for key in ('corpus', 'labels', 'eval_labels'):
        argv += ['--' + key.replace('_', '-'), str(root / p['paths'][key])]
    return argv


def remote_arguments(a, root, receipt, output, weights):
    """Explicit replacement of ONLY the local path/environment guard."""
    root = Path(root)
    protocols, _ = source_contract(root)
    need(a.seed in protocols, 'Wrong seed')
    p = protocols[a.seed]
    need(a.variant == 'CTX' and a.fold == 0 and a.mode == 'full' and not a.validate_only and a.tag == tag(a.seed), 'Wrong training scope')
    need(Path(a.protocol).resolve() == (root / MANIFESTS[a.seed]).resolve(), 'Wrong protocol path')
    for key, val in p['common_args'].items():
        need(getattr(a, key) == val, 'Frozen argument mismatch: ' + key)
    need(a.workers == receipt['execution']['workers_per_seed'] == WORKERS, 'Worker budget mismatch')
    for key in ('corpus', 'labels', 'eval_labels'):
        need(Path(getattr(a, key)).resolve() == (root / p['paths'][key]).resolve(), 'Wrong input path: ' + key)
    need(Path(a.init_weights).resolve() == Path(weights).resolve() and sha(weights) == WEIGHTS_SHA, 'Weights mismatch')
    need(Path(a.output_root).resolve() == Path(output).resolve(), 'Wrong output root')
    for key, val in {**THREADS, 'HF_HUB_OFFLINE': '1', 'KNEE_REMOTE_RELAX_ENV': '1'}.items():
        need(os.environ.get(key) == val, 'Wrong environment flag: ' + key)
    f = p['folds']['0']
    for key in ('ordered_ids', 'groups'):
        need(sha(root / f[key]['path']) == f[key]['sha256'], 'Fold file mismatch')
    ordered = read(root / f['ordered_ids']['path'])
    for key in ('train', 'val'):
        need(hashlib.sha256('\n'.join(ordered[key]).encode()).hexdigest() == f['ordered_ids'][key + '_sha256'], 'Ordered IDs mismatch')
    need(sha(root / p['label_set']['path']) == p['label_set']['sha256'], 'Label byte hash mismatch')
    # Original runner next rechecks semantic labels, membership, pos_weight and steps.
    return p, f, ordered, Path(output)


def require_kaggle():
    need(sys.platform.startswith('linux') and Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir(),
         'Training execution is Kaggle-only; local GPU must never be initialized')


def verify_runtime(environment, count):
    need(count == 1 and 'T4' in environment.get('gpu', ''), 'Worker must see exactly one T4')
    need(environment.get('python', '').startswith('3.13.') and environment.get('torch') == '2.11.0+cu128'
         and environment.get('timm') == '1.0.29', 'Pinned Python/torch/timm differs; no silent default-image fallback')


def worker(seed, receipt_sha, weights, started, output=Path('/kaggle/working'), root=ROOT):
    require_kaggle()
    root, output = Path(root), Path(output)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == str(SEEDS.index(seed)), 'Wrong physical GPU assignment')
    receipt = verify_bundle(root, receipt_sha)
    protocols, _ = source_contract(root)
    p = protocols[seed]
    verify_corpus(root / p['paths']['corpus'], receipt, full=False)
    need(time.time() - started < MAX_WALL_SECONDS, 'Notebook quota deadline already passed')
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'training/src'))
    from tools import exp139_aug1_variants as guard
    import raptor_common
    import raptor_exp139
    import torch
    import torchvision
    environment = raptor_common.environment()
    environment['torchvision'] = torchvision.__version__
    verify_runtime(environment, torch.cuda.device_count())
    # Do not let remote relaxation excuse random head/adapter initialization drift.
    original_hashes = raptor_exp139.base_state_hashes
    def strict_initial_hashes(model):
        digest, keys = original_hashes(model)
        need(digest == p['initial_state_sha256'], 'Registered base initial-state hash mismatch on remote stack')
        return digest, keys
    raptor_exp139.base_state_hashes = strict_initial_hashes
    guard.validate_arguments = lambda args, workspace: remote_arguments(args, workspace, receipt, output, weights)
    spec = importlib.util.spec_from_file_location('exp141_frozen_runner', root / RUNNER)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    run = output / tag(seed)
    need(not run.exists(), 'Refusing existing seed output')
    def stop(signum, frame):
        raise SystemExit(124)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(1, MAX_WALL_SECONDS - (time.time() - started)))
    began = time.time()
    try:
        entry.main(runner_args(p, root, output, weights))
        # Receipt is part of output inventory. Keep original checkpoint bytes untouched.
        prov = read(run / 'provenance.json')
        summary = read(run / 'summary.json')
        extra = dict(provenance_flag='kaggle_t4', exp141_bundle_sha256=receipt_sha,
                     exp141_adapter_sha256=sha(root / TOOL), exp141_docker_image=DOCKER,
                     exp141_cuda_visible_devices=str(SEEDS.index(seed)), bit_parity_claimed=False,
                     full_environment=environment, comparison_caveat=CAVEAT)
        prov.update(extra)
        summary.update(provenance_flag='kaggle_t4', comparison_caveat=CAVEAT)
        summary['provenance'] = prov
        write(run / 'provenance.json', prov)
        write(run / 'summary.json', summary)
        write(run / 'exp141-run.json', dict(schema=SCHEMA, state='COMPLETE', seed=seed, **extra,
              elapsed_seconds=time.time() - began, notebook_elapsed_seconds=time.time() - started,
              original_runner_sha256=sha(root / RUNNER), protocol_sha256=REGISTERED_SHA[seed]))
        inventory = read(run / 'output-inventory.json')
        for name in REQUIRED_OUTPUTS:
            inventory[name] = dict(sha256=sha(run / name), bytes=(run / name).stat().st_size)
        write(run / 'output-inventory.json', inventory)
        verify_remote_run(run, p, REGISTERED_SHA[seed], receipt_sha, sha(root / TOOL), root)
    except BaseException as exc:
        if run.exists():
            write(run / 'exp141-failure.json', dict(schema=SCHEMA, state='INCOMPLETE', seed=seed,
                reason=type(exc).__name__ + ': ' + str(exc), notebook_elapsed_seconds=time.time() - started,
                note='Not resumable; no full-run screen unless both seed receipts and inventories validate.'))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def stop_group(proc, hard=False):
    # A departed leader can leave DataLoader descendants alive in its group.
    try:
        os.killpg(proc.pid, signal.SIGKILL if hard else signal.SIGTERM)
    except ProcessLookupError:
        pass


def supervise(commands, deadline, popen=subprocess.Popen, clock=time.time, sleep=time.sleep,
              stopper=stop_group, heartbeat=None):
    """Start BOTH processes before polling. On any failure/deadline, stop the whole group."""
    procs, reason = [], None
    try:
        for command, kwargs in commands:
            need(clock() < deadline, 'Deadline before launch')
            procs.append(popen(command, **kwargs))
        while True:
            codes = [p.poll() for p in procs]
            if any(c not in (None, 0) for c in codes):
                reason = 'child_failure'
                break
            if all(c == 0 for c in codes):
                break
            if clock() >= deadline:
                reason = 'wall_clock_guard'
                break
            if heartbeat:
                heartbeat(codes)
            sleep(min(10, max(0, deadline - clock())))
    except BaseException:
        reason = 'supervisor_exception'
        raise
    finally:
        if reason:
            for p in procs:
                stopper(p, False)
            until = clock() + 30
            while any(p.poll() is None for p in procs) and clock() < until:
                sleep(1)
            for p in procs:
                stopper(p, True)
        for p in procs:
            p.wait(timeout=10)
    return reason, [p.returncode for p in procs]


def launch(receipt_sha, started, root=ROOT, output=Path('/kaggle/working')):
    require_kaggle()
    root, output = Path(root), Path(output)
    status_path = output / 'exp141-status.json'
    need(not status_path.exists(), 'Refusing existing EXP141 supervisor output')
    deadline = started + min(MAX_WALL_SECONDS, ABSOLUTE_WALL_SECONDS)
    status = dict(schema=SCHEMA, state='PREFLIGHT', started=started, provenance_flag='kaggle_t4',
                  deadline=deadline, exp141_bundle_sha256=receipt_sha, caveat=CAVEAT)
    write(status_path, status)
    def timeout(signum, frame):
        raise TimeoutError('EXP141 notebook quota deadline')
    signal.signal(signal.SIGTERM, timeout)
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1, deadline - time.time()))
    try:
        receipt = verify_bundle(root, receipt_sha)
        # nvidia-smi inspects hardware without initializing a CUDA context in supervisor.
        devices = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
                                 capture_output=True, text=True, check=True, timeout=20).stdout.strip().splitlines()
        need(len(devices) == 2 and all('T4' in x for x in devices), 'Exactly two T4 GPUs required')
        status['hardware'] = devices
        p = read(root / MANIFESTS[42])
        corpus = one_mount('/kaggle/input', 'vols_shard00.npy').parent
        weights = one_mount('/kaggle/input', 'ResNet50.pt', WEIGHTS_SHA)
        data_receipt = verify_corpus(corpus, receipt, full=True)
        write(output / 'exp141-data-verification.json', data_receipt)
        destination = root / p['paths']['corpus']
        destination.parent.mkdir(parents=True, exist_ok=True)
        need(not destination.exists(), 'Refusing existing corpus path')
        destination.symlink_to(corpus, target_is_directory=True)
        shutil.copyfile(root / RECEIPT, output / RECEIPT)
        commands = []
        with contextlib.ExitStack() as stack:
            for seed in SEEDS:
                need(not (output / tag(seed)).exists(), 'Refusing existing run')
                log = stack.enter_context((output / f'exp141-seed{seed}.log').open('x', encoding='utf-8'))
                cmd = [sys.executable, str(root / TOOL), 'worker', '--seed', str(seed),
                       '--receipt-sha', receipt_sha, '--weights', str(weights), '--started', str(started)]
                commands.append((cmd, dict(cwd=root, env=child_environment(seed, root),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)))
            def heartbeat(codes):
                status.update(state='RUNNING', returncodes=codes, elapsed_seconds=time.time() - started)
                write(status_path, status)
            reason, codes = supervise(commands, deadline, heartbeat=heartbeat)
        need(reason is None and codes == [0, 0], 'Two-seed execution incomplete: ' + str(reason))
        for seed in SEEDS:
            verify_remote_run(output / tag(seed), read(root / MANIFESTS[seed]), REGISTERED_SHA[seed],
                              receipt_sha, sha(root / TOOL), root)
        status.update(state='COMPLETE', returncodes=codes)
    except BaseException as exc:
        status.update(state='FAILED_OR_INCOMPLETE', reason=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        status.update(elapsed_seconds=time.time() - started)
        status['physical_gpu_hours_upper_bound'] = 2 * status['elapsed_seconds'] / 3600
        write(status_path, status)


def verify_predictions(path, ids):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [UID, *TARGETS], 'Prediction columns mismatch')
        seen = []
        for row in reader:
            seen.append(row[UID])
            vals = [float(row[t]) for t in TARGETS]
            need(all(math.isfinite(v) and 0 <= v <= 1 for v in vals), 'Invalid predictions')
    need(seen == ids and len(set(seen)) == len(ids), 'Prediction UID order/coverage mismatch')


def verify_remote_run(run, p, protocol_sha, receipt_sha, adapter_sha, root=ROOT):
    """All EXP139 full-run gates retained except its local-environment assertion."""
    run, root = Path(run), Path(root)
    inv = read(run / 'output-inventory.json')
    need(REQUIRED_OUTPUTS.issubset(inv), 'Incomplete run inventory')
    for name, rec in inv.items():
        need(Path(name).name == name and '/' not in name and '\\' not in name, 'Unsafe inventory path')
        need(sha(run / name) == rec['sha256'], 'Output inventory mismatch: ' + name)
        need((run / name).stat().st_size == rec['bytes'], 'Output size mismatch: ' + name)
    s, prov, rr = (read(run / n) for n in ('summary.json', 'provenance.json', 'exp141-run.json'))
    f = p['folds']['0']
    need(rr.get('state') == 'COMPLETE' and rr.get('schema') == SCHEMA, 'No complete remote receipt')
    expected = dict(fold=0, seed=p['seed'], training_mode='full', checkpoint_selection='fixed-final',
        protocol_sha256=protocol_sha, label_set='dread', label_file_sha256=p['label_set']['sha256'],
        label_semantic_sha256=p['label_set']['semantic_sha256'], arch=p['common_args']['arch'],
        train_ids_ordered_sha256=f['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=f['ordered_ids']['val_sha256'],
        initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True,
        environment_matches_baseline=False, remote_relaxed=True, input_policy='center3', norm='radimagenet',
        runner_sha256=p['source_bindings'][RUNNER], guard_sha256=p['source_bindings'][GUARD],
        k_eval=24, k_eval_extra=36, exp139_variant='CTX', exp139_variant_config=p['variant_config'],
        exp139_adapter_initial_state=p['adapter_initial_state'], exp139_augmentation_config=p['augmentation_config'],
        provenance_flag='kaggle_t4', exp141_bundle_sha256=receipt_sha, exp141_adapter_sha256=adapter_sha,
        bit_parity_claimed=False, exp141_docker_image=DOCKER,
        exp141_cuda_visible_devices=str(SEEDS.index(p['seed'])),
        exp139_code_sha256={n: p['source_bindings'][n] for n in (
            GUARD, RUNNER, 'training/src/raptor_exp139.py', 'training/src/raptor_exp127_aug1.py',
            'training/src/raptor_exp129.py', 'training/src/raptor_exp129_late.py')})
    for key, val in expected.items():
        need(prov.get(key) == val, 'Remote provenance mismatch: ' + key)
    for key in ('provenance_flag', 'exp141_bundle_sha256', 'exp141_adapter_sha256', 'bit_parity_claimed',
                'exp141_docker_image', 'exp141_cuda_visible_devices', 'protocol_sha256'):
        need(rr.get(key) == prov.get(key), 'Remote receipt disagreement: ' + key)
    need(s.get('provenance') == prov and s.get('provenance_flag') == 'kaggle_t4', 'Summary provenance mismatch')
    need(prov.get('actual_pos_weight') == f['pos_weight'], 'pos_weight mismatch')
    need(prov.get('init_weights', {}).get('weights_sha256') == WEIGHTS_SHA, 'Initial weights mismatch')
    verify_runtime(prov.get('full_environment', {}), 1)
    for key in p['environment']:
        need(prov['environment'].get(key) == prov['full_environment'].get(key), 'Environment receipt mismatch: ' + key)
    args = s.get('args', {})
    need(all(args.get(k) == v for k, v in p['common_args'].items()), 'Recorded recipe mismatch')
    need((args.get('seed'), args.get('variant'), args.get('tag'), args.get('workers')) ==
         (p['seed'], 'CTX', f['tag'], WORKERS), 'Recorded seed/variant/tag/workers mismatch')
    need(run.name == f['tag'] and s.get('mode') == 'full' and s.get('selected_epoch_1based') == p['common_args']['epochs'], 'Not a full fixed-final run')
    need((s.get('train'), s.get('val'), s.get('optimizer_updates')) == (f['n_train'], f['n_val'], f['steps']), 'Counts/updates mismatch')
    need(0 <= s.get('skipped_updates', 10**9) <= f['max_fp16_skips'], 'FP16 skip gate failed')
    history = read(run / 'history.json')
    need([h.get('epoch') for h in history] == list(range(p['common_args']['epochs'])) and history == s.get('history'), 'Epoch history mismatch')
    ids = read(root / f['ordered_ids']['path'])['val']
    for name in ('oof_best.csv', 'oof_k36.csv'):
        verify_predictions(run / name, ids)
    return {str((run / n).resolve()): sha(run / n) for n in {*REQUIRED_OUTPUTS, 'output-inventory.json'}}


def import_outputs(download, stage_dir=DEFAULT_STAGE, root=ROOT):
    download, stage_dir, root = Path(download), Path(stage_dir), Path(root)
    sr = verify_stage(stage_dir, root)
    status = read(download / 'exp141-status.json')
    need(status['state'] == 'COMPLETE' and status['exp141_bundle_sha256'] == sr['receipt_sha256'], 'Incomplete/wrong remote job')
    need(sha(download / RECEIPT) == sr['receipt_sha256'], 'Downloaded bundle receipt mismatch')
    protocols, _ = source_contract(root)
    destinations = [root / 'artifacts/experiments' / tag(seed) for seed in SEEDS]
    need(all(not p.exists() for p in destinations), 'Refusing to overwrite local/existing CTX runs')
    # Verify BOTH seeds before copying either. Never interpret process exit alone as success.
    for seed in SEEDS:
        verify_remote_run(download / tag(seed), protocols[seed], REGISTERED_SHA[seed],
                          sr['receipt_sha256'], sha(root / TOOL), root)
    for seed, dest in zip(SEEDS, destinations):
        source = download / tag(seed)
        inv = read(source / 'output-inventory.json')
        dest.mkdir(parents=True)
        for name in [*inv, 'output-inventory.json']:
            shutil.copyfile(source / name, dest / name)
        if (source / 'train.log').is_file():
            shutil.copyfile(source / 'train.log', dest / 'train.log')
        verify_remote_run(dest, protocols[seed], REGISTERED_SHA[seed], sr['receipt_sha256'], sha(root / TOOL), root)
    return [str(x) for x in destinations]


def screen(stage_dir=DEFAULT_STAGE, root=ROOT):
    """Call the frozen EXP139 screen with explicit remote verification, new output dir.

    No false local provenance is fabricated and no frozen source hash is changed.
    The direct EXP139 CLI intentionally still rejects these external trajectories.
    """
    sr = verify_stage(stage_dir, root)
    sys.path.insert(0, str(root))
    from tools import exp139_aug1_variants as guard
    guard.forbid_cuda_init()
    original = (guard.verify_run, guard.screen_dir, guard.metric_contrast, guard.screen_markdown)
    def remote_verify(run, p, protocol_sha, root=root, candidate=True):
        if not candidate:
            return original[0](run, p, protocol_sha, root, candidate=False)
        bindings = verify_remote_run(run, p, protocol_sha, sr['receipt_sha256'], sha(Path(root) / TOOL), root)
        # EXP139 check_bindings accepts only workspace-relative names.
        return {Path(n).relative_to(Path(root).resolve()).as_posix(): v for n, v in bindings.items()}
    def descriptive(*args, **kwargs):
        metrics = original[2](*args, **kwargs)
        metrics['descriptive_local_control_thresholds'] = metrics['tiers']
        metrics['tiers'] = {k: {**v, 'screen_verdict': 'DESCRIPTIVE_ONLY_REMOTE', 'adoption_authorized': False}
                            for k, v in metrics['tiers'].items()}
        metrics['descriptive_per_seed_thresholds'] = metrics['per_seed_tiers']
        metrics['per_seed_tiers'] = {seed: {k: {**v, 'screen_verdict': 'DESCRIPTIVE_ONLY_REMOTE',
            'adoption_authorized': False} for k, v in tiers.items()} for seed, tiers in metrics['per_seed_tiers'].items()}
        metrics['comparison_caveat'] = CAVEAT
        return metrics
    guard.verify_run = remote_verify
    guard.screen_dir = lambda variant: 'artifacts/experiments/EXP139_AUG1_VARIANTS/CTX/screen-kaggle-t4'
    guard.metric_contrast = descriptive
    guard.screen_markdown = lambda body: '# EXP141 remote trajectory screen\n\n' + CAVEAT + '\n\n' + original[3](body).replace(
        'matched local F0 screen', 'remote F0 descriptive screen')
    try:
        body = guard.screen('CTX', root)
        body.update(schema=SCHEMA + '-screen', provenance_flag='kaggle_t4', comparison_caveat=CAVEAT,
                    exp141_bundle_sha256=sr['receipt_sha256'])
        write(Path(root) / guard.screen_dir('CTX') / 'result.json', body)
        return body
    finally:
        guard.verify_run, guard.screen_dir, guard.metric_contrast, guard.screen_markdown = original


def readme(out):
    return f'''# EXP141 delivery (prepared only)

Private kernel: {KERNEL}. Code dataset: {CODE_DATASET}.
No account changes or jobs are performed by staging. Competition is attached but
never read; only target-only Dread labels from the frozen registration are used.

## Execution and evidence

Exact EXP139 CTX seed 42/43 manifests and original source bytes are in the code ZIP.
Seed 42 gets physical GPU 0, seed 43 physical GPU 1 via separate subprocesses and
CUDA_VISIBLE_DEVICES. Each sees logical cuda:0. Two loader workers each (four total).
F0: 3455 train / 888 validation, 20 fixed-final epochs, 431 batches/epoch, 8620
scheduled updates, registered clipped pos_weight, bs8, K12 train / K24 and K36 OOF.
No change to AMP, augmentation, optimizer, LR, data order, seed or initialization.
Source, manifest, labels, fold metadata, ids/masks and all 12 corpus shard hashes
are checked; a mismatch aborts. Initial base and CTX adapter state hashes must
match exactly even on the remote stack. Nothing is downloaded or pip-installed.

The image is the latest EXP133/EXP140 pin `{DOCKER}`: Python 3.13,
torch 2.11.0+cu128, timm 1.0.29. It already served RadR50; training is untested.
The older EXP107 pin 37c64f... used Python 3.12. There is no default-image fallback.
Each run records Python, platform, torch, torchvision, timm, numpy, cv2, CUDA/cuDNN,
driver, GPU, exact code hashes, registered protocol hash and `kaggle_t4` provenance.
The original model checkpoint is unchanged; the inventory binds it to the added
EXP141 receipt and full summary/provenance. {CAVEAT}

## Budget and stop behavior

Planning estimate: **120-180 min wall for BOTH concurrent seeds**, including
staging, full-corpus hashing and K24/K36 evaluation. Supplied local AUG135-50min
and recorded AUG146.7/50.7min are not direct T4 timing. Comparable unaugmented
local EXP113 F0 took1550s (25.8min), Kaggle EXP107 F0 took3493s (58.2min), a2.25x
ratio. Applying that ratio to EXP139's local CTX50-70min predicts113-158min before
extra AUG1 affine CPU cost and shared four-vCPU/I/O contention. This is an estimate,
not a measured EXP141 runtime; an optimistic75-100min case must not drive approval.

Two physical T4s imply **4-6 physical GPU hours** for that range. Kaggle's billing
multiplier is not verified here; budget assumes TWO quota hours per wall hour.
At dispatch 21:37 used leaves 3:23 until cap25. The immutable quota guard signals
both process groups at **200 minutes from the first notebook cell**, waits at most
30 seconds then kills survivors; notebook outer watchdog is 97 minutes. Worst
planned charge is about3.23h plus Kaggle startup/shutdown overhead (about9min
quota margin to cap25). Recheck live quota/other running reservations before push.
The likely runtime EXCEEDS this96min quota guard: the package is built as requested,
but completion within the authorized budget is NOT established. B should resolve
the actual T4x2 quota multiplier/live reservations and timing feasibility before
push; never quietly extend the guard. A stopped run is NOT resumable and MUST NOT
be screened. Changing the guard requires a newly hashed package and budget authority.
The absolute safety ceiling is10h45, with the much tighter96min quota guard active.
If a child fails, the sibling is stopped to avoid wasting quota. No fallback
batch size, changed epochs, retry, second kernel, or local GPU job is started.

## Exact commands for B (PowerShell, workspace root)

```powershell
$gpuPython = 'C:\\Users\\Taeyang\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'
$env:OMP_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
# Already staged. This verifies every archived file and current local source,
# and calls compile() on EVERY notebook code cell:
& $gpuPython tools/exp141_kaggle_ctx.py verify-stage
# For a fresh NEW directory only (never overwrite this reviewed stage):
# & $gpuPython tools/exp141_kaggle_ctx.py stage --out .local/exp141-kaggle-ctx/rebuilt-v2
& $gpuPython -m pytest tests/test_exp141_kaggle_ctx.py --basetemp .local/pytest-exp141 -p no:cacheprovider

# Account actions below are B's responsibility; not executed by builder.
# create is PRIVATE by default; DO NOT pass --public. New dataset, no version overwrite.
& .venv/Scripts/python.exe -X utf8 -m kaggle datasets create -p .local/exp141-kaggle-ctx/code-dataset --dir-mode zip
& .venv/Scripts/python.exe -X utf8 -m kaggle datasets status {CODE_DATASET}
# Push wrapper waits for all attached datasets, explicitly selects T4, and strips
# machine_shape (saving its original); this expected metadata edit does not alter code ZIP.
& .venv/Scripts/python.exe -X utf8 tools/kaggle_push.py .local/exp141-kaggle-ctx --accelerator NvidiaTeslaT4
& .venv/Scripts/python.exe -X utf8 -m kaggle kernels status {KERNEL}
& .venv/Scripts/python.exe -X utf8 -m kaggle kernels output {KERNEL} -p .local/exp141-kaggle-ctx/download --page-size 200

# Verifies COMPLETE status, both inventories, OOF roster/columns/probabilities,
# protocol, provenance, recipe, epochs/steps/FP16 skips BEFORE copying either seed.
# Refuses any existing target directories. Exact destinations:
# artifacts/experiments/EXP139_AUG1_CTX_S42_F0
# artifacts/experiments/EXP139_AUG1_CTX_S43_F0
& $gpuPython tools/exp141_kaggle_ctx.py import-outputs --download .local/exp141-kaggle-ctx/download
# Calls tools/exp139_aug1_variants.py screen('CTX') IN PROCESS, with explicit
# remote verifier and descriptive-only verdicts. No EXP139 source is changed.
& $gpuPython tools/exp141_kaggle_ctx.py screen
```

Direct `tools/exp139_aug1_variants.py screen --variant CTX` refuses remote runs by
design; copying cannot satisfy its local environment guard. Use the EXP141 adapter
above. Output is CTX/screen-kaggle-t4, preserving the local paired screen directory.
Same EXP139 metrics/bootstrap/partner calculations, with remote confounding shown
and Tier A/B adoption disabled. No public submission or fold expansion is implied.

## Open risks / licence

The requested local Python executable is inaccessible in the builder sandbox.
Pytest and notebook compile() are UNRUN here; B must run both commands above.
PowerShell/.NET staging verifies byte hashes but does not validate Python syntax.
No CUDA or Kaggle execution was performed by this builder. Python3.13 training,
CTX initial-state equality on torch2.11/timm1.0.29, AMP finite/skip gate, GPU memory,
CPU augmentation contention and deadline feasibility need the first remote run.
Fail closed if any of these fail. Failed/incomplete outputs remain diagnostic.
Full corpus verification reads~31.9GB once; do not skip hashes to save time.
RadImageNet mounted source: {WEIGHTS_DATASET}, SHA256 {WEIGHTS_SHA},
CC-BY-NC-SA-4.0 per the frozen registration. Upstream BMEII-AI/RadImageNet repo MIT;
equality to upstream weights was not established. Dread label source provenance
and all inherited registrations are preserved; no third-party rights are relicensed.
Receipt lists screening-only bindings omitted from remote training (local AUG1
controls, alternate labels and partner OOF); frozen screen checks them locally.
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('stage'); s.add_argument('--out', type=Path, default=DEFAULT_STAGE)
    v = sub.add_parser('verify-stage'); v.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    i = sub.add_parser('import-outputs'); i.add_argument('--download', type=Path, required=True)
    i.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    s = sub.add_parser('screen'); s.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    for name in ('launch', 'worker'):
        p = sub.add_parser(name)
        p.add_argument('--receipt-sha', required=True)
        p.add_argument('--started', type=float, required=True)
        if name == 'worker':
            p.add_argument('--seed', type=int, choices=SEEDS, required=True)
            p.add_argument('--weights', type=Path, required=True)
    a = ap.parse_args(argv)
    if a.command == 'stage': result = stage(out=a.out)
    elif a.command == 'verify-stage': result = verify_stage(a.stage)
    elif a.command == 'import-outputs': result = import_outputs(a.download, a.stage)
    elif a.command == 'screen': result = screen(a.stage)
    elif a.command == 'launch': return launch(a.receipt_sha, a.started)
    else: return worker(a.seed, a.receipt_sha, a.weights, a.started)
    if a.command != 'screen':
        print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
