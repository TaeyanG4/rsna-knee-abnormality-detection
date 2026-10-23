"""EXP150 v1.0.0: registered paired consensus-weighted AUG1 RadR50 screen.

Only `remote-run` trains. All local commands are CPU-only and never use a network.
Remote-run refuses Windows and requires an explicit Kaggle or Vast marker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = 'artifacts/experiments/EXP150_CONSENSUS_WEIGHT'
BASE = 'artifacts/experiments/EXP127_AUG1_PROTOCOL_V1/s42/manifest.json'
RUNNER = 'training/runners/exp150_consensus_weight.py'
TOOL = 'tools/exp150_consensus_weight.py'
UID = 'StudyInstanceUID'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
     'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
SETS = {
    'dread': 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv',
    'steven_v4': 'storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
    'pilkwang_v2': 'storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv',
    'relabel_v1': 'storage/labels/derived_relabel/relabel_v1.csv',
}
JEV = 'storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv'
AQUILA = 'storage/labels/derived_constellation/aquila_v1.csv'
FLIGHT = 'storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv'
COMPONENTS = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
WEIGHTS = f'{EXP}/consensus-weights.csv'
INITIAL = f'{EXP}/initial-state.pt'
CORPUS_LOCAL = f'{EXP}/corpus-manifest.json'
MANIFEST = f'{EXP}/manifest.json'
PROTOCOL = f'{EXP}/PROTOCOL.md'
RECEIPT = 'exp150-bundle.json'
CORPUS_HASH = 'f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
RAD_HASH = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
DATASET = 'taeyangg4/knee-exp150-consensus-code'
KERNEL = 'taeyangg4/exp150-consensus-weight-f0-pair42'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def safe(name):
    p = PurePosixPath(name)
    need(str(p) == name and not p.is_absolute() and '..' not in p.parts and
         '\\' not in name and ':' not in name, 'Unsafe package path')
    return p


def table(path, ids):
    """Read only UID and target columns; never materialise report text."""
    import pandas as pd
    d = pd.read_csv(path, usecols=lambda c: c in {UID, *T}, dtype={UID: str})
    need(set(d.columns) == {UID, *T} and not d[UID].isna().any() and
         not d[UID].duplicated().any(), 'Invalid target-only label table')
    d = d.set_index(UID).reindex(ids)
    return d[T].apply(pd.to_numeric, errors='coerce').to_numpy(dtype='float64')


def weight_matrix(arrays):
    import numpy as np
    stack = np.stack(arrays)
    valid = np.isfinite(stack).all(axis=0)
    binary = stack >= .5
    agree = valid & (binary.all(axis=0) | (~binary).all(axis=0))
    return np.where(agree, 1., .4).astype(np.float32)


def canonical_ids(root):
    p = read(root / BASE)
    f = p['folds']['0']
    ordered = read(root / f['ordered_ids']['path'])
    ids = ordered['train'] + ordered['val']
    need(len(ids) == 4343 and len(set(ids)) == len(ids) and len(ordered['val']) == 888,
         'EXP099 fold-0 roster changed')
    return p, ordered, ids


def prepare(root=ROOT):
    """CPU only; exclusive paths. Register weights and one seed-42 model state for all arms."""
    root = Path(root)
    p, roster, ids = canonical_ids(root)
    dest = root / EXP
    need(not dest.exists(), 'EXP150 registration already exists; never overwrite')
    for name, digest in p['source_bindings'].items():
        if name in (TOOL, RUNNER):
            continue
        need(sha(root / name) == digest, 'Frozen binding changed: ' + name)
    arrays = [table(root / path, ids) for path in SETS.values()]
    w = weight_matrix(arrays)
    dest.mkdir(parents=True)
    corpus_source = root / p['paths']['corpus'] / 'manifest.json'
    need(sha(corpus_source) == CORPUS_HASH, 'Local corpus manifest changed')
    with (root / CORPUS_LOCAL).open('xb') as f:
        f.write(corpus_source.read_bytes())
    cm = read(root / CORPUS_LOCAL)
    corpus_list = [dict(file=n, bytes=(corpus_source.parent / n).stat().st_size, sha256=d)
                   for n, d in [('ids.npy', cm['ids_sha256']), ('masks.npy', cm['masks_sha256'])] +
                   [(s['file'], s['sha256']) for s in cm['shards']]]
    with (root / WEIGHTS).open('x', newline='', encoding='utf-8') as f:
        cw = csv.writer(f, lineterminator='\n')
        cw.writerow([UID, *T])
        for uid, row in zip(ids, w):
            cw.writerow([uid, *('1.0' if x == 1. else '0.4' for x in row)])
    # The CPU state is created ONCE. Remote model constructors may consume RNG,
    # but all constructed tensors are overwritten by this checked state before use.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    for key in THREADS:
        os.environ[key] = '1'
    sys.path.insert(0, str(root / 'training/src'))
    from tools import exp127_radr50_single as g
    from raptor_common import seed_all
    from raptor_radimagenet import build_radimagenet_model
    import torch
    need(not torch.cuda.is_initialized(), 'prepare must not touch CUDA')
    torch.set_num_threads(1)
    g.worker_init(0)
    seed_all(42)
    model, _ = build_radimagenet_model('resnet50', root / p['init_weights']['local'], RAD_HASH)
    state_hash, _ = g.state_hashes(model)
    need(state_hash == p['initial_state_sha256'], 'Local CPU init differs from EXP127 registered state')
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, root / INITIAL)
    need(not torch.cuda.is_initialized(), 'prepare initialized CUDA')
    aug1_summary = root / 'artifacts/experiments/EXP127_AUG1_S42_F0/summary.json'
    aug1_args = read(aug1_summary)['args']
    need(aug1_args['variant'] == 'AUG1' and aug1_args['seed'] == 42 and
         aug1_args['fold'] == 0, 'Source AUG1 argument list changed')
    manifest = dict(schema='exp150-consensus-weight-v1', status='REGISTERED_NO_EXECUTION',
        plan='H-B-44', base_manifest=BASE, base_manifest_sha256=sha(root / BASE),
        corpus_manifest_sha256=CORPUS_HASH, fold=0, seeds=[42, 43], arms=['C', 'W'],
        initial_state=INITIAL, initial_state_file_sha256=sha(root / INITIAL),
        initial_state_sha256=state_hash, weights=WEIGHTS, weights_sha256=sha(root / WEIGHTS),
        corpus_local_manifest=CORPUS_LOCAL, corpus_local_manifest_sha256=sha(root / CORPUS_LOCAL),
        corpus_file_list=corpus_list,
        label_sources={n: dict(path=path, sha256=sha(root / path)) for n, path in SETS.items()},
        source_sha256={TOOL: sha(root / TOOL), RUNNER: sha(root / RUNNER)},
        aug1_summary_sha256=sha(aug1_summary),
        readout_sha256={n: sha(root / n) for n in (JEV, AQUILA, FLIGHT, COMPONENTS)},
        roster=dict(train=len(roster['train']), val=len(roster['val']),
                    train_sha256=fhash(roster['train']), val_sha256=fhash(roster['val'])),
        weight_values=[.4, 1.], weighted_cells=int((w == .4).sum()),
        total_cells=int(w.size), bootstrap=dict(draws=300, seed=20261003))
    write_new(root / MANIFEST, manifest)
    lines = [
        '# EXP150 consensus-weighted supervision — registered before any run', '',
        'H-B-44; v1.0.0. No GPU run or submission is authorized by this protocol.', '',
        '## Fixed recipe', '',
        'EXP127 AUG1 RadImageNet ResNet50 single head, EXP099 fold 0, Dread labels,',
        '224 px, K12 train, K36 primary OOF (K24 auxiliary), 20 epochs fixed-final,',
        'bs8, AdamW bb/head LR 5e-4/1e-3, wd .02, OneCycle pct .15, fp16 strict.',
        'No Gold58 labels or predictions enter train, tuning or the registered gate.',
        'The exact source AUG1 arguments and variant options follow below; EXP150',
        'changes seed, tag, worker allocation, remote paths and the registered weight only.', '',
        '```json', json.dumps(aug1_args, indent=2, ensure_ascii=False), '```', '',
        '```json', json.dumps(p['variant_config'], indent=2, ensure_ascii=False), '```', '',
        '## Variable and pairs', '',
        'Per UID × target, C uses weight 1.0. W uses weight 1.0 if all four',
        'Dread/Steven v4/Pilkwang v2/relabel v1 labels exist and agree after >=0.5',
        'binarisation; otherwise 0.4. BCE positive term uses the unchanged fold-0',
        'Dread pos_weight. Weighted loss is sum of cell weights × BCE / (batch × 12),',
        'with no re-normalisation by sum of weights. C42/W42 share Kaggle T4 image',
        'and one 2×T4 notebook; one process per GPU. C43/W43 share the single Vast',
        'RTX3090, sequentially. Seed controls data ordering/augmentation; all four',
        'load the one CPU-generated initial state file. No cross-GPU paired deltas.',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8 is set before CUDA initialization.', '',
        '## Registered reading', '',
        'Primary independent references: JEV HYBRID binary and aquila_v1 >=0.5.',
        'Dread guard. Per pair, W minus C macro AUC on fold-0 K36 OOF, 888 studies,',
        'with paired component bootstrap, 300 draws, seed 20261003. ADVANCE to a',
        'separately registered five-fold test iff mean(JEV Δ, aquila Δ)>0 in BOTH',
        'seed pairs and Dread Δ>=-0.003 in BOTH. Otherwise HOLD/STOP. Report all',
        '12 target deltas, CIs and consensus/flight/relabel as biased report-only.',
        'No public leaderboard ratio/routing tuning.', '',
        '## Hash bindings', '',
        f'- weights CSV sha256: {manifest["weights_sha256"]}',
        f'- initial state file sha256: {manifest["initial_state_file_sha256"]}',
        f'- initial state tensor hash: {state_hash}',
        f'- base protocol sha256: {manifest["base_manifest_sha256"]}',
        f'- corpus manifest sha256: {CORPUS_HASH}',
        f'- EXP150 manifest sha256: {sha(root / MANIFEST)}',
    ]
    with (root / PROTOCOL).open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    return manifest


def fhash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def registration(root):
    root = Path(root)
    m = read(root / MANIFEST)
    need(m['schema'] == 'exp150-consensus-weight-v1' and sha(root / BASE) == m['base_manifest_sha256']
         and sha(root / WEIGHTS) == m['weights_sha256'] and sha(root / INITIAL) == m['initial_state_file_sha256'],
         'Registration hash mismatch')
    need(sha(root / CORPUS_LOCAL) == m['corpus_local_manifest_sha256'] == CORPUS_HASH,
         'Packaged local corpus list mismatch')
    for n, h in m['source_sha256'].items():
        need(sha(root / n) == h, 'EXP150 source changed: ' + n)
    return m, read(root / BASE)


def load_initial_state(model, path, protocol):
    import torch
    need(sha(path) == protocol['initial_state_file_sha256'], 'Initial state file mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    need(set(state) == set(model.state_dict()), 'Initial state keys differ')
    model.load_state_dict(state, strict=True)


def load_weights(path, train_ids, root, protocol):
    import numpy as np
    need(sha(path) == protocol['weights_sha256'], 'Weight CSV hash mismatch')
    with Path(path).open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == [UID, *T], 'Weight CSV columns changed')
        rows = {r[UID]: np.array([float(r[t]) for t in T], np.float32) for r in reader}
    need(len(rows) == protocol['roster']['train'] + protocol['roster']['val'] and
         set(train_ids).issubset(rows) and all(np.isin(x, np.array([.4, 1.], np.float32)).all() for x in rows.values()),  # B: float32 rows vs float64 .4 never matched
         'Weight CSV rows/values changed')
    return {uid: rows[uid] for uid in train_ids}


def validate_runner_args(a, root):
    root = Path(root)
    m, p = registration(root)
    f = p['folds']['0']
    need(a.variant == 'AUG1' and a.fold == 0 and a.seed in (42, 43) and
         a.tag == f'EXP150_{a.arm}{a.seed}_F0' and
         Path(a.protocol).resolve() == (root / MANIFEST).resolve(), 'Wrong EXP150 scope')
    need(Path(a.initial_state).resolve() == (root / INITIAL).resolve() and
         Path(a.consensus_weights).resolve() == (root / WEIGHTS).resolve(), 'Wrong EXP150 inputs')
    for k, value in p['common_args'].items():
        need(getattr(a, k) == value or (a.mode == 'smoke' and k == 'device' and a.device == 'cpu'),
             'Recipe argument differs: ' + k)
    need(a.mode == 'full' or (a.mode == 'smoke' and a.device == 'cpu'), 'Invalid run mode')
    need(a.workers in range(0, 13), 'Invalid workers')
    for k in ('corpus', 'labels', 'eval_labels'):
        need(Path(getattr(a, k)).resolve() == (root / p['paths'][k]).resolve(), 'Wrong input: ' + k)
    need(Path(a.init_weights).resolve() == (root / INITIAL).resolve(), 'Wrong model input file')
    ordered = read(root / f['ordered_ids']['path'])
    need(fhash(ordered['train']) == m['roster']['train_sha256'] and
         fhash(ordered['val']) == m['roster']['val_sha256'], 'Fold roster changed')
    for k in ('ordered_ids', 'groups'):
        need(sha(root / f[k]['path']) == f[k]['sha256'], 'Fold metadata changed')
    if a.mode == 'full':
        need(sys.platform.startswith('linux') and os.environ.get('EXP150_REMOTE') in ('kaggle', 'vast'),
             'Full jobs only on registered remote surfaces')
    out = Path(a.output_root).resolve()
    need(out != (root / 'artifacts/experiments').resolve(), 'Use isolated output root')
    return {**p, **m, 'paths': p['paths'], 'folds': p['folds'], 'label_set': p['label_set'],
            'common_args': p['common_args'], 'extra_eval': p['extra_eval'],
            'variant_config': p['variant_config'], 'environment': p['environment'],
            'seed': a.seed, 'init_weights': p['init_weights'], 'initial_state_sha256': m['initial_state_sha256']}, f, ordered, out


def runner_args(root, arm, seed, weights_path, output, *, mode='full'):
    p = read(Path(root) / BASE)
    args = ['--variant', 'AUG1', '--protocol', str(Path(root) / MANIFEST), '--fold', '0',
            '--mode', mode, '--tag', f'EXP150_{arm}{seed}_F0', '--arm', arm,
            '--seed', str(seed), '--workers', '2' if os.environ.get('EXP150_REMOTE') == 'kaggle' else '8',
            '--output-root', str(output), '--init-weights', str(Path(root) / INITIAL),
            '--initial-state', str(Path(root) / INITIAL),
            '--consensus-weights', str(Path(root) / WEIGHTS)]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v: args.append('--' + k.replace('_', '-'))
        else:
            args += ['--' + k.replace('_', '-'), str(v)]
    for k in ('corpus', 'labels', 'eval_labels'):
        args += ['--' + k.replace('_', '-'), str(Path(root) / p['paths'][k])]
    if mode == 'smoke':
        i = args.index('--device'); args[i+1] = 'cpu'
        args += ['--smoke-train', '8', '--smoke-val', '1', '--smoke-epochs', '1']
    return args


def package_files(root):
    root = Path(root)
    m, p = registration(root)
    f = p['folds']['0']
    names = {MANIFEST, PROTOCOL, WEIGHTS, INITIAL, CORPUS_LOCAL, BASE, RUNNER, TOOL,
             'training/runners/exp127_radr50_single.py',
             'tools/__init__.py', p['label_set']['path'], p['paths']['base_ordered_ids'],
             p['paths']['base_groups'], p['paths']['folds'], f['ordered_ids']['path'],
             f['groups']['path']}
    names |= {x.relative_to(root).as_posix() for folder in ('training/src', 'tools')
              for x in (root / folder).glob('*.py')}
    files = {}
    for name in sorted(names):
        safe(name)
        source = root / name
        need(source.is_file() and not source.is_symlink(), 'Missing/unsafe package file: ' + name)
        files[name] = dict(sha256=sha(source), bytes=source.stat().st_size)
    need(files[BASE]['sha256'] == m['base_manifest_sha256'] and
         files[WEIGHTS]['sha256'] == m['weights_sha256'] and
         files[INITIAL]['sha256'] == m['initial_state_file_sha256'], 'Package binding mismatch')
    return files


def verify_corpus(directory, local_manifest, file_list, expected=CORPUS_HASH):
    directory = Path(directory)
    need(sha(directory / 'manifest.json') == expected, 'Corpus manifest hash mismatch')
    need(sha(local_manifest) == expected, 'Local corpus file list hash mismatch')
    m = read(directory / 'manifest.json')
    need(m == read(local_manifest), 'Corpus differs from packaged local file list')
    need(m['contract'] == 'raptor64_336_c140_s02' and m['finished'], 'Incomplete corpus')
    checks = [('ids.npy', m['ids_sha256']), ('masks.npy', m['masks_sha256'])]
    for s in m['shards']:
        need(s['done'] and Path(s['file']).name == s['file'], 'Incomplete/unsafe shard')
        need((directory / s['file']).stat().st_size == s['bytes'], 'Corpus shard size mismatch')
        checks.append((s['file'], s['sha256']))
    for name, digest in checks:
        rec = [x for x in file_list if x['file'] == name]
        need(len(rec) == 1 and rec[0]['sha256'] == digest and
             (directory / name).stat().st_size == rec[0]['bytes'], 'Local corpus file list mismatch: ' + name)
        need(sha(directory / name) == digest, 'Corpus file hash mismatch: ' + name)
    return {'manifest_sha256': expected, 'files': len(checks), 'total_bytes': sum((directory / n).stat().st_size for n, _ in checks)}


def stage(kind, out, root=ROOT):
    root, out = Path(root), Path(out)
    need(kind in ('kaggle', 'vast') and not out.exists(), 'Use a new stage directory')
    files = package_files(root)
    receipt = dict(schema='exp150-package-v1', surface=kind, files=files,
                   manifest_sha256=sha(root / MANIFEST), protocol_sha256=sha(root / PROTOCOL),
                   corpus_manifest_sha256=CORPUS_HASH, no_execution=True)
    out.mkdir(parents=True)
    code = out / ('code-dataset' if kind == 'kaggle' else 'package')
    code.mkdir()
    marker = b'# EXP150 isolated workspace; no credentials.\n'
    receipt['files']['config/workspace.toml'] = dict(sha256=hashlib.sha256(marker).hexdigest(), bytes=len(marker))
    rb = (json.dumps(receipt, indent=2, ensure_ascii=False) + '\n').encode()
    archive = code / 'exp150-code.zip.bin'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.write(root / name, name)
        z.writestr('config/workspace.toml', marker)
        z.writestr(RECEIPT, rb)
    (out / RECEIPT).write_bytes(rb)
    write_new(out / 'stage-receipt.json', dict(schema='exp150-stage-v1', surface=kind,
        archive_sha256=sha(archive), bundle_sha256=hashlib.sha256(rb).hexdigest(),
        manifest_sha256=receipt['manifest_sha256'], protocol_sha256=receipt['protocol_sha256']))
    if kind == 'kaggle':
        write_new(code / 'dataset-metadata.json', dict(id=DATASET, title='EXP150 consensus weight code private',
            isPrivate=True, licenses=[{'name': 'other'}],
            description='Private experiment package; source licences retained. No new rights granted.'))
        source = notebook_source(sha(archive), hashlib.sha256(rb).hexdigest())
        write_new(out / 'run.ipynb', dict(nbformat=4, nbformat_minor=5,
            metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
            cells=[dict(cell_type='code', id='exp150', metadata={}, execution_count=None,
                        outputs=[], source=source.splitlines(keepends=True))]))
        write_new(out / 'kernel-metadata.json', dict(id=KERNEL,
            title='EXP150 consensus weight F0 pair42', code_file='run.ipynb', language='python',
            kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
            enable_internet=False, machine_shape='NvidiaTeslaT4', docker_image=DOCKER,
            dataset_sources=['taeyangg4/knee-raptor-corpus-336',
                             'marwanmath/resnet-50-radimagenet-marwan', DATASET],
            competition_sources=['rsna-knee-abnormality-detection'], kernel_sources=[], model_sources=[]))
        compile(source, '<exp150-notebook>', 'exec')
    else:
        write_new(out / 'vast-queue.json', [dict(tag='EXP150_CW43_VAST_F0', command=[
            'python', TOOL, 'remote-run', '--surface', 'vast', '--bundle-sha',
            hashlib.sha256(rb).hexdigest(), '--started', '0'])])
    verify_stage(kind, out, root)
    return read(out / 'stage-receipt.json')


def verify_stage(kind, out, root=ROOT):
    root, out = Path(root), Path(out)
    r = read(out / 'stage-receipt.json')
    need(r['surface'] == kind and r['schema'] == 'exp150-stage-v1', 'Wrong stage')
    archive = out / ('code-dataset' if kind == 'kaggle' else 'package') / 'exp150-code.zip.bin'
    need(sha(archive) == r['archive_sha256'] and sha(out / RECEIPT) == r['bundle_sha256'], 'Stage hash mismatch')
    receipt = read(out / RECEIPT)
    with zipfile.ZipFile(archive) as z:
        need(set(z.namelist()) == {*receipt['files'], RECEIPT}, 'Package entry mismatch')
        for n, rec in receipt['files'].items():
            safe(n)
            data = z.read(n)
            need(len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256'],
                 'Package file mismatch: ' + n)
            if n != 'config/workspace.toml':
                need(sha(root / n) == rec['sha256'], 'Local file changed since stage: ' + n)
    if kind == 'kaggle':
        meta = read(out / 'kernel-metadata.json')
        need(meta['is_private'] and meta['machine_shape'] == 'NvidiaTeslaT4' and
             meta['docker_image'] == DOCKER, 'Kaggle metadata mismatch')
        need(''.join(read(out / 'run.ipynb')['cells'][0]['source']) ==
             notebook_source(r['archive_sha256'], r['bundle_sha256']), 'Notebook drift')
    else:
        queue = read(out / 'vast-queue.json')
        need(len(queue) == 1 and queue[0]['tag'] == 'EXP150_CW43_VAST_F0' and
             queue[0]['command'] == ['python', TOOL, 'remote-run', '--surface', 'vast',
                                    '--bundle-sha', r['bundle_sha256'], '--started', '0'],
             'Vast queue drift')
    return r


def notebook_source(archive_sha, bundle_sha):
    return f'''# EXP150 private paired training; no network, credentials or submission.
import hashlib, json, os, pathlib, subprocess, sys, time, zipfile
started = time.time()
inp = pathlib.Path('/kaggle/input'); work = pathlib.Path('/kaggle/working')
hits = list(inp.rglob('exp150-code.zip.bin'))
assert len(hits) == 1 and hashlib.sha256(hits[0].read_bytes()).hexdigest() == {archive_sha!r}
ws = pathlib.Path('/tmp/exp150-workspace')
assert not ws.exists()
with zipfile.ZipFile(hits[0]) as z:
    names = z.namelist(); assert len(set(names)) == len(names)
    for name in names:
        p = pathlib.PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
    z.extractall(ws)
env = dict(os.environ, KNEE_WORKSPACE_ROOT=str(ws), EXP150_REMOTE='kaggle',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
           HF_HUB_OFFLINE='1')
for key in {THREADS!r}: env[key] = '1'
cmd = [sys.executable, str(ws / {TOOL!r}), 'remote-run', '--surface', 'kaggle',
       '--bundle-sha', {bundle_sha!r}, '--started', str(started)]
p = subprocess.run(cmd, cwd=ws, env=env, timeout=200*60+60)
assert p.returncode == 0, 'EXP150 incomplete; inspect exp150-status.json and logs'
print((work / 'exp150-status.json').read_text())
'''


def kaggle_push_plan(out):
    verify_stage('kaggle', out)
    out = Path(out)
    return '\n'.join([
        '# B checks live Kaggle GPU quota, slot availability, then executes:',
        "$gpuPython = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'",
        f'& .venv/Scripts/python.exe -X utf8 -m kaggle datasets create -p {out.as_posix()}/code-dataset --dir-mode zip',
        f'# If that dataset ID already exists instead: & .venv/Scripts/python.exe -X utf8 -m kaggle datasets version -p {out.as_posix()}/code-dataset -m "EXP150 v1.0.0" --dir-mode zip',
        f'& .venv/Scripts/python.exe -X utf8 -m kaggle kernels push -p {out.as_posix()}',
        f'& .venv/Scripts/python.exe -X utf8 -m kaggle kernels output {KERNEL} -p {out.as_posix()}/download --page-size 200',
        f'& $gpuPython tools/exp150_consensus_weight.py collect --surface kaggle --source {out.as_posix()}/download',
        '& $gpuPython tools/exp150_consensus_weight.py screen',
    ])


def vast_plan(out, *, host, port, user='root'):
    verify_stage('vast', out)
    need(host and all(c.isalnum() or c in '.-' for c in host) and
         isinstance(port, int) and 1 <= port <= 65535 and user == 'root',
         'Live reviewed root endpoint required')
    out = Path(out)
    kh = '.local/hc06-illinois-stable-20261009-v10/known_hosts-54941539-v1'
    key = 'C:/Users/Taeyang/.ssh/id_ed25519'
    archive = (out / 'package/exp150-code.zip.bin').as_posix()
    options = f'-i {key} -o UserKnownHostsFile={kh} -o StrictHostKeyChecking=yes -P {port}'
    sshopt = f'-i {key} -o UserKnownHostsFile={kh} -o StrictHostKeyChecking=yes -p {port}'
    return '\n'.join([
        '# B first verifies the live instance ID, endpoint, host key, budget and idle GPU; claims Active compute.',
        "$gpuPython = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'",
        f'scp {options} {archive} {user}@{host}:/workspace/exp150-code.zip.bin',
        f'ssh {sshopt} {user}@{host} "mkdir -p /workspace/exp150 && python -m zipfile -e /workspace/exp150-code.zip.bin /workspace/exp150"',
        f'scp {options} {(out / "vast-queue.json").as_posix()} {user}@{host}:/workspace/exp150/vast-queue.json',
        f'ssh {sshopt} {user}@{host} "cd /workspace/exp150 && EXP150_REMOTE=vast KNEE_WORKSPACE_ROOT=/workspace/exp150 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 python tools/gpu_queue.py vast-queue.json --foreground"',
        f'scp {options} -r {user}@{host}:/workspace/exp150/results {out.as_posix()}/download',
        f'& $gpuPython tools/exp150_consensus_weight.py collect --surface vast --source {out.as_posix()}/download',
        '& $gpuPython tools/exp150_consensus_weight.py screen',
    ])


def verify_bundle(root, bundle_sha):
    root = Path(root)
    need(sha(root / RECEIPT) == bundle_sha, 'Bundle receipt hash mismatch')
    r = read(root / RECEIPT)
    need(r['schema'] == 'exp150-package-v1', 'Wrong bundle schema')
    for name, rec in r['files'].items():
        safe(name)
        p = root / name
        need(p.is_file() and not p.is_symlink() and p.stat().st_size == rec['bytes'] and
             sha(p) == rec['sha256'], 'Bundle file mismatch: ' + name)
    registration(root)
    return r


def locate_corpus(surface):
    base = Path('/kaggle/input') if surface == 'kaggle' else Path('/workspace/rsna-hc06-illinois-pair-v2')
    need(base.is_dir(), 'Expected corpus search root missing')
    hits = []
    for candidate in base.rglob('manifest.json'):
        if sha(candidate) == CORPUS_HASH:
            hits.append(candidate.parent)
    need(len(hits) == 1, 'Expected exactly one hash-matched corpus')
    return hits[0]


def remote_run(surface, bundle_sha, started, root=ROOT):
    """Remote-only worker supervisor; never invoked by prepare/stage/validate/smoke."""
    root = Path(root)
    need(sys.platform.startswith('linux') and os.environ.get('EXP150_REMOTE') == surface and
         surface in ('kaggle', 'vast') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
         'Wrong execution surface or determinism environment')
    if surface == 'kaggle':
        need(Path('/kaggle/working').is_dir(), 'Not a Kaggle notebook')
        output = Path('/kaggle/working')
        pairs = [('C', 42, '0'), ('W', 42, '1')]
    else:
        need(str(root) == '/workspace/exp150', 'Wrong Vast root')
        output = root / 'results'
        pairs = [('C', 43, '0'), ('W', 43, '0')]
    need(not (output / 'exp150-status.json').exists(), 'Refusing previous supervisor output')
    receipt = verify_bundle(root, bundle_sha)
    need(receipt['surface'] == surface, 'Wrong surface bundle')
    started = started or time.time()
    deadline = started + (200 * 60 if surface == 'kaggle' else 5 * 3600)
    status = dict(schema='exp150-run-v1', surface=surface, state='PREFLIGHT',
                  started=started, bundle_sha256=bundle_sha)
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / 'exp150-status.json', status)
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True,
                             check=True, timeout=30).stdout.strip().splitlines()
        need(len(smi) == (2 if surface == 'kaggle' else 1), 'Wrong GPU count')
        need(all(('T4' if surface == 'kaggle' else '3090') in x for x in smi), 'Wrong GPU type')
        import torch, timm, numpy
        need(not torch.cuda.is_initialized(), 'Supervisor unexpectedly initialized CUDA')
        if surface == 'kaggle':
            need(sys.version_info[:2] == (3, 13) and torch.__version__ == '2.11.0+cu128'
                 and timm.__version__ == '1.0.29', 'Pinned Kaggle image stack changed')
        status['software'] = dict(python=sys.version.split()[0], torch=torch.__version__,
                                  timm=timm.__version__, numpy=numpy.__version__)
        status['gpu'] = smi
        corpus = locate_corpus(surface)
        status['corpus'] = verify_corpus(corpus, root / CORPUS_LOCAL,
                                          read(root / MANIFEST)['corpus_file_list'])
        p = read(root / BASE)
        destination = root / p['paths']['corpus']
        destination.parent.mkdir(parents=True, exist_ok=True)
        need(not destination.exists(), 'Refusing existing corpus path')
        destination.symlink_to(corpus, target_is_directory=True)
        shutil.copyfile(root / RECEIPT, output / RECEIPT)
        procs = []
        for arm, seed, gpu in pairs:
            tag = f'EXP150_{arm}{seed}_F0'
            need(not (output / tag).exists(), 'Refusing existing run directory')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       KNEE_REMOTE_RELAX_ENV='1', KNEE_WORKSPACE_ROOT=str(root),
                       EXP150_REMOTE=surface, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       HF_HUB_OFFLINE='1', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
            for key in THREADS: env[key] = '1'
            log = (output / f'{tag}.launcher.log').open('x', encoding='utf-8')
            cmd = [sys.executable, str(root / RUNNER)] + runner_args(root, arm, seed, root / INITIAL, output)
            child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            procs.append((child, log, arm, seed, gpu))
            if surface == 'vast':
                code = child.wait(timeout=max(1, deadline - time.time()))
                log.close()
                need(code == 0, f'Vast {tag} failed with exit {code}')
        if surface == 'kaggle':
            codes = [child.wait(timeout=max(1, deadline - time.time())) for child, *_ in procs]
            for _, log, *_ in procs: log.close()
            need(codes == [0, 0], f'Kaggle pair incomplete: {codes}')
        for _, _, arm, seed, gpu in procs:
            tag = f'EXP150_{arm}{seed}_F0'
            run = output / tag
            prov = read(run / 'provenance.json')
            summary = read(run / 'summary.json')
            hardware = dict(physical_gpu=smi[int(gpu)], gpu_visible_device=gpu,
                            torch=torch.__version__, timm=timm.__version__, numpy=numpy.__version__,
                            cuda=torch.version.cuda, driver=smi[int(gpu)].split(',')[-1].strip())
            prov['exp150'].update(surface=surface, hardware=hardware, bundle_sha256=bundle_sha,
                                  manifest_sha256=sha(root / MANIFEST), protocol_sha256=sha(root / PROTOCOL))
            summary['provenance'] = prov
            with (run / 'provenance.json').open('w', encoding='utf-8') as f: json.dump(prov, f, indent=2)
            with (run / 'summary.json').open('w', encoding='utf-8') as f: json.dump(summary, f, indent=2)
            write_new(run / 'exp150-run.json', dict(schema='exp150-run-v1', state='COMPLETE',
                arm=arm, seed=seed, surface=surface, hardware=hardware, bundle_sha256=bundle_sha,
                initial_state_sha256=prov['initial_state_sha256'],
                initial_file_sha256=sha(root / INITIAL), weight_csv_sha256=sha(root / WEIGHTS),
                oof_k36_sha256=sha(run / 'oof_k36.csv')))
            names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                     'oof_best.csv', 'oof_k36.csv', 'exp150-run.json']
            with (run / 'output-inventory.json').open('w', encoding='utf-8') as f:
                json.dump({n: dict(sha256=sha(run / n), bytes=(run / n).stat().st_size)
                           for n in names}, f, indent=2)
        status.update(state='COMPLETE', tags=[f'EXP150_{a}{s}_F0' for _, _, a, s, _ in procs])
    except BaseException as exc:
        status.update(state='FAILED_OR_INCOMPLETE', reason=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        status['elapsed_seconds'] = time.time() - started
        (output / 'exp150-status.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')


def verify_run(run, arm, seed, surface, root=ROOT):
    root, run = Path(root), Path(run)
    m, p = registration(root)
    need(run.name == f'EXP150_{arm}{seed}_F0', 'Run tag mismatch')
    inv = read(run / 'output-inventory.json')
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                'oof_best.csv', 'oof_k36.csv', 'exp150-run.json'}
    need(required == set(inv), 'Incomplete output inventory')
    for n, r in inv.items():
        need(Path(n).name == n and sha(run / n) == r['sha256'] and
             (run / n).stat().st_size == r['bytes'], 'Output hash/size mismatch: ' + n)
    summary, prov, receipt = (read(run / n) for n in ('summary.json', 'provenance.json', 'exp150-run.json'))
    f = p['folds']['0']
    need(receipt['state'] == 'COMPLETE' and receipt['arm'] == arm and
         receipt['seed'] == seed and receipt['surface'] == surface and
         receipt['oof_k36_sha256'] == sha(run / 'oof_k36.csv'), 'Run receipt mismatch')
    need(prov['exp150']['arm'] == arm and prov['exp150']['surface'] == surface and
         prov['exp150']['initial_state_file_sha256'] == m['initial_state_file_sha256'] and
         prov['exp150']['consensus_weights_sha256'] == m['weights_sha256'] and
         prov['initial_state_sha256'] == m['initial_state_sha256'] and
         prov['corpus_manifest_sha256'] == CORPUS_HASH, 'Run provenance mismatch')
    need(prov['train_ids_ordered_sha256'] == m['roster']['train_sha256'] and
         prov['eval_ids_ordered_sha256'] == m['roster']['val_sha256'] and
         prov['actual_pos_weight'] == f['pos_weight'], 'Roster/pos_weight mismatch')
    need(summary['provenance'] == prov and summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == f['steps'] and summary['rows'] == len(read(root / f['ordered_ids']['path'])['val']),
         'Full fixed-final recipe mismatch')
    args = summary['args']
    need(all(args.get(k) == v for k, v in p['common_args'].items()) and
         args['seed'] == seed and args['arm'] == arm and args['variant'] == 'AUG1' and
         args['fold'] == 0 and args['mode'] == 'full' and
         args['tag'] == f'EXP150_{arm}{seed}_F0' and
         summary['train'] == f['n_train'] and summary['val'] == f['n_val'] and
         summary['gold'] == 0 and summary['skipped_updates'] <= f['max_fp16_skips'],
         'Recorded argument/count/skip contract mismatch')
    history = read(run / 'history.json')
    need([x['epoch'] for x in history] == list(range(20)) and history == summary['history'],
         'Epoch history mismatch')
    need(all(x in prov['exp150']['hardware'] for x in ('driver', 'torch', 'timm', 'numpy', 'cuda', 'physical_gpu')),
         'Missing platform provenance')
    if surface == 'kaggle':
        need('T4' in prov['exp150']['hardware']['physical_gpu'], 'Wrong Kaggle GPU')
    else:
        need('3090' in prov['exp150']['hardware']['physical_gpu'], 'Wrong Vast GPU')
    ids = read(root / f['ordered_ids']['path'])['val']
    with (run / 'oof_k36.csv').open(newline='', encoding='utf-8') as stream:
        dr = csv.DictReader(stream)
        need(dr.fieldnames == [UID, *T], 'OOF columns changed')
        seen = []
        for row in dr:
            seen.append(row[UID])
            need(all(0. <= float(row[t]) <= 1. for t in T), 'Invalid OOF probability')
    need(seen == ids, 'OOF roster/order mismatch')
    return {n: r['sha256'] for n, r in inv.items()}


def collect(surface, source, root=ROOT):
    root, source = Path(root), Path(source)
    need(surface in ('kaggle', 'vast'), 'Wrong surface')
    seed = 42 if surface == 'kaggle' else 43
    need(read(source / 'exp150-status.json')['state'] == 'COMPLETE', 'Remote supervisor incomplete')
    for arm in ('C', 'W'):
        verify_run(source / f'EXP150_{arm}{seed}_F0', arm, seed, surface, root)
    dest = root / EXP / 'outputs' / surface
    need(not dest.exists(), 'Refusing existing collected output')
    shutil.copytree(source, dest)
    for arm in ('C', 'W'):
        verify_run(dest / f'EXP150_{arm}{seed}_F0', arm, seed, surface, root)
    return {'surface': surface, 'destination': str(dest), 'status': 'VERIFIED'}


def paired_screen(root=ROOT):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp125_radr50_k16_local as e125
    root = Path(root)
    m, p = registration(root)
    ids = read(root / p['folds']['0']['ordered_ids']['path'])['val']
    need(len(ids) == 888, 'Wrong fold-0 screen roster')
    for name, digest in m['readout_sha256'].items():
        need(sha(root / name) == digest, 'Registered readout changed: ' + name)
    draws = e125.component_draws(ids, read(root / COMPONENTS), draws=300, seed=20261003)
    refs = {}
    jev = table(root / JEV, ids)
    need(np.isfinite(jev).all() and np.isin(jev, [0., 1.]).all(), 'JEV must be complete binary')
    refs['jev_hybrid'] = (jev, np.ones(jev.shape, bool), 1)
    aquila = table(root / AQUILA, ids)
    need(np.isfinite(aquila).all(), 'Aquila must be complete')
    refs['aquila_v1'] = ((aquila >= .5).astype(float), np.ones(aquila.shape, bool), 1)
    dread = table(root / SETS['dread'], ids)
    need(np.isfinite(dread).all(), 'Dread must be complete')
    refs['dread'] = (np.rint(dread.astype(np.float32)).astype(float), np.ones(dread.shape, bool), 1)
    for name, path in [('flight_v4', FLIGHT), ('relabel_v1', SETS['relabel_v1'])]:
        x = table(root / path, ids)
        need(np.isfinite(x).all(), 'Report reference incomplete: ' + name)
        refs[name] = ((x >= .5).astype(float), np.ones(x.shape, bool), 1)
    stack = np.stack([table(root / path, ids) for path in SETS.values()])
    valid = np.isfinite(stack).all(axis=0)
    binary = stack >= .5
    agree = valid & (binary.all(axis=0) | (~binary).all(axis=0))
    refs['consensus'] = (binary[0].astype(float), agree, 2)
    results = {}
    bindings = {}
    for surface, seed in [('kaggle', 42), ('vast', 43)]:
        base = root / EXP / 'outputs' / surface
        a = {}
        for arm in ('C', 'W'):
            run = base / f'EXP150_{arm}{seed}_F0'
            verify_run(run, arm, seed, surface, root)
            pred = run / 'oof_k36.csv'
            a[arm] = e125.load_table(pred, ids, prediction=True)
            bindings[str(pred.relative_to(root)).replace('\\', '/')] = sha(pred)
        pair = {}
        for name, ref in refs.items():
            body = ev.compare(a['W'], a['C'], ref, draws)
            need(np.isfinite(body['delta']) and np.isfinite(body['ci95']).all(),
                 'Nonfinite paired macro delta')
            body['targets'] = {k: (v if np.isfinite(v) else None) for k, v in body['targets'].items()}
            pair[name] = body
        mean_primary = (pair['jev_hybrid']['delta'] + pair['aquila_v1']['delta']) / 2
        pair['registered_gate'] = dict(mean_primary_delta=mean_primary,
            independent_positive=mean_primary > 0,
            dread_guard=pair['dread']['delta'] >= -.003)
        results[str(seed)] = pair
    advance = all(results[str(seed)]['registered_gate']['independent_positive'] and
                  results[str(seed)]['registered_gate']['dread_guard'] for seed in (42, 43))
    out = root / EXP / 'screen'
    need(not out.exists(), 'Screen already exists; refuse overwrite')
    out.mkdir()
    body = dict(schema='exp150-screen-v1', registered_rule='mean(JEV,aquila)>0 and Dread>=-0.003 in both pairs',
                verdict='ADVANCE_TO_REGISTERED_FIVE_FOLDS' if advance else 'HOLD_STOP',
                rows=len(ids), bootstrap=dict(draws=300, seed=20261003, unit='validation component'),
                manifest_sha256=sha(root / MANIFEST), protocol_sha256=sha(root / PROTOCOL),
                readout_sha256=m['readout_sha256'], predictions_sha256=bindings,
                pairs=results, report_only=['consensus', 'flight_v4', 'relabel_v1'])
    write_new(out / 'result.json', body)
    return body


def validate_fake():
    import numpy as np
    a = [np.array([[.1, .9, np.nan], [.5, .2, .7]], np.float32),
         np.array([[.1, .9, .1], [.5, .2, .7]], np.float32),
         np.array([[.1, .1, .1], [.5, .2, .7]], np.float32),
         np.array([[.1, .9, .1], [.5, .2, .7]], np.float32)]
    w = weight_matrix(a)
    need(w.tolist() == [[1., .4000000059604645, .4000000059604645],
                        [1., 1., 1.]], 'Fake weight semantics failed')
    return {'status': 'VALIDATED_CPU_FAKES', 'weights': w.shape, 'cuda_initialized': False}


def cpu_smoke():
    import torch
    import torch.nn.functional as F
    need(not torch.cuda.is_initialized(), 'CPU smoke touched CUDA')
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 12)
    x = torch.ones((2, 3), dtype=torch.float32)
    y = torch.zeros((2, 12), dtype=torch.float32)
    w = torch.tensor([[1.] * 12, [.4] * 12])
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction='none') * w).mean()
    loss.backward(); opt.step()
    need(torch.isfinite(loss) and not torch.cuda.is_initialized(), 'CPU smoke failed')
    return {'status': 'ONE_CPU_STEP', 'loss': float(loss), 'cuda_initialized': False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('validate')
    sub.add_parser('cpu-smoke')
    for name in ('kaggle-stage', 'vast-stage', 'kaggle-push-plan', 'vast-plan'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        if name == 'vast-plan':
            p.add_argument('--host', required=True)
            p.add_argument('--port', required=True, type=int)
            p.add_argument('--user', default='root')
    rr = sub.add_parser('remote-run')
    rr.add_argument('--surface', required=True, choices=['kaggle', 'vast'])
    rr.add_argument('--bundle-sha', required=True)
    rr.add_argument('--started', required=True, type=float)
    co = sub.add_parser('collect')
    co.add_argument('--surface', required=True, choices=['kaggle', 'vast'])
    co.add_argument('--source', required=True)
    sub.add_parser('screen')
    a = ap.parse_args(argv)
    if a.command == 'prepare': value = prepare()
    elif a.command == 'validate': value = validate_fake()
    elif a.command == 'cpu-smoke': value = cpu_smoke()
    elif a.command == 'kaggle-stage': value = stage('kaggle', a.out)
    elif a.command == 'vast-stage': value = stage('vast', a.out)
    elif a.command == 'kaggle-push-plan': value = kaggle_push_plan(a.out)
    elif a.command == 'vast-plan': value = vast_plan(a.out, host=a.host, port=a.port, user=a.user)
    elif a.command == 'remote-run': value = remote_run(a.surface, a.bundle_sha, a.started)
    elif a.command == 'collect': value = collect(a.surface, a.source)
    else: value = paired_screen()
    print(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
