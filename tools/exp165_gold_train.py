"""EXP165 v1.1.0: private Gold58 training harm screen. Local commands are CPU-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp150_consensus_weight as e150

EXP = 'artifacts/experiments/EXP165_GOLD_TRAIN'
PROTOCOL = f'{EXP}/PROTOCOL.md'
GOLD = f'{EXP}/gold-labels.csv'
AUDIT = f'{EXP}/label-audit.json'
MANIFEST = f'{EXP}/manifest.json'
RUNNER = 'training/runners/exp165_gold_train.py'
TOOL = 'tools/exp165_gold_train.py'
RECEIPT = 'exp165-bundle.json'
DATASET = 'taeyangg4/knee-exp165-gold-train-private'
KERNEL = 'taeyangg4/exp165-gold-train-f0-s42'
THREADS = e150.THREADS
WALL_SECONDS = {'kaggle': 7800, 'vast': 2100}
SEEDS = {'kaggle': 42, 'vast': 43}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return e150.sha(path)


def write_new(path, value):
    e150.write_new(path, value)


def labels_from_official(path):
    """Target-only read. Errors and receipts never contain UIDs or target values."""
    import numpy as np
    import pandas as pd
    columns = {e150.UID, *e150.T}
    frame = pd.read_csv(path, usecols=lambda name: name in columns, dtype={e150.UID: str})
    need(set(frame.columns) == columns, 'Official target columns missing')
    result = {}
    for uid, values in zip(frame[e150.UID], frame[e150.T].to_numpy()):
        present = pd.notna(values)
        if not present.any():
            continue
        need(present.all(), 'Official labels must be complete binary rows')
        numbers = pd.to_numeric(values, errors='coerce').astype(float)
        need(np.isfinite(numbers).all() and np.isin(numbers, [0., 1.]).all(),
             'Official labels must be complete binary rows')
        need(isinstance(uid, str) and bool(uid) and uid not in result,
             'Official UID missing or duplicated')
        result[uid] = tuple(str(int(x)) for x in numbers)
    need(len(result) == 58, 'Expected exactly 58 official labelled studies')
    return result


def disjoint_audit(gold_ids, fold_rosters, components, corpus_ids, report_groups=None):
    gold = set(gold_ids)
    need(len(gold) == 58, 'Gold roster changed')
    need(gold.issubset(set(corpus_ids)), 'Gold58 missing from registered RAPTOR corpus')
    all_fold = set()
    for roster in fold_rosters:
        all_fold.update(roster['train'])
        all_fold.update(roster['val'])
    need(not (gold & all_fold), 'Gold UID overlaps an EXP099 fold')
    need(len(fold_rosters[0]['val']) == 888 and not (gold & set(fold_rosters[0]['val'])),
         'Gold UID overlaps V888')
    membership = {}
    for component in components:
        for uid in component['members']:
            need(uid not in membership, 'Duplicate validation component membership')
            membership[uid] = component['component_id']
    fold_components = {membership[u] for u in all_fold if u in membership}
    gold_components = {membership[u] for u in gold if u in membership}
    need(not (gold_components & fold_components), 'Gold component overlaps EXP099 fold')
    need(not (gold_components & {membership[u] for u in fold_rosters[0]['val'] if u in membership}),
         'Gold component overlaps V888')
    # Gold rows are outside the 4,343-member validation graph. Extend its
    # report-equality edges from the precomputed EXP026 hash metadata; this
    # does not open report text. Known duplicate edges are already in components.
    if report_groups is not None:
        need((gold | all_fold).issubset(report_groups), 'Report-group metadata incomplete')
        need(not ({report_groups[u] for u in gold} & {report_groups[u] for u in all_fold}),
             'Gold report component overlaps EXP099 fold')
    return dict(gold_rows=58, corpus_covered=58, uid_overlap=0, component_overlap=0,
                report_component_overlap=0, folds_checked=len(fold_rosters), v888_rows=888)


def report_group_map(path):
    result = {}
    gold_flags = set()
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = csv.DictReader(stream)
        need(rows.fieldnames is not None and {e150.UID, 'report_group_sha256', 'is_gold'}.issubset(rows.fieldnames),
             'EXP026 group columns missing')
        for row in rows:
            uid, group = row[e150.UID], row['report_group_sha256']
            need(uid and uid not in result and len(group) == 64, 'Invalid report-group metadata')
            result[uid] = group
            if row['is_gold'].lower() == 'true':
                gold_flags.add(uid)
    need(len(gold_flags) == 58, 'EXP026 Gold metadata count changed')
    return result, gold_flags


def build_labels(root=ROOT):
    import numpy as np
    root = Path(root)
    destination = root / GOLD
    need(not destination.exists() and not (root / MANIFEST).exists(), 'Registration already exists')
    protocol = root / PROTOCOL
    need(protocol.is_file(), 'Protocol must exist before building labels')
    base = read(root / e150.BASE)
    e150.registration(root)
    official = root / 'storage/competition/train.csv'
    labels = labels_from_official(official)
    # B hotfix 2026-10-10 (1.1.1): e150.BASE holds only fold 0; the five rosters live in the EXP134 k-fold protocol.
    kfold = read(root / 'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1/s42/manifest.json')
    rosters = [read(root / kfold['folds'][str(i)]['ordered_ids']['path']) for i in range(5)]
    components = read(root / e150.COMPONENTS)
    corpus_dir = root / base['paths']['corpus']
    need(sha(corpus_dir / 'manifest.json') == e150.CORPUS_HASH, 'Corpus manifest changed')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=False).astype(str).tolist()
    group_path = root / 'artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv'
    groups, metadata_gold = report_group_map(group_path)
    need(set(labels) == metadata_gold, 'Official Gold roster differs from EXP026 metadata')
    audit = disjoint_audit(labels, rosters, components, corpus_ids, groups)
    controls = {}
    for surface, seed in SEEDS.items():
        control = root / e150.EXP / 'outputs' / surface / f'EXP150_C{seed}_F0'
        e150.verify_run(control, 'C', seed, surface, root)
        control_prov = read(control / 'provenance.json')
        for name, digest in control_prov['code_sha256'].items():
            need(sha(root / 'training/src' / name) == digest, 'Core recipe code changed: ' + name)
        for name, digest in control_prov['exp127_code_sha256'].items():
            need(sha(root / name) == digest, 'AUG1 recipe code changed: ' + name)
        controls[surface] = dict(oof_sha256=sha(control / 'oof_k36.csv'),
            hardware=control_prov['exp150']['hardware'],
            code_sha256=control_prov['code_sha256'], exp127_code_sha256=control_prov['exp127_code_sha256'],
            environment=read(control / 'summary.json')['environment'])
    em = read(root / e150.MANIFEST)
    need(em['initial_state_sha256'].startswith('ce44133b') and audit['gold_rows'] == 58,
         'Registered initial state or Gold count changed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow([e150.UID, *e150.T])
        for uid in sorted(labels):
            writer.writerow([uid, *labels[uid]])
    write_new(root / AUDIT, dict(schema='exp165-label-audit-v1', **audit,
        official_file_sha256=sha(official), gold_file_sha256=sha(destination),
        corpus_manifest_sha256=e150.CORPUS_HASH, components_sha256=sha(root / e150.COMPONENTS),
        report_group_file_sha256=sha(group_path)))
    m = dict(schema='exp165-gold-train-v1', version='1.1.0', plan='H-B-38',
        protocol_sha256=sha(protocol), official_file_sha256=sha(official),
        gold_file_sha256=sha(destination), audit_sha256=sha(root / AUDIT),
        base_manifest_sha256=sha(root / e150.BASE), exp150_manifest_sha256=sha(root / e150.MANIFEST),
        controls=controls,
        initial_state_file_sha256=em['initial_state_file_sha256'],
        initial_state_sha256=em['initial_state_sha256'], corpus_manifest_sha256=e150.CORPUS_HASH,
        roster_train_sha256=em['roster']['train_sha256'], roster_val_sha256=em['roster']['val_sha256'],
        train_with_gold_sha256=e150.fhash(rosters[0]['train'] + sorted(labels) * 4),
        source_sha256={TOOL: sha(root / TOOL), RUNNER: sha(root / RUNNER)},
        gold_copies=4, fold=0, seeds=SEEDS, train=3687, val=888,
        train_loader=460, scheduler_steps=9200, max_fp16_skips=92,
        bootstrap={'draws': 300, 'seed': 20261003}, wall_seconds=WALL_SECONDS)
    write_new(root / MANIFEST, m)
    return {'status': 'BUILT_NO_EXECUTION', 'gold_rows': 58, 'audit': audit,
            'manifest_sha256': sha(root / MANIFEST)}


def registration(root=ROOT):
    root = Path(root)
    m = read(root / MANIFEST)
    need(m['schema'] == 'exp165-gold-train-v1' and m['version'] == '1.1.0', 'Wrong EXP165 registration')
    for name, key in ((PROTOCOL, 'protocol_sha256'), (GOLD, 'gold_file_sha256'),
                      (AUDIT, 'audit_sha256'), (e150.BASE, 'base_manifest_sha256'),
                      (e150.MANIFEST, 'exp150_manifest_sha256')):
        need(sha(root / name) == m[key], 'Registration hash mismatch: ' + name)
    for name, digest in m['source_sha256'].items():
        need(sha(root / name) == digest, 'Source changed: ' + name)
    for control in m['controls'].values():
        for name, digest in control['code_sha256'].items():
            need(sha(root / 'training/src' / name) == digest, 'Core recipe code changed: ' + name)
        for name, digest in control['exp127_code_sha256'].items():
            need(sha(root / name) == digest, 'AUG1 recipe code changed: ' + name)
    need(sha(root / e150.INITIAL) == m['initial_state_file_sha256'], 'Initial state changed')
    audit = read(root / AUDIT)
    need(audit['gold_rows'] == 58 and audit['corpus_covered'] == 58 and
         audit['uid_overlap'] == audit['component_overlap'] == audit['report_component_overlap'] == 0,
         'Gold audit failed')
    official = root / 'storage/competition/train.csv'
    if official.is_file():
        need(sha(official) == m['official_file_sha256'], 'Official CSV changed since registration')
    group_path = root / 'artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv'
    if group_path.is_file():
        need(sha(group_path) == audit['report_group_file_sha256'], 'Report-group metadata changed')
    return m, read(root / e150.BASE)


def load_gold(path, expected_sha):
    import numpy as np
    need(sha(path) == expected_sha, 'Gold label hash changed')
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [e150.UID, *e150.T], 'Gold label columns changed')
        result = {}
        for row in reader:
            uid = row[e150.UID]
            need(uid and uid not in result, 'Invalid Gold UID')
            values = [row[t] for t in e150.T]
            need(all(x in ('0', '1') for x in values), 'Invalid Gold label value')
            result[uid] = np.asarray([int(x) for x in values], dtype=np.float32)
    need(len(result) == 58, 'Gold count changed')
    return result


def validate_runner_args(a, root=ROOT):
    root = Path(root)
    m, p = registration(root)
    fold = p['folds']['0']
    ordered = read(root / fold['ordered_ids']['path'])
    surface = os.environ.get('EXP165_REMOTE')
    need(a.variant == 'AUG1' and a.fold == 0 and surface in SEEDS and
         a.seed == SEEDS[surface] and a.tag == f'EXP165_G4_S{a.seed}_F0' and
         a.mode == 'full', 'Wrong EXP165 scope')
    need(Path(a.protocol).resolve() == (root / MANIFEST).resolve() and
         Path(a.initial_state).resolve() == (root / e150.INITIAL).resolve() and
         Path(a.gold_labels).resolve() == (root / GOLD).resolve(), 'Wrong EXP165 inputs')
    for key, value in p['common_args'].items():
        need(getattr(a, key) == value, 'Recipe argument changed: ' + key)
    need(a.workers == 2, 'Worker count changed')
    for key in ('corpus', 'labels', 'eval_labels'):
        need(Path(getattr(a, key)).resolve() == (root / p['paths'][key]).resolve(), 'Wrong input: ' + key)
    need(Path(a.init_weights).resolve() == (root / e150.INITIAL).resolve(), 'Wrong init file')
    need(e150.fhash(ordered['train']) == m['roster_train_sha256'] and
         e150.fhash(ordered['val']) == m['roster_val_sha256'], 'Fold roster changed')
    need(sys.platform.startswith('linux') and
         os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Full run only on registered Kaggle')
    out = Path(a.output_root).resolve()
    return {**p, **m, 'paths': p['paths'], 'folds': p['folds'], 'label_set': p['label_set'],
            'common_args': p['common_args'], 'extra_eval': p['extra_eval'],
            'variant_config': p['variant_config'], 'environment': p['environment'],
            'initial_state_sha256': m['initial_state_sha256'], 'init_weights': p['init_weights'],
            'seed': a.seed}, fold, ordered, out


def runner_args(root, surface, output):
    root = Path(root)
    p = read(root / e150.BASE)
    seed = SEEDS[surface]
    args = ['--variant', 'AUG1', '--protocol', str(root / MANIFEST), '--fold', '0',
            '--mode', 'full', '--tag', f'EXP165_G4_S{seed}_F0', '--seed', str(seed),
            '--workers', '2', '--output-root', str(output), '--init-weights', str(root / e150.INITIAL),
            '--initial-state', str(root / e150.INITIAL), '--gold-labels', str(root / GOLD)]
    for key, value in p['common_args'].items():
        if isinstance(value, bool):
            if value:
                args.append('--' + key.replace('_', '-'))
        else:
            args += ['--' + key.replace('_', '-'), str(value)]
    for key in ('corpus', 'labels', 'eval_labels'):
        args += ['--' + key.replace('_', '-'), str(root / p['paths'][key])]
    return args


def package_files(root=ROOT):
    root = Path(root)
    m, p = registration(root)
    f = p['folds']['0']
    names = {MANIFEST, PROTOCOL, GOLD, AUDIT, e150.MANIFEST, e150.PROTOCOL,
             e150.WEIGHTS, e150.INITIAL, e150.CORPUS_LOCAL, e150.BASE, RUNNER, TOOL,
             e150.RUNNER, e150.TOOL, p['label_set']['path'],
             p['paths']['base_ordered_ids'], p['paths']['base_groups'],
             p['paths']['folds'], f['ordered_ids']['path'], f['groups']['path']}
    # Include every transitive project Python module; EXP162 missed raptor_radimagenet.py.
    names |= {x.relative_to(root).as_posix() for folder in ('training/src', 'training/runners', 'tools')
              for x in (root / folder).glob('*.py')}
    files = {}
    for name in sorted(names):
        e150.safe(name)
        source = root / name
        need(source.is_file() and not source.is_symlink(), 'Missing/unsafe package file: ' + name)
        files[name] = dict(sha256=sha(source), bytes=source.stat().st_size)
    need('training/src/raptor_radimagenet.py' in files, 'Missing RadImageNet transitive import')
    need(files[GOLD]['sha256'] == m['gold_file_sha256'] and
         files[e150.INITIAL]['sha256'] == m['initial_state_file_sha256'], 'Package binding mismatch')
    return files


def guard_interface():
    """Static preflight: catch runner/guard API mismatch without importing GPU modules."""
    import ast
    source = (ROOT / RUNNER).read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == 'guard'}
    from tools import exp127_radr50_single as guard
    missing = sorted(n for n in names if not hasattr(guard, n))
    need(not missing, 'Guard API missing: ' + ', '.join(missing))
    return len(names)


def notebook_source(archive_sha, bundle_sha):
    return f'''# EXP165 private Gold training; one T4, no submission or network.
import hashlib, os, pathlib, subprocess, sys, time, zipfile
started = time.time()
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
inp = pathlib.Path('/kaggle/input')
hits = list(inp.rglob('exp165-code.zip.bin'))
assert len(hits) == 1 and hashlib.sha256(hits[0].read_bytes()).hexdigest() == {archive_sha!r}
ws = pathlib.Path('/tmp/exp165-workspace')
assert not ws.exists()
with zipfile.ZipFile(hits[0]) as z:
    names = z.namelist(); assert len(names) == len(set(names))
    for name in names:
        p = pathlib.PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
    z.extractall(ws)
env = dict(os.environ, KNEE_WORKSPACE_ROOT=str(ws), EXP165_REMOTE='kaggle',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
for key in {THREADS!r}: env[key] = '1'
cmd = [sys.executable, str(ws / {TOOL!r}), 'remote-run', '--surface', 'kaggle', '--bundle-sha', {bundle_sha!r}, '--started', str(started)]
p = subprocess.run(cmd, cwd=ws, env=env, timeout={WALL_SECONDS['kaggle'] + 120})
assert p.returncode == 0, 'EXP165 incomplete; inspect exp165-status.json and launcher logs'
print(pathlib.Path('/kaggle/working/exp165-status.json').read_text())
'''


def stage(out, root=ROOT, surface='kaggle'):
    root, out = Path(root), Path(out)
    need(surface in SEEDS and not out.exists(), 'Use a new stage directory')
    guard_interface()
    files = package_files(root)
    receipt = dict(schema='exp165-package-v1', surface=surface, files=dict(files), manifest_sha256=sha(root / MANIFEST),
                   protocol_sha256=sha(root / PROTOCOL), no_execution=True)
    out.mkdir(parents=True)
    code = out / ('code-dataset' if surface == 'kaggle' else 'package')
    code.mkdir()
    marker = b'# EXP165 isolated private workspace.\n'
    receipt['files']['config/workspace.toml'] = dict(sha256=hashlib.sha256(marker).hexdigest(), bytes=len(marker))
    blob = (json.dumps(receipt, indent=2, ensure_ascii=False) + '\n').encode()
    archive = code / 'exp165-code.zip.bin'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.write(root / name, name)
        z.writestr('config/workspace.toml', marker)
        z.writestr(RECEIPT, blob)
    (out / RECEIPT).write_bytes(blob)
    write_new(out / 'stage-receipt.json', dict(schema='exp165-stage-v1', surface=surface, archive_sha256=sha(archive),
        bundle_sha256=hashlib.sha256(blob).hexdigest(), manifest_sha256=sha(root / MANIFEST)))
    if surface == 'kaggle':
        write_new(code / 'dataset-metadata.json', dict(id=DATASET, title='EXP165 Gold training private package',
        isPrivate=True, licenses=[{'name': 'other'}],
        description='Private competition experiment package; no external raw data.'))
        source = notebook_source(sha(archive), hashlib.sha256(blob).hexdigest())
        compile(source, '<exp165-notebook>', 'exec')
        write_new(out / 'run.ipynb', dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        cells=[dict(cell_type='code', id='exp165', metadata={}, execution_count=None,
                    outputs=[], source=source.splitlines(keepends=True))]))
        write_new(out / 'kernel-metadata.json', dict(id=KERNEL, title='EXP165 Gold train F0 S42',
        code_file='run.ipynb', language='python', kernel_type='notebook', is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=False, machine_shape='NvidiaTeslaT4',
        docker_image=e150.DOCKER,
        dataset_sources=['taeyangg4/knee-raptor-corpus-336', DATASET],
        competition_sources=['rsna-knee-abnormality-detection'], kernel_sources=[], model_sources=[]))
    else:
        write_new(out / 'vast-queue.json', [dict(tag='EXP165_G4_S43_F0', command=[
            'python', TOOL, 'remote-run', '--surface', 'vast', '--bundle-sha',
            hashlib.sha256(blob).hexdigest(), '--started', '0'])])
    verify_stage(out, root)
    return read(out / 'stage-receipt.json')


def verify_stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    record = read(out / 'stage-receipt.json')
    surface = record['surface']
    archive = out / ('code-dataset' if surface == 'kaggle' else 'package') / 'exp165-code.zip.bin'
    need(sha(archive) == record['archive_sha256'] and sha(out / RECEIPT) == record['bundle_sha256'],
         'Stage hash mismatch')
    bundle = read(out / RECEIPT)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == {*bundle['files'], RECEIPT},
             'Archive entries changed or duplicate')
        for name, rec in bundle['files'].items():
            e150.safe(name)
            data = z.read(name)
            need(len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256'],
                 'Archive file mismatch: ' + name)
            if name != 'config/workspace.toml':
                need(sha(root / name) == rec['sha256'], 'Local file changed since stage: ' + name)
    if surface == 'kaggle':
        meta = read(out / 'kernel-metadata.json')
        need(meta['is_private'] and meta['machine_shape'] == 'NvidiaTeslaT4' and
             DATASET in meta['dataset_sources'], 'Unsafe Kaggle kernel metadata')
        need(meta['id'].split('/')[-1] == '-'.join(meta['title'].lower().split()),
             'Kernel slug must follow title')
    else:
        need(read(out / 'vast-queue.json') == [dict(tag='EXP165_G4_S43_F0', command=[
            'python', TOOL, 'remote-run', '--surface', 'vast', '--bundle-sha',
            record['bundle_sha256'], '--started', '0'])], 'Vast queue drift')
    return record


def push_plan(out):
    need(verify_stage(out)['surface'] == 'kaggle', 'Wrong Kaggle stage')
    out = Path(out).as_posix()
    return '\n'.join([
        '# B: after 09:00 KST quota reset, check quota/slot, claim Kaggle compute, then execute in order.',
        "$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'",
        "$kagglePython = (Resolve-Path '.venv/Scripts/python.exe').Path",
        f'Push-Location {out}/code-dataset',
        '& $kagglePython -X utf8 -m kaggle datasets create -p . --dir-mode zip',
        '# If the dataset already exists, use: & $kagglePython -X utf8 -m kaggle datasets version -p . -m "EXP165 v1.1.0" --dir-mode zip',
        'Pop-Location',
        f'Push-Location {out}',
        '& $kagglePython -X utf8 -m kaggle kernels push -p .',
        'Pop-Location',
        f'& $kagglePython -X utf8 -m kaggle kernels output {KERNEL} -p {out}/download --page-size 200',
        f'& $py tools/exp165_gold_train.py collect --surface kaggle --source {out}/download',
        '& $py tools/exp165_gold_train.py screen',
    ])


def vast_plan(out, host, port, known_hosts, user='root'):
    need(verify_stage(out)['surface'] == 'vast', 'Wrong Vast stage')
    need(host and all(c.isalnum() or c in '.-' for c in host) and
         isinstance(port, int) and 1 <= port <= 65535 and user == 'root', 'Live Vast endpoint required')
    need(Path(known_hosts).is_file(), 'Fresh verified known_hosts file required')
    out = Path(out).as_posix()
    key = 'C:/Users/Taeyang/.ssh/id_ed25519'
    scp_opts = f'-i {key} -o UserKnownHostsFile={known_hosts} -o StrictHostKeyChecking=yes -P {port}'
    ssh_opts = f'-i {key} -o UserKnownHostsFile={known_hosts} -o StrictHostKeyChecking=yes -p {port}'
    remote = f'{user}@{host}'
    return '\n'.join([
        '# After EXP164 ends: verify live instance, host key and idle 3090, then claim Active compute.',
        f'scp {scp_opts} {out}/package/exp165-code.zip.bin {remote}:/workspace/exp165-code.zip.bin',
        f'ssh {ssh_opts} {remote} "mkdir -p /workspace/exp165 && python -m zipfile -e /workspace/exp165-code.zip.bin /workspace/exp165"',
        f'scp {scp_opts} {out}/vast-queue.json {remote}:/workspace/exp165/vast-queue.json',
        f"ssh {ssh_opts} {remote} 'cd /workspace/exp165 && EXP165_REMOTE=vast KNEE_WORKSPACE_ROOT=/workspace/exp165 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 setsid nohup python tools/gpu_queue.py vast-queue.json --foreground > exp165-queue.log 2>&1 < /dev/null & echo $! > /workspace/exp165/exp165-queue.pid'",
        f'scp {scp_opts} -r {remote}:/workspace/exp165/results {out}/download',
        f'& C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe tools/exp165_gold_train.py collect --surface vast --source {out}/download',
        '& C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe tools/exp165_gold_train.py screen',
    ])


def verify_bundle(root, bundle_sha):
    root = Path(root)
    need(sha(root / RECEIPT) == bundle_sha, 'Bundle hash mismatch')
    bundle = read(root / RECEIPT)
    need(bundle['schema'] == 'exp165-package-v1', 'Wrong bundle schema')
    for name, rec in bundle['files'].items():
        e150.safe(name)
        p = root / name
        need(p.is_file() and not p.is_symlink() and p.stat().st_size == rec['bytes'] and
             sha(p) == rec['sha256'], 'Bundle file changed: ' + name)
    registration(root)
    return bundle


def remote_run(surface, bundle_sha, started, root=ROOT):
    root = Path(root)
    need(sys.platform.startswith('linux') and surface in SEEDS and
         os.environ.get('EXP165_REMOTE') == surface and
         os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Wrong remote surface')
    if surface == 'kaggle':
        need(Path('/kaggle/working').is_dir(), 'Kaggle mount missing')
    else:
        need(root == Path('/workspace/exp165'), 'Wrong Vast root')
    output = Path('/kaggle/working') if surface == 'kaggle' else root / 'results'
    output.mkdir(parents=True, exist_ok=True)
    need(not (output / 'exp165-status.json').exists(), 'Previous supervisor output exists')
    bundle = verify_bundle(root, bundle_sha)
    need(bundle['surface'] == surface, 'Wrong surface bundle')
    m, p = registration(root)
    started = started or time.time()
    deadline = started + m['wall_seconds'][surface]
    status = dict(schema='exp165-run-v1', state='PREFLIGHT', started=started,
                  surface=surface, bundle_sha256=bundle_sha)
    write_new(output / 'exp165-status.json', status)
    procs = []
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True,
                             check=True, timeout=30).stdout.strip().splitlines()
        need((len(smi) in (1, 2) if surface == 'kaggle' else len(smi) == 1) and
             all(('T4' if surface == 'kaggle' else '3090') in x for x in smi), 'Wrong GPU type/count')
        import torch, timm, numpy
        need(not torch.cuda.is_initialized(), 'Supervisor touched CUDA')
        status['software'] = dict(python=sys.version.split()[0], torch=torch.__version__,
                                  timm=timm.__version__, numpy=numpy.__version__, cuda=torch.version.cuda)
        baseline = m['controls'][surface]
        status['control_software'] = baseline['hardware']
        status['software_drift'] = {k: dict(control=(baseline['environment']['python'] if k == 'python' else baseline['hardware'][k]),
                                             actual=status['software'][k])
                                    for k in ('python', 'torch', 'timm', 'numpy', 'cuda')
                                    if status['software'][k] != (baseline['environment']['python'] if k == 'python' else baseline['hardware'][k])}
        driver = smi[0].split(',')[-1].strip()
        if driver != baseline['hardware']['driver']:
            status['software_drift']['driver'] = dict(control=baseline['hardware']['driver'], actual=driver)
        status['image_changed'] = bool(status['software_drift'])
        status['gpu'] = smi
        if surface == 'kaggle':
            corpus = Path('/kaggle/input/datasets/taeyangg4/knee-raptor-corpus-336')
            if not (corpus / 'manifest.json').exists():
                hits = [x.parent for x in Path('/kaggle/input').rglob('manifest.json')
                        if sha(x) == e150.CORPUS_HASH]
                need(len(hits) == 1, 'Corpus mount missing or ambiguous')
                corpus = hits[0]
        else:
            corpus = e150.locate_corpus('vast')
        em = read(root / e150.MANIFEST)
        status['corpus'] = e150.verify_corpus(corpus, root / e150.CORPUS_LOCAL,
                                               em['corpus_file_list'])
        import numpy as np
        corpus_ids = set(np.load(corpus / 'ids.npy', allow_pickle=False).astype(str).tolist())
        gold_ids = set(load_gold(root / GOLD, m['gold_file_sha256']))
        need(gold_ids.issubset(corpus_ids), 'Mounted corpus lacks Gold58')
        destination = root / p['paths']['corpus']
        destination.parent.mkdir(parents=True, exist_ok=True)
        need(not destination.exists(), 'Corpus path already exists')
        if surface == 'kaggle':
            destination.symlink_to(corpus, target_is_directory=True)
        else:
            destination.mkdir()
            for source in corpus.rglob('*'):
                relative = source.relative_to(corpus)
                target = destination / relative
                if source.is_dir():
                    target.mkdir(exist_ok=True)
                else:
                    need(source.is_file() and not source.is_symlink(), 'Unsafe corpus entry')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, target)
                    need(source.stat().st_ino == target.stat().st_ino and
                         source.stat().st_dev == target.stat().st_dev, 'Corpus hardlink failed')
            need(sha(destination / 'manifest.json') == e150.CORPUS_HASH, 'Hardlinked corpus changed')
            prior = Path('/workspace/exp150') / e150.INITIAL
            current = root / e150.INITIAL
            if prior.is_file() and sha(prior) == m['initial_state_file_sha256']:
                current.unlink()
                os.link(prior, current)
                need(current.stat().st_ino == prior.stat().st_ino, 'Initial state hardlink failed')
                status['exp150_package_reused'] = True
            else:
                status['exp150_package_reused'] = False
        shutil.copyfile(root / RECEIPT, output / RECEIPT)
        for gpu in ('0',):
            seed = SEEDS[surface]
            tag = f'EXP165_G4_S{seed}_F0'
            need(not (output / tag).exists(), 'Run directory already exists')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       KNEE_REMOTE_RELAX_ENV='1', KNEE_WORKSPACE_ROOT=str(root),
                       EXP165_REMOTE=surface, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       HF_HUB_OFFLINE='1', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
            for key in THREADS:
                env[key] = '1'
            log = (output / f'{tag}.launcher.log').open('x', encoding='utf-8')
            child = subprocess.Popen([sys.executable, str(root / RUNNER)] + runner_args(root, surface, output),
                                     cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            procs.append((child, log, seed, gpu))
        codes = []
        for child, log, seed, gpu in procs:
            try:
                codes.append(child.wait(timeout=max(1, deadline - time.time())))
            finally:
                log.close()
        need(codes == [0], 'Gold run failed; check launcher log')
        for _, _, seed, gpu in procs:
            run = output / f'EXP165_G4_S{seed}_F0'
            prov, summary = read(run / 'provenance.json'), read(run / 'summary.json')
            hardware = dict(physical_gpu=smi[int(gpu)], visible_device=gpu,
                            driver=smi[int(gpu)].split(',')[-1].strip(), **status['software'])
            prov['exp165'].update(surface=surface, seed=seed, hardware=hardware,
                                   software_drift=status['software_drift'], image_changed=status['image_changed'],
                                   bundle_sha256=bundle_sha, manifest_sha256=sha(root / MANIFEST))
            summary['provenance'] = prov
            (run / 'provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf-8')
            (run / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            write_new(run / 'exp165-run.json', dict(schema='exp165-run-v1', state='COMPLETE', surface=surface,
                seed=seed, hardware=hardware, bundle_sha256=bundle_sha, oof_k36_sha256=sha(run / 'oof_k36.csv')))
            names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                     'oof_best.csv', 'oof_k36.csv', 'exp165-run.json']
            # B hotfix 2026-10-10 (A2): the derived runner already writes a six-entry inventory before this
            # supervisor rewrites summary/provenance and adds exp165-run.json, so replace it (Vast S43 failed here).
            (run / 'output-inventory.json').write_text(json.dumps(
                {n: dict(sha256=sha(run / n), bytes=(run / n).stat().st_size) for n in names}, indent=1), encoding='utf-8')
        status.update(state='COMPLETE', tags=[f'EXP165_G4_S{SEEDS[surface]}_F0'])
    except BaseException as exc:
        for child, log, _, _ in procs:
            if child.poll() is None:
                child.terminate()
            if not log.closed:
                log.close()
        status.update(state='FAILED_OR_INCOMPLETE', reason=type(exc).__name__)
        raise
    finally:
        status['elapsed_seconds'] = time.time() - started
        (output / 'exp165-status.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')


def verify_run(run, surface, root=ROOT):
    root, run = Path(root), Path(run)
    m, p = registration(root)
    seed = SEEDS[surface]
    need(run.name == f'EXP165_G4_S{seed}_F0', 'Wrong run tag')
    inventory = read(run / 'output-inventory.json')
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                'oof_best.csv', 'oof_k36.csv', 'exp165-run.json'}
    need(set(inventory) == required, 'Incomplete output inventory')
    for name, rec in inventory.items():
        need(Path(name).name == name and sha(run / name) == rec['sha256'] and
             (run / name).stat().st_size == rec['bytes'], 'Output file mismatch')
    summary, prov, receipt = (read(run / n) for n in ('summary.json', 'provenance.json', 'exp165-run.json'))
    need(receipt['state'] == 'COMPLETE' and receipt['surface'] == surface and receipt['seed'] == seed and
         receipt['oof_k36_sha256'] == sha(run / 'oof_k36.csv'), 'Incomplete arm receipt')
    need(summary['provenance'] == prov and prov['exp165']['gold_copies'] == 4 and prov['exp165']['seed'] == seed and
         prov['initial_state_sha256'] == m['initial_state_sha256'] and
         prov['train_ids_ordered_sha256'] == m['train_with_gold_sha256'] and
         prov['eval_ids_ordered_sha256'] == m['roster_val_sha256'] and
         prov['actual_pos_weight'] == p['folds']['0']['pos_weight'] and
         summary['train'] == m['train'] and summary['val'] == 888 and summary['gold'] == 58 and
         summary['optimizer_updates'] == m['scheduler_steps'] and summary['skipped_updates'] <= m['max_fp16_skips'] and
         summary['selected_epoch_1based'] == 20, 'Arm provenance/recipe mismatch')
    need(len(read(run / 'history.json')) == 20 and
         ('T4' if surface == 'kaggle' else '3090') in receipt['hardware']['physical_gpu'],
         'Epoch or GPU mismatch')
    ids = read(root / p['folds']['0']['ordered_ids']['path'])['val']
    with (run / 'oof_k36.csv').open(newline='', encoding='utf-8') as stream:
        rows = csv.DictReader(stream)
        need(rows.fieldnames == [e150.UID, *e150.T], 'OOF columns changed')
        seen = []
        for row in rows:
            seen.append(row[e150.UID])
            need(all(0 <= float(row[t]) <= 1 for t in e150.T), 'Invalid OOF probability')
    need(seen == ids, 'OOF order/membership changed')
    return {n: rec['sha256'] for n, rec in inventory.items()}


def collect(source, surface='kaggle', root=ROOT):
    root, source = Path(root), Path(source)
    seed = SEEDS[surface]
    need(read(source / 'exp165-status.json')['state'] == 'COMPLETE', 'Remote supervisor incomplete')
    verify_run(source / f'EXP165_G4_S{seed}_F0', surface, root)
    dest = root / EXP / 'outputs' / surface
    need(not dest.exists(), 'Collected output already exists')
    shutil.copytree(source, dest)
    verify_run(dest / f'EXP165_G4_S{seed}_F0', surface, root)
    return {'status': 'COLLECTED_VALIDATED', 'destination': str(dest)}


def paired_mean_draws(candidate, control, refs, draws):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    deltas = []
    for ix in draws:
        pair = []
        for y, mask, mc in refs:
            a = ev.macro(ev.auc_targets(y[ix], candidate[ix], mask[ix], mc))
            b = ev.macro(ev.auc_targets(y[ix], control[ix], mask[ix], mc))
            pair.append(a - b)
        deltas.append(float(np.mean(pair)))
    need(np.isfinite(deltas).all(), 'Nonfinite primary bootstrap draw')
    return np.asarray(deltas)


def paired_mean_ci(candidate, control, refs, draws):
    import numpy as np
    return np.percentile(paired_mean_draws(candidate, control, refs, draws), [2.5, 97.5]).tolist()


def harm_decision(seed_results):
    need(set(seed_results) == set(SEEDS), 'Both registered seeds required')
    harmed = sum(body['primary_mean_delta'] < -.003 or body['metrics']['dread']['delta'] < -.003
                 for body in seed_results.values())
    return 'ADOPT_GOLD_X4' if harmed == 0 else ('USE_GOLD_X1_NO_RESCREEN' if harmed == 1 else 'EXCLUDE_GOLD')


def screen(root=ROOT):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp125_radr50_k16_local as e125
    root = Path(root)
    m, p = registration(root)
    ids = read(root / p['folds']['0']['ordered_ids']['path'])['val']
    need(len(ids) == 888, 'Wrong V888')
    em = read(root / e150.MANIFEST)
    for name, digest in em['readout_sha256'].items():
        need(sha(root / name) == digest, 'Registered readout changed: ' + name)
    draws = e125.component_draws(ids, read(root / e150.COMPONENTS), draws=300, seed=20261003)
    refs = {}
    y = e150.table(root / e150.JEV, ids)
    need(np.isfinite(y).all() and np.isin(y, [0, 1]).all(), 'Invalid JEV reference')
    refs['jev_hybrid'] = (y, np.ones(y.shape, bool), 1)
    y = e150.table(root / e150.AQUILA, ids)
    need(np.isfinite(y).all(), 'Invalid aquila reference')
    refs['aquila_v1'] = ((y >= .5).astype(float), np.ones(y.shape, bool), 1)
    y = e150.table(root / e150.SETS['dread'], ids)
    need(np.isfinite(y).all(), 'Invalid Dread reference')
    refs['dread'] = (np.rint(y.astype(np.float32)).astype(float), np.ones(y.shape, bool), 1)
    stack = np.stack([e150.table(root / path, ids) for path in e150.SETS.values()])
    valid = np.isfinite(stack).all(axis=0)
    binary = stack >= .5
    agree = valid & (binary.all(axis=0) | (~binary).all(axis=0))
    refs['consensus'] = (binary[0].astype(float), agree, 2)
    results, predictions = {}, {}
    for surface, seed in SEEDS.items():
        control = root / e150.EXP / 'outputs' / surface / f'EXP150_C{seed}_F0'
        run = root / EXP / 'outputs' / surface / f'EXP165_G4_S{seed}_F0'
        e150.verify_run(control, 'C', seed, surface, root)
        need(sha(control / 'oof_k36.csv') == m['controls'][surface]['oof_sha256'], 'Control prediction changed')
        verify_run(run, surface, root)
        c = e125.load_table(control / 'oof_k36.csv', ids, prediction=True)
        a = e125.load_table(run / 'oof_k36.csv', ids, prediction=True)
        predictions[surface] = (a, c)
        metrics = {name: ev.compare(a, c, ref, draws) for name, ref in refs.items()}
        for body in metrics.values():
            need(np.isfinite(body['delta']) and np.isfinite(body['ci95']).all(), 'Nonfinite delta')
            body['targets'] = {k: float(v) if np.isfinite(v) else None for k, v in body['targets'].items()}
        mean = (metrics['jev_hybrid']['delta'] + metrics['aquila_v1']['delta']) / 2
        ci = paired_mean_ci(a, c, (refs['jev_hybrid'], refs['aquila_v1']), draws)
        harm = mean < -.003 or metrics['dread']['delta'] < -.003
        candidate_prov = read(run / 'provenance.json')
        control_prov = read(control / 'provenance.json')
        core_code_drift = {name: dict(control=digest, actual=candidate_prov['code_sha256'].get(name))
                           for name, digest in control_prov['code_sha256'].items()
                           if candidate_prov['code_sha256'].get(name) != digest}
        results[surface] = dict(seed=seed, metrics=metrics, primary_mean_delta=mean, primary_ci95=ci,
                            harm=harm, software_drift=candidate_prov['exp165']['software_drift'],
                            core_code_drift=core_code_drift,
                            prediction_sha256=sha(run / 'oof_k36.csv'))
    decision = harm_decision(results)
    pooled = {}
    for name, ref in refs.items():
        bodies = [results[s]['metrics'][name] for s in SEEDS]
        draw_values = [paired_mean_draws(*predictions[s], (ref,), draws) for s in SEEDS]
        target_names = bodies[0]['targets']
        pooled[name] = dict(delta=float(np.mean([b['delta'] for b in bodies])),
                            ci95=np.percentile(np.mean(draw_values, axis=0), [2.5, 97.5]).tolist(),
                            targets={t: (float(np.mean([b['targets'][t] for b in bodies]))
                                         if all(b['targets'][t] is not None for b in bodies) else None)
                                     for t in target_names})
    primary_draws = [paired_mean_draws(*predictions[s],
                                      (refs['jev_hybrid'], refs['aquila_v1']), draws) for s in SEEDS]
    primary_pooled = dict(delta=float(np.mean([results[s]['primary_mean_delta'] for s in SEEDS])),
                          ci95=np.percentile(np.mean(primary_draws, axis=0), [2.5, 97.5]).tolist())
    primary_pooled['non_inferior'] = bool(primary_pooled['ci95'][0] > -.003)
    out = root / EXP / 'screen'
    need(not out.exists(), 'Screen already exists')
    out.mkdir()
    result = dict(schema='exp165-screen-v1.1', gate='two-seed harm check at -0.003',
                  gold58_metric='FORBIDDEN_TRAINING_EXPOSURE', rows=888,
                  bootstrap=m['bootstrap'], decision=decision, benefit='INCONCLUSIVE',
                  pooled_primary=primary_pooled, pooled_metrics=pooled, seeds=results)
    write_new(out / 'result.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('build-labels')
    st = sub.add_parser('stage'); st.add_argument('--out', required=True)
    vs = sub.add_parser('vast-stage'); vs.add_argument('--out', required=True)
    pp = sub.add_parser('push-plan'); pp.add_argument('--out', required=True)
    vp = sub.add_parser('vast-plan'); vp.add_argument('--out', required=True); vp.add_argument('--host', required=True); vp.add_argument('--port', type=int, required=True); vp.add_argument('--known-hosts', required=True)
    rr = sub.add_parser('remote-run'); rr.add_argument('--surface', required=True, choices=['kaggle', 'vast']); rr.add_argument('--bundle-sha', required=True); rr.add_argument('--started', type=float, required=True)
    co = sub.add_parser('collect'); co.add_argument('--source', required=True); co.add_argument('--surface', required=True, choices=['kaggle', 'vast'])
    sub.add_parser('screen')
    a = ap.parse_args(argv)
    if a.command == 'build-labels':
        value = build_labels()
    elif a.command == 'stage':
        value = stage(a.out)
    elif a.command == 'vast-stage':
        value = stage(a.out, surface='vast')
    elif a.command == 'push-plan':
        value = push_plan(a.out)
    elif a.command == 'vast-plan':
        value = vast_plan(a.out, a.host, a.port, a.known_hosts)
    elif a.command == 'remote-run':
        value = remote_run(a.surface, a.bundle_sha, a.started)
    elif a.command == 'collect':
        value = collect(a.source, a.surface)
    else:
        value = screen()
    print(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
