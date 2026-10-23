"""EXP173 v1.2.0 A1: matched-label W4 and exploratory Gold58 cross-fit.

Local commands are CPU-only. No command pushes a dataset or starts a kernel.
Changelog 1.2.0 (2026-10-12): elapsed/projection values are telemetry; no self-imposed run deadline.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp165_gold_train as e165
from tools import exp150_consensus_weight as e150

VERSION = '1.2.0'
EXP = 'artifacts/experiments/EXP173_GOLD_CROSSFIT'
PROTOCOL = f'{EXP}/PROTOCOL.md'
SPLIT = f'{EXP}/split.json'
TOOL = 'tools/exp173_gold_crossfit.py'
CONTROL = 'artifacts/experiments/EXP150_CONSENSUS_WEIGHT/outputs/kaggle/EXP150_C42_F0/model_best.pt'
ASSET = 'assets/exp150-control/model_best.pt'
BASE_ARCHIVE = '.local/exp165-v110-kaggle/code-dataset/exp165-code.zip.bin'
BASE_SHA = '8c4c0ae6c4bd5d6c2183fb25919118d5b87d9aabf69a52ccfcdc0a1ec88c0c57'
VAST_BASE_ARCHIVE = '.local/exp165-v110-vast/package/exp165-code.zip.bin'
VAST_BASE_SHA = '9bffbfdb495d96d29a2b53970108f0b05aa4fce0c67cac70e1538fe2151fb7b6'
DATASET = 'taeyangg4/knee-exp173-gold-crossfit-private'
ARMS = ('W4-S42', 'G4-H1', 'J4-H1', 'G4-H2', 'J4-H2')
CROSS_ARMS = ARMS[1:]
VAST_ARM = 'W4-S43'
ALL_ARMS = ARMS + (VAST_ARM,)
BOOT_DRAWS = 2000
BOOT_SEED = 20261003


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return e150.sha(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    e150.write_new(path, data)


def arm_parts(arm):
    need(arm in CROSS_ARMS, 'Unknown cross-fit arm')
    return arm[0], int(arm[-1]), 3 - int(arm[-1])


def load_split(root=ROOT):
    root = Path(root)
    obj = read(root / SPLIT)
    need(obj['schema'] == 'exp173-split-v1.1' and obj['seed'] == 20261010 and obj['searches'] == 1000 and
         obj['gold_sha256'] == sha(root / e165.GOLD), 'Split registration changed')
    audit = obj['duplicate_audit']
    need(audit['report_groups'] == 58 and audit['gold_series_studies'] == 58 and
         audit['shared_series_uids'] == 0 and audit['gold_graph_components'] == 0 and
         audit['patient_image_near_duplicate_coverage'] == 'unavailable', 'Duplicate audit registration changed')
    gold = e165.load_gold(root / e165.GOLD, obj['gold_sha256'])
    halves = obj['halves']
    need(set(halves) == {'1', '2'} and all(len(halves[h]) == 29 and halves[h] == sorted(halves[h])
         for h in halves) and set(halves['1']).isdisjoint(halves['2']) and
         set(halves['1'] + halves['2']) == set(gold), 'Split roster invalid')
    import numpy as np
    for ids in halves.values():
        y = np.stack([gold[u] for u in ids])
        need(((y.sum(0) > 0) & (y.sum(0) < 29)).all(), 'A half lacks target classes')
    expected, score = best_half(gold, searches=1000)
    need(halves['1'] == expected and halves['2'] == sorted(set(gold) - set(expected)) and
         obj['imbalance'] == score, 'Split differs from registered deterministic search')
    return obj


def audit_gold_duplicates(root=ROOT):
    """Metadata-only Gold/Gold audit; never reads a radiology report or label rows."""
    root = Path(root)
    gold = set(e165.load_gold(root / e165.GOLD, sha(root / e165.GOLD)))
    groups, marked = e165.report_group_map(root / 'artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv')
    need(gold == marked and len({groups[u] for u in gold}) == 58, 'Gold report group collision')
    components = read(root / e150.COMPONENTS)
    touched = [c for c in components if gold.intersection(c['members'])]
    need(not touched, 'Gold unexpectedly present in image-near-duplicate graph')
    series = root / 'storage/competition/train_series.csv'
    owners, covered = {}, set()
    with series.open(newline='', encoding='utf-8') as stream:
        rows = csv.DictReader(stream)
        need(rows.fieldnames is not None and
             {e150.UID, 'SeriesInstanceUID'}.issubset(rows.fieldnames), 'Series metadata columns missing')
        for row in rows:
            uid = row[e150.UID]
            if uid not in gold:
                continue
            sid = row['SeriesInstanceUID']
            need(sid and (sid not in owners or owners[sid] == uid), 'Gold studies share a series UID')
            owners[sid] = uid
            covered.add(uid)
    need(covered == gold, 'Gold series metadata incomplete')
    return dict(report_groups=58, gold_series_studies=58, gold_series_uids=len(owners),
                shared_series_uids=0, gold_graph_components=0,
                patient_image_near_duplicate_coverage='unavailable',
                limitation='EXP026 checks report equality; Gold58 is outside the existing image-near-duplicate graph, and available series metadata has no patient identifier.')


def best_half(gold, searches=1000):
    import numpy as np
    ids = sorted(gold)
    y = np.stack([gold[u] for u in ids]).astype(int)
    best, score = None, None
    for attempt in range(searches):
        indices = np.asarray(sorted(range(58), key=lambda i: hashlib.sha256(
            f'20261010:{attempt}:{ids[i]}'.encode('utf-8')).hexdigest()), dtype=int)
        left = y[indices[:29]].sum(0)
        right = y.sum(0) - left
        if ((left > 0) & (left < 29) & (right > 0) & (right < 29)).all():
            candidate_score = int(np.abs(left - right).sum())
            if score is None or candidate_score < score:
                best, score = indices[:29].copy(), candidate_score
    need(best is not None, 'No stratified split found')
    return sorted(ids[i] for i in best), score


def split(root=ROOT, searches=1000):
    """Deterministic pre-training stratification; returns only counts and path."""
    import numpy as np
    root = Path(root)
    need(searches == 1000, 'Split search count is frozen at 1000')
    need((root / PROTOCOL).is_file() and not (root / SPLIT).exists(), 'Protocol first; split immutable')
    e165.registration(root)
    gold = e165.load_gold(root / e165.GOLD, sha(root / e165.GOLD))
    half1, score = best_half(gold, searches)
    half2 = sorted(set(gold) - set(half1))
    audit = audit_gold_duplicates(root)
    write_new(root / SPLIT, dict(schema='exp173-split-v1.1', seed=20261010,
         searches=searches, criterion='all targets both classes per half; minimum L1 positive imbalance; first tie',
         imbalance=score, gold_sha256=sha(root / e165.GOLD), duplicate_audit=audit,
         halves={'1': half1, '2': half2}))
    load_split(root)
    return {'status': 'SPLIT_REGISTERED', 'half_sizes': [29, 29], 'imbalance': score,
            'split_sha256': sha(root / SPLIT)}


def guard_interface(root=ROOT):
    """Audit every guard.<attribute> used by the frozen runner, before packaging."""
    import ast
    from tools import exp127_radr50_single as guard
    source = (Path(root) / e165.RUNNER).read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == 'guard'}
    missing = sorted(n for n in names if not hasattr(guard, n))
    need(not missing, 'Frozen guard API missing: ' + ', '.join(missing))
    return len(names)


def verify_base(root=ROOT, surface='kaggle'):
    root = Path(root)
    archive = BASE_ARCHIVE if surface == 'kaggle' else VAST_BASE_ARCHIVE
    if surface == 'kaggle':
        need(sha(root / archive) == BASE_SHA, 'EXP165 Kaggle archive changed')
    else:
        need(sha(root / archive) == VAST_BASE_SHA, 'EXP165 Vast archive changed')
        stage = read(root / '.local/exp165-v110-vast/stage-receipt.json')
        need(stage['surface'] == 'vast' and stage['archive_sha256'] == VAST_BASE_SHA,
             'EXP165 Vast stage receipt changed')
    with zipfile.ZipFile(root / archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and TOOL not in names and SPLIT not in names,
             'Base archive overlaps EXP173 overlay')
        need('config/workspace.toml' in names, 'Base workspace config absent')
        if surface == 'vast':
            bundle = json.loads(z.read(e165.RECEIPT))
            need(bundle['surface'] == 'vast' and
                 hashlib.sha256(z.read(e165.RECEIPT)).hexdigest() == stage['bundle_sha256'] and
                 set(names) == {*bundle['files'], e165.RECEIPT}, 'EXP165 Vast bundle receipt changed')
            for name, rec in bundle['files'].items():
                data = z.read(name)
                need(len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256'],
                     'EXP165 Vast archive file changed: ' + name)
            old = json.loads(z.read(e165.MANIFEST))
            current = read(root / e165.MANIFEST)
            for key in ('gold_file_sha256', 'initial_state_sha256', 'roster_train_sha256',
                        'roster_val_sha256', 'train_with_gold_sha256', 'scheduler_steps', 'controls'):
                need(old[key] == current[key], 'Vast frozen recipe/control drift: ' + key)
    e165.registration(root)
    return sha(root / archive)


def package_files(root=ROOT):
    root = Path(root)
    load_split(root)
    need(audit_gold_duplicates(root) == read(root / SPLIT)['duplicate_audit'], 'Gold duplicate audit drift')
    guard_interface(root)
    verify_base(root)
    e150.verify_run(root / Path(CONTROL).parent, 'C', 42, 'kaggle', root)
    for arm in ('W4-S42', 'J4-H1', 'J4-H2'):
        need(len(arm_csv_bytes(root, arm)) > 100, 'JEV Gold roster not covered')
    names = (TOOL, PROTOCOL, SPLIT, CONTROL, e150.JEV)
    return {name: {'sha256': sha(root / name), 'bytes': (root / name).stat().st_size}
            for name in names}


def notebook_source(arm, overlay_sha):
    return f'''# EXP173 {arm}: private T4, no network or submission.
import hashlib, os, pathlib, subprocess, sys, time, zipfile
started = time.time()
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
inp = pathlib.Path('/kaggle/input')
def unique(name, digest):
    hits = [p for p in inp.rglob(name) if hashlib.sha256(p.read_bytes()).hexdigest() == digest]
    assert len(hits) == 1, 'Missing/ambiguous pinned package: ' + name
    return hits[0]
base = unique('exp165-code.zip.bin', {BASE_SHA!r})
overlay = unique('exp173-overlay.zip.bin', {overlay_sha!r})
ws = pathlib.Path('/tmp/exp173-workspace')
assert not ws.exists()
ws.mkdir()
for archive in (base, overlay):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist(); assert len(names) == len(set(names))
        for name in names:
            p = pathlib.PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
            assert not (ws / name).exists(), 'Archive collision'
        z.extractall(ws)
env = dict(os.environ, KNEE_WORKSPACE_ROOT=str(ws), EXP165_REMOTE='kaggle',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', KNEE_REMOTE_RELAX_ENV='1',
           PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '1'
cmd = [sys.executable, str(ws / {TOOL!r}), 'remote-run', '--arm', {arm!r}, '--started', str(started)]
done = subprocess.run(cmd, cwd=ws, env=env)
assert done.returncode == 0, 'EXP173 incomplete; inspect exp173-status.json and launcher log'
print(pathlib.Path('/kaggle/working/exp173-status.json').read_text())
'''


def stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    need(not out.exists(), 'Use a new stage directory')
    files = package_files(root)
    e165.verify_run(root / e165.EXP / 'outputs/kaggle/EXP165_G4_S42_F0', 'kaggle', root)
    out.mkdir(parents=True)
    code = out / 'code-dataset'
    code.mkdir()
    overlay = code / 'exp173-overlay.zip.bin'
    with zipfile.ZipFile(overlay, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            archive_name = ASSET if name == CONTROL else name
            z.write(root / name, archive_name)
        z.writestr('exp173-overlay-manifest.json', json.dumps({'files': files}, sort_keys=True))
    manifest = {'schema': 'exp173-stage-v1', 'version': VERSION,
                'base_archive_sha256': BASE_SHA, 'overlay_sha256': sha(overlay),
                'split_sha256': sha(root / SPLIT), 'files': files,
                'control_checkpoint_sha256': files[CONTROL]['sha256'], 'arms': list(ARMS)}
    write_new(out / 'stage-receipt.json', manifest)
    write_new(code / 'dataset-metadata.json', dict(id=DATASET, title='EXP173 Gold crossfit private assets',
              isPrivate=True, licenses=[{'name': 'other'}],
              description='Private EXP173 wrapper, split and control checkpoint.'))
    for arm in ARMS:
        dest = out / arm
        dest.mkdir()
        source = notebook_source(arm, manifest['overlay_sha256'])
        compile(source, '<exp173-notebook>', 'exec')
        write_new(dest / 'run.ipynb', dict(nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
                  cells=[dict(cell_type='code', id='exp173', metadata={}, execution_count=None,
                              outputs=[], source=source.splitlines(keepends=True))]))
        slug = 'exp173-gold-crossfit-' + arm.lower().replace('-', '-')
        write_new(dest / 'kernel-metadata.json', dict(id='taeyangg4/' + slug,
                  title='EXP173 Gold crossfit ' + arm, code_file='run.ipynb', language='python',
                  kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
                  enable_internet=False, machine_shape='NvidiaTeslaT4', docker_image=e150.DOCKER,
                  dataset_sources=['taeyangg4/knee-raptor-corpus-336', e165.DATASET, DATASET],
                  competition_sources=['rsna-knee-abnormality-detection'],
                  kernel_sources=[], model_sources=[]))
    verify_stage(out, root)
    return {'status': 'STAGED_NO_PUSH', 'stage': str(out), 'arms': list(ARMS),
            'overlay_sha256': manifest['overlay_sha256']}


def verify_stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    m = read(out / 'stage-receipt.json')
    need(m['schema'] == 'exp173-stage-v1' and m['base_archive_sha256'] == BASE_SHA and
         m['split_sha256'] == sha(root / SPLIT), 'Stage registration mismatch')
    overlay = out / 'code-dataset/exp173-overlay.zip.bin'
    need(sha(overlay) == m['overlay_sha256'], 'Overlay archive mismatch')
    with zipfile.ZipFile(overlay) as z:
        expected = {ASSET if n == CONTROL else n for n in m['files']}
        expected.add('exp173-overlay-manifest.json')
        need(set(z.namelist()) == expected and len(z.namelist()) == len(expected), 'Overlay entries changed')
        need(json.loads(z.read('exp173-overlay-manifest.json')) == {'files': m['files']},
             'Overlay internal manifest changed')
        for name, rec in m['files'].items():
            actual = ASSET if name == CONTROL else name
            data = z.read(actual)
            need(len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256'] and
                 sha(root / name) == rec['sha256'], 'Overlay file changed: ' + name)
    for arm in ARMS:
        meta = read(out / arm / 'kernel-metadata.json')
        need(meta['is_private'] and meta['enable_gpu'] and meta['machine_shape'] == 'NvidiaTeslaT4' and
             DATASET in meta['dataset_sources'] and e165.DATASET in meta['dataset_sources'] and
             meta['id'].endswith(arm.lower()), 'Unsafe kernel metadata')
    return m


def push_plan(out, arm=None, root=ROOT):
    verify_stage(out, root)
    need(arm is None or arm in ARMS, 'Unknown Kaggle arm')
    out = Path(out).as_posix()
    lines = [
        '# B: check live quota and both batch slots; maximum two concurrent batch GPU sessions across notebooks.',
        '# Order: W4-S42 and G4-H1 first, then J4-H1, G4-H2, J4-H2 as slots free.',
        "$kagglePython = (Resolve-Path '.venv/Scripts/python.exe').Path",
        '# Publish the private overlay dataset once, before any arm. If it exists, check privacy/version and use version:',
        f'# Push-Location {out}/code-dataset',
        '# & $kagglePython -X utf8 -m kaggle datasets create -p . --dir-mode zip',
        '# & $kagglePython -X utf8 -m kaggle datasets version -p . -m "EXP173 A1 v1.1.0" --dir-mode zip',
        '# Pop-Location',
        '# Invoke push-plan --arm ARM to get one actionable arm command after a slot is free.'
    ]
    for selected in ((arm,) if arm else ARMS):
        slug = 'taeyangg4/exp173-gold-crossfit-' + selected.lower()
        prefix = '' if arm else '# '
        lines += [f'{prefix}Push-Location {out}/{selected}',
                  f'{prefix}& $kagglePython -X utf8 -m kaggle kernels push -p .',
                  f'{prefix}Pop-Location',
                  f'# After COMPLETE: & $kagglePython -X utf8 -m kaggle kernels output {slug} -p {out}/{selected}/download --page-size 200',
                  f'# Verify each file through collect: & C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe tools/exp173_gold_crossfit.py collect --arm {selected} --source {out}/{selected}/download']
    lines += ['# After W4-S42 and W4-S43 collected: python tools/exp173_gold_crossfit.py screen-w4',
              '# After four cross-fit arms collected: python tools/exp173_gold_crossfit.py screen-crossfit']
    return '\n'.join(lines)


def vast_stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    need(not out.exists(), 'Use a new Vast stage directory')
    files = package_files(root)
    base_sha = verify_base(root, 'vast')
    e150.verify_run(root / e150.EXP / 'outputs/vast/EXP150_C43_F0', 'C', 43, 'vast', root)
    e165.verify_run(root / e165.EXP / 'outputs/vast/EXP165_G4_S43_F0', 'vast', root)
    out.mkdir(parents=True)
    package = out / 'package'
    package.mkdir()
    shutil.copyfile(root / VAST_BASE_ARCHIVE, package / 'exp165-code.zip.bin')
    overlay = package / 'exp173-overlay.zip.bin'
    with zipfile.ZipFile(overlay, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.write(root / name, ASSET if name == CONTROL else name)
        z.writestr('exp173-overlay-manifest.json', json.dumps({'files': files}, sort_keys=True))
    record = dict(schema='exp173-vast-stage-v1.1', version=VERSION, arm=VAST_ARM,
                  base_archive_sha256=base_sha, overlay_sha256=sha(overlay), files=files,
                  split_sha256=sha(root / SPLIT))
    write_new(out / 'stage-receipt.json', record)
    verify_vast_stage(out, root)
    return {'status': 'VAST_STAGED_NO_LAUNCH', 'stage': str(out), 'base_sha256': base_sha,
            'overlay_sha256': record['overlay_sha256']}


def verify_vast_stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    m = read(out / 'stage-receipt.json')
    need(m['schema'] == 'exp173-vast-stage-v1.1' and m['version'] in ('1.0.0', '1.1.0', VERSION) and
         m['arm'] == VAST_ARM and m['split_sha256'] == sha(root / SPLIT) and
         sha(out / 'package/exp165-code.zip.bin') == m['base_archive_sha256'] and
         m['base_archive_sha256'] == sha(root / VAST_BASE_ARCHIVE), 'Vast stage registration mismatch')
    overlay = out / 'package/exp173-overlay.zip.bin'
    need(sha(overlay) == m['overlay_sha256'], 'Vast overlay changed')
    with zipfile.ZipFile(overlay) as z:
        expected = {ASSET if n == CONTROL else n for n in m['files']}
        expected.add('exp173-overlay-manifest.json')
        need(set(z.namelist()) == expected and len(z.namelist()) == len(expected), 'Vast overlay entries changed')
        need(json.loads(z.read('exp173-overlay-manifest.json')) == {'files': m['files']},
             'Vast internal manifest changed')
        for name, rec in m['files'].items():
            data = z.read(ASSET if name == CONTROL else name)
            need(len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256'] and
                 sha(root / name) == rec['sha256'], 'Vast overlay file changed: ' + name)
    return m


def vast_plan(out, host, port, known_hosts, root=ROOT):
    m = verify_vast_stage(out, root)
    need(host == '<remote-host>' and port == 32358 and Path(known_hosts).is_file(),
         'Verify live Vast endpoint and a fresh known_hosts file')
    out = Path(out).as_posix()
    key = 'C:/Users/Taeyang/.ssh/id_ed25519'
    opts = f'-i {key} -o UserKnownHostsFile={known_hosts} -o StrictHostKeyChecking=yes'
    remote = f'root@{host}'
    return '\n'.join([
        '# Git Bash. Check live 3090 and Active compute before launch. Reuse the prior EXP165 archive only if its SHA matches.',
        'export MSYS_NO_PATHCONV=1',
        f'if ssh {opts} -p {port} {remote} "test -f /workspace/exp165-code.zip.bin && echo {m["base_archive_sha256"]}  /workspace/exp165-code.zip.bin | sha256sum -c -"; then echo "reusing EXP165 archive"; else scp {opts} -P {port} {out}/package/exp165-code.zip.bin {remote}:/workspace/exp165-code.zip.bin; fi',
        f'scp {opts} -P {port} {out}/package/exp173-overlay.zip.bin {remote}:/workspace/exp173-overlay.zip.bin',
        f'ssh {opts} -p {port} {remote} "test ! -e /workspace/exp173 && mkdir /workspace/exp173 && python -m zipfile -e /workspace/exp165-code.zip.bin /workspace/exp173 && python -m zipfile -e /workspace/exp173-overlay.zip.bin /workspace/exp173"',
        f'ssh {opts} -p {port} {remote} "test -f /workspace/exp173/config/workspace.toml && cd /workspace/exp173 && (EXP165_REMOTE=vast KNEE_REMOTE_RELAX_ENV=1 KNEE_WORKSPACE_ROOT=/workspace/exp173 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 setsid nohup python tools/exp173_gold_crossfit.py remote-run --arm W4-S43 --started 0 > exp173-queue.log 2>&1 < /dev/null & echo \\$! > /workspace/exp173/exp173-queue.pid)"',
        '# Wait for /workspace/exp173/results/exp173-status.json state COMPLETE, then run each line below.',
        f'# scp {opts} -P {port} -r {remote}:/workspace/exp173/results {out}/download',
        f'# C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe tools/exp173_gold_crossfit.py collect --arm W4-S43 --source {out}/download',
        '# C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe tools/exp173_gold_crossfit.py screen-w4',
    ])


def arm_csv_bytes(root, arm):
    import numpy as np
    root = Path(root)
    need(arm in ALL_ARMS, 'Unknown arm')
    ids = sorted(e165.load_gold(root / e165.GOLD, sha(root / e165.GOLD))) if arm.startswith('W') else \
        load_split(root)['halves'][str(arm_parts(arm)[1])]
    if arm.startswith('G'):
        gold = e165.load_gold(root / e165.GOLD, sha(root / e165.GOLD))
        values = np.stack([gold[u] for u in ids])
    else:
        values = e150.table(root / e150.JEV, ids)
        need(np.isfinite(values).all() and np.isin(values, [0, 1]).all(), 'JEV hard labels absent/nonbinary')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow([e150.UID, *e150.T])
    for uid, row in zip(ids, values):
        writer.writerow([uid, *(str(int(v)) for v in row)])
    return stream.getvalue().encode('utf-8')


def arm_labels(root, arm):
    root = Path(root)
    path = root / EXP / (arm.replace('-', '_') + '-labels.csv')
    need(not path.exists(), 'Arm label file already exists')
    with path.open('xb') as stream:
        stream.write(arm_csv_bytes(root, arm))
    return path


def train_arm(root, arm, output, surface):
    """Wrap, never edit, the frozen EXP165 runner with the matched label contract."""
    root, output = Path(root), Path(output)
    need(arm in ALL_ARMS and surface in e165.SEEDS and
         (arm == VAST_ARM) == (surface == 'vast'), 'Arm/surface mismatch')
    seed = e165.SEEDS[surface]
    rows = 58 if arm.startswith('W') else 29
    labels = arm_labels(root, arm)
    expected_sha = sha(labels)
    from training.runners import exp165_gold_train as runner
    old_validate, old_load = e165.validate_runner_args, e165.load_gold
    def adapted_validate(a, r=root):
        frozen = copy.copy(a)
        frozen.tag = f'EXP165_G4_S{seed}_F0'
        frozen.gold_labels = str(root / e165.GOLD)
        p, fold, ordered, out = old_validate(frozen, r)
        need(a.tag == 'EXP173_' + arm.replace('-', '_') + '_F0' and
             Path(a.gold_labels).resolve() == labels.resolve(), 'Wrong cross-fit args')
        batches = (fold['n_train'] + rows * 4) // a.bs
        steps = batches * a.epochs
        p.update(gold_copies=4, gold_file_sha256=expected_sha,
                 train_loader=batches, scheduler_steps=steps, max_fp16_skips=steps // 100,
                 seed=seed)
        return p, fold, ordered, out
    def adapted_load(path, expected):
        need(Path(path).resolve() == labels.resolve() and expected == expected_sha,
             'Wrong half label input')
        need(sha(labels) == expected_sha, 'Half label hash changed')
        import numpy as np
        with labels.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        need(len(rows) == (58 if arm.startswith('W') else 29), 'Arm labels changed')
        return {row[e150.UID]: np.asarray([int(row[t]) for t in e150.T], np.float32)
                for row in rows}
    argv = e165.runner_args(root, surface, output)
    argv[argv.index('--tag') + 1] = 'EXP173_' + arm.replace('-', '_') + '_F0'
    argv[argv.index('--gold-labels') + 1] = str(labels)
    e165.validate_runner_args, e165.load_gold = adapted_validate, adapted_load
    try:
        runner.main(argv)
    finally:
        e165.validate_runner_args, e165.load_gold = old_validate, old_load
    return output / ('EXP173_' + arm.replace('-', '_') + '_F0')


def serve_pair(root, run, arm, output):
    """K36 held-out inference for candidate and C42 on the same T4 and loader."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    root, run, output = Path(root), Path(run), Path(output)
    sys.path.insert(0, str(root / 'training/src'))
    from raptor_common import Corpus, build_model, write_predictions
    from raptor_data import collate
    from raptor_exp127 import loader_factory
    from raptor_runtime import configure_runtime
    from tools import exp127_radr50_single as guard
    _, training_half, held_half = arm_parts(arm)
    ids = load_split(root)['halves'][str(held_half)]
    p = read(root / e150.BASE)
    corpus = Corpus(root / p['paths']['corpus'])
    need(set(ids).issubset(set(corpus.ids.astype(str))), 'Held-out Gold missing from corpus')
    configure_runtime('fp16', 'cuda', policy='strict')
    labels = {u: np.zeros(12, np.float32) for u in ids}
    dataset = loader_factory('AUG1')(corpus.dir, ids, labels, 224, 36, False, 'radimagenet', 42,
                                     window_policy='within-slot-valid', input_policy='center3')
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate,
                        persistent_workers=False, worker_init_fn=guard.worker_init)
    paths = {'candidate': run / 'model_best.pt', 'control': root / ASSET}
    out = {}
    for name, path in paths.items():
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        need(ckpt['epoch'] == 19 and ckpt['k_eval'] == 24 and ckpt['norm'] == 'radimagenet' and
             ckpt['input_policy'] == 'center3', 'Checkpoint serving recipe changed')
        model = build_model('resnet50', pretrained=False)
        model.load_state_dict(ckpt['model'], strict=True)
        model = model.cuda().eval()
        probs, seen = [], []
        with torch.no_grad():
            for x, _, uids in loader:
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x.cuda())
                need(tuple(logits.shape) == (len(uids), 12) and bool(torch.isfinite(logits).all()),
                     'Invalid held-out logits')
                probs.append(torch.sigmoid(logits.float()).cpu().numpy())
                seen.extend(uids)
        need(seen == ids, 'Held-out inference roster/order changed')
        dest = output / (name + '-heldout-k36.csv')
        write_predictions(dest, seen, np.concatenate(probs))
        out[name] = sha(dest)
        del model, ckpt
        torch.cuda.empty_cache()
    return out


def remote_run(arm, started, root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output is not None else (Path('/workspace/exp173/results') if arm == VAST_ARM else Path('/kaggle/working'))
    surface = 'vast' if arm == VAST_ARM else 'kaggle'
    if surface == 'vast':
        output.mkdir(parents=True, exist_ok=True)
    need(sys.platform.startswith('linux') and os.environ.get('EXP165_REMOTE') == surface and
         os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and arm in ALL_ARMS and
         output.is_dir(), 'Wrong remote execution surface')
    import subprocess
    smi = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    need((len(smi) in (1, 2) if surface == 'kaggle' else len(smi) == 1) and
         all(('T4' if surface == 'kaggle' else '3090') in x for x in smi), 'Wrong GPU')
    m = load_split(root)
    # Frozen EXP165 package and overlay are verified independently, including the C42 checkpoint.
    e165.registration(root)
    overlay_manifest = read(root / 'exp173-overlay-manifest.json')
    for name, rec in overlay_manifest['files'].items():
        local = ASSET if name == CONTROL else name
        need(sha(root / local) == rec['sha256'] and (root / local).stat().st_size == rec['bytes'],
             'Overlay file changed: ' + name)
    need(sha(root / ASSET) == overlay_manifest['files'][CONTROL]['sha256'], 'Control checkpoint changed')
    em = read(root / e150.MANIFEST)
    if surface == 'kaggle':
        corpus = Path('/kaggle/input/datasets/taeyangg4/knee-raptor-corpus-336')
        if not (corpus / 'manifest.json').exists():
            hits = [x.parent for x in Path('/kaggle/input').rglob('manifest.json')
                    if sha(x) == e150.CORPUS_HASH]
            need(len(hits) == 1, 'Corpus mount missing or ambiguous')
            corpus = hits[0]
    else:
        corpus = e150.locate_corpus('vast')
    e150.verify_corpus(corpus, root / e150.CORPUS_LOCAL, em['corpus_file_list'])
    destination = root / read(root / e150.BASE)['paths']['corpus']
    need(not destination.exists(), 'Corpus path already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if surface == 'kaggle':
        destination.symlink_to(corpus, target_is_directory=True)
    else:
        destination.mkdir()
        for source in corpus.rglob('*'):
            target = destination / source.relative_to(corpus)
            if source.is_dir():
                target.mkdir(exist_ok=True)
            else:
                need(source.is_file() and not source.is_symlink(), 'Unsafe corpus entry')
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
                need(source.stat().st_ino == target.stat().st_ino and
                     source.stat().st_dev == target.stat().st_dev, 'Corpus hardlink failed')
        need(sha(destination / 'manifest.json') == e150.CORPUS_HASH, 'Hardlinked corpus changed')
    tag = 'EXP173_' + arm.replace('-', '_') + '_F0'
    status_path = output / 'exp173-status.json'
    need(not status_path.exists() and not (output / tag).exists(), 'Remote output exists')
    started = started or time.time()
    import torch, timm, numpy
    need(not torch.cuda.is_initialized(), 'Preflight touched CUDA')
    software = dict(python=sys.version.split()[0], torch=torch.__version__,
                    timm=timm.__version__, numpy=numpy.__version__, cuda=torch.version.cuda)
    status = dict(schema='exp173-remote-v1.1', arm=arm, surface=surface, state='PREFLIGHT', started=started,
                  split_sha256=sha(root / SPLIT), control_checkpoint_sha256=sha(root / ASSET),
                  gpu=smi, software=software)
    write_new(status_path, status)
    try:
        run = train_arm(root, arm, output, surface)
        scores = serve_pair(root, run, arm, output) if arm in CROSS_ARMS else {}
        inv = read(run / 'output-inventory.json')
        need(set(inv) == {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                          'oof_best.csv', 'oof_k36.csv'}, 'Runner inventory changed')
        for name, record in inv.items():
            need(sha(run / name) == record['sha256'] and (run / name).stat().st_size == record['bytes'],
                 'Runner output incomplete: ' + name)
        summary, prov = read(run / 'summary.json'), read(run / 'provenance.json')
        training_half = None if arm.startswith('W') else arm_parts(arm)[1]
        held_half = None if arm.startswith('W') else arm_parts(arm)[2]
        rows = 58 if arm.startswith('W') else 29
        expected_train = 3455 + rows * 4
        expected_steps = (expected_train // 8) * 20
        need(summary['train'] == expected_train and summary['optimizer_updates'] == expected_steps and
             summary['selected_epoch_1based'] == 20 and summary['skipped_updates'] <= expected_steps // 100 and
             prov['initial_state_matches_registered'] and prov['exp165']['gold_copies'] == 4,
             'Frozen training contract failed')
        receipt = dict(schema='exp173-arm-v1.1', state='COMPLETE', arm=arm, seed=e165.SEEDS[surface],
                       surface=surface, train_half=training_half, held_half=held_half,
                       train_added_rows=rows, label_source='Gold' if arm.startswith('G') else 'JEV',
                       copies=4, train_rows=expected_train, optimizer_updates=expected_steps,
                       split_sha256=sha(root / SPLIT), gold_sha256=m['gold_sha256'],
                       arm_label_sha256=sha(root / EXP / (arm.replace('-', '_') + '-labels.csv')),
                       control_checkpoint_sha256=sha(root / ASSET), heldout_sha256=scores,
                       runner_inventory_sha256=sha(run / 'output-inventory.json'),
                       runner_oof_sha256=sha(run / 'oof_k36.csv'),
                       note='Frozen runner summary/provenance has legacy gold=58 field; actual added rows are bound by arm label hash, roster hash and train count.')
        write_new(output / 'exp173-arm.json', receipt)
        status.update(state='COMPLETE', elapsed_seconds=time.time() - started)
    except BaseException as exc:
        status.update(state='FAILED_OR_INCOMPLETE', reason=type(exc).__name__,
                      elapsed_seconds=time.time() - started)
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    return status


def verify_output(source, arm, root=ROOT):
    root, source = Path(root), Path(source)
    need(arm in ALL_ARMS, 'Unknown arm')
    surface = 'vast' if arm == VAST_ARM else 'kaggle'
    status = read(source / 'exp173-status.json')
    need(status['schema'] == 'exp173-remote-v1.1' and status['state'] == 'COMPLETE' and
         status['arm'] == arm and status['surface'] == surface and status['gpu'] and
         status['split_sha256'] == sha(root / SPLIT) and
         status['control_checkpoint_sha256'] == sha(root / CONTROL) and
         all(('T4' if surface == 'kaggle' else '3090') in line for line in status['gpu']),
         'Kernel incomplete or wrong GPU')
    r = read(source / 'exp173-arm.json')
    training_half = None if arm.startswith('W') else arm_parts(arm)[1]
    held_half = None if arm.startswith('W') else arm_parts(arm)[2]
    rows = 58 if arm.startswith('W') else 29
    need(r['schema'] == 'exp173-arm-v1.1' and r['state'] == 'COMPLETE' and r['arm'] == arm and
         r['surface'] == surface and r['seed'] == e165.SEEDS[surface] and r['copies'] == 4 and
         r['train_half'] == training_half and r['held_half'] == held_half and
         r['split_sha256'] == sha(root / SPLIT) and r['control_checkpoint_sha256'] == sha(root / CONTROL),
         'Arm receipt mismatch')
    need(r['train_added_rows'] == rows and r['gold_sha256'] == sha(root / e165.GOLD) and
         r['train_rows'] == 3455 + rows * 4 and
         r['optimizer_updates'] == ((3455 + rows * 4) // 8) * 20 and
         r['label_source'] == ('Gold' if arm.startswith('G') else 'JEV') and
         r['arm_label_sha256'] == hashlib.sha256(arm_csv_bytes(root, arm)).hexdigest(),
         'Training arm label binding mismatch')
    run = source / ('EXP173_' + arm.replace('-', '_') + '_F0')
    inv = read(run / 'output-inventory.json')
    need(sha(run / 'output-inventory.json') == r['runner_inventory_sha256'] and
         set(inv) == {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                      'oof_best.csv', 'oof_k36.csv'}, 'Missing runner inventory')
    for name, rec in inv.items():
        need(sha(run / name) == rec['sha256'] and (run / name).stat().st_size == rec['bytes'],
             'Missing/truncated runner file: ' + name)
    need(sha(run / 'oof_k36.csv') == r['runner_oof_sha256'], 'OOF receipt mismatch')
    from tools import exp125_radr50_k16_local as e125
    ids888 = read(root / read(root / e150.BASE)['folds']['0']['ordered_ids']['path'])['val']
    e125.load_table(run / 'oof_k36.csv', ids888, prediction=True)
    summary = read(run / 'summary.json')
    prov = read(run / 'provenance.json')
    base = read(root / e150.BASE)
    fold = base['folds']['0']
    ordered = read(root / fold['ordered_ids']['path'])
    added_ids = sorted(e165.load_gold(root / e165.GOLD, sha(root / e165.GOLD))) if arm.startswith('W') else \
        load_split(root)['halves'][str(training_half)]
    need(summary['train'] == 3455 + rows * 4 and
         summary['optimizer_updates'] == ((3455 + rows * 4) // 8) * 20 and
         summary['selected_epoch_1based'] == 20 and summary['val'] == 888 and
         summary['skipped_updates'] <= ((3455 + rows * 4) // 8) * 20 // 100 and
         len(read(run / 'history.json')) == 20 and summary['provenance'] == prov and
         prov['exp165']['gold_copies'] == 4 and
         prov['exp165']['gold_labels_sha256'] == r['arm_label_sha256'] and
         prov['train_ids_ordered_sha256'] == e150.fhash(ordered['train'] + added_ids * 4) and
         prov['eval_ids_ordered_sha256'] == e150.fhash(ordered['val']) and
         prov['initial_state_sha256'] == read(root / e150.MANIFEST)['initial_state_sha256'] and
         prov['actual_pos_weight'] == fold['pos_weight'], 'Training receipt/count mismatch')
    if arm.startswith('W'):
        need(r['heldout_sha256'] == {}, 'W4 must have no Gold-heldout inference')
    else:
        ids = load_split(root)['halves'][str(held_half)]
        for name in ('candidate', 'control'):
            path = source / (name + '-heldout-k36.csv')
            need(sha(path) == r['heldout_sha256'][name], 'Missing/truncated heldout file: ' + name)
            e125.load_table(path, ids, prediction=True)
    return r


def collect(source, arm, root=ROOT):
    root, source = Path(root), Path(source)
    verify_output(source, arm, root)
    dest = root / EXP / 'outputs' / arm
    need(not dest.exists(), 'Already collected')
    shutil.copytree(source, dest)
    verify_output(dest, arm, root)
    return {'status': 'COLLECTED_VALIDATED', 'arm': arm, 'destination': str(dest)}


def _auc(y, p):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    result = ev.auc_targets(y, p, np.ones(y.shape, bool), 1)
    need(np.isfinite(result).all(), 'Gold half lacks target classes')
    return result


def half_bootstrap(halves, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Paired, independently resampled studies within each scored half."""
    import numpy as np
    need(len(halves) == 2 and all({'y', 'G4', 'J4', 'C', 'C_J'}.issubset(h) and
         all(h[k].shape == (29, 12) for k in ('y', 'G4', 'J4', 'C', 'C_J'))
         for h in halves), 'Half inputs')
    pairs = {'G4-J4': ('G4', 'J4'), 'G4-C': ('G4', 'C'), 'J4-C': ('J4', 'C_J')}
    half_targets = {label: [_auc(h['y'], h[a]) - _auc(h['y'], h[b]) for h in halves]
                    for label, (a, b) in pairs.items()}
    targets = {name: np.mean(per_half, axis=0)
               for name, per_half in half_targets.items()}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in targets}
    attempts = 0
    while len(values['G4-J4']) < draws:
        attempts += 1
        need(attempts <= draws * 1000, 'Too many degenerate half-stratified draws')
        ix = [rng.integers(0, 29, 29) for _ in halves]
        if any(not (((h['y'][i].sum(0) > 0) & (h['y'][i].sum(0) < 29)).all())
               for h, i in zip(halves, ix)):
            continue
        auc = [{name: _auc(h['y'][i], h[name][i]) for name in ('G4', 'J4', 'C', 'C_J')}
               for h, i in zip(halves, ix)]
        for label, (a, b) in pairs.items():
            values[label].append(float(np.mean([np.mean(v[a] - v[b]) for v in auc])))
    result = {}
    for name, target in targets.items():
        arr = np.asarray(values[name])
        result[name] = dict(delta=float(np.mean(target)), ci95=np.percentile(arr, [2.5, 97.5]).tolist(),
                            paired_se=float(arr.std(ddof=1)),
                            targets={t: float(v) for t, v in zip(e150.T, target)})
    by_half = {str(i+1): {name: dict(delta=float(np.mean(vector)),
                                      targets={t: float(v) for t, v in zip(e150.T, vector)})
                             for name, vectors in half_targets.items() for vector in (vectors[i],)}
               for i in range(2)}
    return dict(comparisons=result, scored_halves=by_half,
                bootstrap=dict(draws=draws, seed=seed, attempts=attempts,
                 unit='study within scored half', degenerate_draws_rejected=attempts-draws))


def _normalized_ranks(values):
    import numpy as np
    from scipy.stats import rankdata
    need(values.shape == (29, 12), 'Rank half shape')
    return np.stack([(rankdata(values[:, j], method='average') - 1) / 28 for j in range(12)], axis=1)


def cell_readout(halves):
    """Gold-oriented rank movement on JEV-disagreeing versus agreeing cells."""
    import numpy as np
    out = {}
    for a, b in (('G4', 'J4'), ('G4', 'C')):
        groups = {'JEV_disagrees_with_Gold': [], 'JEV_agrees_with_Gold': []}
        by_target = {t: {'disagree': [], 'agree': []} for t in e150.T}
        for h in halves:
            direction = 2 * h['y'] - 1
            movement = direction * (_normalized_ranks(h[a]) - _normalized_ranks(h[b]))
            disagreement = h['jev'] != h['y']
            groups['JEV_disagrees_with_Gold'].extend(movement[disagreement].tolist())
            groups['JEV_agrees_with_Gold'].extend(movement[~disagreement].tolist())
            for j, target in enumerate(e150.T):
                by_target[target]['disagree'].extend(movement[disagreement[:, j], j].tolist())
                by_target[target]['agree'].extend(movement[~disagreement[:, j], j].tolist())
        out[f'{a}-{b}'] = dict(cells={name: dict(n=len(v), mean_toward_Gold=float(np.mean(v)) if v else None)
                                        for name, v in groups.items()},
                             targets={t: {k: dict(n=len(v), mean_toward_Gold=float(np.mean(v)) if v else None)
                                          for k, v in body.items()} for t, body in by_target.items()})
    return out


def _refs(root, ids):
    import numpy as np
    em = read(root / e150.MANIFEST)
    for name, digest in em['readout_sha256'].items():
        need(sha(root / name) == digest, 'Weak reference changed: ' + name)
    refs = {}
    y = e150.table(root / e150.JEV, ids)
    need(np.isfinite(y).all() and np.isin(y, [0, 1]).all(), 'JEV weak reference changed')
    refs['jev_hybrid'] = (y, np.ones(y.shape, bool), 1)
    y = e150.table(root / e150.AQUILA, ids)
    need(np.isfinite(y).all(), 'Aquila reference changed')
    refs['aquila_v1'] = ((y >= .5).astype(float), np.ones(y.shape, bool), 1)
    y = e150.table(root / e150.SETS['dread'], ids)
    need(np.isfinite(y).all(), 'Dread reference changed')
    refs['dread'] = (np.rint(y.astype(np.float32)).astype(float), np.ones(y.shape, bool), 1)
    stack = np.stack([e150.table(root / path, ids) for path in e150.SETS.values()])
    valid = np.isfinite(stack).all(0)
    binary = stack >= .5
    agree = valid & (binary.all(0) | (~binary).all(0))
    refs['consensus'] = (binary[0].astype(float), agree, 2)
    return refs


def screen_crossfit(root=ROOT):
    import numpy as np
    from tools import exp125_radr50_k16_local as e125
    from tools import exp115_vlm_eval as ev
    root = Path(root)
    split_obj = load_split(root)
    gold = e165.load_gold(root / e165.GOLD, sha(root / e165.GOLD))
    e150.verify_run(root / Path(CONTROL).parent, 'C', 42, 'kaggle', root)
    ids888 = read(root / read(root / e150.BASE)['folds']['0']['ordered_ids']['path'])['val']
    refs = _refs(root, ids888)
    control_oof = e125.load_table(root / e150.EXP / 'outputs/kaggle/EXP150_C42_F0/oof_k36.csv', ids888, prediction=True)
    halves, weak = [], {}
    for scored_half in (1, 2):
        training_half = 3 - scored_half
        ids = split_obj['halves'][str(scored_half)]
        h = {'y': np.stack([gold[u] for u in ids]), 'jev': e150.table(root / e150.JEV, ids)}
        need(np.isfinite(h['jev']).all() and np.isin(h['jev'], [0, 1]).all(), 'Missing JEV Gold cells')
        for prefix in ('G4', 'J4'):
            arm = f'{prefix}-H{training_half}'
            source = root / EXP / 'outputs' / arm
            verify_output(source, arm, root)
            h[prefix] = e125.load_table(source / 'candidate-heldout-k36.csv', ids, prediction=True)
            control = e125.load_table(source / 'control-heldout-k36.csv', ids, prediction=True)
            if 'C' in h:
                need(np.allclose(control, h['C'], atol=1e-5, rtol=0), 'Same-half C42 serving drift')
                h['C_J'] = control
            else:
                h['C'] = control
            run = source / ('EXP173_' + arm.replace('-', '_') + '_F0')
            pred = e125.load_table(run / 'oof_k36.csv', ids888, prediction=True)
            weak[arm] = {name: float(ev.macro(ev.auc_targets(ref[0], pred, ref[1], ref[2])) -
                                     ev.macro(ev.auc_targets(ref[0], control_oof, ref[1], ref[2])))
                         for name, ref in refs.items()}
        halves.append(h)
    primary = half_bootstrap(halves)
    pooled = {name: float(np.mean(_auc(np.concatenate([h['y'] for h in halves]),
                                      np.concatenate([h[a] for h in halves])) -
                                  _auc(np.concatenate([h['y'] for h in halves]),
                                       np.concatenate([h[b] for h in halves]))))
              for name, (a, b) in {'G4-J4': ('G4', 'J4'), 'G4-C': ('G4', 'C'),
                                    'J4-C': ('J4', 'C_J')}.items()}
    result = dict(schema='exp173-crossfit-screen-v1.1', status='EXPLORATORY',
                  split_sha256=sha(root / SPLIT), gold58_exposed=True,
                  expected_paired_se='approximately 0.008-0.010; planning proxy, not observed precision',
                  primary='mean within-half paired macro-12 Gold AUC delta G4-J4',
                  primary_and_secondary=primary,
                  pooled_cross_half_auc_sensitivity=pooled,
                  cell_rank_readout=cell_readout(halves), v888_weak_reference_deltas=weak,
                  interpretation='No automatic Gold-training decision; Taeyang decides after results.')
    dest = root / EXP / 'screen-crossfit'
    need(not dest.exists(), 'Cross-fit screen already exists')
    dest.mkdir()
    write_new(dest / 'result.json', result)
    return result


def screen_w4(root=ROOT):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp125_radr50_k16_local as e125
    root = Path(root)
    m, p = e165.registration(root)
    prior_path = root / e165.EXP / 'screen/result.json'
    prior = read(prior_path)
    need(prior['schema'] == 'exp165-screen-v1.1' and prior['rows'] == 888,
         'EXP165 G4-C screen missing or changed')
    ids = read(root / p['folds']['0']['ordered_ids']['path'])['val']
    need(len(ids) == 888, 'V888 changed')
    refs = _refs(root, ids)
    draws = e125.component_draws(ids, read(root / e150.COMPONENTS), draws=300, seed=BOOT_SEED)
    seeds, paired = {}, {}
    for surface, seed in e165.SEEDS.items():
        control = root / e150.EXP / 'outputs' / surface / f'EXP150_C{seed}_F0'
        gold_run = root / e165.EXP / 'outputs' / surface / f'EXP165_G4_S{seed}_F0'
        arm = f'W4-S{seed}'
        weak_run = root / EXP / 'outputs' / arm
        e150.verify_run(control, 'C', seed, surface, root)
        need(sha(control / 'oof_k36.csv') == m['controls'][surface]['oof_sha256'], 'Control drift')
        e165.verify_run(gold_run, surface, root)
        verify_output(weak_run, arm, root)
        predictions = {
            'C': e125.load_table(control / 'oof_k36.csv', ids, prediction=True),
            'G4': e125.load_table(gold_run / 'oof_k36.csv', ids, prediction=True),
            'W4': e125.load_table(weak_run / f'EXP173_{arm.replace("-", "_")}_F0/oof_k36.csv', ids, prediction=True)}
        paired[surface] = predictions
        comparisons = {}
        for label, a, b in (('W4-C', 'W4', 'C'), ('G4-W4', 'G4', 'W4')):
            metrics = {name: ev.compare(predictions[a], predictions[b], ref, draws)
                       for name, ref in refs.items()}
            for body in metrics.values():
                need(np.isfinite(body['delta']) and np.isfinite(body['ci95']).all(), 'Nonfinite V888 delta')
                body['targets'] = {t: float(v) if np.isfinite(v) else None
                                   for t, v in body['targets'].items()}
            primary = float(np.mean([metrics[n]['delta'] for n in ('jev_hybrid', 'aquila_v1')]))
            ci = e165.paired_mean_ci(predictions[a], predictions[b],
                                    (refs['jev_hybrid'], refs['aquila_v1']), draws)
            comparisons[label] = dict(primary_mean_delta=primary, primary_ci95=ci, metrics=metrics)
        seeds[surface] = dict(seed=seed, comparisons=comparisons,
                              prediction_sha256={name: sha(path) for name, path in {
                                  'C': control / 'oof_k36.csv', 'G4': gold_run / 'oof_k36.csv',
                                  'W4': weak_run / f'EXP173_{arm.replace("-", "_")}_F0/oof_k36.csv'}.items()})
    pooled = {}
    for label, a, b in (('W4-C', 'W4', 'C'), ('G4-W4', 'G4', 'W4')):
        metrics = {}
        for name, ref in refs.items():
            bodies = [seeds[s]['comparisons'][label]['metrics'][name] for s in e165.SEEDS]
            sampled = [e165.paired_mean_draws(paired[s][a], paired[s][b], (ref,), draws)
                       for s in e165.SEEDS]
            metrics[name] = dict(delta=float(np.mean([x['delta'] for x in bodies])),
                                 ci95=np.percentile(np.mean(sampled, axis=0), [2.5, 97.5]).tolist(),
                                 targets={t: (float(np.mean([x['targets'][t] for x in bodies]))
                                              if all(x['targets'][t] is not None for x in bodies) else None)
                                          for t in e150.T})
        sampled = [e165.paired_mean_draws(paired[s][a], paired[s][b],
                   (refs['jev_hybrid'], refs['aquila_v1']), draws) for s in e165.SEEDS]
        pooled[label] = dict(primary_mean_delta=float(np.mean([seeds[s]['comparisons'][label]['primary_mean_delta']
                                                               for s in e165.SEEDS])),
                             primary_ci95=np.percentile(np.mean(sampled, axis=0), [2.5, 97.5]).tolist(),
                             metrics=metrics)
    result = dict(schema='exp173-w4-screen-v1.1', status='EXPLORATORY', rows=888,
                  references=list(refs), bootstrap=dict(unit='EXP165 component', draws=300, seed=BOOT_SEED),
                  comparisons=['W4-C', 'G4-W4'], seeds=seeds, pooled=pooled,
                  existing_g4_minus_c=dict(source=str(prior_path.relative_to(root)), sha256=sha(prior_path),
                                           seeds={s: dict(primary_mean_delta=prior['seeds'][s]['primary_mean_delta'],
                                                          metrics=prior['seeds'][s]['metrics']) for s in e165.SEEDS},
                                           pooled_primary=prior['pooled_primary'],
                                           pooled_metrics=prior['pooled_metrics']),
                  interpretation=[
                      'W4-C approximately G4-C: EXP165 weak-reference harm tracks duplication/case mix, not Gold definition.',
                      'W4-C approximately zero and G4-W4 negative, concentrated in Synovitis/Contusion/Effusion/Fracture/Lateral OA: definition shift penalised by weak rulers; not yet a leaderboard benefit.'],
                  decision='No automatic Gold-training or leaderboard decision; Taeyang decides after results.')
    dest = root / EXP / 'screen-w4'
    need(not dest.exists(), 'W4 screen already exists')
    dest.mkdir()
    write_new(dest / 'result.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('split')
    stage_cmd = sub.add_parser('stage'); stage_cmd.add_argument('--out', required=True)
    vast_stage_cmd = sub.add_parser('vast-stage'); vast_stage_cmd.add_argument('--out', required=True)
    plan_cmd = sub.add_parser('push-plan'); plan_cmd.add_argument('--out', required=True)
    plan_cmd.add_argument('--arm', choices=ARMS)
    vast_plan_cmd = sub.add_parser('vast-plan'); vast_plan_cmd.add_argument('--out', required=True)
    vast_plan_cmd.add_argument('--host', required=True); vast_plan_cmd.add_argument('--port', type=int, required=True)
    vast_plan_cmd.add_argument('--known-hosts', required=True)
    remote = sub.add_parser('remote-run'); remote.add_argument('--arm', choices=ALL_ARMS, required=True)
    remote.add_argument('--started', type=float, default=0)
    collect_cmd = sub.add_parser('collect'); collect_cmd.add_argument('--arm', choices=ALL_ARMS, required=True)
    collect_cmd.add_argument('--source', required=True)
    sub.add_parser('screen-crossfit')
    sub.add_parser('screen-w4')
    a = ap.parse_args(argv)
    if a.command == 'split':
        value = split()
    elif a.command == 'stage':
        value = stage(a.out)
    elif a.command == 'vast-stage':
        value = vast_stage(a.out)
    elif a.command == 'push-plan':
        value = push_plan(a.out, a.arm)
    elif a.command == 'vast-plan':
        value = vast_plan(a.out, a.host, a.port, a.known_hosts)
    elif a.command == 'remote-run':
        value = remote_run(a.arm, a.started)
    elif a.command == 'collect':
        value = collect(a.source, a.arm)
    elif a.command == 'screen-crossfit':
        value = screen_crossfit()
    else:
        value = screen_w4()
    print(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
