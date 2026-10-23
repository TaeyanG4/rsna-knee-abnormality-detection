"""EXP162 v1.0.0: offline Kaggle stage, two-T4 coordinator, collection, CPU screen.

Only ``run``/``worker`` are notebook entrypoints. No command publishes or trains
on the local machine. The EXP158 runner and C's source are copied, never edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'artifacts/experiments/EXP162_DINO_FOLDS'
STAGE = ROOT / '.local/exp162-dino-folds-v1'
DATASET = 'taeyangg4/knee-exp162-dino-folds-assets'
KERNEL = 'taeyangg4/knee-exp162-dino-folds'
CORPUS = 'taeyangg4/knee-raptor-corpus-336'
CORPUS_SHA = 'f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
INITIAL_SHA = '688756a1dff13683a5d55a60dd667e0bbd88b17df324d2598c5baf3122c41ad5'
FULL_SHA = '4ea6767e85f8090e16d4238593d8d16809735002de1cb32255a6a7bf3d03f563'
WEIGHT_SHA = '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
PARENT_SHA = '4e21fcc33ee7f27468f0e183710110326a2f9a0225037d87ac34395f62f311af'
PARENT = '.local/hc06-illinois-pair-v2/manifest.json'
SOURCE_RUNNER = '.local/exp158-dino-strong-v1/training/runners/exp158_dino_strong.py'
RUNNER = 'training/runners/exp162_dino_fold.py'
SCHEMA = 'exp162-dino-folds-v1'
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
UID = 'StudyInstanceUID'
FOLD_COUNTS = {0: 888, 1: 864, 2: 864, 3: 864, 4: 863}
STUDIES = 4343
ASSIGNMENT = ((1, 3), (2, 4))
MAX_WALL = 4 * 3600
FOLD_RESERVE = 105 * 60
OUTPUT_RESERVE = 10 * 60
JEV = 'storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv'
AQUILA = 'storage/labels/derived_constellation/aquila_v1.csv'
DREAD = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
CONSENSUS = (DREAD,
             'storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
             'storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv',
             'storage/labels/derived_relabel/relabel_v1.csv')
CODE = ('training/src/raptor_train.py', 'training/src/raptor_common.py',
        'training/src/raptor_data.py', 'training/src/raptor_runtime.py',
        'training/src/raptor_exp127_aug1.py', 'training/src/exp158_aug1_triplet.py',
        'training/adapters/dinov2_cpu_pos_v2.py', 'training/adapters/dinov2_patch_mil.py',
        'tools/exp084_group_loss.py', 'tools/exp086_dinov2s_v2.py',
        'tools/vast_hc06.py', 'tools/exp158_dino_strong.py',
        'tools/exp162_dino_folds_kaggle.py',
        # B hotfix 2026-10-09 (1.0.2): transitive imports missing from v1 (kernel v2 failed on raptor_radimagenet)
        'training/src/raptor_radimagenet.py', 'tools/process_resource_receipt.py', 'tools/exp082_r34_loss.py')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def safe(name):
    p = PurePosixPath(name)
    need(isinstance(name, str) and name and p.as_posix() == name and not p.is_absolute()
         and all(x not in ('.', '..') for x in p.parts) and ':' not in name and '\\' not in name,
         'unsafe archive member')
    return name


def roster(root=ROOT):
    """Prove EXP158's historical fold 2 is OUR fold 0, then return all rosters."""
    root = Path(root)
    parent = read(root / PARENT)
    need(sha(root / PARENT) == PARENT_SHA, 'C parent manifest changed')
    old = read(root / parent['ordered_ids']['path'])
    proto = read(root / 'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1/s42/manifest.json')
    result, covered = {}, set()
    for fold in range(5):
        path = proto['folds'][str(fold)]['ordered_ids']['path']
        ids = read(root / path)
        val, train = ids['val'], ids['train']
        need(sha(root / path) == proto['folds'][str(fold)]['ordered_ids']['sha256'], 'roster hash differs')
        need(len(val) == FOLD_COUNTS[fold] and len(train) + len(val) == STUDIES
             and len(set(val)) == len(val) and not set(val) & set(train), 'fold counts/membership differ')
        need(not set(val) & covered, 'held-out folds overlap')
        covered.update(val)
        result[fold] = (path, ids)
    need(result[0][1] == old and len(covered) == STUDIES
         and all(set(v['train']) == covered - set(v['val']) for _, v in result.values()),
         'EXP158/EXP127 fold mapping or five-fold coverage differs')
    return result


def tag(fold):
    need(fold in range(1, 5), 'EXP162 trains only our folds 1-4')
    return f'EXP162_DINO_S42_F{fold}'


def runner_source(root=ROOT):
    source = (Path(root) / SOURCE_RUNNER).read_text(encoding='utf-8')
    old = 'from tools import exp158_dino_strong as guard'
    need(source.count(old) == 1, 'EXP158 frozen runner import changed')
    return source.replace(old, 'from tools import exp162_dino_folds_kaggle as guard')


def pack(path, members):
    need(str(path).endswith('.zip.bin') and not Path(path).exists(), 'fresh .zip.bin required')
    need(len(members) == len(set(members)), 'duplicate archive member')
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, source in members.items():
            safe(name)
            if isinstance(source, bytes):
                archive.writestr(name, source)
            else:
                archive.write(source, name)
    return sha(path)


def _receipt(members):
    return {name: {'sha256': hashlib.sha256(src).hexdigest() if isinstance(src, bytes) else sha(src),
                   'bytes': len(src) if isinstance(src, bytes) else Path(src).stat().st_size}
            for name, src in members.items()}


def stage(out=STAGE, root=ROOT):
    out, root = Path(out).resolve(), Path(root).resolve()
    need(not out.exists(), 'fresh stage directory required')
    need((root / 'artifacts/experiments/EXP162_DINO_FOLDS/PROTOCOL.md').is_file(), 'register protocol first')
    folds = roster(root)
    parent = read(root / PARENT)
    files = {n: root / n for n in CODE}
    # The live workspace helper was revised after C's frozen run. Package C's
    # registered copy so the import chain stays bound to EXP158 provenance.
    files['tools/vast_hc06.py'] = root / '.local/hc06-illinois-pair-v2/vast_hc06.py'
    files['config/workspace.toml'] = b'# isolated EXP162 Kaggle workspace\n'
    exp158_manifest = read(root / '.local/exp158-dino-strong-v1/manifest.json')
    need(sha(root / SOURCE_RUNNER) ==
         exp158_manifest['files']['training/runners/exp158_dino_strong.py']['sha256'],
         'frozen EXP158 runner changed')
    files[RUNNER] = runner_source(root).encode()
    files[DREAD] = root / DREAD
    for name in (parent['groups']['path'], parent['paths']['exclude_ids']):
        files[name] = root / name
    # B hotfix 2026-10-09 (1.0.3): C's exp084 validate_membership requires SORTED train/val lists; our
    # EXP099 folds 1-4 keep an unsorted train order (fold 0 = EXP158 is sorted). Same membership, sorted copies.
    sorted_paths = {}
    for fold in range(1, 5):
        path, ids = folds[fold]
        rel = f'artifacts/experiments/EXP162_DINO_FOLDS/rosters/fold{fold}-sorted.json'
        body = (json.dumps({'train': sorted(ids['train']), 'val': sorted(ids['val'])}, indent=1) + chr(10)).encode()
        local = root / rel
        if local.exists():
            need(local.read_bytes() == body, 'sorted roster differs: ' + rel)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(body)
        files[rel] = local
        sorted_paths[fold] = rel
    for name in CODE:
        expected = exp158_manifest['files'].get(name, parent['files'].get(name))
        if expected:
            need(sha(files[name]) == expected['sha256'], 'frozen C source changed: ' + name)
    weight_dir = root / '.local/vast-hc06-package-54849265-v2/storage/weights/vast_hc06'
    assets = {'storage/weights/vast_hc06/dinov2s.safetensors': weight_dir / 'dinov2s.safetensors',
              'storage/weights/vast_hc06/full_initial_state.safetensors': weight_dir / 'full_initial_state.safetensors'}
    need(sha(assets['storage/weights/vast_hc06/dinov2s.safetensors']) == WEIGHT_SHA
         and sha(assets['storage/weights/vast_hc06/full_initial_state.safetensors']) == FULL_SHA,
         'canonical DINO asset changed')
    files.update(assets)
    need(sha(files[DREAD]) == parent['files'][DREAD]['sha256'], 'Dread label file changed')
    (out / 'dataset').mkdir(parents=True)
    (out / 'kernel').mkdir()
    for fold in range(1, 5):
        _, ids = folds[fold]
        path = sorted_paths[fold]
        config = {k: v for k, v in parent.items() if k not in ('files', 'phases', 'execution', 'arms')}
        args = dict(parent['arms']['PATCH']['expected_args'])
        args.update(epochs=20, fixed_epoch=20, k_eval=36, fold=fold, workers=2)
        config.update(schema=SCHEMA, fold=fold, tag=tag(fold), expected_args=args,
                      ordered_ids={'path': path, 'sha256': sha(root / path)},
                      expected_ids={'train': len(ids['train']), 'val': len(ids['val'])},
                      arm={**parent['arms']['PATCH'], 'id': 'PATCH', 'tag': tag(fold),
                           'expected_args': args, 'expected_attempts': (len(ids['train']) // 8) * 20},
                      aug1_sha256=sha(root / 'training/src/raptor_exp127_aug1.py'),
                      aug1_triplet_sha256=sha(root / 'training/src/exp158_aug1_triplet.py'),
                      protocol_sha256=sha(root / 'artifacts/experiments/EXP162_DINO_FOLDS/PROTOCOL.md'))
        files[f'protocols/fold{fold}.json'] = (json.dumps(config, indent=2, ensure_ascii=False) + '\n').encode()
    members = _receipt(files)
    archive = out / 'dataset/exp162-package.zip.bin'
    archive_sha = pack(archive, files)
    manifest = {'schema': SCHEMA, 'version': '1.0.0', 'protocol_sha256': sha(root / 'artifacts/experiments/EXP162_DINO_FOLDS/PROTOCOL.md'),
                'dataset': DATASET, 'kernel': KERNEL, 'corpus': CORPUS, 'corpus_sha256': CORPUS_SHA,
                'initial_state_sha256': INITIAL_SHA, 'full_asset_sha256': FULL_SHA,
                'generic_asset_sha256': WEIGHT_SHA, 'max_wall_seconds': MAX_WALL,
                'assignment': [list(x) for x in ASSIGNMENT], 'members': members,
                'archive_sha256': archive_sha}
    save(out / 'dataset/manifest.json', manifest)
    save(out / 'dataset/dataset-metadata.json', {'id': DATASET, 'title': 'knee-exp162-dino-folds-assets',
         'isPrivate': True, 'licenses': [{'name': 'other'}],
         'description': 'Private canonical DINO state, Dread labels, rosters and frozen EXP158/C code.'})
    save(out / 'kernel/kernel-metadata.json', {'id': KERNEL, 'title': 'knee-exp162-dino-folds',
         'code_file': 'run.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
         'enable_gpu': True, 'enable_tpu': False, 'machine_shape': 'NvidiaTeslaT4',
         'enable_internet': False, 'dataset_sources': [DATASET, CORPUS],
         'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
    save(out / 'kernel/run.ipynb', notebook(sha(out / 'dataset/manifest.json')))
    result = {'status': 'STAGED_NO_GPU', 'stage': str(out), 'dataset': DATASET, 'kernel': KERNEL,
              'asset_bytes': archive.stat().st_size, 'manifest_sha256': sha(out / 'dataset/manifest.json'),
              'notebook_sha256': sha(out / 'kernel/run.ipynb')}
    save(out / 'stage-receipt.json', result)
    return result


def notebook(manifest_sha):
    cell = """import os, time
STARTED = time.monotonic()
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib, json, pathlib, shutil, subprocess, sys, zipfile
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8<<20), b''): h.update(b)
    return h.hexdigest()
def mount(owner, slug, marker):
    roots = [pathlib.Path('/kaggle/input/datasets')/owner/slug, pathlib.Path('/kaggle/input')/slug]
    hits = [p for p in roots if (p/marker).is_file()]
    assert len(hits) == 1, (slug, len(hits))
    return hits[0]
assets = mount('taeyangg4', 'knee-exp162-dino-folds-assets', 'manifest.json')
corpus = mount('taeyangg4', 'knee-raptor-corpus-336', 'manifest.json')
m = json.loads((assets/'manifest.json').read_text())
assert sha(assets/'manifest.json') == EXPECTED_MANIFEST_SHA
assert m['schema'] == 'exp162-dino-folds-v1'
assert sha(corpus/'manifest.json') == m['corpus_sha256']
archive = assets/'exp162-package.zip.bin'
assert sha(archive) == m['archive_sha256']
work = pathlib.Path('/tmp/exp162-work')
work.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert len(names) == len(set(names)) and set(names) == set(m['members'])
    for name in names:
        p = pathlib.PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
        dest = work/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as src, dest.open('xb') as target: shutil.copyfileobj(src, target)
        assert dest.stat().st_size == m['members'][name]['bytes'] and sha(dest) == m['members'][name]['sha256']
link = work/'storage/cache/derived/RAPTOR64_336_C140_S02'
link.parent.mkdir(parents=True, exist_ok=True)
link.symlink_to(corpus, target_is_directory=True)
# B hotfix 2026-10-09 (tool 1.0.1): the package already ships artifacts/experiments/* protocol files, so link only
# the four fold output dirs to /kaggle/working instead of replacing the whole directory.
(work/'artifacts/experiments').mkdir(parents=True, exist_ok=True)
for fold in (1, 2, 3, 4):
    target = pathlib.Path('/kaggle/working')/f'EXP162_DINO_S42_F{fold}'
    target.mkdir(exist_ok=False)
    (work/'artifacts/experiments'/target.name).symlink_to(target, target_is_directory=True)
env = dict(os.environ, KNEE_WORKSPACE_ROOT=str(work), PYTHONUTF8='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
env['PYTHONPATH'] = str(work)
assert time.monotonic()-STARTED < 900, 'EXP162 mount/extract startup exceeded 15 minutes'
cmd = [sys.executable, str(work/'tools/exp162_dino_folds_kaggle.py'), 'run',
       '--work', str(work), '--corpus', str(corpus), '--manifest', str(assets/'manifest.json'),
       '--started', str(STARTED)]
subprocess.run(cmd, env=env, timeout=max(1, m['max_wall_seconds']-(time.monotonic()-STARTED)))
"""
    cell = f'EXPECTED_MANIFEST_SHA = {manifest_sha!r}\n' + cell
    compile(cell, '<EXP162 notebook>', 'exec')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
            'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': '# EXP162 DINO five-fold completion'},
                      {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': cell}]}


def validate_package(work, corpus, manifest):
    work, corpus, manifest = map(Path, (work, corpus, manifest))
    m = read(manifest)
    need(m['schema'] == SCHEMA and m['version'] == '1.0.0' and
         m['assignment'] == [list(x) for x in ASSIGNMENT] and
         m['initial_state_sha256'] == INITIAL_SHA and m['full_asset_sha256'] == FULL_SHA and
         m['generic_asset_sha256'] == WEIGHT_SHA and m['corpus_sha256'] == CORPUS_SHA,
         'wrong EXP162 manifest')
    need(sha(corpus / 'manifest.json') == CORPUS_SHA, 'corpus manifest changed')
    for name, entry in m['members'].items():
        target = (work / safe(name)).resolve()
        need(target.is_relative_to(work.resolve()) and target.is_file() and
             target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'],
             'package member changed: ' + name)
    return m


def runner_argv(protocol, work):
    p = read(protocol)
    args = [sys.executable, str(Path(work) / RUNNER), '--protocol', str(protocol),
            '--phase', 'full', '--tag', p['tag']]
    for key, value in {**p['expected_args'], **p['paths']}.items():
        if isinstance(value, bool):
            if value:
                args.append('--' + key.replace('_', '-'))
        else:
            args.extend(('--' + key.replace('_', '-'),
                         str(Path(work) / value) if key in p['paths'] else str(value)))
    return args


def _status_path(fold):
    return Path('/kaggle/working') / f'exp162-fold{fold}-status.json'


def admit_fold(remaining):
    """Only start a fold when its registered cap and output reserve both fit."""
    return math.isfinite(remaining) and remaining >= FOLD_RESERVE + OUTPUT_RESERVE


def worker(work, corpus, manifest, started, worker_id):
    need(sys.platform.startswith('linux') and worker_id in (0, 1), 'Kaggle Linux worker required')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and
         os.environ.get('CUDA_VISIBLE_DEVICES') == str(worker_id), 'worker GPU pin missing')
    m = validate_package(work, corpus, manifest)
    import torch  # CUDA is first queried only after pin and CUBLAS configuration.
    need(torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0),
         'one visible T4 required per worker')
    results = []
    for fold in ASSIGNMENT[worker_id]:
        protocol = Path(work) / f'protocols/fold{fold}.json'
        p = read(protocol)
        need(p['fold'] == fold and p['tag'] == tag(fold), 'worker fold binding differs')
        remaining = MAX_WALL - OUTPUT_RESERVE - (time.monotonic() - started)
        status = {'schema': SCHEMA, 'fold': fold, 'worker': worker_id,
                  'gpu': torch.cuda.get_device_name(0), 'remaining_seconds_at_start': round(remaining, 1),
                  'protocol_sha256': sha(protocol), 'status': 'SKIPPED_DEADLINE'}
        if not admit_fold(remaining + OUTPUT_RESERVE):
            save(_status_path(fold), status)
            results.append(status)
            continue
        status['status'] = 'RUNNING'
        save(_status_path(fold), status)
        log = Path('/kaggle/working') / f'exp162-fold{fold}.log'
        env = dict(os.environ, PYTHONPATH=str(work), KNEE_WORKSPACE_ROOT=str(work),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES=str(worker_id),
                   PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        started_fold = time.monotonic()
        try:
            with log.open('x', encoding='utf-8') as stream:
                completed = subprocess.run(runner_argv(protocol, work), cwd=work, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=max(1, min(FOLD_RESERVE, remaining)))
            if completed.returncode:
                status.update(status='FAILED', returncode=completed.returncode)
            else:
                receipt = read(Path('/kaggle/working') / tag(fold) / 'receipt.json')
                need(receipt['status'] == 'PASS' and receipt['fold'] == fold and
                     receipt['protocol_sha256'] == sha(protocol), 'runner receipt differs')
                status.update(status='COMPLETE', receipt_sha256=sha(Path('/kaggle/working') / tag(fold) / 'receipt.json'))
        except subprocess.TimeoutExpired:
            status['status'] = 'TIMED_OUT'
        except Exception as exc:
            status.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc)[:200])
        status['elapsed_seconds'] = round(time.monotonic() - started_fold, 1)
        save(_status_path(fold), status)
        results.append(status)
        print(json.dumps({'fold': fold, 'status': status['status'],
                          'elapsed_seconds': status['elapsed_seconds']}), flush=True)
    return results


def run(work, corpus, manifest, started):
    need(sys.platform.startswith('linux') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
         'Kaggle runtime configuration required')
    validate_package(work, corpus, manifest)
    # nvidia-smi is a subprocess, so the coordinator never initialises CUDA.
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    need(len(names) == 2 and all('T4' in n for n in names), 'Kaggle 2xT4 required')
    children = []
    for worker_id in (0, 1):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker_id), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        args = [sys.executable, str(Path(work) / 'tools/exp162_dino_folds_kaggle.py'), 'worker',
                '--work', str(work), '--corpus', str(corpus), '--manifest', str(manifest),
                '--started', str(started), '--worker-id', str(worker_id)]
        stream = (Path('/kaggle/working') / f'exp162-worker{worker_id}.log').open('x', encoding='utf-8')
        children.append((subprocess.Popen(args, cwd=work, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT), stream))
    exit_codes = []
    for child, stream in children:
        try:
            exit_codes.append(child.wait(timeout=max(1, MAX_WALL-120-(time.monotonic()-started))))
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(); exit_codes.append(124)
        finally:
            stream.close()
    statuses = {}
    for fold in range(1, 5):
        path = _status_path(fold)
        if not path.is_file():
            save(path, {'schema': SCHEMA, 'status': 'NO_STATUS', 'fold': fold,
                        'worker': next(i for i, folds in enumerate(ASSIGNMENT) if fold in folds)})
        statuses[str(fold)] = read(path)
    result = {'schema': SCHEMA, 'status': 'COMPLETE' if all(x == 0 for x in exit_codes) and
              all(s['status'] == 'COMPLETE' for s in statuses.values()) else 'INCOMPLETE',
              'worker_exit_codes': exit_codes, 'folds': statuses,
              'elapsed_seconds': round(time.monotonic() - started, 1)}
    save('/kaggle/working/exp162-status.json', result)
    return result


# Methods consumed by the staged copy of C's EXP158 runner. Imports stay lazy.
sha256 = sha
AGGREGATION_POLICY = 'dinov2-normalized-patch256-existing-mil-v1'
POSITION_POLICY = 'canonical518-trainable-cpu-fp32-bicubic-aa-v2'
CODE_NAMES = ('raptor_train.py', 'raptor_common.py', 'raptor_data.py', 'raptor_runtime.py')
_active_protocol = None


def old_guard():
    from tools import exp158_dino_strong
    return exp158_dino_strong


def validate_arguments(args, root):
    global _active_protocol
    root = Path(root).resolve()
    p = read(args.protocol)
    fold = p.get('fold')
    need(p.get('schema') == SCHEMA and fold in range(1, 5) and
         Path(args.protocol).resolve() == root / f'protocols/fold{fold}.json' and
         args.tag == tag(fold) and args.phase == 'full', 'wrong EXP162 fold protocol')
    for key, expected in p['expected_args'].items():
        need(getattr(args, key, None) == expected, 'training argument differs: ' + key)
    for key, value in p['paths'].items():
        need(Path(getattr(args, key)).resolve() == (root / value).resolve(), 'input path differs: ' + key)
    need(p['expected_args']['seed'] == 42 and p['expected_args']['epochs'] == p['expected_args']['fixed_epoch'] == 20
         and p['expected_args']['k'] == 12 and p['expected_args']['k_eval'] == 36
         and p['expected_args']['amp'] == p['expected_args']['eval_amp'] == 'fp16'
         and p['arm']['initial_state_sha256'] == INITIAL_SHA, 'EXP158 recipe changed')
    ids = read(root / p['ordered_ids']['path'])
    need(sha(root / p['ordered_ids']['path']) == p['ordered_ids']['sha256'] and
         len(ids['val']) == FOLD_COUNTS[fold] and len(ids['train']) == 4343-FOLD_COUNTS[fold],
         'fold roster differs')
    _active_protocol = p
    return p, ids


def arm_for(protocol, name):
    need(name == protocol['tag'], 'wrong PATCH arm')
    return protocol['arm']


def build_offline_model(arch, asset):
    need(_active_protocol is not None and arch == 'vit_small_patch14_dinov2.lvd142m',
         'admitted fold required before model build')
    model = old_guard().build_offline_model(arch, asset)
    from safetensors.torch import load_file
    path = ROOT / 'storage/weights/vast_hc06/full_initial_state.safetensors'
    need(sha(path) == FULL_SHA, 'canonical full initial state differs')
    model.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    return model


def read_rows(path):
    return old_guard().read_rows(path)


def worker_init(index):
    return old_guard().worker_init(index)


def install_gold_blocker(root):
    return old_guard().install_gold_blocker(root)


def validate_membership(*args):
    return old_guard().validate_membership(*args)


def validate_label_contract(label_hash, eval_hash, split_hash, protocol):
    need(label_hash == eval_hash == protocol['label_hash'] and
         split_hash == protocol['split_hash'], 'label/split semantic hashes differ')


def loss_weights(baseline, protocol, arm):
    import numpy as np
    need(baseline.dtype == np.float32 and baseline.shape == (12,) and
         np.isfinite(baseline).all() and ((baseline >= 1) & (baseline <= 10)).all() and
         arm['loss_policy'] == 'clipped_positive_term', 'fold-local T-only loss weights invalid')
    return baseline


def uid_hash(ids):
    return old_guard().uid_hash(ids)


def validate_pretraining(provenance, env, protocol, arm):
    need(provenance['initial_state_sha256'] == INITIAL_SHA and
         provenance['initial_asset_sha256'] == WEIGHT_SHA and
         provenance['aggregation_policy'] == AGGREGATION_POLICY and
         provenance['position_policy'] == POSITION_POLICY and
         provenance['protocol_sha256'] == sha(ROOT / f"protocols/fold{protocol['fold']}.json") and
         provenance['corpus_manifest_sha256'] == CORPUS_SHA and
         provenance['aug1_sha256'] == protocol['aug1_sha256'] and
         provenance['aug1_triplet_sha256'] == protocol['aug1_triplet_sha256'] and
         provenance['train_ids_ordered_sha256'] == uid_hash(read(ROOT / protocol['ordered_ids']['path'])['train']) and
         provenance['eval_ids_ordered_sha256'] == uid_hash(read(ROOT / protocol['ordered_ids']['path'])['val']) and
         provenance['checkpoint_selection'] == 'fixed-final' and
         provenance['gold_access'] == 'disabled', 'pretraining provenance differs')
    for key in ('runtime', 'eval_runtime'):
        r = provenance[key]
        need(r['requested_amp'] == r['effective_amp'] == 'fp16' and r['policy'] == 'strict'
             and r['device'] == 'cuda' and r['compute_capability'] == [7, 5]
             and not r['precision_fallback_allowed'], 'T4 fp16 runtime differs')
    need(provenance['code_sha256'] == protocol['training_code_sha256'], 'frozen training source differs')
    provenance.update(remote_environment={'gpu': 'Tesla T4', 'device': os.environ.get('CUDA_VISIBLE_DEVICES')},
                      arm='PATCH', gpu_family='T4')


def check_prefix_trace(trace):
    need(_active_protocol is not None and len(trace) == 64, 'missing fold trace')
    save(ROOT / 'artifacts/experiments' / _active_protocol['tag'] / 'trace_prefix.json',
         {'schema': SCHEMA, 'epochs': 8, 'training_trace_sha256': trace})


def finish_full(out, protocol, provenance, trace, elapsed):
    out = Path(out)
    summary, history = read(out / 'summary.json'), read(out / 'history.json')
    fold = protocol['fold']
    updates = protocol['arm']['expected_attempts']
    need(summary['fold'] == fold and summary['train'] == 4343-FOLD_COUNTS[fold] and
         summary['val'] == FOLD_COUNTS[fold] and summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == updates and summary['skipped_updates'] == 0 and
         summary['gold_access'] == 'disabled' and summary['weak_oof_macro_auc'] is None and
         len(history) == 20 and all(h['epoch'] == i and h['optimizer_updates'] == (i+1)*(updates//20)
                                    and h['skipped_updates'] == 0 for i, h in enumerate(history)),
         'incomplete fixed-final fold training')
    ids = read(ROOT / protocol['ordered_ids']['path'])['val']
    _prediction_values(out / 'oof_best.csv', ids)
    need(len(trace) == 64 and read(out / 'trace_prefix.json')['epochs'] == 8 and
         provenance['initial_state_sha256'] == INITIAL_SHA, 'training trace/state missing')
    save(out / 'trace.json', {'schema': SCHEMA, 'fold': fold, 'epochs': 20,
                             'updates': updates, 'training_trace_sha256': trace})
    outputs = {name: {'sha256': sha(out / name), 'bytes': (out / name).stat().st_size}
               for name in ('summary.json', 'provenance.json', 'history.json', 'trace.json',
                            'trace_prefix.json', 'model_best.pt', 'oof_best.csv')}
    save(out / 'receipt.json', {'schema': SCHEMA, 'status': 'PASS', 'fold': fold, 'seed': 42,
         'protocol_sha256': sha(ROOT / f'protocols/fold{fold}.json'),
         'initial_state_sha256': INITIAL_SHA, 'gpu_family': 'T4', 'amp': 'fp16',
         'updates': updates, 'skips': 0, 'validation_rows': len(ids),
         'elapsed_seconds': elapsed, 'outputs': outputs})


def _prediction_values(path, ids):
    import numpy as np
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [UID, *TARGETS], 'OOF target columns differ')
        rows = list(reader)
    need([r[UID] for r in rows] == list(ids) and len(set(ids)) == len(ids),
         'OOF ordered study IDs differ')
    values = np.asarray([[float(r[t]) for t in TARGETS] for r in rows], np.float64)
    need(values.shape == (len(ids), 12) and np.isfinite(values).all() and
         ((values >= 0) & (values <= 1)).all(), 'OOF probabilities invalid')
    return values


def _verify_fold(run, fold, root=ROOT):
    root, run = Path(root), Path(run)
    ids = roster(root)[fold][1]['val']
    receipt, summary, provenance, history = (read(run / n) for n in
                                             ('receipt.json', 'summary.json', 'provenance.json', 'history.json'))
    updates = ((4343-FOLD_COUNTS[fold]) // 8) * 20
    need(receipt['schema'] == SCHEMA and receipt['status'] == 'PASS' and
         receipt['fold'] == fold and receipt['seed'] == 42 and receipt['gpu_family'] == 'T4' and
         receipt['amp'] == 'fp16' and receipt['initial_state_sha256'] == INITIAL_SHA and
         receipt['updates'] == updates and receipt['skips'] == 0 and
         receipt['validation_rows'] == len(ids), 'fold receipt differs')
    need(summary['fold'] == fold and summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == updates and summary['skipped_updates'] == 0 and
         summary['val'] == len(ids) and summary['train'] == 4343-len(ids) and
         summary['gold_access'] == 'disabled' and summary['weak_oof_macro_auc'] is None,
         'fold training summary differs')
    need(len(history) == 20 and all(h['epoch'] == i and
         h['optimizer_updates'] == (i+1)*(updates//20) and h['skipped_updates'] == 0
         for i, h in enumerate(history)), 'fold history incomplete')
    need(provenance['initial_state_sha256'] == INITIAL_SHA and
         provenance['aggregation_policy'] == AGGREGATION_POLICY and
         provenance['gpu_family'] == 'T4' and
         provenance['runtime']['effective_amp'] == 'fp16' and
         provenance['runtime']['compute_capability'] == [7, 5] and
         provenance['train_ids_ordered_sha256'] == uid_hash(roster(root)[fold][1]['train']) and
         provenance['eval_ids_ordered_sha256'] == uid_hash(ids),
         'fold provenance differs')
    need(set(receipt['outputs']) == {'summary.json', 'provenance.json', 'history.json',
         'trace.json', 'trace_prefix.json', 'model_best.pt', 'oof_best.csv'}, 'fold output inventory differs')
    for name, record in receipt['outputs'].items():
        target = run / name
        need(target.is_file() and target.stat().st_size == record['bytes'] and
             sha(target) == record['sha256'], 'fold output hash differs: ' + name)
    _prediction_values(run / 'oof_best.csv', ids)
    return receipt


def collect(source, stage_dir=STAGE, destination=EXP / 'outputs', root=ROOT):
    source, stage_dir, destination, root = map(Path, (source, stage_dir, destination, root))
    need(not destination.exists(), 'fresh collection directory required')
    manifest_path = stage_dir / 'dataset/manifest.json'
    need(sha(manifest_path) == read(stage_dir / 'stage-receipt.json')['manifest_sha256'],
         'stage manifest changed')
    m = read(manifest_path)
    need(m['schema'] == SCHEMA and m['version'] == '1.0.0', 'wrong stage')
    status = read(source / 'exp162-status.json')
    need(status['schema'] == SCHEMA and set(status['folds']) == {'1', '2', '3', '4'},
         'Kaggle coordinator status missing')
    destination.mkdir(parents=True)
    completed, failures = [], {}
    for fold in range(1, 5):
        fold_status = read(source / f'exp162-fold{fold}-status.json')
        need(fold_status == status['folds'][str(fold)] and fold_status['fold'] == fold,
             'fold/coordinator status disagree')
        shutil.copyfile(source / f'exp162-fold{fold}-status.json',
                        destination / f'exp162-fold{fold}-status.json')
        if fold_status['status'] != 'COMPLETE':
            failures[str(fold)] = fold_status['status']
            continue
        run = source / tag(fold)
        need(sha(run / 'receipt.json') == fold_status['receipt_sha256'], 'fold receipt hash differs')
        receipt = _verify_fold(run, fold, root)
        need(receipt['protocol_sha256'] ==
             m['members'][f'protocols/fold{fold}.json']['sha256'],
             'fold protocol differs from staged dataset')
        out = destination / tag(fold)
        out.mkdir()
        for name in (*receipt['outputs'], 'receipt.json'):
            shutil.copyfile(run / name, out / name)
        completed.append(fold)
    shutil.copyfile(source / 'exp162-status.json', destination / 'exp162-status.json')
    result = {'schema': SCHEMA, 'status': 'VERIFIED' if len(completed) == 4 else 'INCOMPLETE',
              'completed_folds': completed, 'incomplete_folds': failures,
              'stage_manifest_sha256': sha(manifest_path),
              'coordinator_sha256': sha(destination / 'exp162-status.json')}
    save(destination / 'collection.json', result)
    return result


def _score(pred, ref, evaluator):
    import numpy as np
    a = evaluator.auc_targets(ref[0], pred, ref[1], ref[2])
    macro = evaluator.macro(a)
    return {'macro': float(macro) if np.isfinite(macro) else None,
            'per_target': {name: (float(value) if np.isfinite(value) else None)
                           for name, value in zip(TARGETS, a)}}


def _paired(primary, baseline, refs, draws, evaluator):
    import numpy as np
    details, mean_draws = {}, []
    for name, (y, mask, minimum) in refs.items():
        ac = evaluator.auc_targets(y, primary, mask, minimum)
        bc = evaluator.auc_targets(y, baseline, mask, minimum)
        if name in ('jev', 'aquila'):
            need(np.isfinite(ac).all() and np.isfinite(bc).all(),
                 'primary 12-target AUC undefined')
        values = []
        for ix in draws:
            ai = evaluator.auc_targets(y[ix], primary[ix], mask[ix], minimum)
            bi = evaluator.auc_targets(y[ix], baseline[ix], mask[ix], minimum)
            values.append(float(evaluator.macro(ai) - evaluator.macro(bi)))
        values = np.asarray(values)
        if name in ('jev', 'aquila'):
            need(np.isfinite(values).all(), 'primary bootstrap AUC undefined')
            mean_draws.append(values)
        finite = values[np.isfinite(values)]
        ca, ba = evaluator.macro(ac), evaluator.macro(bc)
        delta = ca-ba
        details[name] = {'candidate': float(ca) if np.isfinite(ca) else None,
                         'baseline': float(ba) if np.isfinite(ba) else None,
                         'delta': float(delta) if np.isfinite(delta) else None,
                         'ci95': np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
                         'per_target_delta': {t: float(a-b) if np.isfinite(a-b) else None
                                              for t, a, b in zip(TARGETS, ac, bc)}}
    paired = (mean_draws[0] + mean_draws[1]) / 2
    point = (details['jev']['delta'] + details['aquila']['delta']) / 2
    ci = np.percentile(paired, [2.5, 97.5]).tolist()
    return {'references': details, 'primary_mean_delta': point, 'primary_mean_ci95': ci,
            'pass': bool(point > 0 and ci[0] > 0)}


def screen(outputs=EXP / 'outputs', destination=EXP / 'screen', root=ROOT):
    outputs, destination, root = map(Path, (outputs, destination, root))
    need(not destination.exists(), 'fresh screen directory required')
    collected = read(outputs / 'collection.json')
    need(collected['schema'] == SCHEMA, 'wrong EXP162 collection')
    if collected['status'] != 'VERIFIED':
        result = {'schema': SCHEMA, 'verdict': 'INCOMPLETE',
                  'completed_folds': collected['completed_folds'],
                  'incomplete_folds': collected['incomplete_folds']}
        save(destination / 'result.json', result)
        return result
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp156_k_sweep as e156
    folds = roster(root)
    ids = [uid for fold in range(5) for uid in folds[fold][1]['val']]
    need(len(ids) == len(set(ids)) == 4343, 'pooled five-fold roster differs')
    exp158 = root / '.local/exp158-collected-v1'
    old_receipt = read(exp158 / 'collect-receipt.json')
    need(old_receipt['status'] == 'COLLECTED_VALIDATED' and
         old_receipt['prediction_sha256'] == sha(exp158 / 'oof_best.csv') and
         read(exp158 / 'provenance.json')['initial_state_sha256'] == INITIAL_SHA,
         'EXP158 fold-0 collection changed')
    fold_pred, fold_hash = {}, {}
    fold_pred[0] = _prediction_values(exp158 / 'oof_best.csv', folds[0][1]['val'])
    fold_hash['0'] = sha(exp158 / 'oof_best.csv')
    for fold in range(1, 5):
        run = outputs / tag(fold)
        _verify_fold(run, fold, root)
        fold_pred[fold] = _prediction_values(run / 'oof_best.csv', folds[fold][1]['val'])
        fold_hash[str(fold)] = sha(run / 'oof_best.csv')
    dino = np.concatenate([fold_pred[i] for i in range(5)])
    destination.mkdir(parents=True)
    with (destination / 'pooled_oof.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([UID, *TARGETS])
        for study, row in zip(ids, dino):
            writer.writerow([study, *(format(float(v), '.9g') for v in row)])
    refs = e156.references(ids, root)
    refs = {'jev': refs['jev_hybrid'], 'aquila': refs['aquila_v1'],
            'dread': refs['dread'], 'consensus': refs['consensus']}
    standalone = {'pooled': {name: _score(dino, ref, ev) for name, ref in refs.items()},
                  'per_fold': {str(f): {name: _score(fold_pred[f],
                        tuple(x[sum(FOLD_COUNTS[i] for i in range(f)):sum(FOLD_COUNTS[i] for i in range(f+1))]
                              if j < 2 else x for j, x in enumerate(ref)), ev)
                        for name, ref in refs.items()} for f in range(5)}}
    partners, hashes = {}, {}
    for name in ('radr50', 'h1'):
        arrays = []
        for fold in range(5):
            uid = folds[fold][1]['val']
            if name == 'radr50':
                tg = f'EXP127_AUG1_S42_F0' if fold == 0 else f'EXP134_AUG1_S42_F{fold}'
            else:
                tg = f'EXP149_H1_S42_F{fold}' if fold < 2 else f'EXP153_H1_S42_F{fold}'
            path = root / 'artifacts/experiments' / tg / 'oof_k36.csv'
            inventory = read(path.parent / 'output-inventory.json')['oof_k36.csv']
            need(path.stat().st_size == inventory['bytes'] and sha(path) == inventory['sha256'],
                 'partner OOF inventory differs: ' + tg)
            arrays.append(_prediction_values(path, uid))
            hashes[f'{name}_f{fold}'] = sha(path)
        partners[name] = np.concatenate(arrays)
    draws = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    pairs = {}
    for name, base in partners.items():
        ranked_base, ranked_dino = ev.pct_rank(base), ev.pct_rank(dino)
        blend = .9*ranked_base + .1*ranked_dino
        pairs[name] = _paired(blend, ranked_base, refs, draws, ev)
    result = {'schema': SCHEMA, 'verdict': 'PASS' if pairs['radr50']['pass'] else 'FAIL',
              'rows': len(ids), 'folds': list(range(5)), 'seed': 42,
              'bootstrap': {'draws': 300, 'seed': 20261003, 'unit': 'validation component'},
              'weight_dino': .1, 'standalone': standalone, 'partners': pairs,
              'prediction_sha256': fold_hash, 'partner_sha256': hashes,
              'pooled_oof_sha256': sha(destination / 'pooled_oof.csv'),
              'reference_sha256': {n: sha(root / p) for n, p in
                   (('jev', JEV), ('aquila', AQUILA), ('dread', DREAD),
                    *[(f'consensus_{i}', p) for i, p in enumerate(CONSENSUS)])},
              'cross_gpu_caveat': 'fold0 RTX3090; folds1-4 T4; DINO device drift unchecked'}
    save(destination / 'result.json', result)
    return {'verdict': result['verdict'], 'rows': len(ids),
            'radr50_primary_mean_delta': pairs['radr50']['primary_mean_delta'],
            'radr50_primary_mean_ci95': pairs['radr50']['primary_mean_ci95'],
            'h1_primary_mean_delta': pairs['h1']['primary_mean_delta'],
            'result': str(destination / 'result.json')}


def push_plan(out=STAGE):
    out = Path(out).resolve()
    need((out / 'dataset/manifest.json').is_file() and (out / 'kernel/run.ipynb').is_file(),
         'stage first')
    return {'dataset': DATASET, 'kernel': KERNEL, 'commands': [
        f'Set-Location -LiteralPath "{out / "dataset"}"',
        'kaggle datasets create -p .',
        f'Set-Location -LiteralPath "{out / "kernel"}"',
        'kaggle kernels push -p .',
        f'Set-Location -LiteralPath "{ROOT}"',
        f'kaggle kernels output {KERNEL} -p "{out / "download"}"',
        f'python -m tools.exp162_dino_folds_kaggle collect --source "{out / "download"}" --stage-dir "{out}"',
        'python -m tools.exp162_dino_folds_kaggle screen']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('stage'); s.add_argument('--out', type=Path, default=STAGE)
    s = sub.add_parser('push-plan'); s.add_argument('--out', type=Path, default=STAGE)
    s = sub.add_parser('collect'); s.add_argument('--source', type=Path, required=True)
    s.add_argument('--stage-dir', type=Path, default=STAGE)
    s.add_argument('--destination', type=Path, default=EXP / 'outputs')
    s = sub.add_parser('screen'); s.add_argument('--outputs', type=Path, default=EXP / 'outputs')
    s.add_argument('--destination', type=Path, default=EXP / 'screen')
    for action in ('run', 'worker'):
        s = sub.add_parser(action)
        s.add_argument('--work', type=Path, required=True)
        s.add_argument('--corpus', type=Path, required=True)
        s.add_argument('--manifest', type=Path, required=True)
        s.add_argument('--started', type=float, required=True)
        if action == 'worker':
            s.add_argument('--worker-id', type=int, required=True)
    a = p.parse_args(argv)
    if a.action == 'stage': result = stage(a.out)
    elif a.action == 'push-plan': result = push_plan(a.out)
    elif a.action == 'collect': result = collect(a.source, a.stage_dir, a.destination)
    elif a.action == 'screen': result = screen(a.outputs, a.destination)
    elif a.action == 'run': result = run(a.work, a.corpus, a.manifest, a.started)
    else: result = worker(a.work, a.corpus, a.manifest, a.started, a.worker_id)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
