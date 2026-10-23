"""EXP156 v1.1.0: offline stage, Kaggle two-T4 inference, local collection/screen.

No command here contacts Kaggle or submits. B alone publishes and launches the
staged private dataset/kernel. `run` and `worker` are notebook-only entrypoints.
"""
from __future__ import annotations

import argparse
import ast
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
HERE = ROOT / 'tools'
EXP = ROOT / 'artifacts/experiments/EXP156_K_SWEEP'
DEFAULT_OUT = ROOT / '.local/exp156-k-sweep'
DATASET = 'taeyangg4/knee-exp156-k-sweep-assets'
KERNEL = 'taeyangg4/knee-exp156-k-sweep'
CORPUS = 'taeyangg4/knee-raptor-corpus-336'
WEIGHTS = 'marwanmath/resnet-50-radimagenet-marwan'
WEIGHTS_SHA = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
CORPUS_SHA = 'f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
K_VALUES = (36, 62)
SEEDS = (42,)
WORKER_FOLDS = ((0, 2, 4), (1, 3))
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
UID = 'StudyInstanceUID'
PROTOCOL0 = 'artifacts/experiments/EXP127_AUG1_PROTOCOL_V1'
PROTOCOL14 = 'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1'
COMPONENTS = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
JEV = 'storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv'
AQUILA = 'storage/labels/derived_constellation/aquila_v1.csv'
DREAD = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
CONSENSUS = (DREAD,
             'storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
             'storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv',
             'storage/labels/derived_relabel/relabel_v1.csv')
MAX_WALL = 75 * 60


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def safe_name(name):
    p = PurePosixPath(name)
    need(name and p.as_posix() == name and not p.is_absolute() and
         all(x not in ('.', '..') for x in p.parts) and ':' not in name and '\\' not in name,
         'Unsafe archive path')
    return name


def tag(seed, fold):
    need(seed in SEEDS and fold in range(5), 'Wrong seed/fold')
    return (f'EXP127_AUG1_S{seed}_F0' if fold == 0 else
            f'EXP134_AUG1_S{seed}_F{fold}')


def protocol(seed, fold):
    return f'{PROTOCOL0 if fold == 0 else PROTOCOL14}/s{seed}/manifest.json'


def _function_source(path, names):
    source = Path(path).read_text(encoding='utf-8')
    found = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in names}
    need(set(found) == set(names), 'Frozen serving function missing')
    return '\n\n'.join(found[n] for n in names)


def components_source(root):
    paths = (root / 'training/src/raptor_common.py', root / 'training/src/raptor_radimagenet.py',
             root / 'tools/exp124_flip_tta.py')
    body = 'import numpy as np\n\n'
    body += _function_source(paths[0], ('select_centers',)) + '\n\n'
    body += _function_source(paths[1], ('center3_windows_from', 'normalise')) + '\n\n'
    body += _function_source(paths[2], ('plane_ranges', 'flip_mask')) + '\n'
    compile(body, 'exp156_components.py', 'exec')
    return body.encode('utf-8')


def inventory(root=ROOT):
    root = Path(root)
    entries, files = [], {}
    for seed in SEEDS:
        for fold in range(5):
            p = protocol(seed, fold)
            config = read(root / p)
            need(config['corpus_manifest_sha256'] == CORPUS_SHA, 'Corpus contract differs')
            fold_data = config['folds'][str(fold)]
            ids_path = fold_data['ordered_ids']['path']
            need(sha(root / ids_path) == fold_data['ordered_ids']['sha256'], 'Fold roster hash differs')
            ids = read(root / ids_path)['val']
            need(len(ids) == fold_data['n_val'] and len(set(ids)) == len(ids), 'Fold roster invalid')
            name = tag(seed, fold)
            run = root / 'artifacts/experiments' / name
            original = __import__('exp140_aug1_release').checkpoint_entry(root, seed, fold)
            oof = run / 'oof_k36.csv'
            stored = read(run / 'output-inventory.json')['oof_k36.csv']
            need(sha(oof) == stored['sha256'] and oof.stat().st_size == stored['bytes'],
                 'Stored K36 OOF differs: ' + name)
            need(original['protocol_sha256'] == sha(root / p), 'Checkpoint protocol differs')
            files[f'protocols/s{seed}_f{fold}.json'] = root / p
            files[f'folds/f{fold}.json'] = root / ids_path
            files[f'control/s{seed}_f{fold}.csv'] = oof
            files[f'checkpoints/s{seed}_f{fold}.pt'] = run / 'model_best.pt'
            entries.append(dict(seed=seed, fold=fold, tag=name, n_val=len(ids),
                                protocol_sha256=sha(root / p), roster_sha256=sha(root / ids_path),
                                checkpoint_sha256=original['sha256'], stored_oof_sha256=stored['sha256']))
    need(sum(e['n_val'] for e in entries) == 4343, 'Registered seed-42 roster differs')
    return entries, files


def pack(path, members):
    need(not Path(path).exists() and str(path).endswith('.zip.bin'), 'Fresh .zip.bin required')
    need(len(members) == len(set(members)), 'Duplicate archive entries')
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, source in members.items():
            safe_name(name)
            if isinstance(source, bytes):
                z.writestr(name, source)
            else:
                z.write(source, name)
    return sha(path)


def notebook(manifest_sha):
    cell = ("STARTED = __import__('time').monotonic()\n"
            "import os\n"
            "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
            "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
            f'EXPECTED_MANIFEST_SHA = {manifest_sha!r}\n' + '''import hashlib, json, pathlib, subprocess, sys, time, zipfile
assert sys.platform.startswith('linux'), 'Kaggle Linux required'
def file_sha(path):
    d = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): d.update(b)
    return d.hexdigest()
def mount(slug, marker):
    roots = [pathlib.Path('/kaggle/input') / slug,
             pathlib.Path('/kaggle/input/datasets') / ('taeyangg4' if slug != 'resnet-50-radimagenet-marwan' else 'marwanmath') / slug]
    hits = [p for p in roots if (p / marker).is_file()]
    assert len(hits) == 1, (slug, hits)
    return hits[0]
assets = mount('knee-exp156-k-sweep-assets', 'manifest.json')
corpus = mount('knee-raptor-corpus-336', 'manifest.json')
weights = mount('resnet-50-radimagenet-marwan', 'ResNet50.pt')
m = json.loads((assets / 'manifest.json').read_text())
assert file_sha(assets / 'manifest.json') == EXPECTED_MANIFEST_SHA
assert m['schema'] == 'exp156-k-sweep-v1'
assert file_sha(corpus / 'manifest.json') == m['corpus_sha256']
assert file_sha(weights / 'ResNet50.pt') == m['radimagenet_weights_sha256']
assert time.monotonic() - STARTED < 600, 'Mount preflight too slow'
work = pathlib.Path('/tmp/exp156-work')
work.mkdir(exist_ok=False)
for archive_name, entries in (('code.zip.bin', m['code']), ('checkpoints.zip.bin', m['assets'])):
    archive = assets / archive_name
    assert file_sha(archive) == m['archives'][archive_name]
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and set(names) == set(entries)
        for name in names:
            p = pathlib.PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
            dst = work / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, dst.open('xb') as out:
                import shutil
                shutil.copyfileobj(src, out)
            assert dst.stat().st_size == entries[name]['bytes'] and file_sha(dst) == entries[name]['sha256']
(work / 'config').mkdir()
(work / 'config/workspace.toml').write_text('# isolated EXP156 workspace\\n')
env = dict(os.environ, KNEE_WORKSPACE_ROOT=str(work), PYTHONUTF8='1', OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
cmd = [sys.executable, str(work / 'tools/exp156_k_sweep.py'), 'run', '--work', str(work),
       '--corpus', str(corpus), '--manifest', str(assets / 'manifest.json'),
       '--started', str(STARTED)]
remaining = 75 * 60 - (time.monotonic() - STARTED)
assert remaining > 0, 'EXP156 preflight exhausted wall guard'
p = subprocess.run(cmd, env=env, timeout=remaining)
assert p.returncode == 0, 'EXP156 incomplete; inspect status.json; do not score partial output'
''')
    compile(cell, '<EXP156 notebook>', 'exec')
    return dict(nbformat=4, nbformat_minor=5,
                metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
                cells=[{'cell_type': 'markdown', 'metadata': {}, 'source':
                        '# EXP156 v1.1 registered seed-42 K36/K62 sweep on two T4s'},
                       {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                        'outputs': [], 'source': cell}])


def stage(out=DEFAULT_OUT, root=ROOT):
    out, root = Path(out).resolve(), Path(root)
    need(not out.exists(), 'Use a fresh stage directory')
    need((root / 'artifacts/experiments/EXP156_K_SWEEP/PROTOCOL.md').is_file(), 'Register protocol first')
    if str(root / 'tools') not in sys.path:
        sys.path.insert(0, str(root / 'tools'))
    entries, asset_files = inventory(root)
    code_files = {f'tools/{name}': root / 'tools' / name for name in ('exp156_k_sweep.py',)}
    code_files.update({f'training/src/{name}': root / 'training/src' / name
                       for name in ('raptor_common.py', 'raptor_data.py', 'raptor_radimagenet.py', 'raptor_runtime.py')})
    code_files['exp156_components.py'] = components_source(root)
    (out / 'dataset').mkdir(parents=True)
    (out / 'kernel').mkdir()
    def receipt(members):
        return {name: dict(sha256=(hashlib.sha256(src).hexdigest() if isinstance(src, bytes) else sha(src)),
                           bytes=(len(src) if isinstance(src, bytes) else Path(src).stat().st_size))
                for name, src in members.items()}
    manifest = dict(schema='exp156-k-sweep-v1', version='1.1.0', plan='H-B-56',
                    protocol_sha256=sha(root / 'artifacts/experiments/EXP156_K_SWEEP/PROTOCOL.md'),
                    dataset=DATASET, kernel=KERNEL, corpus=CORPUS, weights=WEIGHTS,
                    corpus_sha256=CORPUS_SHA, radimagenet_weights_sha256=WEIGHTS_SHA,
                    k_values=list(K_VALUES), seeds=list(SEEDS), max_wall_seconds=MAX_WALL,
                    entries=entries, code=receipt(code_files), assets=receipt(asset_files), archives={})
    manifest['archives']['code.zip.bin'] = pack(out / 'dataset/code.zip.bin', code_files)
    manifest['archives']['checkpoints.zip.bin'] = pack(out / 'dataset/checkpoints.zip.bin', asset_files)
    write_new(out / 'dataset/manifest.json', manifest)
    write_new(out / 'dataset/dataset-metadata.json', dict(id=DATASET, title='EXP156 K sweep assets',
              isPrivate=True, licenses=[{'name': 'other'}],
              description='Private AUG1 fold checkpoints and frozen inference code; no reference labels.'))
    write_new(out / 'kernel/kernel-metadata.json', dict(id=KERNEL, title='knee-exp156-k-sweep',
              code_file='run.ipynb', language='python', kernel_type='notebook', is_private=True,
              enable_gpu=True, enable_tpu=False, machine_shape='NvidiaTeslaT4', enable_internet=False,
              dataset_sources=[DATASET, CORPUS, WEIGHTS], competition_sources=[], kernel_sources=[], model_sources=[]))
    write_new(out / 'kernel/run.ipynb', notebook(sha(out / 'dataset/manifest.json')))
    result = dict(status='STAGED', dataset=DATASET, kernel=KERNEL, checkpoints=len(entries),
                  stage=str(out), manifest_sha256=sha(out / 'dataset/manifest.json'),
                  notebook_sha256=sha(out / 'kernel/run.ipynb'), network_calls=0, gpu_runs=0)
    write_new(out / 'stage-receipt.json', result)
    return result


def csv_values(path, ids):
    import numpy as np
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [UID, *TARGETS], 'Prediction columns differ')
        rows = list(reader)
    need([r[UID] for r in rows] == list(ids) and len(set(ids)) == len(ids), 'Prediction UID order differs')
    values = np.asarray([[float(r[t]) for t in TARGETS] for r in rows], np.float64)
    need(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'Invalid prediction values')
    return values


def write_csv(path, ids, values):
    import numpy as np
    a = np.asarray(values)
    need(a.shape == (len(ids), 12) and np.isfinite(a).all() and len(set(ids)) == len(ids), 'Invalid OOF')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        w = csv.writer(stream)
        w.writerow([UID, *TARGETS])
        for uid, row in zip(ids, a):
            w.writerow([uid, *(format(float(v), '.9g') for v in row)])


def validated_manifest(work, manifest_path):
    m = read(manifest_path)
    need(m['schema'] == 'exp156-k-sweep-v1' and m['version'] == '1.1.0' and
         m['k_values'] == list(K_VALUES) and m['seeds'] == list(SEEDS) and
         len(m['entries']) == 5 and sum(e['n_val'] for e in m['entries']) == 4343 and
         {e['fold'] for e in m['entries']} == set(range(5)) and
         all(e['seed'] == 42 for e in m['entries']), 'Manifest contract differs')
    for group in ('code', 'assets'):
        for name, record in m[group].items():
            safe_name(name)
            p = Path(work) / name
            need(p.stat().st_size == record['bytes'] and sha(p) == record['sha256'],
                 'Extracted input differs: ' + name)
    return m


def one_mount(slug, marker, owner):
    roots = (Path('/kaggle/input') / slug, Path('/kaggle/input/datasets') / owner / slug)
    hits = [p for p in roots if (p / marker).is_file()]
    need(len(hits) == 1, 'Expected exactly one mounted path for ' + slug)
    return hits[0]


def _load_model(torch, build_model, path, entry):
    ck = torch.load(path, map_location='cpu', weights_only=False)  # own SHA-bound checkpoint
    expected = dict(arch='resnet50', res=224, lab=list(TARGETS), contract='raptor64_336_c140_s02',
                    norm='radimagenet', window_policy='within-slot-valid', input_policy='center3',
                    tag=entry['tag'], fold=entry['fold'])
    need(all(ck.get(k) == v for k, v in expected.items()), 'Checkpoint metadata differs')
    need(ck['provenance']['protocol_sha256'] == entry['protocol_sha256'], 'Checkpoint protocol differs')
    model = build_model('resnet50', pretrained=False)
    model.load_state_dict(ck['model'], strict=True)
    del ck
    return model.eval().to('cuda')


def worker(worker_id, work, corpus_path, manifest_path, started):
    """A fixed subset of seed-42 folds on one T4; K values share encoded centres."""
    import numpy as np
    import torch
    import timm
    import cv2
    work, corpus_path = Path(work), Path(corpus_path)
    need(worker_id in range(len(WORKER_FOLDS)) and
         os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Worker environment differs')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == str(worker_id) and torch.cuda.device_count() == 1 and
         'T4' in torch.cuda.get_device_name(0), 'One T4 per worker required')
    m = validated_manifest(work, manifest_path)
    need(sha(corpus_path / 'manifest.json') == CORPUS_SHA, 'Corpus manifest differs')
    sys.path.insert(0, str(work / 'training/src'))
    sys.path.insert(0, str(work))
    import exp156_components as c
    from raptor_common import Corpus, build_model
    from raptor_runtime import configure_runtime
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    configure_runtime('fp16', 'cuda', 'strict')
    corpus = Corpus(corpus_path)
    need(corpus.contract == 'raptor64_336_c140_s02', 'Corpus contract differs')
    ranges = c.plane_ranges(corpus.spec)
    seed = 42
    out = Path('/kaggle/working') / f'exp156-worker{worker_id}'
    out.mkdir(exist_ok=False)
    all_ids, timings, unique_stats, diffs, outputs = [], {}, {}, {}, {}
    for fold in WORKER_FOLDS[worker_id]:
        entry = next(e for e in m['entries'] if (e['seed'], e['fold']) == (seed, fold))
        ids = read(work / f'folds/f{fold}.json')['val']
        need(len(ids) == entry['n_val'] and sha(work / f'folds/f{fold}.json') == entry['roster_sha256'],
             'Worker fold roster differs')
        need(set(ids) <= set(corpus.ids) and not (set(ids) & set(all_ids)), 'Corpus/fold coverage differs')
        model = _load_model(torch, build_model, work / f'checkpoints/s{seed}_f{fold}.pt', entry)
        fold_preds = {k: [] for k in K_VALUES}
        fold_seconds = {str(k): 0.0 for k in K_VALUES}
        direct_seconds = {str(k): 0.0 for k in K_VALUES}
        shared_seconds = 0.0
        fold_unique = {str(k): [] for k in K_VALUES}
        plain36 = []
        with torch.inference_mode():
            for i, uid in enumerate(ids):
                need(time.monotonic() - started < MAX_WALL - 120, 'EXP156 wall guard: incomplete OOF')
                mask = corpus.mask(uid)
                centers = {k: c.select_centers(mask, corpus.depth, k, corpus.spec,
                                               policy='within-slot-valid') for k in K_VALUES}
                union = list(dict.fromkeys(x for k in K_VALUES for x in centers[k]))
                index = {x: j for j, x in enumerate(union)}
                flags = c.flip_mask(union, ranges)
                vol = corpus.volume(uid)
                x = c.normalise(c.center3_windows_from(vol, union, 224), 'radimagenet')
                xm = x.copy()
                xm[flags] = xm[flags, ..., ::-1]
                scores = {k: [] for k in K_VALUES}
                for view in (x, xm):
                    shared_started = time.monotonic()
                    xt = torch.from_numpy(np.ascontiguousarray(view)[None]).to('cuda')
                    with torch.autocast('cuda', dtype=torch.float16):
                        # Bound activation memory on a 16 GB T4. Attention still
                        # sees every K-specific centre in its registered order.
                        feats = torch.cat([model.encode(xt[:, j:j+12])
                                           for j in range(0, xt.shape[1], 12)], dim=1)
                        torch.cuda.synchronize()
                        shared_seconds += time.monotonic() - shared_started
                        for k in K_VALUES:
                            head_started = time.monotonic()
                            take = [index[v] for v in centers[k]]
                            logits = model.head(feats[:, take])
                            need(tuple(logits.shape) == (1, 12) and bool(torch.isfinite(logits).all()),
                                 'Invalid model output')
                            scores[k].append(torch.sigmoid(logits.float())[0].cpu().numpy())
                            torch.cuda.synchronize()
                            fold_seconds[str(k)] += time.monotonic() - head_started
                    del xt, feats
                for k in K_VALUES:
                    fold_preds[k].append((scores[k][0] + scores[k][1]) / 2)
                    fold_unique[str(k)].append(len(set(centers[k]) - {-1}))
                plain36.append(scores[36][0])
                if i < 2:
                    # Direct K-specific probe gives a comparable measured time
                    # without charging the entire cohort three separate encodes.
                    for k in K_VALUES:
                        probe_start = time.monotonic()
                        xx = c.normalise(c.center3_windows_from(vol, centers[k], 224), 'radimagenet')
                        mm = xx.copy()
                        ff = c.flip_mask(centers[k], ranges)
                        mm[ff] = mm[ff, ..., ::-1]
                        direct = []
                        for view in (xx, mm):
                            xt = torch.from_numpy(np.ascontiguousarray(view)[None]).to('cuda')
                            with torch.autocast('cuda', dtype=torch.float16):
                                features = torch.cat([model.encode(xt[:, j:j+12])
                                                      for j in range(0, xt.shape[1], 12)], dim=1)
                                direct.append(torch.sigmoid(model.head(features).float())[0].cpu().numpy())
                        torch.cuda.synchronize()
                        direct_seconds[str(k)] += time.monotonic() - probe_start
                        need(np.max(np.abs((direct[0]+direct[1])/2 - fold_preds[k][-1])) <= .002,
                             'Shared-feature/direct-K prediction parity differs')
                if (i + 1) % 100 == 0:
                    print(f'EXP156 seed {seed} fold {fold}: {i+1}/{len(ids)} elapsed {time.monotonic()-started:.1f}s', flush=True)
        for k in K_VALUES:
            arr = np.asarray(fold_preds[k], np.float32)
            fold_unique[str(k)] = dict(min=min(fold_unique[str(k)]), max=max(fold_unique[str(k)]),
                                       mean=float(np.mean(fold_unique[str(k)])))
        stored = csv_values(work / f'control/s{seed}_f{fold}.csv', ids)
        diff = float(np.max(np.abs(np.asarray(plain36) - stored)))
        need(diff <= .002, f'K36 plain control mismatch seed={seed} fold={fold} diff={diff:.6g}')
        diffs[str(fold)] = diff
        for k in K_VALUES:
            name = f'oof_k{k}_f{fold}.csv'
            write_csv(out / name, ids, np.asarray(fold_preds[k], np.float32))
            outputs[name] = sha(out / name)
        timings[str(fold)] = dict(head_seconds=fold_seconds, shared_encode_seconds=shared_seconds,
                                  direct_two_study_seconds=direct_seconds)
        unique_stats[str(fold)] = fold_unique
        all_ids += ids
        del model
        torch.cuda.empty_cache()
    need(len(all_ids) == sum(e['n_val'] for e in m['entries']
                             if e['fold'] in WORKER_FOLDS[worker_id]), 'Worker rows differ')
    receipt = dict(schema='exp156-worker-v1', status='COMPLETE', seed=seed,
                   worker_id=worker_id, assigned_folds=list(WORKER_FOLDS[worker_id]), rows=len(all_ids),
                   folds=timings, unique_centers=unique_stats, plain_k36_max_abs_diff=diffs,
                   elapsed_seconds=time.monotonic()-started, gpu=torch.cuda.get_device_name(0),
                   torch=str(torch.__version__), timm=timm.__version__, outputs=outputs)
    write_new(out / 'receipt.json', receipt)
    return receipt


def run(work, corpus, manifest, started):
    """GPU-free supervisor; CUDA_VISIBLE_DEVICES is set in each child before torch import."""
    work, corpus = Path(work), Path(corpus)
    m = validated_manifest(work, manifest)
    need(m['corpus_sha256'] == sha(corpus / 'manifest.json'), 'Corpus differs')
    status_path = Path('/kaggle/working/exp156-status.json')
    write_new(status_path, dict(state='RUNNING', started=started, max_wall_seconds=MAX_WALL))
    processes = []
    try:
        for worker_id in range(len(WORKER_FOLDS)):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker_id), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       KNEE_WORKSPACE_ROOT=str(work), PYTHONPATH=str(work / 'training/src'))
            log = (Path('/kaggle/working') / f'exp156-worker{worker_id}.log').open('x', encoding='utf-8')
            cmd = [sys.executable, str(work / 'tools/exp156_k_sweep.py'), 'worker', '--worker-id', str(worker_id),
                   '--work', str(work), '--corpus', str(corpus), '--manifest', str(manifest),
                   '--started', str(started)]
            processes.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        while any(p.poll() is None for p, _ in processes):
            need(all(p.poll() in (None, 0) for p, _ in processes), 'Worker failed; inspect worker logs')
            need(time.monotonic() - started < MAX_WALL - 30, 'EXP156 wall guard exceeded')
            time.sleep(3)
        need(all(p.returncode == 0 for p, _ in processes), 'Worker failed; inspect worker logs')
        receipts = [read(Path('/kaggle/working') / f'exp156-worker{i}/receipt.json')
                    for i in range(len(WORKER_FOLDS))]
        need(all(r['status'] == 'COMPLETE' and r['seed'] == 42 and r['worker_id'] == i and
                 r['assigned_folds'] == list(WORKER_FOLDS[i]) and 'T4' in r['gpu']
                 for i, r in enumerate(receipts)), 'Worker receipts incomplete')
        out = Path('/kaggle/working/exp156-seed42')
        out.mkdir(exist_ok=False)
        pooled = {k: [] for k in K_VALUES}
        all_ids = []
        for fold in range(5):
            worker_id = next(i for i, folds in enumerate(WORKER_FOLDS) if fold in folds)
            source = Path('/kaggle/working') / f'exp156-worker{worker_id}'
            receipt = receipts[worker_id]
            entry = next(e for e in m['entries'] if e['fold'] == fold)
            ids = read(work / f'folds/f{fold}.json')['val']
            need(len(ids) == entry['n_val'] and not (set(ids) & set(all_ids)) and
                 str(fold) in receipt['plain_k36_max_abs_diff'] and
                 receipt['plain_k36_max_abs_diff'][str(fold)] <= .002,
                 'Fold roster/control differs at merge')
            all_ids.extend(ids)
            for k in K_VALUES:
                name = f'oof_k{k}_f{fold}.csv'
                path = source / name
                need(sha(path) == receipt['outputs'][name], 'Worker output hash differs')
                pooled[k].append(csv_values(path, ids))
        need(len(all_ids) == 4343 and sum(r['rows'] for r in receipts) == 4343,
             'Merged seed-42 roster differs')
        import numpy as np
        for k in K_VALUES:
            write_csv(out / f'oof_k{k}_s42.csv', all_ids, np.concatenate(pooled[k]))
        fold_timings = {fold: timing for r in receipts for fold, timing in r['folds'].items()}
        pooled_timing = dict(shared_encode_seconds=sum(t['shared_encode_seconds']
                                                       for t in fold_timings.values()),
                             head_seconds={str(k): sum(t['head_seconds'][str(k)]
                                                       for t in fold_timings.values()) for k in K_VALUES})
        merged_receipt = dict(schema='exp156-merged-v1', status='COMPLETE', seed=42,
                              rows=len(all_ids), workers=[r['worker_id'] for r in receipts],
                              assigned_folds={str(i): list(folds) for i, folds in enumerate(WORKER_FOLDS)},
                              folds=fold_timings, pooled_timing=pooled_timing,
                              unique_centers={f: v for r in receipts for f, v in
                                              r['unique_centers'].items()},
                              plain_k36_max_abs_diff={f: v for r in receipts for f, v in
                                                      r['plain_k36_max_abs_diff'].items()},
                              gpu_names=[r['gpu'] for r in receipts],
                              outputs={f'oof_k{k}_s42.csv': sha(out / f'oof_k{k}_s42.csv')
                                       for k in K_VALUES})
        write_new(out / 'receipt.json', merged_receipt)
        status = dict(state='COMPLETE', elapsed_seconds=time.monotonic()-started,
                      billed_gpu_hours_upper_bound=2*(time.monotonic()-started)/3600,
                      rows=len(all_ids), gpu_names=[r['gpu'] for r in receipts],
                      torch=[r['torch'] for r in receipts], timm=[r['timm'] for r in receipts])
    except BaseException as exc:
        status = dict(state='INCOMPLETE', reason=type(exc).__name__ + ': ' + str(exc),
                      elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        for p, log in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
            log.close()
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return status


def collect(source, destination=EXP / 'outputs', stage_dir=DEFAULT_OUT):
    source, destination, stage_dir = Path(source), Path(destination), Path(stage_dir)
    need(not destination.exists(), 'Use a fresh collection destination')
    status = read(source / 'exp156-status.json')
    need(status['state'] == 'COMPLETE' and status['rows'] == 4343,
         'Kaggle session incomplete; refuse partial OOF')
    m = read(stage_dir / 'dataset/manifest.json')
    need(sha(stage_dir / 'dataset/manifest.json') == read(stage_dir / 'stage-receipt.json')['manifest_sha256'],
         'Stage manifest differs')
    need(m['version'] == '1.1.0' and m['seeds'] == list(SEEDS) and
         m['k_values'] == list(K_VALUES), 'Staged sweep scope differs')
    expected = {s: [e for e in m['entries'] if e['seed'] == s] for s in SEEDS}
    for seed in SEEDS:
        run_dir = source / f'exp156-seed{seed}'
        receipt = read(run_dir / 'receipt.json')
        need(receipt['status'] == 'COMPLETE' and receipt['seed'] == seed and
             receipt['rows'] == sum(e['n_val'] for e in expected[seed]) == status['rows'] and
             len(receipt['gpu_names']) == 2 and all('T4' in g for g in receipt['gpu_names']) and
             receipt['assigned_folds'] == {str(i): list(folds) for i, folds in enumerate(WORKER_FOLDS)} and
             len(receipt['plain_k36_max_abs_diff']) == 5 and
             max(receipt['plain_k36_max_abs_diff'].values()) <= .002,
             'Seed receipt/roster/control differs')
        ids = [u for e in expected[seed] for u in read(ROOT /
               read(ROOT / protocol(seed, e['fold']))['folds'][str(e['fold'])]['ordered_ids']['path'])['val']]
        need(len(set(ids)) == len(ids), 'Pooled folds overlap')
        for k in K_VALUES:
            name = f'oof_k{k}_s{seed}.csv'
            need(sha(run_dir / name) == receipt['outputs'][name], 'OOF hash differs')
            csv_values(run_dir / name, ids)
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns('*.log', 'exp156-work'))
    write_new(destination / 'collection.json', dict(schema='exp156-collection-v1', status='VERIFIED',
              source=str(source), stage_manifest_sha256=sha(stage_dir / 'dataset/manifest.json'),
              rows=status['rows'], seeds=list(SEEDS), k_values=list(K_VALUES)))
    return dict(status='VERIFIED', rows=status['rows'], destination=str(destination))


def _reference(path, ids, rule):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    a = ev.load_soft(path, ids)
    need(np.isfinite(a).all(), 'Reference has missing values: ' + str(path))
    if rule == 'binary':
        need(np.isin(a, (0, 1)).all(), 'Reference is not binary')
        return a
    if rule == 'rint':
        return np.rint(a.astype(np.float32)).astype(np.float64)
    return (a >= .5).astype(np.float64)


def references(ids, root=ROOT):
    import numpy as np
    root = Path(root)
    ones = np.ones((len(ids), 12), bool)
    refs = {'jev_hybrid': (_reference(root / JEV, ids, 'binary'), ones, 1),
            'aquila_v1': (_reference(root / AQUILA, ids, 'ge'), ones, 1),
            'dread': (_reference(root / DREAD, ids, 'rint'), ones, 1)}
    arrays = np.stack([_reference(root / p, ids, 'ge') for p in CONSENSUS])
    agree = arrays.all(0) | (~arrays.astype(bool)).all(0)
    refs['consensus'] = (arrays[0], agree, 2)
    return refs


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def timing_gate(path, k):
    if path is None:
        return dict(pass_=False, reason='Full B3/H1 serving timing not supplied')
    t = read(path)
    need(t['schema'] == 'exp156-serving-time-v1' and t['units'] == 'seconds' and
         t['cohort'] == 'full competition test' and set(t['releases']) == {'B3', 'H1'},
         'Serving timing evidence differs')
    detail = {}
    for release in ('B3', 'H1'):
        r = t['releases'][release]
        value = r[str(k)]
        need(value > 0 and r['36'] > 0 and r['reserve'] >= 0 and r['source'],
             'Serving timing incomplete')
        detail[release] = dict(control_seconds=r['36'], candidate_seconds=value,
                               reserve_seconds=r['reserve'], within_budget=value+r['reserve'] <= 9*3600)
    return {'pass_': all(v['within_budget'] for v in detail.values()), 'releases': detail,
            'evidence_sha256': sha(path)}


def screen(outputs=EXP / 'outputs', timing=None, root=ROOT, destination=EXP / 'screen'):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp125_radr50_k16_local as e125
    outputs, root, destination = Path(outputs), Path(root), Path(destination)
    collection = read(outputs / 'collection.json')
    need(not destination.exists() and collection['status'] == 'VERIFIED' and
         collection['seeds'] == list(SEEDS) and collection['k_values'] == list(K_VALUES),
         'Collect complete output first; use a fresh screen directory')
    ids = [u for fold in range(5) for u in read(root /
           read(root / protocol(42, fold))['folds'][str(fold)]['ordered_ids']['path'])['val']]
    need(len(set(ids)) == len(ids) == collection['rows'] == 4343,
         'Screen fold roster differs')
    preds = {}
    hashes = {}
    for seed in SEEDS:
        for k in K_VALUES:
            path = outputs / f'exp156-seed{seed}/oof_k{k}_s{seed}.csv'
            preds[seed, k] = csv_values(path, ids)
            hashes[f's{seed}_k{k}'] = sha(path)
    ranked = {k: ev.pct_rank(preds[42, k]) for k in K_VALUES}
    refs = references(ids, root)
    draws = e125.component_draws(ids, read(root / COMPONENTS), draws=300, seed=20261003)
    pairs, gates = {}, {}
    for k in (62,):
        pairs[str(k)] = {name: _clean(ev.compare(ranked[k], ranked[36], ref, draws))
                         for name, ref in refs.items()}
        primary = (pairs[str(k)]['jev_hybrid']['delta'] + pairs[str(k)]['aquila_v1']['delta']) / 2
        accuracy = primary > .001 and pairs[str(k)]['dread']['delta'] >= -.002
        cost = timing_gate(timing, k)
        gates[str(k)] = dict(primary_mean_delta=primary, accuracy_pass=accuracy,
                             dread_delta=pairs[str(k)]['dread']['delta'], timing=cost,
                             adopt=accuracy and cost['pass_'])
    passing = [k for k in (62,) if gates[str(k)]['adopt']]
    chosen = min(passing, key=lambda k: max(v['candidate_seconds'] for v in
                 gates[str(k)]['timing']['releases'].values())) if passing else None
    verdict = f'ADOPT_K{chosen}' if chosen else ('HOLD_TIMING' if timing is None and any(
        gates[str(k)]['accuracy_pass'] for k in (62,)) else 'KEEP_K36')
    destination.mkdir(parents=True)
    result = dict(schema='exp156-screen-v1', verdict=verdict, rows=len(ids),
                  bootstrap=dict(draws=300, seed=20261003, unit='validation component'),
                  registered_rule='seed42 rank; mean(JEV,aquila)>+0.001; Dread>=-0.002; B3/H1 full serving <=9h',
                  prediction_sha256=hashes, reference_sha256={n: sha(root / p) for n, p in
                  [('jev_hybrid', JEV), ('aquila_v1', AQUILA), ('dread', DREAD),
                   *[(f'consensus_{i}', p) for i, p in enumerate(CONSENSUS)]]},
                  pairs=pairs, gates=gates)
    write_new(destination / 'result.json', result)
    return dict(verdict=verdict, rows=len(ids), result=str(destination / 'result.json'))


def push_plan(out=DEFAULT_OUT):
    out = Path(out).resolve()
    need((out / 'dataset/manifest.json').is_file() and (out / 'kernel/run.ipynb').is_file(), 'Stage first')
    return dict(dataset=DATASET, kernel=KERNEL, commands=[
        f'Set-Location -LiteralPath "{out / "dataset"}"',
        'kaggle datasets create -p .',
        f'Set-Location -LiteralPath "{out / "kernel"}"',
        'kaggle kernels push -p .',
        f'Set-Location -LiteralPath "{ROOT}"',
        f'kaggle kernels output {KERNEL} -p "{out / "download"}"',
        f'python tools/exp156_k_sweep.py collect --source "{out / "download"}" --stage-dir "{out}"',
        'python tools/exp156_k_sweep.py screen'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('stage'); s.add_argument('--out', type=Path, default=DEFAULT_OUT)
    s = sub.add_parser('push-plan'); s.add_argument('--out', type=Path, default=DEFAULT_OUT)
    s = sub.add_parser('collect'); s.add_argument('--source', type=Path, required=True)
    s.add_argument('--destination', type=Path, default=EXP / 'outputs')
    s.add_argument('--stage-dir', type=Path, default=DEFAULT_OUT)
    s = sub.add_parser('screen'); s.add_argument('--outputs', type=Path, default=EXP / 'outputs')
    s.add_argument('--timing', type=Path); s.add_argument('--destination', type=Path, default=EXP / 'screen')
    for action in ('run', 'worker'):
        s = sub.add_parser(action)
        s.add_argument('--work', type=Path, required=True); s.add_argument('--corpus', type=Path, required=True)
        s.add_argument('--manifest', type=Path, required=True); s.add_argument('--started', type=float, required=True)
        if action == 'worker': s.add_argument('--worker-id', type=int, required=True)
    a = p.parse_args(argv)
    if a.action == 'stage': r = stage(a.out)
    elif a.action == 'push-plan': r = push_plan(a.out)
    elif a.action == 'collect': r = collect(a.source, a.destination, a.stage_dir)
    elif a.action == 'screen': r = screen(a.outputs, a.timing, destination=a.destination)
    elif a.action == 'run': r = run(a.work, a.corpus, a.manifest, a.started)
    else: r = worker(a.worker_id, a.work, a.corpus, a.manifest, a.started)
    print(json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
