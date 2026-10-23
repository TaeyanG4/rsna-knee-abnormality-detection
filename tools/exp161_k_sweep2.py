"""EXP161 v1.0.0: offline stage/collect/screen and private two-T4 sweep worker.

No public command here contacts Kaggle. B alone creates the private dataset and
pushes the kernel. The H1 weights are reused from the EXP154 private dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from tools import exp156_k_sweep as base

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'artifacts/experiments/EXP161_K_SWEEP2'
OUT = ROOT / '.local/exp161-k-sweep2'
DATASET = 'taeyangg4/knee-exp161-k-sweep2-assets'
KERNEL = 'taeyangg4/knee-exp161-k-sweep2'
H1_DATASET = 'taeyangg4/knee-exp154-b3-h1-assets'
H1_LOCAL = ROOT / '.local/exp154-b3-h1-release/dataset'
K_VALUES = (36, 62, 94)
FAMILIES = ('aug1', 'h1')
HEADS = ('dread', 'relabel_v1', 'steven_v4', 'pilkwang_v2', 'aquila_v1')
MAX_WALL = 100 * 60  # B amendment A1 2026-10-09: v1 hit the 44-min guard (worker0 needed ≈3,500 s)
DIRECT_N = 8  # each fold/family; measured direct-K serving samples
SCHEMA = 'exp161-k-sweep2-v1'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return base.read(path)


def sha(path):
    return base.sha(path)


def write_new(path, value):
    base.write_new(path, value)


def h1_tag(fold):
    return f'{"EXP149" if fold < 2 else "EXP153"}_H1_S42_F{fold}'


def h1_source_manifest(root=ROOT):
    """Bind only the five seed-42 entries of the already staged EXP154 assets."""
    source = Path(root) / '.local/exp154-b3-h1-release/dataset'
    m = read(source / 'release-manifest.json')
    need(m['schema'] == 'exp154-b3-h1-release-v1' and
         m['dataset'] == H1_DATASET and m['heads'] == list(HEADS), 'EXP154 manifest differs')
    entries = [next(e for e in m['entries'] if e['seed'] == 42 and e['fold'] == fold)
               for fold in range(5)]
    for fold, e in enumerate(entries):
        need(e['tag'] == h1_tag(fold) and e['path'] == f'checkpoints/s42_f{fold}.pt' and
             e['heads'] == list(HEADS), 'H1 entry differs')
        local = Path(root) / e['source']
        need(local.stat().st_size == e['bytes'] and sha(local) == e['sha256'],
             'H1 local checkpoint differs: ' + e['tag'])
    need(sha(source / 'checkpoints.zip.bin') == m['archives']['checkpoints.zip.bin'],
         'EXP154 archive differs')
    return m, entries


def receipt(files):
    return {name: {'sha256': hashlib.sha256(src).hexdigest() if isinstance(src, bytes) else sha(src),
                   'bytes': len(src) if isinstance(src, bytes) else Path(src).stat().st_size}
            for name, src in files.items()}


def notebook(manifest_sha):
    # Runs in a fresh Kaggle process. CUDA is never queried in the coordinator.
    cell = f'''import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib, json, pathlib, shutil, subprocess, sys, time, zipfile
STARTED = time.monotonic()
EXPECTED = {manifest_sha!r}
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''): h.update(part)
    return h.hexdigest()
def mount(owner, slug, marker):
    roots = [pathlib.Path('/kaggle/input/datasets') / owner / slug,
             pathlib.Path('/kaggle/input') / slug]
    hits = [p for p in roots if (p / marker).is_file()]
    assert len(hits) == 1, (slug, hits)
    return hits[0]
assets = mount('taeyangg4', 'knee-exp161-k-sweep2-assets', 'manifest.json')
h1 = mount('taeyangg4', 'knee-exp154-b3-h1-assets', 'release-manifest.json')
corpus = mount('taeyangg4', 'knee-raptor-corpus-336', 'manifest.json')
weights = mount('marwanmath', 'resnet-50-radimagenet-marwan', 'ResNet50.pt')
assert digest(assets / 'manifest.json') == EXPECTED
m = json.loads((assets / 'manifest.json').read_text())
assert m['schema'] == 'exp161-k-sweep2-v1' and m['version'] == '1.0.0'
assert digest(h1 / 'release-manifest.json') == m['h1_manifest_sha256']
assert digest(h1 / 'checkpoints.zip.bin') == m['h1_archive_sha256']
assert digest(corpus / 'manifest.json') == m['corpus_sha256']
assert digest(weights / 'ResNet50.pt') == m['radimagenet_weights_sha256']
work = pathlib.Path('/tmp/exp161-work')
work.mkdir(exist_ok=False)
def extract(archive, expected, selected=None):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert set(names) == set(expected) if selected is None else set(selected) <= set(names)
        for name in names if selected is None else selected:
            p = pathlib.PurePosixPath(name)
            assert name == p.as_posix() and not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
            dst = work / ('h1/' + name if selected is not None else name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, dst.open('xb') as target: shutil.copyfileobj(source, target)
            assert dst.stat().st_size == expected[name]['bytes'] and digest(dst) == expected[name]['sha256']
for archive, group in [('code.zip.bin', 'code'), ('checkpoints.zip.bin', 'assets')]:
    assert digest(assets / archive) == m['archives'][archive]
    extract(assets / archive, m[group])
hm = json.loads((h1 / 'release-manifest.json').read_text())
assert hm['schema'] == 'exp154-b3-h1-release-v1' and hm['heads'] == m['heads']
assert hm['archives']['checkpoints.zip.bin'] == m['h1_archive_sha256']
selected = [e['path'] for e in m['h1_entries']]
for e in m['h1_entries']:
    assert next(x for x in hm['entries'] if x['seed'] == 42 and x['fold'] == e['fold'])['sha256'] == e['sha256']
extract(h1 / 'checkpoints.zip.bin', hm['checkpoints'], selected)
(work / 'config').mkdir()
(work / 'config/workspace.toml').write_text('# EXP161 isolated workspace\\n')
env = dict(os.environ, KNEE_WORKSPACE_ROOT=str(work), PYTHONPATH=str(work), PYTHONUTF8='1',
           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
cmd = [sys.executable, str(work / 'tools/exp161_k_sweep2.py'), 'run', '--work', str(work),
       '--corpus', str(corpus), '--manifest', str(assets / 'manifest.json'), '--started', str(STARTED)]
remaining = m['max_wall_seconds'] - (time.monotonic() - STARTED)
assert remaining > 0
p = subprocess.run(cmd, env=env, timeout=remaining)
assert p.returncode == 0, 'EXP161 incomplete: inspect exp161-status.json and worker logs'
'''
    compile(cell, '<EXP161 notebook>', 'exec')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
            'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': '# EXP161 registered K36/K62/K94 AUG1 and H1 held-out sweep'},
                      {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                       'outputs': [], 'source': cell}]}


def stage(out=OUT, root=ROOT):
    out, root = Path(out).resolve(), Path(root)
    need(not out.exists() and (root / 'artifacts/experiments/EXP161_K_SWEEP2/PROTOCOL.md').is_file(),
         'Fresh stage and registered protocol required')
    if str(root / 'tools') not in sys.path:
        sys.path.insert(0, str(root / 'tools'))
    aug, assets = base.inventory(root)
    hm, h1 = h1_source_manifest(root)
    need([e['fold'] for e in h1] == list(range(5)) and
         [e['fold'] for e in aug] == list(range(5)), 'Family fold coverage differs')
    from tools import exp151_bundle_release as e151
    code = {'tools/exp156_k_sweep.py': root / 'tools/exp156_k_sweep.py',
            'tools/exp161_k_sweep2.py': root / 'tools/exp161_k_sweep2.py',
            'exp156_components.py': base.components_source(root),
            'exp151_components.py': e151.components_source().encode('utf-8')}
    code.update({f'training/src/{name}': root / 'training/src' / name for name in
                 ('raptor_common.py', 'raptor_data.py', 'raptor_radimagenet.py', 'raptor_runtime.py')})
    (out / 'dataset').mkdir(parents=True)
    (out / 'kernel').mkdir()
    manifest = {'schema': SCHEMA, 'version': '1.0.0', 'plan': 'H-B-73',
                'protocol_sha256': sha(root / 'artifacts/experiments/EXP161_K_SWEEP2/PROTOCOL.md'),
                'dataset': DATASET, 'kernel': KERNEL, 'h1_dataset': H1_DATASET,
                'corpus_sha256': base.CORPUS_SHA, 'radimagenet_weights_sha256': base.WEIGHTS_SHA,
                'h1_manifest_sha256': sha(Path(root) / '.local/exp154-b3-h1-release/dataset/release-manifest.json'),
                'h1_archive_sha256': hm['archives']['checkpoints.zip.bin'],
                'heads': list(HEADS), 'h1_entries': h1, 'entries': aug,
                'k_values': list(K_VALUES), 'max_wall_seconds': MAX_WALL,
                'code': receipt(code), 'assets': receipt(assets), 'archives': {}}
    manifest['archives']['code.zip.bin'] = base.pack(out / 'dataset/code.zip.bin', code)
    manifest['archives']['checkpoints.zip.bin'] = base.pack(out / 'dataset/checkpoints.zip.bin', assets)
    write_new(out / 'dataset/manifest.json', manifest)
    write_new(out / 'dataset/dataset-metadata.json',
              {'id': DATASET, 'title': 'EXP161 K sweep2 assets', 'isPrivate': True,
               'licenses': [{'name': 'other'}],
               'description': 'Private AUG1 seed-42 checkpoints and sweep code; H1 reused from EXP154. No reference labels.'})
    write_new(out / 'kernel/kernel-metadata.json',
              {'id': KERNEL, 'title': KERNEL.split('/')[-1], 'code_file': 'run.ipynb',
               'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
               'enable_gpu': True, 'enable_tpu': False, 'machine_shape': 'NvidiaTeslaT4',
               'enable_internet': False, 'dataset_sources': [DATASET, H1_DATASET, base.CORPUS, base.WEIGHTS],
               'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
    write_new(out / 'kernel/run.ipynb', notebook(sha(out / 'dataset/manifest.json')))
    result = {'status': 'STAGED', 'stage': str(out), 'manifest_sha256': sha(out / 'dataset/manifest.json'),
              'dataset': DATASET, 'kernel': KERNEL, 'aug_checkpoints': 5, 'h1_reused_checkpoints': 5,
              'gpu_runs': 0, 'network_calls': 0}
    write_new(out / 'stage-receipt.json', result)
    return result


def validated_manifest(work, path):
    m = read(path)
    need(m['schema'] == SCHEMA and m['version'] == '1.0.0' and
         m['k_values'] == list(K_VALUES) and m['heads'] == list(HEADS) and
         m['max_wall_seconds'] == MAX_WALL and len(m['entries']) == len(m['h1_entries']) == 5,
         'EXP161 manifest contract differs')
    need([e['fold'] for e in m['entries']] == list(range(5)) and
         [e['fold'] for e in m['h1_entries']] == list(range(5)) and
         sum(e['n_val'] for e in m['entries']) == 4343, 'Family roster differs')
    for group in ('code', 'assets'):
        for name, record in m[group].items():
            base.safe_name(name)
            path = Path(work) / name
            need(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                 'Extracted input differs: ' + name)
    for e in m['h1_entries']:
        path = Path(work) / 'h1' / e['path']
        need(path.stat().st_size == e['bytes'] and sha(path) == e['sha256'],
             'H1 checkpoint differs: ' + e['tag'])
    return m


def _load_model(torch, build_model, multihead, path, entry, family):
    ck = torch.load(path, map_location='cpu', weights_only=False)  # hash-bound own checkpoint
    expect = {'arch': 'resnet50', 'res': 224, 'lab': list(base.TARGETS),
              'contract': 'raptor64_336_c140_s02', 'norm': 'radimagenet',
              'window_policy': 'within-slot-valid', 'input_policy': 'center3',
              'tag': entry['tag'], 'fold': entry['fold']}
    if family == 'h1':
        expect.update(heads=list(HEADS), label_set='h1', epoch=19)
    need(all(ck.get(k) == v for k, v in expect.items()), 'Checkpoint metadata differs')
    need(ck['provenance']['protocol_sha256'] == entry['protocol_sha256'],
         'Checkpoint protocol differs')
    if family == 'h1':
        need(ck['provenance']['full_model_initial_state_sha256'] ==
             'a56516a289178dce17d2fe736147ddbbcb52a571cb25e6c8b59b4342a3d3e636',
             'H1 initial state differs')
    model = build_model('resnet50', pretrained=False)
    if family == 'h1':
        model = multihead(model, len(HEADS))
        need({int(key.split('.')[1]) for key in ck['model'] if key.startswith('heads.')} == set(range(5)),
             'H1 state head inventory differs')
    model.load_state_dict(ck['model'], strict=True)
    del ck
    return model.eval().to('cuda')


def direct_probe(model, family, centers, vol, ranges, c, torch, np):
    """Direct K-specific two-view wall/GPU elapsed for one model and study."""
    wall_start = time.monotonic()
    x = c.normalise(c.center3_windows_from(vol, centers, 224), 'radimagenet')
    xm = x.copy()
    flags = c.flip_mask(centers, ranges)
    xm[flags] = xm[flags, ..., ::-1]
    torch.cuda.synchronize()
    gpu_start = time.monotonic()
    scores = []
    for view in (x, xm):
        xt = torch.from_numpy(np.ascontiguousarray(view)[None]).to('cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            feats = torch.cat([model.encode(xt[:, j:j+12])
                               for j in range(0, xt.shape[1], 12)], dim=1)
            logits = (torch.stack([head(feats) for head in model.heads], 1)
                      if family == 'h1' else model.head(feats))
        scores.append(torch.sigmoid(logits.float())[0].cpu().numpy())
        del xt, feats
    torch.cuda.synchronize()
    gpu_seconds = time.monotonic() - gpu_start
    return scores, {'gpu_seconds': gpu_seconds, 'wall_seconds': time.monotonic() - wall_start}


def worker(worker_id, work, corpus_path, manifest_path, started):
    import numpy as np
    import torch
    import timm
    import cv2
    work, corpus_path = Path(work), Path(corpus_path)
    need(worker_id in (0, 1) and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
         'Worker environment differs')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == str(worker_id) and
         torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0),
         'One T4 per worker required')
    m = validated_manifest(work, manifest_path)
    need(sha(corpus_path / 'manifest.json') == base.CORPUS_SHA, 'Corpus differs')
    sys.path.insert(0, str(work / 'training/src'))
    sys.path.insert(0, str(work))
    import exp156_components as c
    import exp151_components as h
    from raptor_common import Corpus, build_model
    from raptor_runtime import configure_runtime
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    configure_runtime('fp16', 'cuda', 'strict')
    corpus = Corpus(corpus_path)
    need(corpus.contract == 'raptor64_336_c140_s02', 'Corpus contract differs')
    ranges = c.plane_ranges(corpus.spec)
    out = Path('/kaggle/working') / f'exp161-worker{worker_id}'
    out.mkdir(exist_ok=False)
    folds, outputs, checks = {}, {}, {}
    with torch.inference_mode():
        for family in FAMILIES:
            for fold in base.WORKER_FOLDS[worker_id]:
                need(time.monotonic() - started < MAX_WALL - 150, 'EXP161 wall guard: incomplete OOF')
                entry = m['entries' if family == 'aug1' else 'h1_entries'][fold]
                ids = read(work / f'folds/f{fold}.json')['val']
                need(len(ids) == m['entries'][fold]['n_val'] and
                     sha(work / f'folds/f{fold}.json') == m['entries'][fold]['roster_sha256'] and
                     set(ids) <= set(corpus.ids), 'Fold roster differs')
                ck = (work / f'checkpoints/s42_f{fold}.pt' if family == 'aug1'
                      else work / 'h1' / f'checkpoints/s42_f{fold}.pt')
                model = _load_model(torch, build_model, h.build_multihead, ck, entry, family)
                preds = {k: [] for k in K_VALUES}
                plain36 = []
                direct = {str(k): {'gpu_seconds': 0.0, 'wall_seconds': 0.0, 'studies': 0}
                          for k in K_VALUES}
                head_seconds = {str(k): 0.0 for k in K_VALUES}
                shared_seconds = 0.0
                unique = {str(k): [] for k in K_VALUES}
                fold_wall_start = time.monotonic()
                probe_indices = {round(j * (len(ids) - 1) / (DIRECT_N - 1))
                                 for j in range(DIRECT_N)}
                need(len(probe_indices) == DIRECT_N, 'Direct probe indices differ')
                for i, uid in enumerate(ids):
                    need(time.monotonic() - started < MAX_WALL - 150,
                         'EXP161 wall guard: incomplete OOF')
                    mask = corpus.mask(uid)
                    centers = {k: c.select_centers(mask, corpus.depth, k, corpus.spec,
                                                   policy='within-slot-valid') for k in K_VALUES}
                    union = list(dict.fromkeys(v for k in K_VALUES for v in centers[k]))
                    index = {v: j for j, v in enumerate(union)}
                    flags = c.flip_mask(union, ranges)
                    vol = corpus.volume(uid)
                    x = c.normalise(c.center3_windows_from(vol, union, 224), 'radimagenet')
                    xm = x.copy()
                    xm[flags] = xm[flags, ..., ::-1]
                    scores = {k: [] for k in K_VALUES}
                    for view in (x, xm):
                        then = time.monotonic()
                        xt = torch.from_numpy(np.ascontiguousarray(view)[None]).to('cuda')
                        with torch.autocast('cuda', dtype=torch.float16):
                            feats = torch.cat([model.encode(xt[:, j:j+12])
                                               for j in range(0, xt.shape[1], 12)], dim=1)
                            torch.cuda.synchronize()
                            shared_seconds += time.monotonic() - then
                            for k in K_VALUES:
                                then = time.monotonic()
                                take = [index[v] for v in centers[k]]
                                logits = (torch.stack([head(feats[:, take]) for head in model.heads], 1)
                                          if family == 'h1' else model.head(feats[:, take]))
                                expect = (1, 5, 12) if family == 'h1' else (1, 12)
                                need(tuple(logits.shape) == expect and bool(torch.isfinite(logits).all()),
                                     'Model output differs')
                                scores[k].append(torch.sigmoid(logits.float())[0].cpu().numpy())
                                torch.cuda.synchronize()
                                head_seconds[str(k)] += time.monotonic() - then
                        del xt, feats
                    for k in K_VALUES:
                        preds[k].append(scores[k] if family == 'h1' else
                                        (scores[k][0] + scores[k][1]) / 2)
                        unique[str(k)].append(len(set(centers[k]) - {-1}))
                    plain36.append(scores[36][0])
                    if i in probe_indices:
                        for k in K_VALUES:
                            check, timing = direct_probe(model, family, centers[k], vol,
                                                         ranges, c, torch, np)
                            target = (np.asarray(scores[k]) if family == 'h1' else
                                      np.asarray(scores[k]))
                            need(np.max(np.abs(np.asarray(check) - target)) <= .002,
                                 'Shared/direct K parity differs')
                            for key in ('gpu_seconds', 'wall_seconds'):
                                direct[str(k)][key] += timing[key]
                            direct[str(k)]['studies'] += 1
                    if (i + 1) % 100 == 0:
                        print(f'EXP161 {family} fold {fold}: {i+1}/{len(ids)} '
                              f'elapsed {time.monotonic()-started:.1f}s', flush=True)
                if family == 'aug1':
                    control = base.csv_values(work / f'control/s42_f{fold}.csv', ids)
                    diff = float(np.max(np.abs(np.asarray(plain36) - control)))
                    need(diff <= .002, f'AUG1 K36 plain control mismatch {fold}: {diff}')
                    checks[str(fold)] = diff
                for k in K_VALUES:
                    name = f'{family}_k{k}_f{fold}'
                    if family == 'aug1':
                        path = out / f'{name}.csv'
                        base.write_csv(path, ids, np.asarray(preds[k], np.float32))
                    else:
                        path = out / f'{name}.npz'
                        arr = np.asarray(preds[k], np.float32)  # rows, views, heads, targets
                        need(arr.shape == (len(ids), 2, 5, 12), 'H1 raw shape differs')
                        np.savez(path, scores=arr)
                    outputs[path.name] = sha(path)
                folds[f'{family}_f{fold}'] = {
                    'rows': len(ids), 'wall_seconds': time.monotonic() - fold_wall_start,
                    'shared_encode_seconds': shared_seconds, 'head_seconds': head_seconds,
                    'direct': direct,
                    'unique_centers': {str(k): {'min': min(unique[str(k)]),
                                                'max': max(unique[str(k)]),
                                                'mean': float(np.mean(unique[str(k)]))} for k in K_VALUES}}
                shared_seconds = 0.0
                del model
                torch.cuda.empty_cache()
    result = {'schema': 'exp161-worker-v1', 'status': 'COMPLETE', 'worker_id': worker_id,
              'assigned_folds': list(base.WORKER_FOLDS[worker_id]), 'folds': folds,
              'aug_plain_k36_max_abs_diff': checks, 'outputs': outputs,
              'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
              'timm': timm.__version__, 'elapsed_seconds': time.monotonic() - started}
    write_new(out / 'receipt.json', result)
    return result


def run(work, corpus, manifest, started):
    """GPU-free coordinator; each child sees one T4 before importing torch."""
    work, corpus = Path(work), Path(corpus)
    m = validated_manifest(work, manifest)
    need(sha(corpus / 'manifest.json') == m['corpus_sha256'], 'Corpus manifest differs')
    status_path = Path('/kaggle/working/exp161-status.json')
    write_new(status_path, {'state': 'RUNNING', 'started': started, 'max_wall_seconds': MAX_WALL})
    processes = []
    status = None
    try:
        for worker_id in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker_id),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', KNEE_WORKSPACE_ROOT=str(work),
                       PYTHONPATH=str(work))
            log = (Path('/kaggle/working') / f'exp161-worker{worker_id}.log').open('x', encoding='utf-8')
            cmd = [sys.executable, str(work / 'tools/exp161_k_sweep2.py'), 'worker',
                   '--worker-id', str(worker_id), '--work', str(work), '--corpus', str(corpus),
                   '--manifest', str(manifest), '--started', str(started)]
            processes.append((subprocess.Popen(cmd, env=env, stdout=log,
                                               stderr=subprocess.STDOUT), log))
        while any(p.poll() is None for p, _ in processes):
            need(all(p.poll() in (None, 0) for p, _ in processes),
                 'Worker failed: inspect worker logs')
            need(time.monotonic() - started < MAX_WALL - 30, 'EXP161 wall guard exceeded')
            time.sleep(3)
        need(all(p.returncode == 0 for p, _ in processes), 'Worker failed')
        receipts = [read(Path('/kaggle/working') / f'exp161-worker{i}/receipt.json') for i in (0, 1)]
        need(all(r['status'] == 'COMPLETE' and r['worker_id'] == i and
                 r['assigned_folds'] == list(base.WORKER_FOLDS[i]) and 'T4' in r['gpu']
                 for i, r in enumerate(receipts)), 'Worker receipts differ')
        out = Path('/kaggle/working/exp161-pooled')
        out.mkdir(exist_ok=False)
        ids = [u for fold in range(5) for u in read(work / f'folds/f{fold}.json')['val']]
        need(len(ids) == len(set(ids)) == 4343, 'Pooled roster differs')
        import numpy as np
        outputs = {}
        for family in FAMILIES:
            for k in K_VALUES:
                parts = []
                for fold in range(5):
                    worker_id = next(i for i, fs in enumerate(base.WORKER_FOLDS) if fold in fs)
                    source = Path('/kaggle/working') / f'exp161-worker{worker_id}'
                    filename = f'{family}_k{k}_f{fold}.{ "csv" if family == "aug1" else "npz"}'
                    need(sha(source / filename) == receipts[worker_id]['outputs'][filename],
                         'Worker prediction hash differs')
                    fold_ids = read(work / f'folds/f{fold}.json')['val']
                    if family == 'aug1':
                        parts.append(base.csv_values(source / filename, fold_ids))
                    else:
                        with np.load(source / filename, allow_pickle=False) as z:
                            need(z.files == ['scores'], 'H1 archive entries differ')
                            arr = z['scores']
                        need(arr.shape == (len(fold_ids), 2, 5, 12) and np.isfinite(arr).all(),
                             'H1 raw predictions differ')
                        parts.append(arr)
                arr = np.concatenate(parts)
                filename = f'{family}_k{k}_s42.{ "csv" if family == "aug1" else "npz"}'
                path = out / filename
                if family == 'aug1':
                    base.write_csv(path, ids, arr)
                else:
                    np.savez(path, scores=arr)
                outputs[filename] = sha(path)
        timing = timing_summary(receipts)
        merged = {'schema': 'exp161-merged-v1', 'status': 'COMPLETE',
                  'rows_per_family': 4343, 'k_values': list(K_VALUES),
                  'families': list(FAMILIES), 'outputs': outputs,
                  'worker_receipt_sha256': {str(i): sha(Path('/kaggle/working') /
                                                         f'exp161-worker{i}/receipt.json') for i in (0, 1)},
                  'gpu_names': [r['gpu'] for r in receipts], 'timing': timing,
                  'plain_k36_max_abs_diff': {fold: value for r in receipts
                                             for fold, value in r['aug_plain_k36_max_abs_diff'].items()}}
        write_new(out / 'receipt.json', merged)
        elapsed = time.monotonic() - started
        status = {'state': 'COMPLETE', 'elapsed_seconds': elapsed,
                  'billed_gpu_hours_upper_bound': 2 * elapsed / 3600,
                  'rows_per_family': 4343, 'gpu_names': merged['gpu_names'],
                  'torch': [r['torch'] for r in receipts], 'timm': [r['timm'] for r in receipts]}
    except BaseException as exc:
        status = {'state': 'INCOMPLETE', 'reason': type(exc).__name__ + ': ' + str(exc),
                  'elapsed_seconds': time.monotonic() - started}
        raise
    finally:
        for p, log in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            log.close()
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return status


def timing_summary(receipts):
    """Direct per-model measurements stay separate from shared-union timings."""
    result = {}
    for family in FAMILIES:
        rows = [t for r in receipts for key, t in r['folds'].items()
                if key.startswith(family + '_')]
        need(len(rows) == 5 and sum(t['rows'] for t in rows) == 4343,
             'Timing fold coverage differs')
        direct = {}
        for k in K_VALUES:
            samples = sum(t['direct'][str(k)]['studies'] for t in rows)
            need(samples == DIRECT_N * 5, 'Direct timing sample differs')
            direct[str(k)] = {
                'studies': samples, 'models_per_study': 1,
                'gpu_seconds_per_model_study': sum(t['direct'][str(k)]['gpu_seconds'] for t in rows) / samples,
                'wall_seconds_per_model_study': sum(t['direct'][str(k)]['wall_seconds'] for t in rows) / samples,
                'shared_head_seconds_per_model_study': sum(t['head_seconds'][str(k)] for t in rows) / 4343}
        direct_extra = sum(t['direct'][str(k)]['wall_seconds'] for t in rows for k in K_VALUES)
        result[family] = {'studies': 4343, 'models': 5,
                          'shared_union_wall_seconds_per_study':
                              (sum(t['wall_seconds'] for t in rows) - direct_extra) / 4343,
                          'shared_encode_seconds_per_study': sum(t['shared_encode_seconds'] for t in rows) / 4343,
                          'direct_probe_extra_wall_seconds': direct_extra,
                          'direct': direct}
    return result


def collect(source, destination=EXP / 'outputs', stage_dir=OUT, root=ROOT):
    source, destination, stage_dir, root = map(Path, (source, destination, stage_dir, root))
    need(not destination.exists(), 'Fresh collection destination required')
    status = read(source / 'exp161-status.json')
    need(status['state'] == 'COMPLETE' and status['rows_per_family'] == 4343 and
         status['billed_gpu_hours_upper_bound'] <= 2 * MAX_WALL / 3600, 'Kernel incomplete or over budget')  # B A2: budget follows the A1 wall
    m = read(stage_dir / 'dataset/manifest.json')
    need(m['schema'] == SCHEMA and sha(stage_dir / 'dataset/manifest.json') ==
         read(stage_dir / 'stage-receipt.json')['manifest_sha256'], 'Stage manifest differs')
    r = read(source / 'exp161-pooled/receipt.json')
    need(r['schema'] == 'exp161-merged-v1' and r['status'] == 'COMPLETE' and
         r['rows_per_family'] == 4343 and len(r['gpu_names']) == 2 and
         all('T4' in g for g in r['gpu_names']) and
         max(r['plain_k36_max_abs_diff'].values()) <= .002, 'Merged receipt differs')
    for worker_id, assigned in enumerate(base.WORKER_FOLDS):
        work_dir = source / f'exp161-worker{worker_id}'
        worker_receipt = work_dir / 'receipt.json'
        need(sha(worker_receipt) == r['worker_receipt_sha256'][str(worker_id)],
             'Worker receipt hash differs')
        wr = read(worker_receipt)
        need(wr['schema'] == 'exp161-worker-v1' and wr['status'] == 'COMPLETE' and
             wr['worker_id'] == worker_id and wr['assigned_folds'] == list(assigned) and
             wr['gpu'] == r['gpu_names'][worker_id] and
             set(wr['aug_plain_k36_max_abs_diff']) == {str(f) for f in assigned} and
             max(wr['aug_plain_k36_max_abs_diff'].values()) <= .002 and
             set(wr['folds']) == {f'{family}_f{fold}' for family in FAMILIES for fold in assigned},
             'Worker coverage/control differs')
        expected = {f'{family}_k{k}_f{fold}.{ "csv" if family == "aug1" else "npz"}'
                    for family in FAMILIES for k in K_VALUES for fold in assigned}
        need(set(wr['outputs']) == expected, 'Worker output inventory differs')
        for name, digest in wr['outputs'].items():
            need(sha(work_dir / name) == digest, 'Worker output hash differs: ' + name)
    ids = [u for fold in range(5) for u in read(root / base.read(root / base.protocol(42, fold))
           ['folds'][str(fold)]['ordered_ids']['path'])['val']]
    need(len(ids) == len(set(ids)) == 4343, 'Collection roster differs')
    import numpy as np
    for family in FAMILIES:
        for k in K_VALUES:
            name = f'{family}_k{k}_s42.{ "csv" if family == "aug1" else "npz"}'
            path = source / 'exp161-pooled' / name
            need(sha(path) == r['outputs'][name], 'Pooled output hash differs')
            if family == 'aug1':
                base.csv_values(path, ids)
            else:
                with np.load(path, allow_pickle=False) as z:
                    need(z.files == ['scores'] and z['scores'].shape == (4343, 2, 5, 12) and
                         np.isfinite(z['scores']).all(), 'H1 output differs')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('*.log'))
    write_new(destination / 'collection.json',
              {'schema': 'exp161-collection-v1', 'status': 'VERIFIED',
               'stage_manifest_sha256': sha(stage_dir / 'dataset/manifest.json'),
               'rows': 4343, 'families': list(FAMILIES), 'k_values': list(K_VALUES),
               'billed_gpu_hours_upper_bound': status['billed_gpu_hours_upper_bound']})
    return {'status': 'VERIFIED', 'destination': str(destination), 'rows': 4343}


def h1_served_oof(raw):
    """Five-head B3-H1 reduction on the honest one-model-per-study OOF pool."""
    import numpy as np
    from scipy.stats import rankdata
    a = np.asarray(raw, np.float64)
    need(a.ndim == 4 and a.shape[1:] == (2, 5, 12) and
         np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(),
         'H1 raw score shape/range differs')
    n = len(a)
    views = np.empty((2, n, 12), np.float64)
    for view in range(2):
        views[view] = np.mean([rankdata(a[:, view, head, :], axis=0,
                                        method='average') / n for head in range(5)], axis=0)
    return rankdata(views.mean(axis=0), axis=0, method='average') / n


def paired_primary_ci(candidate, baseline, refs, draws):
    """Bootstrap the mean of the two reference deltas on identical components."""
    import numpy as np
    from tools import exp115_vlm_eval as ev
    deltas = []
    for ii in draws:
        values = []
        for name in ('jev_hybrid', 'aquila_v1'):
            y, mask, mc = refs[name]
            a = ev.macro(ev.auc_targets(y[ii], candidate[ii], mask[ii], mc))
            b = ev.macro(ev.auc_targets(y[ii], baseline[ii], mask[ii], mc))
            values.append(a - b)
        deltas.append(sum(values) / 2)
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def accuracy_gate(pairs, primary_ci, require_ci):
    primary = (pairs['jev_hybrid']['delta'] + pairs['aquila_v1']['delta']) / 2
    dread = pairs['dread']['delta']
    ci_pass = primary_ci[0] > 0
    return {'primary_mean_delta': primary, 'primary_ci95': primary_ci,
            'dread_delta': dread, 'point_pass': primary > .001,
            'ci_pass': ci_pass, 'dread_pass': dread >= -.002,
            'accuracy_pass': primary > .001 and dread >= -.002 and
            (ci_pass or not require_ci)}


def serving_time_gate(path, k, family):
    if path is None:
        return {'pass_': False, 'reason': 'Full-cohort B3-H1/H1 serving timing not supplied'}
    t = read(path)
    need(t['schema'] == 'exp161-full-serving-time-v1' and t['units'] == 'seconds' and
         t['cohort'] == 'full competition test' and
         set(t['releases']) == {'B3_H1', 'H1'}, 'Full serving timing contract differs')
    detail = {}
    for name in ('B3_H1', 'H1'):
        r = t['releases'][name]
        seconds = r[str(k)]
        need(seconds > 0 and r['reserve'] >= 0 and r['source'], 'Timing receipt incomplete')
        detail[name] = {'elapsed_seconds': seconds, 'reserve_seconds': r['reserve'],
                        'within_9h': seconds + r['reserve'] <= 9 * 3600}
    return {'pass_': all(v['within_9h'] for v in detail.values()),
            'detail': detail, 'evidence_sha256': sha(path), 'family': family}


def screen(outputs=EXP / 'outputs', destination=EXP / 'screen', root=ROOT,
           timing=None, hidden_studies=1300, partner_seconds=None, reserve_seconds=120):
    """CPU-only screen. Run as `python -m tools.exp161_k_sweep2 screen`."""
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp125_radr50_k16_local as e125
    outputs, destination, root = map(Path, (outputs, destination, root))
    collection = read(outputs / 'collection.json')
    need(not destination.exists() and collection['status'] == 'VERIFIED' and
         collection['rows'] == 4343 and collection['families'] == list(FAMILIES) and
         collection['k_values'] == list(K_VALUES), 'Collect complete output; choose fresh screen')
    receipt = read(outputs / 'exp161-pooled/receipt.json')
    need(receipt['schema'] == 'exp161-merged-v1' and receipt['status'] == 'COMPLETE',
         'Pooled receipt differs')
    ids = [u for fold in range(5) for u in read(root /
           read(root / base.protocol(42, fold))['folds'][str(fold)]['ordered_ids']['path'])['val']]
    need(len(ids) == len(set(ids)) == 4343, 'Screen roster differs')
    preds, hashes = {}, {}
    for family in FAMILIES:
        for k in K_VALUES:
            name = f'{family}_k{k}_s42.{ "csv" if family == "aug1" else "npz"}'
            path = outputs / 'exp161-pooled' / name
            need(sha(path) == receipt['outputs'][name], 'Pooled prediction changed')
            hashes[f'{family}_k{k}'] = sha(path)
            if family == 'aug1':
                preds[family, k] = ev.pct_rank(base.csv_values(path, ids))
            else:
                with np.load(path, allow_pickle=False) as z:
                    need(z.files == ['scores'], 'H1 raw archive differs')
                    preds[family, k] = h1_served_oof(z['scores'])
    old_screen = read(root / 'artifacts/experiments/EXP156_K_SWEEP/screen/result.json')
    need(old_screen['schema'] == 'exp156-screen-v1' and old_screen['rows'] == 4343,
         'EXP156 screen control differs')
    replicate = {}
    for k in (36, 62):
        old_path = root / f'artifacts/experiments/EXP156_K_SWEEP/outputs/exp156-seed42/oof_k{k}_s42.csv'
        need(sha(old_path) == old_screen['prediction_sha256'][f's42_k{k}'],
             'EXP156 control hash differs')
        new_path = outputs / f'exp161-pooled/aug1_k{k}_s42.csv'
        old = base.csv_values(old_path, ids)
        new = base.csv_values(new_path, ids)
        replicate[str(k)] = {'exp156_sha256': sha(old_path), 'exp161_sha256': sha(new_path),
                             'max_abs_prediction_delta': float(np.max(np.abs(new-old))),
                             'mean_abs_prediction_delta': float(np.mean(np.abs(new-old)))}
    control_pass = all(v['max_abs_prediction_delta'] <= .002 for v in replicate.values())
    refs = base.references(ids, root)
    draws = e125.component_draws(ids, read(root / base.COMPONENTS), draws=300, seed=20261003)
    pairs, gates = {}, {}
    for family in FAMILIES:
        pairs[family], gates[family] = {}, {}
        for candidate_k, baseline_k in ((62, 36), (94, 62)):
            key = f'{candidate_k}_vs_{baseline_k}'
            a, b = preds[family, candidate_k], preds[family, baseline_k]
            pairs[family][key] = {name: base._clean(ev.compare(a, b, ref, draws))
                                  for name, ref in refs.items()}
            primary_ci = paired_primary_ci(a, b, refs, draws)
            require_ci = candidate_k == 94
            gate = accuracy_gate(pairs[family][key], primary_ci, require_ci)
            gate['timing'] = serving_time_gate(timing, candidate_k, family)
            gate['adopt'] = bool(gate['accuracy_pass'] and gate['timing']['pass_'] and
                                 (control_pass if family == 'aug1' else True))
            gates[family][key] = gate
    old_primary = (old_screen['pairs']['62']['jev_hybrid']['delta'] +
                   old_screen['pairs']['62']['aquila_v1']['delta']) / 2
    new_primary = gates['aug1']['62_vs_36']['primary_mean_delta']
    replicate['primary_delta_vs_exp156'] = new_primary - old_primary
    replicate['per_reference_delta_vs_exp156'] = {
        name: pairs['aug1']['62_vs_36'][name]['delta'] - old_screen['pairs']['62'][name]['delta']
        for name in refs}
    replicate['control_pass_max_abs_le_0_002'] = control_pass
    timing_report = receipt['timing']
    projections = {}
    need(hidden_studies > 0 and reserve_seconds >= 0 and
         (partner_seconds is None or partner_seconds >= 0), 'Projection inputs invalid')
    for k in K_VALUES:
        per = timing_report['h1']['direct'][str(k)]
        h1_wall = hidden_studies * 10 * per['wall_seconds_per_model_study']
        projections[str(k)] = {
            'hidden_studies_assumed': hidden_studies, 'h1_models': 10,
            'h1_conservative_seconds': h1_wall,
            'h1_gpu_seconds': hidden_studies * 10 * per['gpu_seconds_per_model_study'],
            'partner_seconds': partner_seconds, 'reserve_seconds': reserve_seconds,
            'b3_h1_upper_proxy_seconds': None if partner_seconds is None else
                h1_wall + partner_seconds + reserve_seconds,
            'caution': 'Direct probe wall includes per-model preparation; multiplying by ten overcounts shared decode. Full hidden-test timing remains the adoption gate.'}
    verdict = {family: {key: ('ADOPT' if gate['adopt'] else
                               'HOLD_TIMING' if gate['accuracy_pass'] and timing is None else 'KEEP_BASELINE')
                        for key, gate in gates[family].items()} for family in FAMILIES}
    if not control_pass:
        verdict['aug1'] = {key: 'CONTROL_MISMATCH' for key in verdict['aug1']}
    result = {'schema': 'exp161-screen-v1', 'verdict': verdict, 'rows': 4343,
              'bootstrap': {'draws': 300, 'seed': 20261003, 'unit': 'validation component'},
              'h1_oof_rule': 'per-head pooled target rank/view; mean five heads; mean views; outer pooled target rank; one held-out model/study',
              'h1_release_difference': 'Full competition test averages ten models; OOF has one held-out model per study.',
              'prediction_sha256': hashes,
              'reference_sha256': {name: sha(root / path) for name, path in
                                   [('jev_hybrid', base.JEV), ('aquila_v1', base.AQUILA),
                                    ('dread', base.DREAD),
                                    *[(f'consensus_{i}', p) for i, p in enumerate(base.CONSENSUS)]]},
              'replicate_vs_exp156': replicate, 'pairs': pairs, 'gates': gates,
              'per_study_timing': timing_report, 'b3_h1_projection': projections,
              'serving_timing_gate': 'Full competition test B3-H1 and H1 <=9h including reserve.'}
    destination.mkdir(parents=True)
    served = {}
    for k in K_VALUES:
        path = destination / f'h1_k{k}_served_s42.csv'
        base.write_csv(path, ids, preds['h1', k])
        served[str(k)] = sha(path)
    result['h1_served_oof_sha256'] = served
    write_new(destination / 'result.json', result)
    return {'verdict': verdict, 'result': str(destination / 'result.json'),
            'aug_control_max_abs': max(replicate[str(k)]['max_abs_prediction_delta'] for k in (36, 62))}


def push_plan(out=OUT):
    out = Path(out).resolve()
    need((out / 'dataset/manifest.json').is_file() and
         (out / 'kernel/run.ipynb').is_file(), 'Stage first')
    python = r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe'
    kaggle = ROOT / '.venv/Scripts/kaggle.exe'
    return {'status': 'PLAN_ONLY_NO_NETWORK', 'dataset': DATASET, 'kernel': KERNEL,
            'reused_h1_dataset': H1_DATASET, 'commands': [
                f'& "{kaggle}" datasets status {H1_DATASET}',
                f'Set-Location -LiteralPath "{out / "dataset"}"',
                f'& "{kaggle}" datasets create -p .',
                f'& "{kaggle}" datasets status {DATASET}',
                f'Set-Location -LiteralPath "{out / "kernel"}"',
                f'& "{kaggle}" kernels push -p .',
                f'& "{kaggle}" kernels status {KERNEL}',
                f'& "{kaggle}" kernels output {KERNEL} -p "{out / "download"}"',
                f'Set-Location -LiteralPath "{ROOT}"',
                f'& "{python}" -m tools.exp161_k_sweep2 collect --source "{out / "download"}" --stage-dir "{out}"',
                f'& "{python}" -m tools.exp161_k_sweep2 screen']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('stage'); s.add_argument('--out', type=Path, default=OUT)
    s = sub.add_parser('push-plan'); s.add_argument('--out', type=Path, default=OUT)
    s = sub.add_parser('collect'); s.add_argument('--source', type=Path, required=True)
    s.add_argument('--destination', type=Path, default=EXP / 'outputs')
    s.add_argument('--stage-dir', type=Path, default=OUT)
    s = sub.add_parser('screen'); s.add_argument('--outputs', type=Path, default=EXP / 'outputs')
    s.add_argument('--destination', type=Path, default=EXP / 'screen')
    s.add_argument('--timing', type=Path); s.add_argument('--hidden-studies', type=int, default=1300)
    s.add_argument('--partner-seconds', type=float); s.add_argument('--reserve-seconds', type=float, default=120)
    for action in ('run', 'worker'):
        s = sub.add_parser(action)
        s.add_argument('--work', type=Path, required=True); s.add_argument('--corpus', type=Path, required=True)
        s.add_argument('--manifest', type=Path, required=True); s.add_argument('--started', type=float, required=True)
        if action == 'worker': s.add_argument('--worker-id', type=int, required=True)
    args = p.parse_args(argv)
    if args.action == 'stage': result = stage(args.out)
    elif args.action == 'push-plan': result = push_plan(args.out)
    elif args.action == 'collect': result = collect(args.source, args.destination, args.stage_dir)
    elif args.action == 'screen':
        result = screen(args.outputs, args.destination, timing=args.timing,
                        hidden_studies=args.hidden_studies, partner_seconds=args.partner_seconds,
                        reserve_seconds=args.reserve_seconds)
    elif args.action == 'run': result = run(args.work, args.corpus, args.manifest, args.started)
    else: result = worker(args.worker_id, args.work, args.corpus, args.manifest, args.started)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
