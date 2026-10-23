"""EXP151 private, non-OAI four-head AUG1 release (v1.0.3).

`stage` is offline. `public3` is an explicit local CUDA integration run. This
file never contacts Kaggle, uploads, pushes, or submits a prediction.

Changelog: v1.0.1 finds the mounted dataset by manifest; v1.0.2 makes one T4
visible before the generated notebook imports the serving code.
v1.0.3 configures CUBLAS before CUDA, probes only two asset mounts, and starts
the deadline clock before mount checks, hashing, and extraction.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / '.local/exp151-bundle-release'
DATASET = 'taeyangg4/knee-exp151-bundle-aug1-assets'
KERNEL = 'taeyangg4/knee-exp151-bundle-aug1'
HEADS = ('dread', 'relabel_v1', 'steven_v4', 'pilkwang_v2')
RATE_60 = 301.32550326700016 / 60  # EXP133, ten RadR50 models with flip on T4
CUTOFF, RESERVE = 30600, 120
VERSION = '1.0.3'


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def projection(n=1300):
    need(type(n) is int and n > 0, 'Positive study count required')
    seconds = 1.3 * RATE_60 * n + 600
    return {'studies': n, 'measured_seconds_per_study': RATE_60,
            'basis': 'EXP133 timing60: 301.325503267 s/60 studies, ten RadR50 K36 models, two views, one visible T4 in 2xT4 image',
            'projected_seconds': seconds, 'deadline_seconds': CUTOFF,
            'publication_reserve_seconds': RESERVE,
            'margin_seconds': CUTOFF - RESERVE - seconds,
            'limitation': 'Four-head forward/ranking and full-cohort decode have not been timed; Public3 is required.'}


def checkpoint_entry(root, seed, fold, hash_weight=True):
    tag = f'EXP138_BUNDLE_AUG1_MH_S{seed}_F{fold}'
    run = Path(root) / 'artifacts/experiments' / tag
    inventory = json.loads((run / 'output-inventory.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    prov = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
    protocol = Path(root) / 'artifacts/experiments/EXP138_BUNDLE_AUG1_MH' / f's{seed}/manifest.json'
    expected = inventory['model_best.pt']
    model = run / 'model_best.pt'
    for name in ('summary.json', 'provenance.json'):
        need(sha(run / name) == inventory[name]['sha256'] and (run / name).stat().st_size == inventory[name]['bytes'],
             'EXP138 inventory differs: ' + tag + '/' + name)
    need(model.stat().st_size == expected['bytes'], 'Checkpoint bytes differ: ' + tag)
    if hash_weight:
        need(sha(model) == expected['sha256'], 'Checkpoint SHA differs: ' + tag)
    need(summary['mode'] == 'full' and summary['fold'] == fold and summary['selected_checkpoint'] == 'fixed-final'
         and summary['worker_cleanup']['clean'] is True and summary['total_seconds'] > 0
         and summary['heads'] == list(HEADS), 'EXP138 completion/heads differ: ' + tag)
    need(prov['seed'] == seed and prov['fold'] == fold and prov['heads'] == list(HEADS)
         and prov['exp127_variant'] == 'AUG1' and prov['arch'] == 'resnet50'
         and prov['norm'] == 'radimagenet' and prov['input_policy'] == 'center3'
         and prov['k_eval_extra'] == 36 and prov['protocol_sha256'] == sha(protocol)
         and summary['provenance']['protocol_sha256'] == prov['protocol_sha256'],
         'EXP138 serving provenance differs: ' + tag)
    return {'seed': seed, 'fold': fold, 'tag': tag,
            'path': f'checkpoints/s{seed}_f{fold}.pt', 'sha256': expected['sha256'],
            'bytes': expected['bytes'], 'protocol_sha256': prov['protocol_sha256']}


def checkpoints(root=ROOT):
    entries = [checkpoint_entry(root, seed, fold) for seed in (42, 43) for fold in range(5)]
    need(len(entries) == 10 and len({x['sha256'] for x in entries}) == 10, 'Ten distinct checkpoints required')
    return entries


def _function_source(path, names):
    source = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(source)
    found = {n.name: ast.get_source_segment(source, n) for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names}
    need(set(found) == set(names), 'Frozen function missing: ' + str(path))
    return '\n\n'.join(found[n] for n in names) + '\n'


def components_source():
    # exp138_flip_tta delegates its inference functions to exp124_flip_tta.
    # Copy their exact definitions into the offline dataset and call them there.
    registry = json.loads((ROOT / 'artifacts/experiments/EXP138_BUNDLE_AUG1_MH/s42/manifest.json').read_text(encoding='utf-8'))
    for name in ('tools/exp138_flip_tta.py', 'tools/exp124_flip_tta.py',
                 'tools/exp117_multihead.py', 'tools/exp099_kfold.py',
                 'tools/exp138_bundle_aug1_mh.py'):
        need(sha(ROOT / name) == registry['source_bindings'][name],
             'EXP138-bound source changed: ' + name)
    bridge = (HERE / 'exp138_flip_tta.py').read_text(encoding='utf-8')
    need('from tools import exp124_flip_tta as flip' in bridge,
         'EXP138 flip TTA delegation changed')
    body = ('from __future__ import annotations\nimport copy\n'
            'from types import SimpleNamespace\n\n')
    body += _function_source(HERE / 'exp124_flip_tta.py', ('plane_ranges', 'flip_mask'))
    body += '\n' + _function_source(HERE / 'exp099_kfold.py', ('avg_rank',))
    body += '\nk99 = SimpleNamespace(avg_rank=avg_rank)\n\n'
    body += _function_source(HERE / 'exp117_multihead.py', ('build_multihead', 'rank_mean'))
    body += '\n' + _function_source(HERE / 'exp138_bundle_aug1_mh.py', ('fold_ranks',))
    compile(body, 'exp151_components.py', 'exec')
    return body


def code_files():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import exp133_oai_b3 as e133
    import exp128_oai_light as e128
    payload = e133.payload()['files']  # EXP140/EXP133 frozen serving sources and SHA gates
    files = {'exp151_bundle_release.py': Path(__file__).read_bytes(),
             'exp151_components.py': components_source().encode('utf-8'),
             'exp133_oai_b3.py': (HERE / 'exp133_oai_b3.py').read_bytes(),
             'exp128_oai_light.py': (HERE / 'exp128_oai_light.py').read_bytes()}
    for name, item in payload.items():
        content = e128.unpack(item['z']).encode('utf-8')
        need(hashlib.sha256(content).hexdigest() == item['sha256'], 'Frozen serving code differs: ' + name)
        files[name] = content
    for name, data in files.items():
        if name.endswith('.py'):
            compile(data.decode('utf-8'), name, 'exec')
    return files


def pack_archive(path, members):
    """Write a fresh .zip.bin; Kaggle must not auto-extract these archives."""
    path = Path(path)
    need(not path.exists() and path.suffix == '.bin' and len(members) == len(set(members)), 'Unsafe archive target')
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, source in members.items():
            pure = Path(name)
            need(not pure.is_absolute() and '..' not in pure.parts and name.replace('\\', '/') == name,
                 'Unsafe archive member')
            if isinstance(source, bytes):
                archive.writestr(name, source)
            else:
                archive.write(source, name)
    return sha(path)


def notebook(manifest_sha=None):
    cell = "NOTEBOOK_STARTED = __import__('time').monotonic()\nimport os\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'\nos.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n" + 'EXPECTED_MANIFEST_SHA = ' + repr(manifest_sha or '') + '\n' + '''import json, pathlib, sys, time, zipfile, hashlib
assert sys.platform.startswith('linux') and sys.version_info[:2] == (3, 13), 'Pinned Kaggle Linux Python 3.13 required'
import torch, timm
assert torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0), 'One visible T4 required'
assert str(torch.__version__) == '2.11.0+cu128' and timm.__version__ == '1.0.29', 'Pinned image versions differ'
_comp = [d for d in (pathlib.Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), pathlib.Path('/kaggle/input/rsna-knee-abnormality-detection')) if (d / 'test.csv').exists()]
assert _comp, 'competition data not mounted'
_slug = 'knee-exp151-bundle-aug1-assets'
_roots = [pathlib.Path('/kaggle/input') / _slug, pathlib.Path('/kaggle/input/datasets/taeyangg4') / _slug]
_hits = [p for p in _roots if (p / 'release-manifest.json').is_file()]
assert len(_hits) == 1, _hits
dataset = _hits[0]
manifest = json.loads((dataset / 'release-manifest.json').read_text(encoding='utf-8'))
work = pathlib.Path('/kaggle/working/exp151')
work.mkdir(parents=True, exist_ok=False)
def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
if EXPECTED_MANIFEST_SHA:
    assert file_sha(dataset / 'release-manifest.json') == EXPECTED_MANIFEST_SHA
for archive_name, entries in (('code.zip.bin', manifest['code']), ('checkpoints.zip.bin', manifest['checkpoints'])):
    archive_path = dataset / archive_name
    h = file_sha(archive_path)
    assert h == manifest['archives'][archive_name], archive_name
    with zipfile.ZipFile(archive_path) as z:
        assert set(z.namelist()) == set(entries)
        for name, info in entries.items():
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, target.open('xb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
            assert target.stat().st_size == info['bytes']
            assert file_sha(target) == info['sha256']
sys.path.insert(0, str(work / 'code'))
import exp151_bundle_release as release
release.serve(work, _comp[0],
              pathlib.Path('/kaggle/working'), manifest, notebook_started=NOTEBOOK_STARTED)
'''
    compile(cell, '<EXP151 release cell>', 'exec')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}},
            'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source':
                       '# EXP151 own non-OAI RadImageNet AUG1 bundle\nK36, EXP138 plane-aware flip, four-head per-target rank mean for each view, view mean, per-model target rank, ten-model mean.\n'},
                      {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                       'outputs': [], 'source': cell}]}


def stage(out=OUT, root=ROOT):
    out, root = Path(out), Path(root)
    need(not out.exists(), 'Choose a fresh stage directory')
    entries = checkpoints(root)
    codes = code_files()
    dataset, kernel = out / 'dataset', out / 'kernel'
    dataset.mkdir(parents=True)
    kernel.mkdir()
    ck = {x['path']: root / 'artifacts/experiments' / x['tag'] / 'model_best.pt' for x in entries}
    code = {'code/' + name: data for name, data in codes.items()}
    archives = {'checkpoints.zip.bin': pack_archive(dataset / 'checkpoints.zip.bin', ck),
                'code.zip.bin': pack_archive(dataset / 'code.zip.bin', code)}
    manifest = {'schema': 'exp151-bundle-release-v1', 'version': VERSION, 'dataset': DATASET,
                'kernel': KERNEL, 'heads': list(HEADS), 'archives': archives,
                'checkpoints': {x['path']: {'sha256': x['sha256'], 'bytes': x['bytes']} for x in entries},
                'code': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                         for name, data in code.items()}, 'entries': entries,
                'flip_source_sha256': sha(HERE / 'exp138_flip_tta.py'),
                'flip_functions_sha256': sha(HERE / 'exp124_flip_tta.py'),
                'runtime_projection_1300': projection()}
    save(dataset / 'release-manifest.json', manifest)
    save(dataset / 'dataset-metadata.json', {'title': DATASET.split('/')[-1], 'id': DATASET,
         'isPrivate': True, 'licenses': [{'name': 'other'}],
         'description': 'Private own EXP138 four-head AUG1 checkpoints and frozen serving code.'})
    import exp128_oai_light as e128
    metadata = e128.metadata('fidelity')
    metadata.update(id=KERNEL, title=KERNEL.split('/')[-1], dataset_sources=[DATASET])
    save(kernel / 'kernel-metadata.json', metadata)
    save(kernel / 'release.ipynb', notebook(sha(dataset / 'release-manifest.json')))
    save(out / 'stage-receipt.json', {'status': 'STAGED_HASH_VERIFIED', 'dataset': DATASET,
         'kernel': KERNEL, 'manifest_sha256': sha(dataset / 'release-manifest.json'),
         'notebook_sha256': sha(kernel / 'release.ipynb'), 'archives': archives,
         'checkpoints': len(entries), 'network_calls': 0, 'gpu_runs': 0, 'submissions': 0})
    return {'status': 'STAGED_HASH_VERIFIED', 'out': str(out), 'checkpoints': len(entries),
            'archives': archives, 'projection': projection()}


def extract(dataset, work, manifest):
    dataset, work = Path(dataset), Path(work)
    need(not work.exists(), 'Choose a fresh extraction directory')
    work.mkdir(parents=True)
    for archive_name, entries in (('code.zip.bin', manifest['code']),
                                  ('checkpoints.zip.bin', manifest['checkpoints'])):
        archive = dataset / archive_name
        need(sha(archive) == manifest['archives'][archive_name], 'Archive hash differs')
        with zipfile.ZipFile(archive) as z:
            need(set(z.namelist()) == set(entries), 'Archive inventory differs')
            for name, info in entries.items():
                need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe archive path')
                dst = work / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, dst.open('xb') as sink:
                    shutil.copyfileobj(src, sink)
                need(dst.stat().st_size == info['bytes'] and sha(dst) == info['sha256'],
                     'Extracted file differs: ' + name)
    return work


def reduce_heads(plain, mirror, rank_mean, fold_ranks):
    """EXP138: head ranks/view, view mean, per-fold score rank, ten-model mean."""
    import numpy as np
    a, b = np.asarray(plain), np.asarray(mirror)
    need(a.shape == b.shape and a.ndim == 4 and a.shape[0] == 10
         and a.shape[2:] == (4, 12) and a.shape[1] > 0, 'Model/head/target shape differs')
    need(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite head probabilities')
    served = []
    for i in range(10):
        views = (rank_mean(a[i]) + rank_mean(b[i])) / 2
        served.append(fold_ranks(views))
    return np.mean(served, axis=0)


def serve(work, competition, output, manifest, local_public3=False, notebook_started=None):
    """Same release path for Kaggle and local Public3; CUDA is initialized here only."""
    started = time.monotonic() if notebook_started is None else notebook_started
    work, competition, output = Path(work), Path(competition), Path(output)
    sys.path.insert(0, str(work / 'code'))
    config = work / 'config'
    config.mkdir(exist_ok=True)
    marker = config / 'workspace.toml'
    if not marker.exists():
        marker.write_text('# EXP151 test-only serving root\n', encoding='utf-8')
    os.environ['KNEE_WORKSPACE_ROOT'] = str(work)
    import numpy as np
    import torch
    import cv2
    import timm
    import pydicom
    import exp133_oai_b3 as e133
    import exp151_components as c
    import exp111_release_v6 as v6
    import raptor_runtime as rt
    from raptor_common import build_model, select_centers
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    if local_public3:
        need(torch.cuda.device_count() >= 1, 'Local Public3 requires a CUDA GPU')
    else:
        need(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 13), 'Pinned Kaggle Linux Python 3.13 required')
        need(torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0), 'One visible T4 required')
        need(str(torch.__version__) == '2.11.0+cu128' and timm.__version__ == '1.0.29', 'Pinned image versions differ')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    rt.configure_runtime('fp16', 'cuda', 'strict')
    ids = e133.sample_ids(competition / 'sample_submission.csv')
    estimate = projection(len(ids))
    need(estimate['projected_seconds'] <= CUTOFF - RESERVE, 'Projected runtime exceeds deadline')
    need(time.monotonic() - started <= CUTOFF - RESERVE, 'EXP151 deadline exceeded before model load')
    models = []
    for entry in manifest['entries']:
        path = work / entry['path']
        need(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Checkpoint differs')
        ck = torch.load(path, map_location='cpu', weights_only=False)  # own trusted SHA-bound checkpoint
        expect = {'arch': 'resnet50', 'res': 224, 'lab': e133.TARGETS,
                  'contract': 'raptor64_336_c140_s02', 'norm': 'radimagenet',
                  'window_policy': 'within-slot-valid', 'input_policy': 'center3',
                  'tag': entry['tag'], 'fold': entry['fold'], 'heads': list(HEADS)}
        need(all(ck.get(k) == v for k, v in expect.items()), 'Checkpoint metadata differs')
        need(ck['provenance']['protocol_sha256'] == entry['protocol_sha256'], 'Checkpoint protocol differs')
        model = c.build_multihead(build_model('resnet50', pretrained=False), 4)
        model.load_state_dict(ck['model'], strict=True)
        models.append(model.eval().to('cuda'))
        del ck
    loaded = time.monotonic()
    corpus = DicomCorpus(competition / 'test.csv', competition / 'test_series.csv',
                         DirectoryDicomSource(competition / 'test_series'))
    need(set(corpus.ids) == set(ids), 'Competition cohort differs')
    ranges = c.plane_ranges(dict(corpus.spec, depth=corpus.depth))
    scores = np.empty((10, 2, len(ids), 4, 12), np.float32)
    decode_seconds = 0.0
    with torch.inference_mode(), v6.serving_input(rt, 'center3'):
        for row, uid in enumerate(ids):
            then = time.monotonic()
            corpus.volume(uid)
            mask = corpus.mask(uid)
            x = rt.prepare_input(corpus, uid, 36, 224, 'radimagenet', 'within-slot-valid')
            centers = select_centers(mask, corpus.depth, 36, corpus.spec, policy='within-slot-valid')
            flags = c.flip_mask(centers, ranges)
            xm = x.copy()
            xm[flags] = xm[flags, ..., ::-1]
            decode_seconds += time.monotonic() - then
            for view, array in enumerate((x, xm)):
                xt = torch.from_numpy(np.ascontiguousarray(array)[None]).to('cuda')
                for i, model in enumerate(models):
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(xt)
                    need(tuple(logits.shape) == (1, 4, 12) and bool(torch.isfinite(logits).all()),
                         'Nonfinite or wrong-shape multihead logits')
                    scores[i, view, row] = torch.sigmoid(logits.float())[0].cpu().numpy()
            if (row + 1) % 50 == 0 or row + 1 == len(ids):
                print('EXP151', row + 1, '/', len(ids), 'seconds', round(time.monotonic() - started, 1), flush=True)
            need(time.monotonic() - started <= CUTOFF - RESERVE, 'EXP151 deadline exceeded')
    result = reduce_heads(scores[:, 0], scores[:, 1], c.rank_mean, c.fold_ranks)
    need(time.monotonic() - started <= CUTOFF - RESERVE, 'EXP151 deadline exceeded before publication')
    output.mkdir(parents=True, exist_ok=True)
    final = output / 'submission.csv'
    need(not final.exists(), 'Refuse to overwrite submission.csv')
    e133.write_csv(final, ids, result.tolist())
    checked_ids, checked = e133.validate_csv(final, competition / 'sample_submission.csv')
    need(checked_ids == ids and np.array_equal(np.asarray(checked), result), 'Submission CSV roundtrip differs')
    receipt = {'schema': 'exp151-bundle-serving-v1', 'status': 'PASS', 'studies': len(ids),
               'heads': list(HEADS), 'models': len(models), 'k_eval': 36,
               'rule': 'per-target rank mean of four heads per model/view across full cohort; mean views; per-model target rank; mean ten models',
               'submission_sha256': sha(final), 'elapsed_seconds': time.monotonic() - started,
               'load_seconds': loaded - started, 'decode_prepare_seconds': decode_seconds,
               'projection': estimate, 'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               'training': 0, 'label_reads': 0}
    save(output / 'exp151_stage_timing.json', receipt)
    return receipt


def public3(out, competition, output, python=sys.executable):
    out, competition, output = Path(out), Path(competition), Path(output)
    need(not output.exists(), 'Choose fresh Public3 output directory')
    manifest = json.loads((out / 'dataset/release-manifest.json').read_text(encoding='utf-8'))
    direct = extract(out / 'dataset', output / 'direct-assets', manifest)
    local = serve(direct, competition, output / 'direct', manifest, local_public3=True)
    packaged = extract(out / 'dataset', output / 'packaged-assets', manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', PYTHONUTF8='1')
    command = [python, str(packaged / 'code/exp151_bundle_release.py'), 'serve',
               '--work', str(packaged), '--competition', str(competition),
               '--output', str(output / 'packaged'), '--manifest', str(out / 'dataset/release-manifest.json'),
               '--local-public3']
    subprocess.run(command, check=True, env=env)
    a, b = output / 'direct/submission.csv', output / 'packaged/submission.csv'
    need(a.read_bytes() == b.read_bytes(), 'Public3 local/release CSV bytes differ')
    need(local['studies'] == 3 and sha(a) == local['submission_sha256'], 'Public3 receipt/cohort differs')
    result = {'status': 'PASS_PUBLIC3_BYTE_IDENTICAL', 'studies': 3,
              'submission_sha256': sha(a), 'local': str(a), 'release': str(b)}
    save(output / 'public3-verification.json', result)
    return result


def push_plan(out=OUT):
    out = Path(out)
    need((out / 'dataset/release-manifest.json').is_file(), 'Stage first')
    return {'commands': [
        f'cd "{out / "dataset"}"',
        'kaggle datasets create -p .',
        '# For a later revision from inside the same dataset folder: kaggle datasets version -p . -m "EXP151 bundle update"',
        f'cd "{out / "kernel"}"',
        'kaggle kernels push -p .',
        '# B verifies the private Public3 output before any submission; submission only on Taeyang\'s word.'
    ], 'dataset': DATASET, 'kernel': KERNEL}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('stage'); s.add_argument('--out', type=Path, default=OUT)
    s = sub.add_parser('public3'); s.add_argument('--out', type=Path, default=OUT)
    s.add_argument('--competition', type=Path, required=True)
    s.add_argument('--output', type=Path, default=OUT / 'public3-output')
    s.add_argument('--python', default=sys.executable)
    s = sub.add_parser('push-plan'); s.add_argument('--out', type=Path, default=OUT)
    s = sub.add_parser('serve'); s.add_argument('--work', type=Path, required=True)
    s.add_argument('--competition', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--manifest', type=Path, required=True)
    s.add_argument('--local-public3', action='store_true')
    a = p.parse_args(argv)
    if a.action == 'stage':
        result = stage(a.out)
    elif a.action == 'public3':
        result = public3(a.out, a.competition, a.output, a.python)
    elif a.action == 'push-plan':
        result = push_plan(a.out)
    else:
        result = serve(a.work, a.competition, a.output,
                       json.loads(a.manifest.read_text(encoding='utf-8')), a.local_public3)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
