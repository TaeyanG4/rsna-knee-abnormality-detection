"""EXP153 five-head H1 private release builder (v1.0.1).

Stage and push-plan are offline. Public3 and serve are explicit CUDA actions;
this module never starts a Kaggle job, publishes an asset, or submits a CSV.
EXP151 remains frozen and supplies the archive, extraction, source and plane
flip/rank components used here.
Changelog: v1.0.1 inherits EXP151's early CUDA checks and bounded startup
deadline through its regenerated notebook header.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import exp151_bundle_release as base

VERSION = '1.0.1'
OUT = ROOT / '.local/exp153-h1-release'
DATASET = 'taeyangg4/knee-exp153-h1-aquila-head-assets'
KERNEL = 'taeyangg4/knee-exp153-h1-aquila-head'
HEADS = (*base.HEADS, 'aquila_v1')
INITIAL_SHA = 'a56516a289178dce17d2fe736147ddbbcb52a571cb25e6c8b59b4342a3d3e636'
EXP149_MANIFEST = 'artifacts/experiments/EXP149_LABEL_SOURCES/students/manifest.json'
MODEL_COUNT = 10
VERSION_SCHEMA = 'exp153-h1-release-v1'


def tag_for(seed, fold):
    base.need(seed in (42, 43) and fold in range(5), 'Invalid H1 seed/fold')
    prefix = 'EXP149' if seed == 42 and fold < 2 else 'EXP153'
    return f'{prefix}_H1_S{seed}_F{fold}'


def checkpoint_entry(root, seed, fold, hash_weight=True):
    """Fail closed on training inventory and the H1 fixed-final contract."""
    root = Path(root)
    tag = tag_for(seed, fold)
    run = root / 'artifacts/experiments' / tag
    inv = json.loads((run / 'output-inventory.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    prov = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                'oof_best.csv', 'oof_k36.csv'} | {
                    f'oof_head_{head}{suffix}.csv' for head in HEADS for suffix in ('', '_k36')}
    base.need(set(inv) == required, f'Incomplete H1 inventory: {tag}')
    for name, item in inv.items():
        path = run / name
        base.need(path.is_file(), f'H1 inventory file missing: {tag}/{name}')
        base.need(path.stat().st_size == item['bytes'], f'H1 inventory bytes differ: {tag}/{name}')
        if hash_weight or name != 'model_best.pt':
            base.need(base.sha(path) == item['sha256'], f'H1 inventory SHA differs: {tag}/{name}')
    history = json.loads((run / 'history.json').read_text(encoding='utf-8'))
    base.need(len(history) == 20 and all(x.get('epoch') == i for i, x in enumerate(history)),
              f'H1 history differs: {tag}')
    base.need(summary.get('run') == tag and summary.get('fold') == fold
              and summary.get('mode') == 'full' and summary.get('label_set') == 'h1'
              and summary.get('heads') == list(HEADS)
              and summary.get('selected_checkpoint') == 'fixed-final'
              and summary.get('selected_epoch_1based') == 20
              and summary.get('optimizer_updates') == prov.get('steps')
              and isinstance(prov.get('steps'), int) and prov['steps'] > 0
              and summary.get('skipped_updates') == 0
              and summary.get('gold') == 0 and summary.get('gold_access') == 'disabled'
              and summary.get('worker_cleanup', {}).get('clean') is True
              and summary.get('total_seconds', 0) > 0,
              f'H1 completion differs: {tag}')
    base.need(prov.get('run') == tag and prov.get('seed') == seed and prov.get('fold') == fold
              and prov.get('heads') == list(HEADS) and prov.get('label_set') == 'h1'
              and prov.get('arch') == 'resnet50' and prov.get('norm') == 'radimagenet'
              and prov.get('input_policy') == 'center3' and prov.get('k_eval_extra') == 36
              and prov.get('train_folds') == [k for k in range(5) if k != fold]
              and prov.get('eval_folds') == [fold]
              and prov.get('checkpoint_selection') == 'fixed-final'
              and prov.get('epochs') == 20 and prov.get('gold_access') == 'disabled'
              and prov.get('full_model_initial_state_sha256') == INITIAL_SHA
              and prov.get('fp16_skipped_updates') == 0,
              f'H1 provenance differs: {tag}')
    protocol_sha = prov.get('protocol_sha256')
    base.need(isinstance(protocol_sha, str) and len(protocol_sha) == 64
              and summary.get('provenance', {}).get('protocol_sha256') == protocol_sha,
              f'H1 protocol provenance differs: {tag}')
    if tag.startswith('EXP149'):
        manifest = root / EXP149_MANIFEST
        registered = json.loads(manifest.read_text(encoding='utf-8'))['runs'][tag]
        base.need(base.sha(manifest) == protocol_sha
                  and registered['initial_state_sha256'] == INITIAL_SHA
                  and registered['heads'] == list(HEADS),
                  f'EXP149 registration differs: {tag}')
    else:
        # EXP153 uses a separately registered protocol; the advance tool
        # validates its own registration and records the matching path here.
        protocol_path = prov.get('protocol_path')
        base.need(isinstance(protocol_path, str) and protocol_path.startswith('artifacts/experiments/EXP153_')
                  and '..' not in Path(protocol_path).parts
                  and base.sha(root / protocol_path) == protocol_sha,
                  f'EXP153 registration differs: {tag}')
    return {'seed': seed, 'fold': fold, 'tag': tag,
            'path': f'checkpoints/s{seed}_f{fold}.pt',
            'sha256': inv['model_best.pt']['sha256'], 'bytes': inv['model_best.pt']['bytes'],
            'protocol_sha256': protocol_sha,
            'full_model_initial_state_sha256': INITIAL_SHA}


def checkpoints(root=ROOT):
    entries = [checkpoint_entry(root, seed, fold) for seed in (42, 43) for fold in range(5)]
    base.need(len(entries) == MODEL_COUNT and len({x['sha256'] for x in entries}) == MODEL_COUNT,
              'Ten distinct H1 checkpoints required')
    return entries


def reduce_heads(plain, mirror, rank_mean, fold_ranks):
    """Head ranks per view, view mean, model rank, equal ten-model mean."""
    import numpy as np
    a, b = np.asarray(plain), np.asarray(mirror)
    base.need(a.shape == b.shape and a.ndim == 4 and a.shape[0] == MODEL_COUNT
              and a.shape[2:] == (len(HEADS), 12) and a.shape[1] > 0,
              'Model/head/target shape differs')
    base.need(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite head probabilities')
    return np.mean([fold_ranks((rank_mean(a[i]) + rank_mean(b[i])) / 2)
                    for i in range(MODEL_COUNT)], axis=0)


def code_files():
    files = base.code_files()
    files['exp153_h1_release.py'] = Path(__file__).read_bytes()
    compile(files['exp153_h1_release.py'].decode('utf-8'), 'exp153_h1_release.py', 'exec')
    return files


def notebook(manifest_sha=None):
    """Reuse EXP151's verified extraction cell, changing only entry points."""
    nb = base.notebook(manifest_sha)
    cell = nb['cells'][-1]['source']
    substitutions = {
        'knee-exp151-bundle-aug1-assets': DATASET.split('/')[-1],
        "pathlib.Path('/kaggle/working/exp151')": "pathlib.Path('/kaggle/working/exp153')",
        'import exp151_bundle_release as release': 'import exp153_h1_release as release',
    }
    for old, new in substitutions.items():
        base.need(cell.count(old) == 1, 'Frozen EXP151 notebook entry point changed: ' + old)
        cell = cell.replace(old, new)
    compile(cell, '<EXP153 release cell>', 'exec')
    nb['cells'][-1]['source'] = cell
    nb['cells'][0]['source'] = ('# EXP153 own non-OAI RadImageNet AUG1 H1 bundle\n'
                                 'K36, plane-aware flip, five-head target rank mean per view, '
                                 'view mean, per-model target rank, ten-model mean.\n')
    return nb


def projection(n=1300):
    estimate = base.projection(n)
    estimate['limitation'] = ('EXP133 ten-model timing was measured with fewer heads; '
                              'five-head full-cohort decode and ranking require Public3 and full runtime checks.')
    return estimate


def stage(out=OUT, root=ROOT):
    out, root = Path(out), Path(root)
    base.need(not out.exists(), 'Choose a fresh stage directory')
    entries, codes = checkpoints(root), code_files()
    dataset, kernel = out / 'dataset', out / 'kernel'
    dataset.mkdir(parents=True)
    kernel.mkdir()
    ck = {x['path']: root / 'artifacts/experiments' / x['tag'] / 'model_best.pt' for x in entries}
    code = {'code/' + name: data for name, data in codes.items()}
    archives = {'checkpoints.zip.bin': base.pack_archive(dataset / 'checkpoints.zip.bin', ck),
                'code.zip.bin': base.pack_archive(dataset / 'code.zip.bin', code)}
    manifest = {'schema': VERSION_SCHEMA, 'version': VERSION, 'dataset': DATASET,
                'kernel': KERNEL, 'heads': list(HEADS), 'archives': archives,
                'checkpoints': {x['path']: {'sha256': x['sha256'], 'bytes': x['bytes']} for x in entries},
                'code': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                         for name, data in code.items()}, 'entries': entries,
                'flip_source_sha256': base.sha(HERE / 'exp138_flip_tta.py'),
                'flip_functions_sha256': base.sha(HERE / 'exp124_flip_tta.py'),
                'runtime_projection_1300': projection()}
    base.save(dataset / 'release-manifest.json', manifest)
    base.save(dataset / 'dataset-metadata.json',
              {'title': DATASET.split('/')[-1], 'id': DATASET, 'isPrivate': True,
               'licenses': [{'name': 'other'}],
               'description': 'Private own EXP153 H1 five-head AUG1 checkpoints and frozen serving components.'})
    import exp128_oai_light as e128
    metadata = e128.metadata('fidelity')
    metadata.update(id=KERNEL, title=KERNEL.split('/')[-1], dataset_sources=[DATASET])
    base.save(kernel / 'kernel-metadata.json', metadata)
    base.save(kernel / 'release.ipynb', notebook(base.sha(dataset / 'release-manifest.json')))
    base.save(out / 'stage-receipt.json',
              {'status': 'STAGED_HASH_VERIFIED', 'dataset': DATASET, 'kernel': KERNEL,
               'manifest_sha256': base.sha(dataset / 'release-manifest.json'),
               'notebook_sha256': base.sha(kernel / 'release.ipynb'), 'archives': archives,
               'checkpoints': len(entries), 'network_calls': 0, 'gpu_runs': 0, 'submissions': 0})
    return {'status': 'STAGED_HASH_VERIFIED', 'out': str(out), 'checkpoints': len(entries),
            'archives': archives, 'projection': projection()}


def serve(work, competition, output, manifest, local_public3=False, notebook_started=None):
    """Use EXP151 components with five H1 heads; CUDA starts only here."""
    started = time.monotonic() if notebook_started is None else notebook_started
    work, competition, output = Path(work), Path(competition), Path(output)
    base.need(manifest.get('schema') == VERSION_SCHEMA and manifest.get('heads') == list(HEADS)
              and len(manifest.get('entries', [])) == MODEL_COUNT, 'EXP153 release manifest differs')
    sys.path.insert(0, str(work / 'code'))
    config = work / 'config'
    config.mkdir(exist_ok=True)
    marker = config / 'workspace.toml'
    if not marker.exists():
        marker.write_text('# EXP153 serving root\n', encoding='utf-8')
    os.environ['KNEE_WORKSPACE_ROOT'] = str(work)
    import numpy as np
    import torch
    import cv2
    import timm
    import pydicom  # noqa: F401; bundled runtime dependency
    import exp133_oai_b3 as e133
    import exp151_components as c
    import exp111_release_v6 as v6
    import raptor_runtime as rt
    from raptor_common import build_model, select_centers
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    if local_public3:
        base.need(torch.cuda.device_count() >= 1, 'Local Public3 requires a CUDA GPU')
    else:
        base.need(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 13),
                  'Pinned Kaggle Linux Python 3.13 required')
        base.need(torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0),
                  'One visible T4 required')
        base.need(str(torch.__version__) == '2.11.0+cu128' and timm.__version__ == '1.0.29',
                  'Pinned image versions differ')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    rt.configure_runtime('fp16', 'cuda', 'strict')
    ids = e133.sample_ids(competition / 'sample_submission.csv')
    estimate = projection(len(ids))
    base.need(estimate['projected_seconds'] <= base.CUTOFF - base.RESERVE,
              'Projected runtime exceeds deadline')
    base.need(time.monotonic() - started <= base.CUTOFF - base.RESERVE,
              'EXP153 deadline exceeded before model load')
    models = []
    for entry in manifest['entries']:
        path = work / entry['path']
        base.need(path.stat().st_size == entry['bytes'] and base.sha(path) == entry['sha256'],
                  'Checkpoint differs')
        ck = torch.load(path, map_location='cpu', weights_only=False)  # own SHA-bound checkpoint
        expect = {'arch': 'resnet50', 'res': 224, 'lab': e133.TARGETS,
                  'contract': 'raptor64_336_c140_s02', 'norm': 'radimagenet',
                  'window_policy': 'within-slot-valid', 'input_policy': 'center3',
                  'tag': entry['tag'], 'fold': entry['fold'], 'heads': list(HEADS),
                  'label_set': 'h1', 'epoch': 19}
        base.need(all(ck.get(k) == v for k, v in expect.items()), 'Checkpoint metadata differs')
        base.need(ck['provenance']['protocol_sha256'] == entry['protocol_sha256']
                  and ck['provenance']['full_model_initial_state_sha256'] == INITIAL_SHA,
                  'Checkpoint provenance differs')
        model = c.build_multihead(build_model('resnet50', pretrained=False), len(HEADS))
        model.load_state_dict(ck['model'], strict=True)
        models.append(model.eval().to('cuda'))
        del ck
    loaded = time.monotonic()
    corpus = DicomCorpus(competition / 'test.csv', competition / 'test_series.csv',
                         DirectoryDicomSource(competition / 'test_series'))
    base.need(set(corpus.ids) == set(ids), 'Competition cohort differs')
    ranges = c.plane_ranges(dict(corpus.spec, depth=corpus.depth))
    scores = np.empty((MODEL_COUNT, 2, len(ids), len(HEADS), 12), np.float32)
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
                    base.need(tuple(logits.shape) == (1, len(HEADS), 12)
                              and bool(torch.isfinite(logits).all()),
                              'Nonfinite or wrong-shape multihead logits')
                    scores[i, view, row] = torch.sigmoid(logits.float())[0].cpu().numpy()
            if (row + 1) % 50 == 0 or row + 1 == len(ids):
                print('EXP153', row + 1, '/', len(ids), 'seconds',
                      round(time.monotonic() - started, 1), flush=True)
            base.need(time.monotonic() - started <= base.CUTOFF - base.RESERVE,
                      'EXP153 deadline exceeded')
    result = reduce_heads(scores[:, 0], scores[:, 1], c.rank_mean, c.fold_ranks)
    base.need(time.monotonic() - started <= base.CUTOFF - base.RESERVE,
              'EXP153 deadline exceeded before publication')
    output.mkdir(parents=True, exist_ok=True)
    final = output / 'submission.csv'
    base.need(not final.exists(), 'Refuse to overwrite submission.csv')
    e133.write_csv(final, ids, result.tolist())
    checked_ids, checked = e133.validate_csv(final, competition / 'sample_submission.csv')
    base.need(checked_ids == ids and np.array_equal(np.asarray(checked), result),
              'Submission CSV roundtrip differs')
    receipt = {'schema': 'exp153-h1-serving-v1', 'status': 'PASS', 'studies': len(ids),
               'heads': list(HEADS), 'models': len(models), 'k_eval': 36,
               'rule': 'five head target ranks/model/view; mean views; target rank/model; mean ten models',
               'submission_sha256': base.sha(final), 'elapsed_seconds': time.monotonic() - started,
               'load_seconds': loaded - started, 'decode_prepare_seconds': decode_seconds,
               'projection': estimate,
               'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               'training': 0, 'label_reads': 0}
    base.save(output / 'exp153_stage_timing.json', receipt)
    return receipt


def public3(out, competition, output, python=sys.executable):
    out, competition, output = Path(out), Path(competition), Path(output)
    base.need(not output.exists(), 'Choose fresh Public3 output directory')
    manifest = json.loads((out / 'dataset/release-manifest.json').read_text(encoding='utf-8'))
    direct = base.extract(out / 'dataset', output / 'direct-assets', manifest)
    local = serve(direct, competition, output / 'direct', manifest, local_public3=True)
    packaged = base.extract(out / 'dataset', output / 'packaged-assets', manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', PYTHONUTF8='1')
    command = [python, str(packaged / 'code/exp153_h1_release.py'), 'serve',
               '--work', str(packaged), '--competition', str(competition),
               '--output', str(output / 'packaged'),
               '--manifest', str(out / 'dataset/release-manifest.json'), '--local-public3']
    subprocess.run(command, check=True, env=env)
    a, b = output / 'direct/submission.csv', output / 'packaged/submission.csv'
    base.need(a.read_bytes() == b.read_bytes(), 'Public3 local/release CSV bytes differ')
    base.need(local['studies'] == 3 and base.sha(a) == local['submission_sha256'],
              'Public3 receipt/cohort differs')
    result = {'status': 'PASS_PUBLIC3_BYTE_IDENTICAL', 'studies': 3,
              'submission_sha256': base.sha(a), 'local': str(a), 'release': str(b)}
    base.save(output / 'public3-verification.json', result)
    return result


def push_plan(out=OUT):
    out = Path(out)
    base.need((out / 'dataset/release-manifest.json').is_file(), 'Stage first')
    return {'commands': [f'cd "{out / "dataset"}"', 'kaggle datasets create -p .',
                         '# Later revision: kaggle datasets version -p . -m "EXP153 H1 update"',
                         f'cd "{out / "kernel"}"', 'kaggle kernels push -p .',
                         '# Verify private Public3 output before submission; submit only on Taeyang\'s word.'],
            'dataset': DATASET, 'kernel': KERNEL}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
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
    a = parser.parse_args(argv)
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
