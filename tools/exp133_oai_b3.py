"""Local-only EXP133 builder; guarded B3 runtime executes only inside Kaggle.

B3 = .8 rank(OAI) + .1 rank(ConvNeXt) + .1 rank(Rad two-seed TTA).
The original EXP128 builder and public 0.949 code cells are never modified.
No command uploads, pushes, submits, reads credentials, or starts local inference.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import exp128_oai_light as e128

UID, TARGETS = e128.UID, e128.TARGETS
need, sha, write_json = e128.need, e128.sha, e128.write_json
OUT = ROOT / '.local/exp133-oai-b3'
KERNEL = 'taeyangg4/knee-exp133-oai-b3'
# Proposed NEW private dataset: not uploaded or confirmed on the website.
RAD_DATASET = 'taeyangg4/knee-exp133-radr50-2seed'
CUTOFF_S, RESERVE_S = 30_600, 120
OAI_S_PER_STUDY, CNX_S_PER_STUDY = 1.50, 2.05
FIXED_S = e128.FIXED_S  # 600 s per partner for load/install/hash/start-up.
SEEDS = {42: ('EXP113_RADR50_LOCAL_PROTOCOL_V1', 'EXP113_RADR50_LOCAL_F'),
         43: ('EXP118_RADR50_LOCAL_S43_PROTOCOL_V1', 'EXP118_RADR50_LOCAL_S43_F')}
WEIGHTS = {'B3': {'oai': .8, 'cnx': .1, 'rad': .1},
           'B2': {'oai': .9, 'rad': .1}, 'B1': {'oai': .85, 'cnx': .15},
           'pure_oai': {'oai': 1.0}}


def matrix(values, n):
    need(len(values) == n and n > 0 and all(len(r) == len(TARGETS) for r in values),
         'Partner cohort/target shape differs')
    need(all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for r in values for v in r),
         'Partner probabilities invalid')
    return [[float(v) for v in row] for row in values]


def selected_path(rad, cnx):
    return 'B3' if rad is not None and cnx is not None else (
        'B2' if rad is not None else ('B1' if cnx is not None else 'pure_oai'))


def blend(oai, rad=None, cnx=None):
    """EXP128 ranks, independently per target over the entire SAME cohort.

    Do not re-rank the final weighted sum. The Rad input is the 50:50 mean
    of per-seed ranks, which is ranked once here (as in EXP130/131).
    """
    path = selected_path(rad, cnx)
    values = {'oai': matrix(oai, len(oai))}
    if rad is not None:
        values['rad'] = matrix(rad, len(oai))
    if cnx is not None:
        values['cnx'] = matrix(cnx, len(oai))
    if path == 'pure_oai':
        return path, values['oai']
    ranks = {name: [e128.rank_average([r[t] for r in rows]) for t in range(len(TARGETS))]
             for name, rows in values.items()}
    return path, [[sum(w * ranks[name][t][i] for name, w in WEIGHTS[path].items())
                   for t in range(len(TARGETS))] for i in range(len(oai))]


def projection(n=1300):
    need(type(n) is int and n > 0, 'Positive integer study count required')
    # No measured EXP113+EXP118 K36/TTA T4 release exists. EXP128 screen,
    # expanded from five to TEN models; one shared Rad decode, twenty passes.
    decode = 1.3 * 6.52503319258 * n
    forward = 1.3 * 10 * 2 * .25 * n
    stages = {'oai': OAI_S_PER_STUDY * n,
              'rad': decode + forward + FIXED_S,
              'cnx': CNX_S_PER_STUDY * n + FIXED_S}
    return {'studies': n, 'stages_seconds': stages,
            'rad_decode_seconds': decode, 'rad_forward_seconds': forward,
            'fixed_per_partner_seconds': FIXED_S,
            'total_seconds': sum(stages.values()),
            'margin_to_launch_deadline_seconds': CUTOFF_S - RESERVE_S - sum(stages.values()),
            'partner_deadline_seconds': CUTOFF_S - RESERVE_S,
            'publication_reserve_seconds': RESERVE_S, 'cutoff_seconds': CUTOFF_S,
            'basis': {
                'oai': '1.50 s/study, rounded registered rate; EXP130 gold58_timing.json: '
                       '87.1797/58=1.503098 s/study, K94+mirror, 2xT4. Small cohort, not a hidden bound.',
                'cnx': '2.05 s/study including decode; EXP131 gold58_cnx_timing.json: '
                       '118.6461/58=2.045622, three folds, one T4; plus 600 s allowance.',
                'rad': 'EXP128 screen constants, not measured two-seed time: '
                       '1.3*(6.52503319258 decode + 10 models*2 views*0.25)*N+600. '
                       'Nearest actual release: EXP111 out-rad5-v2, EXP107 five models/Public3, '
                       'model_seconds=11.944 and elapsed=50.790 for 3 studies; startup-inclusive '
                       'and different checkpoints, not a transferable two-seed estimate.'}}


def can_start(elapsed, projected):
    return (math.isfinite(elapsed) and math.isfinite(projected) and
            elapsed >= 0 and projected >= 0 and elapsed + projected <= CUTOFF_S - RESERVE_S)


def write_csv(path, ids, values):
    matrix(values, len(ids))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([UID, *TARGETS])
        writer.writerows([[uid, *row] for uid, row in zip(ids, values)])


def sample_ids(sample):
    with Path(sample).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == [UID, *TARGETS], 'Sample target order differs')
        ids = [r[UID] for r in reader]
    need(ids and all(ids) and len(ids) == len(set(ids)), 'Sample IDs invalid')
    return ids


def validate_csv(path, sample):
    ids, values = e128.read_submission(path)
    need(ids == sample_ids(sample), 'CSV IDs/order differ from sample_submission')
    return ids, values


def publish(working, ids, values):
    """Check before an atomic replace; current good submission survives a failure."""
    final = Path(working) / 'submission.csv'
    tmp = Path(working) / 'exp133-final.tmp.csv'
    try:
        write_csv(tmp, ids, values)
        got_ids, got = e128.read_submission(tmp)
        need(got_ids == ids and got == values, 'Blend CSV round trip differs')
        digest = sha(tmp)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    return digest


def input_dir(dataset, marker, input_root=Path('/kaggle/input')):
    candidates = [Path(input_root) / dataset.split('/')[-1],
                  Path(input_root) / 'datasets' / dataset,
                  Path(input_root) / 'competitions' / dataset]
    found = [p for p in candidates if (p / marker).is_file()]
    need(len(found) == 1, 'Expected one input root for ' + dataset)
    return found[0]


def child_timeout(start):
    remaining = CUTOFF_S - RESERVE_S - (time.monotonic() - start)
    need(remaining > 0, 'Partner deadline reached')
    return remaining


def child_env(*python_paths):
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_paths)),
                CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8')


def release_public_cuda_cache():
    """Kaggle stage only: return unused parent allocator memory to the child."""
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def reader_stage(payload, start, ids, record, working):
    release_public_cuda_cache()
    assets = input_dir(e128.READER_DATASET, 'infer.py')
    comp = input_dir(e128.COMPETITION, 'test_series.csv')
    checks = [assets / ('cnxt_v0_fold%d.pt' % i) for i in range(3)]
    need(all(p.is_file() for p in checks), 'ConvNeXt three-fold weights missing')
    need(all((assets / n).is_file() for n in ('preprocess.py', 'knee.py')), 'Reader source missing')
    env_dir = Path(working) / 'exp133_reader_env'
    env_dir.mkdir(exist_ok=True)
    # EXP131 v1.1 / Apex: incompatible cp311/cp312 wheels on Python 3.13
    # are OPTIONAL. The pinned image's existing decoder successfully ran EXP131.
    with (Path(working) / 'exp133_reader_setup.log').open('w', encoding='utf-8') as log:
        install = e128.run_group([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                                  '--target', str(env_dir), '--find-links', str(assets),
                                  'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'],
                                 check=False, timeout=min(900, child_timeout(start)),
                                 stdout=log, stderr=subprocess.STDOUT)
    record['optional_decoder_install_returncode'] = install.returncode
    record['checkpoint_sha256'] = {p.name: sha(p) for p in checks}
    record['source_sha256'] = {n: sha(assets / n) for n in ('infer.py', 'preprocess.py', 'knee.py')}
    out = Path(working) / 'exp133_cnx.csv'
    child = '''import importlib.util, json, pathlib, sys
assets, data, out, ids, *ckpts = sys.argv[1:]
sys.path.insert(0, assets)
spec = importlib.util.spec_from_file_location('infer', str(pathlib.Path(assets) / 'infer.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.run(data, 'test', ckpts, studies=json.loads(ids))
result.to_csv(out)
'''
    with (Path(working) / 'exp133_reader.log').open('w', encoding='utf-8') as log:
        e128.run_group([sys.executable, '-c', child, str(assets), str(comp), str(out),
                        json.dumps(ids), *map(str, checks)],
                       env=child_env(assets, env_dir), check=True, timeout=child_timeout(start),
                       stdout=log, stderr=subprocess.STDOUT)
    got_ids, rows = e128.read_submission(out)
    need(set(got_ids) == set(ids), 'Reader cohort differs')
    record['csv_sha256'] = sha(out)
    by_id = dict(zip(got_ids, rows))
    return [by_id[uid] for uid in ids]


def materialize(payload, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, entry in payload['files'].items():
        source = e128.unpack(entry['z'])
        need(hashlib.sha256(source.encode('utf-8')).hexdigest() == entry['sha256'],
             'Embedded source differs: ' + name)
        need(Path(name).name == name, 'Unsafe embedded filename')
        (directory / name).write_text(source, encoding='utf-8', newline='\n')
    # Frozen raptor_common only needs the marker; no settings are read from it.
    (directory / 'config').mkdir()
    (directory / 'config/workspace.toml').write_text('# EXP133 test-only runtime\n', encoding='utf-8')
    write_json(directory / 'rad-manifest.json', payload['rad_manifest'])


def rad_stage(payload, start, ids, record, working):
    release_public_cuda_cache()
    need(payload is not None, 'Rad payload missing')
    assets = input_dir(RAD_DATASET, 'exp133-rad-assets.json')
    comp = input_dir(e128.COMPETITION, 'test_series.csv')
    manifest = payload['rad_manifest']
    need(json.loads((assets / 'exp133-rad-assets.json').read_text(encoding='utf-8')) == manifest,
         'Rad assets manifest differs')
    src = Path(working) / 'exp133_rad_src'
    materialize(payload, src)
    write_json(src / 'ids.json', ids)
    out = Path(working) / 'exp133_rad'
    environment = child_env(src)
    environment['KNEE_WORKSPACE_ROOT'] = str(src)
    with (Path(working) / 'exp133_rad.log').open('w', encoding='utf-8') as log:
        e128.run_group([sys.executable, str(HERE / 'exp133_oai_b3.py'), 'rad-worker',
                        '--manifest', str(src / 'rad-manifest.json'), '--assets', str(assets),
                        '--source', str(src), '--competition', str(comp),
                        '--ids', str(src / 'ids.json'), '--out', str(out)],
                       check=True, env=environment, timeout=child_timeout(start),
                       stdout=log, stderr=subprocess.STDOUT)
    worker = json.loads((out / 'receipt.json').read_text(encoding='utf-8'))
    need(worker['status'] == 'PASS' and worker['checkpoints'] == manifest['checkpoints']
         and worker['studies'] == len(ids), 'Rad worker receipt differs')
    seed_rows = []
    for seed in SEEDS:
        path = out / ('seed%d.csv' % seed)
        got_ids, rows = e128.read_submission(path)
        need(got_ids == ids and sha(path) == worker['csv_sha256'][str(seed)], 'Rad seed CSV differs')
        seed_rows.append(rows)
    record['worker'] = worker
    partner = e128.rank_blend(*seed_rows, (.5, .5))
    write_csv(Path(working) / 'exp133_rad.csv', ids, partner)
    record['csv_sha256'] = sha(Path(working) / 'exp133_rad.csv')
    return partner


def check_checkpoint(meta, entry):
    expected = {'arch': 'resnet50', 'res': 224, 'lab': TARGETS,
                'contract': 'raptor64_336_c140_s02', 'norm': 'radimagenet',
                'window_policy': 'within-slot-valid', 'input_policy': 'center3',
                'tag': entry['tag'], 'fold': entry['fold']}
    need(all(meta.get(k) == v for k, v in expected.items()), 'Rad checkpoint metadata differs')
    need(meta.get('provenance', {}).get('protocol_sha256') == entry['protocol_sha256'],
         'Rad checkpoint protocol differs')


def rad_worker(args):
    """Kaggle only; frozen decode/input/FP16/TTA functions, current OAI image.

    New orchestration avoids the old worker's Python-3.12-only environment check.
    It does NOT claim cross-environment numerical parity before a Public3 run.
    One DICOM decode/study; ten resident ResNet50 models, one visible GPU.
    """
    need(sys.platform.startswith('linux') and Path('/kaggle/input').is_dir(), 'Kaggle-only worker')
    sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    import timm
    import cv2
    import pydicom
    import exp111_release_v6 as v6
    import exp126_anchor_blend_v8 as e126
    import raptor_runtime as rt
    from raptor_common import load_checkpoint
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    need(sys.version_info[:2] == (3, 13) and torch.cuda.device_count() == 1 and
         'T4' in torch.cuda.get_device_name(0), 'Expected pinned Python 3.13 / one visible T4')
    need(str(torch.__version__) == '2.11.0+cu128' and timm.__version__ == '1.0.29',
         'Pinned OAI torch/timm versions differ')
    rt.configure_runtime('fp16', 'cuda', 'strict')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    ids = json.loads(args.ids.read_text(encoding='utf-8'))
    models = {seed: [] for seed in SEEDS}
    started = time.monotonic()
    for ck in manifest['checkpoints']:
        path = args.assets / ck['path']
        need(path.stat().st_size == ck['bytes'] and sha(path) == ck['sha256'], 'Rad weight hash differs')
        model, meta = load_checkpoint(path)  # trusted own checkpoint, only after full SHA check
        check_checkpoint(meta, ck)
        models[ck['seed']].append(model.eval().to('cuda'))
        del meta
    need(all(len(m) == 5 for m in models.values()), 'Five folds per seed required')
    loaded = time.monotonic()
    corpus = DicomCorpus(args.competition / 'test.csv', args.competition / 'test_series.csv',
                         DirectoryDicomSource(args.competition / 'test_series'))
    need(set(corpus.ids) == set(ids), 'Rad competition cohort differs')
    tta = e126.TTARuntime(rt, True)
    probabilities = {seed: [] for seed in SEEDS}
    decode_s = 0.0
    input_hashes = hashlib.sha256()
    for i, uid in enumerate(ids):
        then = time.monotonic()
        corpus.volume(uid); corpus.mask(uid)
        decode_s += time.monotonic() - then
        # DicomCorpus retains this ONE study; both seeds/views read identical bytes.
        with v6.serving_input(tta, 'center3'):
            for seed, members in models.items():
                fold_prob = []
                for model in members:
                    p, timing = tta.predict(model, corpus, [uid], 36, 224, 'fp16', 'cuda',
                                           norm='radimagenet', batch=1, log=lambda *_: None,
                                           window_policy='within-slot-valid')
                    need(timing['all_finite'] and timing['fp32_fallbacks'] == 0, 'Rad precision failure')
                    fold_prob.append(p[0])
                probabilities[seed].append(np.mean(fold_prob, axis=0).tolist())
        input_hashes.update(json.dumps(corpus.receipts[uid], sort_keys=True).encode())
        if (i + 1) % 50 == 0 or i + 1 == len(ids):
            print('EXP133 Rad', i + 1, '/', len(ids), 'seconds', time.monotonic() - started, flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for seed, values in probabilities.items():
        path = args.out / ('seed%d.csv' % seed)
        write_csv(path, ids, values)
        hashes[str(seed)] = sha(path)
    write_json(args.out / 'receipt.json', {
        'schema': 'exp133-rad-worker-v1', 'status': 'PASS', 'studies': len(ids),
        'checkpoints': manifest['checkpoints'], 'k_eval': 36,
        'fold_reduction': 'mean of five (plain+plane-flip)/2 probabilities per seed',
        'seed_reduction': 'per-target percentile rank mean, 0.5/0.5 (parent)',
        'plane_flip': 'EXP124: coronal/axial horizontal flip; sagittal unchanged',
        'csv_sha256': hashes, 'seconds_total': time.monotonic() - started,
        'seconds_load': loaded - started, 'seconds_decode': decode_s,
        'decode_receipts_sha256': input_hashes.hexdigest(),
        'environment': {'python': sys.version, 'torch': str(torch.__version__),
                        'timm': timm.__version__, 'opencv': cv2.__version__, 'pydicom': pydicom.__version__},
        'training': 0, 'label_reads': 0})


def run_after_oai(payload, start, working='/kaggle/working', sample=None,
                  stages=None, clock=time.monotonic):
    """Immediate safe B2 publication, then B3/B1; every partner failure is caught."""
    working = Path(working)
    final = working / 'submission.csv'
    sample = Path(sample) if sample is not None else input_dir(e128.COMPETITION, 'sample_submission.csv') / 'sample_submission.csv'
    ids, pure = validate_csv(final, sample)
    backup = working / 'exp133_oai_submission.csv'
    shutil.copyfile(final, backup)
    pure_sha = sha(final)
    need(sha(backup) == pure_sha, 'Pure OAI backup differs')
    receipt = {'schema': 'exp133-b3-stages-v1', 'path': 'pure_oai',
               'registered_weights': WEIGHTS, 'final_weights': WEIGHTS['pure_oai'],
               'studies': len(ids), 'pure_oai_sha256': pure_sha,
               'public_source_sha256': e128.SOURCE_SHA,
               'rad_manifest_sha256': hashlib.sha256(json.dumps(payload['rad_manifest'], sort_keys=True).encode()).hexdigest(),
               'projection': projection(len(ids)), 'stages': {},
               'oai_elapsed_seconds': clock() - start, 'final_sha256': pure_sha}
    partners = {'rad': None, 'cnx': None}
    stages = stages or {'rad': rad_stage, 'cnx': reader_stage}

    def save():
        receipt['elapsed_seconds'] = clock() - start
        try:
            write_json(working / 'exp133_stage_timing.json', receipt)
        except BaseException as exc:
            print('EXP133 receipt write failed:', repr(exc), flush=True)

    save()
    for name in ('rad', 'cnx'):
        elapsed = clock() - start
        record = {'started_elapsed_seconds': elapsed,
                  'projection_seconds': projection(len(ids))['stages_seconds'][name]}
        receipt['stages'][name] = record
        if not can_start(elapsed, record['projection_seconds']):
            record.update(status='SKIPPED', reason='elapsed + projection > 30600 - 120 seconds')
            save()
            continue
        alarm = sys.platform.startswith('linux') and hasattr(signal, 'setitimer')
        old = None
        try:
            if alarm:
                old = signal.signal(signal.SIGALRM, e128._timeout)
                signal.setitimer(signal.ITIMER_REAL, CUTOFF_S - RESERVE_S - elapsed)
            record['status'] = 'RUNNING'
            save()
            value = matrix(stages[name](payload, start, ids, record, working), len(ids))
            need(clock() - start <= CUTOFF_S - RESERVE_S, 'Partner completed after deadline')
            partners[name] = value
            record['status'] = 'PASS'
        except BaseException as exc:
            record.update(status='FAILED', reason=type(exc).__name__ + ': ' + str(exc),
                          traceback_tail=traceback.format_exc()[-2000:])
            print('EXP133', name, 'fallback:', record['reason'], flush=True)
        finally:
            if alarm and old is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
        record['seconds'] = clock() - start - elapsed
        try:
            path, result = blend(pure, **partners)
            if path != 'pure_oai':
                receipt['final_sha256'] = publish(working, ids, result)
                receipt['path'], receipt['final_weights'] = path, WEIGHTS[path]
        except BaseException as exc:
            # Publish validates a temporary file before replacement: previous
            # pure/B2 submission stays intact, including when CNX fails later.
            record['publication_error'] = type(exc).__name__ + ': ' + str(exc)
            record['status'] = 'FAILED'
            partners[name] = None
        save()
    return receipt


def payload():
    """Bind exact EXP130 weights without unpickling anything locally."""
    reference = ROOT / '.local/exp130-gold58/local/receipt.json'
    gold = json.loads(reference.read_text(encoding='utf-8'))
    checkpoints = []
    for seed, (protocol, prefix) in SEEDS.items():
        protocol_sha = sha(ROOT / 'artifacts/experiments' / protocol / 'manifest.json')
        need(protocol_sha == gold['models'][str(seed)]['protocol_sha256'], 'Gold58 protocol changed')
        for fold, digest in enumerate(gold['models'][str(seed)]['checkpoints_sha256']):
            source = ROOT / 'artifacts/experiments' / (prefix + str(fold)) / 'model_best.pt'
            need(source.is_file(), 'Missing local checkpoint: ' + str(source))
            checkpoints.append({'seed': seed, 'fold': fold, 'tag': prefix + str(fold),
                                'path': 'checkpoints/rad%d_f%d.pt' % (seed, fold),
                                'sha256': digest, 'bytes': source.stat().st_size,
                                'protocol_sha256': protocol_sha})
    need(len(checkpoints) == 10, 'Exactly ten Rad checkpoints required')
    old = ROOT / '.local/exp089-recipe-transfer/t4-v3/dataset/training/src'
    paths = {n: old / n for n in ('raptor_common.py', 'raptor_dicom.py', 'raptor_runtime.py')}
    paths.update({n: ROOT / 'kaggle/packages/KNEE_RELEASE_ASSETS_V8/code' / n
                  for n in ('raptor_data.py', 'raptor_radimagenet.py')})
    paths.update({n: HERE / n for n in ('exp111_release_v6.py', 'exp126_anchor_blend_v8.py')})
    # Bind frozen serving and centre-slice code to the reviewed release, not
    # mutable training/src. Only LF-normalisation changes embedded file bytes.
    protocol = json.loads((old.parents[1] / 'exp089-t4-protocol.json').read_text(encoding='utf-8'))
    for name in ('raptor_common.py', 'raptor_dicom.py', 'raptor_runtime.py'):
        need(sha(paths[name]) == protocol['files']['training/src/' + name], 'Frozen serving code changed')
    v6 = e128._load('exp133_v6_contract', HERE / 'exp111_release_v6.py',
                    '4e9a74445f5c558cffa56e16f4cd22d7676e444f4a304e3ab30d7badbbd1b36b')
    for name, digest in v6.CODE_FILES.items():
        need(sha(paths[name]) == digest, 'Frozen center3 code changed')
    files = {}
    for name, path in paths.items():
        source = path.read_text(encoding='utf-8')
        compile(source, name, 'exec')
        files[name] = {'sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(), 'z': e128.packed(source)}
    return {'rad_manifest': {'schema': 'exp133-rad-assets-v1', 'dataset': RAD_DATASET,
                            'gold58_receipt_sha256': sha(reference), 'checkpoints': checkpoints},
            'files': files}


def metadata():
    meta = e128.metadata('blend_b1')
    meta.update(id=KERNEL, title=KERNEL.split('/')[1])
    meta['dataset_sources'].append(RAD_DATASET)
    return meta


def literal_bytes(source):
    """Short literal lines avoid Kaggle's long-cell-line startup failure."""
    encoded = e128.packed(source)
    return '(\n' + ''.join('    ' + repr(encoded[i:i+4000]) + '\n' for i in range(0, len(encoded), 4000)) + ')'


def notebook(data):
    public, code = e128.load_source()
    source128 = (HERE / 'exp128_oai_light.py').read_text(encoding='utf-8')
    source133 = Path(__file__).read_text(encoding='utf-8')
    setup = ('import base64, zlib, pathlib, sys\n'
             'def _exp133_write(name, value):\n'
             '    pathlib.Path("/kaggle/working", name).write_text(zlib.decompress(base64.b64decode(value)).decode(), encoding="utf-8", newline="\\n")\n'
             '_exp133_write("exp128_oai_light.py", ' + literal_bytes(source128) + ')\n'
             '_exp133_write("exp133_oai_b3.py", ' + literal_bytes(source133) + ')\n'
             'sys.path.insert(0, "/kaggle/working")\n'
             'import exp133_oai_b3 as _exp133\n'
             '_EXP133_PAYLOAD = json.loads(_exp133.e128.unpack(' + literal_bytes(json.dumps(data)) + '))\n'
             '_EXP133_RECEIPT = _exp133.run_after_oai(_EXP133_PAYLOAD, _EXP133_START)\n'
             'print("EXP133 served:", _EXP133_RECEIPT["path"], _EXP133_RECEIPT["final_sha256"])\n')
    # All code cells are plain Python (no %%writefile), so compile() catches
    # cell-generation mistakes, including the EXP131 inline-comment class.
    tail = ('try:\n    import json\n' + ''.join('    ' + line for line in setup.splitlines(keepends=True)) +
            'except BaseException as _exp133_error:\n'
            '    print("EXP133 keeps current valid submission:", type(_exp133_error).__name__, str(_exp133_error), flush=True)\n'
            '    try:\n'
            '        import json, pathlib, hashlib\n'
            '        _receipt = pathlib.Path("/kaggle/working/exp133_stage_timing.json")\n'
            '        if not _receipt.exists():\n'
            '            _receipt.write_text(json.dumps({"schema": "exp133-bootstrap-error-v1", "path": "pure_oai", "reason": str(_exp133_error), "elapsed_seconds": _exp133_time.monotonic() - _EXP133_START, "final_sha256": hashlib.sha256(pathlib.Path("/kaggle/working/submission.csv").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")\n'
            '    except BaseException:\n'
            '        pass\n')
    credit = (e128.credit().replace('# EXP128: public OAI-trained checkpoint and light blends', '# EXP133: registered B3')
              .replace("in B1.", "in B3/B1."))
    credit += ('\nRegistered B3: 0.8 rank(0.949) + 0.1 rank(ConvNeXt F0-F2 uniform) + '
               '0.1 rank(RadR50 EXP113 seed42 + EXP118 seed43, F0-F4 each, K36, EXP124 flip). '
               'Fallbacks: Rad only B2 0.9/0.1; reader only B1 0.85/0.15; neither pure 0.949. '
               'OAI/gray-area derived submission requires explicit user approval. '
               'Own RadImageNet initialization provenance: marwanmath/resnet-50-radimagenet-marwan, '
               'CC BY-NC-SA Kaggle copy. The Python-3.13 Rad adapter requires Public3 validation.\n')
    cells = [e128.cell('import time as _exp133_time\n_EXP133_START = _exp133_time.monotonic()\n'),
             e128.cell(credit, 'markdown'), *copy.deepcopy(public['cells'][:3]),
             *copy.deepcopy(code), e128.cell(tail)]
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), '<EXP133 cell %d>' % i, 'exec')
    need(cells[5:7] == code, 'Public code changed')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}}, 'cells': cells}


def screen_spec(dest, data):
    return {'schema': 'exp133-submission-screen-v1', 'candidate': 'EXP133_OAI_B3',
            'variant': 'blend_b3', 'basis': 'PUBLIC_ONLY', 'submission_state': 'UNPUSHED',
            'private_kernel': True, 'submit_candidate': True, 'expected_studies_public3': 3,
            'targets': TARGETS, 'members': [
                {'name': 'oai', 'arm': 'oai', 'weight': .8, 'oof': None,
                 'oof_reason': 'Public all-data OAI checkpoint: no held-out OOF', 'checkpoint_sha256': e128.WEIGHT_SHA},
                {'name': 'cnx', 'arm': 'cnx', 'weight': .1, 'oof': None,
                 'oof_reason': 'Public 3-fold reader; no local OOF; hashes recovered by Public3 receipt'},
                {'name': 'rad', 'arm': 'rad', 'weight': .1, 'oof': None,
                 'oof_reason': 'No OOF for this OAI composite; seed weak OOF is context only',
                 'checkpoint_sha256': [c['sha256'] for c in data['rad_manifest']['checkpoints']]}],
            'release': {'kernel_dir': str(dest), 'out_dir': str(OUT / 'public3-output')},
            'decision': {'plan_item': 'H-B-58', 'basis': 'public',
                         'text': 'Registered B3 only; explicit user candidate approval required; never retune weights.'},
            'blend': {'weights': WEIGHTS, 'rank': 'EXP128 per-target average-tie percentile, entire test cohort',
                      'seed_reduction': '0.5 rank(s42 five-fold TTA mean) + 0.5 rank(s43 five-fold TTA mean)'},
            'approval_required': True,
            'screen_limitation': 'v7 accepts this spec but has no EXP133 receipt adapter; keep HOLD, '
                                 'do not invent a score decision/registered_at or treat this build as admission. '
                                 'validate-public3 verifies integration separately; B owns final screen integration.'}


def build(out=OUT):
    out = Path(out)
    need(not out.exists(), 'Choose a fresh build directory')
    data = payload()
    nb = notebook(data)
    dest = out / 'blend_b3'
    dest.mkdir(parents=True)
    write_json(dest / 'release.ipynb', nb)
    write_json(dest / 'kernel-metadata.json', metadata())
    write_json(dest / 'submission_screen_spec.json', screen_spec(dest, data))
    write_json(dest / 'runtime-projection-1300.json', projection())
    asset_dir = out / 'required-assets'
    asset_dir.mkdir()
    write_json(asset_dir / 'exp133-rad-assets.json', data['rad_manifest'])
    uploads = []
    for ck in data['rad_manifest']['checkpoints']:
        uploads.append({'source': str(ROOT / 'artifacts/experiments' / ck['tag'] / 'model_best.pt'),
                        'destination': ck['path'], 'sha256': ck['sha256'], 'bytes': ck['bytes']})
    uploads.append({'source': str(asset_dir / 'exp133-rad-assets.json'),
                    'destination': 'exp133-rad-assets.json', 'sha256': sha(asset_dir / 'exp133-rad-assets.json')})
    write_json(out / 'required-upload-files.json', {
        'dataset': RAD_DATASET, 'private_required': True, 'status': 'NOT_UPLOADED', 'files': uploads,
        'evidence': 'No local release package contains EXP113+EXP118. V8 has EXP107, not these models. '
                    'No live Kaggle query made. Stored EXP130 hashes are checked at Kaggle load; '
                    'local weight contents are not reread/unpickled by this builder.'})
    report = {'schema': 'exp133-build-v1', 'status': 'UNPUSHED_REQUIRES_RAD_DATASET',
              'kernel': KERNEL, 'public_code_cells_unchanged': True,
              'source128_sha256': sha(HERE / 'exp128_oai_light.py'),
              'source133_sha256': sha(Path(__file__)),
              'public_source_sha256': e128.SOURCE_SHA, 'oai_weight_sha256': e128.WEIGHT_SHA,
              'files_sha256': {p.name: sha(p) for p in dest.iterdir() if p.is_file()},
              'rad_manifest_sha256': hashlib.sha256(json.dumps(data['rad_manifest'], sort_keys=True).encode()).hexdigest(),
              'embedded_source_sha256': {name: value['sha256'] for name, value in data['files'].items()},
              'network_calls': 0, 'gpu_runs': 0, 'submissions': 0,
              'runtime_projection_1300': projection()}
    write_json(dest / 'build-receipt.json', report)
    return report


def validate_public3(output, sample, build_dir=OUT / 'blend_b3', allow_fallback=False):
    output, build_dir = Path(output), Path(build_dir)
    ids, final = validate_csv(output / 'submission.csv', sample)
    need(len(ids) == 3, 'Public3 requires exactly three studies')
    r = json.loads((output / 'exp133_stage_timing.json').read_text(encoding='utf-8'))
    build_record = json.loads((build_dir / 'build-receipt.json').read_text(encoding='utf-8'))
    need(r['schema'] == 'exp133-b3-stages-v1' and r['registered_weights'] == WEIGHTS, 'Receipt weights differ')
    need(r['rad_manifest_sha256'] == build_record['rad_manifest_sha256'], 'Rad build/receipt binding differs')
    need(r['final_sha256'] == sha(output / 'submission.csv'), 'Final CSV hash differs')
    need(r['pure_oai_sha256'] == sha(output / 'exp133_oai_submission.csv'), 'Pure OAI backup hash differs')
    _, pure = validate_csv(output / 'exp133_oai_submission.csv', sample)
    partners = {}
    for name in ('rad', 'cnx'):
        if r['stages'][name]['status'] == 'PASS':
            path = output / ('exp133_%s.csv' % name)
            own_ids, own = e128.read_submission(path)
            need(set(own_ids) == set(ids) and sha(path) == r['stages'][name]['csv_sha256'], 'Partner CSV binding differs')
            by_id = dict(zip(own_ids, own))
            partners[name] = [by_id[i] for i in ids]
    path, expected = blend(pure, **partners)
    need(r['path'] == path and r['final_weights'] == WEIGHTS[path], 'Fallback path/weights differ')
    need(final == expected, 'Final blend arithmetic differs')
    need(allow_fallback or path == 'B3', 'Valid fallback CSV, but B3 integration NOT demonstrated: ' + path)
    return {'status': 'PASS_B3_PUBLIC3' if path == 'B3' else 'PASS_FALLBACK_PUBLIC3',
            'path': path, 'studies': 3, 'targets': len(TARGETS),
            'submission_sha256': sha(output / 'submission.csv'), 'submission_authorized': False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--out', type=Path, default=OUT)
    v = sub.add_parser('validate-public3')
    v.add_argument('--output', type=Path, default=OUT / 'public3-output')
    v.add_argument('--sample', type=Path, default=ROOT / 'storage/competition/sample_submission.csv')
    v.add_argument('--build-dir', type=Path, default=OUT / 'blend_b3')
    v.add_argument('--allow-fallback', action='store_true')
    w = sub.add_parser('rad-worker')
    for name in ('manifest', 'assets', 'source', 'competition', 'ids', 'out'):
        w.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args(argv)
    if a.command == 'rad-worker':
        return rad_worker(a)
    result = build(a.out) if a.command == 'build' else validate_public3(
        a.output, a.sample, a.build_dir, a.allow_fallback)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
