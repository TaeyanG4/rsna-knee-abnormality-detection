"""EXP128 local-only builder and guarded Kaggle runtime: fidelity, B1, B2.

The downloaded nartaa notebook is untrusted input: this builder parses its JSON,
copies its two code cells verbatim, and never imports or executes either locally.
Only ``build`` and ``compare`` are intended for local invocation. Kaggle invokes
``run_after_oai`` after the copied public inference has written submission.csv.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import importlib.util
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
import zlib

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'storage/external/nartaa__rsna-knee-0949-anatomical-mirror/rsna-knee-0949-anatomical-mirror.ipynb'
SOURCE_SHA = '8f1f96530b08543bd6caeed48489eea7c01657d6994f22e21a8400047fd4e100'
WEIGHT_SHA = '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
WEIGHT_NAME = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
DATASET = 'nartaa/rsna-knee-publication-swa-weights-20261007'
READER_DATASET = 'goodpjw2008/rsna-knee-2-5d-convnext-reader'
COMPETITION = 'rsna-knee-abnormality-detection'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
KERNELS = {'fidelity': 'knee-exp128-oai-fidelity',
           'blend_b1': 'knee-exp128-oai-reader-b1', 'blend_b2': 'knee-exp128-oai-rad-b2'}
# Fixed H-B-58 REVISED weights; no target-specific or public-score tuning.
WEIGHTS = {'blend_b1': (0.85, 0.15), 'blend_b2': (0.9, 0.1)}
CUTOFF_S = 30_600
RAD_DECODE_S = 1.3 * 6.52503319258 * 1300  # own DICOM decode, not shared with 0.949
RAD_TTA_S = 2 * 1.3 * 5 * 0.25 * 1300
FIXED_S = 600
SOURCE_FILES_B = ('exp126_anchor_blend_v8.py', 'exp119_release_v7.py',
                  'exp111_release_v6.py', 'exp105_combined_release.py',
                  'raptor_shared_input.py')
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
OUT = ROOT / '.local/exp128-oai-light'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load_source():
    need(sha(SOURCE) == SOURCE_SHA, 'Downloaded public notebook SHA differs')
    nb = json.loads(SOURCE.read_text(encoding='utf-8'))
    need(len(nb.get('cells', [])) == 5 and [c['cell_type'] for c in nb['cells']] ==
         ['markdown', 'markdown', 'markdown', 'code', 'code'], 'Public notebook cells differ')
    code = [copy.deepcopy(c) for c in nb['cells'][3:]]
    need(WEIGHT_SHA in ''.join(code[0]['source']) and 'if __name__ == "__main__":' in ''.join(code[1]['source']),
         'Public code guard/entrypoint differs')
    return nb, code


def cell(source, kind='code'):
    return {'cell_type': kind, 'metadata': {}, 'source': source,
            **({'execution_count': None, 'outputs': []} if kind == 'code' else {})}


def packed(value):
    return base64.b64encode(zlib.compress(value.encode('utf-8'), 9)).decode('ascii')


def unpack(value):
    return zlib.decompress(base64.b64decode(value)).decode('utf-8')


def credit():
    return ("# EXP128: public OAI-trained checkpoint and light blends\n\n"
            "Checkpoint: `" + WEIGHT_NAME + "` (SHA-256 `" + WEIGHT_SHA + "`). "
            "Published by **nartaa**, using the **Raptor CoAtNet/MIL recipe by dreaddevelopment**. "
            "Its training included Osteoarthritis Initiative (OAI) data (NIMH Data Archive acknowledgement, "
            "DOI 10.15154/0hcg-f676). The published weight dataset states research/educational model-use "
            "permission and a `licences=[other]` tag; this is separate from the notebook code licence. "
            "No OAI source images or reports are loaded here. The checkpoint dataset is `" + DATASET + "`. "
            "The original notebook code and the ConvNeXt reader dataset publish Apache License, "
            "Version 2.0 (Apache 2.0) notices (http://www.apache.org/licenses/LICENSE-2.0); "
            "their source attribution is preserved here; the reader scripts are used from "
            "goodpjw2008's attached dataset in B1. "
            "Reader author: **goodpjw2008**, dataset `" + READER_DATASET + "`. "
            "The three original notebook markdown cells follow. This private notebook does not claim "
            "to have trained either checkpoint.\n\n"
            "Original notebook: `nartaa/rsna-knee-0949-anatomical-mirror`, SHA-256 `" + SOURCE_SHA + "`. "
            "Original pinned Docker image `" + DOCKER + "` is retained. "
            "The original qualification reported torch 2.11.0+cu128 and timm 1.0.29; "
            "the runtime receipt records actual versions.\n")


def metadata(variant):
    sources = [DATASET]
    if variant == 'blend_b1':
        sources += [READER_DATASET]
    if variant == 'blend_b2':
        sources += ['taeyangg4/knee-exp089-recipe-transfer-v1',
                    'taeyangg4/knee-release-assets-v8',
                    'mattiaangeli/opencv-python-headless-4120088-x86']
    return {'id': 'taeyangg4/' + KERNELS[variant], 'title': KERNELS[variant],
            'code_file': 'release.ipynb', 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': False, 'dataset_sources': list(dict.fromkeys(sources)),
            'competition_sources': [COMPETITION], 'kernel_sources': [], 'model_sources': [],
            'machine_shape': 'NvidiaTeslaT4', 'docker_image': DOCKER}

def compare_csv(ours, public):
    def read(path):
        with Path(path).open(newline='', encoding='utf-8-sig') as f:
            r = csv.DictReader(f)
            need(r.fieldnames == [UID, *TARGETS], 'CSV column order differs: ' + str(path))
            rows = list(r)
        need(len(rows) == len({r[UID] for r in rows}), 'Duplicate IDs: ' + str(path))
        for r in rows:
            need(all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1 for t in TARGETS),
                 'Invalid probability: ' + str(path))
        return rows
    a, b = read(ours), read(public)
    need(len(a) == len(b) and {r[UID] for r in a} == {r[UID] for r in b}, 'CSV ID sets differ')
    by = {r[UID]: r for r in b}
    mismatches = [(r[UID], t, r[t], by[r[UID]][t]) for r in a for t in TARGETS
                  if r[t] != by[r[UID]][t]]
    return {'exact': not mismatches, 'studies': len(a), 'cells': len(a) * len(TARGETS),
            'mismatch_count': len(mismatches),
            'max_abs_diff': max((abs(float(x)-float(y)) for _, _, x, y in mismatches), default=0.0),
            'ours_sha256': sha(ours), 'public_sha256': sha(public)}


def fidelity_receipt():
    """Runs on Kaggle after the exact public cells, including their CSV write."""
    import torch
    import timm
    output = Path('/kaggle/working/submission.csv')
    need(output.is_file(), 'Public source did not publish submission.csv')
    with output.open(newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        need(r.fieldnames == [UID, *TARGETS], 'Public output columns differ')
        rows = list(r)
    need(rows and len(rows) == len({r[UID] for r in rows}), 'Public output cohort invalid')
    values = {r[UID]: [float(r[t]) for t in TARGETS] for r in rows}
    need(all(math.isfinite(v) and 0 <= v <= 1 for row in values.values() for v in row), 'Public output nonfinite')
    record = {'schema': 'exp128-fidelity-v1', 'public_source_sha256': SOURCE_SHA,
              'checkpoint_sha256': WEIGHT_SHA, 'submission_sha256': sha(output),
              'columns': [UID, *TARGETS], 'studies': len(rows), 'probabilities': values,
              'input_volume_hashes': None,
              'input_volume_hashes_reason': 'Public source does not retain decoded volumes after main(); no extra decode in fidelity run',
              'environment': {'python': sys.version, 'torch': str(torch.__version__),
                              'timm': str(timm.__version__),
                              'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
              'original_qualified_environment': {'torch': '2.11.0+cu128', 'timm': '1.0.29',
                  'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'}}
    write_json('/kaggle/working/exp128_fidelity.json', record)
    return record


def rank_average(values):
    """Average-tie percentile ranks, one target. No NumPy/CUDA dependency."""
    need(values and all(math.isfinite(float(v)) for v in values), 'Rank inputs must be finite')
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    at = 0
    while at < len(order):
        end = at + 1
        while end < len(order) and values[order[end]] == values[order[at]]:
            end += 1
        pct = ((at + end - 1) / 2 + 1) / len(order)
        for i in order[at:end]:
            out[i] = pct
        at = end
    return out


def rank_blend(first, second, weights):
    need(len(first) == len(second) and first and len(weights) == 2 and
         all(w >= 0 and math.isfinite(w) for w in weights) and sum(weights) > 0, 'Blend inputs invalid')
    need(all(len(row) == len(TARGETS) for row in first+second), 'Target count differs')
    n = len(first)
    out = [[0.0] * len(TARGETS) for _ in range(n)]
    for t in range(len(TARGETS)):
        a = rank_average([row[t] for row in first])
        b = rank_average([row[t] for row in second])
        for i in range(n):
            out[i][t] = (weights[0]*a[i] + weights[1]*b[i]) / sum(weights)
    return out


def read_submission(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        need(next(reader) == [UID, *TARGETS], 'Submission columns differ')
        rows = list(reader)
    ids = [r[0] for r in rows]
    need(ids and len(ids) == len(set(ids)), 'Duplicate or empty study IDs')
    need(all(len(r) == len(TARGETS)+1 for r in rows), 'Submission row width differs')
    values = [[float(v) for v in r[1:]] for r in rows]
    need(all(math.isfinite(v) and 0 <= v <= 1 for row in values for v in row), 'Submission probabilities invalid')
    return ids, values


def publish(ids, first, second, weights, expected_sha, backup, working='/kaggle/working'):
    """Atomic final write; restore the verified pure 0.949 output on any failure."""
    final = Path(working) / 'submission.csv'
    need(sha(backup) == expected_sha, 'Pure OAI backup differs')
    need(len(ids) == len(first) == len(second), 'Blend cohort differs')
    blend = rank_blend(first, second, weights)
    tmp = final.with_name('exp128-blend.tmp.csv')
    need(not tmp.exists(), 'Old blend temporary output exists')
    try:
        with tmp.open('x', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([UID, *TARGETS])
            w.writerows([[uid, *row] for uid, row in zip(ids, blend)])
        got_ids, got = read_submission(tmp)
        need(got_ids == ids and got == blend, 'Blend round-trip differs')
        # Another stage may have overwritten submission.csv, but the pure backup is stable.
        os.replace(tmp, final)
        need(read_submission(final)[0] == ids, 'Published cohort differs')
        return sha(final)
    except BaseException:
        if tmp.exists(): tmp.unlink()
        shutil.copyfile(backup, final)
        need(sha(final) == expected_sha, 'Could not restore pure OAI submission')
        raise


def _load(name, path, expected=None):
    if expected: need(sha(path) == expected, 'Pinned module changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timeout(_signum, _frame):
    raise TimeoutError('EXP128 wall-clock cutoff reached')


def run_group(command, *, timeout, check=False, env=None, stdout=None, stderr=None):
    """One child process group; a timeout/alarm removes DataLoader descendants too."""
    proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                            start_new_session=True)
    try:
        code = proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=60)
    if check and code:
        raise subprocess.CalledProcessError(code, command)
    return subprocess.CompletedProcess(command, code)


def _rad_stage(payload, start, ids, receipt):
    """EXP126 Rad worker with no shared store; its verified DICOM fallback is charged."""
    import gc
    import torch
    elapsed = time.monotonic() - start
    projected = RAD_DECODE_S + RAD_TTA_S + FIXED_S
    receipt['rad_projection_seconds'] = projected
    need(elapsed + projected <= CUTOFF_S - 120,
         'B RadR50+TTA projection exceeds registered cutoff minus publication reserve')
    root = Path('/kaggle/working/exp128')
    src = root / 'src'; src.mkdir(parents=True, exist_ok=False)
    for name, item in payload['files'].items():
        data = unpack(item['z'])
        need(hashlib.sha256(data.encode()).hexdigest() == item['sha256'], 'Embedded source differs: '+name)
        (src / name).write_text(data, encoding='utf-8')
    manifest = src / 'exp128-rad-manifest.json'
    manifest.write_text(unpack(payload['manifest_z']), encoding='utf-8')
    need(sha(manifest) == payload['manifest_sha256'], 'Rad manifest differs')
    e105 = _load('exp105_for_exp128_rad', src / 'exp105_combined_release.py',
                 '4c0d227c3d0d66d0e40821994ff125f84e78b6f964c240c5e052d06df76c3fd2')
    e126 = _load('exp126_for_exp128_rad', src / 'exp126_anchor_blend_v8.py', payload['files']['exp126_anchor_blend_v8.py']['sha256'])
    release = json.loads(manifest.read_text(encoding='utf-8'))
    input_root = Path('/kaggle/input')
    old = e105.find_one([input_root / 'datasets/taeyangg4/knee-exp089-recipe-transfer-v1',
                         input_root / 'knee-exp089-recipe-transfer-v1'], 'EXP089_RECIPE_TRANSFER_PROTOCOL_V1.json')
    # The verifier below checks the pinned protocol and wheel hashes.
    wheel_dir = e105.find_one([input_root / 'datasets/mattiaangeli/opencv-python-headless-4120088-x86',
                               input_root / 'opencv-python-headless-4120088-x86'],
                              'opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.whl')
    wheel = next(wheel_dir.glob('opencv_python_headless-4.12.0.88-*.whl'))
    e105.install_env(_load('exp111_for_exp128', src / 'exp111_release_v6.py'), old, wheel,
                     root / 'env', runner=run_group)
    comp = next((p for p in [input_root / 'competitions' / COMPETITION, input_root / COMPETITION]
                 if (p / 'test_series.csv').is_file()), None)
    need(comp is not None, 'Competition test series unavailable')
    (root / 'ids.json').write_text(json.dumps(ids), encoding='utf-8')
    gc.collect(); torch.cuda.empty_cache()
    cmd = [sys.executable, str(src / 'exp126_anchor_blend_v8.py'), 'worker',
           '--manifest', str(manifest), '--sha256', sha(manifest), '--assets', str(old),
           '--env', str(root / 'env'), '--store', str(root / 'absent-store'),
           '--ids-json', str(root / 'ids.json'), '--rows-csv', str(comp / 'test_series.csv'),
           '--dicom-root', str(comp / 'test_series'), '--out', str(root / 'rad-worker'),
           '--device', 'cuda', '--amp', 'fp16', '--plane-flip-tta']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    limit = max(1, int(CUTOFF_S - 300 - (time.monotonic() - start)))
    with (root / 'rad-worker.log').open('w', encoding='utf-8') as log:
        result = run_group(cmd, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
    own, worker = e126.checked_own(release, sha(manifest), root / 'rad-worker', ids,
                                    result.returncode, plane_flip_tta=True)
    need(set(own) == set(range(len(ids))) and not worker.get('unscored'), 'Rad worker coverage incomplete')
    receipt['rad_worker'] = worker
    return [own[i] for i in range(len(ids))]


def _reader_stage(start, ids, receipt):
    """Run the author's dataset infer.py in a child; it decodes its own DICOMs."""
    input_root = Path('/kaggle/input')
    patterns = ('*/{name}', '*/*/{name}', '*/*/*/{name}')
    candidates = [p for pattern in patterns
                  for p in input_root.glob(pattern.format(name='cnxt_v0_fold0.pt'))]
    need(len(candidates) == 1, 'Expected one ConvNeXt reader dataset')
    assets = candidates[0].parent
    checks = [assets / ('cnxt_v0_fold%d.pt' % i) for i in range(3)]
    need(all(p.is_file() for p in checks), 'ConvNeXt F0-F2 weights missing')
    for name in ('infer.py', 'preprocess.py', 'knee.py'):
        need((assets / name).is_file(), 'Reader source missing: ' + name)
    comps = [p.parent for pattern in patterns
             for p in input_root.glob(pattern.format(name='test_series.csv'))
             if (p.parent / 'test.csv').is_file()]
    need(len(comps) == 1, 'Expected one competition test root')
    # K16, six slots, 3 folds; an engineering estimate, never a measured bound.
    projected = 600 + len(ids) * (8.5 + 3 * 6 * 16 * 0.02)
    receipt['reader_projection_seconds'] = projected
    need(time.monotonic() - start + projected <= CUTOFF_S - 120,
         'Reader projected to exceed registered wall-clock cutoff')
    import torch
    torch.cuda.empty_cache()
    env_dir = Path('/kaggle/working/exp128_reader_env')
    env_dir.mkdir(exist_ok=False)
    wheels = sorted(assets.glob('pylibjpeg*.whl'))
    need(len(wheels) >= 3, 'Reader decoder wheels missing')
    setup_timeout = max(1, int(CUTOFF_S - 300 - (time.monotonic() - start)))
    run_group([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
               '--target', str(env_dir), *map(str, wheels)], check=True, timeout=min(900, setup_timeout),
              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    out = Path('/kaggle/working/exp128_reader.csv')
    env = dict(os.environ, PYTHONPATH=str(assets) + os.pathsep + str(env_dir) + os.pathsep +
               os.environ.get('PYTHONPATH', ''), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2')
    cmd = [sys.executable, str(assets / 'infer.py'), str(comps[0]), 'test', str(out), *map(str, checks)]
    timeout = max(1, int(CUTOFF_S - 120 - (time.monotonic() - start)))
    with Path('/kaggle/working/exp128_reader.log').open('w', encoding='utf-8') as log:
        run_group(cmd, env=env, check=True, timeout=timeout, stdout=log, stderr=subprocess.STDOUT)
    own_ids, own = read_submission(out)
    need(set(own_ids) == set(ids), 'Reader cohort differs')
    by_id = dict(zip(own_ids, own))
    receipt['reader_csv_sha256'] = sha(out)
    receipt['reader_checkpoint_sha256'] = {p.name: sha(p) for p in checks}
    return [by_id[i] for i in ids]


def run_after_oai(variant, payload=None, start=None, working='/kaggle/working'):
    """Blend after pure 0.949 CSV is published; every stage failure keeps its bytes."""
    start = time.monotonic() if start is None else start
    final = Path(working) / 'submission.csv'
    backup = Path(working) / 'exp128_oai_submission.csv'
    receipt = {'schema': 'exp128-light-blend-v1', 'variant': variant,
               'weights': list(WEIGHTS.get(variant, ())), 'cutoff_seconds': CUTOFF_S,
               'path': 'pure_oai', 'reason': None}
    pure_sha = None
    backup_good = False
    try:
        need(variant in WEIGHTS, 'Unknown blend variant')
        ids, pure = read_submission(final)
        pure_sha = sha(final)
        shutil.copyfile(final, backup)
        need(sha(backup) == pure_sha, 'Pure OAI backup differs')
        backup_good = True
        receipt['pure_oai_sha256'] = pure_sha
        remaining = CUTOFF_S - 120 - (time.monotonic() - start)
        need(remaining > 0, 'Registered cutoff reached before partner stage')
        alarm = sys.platform.startswith('linux') and hasattr(signal, 'setitimer')
        if alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout)
            signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            second = (_reader_stage(start, ids, receipt) if variant == 'blend_b1' else
                      _rad_stage(payload, start, ids, receipt))
            need(time.monotonic() - start <= CUTOFF_S - 120, 'No publication allowance remains')
            receipt['final_sha256'] = publish(ids, pure, second, WEIGHTS[variant], pure_sha, backup, working)
            receipt['path'] = 'blended'
        finally:
            if alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
    except BaseException as exc:
        receipt['reason'] = type(exc).__name__ + ': ' + str(exc)
        receipt['traceback_tail'] = traceback.format_exc()[-2000:]
        if backup_good and backup.is_file():
            try:
                shutil.copyfile(backup, final)
                need(sha(final) == pure_sha, 'Pure OAI restoration differs')
            except BaseException as restore_exc:
                receipt['restore_error'] = type(restore_exc).__name__ + ': ' + str(restore_exc)
    receipt['elapsed_seconds'] = time.monotonic() - start
    try:
        receipt['final_sha256'] = sha(final) if final.is_file() else None
        write_json(Path(working) / 'exp128_blend_receipt.json', receipt)
    except BaseException as diagnostic_exc:
        print('EXP128 diagnostic write failed:', type(diagnostic_exc).__name__, str(diagnostic_exc), flush=True)
    return receipt


def runtime_projection(studies=1300):
    need(type(studies) is int and studies > 0, 'Positive study count required')
    reader = {'dicom_decode_estimate': 8.5 * studies,
              'three_fold_forward_estimate': 3 * 6 * 16 * 0.02 * studies,
              'fixed_allowance': FIXED_S}
    rad = {'own_dicom_decode_screen': 1.3 * 6.52503319258 * studies,
           'five_model_plain_screen': 1.3 * 5 * 0.25 * studies,
           'plane_flip_extra_screen': 1.3 * 5 * 0.25 * studies,
           'fixed_allowance': FIXED_S}
    return {'studies': studies, 'oai_0949_seconds': None,
            'oai_reason': 'No measured 1300-study runtime; fill after fidelity/timing run',
            'reader_b1': dict(reader, total_partner_estimate=sum(reader.values()),
                              basis='infer.py K16 x six slots x three ConvNeXt folds, 384px DICOM decode and 256px forward; 8.5 s/study decode + 0.02 s/window/fold are unmeasured assumptions'),
            'rad_b2': dict(rad, total_partner_screen=sum(rad.values()),
                           basis='EXP119/v6 inflated screen constants; separate own DICOM decode; second forward pass for EXP124 plane-aware TTA'),
            'registered_cutoff_seconds': CUTOFF_S,
            'decision': 'Only stage partner if elapsed 0.949 time + partner projection <= cutoff - 120 s; otherwise retain pure 0.949'}


def rad_payload():
    """Copy the v7 RadR50x5 release arm; no anchor arm is mounted or scored."""
    v7_path = HERE / 'exp119_release_v7.py'
    need(sha(v7_path) == 'b520dbd35b0f026c00c5891406279dec2a2d0364c11dc1b4be8c8caafd72a6a9',
         'Reviewed EXP119 v7 source changed')
    v7 = _load('exp119_for_exp128_builder', v7_path)
    base = ROOT / '.local/exp119-release-v7/coat5rad5/release-manifest.json'
    release = copy.deepcopy(v7.checked_manifest(base, sha(base)))
    release['arms'] = [arm for arm in release['arms'] if arm['name'] == 'rad5']
    need(len(release['arms']) == 1 and len(release['arms'][0]['checkpoints']) == 5,
         'V7 RadR50 five-fold arm unavailable')
    release['arms'][0]['weight'] = 1.0
    release['arm_assets'] = {'rad5': release['arm_assets']['rad5']}
    dataset = release['arm_assets']['rad5']
    release['asset_sets'] = [a for a in release['asset_sets'] if a['dataset'] == dataset]
    release['asset_packages'] = {k: v for k, v in release['asset_packages'].items() if k == dataset}
    release['arm_specs'] = [s for s in release['arm_specs'] if s.startswith('rad5=')]
    release['rank_weights'] = {'rad5': 1.0}
    release['release'] = 'exp128-rad5-tta'
    release['combination'] = 'single arm: raw mean_prob probabilities'
    v7.validate_release(release)
    raw = json.dumps(release, indent=2) + '\n'
    sources = {}
    for name in SOURCE_FILES_B:
        path = ROOT / 'training/src/raptor_shared_input.py' if name == 'raptor_shared_input.py' else HERE / name
        data = path.read_text(encoding='utf-8')
        sources[name] = {'sha256': hashlib.sha256(data.encode()).hexdigest(), 'z': packed(data)}
    return {'files': sources, 'manifest_z': packed(raw),
            'manifest_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'arm': {'name': 'rad5', 'checkpoints': [c['name'] for c in release['arms'][0]['checkpoints']],
                    'dataset': dataset, 'k_eval': release['arms'][0]['k_eval']}}


def notebook(variant, public, code, payload):
    source = Path(__file__).read_text(encoding='utf-8')
    write_module = '%%writefile /kaggle/working/exp128_oai_light.py\n' + source
    prefix = ("import time as _exp128_time\n"
              "_EXP128_START = _exp128_time.monotonic()\n")
    tail = ("import importlib.util as _exp128_importlib\n"
            "_spec = _exp128_importlib.spec_from_file_location('exp128_runtime', '/kaggle/working/exp128_oai_light.py')\n"
            "_exp128 = _exp128_importlib.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_exp128)\n")
    if variant == 'fidelity':
        tail += ("_EXP128_FIDELITY = _exp128.fidelity_receipt()\n"
                 "print('EXP128 fidelity:', _EXP128_FIDELITY['submission_sha256'])\n")
    else:
        literal = repr(payload) if variant == 'blend_b2' else 'None'
        inner = (tail +
                 "try:\n"
                 "    _EXP128_FIDELITY = _exp128.fidelity_receipt()\n"
                 "except BaseException as _diagnostic_error:\n"
                 "    print('EXP128 fidelity diagnostic failed:', type(_diagnostic_error).__name__, str(_diagnostic_error), flush=True)\n"
                 "_EXP128_RECEIPT = _exp128.run_after_oai(" + repr(variant) +
                 ", payload=" + literal + ", start=_EXP128_START)\n"
                 "print('EXP128 path:', _EXP128_RECEIPT['path'], 'reason:', _EXP128_RECEIPT['reason'])\n")
        tail = ("try:\n" + ''.join('    ' + line for line in inner.splitlines(keepends=True)) +
                "except BaseException as _exp128_error:\n"
                "    print('EXP128 pure-0.949 fallback:', type(_exp128_error).__name__, str(_exp128_error), flush=True)\n")
    original = [copy.deepcopy(c) for c in public['cells'][:3]]
    cells = [cell(prefix), cell(credit(), 'markdown'), *original, *copy.deepcopy(code),
             cell(write_module), cell(tail)]
    result = {'nbformat': 4, 'nbformat_minor': 5,
              'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                           'language_info': {'name': 'python'}}, 'cells': cells}
    need(result['cells'][5:7] == code, 'Public 0.949 code cells changed')
    return result


def build(out=OUT):
    public, code = load_source()
    out = Path(out)
    need(not out.exists(), 'Output already exists; choose a fresh directory')
    payload = rad_payload()
    out.mkdir(parents=True)
    report = {'schema': 'exp128-oai-light-build-v1', 'status': 'UNPUSHED',
              'source_notebook_sha256': SOURCE_SHA, 'weights_sha256': WEIGHT_SHA,
              'runtime_projection_1300': runtime_projection(), 'kernels': {}}
    for variant in KERNELS:
        dest = out / variant
        dest.mkdir()
        nb = notebook(variant, public, code, payload if variant == 'blend_b2' else None)
        meta = metadata(variant)
        write_json(dest / 'release.ipynb', nb)
        write_json(dest / 'kernel-metadata.json', meta)
        screen = {'schema': 'exp128-submission-screen-v1', 'variant': variant,
                  'basis': 'PUBLIC_ONLY', 'reason': 'OAI-derived checkpoint trained on all data; no local OOF',
                  'submission_state': 'UNPUSHED', 'private_kernel': True,
                  'submit_candidate': variant != 'fidelity',
                  'expected_studies_public3': 3, 'targets': TARGETS,
                  'fidelity': {'public_source_sha256': SOURCE_SHA,
                               'compare_command': 'python tools/exp128_compare_public3.py --ours <our-submission.csv> --public <public-notebook-output.csv>',
                               'require_same_columns_ids_and_exact_values': True},
                  'blend': None if variant == 'fidelity' else
                           {'weights': list(WEIGHTS[variant]), 'rank': 'per-target average-tie percentile',
                            'partner': 'ConvNeXt three-fold own preprocessing' if variant == 'blend_b1' else
                                       'v7 RadR50x5 own DICOM decode + EXP124 plane-aware flip TTA',
                            'fallback': 'pure 0.949 if failure or 30600-second cutoff'},
                  'public_score_adoption': 'Fidelity uses no submission slot. B chooses one blend after fidelity and runtime consultation; public score only, no weight retuning'}
        write_json(dest / 'submission_screen_spec.json', screen)
        record = {'kernel': meta['id'], 'variant': variant, 'not_pushed': True,
                  'public_code_cells_unchanged': nb['cells'][5:7] == code,
                  'notebook_sha256': sha(dest / 'release.ipynb'),
                  'metadata_sha256': sha(dest / 'kernel-metadata.json'),
                  'push_command_not_run': '.venv\\Scripts\\python.exe tools\\kaggle_push.py ' +
                                          str(dest) + ' --accelerator NvidiaTeslaT4'}
        report['kernels'][variant] = record
        write_json(dest / 'build-receipt.json', record)
    write_json(out / 'build-report.json', report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--out', default=str(OUT))
    c = sub.add_parser('compare'); c.add_argument('--ours', required=True); c.add_argument('--public', required=True)
    a = p.parse_args(argv)
    if a.command == 'build':
        result = build(a.out)
        print(json.dumps({'status': result['status'], 'out': str(a.out),
                          'kernels': list(result['kernels'])}, indent=2))
        return 0
    result = compare_csv(a.ours, a.public)
    print(json.dumps(result, indent=2))
    return 0 if result['exact'] and result['studies'] == 3 else 1


if __name__ == '__main__':
    sys.exit(main())
