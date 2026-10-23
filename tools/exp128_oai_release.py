"""EXP128 local-only builder and guarded Kaggle runtime.

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
STAGE1 = ROOT / '.local/hb15-combined/knee-exp105-anchor-own-v1/knee-exp105-anchor-own-v1.ipynb'
STAGE1_SHA = 'ba8acd5ac2a8f5619ea09ae90dc548c74999ce01ce7e3bf42d8157367af12cd3'
WEIGHT_SHA = '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
WEIGHT_NAME = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
DATASET = 'nartaa/rsna-knee-publication-swa-weights-20261007'
COMPETITION = 'rsna-knee-abnormality-detection'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
KERNELS = {'fidelity': 'knee-exp128-oai-fidelity', 'timing': 'knee-exp128-oai-timing',
           'a': 'knee-exp128-oai-anchor-a', 'b': 'knee-exp128-oai-rad-b'}
# Fixed H-B-58 weights. There is no public-score tuning in this builder.
WEIGHTS = {'a': (0.6, 0.4), 'b': (0.9, 0.1)}
CUTOFF_S = 30_600
PUBLIC_ANCHOR_S = 26_404
RAD_DECODE_S = 1.3 * 6.52503319258 * 1300
RAD_TTA_S = 2 * 1.3 * 5 * 0.25 * 1300
FIXED_S = 600
SOURCE_FILES_B = ('exp128_oai_release.py', 'exp126_anchor_blend_v8.py',
                  'exp119_release_v7.py', 'exp111_release_v6.py',
                  'exp105_combined_release.py', 'raptor_shared_input.py')


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
    return ("# EXP128 — public OAI-trained checkpoint\n\n"
            "Checkpoint: `" + WEIGHT_NAME + "` (SHA-256 `" + WEIGHT_SHA + "`). "
            "Published by **nartaa**, using the **Raptor CoAtNet/MIL recipe by dreaddevelopment**. "
            "Its training included controlled-access Osteoarthritis Initiative (OAI) data. "
            "No OAI source images or reports are loaded by this notebook. The checkpoint dataset "
            "`" + DATASET + "` declares Apache 2.0; retain its original notices and the "
            "three original public notebook markdown cells below. This is our private, unpushed "
            "run, not a claim of training the checkpoint.\n\n"
            "Original notebook: `nartaa/rsna-knee-0949-anatomical-mirror`, SHA-256 `" + SOURCE_SHA + "`. "
            "Original metadata names docker image `gcr.io/kaggle-private-byod/python@sha256:2757e0c7…`; "
            "this build omits it because normal-user access is unverified. The original qualification "
            "reported torch 2.11.0+cu128 and timm 1.0.29; the runtime receipt records actual versions.\n")


def metadata(variant, anchor=None):
    sources = [DATASET]
    if variant == 'a':
        need(anchor is not None, 'Anchor metadata required')
        sources += anchor['dataset_sources']
    if variant == 'b':
        sources += ['taeyangg4/knee-exp089-recipe-transfer-v1',
                    'taeyangg4/knee-release-assets-v8',
                    'mattiaangeli/opencv-python-headless-4120088-x86']
    return {'id': 'taeyangg4/' + KERNELS[variant], 'title': KERNELS[variant],
            'code_file': 'release.ipynb', 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': False, 'dataset_sources': list(dict.fromkeys(sources)),
            'competition_sources': [COMPETITION], 'kernel_sources': [], 'model_sources': [],
            'machine_shape': 'NvidiaTeslaT4'}


def timing_source(public_code):
    """One private TRAIN-only benchmark; never reads train.csv or labels."""
    marker = '\ndef main():\n'
    need(public_code.count(marker) == 1, 'Public main marker differs')
    before, main = public_code.split(marker)
    need('test = pd.read_csv(ROOT + "/test.csv")' in main, 'Public test roster line differs')
    replacements = [
        ('tsdir = ROOT + "/test_series"', 'tsdir = ROOT + "/train_series"'),
        ('tsdir = ROOT + "/test_images"', 'tsdir = ROOT + "/train_images"'),
        ('test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)',
         'test = pd.read_csv(ROOT + "/train_series.csv", usecols=["StudyInstanceUID"], dtype=str).drop_duplicates("StudyInstanceUID").sort_values("StudyInstanceUID").head(100); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)'),
        ('tser = pd.read_csv(ROOT + "/test_series.csv")', 'tser = pd.read_csv(ROOT + "/train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})'),
        ('SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}',
         'tser = tser[tser["StudyInstanceUID"].isin(test_ids)]\n    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}'),
        ('out = "/kaggle/working/submission.csv"', 'out = "/kaggle/working/exp128_timing_predictions.csv"'),
        ('t_pre = time.time()', 't_pre = time.time()'),
    ]
    for old, new in replacements:
        need(old in main, 'Timing patch marker differs: ' + old[:50])
        main = main.replace(old, new, 1)
    # The public source's study builder, model, K94, mirror, GPU and worker policy stay identical.
    main = main.replace('    print(f"[pre] done: {N} studies in {time.time()-t_pre:.0f}s (procs {nproc} x fastread {FASTREAD_VARIANT} {FR_HDR}/{FR_PX} threads)", flush=True)',
                        '    _exp128_pre_done = time.time()\n    print(f"[pre] done: {N} studies in {_exp128_pre_done-t_pre:.0f}s (procs {nproc} x fastread {FASTREAD_VARIANT} {FR_HDR}/{FR_PX} threads)", flush=True)', 1)
    needle = '    print(f"DONE {time.time()-t0:.0f}s", flush=True)'
    need(needle in main and '_exp128_pre_done' in main, 'Timing phase markers differ')
    main = main.replace(needle, needle + '''
    _exp128_end = time.time()
    from pathlib import Path as _Exp128Path
    import json as _exp128_json
    import timm as _exp128_timm
    _exp128_json.dump({"schema":"exp128-timing-v1", "roster":"first 100 lexicographic unique StudyInstanceUID in train_series.csv", "labels_read":False,
        "studies":N, "gpus":ngpu, "torch":str(torch.__version__), "timm":str(_exp128_timm.__version__),
        "phase_seconds":{"setup_models":t_pre-t0,"preprocess_until_pool_done":_exp128_pre_done-t_pre,
          "queue_drain_and_publish":_exp128_end-_exp128_pre_done,"total":_exp128_end-t0},
        "seconds_per_study":(_exp128_end-t0)/N,"projection_1300_seconds":(_exp128_end-t0)/N*1300,
        "prediction_sha256":hashlib.sha256(_Exp128Path(out).read_bytes()).hexdigest()},
        open('/kaggle/working/exp128_timing.json','w'),indent=2)
''', 1)
    return before + marker + main


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
        pct = ((at + end - 1) / 2) / max(1, len(order) - 1)
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


def publish(ids, first, second, weights, expected_sha, backup):
    """Atomic final write; restore the verified pure 0.949 output on any failure."""
    final = Path('/kaggle/working/submission.csv')
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


def _anchor_stage(anchor_cells, start, ids, pure, receipt):
    """Execute unchanged EXP105 setup/anchor/hook cells in the notebook namespace."""
    # The full-stage anchor's 26,404 s is a screen scenario, not a hidden bound.
    elapsed = time.monotonic() - start
    need(elapsed + PUBLIC_ANCHOR_S + 300 <= CUTOFF_S, 'A anchor projection exceeds 30,600 s cutoff')
    ip = get_ipython()  # noqa: F821 - Kaggle notebook runtime only
    need(ip is not None, 'Anchor requires an IPython notebook runtime')
    need(len(anchor_cells) == 33, 'Scored stage-1 cell count differs')
    old = signal.signal(signal.SIGALRM, _timeout)
    try:
        for index, source in enumerate(anchor_cells):
            if source is None:
                continue
            remaining = CUTOFF_S - 300 - (time.monotonic() - start)
            need(remaining > 0, 'Anchor cutoff before cell ' + str(index))
            signal.setitimer(signal.ITIMER_REAL, remaining)
            try:
                result = ip.run_cell(source, store_history=False)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError('Anchor cell %d failed: %s' % (index, result.error_before_exec or result.error_in_exec))
        ns = ip.user_ns
        setup = ns.get('_HB15_SETUP') or {}
        need(setup.get('ok') is True, 'EXP105 setup failed')
        hook = ns.get('_HB15') or {}
        need(callable(hook.get('finalize')), 'EXP105 hook missing')
        summary = hook['finalize'](120)
        need(summary.get('thread_finished') is True and not summary.get('hook_errors') and
             not summary.get('anchor_raised') and not summary.get('dropped_ids'), 'EXP105 hook failed')
        src = Path(setup['src'])
        e105 = _load('exp105_for_exp128', src / 'exp105_combined_release.py',
                     '4c0d227c3d0d66d0e40821994ff125f84e78b6f964c240c5e052d06df76c3fd2')
        gate = e105.coat_family_record(ns, Path('/kaggle/working'))
        need(gate.get('ok') is True and gate.get('members') ==
             ['resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3'] and
             gate.get('audit_status') == 'COMPLETE' and gate.get('audit_members') == 4 and
             gate.get('family_reduction') == 'rank_of_member_probability_mean' and
             not gate.get('failure_events'), 'Four-child anchor gate failed')
        anchor_ids, anchor = read_submission('/kaggle/working/submission.csv')
        need(anchor_ids == ids and len(anchor) == len(pure), 'Anchor cohort differs')
        receipt['anchor_gate'] = gate
        return anchor
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _rad_stage(payload, start, ids, receipt):
    """EXP126 Rad worker with no shared store; its verified DICOM fallback is charged."""
    import gc
    import torch
    elapsed = time.monotonic() - start
    projected = RAD_DECODE_S + RAD_TTA_S + FIXED_S
    receipt['rad_projection_seconds'] = projected
    need(elapsed + projected <= CUTOFF_S, 'B RadR50+TTA projection exceeds 30,600 s cutoff')
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
    e105.install_env(_load('exp111_for_exp128', src / 'exp111_release_v6.py'), old, wheel, root / 'env')
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
        result = subprocess.run(cmd, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
    own, worker = e126.checked_own(release, sha(manifest), root / 'rad-worker', ids,
                                    result.returncode, plane_flip_tta=True)
    need(set(own) == set(range(len(ids))) and not worker.get('unscored'), 'Rad worker coverage incomplete')
    receipt['rad_worker'] = worker
    return [own[i] for i in range(len(ids))]


def run_after_oai(variant, payload=None, start=None):
    """Never propagates a blend-stage error; restores pure public-model CSV."""
    need(variant in ('a', 'b'), 'Unknown blend variant')
    start = time.monotonic() if start is None else start
    final = Path('/kaggle/working/submission.csv')
    backup = Path('/kaggle/working/exp128_oai_submission.csv')
    receipt = {'schema': 'exp128-blend-v1', 'variant': variant,
               'weights': list(WEIGHTS[variant]), 'cutoff_seconds': CUTOFF_S,
               'path': 'pure_oai', 'reason': None}
    try:
        ids, pure = read_submission(final)
        pure_sha = sha(final)
        shutil.copyfile(final, backup)
        need(sha(backup) == pure_sha, 'Pure OAI backup differs')
        receipt['pure_oai_sha256'] = pure_sha
        if variant == 'a':
            second = _anchor_stage(payload['anchor_cells'], start, ids, pure, receipt)
        else:
            second = _rad_stage(payload, start, ids, receipt)
        need(time.monotonic() - start <= CUTOFF_S - 120, 'No publication allowance remains')
        receipt['final_sha256'] = publish(ids, pure, second, WEIGHTS[variant], pure_sha, backup)
        receipt['path'] = 'blended'
    except BaseException as exc:
        receipt['reason'] = type(exc).__name__ + ': ' + str(exc)
        receipt['traceback_tail'] = traceback.format_exc()[-2000:]
        if backup.is_file():
            shutil.copyfile(backup, final)
            receipt['final_sha256'] = sha(final)
    receipt['elapsed_seconds'] = time.monotonic() - start
    write_json('/kaggle/working/exp128_blend_receipt.json', receipt)
    return receipt
