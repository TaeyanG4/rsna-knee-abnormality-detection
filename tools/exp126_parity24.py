"""Real-checkpoint EXP126 CPU parity on the saved 24-study v7 cohort.

Reads series metadata and DICOMs only. Never reads labels or initializes CUDA.
The optional TTA reference uses the same release-v6 checkpoints and two frozen
predict passes, with the EXP124 plane mirror transform on pass two.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / '.local/exp089-recipe-transfer/t4-v3/dataset'
ASSETS = ROOT / 'kaggle/packages/KNEE_RELEASE_ASSETS_V8'
PARITY = ROOT / '.local/exp119-release-v7/cpu-parity-v1.json'
MANIFEST = ROOT / '.local/exp126-anchor-blend/rad5-v8-r9/exp126-own-manifest.json'
UID = 'StudyInstanceUID'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def series_rows(ids):
    want = set(ids)
    with (ROOT / 'storage/competition/train_series.csv').open(newline='', encoding='utf-8-sig') as f:
        return [{k: row.get(k) for k in (UID, 'SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive')}
                for row in csv.DictReader(f) if row[UID] in want]


def archive_paths():
    con = sqlite3.connect((ROOT / '.local/kneeops/app.sqlite').as_uri() + '?mode=ro', uri=True)
    try:
        row = con.execute("SELECT value FROM meta WHERE key='data_library'").fetchone()
    finally:
        con.close()
    archive = json.loads(row[0])['archives'][0]
    return Path(archive['path'].replace('workspace://', str(ROOT) + '/')), Path(archive['index'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--own', choices=('rad5', 'coat5rad5'), default='rad5')
    ap.add_argument('--plane-flip-tta', action='store_true')
    ap.add_argument('--studies', type=int, default=24)
    ap.add_argument('--manifest', type=Path, default=MANIFEST)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.studies <= 24 or args.out.exists():
        raise ValueError('Use 1..24 studies and a fresh output path')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['KNEE_WORKSPACE_ROOT'] = str(FROZEN)
    sys.path[:0] = [str(FROZEN / 'training/src'), str(ASSETS / 'code')]

    import numpy as np
    import torch
    def refuse(*_args, **_kwargs):
        raise RuntimeError('CUDA initialization forbidden in CPU parity')
    torch.cuda._lazy_init = refuse
    torch.cuda.is_available = lambda: False
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    import cv2
    cv2.setNumThreads(1)
    import raptor_common as common
    import raptor_runtime as runtime
    import raptor_dicom as dicom
    exp = load('exp126_parity_release', ROOT / 'tools/exp126_anchor_blend_v8.py')
    _, v7 = exp.modules()
    v6 = v7.v6
    runtime.configure_runtime('none', 'cpu', 'strict')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized')

    release = v7.checked_manifest(args.manifest, exp.sha(args.manifest))
    names = ['rad5'] if args.own == 'rad5' else ['coat5', 'rad5']
    arms = [a for a in release['arms'] if a['name'] in names]
    if [a['name'] for a in arms] != names:
        raise ValueError('Selected arms differ from own manifest')
    ids = json.loads(PARITY.read_text(encoding='utf-8'))['train_studies']['ids'][:args.studies]
    rows = series_rows(ids)
    zip_path, index_path = archive_paths()
    def corpus(source):
        return dicom.DicomCorpus.from_records(ids, rows, source, v7.DECODE_CONTRACT, 'train')

    result = dict(schema='exp126-cpu-parity-v1', device='cpu', amp='none', studies=len(ids),
                  own=args.own, plane_flip_tta=args.plane_flip_tta,
                  release_manifest_sha256=exp.sha(args.manifest), arms={})
    with dicom.IndexedZipDicomSource(zip_path, index_path) as source:
        shared = v7.serve_shared(arms, {a['name']: ASSETS for a in arms}, corpus(source), ids,
                                 'none', 'cpu', {}, log=lambda *_: None,
                                 runtime_module=exp.TTARuntime(runtime, args.plane_flip_tta))
        if shared['scored_ids'] != ids or shared['unscored']:
            raise ValueError('Shared path did not score all studies')
        for arm in arms:
            name = arm['name']
            actual = shared['probs'][name]
            separate, timing, checks, _ = v6.serve_arm(arm, ASSETS, corpus(source), ids,
                                                        'none', 'cpu', {}, log=lambda *_: None)
            reference = separate
            if args.plane_flip_tta and name == 'rad5':
                from raptor_common import select_centers
                models = [common.load_checkpoint(ASSETS / ck['path'])[0].eval().to('cpu')
                          for ck in arm['checkpoints']]
                model = v6.mean_prob_module(models)
                view = corpus(source)
                with v6.serving_input(runtime, 'center3'):
                    center3 = runtime.prepare_input
                    def mirror(c, uid, k, res, norm, window_policy='legacy'):
                        x = center3(c, uid, k, res, norm, window_policy)
                        centers = select_centers(c.mask(uid), c.depth, k, c.spec, policy=window_policy)
                        flags = exp.plane_flags(centers, dict(c.spec, depth=c.depth))
                        return np.ascontiguousarray(np.stack([x[i, ..., ::-1] if flag else x[i]
                                                               for i, flag in enumerate(flags)]))
                    try:
                        runtime.prepare_input = mirror
                        flipped, _ = runtime.predict(model, view, ids, arm['k_eval'], arm['res'],
                                                      'none', 'cpu', norm=arm['norm'], batch=1,
                                                      log=lambda *_: None, window_policy=arm['window_policy'])
                    finally:
                        runtime.prepare_input = center3
                reference = (separate + flipped) / np.float32(2)
            delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
            result['arms'][name] = dict(max_abs_diff=float(delta.max()), mean_abs_diff=float(delta.mean()),
                                       exact=bool(np.array_equal(actual, reference)),
                                       checkpoints_match=([{k: v for k, v in c.items() if k != 'input_policy'}
                                                           for c in shared['checks'][name]]
                                                          == [{k: v for k, v in c.items() if k != 'input_policy'}
                                                              for c in checks]),
                                       input_hashes_match=(shared['timing'][name]['ordered_inputs_sha256']
                                                           == timing['ordered_inputs_sha256']) if not args.plane_flip_tta else None)
    result['cuda_initialized'] = torch.cuda.is_initialized()
    result['pass'] = (not result['cuda_initialized'] and all(x['exact'] and x['checkpoints_match']
                      for x in result['arms'].values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}), flush=True)
    print({k: v['max_abs_diff'] for k, v in result['arms'].items()}, flush=True)
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
