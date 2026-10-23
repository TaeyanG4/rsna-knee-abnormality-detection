"""v1.0.0: bounded CPU real-DICOM flip path smoke; no full-study qualification.

One decoded three-slice window from each visible study, resized to 384 and
centre-cropped to 352. Uses all four actual checkpoint models and the frozen
worker's mirror/score/state functions. This local window preparation is NOT the
production anatomical renderer; full Linux visible qualification remains required.
"""
from __future__ import annotations
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_flipall_v1115 as b


def node(source, name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import pydicom
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    b.validate()
    files = b.unpack(b.read(b.OUT / 'kernel/release.ipynb'))[0]
    source_key = next(k for k in files if k.endswith('/compact_infer.py'))
    trainer_key = next(k for k in files if k.endswith('/compact_train.py'))
    source = files[source_key].decode()
    trainer_path = b.SCRATCH / 'candidate-runtime' / trainer_key
    spec = importlib.util.spec_from_file_location('flip_smoke_frozen_trainer', trainer_path)
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    bundle = ROOT / 'storage/external/heliosli-r7cnxfsss-v1'
    manifest = b.read(bundle / 'manifest.json')
    b.pin(bundle / 'manifest.json', 'c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6')
    state = b.read(bundle / 'state_spec.json')
    for name, sha in manifest['sha256'].items():
        b.pin(bundle / name, sha)
    data = ROOT / '.local/exp189-cpu-v120/input/visible_data/test_series'
    ids = sorted(p.name for p in data.iterdir() if p.is_dir())
    b.need(len(ids) == 3, 'expected all real3 studies')
    images, pins = [], {}
    for uid in ids:
        paths = sorted((data / uid).rglob('*.dcm'))
        middle = len(paths) // 2
        paths = paths[middle-1:middle+2]
        pixels = []
        for path in paths:
            pixels.append(pydicom.dcmread(path).pixel_array.astype('float32'))
            pins[str(path.relative_to(ROOT))] = b.digest(path.read_bytes())
        arr = np.stack(pixels)
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip((arr-lo)/max(hi-lo, 1)*255, 0, 255)
        img = F.interpolate(torch.from_numpy(arr)[None], size=(384, 384), mode='bilinear', align_corners=False)[0]
        images.append(img[:, 16:368, 16:368].to(torch.uint8))
    # Each study contributes a window; exercise all plane/view transformations.
    img = torch.stack(images)[None]
    meta = torch.zeros((1, 3, 9))
    meta[0, :, 7] = torch.tensor([0, 1, 2])
    meta[0, :, 1] = torch.tensor([0, 2, 3])
    meta[0, :, 2] = .75
    meta[0, :, 6] = 1
    mask = torch.ones((1, 3), dtype=torch.bool)
    ns = dict(torch=torch, MIRROR_PAIRS=(("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")))
    for name in ('mirror_perm', 'mirror_windows', 'state_layout', 'state_readout', 'score_v4c'):
        exec(compile(ast.Module(body=[node(source, name)], type_ignores=[]), '<frozen-flip>', 'exec'), ns)
    bins = [(k, label) for k, labels in state['binaries'].items() for label in labels]
    perm, bp = ns['mirror_perm'](ct.LABELS, state, bins)
    layout = ns['state_layout'](state, ct.LABELS)
    mcl = ct.LABELS.index('MCL')
    cols = [i for col in layout['columns'] if col['class'] == 'MCL'
            for i in (range(col['start'], col['stop']) if 'categories' in col else [col['column']])]
    ns.update(FLIP=dict(perm=perm, bin_perm=bp, mcl=mcl), MCL_STATE_COLS=cols, EXTRA={})
    flipped, fm = ns['mirror_windows'](img, meta)
    back, bm = ns['mirror_windows'](flipped, fm)
    assert torch.equal(img, back) and torch.equal(meta, bm)
    started, folds, calls = time.perf_counter(), [], []
    def one_pass(model, image, metadata, valid):
        result = model(image, metadata, valid, chunk=32, return_aux=True)
        calls.append(result)
        return result
    ns['one_pass'] = one_pass
    with torch.inference_mode():
        for index, name in enumerate(manifest['models']):
            sd = torch.load(bundle / name, map_location='cpu', weights_only=True)
            model = ct.Net(manifest['backbone'], None, crop=352, state_spec=state, in_chans=3,
                           spatial='sp_q' in sd, extent='ext_u' in sd, syn_decomp='syn_beta' in sd)
            model.load_state_dict(sd, strict=True)
            model.eval().requires_grad_(False)
            calls.clear()
            actual = ns['score_v4c']([model], img, meta, mask)[0]
            (z, sl, bl), (zm, slm, blm) = calls
            expected = (z[0] + zm[0, perm]) / 2
            expected[mcl] = z[0, mcl]
            assert torch.equal(actual, expected) and torch.isfinite(actual).all()
            plain = ns['state_readout'](model, sl, bl)[0]
            mirrored = ns['state_readout'](model, slm[:, perm], blm[:, bp])[0]
            expected_state = (plain + mirrored) / 2
            expected_state[cols] = plain[cols]
            assert torch.equal(ns['EXTRA']['group1']['state'][0], expected_state)
            folds.append(dict(fold=index, plain_mcl_exact=True, state_mcl_exact=True,
                              non_mcl_changed=int((actual != z[0]).sum())))
            print(json.dumps(dict(fold=index, status='PASS')), flush=True)
            del model, sd
    report = dict(status='PASS', device='cpu', folds=folds, studies=3, decoded_dicoms=9,
                  input_sha256=pins, worker_sha256=b.digest(files[source_key]), trainer_sha256=b.digest(files[trainer_key]),
                  seconds=time.perf_counter()-started, exposure='EXPOSED', clean_oof=False,
                  scope='Reduced real-pixel window smoke; real checkpoints and frozen flip/state math. Eager CPU; no full renderer, compiled GPU, full-study prediction or timing qualification.')
    # Fixed scratch receipt may be overwritten for a rerun, never release bytes.
    from tools.hb145_output_io import atomic_json
    atomic_json(b.SCRATCH / 'flip-cpu-smoke.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
