"""H-B-113 Gold58 paired diagnostic v1.0.0; never launches a GPU job.

Reuse P956 R functions verbatim. CPU/fp32 is an engineering smoke only.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp169_release_p956 as p956
from tools import exp152_own0949_release as helper
from tools import multi_lens_readout as lens
from tools.hb113_gold58_archive import ArchiveReader

VERSION = '1.0.0'
STAGE = ROOT / 'artifacts/experiments/H_B_113_GOLD58/v1.0.0'
SCRATCH = ROOT / '.local/scratch/sol-2/hb113-gold58'
GOLD = ROOT / 'storage/competition/train.csv'
SERIES = ROOT / 'storage/competition/train_series.csv'
SERVING = p956.VISIBLE / 'exp155-assets/public-serving.py'
PARENT = ROOT / '.local/rplus-v1/public-R-download.bin'
MEMBER = ROOT / 'artifacts/experiments/H_B_109_RPLUS_PUBLIC/v1.0.0/member/outputs/model_best.pt'
LABELS = tuple(p956.LABELS)
UID = p956.UID
CAVEAT = ('Public R was trained on all data including Gold58 using report labels. '
          'R+ inherits that ancestry, then continued on 4,347 studies excluding Gold58. '
          'Forgetting biases this comparison against R+. This is an exposed diagnostic, '
          'not honest OOF, independent generalization evidence, or a release gate.')
need, sha = p956.need, p956.sha


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(p956.json_bytes(obj))


def scratch(path):
    path = Path(path).resolve()
    need(path.is_relative_to(SCRATCH.resolve()) and path != SCRATCH.resolve(), 'output must be below fixed scratch root')
    return path


def gold():
    frame = pd.read_csv(GOLD, usecols=[UID, *LABELS], dtype={UID: str})
    frame = frame.loc[frame['ACL'].notna()].sort_values(UID).reset_index(drop=True)
    need(len(frame) == 58 and frame[UID].nunique() == 58, 'official Gold58 roster invalid')
    y = frame[list(LABELS)].to_numpy(float)
    need(np.isfinite(y).all() and np.isin(y, [0, 1]).all(), 'official Gold labels must be complete binary values')
    return frame[UID].tolist(), y


def source_files():
    # All project Python modules actually imported by this runner, including the
    # paired bootstrap implementation and its transitive helper imports.
    result = {str(Path(__file__).resolve().relative_to(ROOT)): Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(ROOT / 'tools') and path.suffix == '.py':
                result[str(path.relative_to(ROOT))] = path
    result[str(SERVING.relative_to(ROOT))] = SERVING
    return dict(sorted(result.items()))


def register():
    ids, _ = gold()
    p956.check_pin(PARENT, p956.R_SHA)
    for name, path in [('public-serving.py', SERVING), ('exp152_own0949_release.py', Path(helper.__file__))]:
        p956.check_pin(path, p956.PINS[name])
    ck = helper.safe_load(MEMBER)
    need(ck.get('public_parent_sha256') == p956.R_SHA and
         ck.get('ancestry_scope') == 'PRODUCTION_PUBLIC_R_ANCESTRY' and
         ck.get('swa_epochs_1based') == [1, 2, 3] and ck.get('checkpoint_kind') == 'fixed-swa-bn',
         'unexpected member ancestry or checkpoint kind')
    need(ck.get('lab') == list(LABELS) or ck.get('lab') == LABELS, 'member label order mismatch')
    binding = dict(version=VERSION, caveat=CAVEAT, scope='PRODUCTION_PUBLIC_R_ANCESTRY_GOLD58_DIAGNOSTIC',
                   gold_count=58, ids=ids, targets=LABELS,
                   serving='P956 R native320 centre crop / K94 / plain+anatomical mirror probability mean',
                   bootstrap=dict(draws=lens.DRAWS, seed=lens.SEED, unit='Gold study singleton', weak_indices=lens.WEAK),
                   files={str(p.relative_to(ROOT)): sha(p) for p in [GOLD, SERIES, PARENT, MEMBER]},
                   sources={name: sha(path) for name, path in source_files().items()})
    p956.write_once(STAGE / 'registration.json', p956.json_bytes(binding))
    return dict(status='REGISTERED', studies=58, binding_sha256=sha(STAGE / 'registration.json'))


def validate():
    binding = json.loads((STAGE / 'registration.json').read_text())
    need(binding['version'] == VERSION and binding['caveat'] == CAVEAT, 'binding policy drift')
    need(binding['targets'] == list(LABELS), 'target order drift')
    for name, digest in {**binding['files'], **binding['sources']}.items():
        p956.check_pin(ROOT / name, digest)
    ids, _ = gold()
    need(ids == binding['ids'], 'Gold roster drift')
    return binding


def runtime():
    # Execute P956's authenticated source in memory: no extracted fastread file
    # beside frozen sources, no alternative training decoder or corpus cache.
    module = types.ModuleType('hb113_p956_r')
    module.__file__ = str(SERVING)
    exec(compile(p956.r_runtime_source(SERVING.read_text(encoding='utf-8')), str(SERVING), 'exec'), module.__dict__)
    module._E152 = helper
    return module


def fastread(module):
    reader = types.ModuleType('hb113_p956_fastread')
    exec(compile(base64.b64decode(module._FASTREAD_B64), '<P956 embedded fastread>', 'exec'), reader.__dict__)
    if not hasattr(os, 'pread'):
        # Windows equivalent of positional binary reads. One lock protects each
        # seek+read pair; decoded bytes and P956 math remain unchanged.
        lock = threading.Lock()
        def pread(fd, n, offset):
            with lock:
                os.lseek(fd, offset, os.SEEK_SET)
                return os.read(fd, n)
        reader._pread = pread
        reader._O_FLAGS |= getattr(os, 'O_BINARY', 0)
    return reader


def cpu_probs(model, x, device):
    import torch
    need(str(device) == 'cpu', 'CPU smoke cannot use CUDA')
    with torch.no_grad():
        logits = model(x.unsqueeze(0)).float()
        need(bool(torch.isfinite(logits).all()), 'nonfinite CPU logits')
        return torch.sigmoid(logits)[0].numpy()


def check_predictions(arrays, n):
    for name in ('parent', 'member'):
        for suffix in ('', '_plain', '_mirror'):
            a = arrays[name + suffix]
            need(a.shape == (n, 12) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), 'invalid paired probabilities')
        need(np.array_equal(arrays[name], .5 * (arrays[name + '_plain'] + arrays[name + '_mirror'])), 'TTA mean mismatch')


def infer(device, output, dicom_root, smoke=False):
    import torch
    binding = validate()
    need((device == 'cpu') == smoke, 'CPU requires --smoke; full check requires CUDA')
    output = scratch(output)
    need(smoke or not (output / 'receipt.json').exists(), 'full output is immutable')
    dicom_root = Path(dicom_root)
    need(dicom_root.is_dir() or (dicom_root.is_file() and dicom_root.suffix == '.zip'), 'missing local DICOM root or ZIP')
    ids = binding['ids'][:2] if smoke else binding['ids']
    write(output / 'receipt.json', dict(status='RUNNING', caveat=CAVEAT))
    torch.set_num_threads(4)
    module = runtime()
    module._E152_FILES = {'parent': PARENT, 'member': MEMBER}
    module._E152_MANIFEST = {'checkpoints': {name: {'sha256': binding['files'][str(path.relative_to(ROOT))]}
                                           for name, path in module._E152_FILES.items()}}
    native_infer = module.infer_probs
    if smoke:
        module.infer_probs = cpu_probs
    metadata = pd.read_csv(SERIES, dtype={UID: str, 'SeriesInstanceUID': str})
    records = {u: f.to_dict('records') for u, f in metadata.loc[metadata[UID].isin(ids)].groupby(UID)}
    need(set(records) == set(ids), 'missing Gold series metadata')
    decoder = fastread(module)
    archive = ArchiveReader(decoder, dicom_root, ids) if dicom_root.is_file() else None
    input_root = archive.root if archive else str(dicom_root)
    reader = decoder.make_builder(module.FASTREAD_VARIANT, module.IMG, module.SLOTS,
                                            hdr_threads=module.FR_HDR, px_threads=module.FR_PX)
    models = {name: module.load_model(str(path), helper.ARCH, 384, device)[0]
              for name, path in module._E152_FILES.items()}
    arrays = {name + suffix: [] for name in models for suffix in ('', '_plain', '_mirror')}
    inputs, times = [], []
    started = time.perf_counter()
    for index, uid in enumerate(ids):
        tick = time.perf_counter()
        vol, mask = reader(uid, records, input_root)
        need(not archive or archive.read_errors == 0, 'archive DICOM read failed')
        need(vol.shape == (96, 384, 384) and vol.dtype == np.uint8 and mask.any(), 'empty or invalid P956 decode')
        plain = module.gpu_windows(vol, mask, 94, 320, device)
        mirror = module.gpu_windows(vol, mask, 94, 320, device, mirror=True)
        if smoke:
            need(torch.equal(plain, module.eval_windows(vol, mask, 94, 320)), 'plain CPU/LUT window mismatch')
            need(torch.equal(mirror, module.eval_windows(vol, mask, 94, 320, mirror=True)), 'mirror CPU/LUT window mismatch')
        inputs.append(helper.input_record(module.__dict__, vol, mask, plain.cpu().numpy(), mirror.cpu().numpy()))
        for name, model in models.items():
            views = []
            def capture(m, x, d):
                p = (cpu_probs if smoke else native_infer)(m, x, d)
                views.append(p)
                return p
            module.infer_probs = capture
            mean = module.infer_pair_probs(model, plain, mirror, device)
            arrays[name].append(mean)
            arrays[name + '_plain'].append(views[0])
            arrays[name + '_mirror'].append(views[1])
        times.append(time.perf_counter() - tick)
        print(json.dumps(dict(completed=index + 1, total=len(ids), seconds=times[-1])), flush=True)
    if archive:
        archive.close()
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    check_predictions(arrays, len(ids))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'predictions.npz', ids=np.asarray(ids), targets=np.asarray(LABELS), **arrays)
    receipt = dict(version=VERSION, status='CPU_SMOKE_ONLY' if smoke else 'PAIRED_GOLD58_INFERENCE_COMPLETE',
                   caveat=CAVEAT, binding_sha256=sha(STAGE / 'registration.json'),
                   prediction_sha256=sha(output / 'predictions.npz'), studies=len(ids), device=device,
                   precision='float32 engineering only' if smoke else 'P956 CUDA autocast fp16',
                   inputs=inputs, decode='P956 embedded study_lut; no persistent pixel cache',
                   input_container='CRC-checked ZIP bytes' if archive else 'DICOM directory',
                   seconds_per_study=times, seconds=time.perf_counter() - started,
                   torch=str(torch.__version__))
    write(output / 'receipt.json', receipt)
    return {k: receipt[k] for k in ('status', 'studies', 'seconds', 'prediction_sha256')}


def summary(values):
    valid = values[1:][np.isfinite(values[1:])]
    return dict(observed=float(values[0]) if np.isfinite(values[0]) else None,
                ci95=[float(x) for x in np.percentile(valid, [2.5, 97.5])] if len(valid) else None,
                valid_draws=len(valid), requested_draws=len(values) - 1)


def paired_stats(ids, parent, member, y):
    need(parent.shape == member.shape == y.shape == (len(ids), 12), 'paired shape mismatch')
    need(len(set(ids)) == len(ids), 'duplicate paired studies')
    w = lens.weights(ids, [{'component_id': u, 'members': [u]} for u in ids])
    a, b = [lens.auc_draws(p, y, np.isfinite(y), w) for p in (parent, member)]
    # np.mean deliberately invalidates an entire draw if ANY fixed target lacks
    # support; nanmean would silently change the macro target set per draw.
    def block(x, z):
        return dict(parent=summary(x), member=summary(z), delta_member_minus_parent=summary(z - x))
    return dict(targets={t: block(a[:, i], b[:, i]) for i, t in enumerate(LABELS)},
                macro12=block(a.mean(1), b.mean(1)),
                weak_four=block(a[:, lens.WEAK].mean(1), b[:, lens.WEAK].mean(1)))


def score(output, smoke=False):
    binding = validate()
    output = scratch(output)
    receipt = json.loads((output / 'receipt.json').read_text())
    expected = 'CPU_SMOKE_ONLY' if smoke else 'PAIRED_GOLD58_INFERENCE_COMPLETE'
    need(receipt['status'] == expected, 'smoke/full receipt mismatch')
    need(receipt['caveat'] == CAVEAT and receipt['binding_sha256'] == sha(STAGE / 'registration.json'), 'receipt binding drift')
    p956.check_pin(output / 'predictions.npz', receipt['prediction_sha256'])
    ids, y = gold()
    if smoke:
        ids, y = ids[:2], y[:2]
    need(receipt['studies'] == len(ids) and len(receipt['inputs']) == len(ids), 'receipt study count mismatch')
    need(receipt['device'] == ('cpu' if smoke else 'cuda'), 'receipt device mismatch')
    with np.load(output / 'predictions.npz', allow_pickle=False) as data:
        need(data['ids'].tolist() == ids and data['targets'].tolist() == list(LABELS), 'prediction roster/target mismatch')
        check_predictions(data, len(ids))
        stats = paired_stats(ids, data['parent'], data['member'], y)
    result = dict(version=VERSION, status='CPU_SMOKE_SCORING_ONLY' if smoke else 'GOLD58_EXPOSED_PAIRED_DIAGNOSTIC',
                  caveat=CAVEAT, studies=len(ids), receipt_sha256=sha(output / 'receipt.json'),
                  binding_sha256=sha(STAGE / 'registration.json'), bootstrap=binding['bootstrap'],
                  bootstrap_policy='Paired study-singleton percentile CI; fixed targets; invalid class draws omitted and counted. No patient-component contract.',
                  decision='Diagnostic only; no automatic release/public-comparison authorization', **stats)
    write(output / 'score.json', result)
    return dict(status=result['status'], studies=len(ids), score_sha256=sha(output / 'score.json'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('register')
    sub.add_parser('validate')
    p = sub.add_parser('infer')
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--dicom-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p = sub.add_parser('score')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'register':
        result = register()
    elif args.command == 'validate':
        validate()
        result = dict(status='VALID', studies=58)
    elif args.command == 'infer':
        result = infer(args.device, args.output, args.dicom_root, args.smoke)
    else:
        result = score(args.output, args.smoke)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
