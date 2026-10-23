"""B_RN3F exact100 numerical gate. Passive import; explicit CPU input/evaluation only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

SCHEMA = 'B_RN3F_PARITY_V1'
SERVING_SHA = '40f5a04dd0f770f05e9af5cf9a3827ccf5748c7989847c0778ba189fde40f80e'
PRODUCER_SHA = 'bd0fcaf4a2ddb9c8eb975a80f9ef1ae5250bd36d7b5ea6a4eefbf371014c7b5b'
UID_SHA = '824251f64754854339b3972f5fe3c30fce8b347e2b4712f34a3fdd3eaae46981'
MEMBERS = ['B_F0', 'RN_F0', 'B_F1', 'RN_F1', 'B_F2', 'RN_F2']
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
SYNTHETIC_NAMES = ['allzero', 'slot_boundary', 'sparse_isolated', 'sparse_valid', 'highcontrast', 'ramp']
SOURCE_SHA = {'raptor_common.py': 'bc6db5b3e8cd13b52fbff8c039d3674cdf573f86c1959e22fe9b2fd94255d7dc',
              'raptor_runtime.py': 'afe394b631c681e3826ad976e71c793214dee81c49c0d9cd76046590cc4513ad',
              'raptor_dicom.py': 'e478626baaf757260de3f8b03f143e727ebb130f575404788e2d9dad1264867e'}
LIMITS = {'window': {'max_abs': 5e-7, 'p99_abs': 5e-7, 'mean_abs': 5e-7},
          'normalized': {'max_abs': 3e-6, 'p99_abs': 3e-6, 'mean_abs': 3e-6},
          'probability_per_member_target': {'max_abs': .002, 'p99_abs': .002, 'mean_abs': .0005},
          'rank': 'Elementwise average-rank displacement <= reference rank interval bound from +/-0.002 per raw probability; all comparisons over complete100',
          'cross_host_normalized_envelope': 6e-6, 'rtol': 0.0}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_sha(value):
    import numpy as np
    value = np.ascontiguousarray(value, dtype='<f4')
    header = json.dumps({'shape': list(value.shape), 'dtype': '<f4'}, sort_keys=True, separators=(',', ':')).encode()
    h = hashlib.sha256(header + b'\n'); h.update(memoryview(value).cast('B')); return h.hexdigest()


def constants():
    import numpy as np
    return (np.asarray([.485, .456, .406], dtype=np.float32).reshape(1, 3, 1, 1),
            np.asarray([.229, .224, .225], dtype=np.float32).reshape(1, 3, 1, 1))


def centers_reference(mask):
    """Independent integer selection; K24 has denominator23, hence no .5 ties."""
    import numpy as np
    mask = np.asarray(mask)
    need(mask.dtype == np.uint8 and mask.shape == (64,) and bool(np.isin(mask, [0, 1]).all()), 'binary uint8 mask64 required')
    candidates, offset = [], 0
    for size in [18, 14, 12, 8, 12]:
        candidates.extend(i for i in range(offset + 1, offset + size - 1) if all(int(mask[j]) == 1 for j in (i-1, i, i+1)))
        offset += size
    if not candidates: return [-1] * 24
    return [candidates[(2 * j * (len(candidates)-1) + 23) // 46] for j in range(24)]


def analytical_reference(volume, centers):
    """No torch/OpenCV: exact-coordinate float64 bilinear plus frozen float32 constants."""
    import numpy as np
    need(volume.dtype == np.uint8 and volume.shape == (64, 336, 336), 'uint8 volume64x336x336 required')
    need(len(centers) == 24 and all(type(c) is int and (c == -1 or 1 <= c <= 62) for c in centers), 'K24 integer centers required')
    q = (np.arange(224, dtype=np.float64) + .5) * 1.5 - .5
    lo = np.floor(q).astype(int); hi = lo + 1; weight = q - lo
    need(set(np.unique(weight)) == {.25, .75}, 'bilinear coordinates changed')
    windows = np.zeros((24, 3, 224, 224), dtype=np.float64)
    for i, center in enumerate(centers):
        if center == -1: continue
        value = volume[center-1:center+2].astype(np.float64) / 255.0
        horizontal = value[..., lo] * (1-weight) + value[..., hi] * weight
        windows[i] = horizontal[..., lo, :] * (1-weight)[:, None] + horizontal[..., hi, :] * weight[:, None]
    mean, std = constants()
    return windows, (windows - mean.astype(np.float64)) / std.astype(np.float64)


def errors(actual, reference, limits):
    import numpy as np
    need(actual.shape == reference.shape and actual.dtype == np.float32, 'float32 tensor shape mismatch')
    need(bool(np.isfinite(actual).all() and np.isfinite(reference).all()), 'nonfinite tensor')
    delta = np.abs(actual.astype(np.float64) - reference)
    observed = {'max_abs': float(delta.max()), 'p99_abs': float(np.quantile(delta, .99)), 'mean_abs': float(delta.mean())}
    return {**observed, 'limits': limits, 'pass': all(observed[k] <= v for k, v in limits.items())}


def input_case(uid, volume, mask, centers, windows, normalized):
    import numpy as np
    expected_centers = centers_reference(mask)
    need(list(centers) == expected_centers, 'within-slot-valid centers differ from independent selection')
    need(volume.dtype == np.uint8 and volume.shape == (64, 336, 336), 'input volume contract')
    need(np.array_equal(mask, (volume.reshape(64, -1).sum(1) > 0).astype(np.uint8)), 'mask does not match uint8 volume')
    wref, nref = analytical_reference(volume, expected_centers)
    wc, nc = errors(windows, wref, LIMITS['window']), errors(normalized, nref, LIMITS['normalized'])
    return {'uid': uid, 'volume_sha256': array_sha(volume), 'mask_sha256': array_sha(mask),
            'selected_centers': expected_centers, 'empty_windows': expected_centers.count(-1),
            'normalized_sha256': tensor_sha(normalized), 'window_sha256': tensor_sha(windows),
            'shape': [24, 3, 224, 224], 'dtype': 'float32', 'window': wc, 'normalized': nc,
            'pass': wc['pass'] and nc['pass']}


def synthetics():
    import numpy as np
    y, x = np.indices((336, 336))
    for name in SYNTHETIC_NAMES:
        volume = np.zeros((64, 336, 336), np.uint8)
        if name == 'slot_boundary': volume[16:20] = 255
        if name == 'sparse_isolated': volume[[1, 20, 45]] = 255
        if name == 'sparse_valid': volume[[2, 3, 4, 33, 34, 35]] = 255
        if name == 'highcontrast': volume[:] = ((x+y) % 2 * 255).astype(np.uint8)
        if name == 'ramp': volume[:] = ((3*x + 7*y) % 256).astype(np.uint8)
        mask = (volume.reshape(64, -1).sum(1) > 0).astype(np.uint8)
        yield name, volume, mask


def rank_pct(values):
    import numpy as np
    # Pairwise enumeration is independent of the producer's stable-sort grouping.
    out = np.empty(values.shape, np.float64)
    for j in range(values.shape[1]):
        column = values[:, j]
        out[:, j] = ((column[None, :] < column[:, None]).sum(1) +
                     .5 * ((column[None, :] == column[:, None]).sum(1) + 1)) / len(column)
    return out


def rank_bound(reference, epsilon=.002):
    """Possible rank interval: only reference neighbours within2epsilon can cross."""
    import numpy as np
    value = reference.astype(np.float64); reference_rank = rank_pct(value)
    low = np.empty(value.shape); high = np.empty(value.shape)
    for j in range(value.shape[1]):
        column = value[:, j]
        low[:, j] = ((column[None, :] < column[:, None] - 2*epsilon).sum(1) + 1) / len(column)
        high[:, j] = (column[None, :] <= column[:, None] + 2*epsilon).sum(1) / len(column)
    return np.maximum(reference_rank-low, high-reference_rank)


def metric_delta(delta):
    import numpy as np
    return {'max_abs': float(delta.max()), 'p99_abs': float(np.quantile(delta, .99)), 'mean_abs': float(delta.mean())}


def probability_gate(actual, reference):
    import numpy as np
    for values in (actual, reference):
        need(values.dtype == np.float32 and values.shape == (6, 100, 12), 'exact six x100x12 float32 probabilities required')
        need(bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()), 'invalid probabilities')
    delta = np.abs(actual.astype(np.float64)-reference.astype(np.float64))
    arank = np.stack([rank_pct(v) for v in actual]); rrank = np.stack([rank_pct(v) for v in reference])
    bounds = np.stack([rank_bound(v) for v in reference]); rankdelta = np.abs(arank-rrank)
    rows = []
    for i, member in enumerate(MEMBERS):
        for j, target in enumerate(TARGETS):
            observed = metric_delta(delta[i, :, j]); rd = metric_delta(rankdelta[i, :, j]); rb = metric_delta(bounds[i, :, j])
            probability_pass = all(observed[k] <= v for k, v in LIMITS['probability_per_member_target'].items())
            rank_pass = bool((rankdelta[i, :, j] <= bounds[i, :, j] + 1e-15).all())
            rows.append({'member': member, 'target': target, 'probability': observed, 'rank_delta': rd,
                         'rank_allowed': rb, 'pass': probability_pass and rank_pass})
    blend = lambda values: (.9 * values[::2] + .1 * values[1::2]).mean(0)
    bd = np.abs(blend(arank)-blend(rrank)); bb = blend(bounds)
    return {'pass': all(r['pass'] for r in rows) and bool((bd <= bb + 1e-15).all()), 'member_target': rows,
            'blend_rank_delta': metric_delta(bd), 'blend_rank_allowed': metric_delta(bb),
            'rank_bound_roundoff_atol': 1e-15, 'limits': LIMITS, 'quality_claim': False}


def freeze(out):
    out = Path(out); need(not out.exists(), 'new immutable protocol file required')
    protocol = {'schema': SCHEMA, 'frozen_at': datetime.now(timezone.utc).isoformat(), 'helper_sha256': sha(__file__),
                'serving_manifest_sha256': SERVING_SHA, 'producer_sha256': PRODUCER_SHA, 'ordered_uids_sha256': UID_SHA,
                'members': MEMBERS, 'targets': TARGETS, 'studies': 100, 'limits': LIMITS, 'synthetic_names': SYNTHETIC_NAMES,
                'selection': 'existing engineering100; no clinical independence or quality claim',
                'status': 'FROZEN_BEFORE_NEW_B_REAL100_INPUT_OR_OUTPUT_OBSERVATION',
                'input_math': '336/224=1.5, quarter/three-quarter exact coordinate weights. Standard float32 bilinear envelope4.7684e-7 windows/2.4435e-6 normalized, not a compiler proof; -1 means zeros before ImageNet normalization.',
                'probability_basis': 'Prior D01 raw max.002/mean.0005 are prospective engineering budgets, NOT measured B host differences or inherited PASS. Apply per-member/target, stricter scope than prior pooled-member mean.',
                'p99_basis': 'p99 uses existing max bound; no independent empirical percentile threshold invented.',
                'rank_basis': 'Each raw probability may move +/-epsilon; only reference values within2epsilon can cross. Fixed formula yields per-study/target rank interval; measured reference density is not threshold tuning.',
                'no_post_observation_tolerance_relaxation': True, 'submission_ready': False,
                'separate_gates': ['actual compressed codec fixture', 'RSS/cgroup and whole-session runtime', 'hidden full-cohort release', 'competitive quality']}
    write_new(out, protocol); return protocol


def validate_protocol(path, digest):
    need(sha(path) == digest, 'protocol SHA mismatch')
    p = read(path)
    need(p['schema'] == SCHEMA and p['helper_sha256'] == sha(__file__) and p['limits'] == LIMITS
         and p['members'] == MEMBERS and p['targets'] == TARGETS and p['studies'] == 100
         and p['serving_manifest_sha256'] == SERVING_SHA and p['producer_sha256'] == PRODUCER_SHA
         and p['ordered_uids_sha256'] == UID_SHA, 'frozen protocol changed')
    need(p['synthetic_names'] == SYNTHETIC_NAMES, 'synthetic suite changed')
    return p


def passive_sources(assets):
    assets = Path(assets)
    need(sha(assets / 'b-rn3f-assets.json') == SERVING_SHA, 'serving stage identity mismatch')
    for name, digest in SOURCE_SHA.items(): need(sha(assets / 'training/src' / name) == digest, 'passive source differs')
    need(not any(n in sys.modules for n in ('raptor_common', 'raptor_runtime', 'raptor_dicom', 'torch')), 'fresh CPU process required')
    os.environ.update(CUDA_VISIBLE_DEVICES='', KNEE_WORKSPACE_ROOT=str(assets.resolve()), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(assets / 'training/src'))
    import raptor_common, raptor_runtime, torch
    torch.set_num_threads(1)
    need(not torch.cuda.is_initialized(), 'CPU gate must not initialize CUDA')
    return raptor_common, raptor_runtime


def run_input(args):
    validate_protocol(args.protocol, args.protocol_sha256)
    need(not Path(args.out).exists(), 'new input report required')
    common, runtime = passive_sources(args.assets)
    mean, std = constants()
    need(array_sha(runtime.IMAGENET_MEAN) == array_sha(mean) and array_sha(runtime.IMAGENET_STD) == array_sha(std), 'normalization constants changed')
    ids = read(Path(args.assets) / 'engineering100.json')['ids']
    need(len(ids) == 100 and hashlib.sha256('\n'.join(ids).encode()).hexdigest() == UID_SHA, 'exact100 UID contract')
    source = None
    if args.target == 'local-reference':
        need(args.cache_root and not args.competition, 'explicit frozen cache required')
        manifest_sha = read(Path(args.assets) / 'b-rn3f-assets.json')['reference_cache_manifest_sha256']
        need(sha(Path(args.cache_root) / 'manifest.json') == manifest_sha, 'cache manifest identity changed')
        corpus = common.Corpus(args.cache_root)
    else:
        need(args.competition and not args.cache_root, 'explicit competition DICOM source required')
        import cv2, pydicom, gdcm
        need((cv2.__version__, pydicom.__version__, gdcm.Version.GetVersion()) == ('4.12.0', '3.0.2', '3.2.6'), 'activate fixed producer offline environment before CPU input gate')
        from raptor_dicom import DirectoryDicomSource, DicomCorpus
        import csv
        with (Path(args.competition) / 'train_series.csv').open(encoding='utf-8-sig', newline='') as stream:
            series = list(csv.DictReader(stream))
        source = DirectoryDicomSource(Path(args.competition) / 'train_series')
        corpus = DicomCorpus.from_records(ids, series, source, contract='raptor64_336_c140_s02', split='train')
    def check(uid, volume, mask):
        centers = common.select_centers(mask, 64, 24, {'slots': common.SLOTS64}, policy='within-slot-valid')
        windows = common.windows_from(volume, centers, 224)
        return input_case(uid, volume, mask, centers, windows, runtime.normalize_input(windows, 'imagenet'))
    try:
        synthetic = [check(name, volume, mask) for name, volume, mask in synthetics()]
        rows = [check(uid, corpus.volume(uid), corpus.mask(uid)) for uid in ids]
        report = {'schema': SCHEMA, 'protocol_sha256': args.protocol_sha256, 'helper_sha256': sha(__file__),
                  'serving_manifest_sha256': SERVING_SHA, 'target': args.target,
                  'observed_at': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                  'rows': rows, 'synthetic': synthetic, 'pass': all(r['pass'] for r in rows + synthetic),
                  'gpu_execution': False, 'cpu_workers': 1, 'reason': 'sequential bounded tensor/reference memory; no nested pool'}
        write_new(args.out, report); return report
    finally:
        if source is not None: source.close()


def validate_input_report(report, receipt, ids, protocol_sha, target):
    need(report.get('schema') == SCHEMA and report.get('protocol_sha256') == protocol_sha
         and report.get('helper_sha256') == sha(__file__) and report.get('serving_manifest_sha256') == SERVING_SHA
         and report.get('target') == target, 'input report provenance mismatch')
    rows = report.get('rows', [])
    need([r['uid'] for r in rows] == ids and report.get('pass') is True, 'input100 report incomplete/failed')
    need([r['uid'] for r in report.get('synthetic', [])] == SYNTHETIC_NAMES, 'synthetic suite incomplete')
    for r in rows + report['synthetic']:
        need(r.get('pass') is True and r['shape'] == [24, 3, 224, 224] and r['dtype'] == 'float32', 'input case failed')
        for name in ['window', 'normalized']:
            need(r[name]['limits'] == LIMITS[name] and r[name]['pass'] is True
                 and all(0 <= r[name][k] <= limit for k, limit in LIMITS[name].items()), 'input numerical limit failure')
    actual_inputs = receipt['runtime']['input_evidence']
    need(len(actual_inputs) == 100, 'producer input evidence incomplete')
    for row, actual, normalized in zip(rows, actual_inputs, receipt['inference']['input_tensor_hashes']):
        need(all(row[key] == actual[key] for key in ['uid', 'volume_sha256', 'mask_sha256', 'selected_centers', 'empty_windows'])
             and row['normalized_sha256'] == normalized['sha256'], 'input gate does not bind actual producer tensors')
    return rows


def evaluate(args):
    import numpy as np
    validate_protocol(args.protocol, args.protocol_sha256)
    need(not Path(args.out).exists(), 'new paired report required')
    assets = Path(args.assets)
    need(sha(assets / 'b-rn3f-assets.json') == SERVING_SHA and sha(assets / 'b_rn3f_serving.py') == PRODUCER_SHA, 'producer identity differs')
    spec = importlib.util.spec_from_file_location('frozen_b_producer', assets / 'b_rn3f_serving.py')
    producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
    ids = read(assets / 'engineering100.json')['ids']
    need(len(ids) == 100 and producer.ids_sha(ids) == UID_SHA, 'wrong paired cohort')
    receipts, reports, arrays = [], [], []
    for work, path, target in [(args.local_work, args.local_input, 'local-reference'), (args.t4_work, args.t4_input, 't4-engineering')]:
        receipt = producer.validate_result(work, assets, ids, target, 'engineering100')
        report = read(path); rows = validate_input_report(report, receipt, ids, args.protocol_sha256, target)
        receipts.append(receipt); reports.append(rows)
        with np.load(Path(work) / 'result/members.npz', allow_pickle=False) as data: arrays.append(data['member_probabilities'])
    exact_keys = ['uid', 'volume_sha256', 'mask_sha256', 'selected_centers', 'empty_windows', 'shape', 'dtype']
    exact = all(all(a[k] == b[k] for k in exact_keys) for a, b in zip(*reports))
    probability = probability_gate(arrays[1], arrays[0])
    result = {'schema': SCHEMA, 'protocol_sha256': args.protocol_sha256, 'serving_manifest_sha256': SERVING_SHA,
              'status': 'PASS_NUMERICAL_ONLY' if exact and probability['pass'] else 'FAILED',
              'integer_inputs_exact': exact, 'normalized_cross_host_bound': LIMITS['cross_host_normalized_envelope'],
              'normalized_cross_host_basis': 'triangle inequality of independently checked same-integer-input float64 reference; actual hashes retained, not asserted equal',
              'normalized_hash_mismatches': sum(a['normalized_sha256'] != b['normalized_sha256'] for a, b in zip(*reports)),
              'probabilities': probability, 'submission_ready': False, 'competitive_quality_claim': False,
              'input_report_sha256': [sha(args.local_input), sha(args.t4_input)],
              'producer_receipt_sha256': [sha(Path(w) / 'result/receipt.json') for w in [args.local_work, args.t4_work]],
              'observed_at': datetime.now(timezone.utc).isoformat()}
    write_new(args.out, result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--out', required=True)
    for action in ['input', 'evaluate']:
        q = sub.add_parser(action)
        for name in ['protocol', 'protocol-sha256', 'assets', 'out']: q.add_argument('--' + name, required=True)
        if action == 'input':
            q.add_argument('--target', choices=['local-reference', 't4-engineering'], required=True)
            q.add_argument('--cache-root'); q.add_argument('--competition')
        else:
            for name in ['local-work', 't4-work', 'local-input', 't4-input']: q.add_argument('--' + name, required=True)
    args = p.parse_args()
    try:
        result = freeze(args.out) if args.action == 'freeze' else run_input(args) if args.action == 'input' else evaluate(args)
    except Exception as exc:
        if args.action != 'freeze' and not Path(args.out).exists():
            write_new(args.out, {'schema': SCHEMA, 'status': 'FAILED', 'exception_type': type(exc).__name__,
                                 'error': str(exc), 'submission_ready': False})
        raise
    print(json.dumps({'schema': SCHEMA, 'status': result.get('status'), 'pass': result.get('pass')}, indent=2))


if __name__ == '__main__': main()
