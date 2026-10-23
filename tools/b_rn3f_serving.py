"""B_RN3F_SERVING_V1: immutable CPU staging and explicit engineering inference.

No training, remote publication, submission, model execution or backend change
occurs on import/build. Engineering outputs do not establish a numerical gate,
hidden-runtime PASS or a released competition candidate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'B_RN3F_SERVING_V1'
MARKER = 'b-rn3f-assets.json'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
MEMBERS = [
    ('B_F0', 'B', 0, 'EXP078_INPUT_B_F0', '74a70e8f4a684a53ad3b3489144eac16e5c5c500a602440cf8f8756ac03c9b17'),
    ('RN_F0', 'RN', 0, 'EXP079_R34_B_F0', '59304302e4d0bfb084f8e47c73cc91f7e290c2faef6b62f58124bdd05287e1d4'),
    ('B_F1', 'B', 1, 'EXP078_INPUT_B_F1', 'b2ebdc06490174048efaa5b559aa99258e013cfe075e60d62c89f326c72b58f9'),
    ('RN_F1', 'RN', 1, 'EXP079_R34_B_F1', 'a3cf3f22eeade5115270df81e47c673420ee7ce77416ade716e7620a2e07cfdb'),
    ('B_F2', 'B', 2, 'EXP080_INPUT_B_F2', '5e3399bb3f44bb43f81b58c4048cabad1c4657f78a4952d30b15a957f022992f'),
    ('RN_F2', 'RN', 2, 'EXP080_R34_B_F2', 'd198282aa26b2e06a26412942ef61ef318d789c16891801ffe363feb44ca9abe'),
]
CODE_SHA = {
    'raptor_common.py': 'bc6db5b3e8cd13b52fbff8c039d3674cdf573f86c1959e22fe9b2fd94255d7dc',
    'raptor_runtime.py': 'afe394b631c681e3826ad976e71c793214dee81c49c0d9cd76046590cc4513ad',
    'raptor_ensemble.py': 'e8cc79294be6fbc1c960eea1f5e38ab0ce8abb7d41e674d85274d03c15259de1',
    'raptor_dicom.py': 'e478626baaf757260de3f8b03f143e727ebb130f575404788e2d9dad1264867e',
}
AUDIT = 'docs/handoffs/evidence/2026-10/2026-10-04-candidate-readiness/artifact-audit.json'
AUDIT_SHA = 'b397775e24646e107a1b45cf6b82d0dacb5624c711ae76502789f466b555c679'
LEGACY_ASSETS = '.local/d01-staging/20261003-v1/assets'
LEGACY_MANIFEST_SHA = 'e62893ebb1e22dbb1288ea6c805097bbc96ef5e43759f42a22d7cfb3bbef036b'
SELECTION = '.local/d01-runtime-staging/20261003-v2/selection-manifest.json'
SELECTION_SHA = '599c7e81784fbccfafc213602bd23493bd0c6dfb61c0c25507f89213031d4a61'
ENGINEERING100_UID_SHA = '824251f64754854339b3972f5fe3c30fce8b347e2b4712f34a3fdd3eaae46981'
OPENCV_SHA = '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'
CACHE_MANIFEST_SHA = 'f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
CONFIG_SHA = '2485feb52868267ad311ff631537a81f5434ec89d2c7e26ad408d86345db7f1f'
WHEEL_SHA = {
    'timm-1.0.19-py3-none-any.whl': 'c07b56c32f3d3226c656f75c1b5479c08eb34eefed927c82fd8751a852f47931',
    'pydicom-3.0.2-py3-none-any.whl': 'abf971a5440f84dbaf42c4b6758e30e62480902584f8b270b9a5d146e278a07b',
    'python_gdcm-3.2.6-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl':
        '512d0d903e12c5ad389939a257817bf182ce288431b3d474d62cb0c0833b8b56',
}
INPUT = {'contract': 'raptor64_336_c140_s02', 'targets': TARGETS, 'res': 224, 'k_eval': 24,
         'norm': 'imagenet', 'window_policy': 'within-slot-valid'}
POLICY = {'member_order': [m[0] for m in MEMBERS], 'input': INPUT,
          'formula': 'mean_f[0.9*rank_pct(B_f)+0.1*rank_pct(RN_f)]',
          'folds': [0, 1, 2], 'weights': {'B': .9, 'RN': .1}, 'rank_ties': 'average',
          'rank_denominator': 'complete supplied cohort N, separately for each target and member',
          'rank_dtype': 'float64', 'raw_dtype': 'float32', 'final_rerank': False,
          'shard_local_rank': False, 'anchor_included': False,
          'amp': 'fp16', 'runtime_policy': 'strict', 'batch': 1, 'residency': 'resident',
          'devices': 1, 'header_policy': 'pinned-legacy-full-read', 'automatic_retries': 0}
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def valid_ids(ids):
    ids = list(ids)
    need(bool(ids) and all(isinstance(u, str) and u and not any(c in u for c in ('/', '\\', '\x00', '..')) for u in ids),
         'nonempty safe string UIDs required')
    need(len(ids) == len(set(ids)), 'duplicate UIDs')
    return ids


def ids_sha(ids):
    return hashlib.sha256('\n'.join(valid_ids(ids)).encode()).hexdigest()


def rank_pct(values):
    """Average-tie percentile ranks; complete cohort required by the caller."""
    import numpy as np
    values = np.asarray(values)
    need(values.ndim == 2 and values.shape[1] == 12 and len(values) > 0, 'expected nonempty Nx12 probabilities')
    need(bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()), 'nonfinite/out-of-range probabilities')
    result = np.empty(values.shape, dtype=np.float64)
    for j in range(12):
        order = np.argsort(values[:, j], kind='stable')
        column = values[order, j]
        starts = np.r_[0, np.flatnonzero(column[1:] != column[:-1]) + 1]
        stops = np.r_[starts[1:], len(column)]
        for start, stop in zip(starts, stops):
            result[order[start:stop], j] = (float(start + 1) + float(stop)) / (2 * len(column))
    return result


def aggregate(raw, ids, expected_ids, member_names):
    """Rank only after all six members and the complete ordered cohort exist."""
    import numpy as np
    ids, expected_ids = valid_ids(ids), valid_ids(expected_ids)
    need(ids == expected_ids, 'missing/unexpected/reordered cohort UIDs; align raw before rank')
    need(list(member_names) == POLICY['member_order'], 'exact ordered six members required')
    raw = np.asarray(raw)
    need(raw.dtype == np.float32 and raw.shape == (6, len(ids), 12), 'raw must be float32[6,N,12]')
    need(bool(np.isfinite(raw).all() and (raw >= 0).all() and (raw <= 1).all()), 'invalid member probabilities')
    ranks = np.stack([rank_pct(member) for member in raw])
    paired = np.stack([.9 * ranks[2 * f] + .1 * ranks[2 * f + 1] for f in range(3)])
    output = paired.mean(axis=0, dtype=np.float64)
    need(bool(np.isfinite(output).all() and (output >= 0).all() and (output <= 1).all()), 'invalid rank blend')
    return output


def merge_raw_shards(expected_ids, shards):
    """Pure raw-only join helper; production V1 still executes on one device."""
    import numpy as np
    expected_ids = valid_ids(expected_ids)
    location = {}
    for shard_ids, member_names, raw in shards:
        shard_ids = valid_ids(shard_ids)
        need(list(member_names) == POLICY['member_order'], 'shard member identity/order mismatch')
        raw = np.asarray(raw)
        need(raw.dtype == np.float32 and raw.shape == (6, len(shard_ids), 12), 'shards must carry raw float32 probabilities')
        need(bool(np.isfinite(raw).all() and (raw >= 0).all() and (raw <= 1).all()), 'invalid shard probabilities')
        for i, uid in enumerate(shard_ids):
            need(uid not in location, 'duplicate shard UID')
            location[uid] = raw[:, i, :]
    need(set(location) == set(expected_ids), 'missing/unexpected shard UIDs')
    return np.stack([location[u] for u in expected_ids], axis=1)


def validate_frame(path, expected_ids):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        need(next(reader, None) == [UID, *TARGETS], 'output target schema/order mismatch')
        rows = list(reader)
    need(all(len(row) == 13 for row in rows), 'output row width')
    need([r[0] for r in rows] == valid_ids(expected_ids), 'output UID order/completeness mismatch')
    for row in rows:
        need(len(row) == 13, 'output row width')
        values = [float(v) for v in row[1:]]
        need(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'invalid output value')
    return len(rows)


def tensor_sha256(value):
    """Same shape/little-endian float32 hash as the pinned passive runtime."""
    import numpy as np
    value = np.ascontiguousarray(value, dtype='<f4')
    header = json.dumps({'shape': list(value.shape), 'dtype': '<f4'}, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(header + b'\n'); digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def validate_inference(inference, ids, raw):
    """A COMPLETE word cannot publish missing members, fallback or partial inputs."""
    need(inference.get('status') == 'COMPLETE' and inference.get('all_finite') is True,
         'inference not complete/finite')
    need(inference.get('input_contract') == INPUT and inference.get('studies') == len(ids)
         and inference.get('member_count') == 6, 'inference scope mismatch')
    need(inference.get('requested_amp') == inference.get('effective_amp') == 'fp16'
         and inference.get('batch_size') == 1 and inference.get('residency') == 'resident', 'inference policy mismatch')
    need(inference.get('runtime', {}).get('policy') == 'strict', 'strict inference receipt required')
    need(inference.get('fp32_fallbacks') == inference.get('neutral_fills') == 0
         and not inference.get('failures') and not inference.get('cleanup_failures'), 'inference failure/fallback')
    need(inference.get('input_preparations') == inference.get('completed_batches') == len(ids), 'incomplete inputs/batches')
    records = inference.get('input_tensor_hashes', [])
    need([r.get('uid') for r in records] == ids, 'input receipt UID order mismatch')
    need(all(isinstance(r.get('sha256'), str) and len(r['sha256']) == 64
             and all(c in '0123456789abcdef' for c in r['sha256']) for r in records), 'invalid input digest')
    digest = hashlib.sha256()
    for record in records:
        digest.update((json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
    need(inference.get('ordered_inputs_sha256') == digest.hexdigest(), 'input receipt digest mismatch')
    results = inference.get('member_results', [])
    need([r.get('name') for r in results] == POLICY['member_order'], 'incomplete/misordered member receipts')
    for i, (result, member) in enumerate(zip(results, expected_members())):
        need(result.get('status') == 'COMPLETE' and result.get('completed_studies') == len(ids)
             and result.get('checkpoint_sha256') == member['sha256']
             and result.get('ordered_inputs_sha256') == digest.hexdigest()
             and result.get('probabilities_sha256') == tensor_sha256(raw[i]), 'member output provenance mismatch')


def atomic_outputs(work, ids, raw, manifest_sha256, inference, runtime_metadata):
    """Atomically publish one complete result directory, never a partial final CSV."""
    import numpy as np
    work = Path(work)
    pending, final = work / 'result.pending', work / 'result'
    need(not pending.exists() and not final.exists(), 'refuse existing result directory')
    ids = valid_ids(ids)
    values = aggregate(raw, ids, ids, POLICY['member_order'])
    validate_inference(inference, ids, raw)
    pending.mkdir()
    with (pending / 'members.npz').open('xb') as stream:
        np.savez_compressed(stream, study_uids=np.asarray(ids), member_names=np.asarray(POLICY['member_order']),
                            targets=np.asarray(TARGETS), member_probabilities=raw)
    with (pending / 'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow([UID, *TARGETS])
        writer.writerows([[uid, *row] for uid, row in zip(ids, values)])
        stream.flush(); os.fsync(stream.fileno())
    validate_frame(pending / 'predictions.csv', ids)
    receipt = {'schema': SCHEMA, 'status': 'COMPLETE', 'artifact_kind': 'unreleased_engineering_rank_blend',
               'serving_parity_gate': 'NOT_EVALUATED', 'submission_ready': False, 'accuracy_claim': False,
               'manifest_sha256': manifest_sha256, 'policy_sha256': POLICY_SHA,
               'studies': len(ids), 'ordered_uids_sha256': ids_sha(ids), 'members': POLICY['member_order'],
               'targets': TARGETS, 'policy': POLICY, 'inference': inference, 'runtime': runtime_metadata,
               'output_sha256': {n: sha(pending / n) for n in ('members.npz', 'predictions.csv')}}
    write_new(pending / 'receipt.json', receipt)
    # One same-volume directory rename publishes CSV, raw arrays and their receipt together.
    os.replace(pending, final)
    return receipt


def validate_result(work, assets, ids, target, cohort):
    import numpy as np
    result = Path(work) / 'result'
    final = read(result / 'receipt.json')
    need(final.get('schema') == SCHEMA and final.get('status') == 'COMPLETE'
         and final.get('manifest_sha256') == sha(Path(assets) / MARKER)
         and final.get('policy_sha256') == POLICY_SHA and final.get('policy') == POLICY,
         'wrong worker output receipt')
    need(final.get('submission_ready') is False and final.get('serving_parity_gate') == 'NOT_EVALUATED',
         'engineering producer cannot claim a parity/release gate')
    need(final.get('studies') == len(ids) and final.get('ordered_uids_sha256') == ids_sha(ids)
         and final.get('members') == POLICY['member_order'] and final.get('targets') == TARGETS, 'output scope mismatch')
    need(final.get('runtime', {}).get('target') == target and final['runtime'].get('cohort') == cohort,
         'output runtime/cohort mismatch')
    need(set(final.get('output_sha256', {})) == {'members.npz', 'predictions.csv'}, 'output inventory mismatch')
    for name, expected in final['output_sha256'].items():
        need(sha(result / name) == expected, 'published output hash mismatch')
    validate_frame(result / 'predictions.csv', ids)
    with np.load(result / 'members.npz', allow_pickle=False) as data:
        need(set(data.files) == {'study_uids', 'member_names', 'targets', 'member_probabilities'}, 'raw output schema mismatch')
        need(data['study_uids'].tolist() == ids and data['targets'].tolist() == TARGETS, 'raw UID/target order mismatch')
        raw = data['member_probabilities']
        values = aggregate(raw, ids, ids, data['member_names'].tolist())
        validate_inference(final['inference'], ids, raw)
    with (result / 'predictions.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.reader(stream))[1:]
    observed = np.asarray([[float(x) for x in row[1:]] for row in rows], dtype=np.float64)
    need(np.array_equal(values, observed), 'published rank blend differs from raw members')
    return final


def expected_members():
    return [{'name': name, 'family': family, 'fold': fold, 'training_run': tag,
             'path': 'weights/' + name + '.pt', 'sha256': digest,
             'arch': 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k' if family == 'B' else 'resnet34.a1_in1k',
             'epoch_1based': 8 if family == 'B' else 20}
            for name, family, fold, tag, digest in MEMBERS]


def validate_assets(assets):
    assets = Path(assets).resolve()
    manifest = read(assets / MARKER)
    need(manifest.get('schema') == SCHEMA and manifest.get('policy') == POLICY
         and manifest.get('policy_sha256') == POLICY_SHA, 'serving policy changed')
    need(manifest.get('members') == expected_members(), 'exact member metadata changed')
    need(manifest.get('validation_status') == 'CPU_STAGED_NOT_INFERENCE_VALIDATED', 'unexpected validation status')
    required = {'b_rn3f_serving.py', 'config/workspace.toml', 'engineering100.json'}
    required.update('training/src/' + name for name in CODE_SHA)
    required.update(m['path'] for m in expected_members())
    required.update('wheels/' + name for name in WHEEL_SHA)
    need(required <= manifest['files'].keys(), 'incomplete asset inventory')
    need(manifest.get('reference_cache_manifest_sha256') == CACHE_MANIFEST_SHA, 'reference cache identity changed')
    need(manifest.get('wheels') == [{'path': 'wheels/' + name, 'sha256': digest} for name, digest in WHEEL_SHA.items()],
         'offline wheel contract changed')
    for relative, expected in manifest['files'].items():
        path = Path(relative)
        need(not path.is_absolute() and not path.drive and '..' not in path.parts, 'unsafe asset relative path')
        path = (assets / path).resolve()
        need(path.is_relative_to(assets) and path.is_file() and sha(path) == expected, 'asset missing/changed: ' + relative)
    need(sha(assets / 'b_rn3f_serving.py') == sha(__file__), 'executing helper differs from staged helper')
    for name, expected in CODE_SHA.items():
        need(manifest['files']['training/src/' + name] == expected, 'pinned inference source changed')
    for member in expected_members():
        need(manifest['files'][member['path']] == member['sha256'], 'checkpoint inventory identity changed')
    for name, digest in WHEEL_SHA.items():
        need(manifest['files']['wheels/' + name] == digest, 'wheel inventory identity changed')
    need(manifest['files']['config/workspace.toml'] == CONFIG_SHA, 'workspace config changed')
    engineering = read(assets / 'engineering100.json')
    need(len(valid_ids(engineering['ids'])) == 100
         and ids_sha(engineering['ids']) == manifest['engineering100_uid_sha256'] == ENGINEERING100_UID_SHA
         and engineering.get('selection_source_sha256') == manifest.get('engineering_selection_source_sha256') == SELECTION_SHA,
         'engineering100 cohort changed')
    return manifest


def build(root, out, root_decision):
    """Copy exact existing assets to a new stage; never import torch or run a model."""
    root, out, root_decision = Path(root), Path(out), Path(root_decision)
    need(not out.exists(), 'new immutable stage required')
    need(root_decision.is_file(), 'root decision record must exist before staging')
    decision = read(root_decision)['selected_cpu_serving']
    need(decision['id'] == SCHEMA and decision['anchor_included'] is False, 'wrong serving decision')
    need(sha(root / AUDIT) == AUDIT_SHA, 'completed asset audit identity changed')
    audit = read(root / AUDIT)
    legacy = root / LEGACY_ASSETS
    need(sha(legacy / 'd01-assets-manifest.json') == LEGACY_MANIFEST_SHA, 'legacy source bundle changed')
    old_manifest = read(legacy / 'd01-assets-manifest.json')
    need(sha(root / SELECTION) == SELECTION_SHA, 'existing engineering selection changed')
    selection = read(root / SELECTION)
    engineering = {'ids': valid_ids(selection['ids']), 'selection_source_sha256': sha(root / SELECTION),
                   'scope': 'existing engineering100; no labels/Report or model predictions copied'}
    need(len(engineering['ids']) == 100, 'expected existing100 engineering IDs')
    assets = out / 'assets'; assets.mkdir(parents=True)
    def copy(source, relative, expected):
        need(sha(source) == expected, 'source identity mismatch: ' + str(source))
        target = assets / relative; target.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
        need(sha(target) == expected, 'copied bytes differ')
    for member in expected_members():
        copy(root / 'artifacts/experiments' / member['training_run'] / 'model_best.pt', member['path'], member['sha256'])
    for name, digest in CODE_SHA.items():
        copy(legacy / 'training/src' / name, 'training/src/' + name, digest)
    copy(legacy / 'config/workspace.toml', 'config/workspace.toml', old_manifest['files']['config/workspace.toml'])
    copy(Path(__file__), 'b_rn3f_serving.py', sha(__file__))
    wheel_sources = [key for key in audit['sources'] if key.endswith('.whl')]
    need(len(wheel_sources) == 3, 'expected timm/pydicom/gdcm offline wheel inventory')
    wheels = []
    for uri in wheel_sources:
        source = root / uri.removeprefix('workspace://'); digest = audit['sources'][uri]['sha256']
        relative = 'wheels/' + source.name; copy(source, relative, digest)
        wheels.append({'path': relative, 'sha256': digest})
    write_new(assets / 'engineering100.json', engineering)
    manifest = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
                'validation_status': 'CPU_STAGED_NOT_INFERENCE_VALIDATED', 'members': expected_members(),
                'policy': POLICY, 'policy_sha256': POLICY_SHA, 'wheels': wheels,
                'external_opencv': {'dataset': 'mattiaangeli/opencv-python-headless-4120088-x86', 'sha256': OPENCV_SHA},
                'engineering100_uid_sha256': ids_sha(engineering['ids']), 'engineering_selection_source_sha256': sha(root / SELECTION),
                'completed_audit_sha256': AUDIT_SHA, 'root_decision_sha256': sha(root_decision),
                'reference_cache_manifest_sha256': CACHE_MANIFEST_SHA,
                'cpu_policy': 'New parallel CPU default12 capped by effective allocation; reduce after throughput/RAM/I/O evidence. Small staging reads/copies serial; frozen training workers unchanged.',
                'readiness': 'No measured B input or exact6 local/T4 parity; no hidden-runtime evidence or release authorization.',
                'files': {p.relative_to(assets).as_posix(): sha(p) for p in sorted(assets.rglob('*')) if p.is_file()}}
    write_new(assets / MARKER, manifest)
    validate_assets(assets)
    review = {'schema': SCHEMA, 'status': 'CPU_STAGE_COMPLETE_NOT_GPU_VALIDATED',
              'assets': str(assets.resolve()), 'manifest_sha256': sha(assets / MARKER), 'policy_sha256': POLICY_SHA,
              'checkpoint_bytes': sum((assets / m['path']).stat().st_size for m in expected_members()),
              'source_root_decision_sha256': sha(root_decision), 'gpu_executed': False, 'remote_operations': 0,
              'implementation_limits': ['Engineering-only wrapper; no production release gate exists yet.',
                  'Actual numerical/reference/codec/runtime gates must be measured and reviewed; no D01 PASS inherited.']}
    write_new(out / 'stage-review.json', review)
    return review


def cpu_allocation(requested=12):
    need(type(requested) is int and requested > 0, 'positive CPU thread request')
    limits = [os.cpu_count() or 1]
    if hasattr(os, 'sched_getaffinity'):
        limits.append(len(os.sched_getaffinity(0)))
    for path, period_path in ((Path('/sys/fs/cgroup/cpu.max'), None),
                               (Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'), Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us'))):
        if path.is_file():
            parts = path.read_text().strip().split()
            if parts and parts[0] not in ('max', '-1'):
                period = int(period_path.read_text()) if period_path else int(parts[1])
                limits.append(max(1, int(int(parts[0]) / period)))
    effective = max(1, min(limits))
    return {'requested': requested, 'effective_allocation': effective, 'selected_threads': min(requested, effective),
            'opencv_threads': 1, 'torch_interop_threads': 1, 'data_loader_workers': 0}


def activate(assets, work, target, opencv_wheel=None):
    """Fresh child only; Linux offline installation is limited to engineering T4."""
    assets, work = Path(assets), Path(work)
    need(not any(name in sys.modules for name in ('torch', 'timm', 'cv2', 'pydicom', 'raptor_common')),
         'runtime activation requires a fresh child process')
    manifest = validate_assets(assets)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      KNEE_WORKSPACE_ROOT=str(assets.resolve()))
    if target == 't4-engineering':
        need(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 12), 'offline wheels require Linux CPython3.12')
        need(opencv_wheel and sha(opencv_wheel) == OPENCV_SHA, 'exact offline OpenCV wheel required')
        environment = work / 'environment'
        need(not environment.exists(), 'fresh isolated dependency directory required')
        wheels = [str(assets / w['path']) for w in manifest['wheels']] + [str(opencv_wheel)]
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(environment), *wheels], check=True)
        sys.path.insert(0, str(environment))
    else:
        need(target == 'local-reference', 'unknown runtime target')
    sys.path.insert(0, str(assets / 'training/src'))
    import timm
    need(timm.__version__ == '1.0.19', 'timm version differs from fixed architecture')
    if target == 't4-engineering':
        import cv2, pydicom, gdcm
        need((cv2.__version__, pydicom.__version__, gdcm.Version.GetVersion()) == ('4.12.0', '3.0.2', '3.2.6'),
             'offline decoder dependencies not respected')
    return manifest


def resolve_ids(assets, cohort, competition=None):
    if cohort == 'engineering100':
        return valid_ids(read(Path(assets) / 'engineering100.json')['ids'])
    need(cohort == 'public3' and competition is not None, 'explicit cohort/competition required')
    with (Path(competition) / 'test.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream); need(UID in rows.fieldnames, 'test UID field missing')
        ids = valid_ids(row[UID] for row in rows)
    need(len(ids) == 3, 'V1 public3 check does not authorize hidden/full-cohort execution')
    return ids


class ObservedCorpus:
    """Record uint8 input and center identities without retaining extra pixels."""
    def __init__(self, corpus):
        self.base = corpus
        self.ids, self.contract, self.depth, self.spec = corpus.ids, corpus.contract, corpus.depth, corpus.spec
        self.records = {}

    def __contains__(self, uid):
        return uid in self.base

    def mask(self, uid):
        import numpy as np
        from raptor_common import select_centers
        value = self.base.mask(uid)
        need(value.dtype == np.uint8 and value.shape == (64,), 'mask contract changed')
        centers = select_centers(value, self.depth, INPUT['k_eval'], self.spec, policy=INPUT['window_policy'])
        self.records[uid] = {'uid': uid, 'mask_sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                             'selected_centers': centers, 'empty_windows': sum(c == -1 for c in centers)}
        return value

    def volume(self, uid):
        import numpy as np
        value = self.base.volume(uid)
        need(value.dtype == np.uint8 and value.shape == (64, 336, 336), 'volume contract changed')
        self.records[uid]['volume_sha256'] = hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()
        return value


def worker(args):
    started = time.monotonic(); assets, work = Path(args.assets), Path(args.work)
    need(sha(assets / MARKER) == args.manifest_sha256, 'requested asset manifest differs')
    manifest = activate(assets, work, args.target, args.opencv_wheel)
    import numpy as np
    import torch
    from raptor_common import Corpus, environment
    from raptor_ensemble import load_members, predict_ensemble, validate_members
    from raptor_runtime import configure_runtime
    allocation = cpu_allocation(args.cpu_threads)
    torch.set_num_threads(allocation['selected_threads']); torch.set_num_interop_threads(1)
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one visible CUDA device required')
    if args.target == 't4-engineering':
        need(torch.cuda.get_device_capability(0) == (7, 5) and 'T4' in torch.cuda.get_device_name(0), 'actual NVIDIA T4 required')
        import cv2
        cv2.setNumThreads(1)
    runtime = configure_runtime('fp16', 'cuda', policy='strict')
    ids = resolve_ids(assets, args.cohort, args.competition)
    members, _ = load_members(assets / MARKER)
    need(validate_members(members) == INPUT, 'loaded checkpoint input metadata differs')
    for member, expected in zip(members, expected_members()):
        need(member.metadata['arch'] == expected['arch'] and member.metadata['epoch'] + 1 == expected['epoch_1based']
             and member.metadata['tag'] == expected['training_run'], 'loaded model provenance differs')
    source = None
    if args.target == 'local-reference':
        need(args.cache_root and not args.opencv_wheel, 'local-reference uses explicit frozen cache; no Linux wheel install')
        need(sha(Path(args.cache_root) / 'manifest.json') == CACHE_MANIFEST_SHA, 'reference cache manifest differs')
        corpus = Corpus(args.cache_root)
        need(set(ids) <= set(map(str, corpus.ids)), 'reference cache lacks requested cohort')
    else:
        from raptor_dicom import DicomCorpus, DirectoryDicomSource
        competition = Path(args.competition)
        split = 'train' if args.cohort == 'engineering100' else 'test'
        # UID-only input prevents report/target columns from entering the runtime adapter.
        uid_csv = work / 'cohort.csv'
        with uid_csv.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream); writer.writerow([UID]); writer.writerows([[u] for u in ids])
        source = DirectoryDicomSource(competition / (split + '_series'))
        corpus = DicomCorpus(uid_csv, competition / (split + '_series.csv'), source, INPUT['contract'], split)
    try:
        observed = ObservedCorpus(corpus)
        mean, raw, inference = predict_ensemble(members, observed, ids, amp='fp16', device='cuda', batch=1,
                                               residency='resident', runtime_policy='strict')
        need(raw.dtype == np.float32 and mean.shape == (len(ids), 12), 'raw inference shape/dtype')
        need(inference['status'] == 'COMPLETE' and inference['fp32_fallbacks'] == 0 and inference['neutral_fills'] == 0,
             'incomplete/fallback inference')
        info = {'target': args.target, 'cohort': args.cohort, 'environment': environment(), 'runtime_policy': runtime,
                'allocation': allocation, 'elapsed_seconds_before_publish': time.monotonic() - started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'whole_process_rss_and_cgroup_peak': 'NOT_MEASURED_BY_V1; required before runtime release',
                'input_evidence': [observed.records[uid] for uid in ids],
                'corpus_manifest': corpus.manifest,
                'dicom_receipts': getattr(corpus, 'receipts', None),
                'cache_integrity_scope': 'Pinned manifest; per-study volume/mask hashes measured, not preasserted equal to original bytes',
                'parity_scope': 'Raw output producer only; local/T4 comparison not evaluated here'}
        return atomic_outputs(work, ids, raw, sha(assets / MARKER), inference, info)
    finally:
        if source is not None:
            source.close()


def run(args):
    """Bounded local child launcher; calling this function performs explicit GPU work."""
    assets, work = Path(args.assets).resolve(), Path(args.work).resolve()
    need(sha(assets / MARKER) == args.manifest_sha256, 'requested asset manifest differs')
    validate_assets(assets)
    need(not work.exists(), 'new work directory required; no overwrite/retry')
    need(0 < args.max_seconds <= 1800, 'engineering wall-time cap must be <=1800 seconds')
    if args.target == 't4-engineering':
        need(args.competition and args.opencv_wheel and not args.cache_root, 'T4 requires DICOM competition + pinned OpenCV wheel')
    else:
        need(args.cache_root and not args.opencv_wheel, 'local reference requires frozen cache only')
    ids = resolve_ids(assets, args.cohort, args.competition)
    work.mkdir(parents=True)
    allocation = cpu_allocation(args.cpu_threads)
    command = [sys.executable, str(assets / 'b_rn3f_serving.py'), '_worker', '--assets', str(assets), '--work', str(work),
               '--manifest-sha256', args.manifest_sha256,
               '--target', args.target, '--cohort', args.cohort, '--cpu-threads', str(allocation['selected_threads'])]
    for name in ('competition', 'cache_root', 'opencv_wheel'):
        if getattr(args, name):
            command += ['--' + name.replace('_', '-'), str(Path(getattr(args, name)).resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS=str(allocation['selected_threads']),
               MKL_NUM_THREADS=str(allocation['selected_threads']), OPENBLAS_NUM_THREADS=str(allocation['selected_threads']),
               PYTHONDONTWRITEBYTECODE='1')
    write_new(work / 'launch.json', {'command': command, 'max_seconds': args.max_seconds, 'allocation': allocation,
                                  'manifest_sha256': sha(assets / MARKER), 'automatic_retries': 0})
    started = time.monotonic(); result = {'schema': SCHEMA, 'status': 'RUNNING', 'submission_ready': False}
    with (work / 'run.log').open('x', encoding='utf-8') as log:
        process = None
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       start_new_session=(os.name != 'nt'),
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            result['pid'] = process.pid
            write_new(work / 'pid.json', {'pid': process.pid, 'started_at': datetime.now(timezone.utc).isoformat()})
            code = process.wait(timeout=args.max_seconds)
            need(code == 0, 'engineering worker failed; inspect run.log')
            validate_result(work, assets, ids, args.target, args.cohort)
            result.update(status='COMPLETE_ENGINEERING_OUTPUT_ONLY', worker_receipt_sha256=sha(work / 'result/receipt.json'))
        except BaseException as exc:
            if process is not None and process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=log, stderr=log, check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            result.update(status='FAILED', exception_type=type(exc).__name__, error=str(exc))
        finally:
            result['elapsed_seconds'] = time.monotonic() - started
            write_new(work / 'run-receipt.json', result)
    need(result['status'] == 'COMPLETE_ENGINEERING_OUTPUT_ONLY', result.get('error', 'worker did not complete'))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build'); b.add_argument('--root', default=str(ROOT)); b.add_argument('--out', required=True); b.add_argument('--root-decision', required=True)
    v = sub.add_parser('verify'); v.add_argument('--assets', required=True)
    for action in ('run', '_worker'):
        r = sub.add_parser(action)
        r.add_argument('--assets', required=True); r.add_argument('--work', required=True)
        r.add_argument('--manifest-sha256', required=True)
        r.add_argument('--target', choices=['local-reference', 't4-engineering'], required=True)
        r.add_argument('--cohort', choices=['engineering100', 'public3'], required=True)
        r.add_argument('--competition'); r.add_argument('--cache-root'); r.add_argument('--opencv-wheel')
        r.add_argument('--cpu-threads', type=int, default=12)
        if action == 'run': r.add_argument('--max-seconds', type=int, default=1800)
    return p


def main():
    args = parser().parse_args()
    if args.action == 'build':
        result = build(args.root, args.out, args.root_decision)
    elif args.action == 'verify':
        manifest = validate_assets(args.assets)
        result = {'status': 'ASSET_IDENTITIES_VALID_NO_INFERENCE', 'manifest_sha256': sha(Path(args.assets) / MARKER),
                  'validation_status': manifest['validation_status']}
    elif args.action == 'run': result = run(args)
    else:
        try:
            result = worker(args)
        except BaseException as exc:
            failure = {'schema': SCHEMA, 'status': 'FAILED', 'submission_ready': False,
                       'error': str(exc), 'exception_type': type(exc).__name__}
            if hasattr(exc, 'receipt'):
                failure['source_receipt'] = exc.receipt
            write_new(Path(args.work) / 'worker-failure.json', failure)
            raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
