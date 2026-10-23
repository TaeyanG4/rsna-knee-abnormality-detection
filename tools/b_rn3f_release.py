"""Versioned test-only B/RN runner. Import/build never load models or use CUDA.

Frozen engineering modules/assets are read-only. Only explicit remote worker/run
actions perform inference. Atomic competition CSV requires natural worker exit,
parent provenance, resource cleanup and exact full-cohort aggregation gates.
"""
from __future__ import annotations
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'B_RN3F_TEST_RELEASE_V1'
SERVING_SHA = '40f5a04dd0f770f05e9af5cf9a3827ccf5748c7989847c0778ba189fde40f80e'
PRODUCER_SHA = 'bd0fcaf4a2ddb9c8eb975a80f9ef1ae5250bd36d7b5ea6a4eefbf371014c7b5b'
TRIAL_SHA = 'e358076fa07b732ab69aeb778a5b6c593f3137e2911ec26dbfbb8f86a207f234'
RESOURCE_SHA = '15c0fe61d5d0b350557093ec38a1dc902b73e12c76328c4951934cc88f1da954'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
KERNEL = 'taeyangg4/knee-b-rn3f-release-v1'
DATASET = 'taeyangg4/knee-b-rn3f-v1-assets'
TIME_POLICY = {'producer100_seconds': 877.354403408, 'inflation': 1.3, 'fixed_seconds': 300,
               'screen_max_seconds': 30600, 'platform_seconds': 32400,
               'full_internal_seconds': 32100, 'public_up_to3_internal_seconds': 600,
               'public3_session_seconds': 900, 'finalization_reserve_seconds': 30,
               'automatic_retries': 0}
SYNTAXES = ['1.2.840.10008.1.2', '1.2.840.10008.1.2.1',
            *['1.2.840.10008.1.2.4.' + str(n) for n in (57, 70, 90, 91)]]
FIXTURE_HASHES = {
    'SC_rgb_jpeg_gdcm.dcm': 'a492ed4a120c51a076126a6021e8cab1acb0172da3d42c62843b2a34a8ddd252',
    'MR_small_jp2klossless.dcm': '4c0049e0355b560c8c846538d827afbdae5311b20fc5e5a93a3892e109bb140d',
    'JPEG2000.dcm': '5be539024e6803029a7b73c0f8e72e88d032e3a0bc05922c0c047344780aa8e1',
    'MR_small.dcm': '3f27d1c22f1a66e80d7bb7c911e8610fd0bb70325a76746a7adb1c0ddefcf2bb'}


def need(ok, message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def load(path, name, expected):
    need(sha(path) == expected, 'source identity changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def producer(assets):
    assets = Path(assets)
    need(sha(assets / 'b-rn3f-assets.json') == SERVING_SHA, 'frozen assets changed')
    return load(assets / 'b_rn3f_serving.py', 'unchanged_b_producer', PRODUCER_SHA)


def safe_ids(values):
    ids = list(values)
    need(bool(ids) and all(isinstance(v, str) and v and not any(c in v for c in ('/', '\\', ':', '\x00', '..')) for v in ids), 'invalid test UID')
    need(len(ids) == len(set(ids)), 'duplicate test UID'); return ids


def csv_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    need(len(rows) > 1 and len(rows[0]) == len(set(rows[0])), 'empty/duplicate CSV header')
    need(all(len(r) == len(rows[0]) for r in rows[1:]), 'CSV row width mismatch')
    return rows[0], rows[1:]


def test_contract(competition):
    """Canonical output follows the official sample order; no train/Report read."""
    competition = Path(competition)
    need((competition / 'test_series').is_dir(), 'test DICOM directory absent')
    th, tr = csv_table(competition / 'test.csv')
    need(UID in th, 'test UID column absent')
    test_ids = safe_ids(r[th.index(UID)] for r in tr)
    sh, sr = csv_table(competition / 'sample_submission.csv')
    need(sh == [UID, *TARGETS], 'official target schema/order differs')
    ids = safe_ids(r[0] for r in sr)
    need(set(ids) == set(test_ids), 'sample/test UID membership differs')
    need((competition / 'test_series.csv').is_file(), 'test series metadata absent')
    return {'ids': ids, 'test_ids': test_ids, 'columns': sh, 'studies': len(ids),
            'metadata_sha256': {n: sha(competition / n) for n in ['test.csv', 'test_series.csv', 'sample_submission.csv']},
            'scope': 'test metadata and official sample only; no labels/Reports/train data'}


def operational_plan(n):
    need(type(n) is int and n > 0, 'positive study count required')
    projected = 1.3 * (877.354403408 * n / 100) + 300
    need(projected <= 30600, 'N-dependent operational screen failed; no automatic workaround')
    return {'studies': n, 'projected_seconds': projected, 'screen_max_seconds': 30600,
            'internal_seconds': 600 if n <= 3 else 32100,
            'scope': 'fixed operational screen only; not hidden runtime proof', 'retries': 0}


def check_bundle(bundle, manifest_sha):
    bundle = Path(bundle); need(sha(bundle / 'release-manifest.json') == manifest_sha, 'release manifest changed')
    manifest = read(bundle / 'release-manifest.json')
    need(manifest.get('schema') == SCHEMA and manifest.get('time_policy') == TIME_POLICY
         and manifest.get('serving_sha256') == SERVING_SHA and manifest.get('producer_sha256') == PRODUCER_SHA,
         'release contract changed')
    for name, expected in manifest['files'].items():
        path = PurePosixPath(name)
        need(not path.is_absolute() and '..' not in path.parts and ':' not in name and '\\' not in name, 'unsafe bundle member')
        need(sha(bundle / name) == expected, 'release payload changed: ' + name)
    need(manifest['files']['b_rn3f_release.py'] == sha(__file__), 'executing release source differs')
    return manifest


def validate_codec(value):
    need(value.get('schema') == 'B_RN3F_FINAL_CODEC_V1' and value.get('status') == 'PASS'
         and value.get('pydicom_version') == '3.0.2' and value.get('gdcm_version') == '3.2.6', 'final codec environment failed')
    need([r['uid'] for r in value['decoders']] == SYNTAXES and all(r['available'] is True for r in value['decoders']), 'decoder capabilities incomplete')
    rows = value['fixtures']
    need([r['file'] for r in rows] == list(FIXTURE_HASHES)[:3] and all(r['decode'] == 'SUCCESS'
         and r['default_equals_fixed_reference'] is True and r['file_sha256'] == FIXTURE_HASHES[r['file']] for r in rows), 'actual fixture decode failed')
    need(rows[1]['matches_uncompressed_fixture'] is True, 'lossless fixture differs')
    return value


def codec_check(bundle, work):
    """Actual default decoder in the worker's installed final environment."""
    import numpy as np
    import pydicom, gdcm
    from pydicom.pixels import get_decoder
    bundle, work = Path(bundle), Path(work)
    reference = read(bundle / 'codec-reference.json')
    expected = {r['file']: r for r in reference['fixtures']}
    value = {'schema': 'B_RN3F_FINAL_CODEC_V1', 'status': 'FAILED', 'pydicom_version': pydicom.__version__,
             'gdcm_version': gdcm.Version.GetVersion(), 'fixtures': [],
             'scope': 'bundled compatibility fixtures, not hidden competition codec coverage',
             'decoders': [{'uid': u, 'available': get_decoder(u).is_available,
                           'plugins': list(get_decoder(u).available_plugins)} for u in SYNTAXES]}
    for name in list(FIXTURE_HASHES)[:3]:
        path = bundle / 'fixtures' / name
        need(sha(path) == FIXTURE_HASHES[name], 'codec fixture identity changed')
        ds = pydicom.dcmread(path); pixels = ds.pixel_array
        row = {'file': name, 'file_sha256': sha(path), 'syntax': str(ds.file_meta.TransferSyntaxUID),
               'shape': list(pixels.shape), 'dtype': str(pixels.dtype),
               'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(), 'decode': 'SUCCESS'}
        row['default_equals_fixed_reference'] = bool(np.isfinite(pixels).all() and pixels.size > 0
            and all(row[k] == expected[name][k] for k in ['syntax', 'shape', 'dtype', 'pixel_sha256']))
        if name == 'MR_small_jp2klossless.dcm':
            uncompressed = bundle / 'fixtures/MR_small.dcm'
            need(sha(uncompressed) == FIXTURE_HASHES['MR_small.dcm'], 'uncompressed fixture changed')
            row['matches_uncompressed_fixture'] = bool(np.array_equal(pixels, pydicom.dcmread(uncompressed).pixel_array))
        value['fixtures'].append(row)
    value['status'] = 'PASS'; validate_codec(value)
    write_new(work / 'codec.json', value); return value


def validate_resource(value, argv, worker_pid, timeout):
    need(value.get('schema') == 'process-resource-receipt-v1' and value.get('supervisor_sha256') == RESOURCE_SHA,
         'resource source mismatch')
    need(value.get('status') == 'COMPLETE' and value.get('child_exit_code') == value.get('wrapper_exit_code') == 0
         and value.get('child_exit_code_before_cleanup') == 0 and value.get('failure') is None, 'resource process failed')
    need(value.get('argv') == argv and value.get('child_identity', {}).get('pid') == worker_pid
         and worker_pid in {r['pid'] for r in value.get('observed_identities', [])}, 'resource/worker identity mismatch')
    need(value.get('sample_count', 0) > 0 and value.get('sampled_tree_rss_max_bytes', 0) > 0
         and value.get('failed_sample_attempts') == value.get('access_failure_count') == 0
         and not value.get('access_failures'), 'resource telemetry incomplete')
    cleanup = value.get('cleanup', {})
    need(set(cleanup) >= {'actions', 'errors', 'survivors'} and all(not cleanup[k] for k in ['actions', 'errors', 'survivors']), 'worker did not cleanly exit')
    need(value.get('retries') == 0 and value.get('timeout_seconds') == timeout
         and 0 <= value.get('elapsed_seconds', -1) < timeout, 'resource deadline mismatch')
    need(value.get('platform', '').startswith('linux') and value.get('cgroup_telemetry_complete') is True, 'Linux cgroup telemetry missing')
    snapshots = [value['cgroup_start'], value['cgroup_end']]
    for snapshot in snapshots:
        need(not snapshot['errors'] and snapshot['location']['version'] == 2, 'incomplete cgroup v2 snapshot')
        v = snapshot['values']
        need(type(v['memory.max']) is int and 0 < v['memory.max']
             and 0 < v['memory.peak'] <= v['memory.max'] and 0 <= v['memory.current'] <= v['memory.max'], 'memory accounting exceeds allocation')
        need(all(type(v['memory.events'].get(k)) is int and v['memory.events'][k] == 0
                 for k in ['max', 'oom', 'oom_kill']), 'memory limit or OOM event observed')
    return value


def worker(args):
    started = time.monotonic(); bundle, assets, work = map(Path, (args.bundle, args.assets, args.work))
    check_bundle(bundle, args.manifest_sha256)
    context = read(work / 'launch.json'); contract = test_contract(args.competition)
    need(contract == context['test_contract'] and time.monotonic() < context['deadline_monotonic'], 'test metadata/deadline changed')
    p = producer(assets)
    p.activate(assets, work, 't4-engineering', args.opencv_wheel)
    codec_check(bundle, work)
    import numpy as np
    import torch
    from raptor_common import environment
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_ensemble import load_members, predict_ensemble, validate_members
    from raptor_runtime import configure_runtime
    allocation = p.cpu_allocation(12)
    torch.set_num_threads(allocation['selected_threads']); torch.set_num_interop_threads(1)
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1
         and torch.cuda.get_device_capability(0) == (7, 5) and 'T4' in torch.cuda.get_device_name(0), 'one actual T4 required')
    import cv2
    cv2.setNumThreads(1)
    runtime = configure_runtime('fp16', 'cuda', policy='strict')
    members, _ = load_members(assets / 'b-rn3f-assets.json')
    need(validate_members(members) == p.INPUT, 'frozen member input differs')
    for member, expected in zip(members, p.expected_members()):
        need(member.metadata['arch'] == expected['arch'] and member.metadata['epoch'] + 1 == expected['epoch_1based']
             and member.metadata['tag'] == expected['training_run'], 'frozen member provenance differs')
    competition = Path(args.competition); ids = contract['ids']
    uid_csv = work / 'cohort.csv'
    with uid_csv.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow([UID]); writer.writerows([[u] for u in ids])
    with DirectoryDicomSource(competition / 'test_series') as source:
        corpus = DicomCorpus(uid_csv, competition / 'test_series.csv', source, p.INPUT['contract'], 'test')
        observed = p.ObservedCorpus(corpus)
        mean, raw, inference = predict_ensemble(members, observed, ids, amp='fp16', device='cuda', batch=1,
                                               residency='resident', runtime_policy='strict')
        need(raw.dtype == np.float32 and mean.shape == (len(ids), 12), 'raw output shape/dtype changed')
        need(test_contract(competition) == contract and time.monotonic() < context['deadline_monotonic'], 'metadata changed/deadline expired')
        info = {'target': 't4-engineering', 'cohort': 'test-full', 'pid': os.getpid(),
                'release_schema': SCHEMA, 'release_manifest_sha256': args.manifest_sha256, 'release_source_sha256': sha(__file__),
                'environment': environment(), 'runtime_policy': runtime, 'allocation': allocation,
                'input_evidence': [observed.records[u] for u in ids], 'corpus_manifest': corpus.manifest,
                'dicom_receipts': corpus.receipts, 'test_contract': contract,
                'elapsed_seconds_before_publish': time.monotonic() - started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'codec_sha256': sha(work / 'codec.json'), 'launch_sha256': sha(work / 'launch.json'),
                'scope': 'unreleased full test cohort outputs; outer parent/resource gate still required'}
        return p.atomic_outputs(work, ids, raw, SERVING_SHA, inference, info)


def checked_outputs(work, assets, context, manifest_sha):
    work = Path(work); p = producer(assets)
    need(not (work / 'worker-failure.json').exists(), 'worker failure marker exists')
    receipt = p.validate_result(work, assets, context['test_contract']['ids'], 't4-engineering', 'test-full')
    rt = receipt['runtime']
    need(rt['release_schema'] == SCHEMA and rt['release_manifest_sha256'] == manifest_sha
         and rt['release_source_sha256'] == sha(__file__) and rt['test_contract'] == context['test_contract']
         and rt['launch_sha256'] == sha(work / 'launch.json') and rt['codec_sha256'] == sha(work / 'codec.json'), 'worker does not bind release context')
    validate_codec(read(work / 'codec.json'))
    need(len(rt['input_evidence']) == len(context['test_contract']['ids'])
         and [r['uid'] for r in rt['input_evidence']] == context['test_contract']['ids'], 'input evidence incomplete')
    return receipt


def publish_final(work, assets, context, manifest_sha, destination, elapsed, archive_info=None):
    """No final CSV is created until the entire owned tree has exited naturally."""
    work, destination = Path(work), Path(destination)
    need(not destination.exists(), 'existing final submission is preserved')
    need(math.isfinite(elapsed) and 0 <= elapsed < context['operational_plan']['internal_seconds'], 'whole internal deadline expired')
    receipt = checked_outputs(work, assets, context, manifest_sha)
    parent = read(work / 'parent-receipt.json')
    need(parent.get('schema') == SCHEMA and parent.get('status') == 'PARENT_COMPLETE_RESOURCE_CLEAN'
         and parent.get('worker_exit_code') == parent.get('supervisor_exit_code') == 0
         and parent.get('worker_sha256') == sha(work / 'result/receipt.json')
         and parent.get('resource_sha256') == sha(work / 'resource/receipt.json')
         and parent.get('launch_sha256') == sha(work / 'launch.json'), 'parent completion unbound')
    resource = validate_resource(read(work / 'resource/receipt.json'), context['worker_argv'], receipt['runtime']['pid'], context['worker_timeout_seconds'])
    need(test_contract(context['competition']) == context['test_contract'], 'test metadata changed before publication')
    temporary = destination.with_name(destination.name + '.pending')
    need(not temporary.exists(), 'existing pending submission is preserved')
    with temporary.open('xb') as stream:
        stream.write((work / 'result/predictions.csv').read_bytes()); stream.flush(); os.fsync(stream.fileno())
    p = producer(assets); p.validate_frame(temporary, context['test_contract']['ids'])
    need(sha(temporary) == receipt['output_sha256']['predictions.csv'], 'atomic copy differs')
    need(time.monotonic() < context['deadline_monotonic'], 'deadline expired before final publication')
    # Same-directory rename is the only operation that exposes the final filename.
    published = False
    try:
        os.replace(temporary, destination); published = True
        final_sha = sha(destination)
        need(final_sha == receipt['output_sha256']['predictions.csv'], 'final file differs after rename')
        need(time.monotonic() < context['deadline_monotonic'], 'deadline expired during final rename')
        result = {'schema': SCHEMA, 'status': 'PASS_TEST_COHORT_ARTIFACT_ONLY', 'observed_at': datetime.now(timezone.utc).isoformat(),
                  'release_manifest_sha256': manifest_sha, 'studies': context['test_contract']['studies'],
                  'submission_sha256': final_sha, 'submission_bytes': destination.stat().st_size,
                  'worker_sha256': sha(work / 'result/receipt.json'), 'parent_sha256': sha(work / 'parent-receipt.json'),
                  'resource_sha256': sha(work / 'resource/receipt.json'), 'codec_sha256': sha(work / 'codec.json'),
                  'operational_plan': context['operational_plan'],
                  'sampled_tree_rss_max_bytes': resource['sampled_tree_rss_max_bytes'], 'automatic_retries': 0,
                  'submission_authorized': False, 'competitive_quality_claim': False,
                  'scope': 'artifact for actual supplied test cohort only; public3 does not prove unseen full runtime'}
        result.update(archive_info or {})
        finish_record(work, result, destination, context)
        return result
    except Exception:
        # Remove only the artifact created by this call, and only while its exact
        # bytes still match our internal result. Pre-existing/external files stay.
        if published and destination.is_file() and sha(destination) == receipt['output_sha256']['predictions.csv']:
            destination.unlink()
        raise


def archive_evidence(work):
    work = Path(work); names = ['launch.json', 'parent-receipt.json', 'worker-failure.json', 'codec.json',
        'resource/receipt.json', 'resource/stdout.log', 'resource/stderr.log', 'resource/samples.jsonl',
        'result/receipt.json', 'result/members.npz', 'result/predictions.csv']
    with zipfile.ZipFile(work / 'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            if (work / name).is_file(): z.write(work / name, name)
    return {'evidence_zip_sha256': sha(work / 'evidence.zip'), 'evidence_zip_bytes': (work / 'evidence.zip').stat().st_size,
            'artifacts': {n: sha(work / n) for n in names if (work / n).is_file()}}


def finish_record(work, result, destination, context=None):
    """A failed final acknowledgement must not leave our submission exposed."""
    acknowledgement = Path(work) / 'result.json'
    pending = Path(work) / 'result.json.pending'
    acknowledged = False
    acknowledgement_sha = None
    try:
        need(not acknowledgement.exists(), 'existing acknowledgement is preserved')
        if context is not None:
            before_ack = time.monotonic()
            need(before_ack < context['deadline_monotonic'], 'deadline expired before final acknowledgement')
            result['elapsed_seconds'] = before_ack - context['started_monotonic']
            result['elapsed_scope'] = 'through pre-acknowledgement; actual post-ack clock checked and returned in notebook output'
        write_new(pending, result)
        acknowledgement_sha = sha(pending)
        os.replace(pending, acknowledgement); acknowledged = True
        if context is not None:
            after_ack = time.monotonic()
            need(after_ack < context['deadline_monotonic'], 'deadline expired during final acknowledgement')
            # This observation is returned/printed after the receipt write. It is
            # not claimed to be embedded retroactively in that same receipt.
            result['post_ack_elapsed_seconds'] = after_ack - context['started_monotonic']
            result['post_ack_deadline_pass'] = True
    except Exception:
        if acknowledged and acknowledgement.is_file() and sha(acknowledgement) == acknowledgement_sha:
            acknowledgement.unlink()
        destination = Path(destination)
        if result.get('status') == 'PASS_TEST_COHORT_ARTIFACT_ONLY' and destination.is_file() and sha(destination) == result['submission_sha256']:
            destination.unlink()
        raise


def run(args):
    bundle, assets, work = map(Path, (args.bundle, args.assets, args.work))
    check_bundle(bundle, args.manifest_sha256)
    need(not work.exists() and not Path(args.out).exists(), 'fresh work/final paths required')
    work.mkdir(parents=True)
    result = {'schema': SCHEMA, 'status': 'FAILED', 'submission_authorized': False, 'competitive_quality_claim': False}
    archive_info = None
    try:
        started = args.started_monotonic if args.started_monotonic is not None else time.monotonic()
        need(math.isfinite(started) and 0 <= time.monotonic() - started < 600, 'invalid bootstrap start time')
        p = producer(assets); p.validate_assets(assets)
        contract = test_contract(args.competition); plan = operational_plan(contract['studies'])
        deadline = started + plan['internal_seconds']; timeout = deadline - time.monotonic() - 30
        need(timeout > 0, 'no owned-worker budget remains')
        command = [sys.executable, str(bundle / 'b_rn3f_release.py'), '_worker', '--bundle', str(bundle),
                   '--manifest-sha256', args.manifest_sha256, '--assets', str(assets), '--work', str(work),
                   '--competition', str(args.competition), '--opencv-wheel', str(args.opencv_wheel)]
        context = {'schema': SCHEMA, 'release_manifest_sha256': args.manifest_sha256, 'test_contract': contract,
                   'operational_plan': plan, 'competition': str(args.competition), 'worker_argv': command,
                   'worker_timeout_seconds': timeout, 'deadline_monotonic': deadline, 'started_monotonic': started,
                   'automatic_retries': 0}
        write_new(work / 'launch.json', context)
        allocation = p.cpu_allocation(12); threads = str(allocation['selected_threads'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
        code = subprocess.run([sys.executable, str(bundle / 'process_resource_receipt.py'), '--out', str(work / 'resource'),
                               '--timeout', str(timeout), '--', *command], env=env).returncode
        need(code == 0, 'resource supervisor failed')
        receipt = checked_outputs(work, assets, context, args.manifest_sha256)
        resource = validate_resource(read(work / 'resource/receipt.json'), command, receipt['runtime']['pid'], timeout)
        parent = {'schema': SCHEMA, 'status': 'PARENT_COMPLETE_RESOURCE_CLEAN', 'supervisor_exit_code': code,
                  'worker_exit_code': resource['child_exit_code'], 'worker_sha256': sha(work / 'result/receipt.json'),
                  'resource_sha256': sha(work / 'resource/receipt.json'), 'launch_sha256': sha(work / 'launch.json'),
                  'elapsed_seconds': time.monotonic() - started}
        write_new(work / 'parent-receipt.json', parent)
        archive_info = archive_evidence(work)
        return publish_final(work, assets, context, args.manifest_sha256, args.out, time.monotonic() - started, archive_info)
    except Exception as exc:
        result.update(status='FAILED', error=str(exc), exception_type=type(exc).__name__)
    if archive_info is None: archive_info = archive_evidence(work)
    result.update(archive_info)
    finish_record(work, result, args.out); return result


def find_test_competition():
    paths = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
             Path('/kaggle/input/rsna-knee-abnormality-detection')]
    found = [p for p in paths if all((p / n).is_file() for n in ['test.csv', 'test_series.csv', 'sample_submission.csv'])
             and (p / 'test_series').is_dir()]
    need(len(found) == 1, 'test-only competition mount absent/ambiguous'); return found[0]


def build(args):
    """Small offline payload only; existing weights dataset remains unchanged."""
    root, out = Path(args.root), Path(args.out)
    need(not out.exists(), 'new release stage required')
    trial = load(root / 'tools/b_rn3f_t4_trial.py', 'frozen_trial_packaging', TRIAL_SHA)
    evidence = root / 'docs/handoffs/evidence/2026-10'
    previous = evidence / '2026-10-04-b-rn3f-t4-trial-v1'
    decision_path = evidence / '2026-10-04-b-rn3f-release-v1/root-cpu-decision.json'
    timeout_decision_path = evidence / '2026-10-04-b-rn3f-release-v1/root-timeout-decision.json'
    decision = read(decision_path)
    need(decision['status'] == 'AUTHORIZE_BOUNDED_CPU_IMPLEMENTATION_ONLY' and decision['automatic_retries'] == 0,
         'root CPU decision missing')
    need(read(timeout_decision_path)['status'] == 'PUBLIC3_QUALIFICATION_ONLY_KEEP_PLATFORM900',
         'root public-only timeout decision missing')
    rd = decision['runtime_release_design']
    need(rd['full_internal_deadline_seconds'] == 32100 and rd['public_N3_internal_seconds'] == 600
         and rd['platform_seconds'] == 32400 and rd['reserve_seconds'] == 1800, 'root time policy differs')
    need(read(previous / 'root-result-review.json')['status'] == 'ROOT_ENGINEERING_100_NUMERICAL_RESOURCE_PASS'
         and read(previous / 'independent-result-review.json')['status'] == 'PASS_ENGINEERING100_NUMERICAL_AND_RESOURCE_ONLY',
         'actual engineering100 review missing')
    assets = Path(args.assets); p = producer(assets)
    codec = root / '.local/d01-codec-staging/20261003-v1/preflight-v2/output/d01-codec-check.json'
    need(sha(codec) == 'eb26515f89b9be3d7f6f33c9667cbe6e079227b4f7b67ee078a3581f087ac8d1', 'codec reference changed')
    bundle = out / 'bundle'; bundle.mkdir(parents=True)
    files = {'b_rn3f_release.py': Path(__file__), 'b_rn3f_t4_trial.py': root / 'tools/b_rn3f_t4_trial.py',
             'process_resource_receipt.py': root / 'tools/process_resource_receipt.py', 'root-cpu-decision.json': decision_path,
             'root-timeout-decision.json': timeout_decision_path,
             'engineering-root-review.json': previous / 'root-result-review.json',
             'engineering-independent-review.json': previous / 'independent-result-review.json', 'codec-reference.json': codec}
    need(sha(files['process_resource_receipt.py']) == RESOURCE_SHA, 'resource source changed')
    for name, source in files.items(): shutil.copyfile(source, bundle / name)
    wheel = assets / 'wheels/pydicom-3.0.2-py3-none-any.whl'
    need(sha(wheel) == p.WHEEL_SHA[wheel.name], 'fixture wheel changed')
    (bundle / 'fixtures').mkdir()
    with zipfile.ZipFile(wheel) as z:
        for name, expected in FIXTURE_HASHES.items():
            data = z.read('pydicom/data/test_files/' + name)
            need(hashlib.sha256(data).hexdigest() == expected, 'bundled fixture identity differs')
            (bundle / 'fixtures' / name).write_bytes(data)
    manifest = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
                'serving_sha256': SERVING_SHA, 'producer_sha256': PRODUCER_SHA, 'time_policy': TIME_POLICY,
                'policy': p.POLICY, 'policy_sha256': p.POLICY_SHA, 'original_assets_dataset': DATASET,
                'files': {f.relative_to(bundle).as_posix(): sha(f) for f in sorted(bundle.rglob('*')) if f.is_file()},
                'fixture_scope': 'actual default .70/.90/.91 decode and lossless pair; .57 availability only; not all MRI formats',
                'scope': 'same test-only code for public3 and dynamic full test; no performance or submission authorization',
                'root_amendment': '2026-10-04 root message: N<=3 uses600s; full32100s; first notebook cell clock',
                'timeout_contract': {'platform_request_seconds_actual': 900, 'public_internal_seconds': 600,
                                     'future_intended_full_platform_seconds': 32400, 'hidden_scoring_timeout': 'UNKNOWN',
                                     'full_readiness': 'HOLD', 'scope': 'public3 qualification only; no submission readiness claim'},
                'train_data_runtime_reads': 0, 'submission_authorized': False}
    write_new(bundle / 'release-manifest.json', manifest)
    digest = sha(bundle / 'release-manifest.json'); check_bundle(bundle, digest)
    raw = {f.relative_to(bundle).as_posix(): f.read_bytes() for f in bundle.rglob('*') if f.is_file()}
    blob = trial.pack_files(raw); blob_sha = hashlib.sha256(blob).hexdigest()
    bootstrap = '''import time
started=time.monotonic()
from pathlib import Path, PurePosixPath
import base64,hashlib,importlib.util,io,json,lzma,os,sys,tarfile
'''+inspect.getsource(trial.unpack_files)+f'''
blob=base64.b85decode({base64.b85encode(blob).decode()!r})
assert hashlib.sha256(blob).hexdigest()=={blob_sha!r}
base=Path('/kaggle/working/b-rn3f-release-v1')
unpack_files(blob,base/'bundle')
spec=importlib.util.spec_from_file_location('release',base/'bundle/b_rn3f_release.py')
release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)
trial=release.load(base/'bundle/b_rn3f_t4_trial.py','frozen_mount',release.TRIAL_SHA)
assets=trial.find_mount('b-rn3f-assets.json',release.SERVING_SHA)
opencv=trial.find_mount('opencv_python_headless-4.12.0.88-*.whl',trial.OPENCV_SHA,return_file=True,dataset='mattiaangeli/opencv-python-headless-4120088-x86')
competition=release.find_test_competition()
from types import SimpleNamespace
args=SimpleNamespace(bundle=str(base/'bundle'),assets=str(assets),work=str(base/'work'),competition=str(competition),opencv_wheel=str(opencv),manifest_sha256={digest!r},out='/kaggle/working/submission.csv',started_monotonic=started)
result=release.run(args)
print(json.dumps(result,indent=2))
assert result['status']=='PASS_TEST_COHORT_ARTIFACT_ONLY'
assert release.sha('/kaggle/working/submission.csv')==result['submission_sha256']
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 4,
                'metadata': {'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': bootstrap.splitlines(keepends=True)}]}
    size = len(trial.sdk_text(notebook).encode('utf-8')); need(size < 1_000_000, 'release notebook exceeds1MB')
    directory = out / 'notebook'; directory.mkdir()
    write_new(directory / 'release.ipynb', notebook)
    meta = read(root / '.local/b-rn3f-t4-staging/20261004-v1/notebook/kernel-metadata.json')
    meta.update(id=KERNEL, title=KERNEL.split('/')[1], code_file='release.ipynb', is_private=True,
                enable_gpu=True, enable_tpu=False, enable_internet=False,
                dataset_sources=[DATASET, 'mattiaangeli/opencv-python-headless-4120088-x86'],
                competition_sources=['rsna-knee-abnormality-detection'], kernel_sources=[], model_sources=[])
    write_new(directory / 'kernel-metadata.json', meta)
    review = {'schema': SCHEMA, 'status': 'CPU_PACKAGED_NOT_PUBLIC3_EXECUTED', 'manifest_sha256': digest,
              'notebook_sha256': sha(directory / 'release.ipynb'), 'metadata_sha256': sha(directory / 'kernel-metadata.json'),
              'script_body_bytes': size, 'payload_raw_bytes': sum(map(len, raw.values())), 'payload_lzma_bytes': len(blob),
              'payload_lzma_sha256': blob_sha, 'fixture_bytes': sum((bundle / 'fixtures' / n).stat().st_size for n in FIXTURE_HASHES),
              'weight_upload_bytes': 0, 'network_calls': 0, 'gpu_execution': False,
              'platform_request_seconds_actual': 900, 'future_intended_full_platform_seconds': 32400,
              'hidden_scoring_timeout': 'UNKNOWN', 'full_readiness': 'HOLD', 'scope': 'public3 qualification only',
              'push_argv': [str(root / '.venv/Scripts/kaggle.exe'), 'kernels', 'push', '-p', str(directory), '--timeout', '900'],
              'first_output': 'b-rn3f-release-v1/work/result.json', 'evidence_output': 'b-rn3f-release-v1/work/evidence.zip',
              'competition_output': 'submission.csv', 'submission_authorized': False}
    write_new(out / 'stage-review.json', review); return review


def sdk_check(stage):
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    stage = Path(stage); bundle = stage / 'bundle'
    trial = load(bundle / 'b_rn3f_t4_trial.py', 'release_sdk_format', TRIAL_SHA)
    meta = read(stage / 'notebook/kernel-metadata.json')
    text = trial.sdk_text(read(stage / 'notebook' / meta['code_file']))
    req = ApiSaveKernelRequest()
    for field, key in [('slug', 'id'), ('new_title', 'title'), ('language', 'language'), ('kernel_type', 'kernel_type'),
                       ('is_private', 'is_private'), ('enable_gpu', 'enable_gpu'), ('enable_tpu', 'enable_tpu'),
                       ('enable_internet', 'enable_internet'), ('dataset_data_sources', 'dataset_sources'),
                       ('competition_data_sources', 'competition_sources'), ('kernel_data_sources', 'kernel_sources'),
                       ('model_data_sources', 'model_sources'), ('docker_image', 'docker_image'), ('machine_shape', 'machine_shape')]:
        setattr(req, field, meta[key])
    req.text = text; req.category_ids = meta.get('keywords', []); req.session_timeout_seconds = 900
    encoded = req.to_json().encode('utf-8')
    need(max(len(encoded), len(text.encode('utf-8'))) < 1_000_000, 'actual SDK payload exceeds1MB')
    value = {'status': 'CPU_SDK_PAYLOAD_SIZE_PASS', 'actual_sdk_request_bytes': len(encoded),
             'script_body_bytes': len(text.encode('utf-8')), 'request_sha256': hashlib.sha256(encoded).hexdigest(),
             'session_timeout_seconds': 900, 'credential_reads': 0, 'network_calls': 0}
    write_new(stage / 'sdk-review.json', value); return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build'); b.add_argument('--root', default=str(ROOT)); b.add_argument('--assets', required=True); b.add_argument('--out', required=True)
    s = sub.add_parser('sdk-check'); s.add_argument('--stage', required=True)
    for name in ['run', '_worker']:
        q = sub.add_parser(name)
        for field in ['bundle', 'manifest-sha256', 'assets', 'work', 'competition', 'opencv-wheel']: q.add_argument('--' + field, required=True)
        if name == 'run': q.add_argument('--out', required=True); q.add_argument('--started-monotonic', type=float)
    args = parser.parse_args()
    if args.action == 'build': result = build(args)
    elif args.action == 'sdk-check': result = sdk_check(args.stage)
    elif args.action == 'run':
        result = run(args)
        print(json.dumps(result, indent=2)); return 0 if result['status'] == 'PASS_TEST_COHORT_ARTIFACT_ONLY' else 1
    else:
        try: result = worker(args)
        except BaseException as exc:
            write_new(Path(args.work) / 'worker-failure.json', {'schema': SCHEMA, 'status': 'FAILED',
                      'error': str(exc), 'exception_type': type(exc).__name__})
            raise
    print(json.dumps(result, indent=2)); return 0


if __name__ == '__main__': raise SystemExit(main())
