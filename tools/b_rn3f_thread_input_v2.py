"""Explicit producer-thread replay of frozen V1 input math; no GPU/model/scores."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform

SCHEMA = 'B_RN3F_THREAD_INPUT_V2'
PARITY_SHA = '298f11855c7e8e86007be612bf7e0831469da54d601aeedabab9f85a6259754b'
PASS = 'PASS_INPUT_AND_PRODUCER_LINK_ONLY'
THREAD_POLICY = {'source': 'completed_worker.runtime.allocation.selected_threads',
                 'interop_threads': 1, 'processes': 1, 'inference_mode': True,
                 'runtime_policy': 'strict', 'same_host_normalized_hash': 'exact',
                 'tolerance_change': False}

def need(ok, message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')

def load_math(path):
    need(sha(path) == PARITY_SHA, 'frozen math helper changed')
    spec = importlib.util.spec_from_file_location('frozen_thread_input_math', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper

def contract(helper, protocol_sha):
    return {'schema': SCHEMA, 'adapter_sha256': sha(__file__), 'math_helper_sha256': PARITY_SHA,
            'math_protocol_sha256': protocol_sha, 'source_sha256': helper.SOURCE_SHA,
            'serving_manifest_sha256': helper.SERVING_SHA, 'producer_sha256': helper.PRODUCER_SHA,
            'ordered_uids_sha256': helper.UID_SHA, 'studies': 100, 'limits': helper.LIMITS,
            'thread_policy': THREAD_POLICY}

def freeze(args, helper):
    helper.validate_protocol(args.protocol, args.protocol_sha256)
    value = {**contract(helper, args.protocol_sha256), 'frozen_at': datetime.now(timezone.utc).isoformat(),
             'status': 'FROZEN_BEFORE_V2_FULL100_REPLAY',
             'prior_observation': 'V1 CPU1 input100 passed math but failed producer12 binding; single first-UID 1/12 diagnostic established thread-sensitive CPU bilinear rounding.',
             'scope': 'Reuse unchanged V1 math/tolerances; replay actual completed producer CPU threads. No retrospective V1 PASS or tolerance relaxation.',
             'submission_ready': False}
    write_new(args.out, value); return value

def validate_contract(args, helper):
    helper.validate_protocol(args.protocol, args.protocol_sha256)
    need(sha(args.thread_protocol) == args.thread_protocol_sha256, 'thread protocol hash mismatch')
    value = read(args.thread_protocol)
    need(all(value.get(k) == v for k, v in contract(helper, args.protocol_sha256).items()), 'thread contract changed')

def completed_input_receipt(work, helper, ids, target):
    work = Path(work)
    need(not (work/'worker-failure.json').exists(), 'worker failure marker exists')
    parent, worker = read(work/'run-receipt.json'), read(work/'result/receipt.json')
    need(parent.get('schema') == 'B_RN3F_SERVING_V1' and parent.get('status') == 'COMPLETE_ENGINEERING_OUTPUT_ONLY', 'parent not complete')
    elapsed = parent.get('elapsed_seconds')
    need(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, 'invalid parent elapsed')
    need(parent.get('worker_receipt_sha256') == sha(work/'result/receipt.json'), 'parent worker hash mismatch')
    need(worker.get('schema') == 'B_RN3F_SERVING_V1' and worker.get('status') == 'COMPLETE'
         and worker.get('manifest_sha256') == helper.SERVING_SHA, 'worker incomplete or wrong manifest')
    need(worker.get('studies') == 100 and worker.get('ordered_uids_sha256') == helper.UID_SHA
         and worker.get('members') == helper.MEMBERS and worker.get('targets') == helper.TARGETS, 'worker cohort mismatch')
    rt, inf = worker['runtime'], worker['inference']
    need(rt.get('target') == target and rt.get('cohort') == 'engineering100', 'worker target mismatch')
    need(inf.get('status') == 'COMPLETE' and inf.get('studies') == 100 and inf.get('member_count') == 6
         and inf.get('input_hash_scheme') == 'normalized-float32-shape-v1'
         and inf.get('fp32_fallbacks') == 0 and inf.get('neutral_fills') == 0, 'worker inference incomplete')
    for values in [rt.get('input_evidence', []), inf.get('input_tensor_hashes', [])]:
        need(len(values) == 100 and [v.get('uid') for v in values] == ids, 'worker input records incomplete/reordered')
    need(all(isinstance(v.get('sha256'), str) and len(v['sha256']) == 64
             and all(c in '0123456789abcdef' for c in v['sha256']) for v in inf['input_tensor_hashes']), 'invalid input digest')
    allocation = rt['allocation']; threads = allocation.get('selected_threads')
    need(type(threads) is int and threads > 0 and type(allocation.get('effective_allocation')) is int
         and threads <= allocation['effective_allocation'] and allocation.get('torch_interop_threads') == 1,
         'invalid completed producer thread allocation')
    policy = rt['runtime_policy']
    need(policy.get('policy') == 'strict' and policy.get('backends', {}).get('deterministic_algorithms') is True,
         'producer strict execution contract missing')
    return worker, {'parent_sha256': sha(work/'run-receipt.json'), 'worker_sha256': sha(work/'result/receipt.json'),
                    'selected_threads': threads, 'allocation': allocation,
                    'verification_scope': 'completion/worker/input metadata only; prediction artifacts are not read'}

def set_threads(torch, threads):
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    need(torch.get_num_threads() == threads and torch.get_num_interop_threads() == 1, 'actual CPU thread mismatch')
    need(not torch.cuda.is_initialized(), 'CUDA initialized in CPU bridge')

def reconstruct(args, helper, worker, binding, ids):
    common, runtime = helper.passive_sources(args.assets)
    import torch
    threads = binding['selected_threads']
    # passive_sources deliberately initializes one thread. Override explicitly and record actual replay.
    os.environ.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    set_threads(torch, threads)
    policy = runtime.configure_runtime('none', 'cpu', policy='strict')
    mean, std = helper.constants()
    need(helper.array_sha(runtime.IMAGENET_MEAN) == helper.array_sha(mean)
         and helper.array_sha(runtime.IMAGENET_STD) == helper.array_sha(std), 'normalization constants changed')
    source = None
    if args.target == 'local-reference':
        need(args.cache_root and not args.competition, 'explicit frozen cache required')
        manifest = read(Path(args.assets)/'b-rn3f-assets.json')['reference_cache_manifest_sha256']
        need(sha(Path(args.cache_root)/'manifest.json') == manifest, 'cache manifest changed')
        corpus = common.Corpus(args.cache_root)
    else:
        need(args.competition and not args.cache_root, 'explicit competition DICOM source required')
        import csv, cv2, pydicom, gdcm
        need((cv2.__version__, pydicom.__version__, gdcm.Version.GetVersion()) == ('4.12.0','3.0.2','3.2.6'), 'fixed offline DICOM environment required')
        cv2.setNumThreads(1)
        from raptor_dicom import DirectoryDicomSource, DicomCorpus
        with (Path(args.competition)/'train_series.csv').open(encoding='utf-8-sig', newline='') as stream:
            series = list(csv.DictReader(stream))
        source = DirectoryDicomSource(Path(args.competition)/'train_series')
        corpus = DicomCorpus.from_records(ids, series, source, contract='raptor64_336_c140_s02', split='train')
    def check(uid, volume, mask):
        centers = common.select_centers(mask, 64, 24, {'slots': common.SLOTS64}, policy='within-slot-valid')
        windows = common.windows_from(volume, centers, 224)
        return helper.input_case(uid, volume, mask, centers, windows, runtime.normalize_input(windows, 'imagenet'))
    try:
        with torch.inference_mode():
            synthetic = [check(name, volume, mask) for name, volume, mask in helper.synthetics()]
            rows = [check(uid, corpus.volume(uid), corpus.mask(uid)) for uid in ids]
        need(not torch.cuda.is_initialized(), 'CUDA initialized during input replay')
        return {'schema': helper.SCHEMA, 'protocol_sha256': args.protocol_sha256, 'helper_sha256': PARITY_SHA,
                'serving_manifest_sha256': helper.SERVING_SHA, 'target': args.target,
                'observed_at': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                'rows': rows, 'synthetic': synthetic, 'pass': all(r['pass'] for r in rows+synthetic),
                'gpu_execution': False, 'cpu_workers': 1, 'cpu_threads': torch.get_num_threads(),
                'thread_adapter': {'schema': SCHEMA, 'adapter_sha256': sha(__file__),
                    'thread_protocol_sha256': args.thread_protocol_sha256,
                    'math_helper_sha256': PARITY_SHA, 'math_reuse': 'Original helper validates unchanged math; this adapter executes producer-thread replay.',
                    'passive_initial_threads': 1, 'actual_threads': torch.get_num_threads(),
                    'actual_interop_threads': torch.get_num_interop_threads(), 'inference_mode': True,
                    'producer_binding': binding, 'cpu_runtime_policy': policy},
                'reason': 'One process, recorded producer CPU threads; original V1 input evidence is preserved.'}
    finally:
        if source is not None: source.close()

def finish_report(args, helper, worker, ids, report):
    write_new(args.out, report)
    need(report.get('pass') is True, 'explicit input math PASS required')
    helper.validate_input_report(report, worker, ids, args.protocol_sha256, args.target)
    return report

def execute(args):
    need(not Path(args.out).exists() and not Path(args.acceptance_receipt).exists(), 'new output/receipt required')
    need(Path(args.out).resolve() != Path(args.acceptance_receipt).resolve(), 'output and receipt must differ')
    record = {'schema': SCHEMA, 'status': 'FAILED', 'adapter_sha256': sha(__file__),
              'observed_at': datetime.now(timezone.utc).isoformat(), 'submission_ready': False,
              'runtime_release_claim': False, 'gpu_execution': False}
    try:
        helper = load_math(args.parity_helper); validate_contract(args, helper)
        assets = Path(args.assets)
        need(sha(assets/'b-rn3f-assets.json') == helper.SERVING_SHA
             and sha(assets/'b_rn3f_serving.py') == helper.PRODUCER_SHA, 'producer assets changed')
        ids = read(assets/'engineering100.json')['ids']
        need(len(ids) == 100 and hashlib.sha256('\n'.join(ids).encode()).hexdigest() == helper.UID_SHA, 'wrong exact100 cohort')
        worker, binding = completed_input_receipt(args.work, helper, ids, args.target)
        record.update(producer_binding=binding, thread_protocol_sha256=args.thread_protocol_sha256,
                      math_protocol_sha256=args.protocol_sha256, math_helper_sha256=PARITY_SHA)
        report = reconstruct(args, helper, worker, binding, ids)
        finish_report(args, helper, worker, ids, report)
        # Reject receipt mutation across replay; completed inputs remain bound to the same producer.
        _, end_binding = completed_input_receipt(args.work, helper, ids, args.target)
        need(end_binding == binding, 'producer receipt changed during replay')
        record.update(status=PASS, exact_bound_studies=100, actual_threads=report['cpu_threads'])
    except Exception as exc:
        record.update(error=str(exc), exception_type=type(exc).__name__)
    if Path(args.out).exists(): record.update(input_report_sha256=sha(args.out), input_report=str(args.out))
    write_new(args.acceptance_receipt, record)
    return 0 if record['status'] == PASS else 1

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','input'])
    for name in ['parity-helper','protocol','protocol-sha256','out']: parser.add_argument('--'+name, required=True)
    for name in ['thread-protocol','thread-protocol-sha256','assets','work','target','cache-root','competition','acceptance-receipt']:
        parser.add_argument('--'+name)
    args = parser.parse_args(argv)
    if args.action == 'freeze': freeze(args, load_math(args.parity_helper)); return 0
    need(all(getattr(args,k) for k in ['thread_protocol','thread_protocol_sha256','assets','work','target','acceptance_receipt']), 'missing input arguments')
    need(args.target in ['local-reference','t4-engineering'], 'invalid target')
    return execute(args)

if __name__ == '__main__': raise SystemExit(main())
