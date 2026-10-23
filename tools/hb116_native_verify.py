"""Offline visible-study proof of the fused native reader/input contracts."""
import sys
from pathlib import Path
from tools import hb116_vnext_release as h


def failure_smoke(job, ns):
    """Real visible DICOM/preprocessing; tiny deterministic inference probes.

    These exercise the actual worker/journal/CSV path without rerunning the
    unchanged 31 trained models. They are not model parity or target timings.
    """
    import time
    import numpy as np
    import torch
    import os
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools import hb116_native_runtime as native
    from tools import exp169_release_v6 as v6
    import raptor_radimagenet, raptor_runtime, raptor_dicom
    reg = h.read(h.REG/'consults/astra3-registration-draft.json')
    entries = [next(e for e in job['entries'] if e.get('family') == family)
               for family in ('R50', 'C4', 'S')]
    entries.append(next(e for e in job['entries'] if e['id'] == 'EXP143v2'))
    identity = dict(version=h.VERSION, members={e['id']:e['sha256'] for e in entries})
    def model(*args):
        value = torch.nn.Linear(1, 1)
        if fault == 'resident_model_bytes':
            value.buffers = lambda: (_ for _ in ()).throw(RuntimeError('injected resident stats'))
        return value
    def infer(model, tensor, device, **kwargs):
        # Depend on every prepared element, with no random model state.
        value = torch.sigmoid(tensor.double().mean()).item()
        return np.full(12, value, np.float32)
    original_enter, original_decode = native.SharedPixels.__enter__, native.SharedPixels._decode
    original_normalise = raptor_radimagenet.normalise
    original_imagenet = raptor_runtime.normalize_input
    original_q_decode = raptor_dicom._decode
    original_to, original_dump = torch.Tensor.to, h.dump
    original_windows = ns['gpu_windows']
    results, baseline = {}, None
    for fault in ('normal', 'cache_enter', 'cache_decode', 'r50_normalize', 'c4_normalize',
                  'mock_cuda_0', 'mock_cuda_1', 'telemetry_all_0', 'telemetry_all_1',
                  'monotonic', 'timing_write', 'print', 'resident_model_bytes'):
        folder = h.SCRATCH/'v112-failure-smoke'/fault
        out = folder/'worker'
        out.mkdir(parents=True, exist_ok=True)
        (out/'predictions.jsonl').write_text('', encoding='utf-8')
        spec = dict(job, entries=entries, identity=identity, output=str(out))
        h.dump(folder/'job.json', spec)
        calls = {'enter':0, 'norm':0, 'imagenet':0}
        def enter(cache):
            calls['enter'] += 1
            value = original_enter(cache)
            if fault == 'cache_enter' and calls['enter'] == 1:
                raise OSError('injected after cache hooks installed')
            return value
        def decode(cache, raw):
            if fault == 'cache_decode' and calls['enter'] == 1:
                raise OSError('injected shared pixel decode')
            return original_decode(cache, raw)
        def normalise(value, kind):
            calls['norm'] += 1
            if fault == 'r50_normalize' and calls['norm'] == 1:
                raise ValueError('injected R50 normalization')
            return original_normalise(value, kind)
        def imagenet(value, kind):
            calls['imagenet'] += 1
            if fault == 'c4_normalize' and calls['imagenet'] == 1:
                raise ValueError('injected C4/S normalization')
            return original_imagenet(value, kind)
        mocked_cuda = fault.startswith(('mock_cuda', 'telemetry_all'))
        device = 'cuda:0' if mocked_cuda else 'cpu'
        smoke_ns = dict(ns)
        def raising(*args, **kwargs):
            raise RuntimeError('injected telemetry failure')
        with ExitStack() as patches, patch.object(h, 'q_model', model), \
             patch.object(h, 'anchor_setup', lambda *args:(smoke_ns, {'EXP143v2':model()}, None)), \
             patch.object(v6, 'infer', infer), \
             patch.object(native.SharedPixels, '__enter__', enter), \
             patch.object(native.SharedPixels, '_decode', decode), \
             patch.object(raptor_radimagenet, 'normalise', normalise), \
             patch.object(raptor_runtime, 'normalize_input', imagenet):
            if mocked_cuda:
                patches.enter_context(patch.dict(os.environ, CUDA_VISIBLE_DEVICES=fault[-1]))
                def cuda_call(name):
                    def call(value):
                        assert str(value) == 'cuda:0'
                        if fault.startswith('telemetry_all'):
                            raise RuntimeError('injected '+name)
                        return {'get_device_name':'MOCK T4 (CPU tensors)',
                            'get_device_properties':SimpleNamespace(total_memory=16*1024**3)}.get(name, 128)
                    return call
                for name in ('reset_peak_memory_stats','synchronize','get_device_name',
                             'max_memory_allocated','max_memory_reserved','get_device_properties'):
                    patches.enter_context(patch.object(torch.cuda, name, cuda_call(name)))
                def cpu_to(tensor, target, *args, **kwargs):
                    if str(target).startswith('cuda'):
                        assert str(target) == 'cuda:0'
                        target = 'cpu'
                    return original_to(tensor, target, *args, **kwargs)
                patches.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
                smoke_ns['gpu_windows'] = lambda vol, mask, k, size, dev, **kw: original_windows(
                    vol, mask, k, size, 'cpu', **kw)
            # Any unexpected CUDA initialization is a verification failure,
            # including from the real readers, regardless of the chosen case.
            patches.enter_context(patch.object(torch.cuda, '_lazy_init',
                side_effect=AssertionError('Hardware CUDA forbidden during CPU smoke')))
            if fault == 'monotonic':
                patches.enter_context(patch.object(native, 'time', SimpleNamespace(monotonic=raising)))
            if fault == 'print':
                patches.enter_context(patch.object(native, 'print', raising, create=True))
            if fault == 'timing_write':
                def dump(path, data):
                    if Path(path).name == 'timing.json': raising()
                    return original_dump(path, data)
                patches.enter_context(patch.object(h, 'dump', dump))
            native.worker(folder/'job.json', device)
        assert raptor_dicom._decode is original_q_decode
        values, errors = h.c.read_journal(out, job['ids'], list(identity['members']), identity)
        expected_bad = ([entries[0]['id']] if fault == 'r50_normalize' else
                        [entries[1]['id'], entries[2]['id']] if fault == 'c4_normalize' else [])
        for member, failures in errors.items():
            assert set(failures) == ({job['ids'][0]} if member in expected_bad else set())
            assert np.isfinite(values[member]).all()
            if baseline is not None:
                for i, uid in enumerate(job['ids']):
                    if uid not in failures:
                        np.testing.assert_array_equal(values[member][i], baseline[member][i])
            for i, uid in enumerate(job['ids']):
                if uid in failures:
                    values[member][i] = np.nan
        if baseline is None:
            baseline = {k:v.copy() for k,v in values.items()}
        timing = h.read(out/'timing.json') if fault != 'timing_write' else None
        if timing is not None:
            assert len(timing['details']) == 3
        if fault.startswith('telemetry_all') or fault in ('monotonic','timing_write','print','resident_model_bytes'):
            assert h.read(out/'telemetry-errors.json')
        if fault.startswith('cache'):
            assert timing['details'][job['ids'][0]]['shared_cache_error']
            assert all('shared_cache_error' not in timing['details'][uid] for uid in job['ids'][1:])
        runtime = folder/'runtime'
        h.dump(runtime/'build-config.json', dict(version=h.VERSION, candidate='A',
            constants=h.CONSTANTS['A'], rx='R', s43=True, stress=False))
        receipt = h.finish_run(runtime, folder/'export', reg, job['ids'], values,
                              {}, {}, 0., time.monotonic(), device='cpu-probe')
        ids, csv_values = h.c.v5q._read_csv(folder/'export/submission.csv')
        assert ids == job['ids'] and np.isfinite(csv_values).all()
        results[fault] = dict(status='PASS', studies=len(ids), failed_members_first_study=expected_bad,
            healthy_probe_outputs_bit_identical=True, csv_sha256=receipt['prediction_sha256'],
            execution='CPU tensors; CUDA APIs mocked' if mocked_cuda else 'CPU',
            telemetry_errors=h.read(out/'telemetry-errors.json') if (
                fault.startswith('telemetry_all') or fault in ('monotonic','timing_write','print','resident_model_bytes')) else [])
        print('REAL_CPU_FAILURE_SMOKE_PASS', fault, flush=True)
    h.dump(h.OUT/'validation/failure-smoke.json', dict(status='PASS', version=h.VERSION,
        runtime_sha256=h.sha(h.ROOT/'tools/hb116_native_runtime.py'),
        verifier_sha256=h.sha(Path(__file__)), cases=results,
        scope='Real DICOM, preprocessing, worker, journals and CSV; deterministic inference probes, no trained-model rerun'))
    return results


def verify():
    h.configure()
    import cv2
    import numpy as np
    from tools import exp169_release_v6 as v6
    from tools.hb116_native_runtime import SharedPixels, shared_q_inputs, prefetched
    cv2.setNumThreads(1)
    job = h.read(h.SCRATCH/'cpu-NATIVE/NATIVE/worker0/job.json')
    sys.path.insert(0, job['legacy'])
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_common import select_centers
    ns, _, _ = h.anchor_setup(job, [], 'cpu')
    ns.update(FR_HDR=1, FR_PX=1)
    reader = v6.anchor_reader(ns)
    rows = {}
    for row in job['rows']:
        rows.setdefault(row[h.UID], []).append(row)
    def corpus():
        return DicomCorpus.from_records(job['ids'], job['rows'], DirectoryDicomSource(job['images']))
    results = []
    for uid in job['ids']:
        old, new = corpus(), corpus()
        q_old, qm_old = old.volume(uid), old.mask(uid)
        a_old, am_old = reader(uid, rows, job['images'])
        with SharedPixels(reader) as cache:
            q_new, qm_new = new.volume(uid), new.mask(uid)
            a_new, am_new = reader(uid, rows, job['images'])
            counts = dict(unique_decoded_slices=cache.misses, shared_pixel_hits=cache.hits)
        for a,b in ((q_old,q_new),(qm_old,qm_new),(a_old,a_new),(am_old,am_new)):
            np.testing.assert_array_equal(a,b)
        centers = select_centers(qm_old,64,36,old.spec,'within-slot-valid')
        shared = shared_q_inputs(q_new,centers)
        for family in ('R50','C4','S'):
            np.testing.assert_array_equal(h.q_inputs(q_old,centers,family),shared[family])
        assert shared['C4'] is shared['S']
        # Repeat after an interrupted cache scope: restoration must be exact.
        pixels = SharedPixels(reader)
        original = pixels.module._decode
        try:
            with pixels:
                raise RuntimeError('injected visible-study prepare failure')
        except RuntimeError:
            pass
        assert pixels.module._decode is original and pixels.ctx.px is pixels.decoder
        np.testing.assert_array_equal(reader(uid,rows,job['images'])[0],a_old)
        assert counts['shared_pixel_hits'] > 0
        results.append(dict(study_index=len(results),bit_identical=True,**counts))
        print('VISIBLE_INPUT_PASS',len(results),counts,flush=True)
    # Real-study preparations survive a reader failure and continue to next UID.
    def fail_one(uid):
        try:
            if uid == job['ids'][0]:
                value = reader(uid, {}, job['images'])
                h.need(np.any(value[0]), 'injected empty study')
                return value
            return reader(uid, rows, job['images'])
        except Exception as exc:
            return {'error':type(exc).__name__}
    outputs=list(prefetched(job['ids'],fail_one))
    assert 'error' in outputs[0][1]
    assert all(np.any(value[0]) for _,value,_ in outputs[1:])
    failures = failure_smoke(job, ns)
    result=dict(status='PASS',studies=results,real_failure_recovery=True,
        failure_smoke_cases=list(failures),
        runtime_sha256=h.sha(h.ROOT/'tools/hb116_native_runtime.py'),
        contract='Q64/anchor96 volumes+masks and all K36 normalized inputs exactly equal; shared pixels once, separate frozen renders')
    h.dump(h.OUT/'validation/input-equivalence.json',result)
    return result


if __name__=='__main__':
    verify()
