"""HB116 v1.1 native study pipeline. S6 and frozen readers remain untouched.

One resident worker per GPU, one outstanding CPU study, two distinct decode
contracts (Q64 and anchor96), and shared immutable windows within each contract.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import inspect
import sys
import time
from threading import Lock


class Telemetry:
    """Diagnostics cannot remove a prediction; missing measurements fail admission.

    One instance is shared with the CPU producer. Coalesce repeated errors (a
    broken clock could otherwise emit one record per slice) and keep the last
    good clock value so failed measurements cannot produce negative durations.
    """
    def __init__(self):
        self.errors = []
        self.lock = Lock()
        self.last_time = 0.

    def call(self, name, function, *args, default=None, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            with self.lock:
                for row in self.errors:
                    if row['operation'] == name and row['error'] == error:
                        row['count'] += 1
                        break
                else:
                    self.errors.append(dict(operation=name, error=error, count=1))
            return default

    def now(self):
        value = self.call('monotonic', time.monotonic)
        with self.lock:
            if value is not None:
                self.last_time = max(self.last_time, value)
            return self.last_time

    def snapshot(self):
        with self.lock:
            return [dict(row) for row in self.errors]

    def write(self, dump, out, values):
        values['telemetry_errors'] = self.snapshot()
        self.call('timing.json', dump, out/'timing.json', values)
        # A failed timing write cannot persist its own error in that file.
        if self.snapshot():
            self.call('telemetry-errors.json', dump, out/'telemetry-errors.json', self.snapshot())
            self.call('print', print, 'TELEMETRY_ERRORS', self.snapshot(), flush=True)


def worker_device(device):
    """Use the same process-local index as v1.0.1's per-family workers.

    The launcher masks physical shard 0/1 with CUDA_VISIBLE_DEVICES and passes
    cuda:0 to BOTH children. Never interpret that mask as a logical index.
    Constructing torch.device does not initialize CUDA or query hardware.
    """
    import torch
    resolved = torch.device(device)
    if resolved.type == 'cuda':
        resolved = torch.device('cuda', 0 if resolved.index is None else resolved.index)
    return str(resolved)


class SharedPixels:
    """One decode per selected byte-identical slice, scoped to one study.

    Retain the anchor reader's frozen RawSlice/LUT path and materialize its
    exact float32 recipe for Q; compressed/unsupported slices use pydicom.
    Keep their sampling, header ordering and rendering completely separate.
    This adapter is private to the native child; S6 has its own process.
    """
    def __init__(self, reader):
        self.telemetry = Telemetry()
        import raptor_dicom
        self.module = raptor_dicom
        self.original = raptor_dicom._decode
        self.ctx = inspect.getclosurevars(reader).nonlocals['ctx']
        if not self.ctx.cfg.lut:
            raise ValueError('shared decoder requires pinned study_lut')
        self.decoder = self.ctx.px
        self.scan = self.decoder.__func__.__globals__['_scan_full']
        self.clear()

    def clear(self):
        self.cache = {}
        self.hits = self.misses = 0
        self.seconds = 0.
        self.failed = False

    def decode(self, raw):
        try:
            return self._decode(raw)
        except Exception:
            # Frozen readers may absorb decoder exceptions and return a partial
            # volume. Retry independently even when they do not raise to us.
            self.failed = True
            raise

    def _decode(self, raw):
        key = hashlib.sha256(raw).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key], key
        t = self.telemetry.now()
        value, hist = self.decoder(raw, self.scan(raw))
        self.seconds += self.telemetry.now() - t
        if value is None:
            raise ValueError('frozen DICOM decoder failed')
        self.cache[key] = (value, hist)
        self.misses += 1
        return (value, hist), key

    def q_decode(self, source, member):
        import numpy as np
        (value, _), digest = self.decode(source.read(member))
        if hasattr(value, 'to_float'):
            value = value.to_float()
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError('expected finite 2-D DICOM pixels')
        return value, digest

    def anchor_decode(self, raw, info):
        if raw is None:
            return None, None
        try:
            return self.decode(raw)[0]
        except Exception:
            return None, None

    def __enter__(self):
        self.module._decode = self.q_decode
        self.ctx.px = self.anchor_decode
        return self

    def __exit__(self, *args):
        self.module._decode = self.original
        self.ctx.px = self.decoder
        self.cache.clear()


def shared_q_inputs(vol, centers, errors=None):
    """Exactly the frozen center-plane interpolation and normalization order."""
    from raptor_radimagenet import center3_windows_from, normalise
    from raptor_runtime import normalize_input
    full = center3_windows_from(vol, centers, 224)
    result = {}
    failures = {} if errors is None else errors
    try:
        mixed = full.copy()
        mixed[1::2] = center3_windows_from(vol[:, 48:288, 48:288], centers[1::2], 224)
        mixed = normalize_input(mixed, 'imagenet')
        result.update(C4=mixed, S=mixed)
    except Exception as exc:
        failures['C4'] = failures['S'] = f'{type(exc).__name__}: {exc}'
    try:
        result['R50'] = normalise(full, 'radimagenet')
    except Exception as exc:
        failures['R50'] = f'{type(exc).__name__}: {exc}'
    return result


def prepare_study(uid, reader, prepare_uncached, telemetry=None):
    """Contain cache faults to a study; retry using original readers/corpus.

    Explicit finally also restores partially installed hooks if __enter__
    raises (a with statement would not call __exit__ in that case).
    """
    telemetry = telemetry if telemetry is not None else Telemetry()
    began = telemetry.now()
    try:
        pixels = SharedPixels(reader)
        pixels.telemetry = telemetry
        try:
            pixels.__enter__()
            result = prepare_uncached(uid)
            if pixels.failed or any(k in result['errors'] for k in ('Q', 'ANCHOR')):
                raise RuntimeError('shared decode/preparation failed')
            result['timing'].update(pixel_decode_seconds=pixels.seconds,
                unique_decoded_slices=pixels.misses, shared_pixel_hits=pixels.hits)
        finally:
            pixels.__exit__(*sys.exc_info())
        return result
    except Exception as exc:
        cache_error = f'{type(exc).__name__}: {exc}'
    retry = telemetry.now()
    try:
        result = prepare_uncached(uid, independent=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        result = {'errors': {'Q': error, 'ANCHOR': error}, 'timing': {}}
    result['timing'].update(shared_cache_error=cache_error,
        shared_cache_attempt_seconds=retry-began,
        independent_retry_seconds=telemetry.now()-retry)
    return result


def prefetched(ids, prepare, telemetry=None):
    """At most current + next study; never schedules a whole-cohort cache."""
    telemetry = telemetry if telemetry is not None else Telemetry()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='hb116-prepare') as pool:
        iterator = iter(ids)
        uid = next(iterator, None)
        future = pool.submit(prepare, uid) if uid is not None else None
        while future is not None:
            start = telemetry.now()
            value = future.result()
            wait = telemetry.now() - start
            following = next(iterator, None)
            future = pool.submit(prepare, following) if following is not None else None
            yield uid, value, wait
            uid = following


def worker(job_path, device):
    from tools import hb116_vnext_release as h
    h.configure()
    import cv2
    import numpy as np
    import torch
    device = worker_device(device)
    telemetry = Telemetry()
    from tools import exp169_release_v6 as v6
    cv2.setNumThreads(1)
    job = h.read(job_path)
    sys.path.insert(0, job['legacy'])
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_common import select_centers
    out = Path(job['output'])
    journal = h.c.Journal(out, job['ids'], job['identity'])
    began = telemetry.now()
    cuda = str(device).startswith('cuda')
    # Each GPU worker has one CPU producer. Serial native libraries keep the
    # two-shard target at <=4 runnable CPU threads, including reader workers.
    torch.set_num_threads(1 if cuda else 2)
    models, failed, model_setup = {}, {}, {}
    q_entries = [e for e in job['entries'] if e.get('family') in ('R50', 'C4', 'S')]
    anchor_entries = [e for e in job['entries'] if e not in q_entries]
    families = {e['id']: e.get('family', 'ANCHOR') for e in job['entries']}
    original_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                     torch.get_float32_matmul_precision())
    for e in q_entries:
        t = telemetry.now()
        try:
            models[e['id']] = h.q_model(e, device)
        except Exception as exc:
            failed[e['id']] = f'{type(exc).__name__}: {exc}'
        model_setup[e['id']] = telemetry.now() - t
    ns, anchor_models, _ = h.anchor_setup(job, anchor_entries, device, failed)
    # Override only pool widths; frozen numerical reader code is unchanged.
    ns.update(FR_HDR=1, FR_PX=1)
    reader = v6.anchor_reader(ns)
    models.update(anchor_models)
    # Model .to(cuda:0) lazily initializes the worker's CUDA context, just as
    # v1.0.1 did. An early reset can fail before initialization on Kaggle.
    # Resetting after model setup retains resident allocations in the peak.
    if cuda:
        telemetry.call('reset_peak_memory_stats', torch.cuda.reset_peak_memory_stats, device)
    corpus = DicomCorpus.from_records(job['ids'], job['rows'], DirectoryDicomSource(job['images']))
    rows = {}
    for row in job['rows']:
        rows.setdefault(row[h.UID], []).append(row)
    setup = telemetry.now() - began

    def prepare_uncached(uid, independent=False):
        result = {'errors': {}, 'timing': {}}
        t = telemetry.now()
        if q_entries:
            try:
                # Do not reuse a partially cached volume from a failed shared
                # attempt; each family contract now uses its original decoder.
                source = (DicomCorpus.from_records(job['ids'], job['rows'],
                    DirectoryDicomSource(job['images'])) if independent else corpus)
                vol, mask = source.volume(uid), source.mask(uid)
                h.need(np.count_nonzero(mask) >= 3 and np.any(vol), 'empty Q study')
                result['timing']['q_decode_render_seconds'] = telemetry.now() - t
                t = telemetry.now()
                centers = select_centers(mask, 64, 36, corpus.spec, 'within-slot-valid')
                result['q'] = shared_q_inputs(vol, centers, result['errors'])
                result['flags'] = np.asarray(h.c.a172.base.plane_flags(centers, corpus.spec), bool)
                result['timing']['q_windows_seconds'] = telemetry.now() - t
            except Exception as exc:
                result['errors']['Q'] = f'{type(exc).__name__}: {exc}'
        t = telemetry.now()
        if anchor_entries:
            try:
                vol, mask = reader(uid, rows, job['images'])
                h.need(np.count_nonzero(mask) >= 3 and np.any(vol), 'empty own/R study')
                result['anchor'] = (vol, mask)
            except Exception as exc:
                result['errors']['ANCHOR'] = f'{type(exc).__name__}: {exc}'
            result['timing']['anchor_decode_render_seconds'] = telemetry.now() - t
        return result

    def prepare(uid):
        # The producer alone owns this cache; its reader pool is serial.
        return prepare_study(uid, reader, prepare_uncached, telemetry)

    study_times, details = {}, {}
    for uid, prepared, wait in prefetched(job['ids'], prepare, telemetry):
        started = telemetry.now()
        metrics = dict(prepared['timing'], prefetch_wait_seconds=wait,
                       inference_seconds=0., device_prepare_seconds=0.)
        for member, error in failed.items():
            journal.record(uid, member, error='model load: ' + error, seconds=0.)
        inputs, input_errors = {}, dict(prepared['errors'])
        t = telemetry.now()
        for family in ('R50', 'C4'):
            if 'q' not in prepared:
                continue
            if family in input_errors:
                continue
            try:
                plain = prepared['q'][family]
                mirror = plain.copy()
                flags = prepared['flags']
                mirror[flags] = mirror[flags, :, :, ::-1]
                inputs[family] = tuple(torch.from_numpy(x).to(device) for x in (plain, mirror))
                if family == 'C4':
                    inputs['S'] = inputs['C4']
            except Exception as exc:
                input_errors[family] = f'{type(exc).__name__}: {exc}'
                if family == 'C4': input_errors['S'] = input_errors[family]
        if 'anchor' in prepared:
            try:
                vol, mask = prepared['anchor']
                inputs['ANCHOR'] = (ns['gpu_windows'](vol, mask, 94, 320, device),
                                    ns['gpu_windows'](vol, mask, 94, 320, device, mirror=True))
            except Exception as exc:
                input_errors['ANCHOR'] = f'{type(exc).__name__}: {exc}'
        if cuda: telemetry.call('synchronize', torch.cuda.synchronize, device)
        metrics['device_prepare_seconds'] = telemetry.now() - t
        for member, model in models.items():
            family = families[member]
            error = input_errors.get(family, input_errors.get('Q') if family != 'ANCHOR' else None)
            if error or family not in inputs:
                journal.record(uid, member, error=error or 'input unavailable', seconds=0.)
                continue
            t = telemetry.now()
            try:
                # The old S phase disabled TF32, while every other phase began
                # in a fresh process. Restore that exact family-specific state.
                if family == 'S':
                    torch.set_float32_matmul_precision('highest')
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                else:
                    torch.set_float32_matmul_precision(original_tf32[2])
                    torch.backends.cuda.matmul.allow_tf32 = original_tf32[0]
                    torch.backends.cudnn.allow_tf32 = original_tf32[1]
                plain, mirror = inputs[family]
                value = (v6.infer(model, plain, device, anchor=family == 'ANCHOR') +
                         v6.infer(model, mirror, device, anchor=family == 'ANCHOR')) * np.float32(.5)
                journal.record(uid, member, value, seconds=telemetry.now() - t)
                telemetry.call('print', print, 'MEMBER_DONE', member, uid, round(telemetry.now() - t, 3), flush=True)
            except Exception as exc:
                journal.record(uid, member, error=f'{type(exc).__name__}: {exc}', seconds=telemetry.now() - t)
            metrics['inference_seconds'] += telemetry.now() - t
        study_times[uid] = wait + telemetry.now() - started
        metrics['consumer_seconds'] = study_times[uid]
        details[uid] = metrics
        # Release study tensors before the next device preparation allocation.
        inputs.clear()
        plain = mirror = None
        telemetry.call('print', print, 'STUDY_DONE NATIVE', uid, round(study_times[uid], 3), flush=True)
        telemetry.write(h.dump, out, dict(family='NATIVE', device=device, setup_seconds=setup,
            load_failures=failed, model_setup_seconds=model_setup,
            gpu_name=telemetry.call('get_device_name', torch.cuda.get_device_name, device) if cuda else None,
            peak_allocated_bytes=telemetry.call('max_memory_allocated', torch.cuda.max_memory_allocated, device) if cuda else 0,
            peak_reserved_bytes=telemetry.call('max_memory_reserved', torch.cuda.max_memory_reserved, device) if cuda else 0,
            total_device_bytes=telemetry.call('get_device_properties', lambda: torch.cuda.get_device_properties(device).total_memory) if cuda else 0,
            resident_model_bytes=telemetry.call('resident_model_bytes', lambda: sum(t.numel()*t.element_size()
                for model in models.values() for t in list(model.parameters())+list(model.buffers()))),
            wall_seconds=telemetry.now()-began, study_seconds=study_times,
            details=details, threads=1 if cuda else 2, prefetch_depth=1, decode_contracts=['Q64', 'anchor96']))
