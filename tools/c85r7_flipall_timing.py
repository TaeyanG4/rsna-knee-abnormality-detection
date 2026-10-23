"""H-B-153 local timing v2.1.3; never embedded in the release.

Changelog 2.1.3: lossless sampler-thread VRAM guard and fail-closed sampler
health; shared hb149 monitor unchanged. Backup: .local/backups/hb153-v213/.

Changelog 2.1.2: no wall deadline; dual-signal idle watchdog; measured 12 GiB
plus desktop/headroom admission and live device guard. Backup: .local/backups/hb153-v212/.

Changelog 2.1.1: actual serving CPU admission, largest admissible N4,
workers=2, no total-window cap. Backup: .local/backups/hb153-v211/.

Changelog 2.1.0: ten-minute N4 measured calibration and receipt-bound admit CLI.
Backup: .local/backups/hb153-v210/. Frozen release unchanged.

Changelog 2.0.0: authenticated N3/N32/hidden baseline; ABBA paired N32;
full-process CUDA/window journals; preparation watchdog and 75-minute deadline;
mandatory runtime/T4-memory gates. Backup: .local/backups/hb153-v2/.
Local diagnostic uses the serving workers=2 and sampled device-wide peaks.
This scheduling proxy is NOT Linux qualification or submission authorization.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_flipall_v1115 as b
from tools import hb145_output_io as io
from tools import hb149_monitor_v125 as monitor
from tools import c85r7_flipall_baseline as baseline_source
from tools import c85r7_flipall_measure as measure

VERSION = '2.1.3'
WORK = ROOT / '.local/scratch/slot-b/H-B-153/timing-v213'
ADMISSION = WORK / 'admission.json'
SERIES = ROOT / 'storage/competition/train_series.csv'
ZIP = ROOT / 'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip'
ORDER = ('plain-1', 'flip-1', 'flip-2', 'plain-2')
IDLE_SECONDS = 15*60
SAMPLE_SECONDS = 0.1
SAMPLER_STALL_SECONDS = 30
GIB = 2**30
CALIBRATING = False
CALIBRATION_ORDER = ('plain-cal', 'flip-cal')
MANIFEST = ROOT/'storage/external/heliosli-r7cnxfsss-v1/manifest.json'


def desktop_snapshot():
    """Measured on device zero before starting any CUDA child."""
    query = subprocess.run(['nvidia-smi', '-i', '0',
                            '--query-gpu=name,memory.total,memory.used,uuid',
                            '--format=csv,noheader,nounits'], capture_output=True,
                           text=True, check=True, timeout=5)
    device, total, used, uuid = next(csv.reader(query.stdout.splitlines()))
    value = dict(device_name=device.strip(), total_mib=float(total), used_mib=float(used), uuid=uuid.strip())
    desktop_values(value)
    return value


def desktop_values(desktop):
    total, used = desktop['total_mib']*2**20, desktop['used_mib']*2**20
    b.need('4070 Ti SUPER' in desktop['device_name'] and
           math.isfinite(total) and 16000*2**20 <= total <= 16400*2**20 and
           math.isfinite(used) and 0 <= used < total, 'measured desktop memory required')
    return used, total


def memory_admission(desktop, peaks, total_peak=None):
    used, total = desktop_values(desktop)
    b.need(bool(peaks) and all(math.isfinite(x) and 0 < x <= 12*GIB for x in peaks),
           'DO_NOT_SUBMIT: each measured peak must be <= 12 GiB')
    maximum = max(peaks)
    if total_peak is not None:
        b.need(math.isfinite(total_peak) and 0 < total_peak < total,
               'DO_NOT_SUBMIT: measured total VRAM required')
        # Device delta includes auxiliary CUDA contexts absent from worker peaks.
        maximum = max(maximum, total_peak-used)
        b.need(maximum <= 12*GIB, 'DO_NOT_SUBMIT: aggregate measured peak exceeds 12 GiB')
    b.need(used + maximum + 2*GIB <= total, 'DO_NOT_SUBMIT: desktop + max peak + 2 GiB headroom')
    return maximum


def live_memory_guard(used_mib, desktop):
    _, total = desktop_values(desktop)
    b.need(math.isfinite(used_mib) and used_mib >= 0, 'invalid live VRAM telemetry')
    # Trip one GiB EARLIER than the requested total-1GiB ceiling, allowing cleanup.
    b.need(used_mib*2**20 < total-2*GIB, 'live VRAM guard: total device use reached card total - 2 GiB')


class _GuardedTelemetry(monitor._Telemetry):
    """HB153-only guard state is lossless; the inherited queue is logging only."""
    def __init__(self, env, interval, desktop):
        super().__init__(env, interval)
        self.desktop = desktop
        self.guard_lock = threading.Lock()
        self.last_sample = time.perf_counter()
        self.last_activity = self.last_sample
        self.peak = 0
        self.failure = None

    def _run(self):
        while not self.stop.is_set():
            try:
                gpu = monitor._gpu_sample(self.env)
                now = time.perf_counter()
                # Evaluate every sample BEFORE the bounded logging queue.
                with self.guard_lock:
                    b.need(now-self.last_sample < SAMPLER_STALL_SECONDS,
                           'telemetry sampler stalled')
                    self.peak = max(self.peak, gpu[0])
                    live_memory_guard(gpu[0], self.desktop)
                    b.need(math.isfinite(gpu[1]) and 0 <= gpu[1] <= 100,
                           'invalid GPU activity')
                    self.last_sample = now
                    if gpu[1] > 0:
                        self.last_activity = now
                result = (gpu, None)
            except BaseException as exc:
                with self.guard_lock:
                    self.failure = f'telemetry sampler failed: {type(exc).__name__}: {exc}'
                return  # latched failure is consumed independently of the queue
            try:
                self.results.put_nowait(result)
            except queue.Full:
                pass
            self.stop.wait(self.interval)

    def guard_snapshot(self):
        with self.guard_lock:
            if self.failure is None and time.perf_counter()-self.last_sample >= SAMPLER_STALL_SECONDS:
                self.failure = 'telemetry sampler stalled'
            b.need(self.failure is None, self.failure)
            return self.peak, self.last_activity


def idle_guard(now, last_progress, last_activity):
    b.need(now-max(last_progress, last_activity) < IDLE_SECONDS,
           'idle watchdog: no study/window progress AND no GPU activity')


def paired_memory_admission(arms):
    """Apply the largest measured mode peak to every pre-CUDA desktop snapshot."""
    peaks = [memory_admission(a['desktop_before_cuda'],
                             [r['peak_reserved_bytes'] for r in a['rows']],
                             a['peak_total_vram_bytes']) for a in arms.values()]
    for arm in arms.values():
        memory_admission(arm['desktop_before_cuda'], peaks)


def gate(plain, flip, baseline, flip_peak):
    result = projection(plain, flip, baseline['hidden_seconds'], baseline['r7_seconds'])
    b.need(math.isfinite(flip_peak) and flip_peak > 0, 'DO_NOT_SUBMIT: measured flip peak required')
    # Explicit cross-architecture assumption, NOT a substitute for Linux/T4 qualification.
    t4_peak = 1.25*flip_peak + GIB
    result.update(t4_peak_bytes=t4_peak, measured_flip_peak_bytes=flip_peak,
                  memory_gate='PASS' if t4_peak <= 13.5*GIB else 'DO_NOT_SUBMIT',
                  t4_margin='1.25 * 4070 cumulative reserved peak + 1 GiB; empirical T4 validation pending',
                  status='PASS_LOCAL_GATES_ONLY', version=VERSION,
                  exposure='EXPOSED', clean_oof=False)
    if result['runtime_gate'] != 'PASS' or result['memory_gate'] != 'PASS':
        result['status'] = 'DO_NOT_SUBMIT'
    return result


def projection(plain, flip, hidden, r7=None):
    values = [plain, flip, hidden] + ([] if r7 is None else [r7])
    b.need(all(math.isfinite(x) and x > 0 for x in values), 'positive finite seconds required')
    b.need(r7 is None or r7 <= hidden, 'R7 phase exceeds total hidden duration')
    ratio = max(1., flip / plain)  # no speedup credit
    seconds = hidden * ratio if r7 is None else hidden + (ratio - 1) * r7
    return dict(ratio=flip/plain, projected_hidden_seconds=seconds,
                runtime_gate='PASS' if seconds <= 8*3600 else 'DO_NOT_SUBMIT',
                qualified_for_submission=False)


def cohort():
    with SERIES.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))  # series metadata only, never labels
    counts = {}
    for row in rows:
        uid = row['StudyInstanceUID']
        counts[uid] = counts.get(uid, 0) + 1
    ordered = sorted(counts, key=lambda uid: (counts[uid], uid))
    b.need(len(ordered) >= 32, 'need 32 studies')
    ids = [ordered[round(i*(len(ordered)-1)/31)] for i in range(32)]
    return ids, [row for row in rows if row['StudyInstanceUID'] in ids]


def bindings():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): b.digest(p.read_bytes()) for p in
            [Path(__file__), b.OUT/'build-receipt.json', b.SCRATCH/'replay.json',
             b.SCRATCH/'flip-cpu-smoke.json', SERIES,
             ROOT/'tools/hb149_monitor_v125.py', ROOT/'tools/hb145_output_io.py',
             ROOT/'tools/c85r7_flipall_measure.py', ROOT/'tools/c85r7_flipall_baseline.py',
             ROOT/'tools/exp189_s6_port.py', ROOT/'tools/gpu_job.py', ROOT/'tools/job_runtime.py', MANIFEST]}


def selection_inventory():
    """Rank N32 by complete study DICOM count, with archive membership bound."""
    ids, rows = cohort()
    allowed = {(r['StudyInstanceUID'], r['SeriesInstanceUID']) for r in rows}
    counts, entries = {}, []
    with zipfile.ZipFile(ZIP) as archive:
        for item in archive.infolist():
            p = item.filename.split('/')
            if len(p) == 4 and p[0] in ('train_series', 'train_images') and p[3].endswith('.dcm') and (p[1], p[2]) in allowed:
                key = (p[1], p[2])
                counts[key] = counts.get(key, 0)+1
                entries.append((item.filename, item.CRC, item.file_size))
    b.need(set(counts) == allowed, 'calibration cohort archive membership incomplete')
    totals = {uid: sum(n for (study, _), n in counts.items() if study == uid) for uid in ids}
    return dict(ranked=sorted(ids, key=lambda u: (-totals[u], u)), study_images=totals,
                archive_members_sha256=b.digest(json.dumps(sorted(entries)).encode()))


def selection_binding(inventory):
    files = b.parent()[1]  # Authenticate the hidden-proven notebook and EVERY embedded file.
    manifest = b.read(MANIFEST)
    pins = {name: b.digest(raw) for name, raw in files.items()}
    for path in (Path(__file__), SERIES, MANIFEST, ROOT/'.local/exp189-cpu-v120/input/anatomy-fast.pt',
                 MANIFEST.parent/manifest['side_model']):
        pins[str(path.relative_to(ROOT))] = b.digest(path.read_bytes())
    # Bind the native/geometry implementation as well as the frozen R7 modules.
    code = ROOT/'.local/exp189-cpu-v120/input/cloud_code'
    pins.update({str(p.relative_to(ROOT)): b.digest(p.read_bytes()) for p in code.rglob('*.py')})
    return dict(version=VERSION, inventory=inventory, source_pins=pins)


def select_admissible(inventory, probe):
    """Try in descending size order, retaining v1.1.0 refusals instead of hiding them."""
    selected, checked, refused = [], [], []
    for uid in inventory['ranked']:
        result = dict(study=uid, **probe(uid))
        checked.append(result)
        if result['error'] is None and result['windows'] > 0:
            selected.append(uid)
        else:
            refused.append(result)
        if len(selected) == 4:
            break
    return dict(studies=selected, largest_admissible=selected[0] if selected else None,
                study_images={u: inventory['study_images'][u] for u in selected},
                checked=checked, serving_refusals=refused,
                unassessed=[u for u in inventory['ranked'] if u not in {r['study'] for r in checked}],
                rule='Four largest N32 studies by total DICOMs that pass the exact v1.1.0 CStudies CPU path; ties lexical. No total-window cap.',
                qualification='CPU decode/geometry/side/anchor/render/window admission only; GPU model execution still unmeasured.')


def calibration_selection():
    record = b.read(ADMISSION.parent/'cohort-admission.json')
    b.need(record['binding'] == selection_binding(selection_inventory()), 'CPU cohort admission stale; rerun select')
    value = record['selection']
    results = {r['study']: {k: v for k, v in r.items() if k != 'study'} for r in value['checked']}
    b.need(value == select_admissible(record['binding']['inventory'], results.__getitem__) and len(value['studies']) == 4,
           'CPU cohort admission incomplete or selection mismatch')
    return value


def select_cpu():
    """Run the frozen serving dataset on CPU, including real anatomy/side models.

    No classifier or CUDA execution. Retain actual refusals and incremental
    receipts. Never substitute DICOM count for window count or truncate windows.
    """
    import importlib.util
    import torch
    b.need(not torch.cuda.is_initialized(), 'CPU selector must not initialize CUDA')
    inventory = selection_inventory()
    binding = selection_binding(inventory)
    base = ADMISSION.parent/'cpu-selection'
    runtime = base/'runtime'
    for name, raw in b.parent()[1].items():
        b.write(runtime/name, raw)
    deps = ROOT/'.local/scratch/claude-hb128/runtime/deps'
    sys.path[:0] = [str(runtime), str(deps)]
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    r7 = module('exp169_r7_port', runtime/'exp169_r7_port.py')
    diagnostic = base/'r7_runtime'
    if diagnostic.exists():
        pins = b.read(diagnostic/'runtime_sha256.json')
        for name, sha in pins.items():
            b.pin(diagnostic/name, sha)
    else:
        diagnostic = r7.diagnostic_source(runtime/'exp169_r7_src', base)
    sys.path.insert(0, str(diagnostic))
    worker = module('hb153_cpu_worker', diagnostic/'compact_infer.py')
    manifest = b.read(MANIFEST)
    data = base/'data'
    data.mkdir(parents=True, exist_ok=True)
    # Series metadata only; no labels or reports.
    b.write(data/'test_series.csv', SERIES.read_bytes())
    # Match exp189_s6_port._stage_studies: native_cache also requires the
    # UID-only test table. These are identifiers, never training targets.
    b.write(data/'test.csv', ('StudyInstanceUID\n'+'\n'.join(inventory['ranked'])+'\n').encode())
    tmp = base/'native'
    tmp.mkdir(exist_ok=True)
    ds = worker.CStudies(data, 'test', inventory['ranked'], tmp,
        ROOT/'.local/exp189-cpu-v120/input/cloud_code', manifest['size'],
        manifest['eval_stride'], manifest['views'], manifest['slices'],
        ROOT/'.local/exp189-cpu-v120/input/anatomy-fast.pt', manifest['phys_mm'],
        side_model=MANIFEST.parent/manifest['side_model'], side_threshold=manifest['side_threshold'],
        side_max_x=manifest['side_max_x'], side_position_mm=manifest['side_position_mm'],
        extent_meta=True, fluid=True, device='cpu')
    with zipfile.ZipFile(ZIP) as archive:
        members = {}
        for item in archive.infolist():
            p = item.filename.split('/')
            if len(p) == 4 and p[0] in ('train_images', 'train_series') and p[1] in inventory['ranked'] and p[3].endswith('.dcm'):
                members.setdefault(p[1], []).append((item, p))
        def probe(uid):
            for item, p in members[uid]:
                b.need(all(re.fullmatch(r'[0-9.]+', s) for s in p[1:3]) and re.fullmatch(r'[0-9.]+\.dcm', p[3]), 'unsafe DICOM path')
                b.write(data/'test_series'/Path(*p[1:]), archive.read(item))
            result = ds[inventory['ranked'].index(uid)]
            row = dict(error=result['error'], windows=len(result['images']), seconds=result['seconds'])
            io.journal(base/(uid+'.json'), dict(binding=binding, result=row), base)
            print('CPU_SERVING_ADMISSION '+json.dumps(dict(study=uid, **row)), flush=True)
            return row
        selection = select_admissible(inventory, probe)
    b.need(not torch.cuda.is_initialized(), 'CPU selector initialized CUDA')
    io.journal(ADMISSION.parent/'cohort-admission.json', dict(binding=binding, selection=selection), ADMISSION.parent)
    b.need(len(selection['studies']) == 4, 'fewer than four serving-admissible studies; see cohort-admission.json')
    return selection


def runtime_check(value, flip):
    manifest = b.read(MANIFEST)
    expected = dict(models=manifest['models'], models2=[], flip_tta=flip, workers=2,
                    **{k: manifest[k] for k in ('size', 'crop', 'views', 'slices')})
    b.need(value.get('runtime') == expected and len(expected['models']) == 4,
           'calibration runtime must prove full shape, four folds, workers=2 and correct flip mode')


def calibration_metrics(arms, ids):
    reference = None
    for name in CALIBRATION_ORDER:
        value = arms[name]
        runtime_check(value, name.startswith('flip'))
        b.need(value['done'] == len(value['rows']) == 4 and {r['study'] for r in value['rows']} == set(ids),
               'calibration incomplete four-study evidence')
        b.need(math.isfinite(value['seconds']) and 0 < value['seconds'],
               'calibration timing invalid')
        memory_admission(value['desktop_before_cuda'],
                         [r['peak_reserved_bytes'] for r in value['rows']], value['peak_total_vram_bytes'])
        current = [(r['study'], r['windows'], r['batch_windows']) for r in value['rows']]
        b.need(reference is None or current == reference, 'calibration paired windows differ')
        reference = current
        for row in value['rows']:
            measure.window_gate(row['windows'], row['batch_windows'])
            b.need(0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes'] <= 12*GIB and
                   math.isfinite(row['gpu_seconds']) and row['gpu_seconds'] > 0,
                   'calibration invalid measured CUDA peak/time')
        for key in ('peak_reserved_bytes', 'peak_allocated_bytes'):
            b.need(value[key] == max(r[key] for r in value['rows']), 'calibration peak summary mismatch')
    paired_memory_admission(arms)
    plain, flip = (arms[a] for a in CALIBRATION_ORDER)
    return dict(plain_peak_reserved_bytes=plain['peak_reserved_bytes'], peak_reserved_bytes=flip['peak_reserved_bytes'],
                plain_peak_allocated_bytes=plain['peak_allocated_bytes'], flip_peak_allocated_bytes=flip['peak_allocated_bytes'],
                peak_total_vram_bytes=max(a['peak_total_vram_bytes'] for a in arms.values()),
                healthy_studies_per_minute=min(4*60/a['seconds'] for a in arms.values()),
                max_observed_windows=max(r['windows'] for a in arms.values() for r in a['rows']))


def validate_calibration(memory):
    b.need(memory.get('schema') == 'HB153-calibration/v1' and memory.get('status') == 'PASS' and
           memory.get('timing_version') == VERSION and memory.get('bindings') == bindings(),
           'calibration identity/bindings invalid')
    b.need(memory.get('full_shape') is True and memory.get('four_folds') is True and memory.get('flip_tta') is True and
           memory.get('workers') == 2 and memory.get('max_windows') == measure.MAX_WINDOWS and
           memory.get('max_batch_windows') == measure.MAX_BATCH_WINDOWS and
           '4070 Ti SUPER' in memory.get('device', '') and memory.get('device_uuid') and
           memory.get('notebook_sha256') == b.digest((b.OUT/'kernel/release.ipynb').read_bytes()),
           'calibration full-shape identity invalid')
    b.need(0 <= memory['finished']-memory['started'] and
           math.isfinite(memory['finished']) and math.isfinite(memory['started']), 'invalid calibration timestamps')
    selection = memory['selection']
    b.need(selection == calibration_selection(), 'calibration longest-series selection changed')
    metrics = calibration_metrics(memory['per_mode'], selection['studies'])
    b.need(all(memory.get(k) == v for k, v in metrics.items()), 'calibration measured summary mismatch')
    evidence = memory['evidence']
    b.need(set(evidence) == {'cohort', 'selection', *CALIBRATION_ORDER, *(a+'-timing' for a in CALIBRATION_ORDER)},
           'calibration evidence incomplete')
    records = {}
    for name, pin in evidence.items():
        path = ROOT/pin['path']
        b.pin(path, pin['sha256'])
        records[name] = b.read(path)
    b.need(records['selection'] == selection and records['cohort']['studies'] == selection['studies'] and
           records['cohort']['files'], 'calibration prepared cohort mismatch')
    for name in CALIBRATION_ORDER:
        raw, timing, measured = records[name], records[name+'-timing'], memory['per_mode'][name]
        b.need(raw['rows'] == measured['rows'] and raw['runtime'] == measured['runtime'] and
               raw['device'] == memory['device'] and raw['done'] == 4 and raw['identity'] == 'HB153-v2' and
               timing['error'] is None and timing['exit_code'] == 0 and not timing['cleanup'] and
               timing['desktop_before_cuda'] == measured['desktop_before_cuda'] and
               timing['seconds'] == measured['seconds'] and timing['peak_gpu_mib']*2**20 == measured['peak_total_vram_bytes'],
               'calibration raw evidence mismatch')
    return memory


def validate_review(path, memory_path, memory):
    review = b.read(path)
    b.need(review.get('schema') == 'HB153-calibration-review/v1' and review.get('verdict') == 'PASS' and
           isinstance(review.get('independent_reviewer'), str) and review['independent_reviewer'].strip() and
           review.get('independent') is True and review.get('bindings') == memory['bindings'] and
           review.get('memory_receipt_sha256') == b.digest(memory_path.read_bytes()),
           'independent PASS review must bind exact calibration receipt and code')
    return review


def admit(review_path, dry_run=False):
    memory_path = WORK/'calibration/memory-receipt.json'
    memory = validate_calibration(b.read(memory_path))
    review_path = Path(review_path).resolve()
    review = validate_review(review_path, memory_path, memory)
    value = dict(verdict='PASS', independent_reviewer=review['independent_reviewer'], bindings=bindings(),
                 review_path=str(review_path.relative_to(ROOT)), review_sha256=b.digest(review_path.read_bytes()),
                 memory_receipt_path=str(memory_path.relative_to(ROOT)), memory_receipt_sha256=b.digest(memory_path.read_bytes()),
                 healthy_studies_per_minute=memory['healthy_studies_per_minute'], version=VERSION,
                 qualified_for_submission=False)
    admission(value)
    if not dry_run:
        io.journal(ADMISSION, value, WORK)
    print('PASS_LOCAL_TIMING_ADMISSION_ONLY'+(' (dry-run; no write)' if dry_run else ''))
    return value


def calibrate():
    """Explicitly authorized safety measurement, preceding independent run review."""
    started = time.time()
    b.need(CALIBRATING, 'calibration context required')
    b.need(not (WORK/'attempt.json').exists() and not (WORK/'memory-receipt.json').exists(),
           'preserve previous calibration attempt; B must review before reuse')
    device = lane_admission(None)
    state = dict(identity='HB153-calibration/v1', version=VERSION, state='RUNNING', started=started,
                 attempt_started=started, idle_seconds=IDLE_SECONDS, bindings=bindings(), device_uuid=device, controller_pid=os.getpid(), completed={},
                 exposure='EXPOSED', clean_oof=False)
    io.journal(WORK/'attempt.json', state, WORK)
    try:
        supervise('prepare', None, started)
        selection = b.read(WORK/'selection.json')
        ids = b.read(WORK/'cohort.json')['studies']
        b.need(ids == selection['studies'] and len(set(ids)) == 4 and selection['largest_admissible'] in ids,
               'calibration cohort mismatch')
        memory = None
        for arm in CALIBRATION_ORDER:
            b.need(lane_admission(memory) == device, 'calibration device changed')
            b.need(not (WORK/(arm+'-output')).exists() and not (WORK/(arm+'-measure-status.json')).exists(),
                   'preserve previous calibration arm')
            supervise(arm, None if memory is None else memory['healthy_studies_per_minute'], started)
            measured = measurements(arm, ids)
            runtime_check(measured, arm.startswith('flip'))
            timing = b.read(WORK/(arm+'-timing.json'))
            measured.update(seconds=timing['seconds'], peak_total_vram_bytes=timing['peak_gpu_mib']*2**20,
                            desktop_before_cuda=timing['desktop_before_cuda'])
            memory_admission(measured['desktop_before_cuda'], [r['peak_reserved_bytes'] for r in measured['rows']],
                             measured['peak_total_vram_bytes'])
            state['completed'][arm] = measured
            memory = dict(plain_peak_reserved_bytes=measured['peak_reserved_bytes'], healthy_studies_per_minute=240/measured['seconds'])
            io.journal(WORK/'attempt.json', state, WORK)
        arms = state['completed']
        metrics = calibration_metrics(arms, ids)
        paths = {'cohort': WORK/'cohort.json', 'selection': WORK/'selection.json'}
        paths.update({a: WORK/(a+'-measure-status.json') for a in CALIBRATION_ORDER})
        paths.update({a+'-timing': WORK/(a+'-timing.json') for a in CALIBRATION_ORDER})
        result = dict(schema='HB153-calibration/v1', status='PASS', timing_version=VERSION,
                      started=started, finished=time.time(), device=arms['plain-cal']['device'], device_uuid=device,
                      bindings=bindings(), notebook_sha256=b.digest((b.OUT/'kernel/release.ipynb').read_bytes()),
                      full_shape=True, four_folds=True, flip_tta=True, workers=2, max_windows=measure.MAX_WINDOWS,
                      max_batch_windows=measure.MAX_BATCH_WINDOWS, per_mode=arms, selection=selection,
                      evidence={k: dict(path=str(p.relative_to(ROOT)), sha256=b.digest(p.read_bytes())) for k,p in paths.items()},
                      exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
                      assumptions=['Healthy rate = minimum of four commits / full arm wall time, including cold startup.',
                                   'N4 includes the largest CPU-admissible N32 study; no total-window ceiling. GPU failures remain possible.',
                                   'No wall deadline: abort only after registered dual-signal idle period; live sampled VRAM guard protects desktop headroom.'], **metrics)
        validate_calibration(result)
        result['finished'] = time.time()
        io.journal(WORK/'memory-receipt.json', result, WORK)
        template = dict(schema='HB153-calibration-review/v1', verdict='HOLD', independent_reviewer='', independent=False,
                        bindings=bindings(), memory_receipt_sha256=b.digest((WORK/'memory-receipt.json').read_bytes()),
                        scope='Independently review code and raw calibration evidence, shape/folds, memory and measured healthy rate before PASS.')
        io.receipt(WORK/'review-template.json', template)
        state['state'] = 'PASS_AWAITING_INDEPENDENT_REVIEW'
        return result
    except BaseException as exc:
        state.update(state='ABORTED_DO_NOT_SUBMIT', error=str(exc), partial_attempt_preserved=True)
        raise
    finally:
        io.journal(WORK/'attempt.json', state, WORK)


def admission(value):
    b.need(value.get('verdict') == 'PASS' and value.get('independent_reviewer'), 'independent PASS required')
    b.need(value.get('bindings') == bindings(), 'review must bind exact build, smoke, helper and inputs')
    b.pin(ROOT/value['review_path'], value['review_sha256'])
    b.pin(ROOT/value['memory_receipt_path'], value['memory_receipt_sha256'])
    memory = b.read(ROOT/value['memory_receipt_path'])
    b.need(memory.get('status') == 'PASS' and memory.get('full_shape') is True and
           memory.get('four_folds') is True and memory.get('flip_tta') is True and
           memory.get('notebook_sha256') == b.digest((b.OUT/'kernel/release.ipynb').read_bytes()) and
           '4070' in memory.get('device', '') and
           0 < memory.get('peak_reserved_bytes', math.inf) <= 12*GIB and
           math.isfinite(memory.get('peak_total_vram_bytes', math.inf)) and memory.get('peak_total_vram_bytes', 0) > 0,
           'measured full-shape all-fold flip 4070 memory gate required')
    b.need(memory.get('timing_version') == VERSION and memory.get('bindings') == bindings() and
           memory.get('workers') == 2 and memory.get('max_windows', 'missing') is measure.MAX_WINDOWS and
           0 < memory.get('plain_peak_reserved_bytes', math.inf) <= 12*GIB,
           'v2 measured plain/full-shape flip memory receipt required')
    b.need(math.isfinite(value['healthy_studies_per_minute']) and value['healthy_studies_per_minute'] > 0,
           'measured healthy commit rate required')
    validate_calibration(memory)
    b.need(value['healthy_studies_per_minute'] == memory['healthy_studies_per_minute'], 'healthy rate differs from measurement')
    validate_review(ROOT/value['review_path'], ROOT/value['memory_receipt_path'], memory)
    return value


def prepare_data():
    if CALIBRATING:
        selection = calibration_selection()
        ids = selection['studies']
        rows = [r for r in cohort()[1] if r['StudyInstanceUID'] in ids]
        io.journal(WORK/'selection.json', selection, WORK)
    else:
        ids, rows = cohort()
    data = WORK/'data'
    data.mkdir(parents=True, exist_ok=True)
    with (data/'train_series.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected, found = set(ids), set()
    pins = {}
    with zipfile.ZipFile(ZIP) as archive:
        for item in archive.infolist():
            parts = item.filename.split('/')
            if len(parts) != 4 or parts[0] not in ('train_series', 'train_images') or parts[1] not in selected or not parts[3].endswith('.dcm'):
                continue
            b.need(all(re.fullmatch(r'[0-9.]+', s) for s in parts[1:3]) and re.fullmatch(r'[0-9.]+\.dcm', parts[3]), 'unsafe DICOM path')
            target = data/'train_series'/Path(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read(item)
            if not target.exists():
                temporary = target.with_suffix('.partial')
                temporary.write_bytes(raw)
                io.replace(temporary, target)
            b.need(target.read_bytes() == raw, 'staged DICOM changed')
            pins[str(target.relative_to(data))] = b.digest(raw)
            found.add(parts[1])
            if len(pins) % 100 == 0:
                print('PREP_PROGRESS '+str(len(pins)), flush=True)
    b.need(found == selected and all(any(k.startswith('train_series/'+r['StudyInstanceUID']+'/'+r['SeriesInstanceUID']+'/')
                                   for k in (p.replace('\\', '/') for p in pins)) for r in rows),
           'N32 image membership incomplete')
    io.journal(WORK/'cohort.json', dict(identity='HB153-v2', studies=ids, files=pins,
               series_sha256=b.digest(SERIES.read_bytes())), WORK)
    return ids


def progress(line):
    match = re.search(r'PROGRESS worker (\d+) of \d+ (\{.*\})', line)
    if match:
        return int(match[1]), int(json.loads(match[2]).get('done', 0))
    return None


def gpu_python_pids(output, own):
    """Windows desktop C+G apps are allowed; Python CUDA processes block."""
    result = []
    for row in csv.reader(output.splitlines()):
        if len(row) == 2 and 'python' in row[1].lower() and int(row[0]) not in own:
            result.append(int(row[0]))
    return result


def supervise(arm, healthy, attempt_started=None):
    """Only stdout and telemetry while live; no scans of running output dirs."""
    import psutil
    attempt_started = time.time() if attempt_started is None else attempt_started
    desktop = desktop_snapshot()
    live_memory_guard(desktop['used_mib'], desktop)
    command = [sys.executable, '-u', '-B', str(Path(__file__)),
               '_prepare' if arm == 'prepare' else '_arm', '--arm', arm, '--attempt-started', str(attempt_started)]
    if CALIBRATING:
        command.append('--calibration')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    events = queue.Queue()
    child = job = None
    tracked, completed = {}, {}
    started = last_progress = last_activity = time.perf_counter()
    telemetry = _GuardedTelemetry(env, SAMPLE_SECONDS, desktop)
    status = monitor._StatusWriter(WORK/'status.json')
    error, cleanup, peak, code = None, [], 0, None
    phase = 'load/verify/preprocess'
    try:
        status.start()
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', start_new_session=os.name != 'nt')
        root = psutil.Process(child.pid)
        tracked[root.pid] = root
        job = monitor._WindowsJob(child)
        def consume():
            for line in child.stdout:
                events.put(line)
        threading.Thread(target=consume, daemon=True).start()
        telemetry.start()
        while child.poll() is None:
            peak, last_activity = telemetry.guard_snapshot()
            now = time.perf_counter()
            while not events.empty():
                line = events.get_nowait()
                print(line, end='', flush=True)
                event = progress(line)
                if event and event[1] > completed.get(event[0], 0):
                    completed[event[0]] = event[1]
                    last_progress, phase = now, 'study inference'
                if line.startswith('PREP_PROGRESS '):
                    count = int(line.split()[1])
                    if count > completed.get(-1, 0):
                        completed[-1], last_progress, phase = count, now, 'prepare'
            # Only the explicitly named status journal is read while the child runs.
            status_path = WORK/(arm+'-measure-status.json')
            try:
                current = b.read(status_path)
                if current['done'] > completed.get(0, 0):
                    completed[0], last_progress, phase = current['done'], now, 'study inference'
            except (OSError, ValueError, KeyError):
                pass
            try:
                processes = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                processes = []  # normal exit between poll() and enumeration
            ram = 0
            for process in processes:
                try:
                    tracked[process.pid] = process
                    mem = process.memory_info()
                    ram += max(mem.rss, getattr(mem, 'private', mem.rss))
                except psutil.NoSuchProcess:
                    pass
            b.need(ram < 38*10**9, 'RAM limit')
            telemetry.take()  # optional logging queue cannot affect guard decisions
            peak, last_activity = telemetry.guard_snapshot()
            idle_guard(now, last_progress, last_activity)
            status.submit(dict(arm=arm, phase=phase, completed=sum(completed.values()), seconds=now-started, peak_gpu_mib=peak))
            time.sleep(SAMPLE_SECONDS)
        code = child.wait(timeout=2)
        b.need(telemetry.close(), 'telemetry sampler stalled during shutdown')
        peak, last_activity = telemetry.guard_snapshot()
        b.need(code == 0, 'R7 arm failed')
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        elapsed = time.perf_counter()-started
        cleanup = monitor._cleanup(child, tracked, job, psutil)
        if telemetry.thread.ident is not None:
            telemetry.close()
        with telemetry.guard_lock:
            peak = max(peak, telemetry.peak)  # retain even the sample that aborted the run
        status.submit(dict(arm=arm, phase='STOPPED' if error else 'COMPLETE', error=error, cleanup=cleanup, seconds=elapsed))
        status.close()
        io.receipt(WORK/(arm+'-timing.json'), dict(seconds=elapsed, error=error, exit_code=code,
                       cleanup=cleanup, peak_gpu_mib=peak, desktop_before_cuda=desktop,
                       idle_seconds=IDLE_SECONDS, sample_seconds=SAMPLE_SECONDS, exposure='EXPOSED', clean_oof=False))
    b.need(not cleanup, 'owned process cleanup incomplete')
    return elapsed


def arm_run(arm):
    mode = arm.split('-')[0]
    files = b.parent()[1] if mode == 'plain' else b.unpack(b.read(b.OUT/'kernel/release.ipynb'))[0]
    runtime = WORK/(arm+'-runtime')
    for name, raw in files.items():
        b.write(runtime/name, raw)
    deps = ROOT/'.local/scratch/claude-hb128/runtime/deps'
    os.environ['PYTHONPATH'] = os.pathsep.join([str(runtime), str(deps), str(ROOT)])
    os.environ['HB153_MEASURE_STATUS'] = str(WORK/(arm+'-measure-status.json'))
    sys.path[:0] = [str(runtime), str(deps)]
    import exp169_r7_port as r7
    frozen_diagnostic = r7.diagnostic_source
    def diagnostic(*args, **kwargs):
        path = frozen_diagnostic(*args, **kwargs)
        worker = path/'compact_infer.py'
        worker.write_text(measure.instrument(worker.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
        return path
    r7.diagnostic_source = diagnostic
    ids = b.read(WORK/'cohort.json')['studies']
    r7.infer_studies([WORK/'data/train_series'/uid for uid in ids],
        bundle_dir=ROOT/'storage/external/heliosli-r7cnxfsss-v1',
        code_dir=ROOT/'.local/exp189-cpu-v120/input/cloud_code',
        anatomy_weights=ROOT/'.local/exp189-cpu-v120/input/anatomy-fast.pt',
        output_dir=WORK/(arm+'-output'), source_dir=runtime/'exp169_r7_src',
        workers=2, amp='auto', device='cuda', max_fallback_fraction=1.0)


def lane_admission(memory):
    """Live checks only on B's explicitly requested GPU run, never CPU dry-runs."""
    import psutil
    parents = psutil.Process().parents()
    b.need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in parents), 'gpu_job lock required')
    from tools.job_runtime import LOGS, job_status
    job = job_status(LOGS, 'HB153_R7_CAL' if CALIBRATING else 'HB153_R7_N32', tail=False)
    b.need(job['state'] == 'running' and job['pid'] in {p.pid for p in parents},
           'exclusive timing lane required')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=5)
    b.need(not gpu_python_pids(gpu.stdout, {os.getpid(), *[p.pid for p in parents]}),
           'another Python ML process owns the GPU')
    desktop = desktop_snapshot()
    live_memory_guard(desktop['used_mib'], desktop)
    b.need(memory is not None or CALIBRATING, 'calibration-only initial memory admission')
    if memory is not None:
        memory_admission(desktop, [memory['plain_peak_reserved_bytes'],
                                  memory.get('peak_reserved_bytes', memory['plain_peak_reserved_bytes'])])
    return desktop['uuid']



def measurements(arm, ids):
    value = b.read(WORK/(arm+'-measure-status.json'))  # child already stopped
    rows = value['rows']
    b.need(value['identity'] == 'HB153-v2' and value['done'] == len(ids) and len(rows) == len(ids) and
           {r['study'] for r in rows} == set(ids) and '4070' in value['device'],
           'DO_NOT_SUBMIT: incomplete or mismatched N32 measurement')
    for row in rows:
        measure.window_gate(row['windows'], row['batch_windows'])
        b.need(all(math.isfinite(row[k]) and row[k] > 0 for k in
                   ('gpu_seconds', 'peak_reserved_bytes', 'peak_allocated_bytes')) and
               row['peak_allocated_bytes'] <= row['peak_reserved_bytes'] <= 12*GIB,
               'DO_NOT_SUBMIT: invalid measured peak/time')
    value.update(peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
                 peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                 max_observed_windows=max(r['windows'] for r in rows))
    return value


def run(value):
    """Preparation and GPU children share a persisted attempt identity, without a wall deadline.

    An aborted attempt is preserved and cannot silently become a fresh timing
    sample. B must review/archive it before reuse; partial journals remain usable
    for diagnosis. No incomplete study or mode is included in a ratio.
    """
    started = time.time()
    base = baseline_source.baseline()
    memory = b.read(ROOT/value['memory_receipt_path'])
    device = lane_admission(memory)
    b.need(not (WORK/'attempt.json').exists(), 'preserve previous attempt; B must review before reuse')
    state = dict(identity='HB153-v2', version=VERSION, state='RUNNING', started=started,
                 attempt_started=started, idle_seconds=IDLE_SECONDS, bindings=bindings(), baseline=base, device_uuid=device,
                 order=ORDER, completed={}, exposure='EXPOSED', clean_oof=False)
    io.journal(WORK/'attempt.json', state, WORK)
    try:
        supervise('prepare', value['healthy_studies_per_minute'], started)
        ids = b.read(WORK/'cohort.json')['studies']
        b.need(len(ids) == len(set(ids)) == 32, 'N32 required')
        for arm in ORDER:
            b.need(lane_admission(memory) == device, 'device changed between paired arms')
            b.need(not (WORK/(arm+'-output')).exists() and not (WORK/(arm+'-measure-status.json')).exists(),
                   'preserve previous arm; no stale measurements')
            elapsed = supervise(arm, value['healthy_studies_per_minute'], started)
            measured = measurements(arm, ids)
            timing = b.read(WORK/(arm+'-timing.json'))
            memory_admission(timing['desktop_before_cuda'], [r['peak_reserved_bytes'] for r in measured['rows']],
                             timing['peak_gpu_mib']*2**20)
            state['completed'][arm] = dict(seconds=elapsed, desktop_before_cuda=timing['desktop_before_cuda'],
                                          peak_total_vram_bytes=timing['peak_gpu_mib']*2**20, **measured)
            io.journal(WORK/'attempt.json', state, WORK)
        arms = state['completed']
        paired_memory_admission(arms)
        reference = [(r['study'], r['windows'], r['batch_windows']) for r in arms[ORDER[0]]['rows']]
        b.need(all([(r['study'], r['windows'], r['batch_windows']) for r in a['rows']] == reference
                   for a in arms.values()), 'DO_NOT_SUBMIT: mode window/cohort mismatch')
        # AB then BA; take the largest pair ratio AND GPU-only ratio. This avoids
        # startup/CPU overhead hiding extra GPU work in the serial-loader proxy.
        ratios = [arms[f'flip-{i}']['seconds']/arms[f'plain-{i}']['seconds'] for i in (1, 2)]
        ratios += [sum(r['gpu_seconds'] for r in arms[f'flip-{i}']['rows']) /
                   sum(r['gpu_seconds'] for r in arms[f'plain-{i}']['rows']) for i in (1, 2)]
        flip_peak = max(r['peak_reserved_bytes'] for name, a in arms.items() if name.startswith('flip') for r in a['rows'])
        result = gate(1., max(ratios), base, flip_peak)
        result.update(baseline=base, pair_ratios=ratios, bindings=bindings(), device_uuid=device,
                      per_mode=arms, order=ORDER, workers=2, max_windows=measure.MAX_WINDOWS,
                      max_batch_windows=measure.MAX_BATCH_WINDOWS,
                      assumptions=['ABBA same 32 studies; cold compile included in each process; max pair and GPU ratios, no speedup credit.',
                                   'workers=2 matches serving; allocated/reserved peaks cover the main worker; sampled device VRAM includes loader auxiliary models and desktop.',
                                   'No full-study window cap; encoder chunk192 is the frozen ConvNeXt serving default.',
                                   'T4 memory margin and local timing transfer are assumptions; hidden windows and Linux scheduling still require qualification.'])
        io.journal(WORK/'paired.json', dict(identity='HB153-v2', **result), WORK)
        b.need(result['status'] != 'DO_NOT_SUBMIT', 'DO_NOT_SUBMIT: runtime or T4 VRAM gate failed')
        state['state'] = 'PASS_LOCAL_GATES_ONLY'
        return result
    except BaseException as exc:
        state.update(state='ABORTED_DO_NOT_SUBMIT', error=str(exc), partial_attempt_preserved=True)
        raise
    finally:
        io.journal(WORK/'attempt.json', state, WORK)


def prepare_dry_run(calibration=False):
    """CPU real-archive check: membership plus one decoded DICOM per N32 study."""
    import pydicom
    import io as stdio
    ids, rows = cohort()
    selection = calibration_selection() if calibration else None
    if selection:
        ids = selection['studies']
        rows = [r for r in rows if r['StudyInstanceUID'] in ids]
    selected = set(ids)
    samples, series = {}, set()
    with zipfile.ZipFile(ZIP) as archive:
        for item in archive.infolist():
            parts = item.filename.split('/')
            if len(parts) == 4 and parts[0] in ('train_series', 'train_images') and parts[1] in selected and parts[3].endswith('.dcm'):
                series.add((parts[1], parts[2]))
                if parts[1] not in samples:
                    raw = archive.read(item)
                    ds = pydicom.dcmread(stdio.BytesIO(raw))
                    pixels = ds.pixel_array
                    b.need(pixels.size > 0, 'DICOM decode failed')
                    samples[parts[1]] = dict(sha256=b.digest(raw), shape=list(pixels.shape))
    b.need(set(samples) == selected and all((r['StudyInstanceUID'], r['SeriesInstanceUID']) in series for r in rows),
           'N32 archive membership incomplete')
    result = dict(status='PASS_CPU_CALIBRATION_DRY_RUN' if calibration else 'PASS_CPU_PREPARE_DRY_RUN',
                  studies=len(ids), samples=samples, selection=selection,
                  full_extraction=False, qualified_for_submission=False)
    io.receipt(WORK/('calibrate-dry-run.json' if calibration else 'prepare-dry-run.json'), result)
    print(json.dumps(dict(status=result['status'], studies=len(ids), series=len(series))))


def main(argv=None):
    global WORK, CALIBRATING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'plan', 'validate', 'prepare', 'calibrate', 'admit', 'run', '_arm', '_prepare'])
    parser.add_argument('--arm', choices=[*ORDER, *CALIBRATION_ORDER, 'prepare'])
    parser.add_argument('--attempt-started', type=float)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--review', type=Path)
    parser.add_argument('--calibration', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dry_run and args.command not in ('prepare', 'admit', 'calibrate'):
        parser.error('--dry-run is only valid for prepare/calibrate/admit')
    if args.calibration and args.command not in ('_arm', '_prepare'):
        parser.error('--calibration is internal only')
    if args.review and args.command != 'admit':
        parser.error('--review is only valid for admit')
    CALIBRATING = args.command == 'calibrate' or args.calibration
    if CALIBRATING:
        WORK = ADMISSION.parent/'calibration'
    b.validate()
    if args.command == 'select':
        select_cpu()
        return 0
    if args.command == 'plan':
        ids, rows = cohort()
        value = dict(status='HOLD_REVIEW_AND_MEASURED_MEMORY', studies=len(ids), series=len(rows),
                     bindings=bindings(), verdict='HOLD', independent_reviewer=None,
                     review_path=None, review_sha256=None, memory_receipt_path=None,
                     memory_receipt_sha256=None, healthy_studies_per_minute=None,
                     qualified_for_submission=False, version=VERSION,
                     baseline=baseline_source.baseline(), order=ORDER, idle_seconds=IDLE_SECONDS,
                     memory_requirements='calibrate produces N4 all-fold plain+flip 4070 receipt, workers=2, no total-window ceiling; each measured peak <=12GiB; desktop + max peak +2GiB <= card total',
                     review_scope='calibrate -> independent review.json from review-template.json -> admit --review -> run')
        io.atomic_json(WORK/'admission-template.json', value)
        print(json.dumps(value, indent=2))
        return
    if args.command == 'validate':
        base = baseline_source.baseline()
        print(json.dumps(dict(status='PASS_CPU_IDENTITY', hidden_seconds=base['hidden_seconds'], r7_seconds=base['r7_seconds'])))
        return
    if args.command == 'prepare':
        b.need(args.dry_run, 'CPU prepare requires --dry-run; full preparation occurs only inside run watchdog')
        prepare_dry_run()
        return
    if args.command == 'calibrate':
        if args.dry_run:
            prepare_dry_run(calibration=True)
        else:
            result = calibrate()
            print(json.dumps({k: result[k] for k in ('status', 'peak_reserved_bytes', 'healthy_studies_per_minute')}))
        return 0
    if args.command == 'admit':
        if args.review is None:
            if args.dry_run and not ADMISSION.exists():
                print('DO_NOT_SUBMIT: calibration receipt and --review required')
                return 2
            b.need(args.dry_run, 'admit requires --review independent review.json')
            admission(b.read(ADMISSION))
            print('PASS_LOCAL_TIMING_ADMISSION_ONLY (existing admission dry-run)')
        else:
            admit(args.review, args.dry_run)
        return 0
    if not CALIBRATING:
        value = admission(b.read(ADMISSION))
    if args.command in ('_arm', '_prepare'):
        b.need(args.attempt_started is not None and math.isfinite(args.attempt_started), 'supervised attempt required')
        state = b.read(WORK/'attempt.json')
        b.need(state['state'] == 'RUNNING' and state['attempt_started'] == args.attempt_started and state['bindings'] == bindings(),
               'supervised attempt identity required')
        if CALIBRATING:
            b.need(state['identity'] == 'HB153-calibration/v1' and state['version'] == VERSION and
                   state['attempt_started'] == state['started'] and state['idle_seconds'] == IDLE_SECONDS and os.getppid() == state['controller_pid'],
                   'authenticated calibration controller required')
            b.need(args.arm in (*CALIBRATION_ORDER, 'prepare'), 'calibration arm required')
        else:
            b.need(args.arm in (*ORDER, 'prepare'), 'timing arm required')
        if args.command == '_prepare':
            prepare_data()
            return
        arm_run(args.arm)
        return
    result = run(value)
    print(json.dumps({k:result[k] for k in ('status', 'projected_hidden_seconds', 't4_peak_bytes')}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('DO_NOT_SUBMIT: '+str(exc), file=sys.stderr)
        raise SystemExit(1)
