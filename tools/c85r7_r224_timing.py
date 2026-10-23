"""H-B-164 R224-only N32 timing, v1.0.1 (local diagnostic, never packaged).

Changelog 1.0.1: telemetry failures preserve inference but invalidate timing;
lossless measured-memory guard remains binding. Backup: hb164/timing-v100-backup.

Changelog 1.0.0: AST-only public functions, sealed CPU preparation and independent
admission; supervised B-only GPU measurement. No total execution deadline.
Windows uses the public canonical reader sequentially, not Linux fork/fastread.
This measures a conservative local path, not Linux serving parity or clean OOF.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io as streams
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_flipall_timing as shared
from tools import hb145_output_io as io
from tools import hb149_monitor_v125 as monitor

VERSION = '1.0.1'
TAG = 'HB164_R224_N32'
GPU_PYTHON = Path(r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe')
WORK = ROOT / '.local/scratch/hb164-r224/hb164/timing'
SOURCE = ROOT / '.local/scratch/hb164-r224/hb164/nartaa-cell-4.txt'
SOURCE_SHA = '211858b5f276c60e4bbcd4f81e9519ca63fb01aef252e69db60cbed32a601757'
CHECKPOINT = SOURCE.parent / 'nartaa-original/raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'
CHECKPOINT_SHA = '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'
R7_PAIRED = ROOT / '.local/scratch/slot-b/H-B-153/timing-v213/paired.json'
R7_PAIRED_SHA = 'bc765b6413fac2df506bdb9df1072b19aa3ee978720ca9482e014b38e07ed620'
BASELINE_HOURS = 5.38
LIMIT_HOURS = 8.5
GIB = 2**30
SAMPLER_GAP_SECONDS = 30
FUNCTIONS = ('_crop', 'build_backbone', 'RaptorClassifier', 'load_model',
             '_eval_centers', 'eval_windows', 'gpu_windows', 'infer_probs',
             '_make_reader', '_pick_series_for_slot', 'build_study')
CONSTANTS = ('IMG', 'CROP_MM', 'SLOTS', 'K_EVAL', 'HDR_THREADS', 'PX_THREADS',
             'NORM', 'LAB', 'CROP_FRAC', 'ARMS')
DIFFERENCES = [
    'AST allowlist only; public main, Kaggle discovery, XLA and notebook not executed',
    'Windows sequential canonical _make_reader/build_study, not Linux fork/fastread',
    'One local CUDA device; no TTA, recipe changes, batching changes or ensemble',
    'Explicit local paths; startup plus sequential decode and inference included',
    'Local 4070 timing does not establish Kaggle T4/Linux parity',
    'Transparent model-call observation records public fp16-to-fp32 retries without changing inference',
]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def stamp(**value):
    return dict(identity=TAG, version=VERSION, exposure='EXPOSED', clean_oof=False, **value)


def source_parts(path=SOURCE):
    """Parse source without running imports, notebook main or arbitrary statements."""
    need(sha(path) == SOURCE_SHA, 'public source hash changed')
    text = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(text)
    constants, functions = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in CONSTANTS:
                need(name not in constants, 'duplicate source constant')
                constants[name] = ast.literal_eval(node.value)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in FUNCTIONS:
            need(node.name not in functions, 'duplicate source function')
            # Keep original decorated function lines, not a reimplementation.
            start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            functions[node.name] = '\n'.join(text.splitlines()[start-1:node.end_lineno])
    need(set(constants) == set(CONSTANTS) and set(functions) == set(FUNCTIONS), 'source allowlist incomplete')
    need(constants['IMG'] == 384 and constants['CROP_MM'] == 140.0 and
         constants['CROP_FRAC'] == .8 and constants['K_EVAL'] == 94 and
         constants['NORM'] == 'imagenet' and sum(x[2] for x in constants['SLOTS']) == 96 and
         constants['HDR_THREADS'] == 16 and constants['PX_THREADS'] == 8 and
         len(constants['LAB']) == 12 and len(constants['ARMS']) == 1 and
         constants['ARMS'][0] == dict(file=CHECKPOINT.name,
             arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', res=224, w=1.0), 'R224 recipe changed')
    extracted = '\n\n'.join(functions[name] for name in FUNCTIONS)
    compile(extracted, '<R224-reviewed-functions-only>', 'exec')
    return constants, extracted


def public_namespace():
    """Only called by CPU replay or authenticated worker; never execute whole source."""
    import glob
    import gc
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    import timm
    constants, extracted = source_parts()
    ns = dict(constants, np=np, torch=torch, nn=torch.nn, F=torch.nn.functional,
              timm=timm, glob=glob, gc=gc, ThreadPoolExecutor=ThreadPoolExecutor,
              MAXS=sum(x[2] for x in constants['SLOTS']), _WLUT={},
              _MEAN=torch.tensor([.485, .456, .406]).view(3, 1, 1),
              _STD=torch.tensor([.229, .224, .225]).view(3, 1, 1))
    exec(compile(extracted, '<R224-reviewed-functions-only>', 'exec'), ns)
    return ns


def inventory():
    ids, rows = shared.cohort()
    need(len(ids) == len(set(ids)) == 32, 'exact shared N32 cohort required')
    members, found = [], set()
    expected = {(r['StudyInstanceUID'], r['SeriesInstanceUID']) for r in rows}
    with zipfile.ZipFile(shared.ZIP) as archive:
        for item in archive.infolist():
            parts = item.filename.split('/')
            if len(parts) != 4 or parts[0] not in ('train_series', 'train_images') or parts[1] not in ids or not parts[3].endswith('.dcm'):
                continue
            need(all(re.fullmatch(r'[0-9.]+', x) for x in parts[1:3]) and
                 re.fullmatch(r'[0-9.]+\.dcm', parts[3]), 'unsafe DICOM path')
            need((parts[1], parts[2]) in expected, 'DICOM series absent from metadata')
            members.append(dict(member=item.filename, target='/'.join(['train_series'] + parts[1:]),
                                size=item.file_size, crc=item.CRC))
            found.add((parts[1], parts[2]))
    need(found == expected and len({m['target'] for m in members}) == len(members), 'missing or duplicate DICOM membership')
    stat = shared.ZIP.stat()
    return dict(studies=ids, series=rows, members=sorted(members, key=lambda m: m['target']),
                series_sha256=sha(shared.SERIES), archive_size=stat.st_size,
                archive_mtime_ns=stat.st_mtime_ns)


def cpu_decode_probe(inv):
    """One actual DICOM pixel decode per study; metadata only, never labels."""
    import pydicom
    import numpy as np
    transfers = set()
    by_study = {}
    for item in inv['members']:
        by_study.setdefault(item['target'].split('/')[1], item)
    with zipfile.ZipFile(shared.ZIP) as archive:
        for uid in inv['studies']:
            ds = pydicom.dcmread(streams.BytesIO(archive.read(by_study[uid]['member'])))
            pixels = ds.pixel_array
            need(pixels.size > 0 and np.isfinite(pixels).all(), 'CPU DICOM decode failed')
            transfers.add(str(ds.file_meta.TransferSyntaxUID))
    return dict(decoded_studies=len(by_study), transfer_syntaxes=sorted(transfers))


def prepare(dry_run=False):
    constants, extracted = source_parts()
    inv = inventory()
    probe = cpu_decode_probe(inv)
    result = stamp(state='CPU_DRY_RUN' if dry_run else 'CPU_PREPARED',
                   inventory_sha256=digest(inv), cohort_sha256=digest(inv['studies']),
                   source_sha256=sha(SOURCE), extracted_sha256=hashlib.sha256(extracted.encode()).hexdigest(),
                   tool_sha256=sha(__file__), recipe=constants, differences=DIFFERENCES, probe=probe)
    if not dry_run:
        pins = {}
        with zipfile.ZipFile(shared.ZIP) as archive:
            for item in inv['members']:
                raw = archive.read(item['member'])
                target = WORK / 'data' / item['target']
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    temporary = target.with_suffix('.partial')
                    io.retry(temporary.write_bytes, raw)
                    io.replace(temporary, target)
                value = hashlib.sha256(raw).hexdigest()
                need(sha(target) == value, 'staged DICOM changed')
                pins[item['target']] = value
        result.update(files=pins, studies=inv['studies'], series=inv['series'])
        # Real canonical reader replay on one complete study, without touching CUDA.
        ns = public_namespace()
        grouped = group_series(inv['series'])
        vol, mask = ns['build_study'](inv['studies'][0], grouped, str(WORK/'data/train_series'), ns['_make_reader']())
        validate_volume(vol, mask)
        windows = ns['gpu_windows'](vol, mask, 94, 224, 'cpu')
        reference = ns['eval_windows'](vol, mask, 94, 224)
        need(ns['torch'].equal(windows, reference), 'CPU window parity failed')
        result['real_study_cpu_replay'] = dict(shape=list(windows.shape), exact_window_parity=True)
    io.journal(WORK/('prepare-dry-run.json' if dry_run else 'prepare.json'), result, WORK)
    return result


def group_series(rows):
    result = {}
    for row in rows:
        result.setdefault(row['StudyInstanceUID'], []).append(row)
    return result


def validate_volume(vol, mask):
    import numpy as np
    need(vol.shape == (96, 384, 384) and vol.dtype == np.uint8 and
         mask.shape == (96,) and np.isfinite(vol).all() and int(mask.sum()) > 0,
         'invalid/empty preprocessing volume; no silent fallback timing')


def transfer_calibration():
    """Authenticate D-B-188 raw timings; cross-architecture transfer is an assumption."""
    need(sha(R7_PAIRED) == R7_PAIRED_SHA, 'R7 calibration receipt hash changed')
    paired = io.read_json(R7_PAIRED)
    base = shared.baseline_source.baseline()
    need(paired.get('identity') == 'HB153-v2' and paired.get('version') == '2.1.3' and
         paired.get('exposure') == 'EXPOSED' and paired.get('clean_oof') is False,
         'R7 calibration identity mismatch')
    need(all(paired['baseline'][k] == base[k] for k in ('hidden_seconds', 'r7_seconds')),
         'R7 hidden baseline changed')
    ids = shared.cohort()[0]
    seconds = []
    for name in ('plain-1', 'plain-2'):
        arm = paired['per_mode'][name]
        need(arm['done'] == 32 and len(arm['rows']) == 32 and
             {r['study'] for r in arm['rows']} == set(ids) and
             arm['runtime']['flip_tta'] is False and arm['runtime']['workers'] == 2 and
             '4070 Ti SUPER' in arm['device'] and math.isfinite(arm['seconds']) and arm['seconds'] > 0,
             'R7 calibration cohort/runtime mismatch')
        seconds.append(arm['seconds'])
    local = math.floor(min(seconds))  # smaller denominator conservatively increases transferred cost
    need(local > 0 and 0 < base['r7_seconds'] <= base['hidden_seconds'] <= BASELINE_HOURS*3600,
         'invalid R7 baseline or local denominator')
    return dict(source='D-B-188 raw authenticated receipts; discovery cross-check PENDING',
                paired_path=str(R7_PAIRED.relative_to(ROOT)).replace('\\', '/'), paired_sha256=R7_PAIRED_SHA,
                baseline=base, local_plain_seconds=seconds, conservative_plain_seconds=local,
                r7_hidden_seconds=base['r7_seconds'], scale=base['r7_seconds']/local,
                cohort_sha256=digest(ids), device_uuid=paired['device_uuid'],
                assumption='Transfer R7 local-N32 to Kaggle-hidden scale to R224; different model/reader paths and Linux/T4 behavior remain unqualified. '
                           'The authenticated baseline assumes 1300 hidden studies; this is not a measured hidden count. '
                           'Sequential local decode/startup is included and may overestimate cost; no speedup credit or GPU-count division.')


def input_binding(build_review):
    need(build_review is not None and Path(build_review).is_file(), 'independent build review file required')
    need(sha(CHECKPOINT) == CHECKPOINT_SHA, 'checkpoint hash mismatch')
    source_parts()
    from tools import c85r7_r224q_v1123 as builder
    # The builder owns stage validation. A validation exception is binding.
    builder.validate()
    stage = Path(builder.OUT)
    need((stage/'build-receipt.json').is_file(), 'stage build receipt missing')
    stage_pins = {str(p.relative_to(stage)).replace('\\', '/'): sha(p)
                  for p in sorted(stage.rglob('*')) if p.is_file()}
    inv = inventory()
    prepared = io.read_json(WORK/'prepare.json')
    need(prepared.get('state') == 'CPU_PREPARED' and prepared.get('tool_sha256') == sha(__file__) and
         prepared.get('inventory_sha256') == digest(inv) and prepared.get('studies') == inv['studies'] and
         prepared.get('real_study_cpu_replay', {}).get('exact_window_parity') is True,
         'fresh complete CPU preparation/replay required')
    expected = {m['target'] for m in inv['members']}
    need(set(prepared['files']) == expected, 'prepared file set changed')
    for name, value in prepared['files'].items():
        need(sha(WORK/'data'/name) == value, 'prepared DICOM hash mismatch')
    helpers = ('c85r7_r224_timing.py', 'c85r7_flipall_timing.py', 'c85r7_flipall_v1115.py',
               'c85r7_flipall_measure.py', 'c85r7_flipall_baseline.py', 'hb145_output_io.py',
               'hb149_monitor_v125.py', 'gpu_job.py', 'gpu_queue.py', 'job_runtime.py',
               'c85r7_r224q_v1123.py')
    return stamp(transfer_calibration=transfer_calibration(), baseline_hours=BASELINE_HOURS,
                 runtime_limit_hours=LIMIT_HOURS, checkpoint_sha256=CHECKPOINT_SHA,
                 source_sha256=SOURCE_SHA, cohort_sha256=digest(inv['studies']),
                 inventory_sha256=digest(inv), prepare_sha256=sha(WORK/'prepare.json'),
                 build_review=str(Path(build_review).resolve()), build_review_sha256=sha(build_review),
                 stage_files=stage_pins, helper_files={n: sha(ROOT/'tools'/n) for n in helpers},
                 test_sha256=sha(ROOT/'tests/test_c85r7_r224_timing.py'), differences=DIFFERENCES)


def validate_review(review, binding, smoke_sha):
    need(review.get('verdict') == 'PASS' and isinstance(review.get('reviewer'), str) and
         bool(review['reviewer'].strip()) and isinstance(review.get('builder'), str) and
         bool(review['builder'].strip()) and review['reviewer'] != review['builder'],
         'independent timing review PASS required')
    need(review.get('binding_sha256') == digest(binding) and
         review.get('build_review_sha256') == binding['build_review_sha256'] and
         review.get('smoke_sha256') == smoke_sha and review.get('execution_authority') == 'B' and
         review.get('approved_differences') == DIFFERENCES and
         review.get('reviewed_functions') == list(FUNCTIONS), 'review evidence stale or incomplete')


def validate(build_review):
    binding = input_binding(build_review)
    result = stamp(state='CPU_VALIDATED', binding=binding, binding_sha256=digest(binding))
    io.journal(WORK/'validate.json', result, WORK)
    return result


def admit(build_review, review_path, dry_run=False):
    binding = input_binding(build_review)
    smoke = io.read_json(WORK/'validate.json')
    need(smoke.get('state') == 'CPU_VALIDATED' and smoke.get('binding') == binding,
         'fresh CPU validation receipt required')
    need(review_path is not None, 'timing review required; dry-run never bypasses review')
    review = io.read_json(review_path)
    validate_review(review, binding, sha(WORK/'validate.json'))
    value = stamp(state='DRY_RUN_PASS' if dry_run else 'ADMITTED', binding=binding,
                  binding_sha256=digest(binding), review=str(Path(review_path).resolve()),
                  review_sha256=sha(review_path), smoke_sha256=sha(WORK/'validate.json'))
    io.journal(WORK/('admit-dry-run.json' if dry_run else 'admission.json'), value, WORK)
    if not dry_run:
        from tools.gpu_queue import validate_plan
        plan = validate_plan([dict(tag=TAG, command=launch_command())])
        io.atomic_json(WORK/'gpu-plan.json', plan)
    return value


def admitted():
    value = io.read_json(WORK/'admission.json')
    need(value.get('state') == 'ADMITTED', 'GPU admission missing; dry-run is not authorization')
    old = value['binding']
    current = input_binding(old['build_review'])
    need(current == old and value.get('binding_sha256') == digest(current), 'GPU admission input drift')
    need(sha(value['review']) == value['review_sha256'] and
         sha(WORK/'validate.json') == value['smoke_sha256'], 'GPU review/smoke drift')
    validate_review(io.read_json(value['review']), current, value['smoke_sha256'])
    return value


def projection(seconds, calibration, measured_count=32):
    local, hidden = calibration['conservative_plain_seconds'], calibration['r7_hidden_seconds']
    need(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in (seconds, local, hidden)) and
         measured_count == 32, 'invalid additive projection inputs')
    extra = seconds * hidden / local / 3600
    hours = BASELINE_HOURS + extra
    return dict(baseline_hours=BASELINE_HOURS, r224_hours=extra, projected_hours=hours,
                limit_hours=LIMIT_HOURS, runtime_pass=hours <= LIMIT_HOURS,
                method='5.38 + measured local R224 N32 seconds * authenticated R7 hidden seconds / floor(min(local R7 plain N32 seconds)) / 3600',
                local_r7_seconds=local, hidden_r7_seconds=hidden,
                limitation=calibration['assumption'])


def launch_command():
    return [str(GPU_PYTHON), '-B', str(Path(__file__).resolve()), 'run']


def watchdog(now, last_progress, gpu, desktop):
    need(now-last_progress < 900, 'idle watchdog: 15 minutes without progress')
    if gpu is not None:
        used, util, power = gpu
        need(all(math.isfinite(x) for x in gpu) and 0 <= util <= 100 and power >= 0,
             'invalid watchdog telemetry')
        if now-last_progress >= 600 and util >= 95 and power < 90 and used >= desktop['total_mib']-4096:
            raise ValueError('WDDM watchdog: 10 minutes without progress near memory limit')


class _TimingTelemetry(monitor._Telemetry):
    """Lossless memory violations abort; unavailable diagnostics only invalidate timing."""
    def __init__(self, env, interval, desktop):
        super().__init__(env, interval)
        self.desktop = desktop
        self.guard_lock = threading.Lock()
        self.started = time.perf_counter()
        self.last_sample = self.started
        self.samples = 0
        self.peak = 0
        self.latest_gpu = None
        self.memory_failure = None
        self.error_count = 0
        self.first_error = self.last_error = None
        self.gap_count = 0
        self.max_gap_seconds = 0
        self.first_gap_elapsed_seconds = None
        self.gap_active = False
        self.sampler_stopped = None

    def _gap_locked(self, now):
        gap = now-self.last_sample
        self.max_gap_seconds = max(self.max_gap_seconds, gap)
        if gap >= SAMPLER_GAP_SECONDS and not self.gap_active:
            self.gap_count += 1
            self.gap_active = True
            if self.first_gap_elapsed_seconds is None:
                self.first_gap_elapsed_seconds = now-self.started

    def _record_error(self, exc):
        with self.guard_lock:
            value = dict(elapsed_seconds=time.perf_counter()-self.started,
                         reason=f'{type(exc).__name__}: {exc}')
            self.error_count += 1
            self.first_error = self.first_error or value
            self.last_error = value
            self.latest_gpu = None  # Do not apply WDDM activity checks using a pre-error sample.

    def _sample(self):
        gpu = monitor._gpu_sample(self.env)
        need(len(gpu) == 3 and math.isfinite(gpu[0]) and gpu[0] >= 0,
             'invalid GPU memory telemetry')
        now = time.perf_counter()
        with self.guard_lock:
            self._gap_locked(now)
            # Consume valid memory even if activity/power parsing is unusable.
            self.peak = max(self.peak, gpu[0])
            try:
                shared.live_memory_guard(gpu[0], self.desktop)
            except ValueError as exc:
                self.memory_failure = self.memory_failure or str(exc)
            need(math.isfinite(gpu[1]) and 0 <= gpu[1] <= 100 and
                 math.isfinite(gpu[2]) and gpu[2] >= 0, 'invalid GPU activity/power telemetry')
            self.samples += 1
            self.last_sample, self.latest_gpu = now, list(gpu)
            self.gap_active = False

    def _run(self):
        while not self.stop.is_set():
            try:
                self._sample()
            except Exception as exc:
                self._record_error(exc)
            self.stop.wait(self.interval)

    def start(self):
        try:
            super().start()
        except Exception as exc:
            self._record_error(exc)

    def close(self):
        try:
            stopped = super().close() if self.thread.ident is not None else False
        except Exception as exc:
            self._record_error(exc)
            stopped = False
        with self.guard_lock:
            # Once a close fails, a later call cannot erase the evidence.
            self.sampler_stopped = stopped if self.sampler_stopped is None else self.sampler_stopped and stopped
        return stopped

    def snapshot(self, final=False):
        now = time.perf_counter()
        with self.guard_lock:
            self._gap_locked(now)
            complete = (self.samples > 0 and not self.error_count and not self.gap_count and
                        self.memory_failure is None and self.sampler_stopped is not False and
                        (not final or self.sampler_stopped is True))
            return dict(complete=complete, sample_count=self.samples, peak_gpu_mib=self.peak,
                        error_count=self.error_count, first_error=self.first_error, last_error=self.last_error,
                        gap_count=self.gap_count, max_gap_seconds=self.max_gap_seconds,
                        first_gap_elapsed_seconds=self.first_gap_elapsed_seconds,
                        gap_threshold_seconds=SAMPLER_GAP_SECONDS, sampler_stopped=self.sampler_stopped,
                        memory_failure=self.memory_failure,
                        latest_gpu=self.latest_gpu if now-self.last_sample < SAMPLER_GAP_SECONDS else None)


def lane_admission():
    import psutil
    from tools.job_runtime import LOGS, job_status
    parents = psutil.Process().parents()
    need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in parents), 'gpu_job lock required')
    job = job_status(LOGS, TAG, expected_command=launch_command(), tail=False)
    need(job['state'] == 'running' and job['pid'] in {p.pid for p in parents}, 'exclusive HB164 lane required')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=5)
    need(not shared.gpu_python_pids(gpu.stdout, {os.getpid(), *[p.pid for p in parents]}),
         'another Python ML process owns GPU')
    desktop = shared.desktop_snapshot()
    shared.live_memory_guard(desktop['used_mib'], desktop)
    return desktop


def validate_measurement(value, ids):
    need(value.get('identity') == TAG and value.get('done') == 32 and
         value.get('state') == 'COMPLETE' and value.get('clean_oof') is False,
         'incomplete R224 measurement')
    rows = value['rows']
    need([r['study'] for r in rows] == ids and len(set(ids)) == 32, 'N32 output identity/order drift')
    for row in rows:
        need(len(row['probabilities']) == 12 and all(math.isfinite(x) and 0 <= x <= 1 for x in row['probabilities']),
             'N32 x 12 finite probabilities required')
        need(row['failed_study_fallback'] is False and row['windows'] == 94 and row['resolution'] == 224 and
             row['forward_precision'] in (['float16'], ['float16', 'float32'], ['float32']) and
             row['fp32_retry'] is ('float32' in row['forward_precision']) and
             math.isfinite(row['seconds']) and row['seconds'] > 0 and
             0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes'] <= 12*GIB,
             'invalid recipe, timing, fallback or memory measurement')


def observed_inference(ns, model, windows):
    """Observe, never rewrite, the public function's RuntimeError fp32 retry."""
    precision = []
    def forward(value):
        precision.append('float16' if ns['torch'].is_autocast_enabled('cuda') else 'float32')
        return model(value)
    result = ns['infer_probs'](forward, windows, 'cuda:0')
    need(precision in (['float16'], ['float16', 'float32'], ['float32']), 'unexpected public inference path')
    return result, precision


def runtime_snapshot(ns):
    import platform
    import pydicom
    torch = ns['torch']
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=str(torch.__version__), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), timm=ns['timm'].__version__,
                numpy=ns['np'].__version__, pydicom=pydicom.__version__,
                device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def worker():
    """Private worker still revalidates admission/lock; CLI cannot bypass either."""
    need(os.environ.get('HB164_CONTROLLER_PID') == str(os.getppid()), 'supervised worker only')
    admission = admitted()
    desktop = lane_admission()
    prepared = io.read_json(WORK/'prepare.json')
    ids, rows = prepared['studies'], prepared['series']
    started = time.perf_counter()
    ns = public_namespace()
    torch, np = ns['torch'], ns['np']
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one visible CUDA GPU required')
    need('4070 Ti SUPER' in torch.cuda.get_device_name(0), '4070 Ti SUPER required')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    runtime = runtime_snapshot(ns)
    torch.cuda.reset_peak_memory_stats()
    arm = ns['ARMS'][0]
    model, res = ns['load_model'](str(CHECKPOINT), arm['arch'], arm['res'], 'cuda:0')
    need(res == 224, 'checkpoint resolution mismatch')
    reader, grouped, measurements = ns['_make_reader'](), group_series(rows), []
    def publish(phase):
        io.journal(WORK/'worker-status.json', stamp(state=phase, done=len(measurements), rows=measurements,
                   runtime=runtime, binding_sha256=admission['binding_sha256'], seconds=time.perf_counter()-started), WORK)
    publish('MODEL_LOADED')
    for uid in ids:
        mark = time.perf_counter()
        vol, mask = ns['build_study'](uid, grouped, str(WORK/'data/train_series'), reader)
        validate_volume(vol, mask)
        windows = ns['gpu_windows'](vol, mask, 94, res, 'cuda:0')
        probabilities, precision = observed_inference(ns, model, windows)
        torch.cuda.synchronize()
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        shared.memory_admission(desktop, [reserved])
        need(probabilities.shape == (12,) and np.isfinite(probabilities).all(), 'nonfinite R224 output')
        measurements.append(dict(study=uid, probabilities=probabilities.tolist(), windows=94,
                                 resolution=res, failed_study_fallback=False, forward_precision=precision,
                                 fp32_retry='float32' in precision, seconds=time.perf_counter()-mark,
                                 peak_allocated_bytes=allocated, peak_reserved_bytes=reserved))
        publish('INFERENCE')
        del windows, probabilities, vol, mask
    publish('COMPLETE')
    return 0


def supervise(desktop):
    import psutil
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1', HB164_CONTROLLER_PID=str(os.getpid()))
    child = job = None
    tracked = {}
    telemetry = _TimingTelemetry(env, .1, desktop)
    status = monitor._StatusWriter(WORK/'status.json')
    started = last_progress = time.perf_counter()
    completed, healthy, rate, peak, last_gpu, last_phase = 0, None, None, 0, None, None
    error, cleanup, code, result = None, [], None, None
    try:
        status.start()
        # stderr/stdout inherit gpu_job's owned log. Live reads use only status JSON.
        child = subprocess.Popen([str(GPU_PYTHON), '-u', '-B', str(Path(__file__).resolve()), '_worker'],
                                 env=env, cwd=ROOT, start_new_session=os.name != 'nt')
        root = psutil.Process(child.pid)
        tracked[root.pid] = root
        job = monitor._WindowsJob(child)
        telemetry.start()
        while child.poll() is None:
            now = time.perf_counter()
            metrics = telemetry.snapshot()
            peak, last_gpu = metrics['peak_gpu_mib'], metrics['latest_gpu']
            need(metrics['memory_failure'] is None, metrics['memory_failure'])
            try:
                current = io.read_json(WORK/'worker-status.json')
            except (OSError, ValueError):
                current = {}
            if current.get('state') == 'MODEL_LOADED' and last_phase is None:
                last_progress, last_phase = now, 'MODEL_LOADED'
            if current.get('done', 0) > completed:
                completed = current['done']
                last_progress = now
                if healthy is None and completed >= 2:
                    seconds = sum(r['seconds'] for r in current['rows'])
                    healthy = completed*60/seconds
                    rate = monitor.RateGuard(healthy, now)
                    # RateGuard's initial count is zero; align it to this warmup boundary.
                    rate.history.clear()
                    rate.history.append((now, completed))
            watchdog(now, last_progress, last_gpu, desktop)
            if rate is not None:
                reason = rate.update(now, completed)
                need(reason is None, reason)
            try:
                processes = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                processes = []
            for process in processes:
                tracked.setdefault(process.pid, process)
            status.submit(stamp(state='RUNNING', done=completed, seconds=now-started,
                                peak_gpu_mib=peak, healthy_studies_per_minute=healthy,
                                telemetry=metrics))
            time.sleep(.1)
        code = child.wait(timeout=2)
        need(code == 0, 'R224 child failed')
        result = io.read_json(WORK/'worker-status.json')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        cleanup = monitor._cleanup(child, tracked, job, psutil)
        telemetry.close()
        metrics = telemetry.snapshot(final=True)
        peak = metrics['peak_gpu_mib']
        error = error or metrics['memory_failure']
        status.submit(stamp(state='FAILED' if error or cleanup else 'STOPPED', error=error,
                            cleanup=cleanup, seconds=time.perf_counter()-started, telemetry=metrics))
        status.close()
        io.receipt(WORK/'supervision.json', stamp(error=error, exit_code=code, cleanup=cleanup,
                   seconds=time.perf_counter()-started, desktop_before_cuda=desktop, peak_gpu_mib=peak,
                   telemetry=metrics,
                   healthy_studies_per_minute=healthy, idle_seconds=900, wddm_seconds=600,
                   throughput_window_seconds=1800, wall_cap=None))
    need(not cleanup, 'owned cleanup incomplete')
    need(metrics['memory_failure'] is None, metrics['memory_failure'])
    return result, metrics


def run():
    value = admitted()
    desktop = lane_admission()
    # Preserve attempts. Reuse requires B to move the prior files deliberately.
    need(not (WORK/'worker-status.json').exists(), 'existing attempt present; do not overwrite')
    result, telemetry = supervise(desktop)
    device_peak_mib = telemetry['peak_gpu_mib']
    ids = shared.cohort()[0]
    validate_measurement(result, ids)
    need(result['binding_sha256'] == value['binding_sha256'], 'measurement admission mismatch')
    maximum = shared.memory_admission(desktop, [r['peak_reserved_bytes'] for r in result['rows']],
                                       device_peak_mib*2**20 if device_peak_mib > 0 else None)
    projected = projection(result['seconds'], value['binding']['transfer_calibration'])
    complete = telemetry['complete']
    if not complete:
        projected['arithmetic_within_limit'] = projected['runtime_pass']
        projected['runtime_pass'] = None  # An incomplete measurement never publishes a timing PASS.
    receipt = stamp(state='MEASURED_REQUIRES_INDEPENDENT_RESULT_REVIEW' if complete else 'HOLD', binding=value['binding'],
                    binding_sha256=value['binding_sha256'], measured_seconds=result['seconds'],
                    measured_studies=32, peak_reserved_bytes=maximum, desktop_before_cuda=desktop,
                    runtime=result['runtime'], fp32_retries=sum(r['fp32_retry'] for r in result['rows']),
                    device_peak_mib=device_peak_mib, projection=projected, telemetry=telemetry,
                    decision=('TIMING_PASS_PENDING_REVIEW' if projected['runtime_pass'] else 'DO_NOT_SUBMIT')
                             if complete else 'HOLD_TELEMETRY_INCOMPLETE')
    io.journal(WORK/'measurement.json', receipt, WORK)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate', 'admit', 'run', '_worker'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--build-review', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == 'prepare':
            result = prepare(args.dry_run)
        elif args.action == 'validate':
            result = validate(args.build_review)
        elif args.action == 'admit':
            result = admit(args.build_review, args.review, args.dry_run)
        elif args.action == 'run':
            need(not args.dry_run, 'use admit --dry-run; run never silently ignores dry-run')
            result = run()
        else:
            need(not args.dry_run, 'worker has no dry-run bypass')
            return worker()
        print(json.dumps({k: result[k] for k in ('identity', 'version', 'state') if k in result}))
        return 2 if result.get('state') == 'HOLD' else 0
    except (OSError, ValueError, ImportError) as exc:
        value = stamp(state='HOLD', action=args.action, reason=str(exc))
        if args.action in ('prepare', 'validate', 'admit'):
            io.receipt(WORK/(args.action+'-hold.json'), value)
        print(json.dumps(value))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
