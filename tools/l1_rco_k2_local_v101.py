"""K2 external B-only local qualification supervisor v1.0.2.

v1.0.2: stream large pinned assets during admission replay; preserve draft1.

No predictor edits. Static register/validate/prepare/admit never touch CUDA.
Only execute, inside the exact gpu_job identity and reviewed admission, starts
the fixed worker child. No wall-time cap, network, Kaggle or submission route.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from tools import job_runtime as jr
from tools import hb150_io_v1 as io
from tools.hb155_watchdog_v1 import collect_owned, terminate_owned
from tools.c85r7_v150_safety import ThroughputGuard
from tools import l1_rco_runtime_v120 as legacy
from tools import strict_variant_gate as base
from tools import strict_variant_gate_k2_v101 as gate
from tools import l1_rco_k2_v101 as build
from tools import l1_rco_k2_worker_v101 as worker
from tools.gpu_queue import validate_plan

VERSION = '1.0.2'
REL = Path('.local/scratch/slot-b/k2-release/fix1/local-wrapper')
STAGE = Path('artifacts/experiments/L1_RCO_K2/v1.0.1-rcof0')
FIX1 = REL.parent
TAG = 'K2_RCOF0_FIX1_LOCAL_WRAPPED_1'
CANDIDATE = 'K2-RCOF0-V1'
REG_SHA = '064ce70162a40b4782b45bcf1f884b3220a10ed091dab1f69d1ec67bde79302c'
WORKER_SHA = 'cfcba1a26dddda54ebfeadd874d58538c1ac358a81e8b45bbabb2563c8b45e1f'
ASSET_SHA = 'a529559ad7efac3e88111b0007338ca931bde3e3a35b8e88b1d09ca3f44e68f3'
DRIVER_PATH = r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
DRIVER_SHA = '712cde267b942a6dbbea6e860a8a53372ad71ee6aadd94cca6bd1a424ab7e14b'
SELF = 'tools/l1_rco_k2_local_v101.py'
SOURCES = [SELF, 'tools/test_l1_rco_k2_local_v101.py', 'tools/job_runtime.py',
           'tools/gpu_queue.py', 'tools/gpu_job.py', 'tools/hb150_io_v1.py',
           'tools/hb155_watchdog_v1.py', 'tools/c85r7_v150_safety.py',
           'tools/l1_rco_runtime_v120.py', 'tools/l1_rco_stage0_v120.py', 'tools/s6plus.py']
GIB = 2**30


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path, root=ROOT):
    path = Path(path).resolve()
    need(path.is_relative_to(root.resolve()), 'evidence outside root')
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path))


def fresh_json(path, value):
    need(not Path(path).exists(), 'create-only evidence already exists')
    io.atomic_json(path, value)


def diagnostic(path, value):
    try:
        return bool(io.atomic_json(path, value, critical=False))
    except BaseException:
        return False  # Diagnostics never prevent owned-process cleanup.


def worker_arguments(root=ROOT):
    out = root / REL / 'run1'
    return ['--assets', str(root / FIX1 / 'assets-v1'), '--manifest-sha', ASSET_SHA,
            '--data', str(root / '.local/exp189-cpu-v120/input/visible_data'),
            '--template', str(root / '.local/exp189-cpu-v120/input/visible_data/sample_submission.csv'),
            '--output', str(out / 'submission.csv'), '--scratch', str(out / 'scratch'),
            '--device', 'cuda:0']


def queue(root=ROOT):
    return [dict(tag=TAG, after=[], continue_on_failure=False,
                 command=[sys.executable, '-X', 'utf8', '-B', str(root / SELF),
                          'execute', '--admission', str(root / REL / 'admission.json')])]


def register(driver, driver_sha, root=ROOT):
    root = root.resolve(); work = root / REL
    need(sha(root / STAGE / 'registration.json') == REG_SHA and
         sha(root / 'tools/l1_rco_k2_worker_v101.py') == WORKER_SHA and
         sha(root / FIX1 / 'assets-v1/asset-manifest.json') == ASSET_SHA, 'exact FIX1 identity')
    driver = Path(driver).resolve()
    need(driver.name.lower() == 'nvdxgdmal64.dll' and sha(driver) == driver_sha, 'installed DLL pin')
    if root == ROOT:
        need(driver == Path(DRIVER_PATH) and driver_sha == DRIVER_SHA, 'B-observed exact installed DLL')
    inputs = base.Inputs(root); material = gate.snapshot(root / STAGE, inputs)
    for n in SOURCES:
        inputs.read(n)
    inputs.recheck()
    value = dict(schema='k2-local-registration/v1', version=VERSION, candidate_id=CANDIDATE,
        workspace=str(root), fixture_only=root != ROOT, stage=STAGE.as_posix(),
        registration_sha256=REG_SHA, worker_sha256=WORKER_SHA, asset_manifest_sha256=ASSET_SHA,
        kernel_sha256=material['kernel_sha256'], input_sha256=inputs.hashes(),
        worker_arguments=worker_arguments(root), queue=queue(root),
        driver=dict(path=str(driver), sha256=driver_sha, technique='s6plus.pin_windows_driver'),
        policy=dict(zero_progress_seconds=900, wddm_seconds=600, utilization=95, power_below=90,
                    vram_fraction=.90, throughput_seconds=1800, throughput_fraction=.25,
                    healthy_measurement_seconds=120, memory_peak_bytes=12*GIB, margin_bytes=2*GIB),
        qualified_for_submission=False, execution_authorized=False)
    validate_plan(value['queue'])
    work.mkdir(parents=True, exist_ok=True)
    fresh_json(work / 'registration.json', value)
    return value


def validate(path, root=ROOT):
    root = root.resolve(); inputs = base.Inputs(root); reg = inputs.json(path)
    need(reg['schema'] == 'k2-local-registration/v1' and reg['version'] == VERSION and
         reg['candidate_id'] == CANDIDATE and reg['workspace'] == str(root) and
         reg['fixture_only'] is (root != ROOT), 'local registration root/schema/fixture')
    need(reg['registration_sha256'] == REG_SHA and reg['worker_sha256'] == WORKER_SHA and
         reg['asset_manifest_sha256'] == ASSET_SHA and reg['stage'] == STAGE.as_posix(), 'FIX1 pin identity')
    need(reg['queue'] == queue(root) and reg['worker_arguments'] == worker_arguments(root), 'fixed command identity')
    for name, pin in reg['input_sha256'].items():
        need(inputs.digest(name) == pin, 'source/asset changed: ' + name)
    need(set(SOURCES) <= set(reg['input_sha256']), 'missing supervisor dependency')
    material = gate.snapshot(root / STAGE, inputs)
    need(material['kernel_sha256'] == reg['kernel_sha256'], 'kernel changed')
    need(sha(reg['driver']['path']) == reg['driver']['sha256'] and
         Path(reg['driver']['path']).name.lower() == 'nvdxgdmal64.dll', 'driver changed')
    if root == ROOT:
        need(reg['driver'] == dict(path=str(Path(DRIVER_PATH)), sha256=DRIVER_SHA,
             technique='s6plus.pin_windows_driver'), 'reviewed DLL environment changed')
    inputs.recheck(); return reg, inputs


def prepare(registration, root=ROOT):
    reg, _ = validate(registration, root); work = root / REL
    value = dict(schema='k2-local-prepare/v1', registration=ref(registration, root),
        queue=reg['queue'], build_gate=(FIX1 / 'build-gate.json').as_posix(),
        wrapper_builder=(REL / 'BUILD-REPORT.md.receipt.json').as_posix(),
        wrapper_review=(REL / 'BUILD-REVIEW.md').as_posix(),
        launch_review=(REL / 'LAUNCH-REVIEW.md').as_posix(),
        resource_claim=(REL / 'resource-claim.json').as_posix(),
        executor='B', runtime_checks=['exact live gpu_job identity', 'exclusive physical4070',
            'CUDA_VISIBLE_DEVICES=0', 'measured memory', 'fresh run directory'],
        fixture_only=root != ROOT, execution_authorized=False)
    fresh_json(work / 'prepare.json', value)
    fresh_json(work / 'queue.json', reg['queue'])
    return value


def authentic_review(path, builder_lane, inputs, required):
    text = base.verdict(inputs.read(path))
    lane, report, actual = base.dispatch(str(path) + '.receipt.json', inputs)
    need(lane != builder_lane and actual == path and report == inputs.read(path), 'independent authentic review')
    need(re.findall(r'^Reviewer-Lane: (\S+)$', text, re.M) == [lane], 'review lane')
    for line in required:
        need(text.splitlines().count(line) == 1, 'review missing/ambiguous binding: ' + line)
    return lane


def check_resource(value):
    need(value['schema'] == 'k2-local-resource/v1' and value['recorded_by'] == 'B' and
         value['candidate_id'] == CANDIDATE and value['tag'] == TAG and
         value['state'] == 'CLAIMED' and value['exclusive'] is True and
         value['physical_index'] == 0 and value['logical_device'] == 'cuda:0' and
         value['gpu_name'] == 'NVIDIA GeForce RTX 4070 Ti SUPER' and
         isinstance(value['reservation'], str) and len(value['reservation']) >= 8 and
         isinstance(value['shared_claim_evidence'], str) and bool(value['shared_claim_evidence']),
         'exclusive B resource claim required')


def authorize(prepared, root=ROOT):
    root = root.resolve(); inputs = base.Inputs(root); spec = inputs.json(prepared)
    need(spec['schema'] == 'k2-local-prepare/v1' and spec['executor'] == 'B' and
         spec['fixture_only'] is (root != ROOT), 'prepare identity')
    regpath = base.local_path(spec['registration']['path'], root)
    need(inputs.digest(regpath) == spec['registration']['sha256'], 'prepare registration changed')
    reg, static_inputs = validate(regpath, root)
    for name in static_inputs.hashes(): inputs.digest(name)
    need(spec['queue'] == queue(root), 'prepared command changed')
    bgpath = base.local_path(spec['build_gate'], root)
    bg = gate.verify_receipt(bgpath, inputs, 'BUILD_READY')
    need(bg['stage'] == STAGE.as_posix() and bg['kernel_sha256'] == reg['kernel_sha256'], 'wrong BUILD_READY')
    builder, report, _ = base.dispatch(spec['wrapper_builder'], inputs)
    label = 'K2-Local-Registration-SHA256: ' + inputs.digest(regpath)
    need(report.decode('utf-8-sig').splitlines().count(label) == 1, 'wrapper builder registration binding')
    reviewer = authentic_review(base.local_path(spec['wrapper_review'], root), builder, inputs,
        [label, 'Candidate-ID: ' + CANDIDATE, 'Local-Wrapper-Build: PASS'])
    resource_path = base.local_path(spec['resource_claim'], root)
    resource = inputs.json(resource_path); check_resource(resource)
    launcher = authentic_review(base.local_path(spec['launch_review'], root), builder, inputs,
        ['K2-Local-Prepare-SHA256: ' + inputs.digest(prepared), label,
         'K2-Build-Gate-SHA256: ' + inputs.digest(bgpath),
         'Resource-Claim-SHA256: ' + inputs.digest(resource_path),
         'Candidate-ID: ' + CANDIDATE, 'Local-CUDA-Launch: PASS'])
    inputs.recheck()
    return dict(schema='k2-local-admission/v1', status='READY_FOR_GPU_JOB',
        registration=ref(regpath, root), prepared=ref(prepared, root),
        builder_lane=builder, reviewer_lane=reviewer, launch_reviewer_lane=launcher,
        fixture_only=root != ROOT, input_sha256=inputs.hashes(), candidate_id=CANDIDATE,
        tag=TAG, qualified_for_submission=False, runtime_checks_pending=True)


def admit(prepared, dry_run=False, root=ROOT):
    try:
        value = authorize(prepared, root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return dict(status='BLOCKED', error=str(exc), execution_authorized=False,
                    qualified_for_submission=False, dry_run=dry_run)
    if not dry_run:
        fresh_json(root / REL / 'admission.json', value)
    return dict(value, dry_run=dry_run, execution_authorized=False)


def replay_admission(path, root=ROOT):
    value = jr.read_json(path)
    need(value['schema'] == 'k2-local-admission/v1' and value['status'] == 'READY_FOR_GPU_JOB' and
         value['fixture_only'] is (root != ROOT), 'actual local admission required')
    for name, pin in value['input_sha256'].items():
        need(sha(root / name) == pin, 'admission input mutation')
    need(sha(root / value['prepared']['path']) == value['prepared']['sha256'], 'prepare mutation')
    current = authorize(root / value['prepared']['path'], root)
    need(current == value, 'admission replay differs')
    return value


def bounded_output(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=3,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    result.check_returncode(); return result.stdout


def bounded_legacy(function):
    # Private namespace, preserving existing parser/policy without an unbounded subprocess.
    namespace = dict(function.__globals__, subprocess=types.SimpleNamespace(check_output=bounded_output))
    return types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)


def telemetry():
    try:
        return bounded_legacy(legacy.gpu_snapshot)()
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None  # Optional ongoing samples cannot independently abort work.


def memory_ok(peak, before, total):
    return all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (peak, before, total)) and \
        peak <= 12*GIB and before + peak + 2*GIB <= total


def check_job(record, own, parent, command, root=ROOT):
    need(record['schema'] == 'gpu-job/v2' and record['tag'] == TAG and record['phase'] == 'running' and
         record['command'] == command and record['command_sha256'] == jr.command_digest(command, root) and
         Path(record['cwd']).resolve() == root.resolve() and record['child_identity'] == own and
         dict(pid=record['pid'], create_time=record['create_time']) == parent,
         'gpu_job command/PID/start identity mismatch')
    need(jr.liveness(own) == jr.liveness(parent) == 'alive', 'gpu_job identity not alive')
    return record


def job_context(own=None):
    own = own or jr.identity()
    process = psutil.Process(own['pid']); parent = jr.identity(process.ppid())
    command = queue()[0]['command']
    need(process.cmdline() == command, 'live supervisor command differs')
    parent_args = psutil.Process(parent['pid']).cmdline()
    path = ROOT / 'artifacts/logs' / TAG / 'job.json'
    last = None
    for _ in range(40):  # gpu_job publishes child_identity immediately after Popen.
        try:
            value = check_job(jr.read_json(path), own, parent, command)
            need(parent_args[-3:] == ['--_run-job', TAG, value['run_id']] and
                 str(ROOT / 'tools/gpu_job.py') in parent_args, 'actual gpu_job parent command')
            return value
        except (OSError, ValueError, KeyError) as exc:
            last = exc
            time.sleep(.05)
    raise ValueError('gpu_job child handoff not authenticated') from last


class Progress:
    """Adapt sealed K2 PID/phase/completed/windows, never mtime as progress."""
    ranks = {'render': 0, 'classifier': 1, 'classifier-chunk': 2, 'complete': 3}

    def __init__(self, started, child, studies=3):
        self.last = self.previous = started
        self.child, self.studies = child, studies
        self.completed, self.windows, self.rank = -1, 0, -1
        self.phase = 'startup'; self.stamp = child['create_time'] - 1
        self.active = 0.; self.baseline_at = 0.; self.baseline_windows = 0
        self.healthy = None; self.guard = None; self.events = 0

    def update(self, now, heartbeat, sample, wall):
        need(now >= self.previous, 'monotonic clock moved backwards')
        if self.phase in ('classifier', 'classifier-chunk'):
            self.active += now - self.previous
        self.previous = now
        if heartbeat is not None:
            h = heartbeat
            need(h['pid'] == self.child['pid'] and type(h['pid']) is int, 'forged progress PID')
            stamp = h['time']; completed = h['completed']; windows = h['windows']; phase = h['phase']
            need(type(stamp) in (int, float) and math.isfinite(stamp) and
                 self.child['create_time']-1 <= stamp <= wall+2 and stamp >= self.stamp,
                 'forged/stale progress timestamp')
            need(type(completed) is int and type(windows) is int and windows >= self.windows and
                 0 <= completed <= self.studies and phase in self.ranks, 'progress counter/schema')
            rank = self.ranks[phase]
            need((phase == 'complete') is (completed == self.studies), 'completion phase count')
            need(completed >= self.completed and (completed > self.completed or rank >= self.rank),
                 'regressing phase/study progress')
            if completed == self.completed and rank < 2:
                need(windows == self.windows, 'windows outside classifier progress')
            meaningful = (completed, rank, windows) != (self.completed, self.rank, self.windows)
            if meaningful:
                self.last = now; self.events += 1
            self.completed, self.rank, self.windows, self.phase, self.stamp = completed, rank, windows, phase, stamp
        # Measure only accumulated classifier-active time. Render/setup pauses it;
        # phase/file touches cannot reset this clock or the inherited low-rate guard.
        span = self.active - self.baseline_at
        if span >= 120:
            rate = (self.windows - self.baseline_windows) / span
            if rate > 0 and (self.healthy is None or rate > self.healthy):
                self.healthy = rate
                if self.guard is None:
                    self.guard = ThroughputGuard(rate, self.active)
                    self.guard.count = self.windows
                else:
                    self.guard.healthy_rate = rate
            self.baseline_at, self.baseline_windows = self.active, self.windows
        reason = legacy.stale_reason({'time': self.last}, now, sample)
        if not reason and self.guard:
            reason = self.guard.update(self.active, self.windows)
        return reason


def _supervise(command, attempt, environment, *, sample=telemetry, poll=5., clock=time.monotonic,
               wall=time.time, sleep=time.sleep, studies=3):
    """Private engine; CPU fixtures inject clocks/samples, production CLI cannot."""
    attempt = Path(attempt); started = clock(); start_wall = wall()
    child = root_process = policy = samples = None; owned = {}; survivors = []
    code, reason = 125, 'supervisor_error'; diagnostics_ok = True
    try:
        try:
            samples = (attempt / 'telemetry.jsonl').open('x', encoding='utf-8')
        except OSError:
            diagnostics_ok = False
        with (attempt / 'worker.log').open('xb', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            root_process = psutil.Process(child.pid)
            identity = jr.identity(child.pid)
            policy = Progress(started, identity, studies)
            diagnostics_ok &= diagnostic(attempt / 'child-identity.json', identity)
            while True:
                collect_owned(root_process, owned)
                if child.poll() is not None:
                    code = child.returncode; reason = 'completed' if code == 0 else 'worker_failed'
                    break
                heartbeat = None
                try:
                    heartbeat = jr.read_json(attempt / 'scratch/status.json')
                except (OSError, ValueError):
                    pass
                current_sample = sample()
                reason = policy.update(clock(), heartbeat, current_sample, wall())
                observation = dict(
                    pid=os.getpid(), child=identity, phase=policy.phase, completed=policy.completed,
                    windows=policy.windows, idle_seconds=clock()-policy.last, telemetry=current_sample,
                    classifier_active_seconds=policy.active, healthy_windows_per_second=policy.healthy,
                    decision=reason, elapsed_seconds=clock()-started)
                diagnostics_ok &= diagnostic(attempt / 'monitor-status.json', observation)
                try:
                    if samples is not None:
                        io._retry(lambda: samples.write(json.dumps(observation, allow_nan=False)+'\n'))
                        io._retry(samples.flush)
                except BaseException:
                    diagnostics_ok = False
                if reason:
                    code = 124; break
                sleep(poll)
    except BaseException as exc:
        reason, code = 'supervisor_' + type(exc).__name__, 125
    finally:
        # Cleanup always precedes any failure receipt, including publication errors.
        try:
            if root_process is not None:
                survivors = terminate_owned(root_process, owned)
        except BaseException as exc:
            survivors = [dict(cleanup_error=type(exc).__name__)]
        if child is not None:
            try:
                if child.poll() is None: child.kill()  # Owned Popen handle, never a guessed PID.
                child.wait(timeout=5)
            except BaseException as exc:
                survivors.append(dict(pid=child.pid, cleanup_error=type(exc).__name__))
        if survivors: code, reason = 125, 'termination_unconfirmed'
        if samples is not None:
            try: samples.close()
            except BaseException: diagnostics_ok = False
        result = dict(schema='k2-local-process/v1', version=VERSION, exit_code=code, reason=reason,
            started_epoch=start_wall, completed_epoch=wall(), full_wall_seconds=clock()-started,
            child=jr.identity(child.pid) if child is not None and child.poll() is None else
                (policy.child if policy else None), owned_descendants=[dict(pid=p, create_time=t) for p,t in owned],
            survivors=survivors, last_phase=policy.phase if policy else 'startup',
            windows=policy.windows if policy else 0, progress_events=policy.events if policy else 0,
            classifier_active_seconds=policy.active if policy else 0,
            healthy_windows_per_second=policy.healthy if policy else None,
            diagnostics_complete=bool(diagnostics_ok), qualified_for_submission=False)
        diagnostic(attempt / 'process-receipt.json', result)
    return result


def validate_result(attempt, reg, process, before):
    need(process['exit_code'] == 0 and not process['survivors'], 'worker did not complete/terminate')
    result = jr.read_json(attempt / 'worker-receipt.json')
    registration = jr.read_json(ROOT / STAGE / 'registration.json')
    need(result['status'] == 'COMPLETE' and result['version'] == worker.VERSION and
         result['contract'] == worker.CONTRACT and result['manifest_sha256'] == ASSET_SHA and
         result['study_count'] == result['prediction_count'] == 3 and
         result['roster_sha256'] == worker.digest(registration['visible_ids']) and
         result['fallbacks']['prediction'] == 0 and not result['fast_path'] and not result['placeholder'] and
         result['output_sha256'] == sha(attempt / 'submission.csv'), 'worker output identity')
    # Existing finite table/roster/column validator; never print individual values.
    build.prediction_table(attempt / 'submission.csv', registration['visible_ids'])
    memory = result['memory']; peak = result['peak_reserved_bytes']
    need(memory_ok(peak, max(before['desktop_bytes'], memory['used_before_bytes']),
                   min(before['total_bytes'], memory['total_bytes'])), 'measured memory/headroom FAIL')
    env = result['environment']
    need(env['device'] == 'cuda:0' and env['cuda_initialized'] and not env['tf32'] and
         not env['autocast'] and '4070 Ti SUPER' in env['gpu'] and 'Windows' in env['platform'] and
         memory['device'] == env['gpu'] and memory['source'] == 'measured before inference', 'local CUDA contract')
    codec = result['codecs']
    need(codec['status'] == 'ACTIVE' and codec['python_abi'] == env['python_abi'] and
         codec['syntaxes'] and all(s['decoded'] is True for s in codec['syntaxes'].values()), 'local codec evidence')
    need(len(result['studies']) == 3 and [x['index'] for x in result['studies']] == list(range(3)) and
         all(x['windows'] > 0 and x['error'] is None and x['prediction_fallback'] == 0 for x in result['studies']),
         'local complete study evidence')
    from tools.l1_rco_k2_admission_v101 import timing_components
    total = max(process['full_wall_seconds'], result['total_seconds'])
    timing = timing_components(result, dict(schema='k2-bootstrap/v1', fixed_startup_seconds=0.,
        total_seconds=total, timing_policy=build.TIMING_POLICY['schema']), total)
    return dict(schema='k2-local-result/v1', status='LOCAL_COMPLETE_PENDING_INDEPENDENT_REVIEW',
        process=process, worker_receipt=ref(attempt / 'worker-receipt.json'),
        submission=ref(attempt / 'submission.csv'), memory_before=before, memory=memory,
        peak_reserved_bytes=peak, local_all_in_timing=timing,
        runtime_projection='NOT_GRANTED; actual Kaggle visible/N32 required', qualified_for_submission=False)


def execute(admission):
    need(os.name == 'nt' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Windows logical cuda0 required')
    need(Path(admission).resolve() == ROOT / REL / 'admission.json', 'fixed admission path')
    admitted = replay_admission(admission)
    reg, _ = validate(ROOT / admitted['registration']['path'])
    need(not reg['fixture_only'], 'fixture cannot execute GPU')
    job = job_context(); own = jr.identity()
    need(not jr.active_jobs(ROOT / 'artifacts/logs', exclude=TAG), 'other active GPU reservation')
    # Initial memory/identity is required admission evidence; later telemetry is optional.
    before = bounded_legacy(legacy.gpu_snapshot)()
    need(before['index'] == 0 and memory_ok(0, before['desktop_bytes'], before['total_bytes']), 'initial measured memory')
    processes = bounded_legacy(legacy.active_ml_processes)()
    permitted = {str(own['pid']), str(job['pid'])}
    need(not [line for line in processes if line.split(',')[0].strip() not in permitted], 'other Python ML GPU owner')
    attempt = ROOT / REL / 'run1'; attempt.mkdir(exist_ok=False)
    token = uuid.uuid4().hex
    ticket = dict(schema='k2-local-ticket/v1', parent=own, token_sha256=jr.digest(token),
                  admission=ref(admission), registration=ref(ROOT / REL / 'registration.json'))
    fresh_json(attempt / 'ticket.json', ticket)
    fresh_json(attempt / 'prelaunch.json', dict(job=job, memory=before, resource='replayed admission',
        command=reg['worker_arguments'], registration_sha256=sha(ROOT / REL / 'registration.json')))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', S6PLUS_PIN_DLL=reg['driver']['path'],
               K2_LOCAL_TOKEN=token, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    command = [sys.executable, '-X', 'utf8', '-B', str(ROOT / SELF), 'child', '--ticket', str(attempt / 'ticket.json')]
    process = _supervise(command, attempt, env)
    try:
        result = validate_result(attempt, reg, process, before)
        need(diagnostic(attempt / 'result.json', result), 'result publication failed; not qualified')
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        diagnostic(attempt / 'result-failure.json', dict(status='FAILED', error=str(exc), process=process))
        return process['exit_code'] or 126


def collect():
    """B-only post-completion collection; no live output-directory reads or GPU calls."""
    status = jr.job_status(ROOT / 'artifacts/logs', TAG, expected_command=queue()[0]['command'])
    need(status['state'] in ('SUCCESS', 'FAILED') and status.get('receipt_kind') == 'bound',
         'collect requires completed bound gpu_job and dead child')
    logdir = ROOT / 'artifacts/logs' / TAG; attempt = ROOT / REL / 'run1'
    job = jr.read_json(logdir / 'job.json'); receipt = jr.read_json(logdir / 'receipt.json')
    need(job['command'] == queue()[0]['command'], 'collection command identity')
    value = dict(schema='k2-local-collection/v1', status=status['state'],
        gpu_job=ref(logdir / 'job.json'), gpu_job_receipt=ref(logdir / 'receipt.json'),
        full_gpu_job_wall_seconds=receipt['elapsed_seconds'], qualified_for_submission=False,
        runtime_projection='NOT_GRANTED; actual Kaggle visible/N32 required')
    if status['state'] == 'SUCCESS':
        result = jr.read_json(attempt / 'result.json')
        value.update(result=ref(attempt / 'result.json'),
                     all_in_wall_seconds=max(receipt['elapsed_seconds'], result['process']['full_wall_seconds']),
                     status='LOCAL_COMPLETE_PENDING_INDEPENDENT_REVIEW')
    fresh_json(ROOT / REL / 'collection.json', value)
    return value


def child(ticket_path):
    need(Path(ticket_path).resolve() == ROOT / REL / 'run1/ticket.json', 'fixed child ticket path')
    ticket = jr.read_json(ticket_path)
    need(ticket['schema'] == 'k2-local-ticket/v1' and ticket['parent'] == jr.identity(os.getppid()) and
         jr.digest(os.environ.get('K2_LOCAL_TOKEN', '')) == ticket['token_sha256'], 'owned child token/parent')
    job_context(ticket['parent'])
    need(sha(ROOT / ticket['admission']['path']) == ticket['admission']['sha256'], 'child admission mutation')
    replay_admission(ROOT / ticket['admission']['path'])
    reg, _ = validate(ROOT / ticket['registration']['path'])
    need(not reg['fixture_only'] and sha(ROOT / ticket['registration']['path']) == ticket['registration']['sha256'] and
         os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('S6PLUS_PIN_DLL') == reg['driver']['path'],
         'child fixed environment/registration')
    # Exact reviewed process-local WinDLL pin, before any torch import. No global environment change.
    from tools import s6plus
    need(s6plus.PINNED_DRIVER_SHA256 == reg['driver']['sha256'], 'process-local driver pin differs')
    diagnostic(ROOT / REL / 'run1/environment-pin.json', dict(driver=reg['driver'], pid=os.getpid(),
        create_time=jr.identity()['create_time'], python=sys.version, cuda_visible_devices='0'))
    sys.argv = [str(ROOT / 'tools/l1_rco_k2_worker_v101.py')] + reg['worker_arguments']
    runpy.run_path(sys.argv[0], run_name='__main__')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('register'); q.add_argument('--driver', required=True); q.add_argument('--driver-sha', required=True)
    for name in ('validate', 'prepare', 'collect'):
        sub.add_parser(name)
    q = sub.add_parser('admit'); q.add_argument('--dry-run', action='store_true')
    q = sub.add_parser('execute'); q.add_argument('--admission', type=Path, required=True)
    q = sub.add_parser('child'); q.add_argument('--ticket', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'register': result = register(args.driver, args.driver_sha)
        elif args.command == 'validate':
            validate(ROOT / REL / 'registration.json'); result = dict(status='STATIC_VALIDATED_NOT_ADMITTED')
        elif args.command == 'prepare': result = prepare(ROOT / REL / 'registration.json')
        elif args.command == 'collect': result = collect()
        elif args.command == 'admit': result = admit(ROOT / REL / 'prepare.json', args.dry_run)
        elif args.command == 'execute': return execute(args.admission)
        else: return child(args.ticket)
        print(json.dumps({k:v for k,v in result.items() if k in ('schema','status','execution_authorized','error','fixture_only','dry_run')}, indent=2))
        return 2 if result.get('status') == 'BLOCKED' else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status='BLOCKED', error=str(exc), execution_authorized=False))); return 2


if __name__ == '__main__':
    raise SystemExit(main())
