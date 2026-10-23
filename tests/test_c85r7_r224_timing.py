"""CPU-only failure-oriented checks for H-B-164 local measurement v1.0.1."""
import ast
import copy
import json
from pathlib import Path

import pytest

from tools import c85r7_r224_timing as t


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'WORK', tmp_path)
    return tmp_path


def desktop(used=1800):
    return dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER', total_mib=16376,
                used_mib=used, uuid='test-only')


def review_fixture():
    binding = dict(build_review_sha256='build', example='bound evidence')
    review = dict(verdict='PASS', reviewer='independent-lane', builder='builder-lane',
                  binding_sha256=t.digest(binding), build_review_sha256='build',
                  smoke_sha256='smoke', execution_authority='B',
                  approved_differences=t.DIFFERENCES, reviewed_functions=list(t.FUNCTIONS))
    return binding, review


def test_source_allowlist_excludes_notebook_and_kaggle_main():
    config, code = t.source_parts()
    tree = ast.parse(code)
    assert {n.name for n in tree.body} == set(t.FUNCTIONS)
    assert all(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body)
    assert 'def main(' not in code and 'find_weight_file' not in code
    assert config['K_EVAL'] == 94 and config['ARMS'][0]['res'] == 224
    assert sum(x[2] for x in config['SLOTS']) == 96


def test_source_mutation_is_rejected(tmp_path):
    source = tmp_path/'changed.py'
    source.write_bytes(t.SOURCE.read_bytes() + b'\nraise RuntimeError("never run")\n')
    with pytest.raises(ValueError, match='source hash'):
        t.source_parts(source)


@pytest.mark.parametrize('change', [
    {'verdict': 'FAIL'}, {'reviewer': 'builder-lane'}, {'reviewer': ''},
    {'binding_sha256': 'stale'}, {'build_review_sha256': 'stale'},
    {'smoke_sha256': 'stale'}, {'execution_authority': 'builder'},
    {'approved_differences': []}, {'reviewed_functions': []},
])
def test_independent_review_rejects_bypass_or_stale_evidence(change):
    binding, review = review_fixture()
    t.validate_review(review, binding, 'smoke')
    with pytest.raises(ValueError):
        t.validate_review(dict(review, **change), binding, 'smoke')


def test_admit_dry_run_still_requires_review(work, monkeypatch):
    binding, _ = review_fixture()
    monkeypatch.setattr(t, 'input_binding', lambda *a: binding)
    t.io.atomic_json(work/'validate.json', dict(state='CPU_VALIDATED', binding=binding))
    with pytest.raises(ValueError, match='review required'):
        t.admit('build', None, dry_run=True)
    assert not (work/'admission.json').exists()


def test_dry_run_cannot_be_used_as_admission(work):
    t.io.atomic_json(work/'admission.json', dict(state='DRY_RUN_PASS'))
    with pytest.raises(ValueError, match='dry-run'):
        t.admitted()


def test_build_review_required():
    with pytest.raises(ValueError, match='build review'):
        t.input_binding(None)


def test_calibration_authenticates_same_n32_and_conservative_scale():
    value = t.transfer_calibration()
    assert value['conservative_plain_seconds'] == 188
    assert value['r7_hidden_seconds'] == pytest.approx(13048.471554289028)
    assert '1300' in value['assumption'] and 'PENDING' in value['source']
    assert value['scale'] > 69


def test_calibration_changed_receipt_is_binding_fail(tmp_path, monkeypatch):
    altered = tmp_path/'paired.json'
    altered.write_bytes(t.R7_PAIRED.read_bytes()+b'\n')
    monkeypatch.setattr(t, 'R7_PAIRED', altered)
    with pytest.raises(ValueError, match='hash changed'):
        t.transfer_calibration()


def synthetic_calibration(local=188, hidden=13048):
    return dict(conservative_plain_seconds=local, r7_hidden_seconds=hidden, assumption='synthetic only')


def test_additive_runtime_gate_boundary_and_overflow():
    cal = synthetic_calibration()
    seconds = (8.5-5.38)*3600*188/13048
    assert t.projection(seconds, cal)['projected_hours'] == pytest.approx(8.5)
    assert t.projection(seconds, cal)['runtime_pass']
    assert not t.projection(seconds+1, cal)['runtime_pass']


@pytest.mark.parametrize('seconds,local,n', [(0, 10, 32), (float('nan'), 10, 32),
                                           (1, 0, 32), (1, 10, 31), (1, True, 32)])
def test_invalid_projection_fails_closed(seconds, local, n):
    with pytest.raises(ValueError):
        t.projection(seconds, synthetic_calibration(local=local), n)


def test_measured_memory_not_fixed_ten_gib():
    assert t.shared.memory_admission(desktop(), [10.9*t.GIB]) == 10.9*t.GIB
    with pytest.raises(ValueError, match='12 GiB'):
        t.shared.memory_admission(desktop(), [12.01*t.GIB])
    with pytest.raises(ValueError, match='headroom'):
        t.shared.memory_admission(desktop(3000), [12*t.GIB])
    with pytest.raises(ValueError):
        t.shared.memory_admission(desktop(), [float('nan')])


def test_total_device_peak_included_in_memory_gate():
    with pytest.raises(ValueError, match='aggregate'):
        t.shared.memory_admission(desktop(), [10*t.GIB], total_peak=(1800*2**20)+12.1*t.GIB)


def test_watchdogs_have_progress_semantics_and_no_total_wall_cap():
    t.watchdog(1000000, 999999, [3000, 100, 200], desktop())
    with pytest.raises(ValueError, match='idle'):
        t.watchdog(901, 0, [4000, 100, 200], desktop())
    with pytest.raises(ValueError, match='WDDM'):
        t.watchdog(600, 0, [13000, 100, 80], desktop())
    t.watchdog(599, 0, [13000, 100, 80], desktop())


def test_throughput_collapse_even_with_some_progress():
    guard = t.monitor.RateGuard(1.0, 0)
    for sec in range(60, 1800, 60):
        assert guard.update(sec, sec//600) is None
    assert '25 percent' in guard.update(1800, 3)


def valid_measurement():
    ids = [str(i) for i in range(32)]
    value = t.stamp(state='COMPLETE', done=32, rows=[dict(study=i, probabilities=[.5]*12,
                        windows=94, resolution=224, failed_study_fallback=False,
                        forward_precision=['float16'], fp32_retry=False, seconds=1,
                        peak_allocated_bytes=t.GIB, peak_reserved_bytes=2*t.GIB) for i in ids])
    return ids, value


@pytest.mark.parametrize('defect', ['nan', 'missing', 'duplicate', 'fallback', 'resolution', 'memory'])
def test_measurement_rejects_invalid_output(defect):
    ids, value = valid_measurement()
    t.validate_measurement(value, ids)
    if defect == 'nan':
        value['rows'][0]['probabilities'][0] = float('nan')
    elif defect == 'missing':
        value['rows'].pop()
    elif defect == 'duplicate':
        value['rows'][1]['study'] = ids[0]
    elif defect == 'fallback':
        value['rows'][0]['failed_study_fallback'] = True
    elif defect == 'resolution':
        value['rows'][0]['resolution'] = 384
    elif defect == 'memory':
        value['rows'][0]['peak_reserved_bytes'] = 13*t.GIB
    with pytest.raises(ValueError):
        t.validate_measurement(value, ids)


def test_desktop_cg_apps_allowed_foreign_python_blocked():
    output = '42, C:\\Windows\\explorer.exe\n43, C:\\Apps\\chrome.exe\n44, C:\\Python\\python.exe\n'
    assert t.shared.gpu_python_pids(output, {42}) == [44]
    assert t.shared.gpu_python_pids(output, {44}) == []


def test_actual_desktop_snapshot_parser_and_injected_foreign_python():
    output = (t.SOURCE.parent/'nvidia-apps-review.txt').read_text()
    assert 'explorer.exe' in output and 'NVIDIA Overlay.exe' in output
    assert t.shared.gpu_python_pids(output, set()) == []
    assert t.shared.gpu_python_pids(output+'\n987654, C:\\Python\\python.exe\n', set()) == [987654]


def test_cpu_cli_hold_does_not_probe_gpu(work, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU command touched GPU')
    monkeypatch.setattr(t, 'lane_admission', forbidden)
    monkeypatch.setattr(t.shared, 'desktop_snapshot', forbidden)
    assert t.main(['validate']) == 2
    assert t.main(['admit', '--dry-run']) == 2
    assert t.main(['run', '--dry-run']) == 2


def test_private_worker_cannot_bypass_supervisor(monkeypatch):
    monkeypatch.delenv('HB164_CONTROLLER_PID', raising=False)
    with pytest.raises(ValueError, match='supervised worker'):
        t.worker()


def test_queue_command_pins_gpu_interpreter_and_entrypoint():
    from tools.gpu_queue import validate_plan
    command = t.launch_command()
    assert command[0] == r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe'
    assert command[-1] == 'run'
    validate_plan([dict(tag=t.TAG, command=command)])


def test_changed_admission_binding_blocks_before_gpu(work, monkeypatch):
    binding, _ = review_fixture()
    t.io.atomic_json(work/'admission.json', dict(state='ADMITTED', binding=dict(binding,
        build_review='build'), binding_sha256='old'))
    monkeypatch.setattr(t, 'input_binding', lambda *a: {'changed': True})
    with pytest.raises(ValueError, match='input drift'):
        t.admitted()


@pytest.mark.parametrize('retry', [False, True])
def test_precision_observer_preserves_result_and_records_retry(retry):
    from types import SimpleNamespace
    state = {'autocast': True, 'calls': 0}
    def model(value):
        state['calls'] += 1
        if retry and state['autocast']:
            raise RuntimeError('synthetic fp16 engine failure')
        return value
    def infer(forward, windows, device):
        try:
            return forward(windows)
        except RuntimeError:
            state['autocast'] = False
            return forward(windows)
    ns = dict(torch=SimpleNamespace(is_autocast_enabled=lambda device: state['autocast']), infer_probs=infer)
    result, precision = t.observed_inference(ns, model, 'unchanged')
    assert result == 'unchanged' and state['calls'] == (2 if retry else 1)
    assert precision == (['float16', 'float32'] if retry else ['float16'])


@pytest.mark.parametrize('exit_code,cleanup', [(1, []), (0, ['synthetic cleanup failure'])])
def test_supervisor_always_cleans_owned_process_and_fails_closed(work, monkeypatch, exit_code, cleanup):
    from types import SimpleNamespace
    import psutil
    cleaned, events = [], []
    child = SimpleNamespace(pid=123456, poll=lambda: exit_code, wait=lambda timeout: exit_code)
    process = SimpleNamespace(pid=child.pid)
    telemetry = SimpleNamespace(start=lambda: None, close=lambda: True,
        snapshot=lambda **kw: dict(complete=True, peak_gpu_mib=3000, memory_failure=None, latest_gpu=None))
    writer = SimpleNamespace(start=lambda: events.append('start'),
                             submit=lambda value: events.append(value['state']),
                             close=lambda: events.append('close'))
    monkeypatch.setattr(t.subprocess, 'Popen', lambda *a, **kw: child)
    monkeypatch.setattr(psutil, 'Process', lambda pid: process)
    monkeypatch.setattr(t, '_TimingTelemetry', lambda *a: telemetry)
    monkeypatch.setattr(t.monitor, '_StatusWriter', lambda *a: writer)
    monkeypatch.setattr(t.monitor, '_WindowsJob', lambda actual: 'owned-job')
    def do_cleanup(actual, tracked, job, module):
        cleaned.append((actual, tracked.copy(), job))
        return cleanup
    monkeypatch.setattr(t.monitor, '_cleanup', do_cleanup)
    t.io.atomic_json(work/'worker-status.json', {'state': 'COMPLETE'})
    with pytest.raises(ValueError, match='child failed' if exit_code else 'cleanup incomplete'):
        t.supervise(desktop())
    assert cleaned == [(child, {child.pid: process}, 'owned-job')]
    assert events[0] == 'start' and events[-2:] == ['FAILED', 'close']
    receipt = t.io.read_json(work/'supervision.json')
    assert receipt['cleanup'] == cleanup and receipt['exit_code'] == exit_code


def test_telemetry_api_failure_recovers_but_latches_incomplete(monkeypatch):
    telemetry = t._TimingTelemetry({}, .1, desktop())
    calls = []
    def sample(env):
        calls.append(env)
        if len(calls) == 1:
            raise TimeoutError('synthetic unavailable telemetry API')
        telemetry.stop.set()
        return [3000, 50, 120]
    monkeypatch.setattr(t.monitor, '_gpu_sample', sample)
    monkeypatch.setattr(telemetry.stop, 'wait', lambda seconds: None)
    telemetry._run()
    result = telemetry.snapshot()
    assert len(calls) == 2 and result['sample_count'] == 1
    assert result['peak_gpu_mib'] == 3000 and result['memory_failure'] is None
    assert result['error_count'] == 1 and not result['complete']
    assert 'TimeoutError' in result['first_error']['reason']


def test_telemetry_memory_breach_is_latched_even_with_bad_activity(monkeypatch):
    telemetry = t._TimingTelemetry({}, .1, desktop())
    monkeypatch.setattr(t.monitor, '_gpu_sample', lambda env: [15000, float('nan'), 80])
    with pytest.raises(ValueError, match='activity'):
        telemetry._sample()
    result = telemetry.snapshot()
    assert result['peak_gpu_mib'] == 15000
    assert 'live VRAM guard' in result['memory_failure'] and not result['complete']


def test_telemetry_stale_sample_and_no_samples_cannot_pass(monkeypatch):
    telemetry = t._TimingTelemetry({}, .1, desktop())
    assert not telemetry.snapshot()['complete']
    monkeypatch.setattr(t.monitor, '_gpu_sample', lambda env: [3000, 50, 120])
    telemetry._sample()
    assert telemetry.snapshot()['complete']
    monkeypatch.setattr(t.time, 'perf_counter', lambda: telemetry.last_sample + 31)
    result = telemetry.snapshot()
    assert result['gap_count'] == 1 and result['max_gap_seconds'] >= 31
    assert result['latest_gpu'] is None and not result['complete']


def test_telemetry_failed_start_and_shutdown_are_nonfatal_and_incomplete(monkeypatch):
    telemetry = t._TimingTelemetry({}, .1, desktop())
    def fail_start():
        raise RuntimeError('synthetic thread start failure')
    monkeypatch.setattr(telemetry.thread, 'start', fail_start)
    telemetry.start()
    assert not telemetry.close()
    result = telemetry.snapshot(final=True)
    assert not result['complete'] and result['sampler_stopped'] is False
    assert result['error_count'] == 1 and 'start failure' in result['last_error']['reason']


@pytest.mark.parametrize('memory_breach', [False, True])
def test_supervisor_preserves_inference_on_telemetry_error_but_aborts_measured_breach(work, monkeypatch, memory_breach):
    from types import SimpleNamespace
    import psutil
    polls, cleaned = iter([None, 0]), []
    child = SimpleNamespace(pid=123456, poll=lambda: next(polls), wait=lambda timeout: 0)
    process = SimpleNamespace(pid=child.pid, children=lambda **kw: [])
    metrics = dict(complete=False, peak_gpu_mib=15000 if memory_breach else 0,
                   error_count=1, last_error='synthetic API failure', latest_gpu=None,
                   memory_failure='live VRAM guard: synthetic measured breach' if memory_breach else None)
    telemetry = SimpleNamespace(start=lambda: None, close=lambda: True,
                                snapshot=lambda **kw: metrics.copy())
    writer = SimpleNamespace(start=lambda: None, submit=lambda value: None, close=lambda: None)
    monkeypatch.setattr(t.subprocess, 'Popen', lambda *a, **kw: child)
    monkeypatch.setattr(psutil, 'Process', lambda pid: process)
    monkeypatch.setattr(t, '_TimingTelemetry', lambda *a: telemetry)
    monkeypatch.setattr(t.monitor, '_StatusWriter', lambda *a: writer)
    monkeypatch.setattr(t.monitor, '_WindowsJob', lambda actual: 'owned-job')
    monkeypatch.setattr(t.monitor, '_cleanup', lambda *a: cleaned.append(a[0]) or [])
    monkeypatch.setattr(t.time, 'sleep', lambda seconds: None)
    t.io.atomic_json(work/'worker-status.json', {'state': 'COMPLETE'})
    if memory_breach:
        with pytest.raises(ValueError, match='measured breach'):
            t.supervise(desktop())
    else:
        result, observed = t.supervise(desktop())
        assert result['state'] == 'COMPLETE' and observed['error_count'] == 1
    assert cleaned == [child]
    receipt = t.io.read_json(work/'supervision.json')
    assert receipt['telemetry']['complete'] is False
    assert bool(receipt['error']) is memory_breach


@pytest.mark.parametrize('complete', [False, True])
def test_final_measurement_with_incomplete_telemetry_is_hold_never_timing_pass(work, monkeypatch, complete):
    ids, result = valid_measurement()
    result.update(binding_sha256='bound', seconds=100, runtime={'synthetic': True})
    admission = dict(binding_sha256='bound', binding=dict(transfer_calibration=synthetic_calibration()))
    metrics = dict(complete=complete, peak_gpu_mib=4000 if complete else 0, error_count=0 if complete else 1)
    monkeypatch.setattr(t, 'admitted', lambda: admission)
    monkeypatch.setattr(t, 'lane_admission', desktop)
    monkeypatch.setattr(t, 'supervise', lambda value: (result, metrics))
    monkeypatch.setattr(t.shared, 'cohort', lambda: (ids, []))
    measurement = t.run()
    saved = t.io.read_json(work/'measurement.json')
    assert measurement == saved
    if complete:
        assert measurement['decision'] == 'TIMING_PASS_PENDING_REVIEW'
        assert measurement['projection']['runtime_pass'] is True
    else:
        assert measurement['state'] == 'HOLD'
        assert measurement['decision'] == 'HOLD_TELEMETRY_INCOMPLETE'
        assert measurement['projection']['runtime_pass'] is None
        assert measurement['projection']['arithmetic_within_limit'] is True
