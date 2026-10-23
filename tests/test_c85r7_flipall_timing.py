import math
import pytest
from tools import c85r7_flipall_timing as t
from tools import c85r7_flipall_measure as m


def desktop(used=2):
    return dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER', total_mib=16376,
                used_mib=used*1024, uuid='GPU-test')


def test_projection_gate_and_conservative_missing_breakdown():
    assert t.projection(100, 200, 4*3600)['runtime_gate'] == 'PASS'
    assert t.projection(100, 201, 4*3600)['runtime_gate'] == 'DO_NOT_SUBMIT'
    assert t.projection(100, 200, 5*3600, 2*3600)['projected_hidden_seconds'] == 7*3600
    assert t.projection(100, 50, 5*3600)['projected_hidden_seconds'] == 5*3600


@pytest.mark.parametrize('bad', [0, -1, math.nan, math.inf])
def test_no_invalid_baseline_or_timing(bad):
    with pytest.raises(ValueError):
        t.projection(1, 1, bad)
    with pytest.raises(ValueError):
        t.projection(bad, 1, 1)


def test_bad_phase_breakdown():
    with pytest.raises(ValueError):
        t.projection(1, 2, 1, 2)


def test_no_review_bypass():
    with pytest.raises(ValueError, match='independent PASS'):
        t.admission({})
    with pytest.raises(ValueError, match='bind exact'):
        t.admission(dict(verdict='PASS', independent_reviewer='another lane', bindings={}))


def test_memory_gate_binds_four_fold_real_shape(monkeypatch):
    monkeypatch.setattr(t, 'bindings', lambda: {'x': 'abc'})
    monkeypatch.setattr(t.b, 'pin', lambda *a: None)
    monkeypatch.setattr(t.b, 'read', lambda *a: dict(status='PASS', full_shape=False))
    value = dict(verdict='PASS', independent_reviewer='reviewer', bindings={'x':'abc'},
                 review_path='review', review_sha256='x', memory_receipt_path='memory', memory_receipt_sha256='x')
    with pytest.raises(ValueError, match='full-shape'):
        t.admission(value)


def test_progress_repeated_heartbeat_is_not_new_commit():
    assert t.progress('PROGRESS worker 0 of 32 {"done": 3, "scored": 3}') == (0, 3)
    assert t.progress('PROGRESS worker 0 of 32 (starting)') is None
    assert t.progress('WORKER_START 0 studies 32') is None


def test_actual_desktop_snapshot_python_only():
    # Actual 2026-10-15 query: desktop apps and one occupied Python GPU process.
    snapshot = (t.b.SCRATCH/'nvidia-compute.csv').read_text()
    assert t.gpu_python_pids(snapshot, set()) == [44812]
    assert t.gpu_python_pids(snapshot, {44812}) == []


def test_real_metadata_cohort_has_exactly_32_no_labels():
    ids, rows = t.cohort()
    assert len(ids) == len(set(ids)) == 32
    assert set(r['StudyInstanceUID'] for r in rows) == set(ids)
    assert set(rows[0]) == {'StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane'}


def test_selective_archive_staging_and_rerun_identity(monkeypatch, tmp_path):
    import zipfile
    archive = tmp_path/'input.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('train_series/1.2/3.4/5.6.dcm', b'synthetic DICOM bytes')
        z.writestr('train_series/9/8/7.dcm', b'not selected')
    monkeypatch.setattr(t, 'ZIP', archive)
    monkeypatch.setattr(t, 'WORK', tmp_path/'work')
    monkeypatch.setattr(t, 'cohort', lambda: (['1.2'], [{'StudyInstanceUID':'1.2', 'SeriesInstanceUID':'3.4'}]))
    assert t.prepare_data() == ['1.2']
    assert t.prepare_data() == ['1.2']
    assert len(t.b.read(t.WORK/'cohort.json')['files']) == 1
    (t.WORK/'data/train_series/1.2/3.4/5.6.dcm').write_bytes(b'changed')
    with pytest.raises(ValueError, match='staged DICOM changed'):
        t.prepare_data()


@pytest.mark.parametrize('case', ['complete', 'child_failure', 'memory_abort'])
def test_supervisor_real_cpu_child_and_owned_cleanup(monkeypatch, tmp_path, case):
    import subprocess
    import sys
    import psutil
    monkeypatch.setattr(t, 'WORK', tmp_path)
    monkeypatch.setattr(t, 'desktop_snapshot', desktop)
    # Actual subprocess, status writer, Job Object and cleanup. Only GPU telemetry
    # and the inference command are replaced; no CUDA or nvidia-smi is invoked.
    monkeypatch.setattr(t.monitor, '_gpu_sample', lambda env: (14500 if case == 'memory_abort' else 100, 0, 0))
    real_popen = subprocess.Popen
    children = []
    def cpu_child(command, **kwargs):
        source = 'import time; time.sleep(1.5); raise SystemExit(%d)' % (2 if case == 'child_failure' else 0)
        if case == 'memory_abort':
            source = 'import time; time.sleep(30)'
        child = real_popen([sys.executable, '-B', '-c', source], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(t.subprocess, 'Popen', cpu_child)
    if case == 'complete':
        assert t.supervise('plain', 1) > 0
    else:
        with pytest.raises(ValueError, match='R7 arm failed' if case == 'child_failure' else 'live VRAM guard'):
            t.supervise('plain', 1)
    receipt = t.b.read(tmp_path/'plain-timing.json')
    assert receipt['cleanup'] == []
    assert bool(receipt['error']) == (case != 'complete')
    assert all(child.poll() is not None and not psutil.pid_exists(child.pid) for child in children)


@pytest.mark.parametrize('used,peak,ok', [(2, 10.9, True), (1, 12, True), (1, 12.5, False),
                                          (3, 12, False), (2, 0, False), (math.nan, 6, False)])
def test_measured_peak_and_desktop_headroom(used, peak, ok):
    if ok:
        t.memory_admission(desktop(used), [peak*t.GIB])
    else:
        with pytest.raises(ValueError):
            t.memory_admission(desktop(used), [peak*t.GIB])


def test_auxiliary_contexts_count_against_headroom():
    with pytest.raises(ValueError, match='headroom'):
        t.memory_admission(desktop(3), [10*t.GIB], 14.5*t.GIB)


def test_cross_mode_max_peak_applies_to_every_desktop_snapshot():
    arms = {name: dict(desktop_before_cuda=desktop(used), rows=[dict(peak_reserved_bytes=peak*t.GIB)],
                       peak_total_vram_bytes=(used+peak)*t.GIB)
            for name, used, peak in [('plain', 3, 10), ('flip', 2, 11)]}
    for arm in arms.values():
        t.memory_admission(arm['desktop_before_cuda'], [arm['rows'][0]['peak_reserved_bytes']],
                           arm['peak_total_vram_bytes'])
    with pytest.raises(ValueError, match='headroom'):
        t.paired_memory_admission(arms)


@pytest.mark.parametrize('ratio,peak,status', [(1, 10, 'PASS_LOCAL_GATES_ONLY'), (3, 8, 'DO_NOT_SUBMIT'), (1, 10.001, 'DO_NOT_SUBMIT')])
def test_joint_gate_violations(ratio, peak, status):
    result = t.gate(1, ratio, {'hidden_seconds':5*3600, 'r7_seconds':2*3600}, peak*t.GIB)
    assert result['status'] == status
    assert result['qualified_for_submission'] is False


@pytest.mark.parametrize('windows,chunk,ok', [(1024,192,True), (1025,192,True), (100000,192,True), (1,193,False), (0,96,False)])
def test_window_admission_no_truncation(windows, chunk, ok):
    if ok:
        m.window_gate(windows,chunk)
    else:
        with pytest.raises(SystemExit, match='DO_NOT_SUBMIT'):
            m.window_gate(windows,chunk)


def test_real_frozen_diagnostic_instrumentation(tmp_path):
    import importlib.util
    files = t.b.unpack(t.b.read(t.b.OUT/'kernel/release.ipynb'))[0]
    for name, raw in files.items():
        t.b.write(tmp_path/name, raw)
    spec = importlib.util.spec_from_file_location('hb153_frozen_adapter', tmp_path/'exp169_r7_port.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    path = adapter.diagnostic_source(tmp_path/'exp169_r7_src', tmp_path/'diagnostic')
    worker = (path/'compact_infer.py').read_text(encoding='utf-8')
    patched = m.instrument(worker)
    assert 'torch.cuda.reset_peak_memory_stats()' in patched
    assert 'hb153.commit(' in patched and 'hb153.window_gate(len(item["images"]), a.chunk)' in patched
    assert 'a.chunk = 96 if' in patched
    compile(patched, '<real-worker>', 'exec')
    with pytest.raises(ValueError, match='anchor mismatch'):
        m.instrument(patched)


def test_commit_mock_cuda_cumulative_peaks(tmp_path):
    from types import SimpleNamespace
    cuda = SimpleNamespace(synchronize=lambda:None, max_memory_reserved=lambda:5*t.GIB,
                           max_memory_allocated=lambda:4*t.GIB, get_device_name=lambda:'4070 Ti SUPER')
    rows = []
    m.commit(SimpleNamespace(cuda=cuda), tmp_path/'status.json', rows, '1.2', 192, 192,
             t.time.perf_counter(), 1., 2., 3.)
    saved = t.b.read(tmp_path/'status.json')
    assert saved['done'] == 1 and saved['rows'][0]['peak_reserved_bytes'] == 5*t.GIB
    assert saved['rows'][0]['peak_allocated_bytes'] == 4*t.GIB
    cuda.max_memory_reserved = lambda:12.5*t.GIB
    with pytest.raises(SystemExit, match='12 GiB'):
        m.commit(SimpleNamespace(cuda=cuda), tmp_path/'status.json', rows, '2.3', 192, 192,
                 t.time.perf_counter(), 1., 2., 3.)
    assert t.b.read(tmp_path/'status.json')['done'] == 1


def simulated_run(monkeypatch, tmp_path, ratio=1.2, peak=6*t.GIB, fail_prepare=False, real_inputs=False):
    monkeypatch.setattr(t, 'WORK', tmp_path)
    monkeypatch.setattr(t, 'desktop_snapshot', desktop)
    monkeypatch.setattr(t, 'bindings', lambda:{'frozen':'test'})
    if not real_inputs:
        monkeypatch.setattr(t.baseline_source, 'baseline', lambda:{'hidden_seconds':5*3600, 'r7_seconds':2*3600})
    monkeypatch.setattr(t, 'lane_admission', lambda memory:'GPU-4070-same')
    memory = tmp_path/'memory.json'
    t.io.atomic_json(memory, {'plain_peak_reserved_bytes':5*t.GIB})
    ids = t.cohort()[0] if real_inputs else [str(i) for i in range(32)]
    calls = []
    def supervise(arm, healthy, deadline):
        calls.append((arm, deadline))
        if arm == 'prepare':
            if fail_prepare:
                raise ValueError('DO_NOT_SUBMIT: preparation failed')
            t.io.atomic_json(tmp_path/'cohort.json', {'studies':ids})
            return 3.
        flip = arm.startswith('flip')
        # The frozen controller cost-sorts assignments, rather than preserving
        # request order. Both arms must retain the SAME resulting order.
        rows = [dict(study=uid, windows=200, batch_windows=192, gpu_seconds=ratio if flip else 1.,
                     peak_reserved_bytes=peak, peak_allocated_bytes=4*t.GIB) for uid in reversed(ids)]
        t.io.atomic_json(tmp_path/(arm+'-measure-status.json'),
                         dict(identity='HB153-v2', done=32, rows=rows, device='4070 Ti SUPER'))
        t.io.atomic_json(tmp_path/(arm+'-timing.json'), dict(desktop_before_cuda=desktop(), peak_gpu_mib=8000))
        return 32*(ratio if flip else 1.)
    monkeypatch.setattr(t, 'supervise', supervise)
    return dict(memory_receipt_path=str(memory), healthy_studies_per_minute=2), calls


@pytest.mark.parametrize('ratio,peak,passes', [(1.2,6,True), (3,6,False), (1.2,10.01,False)])
def test_run_always_projects_and_gates(monkeypatch, tmp_path, ratio, peak, passes):
    # Isolate T4 projection from local measurement validation for the synthetic T4 gate case.
    value, calls = simulated_run(monkeypatch, tmp_path, ratio, peak*t.GIB)
    if peak > 10:
        monkeypatch.setattr(t, 'measurements', lambda arm,ids:t.b.read(tmp_path/(arm+'-measure-status.json')))
    if passes:
        assert t.run(value)['status'] == 'PASS_LOCAL_GATES_ONLY'
    else:
        with pytest.raises(ValueError, match='DO_NOT_SUBMIT'):
            t.run(value)
    assert [a for a,d in calls] == ['prepare', *t.ORDER]
    assert len({d for a,d in calls}) == 1
    paired = t.b.read(tmp_path/'paired.json')
    assert paired['status'] == ('PASS_LOCAL_GATES_ONLY' if passes else 'DO_NOT_SUBMIT')
    assert paired['projected_hidden_seconds'] == 5*3600+(ratio-1)*2*3600
    with pytest.raises(ValueError, match='preserve previous attempt'):
        t.run(value)


def test_preparation_failure_preserves_attempt(monkeypatch, tmp_path):
    value, calls = simulated_run(monkeypatch,tmp_path,fail_prepare=True)
    with pytest.raises(ValueError, match='preparation failed'):
        t.run(value)
    assert [a for a,d in calls] == ['prepare']
    assert t.b.read(tmp_path/'attempt.json')['state'] == 'ABORTED_DO_NOT_SUBMIT'
    assert not (tmp_path/'paired.json').exists()


def test_idle_watchdog_covers_preparation_cpu_child(monkeypatch, tmp_path):
    import subprocess
    import sys
    monkeypatch.setattr(t, 'WORK', tmp_path)
    monkeypatch.setattr(t, 'desktop_snapshot', desktop)
    monkeypatch.setattr(t.monitor, '_gpu_sample', lambda env:(100,0,0))
    popen = subprocess.Popen
    children=[]
    def child(command, **kwargs):
        assert '_prepare' in command
        p=popen([sys.executable,'-B','-c','import time; time.sleep(30)'], **kwargs)
        children.append(p)
        return p
    monkeypatch.setattr(t.subprocess,'Popen',child)
    monkeypatch.setattr(t, 'IDLE_SECONDS', .3)
    with pytest.raises(ValueError, match='idle watchdog'):
        t.supervise('prepare',1)
    assert all(p.poll() is not None for p in children)
    assert t.b.read(tmp_path/'prepare-timing.json')['error']


def test_throughput_collapse_gate():
    guard=t.monitor.RateGuard(4,0)
    assert guard.update(1799,1) is None
    assert guard.update(1800,2) is not None


@pytest.mark.parametrize('signal', ['study', 'gpu'])
def test_slow_progressing_study_survives_multiple_idle_periods(monkeypatch, tmp_path, signal):
    import subprocess
    import sys
    monkeypatch.setattr(t, 'WORK', tmp_path)
    monkeypatch.setattr(t, 'desktop_snapshot', desktop)
    monkeypatch.setattr(t, 'IDLE_SECONDS', .5)
    monkeypatch.setattr(t.monitor, '_gpu_sample', lambda env: (3000, 40 if signal == 'gpu' else 0, 100))
    popen = subprocess.Popen
    children = []
    def child(command, **kwargs):
        source = ('import time\nfor i in range(15):\n'
                  ' print(\'PROGRESS worker 0 of 32 {"done": %d}\' % (i+1), flush=True)\n'
                  ' time.sleep(.1)\n') if signal == 'study' else 'import time; time.sleep(1.5)'
        process = popen([sys.executable, '-B', '-c', source], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(t.subprocess, 'Popen', child)
    assert t.supervise('plain', 100000, attempt_started=t.time.time()-10000) >= 1.5
    receipt = t.b.read(tmp_path/'plain-timing.json')
    assert receipt['error'] is None and receipt['cleanup'] == []
    assert all(p.poll() == 0 for p in children)


def test_idle_is_dual_signal_and_has_no_total_duration_limit():
    t.idle_guard(100000, 99999, 0)
    t.idle_guard(100000, 0, 99999)
    with pytest.raises(ValueError, match='no study/window progress AND no GPU activity'):
        t.idle_guard(100000, 0, 0)


def test_live_guard_trips_before_one_gib_ceiling():
    _, total = t.desktop_values(desktop())
    t.live_memory_guard((total-2*t.GIB-1)/2**20, desktop())
    with pytest.raises(ValueError, match='live VRAM guard'):
        t.live_memory_guard((total-2*t.GIB)/2**20, desktop())
    assert total-2*t.GIB < total-t.GIB


@pytest.mark.parametrize('second', [15600, 11000, 'failure'])
def test_sampler_guard_precedes_full_logging_queue(monkeypatch, second):
    samples = iter([(10000, 30, 100), (second, 30, 100)])
    def sample(env):
        value = next(samples)
        if value[0] == 'failure':
            raise OSError('sample unavailable')
        if value[0] == 11000:
            telemetry.stop.set()
        return value
    monkeypatch.setattr(t.monitor, '_gpu_sample', sample)
    telemetry = t._GuardedTelemetry({}, .05, desktop())
    telemetry.start()
    telemetry.thread.join(timeout=2)
    assert not telemetry.thread.is_alive()
    # First value remains queued; the second was evaluated without dequeuing it.
    assert telemetry.results.full()
    assert telemetry.take() == ((10000, 30, 100), None)
    assert telemetry.take() is None
    if second == 11000:
        assert telemetry.guard_snapshot()[0] == 11000
    else:
        with pytest.raises(ValueError, match='live VRAM guard' if second == 15600 else 'sample unavailable'):
            telemetry.guard_snapshot()
        with pytest.raises(ValueError):  # failure remains latched after queue drain
            telemetry.guard_snapshot()
    assert telemetry.close()


@pytest.mark.parametrize('case', ['queue_full', 'stall', 'failure', 'normal'])
def test_guarded_sampler_real_dicom_cpu_child_cleanup(monkeypatch, tmp_path, case):
    import subprocess
    import sys
    import threading
    import zipfile
    import psutil
    with zipfile.ZipFile(t.ZIP) as source:
        entry = next(x for x in source.infolist() if x.filename.endswith('.dcm'))
        dicom = tmp_path/'sample.dcm'
        dicom.write_bytes(source.read(entry))
    monkeypatch.setattr(t, 'WORK', tmp_path)
    monkeypatch.setattr(t, 'desktop_snapshot', desktop)
    release = threading.Event()
    calls = []
    def sample(env):
        calls.append(1)
        if case == 'stall' and len(calls) > 1:
            release.wait(5)
        if case == 'failure' and len(calls) > 1:
            raise OSError('sample failed')
        return (15600 if case == 'queue_full' and len(calls) > 1 else 10000, 30, 100)
    monkeypatch.setattr(t.monitor, '_gpu_sample', sample)
    monkeypatch.setattr(t, 'SAMPLER_STALL_SECONDS', .4 if case == 'stall' else 30)
    if case == 'queue_full':
        # Hold the supervisor until the producer has seen both samples with no
        # queue consumer. This reproduces the independent review's lost spike.
        original_start = t._GuardedTelemetry.start
        def start(self):
            original_start(self)
            self.thread.join(timeout=2)
            assert not self.thread.is_alive() and self.results.full()
        monkeypatch.setattr(t._GuardedTelemetry, 'start', start)
    popen = subprocess.Popen
    children = []
    def child(command, **kwargs):
        source = ('import pydicom,sys,time; '
                  'assert pydicom.dcmread(sys.argv[1]).pixel_array.size > 0; '
                  'print("CPU_DICOM_DECODE_OK", flush=True); time.sleep(%s)' %
                  ('.2' if case == 'normal' else '30'))
        p = popen([sys.executable, '-B', '-c', source, str(dicom)], **kwargs)
        children.append(p)
        return p
    monkeypatch.setattr(t.subprocess, 'Popen', child)
    try:
        if case == 'normal':
            assert t.supervise('plain', None) > 0
        else:
            with pytest.raises(ValueError, match={'queue_full':'live VRAM guard',
                                                 'stall':'sampler stalled',
                                                 'failure':'sample failed'}[case]):
                t.supervise('plain', None)
    finally:
        release.set()
    receipt = t.b.read(tmp_path/'plain-timing.json')
    assert receipt['cleanup'] == []
    assert bool(receipt['error']) == (case != 'normal')
    assert all(p.poll() is not None and not psutil.pid_exists(p.pid) for p in children)


def test_cli_run_failure_nonzero_and_dry_run_no_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(t.b,'validate',lambda:None)
    monkeypatch.setattr(t,'ADMISSION',tmp_path/'missing.json')
    monkeypatch.setattr(t,'lane_admission',lambda *a:pytest.fail('CPU admission touched GPU'))
    assert t.main(['admit','--dry-run']) == 2
    monkeypatch.setattr(t,'admission',lambda x:x)
    monkeypatch.setattr(t.b,'read',lambda x:{})
    monkeypatch.setattr(t,'run',lambda x:(_ for _ in ()).throw(ValueError('DO_NOT_SUBMIT')))
    with pytest.raises(ValueError,match='DO_NOT_SUBMIT'):
        t.main(['run'])


def test_real_inputs_cpu_run_smoke_gpu_is_mocked(monkeypatch,tmp_path):
    value,calls=simulated_run(monkeypatch,tmp_path,real_inputs=True)
    result=t.run(value)
    assert result['baseline']['submission_ref'] == '57133175'
    assert result['projected_hidden_seconds'] > result['baseline']['hidden_seconds']
    assert result['status'] == 'PASS_LOCAL_GATES_ONLY'
    assert [a for a,d in calls] == ['prepare',*t.ORDER]
    assert {r['study'] for r in result['per_mode']['plain-1']['rows']} == set(t.cohort()[0])
    assert result['qualified_for_submission'] is False


def test_real_dicom_cpu_staging_resume_and_measurement_smoke(monkeypatch,tmp_path):
    """Real selected DICOM bytes through staging/decoding; CUDA is mocked."""
    import io
    import zipfile
    import pydicom
    from types import SimpleNamespace
    ids, rows = t.cohort()
    samples = {}
    archive_path = tmp_path/'real-samples.zip'
    with zipfile.ZipFile(t.ZIP) as source, zipfile.ZipFile(archive_path,'w') as target:
        for entry in source.infolist():
            parts = entry.filename.split('/')
            if (len(parts) == 4 and parts[0] in ('train_series','train_images')
                    and parts[1] in ids and parts[1] not in samples and parts[3].endswith('.dcm')):
                raw = source.read(entry)
                assert pydicom.dcmread(io.BytesIO(raw)).pixel_array.size > 0
                target.writestr(entry.filename, raw)
                samples[parts[1]] = parts[2]
    selected = [r for r in rows if samples.get(r['StudyInstanceUID']) == r['SeriesInstanceUID']]
    assert len(samples) == len(selected) == 32
    monkeypatch.setattr(t,'ZIP',archive_path)
    monkeypatch.setattr(t,'WORK',tmp_path/'work')
    monkeypatch.setattr(t,'cohort',lambda:(ids,selected))
    assert t.prepare_data() == ids
    assert t.prepare_data() == ids
    assert len(t.b.read(t.WORK/'cohort.json')['files']) == 32
    cuda = SimpleNamespace(synchronize=lambda:None, max_memory_reserved=lambda:5*t.GIB,
                           max_memory_allocated=lambda:4*t.GIB, get_device_name=lambda:'4070 Ti SUPER')
    measured=[]
    for uid in ids:
        m.commit(SimpleNamespace(cuda=cuda),t.WORK/'plain-1-measure-status.json',measured,
                 uid,1,192,t.time.perf_counter(),0.,0.,0.)
    assert t.measurements('plain-1',ids)['done'] == 32
