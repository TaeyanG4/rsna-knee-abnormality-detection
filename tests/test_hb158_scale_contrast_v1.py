"""H-B-158 contract tests; no CUDA, no label rows printed."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from tools import hb158_scale_contrast_v1 as h


def pair():
    return dict(primary_mean_delta=.001, references={name:dict(ci95=[.0001,.01],
        per_target_delta={t:.001 for t in h.TARGETS}) for name in h.RULERS})


def test_registered_gate_boundaries():
    p = pair()
    assert h.scientific_gate(p)['status'] == 'GO_LOCAL_ONLY'
    p['primary_mean_delta'] = .000999
    assert h.scientific_gate(p)['status'] == 'HOLD'
    p = pair()
    p['references']['dread']['per_target_delta']['MCL'] = -.003001
    assert h.scientific_gate(p)['status'] == 'HOLD'
    p['references']['dread']['per_target_delta']['MCL'] = -.003
    assert h.scientific_gate(p)['status'] == 'GO_LOCAL_ONLY'
    for name in ('jev','aquila','consensus'):
        p['references'][name]['ci95'][0] = 0
    assert h.scientific_gate(p)['status'] == 'HOLD'


@pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
def test_missing_target_never_passes(value):
    p = pair()
    p['references']['aquila']['per_target_delta']['ACL'] = value
    assert h.scientific_gate(p)['status'] == 'HOLD'


def test_no_probability_fallback():
    for values in (np.zeros((2,11)), np.full((2,12), np.nan), np.full((2,12), 1.1)):
        with pytest.raises(ValueError):
            h.valid_probs(values, 2)


def test_review_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'ROOT', tmp_path)
    evidence = tmp_path/'independent.md'
    evidence.write_text('Independent scoped review evidence', encoding='utf-8')
    r = dict(verdict='PASS', reviewer='reviewer-other', independent=True, scope='HB158_SCALE',
             registration_sha256='reg', cpu_smoke_sha256='smoke',
             evidence_path=str(evidence), evidence_sha256=h.sha(evidence))
    h.check_review(r, 'reg', 'smoke')
    for field, value in [('verdict','FAIL'), ('reviewer',h.BUILDER), ('independent',False),
                         ('registration_sha256','old'), ('cpu_smoke_sha256','old'),
                         ('evidence_sha256','wrong')]:
        bad = dict(r, **{field:value})
        with pytest.raises(ValueError):
            h.check_review(bad, 'reg', 'smoke')
    with pytest.raises(ValueError):
        h.check_review({}, 'reg', 'smoke')


def test_admit_missing_review_only_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'WORK', tmp_path)
    monkeypatch.setattr(h, 'registered', lambda: {})
    h.io.atomic_json(tmp_path/'registration.json', {})
    h.io.atomic_json(tmp_path/'cpu-smoke.json', dict(status='CPU_SMOKE_PASS', studies=2,
        registration_sha256=h.sha(tmp_path/'registration.json')))
    assert h.admit(True)['status'] == 'HOLD_REVIEW_REQUIRED'
    with pytest.raises(FileNotFoundError):
        h.admit()
    h.io.atomic_json(tmp_path/'review.json', dict(verdict='FAIL'))
    with pytest.raises(ValueError):
        h.admit(True)


def test_stale_registration_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'WORK', tmp_path)
    h.io.atomic_json(tmp_path/'registration.json', {'identity':'partial/old'})
    monkeypatch.setattr(h, 'registration', lambda: {'identity':'complete/current'})
    with pytest.raises(ValueError, match='stale'):
        h.registered()


def test_frozen_serving_definition_diff_and_crop():
    import torch
    a, proof = h.serving(320)
    b, proof384 = h.serving(384)
    assert proof == proof384
    assert {n for n,v in proof['definitions'].items() if v['change'] != 'byte-identical'} == {'infer_pair_probs'}
    volume = np.random.default_rng(3).integers(0,256,(96,384,384),dtype=np.uint8)
    mask = np.ones(96,np.uint8)
    for mirrored in (False, True):
        x = a['gpu_windows'](volume,mask,4,320,'cpu',mirror=mirrored)
        y = b['gpu_windows'](volume,mask,4,384,'cpu',mirror=mirrored)
        assert torch.equal(x,y[:,:,32:352,32:352])
        assert torch.equal(x,a['eval_windows'](volume,mask,4,320,mirror=mirrored))
        assert torch.equal(y,b['eval_windows'](volume,mask,4,384,mirror=mirrored))
    with pytest.raises(ValueError):
        h.serving(352)


def test_mean_before_ranks_and_identical_masks(monkeypatch):
    from tools import exp170_convnext_t_l2 as e
    calls = []
    refs, draws = object(), object()
    def fake(candidate, baseline, references, bootstrap, evaluator):
        assert references is refs and bootstrap is draws
        calls.append((np.array(candidate),np.array(baseline)))
        return pair()
    monkeypatch.setattr(e,'paired_jev_consensus',fake)
    a = np.tile(np.array([.1,.4,.9])[:,None],(1,12))
    b = np.tile(np.array([.9,.3,.2])[:,None],(1,12))
    h.comparisons({'320':a,'384':b},refs,draws)
    assert np.array_equal(calls[1][0],.5*(a+b))
    assert np.array_equal(calls[0][1],calls[1][1])


def test_memory_gate_and_desktop_apps():
    from tools.c85r7_flipall_timing import memory_admission, live_memory_guard
    desktop = dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_mib=16376,used_mib=1200)
    memory_admission(desktop,[12*h.GIB])
    with pytest.raises(ValueError):
        memory_admission(desktop,[12*h.GIB+1])
    with pytest.raises(ValueError):
        memory_admission(dict(desktop,used_mib=3000),[12*h.GIB])
    with pytest.raises(ValueError):
        live_memory_guard(16376-2048,desktop)


def test_watchdogs_and_no_wall_cap():
    from tools.hb149_monitor_v125 import RateGuard
    assert h.guard_reason(900,0,[1000,100,200],16376)
    assert h.guard_reason(600,0,[15000,99,70],16376)
    assert h.guard_reason(100000,99999,[1000,100,200],16376) is None
    r = RateGuard(10,0)
    assert r.update(1800,1)


def test_authenticate_real_control_and_fixed_rulers():
    ids, refs, draws, meta = h.authenticate()
    assert len(set(ids)) == 888 and len(draws) == 300
    assert meta['train_n'] == 3455 and meta['train_val_overlap'] == 0
    assert meta['checkpoint_sha256'] == h.MODEL_SHA
    assert set(refs) == set(h.RULERS)
    assert all(y.shape == m.shape == (888,12) for y,m,_ in refs.values())


def test_actual_desktop_admission_query():
    # Read-only live Windows query; NO CUDA context, GPU model or inference.
    import psutil
    allowed = {p.pid for p in psutil.process_iter(['name'])
               if 'python' in (p.info['name'] or '').lower()}
    h.no_other_ml(allowed)


def test_desktop_apps_allowed_but_competing_ml_rejected(monkeypatch):
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(h.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout='101, explorer.exe\n102, python.exe\n'))
    monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(
        name=lambda: 'explorer.exe' if pid == 101 else 'python.exe'))
    h.no_other_ml({102})
    with pytest.raises(ValueError, match='other GPU ML'):
        h.no_other_ml(set())


def test_internal_worker_requires_live_controller(monkeypatch):
    import psutil
    from types import SimpleNamespace
    argv = [h.sys.executable, str(Path(h.__file__).resolve()), 'run']
    parent = SimpleNamespace(pid=123, create_time=lambda: 10., cmdline=lambda: argv)
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(parent=lambda: parent))
    monkeypatch.setattr(h, 'run_binding', lambda: {'source':'current'})
    monkeypatch.delenv('HB158_CONTROLLER', raising=False)
    with pytest.raises(ValueError, match='controller'):
        h.require_controller('calibration')
    ticket = dict(pid=123, created=10., phase='calibration', binding={'source':'current'})
    monkeypatch.setenv('HB158_CONTROLLER', json.dumps(ticket))
    h.require_controller('calibration')
    for field, value in [('pid',999), ('created',11.), ('phase','full'), ('binding',{})]:
        monkeypatch.setenv('HB158_CONTROLLER', json.dumps(dict(ticket, **{field:value})))
        with pytest.raises(ValueError, match='controller'):
            h.require_controller('calibration')
    monkeypatch.setenv('HB158_CONTROLLER', json.dumps(ticket))
    argv[-1] = '_worker'
    with pytest.raises(ValueError, match='controller'):
        h.require_controller('calibration')


def test_gpu_wrapper_identity_and_device_fail_closed(tmp_path, monkeypatch):
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(h, 'ROOT', tmp_path)
    wrapper = SimpleNamespace(pid=321, create_time=lambda: 100.)
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(parents=lambda: [wrapper]))
    monkeypatch.setattr(h, 'no_other_ml', lambda allowed: None)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    path = tmp_path/'artifacts/logs/HB158_SCALE/job.json'
    valid = dict(tag='HB158_SCALE', pid=321, create_time=100., phase='running')
    h.io.atomic_json(path, valid)
    h.require_gpu_job()
    for field, value in [('tag','OTHER'), ('pid',999), ('create_time',99.), ('phase','done')]:
        h.io.atomic_json(path, dict(valid, **{field:value}))
        with pytest.raises(ValueError, match='gpu_job'):
            h.require_gpu_job()
    h.io.atomic_json(path, valid)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(ValueError, match='physical GPU zero'):
        h.require_gpu_job()


def test_completed_output_and_memory_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'WORK', tmp_path)
    monkeypatch.setattr(h, 'run_binding', lambda: {'source':'current'})
    path = tmp_path/'gpu/calibration/journal/320-0000.json'
    h.io.atomic_json(path, {'peak_reserved_bytes':h.GIB})
    controller = path.parent.parent/'controller.json'
    proof = dict(exit_code=0, cleanup_errors=[], phase='calibration', binding={'source':'current'},
        journal_sha256={path.name:h.sha(path)}, peak_device_mib=2200,
        desktop=dict(device_name='RTX 4070 Ti SUPER', total_mib=16376, used_mib=1200))
    with pytest.raises(FileNotFoundError):
        h.completed_phase('calibration', 1)
    h.io.atomic_json(controller, proof)
    assert h.completed_phase('calibration', 1)['exit_code'] == 0
    for field, value in [('exit_code',1), ('cleanup_errors',['alive']), ('binding',{}),
                         ('journal_sha256',{}), ('peak_device_mib',16000)]:
        h.io.atomic_json(controller, dict(proof, **{field:value}))
        with pytest.raises(ValueError):
            h.completed_phase('calibration', 1)
    h.io.atomic_json(controller, proof)
    admission = dict(status='MEASURED_PASS', binding={'source':'current'},
                     calibration_controller_sha256=h.sha(controller))
    h.io.atomic_json(tmp_path/'gpu/memory-admission.json', admission)
    h.check_memory_receipt()
    h.io.atomic_json(controller, dict(proof, edited=True))
    with pytest.raises(ValueError, match='stale'):
        h.check_memory_receipt()
    h.io.atomic_json(path, {'peak_reserved_bytes':2*h.GIB})
    with pytest.raises(ValueError, match='journal changed'):
        h.completed_phase('calibration', 1)


@pytest.mark.parametrize('fails', [False, True, 'guard'])
def test_supervisor_cpu_real_child_cleanup(tmp_path, monkeypatch, fails):
    from tools import c85r7_flipall_timing as timing
    import threading
    # Real Windows process/job/cleanup; only GPU sampling is replaced.
    class Telemetry:
        def __init__(self, *args):
            self.thread = threading.Thread(target=lambda: None)
        def start(self): self.thread.start()
        def close(self): self.thread.join()
        def take(self): return ([2200,50,150], None)
        def guard_snapshot(self):
            if fails == 'guard': raise ValueError('synthetic live guard')
            return (2200, 0)
    monkeypatch.setattr(timing, '_GuardedTelemetry', Telemetry)
    monkeypatch.setattr(h, 'run_binding', lambda: {'source':'current'})
    folder = tmp_path/'calibration'
    script = tmp_path/'child.py'
    script.write_text('import json, pathlib, sys, time\n'
        'p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True)\n'
        'p.joinpath("320-0000.json").write_text(json.dumps({"peak_reserved_bytes":1073741824}))\n'
        f'time.sleep({30 if fails == "guard" else 0})\n'
        f'sys.exit({1 if fails else 0})\n', encoding='utf-8')
    command = [h.sys.executable, str(script), str(folder/'journal')]
    desktop = dict(device_name='RTX 4070 Ti SUPER', total_mib=16376, used_mib=1200)
    if fails:
        with pytest.raises(ValueError, match='synthetic live guard' if fails == 'guard' else 'worker failed'):
            h.supervise(command, folder, desktop)
    else:
        h.supervise(command, folder, desktop)
    receipt = h.io.read_json(folder/'controller.json')
    assert receipt['exit_code'] == int(bool(fails))
    assert receipt['cleanup_errors'] == [] and receipt['no_wall_cap']


def test_evaluation_refuses_incomplete_controller(monkeypatch):
    monkeypatch.setattr(h, 'admit', lambda: {})
    def incomplete(*args): raise ValueError('incomplete controller')
    monkeypatch.setattr(h, 'completed_phase', incomplete)
    monkeypatch.setattr(h, 'authenticate', lambda: pytest.fail('read inputs before completion'))
    with pytest.raises(ValueError, match='incomplete controller'):
        h.evaluate_outputs({'ids':[]})
