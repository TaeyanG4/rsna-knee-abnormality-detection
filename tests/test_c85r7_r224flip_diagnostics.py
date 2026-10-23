"""CPU contracts for HB167; no GPU/model/network execution."""
# v1.0.6: delayed owned-process cleanup and authenticated preserved-pair recovery.
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import c85r7_r224flip_diagnostics as d


def test_cleanup_real_cpu_child_and_recorded_delayed_exit(monkeypatch, tmp_path):
    import subprocess
    import sys
    import psutil
    monkeypatch.setattr(d.r224, 'WORK', tmp_path)
    child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(.3)'],
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    process = psutil.Process(child.pid)
    try:
        # Reproduce the old cleanup's stale alive report with a real exiting CPU child.
        monkeypatch.setattr(d, '_CLEANUP', lambda *a: [f'still alive {child.pid}'])
        assert d.cleanup_owned(child, {child.pid:process}, None, psutil) == []
        assert child.wait(timeout=2) == 0
        receipt = json.loads((tmp_path/'cleanup-recovery.json').read_text())
        assert receipt['recovered'] == [f'still alive {child.pid}']
        assert receipt['remaining'] == []
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


@pytest.mark.parametrize('kind', ['live', 'denied', 'untracked', 'other'])
def test_cleanup_never_waives_unproven_or_other_errors(monkeypatch, tmp_path, kind):
    import psutil
    monkeypatch.setattr(d.r224, 'WORK', tmp_path)
    error = 'terminate 12: AccessDenied' if kind == 'other' else 'still alive 12'
    monkeypatch.setattr(d, '_CLEANUP', lambda *a: [error])
    clock = iter([0., 11., 12.])
    monkeypatch.setattr(d.time, 'monotonic', lambda: next(clock))
    def live():
        if kind == 'denied': raise psutil.AccessDenied(12)
        return True
    tracked = {} if kind == 'untracked' else {12:SimpleNamespace(is_running=live)}
    assert d.cleanup_owned(None, tracked, None, psutil) == [error]
    assert json.loads((tmp_path/'cleanup-recovery.json').read_text())['remaining'] == [error]


def test_historical_cleanup_requires_absence_not_pid_reuse(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: True)
    with pytest.raises(ValueError, match='PID still exists'):
        d.resolved_cleanup(['still alive 25780'])
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: False)
    assert d.resolved_cleanup(['still alive 25780'])[0]['original'] == 'still alive 25780'
    with pytest.raises(ValueError, match='unrecoverable'):
        d.resolved_cleanup(['terminate 25780: AccessDenied'])


@pytest.fixture
def preserved():
    source = d.WORK/'gold/v1.0.5'
    pins = d.WORK/'gold/v1.0.6/preserved-pins.json'
    pin = 'a3debbd5a122a0763756fcbacf0c6e85e0f60d11ffefab776d10805c4309dcd6'
    if not pins.exists(): pytest.skip('local preserved Gold58 evidence unavailable')
    return source, pins, pin


@pytest.mark.parametrize('damage', ['seal','binding','admission','review','threshold','common','attempt','job'])
def test_preserved_context_rejects_tampering(preserved, monkeypatch, damage):
    source, pins, pin = preserved
    original = d.io.read_json
    def read(path):
        value = copy.deepcopy(original(path))
        name = Path(path).name
        if damage == 'binding' and name == 'registration.json': value['binding_sha256'] = 'bad'
        if damage == 'admission' and name == 'admission.json': value['state'] = 'DRY_RUN_PASS'
        if damage == 'review' and name == 'independent-review.json': value['verdict'] = 'FAIL'
        if damage == 'threshold' and name == 'registration.json': value['binding']['rules']['gold_macro_delta'] = -.1
        if damage == 'common' and name == 'registration.json': value['binding']['common']['sha256'] = 'bad'
        if damage == 'attempt' and name == 'attempt.json': value['reason'] = 'inference failed'
        if damage == 'job' and name == 'receipt.json' and 'HB167_GOLD58' in str(path): value['state'] = 'RUNNING'
        return value
    monkeypatch.setattr(d.io, 'read_json', read)
    with pytest.raises(ValueError):
        d.preserved_context(source, pins, 'bad' if damage == 'seal' else pin)


@pytest.mark.parametrize('damage', ['partial','binding','flip','predictions','duplicate','telemetry','fallback','exit'])
def test_preserved_arm_rejects_invalid_outputs(preserved, damage):
    source, pins, pin = preserved
    reg, _, _, read = d.preserved_context(source, pins, pin)
    def damaged(path):
        value = copy.deepcopy(read(path))
        name = Path(path).name
        if name == 'worker-status.json':
            if damage == 'partial': value['done'] = 57
            if damage == 'binding': value['binding_sha256'] = 'bad'
            if damage == 'flip': value['runtime']['flip_tta'] = False
            if damage == 'predictions': value['predictions'][0][0] += .0001
            if damage == 'duplicate': value['rows'][0] = value['rows'][1]
        if name == 'supervision.json':
            if damage == 'telemetry': value['telemetry']['complete'] = False
            if damage == 'exit': value['exit_code'] = 1
        if name == 'submission_receipt.json' and damage == 'fallback': value['fallback_studies'] = 1
        return value
    with pytest.raises(ValueError): d.preserved_arm(source, 'flip-1', reg, damaged)


def test_evaluate_real_preserved_pair_cpu_and_no_mutation(preserved, monkeypatch, tmp_path):
    import torch
    source, pins, pin = preserved
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(d, 'save', lambda mode, name, value: value)
    monkeypatch.setattr(d, 'lane', lambda *a: pytest.fail('GPU lane reached'))
    value = d.evaluate(source, pins, pin)
    assert value['result']['macro_delta'] == pytest.approx(-.0013373415279985164)
    assert value['result']['weak_four_delta'] == pytest.approx(-.004820518271325486)
    assert value['result']['no_harm_pass'] is False
    assert value['measured_pair'] is True and not torch.cuda.is_initialized()
    assert value['cleanup_warnings']['flip-1'][0]['original'] == 'still alive 25780'


def test_evaluate_requires_explicit_cpu_environment(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with pytest.raises(ValueError, match='present-empty'):
        d.evaluate(None, None, None)


@pytest.fixture
def measured(monkeypatch):
    ids = [str(i) for i in range(32)]
    monkeypatch.setattr(d.r7, 'cohort', lambda: (ids, []))
    arms = {name: dict(seconds=100., rows=[dict(study=u, gpu_seconds=1.,
             windows=94, batch_windows=192) for u in ids]) for name in d.ORDER}
    for name in ('flip-1', 'flip-2'):
        arms[name]['seconds'] = 200.
        for row in arms[name]['rows']:
            row['gpu_seconds'] = 2.
    return arms, dict(r7_hidden_seconds=3600.)


def test_measured_projection_and_empirical_separation(measured):
    value = d.projection(*measured)
    assert value['projected_hours'] == pytest.approx(7.38)
    assert value['runtime_pass'] is True
    assert value['qualified_for_submission'] is False
    assert value['empirical']['conditional_additive_estimate_hours'] == pytest.approx(6.22)
    assert value['empirical']['rigorous_upper_bound'] is False
    assert value['empirical']['runtime_gate_uses_this'] is False


def test_no_speedup_credit_and_gate_boundary(measured):
    arms, cal = measured
    for name in ('flip-1', 'flip-2'):
        arms[name]['seconds'] = 50.
        for row in arms[name]['rows']:
            row['gpu_seconds'] = .5
    arms['r224']['seconds'] = 312.
    value = d.projection(arms, cal)
    assert value['r7_flip_increment_hours'] == 0
    assert value['projected_hours'] == pytest.approx(8.5)
    assert value['runtime_pass'] is True
    arms['r224']['seconds'] = 312.01
    assert d.projection(arms, cal)['runtime_pass'] is False


@pytest.mark.parametrize('damage', ['missing','order','windows','zero','nan','calibration','r224rows'])
def test_invalid_measurement_fails(measured, damage):
    arms, cal = measured
    if damage == 'missing': del arms['flip-2']
    elif damage == 'order': arms['flip-1']['rows'].reverse()
    elif damage == 'windows': arms['flip-1']['rows'][0]['windows'] += 1
    elif damage == 'zero': arms['plain-1']['seconds'] = 0
    elif damage == 'nan': arms['flip-1']['rows'][0]['gpu_seconds'] = math.nan
    elif damage == 'calibration': cal['r7_hidden_seconds'] = -1
    elif damage == 'r224rows': arms['r224']['rows'].pop()
    with pytest.raises(ValueError): d.projection(arms, cal)


@pytest.mark.parametrize('size', [32,58])
def test_canonicalize_shuffled_rows_preserves_predictions_and_completion_audit(size):
    ids = [str(i) for i in range(size)]
    rows = [dict(study=u, measured=i) for i,u in enumerate(reversed(ids))]
    value = dict(rows=rows, predictions=[[i]*12 for i in range(size)], seconds=230.2)
    before = copy.deepcopy(value)
    ordered = d.canonicalize_rows(value, ids)
    assert [r['study'] for r in ordered['rows']] == ids
    assert ordered['completion_order'] == list(reversed(ids))
    assert ordered['predictions'] is value['predictions']
    assert ordered['seconds'] == value['seconds']
    assert value == before
    assert {r['study']:r for r in ordered['rows']} == {r['study']:r for r in rows}


@pytest.mark.parametrize('damage', ['duplicate','missing','extra','replacement','cohort_duplicate'])
def test_canonicalize_rejects_nonexact_or_duplicate_cohort(damage):
    ids = ['a','b','c']
    value = dict(rows=[dict(study=u) for u in reversed(ids)])
    if damage == 'duplicate': value['rows'][0] = value['rows'][1]
    elif damage == 'missing': value['rows'].pop()
    elif damage == 'extra': value['rows'].append(dict(study='extra'))
    elif damage == 'replacement': value['rows'][0] = dict(study='extra')
    else: ids.append('a')
    with pytest.raises(ValueError, match='unique exact cohort'):
        d.canonicalize_rows(value, ids)


def test_shuffled_all_arms_canonicalize_before_paired_projection(measured):
    arms,cal = measured
    expected = d.projection(arms,cal)
    ids = d.r7.cohort()[0]
    for i,arm in enumerate(arms.values(),1):
        arm['rows'] = arm['rows'][i:]+arm['rows'][:i]
    normalized = {name:d.canonicalize_rows(value,ids) for name,value in arms.items()}
    assert d.projection(normalized,cal) == expected
    normalized['flip-1']['rows'][0]['windows'] += 1
    with pytest.raises(ValueError, match='paired R7 windows drift'):
        d.projection(normalized,cal)


@pytest.mark.parametrize('mode,arm', [('timing',a) for a in d.ORDER] +
                         [('gold','plain-1'),('gold','flip-1')])
def test_every_arm_checks_canonicalized_rows_without_relaxing_validators(monkeypatch,mode,arm):
    ids = [str(i) for i in range(32 if mode == 'timing' else 58)]
    monkeypatch.setattr(d,'roster',lambda mode:ids)
    manifest = d.io.read_json(d.r7.MANIFEST)
    runtime = dict(models=manifest['models'], models2=[], flip_tta=arm.startswith('flip'), workers=2,
                   **{k:manifest[k] for k in ('size','crop','views','slices')})
    value = dict(identity=d.r224.TAG, state='COMPLETE', done=len(ids), clean_oof=False,
                 runtime=runtime, predictions=[[.5]*12 for _ in ids], rows=[dict(
                     study=u, windows=94, batch_windows=192, resolution=224, seconds=1., gpu_seconds=1.,
                     probabilities=[.5]*12, failed_study_fallback=False, forward_precision=['float16'],
                     fp32_retry=False, peak_allocated_bytes=2**29, peak_reserved_bytes=2**30) for u in reversed(ids)])
    desktop = dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_mib=16376.,used_mib=1445.)
    telemetry = dict(complete=True,peak_gpu_mib=3000.)
    with pytest.raises(ValueError,match='complete ordered cohort'):
        d.check_arm(mode,arm,value,desktop,telemetry)
    ordered = d.canonicalize_rows(value,ids)
    d.check_arm(mode,arm,ordered,desktop,telemetry)
    # Ordering repair must not mask malformed completion, telemetry or row data.
    with pytest.raises(ValueError,match='complete ordered cohort'):
        d.check_arm(mode,arm,dict(ordered,done=len(ids)-1),desktop,telemetry)
    with pytest.raises(ValueError,match='incomplete telemetry'):
        d.check_arm(mode,arm,ordered,desktop,dict(telemetry,complete=False))
    if arm == 'r224': ordered['rows'][0]['resolution'] = 223
    else: ordered['rows'][0]['gpu_seconds'] = math.nan
    with pytest.raises(ValueError):
        d.check_arm(mode,arm,ordered,desktop,telemetry)


def test_commit_status_is_atomic_and_supervisor_compatible(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(d.io, 'journal', lambda p,v,w: captured.append(copy.deepcopy(v)))
    gpu = SimpleNamespace(synchronize=lambda:None, max_memory_reserved=lambda:1024,
                         max_memory_allocated=lambda:512, get_device_name=lambda:'CPU synthetic')
    ticks = iter([12.,13.,22.,23.])
    monkeypatch.setattr(d.measure.time, 'perf_counter', lambda: next(ticks))
    rows = []
    for i in range(2):
        d.commit(SimpleNamespace(cuda=gpu), tmp_path/'status.json', rows, str(i),94,192,10.+10*i,1.,1.,0.)
    assert len(captured) == 2
    assert all(r['seconds'] == r['study_seconds'] for v in captured for r in v['rows'])
    healthy = 2*60/sum(r['seconds'] for r in captured[-1]['rows'])
    assert healthy == 20.


def synthetic_gold():
    ids = [str(i) for i in range(58)]
    labels = np.tile((np.arange(58)%2)[:,None], (1,12))
    common = {name: np.full((58,12), .4) for name in d.COMMON}
    failed = {name: np.zeros(58, dtype=bool) for name in d.COMMON}
    plain = labels*.8+.1
    return ids,labels,common,failed,plain


def test_gold_full_blend_identical_pass_and_harm_fails():
    ids,labels,common,failed,plain = synthetic_gold()
    same = d.gold_no_harm(ids, labels, common, failed, plain, plain)
    assert same['no_harm_pass'] and same['macro_delta'] == same['weak_four_delta'] == 0
    damaged = d.gold_no_harm(ids, labels, common, failed, plain, 1-plain)
    assert not damaged['no_harm_pass'] and damaged['macro_delta'] == -1
    assert damaged['clean_oof'] is False and damaged['qualified_for_submission'] is False


def test_weak_four_harm_and_full_formula_reference():
    ids,labels,common,failed,plain = synthetic_gold()
    flip = plain.copy()
    flip[:,list(d.gold.WEAK)] = 1-flip[:,list(d.gold.WEAK)]
    result = d.gold_no_harm(ids,labels,common,failed,plain,flip)
    assert result['weak_four_delta'] == -1 and not result['no_harm_pass']
    assert result['formula'] == d.formula.FORMULA
    expected = d.formula.reference(common['R'],common['R224'],plain,common['EXP143v2'],common['EXP155'],common['Q_A5q'])
    observed = d.formula.blend(ids,common['R'],common['R224'],plain,common['EXP143v2'],common['EXP155'],common['Q_A5q'])
    np.testing.assert_array_equal(expected, observed)


@pytest.mark.parametrize('damage', ['dedup','nan','mask','missing','label'])
def test_gold_invalid_data_fails(damage):
    ids,labels,common,failed,plain = synthetic_gold()
    if damage == 'dedup': ids[-1] = ids[0]
    elif damage == 'nan': plain[0,0] = np.nan
    elif damage == 'mask': failed['R'] = np.zeros(58)
    elif damage == 'missing': del common['R224']
    elif damage == 'label': labels[:,0] = 0
    with pytest.raises(ValueError): d.gold_no_harm(ids,labels,common,failed,plain,plain)


def test_missing_common_is_binding_hold():
    with pytest.raises(ValueError, match='authenticated parent'):
        d.read_common(None, [])


@pytest.mark.parametrize('damage', [None, 'cohort', 'fallback', 'validation', 'cuda_env'])
def test_gold_preserved_replay_is_identity_control_only(monkeypatch, tmp_path, damage):
    ids, labels, common, failed, plain = synthetic_gold()
    monkeypatch.setattr(d, 'WORK', tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '' if damage != 'cuda_env' else '-1')
    monkeypatch.setattr(d, 'current', lambda mode: dict(binding_sha256='sealed', binding=dict(common=dict(path='cache'))))
    d.save('gold', 'validate', dict(binding_sha256='wrong' if damage == 'validation' else 'sealed'))
    monkeypatch.setattr(d.gold, 'gold', lambda: (ids, labels))
    monkeypatch.setattr(d, 'read_common', lambda path, cohort: (dict(sha256='cache-sha'), common, failed))
    old_failed = dict(R7=np.zeros(58, dtype=bool))
    if damage == 'fallback': old_failed['R7'][0] = True
    receipt = dict(studies=58, limit=None, fallback_counts=dict(R7=0), prediction_sha256='pred-sha', binding_sha256='reg-sha')
    monkeypatch.setattr(d.gold, 'load_run', lambda run, mode: (ids[::-1] if damage == 'cohort' else ids,
                        dict(R7=plain), old_failed, receipt))
    (tmp_path/'receipt.json').write_text('{}')
    if damage:
        with pytest.raises(ValueError): d.replay_gold(tmp_path)
        assert not d.state_path('gold', 'cpu-replay').exists()
    else:
        result = d.replay_gold(tmp_path)
        assert result['state'] == 'CPU_IDENTITY_CONTROL_PASS'
        assert result['macro_delta'] == result['weak_four_delta'] == 0
        assert result['measured_pair'] is result['qualified_for_submission'] is False
        assert 'no_harm_pass' not in result and 'labels' not in result


def test_admission_dry_run_cannot_bypass_review(monkeypatch, tmp_path):
    monkeypatch.setattr(d, 'WORK', tmp_path)
    reg = dict(binding_sha256='sealed', binding=dict(common=None))
    monkeypatch.setattr(d, 'current', lambda mode:reg)
    d.save('timing','validate',dict(binding_sha256='sealed'))
    with pytest.raises(ValueError, match='independent review'):
        d.admit('timing', None, dry_run=True)
    assert not d.state_path('timing','admission').exists()


@pytest.mark.parametrize('damage', ['same_builder','binding','differences','gold_common'])
def test_review_fail_closed(monkeypatch, tmp_path, damage):
    monkeypatch.setattr(d, 'WORK', tmp_path)
    mode = 'gold' if damage == 'gold_common' else 'timing'
    d.save(mode,'validate',dict(state='CPU_VALIDATED'))
    reg = dict(binding_sha256='sealed', binding=dict(common=None))
    review = dict(verdict='PASS',builder='builder',reviewer='reviewer',execution_authority='B',
                  binding_sha256='sealed',validation_sha256=d.sha(d.state_path(mode,'validate')),
                  approved_differences=d.DIFFERENCES,mode=mode,common_member_identity_checked=True)
    if damage == 'same_builder': review['reviewer'] = 'builder'
    elif damage == 'binding': review['binding_sha256'] = 'wrong'
    elif damage == 'differences': review['approved_differences'] = []
    with pytest.raises(ValueError): d.review_check(review,mode,reg)


def test_private_worker_bypass_rejected(monkeypatch):
    monkeypatch.delenv('HB164_CONTROLLER_PID', raising=False)
    with pytest.raises(ValueError,match='supervised worker'):
        d.worker('timing','plain-1')


def test_worker_cli_dispatch_uses_controller_environment(monkeypatch):
    calls=[]
    monkeypatch.setenv('HB167_MODE','gold')
    monkeypatch.setenv('HB167_ARM','flip-1')
    monkeypatch.setattr(d,'worker',lambda mode,arm:calls.append((mode,arm)) or 0)
    assert d.main(['_worker']) == 0
    assert calls == [('gold','flip-1')]


def test_r224_worker_descriptor_and_private_configuration(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'WORK',tmp_path)
    monkeypatch.setenv('HB164_CONTROLLER_PID',str(os.getppid()))
    monkeypatch.setattr(d,'admitted',lambda mode:dict(binding_sha256='sealed'))
    monkeypatch.setattr(d,'lane',lambda mode:dict(uuid='synthetic'))
    monkeypatch.setattr(d,'roster',lambda mode:['a','b'])
    for name in ('WORK','TAG','__file__','admitted','lane_admission'):
        monkeypatch.setattr(d.r224,name,getattr(d.r224,name))
    d.save('timing','prepare',dict(series=[dict(StudyInstanceUID='a')]))
    (d.arm_work('timing','r224')/'data').mkdir(parents=True)
    d.io.atomic_json(d.arm_work('timing','r224')/'prepare.json',
                     dict(studies=['a','b'],series=[dict(StudyInstanceUID='a')]))
    def inherited_worker():
        assert d.r224.TAG == 'HB167_N32'
        assert d.r224.admitted()['binding_sha256'] == 'sealed'
        assert d.r224.lane_admission()['uuid'] == 'synthetic'
        descriptor=d.io.read_json(d.r224.WORK/'prepare.json')
        assert descriptor == dict(studies=['a','b'],series=[dict(StudyInstanceUID='a')])
        return 0
    monkeypatch.setattr(d.r224,'worker',inherited_worker)
    assert d.worker('timing','r224') == 0


def test_configure_preserves_frozen_module_file(monkeypatch,tmp_path):
    saved = d.sha(Path(d.r224.__file__))
    for name in ('WORK','TAG','__file__','admitted','lane_admission'):
        monkeypatch.setattr(d.r224,name,getattr(d.r224,name))
    monkeypatch.setattr(d,'WORK',tmp_path)
    monkeypatch.setattr(d,'admitted',lambda mode:dict(mode=mode))
    monkeypatch.setattr(d,'lane',lambda mode:mode)
    d.configure('gold','flip-1')
    assert d.r224.WORK == d.arm_work('gold','flip-1')
    assert d.r224.admitted() == dict(mode='gold')
    assert d.r224.lane_admission() == 'gold'
    assert d.r224.TAG == 'HB167_GOLD58'
    assert d.r224.__file__ == str(d.SCRIPT)
    assert d.sha(d.ROOT/'tools/c85r7_r224_timing.py') == saved


def test_exact_stage_and_instrumentation_contract():
    files = d.stage_files(d.STAGE)
    assert len(files) == 31
    raw = files['exp169_r7_src/compact_infer.py'].decode()
    code = d.instrument(raw)
    assert 'from tools import c85r7_r224flip_diagnostics as hb153' in code
    compile(code,'<diagnostic>','exec')
    assert d.command('timing')[-2:] == ['--mode','timing']
    assert d.TAGS['timing'] == 'HB167_N32'


@pytest.mark.parametrize('validation_seconds', [0.,5000.])
def test_run_preserves_timing_boundary_and_final_attempt(monkeypatch,tmp_path,measured,validation_seconds):
    arms,cal=measured
    clock=[0.]
    monkeypatch.setattr(d.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(d,'WORK',tmp_path)
    monkeypatch.setattr(d,'admitted',lambda mode:dict(binding_sha256='sealed'))
    monkeypatch.setattr(d,'preflight',lambda mode:dict(state='CPU_PATH_PREFLIGHT_PASS'))
    d.save('timing','validate',dict(preflight_sha256=d.digest(d.preflight('timing'))))
    monkeypatch.setattr(d,'lane',lambda mode:{})
    monkeypatch.setattr(d,'configure',lambda mode,arm:tmp_path/mode/arm)
    normalize = d.canonicalize_rows
    def canonicalize_rows(value,ids):
        clock[0]+=validation_seconds
        return normalize(value,ids)
    monkeypatch.setattr(d,'canonicalize_rows',canonicalize_rows)
    def check_arm(*args):
        assert [r['study'] for r in args[2]['rows']] == d.r7.cohort()[0]
        assert args[2]['completion_order'] == list(reversed(d.r7.cohort()[0]))
        clock[0]+=validation_seconds
    monkeypatch.setattr(d,'check_arm',check_arm)
    monkeypatch.setattr(d,'current',lambda mode:dict(binding=dict(calibration=cal)))
    (tmp_path/'timing/r224/data').mkdir(parents=True)
    def supervise(desktop):
        arm=os.environ['HB167_ARM']
        value=copy.deepcopy(arms[arm])
        value['rows'].reverse()
        clock[0]+=value['seconds']+(25. if arm=='r224' else 0.)
        if arm!='r224':
            value['seconds']/=2.
        value['binding_sha256']='sealed'
        return value,dict(complete=True)
    monkeypatch.setattr(d.r224,'supervise',supervise)
    value=d.run('timing')
    assert value['attempt_sha256']==d.sha(d.state_path('timing','attempt'))
    attempt=d.io.read_json(d.state_path('timing','attempt'))
    assert attempt['state']==value['state']
    for arm,measured_arm in attempt['arms'].items():
        assert measured_arm['seconds']==arms[arm]['seconds']
        if arm=='r224':
            assert measured_arm['controller_seconds']==125.
            assert 'worker_seconds' not in measured_arm
        else:
            assert measured_arm['controller_seconds']==arms[arm]['seconds']
            assert measured_arm['worker_seconds']==arms[arm]['seconds']/2.
    assert value['result']['projected_hours']==pytest.approx(7.38)
    with pytest.raises(ValueError, match='preserve previous attempt'):
        d.run('timing')


@pytest.fixture
def paths(monkeypatch, tmp_path):
    from tools import exp169_r7_port as port
    from tools import exp189_s6_port as s6
    monkeypatch.setattr(d, 'WORK', tmp_path/'owned')
    monkeypatch.setattr(d, 'R224_DATA', tmp_path/'hb164')
    rows = [dict(StudyInstanceUID='2', SeriesInstanceUID='2.2', Anatomical_Plane='Sagittal'),
            dict(StudyInstanceUID='1', SeriesInstanceUID='1.1', Anatomical_Plane='Coronal')]
    ids = ['1','2']  # roster and CSV have distinct, deliberately retained order
    monkeypatch.setattr(d.r7, 'cohort', lambda:(ids,rows))
    monkeypatch.setattr(d, 'roster', lambda mode:ids)
    monkeypatch.setattr(d, 'series_rows', lambda mode:rows)
    for row in rows:
        p=d.R224_DATA/'data/train_series'/row['StudyInstanceUID']/row['SeriesInstanceUID']/'1.dcm'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'real-file-open-synthetic-dicom')
    # Original failure: authenticated data contains only train_series/, no CSV.
    assert not (d.R224_DATA/'data/train_series.csv').exists()
    monkeypatch.setattr(d.gold, 'GOLD_DICOM', tmp_path/'gold-data')
    d.gold.GOLD_DICOM.mkdir()
    (d.gold.GOLD_DICOM/'train_series.csv').write_bytes(d.series_bytes(rows))
    d.junction(d.WORK/'gold-source', d.R224_DATA/'data/train_series')
    # Gold uses ordinary directories, matching the real prepared Gold source.
    for row in rows:
        p=d.gold.GOLD_DICOM/'train_series'/row['StudyInstanceUID']/row['SeriesInstanceUID']/'1.dcm'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'gold-input')
    monkeypatch.setattr(port, 'verify_bundle', lambda *a,**kw:{})
    monkeypatch.setattr(s6, 'verify_code', lambda *a,**kw:dict(tree_sha256='code'))
    checkpoint=tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    monkeypatch.setattr(d.r224,'CHECKPOINT',checkpoint)
    monkeypatch.setattr(d.r224,'CHECKPOINT_SHA',d.sha(checkpoint))
    monkeypatch.setattr(d.r224,'source_parts',lambda:None)
    monkeypatch.setattr(d.r224,'GPU_PYTHON',checkpoint)
    monkeypatch.setattr(d.gold,'RUNTIME',tmp_path/'runtime')
    for name in ('pydicom','timm'):
        p=d.gold.RUNTIME/'deps'/name/'__init__.py'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'# test dependency')
    monkeypatch.setattr(d,'stage_files',lambda stage:{'port.py':b'# frozen source'})
    for mode in ('timing','gold'):
        d.prepare_data(mode)
        for variant in ('plain','flip'):
            p=d.mode_work(mode)/'cpu-runtime'/variant/'embedded/port.py'
            p.parent.mkdir(parents=True)
            p.write_bytes(b'# frozen source')
    work=d.arm_work('timing','r224')
    d.io.atomic_json(work/'prepare.json',dict(studies=ids,series=rows))
    d.junction(work/'data',d.data_source('timing'))
    return ids,rows


def test_owned_csv_resolved_worker_staging_and_hash_order(paths, tmp_path):
    from tools import exp189_s6_port as s6
    ids,rows=paths
    contract=d.data_contract('timing')
    assert contract['csv_sha256']==d.pack.digest(d.series_bytes(rows))
    resolved_ids,resolved=s6._study_ids([d.data_source('timing')/'train_series'/u for u in ids])
    assert resolved_ids==ids
    assert all(p.parent.parent==d.data_source('timing').resolve() for p in resolved)
    stage=tmp_path/'staging'
    stage.mkdir()
    s6._stage_studies(resolved,ids,stage)  # exact formerly failing CSV-open path
    assert (stage/'test_series.csv').read_bytes()==d.series_bytes(rows)
    assert not (d.R224_DATA/'data/train_series.csv').exists()
    before=d.data_contract('timing')
    d.prepare_data('timing')
    assert d.data_contract('timing')==before
    (d.data_source('timing')/'train_series.csv').write_bytes(d.series_bytes(list(reversed(rows))))
    with pytest.raises(ValueError,match='row/order drift'):
        d.data_contract('timing')


@pytest.mark.parametrize('mode',['timing','gold'])
def test_preflight_opens_all_arms_before_gpu(paths, mode):
    value=d.preflight(mode)
    assert set(value['arms'])==set(d.ORDER if mode=='timing' else ('plain-1','flip-1'))
    assert value['cuda_executed'] is False
    assert all(v['dicom_files']==2 for v in value['arms'].values())


@pytest.mark.parametrize('damage',['csv','dicom','checkpoint','deps','runtime','descriptor','root_junction'])
def test_preflight_missing_worker_inputs_rejected(paths, monkeypatch, damage, tmp_path):
    if damage=='csv': (d.data_source('timing')/'train_series.csv').unlink()
    elif damage=='dicom':
        (d.R224_DATA/'data/train_series/1/1.1/1.dcm').unlink()
    elif damage=='checkpoint': d.r224.CHECKPOINT.unlink()
    elif damage=='deps': (d.gold.RUNTIME/'deps/timm/__init__.py').unlink()
    elif damage=='runtime': (d.mode_work('timing')/'cpu-runtime/flip/embedded/port.py').unlink()
    elif damage=='descriptor': (d.arm_work('timing','r224')/'prepare.json').unlink()
    elif damage=='root_junction':
        alternate=d.WORK/'wrong-root'
        alternate.mkdir()
        (alternate/'train_series.csv').write_bytes(d.series_bytes(paths[1]))
        d.junction(alternate/'train_series',d.R224_DATA/'data/train_series')
        monkeypatch.setattr(d,'data_source',lambda mode:alternate)
    with pytest.raises((OSError,ValueError)):
        d.preflight('timing')


def test_preflight_blocks_run_before_gpu_and_preserves_legacy(paths, monkeypatch):
    legacy=d.WORK/'timing'
    (legacy/'plain-1').mkdir(exist_ok=True)
    (legacy/'attempt.json').write_bytes(b'failed evidence')
    (legacy/'plain-1/worker-status.json').write_bytes(b'failed worker')
    before={p:d.sha(p) for p in (legacy/'attempt.json',legacy/'plain-1/worker-status.json')}
    monkeypatch.setattr(d,'admitted',lambda mode:dict(binding_sha256='sealed'))
    monkeypatch.setattr(d,'lane',lambda mode:pytest.fail('GPU reached'))
    (d.data_source('timing')/'train_series.csv').unlink()
    with pytest.raises(OSError): d.run('timing')
    assert before=={p:d.sha(p) for p in before}
    assert not d.state_path('timing','attempt').exists()
