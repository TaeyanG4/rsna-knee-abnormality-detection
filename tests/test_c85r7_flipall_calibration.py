"""v2.1 calibration/admission tests: CPU only, all CUDA calls mocked."""
import io
import zipfile
from types import SimpleNamespace

import pytest
from tools import c85r7_flipall_timing as t
from tools import c85r7_flipall_measure as m


def desktop():
    return dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER', total_mib=16376,
                used_mib=2048, uuid='GPU-same')


def calibrated(monkeypatch, tmp_path, real=False, failure=None, seconds_scale=1):
    selection = t.calibration_selection() if real else None
    parent = tmp_path/'timing'
    work = parent/'calibration'
    monkeypatch.setattr(t, 'WORK', work)
    monkeypatch.setattr(t, 'ADMISSION', parent/'admission.json')
    monkeypatch.setattr(t, 'CALIBRATING', True)
    if real:
        ids = selection['studies']
        rows = [r for r in t.cohort()[1] if r['StudyInstanceUID'] in ids]
        archive = tmp_path/'real-series-samples.zip'
        found = set()
        import pydicom
        with zipfile.ZipFile(t.ZIP) as source, zipfile.ZipFile(archive, 'w') as target:
            for item in source.infolist():
                p = item.filename.split('/')
                if len(p) == 4 and p[0] in ('train_images', 'train_series') and p[1] in ids and p[3].endswith('.dcm') and (p[1],p[2]) not in found:
                    raw = source.read(item)
                    assert pydicom.dcmread(io.BytesIO(raw)).pixel_array.size > 0
                    target.writestr(item.filename, raw)
                    found.add((p[1],p[2]))
        assert found == {(r['StudyInstanceUID'], r['SeriesInstanceUID']) for r in rows}
        monkeypatch.setattr(t, 'ZIP', archive)
        monkeypatch.setattr(t, 'cohort', lambda: (ids, rows))
    else:
        ids = ['1','2','3','4']
        selection = dict(studies=ids, largest_admissible='1')
        monkeypatch.setattr(t, 'bindings', lambda: {'test': 'bound'})
    monkeypatch.setattr(t, 'calibration_selection', lambda deadline=None: selection)
    lane_calls = []
    def lane(memory):
        lane_calls.append(memory)
        return 'GPU-same' if failure != 'device' or len(lane_calls) < 3 else 'GPU-different'
    monkeypatch.setattr(t, 'lane_admission', lane)
    calls = []
    manifest = t.b.read(t.MANIFEST)
    def supervise(arm, healthy, deadline):
        calls.append((arm, healthy, deadline))
        assert deadline <= t.time.time()  # attempt start identity, not an expiry
        if failure == 'prepare':
            raise ValueError('preparation failed')
        if arm == 'prepare':
            if real:
                t.prepare_data()
                t.prepare_data()  # verified, resume-safe reuse
            else:
                t.io.journal(work/'cohort.json', dict(studies=ids, files={'synthetic': 'cpu-only'}), work)
                t.io.journal(work/'selection.json', selection, work)
            return 1.
        flip = arm.startswith('flip')
        runtime = dict(models=manifest['models'], models2=[], flip_tta=flip, workers=2,
                       **{k:manifest[k] for k in ('size','crop','views','slices')})
        if failure == 'shape':
            runtime['crop'] = 224
        cuda = SimpleNamespace(synchronize=lambda:None, max_memory_reserved=lambda:6e9 if flip else 5e9,
                               max_memory_allocated=lambda:4e9, get_device_name=lambda:'NVIDIA GeForce RTX 4070 Ti SUPER')
        measured=[]
        for uid in ids:
            m.commit(SimpleNamespace(cuda=cuda), work/(arm+'-measure-status.json'), measured,
                     uid, 200, 192, t.time.perf_counter()-1, 0., 0., 0., runtime=runtime)
        seconds = (80. if flip else 40.)*seconds_scale
        t.io.journal(work/(arm+'-timing.json'), dict(seconds=seconds, error=None, exit_code=0, cleanup=[],
                                                  desktop_before_cuda=desktop(), peak_gpu_mib=0 if failure == 'telemetry' else 7000), work)
        return seconds
    monkeypatch.setattr(t, 'supervise', supervise)
    result = t.calibrate()
    return result, calls, lane_calls, parent


def make_review(parent):
    path = parent/'calibration/review.json'
    template = t.b.read(parent/'calibration/review-template.json')
    template.update(verdict='PASS', independent_reviewer='independent test reviewer; SYNTHETIC ONLY', independent=True)
    t.io.journal(path, template, parent)
    return path


def test_producer_review_admit_flow(monkeypatch, tmp_path):
    result, calls, lane, parent = calibrated(monkeypatch, tmp_path)
    assert [a for a,h,d in calls] == ['prepare', *t.CALIBRATION_ORDER]
    assert len({d for a,h,d in calls}) == 1
    assert calls[0][1] is None and calls[-1][1] == 6.
    assert lane[0] is None and lane[-1]['plain_peak_reserved_bytes'] == 5e9
    assert result['healthy_studies_per_minute'] == 3.
    assert result['max_windows'] is None and result['max_observed_windows'] == 200
    review = make_review(parent)
    monkeypatch.setattr(t, 'WORK', parent)
    t.admit(review, dry_run=True)
    assert not t.ADMISSION.exists()
    value = t.admit(review)
    assert t.admission(t.b.read(t.ADMISSION)) == value
    assert value['memory_receipt_sha256'] == t.b.digest((parent/'calibration/memory-receipt.json').read_bytes())
    # Production run CLI consumes the generated admission, with no manual JSON.
    monkeypatch.setattr(t.b, 'validate', lambda: None)
    seen=[]
    monkeypatch.setattr(t, 'run', lambda v: seen.append(v) or dict(status='TEST',projected_hidden_seconds=1,t4_peak_bytes=1))
    assert t.main(['run']) == 0 and seen == [value]


def test_complete_slow_measurements_are_not_rejected_by_elapsed_time(monkeypatch, tmp_path):
    result, _, _, parent = calibrated(monkeypatch, tmp_path, seconds_scale=150)
    result['finished'] = result['started']+19000
    assert t.validate_calibration(result)['status'] == 'PASS'
    assert result['per_mode']['plain-cal']['seconds'] == 6000
    assert result['per_mode']['flip-cal']['seconds'] == 12000


@pytest.mark.parametrize('field,value', [('peak_reserved_bytes', 12.5*t.GIB),
                                       ('desktop_before_cuda', dict(device_name='4070 Ti SUPER',
                                        total_mib=16376, used_mib=12000, uuid='GPU-same'))])
def test_measured_calibration_gate_rejects_peak_or_headroom(monkeypatch, tmp_path, field, value):
    result, _, _, _ = calibrated(monkeypatch, tmp_path)
    arm = result['per_mode']['flip-cal']
    if field == 'peak_reserved_bytes':
        for row in arm['rows']:
            row[field] = value
        arm[field] = value
    else:
        arm[field] = value
    with pytest.raises(ValueError, match='12 GiB|headroom'):
        t.validate_calibration(result)


@pytest.mark.parametrize('failure', ['prepare','shape','telemetry','device'])
def test_failed_calibration_preserves_partial_no_memory_pass(monkeypatch, tmp_path, failure):
    with pytest.raises(ValueError):
        calibrated(monkeypatch, tmp_path, failure=failure)
    work = tmp_path/'timing/calibration'
    assert t.b.read(work/'attempt.json')['state'] == 'ABORTED_DO_NOT_SUBMIT'
    assert not (work/'memory-receipt.json').exists()


@pytest.mark.parametrize('key,value', [('verdict','HOLD'), ('independent',False), ('memory_receipt_sha256','bad'),
                                       ('bindings',{}), ('independent_reviewer','')])
def test_review_spoof_or_stale_binding_blocks(monkeypatch, tmp_path, key, value):
    result, _, _, parent = calibrated(monkeypatch,tmp_path)
    review = make_review(parent)
    data = t.b.read(review)
    data[key] = value
    t.io.journal(review,data,parent)
    monkeypatch.setattr(t,'WORK',parent)
    with pytest.raises(ValueError,match='independent PASS'):
        t.admit(review)
    assert not t.ADMISSION.exists()


@pytest.mark.parametrize('key,value', [('healthy_studies_per_minute',300.), ('peak_reserved_bytes',11e9),
                                       ('full_shape',False), ('timing_version','2.0.0'), ('finished',float('inf'))])
def test_memory_tampering_blocks(monkeypatch,tmp_path,key,value):
    result, _, _, parent = calibrated(monkeypatch,tmp_path)
    result[key] = value
    with pytest.raises(ValueError):
        t.validate_calibration(result)


def test_raw_evidence_mutation_and_attempt_reuse_block(monkeypatch,tmp_path):
    result, _, _, parent = calibrated(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='preserve previous calibration'):
        t.calibrate()
    (parent/'calibration/plain-cal-timing.json').write_text('{}')
    with pytest.raises(ValueError):
        t.validate_calibration(result)


def test_real_archive_serving_selection():
    selection = t.calibration_selection()
    inventory = t.selection_inventory()
    assert len(set(selection['studies'])) == 4
    assert selection['largest_admissible'] == selection['studies'][0]
    checked = {r['study']:r for r in selection['checked']}
    assert all(checked[u]['error'] is None and checked[u]['windows'] > 0 for u in selection['studies'])
    assert max(checked[u]['windows'] for u in selection['studies']) > 1024
    assert selection['studies'][0] == next(u for u in inventory['ranked'] if checked[u]['error'] is None and checked[u]['windows'] > 0)


def test_real_selection_algorithm_preserves_serving_refusals():
    inventory = dict(ranked=['a','b','c','d','e','f'],study_images={u:6-i for i,u in enumerate('abcdef')})
    def probe(u):
        return dict(error='no usable window' if u == 'a' else None, windows=0 if u == 'a' else 1557)
    got = t.select_admissible(inventory,probe)
    assert got['studies'] == list('bcde') and got['largest_admissible'] == 'b'
    assert got['serving_refusals'] == [dict(study='a',error='no usable window',windows=0)]
    assert got['unassessed'] == ['f']


def test_real_data_cpu_calibrate_admit_smoke(monkeypatch,tmp_path):
    # Real selected series DICOMs, preparation/reuse, frozen manifest and hash
    # bindings. Only GPU lane, CUDA/model calls and telemetry are substituted.
    result, _, _, parent = calibrated(monkeypatch,tmp_path,real=True)
    assert result['full_shape'] and result['four_folds']
    review = make_review(parent)
    monkeypatch.setattr(t,'WORK',parent)
    assert t.main(['admit','--review',str(review),'--dry-run']) == 0
    assert not t.ADMISSION.exists()
    assert t.main(['admit','--review',str(review)]) == 0
    assert t.main(['admit','--dry-run']) == 0


def test_calibration_serving_call_flags_match_authenticated_parent():
    import ast
    import inspect
    parent = t.b.parent()[1]['exp169_release_c85r7.py'].decode()
    def flags(source):
        calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'infer_studies']
        return [{k.arg: ast.literal_eval(k.value) for k in n.keywords
                 if k.arg in ('workers','amp','device','max_fallback_fraction')} for n in calls]
    assert flags(inspect.getsource(t.arm_run)) == flags(parent) == [
        dict(workers=2,amp='auto',device='cuda',max_fallback_fraction=1.0)]


def test_cohort_admission_tamper_rejected(monkeypatch,tmp_path):
    original = t.b.read(t.ADMISSION.parent/'cohort-admission.json')
    monkeypatch.setattr(t,'ADMISSION',tmp_path/'admission.json')
    original['selection']['studies'].reverse()
    t.io.journal(tmp_path/'cohort-admission.json',original,tmp_path)
    with pytest.raises(ValueError,match='selection mismatch'):
        t.calibration_selection()


def test_calibration_cli_cannot_bypass_run_review(monkeypatch,tmp_path):
    monkeypatch.setattr(t,'WORK',tmp_path)
    monkeypatch.setattr(t,'ADMISSION',tmp_path/'missing.json')
    monkeypatch.setattr(t,'CALIBRATING',False)
    monkeypatch.setattr(t.b,'validate',lambda:None)
    monkeypatch.setattr(t,'lane_admission',lambda *a:pytest.fail('unreviewed run touched GPU'))
    with pytest.raises(SystemExit):
        t.main(['run','--calibration'])
    with pytest.raises(OSError):
        t.main(['run'])
    with pytest.raises(OSError):
        t.main(['_arm','--calibration','--arm','flip-cal','--attempt-started',str(t.time.time())])


def test_calibration_idle_watchdog_real_cpu_child(monkeypatch,tmp_path):
    import subprocess
    import sys
    monkeypatch.setattr(t, 'WORK', tmp_path)
    monkeypatch.setattr(t, 'desktop_snapshot', desktop)
    monkeypatch.setattr(t, 'CALIBRATING', True)
    monkeypatch.setattr(t.monitor, '_gpu_sample', lambda env: (100, 0, 0))
    popen = subprocess.Popen
    children = []
    def child(command, **kwargs):
        assert '--calibration' in command and '_prepare' in command
        process = popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(t.subprocess, 'Popen', child)
    monkeypatch.setattr(t, 'IDLE_SECONDS', .3)
    with pytest.raises(ValueError, match='idle watchdog'):
        t.supervise('prepare', None)
    assert all(p.poll() is not None for p in children)
    assert t.b.read(tmp_path/'prepare-timing.json')['error']
