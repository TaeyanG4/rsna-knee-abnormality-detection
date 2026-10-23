"""H-B-167 diagnostics v1.0.7: measured-share N32 and full-blend Gold58.

Changelog 1.0.7: candidates-* delegates authenticated stack/mirror Gold58
evaluation, registered paired bootstrap and one review-bound R224 pair job.
Backup: .local/backups/hb167-gold-candidates-v107. Historical paths unchanged.

Changelog 1.0.6: bounded identity-safe post-cleanup wait with recorded recovery;
CPU-only preserved-arm evaluation, sealed evidence and historical admission checks.

Changelog 1.0.5: fresh owned work folder for COMPLETE common-cache binding;
CPU identity-control replay with authenticated preserved Gold58 R7 output.

Changelog 1.0.4: share unchanged common-cache validation with the producer so
it can validate an in-memory candidate before publishing COMPLETE.

Changelog 1.0.3: canonicalize completed arm rows by exact unique cohort identity
outside measurement clocks; retain completion order for audit and all validators.
Changelog 1.0.2: owned N32 metadata/series junctions, CPU path preflight and
versioned attempt isolation; preserve all v1.0.1 failure evidence.
Changelog 1.0.1: preserve prior R7 supervisor-wall and R224 worker timing boundaries.
Changelog 1.0.0: compose the existing R7 ABBA and R224 measurement methods.
No serving stage is modified; no Kaggle API, downloads or automatic GPU launch.
GPU actions require B's gpu_queue/gpu_job lock and an independent admission.
Gold58 is EXPOSED, clean_oof=false; a full-blend check requires authenticated
common-member predictions from the exact plain R224 parent, never an R7-only
proxy. Its absence is a binding HOLD, not a passing no-harm result.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import io as streams
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r224_timing as r224
from tools import c85r7_flipall_timing as r7
from tools import c85r7_flipall_measure as measure
from tools import c85r7_flipall_v1115 as pack
from tools import hb145_output_io as io
from tools import hb128_gold58_pbranch_v1 as gold
from tools import exp169_r224 as formula

VERSION = '1.0.7'
WORK = ROOT / '.local/scratch/astra-b1/hb167-gold-bind'
PARENT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.23-r224q'
STAGE = PARENT.parent / 'v1.1.24-r224flipq'
SCRIPT = Path(__file__).resolve()
TAGS = dict(timing='HB167_N32', gold='HB167_GOLD58')
ORDER = ('plain-1', 'flip-1', 'flip-2', 'plain-2', 'r224')
COMMON = ('R', 'R224', 'EXP143v2', 'EXP155', 'Q_A5q')
R224_DATA = ROOT / '.local/scratch/hb164-r224/hb164/timing'
CODE = ROOT/'.local/exp189-cpu-v120/input/cloud_code'
ANATOMY = ROOT/'.local/exp189-cpu-v120/input/anatomy-fast.pt'
need, sha, digest = r224.need, r224.sha, r224.digest
_MONITOR = r224.monitor
_CLEANUP = _MONITOR._cleanup
HISTORICAL_BACKUP = ROOT/'.local/backups/hb167-gold-cleanup-v106'
DIFFERENCES = [
    'Local diagnostics only: no changes to packaged serving sources or launch mechanics.',
    'N32: exact c85r7_flipall_timing.cohort, R7 ABBA, then original R224 functions; one 4070, sequential.',
    'R7 uses original workers=2, chunk192 and auto precision as the prior N32 method; local Ada/T4 transfer is assumed.',
    'R224 uses the prior AST-only canonical sequential reader and original 94-window fp16/fp32 retry path.',
    'R7 supervisor wall and R224 worker seconds match prior tools; startup/decode included; maximum wall/GPU R7 pair ratio; no speedup or GPU-count credit.',
    'Gold58 uses the exact exp169_r224.blend, common parent members, newly paired plain/full-flip R7; exposed only.',
    'Historical hidden durations are user-supplied upper bounds, not authenticated runtimes or a rigorous combined bound.',
    'v1.0.2 owns N32 CSV and study directories; series junctions are read-only inputs. CPU preflight opens every arm input before GPU; versioned attempts preserve v1.0.1.',
    'v1.0.3 canonicalizes each completed arm row list to the exact unique cohort after timer capture, retaining completion_order; predictions and all validation/timing rules are unchanged.',
]


def stamp(mode, **value):
    return dict(identity=TAGS[mode], version=VERSION, exposure='EXPOSED', clean_oof=False, **value)


def mode_work(mode):
    need(mode in TAGS, 'unknown mode')
    return WORK / mode / ('v' + VERSION)


def arm_work(mode, arm):
    need(arm in (ORDER if mode == 'timing' else ('plain-1', 'flip-1')), 'unknown arm')
    return mode_work(mode) / arm


def state_path(mode, name):
    return mode_work(mode) / (name + '.json')


def save(mode, name, value):
    io.journal(state_path(mode, name), value, mode_work(mode))
    return value


def stage_files(stage):
    from tools import c85r7_flipallq_v1122 as flipq
    return pack.unpack(flipq.restored(io.read_json(stage / 'kernel/release.ipynb')))[0]


def validate_stage():
    from tools import c85r7_r224flipq_v1124 as builder
    builder.validate()
    parent, candidate = stage_files(PARENT), stage_files(STAGE)
    need(set(parent) == set(candidate), 'stage embedded file inventory changed')
    need(candidate['exp169_r7_port.py'] == pack.replace(parent['exp169_r7_port.py'], pack.FLAG_OLD, pack.FLAG_NEW),
         'exact FULL flip port patch required')
    need(candidate['exp169_r224.py'] == parent['exp169_r224.py'] == (ROOT/'tools/exp169_r224.py').read_bytes(),
         'common R224 formula bytes changed')
    return {str(stage.relative_to(ROOT)): {name: sha(stage/name)
            for name in ('kernel/release.ipynb', 'kernel/kernel-metadata.json', 'build-receipt.json')}
            for stage in (PARENT, STAGE)}


window_gate = measure.window_gate


def commit(*args, **kwargs):
    """Preserve the frozen measurement, add supervisor's seconds alias atomically."""
    def journal(path, value, work):
        for row in value['rows']:
            row['seconds'] = row['study_seconds']
        io.journal(path, value, work)
    namespace = dict(measure.commit.__globals__, io=types.SimpleNamespace(journal=journal))
    function = types.FunctionType(measure.commit.__code__, namespace, argdefs=measure.commit.__defaults__)
    return function(*args, **kwargs)


def instrument(source):
    value = measure.instrument(source)
    return pack.replace(value.encode(),
        b'from tools import c85r7_flipall_measure as hb153',
        b'from tools import c85r7_r224flip_diagnostics as hb153').decode()


def roster(mode):
    return r7.cohort()[0] if mode == 'timing' else gold.gold()[0]


def data_source(mode):
    return mode_work(mode)/'data' if mode == 'timing' else gold.GOLD_DICOM


def series_rows(mode):
    if mode == 'timing':
        return r7.cohort()[1]  # exact original metadata order, no label access
    ids = set(roster(mode))
    with r7.SERIES.open(newline='', encoding='utf-8-sig') as stream:
        return [r for r in csv.DictReader(stream) if r['StudyInstanceUID'] in ids]


def series_bytes(rows):
    stream = streams.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def junction(link, target):
    """Create only owned links; targets are opened read-only, never modified.

    NTFS junctions do not enforce read-only access: this is an I/O contract,
    not an ACL change to shared data.
    """
    link, target = Path(link), Path(target).resolve(strict=True)
    need(link.parent.resolve().is_relative_to(WORK.resolve()), 'junction outside harness scratch')
    if link.exists() or link.is_symlink():
        need(link.resolve() == target, 'existing junction target drift')
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    need(os.name == 'nt', 'Windows junction required')
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                   check=True, capture_output=True, text=True)
    need(link.resolve() == target, 'junction creation failed')


def prepare_data(mode):
    if mode != 'timing':
        return
    data, rows = data_source(mode), series_rows(mode)
    data.mkdir(parents=True, exist_ok=True)
    raw = series_bytes(rows)
    csv_path = data/'train_series.csv'
    if csv_path.exists():
        need(csv_path.read_bytes() == raw, 'owned CSV drift; do not overwrite')
    else:
        temporary = csv_path.with_suffix('.csv.tmp')
        io.retry(temporary.write_bytes, raw)
        io.replace(temporary, csv_path)
    # _study_ids resolves study paths. A whole-root or study-level junction
    # would escape back to HB164's CSV-less root. Only series are junctions.
    for row in rows:
        uid, sid = row['StudyInstanceUID'], row['SeriesInstanceUID']
        need(all(part and set(part) <= set('0123456789.') for part in (uid, sid)), 'unsafe series path')
        study = data/'train_series'/uid
        study.mkdir(parents=True, exist_ok=True)
        junction(study/sid, R224_DATA/'data/train_series'/uid/sid)


def data_contract(mode):
    rows = series_rows(mode)
    expected = series_bytes(rows)
    path = data_source(mode)/'train_series.csv'
    # Registration binds exact intended bytes; prepare materializes them.
    if path.exists():
        with path.open(newline='', encoding='utf-8-sig') as stream:
            need(list(csv.DictReader(stream)) == rows, 'series CSV row/order drift')
        if mode == 'timing':
            need(path.read_bytes() == expected, 'N32 CSV byte drift')
    return dict(path=str(path), rows_sha256=digest(rows),
                csv_sha256=pack.digest(expected) if mode == 'timing' else sha(path),
                series_links='per-series; real study directories' if mode == 'timing' else 'existing authenticated Gold tree')


def preflight(mode):
    """Open worker inputs on CPU, including the exact resolved R7 CSV path."""
    from tools import exp189_s6_port as s6
    from tools import exp169_r7_port as port
    ids, rows = roster(mode), series_rows(mode)
    data = data_source(mode)
    contract = data_contract(mode)
    resolved_ids, paths = s6._study_ids([data/'train_series'/uid for uid in ids])
    need(resolved_ids == ids and {p.parent.parent for p in paths} == {data.resolve()},
         'resolved R7 paths escape owned metadata root')
    csv_path = paths[0].parent.parent/'train_series.csv'
    need(sha(csv_path) == contract['csv_sha256'], 'worker CSV pin mismatch')
    opened = []
    for row in rows:
        directory = data/'train_series'/row['StudyInstanceUID']/row['SeriesInstanceUID']
        if mode == 'timing':
            need(directory.resolve(strict=True) ==
                 (R224_DATA/'data/train_series'/row['StudyInstanceUID']/row['SeriesInstanceUID']).resolve(strict=True),
                 'N32 series junction target drift')
        files = sorted(directory.glob('*.dcm'))
        need(files, 'missing required DICOM series')
        for path in files:
            with path.open('rb') as stream:
                need(bool(stream.read(1)), 'empty DICOM input')
            opened.append(str(path.relative_to(data)))
    # Opens all model, code and anatomy files via the original hash verifiers.
    port.verify_bundle(r7.MANIFEST.parent, expected_manifest_sha256=gold.R7_MANIFEST_SHA)
    code = s6.verify_code(CODE, ANATOMY)
    need(sha(r224.CHECKPOINT) == r224.CHECKPOINT_SHA, 'R224 checkpoint drift')
    r224.source_parts()
    with r224.GPU_PYTHON.open('rb') as stream:
        need(bool(stream.read(1)), 'GPU Python missing')
    deps = gold.RUNTIME/'deps'
    for name in ('pydicom', 'timm'):
        with (deps/name/'__init__.py').open('rb') as stream:
            need(bool(stream.read(1)), 'runtime dependency missing')
    arms = {}
    for arm in ORDER if mode == 'timing' else ('plain-1', 'flip-1'):
        work = arm_work(mode, arm)
        if arm == 'r224':
            descriptor = io.read_json(work/'prepare.json')
            need(descriptor == dict(studies=ids, series=rows), 'R224 descriptor drift')
            need((work/'data').resolve() == data.resolve(), 'R224 data path drift')
            with (work/'data/train_series.csv').open('rb') as stream:
                need(bool(stream.read(1)), 'R224 data metadata missing')
        else:
            runtime = mode_work(mode)/'cpu-runtime'/('plain' if arm.startswith('plain') else 'flip')/'embedded'
            for name, raw in stage_files(PARENT if arm.startswith('plain') else STAGE).items():
                need(sha(runtime/name) == pack.digest(raw), 'prepared runtime source drift')
        arms[arm] = dict(data=str(data), csv=str(csv_path), dicom_files=len(opened),
                         work=str(work), opened_paths_sha256=digest(opened))
    return dict(state='CPU_PATH_PREFLIGHT_PASS', arms=arms, metadata=contract,
                code_tree_sha256=code['tree_sha256'], cuda_executed=False)


def verify_data(mode):
    ids = roster(mode)
    if mode == 'timing':
        record = io.read_json(R224_DATA/'prepare.json')
        need(record.get('state') == 'CPU_PREPARED' and record['studies'] == ids and
             record.get('real_study_cpu_replay', {}).get('exact_window_parity') is True,
             'complete existing R224 N32 preparation required')
        for name, value in record['files'].items():
            need(sha(R224_DATA/'data'/name) == value, 'existing N32 DICOM hash drift')
        result = dict(manifest_sha256=sha(R224_DATA/'prepare.json'), files=len(record['files']))
    else:
        result = gold.verify_dicom(gold.GOLD_DICOM, ids)
    return dict(source=str(data_source(mode)), **result)


def read_common(path, ids):
    """Binding plus independent admission authenticates externally generated common caches."""
    need(path is not None, 'Gold58 requires --common authenticated parent common-member receipt')
    path = Path(path).resolve()
    need(path.is_relative_to(ROOT), 'common receipt must be in project')
    receipt = io.read_json(path)
    arrays, failed = validate_common(receipt, ids)
    return dict(path=str(path), sha256=sha(path), receipt=receipt), arrays, failed


def validate_common(receipt, ids):
    """Validate a candidate without publishing it; read_common remains the disk API."""
    import numpy as np
    need(receipt.get('schema') == 'HB167-parent-common/v1' and receipt.get('state') == 'COMPLETE' and
         receipt.get('parent_kernel_sha256') == sha(PARENT/'kernel/release.ipynb') and
         receipt.get('clean_oof') is False and receipt.get('exposure') == 'EXPOSED',
         'plain R224 common-cache identity or exposure mismatch')
    need(receipt.get('formula_sha256') == sha(ROOT/'tools/exp169_r224.py') and
         set(receipt.get('members', {})) == set(COMMON), 'common formula/member inventory mismatch')
    for member, proof in receipt['members'].items():
        need(proof.get('serving_source_sha256') in {pack.digest(v) for v in stage_files(PARENT).values()},
             'common member serving source not embedded in parent: ' + member)
        evidence = (ROOT/proof['receipt_path']).resolve()
        need(evidence.is_relative_to(ROOT) and sha(evidence) == proof['receipt_sha256'], 'common member receipt pin mismatch')
        need(proof.get('checkpoint_sha256') and len(proof['checkpoint_sha256']) == 64, 'checkpoint identity required')
    need(receipt['members']['R224']['checkpoint_sha256'] == r224.CHECKPOINT_SHA, 'R224 common checkpoint mismatch')
    predictions = (ROOT/receipt['predictions_path']).resolve()
    need(predictions.is_relative_to(ROOT) and sha(predictions) == receipt['predictions_sha256'], 'common predictions pin mismatch')
    with np.load(predictions, allow_pickle=False) as data:
        need(data['ids'].tolist() == ids and data['targets'].tolist() == list(gold.LABELS), 'common prediction order mismatch')
        arrays = {m: np.array(data[m], dtype=np.float64) for m in COMMON}
        failed = {m: np.array(data['failed_'+m]) for m in COMMON}
    for m in COMMON:
        need(arrays[m].shape == (58, 12) and np.isfinite(arrays[m]).all() and
             ((arrays[m] >= 0) & (arrays[m] <= 1)).all(), 'invalid common probabilities')
        need(failed[m].shape == (58,) and failed[m].dtype == np.bool_, 'explicit boolean failure masks required')
    formula.failure_gate(failed['R224'] | np.all(arrays['R224'] == .5, axis=1))
    return arrays, failed


def empirical_estimate():
    # Two separate upper bounds do not identify a combined upper bound: baseline
    # and interaction are unknown. Preserve the user's estimate separately.
    return dict(source='User evidence 2026-10-15 15:22 KST; not polled',
                r224_submission=57200805, flip_submission=57198522,
                r224_upper_hours=5.3, flip_upper_hours=6.3, baseline_reference_hours=5.38,
                conditional_additive_estimate_hours=5.3+6.3-5.38,
                user_combined_planning_estimate_hours=7.1, rigorous_upper_bound=False,
                runtime_gate_uses_this=False,
                caveat='6.22h additive estimate assumes comparable nodes, same baseline and no interaction; '
                       '7.1h is the supplied planning envelope. Neither is a proven combined bound.')


def projection(arms, calibration):
    ids = r7.cohort()[0]
    need(len(ids) == len(set(ids)) == 32 and set(arms) == set(ORDER), 'exact N32 and five arms required')
    need(type(calibration['r7_hidden_seconds']) in (int, float) and
         math.isfinite(calibration['r7_hidden_seconds']) and
         0 < calibration['r7_hidden_seconds'] <= r224.BASELINE_HOURS*3600, 'invalid calibration')
    for name, arm in arms.items():
        need(type(arm['seconds']) in (int, float) and math.isfinite(arm['seconds']) and arm['seconds'] > 0,
             'invalid arm elapsed time')
        need([row['study'] for row in arm['rows']] == ids, 'incomplete or reordered N32 arm')
        if name != 'r224':
            need(all(math.isfinite(row['gpu_seconds']) and row['gpu_seconds'] > 0 for row in arm['rows']),
                 'invalid GPU elapsed time')
    ratios = []
    for i in (1, 2):
        a, b = arms['plain-'+str(i)], arms['flip-'+str(i)]
        need(len(a['rows']) == len(b['rows']) == 32, 'complete paired N32 required')
        need([r['study'] for r in a['rows']] == [r['study'] for r in b['rows']] == ids, 'paired N32 order drift')
        need([(r['windows'], r['batch_windows']) for r in a['rows']] ==
             [(r['windows'], r['batch_windows']) for r in b['rows']], 'paired R7 windows drift')
        ratios.extend([b['seconds']/a['seconds'], sum(r['gpu_seconds'] for r in b['rows'])/
                       sum(r['gpu_seconds'] for r in a['rows'])])
    need(all(math.isfinite(x) and x > 0 for x in ratios), 'invalid measured R7 ratio')
    plain = math.floor(min(arms['plain-1']['seconds'], arms['plain-2']['seconds']))
    need(plain > 0 and arms['r224']['seconds'] > 0, 'invalid measured elapsed seconds')
    scale = calibration['r7_hidden_seconds']/plain
    baseline = r224.BASELINE_HOURS
    flip_hours = (max(1., *ratios)-1)*calibration['r7_hidden_seconds']/3600
    r224_hours = arms['r224']['seconds']*scale/3600
    hours = baseline+flip_hours+r224_hours
    return dict(baseline_hours=baseline, r7_flip_increment_hours=flip_hours,
                r224_increment_hours=r224_hours, projected_hours=hours,
                r7_ratios=ratios, local_plain_seconds=plain, limit_hours=8.5,
                runtime_pass=hours <= 8.5, calibration=calibration,
                method='5.38 + max(0,max(R7 wall and GPU ABBA ratios)-1)*hidden_R7_hours + '
                       'new_R224_N32_seconds*hidden_R7_seconds/floor(min(new_plain_N32_seconds))/3600',
                empirical=empirical_estimate(), qualified_for_submission=False)


def binding(mode, common=None):
    pins = validate_stage()
    need(sha(r224.CHECKPOINT) == r224.CHECKPOINT_SHA, 'R224 checkpoint drift')
    r224.source_parts()
    manifest = io.read_json(r7.MANIFEST)
    from tools import exp169_r7_port as port
    port.verify_bundle(r7.MANIFEST.parent, expected_manifest_sha256=gold.R7_MANIFEST_SHA)
    ids = roster(mode)
    helpers = ('c85r7_r224flip_diagnostics.py', 'c85r7_r224flipq_v1124.py',
               'c85r7_r224_timing.py', 'c85r7_flipall_timing.py', 'c85r7_flipall_measure.py',
               'c85r7_flipall_v1115.py', 'c85r7_flipall_baseline.py', 'hb128_gold58_pbranch_v1.py',
               'hb113_gold58.py', 'exp169_r224.py', 'exp169_r7_port.py', 'exp189_s6_port.py',
               'hb149_monitor_v125.py', 'hb145_output_io.py', 'multi_lens_readout.py',
               'gpu_job.py', 'gpu_queue.py', 'job_runtime.py')
    common_info = read_common(common, ids)[0] if mode == 'gold' and common else None
    return stamp(mode, stage_files=pins, ids=ids, cohort_sha256=digest(ids),
                 series_sha256=sha(r7.SERIES), data=verify_data(mode), metadata=data_contract(mode),
                 common=common_info, calibration=r224.transfer_calibration(),
                 source_sha256=r224.SOURCE_SHA, checkpoint_sha256=r224.CHECKPOINT_SHA,
                 r7_manifest_sha256=sha(r7.MANIFEST), r7_models=manifest['models'],
                 helpers={n: sha(ROOT/'tools'/n) for n in helpers},
                 tests_sha256=sha(ROOT/'tests/test_c85r7_r224flip_diagnostics.py'),
                 rules=dict(runtime_hours=8.5, gold_macro_delta=-.001, gold_weak_four_delta=-.002,
                            public_noninferiority_delta=-.001, parent_alive_min=.957),
                 differences=DIFFERENCES, empirical=empirical_estimate())


def register(mode, common=None):
    value = binding(mode, common)
    return save(mode, 'registration', stamp(mode, state='REGISTERED', binding=value, binding_sha256=digest(value)))


def current(mode):
    value = io.read_json(state_path(mode, 'registration'))
    common = value['binding']['common']
    fresh = binding(mode, common['path'] if common else None)
    need(value['binding'] == fresh and value['binding_sha256'] == digest(fresh), 'registration input drift')
    return value


def prepare(mode):
    """Reuse already staged DICOM bytes; CPU canonical reader/window replay only."""
    registered = current(mode)
    ids = registered['binding']['ids']
    prepare_data(mode)
    rows = series_rows(mode)
    if mode == 'timing':
        work = arm_work(mode, 'r224')
        io.atomic_json(work/'prepare.json', dict(studies=ids, series=rows))
        junction(work/'data', data_source(mode))
    ns = r224.public_namespace()
    torch = ns['torch']
    need(not torch.cuda.is_initialized(), 'CPU prepare must not initialize CUDA')
    vol, mask = ns['build_study'](ids[0], r224.group_series(rows), str(data_source(mode)/'train_series'), ns['_make_reader']())
    r224.validate_volume(vol, mask)
    window = ns['gpu_windows'](vol, mask, 94, 224, 'cpu')
    need(torch.equal(window, ns['eval_windows'](vol, mask, 94, 224)), 'real CPU window parity failed')
    # Exact local instrumentation compiles against both embedded runtime copies.
    compiled = {}
    for label, stage in [('plain', PARENT), ('flip', STAGE)]:
        files = stage_files(stage)
        base = mode_work(mode)/'cpu-runtime'/label
        runtime = base/'embedded'
        for name, raw in files.items():
            pack.write(runtime/name, raw)
        sys.path[:0] = [str(runtime), str(gold.RUNTIME/'deps')]
        port = import_path('hb167_cpu_'+label, runtime/'exp169_r7_port.py')
        # copytree is redirected to overwrite only our fixed generated scratch.
        # The frozen adapter and original embedded files remain unchanged.
        original_copy = port.shutil.copytree
        def scratch_copy(src, dst, *args, **kwargs):
            need(Path(dst).resolve().is_relative_to((mode_work(mode)/'cpu-runtime').resolve()), 'scratch boundary')
            return original_copy(src, dst, *args, **dict(kwargs, dirs_exist_ok=True))
        port.shutil = types.SimpleNamespace(copytree=scratch_copy)
        diagnostic = port.diagnostic_source(runtime/'exp169_r7_src', base)
        raw = instrument((diagnostic/'compact_infer.py').read_text(encoding='utf-8'))
        compile(files['exp169_r7_port.py'], label+'-port', 'exec')
        compile(raw, label+'-compact', 'exec')
        pack.write(base/'instrumented.py', raw.encode())
        compiled[label] = dict(port=pack.digest(files['exp169_r7_port.py']), instrumented=pack.digest(raw.encode()))
    need(not torch.cuda.is_initialized(), 'CPU prepare initialized CUDA')
    return save(mode, 'prepare', stamp(mode, state='CPU_PREPARED', binding_sha256=registered['binding_sha256'],
                studies=len(ids), series=rows, real_cpu_window_shape=list(window.shape),
                exact_window_parity=True, port_sha256=compiled, preflight=preflight(mode),
                limitation='No GPU/model execution; original CPU data path reused. CUDA and Linux remain unqualified.'))


def validate(mode):
    reg = current(mode)
    prep = io.read_json(state_path(mode, 'prepare'))
    need(prep['state'] == 'CPU_PREPARED' and prep['binding_sha256'] == reg['binding_sha256'] and
         prep['exact_window_parity'] is True, 'fresh CPU preparation required')
    checked = preflight(mode)
    need(checked == prep['preflight'], 'CPU preflight drift')
    return save(mode, 'validate', stamp(mode, state='CPU_VALIDATED', binding_sha256=reg['binding_sha256'],
                                       prepare_sha256=sha(state_path(mode, 'prepare')), preflight_sha256=digest(checked)))


def replay_gold(r7_run):
    """CPU plumbing control, never a measured plain/full-flip comparison."""
    import numpy as np
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
         'replay requires present-empty CUDA_VISIBLE_DEVICES and uninitialized CUDA')
    need(r7_run is not None, 'preserved Gold58 R7 run required')
    reg = current('gold')
    valid = io.read_json(state_path('gold', 'validate'))
    need(valid['binding_sha256'] == reg['binding_sha256'], 'validation drift')
    ids, labels = gold.gold()
    info, arrays, failed = read_common(reg['binding']['common']['path'], ids)
    run = Path(r7_run).resolve()
    old_ids, members, old_failed, receipt = gold.load_run(run, 'gold')
    need(old_ids == ids and receipt['studies'] == 58 and receipt['limit'] is None,
         'preserved R7 exact Gold58 cohort required')
    need(not old_failed['R7'].any() and receipt['fallback_counts']['R7'] == 0,
         'identity replay requires preserved R7 without fallbacks')
    result = gold_no_harm(ids, labels, arrays, failed, members['R7'], members['R7'])
    need(result['macro_delta'] == result['weak_four_delta'] == 0 and result['no_harm_pass'],
         'same-input real-cache replay failed')
    need(not torch.cuda.is_initialized(), 'replay initialized CUDA')
    return save('gold', 'cpu-replay', stamp('gold', state='CPU_IDENTITY_CONTROL_PASS',
        binding_sha256=reg['binding_sha256'], common_receipt_sha256=info['sha256'],
        r7_run=str(run), r7_receipt_sha256=sha(run/'receipt.json'),
        r7_predictions_sha256=receipt['prediction_sha256'],
        r7_registration_sha256=receipt['binding_sha256'], studies=len(ids), targets=12,
        common_failure_counts={m:int(np.sum(failed[m])) for m in COMMON},
        macro_delta=result['macro_delta'], weak_four_delta=result['weak_four_delta'],
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_initialized=False,
        measured_pair=False, qualified_for_submission=False,
        limitation='Same preserved plain R7 used twice: input/formula plumbing only. '
                   'New paired plain/full-flip Gold58 GPU outputs and independent result review remain required.'))


def review_check(review, mode, reg):
    need(review.get('verdict') == 'PASS' and review.get('builder') and review.get('reviewer') and
         review['builder'] != review['reviewer'] and review.get('execution_authority') == 'B',
         'independent review PASS and B execution authority required')
    need(review.get('binding_sha256') == reg['binding_sha256'] and
         review.get('validation_sha256') == sha(state_path(mode, 'validate')) and
         review.get('approved_differences') == DIFFERENCES and
         review.get('mode') == mode, 'review binding/smoke/differences mismatch')
    need(mode != 'gold' or (reg['binding']['common'] is not None and
         review.get('common_member_identity_checked') is True), 'Gold58 full-blend common identities not reviewed')


def command(mode):
    return [str(r224.GPU_PYTHON), '-B', str(SCRIPT), 'run', '--mode', mode]


def admit(mode, review, dry_run=False):
    reg = current(mode)
    valid = io.read_json(state_path(mode, 'validate'))
    need(valid['binding_sha256'] == reg['binding_sha256'], 'validation drift')
    need(review is not None, 'independent review is required; dry-run cannot bypass it')
    review = Path(review).resolve()
    review_check(io.read_json(review), mode, reg)
    value = stamp(mode, state='DRY_RUN_PASS' if dry_run else 'ADMITTED',
                  binding_sha256=reg['binding_sha256'], review=str(review), review_sha256=sha(review),
                  validation_sha256=sha(state_path(mode, 'validate')))
    save(mode, 'admit-dry-run' if dry_run else 'admission', value)
    if not dry_run:
        from tools.gpu_queue import validate_plan
        io.atomic_json(mode_work(mode)/'gpu-plan.json', validate_plan([dict(tag=TAGS[mode], command=command(mode))]))
    return value


def admitted(mode):
    value = io.read_json(state_path(mode, 'admission'))
    reg = current(mode)
    need(value['state'] == 'ADMITTED' and value['binding_sha256'] == reg['binding_sha256'] and
         value['validation_sha256'] == sha(state_path(mode, 'validate')) and
         value['review_sha256'] == sha(value['review']), 'admission drift')
    review_check(io.read_json(value['review']), mode, reg)
    return value


def lane(mode):
    import psutil
    from tools.job_runtime import LOGS, job_status
    parents = psutil.Process().parents()
    need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in parents), 'gpu_job lock required')
    job = job_status(LOGS, TAGS[mode], expected_command=command(mode), tail=False)
    need(job['state'] == 'running' and job['pid'] in {p.pid for p in parents}, 'exclusive command-bound HB167 lane required')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=5)
    need(not r7.gpu_python_pids(result.stdout, {os.getpid(), *[p.pid for p in parents]}), 'another Python ML process owns GPU')
    desktop = r7.desktop_snapshot()
    r7.live_memory_guard(desktop['used_mib'], desktop)
    need(desktop['uuid'] == current(mode)['binding']['calibration']['device_uuid'], 'must use same 4070 device as calibration')
    return desktop


def cleanup_owned(child, tracked, job, psutil):
    """Keep the frozen termination, then allow delayed owned-process teardown.

    No directory deletion or extra signalling. Only a cached, identity-aware
    Process object confirmed gone can clear a 'still alive' diagnostic.
    """
    errors = _CLEANUP(child, tracked, job, psutil)
    pending = {}
    for error in errors:
        parts = error.split()
        if len(parts) == 3 and parts[:2] == ['still', 'alive'] and parts[2].isdigit():
            process = tracked.get(int(parts[2]))
            if process is not None:
                pending[error] = process
    recovered = []
    started = time.monotonic()
    while pending:
        for error, process in list(pending.items()):
            try:
                gone = not process.is_running()  # psutil checks PID + creation time
            except psutil.NoSuchProcess:
                gone = True
            except psutil.AccessDenied:
                gone = False
            if gone:
                recovered.append(error)
                del pending[error]
        if not pending or time.monotonic()-started >= 10.:
            break
        time.sleep(.1)
    remaining = [error for error in errors if error not in recovered]
    if errors:
        io.journal(r224.WORK/'cleanup-recovery.json', dict(version=VERSION,
            original_errors=errors, recovered=recovered, remaining=remaining,
            extra_wait_seconds=time.monotonic()-started,
            policy='No file deletion or new signals; cached process identity confirmed gone'), r224.WORK)
    return remaining


def configure(mode, arm):
    """Private process-local diagnostic configuration; no frozen file edits."""
    work = arm_work(mode, arm)
    r224.WORK, r224.TAG, r224.__file__ = work, TAGS[mode], str(SCRIPT)
    r224.admitted = lambda: admitted(mode)
    r224.lane_admission = lambda: lane(mode)
    # Scope the recovery to this diagnostics process; the shared monitor stays frozen.
    r224.monitor = types.SimpleNamespace(**dict(vars(_MONITOR), _cleanup=cleanup_owned))
    return work


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(mode, arm):
    need(os.environ.get('HB164_CONTROLLER_PID') == str(os.getppid()), 'supervised worker only')
    admitted(mode)
    lane(mode)
    work = configure(mode, arm)
    prep = io.read_json(state_path(mode, 'prepare'))
    ids = roster(mode)
    if arm == 'r224':
        # The inherited worker reads only this sealed descriptor and existing
        # prepared data path; it is otherwise byte-identical original code.
        need(io.read_json(work/'prepare.json') == dict(studies=ids, series=prep['series']), 'R224 descriptor drift')
        need((work/'data').exists(), 'reviewed data junction missing')
        return r224.worker()
    files = stage_files(PARENT if arm.startswith('plain') else STAGE)
    runtime = work/'runtime'
    for name, raw in files.items():
        pack.write(runtime/name, raw)
    deps = gold.RUNTIME/'deps'
    sys.path[:0] = [str(runtime), str(deps)]
    os.environ['PYTHONPATH'] = os.pathsep.join([str(runtime), str(deps), str(ROOT)])
    os.environ['HB153_MEASURE_STATUS'] = str(work/'worker-status.json')
    port = import_path('exp169_r7_port', runtime/'exp169_r7_port.py')
    frozen = port.diagnostic_source
    def diagnostic(*args, **kwargs):
        path = frozen(*args, **kwargs)
        target = path/'compact_infer.py'
        raw = instrument(target.read_text(encoding='utf-8')).encode()
        io.retry(target.write_bytes, raw)
        return path
    port.diagnostic_source = diagnostic
    started = time.perf_counter()
    predictions = port.infer_studies([data_source(mode)/'train_series'/uid for uid in ids],
        bundle_dir=r7.MANIFEST.parent, code_dir=CODE,
        anatomy_weights=ANATOMY,
        output_dir=work/'output', source_dir=runtime/'exp169_r7_src', workers=2,
        amp='auto', device='cuda', max_fallback_fraction=0.)
    value = io.read_json(work/'worker-status.json')
    need([p[gold.UID] for p in predictions] == ids, 'R7 predictions order mismatch')
    r7.runtime_check(value, arm.startswith('flip'))
    result = stamp(mode, state='COMPLETE', done=len(ids), rows=value['rows'], runtime=value['runtime'],
                   predictions=[[p[t] for t in gold.LABELS] for p in predictions],
                   seconds=time.perf_counter()-started, device=value['device'],
                   binding_sha256=io.read_json(state_path(mode, 'admission'))['binding_sha256'])
    io.journal(work/'worker-status.json', result, work)
    return 0


def canonicalize_rows(result, ids):
    """Normalize measurement rows only; predictions already have cohort order."""
    rows = result['rows']
    completion_order = [row['study'] for row in rows]
    need(len(ids) == len(set(ids)) and len(completion_order) == len(set(completion_order)) and
         set(completion_order) == set(ids), 'unique exact cohort required for row canonicalization')
    by_study = {row['study']: row for row in rows}
    return dict(result, rows=[by_study[uid] for uid in ids], completion_order=completion_order)


def check_arm(mode, arm, result, desktop, telemetry):
    import numpy as np
    ids = roster(mode)
    need(result.get('state') == 'COMPLETE' and result.get('done') == len(ids) and
         [r['study'] for r in result['rows']] == ids, 'complete ordered cohort required')
    need(telemetry['complete'], 'incomplete telemetry cannot publish timing or no-harm PASS')
    r7.memory_admission(desktop, [r['peak_reserved_bytes'] for r in result['rows']], telemetry['peak_gpu_mib']*2**20)
    if arm == 'r224':
        r224.validate_measurement(result, ids)
    else:
        r7.runtime_check(result, arm.startswith('flip'))
        predictions = np.asarray(result['predictions'], float)
        need(predictions.shape == (len(ids),12) and np.isfinite(predictions).all() and
             ((predictions >= 0) & (predictions <= 1)).all(), 'invalid R7 predictions')
        for row in result['rows']:
            measure.window_gate(row['windows'], row['batch_windows'])
            need(math.isfinite(row['gpu_seconds']) and row['gpu_seconds'] > 0, 'invalid R7 timing')


def gold_no_harm(ids, labels, common, failed, plain, flip):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    need(len(ids) == len(set(ids)) == 58 and labels.shape == (58,12) and
         np.isin(labels, [0,1]).all() and all(len(np.unique(labels[:,j])) == 2 for j in range(12)),
         'exact binary nondegenerate Gold58 required')
    need(set(common) == set(failed) == set(COMMON), 'full common members required')
    need(all(np.asarray(failed[m]).shape == (58,) and np.asarray(failed[m]).dtype == np.bool_ for m in COMMON),
         'boolean member failure masks required')
    masks = {m: {u: 'authenticated parent failure' for u, f in zip(ids, failed[m]) if f} for m in COMMON}
    outputs = {}
    for name, member in [('plain', plain), ('flip', flip)]:
        values = formula.blend(ids, common['R'], common['R224'], member,
                               common['EXP143v2'], common['EXP155'], common['Q_A5q'], masks)
        outputs[name] = [float(roc_auc_score(labels[:,j], values[:,j])) for j in range(12)]
    delta = np.asarray(outputs['flip'])-np.asarray(outputs['plain'])
    macro, weak = float(delta.mean()), float(delta[list(gold.WEAK)].mean())
    return dict(scope='full final R224 candidate vs plain R224 parent; fixed Gold58 cohort',
                formula=formula.FORMULA, per_target_auc=outputs, macro_delta=macro, weak_four_delta=weak,
                macro_floor=-.001, weak_four_floor=-.002,
                no_harm_pass=macro >= -.001 and weak >= -.002,
                exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
                caveat='Small exposed diagnostic; requires independent result review; not clean OOF.')


def run(mode):
    admission = admitted(mode)
    need(not state_path(mode, 'attempt').exists(), 'preserve previous attempt; B must archive deliberately')
    checked = preflight(mode)
    need(digest(checked) == io.read_json(state_path(mode, 'validate'))['preflight_sha256'], 'launch preflight drift')
    arms = ORDER if mode == 'timing' else ('plain-1', 'flip-1')
    for arm in arms:
        need(not (arm_work(mode, arm)/'worker-status.json').exists(), 'preserve old arm output')
    attempt = stamp(mode, state='RUNNING', binding_sha256=admission['binding_sha256'], arms={}, order=list(arms))
    save(mode, 'attempt', attempt)
    try:
        for arm in arms:
            desktop = lane(mode)
            work = configure(mode, arm)
            need(not (work/'worker-status.json').exists(), 'preserve old arm output')
            os.environ['HB167_MODE'], os.environ['HB167_ARM'] = mode, arm
            started = time.perf_counter()
            value, telemetry = r224.supervise(desktop)
            elapsed = time.perf_counter()-started
            value = canonicalize_rows(value, roster(mode))
            check_arm(mode, arm, value, desktop, telemetry)
            need(value['binding_sha256'] == admission['binding_sha256'], 'arm admission mismatch')
            # Match flipall_timing's supervisor wall for R7 and r224_timing's
            # worker timer for R224. Post-run validation is outside both clocks.
            if arm != 'r224':
                value['worker_seconds'], value['seconds'] = value['seconds'], elapsed
            value.update(controller_seconds=elapsed, desktop_before_cuda=desktop,
                         telemetry=telemetry)
            attempt['arms'][arm] = value
            save(mode, 'attempt', attempt)
        if mode == 'timing':
            result = projection(attempt['arms'], current(mode)['binding']['calibration'])
        else:
            ids, labels = gold.gold()
            common = current(mode)['binding']['common']
            _, arrays, failed = read_common(common['path'], ids)
            result = gold_no_harm(ids, labels, arrays, failed,
                                 attempt['arms']['plain-1']['predictions'], attempt['arms']['flip-1']['predictions'])
        attempt['state'] = 'COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW'
        save(mode, 'attempt', attempt)
        return save(mode, 'result', stamp(mode, state=attempt['state'], binding_sha256=admission['binding_sha256'],
                                        result=result, attempt_sha256=sha(state_path(mode, 'attempt'))))
    except BaseException as exc:
        attempt.update(state='HOLD', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(mode, 'attempt', attempt)


def preserved_context(source, pins, pins_sha256):
    """Verify sealed local evidence and the original reviewed admission, read-only."""
    source, pins = Path(source).resolve(), Path(pins).resolve()
    need(source.is_relative_to(WORK.resolve()) and source != mode_work('gold').resolve(),
         'preserved source must be a separate historical harness folder')
    need(pins.is_relative_to(ROOT) and sha(pins) == pins_sha256, 'preserved evidence seal mismatch')
    seal = io.read_json(pins)
    need(seal['schema'] == 'HB167-preserved-evidence/v1' and Path(seal['source']).resolve() == source,
         'preserved source seal mismatch')
    for name, expected in seal['files'].items():
        path = (ROOT/name).resolve()
        need(path.is_relative_to(ROOT) and sha(path) == expected, 'preserved input hash drift: '+name)
    def read(path):
        path = Path(path).resolve()
        need(str(path.relative_to(ROOT)) in seal['files'], 'unsealed historical input')
        return io.read_json(path)
    reg, admission, valid = [read(source/(name+'.json')) for name in ('registration','admission','validate')]
    bound = reg['binding']
    need(reg['state'] == 'REGISTERED' and bound['identity'] == TAGS['gold'] and
         bound['version'] == '1.0.5' and reg['binding_sha256'] == digest(bound), 'historical registration drift')
    need(admission['state'] == 'ADMITTED' and admission['binding_sha256'] ==
         valid['binding_sha256'] == reg['binding_sha256'] and valid['state'] == 'CPU_VALIDATED' and
         admission['validation_sha256'] == sha(source/'validate.json') and
         valid['prepare_sha256'] == sha(source/'prepare.json'), 'historical admission/validation drift')
    review = read(admission['review'])
    need(sha(admission['review']) == admission['review_sha256'] and review['verdict'] == 'PASS' and
         review['builder'] != review['reviewer'] and review['execution_authority'] == 'B' and
         review['mode'] == 'gold' and review['binding_sha256'] == reg['binding_sha256'] and
         review['validation_sha256'] == admission['validation_sha256'] and
         review['approved_differences'] == bound['differences'] == DIFFERENCES and
         review['common_member_identity_checked'] is True, 'historical independent review drift')
    for name, expected in bound['helpers'].items():
        path = HISTORICAL_BACKUP/'tools'/name if name == SCRIPT.name else ROOT/'tools'/name
        need(sha(path) == expected, 'historical helper drift: '+name)
    need(sha(HISTORICAL_BACKUP/'tests/test_c85r7_r224flip_diagnostics.py') == bound['tests_sha256'],
         'historical tests drift')
    need(validate_stage() == bound['stage_files'], 'historical stage drift')
    need(bound['rules']['gold_macro_delta'] == -.001 and bound['rules']['gold_weak_four_delta'] == -.002,
         'registered no-harm thresholds drift')
    ids = roster('gold')
    need(bound['ids'] == ids and bound['cohort_sha256'] == digest(ids), 'historical cohort drift')
    need(sha(r7.SERIES) == bound['series_sha256'] and sha(r7.MANIFEST) == bound['r7_manifest_sha256'],
         'historical series/manifest drift')
    common, arrays, failed = read_common(bound['common']['path'], ids)
    need(common == bound['common'], 'historical common cache drift')
    attempt = read(source/'attempt.json')
    need(attempt['state'] == 'HOLD' and attempt['reason'] == 'ValueError: owned cleanup incomplete' and
         attempt['binding_sha256'] == reg['binding_sha256'] and
         attempt['order'] == ['plain-1','flip-1'], 'only completed-arm cleanup HOLD is recoverable')
    from tools.job_runtime import command_digest
    job = read(ROOT/'artifacts/logs/HB167_GOLD58/receipt.json')
    need(job['tag'] == TAGS['gold'] and job['state'] == 'FAILED' and job['exit_code'] == 2 and
         job['command_sha256'] == command_digest(command('gold'), ROOT), 'historical job identity mismatch')
    return reg, arrays, failed, read


def resolved_cleanup(errors):
    """Historical receipt lacks create_time: only PID absence is accepted, no kills."""
    import psutil
    warnings = []
    for error in errors:
        parts = error.split()
        need(len(parts) == 3 and parts[:2] == ['still','alive'] and parts[2].isdigit(),
             'unrecoverable cleanup error')
        need(not psutil.pid_exists(int(parts[2])), 'historical cleanup PID still exists; cannot prove identity')
        warnings.append(dict(original=error, resolution='PID now absent; no process signalled or files deleted',
                             cause='termination did not settle within original waits; exact OS cause unrecorded'))
    return warnings


def preserved_arm(source, arm, reg, read):
    import numpy as np
    from tools import exp189_s6_port as s6
    work = Path(source)/arm
    raw = read(work/'worker-status.json')
    supervision = read(work/'supervision.json')
    need(supervision['exit_code'] == 0 and supervision['error'] is None and
         supervision['telemetry']['memory_failure'] is None, 'arm supervision failed')
    warnings = resolved_cleanup(supervision['cleanup'])
    need(raw['identity'] == TAGS['gold'] and raw['version'] == '1.0.5' and
         raw['binding_sha256'] == reg['binding_sha256'], 'preserved arm binding drift')
    ids = reg['binding']['ids']
    value = canonicalize_rows(raw, ids)
    check_arm('gold', arm, value, supervision['desktop_before_cuda'], supervision['telemetry'])
    need(supervision['desktop_before_cuda']['uuid'] == reg['binding']['calibration']['device_uuid'],
         'preserved GPU identity drift')
    controller = read(work/'output/controller/submission_receipt.json')
    need(controller['status'] == 'PASS' and controller['rows'] == controller['scored'] == 58 and
         controller['fallback_studies'] == 0 and controller['fallback_reasons'] == {} and
         controller['worker_exit_codes'] and all(v == 0 for v in controller['worker_exit_codes'].values()),
         'incomplete or fallback controller output')
    need(math.isfinite(raw['seconds']) and raw['seconds'] > 0, 'invalid worker time')
    for path in (work/'output/predictions.csv', work/'output/controller/submission.csv'):
        # read_predictions validates unique exact IDs and target inventory, then orders by IDs.
        rows = s6.read_predictions(path, ids)
        predictions = np.asarray([[row[t] for t in gold.LABELS] for row in rows], float)
        need(np.array_equal(predictions, np.asarray(raw['predictions'], float)), 'preserved predictions disagree')
    files = stage_files(PARENT if arm == 'plain-1' else STAGE)
    for name, expected in files.items():
        need((work/'runtime'/name).read_bytes() == expected, 'preserved embedded source drift: '+name)
    runtime = work/'output/r7_runtime'
    label = 'plain' if arm == 'plain-1' else 'flip'
    prepared = Path(source)/'cpu-runtime'/label/'instrumented.py'
    prep = read(Path(source)/'prepare.json')
    need(sha(prepared) == prep['port_sha256'][label]['instrumented'], 'reviewed instrumentation pin drift')
    expected = prepared.read_text(encoding='utf-8')
    need((runtime/'compact_infer.py').read_text(encoding='utf-8') == expected,
         'preserved instrumentation drift')
    invocation = read(work/'output/invocation.json')
    args = invocation['args']
    worker_args = json.loads(args[args.index('--worker-args')+1])
    need(('--flip-tta' in worker_args) == arm.startswith('flip') and
         args[args.index('--max-fallback-fraction')+1] == '0.0', 'preserved invocation variant drift')
    provenance = read(work/'output/provenance.json')
    need(provenance['manifest_sha256'] == reg['binding']['r7_manifest_sha256'], 'arm model manifest drift')
    for name, expected_sha in provenance['file_sha256'].items():
        need(sha(r7.MANIFEST.parent/name) == expected_sha, 'arm checkpoint/asset drift')
    return value, warnings, dict(controller_seconds=controller['seconds'], worker_seconds=raw['seconds'],
                                 supervisor_seconds=supervision['seconds'])


def evaluate(source, pins, pins_sha256):
    """Evaluate a preserved reviewed pair; never admit, launch, or rewrite its receipts."""
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
         'evaluate requires present-empty CUDA_VISIBLE_DEVICES and uninitialized CUDA')
    need(source is not None and pins is not None and pins_sha256, 'source and pinned evidence required')
    reg, arrays, failed, read = preserved_context(source, pins, pins_sha256)
    arms, warnings, times = {}, {}, {}
    for arm in ('plain-1','flip-1'):
        arms[arm], warnings[arm], times[arm] = preserved_arm(source, arm, reg, read)
    need([(r['study'],r['windows'],r['batch_windows']) for r in arms['plain-1']['rows']] ==
         [(r['study'],r['windows'],r['batch_windows']) for r in arms['flip-1']['rows']], 'paired window drift')
    ids, labels = gold.gold()
    result = gold_no_harm(ids, labels, arrays, failed, arms['plain-1']['predictions'], arms['flip-1']['predictions'])
    # Detect concurrent edits during evaluation, before publishing the new receipt.
    seal = io.read_json(pins)
    need(sha(pins) == pins_sha256 and all(sha(ROOT/p) == h for p,h in seal['files'].items()),
         'preserved evidence changed during evaluation')
    need(not torch.cuda.is_initialized(), 'evaluate initialized CUDA')
    return save('gold', 'evaluation', stamp('gold', state='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW',
        historical_binding_sha256=reg['binding_sha256'], evidence_sha256=pins_sha256,
        source=str(Path(source).resolve()), evaluator_sha256=sha(SCRIPT), result=result,
        cleanup_warnings=warnings, timing=times, measured_pair=True,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_initialized=False,
        provenance_limit='Original files cross-checked against reviewed admission and duplicate outputs; '
                         'evidence snapshot sealed at recovery, not a cryptographic run-time signature.'))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith('candidates-'):
        from tools import c85r7_gold58_candidates as candidates
        return candidates.main([argv[0].removeprefix('candidates-'), *argv[1:]])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'prepare', 'validate', 'replay', 'evaluate', 'admit', 'run', '_worker', 'empirical'])
    parser.add_argument('--mode', choices=['timing','gold'], default='timing')
    parser.add_argument('--common', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--r7-run', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--pins', type=Path)
    parser.add_argument('--pins-sha256')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        need(not args.dry_run or args.action == 'admit', '--dry-run is supported only for admission')
        if args.action == '_worker':
            return worker(os.environ['HB167_MODE'], os.environ['HB167_ARM'])
        if args.action == 'empirical':
            print(json.dumps(empirical_estimate())); return 0
        if args.action == 'register': value = register(args.mode, args.common)
        elif args.action == 'prepare': value = prepare(args.mode)
        elif args.action == 'validate': value = validate(args.mode)
        elif args.action == 'replay':
            need(args.mode == 'gold', 'replay supports gold mode only')
            value = replay_gold(args.r7_run)
        elif args.action == 'evaluate':
            need(args.mode == 'gold', 'evaluate supports gold mode only')
            value = evaluate(args.source, args.pins, args.pins_sha256)
        elif args.action == 'admit': value = admit(args.mode, args.review, args.dry_run)
        else: value = run(args.mode)
        print(json.dumps({k:value[k] for k in ('identity','version','state')}))
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(json.dumps(stamp(args.mode, state='HOLD', action=args.action, reason=str(exc))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
