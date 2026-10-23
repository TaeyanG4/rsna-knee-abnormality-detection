"""Q12b public-R cohort readout v1.0.1, EXPOSED, CPU build / B-only GPU launch.

Changelog 1.0.1: explicit CPU validation guard, undefined-draw regression tests.
Prelaunch v1.0.0 backup: .local/backups/q12b-v100. No GPU attempt existed.
Changelog 1.0.0: exact HB167 R384 mirror + R224 plain; frozen fable cohort/rulers,
paired study bootstrap, receipt-bound admission, archive-only read adapter.
No release sources are modified. No labels or report text are printed.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import copy
import csv
import gc
import hashlib
import io as streams
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hb145_output_io as io
from tools import hb167_common_gold58_v1 as common

VERSION = '1.0.1'
TAG = 'Q12B_R_COHORT976_V1'
AFTER = 'L1_RCO_K0_V2'
BUILDER = 'Codex/q12b_build'
WORK = ROOT / '.local/scratch/slot-b/q12b'
SCRIPT = Path(__file__).resolve()
RUNTIME = WORK / 'runtime'
PYTHON = Path(r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe')
FABLE = ROOT / '.local/scratch/slot-b/routing-1016/fable/cohort977_compact_vs_own.py'
FABLE_SHA = '3a15f5c6de6fc80d3c2fb4fe2233e71877e2f00000eea6b7b5bc849afdc7bbb1'
RULER_SHA = '8c4658ca186e77d25c1da3e2d1704c77b596678726a10f02bf80fff03cc3efe2'
HB167 = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq'
ZIP = ROOT / 'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip'
SERIES = ROOT / 'storage/competition/train_series.csv'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
OTHER5 = [7, 9, 10, 8, 11]
MEMBERS = ('R384', 'R224')
READING_RULE = ('Only AUC(held-out R7-f0) >= AUC(in-sample R) is directionally informative; '
                'R > R7-f0 is uninterpretable. R was trained on all RSNA train and OAI. '
                'Descriptive and EXPOSED; report rulers are not image truth. Equality is not '
                'equivalence; a CI spanning zero does not establish superiority. No route GO decision.')
RULES = dict(reading_rule=READING_RULE, draws=1000, seed=20261016, primary=[5, 6], other5=OTHER5,
             ranks='average percentile on all 976 studies once, before bootstrap',
             bootstrap='paired study indices, same draw across all rulers, targets and members; '
                       'single-class cells omitted only from that cell; aggregate draws require all constituent cells',
             formula='R = 0.65 rank(R384) + 0.35 rank(R224)',
             decisions='No selection or GO; independent result review required before scientific use')
need, sha, digest = common.need, common.sha, common.digest


def stamp(**kw):
    return dict(version=VERSION, identity=TAG, exposure='EXPOSED', clean_oof=False, **kw)


def save(name, value):
    io.atomic_json(WORK / (name + '.json'), value)
    return value


def cpu_only():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'CUDA_VISIBLE_DEVICES' in os.environ,
         'present-empty CUDA_VISIBLE_DEVICES required')
    import torch
    need(not torch.cuda.is_initialized(), 'CPU command initialized CUDA')


def fable_context():
    """Execute the original cohort statements byte-for-byte, stopping before print/AUC."""
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    from tools import exp156_k_sweep as e156
    need(sha(FABLE) == FABLE_SHA and sha(Path(e156.__file__)) == RULER_SHA, 'fable/ruler source drift')
    source = FABLE.read_text(encoding='utf-8')
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'inter' for t in n.targets))
    nodes = copy.deepcopy(main.body[:end + 1])
    # The selected source bytes, including the original nested ids_of, are sealed.
    fragment = '\n'.join(source.splitlines()[main.body[0].lineno - 1:main.body[end].end_lineno])
    code = ast.Module(body=nodes, type_ignores=[])
    ns = dict(os=os, json=json, np=np, pd=pd, rankdata=rankdata, e156=e156,
              ROOT=str(ROOT), LAB=LABELS)
    exec(compile(ast.fix_missing_locations(code), str(FABLE), 'exec'), ns)
    need(ns['t']['model_fold'].astype(str).eq('0').all(), 'teacher must use held-out model fold zero')
    need(not ns['t']['StudyInstanceUID'].duplicated().any(), 'duplicate teacher IDs')
    need(len(ns['roster']) == len(set(ns['roster'])) == 977, 'duplicate nominal roster IDs')
    need(all(not frame.index.has_duplicates for frame in ns['cs']), 'duplicate control IDs')
    ids = ns['cohort']
    need(len(ids) == 976 and len(set(ids)) == 976 and ids == sorted(ids), 'exact fable cohort required')
    compact = ns['teach'].loc[ids].to_numpy()
    need(compact.shape == (976, 12) and np.isfinite(compact).all(), 'compact prediction coverage')
    return ids, compact, e156.references(ids), hashlib.sha256(fragment.encode()).hexdigest()


def stage_sources():
    from tools import c85r7_r224flip_diagnostics as d
    files = d.stage_files(HB167)
    reg = io.read_json(common.STAGE / 'registration.json')
    for member, old, name in [('R384', 'R', 'public-serving.py'), ('R224', 'R224', 'raptor224_run.py')]:
        proof = io.read_json(common.WORK / ('gpu/' + old + '-receipt.json'))
        need(proof['state'] == 'COMPLETE' and proof['registration_sha256'] == sha(common.STAGE / 'registration.json'),
             'HB167 historical identity receipt')
        need(hashlib.sha256(files[name]).hexdigest() == proof['serving_source_sha256'] == reg['embedded'][name],
             member + ' differs from admitted H-B-167')
        checkpoint = reg['checkpoints'][old]
        need(checkpoint['path'] == proof['checkpoint_manifest']['path'] and
             checkpoint['sha256'] == proof['checkpoint_sha256'] == sha(ROOT/checkpoint['path']), 'checkpoint identity/drift')
    for name in ('public-serving.py', 'raptor224_run.py', 'exp152_own0949_release.py', 'exp169_r224.py'):
        need(hashlib.sha256(files[name]).hexdigest() == reg['embedded'][name], 'embedded dependency drift')
    checkpoints = {m: reg['checkpoints']['R' if m == 'R384' else m] for m in MEMBERS}
    return files, checkpoints


def archive_manifest(ids):
    wanted = set(ids)
    with zipfile.ZipFile(ZIP) as archive:
        entries = [[i.filename, i.CRC, i.file_size, i.compress_size, i.header_offset]
                   for i in archive.infolist() if len(i.filename.split('/')) == 4
                   and i.filename.split('/')[0] == 'train_series'
                   and i.filename.split('/')[1] in wanted and i.filename.endswith('.dcm')]
    need({e[0].split('/')[1] for e in entries} == wanted and len({e[0] for e in entries}) == len(entries),
         'DICOM archive missing/duplicate membership')
    return dict(path=str(ZIP), size=ZIP.stat().st_size, mtime_ns=ZIP.stat().st_mtime_ns,
                members=len(entries), uncompressed_bytes=sum(e[2] for e in entries),
                central_directory_sha256=digest(entries),
                binding='SHA256 of requested ZIP member names/CRC32/sizes/offsets; CRC checked on every decoded read; not full ZIP SHA256')


def launch_command():
    return [str(PYTHON), '-B', str(SCRIPT), 'run']


def plan():
    return [dict(tag=TAG, command=launch_command(), after=AFTER, continue_on_failure=True)]


def register():
    cpu_only()
    ids, _, _, fragment = fable_context()
    files, checkpoints = stage_sources()
    from tools import exp156_k_sweep as e156
    from tools import gpu_queue, gpu_job, job_runtime, hb149_monitor_v125, c85r7_r224_timing, c85r7_flipall_timing, hb113_gold58, hb113_gold58_archive
    base = ROOT / '.local/scratch/astra-b4/H-B-146/launch-v115'
    inputs = [SCRIPT, ROOT/'tests/test_q12b_r_cohort_v1.py', WORK/'SPEC.md', FABLE,
              HB167/'kernel/release.ipynb', common.STAGE/'registration.json', SERIES,
              base/'registration.json', base/'teacher-f0.csv', base/'fits/control-s42/val.csv', base/'fits/control-s43/val.csv',
              ROOT/'artifacts/experiments/EXP181_COATNET_F0_BRIDGE/CONTROL/oof_k94_mirror_tta.csv',
              *[ROOT/p for p in (e156.JEV, e156.AQUILA, e156.DREAD, *e156.CONSENSUS)],
              *[ROOT/c['path'] for c in checkpoints.values()]]
    inputs += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
               and Path(m.__file__).resolve().is_relative_to(ROOT/'tools') and Path(m.__file__).suffix == '.py']
    with SERIES.open(newline='', encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if r['StudyInstanceUID'] in set(ids)]
    need({r['StudyInstanceUID'] for r in rows} == set(ids), 'series metadata coverage')
    value = stamp(state='REGISTERED', builder=BUILDER, rules=RULES, ids=ids, targets=LABELS, rows=rows,
                  cohort_source_fragment_sha256=fragment, archive=archive_manifest(ids), checkpoints=checkpoints,
                  embedded={n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
                  pins={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(inputs))}, plan=plan(),
                  recipe=dict(R384='94 windows, native320, plain + anatomical mirror mean, checkpoint res384',
                              R224='94 windows, 224, PLAIN (HB167 flips R7 only)'))
    path = WORK/'registration.json'
    if path.exists(): need(io.read_json(path) == value, 'registration immutable; stale build must be explicitly revised')
    save('registration', value)
    save('gpu-queue', gpu_queue.validate_plan(plan()))
    return stamp(state='REGISTERED', studies=976, reading_rule=READING_RULE)


def validate(runtime=True):
    reg = io.read_json(WORK/'registration.json')
    need(reg['version'] == VERSION and reg['builder'] == BUILDER and reg['rules'] == RULES and reg['plan'] == plan(), 'policy drift')
    for path, expected in reg['pins'].items(): need(sha(ROOT/path) == expected, 'registered input drift: '+path)
    need(io.read_json(WORK/'gpu-queue.json') == plan(), 'queue drift')
    need(archive_manifest(reg['ids']) == reg['archive'], 'DICOM archive drift')
    ids, _, _, fragment = fable_context()
    need(ids == reg['ids'] and fragment == reg['cohort_source_fragment_sha256'], 'cohort drift')
    if runtime:
        for name, expected in reg['embedded'].items(): need(sha(RUNTIME/name) == expected, 'extracted source drift')
    return reg


def prepare():
    cpu_only(); reg = validate(False)
    files, _ = stage_sources()
    for name, data in files.items():
        path = RUNTIME/name; path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        io.retry(temporary.write_bytes, data); io.replace(temporary, path)
    # Exactly two real studies, ordinary filesystem oracle for archive IO parity.
    chosen = set(reg['ids'][:2]); count = 0
    with zipfile.ZipFile(ZIP) as archive:
        for item in archive.infolist():
            parts = item.filename.split('/')
            if len(parts) != 4 or parts[0] != 'train_series' or parts[1] not in chosen or not parts[-1].endswith('.dcm'): continue
            need(all(p and p not in ('.', '..') and ':' not in p for p in parts), 'unsafe archive path')
            path = WORK/'data'/item.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read(item)
            if path.exists(): need(path.read_bytes() == raw, 'existing owned parity input differs')
            else: io.retry(path.write_bytes, raw)
            count += 1
    return save('prepare', stamp(state='CPU_PREPARED', registration_sha256=sha(WORK/'registration.json'),
                                 extracted_studies=2, extracted_dicom_files=count, cuda_initialized=False))


class CanonicalZip:
    """Read-only file-opening adapter; the admitted R224 reader functions stay intact."""
    def __init__(self, ns, ids):
        import fnmatch
        import pydicom
        self.ns, self.oldglob, self.dcmread, self.pydicom = ns, ns['glob'], pydicom.dcmread, pydicom
        self.archive = zipfile.ZipFile(ZIP); self.lock = threading.Lock(); self.errors = 0
        self.members = {'archive/'+i.filename: i for i in self.archive.infolist()
                        if len(i.filename.split('/')) == 4 and i.filename.split('/')[0] == 'train_series'
                        and i.filename.split('/')[1] in set(ids) and i.filename.endswith('.dcm')}
        self.index = series_index(self.members)
        ns['glob'] = types.SimpleNamespace(glob=lambda pattern: indexed_glob(self.index, pattern))
        def read(path, *args, **kwargs):
            if isinstance(path, str) and path.startswith('archive/'):
                try:
                    with self.lock: raw = self.archive.read(self.members[path])
                    path = streams.BytesIO(raw)
                except Exception:
                    self.errors += 1; raise
            return self.dcmread(path, *args, **kwargs)
        pydicom.dcmread = read

    def close(self):
        self.pydicom.dcmread = self.dcmread; self.ns['glob'] = self.oldglob; self.archive.close()


def series_index(members):
    result = {}
    for path in members: result.setdefault(path.rsplit('/', 1)[0], []).append(path)
    return result


def indexed_glob(index, pattern):
    import fnmatch
    directory = pattern.rsplit('/', 1)[0]
    need(not any(c in directory for c in '*?['), 'series glob must name a literal directory')
    return [p for p in index.get(directory, []) if fnmatch.fnmatchcase(p, pattern)]


def namespace(member, cpu=False):
    from tools import hb113_gold58 as r
    if member == 'R224':
        fn = types.FunctionType(common.r224_namespace.__code__, dict(common.r224_namespace.__globals__, RUNTIME=RUNTIME))
        return fn()
    module = common.load('q12b_public_serving', RUNTIME/'public-serving.py')
    helper = common.load('q12b_public_helper', RUNTIME/'exp152_own0949_release.py')
    module._E152 = helper
    if cpu: module.infer_probs = r.cpu_probs
    return module.__dict__


def model_for(member, ns, reg, device):
    ck = reg['checkpoints'][member]
    if member == 'R224': return ns['load_model'](str(ROOT/ck['path']), ns['ARMS'][0]['arch'], 224, device)[0]
    ns.update(_E152_MANIFEST={'checkpoints': {Path(ck['path']).name: {'sha256': ck['sha256']}}},
              _E152_FILES={Path(ck['path']).name: ROOT/ck['path']})
    return ns['_E152'].checked_model(ns, ROOT/ck['path'], common.ARCH, 384, device)[0]


@contextlib.contextmanager
def decoding(member, ns, ids, archive):
    from tools import hb113_gold58 as r
    adapter = None
    try:
        if member == 'R224':
            if archive: adapter = CanonicalZip(ns, ids)
            reader = ns['_make_reader']()
            fn = lambda uid, grouped, root: ns['build_study'](uid, grouped, root, reader)
        else:
            decoder = r.fastread(types.SimpleNamespace(**ns))
            if archive:
                adapter = r.ArchiveReader(decoder, ZIP, set(ids))
                index = series_index(adapter.members)
                decoder.glob = types.SimpleNamespace(glob=lambda pattern: indexed_glob(index, pattern))
            fn = decoder.make_builder(ns['FASTREAD_VARIANT'], ns['IMG'], ns['SLOTS'], hdr_threads=ns['FR_HDR'], px_threads=ns['FR_PX'])
        yield fn, 'archive/train_series' if archive else str(WORK/'data/train_series'), adapter
    finally:
        if adapter: adapter.close()


def infer_study(ns, model, member, vol, mask, device):
    plain = ns['gpu_windows'](vol, mask, 94, 224 if member == 'R224' else 320, device)
    if member == 'R224': return ns['infer_probs'](model, plain, device)
    mirror = ns['gpu_windows'](vol, mask, 94, 320, device, mirror=True)
    return ns['infer_pair_probs'](model, plain, mirror, device)


def smoke():
    cpu_only(); reg = validate()
    need(io.read_json(WORK/'prepare.json')['registration_sha256'] == sha(WORK/'registration.json'), 'prepare stale')
    import numpy as np
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    ids = reg['ids'][:2]; grouped = {u: [r for r in reg['rows'] if r['StudyInstanceUID'] == u] for u in ids}
    results = []; start = time.monotonic()
    for member in MEMBERS:
        ns = namespace(member, True); model = model_for(member, ns, reg, 'cpu')
        need(all(p.device.type == 'cpu' for p in model.parameters()), 'CPU model required')
        for uid in ids:
            with decoding(member, ns, [uid], True) as (reader, root, adapter):
                vol, mask = reader(uid, grouped, root)
                need(getattr(adapter, 'errors', getattr(adapter, 'read_errors', 0)) == 0, 'archive read failure')
            with decoding(member, ns, [uid], False) as (reader, root, _): original, omask = reader(uid, grouped, root)
            need(np.array_equal(vol, original) and np.array_equal(mask, omask) and mask.any(), 'archive/stage decode parity')
            actual = infer_study(ns, model, member, vol, mask, 'cpu')
            # Independent call site uses the stage's own inference functions on identical decoded bytes.
            x = ns['gpu_windows'](original, omask, 94, 224 if member == 'R224' else 320, 'cpu')
            need(torch.equal(x, ns['eval_windows'](original, omask, 94, 224 if member == 'R224' else 320)), 'stage window tensor parity')
            if member == 'R384':
                mirror = ns['gpu_windows'](original, omask, 94, 320, 'cpu', mirror=True)
                need(torch.equal(mirror, ns['eval_windows'](original, omask, 94, 320, mirror=True)), 'stage mirror tensor parity')
                expected = ns['infer_pair_probs'](model, x, mirror, 'cpu'); del mirror
            else: expected = ns['infer_probs'](model, x, 'cpu')
            common.check_values(np.asarray([actual]), 1)
            error = float(np.max(np.abs(actual-expected)))
            need(error <= 1e-6, 'stage CPU output parity')
            results.append(dict(member=member, study_index=len(results)%2, decode_byte_equal=True,
                                window_tensor_equal=True,
                                input_sha256=hashlib.sha256(vol.tobytes()+mask.tobytes()).hexdigest(), max_abs_error=error))
            del x, vol, original
        del model; gc.collect()
    need(not torch.cuda.is_initialized(), 'CPU parity initialized CUDA')
    return save('smoke', stamp(state='CPU_SMOKE_ONLY', registration_sha256=sha(WORK/'registration.json'),
                              studies=2, results=results, seconds=time.monotonic()-start, cuda_initialized=False,
                              precision='fp32 engineering parity; R384 infer_probs uses existing hb113 CPU shim; not CUDA parity'))


def review_binding():
    return {name+'_sha256': sha(WORK/(name+'.json')) for name in ('registration', 'smoke', 'gpu-queue', 'eval-dry-run')} | {'tool_sha256': sha(SCRIPT)}


def check_review(review, binding):
    need(review.get('schema') == 'Q12b-review/v1' and review.get('verdict') == 'PASS', 'independent PASS required')
    need(review.get('builder') == BUILDER and review.get('reviewer') and review['reviewer'] != BUILDER, 'independent reviewer required')
    need(review.get('binding') == binding, 'review binding stale')
    for key in ('member_identity_checked', 'cpu_parity_checked', 'evaluation_checked', 'memory_watchdog_checked'):
        need(review.get(key) is True, 'review check absent: '+key)
    report = (ROOT/review['report_path']).resolve()
    need(report.is_relative_to(ROOT) and sha(report) == review['report_sha256'], 'review report drift')


def admit():
    reg = validate()
    smoke_value = io.read_json(WORK/'smoke.json')
    need(smoke_value['state'] == 'CPU_SMOKE_ONLY' and smoke_value['studies'] == 2 and
         smoke_value['registration_sha256'] == sha(WORK/'registration.json') and
         len(smoke_value['results']) == 4 and max(r['max_abs_error'] for r in smoke_value['results']) <= 1e-6,
         'fresh real CPU parity required')
    dry = io.read_json(WORK/'eval-dry-run.json')
    need(dry['state'] == 'CPU_DRY_RUN_ONLY' and dry['registration_sha256'] == sha(WORK/'registration.json'), 'real eval dry run stale')
    check_review(io.read_json(WORK/'independent-review.json'), review_binding())
    return stamp(state='ADMITTED_CPU_ONLY', studies=len(reg['ids']), independent_review_sha256=sha(WORK/'independent-review.json'))


def check_predictions(path, reg):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        need(set(data.files) == {'ids', 'targets', *MEMBERS}, 'prediction archive inventory')
        need(data['ids'].tolist() == reg['ids'] and data['targets'].tolist() == LABELS, 'prediction cohort/order/target drift')
        arrays = {m: data[m].copy() for m in MEMBERS}
    for a in arrays.values():
        common.check_values(a, len(reg['ids']))
        need(not np.any(np.all(a == .5, axis=1)), 'neutral fallback prohibited')
        need(np.all(np.ptp(a,axis=0)>0), 'constant target predictions prohibited')
    return arrays


def throughput_guard(history, now):
    """Per-member commit rate; no elapsed deadline. First 10 completed studies calibrate."""
    if len(history) < 11 or now-history[10] < 1800: return
    healthy = 10/max(history[10]-history[0], .001)
    recent = sum(t > now-1800 for t in history)/1800
    need(recent >= .25*healthy, 'throughput watchdog: below 25% calibrated commit rate for 30 minutes')


def worker():
    admit()
    execution_gate()
    import numpy as np
    import psutil
    import torch
    from tools import c85r7_flipall_timing as memory
    need(any('_supervisor' in p.cmdline() for p in psutil.Process().parents()), 'supervised private worker required')
    need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in psutil.Process().parents()), 'gpu_job ancestor required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'CUDA convention must be 0')
    reg = io.read_json(WORK/'registration.json'); ids = reg['ids']
    grouped = {u: [r for r in reg['rows'] if r['StudyInstanceUID'] == u] for u in ids}
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True
    out = WORK/'gpu'; desktop = io.read_json(out/'desktop.json'); arrays = {}; peaks = []; sequence = 0; start = time.monotonic()
    commits = {m: [] for m in MEMBERS}
    def progress(phase, member='', committed=False):
        nonlocal sequence
        sequence += 1
        if committed: commits[member].append(time.monotonic())
        io.receipt(out/'status.json', stamp(state='RUNNING', sequence=sequence, phase=phase, member=member,
                                           committed=committed, commit_times=commits, elapsed=time.monotonic()-start))
    for member in MEMBERS:
        progress('load', member); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ns = namespace(member); model = model_for(member, ns, reg, 'cuda:0'); values = []
        with decoding(member, ns, ids, True) as (reader, root, adapter):
            for index, uid in enumerate(ids):
                progress('decode', member); vol, mask = reader(uid, grouped, root)
                need(vol.shape == (96,384,384) and mask.any() and np.any(vol), 'empty stage decode')
                need(getattr(adapter, 'errors', getattr(adapter, 'read_errors', 0)) == 0, 'archive read failure')
                progress('infer', member); value = infer_study(ns, model, member, vol, mask, 'cuda:0')
                common.check_values(np.asarray([value]), 1); values.append(value)
                torch.cuda.synchronize(); peak = torch.cuda.max_memory_reserved(); memory.memory_admission(desktop,[peak])
                peaks.append(dict(member=member, study_index=index, peak_reserved_bytes=peak))
                common.atomic_npz(out/(member+'-partial.npz'), ids=np.asarray(ids[:index+1]), values=np.asarray(values))
                progress('commit', member, True); del vol, mask
        arrays[member] = np.asarray(values); del model; gc.collect()
    path = out/'predictions.npz'
    common.atomic_npz(path, ids=np.asarray(ids), targets=np.asarray(LABELS), **arrays)
    check_predictions(path, reg)
    io.atomic_json(out/'worker.json', stamp(state='INFERENCE_COMPLETE', studies=976, predictions_sha256=sha(path),
                                          registration_sha256=sha(WORK/'registration.json'), memory=peaks, seconds=time.monotonic()-start,
                                          checkpoints=reg['checkpoints'], embedded=reg['embedded'], device='cuda:0', torch=str(torch.__version__)))


def supervisor():
    admit()
    execution_gate()
    import psutil
    from tools import c85r7_flipall_timing as memory
    from tools import c85r7_r224_timing as guards
    from tools import hb149_monitor_v125 as monitor
    parents = psutil.Process().parents()
    need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in parents), 'gpu_job ancestor required')
    out = WORK/'gpu'; out.mkdir(parents=True, exist_ok=True)
    need(not any((out/n).exists() for n in ('receipt.json','worker.json','desktop.json')), 'no automatic replay of GPU attempt')
    query = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5)
    need(not memory.gpu_python_pids(query.stdout, {os.getpid(), *[p.pid for p in parents]}), 'another Python GPU owner')
    desktop = memory.desktop_snapshot(); io.atomic_json(out/'desktop.json', desktop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    telemetry = guards._TimingTelemetry(env,.1,desktop); child = None; owner = None; tracked = {}; error = None
    last = time.monotonic(); sequence = None; history = {m: [] for m in MEMBERS}
    try:
        telemetry.start()
        with (out/'worker.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([str(PYTHON),'-B',str(SCRIPT),'_worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            owner = monitor._WindowsJob(child)
            while child.poll() is None:
                now = time.monotonic(); status = io.read_receipt(out/'status.json')
                if status.get('sequence') != sequence:
                    last = now; sequence = status.get('sequence')
                    history = status.get('commit_times', history)
                snapshot = telemetry.snapshot(); need(not snapshot['memory_failure'],snapshot['memory_failure'])
                guards.watchdog(now,last,snapshot['latest_gpu'],desktop)
                if status.get('member') in history: throughput_guard(history[status['member']], now)
                time.sleep(.2)
            need(child.returncode == 0, 'worker failed; inspect completed worker.log')
    except BaseException as exc:
        error = exc
    finally:
        try:
            if child is not None: need(not monitor._cleanup(child,tracked,owner,psutil), 'owned worker cleanup failed')
        except BaseException as exc: error = error or exc
        try: telemetry.close()
        except BaseException as exc: error = error or exc
    if error:
        io.atomic_json(out/'receipt.json',stamp(state='FAILED',error=type(error).__name__+': '+str(error))); raise error
    try:
        observed = telemetry.snapshot(final=True); need(observed['complete'], 'complete telemetry required for result')
        wr = io.read_json(out/'worker.json'); reg = io.read_json(WORK/'registration.json')
        need(wr['state']=='INFERENCE_COMPLETE' and wr['studies']==976 and wr['registration_sha256']==sha(WORK/'registration.json'), 'worker stale/partial')
        need(sha(out/'predictions.npz') == wr['predictions_sha256'], 'prediction hash mismatch'); check_predictions(out/'predictions.npz', reg)
        memory.memory_admission(desktop,[p['peak_reserved_bytes'] for p in wr['memory']],observed['peak_gpu_mib']*2**20)
        io.atomic_json(out/'receipt.json',stamp(state='COMPLETE',registration_sha256=sha(WORK/'registration.json'),
                                               predictions_sha256=wr['predictions_sha256'], worker_sha256=sha(out/'worker.json'),
                                               independent_review_sha256=sha(WORK/'independent-review.json'), telemetry=observed))
    except BaseException as exc:
        io.atomic_json(out/'receipt.json',stamp(state='FAILED',error=type(exc).__name__+': '+str(exc))); raise
    return stamp(state='COMPLETE',studies=976)


def execution_gate():
    import psutil
    from tools.job_runtime import job_status, LOGS
    job = job_status(LOGS, TAG, expected_command=launch_command(), tail=False)
    need(job['state']=='running' and job.get('pid') in {p.pid for p in psutil.Process().parents()}, 'exclusive gpu_job lock required')
    need(job_status(LOGS,AFTER,tail=False)['state'] in ('SUCCESS','FAILED'), 'K0 predecessor must exist and have ended; queue must not skip an unregistered predecessor')


def run():
    admit()
    execution_gate()
    # Supervisor remains distinct so the private worker can prove both ancestry gates.
    result = subprocess.run([str(PYTHON),'-B',str(SCRIPT),'_supervisor'],cwd=ROOT)
    need(result.returncode==0,'GPU supervisor failed')
    return stamp(state='COMPLETE', studies=976)


def auc(y, scores):
    import numpy as np
    from scipy.stats import rankdata
    yy = np.asarray(y, bool); n1 = yy.sum(); n0 = len(yy)-n1
    if not n1 or not n0: return float('nan')
    return float((rankdata(scores)[yy].sum()-n1*(n1+1)/2)/(n1*n0))


def summarize(compact, arrays, refs, draws=1000, seed=20261016):
    import numpy as np
    from scipy.stats import rankdata
    need(set(refs)=={'jev_hybrid','aquila_v1','dread','consensus'}, 'exact four rulers required')
    n = len(compact); common.check_values(compact,n)
    for values in arrays.values(): common.check_values(values,n)
    need(set(arrays)==set(MEMBERS), 'two exact raw members required')
    sources = dict(arrays, R_blend=.65*rankdata(arrays['R384'],axis=0,method='average')/n+.35*rankdata(arrays['R224'],axis=0,method='average')/n)
    names = list(refs); snames = list(sources); point = np.full((3,4,12),np.nan); compact_auc = np.full((4,12),np.nan)
    for r, name in enumerate(names):
        y, mask, _weight = refs[name]
        need(y.shape == mask.shape == (n,12) and mask.dtype==np.bool_ and np.isfinite(y).all() and np.isin(y,[0,1]).all(), 'ruler mask/value contract')
        for j in range(12):
            m=mask[:,j]; compact_auc[r,j]=auc(y[m,j],compact[m,j])
            for s,name2 in enumerate(snames): point[s,r,j]=compact_auc[r,j]-auc(y[m,j],sources[name2][m,j])
    need(np.isfinite(point).all(), 'single-class real cohort target')
    boots=np.full((draws,3,4,12),np.nan); rng=np.random.default_rng(seed)
    for draw in range(draws):
        idx=rng.integers(0,n,n)
        for r,name in enumerate(names):
            y,mask,_=refs[name]
            for j in range(12):
                selected=idx[mask[idx,j]]; yy=y[selected,j]; a=auc(yy,compact[selected,j])
                for s,name2 in enumerate(snames): boots[draw,s,r,j]=a-auc(yy,sources[name2][selected,j])
    def interval(values):
        v=np.asarray(values); good=v[np.isfinite(v)]
        need(len(good)>=max(1,int(.9*draws)), 'too few valid bootstrap draws')
        return dict(ci95=np.quantile(good,[.025,.975],method='linear').tolist(), valid_draws=len(good),
                    undefined_draws=int(draws-len(good)), p_positive=float(np.mean(good>0)))
    result={}
    for s,name in enumerate(snames):
        result[name]={'per_ruler': {}, 'four_ruler_mean': {}}
        for r,ruler in enumerate(names):
            result[name]['per_ruler'][ruler]={lab:dict(compact_auc=float(compact_auc[r,j]),source_auc=float(compact_auc[r,j]-point[s,r,j]),
                delta=float(point[s,r,j]),**interval(boots[:,s,r,j])) for j,lab in enumerate(LABELS)}
        for j,lab in enumerate(LABELS):
            mean=float(point[s,:,j].mean())
            result[name]['four_ruler_mean'][lab]=dict(delta=mean,**interval(boots[:,s,:,j].mean(axis=1)),
                reading='directionally informative descriptive point estimate' if mean>=0 else 'uninterpretable: in-sample R advantage')
        result[name]['other5_macro']=dict(targets=[LABELS[j] for j in OTHER5],delta=float(point[s][:,OTHER5].mean()),
                                        **interval(boots[:,s][:,:,OTHER5].mean(axis=(1,2))))
        result[name]['all12_macro']=dict(delta=float(point[s].mean()),**interval(boots[:,s].mean(axis=(1,2))))
    return result


def evaluate(dry=False):
    cpu_only(); reg=validate(); ids,compact,refs,_=fable_context()
    if dry:
        # Identity null runs real roster/rulers through every statistic without deriving any R result.
        arrays={m:compact.copy() for m in MEMBERS}; draws=20
    else:
        admit(); receipt=io.read_json(WORK/'gpu/receipt.json')
        need(receipt['state']=='COMPLETE' and receipt['registration_sha256']==sha(WORK/'registration.json') and
             receipt['independent_review_sha256']==sha(WORK/'independent-review.json'), 'incomplete/stale GPU evidence')
        need(sha(WORK/'gpu/predictions.npz')==receipt['predictions_sha256'], 'GPU prediction hash mismatch')
        need(sha(WORK/'gpu/worker.json')==receipt['worker_sha256'], 'GPU worker hash mismatch')
        arrays=check_predictions(WORK/'gpu/predictions.npz',reg); draws=RULES['draws']
    result=summarize(compact,arrays,refs,draws=draws)
    if dry:
        need(all(abs(cell['delta'])<1e-12 for src in result.values() for cell in src['four_ruler_mean'].values()), 'identity-null evaluator failed')
        return save('eval-dry-run',stamp(state='CPU_DRY_RUN_ONLY',registration_sha256=sha(WORK/'registration.json'),studies=len(ids),draws=draws,
            real_roster_rulers=True, identity_null_pass=True, scientific_output=False, reading_rule=READING_RULE))
    save('evaluation',stamp(state='RESULT_PENDING_INDEPENDENT_REVIEW',registration_sha256=sha(WORK/'registration.json'),
        predictions_sha256=receipt['predictions_sha256'],reading_rule=READING_RULE,rules=RULES,studies=len(ids),results=result))
    return stamp(state='RESULT_PENDING_INDEPENDENT_REVIEW',studies=len(ids),output=str(WORK/'evaluation.json'),reading_rule=READING_RULE)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('register','prepare','validate','smoke','admit','review-binding','evaluate','run','_supervisor','_worker'))
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.command not in ('run','_supervisor','_worker'): os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if args.command=='validate': cpu_only(); validate(); result=stamp(state='CPU_VALIDATED')
    elif args.command=='evaluate': result=evaluate(args.dry_run)
    elif args.command=='review-binding': result=review_binding()
    elif args.command=='_worker': worker(); return
    elif args.command=='_supervisor': result=supervisor()
    else: result=globals()[args.command]()
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
