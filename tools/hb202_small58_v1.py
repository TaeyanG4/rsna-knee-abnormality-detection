"""H202 Small58 cache v1.0.0: static preparation; B-only reviewed GPU execution.

One missing Small10 arm, unchanged serving helpers, no labels/metrics/network.
Never invokes the old HB199 registration/admission/CLI or a release notebook.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb145_output_io as io
from tools import job_runtime as jr

VERSION = '1.0.0'
BUILDER = '/root/k1_result_review'
SELF = 'tools/hb202_small58_v1.py'
TEST = 'tools/test_hb202_small58_v1.py'
PREP = ROOT / '.local/scratch/slot-b/research-resume-1018/h202-prep'
RUN = PREP.parent / 'h202-small58/run1'
TAG = 'H202_INF2_SMALL58_V1'
KEY = 'R6W958-INF2-V1'
PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
INVENTORY_SHA = '4dfe825eb8d69b56bbfbc747e4e185950e4038610281d24e6f883b6c05776c07'
MEMBERS = [f'S-s{s}-f{f}' for s in (42, 43) for f in range(5)]
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ROSTER_SHA = '3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea'
PARENT = ROOT / '.local/scratch/hb167-common/hb167/gpu/predictions.npz'
FUNCS = ('inf2_small_meta', 'inf2_load_small', 'inf2_small_windows',
         'plane_flags', '_infer_one', 'rank', 'aggregate')
EXTRA_SOURCES = (SELF, TEST, 'tools/hb145_output_io.py', 'tools/job_runtime.py',
    'tools/gpu_queue.py', 'tools/gpu_job.py', 'tools/hb155_watchdog_v1.py',
    'tools/hb150_io_v1.py', 'tools/native_codex_dispatch_record_v1.py')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return io.read_json(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def digest_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def infra():
    # Utilities only: no Context/register/admission/run/worker from the old tool.
    from tools import hb199_inf2_serving_v1
    return hb199_inf2_serving_v1


def command(worker=False):
    return [PYTHON, '-X', 'utf8', '-B', str(ROOT / SELF), '_worker' if worker else 'run']


def queue():
    return [dict(tag=TAG, command=command(), continue_on_failure=False)]


def recipe():
    return dict(candidate=KEY, members=MEMBERS, targets=TARGETS, rows=58,
        ordered_uid_sha256=ROSTER_SHA, shape=[10, 58, 12], dtype='float32',
        centers=36, input_shape=[64, 336, 336], windows_shape=[36, 3, 224, 224],
        window_policy='within-slot-valid', crop=[48, 288], normalization='ImageNet',
        device='cuda:0', precision='FP16 forward; FP32 sigmoid and TTA probability mean',
        aggregation='((small[:5] + small[5:]) / 2).mean(0), float32; CPU follow-up only',
        reference_indices=[0], renderer_rows=58, reference_max_abs=1e-6,
        peak_limit_bytes=12 * 2**30, reserve_bytes=2 * 2**30,
        idle_seconds=900, wddm_seconds=600, collapse_seconds=1800, collapse_fraction=.25,
        exposure='EXPOSED_GOLD58', clean_oof=False, qualified_for_submission=False)


def verify_pins(pins):
    for name, expected in pins.items():
        path = (ROOT / name).resolve()
        need(path.is_relative_to(ROOT) and not path.is_relative_to(RUN), 'unsafe pin path')
        need(pin(path) == expected, 'source/input pin changed: ' + name)


def inventory():
    need(sha(PREP / 'INVENTORY.json') == INVENTORY_SHA, 'inventory identity')
    value = read(PREP / 'INVENTORY.json')
    need(value['candidate'] == KEY and value['gold58']['ordered_uid_sha256'] == ROSTER_SHA
         and value['gold58']['targets'] == TARGETS and value['exposure'] == 'EXPOSED_GOLD58'
         and value['clean_oof'] is False, 'inventory scientific identity')
    verify_pins(value['pins'])  # Small source/metadata only; no weights/volume payloads.
    return value


def members(inv):
    manifest = read(ROOT / inv['stage'] / 'inf2-small-manifest.json')
    entries = manifest['entries']
    need(manifest['schema'] == 'inf2-small10-v1' and manifest['candidate'] == KEY
         and [e['id'] for e in entries] == MEMBERS, 'fixed Small manifest')
    need(len(set(e['sha256'] for e in entries)) == 10, 'duplicate checkpoint')
    for e, src, (seed, fold) in zip(entries, inv['members'], [(s, f) for s in (42, 43) for f in range(5)]):
        need(e['family'] == 'small' and e['arch'] == 'convnext_small.fb_in22k_ft_in1k'
             and (e['seed'], e['fold']) == (seed, fold) and e['id'] == src['id']
             and e['source'] == src['source'] and e['path'] == src['release_path']
             and e['sha256'] == e['source_sha256'] == src['expected_sha256']
             and e['bytes'] == src['bytes'] == 198775478 and e['spec_sha256'], 'member identity')
        identity = e['source_identity']
        need(all(identity[k] == e[k] for k in ('id', 'arch', 'seed', 'fold', 'sha256', 'bytes', 'spec_sha256'))
             and identity['path'] == e['source'] and identity['family'] == 'S', 'original model identity')
        need((ROOT / e['source']).resolve().is_relative_to(ROOT), 'checkpoint path escape')
    return entries


def embedded(inv):
    from tools.c85r7_r6w958inf2_v1148 import unpack
    path = ROOT / inv['stage'] / 'kernel/release.ipynb'
    need(sha(path) == inv['kernel_sha256'], 'candidate notebook identity')
    files, _, _ = unpack(read(path))  # Literal AST/archive extraction; never executes cells.
    code = files['exp169_partner_bundle_release.py']
    helper = (ROOT / 'tools/c85r7_inf2_small_v1.py').read_bytes()
    need(code.endswith(b'\n\n' + helper), 'actual embedded Small helper differs')
    tree = ast.parse(code)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    need(sorted(n.name for n in selected) == sorted(FUNCS), 'exact pure reference functions')
    ns = dict(TARGETS=TARGETS, need=need, sha=sha)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<authenticated-INF2-pure-functions>', 'exec'), ns)
    return ns, {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def roster():
    import numpy as np
    with np.load(PARENT, allow_pickle=False) as z:
        ids, targets = z['ids'].astype(str).tolist(), z['targets'].astype(str).tolist()
    need(len(ids) == len(set(ids)) == 58 and uid_hash(ids) == ROSTER_SHA and targets == TARGETS,
         'fixed parent roster/target order')
    return ids


def prepare():
    inv = inventory()
    entries = members(inv)
    _, file_hashes = embedded(inv)
    roster()
    paths = [*EXTRA_SOURCES, *[(PREP / n).relative_to(ROOT).as_posix()
              for n in ('INVENTORY.json', 'LAUNCH-SPEC.md', 'WORKER-CONTRACT.md')]]
    pins = dict(inv['pins'])
    pins.update({p: pin(ROOT / p) for p in paths})
    from tools.gpu_queue import validate_plan
    validate_plan(queue())
    value = dict(schema='h202-small58-contract/v1', version=VERSION,
        status='PREPARED_NOT_REVIEWED_NOT_RUN', builder=BUILDER, inventory_sha256=INVENTORY_SHA,
        recipe=recipe(), pins=pins, members=entries, embedded_sha256=file_hashes,
        queue=queue(), output=RUN.relative_to(ROOT).as_posix(),
        raw_row_hashes_measured=False, weight_hashes_measured=False,
        rendering_measured=False, model_forwards=0, gpu_execution=False)
    infra().fresh(PREP / 'CONTRACT.json', value)
    infra().fresh(PREP / 'QUEUE.json', queue())
    return dict(status=value['status'], pins=len(pins), contract_sha256=sha(PREP / 'CONTRACT.json'))


def validate():
    need(jr.ROOT == ROOT and os.environ.get('KNEE_WORKSPACE_ROOT') in (None, str(ROOT)), 'fixed workspace')
    value = read(PREP / 'CONTRACT.json')
    need(value['schema'] == 'h202-small58-contract/v1' and value['version'] == VERSION
         and value['status'] == 'PREPARED_NOT_REVIEWED_NOT_RUN' and value['builder'] == BUILDER
         and value['inventory_sha256'] == INVENTORY_SHA and value['recipe'] == recipe()
         and value['queue'] == queue() and value['output'] == RUN.relative_to(ROOT).as_posix(), 'frozen contract')
    verify_pins(value['pins'])
    inv = inventory()
    need(value['members'] == members(inv), 'frozen members')
    reference, hashes = embedded(inv)
    need(value['embedded_sha256'] == hashes and read(PREP / 'QUEUE.json') == queue(), 'frozen source/queue')
    return value, inv, reference


def parse_review(text, contract_sha):
    def field(name):
        values = re.findall(r'^' + re.escape(name) + r': ([^\r\n]+)$', text, re.M)
        need(len(values) == 1, 'one exact review field: ' + name)
        return values[0]
    need(field('Contract-SHA256') == contract_sha and field('Verdict') == 'PASS', 'independent PASS required')
    reviewer = field('Reviewer-Agent')
    need(re.fullmatch(r'/root/[a-z0-9_]+', reviewer) and reviewer != BUILDER, 'distinct reviewer required')
    return reviewer


def review():
    path = PREP / 'H202-LAUNCH-REVIEW.md'
    reviewer = parse_review(path.read_text(encoding='utf-8'), sha(PREP / 'CONTRACT.json'))
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), reviewer=reviewer,
                contract_sha256=sha(PREP / 'CONTRACT.json'))


def check_job(job, own, parent, actual_command, parent_args, alive):
    need(job['schema'] == 'gpu-job/v2' and job['phase'] == 'running' and job['tag'] == TAG
         and job['cwd'] == str(ROOT) and job['command'] == command()
         and job['command_sha256'] == jr.command_digest(command(), ROOT)
         and job['child_identity'] == own and actual_command == command()
         and dict(pid=job['pid'], create_time=job['create_time']) == parent, 'actual gpu_job identity')
    need(str(ROOT / 'tools/gpu_job.py') in parent_args
         and parent_args[-3:] == ['--_run-job', TAG, job['run_id']]
         and alive(own) == alive(parent) == 'alive', 'actual gpu_job parent')
    return job


def job_identity(own=None):
    import psutil
    own = own or jr.identity()
    proc = psutil.Process(own['pid'])
    parent = jr.identity(proc.ppid())
    # Bounded publication-race retry only, not queue/resource waiting.
    for attempt in range(40):
        try:
            return check_job(read(ROOT / 'artifacts/logs' / TAG / 'job.json'), own, parent,
                             proc.cmdline(), psutil.Process(parent['pid']).cmdline(), jr.liveness)
        except (ValueError, KeyError, OSError):
            if attempt == 39:
                raise
            time.sleep(.05)


def check_compute_apps(text, allowed):
    for line in text.splitlines():
        parts = line.split(',', 1)
        if len(parts) == 2 and 'python' in parts[1].lower():
            need(parts[0].strip().isdigit() and int(parts[0].strip()) in allowed, 'another Python GPU owner')


def validate_values(values):
    import numpy as np
    need(values.shape == (10, 58, 12) and values.dtype == np.float32
         and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'complete finite FP32 Small10')


def seed_pair_fold_mean(values):
    """Pure future CPU helper only; worker preserves leaves and never ranks."""
    validate_values(values)
    return ((values[:5] + values[5:]) / 2).mean(0)


def check_npz(path, ids):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        need(set(z.files) == {'ids', 'targets', 'member_ids', 'probabilities'}
             and z['ids'].tolist() == ids and len(set(ids)) == 58 and uid_hash(ids) == ROSTER_SHA
             and z['targets'].tolist() == TARGETS and z['member_ids'].tolist() == MEMBERS, 'exact output order')
        validate_values(z['probabilities'])


def completed(worker, process, contract_sha, reviewed, npz_sha, data_sha, alive, entries):
    need(process['schema'] == 'h199-serving-process/v1' and process['child_exit_code'] == 0
         and process['owned_cleanup_complete'] is True and process['survivors'] == []
         and process['failure'] is None and alive(process['child']) == 'dead', 'clean completed process')
    need(worker['schema'] == 'h202-small58-worker/v1' and worker['version'] == VERSION
         and worker['state'] == 'COMPLETE' and worker['recipe'] == recipe()
         and worker['contract_sha256'] == contract_sha and worker['review'] == reviewed
         and worker['output_sha256'] == npz_sha and worker['data_sha256'] == data_sha
         and worker['predictions'] == 580 and worker['failures'] == worker['fallback_count'] == 0
         and worker['pid'] == process['child']['pid'] and worker['member_ids'] == MEMBERS
         and worker['ordered_uid_sha256'] == ROSTER_SHA and worker['targets'] == TARGETS
         and worker['renderer_rows'] == 58 and len(worker['model_measurements']) == 10
         and worker['exposure'] == 'EXPOSED_GOLD58' and worker['clean_oof'] is False
         and worker['qualified_for_submission'] is False,
         'complete scientific cache required')
    environment = worker['environment']
    need(environment['device'] == 'cuda:0' and environment['autocast_dtype'] == 'torch.float16'
         and environment['cudnn_tf32'] is False and environment['matmul_tf32'] is False
         and environment['cudnn_benchmark'] is False and environment['cudnn_deterministic'] is True
         and environment['deterministic'] is True and environment['matmul_precision'] == 'highest',
         'actual serving backend contract')
    for i, row in enumerate(worker['model_measurements']):
        need(row['member'] == MEMBERS[i] and row['studies'] == 58 and row['reference_index'] == 0
             and row['model_sha256'] == entries[i]['sha256']
             and math.isfinite(row['reference_max_abs']) and 0 <= row['reference_max_abs'] <= 1e-6
             and row['deleted'] is True and row['views'] == 2
             and 0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
             and row['reference_peak_reserved_bytes'] > 0, 'complete reference/member evidence')
        infra().memory_ok(worker['desktop'], [row['reference_peak_reserved_bytes'], row['peak_reserved_bytes']])


def launch():
    need(os.name == 'nt', 'reviewed Windows local execution only')
    contract, _, _ = validate()
    reviewed = review()
    job = job_identity()
    need(not jr.active_jobs(ROOT / 'artifacts/logs', exclude=TAG), 'exclusive GPU job required')
    cp = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                        capture_output=True, text=True, check=True, timeout=3,
                        creationflags=subprocess.CREATE_NO_WINDOW)
    check_compute_apps(cp.stdout, {os.getpid(), job['pid']})
    desktop = infra().snapshot()
    infra().memory_ok(desktop, [0])
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(exist_ok=False)  # No implicit reuse/retry of any attempted cache.
    (RUN / 'journal').mkdir()
    token = uuid.uuid4().hex
    infra().fresh(RUN / 'ticket.json', dict(parent=jr.identity(), token_sha256=jr.digest(token),
        contract_sha256=sha(PREP / 'CONTRACT.json'), review=reviewed))
    infra().fresh(RUN / 'desktop.json', desktop)
    infra().fresh(RUN / 'bound-job.json', job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', INF2_SERVING_TOKEN=token,
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUTF8='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        process = infra().supervise(command(True), RUN, env, token, False)
        worker = read(RUN / 'worker.json')
        validate()
        need(reviewed == review(), 'review changed during execution')
        ids = roster()
        check_npz(RUN / 'probabilities.npz', ids)
        completed(worker, process, sha(PREP / 'CONTRACT.json'), reviewed,
                  sha(RUN / 'probabilities.npz'), sha(RUN / 'data.json'), jr.liveness, contract['members'])
        result = dict(schema='h202-small58-result/v1', version=VERSION,
            state='COMPLETE_PENDING_INDEPENDENT_RESULT_REVIEW', candidate=KEY, tag=TAG,
            contract_sha256=sha(PREP / 'CONTRACT.json'), review=reviewed,
            hashes={n: sha(RUN / n) for n in ('worker.json', 'process.json', 'probabilities.npz',
                    'data.json', 'desktop.json', 'pre-cuda.json', 'ticket.json', 'bound-job.json')},
            studies=58, members=10, targets=12, predictions=580, failures=0, fallback_count=0,
            exposure='EXPOSED_GOLD58', clean_oof=False, qualified_for_submission=False)
        infra().fresh(RUN / 'result.json', result)
        return dict(state=result['state'], predictions=580)
    except BaseException as exc:
        infra().diagnostic(RUN / 'failure.json', dict(state='FAILED', error=type(exc).__name__,
            automatic_retry=False, partial_journals_preserved=True))
        raise


def bind_row(volume, mask, centers, index):
    import numpy as np
    need(volume.shape == (64, 336, 336) and volume.dtype == np.uint8
         and mask.shape == (64,) and np.isin(mask, [0, 1]).all()
         and 50 <= int(mask.sum()) <= 64 and len(centers) == 36
         and all(type(c) is int and -1 <= c <= 62 for c in centers), 'fixed real row contract')
    return dict(index=index, volume_sha256=digest_array(volume), mask_sha256=digest_array(mask),
                centers=centers, global_row=index, shard_row=index, bytes=volume.nbytes)


def worker():
    import psutil
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'literal logical CUDA visibility')
    ticket = read(RUN / 'ticket.json')
    need(ticket['parent'] == jr.identity(os.getppid())
         and ticket['token_sha256'] == jr.digest(os.environ.get('INF2_SERVING_TOKEN', ''))
         and psutil.Process(os.getppid()).cmdline() == command(), 'owned worker identity/token')
    job = job_identity(ticket['parent'])
    need(read(RUN / 'bound-job.json') == job, 'bound controller job')
    contract, inv, reference = validate()
    reviewed = review()
    need(ticket['contract_sha256'] == sha(PREP / 'CONTRACT.json') and ticket['review'] == reviewed,
         'worker contract/review binding')
    sequence = units = 0
    rate = None
    start = time.monotonic()
    journals = []

    def progress(phase, committed=False):
        nonlocal sequence
        sequence += 1
        infra().diagnostic(RUN / 'phase.json', dict(pid=os.getpid(), token=os.environ['INF2_SERVING_TOKEN'],
            wall_time=time.time(), completed_at=time.monotonic(), sequence=sequence,
            event='commit' if committed else 'enter', phase=phase,
            detail=dict(progress_kind='prediction' if phase in ('prediction', 'study_complete') else phase,
                        cumulative_units=units, measured_rate=rate)))

    def journal(phase, **detail):
        path = RUN / 'journal' / f'{len(journals):04}.json'
        io.journal(path, dict(identity=KEY, phase=phase, elapsed=time.monotonic()-start, **detail), RUN)
        journals.append(dict(path=path.relative_to(RUN).as_posix(), sha256=sha(path)))

    progress('startup')
    from tools import hb105_serving_v101 as helper
    modules = helper.imports()
    np, torch, mil, serving = modules
    need(not torch.cuda.is_initialized(), 'desktop snapshot must precede worker CUDA')
    ids = roster()
    corpus = mil.common.Corpus(ROOT / inv['corpus']['path'])
    need(corpus.contract == 'raptor64_336_c140_s02' and corpus.depth == 64
         and len(set(corpus.ids)) == len(corpus.ids)
         and [corpus.row[u] for u in ids] == list(range(58)), 'exact authenticated corpus rows')
    for shard in inv['corpus']['shards']:
        need((corpus.dir / shard['file']).stat().st_size == shard['bytes'], 'shard size changed')
    rows = []
    for i, uid in enumerate(ids):
        tick = time.monotonic()
        progress('render')
        volume, mask = corpus.volume(uid), corpus.mask(uid)
        centers = [int(x) for x in mil.common.select_centers(mask, 64, 36, corpus.spec, 'within-slot-valid')]
        row = bind_row(volume, mask, centers, i)
        x = serving.mil_windows(volume, centers)
        expected = reference['inf2_small_windows'](volume, centers)
        need(x.shape == (36, 3, 224, 224) and x.dtype == np.float32 and np.isfinite(x).all()
             and np.array_equal(x, expected), 'actual embedded renderer parity')
        flags = reference['plane_flags'](centers, corpus.spec)
        mirrored = x.copy(); mirrored[flags] = mirrored[flags, :, :, ::-1]
        original_flip = mil.flip_inputs(torch.from_numpy(x)[None], [uid], corpus, 36)[0].numpy()
        need(np.array_equal(original_flip, mirrored), 'actual embedded plane flip parity')
        row.update(windows_sha256=digest_array(x), flip_sha256=digest_array(mirrored), renderer_exact=True)
        rows.append(row)
        journal('row_bound_and_renderer_parity', index=i, seconds=time.monotonic()-tick,
                **{k: v for k, v in row.items() if k != 'index'})
        progress('render', True)
    need(sum(row['bytes'] for row in rows) == 419069952, 'exact selected raw bytes')
    infra().fresh(RUN / 'data.json', dict(ordered_uid_sha256=ROSTER_SHA, corpus=inv['corpus']['path'],
        rows=rows, selected_bytes=419069952, full_shard_hash_rechecked=False,
        exposure='EXPOSED_GOLD58', clean_oof=False))

    def windows(i):
        volume, mask = corpus.volume(ids[i]), corpus.mask(ids[i])
        need(digest_array(volume) == rows[i]['volume_sha256'] and digest_array(mask) == rows[i]['mask_sha256'],
             'selected row bytes changed')
        x = serving.mil_windows(volume, rows[i]['centers'])
        need(digest_array(x) == rows[i]['windows_sha256'], 'render changed')
        return x

    desktop = read(RUN / 'desktop.json')
    current = infra().snapshot()
    desktop = dict(desktop, desktop_bytes=max(desktop['desktop_bytes'], current['desktop_bytes']),
                   total_bytes=min(desktop['total_bytes'], current['total_bytes']))
    infra().memory_ok(desktop, [0])
    infra().fresh(RUN / 'pre-cuda.json', desktop)
    serving.backend(strict=True)
    torch.cuda.set_device(0)
    need(torch.cuda.device_count() == 1, 'one visible GPU')
    import timm
    environment = dict(os=sys.platform, python=sys.version, numpy=np.__version__, torch=torch.__version__,
        timm=timm.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0), device='cuda:0', cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, matmul_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled(), autocast_dtype='torch.float16')
    values = np.full((10, 58, 12), np.nan, np.float32)
    measurements = []
    for j, entry in enumerate(contract['members']):
        path = ROOT / entry['source']
        tick = time.monotonic()
        progress('model_load')
        need(pin(path) == {k: entry[k] for k in ('bytes', 'sha256')}, 'actual weight SHA256')
        journal('weight_hash', member=MEMBERS[j], seconds=time.monotonic()-tick, sha256=entry['sha256'])
        model = None
        torch.cuda.reset_peak_memory_stats(0)
        try:
            tick = time.monotonic()
            model = reference['inf2_load_small'](path, entry, 'cuda:0')
            torch.cuda.synchronize(0)
            journal('reference_load', member=MEMBERS[j], seconds=time.monotonic()-tick)
            progress('model_load', True)
            x = windows(0)
            flags = reference['plane_flags'](rows[0]['centers'], corpus.spec)
            mirror = x.copy(); mirror[flags] = mirror[flags, :, :, ::-1]
            progress('prediction')
            tick = time.monotonic()
            baseline = (reference['_infer_one'](model, x, 'cuda:0')
                        + reference['_infer_one'](model, mirror, 'cuda:0')) / 2
            torch.cuda.synchronize(0)
            reference_peak = int(torch.cuda.max_memory_reserved(0))
            infra().memory_ok(desktop, [reference_peak])
            journal('release_reference', member=MEMBERS[j], index=0,
                    seconds=time.monotonic()-tick, peak_reserved_bytes=reference_peak)
            progress('model_load', True)
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache(); torch.cuda.synchronize(0)
        model = None
        torch.cuda.reset_peak_memory_stats(0)
        try:
            progress('model_load')
            tick = time.monotonic()
            model = helper.load_model(path, entry, 'cuda:0', modules)
            torch.cuda.synchronize(0)
            journal('original_load', member=MEMBERS[j], seconds=time.monotonic()-tick)
            progress('model_load', True)
            for i in range(58):
                progress('render'); tick = time.monotonic(); x = windows(i)
                render_seconds = time.monotonic()-tick
                progress('prediction')
                tick = time.monotonic()
                value, timings = helper.prediction(model, x, ids[i], corpus, 'cuda:0', modules)
                duration = time.monotonic() - tick
                need(value.shape == (12,) and value.dtype == np.float32 and np.isfinite(value).all()
                     and ((value >= 0) & (value <= 1)).all(), 'finite original prediction')
                if i == 0:
                    error = float(np.max(np.abs(value - baseline)))
                    need(math.isfinite(error) and error <= 1e-6, 'actual embedded model/TTA parity')
                torch.cuda.synchronize(0)
                peak = int(torch.cuda.max_memory_reserved(0))
                infra().memory_ok(desktop, [peak])
                values[j, i] = value
                journal('prediction', member=MEMBERS[j], index=i, probabilities=value.tolist(),
                        seconds=duration, render_seconds=render_seconds,
                        timings=timings, peak_reserved_bytes=peak)
                units += 1
                rate = max(rate or 0, 1 / max(duration, 1e-9))
                progress('study_complete', True)
            allocated = int(torch.cuda.max_memory_allocated(0))
            need(0 < allocated <= peak, 'measured positive memory')
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache(); torch.cuda.synchronize(0)
        measurements.append(dict(member=MEMBERS[j], model_sha256=entry['sha256'], studies=58,
            reference_index=0, reference_max_abs=error, reference_peak_reserved_bytes=reference_peak,
            peak_reserved_bytes=peak, peak_allocated_bytes=allocated, deleted=True, views=2))
        journal('model_complete', **measurements[-1])
    need(units == 580, 'all ten x58 predictions')
    validate_values(values)
    infra().atomic_npz(RUN / 'probabilities.npz', ids=np.asarray(ids), targets=np.asarray(TARGETS),
                       member_ids=np.asarray(MEMBERS), probabilities=values)
    check_npz(RUN / 'probabilities.npz', ids)
    validate()
    need(review() == reviewed, 'review unchanged')
    result = dict(schema='h202-small58-worker/v1', version=VERSION, state='COMPLETE', pid=os.getpid(),
        recipe=recipe(), contract_sha256=sha(PREP / 'CONTRACT.json'), review=reviewed,
        member_ids=MEMBERS, targets=TARGETS, ordered_uid_sha256=ROSTER_SHA,
        output_sha256=sha(RUN / 'probabilities.npz'), data_sha256=sha(RUN / 'data.json'),
        desktop=desktop, environment=environment, renderer_rows=58, model_measurements=measurements,
        predictions=units, failures=0, fallback_count=0, journals=journals,
        seconds=time.monotonic()-start, exposure='EXPOSED_GOLD58', clean_oof=False,
        qualified_for_submission=False)
    infra().fresh(RUN / 'worker.json', result)
    progress('finished', True)
    return dict(state='COMPLETE', predictions=units)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'run', '_worker'))
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare()
    elif args.command == 'validate':
        value, _, _ = validate()
        result = dict(status='STATIC_VALID_NOT_ADMITTED', pins=len(value['pins']), model_forwards=0)
    else:
        result = worker() if args.command == '_worker' else launch()
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
