"""Build a NEW private EXP133 Gold58 timing kernel. No upload or local inference.

Only the copied notebook is instrumented; released B3 bytes remain immutable.
CNX hooks fail closed if the attached reader cannot attribute measurements to UIDs.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import exp133_oai_b3 as b3
import exp131_gold58_convnext as gold

OUT = ROOT / '.local/exp133-oai-b3/timing60'
RELEASE = ROOT / '.local/exp133-oai-b3/blend_b3'
KERNEL = 'taeyangg4/knee-exp133-b3-timing60'


def replace(source, old, new):
    b3.need(source.count(old) == 1, 'Timing instrumentation source drift: ' + old[:90])
    return source.replace(old, new, 1)


def projection(stages, n=1300):
    """Measured steady-state wall rate; retain B3's fixed partner allowances.

    Decode/forward sums overlap for pipelined OAI, so never sum them to project
    its throughput. Also report conservative per-study p95 diagnostics.
    """
    import math
    b3.need(type(n) is int and n > 0, 'Invalid projection cohort size')
    b3.need(set(stages) == {'oai', 'rad', 'cnx'}, 'Missing/extra timing stages')
    out = {}
    cohort = None
    for name in ('oai', 'rad', 'cnx'):
        s = stages[name]
        ids = [r['uid'] for r in s['studies']]
        b3.need(len(ids) == 58 and len(set(ids)) == 58 and
                all(isinstance(uid, str) and uid for uid in ids),
                'Incomplete timing cohort: ' + name)
        if cohort is None:
            cohort = ids
        b3.need(ids == cohort, 'Timing stage cohort/order differs: ' + name)
        for row in s['studies']:
            b3.need(all(isinstance(row[k], (int, float)) and
                        math.isfinite(row[k]) and row[k] > 0
                        for k in ('decode_seconds', 'forward_seconds')),
                    'Unattributed decode/forward timing: ' + name)
        b3.need(all(isinstance(s[k], (int, float)) and math.isfinite(s[k]) and s[k] > 0
                    for k in ('steady_seconds',) + (('seconds_total',) if 'seconds_total' in s else ())),
                'Invalid measured rate')
        b3.need(isinstance(s['peak_gpu_allocated_bytes'], list) and s['peak_gpu_allocated_bytes'] and
                all(type(v) is int and v >= 0 for v in s['peak_gpu_allocated_bytes']),
                'Invalid GPU peak memory')
        rate = (s.get('seconds_total', s['steady_seconds']) if name == 'oai' else s['steady_seconds']) / 58
        factor = 1.3 if name == 'rad' else 1.0
        b3.need(rate > 0, 'Invalid measured rate')
        tail = sorted(r['decode_seconds'] + r['forward_seconds'] for r in s['studies'])[54]
        fixed = 0 if name == 'oai' else b3.FIXED_S
        out[name] = {'seconds_per_study': rate, 'projected_seconds': n * rate * factor + fixed,
                     'guard_multiplier': factor,
                     'fixed_seconds': fixed, 'serial_p95_seconds_per_study': tail,
                     'peak_gpu_allocated_bytes': s['peak_gpu_allocated_bytes']}
    total = sum(s['projected_seconds'] for s in out.values())
    return {'studies': n, 'stages': out, 'total_seconds': total,
            'partner_deadline_seconds': b3.CUTOFF_S - b3.RESERVE_S,
            'margin_seconds': b3.CUTOFF_S - b3.RESERVE_S - total,
            'can_start_all': b3.can_start(0, total),
            'limitation': '58 training studies; measured rates, not a hidden-test tail guarantee. '
                          'Synchronized instrumentation adds overhead; OAI decode/forward overlap.'}


def oai_source(source):
    """Keep original B3 reader/model/windows/mirror and parallel producer/consumer."""
    source = replace(source, '    ROOT = find_test_root()', '    ROOT = str(_TIMING_SHADOW)\n'
                     '    assert ngpu == 2 and all("T4" in torch.cuda.get_device_name(g) for g in range(ngpu)), "Expected 2 x T4"')
    source = replace(source, '    return sid, vol, mask',
                     '    return sid, vol, mask, time.perf_counter() - _decode_start')
    source = replace(source, 'def _pre_one(sid):\n',
                     'def _pre_one(sid):\n    _decode_start = time.perf_counter()\n')
    source = replace(source, '    q = queue.Queue(maxsize=QUEUE_MAX);',
                     '    _steady_start = time.perf_counter()\n'
                     '    for _g in range(ngpu): torch.cuda.reset_peak_memory_stats(_g)\n'
                     '    _rows = {}\n    q = queue.Queue(maxsize=QUEUE_MAX);')
    source = replace(source, '            sid, vol, mask = item; i = pos[sid]',
                     '            sid, vol, mask, _decode = item; i = pos[sid]\n'
                     '            torch.cuda.synchronize(d)\n            _forward_start = time.perf_counter()')
    source = replace(source, '            try:\n                for a, (model, res)',
                     '            try:\n                assert mask.any(), "Failed DICOM preparation"\n'
                     '                for a, (model, res)')
    source = replace(source, '            with lock:\n                done_n[0] += 1',
                     '            torch.cuda.synchronize(d)\n'
                     '            with lock:\n'
                     '                _rows[sid] = dict(uid=sid, decode_seconds=_decode, '
                     'forward_seconds=time.perf_counter()-_forward_start)\n'
                     '                done_n[0] += 1')
    source = replace(source, 'for i, (sid, vol, mask) in enumerate(pool.imap_unordered',
                     'for i, (sid, vol, mask, _decode) in enumerate(pool.imap_unordered')
    source = replace(source, '            q.put((sid, vol, mask))',
                     '            q.put((sid, vol, mask, _decode))')
    start = source.index('    # WEIGHTED rank-mean blend across the test set')
    end = source.index('\n\nif __name__ == "__main__":', start)
    source = source[:start] + '''    assert all(np.isfinite(p).all() for p in arm_probs)
    _timing = dict(studies=[_rows[u] for u in test_ids],
                   steady_seconds=time.perf_counter()-_steady_start,
                   peak_gpu_allocated_bytes=[torch.cuda.max_memory_allocated(g) for g in range(ngpu)],
                   seconds_total=time.time()-t0)
    pathlib.Path('/kaggle/working/timing-oai.json').write_text(json.dumps(_timing))
''' + source[end:]
    # Existing OAI fallback must not turn a timing run into silent success.
    source = replace(source, '0.5, np.float32) for _ in range(A)', 'np.nan, np.float32) for _ in range(A)')
    compile(source, '<timing OAI>', 'exec')
    return source


def rad_source():
    source = inspect.getsource(b3.rad_worker)
    source = replace(source, '    decode_s = 0.0',
                     '    decode_s = 0.0\n    _rows = []\n'
                     '    torch.cuda.reset_peak_memory_stats()\n    _steady = time.perf_counter()')
    source = replace(source, '        decode_s += time.monotonic() - then',
                     '        _decode = time.monotonic() - then\n        decode_s += _decode\n'
                     '        torch.cuda.synchronize()\n        _forward = time.perf_counter()')
    source = replace(source, '        input_hashes.update(',
                     '        torch.cuda.synchronize()\n'
                     '        _rows.append(dict(uid=uid, decode_seconds=_decode, '
                     'forward_seconds=time.perf_counter()-_forward))\n        input_hashes.update(')
    source += '''
    write_json(Path('/kaggle/working/timing-rad.json'), dict(studies=_rows,
        steady_seconds=time.perf_counter()-_steady,
        peak_gpu_allocated_bytes=[torch.cuda.max_memory_allocated()],
        seconds_total=time.monotonic()-started))
'''
    compile(source, '<timing Rad>', 'exec')
    return source


# This is executed only in the private Kaggle reader child. Hooks preserve the
# original run(data, 'test', ckpts, studies=ids), all model calls and checkpoints.
CNX_HOOKS = r'''
import functools, inspect, os, threading, time
import torch, pydicom
_rows = {u: dict(uid=u, decode_seconds=0., forward_seconds=0.) for u in json.loads(ids)}
_events = pathlib.Path('/kaggle/working/timing-cnx-events.jsonl')
assert not _events.exists(), 'Refuse stale timing events'
_state = threading.local()
def _uids():
    frame = inspect.currentframe().f_back
    while frame:
        if frame.f_code.co_name in ('_decode', '_pixels', '_forward', '_uids'):
            frame = frame.f_back
            continue
        found = set()
        for value in frame.f_locals.values():
            if isinstance(value, str):
                found.update(u for u in _rows if u == value or u in value.split('/'))
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
                found.update(v for v in value if v in _rows)
        # Prefer the nearest per-study/per-batch context, never the full run IDs.
        if found and len(found) < len(_rows): return sorted(found)
        frame = frame.f_back
    return []
def _event(uids, kind, elapsed):
    if not uids: return
    line = json.dumps(dict(uids=uids, kind=kind, seconds=elapsed/len(uids)))+chr(10)  # B fix: no escape survives two string levels
    fd = os.open(str(_events), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try: os.write(fd, line.encode())
    finally: os.close(fd)
_original_decode = pydicom.dcmread
@functools.wraps(_original_decode)
def _decode(*a, **k):
    u = _uids(); started = time.perf_counter()
    try:
        result = _original_decode(*a, **k)
        return result
    finally: _event(u, 'decode_seconds', time.perf_counter()-started)
pydicom.dcmread = _decode
# Include deferred pixel decompression; dcmread alone can measure header only.
_original_pixels = pydicom.dataset.Dataset.pixel_array.fget
@functools.wraps(_original_pixels)
def _pixels(self):
    u = _uids(); started = time.perf_counter()
    try: return _original_pixels(self)
    finally: _event(u, 'decode_seconds', time.perf_counter()-started)
pydicom.dataset.Dataset.pixel_array = property(_pixels)
_original_call = torch.nn.Module._call_impl
@functools.wraps(_original_call)
def _forward(self, *a, **k):
    depth = getattr(_state, 'depth', 0)
    _state.depth = depth + 1
    u = _uids() if depth == 0 else []
    if depth == 0: torch.cuda.synchronize(); started = time.perf_counter()
    try: return _original_call(self, *a, **k)
    finally:
        if depth == 0:
            torch.cuda.synchronize()
            if not u: u = _uids()
            _event(u, 'forward_seconds', time.perf_counter()-started)
        _state.depth = depth
torch.nn.Module._call_impl = _forward
torch.cuda.reset_peak_memory_stats()
_steady = time.perf_counter()
'''


def reader_source():
    source = inspect.getsource(b3.reader_stage)
    source = replace(source, "comp = input_dir(e128.COMPETITION, 'test_series.csv')",
                     "comp = Path('/kaggle/working/timing-cohort')")
    source = replace(source, 'spec.loader.exec_module(module)\nresult = module.run',
                     CNX_HOOKS + '\nspec.loader.exec_module(module)\nresult = module.run')
    source = replace(source, 'result.to_csv(out)\n', '''result.to_csv(out)
_elapsed = time.perf_counter()-_steady
for line in _events.read_text().splitlines():
    event = json.loads(line)
    for uid in event['uids']: _rows[uid][event['kind']] += event['seconds']
assert all(r['decode_seconds'] > 0 and r['forward_seconds'] > 0 for r in _rows.values()), 'CNX timing attribution incomplete'
pathlib.Path('/kaggle/working/timing-cnx.json').write_text(json.dumps(dict(
    studies=list(_rows.values()), steady_seconds=_elapsed,
    peak_gpu_allocated_bytes=[torch.cuda.max_memory_allocated()],
    attribution='DICOM header+pixel decode; outermost model forward; batch calls amortized per study')))
''')
    compile(source, '<timing CNX stage>', 'exec')
    return source


def runtime_setup(ids):
    return '''import csv, json, pathlib, time, os
_TIMING_IDS = ''' + repr(ids) + '''
_TIMING_SHADOW = pathlib.Path('/kaggle/working/timing-cohort')
_TIMING_SHADOW.mkdir(exist_ok=False)
_candidates = [pathlib.Path('/kaggle/input/rsna-knee-abnormality-detection'), pathlib.Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
_comp = next(p for p in _candidates if (p/'train_series.csv').is_file())
with (_comp/'train_series.csv').open() as f:
    _reader = csv.DictReader(f); _fields = _reader.fieldnames
    _series = [r for r in _reader if r['StudyInstanceUID'] in _TIMING_IDS]
assert set(r['StudyInstanceUID'] for r in _series) == set(_TIMING_IDS)
with (_TIMING_SHADOW/'test_series.csv').open('x', newline='') as f:
    _writer=csv.DictWriter(f, fieldnames=_fields); _writer.writeheader(); _writer.writerows(_series)
with (_TIMING_SHADOW/'test.csv').open('x', newline='') as f:
    _writer=csv.writer(f); _writer.writerow(['StudyInstanceUID']); _writer.writerows([[u] for u in _TIMING_IDS])
os.symlink(_comp/'train_series', _TIMING_SHADOW/'test_series', target_is_directory=True)
assert not pathlib.Path('/kaggle/working/submission.csv').exists()
'''


def build(out=OUT, release=RELEASE, gold_notebook=None):
    out, release = Path(out), Path(release)
    b3.need(not out.exists(), 'Timing directory exists; refusing overwrite')
    record = json.loads((release / 'build-receipt.json').read_text(encoding='utf-8'))
    b3.need(b3.sha(HERE / 'exp133_oai_b3.py') == record['source133_sha256'] and
            b3.sha(HERE / 'exp128_oai_light.py') == record['source128_sha256'],
            'B3 helper sources differ from release receipt')
    original = (release / 'release.ipynb').read_bytes()
    nb = json.loads(original)
    meta = json.loads((release / 'kernel-metadata.json').read_text(encoding='utf-8'))
    ids = gold.gold_uids(gold_notebook or ROOT / '.local/exp130-gold58/oai_gold58/release.ipynb')
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    main = next(s for s in code if 'def main():' in s)
    tail = next(s for s in code if '_EXP133_PAYLOAD =' in s)
    # Reuse EXACT embedded B3 payload and helper sources; omit publication call.
    tree = ast.parse(tail)
    setup_nodes = tree.body[0].body
    stops = [node for node in setup_nodes if isinstance(node, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == '_EXP133_RECEIPT' for t in node.targets)]
    b3.need(len(stops) == 1, 'B3 publication assignment drift')
    # Dedent the whole block, including embedded function suites and adjacent
    # compressed string literals; per-node source extraction retains inner
    # indentation and is unnecessarily fragile.
    setup = textwrap.dedent(''.join(tail.splitlines(keepends=True)[
        setup_nodes[0].lineno - 1:stops[0].lineno - 1]))
    setup += "_EXP133_START = globals().get('_EXP133_START', time.monotonic())\n"
    setup += "_exp133.input_dir_original = _exp133.input_dir\n"
    setup += "def _timing_input(dataset, marker, input_root=pathlib.Path('/kaggle/input')):\n    return _TIMING_SHADOW if dataset == _exp133.e128.COMPETITION else _exp133.input_dir_original(dataset, marker, input_root)\n_exp133.input_dir = _timing_input\n"
    # Rad launches the original CLI; write a NEW timing copy at working path,
    # replacing ONLY its rad_worker function, and redirect HERE to it.
    original133 = (HERE / 'exp133_oai_b3.py').read_text(encoding='utf-8')
    instrumented133 = replace(original133, inspect.getsource(b3.rad_worker), rad_source())
    setup += "_exp133_write('exp133_oai_b3.py', " + b3.literal_bytes(instrumented133) + ")\n"
    setup += 'exec(_exp133.e128.unpack(' + b3.literal_bytes(reader_source()) + '), _exp133.__dict__)\n'
    setup += "_exp133.rad_stage(_EXP133_PAYLOAD, _EXP133_START, _TIMING_IDS, {}, '/kaggle/working')\n"
    setup += "_exp133.reader_stage(_EXP133_PAYLOAD, _EXP133_START, _TIMING_IDS, {}, '/kaggle/working')\n"
    setup += 'exec(_exp133.e128.unpack(' + b3.literal_bytes(inspect.getsource(projection)) + "), dict(b3=_exp133), _projection_ns := {})\n"
    setup += "_stages = {n: json.loads(pathlib.Path('/kaggle/working/timing-'+n+'.json').read_text()) for n in ('oai','rad','cnx')}\n"
    setup += "_receipt = dict(schema='exp133-timing60-v1', status='PASS', studies=58, training=0, label_reads=0, submissions=0, stages=_stages, projection_1300=_projection_ns['projection'](_stages))\n"
    setup += "assert not pathlib.Path('/kaggle/working/submission.csv').exists()\npathlib.Path('/kaggle/working/timing60.json').write_text(json.dumps(_receipt, indent=2))\n"
    cells = []
    for cell in nb['cells']:
        c = copy.deepcopy(cell)
        s = ''.join(c['source'])
        if c['cell_type'] == 'code':
            if s == main: s = runtime_setup(ids) + oai_source(s)
            elif s == tail: s = setup
            compile(s, '<timing notebook>', 'exec')
            c.update(source=s.splitlines(keepends=True), outputs=[], execution_count=None)
        cells.append(c)
    nb['cells'] = cells
    meta.update(id=KERNEL, title=KERNEL.split('/')[1], is_private=True, machine_shape='NvidiaTeslaT4')
    b3.need(meta['enable_internet'] is False and meta['enable_gpu'] is True and meta['docker_image'], 'Release metadata differs')
    out.mkdir(parents=True)
    b3.write_json(out / 'release.ipynb', nb)
    b3.write_json(out / 'kernel-metadata.json', meta)
    b3.need((release / 'release.ipynb').read_bytes() == original, 'Released notebook changed')
    receipt = dict(status='BUILT_NOT_RUN', kernel=KERNEL, studies=58,
                   released_notebook_sha256=hashlib.sha256(original).hexdigest(),
                   notebook_sha256=b3.sha(out/'release.ipynb'),
                   metadata_sha256=b3.sha(out/'kernel-metadata.json'),
                   network_calls=0, gpu_runs=0, submissions=0,
                   limitations=['Gold58 is not a hidden-test tail bound',
                     'CNX runtime hooks fail closed if per-study decode/forward attribution is incomplete',
                     'CNX steady rate conservatively includes model loading inside original run()'])
    b3.write_json(out / 'build-receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out), indent=2))


if __name__ == '__main__':
    main()


