"""H-B-164 CPU preprocessing replay v1.0.0; no notebook main or model execution.

Changelog: two complete real studies, original study_lut versus canonical reader,
then original crop/window functions on CPU. Windows pread adapter is diagnostic
only and is never embedded in the release.
"""
from __future__ import annotations
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r224q_v1123 as b
from tools import c85r7_r224_timing as timing
from tools import hb145_output_io as output_io

FASTREAD_SHA = '5a59b6f82e846bcdfd452aaee6ff518cdd9d0b04aab03691b45c3f70a5198cb0'
WORK = b.SCRATCH / 'cpu-preprocessing'


def fastread_namespace():
    source, _ = b.source()
    tree = ast.parse(source)
    encoded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_FASTREAD_B64' for t in n.targets))
    raw = base64.b64decode(encoded)
    b.old.need(hashlib.sha256(raw).hexdigest() == FASTREAD_SHA, 'reviewed embedded reader drift')
    tree = ast.parse(raw)
    # Only library declarations/constants/imports. Omit convenience wrapper loop.
    allowed = (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Assign)
    tree.body = [n for n in tree.body if isinstance(n, allowed)]
    ns = {'__name__': 'hb164_reviewed_fastread'}
    exec(compile(tree, '<reviewed-fastread-library-only>', 'exec'), ns)
    if os.name == 'nt':
        lock = threading.Lock()
        def pread(fd, size, offset):
            with lock:
                os.lseek(fd, offset, os.SEEK_SET)
                return os.read(fd, size)
        ns['_pread'] = pread
        ns['_O_FLAGS'] |= os.O_BINARY
    return ns


def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    inv = timing.inventory()
    ids = inv['studies'][:2]
    pins = {}
    with zipfile.ZipFile(timing.shared.ZIP) as archive:
        for item in inv['members']:
            if item['target'].split('/')[1] not in ids:
                continue
            raw = archive.read(item['member'])
            path = WORK / 'data' / item['target']
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                output_io.retry(path.write_bytes, raw)
            pins[item['target']] = hashlib.sha256(raw).hexdigest()
            b.old.need(timing.sha(path) == pins[item['target']], 'smoke input hash mismatch')
    ns, fr = timing.public_namespace(), fastread_namespace()
    np, torch = ns['np'], ns['torch']
    torch.set_num_threads(4)  # diagnostic only; release threading bytes unchanged
    grouped = timing.group_series(inv['series'])
    canonical = ns['_make_reader']()
    builder = fr['make_builder']('study_lut', 384, ns['SLOTS'], hdr_threads=32, px_threads=16)
    results = []
    for uid in ids:
        vol, mask = builder(uid, grouped, str(WORK/'data/train_series'))
        reference, refmask = ns['build_study'](uid, grouped, str(WORK/'data/train_series'), canonical)
        timing.validate_volume(vol, mask)
        b.old.need(np.array_equal(vol, reference) and np.array_equal(mask, refmask), 'study_lut canonical mismatch')
        win = ns['gpu_windows'](vol, mask, 94, 224, 'cpu')
        refwin = ns['eval_windows'](vol, mask, 94, 224)
        b.old.need(torch.equal(win, refwin) and torch.isfinite(win).all(), 'crop/window CPU mismatch')
        results.append(dict(study=uid, nonempty_slices=int(mask.sum()), volume_shape=list(vol.shape),
            windows_shape=list(win.shape), volume_sha256=hashlib.sha256(vol.tobytes()).hexdigest(),
            windows_sha256=hashlib.sha256(win.numpy().tobytes()).hexdigest(),
            canonical_byte_equal=True, windows_byte_equal=True))
    result = dict(status='PASS_CPU_TWO_REAL_STUDIES', version='1.0.0', studies=results,
        input_files=pins, inventory_sha256=timing.digest(inv), source_sha256=b.old.digest(b.source()[0]),
        fastread_sha256=FASTREAD_SHA, tool_sha256=timing.sha(__file__), cpu_only=True,
        limitation='Windows binary-open/locked pread adapter, sequential studies; not Linux fork or CUDA qualification. No model forward or label access.')
    output_io.journal(WORK/'receipt.json', result, WORK)
    print(json.dumps(dict(status=result['status'], studies=len(results), input_files=len(pins))))
    return result


if __name__ == '__main__':
    run()
