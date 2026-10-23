"""Builder 1.0.0: own25q3 changes only cell zero to nonfatal detection/write.

Changelog: 1.1.18-own25q3 fixes the binding own25q2 fallback failure.
All later cells, runtime archive and non-identity metadata remain frozen.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_own25q2_v1114 as reference

old = reference.old
BASE = reference.OUT
OUT = BASE.parent / 'v1.1.18-own25q3'
SCRATCH = ROOT / '.local/scratch/own25q3-v1118/build'
VERSION = '1.1.18-own25q3'
SLUG = 'taeyangg4/knee-lb-own25q3-v1'
BASE_SHA = 'a7f12d0f0f21ce8944e3c5e7fe31f38170f4891722445f6a2f59174fe4672923'
VISIBLE_IDS = reference.VISIBLE_IDS


def source():
    original = ''.join(old.read(BASE / 'kernel/release.ipynb')['cells'][0]['source'])
    body = original.split('FAST = False\n', 1)[1]
    body = old.replace(body.encode(),
        '                FAST = len(_own25q_test_ids) == 3 and set(_own25q_test_ids) == set(_own25q_visible_ids)',
        '                _own25q_eligible = len(_own25q_test_ids) == 3 and set(_own25q_test_ids) == set(_own25q_visible_ids)').decode()
    body = old.replace(body.encode(), '    if FAST:\n', '    if _own25q_eligible:\n').decode()
    body = old.replace(body.encode(),
        '        with _own25q_Path("/kaggle/working/submission.csv").open("w", newline="", encoding="utf-8") as _own25q_stream:\n',
        '        _own25q_output = _own25q_Path("/kaggle/working/submission.csv")\n'
        '        _own25q_stream = _own25q_output.open("x", newline="", encoding="utf-8")\n'
        '        _own25q_created = True\n'
        '        with _own25q_stream:\n').decode()
    body = old.replace(body.encode(), '        print("FAST_PATH_VISIBLE", flush=True)\n',
        '        FAST = True\n        print("FAST_PATH_VISIBLE", flush=True)\n').decode()
    return '''# Visible-only fallback: detection/write failures never block the frozen pipeline.
FAST = False
_own25q_created = False
_own25q_eligible = False
try:
    import os as _own25q_os
''' + ''.join('    ' + line for line in body.splitlines(True)) + '''    if not FAST:
        print("FAST_PATH_DISABLED context is not exact visible test", flush=True)
except Exception as _own25q_error:
    FAST = False
    _own25q_cleanup_error = None
    if _own25q_created:
        try:
            # Bounded Windows sharing-violation retry; never remove an existing file.
            import time as _own25q_time
            for _own25q_attempt in range(40):
                try:
                    _own25q_output.unlink(missing_ok=True)
                    break
                except OSError as _own25q_unlink_error:
                    if (isinstance(_own25q_unlink_error, PermissionError) or
                            getattr(_own25q_unlink_error, "winerror", None) in (5, 32)) and _own25q_attempt < 39:
                        _own25q_time.sleep(0.05)
                    else:
                        raise
        except Exception as _own25q_cleanup_error_caught:
            _own25q_cleanup_error = type(_own25q_cleanup_error_caught).__name__
    try:
        print("FAST_PATH_DISABLED", type(_own25q_error).__name__, str(_own25q_error),
              "cleanup=" + str(_own25q_cleanup_error), flush=True)
    except Exception:
        pass  # A broken output stream must not prevent the full pipeline either.
'''


def parent():
    old.pin(BASE / 'build-receipt.json', BASE_SHA)
    reference.validate()
    return old.read(BASE / 'kernel/release.ipynb'), old.read(BASE / 'kernel/kernel-metadata.json')


def payload():
    nb, meta = parent()
    nb['cells'][0]['source'] = source().splitlines(True)
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta


def scope(nb, meta):
    expected, expected_meta = payload()
    old.need(nb == expected and meta == expected_meta, 'forbidden notebook/metadata diff')
    before, before_meta = parent()
    restored = copy.deepcopy(nb)
    restored['cells'][0] = before['cells'][0]
    old.need(restored == before, 'changes outside cell zero')
    return reference.scope(restored, before_meta)


def full_diff(nb, meta):
    files, blob = scope(nb, meta)
    before, before_meta = parent()
    lines = [f'PASS: all 29 embedded files and entire archive IDENTICAL; SHA256 {old.digest(blob)}\n',
             'PASS: removing the unchanged guard recovers v1.1.10-own25 source byte-for-byte.\n']
    for name, data in sorted(files.items()):
        lines.append(f'IDENTICAL {name} {old.digest(data)} -> {old.digest(data)}\n')
    pairs = [('notebook-top-level', {k:v for k,v in before.items() if k != 'cells'},
              {k:v for k,v in nb.items() if k != 'cells'})]
    pairs += [(f'notebook-cell-{i}', a, b) for i, (a,b) in enumerate(zip(before['cells'], nb['cells']))]
    pairs += [('kernel-metadata.json', before_meta, meta)]
    for name, a, b in pairs:
        lines.append(f'\n{name}: ' + ('IDENTICAL\n' if a == b else 'ALLOWED_CHANGE\n'))
        lines.extend(difflib.unified_diff(old.encoded(a).decode().splitlines(True), old.encoded(b).decode().splitlines(True),
                                        fromfile='v1.1.14-own25q2/' + name, tofile=VERSION + '/' + name))
    return ''.join(lines).encode()


def prepare():
    nb, meta = payload()
    files, blob = scope(nb, meta)
    for name in ['release.ipynb', 'kernel-metadata.json']:
        old.write(ROOT / '.local/backups/own25q3-v1118/own25q2' / name, (BASE / 'kernel' / name).read_bytes())
    old.write(OUT / 'kernel/release.ipynb', old.encoded(nb))
    old.write(OUT / 'kernel/kernel-metadata.json', old.encoded(meta))
    old.write(OUT / 'full-diff-vs-own25q2.txt', full_diff(nb, meta))
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, embedded_runtime_version='1.1.10-own25',
               parent_receipt_sha256=BASE_SHA, notebook_sha256=old.digest(old.encoded(nb)), metadata_sha256=old.digest(old.encoded(meta)),
               archive_sha256=old.digest(blob), files={k:old.digest(v) for k,v in files.items()}, identical_files=29,
               builder_sha256=old.digest(Path(__file__).read_bytes()), visible_identity=reference.authenticate_visible(),
               qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    old.write(OUT / 'build-receipt.json', old.encoded(rec))
    old.write(OUT / 'CHANGELOG.md', b'# 1.1.18-own25q3 / builder 1.0.0\nCell zero catches detection/write failures, commits FAST only after CSV close, and cleans up only its own partial file with bounded retries. Existing outputs are preserved by exclusive creation. Only cell zero and id/title change from own25q2. Frozen originals untouched; backups: .local/backups/own25q3-v1118/own25q2/.\n')
    return validate()


def validate():
    nb, meta = old.read(OUT / 'kernel/release.ipynb'), old.read(OUT / 'kernel/kernel-metadata.json')
    files, blob = scope(nb, meta)
    rec = old.read(OUT / 'build-receipt.json')
    for name, key in [('release.ipynb', 'notebook_sha256'), ('kernel-metadata.json', 'metadata_sha256')]:
        old.pin(OUT / 'kernel' / name, rec[key])
    old.pin(Path(__file__), rec['builder_sha256'])
    old.need(rec['version'] == VERSION and rec['parent_receipt_sha256'] == BASE_SHA and rec['qualified_for_submission'] is False, 'receipt identity')
    old.need(rec['files'] == {k:old.digest(v) for k,v in files.items()} and rec['archive_sha256'] == old.digest(blob), 'runtime pins')
    old.need(rec['visible_identity'] == reference.authenticate_visible(), 'visible identity pins')
    old.need((OUT / 'full-diff-vs-own25q2.txt').read_bytes() == full_diff(nb, meta), 'full diff drift')
    return dict(status='PASS', runtime_files_identical=29, other_cells_identical=True,
                notebook_changed_cells=[0], metadata_changed=['id', 'title'], original_own25_source_identical=True)


def smoke():
    from tools.c85r7_own25q3_cpu_fixtures import run_smoke
    validate()
    report = run_smoke(SCRATCH)
    report.update(notebook_sha256=old.digest((OUT / 'kernel/release.ipynb').read_bytes()),
                  visible_identity=reference.authenticate_visible())
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / 'smoke.json').write_bytes(old.encoded(report))
    return report


def admit():
    validate()
    report = old.read(SCRATCH / 'smoke.json')
    old.need(report['status'] == 'PASS' and report['notebook_sha256'] == old.digest((OUT / 'kernel/release.ipynb').read_bytes()), 'smoke missing/stale/failed')
    old.need(report['cases'] and all(c['status'] == 'PASS' for c in report['cases']), 'failed smoke case')
    return dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False,
                smoke_sha256=old.digest((SCRATCH / 'smoke.json').read_bytes()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'validate', 'register', 'cpu-smoke', 'admit'])
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if args.command in ['prepare', 'admit']:
        old.need(args.dry_run, 'dry-run only')
    print(json.dumps({'prepare':prepare, 'validate':validate, 'register':validate, 'cpu-smoke':smoke, 'admit':admit}[args.command](), indent=2))


if __name__ == '__main__':
    main()
