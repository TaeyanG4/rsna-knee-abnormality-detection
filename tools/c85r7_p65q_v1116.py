"""Builder 1.0.0: mechanical own25q2 fast path on the frozen strict p65 stage.

1.0.0: copy cell zero exactly; apply the identical guard; preserve all 29 files.
CPU-only build/validation, with the authenticated v1.1.14 CPU smoke harness.
"""
from __future__ import annotations
import argparse
import ast
import copy
import difflib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_p65_v1112 as old
from tools import c85r7_own25q2_v1114 as harness

BASE = old.OUT
DONOR = BASE.parent / 'v1.1.14-own25q2'
OUT = BASE.parent / 'v1.1.16-p65q'
SCRATCH = ROOT / '.local/scratch/p65q-v1116/build'
BACKUP = ROOT / '.local/backups/p65q-v1116'
VERSION = '1.1.16-p65q'
SLUG = 'taeyangg4/knee-lb-p65q-v1'
BASE_SHA = '640c95acda92d624310d92af88fb385a4d973281afbb4861481aa38d4f29b107'
DONOR_SHA = 'a7f12d0f0f21ce8944e3c5e7fe31f38170f4891722445f6a2f59174fe4672923'
HARNESS_SHA = '5b16ad1f0053dcc027ad5ad63d3409dfc6527c53f52e2f493e01b9718008c720'
VISIBLE_IDS = harness.VISIBLE_IDS


def authenticated(stage, receipt_sha):
    old.pin(stage / 'build-receipt.json', receipt_sha)
    rec = old.read(stage / 'build-receipt.json')
    for name, key in [('release.ipynb', 'notebook_sha256'), ('kernel-metadata.json', 'metadata_sha256')]:
        old.pin(stage / 'kernel' / name, rec[key])
    return old.read(stage / 'kernel/release.ipynb'), old.read(stage / 'kernel/kernel-metadata.json')


def parent():
    result = authenticated(BASE, BASE_SHA)
    old.validate()
    return result


def donor():
    old.pin(Path(harness.__file__), HARNESS_SHA)
    return authenticated(DONOR, DONOR_SHA)[0]


def source():
    return ''.join(donor()['cells'][0]['source'])


def guard(source_text):
    return 'if not FAST:\n' + ''.join('    ' + line for line in source_text.splitlines(True))


def unguard(source_text):
    lines = source_text.splitlines(True)
    old.need(lines[0] == 'if not FAST:\n' and all(x.startswith('    ') for x in lines[1:]), 'guard shape')
    return ''.join(x[4:] for x in lines[1:])


def payload():
    nb, meta = parent()
    original = ''.join(nb['cells'][1]['source'])
    nb['cells'][1]['source'] = guard(original).splitlines(True)
    nb['cells'].insert(0, copy.deepcopy(donor()['cells'][0]))
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta


def scope(nb, meta):
    expected, expected_meta = payload()
    old.need(nb == expected, 'forbidden notebook diff')
    old.need(meta == expected_meta, 'forbidden metadata diff')
    original, _ = parent()
    donor_nb = donor()
    donor_guard = ''.join(donor_nb['cells'][2]['source'])
    old.need(guard(unguard(donor_guard)) == donor_guard, 'donor guard differs')
    old.need(nb['cells'][0] == donor_nb['cells'][0], 'donor cell zero differs')
    restored = copy.deepcopy(nb)
    restored['cells'].pop(0)
    restored['cells'][1]['source'] = unguard(''.join(restored['cells'][1]['source'])).splitlines(True)
    old.need(restored == original, 'original cells not byte identical after removing guard')
    files, blob, _ = old.unpack(restored)
    original_files, original_blob, _ = old.unpack(original)
    old.need(files == original_files and blob == original_blob and len(files) == 29, 'runtime changed')
    for name, data in files.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    return files, blob


def full_diff(nb, meta):
    files, blob = scope(nb, meta)
    before, before_meta = parent()
    lines = [f'PASS: ALL 29 embedded files and archive IDENTICAL to strict p65; archive {old.digest(blob)}\n']
    for name, data in sorted(files.items()):
        lines.append(f'IDENTICAL {name} {old.digest(data)} -> {old.digest(data)}\n')
    pairs = [('notebook-top-level', {k:v for k,v in before.items() if k != 'cells'}, {k:v for k,v in nb.items() if k != 'cells'}),
             ('inserted-cell-0', {}, nb['cells'][0])]
    pairs += [(f'parent-cell-{i}-to-candidate-cell-{i+1}', a, b) for i,(a,b) in enumerate(zip(before['cells'], nb['cells'][1:]))]
    pairs += [('kernel-metadata.json', before_meta, meta)]
    for name, a, b in pairs:
        lines.append('\n' + name + ': ' + ('IDENTICAL\n' if a == b else 'ALLOWED_CHANGE\n'))
        lines.extend(difflib.unified_diff(old.encoded(a).decode().splitlines(True), old.encoded(b).decode().splitlines(True),
                                        fromfile='v1.1.12-p65/' + name, tofile=VERSION + '/' + name))
    return ''.join(lines).encode()


def cell_zero_diff(nb):
    before = old.encoded(donor()['cells'][0])
    after = old.encoded(nb['cells'][0])
    old.need(before == after, 'cell zero diff is nonempty')
    return (f'IDENTICAL whole cell object: {old.digest(before)} -> {old.digest(after)}\n'
            f'IDENTICAL UTF-8 source: {old.digest(source().encode())}\n'
            'Unified diff vs v1.1.14-own25q2 cell 0: EMPTY (0 changed bytes).\n').encode()


def registration(files):
    return dict(version=VERSION, kernel_id=SLUG, parent_receipt_sha256=BASE_SHA, donor_receipt_sha256=DONOR_SHA,
                registered_change='Exact own25q2 cell zero + identical if not FAST guard + kernel id/title only',
                formula=old.FORMULA, files={k:old.digest(v) for k,v in files.items()},
                exposure='EXPOSED', clean_oof=False, qualified_for_submission=False)


def prepare():
    nb, meta = payload()
    files, blob = scope(nb, meta)
    for stage in [BASE, DONOR]:
        for name in ['build-receipt.json', 'kernel/release.ipynb', 'kernel/kernel-metadata.json']:
            old.write(BACKUP / stage.name / name, (stage / name).read_bytes())
    for kind in ['parent-runtime', 'candidate-runtime']:
        for name, data in files.items():
            old.write(SCRATCH / kind / name, data)
    old.write(OUT / 'kernel/release.ipynb', old.encoded(nb))
    old.write(OUT / 'kernel/kernel-metadata.json', old.encoded(meta))
    old.write(OUT / 'full-diff-vs-p65.txt', full_diff(nb, meta))
    old.write(OUT / 'cell0-diff-vs-own25q2.txt', cell_zero_diff(nb))
    old.write(OUT / 'registration.json', old.encoded(registration(files)))
    old.write(OUT / 'CHANGELOG.md', b'# 1.1.16-p65q / builder 1.0.0\nMechanical own25q2 cell-zero and guard transplant onto strict p65; id/title updated. All 29 embedded files unchanged, embedded runtime stays 1.1.12-p65. Frozen sources untouched; backups: .local/backups/p65q-v1116/.\n')
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, embedded_runtime_version='1.1.12-p65',
               parent_receipt_sha256=BASE_SHA, donor_receipt_sha256=DONOR_SHA,
               notebook_sha256=old.digest(old.encoded(nb)), metadata_sha256=old.digest(old.encoded(meta)),
               archive_sha256=old.digest(blob), files={k:old.digest(v) for k,v in files.items()}, identical_files=29,
               builder_sha256=old.digest(Path(__file__).read_bytes()), harness_sha256=HARNESS_SHA,
               visible_identity=harness.authenticate_visible(), qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    old.write(OUT / 'build-receipt.json', old.encoded(rec))
    return validate()


def validate():
    nb, meta = old.read(OUT / 'kernel/release.ipynb'), old.read(OUT / 'kernel/kernel-metadata.json')
    files, blob = scope(nb, meta)
    rec = old.read(OUT / 'build-receipt.json')
    for name, key in [('release.ipynb', 'notebook_sha256'), ('kernel-metadata.json', 'metadata_sha256')]:
        old.pin(OUT / 'kernel' / name, rec[key])
    old.pin(Path(__file__), rec['builder_sha256'])
    old.need(rec['version'] == VERSION and rec['parent_receipt_sha256'] == BASE_SHA and rec['donor_receipt_sha256'] == DONOR_SHA, 'receipt identity')
    old.need(rec['qualified_for_submission'] is False and rec['exposure'] == 'EXPOSED' and rec['clean_oof'] is False, 'qualification/exposure')
    old.need(rec['files'] == {k:old.digest(v) for k,v in files.items()} and rec['archive_sha256'] == old.digest(blob), 'runtime pins')
    old.need(rec['harness_sha256'] == HARNESS_SHA and rec['visible_identity'] == harness.authenticate_visible(), 'harness/visible pins')
    old.need(old.read(OUT / 'registration.json') == registration(files), 'registration drift')
    old.need((OUT / 'full-diff-vs-p65.txt').read_bytes() == full_diff(nb, meta), 'full diff drift')
    old.need((OUT / 'cell0-diff-vs-own25q2.txt').read_bytes() == cell_zero_diff(nb), 'cell zero diff drift')
    return dict(status='PASS', runtime_files_identical=29, archive_identical=True, cell0_identical_to_own25q2=True,
                guards_identical=True, original_cells_identical_after_unguard=True, metadata_changed=['id','title'])


def run_case(area, **kwargs):
    # Reuse the reviewed CPU-only fixture; execute this candidate's actual cell 0.
    with patch.object(harness, 'source', lambda: ''.join(old.read(OUT / 'kernel/release.ipynb')['cells'][0]['source'])), \
         patch.object(harness, 'OUT', OUT), patch.object(harness, 'SCRATCH', SCRATCH):
        return harness.run_case(area, **kwargs)


def smoke():
    with patch.object(harness, 'source', lambda: ''.join(old.read(OUT / 'kernel/release.ipynb')['cells'][0]['source'])), \
         patch.object(harness, 'OUT', OUT), patch.object(harness, 'SCRATCH', SCRATCH), patch.object(harness, 'validate', validate):
        return harness.smoke()


def replay():
    validate()
    # Byte identity established above; run the frozen p65 arithmetic on real visible predictions.
    with patch.object(old, 'SCRATCH', SCRATCH / 'arithmetic'):
        return old.replay()


def admit():
    validate()
    report = old.read(SCRATCH / 'smoke.json')
    old.need(report['status'] == 'PASS' and report['notebook_sha256'] == old.digest((OUT / 'kernel/release.ipynb').read_bytes()), 'smoke missing/stale')
    old.need(len(report['cases']) == 33 and all(c['status'] == 'PASS' for c in report['cases']), 'smoke coverage')
    replay_result = old.read(SCRATCH / 'arithmetic/replay.json')
    old.need(replay_result['status'] == 'PASS' and replay_result['independent_pandas_exact'], 'real replay required')
    return dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False,
                smoke_sha256=old.digest((SCRATCH / 'smoke.json').read_bytes()), replay_sha256=old.digest((SCRATCH / 'arithmetic/replay.json').read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'register', 'cpu-smoke', 'cpu-replay', 'admit'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if args.command == 'admit':
        old.need(args.dry_run, 'dry-run only; no launch authority')
    action = {'prepare':prepare, 'validate':validate, 'register':validate, 'cpu-smoke':smoke, 'cpu-replay':replay, 'admit':admit}[args.command]
    print(json.dumps(action(), indent=2))


if __name__ == '__main__':
    main()
