"""Builder 1.0.0: mechanically transplant own25q2 fast path onto strict own10.

Only the notebook cell zero, identical guard/indent, and id/title change.
No runtime, launch, process, import, or arithmetic changes; CPU operations only.
"""
from __future__ import annotations
import argparse
import copy
import difflib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_own10_v1113 as old
from tools import c85r7_own25q2_v1114 as reference

BASE = old.OUT
REF = reference.OUT
OUT = BASE.parent / 'v1.1.17-own10q'
SCRATCH = ROOT / '.local/scratch/astra-x1/own10q'
BACKUP = ROOT / '.local/backups/astra-x1/own10q'
VERSION = '1.1.17-own10q'
SLUG = 'taeyangg4/knee-lb-own10q-v1'
BASE_SHA = 'be72a247147b0b2659250b6dde502e9ce2a0df63652fe9935291cf9656b73824'
REF_SHA = 'a7f12d0f0f21ce8944e3c5e7fe31f38170f4891722445f6a2f59174fe4672923'
VISIBLE_IDS = reference.VISIBLE_IDS


def parents():
    old.pin(BASE / 'build-receipt.json', BASE_SHA)
    old.pin(REF / 'build-receipt.json', REF_SHA)
    old.validate()
    reference.validate()
    return (old.read(BASE / 'kernel/release.ipynb'),
            old.read(BASE / 'kernel/kernel-metadata.json'),
            old.read(REF / 'kernel/release.ipynb'))


def guard(text):
    return 'if not FAST:\n' + ''.join('    ' + line for line in text.splitlines(True))


def source():
    return ''.join(old.read(OUT / 'kernel/release.ipynb')['cells'][0]['source'])


def payload():
    nb, meta, ref = parents()
    old.need(len(nb['cells']) == 2 and nb['cells'][1]['cell_type'] == 'code', 'parent cell layout')
    # Prove this is the exact guard transformation used in the reviewed source.
    ref_original = old.read(reference.parent_builder.BASE / 'kernel/release.ipynb')
    old.need(''.join(ref['cells'][2]['source']) == guard(''.join(ref_original['cells'][1]['source'])), 'reference guard drift')
    nb['cells'][1]['source'] = guard(''.join(nb['cells'][1]['source'])).splitlines(True)
    nb['cells'].insert(0, copy.deepcopy(ref['cells'][0]))
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta


def scope(nb, meta):
    expected, expected_meta = payload()
    old.need(nb == expected, 'forbidden notebook diff')
    old.need(meta == expected_meta, 'forbidden metadata diff')
    before, _, ref = parents()
    old.need(old.encoded(nb['cells'][0]) == old.encoded(ref['cells'][0]), 'cell zero differs from reference')
    old.need(''.join(nb['cells'][0]['source']).encode() == ''.join(ref['cells'][0]['source']).encode(), 'cell zero source bytes differ')
    restored = copy.deepcopy(nb)
    restored['cells'].pop(0)
    lines = ''.join(restored['cells'][1]['source']).splitlines(True)
    old.need(lines[0] == 'if not FAST:\n' and all(line.startswith('    ') for line in lines[1:]), 'guard shape')
    restored['cells'][1]['source'] = ''.join(line[4:] for line in lines[1:]).splitlines(True)
    old.need(restored == before, 'later cells changed beyond exact guard')
    files, blob, _ = old.unpack(restored)
    base_files, base_blob, _ = old.unpack(before)
    old.need(files == base_files and blob == base_blob and len(files) == 29, 'runtime bytes changed')
    return files, blob


def full_diff(nb, meta):
    files, blob = scope(nb, meta)
    before, before_meta, _ = parents()
    lines = [f'PASS: ALL 29 embedded files and archive byte-identical to v1.1.13-own10; archive {old.digest(blob)}\n']
    for name, data in sorted(files.items()):
        lines.append(f'IDENTICAL {name} {old.digest(data)} -> {old.digest(data)}\n')
    pairs = [('notebook-top-level', {k:v for k,v in before.items() if k != 'cells'}, {k:v for k,v in nb.items() if k != 'cells'}),
             ('new-cell-0', {}, nb['cells'][0]),
             ('parent-cell-0-to-candidate-cell-1', before['cells'][0], nb['cells'][1]),
             ('parent-cell-1-to-candidate-cell-2', before['cells'][1], nb['cells'][2]),
             ('kernel-metadata.json', before_meta, meta)]
    for name, left, right in pairs:
        lines.append(f'\n{name}: ' + ('IDENTICAL\n' if left == right else 'ALLOWED_CHANGE\n'))
        lines.extend(difflib.unified_diff(old.encoded(left).decode().splitlines(True), old.encoded(right).decode().splitlines(True),
                                        fromfile='v1.1.13-own10/' + name, tofile=VERSION + '/' + name))
    return ''.join(lines).encode()


def cell0_proof(nb):
    ref = old.read(REF / 'kernel/release.ipynb')['cells'][0]
    actual = nb['cells'][0]
    left, right = ''.join(ref['source']).encode(), ''.join(actual['source']).encode()
    old.need(left == right and old.encoded(ref) == old.encoded(actual), 'cell zero bytes')
    return old.encoded(dict(status='IDENTICAL', source_bytes=len(left), source_sha256=old.digest(left),
                            serialized_cell_sha256=old.digest(old.encoded(ref)), unified_diff=[],
                            reference=str(REF.relative_to(ROOT)), candidate=VERSION))


def prepare():
    nb, meta = payload()
    files, blob = scope(nb, meta)
    for label, parent in [('own10', BASE), ('own25q2', REF)]:
        for name in ['build-receipt.json', 'kernel/release.ipynb', 'kernel/kernel-metadata.json']:
            old.write(BACKUP / label / name, (parent / name).read_bytes())
    # Extract every embedded file from both actual notebooks into the one fixed scratch.
    before = old.read(BASE / 'kernel/release.ipynb')
    for label, entries in [('parent-runtime', old.unpack(before)[0]), ('candidate-runtime', files)]:
        for name, data in entries.items():
            old.write(SCRATCH / label / name, data)
    old.write(OUT / 'kernel/release.ipynb', old.encoded(nb))
    old.write(OUT / 'kernel/kernel-metadata.json', old.encoded(meta))
    old.write(OUT / 'full-diff-vs-own10.txt', full_diff(nb, meta))
    old.write(OUT / 'cell0-diff-vs-own25q2.json', cell0_proof(nb))
    receipt = dict(status='PREPARED_CPU_ONLY', version=VERSION, embedded_runtime_version=old.VERSION,
                   parent_receipt_sha256=BASE_SHA, fast_path_receipt_sha256=REF_SHA,
                   notebook_sha256=old.digest(old.encoded(nb)), metadata_sha256=old.digest(old.encoded(meta)),
                   archive_sha256=old.digest(blob), files={k:old.digest(v) for k,v in files.items()}, identical_files=29,
                   builder_sha256=old.digest(Path(__file__).read_bytes()), visible_identity=reference.authenticate_visible(),
                   qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    old.write(OUT / 'build-receipt.json', old.encoded(receipt))
    old.write(OUT / 'CHANGELOG.md', b'# 1.1.17-own10q\nMechanical copy of v1.1.14-own25q2 cell 0 and guard onto strict v1.1.13-own10; only id/title additionally changed. All 29 embedded files and recovered later cells unchanged. Original/frozen stages untouched. Backups: .local/backups/astra-x1/own10q.\n')
    return validate()


def validate():
    nb = old.read(OUT / 'kernel/release.ipynb')
    meta = old.read(OUT / 'kernel/kernel-metadata.json')
    files, blob = scope(nb, meta)
    rec = old.read(OUT / 'build-receipt.json')
    for name, key in [('release.ipynb', 'notebook_sha256'), ('kernel-metadata.json', 'metadata_sha256')]:
        old.pin(OUT / 'kernel' / name, rec[key])
    old.pin(Path(__file__), rec['builder_sha256'])
    old.need(rec['status'] == 'PREPARED_CPU_ONLY' and rec['version'] == VERSION and rec['embedded_runtime_version'] == old.VERSION and
             rec['parent_receipt_sha256'] == BASE_SHA and rec['fast_path_receipt_sha256'] == REF_SHA and
             rec['qualified_for_submission'] is False and rec['exposure'] == 'EXPOSED' and rec['clean_oof'] is False, 'receipt identity')
    old.need(rec['files'] == {k:old.digest(v) for k,v in files.items()} and rec['archive_sha256'] == old.digest(blob) and rec['identical_files'] == 29, 'runtime pins')
    old.need(rec['visible_identity'] == reference.authenticate_visible(), 'visible identity pins')
    old.need((OUT / 'full-diff-vs-own10.txt').read_bytes() == full_diff(nb, meta), 'full diff drift')
    old.need((OUT / 'cell0-diff-vs-own25q2.json').read_bytes() == cell0_proof(nb), 'cell zero proof drift')
    return dict(status='PASS', runtime_files_identical=29, cell0_reference_identical=True,
                original_cells_recovered_byte_identical=True, guards_reference_identical=True, metadata_changed=['id', 'title'])


def run_case(area, **kwargs):
    # Reuse reviewed CPU-only fixture; production candidate sources stay unchanged.
    with patch.object(reference, 'OUT', OUT), patch.object(reference, 'SCRATCH', SCRATCH), patch.object(reference, 'source', source):
        return reference.run_case(area, **kwargs)


def smoke():
    # Same 33 cases, same full-notebook FAST=True and original-first-import FAST=False checks.
    checked = validate()
    with patch.object(reference, 'OUT', OUT), patch.object(reference, 'SCRATCH', SCRATCH), \
         patch.object(reference, 'source', source), patch.object(reference, 'validate', lambda: checked):
        return reference.smoke()


def admit():
    validate()
    rec = old.read(SCRATCH / 'smoke.json')
    old.need(rec['status'] == 'PASS' and rec['notebook_sha256'] == old.digest((OUT / 'kernel/release.ipynb').read_bytes()), 'smoke missing/stale')
    old.need(len(rec['cases']) == 33 and all(c['status'] == 'PASS' for c in rec['cases']), 'incomplete smoke')
    return dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False,
                smoke_sha256=old.digest((SCRATCH / 'smoke.json').read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'register', 'cpu-smoke', 'admit'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if args.command == 'admit':
        old.need(args.dry_run, 'dry-run only; independent review required')
    print(json.dumps({'prepare':prepare, 'validate':validate, 'register':validate, 'cpu-smoke':smoke, 'admit':admit}[args.command](), indent=2))


if __name__ == '__main__':
    main()
