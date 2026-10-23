"""1.0.0: single own25q notebook-only fast path; all runtime bytes frozen."""
from __future__ import annotations
import argparse
import ast
import copy
import csv
import difflib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_own25_v1110 as old

BASE = old.OUT
OUT = BASE.parent / 'v1.1.11-own25q'
SCRATCH = ROOT / '.local/scratch/own25q-v1111/build'
VERSION = '1.1.11-own25q'
SLUG = 'taeyangg4/knee-lb-own25q-v1'
BASE_RECEIPT = 'eeaf41b9151ee6c630f9b5fab34cab70916f6b9095c76c1209b17b4fdadba812'
HEADER = ['StudyInstanceUID', 'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
          'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FAST_SOURCE = '''# Visible-only quota fallback; hidden reruns retain the frozen own25 cell.
import os as _own25q_os
FAST = False
if not _own25q_os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    import csv as _own25q_csv
    from pathlib import Path as _own25q_Path
    # Same competition_root / _mount resolution as v1.1.0, including override.
    if _own25q_os.environ.get("C85_COMPETITION"):
        _own25q_comp = _own25q_Path(_own25q_os.environ["C85_COMPETITION"])
    else:
        _own25q_name = "rsna-knee-abnormality-detection"
        _own25q_root = _own25q_Path("/kaggle/input")
        _own25q_candidates = [_own25q_root / _own25q_name,
            _own25q_root / "competitions" / _own25q_name,
            _own25q_root / "datasets" / _own25q_name]
        _own25q_candidates.extend(_own25q_root.glob("datasets/*/" + _own25q_name))
        _own25q_candidates.extend(_own25q_root.glob("datasets/*/" + _own25q_name + "/*"))
        _own25q_candidates.extend(_own25q_root.glob(_own25q_name + "/*"))
        _own25q_matches = list(dict.fromkeys(p for p in _own25q_candidates if (p / "test.csv").is_file()))
        if len(_own25q_matches) != 1:
            raise ValueError("dataset mount absent or ambiguous: " + _own25q_name)
        _own25q_comp = _own25q_matches[0]
    _own25q_ids = sorted(p.name for p in (_own25q_comp / "test_series").iterdir() if p.is_dir())
    FAST = len(_own25q_ids) <= 3
    if FAST:
        with _own25q_Path("/kaggle/working/submission.csv").open("w", newline="", encoding="utf-8") as _own25q_stream:
            _own25q_writer = _own25q_csv.writer(_own25q_stream)
            _own25q_writer.writerow(["StudyInstanceUID", "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"])
            _own25q_writer.writerows([uid, *([0.5] * 12)] for uid in _own25q_ids)
        print("FAST_PATH_VISIBLE", flush=True)
'''

def parent():
    old.pin(BASE / 'build-receipt.json', BASE_RECEIPT)
    old.validate()
    return old.read(BASE / 'kernel/release.ipynb'), old.read(BASE / 'kernel/kernel-metadata.json')

def payload():
    nb, meta = parent()
    nb = copy.deepcopy(nb)
    original = ''.join(nb['cells'][1]['source'])
    nb['cells'][1]['source'] = ('if not FAST:\n' + ''.join('    ' + line for line in original.splitlines(True))).splitlines(True)
    nb['cells'].insert(0, dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=FAST_SOURCE.splitlines(True)))
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta

def scope(nb, meta):
    expected_nb, expected_meta = payload()
    old.need(nb == expected_nb, 'forbidden notebook diff')
    old.need(meta == expected_meta, 'forbidden metadata diff')
    original, _ = parent()
    guarded = ''.join(nb['cells'][2]['source'])
    lines = guarded.splitlines(True)
    old.need(lines[0] == 'if not FAST:\n' and all(x.startswith('    ') for x in lines[1:]), 'guard shape')
    recovered = ''.join(x[4:] for x in lines[1:])
    old.need(recovered == ''.join(original['cells'][1]['source']), 'original source not byte identical')
    old.need(ast.dump(ast.parse(guarded).body[0].body[0]) == ast.dump(ast.parse(recovered).body[0]), 'first statement changed')
    restored = copy.deepcopy(original)
    restored['cells'][1]['source'] = recovered.splitlines(True)
    files, blob, _ = old.unpack(restored)
    base_files, base_blob, _ = old.unpack(original)
    old.need(files == base_files and blob == base_blob and len(files) == 29, 'runtime bytes changed')
    return files, blob

def full_diff(nb, meta):
    files, _ = scope(nb, meta)
    parent_nb, parent_meta = parent()
    lines = ['PASS: all 29 embedded files and entire archive IDENTICAL to v1.1.10-own25.\n']
    for name, data in sorted(files.items()):
        lines.append(f'IDENTICAL {name} {old.digest(data)} -> {old.digest(data)}\n')
    for name, before, after in [('notebook-top-level', {k:v for k,v in parent_nb.items() if k != 'cells'}, {k:v for k,v in nb.items() if k != 'cells'}),
                                ('new-first-cell', {}, nb['cells'][0]),
                                ('markdown-cell', parent_nb['cells'][0], nb['cells'][1]),
                                ('original-code-cell', parent_nb['cells'][1], nb['cells'][2]),
                                ('kernel-metadata.json', parent_meta, meta)]:
        lines.append(f'\n{name}: ' + ('IDENTICAL\n' if before == after else 'ALLOWED_CHANGE\n'))
        lines.extend(difflib.unified_diff(old.encoded(before).decode().splitlines(True), old.encoded(after).decode().splitlines(True), fromfile='own25/'+name, tofile='own25q/'+name))
    return ''.join(lines).encode()

def prepare():
    nb, meta = payload()
    files, blob = scope(nb, meta)
    for name in ('release.ipynb', 'kernel-metadata.json'):
        old.write(ROOT / '.local/backups/own25q-v1111/own25' / name, (BASE / 'kernel' / name).read_bytes())
    old.write(OUT / 'kernel/release.ipynb', old.encoded(nb))
    old.write(OUT / 'kernel/kernel-metadata.json', old.encoded(meta))
    old.write(OUT / 'full-diff-vs-own25.txt', full_diff(nb, meta))
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, embedded_runtime_version='1.1.10-own25',
        base_receipt_sha256=BASE_RECEIPT, notebook_sha256=old.digest(old.encoded(nb)), metadata_sha256=old.digest(old.encoded(meta)),
        archive_sha256=old.digest(blob), files={k:old.digest(v) for k,v in files.items()}, identical_files=29,
        builder_sha256=old.digest(Path(__file__).read_bytes()), qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    old.write(OUT / 'build-receipt.json', old.encoded(rec))
    return validate()

def validate():
    nb, meta = old.read(OUT / 'kernel/release.ipynb'), old.read(OUT / 'kernel/kernel-metadata.json')
    files, blob = scope(nb, meta)
    rec = old.read(OUT / 'build-receipt.json')
    for name, key in [('release.ipynb','notebook_sha256'), ('kernel-metadata.json','metadata_sha256')]:
        old.pin(OUT / 'kernel' / name, rec[key])
    old.pin(Path(__file__), rec['builder_sha256'])
    old.need(rec['files'] == {k:old.digest(v) for k,v in files.items()} and rec['archive_sha256'] == old.digest(blob), 'receipt runtime pins')
    old.need((OUT / 'full-diff-vs-own25.txt').read_bytes() == full_diff(nb, meta), 'diff receipt mismatch')
    return dict(status='PASS', embedded_files_identical=29, original_cell_recovered_byte_identical=True, archive_sha256=old.digest(blob))

def smoke():
    # Test-side path shim only: production notebook is executed without edits.
    import contextlib
    import io
    import time
    from unittest.mock import patch
    import nbclient
    import nbformat
    validate()
    nb = old.read(OUT / 'kernel/release.ipynb')
    old.pin(old.VISIBLE / 'c85r7-receipt.json', old.VISIBLE_SHA)
    receipt = old.read(old.VISIBLE / 'c85r7-receipt.json')
    old.pin(old.VISIBLE / 'submission.csv', receipt['submission_sha256'])
    with (old.VISIBLE / 'submission.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        old.need(reader.fieldnames == HEADER, 'real header mismatch')
        real_ids = [row['StudyInstanceUID'] for row in reader]
    old.need(len(set(real_ids)) == 3, 'real visible cohort')
    results = []
    cases = [('flat3',3,'flat','',False), ('nested3',3,'nested','',False), ('real3',3,'flat','',True),
             ('four',4,'flat','',False), ('rerun3',3,'nested','1',False), ('rerun4',4,'flat','1',False),
             ('override3',3,'override','',False), ('nonempty_env',3,'flat','0',False)]
    for name, n, layout, rerun, real in cases:
        area = SCRATCH / name
        root = area / 'input'
        comp = root / ('competitions' if layout == 'nested' else '') / 'rsna-knee-abnormality-detection'
        ids = real_ids if real else [f'fake-{i}' for i in range(n)]
        for uid in ids: (comp / 'test_series' / uid).mkdir(parents=True, exist_ok=True)
        (comp / 'test.csv').write_text('StudyInstanceUID\n', encoding='utf-8')
        (comp / 'test_series' / 'ignored-file').write_text('not a study', encoding='utf-8')
        work = area / 'working'
        work.mkdir(parents=True, exist_ok=True)
        # Path factory installed only in this test kernel; no runtime source edits.
        setup = f'''import pathlib, os, sys
_test_path = pathlib.Path
def _mapped_path(value, *parts):
    value = str(value)
    if value == '/kaggle/input': value = {str(root.resolve())!r}
    elif value == '/kaggle/working/submission.csv': value = {str((work/'submission.csv').resolve())!r}
    return _test_path(value, *parts)
pathlib.Path = _mapped_path
os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
os.environ.pop('C85_COMPETITION', None)
'''
        if rerun: setup += f"os.environ['KAGGLE_IS_COMPETITION_RERUN'] = {rerun!r}\n"
        if layout == 'override': setup += f"os.environ['C85_COMPETITION'] = {str(comp.resolve())!r}\n"
        want_fast = not rerun and n <= 3
        started = time.monotonic()
        if want_fast:
            # A real Jupyter kernel executes every cell and returns normally.
            test_nb = nbformat.reads(json.dumps(nb), as_version=4)
            test_nb.cells.insert(0, nbformat.v4.new_code_cell(setup))
            test_nb.cells.append(nbformat.v4.new_code_cell("assert FAST is True\nassert 'BLOB' not in globals()\nassert 'exp169_release_c85r7codec' not in sys.modules\nassert 'torch' not in sys.modules\npathlib.Path = _test_path\nprint('NOTEBOOK_COMPLETED')"))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', IPYTHONDIR=str((SCRATCH/'ipython').resolve()), JUPYTER_RUNTIME_DIR=str((SCRATCH/'jupyter').resolve()))
            executed = nbclient.NotebookClient(test_nb, timeout=60, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute(env=env)
            nbformat.write(executed, area / 'executed.ipynb')
            outputs = ''.join(o.get('text','') for c in executed.cells for o in c.get('outputs',[]))
            old.need('FAST_PATH_VISIBLE' in outputs and 'NOTEBOOK_COMPLETED' in outputs, 'notebook incomplete')
            with (work/'submission.csv').open(newline='', encoding='utf-8') as f: rows = list(csv.reader(f))
            old.need(rows[0] == HEADER and [r[0] for r in rows[1:]] == sorted(ids), 'CSV schema or cohort')
            old.need(all(len(r) == 13 and all(float(x) == .5 for x in r[1:]) for r in rows[1:]), 'CSV predictions')
        else:
            ns = {}
            import pathlib
            real_path = pathlib.Path
            with patch.dict(os.environ, {}, clear=False):
                try:
                    exec(setup, ns)
                    exec(FAST_SOURCE, ns)
                    old.need(ns['FAST'] is False, 'hidden bypass failed')
                    # Execute exact first original statement, stop via test AST truncation.
                    tree = ast.parse(''.join(nb['cells'][2]['source']))
                    tree.body[0].body = tree.body[0].body[:1]
                    exec(compile(tree, '<original-first-statement>', 'exec'), ns)
                    old.need(all(k in ns for k in ('base64','hashlib','io','sys','zipfile')), 'original cell not entered')
                    old.need('BLOB' not in ns, 'test exceeded first statement')
                finally: pathlib.Path = real_path
        results.append(dict(case=name, studies=n, FAST=bool(want_fast), status='PASS', seconds=round(time.monotonic()-started,3), complete_notebook=bool(want_fast), original_first_statement=not want_fast))
    report = dict(status='PASS', cases=results, notebook_sha256=old.digest((OUT/'kernel/release.ipynb').read_bytes()),
        real_visible_receipt_sha256=old.VISIBLE_SHA, real_visible_csv_sha256=receipt['submission_sha256'],
        limitation='Real visible IDs from authenticated completed output; CPU notebook/CSV smoke only. No DICOM inference, CUDA, pip, or Kaggle calls.')
    (SCRATCH/'smoke.json').write_bytes(old.encoded(report))
    return report

def admit():
    validate()
    report = old.read(SCRATCH/'smoke.json')
    old.need(report['status'] == 'PASS' and report['notebook_sha256'] == old.digest((OUT/'kernel/release.ipynb').read_bytes()), 'smoke required')
    return dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False, smoke_sha256=old.digest((SCRATCH/'smoke.json').read_bytes()))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','validate','register','cpu-smoke','admit'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if args.command == 'admit': old.need(args.dry_run, 'dry-run only')
    action = {'prepare':prepare,'validate':validate,'register':validate,'cpu-smoke':smoke,'admit':admit}[args.command]
    print(json.dumps(action(), indent=2))

if __name__ == '__main__': main()
