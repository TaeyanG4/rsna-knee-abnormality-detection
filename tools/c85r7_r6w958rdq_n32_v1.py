"""H-B-190 N32 runtime-qualification kernel builder 1.0.1 (diagnostic slug; never submit).

Changelog 1.0.1: projection gate = start-up + whole worker time per study x 1300 (independent review: DataLoader
prefetch during model load made the 1.0.0 central figure low); 1.0.0 stress-kernel (never pushed) moved to
.local/backups/hb190-build/stress-kernel-v1.0.0/.

Changelog 1.0.0 (claude-opus-b2, 2026-10-17): builds stress-kernel/ next to the H-B-190 stage from the bound
gate-copy kernel bytes. Changes against that kernel:
- cell 0 = the diagnostic setup (FAST off, harness) from the pinned tools/c85r7_r6w958rdq_n32_v1_cell0.txt;
- cell 1 = a diagnostic note;
- cell 2 = the candidate bootstrap with exactly one line changed: run_release -> _hb190_n32_run(release, runtime).
The embedded archive (BLOB/PINS, 31 members) is byte-identical. Metadata differs from the candidate's only in
id/title. Why it is not the old C85R7 stress_notebook: R224 (raptor224_run.py find_test_root) ignores
C85_COMPETITION, so the full path is qualified on the real public test (part A) and the reader is timed on
32 train studies (part B).
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r6w958rdq_v1140 as b

VERSION = '1.0.1'
SLUG = 'taeyangg4/knee-lb-r6w958rdq-n32-v1'
GATE_KERNEL = ROOT / '.local/scratch/slot-b/hb190/gate/gate-copy-v145/kernel'
OUT = b.OUT / 'stress-kernel'
CELL0 = ROOT / 'tools/c85r7_r6w958rdq_n32_v1_cell0.txt'
ANCHOR = "    release.run_release(runtime, Path('/kaggle/working'))\n"
REPLACEMENT = '    _hb190_n32_run(release, runtime)\n'
NOTE = ('# H-B-190 N32 diagnostic (never submit)\n'
        'Part A runs the exact v1.1.40-r6w958rdq candidate path with FAST disabled on the public test; part B runs '
        'the same embedded reader phase on 32 label-free train studies; hb190-n32-report.json holds per-stage '
        'seconds, assertions and the hidden-runtime projection. No labels or reports are read.\n')
need = b.need


def candidate():
    """Bound gate-copy kernel == builder stage kernel == the validated candidate (aggregate pinned)."""
    for name in ('release.ipynb', 'kernel-metadata.json'):
        need((GATE_KERNEL / name).read_bytes() == (b.OUT / 'kernel' / name).read_bytes(), 'gate copy differs: ' + name)
    need(b.validate()['kernel_sha256'] == '59f1d2d33f8888c1dbef385d0f95247b24876a48cbea3d1e38c661231b6c9e2b',
         'candidate kernel aggregate')
    return b.old.read(GATE_KERNEL / 'release.ipynb'), b.old.read(GATE_KERNEL / 'kernel-metadata.json')


def payload():
    nb, meta = candidate()
    stress = copy.deepcopy(nb)
    need(len(nb['cells']) == 3 and nb['cells'][1]['cell_type'] == 'markdown', 'candidate cell layout')
    cell0 = CELL0.read_text(encoding='utf-8')
    compile(cell0, '<hb190-n32-cell0>', 'exec')
    stress['cells'][0]['source'] = cell0.splitlines(True)
    stress['cells'][1]['source'] = NOTE.splitlines(True)
    code = ''.join(nb['cells'][2]['source'])
    need(code.count(ANCHOR) == 1, 'run_release anchor')
    stress['cells'][2]['source'] = code.replace(ANCHOR, REPLACEMENT).splitlines(True)
    for cell in stress['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<hb190-n32>', 'exec')
    smeta = dict(meta)
    smeta.update(id=SLUG, title=SLUG.split('/')[1])
    return stress, smeta


def prepare():
    stress, meta = payload()
    for name, value in (('release.ipynb', stress), ('kernel-metadata.json', meta)):
        b.old.write(OUT / name, b.old.encoded(value))  # immutable: identical rerun only
    return validate()


def validate():
    stress, meta = payload()
    for name, value in (('release.ipynb', stress), ('kernel-metadata.json', meta)):
        need((OUT / name).read_bytes() == b.old.encoded(value), 'stress kernel drift: ' + name)
    nb, cmeta = candidate()
    files, blob, _ = b.old.unpack(b.fast.restored(nb))
    sfiles, sblob = b.old.unpack(b.fast.restored(stress))[:2]
    need(sfiles == files and sblob == blob, 'stress kernel embedded archive differs from the candidate')
    need({k for k in meta if meta[k] != cmeta[k]} == {'id', 'title'} and meta['id'] != cmeta['id'], 'metadata scope')
    return dict(status='PREPARED_DIAGNOSTIC_NEVER_SUBMIT', kernel_id=SLUG, kernel_dir=OUT.relative_to(ROOT).as_posix(),
                candidate_kernel_sha256='59f1d2d33f8888c1dbef385d0f95247b24876a48cbea3d1e38c661231b6c9e2b',
                cell0_sha256=b.digest(CELL0.read_bytes()),
                kernel_files={n: b.sha_file(OUT / n) for n in ('release.ipynb', 'kernel-metadata.json')},
                embedded_members=len(sfiles), dataset_sources=meta['dataset_sources'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'validate'])
    a = p.parse_args()
    result = prepare() if a.command == 'prepare' else validate()
    (b.SCRATCH / ('n32-' + a.command + '-result.json')).write_bytes(b.old.encoded(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
