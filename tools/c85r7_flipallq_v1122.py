"""Builder 1.0.0: v1.1.0 + reviewed flipall bytes + reviewed FAST cell/guard.

Changelog 1.1.22-flipallq: one runtime-only public reading; CPU-only builder.
Original tools/stages are untouched; source snapshots are backed up.
"""
from __future__ import annotations
import argparse
import copy
import difflib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_flipall_v1115 as flip
from tools import c85r7_own25q3_v1118 as fast
from tools import strict_variant_gate as gate

VERSION = '1.1.22-flipallq'
SLUG = 'taeyangg4/knee-lb-flipallq-v1'
OUT = flip.BASE.parent / ('v' + VERSION)
BOUND = OUT.with_name(OUT.name + '-gate')
SCRATCH = ROOT / '.local/scratch/slot-b/flipallq'
BACKUP = ROOT / '.local/backups/slot-b/flipallq'
REVIEW = ROOT / '.local/scratch/slot-b/presub/flipallq-review-out.md'
BUILDER_RECEIPT = ROOT / '.local/scratch/slot-b/presub/flipallq-build-out.md.receipt.json'
TESTS = ['test_exp169_release_c85', 'test_exp169_release_c85r7',
         'test_exp169_release_c85r7codec', 'test_hb126_codecs',
         'test_c85r7_flipall_v1115', 'test_c85r7_own25q3_v1118',
         'test_c85r7_flipallq_v1122', 'test_strict_variant_gate']


def payload():
    flip.validate()
    inputs = gate.Inputs(ROOT)
    ref = gate.kernel(fast.OUT, inputs)
    flip.need(ref[2]['kernel_files'] == gate.HISTORICAL[fast.OUT.name]['kernel_files'], 'FAST reference drift')
    flip.need(gate.sha(inputs.read(gate.FAST_REFERENCE['review'])) == gate.FAST_REFERENCE['review_sha256'], 'FAST review drift')
    gate.verdict(inputs.read(gate.FAST_REFERENCE['review']))
    nb = flip.read(flip.OUT / 'kernel/release.ipynb')
    nb['cells'].insert(0, copy.deepcopy(flip.read(fast.OUT / 'kernel/release.ipynb')['cells'][0]))
    nb['cells'][2]['source'] = ('if not FAST:\n' + ''.join('    ' + line for line in nb['cells'][2]['source'])).splitlines(True)
    meta = flip.read(flip.OUT / 'kernel/kernel-metadata.json')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta


def restored(nb):
    result = copy.deepcopy(nb)
    result['cells'] = result['cells'][1:]
    lines = ''.join(result['cells'][1]['source']).splitlines(True)
    flip.need(lines[0] == 'if not FAST:\n' and all(s.startswith('    ') for s in lines[1:]), 'guard drift')
    result['cells'][1]['source'] = [s[4:] for s in lines[1:]]
    return result


def scope(nb, meta):
    expected, expected_meta = payload()
    flip.need(nb == expected and meta == expected_meta, 'forbidden notebook/metadata diff')
    plain = restored(nb)
    flip.need(plain == flip.read(flip.OUT / 'kernel/release.ipynb'), 'later-cell drift')
    files, blob, _ = flip.unpack(plain)
    original, old_blob, _ = flip.unpack(flip.read(flip.OUT / 'kernel/release.ipynb'))
    flip.need(files == original and blob == old_blob and len(files) == 29, 'flipall runtime drift')
    flip.scope_check(flip.parent()[1], files)
    return files, blob


def registration():
    return dict(version=VERSION, embedded_runtime_version=flip.VERSION, kernel_id=SLUG,
        formula=flip.FORMULA, parent=flip.BASE.relative_to(ROOT).as_posix(),
        parent_receipt_sha256=flip.BASE_SHA,
        registered_changes=['Exact reviewed v1.1.15-flipall embedded bytes, including chained pins and receipt/version metadata.',
                            'Exact reviewed v1.1.18-own25q3 FAST cell object and if-not-FAST guard.'],
        public_reading=dict(purpose='runtime evidence only', score_is_gain_test=False,
            decision_rule='Hidden COMPLETE within 9 hours is runtime-admissible for finals; not evidence of score gain.',
            failure_rule='Timeout, non-COMPLETE, missing authenticated runtime, or elapsed >9 hours is not runtime-admissible.',
            source='D-B-188', prior_projection_hours=[8.3, 8.7]),
        qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)


def runtime_admissible(status, seconds):
    import math
    return (status == 'COMPLETE' and type(seconds) in (int, float)
            and math.isfinite(seconds) and 0 <= seconds <= 9 * 3600)


def diff_against(base):
    inputs = gate.Inputs(ROOT)
    before, after = gate.kernel(base, inputs), gate.kernel(OUT, inputs)
    normalized = gate.fast_inventory(before, after, gate.FAST_REFERENCE, inputs)
    rules = gate.exact_rules(before[1], normalized)
    errors, decoded = gate.compare(before[1], normalized, rules)
    flip.need(not errors, 'decoded comparison failed')
    inventory = ''.join(f'{name}: {gate.sha(before[1][name])} -> {gate.sha(normalized[name])}' +
                        (' IDENTICAL\n' if before[1][name] == normalized[name] else ' ALLOWED_CHANGE\n')
                        for name in sorted(normalized))
    raw = ''.join(''.join(difflib.unified_diff(before[3][name].decode().splitlines(True),
                    after[3][name].decode().splitlines(True), fromfile=base.name + '/' + name,
                    tofile=OUT.name + '/' + name)) for name in sorted(before[3]))
    return (inventory + '\nFULL NORMALIZED DIFF\n' + decoded + '\nFULL RAW KERNEL DIFF\n' + raw).encode(), rules


def prepare():
    nb, meta = payload()
    files, blob = scope(nb, meta)
    for base in [flip.BASE, flip.OUT, fast.OUT]:
        for name in ['release.ipynb', 'kernel-metadata.json']:
            flip.write(BACKUP / base.name / name, (base / 'kernel' / name).read_bytes())
    flip.write(OUT / 'kernel/release.ipynb', flip.encoded(nb))
    flip.write(OUT / 'kernel/kernel-metadata.json', flip.encoded(meta))
    for base in [flip.OUT, flip.BASE]:
        diff, rules = diff_against(base)
        flip.write(OUT / ('full-diff-vs-' + base.name + '.txt'), diff)
        flip.write(OUT / ('diff-rules-vs-' + base.name + '.json'), gate.encoded(rules))
    flip.write(OUT / 'registration.json', gate.encoded(registration()))
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, embedded_runtime_version=flip.VERSION,
        archive_sha256=gate.sha(blob), builder_sha256=gate.sha(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'],
        embedded={k:gate.sha(v) for k,v in files.items()}, identical_to_flipall=29,
        identical_to_v110=26, changed_vs_v110=flip.scope_check(flip.parent()[1], files),
        cell0_sha256=gate.sha(''.join(nb['cells'][0]['source']).encode()),
        visible_identity=fast.reference.authenticate_visible(), qualified_for_submission=False,
        exposure='EXPOSED', clean_oof=False)
    flip.write(OUT / 'build-receipt.json', gate.encoded(rec))
    names = ['kernel/release.ipynb', 'kernel/kernel-metadata.json', 'registration.json', 'build-receipt.json']
    flip.write(OUT / 'SHA256SUMS', ''.join(gate.sha((OUT / n).read_bytes()) + '  ' + n + '\n' for n in sorted(names)).encode())
    flip.write(OUT / 'CHANGELOG.md', b'# 1.1.22-flipallq / builder 1.0.0\nExact v1.1.15-flipall archive plus reviewed own25q3 FAST cell and guard. The embedded runtime retains its reviewed 1.1.15-flipall version; this stage version is 1.1.22-flipallq. Runtime-only public reading; no gain test. Original tools and frozen stages untouched; source snapshots: .local/backups/slot-b/flipallq/.\n')
    for name, data in files.items():
        flip.write(SCRATCH / 'candidate-runtime' / name, data)
        flip.write(SCRATCH / 'flip-reference/candidate-runtime' / name, data)
    return validate()


def validate():
    nb, meta = flip.read(OUT / 'kernel/release.ipynb'), flip.read(OUT / 'kernel/kernel-metadata.json')
    files, blob = scope(nb, meta)
    rec = flip.read(OUT / 'build-receipt.json')
    flip.need(rec['kernel_files'] == gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], 'kernel hash drift')
    flip.need(rec['builder_sha256'] == gate.sha(Path(__file__).read_bytes()), 'builder drift')
    flip.need(rec['archive_sha256'] == gate.sha(blob), 'archive drift')
    flip.need(flip.read(OUT / 'registration.json') == registration(), 'registration drift')
    gate.manifest(OUT, gate.Inputs(ROOT), [OUT / n for n in ['kernel/release.ipynb', 'kernel/kernel-metadata.json', 'registration.json', 'build-receipt.json']])
    for base in [flip.OUT, flip.BASE]:
        diff, rules = diff_against(base)
        flip.need((OUT / ('full-diff-vs-' + base.name + '.txt')).read_bytes() == diff, 'diff drift')
        flip.need(flip.read(OUT / ('diff-rules-vs-' + base.name + '.json')) == rules, 'diff rules drift')
    return dict(status='PASS', identical_to_flipall=29, identical_to_v110=26,
        changed_vs_v110=flip.scope_check(flip.parent()[1], files), cell0_exact_reviewed_copy=True,
        restored_later_cells_identical_to_flipall=True)


def fixture_builder():
    return SimpleNamespace(ROOT=ROOT, OUT=OUT, old=flip, reference=fast.reference, VISIBLE_IDS=fast.VISIBLE_IDS)


def smoke():
    from tools import c85r7_own25q3_cpu_fixtures as fixtures
    from tools import c85r7_flipall_cpu_smoke as pixel
    validate()
    results = []
    with patch.object(fixtures, 'b', fixture_builder()):
        for name, kwargs in fixtures.case_specs():
            area = (SCRATCH / 'fast-smoke' / name).resolve()
            flip.need(area.is_relative_to(SCRATCH.resolve()), 'scratch containment')
            if area.exists():
                shutil.rmtree(area)
            results.append(dict(case=name, **fixtures.run_case(area, **kwargs)))
    # The same reviewed 29 files, authenticated above, run with the candidate
    # stage as input. Unwrapping is for the CPU inspector only, never serving.
    adapter = SimpleNamespace(**{n:getattr(flip, n) for n in dir(flip) if not n.startswith('__')})
    adapter.OUT, adapter.SCRATCH, adapter.validate = OUT, SCRATCH, validate
    adapter.unpack = lambda nb: flip.unpack(restored(nb))
    with patch.object(pixel, 'b', adapter):
        pixel.main()
    # Replay the byte-identical arithmetic on the authenticated real visible
    # sufficient statistics, keeping inherited suite outputs in this lane.
    with patch.object(flip, 'SCRATCH', SCRATCH / 'flip-reference'):
        replay = flip.replay()
    shutil.copyfile(SCRATCH / 'flip-cpu-smoke.json', SCRATCH / 'flip-reference/flip-cpu-smoke.json')
    result = dict(status='PASS', cases=results, notebook_sha256=gate.sha((OUT / 'kernel/release.ipynb').read_bytes()),
        visible_identity=fast.reference.authenticate_visible(), replay=replay,
        real_pixel_smoke=flip.read(SCRATCH / 'flip-cpu-smoke.json'),
        limitation='FAST: real test.csv with synthetic study directories. FAST=False executes only the original first import. Flip: real checkpoints/9 DICOMs, reduced windows on CPU. Arithmetic: real visible sufficient statistics. No full-study/GPU/Linux/runtime qualification.')
    (SCRATCH / 'smoke.json').write_bytes(gate.encoded(result))
    return dict(status='PASS', cases=len(results), fast_true=sum(r['FAST'] for r in results),
        fast_false=sum(not r['FAST'] for r in results), real3_arithmetic=replay['status'],
        real_pixel_folds=len(result['real_pixel_smoke']['folds']))


def admit():
    validate()
    report = flip.read(SCRATCH / 'smoke.json')
    flip.need(report['status'] == 'PASS' and report['notebook_sha256'] == gate.sha((OUT / 'kernel/release.ipynb').read_bytes()) and
              len(report['cases']) == 77 and all(r['status'] == 'PASS' for r in report['cases']), 'smoke missing/stale/failed')
    return dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False)


def tests():
    import pytest
    with patch.object(flip, 'SCRATCH', SCRATCH / 'flip-reference'):
        return pytest.main(['tests/' + n + '.py' for n in TESTS] + ['--basetemp=.local/pytest-tmp/slot-b', '-q'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'register', 'cpu-smoke', 'admit', 'tests'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if args.command in ['prepare', 'admit']:
        flip.need(args.dry_run, 'dry-run only')
    if args.command == 'tests':
        return tests()
    print(json.dumps({'prepare':prepare, 'validate':validate, 'register':validate, 'cpu-smoke':smoke, 'admit':admit}[args.command](), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
