"""Builder 1.0.0: mechanical own10 + the reviewed own25q3 FAST cell/guard.

Changelog 1.1.21-own10q2: one CPU-only candidate; frozen runtime untouched.
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
from tools import c85r7_own10_v1113 as own
from tools import c85r7_own25q3_v1118 as fast
from tools import strict_variant_gate as gate

VERSION = '1.1.21-own10q2'
SLUG = 'taeyangg4/knee-lb-own10q2-v1'
OUT = own.OUT.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/astra-x1/own10q2'
REVIEW = ROOT / '.local/scratch/slot-b/presub/own10q2-review-out.md'
BUILDER_RECEIPT = ROOT / '.local/scratch/slot-b/presub/own10q2-build-out.md.receipt.json'
BOUND = OUT.with_name(OUT.name + '-gate')
TESTS = ['test_exp169_release_c85', 'test_exp169_release_c85r7',
         'test_exp169_release_c85r7codec', 'test_hb126_codecs',
         'test_c85r7_own10_v1113', 'test_c85r7_own25q3_v1118',
         'test_c85r7_own10q2_v1121', 'test_strict_variant_gate']


def payload():
    own.validate()
    inputs = gate.Inputs(ROOT)
    ref = gate.kernel(fast.OUT, inputs)
    own.need(ref[2]['kernel_files'] == gate.HISTORICAL[fast.OUT.name]['kernel_files'], 'FAST reference drift')
    gate.verdict(inputs.read(gate.FAST_REFERENCE['review']))
    own.need(gate.sha(inputs.read(gate.FAST_REFERENCE['review'])) == gate.FAST_REFERENCE['review_sha256'], 'FAST review drift')
    nb = own.read(own.OUT / 'kernel/release.ipynb')
    nb['cells'].insert(0, copy.deepcopy(own.read(fast.OUT / 'kernel/release.ipynb')['cells'][0]))
    nb['cells'][2]['source'] = ('if not FAST:\n' + ''.join('    ' + line for line in nb['cells'][2]['source'])).splitlines(True)
    meta = own.read(own.OUT / 'kernel/kernel-metadata.json')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta


def scope(nb, meta):
    expected, expected_meta = payload()
    own.need(nb == expected and meta == expected_meta, 'forbidden notebook/metadata diff')
    restored = copy.deepcopy(nb)
    restored['cells'] = restored['cells'][1:]
    wrapped = ''.join(restored['cells'][1]['source']).splitlines(True)
    own.need(wrapped[0] == 'if not FAST:\n' and all(s.startswith('    ') for s in wrapped[1:]), 'guard drift')
    restored['cells'][1]['source'] = [s[4:] for s in wrapped[1:]]
    own.need(restored == own.read(own.OUT / 'kernel/release.ipynb'), 'later-cell drift')
    files, blob, _ = own.unpack(restored)
    original, old_blob, _ = own.unpack(own.read(own.OUT / 'kernel/release.ipynb'))
    own.need(files == original and blob == old_blob and len(files) == 29, 'runtime drift')
    _, v110 = own.parent()
    own.scope_check(v110, files)
    return files, blob


def diff_against(base):
    inputs = gate.Inputs(ROOT)
    before, after = gate.kernel(base, inputs), gate.kernel(OUT, inputs)
    normalized = gate.fast_inventory(before, after, gate.FAST_REFERENCE, inputs)
    rules = gate.exact_rules(before[1], normalized)
    errors, decoded = gate.compare(before[1], normalized, rules)
    own.need(not errors, 'decoded comparison failed')
    inventory = ''.join(f'{name}: {gate.sha(before[1][name])} -> {gate.sha(normalized[name])}\n' for name in sorted(normalized))
    raw = ''.join(difflib.unified_diff(before[3]['release.ipynb'].decode().splitlines(True),
                after[3]['release.ipynb'].decode().splitlines(True), fromfile=base.name + '/release.ipynb', tofile=OUT.name + '/release.ipynb'))
    return (inventory + '\nFULL NORMALIZED DIFF\n' + decoded + '\nFULL RAW NOTEBOOK DIFF\n' + raw).encode(), rules


def prepare():
    nb, meta = payload()
    files, blob = scope(nb, meta)
    for base in [own.OUT, fast.OUT]:
        for name in ['release.ipynb', 'kernel-metadata.json']:
            own.write(ROOT / '.local/backups/astra-x1/own10q2' / base.name / name, (base / 'kernel' / name).read_bytes())
    own.write(OUT / 'kernel/release.ipynb', own.encoded(nb))
    own.write(OUT / 'kernel/kernel-metadata.json', own.encoded(meta))
    for base in [own.OUT, own.BASE]:
        diff, rules = diff_against(base)
        own.write(OUT / ('full-diff-vs-' + base.name + '.txt'), diff)
        own.write(OUT / ('diff-rules-vs-' + base.name + '.json'), gate.encoded(rules))
    own.write(OUT / 'registration.json', gate.encoded(dict(version=VERSION, embedded_runtime_version=own.VERSION,
        kernel_id=SLUG, formula=own.FORMULA, parent=own.OUT.relative_to(ROOT).as_posix(),
        qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)))
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, archive_sha256=gate.sha(blob),
        builder_sha256=gate.sha(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'],
        embedded={k:gate.sha(v) for k,v in files.items()}, identical_files=29,
        cell0_sha256=gate.sha(''.join(nb['cells'][0]['source']).encode()),
        visible_identity=fast.reference.authenticate_visible(), qualified_for_submission=False,
        exposure='EXPOSED', clean_oof=False)
    own.write(OUT / 'build-receipt.json', gate.encoded(rec))
    names = ['kernel/release.ipynb', 'kernel/kernel-metadata.json', 'registration.json', 'build-receipt.json']
    own.write(OUT / 'SHA256SUMS', ''.join(gate.sha((OUT / n).read_bytes()) + '  ' + n + '\n' for n in sorted(names)).encode())
    own.write(OUT / 'CHANGELOG.md', b'# 1.1.21-own10q2 / builder 1.0.0\nMechanical copy of reviewed own25q3 cell 0 and exact if-not-FAST four-space guard onto own10. All 29 runtime files and restored later cells unchanged. Frozen stages untouched; backups under .local/backups/astra-x1/own10q2/.\n')
    # Actual extracted bytes, only in this lane's fixed scratch folder.
    for name, data in files.items():
        own.write(SCRATCH / 'candidate-runtime' / name, data)
    return validate()


def validate():
    nb, meta = own.read(OUT / 'kernel/release.ipynb'), own.read(OUT / 'kernel/kernel-metadata.json')
    files, blob = scope(nb, meta)
    rec = own.read(OUT / 'build-receipt.json')
    own.need(rec['kernel_files'] == gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], 'kernel hash drift')
    own.need(rec['builder_sha256'] == gate.sha(Path(__file__).read_bytes()), 'builder drift')
    own.need(rec['archive_sha256'] == gate.sha(blob), 'archive drift')
    gate.manifest(OUT, gate.Inputs(ROOT), [OUT / n for n in ['kernel/release.ipynb', 'kernel/kernel-metadata.json', 'registration.json', 'build-receipt.json']])
    for base in [own.OUT, own.BASE]:
        diff, rules = diff_against(base)
        own.need((OUT / ('full-diff-vs-' + base.name + '.txt')).read_bytes() == diff, 'diff drift')
        own.need(own.read(OUT / ('diff-rules-vs-' + base.name + '.json')) == rules, 'diff rules drift')
    return dict(status='PASS', identical_embedded=29, restored_later_cells_identical=True,
                cell0_exact_reviewed_copy=True, v110_changed_runtime_files=own.scope_check(own.parent()[1], files))


def fixture_builder():
    return SimpleNamespace(ROOT=ROOT, OUT=OUT, old=own, reference=fast.reference, VISIBLE_IDS=fast.VISIBLE_IDS)


def smoke():
    from tools import c85r7_own25q3_cpu_fixtures as fixtures
    validate()
    results = []
    with patch.object(fixtures, 'b', fixture_builder()):
        for name, kwargs in fixtures.case_specs():
            area = (SCRATCH / 'fast-smoke' / name).resolve()
            own.need(area.is_relative_to(SCRATCH.resolve()), 'scratch containment')
            if area.exists():
                shutil.rmtree(area)
            results.append(dict(case=name, **fixtures.run_case(area, **kwargs)))
    with patch.object(own, 'SCRATCH', SCRATCH):
        replay = own.replay()
    result = dict(status='PASS', cases=results, notebook_sha256=gate.sha((OUT / 'kernel/release.ipynb').read_bytes()),
        visible_identity=fast.reference.authenticate_visible(), replay=replay,
        limitation='Real test.csv; synthetic study directories. FAST=False reaches original first statement only, before model code. Real3 arithmetic uses authenticated sufficient statistics, not raw member inference. No GPU/Linux qualification.')
    (SCRATCH / 'smoke.json').write_bytes(gate.encoded(result))
    return dict(status='PASS', cases=len(results), fast_true=sum(r['FAST'] for r in results),
                fast_false=sum(not r['FAST'] for r in results), real3_arithmetic=replay['status'])


def admit():
    validate()
    report = own.read(SCRATCH / 'smoke.json')
    own.need(report['status'] == 'PASS' and report['notebook_sha256'] == gate.sha((OUT / 'kernel/release.ipynb').read_bytes()) and
             len(report['cases']) == 77 and all(r['status'] == 'PASS' for r in report['cases']), 'smoke missing/stale/failed')
    return dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False)


def tests():
    import pytest
    # The inherited own10 suite writes replay/import evidence; keep it in our lane.
    with patch.object(own, 'SCRATCH', SCRATCH):
        return pytest.main(['tests/' + n + '.py' for n in TESTS] + ['--basetemp=.local/pytest-tmp/astra-x1', '-q'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'register', 'cpu-smoke', 'admit', 'tests'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if args.command in ['prepare', 'admit']:
        own.need(args.dry_run, 'dry-run only')
    if args.command == 'tests':
        return tests()
    print(json.dumps({'prepare':prepare, 'validate':validate, 'register':validate, 'cpu-smoke':smoke, 'admit':admit}[args.command](), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
