"""H-B-173 strict one-candidate CPU builder 1.0.0.

Changelog: admitted R224 mirror on hidden-completed no-FLIP_ALL R224 parent.
Extends existing mirror, archive and gate helpers; no frozen file modifications.
"""
from __future__ import annotations
import argparse
import ast
import base64
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_flipall_v1115 as old
from tools import c85r7_flipallq_v1122 as flipq
from tools import c85r7_r224q_v1123 as r224
from tools import c85r7_r224mirq_v1126 as mirror
from tools import strict_variant_gate as gate
from tools import hb145_output_io as output_io

VERSION = '1.1.29-r224mirnfq'
SLUG = 'taeyangg4/knee-lb-r224mirnfq-v1'
BASE = ROOT / '.local/scratch/astra-b5/r224-gate/gate-copy-v142'
REFERENCE = ROOT / '.local/scratch/astra-b5/hb170-gate/gate-copy-v144'
OUT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7' / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb173-build/hb173'
BACKUP = ROOT / '.local/backups/hb173-build/hb173'
SOURCE, WRAPPER = mirror.SOURCE, mirror.WRAPPER
PORT, CHAIN = 'exp169_r7_port.py', 'exp169_release_c85r7.py'
ADMISSIONS = {
    BASE: ('knee-lb-r224q-v1', '1.4.2', '25e9c03e00293e4cbb0ac5088afca7875dfaa1d56d9d16b6ff7248d98e69afe3'),
    REFERENCE: ('knee-lb-r224mirq-v1', '1.4.4', '5cb51fea0a688b42233da8651042fa5dfc15edda82ec0f91c6c214863558273b'),
}


def authenticated(stage):
    slug, version, pin = ADMISSIONS[stage]
    receipt = ROOT / '.local/state/gates' / (slug + '.json')
    old.pin(receipt, pin)
    rec = old.read(receipt)
    old.need(rec['status'] == 'PASS' and rec['errors'] == [] and rec['fixture_only'] is False
             and rec['gate_version'] == version and rec['kernel_id'] == 'taeyangg4/' + slug
             and Path(rec['stage']) == stage, 'admitted source identity mismatch')
    inventory = gate.kernel(stage, gate.Inputs(ROOT))[2]
    old.need(inventory == rec['stage_inventory'], 'admitted source inventory drift')
    nb = old.read(stage / 'kernel/release.ipynb')
    files, blob, assignments = old.unpack(flipq.restored(nb))
    return nb, files, blob, assignments


def patch_files(original, reference):
    files = dict(original)
    # The admitted source is copied, never regenerated from current public code.
    files[SOURCE] = reference[SOURCE]
    s = old.replace(original[WRAPPER], old.digest(original[SOURCE]), old.digest(files[SOURCE]))
    for before, after in [('1.1.23-r224q', VERSION), ('knee-lb-r224q-v1', SLUG.split('/')[1])]:
        # Version occurs in VERSION and OUT, both explicitly allowed metadata.
        old.need(s.count(before.encode()) == (2 if before.startswith('1.') else 1), 'metadata anchor drift')
        s = s.replace(before.encode(), after.encode())
    files[WRAPPER] = s
    return files


def reverse_reference(reference):
    """Independent second derivation: remove only admitted flip hunk + its pins."""
    files = dict(reference)
    files[PORT] = old.replace(reference[PORT], old.FLAG_NEW, old.FLAG_OLD)
    files[CHAIN] = old.replace(reference[CHAIN], old.digest(reference[PORT]), old.digest(files[PORT]))
    s = old.replace(reference[WRAPPER], old.digest(reference[CHAIN]), old.digest(files[CHAIN]))
    for before, after in [
        ('VERSION = "1.1.26-r224mirq"', 'VERSION = "' + VERSION + '"'),
        ('KERNEL_ID = "taeyangg4/knee-lb-r224mirq-v1"', 'KERNEL_ID = "' + SLUG + '"'),
        ('RELEASE = "C85R7+codecs R224 FLIP_ALL"', 'RELEASE = "C85R7+codecs R224"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq"',
         'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + VERSION + '"')]:
        s = old.replace(s, before, after)
    files[WRAPPER] = s
    return files


def repack(nb, original_blob, assignments, files):
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original_blob)) as src, zipfile.ZipFile(buf, 'w') as dst:
        for info in src.infolist():
            dst.writestr(info, files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before, after in [
        ('BLOB = ' + repr(assignments['BLOB']), 'BLOB = ' + repr(base64.b64encode(blob).decode())),
        ('PINS = ' + repr(assignments['PINS']), 'PINS = ' + repr({k: old.digest(v) for k, v in files.items()})),
        (repr(old.digest(original_blob)), repr(old.digest(blob)))]:
        code = old.replace(code, before, after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    return nb, blob


def payload():
    nb, original, original_blob, assignments = authenticated(BASE)
    rn, reference, rb, ra = authenticated(REFERENCE)
    files = patch_files(original, reference)
    old.need(files == reverse_reference(reference), 'two-way exact embedded allowlist mismatch')
    # Prove the donor source is precisely the admitted mirror patch to plain R224.
    donor_parent_nb = old.read(REFERENCE / 'parent/kernel/release.ipynb')
    donor_parent = old.unpack(flipq.restored(donor_parent_nb))[0]
    old.need(donor_parent[SOURCE] == original[SOURCE], 'mirror parent source mismatch')
    old.need(mirror.patch_files(donor_parent)[SOURCE] == reference[SOURCE], 'admitted mirror hunk mismatch')
    old.need(files[PORT] == original[PORT] and files[CHAIN] == original[CHAIN], 'flip must stay OFF')
    nb, blob = repack(nb, original_blob, assignments, files)
    other_nb, other_blob = repack(rn, rb, ra, reverse_reference(reference))
    old.need(nb == other_nb and blob == other_blob, 'two-way notebook/bootstrap/archive mismatch')
    meta = old.read(BASE / 'kernel/kernel-metadata.json')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    rmeta = old.read(REFERENCE / 'kernel/kernel-metadata.json')
    rmeta.update(id=SLUG, title=SLUG.split('/')[1])
    old.need(meta == rmeta, 'metadata beyond id/title changed')
    return nb, meta, files, blob


def scope(nb, meta):
    expected, emeta, files, blob = payload()
    old.need(nb == expected and meta == emeta, 'forbidden notebook/metadata difference')
    actual, ab, _ = old.unpack(flipq.restored(nb))
    old.need(actual == files and ab == blob and len(actual) == 31, 'forbidden embedded difference')
    for name, data in files.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    return files, blob


def registration():
    timing_path = ROOT / '.local/scratch/hb170-build/hb170/timing/measurement.json'
    timing_review = ROOT / '.local/scratch/slot-b/hb170/timing-result-review-out.md'
    old.pin(timing_path, '7957b9cde0781c43958f85410fd007ff118345b92f1097ddc650dd6ea35ea659')
    old.pin(timing_review, '749da5aa83a8e8eda47f7c0077f3a5fa1dafcb7cbc3713efacd2ae86d1dbdf9c')
    timing = old.read(timing_path)
    old.need(timing['identity'] == 'HB170_N32' and timing['result']['failed_study_fallbacks'] == 0,
             'timing source identity')
    old.need(b'VERDICT: PASS' in timing_review.read_bytes(), 'increment result review required')
    increment = timing['result']['projected_hidden_increment_seconds']
    old.need(type(increment) in (int, float) and 0 < increment < 32400, 'invalid mirror increment')
    return dict(version=VERSION, item='H-B-173', kernel_id=SLUG,
        parent=str(BASE.relative_to(ROOT)), parent_submission=57200805,
        source_admissions={str(p.relative_to(ROOT)): dict(slug=v[0], gate_version=v[1], receipt_sha256=v[2]) for p,v in ADMISSIONS.items()},
        spec_sha256=old.digest((SCRATCH / 'SPEC.md').read_bytes()), formula=r224.FORMULA,
        purpose='final-pair A candidate; hedge without disputed FLIP_ALL (D-B-199, scheme v2 rule 8)',
        mirror_source='admitted v1.1.26-r224mirq raptor224_run.py, byte-identical',
        flip_tta=False, public_reading=dict(parent=.957, minimum_delta=-.001, keep_min=.956,
            rule='candidate >= R-224 reading - 0.001', source='user supplied; no new reading'),
        timing=dict(formula='R-224 hidden approximately 5.3 h + HB170_N32 conservative mirror increment',
            baseline_hours_approx=5.3, increment_source='.local/scratch/hb170-build/hb170/timing/measurement.json',
            increment_review='.local/scratch/slot-b/hb170/timing-result-review-out.md', runtime_limit_hours=9,
            increment_source_sha256=old.digest(timing_path.read_bytes()),
            increment_review_sha256=old.digest(timing_review.read_bytes()),
            projected_increment_seconds=increment, planning_total_hours=5.3 + increment / 3600,
            measured_candidate_runtime=False,
            limitation='Planning projection; approximate baseline, cross-device uncertainty; not measured candidate hidden runtime'),
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        gate_binding='DEFERRED independent exact-diff gate job; no gate copy')


def comparison(stage, nb, meta, files):
    before = gate.kernel(stage, gate.Inputs(ROOT))
    inventory, _ = gate.unpack_notebook(old.encoded(nb))
    inventory['kernel/kernel-metadata.json'] = old.encoded(meta)
    old.need(set(inventory) == set(before[1]), 'inventory member set changed')
    original = authenticated(stage)[1]
    changed = sorted(k for k in files if original[k] != files[k])
    expected = sorted([SOURCE, WRAPPER] if stage == BASE else [PORT, CHAIN, WRAPPER])
    old.need(changed == expected, 'changed-file scope drift')
    manifest = dict(changed=changed, unchanged=[k for k in sorted(files) if k not in changed],
        added=[], removed=[], inventory={k:dict(before_sha256=gate.sha(before[1][k]),
            after_sha256=gate.sha(inventory[k]), identical=before[1][k] == inventory[k]) for k in sorted(inventory)})
    full = 'FULL DECODED INVENTORY, ALL EMBEDDED FILES / NOTEBOOK CELLS / METADATA\n'
    for name in sorted(inventory):
        full += name + ' ' + json.dumps(manifest['inventory'][name], sort_keys=True) + '\n'
        full += r224.diff(before[1][name], inventory[name], stage.name + '/' + name, OUT.name + '/' + name)
    full += '\nFULL RAW NOTEBOOK AND METADATA DIFF\n'
    for name, data in [('release.ipynb', old.encoded(nb)), ('kernel-metadata.json', old.encoded(meta))]:
        full += r224.diff(before[3][name], data, stage.name + '/' + name, OUT.name + '/' + name)
    return manifest, full.encode()


def artifacts():
    nb, meta, files, blob = payload()
    scope(nb, meta)
    values = {'kernel/release.ipynb': old.encoded(nb), 'kernel/kernel-metadata.json': old.encoded(meta),
              'registration.json': gate.encoded(registration())}
    for stage, name in [(BASE, 'v1.1.23-r224q'), (REFERENCE, 'v1.1.26-r224mirq')]:
        manifest, full = comparison(stage, nb, meta, files)
        values['inventory-vs-' + name + '.json'] = gate.encoded(manifest)
        values['full-diff-vs-' + name + '.txt'] = full
    return values, files, blob


def prepare():
    values, files, blob = artifacts()
    for stage, (slug, _, _) in ADMISSIONS.items():
        for name in ('kernel/release.ipynb', 'kernel/kernel-metadata.json', 'registration.json'):
            old.write(BACKUP / slug / name, (stage / name).read_bytes())
        old.write(BACKUP / (slug + '-admission.json'), (ROOT / '.local/state/gates' / (slug + '.json')).read_bytes())
    for name, data in values.items():
        old.write(OUT / name, data)
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()}, unchanged_vs_parent=29,
        unchanged_vs_mirror=28, qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    old.write(OUT / 'build-receipt.json', gate.encoded(rec))
    old.write(OUT / 'CHANGELOG.md', b'# v1.1.29-r224mirnfq / builder 1.0.0\nAdmitted R224 mirror copied byte-identically onto hidden-completed r224q; R7 FLIP_ALL remains OFF. Only source, chained pins and version/id/output metadata change. 29/31 members identical to parent; 28/31 identical to mirror donor. No frozen edits; backups .local/backups/hb173-build/hb173. No gate copy or GPU/Kaggle execution.\n')
    for name, data in files.items():
        old.write(SCRATCH / 'candidate-runtime' / name, data)
    return validate()


def validate():
    values, files, blob = artifacts()
    for name, data in values.items():
        old.need((OUT / name).read_bytes() == data, 'stage drift: ' + name)
    rec = old.read(OUT / 'build-receipt.json')
    old.need(rec['builder_sha256'] == old.digest(Path(__file__).read_bytes()), 'builder drift')
    old.need(rec['embedded'] == {k:old.digest(v) for k,v in files.items()} and rec['archive_sha256'] == old.digest(blob), 'receipt drift')
    old.need(rec['kernel_files'] == gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], 'kernel drift')
    return dict(status='PASS_CPU_VALIDATE', unchanged_vs_parent=29, unchanged_vs_mirror=28,
        kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),
        notebook_sha256=rec['kernel_files']['release.ipynb'], qualified_for_submission=False)


def smoke():
    from tools import c85r7_r224mir_cpu as cpu
    validate()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    class EmptyVisibility:
        """Adapt only the reused CPU helper's -1 assignment to requested empty visibility."""
        def __setitem__(self, key, value):
            old.need(key == 'CUDA_VISIBLE_DEVICES' and value in ('', '-1'), 'unexpected CPU environment write')
            os.environ[key] = ''
    with patch.object(cpu, 'b', sys.modules[__name__]), \
         patch.object(cpu, 'os', SimpleNamespace(environ=EmptyVisibility())):
        mirror_result = cpu.run()
    # Reuse sealed real3 arithmetic replay; this deliberately proves only the
    # unchanged outer/own arithmetic, not new R224 or R7 neural predictions.
    saved_unpack = old.unpack
    def unpack(nb):
        return saved_unpack(flipq.restored(nb) if len(nb['cells']) == 3 else nb)
    with patch.object(old, 'OUT', OUT), patch.object(old, 'SCRATCH', SCRATCH / 'replay'), \
         patch.object(old, 'validate', validate), patch.object(old, 'unpack', unpack):
        replay = old.replay()
    result = dict(status='PASS', kernel_sha256=validate()['kernel_sha256'],
        mirror=mirror_result, real3_arithmetic=replay,
        limitation='Two real studies with actual CPU pair model forward; unchanged outer/own formula on sealed real3 sufficient statistics. Not full blended neural predictions, CUDA, Linux parity, or runtime qualification.')
    output_io.atomic_json(SCRATCH / 'smoke.json', result)
    return dict(status='PASS', real_mirror_studies=2, anatomical_slots=5,
        real3_arithmetic=replay['status'], kernel_sha256=result['kernel_sha256'])


def decision(parent_score, candidate_score, hidden_seconds, increment):
    return mirror.decision(parent_score, candidate_score, hidden_seconds, increment)


def admit():
    current = validate()
    result = old.read(SCRATCH / 'smoke.json')
    old.need(result['status'] == 'PASS' and result['kernel_sha256'] == current['kernel_sha256'], 'CPU smoke missing/stale')
    return dict(status='HOLD_INDEPENDENT_EXACT_DIFF_GATE_AND_QUALIFICATION',
        kernel_sha256=current['kernel_sha256'], qualified_for_submission=False, gate_copy_created=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register','prepare','validate','cpu-smoke','admit'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if args.command in ('prepare','admit'):
        old.need(args.dry_run, 'CPU dry-run required')
    if args.command == 'register':
        old.write(SCRATCH / 'registration.json', gate.encoded(registration()))
        result = {'status':'REGISTERED_CPU_ONLY'}
    else:
        result = {'prepare':prepare, 'validate':validate, 'cpu-smoke':smoke, 'admit':admit}[args.command]()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
