"""H-B-179 strict R6 + anatomical-mirror composition builder v1.0.0.

Changelog: one new stage, authenticated branch composition and CPU-only review
evidence. Originals untouched; source kernel backups under .local/backups/hb179.
"""
from __future__ import annotations
import argparse
import ast
import base64
import difflib
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r224r6fq_v1130 as r6
from tools import c85r7_r224mirq_v1126 as mirror
from tools import strict_variant_gate_r224_v144 as gate144
old, gate, fast = r6.old, r6.gate, r6.fast
VERSION = '1.1.32-r224r6mirfq'
SLUG = 'taeyangg4/knee-lb-r224r6mirfq-v1'
OUT = r6.OUT.parent / ('v' + VERSION)
WORK = ROOT / '.local/scratch/astra-b1/hb179'
BACKUP = ROOT / '.local/backups/hb179'
STAGES = dict(parent=r6.BASE, r6=r6.OUT,
              mirror=ROOT / '.local/scratch/astra-b5/hb170-gate/gate-copy-v144')
R6_KERNEL = 'c073de7e769251bc042b4754d9a0feae88ddf8bdd2802db610c699a2cd845716'
SOURCE, WRAPPER = mirror.SOURCE, r6.WRAPPER
HELPER = 'exp169_r224.py'


def cpu_only():
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')


def snapshots():
    r6.parent_builder.parent()
    validate_r6()
    receipt = ROOT / '.local/state/gates/knee-lb-r224mirq-v1.json'
    inputs = gate.Inputs(ROOT)
    rec = gate144.verify_receipt(receipt, inputs)
    old.need(gate.local_path(rec['stage'], ROOT) == STAGES['mirror'], 'H170 admitted stage mismatch')
    result = {}
    for key, path in STAGES.items():
        nb = old.read(path / 'kernel/release.ipynb')
        files, blob, assignments = old.unpack(fast.restored(nb))
        old.need(len(files) == 31, 'member count')
        result[key] = dict(nb=nb, files=files, blob=blob, assignments=assignments,
                           meta=old.read(path / 'kernel/kernel-metadata.json'),
                           kernel_files=gate.kernel(path, inputs)[2]['kernel_files'])
    old.need(result['mirror']['kernel_files'] == rec['stage_inventory']['kernel_files'], 'mirror receipt inventory')
    inputs.recheck()
    return result


def validate_r6():
    """Authenticate frozen kernel independently of B's additive registration fields."""
    nb, meta, files, blob = r6.payload()
    old.need(old.read(r6.OUT / 'kernel/release.ipynb') == nb and
             old.read(r6.OUT / 'kernel/kernel-metadata.json') == meta, 'H174 payload drift')
    kfiles = gate.kernel(r6.OUT, gate.Inputs(ROOT))[2]['kernel_files']
    old.need(gate.sha(gate.encoded(kfiles)) == R6_KERNEL, 'H174 fixed kernel drift')
    rec = old.read(r6.OUT / 'build-receipt.json')
    old.need(rec['kernel_files'] == kfiles and rec['embedded'] == {k: old.digest(v) for k, v in files.items()}
             and rec['archive_sha256'] == old.digest(blob), 'H174 receipt drift')
    expected = r6.registration()
    stored = old.read(r6.OUT / 'registration.json')
    extra = {'kernel_id': r6.SLUG, 'dataset_sources_delta_version': '1.0.0',
             'dataset_sources_delta': {'added': [r6.PRIVATE], 'removed': ['heliosli/rsna-knee-compact-r7cnxfsss']}}
    old.need(stored == expected or stored == dict(expected, **extra), 'H174 registration drift')
    return R6_KERNEL


def versioned(raw, before_version, before_slug):
    raw = old.replace(raw, 'VERSION = "' + before_version + '"', 'VERSION = "' + VERSION + '"')
    raw = old.replace(raw, 'KERNEL_ID = "' + before_slug + '"', 'KERNEL_ID = "' + SLUG + '"')
    return old.replace(raw, '/v' + before_version + '"', '/v' + VERSION + '"')


def merge(s):
    p, r, m = (s[k]['files'] for k in ('parent', 'r6', 'mirror'))
    old.need(set(p) == set(r) == set(m), 'member roster drift')
    old.need(r6.patched(p) == r and mirror.patch_files(p) == m, 'source branch transform drift')
    files = dict(r)
    files[SOURCE] = m[SOURCE]
    w = old.replace(r[WRAPPER], old.digest(p[SOURCE]), old.digest(m[SOURCE]))
    files[WRAPPER] = versioned(w, r6.VERSION, r6.SLUG)
    # Independently apply H174 wrapper hunks to the admitted H170 branch.
    reverse = old.replace(m[WRAPPER], mirror.VERSION, '1.1.24-r224flipq')
    reverse = old.replace(reverse, mirror.SLUG, 'taeyangg4/knee-lb-r224flipq-v1')
    a, z = p[WRAPPER].splitlines(True), r[WRAPPER].splitlines(True)
    hunks = []
    for tag, i, j, x, y in difflib.SequenceMatcher(a=a, b=z, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        before, after = b''.join(a[i:j]), b''.join(z[x:y])
        reverse = old.replace(reverse, before, after)
        hunks.append(dict(before=before.decode(), after=after.decode()))
    reverse = versioned(reverse, r6.VERSION, r6.SLUG)
    old.need(reverse == files[WRAPPER], 'reverse merge order differs')
    old.need({k for k in p if p[k] != files[k]} == {SOURCE, WRAPPER, r6.PORT, r6.CHAIN}, 'parent scope')
    old.need({k for k in r if r[k] != files[k]} == {SOURCE, WRAPPER}, 'R6 scope')
    return files, dict(both_merge_orders_equal=True, r6_wrapper_hunks=hunks,
        mirror_source_sha256=old.digest(m[SOURCE]), mirror_source_byte_identical=True,
        r6_port_and_chain_byte_identical=True, helper_byte_identical=True)


def payload(s=None):
    s = s or snapshots()
    p = s['parent']
    files, proof = merge(s)
    nb = json.loads(json.dumps(p['nb']))
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(p['blob'])) as src, zipfile.ZipFile(buf, 'w') as dst:
        for info in src.infolist():
            dst.writestr(info, files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before, after in [('BLOB = ' + repr(p['assignments']['BLOB']), 'BLOB = ' + repr(base64.b64encode(blob).decode())),
        ('PINS = ' + repr(p['assignments']['PINS']), 'PINS = ' + repr({k: old.digest(v) for k, v in files.items()})),
        (repr(old.digest(p['blob'])), repr(old.digest(blob)))]:
        code = old.replace(code, before, after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    meta = dict(s['r6']['meta'], id=SLUG, title=SLUG.split('/')[1])
    for name, data in files.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    return nb, meta, files, blob, proof


def registration(s=None):
    s = s or snapshots()
    review = WORK / 'PRE-REVIEW.md'
    text = review.read_text(encoding='utf-8')
    spec_sha = old.digest((WORK / 'SPEC.md').read_bytes())
    old.need(text.rstrip().endswith(('PASS', 'PASS-WITH-NOTES')) and spec_sha in text
             and 'distinct from builder' in text, 'independent prereview required')
    return dict(item='H-B-179', version=VERSION, kernel_id=SLUG,
        purpose='Test joint pinned R6 FULL-flip and admitted R224 anatomical-mirror composition; gain is a hypothesis.',
        sources={k: dict(stage=str(STAGES[k].relative_to(ROOT)), kernel_files=v['kernel_files']) for k, v in s.items()},
        spec_sha256=spec_sha, prereview_sha256=old.digest(review.read_bytes()),
        change='H174 R6 hunks plus byte-identical admitted H170 mirror hunk and dependent pins/version/id',
        keep_score_min=.956, keep_rule='User-provided score >= .956 and independently reviewed runtime <= 32400 seconds',
        parent_hidden_submission=57224580, exposure='EXPOSED', clean_oof=False,
        dataset_sources=s['r6']['meta']['dataset_sources'], manifest_sha256=r6.MANIFEST_SHA,
        runtime=dict(parent_hidden_upper_hours=6.73,
            HB170_N32_mirror_increment_seconds=None, HB174_N32_R6_increment_seconds=None,
            equation='6.73*3600 + HB170_N32_mirror_increment_seconds + HB174_N32_R6_increment_seconds',
            limit_seconds=32400, status='PENDING_COMPLETED_REVIEWED_TIMING_AND_REAL_OUTPUT_POSTRUN_REPLAY',
            assumption='Additive local-4070-to-hidden cost transfer; no speedup credit; combined GPU/Linux interaction unmeasured'),
        qualified_for_submission=False, gate='No gate copy; B must obtain independent final review, timing postchecks and qualification')


def artifacts():
    s = snapshots()
    nb, meta, files, blob, proof = payload(s)
    result = {'kernel/release.ipynb': old.encoded(nb), 'kernel/kernel-metadata.json': old.encoded(meta),
              'registration.json': gate.encoded(registration(s)), 'merge-proof.json': gate.encoded(proof)}
    inventory = {}
    for key, ref in s.items():
        manifest = dict(changed=[], unchanged=[], added=[], removed=[])
        full = 'ALL 31 EMBEDDED FILES, NOTEBOOK AND METADATA; NO ADDED/REMOVED MEMBERS\n'
        for name, data in sorted(files.items()):
            before = ref['files'][name]
            manifest['unchanged' if before == data else 'changed'].append(dict(name=name,
                before_sha256=old.digest(before), after_sha256=old.digest(data)))
            full += f'{name}: {old.digest(before)} -> {old.digest(data)} ' + ('IDENTICAL' if before == data else 'REGISTERED_CHANGE') + '\n'
            full += mirror.r224.diff(before, data, key + '/' + name, VERSION + '/' + name)
        for name, before, after in [('release.ipynb', ref['nb'], nb), ('kernel-metadata.json', ref['meta'], meta)]:
            full += mirror.r224.diff(old.encoded(before), old.encoded(after), key + '/' + name, VERSION + '/' + name)
        inventory[key] = manifest
        result['full-diff-vs-' + key + '.txt'] = full.encode()
    result['embedded-diff.json'] = gate.encoded(inventory)
    return result, files, blob


def register():
    old.write(WORK / 'registration.json', gate.encoded(registration()))
    return dict(status='REGISTERED_BEFORE_EXECUTION', keep_score_min=.956)


def prepare():
    result, files, blob = artifacts()
    old.need((WORK / 'registration.json').read_bytes() == result['registration.json'], 'register first')
    old.need(not (OUT / 'kernel').exists(), 'refuse existing stage overwrite')
    for key, path in STAGES.items():
        for name in ('kernel/release.ipynb', 'kernel/kernel-metadata.json'):
            old.write(BACKUP / key / name, (path / name).read_bytes())
    for name, data in result.items():
        old.write(OUT / name, data)
    for name, data in files.items():
        old.write(WORK / 'runtime' / name, data)
    rec = dict(status='PREPARED_CPU_ONLY', builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], archive_sha256=old.digest(blob),
        embedded={k: old.digest(v) for k, v in files.items()}, qualified_for_submission=False)
    old.write(OUT / 'build-receipt.json', gate.encoded(rec))
    old.write(OUT / 'CHANGELOG.md', b'# v1.1.32-r224r6mirfq / builder 1.0.0\nCompose H174 R6 with byte-identical admitted H170 R224 mirror. Four changed/27 identical members vs H167; two changed/29 identical vs H174. No launch, blend, O, FAST, pool or thread change. Sources untouched; backups .local/backups/hb179. CPU only; no gate binding.\n')
    return validate()


def validate():
    result, files, blob = artifacts()
    for name, data in result.items():
        old.need((OUT / name).read_bytes() == data, 'stage drift: ' + name)
    rec = old.read(OUT / 'build-receipt.json')
    old.need(rec['builder_sha256'] == old.digest(Path(__file__).read_bytes()), 'builder drift')
    old.need(rec['archive_sha256'] == old.digest(blob) and rec['embedded'] == {k: old.digest(v) for k, v in files.items()}, 'embedded receipt drift')
    old.need(rec['kernel_files'] == gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], 'kernel drift')
    return dict(status='PASS_CPU_VALIDATE', kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),
                parent_changed=4, parent_identical=27, r6_changed=2, r6_identical=29)


def decision(score, mirror_increment, r6_increment):
    vals = (score, mirror_increment, r6_increment)
    old.need(all(type(v) in (int, float) and math.isfinite(v) for v in vals), 'completed finite measurements required')
    old.need(0 <= score <= 1 and mirror_increment >= 0 and r6_increment >= 0, 'decision domain; no speedup credit')
    return score >= .956 and 6.73 * 3600 + mirror_increment + r6_increment <= 32400


def replay():
    from tools import c85r7_r224r6mir_cpu as cpu
    return cpu.run()


def admit():
    valid = validate()
    rec = old.read(WORK / 'cpu-replay.json')
    old.need(rec['status'] == 'PASS_REAL_TWO_STUDY_COMPOSITION' and rec['kernel_sha256'] == valid['kernel_sha256'], 'fresh CPU replay required')
    for name, digest in rec['evidence_sha256'].items():
        old.need(old.digest((ROOT / name).read_bytes()) == digest, 'replay evidence drift: ' + name)
    return dict(status='HOLD_FINAL_REVIEW_TIMING_POSTRUN_REPLAY_AND_QUALIFICATION',
                kernel_sha256=valid['kernel_sha256'], qualified_for_submission=False, gate_bound=False)


def main():
    cpu_only()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register', 'prepare', 'validate', 'cpu-replay', 'admit'])
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.command in ('prepare', 'admit'):
        old.need(a.dry_run, 'CPU dry-run required')
    print(json.dumps({'register': register, 'prepare': prepare, 'validate': validate, 'cpu-replay': replay, 'admit': admit}[a.command](), indent=2))


if __name__ == '__main__':
    main()
