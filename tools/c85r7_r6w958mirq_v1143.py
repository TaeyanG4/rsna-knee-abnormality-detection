"""H-B-195 builder 1.0.0: admitted R-224 anatomical mirror TTA on admitted H-B-189.

Changelog 1.0.0 (claude-opus-b6, 2026-10-17): one candidate on the admitted
H-B-189 0.958-weights parent. raptor224_run.py is replaced by the admitted
H-B-179 member (byte-identical to the admitted H-B-170 member); the wrapper
changes only its raptor224 pin and VERSION/OUT/KERNEL_ID/RELEASE. The blend
helper and every other member stay byte-identical to H-B-189. Reuses reviewed
H189/H187/H179 infrastructure without editing it. The CPU replay lives in
tools/c85r7_r6w958mirq_cpu_v1143.py. No frozen/scored/original edits, no
GPU/Kaggle/upload.
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

# A present-empty value is required; PowerShell assignment of '' deletes it.
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r6w958q_v1139 as h189
from tools import c85r7_stackfxq_v1137 as h187

old, fast, rbuild, gate = h189.old, h189.fast, h189.rbuild, h189.gate
reuse = h189.reuse
HELPER, WRAPPER = h189.HELPER, h189.WRAPPER
SOURCE = 'raptor224_run.py'
MEMBERS, PRIVATE_DATASETS = h189.MEMBERS, h189.PRIVATE_DATASETS
VERSION = '1.1.43-r6w958mirq'
SLUG = 'taeyangg4/knee-lb-r6w958mirq-v1'
PARENT_VERSION = '1.1.39-r6w958q'
PARENT_SLUG = 'taeyangg4/knee-lb-r6w958q-v1'
BASE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q'
PARENT_GATE = ROOT / '.local/scratch/astra-b5/hb189-gate/gate-copy-v145'
RECEIPT = ROOT / '.local/state/gates/knee-lb-r6w958q-v1.json'
PARENT_KERNEL = 'c76b1ddb4d83110ee6556f5f5282ad843af9f6d0cbc932dc6630e3868d4e084f'
PARENT_RECEIPT = '8781c905cc86a7ad35bdaf49720796631e352ce8b58e5e0532bbe6ccbadbd7a6'
PLAIN_SHA = 'bc29b23a7aac8ace281619b71c8b49d8c4ac83e645d4e29e457e7e83f477fe2c'   # H174/H189 raptor224_run.py
MIRROR_SHA = '699ee7d15e21d62fe83df76810f0949bdcda79f28b19c14ff67f931c2b78f428'  # admitted H179 = H170 member
RELEASE_PARENT = 'C85R7+codecs R224 R6 FLIP_ALL per-target W958'
RELEASE = RELEASE_PARENT + ' R224 mirror'
# Admitted mirror references (authenticated; their members are only read).
REFERENCES = {
    'h179': dict(stage=ROOT / '.local/scratch/astra-b5/hb179-gate/gate-copy-v145',
                 source=ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.32-r224r6mirfq',
                 receipt=ROOT / '.local/state/gates/knee-lb-r224r6mirfq-v1.json',
                 receipt_sha='10722ccd8232c23cd7b87d8f693da568b8c3e024911cb69e917da1411fe6a1bf',
                 kernel='baf0acc4651d4a9aecb5ae18c4bb60d46fe42e76e0181d3d3620d5af2342972b',
                 version='1.1.32-r224r6mirfq', slug='taeyangg4/knee-lb-r224r6mirfq-v1'),
    'h170': dict(stage=ROOT / '.local/scratch/astra-b5/hb170-gate/gate-copy-v144',
                 source=ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.26-r224mirq',
                 receipt=ROOT / '.local/state/gates/knee-lb-r224mirq-v1.json',
                 receipt_sha='5cb51fea0a688b42233da8651042fa5dfc15edda82ec0f91c6c214863558273b',
                 kernel='b7e0e38a610041c3782b3149667704647c50ebd1dcf4eb78f097c40c764156f9',
                 version='1.1.26-r224mirq', slug='taeyangg4/knee-lb-r224mirq-v1')}
H174_VERSION, H174_SLUG = '1.1.30-r224r6fq', 'taeyangg4/knee-lb-r224r6fq-v1'
OUT = old.BASE.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb195-build/hb195'
BACKUP = ROOT / '.local/backups/hb195-build/hb195-parent'
SPEC = SCRATCH / 'SPEC.md'
FABLE = h189.FABLE
DIFF = 'full-diff-vs-v1.1.39-r6w958q.txt'
PROOF = 'reference-proof.json'
MERGE = 'merge-proof.json'
WEIGHTS = h189.WEIGHTS
W_COLUMNS = (5, 6, 8, 9)                                      # H-B-189 weights != .5
HALF_COLUMNS = tuple(i for i in range(12) if i not in W_COLUMNS)  # == H-B-179 runtime on same inputs
MIRROR_TIMING = ROOT / '.local/scratch/hb170-build/hb170/timing/measurement.json'
MIRROR_TIMING_SHA = '7957b9cde0781c43958f85410fd007ff118345b92f1097ddc650dd6ea35ea659'
PARENT_HIDDEN_UPPER_SECONDS = 7*3600 + 13*60 + 59  # 09:02:00 -> 16:15:59 KST worst-case minute rounding
LIMIT_SECONDS = 32400


def authenticated(stage, source, receipt, receipt_sha, kernel_sha):
    """Reviewed H-B-187 admitted-stage authentication (receipt, inventory, source bytes)."""
    return h187.authenticated(stage, source, receipt, receipt_sha, kernel_sha)


def parent():
    value = authenticated(PARENT_GATE, BASE, RECEIPT, PARENT_RECEIPT, PARENT_KERNEL)
    old.need(value[1] == h189.patched(h189.parent()[1]), 'admitted H189 != H189 weights patch of admitted H174')
    old.need(old.digest(value[1][SOURCE]) == PLAIN_SHA, 'parent raptor224 member is not the plain H174 member')
    old.need(old.read(BASE/'kernel/kernel-metadata.json')['dataset_sources'] == PRIVATE_DATASETS,
             'fixed private R6 dataset list drift')
    return value


def reference(label):
    ref = REFERENCES[label]
    return authenticated(ref['stage'], ref['source'], ref['receipt'], ref['receipt_sha'], ref['kernel'])


def mirror_member():
    members = [reference(label)[1][SOURCE] for label in ('h179', 'h170')]
    old.need(members[0] == members[1] and old.digest(members[0]) == MIRROR_SHA,
             'admitted H179/H170 mirror members differ or drifted')
    return members[0]


def versioned(raw):
    """H-B-189 wrapper version strings -> child; nothing else."""
    for before, after in [
            ('VERSION = "' + PARENT_VERSION + '"', 'VERSION = "' + VERSION + '"'),
            ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + PARENT_VERSION + '"',
             'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + VERSION + '"'),
            ('KERNEL_ID = "' + PARENT_SLUG + '"', 'KERNEL_ID = "' + SLUG + '"'),
            ('RELEASE = "' + RELEASE_PARENT + '"', 'RELEASE = "' + RELEASE + '"')]:
        raw = old.replace(raw, before, after)
    return raw


def patched(original, mirror=None):
    files = dict(original)
    old.need(old.digest(original[SOURCE]) == PLAIN_SHA, 'parent raptor224 member drift')
    files[SOURCE] = mirror_member() if mirror is None else mirror
    files[WRAPPER] = versioned(old.replace(original[WRAPPER], PLAIN_SHA, old.digest(files[SOURCE])))
    return files


def mirror_hunks(before, after):
    """Added/replacement text of every non-equal line opcode (the mirror hunk blocks)."""
    a, z = before.splitlines(True), after.splitlines(True)
    return [b''.join(z[x:y]) for tag, i, j, x, y in
            difflib.SequenceMatcher(a=a, b=z, autojunk=False).get_opcodes() if tag != 'equal']


def merge_proof():
    """Forward (H189 + mirror member) == reverse (H179 + H189 helper/wrapper hunks)."""
    p189 = parent()[1]
    h174 = h189.parent()[1]
    h179 = reference('h179')[1]
    forward = patched(p189)
    old.need({k for k in h174 if h174[k] != h179[k]} == {SOURCE, WRAPPER}, 'H179 scope vs H174')
    old.need({k for k in h174 if h174[k] != p189[k]} == {HELPER, WRAPPER}, 'H189 scope vs H174')
    reverse = dict(h179)
    reverse[HELPER] = p189[HELPER]
    w = h179[WRAPPER]
    for before, after in [
            ('VERSION = "' + REFERENCES['h179']['version'] + '"', 'VERSION = "' + H174_VERSION + '"'),
            ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + REFERENCES['h179']['version'] + '"',
             'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + H174_VERSION + '"'),
            ('KERNEL_ID = "' + REFERENCES['h179']['slug'] + '"', 'KERNEL_ID = "' + H174_SLUG + '"')]:
        w = old.replace(w, before, after)
    old.need(w == old.replace(h174[WRAPPER], PLAIN_SHA, MIRROR_SHA), 'H179 wrapper != H174 wrapper + mirror pin')
    a, z = h174[WRAPPER].splitlines(True), p189[WRAPPER].splitlines(True)
    hunks = []
    for tag, i, j, x, y in difflib.SequenceMatcher(a=a, b=z, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        before, after = b''.join(a[i:j]), b''.join(z[x:y])
        w = old.replace(w, before, after)
        hunks.append(dict(before=before.decode(), after=after.decode()))
    reverse[WRAPPER] = versioned(w)
    old.need(reverse == forward, 'merge orders differ')
    old.need({k for k in p189 if p189[k] != forward[k]} == {SOURCE, WRAPPER}, 'child scope vs H189')
    old.need({k for k in h179 if h179[k] != forward[k]} == {HELPER, WRAPPER}, 'child scope vs H179')
    old.need(mirror_hunks(p189[SOURCE], forward[SOURCE]) == mirror_hunks(h174[SOURCE], h179[SOURCE]),
             'child mirror hunk differs from admitted H174->H179 hunk')
    return dict(both_merge_orders_equal=True, mirror_source_sha256=MIRROR_SHA, plain_source_sha256=PLAIN_SHA,
                mirror_source_byte_identical_to=['H-B-179 hb179-gate/gate-copy-v145', 'H-B-170 hb170-gate/gate-copy-v144'],
                helper_sha256=old.digest(p189[HELPER]), helper_byte_identical_to_h189=True,
                h189_wrapper_hunks_applied_to_h179=hunks, identical_to_h189=29, identical_to_h179=29)


def reference_proof():
    child = patched(parent()[1])[SOURCE]
    hunks = mirror_hunks(parent()[1][SOURCE], child)
    old.need(len(hunks) == 4, 'expected exactly four mirror hunks')
    blocks = []
    for label in ('h179', 'h170'):
        member = reference(label)[1][SOURCE]
        stage = REFERENCES[label]['stage'].relative_to(ROOT).as_posix()
        for text in [member] + hunks:
            old.need(member.count(text) == 1 and child.count(text) == 1, 'mirror block not unique in ' + label)
            blocks.append(dict(reference_stage=stage, reference_member=SOURCE, child_member=SOURCE, text=text.decode()))
    return dict(blocks=blocks)


def timing():
    old.pin(MIRROR_TIMING, MIRROR_TIMING_SHA)
    measured = old.read(MIRROR_TIMING)
    r = measured['result']
    old.need(measured['identity'] == 'HB170_N32' and r['failed_study_fallbacks'] == 0, 'HB170_N32 measurement identity')
    conservative = r['projected_hidden_increment_seconds']
    scale = conservative / r['conservative_increment_seconds']
    inference_only = r['inference_only_increment_seconds'] * scale
    return dict(
        basis='H-B-189 hidden COMPLETE + measured HB170_N32 mirror increment (additive)',
        parent_submission=57287199, parent_submitted_kst='2026-10-17 09:02',
        parent_complete_observed_kst='2026-10-17 16:15 (API read, D-B-213)',
        parent_hidden_upper_seconds=PARENT_HIDDEN_UPPER_SECONDS,
        parent_hidden_note='upper bound incl. queueing; minute-rounded submit/read times taken worst-case',
        mirror_increment=dict(source=MIRROR_TIMING.relative_to(ROOT).as_posix(), sha256=MIRROR_TIMING_SHA,
            local_n32_conservative_seconds=r['conservative_increment_seconds'],
            local_n32_inference_only_seconds=r['inference_only_increment_seconds'],
            projected_hidden_conservative_seconds=conservative,
            projected_hidden_inference_only_seconds=inference_only,
            per_target_spearman_plain_vs_mirror=[min(r['per_target_spearman']), max(r['per_target_spearman'])],
            review='HB170_N32 result review astra-6 PASS (handoff 2026-10-15 21:xx)',
            limitation=r['limitation']),
        projected_hidden_seconds_conservative=PARENT_HIDDEN_UPPER_SECONDS + conservative,
        projected_hidden_seconds_inference_only=PARENT_HIDDEN_UPPER_SECONDS + inference_only,
        limit_seconds=LIMIT_SECONDS,
        runtime_pass=PARENT_HIDDEN_UPPER_SECONDS + conservative <= LIMIT_SECONDS,
        cross_check='H-B-179 (H-B-174 + same mirror) review projection 7.31 h',
        assumption='H-B-189 hidden already includes the plain R224 forward; the mirror adds one more R224 forward per '
                   'study; HB170_N32 local-4070-to-hidden scale; no speedup credit; Linux/T4 mirror forward unmeasured')


def registration():
    parent(); reference_proof(); merge_proof()
    return dict(item='H-B-195', version=VERSION, kernel_id=SLUG,
        purpose='Complete the public 0.958-recipe transfer: add the admitted R-224 anatomical mirror TTA '
                '(byte-identical H-B-179/H-B-170 raptor224_run.py) to admitted H-B-189; own O branch kept',
        parent=PARENT_GATE.relative_to(ROOT).as_posix(), parent_source_stage=BASE.relative_to(ROOT).as_posix(),
        parent_kernel_sha256=PARENT_KERNEL, parent_receipt_sha256=PARENT_RECEIPT,
        parent_submission=57287199, parent_displayed_public=.958,
        reference_kernels={k: v['kernel'] for k, v in REFERENCES.items()},
        reference_receipts_sha256={k: v['receipt_sha'] for k, v in REFERENCES.items()},
        changed_member=SOURCE, mirror_member_sha256=MIRROR_SHA, parent_member_sha256=PLAIN_SHA,
        formula=h189.FORMULA + '; R224 = 0.5*(p_plain + p_mirror), uint8 anatomical mirror before crop '
                '(H-B-170 raptor224_run.py)',
        weights=list(WEIGHTS), blend_helper='exp169_r224.py byte-identical to H-B-189 (no blend change)',
        columns_affected='all 12 through the R224 member input only; columns 0-4,7,10,11 follow the H-B-179 '
                         'blend, 5,6,8,9 the H-B-189 weights',
        pre_execution=dict(spec_sha256=old.digest(SPEC.read_bytes()), spec_path=SPEC.relative_to(ROOT).as_posix(),
            review='one combined independent review dispatched by B after the build (process lightening 28(a): '
                   'byte-patch child of a hidden-proven parent with only admitted hunks)'),
        preregistration='registered before prepare, stage tests, neural parity, cached replay and the descriptive '
                        'Gold58 delta; nothing registered is chosen after a result (all fixed by B\'s build prompt).',
        failure_rule='H-B-189 / H-B-179 unchanged: an exception in the plain or mirror window/forward leaves the '
                     'study\'s R224 row at the 0.5 sentinel (raptor224_run.py FALLBACK); the blend marks all-0.5 rows '
                     'R224-failed (success-only ranks, rr = r(R384) for that row); >3% R224 failures abort. '
                     'Own 17/17/12 survivor renormalisation and all other overrides unchanged.',
        dataset_sources=PRIVATE_DATASETS,
        dataset_sources_note='identical to H-B-189 (and to H-B-179, which ran this mirror member with this list); '
                             'the mirror reuses the R224 checkpoint 3394fdd7 already attached via '
                             'nartaa/rsna-knee-publication-swa-weights-20261007; no new dataset',
        public_reading=dict(reference_value=.958, expected=.958, upside=.959,
            rule='RETAIN iff displayed >= 0.958 (= H-B-189 displayed); DROP iff displayed <= 0.957',
            written='before submission (this registration)', source='user-supplied readings only; no score polling'),
        timing=timing(),
        gold58=dict(exposure='EXPOSED', use='descriptive only, not a gate', weak_four=[3, 5, 6, 8],
            replay=FABLE.relative_to(ROOT).as_posix(), replay_sha256=old.digest(FABLE.read_bytes()),
            mirror_predictions='HB170_GOLD58_V107 measured 4070 fp16 plain/mirror pair (admitted H-B-170 raptor '
                               'functions = child member bytes); result review astra-1 PASS',
            comparison='actual embedded child (mirror R224) vs H-B-189 (plain R224), healthy 58 studies'),
        sources=['Taeyang 2026-10-17 16:1x: think in order of likelihood and submit (handoff 16:21, slot 5)',
                 'public 0.958 notebooks run R-224 mirror on both Raptor resolutions (H-B-189 SPEC source notes)',
                 'Gold58 mirror HB170-HB167 +0.000017 macro (exposed, ~0); public H-B-170 mirror tie'],
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        runtime_qualification='CPU real R224 neural parity (2 visible studies) and cached-serving arithmetic; no '
                              'CUDA/T4/Linux mirror forward or hidden-runtime measurement',
        gate_binding='B must pin slug-generic strict_variant_gate_r224_v145 with authentic dispatcher '
                     'builder/reviewer receipts and the stage reference-proof.json')


def decision(displayed):
    old.need(type(displayed) in (int, float) and math.isfinite(displayed) and 0 <= displayed <= 1,
             'finite displayed public score required')
    return 'RETAIN' if displayed >= .958 else 'DROP'


def payload():
    nb, original, old_blob, assignments = parent()
    files = patched(original)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buf, 'w') as target:
        for info in source.infolist(): target.writestr(info, files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before, after in [
            ('BLOB = ' + repr(assignments['BLOB']), 'BLOB = ' + repr(base64.b64encode(blob).decode())),
            ('PINS = ' + repr(assignments['PINS']), 'PINS = ' + repr({k: old.digest(v) for k, v in files.items()})),
            (repr(old.digest(old_blob)), repr(old.digest(blob)))]:
        code = old.replace(code, before, after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    meta = old.read(BASE/'kernel/kernel-metadata.json')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta, files, blob


def scope(nb, meta):
    expected, em, files, blob = payload()
    old.need(nb == expected and meta == em, 'forbidden notebook/metadata diff')
    actual, ab, _ = old.unpack(fast.restored(nb))
    old.need(actual == files and ab == blob, 'forbidden embedded diff')
    original = parent()[1]
    old.need(set(files) == set(original) and {k for k in files if files[k] != original[k]} == {SOURCE, WRAPPER},
             'member scope')
    old.need(files[HELPER] == original[HELPER], 'blend helper must stay byte-identical to H-B-189')
    for name, data in files.items():
        if name.endswith('.py'): compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))
    return files, blob


def artifacts():
    nb, meta, files, blob = payload()
    pnb, original, _, _ = parent()
    manifest = dict(version=VERSION, added=[], removed=[], changed=[], unchanged=[])
    full = 'ALL 31 EMBEDDED FILES: 29 IDENTICAL, 2 REGISTERED CHANGES; NO ADDED/REMOVED MEMBERS\n'
    for k in sorted(files):
        a, b = old.digest(original[k]), old.digest(files[k])
        manifest['changed' if a != b else 'unchanged'].append(dict(name=k, before_sha256=a, after_sha256=b))
        full += f'{k}: {a} -> {b} ' + ('REGISTERED_CHANGE' if a != b else 'IDENTICAL') + '\n'
        full += rbuild.diff(original[k], files[k], PARENT_VERSION + '/' + k, VERSION + '/' + k)
    for name, before, after in [('notebook', pnb, nb), ('kernel-metadata.json', old.read(BASE/'kernel/kernel-metadata.json'), meta)]:
        full += rbuild.diff(old.encoded(before), old.encoded(after), 'parent/' + name, 'candidate/' + name)
    return nb, meta, files, blob, manifest, full.encode()


CHANGELOG = b'''# v1.1.43-r6w958mirq
H-B-195: admitted H-B-189 (0.958 per-target weights on H174 R6) with the
admitted R-224 anatomical mirror TTA: raptor224_run.py byte-identical to the
admitted H-B-179 / H-B-170 member (sha 699ee7d1). Wrapper: raptor224 pin and
VERSION/OUT/KERNEL_ID/RELEASE only. Blend helper, w vector, outer .85/.15, own,
R6 FULL flip, FAST, serving, datasets unchanged; 29/31 embedded files identical.
Parent backup .local/backups/hb195-build/hb195-parent. EXPOSED. Keep displayed
>=.958 (= H-B-189); expected .958 (.959 upside).
'''


def register():
    data = old.encoded(registration())
    path = OUT/'registration.json'
    if path.exists(): old.need(path.read_bytes() == data, 'immutable registration drift')
    else:
        old.need(not (OUT/'kernel').exists(), 'registration must precede kernel')
        old.write(path, data)
    old.write(OUT/'registration.sha256', (old.digest(data) + '\n').encode())
    return dict(status='REGISTERED_BEFORE_EVALUATION', sha256=old.digest(data), exposure='EXPOSED')


def stage_files():
    nb, meta, files, blob, manifest, full = artifacts()
    written = [('kernel/release.ipynb', old.encoded(nb)), ('kernel/kernel-metadata.json', old.encoded(meta)),
               ('CHANGELOG.md', CHANGELOG), ('new-members.json', old.encoded(manifest)), (DIFF, full),
               ('SPEC.md', SPEC.read_bytes()), (PROOF, old.encoded(reference_proof())),
               (MERGE, old.encoded(merge_proof()))]
    return written, nb, meta, files, blob


def prepare():
    old.need((OUT/'registration.json').read_bytes() == old.encoded(registration()), 'register first')
    written, nb, meta, files, blob = stage_files()
    scope(nb, meta)
    for name in ('kernel/release.ipynb', 'kernel/kernel-metadata.json'):
        old.write(BACKUP/name, (BASE/name).read_bytes())
    old.write(BACKUP/'parent-admission-receipt.json', RECEIPT.read_bytes())
    for name, data in written: old.write(OUT/name, data)
    for label, contents in [('candidate', files), ('parent', parent()[1]), ('h179', reference('h179')[1])]:
        for name, data in contents.items(): old.write(SCRATCH/(label + '-runtime')/name, data)
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], archive_sha256=old.digest(blob),
        embedded={k: old.digest(v) for k, v in files.items()}, unchanged=29, changed=[SOURCE, WRAPPER],
        added=[], removed=[], parent_kernel_sha256=PARENT_KERNEL, parent_receipt_sha256=PARENT_RECEIPT,
        reference_proof_sha256=old.digest(old.encoded(reference_proof())),
        merge_proof_sha256=old.digest(old.encoded(merge_proof())),
        qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    old.write(OUT/'build-receipt.json', old.encoded(rec))
    return validate()


def validate():
    written, _, _, files, blob = stage_files()
    for name, data in [('registration.json', old.encoded(registration()))] + written:
        old.need((OUT/name).read_bytes() == data, 'stage drift: ' + name)
    scope(old.read(OUT/'kernel/release.ipynb'), old.read(OUT/'kernel/kernel-metadata.json'))
    rec = old.read(OUT/'build-receipt.json')
    old.need(rec['builder_sha256'] == old.digest(Path(__file__).read_bytes()), 'builder drift')
    old.need(rec['embedded'] == {k: old.digest(v) for k, v in files.items()} and rec['archive_sha256'] == old.digest(blob),
             'archive drift')
    old.need(rec['kernel_files'] == gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], 'kernel drift')
    old.need(rec['qualified_for_submission'] is False and rec['unchanged'] == 29, 'receipt scope')
    from tools import strict_variant_gate_r224_v145 as generic
    pins = gate.Inputs(ROOT)
    child = gate.kernel(OUT, pins)
    generic.metadata_delta(gate.kernel(PARENT_GATE, pins), child, SLUG, registration())
    proofs = generic.reference_blocks(OUT/PROOF, child, pins)  # the gate's own reference-proof check
    old.need(len(proofs) == 10, 'reference proof blocks')
    pins.recheck()
    return dict(status='PASS_CPU_VALIDATE', kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),
                identical=29, changed=2, reference_blocks=len(proofs), both_merge_orders_equal=True,
                metadata_raw_values_identical_except_id_title=True)


def math_functions(files):
    """Test-only extraction; the real replay imports the actual embedded archives."""
    return reuse.math_functions(files)


def oracle(arrays, masks):
    """Independent column-wise pandas oracle of the unchanged H-B-189 formula."""
    return h189.oracle(arrays, masks)


def admit():
    from tools import c85r7_r6w958mirq_cpu_v1143 as cpu
    valid = validate()
    smoke, neural = old.read(SCRATCH/'replay.json'), old.read(SCRATCH/'neural-parity.json')
    old.need(smoke['status'] == 'PASS' and smoke['kernel_sha256'] == valid['kernel_sha256'], 'CPU replay missing/stale')
    old.need(neural['status'] == 'PASS_REAL_TWO_STUDY_MIRROR_NEURAL_PARITY' and neural['kernel_sha256'] == valid['kernel_sha256'],
             'neural parity missing/stale')
    for value in (smoke, neural):
        old.need(value['replay_tool_sha256'] == old.digest(Path(cpu.__file__).read_bytes()), 'replay tool drift')
    old.need(smoke['all12_vs_h189_float64_csv_exact'] and smoke['half_columns_vs_h179_float64_csv_exact']
             and smoke['studies'] == [2, 58], 'CPU parity missing')
    return dict(status='HOLD_B_SLUG_GENERIC_GATE_AND_INDEPENDENT_REVIEW', qualified_for_submission=False,
                kernel_sha256=valid['kernel_sha256'], gate='tools/strict_variant_gate_r224_v145.py',
                reference_proof=(OUT/PROOF).relative_to(ROOT).as_posix())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register', 'prepare', 'validate', 'cpu-replay', 'admit'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.command in ('prepare', 'admit'): old.need(args.dry_run, 'CPU dry-run required')
    if args.command == 'cpu-replay':
        from tools import c85r7_r6w958mirq_cpu_v1143 as cpu
        result = cpu.run()
    else:
        result = {'register': register, 'prepare': prepare, 'validate': validate, 'admit': admit}[args.command]()
    old.write(SCRATCH/(args.command + '-result.json'), old.encoded(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
