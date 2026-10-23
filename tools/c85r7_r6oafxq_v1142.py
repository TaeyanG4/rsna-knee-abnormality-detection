"""H-B-193 builder 1.0.0: admitted R-only Fracture route on the auto-selected H-B-181.

Changelog 1.0.0: routed-from-R column set [] -> [11] on H-B-181 (R224 + FULL flip +
R6 compact + C1 hard OA). The three hunks are the admitted H-B-185 fluid hunks
with the selector restricted [7,9] -> [11], which equals the admitted H-B-187
R-only Fracture rule restricted [7,9,11] -> [11]. Dependent pins/version/id only.
No frozen edits, launch/import/pool/thread/timing changes or added inference.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

# A present-empty value is required; PowerShell assignment of '' deletes it.
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_stackq_v1125 as infrastructure

old, fast, rbuild, gate = (infrastructure.old, infrastructure.fast,
                          infrastructure.rbuild, infrastructure.gate)
BUILDER_VERSION = '1.0.0'
VERSION = '1.1.42-r6oafxq'
SLUG = 'taeyangg4/knee-lb-r6oafxq-v1'
RELEASES = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7'
BASE = RELEASES / 'v1.1.34-r224r6oaq'
PARENT = dict(gate=ROOT / '.local/scratch/astra-b5/hb181-gate/gate-copy-v145', source=BASE,
              receipt=ROOT / '.local/state/gates/knee-lb-r224r6oaq-v1.json',
              receipt_sha='a732818c76b9ee2bec998de2d088cae7e38f5fe676f508a036e6926329cdb77c',
              kernel='b24cfd0205bcadd365fdab92cb775676bc2763969ca0cd4eee610e0f932d1517')
# Admitted R-only Fracture rule (H-B-187 on H-B-168, routed [7,9,11]).
FX = dict(gate=ROOT / '.local/scratch/astra-b5/hb187-gate/gate-copy-v145', source=RELEASES / 'v1.1.37-stackfxq',
          receipt=ROOT / '.local/state/gates/knee-lb-stackfxq-v1.json',
          receipt_sha='330a5f0ac5e4bdc37f792f61c906eb9b38e0819d422c623d17e79e8eb2475fe3',
          kernel='c61907d2e98ec00038ff5fba4bfe1eb21e41d292ee5ebfa6884e7b9b5349d4d6')
# Admitted R6-line port of the same route code (H-B-185 on H-B-181, routed [7,9]).
SIBLING = dict(gate=ROOT / '.local/scratch/astra-b5/hb185-gate/gate-copy-v145', source=RELEASES / 'v1.1.36-r6stackoaq',
               receipt=ROOT / '.local/state/gates/knee-lb-r6stackoaq-v1.json',
               receipt_sha='7153b30a5474418f4ccb4c5c77ee9764f363c8a4521120e49e563cc2a8ad930c',
               kernel='8d1599e51a1bdf4b5170b660838f5b6b493336e1420afd9d3fd75f96b2f3b517')
OUT = RELEASES / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb193-build/hb193'
BACKUP = ROOT / '.local/backups/hb193-build/hb193-parent'
SPEC = SCRATCH / 'SPEC.md'
SPEC_SHA = '16084bbf52948b8f44806be524ac71eff077873ffd1f547c3bb357b3adbca16b'
HELPER, WRAPPER = infrastructure.HELPER, infrastructure.WRAPPER
DIFF = 'full-diff-vs-v1.1.34-r224r6oaq.txt'
PROOF = 'fracture-reference-proof.json'
FORMULA = ('0.85*r(P)+0.15*r(O); r_R=0.65*r(R384)+0.35*r(R224); '
           'P_t=r(R6flip)_t for Lateral OA and PF OA when compact succeeds, else r_R_t; '
           'P_t=r_R_t for Fracture, with R-only failure eligibility; other P=0.5*r_R+0.5*r(R6flip); '
           'O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46')
PROTECTED = tuple(range(11))
ROUTED = (11,)
MEMBERS = ('R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q_A5q')
SELECTOR, SIBLING_SELECTOR, FX_SELECTOR = '[11]', '[7,9]', '[7,9,11]'
# The three child hunks (insertions into H-B-181). Each equals an admitted H-B-185
# hunk with [7,9] -> [11] and an admitted H-B-187 hunk with [7,9,11] -> [11].
HUNKS = ('    public[failed], public[sf], public[failed & sf] = ss[failed], rr[sf], .5\n'
         '    public[:, [11]] = rr[:, [11]]\n',
         '    rr, rf = r_half(r, r224, rf, f224)\n'
         '    rr = c85.scored_rank(rr, rf)\n'
         '    routed = .85 * rr + .15 * oo\n'
         '    routed[of], routed[rf], routed[rf & of] = rr[of], oo[rf], .5\n'
         '    result[:, [11]] = routed[:, [11]]\n',
         '    rr = .65 * rank(r) + .35 * rank(r224)\n'
         '    public = .5 * rr + .5 * rank(ours)\n'
         '    public[:, [11]] = rr[:, [11]]\n')
ANCHORS = ('    public[failed], public[sf], public[failed & sf] = ss[failed], rr[sf], .5\n',
           '    return result\n\n\ndef reference(',
           '    public = .5 * (.65 * rank(r) + .35 * rank(r224)) + .5 * rank(ours)\n')
OA_LINES = ('    public[:, [5,6]] = np.where(sf[:, None], rr[:, [5,6]], ss[:, [5,6]])\n',
            '    public[:, [5,6]] = rank(ours)[:, [5,6]]\n')
ROUTE_FUNCTIONS = ('p_branch', 'blend', 'reference')
WRAPPER_EDITS = [
    ('VERSION = "1.1.34-r224r6oaq"', 'VERSION = "' + VERSION + '"'),
    ('RELEASE = "C85R7+codecs R224 R6 FLIP_ALL hard OA"', 'RELEASE = "C85R7+codecs R224 R6 FLIP_ALL hard OA FractureR"'),
    ('KERNEL_ID = "taeyangg4/knee-lb-r224r6oaq-v1"', 'KERNEL_ID = "' + SLUG + '"'),
    ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.34-r224r6oaq"',
     'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + VERSION + '"'),
    ('formula="public OA=rank(S_R6flip) with combined-R fallback; other public=.5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",',
     'formula="public OA=rank(S_R6flip) with combined-R fallback; Fracture=combined-R with R-only eligibility; other public=.5*(.65*r(R384)+.35*r(R224))+.5*r(S_R6flip); own-success ranks; float32",')]
KEEP_RULE = ('RETAIN iff displayed >= 0.957 (a 0.957 tie is an unresolved alternative, not a proven gain); '
             'DROP at <= 0.956; absent reading HOLD')


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def authenticated(spec):
    """Admitted receipt + gate-copy inventory + identical fixed source kernel."""
    old.pin(spec['receipt'], spec['receipt_sha'])
    saved = old.read(spec['receipt'])
    old.need(saved['status'] == 'PASS' and saved['errors'] == [] and saved['fixture_only'] is False,
             'not an admitted parent/reference')
    old.need(Path(saved['stage']).resolve() == spec['gate'].resolve(), 'receipt stage differs from gate copy')
    inv = gate.kernel(spec['gate'], gate.Inputs(ROOT))[2]
    old.need(inv == saved['stage_inventory'], 'admitted inventory mismatch')
    old.need(gate.sha(gate.encoded(inv['kernel_files'])) == spec['kernel'], 'kernel identity')
    old.need(gate.kernel(spec['source'], gate.Inputs(ROOT))[3] == gate.kernel(spec['gate'], gate.Inputs(ROOT))[3],
             'fixed source differs from admitted gate kernel')
    nb = old.read(spec['source'] / 'kernel/release.ipynb')
    files, blob, assignments = old.unpack(fast.restored(nb))
    old.need(len(files) == 31, 'embedded roster')
    return nb, files, blob, assignments


def parent():
    return authenticated(PARENT)


def function_bytes(raw, name):
    text = raw.decode()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ''.join(text.splitlines(True)[node.lineno - 1:node.end_lineno]).encode()


def admitted_hunks():
    """Prove each child hunk is an admitted H-B-185 and H-B-187 hunk with only the selector restricted."""
    sib, fx = authenticated(SIBLING)[1][HELPER], authenticated(FX)[1][HELPER]
    old.need(sib.count(SIBLING_SELECTOR.encode()) == 6 and fx.count(FX_SELECTOR.encode()) == 6, 'admitted selector count drift')
    for hunk in HUNKS:
        old.need(hunk.count(SELECTOR) == 2, 'hunk selector count')
        old.need(sib.count(hunk.replace(SELECTOR, SIBLING_SELECTOR).encode()) == 1, 'hunk is not the admitted H-B-185 hunk')
        old.need(fx.count(hunk.replace(SELECTOR, FX_SELECTOR).encode()) == 1, 'hunk is not the admitted H-B-187 hunk')
    return sib, fx


def route_blocks(fx):
    """Selector-free admitted bytes: each route function from its def line up to its selector line."""
    blocks = []
    for name in ROUTE_FUNCTIONS:
        lines = function_bytes(fx, name).decode().splitlines(True)
        cut = next(i for i, line in enumerate(lines) if FX_SELECTOR in line)
        blocks.append(''.join(lines[:cut]))
    return blocks


def reference_proof(files=None):
    sib, fx = admitted_hunks()
    child = (files or payload()[2])[HELPER]
    blocks = []
    for spec, member in ((FX, fx), (SIBLING, sib)):
        for text in route_blocks(fx):
            old.need(text and member.count(text.encode()) == 1 and child.count(text.encode()) == 1,
                     'reference block absent or ambiguous')
            blocks.append(dict(reference_stage=rel(spec['gate']), reference_member=HELPER, child_member=HELPER, text=text))
    return dict(blocks=blocks)


def pre_execution():
    old.pin(SPEC, SPEC_SHA)
    return dict(spec_sha256=SPEC_SHA, spec_path=rel(SPEC),
                review='process-lightening 28(a): byte-patch child of the hidden-proven H-B-181 with only admitted hunks; '
                       'ONE combined independent review dispatched by B and bound to the v1.4.5 gate; no builder self-review')


def registration():
    parent(); admitted_hunks()
    return dict(item='H-B-193', version=VERSION, kernel_id=SLUG,
        purpose='R6-line stack of our routes on the auto-selected H-B-181 (R6 + C1 hard OA): add the admitted R-only Fracture route; '
                'Gold58 Fracture on the R6 line +0.0016 macro, P ~ 0.85 (EXPOSED, selected from a 12-target map; motivation only)',
        parent=rel(BASE), parent_gate=rel(PARENT['gate']), parent_kernel_sha256=PARENT['kernel'],
        parent_receipt_sha256=PARENT['receipt_sha'], parent_submission=57263371,
        parent_hidden_status='COMPLETE; user supplied (D-B-211); auto-selected with H-B-174',
        parent_displayed_public=.957, expected_public=.957, expected_public_upside=.958,
        formula=FORMULA, routed_columns=list(ROUTED), added_route=list(ROUTED), unchanged_oa_columns=[5, 6],
        protected_columns=list(PROTECTED),
        rule_reference=dict(item='H-B-187', kernel_id='taeyangg4/knee-lb-stackfxq-v1', gate=rel(FX['gate']),
                            kernel_sha256=FX['kernel'], receipt_sha256=FX['receipt_sha'], selector=FX_SELECTOR),
        code_reference=dict(item='H-B-185', kernel_id='taeyangg4/knee-lb-r6stackoaq-v1', gate=rel(SIBLING['gate']),
                            kernel_sha256=SIBLING['kernel'], receipt_sha256=SIBLING['receipt_sha'], selector=SIBLING_SELECTOR),
        pre_execution=pre_execution(),
        preregistration='register precedes prepare/tests/replay/Gold58 description; no label-based selection; aggregates only, no printed labels',
        failure_rule='Fracture uses the admitted R-only rule: independent success-only R384/R224 ranks, .65/.35 mix, single R survivor alone; '
                     'outer .85 r(R) + .15 r(O); O failed -> R only; both R failed -> O only even if compact succeeds; both failed -> .5. '
                     'Columns 0..10 keep the exact H-B-181 rule (OA 5/6 success-only R6 ranks with combined-R fallback; others .5 r_R + .5 r(R6)). '
                     'Own survivor 17/17/12 and R224 >3% abort unchanged.',
        parent_clarification='H-B-181: R224 + FULL flip + R6 compact (legacy S_R7 slot name) + C1 hard OA; no fluid route, no compo15; '
                             'own 17/17/12 EXP143v2/EXP155/Q_A5q and private dataset sources unchanged.',
        public_reading=dict(reference_value=.957, expected=.957, upside=.958, rule=KEEP_RULE,
                            expected_written='before submission, at build registration 2026-10-17 KST',
                            source='user-supplied reading only; no score polling'),
        timing=dict(source='user-supplied H-B-181 hidden COMPLETE submission 57263371 (D-B-211)',
                    parent_hidden_hours_upper=6.3, includes_queue=True, inference_delta='none', child_measured_hours=None,
                    limitation='Parent completion is the runtime basis; arithmetic-only change, no new inference, unmeasured child duration.'),
        sources=['plan.md H-B-193; 2026-10-16 stacking audit fable stack_gold58 HB181(R6+OA) Fx_to_R macro +0.00162 '
                 '[-0.00097, +0.00521], P ~ 0.85; motivation only',
                 'D-B-211 H-B-181 COMPLETE 0.957, auto-selected with H-B-174'],
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        runtime_qualification='CPU cached arithmetic replay only; no fresh model forward/CUDA/Linux qualification',
        gate_binding='HOLD: B slug-generic v1.4.5 gate pin/bind/evaluate with authentic dispatcher builder/reviewer receipts')


def decision(displayed_score):
    if displayed_score is None:
        return 'HOLD'
    old.need(type(displayed_score) in (int, float) and math.isfinite(displayed_score) and 0 <= displayed_score <= 1,
             'finite displayed public score required')
    old.need(abs(displayed_score * 1000 - round(displayed_score * 1000)) < 1e-6, 'three-decimal displayed score required')
    return 'RETAIN' if round(displayed_score, 3) >= .957 else 'DROP'


def patched(original):
    files = dict(original)
    sib, _ = admitted_hunks()
    raw = files[HELPER]
    old.need(raw.count(SELECTOR.encode()) == 0 and raw.count(b'[7,9') == 0, 'parent already routes from R')
    line = next(x for x in raw.decode().splitlines() if x.startswith('FORMULA = '))
    raw = old.replace(raw, line, 'FORMULA = ' + repr(FORMULA))
    raw = old.replace(raw, ANCHORS[0], HUNKS[0])
    raw = old.replace(raw, ANCHORS[1], HUNKS[1] + ANCHORS[1])
    raw = old.replace(raw, ANCHORS[2], HUNKS[2])
    # Independent construction: the admitted H-B-185 helper restricted to column 11.
    sline = next(x for x in sib.decode().splitlines() if x.startswith('FORMULA = '))
    expected = old.replace(sib, sline, 'FORMULA = ' + repr(FORMULA)).replace(SIBLING_SELECTOR.encode(), SELECTOR.encode())
    old.need(raw == expected, 'child helper differs from the restricted H-B-185 helper')
    files[HELPER] = raw
    wrapper = old.replace(files[WRAPPER], old.digest(original[HELPER]), old.digest(raw))
    for before, after in WRAPPER_EDITS:
        wrapper = old.replace(wrapper, before, after)
    files[WRAPPER] = wrapper
    return files


def payload():
    nb, original, old_blob, assignments = parent()
    files = patched(original)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buf, 'w') as target:
        for info in source.infolist():
            target.writestr(info, files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before, after in [
            ('BLOB = ' + repr(assignments['BLOB']), 'BLOB = ' + repr(base64.b64encode(blob).decode())),
            ('PINS = ' + repr(assignments['PINS']), 'PINS = ' + repr({k: old.digest(v) for k, v in files.items()})),
            (repr(old.digest(old_blob)), repr(old.digest(blob)))]:
        code = old.replace(code, before, after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    meta = old.read(BASE / 'kernel/kernel-metadata.json')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta, files, blob


def scope(nb, meta):
    expected, em, files, blob = payload()
    old.need(nb == expected and meta == em, 'forbidden notebook/metadata diff')
    actual, ab, _ = old.unpack(fast.restored(nb))
    old.need(actual == files and ab == blob, 'forbidden embedded diff')
    original = parent()[1]
    old.need(set(files) == set(original) and {k for k in files if files[k] != original[k]} == {HELPER, WRAPPER}, 'member scope')
    for name, data in files.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
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
        full += rbuild.diff(original[k], files[k], 'v1.1.34-r224r6oaq/' + k, VERSION + '/' + k)
    for name, before, after in [('notebook', pnb, nb), ('kernel-metadata.json', old.read(BASE / 'kernel/kernel-metadata.json'), meta)]:
        full += rbuild.diff(old.encoded(before), old.encoded(after), 'parent/' + name, 'candidate/' + name)
    return nb, meta, files, blob, manifest, full.encode()


CHANGELOG = (b'# v1.1.42-r6oafxq / H-B-193\n'
             b'Fracture (column 11) follows the admitted R-only route on the auto-selected H-B-181: routed-from-R set [] -> [11]. '
             b'Hunks = admitted H-B-185 fluid hunks with [7,9] -> [11] (= admitted H-B-187 hunks with [7,9,11] -> [11]). '
             b'Formula/version/slug/release strings and the chained helper hash only. All 29 other embedded members, R6 datasets, '
             b'OA hunk, FULL flip and FAST unchanged; no fluid route, compo15 or new inference. Builder tools/c85r7_r6oafxq_v1142.py 1.0.0. '
             b'Parent backup .local/backups/hb193-build/hb193-parent; no frozen edits.\n')


def register():
    value = registration()
    old.write(OUT / 'registration.json', old.encoded(value))
    return dict(status='REGISTERED_BEFORE_EVALUATION', version=VERSION, exposure='EXPOSED', rule=value['public_reading']['rule'])


def stage_files():
    nb, meta, files, blob, manifest, full = artifacts()
    return dict(nb=nb, meta=meta, files=files, blob=blob, written=[
        ('kernel/release.ipynb', old.encoded(nb)), ('kernel/kernel-metadata.json', old.encoded(meta)),
        ('new-members.json', old.encoded(manifest)), (DIFF, full), ('SPEC.md', SPEC.read_bytes()),
        ('CHANGELOG.md', CHANGELOG), (PROOF, old.encoded(reference_proof(files)))])


def prepare():
    old.need((OUT / 'registration.json').read_bytes() == old.encoded(registration()), 'register first')
    built = stage_files()
    scope(built['nb'], built['meta'])
    for name in ('kernel/release.ipynb', 'kernel/kernel-metadata.json'):
        old.write(BACKUP / name, (BASE / name).read_bytes())
    old.write(BACKUP / 'parent-admission-receipt.json', PARENT['receipt'].read_bytes())
    for name, data in built['written']:
        old.write(OUT / name, data)
    for label, contents in [('candidate', built['files']), ('parent', parent()[1]), ('fx', authenticated(FX)[1])]:
        for name, data in contents.items():
            old.write(SCRATCH / (label + '-runtime') / name, data)
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, builder_version=BUILDER_VERSION,
               builder_sha256=old.digest(Path(__file__).read_bytes()),
               kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], archive_sha256=old.digest(built['blob']),
               embedded={k: old.digest(v) for k, v in built['files'].items()}, unchanged=29, changed=[HELPER, WRAPPER],
               added=[], removed=[], parent_kernel_sha256=PARENT['kernel'], parent_receipt_sha256=PARENT['receipt_sha'],
               reference_proof_sha256=old.digest((OUT / PROOF).read_bytes()),
               qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    old.write(OUT / 'build-receipt.json', old.encoded(rec))
    return validate()


def validate():
    built = stage_files()
    old.need((OUT / 'registration.json').read_bytes() == old.encoded(registration()), 'stage drift: registration.json')
    for name, data in built['written']:
        old.need((OUT / name).read_bytes() == data, 'stage drift: ' + name)
    scope(old.read(OUT / 'kernel/release.ipynb'), old.read(OUT / 'kernel/kernel-metadata.json'))
    rec = old.read(OUT / 'build-receipt.json')
    old.need(rec['builder_sha256'] == old.digest(Path(__file__).read_bytes()), 'builder drift')
    old.need(rec['embedded'] == {k: old.digest(v) for k, v in built['files'].items()} and
             rec['archive_sha256'] == old.digest(built['blob']), 'archive drift')
    old.need(rec['kernel_files'] == gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], 'kernel drift')
    old.need(rec['qualified_for_submission'] is False and rec['unchanged'] == 29, 'receipt scope')
    old.need(rec['reference_proof_sha256'] == old.digest((OUT / PROOF).read_bytes()), 'reference proof drift')
    from tools import strict_variant_gate_r224_v145 as generic
    pins = gate.Inputs(ROOT)
    generic.metadata_delta(gate.kernel(BASE, pins), gate.kernel(OUT, pins), SLUG, registration())
    # Read-only check with the gate's own reference-proof validator (no pin/bind/evaluate).
    proofs = generic.reference_blocks(OUT / PROOF, gate.kernel(OUT, pins), pins)
    old.need(len(proofs) == 2 * len(ROUTE_FUNCTIONS), 'reference proof block count')
    return dict(status='PASS_CPU_VALIDATE', kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),
                identical=29, changed=2, reference_blocks=len(proofs), metadata_raw_values_identical_except_id_title=True)


def math_functions(files):
    """Test-only extraction; the real replay imports the actual embedded archives."""
    return infrastructure.math_functions(files)


def oracle(arrays, masks):
    """Independent column-wise survivor oracle, using NaN rather than zero ranks."""
    import numpy as np
    import pandas as pd

    def rank(v, failed):
        out = np.full(len(v), np.nan)
        out[~failed] = pd.Series(v[~failed]).rank(method='average', pct=True).to_numpy()
        return out
    result = np.empty_like(arrays[0])
    rf, f224, sf, *ofs = masks
    for col in range(12):
        a, b, s, *owns = [rank(v[:, col], f) for v, f in zip(arrays, masks)]
        rr = np.where(rf, b, np.where(f224, a, .65 * a + .35 * b))
        rr[rf & f224] = .5
        if col in ROUTED:
            public = rr.copy()
        elif col in (5, 6):
            public = s.copy()
            public[sf] = rr[sf]
        else:
            public = .5 * rr + .5 * s
            public[rf & f224] = s[rf & f224]
            public[sf] = rr[sf]
        pf = (rf & f224) if col in ROUTED else (rf & f224 & sf)
        public[pf] = .5
        den = sum(w * (~f).astype(int) for w, f in zip((17, 17, 12), ofs))
        own = sum(w * np.nan_to_num(v) for w, v in zip((17, 17, 12), owns)) / np.maximum(den, 1)
        pr, oo = rank(public, pf), rank(own, den == 0)
        out = .85 * pr + .15 * oo
        out[den == 0] = pr[den == 0]
        out[pf] = oo[pf]
        out[pf & (den == 0)] = .5
        result[:, col] = out
    return result


REPLAY_SCRIPT = r'''
import os,sys,json,itertools,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
runtime,source,dest=map(Path,sys.argv[1:4])
sys.path.insert(0,str(runtime))
import exp169_release_c85r7codec as c
import exp169_r224 as h
assert c.r224 is h and Path(h.__file__).resolve().parent==runtime.resolve()
assert Path(c.__file__).resolve().parent==runtime.resolve()
assert all(not name.startswith('tools.') for name in sys.modules)
data=np.load(source,allow_pickle=False)
ids=list(data['ids']); arrays=[data['a'+str(i)] for i in range(6)]
names=('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')
outputs={}; checks=[]
def csv_hash(raw,cols):
    v=b'\n'.join(b','.join(row.split(b',')[j] for j in (0,*[i+1 for i in cols])) for row in raw.splitlines())
    return hashlib.sha256(v).hexdigest()
for n in (2,len(ids)):
    aa=[a[:n] for a in arrays]
    for case,bits in enumerate(itertools.product((False,True),repeat=6)):
        key=f'n{n}-case{case}'
        failed={name:({ids[0]:'injected cached member failure'} if bit else {}) for name,bit in zip(names,bits)}
        try:
            result=h.blend(ids[:n],*aa,failed)
        except ValueError as exc:
            assert n==2 and bits[1] and 'more than 3%' in str(exc),str(exc)
            checks.append(dict(key=key,status='EXPECTED_R224_ABORT'))
            continue
        assert not(n==2 and bits[1]),'R224 threshold bypass'
        assert np.isfinite(result).all() and np.all((result>=0)&(result<=1))
        if not any(bits): assert np.array_equal(result,h.reference(*aa))
        outputs[key]=result
        raw=c.p.csv_bytes(ids[:n],result)
        checks.append(dict(key=key,status='PASS',all_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
            protected_csv_sha256=csv_hash(raw,tuple(range(11))),
            routed_csv_sha256=csv_hash(raw,(11,))))
np.savez(dest.with_suffix('.npz'),**outputs)
import torch
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
dest.with_suffix('.json').write_text(json.dumps(dict(version=c.VERSION,checks=checks,cuda_initialized=False,embedded_imports=True),indent=2))
print(json.dumps(dict(status='PASS',version=c.VERSION,cases=len(checks),successes=len(outputs),expected_aborts=len(checks)-len(outputs),cuda_initialized=False)))
'''
# Runtime label -> input compact arrays. fx7 = H-B-187 with its genuine R7 compact input.
RUNS = (('parent', 'r6'), ('candidate', 'r6'), ('fx7', 'r7'), ('fx6', 'r6'))


def real_inputs():
    import numpy as np
    from tools import c85r7_gold58_candidates as cache
    from tools import c85r7_r224r6oaq_v1134 as h181
    reg, common, failed, arms = cache.context()
    r6_reg, compact, r6_auth = h181.authenticated_r6()
    old.need(r6_reg['binding']['ids'] == reg['binding']['ids'] and len(reg['binding']['ids']) == 58, 'ordered real cohort drift')
    ids = reg['binding']['ids']
    arrays = [np.asarray(common[k], float) for k in ('R', 'R224')] + [compact] + [np.asarray(common[k], float) for k in MEMBERS[3:]]
    r7_arrays = [a.copy() for a in arrays]
    r7_arrays[2] = np.asarray(arms['flip-1']['predictions'], float)
    masks = [np.asarray(failed[k], bool) if k in failed else np.zeros(len(ids), bool) for k in MEMBERS]
    old.need(not any(m.any() for m in masks), 'real replay requires healthy recorded rows')
    old.need(not any(np.all(a == .5, axis=1).any() for a in arrays[:2]), 'unexpected real R sentinel')
    old.need(not np.array_equal(arrays[2], r7_arrays[2]), 'R6 and R7 compact inputs must differ')
    for name, digest in r6_auth['admitted_H174_exact_contracts'].items():
        old.need(old.digest(parent()[1][name]) == digest and old.digest(payload()[2][name]) == digest, 'R6 measured contract differs')
    return reg, ids, arrays, r7_arrays, r6_auth


def replay():
    import itertools
    import numpy as np
    import torch
    old.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only replay required')
    valid = validate()
    reg, ids, arrays, r7_arrays, r6_auth = real_inputs()
    for label, files in [('candidate', payload()[2]), ('parent', parent()[1]), ('fx', authenticated(FX)[1])]:
        for name, data in files.items():
            old.need((SCRATCH / (label + '-runtime') / name).read_bytes() == data, 'extracted runtime drift: ' + label + '/' + name)
    for label, values in [('r6', arrays), ('r7', r7_arrays)]:
        np.savez(SCRATCH / (label + '-real-input.npz'), ids=np.asarray(ids), **{'a' + str(i): a for i, a in enumerate(values)})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    runs, outputs = {}, {}
    for label, source in RUNS:
        runtime = SCRATCH / (('fx' if label.startswith('fx') else label) + '-runtime')
        output = SCRATCH / ('real-' + label)
        proc = subprocess.run([sys.executable, '-B', '-I', '-c', REPLAY_SCRIPT, str(runtime),
                               str(SCRATCH / (source + '-real-input.npz')), str(output)],
                              cwd=SCRATCH, env=env, capture_output=True, text=True)
        old.need(proc.returncode == 0, 'isolated replay failed: ' + proc.stderr[-6000:])
        runs[label] = old.read(output.with_suffix('.json'))
        with np.load(output.with_suffix('.npz'), allow_pickle=False) as data:
            outputs[label] = {k: data[k] for k in data.files}
    old.need(runs['candidate']['version'] == VERSION and runs['parent']['version'] == '1.1.34-r224r6oaq'
             and runs['fx7']['version'] == runs['fx6']['version'] == '1.1.37-stackfxq', 'embedded runtime identity')
    checks = {label: {row['key']: row for row in run['checks']} for label, run in runs.items()}
    cases = []
    for n in (2, len(ids)):
        for case, bits in enumerate(itertools.product((False, True), repeat=6)):
            key = f'n{n}-case{case}'
            row = checks['candidate'][key]
            old.need(all(checks[label][key]['status'] == row['status'] for label in checks), 'failure gate differs')
            if row['status'] == 'EXPECTED_R224_ABORT':
                cases.append(dict(key=key, status=row['status']))
                continue
            current = [np.zeros(n, bool) for _ in arrays]
            for mask, bit in zip(current, bits):
                mask[0] = bit
            actual = outputs['candidate'][key]
            old.need(actual.tobytes() == oracle([a[:n] for a in arrays], current).tobytes(), 'independent pandas byte parity: ' + key)
            old.need(np.ascontiguousarray(actual[:, PROTECTED]).tobytes() ==
                     np.ascontiguousarray(outputs['parent'][key][:, PROTECTED]).tobytes(), 'protected-eleven bytes: ' + key)
            old.need(row['protected_csv_sha256'] == checks['parent'][key]['protected_csv_sha256'], 'protected-eleven CSV: ' + key)
            for label in ('fx7', 'fx6'):
                old.need(np.ascontiguousarray(actual[:, ROUTED]).tobytes() ==
                         np.ascontiguousarray(outputs[label][key][:, ROUTED]).tobytes(), 'fracture bytes vs ' + label + ': ' + key)
                old.need(row['routed_csv_sha256'] == checks[label][key]['routed_csv_sha256'], 'fracture CSV vs ' + label + ': ' + key)
            changed = int(np.sum(actual[:, 11] != outputs['parent'][key][:, 11]))
            cases.append(dict(key=key, status='PASS', independent_pandas_bytes_exact=True, protected_eleven_bytes_exact=True,
                              fracture_equals_h187_r7_and_r6_inputs=True, csv_fields_exact=True, fracture_rows_changed_vs_parent=changed))
    healthy = [c for c in cases if c['key'] == f'n{len(ids)}-case0'][0]
    old.need(healthy['fracture_rows_changed_vs_parent'] > 0, 'route has no effect on real healthy cohort')
    report = dict(status='PASS', builder_version=BUILDER_VERSION, kernel_sha256=valid['kernel_sha256'], real_two_studies=True,
        full_real_cohort=58, scenarios=64, two_study_successes=sum(c['status'] == 'PASS' and c['key'].startswith('n2-') for c in cases),
        two_study_expected_r224_aborts=sum(c['status'] == 'EXPECTED_R224_ABORT' for c in cases),
        full_cohort_successes=sum(c['status'] == 'PASS' and c['key'].startswith('n58-') for c in cases),
        protected_eleven_float64_bytes_exact=True, protected_eleven_csv_fields_exact=True,
        fracture_equals_h187_float64_bytes=True, fracture_equals_h187_csv_fields=True, independent_pandas_bytes_exact=True,
        runs={label: dict(runtime=('fx' if label.startswith('fx') else label) + '-runtime', compact_input=src) for label, src in RUNS},
        cases=cases, embedded_runs=runs, output_sha256={label: {k: old.digest(v.tobytes()) for k, v in o.items()} for label, o in outputs.items()},
        cohort_sha256=old.digest(old.encoded(ids)), historical_binding_sha256=reg['binding_sha256'], r6_authentication=r6_auth,
        inputs_sha256={'R6': [old.digest(a.tobytes()) for a in arrays], 'R7': [old.digest(a.tobytes()) for a in r7_arrays]},
        exposure='EXPOSED', clean_oof=False,
        limitation='Authenticated real cached serving arithmetic; no labels used or printed; no AUC/model forward/CUDA/Linux '
                   'qualification. Explicit injected missing-member masks; R224 >3% abort preserved, not bypassed.')
    old.need(report['two_study_successes'] == 32 and report['two_study_expected_r224_aborts'] == 32 and
             report['full_cohort_successes'] == 64, 'scenario coverage')
    (SCRATCH / 'replay.json').write_bytes(old.encoded(report))
    return {k: v for k, v in report.items() if k not in ('cases', 'embedded_runs', 'r6_authentication', 'output_sha256')}


def gold58():
    """Descriptive EXPOSED Gold58 delta child - H-B-181 (aggregates only; not a gate)."""
    from unittest import mock
    import numpy as np
    import torch
    from tools import c85r7_gold58_candidates as cache
    old.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only required')
    valid = validate()
    smoke = old.read(SCRATCH / 'replay.json')
    old.need(smoke['status'] == 'PASS' and smoke['kernel_sha256'] == valid['kernel_sha256'], 'CPU replay missing/stale')
    key = 'n58-case0'
    outputs = {}
    for label in ('candidate', 'parent', 'fx7'):
        with np.load(SCRATCH / ('real-' + label + '.npz'), allow_pickle=False) as data:
            outputs[label] = np.asarray(data[key])
        old.need(old.digest(outputs[label].tobytes()) == smoke['output_sha256'][label][key], 'replayed output drift: ' + label)
    reg = cache.context()[0]
    ids, labels = cache.d.gold.gold()
    old.need(list(ids) == list(reg['binding']['ids']), 'Gold58 roster/order drift')
    with mock.patch.object(cache, 'PAIRS', [('child', 'H181')]):
        metrics = cache.paired_metrics(labels, dict(child=outputs['candidate'], H181=outputs['parent']))
    delta = metrics['comparisons']['child-minus-H181']
    per = delta['per_target_delta']
    old.need(all(abs(v) < 1e-12 for i, v in enumerate(per) if i != 11), 'non-Fracture Gold58 AUC changed')
    names = list(cache.d.gold.LABELS)
    result = dict(status='DESCRIPTIVE_EXPOSED_NOT_A_GATE', kernel_sha256=valid['kernel_sha256'], studies=58,
                  comparison='child v1.1.42-r6oafxq minus H-B-181 v1.1.34-r224r6oaq (actual embedded runtime outputs, healthy real cohort)',
                  macro_delta=delta['macro_delta'], macro_ci95=delta['macro_ci95'],
                  weak_four_delta=delta['weak_four_delta'], fracture_auc_delta=per[11],
                  fracture_auc={'child': metrics['per_target_auc']['child'][11], 'H181': metrics['per_target_auc']['H181'][11]},
                  per_target_delta=dict(zip(names, per)), macro_auc={k: float(np.mean(v)) for k, v in metrics['per_target_auc'].items()},
                  bootstrap=metrics['bootstrap'], fx7_fracture_identical=bool(np.array_equal(outputs['candidate'][:, 11], outputs['fx7'][:, 11])),
                  exposure='EXPOSED', clean_oof=False, selection='Fracture route selected from a 12-target Gold58 map (selection-exposed)',
                  labels_printed=False, qualified_for_submission=False)
    (SCRATCH / 'gold58-descriptive.json').write_bytes(old.encoded(result))
    return result


def admit():
    valid = validate()
    smoke = old.read(SCRATCH / 'replay.json')
    old.need(smoke['status'] == 'PASS' and smoke['kernel_sha256'] == valid['kernel_sha256'], 'CPU replay missing/stale')
    old.need(smoke['protected_eleven_float64_bytes_exact'] and smoke['protected_eleven_csv_fields_exact'], 'protected-eleven parity missing')
    old.need(smoke['fracture_equals_h187_float64_bytes'] and smoke['fracture_equals_h187_csv_fields'], 'fracture route parity missing')
    return dict(status='HOLD_B_SLUG_GENERIC_GATE_AND_COMBINED_INDEPENDENT_REVIEW', qualified_for_submission=False,
                kernel_sha256=valid['kernel_sha256'], gate='tools/strict_variant_gate_r224_v145.py', reference_proof=rel(OUT / PROOF))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register', 'prepare', 'validate', 'cpu-replay', 'gold58', 'admit'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.command in ('prepare', 'admit'):
        old.need(args.dry_run, 'CPU dry-run required')
    result = {'register': register, 'prepare': prepare, 'validate': validate, 'cpu-replay': replay,
              'gold58': gold58, 'admit': admit}[args.command]()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
