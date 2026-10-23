"""H-B-191 builder 1.0.0: w_Effusion = w_Fracture = 1.0 on admitted H-B-189.

Changelog 1.0.0 (claude-opus-b1, 2026-10-17): one candidate on the admitted
H-B-189 0.958-weights parent. Weight entries 7/11 -> 1.0 plus the admitted
H-B-187/H-B-168 fluid survivor block for columns [7,11]; dependent wrapper
pins/version/id only. Reuses reviewed H189/H187/H181 infrastructure without
editing it. A Codex partial of this file (astra-b4, died 00:28, unreviewed) is
preserved in .local/backups/partial-codex-1017/hb191/ and was NOT reused.
No frozen/scored/original edits, no GPU/Kaggle/upload.
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
from tools import c85r7_r6w958q_v1139 as h189
from tools import c85r7_stackfxq_v1137 as h187

old, fast, rbuild, gate = h189.old, h189.fast, h189.rbuild, h189.gate
reuse = h189.reuse
HELPER, WRAPPER = h189.HELPER, h189.WRAPPER
MEMBERS, PRIVATE_DATASETS = h189.MEMBERS, h189.PRIVATE_DATASETS
VERSION = '1.1.41-r6w958fxq'
SLUG = 'taeyangg4/knee-lb-r6w958fxq-v1'
BASE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q'
PARENT_GATE = ROOT / '.local/scratch/astra-b5/hb189-gate/gate-copy-v145'
RECEIPT = ROOT / '.local/state/gates/knee-lb-r6w958q-v1.json'
PARENT_KERNEL = 'c76b1ddb4d83110ee6556f5f5282ad843af9f6d0cbc932dc6630e3868d4e084f'
PARENT_RECEIPT = '8781c905cc86a7ad35bdaf49720796631e352ce8b58e5e0532bbe6ccbadbd7a6'
PARENT_REPLAY = ROOT / '.local/scratch/hb189-build/hb189/replay.json'
PARENT_REPLAY_SHA = '5f176aa8ff2fe2f6909eef11dbc84517f41be4fd1d7ed1bc901dc5f03a4fc7fb'
PARENT_HEALTHY58_SHA = '97b35fd7f307ae53fd99e8c71bf2cc69fb7dbb46278a54bbda339d1d5019c805'
# Admitted R-only route references (authenticated; their helpers are only read).
REFERENCES = {
    'h187': dict(stage=ROOT / '.local/scratch/astra-b5/hb187-gate/gate-copy-v145',
                 source=ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.37-stackfxq',
                 receipt=ROOT / '.local/state/gates/knee-lb-stackfxq-v1.json',
                 receipt_sha='330a5f0ac5e4bdc37f792f61c906eb9b38e0819d422c623d17e79e8eb2475fe3',
                 kernel='c61907d2e98ec00038ff5fba4bfe1eb21e41d292ee5ebfa6884e7b9b5349d4d6',
                 routed=(7, 9, 11)),
    'h168': dict(stage=ROOT / '.local/scratch/astra-b5/hb168-gate/gate-copy-v144',
                 source=ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.25-stackq',
                 receipt=ROOT / '.local/state/gates/knee-lb-stackq-v1.json',
                 receipt_sha='b3a9b5579f9a476e6b9112830850f81d15badea41cda59029e2011b4789ae640',
                 kernel='9e184865452a3a4a7a5b637c0be9701fac965f7546b6a0dccb6956bb1dbce89b',
                 routed=(7, 9))}
OUT = old.BASE.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb191-build/hb191'
BACKUP = ROOT / '.local/backups/hb191-build/hb191-parent'
SPEC = SCRATCH / 'SPEC.md'
PRE_REVIEW = SCRATCH / 'PRE-REVIEW.md'
FABLE = h189.FABLE
DIFF = 'full-diff-vs-v1.1.39-r6w958q.txt'
PROOF = 'reference-proof.json'
WEIGHTS = (.5, .5, .5, .5, .5, .2, .6, 1.0, .8, .35, .5, 1.0)
ROUTED = (7, 11)
PROTECTED = tuple(i for i in range(12) if i not in ROUTED)
SURVIVOR = ('    rr, rf = r_half(r, r224, rf, f224)\n'
            '    rr = c85.scored_rank(rr, rf)\n'
            '    routed = .85 * rr + .15 * oo\n'
            '    routed[of], routed[rf], routed[rf & of] = rr[of], oo[rf], .5\n')
SELECT = '    result[:, [7,11]] = routed[:, [7,11]]\n'
FORMULA = ('0.85*r(P)+0.15*r(O); rr=0.65*r(R384)+0.35*r(R224); '
    'P_t=w_t*rr_t+(1-w_t)*r(R6flip)_t; w=.5 except Lateral OA=.2, PF OA=.6, Synovitis=.8, Baker\'s=.35, '
    'Effusion=1, Fracture=1 (Effusion/Fracture: R-only failure eligibility, fluid survivor rule); '
    'O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46')
WRAPPER_FORMULA = ('public w*(.65*r(R384)+.35*r(R224))+(1-w)*r(S_R6flip); '
    'w=.5 except LatOA=.2/PFOA=.6/Syn=.8/Baker=.35/Effusion=1/Fracture=1; '
    'Effusion/Fracture pre-route here (final 7/11 use R-only eligibility); own-success ranks; float32')


def authenticated(stage, source, receipt, receipt_sha, kernel_sha):
    """Reviewed H-B-187 admitted-stage authentication (receipt, inventory, source bytes)."""
    return h187.authenticated(stage, source, receipt, receipt_sha, kernel_sha)


def parent():
    value = authenticated(PARENT_GATE, BASE, RECEIPT, PARENT_RECEIPT, PARENT_KERNEL)
    old.need(value[1] == h189.patched(h189.parent()[1]), 'admitted H189 != H189 weights patch of admitted H174')
    old.need(old.read(BASE/'kernel/kernel-metadata.json')['dataset_sources'] == PRIVATE_DATASETS,
             'fixed private R6 dataset list drift')
    return value


def reference(label):
    ref = REFERENCES[label]
    return authenticated(ref['stage'], ref['source'], ref['receipt'], ref['receipt_sha'], ref['kernel'])


def reference_proof():
    blocks = []
    for label in ('h187', 'h168'):
        helper = reference(label)[1][HELPER]
        old.need(helper.count(SURVIVOR.encode()) == 1, 'admitted survivor block not unique in ' + label)
        blocks.append(dict(reference_stage=REFERENCES[label]['stage'].relative_to(ROOT).as_posix(),
                           reference_member=HELPER, child_member=HELPER, text=SURVIVOR))
    return dict(blocks=blocks)


def pre_review():
    import re
    raw = PRE_REVIEW.read_text(encoding='utf-8')
    sha = old.digest(SPEC.read_bytes())
    old.need('Verdict: PASS for CPU-only build/tests/replay of SPEC.md v1.0.0' in raw
             and re.search(r'\b' + sha + r'\b', raw), 'independent pre-execution PASS bound to spec required')
    return dict(spec_sha256=sha, review_sha256=old.digest(PRE_REVIEW.read_bytes()),
                review_path=PRE_REVIEW.relative_to(ROOT).as_posix())


def registration():
    parent(); reference_proof()
    return dict(item='H-B-191', version=VERSION, kernel_id=SLUG,
        purpose='Taeyang-requested stack on the 0.958 base: admitted H-B-189 per-target weights plus our '
                'non-colliding Effusion -> R (fluid route) and Fracture -> R routes (w=1.0)',
        parent=PARENT_GATE.relative_to(ROOT).as_posix(), parent_source_stage=BASE.relative_to(ROOT).as_posix(),
        parent_kernel_sha256=PARENT_KERNEL, parent_receipt_sha256=PARENT_RECEIPT,
        parent_displayed_public=None, runtime_parent_submission=57251520,
        reference_kernels={k: v['kernel'] for k, v in REFERENCES.items()},
        formula=FORMULA, weights=list(WEIGHTS), routed_columns=list(ROUTED), protected_columns=list(PROTECTED),
        collisions_not_applied='0.958 weights win where they collide: Lateral OA .2, PF OA .6 (no OA route), '
                               'Synovitis .8, Baker\'s .35 (Baker\'s not routed)',
        pre_execution=pre_review(),
        preregistration='register precedes prepare, stage tests and the recorded CPU replay; Gold58 labels read '
                        'only for the descriptive replay AUC, never printed or used to select. Disclosure: the '
                        'builder smoke-ran the replay code (incl. the descriptive Gold58 lines) once in lane scratch '
                        '.local/scratch/claude-opus-b1/hb191/dryreplay before registering; no registered item '
                        '(weights, routes, rule, keep rule) was chosen or changed after it - all are fixed by B\'s '
                        'build prompt.',
        failure_rule='Columns 7/11: admitted H-B-168/H-B-187 fluid survivor rule: success-only R384/R224 ranks '
                     '.65/.35, sole survivor if one fails; both R failed => O only even if compact succeeds; '
                     'O failed => r(R) only; both => .5. "Fallback when R is missing = the parent mix" read as '
                     'H-B-187 read it (existing fluid survivor rule). Other columns: H-B-189 rule unchanged. '
                     'Own 17/17/12 survivor renormalisation and R224 >3% abort unchanged.',
        alternative_not_built='Vector-only reading (both-R-failed rows fall back to the compact inside P_7/P_11) '
                              'is not built; it differs from this build in columns 7/11 on any cohort containing '
                              '>=1 study where R384 and R224 both fail (ranks of every row of 7/11 shift there). '
                              'p_branch diagnostic CSV columns 7/11 keep the parent override (compact rank on '
                              'both-R-failed rows); served columns 7/11 never use the compact.',
        public_reading=dict(reference_value=.957, expected=.958, expected_if_transfer_fails=.957,
            rule='RETAIN iff displayed >=0.957 AND displayed >= H-B-189 displayed (both read); displayed '
                 '>=0.957 but < H-B-189 displayed => DROP; missing H-B-189 reading => HOLD (conservative '
                 'reading of "when both read", B may relax); displayed <=0.956 => DROP',
            source='user-supplied readings only; no score polling'),
        timing=dict(source='H-B-174 hidden completion (submission 57251520); weights-only inheritance via H-B-189',
            inference_delta='none', child_measured_hours=None, hidden_hours=None),
        gold58=dict(exposure='EXPOSED', use='descriptive only, not a gate', weak_four=[3, 5, 6, 8],
            replay=FABLE.relative_to(ROOT).as_posix(), replay_sha256=old.digest(FABLE.read_bytes()),
            comparison='actual embedded child vs H-B-189 parent outputs, healthy 58 studies'),
        sources=['Taeyang 2026-10-17 00:2x stack request (handoff 00:26)',
                 'Fracture R7 Gold58 +0.001273 [-0.000993,+0.004161] P~0.83 (motivation only)',
                 'Effusion R route on R6 line Gold58 -0.00025 (handoff; no gain claim)'],
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        runtime_qualification='CPU cached-serving arithmetic only, no CUDA/Linux/model-forward qualification',
        gate_binding='B must pin slug-generic strict_variant_gate_r224_v145 with authentic dispatcher '
                     'builder/reviewer receipts and the stage reference-proof.json')


def decision(displayed, parent_displayed=None):
    for value in (displayed, parent_displayed):
        if value is not None:
            old.need(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                     'finite displayed public score required')
    if displayed is not None and displayed < .957: return 'DROP'
    if displayed is None or parent_displayed is None: return 'HOLD'
    return 'RETAIN' if displayed >= parent_displayed else 'DROP'


def patched(original):
    files = dict(original)
    s = files[HELPER]
    line = next(x for x in s.decode().splitlines() if x.startswith('FORMULA = '))
    s = old.replace(s, line, 'FORMULA = ' + repr(FORMULA))
    before, after = 'w = np.asarray(' + repr(h189.WEIGHTS) + ')\n', 'w = np.asarray(' + repr(WEIGHTS) + ')\n'
    old.need(s.count(before.encode()) == 2, 'parent weight vector must occur in p_branch and reference only')
    s = old.replace(s, '    ' + before + '    public = w * rr + (1.0 - w) * ss\n',
                    '    ' + after + '    public = w * rr + (1.0 - w) * ss\n')
    s = old.replace(s, '    ' + before + '    return .85 * rank(w * ', '    ' + after + '    return .85 * rank(w * ')
    s = old.replace(s, '    result[of], result[pf], result[pf & of] = pr[of], oo[pf], .5\n    return result\n',
                    '    result[of], result[pf], result[pf & of] = pr[of], oo[pf], .5\n' + SURVIVOR + SELECT
                    + '    return result\n')
    files[HELPER] = s
    s = old.replace(files[WRAPPER], old.digest(original[HELPER]), old.digest(s))
    for before, after in [
        ('VERSION = "1.1.39-r6w958q"', 'VERSION = "' + VERSION + '"'),
        ('RELEASE = "C85R7+codecs R224 R6 FLIP_ALL per-target W958"',
         'RELEASE = "C85R7+codecs R224 R6 FLIP_ALL per-target W958 EffusionR FractureR"'),
        ('KERNEL_ID = "taeyangg4/knee-lb-r6w958q-v1"', 'KERNEL_ID = "' + SLUG + '"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q"',
         'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + VERSION + '"'),
        ('formula="public w*(.65*r(R384)+.35*r(R224))+(1-w)*r(S_R6flip); w=.5 except LatOA=.2/PFOA=.6/Syn=.8/Baker=.35; own-success ranks; float32",',
         'formula="' + WRAPPER_FORMULA + '",')]:
        s = old.replace(s, before, after)
    files[WRAPPER] = s
    return files


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
    old.need(set(files) == set(original) and {k for k in files if files[k] != original[k]} == {HELPER, WRAPPER},
             'member scope')
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
        full += rbuild.diff(original[k], files[k], 'v1.1.39-r6w958q/' + k, VERSION + '/' + k)
    for name, before, after in [('notebook', pnb, nb), ('kernel-metadata.json', old.read(BASE/'kernel/kernel-metadata.json'), meta)]:
        full += rbuild.diff(old.encoded(before), old.encoded(after), 'parent/' + name, 'candidate/' + name)
    return nb, meta, files, blob, manifest, full.encode()


CHANGELOG = b'''# v1.1.41-r6w958fxq
H-B-191: admitted H-B-189 (0.958 per-target weights on H174 R6) with
w_Effusion = w_Fracture = 1.0 and the admitted H-B-187/H-B-168 fluid survivor
block for columns [7,11] (both R failed => O only). Baker's stays .35 and the
OA weights stay .2/.6 (0.958 choices win on collision). Outer .85/.15, own,
R224, R6 FULL flip, FAST, serving, datasets unchanged. Only helper arithmetic/
formula and wrapper version/id/formula/pin; 29/31 embedded files identical.
Parent backup .local/backups/hb191-build/hb191-parent. EXPOSED; no new
inference. Keep displayed >=.957 and >= H-B-189; expected .958 (.957).
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
               ('SPEC.md', SPEC.read_bytes()), (PROOF, old.encoded(reference_proof()))]
    return written, nb, meta, files, blob


def prepare():
    old.need((OUT/'registration.json').read_bytes() == old.encoded(registration()), 'register first')
    written, nb, meta, files, blob = stage_files()
    scope(nb, meta)
    for name in ('kernel/release.ipynb', 'kernel/kernel-metadata.json'):
        old.write(BACKUP/name, (BASE/name).read_bytes())
    old.write(BACKUP/'parent-admission-receipt.json', RECEIPT.read_bytes())
    for name, data in written: old.write(OUT/name, data)
    for label, contents in [('candidate', files), ('parent', parent()[1])]:
        for name, data in contents.items(): old.write(SCRATCH/(label + '-runtime')/name, data)
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], archive_sha256=old.digest(blob),
        embedded={k: old.digest(v) for k, v in files.items()}, unchanged=29, changed=[HELPER, WRAPPER],
        added=[], removed=[], parent_kernel_sha256=PARENT_KERNEL, parent_receipt_sha256=PARENT_RECEIPT,
        reference_proof_sha256=old.digest(old.encoded(reference_proof())),
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
    old.need(len(proofs) == 2, 'reference proof blocks')
    pins.recheck()
    return dict(status='PASS_CPU_VALIDATE', kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),
                identical=29, changed=2, reference_blocks=len(proofs),
                metadata_raw_values_identical_except_id_title=True)


def math_functions(files):
    """Test-only extraction; the real replay imports the actual embedded archives."""
    return reuse.math_functions(files)


def oracle(arrays, masks):
    """Independent column-wise pandas oracle, NaN failure representation."""
    import numpy as np
    import pandas as pd
    def rank(v, failed):
        out = np.full(len(v), np.nan)
        out[~failed] = pd.Series(v[~failed]).rank(method='average', pct=True).to_numpy()
        return out
    result = np.empty_like(arrays[0])
    rf, f224, sf, *ofs = masks
    for col, w in enumerate(WEIGHTS):
        a, b, s, *owns = [rank(v[:, col], f) for v, f in zip(arrays, masks)]
        rr = np.where(rf, b, np.where(f224, a, .65*a + .35*b)); rr[rf & f224] = .5
        if col in ROUTED:  # admitted fluid survivor rule: compact never used
            public = rr.copy(); pf = rf & f224
        else:
            public = w*rr + (1-w)*s
            public[rf & f224] = s[rf & f224]; public[sf] = rr[sf]
            pf = rf & f224 & sf
        public[pf] = .5
        den = sum(weight*(~f).astype(int) for weight, f in zip((17, 17, 12), ofs))
        own = sum(weight*np.nan_to_num(v) for weight, v in zip((17, 17, 12), owns))/np.maximum(den, 1)
        pr, oo = rank(public, pf), rank(own, den == 0)
        out = .85*pr + .15*oo
        out[den == 0] = pr[den == 0]; out[pf] = oo[pf]; out[pf & (den == 0)] = .5
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
            assert n==2 and bits[1] and 'Raptor-224 failed on more than 3% of studies' in str(exc),str(exc)
            checks.append(dict(key=key,status='EXPECTED_R224_ABORT'))
            continue
        assert not(n==2 and bits[1]),'R224 threshold bypass'
        assert np.isfinite(result).all() and np.all((result>=0)&(result<=1))
        if not any(bits): assert np.array_equal(result,h.reference(*aa))
        outputs[key]=result
        raw=c.p.csv_bytes(ids[:n],result)
        checks.append(dict(key=key,status='PASS',all_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
            protected_csv_sha256=csv_hash(raw,(0,1,2,3,4,5,6,8,9,10)),
            routed_csv_sha256=csv_hash(raw,(7,11)),effusion_csv_sha256=csv_hash(raw,(7,))))
np.savez(dest.with_suffix('.npz'),**outputs)
import torch
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
dest.with_suffix('.json').write_text(json.dumps(dict(version=c.VERSION,checks=checks,cuda_initialized=False,embedded_imports=True),indent=2))
print(json.dumps(dict(status='PASS',version=c.VERSION,cases=len(checks),successes=len(outputs),expected_aborts=len(checks)-len(outputs),cuda_initialized=False)))
'''


def replay():
    import importlib.util
    import itertools
    import numpy as np
    import pandas as pd
    import torch
    from tools import c85r7_gold58_candidates as cache
    old.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only required')
    valid = validate()
    reg, common, failed, arms = cache.context()
    r6_reg, compact, r6_auth = reuse.authenticated_r6()
    ids = reg['binding']['ids']
    old.need(r6_reg['binding']['ids'] == ids and len(ids) == 58, 'R6/H174 Gold58 ordered cohort drift')
    arrays = [np.asarray(common[k], float) for k in ('R', 'R224')] + [compact] + [np.asarray(common[k], float) for k in MEMBERS[3:]]
    r7_arrays = [a.copy() for a in arrays]; r7_arrays[2] = np.asarray(arms['flip-1']['predictions'], float)
    masks = [np.asarray(failed[k], bool) if k in failed else np.zeros(58, bool) for k in MEMBERS]
    old.need(not any(m.any() for m in masks), 'Gold58 observed failures require review')
    old.need(not any(np.all(a == .5, axis=1).any() for a in arrays[:2]), 'unexpected real R sentinel')
    old.need(not np.array_equal(r7_arrays[2], arrays[2]), 'R7 and R6 compact must differ for the any-compact proof')
    for name, digest in r6_auth['admitted_H174_exact_contracts'].items():
        old.need(old.digest(parent()[1][name]) == digest and old.digest(payload()[2][name]) == digest,
                 'R6 measured contract differs: ' + name)
    # Same authenticated inputs as the admitted H-B-189 replay; its healthy-58 output is pinned below.
    old.pin(PARENT_REPLAY, PARENT_REPLAY_SHA)
    old.need(old.read(PARENT_REPLAY)['inputs_sha256'] == [old.digest(a.tobytes()) for a in arrays], 'H189 replay input drift')
    for label in REFERENCES:
        for name, data in reference(label)[1].items(): old.write(SCRATCH/(label + '-runtime')/name, data)
    for label, values in [('r6', arrays), ('r7', r7_arrays)]:
        np.savez(SCRATCH/(label + '-real-input.npz'), ids=np.asarray(ids), **{'a' + str(i): a for i, a in enumerate(values)})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONPATH', None)
    plan = {'parent': ('parent', 'r6'), 'candidate': ('candidate', 'r6'), 'h187_r6': ('h187', 'r6'),
            'h187_r7': ('h187', 'r7'), 'h168_r7': ('h168', 'r7')}
    runs, outputs = {}, {}
    for label, (runtime, inp) in plan.items():
        output = SCRATCH/('real-' + label)
        proc = subprocess.run([sys.executable, '-B', '-I', '-c', REPLAY_SCRIPT, str(SCRATCH/(runtime + '-runtime')),
                               str(SCRATCH/(inp + '-real-input.npz')), str(output)], cwd=SCRATCH, env=env,
                              capture_output=True, text=True)
        old.need(proc.returncode == 0, 'isolated replay failed (' + label + '): ' + proc.stderr[-6000:])
        runs[label] = old.read(output.with_suffix('.json'))
        with np.load(output.with_suffix('.npz'), allow_pickle=False) as data: outputs[label] = {k: data[k] for k in data.files}
    old.need(runs['candidate']['version'] == VERSION and runs['parent']['version'] == '1.1.39-r6w958q', 'runtime versions')
    checks = {label: {row['key']: row for row in run['checks']} for label, run in runs.items()}
    col = lambda x, cols: np.ascontiguousarray(x[:, list(cols)]).tobytes()
    cases = []
    for n in (2, len(ids)):
        for case, bits in enumerate(itertools.product((False, True), repeat=6)):
            key = f'n{n}-case{case}'
            row = checks['candidate'][key]
            old.need(all(checks[label][key]['status'] == row['status'] for label in checks), 'failure gate differs: ' + key)
            if row['status'] == 'EXPECTED_R224_ABORT':
                cases.append(dict(key=key, failed=[m for m, b in zip(MEMBERS, bits) if b], status=row['status'])); continue
            current = [np.zeros(n, bool) for _ in arrays]
            for mask, bit in zip(current, bits): mask[0] = bit
            actual = outputs['candidate'][key]
            old.need(actual.tobytes() == oracle([a[:n] for a in arrays], current).tobytes(), 'independent pandas byte parity: ' + key)
            old.need(col(actual, PROTECTED) == col(outputs['parent'][key], PROTECTED), 'protected-ten float64 bytes: ' + key)
            old.need(row['protected_csv_sha256'] == checks['parent'][key]['protected_csv_sha256'], 'protected-ten CSV: ' + key)
            for label in ('h187_r6', 'h187_r7'):
                old.need(col(actual, ROUTED) == col(outputs[label][key], ROUTED), 'routed 7/11 float64 vs ' + label + ': ' + key)
                old.need(row['routed_csv_sha256'] == checks[label][key]['routed_csv_sha256'], 'routed 7/11 CSV vs ' + label + ': ' + key)
            old.need(col(actual, (7,)) == col(outputs['h168_r7'][key], (7,)), 'Effusion float64 vs H168: ' + key)
            old.need(row['effusion_csv_sha256'] == checks['h168_r7'][key]['effusion_csv_sha256'], 'Effusion CSV vs H168: ' + key)
            cases.append(dict(key=key, failed=[m for m, b in zip(MEMBERS, bits) if b], status='PASS',
                independent_pandas_bytes_exact=True, protected_ten_vs_h189_float64_csv_exact=True,
                routed_7_11_vs_h187_r6_and_r7_float64_csv_exact=True, effusion_vs_h168_float64_csv_exact=True,
                candidate_output_sha256=row['all_sha256']))
    healthy = 'n58-case0'
    old.need(old.digest(outputs['parent'][healthy].tobytes()) == PARENT_HEALTHY58_SHA, 'parent != admitted H189 replay output')
    # Real-data p_branch truth table with the registered weights (healthy 58).
    pb = math_functions(payload()[2])['p_branch'](*arrays[:3], *masks[:3])
    rank = lambda a: pd.DataFrame(a).rank(method='average', pct=True).to_numpy()
    rr = .65*rank(arrays[0]) + .35*rank(arrays[1]); w = np.asarray(WEIGHTS)
    old.need(pb.tobytes() == (w*rr + (1-w)*rank(arrays[2])).tobytes(), 'real p_branch weight truth table')
    old.need(col(pb, ROUTED) == col(rr, ROUTED), 'real p_branch 7/11 == rr bit-exactly')
    # Descriptive EXPOSED Gold58 comparison (not a gate); labels never printed.
    spec = importlib.util.spec_from_file_location('hb191_fable', FABLE)
    fable = importlib.util.module_from_spec(spec); spec.loader.exec_module(fable)
    label_ids, labels = fable.d.gold.gold(); labels = np.asarray(labels, float)
    old.need(list(label_ids) == ids and labels.shape == (58, 12), 'Gold58 labels/roster alignment')
    metrics = {}
    for key, value in (('parent', outputs['parent'][healthy]), ('child', outputs['candidate'][healthy])):
        aucs = np.asarray([fable.auc(labels[:, j], value[:, j]) for j in range(12)])
        metrics[key] = dict(macro=float(aucs.mean()), weak_four=float(aucs[fable.WEAK].mean()),
                            per_target={fable.LAB[j]: float(aucs[j]) for j in ROUTED})
    metrics['delta'] = {k: metrics['child'][k] - metrics['parent'][k] for k in ('macro', 'weak_four')}
    metrics['delta']['per_target'] = {k: metrics['child']['per_target'][k] - metrics['parent']['per_target'][k]
                                      for k in metrics['child']['per_target']}
    old.need(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    report = dict(status='PASS', kernel_sha256=valid['kernel_sha256'], studies=[2, 58], targets=12,
        scenarios=64, two_study_successes=sum(c['status'] == 'PASS' for c in cases if c['key'].startswith('n2-')),
        two_study_expected_r224_aborts=sum(c['status'] != 'PASS' for c in cases if c['key'].startswith('n2-')),
        full_cohort_successes=sum(c['status'] == 'PASS' for c in cases if c['key'].startswith('n58-')),
        independent_pandas_bytes_exact=True, protected_ten_vs_h189_float64_csv_exact=True,
        routed_7_11_vs_h187_r6_and_r7_float64_csv_exact=True, effusion_vs_h168_float64_csv_exact=True,
        parent_healthy58_equals_admitted_h189_replay=True, real_p_branch_weight_truth_table=True,
        cases=cases, embedded_runs=runs, cohort_sha256=old.digest(old.encoded(ids)),
        historical_binding_sha256=reg['binding_sha256'], r6_authentication=r6_auth,
        preserved_pins_sha256=cache.PINS_SHA,
        inputs_sha256={'R6': [old.digest(a.tobytes()) for a in arrays], 'R7': [old.digest(a.tobytes()) for a in r7_arrays]},
        gold58=metrics, exposure='EXPOSED', clean_oof=False, gold58_gate=False,
        limitation='Authenticated real cached-serving arithmetic and actual embedded imports, single-row injected '
                   'failures; no model-forward/CUDA/Linux qualification or runtime estimate. Gold58 labels read '
                   'internally for descriptive AUC only, never printed.')
    old.write(SCRATCH/'replay.json', old.encoded(report))
    return {k: v for k, v in report.items() if k not in ('cases', 'embedded_runs', 'r6_authentication')}


def admit():
    valid = validate(); smoke = old.read(SCRATCH/'replay.json')
    old.need(smoke['status'] == 'PASS' and smoke['kernel_sha256'] == valid['kernel_sha256'], 'CPU replay missing/stale')
    old.need(smoke['protected_ten_vs_h189_float64_csv_exact'] and smoke['routed_7_11_vs_h187_r6_and_r7_float64_csv_exact']
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
    result = {'register': register, 'prepare': prepare, 'validate': validate, 'cpu-replay': replay, 'admit': admit}[args.command]()
    old.write(SCRATCH/(args.command + '-result.json'), old.encoded(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
