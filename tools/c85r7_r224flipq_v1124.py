"""H-B-167 builder 1.0.0: strict R224 parent + the hidden-completed FULL flip.

Changelog: one flag patch, chained pins and version metadata only. CPU only.
No original/frozen files are changed. Independent exact-diff gate is deferred.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import io
import json
import math
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
from tools import c85r7_r224q_v1123 as parent_builder
from tools import strict_variant_gate as gate
from tools import hb145_output_io as output_io

VERSION = '1.1.24-r224flipq'
SLUG = 'taeyangg4/knee-lb-r224flipq-v1'
BASE = parent_builder.OUT
REFERENCE = flipq.OUT
OUT = BASE.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb167-build/hb167'
BACKUP = ROOT / '.local/backups/hb167-build/hb167'
PARENT_RECEIPT_SHA = 'bd76c883e7a3663a45ffee606dede8951459c89d9942ebea944c1f5989217b37'
FLIP_RECEIPT_SHA = '42b91ecac6fbbff32c41225f7f5bed7306259712e54f93286c26d22492c22e79'
PORT = 'exp169_r7_port.py'
CHAIN = 'exp169_release_c85r7.py'
WRAPPER = 'exp169_release_c85r7codec.py'


def authenticated(stage, receipt_sha):
    old.pin(stage / 'build-receipt.json', receipt_sha)
    rec = old.read(stage / 'build-receipt.json')
    old.need(rec['kernel_files'] == gate.kernel(stage, gate.Inputs(ROOT))[2]['kernel_files'], 'source kernel drift')
    nb = old.read(stage / 'kernel/release.ipynb')
    files, blob, assignments = old.unpack(flipq.restored(nb))
    old.need(rec['archive_sha256'] == old.digest(blob), 'source archive drift')
    old.need(rec['embedded'] == {k: old.digest(v) for k,v in files.items()}, 'source embedded drift')
    return nb, files, blob, assignments


def allowed_files(original, reference):
    files = dict(original)
    files[PORT] = old.replace(original[PORT], old.FLAG_OLD, old.FLAG_NEW)
    old.need(files[PORT] == reference[PORT], 'FULL flip differs from hidden-completed v1.1.22')
    files[CHAIN] = old.replace(original[CHAIN], old.digest(original[PORT]), old.digest(files[PORT]))
    old.need(files[CHAIN] == reference[CHAIN], 'chained serving file differs from flip reference')
    s = old.replace(original[WRAPPER], old.digest(original[CHAIN]), old.digest(files[CHAIN]))
    for before, after in [
        ('VERSION = "1.1.23-r224q"', 'VERSION = "' + VERSION + '"'),
        ('RELEASE = "C85R7+codecs R224"', 'RELEASE = "C85R7+codecs R224 FLIP_ALL"'),
        ('KERNEL_ID = "' + parent_builder.SLUG + '"', 'KERNEL_ID = "' + SLUG + '"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.23-r224q"',
         'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + VERSION + '"')]:
        s = old.replace(s, before, after)
    files[WRAPPER] = s
    return files


def payload():
    nb, original, old_blob, assignments = authenticated(BASE, PARENT_RECEIPT_SHA)
    reference = authenticated(REFERENCE, FLIP_RECEIPT_SHA)[1]
    files = allowed_files(original, reference)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buf, 'w') as target:
        for info in source.infolist():
            target.writestr(info, files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before, after in [
        ('BLOB = ' + repr(assignments['BLOB']), 'BLOB = ' + repr(base64.b64encode(blob).decode())),
        ('PINS = ' + repr(assignments['PINS']), 'PINS = ' + repr({k:old.digest(v) for k,v in files.items()})),
        (repr(old.digest(old_blob)), repr(old.digest(blob)))]:
        code = old.replace(code, before, after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    meta = old.read(BASE / 'kernel/kernel-metadata.json')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta, files, blob


def scope(nb, meta):
    expected, expected_meta, files, blob = payload()
    old.need(nb == expected and meta == expected_meta, 'forbidden notebook/metadata diff')
    actual, actual_blob, _ = old.unpack(flipq.restored(nb))
    old.need(actual == files and actual_blob == blob, 'forbidden embedded diff')
    original = authenticated(BASE, PARENT_RECEIPT_SHA)[1]
    old.need(set(actual) == set(original) and len(actual) == 31, 'embedded member set changed')
    for name, data in actual.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    return files, blob


def registration():
    return dict(version=VERSION, item='H-B-167', kernel_id=SLUG, parent=BASE.relative_to(ROOT).as_posix(),
        parent_receipt_sha256=PARENT_RECEIPT_SHA, flip_reference=REFERENCE.relative_to(ROOT).as_posix(),
        flip_reference_receipt_sha256=FLIP_RECEIPT_SHA, formula=parent_builder.FORMULA,
        purpose='final candidate if R224 is ALIVE >=0.957; FULL flip supersedes the old half-flip plan',
        registered_change='R7 existing --flip-tta unconditional for all four folds; exact v1.1.22-flipallq bytes and targets, including MCL plain-only',
        permitted_metadata='version/release/id/output strings and dependent hash pins only',
        public_reading=dict(parent_alive_min=.957, candidate_min_delta=-.001, parent_reading=.957,
                            source='Taeyang supplied 2026-10-15 15:22; no new score read', keep_min=.956),
        gold58=dict(reference='plain v1.1.23-r224q final blend', macro_min_delta=-.001, weak_four_min_delta=-.002),
        timing=dict(tag='HB167_N32', max_projected_hours=8.5, actual_kaggle_limit_hours=9,
                    projection='measured R7 flip increment plus measured R224 share; no double count of baseline', measured=False),
        empirical_hidden=dict(source='Taeyang 2026-10-15 15:22', r224_submission=57200805,
            flip_submission=57198522, r224_completed_upper_hours=5.3, flip_completed_upper_hours=6.3,
            planning_upper_hours=7.1, derived_additive_hours=round(5.3+6.3-19357/3600,6),
            method='Additive proxy 5.3+6.3-(19357/3600); separate user-supplied conservative planning estimate approximately 7.1 h',
            limitation='Conditional additive planning estimate from two completed upper bounds; not a measured combined run or guaranteed upper bound. Hardware/load interactions unknown.'),
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        gate_binding='DEFERRED: separate exact-diff-pin extension job; no gate copy produced')


def decision(parent_score, candidate_score, macro_delta, weak_four_delta, projected_hours):
    numbers = [parent_score, candidate_score, macro_delta, weak_four_delta, projected_hours]
    old.need(all(type(x) in (int,float) and math.isfinite(x) for x in numbers), 'finite measured decisions required')
    old.need(0 <= parent_score <= 1 and 0 <= candidate_score <= 1 and projected_hours > 0, 'decision domain')
    return (parent_score >= .957 and candidate_score >= parent_score-.001 and
            macro_delta >= -.001 and weak_four_delta >= -.002 and projected_hours <= 8.5)


def artifacts():
    nb, meta, files, blob = payload()
    original_nb, original, _, _ = authenticated(BASE, PARENT_RECEIPT_SHA)
    changed = sorted(k for k in files if files[k] != original[k])
    old.need(set(changed) == {PORT,CHAIN,WRAPPER}, 'unexpected changed members')
    manifest = dict(version=VERSION, added=[], removed=[],
        changed=[dict(name=k, before_sha256=old.digest(original[k]), after_sha256=old.digest(files[k])) for k in changed],
        unchanged=[dict(name=k,sha256=old.digest(v)) for k,v in sorted(files.items()) if k not in changed],
        exact_flip_reference=True, rules='Exact generated bytes only; any deviation fails scope()')
    full = 'ALL 31 EMBEDDED FILES: 28 IDENTICAL, 3 REGISTERED CHANGES, 0 ADDED/REMOVED\n'
    for k in sorted(files):
        full += f'{k}: {old.digest(original[k])} -> {old.digest(files[k])} ' + ('IDENTICAL' if k not in changed else 'REGISTERED_CHANGE') + '\n'
        full += parent_builder.diff(original[k],files[k],BASE.name+'/'+k,OUT.name+'/'+k)
    for name,before,after in [('notebook',original_nb,nb), ('kernel-metadata.json',old.read(BASE/'kernel/kernel-metadata.json'),meta)]:
        full += parent_builder.diff(old.encoded(before),old.encoded(after),BASE.name+'/'+name,OUT.name+'/'+name)
    return nb,meta,files,blob,manifest,full.encode()


def prepare():
    nb,meta,files,blob,manifest,full = artifacts()
    for name in ('build-receipt.json','kernel/release.ipynb','kernel/kernel-metadata.json'):
        old.write(BACKUP/BASE.name/name,(BASE/name).read_bytes())
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('registration.json',gate.encoded(registration())),('new-members.json',gate.encoded(manifest)),
        ('full-diff-vs-v1.1.23-r224q.txt',full)]:
        old.write(OUT/name,data)
    rec = dict(status='PREPARED_CPU_ONLY',version=VERSION,builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'], archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()}, unchanged=28, changed=[x['name'] for x in manifest['changed']],
        added=[],removed=[],qualified_for_submission=False, exposure='EXPOSED',clean_oof=False)
    old.write(OUT/'build-receipt.json',gate.encoded(rec))
    old.write(OUT/'CHANGELOG.md', b'# 1.1.24-r224flipq / builder 1.0.0\nOne candidate from hidden-completed v1.1.23-r224q: exact v1.1.22-flipallq R7 flag plus chained hash pins and version metadata. All 28 other embedded files, FAST cell/guard and bootstrap mechanics unchanged. No frozen file edits; parent backup .local/backups/hb167-build/hb167/. Separate strict-gate pin extension required. CPU-only build; GPU/Kaggle never launched.\n')
    for name,data in files.items():
        old.write(SCRATCH/'candidate-runtime'/name,data)
    return validate()


def validate():
    nb,meta,files,blob,manifest,full = artifacts()
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('registration.json',gate.encoded(registration())),('new-members.json',gate.encoded(manifest)),
        ('full-diff-vs-v1.1.23-r224q.txt',full)]:
        old.need((OUT/name).read_bytes() == data,'stage drift: '+name)
    scope(old.read(OUT/'kernel/release.ipynb'),old.read(OUT/'kernel/kernel-metadata.json'))
    rec=old.read(OUT/'build-receipt.json')
    old.need(rec['builder_sha256']==old.digest(Path(__file__).read_bytes()),'builder drift')
    old.need(rec['embedded']=={k:old.digest(v) for k,v in files.items()} and rec['archive_sha256']==old.digest(blob),'receipt drift')
    old.need(rec['kernel_files']==gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],'kernel drift')
    old.need(rec['unchanged']==28 and rec['added']==rec['removed']==[] and rec['qualified_for_submission'] is False,'receipt scope')
    return dict(status='PASS_CPU_VALIDATE',unchanged=28,changed=3,added=0,
                kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),qualified_for_submission=False)


def smoke():
    from tools import c85r7_flipall_cpu_smoke as pixel
    validate()
    adapter=SimpleNamespace(**{n:getattr(old,n) for n in dir(old) if not n.startswith('__')})
    adapter.OUT,adapter.SCRATCH,adapter.validate=OUT,SCRATCH,validate
    adapter.unpack=lambda nb:old.unpack(flipq.restored(nb))
    with patch.object(pixel,'b',adapter):
        pixel.main()
    # Existing final/own arithmetic is unchanged and is replayed on sealed real3
    # sufficient statistics. This does not claim new full-study R224 predictions.
    with patch.object(old,'OUT',OUT),patch.object(old,'SCRATCH',SCRATCH/'replay'),patch.object(old,'validate',validate):
        original_unpack=old.unpack
        def unpack(nb):
            if len(nb['cells'])==3:
                nb=flipq.restored(nb)
            return original_unpack(nb)
        with patch.object(old,'unpack',unpack):
            replay=old.replay()
    report=dict(status='PASS',kernel_sha256=validate()['kernel_sha256'],
        pixel=old.read(SCRATCH/'flip-cpu-smoke.json'),real3_arithmetic=replay,
        limitation='Actual four R7 checkpoints, reduced real DICOM windows on CPU; unchanged outer/own formula on sealed real3. Full R224+flip blend and GPU timing are not measured.')
    output_io.atomic_json(SCRATCH/'smoke.json',report)
    return dict(status='PASS',folds=4,real_studies=3,decoded_dicoms=9,real3_arithmetic=replay['status'])


def admit():
    value=validate()
    smoke_result=old.read(SCRATCH/'smoke.json')
    old.need(smoke_result['status']=='PASS' and smoke_result['kernel_sha256']==value['kernel_sha256'],'CPU smoke missing/stale')
    return dict(status='HOLD_INDEPENDENT_REVIEW_EXACT_DIFF_PIN_AND_GPU_MEASUREMENTS',qualified_for_submission=False,
                kernel_sha256=value['kernel_sha256'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','register','validate','cpu-smoke','admit'])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    if args.command in ('prepare','admit'):
        old.need(args.dry_run,'CPU dry-run required')
    result={'prepare':prepare,'register':validate,'validate':validate,'cpu-smoke':smoke,'admit':admit}[args.command]()
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
