"""H-B-175 builder v1.0.0: one R224 + sequential R7/R6 flip candidate.

Changelog: additive R6 bundle identity, execution and successful-row slot ranks.
No original/frozen source edits. Parent and all historical receipts are read-only.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_flipall_v1115 as old, c85r7_flipallq_v1122 as flipq
from tools import strict_variant_gate as gate
from tools.hb145_output_io import atomic_json

VERSION = '1.1.31-r224r76fq'
SLUG = 'taeyangg4/knee-lb-r224r76fq-v1'
BASE = ROOT / '.local/scratch/astra-b5/hb167-gate/gate-copy-v143'
OUT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7' / ('v'+VERSION)
SCRATCH = ROOT / '.local/scratch/astra-b5/hb175'
PARENT_RECEIPT = ROOT / '.local/state/gates/knee-lb-r224flipq-v1.json'
PARENT_RECEIPT_SHA = 'd76d96d085d0b7c8f467d292f94f2c5e15f649176e6f29c63591b2e8df571378'
PARENT_KERNEL = {'kernel-metadata.json':'361bc5f487244d178669c0faec6be935207da6db14b0ed72888dea9b7568c078',
                 'release.ipynb':'c96f5d0d9ed53c5a57870b97eca4e03fd4226c131fa79489d6bea1d02956c95a'}
R6_ID = 'fe055406fad6a12f'
R6_SHA = '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
R6_PRIVATE = 'taeyangg4/knee-r6-fe055406-v1'
R6_PUBLIC = 'rostigor/rsna-knee-r6-fe055406'
R6_LOCAL = ROOT / 'storage/external/heliosli-r6-fe055406-v1'
PORT, CHAIN, WRAPPER, MATH = 'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py', 'exp169_r224.py'
FORMULA = '0.85*r(P)+0.15*r(O); P=0.5*(0.65*r(R384)+0.35*r(R224))+0.5*(0.5*r(R7flip)+0.5*r(R6flip)); O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46'


def parent():
    old.pin(PARENT_RECEIPT, PARENT_RECEIPT_SHA)
    receipt = old.read(PARENT_RECEIPT)
    old.need(receipt['status']=='PASS' and receipt['kernel_id']==SLUG.replace('r224r76fq','r224flipq'), 'parent admission')
    for name, digest in PARENT_KERNEL.items(): old.pin(BASE/'kernel'/name, digest)
    nb = old.read(BASE/'kernel/release.ipynb')
    files, blob, assignments = old.unpack(flipq.restored(nb))
    expected = receipt['comparison_sha256']['stage']
    old.need(all(expected['embedded/'+k]==old.digest(v) for k,v in files.items()), 'admitted embedded identity')
    return nb,files,blob,assignments


def registration():
    return dict(item='H-B-175', version=VERSION, kernel_id=SLUG, parent=str(BASE.relative_to(ROOT)),
        parent_receipt_sha256=PARENT_RECEIPT_SHA, formula=FORMULA,
        registered_change='ADD R6 flip beside R7 flip, sequential; average separate successful-row ranks per target; no extra rank of slot average',
        failures='one bundle fails: surviving bundle rank alone; both fail: parent slot failure and existing branch survivor rules; R224 >3% abort unchanged',
        public_reading=dict(parent=.957, keep_min=.956, candidate_min_delta=-.001, source='user supplied; no score query'),
        r6=dict(bundle_id=R6_ID, manifest_sha256=R6_SHA, private_copy=R6_PRIVATE, public_source=R6_PUBLIC,
                original='heliosli/rsna-knee-r6-frozen-fe055406 version 1', licence='other', flip_tta=True),
        timing=dict(tag='HB175_N32', baseline_hours=6.73, baseline='H-B-167 hidden 57224580; user reported <=6.73h',
                    hard_limit_hours=9, available_increment_hours=2.27, measured=False,
                    method='6.73h + measured R6 flip N32 seconds * authenticated R7 hidden/local-N32 scale / 3600',
                    admission_requires='independent review, actual 4070 N32 R6 measurement, authenticated real-output post-run replay, projected <=9h'),
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False, gate_binding='NOT REQUESTED; no gate copy')


def allowed_files(original):
    files = dict(original)
    def edit(name, before, after): files[name] = old.replace(files[name], before, after)
    # Same adapter and frozen worker: bundle identity is an explicit per-call argument.
    edit(PORT, 'MANIFEST_SHA256: str | None = None',
         'MANIFEST_SHA256: str | None = None\nR6_BUNDLE_ID = '+repr(R6_ID)+'\nR6_MANIFEST_SHA256 = '+repr(R6_SHA)+'\nR6_DATASET_SLUG = '+repr(R6_PRIVATE)+'\nR6_PUBLIC_SLUG = '+repr(R6_PUBLIC))
    edit(PORT, 'source_dir: str | Path = SOURCE) -> tuple[dict, dict]:',
         'source_dir: str | Path = SOURCE, *, bundle_id: str = BUNDLE_ID, dataset_slug: str = DATASET_SLUG) -> tuple[dict, dict]:')
    edit(PORT, '    bundle = Path(bundle_dir)\n    missing',
         '    require(bundle_id in (BUNDLE_ID, R6_BUNDLE_ID), "unregistered bundle")\n'
         '    if bundle_id == R6_BUNDLE_ID:\n'
         '        require(dataset_slug in (R6_DATASET_SLUG, R6_PUBLIC_SLUG), "unregistered R6 dataset")\n'
         '        require(expected_manifest_sha256 in (None, R6_MANIFEST_SHA256), "R6 pin override")\n'
         '        expected_manifest_sha256 = R6_MANIFEST_SHA256\n'
         '    else:\n        require(dataset_slug == DATASET_SLUG, "unregistered R7 dataset")\n'
         '    bundle = Path(bundle_dir)\n    missing')
    edit(PORT, 'manifest.get("bundle_id") == BUNDLE_ID', 'manifest.get("bundle_id") == bundle_id')
    edit(PORT, 'identity = {"dataset_slug": DATASET_SLUG, "bundle_id": BUNDLE_ID,', 'identity = {"dataset_slug": dataset_slug, "bundle_id": bundle_id,')
    edit(PORT, '"dataset_slug": DATASET_SLUG, "bundle_id": BUNDLE_ID, "licence":', '"dataset_slug": identity["dataset_slug"], "bundle_id": identity["bundle_id"], "licence":')
    edit(PORT, 'expected_manifest_sha256: str | None = None) -> list[dict[str, str | float]]:',
         'expected_manifest_sha256: str | None = None, bundle_id: str = BUNDLE_ID,\n                  dataset_slug: str = DATASET_SLUG) -> list[dict[str, str | float]]:')
    edit(PORT, 'verify_bundle(bundle_dir, expected_manifest_sha256, source_dir)',
         'verify_bundle(bundle_dir, expected_manifest_sha256, source_dir, bundle_id=bundle_id, dataset_slug=dataset_slug)')
    edit(CHAIN, old.digest(original[PORT]), old.digest(files[PORT]))
    # Parameterize the existing authenticated reader, not a parallel reader with weaker checks.
    begin = files[CHAIN].decode().index('def authenticated_pr7(')
    end = files[CHAIN].decode().index('\n\ndef candidate_receipt', begin)
    prior = files[CHAIN].decode()[begin:end]
    updated = prior.replace('def authenticated_pr7(directory):',
        'def authenticated_pr7(directory, *, member="S_R7", bundle_id=r7.BUNDLE_ID,\n'
        '                      dataset_slug=r7.DATASET_SLUG, manifest_pin=r7.MANIFEST_SHA256):')
    updated = updated.replace('    import csv\n', '    need(member in ("S_R7", "S_R6"), "invalid S member")\n    import csv\n')
    updated = updated.replace('"s_r7/', 'member.lower() + "/').replace('"s_r7"', 'member.lower()')
    updated = updated.replace('"r7-identity.json"', '(member[2:].lower() + "-identity.json")')
    updated = updated.replace('r7.DATASET_SLUG', 'dataset_slug').replace('r7.BUNDLE_ID', 'bundle_id').replace('r7.MANIFEST_SHA256', 'manifest_pin')
    # Restore default expressions altered by the body substitution above.
    updated = updated.replace('bundle_id=bundle_id', 'bundle_id=r7.BUNDLE_ID').replace('dataset_slug=dataset_slug, manifest_pin=manifest_pin', 'dataset_slug=r7.DATASET_SLUG, manifest_pin=r7.MANIFEST_SHA256')
    updated = updated.replace('"S_R7": sf', 'member: sf').replace('["R", "S_R7"]', '["R", member]').replace('("R", "S_R7")', '("R", member)')
    # Path expressions require parentheses after /, since + has lower precedence.
    updated = updated.replace('work / member.lower() + "/', 'work / (member.lower() + "/')
    # Close just the path string expression, preserving subsequent suffix operations.
    import re
    updated = re.sub(r'work / \(member.lower\(\) \+ "(/[^"\n]*)"', r'work / (member.lower() + "\1")', updated)
    edit(CHAIN, prior, updated)
    helper = '''def r6_mount():
    """Private copy preferred; public source accepted. Every selected mount is pinned."""
    root = Path("/kaggle/input")
    for slug in (r7.R6_DATASET_SLUG, r7.R6_PUBLIC_SLUG):
        name = slug.split("/")[-1]
        present = (root / name).exists() or (root / "datasets" / slug).exists() or any(root.glob("datasets/*/" + name))
        if present:
            return v5q._mount(slug, "manifest.json"), slug
    raise ValueError("R6 dataset mount missing")


'''
    edit(CHAIN, 'def preflight(runtime, output):', helper+'def preflight(runtime, output):')
    edit(CHAIN, '    write_once(work / "r7-identity.json", json_bytes(identity))',
         '    write_once(work / "r7-identity.json", json_bytes(identity))\n'
         '    r6_bundle, r6_slug = r6_mount()\n'
         '    _, r6_identity = r7.verify_bundle(r6_bundle, source_dir=source, bundle_id=r7.R6_BUNDLE_ID, dataset_slug=r6_slug)\n'
         '    write_once(work / "r6-identity.json", json_bytes(r6_identity))')
    # Slot average is fed directly into P (no second rank at this level).
    edit(MATH, files[MATH].decode().split('FORMULA = ')[1].split('\n')[0], repr(FORMULA))
    edit(MATH, 'def p_branch(r, r224, ours, rf, f224, sf):',
         'def s_slot(ours, r6, sf, f6):\n'
         '    a, b = c85.scored_rank(ours, sf), c85.scored_rank(r6, f6)\n'
         '    slot = .5 * a + .5 * b\n'
         '    slot[sf], slot[f6], slot[sf & f6] = b[sf], a[f6], .5\n'
         '    return slot, sf & f6\n\n\n'
         'def p_branch(r, r224, ours, rf, f224, sf, r6, f6):')
    edit(MATH, '    ss = c85.scored_rank(ours, sf)', '    ss, sf = s_slot(ours, r6, sf, f6)')
    edit(MATH, 'def blend(ids, r, r224, ours, exp143, exp155, q, failures=None):', 'def blend(ids, r, r224, ours, exp143, exp155, q, failures=None, *, r6):')
    edit(MATH, "'EXP155', 'Q_A5q']", "'EXP155', 'Q_A5q', 'S_R6']")
    edit(MATH, '(r, r224, ours, exp143, exp155, q))]', '(r, r224, ours, exp143, exp155, q, r6))]')
    edit(MATH, 'r, r224, ours, exp143, exp155, q = values', 'r, r224, ours, exp143, exp155, q, r6 = values')
    edit(MATH, 'rf, f224, sf, f143, f155, fq = masks', 'rf, f224, sf, f143, f155, fq, f6 = masks')
    edit(MATH, 'public = p_branch(r, r224, ours, rf, f224, sf)', 'public = p_branch(r, r224, ours, rf, f224, sf, r6, f6)')
    edit(MATH, 'masks[3:])', 'masks[3:6])')
    edit(MATH, 'pf, of = rf & f224 & sf, denom == 0', 'pf, of = rf & f224 & sf & f6, denom == 0')
    edit(MATH, 'def reference(r, r224, ours, exp143, exp155, q):', 'def reference(r, r224, ours, exp143, exp155, q, *, r6):')
    edit(MATH, '+ .5 * rank(ours))', '+ .5 * (.5 * rank(ours) + .5 * rank(r6)))')
    edit(WRAPPER, old.digest(original[CHAIN]), old.digest(files[CHAIN]))
    edit(WRAPPER, old.digest(original[MATH]), old.digest(files[MATH]))
    # Required version / execution provenance metadata.
    files[WRAPPER] = files[WRAPPER].replace(b'1.1.24-r224flipq', VERSION.encode()).replace(b'taeyangg4/knee-lb-r224flipq-v1',SLUG.encode())
    edit(WRAPPER, 'RELEASE = "C85R7+codecs R224 FLIP_ALL"', 'RELEASE = "C85R7+codecs R224 R7_R6 FLIP_ALL"')
    edit(WRAPPER, 'DATASETS = [*base.DATASETS, codecs.DATASET_SLUG]', 'DATASETS = [*base.DATASETS, codecs.DATASET_SLUG, r7.R6_DATASET_SLUG]')
    files[WRAPPER] = files[WRAPPER].replace(b'[*MEMBERS, "R224"]', b'[*MEMBERS, "R224", "S_R6"]')
    edit(WRAPPER, '    receipt["fallbacks"]["r224_prediction"]',
         '    receipt["fallbacks"]["s_r6_prediction"] = len(b["failures"]["S_R6"])\n'
         '    receipt["r6_identity"] = b["r6"]["r7_identity"]\n'
         '    receipt["execution_passes"].append(dict(serving="exp169_r7_port sequential R6 flip", arms=["S_R6"]))\n'
         '    receipt["fallbacks"]["r224_prediction"]')
    edit(WRAPPER, '("S_R7", pr7_dir / "pr7-run/s_r7/controller/submission_receipt.json")):',
         '("S_R7", pr7_dir / "pr7-run/s_r7/controller/submission_receipt.json"),\n'
         '                        ("S_R6", pr7_dir / "pr7-run/s_r6/controller/submission_receipt.json")):')
    edit(WRAPPER, '        completed.append("R224")\n',
         '        completed.append("R224")\n'
         '        r6_identity = read_json(pr7_dir / "pr7-run/r6-identity.json")\n'
         '        r6_ids, _, r6_values, b["r6"] = authenticated_pr7(pr7_dir, member="S_R6",\n'
         '            bundle_id=r7.R6_BUNDLE_ID, dataset_slug=r6_identity["dataset_slug"], manifest_pin=r7.R6_MANIFEST_SHA256)\n'
         '        need(set(r6_ids) == set(ids), "R6 cohort mismatch")\n'
         '        r6_values = r6_values[[r6_ids.index(uid) for uid in ids]]\n'
         '        b["failures"]["S_R6"] = b["r6"]["failures"]["S_R6"]\n'
         '        completed.append("S_R6")\n')
    edit(WRAPPER, '{**a["failures"], **b["failures"]})', '{**a["failures"], **b["failures"]}, r6=r6_values)')
    edit(WRAPPER, 'r224.reference(r, r224_values, ours, exp143, exp155, q)', 'r224.reference(r, r224_values, ours, exp143, exp155, q, r6=r6_values)')
    edit(WRAPPER, 'f224, np.asarray(failed)))', 'f224, np.asarray(failed), r6_values,\n                np.asarray([uid in b["failures"]["S_R6"] for uid in ids])))')
    edit(WRAPPER, '.5*r(S_R7); own-success ranks;', '.5*(.5*r(S_R7)+.5*r(S_R6)); own-success ranks;')
    edit(WRAPPER, '    c85.phase(output, "S_R7", infer_s)\n',
         '    c85.phase(output, "S_R7", infer_s)\n'
         '    # Synchronous phase completion includes worker teardown before the next bundle.\n'
         '    def infer_s6():\n'
         '        r6_identity = read_json(work / "r6-identity.json")\n'
         '        bundle, slug = base.r6_mount()\n'
         '        need(slug == r6_identity["dataset_slug"], "R6 mount changed since preflight")\n'
         '        r7.infer_studies([comp / "test_series" / uid for uid in ids], bundle_dir=bundle, code_dir=code,\n'
         '            anatomy_weights=anatomy, output_dir=work / "s_r6", workers=2, amp="auto", device="cuda",\n'
         '            max_fallback_fraction=1.0, source_dir=runtime / "exp169_r7_src",\n'
         '            expected_manifest_sha256=r7.R6_MANIFEST_SHA256, bundle_id=r7.R6_BUNDLE_ID, dataset_slug=slug)\n'
         '    c85.phase(output, "S_R6", infer_s6)\n')
    return files


def payload():
    nb,original,old_blob,assignments = parent()
    files = allowed_files(original)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buf,'w') as dest:
        for info in source.infolist(): dest.writestr(info,files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before,after in [('BLOB = '+repr(assignments['BLOB']),'BLOB = '+repr(base64.b64encode(blob).decode())),
                         ('PINS = '+repr(assignments['PINS']),'PINS = '+repr({k:old.digest(v) for k,v in files.items()})),
                         (repr(old.digest(old_blob)),repr(old.digest(blob)))]: code=old.replace(code,before,after)
    nb['cells'][-1]['source']=code.decode().splitlines(True)
    meta=old.read(BASE/'kernel/kernel-metadata.json')
    meta.update(id=SLUG,title=SLUG.split('/')[1])
    meta['dataset_sources'].append(R6_PRIVATE)
    return nb,meta,files,blob


def artifacts():
    nb,meta,files,blob=payload()
    pnb,original,_,_=parent()
    changed=sorted(k for k in files if files[k]!=original[k])
    old.need(set(changed)=={PORT,CHAIN,WRAPPER,MATH} and set(files)==set(original),'unexpected scope')
    for k,v in files.items():
        if k.endswith('.py'): compile(v,k,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    manifest=dict(changed=changed,unchanged=[k for k in files if k not in changed],added=[],removed=[],
        files={k:dict(before=old.digest(original[k]),after=old.digest(v)) for k,v in files.items()})
    full='ALL 31 EMBEDDED FILES: 27 IDENTICAL, 4 REGISTERED CHANGES, 0 ADDED/REMOVED\n'
    pairs=[(k,original[k],files[k]) for k in sorted(files)]
    pairs += [('notebook.json',old.encoded(pnb),old.encoded(nb)),('kernel-metadata.json',old.encoded(old.read(BASE/'kernel/kernel-metadata.json')),old.encoded(meta))]
    for name,a,b in pairs:
        full+=f'{name}: {old.digest(a)} -> {old.digest(b)} '+('IDENTICAL' if a==b else 'REGISTERED_CHANGE')+'\n'
        full+=''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='parent/'+name,tofile='child/'+name))
    return nb,meta,files,blob,manifest,full.encode()


def expected_artifacts():
    nb,meta,files,blob,manifest,full=artifacts()
    data={'kernel/release.ipynb':old.encoded(nb),'kernel/kernel-metadata.json':old.encoded(meta),
          'registration.json':gate.encoded(registration()),'new-members.json':gate.encoded(manifest),
          'full-diff-vs-v1.1.24-r224flipq.txt':full}
    rec=dict(status='PREPARED_CPU_ONLY',version=VERSION,builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files={k:old.digest(v) for k,v in data.items() if k.startswith('kernel/')},
        archive_sha256=old.digest(blob),embedded={k:old.digest(v) for k,v in files.items()},
        unchanged=27,changed=manifest['changed'],added=[],removed=[],qualified_for_submission=False)
    data['build-receipt.json']=gate.encoded(rec)
    return data,files,rec


def prepare():
    data,files,_=expected_artifacts()
    for k,v in data.items(): old.write(OUT/k,v)
    for k,v in files.items(): old.write(SCRATCH/'candidate-runtime'/k,v)
    old.write(OUT/'CHANGELOG.md',b'# v1.1.31-r224r76fq / builder 1.0.0\nH-B-175: add R6 flip sequentially beside R7 flip; independent ranks and survivor slot average. 27/31 embedded members byte-identical. No original/frozen code edited; no gate copy. CPU-only; independent review and HB175_N32 plus real-output replay required before admission.\n')
    return validate()


def validate():
    data,_,rec=expected_artifacts()
    for k,v in data.items(): old.need((OUT/k).read_bytes()==v,'stage drift: '+k)
    return dict(status='PASS_CPU_VALIDATE',kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),
                unchanged=27,changed=4,qualified_for_submission=False)


def admit():
    val=validate()
    smoke=old.read(SCRATCH/'smoke.json')
    old.need(smoke['status']=='PASS' and smoke['kernel_sha256']==val['kernel_sha256'],'CPU smoke missing/stale')
    return dict(val,status='HOLD_INDEPENDENT_REVIEW_HB175_N32_AND_REAL_OUTPUT_REPLAY',qualified_for_submission=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','prepare','validate','cpu-smoke','admit'])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    if args.command in ('prepare','admit'): old.need(args.dry_run,'CPU dry-run required')
    if args.command=='register':
        atomic_json(SCRATCH/'registration-before-results.json',registration()); result=registration()
    elif args.command=='cpu-smoke':
        from tools import c85r7_r76_cpu_smoke as smoke
        result=smoke.run()
    else: result={'prepare':prepare,'validate':validate,'admit':admit}[args.command]()
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
