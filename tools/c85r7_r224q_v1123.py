"""H-B-164 builder 1.0.0: one strict C85-R7 v1.1.0 + R224 + reviewed FAST candidate.

Changelog: 1.0.0 adds the registered R224 half; no gate binding or GPU execution.
Frozen originals are read-only; every existing runtime byte except the integration
wrapper is retained. The complete exact patch is the allowlist, not a line regex.
"""
from __future__ import annotations
import argparse
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
from tools import c85r7_flipall_v1115 as old
from tools import c85r7_own25q3_v1118 as fast
from tools import c85r7_flipallq_v1122 as fastq
from tools import strict_variant_gate as gate
from tools import exp169_r224 as r224

VERSION = '1.1.23-r224q'
OUT = old.BASE.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb164-r224/hb164'
SOURCE_DIR = SCRATCH / 'nartaa-original'
SOURCE_NOTEBOOK = SOURCE_DIR / 'rsna-knee-0945-efficient-224crop.ipynb'
NOTEBOOK_SHA = 'b80d3b72a1c34f2b39b3ea0d2952f039b8d27c3feb5c4db0a7d0ab62ae75354e'
SCORED_SHA = 'e85d994fa93caf1bbc2df04f396649207a69d2463924c41fa3b05ba8801cd38c'
CHECK_CELL_SHA = 'c5f79d1c8240a264014b1f88c5ea9cbc7d8f9fbe3b53c2ab6a5f75e470e5d072'
SLUG = 'taeyangg4/knee-lb-r224q-v1'
WRAPPER = 'exp169_release_c85r7codec.py'
HELPER = 'exp169_r224.py'
FORMULA = r224.FORMULA


def diff(a, b, before, after):
    return ''.join(difflib.unified_diff(a.decode().splitlines(True), b.decode().splitlines(True), fromfile=before, tofile=after))


def source():
    old.pin(SOURCE_NOTEBOOK, NOTEBOOK_SHA)
    nb = old.read(SOURCE_NOTEBOOK)
    check, scored = [''.join(nb['cells'][i]['source']).encode() for i in (3, 4)]
    old.need(old.digest(check) == CHECK_CELL_SHA and old.digest(scored) == SCORED_SHA, 'original source cell hash')
    raw = ('    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)   # [blend] raw probabilities; a failed study stays 0.5\n'
           '    _raw.insert(0, "StudyInstanceUID", test_ids)\n'
           '    _raw.to_csv("/kaggle/working/raptor224_raw.csv", index=False)\n')
    anchor = '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n'
    modified = old.replace(scored, anchor, anchor + raw)
    modified = old.replace(modified, '    out = "/kaggle/working/submission.csv"',
        '    out = "/kaggle/working/raptor224_sub.csv"   # [blend] Raptor-224 stage output; the blend cell writes submission.csv')
    helios = (SCRATCH / 'heliosli-cell-4.txt').read_text(encoding='utf-8').encode()
    old.need(helios.endswith(modified), 'two changes differ from heliosli scored segment')
    combined = check + b'\n' + modified
    compile(combined, r224.SOURCE, 'exec')
    return combined, dict(notebook_url='https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop',
        notebook_sha256=NOTEBOOK_SHA, original_scored_source_sha256=SCORED_SHA,
        checkpoint_cell_sha256=CHECK_CELL_SHA, concatenation='original cell 3 + one LF + patched cell 4',
        edits_unified_diff=diff(scored, modified, 'nartaa/cell4.py', 'raptor224_run.py/scored-cell4'),
        heliosli_scored_segment_identical=True,
        permission='D-B-174: explicit non-commercial adaptation permission; source retained')


def patched(original):
    files = dict(original)
    files[r224.SOURCE] = source()[0]
    files[HELPER] = (ROOT / 'tools' / HELPER).read_bytes()
    s = files[WRAPPER]
    changes = [
        ('from tools import exp169_release_c85r7 as base, hb126_codecs as codecs',
         'from tools import exp169_release_c85r7 as base, hb126_codecs as codecs\n    from tools import exp169_r224 as r224'),
        ('    import hb126_codecs as codecs', '    import hb126_codecs as codecs\n    import exp169_r224 as r224'),
        ('VERSION = "1.1.0"', 'VERSION = "' + VERSION + '"'),
        ('RELEASE = "C85R7+codecs"', 'RELEASE = "C85R7+codecs R224"'),
        ('FORMULA = base.FORMULA', 'FORMULA = r224.FORMULA'),
        ('intended_members=MEMBERS,', 'intended_members=[*MEMBERS, "R224"],'),
        ('KERNEL_ID = "taeyangg4/knee-exp169-c85r7codec-v1"', 'KERNEL_ID = "' + SLUG + '"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0"',
         'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + VERSION + '"'),
        ('receipt.update(version=VERSION, release=RELEASE, codecs=codec)',
         'receipt.update(version=VERSION, release=RELEASE, codecs=codec, formula=FORMULA,\n'
         '                   exposure="EXPOSED", clean_oof=False, r224=b["r224"],\n'
         '                   intended_members=[*MEMBERS, "R224"], executed_members=[*MEMBERS, "R224"])\n'
         '    receipt["fallbacks"]["r224_prediction"] = len(b["failures"]["R224"])\n'
         '    receipt["execution_passes"].append(dict(serving=r224.SOURCE, arms=["R224"]))'),
        ('        values = blend(ids, r, ours, exp143, exp155, q, {**a["failures"], **b["failures"]})\n'
         '        expected = reference(r, ours, exp143, exp155, q)',
         '        r224_values, r224_failures, b["r224"] = r224.read(pr7_dir, ids)\n'
         '        b["failures"]["R224"] = r224_failures\n'
         '        completed.append("R224")\n'
         '        values = r224.blend(ids, r, r224_values, ours, exp143, exp155, q, {**a["failures"], **b["failures"]})\n'
         '        expected = r224.reference(r, r224_values, ours, exp143, exp155, q)'),
        ('            branch["payload"] = p.csv_bytes(ids, p_branch(r, ours, failed))',
         '            rf = np.asarray([uid in b["failures"]["R"] for uid in ids]) | np.all(r == .5, axis=1)\n'
         '            f224 = np.asarray([uid in r224_failures for uid in ids])\n'
         '            branch["payload"] = p.csv_bytes(ids, r224.p_branch(r, r224_values, ours, rf, f224, np.asarray(failed)))'),
        ('formula="public cell 7: .5 r(R)+.5 r(S_R7), each over own non-failed rows; float32",',
         'formula="public cell 8: .5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",'),
        ('    def infer_s():',
         '    def infer_r224():\n'
         '        working = Path("/kaggle/working")\n'
         '        need(not any((working / name).exists() for name in (r224.RAW, r224.SUB)), "stale R224 outputs")\n'
         '        v5q.run_process_group([sys.executable, "-u", str(runtime / r224.SOURCE)],\n'
         '            label="C85R7 R224", env=codecs.child_env(env, codec, branch="R224", telemetry_dir=tele))\n'
         '        r224.record(work, working)\n'
         '    c85.phase(output, "R224", infer_r224)\n'
         '    def infer_s():'),
        ('    base.runtime_pins(runtime)\n',
         '    base.runtime_pins(runtime)\n'
         '    p.check_pin(runtime / r224.SOURCE, ' + repr(old.digest(files[r224.SOURCE])) + ')\n'
         '    p.check_pin(runtime / "exp169_r224.py", ' + repr(old.digest(files[HELPER])) + ')\n'),
    ]
    for a, b in changes:
        s = old.replace(s, a, b)
    files[WRAPPER] = s
    compile(s, WRAPPER, 'exec')
    return files


def payload():
    nb, original = old.parent()
    files = patched(original)
    _, old_blob, assignments = old.unpack(nb)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as parent, zipfile.ZipFile(buf, 'w') as archive:
        for info in parent.infolist():
            archive.writestr(info, files[info.filename])
        for name in sorted(set(files) - set(original)):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 15, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(info, files[name])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    code = old.replace(code, 'BLOB = ' + repr(assignments['BLOB']), 'BLOB = ' + repr(base64.b64encode(blob).decode()))
    code = old.replace(code, 'PINS = ' + repr(assignments['PINS']), 'PINS = ' + repr({k:old.digest(v) for k,v in files.items()}))
    code = old.replace(code, repr(old.digest(old_blob)), repr(old.digest(blob)))
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    # Authenticate the exact reviewed FAST source independently of the variant archive.
    ref = gate.kernel(fast.OUT, gate.Inputs(ROOT))
    old.need(ref[2]['kernel_files'] == gate.HISTORICAL[fast.OUT.name]['kernel_files'], 'FAST identity drift')
    old.need(gate.sha(gate.Inputs(ROOT).read(gate.FAST_REFERENCE['review'])) == gate.FAST_REFERENCE['review_sha256'], 'FAST review drift')
    nb['cells'].insert(0, copy.deepcopy(old.read(fast.OUT / 'kernel/release.ipynb')['cells'][0]))
    nb['cells'][-1]['source'] = ('if not FAST:\n' + ''.join('    ' + line for line in nb['cells'][-1]['source'])).splitlines(True)
    meta = old.read(old.BASE / 'kernel/kernel-metadata.json')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta, files, blob


def registration():
    return dict(version=VERSION, item='H-B-164', purpose='candidate for the final pair', kernel_id=SLUG,
        parent=str(old.BASE.relative_to(ROOT)).replace('\\','/'), parent_receipt_sha256=old.BASE_SHA,
        formula=FORMULA, flip_tta=False, exposure='EXPOSED', clean_oof=False,
        public_reading=dict(alive_min=.957, dead_max=.955, hold=.956, rule='ALIVE >=0.957; DEAD <=0.955; otherwise HOLD'),
        timing=dict(tag='HB164_R224_N32', baseline_hours=5.38, projection='5.38 + measured R224 hidden share hours',
                    max_projected_hours=8.5, actual_kaggle_limit_hours=9, measured=False),
        registered_changes=['R224 original checkpoint cell and scored source with exactly two heliosli [blend] edits',
            'R half .65 own-success rank R384 + .35 own-success rank R224; survivor ranks; R224 failures >3% abort',
            'Exact reviewed FAST cell and if-not-FAST guard; flip OFF', 'id/title only; dataset_sources unchanged'],
        gate_binding='DEFERRED by B part 2; independent added-member contract required', qualified_for_submission=False)


def reading(score):
    old.need(isinstance(score, (int, float)) and 0 <= score <= 1, 'invalid public score')
    return 'ALIVE' if score >= .957 else 'DEAD' if score <= .955 else 'HOLD'


def artifacts():
    nb, meta, files, blob = payload()
    parent_nb, original = old.parent()
    _, origin = source()
    changed = [k for k in original if original[k] != files[k]]
    old.need(changed == [WRAPPER] and len(files) == 31, 'scope: one wrapper plus two additions')
    manifest = dict(version=VERSION, added=[
        dict(name=r224.SOURCE, sha256=old.digest(files[r224.SOURCE]), origin=origin),
        dict(name=HELPER, sha256=old.digest(files[HELPER]), origin=dict(kind='project-authored integration and arithmetic',
             source='tools/exp169_r224.py', derived_semantics='heliosli notebook cell 8 and frozen C85 outer survivor policy',
             original_scored_source_sha256=SCORED_SHA, nartaa_two_edits_unified_diff=origin['edits_unified_diff']))],
        changed=[dict(name=WRAPPER, before_sha256=old.digest(original[WRAPPER]), after_sha256=old.digest(files[WRAPPER]),
            registered_reason='Add isolated R224 subprocess, authenticate its outputs, replace only R half, metadata and new hash pins')],
        unchanged=[dict(name=k, sha256=old.digest(v)) for k,v in sorted(original.items()) if k not in changed])
    inventory = ''.join(f'{k}: {old.digest(original[k]) if k in original else "ADDED"} -> {old.digest(v)} '
        + ('IDENTICAL' if original.get(k) == v else 'REGISTERED_CHANGE' if k in original else 'REGISTERED_ADDITION') + '\n'
        for k,v in sorted(files.items()))
    full = inventory + '\nFULL EMBEDDED DIFF\n' + ''.join(diff(original.get(k,b''), files[k], 'v1.1.0/'+k, VERSION+'/'+k)
           for k in sorted(files))
    full += '\nFULL NOTEBOOK AND METADATA DIFF (archive encoded as shipped)\n'
    full += diff(old.encoded(parent_nb), old.encoded(nb), 'v1.1.0/release.ipynb', VERSION+'/release.ipynb')
    full += diff(old.encoded(old.read(old.BASE/'kernel/kernel-metadata.json')), old.encoded(meta), 'v1.1.0/metadata', VERSION+'/metadata')
    return nb, meta, files, blob, manifest, full.encode()


def prepare():
    old.pin(SOURCE_DIR / r224.CHECKPOINT, r224.CHECKPOINT_SHA)
    nb, meta, files, blob, manifest, full = artifacts()
    for name in ('release.ipynb', 'kernel-metadata.json'):
        old.write(ROOT / '.local/backups/hb164-r224/v110' / name, (old.BASE / 'kernel' / name).read_bytes())
    for name,data in [('kernel/release.ipynb', old.encoded(nb)), ('kernel/kernel-metadata.json', old.encoded(meta)),
                      ('registration.json', gate.encoded(registration())), ('new-members.json', gate.encoded(manifest)),
                      ('full-diff-vs-v1.1.0.txt', full)]:
        old.write(OUT / name, data)
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()}, unchanged=28, changed=[WRAPPER], added=[HELPER,r224.SOURCE],
        checkpoint=dict(dataset='nartaa/rsna-knee-publication-swa-weights-20261007', filename=r224.CHECKPOINT,
            bytes=(SOURCE_DIR/r224.CHECKPOINT).stat().st_size, sha256=r224.CHECKPOINT_SHA, verified_actual_file=True),
        qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    old.write(OUT / 'build-receipt.json', gate.encoded(rec))
    old.write(OUT / 'CHANGELOG.md', b'# 1.1.23-r224q / builder 1.0.0\nR224 addition to hidden-proven v1.1.0; exact reviewed FAST cell. Existing launch and pooling bytes retained. No gate copy bound (B part 2). No frozen tool/stage edits. Parent snapshots: .local/backups/hb164-r224/v110/. CPU-only preparation; independent review and Linux/CUDA qualification remain required.\n')
    for name,data in files.items():
        old.write(SCRATCH / 'candidate-runtime' / name, data)
    return validate()


def validate(stage=OUT):
    nb, meta, files, blob, manifest, full = artifacts()
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
                      ('registration.json',gate.encoded(registration())),('new-members.json',gate.encoded(manifest)),
                      ('full-diff-vs-v1.1.0.txt',full)]:
        old.need((stage/name).read_bytes() == data, 'stage drift: '+name)
    rec = old.read(stage/'build-receipt.json')
    old.need(rec['builder_sha256'] == old.digest(Path(__file__).read_bytes()), 'builder drift')
    old.need(rec['embedded'] == {k:old.digest(v) for k,v in files.items()}, 'embedded drift')
    old.need(rec['kernel_files'] == gate.kernel(stage,gate.Inputs(ROOT))[2]['kernel_files'], 'kernel drift')
    old.need(old.unpack(fastq.restored(nb))[0] == files, 'archive inspection')
    return dict(status='PASS_CPU_VALIDATE', unchanged=28, changed=1, added=2,
        kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])), qualified_for_submission=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare','validate','register','admit'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.command in ('prepare','admit'):
        old.need(args.dry_run, 'CPU dry-run required')
    result = prepare() if args.command == 'prepare' else validate()
    if args.command == 'admit':
        result.update(status='HOLD_INDEPENDENT_REVIEW_AND_GPU_TIMING', qualified_for_submission=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
