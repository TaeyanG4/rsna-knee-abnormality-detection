"""HB151 kill-test builder 1.0.0. CPU only; never invokes Kaggle or CUDA.

Changelog: checkpoint source/pins plus exact own25q3 cell zero and guard.
Strict scope deliberately rejects checkpoints the frozen parent's own_meta rejects.
"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / '.local/exp152-own0949-v1/stage'
DONOR = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.18-own25q3'
OUT = ROOT / 'artifacts/experiments/HB151_KILLTEST/v1.0.0'
SCRATCH = ROOT / '.local/scratch/astra-b4/HB151_KILLTEST'
EXPORT = ROOT / '.local/scratch/astra-b4/H-B-151/v3.0.2/export/exp143-v2-e14-e11-e12-bn.pt'
NAME = EXPORT.name
ARCHIVE = 'exp152-assets.zip.bin'
SLUG = 'taeyangg4/knee-hb151-pl-coat-v1'
DATASET = 'taeyangg4/knee-hb151-pl-coat-ckpt-v1'
PLACEHOLDER = b'HB151_KILLTEST_PLACEHOLDER_NOT_A_TRAINED_CHECKPOINT\n'
DECISION = dict(alive_at_least=0.950, dead_at_most=0.947, otherwise_seed=43,
                baseline_public=0.946, baseline_ref='57005140', standalone=True)
PINS = {
    'own/release.ipynb': '1045c22cd6de8e28dfeb60fc47e62b0356d069487d520c63e67421586aadb08a',
    'own/kernel-metadata.json': 'cff7e3ea949caca229869d379c26b237696e7fe2c84175af38b47114b91daa58',
    'build-receipt.json': '672ebedf84ebd8fb2f3e1dcf016995f5f44a20fb5184017dfe94b5d30ea0c198',
    'assets/release-manifest.json': 'd85946bd08c6087ef170e705bf695260c5f398e57479bd30b48ba65e0838b295',
    'assets/exp152_own0949_release.py': 'ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2',
    'assets/public-serving.py': '25ef6ab9d2e942ac291433a498246d0507481c2aefa5630cdd601d0a36852cf2',
}
DONOR_PIN = '98668e17d4419249ccce63641b17b90e5ec9bb3612f72ffb875ef3b33d3f5855'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def retry(fn):
    for attempt in range(40):
        try:
            return fn()
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32)) or attempt == 39:
                raise
            time.sleep(0.05)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    def publish():
        with temporary.open('wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    retry(publish)
    retry(lambda: os.replace(temporary, path))


def parent():
    for name, expected in PINS.items():
        need(sha(PARENT / name) == expected, 'parent identity: ' + name)
    ledger = ROOT / 'artifacts/experiments/SUBMISSION_SCREEN/submit_ledger.jsonl'
    rows = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines() if line.strip()]
    need(any(str(r.get('ref')) == '57005140' and r.get('kernel') == 'taeyangg4/knee-exp152-own0949-v2'
             and str(r.get('version')) == '1' and r.get('outcome') == 'SUBMITTED' for r in rows), 'ledger identity')
    need(sha(DONOR / 'kernel/release.ipynb') == DONOR_PIN, 'donor notebook identity')
    return read(PARENT / 'own/release.ipynb'), read(PARENT / 'own/kernel-metadata.json')


def source(cell):
    return ''.join(cell['source'])


def manifest(checkpoint_sha):
    m = read(PARENT / 'assets/release-manifest.json')
    m['checkpoints'][NAME]['sha256'] = checkpoint_sha
    m['asset_sha256'][NAME] = checkpoint_sha
    return m


def payload(checkpoint_sha, archive_sha):
    nb, meta = parent()
    msha = digest(encoded(manifest(checkpoint_sha)))
    original = source(nb['cells'][4])
    changes = {
        'knee-exp152-own0949-v2-assets': DATASET.split('/')[1],
        read(PARENT / 'build-receipt.json')['archive_sha256']: archive_sha,
        PINS['assets/release-manifest.json']: msha,
    }
    for before, after in changes.items():
        need(before in original, 'missing replacement ' + before)
        original = original.replace(before, after)
    # Same guard and exact indentation operation as the donor builder.
    nb['cells'][4]['source'] = 'if not FAST:\n' + ''.join('    ' + line for line in original.splitlines(True))
    nb['cells'].insert(0, copy.deepcopy(read(DONOR / 'kernel/release.ipynb')['cells'][0]))
    meta.update(id=SLUG, title=SLUG.split('/')[1], dataset_sources=[DATASET])
    return nb, meta


def dataset_metadata():
    return dict(title=DATASET.split('/')[1], id=DATASET, licenses=[dict(name='other')],
                description='Private HB151 EXPOSED / clean_oof=false checkpoint and byte-identical EXP143v2 serving code. '
                'The checkpoint retains its exact filename inside exp152-assets.zip.bin. No training data. Do not use --public.')


def full_diff(nb, meta, files):
    before, before_meta = parent()
    pairs = [('notebook-top-level', encoded({k:v for k,v in before.items() if k != 'cells'}),
              encoded({k:v for k,v in nb.items() if k != 'cells'})),
             ('cell-0-added-exact-donor', b'', encoded(nb['cells'][0]))]
    pairs += [(f'parent-cell-{i}-to-child-{i+1}', encoded(a), encoded(b))
              for i, (a,b) in enumerate(zip(before['cells'], nb['cells'][1:]))]
    pairs += [('kernel-metadata.json', encoded(before_meta), encoded(meta))]
    for name in sorted(files):
        if name == NAME:
            pairs.append((name + '-binary-sha256', (read(PARENT/'assets/release-manifest.json')['asset_sha256'][NAME]+'\n').encode(),
                          (files[name]+'\n').encode()))
        else:
            pairs.append(('archive/' + name, (PARENT/'assets'/name).read_bytes(),
                          encoded(manifest(files[NAME])) if name == 'release-manifest.json' else (PARENT/'assets'/name).read_bytes()))
    result = ['ALLOWLIST: checkpoint bytes; two manifest checkpoint hashes; dataset mount/id; archive/manifest pins; kernel id/title; exact donor cell0/guard.\n',
              'No serving source, process/import policy, preprocessing, inference, TTA or output change.\n',
              'Guard placement: parent has one code cell (index4); prepend donor cell0 and wrap that cell (now index5).\n']
    for name,a,b in pairs:
        result.append(f'\n{name}: {"IDENTICAL" if a == b else "ALLOWED_CHANGE"} {digest(a)} -> {digest(b)}\n')
        result.extend(difflib.unified_diff(a.decode().splitlines(True), b.decode().splitlines(True), fromfile='parent/'+name, tofile='HB151/'+name))
    return ''.join(result).encode()


def archive_files(path):
    expected = {NAME, 'release-manifest.json', 'public-serving.py', 'exp152_own0949_release.py'}
    hashes, small = {}, {}
    with zipfile.ZipFile(path) as z:
        need(len(z.namelist()) == 4 and set(z.namelist()) == expected, 'archive member scope')
        for name in sorted(expected):
            with z.open(name) as f:
                hashes[name] = hashlib.file_digest(f, 'sha256').hexdigest()
            if name != NAME:
                small[name] = z.read(name)
    return hashes, small


def authenticate_parent_archive():
    archive = PARENT/'dataset'/ARCHIVE
    need(sha(archive) == read(PARENT/'build-receipt.json')['archive_sha256'], 'parent archive identity')
    hashes, small = archive_files(archive)
    expected = read(PARENT/'assets/release-manifest.json')['asset_sha256']
    need(all(hashes[n] == s for n,s in expected.items()) and
         hashes['release-manifest.json'] == PINS['assets/release-manifest.json'], 'parent archive members differ')
    need(all(data == (PARENT/'assets'/name).read_bytes() for name,data in small.items()), 'parent extracted files differ')
    return dict(status='PASS', members=hashes)


def scope(nb, meta, hashes, small, archive_sha):
    expected, expected_meta = payload(hashes[NAME], archive_sha)
    need(nb == expected and meta == expected_meta, 'forbidden notebook/metadata difference')
    need(small['release-manifest.json'] == encoded(manifest(hashes[NAME])), 'forbidden manifest difference')
    for name in ['public-serving.py', 'exp152_own0949_release.py']:
        need(small[name] == (PARENT/'assets'/name).read_bytes(), 'forbidden runtime difference: '+name)
    return True


def build(checkpoint=None, *, fixture=False):
    parent()
    need(not (OUT/'RELEASED.json').exists(), 'frozen release; create a new version')
    if checkpoint is None:
        need(not (OUT/'build-receipt.json').exists() or read(OUT/'build-receipt.json')['placeholder'], 'refuse to overwrite bound checkpoint')
        checkpoint = SCRATCH/'placeholder'/NAME
        write(checkpoint, PLACEHOLDER)
    checkpoint = Path(checkpoint)
    csha = sha(checkpoint)
    placeholder = csha == digest(PLACEHOLDER)
    if not placeholder:
        compatibility(checkpoint)  # MUST happen before stage mutation.
    outzip = OUT/'dataset'/ARCHIVE
    outzip.parent.mkdir(parents=True, exist_ok=True)
    temporary = outzip.with_suffix('.partial')
    mbytes = encoded(manifest(csha))
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as z:
        for name in sorted([NAME, 'release-manifest.json', 'public-serving.py', 'exp152_own0949_release.py']):
            info = zipfile.ZipInfo(name, date_time=(2026,10,15,0,0,0))
            with z.open(info, 'w', force_zip64=True) as dest:
                if name == NAME:
                    with checkpoint.open('rb') as src:
                        shutil.copyfileobj(src, dest, 4<<20)
                else:
                    dest.write(mbytes if name == 'release-manifest.json' else (PARENT/'assets'/name).read_bytes())
    retry(lambda: os.replace(temporary, outzip))
    asha = sha(outzip)
    nb, meta = payload(csha, asha)
    write(OUT/'kernel/release.ipynb', encoded(nb))
    write(OUT/'kernel/kernel-metadata.json', encoded(meta))
    write(OUT/'dataset/dataset-metadata.json', encoded(dataset_metadata()))
    write(OUT/'dataset'/Path(NAME).with_suffix('.sha256'), (csha+'  '+NAME+'\n').encode())
    hashes, small = archive_files(outzip)
    scope(nb, meta, hashes, small, asha)
    write(OUT/'full-diff-vs-parent.txt', full_diff(nb, meta, hashes))
    rec = dict(version='1.0.0', parent_ref='57005140', parent_pins=PINS, donor_notebook_sha256=DONOR_PIN,
               placeholder=placeholder, fixture=fixture, checkpoint_sha256=csha, archive_sha256=asha,
               notebook_sha256=sha(OUT/'kernel/release.ipynb'), metadata_sha256=sha(OUT/'kernel/kernel-metadata.json'),
               dataset_metadata_sha256=sha(OUT/'dataset/dataset-metadata.json'), files=hashes,
               builder_sha256=sha(Path(__file__)), exposure='EXPOSED', clean_oof=False, decision=DECISION,
               qualified_for_submission=False, status='PLACEHOLDER_CPU_ONLY' if placeholder else 'BOUND_NOT_REVIEWED')
    write(OUT/'build-receipt.json', encoded(rec))
    return validate()


def validate():
    nb, meta = read(OUT/'kernel/release.ipynb'), read(OUT/'kernel/kernel-metadata.json')
    r = read(OUT/'build-receipt.json')
    need(r['builder_sha256'] == sha(Path(__file__)) and r['parent_pins'] == PINS and
         r['donor_notebook_sha256'] == DONOR_PIN and r['decision'] == DECISION and
         r['exposure'] == 'EXPOSED' and r['clean_oof'] is False and r['qualified_for_submission'] is False, 'registration drift')
    for rel,key in [('kernel/release.ipynb','notebook_sha256'), ('kernel/kernel-metadata.json','metadata_sha256'),
                    ('dataset/dataset-metadata.json','dataset_metadata_sha256'), ('dataset/'+ARCHIVE,'archive_sha256')]:
        need(sha(OUT/rel) == r[key], 'artifact hash: '+rel)
    hashes, small = archive_files(OUT/'dataset'/ARCHIVE)
    need(hashes == r['files'] and hashes[NAME] == r['checkpoint_sha256'], 'checkpoint pin')
    need(r['placeholder'] == (hashes[NAME] == digest(PLACEHOLDER)), 'placeholder identity')
    scope(nb, meta, hashes, small, r['archive_sha256'])
    need(read(OUT/'dataset/dataset-metadata.json') == dataset_metadata(), 'dataset identity')
    need((OUT/'dataset'/Path(NAME).with_suffix('.sha256')).read_text().split() == [hashes[NAME],NAME], 'sidecar pin')
    need((OUT/'full-diff-vs-parent.txt').read_bytes() == full_diff(nb, meta, hashes), 'mechanical diff drift')
    return dict(status='PASS_CPU_SCOPE', placeholder=r['placeholder'], runtime_files_identical=2,
                archive_members_checked=4, qualified_for_submission=False)


def runtime():
    spec = importlib.util.spec_from_file_location('_hb151_frozen_parent', PARENT/'assets/exp152_own0949_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compatibility(checkpoint):
    """Use the EXACT frozen checkpoint acceptance path on CPU. No provenance laundering."""
    parent()
    m = runtime()
    ck = m.safe_load(Path(checkpoint))
    m.own_meta(ck, read(PARENT/'assets/release-manifest.json')['protocol_sha256'])
    need(m.state_schema(ck['model']) == read(PARENT/'assets/release-manifest.json')['own_state_schema'], 'checkpoint state schema mismatch')
    return dict(status='PASS_FROZEN_LOADER_CPU')


def bind(checkpoint, pin_file):
    checkpoint, pin_file = Path(checkpoint), Path(pin_file)
    tokens = pin_file.read_text(encoding='utf-8').split()
    need(len(tokens) == 2 and re.fullmatch('[0-9a-f]{64}', tokens[0]) and tokens[1] == NAME, 'invalid export sidecar')
    need(checkpoint.name == NAME and sha(checkpoint) == tokens[0] and tokens[0] != digest(PLACEHOLDER), 'export pin mismatch/placeholder')
    from tools import hb151_build_v3 as fit
    export_meta = read(checkpoint.with_suffix('.json'))
    need(export_meta.get('sha256') == tokens[0] and export_meta.get('exposure') == 'EXPOSED' and
         export_meta.get('clean_oof') is False and export_meta.get('protocol_sha256') == sha(fit.PROTOCOL) and
         export_meta.get('serving_name') == NAME, 'HB151 export provenance missing/mismatched')
    summary = read(fit.RUN/'summary.json')  # B invokes bind only after the fit ends.
    need(summary.get('status') == 'COMPLETE_TRAINING_NO_EVALUATION' and
         summary.get('model_sha256') == tokens[0] and sha(fit.RUN/'model_best.pt') == tokens[0],
         'incomplete fit or canonical/export checkpoint mismatch')
    identity = summary.get('identity', {})
    need(identity.get('protocol_sha256') == sha(fit.PROTOCOL) and
         identity.get('run_dir') == read(fit.PROTOCOL)['run_dirs']['local'] and
         identity.get('seed') == 42 and summary.get('exposure') == 'EXPOSED' and
         summary.get('clean_oof') is False, 'completed fit identity mismatch')
    ck = runtime().safe_load(checkpoint)
    need(ck.get('exposure') == 'EXPOSED' and ck.get('clean_oof') is False and
         ck.get('provenance',{}).get('protocol_sha256') == sha(fit.PROTOCOL) and
         ck['provenance'].get('run_dir') == read(fit.PROTOCOL)['run_dirs']['local'], 'reject older/partial/non-HB151 checkpoint')
    del ck
    return build(checkpoint)


def outcome(score):
    need(isinstance(score, (float,int)) and 0 <= score <= 1, 'invalid public score')
    return 'alive' if score >= DECISION['alive_at_least'] else 'dead' if score <= DECISION['dead_at_most'] else 'seed43'


def smoke():
    from tools.hb151_killtest_cpu_fixtures import run_smoke, loader_smoke
    validate()
    report = run_smoke(SCRATCH/'fast-cases')
    report['parent_archive'] = authenticate_parent_archive()
    report['loader'] = loader_smoke(SCRATCH/'loader')
    from tools.c85r7_own25q2_v1114 import authenticate_visible
    report['visible_identity'] = authenticate_visible()
    report.update(build_receipt_sha256=sha(OUT/'build-receipt.json'),
                  notebook_sha256=sha(OUT/'kernel/release.ipynb'), fixture_source_sha256=sha(ROOT/'tools/hb151_killtest_cpu_fixtures.py'))
    write(OUT/'cpu-smoke.json', encoded(report))
    return dict(status=report['status'], cases=len(report['cases']), FAST_true=sum(c['FAST'] for c in report['cases']),
                FAST_false=sum(not c['FAST'] for c in report['cases']), loader=report['loader'], limitation=report['limitation'])


def admit(review=None):
    validate()
    s = read(OUT/'cpu-smoke.json')
    need(s['status'] == 'PASS' and s['build_receipt_sha256'] == sha(OUT/'build-receipt.json') and
         s['notebook_sha256'] == sha(OUT/'kernel/release.ipynb') and
         s['fixture_source_sha256'] == sha(ROOT/'tools/hb151_killtest_cpu_fixtures.py') and
         bool(s['cases']) and all(c['status'] == 'PASS' for c in s['cases']), 'stale/failed smoke')
    r = read(OUT/'build-receipt.json')
    if r['placeholder'] or r['fixture']:
        return dict(status='HOLD_PLACEHOLDER', qualified_for_submission=False)
    if review is None:
        return dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False)
    v = read(review)
    need(v.get('verdict') == 'PASS' and v.get('independent') is True and bool(v.get('reviewer')) and
         v.get('build_receipt_sha256') == sha(OUT/'build-receipt.json') and
         v.get('smoke_sha256') == sha(OUT/'cpu-smoke.json'), 'independent review missing/stale/failed')
    return dict(status='PASS_LOCAL_ADMISSION_ONLY', qualified_for_submission=False,
                note='B performs private dataset create and one visible push; visible FAST is not model inference qualification.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare','register','validate','cpu-smoke','bind','admit'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--checkpoint', type=Path, default=EXPORT)
    p.add_argument('--sha256-file', type=Path, default=EXPORT.with_suffix('.sha256'))
    p.add_argument('--review', type=Path)
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if a.command in ['prepare','admit']:
        need(a.dry_run, 'CPU dry-run flag required')
    result = {'prepare':lambda:build(), 'register':validate, 'validate':validate, 'cpu-smoke':smoke,
              'bind':lambda:bind(a.checkpoint,a.sha256_file), 'admit':lambda:admit(a.review)}[a.command]()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
