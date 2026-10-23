"""Build one prospective K3 notebook; static packaging only, never runs a model."""
from __future__ import annotations

import argparse
import ast
import base64
import difflib
import hashlib
import io
import json
from pathlib import Path
import zipfile

try:
    from tools import c85r7_rco_overlay_v1 as overlay
except ImportError:
    import c85r7_rco_overlay_v1 as overlay

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q/kernel'
STAGE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.45-r6w958rco'
WORK = ROOT / '.local/scratch/slot-b/k3-release/lean-build'
ASSETS = ROOT / '.local/scratch/slot-b/k2-release/fix1/assets-v1'
PARENT_NB_SHA = 'fcb2df01b47d5cc9b83c8a11e35ceca10d19ba5884e2b8fd40aee13867f3e8c9'
PARENT_META_SHA = 'f794876c887318a85887458dee54c0dffa1ca89edf5823db7c1242f17c15dbf9'
PARENT_BLOB_SHA = '92cec8b3f10a768a73ad221d5aa6ce2fefda41a591d0b39dcba6cbf10a5816bd'
need, sha, encoded, write_once = overlay.need, overlay.sha, overlay.encoded, overlay.write_once


def assignment(code, name):
    matches = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    need(len(matches) == 1, 'missing/duplicate assignment: ' + name)
    return ast.literal_eval(matches[0].value)


def parent_material():
    need(sha(PARENT / 'release.ipynb') == PARENT_NB_SHA and sha(PARENT / 'kernel-metadata.json') == PARENT_META_SHA, 'parent identity')
    notebook = json.loads((PARENT / 'release.ipynb').read_bytes())
    code = ''.join(notebook['cells'][2]['source'])
    raw = base64.b64decode(assignment(code, 'BLOB'))
    need(hashlib.sha256(raw).hexdigest() == PARENT_BLOB_SHA, 'parent archive')
    pins = assignment(code, 'PINS')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        need(len(archive.namelist()) == 31 == len(pins) and set(archive.namelist()) == set(pins), 'parent member inventory')
        for name, pin in pins.items():
            need(hashlib.sha256(archive.read(name)).hexdigest() == pin, 'parent member hash: ' + name)
    # Byte-identical cell; FAST=False is the sole prefix. Parent paths/imports unchanged.
    parent_script = ('FAST = False\n' + code).encode()
    compile(parent_script, '<K3 exact parent child>', 'exec')
    return notebook, raw, pins, parent_script


def package():
    parent, parent_blob, parent_pins, parent_script = parent_material()
    runtime = ROOT / 'tools/c85r7_rco_overlay_v1.py'
    files = {'parent-run.py': parent_script, 'c85r7_rco_overlay_v1.py': runtime.read_bytes()}
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 18, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(info, data)
    blob = out.getvalue()
    pins = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    code = '\n'.join([
        'import base64, hashlib, io, subprocess, sys, zipfile', 'from pathlib import Path',
        f'BLOB = {base64.b64encode(blob).decode()!r}', f'PINS = {pins!r}',
        'blob = base64.b64decode(BLOB)',
        f'assert hashlib.sha256(blob).hexdigest() == {hashlib.sha256(blob).hexdigest()!r}',
        "runtime = Path('/kaggle/working/k3-coordinator-runtime')", "assert not runtime.exists(), 'stale K3 runtime'",
        'with zipfile.ZipFile(io.BytesIO(blob)) as archive:',
        '    assert len(archive.namelist()) == len(PINS) and set(archive.namelist()) == set(PINS)',
        '    for name, pin in PINS.items():',
        "        assert '/' not in name and '\\\\' not in name and ':' not in name and '..' not in name",
        '        assert hashlib.sha256(archive.read(name)).hexdigest() == pin',
        '    runtime.mkdir()', '    archive.extractall(runtime)',
        "subprocess.run([sys.executable, '-B', '-u', str(runtime/'c85r7_rco_overlay_v1.py'), '--runtime', str(runtime)], check=True)", ''])
    compile(code, '<K3 notebook>', 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=parent['metadata'], cells=[
        dict(cell_type='markdown', metadata={}, source=['# K3 fixed RCO overlay\n', overlay.FORMULA + '\n',
            'Full parent inference followed by one exact K2 seed0/fold0 model; all-or-nothing overlay. No format-only path.\n',
            ''.join(parent['cells'][1]['source'])]),
        dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=code.splitlines(True))])
    meta = json.loads((PARENT / 'kernel-metadata.json').read_bytes())
    meta.update(id=overlay.KERNEL, title=overlay.KERNEL.split('/')[1])
    meta['dataset_sources'].append('taeyangg4/knee-l1-rcof0-v1-assets')
    inventory = dict(parent_notebook_sha256=PARENT_NB_SHA, parent_metadata_sha256=PARENT_META_SHA,
        parent_archive_sha256=PARENT_BLOB_SHA, parent_members=parent_pins, parent_members_byte_identical=31,
        parent_cell_byte_identical=True, parent_script_prefix='FAST = False\n', outer_members=pins,
        outer_archive_sha256=hashlib.sha256(blob).hexdigest(), worker_sha256=overlay.WORKER_SHA,
        asset_manifest_sha256=overlay.MANIFEST_SHA, model_sha256=overlay.MODEL_SHA)
    return notebook, meta, inventory


def prepare(stage=STAGE, work=WORK):
    stage, work = Path(stage), Path(work)
    need(not stage.exists(), 'immutable stage already exists')
    notebook, meta, inventory = package()
    need(sha(ASSETS / 'asset-manifest.json') == overlay.MANIFEST_SHA, 'K2 asset manifest drift')
    manifest = json.loads((ASSETS / 'asset-manifest.json').read_bytes())
    for name, pin in manifest['files'].items():
        need(sha(ASSETS / name) == pin, 'K2 asset drift: ' + name)
    need(sha(ASSETS / 'source/l1_rco_k2_worker_v101.py') == overlay.WORKER_SHA, 'worker drift')
    review = ROOT / '.local/scratch/slot-b/k2-release/fix1/BUILD-REVIEW.md'
    inventory['k2_build_review_sha256'] = sha(review)
    inventory['assets'] = dict(files=manifest['files'], bytes=sum((ASSETS / name).stat().st_size for name in manifest['files']))
    inventory['source_sha256'] = {name: sha(ROOT / name) for name in
        ('tools/c85r7_r6w958rco_v1145.py', 'tools/c85r7_rco_overlay_v1.py', 'tools/test_c85r7_r6w958rco_v1145.py')}
    write_once(stage / 'kernel/release.ipynb', encoded(notebook))
    write_once(stage / 'kernel/kernel-metadata.json', encoded(meta))
    write_once(stage / 'embedded-file-manifest.json', encoded(inventory))
    registration = dict(status='PREPARED_LOCAL_ONLY', candidate_id=overlay.CANDIDATE,
        kernel_id=overlay.KERNEL, version=overlay.VERSION, formula=overlay.FORMULA,
        parent=str(PARENT.parent.relative_to(ROOT)).replace('\\', '/'),
        notebook_sha256=sha(stage / 'kernel/release.ipynb'), metadata_sha256=sha(stage / 'kernel/kernel-metadata.json'),
        inventory_sha256=sha(stage / 'embedded-file-manifest.json'), fast_path=False,
        qualified_for_submission=False, execution_evidence='NOT RUN; ordinary visible output required')
    write_once(stage / 'registration.json', encoded(registration))
    write_once(stage / 'CHANGELOG.md', ('# K3 ' + overlay.VERSION + '\n\n'
        'One fixed 90/10 outer rank overlay. All 31 parent members and the exact K2 worker/assets are unchanged. '
        'The parent cell runs in a fresh child at its original paths with FAST=False. The outer runner serializes '
        'parent then RCO, verifies whole-roster success, preserves the parent snapshot, installs pinned pure-Python '
        'wheels into a private target, reuses the parent codec hook, and atomically publishes output plus a distinct actual receipt.\n').encode())
    parent = json.loads((PARENT / 'release.ipynb').read_bytes())
    def readable(value):
        if 'cells' in value:
            return '\n'.join(''.join(c.get('source', [])) for c in value['cells']).splitlines(True)
        return encoded(value).decode().splitlines(True)
    diff = ''.join(difflib.unified_diff(readable(parent), readable(notebook), fromfile='H189 notebook', tofile='K3 notebook'))
    diff += ''.join(difflib.unified_diff((PARENT / 'kernel-metadata.json').read_text().splitlines(True),
        encoded(meta).decode().splitlines(True), fromfile='H189 metadata', tofile='K3 metadata'))
    write_once(stage / 'full-notebook-metadata.diff', diff.encode())
    write_once(work / 'BUILD-IDENTITY.json', encoded(registration))
    return registration


def validate(stage=STAGE):
    stage = Path(stage)
    notebook, meta, inventory = package()
    need((stage / 'kernel/release.ipynb').read_bytes() == encoded(notebook), 'notebook drift')
    need((stage / 'kernel/kernel-metadata.json').read_bytes() == encoded(meta), 'metadata drift')
    # Reopen the actual delivered nested archive, not only generated inventory claims.
    delivered = json.loads((stage / 'kernel/release.ipynb').read_bytes())
    blob = base64.b64decode(assignment(''.join(delivered['cells'][1]['source']), 'BLOB'))
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        parent_script = archive.read('parent-run.py').decode()
        parent_blob = base64.b64decode(assignment(parent_script, 'BLOB'))
        need(hashlib.sha256(parent_blob).hexdigest() == PARENT_BLOB_SHA, 'delivered parent archive changed')
        with zipfile.ZipFile(io.BytesIO(parent_blob)) as inherited:
            actual = {name: hashlib.sha256(inherited.read(name)).hexdigest() for name in inherited.namelist()}
        need(actual == inventory['parent_members'], 'delivered parent members changed')
        need(archive.read('c85r7_rco_overlay_v1.py') == (ROOT / 'tools/c85r7_rco_overlay_v1.py').read_bytes(), 'delivered runner changed')
    recorded = json.loads((stage / 'embedded-file-manifest.json').read_bytes())
    for key, value in inventory.items():
        need(recorded[key] == value, 'inventory mismatch: ' + key)
    for name, pin in recorded['source_sha256'].items():
        need(sha(ROOT / name) == pin, 'source drift')
    reg = json.loads((stage / 'registration.json').read_bytes())
    need(reg['notebook_sha256'] == sha(stage / 'kernel/release.ipynb') and
        reg['metadata_sha256'] == sha(stage / 'kernel/kernel-metadata.json') and
        reg['inventory_sha256'] == sha(stage / 'embedded-file-manifest.json'), 'registration identity')
    return dict(status='PASS_STATIC', parent_members_unchanged=31, worker_unchanged=True,
        fast_path=False, model_executed=False, notebook_sha256=reg['notebook_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else validate(), sort_keys=True))
