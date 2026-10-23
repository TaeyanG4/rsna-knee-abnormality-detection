"""H208 v1.0.0: exact INF3 parent in the unchanged K3 90/10 coordinator.

CPU packaging/replay only; no model loads, GPU, network or submission command.
The Oct19 solo override permits B's direct checks, not an independent PASS.
Old stages and builders are immutable. Runtime qualification remains separate.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import difflib
import hashlib
import io
import json
from pathlib import Path
import zipfile

from tools import c85r7_r6w958rco_v1145 as k3
from tools import c85r7_rco_overlay_v1 as overlay

ROOT = k3.ROOT
VERSION = '1.1.50-r6w958inf3rco'
SLUG = 'taeyangg4/knee-lb-r6w958inf3rco-v1'
KEY = 'R6W958-INF3RCO-V1'
FORMULA = 'rank(0.90*rank(F_INF3_float32_csv)+0.10*rank(RCO_v2_float32)); average ties/full roster; final float32'
INF3 = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.46-r6w958inf3'
K3 = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.45-r6w958rco'
STAGE = INF3.parent / ('v' + VERSION)
WORK = ROOT / '.local/scratch/slot-b/research-resume-1018/h208-build'
INF3_SHA = 'baec47351a8b7823a49bd6013e95b9b5a86721f31e51efa5064bfc3c3bb0d150'
K3_SHA = '641d675360d2b1ee935e1d9e6b74722285d3e03ab6c8234af0a9d4310e68b5e8'
INVENTORY = WORK.parent / 'H208-INPUTS.json'
INVENTORY_SHA = '805e8ed38e654a227c6e72a7c4cc73b61016c23c6448218c3de99067b5f36a38'
need, sha, encoded = overlay.need, overlay.sha, overlay.encoded


def digest(data):
    return hashlib.sha256(data).hexdigest()


def exact(source, before, after):
    need(source.count(before) == 1, 'missing/duplicate patch anchor')
    return source.replace(before, after)


def archive(code):
    blob = base64.b64decode(k3.assignment(code, 'BLOB'))
    pins = k3.assignment(code, 'PINS')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) == len(pins) and set(names) == set(pins), 'archive names')
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names), 'unsafe archive')
        files = {n: z.read(n) for n in names}
    need({n: digest(v) for n, v in files.items()} == pins, 'archive pins')
    return blob, pins, files


def packed(files, original_blob):
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original_blob)) as src, zipfile.ZipFile(buffer, 'w') as dst:
        need(set(src.namelist()) == set(files), 'outer payload roster')
        for info in src.infolist():
            dst.writestr(info, files[info.filename])
    return buffer.getvalue()


def runtime_source(original):
    need(original == Path(overlay.__file__).read_bytes(), 'K3 runtime source drift')
    source = original.decode()
    replacements = {
        'VERSION': VERSION, 'KERNEL': SLUG, 'CANDIDATE': KEY,
        'FORMULA': FORMULA, 'PARENT_VERSION': '1.1.46-r6w958inf3',
    }
    for name, value in replacements.items():
        source = exact(source, name + ' = ' + repr(getattr(overlay, name)), name + ' = ' + repr(value))
    # Only receipt member labels change inside run(); execution/math stay intact.
    need(source.count("'H189'") == 4, 'expected receipt labels')
    source = source.replace("'H189'", "'INF3'")
    normalized = source
    for name, value in replacements.items():
        normalized = exact(normalized, name + ' = ' + repr(value), name + ' = ' + repr(getattr(overlay, name)))
    normalized = normalized.replace("'INF3'", "'H189'")
    need(normalized.encode() == original, 'non-metadata runtime change')
    compile(source, '<H208 existing coordinator>', 'exec')
    return source.encode()


def package():
    need(sha(INF3 / 'kernel/release.ipynb') == INF3_SHA, 'INF3 notebook pin')
    need(sha(K3 / 'kernel/release.ipynb') == K3_SHA, 'K3 notebook pin')
    parent = json.loads((INF3 / 'kernel/release.ipynb').read_bytes())
    base = json.loads((K3 / 'kernel/release.ipynb').read_bytes())
    parent_code = ''.join(parent['cells'][2]['source'])
    parent_blob, parent_pins, parent_files = archive(parent_code)
    need(len(parent_files) == 31, 'INF3 full payload count')
    parent_script = ('FAST = False\n' + parent_code).encode()
    code = ''.join(base['cells'][1]['source'])
    old_blob, old_pins, old_files = archive(code)
    files = {'parent-run.py': parent_script,
             'c85r7_rco_overlay_v1.py': runtime_source(old_files['c85r7_rco_overlay_v1.py'])}
    blob = packed(files, old_blob)
    pins = {n: digest(v) for n, v in files.items()}
    new_code = exact(code, repr(base64.b64encode(old_blob).decode()), repr(base64.b64encode(blob).decode()))
    new_code = exact(new_code, 'PINS = ' + repr(old_pins), 'PINS = ' + repr(pins))
    new_code = exact(new_code, repr(digest(old_blob)), repr(digest(blob)))
    compile(new_code, '<H208 notebook>', 'exec')
    compile(parent_script, '<H208 exact INF3 child>', 'exec')
    nb = copy.deepcopy(base)
    nb['cells'][1]['source'] = new_code.splitlines(True)
    nb['cells'][0]['source'] = ['# H208 fixed INF3 + K1 v2 RCO\n', FORMULA + '\n',
        'Exact INF3 child and unchanged K3 execution/overlay. Qualification pending.\n',
        ''.join(parent['cells'][1]['source'])]
    meta = json.loads((K3 / 'kernel/kernel-metadata.json').read_bytes())
    parent_meta = json.loads((INF3 / 'kernel/kernel-metadata.json').read_bytes())
    need(meta['dataset_sources'] == parent_meta['dataset_sources'] + ['taeyangg4/knee-l1-rcof0-v1-assets'], 'exact mounts')
    for name, value in parent_meta.items():
        if name not in ('id', 'title', 'dataset_sources'):
            need(meta.get(name) == value, 'parent metadata differs: ' + name)
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    need(meta['is_private'] is True and meta['enable_internet'] is False, 'private offline required')
    proof = dict(parent_notebook_sha256=INF3_SHA, coordinator_notebook_sha256=K3_SHA,
        parent_archive_sha256=digest(parent_blob), parent_members=parent_pins,
        parent_members_byte_identical=31, parent_cell_byte_identical=True,
        outer_members=pins, worker_sha256=overlay.WORKER_SHA, model_sha256=overlay.MODEL_SHA,
        asset_manifest_sha256=overlay.MANIFEST_SHA,
        runtime_change='identity constants and four receipt member labels only; all other bytes identical',
        builder='Codex B solo', independent_new_review=False)
    return nb, meta, proof, old_files, files


def materialize(stage=STAGE, work=WORK):
    nb, meta, proof, old, new = package()
    stage, work = Path(stage), Path(work)
    need(not stage.exists(), 'immutable stage already exists')
    registration = dict(status='PREPARED_LOCAL_ONLY', candidate_id=KEY, kernel_id=SLUG,
        version=VERSION, formula=FORMULA, notebook_sha256=digest(encoded(nb)),
        metadata_sha256=digest(encoded(meta)), qualified_for_submission=False,
        runtime='UNMEASURED for H208; ordinary visible output and material runtime check pending',
        independent_review='No new independent review; B direct solo check under Oct19 user override',
        source_sha256=sha(__file__))
    outputs = {'kernel/release.ipynb': encoded(nb), 'kernel/kernel-metadata.json': encoded(meta),
        'embedded-file-manifest.json': encoded(proof), 'registration.json': encoded(registration),
        'CHANGELOG.md': ('# H208 ' + VERSION + '\n\nOnly replace the K3 parent with exact INF3 and update identity metadata. '
        'All31 INF3 embedded files are unchanged; the K3 launch, cleanup, codecs, model, overlay, publication '
        'and timing code remain byte-identical after reversing metadata edits. No GPU or Kaggle execution.\n').encode()}
    diffs = []
    for name in new:
        # Unpack nested source for a reviewable full file diff, rather than base64 only.
        if name == 'parent-run.py':
            _, _, before = archive(old[name].decode())
            _, _, after = archive(new[name].decode())
            for member in sorted(before):
                if before[member] != after[member]:
                    diffs.extend(difflib.unified_diff(before[member].decode().splitlines(True),
                        after[member].decode().splitlines(True), fromfile='H189/' + member, tofile='INF3/' + member))
        else:
            diffs.extend(difflib.unified_diff(old[name].decode().splitlines(True), new[name].decode().splitlines(True),
                fromfile='K3/' + name, tofile='H208/' + name))
    outputs['full-payload.diff'] = ''.join(diffs).encode()
    for name, data in outputs.items():
        overlay.write_once(stage / name, data)
    overlay.write_once(work / 'BUILD-IDENTITY.json', encoded(registration))
    return registration


def validate(stage=STAGE):
    nb, meta, proof, _, files = package()
    stage = Path(stage)
    for name, value in [('kernel/release.ipynb', nb), ('kernel/kernel-metadata.json', meta),
                        ('embedded-file-manifest.json', proof)]:
        need((stage / name).read_bytes() == encoded(value), 'delivered drift: ' + name)
    delivered = json.loads((stage / 'kernel/release.ipynb').read_bytes())
    _, _, actual = archive(''.join(delivered['cells'][1]['source']))
    need(actual == files, 'actual outer archive differs')
    return dict(status='PASS_STATIC_B_DIRECT', parent_members_unchanged=31,
        formula=FORMULA, model_executed=False, qualified_for_submission=False)


def replay(work=WORK):
    """Existing58 cached values only; no labels, statistical study or inference."""
    import numpy as np
    from tools import c85r7_r6w958q_v1139 as parent
    need(sha(INVENTORY) == INVENTORY_SHA, 'reviewed inventory drift')
    inv = json.loads(INVENTORY.read_bytes())
    def pinned(relative):
        path = ROOT / relative
        entry = inv['files'][relative]
        need(sha(path) == entry['sha256'], 'replay input drift: ' + relative)
        return path
    common = '.local/scratch/hb167-common/hb167/gpu/'
    with np.load(pinned(common + 'predictions.npz'), allow_pickle=False) as z:
        a = {n: z[n].copy() for n in z.files}
    with np.load(pinned(common + 'q-leaves.npz'), allow_pickle=False) as z:
        leaves = {n: z[n].copy() for n in z.files}
    ids = a['ids'].tolist()
    need(len(ids) == len(set(ids)) == 58 and ids == leaves['ids'].tolist(), 'exact58 roster')
    need(a['targets'].tolist() == leaves['targets'].tolist() == overlay.LABELS, 'targets')
    need(digest('\n'.join(ids).encode()) == inv['cohort']['ordered_uid_sha256_no_trailing_newline'], 'cohort pin')
    for n in ('R', 'R224', 'EXP143v2', 'EXP155', 'Q_A5q'):
        need(a['failed_' + n].dtype == bool and not a['failed_' + n].any(), 'prediction fallback')
    import pandas as pd
    r6 = pinned('.local/scratch/r6-gold58-build/r6-gold58/gpu/flip/output/predictions.csv')
    frame = pd.read_csv(r6, dtype={overlay.UID: str}, float_precision='round_trip')
    need(frame.columns.tolist() == [overlay.UID, *overlay.LABELS] and frame[overlay.UID].is_unique, 'R6 identity')
    need(set(frame[overlay.UID]) == set(ids), 'R6 cohort')
    a['S_R7'] = frame.set_index(overlay.UID).loc[ids, overlay.LABELS].to_numpy(float)
    nb, _, _, _, files = package()
    _, _, source = archive(files['parent-run.py'].decode())
    ns = parent.math_functions(source)
    # Execute only the existing pure INF3 aggregation and its literal costs.
    tree = ast.parse(source['exp169_release_c85.py'])
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inf3_aggregate'
        or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INF3_COSTS' for t in n.targets)]
    need(len(selected) == 2, 'INF3 pure operator extraction')
    scope = dict(need=need, rank=overlay.ranks)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<pure INF3>', 'exec'), scope)
    args = [a[n] for n in ('R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q_A5q')]
    f189 = ns['p'].csv_bytes(ids, ns['blend'](ids, *args))
    args[-1] = scope['inf3_aggregate']({n: leaves[n] for n in ('l2', 'dino', 'convnext')}, np)
    finf3 = ns['p'].csv_bytes(ids, ns['blend'](ids, *args))
    rco = pinned('.local/scratch/slot-b/research-resume-1018/run1/predictions.csv').read_bytes()
    controls = {'H189': f189, 'INF3': finf3, 'K3': overlay.blend(f189, rco, ids)}
    for name, data in controls.items():
        need(digest(data) == inv['arms'][name]['sha256'], 'old control byte mismatch: ' + name)
    joint = overlay.blend(finf3, rco, ids)
    _, values = overlay.table(joint, ids)
    need(values.shape == (58, 12) and np.isfinite(values).all(), 'joint finite shape')
    record = dict(status='PASS_CPU_REPLAY_B_DIRECT', rows=58, targets=12,
        control_sha256={n: digest(v) for n, v in controls.items()}, joint_sha256=digest(joint),
        formula=FORMULA, exposure='EXPOSED_GOLD58', efficacy_evaluated=False,
        qualified_for_submission=False, model_executed=False,
        rank_identical_to_controls={n: data == joint for n, data in controls.items()})
    overlay.write_once(Path(work) / 'joint58.csv', joint)
    overlay.write_once(Path(work) / 'CPU-REPLAY.json', encoded(record))
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'validate', 'replay'])
    action = p.parse_args().action
    print(json.dumps({'prepare': materialize, 'validate': validate, 'replay': replay}[action](), sort_keys=True))
