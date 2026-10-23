"""EXP184: registered aquila-share labels and one private three-arm Kaggle T4 kernel.

CPU commands: build-labels, stage, verify-stage, push-plan, collect, screen.
The only GPU entry point is remote-run inside the private Kaggle kernel.
Changelog 1.1.0 (2026-10-12): elapsed/projection values are telemetry; no self-imposed run deadline.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = '1.1.0'
SCHEMA = 'exp184-share-v1'
EXP = 'artifacts/experiments/EXP184_AQUILA_SHARE'
PROTOCOL = EXP + '/PROTOCOL.md'
TOOL = 'tools/exp184_aquila_share.py'
RUNNER = 'training/runners/exp184_aquila_share.py'
UID = 'StudyInstanceUID'
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')

# Labels and the pre-registered share rule (PROTOCOL.md).
L2 = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv'
L2_PROVENANCE = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.provenance.json'
L2_SHA = '79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9'
DREAD = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
DREAD_SHA = '8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1'
AQUILA = 'storage/labels/derived_constellation/aquila_v1.csv'
AQUILA_SHA = '2c709b9a5fa053bc791e5f88f63d980a4eb506b2191c2f7d541c87b9140b4425'
DERIVED = {arm: f'storage/labels/derived_constellation/exp184_aquila{share}_v1.csv'
           for arm, share in (('A0', 0), ('A50', 50))}
DERIVED_PROVENANCE = {arm: path[:-4] + '.provenance.json' for arm, path in DERIVED.items()}

# Kaggle package.
ARMS = ('A0', 'A25', 'A50')
LABELS = {'A0': (DERIVED['A0'], 'exp184_aquila0_v1'),
          'A25': (L2, 'l2_dread75_aquila25_v1'),
          'A50': (DERIVED['A50'], 'exp184_aquila50_v1')}
BASE_ARCHIVE = '.local/exp166-vast-stage-v1/package/exp166-code.zip.bin'
BASE_SHA = 'ecac612702c5ac72ca50a7af6a96ff32375cdc025f82f2b2d094a18a0884ac50'
BASE_NAME = 'exp166-code.zip.bin'
OVERLAY_NAME = 'exp184-overlay.zip.bin'
EXP170_PROTOCOL = '.local/exp170-overlay-v1/protocols/fold0.json'
EXP170_MANIFEST = '.local/exp170-overlay-v1/manifest.json'
DATASET = 'taeyangg4/knee-exp184-aquila-share-private'
CORPUS_DATASET = 'taeyangg4/knee-raptor-corpus-336'
KERNEL = 'taeyangg4/exp184-aquila-share-t4'
WORKSPACE = '/tmp/exp184-workspace'
RESULTS = '/tmp/exp184-results'
WORKING = '/kaggle/working'
WORKERS = 4
ARM_CAP_SECONDS = 9000
KERNEL_TIMEOUT_SECONDS = 30600
REQUIRED_OUTPUTS = ('summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                    'oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data, new=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, allow_nan=False, sort_keys=True) + '\n'
    with path.open('x' if new else 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def safe(name):
    p = PurePosixPath(name)
    need(name and not p.is_absolute() and '..' not in p.parts and ':' not in name
         and '\\' not in name, 'unsafe package path: ' + str(name))
    return name


def tag(arm):
    need(arm in ARMS, 'unknown EXP184 arm')
    return f'EXP184_{arm}_CONVNEXT_T_L2_S42_F0'


def protocol_path(arm):
    need(arm in ARMS, 'unknown EXP184 arm')
    return f'{EXP}/protocols/{arm}.json'


# ---------------------------------------------------------------- labels (CPU)

def _rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]
    return header, rows


def _source(path, ids):
    """Select only canonical UIDs; never serialize or print source rows."""
    import numpy as np
    found = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames is not None and UID in reader.fieldnames, 'source UID missing')
        columns = [t if t in reader.fieldnames else t + '_soft' for t in TARGETS]
        need(set(columns) <= set(reader.fieldnames), 'source target missing')
        seen = set()
        for row in reader:
            uid = row[UID]
            need(uid and uid not in seen, 'source duplicate/empty UID')
            seen.add(uid)
            if uid in ids:
                values = np.asarray([float(row[c]) for c in columns], dtype=np.float64)
                need(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                     'source cell invalid')
                found[uid] = values
    need(set(found) == ids, 'source canonical UID coverage differs')
    return found


def mix_rows(ids, dread, aquila, share):
    import numpy as np
    need(share in (0, 25, 50), 'unregistered share')
    d_weight, a_weight = (100-share)/100, share/100
    rows = []
    for uid in ids:
        values = (d_weight * dread[uid] + a_weight * aquila[uid]).astype(np.float32)
        rows.append([uid, *[repr(float(x)) for x in values]])
    return rows


def _csv_text(header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def build_labels(root=ROOT, dry_run=False):
    root = Path(root)
    need((root / PROTOCOL).is_file(), 'Register PROTOCOL.md before building labels')
    for path, digest in ((L2, L2_SHA), (DREAD, DREAD_SHA), (AQUILA, AQUILA_SHA)):
        need(sha(root / path) == digest, 'pinned label input changed: ' + path)
    prov = read(root / L2_PROVENANCE)
    need(prov['csv_sha256'] == L2_SHA and prov['gold58_included'] is False and prov['rows'] == 4347,
         'L2 provenance differs')
    header, rows = _rows(root / L2)
    need(header == [UID, *TARGETS] and len(rows) == 4347 and
         len({r[0] for r in rows}) == len(rows), 'L2 header/UID roster differs')
    need(_csv_text(header, rows).encode() == (root / L2).read_bytes(), 'L2 CSV dialect not reproducible')
    ids = [r[0] for r in rows]
    dread, aquila = _source(root / DREAD, set(ids)), _source(root / AQUILA, set(ids))
    # The historical L2 writer used the same float32 arithmetic; verify it numerically
    # cell by cell, allowing only its CSV number-format difference.
    import numpy as np
    expected25 = mix_rows(ids, dread, aquila, 25)
    need(all(np.array_equal(np.asarray([float(x) for x in old[1:]], np.float32),
                            np.asarray([float(x) for x in new[1:]], np.float32))
             for old, new in zip(rows, expected25)), 'A25 source arithmetic differs from pinned L2')
    result, texts = {}, {}
    for arm, share in (('A0', 0), ('A50', 50)):
        body = _csv_text(header, mix_rows(ids, dread, aquila, share))
        texts[arm] = body
        result[arm] = dict(rows=len(rows), csv_sha256=hashlib.sha256(body.encode()).hexdigest(),
                           share=share, gold58_included=False)
    result['A25'] = dict(rows=len(rows), csv_sha256=L2_SHA, share=25, gold58_included=False)
    if dry_run:
        return result
    for arm in ('A0', 'A50'):
        need(not (root / DERIVED[arm]).exists() and not (root / DERIVED_PROVENANCE[arm]).exists(),
             'versioned share labels already exist')
    for arm in ('A0', 'A50'):
        with (root / DERIVED[arm]).open('x', encoding='utf-8', newline='') as f:
            f.write(texts[arm])
        need(sha(root / DERIVED[arm]) == result[arm]['csv_sha256'], 'share label write differs')
        write_json(root / DERIVED_PROVENANCE[arm], dict(
            name=LABELS[arm][1], **result[arm], oai=False,
            created=time.strftime('%Y-%m-%d %H:%M:%S %z'), builder=TOOL, version=VERSION,
            protocol=PROTOCOL, protocol_sha256=sha(root / PROTOCOL),
            formula=f'float32({(100-result[arm]["share"])/100}*Dread + {result[arm]["share"]/100}*aquila) per cell',
            canonical_order=L2, inputs={L2: L2_SHA, DREAD: DREAD_SHA, AQUILA: AQUILA_SHA},
            label_rows_printed=False))
    return result


# ------------------------------------------------- frozen-runner guard interface

def current_arm():
    arm = os.environ.get('EXP184_ARM')
    need(arm in ARMS, 'EXP184_ARM must name A0, A25 or A50')
    return arm


def __getattr__(name):
    # The derived EXP166 runner calls these unchanged helpers through its guard.
    if name == 'LABEL_SET':
        return LABELS[current_arm()][1]
    if name.startswith('__') or name in ('validate_arguments', 'validate_fold_membership'):
        raise AttributeError(name)
    from tools import exp166_convnext_t as source
    return getattr(source, name)


def validate_fold_membership(corpus_ids, ordered, rows, canonical):
    # Same as EXP170's guard (its tool is not in the Kaggle workspace): canonical list sorted.
    from tools import exp166_convnext_t as source
    return source.validate_fold_membership(corpus_ids, ordered, rows, sorted(canonical))


def validate_arguments(a, root):
    from tools import exp166_convnext_t as source
    root = Path(root)
    arm = current_arm()
    q = read(root / protocol_path(arm))
    need(Path(a.protocol).resolve() == (root / protocol_path(arm)).resolve() and q['arm'] == arm
         and a.variant == 'AUG1' and a.fold == 0 and a.seed == 42 and a.tag == tag(arm)
         and a.mode == 'full' and not a.validate_only, 'wrong EXP184 arm/fold/seed/tag/mode')
    for key, value in q['common_args'].items():
        need(getattr(a, key) == value, 'frozen argument changed: ' + key)
    need(0 <= a.workers <= 12 and a.device == 'cuda' and os.environ.get('HF_HUB_OFFLINE') == '1'
         and all(os.environ.get(k) == '1' for k in source.THREADS), 'runtime environment differs')
    for key in ('corpus', 'labels', 'eval_labels'):
        need(Path(getattr(a, key)).resolve() == (root / q['paths'][key]).resolve(),
             'input path differs: ' + key)
    need(q['paths']['labels'] == q['paths']['eval_labels'] == LABELS[arm][0], 'arm label path differs')
    need(Path(a.init_weights).resolve() == (root / source.WEIGHT).resolve() and
         Path(a.output_root).resolve() == Path(RESULTS).resolve(), 'weight/output path differs')
    info = q['folds']['0']
    return q, info, read(root / info['ordered_ids']['path']), Path(RESULTS)


def runner_source(root=ROOT):
    from tools import exp166_convnext_t as source
    old = source.runner_source((Path(root) / source.BASE_RUNNER).read_text(encoding='utf-8'))
    anchor = 'from tools import exp166_convnext_t as guard'
    need(old.count(anchor) == 1, 'EXP166 runner guard changed')
    new = old.replace(anchor, 'from tools import exp184_aquila_share as guard')
    compile(new, RUNNER, 'exec')
    return new


def guard_attributes(runner):
    names = {n.attr for n in ast.walk(ast.parse(runner)) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == 'guard'}
    module = sys.modules[__name__]
    previous = os.environ.get('EXP184_ARM')
    os.environ['EXP184_ARM'] = 'A25'
    try:
        missing = sorted(n for n in names if not hasattr(module, n))
    finally:
        if previous is None:
            os.environ.pop('EXP184_ARM', None)
        else:
            os.environ['EXP184_ARM'] = previous
    need(not missing, 'runner guard attributes missing: ' + ', '.join(missing))
    return sorted(names)


def runner_args(arm, root=ROOT):
    from tools import exp166_convnext_t as source
    root = Path(root)
    args = source.runner_args(root, RESULTS, workers=WORKERS, pilot=False)
    def set_flag(flag, value):
        args[args.index(flag) + 1] = str(value)
    args[1] = str(root / RUNNER)
    set_flag('--protocol', root / protocol_path(arm))
    set_flag('--fold', 0)
    set_flag('--tag', tag(arm))
    set_flag('--labels', root / LABELS[arm][0])
    set_flag('--eval-labels', root / LABELS[arm][0])
    return args


def build_protocol(arm, root=ROOT):
    """EXP170 fold-0 protocol construction with this arm's labels (fold-local pos weights)."""
    from tools import exp166_convnext_t as source
    root = Path(root)
    registered, derived = source.registration(root)
    label, name = LABELS[arm]
    labels, semantic = source.old.load_label_csv(root / label)
    info = copy.deepcopy(registered['folds']['0'])
    from tools import exp170_convnext_t_l2 as e170
    ids = read(root / e170.FOLD0_ROSTER)
    need(sha(root / e170.FOLD0_ROSTER) == info['ordered_ids']['sha256'], 'EXP166 fold-0 roster differs')
    vector = source.old.pos_weight(labels, ids['train'])
    original = info['pos_weight']
    info['pos_weight'] = [float(x) for x in vector]
    info['pos_weight_equals_exp099'] = info['pos_weight'] == original
    info['tag'] = tag(arm)
    info['ordered_ids'] = {'path': e170.FOLD0_ROSTER, 'sha256': sha(root / e170.FOLD0_ROSTER),
                           'train_sha256': source.old.uid_hash(ids['train']),
                           'val_sha256': source.old.uid_hash(ids['val'])}
    runner = runner_source(root)
    q = copy.deepcopy(registered)
    q.update(schema=SCHEMA, status='STAGED_NOT_LAUNCHED', arm=arm,
             source_manifest_sha256=sha(root / source.MANIFEST), protocol_sha256=sha(root / PROTOCOL),
             runner_sha256=hashlib.sha256(runner.encode()).hexdigest(), tool_sha256=sha(root / TOOL),
             folds={'0': info}, tags={'prefix': tag(arm)}, fold=0)
    q['paths']['labels'] = q['paths']['eval_labels'] = label
    q['label_set'] = {'name': name, 'path': label, 'sha256': sha(root / label), 'semantic_sha256': semantic}
    q['recipe']['changes'] = list(registered['recipe']['changes']) + [
        f'EXP184 {arm}: registered global aquila share; fold-local clipped positive weights recomputed']
    return q


RECIPE_KEYS = ('common_args', 'extra_eval', 'initial_state', 'initial_state_sha256', 'initial_state_file_sha256',
               'init_weights', 'seed', 'variant', 'environment', 'source_manifest_sha256')


def control_matches_exp170(q, root=ROOT):
    """A25 must match EXP170 and the verified EXP182 CONTROL recipe."""
    root = Path(root)
    m = read(root / EXP170_MANIFEST)
    need(sha(root / EXP170_PROTOCOL) == m['folds']['0']['protocol_sha256'], 'EXP170 fold-0 protocol drift')
    ref = read(root / EXP170_PROTOCOL)
    for key in RECIPE_KEYS:
        need(q.get(key) == ref.get(key), 'CONTROL recipe differs from EXP170: ' + key)
    need(q['paths'] == ref['paths'] and q['label_set'] == ref['label_set'], 'CONTROL labels/paths differ from EXP170')
    a, b = dict(q['folds']['0']), dict(ref['folds']['0'])
    a.pop('tag'), b.pop('tag')
    need(a == b, 'CONTROL fold-0 roster/pos weight/steps differ from EXP170')
    from tools import exp182_synovitis_fill as e182
    old = read(root / e182.protocol_path('CONTROL'))
    for key in RECIPE_KEYS:
        need(q.get(key) == old.get(key), 'A25 differs from EXP182 CONTROL: ' + key)
    need(q['paths'] == old['paths'] and q['label_set'] == old['label_set'],
         'A25 labels differ from EXP182 CONTROL')
    old_info = dict(old['folds']['0'])
    old_info.pop('tag')
    need(a == old_info, 'A25 fold recipe differs from EXP182 CONTROL')
    return True


# ------------------------------------------------------------ staging (CPU)

def _overlay_payload(root):
    """Generated runner/protocols live in the repo so collection can re-verify them.

    They are regenerated on every stage; the stage receipt pins their hashes, so a
    restage after launch makes collect fail closed instead of silently passing.
    """
    root = Path(root)
    runner = runner_source(root)
    guard_attributes(runner)
    (root / RUNNER).write_text(runner, encoding='utf-8', newline='\n')
    for arm in ARMS:
        q = build_protocol(arm, root)
        if arm == 'A25':
            control_matches_exp170(q, root)
        write_json(root / protocol_path(arm), q, new=False)
    for arm in ('A0', 'A50'):
        record = read(root / DERIVED_PROVENANCE[arm])
        need(record['csv_sha256'] == sha(root / DERIVED[arm]) and
             record['protocol_sha256'] == sha(root / PROTOCOL), 'derived label provenance differs')
    names = [TOOL, PROTOCOL, RUNNER, 'config/workspace.toml', L2, L2_PROVENANCE,
             *DERIVED.values(), *DERIVED_PROVENANCE.values(),
             *(protocol_path(arm) for arm in ARMS)]
    return {safe(n): {'sha256': sha(root / n), 'bytes': (root / n).stat().st_size} for n in names}


def _base_names(root):
    root = Path(root)
    need(sha(root / BASE_ARCHIVE) == BASE_SHA, 'EXP166 code archive changed')
    with zipfile.ZipFile(root / BASE_ARCHIVE) as z:
        names = z.namelist()
    need(len(names) == len(set(names)), 'duplicate base archive entries')
    return set(names)


def notebook_source(overlay_sha):
    return f'''# EXP184 v{VERSION} H-B-91 A0/A25/A50: private T4, internet off, no submission.
import hashlib, os, pathlib, subprocess, sys, time, zipfile
started = time.time()
inp = pathlib.Path('/kaggle/input')
def unique(name, digest):
    hits = [p for p in inp.rglob(name) if hashlib.sha256(p.read_bytes()).hexdigest() == digest]
    assert len(hits) == 1, 'Missing/ambiguous pinned package: ' + name
    return hits[0]
base = unique({BASE_NAME!r}, {BASE_SHA!r})
overlay = unique({OVERLAY_NAME!r}, {overlay_sha!r})
ws = pathlib.Path({WORKSPACE!r})
assert not ws.exists()
ws.mkdir()
for archive in (base, overlay):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist(); assert len(names) == len(set(names))
        for name in names:
            p = pathlib.PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
            assert not (ws / name).exists(), 'Archive collision'
        z.extractall(ws)
env = dict(os.environ, KNEE_WORKSPACE_ROOT=str(ws), CUBLAS_WORKSPACE_CONFIG=':4096:8',
           KNEE_REMOTE_RELAX_ENV='1', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '1'
cmd = [sys.executable, str(ws / {TOOL!r}), 'remote-run', '--started', str(started)]
done = subprocess.run(cmd, cwd=ws, env=env)
assert done.returncode == 0, 'EXP184 incomplete; inspect exp184-status.json and the launcher logs'
print(pathlib.Path({WORKING!r} + '/exp184-status.json').read_text())
'''


def kernel_metadata():
    from tools import exp150_consensus_weight as e150
    return dict(id=KERNEL, title='EXP184 aquila share T4', code_file='run.ipynb', language='python',
                kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
                enable_internet=False, machine_shape='NvidiaTeslaT4', docker_image=e150.DOCKER,
                dataset_sources=[CORPUS_DATASET, DATASET],
                competition_sources=['rsna-knee-abnormality-detection'], kernel_sources=[], model_sources=[])


def stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    need(not out.exists(), 'Use a new stage directory')
    files = _overlay_payload(root)
    clash = sorted(set(files) & _base_names(root))
    need(not clash, 'overlay collides with EXP166 base archive: ' + ', '.join(clash))
    code = out / 'code-dataset'
    code.mkdir(parents=True)
    overlay = code / OVERLAY_NAME
    with zipfile.ZipFile(overlay, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.write(root / name, name)
        z.writestr('exp184-overlay-manifest.json', json.dumps({'files': files}, sort_keys=True))
    try:
        os.link(root / BASE_ARCHIVE, code / BASE_NAME)
    except OSError:
        shutil.copyfile(root / BASE_ARCHIVE, code / BASE_NAME)
    need(sha(code / BASE_NAME) == BASE_SHA, 'staged base archive differs')
    write_json(code / 'dataset-metadata.json', dict(id=DATASET, title='EXP184 aquila share private code',
               isPrivate=True, licenses=[{'name': 'other'}],
               description='Private EXP184 code: pinned EXP166 archive and EXP184 overlay (three label files).'))
    receipt = dict(schema='exp184-stage-v1', version=VERSION, base_archive_sha256=BASE_SHA,
                   overlay_sha256=sha(overlay), files=files, arms=list(ARMS), kernel=KERNEL, dataset=DATASET,
                   corpus_dataset=CORPUS_DATASET, arm_cap_seconds=ARM_CAP_SECONDS,
                   kernel_timeout_seconds=KERNEL_TIMEOUT_SECONDS, pushed=False)
    write_json(out / 'stage-receipt.json', receipt)
    kernel = out / 'kernel'
    source = notebook_source(receipt['overlay_sha256'])
    compile(source, '<exp184-notebook>', 'exec')
    write_json(kernel / 'run.ipynb', dict(nbformat=4, nbformat_minor=5,
               metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
               cells=[dict(cell_type='code', id='exp184', metadata={}, execution_count=None, outputs=[],
                           source=source.splitlines(keepends=True))]))
    write_json(kernel / 'kernel-metadata.json', kernel_metadata())
    verify_stage(out, root)
    return {'status': 'STAGED_NO_PUSH', 'stage': out.as_posix(), 'overlay_sha256': receipt['overlay_sha256'],
            'overlay_bytes': overlay.stat().st_size, 'base_bytes': (code / BASE_NAME).stat().st_size}


def verify_stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    m = read(out / 'stage-receipt.json')
    need(m['schema'] == 'exp184-stage-v1' and m['version'] in ('1.0.0', '1.1.0', VERSION) and m['base_archive_sha256'] == BASE_SHA
         and m['pushed'] is False, 'stage registration differs')
    overlay = out / 'code-dataset' / OVERLAY_NAME
    need(sha(overlay) == m['overlay_sha256'] and sha(out / 'code-dataset' / BASE_NAME) == BASE_SHA,
         'staged archives differ')
    with zipfile.ZipFile(overlay) as z:
        expected = set(m['files']) | {'exp184-overlay-manifest.json'}
        need(set(z.namelist()) == expected and len(z.namelist()) == len(expected), 'overlay entries changed')
        need(json.loads(z.read('exp184-overlay-manifest.json')) == {'files': m['files']}, 'overlay manifest changed')
        for name, rec in m['files'].items():
            data = z.read(name)
            need(len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256'] ==
                 sha(root / name), 'overlay file changed: ' + name)
    meta = read(out / 'kernel' / 'kernel-metadata.json')
    need(meta == kernel_metadata() and meta['is_private'] and not meta['enable_internet'] and
         meta['machine_shape'] == 'NvidiaTeslaT4', 'unsafe kernel metadata')
    cells = read(out / 'kernel' / 'run.ipynb')['cells']
    need(len(cells) == 1 and ''.join(cells[0]['source']) == notebook_source(m['overlay_sha256']),
         'kernel notebook differs')
    data = read(out / 'code-dataset' / 'dataset-metadata.json')
    need(data['id'] == DATASET and data['isPrivate'] is True, 'dataset metadata not private')
    return m


def push_plan(out, root=ROOT):
    verify_stage(out, root)
    out = Path(out).as_posix()
    return '\n'.join([
        '# EXP184 push plan (not executed by the tool). B: check live Kaggle quota (>= 8.5 T4 h headroom) and batch slots.',
        "$kagglePython = (Resolve-Path '.venv/Scripts/python.exe').Path",
        f'Push-Location {out}/code-dataset',
        '& $kagglePython -X utf8 -m kaggle datasets create -p . --dir-mode zip   # private; use `datasets version` if it exists',
        'Pop-Location',
        '# Wait until the dataset is ready, then:',
        f'Push-Location {out}/kernel',
        '& $kagglePython -X utf8 -m kaggle kernels push -p .',
        'Pop-Location',
        f'# After COMPLETE: & $kagglePython -X utf8 -m kaggle kernels output {KERNEL} -p {out}/download',
        f'# Then: python tools/exp184_cpu.py collect --experiment exp184 --source {out}/download',
        '# CPU screen after collection: python tools/exp184_cpu.py screen --experiment exp184',
        '# EXP182 after its output download: python tools/exp184_cpu.py collect --experiment exp182 --source <exp182-download>',
        '# EXP182 screen: python tools/exp184_cpu.py screen --experiment exp182'])


# ------------------------------------------------------------ GPU (Kaggle only)

def _status(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def verify_overlay(root):
    root = Path(root)
    m = json.loads((root / 'exp184-overlay-manifest.json').read_text(encoding='utf-8'))
    for name, rec in m['files'].items():
        need((root / name).is_file() and (root / name).stat().st_size == rec['bytes'] and
             sha(root / name) == rec['sha256'], 'workspace overlay file differs: ' + name)
    bundle = read(root / 'exp166-bundle.json')
    for name, rec in bundle['files'].items():
        need((root / name).stat().st_size == rec['bytes'] and sha(root / name) == rec['sha256'],
             'workspace EXP166 file differs: ' + name)
    return len(m['files']) + len(bundle['files'])


def _link_corpus(root):
    from tools import exp150_consensus_weight as e150
    from tools import exp166_convnext_t as source
    root = Path(root)
    hits = [x.parent for x in Path('/kaggle/input').rglob('manifest.json') if sha(x) == e150.CORPUS_HASH]
    need(len(hits) == 1, 'RAPTOR64 corpus mount missing or ambiguous')
    receipt = e150.verify_corpus(hits[0], root / e150.CORPUS_LOCAL, read(root / e150.MANIFEST)['corpus_file_list'])
    target = root / read(root / source.MANIFEST)['paths']['corpus']
    need(not target.exists() and not target.is_symlink(), 'corpus path already exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(hits[0], target_is_directory=True)
    return receipt


def verify_run(run, arm, root=ROOT, gpu='T4'):
    from tools import exp162_dino_folds_kaggle as folds
    from tools import exp166_convnext_t as source
    root, run = Path(root), Path(run)
    q = read(root / protocol_path(arm))
    info = q['folds']['0']
    need(run.name == tag(arm), 'wrong run tag')
    inv = read(run / 'output-inventory.json')
    need(set(inv) == set(REQUIRED_OUTPUTS), 'incomplete EXP184 output inventory')
    for name, record in inv.items():
        need(safe(name) == name and (run / name).is_file() and (run / name).stat().st_size == record['bytes']
             and sha(run / name) == record['sha256'], 'output inventory differs: ' + name)
    summary, prov, history = (read(run / n) for n in ('summary.json', 'provenance.json', 'history.json'))
    need(summary['fold'] == 0 and summary['train'] == info['n_train'] == 3455 and
         summary['val'] == summary['rows'] == info['n_val'] == 888 and summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == info['steps'] and summary['skipped_updates'] <= info['max_fp16_skips'] and
         len(history) == 20 and [h['epoch'] for h in history] == list(range(20)) and
         summary['label_set'] == prov['label_set'] == LABELS[arm][1] and summary['gold_access'] == 'disabled' and
         gpu in summary['environment']['gpu'] and gpu in prov['environment']['gpu'], 'incomplete fixed-final EXP184 arm')
    args = summary['args']
    need(all(args.get(k) == v for k, v in q['common_args'].items()) and args['seed'] == 42 and
         args['fold'] == 0 and args['tag'] == tag(arm) and args['mode'] == 'full' and
         args['labels'] == args['eval_labels'] == f'{WORKSPACE}/{LABELS[arm][0]}' and
         prov['initial_state_sha256'] == q['initial_state_sha256'] and
         prov['init_weights']['weights_sha256'] == q['init_weights']['sha256'] and
         prov['runner_sha256'] == sha(root / RUNNER) == q['runner_sha256'] and
         prov['guard_sha256'] == sha(root / TOOL) == q['tool_sha256'] and
         prov['protocol_sha256'] == sha(root / protocol_path(arm)) and
         prov['train_ids_ordered_sha256'] == info['ordered_ids']['train_sha256'] and
         prov['eval_ids_ordered_sha256'] == info['ordered_ids']['val_sha256'] and
         prov['actual_pos_weight'] == info['pos_weight'] and
         prov['flip_tta'] == 'coronal_axial_horizontal_mean_probability' and
         prov['label_file_sha256'] == q['label_set']['sha256'] and
         prov['label_semantic_sha256'] == q['label_set']['semantic_sha256'], 'arm recipe/provenance differs')
    ids = read(root / info['ordered_ids']['path'])['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        folds._prediction_values(run / name, ids)
    return {name: inv[name]['sha256'] for name in REQUIRED_OUTPUTS}


def remote_run(started, root=ROOT):
    root = Path(root).resolve()
    need(sys.platform.startswith('linux') and str(root) == WORKSPACE and Path(WORKING).is_dir() and
         os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'EXP184 remote-run requires the Kaggle workspace')
    status_path = Path(WORKING) / 'exp184-status.json'
    status = dict(version=VERSION, state='RUNNING', arms={}, started=float(started))
    _status(status_path, status)
    status['workspace_files_verified'] = verify_overlay(root)
    smi = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    need(len(smi) in (1, 2) and all('T4' in x for x in smi), 'Kaggle T4 required')
    status['gpu'] = smi
    status['corpus'] = _link_corpus(root)
    import torch
    import timm
    import numpy
    need(not torch.cuda.is_initialized(), 'supervisor touched CUDA')
    status['software'] = dict(python=sys.version.split()[0], torch=torch.__version__, timm=timm.__version__,
                              numpy=numpy.__version__, cuda=torch.version.cuda)
    _status(status_path, status)
    Path(RESULTS).mkdir(parents=True, exist_ok=False)
    from tools import exp166_convnext_t as source
    for arm in ARMS:
        env = dict(os.environ, EXP184_ARM=arm, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   KNEE_WORKSPACE_ROOT=str(root), KNEE_REMOTE_RELAX_ENV='1', HF_HUB_OFFLINE='1',
                   PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env.pop('EXP166_REMOTE', None)
        for key in source.THREADS:
            env[key] = '1'
        run_start = time.time()
        log_path = Path(WORKING) / f'{tag(arm)}.launcher.log'
        state = 'FAILED'
        try:
            with log_path.open('x', encoding='utf-8') as log:
                done = subprocess.run(runner_args(arm, root), cwd=root, env=env, stdout=log,
                                      stderr=subprocess.STDOUT)
            state = 'CHILD_EXIT_%d' % done.returncode
            need(done.returncode == 0, arm + ' runner failed')
            hashes = verify_run(Path(RESULTS) / tag(arm), arm, root)
            dest = Path(WORKING) / tag(arm)
            dest.mkdir()
            for name in (*REQUIRED_OUTPUTS, 'output-inventory.json'):
                shutil.copyfile(Path(RESULTS) / tag(arm) / name, dest / name)
            state = 'COMPLETE'
            status['arms'][arm] = dict(state=state, seconds=time.time() - run_start, sha256=hashes)
        except Exception:
            status['arms'][arm] = dict(state=state, seconds=time.time() - run_start)
            _status(status_path, dict(status, state='FAILED'))
            raise
        _status(status_path, status)
    status.update(state='COMPLETE', total_seconds=time.time() - float(started))
    _status(status_path, status)
    return status


def collect(source_dir, root=ROOT):
    root, source_dir = Path(root), Path(source_dir)
    status = read(source_dir / 'exp184-status.json')
    need(status['state'] == 'COMPLETE' and status['version'] in ('1.0.0', VERSION) and
         all(status['arms'][a]['state'] == 'COMPLETE' for a in ARMS), 'kernel did not complete every arm')
    out = {}
    for arm in ARMS:
        run = source_dir / tag(arm)
        hashes = verify_run(run, arm, root)
        need(hashes == status['arms'][arm]['sha256'], 'status hashes differ: ' + arm)
        dest = root / EXP / 'outputs' / 'kaggle' / tag(arm)
        need(not dest.exists(), 'collected output exists: ' + arm)
        dest.mkdir(parents=True)
        for name in (*REQUIRED_OUTPUTS, 'output-inventory.json'):
            shutil.copyfile(run / name, dest / name)
        out[arm] = dest.as_posix()
    shutil.copyfile(source_dir / 'exp184-status.json', root / EXP / 'outputs' / 'kaggle' / 'exp184-status.json')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build-labels')
    b.add_argument('--dry-run', action='store_true')
    for name in ('stage', 'verify-stage', 'push-plan'):
        sub.add_parser(name).add_argument('--out', type=Path, required=True)
    sub.add_parser('collect').add_argument('--source', type=Path, required=True)
    sub.add_parser('remote-run').add_argument('--started', type=float, required=True)
    sub.add_parser('fingerprint')
    a = ap.parse_args(argv)
    if a.command == 'build-labels':
        result = build_labels(dry_run=a.dry_run)
    elif a.command == 'stage':
        result = stage(a.out)
    elif a.command == 'verify-stage':
        result = verify_stage(a.out)
    elif a.command == 'push-plan':
        print(push_plan(a.out))
        return 0
    elif a.command == 'collect':
        result = collect(a.source)
    elif a.command == 'remote-run':
        result = remote_run(a.started)
    else:
        result = dict(version=VERSION, files={n: sha(ROOT / n) for n in (TOOL, PROTOCOL)})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
