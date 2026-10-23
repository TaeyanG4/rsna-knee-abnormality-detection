"""EXP154 B3-H1 v1.0.1: offline registration, stage, push plan and Public3 check.

OAI-derived, fixed .8 OAI + .1 ConvNeXt + .1 H1 percentile-rank blend.
No CLI command invokes CUDA, network, Kaggle, upload or submission. The generated
notebook alone runs inference. Frozen training and public sources are untouched.
Changelog: v1.0.1 binds the corrected EXP151 v1.0.3 source bytes while
retaining the source hash gate and the two-GPU public OAI notebook cell.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import exp133_oai_b3 as e133

VERSION = '1.0.1'
OUT = ROOT / '.local/exp154-b3-h1-release'
LIST = HERE / 'exp154_b3_h1_checkpoints.json'
REGISTRATION = ROOT / '.local/exp154-b3-h1-checkpoints.json'
DATASET = 'taeyangg4/knee-exp154-b3-h1-assets'
KERNEL = 'taeyangg4/knee-exp154-b3-h1'
BASE_NOTEBOOK = '.local/exp140-aug1-release/blend_b3_aug1/release.ipynb'
BASE_SHA = '1147867bc6f4f4e47c92c8e535138b0af2b13b12c684136e3cb3f87fded6ea17'
HEADS = ('dread', 'relabel_v1', 'steven_v4', 'pilkwang_v2', 'aquila_v1')
WEIGHTS = {'oai': .8, 'cnx': .1, 'h1': .1}
INITIAL_SHA = 'a56516a289178dce17d2fe736147ddbbcb52a571cb25e6c8b59b4342a3d3e636'
PROTOCOL149 = 'artifacts/experiments/EXP149_LABEL_SOURCES/students/manifest.json'
PROTOCOL153 = 'artifacts/experiments/EXP153_H1_AQUILA/manifest.json'
CUTOFF, RESERVE = 30600, 120
RULE = 'five head target ranks/model/view; mean views; target rank/model; mean ten models'
# Observed EXP140 B Public3 commit receipt (SHA d3ceda4733b8cbcc1a244e7c4f3d9cbf02c1aca88881ca218a64b5c7cd04ff3d),
# .local/exp140-aug1-release/public3-b-output/exp133_stage_timing.json.
# Pin the unchanged B3 reader, not merely its mutable dataset slug.
CNX_CHECKPOINTS = {
    'cnxt_v0_fold0.pt': '0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80',
    'cnxt_v0_fold1.pt': '61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb',
    'cnxt_v0_fold2.pt': 'f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6',
}
CNX_SOURCES = {
    'infer.py': '39d4517cc33c4bb1e4a03cfb49a0673a8789055de8c3cc0957c6e3035f6be4af',
    'preprocess.py': '68601ad84ec3a0f030b5c2d37cc0d657cd3e12b17f7130e7385fba9648d324b6',
    'knee.py': '314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1',
}
SOURCE_BINDINGS = {
    'tools/exp140_aug1_release.py': '02de8414878c8a92da76ac33004b2f64b0e5f5a2826277232f7c179433dfe2ca',
    'tools/exp133_oai_b3.py': 'b189718d1b8a2d4735286121d6a9c1c409efe65c4cdb43c0fa035df43b4349d6',
    'tools/exp128_oai_light.py': '4a09da811af2ad0ac05c4eade828abe5dcefa6540d5e9a8baf6e5e02d5f49ce8',
    'tools/exp151_bundle_release.py': '7e081ed7fc50228fb8d254252e779b51cacd67a8e368d37f46e6cda11496800d',
    BASE_NOTEBOOK: BASE_SHA,
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def record(path):
    path = Path(path)
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tag(seed, fold):
    return f'{"EXP149" if seed == 42 and fold < 2 else "EXP153"}_H1_S{seed}_F{fold}'


def checkpoint_list(path=LIST):
    value = read(path)
    expected = [{'seed': s, 'fold': f, 'tag': tag(s, f)} for s in (42, 43) for f in range(5)]
    need(value == {'schema': 'exp154-checkpoint-list-v1', 'heads': list(HEADS), 'entries': expected},
         'Registered checkpoint list/order/heads differs')
    return expected


def checkpoint_entry(root, item):
    """Read receipts/protocols and hash outputs only; never read labels/reports."""
    root = Path(root)
    seed, fold, name = item['seed'], item['fold'], item['tag']
    need(name == tag(seed, fold), 'Checkpoint tag differs')
    run = root / 'artifacts/experiments' / name
    inventory_before = record(run / 'output-inventory.json')
    inv = read(run / 'output-inventory.json')
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                'oof_best.csv', 'oof_k36.csv'} | {
                    f'oof_head_{h}{suffix}.csv' for h in HEADS for suffix in ('', '_k36')}
    need(set(inv) == required, 'Incomplete H1 inventory: ' + name)
    for filename in sorted(required):
        need(record(run / filename) == inv[filename], 'H1 inventory SHA/bytes differ: ' + name + '/' + filename)
    s, p, history = (read(run / n) for n in ('summary.json', 'provenance.json', 'history.json'))
    protocol_path = PROTOCOL149 if name.startswith('EXP149') else PROTOCOL153
    protocol = read(root / protocol_path)
    spec = protocol['runs'][name]
    need(s['provenance'] == p and s['history'] == history, 'Summary provenance/history differs: ' + name)
    need(s['run'] == name and s['fold'] == fold and s['mode'] == 'full' and s['label_set'] == 'h1'
         and s['heads'] == list(HEADS) and s['selected_checkpoint'] == 'fixed-final'
         and s['selected_epoch_1based'] == 20 and s['optimizer_updates'] == spec['steps']
         and s['skipped_updates'] == 0 and s['gold'] == 0 and s['gold_access'] == 'disabled'
         and s['worker_cleanup']['clean'] is True and s['total_seconds'] > 0,
         'H1 completion differs: ' + name)
    need([h['epoch'] for h in history] == list(range(20))
         and all(h['fp16_skips'] == 0 for h in history)
         and history[-1]['optimizer_updates'] == spec['steps'], 'H1 history differs: ' + name)
    expect = {'run': name, 'seed': seed, 'fold': fold, 'heads': list(HEADS), 'label_set': 'h1',
              'arch': 'resnet50', 'norm': 'radimagenet', 'input_policy': 'center3',
              'k_eval_extra': 36, 'training_mode': 'full', 'epochs': 20,
              'checkpoint_selection': 'fixed-final', 'gold_access': 'disabled',
              'full_model_initial_state_sha256': INITIAL_SHA, 'fp16_skipped_updates': 0,
              'protocol_sha256': sha(root / protocol_path), 'train_folds': [f for f in range(5) if f != fold],
              'eval_folds': [fold], 'steps': spec['steps']}
    need(all(p.get(k) == v for k, v in expect.items()), 'H1 serving provenance differs: ' + name)
    need(spec['heads'] == list(HEADS) and spec['initial_state_sha256'] == INITIAL_SHA
         and p['train_ids_ordered_sha256'] == spec['train_ids_sha256']
         and p['eval_ids_ordered_sha256'] == spec['eval_ids_sha256']
         and p['head_label_files'] == {h['name']: h['sha256'] for h in spec['head_labels']},
         'H1 protocol heads/roster differ: ' + name)
    a = s['args']
    need(a['variant'] == 'AUG1' and a['mode'] == 'full' and a['seed'] == seed
         and a['fold'] == fold and a['tag'] == name and a['run'] == name
         and all(a.get(k) == v for k, v in protocol['common_args'].items()),
         'H1 AUG1 recipe differs: ' + name)
    if name.startswith('EXP153'):
        need(p['protocol_path'] == PROTOCOL153
             and protocol['exp153']['source_manifest_sha256'] == sha(root / PROTOCOL149)
             and p['initial_state_file_sha256'] == protocol['exp153']['initial_state_file_sha256']
             and p['exp153_surface'] in ('local', 'vast'), 'EXP153 continuation differs: ' + name)
    entry = {**item, 'source': f'artifacts/experiments/{name}/model_best.pt',
            'path': f'checkpoints/s{seed}_f{fold}.pt', **inv['model_best.pt'],
            'heads': list(HEADS), 'protocol_path': protocol_path,
            'protocol_sha256': expect['protocol_sha256'], 'inventory': inventory_before,
            'receipts': {n: inv[n] for n in ('summary.json', 'provenance.json', 'history.json')},
            'surface': p.get('exp153_surface', 'local-EXP149')}
    # Safe tensor loading maps to CPU only; never build a model or touch CUDA.
    # This checks the actual .pt head inventory, not just its sidecar declaration.
    check_checkpoint(load_checkpoint_header(run / 'model_best.pt'), entry)
    need(record(run / 'output-inventory.json') == inventory_before
         and record(run / 'model_best.pt') == inv['model_best.pt'], 'Checkpoint changed during registration')
    return entry


def register_checkpoints(output=REGISTRATION, root=ROOT, list_path=LIST):
    need(not Path(output).exists(), 'Choose a fresh registration file')
    items = checkpoint_list(list_path)
    missing = [i['tag'] for i in items if not (Path(root) / 'artifacts/experiments' /
               i['tag'] / 'output-inventory.json').is_file()]
    need(not missing, 'Incomplete checkpoints; no registration written: ' + ', '.join(missing))
    entries = [checkpoint_entry(root, i) for i in items]
    need(len({e['sha256'] for e in entries}) == 10, 'Ten distinct H1 checkpoints required')
    value = {'schema': 'exp154-registered-checkpoints-v1', 'version': VERSION,
             'heads': list(HEADS), 'list_sha256': sha(list_path), 'entries': entries}
    save(output, value)
    return {'status': 'REGISTERED', 'file': str(output), 'sha256': sha(output), 'checkpoints': 10}


def registered(path, root=ROOT, list_path=LIST):
    value = read(path)
    items = checkpoint_list(list_path)
    need(value['schema'] == 'exp154-registered-checkpoints-v1' and value['version'] == VERSION
         and value['heads'] == list(HEADS) and value['list_sha256'] == sha(list_path),
         'Checkpoint registration differs')
    entries = [checkpoint_entry(root, i) for i in items]
    need(entries == value['entries'] and len({e['sha256'] for e in entries}) == 10,
         'Registered checkpoint SHA/heads/receipt differs; refuse to rebind')
    return value


def projection(n=1300):
    need(type(n) is int and n > 0, 'Positive study count required')
    baseline = 3.2 * 3600 * n / 1300
    # Deliberately charge 15% of the entire baseline (not merely the H1 stage)
    # for extra heads/ranks; then add 30% uncertainty and 10 min startup reserve.
    central = baseline * 1.15
    budget = central * 1.30 + 600
    return {'studies': n, 'baseline_seconds': baseline, 'projected_seconds': central,
            'budget_seconds': budget, 'deadline_seconds': CUTOFF, 'publication_reserve_seconds': RESERVE,
            'margin_seconds': CUTOFF - RESERVE - budget,
            'basis': 'User-reported submitted EXP140 B3-AUG1 (D-B-080) hidden run about 3.2 h on 2xT4; scaled at 1300 studies.',
            'head_allowance': '15% of whole B3 run; H1 shares each backbone across five heads, not five backbone passes.',
            'uncertainty': 'Unmeasured H1/full hidden cohort. Central heuristic 3.68 h; budget 5.0 h at 1300, not a benchmark.',
            'reference_rad_timing_seconds_per_study': 301.32550326700016 / 60,
            'gpu_layout': 'OAI frozen 2xT4 path; H1 and ConvNeXt sequential on one visible T4.',
            'decode_scope': 'One H1 DICOM corpus/study shared across ten models and both views; OAI/reader retain their native decoders.'}


def reduce_heads(plain, mirror, rank_mean, fold_ranks):
    """Exact EXP151 reduction with five heads in place of four."""
    import numpy as np
    a, b = np.asarray(plain), np.asarray(mirror)
    need(a.shape == b.shape and a.ndim == 4 and a.shape[0] == 10
         and a.shape[2:] == (5, 12) and a.shape[1] > 0, 'Model/head/target shape differs')
    need(np.isfinite(a).all() and np.isfinite(b).all()
         and (a >= 0).all() and (a <= 1).all() and (b >= 0).all() and (b <= 1).all(),
         'Invalid head probabilities')
    return np.mean([fold_ranks((rank_mean(a[i]) + rank_mean(b[i])) / 2) for i in range(10)], axis=0)


def blend(oai, cnx, h1):
    need(cnx is not None and h1 is not None, 'All three fixed B3 partners required')
    path, result = e133.blend(oai, rad=h1, cnx=cnx)
    need(path == 'B3', 'B3 fallback forbidden')
    return result


def function_source(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    need(len(nodes) == 1, 'Frozen function missing: ' + name)
    return ast.get_source_segment(source, nodes[0])


def replace_once(source, before, after):
    need(source.count(before) == 1, 'Frozen serving source differs: ' + before[:80])
    return source.replace(before, after, 1)


def load_checkpoint_header(path):
    import torch
    # mmap avoids loading ten full tensor sets into RAM during registration.
    # Unsupported pickle metadata fails closed, with no unsafe-load fallback.
    # B 2026-10-09: our checkpoints record torch.__version__ (a TorchVersion object); allow-list only that class.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def check_checkpoint(ck, entry):
    expect = {'arch': 'resnet50', 'res': 224, 'lab': e133.TARGETS,
              'contract': 'raptor64_336_c140_s02', 'norm': 'radimagenet',
              'window_policy': 'within-slot-valid', 'input_policy': 'center3',
              'tag': entry['tag'], 'fold': entry['fold'], 'heads': list(HEADS),
              'label_set': 'h1', 'epoch': 19}
    need(all(ck.get(k) == v for k, v in expect.items()), 'Checkpoint metadata/heads differs')
    p = ck['provenance']
    need(p['protocol_sha256'] == entry['protocol_sha256'] and p['seed'] == entry['seed']
         and p['full_model_initial_state_sha256'] == INITIAL_SHA and p['heads'] == list(HEADS),
         'Checkpoint H1 protocol/initial state differs')
    head_indices = {int(m.group(1)) for key in ck['model']
                    if (m := re.match(r'^heads\.(\d+)\.', key))}
    need(head_indices == set(range(5)), 'Checkpoint state must have exactly five heads')


def worker_source(base_source):
    """Copy the bound EXP151 serving function; do not import sol's live files."""
    body = function_source(base_source, 'serve')
    body = replace_once(body, "model = c.build_multihead(build_model('resnet50', pretrained=False), 4)",
                        "check_checkpoint(ck, entry)\n        model = c.build_multihead(build_model('resnet50', pretrained=False), 5)")
    body = replace_once(body, 'scores = np.empty((10, 2, len(ids), 4, 12), np.float32)',
                        'scores = np.empty((10, 2, len(ids), 5, 12), np.float32)')
    body = replace_once(body, 'tuple(logits.shape) == (1, 4, 12)', 'tuple(logits.shape) == (1, 5, 12)')
    body = replace_once(body, "estimate['projected_seconds'] <= CUTOFF - RESERVE",
                        "estimate['budget_seconds'] <= CUTOFF - RESERVE")
    body = replace_once(body, "save(output / 'exp151_stage_timing.json', receipt)",
                        "np.savez(output / 'head-probabilities.npz', scores=scores)\n"
                        "    receipt.update(schema='exp154-h1-worker-v1', rule=RULE, entries=manifest['entries'],\n"
                        "                   raw_scores_sha256=sha(output / 'head-probabilities.npz'),\n"
                        "                   environment={'python': sys.version, 'torch': str(torch.__version__),\n"
                        "                                'timm': timm.__version__, 'gpu': torch.cuda.get_device_name(0)},\n"
                        "                   plane_flip='coronal/axial horizontal; sagittal unchanged')\n"
                        "    save(output / 'exp154_h1_stage_timing.json', receipt)")
    header = ('from __future__ import annotations\nimport os, sys, time, hashlib, json\nfrom pathlib import Path\n'
              'from exp154_b3_h1_release import (need, sha, save, projection, reduce_heads, check_checkpoint, '
              'HEADS, CUTOFF, RESERVE, RULE)\n\n')
    code = header + body + '\n'
    compile(code, 'exp154_h1_worker.py', 'exec')
    return code


def build_sources(root=ROOT):
    root = Path(root)
    for name, digest in SOURCE_BINDINGS.items():
        need(sha(root / name) == digest, 'Frozen source SHA differs: ' + name)
    import exp140_aug1_release as e140
    import exp151_bundle_release as e151
    nb = read(root / BASE_NOTEBOOK)
    _, payload = e140.old_payload_literal(nb['cells'][-1]['source'])
    # The proven B3 notebook contains its already-bound serving payload. No
    # live Gold58 result/old checkpoint/label dependencies enter this package.
    files = {n: e133.e128.unpack(item['z']).encode() for n, item in payload['files'].items()}
    for name, data in files.items():
        need(hashlib.sha256(data).hexdigest() == payload['files'][name]['sha256'], 'Frozen payload SHA differs')
    files.update({n: (root / 'tools' / n).read_bytes() for n in
                  ('exp133_oai_b3.py', 'exp128_oai_light.py', 'exp154_b3_h1_release.py')})
    files['exp151_components.py'] = e151.components_source().encode()
    files['exp154_h1_worker.py'] = worker_source((root / 'tools/exp151_bundle_release.py').read_text(encoding='utf-8')).encode()
    for name, data in files.items():
        need(Path(name).name == name, 'Unsafe source name')
        compile(data, name, 'exec')
    return files, nb


def safe_member(name):
    p = PurePosixPath(name)
    need(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name
         and ':' not in name and str(p) == name, 'Unsafe archive member: ' + name)


def pack_archive(path, members):
    path = Path(path)
    need(path.name.endswith('.zip.bin') and not path.exists(), 'Fresh .zip.bin required')
    for name in members:
        safe_member(name)
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, value in members.items():
            if isinstance(value, bytes):
                z.writestr(name, value)
            else:
                z.write(value, name)
    return sha(path)


def verify_archive(path, entries, extract_to=None):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == set(entries), 'Archive inventory/duplicate entries differ')
        for name in names:
            safe_member(name)
            item = entries[name]
            need(z.getinfo(name).file_size == item['bytes'], 'Archive member size differs')
            h = hashlib.sha256()
            dst = None
            if extract_to is not None:
                target = Path(extract_to) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                dst = target.open('xb')
            try:
                with z.open(name) as src:
                    for block in iter(lambda: src.read(8 << 20), b''):
                        h.update(block)
                        if dst is not None:
                            dst.write(block)
            finally:
                if dst is not None:
                    dst.close()
            need(h.hexdigest() == item['sha256'], 'Archive member SHA differs: ' + name)


def extract(dataset, work, manifest):
    need(not Path(work).exists(), 'Fresh extraction directory required')
    Path(work).mkdir(parents=True)
    for filename, key in (('code.zip.bin', 'code'), ('checkpoints.zip.bin', 'checkpoints')):
        path = Path(dataset) / filename
        need(sha(path) == manifest['archives'][filename], 'Archive SHA differs')
        verify_archive(path, manifest[key], work)


def notebook(base, manifest_sha):
    nb = copy.deepcopy(base)
    nb['cells'] = nb['cells'][:-1]  # only the proven B3 partner adapter is replaced
    description = ('# EXP154 B3-H1 - OAI-derived\n'
                   'Fixed per-target cohort percentile ranks: 0.8 public OAI 0.949 + '
                   '0.1 goodpjw2008 ConvNeXt reader + 0.1 H1 (five label-source heads, '
                   '5 folds x seeds 42/43, K36, EXP151 plane-aware flip and reduction). '
                   'One H1 decode/study is shared by all ten models and both views. '
                   'OAI/ConvNeXt retain their proven native preprocessing. '
                   'OAI-line candidacy approved by Taeyang 2026-10-09; each submission still requires his word. '
                   'No fallback and no weight tuning.\n\n' + e133.e128.credit().replace(
                       '# EXP128: public OAI-trained checkpoint and light blends', '## Source attribution').replace('in B1.', 'in EXP154.'))
    nb['cells'][1] = e133.e128.cell(description, 'markdown')
    # Bootstrap from hash-bound code before OAI. Duplicates and traversal are
    # rejected independently of the extracted implementation.
    bootstrap = '''import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pathlib, hashlib, json, zipfile, shutil, sys
def _exp154_sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()
_exp154_roots = [pathlib.Path('/kaggle/input/knee-exp154-b3-h1-assets'), pathlib.Path('/kaggle/input/datasets/taeyangg4/knee-exp154-b3-h1-assets')]
_exp154_roots = [p for p in _exp154_roots if (p / 'release-manifest.json').is_file()]
assert len(_exp154_roots) == 1
_exp154_dataset = _exp154_roots[0]
assert _exp154_sha(_exp154_dataset / 'release-manifest.json') == EXPECTED_MANIFEST_SHA
_exp154_manifest = json.loads((_exp154_dataset / 'release-manifest.json').read_text())
_exp154_work = pathlib.Path('/kaggle/working/exp154-assets')
_exp154_work.mkdir(exist_ok=False)
for _archive, _entries in [('code.zip.bin', _exp154_manifest['code']), ('checkpoints.zip.bin', _exp154_manifest['checkpoints'])]:
    assert _exp154_sha(_exp154_dataset / _archive) == _exp154_manifest['archives'][_archive]
    with zipfile.ZipFile(_exp154_dataset / _archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and set(names) == set(_entries)
        for name in names:
            p = pathlib.PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\\\' not in name and str(p) == name
            target = _exp154_work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert target.stat().st_size == _entries[name]['bytes'] and _exp154_sha(target) == _entries[name]['sha256']
sys.path.insert(0, str(_exp154_work / 'code'))
import exp154_b3_h1_release as _exp154
_exp154.save(_exp154_work / 'release-manifest.json', _exp154_manifest)
'''
    bootstrap = 'EXPECTED_MANIFEST_SHA = ' + repr(manifest_sha) + '\n' + bootstrap
    nb['cells'].insert(1, e133.e128.cell(bootstrap))
    tail = ('_exp154_receipt = _exp154.run_after_oai(_exp154_work, _exp154_manifest, '
            '_EXP133_START, manifest_sha=EXPECTED_MANIFEST_SHA)\n'
            'print("EXP154 B3-H1", _exp154_receipt["status"], _exp154_receipt["final_sha256"])\n')
    nb['cells'].append(e133.e128.cell(tail))
    need(nb['cells'][3:-1] == base['cells'][2:-1], 'Frozen public OAI cells changed')
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'EXP154 cell {i}', 'exec')
    return nb


def stage(registration=REGISTRATION, out=OUT, root=ROOT, list_path=LIST):
    out, root = Path(out), Path(root)
    need(not out.exists(), 'Choose a fresh stage directory')
    reg = registered(registration, root, list_path)
    codes, base = build_sources(root)
    entries = reg['entries']
    dataset, kernel = out / 'dataset', out / 'kernel'
    dataset.mkdir(parents=True)
    kernel.mkdir()
    code = {'code/' + n: b for n, b in codes.items()}
    checks = {e['path']: root / e['source'] for e in entries}
    manifest = {'schema': 'exp154-b3-h1-release-v1', 'version': VERSION, 'dataset': DATASET, 'kernel': KERNEL,
                'heads': list(HEADS), 'weights': WEIGHTS, 'entries': entries,
                'reader_checkpoints': CNX_CHECKPOINTS, 'reader_sources': CNX_SOURCES,
                'registration_sha256': sha(registration), 'registered_list_sha256': sha(list_path),
                'checkpoints': {e['path']: {k: e[k] for k in ('sha256', 'bytes')} for e in entries},
                'code': {n: {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)} for n, b in code.items()},
                'source_bindings': SOURCE_BINDINGS, 'runtime_projection_1300': projection(),
                'oai_derived': True, 'submission_authorized': False}
    manifest['archives'] = {'code.zip.bin': pack_archive(dataset / 'code.zip.bin', code),
                            'checkpoints.zip.bin': pack_archive(dataset / 'checkpoints.zip.bin', checks)}
    # Verify bytes INSIDE archives, catching replacement/races during copying.
    for name, key in (('code.zip.bin', 'code'), ('checkpoints.zip.bin', 'checkpoints')):
        verify_archive(dataset / name, manifest[key])
    save(dataset / 'release-manifest.json', manifest)
    save(dataset / 'dataset-metadata.json', {'title': DATASET.split('/')[-1], 'id': DATASET,
         'isPrivate': True, 'licenses': [{'name': 'other'}],
         'description': 'OAI-derived EXP154 B3-H1 candidate: frozen B3 serving code and own H1 five-head checkpoints. '
                        'No raw OAI data/reports. Taeyang approved OAI-line candidates 2026-10-09; submission only on his word.'})
    meta = e133.e128.metadata('blend_b1')
    meta.update(id=KERNEL, title=KERNEL.split('/')[-1],
                dataset_sources=[e133.e128.DATASET, e133.e128.READER_DATASET, DATASET])
    save(kernel / 'kernel-metadata.json', meta)
    save(kernel / 'release.ipynb', notebook(base, sha(dataset / 'release-manifest.json')))
    save(kernel / 'runtime-projection-1300.json', projection())
    save(out / 'stage-receipt.json', {'schema': 'exp154-stage-v1', 'status': 'STAGED_HASH_VERIFIED',
         'manifest_sha256': sha(dataset / 'release-manifest.json'), 'registration_sha256': sha(registration),
         'notebook_sha256': sha(kernel / 'release.ipynb'), 'metadata_sha256': sha(kernel / 'kernel-metadata.json'),
         'dataset_metadata_sha256': sha(dataset / 'dataset-metadata.json'),
         'network_calls': 0, 'gpu_runs': 0, 'submissions': 0})
    return {'status': 'STAGED_HASH_VERIFIED', 'out': str(out), 'checkpoints': 10, 'projection': projection()}


def validate_stage(out):
    out = Path(out)
    r = read(out / 'stage-receipt.json')
    for key, name in (('manifest_sha256', 'dataset/release-manifest.json'),
                      ('notebook_sha256', 'kernel/release.ipynb'), ('metadata_sha256', 'kernel/kernel-metadata.json'),
                      ('dataset_metadata_sha256', 'dataset/dataset-metadata.json')):
        need(sha(out / name) == r[key], 'Staged file SHA differs: ' + name)
    m = read(out / 'dataset/release-manifest.json')
    need(m['schema'] == 'exp154-b3-h1-release-v1' and m['version'] == VERSION and m['weights'] == WEIGHTS
         and m['heads'] == list(HEADS) and m['dataset'] == DATASET and m['kernel'] == KERNEL,
         'Release contract differs')
    need(m['reader_checkpoints'] == CNX_CHECKPOINTS and m['reader_sources'] == CNX_SOURCES,
         'Frozen B3 reader binding differs')
    need([(e['seed'], e['fold'], e['tag']) for e in m['entries']] ==
         [(s, f, tag(s, f)) for s in (42, 43) for f in range(5)], 'Release checkpoint order differs')
    for filename, key in (('code.zip.bin', 'code'), ('checkpoints.zip.bin', 'checkpoints')):
        need(sha(out / 'dataset' / filename) == m['archives'][filename], 'Staged archive SHA differs')
        verify_archive(out / 'dataset' / filename, m[key])
    return m, r


def run_after_oai(work, manifest, start, working='/kaggle/working', sample=None,
                  manifest_sha=None, stages=None, clock=time.monotonic):
    """Kaggle-only orchestration in production; injectable CPU fakes in tests.

    Move the OAI file out of the submission location before any partner checks.
    A failed partner can never leave a fallback submission masquerading as H1.
    """
    work, working = Path(work), Path(working)
    final, backup = working / 'submission.csv', working / 'exp154_oai.csv'
    need(not backup.exists(), 'OAI backup exists; fresh run required')
    final.rename(backup)
    sample = Path(sample) if sample else e133.input_dir(e133.e128.COMPETITION, 'sample_submission.csv') / 'sample_submission.csv'
    ids, pure = e133.validate_csv(backup, sample)
    estimate = projection(len(ids))
    need(clock() - start <= CUTOFF - RESERVE and estimate['budget_seconds'] <= CUTOFF - RESERVE,
         'EXP154 runtime budget exceeded')
    r = {'schema': 'exp154-b3-h1-stages-v1', 'status': 'RUNNING', 'path': 'B3_H1', 'weights': WEIGHTS,
         'heads': list(HEADS), 'studies': len(ids), 'manifest_sha256': manifest_sha,
         'public_source_sha256': e133.e128.SOURCE_SHA, 'oai_weight_sha256': e133.e128.WEIGHT_SHA,
         'oai_sha256': sha(backup), 'oai_elapsed_seconds': clock() - start, 'stages': {}, 'projection': estimate}
    stages = stages or {'h1': h1_stage, 'cnx': reader_stage}
    try:
        partners = {}
        for name in ('h1', 'cnx'):
            begun = clock()
            rec = {'status': 'RUNNING'}
            r['stages'][name] = rec
            partners[name] = e133.matrix(stages[name](manifest, start, ids, rec, working, work)
                                         if name == 'h1' else stages[name](None, start, ids, rec, working), len(ids))
            need(clock() - start <= CUTOFF - RESERVE, 'Partner exceeded deadline')
            rec.update(status='PASS', seconds=clock() - begun)
        result = blend(pure, partners['cnx'], partners['h1'])
        # Existing EXP133 publication routine verifies temporary CSV before rename.
        r['final_sha256'] = e133.publish(working, ids, result)
        r['status'] = 'PASS'
        return r
    except BaseException as exc:
        r.update(status='FAILED', reason=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        r['elapsed_seconds'] = clock() - start
        save(working / 'exp154_stage_timing.json', r)


def h1_stage(manifest, start, ids, rec, working, work):
    e133.release_public_cuda_cache()
    comp = e133.input_dir(e133.e128.COMPETITION, 'test_series.csv')
    output = Path(working) / 'exp154_h1'
    child = ('import json, pathlib, sys; import exp154_h1_worker as w; '
             'p=pathlib.Path; w.serve(p(sys.argv[1]),p(sys.argv[2]),p(sys.argv[3]),'
             'json.loads(p(sys.argv[4]).read_text()))')
    with (Path(working) / 'exp154_h1.log').open('x', encoding='utf-8') as log:
        e133.e128.run_group([sys.executable, '-c', child, str(work), str(comp), str(output),
                            str(Path(work) / 'release-manifest.json')],
                           env=e133.child_env(Path(work) / 'code'), check=True,
                           timeout=e133.child_timeout(start), stdout=log, stderr=subprocess.STDOUT)
    worker = read(output / 'exp154_h1_stage_timing.json')
    need(worker['status'] == 'PASS' and worker['studies'] == len(ids)
         and worker['entries'] == manifest['entries'] and worker['heads'] == list(HEADS), 'H1 worker differs')
    own_ids, values = e133.e128.read_submission(output / 'submission.csv')
    need(own_ids == ids and sha(output / 'submission.csv') == worker['submission_sha256'], 'H1 CSV binding differs')
    target = Path(working) / 'exp154_h1.csv'
    with (output / 'submission.csv').open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    rec.update(csv_sha256=sha(target), worker=worker)
    return values


def reader_stage(payload, start, ids, rec, working):
    assets = e133.input_dir(e133.e128.READER_DATASET, 'infer.py')
    for name, digest in {**CNX_CHECKPOINTS, **CNX_SOURCES}.items():
        need(sha(assets / name) == digest, 'Submitted B3 reader asset differs: ' + name)
    values = e133.reader_stage(payload, start, ids, rec, working)
    need(rec['checkpoint_sha256'] == CNX_CHECKPOINTS and rec['source_sha256'] == CNX_SOURCES,
         'B3 reader changed during inference')
    return values


def validate_public3(output, sample, out=OUT):
    """CPU-only verification of downloaded Kaggle COMMIT output, not local CUDA."""
    import numpy as np
    output, out = Path(output), Path(out)
    manifest, staged = validate_stage(out)
    ids, final = e133.validate_csv(output / 'submission.csv', sample)
    need(len(ids) == 3, 'Public3 requires exactly three studies')
    r = read(output / 'exp154_stage_timing.json')
    need(r['schema'] == 'exp154-b3-h1-stages-v1' and r['status'] == 'PASS' and r['path'] == 'B3_H1'
         and r['weights'] == WEIGHTS and r['heads'] == list(HEADS) and r['studies'] == 3
         and r['manifest_sha256'] == staged['manifest_sha256']
         and r['public_source_sha256'] == e133.e128.SOURCE_SHA and r['oai_weight_sha256'] == e133.e128.WEIGHT_SHA,
         'B3-H1 receipt/manifest/weights differ')
    need(0 < r['elapsed_seconds'] <= CUTOFF - RESERVE and math.isfinite(r['elapsed_seconds']), 'Runtime deadline differs')
    need(sha(output / 'submission.csv') == r['final_sha256'], 'Final CSV SHA differs')
    need(sha(output / 'exp154_oai.csv') == r['oai_sha256'], 'OAI CSV SHA differs')
    _, pure = e133.validate_csv(output / 'exp154_oai.csv', sample)
    partners = {}
    for name, filename in (('h1', 'exp154_h1.csv'), ('cnx', 'exp133_cnx.csv')):
        rec = r['stages'][name]
        need(rec['status'] == 'PASS' and rec['csv_sha256'] == sha(output / filename), 'Partner status/SHA differs')
        own_ids, rows = e133.e128.read_submission(output / filename)
        need(set(own_ids) == set(ids), 'Partner cohort differs')
        by_id = dict(zip(own_ids, rows))
        partners[name] = [by_id[uid] for uid in ids]
    need(final == blend(pure, partners['cnx'], partners['h1']), 'Fixed blend arithmetic differs')
    need(r['stages']['cnx']['checkpoint_sha256'] == CNX_CHECKPOINTS
         and r['stages']['cnx']['source_sha256'] == CNX_SOURCES, 'B3 reader receipt differs')
    worker = read(output / 'exp154_h1/exp154_h1_stage_timing.json')
    need(worker == r['stages']['h1']['worker'] and worker['schema'] == 'exp154-h1-worker-v1'
         and worker['status'] == 'PASS' and worker['studies'] == 3 and worker['models'] == 10
         and worker['heads'] == list(HEADS) and worker['entries'] == manifest['entries']
         and worker['k_eval'] == 36 and worker['rule'] == RULE and worker['training'] == 0
         and worker['label_reads'] == 0 and worker['manifest_sha256'] == object_sha(manifest), 'H1 worker binding differs')
    need(sha(output / 'exp154_h1/submission.csv') == worker['submission_sha256'] ==
         r['stages']['h1']['csv_sha256'], 'H1 worker CSV differs')
    raw = output / 'exp154_h1/head-probabilities.npz'
    need(sha(raw) == worker['raw_scores_sha256'], 'Raw H1 scores SHA differs')
    with np.load(raw, allow_pickle=False) as z:
        need(z.files == ['scores'], 'Unexpected raw score entries')
        scores = z['scores']
    need(scores.shape == (10, 2, 3, 5, 12), 'H1 raw score shape differs')
    with zipfile.ZipFile(out / 'dataset/code.zip.bin') as z:
        component_source = z.read('code/exp151_components.py').decode()
    scope = {}
    exec(compile(component_source, '<bound rank functions>', 'exec'), scope)
    recomputed = reduce_heads(scores[:, 0], scores[:, 1], scope['rank_mean'], scope['fold_ranks'])
    need(np.array_equal(np.asarray(partners['h1']), recomputed), 'H1 head/view/fold arithmetic differs')
    return {'status': 'PASS_B3_H1_PUBLIC3', 'studies': 3, 'targets': 12,
            'submission_sha256': sha(output / 'submission.csv'), 'manifest_sha256': staged['manifest_sha256'],
            'projection': projection(), 'submission_authorized': False,
            'limitation': 'Integration/arithmetic verified; no hidden score, full runtime or cross-GPU numerical parity inferred.'}


def push_plan(out=OUT):
    out = Path(out).resolve()
    validate_stage(out)
    kaggle = ROOT / '.venv/Scripts/kaggle.exe'
    py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
    # Text only. Dataset creation must occur INSIDE the dataset directory.
    return {'status': 'PLAN_ONLY_NO_NETWORK', 'kernel': KERNEL, 'dataset': DATASET, 'commands': [
        f'Set-Location -LiteralPath "{out / "dataset"}"',
        f'& "{kaggle}" datasets create -p .',
        '# Wait for the private dataset to be ready; check live Kaggle quota before GPU dispatch.',
        f'& "{kaggle}" datasets status {DATASET}',
        f'Set-Location -LiteralPath "{out / "kernel"}"',
        f'& "{kaggle}" kernels push -p .',
        f'& "{kaggle}" kernels status {KERNEL}',
        '# After COMPLETE, download this committed version into a fresh output directory.',
        f'& "{kaggle}" kernels output {KERNEL} -p "{out / "public3-output"}"',
        f'Set-Location -LiteralPath "{ROOT}"',
        f'& "{py}" tools/exp154_b3_h1_release.py validate-public3 --out "{out}" --output "{out / "public3-output"}"',
        '# STOP: no submission command is authorized. Taeyang must say to submit this version.'
    ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=VERSION)
    sub = parser.add_subparsers(dest='command', required=True)
    reg = sub.add_parser('register-checkpoints')
    reg.add_argument('--output', type=Path, default=REGISTRATION)
    reg.add_argument('--list', type=Path, default=LIST)
    st = sub.add_parser('stage')
    st.add_argument('--registration', type=Path, default=REGISTRATION)
    st.add_argument('--out', type=Path, default=OUT)
    st.add_argument('--list', type=Path, default=LIST)
    pp = sub.add_parser('push-plan')
    pp.add_argument('--out', type=Path, default=OUT)
    val = sub.add_parser('validate-public3')
    val.add_argument('--out', type=Path, default=OUT)
    val.add_argument('--output', type=Path, required=True)
    val.add_argument('--sample', type=Path, default=ROOT / 'storage/competition/sample_submission.csv')
    a = parser.parse_args(argv)
    if a.command == 'register-checkpoints':
        result = register_checkpoints(a.output, list_path=a.list)
    elif a.command == 'stage':
        result = stage(a.registration, a.out, list_path=a.list)
    elif a.command == 'push-plan':
        result = push_plan(a.out)
    else:
        result = validate_public3(a.output, a.sample, a.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
