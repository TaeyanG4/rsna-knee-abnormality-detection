"""Legacy artifact compatibility v2.0.0.

Changelog 2.0.0: retire source generation and fresh launches; retain historical
validation, collection and scientific readouts. Timing is measurement only.

EXP191: registered report-silence soft priors, paired EXP185 fold-0 run. Version 1.0.1.

Changelog v1.0.1 (B, 2026-10-11 12:5x): `run` compared the Vast runtime with the REGISTRATION environment
of protocols/S42.json (local 4070, cv2 5.0.0) and refused the 3090 (cv2 4.10.0) before training; it now
compares with VAST_ENVIRONMENT, the runtime recorded by the reused EXP185 controls on this Vast host.

All packaging and label commands are CPU only. ``run`` is an explicit Vast GPU
command and is never called by preparation, staging, admission, or screening.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp185_finding_mil as mil
from tools import exp182_synovitis_fill as fill
from tools import exp170_convnext_t_l2 as base

VERSION = '2.0.0'
HISTORICAL_VERSION = '1.0.1'
# Runtime of the reused EXP185 S42/S43 controls on the same Vast 3090: EXP185 outputs manifest
# (sha256 f121fc26…, = controls manifest_sha256) controls/<seed>/provenance/environment.
VAST_ENVIRONMENT = {'torch': '2.5.1+cu121', 'cuda': '12.1', 'cudnn': 90100, 'gpu': 'NVIDIA GeForce RTX 3090',
                    'timm': '1.0.19', 'numpy': '2.0.2', 'pydicom': '3.0.2', 'cv2': '4.10.0'}
SCHEMA = 'exp191-softprior-v1'
EXP = 'artifacts/experiments/EXP191_SOFTPRIOR'
PROTOCOL = EXP + '/PROTOCOL.md'
TOOL = 'tools/exp191_softprior.py'
RUNNER = 'training/runners/exp191_softprior.py'
INIT = 'tools/__init__.py'
FILL_TOOL = 'tools/exp182_synovitis_fill.py'
MODEL = mil.MODEL
SOURCE_OVERLAY = mil.BASE_OVERLAY
MIL_OVERLAY = '.local/exp185-overlay'
SOURCE_ROOT = mil.SOURCE_ROOT
REMOTE_ROOT = '/workspace/exp191'
REMOTE_RESULTS = '/workspace/exp191-results'
SEEDS = (42, 43)
ARMS = ('CONTROL', 'P', 'PA')
TARGETS = fill.TARGETS
TARGETS_CHANGED = ('Synovitis', 'PF OA')
BASE_RATES = {'Synovitis': 0.3148, 'PF OA': 0.2000}
S6 = '.local/idea-scout/20261011-1030/s6-small/state_spec.json'
ASTRA = '.local/astra-improve/decisions/hb102-softprior-astra.md'
L2 = fill.L2
AQUILA = 'storage/labels/derived_constellation/aquila_v1.csv'
MARKER = fill.MARKER
RELABEL = fill.RELABEL
LABEL_DIR = 'storage/labels/derived_constellation'
LABELS = {'CONTROL': L2,
          'P': LABEL_DIR + '/l2_exp191_softprior_p.csv',
          'PA': LABEL_DIR + '/l2_exp191_softprior_pa.csv'}
ARM_CAP = 5400
PILOT_CAP = 300
OUTPUTS = EXP + '/outputs'
need, read, sha, write, safe = mil.need, mil.read, mil.sha, mil.write, mil.safe


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    need(header == [fill.UID, *TARGETS], 'label columns/order differ')
    need(all(len(row) == len(header) for row in rows), 'malformed label row')
    need(len({row[0] for row in rows}) == len(rows), 'duplicate label UID')
    return header, rows


def _text(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _lookup(path):
    header, rows = _rows(path)
    return {row[0]: {name: float(row[i]) for i, name in enumerate(header) if i}
            for row in rows}


def build_label_arrays(root=ROOT):
    """Return CSV bytes, provenance counts. Never expose study-level data."""
    root = Path(root)
    need(sha(root / L2) == fill.L2_SHA and sha(root / MARKER) == fill.MARKER_SHA and
         sha(root / RELABEL) == fill.RELABEL_SHA, 'pinned label input changed')
    aquila_sha = '2c709b9a5fa053bc791e5f88f63d980a4eb506b2191c2f7d541c87b9140b4425'
    need(sha(root / AQUILA) == aquila_sha, 'pinned aquila input changed')
    spec = read(root / S6)
    need(all(float(spec['base'][t]) == rate for t, rate in BASE_RATES.items()),
         'S6 fixed prior changed')
    header, rows = _rows(root / L2)
    need(_text(header, rows) == (root / L2).read_bytes(), 'L2 CSV dialect differs')
    marker, relabel, aquila = (_lookup(root / path) for path in (MARKER, RELABEL, AQUILA))
    needs = {arm: [] for arm in ('P', 'PA')}
    counts = {t: {'silent': 0, 'eligible': 0, 'changed_P': 0, 'changed_PA': 0,
                  'missing_marker': 0,
                  'missing_relabel': 0, 'explicit_marker': 0,
                  'explicit_relabel_guard': 0, 'missing_aquila': 0,
                  'aquila_on_silent_n': 0, 'aquila_on_silent_sum': 0.0}
              for t in TARGETS_CHANGED}
    for row in rows:
        uid = row[0]
        output = {'P': list(row), 'PA': list(row)}
        for t in TARGETS_CHANGED:
            n = counts[t]
            m = marker.get(uid, {}).get(t)
            g = relabel.get(uid, {}).get(t)
            a = aquila.get(uid, {}).get(t)
            if m is None:
                n['missing_marker'] += 1
                continue
            if m != fill.SILENT_VALUE:
                n['explicit_marker'] += 1
                continue
            n['silent'] += 1
            if a is not None:
                need(0 <= a <= 1, 'aquila probability outside [0,1]')
                n['aquila_on_silent_n'] += 1
                n['aquila_on_silent_sum'] += a
            if g is None:
                n['missing_relabel'] += 1
                continue
            if g >= fill.EXPLICIT_POS_MIN:
                n['explicit_relabel_guard'] += 1
                continue
            if a is None:
                n['missing_aquila'] += 1
                continue
            col = header.index(t)
            output['P'][col] = repr(f32(0.75 * BASE_RATES[t] + 0.25 * a))
            output['PA'][col] = repr(f32(a))
            n['eligible'] += 1
            n['changed_P'] += output['P'][col] != row[col]
            n['changed_PA'] += output['PA'][col] != row[col]
        for arm in ('P', 'PA'):
            needs[arm].append(output[arm])
    for t, n in counts.items():
        need(n['silent'] + n['explicit_marker'] + n['missing_marker'] == len(rows),
             t + ' marker accounting differs')
        need(n['eligible'] + n['missing_relabel'] + n['explicit_relabel_guard'] +
             n['missing_aquila'] == n['silent'], t + ' edit accounting differs')
        n['mean_aquila_on_silent'] = (n.pop('aquila_on_silent_sum') /
                                     n['aquila_on_silent_n'] if n['aquila_on_silent_n'] else None)
    # The full string value, column order and row order of every untouched cell survive.
    for arm in ('P', 'PA'):
        for old, new in zip(rows, needs[arm]):
            need(old[0] == new[0] and all(old[i] == new[i] for i, t in
                 enumerate(header) if t not in TARGETS_CHANGED), 'untouched cell changed')
            for t in TARGETS_CHANGED:
                col = header.index(t)
                if old[col] != new[col]:
                    need(float(new[col]) == f32(float(new[col])) and
                         math.isfinite(float(new[col])), 'edited label is not finite float32')
    return {arm: _text(header, needs[arm]) for arm in ('P', 'PA')}, counts, len(rows)


def build_labels(root=ROOT, dry_run=False):
    root = Path(root)
    need((root / PROTOCOL).is_file(), 'register protocol before building labels')
    data, counts, nrows = build_label_arrays(root)
    result = {'schema': SCHEMA + '-labels', 'rows': nrows, 'counts': counts,
              'sha256': {arm: digest_bytes(value) for arm, value in data.items()},
              'inputs': {path: sha(root / path) for path in (L2, AQUILA, MARKER, RELABEL, S6)},
              'protocol_sha256': sha(root / PROTOCOL)}
    if dry_run:
        return result
    for arm, value in data.items():
        label = root / LABELS[arm]
        provenance = label.with_suffix('.provenance.json')
        qa = label.with_suffix('.qa.json')
        need(not any(path.exists() for path in (label, provenance, qa)),
             'EXP191 labels/provenance already exist: ' + arm)
        label.parent.mkdir(parents=True, exist_ok=True)
        with label.open('xb') as stream:
            stream.write(value)
        need(sha(label) == result['sha256'][arm], 'label write hash differs')
        common = {'schema': SCHEMA + '-label-provenance', 'arm': arm,
                  'name': 'l2_exp191_softprior_' + arm.lower(), 'rows': nrows,
                  'csv_sha256': result['sha256'][arm], 'inputs': result['inputs'],
                  'protocol': PROTOCOL, 'protocol_sha256': result['protocol_sha256'],
                  'builder': TOOL, 'version': VERSION, 'gold58_included': False,
                  'oai': False, 'formula': 'P: .75*S6 base + .25*aquila; PA: aquila',
                  'silent_definition': 'marker == 0.5 and relabel < 0.59; missing values skipped',
                  'targets': list(TARGETS_CHANGED), 'counts': counts,
                  'created': time.strftime('%Y-%m-%d %H:%M:%S %z')}
        write(provenance, common)
        write(qa, {'schema': SCHEMA + '-label-qa', 'arm': arm, 'rows': nrows,
                   'counts': counts, 'csv_sha256': result['sha256'][arm],
                   'untouched_cells_byte_identical': True, 'label_rows_printed': False})
    return result


def tag(arm, seed):
    need(arm in ARMS and seed in SEEDS, 'unknown arm/seed')
    return f'EXP191_SOFTPRIOR_{arm}_S{seed}_F0'


def protocol_path(arm, seed):
    return f'protocols/{arm}-S{seed}.json'


def queue(tasks=(('P', 42), ('P', 43), ('PA', 42), ('PA', 43))):
    return [{'tag': tag(arm, seed), 'command': ['python', TOOL, 'run',
             '--arm', arm, '--seed', str(seed)], 'continue_on_failure': False}
            for arm, seed in tasks]




def dataset_factory(value):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL dataset_factory is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def evaluate_flip_tta(model, corpus, ids, labels, a, dev, eval_dtype, autocast):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL evaluate_flip_tta is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def stamp_outputs(out, protocol):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL stamp_outputs is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def runner_source(root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL runner_source is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def _entry(path, origin='overlay'):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size, 'origin': origin}


def control_reuse(seed, root=ROOT):
    """Only a fully validated, exact EXP185 S42/S43 run can be reused."""
    root = Path(root)
    overlay = root / MIL_OVERLAY
    mil.validate(overlay)
    run = root / mil.OUTPUTS / f'S{seed}' / 'results' / mil.tag(seed)
    receipt = read(root / mil.OUTPUTS / f'S{seed}' / 'collection.json')
    need(receipt['status'] == 'VERIFIED' and receipt['seed'] == seed and
         receipt['manifest_sha256'] == sha(overlay / 'manifest.json'),
         'EXP185 collection differs')
    try:
        hashes = mil.verify_run(run, seed, overlay)
    except ModuleNotFoundError as exc:
        # The local packaging venv has no NumPy/Torch. The original EXP185
        # validation receipt plus a complete byte-for-byte inventory is a
        # sufficient CPU-only reuse check; any other verification error fails.
        need(exc.name in ('numpy', 'torch'), 'unexpected verifier dependency')
        inventory = read(run / 'output-inventory.json')
        need(set(inventory) == base.REQUIRED_OUTPUTS,
             'EXP185 control inventory incomplete')
        for name, entry in inventory.items():
            safe(name)
            path = run / name
            need(path.is_file() and path.stat().st_size == entry['bytes'] and
                 sha(path) == entry['sha256'], 'EXP185 control output drift: ' + name)
        hashes = {name: entry['sha256'] for name, entry in inventory.items()}
    done = read(root / mil.OUTPUTS / f'S{seed}' / f'validated-S{seed}.json')
    need(done['status'] == 'VALIDATED_SUCCESS' and done['output_sha256'] == hashes and
         done['manifest_sha256'] == receipt['manifest_sha256'], 'EXP185 result differs')
    q = read(overlay / f'protocols/S{seed}.json')
    summary, prov = (read(run / name) for name in ('summary.json', 'provenance.json'))
    need(q['exp179']['crop_policy'] == 'mixed' and q['exp185']['model_sha256'] == sha(root / MODEL) and
         q['paths']['labels'] == q['paths']['eval_labels'] == L2 and
         q['label_set']['sha256'] == sha(root / L2) and
         q['common_args']['epochs'] == q['common_args']['fixed_epoch'] == 20 and
         q['folds']['0']['pos_weight'] == prov['actual_pos_weight'] and
         summary['args']['seed'] == seed and summary['args']['fold'] == 0 and
         all(summary['args'].get(k) == v for k, v in q['common_args'].items()) and
         prov['label_file_sha256'] == q['label_set']['sha256'] and
         prov['initial_state_sha256'] == q['initial_state_sha256'] and
         prov['runner_sha256'] == sha(overlay / mil.RUNNER) and
         prov['guard_sha256'] == sha(overlay / mil.TOOL) and
         summary['exp185'] == prov['exp185'] == q['exp185'],
         'EXP185 scientific recipe/model differs')
    return {'path': str(run.relative_to(root)), 'prediction_sha256': hashes['oof_k36_flip_tta.csv'],
            'manifest_sha256': receipt['manifest_sha256'], 'protocol_sha256': sha(overlay / f'protocols/S{seed}.json')}



@dataclass(frozen=True)
class SoftpriorHistory:
    """Explicit immutable identity for reading the two historical softprior families."""
    schema: str
    version: str
    labels: dict
    tool: str
    runner: str
    remote_root: str
    remote_results: str
    tag: object
    validate: object


def historical_config():
    return SoftpriorHistory(SCHEMA, HISTORICAL_VERSION, LABELS, TOOL, RUNNER,
                            REMOTE_ROOT, REMOTE_RESULTS, tag, validate)


def _protocol(arm, seed, package, manifest, runner_sha, tool_sha, *, history=None):
    h = history or historical_config()
    q = copy.deepcopy(read(package / f'protocols/S{seed}.json'))
    q.update(schema=h.schema, protocol_sha256=manifest['protocol_sha256'],
             runner_sha256=runner_sha, tool_sha256=tool_sha)
    q['folds']['0']['tag'] = h.tag(arm, seed)
    q['tags']['prefix'] = h.tag(arm, seed)
    q['paths']['labels'] = h.labels[arm]
    q['paths']['eval_labels'] = L2
    q['label_set'] = {'name': 'l2_exp191_softprior_' + arm.lower() if arm != 'CONTROL'
                      else 'l2_dread75_aquila25_v1', 'path': h.labels[arm],
                      'sha256': manifest['files'][h.labels[arm]]['sha256'],
                      'semantic_sha256': q['label_set']['semantic_sha256'] if arm == 'CONTROL'
                      else None}
    q['exp191'] = {'version': h.version, 'arm': arm, 'seed': seed,
                   'control': mil.tag(seed), 'source_exp185_protocol_sha256':
                       sha(package / f'protocols/S{seed}.json'),
                   'frozen_pos_weight': True,
                   'eval_label_semantic_sha256':
                       read(package / f'protocols/S{seed}.json')['label_set']['semantic_sha256'],
                   'source_manifest_sha256': manifest['source_manifest_sha256']}
    return q


def prepare(out, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL prepare is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def validate(package, full=False):
    package = Path(package)
    m = read(package / 'manifest.json')
    need(m['schema'] == SCHEMA and m['version'] == HISTORICAL_VERSION and
         m['source_root'] == SOURCE_ROOT and m['remote_root'] == REMOTE_ROOT and
         m['protocol_sha256'] == m['files'][PROTOCOL]['sha256'] and
         set(m['runs']) == {tag(row['arm'], row['seed']) for row in m['tasks']} and
         m['caps_seconds']['total'] == ARM_CAP * len(m['tasks']) and
         m['caps_seconds']['pilot'] == PILOT_CAP and
         read(package / 'queue.json') == queue([(x['arm'], x['seed']) for x in m['tasks']]),
         'EXP191 manifest differs')
    need(set(m['files']) >= {TOOL, FILL_TOOL, MODEL, RUNNER, PROTOCOL, L2, LABELS['P'],
                             LABELS['PA'], 'queue.json', INIT},
         'EXP191 transitive files missing')
    need(m['files'][INIT]['origin'] == 'overlay' and
         m['files'][INIT]['sha256'] == digest_bytes(b''),
         'EXP191 tools package import path differs')
    for name, entry in m['files'].items():
        safe(name)
        need(entry['origin'] in ('exp179', 'overlay'), 'unknown origin')
        if full or entry['origin'] == 'overlay':
            p = package / name
            need(p.is_file() and not p.is_symlink() and p.stat().st_size == entry['bytes'] and
                 sha(p) == entry['sha256'], 'package drift: ' + name)
    need(sha(package / PROTOCOL) == m['protocol_sha256'] and
         sha(package / MODEL) == m['files'][MODEL]['sha256'] and
         m['files'][MODEL]['sha256'] ==
             read(package / 'protocols/S42.json')['exp185']['model_sha256'],
         'model/protocol changed')
    for key, run in m['runs'].items():
        arm, seed = run['arm'], run['seed']
        q = read(package / run['protocol'])
        reference = read(package / f'protocols/S{seed}.json')
        fixed = ('common_args', 'extra_eval', 'initial_state', 'initial_state_sha256',
                 'initial_state_file_sha256', 'init_weights', 'seed', 'variant',
                 'environment', 'exp179', 'exp185')
        need(key == tag(arm, seed) and sha(package / run['protocol']) == run['sha256'] and
             all(q.get(x) == reference.get(x) for x in fixed) and
             q['folds']['0'] == {**reference['folds']['0'], 'tag': key} and
             q['paths'] == {**reference['paths'], 'labels': LABELS[arm],
                             'eval_labels': L2} and
             q['label_set']['sha256'] == m['files'][LABELS[arm]]['sha256'] and
             q['runner_sha256'] == m['files'][RUNNER]['sha256'] and
             q['tool_sha256'] == m['files'][TOOL]['sha256'] and
             q['protocol_sha256'] == m['protocol_sha256'] and
             q['exp191']['frozen_pos_weight'] is True,
             'fold-0 frozen recipe differs: ' + key)
    if full:
        need(str(package.resolve()) == REMOTE_ROOT and
             sha(package / 'source-manifest.json') == m['source_manifest_sha256'],
             'remote root/source differs')
    return m


def vast_stage(overlay, source=SOURCE_ROOT, destination=REMOTE_ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL vast_stage is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def _task(root, arm, seed, *, history=None):
    h = history or historical_config()
    m = h.validate(root)
    key = h.tag(arm, seed)
    need(key in m['runs'], 'arm/seed was not staged')
    q = read(Path(root) / m['runs'][key]['protocol'])
    return m, q, q['folds']['0']


def validate_arguments(a, root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL validate_arguments is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def runner_args(arm, seed, pilot=False, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL runner_args is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def verify_run(out, arm, seed, package=ROOT, *, history=None):
    h = history or historical_config()
    from tools.exp162_dino_folds_kaggle import _prediction_values
    package, out = Path(package), Path(out)
    m, q, info = _task(package, arm, seed, history=h)
    need(out.name == h.tag(arm, seed), 'wrong EXP191 output identity')
    inv = read(out / 'output-inventory.json')
    need(set(inv) == base.REQUIRED_OUTPUTS, 'incomplete output inventory')
    for name, entry in inv.items():
        safe(name)
        p = out / name
        need(p.is_file() and p.stat().st_size == entry['bytes'] and
             sha(p) == entry['sha256'], 'output inventory drift: ' + name)
    summary, prov, history = (read(out / name) for name in
                              ('summary.json', 'provenance.json', 'history.json'))
    need(summary['fold'] == 0 and summary['train'] == info['n_train'] == 3455 and
         summary['val'] == summary['rows'] == info['n_val'] == 888 and
         summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == info['steps'] == 8620 and
         summary['skipped_updates'] <= info['max_fp16_skips'] and
         [x['epoch'] for x in history] == list(range(20)) and
         summary['gold_access'] == 'disabled' and
         summary['exp191'] == prov['exp191'] == q['exp191'] and
         summary['exp185'] == prov['exp185'] == q['exp185'] and
         prov['initial_state_sha256'] == q['initial_state_sha256'] and
         prov['actual_pos_weight'] == info['pos_weight'] and
         prov['runner_sha256'] == m['files'][h.runner]['sha256'] and
         prov['guard_sha256'] == m['files'][h.tool]['sha256'] and
         prov['protocol_sha256'] == m['runs'][h.tag(arm, seed)]['sha256'] and
         prov['label_file_sha256'] == m['files'][h.labels[arm]]['sha256'] and
         (arm != 'CONTROL' or prov['label_semantic_sha256'] == q['label_set']['semantic_sha256']) and
         prov['eval_label_hash'] == q['exp191']['eval_label_semantic_sha256'] and
         prov['train_ids_ordered_sha256'] == info['ordered_ids']['train_sha256'] and
         prov['eval_ids_ordered_sha256'] == info['ordered_ids']['val_sha256'] and
         prov['flip_tta'] == 'coronal_axial_horizontal_mean_probability',
         'incomplete or mismatched EXP191 fold-0 run')
    args = summary['args']
    need(args['seed'] == seed and args['tag'] == h.tag(arm, seed) and
         all(args.get(k) == v for k, v in q['common_args'].items()) and
         args['labels'] == f'{h.remote_root}/{h.labels[arm]}' and
         args['eval_labels'] == f'{h.remote_root}/{L2}', 'run args differ')
    ids = read(mil.inherited_path(package, base.FOLD0_ROSTER))['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        _prediction_values(out / name, ids)
    import torch
    ck = torch.load(out / 'model_best.pt', map_location='cpu', weights_only=False)
    keys = read(mil.inherited_path(package, q['initial_state']['keys_file']))['keys']
    need(ck.get('model_kind') == 'finding_spatial_tokens_v1' and
         ck['tag'] == h.tag(arm, seed) and set(ck['model']) == set(keys),
         'checkpoint model differs')
    return {name: inv[name]['sha256'] for name in inv}


def admit(root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL admit is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def run(arm, seed):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL run is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def export(arm, seed, out, root=ROOT, *, history=None):
    h = history or historical_config()
    root, out = Path(root), Path(out)
    need(str(root.resolve()) == h.remote_root and not out.exists(),
         'fresh exact-root export required')
    m = h.validate(root, full=True)
    hashes = verify_run(Path(h.remote_results) / h.tag(arm, seed), arm, seed, root, history=h)
    receipt = read(root / f'validated-{h.tag(arm, seed)}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and
         receipt['output_sha256'] == hashes and
         receipt['manifest_sha256'] == sha(root / 'manifest.json'),
         'validated output receipt differs')
    with tarfile.open(out, 'x') as archive:
        archive.add(root / 'manifest.json', arcname='manifest.json')
        archive.add(root / f'validated-{h.tag(arm, seed)}.json',
                    arcname=f'validated-{h.tag(arm, seed)}.json')
        for name in sorted(base.REQUIRED_OUTPUTS | {'output-inventory.json'}):
            archive.add(Path(h.remote_results) / h.tag(arm, seed) / name,
                        arcname=f'results/{h.tag(arm, seed)}/{name}')
    return {'status': 'EXPORTED_VALIDATED', 'path': str(out), 'sha256': sha(out)}


def collect(archive, overlay, out, arm, seed, *, history=None):
    h = history or historical_config()
    archive, overlay, out = map(Path, (archive, overlay, out))
    h.validate(overlay)
    need(not out.exists(), 'fresh collection required')
    expected = {'manifest.json', f'validated-{h.tag(arm, seed)}.json'} | {
        f'results/{h.tag(arm, seed)}/{name}' for name in
        base.REQUIRED_OUTPUTS | {'output-inventory.json'}}
    with tarfile.open(archive, 'r') as source:
        members = source.getmembers()
        need(len(members) == len(expected) and {x.name for x in members} == expected and
             all(x.isfile() and safe(x.name) == x.name for x in members),
             'unsafe/incomplete export')
        out.mkdir(parents=True)
        for member in members:
            dest = out / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    need(sha(out / 'manifest.json') == sha(overlay / 'manifest.json'),
         'export manifest differs')
    hashes = verify_run(out / 'results' / h.tag(arm, seed), arm, seed, overlay, history=h)
    receipt = read(out / f'validated-{h.tag(arm, seed)}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and receipt['arm'] == arm and receipt['seed'] == seed and
         receipt['manifest_sha256'] == sha(overlay / 'manifest.json') and receipt['output_sha256'] == hashes and
         0 <= receipt['elapsed_seconds'] < float('inf'), 'exported run invalid')
    write(out / 'collection.json', {'status': 'VERIFIED', 'arm': arm, 'seed': seed,
         'manifest_sha256': sha(overlay / 'manifest.json'), 'archive_sha256': sha(archive)})
    return {'status': 'VERIFIED', 'arm': arm, 'seed': seed, 'path': str(out)}


def _binary_reference(path, ids, required_targets=()):
    """A blinded binary reference; blank cells are masked, no row data logged."""
    import numpy as np
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames is not None and fill.UID in reader.fieldnames and
             all(t in reader.fieldnames for t in required_targets),
             'binary reference header differs')
        available = set(reader.fieldnames) & set(TARGETS)
        need(available, 'binary reference has no target columns')
        entries = {}
        for row in reader:
            uid = row[fill.UID]
            need(uid not in entries, 'duplicate binary reference UID')
            entries[uid] = row
    need(set(entries) <= set(ids), 'reference contains out-of-roster UIDs')
    y = np.full((len(ids), 12), np.nan)
    clusters = []
    for i, uid in enumerate(ids):
        row = entries.get(uid)
        clusters.append((row.get('PatientID') or row.get('cluster') or '') if row else '')
        if row is None:
            continue
        for j, target in enumerate(TARGETS):
            value = (row[target] or '').strip() if target in available else ''
            if value:
                need(value in ('0', '1', '0.0', '1.0'), 'panel labels must be binary')
                y[i, j] = float(value)
    return y, clusters


def _gold58_pending(path, fold_ids):
    """Validate the separate exposed cohort without scoring fold-0 predictions."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames is not None and fill.UID in reader.fieldnames,
             'Gold58 UID header missing')
        ids = [row[fill.UID] for row in reader]
    need(len(ids) == 58 and len(set(ids)) == 58 and
         set(ids).isdisjoint(fold_ids),
         'Gold58 must contain 58 unique UIDs disjoint from fold-0 V888')
    _binary_reference(path, ids)
    empty = {target: None for target in TARGETS}
    return {'status': 'PENDING_SEPARATE_INFERENCE', 'studies': len(ids),
            'sha256': sha(path), 'control': dict(empty),
            'candidate': dict(empty), 'delta': dict(empty),
            'diagnostic_only': True,
            'required_evidence': 'Gold58 predictions for CONTROL, P and PA from authenticated seed-42/43 checkpoints, with cohort, serving and checkpoint receipts'}


def _auc_binary(y, score):
    """Binary ROC AUC with midranks for ties; None if either class is absent."""
    import numpy as np
    valid = np.isfinite(y) & np.isfinite(score)
    y, score = y[valid], score[valid]
    pos = int(np.count_nonzero(y == 1))
    neg = int(np.count_nonzero(y == 0))
    if not pos or not neg:
        return None
    order = np.argsort(score, kind='stable')
    sorted_scores, sorted_y = score[order], y[order]
    indices = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1], True])
    rank_sum = 0.0
    for first, last in zip(indices[:-1], indices[1:]):
        rank_sum += float(np.count_nonzero(sorted_y[first:last] == 1)) * (first + last + 1) / 2
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _binary_auc_targets(y, pred):
    return {target: _auc_binary(y[:, i], pred[:, i]) for i, target in enumerate(TARGETS)}


def _delta(candidate, control):
    return {target: (None if candidate[target] is None or control[target] is None
                     else float(candidate[target] - control[target])) for target in TARGETS}


def _macro(scores):
    values = [scores[t] for t in TARGETS]
    return sum(values) / len(values) if all(v is not None for v in values) else None


def _mean_jev_consensus(jev, consensus):
    return {t: (None if jev[t] is None or consensus[t] is None else
                (jev[t] + consensus[t]) / 2) for t in TARGETS}


def _panel_lock(panel, lock):
    """Require a separately frozen, independent image-label roster."""
    if not lock:
        return False
    record = read(lock)
    return (record.get('panel_sha256') == sha(panel) and
            record.get('locked_before_predictions') is True and
            record.get('independent_blinded_image_labels') is True and
            all(record.get('definition_matched', {}).get(t) is True
                for t in TARGETS_CHANGED))


def _report_ref(y, mask, minimum, pred, ev):
    import numpy as np
    scores = ev.auc_targets(y, pred, mask, minimum)
    return {target: (float(value) if np.isfinite(value) else None)
            for target, value in zip(TARGETS, scores)}


def _panel_readout(panel_y, clusters, candidate, control, per_seed, draws=1000):
    import numpy as np
    indices = [TARGETS.index(t) for t in TARGETS_CHANGED]
    coverage = {t: {'positive': int(np.nansum(panel_y[:, i] == 1)),
                    'negative': int(np.nansum(panel_y[:, i] == 0))}
                for t, i in zip(TARGETS_CHANGED, indices)}
    if any(x['positive'] < 10 or x['negative'] < 10 for x in coverage.values()):
        return {'status': 'HOLD_INADEQUATE_IMAGE_PANEL', 'coverage': coverage}
    included = np.flatnonzero(np.isfinite(panel_y[:, indices]).any(axis=1))
    if any(not clusters[i] for i in included):
        return {'status': 'HOLD_MISSING_PATIENT_CLUSTERS', 'coverage': coverage}

    def primary(ix, a, b):
        differences = []
        for j in indices:
            x, z = _auc_binary(panel_y[ix, j], a[ix, j]), _auc_binary(panel_y[ix, j], b[ix, j])
            if x is None or z is None:
                return None
            differences.append(x - z)
        return sum(differences) / len(differences)

    full = np.arange(len(panel_y))
    gain = primary(full, candidate, control)
    seed_gains = {str(seed): primary(full, *per_seed[seed]) for seed in SEEDS}
    cluster_ids = sorted({clusters[i] for i in included})
    members = {cluster: np.array([i for i in included if clusters[i] == cluster], dtype=int)
               for cluster in cluster_ids}
    rng = np.random.default_rng(20261011)
    boot = []
    for _ in range(draws):
        sampled = rng.choice(cluster_ids, len(cluster_ids), replace=True)
        ix = np.concatenate([members[cluster] for cluster in sampled])
        value = primary(ix, candidate, control)
        if value is not None:
            boot.append(value)
    if gain is None or any(v is None for v in seed_gains.values()) or len(boot) < 950:
        return {'status': 'HOLD_UNDEFINED_IMAGE_AUC', 'coverage': coverage,
                'valid_bootstrap_draws': len(boot)}
    return {'status': 'EVALUABLE', 'coverage': coverage,
             'primary_gain': gain, 'per_seed_primary_gain': seed_gains,
             'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
             'simultaneous_ci95_bonferroni': np.percentile(boot, [1.25, 98.75]).tolist(),
             'valid_bootstrap_draws': len(boot), 'cluster_count': len(cluster_ids)}


def screen(outputs, overlay, out, gold58, image_panel=None, image_panel_lock=None, root=ROOT):
    import numpy as np
    from tools import exp179_screen as helper
    from tools import exp115_vlm_eval as ev
    from tools import exp169_bundle_screen as bundle
    root, outputs, overlay, out = map(Path, (root, outputs, overlay, out))
    m = validate(overlay)
    need(not out.exists(), 'fresh screen required')
    ids, refs, draws, context = helper.load_context(root)
    need(set(refs) >= {'jev', 'consensus', 'dread', 'aquila'} and len(ids) == 888,
         'fixed fold-0 reference context differs')
    gold_pending = _gold58_pending(gold58, ids)
    control = {}
    for seed in SEEDS:
        record = m['controls'][str(seed)]
        if record.get('retrain'):
            collected = outputs / f'CONTROL-S{seed}'
            need(read(collected / 'collection.json')['status'] == 'VERIFIED',
                 'CONTROL collection missing')
            run = collected / 'results' / tag('CONTROL', seed)
            verify_run(run, 'CONTROL', seed, overlay)
            path = run / 'oof_k36_flip_tta.csv'
        else:
            current = control_reuse(seed, root)
            need(current == record, 'EXP185 control drift since preparation')
            path = root / record['path'] / 'oof_k36_flip_tta.csv'
        control[seed] = {'path': str(path), 'sha256': sha(path),
                         'pred': bundle.load_oof(path, ids)}
    baseline = np.mean(np.stack([control[s]['pred'] for s in SEEDS]), axis=0)
    report = {'schema': SCHEMA + '-screen', 'version': VERSION,
              'manifest_sha256': sha(overlay / 'manifest.json'),
              'gold58_sha256': sha(gold58), 'image_panel_sha256': sha(image_panel) if image_panel else None,
              'image_panel_lock_sha256': sha(image_panel_lock) if image_panel_lock else None,
              'context': context, 'studies': len(ids), 'seeds': list(SEEDS),
              'control_sha256': {str(s): control[s]['sha256'] for s in SEEDS},
              'arms': {}, 'limitations': [
                  'fold-0 exposed V888; not a private leaderboard estimate',
                  'JEV/consensus/Dread report silence may encode negatives',
                  'aquila is partly self-referenced in PA; report-only',
                  'Gold58 is exposed; diagnostic only']}
    report_control = {name: _report_ref(*refs[name], baseline, ev)
                      for name in ('jev', 'consensus', 'dread', 'aquila')}
    for arm in ('P', 'PA'):
        by_seed, input_hashes, prediction_paths = {}, {}, {}
        for seed in SEEDS:
            collected = outputs / f'{arm}-S{seed}'
            receipt = read(collected / 'collection.json')
            need(receipt['status'] == 'VERIFIED' and receipt['arm'] == arm and
                 receipt['seed'] == seed and
                 receipt['manifest_sha256'] == sha(overlay / 'manifest.json'),
                 'unverified EXP191 candidate')
            run = collected / 'results' / tag(arm, seed)
            verify_run(run, arm, seed, overlay)
            path = run / 'oof_k36_flip_tta.csv'
            input_hashes[str(seed)] = sha(path)
            prediction_paths[str(seed)] = {'path': str(path), 'sha256': sha(path)}
            by_seed[seed] = bundle.load_oof(path, ids)
        candidate = np.mean(np.stack([by_seed[s] for s in SEEDS]), axis=0)
        details = {}
        for name in ('jev', 'consensus', 'dread', 'aquila'):
            arm_auc = _report_ref(*refs[name], candidate, ev)
            details[name] = {'control': report_control[name], 'candidate': arm_auc,
                             'delta': _delta(arm_auc, report_control[name])}
        details['gold58'] = copy.deepcopy(gold_pending)
        mean_control = _mean_jev_consensus(report_control['jev'], report_control['consensus'])
        mean_candidate = _mean_jev_consensus(details['jev']['candidate'],
                                             details['consensus']['candidate'])
        mean_delta = _delta(mean_candidate, mean_control)
        details['mean_jev_consensus'] = {'control': mean_control,
                                         'candidate': mean_candidate,
                                         'delta': mean_delta,
                                         'macro_delta': _macro(mean_delta)}
        harm = {'macro_ge_minus_0_001': _macro(mean_delta) is not None and
                _macro(mean_delta) >= -.001,
                'other_targets_ge_minus_0_010': all(mean_delta[t] is not None and
                    mean_delta[t] >= -.010 for t in TARGETS if t not in TARGETS_CHANGED),
                'dread_macro_ge_minus_0_003': _macro(details['dread']['delta']) is not None and
                    _macro(details['dread']['delta']) >= -.003}
        if image_panel and _panel_lock(image_panel, image_panel_lock):
            panel_y, clusters = _binary_reference(image_panel, ids, TARGETS_CHANGED)
            panel = _panel_readout(panel_y, clusters, candidate, baseline,
                                   {s: (by_seed[s], control[s]['pred']) for s in SEEDS})
        else:
            panel = {'status': 'HOLD_NO_LOCKED_BLINDED_IMAGE_PANEL'}
        pass_primary = (panel['status'] == 'EVALUABLE' and panel['primary_gain'] >= .002 and
                        all(v > 0 for v in panel['per_seed_primary_gain'].values()) and
                        panel['simultaneous_ci95_bonferroni'][0] > 0)
        status = ('ADVANCE_NEEDS_IMAGE_PANEL_EVIDENCE' if pass_primary and all(harm.values())
                  else 'HOLD')
        report['arms'][arm] = {'status': status, 'primary_image_panel': panel,
                                'harm_guard': harm, 'report_reference': details,
                                'input_sha256': input_hashes,
                                'adoption': 'prohibited without fresh definition-matched image evidence'}
        mil.attach_holistic_readout(report['arms'][arm], {
            'name': f'EXP191 {arm} historical paired screen', 'family': 'convnext',
            'label_change': True, 'seeds': {str(seed): {
                'arm': prediction_paths[str(seed)],
                'control': {'path': control[seed]['path'], 'sha256': control[seed]['sha256']}}
                for seed in SEEDS}, 'partners': mil.historical_partners(root)}, root)
    report['status'] = 'B_DECISION_PENDING'
    report['launch_authorized'] = False
    out.mkdir(parents=True)
    write(out / 'result.json', report)
    return {'status': {arm: report['arms'][arm]['status'] for arm in ('P', 'PA')},
            'path': str(out / 'result.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build-labels'); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('prepare'); p.add_argument('--out', required=True)
    p = sub.add_parser('validate'); p.add_argument('--package', required=True)
    p.add_argument('--full', action='store_true')
    p = sub.add_parser('vast-stage'); p.add_argument('--overlay', required=True)
    sub.add_parser('admit')
    p = sub.add_parser('run'); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p = sub.add_parser('export'); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('collect'); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--archive', required=True); p.add_argument('--overlay', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('screen'); p.add_argument('--outputs', default=OUTPUTS)
    p.add_argument('--overlay', required=True); p.add_argument('--out', default=EXP + '/screen')
    p.add_argument('--gold58', required=True); p.add_argument('--image-panel')
    p.add_argument('--image-panel-lock')
    args = parser.parse_args(argv)
    if args.command == 'build-labels':
        result = build_labels(dry_run=args.dry_run)
    elif args.command == 'prepare':
        result = prepare(args.out)
    elif args.command == 'validate':
        result = {'status': 'VERIFIED_PACKAGE'} if validate(args.package, args.full) else None
    elif args.command == 'vast-stage':
        result = vast_stage(args.overlay)
    elif args.command == 'admit':
        result = admit()
    elif args.command == 'run':
        result = run(args.arm, args.seed)
    elif args.command == 'export':
        result = export(args.arm, args.seed, args.out)
    elif args.command == 'collect':
        result = collect(args.archive, args.overlay, args.out, args.arm, args.seed)
    else:
        result = screen(args.outputs, args.overlay, args.out, args.gold58,
                        args.image_panel, args.image_panel_lock)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
