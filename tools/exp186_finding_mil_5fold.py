"""Legacy artifact compatibility v2.0.0.

Changelog 2.0.0: retire source generation and fresh launches; retain historical
validation, collection and scientific readouts. Timing is measurement only.

EXP186B: CPU package/collect/screen and serial Vast S42 folds 1-4.

Changelog v1.0.1: fresh rerun after EXP186 fold-1 guard import failure;
forward inherited guard methods, package the overlay tools namespace, and
bind runtime environment checks to the validated Vast controls.
"""
from __future__ import annotations

import argparse
import copy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp179_multiscale as prior
from tools import exp183_multiscale_5fold as mixed
from tools import exp185_finding_mil as mil
from tools import exp170_convnext_t_l2 as base

VERSION = '2.0.0'
HISTORICAL_VERSION = '1.0.1'
SCHEMA = 'exp186b-finding-mil-5fold-v1'
EXP = 'artifacts/experiments/EXP186_FINDING_MIL_5FOLD'
TOOL = 'tools/exp186_finding_mil_5fold.py'
PROTOCOL = EXP + '/PROTOCOL.md'
MODEL = mil.MODEL
RUNNER = 'training/runners/exp186_finding_mil_5fold.py'
VERIFICATION = 'verification-b.md'
SOURCE_ROOT = '/workspace/exp179'
CONTROL_ROOT = '/workspace/exp183'
CONTROL_RESULTS = '/workspace/exp183-results'
REMOTE_ROOT = '/workspace/exp186b'
REMOTE_RESULTS = '/workspace/exp186b-results'
BASE_OVERLAY = '.local/exp179-overlay'
CONTROL_OVERLAY = '.local/exp183-overlay'
FOLDS = (1, 2, 3, 4)
ARM_CAP = 3600
TOTAL_CAP = 14400
PILOT_CAP = 300
OUTPUTS = EXP + '/outputs-b'
TOOLS_INIT = 'tools/__init__.py'
TOOLS_INIT_BYTES = (b'from pkgutil import extend_path\n'
                    b'__path__ = extend_path(__path__, __name__)\n')

need, read, sha, write, safe = prior.need, prior.read, prior.sha, prior.write, prior.safe




def tag(fold):
    need(fold in FOLDS, 'fold must be 1-4')
    return f'EXP186B_FINDING_MIL_S42_F{fold}'


def fold_from_tag(value):
    found = [f for f in FOLDS if tag(f) == value]
    need(len(found) == 1, 'unknown EXP186B tag')
    return found[0]


def dataset_factory(value):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL dataset_factory is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def evaluate_flip_tta(model, corpus, ids, labels, a, dev, eval_dtype, autocast):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL evaluate_flip_tta is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def build_offline_model_mil(arch, weights_path, expected_sha256):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL build_offline_model_mil is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def stamp_outputs(out, protocol):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL stamp_outputs is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def runner_source(root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL runner_source is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def queue():
    return [{'tag': tag(f), 'command': ['python', TOOL, 'run', '--fold', str(f)],
             'continue_on_failure': False} for f in FOLDS]


def check_verification(path, root=ROOT):
    note = Path(path).read_text(encoding='utf-8')
    lines = [line.strip() for line in note.splitlines() if line.strip()]
    reviewers = [line for line in lines if line.startswith('Reviewer:')]
    verdicts = [line for line in lines if line.startswith('Verdict:')]
    need(lines and lines[-1] == 'Verdict: PASS' and
         reviewers == ['Reviewer: Codex / astra'] and verdicts == ['Verdict: PASS'],
         'independent exact PASS review required')
    for name in (TOOL, MODEL, PROTOCOL):
        need(sha(Path(root) / name) in note, 'review lacks exact hash: ' + name)
    return sha(path)


def control_receipts(root=ROOT):
    root = Path(root)
    overlay = root / CONTROL_OVERLAY
    m = mixed.validate(overlay)
    result = {}
    for fold in FOLDS:
        collected = root / mixed.EXP / 'outputs' / f'fold{fold}'
        receipt = read(collected / 'collection.json')
        need(receipt['status'] == 'VERIFIED' and receipt['fold'] == fold and
             receipt['manifest_sha256'] == sha(overlay / 'manifest.json'),
             'unverified EXP183 control')
        run = collected / 'results' / mixed.tag(fold)
        hashes = mixed.verify_run(run, fold, overlay)
        done = read(collected / f'validated-fold{fold}.json')
        need(done['status'] == 'VALIDATED_SUCCESS' and done['output_sha256'] == hashes and
             done['manifest_sha256'] == receipt['manifest_sha256'],
             'EXP183 control receipt differs')
        result[str(fold)] = {'path': str(run.relative_to(root)), 'hashes': hashes,
            'environment': read(run / 'provenance.json')['environment'],
            'hostname': read(run / 'summary.json')['environment']['hostname'],
            'initial_state_sha256': read(run / 'provenance.json')['initial_state_sha256'],
            'args': read(run / 'summary.json')['args'],
            'control_manifest_sha256': receipt['manifest_sha256']}
        need(m['folds'][str(fold)]['sha256'] == sha(overlay / f'protocols/fold{fold}.json'),
             'EXP183 protocol drift')
    return result


def fold0_receipt(root=ROOT):
    root = Path(root)
    overlay = root / '.local/exp185-overlay'
    collected = root / mil.EXP / 'outputs/S42'
    c = read(collected / 'collection.json')
    need(c['status'] == 'VERIFIED' and c['seed'] == 42 and
         c['manifest_sha256'] == sha(overlay / 'manifest.json'),
         'EXP185 fold-0 collection differs')
    hashes = mil.verify_run(collected / 'results' / mil.tag(42), 42, overlay)
    done = read(collected / 'validated-S42.json')
    need(done['status'] == 'VALIDATED_SUCCESS' and done['output_sha256'] == hashes,
         'EXP185 fold-0 receipt differs')
    control = mil.control_receipt(42, root)
    provenance = read(collected / 'results' / mil.tag(42) / 'provenance.json')
    summary = read(collected / 'results' / mil.tag(42) / 'summary.json')
    return {'manifest_sha256': c['manifest_sha256'], 'hashes': hashes,
            'control_path': control['path'], 'control_hashes': control['hashes'],
            'environment': provenance['environment'],
            'hostname': summary['environment']['hostname']}


def paired_control_path(fold, manifest, root=ROOT):
    need(fold in range(5), 'unknown pooled fold')
    name = (manifest['fold0']['control_path'] if fold == 0 else
            manifest['controls'][str(fold)]['path'])
    # Historical Windows manifests used str(Path); validate the portable form.
    name = name.replace('\\', '/')
    safe(name)
    return Path(root) / name


def prepare(out, verification, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL prepare is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def validate(package, full=False):
    package = Path(package)
    m = read(package / 'manifest.json')
    need(m['schema'] == SCHEMA and m['version'] == HISTORICAL_VERSION and
         m['source_root'] == SOURCE_ROOT and m['control_root'] == CONTROL_ROOT and
         m['remote_root'] == REMOTE_ROOT and set(m['folds']) == {str(f) for f in FOLDS} and
         m['caps_seconds'] == {'arm': ARM_CAP, 'total': TOTAL_CAP, 'pilot': PILOT_CAP} and
         ARM_CAP * len(FOLDS) == TOTAL_CAP <= 14400 and
         read(package / 'queue.json') == queue(), 'manifest/queue contract differs')
    need(check_verification(package / VERIFICATION, package) == m['verification_sha256'] and
         sha(package / PROTOCOL) == m['protocol_sha256'], 'review/protocol binding differs')
    required = base.REQUIRED_MODULES | {TOOL, mixed.TOOL, mil.TOOL, MODEL,
        RUNNER, PROTOCOL, VERIFICATION,
        prior.TOOL, prior.LOADER, TOOLS_INIT, base.LABEL, base.LABEL_PROVENANCE,
        base.SOURCE_WEIGHT, 'config/workspace.toml', 'queue.json'} | {
        f'protocols/fold{f}.json' for f in FOLDS}
    need(required <= set(m['files']), 'missing transitive runtime files')
    need(m['files'][TOOLS_INIT]['origin'] == 'overlay' and
         m['files'][TOOLS_INIT]['sha256'] == hashlib.sha256(TOOLS_INIT_BYTES).hexdigest(),
         'overlay tools namespace differs')
    need(all(m['controls'][str(f)]['environment'] == m['fold0']['environment'] and
             m['controls'][str(f)]['hostname'] == m['fold0']['hostname']
             for f in FOLDS), 'Vast controls have different environments')
    for name, entry in m['files'].items():
        safe(name)
        need(entry['origin'] in ('exp179', 'overlay'), 'unknown file origin')
        if full or entry['origin'] == 'overlay':
            path = package / name
            need(path.is_file() and not path.is_symlink() and
                 path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                 'package drift: ' + name)
    need(m['files'][MODEL]['sha256'] == sha(package / MODEL) and
         m['files'][base.LABEL]['sha256'] == base.LABEL_SHA and
         m['files'][base.SOURCE_WEIGHT]['sha256'] == base.WEIGHT_SHA,
         'model/label/weight drift')
    for fold in FOLDS:
        record = m['folds'][str(fold)]
        q = read(package / record['protocol'])
        info = q['folds'][str(fold)]
        ids = read(package / record['roster'])
        control = m['controls'][str(fold)]
        need(sha(package / record['protocol']) == record['sha256'] and
             q['schema'] == SCHEMA and q['fold'] == fold and q['seed'] == 42 and
             info['tag'] == record['tag'] == tag(fold) and
             q['tags']['prefix'] == tag(fold) and
             q['tool_sha256'] == m['files'][TOOL]['sha256'] and
             q['runner_sha256'] == m['files'][RUNNER]['sha256'] and
             q['protocol_sha256'] == m['protocol_sha256'] and
             q['initial_state_sha256'] == control['initial_state_sha256'] and
             q['environment'] == control['environment'] and
             q['exp179']['crop_policy'] == 'mixed' and
             q['exp185']['model_sha256'] == m['files'][MODEL]['sha256'] and
             q['exp185']['control'] == mixed.tag(fold) and
             q['exp186'] == {'version': HISTORICAL_VERSION, 'fold': fold,
                 'control': mixed.tag(fold), 'verification_sha256': m['verification_sha256'],
                 'source_manifest_sha256': m['source_manifest_sha256']} and
             info['ordered_ids']['path'] == record['roster'] and
             info['ordered_ids']['sha256'] == sha(package / record['roster']) and
             info['ordered_ids']['train_sha256'] == base.uid_hash(ids['train']) and
             info['ordered_ids']['val_sha256'] == base.uid_hash(ids['val']) and
             len(ids['train']) == record['train'] and len(ids['val']) == record['val'] and
             not set(ids['train']) & set(ids['val']) and
             info['steps'] == record['steps'] and
             all(control['args'].get(k) == v for k, v in q['common_args'].items()) and
             q['paths']['labels'] == q['paths']['eval_labels'] == base.LABEL and
             q['label_set']['sha256'] == base.LABEL_SHA and
             q['extra_eval']['k_eval'] == 36 and
             q['common_args']['k'] == 12 and q['common_args']['res'] == 224 and
             q['common_args']['epochs'] == q['common_args']['fixed_epoch'] == 20,
             'fold scientific contract differs: ' + str(fold))
    if full:
        need(str(package.resolve()) == REMOTE_ROOT and
             sha(package / 'source-manifest.json') == m['source_manifest_sha256'],
             'remote source/root differs')
    return m


def vast_stage(overlay, source=SOURCE_ROOT, destination=REMOTE_ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL vast_stage is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def validate_arguments(a, root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL validate_arguments is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def runner_args(fold, pilot=False, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL runner_args is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def verify_run(run_dir, fold, package=ROOT):
    from tools.exp162_dino_folds_kaggle import _prediction_values
    package, out = Path(package), Path(run_dir)
    m = validate(package)
    q = read(package / m['folds'][str(fold)]['protocol'])
    info = q['folds'][str(fold)]
    need(out.name == tag(fold), 'wrong output tag')
    inv = read(out / 'output-inventory.json')
    need(set(inv) == base.REQUIRED_OUTPUTS, 'incomplete output inventory')
    for name, entry in inv.items():
        safe(name)
        path = out / name
        need(path.is_file() and not path.is_symlink() and
             path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
             'output inventory differs: ' + name)
    s, p, history = (read(out / n) for n in ('summary.json', 'provenance.json', 'history.json'))
    control = m['controls'][str(fold)]
    need(s['fold'] == fold and s['train'] == info['n_train'] and
         s['val'] == s['rows'] == info['n_val'] and
         s['selected_epoch_1based'] == 20 and s['optimizer_updates'] == info['steps'] and
         s['skipped_updates'] <= info['max_fp16_skips'] and
         [row['epoch'] for row in history] == list(range(20)) and
         s['gold_access'] == 'disabled' and s['weak_oof_macro_auc'] is None and
         s['label_set'] == base.LABEL_SET and
         s['exp179'] == p['exp179'] == q['exp179'] and
         s['exp185'] == p['exp185'] == q['exp185'] and
         s['exp186'] == p['exp186'] == q['exp186'] and
         p['environment'] == control['environment'] and
         p['environment_matches_baseline'] is True and
         s['environment']['hostname'] == control['hostname'] and
         p['initial_state_sha256'] == q['initial_state_sha256'] and
         p['init_weights']['weights_sha256'] == base.WEIGHT_SHA and
         p['runner_sha256'] == m['files'][RUNNER]['sha256'] and
         p['guard_sha256'] == m['files'][TOOL]['sha256'] and
         p['protocol_sha256'] == m['folds'][str(fold)]['sha256'] and
         p['train_ids_ordered_sha256'] == info['ordered_ids']['train_sha256'] and
         p['eval_ids_ordered_sha256'] == info['ordered_ids']['val_sha256'] and
         p['actual_pos_weight'] == info['pos_weight'] and
         p['label_file_sha256'] == base.LABEL_SHA and
         p['label_semantic_sha256'] == q['label_set']['semantic_sha256'] and
         p['flip_tta'] == 'coronal_axial_horizontal_mean_probability',
         'incomplete or mismatched fold run')
    args = s['args']
    need(args['seed'] == 42 and args['fold'] == fold and args['tag'] == tag(fold) and
         args['mode'] == 'full' and all(args.get(k) == v for k, v in q['common_args'].items()) and
         args['labels'] == args['eval_labels'] == f'{REMOTE_ROOT}/{base.LABEL}',
         'run arguments differ')
    ids = read(package / base.roster_path(fold))['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        _prediction_values(out / name, ids)
    import torch
    ck = torch.load(out / 'model_best.pt', map_location='cpu', weights_only=False)
    keys = read(mil.inherited_path(package, q['initial_state']['keys_file']))['keys']
    need(ck.get('model_kind') == 'finding_spatial_tokens_v1' and
         ck['tag'] == tag(fold) and set(ck['model']) == set(keys),
         'checkpoint model kind/parameters differ')
    return {name: inv[name]['sha256'] for name in inv}


def remote_control(fold, m):
    run = Path(CONTROL_RESULTS) / mixed.tag(fold)
    receipt = read(Path(CONTROL_ROOT) / f'validated-fold{fold}.json')
    expected = m['controls'][str(fold)]
    need(receipt['status'] == 'VALIDATED_SUCCESS' and
         receipt['manifest_sha256'] == expected['control_manifest_sha256'] and
         receipt['output_sha256'] == expected['hashes'] and
         read(run / 'summary.json')['environment']['hostname'] == expected['hostname'] and
         read(run / 'provenance.json')['environment'] == expected['environment'],
         'remote EXP183 control differs')
    for name, digest in expected['hashes'].items():
        need(sha(run / name) == digest, 'remote EXP183 output drift: ' + name)


def stage_corpus(root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL stage_corpus is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def check_vast_environment(manifest, fold, hostname, gpu, versions):
    control = manifest['controls'][str(fold)]
    expected = control['environment']
    need(expected == manifest['fold0']['environment'] and
         hostname == control['hostname'] == manifest['fold0']['hostname'] and
         gpu == expected['gpu'] and
         set(versions) == set(expected) - {'gpu'} and
         all(versions[k] == expected[k] for k in versions),
         'runtime differs from recorded EXP183/EXP185 Vast controls')


def run(fold):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL run is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def allowed_launcher_pids(fold, parent=None, proc_root=Path('/proc')):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL allowed_launcher_pids is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def admit():
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL admit is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def export(fold, out, root=ROOT):
    root, out = Path(root), Path(out)
    need(str(root.resolve()) == REMOTE_ROOT and not out.exists(),
         'fresh exact-root export required')
    validate(root, full=True)
    hashes = verify_run(Path(REMOTE_RESULTS) / tag(fold), fold, root)
    receipt = read(root / f'validated-fold{fold}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and
         receipt['output_sha256'] == hashes and
         receipt['manifest_sha256'] == sha(root / 'manifest.json'),
         'receipt mismatch')
    with tarfile.open(out, 'x') as archive:
        archive.add(root / 'manifest.json', arcname='manifest.json')
        archive.add(root / f'validated-fold{fold}.json',
                    arcname=f'validated-fold{fold}.json')
        for name in sorted(base.REQUIRED_OUTPUTS | {'output-inventory.json'}):
            archive.add(Path(REMOTE_RESULTS) / tag(fold) / name,
                        arcname=f'results/{tag(fold)}/{name}')
    return {'status': 'EXPORTED_VALIDATED', 'path': str(out), 'sha256': sha(out)}


def collect(archive, overlay, out, fold):
    archive, overlay, out = map(Path, (archive, overlay, out))
    need(fold in FOLDS, 'unknown fold')
    m = validate(overlay)
    need(not out.exists(), 'fresh collection required')
    expected = {'manifest.json', f'validated-fold{fold}.json'} | {
        f'results/{tag(fold)}/{name}' for name in
        base.REQUIRED_OUTPUTS | {'output-inventory.json'}}
    with tarfile.open(archive, 'r') as source:
        members = source.getmembers()
        need(len(members) == len(expected) and {x.name for x in members} == expected and
             all(x.isfile() and safe(x.name) == x.name for x in members),
             'unsafe/incomplete archive')
        out.mkdir(parents=True)
        for member in members:
            dest = out / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    need(sha(out / 'manifest.json') == sha(overlay / 'manifest.json'),
         'wrong export manifest')
    hashes = verify_run(out / 'results' / tag(fold), fold, overlay)
    receipt = read(out / f'validated-fold{fold}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and receipt['fold'] == fold and
         receipt['output_sha256'] == hashes and
         receipt['manifest_sha256'] == sha(overlay / 'manifest.json') and
         0 <= receipt['elapsed_seconds'] < float('inf'),
         'invalid exported success')
    write(out / 'collection.json', {'status': 'VERIFIED', 'fold': fold,
          'manifest_sha256': sha(overlay / 'manifest.json'),
          'archive_sha256': sha(archive), 'verification_sha256': m['verification_sha256']})
    return {'status': 'VERIFIED', 'fold': fold, 'path': str(out)}


def score_arrays(fold_values, refs, draws, evaluator):
    import numpy as np
    from tools import exp169_bundle_screen as bundle
    from tools.exp162_dino_folds_kaggle import TARGETS
    need(set(fold_values) == set(range(5)), 'five folds required')
    direct_new, direct_old, new_bundles, old_bundles = [], [], [], []
    per_fold = {}
    offset = 0
    for fold in range(5):
        row = fold_values[fold]
        need(set(row) == {'candidate', 'control', 'rad42', 'rad43', 'dino'},
             'wrong fixed family inputs')
        shapes = {np.asarray(v).shape for v in row.values()}
        need(len(shapes) == 1 and next(iter(shapes))[1] == 12,
             'fold shape differs')
        length = len(row['candidate'])
        radial = (evaluator.pct_rank(row['rad42']) +
                  evaluator.pct_rank(row['rad43'])) / 2
        fixed = {'l2': radial, 'dino': evaluator.pct_rank(row['dino'])}
        new_bundles.append({**fixed, 'convnext': evaluator.pct_rank(row['candidate'])})
        old_bundles.append({**fixed, 'convnext': evaluator.pct_rank(row['control'])})
        direct_new.append(row['candidate'])
        direct_old.append(row['control'])
        segment = slice(offset, offset + length)
        per_fold[str(fold)] = {}
        for name, (y, mask, minimum) in refs.items():
            a = evaluator.auc_targets(y[segment], row['candidate'], mask[segment], minimum)
            b = evaluator.auc_targets(y[segment], row['control'], mask[segment], minimum)
            macro_delta = evaluator.macro(a)-evaluator.macro(b)
            per_fold[str(fold)][name] = {
                'macro_delta': float(macro_delta) if np.isfinite(macro_delta) else None,
                'per_target_delta': {t: float(v) if np.isfinite(v) else None
                                     for t, v in zip(TARGETS, a-b)}}
        offset += length
    need(offset == next(iter(refs.values()))[0].shape[0], 'pooled reference length differs')
    candidate, control = np.concatenate(direct_new), np.concatenate(direct_old)
    fixed_new = {name: np.concatenate([row[name] for row in new_bundles])
                 for name in ('l2', 'dino', 'convnext')}
    fixed_old = {name: np.concatenate([row[name] for row in old_bundles])
                 for name in ('l2', 'dino', 'convnext')}
    candidate_bundle, _ = bundle.bundle_of(fixed_new)
    control_bundle, _ = bundle.bundle_of(fixed_old)
    direct = base.paired_jev_consensus(candidate, control, refs, draws, evaluator)
    try:
        weak = mil.weak_target_readout(candidate, control, refs, draws, evaluator)
    except ValueError as error:
        if str(error) != 'undefined weak-target primary/bootstrap':
            raise
        weak = {'primary_weak4_mean_delta': None, 'primary_ci95': None,
                'numeric_gate': 'HOLD_REVIEW', 'status': 'HOLD_REVIEW',
                'reason': str(error)}
    replacement = base.paired_jev_consensus(candidate_bundle, control_bundle,
                                             refs, draws, evaluator)
    return {'direct': direct, 'weak_target_readout': weak,
            'fixed_bundle_replacement_diagnostic': replacement,
            'per_fold_direct': per_fold, 'status': weak['status']}


def screen(outputs, overlay, out, root=ROOT):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp149_label_sources as e149
    from tools import exp156_k_sweep as e156
    from tools import exp169_bundle_screen as bundle
    from tools import exp172_dino_base_l2 as dino_tool
    root, outputs, overlay, out = map(Path, (root, outputs, overlay, out))
    m = validate(overlay)
    need(not out.exists(), 'fresh screen required')
    need(control_receipts(root) == m['controls'] and fold0_receipt(root) == m['fold0'],
         'historical control/candidate changed')
    historical = read(root / mixed.EXP / 'screen/result.json')
    need(historical['manifest_sha256'] == m['control_manifest_sha256'] and
         historical['bootstrap'] == {'draws': 300, 'seed': 20261003,
                                     'unit': 'validation component'},
         'EXP183 fixed screen differs')
    dino_outputs = root / dino_tool.EXP / 'outputs'
    dc = read(dino_outputs / 'collection.json')
    need(dc['status'] == 'VERIFIED' and dc['completed_folds'] == list(range(5)) and
         dc['manifest_sha256'] == sha(dino_outputs / 'manifest.json'),
         'EXP172 DINO collection differs')
    dino_overlay = Path(dc['overlay'])
    dm = dino_tool.validate_manifest(dino_overlay / 'manifest.json', dino_overlay)
    canonical, fold_of = e149.fold_rosters(e149._load_manifest('student', root), root)
    ids_by_fold = {fold: [uid for uid in canonical if fold_of[uid] == fold]
                   for fold in range(5)}
    ids = [uid for fold in range(5) for uid in ids_by_fold[fold]]
    need(len(ids) == len(set(ids)) == 4343 and
         base.uid_hash(ids) == historical['ordered_ids_sha256'],
         'pooled held-out roster differs')
    raw = e156.references(ids, root)
    refs = {name: raw[key] for name, key in
            (('jev', 'jev_hybrid'), ('aquila', 'aquila_v1'),
             ('dread', 'dread'), ('consensus', 'consensus'))}
    draws = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    components_sha = sha(root / ev.COMPONENTS)
    reference_sha = {name: sha(root / path) for name, path in
                     (('jev', e156.JEV), ('aquila', e156.AQUILA),
                      ('dread', e156.DREAD),
                      *[(f'consensus_{i}', path) for i, path in enumerate(e156.CONSENSUS)])}
    coverage = {name: mask.sum(axis=0).astype(int).tolist()
                for name, (_, mask, _) in refs.items()}
    need(components_sha == historical['components_sha256'] and
         reference_sha == historical['reference_sha256'] and
         coverage == historical['mask_coverage'],
         'EXP183 fixed evaluation context differs')
    inputs, values = {}, {}
    fold0_run = root / mil.EXP / 'outputs/S42/results' / mil.tag(42)
    mil.verify_run(fold0_run, 42, root / '.local/exp185-overlay')
    for fold in range(5):
        fid = ids_by_fold[fold]
        if fold:
            need(set(fid) == set(read(overlay / base.roster_path(fold))['val']),
                 'fold roster membership differs')
            collected = outputs / f'fold{fold}'
            receipt = read(collected / 'collection.json')
            need(receipt['status'] == 'VERIFIED' and receipt['fold'] == fold and
                 receipt['manifest_sha256'] == sha(overlay / 'manifest.json'),
                 'EXP186B collection differs')
            candidate_run = collected / 'results' / tag(fold)
            hashes = verify_run(candidate_run, fold, overlay)
            need(read(collected / f'validated-fold{fold}.json')['output_sha256'] == hashes,
                 'candidate receipt differs')
        else:
            candidate_run = fold0_run
            need(mil.control_receipt(42, root)['hashes'] == m['fold0']['control_hashes'],
                 'EXP179 M42 fold-0 control changed')
        control_run = paired_control_path(fold, m, root)
        dino_run = dino_outputs / 'artifacts/experiments' / dino_tool.tag(fold)
        dq = dm['fold_protocols'][str(fold)]
        dino_tool._verify_fold(dino_run, fold, read(dino_overlay / dq['path']),
                               dq['sha256'], dino_overlay)
        fixed = bundle.paths(root, fold)
        paths = {'candidate': candidate_run / 'oof_k36_flip_tta.csv',
                 'control': control_run / 'oof_k36_flip_tta.csv',
                 'rad42': fixed['l2_s42'], 'rad43': fixed['l2_s43'],
                 'dino': dino_run / 'oof_best.csv'}
        for name, path in paths.items():
            bundle._verify_inventory(path)
            digest = sha(path)
            if name in ('rad42', 'rad43', 'dino'):
                need(digest == historical['input_files'][f'{name}_f{fold}']['sha256'],
                     'fixed bundle input drift')
            if name == 'control':
                need(digest == historical['input_files'][f'mixed_f{fold}']['sha256'],
                     'EXP183 mixed control prediction drift')
            inputs[f'{name}_f{fold}'] = {'path': str(path), 'sha256': digest}
        values[fold] = {name: bundle.load_oof(path, fid) for name, path in paths.items()}
    result = score_arrays(values, refs, draws, ev)
    result.update(schema=SCHEMA + '-screen', version=VERSION, folds=list(range(5)),
         studies=len(ids), seed=42, controls=['EXP179_M42_F0'] +
         [mixed.tag(f) for f in FOLDS],
         bootstrap={'draws': len(draws), 'seed': 20261003,
                    'unit': 'validation component'},
         input_files=inputs, manifest_sha256=sha(overlay / 'manifest.json'),
         verification_sha256=m['verification_sha256'],
         ordered_ids_sha256=base.uid_hash(ids),
         components_sha256=components_sha,
         reference_sha256=reference_sha, mask_coverage=coverage,
         limitations=['exposed five-fold OOF; no private-LB estimate',
                      'report-derived Synovitis ruler has LOW validity'])
    mil.attach_holistic_readout(result, {'name': 'EXP186B historical pooled screen',
        'family': 'convnext', 'label_change': False, 'fold_rank': True,
        'seeds': {'42': {'arm': [inputs[f'candidate_f{f}'] for f in range(5)],
                         'control': [inputs[f'control_f{f}'] for f in range(5)]}},
        'partners': {'radial': {'blend': [
            {'files': [inputs[f'rad42_f{f}'] for f in range(5)], 'weight': .5},
            {'files': [inputs[f'rad43_f{f}'] for f in range(5)], 'weight': .5}], 'rank': True},
            'dino': [inputs[f'dino_f{f}'] for f in range(5)]}}, root)
    write(out / 'result.json', result)
    return {'status': result['status'], 'path': str(out / 'result.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', required=True); p.add_argument('--verification', required=True)
    p = sub.add_parser('validate'); p.add_argument('--package', required=True); p.add_argument('--full', action='store_true')
    p = sub.add_parser('vast-stage'); p.add_argument('--overlay', required=True)
    sub.add_parser('admit')
    p = sub.add_parser('run'); p.add_argument('--fold', type=int, choices=FOLDS, required=True)
    p = sub.add_parser('export'); p.add_argument('--fold', type=int, choices=FOLDS, required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('collect'); p.add_argument('--fold', type=int, choices=FOLDS, required=True); p.add_argument('--archive', required=True); p.add_argument('--overlay', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('screen'); p.add_argument('--outputs', default=OUTPUTS); p.add_argument('--overlay', required=True); p.add_argument('--out', default=EXP + '/screen-b')
    args = parser.parse_args(argv)
    if args.command == 'prepare': result = prepare(args.out, args.verification)
    elif args.command == 'validate': result = {'status': 'VERIFIED_PACKAGE', 'version': VERSION} if validate(args.package, args.full) else None
    elif args.command == 'vast-stage': result = vast_stage(args.overlay)
    elif args.command == 'admit': result = admit()
    elif args.command == 'run': result = run(args.fold)
    elif args.command == 'export': result = export(args.fold, args.out)
    elif args.command == 'collect': result = collect(args.archive, args.overlay, args.out, args.fold)
    else: result = screen(args.outputs, args.overlay, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
