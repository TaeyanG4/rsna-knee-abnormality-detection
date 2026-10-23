"""Legacy artifact compatibility v2.0.0.

Changelog 2.0.0: retire source generation and fresh launches; retain historical
validation, collection and scientific readouts. Timing is measurement only.

EXP185: CPU package/screen and explicitly admitted serial Vast fold-0 runs."""
from __future__ import annotations

import argparse
import copy
from functools import partial
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
from tools import exp170_convnext_t_l2 as base

VERSION = '2.0.0'
HISTORICAL_VERSION = '1.1.0'
SCHEMA = 'exp185-finding-mil-v1'
EXP = 'artifacts/experiments/EXP185_FINDING_MIL'
PROTOCOL = EXP + '/PROTOCOL.md'
TOOL = 'tools/exp185_finding_mil.py'
MODEL = 'training/src/raptor_exp185.py'
RUNNER = 'training/runners/exp185_finding_mil.py'
VERIFICATION = 'verification.md'
SOURCE_ROOT = '/workspace/exp179'
REMOTE_ROOT = '/workspace/exp185'
REMOTE_RESULTS = '/workspace/exp185-results'
BASE_OVERLAY = '.local/exp179-overlay'
SEEDS = (42, 43)
ARMS = {seed: f'S{seed}' for seed in SEEDS}
OUTPUTS = EXP + '/outputs'
TAG = 'EXP185_FINDING_MIL_F0'
PILOT_CAP = 300
ARM_CAP = 5400
TOTAL_CAP = ARM_CAP * len(SEEDS)


def tag(seed):
    need(seed in SEEDS, 'unknown seed')
    return f'{TAG}_S{seed}'


def control_path(seed):
    need(seed in SEEDS, 'unknown seed')
    return (f'artifacts/experiments/EXP179_MULTISCALE/outputs/M{seed}/'
            f'results/EXP179_M{seed}_F0')

need, read, sha, write, safe = prior.need, prior.read, prior.sha, prior.write, prior.safe


def inherited_path(package, name):
    """Resolve an EXP179 hardlink before or after remote staging."""
    local = Path(package) / name
    if local.is_file():
        return local
    sources = ([Path(SOURCE_ROOT) / name] if sys.platform.startswith('linux') else
               [ROOT / BASE_OVERLAY / name, ROOT / name])
    for source in sources:
        if source.is_file():
            return source
    raise ValueError('EXP179 inherited file unavailable: ' + name)




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
    return [{'tag': tag(seed), 'command': ['python', TOOL, 'run', '--seed', str(seed)],
             'continue_on_failure': False} for seed in SEEDS]


def check_verification(path, root=ROOT):
    note = Path(path).read_text(encoding='utf-8')
    lines = [line.strip() for line in note.splitlines() if line.strip()]
    reviewers = [line for line in lines if line.startswith('Reviewer:')]
    verdicts = [line for line in lines if line.startswith('Verdict:')]
    need(lines and lines[-1] == 'Verdict: PASS' and
         reviewers == ['Reviewer: Codex / astra'] and
         verdicts == ['Verdict: PASS'],
         'independent astra review with final exact PASS required')
    for name in (TOOL, MODEL, PROTOCOL):
        need(sha(Path(root) / name) in note, 'review does not bind ' + name)
    return sha(path)


def control_receipt(seed, root=ROOT):
    root = Path(root)
    need(seed in SEEDS, 'unknown control seed')
    outputs = root / f'artifacts/experiments/EXP179_MULTISCALE/outputs/M{seed}'
    receipt = read(outputs / 'collection.json')
    overlay = root / BASE_OVERLAY
    need(receipt['status'] == 'VERIFIED' and receipt['arm'] == f'M{seed}' and
         receipt['manifest_sha256'] == sha(overlay / 'manifest.json'),
         f'EXP179 M{seed} collection/manifest differs')
    hashes = prior.verify_run(root / control_path(seed), f'M{seed}', overlay)
    done = read(outputs / f'validated-M{seed}.json')
    need(done['status'] == 'VALIDATED_SUCCESS' and done['output_sha256'] == hashes,
         f'EXP179 M{seed} receipt differs')
    return {'path': control_path(seed), 'hashes': hashes,
            'summary': read(root / control_path(seed) / 'summary.json'),
            'provenance': read(root / control_path(seed) / 'provenance.json')}


def prepare(out, verification, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL prepare is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def validate(package, full=False):
    package = Path(package)
    m = read(package / 'manifest.json')
    need(m['schema'] == SCHEMA and m['version'] == HISTORICAL_VERSION and
         m['source_root'] == SOURCE_ROOT and m['remote_root'] == REMOTE_ROOT and
         m['caps_seconds'] == {'arm': ARM_CAP, 'total': TOTAL_CAP, 'pilot': PILOT_CAP} and
         set(m['arms']) == {str(seed) for seed in SEEDS} and
         read(package / 'queue.json') == queue(), 'manifest/queue contract differs')
    need(check_verification(package / VERIFICATION, package) == m['verification_sha256'] and
         sha(package / PROTOCOL) == m['protocol_sha256'], 'review/protocol binding differs')
    required = base.REQUIRED_MODULES | {TOOL, MODEL, RUNNER, PROTOCOL, VERIFICATION,
        prior.TOOL, prior.LOADER, base.LABEL, base.LABEL_PROVENANCE,
        base.SOURCE_WEIGHT, 'config/workspace.toml', 'queue.json'} | {
        f'protocols/S{seed}.json' for seed in SEEDS}
    need(required <= set(m['files']), 'missing transitive runtime files')
    for name, entry in m['files'].items():
        safe(name)
        need(entry['origin'] in ('exp179', 'overlay'), 'unknown file origin')
        if full or entry['origin'] == 'overlay':
            path = package / name
            need(path.is_file() and not path.is_symlink() and
                 path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                 'package drift: ' + name)
    ids = read(inherited_path(package, base.FOLD0_ROSTER))
    need(len(ids['train']) == len(set(ids['train'])) == 3455 and
         len(ids['val']) == len(set(ids['val'])) == 888 and
         not set(ids['train']) & set(ids['val']), 'roster duplication/overlap')
    for seed in SEEDS:
        arm = m['arms'][str(seed)]
        q = read(package / arm['protocol'])
        f = q['folds']['0']
        control = m['controls'][str(seed)]
        need(sha(package / arm['protocol']) == arm['sha256'] and arm['tag'] == tag(seed) and
         q['schema'] == SCHEMA and q['seed'] == seed and f['tag'] == tag(seed) and
         q['runner_sha256'] == m['files'][RUNNER]['sha256'] and
         q['tool_sha256'] == m['files'][TOOL]['sha256'] and
         q['protocol_sha256'] == m['protocol_sha256'] and
         q['initial_state_sha256'] == control['provenance']['initial_state_sha256'] and
         q['exp179']['crop_policy'] == 'mixed' and
         q['exp185']['model_sha256'] == m['files'][MODEL]['sha256'] and
         q['exp185']['verification_sha256'] == m['verification_sha256'] and
         q['exp185']['source_manifest_sha256'] == m['source_manifest_sha256'] and
         q['paths']['labels'] == q['paths']['eval_labels'] == base.LABEL and
         q['label_set']['sha256'] == m['files'][base.LABEL]['sha256'] == base.LABEL_SHA and
         m['files'][base.SOURCE_WEIGHT]['sha256'] == base.WEIGHT_SHA and
         f['ordered_ids']['sha256'] == m['files'][base.FOLD0_ROSTER]['sha256'] and
         f['ordered_ids']['train_sha256'] == base.uid_hash(ids['train']) and
         f['ordered_ids']['val_sha256'] == base.uid_hash(ids['val']) and
         f['steps'] == 8620 and
         all(control['summary']['args'].get(k) == v for k, v in q['common_args'].items()) and
         q['common_args']['k'] == 12 and q['common_args']['res'] == 224 and
         q['common_args']['epochs'] == q['common_args']['fixed_epoch'] == 20 and
         q['extra_eval']['k_eval'] == 36, 'candidate/control scientific contract differs')
    if full:
        need(str(package.resolve()) == REMOTE_ROOT and
             sha(package / 'source-manifest.json') == m['source_manifest_sha256'],
             'remote root/source manifest differs')
    return m


def vast_stage(overlay, source=SOURCE_ROOT, destination=REMOTE_ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL vast_stage is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def validate_arguments(a, root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL validate_arguments is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def runner_args(seed, pilot=False, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL runner_args is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def stage_corpus(root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL stage_corpus is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def verify_run(run_dir, seed, package=ROOT):
    from tools.exp162_dino_folds_kaggle import _prediction_values
    package, out = Path(package), Path(run_dir)
    m = validate(package)
    need(seed in SEEDS, 'unknown seed')
    arm = m['arms'][str(seed)]
    q = read(package / arm['protocol'])
    f = q['folds']['0']
    need(out.name == tag(seed), 'wrong run identity')
    inv = read(out / 'output-inventory.json')
    need(set(inv) == base.REQUIRED_OUTPUTS, 'incomplete output inventory')
    for name, entry in inv.items():
        safe(name)
        need((out / name).is_file() and sha(out / name) == entry['sha256'] and
             (out / name).stat().st_size == entry['bytes'], 'output drift: ' + name)
    s, p, history = (read(out / n) for n in ('summary.json', 'provenance.json', 'history.json'))
    need(s['fold'] == 0 and s['train'] == 3455 and s['val'] == s['rows'] == 888 and
         s['selected_epoch_1based'] == 20 and s['optimizer_updates'] == 8620 and
         s['skipped_updates'] <= f['max_fp16_skips'] and
         [v['epoch'] for v in history] == list(range(20)) and
         s['gold_access'] == 'disabled' and s['weak_oof_macro_auc'] is None and
         s['exp185'] == p['exp185'] == q['exp185'] and
         s['exp179'] == p['exp179'] == q['exp179'] and
          p['environment'] == m['controls'][str(seed)]['provenance']['environment'] and
         p['initial_state_sha256'] == q['initial_state_sha256'] and
         p['runner_sha256'] == m['files'][RUNNER]['sha256'] and
         p['guard_sha256'] == m['files'][TOOL]['sha256'] and
          p['protocol_sha256'] == arm['sha256'] and
         p['train_ids_ordered_sha256'] == f['ordered_ids']['train_sha256'] and
         p['eval_ids_ordered_sha256'] == f['ordered_ids']['val_sha256'] and
         p['actual_pos_weight'] == f['pos_weight'] and
         p['label_file_sha256'] == base.LABEL_SHA and
         p['flip_tta'] == 'coronal_axial_horizontal_mean_probability',
         'incomplete or mismatched fold-0 run')
    need(s['args']['tag'] == tag(seed) and s['args']['seed'] == seed and
         all(s['args'].get(k) == v for k, v in q['common_args'].items()) and
         s['args']['labels'] == s['args']['eval_labels'] == f'{REMOTE_ROOT}/{base.LABEL}',
         'run arguments differ')
    ids = read(inherited_path(package, base.FOLD0_ROSTER))['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        _prediction_values(out / name, ids)
    import torch
    ck = torch.load(out / 'model_best.pt', map_location='cpu', weights_only=False)
    keys = read(inherited_path(package, q['initial_state']['keys_file']))['keys']
    need(ck.get('model_kind') == 'finding_spatial_tokens_v1' and
          ck['tag'] == tag(seed) and set(ck['model']) == set(keys),
         'checkpoint model kind/parameter set differs')
    return {name: inv[name]['sha256'] for name in inv}


def allowed_launcher_pids(seed, parent=None, proc_root=Path('/proc')):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL allowed_launcher_pids is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def run(seed):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL run is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def admit():
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL admit is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def export(seed, out, root=ROOT):
    root, out = Path(root), Path(out)
    need(seed in SEEDS, 'unknown seed')
    need(str(root.resolve()) == REMOTE_ROOT and not out.exists(), 'fresh exact-root export required')
    validate(root, full=True)
    hashes = verify_run(Path(REMOTE_RESULTS) / tag(seed), seed, root)
    receipt = read(root / f'validated-S{seed}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and
         receipt['output_sha256'] == hashes and
         receipt['manifest_sha256'] == sha(root / 'manifest.json'), 'receipt mismatch')
    with tarfile.open(out, 'x') as archive:
        archive.add(root / 'manifest.json', arcname='manifest.json')
        archive.add(root / f'validated-S{seed}.json', arcname=f'validated-S{seed}.json')
        for name in sorted(base.REQUIRED_OUTPUTS | {'output-inventory.json'}):
            archive.add(Path(REMOTE_RESULTS) / tag(seed) / name,
                        arcname=f'results/{tag(seed)}/{name}')
    return {'status': 'EXPORTED_VALIDATED', 'path': str(out), 'sha256': sha(out)}


def collect(archive, overlay, out, seed):
    archive, overlay, out = map(Path, (archive, overlay, out))
    need(seed in SEEDS, 'unknown seed')
    m = validate(overlay)
    need(not out.exists(), 'fresh collection required')
    expected = {'manifest.json', f'validated-S{seed}.json'} | {
        f'results/{tag(seed)}/{name}' for name in base.REQUIRED_OUTPUTS | {'output-inventory.json'}}
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
    need(sha(out / 'manifest.json') == sha(overlay / 'manifest.json'), 'wrong export manifest')
    hashes = verify_run(out / 'results' / tag(seed), seed, overlay)
    receipt = read(out / f'validated-S{seed}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and receipt['tag'] == tag(seed) and
          receipt['seed'] == seed and
         receipt['output_sha256'] == hashes and
         receipt['manifest_sha256'] == sha(overlay / 'manifest.json') and
         0 <= receipt['elapsed_seconds'] < float('inf'), 'invalid exported success')
    write(out / 'collection.json', {'status': 'VERIFIED', 'tag': tag(seed), 'seed': seed,
          'manifest_sha256': sha(overlay / 'manifest.json'), 'archive_sha256': sha(archive),
          'verification_sha256': m['verification_sha256']})
    return {'status': 'VERIFIED', 'seed': seed, 'path': str(out)}


def weak_target_readout(candidate, control, refs, draws, evaluator):
    """Pure paired primary and harm guard; masks/draws already frozen by caller."""
    import numpy as np
    from tools.exp162_dino_folds_kaggle import TARGETS

    weak = ('Lateral Meniscus', 'PF OA', 'Lateral OA', 'Synovitis')
    other = [t for t in TARGETS if t not in weak]
    names = list(TARGETS)

    def values(ix):
        per = {}
        macros = {}
        for name in ('jev', 'consensus', 'dread', 'aquila'):
            y, mask, minimum = refs[name]
            a = evaluator.auc_targets(y[ix], candidate[ix], mask[ix], minimum)
            b = evaluator.auc_targets(y[ix], control[ix], mask[ix], minimum)
            per[name] = a - b
            macros[name] = evaluator.macro(a) - evaluator.macro(b)
        avg = (per['jev'] + per['consensus']) / 2
        primary = float(np.mean([avg[names.index(t)] for t in weak]))
        macro = float((macros['jev'] + macros['consensus']) / 2)
        return per, macros, avg, primary, macro

    per, macros, avg, primary, macro = values(np.arange(len(candidate)))
    boot = np.asarray([values(ix)[3] for ix in draws], dtype=np.float64)
    need(np.isfinite(primary) and np.isfinite(macro) and np.isfinite(boot).all(),
         'undefined weak-target primary/bootstrap')
    guard = {'macro12_mean_jev_consensus_ge_minus_0_001': macro >= -.001,
             'other_targets_ge_minus_0_010': all(np.isfinite(avg[names.index(t)]) and
                                                    avg[names.index(t)] >= -.010 for t in other),
             'dread_macro12_ge_minus_0_003': np.isfinite(macros['dread']) and
                                                 macros['dread'] >= -.003}
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    numeric = ('HOLD_REVIEW' if not all(guard.values()) else
               'ADVANCE' if primary > 0 and ci[0] > 0 else 'HOLD')
    # Report-ruler Synovitis cannot by itself qualify the candidate. Gold58
    # is exposed and report-based; an independent image-read check is needed.
    status = 'REVIEW_IMAGE_READ' if numeric == 'ADVANCE' else numeric
    return {'primary_weak4_mean_delta': primary, 'primary_ci95': ci,
            'macro12_mean_jev_consensus_delta': macro,
            'per_target_mean_jev_consensus_delta': {t: float(avg[i]) if np.isfinite(avg[i]) else None
                                                    for i, t in enumerate(TARGETS)},
            'per_reference_macro_delta': {k: float(v) if np.isfinite(v) else None
                                          for k, v in macros.items()},
            'harm_guard': guard, 'numeric_gate': numeric, 'status': status,
            'synovitis_policy': 'report ruler deltas; image-read evidence needed for decision',
            'gold58': 'report-only; no new Gold58 inference in fold-0 screen',
            'aquila': 'report-only; L2 label self-reference'}


def historical_partners(root=ROOT):
    """The pinned historical fold-0 radial/DINO bundle, with explicit seed blending."""
    from tools import exp169_bundle_screen as bundle
    root = Path(root)
    old = read(root / 'artifacts/experiments/EXP179_MULTISCALE/screen/result.json')
    fixed = bundle.paths(root, 0)
    paths = {'R42': fixed['l2_s42'], 'R43': fixed['l2_s43'], 'D172': root /
             'artifacts/experiments/EXP172_DINO_BASE_L2/outputs/artifacts/experiments/EXP172_DINO_BASE_L2_S42_F0/oof_best.csv'}
    entries = {}
    for name, path in paths.items():
        need(sha(path) == old['input_sha256'][name], 'historical partner input drift')
        entries[name] = {'path': str(path), 'sha256': sha(path)}
    return {'radial': {'blend': [{'files': entries['R42'], 'weight': .5},
                                 {'files': entries['R43'], 'weight': .5}], 'rank': True},
            'dino': entries['D172']}


def attach_holistic_readout(result, spec, root=ROOT):
    """Retain the registered numeric gate as context; B makes the holistic decision."""
    from training.arms.collected_screen import readout
    old_status = result['status']
    result['registered_readout'] = {'status': old_status,
        'policy': 'Historical registered gate retained as context; no automatic adoption.'}
    result['holistic_readout'] = readout(dict(spec, old_verdict=old_status), root=root)
    result['status'] = 'B_DECISION_PENDING'
    result['launch_authorized'] = False
    return result


def screen(outputs, overlay, out, root=ROOT):
    import numpy as np
    from tools import exp179_screen as helper
    from tools import exp115_vlm_eval as ev
    from tools import exp169_bundle_screen as bundle
    root, outputs, overlay, out = map(Path, (root, outputs, overlay, out))
    m = validate(overlay)
    need(not out.exists(), 'fresh screen required')
    paths = {}
    for seed in SEEDS:
        collected = outputs / f'S{seed}'
        receipt = read(collected / 'collection.json')
        need(receipt['status'] == 'VERIFIED' and receipt['seed'] == seed and
             receipt['manifest_sha256'] == sha(overlay / 'manifest.json'),
             f'unverified S{seed} candidate')
        hashes = verify_run(collected / 'results' / tag(seed), seed, overlay)
        need(read(collected / f'validated-S{seed}.json')['output_sha256'] == hashes,
             f'S{seed} candidate receipt differs')
        control = control_receipt(seed, root)
        need(control['hashes'] == m['controls'][str(seed)]['hashes'],
             f'M{seed} control changed after preparation')
        paths[seed] = {'candidate': collected / 'results' / tag(seed) / 'oof_k36_flip_tta.csv',
                       'control': root / control_path(seed) / 'oof_k36_flip_tta.csv'}
    ids, refs, draws, context = helper.load_context(root)
    candidates = {seed: bundle.load_oof(paths[seed]['candidate'], ids) for seed in SEEDS}
    baselines = {seed: bundle.load_oof(paths[seed]['control'], ids) for seed in SEEDS}
    candidate = np.stack([candidates[seed] for seed in SEEDS], axis=2)
    baseline = np.stack([baselines[seed] for seed in SEEDS], axis=2)
    mean_evaluator = helper._MeanSeedEvaluator(ev)
    details = base.paired_jev_consensus(candidate, baseline, refs, draws, mean_evaluator)
    weak = weak_target_readout(candidate, baseline, refs, draws, mean_evaluator)
    # Historical EXP172 admission bundle, fixed before EXP185. It is a
    # diagnostic, not an extra gate or a search over family weights.
    historical = read(root / 'artifacts/experiments/EXP179_MULTISCALE/screen/result.json')
    need(historical['context']['ordered_ids_sha256'] == context['ordered_ids_sha256'] and
         historical['context']['bootstrap_seed'] == helper.BOOTSTRAP_SEED,
         'historical bundle screen roster/draws differ')
    fixed = bundle.paths(root, 0)
    dino_path = (root / 'artifacts/experiments/EXP172_DINO_BASE_L2/outputs/'
                 'artifacts/experiments/EXP172_DINO_BASE_L2_S42_F0/oof_best.csv')
    inputs = {'R42': fixed['l2_s42'], 'R43': fixed['l2_s43'], 'D172': dino_path}
    for name, path in inputs.items():
        need(sha(path) == historical['input_sha256'][name],
             'historical fixed bundle input drift: ' + name)
    radial = (ev.pct_rank(bundle.load_oof(inputs['R42'], ids)) +
              ev.pct_rank(bundle.load_oof(inputs['R43'], ids))) / 2
    dino_rank = ev.pct_rank(bundle.load_oof(inputs['D172'], ids))
    candidate_bundles, control_bundles = {}, {}
    for seed in SEEDS:
        candidate_bundles[seed], _ = bundle.bundle_of({'l2': radial, 'dino': dino_rank,
                                                   'convnext': ev.pct_rank(candidates[seed])})
        control_bundles[seed], _ = bundle.bundle_of({'l2': radial, 'dino': dino_rank,
                                                 'convnext': ev.pct_rank(baselines[seed])})
    bundle_details = helper._paired_seed_mean(candidate_bundles, control_bundles,
                                              refs, draws, ev)
    result = {'schema': SCHEMA + '-screen', 'version': VERSION,
               'studies': len(ids), 'seeds': list(SEEDS),
               'tags': [tag(seed) for seed in SEEDS],
               'controls': [f'EXP179_M{seed}_F0' for seed in SEEDS],
              'bootstrap': {'unit': 'validation component', 'draws': len(draws),
                            'seed': helper.BOOTSTRAP_SEED},
              'status': weak['status'], 'weak_target_readout': weak,
               'all_targets_and_references': details,
               'per_seed': {str(seed): base.paired_jev_consensus(
                   candidates[seed], baselines[seed], refs, draws, ev) for seed in SEEDS},
               'fixed_bundle_replacement_diagnostic': bundle_details, 'context': context,
               'input_sha256': {f'{kind}_S{seed}': sha(paths[seed][kind])
                                for seed in SEEDS for kind in ('candidate', 'control')} | {
                                **{name: sha(path) for name, path in inputs.items()}},
              'manifest_sha256': sha(overlay / 'manifest.json'),
              'verification_sha256': m['verification_sha256'],
               'limitations': ['exposed fold-0 V888; two fixed seeds; no private-LB estimate',
                              'report-derived Synovitis ruler has LOW validity']}
    attach_holistic_readout(result, {'name': 'EXP185 historical paired screen',
        'family': 'convnext', 'label_change': False,
        'seeds': {str(seed): {'arm': {'path': str(paths[seed]['candidate']), 'sha256': sha(paths[seed]['candidate'])},
                             'control': {'path': str(paths[seed]['control']), 'sha256': sha(paths[seed]['control'])}}
                  for seed in SEEDS}, 'partners': historical_partners(root)}, root)
    write(out / 'result.json', result)
    return {'status': result['status'], 'path': str(out / 'result.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', required=True); p.add_argument('--verification', required=True)
    p = sub.add_parser('validate'); p.add_argument('--package', required=True); p.add_argument('--full', action='store_true')
    p = sub.add_parser('vast-stage'); p.add_argument('--overlay', required=True)
    sub.add_parser('admit')
    p = sub.add_parser('run'); p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p = sub.add_parser('export'); p.add_argument('--seed', type=int, choices=SEEDS, required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('collect'); p.add_argument('--seed', type=int, choices=SEEDS, required=True); p.add_argument('--archive', required=True); p.add_argument('--overlay', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('screen'); p.add_argument('--outputs', default=OUTPUTS); p.add_argument('--overlay', required=True); p.add_argument('--out', default=EXP + '/screen')
    args = parser.parse_args(argv)
    if args.command == 'prepare': result = prepare(args.out, args.verification)
    elif args.command == 'validate': result = {'status': 'VERIFIED_PACKAGE', 'version': VERSION} if validate(args.package, args.full) else None
    elif args.command == 'vast-stage': result = vast_stage(args.overlay)
    elif args.command == 'admit': result = admit()
    elif args.command == 'run': result = run(args.seed)
    elif args.command == 'export': result = export(args.seed, args.out)
    elif args.command == 'collect': result = collect(args.archive, args.overlay, args.out, args.seed)
    else: result = screen(args.outputs, args.overlay, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
