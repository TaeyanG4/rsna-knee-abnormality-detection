"""Register, validate, queue and screen EXP134 AUG1 five-fold confirmation.

CPU commands only. Training is emitted to tools/gpu_queue.py as JSON; this
module never launches a GPU job or modifies an existing queue.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp127_radr50_single as e127
from tools import exp113_radr50_local as e113
from tools import exp125_radr50_k16_local as e125
from tools import exp129_radr50_single as e129
from tools.exp127_radr50_single import (worker_init, read_rows, install_gold_blocker,
    uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV,
    canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
    DREAD, KEYS_FILE, FOLDS, EXTRA_EVAL, RUN_FILES, fold_contract,
    state_hashes, mismatched_keys, dump, train_python, within, display_path,
    _set_flag, weights_path, LOCAL_WORKERS, RADIMAGENET_SHA256, WEIGHTS_LOCAL)

SCHEMA = 'exp134-aug1-kfold-v1'
SCREEN_SCHEMA = 'exp134-aug1-kfold-screen-v1'
PROTOCOL = 'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1'
SCREEN = 'artifacts/experiments/EXP134_AUG1_KFOLD_SCREEN'
TOOL = 'tools/exp134_aug1_kfold.py'
RUNNER = 'training/runners/exp134_aug1_kfold.py'
TEST = 'tests/test_exp134_aug1_kfold.py'
NEW_CODE = e127.NEW_CODE  # runner records exact EXP127 augmentation sources
SEEDS = e127.SEEDS
LABEL_SET = 'dread'
LABEL_PATHS, REFERENCES, COMPONENTS = e127.LABEL_PATHS, e127.REFERENCES, e127.COMPONENTS
DRAWS, BOOT_SEED = e127.DRAWS, e127.BOOT_SEED
PARTNER = e129.PARTNER
load_label_csv, forbid_cuda_init = e127.load_label_csv, e127.forbid_cuda_init


def tag_for(seed, fold, mode='full'):
    if seed not in SEEDS or fold not in range(1, 5) or mode != 'full':
        raise ValueError('EXP134 registers only seeds 42/43, folds 1-4, full runs')
    return f'EXP134_AUG1_S{seed}_F{fold}'


def manifest_path(seed, root=ROOT):
    if seed not in SEEDS:
        raise ValueError('unregistered seed')
    return Path(root) / PROTOCOL / f's{seed}' / 'manifest.json'


def fixed_bindings(root):
    return {p: sha256(Path(root) / p) for p in (TOOL, RUNNER, TEST, *NEW_CODE,
        e125.TOOL, e127.e118.TOOL, *LABEL_PATHS.values(), COMPONENTS, PARTNER)}


def derive_manifest(base, f0, seed, bindings, f0_inventory, root=ROOT):
    """Copy the matched control, changing only AUG1 and registration metadata."""
    if f0['variant'] != 'AUG1' or f0['common_args'] != base['common_args']:
        raise ValueError('F0/control recipe mismatch')
    p = copy.deepcopy(base)
    folds = copy.deepcopy(base['folds'])
    for k, f in folds.items():
        if k == '0':
            folds[k] = copy.deepcopy(f0['folds']['0'])
        else:
            f['tag'] = tag_for(seed, int(k))
            for key, value in f.items():
                if key != 'tag' and value != base['folds'][k][key]:
                    raise ValueError('matched control fold differs: ' + key)
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        status='REGISTERED_NO_EXECUTION', plan_item='EXP134 delegated by B',
        variant='AUG1', variant_config=copy.deepcopy(e127.CONFIG['AUG1']),
        source_bindings={**base['source_bindings'], **bindings}, folds=folds,
        total_steps=sum(f['steps'] for f in folds.values()),
        tags={'prefix': f'EXP134_AUG1_S{seed}', 'folds': {k: f['tag'] for k, f in folds.items()}},
        local={**copy.deepcopy(base['local']), 'runner': RUNNER, 'guard': TOOL,
            'comparison': 'same-seed EXP113/EXP118 local control, matched fold',
            'queue': 'serial F1-F4 for both seeds; F0 existing EXP127 run is reused'},
        evaluation={**copy.deepcopy(base['evaluation']), 'exp134_screen': screen_spec()},
        derived_from={'control_path': e127.BASES[seed][0] + '/manifest.json',
            'control_sha256': e127.BASES[seed][1],
            'f0_protocol_path': display_path(e127.manifest_path('AUG1', seed, root), root),
            'f0_protocol_sha256': sha256(e127.manifest_path('AUG1', seed, root)),
            'f0_run_inventory': f0_inventory,
            'one_change': 'EXP127 AUG1 training-only augmentation stack; all per-fold control contracts fixed'},
        label_sets={'default': LABEL_SET, 'allowed': [LABEL_SET]})
    assert_single_change(p, base, f0)
    return p


def assert_single_change(p, base, f0):
    if p['variant'] != 'AUG1' or p['variant_config'] != e127.CONFIG['AUG1']:
        raise ValueError('registered AUG1 variant changed')
    if p['common_args'] != base['common_args'] or p['environment'] != base['environment']:
        raise ValueError('control recipe/environment changed')
    if p['initial_state_sha256'] != base['initial_state_sha256'] or p['label_set'] != base['label_set']:
        raise ValueError('control initial state/labels changed')
    if set(p['folds']) != set(base['folds']) or p['folds']['0'] != f0['folds']['0']:
        raise ValueError('F0 registration changed')
    for k in ('1', '2', '3', '4'):
        expected = {**base['folds'][k], 'tag': tag_for(p['seed'], int(k))}
        if p['folds'][k] != expected:
            raise ValueError('matched control fold contract changed: ' + k)
        if p['folds'][k]['steps'] != p['folds'][k]['scheduler_total_steps'] or p['folds'][k]['max_fp16_skips'] != p['folds'][k]['steps'] // 100:
            raise ValueError('scheduler/FP16 gate changed')
    if p['total_steps'] != sum(f['steps'] for f in p['folds'].values()):
        raise ValueError('total steps changed')


def screen_spec():
    return dict(primary='k36', diagnostic='k24', folds=[0, 1, 2, 3, 4],
        seeds=list(SEEDS), draws=DRAWS, bootstrap_seed=BOOT_SEED,
        averaging='mean of two per-seed five-fold macro AUCs; never AUC of seed-mean predictions',
        bootstrap='paired study components, shared draw indices across both arms, both seeds and all references',
        tier_a={'dread_delta_min': .004, 'dread_ci_low_strictly_above': 0., 'consensus_delta_min': 0.},
        tier_b={'dread_delta_min': .001, 'positive_draw_share_min': .90,
            'other_reference_floor': -.001, 'other_references_positive_min': 2,
            'both_seeds_positive': True, 'serving_extra_max': .10},
        partner={'anchor': PARTNER, 'weights': [.20, .15, .30], 'report_only': True},
        interpretation='Pooled five-fold two-seed matched confirmation. Partner proxy is report-only. Tier verdict is evidence for B; no automatic submission or adoption.')


def prepare(root=ROOT):
    root = Path(root)
    out = root / PROTOCOL
    if out.exists() and any(x.name != 'PROTOCOL.md' for x in out.iterdir()):
        raise SystemExit('Refusing existing registration: ' + str(out))
    bindings = fixed_bindings(root)
    manifests = {}
    for seed in SEEDS:
        base, f0 = e127.base_manifest(seed, root), e127.load_protocol('AUG1', seed, root)
        e127.check_control_assets(base, seed, root)
        f0_path = e127.manifest_path('AUG1', seed, root)
        run = root / 'artifacts/experiments' / f0['folds']['0']['tag']
        inventory = e127.verify_run(run, f0, sha256(f0_path), root)
        manifests[seed] = derive_manifest(base, f0, seed, bindings, inventory, root)
    if manifests[42]['common_args'] != manifests[43]['common_args']:
        raise ValueError('seed recipes differ')
    out.mkdir(parents=True, exist_ok=True)
    registry = dict(schema=SCHEMA, status='REGISTERED_NO_EXECUTION', screen=screen_spec(),
        source_bindings=bindings, seeds={},
        estimate={'eight_fold_gpu_minutes': [173, 213], 'basis': '8 x 1300-1600s control; AUG1 overhead unmeasured'})
    for seed, p in manifests.items():
        path = manifest_path(seed, root)
        path.parent.mkdir()
        dump(path, p)
        registry['seeds'][str(seed)] = {'path': display_path(path, root), 'sha256': sha256(path)}
    dump(out / 'manifest.json', registry)
    print(json.dumps({'status': registry['status'], 'output': str(out), 'estimate': registry['estimate']}, indent=2))
    return registry


def load_protocol(seed, root=ROOT):
    root = Path(root)
    registry = read(root / PROTOCOL / 'manifest.json')
    mp = manifest_path(seed, root)
    if registry['schema'] != SCHEMA or sha256(mp) != registry['seeds'][str(seed)]['sha256']:
        raise ValueError('EXP134 registration changed')
    p = read(mp)
    if p['schema'] != SCHEMA or p['seed'] != seed or registry['screen'] != screen_spec():
        raise ValueError('EXP134 protocol/screen changed')
    base, f0 = e127.base_manifest(seed, root), e127.load_protocol('AUG1', seed, root)
    assert_single_change(p, base, f0)
    e127.check_control_assets(base, seed, root)
    e127.check_bindings(root, p['source_bindings'])
    e127.check_bindings(root, p['derived_from']['f0_run_inventory'])
    return p


def validate_arguments(args, root):
    root = Path(root)
    p = load_protocol(args.seed, root)
    if args.variant != 'AUG1' or args.fold not in range(1, 5) or args.mode != 'full':
        raise ValueError('only registered AUG1 full folds 1-4')
    if Path(args.protocol).resolve() != manifest_path(args.seed, root).resolve() or args.tag != tag_for(args.seed, args.fold):
        raise ValueError('wrong protocol/tag/fold')
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol refuses remote relaxation')
    for key, expected in p['common_args'].items():
        got = getattr(args, key, None)
        if got != expected and not (args.validate_only and key == 'device' and got == 'cpu'):
            raise ValueError('frozen argument changed: ' + key)
    if args.validate_only and args.device != 'cpu':
        raise ValueError('validation must explicitly use --device cpu')
    if not 0 <= args.workers <= 12:
        raise ValueError('workers outside registered range')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path changed: ' + key)
    if Path(args.init_weights).resolve() != weights_path(root).resolve():
        raise ValueError('initial weights path changed')
    for key, value in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(key) != value:
            raise ValueError('registered environment required: ' + key)
    out = Path(args.output_root).resolve() if args.output_root else (root / 'artifacts/experiments').resolve()
    if out != (root / 'artifacts/experiments').resolve():
        raise ValueError('full output root changed')
    fold = p['folds'][str(args.fold)]
    ordered = read(root / fold['ordered_ids']['path'])
    return p, fold, ordered, out


def runner_argv(p, fold, root=ROOT, workers=LOCAL_WORKERS, python=None):
    if fold not in range(1, 5):
        raise ValueError('EXP134 runner folds 1-4 only')
    argv = e113.runner_argv(p, fold, 'full', root=root, python=python, workers=workers, label_set='dread')
    argv[1] = str(Path(root) / RUNNER)
    i = argv.index('--label-set')
    del argv[i:i+2]
    argv = _set_flag(argv, '--protocol', str(manifest_path(p['seed'], root)))
    argv = _set_flag(argv, '--tag', tag_for(p['seed'], fold))
    return argv + ['--variant', 'AUG1']


def validate_cli(seed=None, fold=None, root=ROOT):
    targets = [(s, k) for s in SEEDS for k in range(1, 5) if (seed is None or seed == s) and (fold is None or fold == k)]
    for s, k in targets:
        p = load_protocol(s, root)
        argv = _set_flag(runner_argv(p, k, root=root, workers=0), '--device', 'cpu') + ['--validate-only']
        result = subprocess.run(argv, cwd=root, env=e113.local_env(), check=False)
        if result.returncode:
            return result.returncode
    return 0


def queue_entries(after, root=ROOT, workers=12):
    if not 0 <= workers <= 12:
        raise ValueError('workers must be 0-12')
    entries, previous = [], after
    for seed in SEEDS:
        p = load_protocol(seed, root)
        for fold in range(1, 5):
            item = {'tag': tag_for(seed, fold), 'command': runner_argv(p, fold, root=root, workers=workers),
                    'continue_on_failure': True}
            if previous:
                item['after'] = previous
            entries.append(item)
            previous = item['tag']
    return entries


def verify_run(run, p, protocol_sha, root=ROOT):
    """EXP127 inventory/provenance gates generalized to the registered fold."""
    import numpy as np
    run = Path(run)
    fold = int(run.name.rsplit('_F', 1)[1])
    f = p['folds'][str(fold)]
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv'}
    inv = read(run / 'output-inventory.json')
    if not required.issubset(inv):
        raise ValueError('incomplete inventory')
    for name, rec in inv.items():
        if Path(name).name != name or sha256(run / name) != rec['sha256']:
            raise ValueError('inventory mismatch: ' + name)
    summary, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expected = {'fold': fold, 'seed': p['seed'], 'training_mode': 'full',
        'checkpoint_selection': 'fixed-final', 'protocol_sha256': protocol_sha,
        'label_set': 'dread', 'label_file_sha256': p['label_set']['sha256'],
        'label_semantic_sha256': p['label_set']['semantic_sha256'],
        'arch': p['common_args']['arch'], 'train_ids_ordered_sha256': f['ordered_ids']['train_sha256'],
        'eval_ids_ordered_sha256': f['ordered_ids']['val_sha256'],
        'initial_state_sha256': p['initial_state_sha256'], 'initial_state_matches_registered': True,
        'environment_matches_baseline': True, 'remote_relaxed': False,
        'input_policy': 'center3', 'norm': 'radimagenet', 'k_eval': 24, 'k_eval_extra': 36,
        'runner_sha256': p['source_bindings'][RUNNER], 'guard_sha256': p['source_bindings'][TOOL],
        'exp127_variant': 'AUG1', 'exp127_variant_config': p['variant_config'],
        'exp127_code_sha256': {n: p['source_bindings'][n] for n in NEW_CODE}}
    for key, value in expected.items():
        if prov.get(key) != value:
            raise ValueError('provenance mismatch: ' + key)
    if any(prov.get('environment', {}).get(k) != p['environment'][k] for k in ENV_KEYS):
        raise ValueError('runtime environment mismatch')
    if prov.get('init_weights', {}).get('weights_sha256') != RADIMAGENET_SHA256:
        raise ValueError('initial weights mismatch')
    if not np.array_equal(np.asarray(prov.get('actual_pos_weight'), np.float32), np.asarray(f['pos_weight'], np.float32)):
        raise ValueError('pos_weight mismatch')
    args = summary.get('args', {})
    if any(args.get(k) != v for k, v in p['common_args'].items()) or args.get('seed') != p['seed'] or args.get('variant') != 'AUG1' or args.get('tag') != f['tag']:
        raise ValueError('recorded recipe mismatch')
    if run.name != f['tag'] or summary.get('mode') != 'full' or summary.get('selected_epoch_1based') != p['common_args']['epochs']:
        raise ValueError('not registered full fixed-final run')
    if (summary.get('train'), summary.get('val'), summary.get('optimizer_updates')) != (f['n_train'], f['n_val'], f['steps']):
        raise ValueError('counts/updates mismatch')
    if not 0 <= summary.get('skipped_updates', 10**9) <= f['max_fp16_skips']:
        raise ValueError('FP16 skip gate failed')
    history = read(run / 'history.json')
    if [h.get('epoch') for h in history] != list(range(p['common_args']['epochs'])) or history != summary.get('history'):
        raise ValueError('epoch history mismatch')
    return {display_path(run / n, root): sha256(run / n) for n in required | {'output-inventory.json'}}


def verify_control_run(run, base, fold, protocol_sha, root=ROOT):
    """Full control inventory and per-fold provenance; EXP125 helper is F0-only."""
    import numpy as np
    run = Path(run)
    f = base['folds'][str(fold)]
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv'}
    inv = read(run / 'output-inventory.json')
    if not required.issubset(inv):
        raise ValueError('incomplete control inventory')
    for name, rec in inv.items():
        if Path(name).name != name or sha256(run / name) != rec['sha256']:
            raise ValueError('control inventory mismatch: ' + name)
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    guard_tool = e113.TOOL if base['seed'] == 42 else e127.e118.TOOL
    expected = {'fold': fold, 'seed': base['seed'], 'training_mode': 'full',
        'checkpoint_selection': 'fixed-final', 'protocol_sha256': protocol_sha,
        'label_set': 'dread', 'label_file_sha256': base['label_set']['sha256'],
        'label_semantic_sha256': base['label_set']['semantic_sha256'],
        'arch': base['common_args']['arch'], 'train_ids_ordered_sha256': f['ordered_ids']['train_sha256'],
        'eval_ids_ordered_sha256': f['ordered_ids']['val_sha256'],
        'initial_state_sha256': base['initial_state_sha256'], 'initial_state_matches_registered': True,
        'environment_matches_baseline': True, 'remote_relaxed': False,
        'input_policy': 'center3', 'norm': 'radimagenet', 'k_eval': 24, 'k_eval_extra': 36,
        'runner_sha256': base['source_bindings'][e113.SHARED_RUNNER],
        'guard_sha256': base['source_bindings'][guard_tool]}
    for key, value in expected.items():
        if prov.get(key) != value:
            raise ValueError('control provenance mismatch: ' + key)
    if any(prov.get('environment', {}).get(k) != base['environment'][k] for k in ENV_KEYS):
        raise ValueError('control environment mismatch')
    if prov.get('init_weights', {}).get('weights_sha256') != RADIMAGENET_SHA256:
        raise ValueError('control initial weights mismatch')
    if not np.array_equal(np.asarray(prov.get('actual_pos_weight'), np.float32), np.asarray(f['pos_weight'], np.float32)):
        raise ValueError('control pos_weight mismatch')
    args = s.get('args', {})
    if any(args.get(k) != v for k, v in base['common_args'].items()) or args.get('seed') != base['seed'] or args.get('tag') != f['tag']:
        raise ValueError('control recipe mismatch')
    if run.name != f['tag'] or s.get('mode') != 'full' or s.get('selected_epoch_1based') != base['common_args']['epochs']:
        raise ValueError('control fixed-final run mismatch')
    if (s.get('train'), s.get('val'), s.get('optimizer_updates')) != (f['n_train'], f['n_val'], f['steps']):
        raise ValueError('control counts/updates mismatch')
    if not 0 <= s.get('skipped_updates', 10**9) <= f['max_fp16_skips']:
        raise ValueError('control FP16 skip gate failed')
    history = read(run / 'history.json')
    if [h.get('epoch') for h in history] != list(range(base['common_args']['epochs'])) or history != s.get('history'):
        raise ValueError('control history mismatch')
    return {display_path(run / n, root): sha256(run / n) for n in required | {'output-inventory.json'}}


def tier_verdicts(mean, per_seed):
    valid = set(REFERENCES).issubset(mean) and all(math.isfinite(v) for n in REFERENCES
        for v in (mean[n]['delta'], *mean[n]['ci95'], mean[n]['positive_draw_share'])) and all(
        mean[n]['bootstrap_finite_draws'] == mean[n]['bootstrap_draws'] for n in REFERENCES)
    if not valid:
        return {n: {'verdict': 'INVALID', 'conditions': {}, 'adoption_authorized': False} for n in ('tier_a', 'tier_b')}
    a = {'dread_delta': mean['dread']['delta'] >= .004,
         'dread_ci': mean['dread']['ci95'][0] > 0,
         'consensus': mean['consensus']['delta'] >= 0}
    b = {'dread_delta': mean['dread']['delta'] >= .001,
         'positive_draw_share': mean['dread']['positive_draw_share'] >= .90,
         'other_references_floor': all(mean[n]['delta'] >= -.001 for n in REFERENCES if n != 'dread'),
         'two_other_references_positive': sum(mean[n]['delta'] > 0 for n in REFERENCES if n != 'dread') >= 2,
         'both_seeds_positive': all(per_seed[str(s)]['dread']['delta'] > 0 for s in SEEDS),
         'same_family_no_extra_submission': True, 'serving_extra_at_most_10pct': True}
    return {'tier_a': {'verdict': 'THRESHOLDS_MET' if all(a.values()) else 'THRESHOLDS_NOT_MET',
                      'conditions': a, 'adoption_authorized': False},
            'tier_b': {'verdict': 'THRESHOLDS_MET' if all(b.values()) else 'THRESHOLDS_NOT_MET',
                      'conditions': b, 'adoption_authorized': False,
                      'remaining_requirements': ['B decision on bundle composition and frozen joint-parent/public rule; no automatic submission']}}


def screen(root=ROOT):
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    root = Path(root)
    out = root / SCREEN
    if out.exists():
        raise SystemExit('Refusing existing screen: ' + str(out))
    registry_path = root / PROTOCOL / 'manifest.json'
    registry = read(registry_path)
    if registry['schema'] != SCHEMA or registry['screen'] != screen_spec():
        raise ValueError('screen registration mismatch')
    e127.check_bindings(root, registry['source_bindings'])
    bindings = {display_path(registry_path, root): sha256(registry_path), **registry['source_bindings']}
    predictions = {k: {role: {} for role in ('candidate', 'control')} for k in ('k36', 'k24')}
    fold_predictions = {k: {} for k in ('k36', 'k24')}
    pooled_ids = None
    for seed in SEEDS:
        p, base, f0 = load_protocol(seed, root), e127.base_manifest(seed, root), e127.load_protocol('AUG1', seed, root)
        ids = []
        for fold in range(5):
            f = p['folds'][str(fold)]
            val = read(root / f['ordered_ids']['path'])['val']
            if len(val) != f['n_val'] or len(set(val)) != len(val):
                raise ValueError('fold roster invalid')
            ids.extend(val)
            fold_predictions['k36'].setdefault(fold, {})[seed] = {'ids': val}
            fold_predictions['k24'].setdefault(fold, {})[seed] = {'ids': val}
            for role, recipe in (('candidate', p if fold else f0), ('control', base)):
                mp = (manifest_path(seed, root) if fold else e127.manifest_path('AUG1', seed, root)) if role == 'candidate' else root / e127.BASES[seed][0] / 'manifest.json'
                run = root / 'artifacts/experiments' / (f['tag'] if role == 'candidate' else base['folds'][str(fold)]['tag'])
                if role == 'candidate':
                    bindings.update(verify_run(run, recipe, sha256(mp), root) if fold else e127.verify_run(run, recipe, sha256(mp), root))
                else:
                    bindings.update(verify_control_run(run, recipe, fold, sha256(mp), root))
                bindings[display_path(mp, root)] = sha256(mp)
                for k, file in (('k36', 'oof_k36.csv'), ('k24', 'oof_best.csv')):
                    arr = e127.load_table(run / file, val, prediction=True)
                    fold_predictions[k][fold][seed][role] = arr
                    predictions[k][role].setdefault(seed, []).append(arr)
        if len(ids) != 4343 or len(set(ids)) != 4343 or (pooled_ids is not None and pooled_ids != ids):
            raise ValueError('pooled OOF roster must be 4343 identical unique studies')
        pooled_ids = ids
    for k in predictions:
        for role in predictions[k]:
            for seed in SEEDS:
                predictions[k][role][seed] = np.concatenate(predictions[k][role][seed])
    refs = e127.make_references({n: e127.load_table(root / path, pooled_ids) for n, path in LABEL_PATHS.items()})
    draws = e127.component_draws(pooled_ids, read(root / COMPONENTS), draws=DRAWS, seed=BOOT_SEED)
    anchor = e127.load_table(root / PARTNER, pooled_ids, prediction=True)
    metrics = {}
    for k, pp in predictions.items():
        per_seed, mean, spread = {str(s): {} for s in SEEDS}, {}, {}
        for name, ref in refs.items():
            individual, mean[name], spread[name] = e127.compare_seeds(pp['candidate'], pp['control'], ref, draws)
            for seed in SEEDS:
                per_seed[str(seed)][name] = individual[str(seed)]
        per_fold = {}
        start = 0
        for fold in range(5):
            val = fold_predictions[k][fold][42]['ids']
            stop = start + len(val)
            per_fold[str(fold)] = {}
            for name, (y, mask) in refs.items():
                ref = (y[start:stop], mask[start:stop])
                per_fold[str(fold)][name] = {}
                for seed in SEEDS:
                    arm = fold_predictions[k][fold][seed]
                    a, b = e127.auc_vector(arm['candidate'], ref), e127.auc_vector(arm['control'], ref)
                    per_fold[str(fold)][name][str(seed)] = float(np.nanmean(a-b))
            start = stop
        partner = e129.partner_proxy(pp['candidate'], pp['control'], anchor, refs['dread'], draws)
        metrics[k] = {'per_seed': per_seed, 'seed_mean': mean, 'seed_spread': spread,
            'per_fold_delta': per_fold, 'tiers': tier_verdicts(mean, per_seed),
            'partner': partner, 'partner_discordance_flag': mean['dread']['delta'] > 0 and partner['delta_partner'] < 0,
            'diagnostic': k == 'k24'}
    rank_mean = np.mean([rankdata(predictions['k36']['candidate'][s], axis=0, method='average') / len(pooled_ids) for s in SEEDS], axis=0)
    result = {'schema': SCREEN_SCHEMA, 'status': 'COMPLETE', 'observed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'protocol_sha256': sha256(registry_path), 'rows': len(pooled_ids), 'ordered_ids_sha256': uid_hash(pooled_ids),
        'input_bindings': bindings, 'screen_spec': screen_spec(), 'metrics': metrics,
        'tiers': metrics['k36']['tiers'],
        'coverage': {n: {'cells': int(mask.sum()), 'per_target': mask.sum(axis=0).tolist()} for n, (_, mask) in refs.items()}}
    e127.check_bindings(root, bindings)
    out.mkdir(parents=True)
    pd.DataFrame({UID: pooled_ids, **{t: rank_mean[:, j] for j, t in enumerate(T)}}).to_csv(out / 'oof_k36_two_seed_rank_mean.csv', index=False)
    dump(out / 'result.json', result)
    (out / 'result.md').write_text(screen_markdown(result), encoding='utf-8')
    print(json.dumps({'output': str(out), 'tiers': result['tiers'], 'k36': metrics['k36']['seed_mean']}, indent=2))
    return result


def screen_markdown(body):
    lines = ['# EXP134 AUG1 pooled five-fold confirmation', '', body['screen_spec']['interpretation'], '',
             '| eval | scope | reference | candidate | control | delta | CI95 | positive draw share |',
             '|---|---|---|---:|---:|---:|---|---:|']
    for k, metric in body['metrics'].items():
        for scope, values in [*metric['per_seed'].items(), ('seed_mean', metric['seed_mean'])]:
            for name, v in values.items():
                lines.append(f"| {k} | {scope} | {name} | {v['candidate']:.6f} | {v['control']:.6f} | {v['delta']:+.6f} | {v['ci95']} | {v['positive_draw_share']:.4f} |")
        lines += ['', f"{k} per-fold deltas: {json.dumps(metric['per_fold_delta'])}", '',
            f"{k} partner (report only): {metric['partner']['delta_partner']:+.6f}; discordance {metric['partner_discordance_flag']}", '',
            f"{k} tier verdicts: {json.dumps(metric['tiers'])}", '']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    p = sub.add_parser('validate')
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--fold', type=int, choices=range(1, 5))
    p = sub.add_parser('queue-json')
    p.add_argument('--after', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=12)
    sub.add_parser('screen')
    a = ap.parse_args(argv)
    if a.command == 'prepare':
        prepare()
    elif a.command == 'validate':
        return validate_cli(a.seed, a.fold)
    elif a.command == 'queue-json':
        entries = queue_entries(a.after, workers=a.workers)
        out = Path(a.out)
        if out.exists():
            raise SystemExit('Refusing to edit existing queue file: ' + str(out))
        dump(out, entries)
        print(json.dumps({'output': str(out), 'jobs': [e['tag'] for e in entries]}, indent=2))
    else:
        screen()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
