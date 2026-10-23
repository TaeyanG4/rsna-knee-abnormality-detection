"""EXP138 AUG1 plus four label-source heads; registration, CPU checks, queue and screen.

No command starts training. TTA is a separate inference queue on the same checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp134_aug1_kfold as aug
from tools import exp117_multihead as mh
from tools import exp122_multihead_seeds as mh43

SCHEMA = 'exp138-bundle-aug1-mh-v1'
PROTOCOL = 'artifacts/experiments/EXP138_BUNDLE_AUG1_MH'
SCREEN = PROTOCOL + '/screen'
TOOL = 'tools/exp138_bundle_aug1_mh.py'
RUNNER = 'training/runners/exp138_bundle_aug1_mh.py'
TEST = 'tests/test_exp138_bundle_aug1_mh.py'
TTA_TOOL = 'tools/exp138_flip_tta.py'
NEW_CODE = (TOOL, RUNNER, TEST, TTA_TOOL)
SEEDS = (42, 43)
FOLDS = range(5)
LABEL_SET = mh.LABEL_SET
CODE_NAMES, ENV_KEYS = aug.CODE_NAMES, aug.ENV_KEYS
HEAD_NAMES = mh.HEAD_NAMES
worker_init, read_rows, install_gold_blocker = aug.worker_init, aug.read_rows, aug.install_gold_blocker
canonical_order, validate_fold_membership = aug.canonical_order, aug.validate_fold_membership
pos_weight, fold_contract = aug.pos_weight, aug.fold_contract
state_hashes, mismatched_keys = aug.state_hashes, aug.mismatched_keys
read, sha256, uid_hash, load_label_csv = aug.read, aug.sha256, aug.uid_hash, aug.load_label_csv
forbid_cuda_init = aug.forbid_cuda_init
head_pos_weight, stack_sources = mh.head_pos_weight, mh.stack_sources
build_multihead, masked_multihead_bce, rank_mean = mh.build_multihead, mh.masked_multihead_bce, mh.rank_mean


def tag_for(seed, fold):
    if seed not in SEEDS or fold not in FOLDS:
        raise ValueError('unregistered EXP138 seed/fold')
    return f'EXP138_BUNDLE_AUG1_MH_S{seed}_F{fold}'


def manifest_path(seed, root=ROOT):
    return Path(root) / PROTOCOL / f's{seed}' / 'manifest.json'


def mh_protocol(seed, root=ROOT):
    return read(Path(root) / (mh.PROTOCOL if seed == 42 else mh43.paths(43)['protocol']) / 'manifest.json')


def aug_protocol(seed, root=ROOT):
    return aug.e127.load_protocol('AUG1', seed, root) if seed == 42 else aug.e127.load_protocol('AUG1', seed, root)


def aug_fold(seed, fold, root=ROOT):
    if fold == 0:
        return aug_protocol(seed, root)['folds']['0']
    return read(aug.manifest_path(seed, root))['folds'][str(fold)]


def screen_spec():
    return dict(folds=list(FOLDS), seeds=list(SEEDS), primary='k36', diagnostic='k24',
                comparisons={'incremental': 'EXP134/EXP127 AUG1 single head', 'total': 'EXP113/EXP118 single head'},
                serving=['plain', 'plane-aware EXP124 horizontal-flip TTA on same checkpoints'],
                per_fold_rank_normalised=True, references=list(aug.REFERENCES),
                draws=300, bootstrap_seed=20261003, paired='component draw indices shared between arms and seeds',
                tier_a=aug.screen_spec()['tier_a'], tier_b=aug.screen_spec()['tier_b'],
                partner_proxy='report only', decision='joint bundle only; no standalone gain added; B decides adoption')


def derive_manifest(seed, root=ROOT):
    root = Path(root)
    source = mh_protocol(seed, root)
    if source['seed'] != seed or tuple(h['name'] for h in source['heads']) != HEAD_NAMES:
        raise ValueError('multi-head source differs')
    base = aug.e127.base_manifest(seed, root)
    if source['common_args'] != base['common_args']:
        raise ValueError('control recipe differs')
    p = copy.deepcopy(source)
    folds = {}
    for fold in FOLDS:
        a, m = aug_fold(seed, fold, root), source['folds'][str(fold)]
        if {k: v for k, v in m.items() if k not in ('tag', 'head_pos_weight')} != {k: v for k, v in a.items() if k != 'tag'}:
            raise ValueError(f'fold {fold} differs from AUG1 control contract')
        folds[str(fold)] = {**copy.deepcopy(a), 'tag': tag_for(seed, fold),
                            'head_pos_weight': copy.deepcopy(m['head_pos_weight'])}
    paths = (TOOL, RUNNER, TEST, TTA_TOOL, PROTOCOL + '/PROTOCOL.md', 'training/src/raptor_exp127_aug1.py',
             'training/src/raptor_exp127.py', 'tools/exp124_flip_tta.py')
    bindings = {**p['source_bindings'], **{x: sha256(root / x) for x in paths}}
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             status='REGISTERED_NO_EXECUTION', plan_item='H-B-47', seed=seed,
             folds=folds, variant='AUG1', variant_config=copy.deepcopy(aug.e127.CONFIG['AUG1']),
             source_bindings=bindings, tags={'prefix': f'EXP138_BUNDLE_AUG1_MH_S{seed}',
                 'folds': {str(f): tag_for(seed, f) for f in FOLDS}},
             local={**p['local'], 'runner': RUNNER, 'guard': TOOL},
             evaluation={**p['evaluation'], 'exp138_screen': screen_spec()},
             derived_from={'multihead_protocol': mh.PROTOCOL if seed == 42 else mh43.paths(43)['protocol'],
                           'multihead_sha256': sha256(root / (mh.PROTOCOL if seed == 42 else mh43.paths(43)['protocol']) / 'manifest.json'),
                           'aug1_f0_sha256': sha256(aug.e127.manifest_path('AUG1', seed, root)),
                           'aug1_kfold_sha256': sha256(aug.manifest_path(seed, root)),
                           'change': 'EXP127 AUG1 competition training windows on EXP117/122 multi-head model'} )
    return p


def prepare(root=ROOT):
    root = Path(root)
    dest = root / PROTOCOL
    if (dest / 'manifest.json').exists():
        raise FileExistsError('EXP138 registration exists')
    manifests = {s: derive_manifest(s, root) for s in SEEDS}
    dest.mkdir(parents=True, exist_ok=True)
    entries = {}
    for seed, p in manifests.items():
        path = manifest_path(seed, root)
        path.parent.mkdir(exist_ok=True)
        aug.dump(path, p)
        entries[str(seed)] = {'path': aug.display_path(path, root), 'sha256': sha256(path)}
    aug.dump(dest / 'manifest.json', {'schema': SCHEMA, 'status': 'REGISTERED_NO_EXECUTION',
                                     'screen': screen_spec(), 'seeds': entries,
                                     'gpu_minutes_per_training_run_estimate': [50, 100]})
    return entries


def load_protocol(seed, root=ROOT):
    root = Path(root)
    registry = read(root / PROTOCOL / 'manifest.json')
    path = manifest_path(seed, root)
    if registry['schema'] != SCHEMA or registry['screen'] != screen_spec() or sha256(path) != registry['seeds'][str(seed)]['sha256']:
        raise ValueError('EXP138 registration changed')
    p = read(path)
    if p['schema'] != SCHEMA or p['seed'] != seed or tuple(h['name'] for h in p['heads']) != HEAD_NAMES:
        raise ValueError('EXP138 protocol changed')
    if (sha256(aug.e127.manifest_path('AUG1', seed, root)) != p['derived_from']['aug1_f0_sha256']
            or sha256(aug.manifest_path(seed, root)) != p['derived_from']['aug1_kfold_sha256']):
        raise ValueError('AUG1 source protocol changed')
    for f in FOLDS:
        a = aug_fold(seed, f, root)
        registered = p['folds'][str(f)]
        if {k: v for k, v in registered.items() if k not in ('tag', 'head_pos_weight')} != {k: v for k, v in a.items() if k != 'tag'}:
            raise ValueError('AUG1 fold contract changed')
        for key in ('ordered_ids', 'groups'):
            if sha256(root / registered[key]['path']) != registered[key]['sha256']:
                raise ValueError('fold source changed: ' + key)
        ordered = read(root / registered['ordered_ids']['path'])
        for part in ('train', 'val'):
            if uid_hash(ordered[part]) != registered['ordered_ids'][part + '_sha256']:
                raise ValueError('fold order changed: ' + part)
    aug.e127.check_bindings(root, p['source_bindings'])
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('frozen training source changed: ' + name)
    return p


def validate_arguments(args, root=ROOT):
    root = Path(root)
    p = load_protocol(args.seed, root)
    if args.fold not in FOLDS or args.mode != 'full' or Path(args.protocol).resolve() != manifest_path(args.seed, root).resolve() or args.tag != tag_for(args.seed, args.fold):
        raise ValueError('EXP138 registered full run only')
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol refuses relaxed environment')
    for key, expected in p['common_args'].items():
        if getattr(args, key, None) != expected and not (args.validate_only and key == 'device' and args.device == 'cpu'):
            raise ValueError('frozen argument changed: ' + key)
    if args.validate_only and args.device != 'cpu':
        raise ValueError('CPU validation required')
    if not 0 <= args.workers <= 12:
        raise ValueError('workers outside registered range')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != (root / p['paths'][key]).resolve():
            raise ValueError('input path changed: ' + key)
    if Path(args.init_weights).resolve() != aug.weights_path(root).resolve():
        raise ValueError('weights path changed')
    if sha256(args.init_weights) != p['init_weights']['sha256']:
        raise ValueError('weights hash changed')
    for key, value in {**aug.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(key) != value:
            raise ValueError('environment changed: ' + key)
    default = (root / 'artifacts/experiments').resolve()
    out = Path(args.output_root).resolve() if args.output_root else default
    if out != default:
        raise ValueError('full output root changed')
    fold = p['folds'][str(args.fold)]
    return p, fold, read(root / fold['ordered_ids']['path']), out


def runner_argv(p, fold, root=ROOT, workers=12):
    argv = aug.e113.runner_argv(p, fold, 'full', root=root, workers=workers, label_set='dread')
    argv[1] = str(Path(root) / RUNNER)
    i = argv.index('--label-set')
    del argv[i:i+2]
    argv = aug._set_flag(argv, '--protocol', str(manifest_path(p['seed'], root)))
    return aug._set_flag(argv, '--tag', tag_for(p['seed'], fold))


def validate_cli(seed=None, fold=None, root=ROOT):
    for s in SEEDS:
        for f in FOLDS:
            if (seed is not None and seed != s) or (fold is not None and fold != f):
                continue
            p = load_protocol(s, root)
            argv = aug._set_flag(runner_argv(p, f, root, workers=0), '--device', 'cpu') + ['--validate-only']
            result = subprocess.run(argv, cwd=root, env=aug.e113.local_env(), check=False)
            if result.returncode:
                return result.returncode
    return 0


def queue_entries(after, seed=None, root=ROOT, workers=12):
    if not 0 <= workers <= 12:
        raise ValueError('workers outside registered range')
    entries, previous = [], None if after.lower() == 'none' else after
    for s in SEEDS:
        if seed is not None and seed != s:
            continue
        p = load_protocol(s, root)
        for f in FOLDS:
            entry = {'tag': tag_for(s, f), 'command': runner_argv(p, f, root, workers), 'continue_on_failure': True}
            if previous:
                entry['after'] = previous
            entries.append(entry)
            previous = entry['tag']
    return entries


def verify_run(run, p, fold, root=ROOT):
    root, run = Path(root), Path(run)
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                'oof_best.csv', 'oof_k36.csv'} | {f'oof_head_{n}{suffix}.csv' for n in HEAD_NAMES for suffix in ('', '_k36')}
    inventory = read(run / 'output-inventory.json')
    if set(inventory) != required or run.name != tag_for(p['seed'], fold):
        raise ValueError('EXP138 inventory/tag incomplete')
    for name, record in inventory.items():
        if record['sha256'] != sha256(run / name):
            raise ValueError('EXP138 output changed: ' + name)
    prov, summary = read(run / 'provenance.json'), read(run / 'summary.json')
    f = p['folds'][str(fold)]
    expected = {'protocol_sha256': sha256(manifest_path(p['seed'], root)), 'fold': fold, 'seed': p['seed'],
                'runner_sha256': p['source_bindings'][RUNNER], 'guard_sha256': p['source_bindings'][TOOL],
                'heads': list(HEAD_NAMES), 'exp127_variant': 'AUG1', 'exp127_variant_config': p['variant_config'],
                'initial_state_matches_registered': True, 'multihead_initial_state_matches_registered': True,
                'environment_matches_baseline': True, 'train_ids_ordered_sha256': f['ordered_ids']['train_sha256'],
                'eval_ids_ordered_sha256': f['ordered_ids']['val_sha256'], 'gold_access': 'disabled'}
    for key, value in expected.items():
        if prov.get(key) != value:
            raise ValueError('EXP138 provenance mismatch: ' + key)
    if (summary.get('optimizer_updates') != f['steps'] or summary.get('selected_epoch_1based') != p['common_args']['epochs']
            or summary.get('mode') != 'full' or summary.get('val') != f['n_val']
            or not 0 <= summary.get('skipped_updates', -1) <= f['max_fp16_skips']):
        raise ValueError('EXP138 contract/update mismatch')
    return {aug.display_path(run / n, root): sha256(run / n) for n in required | {'output-inventory.json'}}


def fold_ranks(arr):
    from scipy.stats import rankdata
    return rankdata(arr, axis=0, method='average') / len(arr)


def tier_verdicts(mean, per_seed):
    tiers = aug.tier_verdicts(mean, per_seed)
    b = tiers['tier_b']
    b['statistical_verdict'] = b['verdict']
    b['conditions']['serving_extra_at_most_10pct'] = False
    b['serving_cost_status'] = 'UNMEASURED for the four-head bundle and TTA'
    if b['verdict'] == 'THRESHOLDS_MET':
        b['verdict'] = 'PENDING_SERVING_COST'
    b['adoption_authorized'] = False
    return tiers


def screen(root=ROOT):
    """Only full, audited ten-fold runs and their paired TTA outputs are scored."""
    import numpy as np
    from tools import exp138_flip_tta as flip
    root = Path(root)
    out = root / SCREEN
    if out.exists():
        raise FileExistsError(out)
    registry = read(root / PROTOCOL / 'manifest.json')
    if registry['screen'] != screen_spec():
        raise ValueError('screen registration changed')
    arrays = {arm: {s: [] for s in SEEDS} for arm in
              ('plain', 'tta', 'aug1', 'base', 'plain_k24', 'aug1_k24', 'base_k24')}
    per_fold = {}
    bindings, ids = {}, None
    for seed in SEEDS:
        p = load_protocol(seed, root)
        base = aug.e127.base_manifest(seed, root)
        mp = manifest_path(seed, root)
        bindings[aug.display_path(mp, root)] = sha256(mp)
        roster = []
        tta_path = flip.tta_dir(seed, root)
        tta_result = read(tta_path / 'result.json')
        if tta_result['protocol_sha256'] != sha256(manifest_path(seed, root)) or tta_result['rows'] != 4343:
            raise ValueError('TTA protocol/roster mismatch')
        for name, digest in tta_result['outputs'].items():
            if digest != sha256(tta_path / name):
                raise ValueError('TTA output changed')
        tta_table = tta_path / 'oof_tta_k36.csv'
        bindings[aug.display_path(tta_path / 'result.json', root)] = sha256(tta_path / 'result.json')
        bindings[aug.display_path(tta_table, root)] = sha256(tta_table)
        for fold in FOLDS:
            f = p['folds'][str(fold)]
            val = read(root / f['ordered_ids']['path'])['val']
            if len(val) != f['n_val'] or len(set(val)) != len(val):
                raise ValueError('fold roster changed')
            roster += val
            candidate = root / 'artifacts/experiments' / tag_for(seed, fold)
            bindings.update(verify_run(candidate, p, fold, root))
            ac = aug_protocol(seed, root) if fold == 0 else aug.load_protocol(seed, root)
            aug_run = root / 'artifacts/experiments' / ac['folds'][str(fold)]['tag']
            ac_path = aug.e127.manifest_path('AUG1', seed, root) if fold == 0 else aug.manifest_path(seed, root)
            bindings[aug.display_path(ac_path, root)] = sha256(ac_path)
            bindings.update(aug.e127.verify_run(aug_run, ac, sha256(ac_path), root) if fold == 0
                            else aug.verify_run(aug_run, ac, sha256(ac_path), root))
            base_run = root / 'artifacts/experiments' / base['folds'][str(fold)]['tag']
            base_path = root / aug.e127.BASES[seed][0] / 'manifest.json'
            bindings[aug.display_path(base_path, root)] = sha256(base_path)
            bindings.update(aug.verify_control_run(base_run, base, fold, sha256(base_path), root))
            per_fold.setdefault(str(fold), {})[str(seed)] = {}
            for arm, path in (('plain', candidate / 'oof_k36.csv'), ('tta', tta_table),
                              ('aug1', aug_run / 'oof_k36.csv'), ('base', base_run / 'oof_k36.csv'),
                              ('plain_k24', candidate / 'oof_best.csv'),
                              ('aug1_k24', aug_run / 'oof_best.csv'), ('base_k24', base_run / 'oof_best.csv')):
                values = aug.e127.load_table(path, val, prediction=True) if arm != 'tta' else _tta_subset(path, val)
                ranks = fold_ranks(values)
                arrays[arm][seed].append(ranks)
                per_fold[str(fold)][str(seed)][arm] = aug.uid_hash(val)
        if len(roster) != 4343 or len(set(roster)) != 4343 or (ids is not None and roster != ids):
            raise ValueError('pooled roster mismatch')
        ids = roster
    pooled = {arm: {s: np.concatenate(parts) for s, parts in seed_arrays.items()}
              for arm, seed_arrays in arrays.items()}
    refs = aug.e127.make_references({n: aug.e127.load_table(root / path, ids)
                                     for n, path in aug.LABEL_PATHS.items()})
    draws = aug.e127.component_draws(ids, read(root / aug.COMPONENTS), draws=300, seed=20261003)
    anchor = aug.e127.load_table(root / aug.PARTNER, ids, prediction=True)
    contrasts = {}
    for variant, controls in (('plain', ('aug1', 'base')), ('tta', ('aug1', 'base')),
                              ('plain_k24', ('aug1_k24', 'base_k24'))):
        for control in controls:
            per_seed, means, spread = {str(s): {} for s in SEEDS}, {}, {}
            for name, ref in refs.items():
                each, mean, diff = aug.e127.compare_seeds(pooled[variant], pooled[control], ref, draws)
                means[name], spread[name] = mean, diff
                for s in SEEDS:
                    per_seed[str(s)][name] = each[str(s)]
            fold_deltas = {}
            start = 0
            for fold in FOLDS:
                count = len(read(root / p['folds'][str(fold)]['ordered_ids']['path'])['val'])
                stop = start + count
                fold_deltas[str(fold)] = {}
                for name, (y, mask) in refs.items():
                    fold_deltas[str(fold)][name] = {}
                    for s in SEEDS:
                        ref = (y[start:stop], mask[start:stop])
                        va = aug.e127.auc_vector(pooled[variant][s][start:stop], ref)
                        vb = aug.e127.auc_vector(pooled[control][s][start:stop], ref)
                        fold_deltas[str(fold)][name][str(s)] = float(np.nanmean(va - vb))
                start = stop
            tiers = tier_verdicts(means, per_seed) if not variant.endswith('_k24') else {
                'tier_a': {'verdict': 'DIAGNOSTIC_ONLY', 'adoption_authorized': False},
                'tier_b': {'verdict': 'DIAGNOSTIC_ONLY', 'adoption_authorized': False}}
            contrasts[f'{variant}_vs_{control}'] = {
                'per_seed': per_seed, 'seed_mean': means, 'seed_spread': spread,
                'per_fold_delta': fold_deltas, 'diagnostic': variant.endswith('_k24'),
                'tiers': tiers,
                'partner': aug.e129.partner_proxy(pooled[variant], pooled[control], anchor, refs['dread'], draws)}
    result = {'schema': SCHEMA + '-screen', 'status': 'COMPLETE', 'rows': len(ids),
              'ordered_ids_sha256': uid_hash(ids), 'protocol_sha256': sha256(root / PROTOCOL / 'manifest.json'),
              'screen_spec': screen_spec(), 'contrasts': contrasts, 'per_fold_roster_sha256': per_fold,
              'input_bindings': bindings, 'adoption_authorized': False}
    out.mkdir(parents=True)
    aug.dump(out / 'result.json', result)
    lines = ['# EXP138 joint bundle screen', '', 'Partner proxy is report only. No automatic adoption.', '',
             '| Contrast | Reference | Candidate | Control | Delta | CI95 | Tier A | Tier B |',
             '|---|---|---:|---:|---:|---|---|---|']
    for label, contrast in contrasts.items():
        for name, value in contrast['seed_mean'].items():
            lines.append(f"| {label} | {name} | {value['candidate']:.6f} | {value['control']:.6f} | {value['delta']:+.6f} | {value['ci95']} | {contrast['tiers']['tier_a']['verdict']} | {contrast['tiers']['tier_b']['verdict']} |")
    (out / 'result.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(out), 'tiers': {k: v['tiers'] for k, v in contrasts.items()}}, indent=2))
    return result


def _tta_subset(path, ids):
    import pandas as pd
    frame = pd.read_csv(path, dtype={aug.UID: str}).set_index(aug.UID)
    if list(frame.columns) != list(aug.T) or not set(ids) <= set(frame.index):
        raise ValueError('TTA table columns/roster changed')
    return frame.loc[ids, list(aug.T)].to_numpy('float64')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    p = sub.add_parser('validate')
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--fold', type=int, choices=FOLDS)
    p = sub.add_parser('queue-json')
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--after', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=12)
    p = sub.add_parser('queue-tta-json')
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=2)
    sub.add_parser('screen')
    a = ap.parse_args(argv)
    if a.action == 'prepare':
        print(json.dumps(prepare(), indent=2))
    elif a.action == 'validate':
        return validate_cli(a.seed, a.fold)
    elif a.action == 'queue-json':
        entries = queue_entries(a.after, a.seed, workers=a.workers)
        out = Path(a.out)
        if out.exists():
            raise FileExistsError(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        aug.dump(out, entries)
        print(json.dumps({'output': str(out), 'jobs': [e['tag'] for e in entries]}, indent=2))
    elif a.action == 'queue-tta-json':
        from tools import exp138_flip_tta as flip
        entries = flip.queue_entries(a.after, a.seed, a.workers)
        out = Path(a.out)
        if out.exists():
            raise FileExistsError(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        aug.dump(out, entries)
        print(json.dumps({'output': str(out), 'jobs': [e['tag'] for e in entries]}, indent=2))
    else:
        screen()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
