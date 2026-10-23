"""EXP132 KMRI/MRN external auxiliary heads on the EXP127 AUG1 competition base.

Registration and queue output are CPU-only. External batches retain EXP132's
unaugmented CenterSliceWindows rule; AUG1 applies to competition batches only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp132_external_aux as ext
from tools import exp134_aug1_kfold as aug

SCHEMA = 'exp132-aug1-v1'
PROTOCOL = 'artifacts/experiments/EXP132_AUG1'
RUNNER = 'training/src/raptor_exp132_aug1.py'
TOOL = 'tools/exp132_aug1.py'
TEST = 'tests/test_exp132_aug1.py'
SEEDS, VARIANTS = (42, 43), ('KMRI', 'MRN')
RATIO, LOSS_WEIGHT = .25, .5
LABEL_SET, CODE_NAMES, ENV_KEYS = aug.LABEL_SET, aug.CODE_NAMES, aug.ENV_KEYS
worker_init, read_rows, read = aug.worker_init, aug.read_rows, aug.read
canonical_order, validate_fold_membership = aug.canonical_order, aug.validate_fold_membership
pos_weight, fold_contract = aug.pos_weight, aug.fold_contract
state_hashes, mismatched_keys, uid_hash = aug.state_hashes, aug.mismatched_keys, aug.uid_hash
load_label_csv, install_gold_blocker, forbid_cuda_init = aug.load_label_csv, aug.install_gold_blocker, aug.forbid_cuda_init


def tag_for(variant, seed):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError('unregistered EXP132 AUG1 variant/seed')
    return f'EXP132_AUG1_{variant}_S{seed}_F0'


def protocol_path(variant, seed, root=ROOT):
    return Path(root) / PROTOCOL / 'protocol' / variant / f's{seed}.json'


def make_protocol(variant, seed, root=ROOT):
    root = Path(root)
    source = ext.make_protocol(variant, seed, root)
    control = aug.e127.load_protocol('AUG1', seed, root)
    base = aug.e127.base_manifest(seed, root)
    if control['common_args'] != base['common_args']:
        raise ValueError('AUG1/control recipe changed')
    p = copy.deepcopy(source)
    files = (TOOL, RUNNER, TEST, PROTOCOL + '/PROTOCOL.md', 'training/src/raptor_exp127_aug1.py',
             'training/src/raptor_exp127.py', 'training/src/raptor_exp132.py')
    p.update(schema=SCHEMA, tag=tag_for(variant, seed), base_tag=control['folds']['0']['tag'],
             base_protocol_path=aug.display_path(aug.e127.manifest_path('AUG1', seed, root), root),
             base_protocol_sha256=aug.sha256(aug.e127.manifest_path('AUG1', seed, root)),
             fold_contract=copy.deepcopy(control['folds']['0']),
             code_sha256={**source['code_sha256'], **{n: aug.sha256(root / n) for n in files}},
             competition_augmentation='EXP127 AUG1 training batches only',
             external_augmentation='EXP132 unchanged CenterSliceWindows; no AUG1',
             external_batches_per_competition_step=RATIO, auxiliary_bce_weight=LOSS_WEIGHT,
             serving='competition model only; auxiliary heads dropped',
             screen=aug.e127.screen_spec(root),
             frozen_scope='F0 only; same seed EXP127_AUG1 controls; competition OOF only')
    return p


def load_protocol(variant, seed, root=ROOT):
    path = protocol_path(variant, seed, root)
    p = json.loads(path.read_text(encoding='utf-8'))
    if p != make_protocol(variant, seed, root):
        raise ValueError('EXP132 AUG1 protocol/input/code binding changed')
    return p


def register(variant, root=ROOT):
    if variant not in VARIANTS:
        raise ValueError('unregistered variant')
    result = {seed: make_protocol(variant, seed, root) for seed in SEEDS}
    for seed, p in result.items():
        path = protocol_path(variant, seed, root)
        if path.exists():
            load_protocol(variant, seed, root)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                json.dump(p, stream, indent=2)
    return result


def validate_arguments(args, root=ROOT):
    root = Path(root)
    p = load_protocol(args.variant, args.seed, root)
    base = aug.e127.base_manifest(args.seed, root)
    if args.fold != 0 or Path(args.protocol).resolve() != protocol_path(args.variant, args.seed, root).resolve():
        raise ValueError('EXP132 AUG1 protocol/fold changed')
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol refuses relaxed environment')
    for key, value in p['common_args'].items():
        if getattr(args, key, None) != value and not (args.validate_only and key == 'device' and args.device == 'cpu'):
            raise ValueError('frozen AUG1 recipe changed: ' + key)
    if args.validate_only and args.device != 'cpu':
        raise ValueError('validation must use CPU')
    if args.mode != 'full' or args.tag != p['tag'] or not 0 <= args.workers <= 12:
        raise ValueError('registered full F0 run only')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != (root / base['paths'][key]).resolve():
            raise ValueError('input path changed: ' + key)
    if Path(args.init_weights).resolve() != aug.weights_path(root).resolve():
        raise ValueError('weight path changed')
    if aug.sha256(args.init_weights) != base['init_weights']['sha256']:
        raise ValueError('weights hash changed')
    for key in ('ordered_ids', 'groups'):
        if aug.sha256(root / p['fold_contract'][key]['path']) != p['fold_contract'][key]['sha256']:
            raise ValueError('fold input changed: ' + key)
    for key, value in {**aug.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(key) != value:
            raise ValueError('environment changed: ' + key)
    default = (root / 'artifacts/experiments').resolve()
    out = Path(args.output_root).resolve() if args.output_root else default
    if out != default:
        raise ValueError('output root changed')
    control = aug.e127.load_protocol('AUG1', args.seed, root)
    return control, p['fold_contract'], read(root / p['fold_contract']['ordered_ids']['path']), out


def runner_argv(variant, seed, root=ROOT, workers=12):
    p = load_protocol(variant, seed, root)
    base = aug.e127.base_manifest(seed, root)
    argv = aug.e113.runner_argv(base, 0, 'full', root=root, workers=workers, label_set='dread')
    argv[1] = str(Path(root) / RUNNER)
    i = argv.index('--label-set')
    del argv[i:i+2]
    argv = aug._set_flag(argv, '--protocol', str(protocol_path(variant, seed, root)))
    argv = aug._set_flag(argv, '--tag', p['tag'])
    return argv + ['--variant', variant]


def queue_entries(variant, seed, after, root=ROOT, workers=12):
    if not 0 <= workers <= 12:
        raise ValueError('workers outside registered range')
    entry = {'tag': tag_for(variant, seed), 'command': runner_argv(variant, seed, root, workers),
             'continue_on_failure': True}
    if after.lower() != 'none':
        entry['after'] = after
    return [entry]


def screen(variant, root=ROOT):
    import numpy as np
    from tools import exp129_radr50_single as math
    root = Path(root)
    out = root / PROTOCOL / 'screen' / variant
    if out.exists():
        raise FileExistsError(out)
    pp = {k: {'candidate': {}, 'control': {}} for k in ('k36', 'k24')}
    bindings, ids = {}, None
    for seed in SEEDS:
        p = load_protocol(variant, seed, root)
        control = aug.e127.load_protocol('AUG1', seed, root)
        bindings[str(protocol_path(variant, seed, root))] = aug.sha256(protocol_path(variant, seed, root))
        bindings[str(aug.e127.manifest_path('AUG1', seed, root))] = aug.sha256(aug.e127.manifest_path('AUG1', seed, root))
        val = read(root / p['fold_contract']['ordered_ids']['path'])['val']
        if ids is not None and val != ids:
            raise ValueError('seed rosters differ')
        ids = val
        candidate = root / 'artifacts/experiments' / tag_for(variant, seed)
        bindings.update(validate_run(candidate, p, root))
        control_run = root / 'artifacts/experiments' / control['folds']['0']['tag']
        bindings.update(aug.e127.verify_run(control_run, control,
            aug.sha256(aug.e127.manifest_path('AUG1', seed, root)), root))
        for k, name in (('k36', 'oof_k36.csv'), ('k24', 'oof_best.csv')):
            pp[k]['candidate'][seed] = aug.e127.load_table(candidate / name, val, prediction=True)
            pp[k]['control'][seed] = aug.e127.load_table(control_run / name, val, prediction=True)
    refs = aug.e127.make_references({n: aug.e127.load_table(root / path, ids)
                                     for n, path in aug.LABEL_PATHS.items()})
    draws = aug.e127.component_draws(ids, read(root / aug.COMPONENTS), draws=300, seed=20261003)
    all_ids = [u for fold in aug.e127.base_manifest(42, root)['folds'].values()
               for u in read(root / fold['ordered_ids']['path'])['val']]
    anchor_all = aug.e127.load_table(root / aug.PARTNER, all_ids, prediction=True)
    ix = {u: i for i, u in enumerate(all_ids)}
    anchor = anchor_all[[ix[u] for u in ids]]
    metrics = {k: math.metric_contrast(v['candidate'], v['control'], refs, draws, anchor, k != 'k36')
               for k, v in pp.items()}
    body = {'schema': SCHEMA + '-screen', 'variant': variant, 'scope': 'F0 only', 'rows': len(ids),
            'metrics': metrics, 'tiers': metrics['k36']['tiers'], 'input_bindings': bindings,
            'external_validation_rows': 0, 'adoption_authorized': False,
            'caveat': 'Tier A provisional F0; Tier B needs pooled five-fold two-seed evidence; partner report only'}
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps(body, indent=2), encoding='utf-8')
    (out / 'result.md').write_text(f"# EXP132 AUG1 {variant} F0 screen\n\n{body['caveat']}\n\n" +
                                   json.dumps(metrics['k36'], indent=2) + '\n', encoding='utf-8')
    return body


def validate_run(run, p, root=ROOT):
    run = Path(run)
    inv = read(run / 'output-inventory.json')
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv'}
    if set(inv) != required:
        raise ValueError('EXP132 AUG1 output inventory incomplete')
    for name in required:
        if inv[name]['sha256'] != aug.sha256(run / name):
            raise ValueError('EXP132 AUG1 output changed')
    summary, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expected = {'protocol_sha256': aug.sha256(protocol_path(p['variant'], p['seed'], root)),
                'exp132_variant': p['variant'], 'exp132_ratio': .25, 'exp132_loss_weight': .5,
                'exp127_variant': 'AUG1', 'exp127_variant_config': aug.e127.CONFIG['AUG1'],
                'competition_only_augmentation': True, 'external_augmentation': 'EXP132 unchanged',
                'initial_state_matches_registered': True, 'environment_matches_baseline': True}
    for key, value in expected.items():
        if prov.get(key) != value:
            raise ValueError('EXP132 AUG1 provenance mismatch: ' + key)
    fold = p['fold_contract']
    if (summary.get('optimizer_updates') != fold['steps'] or summary.get('auxiliary_parameters_exported') is not False
            or summary.get('auxiliary_batches') != (fold['len_train_loader'] // 4) * p['common_args']['epochs']
            or summary.get('selected_epoch_1based') != p['common_args']['epochs']):
        raise ValueError('EXP132 AUG1 schedule/export changed')
    return {str(run / name): aug.sha256(run / name) for name in required | {'output-inventory.json'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('queue-json')
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=12)
    p = sub.add_parser('screen')
    p.add_argument('--variant', choices=VARIANTS, required=True)
    a = parser.parse_args(argv)
    if a.action == 'queue-json':
        register(a.variant)
        entries = queue_entries(a.variant, a.seed, a.after, workers=a.workers)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf-8') as stream:
            json.dump(entries, stream, indent=2)
        print(json.dumps({'output': str(out), 'jobs': [v['tag'] for v in entries]}))
    else:
        screen(a.variant)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
