"""Register, CPU-validate, queue, and screen the single EXP142 MRNet rescue arm."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp132_aug1 as old

aug, ext = old.aug, old.ext
SEEDS = (42, 43)
SCHEMA = 'exp142-mrn-rescue-v1'
PROTOCOL = 'artifacts/experiments/EXP142_MRN_RESCUE'
RUNNER = 'training/runners/exp142_mrn_rescue.py'
TOOL = 'tools/exp142_mrn_rescue.py'
TEST = 'tests/test_exp142_mrn_rescue.py'
RATIO, LOSS_WEIGHT = .25, .2
LABEL_SET, CODE_NAMES, ENV_KEYS = old.LABEL_SET, old.CODE_NAMES, old.ENV_KEYS
worker_init, read_rows, read = old.worker_init, old.read_rows, old.read
canonical_order, validate_fold_membership = old.canonical_order, old.validate_fold_membership
pos_weight, fold_contract = old.pos_weight, old.fold_contract
state_hashes, mismatched_keys, uid_hash = old.state_hashes, old.mismatched_keys, old.uid_hash
load_label_csv, install_gold_blocker, forbid_cuda_init = old.load_label_csv, old.install_gold_blocker, old.forbid_cuda_init


def tag_for(seed):
    if seed not in SEEDS:
        raise ValueError('unregistered seed')
    return f'EXP142_MRN_RESCUE_S{seed}_F0'


def protocol_path(seed, root=ROOT):
    return Path(root) / PROTOCOL / 'protocol' / f's{seed}.json'


def make_protocol(seed, root=ROOT):
    root = Path(root)
    source = old.make_protocol('MRN', seed, root)
    p = copy.deepcopy(source)
    files = (TOOL, RUNNER, TEST, PROTOCOL + '/PROTOCOL.md')
    p.update(schema=SCHEMA, variant='MRN', seed=seed, tag=tag_for(seed),
             code_sha256={**source['code_sha256'], **{name: aug.sha256(root / name) for name in files}},
             external_batches_per_competition_step=RATIO, auxiliary_bce_weight=LOSS_WEIGHT,
             external_augmentation='EXP127 AUG1 on MRNet windows, distinct exp142_external_aug1 study RNG stream',
             external_batchnorm='BatchNorm eval for external forward/backward only; running stats frozen, affine gradients retained',
             competition_augmentation='EXP127 AUG1 unchanged competition stream',
             external_schedule='one MRNet batch before every fourth competition optimizer step; shared update',
             decision='seed-mean K36 ΔDread > 0 and each seed K36 ΔDread >= -0.001 admits five-fold discussion; otherwise close auxiliary-head configuration',
             serving='competition model only; auxiliary head dropped',
             frozen_scope='F0 only; seeds 42/43 and same-seed EXP127_AUG1 controls; competition OOF only')
    return p


def load_protocol(seed, root=ROOT):
    path = protocol_path(seed, root)
    p = json.loads(path.read_text(encoding='utf-8'))
    if p != make_protocol(seed, root):
        raise ValueError('EXP142 protocol/input/code binding changed')
    return p


def prepare(root=ROOT):
    result = {seed: make_protocol(seed, root) for seed in SEEDS}
    for seed, p in result.items():
        path = protocol_path(seed, root)
        if path.exists():
            load_protocol(seed, root)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                json.dump(p, stream, indent=2)
    return result


def validate_arguments(args, root=ROOT):
    if args.variant != 'MRN':
        raise ValueError('EXP142 is MRNet only')
    root = Path(root)
    p = load_protocol(args.seed, root)
    base = aug.e127.base_manifest(args.seed, root)
    if args.fold != 0 or Path(args.protocol).resolve() != protocol_path(args.seed, root).resolve():
        raise ValueError('EXP142 protocol/fold changed')
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


def runner_argv(seed, root=ROOT, workers=12):
    p = load_protocol(seed, root)
    base = aug.e127.base_manifest(seed, root)
    argv = aug.e113.runner_argv(base, 0, 'full', root=root, workers=workers, label_set='dread')
    argv[1] = str(Path(root) / RUNNER)
    i = argv.index('--label-set')
    del argv[i:i + 2]
    argv = aug._set_flag(argv, '--protocol', str(protocol_path(seed, root)))
    argv = aug._set_flag(argv, '--tag', p['tag'])
    return argv + ['--variant', 'MRN']


def queue_entries(seed, after, root=ROOT, workers=12):
    if not 0 <= workers <= 12:
        raise ValueError('workers outside registered range')
    entry = {'tag': tag_for(seed), 'command': runner_argv(seed, root, workers), 'continue_on_failure': True}
    if after.lower() != 'none':
        entry['after'] = after
    return [entry]


def validate(seed=None, root=ROOT, workers=2):
    seeds = SEEDS if seed is None else (seed,)
    for one in seeds:
        argv = runner_argv(one, root, workers)
        argv = aug._set_flag(argv, '--device', 'cpu') + ['--validate-only']
        env = os.environ.copy()
        env.update(aug.THREAD_ENV)
        env['HF_HUB_OFFLINE'] = '1'
        result = subprocess.run(argv, cwd=root, env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'seed {one} validation failed: {result.stderr[-3000:]}')
        report = json.loads(result.stdout)
        if report.get('status') != 'VALIDATED_NO_TRAINING' or report.get('tag') != tag_for(one):
            raise RuntimeError(f'seed {one} did not validate: {result.stdout[-2000:]}')
        print(result.stdout)


def validate_run(run, p, root=ROOT):
    run = Path(run)
    inv = read(run / 'output-inventory.json')
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv'}
    if set(inv) != required:
        raise ValueError('EXP142 output inventory incomplete')
    for name in required:
        if inv[name]['sha256'] != aug.sha256(run / name):
            raise ValueError('EXP142 output changed')
    summary, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expected = {'protocol_sha256': aug.sha256(protocol_path(p['seed'], root)),
                'exp132_variant': 'MRN', 'exp132_ratio': .25, 'exp132_loss_weight': .2,
                'exp127_variant': 'AUG1', 'exp127_variant_config': aug.e127.CONFIG['AUG1'],
                'competition_only_augmentation': False,
                'external_augmentation': 'EXP127 AUG1 external separate stream exp142_external_aug1',
                'external_batchnorm': 'eval for external forward only; affine trainable',
                'exp142_aux_weight': .2,
                'initial_state_matches_registered': True, 'environment_matches_baseline': True}
    for key, value in expected.items():
        if prov.get(key) != value:
            raise ValueError('EXP142 provenance mismatch: ' + key)
    fold = p['fold_contract']
    if (summary.get('optimizer_updates') != fold['steps'] or summary.get('auxiliary_parameters_exported') is not False
            or summary.get('auxiliary_batches') != (fold['len_train_loader'] // 4) * p['common_args']['epochs']
            or summary.get('selected_epoch_1based') != p['common_args']['epochs']):
        raise ValueError('EXP142 schedule/export changed')
    return {str(run / name): aug.sha256(run / name) for name in required | {'output-inventory.json'}}


def screen_decision(metric):
    mean_delta = metric['seed_mean']['dread']['delta']
    per_seed = {str(s): metric['per_seed'][str(s)]['dread']['delta'] for s in SEEDS}
    passed = mean_delta > 0 and all(v >= -.001 for v in per_seed.values())
    return ('ADMIT_FIVE_FOLD_DISCUSSION' if passed else 'CLOSE_AUXILIARY_HEAD_CONFIGURATION',
            mean_delta, per_seed)


def screen(root=ROOT):
    from tools import exp129_radr50_single as math
    root = Path(root)
    out = root / PROTOCOL / 'screen'
    if out.exists():
        raise FileExistsError(out)
    pp = {k: {'candidate': {}, 'control': {}} for k in ('k36', 'k24')}
    bindings, ids = {}, None
    for seed in SEEDS:
        p = load_protocol(seed, root)
        control = aug.e127.load_protocol('AUG1', seed, root)
        bindings[str(protocol_path(seed, root))] = aug.sha256(protocol_path(seed, root))
        bindings[str(aug.e127.manifest_path('AUG1', seed, root))] = aug.sha256(aug.e127.manifest_path('AUG1', seed, root))
        val = read(root / p['fold_contract']['ordered_ids']['path'])['val']
        if ids is not None and val != ids:
            raise ValueError('seed rosters differ')
        ids = val
        candidate = root / 'artifacts/experiments' / tag_for(seed)
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
    primary = metrics['k36']
    decision, mean_delta, per_seed = screen_decision(primary)
    body = {'schema': SCHEMA + '-screen', 'scope': 'F0 only', 'rows': len(ids),
            'metrics': metrics, 'tiers': primary['tiers'], 'decision': decision,
            'decision_rule': 'K36 seed-mean ΔDread > 0 and both seeds >= -0.001',
            'k36_dread_seed_mean_delta': mean_delta, 'k36_dread_per_seed_delta': per_seed,
            'input_bindings': bindings, 'external_validation_rows': 0, 'adoption_authorized': False,
            'caveat': 'Tier A provisional F0; Tier B needs pooled five-fold two-seed evidence; partner report only'}
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps(body, indent=2), encoding='utf-8')
    (out / 'result.md').write_text('# EXP142 MRNet rescue F0 screen\n\n' +
        f"Decision: {decision}; K36 seed-mean ΔDread {mean_delta:+.6f}; per seed {per_seed}.\n\n" +
        body['caveat'] + '\n\n' + json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'decision': decision, 'k36_dread_seed_mean_delta': mean_delta, 'per_seed': per_seed,
                      'tiers': body['tiers'], 'output': str(out)}, indent=2))
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    p = sub.add_parser('validate')
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--workers', type=int, default=2)
    p = sub.add_parser('queue-json')
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=12)
    sub.add_parser('screen')
    a = parser.parse_args(argv)
    if a.action == 'prepare':
        prepare()
        print(json.dumps({'status': 'PREPARED_NO_TRAINING', 'seeds': SEEDS}))
    elif a.action == 'validate':
        prepare()
        validate(a.seed, workers=a.workers)
    elif a.action == 'queue-json':
        prepare()
        entries = queue_entries(a.seed, a.after, workers=a.workers)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf-8') as stream:
            json.dump(entries, stream, indent=2)
        print(json.dumps({'output': str(out), 'jobs': [v['tag'] for v in entries]}))
    else:
        screen()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
