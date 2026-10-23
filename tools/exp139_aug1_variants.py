"""EXP139: AUG1 plus exactly one EXP129 intervention, POS/LATE then CTX.

prepare --variant / validate --variant --seed / queue-json --variant /
screen --variant / screen-all. These commands never start training or use CUDA.
Full jobs are emitted as queue JSON only. Existing registrations/runs/reports
are immutable. Folds 1-4 are deliberately not executable under this screen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from tools import exp125_radr50_k16_local as e125
from tools import exp113_radr50_local as e113
from tools import exp118_radr50_local_s43 as e118
from tools import exp127_radr50_single as e127
from tools import exp129_radr50_single as e129
from tools.exp113_radr50_local import (worker_init, read_rows, install_gold_blocker,
    uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order,
    validate_fold_membership, pos_weight, contract, UID, T, DREAD, KEYS_FILE, FOLDS,
    EXTRA_EVAL, RUN_FILES, fold_contract, state_hashes, mismatched_keys, dump,
    train_python, within, display_path, _set_flag, weights_path, LOCAL_WORKERS,
    RADIMAGENET_SHA256, WEIGHTS_LOCAL)

SCHEMA = 'exp139-aug1-variants-v1'
SCREEN_SCHEMA = 'exp139-aug1-variants-screen-v1'
TOOL = 'tools/exp139_aug1_variants.py'
RUNNER = 'training/runners/exp139_aug1_variants.py'
TEST = 'tests/test_exp139_aug1_variants.py'
NEW_CODE = (TOOL, RUNNER, 'training/src/raptor_exp139.py',
            'training/src/raptor_exp127_aug1.py', 'training/src/raptor_exp129.py',
            'training/src/raptor_exp129_late.py')
TESTS = (TEST, 'tests/test_exp139_equivalence.py')
PROTOCOL_DOC = 'artifacts/experiments/EXP139_AUG1_VARIANTS/PROTOCOL.md'
VARIANTS = ('POS', 'LATE', 'CTX')
PARTNER = 'artifacts/experiments/EXP099_B8_DREAD_OOF/oof_k36.csv'
SEEDS, LABEL_SET = e125.SEEDS, 'dread'
DRAWS, BOOT_SEED = 300, 20261003
LABEL_PATHS, REFERENCES, COMPONENTS = e125.LABEL_PATHS, e125.REFERENCES, e125.COMPONENTS
check_bindings = e125.check_bindings
load_table, make_references = e125.load_table, e125.make_references
component_draws, auc_vector = e125.component_draws, e125.auc_vector

CONFIG = {v: copy.deepcopy(e129.CONFIG[v]) for v in VARIANTS}
ESTIMATES = {
    'POS': dict(minutes_per_seed=[48, 60], budget_minutes_per_seed=60, serving='tiny projection; latency unmeasured'),
    'LATE': dict(minutes_per_seed=[50, 66], budget_minutes_per_seed=66, serving='each of final/SWA/EMA individually identical control cost'),
    'CTX': dict(minutes_per_seed=[50, 70], budget_minutes_per_seed=70, serving='two-layer transformer overhead unmeasured'),
}
ESTIMATE_BASIS = ('Planning ranges, not measured EXP139 runtime: EXP127_AUG1_S42_F0/summary.json total_seconds=3043 '
    'and EXP127_AUG1_S43_F0/summary.json total_seconds=2803 (50.7/46.7 min; training 2960/2725s, peak 4.4 GiB); '
    'allow POS/CTX overhead and LATE extra evaluation/EMA/checkpoint I/O. '
    'RTX 4070 Ti SUPER, 12 workers, one GPU job at a time; two seeds cost twice the per-seed range.')
CHANGED_FIELDS = {'schema', 'created_at', 'plan_item', 'source_bindings', 'common_args', 'folds',
    'tags', 'local', 'evaluation', 'derived_from', 'label_sets', 'variant', 'variant_config', 'total_steps',
    'adapter_initial_state', 'augmentation_config'}


def base_manifest(seed, root=ROOT):
    """The trained AUG1 F0 registration, never the unaugmented EXP113/118 arm."""
    return e127.load_protocol('AUG1', seed, root)


def control_path(seed, root=ROOT):
    return e127.manifest_path('AUG1', seed, root)


def protocol_dir(variant):
    if variant not in VARIANTS:
        raise ValueError('unregistered variant')
    return f'artifacts/experiments/EXP139_AUG1_VARIANTS/{variant}/protocol-v1'


def screen_dir(variant):
    protocol_dir(variant)
    return f'artifacts/experiments/EXP139_AUG1_VARIANTS/{variant}/screen'


def tag_for(variant, seed, fold=0, mode='full'):
    protocol_dir(variant)
    if seed not in SEEDS or fold != 0 or mode not in ('full', 'smoke'):
        raise ValueError('only registered seeds 42/43 and fold 0 are executable')
    return f'EXP139_AUG1_{variant}_S{seed}_F0' + ('_SMOKE' if mode == 'smoke' else '')


def manifest_path(variant, seed, root=ROOT):
    tag_for(variant, seed)
    return Path(root) / protocol_dir(variant) / f's{seed}' / 'manifest.json'


def expected_args(base, variant):
    args = dict(base['common_args'])
    if variant not in VARIANTS:
        raise ValueError('unregistered variant')
    return args


def adapter_registration(variant, seed):
    """Only tiny CPU head tensors; no backbone, weight files, network or CUDA."""
    src = str(ROOT / 'training/src')
    if src not in sys.path:
        sys.path.insert(0, src)
    from raptor_exp129 import initial_adapter_state
    digest, keys = initial_adapter_state(variant, seed)
    return dict(combined_sha256=digest, keys=keys, seed=seed,
                rule='imported EXP129 adapter only; base state remains exact matched EXP127 AUG1')


def screen_spec(root=ROOT):
    return dict(primary='k36', diagnostic='k24', seeds=list(SEEDS), fold=0,
        draws=DRAWS, bootstrap_seed=BOOT_SEED,
        bindings={p: sha256(Path(root) / p) for p in [*LABEL_PATHS.values(), COMPONENTS, PARTNER]},
        averaging='mean of per-seed macro AUCs; NEVER AUC of mean predictions',
        bootstrap='paired components, same draw indices for both arms AND both seeds; study uncertainty only',
        consensus='finite four-source >=0.5 agreement: dread, steven_v4, pilkwang_v2, relabel_v1; nanmean over targets with both classes',
        contrasts={v: [list(x) for x in contrast_pairs(v)] for v in VARIANTS},
        tier_a=dict(dread_min=.004, dread_ci_low_strictly_above=0., consensus_min=0.),
        tier_b=dict(dread_min=.001, positive_draw_share_min=.90, other_reference_min=-.001,
                    same_sign_second_seed=True, required_folds=5, max_extra_serving_fraction=.10,
                    same_family=True, no_extra_submission=True, positive_other_references_min=2,
                    max_bundle_changes=3, cumulative_serving_fraction=True, joint_bundle_evaluation=True),
        partner=dict(path=PARTNER, primary_variant_weight=.20, sensitivity_variant_weights=[.15, .30],
            ranks='average-tie rank/N once on exact shared F0 V888; fixed blended scores resampled, no re-ranking within draws',
            reference='dread', anchor_eval='CoAtNet5 K36 for both RadR50 K36 and K24',
            report_only=True, scope='F0 subset of 4343-study OOF; pooled 4343 partner confirmation remains required',
            rule='report only during screens; tie-breaker only between later Tier A/B passes; negative delta flags, never blocks'),
        interpretation='F0 exposed V888 screen only. Both tiers are provisional; Tier B cannot adopt without pooled five-fold evidence. '
            'No automatic discard for an inconclusive interval/small gain. No submission or fold-expansion authorization.')


def derive_manifest(base, variant, seed, bindings, spec, root=ROOT):
    p = copy.deepcopy(base)
    args = expected_args(base, variant)
    fold = {**copy.deepcopy(base['folds']['0']), **fold_contract(base['folds']['0']['n_train'], args),
            'tag': tag_for(variant, seed)}
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        plan_item='EXP139 delegated by B after 2026-10-08 17:00 direction review',
        variant=variant, variant_config=copy.deepcopy(CONFIG[variant]),
        augmentation_config=copy.deepcopy(e127.CONFIG['AUG1']),
        source_bindings={**base['source_bindings'], **bindings}, common_args=args,
        folds={'0': fold}, total_steps=fold['steps'],
        tags=dict(prefix=f'EXP139_AUG1_{variant}_S{seed}', screen_dir=screen_dir(variant)),
        local={**copy.deepcopy(base['local']), 'runner': RUNNER, 'guard': TOOL,
               'shared_runner': e125.SHARED_RUNNER,
               'runner_rule': 'new minimal copy; frozen EXP107 optimizer/train/eval loop; static ordered equivalence test',
               'queue': 'F0 only, serial seeds, continue_on_failure true; thread values set by entry before imports',
               'comparison': 'AUG1 + variant minus same-seed EXP127_AUG1 local F0 control',
               'estimate': {**ESTIMATES[variant], 'basis': ESTIMATE_BASIS}},
        evaluation={**base['evaluation'], 'exp139_screen': copy.deepcopy(spec)},
        derived_from=dict(path=display_path(control_path(seed, root), root), sha256=sha256(control_path(seed, root)),
            changed_args={k: dict(before=base['common_args'][k], after=v) for k, v in args.items() if v != base['common_args'][k]},
            changed_fields=sorted(CHANGED_FIELDS), scope='F0 only; other folds require a new prospective protocol'),
        label_sets=dict(default=LABEL_SET, allowed=[LABEL_SET]),
        adapter_initial_state=adapter_registration(variant, seed))
    assert_single_change(p, base)
    return p


def assert_single_change(p, base):
    variant, seed = p['variant'], p['seed']
    if base['variant'] != 'AUG1' or base['variant_config'] != e127.CONFIG['AUG1']:
        raise ValueError('control must be the frozen EXP127 AUG1 recipe')
    if p['augmentation_config'] != base['variant_config']:
        raise ValueError('AUG1 augmentation stack changed')
    if p['common_args'] != expected_args(base, variant) or p['variant_config'] != CONFIG[variant]:
        raise ValueError('recipe differs beyond registered single intervention')
    if p['adapter_initial_state'] != adapter_registration(variant, seed):
        raise ValueError('adapter initial state differs from seed registration')
    for key in set(p) | set(base):
        if key not in CHANGED_FIELDS and p.get(key) != base.get(key):
            raise ValueError('matched control field changed: ' + key)
    expected = {**base['folds']['0'], **fold_contract(base['folds']['0']['n_train'], p['common_args']),
                'tag': tag_for(variant, seed)}
    if p['folds'] != {'0': expected} or p['total_steps'] != expected['steps']:
        raise ValueError('fold contract changed beyond duration/scope/tag')


def check_control_assets(base, seed, root):
    check_bindings(root, base['source_bindings'])
    checks = {'training/src/' + k: v for k, v in base['training_code_sha256'].items()}
    checks.update({base['label_set']['path']: base['label_set']['sha256'],
        base['paths']['corpus'] + '/manifest.json': base['corpus_manifest_sha256'],
        base['fold_protocol']['path']: base['fold_protocol']['sha256'],
        base['paths']['folds']: base['fold_protocol']['folds_csv_sha256'],
        WEIGHTS_LOCAL: base['init_weights']['sha256']})
    for f in base['folds'].values():
        checks.update({f[k]['path']: f[k]['sha256'] for k in ('ordered_ids', 'groups')})
        ordered = read(Path(root) / f['ordered_ids']['path'])
        if any(uid_hash(ordered[k]) != f['ordered_ids'][k + '_sha256'] for k in ('train', 'val')):
            raise ValueError('control ordered IDs changed')
    check_bindings(root, checks)
    keys = read(Path(root) / base['initial_state']['keys_file'])
    if keys['combined_sha256'] != base['initial_state_sha256'] or keys['seed'] != seed:
        raise ValueError('registered initial state mismatch')


def prepare(variant, root=ROOT):
    root = Path(root)
    out = root / protocol_dir(variant)
    if out.exists():
        raise SystemExit('Refusing existing registration: ' + str(out))
    spec = screen_spec(root)
    forbid_cuda_init()
    bindings = {p: sha256(root / p) for p in (*NEW_CODE, *TESTS, PROTOCOL_DOC,
        e125.TOOL, e118.TOOL, e127.TOOL, e127.RUNNER, e129.TOOL, e129.RUNNER)}
    bindings.update(spec['bindings'])
    manifests = {}
    for seed in SEEDS:
        base = base_manifest(seed, root)
        check_control_assets(base, seed, root)
        mp = control_path(seed, root)
        run = root / 'artifacts/experiments' / base['folds']['0']['tag']
        inventory = e127.verify_run(run, base, sha256(mp), root)
        manifests[seed] = derive_manifest(base, variant, seed,
            {**bindings, **inventory, display_path(mp, root): sha256(mp)}, spec, root)
    a, b = manifests.values()
    for key in ('common_args', 'environment', 'paths', 'label_set', 'training_code_sha256'):
        if a[key] != b[key]:
            raise ValueError('seed control mismatch: ' + key)
    for key in ('ordered_ids', 'groups', 'pos_weight', 'n_train', 'n_val', 'steps'):
        if a['folds']['0'][key] != b['folds']['0'][key]:
            raise ValueError('seed fold mismatch: ' + key)
    out.mkdir(parents=True)
    registry = dict(schema=SCHEMA, variant=variant, status='REGISTERED_NO_EXECUTION',
                    screen=spec, estimate={**ESTIMATES[variant], 'basis': ESTIMATE_BASIS}, seeds={})
    for seed, p in manifests.items():
        d = out / f's{seed}'
        d.mkdir()
        dump(d / 'manifest.json', p)
        registry['seeds'][str(seed)] = dict(path=f'{protocol_dir(variant)}/s{seed}/manifest.json', sha256=sha256(d / 'manifest.json'))
    dump(out / 'manifest.json', registry)
    print(json.dumps(dict(status=registry['status'], output=str(out), estimate=registry['estimate']), indent=2))
    return registry


def load_protocol(variant, seed, root=ROOT):
    registry = read(Path(root) / protocol_dir(variant) / 'manifest.json')
    mp = manifest_path(variant, seed, root)
    if registry.get('schema') != SCHEMA or registry.get('variant') != variant or sha256(mp) != registry['seeds'][str(seed)]['sha256']:
        raise ValueError('EXP139 registration changed')
    p = read(mp)
    if p.get('schema') != SCHEMA or p.get('variant') != variant or p['seed'] != seed:
        raise ValueError('wrong EXP139 protocol')
    assert_single_change(p, base_manifest(seed, root))
    if p['evaluation']['exp139_screen'] != registry['screen'] or registry['screen'] != screen_spec(root):
        raise ValueError('screen specification changed')
    check_bindings(root, p['source_bindings'])
    return p


def validate_arguments(args, root):
    root = Path(root)
    p = load_protocol(args.variant, args.seed, root)
    if args.fold != 0 or Path(args.protocol).resolve() != manifest_path(args.variant, args.seed, root).resolve():
        raise ValueError('wrong protocol or non-F0 scope')
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol refuses KNEE_REMOTE_RELAX_ENV')
    check_control_assets(base_manifest(args.seed, root), args.seed, root)
    smoke = args.mode == 'smoke'
    for key, expected in p['common_args'].items():
        got = getattr(args, key, None)
        cpu_validation = args.validate_only and key == 'device' and got == 'cpu'
        if got != expected and not cpu_validation and not (smoke and got in p['smoke_allowed'].get(key, [])):
            raise ValueError('frozen argument changed: ' + key)
    if args.validate_only and args.device != 'cpu':
        raise ValueError('EXP139 validation must explicitly use --device cpu')
    w = p['resource_args']['workers']
    if not w['min'] <= args.workers <= min(w['max'], 12):
        raise ValueError('workers outside registered range 0-12')
    if args.tag != tag_for(args.variant, args.seed, args.fold, args.mode):
        raise ValueError('tag/seed/variant mismatch')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    if Path(args.init_weights).resolve() != weights_path(root).resolve():
        raise ValueError('local weights path changed')
    for key, value in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(key) != value:
            raise ValueError('registered environment required: ' + key)
    fold = p['folds']['0']
    ordered = read(root / fold['ordered_ids']['path'])
    default = (root / 'artifacts/experiments').resolve()
    out = Path(args.output_root).resolve() if args.output_root else default
    if smoke:
        if not args.output_root or within(out, root / 'artifacts'):
            raise ValueError('smoke output must be outside artifacts')
        if args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke requires a full batch, validation and epoch')
        if args.variant == 'LATE' and args.smoke_epochs < 3:
            raise ValueError('LATE smoke needs at least 3 epochs for last-three averaging')
    elif out != default:
        raise ValueError('full output root changed')
    return p, fold, ordered, out


def runner_argv(p, root=ROOT, workers=LOCAL_WORKERS, python=None, mode='full', output_root=None, smoke=None):
    argv = e113.runner_argv(p, 0, mode, root=root, python=python, workers=workers,
                            output_root=output_root, smoke=smoke, label_set='dread')
    argv[1] = str(Path(root) / RUNNER)
    i = argv.index('--label-set')
    del argv[i:i+2]
    argv = _set_flag(argv, '--protocol', str(manifest_path(p['variant'], p['seed'], root)))
    argv = _set_flag(argv, '--tag', tag_for(p['variant'], p['seed'], mode=mode))
    return argv + ['--variant', p['variant']]


def queue_entries(variant, root=ROOT, after=None, workers=LOCAL_WORKERS):
    if not 0 <= workers <= 12:
        raise ValueError('workers must be 0-12')
    entries, previous = [], after
    for seed in SEEDS:
        p = load_protocol(variant, seed, root)
        item = dict(tag=tag_for(variant, seed), command=runner_argv(p, root, workers), continue_on_failure=True)
        if previous:
            item['after'] = previous
        entries.append(item)
        previous = item['tag']
    return entries


def forbid_cuda_init():
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU command started after CUDA initialization')
    def make_fail():
        def fail(*args, **kwargs):
            raise RuntimeError('EXP139 CPU command must never initialize CUDA')
        return fail
    for name in ('_lazy_init', 'init'):
        setattr(torch.cuda, name, make_fail())


def load_label_csv(path, columns=None):
    """EXP107 semantics with usecols restriction: never materialize report text."""
    import numpy as np
    import pandas as pd
    wanted = {UID, *T, *(t + '_soft' for t in T), *(columns or [])}
    df = pd.read_csv(path, dtype={UID: str}, usecols=lambda c: c in wanted)
    if df[UID].isna().any() or df[UID].duplicated().any():
        raise ValueError('missing/duplicate label UID')
    cols = columns or [c if c in df.columns else c + '_soft' for c in T]
    if not set(cols).issubset(df.columns):
        raise ValueError('missing label targets')
    df = df.dropna(subset=cols)
    arr = df[cols].to_numpy(dtype=np.float32)
    if arr.min() < 0 or arr.max() > 1 or not np.isfinite(arr).all():
        raise ValueError('invalid label probabilities')
    digest = hashlib.sha256(df[[UID] + cols].to_csv(index=False, lineterminator='\n').encode('utf-8')).hexdigest()
    return dict(zip(df[UID].astype(str), arr)), digest


def validate_cli(variant, seed=None, root=ROOT):
    # Both seeds by default, and use this interpreter (the user specifies it).
    for s in SEEDS if seed is None else (seed,):
        p = load_protocol(variant, s, root)
        argv = _set_flag(runner_argv(p, root=root, workers=0, python=sys.executable), '--device', 'cpu') + ['--validate-only']
        rc = subprocess.run(argv, cwd=root, env=e113.local_env(), check=False).returncode
        if rc:
            return rc
    return 0


def verify_run(run, p, protocol_sha, root=ROOT, candidate=True):
    import numpy as np
    run = Path(run)
    if not candidate:
        return e127.verify_run(run, p, protocol_sha, root)
    f = p['folds']['0']
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv'}
    if p['variant'] == 'LATE':
        required |= set(late_files())
    inv = read(run / 'output-inventory.json')
    if not required.issubset(inv):
        raise ValueError('incomplete run inventory')
    for name, rec in inv.items():
        if Path(name).name != name or sha256(run / name) != rec['sha256']:
            raise ValueError('run inventory mismatch: ' + name)
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expected = dict(fold=0, seed=p['seed'], training_mode='full', checkpoint_selection='fixed-final',
        protocol_sha256=protocol_sha, label_set='dread', label_file_sha256=p['label_set']['sha256'],
        label_semantic_sha256=p['label_set']['semantic_sha256'], arch=p['common_args']['arch'],
        train_ids_ordered_sha256=f['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=f['ordered_ids']['val_sha256'],
        initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True,
        environment_matches_baseline=True, remote_relaxed=False, input_policy=p['common_args']['input_policy'], norm='radimagenet',
        runner_sha256=p['source_bindings'][RUNNER], guard_sha256=p['source_bindings'][TOOL],
        k_eval=24, k_eval_extra=36, exp139_variant=p['variant'], exp139_variant_config=p['variant_config'],
        exp139_code_sha256={n: p['source_bindings'][n] for n in NEW_CODE})
    expected['exp139_adapter_initial_state'] = p['adapter_initial_state']
    expected['exp139_augmentation_config'] = p['augmentation_config']
    for key, val in expected.items():
        if prov.get(key) != val:
            raise ValueError('provenance mismatch: ' + key)
    if any(prov.get('environment', {}).get(k) != p['environment'][k] for k in ENV_KEYS):
        raise ValueError('runtime environment mismatch')
    if prov.get('init_weights', {}).get('weights_sha256') != RADIMAGENET_SHA256:
        raise ValueError('initial weights mismatch')
    if not np.array_equal(np.asarray(prov.get('actual_pos_weight'), np.float32), np.asarray(f['pos_weight'], np.float32)):
        raise ValueError('pos_weight mismatch')
    args = s.get('args', {})
    if (any(args.get(k) != v for k, v in p['common_args'].items()) or args.get('seed') != p['seed']
            or args.get('variant') != p['variant'] or args.get('tag') != f['tag']):
        raise ValueError('recorded recipe mismatch')
    if run.name != f['tag'] or s.get('mode') != 'full' or s.get('selected_epoch_1based') != p['common_args']['epochs']:
        raise ValueError('not the registered full fixed-final run')
    if (s.get('train'), s.get('val'), s.get('optimizer_updates')) != (f['n_train'], f['n_val'], f['steps']):
        raise ValueError('counts/updates mismatch')
    if not 0 <= s.get('skipped_updates', 10**9) <= f['max_fp16_skips']:
        raise ValueError('FP16 skip gate failed')
    history = read(run / 'history.json')
    if [h.get('epoch') for h in history] != list(range(p['common_args']['epochs'])) or history != s.get('history'):
        raise ValueError('epoch history mismatch')
    if p['variant'] == 'LATE':
        verify_late_metadata(read(run / 'late.json'), s)
    return {display_path(run / n, root): sha256(run / n) for n in required | {'output-inventory.json'}}


def late_files():
    return ['epoch_18.pt', 'epoch_19.pt', 'epoch_20.pt', 'model_swa.pt', 'model_ema.pt',
            'oof_swa_k24.csv', 'oof_swa_k36.csv', 'oof_ema_k24.csv', 'oof_ema_k36.csv', 'late.json']


def verify_late_metadata(meta, summary):
    # The runner adds these simple audit fields alongside the tracker metadata.
    expected = dict(epochs_1based=[18, 19, 20], ema_decay=.998,
        successful_optimizer_updates=summary['optimizer_updates'] - summary['skipped_updates'],
        bn_recomputed=False, swa_buffers=CONFIG['LATE']['swa_buffers'], ema_buffers=CONFIG['LATE']['ema_buffers'])
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('LATE observer policy/update contract mismatch')
    if not all(meta.get('state_sha256', {}).get(k) for k in ('final', 'swa', 'ema')):
        raise ValueError('LATE state hashes missing')


# Reuse the identical registered paired-bootstrap, Tier A/B and partner code.
compare_seeds = e129.compare_seeds
tier_verdicts = e129.tier_verdicts
partner_proxy = e129.partner_proxy
metric_contrast = e129.metric_contrast
contrast_pairs = e129.contrast_pairs

def screen(variant, root=ROOT):
    root = Path(root)
    out = root / screen_dir(variant)
    if out.exists():
        raise SystemExit('Refusing existing screen output: ' + str(out))
    registry_path = root / protocol_dir(variant) / 'manifest.json'
    registry = read(registry_path)
    spec = registry['screen']
    check_bindings(root, spec['bindings'])
    predictions = {k: {'candidate': {}, 'control': {}} for k in ('k36', 'k24')}
    if variant == 'LATE':
        for pp in predictions.values():
            pp.update(swa={}, ema={})
    bindings, ids = {**spec['bindings'], display_path(registry_path, root): sha256(registry_path)}, None
    for seed in SEEDS:
        p, base = load_protocol(variant, seed, root), base_manifest(seed, root)
        val = read(root / p['folds']['0']['ordered_ids']['path'])['val']
        if len(val) != 888 or len(set(val)) != 888 or (ids is not None and ids != val):
            raise ValueError('screen requires exact shared V888 roster')
        ids = val
        for role, recipe, mp in (('candidate', p, manifest_path(variant, seed, root)),
                                ('control', base, control_path(seed, root))):
            check_bindings(root, recipe['source_bindings'])
            bindings.update(recipe['source_bindings'])
            bindings[display_path(mp, root)] = sha256(mp)
            run = root / 'artifacts/experiments' / recipe['folds']['0']['tag']
            bindings.update(verify_run(run, recipe, sha256(mp), root, candidate=role == 'candidate'))
            for k, name in (('k36', 'oof_k36.csv'), ('k24', 'oof_best.csv')):
                predictions[k][role][seed] = load_table(run / name, ids, prediction=True)
                if role == 'candidate' and variant == 'LATE':
                    for arm in ('swa', 'ema'):
                        predictions[k][arm][seed] = load_table(run / f'oof_{arm}_{k}.csv', ids, prediction=True)
    refs = make_references({n: load_table(root / path, ids) for n, path in LABEL_PATHS.items()})
    draws = component_draws(ids, read(root / COMPONENTS), draws=DRAWS, seed=BOOT_SEED)
    # Frozen pooled OOF file must cover every control fold, then subset in F0 order.
    pooled_ids = [u for f in e127.base_manifest(42, root)['folds'].values()
                  for u in read(root / f['ordered_ids']['path'])['val']]
    if len(pooled_ids) != 4343 or len(set(pooled_ids)) != 4343:
        raise ValueError('partner pooled control roster is not 4343 unique studies')
    anchor_all = load_table(root / PARTNER, pooled_ids, prediction=True)
    indices = {u: i for i, u in enumerate(pooled_ids)}
    anchor = anchor_all[[indices[u] for u in ids]]
    comparisons = {name: {k: metric_contrast(pp[left], pp[right], refs, draws, anchor, k != 'k36')
                         for k, pp in predictions.items()} for name, left, right in contrast_pairs(variant)}
    primary_comparison = contrast_pairs(variant)[0][0]
    metrics = comparisons[primary_comparison]
    body = dict(schema=SCREEN_SCHEMA, status='COMPLETE', variant=variant,
        observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), protocol_sha256=sha256(registry_path), screen_spec=spec,
        rows=len(ids), ordered_ids_sha256=uid_hash(ids), input_bindings=bindings, metrics=metrics, tiers=metrics['k36']['tiers'],
        primary_comparison=primary_comparison, comparisons=comparisons,
        gold58=dict(status='NOT_EVALUATED', reason='F0 OOF only; Gold58 direction check belongs to prospective pooled bundle; never a gate'),
        coverage={n: dict(cells=int(m.sum()), per_target=m.sum(axis=0).tolist()) for n, (_, m) in refs.items()},
        caveats=[spec['interpretation'], CONFIG[variant]['caveat'], spec['bootstrap']])
    check_bindings(root, bindings)
    out.mkdir(parents=True)
    dump(out / 'result.json', body)
    with (out / 'result.md').open('x', encoding='utf-8') as f:
        f.write(screen_markdown(body))
    print(json.dumps(dict(output=str(out), tiers=body['tiers'], k36=metrics['k36']['seed_mean']), indent=2))
    return body


def screen_markdown(body):
    return e129.screen_markdown(body).replace('# EXP129 ', '# EXP139 AUG1 + ', 1)

def screen_all(root=ROOT, out=None):
    """Summary of immutable screens, including missing variants; never selects a winner."""
    root = Path(root)
    lines = ['| variant / contrast | status | Dread delta (K36) | share > 0 | consensus delta | flight delta | relabel delta | partner delta | Tier A F0 | Tier B F0 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for variant in VARIANTS:
        path = root / screen_dir(variant) / 'result.json'
        if not path.exists():
            lines.append(f'| {variant} | NOT_SCREENED | | | | | | | | |')
            continue
        body = read(path)
        registry = root / protocol_dir(variant) / 'manifest.json'
        if (body.get('schema') != SCREEN_SCHEMA or body.get('variant') != variant or body.get('status') != 'COMPLETE'
                or body.get('protocol_sha256') != sha256(registry)):
            raise ValueError('summary refuses mismatched screen: ' + variant)
        check_bindings(root, body['input_bindings'])
        if list(body['comparisons']) != [c[0] for c in contrast_pairs(variant)]:
            raise ValueError('summary refuses missing or extra comparisons')
        for contrast, metrics in body['comparisons'].items():
            m = metrics['k36']
            if tier_verdicts(m['seed_mean'], m['per_seed']) != m['tiers']:
                raise ValueError('summary refuses inconsistent verdict: ' + variant)
            v, tiers = m['seed_mean'], m['tiers']
            lines.append(f"| {variant} / {contrast} | COMPLETE_F0 | {v['dread']['delta']:+.6f} | {v['dread']['positive_draw_share']:.4f} | "
                         f"{v['consensus']['delta']:+.6f} | {v['flight_v4']['delta']:+.6f} | {v['relabel_v1']['delta']:+.6f} | "
                         f"{m['partner']['delta_partner']:+.6f} | {tiers['tier_a']['screen_verdict']} | {tiers['tier_b']['screen_verdict']} |")
    lines += ['', 'F0 only: Tier B remains PENDING_POOLED_FIVE_FOLD. No adoption, multiplicity-adjusted claim, automatic ranking or discard.', '']
    md = '\n'.join(lines)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            f.write(md)
    print(md)
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'validate', 'queue-json', 'screen'):
        p = sub.add_parser(name)
        p.add_argument('--variant', choices=VARIANTS, required=True)
        if name == 'validate':
            p.add_argument('--seed', type=int, choices=SEEDS, help='omit to validate both seeds')
        if name == 'queue-json':
            p.add_argument('--after', default='none')
            p.add_argument('--workers', type=int, default=LOCAL_WORKERS)
            p.add_argument('--out', default='-', help='new file or stdout; never edits an active queue')
    p = sub.add_parser('screen-all')
    p.add_argument('--out', help='new optional Markdown file; otherwise stdout')
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.variant)
    elif a.command == 'validate':
        return validate_cli(a.variant, a.seed)
    elif a.command == 'queue-json':
        entries = queue_entries(a.variant, after=None if a.after.lower() == 'none' else a.after, workers=a.workers)
        if a.out != '-':
            dump(a.out, entries)
        print(json.dumps(entries, indent=2))
        print(json.dumps({**ESTIMATES[a.variant], 'basis': ESTIMATE_BASIS}), file=sys.stderr)
    elif a.command == 'screen':
        screen(a.variant)
    else:
        screen_all(out=a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
