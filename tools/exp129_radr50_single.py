"""EXP129 RadR50 borrowed-method F0 screens: POS, LATE, CTX, LRH, LRL.

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
from tools.exp113_radr50_local import (worker_init, read_rows, install_gold_blocker,
    uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order,
    validate_fold_membership, pos_weight, contract, UID, T, DREAD, KEYS_FILE, FOLDS,
    EXTRA_EVAL, RUN_FILES, fold_contract, state_hashes, mismatched_keys, dump,
    train_python, within, display_path, _set_flag, weights_path, LOCAL_WORKERS,
    RADIMAGENET_SHA256, WEIGHTS_LOCAL)

SCHEMA = 'exp129-radr50-single-v1'
SCREEN_SCHEMA = 'exp129-radr50-single-screen-v1'
TOOL = 'tools/exp129_radr50_single.py'
RUNNER = 'training/runners/exp129_radr50_single.py'
TEST = 'tests/test_exp129_radr50_single.py'
NEW_CODE = (TOOL, RUNNER, 'training/src/raptor_exp129.py', 'training/src/raptor_exp129_late.py')
TESTS = (TEST, 'tests/test_exp129_heads.py', 'tests/test_exp129_late.py')
VARIANTS = ('POS', 'LATE', 'CTX', 'LRH', 'LRL')
PARTNER = 'artifacts/experiments/EXP099_B8_DREAD_OOF/oof_k36.csv'
SEEDS, BASES, LABEL_SET = e125.SEEDS, e125.BASES, 'dread'
DRAWS, BOOT_SEED = 300, 20261003
LABEL_PATHS, REFERENCES, COMPONENTS = e125.LABEL_PATHS, e125.REFERENCES, e125.COMPONENTS
check_bindings, base_manifest = e125.check_bindings, e125.base_manifest
load_table, make_references = e125.load_table, e125.make_references
component_draws, auc_vector = e125.component_draws, e125.auc_vector

CONFIG = {
    'POS': dict(change='slot-relative position projected into MIL attention logits',
        position='(centre - slot_start)/(slot_size - 1); empty -1 centre -> 0',
        projection='Linear(1,12,bias=False), zero initialized; no input/head changes otherwise',
        constant_control='zero positions produce identically zero extra logits for every weight; equals EXP113 path',
        source='H-B-60; D-A-037; D-C-009; D-C-014',
        caveat='cache index is not physical mm; prior CoAtNet position result uncertain with Baker loss'),
    'LATE': dict(change='observers only: retain epochs 18-20, SWA18-20 and final EMA .998',
        swa_epochs=[18, 19, 20], ema_decay=.998,
        swa_buffers='floating buffers including BN running stats arithmetic averaged; integer buffers final epoch',
        ema_buffers='EMA floating parameters after successful optimizer steps; buffers including BN copied latest',
        bn_recomputed=False, trajectory='unchanged 20-epoch run; no additional fit, fine-tune or metric-based selection',
        source='H-B-60; D-A-050; D-C-014',
        caveat='old controls lack retained last-three weights and EMA; one replay per seed is needed to collect them; averaged BN not recalibrated'),
    'CTX': dict(change='zero residual two-layer transformer on normalized window embeddings before MIL attention',
        feature_dim=2048, reduced_width=256, heads=4, layers=2, dropout=.2,
        residual='2048->256, TransformerEncoder x2 (FFN1024), 256->2048 zero output projection',
        rng='seeded adapter constructor with restored RNG; contextual dropout restores RNG before baseline head dropout',
        source='H-B-60; D-C-014',
        caveat='not the complete Goodpjw recipe; extra serving time must be measured; no positions in this arm'),
    'LRH': dict(change='backbone LR x2; head LR unchanged', backbone_lr_factor=2.,
        source='H-B-60; D-A-020', caveat='ResNet34 joint LR/duration result does not establish a RadR50 LR effect'),
    'LRL': dict(change='backbone LR x0.5; head LR unchanged', backbone_lr_factor=.5,
        source='H-B-60; D-A-020', caveat='ResNet34 joint LR/duration result does not establish a RadR50 LR effect'),
}
ESTIMATES = {
    'POS': dict(minutes_per_seed=[26, 32], budget_minutes_per_seed=30, serving='tiny projection; latency unmeasured'),
    'LATE': dict(minutes_per_seed=[28, 40], budget_minutes_per_seed=35, serving='each of final/SWA/EMA individually identical control cost'),
    'CTX': dict(minutes_per_seed=[28, 42], budget_minutes_per_seed=35, serving='two-layer transformer overhead unmeasured'),
    'LRH': dict(minutes_per_seed=[26, 30], budget_minutes_per_seed=28, serving='identical to control'),
    'LRL': dict(minutes_per_seed=[26, 30], budget_minutes_per_seed=28, serving='identical to control'),
}
ESTIMATE_BASIS = ('EXP113 F0 alone 1550s, 1494s training + 56s fixed evaluation. '
    'LATE evaluates three fixed states (about 112s extra), EMA and checkpoint I/O overhead unmeasured. '
    'POS/CTX overhead unmeasured. RTX 4070 Ti SUPER, 12 workers. '
    'Two concurrent jobs reportedly take 1.6-2x longer EACH; this family emits serial jobs only.')
CHANGED_FIELDS = {'schema', 'created_at', 'plan_item', 'source_bindings', 'common_args', 'folds',
    'tags', 'local', 'evaluation', 'derived_from', 'label_sets', 'variant', 'variant_config', 'total_steps',
    'adapter_initial_state'}


def protocol_dir(variant):
    if variant not in VARIANTS:
        raise ValueError('unregistered variant')
    return f'artifacts/experiments/EXP129_{variant}_PROTOCOL_V1'


def screen_dir(variant):
    protocol_dir(variant)
    return f'artifacts/experiments/EXP129_{variant}_SCREEN'


def tag_for(variant, seed, fold=0, mode='full'):
    protocol_dir(variant)
    if seed not in SEEDS or fold != 0 or mode not in ('full', 'smoke'):
        raise ValueError('only registered seeds 42/43 and fold 0 are executable')
    return f'EXP129_{variant}_S{seed}_F0' + ('_SMOKE' if mode == 'smoke' else '')


def manifest_path(variant, seed, root=ROOT):
    tag_for(variant, seed)
    return Path(root) / protocol_dir(variant) / f's{seed}' / 'manifest.json'


def expected_args(base, variant):
    args = dict(base['common_args'])
    if variant in ('LRH', 'LRL'):
        args['bb_lr'] *= CONFIG[variant]['backbone_lr_factor']
    elif variant not in VARIANTS:
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
                rule='new adapter only; base state remains exact matched EXP113/EXP118')


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


def derive_manifest(base, variant, seed, bindings, spec):
    p = copy.deepcopy(base)
    args = expected_args(base, variant)
    fold = {**copy.deepcopy(base['folds']['0']), **fold_contract(base['folds']['0']['n_train'], args),
            'tag': tag_for(variant, seed)}
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        plan_item='H-B-60; EXP129 delegated by B; user 2026-10-08',
        variant=variant, variant_config=copy.deepcopy(CONFIG[variant]),
        source_bindings={**base['source_bindings'], **bindings}, common_args=args,
        folds={'0': fold}, total_steps=fold['steps'],
        tags=dict(prefix=f'EXP129_{variant}_S{seed}', screen_dir=screen_dir(variant)),
        local={**copy.deepcopy(base['local']), 'runner': RUNNER, 'guard': TOOL,
               'shared_runner': e125.SHARED_RUNNER,
               'runner_rule': 'new minimal copy; frozen EXP107 optimizer/train/eval loop; static ordered equivalence test',
               'queue': 'F0 only, serial seeds, continue_on_failure true; thread values set by entry before imports',
               'comparison': 'variant minus same-seed EXP113/EXP118 local K12 control',
               'estimate': {**ESTIMATES[variant], 'basis': ESTIMATE_BASIS}},
        evaluation={**base['evaluation'], 'exp129_screen': copy.deepcopy(spec)},
        derived_from=dict(path=BASES[seed][0] + '/manifest.json', sha256=BASES[seed][1],
            changed_args={k: dict(before=base['common_args'][k], after=v) for k, v in args.items() if v != base['common_args'][k]},
            changed_fields=sorted(CHANGED_FIELDS), scope='F0 only; other folds require a new prospective protocol'),
        label_sets=dict(default=LABEL_SET, allowed=[LABEL_SET]),
        adapter_initial_state=adapter_registration(variant, seed))
    assert_single_change(p, base)
    return p


def assert_single_change(p, base):
    variant, seed = p['variant'], p['seed']
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
    bindings = {p: sha256(root / p) for p in (*NEW_CODE, *TESTS, e125.TOOL, e118.TOOL)}
    bindings.update(spec['bindings'])
    manifests = {}
    for seed in SEEDS:
        base = base_manifest(seed, root)
        check_control_assets(base, seed, root)
        manifests[seed] = derive_manifest(base, variant, seed,
            {**bindings, BASES[seed][0] + '/manifest.json': BASES[seed][1]}, spec)
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
        raise ValueError('EXP129 registration changed')
    p = read(mp)
    if p.get('schema') != SCHEMA or p.get('variant') != variant or p['seed'] != seed:
        raise ValueError('wrong EXP129 protocol')
    assert_single_change(p, base_manifest(seed, root))
    if p['evaluation']['exp129_screen'] != registry['screen'] or registry['screen'] != screen_spec(root):
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
        raise ValueError('EXP129 validation must explicitly use --device cpu')
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
    def fail(*args, **kwargs):
        raise RuntimeError('EXP129 CPU command must never initialize CUDA')
    torch.cuda._lazy_init = fail
    torch.cuda.init = fail


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


def validate_cli(variant, seed):
    p = load_protocol(variant, seed)
    argv = _set_flag(runner_argv(p, workers=0), '--device', 'cpu') + ['--validate-only']
    return subprocess.run(argv, cwd=ROOT, env=e113.local_env(), check=False).returncode


def verify_run(run, p, protocol_sha, root=ROOT, candidate=True):
    import numpy as np
    run = Path(run)
    if not candidate:
        return e125.verify_run(run, p, protocol_sha, e113.TOOL if p['seed'] == 42 else e118.TOOL, root)
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
        k_eval=24, k_eval_extra=36, exp129_variant=p['variant'], exp129_variant_config=p['variant_config'],
        exp129_code_sha256={n: p['source_bindings'][n] for n in NEW_CODE})
    expected['exp129_adapter_initial_state'] = p['adapter_initial_state']
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


def compare_seeds(candidates, controls, ref, draws):
    """EXP125 arithmetic plus positive-draw shares; strict >0, ties are not wins."""
    import numpy as np
    result, boots = {}, []
    def boot_summary(ds):
        finite = np.isfinite(ds)
        if not finite.any():
            raise ValueError('no finite bootstrap draws')
        return dict(ci95=np.percentile(ds[finite], [2.5, 97.5]).tolist(),
            positive_draw_share=float(np.count_nonzero(ds > 0) / len(ds)),
            bootstrap_finite_draws=int(finite.sum()), bootstrap_draws=len(ds))
    for seed in SEEDS:
        a, b = auc_vector(candidates[seed], ref), auc_vector(controls[seed], ref)
        if not np.isfinite(a).any() or not np.array_equal(np.isfinite(a), np.isfinite(b)):
            raise ValueError('no comparable targets')
        ds = np.asarray([np.nanmean(auc_vector(candidates[seed], ref, ii)) -
                         np.nanmean(auc_vector(controls[seed], ref, ii)) for ii in draws])
        boots.append(ds)
        result[str(seed)] = dict(candidate=float(np.nanmean(a)), control=float(np.nanmean(b)),
            delta=float(np.nanmean(a) - np.nanmean(b)), **boot_summary(ds),
            scored_targets=int(np.isfinite(a).sum()),
            target_deltas={t: float(v) if np.isfinite(v) else None for t, v in zip(T, a-b)})
    avg = {k: float(np.mean([result[str(s)][k] for s in SEEDS])) for k in ('candidate', 'control', 'delta')}
    avg.update(**boot_summary(np.mean(boots, axis=0)),
        target_deltas={t: float(np.mean([result[str(s)]['target_deltas'][t] for s in SEEDS]))
                       if result['42']['target_deltas'][t] is not None else None for t in T})
    spread = {k: abs(result['43'][k] - result['42'][k]) for k in ('candidate', 'control')}
    return result, avg, spread


def tier_verdicts(mean, per_seed):
    valid = (set(REFERENCES).issubset(mean) and all(
        math.isfinite(v) for n in REFERENCES for v in [mean[n]['delta'], *mean[n]['ci95'], mean[n]['positive_draw_share']])
        and all(mean[n]['bootstrap_finite_draws'] == mean[n]['bootstrap_draws'] for n in REFERENCES))
    if not valid:
        return {t: dict(verdict='INVALID', screen_verdict='INVALID', conditions={}) for t in ('tier_a', 'tier_b')}
    a = dict(dread_delta=mean['dread']['delta'] >= .004, dread_ci=mean['dread']['ci95'][0] > 0,
             consensus=mean['consensus']['delta'] >= 0)
    b = dict(dread_delta=mean['dread']['delta'] >= .001, positive_draw_share=mean['dread']['positive_draw_share'] >= .90,
             other_references=all(mean[n]['delta'] >= -.001 for n in REFERENCES if n != 'dread'),
             at_least_two_other_references_positive=sum(mean[n]['delta'] > 0 for n in REFERENCES if n != 'dread') >= 2,
             second_seed_same_positive_sign=all(per_seed[str(s)]['dread']['delta'] > 0 for s in SEEDS))
    result = {}
    for name, cond in (('tier_a', a), ('tier_b', b)):
        result[name] = dict(verdict='PROVISIONAL_F0_ONLY' if name == 'tier_a' else 'PENDING_POOLED_FIVE_FOLD',
            screen_verdict='THRESHOLDS_MET' if all(cond.values()) else 'THRESHOLDS_NOT_MET', conditions=cond,
            adoption_authorized=False)
    result['tier_b']['remaining_requirements'] = ['pooled five-fold paired evidence', 'no extra submission / bundle only',
        'measured cumulative <=10% extra serving time; POS/CTX unmeasured',
        'at most three changes per bundle; frozen joint parent comparison and public reading rule',
        'Gold58 direction diagnostic only (not available in this F0 screen)',
        'new nested or untouched confirmation if an earlier promotion bar was failed']
    return result


def partner_proxy(candidates, controls, anchor, ref, draws):
    """Fixed F0 ranks then paired component bootstrap; never tune blend weights."""
    import numpy as np
    from scipy.stats import rankdata, spearmanr
    if not np.isfinite(anchor).all() or anchor.shape != candidates[SEEDS[0]].shape:
        raise ValueError('invalid partner predictions')
    ranked_anchor = rankdata(anchor, axis=0, method='average') / len(anchor)
    ranks = {role: {s: rankdata(arr[s], axis=0, method='average') / len(anchor) for s in SEEDS}
             for role, arr in (('candidate', candidates), ('control', controls))}
    contrasts = {}
    for weight in (.20, .15, .30):
        arms = {role: {s: (1 - weight) * ranked_anchor + weight * ranks[role][s] for s in SEEDS}
                for role in ranks}
        per_seed, mean, spread = compare_seeds(arms['candidate'], arms['control'], ref, draws)
        contrasts[f'{weight:.2f}'] = dict(per_seed=per_seed, seed_mean=mean, seed_spread=spread,
            role='primary_report_only' if weight == .20 else 'sensitivity_only')
    correlations = {}
    for role, arr in (('candidate', candidates), ('control', controls)):
        correlations[role] = {}
        for seed in SEEDS:
            values = [float(spearmanr(arr[seed][:, j], anchor[:, j]).statistic)
                      if np.ptp(arr[seed][:, j]) > 0 and np.ptp(anchor[:, j]) > 0 else float('nan')
                      for j in range(len(T))]
            finite = [v for v in values if math.isfinite(v)]
            correlations[role][str(seed)] = dict(mean=float(np.mean(finite)) if finite else None,
                finite_targets=len(finite), per_target={t: v if math.isfinite(v) else None for t, v in zip(T, values)})
        means = [correlations[role][str(s)]['mean'] for s in SEEDS]
        correlations[role]['seed_mean'] = float(np.mean(means)) if all(v is not None for v in means) else None
    return dict(report_only=True, primary_weight=.20, comparisons=contrasts, spearman=correlations,
                delta_partner=contrasts['0.20']['seed_mean']['delta'])


def metric_contrast(candidates, controls, refs, draws, anchor, diagnostic=False):
    per_seed, mean, spread = {str(s): {} for s in SEEDS}, {}, {}
    for n, ref in refs.items():
        individual, mean[n], spread[n] = compare_seeds(candidates, controls, ref, draws)
        for seed in SEEDS:
            per_seed[str(seed)][n] = individual[str(seed)]
    partner = partner_proxy(candidates, controls, anchor, refs['dread'], draws)
    return dict(per_seed=per_seed, seed_mean=mean, seed_spread=spread,
        tiers=tier_verdicts(mean, per_seed), diagnostic=diagnostic, partner=partner,
        per_seed_tiers={str(s): tier_verdicts(per_seed[str(s)], per_seed) for s in SEEDS},
        partner_discordance_flag=mean['dread']['delta'] > 0 and partner['delta_partner'] < 0)


def contrast_pairs(variant):
    if variant == 'LATE':
        return [('swa_vs_final', 'swa', 'candidate'), ('ema_vs_final', 'ema', 'candidate'),
                ('swa_vs_control', 'swa', 'control'), ('ema_vs_control', 'ema', 'control'),
                ('final_vs_control', 'candidate', 'control')]
    return [('variant_vs_control', 'candidate', 'control')]


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
                                ('control', base, root / BASES[seed][0] / 'manifest.json')):
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
    pooled_ids = [u for f in base_manifest(42, root)['folds'].values()
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
    lines = [f"# EXP129 {body['variant']} matched local F0 screen", '', body['screen_spec']['interpretation'], '',
             body['screen_spec']['averaging'], '']
    for contrast, metrics in body['comparisons'].items():
        lines += ['', '## ' + contrast, '']
        for k, m in metrics.items():
            lines += ['| eval | scope | reference | candidate | control | delta | CI95 | share delta > 0 |',
                      '|---|---|---|---:|---:|---:|---|---:|']
            for scope, values in [*m['per_seed'].items(), ('seed_mean', m['seed_mean'])]:
                for n, v in values.items():
                    lines.append(f"| {k} | {scope} | {n} | {v['candidate']:.6f} | {v['control']:.6f} | {v['delta']:+.6f} | {v['ci95']} | {v['positive_draw_share']:.4f} |")
            lines += ['', f"{k} seed-mean tiers: " + json.dumps(m['tiers'], sort_keys=True), '',
                      f"{k} per-seed provisional tiers: " + json.dumps(m['per_seed_tiers'], sort_keys=True), '',
                      f"{k} partner delta (report only): {m['partner']['delta_partner']:+.6f}; discordance flag {m['partner_discordance_flag']}", '',
                      '| target | candidate Spearman s42 / s43 | control Spearman s42 / s43 | partner delta s42 / s43 / mean |',
                      '|---|---|---|---|']
            partner = m['partner']
            for t in T:
                rho = partner['spearman']
                p = partner['comparisons']['0.20']
                lines.append(f"| {t} | {rho['candidate']['42']['per_target'][t]} / {rho['candidate']['43']['per_target'][t]} | "
                    f"{rho['control']['42']['per_target'][t]} / {rho['control']['43']['per_target'][t]} | "
                    f"{p['per_seed']['42']['target_deltas'][t]} / {p['per_seed']['43']['target_deltas'][t]} / {p['seed_mean']['target_deltas'][t]} |")
            lines += ['', 'Partner bootstrap / fixed .15 and .30 sensitivities: ' + json.dumps(partner['comparisons']), '',
                      'Mean per-target Spearman: ' + json.dumps({r: {s: v['mean'] if isinstance(v, dict) else v for s, v in d.items()}
                                                             for r, d in partner['spearman'].items()}), '']
    return '\n'.join(lines) + '\n'


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
            p.add_argument('--seed', type=int, choices=SEEDS, required=True)
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
