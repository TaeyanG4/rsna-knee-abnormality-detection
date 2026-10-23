"""EXP127 RadR50 single-change F0 screens: ADJ3, EP40, AUG1 (in that order).

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

SCHEMA = 'exp127-radr50-single-v1'
SCREEN_SCHEMA = 'exp127-radr50-single-screen-v1'
TOOL = 'tools/exp127_radr50_single.py'
RUNNER = 'training/runners/exp127_radr50_single.py'
TEST = 'tests/test_exp127_radr50_single.py'
NEW_CODE = (TOOL, RUNNER, 'training/src/raptor_exp127.py',
            'training/src/raptor_exp127_adj3.py', 'training/src/raptor_exp127_aug1.py')
VARIANTS = ('ADJ3', 'EP40', 'AUG1')
SEEDS, BASES, LABEL_SET = e125.SEEDS, e125.BASES, 'dread'
DRAWS, BOOT_SEED = 300, 20261003
LABEL_PATHS, REFERENCES, COMPONENTS = e125.LABEL_PATHS, e125.REFERENCES, e125.COMPONENTS
check_bindings, base_manifest = e125.check_bindings, e125.base_manifest
load_table, make_references = e125.load_table, e125.make_references
component_draws, auc_vector = e125.component_draws, e125.auc_vector

CONFIG = {
    'ADJ3': dict(change='input_policy center3 -> adjacent3', input_policy='adjacent3',
        channels=['previous cached slot', 'centre cached slot', 'next cached slot'],
        neighbours='same series slot; clamp at slot edge; invalid neighbour repeats valid centre; -1 is empty',
        sampling='unchanged within-slot-valid centres and streams', normalization='unchanged 2x-1',
        source='user 2026-10-08 12:00 KST; D-B-013 topic 737597 +.018 self-report, different pipeline',
        caveat='cached adjacency, NOT EXP108 native-DICOM adjacency (D-B-028); D-B-013 also reports centre3 +.020 on RadImageNet'),
    'EP40': dict(change='20 -> 40 fixed epochs; OneCycle rebuilt over all 40 epochs',
        epochs=40, fixed_epoch=40, checkpoint_selection='fixed-final',
        source='user 2026-10-08 12:00 KST; independent duration hypothesis',
        caveat='D-B-013 update 2026-10-07 21:41 withdrew unsourced 25->50 epochs +.004; not supporting evidence'),
    'AUG1': dict(change='one training augmentation stack after existing brightness, before normalization',
        rotation_degrees=[-10., 10.], scale=[.9, 1.1], translation_fraction=[-.05, .05],
        gamma=[.8, 1.25], horizontal_flip_probability=.5,
        horizontal_flip_slots=[32, 63], sagittal_flip=False,
        affine_interpolation='bilinear', affine_fill=0., transform_order=['brightness', 'affine', 'gamma', 'flip', 'normalise'],
        rng='existing study_rng(seed,epoch,uid,windows), continued after centres; fixed six draws/window',
        evaluation='unchanged center3, no augmentations, no TTA',
        source='user 2026-10-08 12:00 KST; D-B-036; D-B-056 PENDING',
        caveat='stack screen cannot attribute gain to a component; D-B-043 CHALLENGED compact whole-volume proxy is not a RadR50 conclusion; anatomical orientation unverified'),
}
ESTIMATES = {
    'ADJ3': dict(minutes_per_seed=[26, 35], budget_minutes_per_seed=30, serving='same CNN work; cache triplet resize cost unmeasured'),
    'EP40': dict(minutes_per_seed=[48, 58], budget_minutes_per_seed=52, serving='identical to matched control'),
    'AUG1': dict(minutes_per_seed=[28, 43], budget_minutes_per_seed=35, serving='identical to matched control'),
}
ESTIMATE_BASIS = ('EXP113 F0 alone 1550s, 1494s training + 56s fixed evaluation: '
    'EP40 ~1494*2+56=3044s. ADJ3/AUG1 overhead unmeasured. RTX 4070 Ti SUPER, 12 workers. '
    'Two concurrent jobs reportedly take 1.6-2x longer EACH; this family emits serial jobs only.')
CHANGED_FIELDS = {'schema', 'created_at', 'plan_item', 'source_bindings', 'common_args', 'folds',
    'tags', 'local', 'evaluation', 'derived_from', 'label_sets', 'variant', 'variant_config', 'total_steps'}


def protocol_dir(variant):
    if variant not in VARIANTS:
        raise ValueError('unregistered variant')
    return f'artifacts/experiments/EXP127_{variant}_PROTOCOL_V1'


def screen_dir(variant):
    protocol_dir(variant)
    return f'artifacts/experiments/EXP127_{variant}_SCREEN'


def tag_for(variant, seed, fold=0, mode='full'):
    protocol_dir(variant)
    if seed not in SEEDS or fold != 0 or mode not in ('full', 'smoke'):
        raise ValueError('only registered seeds 42/43 and fold 0 are executable')
    return f'EXP127_{variant}_S{seed}_F0' + ('_SMOKE' if mode == 'smoke' else '')


def manifest_path(variant, seed, root=ROOT):
    tag_for(variant, seed)
    return Path(root) / protocol_dir(variant) / f's{seed}' / 'manifest.json'


def expected_args(base, variant):
    args = dict(base['common_args'])
    if variant == 'ADJ3':
        args['input_policy'] = 'adjacent3'
    elif variant == 'EP40':
        args.update(epochs=40, fixed_epoch=40)
    elif variant != 'AUG1':
        raise ValueError('unregistered variant')
    return args


def screen_spec(root=ROOT):
    return dict(primary='k36', diagnostic='k24', seeds=list(SEEDS), fold=0,
        draws=DRAWS, bootstrap_seed=BOOT_SEED,
        bindings={p: sha256(Path(root) / p) for p in [*LABEL_PATHS.values(), COMPONENTS]},
        averaging='mean of per-seed macro AUCs; NEVER AUC of mean predictions',
        bootstrap='paired components, same draw indices for both arms AND both seeds; study uncertainty only',
        consensus='finite four-source >=0.5 agreement: dread, steven_v4, pilkwang_v2, relabel_v1; nanmean over targets with both classes',
        tier_a=dict(dread_min=.004, dread_ci_low_strictly_above=0., consensus_min=0.),
        tier_b=dict(dread_min=.001, positive_draw_share_min=.90, other_reference_min=-.001,
                    same_sign_second_seed=True, required_folds=5, max_extra_serving_fraction=.10,
                    same_family=True, no_extra_submission=True),
        interpretation='F0 exposed V888 screen only. Both tiers are provisional; Tier B cannot adopt without pooled five-fold evidence. '
            'No automatic discard for an inconclusive interval/small gain. No submission or fold-expansion authorization.')


def derive_manifest(base, variant, seed, bindings, spec):
    p = copy.deepcopy(base)
    args = expected_args(base, variant)
    fold = {**copy.deepcopy(base['folds']['0']), **fold_contract(base['folds']['0']['n_train'], args),
            'tag': tag_for(variant, seed)}
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        plan_item='EXP127 delegated by B; user 2026-10-08 12:00 KST',
        variant=variant, variant_config=copy.deepcopy(CONFIG[variant]),
        source_bindings={**base['source_bindings'], **bindings}, common_args=args,
        folds={'0': fold}, total_steps=fold['steps'],
        tags=dict(prefix=f'EXP127_{variant}_S{seed}', screen_dir=screen_dir(variant)),
        local={**copy.deepcopy(base['local']), 'runner': RUNNER, 'guard': TOOL,
               'shared_runner': e125.SHARED_RUNNER,
               'runner_rule': 'new minimal copy; frozen EXP107 optimizer/train/eval loop; static ordered equivalence test',
               'queue': 'F0 only, serial seeds, continue_on_failure true; thread values set by entry before imports',
               'comparison': 'variant minus same-seed EXP113/EXP118 local K12 control',
               'estimate': {**ESTIMATES[variant], 'basis': ESTIMATE_BASIS}},
        evaluation={**base['evaluation'], 'exp127_screen': copy.deepcopy(spec)},
        derived_from=dict(path=BASES[seed][0] + '/manifest.json', sha256=BASES[seed][1],
            changed_args={k: dict(before=base['common_args'][k], after=v) for k, v in args.items() if v != base['common_args'][k]},
            changed_fields=sorted(CHANGED_FIELDS), scope='F0 only; other folds require a new prospective protocol'),
        label_sets=dict(default=LABEL_SET, allowed=[LABEL_SET]))
    assert_single_change(p, base)
    return p


def assert_single_change(p, base):
    variant, seed = p['variant'], p['seed']
    if p['common_args'] != expected_args(base, variant) or p['variant_config'] != CONFIG[variant]:
        raise ValueError('recipe differs beyond registered single intervention')
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
    bindings = {p: sha256(root / p) for p in (*NEW_CODE, TEST, e125.TOOL, e118.TOOL)}
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
        raise ValueError('EXP127 registration changed')
    p = read(mp)
    if p.get('schema') != SCHEMA or p.get('variant') != variant or p['seed'] != seed:
        raise ValueError('wrong EXP127 protocol')
    assert_single_change(p, base_manifest(seed, root))
    if p['evaluation']['exp127_screen'] != registry['screen'] or registry['screen'] != screen_spec(root):
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
        raise ValueError('EXP127 validation must explicitly use --device cpu')
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
        raise RuntimeError('EXP127 CPU command must never initialize CUDA')
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
        k_eval=24, k_eval_extra=36, exp127_variant=p['variant'], exp127_variant_config=p['variant_config'],
        exp127_code_sha256={n: p['source_bindings'][n] for n in NEW_CODE})
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
    return {display_path(run / n, root): sha256(run / n) for n in required | {'output-inventory.json'}}


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
             second_seed_same_positive_sign=all(per_seed[str(s)]['dread']['delta'] > 0 for s in SEEDS))
    result = {}
    for name, cond in (('tier_a', a), ('tier_b', b)):
        result[name] = dict(verdict='PROVISIONAL_F0_ONLY' if name == 'tier_a' else 'PENDING_POOLED_FIVE_FOLD',
            screen_verdict='THRESHOLDS_MET' if all(cond.values()) else 'THRESHOLDS_NOT_MET', conditions=cond,
            adoption_authorized=False)
    result['tier_b']['remaining_requirements'] = ['pooled five-fold paired evidence', 'no extra submission / bundle only',
        'measured <=10% extra serving time (ADJ3 unmeasured; EP40/AUG1 have identical serving computation)']
    return result


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
    refs = make_references({n: load_table(root / path, ids) for n, path in LABEL_PATHS.items()})
    draws = component_draws(ids, read(root / COMPONENTS), draws=DRAWS, seed=BOOT_SEED)
    metrics = {}
    for k, pp in predictions.items():
        per_seed, mean, spread = {str(s): {} for s in SEEDS}, {}, {}
        for n, ref in refs.items():
            individual, mean[n], spread[n] = compare_seeds(pp['candidate'], pp['control'], ref, draws)
            for seed in SEEDS:
                per_seed[str(seed)][n] = individual[str(seed)]
        metrics[k] = dict(per_seed=per_seed, seed_mean=mean, seed_spread=spread,
                          tiers=tier_verdicts(mean, per_seed), diagnostic=k != 'k36')
    body = dict(schema=SCREEN_SCHEMA, status='COMPLETE', variant=variant,
        observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), protocol_sha256=sha256(registry_path), screen_spec=spec,
        rows=len(ids), ordered_ids_sha256=uid_hash(ids), input_bindings=bindings, metrics=metrics, tiers=metrics['k36']['tiers'],
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
    lines = [f"# EXP127 {body['variant']} matched local F0 screen", '', body['screen_spec']['interpretation'], '',
             body['screen_spec']['averaging'], '',
             '| eval | scope | reference | candidate | control | delta | CI95 | share delta > 0 |',
             '|---|---|---|---:|---:|---:|---|---:|']
    for k, m in body['metrics'].items():
        for scope, values in [*m['per_seed'].items(), ('seed_mean', m['seed_mean'])]:
            for n, v in values.items():
                lines.append(f"| {k} | {scope} | {n} | {v['candidate']:.6f} | {v['control']:.6f} | {v['delta']:+.6f} | {v['ci95']} | {v['positive_draw_share']:.4f} |")
        lines += ['', f"{k} tiers: " + json.dumps(m['tiers'], sort_keys=True), '']
    return '\n'.join(lines) + '\n'


def screen_all(root=ROOT, out=None):
    """Summary of immutable screens, including missing variants; never selects a winner."""
    root = Path(root)
    lines = ['| variant | status | Dread delta (K36) | share > 0 | consensus delta | flight delta | relabel delta | Tier A F0 | Tier B F0 |',
             '|---|---|---:|---:|---:|---:|---:|---|---|']
    for variant in VARIANTS:
        path = root / screen_dir(variant) / 'result.json'
        if not path.exists():
            lines.append(f'| {variant} | NOT_SCREENED | | | | | | | |')
            continue
        body = read(path)
        registry = root / protocol_dir(variant) / 'manifest.json'
        if (body.get('schema') != SCREEN_SCHEMA or body.get('variant') != variant or body.get('status') != 'COMPLETE'
                or body.get('protocol_sha256') != sha256(registry)):
            raise ValueError('summary refuses mismatched screen: ' + variant)
        check_bindings(root, body['input_bindings'])
        m = body['metrics']['k36']
        if tier_verdicts(m['seed_mean'], m['per_seed']) != body['tiers'] or body['tiers'] != m['tiers']:
            raise ValueError('summary refuses inconsistent verdict: ' + variant)
        v, tiers = m['seed_mean'], m['tiers']
        lines.append(f"| {variant} | COMPLETE_F0 | {v['dread']['delta']:+.6f} | {v['dread']['positive_draw_share']:.4f} | "
                     f"{v['consensus']['delta']:+.6f} | {v['flight_v4']['delta']:+.6f} | {v['relabel_v1']['delta']:+.6f} | "
                     f"{tiers['tier_a']['screen_verdict']} | {tiers['tier_b']['screen_verdict']} |")
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
