"""HC20: local paired K36 vs registered EXP113/EXP118 K12, seeds 42/43.

prepare / validate / queue-json / screen; all commands are CPU-only and never
launch training. Queue commands use the unchanged EXP107 training runner via a
new guard entry. F0 is the screen; folds 1-4 require an explicit opt-in AND a
recorded, hash-bound screen PASS. New paths only; existing outputs are refused.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp107_kfold_radr50 as e107
from tools import exp113_radr50_local as e113
from tools import exp118_radr50_local_s43 as e118
# All helpers consumed by the shared training runner are unchanged.
from tools.exp113_radr50_local import (worker_init, read_rows, install_gold_blocker,
    uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order,
    validate_fold_membership, pos_weight, contract, UID, T, DREAD, KEYS_FILE, FOLDS,
    EXTRA_EVAL, RUN_FILES, fold_contract, state_hashes, mismatched_keys, dump,
    train_python, within, display_path, _set_flag, weights_path, LOCAL_WORKERS,
    RADIMAGENET_SHA256, WEIGHTS_LOCAL)

SCHEMA = 'hc20-radr50-k36-local-v1'
SCREEN_SCHEMA = 'hc20-radr50-k36-screen-v1'
PROTOCOL = 'artifacts/experiments/HC20_RADR50_K36_LOCAL_PROTOCOL_V1'
SCREEN_DIR = 'artifacts/experiments/HC20_RADR50_K36_LOCAL_SCREEN'
TOOL = 'tools/hc20_radr50_k36_local_v1.py'
RUNNER = 'training/runners/hc20_radr50_k36_local_v1.py'
TEST = '.local/hc20-local-k36-v1/register_audit.py'
SHARED_RUNNER = e107.RUNNER
SEEDS = (42, 43)
BASES = {42: (e113.PROTOCOL, 'a134327a10bfe051a0d58f3bb83969b2bdb8b9fcc4f54a309893504c960bd31d'),
         43: (e118.PROTOCOL, 'dce43500fa6e38252160e7d8b544c61006599f5ce76ba32af4c93dd16900c659')}
LABEL_SET = 'dread'
ALLOW_PRODUCTION = False
DRAWS, BOOT_SEED = 300, 20261003
REFERENCES = ('dread', 'consensus', 'flight_v4', 'relabel_v1')
LABEL_PATHS = {
    'dread': DREAD,
    'flight_v4': 'storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv',
    'relabel_v1': 'storage/labels/derived_relabel/relabel_v1.csv',
    'steven_v4': 'storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
    'pilkwang_v2': 'storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv',
}
COMPONENTS = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
GATE = dict(dread_min=0.004, dread_ci_low_strictly_above=0.0,
            consensus_min=0.0, flight_v4_min=-0.002, relabel_v1_min=-0.002)
ESTIMATE = dict(seconds_per_job=4800, range_seconds=[4000,6500], basis='Uncontended K12 local1550s scaled3x; admission requires measured K36 smoke. Two serial seeds+pilot+eval hardcap14400s.')
# Every remaining field must equal its matched seed control exactly.
CHANGED_FIELDS = ('schema', 'created_at', 'plan_item', 'source_bindings', 'common_args',
                  'folds', 'tags', 'local', 'evaluation', 'derived_from', 'label_sets')


def tag_for(seed, fold=0, mode='full'):
    if seed not in SEEDS or fold not in FOLDS:
        raise ValueError('unregistered seed/fold')
    return f'HC20_RADR50_K36_LOCAL_S{seed}_F{fold}' + ('_SMOKE' if mode == 'smoke' else '')


def manifest_path(seed, root=ROOT):
    if seed not in SEEDS:
        raise ValueError('unregistered seed')
    return Path(root) / PROTOCOL / f's{seed}' / 'manifest.json'


def check_bindings(root, bindings):
    for path, expected in bindings.items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)


def base_manifest(seed, root=ROOT):
    path, digest = BASES[seed]
    f = Path(root) / path / 'manifest.json'
    if sha256(f) != digest:
        raise ValueError(f'control s{seed} manifest changed')
    p = read(f)
    if p['seed'] != seed or p['common_args'] != e107.EXPECTED_RECIPE or p['label_set']['name'] != LABEL_SET:
        raise ValueError('unexpected matched control')
    return p


def screen_spec(root=ROOT):
    bindings = {p: sha256(Path(root) / p) for p in [*LABEL_PATHS.values(), COMPONENTS]}
    return dict(primary='k36', diagnostic='k24', seeds=list(SEEDS), fold=0,
        draws=DRAWS, bootstrap_seed=BOOT_SEED, gate=GATE, bindings=bindings, retention='Historical EXP125 threshold reported only; positive uncertain gains retained, require seed-sign repeatability before expansion',
        averaging='mean of seed macro AUCs, never AUC of averaged predictions',
        consensus='all four >=0.5 agree and finite: dread, steven_v4, pilkwang_v2, relabel_v1; '
                  'agreed label; at least one positive and negative per target (EXP117 screen)',
        difference_from_colab='local matched controls; 300 draws seed 20261003 instead of 200 seed 0; '
                              'four-source EXP117 consensus instead of Colab three-source Dread/flight/relabel')


def derive_manifest(base, seed, bindings, spec):
    p = copy.deepcopy(base)
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_item='H-C-20',
             source_bindings={**base['source_bindings'], **bindings},
             common_args={**base['common_args'], 'k': 36},
             folds={k: {**copy.deepcopy(v), 'tag': tag_for(seed, int(k))} for k, v in base['folds'].items()},
             tags=dict(prefix=f'HC20_RADR50_K36_LOCAL_S{seed}', screen_dir=SCREEN_DIR),
             local={**copy.deepcopy(base['local']), 'runner': RUNNER, 'guard': TOOL,
                    'shared_runner': SHARED_RUNNER, 'estimate': ESTIMATE,
                    'runner_rule': 'unchanged EXP107 runner, HC20 entry binds new guard',
                    'queue': 'independent seeds, continue_on_failure true; serial by default',
                    'comparison': 'same seed local K36 minus local K12; one changed recipe argument'},
             evaluation={**base['evaluation'], 'hc20_screen': spec},
             derived_from=dict(path=BASES[seed][0] + '/manifest.json', sha256=BASES[seed][1],
                               changed_args={'k': {'from': 12, 'to': 16}}, changed_fields=list(CHANGED_FIELDS)),
             label_sets=dict(default=LABEL_SET, allowed=[LABEL_SET]))
    # Initial state, per-key file, labels, folds, steps, environment all inherited.
    assert_only_k_changed(p, base)
    return p


def assert_only_k_changed(p, base):
    expected = {**base['common_args'], 'k': 36}
    if p['common_args'] != expected:
        raise ValueError('recipe differs beyond k_train 12 -> 36')
    for k in set(p) | set(base):
        if k not in CHANGED_FIELDS and p.get(k) != base.get(k):
            raise ValueError('matched control field changed: ' + k)
    if set(p['folds']) != set(base['folds']):
        raise ValueError('fold set changed')
    for k, f in p['folds'].items():
        if {a: b for a, b in f.items() if a != 'tag'} != {a: b for a, b in base['folds'][k].items() if a != 'tag'}:
            raise ValueError('fold contract changed: ' + k)
        if f['tag'] != tag_for(p['seed'], int(k)):
            raise ValueError('fold tag changed')


def prepare(root=ROOT, dest=None):
    root = Path(root)
    out = Path(dest) if dest else root / PROTOCOL
    if out.exists():
        raise SystemExit('Refusing existing registration: ' + str(out))
    spec = screen_spec(root)
    new_bindings = {p: sha256(root / p) for p in (TOOL, RUNNER, TEST)}
    new_bindings.update(spec['bindings'])
    manifests = {}
    for seed in SEEDS:
        base = base_manifest(seed, root)
        check_bindings(root, base['source_bindings'])
        check_bindings(root, {'training/src/' + k: v for k, v in base['training_code_sha256'].items()})
        checks = {base['label_set']['path']: base['label_set']['sha256'],
                  base['paths']['corpus'] + '/manifest.json': base['corpus_manifest_sha256'],
                  base['fold_protocol']['path']: base['fold_protocol']['sha256'],
                  base['paths']['folds']: base['fold_protocol']['folds_csv_sha256'],
                  WEIGHTS_LOCAL: base['init_weights']['sha256']}
        for f in base['folds'].values():
            checks.update({f[k]['path']: f[k]['sha256'] for k in ('ordered_ids', 'groups')})
            ordered = read(root / f['ordered_ids']['path'])
            if any(uid_hash(ordered[k]) != f['ordered_ids'][k + '_sha256'] for k in ('train', 'val')):
                raise ValueError('control fold UID hash mismatch')
        check_bindings(root, checks)
        keys = read(root / base['initial_state']['keys_file'])
        if keys['combined_sha256'] != base['initial_state_sha256'] or keys['seed'] != seed:
            raise ValueError('registered initial state mismatch')
        manifests[seed] = derive_manifest(base, seed,
            {**new_bindings, BASES[seed][0] + '/manifest.json': BASES[seed][1]}, spec)
    a, b = manifests[42], manifests[43]
    for k in ('common_args', 'environment', 'paths', 'label_set', 'training_code_sha256'):
        if a[k] != b[k]:
            raise ValueError('seed controls incompatible: ' + k)
    for k in a['folds']:
        if {x: v for x, v in a['folds'][k].items() if x != 'tag'} != {x: v for x, v in b['folds'][k].items() if x != 'tag'}:
            raise ValueError('seed controls have different folds')
    out.mkdir(parents=True)
    registry = dict(schema=SCHEMA, status='REGISTERED_NO_EXECUTION', screen=spec, estimate=ESTIMATE, seeds={})
    for seed, p in manifests.items():
        d = out / f's{seed}'
        d.mkdir()
        dump(d / 'manifest.json', p)
        registry['seeds'][str(seed)] = dict(path=f'{PROTOCOL}/s{seed}/manifest.json', sha256=sha256(d / 'manifest.json'))
    dump(out / 'manifest.json', registry)
    print(json.dumps(dict(status=registry['status'], written_to=str(out), estimate=ESTIMATE), indent=2))
    return registry


def load_protocol(seed, root=ROOT):
    registry = read(Path(root) / PROTOCOL / 'manifest.json')
    pth = manifest_path(seed, root)
    if registry.get('schema') != SCHEMA or sha256(pth) != registry['seeds'][str(seed)]['sha256']:
        raise ValueError('HC20 registration changed')
    p = read(pth)
    if p.get('schema') != SCHEMA or p['seed'] != seed:
        raise ValueError('wrong HC20 protocol')
    assert_only_k_changed(p, base_manifest(seed, root))
    check_bindings(root, p['source_bindings'])
    return p


def require_screen_pass(root=ROOT):
    root = Path(root)
    r = read(root / SCREEN_DIR / 'result.json')
    if (r.get('schema') != SCREEN_SCHEMA or r.get('status') != 'COMPLETE' or r.get('gate', {}).get('verdict') != 'PASS'
            or r.get('protocol_sha256') != sha256(root / PROTOCOL / 'manifest.json')
            or r.get('screen_spec') != read(root / PROTOCOL / 'manifest.json')['screen']):
        raise ValueError('no matching recorded HC20 screen PASS')
    if gate(r['metrics']['k36']['seed_mean'])['verdict'] != 'PASS':
        raise ValueError('recorded gate is inconsistent')
    check_bindings(root, r['input_bindings'])


def check_fold(fold, allow=False, root=ROOT):
    if fold != 0:
        raise ValueError('HC20 only fold0 is authorized')
    if fold != 0:
        if not allow:
            raise ValueError('folds 1-4 need --allow-production-after-screen-pass')
        require_screen_pass(root)


def validate_arguments(args, root):
    root = Path(root)
    p = load_protocol(args.seed, root)
    if Path(args.protocol).resolve() != manifest_path(args.seed, root).resolve():
        raise ValueError('wrong protocol path')
    check_fold(args.fold, ALLOW_PRODUCTION, root)
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol refuses KNEE_REMOTE_RELAX_ENV')
    check_bindings(root, {'training/src/' + k: v for k, v in p['training_code_sha256'].items()})
    smoke = args.mode == 'smoke'
    for key, expected in p['common_args'].items():
        got = getattr(args, key, None)
        # Validation is explicitly CPU; shared runner exits before runtime setup.
        cpu_validation = args.validate_only and key == 'device' and got == 'cpu'
        if got != expected and not cpu_validation and not (smoke and got in p['smoke_allowed'].get(key, [])):
            raise ValueError('frozen argument changed: ' + key)
    if args.validate_only and args.device != 'cpu':
        raise ValueError('HC20 validation must explicitly use --device cpu')
    w = p['resource_args']['workers']
    if not w['min'] <= args.workers <= w['max']:
        raise ValueError('workers outside registered range')
    fold = p['folds'][str(args.fold)]
    if args.tag != tag_for(args.seed, args.fold, args.mode):
        raise ValueError('tag/seed/fold/mode mismatch')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    if Path(args.init_weights).resolve() != weights_path(root).resolve() or sha256(weights_path(root)) != RADIMAGENET_SHA256:
        raise ValueError('local RadImageNet weights changed')
    for key, value in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(key) != value:
            raise ValueError('registered environment required: ' + key)
    check_bindings(root, {fold[k]['path']: fold[k]['sha256'] for k in ('ordered_ids', 'groups')})
    ordered = read(root / fold['ordered_ids']['path'])
    if any(uid_hash(ordered[k]) != fold['ordered_ids'][k + '_sha256'] for k in ('train', 'val')):
        raise ValueError('ordered fold ids changed')
    default = (root / 'artifacts/experiments').resolve()
    out = Path(args.output_root).resolve() if args.output_root else default
    if smoke:
        if not args.output_root or within(out, root / 'artifacts'):
            raise ValueError('smoke output must be outside artifacts')
        if args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke needs a full batch, validation and epoch')
    elif out != default:
        raise ValueError('full run output root changed')
    return p, fold, ordered, out


def runner_argv(p, fold=0, mode='full', root=ROOT, workers=LOCAL_WORKERS, python=None,
                output_root=None, smoke=None, allow_production=False):
    check_fold(fold, allow_production, root)
    argv = e113.runner_argv(p, fold, mode, root=root, python=python, workers=workers,
                            output_root=output_root, smoke=smoke, label_set='dread')
    argv[1] = str(Path(root) / RUNNER)
    argv = _set_flag(argv, '--protocol', str(manifest_path(p['seed'], root)))
    argv = _set_flag(argv, '--tag', tag_for(p['seed'], fold, mode))
    if allow_production:
        argv.append('--allow-production-after-screen-pass')
    return argv


def queue_entries(root=ROOT, folds=(0,), after=None, workers=LOCAL_WORKERS,
                  allow_production=False, parallel_seeds=False):
    if not folds or len(set(folds)) != len(folds):
        raise ValueError('empty or duplicate folds')
    entries, previous = [], after
    for seed in SEEDS:
        p = load_protocol(seed, root)
        if parallel_seeds:
            previous = after
        for fold in folds:
            entry = dict(tag=tag_for(seed, fold), command=runner_argv(p, fold, root=root,
                workers=workers, allow_production=allow_production), continue_on_failure=False)
            if previous:
                entry['after'] = previous
            entries.append(entry)
            previous = entry['tag']
    return entries


def validate_cli(seed, fold=0, allow=False):
    p = load_protocol(seed)
    argv = runner_argv(p, fold, workers=0, allow_production=allow)
    argv = _set_flag(argv, '--device', 'cpu') + ['--validate-only']
    result = subprocess.run(argv, cwd=ROOT, env=e113.local_env(), check=False)
    return result.returncode


def verify_run(run, p, protocol_sha, guard_tool, root=ROOT):
    """Accept only complete fixed-final runs, exact provenance and inventory bytes."""
    import numpy as np
    run = Path(run)
    fold = p['folds']['0']
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv'}
    inv = read(run / 'output-inventory.json')
    if not required.issubset(inv):
        raise ValueError('incomplete run inventory: ' + run.name)
    for name, rec in inv.items():
        if Path(name).name != name or sha256(run / name) != rec['sha256']:
            raise ValueError('run inventory mismatch: ' + run.name + '/' + name)
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expected = dict(fold=0, seed=p['seed'], training_mode='full', checkpoint_selection='fixed-final',
        protocol_sha256=protocol_sha, label_set='dread', label_file_sha256=p['label_set']['sha256'],
        label_semantic_sha256=p['label_set']['semantic_sha256'], arch=p['common_args']['arch'],
        train_ids_ordered_sha256=fold['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=fold['ordered_ids']['val_sha256'],
        initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True,
        environment_matches_baseline=True, remote_relaxed=False, input_policy='center3', norm='radimagenet',
        runner_sha256=p['source_bindings'][SHARED_RUNNER], guard_sha256=p['source_bindings'][guard_tool],
        k_eval=24, k_eval_extra=36)
    for key, val in expected.items():
        if prov.get(key) != val:
            raise ValueError(run.name + ': provenance mismatch ' + key)
    if any(prov.get('environment', {}).get(k) != p['environment'][k] for k in ENV_KEYS):
        raise ValueError('environment mismatch: ' + run.name)
    if prov.get('init_weights', {}).get('weights_sha256') != RADIMAGENET_SHA256:
        raise ValueError('initial weights mismatch')
    if not np.array_equal(np.asarray(prov.get('actual_pos_weight'), np.float32), np.asarray(fold['pos_weight'], np.float32)):
        raise ValueError('pos_weight mismatch')
    args = s.get('args', {})
    if any(args.get(k) != v for k, v in p['common_args'].items()) or args.get('seed') != p['seed'] or args.get('tag') != fold['tag']:
        raise ValueError('recorded recipe mismatch: ' + run.name)
    if run.name != fold['tag'] or s.get('mode') != 'full' or s.get('selected_epoch_1based') != p['common_args']['epochs']:
        raise ValueError('not the registered full fixed-final run')
    if (s.get('train'), s.get('val'), s.get('optimizer_updates')) != (fold['n_train'], fold['n_val'], fold['steps']):
        raise ValueError('counts/updates mismatch')
    if not 0 <= s.get('skipped_updates', 10**9) <= fold['max_fp16_skips']:
        raise ValueError('FP16 skip gate failed')
    return {display_path(run / name, root): sha256(run / name) for name in required | {'output-inventory.json'}}


def load_table(path, ids, prediction=False):
    """Read only UID and target columns; never load report text."""
    import numpy as np
    import pandas as pd
    wanted = {UID, *T, *(t + '_soft' for t in T)}
    d = pd.read_csv(path, dtype={UID: str}, usecols=lambda c: c in wanted)
    cols = [t if t in d.columns else t + '_soft' for t in T]
    if d[UID].isna().any() or d[UID].duplicated().any() or not set(cols).issubset(d.columns):
        raise ValueError('invalid UID/target columns: ' + str(path))
    if not set(ids).issubset(d[UID]) or (prediction and (set(d[UID]) != set(ids) or list(d.columns) != [UID, *T])):
        raise ValueError('prediction/label roster mismatch: ' + str(path))
    values = d.set_index(UID).loc[list(ids), cols].to_numpy(np.float64)
    if prediction and (not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError('invalid prediction probabilities')
    return values


def make_references(soft):
    import numpy as np
    refs = {}
    for name in ('dread', 'flight_v4', 'relabel_v1'):
        x = soft[name]
        if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
            raise ValueError('incomplete/non-probability reference: ' + name)
        y = np.rint(x.astype(np.float32)) if name == 'dread' else (x >= 0.5).astype(float)
        refs[name] = (y, np.ones(x.shape, bool))
    x = np.stack([soft[n] for n in ('dread', 'steven_v4', 'pilkwang_v2', 'relabel_v1')])
    valid = np.isfinite(x).all(axis=0)
    b = x >= 0.5
    mask = valid & (b.all(axis=0) | (~b).all(axis=0))
    refs['consensus'] = (b[0].astype(float), mask)
    return {n: refs[n] for n in REFERENCES}


def component_draws(ids, components, draws=DRAWS, seed=BOOT_SEED):
    import numpy as np
    membership = {}
    for c in components:
        for u in c['members']:
            if u in membership:
                raise ValueError('duplicate component membership')
            membership[u] = c['component_id']
    if any(u not in membership for u in ids):
        raise ValueError('missing component membership')
    groups = {}
    for i, u in enumerate(ids):
        groups.setdefault(membership[u], []).append(i)
    arrays = [np.asarray(groups[k], dtype=int) for k in sorted(groups)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([arrays[j] for j in rng.integers(0, len(arrays), len(arrays))]) for _ in range(draws)]


def auc_vector(P, ref, idx=None):
    import numpy as np
    from scipy.stats import rankdata
    Y, M = ref
    if idx is not None:
        P, Y, M = P[idx], Y[idx], M[idx]
    values = []
    for t in range(Y.shape[1]):
        y, p = Y[M[:, t], t], P[M[:, t], t]
        pos, neg = y.sum(), len(y) - y.sum()
        values.append(float((rankdata(p)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else float('nan'))
    return np.asarray(values)


def compare_seeds(candidates, controls, ref, draws):
    """Compute the per-seed contrasts once, reuse draws for their AUC mean."""
    import numpy as np
    result, boots = {}, []
    for seed in SEEDS:
        a, b = auc_vector(candidates[seed], ref), auc_vector(controls[seed], ref)
        if not np.isfinite(a).any() or not np.isfinite(b).any():
            raise ValueError('a reference has no target with both classes')
        ds = np.asarray([np.nanmean(auc_vector(candidates[seed], ref, ii)) -
                         np.nanmean(auc_vector(controls[seed], ref, ii)) for ii in draws])
        boots.append(ds)
        result[str(seed)] = dict(candidate=float(np.nanmean(a)), control=float(np.nanmean(b)),
            delta=float(np.nanmean(a) - np.nanmean(b)), ci95=np.nanpercentile(ds, [2.5, 97.5]).tolist(),
            scored_targets=int(np.isfinite(a).sum()),
            target_deltas={t: float(v) if np.isfinite(v) else None for t, v in zip(T, a-b)})
    ds = np.mean(boots, axis=0)
    avg = {k: float(np.mean([result[str(s)][k] for s in SEEDS])) for k in ('candidate', 'control', 'delta')}
    avg.update(ci95=np.nanpercentile(ds, [2.5, 97.5]).tolist(),
        target_deltas={t: (float(np.mean([result[str(s)]['target_deltas'][t] for s in SEEDS]))
                          if result['42']['target_deltas'][t] is not None else None) for t in T},
        bootstrap_finite_draws=int(np.isfinite(ds).sum()))
    spread = {k: abs(result['43'][k] - result['42'][k]) for k in ('candidate', 'control')}
    return result, avg, spread


def gate(metrics):
    import math
    if not all(n in metrics for n in REFERENCES):
        return dict(verdict='INVALID', conditions={}, reason='references missing')
    if any(not math.isfinite(v) for n in REFERENCES for v in [metrics[n]['delta'], *metrics[n]['ci95']]):
        return dict(verdict='INVALID', conditions={}, reason='nonfinite metric')
    cond = dict(dread=metrics['dread']['delta'] >= GATE['dread_min'] and metrics['dread']['ci95'][0] > 0,
                consensus=metrics['consensus']['delta'] >= GATE['consensus_min'],
                flight_v4=metrics['flight_v4']['delta'] >= GATE['flight_v4_min'],
                relabel_v1=metrics['relabel_v1']['delta'] >= GATE['relabel_v1_min'])
    return dict(verdict='PASS' if all(cond.values()) else 'FAIL', conditions=cond)


def screen(root=ROOT):
    import numpy as np
    root = Path(root)
    out = root / SCREEN_DIR
    if out.exists():
        raise SystemExit('Refusing existing screen output: ' + str(out))
    registry = read(root / PROTOCOL / 'manifest.json')
    spec = registry['screen']
    check_bindings(root, spec['bindings'])
    predictions = {k: {'candidate': {}, 'control': {}} for k in ('k36', 'k24')}
    bindings, ids = dict(spec['bindings']), None
    for seed in SEEDS:
        p, base = load_protocol(seed, root), base_manifest(seed, root)
        val = read(root / p['folds']['0']['ordered_ids']['path'])['val']
        if len(val) != 888 or len(set(val)) != 888 or (ids is not None and ids != val):
            raise ValueError('screen requires the exact shared V888 roster')
        ids = val
        for role, recipe, mp, guard in (('candidate', p, manifest_path(seed, root), TOOL),
                ('control', base, root / BASES[seed][0] / 'manifest.json', e113.TOOL if seed == 42 else e118.TOOL)):
            run = root / 'artifacts/experiments' / recipe['folds']['0']['tag']
            # File digests, provenance, complete training and strict environment before scoring.
            for path, digest in verify_run(run, recipe, sha256(mp), guard, root).items():
                bindings[path] = digest
            for k, name in (('k36', 'oof_k36.csv'), ('k24', 'oof_best.csv')):
                predictions[k][role][seed] = load_table(run / name, ids, prediction=True)
    refs = make_references({n: load_table(root / path, ids) for n, path in LABEL_PATHS.items()})
    draws = component_draws(ids, read(root / COMPONENTS))
    metrics = {}
    for k, p in predictions.items():
        per_seed, mean, spread = {str(s): {} for s in SEEDS}, {}, {}
        for n, ref in refs.items():
            individual, mean[n], spread[n] = compare_seeds(p['candidate'], p['control'], ref, draws)
            for s in SEEDS:
                per_seed[str(s)][n] = individual[str(s)]
        metrics[k] = dict(per_seed=per_seed, seed_mean=mean, seed_spread=spread,
                          gate=gate(mean), diagnostic=(k != 'k36'))
    body = dict(schema=SCREEN_SCHEMA, status='COMPLETE', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        protocol_sha256=sha256(root / PROTOCOL / 'manifest.json'), screen_spec=spec, rows=len(ids),
        ordered_ids_sha256=uid_hash(ids), input_bindings=bindings, metrics=metrics, gate=metrics['k36']['gate'],
        coverage={n: dict(cells=int(m.sum()), per_target=m.sum(axis=0).tolist()) for n, (_, m) in refs.items()},
        caveats=['provisional F0 screen on exposed V888; no public/private improvement established',
                 'component bootstrap covers study sampling, not seed/trajectory uncertainty', spec['difference_from_colab']])
    # Recheck all input bytes before the immutable report can authorize production.
    check_bindings(root, bindings)
    out.mkdir(parents=True)
    dump(out / 'result.json', body)
    lines = ['# HC20 local K36 screen', '', 'K36 primary verdict: ' + body['gate']['verdict'], '',
             spec['averaging'], '', spec['difference_from_colab'], '',
             '| eval | scope | reference | K36 | K12 | delta | CI95 |', '|---|---|---|---:|---:|---:|---|']
    for k, m in metrics.items():
        for scope, values in [*m['per_seed'].items(), ('seed_mean', m['seed_mean'])]:
            for n, v in values.items():
                lines.append(f"| {k} | {scope} | {n} | {v['candidate']:.6f} | {v['control']:.6f} | {v['delta']:+.6f} | {v['ci95']} |")
    with (out / 'result.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(out), gate=body['gate'], k36=metrics['k36']['seed_mean']), indent=2))
    return body


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dry-run', help='new destination outside artifacts; not runnable')
    p = sub.add_parser('validate')
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--fold', type=int, choices=FOLDS, default=0)
    p.add_argument('--allow-production-after-screen-pass', action='store_true')
    p = sub.add_parser('queue-json')
    p.add_argument('--folds', default='0')
    p.add_argument('--after', default='none')
    p.add_argument('--workers', type=int, default=LOCAL_WORKERS)
    p.add_argument('--out', default='-', help='new file, or - for stdout; never edits a live queue')
    p.add_argument('--parallel-seeds', action='store_true', help='omit seed-to-seed dependency; scheduler/memory admission remains external')
    p.add_argument('--allow-production-after-screen-pass', action='store_true')
    sub.add_parser('screen')
    a = ap.parse_args()
    if a.command == 'prepare':
        if a.dry_run and within(a.dry_run, ROOT / 'artifacts'):
            raise SystemExit('--dry-run must be outside artifacts')
        prepare(dest=a.dry_run)
    elif a.command == 'validate':
        return validate_cli(a.seed, a.fold, a.allow_production_after_screen_pass)
    elif a.command == 'queue-json':
        entries = queue_entries(folds=tuple(int(x) for x in a.folds.split(',')),
            after=None if a.after.lower() == 'none' else a.after, workers=a.workers,
            allow_production=a.allow_production_after_screen_pass, parallel_seeds=a.parallel_seeds)
        if a.out != '-':
            dump(a.out, entries)
        print(json.dumps(entries, indent=2))
        print(json.dumps(ESTIMATE), file=sys.stderr)
    else:
        screen()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
