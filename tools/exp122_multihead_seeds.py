"""EXP122: exact EXP117 multi-head protocol replicated with seeds 43 and 44.

CPU registration and validation only; queue-json writes jobs for gpu_queue.py.
Each seed has its own immutable registration and five independent folds.
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
from tools import exp117_multihead as e117
from tools import exp118_radr50_local_s43 as e118
from tools import exp107_kfold_radr50 as e107
from tools import exp099_kfold as k99
from tools.exp117_multihead import (build_multihead, head_pos_weight, stack_sources,
                                    masked_multihead_bce, rank_mean)
from tools.exp113_radr50_local import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,
                                       CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership,
                                       pos_weight, fold_contract, UID, T, DREAD, FOLDS, state_hashes, mismatched_keys,
                                       train_python, within, display_path, _set_flag)

SCHEMA = 'exp122-multihead-seed-v1'
BASE_SHA = '5ee284b1f4f2933d2931c2cd60a90e098d65ae4712dd4cbd2fde49929cee5457'
RUNNER = 'training/runners/exp122_multihead_seeds.py'
TOOL = 'tools/exp122_multihead_seeds.py'
SEEDS = (43, 44)
HEAD_NAMES = e117.HEAD_NAMES
RUN_FILES = e117.RUN_FILES
LABEL_SET = e117.LABEL_SET
LOCAL_WORKERS = e117.LOCAL_WORKERS
BOOT_SEED = 20261003


def paths(seed):
    if seed not in SEEDS:
        raise ValueError('EXP122 seed must be 43 or 44')
    stem = f'EXP122_MULTIHEAD_S{seed}'
    return dict(prefix=stem, protocol=f'artifacts/experiments/{stem}_PROTOCOL_V1',
                oof=f'artifacts/experiments/{stem}_OOF')


def tag_for(fold, mode='full', seed=None):
    if seed not in SEEDS or fold not in FOLDS or mode not in ('full', 'smoke'):
        raise ValueError('seed, fold and mode required for EXP122 tag')
    return f"{paths(seed)['prefix']}_F{fold}" + ('_SMOKE' if mode == 'smoke' else '')


def manifest_path(seed, root=ROOT):
    return Path(root) / paths(seed)['protocol'] / 'manifest.json'


def protocol(seed, root=ROOT):
    p = read(manifest_path(seed, root))
    if p.get('schema') != SCHEMA or p.get('seed') != seed or p.get('derived_from', {}).get('sha256') != BASE_SHA:
        raise ValueError('wrong EXP122 protocol')
    return p


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def assert_initial_state(seed, keys, base_keys, expected=None):
    """Verify RadImageNet backbone identity and a seed-dependent head."""
    if set(keys) != set(base_keys):
        raise ValueError('initial state key set differs from EXP117')
    backbone = [k for k in keys if k.startswith('backbone.')]
    if any(keys[k] != base_keys[k] for k in backbone):
        raise ValueError('RadImageNet backbone differs from EXP117')
    if not any(keys[k] != base_keys[k] for k in keys if not k.startswith('backbone.')):
        raise ValueError(f'seed {seed} head does not differ from seed 42')
    if expected is not None and keys != expected:
        raise ValueError(f'seed {seed} state differs from registered EXP118 state')


def prepare(seed, root=ROOT):
    root = Path(root)
    pp = paths(seed)
    dest = root / pp['protocol']
    if dest.exists():
        raise SystemExit(f'refusing to overwrite registration: {dest}')
    base_file = root / e117.PROTOCOL / 'manifest.json'
    if sha256(base_file) != BASE_SHA:
        raise ValueError('EXP117 screened manifest changed')
    screen = read(root / e117.PROTOCOL / 'f0_screen_pass.json')
    if screen.get('verdict') != 'PASS' or screen.get('protocol_sha256') != BASE_SHA or screen.get('tag') != e117.tag_for(0):
        raise ValueError('EXP117 F0 screen PASS does not bind the frozen protocol')
    bp = read(base_file)
    if bp['schema'] != e117.SCHEMA or bp['seed'] != 42 or tuple(h['name'] for h in bp['heads']) != HEAD_NAMES:
        raise ValueError('unexpected EXP117 registration')
    for name, digest in bp['source_bindings'].items():
        if sha256(root / name) != digest:
            raise ValueError('EXP117 source binding changed: ' + name)
    for name, digest in bp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != digest:
            raise ValueError('EXP117 training source changed: ' + name)
    seed42_keys = read(root / bp['initial_state']['keys_file'])['keys']
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        os.environ[k] = v
    base, report = e107.build_initial_model(bp['common_args']['arch'], seed, e117.e113.weights_path(root))
    digest, keys = state_hashes(base)
    if report != bp['init_weights']['load']:
        raise ValueError('RadImageNet load report changed')
    expected = None
    if seed == 43:
        r118 = read(root / e118.PROTOCOL / 'manifest.json')
        expected = read(root / r118['initial_state']['keys_file'])['keys']
        if digest != r118['initial_state_sha256'] or digest != 'c629d21f3046811e460793d4c62982c25127c1bcb9565f208674a366b5850b2c':
            raise ValueError('seed 43 does not reproduce EXP118 initial state')
    assert_initial_state(seed, keys, seed42_keys, expected)
    multi = build_multihead(base, len(HEAD_NAMES))
    multi_digest, multi_keys = state_hashes(multi)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('prepare initialized CUDA')
    del base, multi
    folds = copy.deepcopy(bp['folds'])
    for k in FOLDS:
        folds[str(k)]['tag'] = tag_for(k, seed=seed)
    state = copy.deepcopy(bp['initial_state'])
    if seed == 43:
        state['keys_file'] = r118['initial_state']['keys_file']
        state['seed43_control'] = dict(protocol=e118.PROTOCOL, sha256=sha256(root / e118.PROTOCOL / 'manifest.json'),
                                       initial_state_sha256=digest)
    else:
        state['keys_file'] = pp['protocol'] + '/initial-state-keys.json'
    state.update(seed=seed, rule=f'EXP117 initial-state construction with seed_all({seed}); same RadImageNet backbone; head copied four times',
                 enforcement='runner checks base and multihead state hashes exactly')
    bindings = copy.deepcopy(bp['source_bindings'])
    bindings.update({e117.PROTOCOL + '/manifest.json': BASE_SHA,
                     e117.PROTOCOL + '/f0_screen_pass.json': sha256(root / e117.PROTOCOL / 'f0_screen_pass.json'),
                     RUNNER: sha256(root / RUNNER), TOOL: sha256(root / TOOL)})
    if seed == 43:
        bindings[e118.PROTOCOL + '/manifest.json'] = sha256(root / e118.PROTOCOL / 'manifest.json')
        bindings[state['keys_file']] = sha256(root / state['keys_file'])
    p = copy.deepcopy(bp)
    p.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_item='H-B-47',
             seed=seed, initial_state_sha256=digest, initial_state=state, source_bindings=bindings, folds=folds,
             tags={**bp['tags'], 'prefix': pp['prefix'], 'pattern': pp['prefix'] + '_F<K>[_SMOKE]', 'oof_dir': pp['oof']},
             local={**bp['local'], 'runner': RUNNER, 'guard': TOOL,
                    'runner_rule': 'byte-identical EXP117 runner except guard import, verified by ordered equivalence test'},
             evaluation={**bp['evaluation'], 'oof_dir': pp['oof']},
             multihead={**bp['multihead'], 'init': f'EXP118 seed-43 state' if seed == 43 else 'registered seed-44 state',
                        'multihead_initial_state_sha256': multi_digest, 'multihead_keys': len(multi_keys)},
             derived_from=dict(path=e117.PROTOCOL + '/manifest.json', sha256=BASE_SHA, changed_args={},
                               changed_seed={'from': 42, 'to': seed},
                               changed_fields=['seed', 'initial_state', 'initial_state_sha256', 'multihead', 'folds',
                                               'tags', 'local', 'evaluation', 'source_bindings', 'derived_from', 'created_at', 'schema']))
    dest.mkdir(parents=True)
    if seed == 44:
        dump_new(root / state['keys_file'], dict(seed=seed, combined_sha256=digest, keys=keys))
        p['source_bindings'][state['keys_file']] = sha256(root / state['keys_file'])
    dump_new(dest / 'manifest.json', p)
    print(json.dumps(dict(protocol=str(dest), manifest_sha256=sha256(dest / 'manifest.json'), seed=seed,
                          initial_state_sha256=digest, multihead_initial_state_sha256=multi_digest), indent=2))
    return p


def validate_arguments(args, root):
    root = Path(root)
    seed = args.seed
    p = protocol(seed, root)
    if Path(args.protocol).resolve() != manifest_path(seed, root).resolve():
        raise ValueError('wrong protocol path')
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('EXP122 is local only')
    for name, digest in p['source_bindings'].items():
        if sha256(root / name) != digest:
            raise ValueError('frozen binding changed: ' + name)
    for name, digest in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != digest:
            raise ValueError('training source changed: ' + name)
    smoke = args.mode == 'smoke'
    for k, v in p['common_args'].items():
        got = getattr(args, k, None)
        if got != v and not (smoke and k in p['smoke_allowed'] and got in p['smoke_allowed'][k]):
            raise ValueError('frozen argument changed: ' + k)
    if not p['resource_args']['workers']['min'] <= args.workers <= p['resource_args']['workers']['max']:
        raise ValueError('workers outside registered range')
    if str(args.fold) not in p['folds']:
        raise ValueError('unregistered fold')
    fold = p['folds'][str(args.fold)]
    if args.tag != tag_for(args.fold, args.mode, seed) or (not smoke and fold['tag'] != args.tag):
        raise ValueError('tag mismatch')
    for k in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, k)).resolve() != (root / p['paths'][k]).resolve():
            raise ValueError('input path mismatch: ' + k)
    if Path(args.init_weights).resolve() != e117.e113.weights_path(root).resolve() or sha256(args.init_weights) != p['init_weights']['sha256']:
        raise ValueError('RadImageNet weight path/hash mismatch')
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    for key in ('ordered_ids', 'groups'):
        if sha256(root / fold[key]['path']) != fold[key]['sha256']:
            raise ValueError('fold contract changed: ' + key)
    ordered = read(root / fold['ordered_ids']['path'])
    for k in ('train', 'val'):
        if uid_hash(ordered[k]) != fold['ordered_ids'][k + '_sha256']:
            raise ValueError('ordered IDs changed')
    default_root = (root / 'artifacts/experiments').resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else default_root
    if smoke:
        if not args.output_root or within(out_root, root / 'artifacts'):
            raise ValueError('smoke output must be outside artifacts')
        if args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke subset too small')
    elif out_root != default_root:
        raise ValueError('full runs write to artifacts/experiments only')
    return p, fold, ordered, out_root


def runner_argv(p, fold, mode='full', root=ROOT, workers=LOCAL_WORKERS, output_root=None):
    seed = p['seed']
    argv = [train_python(), str(Path(root) / RUNNER), '--protocol', str(manifest_path(seed, root)),
            '--fold', str(fold), '--mode', mode, '--tag', tag_for(fold, mode, seed), '--seed', str(seed),
            '--workers', str(workers)]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(Path(root) / p['paths']['corpus']), '--labels', str(Path(root) / p['paths']['labels']),
             '--eval-labels', str(Path(root) / p['paths']['eval_labels']), '--init-weights', str(e117.e113.weights_path(root))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    return argv


def queue_entries(seed, after, root=ROOT, workers=LOCAL_WORKERS):
    p = protocol(seed, root)
    result = []
    previous = after
    for fold in FOLDS:
        item = dict(tag=tag_for(fold, seed=seed), command=runner_argv(p, fold, root=root, workers=workers),
                    continue_on_failure=True)
        if previous:
            item['after'] = previous
        result.append(item)
        previous = item['tag']
    return result


def verify_run(run, p, fold, manifest_sha):
    import numpy as np
    run = Path(run)
    if any(not (run / name).is_file() for name in RUN_FILES):
        return [f'fold {fold}: missing output file']
    inv = read(run / 'output-inventory.json')
    errors = [f'fold {fold}: inventory mismatch for {n}' for n, rec in inv.items()
              if not (run / n).is_file() or sha256(run / n) != rec['sha256']]
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    spec = p['folds'][str(fold)]
    expected = dict(fold=fold, protocol_sha256=manifest_sha, seed=p['seed'],
                    eval_ids_ordered_sha256=spec['ordered_ids']['val_sha256'],
                    train_ids_ordered_sha256=spec['ordered_ids']['train_sha256'],
                    initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True,
                    multihead_initial_state_sha256=p['multihead']['multihead_initial_state_sha256'],
                    multihead_initial_state_matches_registered=True, runner_sha256=p['source_bindings'][RUNNER],
                    guard_sha256=p['source_bindings'][TOOL], training_mode='full', heads=list(HEAD_NAMES),
                    label_set=LABEL_SET, environment_matches_baseline=True, remote_relaxed=False)
    errors += [f'fold {fold}: provenance {k} mismatch' for k, v in expected.items() if prov.get(k) != v]
    errors += [f'fold {fold}: pos_weight mismatch' for n in HEAD_NAMES if not np.array_equal(
        np.asarray(prov.get('head_pos_weight', {}).get(n), np.float32), np.asarray(spec['head_pos_weight'][n], np.float32))]
    if s.get('optimizer_updates') != spec['steps'] or s.get('skipped_updates', 10**9) > spec['max_fp16_skips']:
        errors.append(f'fold {fold}: optimizer/FP16 gate failed')
    if s.get('selected_epoch_1based') != p['common_args']['epochs'] or s.get('train') != spec['n_train'] or s.get('val') != spec['n_val']:
        errors.append(f'fold {fold}: count/epoch mismatch')
    if run.name != tag_for(fold, seed=p['seed']):
        errors.append(f'fold {fold}: run directory mismatch')
    return errors


def assemble(seed, root=ROOT):
    import pandas as pd
    root = Path(root)
    p = protocol(seed, root)
    digest = sha256(manifest_path(seed, root))
    out = root / paths(seed)['oof']
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    names = {('served', 'k24'): 'oof_best.csv', ('served', 'k36'): 'oof_k36.csv'}
    names.update({(n, k): f'oof_head_{n}' + ('_k36' if k == 'k36' else '') + '.csv'
                  for n in HEAD_NAMES for k in ('k24', 'k36')})
    frames = {key: {} for key in names}
    errors = []
    for fold in FOLDS:
        run = root / 'artifacts/experiments' / tag_for(fold, seed=seed)
        errors += verify_run(run, p, fold, digest)
        if not errors:
            for key, name in names.items():
                frames[key][fold] = pd.read_csv(run / name, dtype={UID: str})
    if errors:
        raise SystemExit('\n'.join(errors))
    out.mkdir(parents=True)
    metrics, outputs = {}, {}
    for who in ('served',) + HEAD_NAMES:
        result = k99.assemble_and_score(frames[(who, 'k24')], frames[(who, 'k36')], val_ids, root / DREAD)
        stem = 'oof' if who == 'served' else f'oof_head_{who}'
        for k in ('k24', 'k36'):
            name = f'{stem}_{k}.csv'
            result[f'oof_{k}'].to_csv(out / name, index=False)
            outputs[name] = sha256(out / name)
        metrics[who] = result['metrics']
    body = dict(schema='exp122-oof-v1', seed=seed, protocol_sha256=digest, rows=sum(map(len, val_ids.values())),
                outputs=outputs, metrics=metrics, heads=list(HEAD_NAMES),
                serving=e117.SERVING_RULE, caveat='weak labels; per-fold rank mean pooled across folds')
    dump_new(out / 'result.json', body)
    print(json.dumps({k: v['k36']['pooled']['macro12'] for k, v in metrics.items()}, indent=2))
    return body


def _read_served(seed, root):
    import pandas as pd
    out = Path(root) / (e117.OOF_DIR if seed == 42 else paths(seed)['oof'])
    rec = read(out / 'result.json')
    expected = BASE_SHA if seed == 42 else sha256(manifest_path(seed, root))
    if rec['protocol_sha256'] != expected:
        raise ValueError(f'seed {seed} OOF protocol mismatch')
    path = out / 'oof_k36.csv'
    if sha256(path) != rec['outputs']['oof_k36.csv']:
        raise ValueError(f'seed {seed} OOF hash mismatch')
    return pd.read_csv(path, dtype={UID: str})


def compare(root=ROOT, draws=1000):
    import numpy as np
    from tools import exp115_vlm_eval as mr
    tables = {seed: _read_served(seed, root) for seed in (42, 43, 44)}
    ids = tables[42][UID].tolist()
    if any(t[UID].tolist() != ids or list(t.columns) != [UID] + T for t in tables.values()):
        raise ValueError('seed OOF membership, order or columns differ')
    arrays = {s: t[T].to_numpy(np.float64) for s, t in tables.items()}
    refs = mr.references(ids, root)
    idx = mr.bootstrap_index(mr.component_groups(ids, root), draws, BOOT_SEED)
    fold_ids = {int(k): read(Path(root) / f['ordered_ids']['path'])['val']
                for k, f in read(Path(root) / e117.PROTOCOL / 'manifest.json')['folds'].items()}
    row = {uid: i for i, uid in enumerate(ids)}
    if set(row) != {u for members in fold_ids.values() for u in members}:
        raise ValueError('OOF rows differ from registered fold membership')
    def rankmean(seeds):
        out = np.empty_like(arrays[42])
        for members in fold_ids.values():
            ix = [row[u] for u in members]
            out[ix] = np.mean([mr.pct_rank(arrays[s][ix]) for s in seeds], axis=0)
        return out
    contrasts = {'seed43_minus_42': (arrays[43], arrays[42]), 'seed44_minus_42': (arrays[44], arrays[42]),
                 'seed44_minus_43': (arrays[44], arrays[43]),
                 'mean42_43_minus_42': (rankmean((42, 43)), arrays[42]),
                 'mean42_44_minus_42': (rankmean((42, 44)), arrays[42]),
                 'mean42_43_44_minus_42': (rankmean((42, 43, 44)), arrays[42])}
    metrics = {name: {r: mr.compare(a, b, refs[r], idx) for r in mr.REFERENCES} for name, (a, b) in contrasts.items()}
    result = dict(schema='exp122-compare-v1', draws=draws, bootstrap_seed=BOOT_SEED,
                  references=list(mr.REFERENCES), contrasts=metrics,
                  note='paired component bootstrap conditional on fixed OOF; label-source model has home references')
    out = Path(root) / 'artifacts/experiments/EXP122_MULTIHEAD_SEED_COMPARE'
    dump_new(out / 'result.json', result)
    print(json.dumps({n: {r: v['delta'] for r, v in m.items()} for n, m in metrics.items()}, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'validate', 'queue-json', 'assemble'):
        s = sub.add_parser(name)
        s.add_argument('--seed', type=int, choices=SEEDS, required=True)
        if name == 'validate':
            s.add_argument('--fold', type=int, choices=FOLDS, required=True)
        if name == 'queue-json':
            s.add_argument('--after', required=True)
            s.add_argument('--out', required=True)
            s.add_argument('--workers', type=int, default=LOCAL_WORKERS)
    s = sub.add_parser('compare'); s.add_argument('--draws', type=int, default=1000)
    a = ap.parse_args(argv)
    if a.action == 'prepare':
        prepare(a.seed)
    elif a.action == 'validate':
        cmd = runner_argv(protocol(a.seed), a.fold) + ['--validate-only']
        proc = subprocess.run(cmd, cwd=ROOT, env=e117.e113.local_env(), text=True, capture_output=True)
        print(proc.stdout[-6000:]); print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(proc.returncode)
    elif a.action == 'queue-json':
        entries = queue_entries(a.seed, None if a.after.lower() == 'none' else a.after, workers=a.workers)
        dump_new(a.out, entries)
        print(json.dumps(entries, indent=2))
    elif a.action == 'assemble':
        assemble(a.seed)
    else:
        compare(draws=a.draws)


if __name__ == '__main__':
    main()
