"""EXP101 seed 43: the ResNet34 5-fold protocol EXP101_KFOLD_RN34_PROTOCOL_V1 again with seed 43 (H-B-21 second seed, Kaggle T4).

  python tools/exp101_kfold_rn34_s43.py prepare                   # register EXP101_KFOLD_RN34_S43_PROTOCOL_V1 (CPU; computes the RN34 seed-43 init hash)
  python tools/exp101_kfold_rn34_s43.py argv --fold K [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp101_kfold_rn34_s43.py validate --fold K         # runner --validate-only on CPU, no output
  python tools/exp101_kfold_rn34_s43.py smoke --fold K --out DIR [--train 24 --val 4 --epochs 1 --workers 2]   # tiny CPU fit, DIR outside artifacts/
  python tools/exp101_kfold_rn34_s43.py fetch [--kernel OWNER/SLUG] [--dest DIR]   # download the seed-43 Kaggle output, verify every fold by hash
  python tools/exp101_kfold_rn34_s43.py assemble [--runs-dir DIR]                  # pooled 5-fold OOF -> artifacts/experiments/EXP101_RN34_S43_DREAD_OOF

The seed-42 registration (tools/exp101_kfold_arch.py, manifest sha256 BASE_MANIFEST_SHA256) is not changed. prepare copies it and re-verifies
every fold file, the per-fold pos_weight (recomputed from the Dread training labels), the loader contract and every seed-42 source binding;
the only changes are seed 42 -> 43, the seed-43 initial-state hash (computed on CPU exactly as the seed-42 one), the output tags
(EXP101_RN34_S43_DREAD_F<K>), the OOF directory and the Kaggle kernel. Training runs the seed-42 runner file unchanged
(training/runners/exp101_kfold_arch.py) through training/runners/exp101_kfold_rn34_s43.py, which only points the runner's guard import at
this module, so the training loop is byte-identical to the seed-42 job and the seed is the single changed variable.
Nothing here trains on or touches the GPU.
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
from tools import exp101_kfold_arch as base  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
# names the shared runner reads from its guard module (training/runners/exp101_kfold_arch.py uses `guard.<name>`)
from tools.exp101_kfold_arch import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                     CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                     FOLD_PROTOCOL, RECIPE_SOURCE, RECIPE_ARM, DREAD, LABEL_SET, KEYS_FILE, FOLDS, EXPECTED_RECIPE,
                                     EXTRA_EVAL, RUN_FILES, DOCKER_IMAGE, recipe_args, fold_contract, state_hashes, mismatched_keys,
                                     should_start_fold, dump, train_python, within, display_path, build_initial_model, kaggle_cli, _set_flag)

SCHEMA = base.SCHEMA  # same manifest layout as the seed-42 registration
BASE_PROTOCOL = base.PROTOCOL
BASE_MANIFEST_SHA256 = 'edb5b124dfbc033c2b7ad5b4c546f6f0377430454b05d00f9799f2778bdff0d1'
BASE_SEED = 42
BASE_RUNNER = base.RUNNER
BASE_TOOL = base.TOOL
PROTOCOL = 'artifacts/experiments/EXP101_KFOLD_RN34_S43_PROTOCOL_V1'
RUNNER = 'training/runners/exp101_kfold_rn34_s43.py'
TOOL = 'tools/exp101_kfold_rn34_s43.py'
SEED = 43
TAG_PREFIX = 'EXP101_RN34_S43_DREAD'
OOF_DIR = 'artifacts/experiments/EXP101_RN34_S43_DREAD_OOF'
KERNEL = 'taeyangg4/knee-run-v3-exp101-rn34-s43-5fold'
STATUS_FILE = 'exp101-status.json'
# manifest fields that may differ from the seed-42 registration; everything else must be equal (tests/test_exp101_kfold_rn34_s43.py)
CHANGED_FIELDS = ('created_at', 'source_bindings', 'seed', 'initial_state_sha256', 'initial_state', 'recipe', 'folds', 'tags', 'remote', 'evaluation',
                  'caveats', 'seed_variant_of')
FOLD_CHANGED_FIELDS = ('tag',)
SEED_CHANGE = ('seed 43 instead of 42 (H-B-21 second seed): changes the head initialisation, the window sampling sha256(seed, epoch, uid) and the '
               'batch order; folds, pos_weight, recipe, contract and runner code equal EXP101_KFOLD_RN34_PROTOCOL_V1')


def tag_for(fold, mode='full'):
    return f'{TAG_PREFIX}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def _protocol(root=ROOT):
    return read(Path(root) / PROTOCOL / 'manifest.json')


# ------------------------------------------------------------------ prepare
def prepare(root=ROOT, model_factory=None):
    import numpy as np
    root = Path(root)
    destination = root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + PROTOCOL)
    base_path = root / BASE_PROTOCOL / 'manifest.json'
    if sha256(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError('seed-42 EXP101 manifest changed')
    bp = read(base_path)
    if bp['schema'] != SCHEMA or bp['seed'] != BASE_SEED or bp['tags']['prefix'] == TAG_PREFIX:
        raise ValueError('unexpected seed-42 registration')
    for path, expected in bp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('seed-42 binding changed: ' + path)
    for name, expected in bp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    arm = read(root / RECIPE_SOURCE)['arms'][RECIPE_ARM]
    common = recipe_args(arm['expected_args'])
    if common != EXPECTED_RECIPE or bp['common_args'] != EXPECTED_RECIPE:
        raise ValueError('recipe differs from the H-B-21 recipe')
    labels, semantic = k99._load_labels(root / DREAD)
    if semantic != bp['label_set']['semantic_sha256'] or sha256(root / DREAD) != bp['label_set']['sha256']:
        raise ValueError('Dread file differs from the seed-42 registration')
    if sha256(root / bp['fold_protocol']['path']) != bp['fold_protocol']['sha256'] or sha256(root / bp['paths']['folds']) != bp['fold_protocol']['folds_csv_sha256']:
        raise ValueError('EXP099 fold protocol changed')
    canonical = canonical_order(read(root / bp['paths']['base_ordered_ids']))
    corpus_dir = root / bp['paths']['corpus']
    if sha256(corpus_dir / 'manifest.json') != bp['corpus_manifest_sha256']:
        raise ValueError('corpus manifest differs from the seed-42 registration')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=True).astype(str).tolist()
    folds, cover = {}, []
    for k in FOLDS:
        f42 = bp['folds'][str(k)]
        for key in ('ordered_ids', 'groups'):
            if sha256(root / f42[key]['path']) != f42[key]['sha256']:
                raise ValueError(f'fold {k} {key} file changed')
        ordered = read(root / f42['ordered_ids']['path'])
        if uid_hash(ordered['train']) != f42['ordered_ids']['train_sha256'] or uid_hash(ordered['val']) != f42['ordered_ids']['val_sha256']:
            raise ValueError(f'fold {k} ordered ids changed')
        train, val = validate_fold_membership(corpus_ids, ordered, read_rows(root / f42['groups']['path']), canonical)
        pw = pos_weight(labels, train)
        if not np.array_equal(pw, np.asarray(f42['pos_weight'], np.float32)):
            raise ValueError(f'fold {k} pos_weight differs from the seed-42 registration')
        c = fold_contract(len(train), common)
        if any(c[x] != f42[x] for x in c) or (len(train), len(val)) != (f42['n_train'], f42['n_val']):
            raise ValueError(f'fold {k} contract differs from the seed-42 registration')
        folds[str(k)] = {**copy.deepcopy(f42), 'tag': tag_for(k)}
        cover += val
    if sorted(cover) != sorted(canonical) or len(set(cover)) != len(canonical):
        raise ValueError('fold validation sets do not partition the 4,343 studies')
    model = (model_factory or build_initial_model)(common['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init, keys = state_hashes(model)
    seed42_keys = read(root / bp['initial_state']['keys_file'])['keys']
    changed = mismatched_keys(keys, seed42_keys)
    if init == bp['initial_state_sha256'] or not changed:
        raise ValueError('seed 43 gives the seed-42 initial state')
    if set(keys) != set(seed42_keys) or any(x.startswith('backbone.') for x in changed):
        raise ValueError('seed 43 changed the state_dict layout or the pretrained backbone: ' + ', '.join(changed[:10]))
    import timm
    destination.mkdir(parents=True)
    dump(destination / KEYS_FILE, dict(schema='exp101-initial-state-keys-v1', arch=common['arch'], seed=SEED, combined_sha256=init, keys=keys,
                                       rule='per state_dict key: sha256 of the contiguous CPU tensor bytes; combined = sha256 over sorted (key bytes + tensor bytes)'))
    extra = [BASE_PROTOCOL + '/manifest.json', RUNNER, TOOL, PROTOCOL + '/' + KEYS_FILE]
    bindings = {**bp['source_bindings'], **{s: sha256(root / s) for s in extra}}
    manifest = copy.deepcopy(bp)
    manifest.update(
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', source_bindings=bindings, seed=SEED,
        initial_state_sha256=init,
        initial_state=dict(keys_file=PROTOCOL + '/' + KEYS_FILE, keys=len(keys), computed_on='cpu', seed=SEED, arch=common['arch'], torch=torch.__version__,
                           timm=timm.__version__, rule=f'seed_all({SEED}) after worker_init(0), build_model(arch, pretrained timm); sorted state_dict sha256',
                           seed42_initial_state_sha256=bp['initial_state_sha256'], seed42_keys_file=bp['initial_state']['keys_file'],
                           keys_differing_from_seed42=changed, backbone_equal_to_seed42=True,
                           enforcement=f'runner: seed {SEED} must match, unless KNEE_REMOTE_RELAX_ENV is set (then recorded with mismatching keys; D-B-008)'),
        recipe={**bp['recipe'], 'changes': [*bp['recipe']['changes'], SEED_CHANGE]},
        folds=folds,
        tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=OOF_DIR, seed42_prefix=bp['tags']['prefix']),
        remote={**bp['remote'], 'kernel': KERNEL, 'bundle_dataset': 'taeyangg4/knee-runners-v3b', 'remote_entry': 'tools/remote_run_v3b.py',
                'runner': f'{RUNNER} (loads {BASE_RUNNER} unchanged with this guard)',
                'relax_env': f'KNEE_REMOTE_RELAX_ENV=1: environment and seed-{SEED} initial state recorded, not enforced; independent trajectories, '
                             'never paired with a local arm (D-B-008)'},
        evaluation={**bp['evaluation'], 'oof_dir': OOF_DIR},
        caveats=[*bp['caveats'], 'seed 43 is a second Kaggle T4 trajectory of the same recipe; compare with seed 42 only within the T4 lane'],
        seed_variant_of=dict(path=BASE_PROTOCOL + '/manifest.json', sha256=BASE_MANIFEST_SHA256, seed=BASE_SEED,
                             initial_state_sha256=bp['initial_state_sha256'], tag_prefix=bp['tags']['prefix'], oof_dir=bp['tags']['oof_dir'],
                             kernel=bp['remote']['kernel'], changed_fields=list(CHANGED_FIELDS), fold_changed_fields=list(FOLD_CHANGED_FIELDS)))
    dump(destination / 'manifest.json', manifest)
    report = dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, manifest_sha256=sha256(destination / 'manifest.json'), seed=SEED,
                  initial_state_sha256=init, seed42_initial_state_sha256=bp['initial_state_sha256'], keys_differing_from_seed42=changed,
                  total_steps=manifest['total_steps'], tags=[v['tag'] for v in folds.values()])
    print(json.dumps(report, indent=1))
    return manifest


# ------------------------------------------------------------------ runner guard (same checks as tools/exp101_kfold_arch.py, seed-43 constants)
def validate_arguments(args, root):
    """Called by the shared runner before any training import. Returns (protocol, fold spec, ordered ids, output root)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p.get('schema') != SCHEMA or manifest_path.resolve() != (root / PROTOCOL / 'manifest.json').resolve():
        raise ValueError('wrong protocol')
    if p.get('seed') != SEED or p.get('seed_variant_of', {}).get('sha256') != BASE_MANIFEST_SHA256 or p['tags']['prefix'] != TAG_PREFIX:
        raise ValueError('not the seed-43 registration')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
    smoke = args.mode == 'smoke'
    for k, v in p['common_args'].items():
        got = getattr(args, k, None)
        if got != v and not (smoke and k in p['smoke_allowed'] and got in p['smoke_allowed'][k]):
            raise ValueError('frozen argument changed: ' + k)
    if args.seed != p['seed']:
        raise ValueError('seed changed')
    w = p['resource_args']['workers']
    if not w['min'] <= args.workers <= w['max']:
        raise ValueError('workers outside the registered range')
    if str(args.fold) not in p['folds']:
        raise ValueError('unregistered fold')
    fold = p['folds'][str(args.fold)]
    if tag_for(args.fold, args.mode) != args.tag or (not smoke and fold['tag'] != args.tag):
        raise ValueError('tag/fold/mode mismatch')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    for key in ('ordered_ids', 'groups'):
        if sha256(relpath(root, fold[key]['path'])) != fold[key]['sha256']:
            raise ValueError(f'fold {args.fold} {key} file changed')
    ordered = read(relpath(root, fold['ordered_ids']['path']))
    for k in ('train', 'val'):
        if uid_hash(ordered[k]) != fold['ordered_ids'][k + '_sha256']:
            raise ValueError('ordered fold ids changed')
    default_root = (root / 'artifacts' / 'experiments').resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else default_root
    if smoke:
        if not args.output_root or within(out_root, root / 'artifacts'):
            raise ValueError('smoke runs need --output-root outside artifacts/')
        if args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke needs >= one batch of training studies, >= 1 validation study and >= 1 epoch')
    elif out_root != default_root and not os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('a non-default --output-root is for remote runs (KNEE_REMOTE_RELAX_ENV=1)')
    return p, fold, ordered, out_root


def runner_argv(p, fold, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None):
    root = Path(root)
    argv = [python or train_python(), str(root / RUNNER), '--protocol', str(root / PROTOCOL / 'manifest.json'), '--fold', str(int(fold)),
            '--mode', mode, '--tag', tag_for(fold, mode), '--seed', str(p['seed']),
            '--workers', str(p['resource_args']['workers']['default'] if workers is None else int(workers))]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(relpath(root, p['paths']['corpus'])), '--labels', str(relpath(root, p['paths']['labels'])),
             '--eval-labels', str(relpath(root, p['paths']['eval_labels']))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if mode == 'smoke':
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=24, val=4, epochs=1), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def local_env():
    return base.local_env()


def validate_cli(fold):
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(_protocol(), fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(fold, out, train, val, epochs, workers):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(_protocol(), fold, 'smoke', workers=workers, output_root=out, smoke=dict(train=train, val=val, epochs=epochs))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for(fold, 'smoke')
    report = dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(run), files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json')
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], val=s['val'], seed=s['provenance']['seed'],
                      initial_state_sha256=s['provenance']['initial_state_sha256'],
                      initial_state_matches_registered=s['provenance']['initial_state_matches_registered'], history=s['history'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification, download and assembly of the seed-43 runs
def verify_run(run, p, k, manifest_sha):
    """Problems (empty list = complete and consistent) for one seed-43 fold directory: the seed-42 checks plus seed and tag."""
    problems = base.verify_run(run, p, k, manifest_sha)
    if problems:
        return problems
    run = Path(run)
    prov = read(run / 'provenance.json')
    if prov.get('seed') != SEED:
        problems.append(f"fold {k}: provenance seed = {prov.get('seed')!r}, expected {SEED}")
    s = read(run / 'summary.json')
    if (s.get('args') or {}).get('tag', run.name) != tag_for(k) or run.name != tag_for(k):
        problems.append(f'fold {k}: run directory / tag is not {tag_for(k)}')
    return problems


def fetch(kernel=KERNEL, dest=None, retries=3, root=ROOT):
    """Download the seed-43 kernel output fold by fold (small pages, --force) until every fold verifies by hash or retries run out."""
    root = Path(root); p = _protocol(root); manifest_sha = sha256(root / PROTOCOL / 'manifest.json')
    dest = Path(dest or root / '.local' / 'kaggle-out' / kernel.split('/')[-1]).resolve(); dest.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    st = subprocess.run([*kaggle_cli(), 'kernels', 'status', kernel], capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    print((st.stdout or st.stderr).strip()[-400:])
    state = {}
    for pattern in ['exp101-status'] + [tag_for(k) for k in FOLDS]:
        for _ in range(retries):
            r = subprocess.run([*kaggle_cli(), 'kernels', 'output', kernel, '-p', str(dest), '-o', '--file-pattern', pattern, '--page-size', '200'],
                               capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
            if pattern.startswith(TAG_PREFIX):
                k = int(pattern.rsplit('_F', 1)[1])
                problems = verify_run(dest / pattern, p, k, manifest_sha)
                state[pattern] = problems or 'OK'
                if not problems:
                    break
            else:
                state[pattern] = 'downloaded' if r.returncode == 0 else (r.stderr or r.stdout)[-300:]
                break
    status_file = dest / STATUS_FILE
    status = read(status_file) if status_file.exists() else {}
    if status and status.get('protocol') not in (None, PROTOCOL):
        state['exp101-status'] = f"status file belongs to {status.get('protocol')}, not {PROTOCOL}"
    print(json.dumps(dict(dest=str(dest), folds=state, kernel_status=status.get('state'), job=status.get('job')), indent=1))
    return state


def assemble(runs_dir=None, k36_override=None, root=ROOT):
    import pandas as pd
    root = Path(root); k36_override = k36_override or {}
    manifest_path = root / PROTOCOL / 'manifest.json'
    p = read(manifest_path); manifest_sha = sha256(manifest_path)
    runs = Path(runs_dir).resolve() if runs_dir else root / '.local' / 'kaggle-out' / KERNEL.split('/')[-1]
    out = root / OOF_DIR
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    frames24, frames36, sources, problems = {}, {}, {}, []
    for k in sorted(val_ids):
        run = runs / tag_for(k)
        bad = verify_run(run, p, k, manifest_sha)
        if bad:
            problems += bad; continue
        prov = read(run / 'provenance.json')
        frames24[k] = pd.read_csv(run / 'oof_best.csv', dtype={UID: str})
        k36 = Path(k36_override[k]) if k in k36_override else run / EXTRA_EVAL['file']
        if k36.is_file():
            frames36[k] = pd.read_csv(k36, dtype={UID: str})
        sources[k] = dict(run=display_path(run, root), oof_k24_sha256=sha256(run / 'oof_best.csv'), oof_k36=display_path(k36, root) if k36.is_file() else None,
                          oof_k36_sha256=sha256(k36) if k36.is_file() else None, checkpoint_sha256=read(run / 'output-inventory.json')['model_best.pt']['sha256'],
                          seed=prov.get('seed'), initial_state_sha256=prov.get('initial_state_sha256'),
                          initial_state_matches_registered=prov.get('initial_state_matches_registered'), environment=prov.get('environment'),
                          remote_relaxed=prov.get('remote_relaxed'))
    if problems:
        print(json.dumps(dict(status='INCOMPLETE', problems=problems), indent=1))
        raise SystemExit(2)
    result = k99.assemble_and_score(frames24, frames36, val_ids, root / DREAD)
    out.mkdir(parents=True)
    result['oof_k24'].to_csv(out / 'oof_k24.csv', index=False)
    if result['oof_k36'] is not None:
        result['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp101-oof-v1', label_set=LABEL_SET, label_set_sha256=p['label_set']['sha256'], observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol=PROTOCOL, protocol_sha256=manifest_sha, seed=SEED, seed42_protocol_sha256=BASE_MANIFEST_SHA256, runs_dir=display_path(runs, root),
                rows=int(len(result['oof_k24'])), sources={str(k): v for k, v in sources.items()},
                outputs={f: sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv') if (out / f).exists()},
                scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                metrics=result['metrics'], k36_missing_folds=result['k36_missing'],
                caveats=['weak Dread labels', 'fold 0 val = exposed G1 V888; folds 1-4 models saw G1 val in training', 'pooled AUC mixes 5 models',
                         'Kaggle T4 trajectories (KNEE_REMOTE_RELAX_ENV); compare with local arms only as separate families',
                         'seed 43; the seed-42 OOF is ' + base.OOF_DIR])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    md = k99.result_markdown(LABEL_SET, body).replace(f'# EXP099 B8 5-fold OOF ({LABEL_SET})', f'# EXP101 ResNet34 seed 43 5-fold OOF ({LABEL_SET}, Kaggle T4)', 1)
    (out / 'result.md').write_text(md, encoding='utf-8')
    print(json.dumps(dict(rows=body['rows'], metrics={k: {kk: (round(vv['macro12'], 6) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
                                                      for k, v in body['metrics'].items()}, k36_missing_folds=body['k36_missing_folds']), indent=1))
    return body


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('argv'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--mode', choices=['full', 'smoke'], default='full')
    s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    s = sub.add_parser('validate'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s = sub.add_parser('smoke'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--out', required=True)
    s.add_argument('--train', type=int, default=24); s.add_argument('--val', type=int, default=4); s.add_argument('--epochs', type=int, default=1)
    s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('fetch'); s.add_argument('--kernel', default=KERNEL); s.add_argument('--dest', default=None); s.add_argument('--retries', type=int, default=3)
    s = sub.add_parser('assemble')
    s.add_argument('--runs-dir', default=None, help=f'directory holding {TAG_PREFIX}_F<K>/ (default .local/kaggle-out/{KERNEL.split("/")[-1]})')
    s.add_argument('--k36', action='append', default=[], help='FOLD=PATH oof_k36.csv override')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Kaggle CLI output can contain characters the cp949 console cannot encode (D-B-015)
        stream.reconfigure(encoding='utf-8', errors='replace')
    if a.command == 'prepare':
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        prepare()
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.fold, a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli(a.fold))
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.fold, a.out, a.train, a.val, a.epochs, a.workers))
    elif a.command == 'fetch':
        fetch(a.kernel, a.dest, a.retries)
    else:
        assemble(a.runs_dir, {int(x.split('=', 1)[0]): x.split('=', 1)[1] for x in a.k36})


if __name__ == '__main__':
    main()
