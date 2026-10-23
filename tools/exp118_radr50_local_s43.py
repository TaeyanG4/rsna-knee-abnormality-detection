"""EXP118 (H-B-34): the EXP113 local RadImageNet ResNet50 centre-slice x3 5-fold with seed 43 instead of 42 (the seed-43 local control).

  python tools/exp118_radr50_local_s43.py prepare [--dry-run DIR]   # register EXP118_RADR50_LOCAL_S43_PROTOCOL_V1 (CPU; computes the seed-43 init)
  python tools/exp118_radr50_local_s43.py argv --fold K [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp118_radr50_local_s43.py validate --fold K        # runner --validate-only on CPU, no output
  python tools/exp118_radr50_local_s43.py smoke --fold K --out DIR [--train 16 --val 2 --epochs 1 --workers 2]   # CPU fit, DIR outside artifacts/
  python tools/exp118_radr50_local_s43.py queue-json [--folds 0,1,2,3,4] [--after EXP117_MULTIHEAD_F0] [--out FILE|-]   # tools/gpu_queue.py entries
  python tools/exp118_radr50_local_s43.py assemble [--runs-dir DIR] [--k36 K=PATH]   # pooled OOF -> artifacts/experiments/EXP118_RADR50_LOCAL_S43_OOF
  python tools/exp118_radr50_local_s43.py compare [--draws 1000]   # seed 43 vs EXP113 seed 42 on four references (threshold calibration)

A copy of EXP113_RADR50_LOCAL_PROTOCOL_V1 (manifest sha256 BASE_MANIFEST_SHA256, not changed) that changes ONLY the seed: the same EXP099
folds, Dread labels, per-fold pos_weight, RadImageNet weights (local file, sha256 08629f7e...), RN_clip recipe, K12 / K24 / K36, windows,
input policy and normalisation, local GPU environment (enforced, no KNEE_REMOTE_RELAX_ENV) and 12 loader workers. Seed 43 changes the head
initialisation, the window-sampling streams (sha256(seed, epoch, uid, stream)) and the shuffle order together. The seed-43 initial state is
re-computed on CPU at prepare (backbone keys must equal EXP113's RadImageNet bytes; the seed-42 state is re-computed as a control and must
equal EXP113's) and enforced at run time exactly as EXP113 enforces seed 42: the shared EXP107 runner stops when a run with the registered
seed does not reproduce initial_state_sha256 (every key, head included). Training runs the EXP107 runner file unchanged
(training/runners/exp107_kfold_radr50.py) through training/runners/exp118_radr50_local_s43.py, which points the runner's guard import at this
module. Names: EXP118_RADR50_LOCAL_S43_F<K>, OOF artifacts/experiments/EXP118_RADR50_LOCAL_S43_OOF. Dread only (no label-set registry).
`compare` reports seed 43 - seed 42 (pooled and per fold, four references, component bootstrap): the seed spread that calibrates the
+0.004 adoption threshold of D-B-042. It is descriptive and never passes or fails either run. Nothing here trains on or touches the GPU.
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
from tools import exp113_radr50_local as e113  # noqa: E402
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
# names the shared EXP107 runner reads from its guard module (`guard.<name>`), re-exported unchanged (EXP113 re-exports EXP107's own)
from tools.exp113_radr50_local import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                       CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                       DREAD, KEYS_FILE, FOLDS, EXTRA_EVAL, RUN_FILES, fold_contract, state_hashes, mismatched_keys, dump,
                                       train_python, within, display_path, _set_flag, weights_path, preflight_versions, local_env,
                                       WEIGHTS_LOCAL, RADIMAGENET_SHA256, LOCAL_GPU, LOCAL_WORKERS, DEFAULT_LABEL_SET)

SCHEMA = 'exp118-radr50-local-s43-v1'
BASE_PROTOCOL = e113.PROTOCOL
BASE_MANIFEST_SHA256 = 'a134327a10bfe051a0d58f3bb83969b2bdb8b9fcc4f54a309893504c960bd31d'  # EXP113_RADR50_LOCAL_PROTOCOL_V1/manifest.json
BASE_KEYS_SHA256 = e113.BASE_KEYS_SHA256  # the seed-42 keys file (EXP107 = EXP113)
BASE_SEED = e113.SEED
BASE_OOF_DIR = e113.OOF_DIR  # the seed-42 local OOF that `compare` reads
SHARED_RUNNER = e113.SHARED_RUNNER
PROTOCOL = 'artifacts/experiments/EXP118_RADR50_LOCAL_S43_PROTOCOL_V1'
QUEUE_FILE = PROTOCOL + '/queue-entries.json'
RUNNER = 'training/runners/exp118_radr50_local_s43.py'
TOOL = 'tools/exp118_radr50_local_s43.py'
SEED = 43
TAG_PREFIX = 'EXP118_RADR50_LOCAL_S43'
OOF_DIR = 'artifacts/experiments/EXP118_RADR50_LOCAL_S43_OOF'
COMPARE_DIR = 'artifacts/experiments/EXP118_RADR50_LOCAL_S43_VS_EXP113'
DEFAULT_AFTER = 'EXP117_MULTIHEAD_F0'
LABEL_SET = DEFAULT_LABEL_SET  # read by the shared runner (provenance label_set); EXP118 is dread only
ADOPTION_THRESHOLD = 0.004  # D-B-042 (provisional): adopt a RadR50 change if pooled Dread delta >= +0.004, bootstrap CI > 0, consensus delta >= 0
PRIOR_SEED_PAIR_DREAD = 0.0021  # D-B-042: same-recipe seed pair difference (Dread, 5-fold, 4,343) that set the threshold at about 2x
PRIMARY_K = 'k36'  # D-B-042: pooled 5-fold K36 OOF delta
DRAWS = 1000
BOOT_SEED = 20261003
REFERENCE_SCORER = 'tools/exp115_vlm_eval.py'  # dread rint / flight_v4 >= 0.5 / relabel_v1 >= 0.5 / consensus (D-B-035), component bootstrap
# manifest fields that may differ from the EXP113 registration; everything else must be equal (tests/test_exp118_radr50_local_s43.py)
CHANGED_FIELDS = ('schema', 'created_at', 'source_bindings', 'seed', 'initial_state_sha256', 'initial_state', 'folds', 'tags', 'local', 'evaluation',
                  'caveats', 'derived_from', 'label_sets')
FOLD_CHANGED_FIELDS = ('tag',)
INITIAL_STATE_CHANGED = ('keys_file', 'seed', 'rule', 'enforcement', 'seed42_control')
ENFORCEMENT = ('local runner: the seed-43 state must equal initial_state_sha256 exactly (every key, head included; backbone keys equal the '
               'RadImageNet bytes of EXP113) and the runtime environment must equal the registered local environment; KNEE_REMOTE_RELAX_ENV '
               'is refused by the guard')
ESTIMATE = dict(seconds_per_fold=1450, range_seconds_per_fold=[1300, 1600], five_folds_hours=[1.8, 2.2],
                basis='EXP113 (same recipe, GPU, workers) observed 1,550 / 1,312 / 1,463 s for folds 0-2 (summary.json total_seconds); '
                      'a different seed does not change the step count or the evaluation work')
CAVEATS_S43 = [
    'local RTX 4070 Ti SUPER trajectories with the enforced seed-43 initial state (same Windows window pipeline and torch 2.5.1+cu121 / '
    'timm 1.0.19 stack as EXP113); seed 43 - EXP113 seed 42 is a same-recipe, same-device seed contrast: the seed changes the head '
    'initialisation, the window-sampling streams and the shuffle order together, so it measures trajectory noise, never a recipe effect',
    'one seed pair: seed 43 - seed 42 is a single draw of trajectory noise; with the earlier pair (0.0021 Dread, D-B-042) it calibrates the '
    'provisional +0.004 adoption threshold only roughly',
    'recipe not re-tuned for ResNet50 / RadImageNet',
    'against the Kaggle T4 EXP107 run this is an independent trajectory (D-B-008), never a paired arm',
]
LOCAL_RULE = ('training/runners/exp118_radr50_local_s43.py points the shared runner\'s guard import (tools.exp107_kfold_radr50) at '
              'tools/exp118_radr50_local_s43.py for its own process and calls the unchanged EXP107 runner main(); it reuses the EXP113 entry\'s '
              'supply_registered_env (OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 when absent, before torch is imported; '
              'D-B-017) and accepts no label set other than dread')
QUEUE_RULE = ('tools/exp118_radr50_local_s43.py queue-json -> entries for tools/gpu_queue.py: F0 after ' + DEFAULT_AFTER + ' (default; the '
              'current chain tail at registration), F1-F4 chained; continue_on_failure true on every entry (an upstream failure does not stop '
              'the control; folds are independent fits)')
COMPARE_RULE = ('compare: pooled 5-fold OOF (K24 and K36; K36 primary as D-B-042) of seed 43 minus EXP113 seed 42 scored on dread (rint), '
                'flight_v4 (>= 0.5), relabel_v1 (>= 0.5) and consensus (D-B-035: all four sources agree) with the tools/exp115_vlm_eval.py '
                'scorer, a paired component bootstrap (VALIDATION_GROUP_CONTRACT_V1, shared indices, seed 20261003), per-fold seed '
                'differences and their spread, and whether the D-B-042 adoption rule would "adopt" a pure seed change; descriptive only')


def tag_for(fold, mode='full'):
    return f'{TAG_PREFIX}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def _manifest_path(root=ROOT):
    return Path(root) / PROTOCOL / 'manifest.json'


def _protocol(root=ROOT):
    return read(_manifest_path(root))


def split_keys(keys):
    """(backbone keys, head keys) of a state_hashes key map; the RadImageNet checkpoint fixes every backbone.* byte."""
    backbone = sorted(k for k in keys if k.startswith('backbone.'))
    return backbone, sorted(k for k in keys if not k.startswith('backbone.'))


def seed_state_report(keys43, keys42):
    """Per-key comparison of the seed-43 state with the registered seed-42 state. Raises unless the key sets are equal, every backbone key
    is equal (same checkpoint bytes) and at least one head key differs (the seed reached the head)."""
    if set(keys43) != set(keys42):
        raise ValueError('seed-43 state has different keys from the seed-42 registration')
    backbone, head = split_keys(keys42)
    bad = [k for k in backbone if list(keys43[k]) != list(keys42[k])]
    if bad:
        raise ValueError(f'seed-43 backbone differs from the RadImageNet bytes ({len(bad)} keys, e.g. {bad[:5]})')
    differing = [k for k in head if list(keys43[k]) != list(keys42[k])]
    if not differing:
        raise ValueError('seed-43 head equals the seed-42 head: the seed did not reach the initial state')
    return dict(keys=len(keys42), backbone_keys=len(backbone), backbone_equal_to_seed42=True, head_keys=len(head),
                head_keys_differing_from_seed42=len(differing), head_keys_equal_to_seed42=len(head) - len(differing), differing=differing)


# ------------------------------------------------------------------ prepare
def prepare(root=ROOT, model_factory=None, dest=None):
    import numpy as np
    root = Path(root)
    destination = Path(dest) if dest else root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + str(destination))
    base_path = root / BASE_PROTOCOL / 'manifest.json'
    if sha256(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError('EXP113 manifest changed')
    bp = read(base_path)
    if (bp['schema'] != e113.SCHEMA or bp['seed'] != BASE_SEED or bp['tags']['prefix'] != e113.TAG_PREFIX or bp['label_set']['name'] != DEFAULT_LABEL_SET
            or bp['derived_from']['sha256'] != e113.BASE_MANIFEST_SHA256 or bp['common_args'] != e107.EXPECTED_RECIPE):
        raise ValueError('unexpected EXP113 registration')
    for path, expected in bp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('EXP113 binding changed: ' + path)
    for name, expected in bp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    labels, semantic = k99._load_labels(root / DREAD)
    if semantic != bp['label_set']['semantic_sha256'] or sha256(root / DREAD) != bp['label_set']['sha256']:
        raise ValueError('Dread file differs from the EXP113 registration')
    if sha256(root / bp['fold_protocol']['path']) != bp['fold_protocol']['sha256'] or sha256(root / bp['paths']['folds']) != bp['fold_protocol']['folds_csv_sha256']:
        raise ValueError('EXP099 fold protocol changed')
    canonical = canonical_order(read(root / bp['paths']['base_ordered_ids']))
    corpus_dir = root / bp['paths']['corpus']
    if sha256(corpus_dir / 'manifest.json') != bp['corpus_manifest_sha256']:
        raise ValueError('corpus manifest differs from the EXP113 registration')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=True).astype(str).tolist()
    folds, cover = {}, []
    for k in FOLDS:
        f113 = bp['folds'][str(k)]
        for key in ('ordered_ids', 'groups'):
            if sha256(root / f113[key]['path']) != f113[key]['sha256']:
                raise ValueError(f'fold {k} {key} file changed')
        ordered = read(root / f113['ordered_ids']['path'])
        if uid_hash(ordered['train']) != f113['ordered_ids']['train_sha256'] or uid_hash(ordered['val']) != f113['ordered_ids']['val_sha256']:
            raise ValueError(f'fold {k} ordered ids changed')
        train, val = validate_fold_membership(corpus_ids, ordered, read_rows(root / f113['groups']['path']), canonical)
        if not np.array_equal(pos_weight(labels, train), np.asarray(f113['pos_weight'], np.float32)):
            raise ValueError(f'fold {k} pos_weight differs from the EXP113 registration')
        c = fold_contract(len(train), bp['common_args'])
        if any(c[x] != f113[x] for x in c) or (len(train), len(val)) != (f113['n_train'], f113['n_val']):
            raise ValueError(f'fold {k} contract differs from the EXP113 registration')
        folds[str(k)] = {**copy.deepcopy(f113), 'tag': tag_for(k)}
        cover += val
    if sorted(cover) != sorted(canonical) or len(set(cover)) != len(canonical):
        raise ValueError('fold validation sets do not partition the 4,343 studies')
    weights = weights_path(root)
    if sha256(weights) != RADIMAGENET_SHA256 or bp['init_weights']['sha256'] != RADIMAGENET_SHA256:
        raise ValueError(f'local RadImageNet weight file differs from {RADIMAGENET_SHA256}: {weights}')
    base_keys_path = root / bp['initial_state']['keys_file']
    if sha256(base_keys_path) != BASE_KEYS_SHA256:
        raise ValueError('EXP113 initial-state keys file changed')
    base_keys = read(base_keys_path)
    versions = preflight_versions(bp['environment'])
    factory = model_factory or (lambda arch, seed: e107.build_initial_model(arch, seed, weights))
    arch = bp['common_args']['arch']
    # control: the seed-42 state re-computed here must still be EXP113's (same CPU path the seed-43 state is computed with)
    model, load42 = factory(arch, BASE_SEED)
    init42, keys42 = state_hashes(model)
    del model
    if init42 != bp['initial_state_sha256'] or init42 != base_keys['combined_sha256'] or keys42 != base_keys['keys']:
        bad = mismatched_keys(keys42, base_keys['keys'])
        raise ValueError(f'seed-42 control re-computed on this machine differs from EXP113 ({len(bad)} keys, e.g. {bad[:5]})')
    if load42 != bp['init_weights']['load']:
        raise ValueError('RadImageNet load report (seed 42) differs from the EXP113 registration')
    model, load43 = factory(arch, SEED)
    init43, keys43 = state_hashes(model)
    del model
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    if load43 != bp['init_weights']['load']:
        raise ValueError('RadImageNet load report (seed 43) differs from the EXP113 registration')
    if init43 == init42:
        raise ValueError('seed-43 initial state equals the seed-42 state')
    report = seed_state_report(keys43, base_keys['keys'])
    destination.mkdir(parents=True)
    dump(destination / KEYS_FILE, {**{k: v for k, v in base_keys.items() if k not in ('seed', 'combined_sha256', 'keys')},
                                   'seed': SEED, 'combined_sha256': init43, 'keys': keys43})
    keys_sha = sha256(destination / KEYS_FILE)
    bindings = {**bp['source_bindings'], BASE_PROTOCOL + '/manifest.json': BASE_MANIFEST_SHA256, RUNNER: sha256(root / RUNNER), TOOL: sha256(root / TOOL),
                PROTOCOL + '/' + KEYS_FILE: keys_sha}
    seed42_control = dict(recomputed_on='cpu', initial_state_sha256=init42, equal_to_exp113=True, keys_file=bp['initial_state']['keys_file'],
                          keys_sha256=BASE_KEYS_SHA256, **{k: v for k, v in report.items() if k != 'differing'},
                          head_keys_differing=report['differing'])
    manifest = copy.deepcopy(bp)
    manifest.update(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), source_bindings=bindings, seed=SEED, initial_state_sha256=init43,
        initial_state={**bp['initial_state'], 'keys_file': PROTOCOL + '/' + KEYS_FILE, 'seed': SEED,
                       'rule': bp['initial_state']['rule'].replace(f'seed_all({BASE_SEED})', f'seed_all({SEED})'),
                       'enforcement': ENFORCEMENT, 'seed42_control': seed42_control},
        folds=folds,
        tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=OOF_DIR, compare_dir=COMPARE_DIR,
                  exp113_prefix=bp['tags']['prefix'], exp113_oof_dir=bp['tags']['oof_dir'], exp107_prefix=bp['tags']['exp107_prefix']),
        local={**copy.deepcopy(bp['local']), 'initial_state': ENFORCEMENT, 'runner': RUNNER, 'guard': TOOL, 'runner_rule': LOCAL_RULE,
               'queue': QUEUE_RULE, 'comparison': COMPARE_RULE,
               'seed': dict(value=SEED, control=BASE_SEED, changes='head initialisation, window-sampling streams sha256(seed, epoch, uid, stream) '
                                                                   'and the DataLoader shuffle generator; nothing else'),
               'prepare_preflight': dict(versions=versions, initial_state_recomputed_on='cpu', seed42_control_equal_to_exp113=True,
                                         load_report_equal_to_exp113=True, backbone_equal_to_exp113=True,
                                         head_keys_differing_from_seed42=report['head_keys_differing_from_seed42']),
               'estimate': ESTIMATE},
        evaluation={**bp['evaluation'], 'oof_dir': OOF_DIR, 'label_sets': 'dread only; OOF scored with Dread (np.rint) as EXP099',
                    'seed_comparison': COMPARE_RULE},
        caveats=[c for c in bp['caveats'] if not c.startswith(('single seed', 'local RTX 4070 Ti SUPER trajectories'))] + CAVEATS_S43,
        derived_from=dict(path=BASE_PROTOCOL + '/manifest.json', sha256=BASE_MANIFEST_SHA256, seed=bp['seed'], arch=arch, tag_prefix=bp['tags']['prefix'],
                          oof_dir=bp['tags']['oof_dir'], keys_sha256=BASE_KEYS_SHA256, exp107=bp['derived_from']['path'],
                          exp107_sha256=bp['derived_from']['sha256'], changed_fields=list(CHANGED_FIELDS), fold_changed_fields=list(FOLD_CHANGED_FIELDS),
                          initial_state_changed_fields=list(INITIAL_STATE_CHANGED), changed_args={}, changed_seed={'from': BASE_SEED, 'to': SEED}),
        label_sets=dict(default=DEFAULT_LABEL_SET, allowed=[DEFAULT_LABEL_SET],
                        rule='EXP118 is the dread seed-43 control only (no label-sets/ registry; the runner entry refuses another --label-set); a '
                             'seed-43 run of another label set needs its own registration'))
    dump(destination / 'manifest.json', manifest)
    out = dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, written_to=str(destination), manifest_sha256=sha256(destination / 'manifest.json'),
               seed=SEED, initial_state_sha256=init43, keys_file_sha256=keys_sha, seed42_control_sha256=init42, keys=len(keys43),
               head_keys_differing_from_seed42=report['head_keys_differing_from_seed42'], head_keys=report['head_keys'], versions=versions,
               total_steps=manifest['total_steps'], tags=[v['tag'] for v in folds.values()])
    print(json.dumps(out, indent=1))
    return manifest


# ------------------------------------------------------------------ runner guard (called by the shared EXP107 runner as `guard`)
def validate_arguments(args, root):
    """Called by the runner before any training import. Returns (protocol, fold spec, ordered ids, output root)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p.get('schema') != SCHEMA or manifest_path.resolve() != _manifest_path(root).resolve():
        raise ValueError('wrong protocol')
    if p.get('derived_from', {}).get('sha256') != BASE_MANIFEST_SHA256 or p['tags']['prefix'] != TAG_PREFIX or p.get('seed') != SEED:
        raise ValueError('not the EXP118 registration')
    if p['label_set']['name'] != DEFAULT_LABEL_SET or LABEL_SET != DEFAULT_LABEL_SET:
        raise ValueError('EXP118 runs the dread label set only')
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol: KNEE_REMOTE_RELAX_ENV must not be set (environment and initial state are enforced)')
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
    weights = Path(args.init_weights)
    if weights.resolve() != weights_path(root).resolve():
        raise ValueError('init weights must be the local RadImageNet file ' + WEIGHTS_LOCAL)
    if not weights.is_file() or sha256(weights) != p['init_weights']['sha256'] or p['init_weights']['sha256'] != RADIMAGENET_SHA256:
        raise ValueError('init weights differ from the registered RadImageNet file')
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
    elif out_root != default_root:
        raise ValueError('local full runs write to artifacts/experiments only (no --output-root)')
    return p, fold, ordered, out_root


def runner_argv(p, fold, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None):
    """EXP113's runner argv for the same fold and mode (recipe flags, smoke overrides, input and weight paths from e113.runner_argv, unchanged)
    with the entry, protocol and tag replaced by EXP118's and EXP113's `--label-set dread` dropped; --seed is this manifest's (43)."""
    root = Path(root)
    if p.get('schema') != SCHEMA or p.get('seed') != SEED:
        raise ValueError('runner_argv needs the EXP118 manifest')
    argv = e113.runner_argv(p, fold, mode, root=root, python=python, workers=LOCAL_WORKERS if workers is None else int(workers),
                            output_root=output_root, smoke=smoke, label_set=DEFAULT_LABEL_SET)
    swaps = [(1, str(root / e113.RUNNER), str(root / RUNNER)),
             (argv.index('--protocol') + 1, str(root / e113.PROTOCOL / 'manifest.json'), str(root / PROTOCOL / 'manifest.json')),
             (argv.index('--tag') + 1, e113.tag_for(fold, mode), tag_for(fold, mode)),
             (argv.index('--seed') + 1, str(SEED), str(SEED))]
    for i, old, new in swaps:
        if argv[i] != old:
            raise AssertionError(f'EXP113 argv drifted at {i}: {argv[i]!r} != {old!r}')
        argv[i] = new
    i = argv.index('--label-set')
    if argv[i + 1] != DEFAULT_LABEL_SET:
        raise AssertionError('EXP113 argv label set drifted')
    del argv[i:i + 2]
    return argv


def queue_entries(folds, after=DEFAULT_AFTER, workers=LOCAL_WORKERS, root=ROOT):
    """tools/gpu_queue.py job list: one full fit per fold, chained, continue_on_failure on every entry (F0 runs even if `after` failed)."""
    p = _protocol(root)
    entries, previous = [], after
    for k in folds:
        if str(k) not in p['folds']:
            raise SystemExit(f'unregistered fold {k}')
        e = dict(tag=tag_for(k), command=runner_argv(p, k, 'full', root=root, workers=workers))
        if previous:
            e['after'] = previous
        e['continue_on_failure'] = True
        entries.append(e); previous = e['tag']
    return entries


def validate_cli(fold):
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(_protocol(), fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(fold, out, train=16, val=2, epochs=1, workers=2):
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
        s = read(run / 'summary.json'); prov = s['provenance']
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], val=s['val'], seed=prov['seed'],
                      label_set=prov['label_set'], initial_state_sha256=prov['initial_state_sha256'],
                      initial_state_matches_registered=prov['initial_state_matches_registered'], cuda_initialized=prov.get('cuda_initialized'),
                      guard_sha256=prov['guard_sha256'], init_weights_sha256=prov['init_weights']['weights_sha256'], history=s['history'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification and assembly of local fold runs
def verify_run(run, p, k, manifest_sha):
    """Problems (empty list = complete and consistent) for one local EXP118 fold directory."""
    import numpy as np
    run = Path(run); fold = p['folds'][str(k)]; tag = tag_for(k)
    missing = [n for n in RUN_FILES if not (run / n).is_file()]
    if missing:
        return [f'fold {k}: missing ' + ', '.join(missing)]
    problems = []
    inv = read(run / 'output-inventory.json')
    for name, rec in inv.items():
        if not (run / name).is_file() or sha256(run / name) != rec['sha256']:
            problems.append(f'fold {k}: {name} differs from output-inventory.json')
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expect = dict(fold=k, label_set=DEFAULT_LABEL_SET, label_semantic_sha256=p['label_set']['semantic_sha256'], label_file_sha256=p['label_set']['sha256'],
                  protocol_sha256=manifest_sha, eval_ids_ordered_sha256=fold['ordered_ids']['val_sha256'],
                  train_ids_ordered_sha256=fold['ordered_ids']['train_sha256'], training_mode='full', checkpoint_selection='fixed-final', seed=SEED,
                  arch=p['common_args']['arch'], input_policy='center3', norm='radimagenet', remote_relaxed=False,
                  initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True, environment_matches_baseline=True,
                  runner_sha256=p['source_bindings'][SHARED_RUNNER], guard_sha256=p['source_bindings'][TOOL])
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'fold {k}: provenance {key} = {prov.get(key)!r}, expected {value!r}')
    if (prov.get('init_weights') or {}).get('weights_sha256') != RADIMAGENET_SHA256:
        problems.append(f'fold {k}: init weights are not the registered RadImageNet file')
    if not np.array_equal(np.asarray(prov.get('actual_pos_weight') or [], np.float32), np.asarray(fold['pos_weight'], np.float32)):
        problems.append(f'fold {k}: pos_weight differs from the registration')
    if s.get('optimizer_updates') != fold['steps'] or s.get('skipped_updates', 10 ** 9) > fold['max_fp16_skips']:
        problems.append(f"fold {k}: updates {s.get('optimizer_updates')} (expected {fold['steps']}), skipped {s.get('skipped_updates')}")
    if s.get('selected_epoch_1based') != p['common_args']['epochs'] or s.get('train') != fold['n_train'] or s.get('val') != fold['n_val']:
        problems.append(f'fold {k}: epoch/count mismatch')
    if run.name != tag or (s.get('args') or {}).get('tag') != tag or (s.get('args') or {}).get('seed') != SEED:
        problems.append(f'fold {k}: run directory / tag / seed is not {tag} / {SEED}')
    return problems


def assemble(runs_dir=None, k36_override=None, root=ROOT):
    import pandas as pd
    root = Path(root); k36_override = k36_override or {}
    manifest_path = _manifest_path(root)
    p = read(manifest_path); manifest_sha = sha256(manifest_path)
    runs = Path(runs_dir).resolve() if runs_dir else root / 'artifacts' / 'experiments'
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
        prov, s = read(run / 'provenance.json'), read(run / 'summary.json')
        frames24[k] = pd.read_csv(run / 'oof_best.csv', dtype={UID: str})
        k36 = Path(k36_override[k]) if k in k36_override else run / EXTRA_EVAL['file']
        if k36.is_file():
            frames36[k] = pd.read_csv(k36, dtype={UID: str})
        sources[k] = dict(run=display_path(run, root), oof_k24_sha256=sha256(run / 'oof_best.csv'), oof_k36=display_path(k36, root) if k36.is_file() else None,
                          oof_k36_sha256=sha256(k36) if k36.is_file() else None, checkpoint_sha256=read(run / 'output-inventory.json')['model_best.pt']['sha256'],
                          seed=prov.get('seed'), initial_state_sha256=prov.get('initial_state_sha256'), environment=prov.get('environment'),
                          fp16_skipped_updates=s.get('skipped_updates'), total_seconds=s.get('total_seconds'))
    if problems:
        print(json.dumps(dict(status='INCOMPLETE', problems=problems), indent=1))
        raise SystemExit(2)
    result = k99.assemble_and_score(frames24, frames36, val_ids, root / DREAD)
    out.mkdir(parents=True)
    result['oof_k24'].to_csv(out / 'oof_k24.csv', index=False)
    if result['oof_k36'] is not None:
        result['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp118-oof-v1', label_set=DEFAULT_LABEL_SET, label_set_sha256=p['label_set']['sha256'], observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol=PROTOCOL, protocol_sha256=manifest_sha, seed=SEED, exp113_protocol_sha256=BASE_MANIFEST_SHA256, runs_dir=display_path(runs, root),
                rows=int(len(result['oof_k24'])), sources={str(k): v for k, v in sources.items()},
                outputs={f: sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv') if (out / f).exists()},
                scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                metrics=result['metrics'], k36_missing_folds=result['k36_missing'],
                caveats=['weak Dread labels', 'fold 0 val = exposed G1 V888; folds 1-4 models saw G1 val in training', 'pooled AUC mixes 5 models',
                         'seed 43 of the EXP113 local protocol (enforced environment and seed-43 initial state); pairs with EXP113 seed 42 as a '
                         'same-recipe seed contrast (trajectory noise), see `compare`',
                         'RadImageNet ResNet50, centre slice x3, 2x-1 normalisation; RN_clip recipe otherwise'])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    md = k99.result_markdown(DEFAULT_LABEL_SET, body).replace(f'# EXP099 B8 5-fold OOF ({DEFAULT_LABEL_SET})',
                                                              f'# EXP118 RadImageNet ResNet50 5-fold OOF, seed 43 ({DEFAULT_LABEL_SET}, local RTX 4070 Ti SUPER)', 1)
    (out / 'result.md').write_text(md, encoding='utf-8')
    print(json.dumps(dict(rows=body['rows'], metrics={k: {kk: (round(vv['macro12'], 6) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
                                                      for k, v in body['metrics'].items()}, k36_missing_folds=body['k36_missing_folds']), indent=1))
    return body


# ------------------------------------------------------------------ seed 43 vs seed 42 (threshold calibration, descriptive)
def _spread(values):
    import numpy as np
    v = np.asarray(values, np.float64)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None, min=float(v.min()), max=float(v.max()),
                mean_abs=float(np.abs(v).mean()), max_abs=float(np.abs(v).max()))


def adoption_rule(dread, consensus, threshold=ADOPTION_THRESHOLD):
    """D-B-042 rule applied to a pure seed change in both directions; either 'adoption' is a false positive of the rule."""
    up = dread['delta'] >= threshold and dread['ci95'][0] > 0 and consensus['delta'] >= 0
    down = -dread['delta'] >= threshold and dread['ci95'][1] < 0 and consensus['delta'] <= 0
    return dict(threshold=threshold, adopts_seed43_over_seed42=bool(up), adopts_seed42_over_seed43=bool(down), false_adoption=bool(up or down))


def _clean(value):
    """JSON-safe copy: a NaN (a target without both classes in some scope) becomes null."""
    import math
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return None if isinstance(value, float) and math.isnan(value) else value


def compare(root=ROOT, out_dir=None, draws=DRAWS, seed=BOOT_SEED):
    """EXP118 seed-43 OOF minus EXP113 seed-42 OOF (both dread, same folds and studies) on dread / flight_v4 / relabel_v1 / consensus:
    pooled with a paired component bootstrap, per fold, per target, Spearman, and the D-B-042 rule applied to this pure seed change."""
    import numpy as np
    from tools import exp115_vlm_eval as mr
    root = Path(root)
    out = root / (out_dir or COMPARE_DIR)
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    sr, sf = e113._checked_oof(root / OOF_DIR, 'exp118-oof-v1')
    cr, cf = e113._checked_oof(root / BASE_OOF_DIR, 'exp113-oof-v1')
    if sr.get('protocol_sha256') != sha256(_manifest_path(root)) or sr.get('seed') != SEED:
        raise SystemExit('seed-43 OOF does not belong to the current EXP118 registration')
    if cr.get('protocol_sha256') != BASE_MANIFEST_SHA256 or cr.get('seed') != BASE_SEED:
        raise SystemExit('seed-42 OOF does not belong to the EXP113 registration')
    p = _protocol(root)
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    keys = [k for k in ('k24', 'k36') if k in sf and k in cf]
    if PRIMARY_K not in keys:
        raise SystemExit(f'{PRIMARY_K} OOF missing on one side')
    ids = sf[keys[0]][UID].tolist()
    refs = mr.references(ids, root)
    if tuple(refs) != tuple(mr.REFERENCES):
        raise SystemExit('reference scorer changed its reference set')
    idx = mr.bootstrap_index(mr.component_groups(ids, root), draws, seed)
    index = {u: i for i, u in enumerate(ids)}
    metrics = {}
    for key in keys:
        S, C = sf[key], cf[key]
        if list(S.columns) != [UID] + T or list(C.columns) != [UID] + T or S[UID].tolist() != ids or C[UID].tolist() != ids:
            raise SystemExit(f'{key}: seed-43 and seed-42 OOF tables differ in studies, order or columns')
        A, B = S[T].to_numpy(np.float64), C[T].to_numpy(np.float64)
        pooled = {r: mr.compare(A, B, refs[r], idx) for r in mr.REFERENCES}
        for side, value, recorded in (('seed 43', pooled['dread']['candidate'], sr), ('seed 42', pooled['dread']['baseline'], cr)):
            if abs(value - recorded['metrics'][key]['pooled']['macro12']) > 1e-9:
                raise SystemExit(f'{key}: recomputed {side} pooled Dread macro differs from its result.json')
        per_fold = {}
        for k, fids in sorted(val_ids.items()):
            rows = np.array([index[u] for u in fids])
            m = {}
            for r in mr.REFERENCES:
                Y, M, mc = refs[r]
                sub = (Y[rows], M[rows], mc)
                a, b = mr.score(A[rows], sub)['macro'], mr.score(B[rows], sub)['macro']
                m[r] = dict(seed43=a, seed42=b, delta=a - b)
            for side, value, recorded in (('seed 43', m['dread']['seed43'], sr), ('seed 42', m['dread']['seed42'], cr)):
                if abs(value - recorded['metrics'][key][f'fold{k}']['macro12']) > 1e-9:
                    raise SystemExit(f'{key} fold {k}: recomputed {side} Dread macro differs from its result.json')
            per_fold[f'fold{k}'] = dict(n=len(fids), **m)
        spread = {r: _spread([per_fold[f][r]['delta'] for f in per_fold]) for r in mr.REFERENCES}
        ens = (mr.pct_rank(A) + mr.pct_rank(B)) / 2  # exploratory: two-seed per-target rank mean (pooled ranks)
        diff = np.abs(A - B)
        metrics[key] = dict(pooled=pooled, per_fold=per_fold, fold_delta_spread=spread,
                            spearman_by_target=mr.spearman_by_target(A, B),
                            abs_probability_difference=dict(mean=float(diff.mean()), max=float(diff.max())),
                            two_seed_rank_mean_vs_seed42_exploratory={r: mr.compare(ens, B, refs[r], idx) for r in mr.REFERENCES})
    prim = metrics[PRIMARY_K]
    d = prim['pooled']['dread']
    calibration = dict(primary=PRIMARY_K, seed_delta_dread=d['delta'], seed_delta_dread_ci95=d['ci95'], seed_delta_consensus=prim['pooled']['consensus']['delta'],
                       seed_delta_flight_v4=prim['pooled']['flight_v4']['delta'], seed_delta_relabel_v1=prim['pooled']['relabel_v1']['delta'],
                       abs_seed_delta_dread_over_threshold=abs(d['delta']) / ADOPTION_THRESHOLD, two_x_abs_seed_delta_dread=2 * abs(d['delta']),
                       prior_seed_pair_dread=PRIOR_SEED_PAIR_DREAD, per_fold_dread_delta_sd=prim['fold_delta_spread']['dread']['sd'],
                       per_fold_dread_delta_max_abs=prim['fold_delta_spread']['dread']['max_abs'],
                       rule=adoption_rule(d, prim['pooled']['consensus']),
                       reading='the D-B-042 threshold (+0.004, set at about 2x the earlier seed pair 0.0021) is supported by this pair if |seed delta| '
                               'stays well below it and the rule adopts neither direction; a false adoption or |seed delta| near 0.004 means a '
                               'single-seed RadR50 contrast cannot resolve +0.004 and changes need the two-seed design of H-B-34')
    coverage = {r: dict(pairs=int(v[1].sum()), fraction=float(v[1].mean())) for r, v in refs.items()}
    body = dict(schema='exp118-compare-v1', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), label_set=DEFAULT_LABEL_SET, rows=len(ids),
                seed43=dict(dir=OOF_DIR, protocol_sha256=sr['protocol_sha256'], outputs=sr['outputs']),
                seed42=dict(dir=BASE_OOF_DIR, protocol_sha256=cr['protocol_sha256'], outputs=cr['outputs']),
                reference_scorer=dict(module=REFERENCE_SCORER, sha256=sha256(root / REFERENCE_SCORER) if (root / REFERENCE_SCORER).is_file() else None,
                                      references={r: (mr.LABELS[r][1] + ' ' + mr.LABELS[r][0]) if r in mr.LABELS else
                                                  'all four >= 0.5 agree: ' + ', '.join(mr.CONSENSUS_SETS.values()) for r in mr.REFERENCES},
                                      coverage=coverage),
                bootstrap=dict(unit='component (VALIDATION_GROUP_CONTRACT_V1)', draws=draws, seed=seed, paired='shared indices for both seeds'),
                metrics=metrics, calibration=calibration,
                interpretation='descriptive: seed 43 and seed 42 share protocol, folds, studies, labels, recipe, device and stack; only the seed '
                               '(head initialisation, window sampling, shuffle order) differs, so every delta here is trajectory noise. The '
                               'bootstrap interval covers study sampling only, not trajectory noise; the per-fold spread and the pooled delta '
                               'are the trajectory-noise evidence for the adoption threshold (one seed pair)')
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps(_clean(body), indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (out / 'result.md').write_text(compare_markdown(body), encoding='utf-8')
    print(json.dumps({k: {r: dict(seed43=round(v['pooled'][r]['candidate'], 6), seed42=round(v['pooled'][r]['baseline'], 6),
                                  delta=round(v['pooled'][r]['delta'], 6), ci95=[round(x, 6) for x in v['pooled'][r]['ci95']],
                                  fold_deltas=[round(v['per_fold'][f][r]['delta'], 6) for f in v['per_fold']]) for r in v['pooled']}
                      for k, v in metrics.items()}, indent=1))
    print(json.dumps({k: v for k, v in calibration.items() if k != 'reading'}, indent=1))
    return body


def compare_markdown(body):
    refs = list(next(iter(body['metrics'].values()))['pooled'])
    c = body['calibration']
    lines = ['# EXP118 seed 43 vs EXP113 seed 42 (local RadR50 5-fold, same recipe): seed spread', '', body['interpretation'], '',
             f"Rows {body['rows']}; component bootstrap {body['bootstrap']['draws']} draws, seed {body['bootstrap']['seed']} (paired).", '',
             f"**Calibration ({c['primary']})**: seed delta Dread {c['seed_delta_dread']:+.4f} [{c['seed_delta_dread_ci95'][0]:+.4f}, "
             f"{c['seed_delta_dread_ci95'][1]:+.4f}], consensus {c['seed_delta_consensus']:+.4f}; |delta| / threshold "
             f"{c['abs_seed_delta_dread_over_threshold']:.2f}; per-fold Dread delta SD {c['per_fold_dread_delta_sd']:.4f}, max |delta| "
             f"{c['per_fold_dread_delta_max_abs']:.4f}; earlier seed pair {c['prior_seed_pair_dread']:.4f}; D-B-042 rule false adoption: "
             f"{c['rule']['false_adoption']}", '', c['reading'], '']
    for key, m in body['metrics'].items():
        lines += [f'## {key.upper()}', '', '| reference | seed 43 | seed 42 | 43 - 42 | 95% CI |', '|---|---:|---:|---:|---|']
        for r in refs:
            v = m['pooled'][r]
            lines.append(f"| {r} | {v['candidate']:.6f} | {v['baseline']:.6f} | {v['delta']:+.6f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
        lines += ['', '| fold | n | ' + ' | '.join(f'{r} 43 - 42' for r in refs) + ' |', '|---|---:|' + '---:|' * len(refs)]
        for f, v in m['per_fold'].items():
            lines.append(f"| {f} | {v['n']} | " + ' | '.join(f"{v[r]['delta']:+.4f}" for r in refs) + ' |')
        lines.append('| spread (mean / SD / max abs) | | ' + ' | '.join(
            f"{m['fold_delta_spread'][r]['mean']:+.4f} / {m['fold_delta_spread'][r]['sd']:.4f} / {m['fold_delta_spread'][r]['max_abs']:.4f}" for r in refs) + ' |')
        e = m['two_seed_rank_mean_vs_seed42_exploratory']
        lines += ['', 'Two-seed rank mean minus seed 42 (exploratory): ' + ', '.join(
            f"{r} {e[r]['delta']:+.4f} [{e[r]['ci95'][0]:+.4f}, {e[r]['ci95'][1]:+.4f}]" for r in refs) + '; |p43 - p42| mean '
            f"{m['abs_probability_difference']['mean']:.4f}, max {m['abs_probability_difference']['max']:.4f}", '',
            '| target | Dread 43 - 42 (pooled) | Spearman 43 vs 42 |', '|---|---:|---:|']
        for t in T:
            lines.append(f"| {t} | {m['pooled']['dread']['targets'][t]:+.4f} | {m['spearman_by_target'][t]:.3f} |")
        lines.append('')
    return '\n'.join(lines)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--dry-run', default=None, help='write the registration to this new directory instead (outside artifacts/)')
    s = sub.add_parser('argv'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s.add_argument('--mode', choices=['full', 'smoke'], default='full'); s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    s = sub.add_parser('validate'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s = sub.add_parser('smoke'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--out', required=True)
    s.add_argument('--train', type=int, default=16); s.add_argument('--val', type=int, default=2)
    s.add_argument('--epochs', type=int, default=1); s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('queue-json'); s.add_argument('--folds', default='0,1,2,3,4')
    s.add_argument('--after', default=DEFAULT_AFTER, help=f'tag the first entry waits for (default {DEFAULT_AFTER}); "none" only for an otherwise idle GPU')
    s.add_argument('--workers', type=int, default=LOCAL_WORKERS)
    s.add_argument('--out', default=None, help=f'new JSON file for the entries (default {QUEUE_FILE}; "-" = print only)')
    s = sub.add_parser('assemble')
    s.add_argument('--runs-dir', default=None, help='directory holding the fold run directories (default artifacts/experiments)')
    s.add_argument('--k36', action='append', default=[], help='FOLD=PATH oof_k36.csv override')
    s = sub.add_parser('compare'); s.add_argument('--draws', type=int, default=DRAWS); s.add_argument('--seed', type=int, default=BOOT_SEED)
    s.add_argument('--out', default=None, help=f'new output directory relative to the workspace (default {COMPARE_DIR})')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    if a.command == 'prepare':
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        if a.dry_run and within(a.dry_run, ROOT / 'artifacts'):
            raise SystemExit('--dry-run must be outside artifacts/')
        prepare(dest=a.dry_run)
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.fold, a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli(a.fold))
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.fold, a.out, a.train, a.val, a.epochs, a.workers))
    elif a.command == 'queue-json':
        entries = queue_entries([int(x) for x in a.folds.split(',')], None if a.after.lower() == 'none' else a.after, a.workers)
        out = a.out or str(ROOT / QUEUE_FILE)
        if out != '-':
            dump(out, entries)
            print('written ' + out, file=sys.stderr)
        print(json.dumps(entries, indent=1))
    elif a.command == 'assemble':
        assemble(a.runs_dir, {int(x.split('=', 1)[0]): x.split('=', 1)[1] for x in a.k36})
    else:
        compare(out_dir=a.out, draws=a.draws, seed=a.seed)


if __name__ == '__main__':
    main()
