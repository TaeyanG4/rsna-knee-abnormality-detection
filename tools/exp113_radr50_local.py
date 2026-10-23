"""EXP113 (H-B-34): the EXP107 RadImageNet ResNet50 centre-slice x3 5-fold, trained on the LOCAL RTX 4070 Ti SUPER through tools/gpu_queue.py.

  python tools/exp113_radr50_local.py prepare [--dry-run DIR]       # register EXP113_RADR50_LOCAL_PROTOCOL_V1 (CPU; re-computes the seed-42 init)
  python tools/exp113_radr50_local.py register-labels --name NAME  # bind an EXP099 label set (relabel_v1, flight_v4, ...) with its recorded pos_weight
  python tools/exp113_radr50_local.py argv --fold K [--labels NAME] [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp113_radr50_local.py validate --fold K [--labels NAME]          # runner --validate-only on CPU, no output
  python tools/exp113_radr50_local.py smoke --fold K --out DIR [--labels NAME] [--train 16 --val 2 --epochs 1 --workers 2]   # CPU fit, DIR outside artifacts/
  python tools/exp113_radr50_local.py queue-json --folds 0,1,2,3,4 --after TAG [--labels NAME] [--out FILE|-]   # tools/gpu_queue.py entries
  python tools/exp113_radr50_local.py assemble [--labels NAME] [--runs-dir DIR] [--k36 K=PATH]   # pooled OOF -> artifacts/experiments/EXP113_RADR50_LOCAL[_<NAME>]_OOF
  python tools/exp113_radr50_local.py compare                                     # local vs Kaggle T4 EXP107 OOF (descriptive, D-B-008)

A copy of EXP107_KFOLD_RADR50_PROTOCOL_V1 (manifest sha256 BASE_MANIFEST_SHA256, not changed): the same EXP099 folds, Dread labels, per-fold
pos_weight, seed 42, RadImageNet weights (sha256 08629f7e...), RN_clip recipe, K12 / K24 / K36, windows, input policy and normalisation, and the
same seed-42 initial state (re-computed on CPU at prepare and required to equal EXP107's per key). What changes is where it runs: the local GPU
with the registered local environment and initial state ENFORCED (no KNEE_REMOTE_RELAX_ENV), the local weight file, outputs in
artifacts/experiments/<tag>/, 12 loader workers, and the names (EXP113_RADR50_LOCAL_F<K>). Training runs the EXP107 runner file unchanged
(training/runners/exp107_kfold_radr50.py) through training/runners/exp113_radr50_local.py, which only points the runner's guard import at
this module. Label sets: dread is the manifest's own; another EXP099 label set (pos_weight computed from its own labels) can be bound with
register-labels and selected with --labels / --label-set; OOF is always scored with Dread (np.rint), as EXP099 does.
A local run and the Kaggle T4 run are independent trajectories (D-B-008): `compare` reports the difference, it never passes or fails it.
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
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
# names the shared EXP107 runner reads from its guard module (`guard.<name>`), re-exported unchanged
from tools.exp107_kfold_radr50 import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                       CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                       DREAD, KEYS_FILE, FOLDS, EXTRA_EVAL, RUN_FILES, fold_contract, state_hashes, mismatched_keys, dump,
                                       train_python, within, display_path, _set_flag)

SCHEMA = 'exp113-radr50-local-v1'
LABEL_SCHEMA = 'exp113-label-set-v1'
BASE_PROTOCOL = e107.PROTOCOL
BASE_MANIFEST_SHA256 = 'b51ce3b7e9aeff8bded1cadf1afe50d3cb926df4df139237e3cbaa778836cc81'
BASE_KEYS_SHA256 = '359f3710c7a12d054379bdc2849add4d25bf9732703a9c292086138709b7cc33'
BASE_OOF_DIR = e107.OOF_DIR  # the Kaggle T4 OOF (exp107-oof-v1) that `compare` reads
SHARED_RUNNER = e107.RUNNER
PROTOCOL = 'artifacts/experiments/EXP113_RADR50_LOCAL_PROTOCOL_V1'
LABEL_SETS_DIR = PROTOCOL + '/label-sets'
QUEUE_FILE = PROTOCOL + '/queue-entries.json'
RUNNER = 'training/runners/exp113_radr50_local.py'
TOOL = 'tools/exp113_radr50_local.py'
SEED = 42
TAG_PREFIX = 'EXP113_RADR50_LOCAL'
OOF_DIR = 'artifacts/experiments/EXP113_RADR50_LOCAL_OOF'
COMPARE_DIR = 'artifacts/experiments/EXP113_RADR50_LOCAL_VS_EXP107'
FOLD_PROTOCOL = e107.FOLD_PROTOCOL
WEIGHTS_LOCAL = e107.WEIGHTS_LOCAL
RADIMAGENET_SHA256 = e107.RADIMAGENET_SHA256
LOCAL_GPU = 'NVIDIA GeForce RTX 4070 Ti SUPER'
LOCAL_WORKERS = 12
DEFAULT_LABEL_SET = 'dread'
LABEL_SET = DEFAULT_LABEL_SET  # the label set of this process; the runner entry binds it (bind_label_set) before the shared runner reads it
# manifest fields that may differ from the EXP107 registration; everything else must be equal (tests/test_exp113_radr50_local.py)
CHANGED_FIELDS = ('schema', 'created_at', 'plan_item', 'source_bindings', 'initial_state', 'folds', 'tags', 'remote', 'local', 'evaluation', 'caveats',
                  'derived_from', 'label_sets')
FOLD_CHANGED_FIELDS = ('tag',)
INITIAL_STATE_CHANGED = ('keys_file', 'enforcement')
ENFORCEMENT = ('local runner: the seed-42 state must equal initial_state_sha256 exactly (every key, head included) and the runtime environment '
               'must equal the registered local environment; KNEE_REMOTE_RELAX_ENV is refused by the guard')
ESTIMATE = dict(
    seconds_per_fold=1200, range_seconds_per_fold=[1150, 1450], five_folds_hours=[1.6, 2.0],
    basis='EXP107 on Kaggle T4: 3,452-3,478 s per fold = 168 s/epoch x 20 + K24 77 s + K36 61 s evaluation. Same RN_clip contract with ResNet34: '
          'T4 (EXP101) 104 s/epoch vs local RTX 4070 Ti SUPER (EXP084_RN_CLIP_G1, 12 workers) 34.5 s/epoch, ratio 0.33 (726 s per fit). '
          'ResNet50 at the same ratio: 55 s/epoch -> ~1,110 s training + ~60-90 s K24/K36 evaluation (2 eval workers) = ~20 min per fold. '
          'Upper end: a GPU-bound 2.5x 4070 Ti SUPER / T4 speed-up for ResNet50 FP16 (67 s/epoch) = ~24 min. Peak memory on T4 4.4 GiB.')
CAVEATS_LOCAL = [
    'local RTX 4070 Ti SUPER trajectories (Windows window pipeline, torch 2.5.1+cu121 / timm 1.0.19, enforced seed-42 initial state) differ from '
    'the Kaggle T4 EXP107 trajectories (Linux, torch 2.10.0+cu128 / timm 1.0.26, relaxed initial state); local - T4 is a device/stack/trajectory '
    'difference (D-B-008), never a paired contrast; local recipe contrasts pair with this local control only',
]
LOCAL_RULE = ('training/runners/exp113_radr50_local.py points the shared runner\'s guard import (tools.exp107_kfold_radr50) at '
              'tools/exp113_radr50_local.py for its own process and calls the unchanged EXP107 runner main(); it consumes --label-set NAME '
              '(default dread) and supplies OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 when they are absent, before torch is '
              'imported (D-B-017: a queue runner started without them failed a job in 3 s); any other value is still refused by the guard')


def label_set_name(name):
    if not k99.NAME_RE.match(str(name)):
        raise ValueError('label-set name must match ' + k99.NAME_RE.pattern)
    return str(name)


def tag_for(fold, mode='full', label_set=DEFAULT_LABEL_SET):
    middle = '' if label_set == DEFAULT_LABEL_SET else '_' + label_set_name(label_set).upper()
    return f'{TAG_PREFIX}{middle}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def oof_dir_for(label_set=DEFAULT_LABEL_SET):
    return OOF_DIR if label_set == DEFAULT_LABEL_SET else f'artifacts/experiments/{TAG_PREFIX}_{label_set_name(label_set).upper()}_OOF'


def queue_file_for(label_set=DEFAULT_LABEL_SET):
    return QUEUE_FILE if label_set == DEFAULT_LABEL_SET else f'{PROTOCOL}/queue-entries-{label_set_name(label_set)}.json'


def bind_label_set(name):
    """Select the label set of this process (the runner entry calls this before the shared runner imports the guard)."""
    global LABEL_SET
    LABEL_SET = label_set_name(name)
    return LABEL_SET


def weights_path(root=ROOT):
    """Always the local copy (KNEE_RADIMAGENET_R50 is a remote-only override and is ignored here)."""
    return Path(root) / WEIGHTS_LOCAL


def _protocol(root=ROOT):
    return read(Path(root) / PROTOCOL / 'manifest.json')


def _manifest_path(root=ROOT):
    return Path(root) / PROTOCOL / 'manifest.json'


# ------------------------------------------------------------------ prepare
def preflight_versions(registered):
    """CPU-only version check against the registered local environment (no CUDA call; GPU name and cuDNN are checked by the runner)."""
    import torch
    import timm
    import numpy
    import pydicom
    import cv2
    got = dict(torch=torch.__version__, cuda=torch.version.cuda, timm=timm.__version__, numpy=numpy.__version__, pydicom=pydicom.__version__, cv2=cv2.__version__)
    bad = {k: (v, registered.get(k)) for k, v in got.items() if v != registered.get(k)}
    if bad:
        raise ValueError('local package versions differ from the registered environment: ' + json.dumps(bad))
    return got


def prepare(root=ROOT, model_factory=None, dest=None):
    import numpy as np
    root = Path(root)
    destination = Path(dest) if dest else root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + str(destination))
    base_path = root / BASE_PROTOCOL / 'manifest.json'
    if sha256(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError('EXP107 manifest changed')
    bp = read(base_path)
    if bp['schema'] != e107.SCHEMA or bp['seed'] != SEED or bp['common_args'] != e107.EXPECTED_RECIPE or bp['tags']['prefix'] != e107.TAG_PREFIX:
        raise ValueError('unexpected EXP107 registration')
    for path, expected in bp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('EXP107 binding changed: ' + path)
    for name, expected in bp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    labels, semantic = k99._load_labels(root / DREAD)
    if semantic != bp['label_set']['semantic_sha256'] or sha256(root / DREAD) != bp['label_set']['sha256'] or bp['label_set']['name'] != DEFAULT_LABEL_SET:
        raise ValueError('Dread file differs from the EXP107 registration')
    if sha256(root / bp['fold_protocol']['path']) != bp['fold_protocol']['sha256'] or sha256(root / bp['paths']['folds']) != bp['fold_protocol']['folds_csv_sha256']:
        raise ValueError('EXP099 fold protocol changed')
    canonical = canonical_order(read(root / bp['paths']['base_ordered_ids']))
    corpus_dir = root / bp['paths']['corpus']
    if sha256(corpus_dir / 'manifest.json') != bp['corpus_manifest_sha256']:
        raise ValueError('corpus manifest differs from the EXP107 registration')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=True).astype(str).tolist()
    folds, cover = {}, []
    for k in FOLDS:
        f107 = bp['folds'][str(k)]
        for key in ('ordered_ids', 'groups'):
            if sha256(root / f107[key]['path']) != f107[key]['sha256']:
                raise ValueError(f'fold {k} {key} file changed')
        ordered = read(root / f107['ordered_ids']['path'])
        if uid_hash(ordered['train']) != f107['ordered_ids']['train_sha256'] or uid_hash(ordered['val']) != f107['ordered_ids']['val_sha256']:
            raise ValueError(f'fold {k} ordered ids changed')
        train, val = validate_fold_membership(corpus_ids, ordered, read_rows(root / f107['groups']['path']), canonical)
        if not np.array_equal(pos_weight(labels, train), np.asarray(f107['pos_weight'], np.float32)):
            raise ValueError(f'fold {k} pos_weight differs from the EXP107 registration')
        c = fold_contract(len(train), bp['common_args'])
        if any(c[x] != f107[x] for x in c) or (len(train), len(val)) != (f107['n_train'], f107['n_val']):
            raise ValueError(f'fold {k} contract differs from the EXP107 registration')
        folds[str(k)] = {**copy.deepcopy(f107), 'tag': tag_for(k)}
        cover += val
    if sorted(cover) != sorted(canonical) or len(set(cover)) != len(canonical):
        raise ValueError('fold validation sets do not partition the 4,343 studies')
    weights = weights_path(root)
    if sha256(weights) != RADIMAGENET_SHA256 or bp['init_weights']['sha256'] != RADIMAGENET_SHA256:
        raise ValueError(f'local RadImageNet weight file differs from {RADIMAGENET_SHA256}: {weights}')
    base_keys_path = root / bp['initial_state']['keys_file']
    if sha256(base_keys_path) != BASE_KEYS_SHA256:
        raise ValueError('EXP107 initial-state keys file changed')
    base_keys = read(base_keys_path)
    versions = preflight_versions(bp['environment'])
    model, load_report = (model_factory or (lambda arch, seed: e107.build_initial_model(arch, seed, weights)))(bp['common_args']['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init, keys = state_hashes(model)
    if init != bp['initial_state_sha256'] or init != base_keys['combined_sha256'] or keys != base_keys['keys']:
        bad = mismatched_keys(keys, base_keys['keys'])
        raise ValueError(f'seed-42 initial state re-computed on this machine differs from EXP107 ({len(bad)} keys, e.g. {bad[:5]})')
    if load_report != bp['init_weights']['load']:
        raise ValueError('RadImageNet load report differs from the EXP107 registration')
    destination.mkdir(parents=True)
    (destination / KEYS_FILE).write_bytes(base_keys_path.read_bytes())  # verified equal per key to the re-computation above
    if sha256(destination / KEYS_FILE) != BASE_KEYS_SHA256:
        raise AssertionError('keys file copy differs')
    bindings = {**bp['source_bindings'], BASE_PROTOCOL + '/manifest.json': BASE_MANIFEST_SHA256, RUNNER: sha256(root / RUNNER), TOOL: sha256(root / TOOL),
                PROTOCOL + '/' + KEYS_FILE: sha256(destination / KEYS_FILE)}
    manifest = copy.deepcopy(bp)
    manifest.update(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-34', source_bindings=bindings,
        initial_state={**bp['initial_state'], 'keys_file': PROTOCOL + '/' + KEYS_FILE, 'enforcement': ENFORCEMENT},
        folds=folds,
        tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '[_<LABELSET>]_F<K>[_SMOKE] (no label-set part for dread)', oof_dir=OOF_DIR,
                  oof_dir_pattern='artifacts/experiments/' + TAG_PREFIX + '[_<LABELSET>]_OOF', compare_dir=COMPARE_DIR, exp107_prefix=bp['tags']['prefix']),
        remote=None,
        local=dict(platform=f'local {LOCAL_GPU} (Windows), FP16 + GradScaler 1024, strict runtime (deterministic algorithms)', gpu=LOCAL_GPU,
                   environment='the registered environment (manifest environment: the local GPU, torch 2.5.1+cu121, cuDNN 90100, timm 1.0.19, '
                               'numpy 2.0.2, pydicom 3.0.2, cv2 5.0.0) is enforced by the shared runner',
                   initial_state=ENFORCEMENT, runner=RUNNER, shared_runner=SHARED_RUNNER, guard=TOOL, runner_rule=LOCAL_RULE,
                   workers=LOCAL_WORKERS, workers_note='resource argument (resource_args unchanged: range 0-16, default 4 for Kaggle T4); 12 = the '
                                                      'project default for local loaders; sampling is uid/epoch-deterministic, so speed only',
                   weights=WEIGHTS_LOCAL + ' (KNEE_RADIMAGENET_R50 ignored)', output_root='artifacts/experiments/<tag>/ (default only; a full run '
                                                                                         'refuses any other --output-root)',
                   queue='tools/exp113_radr50_local.py queue-json -> entries for tools/gpu_queue.py (after = the current chain tail; '
                         'continue_on_failure true, folds are independent fits); the queue runner may be started without the thread variables',
                   comparison='EXP107 (Kaggle T4) is an independent trajectory: `compare` reports local - T4 per fold, pooled, per target and '
                              'Spearman as a device/stack/trajectory difference (D-B-008), never as a pass/fail gate',
                   prepare_preflight=dict(versions=versions, initial_state_recomputed_on='cpu', initial_state_equal_to_exp107=True,
                                          load_report_equal_to_exp107=True),
                   estimate=ESTIMATE),
        evaluation={**bp['evaluation'], 'oof_dir': OOF_DIR, 'label_sets': 'every label set is scored with Dread (np.rint), as EXP099'},
        caveats=[c for c in bp['caveats'] if not c.startswith('remote T4 trajectories')] + CAVEATS_LOCAL,
        derived_from=dict(path=BASE_PROTOCOL + '/manifest.json', sha256=BASE_MANIFEST_SHA256, seed=bp['seed'], arch=bp['common_args']['arch'],
                          tag_prefix=bp['tags']['prefix'], oof_dir=bp['tags']['oof_dir'], kernel=bp['remote']['kernel'], keys_sha256=BASE_KEYS_SHA256,
                          changed_fields=list(CHANGED_FIELDS), fold_changed_fields=list(FOLD_CHANGED_FIELDS),
                          initial_state_changed_fields=list(INITIAL_STATE_CHANGED), changed_args={}),
        label_sets=dict(default=DEFAULT_LABEL_SET, dir=LABEL_SETS_DIR, source_registry=FOLD_PROTOCOL + '/label-sets',
                        rule='dread = this manifest (label_set, folds[*].pos_weight). Another name needs label-sets/NAME.json from register-labels: a '
                             'copy of the EXP099 registration (file sha256, semantic sha256, Gold exclusion, fold coverage re-checked) whose recorded '
                             'per-fold pos_weight must equal the EXP099 rule recomputed from its own training labels (pos_weight_source labels; '
                             'the shared runner recomputes it and requires equality). The runner then reads --labels and --eval-labels = that file '
                             '(the EXP107 runner only hash-checks --eval-labels and writes unscored predictions); assemble scores with Dread.'))
    dump(destination / 'manifest.json', manifest)
    report = dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, written_to=str(destination), manifest_sha256=sha256(destination / 'manifest.json'),
                  seed=SEED, initial_state_sha256=init, keys=len(keys), versions=versions, total_steps=manifest['total_steps'],
                  tags=[v['tag'] for v in folds.values()])
    print(json.dumps(report, indent=1))
    return manifest


# ------------------------------------------------------------------ label sets (EXP099 registry semantics)
def register_labels(name, root=ROOT):
    import numpy as np
    import pandas as pd
    root = Path(root)
    name = label_set_name(name)
    if name == DEFAULT_LABEL_SET:
        raise SystemExit('dread is the protocol\'s own label set (manifest label_set); nothing to register')
    manifest_path = _manifest_path(root)
    m = read(manifest_path)
    if m['schema'] != SCHEMA:
        raise SystemExit('wrong protocol')
    out = root / LABEL_SETS_DIR / f'{name}.json'
    if out.exists():
        raise SystemExit('label set already registered: ' + str(out))
    fold_manifest_path = root / FOLD_PROTOCOL / 'manifest.json'
    if sha256(fold_manifest_path) != m['fold_protocol']['sha256']:
        raise SystemExit('EXP099 fold protocol changed')
    src, src_file = k99.load_registration(root, read(fold_manifest_path), name, fold_manifest_path)
    if src.get('pos_weight_source') != 'labels':
        raise SystemExit(f'{name}: pos_weight_source {src.get("pos_weight_source")!r}; the EXP107 runner recomputes pos_weight from the training '
                         'labels and requires equality, so only label sets with pos_weight_source "labels" can be bound')
    file = relpath(root, src['path'])
    labels, semantic = k99._load_labels(file)
    if semantic != src['semantic_sha256'] or sha256(file) != src['sha256']:
        raise SystemExit(f'{name} file differs from its EXP099 registration')
    raw = pd.read_csv(file, dtype={UID: str}, usecols=[UID])[UID].astype(str).tolist()
    trains = {int(k): read(root / v['ordered_ids']['path'])['train'] for k, v in m['folds'].items()}
    k99.check_label_set(labels, raw, k99.gold_uids(root), trains)
    weights = {}
    for k, ids in trains.items():
        pw = pos_weight(labels, ids)
        if not np.array_equal(pw, np.asarray(src['pos_weight'][str(k)], np.float32)):
            raise SystemExit(f'{name} fold {k}: pos_weight recomputed from the labels differs from the EXP099 registration')
        weights[str(k)] = pw.tolist()
    eligible = set().union(*map(set, trains.values()))
    record = dict(schema=LABEL_SCHEMA, name=name, registered_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), path=src['path'], sha256=src['sha256'],
                  semantic_sha256=semantic, rows=len(raw), loaded_rows=len(labels), eligible_covered=len(eligible), gold_overlap=0,
                  extra_rows_outside_g1=len(set(labels) - eligible), protocol=PROTOCOL, protocol_sha256=sha256(manifest_path),
                  source_registration=dict(path=display_path(src_file, root), sha256=sha256(src_file), protocol_sha256=src['protocol_sha256'],
                                           registered_at=src.get('registered_at')),
                  pos_weight_source='labels', pos_weight=weights, pos_weight_equals_source_registration=True,
                  eval_labels=src['path'], scoring=f'Dread {DREAD} (np.rint) at assemble, as EXP099',
                  tags={str(k): tag_for(k, 'full', name) for k in FOLDS}, oof_dir=oof_dir_for(name))
    out.parent.mkdir(exist_ok=True)
    dump(out, record)
    print(json.dumps({k: record[k] for k in ('name', 'path', 'sha256', 'semantic_sha256', 'rows', 'eligible_covered', 'gold_overlap', 'pos_weight_source',
                                             'protocol_sha256', 'source_registration', 'oof_dir')}, indent=1))
    return record


def load_label_registration(root, p, name, manifest_path):
    root = Path(root)
    f = root / LABEL_SETS_DIR / f'{label_set_name(name)}.json'
    if not f.is_file():
        raise ValueError(f'label set {name!r} is not registered for EXP113 ({display_path(f, root)}); run register-labels --name {name}')
    r = read(f)
    if r.get('schema') != LABEL_SCHEMA or r.get('name') != name or r.get('protocol_sha256') != sha256(manifest_path):
        raise ValueError('label-set registration does not belong to this protocol')
    if sha256(relpath(root, r['path'])) != r['sha256']:
        raise ValueError('label file changed since registration: ' + r['path'])
    src = r['source_registration']
    if sha256(relpath(root, src['path'])) != src['sha256']:
        raise ValueError('EXP099 source registration changed: ' + src['path'])
    if sorted(r['pos_weight']) != sorted(p['folds']):
        raise ValueError('label-set registration does not cover every fold')
    return r, f


def protocol_view(root, p, name, manifest_path):
    """The protocol as the shared runner must see it for one label set: dread = the manifest itself; another set replaces label_set,
    paths.labels / paths.eval_labels, every fold's pos_weight and tag with the registered values."""
    name = label_set_name(name)
    view = copy.deepcopy(p)
    if name == DEFAULT_LABEL_SET:
        if p['label_set']['name'] != DEFAULT_LABEL_SET:
            raise ValueError('manifest label set is not dread')
        return view
    reg, f = load_label_registration(root, p, name, manifest_path)
    view['label_set'] = dict(name=name, path=reg['path'], sha256=reg['sha256'], semantic_sha256=reg['semantic_sha256'],
                             registration=display_path(f, root), registration_sha256=sha256(f))
    view['paths'] = {**p['paths'], 'labels': reg['path'], 'eval_labels': reg['path']}
    for k, fold in view['folds'].items():
        fold.update(pos_weight=reg['pos_weight'][k], tag=tag_for(k, 'full', name), pos_weight_source='labels')
    view['tags'] = {**p['tags'], 'oof_dir': oof_dir_for(name)}
    return view


# ------------------------------------------------------------------ runner guard (called by the shared EXP107 runner as `guard`)
def validate_arguments(args, root):
    """Called by the runner before any training import. Returns (protocol view for LABEL_SET, fold spec, ordered ids, output root)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p.get('schema') != SCHEMA or manifest_path.resolve() != _manifest_path(root).resolve():
        raise ValueError('wrong protocol')
    if p.get('derived_from', {}).get('sha256') != BASE_MANIFEST_SHA256 or p['tags']['prefix'] != TAG_PREFIX:
        raise ValueError('not the EXP113 registration')
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
    view = protocol_view(root, p, LABEL_SET, manifest_path)
    fold = view['folds'][str(args.fold)]
    if tag_for(args.fold, args.mode, LABEL_SET) != args.tag or (not smoke and fold['tag'] != args.tag):
        raise ValueError('tag/fold/label-set/mode mismatch')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, view['paths'][key]).resolve():
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
    return view, fold, ordered, out_root


def runner_argv(view, fold, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None, label_set=None):
    """Runner entry argv for one fold; `view` = protocol_view(...) for the label set (its paths are the --labels / --eval-labels files)."""
    root = Path(root)
    label_set = label_set or view['label_set']['name']
    argv = [python or train_python(), str(root / RUNNER), '--protocol', str(root / PROTOCOL / 'manifest.json'), '--label-set', label_set,
            '--fold', str(int(fold)), '--mode', mode, '--tag', tag_for(fold, mode, label_set), '--seed', str(view['seed']),
            '--workers', str(LOCAL_WORKERS if workers is None else int(workers))]
    for k, v in view['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(relpath(root, view['paths']['corpus'])), '--labels', str(relpath(root, view['paths']['labels'])),
             '--eval-labels', str(relpath(root, view['paths']['eval_labels'])), '--init-weights', str(weights_path(root))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if mode == 'smoke':
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=16, val=2, epochs=1), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def view_for(label_set=DEFAULT_LABEL_SET, root=ROOT):
    root = Path(root)
    return protocol_view(root, _protocol(root), label_set, _manifest_path(root))


def queue_entries(folds, after, label_set=DEFAULT_LABEL_SET, workers=LOCAL_WORKERS, root=ROOT):
    """tools/gpu_queue.py job list: one full fit per fold, chained, continue_on_failure (independent folds)."""
    view = view_for(label_set, root)
    entries, previous = [], after
    for k in folds:
        if str(k) not in view['folds']:
            raise SystemExit(f'unregistered fold {k}')
        e = dict(tag=tag_for(k, 'full', label_set), command=runner_argv(view, k, 'full', root=root, workers=workers, label_set=label_set))
        if previous:
            e['after'] = previous
        e['continue_on_failure'] = True
        entries.append(e); previous = e['tag']
    return entries


def local_env():
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(ROOT)}
    for k in ('KNEE_REMOTE_RELAX_ENV', e107.WEIGHTS_ENV):
        env.pop(k, None)
    return env


def validate_cli(fold, label_set=DEFAULT_LABEL_SET):
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(view_for(label_set), fold, 'full', label_set=label_set) + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(fold, out, train=16, val=2, epochs=1, workers=2, label_set=DEFAULT_LABEL_SET):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(view_for(label_set), fold, 'smoke', workers=workers, output_root=out, smoke=dict(train=train, val=val, epochs=epochs), label_set=label_set)
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for(fold, 'smoke', label_set)
    report = dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(run), files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json'); prov = s['provenance']
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], val=s['val'], seed=prov['seed'],
                      label_set=prov['label_set'], initial_state_matches_registered=prov['initial_state_matches_registered'],
                      cuda_initialized=prov.get('cuda_initialized'), init_weights_sha256=prov['init_weights']['weights_sha256'], history=s['history'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification and assembly of local fold runs
def verify_run(run, view, k, manifest_sha, label_set=DEFAULT_LABEL_SET):
    """Problems (empty list = complete and consistent) for one local EXP113 fold directory."""
    import numpy as np
    run = Path(run); fold = view['folds'][str(k)]; tag = tag_for(k, 'full', label_set)
    missing = [n for n in RUN_FILES if not (run / n).is_file()]
    if missing:
        return [f'fold {k}: missing ' + ', '.join(missing)]
    problems = []
    inv = read(run / 'output-inventory.json')
    for name, rec in inv.items():
        if not (run / name).is_file() or sha256(run / name) != rec['sha256']:
            problems.append(f'fold {k}: {name} differs from output-inventory.json')
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expect = dict(fold=k, label_set=label_set, label_semantic_sha256=view['label_set']['semantic_sha256'], label_file_sha256=view['label_set']['sha256'],
                  protocol_sha256=manifest_sha, eval_ids_ordered_sha256=fold['ordered_ids']['val_sha256'],
                  train_ids_ordered_sha256=fold['ordered_ids']['train_sha256'], training_mode='full', checkpoint_selection='fixed-final', seed=SEED,
                  arch=view['common_args']['arch'], input_policy='center3', norm='radimagenet', remote_relaxed=False,
                  initial_state_sha256=view['initial_state_sha256'], initial_state_matches_registered=True, environment_matches_baseline=True,
                  runner_sha256=view['source_bindings'][SHARED_RUNNER], guard_sha256=view['source_bindings'][TOOL])
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'fold {k}: provenance {key} = {prov.get(key)!r}, expected {value!r}')
    if (prov.get('init_weights') or {}).get('weights_sha256') != RADIMAGENET_SHA256:
        problems.append(f'fold {k}: init weights are not the registered RadImageNet file')
    if not np.array_equal(np.asarray(prov.get('actual_pos_weight') or [], np.float32), np.asarray(fold['pos_weight'], np.float32)):
        problems.append(f'fold {k}: pos_weight differs from the {label_set} registration')
    if s.get('optimizer_updates') != fold['steps'] or s.get('skipped_updates', 10 ** 9) > fold['max_fp16_skips']:
        problems.append(f"fold {k}: updates {s.get('optimizer_updates')} (expected {fold['steps']}), skipped {s.get('skipped_updates')}")
    if s.get('selected_epoch_1based') != view['common_args']['epochs'] or s.get('train') != fold['n_train'] or s.get('val') != fold['n_val']:
        problems.append(f'fold {k}: epoch/count mismatch')
    if run.name != tag or (s.get('args') or {}).get('tag') != tag:
        problems.append(f'fold {k}: run directory / tag is not {tag}')
    return problems


def assemble(label_set=DEFAULT_LABEL_SET, runs_dir=None, k36_override=None, root=ROOT):
    import pandas as pd
    root = Path(root); k36_override = k36_override or {}
    manifest_path = _manifest_path(root)
    p = read(manifest_path); manifest_sha = sha256(manifest_path)
    view = protocol_view(root, p, label_set, manifest_path)
    runs = Path(runs_dir).resolve() if runs_dir else root / 'artifacts' / 'experiments'
    out = root / oof_dir_for(label_set)
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in view['folds'].items()}
    frames24, frames36, sources, problems = {}, {}, {}, []
    for k in sorted(val_ids):
        run = runs / tag_for(k, 'full', label_set)
        bad = verify_run(run, view, k, manifest_sha, label_set)
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
    body = dict(schema='exp113-oof-v1', label_set=label_set, label_set_sha256=view['label_set']['sha256'], observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol=PROTOCOL, protocol_sha256=manifest_sha, seed=SEED, exp107_protocol_sha256=BASE_MANIFEST_SHA256, runs_dir=display_path(runs, root),
                rows=int(len(result['oof_k24'])), sources={str(k): v for k, v in sources.items()},
                outputs={f: sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv') if (out / f).exists()},
                scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                metrics=result['metrics'], k36_missing_folds=result['k36_missing'],
                caveats=['weak Dread labels', 'fold 0 val = exposed G1 V888; folds 1-4 models saw G1 val in training', 'pooled AUC mixes 5 models',
                         'local RTX 4070 Ti SUPER trajectories (enforced environment and seed-42 initial state); EXP107 (Kaggle T4) is an independent '
                         'trajectory of the same protocol, not a paired arm (D-B-008)',
                         'RadImageNet ResNet50, centre slice x3, 2x-1 normalisation; RN_clip recipe otherwise'])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    md = k99.result_markdown(label_set, body).replace(f'# EXP099 B8 5-fold OOF ({label_set})',
                                                      f'# EXP113 RadImageNet ResNet50 5-fold OOF ({label_set}, local RTX 4070 Ti SUPER)', 1)
    (out / 'result.md').write_text(md, encoding='utf-8')
    print(json.dumps(dict(rows=body['rows'], metrics={k: {kk: (round(vv['macro12'], 6) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
                                                      for k, v in body['metrics'].items()}, k36_missing_folds=body['k36_missing_folds']), indent=1))
    return body


# ------------------------------------------------------------------ local vs Kaggle T4 (descriptive)
def spearman(a, b):
    """Pearson correlation of average ranks; None when either side is constant."""
    import numpy as np
    ra, rb = k99.avg_rank(np.asarray(a, np.float64)), k99.avg_rank(np.asarray(b, np.float64))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _mean_defined(values):
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def _fmt(value):
    return '-' if value is None else f'{value:.4f}'


def _checked_oof(directory, schema):
    import pandas as pd
    r = read(directory / 'result.json')
    if r.get('schema') != schema or r.get('label_set') != DEFAULT_LABEL_SET:
        raise SystemExit(f'{directory}: expected a {schema} dread OOF')
    frames = {}
    for name, digest in r['outputs'].items():
        if sha256(directory / name) != digest:
            raise SystemExit(f'{directory / name} differs from its result.json hash')
        frames[name.split('.')[0].split('_')[1]] = pd.read_csv(directory / name, dtype={UID: str})
    return r, frames


def compare(root=ROOT, out_dir=None):
    """Local EXP113 OOF vs the Kaggle T4 EXP107 OOF (both dread, same folds and studies): per-fold and pooled macro12, per target, Spearman."""
    import numpy as np
    root = Path(root)
    out = root / (out_dir or COMPARE_DIR)
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    lr, lf = _checked_oof(root / OOF_DIR, 'exp113-oof-v1')
    rr, rf = _checked_oof(root / BASE_OOF_DIR, 'exp107-oof-v1')
    p = _protocol(root)
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    labels, _ = k99._load_labels(root / DREAD)
    metrics = {}
    for key in sorted(set(lf) & set(rf)):
        L, R = lf[key], rf[key]
        if list(L.columns) != [UID] + T or list(R.columns) != [UID] + T or L[UID].tolist() != R[UID].tolist():
            raise SystemExit(f'{key}: local and T4 OOF tables differ in studies or columns')
        index = {u: i for i, u in enumerate(L[UID])}
        Ls, Rs = L[T].to_numpy(np.float64), R[T].to_numpy(np.float64)
        scopes = {f'fold{k}': ids for k, ids in sorted(val_ids.items())}
        scopes['pooled'] = L[UID].tolist()
        m = {}
        for scope, ids in scopes.items():
            rows = np.array([index[u] for u in ids])
            Y = np.rint(np.stack([labels[u] for u in ids]).astype(np.float32)).astype(int)
            ml, pl = k99.macro12(Y, Ls[rows]); mr, pr = k99.macro12(Y, Rs[rows])
            rho = {t: spearman(Ls[rows, j], Rs[rows, j]) for j, t in enumerate(T)}
            m[scope] = dict(n=len(ids), local=ml, t4=mr, delta=ml - mr, mean_spearman=_mean_defined(rho.values()),
                            targets={t: dict(local=pl[t], t4=pr[t], delta=pl[t] - pr[t], spearman=rho[t]) for t in T})
        for side, recorded in (('local', lr), ('t4', rr)):
            if abs(m['pooled'][side] - recorded['metrics'][key]['pooled']['macro12']) > 1e-12:
                raise SystemExit(f'{key}: recomputed {side} pooled macro differs from its result.json')
        # exploratory: equal-weight per-target percentile-rank mean of the two trajectories (a two-run RadR50 ensemble), pooled only
        Y = np.rint(np.stack([labels[u] for u in L[UID]]).astype(np.float32)).astype(int)
        rank = lambda S: np.stack([k99.avg_rank(S[:, j]) / len(S) for j in range(12)], 1)  # noqa: E731
        m['pooled']['rank_mean_local_t4_exploratory'] = k99.macro12(Y, (rank(Ls) + rank(Rs)) / 2)[0]
        diff = np.abs(Ls - Rs)
        m['pooled']['abs_probability_difference'] = dict(mean=float(diff.mean()), max=float(diff.max()))
        metrics[key] = dict(scopes=m, delta_by_fold=[m[f'fold{k}']['delta'] for k in sorted(val_ids)])
    body = dict(schema='exp113-compare-v1', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), label_set=DEFAULT_LABEL_SET,
                local=dict(dir=OOF_DIR, protocol_sha256=lr['protocol_sha256'], outputs=lr['outputs']),
                t4=dict(dir=BASE_OOF_DIR, protocol_sha256=rr['protocol_sha256'], outputs=rr['outputs']),
                scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py; Spearman = Pearson of average ranks'),
                metrics=metrics,
                interpretation='descriptive: the two runs share protocol, folds, labels, seed and recipe but differ in device, OS window pipeline, '
                               'software stack and initial head state (T4 relaxed), so local - T4 is a device/stack/trajectory difference (D-B-008), '
                               'not a pass/fail and not a recipe effect; no interval is reported because two trajectories give no trajectory '
                               'distribution (D-B-017); local recipe contrasts pair with the local control only')
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (out / 'result.md').write_text(compare_markdown(body), encoding='utf-8')
    print(json.dumps({k: dict(pooled_local=round(v['scopes']['pooled']['local'], 6), pooled_t4=round(v['scopes']['pooled']['t4'], 6),
                              delta=round(v['scopes']['pooled']['delta'], 6), delta_by_fold=[round(x, 6) for x in v['delta_by_fold']],
                              mean_spearman=v['scopes']['pooled']['mean_spearman']) for k, v in metrics.items()}, indent=1))
    return body


def compare_markdown(body):
    lines = ['# EXP113 local RadR50 5-fold vs EXP107 Kaggle T4 (dread, descriptive)', '', body['interpretation'], '']
    for key, block in body['metrics'].items():
        m = block['scopes']
        lines += [f'## {key.upper()}', '', '| scope | n | local | T4 | local - T4 | mean Spearman |', '|---|---:|---:|---:|---:|---:|']
        for scope, v in m.items():
            lines.append(f"| {scope} | {v['n']} | {v['local']:.6f} | {v['t4']:.6f} | {v['delta']:+.6f} | {_fmt(v['mean_spearman'])} |")
        lines += ['', f"rank mean of the two trajectories (exploratory, pooled): {m['pooled']['rank_mean_local_t4_exploratory']:.6f}; "
                      f"|p_local - p_T4| mean {m['pooled']['abs_probability_difference']['mean']:.4f}, max {m['pooled']['abs_probability_difference']['max']:.4f}",
                  '', '| target (pooled) | local | T4 | local - T4 | Spearman |', '|---|---:|---:|---:|---:|']
        for t, v in m['pooled']['targets'].items():
            lines.append(f"| {t} | {v['local']:.6f} | {v['t4']:.6f} | {v['delta']:+.6f} | {_fmt(v['spearman'])} |")
        lines.append('')
    return '\n'.join(lines)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--dry-run', default=None, help='write the registration to this new directory instead (outside artifacts/)')
    s = sub.add_parser('register-labels'); s.add_argument('--name', required=True)
    s = sub.add_parser('argv'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--labels', default=DEFAULT_LABEL_SET)
    s.add_argument('--mode', choices=['full', 'smoke'], default='full'); s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    s = sub.add_parser('validate'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--labels', default=DEFAULT_LABEL_SET)
    s = sub.add_parser('smoke'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--out', required=True)
    s.add_argument('--labels', default=DEFAULT_LABEL_SET); s.add_argument('--train', type=int, default=16); s.add_argument('--val', type=int, default=2)
    s.add_argument('--epochs', type=int, default=1); s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('queue-json'); s.add_argument('--folds', default='0,1,2,3,4'); s.add_argument('--labels', default=DEFAULT_LABEL_SET)
    s.add_argument('--after', required=True, help='tag the first entry waits for (the current GPU chain tail); "none" only for an otherwise idle GPU')
    s.add_argument('--workers', type=int, default=LOCAL_WORKERS)
    s.add_argument('--out', default=None, help=f'new JSON file for the entries (default {QUEUE_FILE} or queue-entries-<name>.json; "-" = print only)')
    s = sub.add_parser('assemble'); s.add_argument('--labels', default=DEFAULT_LABEL_SET)
    s.add_argument('--runs-dir', default=None, help='directory holding the fold run directories (default artifacts/experiments)')
    s.add_argument('--k36', action='append', default=[], help='FOLD=PATH oof_k36.csv override')
    sub.add_parser('compare')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    if a.command == 'prepare':
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        if a.dry_run and within(a.dry_run, ROOT / 'artifacts'):
            raise SystemExit('--dry-run must be outside artifacts/')
        prepare(dest=a.dry_run)
    elif a.command == 'register-labels':
        register_labels(a.name)
    elif a.command == 'argv':
        print(json.dumps(runner_argv(view_for(a.labels), a.fold, a.mode, workers=a.workers, output_root=a.output_root, label_set=a.labels), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli(a.fold, a.labels))
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.fold, a.out, a.train, a.val, a.epochs, a.workers, a.labels))
    elif a.command == 'queue-json':
        entries = queue_entries([int(x) for x in a.folds.split(',')], None if a.after.lower() == 'none' else a.after, a.labels, a.workers)
        out = a.out or str(ROOT / queue_file_for(a.labels))
        if out != '-':
            dump(out, entries)
            print('written ' + out, file=sys.stderr)
        print(json.dumps(entries, indent=1))
    elif a.command == 'assemble':
        assemble(a.labels, a.runs_dir, {int(x.split('=', 1)[0]): x.split('=', 1)[1] for x in a.k36})
    else:
        compare()


if __name__ == '__main__':
    main()
