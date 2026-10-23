"""EXP112 (H-B-33): the CoAtNet B8 recipe trained once on ALL 4,343 G1 studies for 15 epochs; model_best.pt = fixed SWA of epochs 12/14/15.

  python tools/exp112_full_swa.py prepare                     # register EXP112_FULL_SWA_PROTOCOL_V1 + queue-entry.json (CPU; init must be B)
  python tools/exp112_full_swa.py prepare-scratch --dest DIR [--tiny]   # engineering registration outside artifacts/ (smoke runs only)
  python tools/exp112_full_swa.py argv [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp112_full_swa.py validate                    # runner --validate-only on CPU (no output, no CUDA)
  python tools/exp112_full_swa.py smoke --out DIR [--protocol MANIFEST] [--train 8 --epochs 3 --swa 1,2,3 --workers 0]   # CPU fit
  python tools/exp112_full_swa.py verify [--run DIR] [--quick]  # check a finished EXP112_B8_FULL_SWA run against the protocol
  python tools/exp112_full_swa.py estimate                    # local GPU time from the EXP094L/EXP099 reference fold runs

Recipe = EXP099 fold runs = EXP094 V4 arm seed42 (B8): coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k, 224 px, cache RAPTOR64_336_C140_S02, K12
within-slot-valid training windows with the per-study intensity jitter, bs 8, AdamW (backbone 3e-5 / head 1e-3, wd 0.02), OneCycleLR
(pct_start 0.15) over every update of the run, grad-norm clip 3.0, BCE with clipped pos_weight, FP16 autocast + GradScaler init 1024 (1% skip
gate), strict deterministic runtime, gradient checkpointing, 12 loader workers, seed 42, initial state = B (e4ea3ae6..., recomputed on CPU).
Changes, all fixed before training and none chosen on any metric:
  1. training set: all 4,343 G1 studies = the union of the five EXP099 fold training lists, in the canonical EXP099 order; Gold58 absent.
  2. pos_weight: the EXP099 (Dread fold-0) rule recomputed on the 4,343 studies; the vector is recorded in the manifest.
  3. epochs 8 -> 15: len_train_loader 4343 // 8 = 542, OneCycle total_steps 542 * 15 = 8130, FP16 skip gate 8130 // 100 = 81.
  4. no validation set, no evaluation, no checkpoint selection: the runner computes no prediction and no metric and cannot read Gold.
  5. a checkpoint after every epoch: <run>/epochs/epoch_NN.pt (NN = 1-based epoch) + epochs/output-inventory.json (file sha256, state hash).
  6. model_best.pt = SWA: arithmetic mean of the epoch-12, -14 and -15 weights. Epoch convention (1-based): epoch e is the model state after
     e complete passes over the training loader (after update e * 542); the runner's loop index is e - 1. BatchNorm running statistics are
     then recomputed with torch.optim.swa_utils.update_bn over the training loader (train mode, no gradient, FP16 autocast as in training,
     window draw index 15 = a draw no training epoch used, 542 batches). model_final.pt = the plain epoch-15 weights.
The release-format checkpoint keys are those of EXP099_B8_DREAD_F1/model_best.pt (tools/exp104_release_v5.py / exp111_release_v6.py).
Nothing here trains on or touches the GPU.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp101_kfold_arch as base  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools.exp101_kfold_arch import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                     CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                     FOLD_PROTOCOL, DREAD, LABEL_SET, KEYS_FILE, FROZEN_KEYS, SMOKE_ALLOWED, GPU_PYTHON,
                                     recipe_args, fold_contract, state_hashes, mismatched_keys, dump, train_python, within, display_path,
                                     build_initial_model, _set_flag)

SCHEMA = 'exp112-full-swa-v1'
PROTOCOL = 'artifacts/experiments/EXP112_FULL_SWA_PROTOCOL_V1'
RUNNER = 'training/runners/exp112_full_swa.py'
TOOL = 'tools/exp112_full_swa.py'
PLAN_ITEM = 'H-B-33'
FOLD_MANIFEST_SHA256 = '707205dbf69ffcaf1c8fdd3df84aa90a2cd63464408d4cfa3a1bdce6e8aee53d'  # EXP099_KFOLD_PROTOCOL_V1/manifest.json
V4_PROTOCOL = 'artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4/manifest.json'
V4_MANIFEST_SHA256 = 'ccac9b24d5cac9c9c87c9eee6cc47b6e3af13dcb50a0176eee5420f0f715205a'
V4_ARM = 'seed42'
B_INITIAL_STATE = 'e4ea3ae625eacde8e3ab563b6c663d715600a0cea039e1c9ee9d201367d5b38e'
SEED = 42
N_STUDIES = 4343
N_GOLD = 58
BASE_EPOCHS = 8
EPOCHS = 15
SWA_EPOCHS = (12, 14, 15)  # 1-based: epoch e = state after e full passes over the training loader (loop index e - 1)
BN_DRAW_EPOCH = EPOCHS  # 0-based window-draw index for the BatchNorm pass: 15 = the draw after the last training epoch (14)
TAG = 'EXP112_B8_FULL_SWA'
FULL_WORKERS = 12
TRAIN_IDS_FILE = 'train-ids.json'
EPOCH_DIR = 'epochs'
INVENTORY = 'output-inventory.json'
QUEUE_FILE = PROTOCOL + '/queue-entry.json'
QUEUE_AFTER = 'EXP099_B8_FLIGHT_V4_F4'
# the B8 recipe (EXP099 common_args without the workers resource argument), written out so a drifted source cannot change it silently
BASE_RECIPE = dict(arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', res=224, k=12, k_eval=24, epochs=BASE_EPOCHS, fixed_epoch=BASE_EPOCHS, bs=8, accum=1,
                   grad_ckpt=True, amp='fp16', amp_init_scale=1024.0, eval_amp='fp16', runtime_policy='strict', norm='imagenet', bb_lr=3e-5,
                   head_lr=1e-3, wd=0.02, pretrained='timm', window_policy='within-slot-valid', device='cuda')
EXPECTED_RECIPE = {**BASE_RECIPE, 'epochs': EPOCHS, 'fixed_epoch': EPOCHS}
RESOURCE_ARGS = dict(workers=dict(default=FULL_WORKERS, min=0, max=16, full=FULL_WORKERS,
                                  note='DataLoader workers; full runs use the B8 value 12; smoke may use fewer. Window sampling is '
                                       'sha256(seed, epoch, uid)-deterministic and batch order comes from a seeded generator'))
# training-loop constants of the EXP099 / EXP101 runner loop (bound through the runner's sha256; listed here for the recipe hash)
LOOP = dict(optimizer='AdamW([backbone.* lr bb_lr, other params lr head_lr], weight_decay wd)',
            scheduler='OneCycleLR(max_lr [bb_lr, head_lr], total_steps = all updates of the run, pct_start 0.15); stepped after every update '
                      'whose FP16 scale did not drop',
            loss='BCEWithLogitsLoss(pos_weight) on float32 logits', grad_clip=3.0, scaler='torch.amp.GradScaler(cuda, init_scale 1024)',
            fp16_skip_gate='skipped updates > steps // 100 fails', sampler='DataLoader(shuffle, drop_last, generator manual_seed(seed))',
            windows='StudyWindows(train=True, K12, within-slot-valid, intensity jitter 0.20, imagenet norm), sha256(seed,epoch,uid,stream)-v1',
            warmup_forward='model.eval()(zeros(1,1,3,res,res)) before training (as EXP094/EXP099)')
SWA = dict(epochs_1based=list(SWA_EPOCHS), loop_indices_0based=[e - 1 for e in SWA_EPOCHS],
           convention='1-based: epoch e is the model state after e complete passes over the training loader (after optimizer update e * '
                      'len_train_loader); the runner loop index is e - 1; files epochs/epoch_{e:02d}.pt',
           rule='arithmetic mean of the member state_dicts: each floating tensor summed in float64 over members, divided by the member count '
                'and cast back to its dtype (float32); integer tensors must be identical across members except BatchNorm num_batches_tracked '
                '(reset by the BatchNorm pass)',
           selection='fixed before training (H-B-33, mirrors public v5 swa_over [14, 12, 15], D-A-050); never chosen on Gold, OOF or any metric',
           members_source='the saved epoch checkpoints, re-read from disk and checked against their recorded state sha256')
BN_RECOMPUTE = dict(function='torch.optim.swa_utils.update_bn', mode='train', gradients='none (update_bn runs under torch.no_grad)',
                    amp='the training autocast (FP16 on CUDA; none on CPU smoke)', momentum='None during the pass = cumulative mean over batches',
                    loader='the training DataLoader itself (all 4,343 studies, bs 8, shuffle with the seeded generator, drop_last): '
                           'one more pass = 542 batches / 4,336 studies',
                    window_draw_epoch_0based=BN_DRAW_EPOCH, grad_checkpointing='off during the pass (forward numerics unchanged)',
                    checks='every BatchNorm num_batches_tracked == batches; running statistics finite; parameters equal the SWA mean')
# keys of artifacts/experiments/EXP099_B8_DREAD_F1/model_best.pt (read by tools/exp104_release_v5.py and exp111_release_v6.py stage-assets)
RELEASE_KEYS = ('model', 'arch', 'res', 'lab', 'contract', 'k_eval', 'norm', 'window_policy', 'epoch', 'tag', 'amp_train', 'amp_eval', 'fold',
                'label_set', 'provenance')
RELEASE_REQUIRED = ('arch', 'res', 'lab', 'contract', 'k_eval', 'norm', 'window_policy')  # REQUIRED_META of both release tools
RELEASE_META = ('arch', 'res', 'lab', 'contract', 'k_eval', 'norm', 'window_policy', 'epoch', 'tag', 'amp_train', 'amp_eval')
RELEASE_CODE = ('raptor_common.py', 'raptor_runtime.py')  # TRAINING_CODE_BOUND of exp111_release_v6
RUN_FILES = ('summary.json', 'provenance.json', 'history.json', 'swa.json', 'model_final.pt', 'model_best.pt', INVENTORY, EPOCH_DIR + '/' + INVENTORY)
REFERENCE_RUNS = ('artifacts/experiments/EXP094L_B8_SEED42', 'artifacts/experiments/EXP099_B8_DREAD_F1', 'artifacts/experiments/EXP099_B8_DREAD_F2',
                  'artifacts/experiments/EXP099_B8_DREAD_F3', 'artifacts/experiments/EXP099_B8_DREAD_F4')
OVERHEAD_SECONDS = 150  # setup + 17 checkpoint saves/hashes (~293 MB each) + SWA averaging
SMOKE_DEFAULTS = dict(device='cpu', amp='none', eval_amp='none', train=8, epochs=3, swa=(1, 2, 3))


# ------------------------------------------------------------------ names
def tag_for(mode='full'):
    if mode not in ('full', 'smoke'):
        raise ValueError('mode is full or smoke')
    return TAG + ('_SMOKE' if mode == 'smoke' else '')


def epoch_file(epoch_1based):
    e = int(epoch_1based)
    if e < 1:
        raise ValueError('epochs are 1-based')
    return f'epoch_{e:02d}.pt'


def parse_epochs(text):
    return tuple(int(x) for x in str(text).split(',') if x.strip())


def check_swa_epochs(swa_epochs, epochs):
    """1-based SWA member epochs: non-empty, distinct, ascending, each in 1..epochs."""
    e = tuple(int(x) for x in swa_epochs)
    if not e or len(set(e)) != len(e) or list(e) != sorted(e) or e[0] < 1 or e[-1] > int(epochs):
        raise ValueError(f'SWA epochs must be distinct, ascending, 1-based and <= {epochs}: {e}')
    return e


def full_recipe(common8):
    """EXP112 common_args from the registered B8 common_args: epochs and fixed_epoch 8 -> 15, nothing else."""
    args = recipe_args(common8)
    if args != BASE_RECIPE:
        raise ValueError('B8 common_args differ from the registered recipe: ' + str({k: (args[k], BASE_RECIPE.get(k)) for k in args if args[k] != BASE_RECIPE.get(k)}))
    out = recipe_args({**args, 'epochs': EPOCHS, 'fixed_epoch': EPOCHS})
    if {k for k in out if out[k] != args[k]} != {'epochs', 'fixed_epoch'}:
        raise AssertionError('more than the epoch count changed')
    return out


def full_contract(n_train, args):
    """EXP099 contract rule without an evaluation batch: loader length, updates (= OneCycle total_steps) and the 1% FP16 skip gate."""
    c = fold_contract(n_train, args)
    return {k: c[k] for k in ('len_train_loader', 'steps', 'scheduler_total_steps', 'max_fp16_skips')}


def canonical_json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def recipe_record(common, pos_weight_vector, n_train):
    record = dict(common_args=common, seed=SEED, loop=LOOP, swa=SWA, bn_recompute=BN_RECOMPUTE, pos_weight=[float(x) for x in pos_weight_vector],
                  n_train=n_train, validation='none', checkpoint_selection='none (fixed SWA of 1-based epochs 12, 14, 15)')
    return record, canonical_json_sha256(record)


# ------------------------------------------------------------------ training set (pure, unit-tested)
def full_training_set(canonical, fold_trains, fold_vals):
    """All G1 studies in canonical order; must equal the union of the fold training lists, each study in exactly one validation list."""
    can = list(canonical)
    if len(can) != len(set(can)):
        raise ValueError('canonical order has duplicates')
    union = set().union(*map(set, fold_trains.values())) if fold_trains else set()
    if union != set(can):
        raise ValueError(f'union of fold training lists ({len(union)}) differs from the G1 population ({len(can)})')
    vals = [u for ids in fold_vals.values() for u in ids]
    if len(vals) != len(set(vals)) or set(vals) != set(can):
        raise ValueError('fold validation lists do not partition the population')
    k = len(fold_trains)
    counts = {}
    for ids in fold_trains.values():
        for u in ids:
            counts[u] = counts.get(u, 0) + 1
    if any(counts[u] != k - 1 for u in can):
        raise ValueError('a study is not in exactly k-1 fold training lists')
    return can


def check_no_gold(train_ids, gold):
    overlap = sorted(set(train_ids) & set(gold))
    if overlap:
        raise ValueError(f'{len(overlap)} Gold uid(s) in the training set, e.g. {overlap[0]}')
    return True


def validate_training_set(corpus_ids, ids, p):
    """Runner-side check of the registered full training list (no Gold read: the manifest records the prepare-time Gold check)."""
    ids = list(ids)
    if len(ids) != len(set(ids)) or len(ids) != p['training_set']['n']:
        raise ValueError('training list size/uniqueness mismatch')
    if uid_hash(ids) != p['training_set']['uid_sha256']:
        raise ValueError('training list differs from the registration')
    if not set(ids) <= set(corpus_ids):
        raise ValueError('training studies missing from the corpus')
    if set(ids) & set(p['excluded_uids']):
        raise ValueError('excluded uid in the training list')
    return ids


# ------------------------------------------------------------------ SWA / BatchNorm / checkpoint logic (pure, unit-tested)
def state_dict_sha256(state):
    """Combined sha256 over sorted keys (key bytes + contiguous CPU tensor bytes): the runner's initial-state rule applied to a state_dict."""
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def swa_average(states):
    """Arithmetic mean of state_dicts (SWA). Floating tensors: float64 sum / n, cast back to the member dtype. Integer tensors must be
    identical across members, except BatchNorm num_batches_tracked (the last member's value; the BatchNorm pass resets it)."""
    import torch
    states = list(states)
    if not states:
        raise ValueError('no SWA members')
    keys = list(states[-1].keys())
    if any(set(s.keys()) != set(keys) for s in states):
        raise ValueError('SWA members have different state_dict keys')
    out = OrderedDict()
    for k in keys:
        vals = [s[k].detach().cpu() for s in states]
        ref = vals[-1]
        if any(v.shape != ref.shape or v.dtype != ref.dtype for v in vals):
            raise ValueError(f'{k}: shape/dtype differ across SWA members')
        if ref.is_floating_point():
            acc = torch.zeros(ref.shape, dtype=torch.float64)
            for v in vals:
                acc += v.to(torch.float64)
            out[k] = (acc / len(vals)).to(ref.dtype)
        else:
            if not k.endswith('num_batches_tracked') and any(not torch.equal(v, ref) for v in vals):
                raise ValueError(f'{k}: integer tensor differs across SWA members')
            out[k] = ref.clone()
    return out


def batchnorm_modules(model):
    from torch.nn.modules.batchnorm import _BatchNorm
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _BatchNorm)]


def recompute_bn(model, loader, device, autocast_ctx=None, update_bn=None):
    """Recompute BatchNorm running statistics with torch.optim.swa_utils.update_bn (reset, momentum None = cumulative mean over batches,
    train mode, no gradient) over `loader` batches (x, y, uids), under `autocast_ctx`. Returns the record of what was used."""
    import torch
    update = update_bn or torch.optim.swa_utils.update_bn
    bns = batchnorm_modules(model)
    if not bns:
        raise ValueError('model has no BatchNorm modules')
    batches = []

    def inputs():
        for x, _y, uids in loader:
            batches.append(list(uids))
            yield x.to(device, non_blocking=True)

    t0 = time.time()
    with (autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()):
        update(inputs(), model)
    tracked = sorted({int(m.num_batches_tracked.item()) for _, m in bns if m.num_batches_tracked is not None})
    if not batches or tracked != [len(batches)]:
        raise RuntimeError(f'BatchNorm pass incomplete: {len(batches)} batches, num_batches_tracked {tracked}')
    bad = [n for n, m in bns if not (torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all())]
    if bad:
        raise RuntimeError('non-finite BatchNorm running statistics: ' + ', '.join(bad[:5]))
    uids = [u for b in batches for u in b]
    return dict(function=f'{update.__module__}.{update.__qualname__}', batches=len(batches), studies=len(uids), unique_studies=len(set(uids)),
                studies_sha256=uid_hash(sorted(uids)), batch_sizes=sorted({len(b) for b in batches}), bn_modules=len(bns), num_batches_tracked=tracked,
                mode='train', gradients='none', seconds=round(time.time() - t0, 1))


def release_payload(state, meta, epoch0, kind, **extra):
    """A checkpoint in the release format (keys of EXP099_B8_DREAD_F1/model_best.pt, in that order) plus EXP112 fields."""
    ck = {'model': state}
    for k in RELEASE_KEYS[1:]:
        ck[k] = int(epoch0) if k == 'epoch' else meta[k]
    ck['checkpoint_kind'] = kind
    ck.update(extra)
    return ck


def release_check(ck, lab):
    """What tools/exp104_release_v5.py and exp111_release_v6.py checkpoint_meta / stage-assets require of a checkpoint."""
    if not isinstance(ck, dict) or not isinstance(ck.get('model'), dict):
        raise ValueError('not a raptor checkpoint')
    missing = [k for k in RELEASE_KEYS if k not in ck]
    if missing:
        raise ValueError('checkpoint lacks release keys: ' + ', '.join(missing))
    if list(ck['lab']) != list(lab):
        raise ValueError('target order differs')
    if type(ck['res']) is not int or type(ck['k_eval']) is not int or ck['res'] <= 0 or ck['k_eval'] <= 0:
        raise ValueError('res / k_eval must be positive ints')
    json.dumps({k: ck[k] for k in RELEASE_META}, allow_nan=False)
    code = (ck.get('provenance') or {}).get('code_sha256') or {}
    if any(not isinstance(code.get(f), str) for f in RELEASE_CODE):
        raise ValueError('provenance lacks the training code hashes the release binds')
    return True


# ------------------------------------------------------------------ models
def tiny_smoke_model(n=12):
    """A few-kilobyte stand-in with the RaptorClassifier interface (backbone.* parameters, set_grad_checkpointing, (B, K, 3, H, W) -> (B, 12))
    and two BatchNorm layers (torch BatchNorm2d and timm BatchNormAct2d, the class CoAtNet uses). For CPU engineering smoke only."""
    import torch
    import torch.nn as nn
    from timm.layers import BatchNormAct2d

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 8, 8, stride=8, bias=False)
            self.bn1 = nn.BatchNorm2d(8)
            self.conv = nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False)
            self.bn2 = BatchNormAct2d(16)
            self.num_features = 16
            self.grad_checkpointing = False

        def set_grad_checkpointing(self, enable=True):
            self.grad_checkpointing = bool(enable)

        def forward(self, x):
            return self.bn2(self.conv(torch.relu(self.bn1(self.stem(x))))).mean((2, 3))

    class TinyRaptor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = TinyBackbone()
            self.norm = nn.LayerNorm(16)
            self.cls = nn.Linear(16, n)

        def forward(self, x):
            b, k = x.shape[:2]
            return self.cls(self.norm(self.backbone(x.flatten(0, 1)).view(b, k, -1)).mean(1))

    return TinyRaptor()


def build_tiny_initial(_arch, seed):
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import seed_all
    worker_init(0)
    seed_all(seed)
    return tiny_smoke_model()


def build_run_model(p, arch, pretrained):
    """The runner's model: the registered kind ('arch' = raptor_common.build_model, or 'tiny-smoke' for scratch smoke registrations)."""
    if p['initial_state']['model'] == 'tiny-smoke':
        return tiny_smoke_model()
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import build_model
    return build_model(arch, pretrained=pretrained)


# ------------------------------------------------------------------ prepare
def prepare(root=ROOT, destination=None, model_factory=None, tiny=False):
    """Register the protocol (default) or, with destination outside artifacts/, a scratch registration for smoke runs only."""
    import numpy as np
    root = Path(root)
    registered = (root / PROTOCOL).resolve()
    dest = Path(destination).resolve() if destination else registered
    scratch = dest != registered
    if scratch and within(dest, root / 'artifacts'):
        raise SystemExit('scratch registrations must live outside artifacts/')
    if not scratch and (tiny or model_factory is not None):
        raise ValueError('the registered protocol is built with the real CoAtNet initial model')
    if dest.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + str(dest))
    fold_manifest_path = root / FOLD_PROTOCOL / 'manifest.json'
    if sha256(fold_manifest_path) != FOLD_MANIFEST_SHA256:
        raise ValueError('EXP099 manifest changed')
    fp = read(fold_manifest_path)
    if fp['schema'] != base.FOLD_SCHEMA or fp['seed'] != SEED or fp['variant'] != 'seed' or fp['initial_state_sha256'] != B_INITIAL_STATE:
        raise ValueError('unexpected EXP099 registration')
    if sha256(root / fp['paths']['folds']) != fp['folds_csv_sha256']:
        raise ValueError('EXP099 folds.csv changed')
    for path, expected in fp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('EXP099 binding changed: ' + path)
    for name, expected in fp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    reg, reg_file = k99.load_registration(root, fp, LABEL_SET, fold_manifest_path)
    if reg['path'] != DREAD or fp['label_sets'][LABEL_SET]['path'] != DREAD or reg['pos_weight_source'] != 'labels':
        raise ValueError('EXP099 dread registration points elsewhere')
    if sha256(root / V4_PROTOCOL) != V4_MANIFEST_SHA256:
        raise ValueError('EXP094 V4 manifest changed')
    v4 = read(root / V4_PROTOCOL)
    arm = v4['arms'][V4_ARM]
    if v4['common_args'] != fp['common_args'] or v4['initial_state_sha256'] != B_INITIAL_STATE or arm['seed'] != SEED or arm['variant'] != 'seed' \
            or arm['slot_dropout'] != 0.0:
        raise ValueError('EXP094 V4 arm seed42 differs from EXP099')
    if fp['common_args']['workers'] != FULL_WORKERS:
        raise ValueError('B8 workers changed')
    common8 = recipe_args(fp['common_args'])
    common = full_recipe(fp['common_args'])
    if common != EXPECTED_RECIPE:
        raise ValueError('EXP112 recipe differs from the registered 15-epoch recipe')
    labels, semantic = k99._load_labels(root / DREAD)
    if semantic != reg['semantic_sha256'] or sha256(root / DREAD) != reg['sha256']:
        raise ValueError('Dread file differs from the EXP099 registration')
    canonical = canonical_order(read(root / fp['paths']['base_ordered_ids']))
    corpus_dir = root / fp['paths']['corpus']
    if sha256(corpus_dir / 'manifest.json') != fp['corpus_manifest_sha256']:
        raise ValueError('corpus manifest differs from EXP099')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=True).astype(str).tolist()
    trains, vals, fold_files = {}, {}, {}
    for k in sorted(fp['folds'], key=int):
        f = fp['folds'][k]
        for key in ('ordered_ids', 'groups'):
            if sha256(root / f[key]['path']) != f[key]['sha256']:
                raise ValueError(f'EXP099 fold {k} {key} file changed')
        ordered = read(root / f['ordered_ids']['path'])
        if uid_hash(ordered['train']) != f['ordered_ids']['train_sha256'] or uid_hash(ordered['val']) != f['ordered_ids']['val_sha256']:
            raise ValueError(f'EXP099 fold {k} ordered ids changed')
        trains[k], vals[k] = validate_fold_membership(corpus_ids, ordered, read_rows(root / f['groups']['path']), canonical)
        if len(trains[k]) != f['n_train'] or len(vals[k]) != f['n_val']:
            raise ValueError(f'EXP099 fold {k} size changed')
        fold_files[k] = dict(path=f['ordered_ids']['path'], sha256=f['ordered_ids']['sha256'], train_sha256=f['ordered_ids']['train_sha256'],
                             n_train=f['n_train'])
    train = full_training_set(canonical, trains, vals)
    if len(train) != N_STUDIES:
        raise ValueError('G1 population drift')
    gold = k99.gold_uids(root)
    if len(gold) != N_GOLD:
        raise ValueError(f'expected {N_GOLD} Gold uids, found {len(gold)}')
    check_no_gold(train, gold)
    if set(train) & set(fp['excluded_uids']):
        raise ValueError('an EXP099 excluded uid is in the training set')
    missing = [u for u in train if u not in labels]
    if missing:
        raise ValueError(f'Dread misses {len(missing)} training uid(s)')
    arr = np.stack([labels[u] for u in train])
    if arr.shape != (N_STUDIES, 12) or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
        raise ValueError('invalid Dread values')
    pw = pos_weight(labels, train)
    pw_f0 = np.asarray(fp['folds']['0']['pos_weight'], np.float32)
    c = full_contract(len(train), common)
    c8 = full_contract(fp['folds']['0']['n_train'], common8)
    if c['len_train_loader'] != N_STUDIES // common['bs'] or c['steps'] != c['len_train_loader'] * EPOCHS or c['max_fp16_skips'] != c['steps'] // 100:
        raise AssertionError('contract rule drifted')
    factory = model_factory or (build_tiny_initial if tiny else build_initial_model)
    model = factory(common['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init, keys = state_hashes(model)
    bn_modules = len(batchnorm_modules(model))
    model_kind = 'tiny-smoke' if (tiny or model_factory is not None) else 'arch'
    if not scratch and init != B_INITIAL_STATE:
        raise ValueError(f'CPU seed-42 initial state {init} is not B ({B_INITIAL_STATE})')
    if model_kind == 'arch' and bn_modules == 0:
        raise ValueError('the CoAtNet has no BatchNorm modules; the BatchNorm pass would be empty')
    import timm
    dest.mkdir(parents=True)
    dump(dest / TRAIN_IDS_FILE, dict(schema='exp112-train-ids-v1', population='all 4,343 G1 studies = union of the EXP099 fold training lists',
                                     order='canonical EXP099 order: EXP084 ordered-ids train list then val list', n=len(train), train=train))
    dump(dest / KEYS_FILE, dict(schema='exp101-initial-state-keys-v1', arch=common['arch'], model=model_kind, seed=SEED, combined_sha256=init, keys=keys,
                                rule='per state_dict key: sha256 of the contiguous CPU tensor bytes; combined = sha256 over sorted (key bytes + tensor bytes)'))
    recipe, recipe_sha = recipe_record(common, pw, len(train))
    bindings = [RUNNER, TOOL, base.TOOL, 'tools/exp099_kfold.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py',
                FOLD_PROTOCOL + '/manifest.json', fp['paths']['folds'], fp['paths']['base_ordered_ids'], fp['paths']['base_groups'],
                *(v['path'] for v in fold_files.values()), display_path(reg_file, root), DREAD, V4_PROTOCOL, fp['paths']['corpus'] + '/manifest.json']
    manifest = dict(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item=PLAN_ITEM,
        registration='scratch' if scratch else 'registered',
        hypothesis='The public-v5 recipe bundle (all labelled G1 studies, 15 epochs, fixed SWA of epochs 12/14/15) applied to our CoAtNet-2 224 B8 '
                   'recipe raises one model to >= 0.923 public (decision rule in plan.md H-B-33; release v5 arm, K62 if H-B-30 passes else K36)',
        source_bindings={s: sha256(root / s) for s in bindings},
        protocol_files={TRAIN_IDS_FILE: sha256(dest / TRAIN_IDS_FILE), KEYS_FILE: sha256(dest / KEYS_FILE)},
        training_code_sha256=fp['training_code_sha256'], environment=fp['environment'], corpus_manifest_sha256=fp['corpus_manifest_sha256'],
        excluded_uids=fp['excluded_uids'],
        recipe_source=dict(fold_protocol=FOLD_PROTOCOL + '/manifest.json', fold_protocol_sha256=FOLD_MANIFEST_SHA256, source_common_args=fp['common_args'],
                           v4_protocol=V4_PROTOCOL, v4_sha256=V4_MANIFEST_SHA256, v4_arm=V4_ARM, base_recipe=common8,
                           runner_loop='training/runners/exp099_kfold.py / exp101_kfold_arch.py loop (= EXP094 V4), without the evaluation'),
        changes=['training set: all 4,343 G1 studies (union of the five EXP099 fold training lists; Gold58 absent)',
                 'pos_weight: EXP099 Dread rule recomputed on the 4,343 studies',
                 'epochs 8 -> 15 (fixed_epoch 15): OneCycle total_steps 8130, FP16 skip gate 81',
                 'no validation set, no evaluation, no checkpoint selection (no OOF, no Gold, no metric)',
                 'checkpoint after every epoch (epochs/epoch_NN.pt, 1-based NN)',
                 'model_best.pt = SWA mean of epochs 12, 14, 15 (1-based) + BatchNorm recompute over the training loader; model_final.pt = epoch 15'],
        common_args=common, seed=SEED, epochs=EPOCHS, resource_args=RESOURCE_ARGS, smoke_allowed=SMOKE_ALLOWED,
        recipe=recipe, recipe_sha256=recipe_sha, swa=SWA, bn_recompute=dict(BN_RECOMPUTE, expected_batches=c['len_train_loader'],
                                                                         expected_studies=c['len_train_loader'] * common['bs'], bn_modules=bn_modules),
        initial_state_sha256=init,
        initial_state=dict(keys_file=KEYS_FILE, keys=len(keys), model=model_kind, computed_on='cpu', seed=SEED, arch=common['arch'],
                           torch=torch.__version__, timm=timm.__version__, equals_B=init == B_INITIAL_STATE, B=B_INITIAL_STATE, bn_modules=bn_modules,
                           rule='seed_all(42) after worker_init(0), build_model(arch, pretrained timm); sorted state_dict sha256',
                           enforcement='runner: the seed-42 initial state must equal this hash (no remote relaxation)'),
        training_set=dict(file=TRAIN_IDS_FILE, n=len(train), uid_sha256=uid_hash(train), sorted_uid_sha256=uid_hash(sorted(train)),
                          rule='canonical EXP099 order (EXP084 ordered-ids train then val) = union of the fold 0-4 training lists',
                          folds=fold_files, each_study_in_fold_train_lists=len(trains) - 1,
                          gold=dict(source=k99.GOLD_SOURCE + ' (rows with non-null ACL; uids only)', n=len(gold), overlap=0,
                                    sorted_uid_sha256=uid_hash(sorted(gold))),
                          excluded_uids_overlap=0, validation='none'),
        paths=dict(corpus=fp['paths']['corpus'], labels=DREAD, base_ordered_ids=fp['paths']['base_ordered_ids'], base_groups=fp['paths']['base_groups'],
                   folds=fp['paths']['folds']),
        label_set=dict(name=LABEL_SET, path=DREAD, sha256=reg['sha256'], semantic_sha256=reg['semantic_sha256'],
                       registration=display_path(reg_file, root), registration_sha256=sha256(reg_file)),
        pos_weight=pw.tolist(), pos_weight_rule=fp['pos_weight_rule'].replace('in fold canonical order', 'of all 4,343 studies in canonical order'),
        pos_weight_fold0=pw_f0.tolist(), pos_weight_minus_fold0=[round(float(a - b), 6) for a, b in zip(pw, pw_f0)],
        contract=c, contract_8_epochs_fold0=c8, contract_rule='len_train_loader = n_train // bs (drop_last); steps = len_train_loader * epochs '
                                                              '(accum 1) = OneCycle total_steps; FP16 skip gate: skipped > steps // 100 fails',
        tags=dict(full=TAG, smoke=tag_for('smoke')),
        outputs=dict(run_dir='artifacts/experiments/' + TAG, files=list(RUN_FILES), epoch_checkpoints=EPOCH_DIR + '/epoch_NN.pt (NN = 01..15)',
                     model_best='SWA(12,14,15) + BatchNorm recompute, release format (keys of EXP099_B8_DREAD_F1/model_best.pt)',
                     model_final='plain epoch-15 weights, release format', predictions='none'),
        release=dict(keys=list(RELEASE_KEYS), required=list(RELEASE_REQUIRED), code_bound=list(RELEASE_CODE),
                     note='tools/exp104_release_v5.py / exp111_release_v6.py stage-assets read model_best.pt; arm K chosen at release (K62 if H-B-30 '
                          'passes, else K36); checkpoint k_eval stays the trained K24 like every B8 checkpoint'),
        budget=estimate(root),
        caveats=['weak Dread labels; no validation set, so the run has no local score by design (judged on public only, plan H-B-33)',
                 'one seed; D-B-017: one-seed differences of ~0.006 occur on fold 0',
                 'the public v5 recipe is not reproduced exactly (labels, uid list, precision, normalization unknown: D-A-050)'])
    if not scratch:
        entry = queue_entry(manifest, root)
        manifest['queue'] = dict(file=QUEUE_FILE, after=QUEUE_AFTER, continue_on_failure=True, python=str(GPU_PYTHON),
                                 environment='queue runner started with OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 (as LOCAL_GPU_CHAIN_V13)',
                                 started=False, entry=entry)
    dump(dest / 'manifest.json', manifest)
    if not scratch:
        dump(root / QUEUE_FILE, [manifest['queue']['entry']])
    report = dict(status='REGISTERED_NO_EXECUTION', registration=manifest['registration'], protocol=display_path(dest, root),
                  manifest_sha256=sha256(dest / 'manifest.json'), initial_state_sha256=init, equals_B=init == B_INITIAL_STATE, bn_modules=bn_modules,
                  n_train=len(train), uid_sha256=manifest['training_set']['uid_sha256'], recipe_sha256=recipe_sha, contract=c,
                  pos_weight=[round(x, 6) for x in pw.tolist()], swa_epochs_1based=list(SWA_EPOCHS),
                  queue=(QUEUE_FILE if not scratch else None), estimate_hours=manifest['budget']['hours'])
    print(json.dumps(report, indent=1))
    return manifest


# ------------------------------------------------------------------ GPU time estimate
def reference_runs(root=ROOT):
    out = {}
    for run in REFERENCE_RUNS:
        s = read(Path(root) / run / 'summary.json')
        ep = sum(h['epoch_seconds'] for h in s['history'])
        out[Path(run).name] = dict(train=s['train'], updates=s['optimizer_updates'], epoch_seconds=ep, total_seconds=s['total_seconds'],
                                   seconds_per_update=round(ep / s['optimizer_updates'], 4))
    return out


def estimate_from(per_update, steps, len_tl, overhead=OVERHEAD_SECONDS):
    """Training updates at the reference per-update pace + one BatchNorm pass over len_tl batches (forward only: between 1/4 of a training
    update, compute-bound, and a full update, loader-bound) + checkpoint/SWA overhead."""
    med, lo, hi = statistics.median(per_update), min(per_update), max(per_update)
    central = med * steps + med * len_tl + overhead
    low = lo * steps + 0.25 * lo * len_tl + overhead
    high = hi * steps + hi * len_tl + overhead
    return dict(seconds=round(central), low_seconds=round(low), high_seconds=round(high), hours=round(central / 3600, 2),
                low_hours=round(low / 3600, 2), high_hours=round(high / 3600, 2))


def estimate(root=ROOT):
    refs = reference_runs(root)
    c = full_contract(N_STUDIES, EXPECTED_RECIPE)
    est = estimate_from([r['seconds_per_update'] for r in refs.values()], c['steps'], c['len_train_loader'])
    return dict(reference_runs=refs, steps=c['steps'], len_train_loader=c['len_train_loader'], overhead_seconds=OVERHEAD_SECONDS, **est,
                rule='median/min/max seconds per update of the reference runs (training epochs only) * 8130 updates + BatchNorm pass '
                     '(542 batches at 0.25-1.0 x the update time) + overhead; the 2026-10-07 flight-V4 runs ran at 1.00-1.06 s/update under '
                     'CPU contention, i.e. near the high end')


# ------------------------------------------------------------------ runner guard
def validate_arguments(args, root):
    """Called by the runner before any training import. Returns (protocol, registered training ids, output root, plan)."""
    root = Path(root)
    manifest_path = Path(args.protocol).resolve()
    p = read(manifest_path)
    smoke = args.mode == 'smoke'
    if p.get('schema') != SCHEMA:
        raise ValueError('wrong protocol')
    if manifest_path == (root / PROTOCOL / 'manifest.json').resolve():
        if p.get('registration') != 'registered':
            raise ValueError('wrong protocol registration')
    elif not smoke or p.get('registration') != 'scratch' or within(manifest_path, root / 'artifacts'):
        raise ValueError('wrong protocol: full runs use ' + PROTOCOL + '; scratch registrations (outside artifacts/) are for smoke only')
    if p.get('seed') != SEED or p['tags']['full'] != TAG or p['common_args'] != EXPECTED_RECIPE or p.get('epochs') != EPOCHS \
            or tuple(p['swa']['epochs_1based']) != SWA_EPOCHS:
        raise ValueError('not the EXP112 registration')
    if p['registration'] == 'registered' and (p['initial_state_sha256'] != B_INITIAL_STATE or p['initial_state']['model'] != 'arch'):
        raise ValueError('registered protocol must start from B')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['protocol_files'].items():
        if sha256(manifest_path.parent / name) != expected:
            raise ValueError('protocol file changed: ' + name)
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
    for k, v in p['common_args'].items():
        got = getattr(args, k, None)
        if got != v and not (smoke and k in p['smoke_allowed'] and got in p['smoke_allowed'][k]):
            raise ValueError('frozen argument changed: ' + k)
    if args.seed != p['seed']:
        raise ValueError('seed changed')
    w = p['resource_args']['workers']
    if not w['min'] <= args.workers <= w['max']:
        raise ValueError('workers outside the registered range')
    if not smoke and args.workers != w['full']:
        raise ValueError(f"full runs use the B8 workers {w['full']}")
    if parse_epochs(args.swa_epochs) != tuple(p['swa']['epochs_1based']):
        raise ValueError('SWA epochs differ from the registration')
    if tag_for(args.mode) != args.tag:
        raise ValueError('tag/mode mismatch')
    for key in ('labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('EXP112 is a local run; KNEE_REMOTE_RELAX_ENV is not allowed')
    ids = read(manifest_path.parent / p['training_set']['file'])['train']
    if uid_hash(ids) != p['training_set']['uid_sha256'] or len(ids) != p['training_set']['n']:
        raise ValueError('registered training list changed')
    default_root = (root / 'artifacts' / 'experiments').resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else default_root
    if smoke:
        if not args.output_root or within(out_root, root / 'artifacts'):
            raise ValueError('smoke runs need --output-root outside artifacts/')
        if args.smoke_train < args.bs or args.smoke_epochs < 1:
            raise ValueError('smoke needs >= one batch of training studies and >= 1 epoch')
        plan = dict(epochs=args.smoke_epochs, swa_epochs=check_swa_epochs(parse_epochs(args.smoke_swa_epochs), args.smoke_epochs), smoke=True)
    else:
        if out_root != default_root:
            raise ValueError('full runs write to artifacts/experiments (no --output-root)')
        plan = dict(epochs=p['epochs'], swa_epochs=check_swa_epochs(p['swa']['epochs_1based'], p['epochs']), smoke=False)
    return p, ids, out_root, plan


def runner_argv(p, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None, protocol_path=None):
    root = Path(root)
    manifest_path = Path(protocol_path) if protocol_path else root / PROTOCOL / 'manifest.json'
    argv = [python or train_python(), str(root / RUNNER), '--protocol', str(manifest_path), '--mode', mode, '--tag', tag_for(mode),
            '--seed', str(p['seed']), '--workers', str(p['resource_args']['workers']['default'] if workers is None else int(workers)),
            '--swa-epochs', ','.join(str(e) for e in p['swa']['epochs_1based'])]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(relpath(root, p['paths']['corpus'])), '--labels', str(relpath(root, p['paths']['labels']))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if mode == 'smoke':
        s = {**SMOKE_DEFAULTS, **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), s[key])
        argv += ['--smoke-train', str(s['train']), '--smoke-epochs', str(s['epochs']), '--smoke-swa-epochs', ','.join(str(e) for e in s['swa'])]
    return argv


def queue_entry(p, root=ROOT):
    """The tools/gpu_queue.py job (LOCAL_GPU_CHAIN_V13 entry format) for the registered full run, after the flight-V4 fold 4."""
    return dict(tag=TAG, command=runner_argv(p, 'full', root=root, python=str(GPU_PYTHON)), after=QUEUE_AFTER, continue_on_failure=True)


def _protocol(root=ROOT):
    return read(Path(root) / PROTOCOL / 'manifest.json')


def local_env():
    return base.local_env()


def validate_cli():
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(_protocol(), 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(out, protocol, train, epochs, swa, workers):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    manifest_path = Path(protocol).resolve() if protocol else ROOT / PROTOCOL / 'manifest.json'
    argv = runner_argv(read(manifest_path), 'smoke', workers=workers, output_root=out, protocol_path=manifest_path,
                       smoke=dict(train=train, epochs=epochs, swa=parse_epochs(swa)))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for('smoke')
    report = dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(run), files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json')
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], swa=s['swa'], history=s['history'],
                      initial_state_matches_registered=s['provenance']['initial_state_matches_registered'], cuda_initialized=s['provenance']['cuda_initialized'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification of the finished run
def verify_run(run, p, root=ROOT, hash_epochs=True):
    """Problems (empty list = complete and consistent) for a finished full EXP112 run directory."""
    root, run = Path(root), Path(run)
    missing = [n for n in RUN_FILES if not (run / n).is_file()]
    if missing:
        return ['missing ' + ', '.join(missing)]
    problems = []
    for name, rec in read(run / INVENTORY).items():
        if not (run / name).is_file() or sha256(run / name) != rec['sha256'] or (run / name).stat().st_size != rec['bytes']:
            problems.append(f'{name} differs from {INVENTORY}')
    epochs = read(run / EPOCH_DIR / INVENTORY)
    if list(epochs) != [epoch_file(e) for e in range(1, p['epochs'] + 1)]:
        problems.append('epoch checkpoints are not epoch_01..epoch_%02d' % p['epochs'])
    elif hash_epochs:
        for name, rec in epochs.items():
            if sha256(run / EPOCH_DIR / name) != rec['sha256']:
                problems.append(f'{EPOCH_DIR}/{name} differs from its inventory')
    if sorted(x.name for x in run.glob('*.csv')):
        problems.append('prediction files present: the full run must not evaluate')
    s, prov, swa = read(run / 'summary.json'), read(run / 'provenance.json'), read(run / 'swa.json')
    expect = dict(seed=SEED, arch=p['common_args']['arch'], initial_state_sha256=B_INITIAL_STATE, initial_state_matches_registered=True,
                  environment_matches_baseline=True, requested_workers=FULL_WORKERS, scheduler_total_steps=p['contract']['steps'],
                  max_fp16_skips=p['contract']['max_fp16_skips'], label_set=LABEL_SET, training_mode='full', validation='none',
                  checkpoint_selection=p['recipe']['checkpoint_selection'], gold_access='disabled', metric_access='none',
                  train_ids_ordered_sha256=p['training_set']['uid_sha256'], label_semantic_sha256=p['label_set']['semantic_sha256'],
                  protocol_sha256=sha256(root / PROTOCOL / 'manifest.json'), recipe_sha256=p['recipe_sha256'], cuda_initialized=True)
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'provenance {key} = {prov.get(key)!r}, expected {value!r}')
    if s.get('optimizer_updates') != p['contract']['steps'] or s.get('skipped_updates', 10 ** 9) > p['contract']['max_fp16_skips']:
        problems.append(f"updates {s.get('optimizer_updates')} (expected {p['contract']['steps']}), skipped {s.get('skipped_updates')}")
    if len(s.get('history') or []) != p['epochs'] or s.get('train') != p['training_set']['n'] or s.get('val') != 0:
        problems.append('history/epoch/count mismatch')
    if swa.get('epochs_1based') != list(SWA_EPOCHS) or [m['epoch_1based'] for m in swa.get('members', [])] != list(SWA_EPOCHS):
        problems.append('SWA members are not epochs 12, 14, 15')
    bn = swa.get('bn_recompute') or {}
    if bn.get('batches') != p['contract']['len_train_loader'] or bn.get('studies') != p['contract']['len_train_loader'] * p['common_args']['bs'] \
            or bn.get('num_batches_tracked') != [p['contract']['len_train_loader']] or bn.get('bn_modules') != p['initial_state']['bn_modules']:
        problems.append(f'BatchNorm pass record differs: {bn}')
    return problems


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('prepare-scratch'); s.add_argument('--dest', required=True); s.add_argument('--tiny', action='store_true')
    s = sub.add_parser('argv'); s.add_argument('--mode', choices=['full', 'smoke'], default='full')
    s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    sub.add_parser('validate')
    s = sub.add_parser('smoke'); s.add_argument('--out', required=True); s.add_argument('--protocol', default=None)
    s.add_argument('--train', type=int, default=SMOKE_DEFAULTS['train']); s.add_argument('--epochs', type=int, default=SMOKE_DEFAULTS['epochs'])
    s.add_argument('--swa', default=','.join(map(str, SMOKE_DEFAULTS['swa']))); s.add_argument('--workers', type=int, default=0)
    s = sub.add_parser('verify'); s.add_argument('--run', default=None, help=f'run directory (default artifacts/experiments/{TAG})')
    s.add_argument('--quick', action='store_true', help='skip re-hashing the 15 epoch checkpoints')
    sub.add_parser('estimate')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    if a.command in ('prepare', 'prepare-scratch'):
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        prepare() if a.command == 'prepare' else prepare(destination=a.dest, tiny=a.tiny)
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli())
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.out, a.protocol, a.train, a.epochs, a.swa, a.workers))
    elif a.command == 'estimate':
        print(json.dumps(estimate(), indent=1))
    else:
        problems = verify_run(Path(a.run) if a.run else ROOT / 'artifacts' / 'experiments' / TAG, _protocol(), hash_epochs=not a.quick)
        print(json.dumps(dict(run=a.run or TAG, status='OK' if not problems else 'PROBLEMS', problems=problems), indent=1))
        sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
