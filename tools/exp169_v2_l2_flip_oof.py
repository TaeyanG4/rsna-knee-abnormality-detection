"""EXP169 v2 pre-reading check B: L2 RadR50 K36 flip-TTA OOF production. v1.0.1

Claude opus for B, 2026-10-10. Job/spec EXP169_V2_L2_K36_FLIP_OOF_v1, registered in
artifacts/experiments/EXP169_RELEASE_V2/PRE_READING_CHECKS.md section B.

Ten frozen fixed-final L2 RadR50 checkpoints (EXP149_L2_S42_F0/F1,
EXP157_L2_S42_F2..F4, EXP157_L2_S43_F0..F4), each evaluated ONLY on its own EXP099
held-out fold (EXP149 ``fold_rosters``), exact RAPTOR64 336/c140/s02 corpus,
within-slot-valid centre3 K36, RadImageNet normalisation, fp16 autocast, cuDNN
benchmark off, deterministic algorithms on, one CPU math thread.

Reused by import (not copied): EXP124 ``plane_ranges``/``flip_mask``/
``mirror_windows``/``PlaneAwareWindows``/``collate_pair``/``infer_pair``/
``served_probabilities`` and the EXP169 serving ``plane_flags`` (both plane rules
must agree for every study). EXP124's CLI is not used. Coronal/axial windows are
mirrored horizontally; sagittal and padding windows are unchanged. The TTA value
is ``(sigmoid(plain.float()) + sigmoid(mirrored.float())) / 2`` before any ranks.

No label file, report text or label row is read; held-out labels are zeros.

Subcommands
  preflight      CPU only: authenticate all ten sources/rosters/corpus, strict-load each state.
  register-lane  write the lane/budget registration once (LOCAL RTX 4070 Ti SUPER, cap 120
                 GPU-min, pilot-first); pilot and run refuse without it.
  pilot          <= 40 held-out studies of one model on CUDA: time, VRAM, projected full-job
                 minutes with overhead; exits 3 on any STOP (projection > 120 min or plain
                 subset gate failed), non-zero on any error.
  run            requires the lane file and a PASS pilot of this version; all ten models
                 serially on CUDA; resumable (a tag whose receipt verifies is skipped); a
                 plain-vs-frozen max abs diff > 0.002 stops the model and the job (exit 2).
  verify         check the ten receipts, outputs, rosters and per-seed coverage.
  queue-json     write a tools/gpu_queue.py plan: pilot, then run only after pilot SUCCESS
                 (launches nothing).

Changelog
  1.0.0 (2026-10-10) initial adapter.
  1.0.1 (2026-10-10) verifier fixes: the pilot exits 3 on STOP, so gpu_job writes FAILED and the
        queue stops before run; lane registration file required by pilot/run; run requires a
        PASS pilot; receipt status is the single constant RECEIPT_STATUS ('ADMITTED'), bound by
        a test to the exp169_v2_dino_marginal v1.1.0 reader; produce() is now the CPU folder
        writer (GPU part: produce_model), plus verify_folder/resume_loop, so tests exercise them.
"""
from __future__ import annotations

import os
import sys

THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
if __name__ == '__main__':  # one CPU math thread and strict cuBLAS before numpy/torch load
    os.environ.update(THREAD_ENV)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp124_flip_tta as e124  # noqa: E402
from tools import exp149_label_sources as guard  # noqa: E402
from tools import exp169_partner_bundle_release as serving  # noqa: E402

VERSION = '1.0.1'
AUTHOR = 'Claude opus for B, 2026-10-10'
JOB_ID = 'EXP169_V2_L2_K36_FLIP_OOF_v1'
SCHEMA = 'exp169-v2-l2-flip-oof-receipt-v1'
# Receipt status read by tools/exp169_v2_dino_marginal.py v1.1.0 check_flip_receipt
# (FLIP_RECEIPT_CONTRACT); tests/test_exp169_v2_l2_flip_oof.py binds the two.
RECEIPT_STATUS = 'ADMITTED'
PILOT_SCHEMA = 'exp169-v2-l2-flip-oof-pilot-v1'
LANE_SCHEMA = 'exp169-v2-l2-flip-oof-lane-v1'
LANE_REL = 'artifacts/experiments/EXP169_RELEASE_V2/pre-reading/l2-flip-oof-lane.json'
LANE_GPU = 'NVIDIA GeForce RTX 4070 Ti SUPER'
TOOL_REL = 'tools/exp169_v2_l2_flip_oof.py'
OUT_REL = 'artifacts/experiments/EXP169_RELEASE_V2/pre-reading/l2-flip-oof'
CHECKS_REL = 'artifacts/experiments/EXP169_RELEASE_V2/PRE_READING_CHECKS.md'
CORPUS_REL = 'storage/cache/derived/RAPTOR64_336_C140_S02'
CONTRACT = 'raptor64_336_c140_s02'
EXP149_PROTOCOL = 'artifacts/experiments/EXP149_LABEL_SOURCES/students/manifest.json'
EXP157_PROTOCOL = 'artifacts/experiments/EXP157_L2_ADVANCE/manifest.json'
# v2 release model manifests (v2.0.1 stages); both list the same ten L2 source checkpoints.
RELEASE_MANIFESTS = {
    'artifacts/releases/EXP169_A172_v2_0_1_stage/assets/release-manifest.json':
        'a4953fe359c8f6d776c09ac671645fac79f90de9d3b3d0506f2f0c2a2fb7365f',
    'artifacts/releases/EXP169_B172_v2_0_1_stage/assets/release-manifest.json':
        '7eeea0ceaf72799244721e404c70d540c467168a40ebb4174ffe2b0d4dabf33f',
}
K = 36
PLAIN_TOL = 0.002          # EXP124 cross-environment reproduction ceiling
SERVING_TOL = 2e-5         # release local/T4 serving parity tolerance; not replaced by PLAIN_TOL
PILOT_MAX = 40
CAP_MINUTES = 120
MARGIN = 1.25              # projection overhead multiplier on top of measured fixed costs
STUDIES_PER_SEED = 4343
UID = e124.UID
TARGETS = list(e124.TARGETS)
PLAIN, TTA = 'oof_plain_k36.csv', 'oof_k36_flip_tta.csv'
INVENTORY, RECEIPT = 'output-inventory.json', 'receipt.json'
CODE_FILES = (TOOL_REL, 'tools/exp124_flip_tta.py', 'tools/exp169_partner_bundle_release.py',
              'tools/exp149_label_sources.py', 'training/src/raptor_common.py',
              'training/src/raptor_radimagenet.py', 'training/src/raptor_runtime.py',
              'training/src/raptor_data.py')

need, sha, read, uid_hash = serving.need, guard.sha256, guard.read, guard.uid_hash


class Source:
    __slots__ = ('tag', 'seed', 'fold', 'protocol')

    def __init__(self, tag, seed, fold, protocol):
        self.tag, self.seed, self.fold, self.protocol = tag, seed, fold, protocol

    def __repr__(self):
        return f'Source({self.tag})'


SOURCES = tuple([Source('EXP149_L2_S42_F0', 42, 0, EXP149_PROTOCOL),
                 Source('EXP149_L2_S42_F1', 42, 1, EXP149_PROTOCOL)] +
                [Source(f'EXP157_L2_S42_F{f}', 42, f, EXP157_PROTOCOL) for f in (2, 3, 4)] +
                [Source(f'EXP157_L2_S43_F{f}', 43, f, EXP157_PROTOCOL) for f in range(5)])
TAGS = tuple(s.tag for s in SOURCES)


class GateError(RuntimeError):
    """Plain regenerated probabilities differ from the frozen oof_k36.csv beyond PLAIN_TOL."""

    def __init__(self, record):
        super().__init__(f"plain reproduction max abs diff {record.get('max_abs_diff')!r} > {PLAIN_TOL}")
        self.record = record


def source_for(tag):
    match = [s for s in SOURCES if s.tag == tag]
    need(len(match) == 1, f'unknown source tag {tag!r}')
    return match[0]


def stamp():
    return time.strftime('%Y%m%dT%H%M%S')


def write_json_x(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=1, allow_nan=False)
        stream.write('\n')


def rel(path, root=ROOT):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


# --------------------------------------------------------------------------- authentication

def release_entries(root=ROOT):
    """L2 entries of the pinned v2 release manifests; both must list the same ten sources."""
    lists = []
    for relpath, expected in RELEASE_MANIFESTS.items():
        path = Path(root) / relpath
        need(sha(path) == expected, 'v2 release model manifest changed: ' + relpath)
        lists.append([e for e in read(path)['entries'] if e.get('family') == 'l2'])
    need(all(x == lists[0] for x in lists), 'v2 release manifests disagree on L2 entries')
    entries = lists[0]
    need([(e['tag'], e['seed'], e['fold']) for e in entries] == [(s.tag, s.seed, s.fold) for s in SOURCES],
         'v2 release L2 entries differ from the ten registered sources')
    return {e['tag']: e for e in entries}


def open_corpus(root=ROOT):
    corpus = e124.Corpus(Path(root) / CORPUS_REL)
    need(corpus.contract == CONTRACT, 'corpus contract differs')
    ranges = e124.plane_ranges(corpus.spec)
    return corpus, ranges, sha(corpus.dir / 'manifest.json')


def held_out_ids(canonical, fold_of, fold):
    """The model's own EXP099 held-out fold only, in canonical EXP099 order."""
    need(fold in set(fold_of.values()), f'fold {fold} absent from roster')
    ids = [u for u in canonical if fold_of[u] == fold]
    need(ids and len(set(ids)) == len(ids), 'held-out roster empty or duplicated')
    return ids


def roster(src, p, spec, root=ROOT):
    """Authenticated held-out IDs for one source; no label or report is read."""
    need(spec.get('fold') == src.fold and list(spec.get('eval_folds', ())) == [src.fold]
         and src.fold not in spec.get('train_folds', ()), f'{src.tag}: run spec fold differs from source fold')
    canonical, fold_of = guard.fold_rosters(p, root)
    ids = held_out_ids(canonical, fold_of, src.fold)
    digest = uid_hash(ids)
    fold_rec = p['folds'][str(src.fold)]
    need(digest == spec['eval_ids_sha256'] == fold_rec['ordered_ids']['val_sha256'],
         f'{src.tag}: held-out roster hash differs from registration')
    need(len(ids) == spec['n_val'] == fold_rec['n_val'], f'{src.tag}: held-out size differs')
    train = {u for u in canonical if fold_of[u] in spec['train_folds']}
    need(not train & set(ids), f'{src.tag}: held-out IDs overlap training folds')
    return ids, dict(canonical_studies=len(canonical), held_out=len(ids), held_out_ids_sha256=digest,
                     exp099_ordered_ids=dict(path=fold_rec['ordered_ids']['path'],
                                            sha256=sha(Path(root) / fold_rec['ordered_ids']['path'])),
                     base_ordered_ids=dict(path=p['paths']['base_ordered_ids'],
                                           sha256=sha(Path(root) / p['paths']['base_ordered_ids'])))


def check_checkpoint_meta(ckpt, src, ids, protocol_sha):
    prov = ckpt.get('provenance') or {}
    need(ckpt.get('tag') == src.tag and ckpt.get('fold') == src.fold and ckpt.get('arch') == 'resnet50'
         and ckpt.get('norm') == 'radimagenet' and ckpt.get('window_policy') == 'within-slot-valid'
         and ckpt.get('input_policy') == 'center3' and ckpt.get('contract') == CONTRACT
         and ckpt.get('label_set') == 'l2' and ckpt.get('init') == 'radimagenet' and 'heads' not in ckpt,
         f'{src.tag}: checkpoint metadata differs from the L2 RadR50 recipe')
    need(prov.get('run') == src.tag and prov.get('seed') == src.seed
         and prov.get('eval_ids_ordered_sha256') == uid_hash(ids) and prov.get('protocol_sha256') == protocol_sha,
         f'{src.tag}: checkpoint provenance differs')


class Auth:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def authenticate(src, corpus, corpus_sha, release, root=ROOT, load=True):
    """Bind one source to inventory, provenance, protocol, roster, corpus and v2 manifest."""
    root = Path(root)
    exp = root / 'artifacts/experiments' / src.tag
    inventory = read(exp / 'output-inventory.json')
    files = {}
    for name in ('model_best.pt', 'provenance.json', 'oof_k36.csv'):
        path = exp / name
        digest, size = sha(path), path.stat().st_size
        need(name in inventory and inventory[name]['sha256'] == digest and inventory[name]['bytes'] == size,
             f'{src.tag}/{name} differs from its output-inventory.json')
        files[name] = dict(sha256=digest, bytes=size)
    entry = release[src.tag]
    need(entry['source_sha256'] == files['model_best.pt']['sha256'] and entry['seed'] == src.seed
         and entry['fold'] == src.fold, f'{src.tag}: checkpoint differs from the v2 release model manifest')
    ppath = root / src.protocol
    protocol_sha, p = sha(ppath), read(ppath)
    prov = read(exp / 'provenance.json')
    need(prov.get('run') == src.tag and prov.get('fold') == src.fold and prov.get('seed') == src.seed
         and prov.get('label_set') == 'l2' and prov.get('eval_folds') == [src.fold]
         and prov.get('arch') == 'resnet50' and prov.get('norm') == 'radimagenet'
         and prov.get('input_policy') == 'center3' and prov.get('k_eval_extra') == K
         and prov.get('serving') == 'single-head probabilities' and prov.get('heads') == []
         and prov.get('checkpoint_selection') == 'fixed-final' and prov.get('protocol_sha256') == protocol_sha,
         f'{src.tag}: provenance.json differs from the L2 K36 contract')
    spec = p['runs'][src.tag]
    a = p['common_args']
    need(spec.get('arm') == 'L2' and spec.get('label_set') == 'l2' and spec.get('heads') == []
         and spec.get('seed', p['seed']) == src.seed, f'{src.tag}: run spec differs')
    need(a['arch'] == 'resnet50' and a['res'] == 224 and a['norm'] == 'radimagenet'
         and a['input_policy'] == 'center3' and a['window_policy'] == 'within-slot-valid'
         and a['eval_amp'] == 'fp16' and a['runtime_policy'] == 'strict', f'{src.tag}: recipe differs')
    need(p['extra_eval']['k_eval'] == K and p['extra_eval']['file'] == 'oof_k36.csv', 'K36 extra eval differs')
    need(p['paths']['corpus'] == CORPUS_REL and p['corpus_manifest_sha256'] == corpus_sha,
         f'{src.tag}: corpus binding differs')
    ids, roster_rec = roster(src, p, spec, root)
    need(prov.get('eval_ids_ordered_sha256') == roster_rec['held_out_ids_sha256'], f'{src.tag}: provenance roster differs')
    need(set(ids) <= set(corpus.row), f'{src.tag}: held-out study absent from corpus')
    eval_batch = spec['eval_batch']
    need(eval_batch == a['bs'] // 2 == p['folds'][str(src.fold)]['eval_batch'], 'eval batch differs')
    ckpt = None
    if load:
        import torch
        ckpt = torch.load(exp / 'model_best.pt', map_location='cpu', weights_only=False)
        check_checkpoint_meta(ckpt, src, ids, protocol_sha)
    record = dict(tag=src.tag, seed=src.seed, fold=src.fold, experiment_dir=rel(exp, root), files=files,
                  output_inventory_sha256=sha(exp / 'output-inventory.json'),
                  protocol=dict(path=src.protocol, sha256=protocol_sha),
                  release_entry={k: entry[k] for k in ('tag', 'seed', 'fold', 'path', 'source_sha256', 'sha256')},
                  release_manifests=dict(RELEASE_MANIFESTS), roster=roster_rec, eval_batch=eval_batch)
    return Auth(src=src, record=record, ids=ids, ckpt=ckpt, p=p, spec=spec,
                stored_path=exp / 'oof_k36.csv', eval_batch=eval_batch)


# --------------------------------------------------------------------------- model / windows / TTA

def load_l2_model(ckpt):
    """Single-head RadR50 RaptorClassifier with an exact state schema and strict load."""
    need(ckpt.get('arch') == 'resnet50' and 'heads' not in ckpt and isinstance(ckpt.get('model'), dict),
         'not a single-head RadR50 checkpoint')
    model = e124.build_model(ckpt['arch'], pretrained=False)
    expected, state = model.state_dict(), ckpt['model']
    need(set(state) == set(expected) and all(tuple(state[k].shape) == tuple(expected[k].shape)
                                             and state[k].dtype == expected[k].dtype for k in expected),
         'model state schema differs (missing/unexpected/shape/dtype)')
    result = model.load_state_dict(state, strict=True)
    return model.eval(), dict(tensors=len(state), params=int(sum(v.numel() for v in state.values())),
                              strict_result=str(result))


def window_flags(centers, ranges, spec, depth):
    """EXP124 flip_mask, cross-checked against the EXP169 serving plane_flags."""
    flags = e124.flip_mask(centers, ranges)
    need(flags == serving.plane_flags(centers, dict(spec, depth=depth)), 'EXP124/EXP169 plane rules disagree')
    return flags


class CheckedPlaneWindows(e124.PlaneAwareWindows):
    """EXP124 PlaneAwareWindows plus the EXP169 serving plane_flags agreement check."""

    def __getitem__(self, i):
        x, y, uid = self.base[i]
        centers = e124.select_centers(self.corpus.mask(uid), self.corpus.depth, self.base.k, self.corpus.spec,
                                      self.base.window_policy, None)
        flags = window_flags(centers, self.ranges, self.corpus.spec, self.corpus.depth)
        return x, e124.mirror_windows(x, flags), y, uid


def window_census(ids, corpus, ranges):
    """Count K36 windows per plane (main process; same centres as the loader)."""
    counts = dict(Sagittal=0, Coronal=0, Axial=0, padding=0, mirrored=0)
    for uid in ids:
        centers = e124.select_centers(corpus.mask(uid), corpus.depth, K, corpus.spec, 'within-slot-valid', None)
        need(len(centers) == K, 'K36 centre count differs')
        flags = window_flags(centers, ranges, corpus.spec, corpus.depth)
        counts['mirrored'] += sum(flags)
        for c in centers:
            if c == -1:
                counts['padding'] += 1
            else:
                counts[next(r['plane'] for r in ranges if r['start'] <= c < r['stop'])] += 1
    counts['studies'] = len(ids)
    need(counts['mirrored'] == counts['Coronal'] + counts['Axial'], 'mirror census differs from plane census')
    return counts


def check_probabilities(values, what='probabilities'):
    import numpy as np
    values = np.asarray(values)
    need(values.ndim == 2 and values.shape[1] == len(TARGETS) and np.isfinite(values).all()
         and ((values >= 0) & (values <= 1)).all(), f'{what} invalid (shape/nonfinite/outside [0,1])')
    return values


def tta_mean(plain, mirrored):
    """Inputs are already sigmoid(logits.float()); returns (plain + mirrored) / 2 (no ranks)."""
    import numpy as np
    plain, mirrored = np.asarray(plain, np.float64), np.asarray(mirrored, np.float64)
    need(plain.shape == mirrored.shape, 'plain/mirrored shapes differ')
    check_probabilities(plain, 'plain')
    check_probabilities(mirrored, 'mirrored')
    return check_probabilities((plain + mirrored) / 2, 'TTA')


def plain_gate(plain, stored, tol=PLAIN_TOL):
    import numpy as np
    a, b = np.asarray(plain, np.float64), np.asarray(stored, np.float64)
    need(a.shape == b.shape and a.ndim == 2 and a.shape[1] == len(TARGETS), 'plain/stored shapes differ')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise GateError(dict(max_abs_diff=None, nonfinite=True, rows=len(a), tolerance=tol))
    diff = np.abs(a - b)
    record = dict(max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                  p99_abs_diff=float(np.percentile(diff, 99)), rows=int(len(a)), tolerance=tol,
                  serving_parity_tolerance_not_replaced=SERVING_TOL, passed=bool(diff.max() <= tol))
    if not record['passed']:
        raise GateError(record)
    return record


def check_output_csv(path, ids):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path, dtype={UID: str})
    need(list(df.columns) == [UID] + TARGETS, f'{Path(path).name}: target columns differ')
    need(df[UID].tolist() == list(ids) and df[UID].is_unique, f'{Path(path).name}: held-out IDs/order differ')
    return check_probabilities(df[TARGETS].to_numpy(np.float64), Path(path).name)


# --------------------------------------------------------------------------- runtime records

def gpu_snapshot():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        name, driver, used, total, util = [x.strip() for x in out.stdout.strip().splitlines()[0].split(',')]
        return dict(name=name, driver=driver, memory_used_mib=int(used), memory_total_mib=int(total),
                    utilization_pct=int(util), time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    except Exception as exc:  # informational only
        return dict(error=f'{type(exc).__name__}: {exc}')


def environment_record():
    import torch
    from training.src.raptor_common import environment
    env = environment()
    env.update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
               thread_env={k: os.environ.get(k) for k in THREAD_ENV}, hf_hub_offline=os.environ.get('HF_HUB_OFFLINE'))
    return env


def memory_record(device):
    rec = {}
    try:
        import psutil
        info = psutil.Process().memory_info()
        rec.update(ram_rss_bytes=int(info.rss), ram_peak_working_set_bytes=getattr(info, 'peak_wset', None))
    except ImportError:
        rec.update(ram_rss_bytes=None)
    if str(device).startswith('cuda'):
        import torch
        free, total = torch.cuda.mem_get_info()
        rec.update(vram_max_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                   vram_max_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                   device_free_bytes=int(free), device_total_bytes=int(total))
    return rec


def setup_runtime(device, min_free_gib):
    import torch
    need(device == 'cuda', 'pilot/run require --device cuda (fp16 release policy)')
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    before = gpu_snapshot()
    runtime = e124.configure_runtime('fp16', 'cuda', policy='strict')
    b = runtime['backends']
    need(b['cudnn_benchmark'] is False and b['cudnn_deterministic'] is True and b['deterministic_algorithms'] is True
         and torch.get_num_threads() == 1, 'backend/thread policy differs')
    free, _total = torch.cuda.mem_get_info()
    need(free >= min_free_gib * 2 ** 30, f'GPU free memory {free / 2 ** 30:.2f} GiB < required {min_free_gib} GiB')
    return runtime, before


def code_hashes(root=ROOT):
    return {name: sha(Path(root) / name) for name in CODE_FILES}


def infer(model, auth, corpus, ranges, ids, device, workers):
    import numpy as np
    from torch.utils.data import DataLoader
    a = auth.p['common_args']
    labels = {u: np.zeros(len(TARGETS), np.float32) for u in ids}  # label-free
    ds = e124.CenterSliceWindows(corpus.dir, list(ids), labels, a['res'], K, False, a['norm'], auth.src.seed,
                                 window_policy=a['window_policy'], input_policy=a['input_policy'])
    loader = DataLoader(CheckedPlaneWindows(ds, corpus, ranges), batch_size=auth.eval_batch, shuffle=False,
                        num_workers=workers, collate_fn=e124.collate_pair, persistent_workers=False,
                        worker_init_fn=guard.worker_init)
    plain, mirrored, got = e124.infer_pair(model, loader, device, 'l2')  # single-head sigmoid(logits.float())
    need(got == list(ids), 'inference UID order changed')
    return check_probabilities(plain, 'plain'), check_probabilities(mirrored, 'mirrored')


# --------------------------------------------------------------------------- receipts

def produce(out_base, src, ids, plain, tta, gate, extra, begun=None):
    """Write one model's ``<tag>/`` folder: stage, read back, inventory, receipt, rename into place.

    The only writer of check-B outputs (CPU; called by produce_model after GPU inference).
    The receipt status equals RECEIPT_STATUS, the value tools/exp169_v2_dino_marginal.py
    v1.1.0 check_flip_receipt reads; its tests parse this function's ``receipt = dict(...)``."""
    t0 = time.perf_counter()
    final = Path(out_base) / src.tag
    need(not final.exists(), f'refusing to overwrite {final}')
    stage = Path(out_base) / f'.{src.tag}.partial-{os.getpid()}-{stamp()}'
    stage.mkdir(parents=True)
    e124.write_predictions(stage / PLAIN, ids, plain)
    e124.write_predictions(stage / TTA, ids, tta)
    for name in (PLAIN, TTA):
        check_output_csv(stage / name, ids)
    inventory = {name: dict(sha256=sha(stage / name), bytes=(stage / name).stat().st_size) for name in (PLAIN, TTA)}
    write_json_x(stage / INVENTORY, inventory)
    extra = dict(extra)
    if 'durations' in extra:
        extra['durations'] = dict(extra['durations'], write_s=time.perf_counter() - t0,
                                  total_s=(time.perf_counter() - begun) if begun is not None else None)
    receipt = dict(
        schema=SCHEMA, job_id=JOB_ID, status='ADMITTED', tag=src.tag, seed=src.seed, fold=src.fold,
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), labels_read=False,
        tool=dict(path=TOOL_REL, version=VERSION, author=AUTHOR), plain_gate=gate,
        outputs=dict(files=inventory, inventory_sha256=sha(stage / INVENTORY), rows=len(ids),
                     ids_sha256=uid_hash(ids), unique_ids=len(set(ids)), targets=TARGETS),
        **extra)
    need(receipt['status'] == RECEIPT_STATUS, 'receipt status literal differs from RECEIPT_STATUS')
    write_json_x(stage / RECEIPT, receipt)
    os.replace(stage, final)
    return final, receipt


def verify_folder(folder, src, ids, source_sha256):
    """Receipt/inventory/CSV/gate checks against the expected roster and source checkpoint sha."""
    folder = Path(folder)
    r = read(folder / RECEIPT)
    need(r.get('schema') == SCHEMA and r.get('job_id') == JOB_ID and r.get('status') == RECEIPT_STATUS
         and (r.get('tag'), r.get('seed'), r.get('fold')) == (src.tag, src.seed, src.fold),
         f'{src.tag}: receipt identity/status differs')
    inventory = read(folder / INVENTORY)
    need(sha(folder / INVENTORY) == r['outputs']['inventory_sha256'] and inventory == r['outputs']['files']
         and set(inventory) == {PLAIN, TTA}, f'{src.tag}: output inventory differs from receipt')
    for name in (PLAIN, TTA):
        need(sha(folder / name) == inventory[name]['sha256'] and (folder / name).stat().st_size == inventory[name]['bytes'],
             f'{src.tag}/{name}: file hash differs from inventory')
    need(r['source']['files']['model_best.pt']['sha256'] == source_sha256, f'{src.tag}: source checkpoint binding differs')
    need(uid_hash(ids) == r['outputs']['ids_sha256'] and r['outputs']['rows'] == len(ids), f'{src.tag}: roster differs')
    for name in (PLAIN, TTA):
        check_output_csv(folder / name, ids)
    gate = r['plain_gate']
    need(gate['passed'] is True and gate['max_abs_diff'] is not None and gate['max_abs_diff'] <= PLAIN_TOL
         and gate['tolerance'] == PLAIN_TOL, f'{src.tag}: plain gate not passed')
    return dict(tag=src.tag, rows=len(ids), ids=ids, plain_max_abs_diff=gate['max_abs_diff'])


def verify_receipt(folder, src, release, root=ROOT):
    folder, root = Path(folder), Path(root)
    ckpt_sha = sha(root / 'artifacts/experiments' / src.tag / 'model_best.pt')
    need(ckpt_sha == release[src.tag]['source_sha256'], f'{src.tag}: checkpoint differs from v2 release manifest')
    r = read(folder / RECEIPT)
    need(sha(root / src.protocol) == r['source']['protocol']['sha256'], f'{src.tag}: protocol changed')
    p = read(root / src.protocol)
    ids, _ = roster(src, p, p['runs'][src.tag], root)
    return verify_folder(folder, src, ids, ckpt_sha)


def resume_loop(sources, out_base, verify, produce_one, log=print):
    """Serial models; a tag whose folder verifies is skipped, an unverifiable folder stops the job."""
    results = []
    for src in sources:
        final = Path(out_base) / src.tag
        if final.exists():
            res = verify(final, src)
            res.pop('ids', None)
            results.append(dict(res, status='SKIPPED_VERIFIED'))
        else:
            results.append(produce_one(src))
        log(json.dumps(results[-1]))
    return results


def verify_all(out_base, root=ROOT):
    release = release_entries(root)
    results, problems, per_seed = [], [], {}
    for src in SOURCES:
        folder = Path(out_base) / src.tag
        try:
            need(folder.is_dir(), f'{src.tag}: output folder missing')
            res = verify_receipt(folder, src, release, root)
            per_seed.setdefault(src.seed, []).append(res.pop('ids'))
            results.append(dict(res, status='VERIFIED'))
        except Exception as exc:
            problems.append(f'{src.tag}: {type(exc).__name__}: {exc}')
            results.append(dict(tag=src.tag, status='FAILED'))
    coverage = {}
    for seed, lists in per_seed.items():
        flat = [u for ids in lists for u in ids]
        coverage[seed] = dict(folds=len(lists), studies=len(flat), unique=len(set(flat)))
        if len(lists) == 5 and not (len(flat) == len(set(flat)) == STUDIES_PER_SEED):
            problems.append(f'seed {seed}: held-out coverage {len(set(flat))}/{len(flat)} != {STUDIES_PER_SEED}')
    return dict(status='PASS' if not problems else 'FAIL', models=results, coverage=coverage, problems=problems)


# --------------------------------------------------------------------------- subcommands

def preflight(root=ROOT):
    """CPU only. Authenticates every source and strict-loads every state; writes nothing."""
    import torch
    release = release_entries(root)
    corpus, ranges, corpus_sha = open_corpus(root)
    out = []
    for src in SOURCES:
        auth = authenticate(src, corpus, corpus_sha, release, root)
        _model, load_rec = load_l2_model(auth.ckpt)
        census = window_census(auth.ids, corpus, ranges)
        stored = e124.stored_predictions(auth.stored_path, auth.ids)
        check_probabilities(stored, 'frozen oof_k36')
        out.append(dict(tag=src.tag, seed=src.seed, fold=src.fold, held_out=len(auth.ids),
                        held_out_ids_sha256=auth.record['roster']['held_out_ids_sha256'],
                        checkpoint_sha256=auth.record['files']['model_best.pt']['sha256'],
                        tensors=load_rec['tensors'], mirrored_windows=census['mirrored'],
                        padding_windows=census['padding']))
        del auth, _model
    need(not torch.cuda.is_initialized(), 'preflight initialised CUDA')
    total = sum(x['held_out'] for x in out)
    return dict(status='PASS', version=VERSION, models=out, total_model_studies=total,
                corpus=dict(path=CORPUS_REL, contract=CONTRACT, manifest_sha256=corpus_sha, plane_ranges=ranges))


def lane_body(registered_at):
    return dict(
        schema=LANE_SCHEMA, job_id=JOB_ID, lane='LOCAL', gpu=LANE_GPU, gpu_count=1, parallel=False,
        reason=('RAPTOR64_336_C140_S02 corpus and the ten checkpoints are local; this is the GPU that trained '
                'these OOFs (EXP149/EXP157 manifests local.gpu), so there is no transfer and no cross-GPU step'),
        cap_gpu_minutes=CAP_MINUTES, pilot_first=True, pilot_max_studies=PILOT_MAX, projection_margin=MARGIN,
        pilot_rule=('pilot (<= 40 label-free held-out studies of EXP149_L2_S42_F0) must PASS first; it exits 3 '
                    'when the projected full-job minutes x 1.25 exceed the cap or its plain subset gate fails, and '
                    'run refuses without a PASS pilot for the same tool version and this lane file'),
        plain_gate_max_abs_diff=PLAIN_TOL, min_free_gib=4.0,
        launcher=('tools/gpu_queue.py plan written by queue-json: pilot, then run after the pilot SUCCESS '
                  '(each through tools/gpu_job.py; a non-zero exit writes FAILED and stops the plan)'),
        spec=CHECKS_REL, spec_lane_table_note=('the section B lane table lists Vast 3090 / Kaggle T4; LOCAL '
                                               'RTX 4070 Ti SUPER chosen by B; this file is the lane/budget registration'),
        registered_at=registered_at, registered_by=AUTHOR, tool=dict(path=TOOL_REL, version=VERSION))


def require_lane(root=ROOT, device_name=None):
    path = Path(root) / LANE_REL
    need(path.is_file(), f'lane registration missing: {LANE_REL} (write it once with register-lane)')
    lane = read(path)
    need(lane.get('schema') == LANE_SCHEMA and lane.get('job_id') == JOB_ID and lane.get('lane') == 'LOCAL'
         and lane.get('gpu') == LANE_GPU and lane.get('cap_gpu_minutes') == CAP_MINUTES
         and lane.get('pilot_first') is True and lane.get('pilot_max_studies') == PILOT_MAX
         and lane.get('parallel') is False and lane.get('projection_margin') == MARGIN,
         'lane registration differs from this tool (lane/GPU/cap/pilot rule)')
    if device_name is not None:
        need(device_name == lane['gpu'], f'GPU {device_name!r} is not the registered lane GPU {lane["gpu"]!r}')
    return lane, sha(path)


def require_pilot_pass(out_base, lane_sha):
    """Pilot-first rule: a PASS pilot of this tool version bound to the current lane file."""
    passes = []
    for path in sorted((Path(out_base) / '_pilot').glob('pilot-*.json')):
        try:
            r = read(path)
        except (OSError, ValueError):
            continue
        projection = r.get('projection') or {}
        if (r.get('schema') == PILOT_SCHEMA and r.get('job_id') == JOB_ID and r.get('status') == 'PASS'
                and (r.get('tool') or {}).get('version') == VERSION and r.get('lane_sha256') == lane_sha
                and isinstance(projection.get('projected_minutes'), (int, float))
                and projection['projected_minutes'] <= CAP_MINUTES):
            passes.append(path)
    need(passes, 'pilot-first rule: no PASS pilot for this tool version and lane registration')
    return passes[-1]


def pilot_status(projected_minutes, gate_passed):
    if projected_minutes > CAP_MINUTES:
        return 'STOP_PROJECTION_EXCEEDS_CAP'
    if not gate_passed:
        return 'STOP_PLAIN_SUBSET_GATE'
    return 'PASS'


def pilot_exit_code(record):
    """0 only for a PASS pilot; every STOP status exits 3 so gpu_job writes FAILED and the queue stops."""
    return 0 if record.get('status') == 'PASS' else 3


def pilot(tag, limit, workers, min_free_gib, root=ROOT, out_base=None):
    import torch
    out_base = Path(out_base or Path(root) / OUT_REL)
    need(1 <= limit <= PILOT_MAX, f'pilot limit must be 1..{PILOT_MAX}')
    src = source_for(tag)
    _lane, lane_sha = require_lane(root)
    runtime, before = setup_runtime('cuda', min_free_gib)
    require_lane(root, torch.cuda.get_device_name(0))
    env = environment_record()
    release = release_entries(root)
    corpus, ranges, corpus_sha = open_corpus(root)
    t0 = time.perf_counter()
    auth = authenticate(src, corpus, corpus_sha, release, root)
    t_auth = time.perf_counter() - t0
    t0 = time.perf_counter()
    model, load_rec = load_l2_model(auth.ckpt)
    model = model.to('cuda')
    auth.ckpt = None
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    ids = auth.ids[:limit]
    census = window_census(ids, corpus, ranges)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    infer(model, auth, corpus, ranges, ids[:auth.eval_batch], 'cuda', workers)  # warm-up batch, not in the rate
    torch.cuda.synchronize()
    t_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    plain, mirrored = infer(model, auth, corpus, ranges, ids, 'cuda', workers)
    torch.cuda.synchronize()
    t_inf = time.perf_counter() - t0
    tta = tta_mean(plain, mirrored)
    try:
        gate = plain_gate(plain, e124.stored_predictions_subset(auth.stored_path, ids))
    except GateError as exc:
        gate = dict(exc.record, passed=False)
    total_studies = sum(read(Path(root) / s.protocol)['runs'][s.tag]['n_val'] for s in SOURCES)
    per_study = t_inf / len(ids)
    fixed = t_auth + t_load
    nominal = (total_studies * per_study + len(SOURCES) * fixed) / 60
    projected = nominal * MARGIN
    after = gpu_snapshot()
    status = pilot_status(projected, gate['passed'])
    record = dict(schema=PILOT_SCHEMA, job_id=JOB_ID, tool=dict(path=TOOL_REL, version=VERSION, author=AUTHOR),
                  created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status=status, tag=tag, studies=len(ids),
                  labels_read=False, workers=workers, eval_batch=auth.eval_batch, lane_sha256=lane_sha,
                  timing=dict(authenticate_s=t_auth, model_load_s=t_load, warmup_batch_s=t_warm,
                              timed_inference_s=t_inf, per_study_two_forwards_s=per_study),
                  projection=dict(total_model_studies=total_studies, forwards_per_study=2, models=len(SOURCES),
                                  per_model_fixed_s=fixed, margin=MARGIN, nominal_minutes=nominal,
                                  projected_minutes=projected, cap_minutes=CAP_MINUTES,
                                  exceeds_cap=projected > CAP_MINUTES,
                                  contended=bool(before.get('utilization_pct', 0) > 5)),
                  plain_subset_gate=gate, tta_minus_plain_max_abs=float(abs(tta - plain).max()),
                  window_census=census, memory=memory_record('cuda'), gpu_before=before, gpu_after=after,
                  runtime=runtime, environment=env, model_load=load_rec, code_sha256=code_hashes(root))
    path = out_base / '_pilot' / f'pilot-{tag}-{stamp()}.json'
    write_json_x(path, record)
    print(json.dumps(dict(status=status, pilot=rel(path, root), per_study_s=round(per_study, 4),
                          projected_minutes=round(projected, 1), nominal_minutes=round(nominal, 1),
                          vram_max_reserved_gib=round(record['memory']['vram_max_reserved_bytes'] / 2 ** 30, 2),
                          plain_subset_max_abs_diff=gate.get('max_abs_diff'), contended=record['projection']['contended']),
                     indent=1), flush=True)
    return record


def produce_model(src, corpus, ranges, corpus_sha, release, runtime, env, workers, out_base, binding, root=ROOT):
    import torch
    begun = time.perf_counter()
    before = gpu_snapshot()
    t0 = time.perf_counter()
    auth = authenticate(src, corpus, corpus_sha, release, root)
    t_auth = time.perf_counter() - t0
    t0 = time.perf_counter()
    model, load_rec = load_l2_model(auth.ckpt)
    auth.ckpt = None
    model = model.to('cuda')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter() - t0
    census = window_census(auth.ids, corpus, ranges)
    t0 = time.perf_counter()
    plain, mirrored = infer(model, auth, corpus, ranges, auth.ids, 'cuda', workers)
    torch.cuda.synchronize()
    t_inf = time.perf_counter() - t0
    memory = memory_record('cuda')
    del model
    torch.cuda.empty_cache()
    stored = e124.stored_predictions(auth.stored_path, auth.ids)
    try:
        gate = plain_gate(plain, stored)
    except GateError as exc:
        write_json_x(Path(out_base) / '_failures' / f'{src.tag}-{stamp()}.json',
                     dict(job_id=JOB_ID, tool_version=VERSION, tag=src.tag, status='STOP_PLAIN_GATE',
                          plain_gate=exc.record, source=auth.record, labels_read=False))
        raise
    tta = tta_mean(plain, mirrored)
    extra = dict(
        spec=dict(path=CHECKS_REL, sha256=sha(Path(root) / CHECKS_REL)), source=auth.record, **binding,
        corpus=dict(path=CORPUS_REL, contract=CONTRACT, manifest_sha256=corpus_sha, plane_ranges=ranges),
        recipe=dict(k_eval=K, window_policy='within-slot-valid', input_policy='center3', norm='radimagenet', res=224,
                    eval_batch=auth.eval_batch, amp='fp16', workers=workers,
                    transform='horizontal mirror of coronal/axial windows only; sagittal and padding unchanged',
                    plane_rule='EXP124 flip_mask == EXP169 serving plane_flags (asserted per study)',
                    probability='single-head sigmoid(logits.float()) per pass (EXP124 served_probabilities)',
                    tta='(sigmoid(plain.float()) + sigmoid(mirrored.float())) / 2 before any ranks'),
        window_census=census,
        tta_vs_plain=dict(max_abs_diff=float(abs(tta - plain).max()), mean_abs_diff=float(abs(tta - plain).mean())),
        model_load=load_rec, runtime=runtime, environment=env, gpu_before=before, gpu_after=gpu_snapshot(),
        durations=dict(authenticate_s=t_auth, model_load_s=t_load, inference_s=t_inf),
        memory=memory, code_sha256=code_hashes(root))
    final, receipt = produce(out_base, src, auth.ids, plain, tta, gate, extra, begun)
    verify_receipt(final, src, release, root)
    return dict(tag=src.tag, status=RECEIPT_STATUS, rows=len(auth.ids), plain_max_abs_diff=gate['max_abs_diff'],
                inference_s=round(t_inf, 1), total_s=round(receipt['durations']['total_s'], 1))


def run(workers, min_free_gib, root=ROOT, out_base=None):
    import torch
    out_base = Path(out_base or Path(root) / OUT_REL)
    _lane, lane_sha = require_lane(root)
    pilot_path = require_pilot_pass(out_base, lane_sha)
    runtime, _before = setup_runtime('cuda', min_free_gib)
    require_lane(root, torch.cuda.get_device_name(0))
    binding = dict(lane=dict(path=LANE_REL, sha256=lane_sha),
                   pilot=dict(path=rel(pilot_path, root), sha256=sha(pilot_path)))
    env = environment_record()
    release = release_entries(root)
    corpus, ranges, corpus_sha = open_corpus(root)
    begun = time.perf_counter()
    try:
        results = resume_loop(
            SOURCES, out_base, lambda folder, src: verify_receipt(folder, src, release, root),
            lambda src: produce_model(src, corpus, ranges, corpus_sha, release, runtime, env, workers, out_base, binding, root),
            log=lambda line: print(line, flush=True))
    except GateError as exc:
        print(json.dumps(dict(status='STOP_PLAIN_GATE', plain_gate=exc.record)), flush=True)
        raise SystemExit(2)
    summary = verify_all(out_base, root)
    record = dict(job_id=JOB_ID, tool_version=VERSION, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  wall_s=time.perf_counter() - begun, results=results, verify=summary, **binding)
    write_json_x(out_base / f'run-{stamp()}.json', record)
    print(json.dumps(dict(status=summary['status'], wall_minutes=round(record['wall_s'] / 60, 1),
                          problems=summary['problems']), indent=1), flush=True)
    if summary['status'] != 'PASS':
        raise SystemExit(4)
    return record


def queue_entries(after=None, root=ROOT):
    """Pilot first, then run only after the pilot's SUCCESS.

    tools/gpu_job.py writes FAILED for any non-zero exit (pilot STOP exits 3), and
    tools/gpu_queue.py then stops the plan before ``run`` (no continue_on_failure)."""
    python, tool = e124.e113.train_python(), str(Path(root) / TOOL_REL)
    pilot_item = dict(tag=JOB_ID + '_PILOT', command=[python, tool, 'pilot', '--tag', TAGS[0], '--limit', str(PILOT_MAX),
                                                      '--device', 'cuda'])
    if after:
        pilot_item['after'] = after
    run_item = dict(tag=JOB_ID, command=[python, tool, 'run', '--device', 'cuda'], after=pilot_item['tag'])
    return [pilot_item, run_item]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', action='version', version=VERSION)
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('preflight')
    s = sub.add_parser('pilot')
    s.add_argument('--tag', choices=TAGS, default=TAGS[0])
    s.add_argument('--limit', type=int, default=PILOT_MAX)
    s.add_argument('--workers', type=int, choices=(0, 1, 2), default=2)
    s.add_argument('--min-free-gib', type=float, default=4.0)
    s.add_argument('--device', choices=['cuda'], default='cuda')
    s = sub.add_parser('run')
    s.add_argument('--workers', type=int, choices=(0, 1, 2), default=2)
    s.add_argument('--min-free-gib', type=float, default=4.0)
    s.add_argument('--device', choices=['cuda'], default='cuda')
    sub.add_parser('verify')
    s = sub.add_parser('queue-json')
    s.add_argument('--after', required=True, help="predecessor gpu_job tag, or 'none'")
    s.add_argument('--out', required=True)
    s = sub.add_parser('register-lane')
    s.add_argument('--registered-at', required=True, help='timestamp taken from `date` in the same command')
    a = ap.parse_args(argv)
    print(f'{JOB_ID} tool exp169_v2_l2_flip_oof v{VERSION} ({AUTHOR}) action={a.action}', flush=True)
    if a.action == 'preflight':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print(json.dumps(preflight(), indent=1), flush=True)
    elif a.action == 'pilot':
        code = pilot_exit_code(pilot(a.tag, a.limit, a.workers, a.min_free_gib))
        if code:
            raise SystemExit(code)
    elif a.action == 'run':
        run(a.workers, a.min_free_gib)
    elif a.action == 'verify':
        summary = verify_all(ROOT / OUT_REL)
        print(json.dumps(summary, indent=1), flush=True)
        if summary['status'] != 'PASS':
            raise SystemExit(4)
    elif a.action == 'register-lane':
        path = ROOT / LANE_REL
        write_json_x(path, lane_body(a.registered_at))
        require_lane(ROOT)
        print(json.dumps(dict(lane=LANE_REL, sha256=sha(path))), flush=True)
    else:
        path = Path(a.out)
        write_json_x(path, queue_entries(None if a.after.lower() == 'none' else a.after))
        print(path)


if __name__ == '__main__':
    main()
