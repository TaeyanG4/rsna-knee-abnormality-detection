"""EXP105 (H-B-15) combined submission: the unchanged 0.943 anchor + our two 5-fold arms on the anchor's shared decode,
final per-target rank blend 0.8 anchor / 0.2 own, with an anchor-only fail-safe.

  # build the kernel directories (local files only; never pushes, publishes or submits)
  python tools/exp105_combined_release.py build --variant full   --out .local/hb15-combined/knee-exp105-anchor-own-v1
  python tools/exp105_combined_release.py build --variant eng100 --out .local/hb15-combined/knee-exp105-anchor-own-100

Notebook cells
  full   : setup | anchor 0..26 | anchor 27a | hook | anchor 27b | anchor 28..29 | post
  eng100 : runtime100 proxy | setup | anchor 0..26 | 27a | hook | 27b | 28..29 | post | runtime100 final check
  * anchor cells are the private anchor copy (kaggle/packages/KNEE_ANCHOR_0943, submission 56773226, public 0.943)
    byte for byte; cell 27 is split between A:4233 and A:4234 (the halves concatenate to the original);
  * hook = .local/hb15-adapter/anchor_hook_cell.py byte for byte (verified on 2xT4 in knee-anchor-shared-100, D-B-018);
  * setup (first cell, standard library only, never raises): first-cell clock, embedded sources written to
    /kaggle/working/hb15/src after a sha256 check, asset bindings (setup_environment), HB15_* environment; on any
    problem no HB15_* variable is set, so the hook stays disabled and the anchor runs unchanged;
  * post (never raises): post_anchor(globals()) below.

post_anchor (the anchor's own cell 29 has already published submission.csv)
  1. finalize the hook; read sample_submission.csv and the anchor's submission.csv (same columns, same id set, finite,
     in [0, 1]) and back it up to exp105/anchor_submission.csv. Anything wrong -> the file is not touched.
  2. time guard (pre-registered, GUARD): run the arms only if
        elapsed since the first cell + max over the two parallel arms of
        [fixed + 1.3 * (per_study * N + decode * (studies without an OK store row + audits))] + margin <= 8.6 h;
  3. pinned wheels (sha256-checked, v5 verify_old_assets + the OpenCV hash) -> one --target environment; two workers
     in parallel (GPU0 coat5 = CoAtNet folds 0-4, GPU1 rn5 = ResNet34 folds 0-4, both K36 mean_prob, reviewed v5 arms);
     both are killed at first cell + 8.75 h;
  4. every worker output is checked (exit 0, receipt PASS and bindings, every study scored or listed as unscored,
     finite, in [0, 1], file hash); unscored studies (input preparation failed) <= 5% of the cohort;
  5. own = v5 rank_mean([coat5, rn5], [0.7, 0.3]) over the studies both arms scored (exp104 combine math, the same
     function); final per target = (0.8 * rank(anchor) + 0.2 * rank(own)) / 1.0, average-tie percentile ranks, the
     anchor over the full cohort and own over the scored studies; a study our arms could not score keeps rank(anchor);
  6. tmp file -> re-read (ids/order/columns = sample_submission, finite, in [0, 1], values = the computed blend) ->
     submission.csv still the anchor's bytes -> os.replace -> re-check (failure restores the anchor bytes);
  7. receipt exp105/receipt.json: path anchor_only|blended, reason, timings, counts, hashes.
Any exception, non-finite value, missing study, failed audit, timeout or failed check keeps the anchor-only file.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'exp105-anchor-own-v1'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
# Pre-registered (plan.md H-B-15, 2026-10-07 05:42, before any combined public score).
OWN_ARMS = (('coat5', 0, 0.7), ('rn5', 1, 0.3))     # (v5 arm name, GPU, own rank weight)
FINAL_WEIGHTS = (0.8, 0.2)                          # (anchor, own)
K_EVAL = 36
N_MODELS = 5
STORE_BYTES_PER_STUDY = 7_225_408   # 64 x 336 x 336 uint8 volume + mask row (raptor_shared_input store)

GUARD = {
    'budget_seconds': 8.6 * 3600,          # elapsed + projection must stay below this to start the arms
    'hard_deadline_seconds': 8.75 * 3600,  # workers are killed at first cell + this (Kaggle limit 9 h)
    'inflation': 1.3,
    # seconds per study on a T4 (D-B-015: CoAtNet-2 224 0.22 s/model at K36, ResNet34 0.091 s/model) x 5 models,
    # + 0.1 s store read / window preparation (hb15 T4: single B224 K24 0.32 s/study including two audit decodes)
    # coat5 raised 1.2 -> 1.6 after the review (B224 on T4 0.20 s/model at K24 ~ 0.30 s at K36, x5 + store read)
    'per_study_seconds': {'coat5': 1.6, 'rn5': 0.55},
    'decode_seconds_per_study': 6.52503319258,   # DICOM fallback per study (v5 TIME constant)
    'audit_studies': 2,
    'fixed_seconds': 300.0,                # env install, start-up, checkpoint hashing/loading, blend
    'margin_seconds': 300.0,
    'max_unscored_fraction': 0.05,
    'finalize_timeout_seconds': 120.0,
    'poll_seconds': 5.0,
}
EXP104_SOURCE = 'exp104_release_v5.py'
EXP104_SOURCE_SHA256 = '2100a5cb5b035c96ecd344b0383425412f54c24c59e5e6cdacec9cc555677aeb'
RELEASE_MANIFEST_NAME = 'coat5rn5-release-manifest.json'
RELEASE_MANIFEST_SHA256 = 'dd1a341fc4c4f4d696c75f75d88227c96b2d721b29b92790c5fcf72931306c84'
ASSETS_MANIFEST_SHA256 = '8d64433d7aa654a48e308ea73f3de0b68d8b22901f6ca6ca182d0ebaf4744423'
SHARED_INPUT_SHA256 = '9112b40c7b51b55887f7e4b2b23d9154efbb74b9008b174728f91052f09036e2'
HOOK_SHA256 = 'c7826695a32e875cd0a8ffc9f985fcc091f7e59c88196e7d51c30d18c65bf2ef'
ANCHOR_SHA256 = 'cd5b8b69f89bcfaa6b6ee065f7b18f7134dc71968be3827c87ca3e5672007cac'
RUNTIME100_SHA256 = '14b6622ceff87380760197ec0acda78fb50adad0ec08593ba573e62492b8a3e1'
OLD_DATASET = 'taeyangg4/knee-exp089-recipe-transfer-v1'
NEW_DATASET = 'taeyangg4/knee-release-assets-v5'
OPENCV_DATASET = 'mattiaangeli/opencv-python-headless-4120088-x86'
OPENCV_GLOB = 'opencv_python_headless-4.12.0.88-*.whl'
DOCKER_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
KERNELS = {'full': 'taeyangg4/knee-exp105-anchor-own-v1', 'eng100': 'taeyangg4/knee-exp105-anchor-own-100'}
WORKER = 'exp105_arm_worker.py'
TMP_SUFFIX = '.exp105-tmp'
MAX_NOTEBOOK_LINE = 30000     # Kaggle never started a notebook with a 66,897-character line (D-B-015 item 12)
MAX_ADDED_LINE = 4000

# local build inputs
ANCHOR_NB = ROOT / 'kaggle/packages/KNEE_ANCHOR_0943/rsna-knee-fast-2xt4-inference.ipynb'
ANCHOR_META = ROOT / 'kaggle/packages/KNEE_ANCHOR_0943/kernel-metadata.json'
RUNTIME100 = ROOT / '.local/d01-anchor-runtime-staging/20261003-count100-v1'
HOOK = ROOT / '.local/hb15-adapter/anchor_hook_cell.py'
RELEASE_MANIFEST = ROOT / '.local/exp104-release-v5/coat5rn5/release-manifest.json'
SOURCES = {  # embedded name -> (local path, pinned sha256 or None for the new EXP105 files)
    'raptor_shared_input.py': (ROOT / 'training/src/raptor_shared_input.py', SHARED_INPUT_SHA256),
    EXP104_SOURCE: (ROOT / 'tools' / EXP104_SOURCE, EXP104_SOURCE_SHA256),
    RELEASE_MANIFEST_NAME: (RELEASE_MANIFEST, RELEASE_MANIFEST_SHA256),
    'exp105_combined_release.py': (ROOT / 'tools/exp105_combined_release.py', None),
    WORKER: (ROOT / 'tools' / WORKER, None),
}
SPLIT_MARKER = ("    _arm['k_eval'] = 94\n", "rsna_phase('public_raptor', 'START')\n")


class AnchorOnly(Exception):
    """Keep the anchor-only submission; the message is the recorded reason."""


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


_V5 = {}


def load_v5(directory=None):
    """The reviewed release v5 module (exp104_release_v5.py, sha256-pinned) from `directory` (default: next to this)."""
    path = Path(directory or Path(__file__).resolve().parent) / EXP104_SOURCE
    key = str(path.resolve())
    if key not in _V5:
        need(sha(path) == EXP104_SOURCE_SHA256, 'exp104_release_v5.py differs from the reviewed v5 source')
        spec = importlib.util.spec_from_file_location('exp104_release_v5_for_exp105', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        need(module.TARGETS == TARGETS, 'v5 target order differs')
        _V5[key] = module
    return _V5[key]


# ------------------------------------------------------------------ release binding
def check_release(release):
    """The v5 coat5rn5 arms with the pre-registered names, order, weights, K and member counts."""
    arms = release['arms']
    need([a['name'] for a in arms] == [n for n, _, _ in OWN_ARMS], 'v5 arm names/order differ')
    for arm, (name, _, weight) in zip(arms, OWN_ARMS):
        need(arm['weight'] == weight and arm['k_eval'] == K_EVAL and arm['aggregator'] == 'mean_prob'
             and len(arm['checkpoints']) == N_MODELS, 'v5 arm differs from the pre-registered arm: ' + name)
    need(release['new_asset_dataset'] == NEW_DATASET
         and release['new_assets_manifest_sha256'] == ASSETS_MANIFEST_SHA256, 'v5 assets binding differs')
    return arms


def find_one(candidates, marker):
    found, seen = [], set()
    for candidate in map(Path, candidates):
        if (candidate / marker).is_file():
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                found.append(candidate)
    need(len(found) == 1, 'expected exactly one mount with %s, found %s' % (marker, [str(p) for p in found]))
    return found[0]


def setup_environment(src, variant='full', kaggle_root='/kaggle', bench_comp=None, bench_ids=None):
    """Called by the setup cell: verify the bindings, return the HB15_* environment (nothing is set here)."""
    src, k = Path(src), Path(kaggle_root)
    need(variant in KERNELS, 'unknown variant ' + repr(variant))
    v5 = load_v5(src)
    release = v5.checked_manifest(src / RELEASE_MANIFEST_NAME, RELEASE_MANIFEST_SHA256)
    arms = check_release(release)
    inputs = k / 'input'
    old = find_one([inputs / 'datasets' / OLD_DATASET, inputs / OLD_DATASET.split('/')[1]], v5.OLD_PROTOCOL)
    protocol = v5.verify_old_assets(old)              # protocol, serving sources and wheels by sha256
    new = find_one([inputs / 'datasets' / NEW_DATASET, inputs / NEW_DATASET.split('/')[1]], v5.NEW_MANIFEST)
    need(sha(new / v5.NEW_MANIFEST) == ASSETS_MANIFEST_SHA256, 'release-assets-v5 manifest differs')
    missing = [c['path'] for a in arms for c in a['checkpoints'] if not (new / c['path']).is_file()]
    need(not missing, 'missing checkpoints: %s' % missing)
    roots = [inputs / 'datasets' / OPENCV_DATASET, inputs / OPENCV_DATASET.split('/')[1]]
    opencv = [p for d in roots if d.is_dir() for p in sorted(d.glob(OPENCV_GLOB))]
    need(len(opencv) == 1, 'expected one OpenCV 4.12.0.88 wheel, found %d' % len(opencv))
    scratch = k / 'temp'
    try:
        scratch.mkdir(parents=True, exist_ok=True)
    except OSError:
        scratch = Path(tempfile.gettempdir())
    store = scratch / 'hb15-store'
    need(not store.exists() or not any(store.iterdir()), 'shared store directory is not empty: %s' % store)
    # The store is a sparse memmap; a full disk would SIGBUS the anchor process, so require 3x the projected
    # store size (7,225,408 bytes per study, 9.55 GB at 1,322) free before enabling it (review 2026-10-07).
    sample = next((p for p in [inputs / 'competitions' / 'rsna-knee-abnormality-detection' / 'sample_submission.csv',
                               inputs / 'rsna-knee-abnormality-detection' / 'sample_submission.csv'] if p.is_file()), None)
    n_cases = (sum(1 for _ in open(sample, encoding='utf-8-sig')) - 1) if sample else 1400
    free = shutil.disk_usage(scratch).free
    need(free >= 3 * STORE_BYTES_PER_STUDY * max(n_cases, 100), 'not enough free disk for the shared store: %d bytes' % free)
    env = {'HB15_SRC': str(src), 'HB15_ASSETS': str(old), 'HB15_NEW_ASSETS': str(new), 'HB15_OPENCV': str(opencv[0]),
           'HB15_STORE': str(store), 'HB15_ENV': str(scratch / 'hb15-env'), 'HB15_WORKING': str(k / 'working'),
           'HB15_RELEASE_MANIFEST': str(src / RELEASE_MANIFEST_NAME), 'HB15_VARIANT': variant, 'HB15_AUDIT': '2'}
    if variant == 'eng100':
        need(bench_comp is not None and bench_ids is not None, 'eng100 needs the runtime100 proxy cell')
        need(list(protocol['ids']) == [str(u) for u in bench_ids], 'EXP089 and runtime100 study order differ')
        comp = Path(bench_comp)
        env.update(HB15_SPLIT='train', HB15_ROWS_CSV=str(comp / 'train_series.csv'),
                   HB15_DICOM_ROOT=str(comp / 'train_series'), HB15_VERIFY_PROTOCOL=str(old / v5.OLD_PROTOCOL))
    else:
        need(bench_comp is None and bench_ids is None, 'the full variant takes no benchmark cohort')
    return env


# ------------------------------------------------------------------ tables
def read_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    need(len(rows) >= 2 and len(rows[0]) == len(set(rows[0])), 'empty table or duplicate header: %s' % path)
    need(all(len(r) == len(rows[0]) for r in rows[1:]), 'CSV width differs: %s' % path)
    return rows[0], rows[1:]


def probability_row(texts, where):
    values = [float(x) for x in texts]
    need(len(values) == len(TARGETS) and all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values),
         'non-finite or out-of-range value: ' + where)
    return values


def read_sample(comp):
    header, rows = read_table(Path(comp) / 'sample_submission.csv')
    need(header == [UID, *TARGETS], 'sample_submission columns differ')
    ids = [r[0] for r in rows]
    need(len(ids) == len(set(ids)) and all(ids), 'sample_submission ids empty or duplicated')
    return ids


def read_anchor(path, ids):
    """The anchor's published submission -> rows in sample order (it is written in test.csv order)."""
    header, rows = read_table(path)
    need(header == [UID, *TARGETS], 'anchor submission columns differ from sample_submission')
    got = [r[0] for r in rows]
    need(len(got) == len(set(got)) and set(got) == set(ids), 'anchor submission ids differ from sample_submission')
    by_id = {r[0]: probability_row(r[1:], 'anchor ' + r[0]) for r in rows}
    return [by_id[u] for u in ids]


def write_csv(v5, path, ids, cols):
    v5.write_columns(path, ids, cols)        # csv.writer rows of repr(float), exclusive create (EXP098 format)
    with Path(path).open('r+b') as f:
        os.fsync(f.fileno())


def verify_csv(path, ids, cols=None):
    """ids/order/columns = sample_submission, finite, in [0, 1]; with cols, every value equals the computed blend."""
    header, rows = read_table(path)
    need(header == [UID, *TARGETS], 'output columns differ')
    need([r[0] for r in rows] == ids, 'output ids/order differ from sample_submission')
    for i, r in enumerate(rows):
        values = probability_row(r[1:], 'output ' + r[0])
        if cols is not None:
            need(all(values[t] == cols[t][i] for t in range(len(TARGETS))), 'output value differs from the blend')
    return sha(path)


# ------------------------------------------------------------------ blend
def own_blend(v5, ids, coat, rn, weights):
    """Studies both arms scored (cohort order) and their v5 rank_mean over that subset: {index: 12 values}."""
    scored = [i for i, u in enumerate(ids) if u in coat and u in rn]
    if not scored:
        return {}
    cols = v5.rank_mean([[coat[ids[i]] for i in scored], [rn[ids[i]] for i in scored]], list(weights))
    return {i: [cols[t][k] for t in range(len(TARGETS))] for k, i in enumerate(scored)}


def final_blend(v5, anchor_rows, own_rows, weights=FINAL_WEIGHTS):
    """Per target: r_a = v5.average_rank of the anchor over the full cohort; r_o = v5.average_rank of the own blend over
    the scored studies; scored study -> (w_a*r_a + w_o*r_o)/(w_a + w_o) with the left-to-right float sums of
    v5.rank_mean (so with every study scored this IS v5.rank_mean([anchor, own], weights)); other studies keep r_a."""
    n = len(anchor_rows)
    need(n > 0 and all(len(r) == len(TARGETS) for r in anchor_rows), 'anchor table malformed')
    index = sorted(own_rows)
    need(all(0 <= i < n and len(own_rows[i]) == len(TARGETS) for i in index), 'own table malformed')
    need(len(weights) == 2 and all(v5.valid_weight(w) for w in weights), 'invalid final weights')
    cols = []
    for t in range(len(TARGETS)):
        r_a = v5.average_rank([row[t] for row in anchor_rows])
        col = list(r_a)
        if index:
            r_o = v5.average_rank([own_rows[i][t] for i in index])
            for k, i in enumerate(index):
                num = den = 0.0
                for w, v in zip(weights, (r_a[i], r_o[k])):
                    num += w * v
                    den += w
                col[i] = num / den
        cols.append(col)
    need(all(math.isfinite(v) and 0.0 <= v <= 1.0 for c in cols for v in c), 'final blend out of range')
    return cols


def rank_correlation(v5, anchor_rows, own_rows):
    """Mean over targets of the Pearson correlation of the anchor and own percentile ranks on the scored studies."""
    index = sorted(own_rows)
    if len(index) < 3:
        return None
    out = []
    for t in range(len(TARGETS)):
        a = v5.average_rank([anchor_rows[i][t] for i in index])
        b = v5.average_rank([own_rows[i][t] for i in index])
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        saa, sbb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
        if saa > 0 and sbb > 0:
            out.append(sab / math.sqrt(saa * sbb))
    return sum(out) / len(out) if out else None


# ------------------------------------------------------------------ time guard
def projection(n, n_fallback, policy=GUARD):
    """Seconds the own arms need from the post cell on: the slower of the two parallel arms plus a margin."""
    need(type(n) is int and n > 0 and type(n_fallback) is int and 0 <= n_fallback <= n, 'invalid counts')
    per_arm = {name: policy['fixed_seconds'] + policy['inflation'] * (
        per_study * n + policy['decode_seconds_per_study'] * (n_fallback + policy['audit_studies']))
        for name, per_study in policy['per_study_seconds'].items()}
    return max(per_arm.values()) + policy['margin_seconds'], per_arm


def guard(elapsed, n, n_fallback, policy=GUARD):
    projected, per_arm = projection(n, n_fallback, policy)
    finish = elapsed + projected
    run = bool(math.isfinite(elapsed) and 0 <= elapsed and finish <= policy['budget_seconds']
               and finish < policy['hard_deadline_seconds'])
    return {'run': run, 'elapsed_seconds': elapsed, 'studies': n, 'fallback_studies': n_fallback,
            'projected_seconds': projected, 'arm_projected_seconds': per_arm, 'projected_finish_seconds': finish,
            'budget_seconds': policy['budget_seconds'], 'hard_deadline_seconds': policy['hard_deadline_seconds'],
            'latest_start_seconds': policy['budget_seconds'] - projected}


# ------------------------------------------------------------------ worker orchestration
def install_env(v5, assets, opencv, envdir, runner=subprocess.run):
    """Pinned offline wheels (v5 hashes + the OpenCV hash of the EXP089 protocol) into one fresh --target directory."""
    envdir = Path(envdir)
    if envdir.exists():
        return {'installed': False, 'reason': 'existing environment directory reused'}
    protocol = v5.verify_old_assets(assets, opencv)
    wheels = [str(Path(assets) / w['path']) for w in protocol['wheels']] + [str(opencv)]
    runner([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--quiet', '--target', str(envdir),
            *wheels], check=True, timeout=900)
    return {'installed': True, 'wheels': [Path(w).name for w in wheels]}


def collect_arm(v5, arm_dir, arm, ids, exit_code):
    """Checked worker output -> ({uid: 12 probabilities}, {uid: reason}, receipt)."""
    name = arm['name']
    need(exit_code == 0, '%s: worker exit code %r' % (name, exit_code))
    receipt = json.loads((Path(arm_dir) / 'receipt.json').read_text(encoding='utf-8'))
    need(receipt.get('status') == 'PASS' and receipt.get('arm') == name, '%s: worker receipt not PASS' % name)
    need(receipt.get('ids_sha256') == ids_sha256(ids) and receipt.get('studies') == len(ids), '%s: cohort differs' % name)
    need(receipt.get('release_manifest_sha256') == RELEASE_MANIFEST_SHA256
         and receipt.get('new_assets_manifest_sha256') == ASSETS_MANIFEST_SHA256, '%s: release binding differs' % name)
    need([c.get('name') for c in receipt.get('checkpoints') or []] == [c['name'] for c in arm['checkpoints']]
         and [c.get('sha256') for c in receipt['checkpoints']] == [c['sha256'] for c in arm['checkpoints']]
         and receipt.get('k_eval') == K_EVAL and receipt.get('rank_weight') == arm['weight'],
         '%s: worker ensemble differs' % name)
    unscored = receipt.get('unscored')
    need(isinstance(unscored, dict) and set(unscored) <= set(ids), '%s: unscored list malformed' % name)
    path = Path(arm_dir) / 'predictions.csv'
    need(sha(path) == receipt.get('predictions_sha256'), '%s: predictions changed' % name)
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f))
    need(rows and rows[0] == [UID, *TARGETS] and all(len(r) == len(TARGETS) + 1 for r in rows[1:]),
         '%s: predictions schema differs' % name)
    need([r[0] for r in rows[1:]] == [u for u in ids if u not in unscored],
         '%s: a study is neither scored nor listed as unscored (or order differs)' % name)
    probs = {r[0]: probability_row(r[1:], name + ' ' + r[0]) for r in rows[1:]}
    return probs, unscored, receipt


def run_workers(commands, deadline, spawn, clock, sleep, poll_seconds, logs):
    """Start every worker, wait; at `deadline` (epoch seconds) kill them all. -> ({name: exit code}, timed_out)."""
    procs = {}
    try:
        for name, cmd, env in commands:
            procs[name] = spawn(cmd, env=env, stdout=logs[name], stderr=subprocess.STDOUT)
        while True:
            codes = {name: p.poll() for name, p in procs.items()}
            if all(c is not None for c in codes.values()):
                return codes, False
            if clock() >= deadline:
                break
            sleep(poll_seconds)
    except BaseException:
        for p in procs.values():
            try:
                p.kill()
            except Exception:
                pass
        raise
    for p in procs.values():
        try:
            p.kill()
        except Exception:
            pass
    for p in procs.values():
        try:
            p.wait(timeout=60)
        except Exception:
            pass
    return {name: p.poll() for name, p in procs.items()}, True


def coat_family_record(ns, working):
    """The anchor's CoAt family state (recorded only; the anchor already published its own output)."""
    names = ['resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3']
    words = ('child_failed', 'output_rejected', 'probability_unavailable', 'family_partial', 'family_unavailable')
    audit = ns.get('_RSNA_AUDIT') or {}
    events = list(audit.get('events') or [])
    try:
        blend = json.loads((Path(working) / '_coat_raptor_blend_receipt.json').read_text())
    except Exception as exc:
        blend = {'error': '%s: %s' % (type(exc).__name__, exc)}
    failures = [e for e in events if isinstance(e, dict) and str(e.get('kind', '')).startswith('coat_')
                and any(w in str(e.get('kind')) for w in words)]
    out = {'members': blend.get('members'), 'family_reduction': blend.get('family_reduction'),
           'audit_members': audit.get('coat_family_members'), 'audit_status': audit.get('status'),
           'failure_events': failures[:20], 'anchor_elapsed_seconds': audit.get('elapsed_seconds')}
    out['ok'] = bool(out['members'] == names and out['audit_members'] == 4 and not failures)
    return out


def phase_seconds(working):
    lines = (Path(working) / 'diagnostics' / 'phase_events.jsonl').read_text().splitlines()
    marks = {}
    for line in lines:
        if line.strip():
            p = json.loads(line)
            marks[(p.get('stage'), p.get('status'))] = p.get('elapsed_seconds')
    return {stage: round(marks[(stage, 'COMPLETE')] - marks[(stage, 'START')], 1)
            for stage, status in marks if status == 'START' and (stage, 'COMPLETE') in marks}


def verify_store(store_dir, protocol_path, ids):
    """eng100 only: stored bytes of every OK row against the EXP089 input_reference hashes (recorded, never gating)."""
    protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    refs = {r['uid']: r for r in protocol['input_reference']}
    records = {}
    for line in (Path(store_dir) / 'records.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[r.get('uid')] = r
    ok = [u for u in ids if (records.get(u) or {}).get('status') == 'OK']
    exact = [u for u in ok if records[u].get('volume_sha256') == (refs.get(u) or {}).get('volume_sha256')
             and records[u].get('mask_sha256') == (refs.get(u) or {}).get('mask_sha256')]
    return {'studies': len(ids), 'ok': len(ok), 'exact_vs_reference': len(exact),
            'ok_but_different': [u for u in ok if u not in exact][:20],
            'not_ok': {u: (records.get(u) or {}).get('status', 'EMPTY') for u in ids if u not in ok}}


def restore_anchor(final, backup, anchor_sha):
    tmp = Path(final).with_name(Path(final).name + '.exp105-restore')
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(backup, tmp)
    need(sha(tmp) == anchor_sha, 'backup differs from the anchor submission')
    os.replace(tmp, final)
    need(sha(final) == anchor_sha, 'restore failed')


def publish_blend(v5, final, ids, cols, anchor_sha, backup):
    """tmp -> re-read checks -> anchor bytes unchanged -> os.replace -> re-check (a failure restores the anchor)."""
    final = Path(final)
    tmp = final.with_name(final.name + TMP_SUFFIX)
    if tmp.exists():
        tmp.unlink()
    try:
        write_csv(v5, tmp, ids, cols)
        blend_sha = verify_csv(tmp, ids, cols)
        need(sha(final) == anchor_sha, 'submission.csv changed after the anchor published it')
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    try:
        need(verify_csv(final, ids, cols) == blend_sha, 'published file differs from the checked blend')
    except BaseException:
        restore_anchor(final, backup, anchor_sha)
        raise
    return blend_sha


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, path)


def post_anchor(ns, environ=None, spawn=None, runner=None, clock=None, sleep=None, policy=None):
    """Post-anchor step (see the module docstring). Never raises; returns the receipt."""
    environ = os.environ if environ is None else environ
    spawn = spawn or subprocess.Popen
    runner = runner or subprocess.run
    clock = clock or time.time
    sleep = sleep or time.sleep
    policy = policy or GUARD
    t_post = clock()
    working = Path(environ.get('HB15_WORKING', '/kaggle/working'))
    out = working / 'exp105'
    final = working / 'submission.csv'
    backup = out / 'anchor_submission.csv'
    receipt = {'schema': SCHEMA, 'path': 'anchor_only', 'reason': None, 'final_weights': list(FINAL_WEIGHTS),
               'own_weights': {n: w for n, _, w in OWN_ARMS}, 'k_eval': K_EVAL, 'guard_policy': policy,
               'timings': {'post_start_epoch': t_post}, 'counts': {}}
    anchor_sha = None
    v5 = None
    try:
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise AnchorOnly('cannot create %s: %s' % (out, exc))
        setup = ns.get('_HB15_SETUP') or {}
        receipt['setup'] = {k: setup.get(k) for k in ('ok', 'error', 'variant', 't0')}
        # 1. hook finalize (render thread stopped, store flushed); never trusted unless the thread finished
        hook = ns.get('_HB15') or {}
        store_ok, hook_ok = 0, False
        if callable(hook.get('finalize')):
            try:
                summary = hook['finalize'](float(policy['finalize_timeout_seconds']))
                receipt['hook'] = summary
                hook_ok = bool(summary.get('thread_finished'))
                store_ok = int(((summary.get('store') or {}).get('status_counts') or {}).get('OK', 0)) if hook_ok else 0
            except Exception as exc:
                receipt['hook'] = {'finalize_error': '%s: %s' % (type(exc).__name__, exc)}
        else:
            receipt['hook'] = {'enabled': False, 'error': hook.get('error', 'hook absent')}
        try:
            receipt['anchor'] = coat_family_record(ns, working)
            receipt['phase_seconds'] = phase_seconds(working)
        except Exception as exc:
            receipt.setdefault('anchor', {})['record_error'] = '%s: %s' % (type(exc).__name__, exc)
        # 2. the anchor's published submission
        try:
            comp = Path(environ.get('RSNA_COMP_ROOT') or '/kaggle/input/competitions/rsna-knee-abnormality-detection')
            ids = read_sample(comp)
            anchor_rows = read_anchor(final, ids)
            anchor_sha = sha(final)
            shutil.copyfile(final, backup)
            need(sha(backup) == anchor_sha, 'anchor backup differs')
        except Exception as exc:
            anchor_sha = None
            raise AnchorOnly('anchor submission not usable (left untouched): %s: %s' % (type(exc).__name__, exc))
        receipt.update(anchor_sha256=anchor_sha, studies=len(ids))
        test_ids = ns.get('_RSNA_TEST_IDS')
        if test_ids is not None and set(map(str, test_ids)) != set(ids):
            raise AnchorOnly('anchor cohort (_RSNA_TEST_IDS) differs from sample_submission')
        if not setup.get('ok'):
            raise AnchorOnly('setup cell failed: %s' % setup.get('error'))
        missing = [k for k in ('HB15_SRC', 'HB15_ASSETS', 'HB15_NEW_ASSETS', 'HB15_OPENCV', 'HB15_STORE', 'HB15_ENV',
                               'HB15_RELEASE_MANIFEST') if not environ.get(k)]
        if missing:
            raise AnchorOnly('setup environment incomplete: %s' % missing)
        if callable(hook.get('finalize')) and not hook_ok:
            raise AnchorOnly('hook render thread did not finish')
        # 3. time guard
        t0 = float(setup.get('t0') if setup.get('t0') is not None else environ['HB15_T0'])
        n = len(ids)
        decision = guard(clock() - t0, n, n - min(store_ok, n), policy)
        receipt['guard'] = decision
        receipt['counts'].update(store_ok=store_ok, fallback_studies=n - min(store_ok, n))
        if not decision['run']:
            raise AnchorOnly('time guard: elapsed %.0f s + projected %.0f s > budget %.0f s'
                             % (decision['elapsed_seconds'], decision['projected_seconds'], policy['budget_seconds']))
        # 4. environment and workers
        src = Path(environ['HB15_SRC'])
        v5 = load_v5(src)
        release = v5.checked_manifest(environ['HB15_RELEASE_MANIFEST'], RELEASE_MANIFEST_SHA256)
        arms = check_release(release)
        t = clock()
        receipt['env'] = install_env(v5, environ['HB15_ASSETS'], environ['HB15_OPENCV'], environ['HB15_ENV'], runner)
        receipt['timings']['env_seconds'] = clock() - t
        try:
            torch = sys.modules.get('torch')   # imported by the anchor: release its cached GPU blocks
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception:
            pass
        (out / 'ids.json').write_text(json.dumps(ids), encoding='utf-8')
        rows_csv = environ.get('HB15_ROWS_CSV') or str(comp / 'test_series.csv')
        dicom_root = environ.get('HB15_DICOM_ROOT') or str(comp / 'test_series')
        device, amp = environ.get('HB15_DEVICE', 'cuda'), environ.get('HB15_AMP', 'fp16')
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
        share = max(1, len(cpus) // len(OWN_ARMS))
        hard_deadline = t0 + policy['hard_deadline_seconds']
        commands, logs = [], {}
        for index, (name, gpu, _) in enumerate(OWN_ARMS):
            mine = cpus[index * share:(index + 1) * share] or cpus
            threads = environ.get('HB15_THREADS') or str(min(4, len(mine)))
            cmd = [sys.executable, str(src / WORKER), '--assets', environ['HB15_ASSETS'],
                   '--new-assets', environ['HB15_NEW_ASSETS'], '--release-manifest', environ['HB15_RELEASE_MANIFEST'],
                   '--src', str(src), '--env', environ['HB15_ENV'], '--store', environ['HB15_STORE'],
                   '--ids-json', str(out / 'ids.json'), '--rows-csv', rows_csv, '--dicom-root', dicom_root,
                   '--split', environ.get('HB15_SPLIT', 'test'), '--arm', name, '--out', str(out / name),
                   '--threads', threads, '--cpus', ','.join(map(str, mine)), '--audit', environ.get('HB15_AUDIT', '2'),
                   '--device', device, '--amp', amp, '--deadline', repr(hard_deadline - 60.0)]
            child = dict(environ, CUDA_VISIBLE_DEVICES=str(gpu) if device == 'cuda' else '', PYTHONUTF8='1',
                         PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
            commands.append((name, cmd, child))
            logs[name] = (out / (name + '.log')).open('w', encoding='utf-8')
        t = clock()
        try:
            codes, timed_out = run_workers(commands, hard_deadline, spawn, clock, sleep, policy['poll_seconds'], logs)
        finally:
            for handle in logs.values():
                handle.close()
        receipt['timings']['workers_seconds'] = clock() - t
        receipt['worker_exit_codes'] = codes
        if timed_out:
            raise AnchorOnly('own arms timed out at first cell + %.0f s' % policy['hard_deadline_seconds'])
        # 5. checked outputs
        tables, unscored, workers = {}, {}, {}
        for arm in arms:
            try:
                probs, miss, r = collect_arm(v5, out / arm['name'], arm, ids, codes.get(arm['name']))
            except Exception as exc:
                detail = ''
                try:
                    r = json.loads((out / arm['name'] / 'receipt.json').read_text(encoding='utf-8'))
                    detail = ' (worker: %s %s)' % (r.get('status'), json.dumps(r.get('error'))[:500])
                except Exception:
                    pass
                raise AnchorOnly('%s: %s%s' % (type(exc).__name__, exc, detail))
            tables[arm['name']], unscored[arm['name']] = probs, miss
            timing = r.get('timing') or {}
            workers[arm['name']] = {k: r.get(k) for k in ('worker_wall_seconds', 'model_load_seconds', 'verify_seconds',
                                                          'versions', 'gpu', 'served', 'audits', 'store_error')}
            workers[arm['name']].update(per_study_s=timing.get('per_study_s'), predict_seconds=timing.get('seconds'),
                                        scored=len(probs), unscored=len(miss))
        receipt['workers'] = workers
        receipt['unscored'] = unscored
        if environ.get('HB15_VERIFY_PROTOCOL'):
            try:
                receipt['verification'] = {'store': verify_store(environ['HB15_STORE'], environ['HB15_VERIFY_PROTOCOL'], ids)}
            except Exception as exc:
                receipt['verification'] = {'error': '%s: %s' % (type(exc).__name__, exc)}
        # 6. blend
        t = clock()
        own = own_blend(v5, ids, tables['coat5'], tables['rn5'], [w for _, _, w in OWN_ARMS])
        n_unscored = n - len(own)
        receipt['counts'].update(scored_both=len(own), unscored_any=n_unscored)
        if not own:
            raise AnchorOnly('no study scored by both arms')
        if n_unscored > math.floor(policy['max_unscored_fraction'] * n):
            raise AnchorOnly('%d of %d studies unscored (> %.0f%%)' % (n_unscored, n, 100 * policy['max_unscored_fraction']))
        cols = final_blend(v5, anchor_rows, own, FINAL_WEIGHTS)
        receipt['anchor_own_rank_correlation'] = rank_correlation(v5, anchor_rows, own)
        anchor_ranks = [v5.average_rank([row[t] for row in anchor_rows]) for t in range(len(TARGETS))]
        receipt['max_abs_change_vs_anchor_rank'] = max(abs(cols[t][i] - anchor_ranks[t][i])
                                                       for t in range(len(TARGETS)) for i in range(n))
        # 7. publish
        blend_sha = publish_blend(v5, final, ids, cols, anchor_sha, backup)
        receipt.update(path='blended', blended_sha256=blend_sha)
        receipt['timings']['blend_publish_seconds'] = clock() - t
    except AnchorOnly as exc:
        receipt['reason'] = str(exc)[:2000]
    except BaseException as exc:
        receipt['reason'] = ('unexpected %s: %s' % (type(exc).__name__, exc))[:2000]
    finally:
        try:
            if receipt['path'] != 'blended' and anchor_sha is not None and final.is_file() and sha(final) != anchor_sha:
                restore_anchor(final, backup, anchor_sha)
                receipt['restored_anchor'] = True
        except BaseException as exc:
            receipt['restore_error'] = '%s: %s' % (type(exc).__name__, exc)
        try:
            receipt['final_sha256'] = sha(final) if final.is_file() else None
            receipt['final_is_anchor'] = receipt['final_sha256'] == anchor_sha
        except BaseException:
            pass
        receipt['timings']['post_seconds'] = clock() - t_post
        try:
            t0 = (ns.get('_HB15_SETUP') or {}).get('t0')
            receipt['timings']['elapsed_since_first_cell'] = clock() - t0 if t0 is not None else None
        except BaseException:
            pass
        try:
            out.mkdir(parents=True, exist_ok=True)
            _write_json(out / 'receipt.json', receipt)
        except BaseException:
            pass
    return receipt


# ------------------------------------------------------------------ notebook build (local)
SETUP_TEMPLATE = r'''# EXP105 setup cell (H-B-15), before the anchor cells. Standard library only; it never raises.
# Records the first-cell clock, writes our embedded sources (each sha256-checked) to /kaggle/working/hb15/src,
# verifies the asset bindings (EXP089 serving code and wheels, release-assets-v5 checkpoints, the OpenCV wheel) and
# sets the HB15_* environment read by the hook cell and the post-anchor cell. If anything fails no HB15_* variable is
# set: the hook stays disabled, the anchor runs unchanged and its own submission.csv is kept.
import time as _hb15_time
_HB15_T0 = _hb15_time.time()
_HB15_KAGGLE = '/kaggle'
_HB15_SETUP = {'t0': _HB15_T0, 'ok': False, 'error': None, 'variant': @@VARIANT@@}
@@SOURCES@@


def _hb15_setup():
    import hashlib, importlib.util, os
    from pathlib import Path
    src = Path(_HB15_KAGGLE) / 'working' / 'hb15' / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for name, (digest, lines) in _HB15_SOURCES.items():
        data = ('\n'.join(lines) + '\n').encode('utf-8')
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError('embedded source differs from its sha256: ' + name)
        (src / name).write_bytes(data)
    _HB15_SETUP['src'] = str(src)
    spec = importlib.util.spec_from_file_location('exp105_combined_release', src / 'exp105_combined_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.setup_environment(src, @@VARIANT@@, _HB15_KAGGLE@@BENCH@@)
    env['HB15_T0'] = repr(_HB15_T0)
    os.environ.update(env)
    return env


try:
    _HB15_SETUP['env'] = _hb15_setup()
    _HB15_SETUP['ok'] = True
except BaseException as _hb15_setup_exc:
    _HB15_SETUP['error'] = (type(_hb15_setup_exc).__name__ + ': ' + str(_hb15_setup_exc))[:1000]
print('[exp105] setup ' + ('OK' if _HB15_SETUP['ok'] else 'FAILED, anchor-only: ' + str(_HB15_SETUP['error'])),
      flush=True)
'''

POST_CELL = r'''# EXP105 post-anchor cell (H-B-15), after the anchor's last cell published submission.csv. It never raises.
# The anchor's submission stays unless our two 5-fold arms ran inside the time guard and every check passed; then
# submission.csv is atomically replaced by the per-target rank blend 0.8 anchor / 0.2 own (studies our arms could not
# score keep the anchor's rank). Evidence: /kaggle/working/exp105/receipt.json (path anchor_only|blended, reason).
def _exp105_post():
    import importlib.util, os
    from pathlib import Path
    src = Path(os.environ.get('HB15_SRC') or globals()['_HB15_SETUP']['src'])
    spec = importlib.util.spec_from_file_location('exp105_combined_release', src / 'exp105_combined_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.post_anchor(globals())


try:
    _EXP105 = _exp105_post()
except BaseException as _exp105_exc:
    _EXP105 = {'path': 'anchor_only', 'reason': ('post cell: ' + type(_exp105_exc).__name__ + ': '
                                                  + str(_exp105_exc))[:1000]}
    try:
        import json as _exp105_json, os as _exp105_os
        _exp105_dir = _exp105_os.path.join(_exp105_os.environ.get('HB15_WORKING', '/kaggle/working'), 'exp105')
        _exp105_os.makedirs(_exp105_dir, exist_ok=True)
        with open(_exp105_os.path.join(_exp105_dir, 'receipt-fallback.json'), 'w') as _exp105_handle:
            _exp105_json.dump(_EXP105, _exp105_handle, indent=1)
    except BaseException:
        pass
try:
    import json as _exp105_json
    print('[exp105] path=' + str(_EXP105.get('path')) + ' reason=' + str(_EXP105.get('reason')), flush=True)
    print('[exp105] ' + _exp105_json.dumps({_k: _EXP105.get(_k) for _k in ('counts', 'guard', 'timings', 'workers',
                                                                           'final_is_anchor')}, default=str)[:6000],
          flush=True)
except BaseException:
    pass
'''


def code_cell(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': source.splitlines(keepends=True)}


def cell_text(cell):
    return ''.join(cell['source'])


def embedded_sources():
    out, digests = ['_HB15_SOURCES = {'], {}
    for name, (path, pinned) in SOURCES.items():
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        need(pinned is None or digest == pinned, '%s differs from its pinned sha256' % name)
        need(b'\r' not in data and data.endswith(b'\n'), '%s: LF-only source ending in a newline required' % name)
        lines = data.decode('utf-8')[:-1].split('\n')
        need(('\n'.join(lines) + '\n').encode('utf-8') == data, '%s: line round trip' % name)
        digests[name] = digest
        out.append(' %r: (%r, [' % (name, digest))
        out.extend('  %r,' % line for line in lines)
        out.append(' ]),')
    out.append('}')
    return '\n'.join(out), digests


def setup_cell(variant):
    sources, digests = embedded_sources()
    bench = ", bench_comp=_bench_comp, bench_ids=_BENCH_CONTRACT['ids']" if variant == 'eng100' else ''
    values = {'VARIANT': repr(variant), 'SOURCES': sources, 'BENCH': bench}
    need(sorted(set(re.findall(r'@@([A-Z_]+)@@', SETUP_TEMPLATE))) == sorted(values), 'template placeholders differ')
    # one pass: inserted text (the embedded sources contain @@...@@ themselves) is never rescanned
    text = re.sub(r'@@([A-Z_]+)@@', lambda m: values[m.group(1)], SETUP_TEMPLATE)
    return text, digests


def split_anchor(cells):
    marker = SPLIT_MARKER[0] + SPLIT_MARKER[1]
    hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and marker in cell_text(c)]
    need(len(hits) == 1, 'exactly one anchor cell holds A:4233-4234')
    whole = cell_text(cells[hits[0]])
    need(whole.count(marker) == 1, 'split marker must be unique')
    cut = whole.index(marker) + len(SPLIT_MARKER[0])
    first, second = copy.deepcopy(cells[hits[0]]), copy.deepcopy(cells[hits[0]])
    first['source'] = whole[:cut].splitlines(keepends=True)
    second['source'] = whole[cut:].splitlines(keepends=True)
    need(cell_text(first) + cell_text(second) == whole, 'split halves do not reproduce the anchor cell')
    return hits[0], first, second


def pretty_proxy(runtime, contract):
    lines = cell_text(runtime['cells'][0]).split('\n')
    head = '_BENCH_CONTRACT = '
    index = [i for i, line in enumerate(lines) if line.startswith(head)]
    need(len(index) == 1, 'one contract line expected in the runtime100 proxy cell')
    value = ast.literal_eval(lines[index[0]][len(head):])
    need(value == contract, 'runtime100 contract literal differs from selection-contract.json')
    pretty = head + pprint.pformat(value, width=110, sort_dicts=False)
    need(ast.literal_eval(pretty[len(head):]) == value, 'pretty literal round trip')
    lines[index[0]] = pretty
    return '\n'.join(lines)


def build_notebook(variant):
    """-> (notebook dict, kernel metadata dict, receipt dict); deterministic, local files only."""
    need(variant in KERNELS, 'unknown variant')
    need(sha(ANCHOR_NB) == ANCHOR_SHA256, 'anchor notebook hash differs')
    need(sha(HOOK) == HOOK_SHA256, 'hook cell hash differs from the verified hook')
    anchor = json.loads(ANCHOR_NB.read_text(encoding='utf-8'))
    split_at, first, second = split_anchor(anchor['cells'])
    hook = HOOK.read_text(encoding='utf-8')
    setup, digests = setup_cell(variant)
    body = (copy.deepcopy(anchor['cells'][:split_at]) + [first, code_cell(hook), second]
            + copy.deepcopy(anchor['cells'][split_at + 1:]))
    added = {'setup': setup, 'hook': hook, 'post': POST_CELL}
    if variant == 'full':
        notebook = copy.deepcopy(anchor)
        notebook['cells'] = [code_cell(setup)] + body + [code_cell(POST_CELL)]
        metadata = json.loads(ANCHOR_META.read_text(encoding='utf-8'))
        order = ['EXP105 setup', 'anchor 0..%d' % (split_at - 1), 'anchor %da' % split_at, 'HB15 hook (verified)',
                 'anchor %db' % split_at, 'anchor %d..%d' % (split_at + 1, len(anchor['cells']) - 1), 'EXP105 post']
    else:
        runtime_nb = RUNTIME100 / 'anchor-runtime100.ipynb'
        need(sha(runtime_nb) == RUNTIME100_SHA256, 'runtime100 notebook hash differs')
        runtime = json.loads(runtime_nb.read_text(encoding='utf-8'))
        need(runtime['cells'][1:-1] == anchor['cells'], 'runtime100 anchor cells differ from the anchor package')
        contract = json.loads((RUNTIME100 / 'selection-contract.json').read_text(encoding='utf-8'))
        proxy = pretty_proxy(runtime, contract)
        added['proxy'] = proxy
        notebook = copy.deepcopy(runtime)
        notebook['cells'] = ([code_cell(proxy), code_cell(setup)] + body + [code_cell(POST_CELL)]
                             + [copy.deepcopy(runtime['cells'][-1])])
        metadata = json.loads((RUNTIME100 / 'kernel-metadata.json').read_text(encoding='utf-8'))
        order = ['runtime100 proxy (100 engineering train studies, IDs one per line)', 'EXP105 setup (eng100)',
                 'anchor 0..%d' % (split_at - 1), 'anchor %da' % split_at, 'HB15 hook (verified)',
                 'anchor %db' % split_at, 'anchor %d..%d' % (split_at + 1, len(anchor['cells']) - 1), 'EXP105 post',
                 'runtime100 final check']
    notebook.setdefault('metadata', {})['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                                                         'name': 'python3'}
    for name, src in added.items():
        compile(src, name, 'exec')
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(cell_text(cell), 'cell%d' % i, 'exec')
    widths = {name: max(len(line) for line in src.splitlines()) for name, src in added.items()}
    need(max(widths.values()) < MAX_ADDED_LINE, 'added line too long: %s' % widths)
    longest = max(len(line) for c in notebook['cells'] for line in cell_text(c).splitlines())
    need(longest < MAX_NOTEBOOK_LINE, 'notebook line too long for Kaggle: %d' % longest)
    kernel = KERNELS[variant]
    slug = kernel.split('/')[1]
    metadata.update(id=kernel, title=slug, code_file=slug + '.ipynb', is_private=True, enable_gpu=True,
                    enable_tpu=False, enable_internet=False, docker_image=DOCKER_IMAGE, machine_shape='NvidiaTeslaT4')
    metadata['dataset_sources'] = [d for d in metadata['dataset_sources'] if d not in (OLD_DATASET, NEW_DATASET)] + [
        OLD_DATASET, NEW_DATASET]
    need(OPENCV_DATASET in metadata['dataset_sources'], 'the anchor OpenCV dataset must be mounted')
    receipt = {'variant': variant, 'kernel': kernel, 'cells': len(notebook['cells']), 'cell_order': order,
               'split_anchor_cell': split_at, 'hook_inserted_between': 'A:4233 | A:4234',
               'embedded_sources_sha256': digests, 'hook_sha256': HOOK_SHA256,
               'setup_sha256': hashlib.sha256(setup.encode()).hexdigest(),
               'post_sha256': hashlib.sha256(POST_CELL.encode()).hexdigest(),
               'added_max_line': widths, 'notebook_max_line': longest, 'anchor_notebook_sha256': ANCHOR_SHA256,
               'docker_image': DOCKER_IMAGE, 'datasets': metadata['dataset_sources'],
               'kernel_sources': metadata['kernel_sources'], 'model_sources': metadata['model_sources'],
               'competition_sources': metadata['competition_sources'], 'final_weights': list(FINAL_WEIGHTS),
               'own_arms': [list(a) for a in OWN_ARMS], 'k_eval': K_EVAL, 'guard_policy': GUARD}
    if variant == 'eng100':
        receipt['runtime100_notebook_sha256'] = RUNTIME100_SHA256
    return notebook, metadata, receipt


def build(a):
    out = Path(a.out)
    need(not out.exists(), 'refuse to overwrite %s' % out)
    notebook, metadata, receipt = build_notebook(a.variant)
    out.mkdir(parents=True)
    nb_path = out / metadata['code_file']
    nb_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8', newline='\n')
    py = str(ROOT / '.venv/Scripts/python.exe')
    receipt.update(state='STAGED_ONLY_NOT_PUSHED', notebook_sha256=sha(nb_path),
                   metadata_sha256=sha(out / 'kernel-metadata.json'),
                   push_argv_not_run=[py, '-m', 'kaggle', 'kernels', 'push', '-p', str(out.resolve()),
                                      '--accelerator', 'NvidiaTeslaT4'],
                   push_note='no --timeout (D-B-015 item 13); set PYTHONIOENCODING=utf-8')
    if a.variant == 'full':
        receipt['submit_argv_template_not_run'] = [py, '-m', 'kaggle', 'competitions', 'submit',
                                                   'rsna-knee-abnormality-detection', '-k', metadata['id'], '-v',
                                                   '<VERSION>', '-f', 'submission.csv', '-m', '<message>']
    (out.parent / (out.name + '.build.json')).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('kernel', 'cells', 'notebook_sha256', 'notebook_max_line')}))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build')
    b.add_argument('--variant', choices=sorted(KERNELS), required=True)
    b.add_argument('--out', required=True)
    a = parser.parse_args(argv)
    {'build': build}[a.action](a)


if __name__ == '__main__':
    main()
