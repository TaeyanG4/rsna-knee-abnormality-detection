"""EXP119 test-only release v7 = shared-decode multi-arm own stack (H-B-32 stage 1; serving runtime is the critical path).

One supervised worker decodes every test study ONCE into the RAPTOR64_336_C140_S02 volume -- the frozen EXP089
raptor_dicom.DicomCorpus, i.e. the very decode/render of every v4-v6 worker -- and feeds that one read-only volume to every
arm of the release: CoAtNet-family 'triplet' arms (cache planes c-1, c, c+1 at their K, ImageNet normalisation) and
RadImageNet ResNet50 'center3' arms (centre plane x3 at their K, 2x-1 normalisation). Each arm is a list of fold
checkpoints averaged by the v5 mean_prob wrapper; arms are combined by the v5 per-target average-tie percentile rank mean
with optional fixed weights (equal weights = the EXP098 equal mean, bit for bit).

  # 1. only for checkpoints that are not yet in a published assets dataset: stage a NEW package (explicit slug + dir)
  python tools/exp119_release_v7.py stage-assets --copy --dataset taeyangg4/knee-release-assets-v11 \
      --package kaggle/packages/KNEE_RELEASE_ASSETS_V11 --checkpoint rad43_f0=<run>/model_best.pt ...
  # 2. generate a Kaggle kernel directory (never pushed here); each --assets DATASET=PACKAGE is mounted read-only
  python tools/exp119_release_v7.py notebook --release coat5rad5 \
      --assets taeyangg4/knee-release-assets-v8=kaggle/packages/KNEE_RELEASE_ASSETS_V8 \
      --arm coat5=coat_f0+coat_f1+coat_f2+coat_f3+coat_f4@36:1 --arm rad5=rad_f0+rad_f1+rad_f2+rad_f3+rad_f4@36:1 \
      --out .local/exp119-release-v7/coat5rad5
  # 3. runtime projections from the v5/v6 constants (local arithmetic only)
  python tools/exp119_release_v7.py project --studies 1300 --out .local/exp119-release-v7/runtime-projection-v1.json

Reuse by import. The reviewed v6 tool (tools/exp111_release_v6.py, which carries every v5 function verbatim) is loaded
from this tool's directory, pinned by sha256 (V6_SOURCE_SHA256), and embedded next to this tool in the kernel. Used
unchanged: arm spec parsing, arm building/validation (input policies, member metadata), checkpoint_check, the mean_prob
wrapper (a non-finite member logit fails closed), the centre-plane input swap (serving_input/center3_prepare_input),
frozen-asset and input-code verification, the test-cohort contract, CSV checks, resource-receipt validation,
launch_budget, atomic publication, rank_mean/write_columns, stage_assets, the TIME policy and the docker pin.

Changes against v6 (everything else keeps v6 semantics):
  A. One worker for the whole release: per study, decode once (DicomCorpus.volume/mask), check the bytes against the
     DicomCorpus receipt (volume_sha256/mask_sha256), wrap them in a read-only one-study view (SharedStudy: no source, so
     nothing can decode again) and run every arm through the unchanged frozen raptor_runtime.predict (batch 1, the
     same numerical path as v5/v6's one call over all studies; the v6 input swap per centre-plane arm). The volume hash
     is re-checked after the last arm. The receipt records, per arm, the predict-format timing (input hashes and the
     ordered aggregate, decode excluded) and, once, the decode trace (decodes == studies, per-study volume/mask hashes).
  B. Time screen with ONE decode: inflation*(decode_seconds_per_study + sum over arms of n_models*model_seconds_per_study)
     *N + screen_fixed_seconds <= screen_seconds (v5/v6: one decode per arm). The launch guard (v6 launch_budget,
     deadline anchored at the first-cell clock) needs that projection minus screen_fixed plus one arm_fixed.
  C. Several assets datasets may be mounted (v5-schema 'knee-release-assets-v5' or v6-schema 'exp111-release-assets-v6');
     each arm is bound to the one dataset that stages all its members, so already published 5-fold arms (v8 coat5/rad5,
     v9 flight5, v7 relabel5) need no new upload. A centre-plane arm needs a v6-schema dataset (it ships the input code).
     Every mounted dataset must serve an arm; a checkpoint name must be unique across the mounted datasets and serves
     one arm only; all members of all arms stay resident on the one T4 (<= MAX_RESIDENT_CHECKPOINT_BYTES).
  D. Every arm must read the shared raptor64_336_c140_s02 volume; arm names 'combine'/'shared' are reserved.
  E. Layout: <base>/shared/work/{launch.json, worker/{receipt.json, <arm>.csv}, supervisor/, evidence.zip, result.json};
     one arm -> /kaggle/working/submission.csv directly; two or more -> <base>/<arm>/arm.csv, then `combine`.
  F. stage-assets requires an explicit --dataset/--package and refuses every published release-assets slug and package
     (v4-v10); it then runs the v6 stage_assets (v6 assets schema, centre-plane code shipped).
  G. Kernel taeyangg4/knee-exp119-<release>-v7; the recorded push uses tools/kaggle_push.py (waits for "ready" datasets)
     or the plain CLI with --accelerator NvidiaTeslaT4, never --timeout (D-B-015 item 13, D-B-039).

stage-assets/notebook/project read local files only: they never push kernels, publish datasets or submit.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V6_NAME = 'exp111_release_v6.py'
V6_SOURCE_SHA256 = '4e9a74445f5c558cffa56e16f4cd22d7676e444f4a304e3ab30d7badbbd1b36b'   # reviewed v6 (EXP111) source


def _load_v6(directory=HERE):
    path = Path(directory) / V6_NAME
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != V6_SOURCE_SHA256:
        raise ValueError('exp111_release_v6.py differs from the reviewed v6 source: ' + digest)
    spec = importlib.util.spec_from_file_location('exp111_release_v6_for_exp119', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v6 = _load_v6()
need, read, sha, write = v6.need, v6.read, v6.sha, v6.write

SCHEMA = 'exp119-test-release-v7'
TOOL_NAME = 'exp119_release_v7.py'
UID, TARGETS, TIME = v6.UID, v6.TARGETS, v6.TIME
DECODE_CONTRACT = 'raptor64_336_c140_s02'
DECODE_POLICY = {'contract': DECODE_CONTRACT, 'decodes_per_study': 1, 'shared_by': 'every arm',
                 'reader': 'frozen EXP089 raptor_dicom.DicomCorpus (prepare_study), one-study cache',
                 'view': 'exp119 SharedStudy (read-only arrays of the one decode, no source)'}
KERNEL_PREFIX, KERNEL_SUFFIX, WORK_PREFIX = 'knee-exp119-', '-v7', 'exp119-'
SHARED_DIR = 'shared'
RESERVED_ARM_NAMES = ('combine', SHARED_DIR)
ASSET_SCHEMAS = {'knee-release-assets-v5': False, v6.ASSETS_SCHEMA: True}   # schema -> ships the centre-plane input code
ASSET_SET_KEYS = ('dataset', 'schema', 'manifest_sha256', 'input_code')
DEFAULT_ASSETS = ['taeyangg4/knee-release-assets-v8=kaggle/packages/KNEE_RELEASE_ASSETS_V8']   # coat_f0-4 + rad_f0-4
# Published (or claimed by other releases) assets datasets and packages: read-only for v7, never (re)staged.
PUBLISHED_ASSET_DATASETS = tuple('taeyangg4/knee-release-assets-v%d' % v for v in range(4, 11))
PUBLISHED_ASSET_PACKAGES = tuple(ROOT / ('kaggle/packages/KNEE_RELEASE_ASSETS_V%d' % v) for v in range(4, 11))
MAX_RESIDENT_CHECKPOINT_BYTES = 10 * 2**30   # every member of every arm stays on the one 16 GB T4
MAX_KERNEL_SLUG = 50
OUT_DIR = ROOT / '.local/exp119-release-v7'
INPUT_FAILURE = 'input preparation failed for '   # the frozen predict's wording, kept for decode failures
COAT = 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'


# ------------------------------------------------------------------ assets datasets (several, read-only)
def check_asset_dataset(dataset):
    """A mounted assets dataset slug: owner/slug, never the frozen EXP089 assets or the OpenCV wheel dataset."""
    need(isinstance(dataset, str) and v6.DATASET_SLUG.fullmatch(dataset)
         and dataset not in (v6.OLD_DATASET, v6.OPENCV_DATASET), 'Unsafe assets dataset slug: ' + repr(dataset))
    return dataset


def parse_pairs(texts, what):
    """['owner/slug=VALUE', ...] -> ordered {dataset: VALUE}; datasets unique, at least one."""
    out = {}
    for text in texts or []:
        dataset, sep, value = (text or '').partition('=')
        need(sep == '=' and bool(value), what + ' must be DATASET=PATH: ' + repr(text))
        check_asset_dataset(dataset)
        need(dataset not in out, 'Duplicate assets dataset: ' + dataset)
        out[dataset] = value
    need(bool(out), 'At least one assets dataset required')
    return out


def local_package(package):
    package = Path(package)
    return package if package.is_absolute() else ROOT / package


def local_asset_set(dataset, package):
    """Local (notebook-time) check of one staged package -> (manifest record, assets manifest, resolved package)."""
    package = local_package(package)
    am_path = package / v6.NEW_MANIFEST
    need(am_path.is_file(), 'Missing assets manifest: ' + str(am_path))
    am = read(am_path)
    need(am.get('schema') in ASSET_SCHEMAS and am.get('dataset') == dataset,
         'Assets manifest schema/dataset differs: ' + dataset)
    need(v6.package_files(package) == am['files'], 'Package files differ from the assets manifest: ' + dataset)
    md = read(package / 'dataset-metadata.json')
    need(md.get('id') == dataset and md.get('isPrivate') is True, 'dataset-metadata.json differs: ' + dataset)
    if ASSET_SCHEMAS[am['schema']]:
        v6.verify_code(package, am)
    record = {'dataset': dataset, 'schema': am['schema'], 'manifest_sha256': sha(am_path),
              'input_code': ASSET_SCHEMAS[am['schema']]}
    return record, am, str(package.resolve())


def verify_asset_sets(release, roots, arms):
    """Bind every mounted assets dataset (v6 verify_new_assets semantics per dataset): manifest digest, schema, slug,
    shipped input code, and the presence and sha256 of every served member file."""
    sets = {s['dataset']: s for s in release['asset_sets']}
    need(set(roots) == set(sets), 'Mounted assets datasets differ from the release')
    manifests = {}
    for dataset, record in sets.items():
        root = Path(roots[dataset])
        need((root / v6.NEW_MANIFEST).is_file(), 'Missing assets manifest: ' + dataset)
        need(sha(root / v6.NEW_MANIFEST) == record['manifest_sha256'], 'Assets manifest changed: ' + dataset)
        am = read(root / v6.NEW_MANIFEST)
        need(am.get('schema') == record['schema'] and am.get('dataset') == dataset, 'Assets schema differs: ' + dataset)
        if record['input_code']:
            v6.verify_code(root, am)
        manifests[dataset] = am
    for arm in arms:
        dataset = release['arm_assets'][arm['name']]
        root, am = Path(roots[dataset]), manifests[dataset]
        for c in arm['checkpoints']:
            staged = am['checkpoints'].get(c['name']) or {}
            need(am['files'].get(c['path']) == c['sha256'] and staged.get('sha256') == c['sha256']
                 and staged.get('path') == c['path'] and staged.get('meta') == c['meta'],
                 'Checkpoint not bound by the assets manifest: ' + c['name'])
            path = root / c['path']
            need(path.is_file(), 'Missing checkpoint: %s/%s' % (dataset, c['path']))
            need(sha(path) == c['sha256'], 'Checkpoint sha256 differs: %s/%s' % (dataset, c['path']))
    return manifests


def code_root(release, roots):
    """The mounted dataset whose shipped input code the centre-plane arms import (None without such an arm)."""
    center = [a['name'] for a in release['arms'] if a['input_policy'] == 'center3']
    return Path(roots[release['arm_assets'][center[0]]]) if center else None


# ------------------------------------------------------------------ release manifest
def validate_release(p):
    """v6 arm validation plus the shared-decode and multi-dataset rules (C, D)."""
    arms = v6.validate_arms(p.get('arms'))
    names = [a['name'] for a in arms]
    need(not set(names) & set(RESERVED_ARM_NAMES), 'Reserved arm name (combine/shared)')
    need(all(a['contract'] == DECODE_CONTRACT for a in arms),
         'Every arm must read the shared %s volume' % DECODE_CONTRACT)
    members = [c['name'] for a in arms for c in a['checkpoints']]
    need(len(members) == len(set(members)), 'A checkpoint may serve only one arm')
    sets = p.get('asset_sets')
    need(isinstance(sets, list) and sets and all(isinstance(s, dict) and set(s) == set(ASSET_SET_KEYS) for s in sets),
         'Asset sets malformed')
    datasets = [s['dataset'] for s in sets]
    need(len(set(datasets)) == len(datasets), 'Duplicate assets dataset')
    for s in sets:
        check_asset_dataset(s['dataset'])
        need(s['schema'] in ASSET_SCHEMAS and s['input_code'] is ASSET_SCHEMAS[s['schema']]
             and isinstance(s['manifest_sha256'], str) and v6.SHA.fullmatch(s['manifest_sha256']),
             'Asset set record differs: ' + str(s['dataset']))
    bound = p.get('arm_assets')
    need(isinstance(bound, dict) and list(bound) == names and set(bound.values()) == set(datasets),
         'Arm-to-assets binding differs (every arm one dataset, every dataset serves an arm)')
    with_code = {s['dataset'] for s in sets if s['input_code']}
    need(all(bound[a['name']] in with_code for a in arms if a['input_policy'] == 'center3'),
         'A center3 arm needs an assets dataset that ships the input code')
    total = sum(c['bytes'] for a in arms for c in a['checkpoints'])
    need(total <= MAX_RESIDENT_CHECKPOINT_BYTES,
         'Resident checkpoint bytes %d exceed %d' % (total, MAX_RESIDENT_CHECKPOINT_BYTES))
    return arms


def checked_manifest(path, digest):
    need(sha(path) == digest, 'Release manifest changed')
    p = read(path)
    need(p.get('schema') == SCHEMA and p.get('time_policy') == TIME and p.get('decode') == DECODE_POLICY
         and p.get('old_asset_protocol_sha256') == v6.ASSETS_PROTOCOL_SHA and p.get('RSS_bytes') == v6.RSS
         and p.get('output_bytes') == v6.OUTPUT and p.get('prob_clamp') == v6.PROB_EPS and p.get('retries') == 0,
         'Release policy differs')
    need(isinstance(p.get('release'), str) and v6.RELEASE_NAME.fullmatch(p['release']), 'Release binding missing')
    need(sha(__file__) == p.get('release_source_sha256'), 'Executing release source changed')
    need(sha(v6.__file__) == V6_SOURCE_SHA256 == p.get('v6_source_sha256'), 'Embedded v6 source changed')
    validate_release(p)
    need(p.get('rank_weights') == v6.rank_weights(p['arms']), 'Recorded rank weights differ from the arms')
    return p


# ------------------------------------------------------------------ time policy (B)
def model_seconds_per_study(arms):
    """Sum over arms of n_models*model_seconds_per_study (plain left-to-right sum, as the notebook screen)."""
    total = 0.0
    for a in arms:
        k, m = len(a['checkpoints']), a['model_seconds_per_study']
        need(type(k) is int and k > 0 and type(m) in (int, float) and math.isfinite(m) and m >= 0, 'Invalid arm cost')
        total += k * m
    return total


def shared_seconds(n, arms):
    """Conservative seconds of the shared worker: ONE decode per study plus every member's forward pass."""
    need(type(n) is int and n > 0, 'Positive count required')
    return TIME['inflation'] * (TIME['decode_seconds_per_study'] + model_seconds_per_study(arms)) * n


def separate_seconds(n, arms):
    """The v5/v6 screen for the same arms (one decode per arm), for comparison only."""
    total = 0.0
    for a in arms:
        total += v6.arm_projection(n, len(a['checkpoints']), a['model_seconds_per_study'])
    return total + TIME['screen_fixed_seconds']


def operational_plan(n, arms):
    need(isinstance(arms, list) and arms and len({a['name'] for a in arms}) == len(arms), 'Unique arms required')
    projected = shared_seconds(n, arms) + TIME['screen_fixed_seconds']
    need(projected <= TIME['screen_seconds'], 'N-dependent time screen failed')
    separate = separate_seconds(n, arms)
    return {'studies': n, 'decode_projected_seconds': TIME['inflation'] * TIME['decode_seconds_per_study'] * n,
            'arm_model_projected_seconds': {a['name']: TIME['inflation'] * len(a['checkpoints'])
                                            * a['model_seconds_per_study'] * n for a in arms},
            'projected_seconds': projected, 'separate_workers_projected_seconds': separate,
            'saved_seconds': separate - projected,
            'internal_seconds': TIME['public3_seconds'] if n <= 3 else TIME['full_seconds'],
            'formula': 'inflation*(decode_seconds_per_study + sum over arms of n_models*model_seconds_per_study)*N'
                       ' + screen_fixed_seconds <= screen_seconds (one shared decode)',
            'scope': 'Conservative engineering scenario, not hidden runtime proof'}


def max_studies(arms):
    per_study = TIME['inflation'] * (TIME['decode_seconds_per_study'] + model_seconds_per_study(arms))
    return int((TIME['screen_seconds'] - TIME['screen_fixed_seconds']) // per_study)


def launch_projection(plan):
    """Seconds still needed when the one shared worker launches: its projection plus one arm_fixed cost."""
    return plan['projected_seconds'] - TIME['screen_fixed_seconds'] + TIME['arm_fixed_seconds']


# ------------------------------------------------------------------ shared decode serving (A)
def volume_sha256(volume):
    import numpy as np
    return hashlib.sha256(memoryview(np.ascontiguousarray(volume)).cast('B')).hexdigest()


def mask_sha256(mask):
    import numpy as np
    return hashlib.sha256(np.asarray(mask).tobytes()).hexdigest()


def read_only(array):
    import numpy as np
    view = np.asarray(array).view()
    view.setflags(write=False)
    return view


class SharedStudy:
    """Corpus view over ONE decoded study with the DicomCorpus interface used by prepare_input/center3_prepare_input
    (depth, spec, volume(uid), mask(uid)). It holds read-only arrays and no source: nothing can decode a second time."""

    def __init__(self, uid, volume, mask, depth, spec):
        self.uid, self.depth, self.spec = str(uid), depth, spec
        self._volume, self._mask = read_only(volume), read_only(mask)
        self.reads = 0

    def _serve(self, uid, value):
        if str(uid) != self.uid:
            raise KeyError(uid)
        self.reads += 1
        return value

    def volume(self, uid):
        return self._serve(uid, self._volume)

    def mask(self, uid):
        return self._serve(uid, self._mask)


def _quiet(*args, **kwargs):
    return None


def load_arms(arms, member_roots, device, state, loader=None):
    """v6 serve_arm's loading, for every arm: frozen load_checkpoint -> v6 checkpoint_check -> device -> v6 mean_prob
    wrapper. state['models'] / state['loading'] keep every module for the caller's cleanup, even on failure."""
    if loader is None:
        from raptor_common import load_checkpoint as loader
    models = state.setdefault('models', {})
    checks = {}
    for arm in arms:
        members = state['loading'] = []
        arm_checks = []
        for ck in arm['checkpoints']:
            member, meta = loader(Path(member_roots[arm['name']]) / ck['path'])
            arm_checks.append(v6.checkpoint_check(meta, arm, ck))
            members.append(member.eval().to(device))   # move now: CPU RSS holds at most one checkpoint
        models[arm['name']] = v6.mean_prob_module(members)
        state['loading'] = []
        checks[arm['name']] = arm_checks
    return models, checks


def serve_shared(arms, member_roots, corpus, ids, amp, device, state=None, log=print, runtime_module=None,
                 loader=None, skip_failed_inputs=False):
    """Decode each study once, then run every arm on that one volume through the frozen raptor_runtime.predict.

    -> {'probs': {arm: float32 (N, 12)}, 'timing': {arm: predict-format timing, decode excluded}, 'checks',
        'input_functions', 'decode': trace, 'scored_ids', 'unscored', 'load_seconds', 'seconds'}.
    A study whose decode fails raises (release semantics, like v5/v6) unless skip_failed_inputs (a stage-2 adapter keeps
    the anchor's rank for it); every other problem raises."""
    import numpy as np
    rt = runtime_module
    if rt is None:
        import raptor_runtime as rt
    state = {} if state is None else state
    ids = [str(u) for u in ids]
    need(bool(ids) and len(set(ids)) == len(ids), 'inference requires a nonempty, unique study ID list')
    names = [a['name'] for a in arms]
    need(bool(arms) and len(set(names)) == len(names), 'Unique arms required')
    started = time.perf_counter()
    models, checks = load_arms(arms, member_roots, device, state, loader)
    load_seconds = time.perf_counter() - started
    probs = {n: np.zeros((len(ids), len(TARGETS)), np.float32) for n in names}
    hashes = {n: [] for n in names}
    aggregates = {n: hashlib.sha256() for n in names}
    arm_seconds = dict.fromkeys(names, 0.0)
    functions, schemes = {}, {}
    decode = {'contract': getattr(corpus, 'contract', DECODE_CONTRACT), 'decodes': 0, 'seconds': 0.0,
              'max_seconds': 0.0, 'studies': [], 'shared_by': names}
    volumes = hashlib.sha256()
    scored, unscored = [], {}
    loop = time.perf_counter()
    for i, uid in enumerate(ids):
        t = time.perf_counter()
        try:
            volume = corpus.volume(uid)
            mask = corpus.mask(uid)
        except Exception as exc:
            if skip_failed_inputs:
                unscored[uid] = ('input: %s: %s' % (type(exc).__name__, exc))[:500]
                continue
            raise RuntimeError(INPUT_FAILURE + '%s: %s' % (uid, exc)) from exc
        dt = time.perf_counter() - t
        decode['decodes'] += 1
        decode['seconds'] += dt
        decode['max_seconds'] = max(decode['max_seconds'], dt)
        vsha, msha = volume_sha256(volume), mask_sha256(mask)
        receipt = (getattr(corpus, 'receipts', None) or {}).get(uid)
        if isinstance(receipt, dict) and 'volume_sha256' in receipt:
            need(receipt['volume_sha256'] == vsha and receipt.get('mask_sha256') == msha,
                 'Decoded volume differs from its DICOM receipt: ' + uid)
        view = SharedStudy(uid, volume, mask, corpus.depth, corpus.spec)
        row = {}
        for arm in arms:
            name = arm['name']
            t = time.perf_counter()
            with v6.serving_input(rt, arm['input_policy']) as how:
                p, timing = rt.predict(models[name], view, [uid], arm['k_eval'], arm['res'], amp, device,
                                       norm=arm['norm'], batch=1, log=_quiet, window_policy=arm['window_policy'])
            arm_seconds[name] += time.perf_counter() - t
            p = np.asarray(p)
            need(p.dtype == np.float32 and p.shape == (1, len(TARGETS)) and bool(np.isfinite(p).all())
                 and timing['fp32_fallbacks'] == 0 and timing['all_finite'] is True,
                 'Incomplete/nonfinite inference: %s %s' % (name, uid))
            record = timing['input_tensor_hashes'][0]
            need(record['uid'] == uid, 'input hash order differs')
            row[name] = (p[0], record)
            functions[name], schemes[name] = how, timing['input_hash_scheme']
        need(volume_sha256(volume) == vsha and mask_sha256(mask) == msha, 'Shared volume changed while serving ' + uid)
        need(view.reads >= 2 * len(arms), 'An arm did not read the shared volume: ' + uid)
        for name, (p, record) in row.items():
            probs[name][i] = p
            hashes[name].append(record)
            aggregates[name].update((json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
        decode['studies'].append({'uid': uid, 'volume_sha256': vsha, 'mask_sha256': msha})
        volumes.update(('%s %s %s\n' % (uid, vsha, msha)).encode())
        scored.append(i)
        if (i + 1) % 50 == 0 or i + 1 == len(ids):
            log('  %d/%d studies, %.0fs (decode %.0fs), unscored %d'
                % (i + 1, len(ids), time.perf_counter() - loop, decode['seconds'], len(unscored)))
    need(bool(scored), 'no study scored')
    n = len(scored)
    decode.update(per_study_s=round(decode['seconds'] / n, 4), seconds=round(decode['seconds'], 3),
                  max_seconds=round(decode['max_seconds'], 3), ordered_volumes_sha256=volumes.hexdigest())
    timing = {a['name']: {'seconds': round(arm_seconds[a['name']], 3),
                          'per_study_s': round(arm_seconds[a['name']] / n, 4), 'fp32_fallbacks': 0, 'all_finite': True,
                          'batch_size': 1, 'window_policy': a['window_policy'], 'norm': a['norm'],
                          'input_hash_scheme': schemes[a['name']],
                          'ordered_inputs_sha256': aggregates[a['name']].hexdigest(),
                          'input_tensor_hashes': hashes[a['name']], 'decode_included': False} for a in arms}
    keep = [ids[i] for i in scored]
    out = {x: (probs[x] if len(keep) == len(ids) else probs[x][scored]) for x in names}
    return {'probs': out, 'timing': timing, 'checks': checks, 'input_functions': functions, 'decode': decode,
            'scored_ids': keep, 'unscored': unscored, 'load_seconds': round(load_seconds, 3),
            'seconds': round(time.perf_counter() - started, 3)}


def ordered_inputs_sha256(records):
    """The frozen predict's ordered aggregate over [{uid, sha256}, ...]."""
    h = hashlib.sha256()
    for record in records:
        h.update((json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
    return h.hexdigest()


# ------------------------------------------------------------------ Kaggle-side commands
def arm_destinations(release, base, final):
    names = [a['name'] for a in release['arms']]
    if len(names) == 1:
        return {names[0]: Path(final)}
    return {n: Path(base) / n / 'arm.csv' for n in names}


def arm_receipt(arm, release, served, input_code, raw_sha256):
    """The worker receipt section of one arm (validated by check_worker_receipt)."""
    name = arm['name']
    return {'aggregator': arm['aggregator'], 'n_models': len(arm['checkpoints']), 'k_eval': arm['k_eval'],
            'rank_weight': arm['weight'], 'rank_weight_normalised': release['rank_weights'][name],
            'input_policy': arm['input_policy'], 'norm': arm['norm'], 'input_function': served['input_functions'][name],
            'input_code': input_code if arm['input_policy'] == 'center3' else {},
            'assets_dataset': release['arm_assets'][name], 'checkpoints': served['checks'][name],
            'timing': served['timing'][name], 'raw_sha256': raw_sha256}


def check_worker_receipt(w, p, digest, contract):
    """The shared worker receipt must bind the release, the cohort, the one-decode trace and every arm."""
    arms, ids = p['arms'], contract['ids']
    names = [a['name'] for a in arms]
    need(w.get('status') == 'PASS' and w.get('schema') == SCHEMA and w.get('release_manifest_sha256') == digest
         and w.get('arm_order') == names and w.get('test_contract') == contract and w.get('studies') == len(ids)
         and w.get('targets') == TARGETS and not w.get('cleanup_errors'), 'Worker contract failed')
    d = w.get('decode') or {}
    need(d.get('contract') == DECODE_CONTRACT and d.get('decodes') == len(ids) and d.get('shared_by') == names
         and [x.get('uid') for x in d.get('studies', [])] == ids, 'Shared decode trace failed')
    dicom = w.get('DICOM_receipts') or {}
    need(all((dicom.get(x['uid']) or {}).get('volume_sha256') == x['volume_sha256']
             and (dicom.get(x['uid']) or {}).get('mask_sha256') == x['mask_sha256'] for x in d['studies']),
         'Shared decode differs from the DICOM receipts')
    rt = w.get('runtime') or {}
    need(rt.get('effective_amp') == 'fp16' and rt.get('policy') == 'strict' and not rt.get('precision_fallback_allowed')
         and not rt['backends']['cudnn_allow_tf32'] and not rt['backends']['matmul_allow_tf32'], 'Runtime differs')
    for arm in arms:
        name = arm['name']
        wa = (w.get('arms') or {}).get(name) or {}
        cks = wa.get('checkpoints') or []
        need([(c['name'], c['sha256'], c['k_eval_release']) for c in cks]
             == [(c['name'], c['sha256'], arm['k_eval']) for c in arm['checkpoints']]
             and wa.get('n_models') == len(arm['checkpoints']) and wa.get('aggregator') == 'mean_prob'
             and wa.get('rank_weight') == arm['weight'] and wa.get('assets_dataset') == p['arm_assets'][name],
             'Worker ensemble differs: ' + name)
        t = wa.get('timing') or {}
        need(wa.get('input_policy') == arm['input_policy'] and wa.get('norm') == arm['norm'] == t.get('norm')
             and all(c['input_policy'] == arm['input_policy'] for c in cks)
             and wa.get('input_code') == (v6.code_record() if arm['input_policy'] == 'center3' else {}),
             'Worker input differs: ' + name)
        records = t.get('input_tensor_hashes') or []
        need(t.get('all_finite') is True and t.get('fp32_fallbacks') == 0 and t.get('batch_size') == 1
             and t.get('window_policy') == arm['window_policy'] and [x['uid'] for x in records] == ids
             and ordered_inputs_sha256(records) == t.get('ordered_inputs_sha256'), 'Input/finite trace failed: ' + name)
    return True


def worker(a):
    p = checked_manifest(a.manifest, a.sha256)
    arms = p['arms']
    names = [x['name'] for x in arms]
    assets, work = Path(a.assets), Path(a.work)
    roots = {d: Path(x) for d, x in parse_pairs(a.asset_root, '--asset-root').items()}
    old = v6.verify_old_assets(assets, a.opencv)
    verify_asset_sets(p, roots, arms)
    need(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 12), 'Linux CPython3.12 required')
    context = read(work / 'launch.json')
    contract = v6.test_contract(a.competition)
    need(contract == context['test_contract'] and context['arms'] == names, 'Test metadata changed')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', KNEE_WORKSPACE_ROOT=str(assets))
    spec = importlib.util.spec_from_file_location('exp100_passive_cpu', assets / 'b_rn3f_serving.py')
    passive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(passive)
    allocation = passive.cpu_allocation(12)
    threads = min(4, allocation['selected_threads'])
    os.environ.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    result = work / 'worker'
    result.mkdir()
    r = {'schema': SCHEMA, 'status': 'FAIL', 'pid': os.getpid(), 'release': p['release'], 'arm_order': names,
         'release_manifest_sha256': a.sha256, 'asset_sets': p['asset_sets'], 'arm_assets': p['arm_assets'],
         'rank_weights': p['rank_weights'], 'arms': {}, 'cleanup_errors': [], 'training': 0, 'label_Gold_reads': 0,
         'scoring': 0, 'retries': 0}
    source = None
    state = {}
    try:
        envdir = Path('/kaggle/temp') / 'exp119-release-env'
        need(not envdir.exists(), 'New dependency dir required')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target',
                        str(envdir), *[str(assets / x['path']) for x in old['wheels']], a.opencv], check=True)
        sys.path[:0] = [str(envdir), str(assets / 'training/src')]
        input_code, code_dir = {}, None
        croot = code_root(p, roots)
        if croot is not None:
            # the EXP107 input modules, verified again right before import and imported AFTER the frozen serving
            # copies, so raptor_common/raptor_runtime stay the frozen EXP089 files (v6 rule)
            input_code = v6.verify_code(croot, read(croot / v6.NEW_MANIFEST))
            code_dir = (croot / v6.CODE_DIR).resolve()
            sys.path.insert(2, str(code_dir))
        import numpy as np
        import pandas as pd
        import torch, timm, cv2, pydicom, gdcm
        need((torch.__version__, timm.__version__, cv2.__version__, pydicom.__version__, gdcm.Version.GetVersion())
             == v6.VERSIONS, 'Completed T4 dependency versions required')
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
             'T4' in torch.cuda.get_device_name(0), 'One visible T4 required')
        from raptor_common import LAB, write_predictions, environment
        import raptor_runtime
        from raptor_dicom import DirectoryDicomSource, DicomCorpus
        frozen = (assets / 'training/src').resolve()
        modules = {'raptor_common': frozen, 'raptor_runtime': frozen, 'raptor_dicom': frozen}
        if code_dir is not None:
            import raptor_radimagenet, raptor_data   # noqa: F401  (bytes verified above)
            modules.update(raptor_radimagenet=code_dir, raptor_data=code_dir)
        need(all(Path(sys.modules[m].__file__).resolve().parent == d for m, d in modules.items()),
             'Serving modules were not imported from the verified asset directories')
        runtime = raptor_runtime.configure_runtime('fp16', 'cuda', 'strict')
        need(LAB == TARGETS, 'Target order changed')
        rows = pd.read_csv(Path(a.competition) / 'test_series.csv', dtype={UID: str, 'SeriesInstanceUID': str})
        source = DirectoryDicomSource(Path(a.competition) / 'test_series')
        corpus = DicomCorpus.from_records(contract['ids'], rows.to_dict('records'), source, DECODE_CONTRACT, 'test')
        member_roots = {x['name']: roots[p['arm_assets'][x['name']]] for x in arms}
        served = serve_shared(arms, member_roots, corpus, contract['ids'], 'fp16', 'cuda', state, log=print,
                              runtime_module=raptor_runtime)
        need(served['scored_ids'] == contract['ids'] and not served['unscored'], 'Incomplete inference')
        for arm in arms:
            name = arm['name']
            probs = served['probs'][name]
            need(probs.dtype == np.float32 and probs.shape == (contract['studies'], 12)
                 and bool(np.isfinite(probs).all()), 'Incomplete/nonfinite inference: ' + name)
            write_predictions(result / (name + '.csv'), contract['ids'], probs)
            r['arms'][name] = arm_receipt(arm, p, served, input_code, sha(result / (name + '.csv')))
        r.update(status='PASS', studies=contract['studies'], targets=LAB, runtime=runtime, decode=served['decode'],
                 load_seconds=served['load_seconds'], serve_seconds=served['seconds'], prob_clamp=v6.PROB_EPS,
                 allocation={**allocation, 'selected_internal_threads': threads, 'data_loader_workers': 0},
                 environment=environment(),
                 source_bindings={n: old['files'][n] for n in old['files'] if n.startswith('training/src/')},
                 test_contract=contract, DICOM_receipts=corpus.receipts,
                 CUDA_peak_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        r['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        held = [*state.get('models', {}).values(), *state.get('loading', [])]
        for label, action in [('models', lambda: [m.to('cpu') for m in held]),
                              ('source', lambda: source.close() if source is not None else None)]:
            try:
                action()
            except Exception as exc:
                r['cleanup_errors'].append(label + ': ' + str(exc))
        if 'torch' in sys.modules:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as exc:
                r['cleanup_errors'].append(str(exc))
        if r['cleanup_errors']:
            r['status'] = 'FAIL'
        write(result / 'receipt.json', r)
    need(r['status'] == 'PASS', 'Worker cleanup failed')


def asset_root_args(roots):
    return [x for d, path in roots.items() for x in ('--asset-root', d + '=' + str(path))]


def run(a):
    p = checked_manifest(a.manifest, a.sha256)
    arms = p['arms']
    names = [x['name'] for x in arms]
    base, final, assets = Path(a.base), Path(a.final), Path(a.assets)
    roots = parse_pairs(a.asset_root, '--asset-root')
    work = base / SHARED_DIR / 'work'
    destinations = arm_destinations(p, base, final)
    need(not work.exists() and not any(d.exists() for d in destinations.values())
         and not (len(names) > 1 and any((base / n).exists() for n in names)), 'Preserve previous outputs')
    work.mkdir(parents=True)
    result = {'status': 'FAIL', 'schema': SCHEMA, 'release': p['release'], 'arm_order': names,
              'release_manifest_sha256': a.sha256, 'accuracy_claim': False}
    published = {}
    try:
        started = a.started
        contract = v6.test_contract(a.competition)
        plan = operational_plan(contract['studies'], arms)
        projection = launch_projection(plan)
        deadline, timeout = v6.launch_budget(started, time.monotonic(), contract['studies'], projection,
                                             a.bootstrap_max_seconds)
        need(sha(assets / v6.OLD_PROTOCOL) == v6.ASSETS_PROTOCOL_SHA, 'Budget/assets failed')
        old = read(assets / v6.OLD_PROTOCOL)
        need(sha(assets / 'process_resource_receipt.py') == old['files']['process_resource_receipt.py'],
             'Supervisor changed')
        need(set(roots) == {s['dataset'] for s in p['asset_sets']}, 'Mounted assets datasets differ from the release')
        for s in p['asset_sets']:
            need(sha(Path(roots[s['dataset']]) / v6.NEW_MANIFEST) == s['manifest_sha256'],
                 'Assets manifest changed: ' + s['dataset'])
        argv = [sys.executable, str(Path(__file__).resolve()), 'worker', '--manifest', a.manifest, '--sha256', a.sha256,
                '--assets', a.assets, *asset_root_args(roots), '--work', str(work), '--competition', a.competition,
                '--opencv', a.opencv]
        context = {'arms': names, 'test_contract': contract, 'operational_plan': plan,
                   'launch_projection_seconds': projection, 'bootstrap_max_seconds': a.bootstrap_max_seconds,
                   'worker_argv': argv, 'deadline_monotonic': deadline, 'worker_timeout_seconds': timeout}
        write(work / 'launch.json', context)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        code = subprocess.run([sys.executable, str(assets / 'process_resource_receipt.py'), '--out',
                               str(work / 'supervisor'), '--timeout', str(timeout), '--', *argv], env=env).returncode
        need(code == 0, 'Supervisor failed')
        w, s = read(work / 'worker/receipt.json'), read(work / 'supervisor/receipt.json')
        check_worker_receipt(w, p, a.sha256, contract)
        cgroup_telemetry = v6.validate_resource(s, argv, w['pid'], timeout)
        raws = {}
        for name in names:
            raw = work / 'worker' / (name + '.csv')
            digest = v6.csv_check(raw, contract['ids'])
            need(digest == w['arms'][name]['raw_sha256'], 'Worker output changed: ' + name)
            raws[name] = (raw, digest)
        need(v6.output_bytes(work) <= v6.OUTPUT, 'Output budget exceeded')
        with zipfile.ZipFile(work / 'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for name in ['launch.json', 'worker/receipt.json', *['worker/%s.csv' % n for n in names],
                         'supervisor/receipt.json', 'supervisor/stdout.log', 'supervisor/stderr.log']:
                z.write(work / name, name)
        need(v6.output_bytes(work) <= v6.OUTPUT, 'Archive output budget exceeded')
        arm_results = {}
        for arm in arms:
            name, wa = arm['name'], w['arms'][arm['name']]
            arm_results[name] = {
                'submission_sha256': raws[name][1], 'path': str(destinations[name]), 'n_models': wa['n_models'],
                'k_eval': arm['k_eval'], 'rank_weight': arm['weight'],
                'rank_weight_normalised': p['rank_weights'][name], 'input_policy': arm['input_policy'],
                'norm': arm['norm'], 'input_function': wa['input_function'], 'assets_dataset': wa['assets_dataset'],
                'checkpoints': wa['checkpoints'], 'model_seconds': wa['timing']['seconds']}
        result.update(status='PASS_TEST_COHORT_ONLY', studies=contract['studies'], columns=contract['columns'],
                      operational_plan=plan, launch_projection_seconds=projection, arms=arm_results,
                      decode={k: w['decode'][k] for k in ('contract', 'decodes', 'seconds', 'per_study_s',
                                                          'max_seconds', 'ordered_volumes_sha256')},
                      load_seconds=w['load_seconds'], serve_seconds=w['serve_seconds'],
                      new_asset_datasets=[x['dataset'] for x in p['asset_sets']], rank_weights=p['rank_weights'],
                      worker_sha256=sha(work / 'worker/receipt.json'),
                      supervisor_sha256=sha(work / 'supervisor/receipt.json'), evidence_sha256=sha(work / 'evidence.zip'),
                      sampled_tree_rss_bytes=s['sampled_tree_rss_max_bytes'], cgroup_telemetry=cgroup_telemetry,
                      elapsed_seconds=time.monotonic() - started,
                      scope='Actual supplied test cohort only; Public3 is not hidden runtime or accuracy proof')
        if len(names) == 1:
            result['submission_sha256'] = raws[names[0]][1]

        def final_check():
            need(v6.test_contract(a.competition) == contract, 'Test metadata changed before finalization')
            need(v6.output_bytes(work) <= v6.OUTPUT, 'Final output budget exceeded')
        for name in names:
            destination = destinations[name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            v6.publish(raws[name][0], destination, raws[name][1], deadline, final_check)
            published[name] = (destination, raws[name][1])
        write(work / 'result.json', result)
        need(time.monotonic() < deadline and v6.output_bytes(work) <= v6.OUTPUT, 'Final acknowledgment budget failed')
        return result
    except BaseException as exc:
        for destination, digest in published.values():
            if destination.is_file() and sha(destination) == digest:
                destination.unlink()
        result.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc)})
        write(work / 'failure.json', result)
        raise


def combine(a):
    """The shared run's validated arm CSVs -> v6 per-target average-tie rank mean -> atomic submission."""
    p = checked_manifest(a.manifest, a.sha256)
    arms = p['arms']
    names = [x['name'] for x in arms]
    need(len(arms) >= 2, 'Rank combination needs two or more arms')
    base, destination = Path(a.base), Path(a.out)
    work = base / 'combine'
    need(not work.exists() and not destination.exists(), 'Preserve previous outputs')
    work.mkdir(parents=True)
    result = {'status': 'FAIL', 'schema': SCHEMA, 'release': p['release'], 'release_manifest_sha256': a.sha256,
              'accuracy_claim': False}
    published_sha = None
    try:
        started = a.started
        contract = v6.test_contract(a.competition)
        ids = contract['ids']
        plan = operational_plan(contract['studies'], arms)
        need(math.isfinite(started) and 0 <= time.monotonic() - started, 'Invalid clock')
        deadline = started + plan['internal_seconds']
        need(time.monotonic() < deadline, 'Deadline passed before combination')
        shared = read(base / SHARED_DIR / 'work/result.json')
        need(shared.get('status') == 'PASS_TEST_COHORT_ONLY' and shared.get('schema') == SCHEMA
             and shared.get('release_manifest_sha256') == a.sha256 and shared.get('studies') == contract['studies']
             and shared.get('arm_order') == names, 'Shared result differs')
        tables, records = [], {}
        for arm in arms:
            name, rec = arm['name'], shared['arms'][arm['name']]
            path = base / name / 'arm.csv'
            need(v6.csv_check(path, ids) == rec['submission_sha256'], 'Arm output changed: ' + name)
            need(rec.get('rank_weight') == arm['weight'], 'Arm result weight differs: ' + name)
            tables.append(v6.load_probabilities(path, ids))
            records[name] = {k: rec[k] for k in ('submission_sha256', 'n_models', 'k_eval', 'rank_weight',
                                                 'model_seconds')}
        weights = [x['weight'] for x in arms]
        cols = v6.rank_mean(tables, weights)
        raw = work / 'combined.csv'
        v6.write_columns(raw, ids, cols)
        published_sha = v6.csv_check(raw, ids)
        equal = len(set(weights)) == 1
        result.update(status='PASS_RANK_MEAN_TEST_COHORT_ONLY', studies=len(ids), submission_sha256=published_sha,
                      formula=('per-target mean of average-tie percentile ranks over the full cohort, equal arm weights '
                               '(EXP098)' if equal else 'per-target weighted mean of average-tie percentile ranks over '
                               'the full cohort: sum_i(W_i*r_i)/sum_i(W_i)'),
                      arm_order=names, arms=records, arm_weights={x['name']: x['weight'] for x in arms},
                      rank_weights=p['rank_weights'], shared_decode=shared['decode'],
                      scope='Actual supplied test cohort only; Public3 is not hidden runtime or accuracy proof')

        def final_check():
            need(v6.test_contract(a.competition) == contract, 'Test metadata changed before finalization')
        v6.publish(raw, destination, published_sha, deadline, final_check)
        result['elapsed_seconds'] = time.monotonic() - started
        write(work / 'result.json', result)
        need(time.monotonic() < deadline, 'Final acknowledgment budget failed')
        return result
    except BaseException as exc:
        if destination.is_file() and published_sha is not None and sha(destination) == published_sha:
            destination.unlink()
        result.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc)})
        write(work / 'failure.json', result)
        raise


# ------------------------------------------------------------------ local preparation (no network, no GPU)
def stage_assets(a):
    """v6 stage_assets into an explicitly named NEW package; every published release-assets slug/package is refused."""
    need(a.dataset not in PUBLISHED_ASSET_DATASETS, 'Refusing a published release-assets dataset (v4-v10): ' + a.dataset)
    package = local_package(a.package)
    need(all(package.resolve() != p.resolve() for p in PUBLISHED_ASSET_PACKAGES),
         'Refusing a published release-assets package directory (v4-v10)')
    v6.check_dataset(a.dataset)
    return v6.stage_assets(argparse.Namespace(checkpoint=a.checkpoint, package=str(package), dataset=a.dataset,
                                              copy=a.copy))


CODE = r'''import time
started=time.monotonic()
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
final=Path('/kaggle/working/submission.csv')
base=Path('/kaggle/working/exp119-@@RELEASE@@')
assert not base.exists() and not final.exists()
base.mkdir()
v6=base64.b85decode(@@V6_SOURCE@@)
assert hashlib.sha256(v6).hexdigest()==@@V6_SOURCE_SHA@@
(base/'exp111_release_v6.py').write_bytes(v6)
source=base64.b85decode(@@SOURCE@@)
assert hashlib.sha256(source).hexdigest()==@@SOURCE_SHA@@
tool=base/'exp119_release_v7.py'; tool.write_bytes(source)
MANIFEST=@@MANIFEST@@
DIGEST=@@DIGEST@@
manifest=base/'release-manifest.json'; manifest.write_bytes(MANIFEST.encode('utf-8'))
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==DIGEST
release=json.loads(MANIFEST); T=release['time_policy']; arms=[a['name'] for a in release['arms']]
def one(paths,marker):
    found=[p for p in paths if (p/marker).is_file()]
    assert len(found)==1,marker
    return found[0]
old=one([Path('/kaggle/input/knee-exp089-recipe-transfer-v1'),Path('/kaggle/input/datasets/taeyangg4/knee-exp089-recipe-transfer-v1')],'exp089-t4-protocol.json')
roots=[]
for s in release['asset_sets']:
    o,n=s['dataset'].split('/')
    roots+=['--asset-root',s['dataset']+'='+str(one([Path('/kaggle/input/'+n),Path('/kaggle/input/datasets/'+o+'/'+n)],'assets-manifest.json'))]
opencv_roots=[Path('/kaggle/input/opencv-python-headless-4120088-x86'),Path('/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86')]
opencv=[p for d in opencv_roots for p in d.glob('opencv_python_headless-4.12.0.88-*.whl')]
assert len(opencv)==1
comp_roots=[Path('/kaggle/input/rsna-knee-abnormality-detection'),Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
competition=[p for p in comp_roots if all((p/n).is_file() for n in ['test.csv','test_series.csv','sample_submission.csv']) and (p/'test_series').is_dir()]
assert len(competition)==1
n_test=sum(1 for _ in open(competition[0]/'sample_submission.csv',encoding='utf-8-sig'))-1
assert n_test>0
per_study=0.0
for a in release['arms']:
    per_study+=len(a['checkpoints'])*a['model_seconds_per_study']
screen=T['inflation']*(T['decode_seconds_per_study']+per_study)*n_test+T['screen_fixed_seconds']
assert screen<=T['screen_seconds'],'v7 time screen failed before any GPU work'
subprocess.run([sys.executable,str(tool),'run','--manifest',str(manifest),'--sha256',DIGEST,'--assets',str(old),*roots,'--base',str(base),'--final',str(final),'--competition',str(competition[0]),'--opencv',str(opencv[0]),'--started',repr(started),'--bootstrap-max-seconds','600'],check=True)
shared=json.loads((base/'shared/work/result.json').read_text()); assert shared['status']=='PASS_TEST_COHORT_ONLY'
for a in arms:
    assert hashlib.sha256(Path(shared['arms'][a]['path']).read_bytes()).hexdigest()==shared['arms'][a]['submission_sha256']
if len(arms)>1:
    subprocess.run([sys.executable,str(tool),'combine','--manifest',str(manifest),'--sha256',DIGEST,'--base',str(base),'--competition',str(competition[0]),'--out',str(final),'--started',repr(started)],check=True)
    summary=json.loads((base/'combine/result.json').read_text()); assert summary['status']=='PASS_RANK_MEAN_TEST_COHORT_ONLY'
else:
    summary=shared
assert hashlib.sha256(final.read_bytes()).hexdigest()==summary['submission_sha256']
print(json.dumps({'release':release['release'],'status':summary['status'],'studies':summary['studies'],'submission_sha256':summary['submission_sha256'],'rank_weights':release['rank_weights'],'decode':shared['decode'],'arms':{a:{k:shared['arms'][a].get(k) for k in ('submission_sha256','n_models','k_eval','rank_weight','model_seconds')} for a in arms},'elapsed_seconds':time.monotonic()-started}))
'''


def chunked(text):
    """A long literal as implicitly concatenated SOURCE_CHUNK-character lines (Kaggle notebook line limit, D-B-015)."""
    chunks = [text[i:i + v6.SOURCE_CHUNK] for i in range(0, len(text), v6.SOURCE_CHUNK)] or ['']
    return '(\n' + '\n'.join(repr(c) for c in chunks) + '\n)'


def render_code(release, v6_source, source, manifest_text, digest):
    need(hashlib.sha256(manifest_text.encode('utf-8')).hexdigest() == digest, 'Manifest text differs from digest')
    need(v6.RELEASE_NAME.fullmatch(release), 'Release name must match [a-z0-9][a-z0-9-]*')
    need(hashlib.sha256(v6_source).hexdigest() == V6_SOURCE_SHA256, 'Embedded v6 source differs from the reviewed v6')
    values = {'RELEASE': release, 'V6_SOURCE': chunked(base64.b85encode(v6_source).decode()),
              'V6_SOURCE_SHA': repr(V6_SOURCE_SHA256), 'SOURCE': chunked(base64.b85encode(source).decode()),
              'SOURCE_SHA': repr(hashlib.sha256(source).hexdigest()), 'MANIFEST': chunked(manifest_text),
              'DIGEST': repr(digest)}
    tokens = re.findall(r'@@([A-Z0-9_]+)@@', CODE)
    need(sorted(set(tokens)) == sorted(values), 'Template placeholders differ')
    code = re.sub(r'@@([A-Z0-9_]+)@@', lambda m: values[m.group(1)], CODE)   # one pass: inserted text is never rescanned
    compile(code, 'exp119-release-notebook', 'exec')
    need(max(len(line) for line in code.split('\n')) <= v6.MAX_NOTEBOOK_LINE, 'Notebook line too long for Kaggle')
    need('--timeout' not in CODE, 'The kernel must never pass --timeout')
    return code


def kernel_metadata(release, docker, datasets):
    need(v6.RELEASE_NAME.fullmatch(release), 'Release name must match [a-z0-9][a-z0-9-]*')
    need(docker == v6.DOCKER_IMAGE and re.fullmatch(r'gcr\.io/[^@\s]+@sha256:[0-9a-f]{64}', docker),
         'Pinned image differs')
    need(isinstance(datasets, list) and datasets and len(set(datasets)) == len(datasets), 'Assets datasets required')
    for d in datasets:
        check_asset_dataset(d)
    slug = KERNEL_PREFIX + release + KERNEL_SUFFIX
    need(len(slug) <= MAX_KERNEL_SLUG, 'Kernel slug too long: ' + slug)
    return {'id': v6.OWNER + '/' + slug, 'title': slug, 'code_file': 'release.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': False, 'keywords': ['gpu'],
            'dataset_sources': [v6.OLD_DATASET, *datasets, v6.OPENCV_DATASET], 'kernel_sources': [],
            'competition_sources': [v6.COMPETITION], 'model_sources': [], 'docker_image': docker,
            'machine_shape': 'NvidiaTeslaT4'}


def build_arms(arm_texts, manifests, old, overrides):
    """--arm specs -> (v6 arms, specs, model-seconds basis, arm -> assets dataset). Every member of an arm must be staged
    in exactly one mounted dataset (the same one for the whole arm)."""
    parsed = [v6.parse_arm_spec(text) for text in arm_texts]
    given = [w is not None for _, _, _, w in parsed]
    need(all(given) or not any(given), 'Give a rank weight :W for every arm or for none (none = equal weights)')
    need(len(parsed) > 1 or not any(given), 'A single-arm release has no rank weight')
    need(set(overrides) <= {name for name, _, _, _ in parsed}, 'Model seconds given for an unknown arm')
    arms, specs, basis, bound = [], [], {}, {}
    for text, (name, members, k, w) in zip(arm_texts, parsed):
        for m in members:
            owners = [d for d, am in manifests.items() if m in am['checkpoints']]
            need(len(owners) == 1, 'Checkpoint %s must be staged in exactly one --assets dataset (found %d)'
                 % (m, len(owners)))
        owners = {d for d, am in manifests.items() for m in members if m in am['checkpoints']}
        need(len(owners) == 1, 'Every member of arm %s must come from one assets dataset' % name)
        dataset = owners.pop()
        am = manifests[dataset]
        arch = am['checkpoints'][members[0]]['meta']['arch']
        seconds, basis[name] = v6.arm_model_seconds(name, arch, overrides)
        arm = v6.build_arm(name, members, k, am, seconds, 1.0 if w is None else w)
        for ck in arm['checkpoints']:
            entry = am['checkpoints'][ck['name']]
            code = entry.get('training_code_sha256') or {}
            for f in v6.TRAINING_CODE_BOUND:
                need(code.get(f) == old['files']['training/src/' + f],
                     'Checkpoint %s was trained with %s differing from the frozen serving copy' % (ck['name'], f))
            if arm['input_policy'] == 'center3':
                need(ASSET_SCHEMAS[am['schema']], 'center3 arm %s needs an assets dataset that ships the input code'
                     % name)
                trained = {**{f: code.get(f) for f in v6.CODE_FILES if f != 'raptor_radimagenet.py'},
                           **(entry.get('input_code_sha256') or {})}
                need(trained == v6.CODE_FILES, 'Checkpoint %s was trained with input code differing from the shipped '
                                               'raptor_data.py/raptor_radimagenet.py' % ck['name'])
        arms.append(arm)
        specs.append(text)
        bound[name] = dataset
    return arms, specs, basis, bound


def notebook(a):
    need(v6.RELEASE_NAME.fullmatch(a.release), 'Release name must match [a-z0-9][a-z0-9-]*')
    out = Path(a.out)
    need(not out.exists(), 'Choose a fresh kernel directory')
    pairs = parse_pairs(a.assets or DEFAULT_ASSETS, '--assets')
    records, manifests, packages = [], {}, {}
    for dataset, package in pairs.items():
        record, am, resolved = local_asset_set(dataset, package)
        records.append(record)
        manifests[dataset] = am
        packages[dataset] = resolved
    old = v6.verify_old_assets(v6.OLD_LOCAL)
    need(read(v6.ACCEPTANCE)['status'] == 'ROOT_T4_V2_ENGINEERING_PASS', 'Completed T4 acceptance required')
    pulled = read(a.docker_from)
    need(pulled.get('id') == v6.DOCKER_SOURCE_KERNEL, 'Docker pin must come from the B224 release v3 kernel')
    docker = pulled.get('docker_image')
    overrides = v6.parse_model_seconds(a.model_seconds)
    arms, specs, basis, bound = build_arms(a.arm, manifests, old, overrides)
    weights = v6.rank_weights(arms)
    equal = len({x['weight'] for x in arms}) == 1
    datasets = [r['dataset'] for r in records]
    manifest = {'schema': SCHEMA, 'release': a.release, 'created_at': datetime.now(timezone.utc).isoformat(),
                'release_source_sha256': sha(__file__), 'v6_source_sha256': V6_SOURCE_SHA256,
                'old_asset_dataset': v6.OLD_DATASET, 'old_asset_protocol_sha256': v6.ASSETS_PROTOCOL_SHA,
                'T4_acceptance_sha256': sha(v6.ACCEPTANCE), 'asset_sets': records, 'asset_packages': packages,
                'arm_assets': bound, 'opencv_dataset': v6.OPENCV_DATASET, 'docker_image': docker, 'time_policy': TIME,
                'decode': DECODE_POLICY, 'RSS_bytes': v6.RSS, 'output_bytes': v6.OUTPUT, 'prob_clamp': v6.PROB_EPS,
                'retries': 0, 'arm_specs': specs, 'arms': arms, 'rank_weights': weights, 'model_seconds_basis': basis,
                'combination': ('single arm: raw mean_prob probabilities' if len(arms) == 1 else
                                'per-target mean of average-tie percentile ranks, equal arm weights (EXP098)' if equal
                                else 'per-target weighted mean of average-tie percentile ranks, '
                                     'sum_i(W_i*r_i)/sum_i(W_i) (weights normalised to 1)'),
                'study_count': 'dynamic complete official test cohort', 'accuracy_claim': False}
    validate_release(manifest)
    plan_hidden = operational_plan(a.expected_studies, arms)
    meta = kernel_metadata(a.release, docker, datasets)
    out.mkdir(parents=True)
    write(out / 'release-manifest.json', manifest)
    digest = sha(out / 'release-manifest.json')
    text = (out / 'release-manifest.json').read_bytes().decode('utf-8')
    code = render_code(a.release, Path(v6.__file__).read_bytes(), Path(__file__).read_bytes(), text, digest)
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
          'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
          'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': [
          {'cell_type': 'code', 'id': 'exp119-release-v7', 'metadata': {},
           'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
    write(out / 'release.ipynb', nb)
    write(out / 'kernel-metadata.json', meta)
    py = str(ROOT / '.venv/Scripts/python.exe')
    plan = {'status': 'CPU_PREPARED_NOT_PUSHED', 'release': a.release, 'kernel': meta['id'], 'arms': [
                {'name': x['name'], 'assets_dataset': bound[x['name']],
                 'checkpoints': [c['name'] + ':' + c['sha256'] for c in x['checkpoints']],
                 'k_eval_release': x['k_eval'], 'k_eval_checkpoints': [c['meta']['k_eval'] for c in x['checkpoints']],
                 'arch': x['arch'], 'input_policy': x['input_policy'], 'norm': x['norm'],
                 'model_seconds_per_study': x['model_seconds_per_study'], 'rank_weight': x['weight'],
                 'rank_weight_normalised': weights[x['name']]} for x in arms],
            'manifest_sha256': digest, 'notebook_sha256': sha(out / 'release.ipynb'),
            'metadata_sha256': sha(out / 'kernel-metadata.json'), 'source_sha256': sha(__file__),
            'v6_source_sha256': V6_SOURCE_SHA256, 'asset_datasets': [
                {**r, 'package': packages[r['dataset']], 'published_before_v7': r['dataset'] in PUBLISHED_ASSET_DATASETS,
                 'publish_argv_if_new': [py, '-m', 'kaggle', 'datasets', 'create', '-p', packages[r['dataset']],
                                         '--dir-mode', 'zip']} for r in records],
            'publish_assets_env': {'PYTHONIOENCODING': 'utf-8'}, 'docker_image': docker,
            'docker_image_source': {'kernel': v6.DOCKER_SOURCE_KERNEL, 'pulled_metadata': str(Path(a.docker_from)),
                                    'pulled_metadata_sha256': sha(a.docker_from)},
            'screen_at_expected_hidden_N': plan_hidden, 'max_studies_passing_screen': max_studies(arms),
            # never --timeout (D-B-015 item 13); kaggle_push.py waits for "ready" datasets (D-B-039)
            'push_argv': [py, str(ROOT / 'tools/kaggle_push.py'), str(out.resolve()), '--accelerator', 'NvidiaTeslaT4'],
            'push_argv_plain_cli': [py, '-m', 'kaggle', 'kernels', 'push', '-p', str(out.resolve()),
                                    '--accelerator', 'NvidiaTeslaT4'],
            'submit_argv_template': [py, '-m', 'kaggle', 'competitions', 'submit', v6.COMPETITION, '-k', meta['id'],
                                     '-v', '<VERSION>', '-f', 'submission.csv', '-m', '<message>'],
            'first_output': WORK_PREFIX + a.release + '/' + ('shared/work/result.json' if len(arms) == 1
                                                            else 'combine/result.json'),
            'private': True, 'model_loads': 0, 'GPU_remote_network': 0, 'dataset_upload': False, 'submission': False}
    need(not any('--timeout' in x for k in ('push_argv', 'push_argv_plain_cli') for x in plan[k]), 'No --timeout')
    write(out / 'execution-plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('status', 'kernel', 'manifest_sha256', 'notebook_sha256',
                                           'max_studies_passing_screen')}))
    return plan


# ------------------------------------------------------------------ runtime projections (local arithmetic)
PROJECTION_ARMS = {   # name -> (arch, n_models, description)
    'coat5': (COAT, 5, 'CoAtNet-2 224 K36, 5 folds, Dread labels (assets v8 coat_f0-f4)'),
    'rad5': ('resnet50', 5, 'RadImageNet ResNet50 centre3 K36, 5 folds, Dread labels (assets v8 rad_f0-f4)'),
    'rad5s43': ('resnet50', 5, 'second RadR50 seed (EXP118 s43, 5 folds; not staged yet)'),
    'fl5': (COAT, 5, 'CoAtNet-2 224 K36, 5 folds, flight_v4 labels (assets v9 fl_f0-f4)'),
    'rl5': (COAT, 5, 'CoAtNet-2 224 K36, 5 folds, relabel_v1 labels (assets v7 rl_f0-f4)'),
}
SCENARIOS = [('2 arms: coat5 + rad5', ['coat5', 'rad5']),
             ('3 arms: coat5 + rad5 + rad5s43', ['coat5', 'rad5', 'rad5s43']),
             ('4 arms: coat5 + rad5 + rad5s43 + fl5', ['coat5', 'rad5', 'rad5s43', 'fl5']),
             ('4 arms: coat5 + rad5 + fl5 + rl5', ['coat5', 'rad5', 'fl5', 'rl5']),
             ('5 arms: coat5 + rad5 + rad5s43 + fl5 + rl5', ['coat5', 'rad5', 'rad5s43', 'fl5', 'rl5']),
             ('3 arms H-B-29 label sets: coat5 + fl5 + rl5', ['coat5', 'fl5', 'rl5'])]
# Measured T4 model seconds per model per study (no decode), for an uninflated central estimate next to the screen.
MEASURED_MODEL_SECONDS = {
    COAT: (0.228, 'EXP105 Kaggle T4 (D-B-018): CoAtNet 5-fold K36 worker 1.345 s/study over 100 studies incl. 7.7 s '
                  'load and 12.8 s verify -> (134.5 - 20.5) / 100 / 5 = 0.228 s per model per study, store read included'),
    'resnet50': (0.069, 'EXP107 K36 OOF passes on a Kaggle T4: 0.060-0.069 s per model per study (cache input, batch 4); '
                        'max taken; batch-1 serving may be slower'),
}


def projection_report(n=1300):
    """Shared (v7) vs separate (v5/v6) projections for the SCENARIOS at N studies, from the v5/v6 constants."""
    need(type(n) is int and n > 0, 'Positive count required')
    out = {'schema': 'exp119-runtime-projection-v1', 'studies': n, 'time_policy': TIME,
           'model_seconds_per_study': {k: v6.MODEL_SECONDS[k] for k in (COAT, 'resnet50')},
           'model_seconds_basis': {k: v6.MODEL_SECONDS_BASIS[k] for k in (COAT, 'resnet50')},
           'measured_model_seconds': {k: {'seconds': v[0], 'basis': v[1]} for k, v in MEASURED_MODEL_SECONDS.items()},
           'arms': {k: {'arch': v[0], 'n_models': v[1], 'what': v[2]} for k, v in PROJECTION_ARMS.items()},
           'formulas': {
               'v7_shared_screen': 'inflation*(decode + sum_arms n_models*model_seconds)*N + screen_fixed <= screen',
               'v5_v6_separate_screen': 'sum_arms inflation*(decode + n_models*model_seconds)*N + screen_fixed <= screen',
               'uninflated_expected': 'arm_fixed + (decode + sum_arms n_models*measured_model_seconds)*N; separate: one '
                                      'arm_fixed and one decode per arm'},
           'scope': 'Arithmetic on the release constants; not a measured hidden-test runtime', 'scenarios': []}
    for label, names in SCENARIOS:
        arms = [{'name': x, 'checkpoints': [None] * PROJECTION_ARMS[x][1],
                 'model_seconds_per_study': v6.MODEL_SECONDS[PROJECTION_ARMS[x][0]]} for x in names]
        shared = shared_seconds(n, arms) + TIME['screen_fixed_seconds']
        separate = separate_seconds(n, arms)
        measured = 0.0
        for x in names:
            measured += PROJECTION_ARMS[x][1] * MEASURED_MODEL_SECONDS[PROJECTION_ARMS[x][0]][0]
        expected_shared = TIME['arm_fixed_seconds'] + (TIME['decode_seconds_per_study'] + measured) * n
        expected_separate = 0.0
        for x in names:
            expected_separate += TIME['arm_fixed_seconds'] + (TIME['decode_seconds_per_study'] + PROJECTION_ARMS[x][1]
                                                              * MEASURED_MODEL_SECONDS[PROJECTION_ARMS[x][0]][0]) * n
        per_study_separate = 0.0
        for x in arms:
            per_study_separate += v6.arm_projection(1, len(x['checkpoints']), x['model_seconds_per_study'])
        out['scenarios'].append({
            'label': label, 'arms': names, 'models': sum(PROJECTION_ARMS[x][1] for x in names),
            'v7_shared_screen_seconds': round(shared, 1), 'v7_shared_screen_hours': round(shared / 3600, 3),
            'v7_passes_screen': shared <= TIME['screen_seconds'], 'v7_max_studies': max_studies(arms),
            'v7_headroom_seconds': round(TIME['screen_seconds'] - shared, 1),
            'v5_v6_separate_screen_seconds': round(separate, 1), 'v5_v6_separate_screen_hours': round(separate / 3600, 3),
            'v5_v6_passes_screen': separate <= TIME['screen_seconds'],
            'v5_v6_max_studies': int((TIME['screen_seconds'] - TIME['screen_fixed_seconds']) // per_study_separate),
            'saved_screen_seconds': round(separate - shared, 1),
            'uninflated_expected_shared_seconds': round(expected_shared, 1),
            'uninflated_expected_separate_seconds': round(expected_separate, 1),
            'decode_share_of_shared_screen': round(TIME['inflation'] * TIME['decode_seconds_per_study'] * n / shared, 3)})
    return out


def project(a):
    report = projection_report(a.studies)
    report['created_at'] = datetime.now(timezone.utc).isoformat()
    report['tool_sha256'] = sha(__file__)
    out = Path(a.out)
    need(not out.exists(), 'Choose a fresh projection file')
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, report)
    for s in report['scenarios']:
        print('%-46s v7 %8.0f s (%s)  v5/v6 %8.0f s (%s)' % (
            s['label'], s['v7_shared_screen_seconds'], 'PASS' if s['v7_passes_screen'] else 'FAIL',
            s['v5_v6_separate_screen_seconds'], 'PASS' if s['v5_v6_passes_screen'] else 'FAIL'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    st = sub.add_parser('stage-assets')
    st.add_argument('--checkpoint', action='append', required=True, help='NAME=PATH (repeatable)')
    st.add_argument('--package', required=True, help='NEW package directory (published v4-v10 packages are refused)')
    st.add_argument('--dataset', required=True, help='NEW private owner/slug (published v4-v10 slugs are refused)')
    st.add_argument('--copy', action='store_true', help='copy instead of hard-linking')
    nb = sub.add_parser('notebook')
    nb.add_argument('--release', required=True)
    nb.add_argument('--arm', action='append', required=True,
                    help='NAME or NAME=CK1+CK2[@K][:W] (repeatable, ordered; W for every arm or none)')
    nb.add_argument('--assets', action='append', default=[],
                    help='DATASET=PACKAGE (repeatable; default %s)' % DEFAULT_ASSETS)
    nb.add_argument('--out', required=True)
    nb.add_argument('--model-seconds', action='append', default=[],
                    help='ARM=S seconds per model per study (default: by architecture, %r)' % v6.MODEL_SECONDS)
    nb.add_argument('--expected-studies', type=int, default=1300)
    nb.add_argument('--docker-from', default=str(v6.DOCKER_PULL))
    pj = sub.add_parser('project')
    pj.add_argument('--studies', type=int, default=1300)
    pj.add_argument('--out', required=True)
    for mode in ['run', 'worker']:
        action = sub.add_parser(mode)
        for key in ['manifest', 'sha256', 'assets', 'competition', 'opencv']:
            action.add_argument('--' + key, required=True)
        action.add_argument('--asset-root', action='append', required=True, help='DATASET=PATH (repeatable)')
        if mode == 'run':
            action.add_argument('--base', required=True)
            action.add_argument('--final', required=True)
            action.add_argument('--started', type=float, required=True)
            action.add_argument('--bootstrap-max-seconds', type=float, default=600.0)
        else:
            action.add_argument('--work', required=True)
    cb = sub.add_parser('combine')
    for key in ['manifest', 'sha256', 'base', 'competition', 'out']:
        cb.add_argument('--' + key, required=True)
    cb.add_argument('--started', type=float, required=True)
    a = parser.parse_args()
    {'stage-assets': stage_assets, 'notebook': notebook, 'project': project, 'run': run, 'worker': worker,
     'combine': combine}[a.action](a)


if __name__ == '__main__':
    main()
