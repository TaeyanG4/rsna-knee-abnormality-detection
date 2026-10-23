"""EXP111 test-only release v6 = reviewed EXP104 release v5 + a RadImageNet ResNet50 (centre plane x3) arm kind (H-B-31).

  # 1. stage checkpoints (copy) and the centre-plane input code into a NEW private assets package (dataset v8)
  python tools/exp111_release_v6.py stage-assets --copy --dataset taeyangg4/knee-release-assets-v8 \
      --package kaggle/packages/KNEE_RELEASE_ASSETS_V8 --checkpoint coat_f0=artifacts/experiments/EXP094L_B8_SEED42/model_best.pt \
      ... --checkpoint rad_f0=.local/kaggle-out/knee-run-v3-exp107-radr50-5fold/EXP107_RADR50_DREAD_F0/model_best.pt ...
  # 2. generate a Kaggle kernel directory (never pushed here)
  python tools/exp111_release_v6.py notebook --release coat5rad5 --arm coat5=coat_f0+...+coat_f4@36:1 \
      --arm rad5=rad_f0+...+rad_f4@36:1 --out .local/exp111-release-v6/coat5rad5

Changes against v5 (everything else is v5 verbatim; names exp104/v5 -> exp111/v6):
  A. Arms carry an input policy. 'triplet' = the v5 input (cache planes c-1, c, c+1 via the frozen
     raptor_runtime.prepare_input; norm imagenet or none). 'center3' = the EXP107 RadImageNet input: the same window
     centres (select_centers), each window the centre cache plane repeated as the three channels, normalisation
     'radimagenet' (x in [0, 1] -> 2x - 1); only arch resnet50 with init radimagenet (RadImageNet ResNet50, sha256
     RADIMAGENET_SHA256) and only norm radimagenet. A checkpoint without an input_policy field is 'triplet'.
  B. The centre-plane input is the training code itself: training/src/raptor_radimagenet.py and raptor_data.py (sha256
     CODE_FILES, the hashes bound in every EXP107 checkpoint's provenance) are staged under code/ in the assets package,
     bound by the assets manifest, verified before import, and imported after the frozen EXP089 serving copies (so
     raptor_common stays the frozen copy). For a center3 arm the worker swaps raptor_runtime.prepare_input for
     center3_prepare_input only while predict runs; predict itself (decode once per study, input hashes, finite checks,
     no precision fallback) is the unchanged frozen copy.
  C. model_seconds_per_study for resnet50 = 0.25 s (see MODEL_SECONDS); the time screen formula is unchanged.
  D. Defaults: dataset taeyangg4/knee-release-assets-v8, package kaggle/packages/KNEE_RELEASE_ASSETS_V8, kernel
     taeyangg4/knee-exp111-<release>-v6; the v4-v7 release-assets slugs and packages are refused.
  E. Staging fails closed when a source checkpoint's sibling output-inventory.json records another sha256.

Changes of v5 against v4 (kept):
  1. The new-assets dataset slug and the local package directory are parameters (--dataset/--package, defaults
     taeyangg4/knee-release-assets-v5 and kaggle/packages/KNEE_RELEASE_ASSETS_V5); the slug is recorded in the assets
     manifest, the release manifest and the kernel's dataset_sources, and the notebook mounts it by that slug. The v4
     dataset slug and package directory are refused, so the v4 assets stay untouched.
  2. Per-arm rank weights: `--arm NAME=CK1+CK2[@K][:W]`; W is optional and defaults to equal weights (give W for every
     arm or for none). combine = per target, average-tie percentile rank over the full cohort (EXP098), weighted mean
     (sum_i W_i*r_i)/(sum_i W_i), i.e. weights normalised to 1; when all weights are equal the v4/EXP098 equal mean is
     used verbatim, so equal weights reproduce v4 bit-for-bit. Raw and normalised weights are recorded in the
     release manifest, the arm results, the worker receipts and the combine result.
  3. model_seconds_per_study is chosen per arm from the members' architecture (CoAtNet-2 224 0.35 s, ResNet34 0.15 s
     per model per study on a T4; an unknown architecture needs --model-seconds NAME=S) and recorded per arm; the
     screen is unchanged: sum over arms of 1.3*(6.52503319258 + n_models*model_seconds_per_study)*N + 600 <= 30600.
  4. A non-finite member logit fails closed (RuntimeError) instead of being clamped by the mean_prob wrapper.
  5. The recorded push command has no --timeout (D-B-015 item 13) and requests the T4 accelerator.

Arms. A release manifest lists one or more arms; each arm is
  {name, arch, res, k_eval, contract, norm, window_policy, input_policy, aggregator: 'mean_prob',
   model_seconds_per_study, weight, checkpoints: [{name, path: 'checkpoints/<name>.pt', sha256, bytes, meta}]}.
In the worker an arm's models are wrapped in one nn.Module returning logit(mean_i sigmoid(model_i(x))) with
probabilities clamped to [1e-6, 1-1e-6], so the unchanged raptor_runtime.predict decodes every study once per arm.

Safety (v4): frozen hashes of the EXP089 assets (wheels, serving code, supervisor), of the assets manifest and every
served checkpoint; dependency versions; Linux CPython 3.12; one visible T4 per worker; test-contract check;
launch_budget anchored at the first-cell clock where a later arm must still fit the conservative projection of itself
and every arm after it; resource validation; full ordered output; atomic publication; receipts and evidence.
stage-assets/notebook read local files only: they never push kernels, publish datasets or submit.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'exp111-test-release-v6'
OWNER = 'taeyangg4'
COMPETITION = 'rsna-knee-abnormality-detection'
OLD_DATASET = 'taeyangg4/knee-exp089-recipe-transfer-v1'
OLD_PROTOCOL = 'exp089-t4-protocol.json'
ASSETS_PROTOCOL_SHA = '3e0e5529191570955b3229458adf511073401dc354de04b33164988f830687a8'
OLD_LOCAL = ROOT / '.local/exp089-recipe-transfer/t4-v3/dataset'
ACCEPTANCE = ROOT / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-v2-root-acceptance.json'
NEW_DATASET = 'taeyangg4/knee-release-assets-v8'          # default; --dataset makes it a recorded parameter
NEW_MANIFEST = 'assets-manifest.json'
ASSETS_SCHEMA = 'exp111-release-assets-v6'
PACKAGE = ROOT / 'kaggle/packages/KNEE_RELEASE_ASSETS_V8'   # default; --package
# v4 (EXP100), v5/v6 (EXP104) and v7 (H-B-28 relabel automation) belong to other releases: never (re)staged here
REFUSED_DATASETS = tuple('taeyangg4/knee-release-assets-v%d' % v for v in range(4, 8))
REFUSED_PACKAGES = tuple(ROOT / ('kaggle/packages/KNEE_RELEASE_ASSETS_V%d' % v) for v in range(4, 8))
OPENCV_DATASET = 'mattiaangeli/opencv-python-headless-4120088-x86'
DATASET_SLUG = re.compile(r'[a-z0-9][a-z0-9-]{0,38}/[a-z0-9][a-z0-9-]{4,49}')
KERNEL_PREFIX, KERNEL_SUFFIX, WORK_PREFIX = 'knee-exp111-', '-v6', 'exp111-'
# Python 3.12 image of the known-good B224 release v3 kernel (COMPLETE, Public3 PASS 2026-10-06); read from
# `kaggle kernels pull taeyangg4/knee-exp089-b224-release-v3 -p .local/exp100-release-v4/docker-pin/b224-release-v3 -m`.
# The Kaggle default image moved to Python 3.13 on 2026-10-06, which fails the CPython 3.12 check: pinning is mandatory.
DOCKER_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
DOCKER_SOURCE_KERNEL = 'taeyangg4/knee-exp089-b224-release-v3'
DOCKER_PULL = ROOT / '.local/exp100-release-v4/docker-pin/b224-release-v3/kernel-metadata.json'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
TIME = {'decode_seconds_per_study': 6.52503319258, 'inflation': 1.3, 'arm_fixed_seconds': 300,
        'screen_fixed_seconds': 600, 'screen_seconds': 30600, 'full_seconds': 32100,
        'public3_seconds': 600, 'finalization_reserve_seconds': 30,
        'platform_public_seconds': 900, 'platform_full_seconds': 32400}
MODEL_SECONDS_DEFAULT = 0.35
# Seconds per model per study on a T4, by architecture (conservative screen constants, not measurements of serving):
# CoAtNet-2 224 keeps the v4 value (Kaggle T4 K24 out-of-fold passes of EXP094T4 seed43/geoaug took 123-137 s for 888
# studies including cache loading = 0.14-0.15 s/study, so about 0.22 s at K36); ResNet34 0.15 s is 1.6x the 0.091 s/study
# measured for the EXP101 ResNet34 K36 out-of-fold pass on a Kaggle T4 (864 studies in 79 s, EXP101 fold logs).
# v6: resnet50 (RadImageNet, centre plane x3) 0.25 s is 3.6-4.2x the 0.060-0.069 s/study measured for the EXP107 K36
# out-of-fold passes on a Kaggle T4 (fold logs: 888 studies in 61 s, 864 in 52 s, 863 in 52 s; FP16, batch 4, cache input).
MODEL_SECONDS = {'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k': MODEL_SECONDS_DEFAULT, 'resnet34.a1_in1k': 0.15, 'resnet50': 0.25}
MODEL_SECONDS_BASIS = {
    'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k': 'v4/v5 constant; EXP094T4 K24 OOF passes 0.14-0.15 s/study on a T4',
    'resnet34.a1_in1k': 'v5 constant; 1.6x the EXP101 K36 OOF pass 0.091 s/study on a T4',
    'resnet50': 'v6 constant; 3.6-4.2x the EXP107 K36 OOF passes 0.060-0.069 s/study on a T4 (FP16, batch 4)'}
PROB_EPS = 1e-6
RSS = 8 * 2**30
OUTPUT = 1 * 2**30
VERSIONS = ('2.10.0+cu128', '1.0.19', '4.12.0', '3.0.2', '3.2.6')
SERVING = ['config/workspace.toml', 'b_rn3f_serving.py', 'process_resource_receipt.py',
           *['training/src/' + n for n in ['raptor_common.py', 'raptor_runtime.py', 'raptor_dicom.py', 'raptor_ensemble.py']]]
TRAINING_CODE_BOUND = ['raptor_common.py', 'raptor_runtime.py']
META_KEYS = ['arch', 'res', 'lab', 'contract', 'k_eval', 'norm', 'window_policy', 'epoch', 'tag', 'amp_train', 'amp_eval',
             'input_policy', 'init', 'init_weights_sha256']
REQUIRED_META = ['arch', 'res', 'lab', 'contract', 'k_eval', 'norm', 'window_policy']
ARM_MATCH = ['arch', 'res', 'contract', 'norm', 'window_policy', 'input_policy']
ARM_KEYS = {'name', 'arch', 'res', 'k_eval', 'contract', 'norm', 'window_policy', 'input_policy', 'aggregator',
            'model_seconds_per_study', 'weight', 'checkpoints'}
WINDOW_POLICIES = ('legacy', 'global-valid', 'within-slot-valid')
# input policy -> accepted normalisations; 'triplet' is every pre-v6 checkpoint (no input_policy field)
INPUT_POLICIES = {'triplet': ('imagenet', 'none'), 'center3': ('radimagenet',)}
DEFAULT_INPUT_POLICY = 'triplet'
CENTER3_ARCH = 'resnet50'
CENTER3_INIT = 'radimagenet'
RADIMAGENET_SHA256 = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'   # EXP107 init, the anchor's encoder
# The centre-plane input code shipped in the assets package: EXP107's own training modules, byte for byte
# (raptor_data.py = code_sha256['raptor_data.py'] and raptor_radimagenet.py = radimagenet_module_sha256 in every EXP107
# checkpoint's provenance).
CODE_DIR = 'code'
CODE_FILES = {'raptor_data.py': '0b922ea0359962a2bb6531d5630639f2c9002abdb80fe1353991087d6bcdea01',
              'raptor_radimagenet.py': '8b368bcfc104f26615934d525dbf37786d1f95c1532601ce8cf4701faa3b6112'}
CODE_SOURCE = ROOT / 'training/src'
CKPT_KEYS = {'name', 'path', 'sha256', 'bytes', 'meta'}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}')
RELEASE_NAME = re.compile(r'[a-z0-9][a-z0-9-]{0,39}')
SOURCE_CHUNK = 4000          # characters per embedded base85 line
MAX_NOTEBOOK_LINE = 30000    # Kaggle failed to start a notebook with a 66,897-character line; 32,646 ran
SHA = re.compile(r'[0-9a-f]{64}')
ARM_SPEC = re.compile(r'(?P<name>[A-Za-z0-9][A-Za-z0-9_-]{0,47})'
                      r'(?:=(?P<members>[A-Za-z0-9][A-Za-z0-9_-]{0,47}(?:\+[A-Za-z0-9][A-Za-z0-9_-]{0,47})*))?'
                      r'(?:@(?P<k>[1-9][0-9]{0,3}))?'
                      r'(?::(?P<w>[0-9]{1,4}(?:\.[0-9]{1,8})?|\.[0-9]{1,8}))?')


# ------------------------------------------------------------------ shared helpers (verbatim v3 semantics)
def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False))


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    need(len(rows) > 1 and len(rows[0]) == len(set(rows[0])), 'Empty or duplicate header')
    need(all(len(r) == len(rows[0]) for r in rows[1:]), 'CSV width differs')
    return rows[0], rows[1:]


def safe_ids(values):
    ids = list(values)
    need(bool(ids) and all(isinstance(u, str) and u and not any(c in u for c in
         ['/', '\\', ':', '\x00', '..']) for u in ids), 'Unsafe/empty UID')
    need(len(ids) == len(set(ids)), 'Duplicate UID')
    return ids


def test_contract(competition):
    competition = Path(competition)
    need((competition / 'test_series').is_dir(), 'Missing test_series directory')
    th, tr = table(competition / 'test.csv')
    need(UID in th, 'Test UID absent')
    test_ids = safe_ids(r[th.index(UID)] for r in tr)
    sh, sr = table(competition / 'sample_submission.csv')
    need(sh == [UID, *TARGETS], 'Official12 column order differs')
    ids = safe_ids(r[0] for r in sr)
    need(set(ids) == set(test_ids), 'Sample/test UID membership differs')
    rh, rr = table(competition / 'test_series.csv')
    need(UID in rh and 'SeriesInstanceUID' in rh, 'Test series schema differs')
    need({r[rh.index(UID)] for r in rr} <= set(ids), 'Unexpected series study')
    safe_ids(dict.fromkeys(r[rh.index('SeriesInstanceUID')] for r in rr))
    return {'ids': ids, 'studies': len(ids), 'columns': sh,
            'metadata_sha256': {n: sha(competition / n) for n in
                                ['test.csv', 'test_series.csv', 'sample_submission.csv']}}


def csv_check(path, ids):
    header, rows = table(path)
    need(header == [UID, *TARGETS] and [r[0] for r in rows] == ids, 'Incomplete/reordered output')
    need(all(math.isfinite(float(x)) and 0 <= float(x) <= 1 for r in rows for x in r[1:]),
         'Nonfinite/out-of-range output')
    return sha(path)


def validate_resource(s, argv, worker_pid, timeout):
    need(s['schema'] == 'process-resource-receipt-v1' and s['status'] == 'COMPLETE'
         and s['failure'] is None and s['child_exit_code'] == s['child_exit_code_before_cleanup']
         == s['wrapper_exit_code'] == 0, 'Child/resource failed')
    need(s['argv'] == argv and s['child_identity']['pid'] == worker_pid, 'Child identity differs')
    need(all(not s['cleanup'][k] for k in ['actions', 'errors', 'survivors']), 'Non-natural cleanup')
    need(s['retries'] == 0 and s['timeout_seconds'] == timeout and 0 <= s['elapsed_seconds'] < timeout,
         'Deadline/retry differs')
    need(s['sample_count'] > 0 and 0 < s['sampled_tree_rss_max_bytes'] <= RSS
         and s['failed_sample_attempts'] == s['access_failure_count'] == 0
         and not s['access_failures'], 'Incomplete/excess resource observation')
    telemetry = {'cgroup_telemetry_complete': bool(s.get('cgroup_telemetry_complete')), 'snapshots': {}}
    for key in ['cgroup_start', 'cgroup_end']:
        snapshot = s.get(key) or {}
        v = snapshot.get('values') or {}
        events = v.get('memory.events') or {}
        telemetry['snapshots'][key] = {'errors': snapshot.get('errors'), 'memory.events': events,
                                       'memory.peak': v.get('memory.peak'), 'memory.max': v.get('memory.max')}
        # Fatal only on a real OOM; max/high count reclaim pressure from page cache and are telemetry.
        need(all(events.get(k, 0) == 0 for k in ['oom', 'oom_kill']), 'Memory OOM event')
    return telemetry


def output_bytes(work):
    # A single stat avoids is_file/stat replacement races; other errors propagate.
    total = 0
    for path in Path(work).rglob('*'):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def publish(raw, destination, expected_sha, deadline, final_check):
    """Own temp/final transaction (atomic os.replace); rollback also covers post-replace failures."""
    destination = Path(destination)
    need(not destination.exists(), 'Preserve existing final CSV')
    temporary = destination.with_name(destination.name + '.exp111-tmp')
    need(not temporary.exists(), 'Preserve existing temp CSV')
    published = False
    try:
        shutil.copyfile(raw, temporary)
        need(sha(temporary) == expected_sha and time.monotonic() < deadline, 'Copy/deadline failed')
        final_check()
        need(not destination.exists(), 'Final CSV appeared during publication')
        os.replace(temporary, destination)
        published = True
        need(sha(destination) == expected_sha and time.monotonic() < deadline, 'Final/deadline failed')
        final_check()
    except BaseException:
        if published and destination.is_file() and sha(destination) == expected_sha:
            destination.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


# ------------------------------------------------------------------ time policy
def arm_projection(n, n_models, model_seconds):
    """Conservative seconds for one arm: one decode per study plus every member's forward pass."""
    need(type(n) is int and n > 0 and type(n_models) is int and n_models > 0, 'Positive counts required')
    need(type(model_seconds) in (int, float) and math.isfinite(model_seconds) and model_seconds >= 0,
         'Invalid model seconds')
    return TIME['inflation'] * (TIME['decode_seconds_per_study'] + n_models * model_seconds) * n


def operational_plan(n, arms):
    need(type(n) is int and n > 0, 'Positive count required')
    need(isinstance(arms, list) and arms and len({a['name'] for a in arms}) == len(arms), 'Unique arms required')
    per_arm = {a['name']: arm_projection(n, len(a['checkpoints']), a['model_seconds_per_study']) for a in arms}
    projected = sum(per_arm.values()) + TIME['screen_fixed_seconds']
    need(projected <= TIME['screen_seconds'], 'N-dependent time screen failed')
    return {'studies': n, 'arm_projected_seconds': per_arm, 'projected_seconds': projected,
            'internal_seconds': TIME['public3_seconds'] if n <= 3 else TIME['full_seconds'],
            'formula': 'sum over arms of inflation*(decode_seconds_per_study + n_models*model_seconds_per_study)*N'
                       ' + screen_fixed_seconds <= screen_seconds',
            'scope': 'Conservative engineering scenario, not hidden runtime proof; later arms may see warmed files'}


def max_studies(arms):
    per_study = sum(arm_projection(1, len(a['checkpoints']), a['model_seconds_per_study']) for a in arms)
    return int((TIME['screen_seconds'] - TIME['screen_fixed_seconds']) // per_study)


def launch_projection(plan, arms, name):
    """Seconds still needed when arm `name` launches: itself and every later arm, plus one arm's fixed cost."""
    names = [a['name'] for a in arms]
    need(name in names, 'Unknown arm: ' + str(name))
    return sum(plan['arm_projected_seconds'][x] for x in names[names.index(name):]) + TIME['arm_fixed_seconds']


def launch_budget(started, now, n, projected_seconds, bootstrap_max=600.0):
    """v3.1 semantics: deadline anchored to the first-cell clock; a late launch must still fit its projection."""
    need(type(n) is int and n > 0, 'Positive count required')
    internal = TIME['public3_seconds'] if n <= 3 else TIME['full_seconds']
    need(math.isfinite(started) and math.isfinite(bootstrap_max) and 0 < bootstrap_max <= internal
         and 0 <= now - started < bootstrap_max, 'Invalid bootstrap clock')
    deadline = started + internal
    timeout = deadline - now - TIME['finalization_reserve_seconds']
    need(timeout > 0, 'Budget failed')
    if n > 3:
        need(math.isfinite(projected_seconds) and timeout >= projected_seconds,
             'Remaining time below the conservative arm projection')
    return deadline, timeout


# ------------------------------------------------------------------ arms and manifests
def parse_arm_spec(text):
    """NAME | NAME=CK1+CK2[@K][:W] | NAME@K -> (arm name, member names, k_eval or None, rank weight or None)."""
    m = ARM_SPEC.fullmatch(text or '')
    need(m is not None, 'Arm spec must be NAME or NAME=CK1+CK2[@K][:W]: ' + repr(text))
    members = m.group('members').split('+') if m.group('members') else [m.group('name')]
    need(len(set(members)) == len(members), 'Duplicate ensemble member in ' + repr(text))
    weight = float(m.group('w')) if m.group('w') else None
    need(weight is None or (math.isfinite(weight) and weight > 0), 'Rank weight must be positive: ' + repr(text))
    return m.group('name'), members, int(m.group('k')) if m.group('k') else None, weight


def valid_weight(w):
    return type(w) is float and math.isfinite(w) and w > 0


def rank_weights(arms):
    """Normalised rank weights W_i / sum(W) (left-to-right float sum), keyed by arm name, in arm order."""
    need(all(valid_weight(a.get('weight')) for a in arms), 'Rank weight must be a positive finite float')
    total = 0.0
    for a in arms:
        total += a['weight']
    return {a['name']: a['weight'] / total for a in arms}


def input_policy_of(meta):
    """A checkpoint's input policy; pre-v6 checkpoints carry no input_policy field and use the cache triplet."""
    policy = meta.get('input_policy', DEFAULT_INPUT_POLICY)
    need(isinstance(policy, str) and policy in INPUT_POLICIES, 'Unknown input policy: ' + repr(policy))
    return policy


def derived_meta(meta):
    """Checkpoint metadata with the effective input policy filled in (the staged/loaded record itself is not changed)."""
    return {**meta, 'input_policy': input_policy_of(meta)}


def input_contract(d, member=False):
    """Allowed (input policy, norm, arch[, init]) combinations; `d` is an arm or a derived member meta."""
    policy = d.get('input_policy')
    need(isinstance(policy, str) and policy in INPUT_POLICIES, 'Unknown input policy: ' + repr(policy))
    need(d.get('norm') in INPUT_POLICIES[policy] and d.get('window_policy') in WINDOW_POLICIES,
         'Unsupported norm/window policy for input policy ' + policy)
    if policy == 'center3':
        need(d.get('arch') == CENTER3_ARCH, 'Input policy center3 is only served for arch ' + CENTER3_ARCH)
        if member:
            need(d.get('init') == CENTER3_INIT and d.get('init_weights_sha256') == RADIMAGENET_SHA256,
                 'center3 member must be initialised from the registered RadImageNet ResNet50')
    return policy


def validate_arms(arms):
    need(isinstance(arms, list) and arms, 'At least one arm required')
    need(all(isinstance(a, dict) for a in arms), 'Arm must be an object')
    need(len({a.get('name') for a in arms}) == len(arms), 'Duplicate arm name')
    for arm in arms:
        need(set(arm) == ARM_KEYS, 'Arm fields differ')
        need(isinstance(arm['name'], str) and NAME.fullmatch(arm['name']) and arm['name'] != 'combine',
             'Unsafe arm name')   # 'combine' is the rank-combination work directory
        need(isinstance(arm['arch'], str) and arm['arch'] and isinstance(arm['contract'], str) and arm['contract'],
             'Arch/contract required')
        need(type(arm['res']) is int and arm['res'] > 0 and type(arm['k_eval']) is int and arm['k_eval'] > 0,
             'res/k_eval must be positive integers')
        need(arm['aggregator'] == 'mean_prob', 'Only aggregator mean_prob is supported')
        input_contract(arm)
        m = arm['model_seconds_per_study']
        need(type(m) in (int, float) and math.isfinite(m) and m > 0, 'model_seconds_per_study must be positive')
        need(valid_weight(arm['weight']), 'Rank weight must be a positive finite float')
        cks = arm['checkpoints']
        need(isinstance(cks, list) and cks and all(isinstance(c, dict) and set(c) == CKPT_KEYS for c in cks),
             'Arm checkpoints malformed')
        for key in ('name', 'path', 'sha256'):
            need(len({c[key] for c in cks}) == len(cks), 'Duplicate checkpoint in arm ' + arm['name'])
        for c in cks:
            need(isinstance(c['name'], str) and NAME.fullmatch(c['name']) and c['path'] == 'checkpoints/' + c['name'] + '.pt'
                 and isinstance(c['sha256'], str) and SHA.fullmatch(c['sha256']) and type(c['bytes']) is int
                 and c['bytes'] > 0, 'Unsafe checkpoint record')
            meta = c['meta']
            need(isinstance(meta, dict) and all(k in meta for k in REQUIRED_META) and meta['lab'] == TARGETS,
                 'Member metadata differs from arm ' + arm['name'])
            dm = derived_meta(meta)
            need(all(dm[k] == arm[k] for k in ARM_MATCH), 'Member metadata differs from arm ' + arm['name'])
            input_contract(dm, member=True)
    return arms


def build_arm(name, members, k_eval, assets_manifest, model_seconds, weight=1.0):
    entries = []
    for c in members:
        need(c in assets_manifest['checkpoints'], 'Unknown staged checkpoint: ' + c)
        e = assets_manifest['checkpoints'][c]
        entries.append({'name': c, 'path': e['path'], 'sha256': e['sha256'], 'bytes': e['bytes'], 'meta': e['meta']})
    derived = [derived_meta(e['meta']) for e in entries]
    first = derived[0]
    for key in ARM_MATCH:
        need(all(d[key] == first[key] for d in derived), 'Ensemble members differ in ' + key)
    if k_eval is None:
        trained = {e['meta']['k_eval'] for e in entries}
        need(len(trained) == 1, 'Members were trained at different k_eval; give NAME=...@K')
        k_eval = trained.pop()
    arm = {'name': name, **{k: first[k] for k in ARM_MATCH}, 'k_eval': k_eval, 'aggregator': 'mean_prob',
           'model_seconds_per_study': model_seconds, 'weight': weight, 'checkpoints': entries}
    validate_arms([arm])
    return arm


def arm_model_seconds(arm_name, arch, overrides):
    """Per-arm seconds per model per study: an explicit --model-seconds NAME=S, else the architecture default."""
    if arm_name in overrides:
        return overrides[arm_name], 'override --model-seconds %s=%r' % (arm_name, overrides[arm_name])
    need(arch in MODEL_SECONDS, 'No model_seconds_per_study default for %s; give --model-seconds %s=S' % (arch, arm_name))
    return MODEL_SECONDS[arch], 'architecture default for %s (%s)' % (arch, MODEL_SECONDS_BASIS[arch])


def arm_by_name(release, name):
    found = [a for a in release['arms'] if a['name'] == name]
    need(len(found) == 1, 'Unknown arm: ' + str(name))
    return found[0]


def checked_manifest(path, digest):
    need(sha(path) == digest, 'Release manifest changed')
    p = read(path)
    need(p.get('schema') == SCHEMA and p.get('time_policy') == TIME and p.get('old_asset_protocol_sha256') == ASSETS_PROTOCOL_SHA
         and p.get('RSS_bytes') == RSS and p.get('output_bytes') == OUTPUT and p.get('prob_clamp') == PROB_EPS
         and p.get('retries') == 0, 'Release policy differs')
    need(isinstance(p.get('new_assets_manifest_sha256'), str) and SHA.fullmatch(p['new_assets_manifest_sha256'])
         and isinstance(p.get('release'), str) and RELEASE_NAME.fullmatch(p['release']), 'Release binding missing')
    need(sha(__file__) == p.get('release_source_sha256'), 'Executing release source changed')
    validate_arms(p.get('arms'))
    check_dataset(p.get('new_asset_dataset'))
    need(p.get('rank_weights') == rank_weights(p['arms']), 'Recorded rank weights differ from the arms')
    return p


def check_dataset(dataset):
    """New-assets dataset slug: owner/slug, never the old assets, OpenCV or the v4-v7 release-assets datasets."""
    need(isinstance(dataset, str) and DATASET_SLUG.fullmatch(dataset)
         and dataset not in (OLD_DATASET, OPENCV_DATASET, *REFUSED_DATASETS),
         'Unsafe new-assets dataset slug: ' + repr(dataset))
    return dataset


def verify_old_assets(root, opencv=None):
    root = Path(root)
    need(sha(root / OLD_PROTOCOL) == ASSETS_PROTOCOL_SHA, 'Frozen assets protocol changed')
    old = read(root / OLD_PROTOCOL)
    for name in [*SERVING, *[r['path'] for r in old['wheels']]]:
        need(sha(root / name) == old['files'][name], 'Selected immutable asset changed: ' + name)
    if opencv is not None:
        need(sha(opencv) == old['opencv_sha256'], 'OpenCV wheel changed')
    return old


def code_record():
    """The assets-manifest record of the shipped centre-plane input code (fixed by CODE_FILES)."""
    return {n: {'path': CODE_DIR + '/' + n, 'sha256': s} for n, s in CODE_FILES.items()}


def verify_code(root, am):
    """The shipped input code must be recorded, present and byte-identical to the EXP107 training modules."""
    root = Path(root)
    need(am.get('code') == code_record(), 'Input code record differs from CODE_FILES')
    for n, s in CODE_FILES.items():
        rel = CODE_DIR + '/' + n
        need(am['files'].get(rel) == s, 'Input code not bound by the assets manifest: ' + rel)
        need((root / rel).is_file() and sha(root / rel) == s, 'Input code changed: ' + rel)
    return code_record()


def verify_new_assets(release, root, arms):
    """Bind the served checkpoints and input code: manifest digest, presence, and sha256 of every member/code file."""
    root = Path(root)
    need((root / NEW_MANIFEST).is_file(), 'Missing new assets manifest')
    need(sha(root / NEW_MANIFEST) == release['new_assets_manifest_sha256'], 'New assets manifest changed')
    am = read(root / NEW_MANIFEST)
    need(am.get('schema') == ASSETS_SCHEMA and am.get('dataset') == check_dataset(release.get('new_asset_dataset')),
         'New assets schema differs')
    verify_code(root, am)
    for arm in arms:
        for c in arm['checkpoints']:
            staged = am['checkpoints'].get(c['name']) or {}
            need(am['files'].get(c['path']) == c['sha256'] and staged.get('sha256') == c['sha256']
                 and staged.get('path') == c['path'] and staged.get('meta') == c['meta'],
                 'Checkpoint not bound by the assets manifest: ' + c['name'])
            path = root / c['path']
            need(path.is_file(), 'Missing checkpoint: ' + c['path'])
            need(sha(path) == c['sha256'], 'Checkpoint sha256 differs: ' + c['path'])
    return am


def checkpoint_check(meta, arm, ck):
    """Loaded metadata must equal the arm and the staged record; k_eval may differ from training (both kept)."""
    meta = {k: jsonable(v) for k, v in meta.items() if k in META_KEYS}
    dm = derived_meta(meta)
    expected = {'lab': TARGETS, **{k: arm[k] for k in ARM_MATCH}}
    bad = sorted(k for k, v in expected.items() if dm.get(k) != v)
    need(not bad, 'Checkpoint metadata differs from arm %s: %s' % (ck['name'], bad))
    input_contract(dm, member=True)
    drift = sorted(k for k in set(ck['meta']) | set(meta) if meta.get(k) != ck['meta'].get(k))
    need(not drift, 'Checkpoint metadata differs from staged record %s: %s' % (ck['name'], drift))
    return {'name': ck['name'], 'path': ck['path'], 'sha256': ck['sha256'], 'tag': meta.get('tag'),
            'epoch': meta.get('epoch'), 'amp_train': meta.get('amp_train'), 'input_policy': dm['input_policy'],
            'k_eval_checkpoint': meta.get('k_eval'), 'k_eval_release': arm['k_eval']}


# ------------------------------------------------------------------ ensemble and rank combination
def mean_prob_module(models):
    """One module: forward = logit(mean_i sigmoid(model_i(x))), probabilities clamped to [1e-6, 1-1e-6].

    v5: a non-finite member logit (nan or +-inf) raises RuntimeError instead of being absorbed by sigmoid/clamp."""
    import torch
    need(len(models) > 0, 'Empty ensemble')

    class MeanProbEnsemble(torch.nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)

        def forward(self, x):
            logits = [m(x) for m in self.members]
            for i, z in enumerate(logits):
                if not bool(torch.isfinite(z).all()):
                    raise RuntimeError('nonfinite member logit (member %d)' % i)
            with torch.autocast(x.device.type, enabled=False):
                p = torch.stack([torch.sigmoid(z.float()) for z in logits]).mean(0)
                p = p.clamp(PROB_EPS, 1.0 - PROB_EPS)
                return torch.log(p) - torch.log1p(-p)

    return MeanProbEnsemble(models)


# ------------------------------------------------------------------ v6: input policies
def center3_prepare_input(corpus, uid, k_eval, res, norm, window_policy='legacy'):
    """EXP107 evaluation input = CenterSliceWindows.__getitem__ with train=False, from any corpus interface.

    Same window centres as the triplet input (select_centers on the corpus mask, no rng), each window the centre plane
    repeated as the three channels (raptor_radimagenet.center3_windows_from), then normalise(x, 'radimagenet') = 2x - 1.
    Signature and contiguous float32 output follow raptor_runtime.prepare_input, so the frozen predict can call it."""
    import numpy as np
    from raptor_common import select_centers
    from raptor_radimagenet import center3_windows_from, normalise
    need(norm in INPUT_POLICIES['center3'], 'center3 input requires norm radimagenet, got ' + repr(norm))
    mask = corpus.mask(uid)
    centers = select_centers(mask, corpus.depth, k_eval, corpus.spec, window_policy, None)
    x = normalise(center3_windows_from(corpus.volume(uid), centers, res), norm)
    need(x.dtype == np.float32 and bool(np.isfinite(x).all()), 'nonfinite or non-float32 model input')
    return np.ascontiguousarray(x, dtype=np.float32)


@contextlib.contextmanager
def serving_input(runtime_module, input_policy):
    """For a center3 arm, swap the frozen runtime's prepare_input for center3_prepare_input while predict runs."""
    need(input_policy in INPUT_POLICIES, 'Unknown input policy: ' + repr(input_policy))
    if input_policy == 'triplet':
        yield 'raptor_runtime.prepare_input (frozen EXP089 serving copy)'
        return
    original = runtime_module.prepare_input
    runtime_module.prepare_input = center3_prepare_input
    try:
        yield 'exp111_release_v6.center3_prepare_input (raptor_radimagenet.center3_windows_from + normalise)'
    finally:
        runtime_module.prepare_input = original


def serve_arm(arm, new_root, corpus, ids, amp, device, state=None, log=print):
    """Worker inference for one arm: load every member (checked against arm and staged record), wrap them in the
    mean_prob module, run the frozen raptor_runtime.predict with the arm's input policy. `state['model']` keeps the
    wrapper for the caller's cleanup even when predict raises."""
    import raptor_runtime
    from raptor_common import load_checkpoint
    state = {} if state is None else state
    members, checks = [], []
    for ck in arm['checkpoints']:
        member, meta = load_checkpoint(Path(new_root) / ck['path'])
        checks.append(checkpoint_check(meta, arm, ck))
        members.append(member.eval().to(device))   # move now: CPU RSS holds at most one checkpoint
    model = state['model'] = mean_prob_module(members)
    with serving_input(raptor_runtime, arm['input_policy']) as input_function:
        probs, timing = raptor_runtime.predict(model, corpus, ids, arm['k_eval'], arm['res'], amp, device,
                                               norm=arm['norm'], batch=1, log=log, window_policy=arm['window_policy'])
    return probs, timing, checks, input_function


def average_rank(col):
    """EXP098 verbatim: average-tie percentile rank over the full cohort, ((i+j)/2+1)/n."""
    n = len(col); order = sorted(range(n), key=lambda i: col[i]); out = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and col[order[j + 1]] == col[order[i]]:
            j += 1
        v = ((i + j) / 2 + 1) / n
        for k in range(i, j + 1):
            out[order[k]] = v
        i = j + 1
    return out


def rank_mean(tables, weights=None):
    """Per target: weighted mean of each arm's average-tie percentile rank.

    weights None or all equal -> the v4/EXP098 equal mean verbatim (bit-identical); otherwise
    (sum_i W_i*r_i)/(sum_i W_i) with left-to-right float sums, which stays inside [0, 1] because rounding is monotone."""
    need(len(tables) >= 2 and len({len(t) for t in tables}) == 1 and len(tables[0]) > 0, 'Two or more equal arm tables')
    need(all(len(row) == len(TARGETS) for t in tables for row in t), 'Arm table width differs')
    weights = [1.0] * len(tables) if weights is None else list(weights)
    need(len(weights) == len(tables) and all(valid_weight(w) for w in weights), 'Invalid rank weights')

    def mean(values):
        total = 0.0   # plain left-to-right float addition: (a+b)/2 for two arms exactly as EXP098;
        for v in values:   # builtin sum() uses compensated summation since Python 3.12
            total += v
        return total / len(values)

    def weighted(values):
        num = den = 0.0
        for w, v in zip(weights, values):
            num += w * v
            den += w
        return num / den

    combine_one = mean if len(set(weights)) == 1 else weighted
    cols = []
    for t in range(len(TARGETS)):
        ranks = [average_rank([row[t] for row in tab]) for tab in tables]
        cols.append([combine_one(v) for v in zip(*ranks)])
    need(all(math.isfinite(v) and 0 <= v <= 1 for c in cols for v in c), 'Rank mean out of range')
    return cols


def load_probabilities(path, ids):
    header, rows = table(path)
    need(header == [UID, *TARGETS] and [r[0] for r in rows] == ids, 'Arm CSV differs')
    return [[float(x) for x in r[1:]] for r in rows]


def write_columns(path, ids, cols):
    """EXP098 byte format: csv.writer rows of repr(float)."""
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([UID, *TARGETS])
        for i, u in enumerate(ids):
            w.writerow([u, *[repr(cols[t][i]) for t in range(len(TARGETS))]])


# ------------------------------------------------------------------ Kaggle-side commands
def worker(a):
    p = checked_manifest(a.manifest, a.sha256)
    arm = arm_by_name(p, a.arm)
    assets, new, work = Path(a.assets), Path(a.new_assets), Path(a.work)
    old = verify_old_assets(assets, a.opencv)
    verify_new_assets(p, new, [arm])
    need(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 12), 'Linux CPython3.12 required')
    context = read(work / 'launch.json')
    contract = test_contract(a.competition)
    need(contract == context['test_contract'] and context['arm'] == arm['name'], 'Test metadata changed')
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
    r = {'status': 'FAIL', 'pid': os.getpid(), 'release': p['release'], 'arm': arm['name'],
         'release_manifest_sha256': a.sha256, 'new_assets_manifest_sha256': p['new_assets_manifest_sha256'],
         'new_asset_dataset': p['new_asset_dataset'], 'aggregator': arm['aggregator'],
         'n_models': len(arm['checkpoints']), 'k_eval': arm['k_eval'], 'rank_weight': arm['weight'],
         'rank_weight_normalised': p['rank_weights'][arm['name']],
         'cleanup_errors': [], 'training': 0, 'label_Gold_reads': 0, 'scoring': 0, 'retries': 0}
    r.update(input_policy=arm['input_policy'], norm=arm['norm'])
    source = None
    state = {}
    try:
        envdir = Path('/kaggle/temp') / ('exp111-release-env-' + arm['name'])
        need(not envdir.exists(), 'New dependency dir required')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target',
                        str(envdir), *[str(assets / x['path']) for x in old['wheels']], a.opencv], check=True)
        sys.path[:0] = [str(envdir), str(assets / 'training/src')]
        input_code = {}
        if arm['input_policy'] == 'center3':
            # v6: the EXP107 input modules, verified (again, right before import) and imported AFTER the frozen serving
            # copies so that raptor_common/raptor_runtime stay the frozen EXP089 files
            input_code = verify_code(new, read(new / NEW_MANIFEST))
            sys.path.insert(2, str(new / CODE_DIR))
        import numpy as np
        import pandas as pd
        import torch, timm, cv2, pydicom, gdcm
        need((torch.__version__, timm.__version__, cv2.__version__, pydicom.__version__, gdcm.Version.GetVersion())
             == VERSIONS, 'Completed T4 dependency versions required')
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
             'T4' in torch.cuda.get_device_name(0), 'One visible T4 required')
        from raptor_common import LAB, write_predictions, environment
        from raptor_runtime import configure_runtime
        from raptor_dicom import DirectoryDicomSource, DicomCorpus
        frozen = (assets / 'training/src').resolve()
        modules = {'raptor_common': frozen, 'raptor_runtime': frozen, 'raptor_dicom': frozen}
        if arm['input_policy'] == 'center3':
            import raptor_radimagenet, raptor_data   # noqa: F401  (bytes verified above)
            modules.update(raptor_radimagenet=(new / CODE_DIR).resolve(), raptor_data=(new / CODE_DIR).resolve())
        need(all(Path(sys.modules[m].__file__).resolve().parent == d for m, d in modules.items()),
             'Serving modules were not imported from the verified asset directories')
        runtime = configure_runtime('fp16', 'cuda', 'strict')
        need(LAB == TARGETS, 'Target order changed')
        rows = pd.read_csv(Path(a.competition) / 'test_series.csv', dtype={UID: str, 'SeriesInstanceUID': str})
        source = DirectoryDicomSource(Path(a.competition) / 'test_series')
        corpus = DicomCorpus.from_records(contract['ids'], rows.to_dict('records'), source, arm['contract'], 'test')
        probs, timing, checks, input_function = serve_arm(arm, new, corpus, contract['ids'], 'fp16', 'cuda', state)
        r.update(input_function=input_function, input_code=input_code)
        need(probs.dtype == np.float32 and probs.shape == (contract['studies'], 12)
             and bool(np.isfinite(probs).all()) and timing['fp32_fallbacks'] == 0, 'Incomplete/nonfinite inference')
        write_predictions(result / 'raw.csv', contract['ids'], probs)
        r.update(status='PASS', studies=contract['studies'], targets=LAB, runtime=runtime, timing=timing,
                 checkpoints=checks, prob_clamp=PROB_EPS,
                 allocation={**allocation, 'selected_internal_threads': threads, 'data_loader_workers': 0},
                 environment=environment(), raw_sha256=sha(result / 'raw.csv'),
                 source_bindings={n: old['files'][n] for n in old['files'] if n.startswith('training/src/')},
                 test_contract=contract, DICOM_receipts=corpus.receipts,
                 CUDA_peak_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        r['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        for label, action in [('model', lambda: state['model'].to('cpu') if state.get('model') is not None else None),
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


def run(a):
    p = checked_manifest(a.manifest, a.sha256)
    arm = arm_by_name(p, a.arm)
    work, assets, new, destination = Path(a.work), Path(a.assets), Path(a.new_assets), Path(a.out)
    need(not work.exists() and not destination.exists(), 'Preserve previous outputs')
    work.mkdir(parents=True)
    result = {'status': 'FAIL', 'schema': SCHEMA, 'release': p['release'], 'arm': arm['name'],
              'release_manifest_sha256': a.sha256, 'accuracy_claim': False}
    published_sha = None
    try:
        started = a.started
        contract = test_contract(a.competition)
        plan = operational_plan(contract['studies'], p['arms'])
        projection = launch_projection(plan, p['arms'], arm['name'])
        deadline, timeout = launch_budget(started, time.monotonic(), contract['studies'], projection,
                                          a.bootstrap_max_seconds)
        need(sha(assets / OLD_PROTOCOL) == ASSETS_PROTOCOL_SHA, 'Budget/assets failed')
        old = read(assets / OLD_PROTOCOL)
        need(sha(assets / 'process_resource_receipt.py') == old['files']['process_resource_receipt.py'], 'Supervisor changed')
        need(sha(new / NEW_MANIFEST) == p['new_assets_manifest_sha256'], 'New assets manifest changed')
        argv = [sys.executable, str(Path(__file__).resolve()), 'worker', '--manifest', a.manifest,
                '--sha256', a.sha256, '--arm', arm['name'], '--assets', a.assets, '--new-assets', a.new_assets,
                '--work', a.work, '--competition', a.competition, '--opencv', a.opencv]
        context = {'arm': arm['name'], 'test_contract': contract, 'operational_plan': plan,
                   'launch_projection_seconds': projection, 'bootstrap_max_seconds': a.bootstrap_max_seconds,
                   'worker_argv': argv, 'deadline_monotonic': deadline, 'worker_timeout_seconds': timeout}
        write(work / 'launch.json', context)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        code = subprocess.run([sys.executable, str(assets / 'process_resource_receipt.py'), '--out',
              str(work / 'supervisor'), '--timeout', str(timeout), '--', *argv], env=env).returncode
        need(code == 0, 'Supervisor failed')
        w, s = read(work / 'worker/receipt.json'), read(work / 'supervisor/receipt.json')
        need(w['status'] == 'PASS' and w['release_manifest_sha256'] == a.sha256 and w['arm'] == arm['name']
             and w['test_contract'] == contract and w['studies'] == contract['studies'] and w['targets'] == TARGETS
             and not w['cleanup_errors'], 'Worker contract failed')
        need([(c['name'], c['sha256'], c['k_eval_release']) for c in w['checkpoints']]
             == [(c['name'], c['sha256'], arm['k_eval']) for c in arm['checkpoints']]
             and w['n_models'] == len(arm['checkpoints']) and w['aggregator'] == 'mean_prob'
             and w['rank_weight'] == arm['weight'], 'Worker ensemble differs')
        need(w['input_policy'] == arm['input_policy'] and w['norm'] == arm['norm'] == w['timing']['norm']
             and all(c['input_policy'] == arm['input_policy'] for c in w['checkpoints'])
             and (w['input_code'] == (code_record() if arm['input_policy'] == 'center3' else {})), 'Worker input differs')
        need(w['timing']['all_finite'] and w['timing']['fp32_fallbacks'] == 0
             and [x['uid'] for x in w['timing']['input_tensor_hashes']] == contract['ids'], 'Input/finite trace failed')
        need(w['runtime']['effective_amp'] == 'fp16' and w['runtime']['policy'] == 'strict'
             and not w['runtime']['precision_fallback_allowed'] and not w['runtime']['backends']['cudnn_allow_tf32']
             and not w['runtime']['backends']['matmul_allow_tf32'], 'Runtime differs')
        cgroup_telemetry = validate_resource(s, argv, w['pid'], timeout)
        raw = work / 'worker/raw.csv'
        published_sha = csv_check(raw, contract['ids'])
        need(published_sha == w['raw_sha256'], 'Worker output changed')
        need(output_bytes(work) <= OUTPUT, 'Output budget exceeded')
        with zipfile.ZipFile(work / 'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for name in ['launch.json', 'worker/receipt.json', 'worker/raw.csv',
                         'supervisor/receipt.json', 'supervisor/stdout.log', 'supervisor/stderr.log']:
                z.write(work / name, name)
        need(output_bytes(work) <= OUTPUT, 'Archive output budget exceeded')
        result.update(status='PASS_TEST_COHORT_ONLY', studies=contract['studies'], columns=contract['columns'],
                      submission_sha256=published_sha, operational_plan=plan, launch_projection_seconds=projection,
                      checkpoints=w['checkpoints'], n_models=w['n_models'], k_eval=arm['k_eval'],
                      input_policy=arm['input_policy'], norm=arm['norm'], input_function=w['input_function'],
                      rank_weight=arm['weight'], rank_weight_normalised=p['rank_weights'][arm['name']],
                      new_asset_dataset=p['new_asset_dataset'], worker_sha256=sha(work / 'worker/receipt.json'),
                      supervisor_sha256=sha(work / 'supervisor/receipt.json'), evidence_sha256=sha(work / 'evidence.zip'),
                      sampled_tree_rss_bytes=s['sampled_tree_rss_max_bytes'], cgroup_telemetry=cgroup_telemetry,
                      model_seconds=w['timing']['seconds'], elapsed_seconds=time.monotonic() - started,
                      scope='Actual supplied test cohort only; Public3 is not hidden runtime or accuracy proof')

        def final_check():
            need(test_contract(a.competition) == contract, 'Test metadata changed before finalization')
            need(output_bytes(work) <= OUTPUT, 'Final output budget exceeded')
        publish(raw, destination, published_sha, deadline, final_check)
        write(work / 'result.json', result)
        need(time.monotonic() < deadline and output_bytes(work) <= OUTPUT, 'Final acknowledgment budget failed')
        return result
    except BaseException as exc:
        if destination.is_file() and published_sha is not None and sha(destination) == published_sha:
            destination.unlink()
        result.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc)})
        write(work / 'failure.json', result)
        raise


def combine(a):
    """Several validated arm CSVs -> per-target average-tie rank mean (EXP098) -> atomic submission."""
    p = checked_manifest(a.manifest, a.sha256)
    need(len(p['arms']) >= 2, 'Rank combination needs two or more arms')
    base, destination = Path(a.base), Path(a.out)
    work = base / 'combine'
    need(not work.exists() and not destination.exists(), 'Preserve previous outputs')
    work.mkdir(parents=True)
    result = {'status': 'FAIL', 'schema': SCHEMA, 'release': p['release'], 'release_manifest_sha256': a.sha256,
              'accuracy_claim': False}
    published_sha = None
    try:
        started = a.started
        contract = test_contract(a.competition)
        ids = contract['ids']
        plan = operational_plan(contract['studies'], p['arms'])
        need(math.isfinite(started) and 0 <= time.monotonic() - started, 'Invalid clock')
        deadline = started + plan['internal_seconds']
        need(time.monotonic() < deadline, 'Deadline passed before combination')
        tables, arms = [], {}
        for arm in p['arms']:
            d = base / arm['name']
            r = read(d / 'work/result.json')
            need(r['status'] == 'PASS_TEST_COHORT_ONLY' and r['arm'] == arm['name'] and r['release_manifest_sha256']
                 == a.sha256 and r['studies'] == contract['studies'], 'Arm result differs: ' + arm['name'])
            need(csv_check(d / 'arm.csv', ids) == r['submission_sha256'], 'Arm output changed: ' + arm['name'])
            tables.append(load_probabilities(d / 'arm.csv', ids))
            need(r.get('rank_weight') == arm['weight'], 'Arm result weight differs: ' + arm['name'])
            arms[arm['name']] = {k: r[k] for k in ('submission_sha256', 'studies', 'elapsed_seconds',
                                                   'sampled_tree_rss_bytes', 'n_models', 'k_eval', 'rank_weight')}
        weights = [x['weight'] for x in p['arms']]
        cols = rank_mean(tables, weights)
        raw = work / 'combined.csv'
        write_columns(raw, ids, cols)
        published_sha = csv_check(raw, ids)
        equal = len(set(weights)) == 1
        result.update(status='PASS_RANK_MEAN_TEST_COHORT_ONLY', studies=len(ids), submission_sha256=published_sha,
                      formula=('per-target mean of average-tie percentile ranks over the full cohort, equal arm weights '
                               '(EXP098)' if equal else 'per-target weighted mean of average-tie percentile ranks over '
                               'the full cohort: sum_i(W_i*r_i)/sum_i(W_i)'),
                      arm_order=[x['name'] for x in p['arms']], arms=arms,
                      arm_weights={x['name']: x['weight'] for x in p['arms']}, rank_weights=p['rank_weights'],
                      scope='Actual supplied test cohort only; Public3 is not hidden runtime or accuracy proof')

        def final_check():
            need(test_contract(a.competition) == contract, 'Test metadata changed before finalization')
        publish(raw, destination, published_sha, deadline, final_check)
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
def checkpoint_meta(path):
    """Read non-weight checkpoint fields on CPU; the values must be plain JSON.

    Returns (meta, training code sha256 from provenance, input-module sha256 from provenance)."""
    if 'torch' not in sys.modules:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    need(isinstance(ck, dict) and isinstance(ck.get('model'), dict), 'Not a raptor checkpoint: ' + str(path))
    missing = [k for k in REQUIRED_META if k not in ck]
    need(not missing, 'Checkpoint metadata missing %s: %s' % (missing, path))
    meta = jsonable({k: ck[k] for k in META_KEYS if k in ck})
    need(meta['lab'] == TARGETS, 'Target order differs: ' + str(path))
    provenance = ck.get('provenance') or {}
    code = jsonable(provenance.get('code_sha256') or {})
    module = provenance.get('radimagenet_module_sha256')
    input_code = {'raptor_radimagenet.py': module} if isinstance(module, str) else {}
    return meta, code, input_code


def package_files(package):
    """sha256 of every package file except the manifest itself and Kaggle's dataset-metadata.json."""
    package = Path(package)
    out = {}
    for path in sorted(package.rglob('*')):
        rel = path.relative_to(package).as_posix()
        if path.is_file() and rel not in (NEW_MANIFEST, 'dataset-metadata.json'):
            out[rel] = sha(path)
    return out


def dataset_metadata(dataset):
    check_dataset(dataset)
    return {'title': dataset.split('/')[1], 'id': dataset, 'licenses': [{'name': 'other'}], 'isPrivate': True,
            'description': 'Own fixed checkpoints and centre-plane input code for the EXP111 v6 two-family release; '
                           'private research only.'}


def check_package(package, dataset):
    """The v4-v7 release-assets package directories and datasets are never (re)staged by v6."""
    check_dataset(dataset)
    need(all(Path(package).resolve() != p.resolve() for p in REFUSED_PACKAGES),
         'Refusing a v4-v7 release-assets package directory')
    return Path(package)


def place(source, target, copy):
    temporary = target.with_name(target.name + '.staging')
    need(not temporary.exists(), 'Stale staging file: ' + str(temporary))
    if not copy:
        try:
            os.link(source, target)
            return 'hardlink'
        except OSError:
            pass
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return 'copy'


def inventory_check(source, source_sha):
    """A run directory's output-inventory.json (EXP094/EXP099/EXP101/EXP107 runners) must record the same bytes."""
    inventory = Path(source).parent / 'output-inventory.json'
    if not inventory.is_file():
        return None
    record = read(inventory).get(Path(source).name)
    need(isinstance(record, dict) and record.get('sha256') == source_sha
         and record.get('bytes') == Path(source).stat().st_size,
         'Source checkpoint differs from its run inventory: ' + str(source))
    return sha(inventory)


def stage_code(package):
    """Copy the EXP107 input modules (fixed sha256) into <package>/code/."""
    (Path(package) / CODE_DIR).mkdir(parents=True, exist_ok=True)
    for n, s in CODE_FILES.items():
        source, target = CODE_SOURCE / n, Path(package) / CODE_DIR / n
        need(source.is_file() and sha(source) == s, 'Training input code differs from CODE_FILES: ' + n)
        if target.exists():
            need(sha(target) == s, 'Refusing to replace staged input code with different bytes: ' + n)
        else:
            place(source, target, True)
        need(sha(target) == s, 'Staged input code hash differs: ' + n)


def stage_assets(a):
    package = check_package(a.package, a.dataset)
    specs = []
    for text in a.checkpoint:
        name, sep, path = text.partition('=')
        need(sep == '=' and NAME.fullmatch(name) and path, 'Checkpoint must be NAME=PATH: ' + repr(text))
        specs.append((name, Path(path) if Path(path).is_absolute() else ROOT / path))
    need(len({n for n, _ in specs}) == len(specs), 'Duplicate checkpoint name')
    (package / 'checkpoints').mkdir(parents=True, exist_ok=True)
    meta_path = package / 'dataset-metadata.json'
    if meta_path.exists():
        need(read(meta_path) == dataset_metadata(a.dataset), 'dataset-metadata.json differs from ' + a.dataset)
    else:
        write(meta_path, dataset_metadata(a.dataset))
    manifest_path = package / NEW_MANIFEST
    previous = read(manifest_path) if manifest_path.exists() else None
    need(previous is None or (previous.get('schema') == ASSETS_SCHEMA and previous.get('dataset') == a.dataset),
         'Existing assets manifest belongs to another schema/dataset')
    stage_code(package)
    checkpoints = dict(previous['checkpoints']) if previous else {}
    for name, source in specs:
        need(source.is_file(), 'Missing source checkpoint: ' + str(source))
        source_sha = sha(source)
        inventory_sha = inventory_check(source, source_sha)
        meta, code, input_code = checkpoint_meta(source)      # before copying: a refused member leaves no file
        input_contract(derived_meta(meta), member=True)
        target = package / 'checkpoints' / (name + '.pt')
        if target.exists():
            need(sha(target) == source_sha, 'Refusing to replace staged checkpoint with different bytes: ' + name)
            mode = (checkpoints.get(name) or {}).get('link_mode', 'existing')
        else:
            mode = place(source, target, a.copy)
        need(sha(target) == source_sha == sha(source), 'Staged checkpoint hash differs: ' + name)
        try:
            rel = source.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(source)
        sidecars = {n: sha(source.parent / n) for n in ('summary.json', 'provenance.json', 'history.json')
                    if (source.parent / n).is_file()}
        checkpoints[name] = {'path': 'checkpoints/' + name + '.pt', 'sha256': source_sha, 'bytes': target.stat().st_size,
                             'meta': meta, 'training_code_sha256': code, 'input_code_sha256': input_code,
                             'source': rel, 'source_sidecars_sha256': sidecars,
                             'source_inventory_sha256': inventory_sha, 'link_mode': mode}
    files = package_files(package)
    expected = {c['path'] for c in checkpoints.values()} | {c['path'] for c in code_record().values()}
    need(set(files) == expected, 'Unexpected or missing package files: %s' % sorted(set(files) ^ expected))
    need(all(files[c['path']] == c['sha256'] for c in [*checkpoints.values(), *code_record().values()]),
         'Package file hash differs')
    body = {'schema': ASSETS_SCHEMA, 'dataset': a.dataset, 'checkpoints': dict(sorted(checkpoints.items())),
            'code': code_record(), 'files': files, 'dataset_metadata_sha256': sha(meta_path)}
    if previous is not None and {k: v for k, v in previous.items() if k != 'staged_at'} == body:
        print(json.dumps({'status': 'UNCHANGED', 'manifest_sha256': sha(manifest_path), 'checkpoints': sorted(checkpoints)}))
        return
    temporary = manifest_path.with_name(NEW_MANIFEST + '.staging')
    need(not temporary.exists(), 'Stale staging manifest')
    write(temporary, {**body, 'staged_at': datetime.now(timezone.utc).isoformat()})
    os.replace(temporary, manifest_path)
    print(json.dumps({'status': 'STAGED', 'package': str(package), 'manifest_sha256': sha(manifest_path),
                      'checkpoints': {n: {'sha256': c['sha256'], 'link_mode': c['link_mode']}
                                      for n, c in checkpoints.items()},
                      'code': code_record()}))


CODE = r'''import time
started=time.monotonic()
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
final=Path('/kaggle/working/submission.csv')
base=Path('/kaggle/working/exp111-@@RELEASE@@')
assert not base.exists() and not final.exists()
base.mkdir()
source=base64.b85decode(@@SOURCE@@)
assert hashlib.sha256(source).hexdigest()==@@SOURCE_SHA@@
tool=base/'exp111_release_v6.py'; tool.write_bytes(source)
MANIFEST=@@MANIFEST@@
DIGEST=@@DIGEST@@
manifest=base/'release-manifest.json'; manifest.write_bytes(MANIFEST.encode('utf-8'))
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==DIGEST
release=json.loads(MANIFEST); T=release['time_policy']; arms=[a['name'] for a in release['arms']]
new_owner,new_slug=release['new_asset_dataset'].split('/')
def one(paths,marker):
    found=[p for p in paths if (p/marker).is_file()]
    assert len(found)==1,marker
    return found[0]
old=one([Path('/kaggle/input/knee-exp089-recipe-transfer-v1'),Path('/kaggle/input/datasets/taeyangg4/knee-exp089-recipe-transfer-v1')],'exp089-t4-protocol.json')
new=one([Path('/kaggle/input/'+new_slug),Path('/kaggle/input/datasets/'+new_owner+'/'+new_slug)],'assets-manifest.json')
opencv_roots=[Path('/kaggle/input/opencv-python-headless-4120088-x86'),Path('/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86')]
opencv=[p for d in opencv_roots for p in d.glob('opencv_python_headless-4.12.0.88-*.whl')]
assert len(opencv)==1
comp_roots=[Path('/kaggle/input/rsna-knee-abnormality-detection'),Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
competition=[p for p in comp_roots if all((p/n).is_file() for n in ['test.csv','test_series.csv','sample_submission.csv']) and (p/'test_series').is_dir()]
assert len(competition)==1
n_test=sum(1 for _ in open(competition[0]/'sample_submission.csv',encoding='utf-8-sig'))-1
assert n_test>0
screen=sum(T['inflation']*(T['decode_seconds_per_study']+len(a['checkpoints'])*a['model_seconds_per_study'])*n_test for a in release['arms'])+T['screen_fixed_seconds']
assert screen<=T['screen_seconds'],'v6 time screen failed before any GPU work'
results={}
for i,arm in enumerate(arms):
    d=base/arm; d.mkdir()
    out=final if len(arms)==1 else d/'arm.csv'
    boot='600' if (i==0 or n_test<=3) else '30000'
    subprocess.run([sys.executable,str(tool),'run','--manifest',str(manifest),'--sha256',DIGEST,'--arm',arm,'--assets',str(old),'--new-assets',str(new),'--work',str(d/'work'),'--competition',str(competition[0]),'--opencv',str(opencv[0]),'--out',str(out),'--started',repr(started),'--bootstrap-max-seconds',boot],check=True)
    r=json.loads((d/'work/result.json').read_text()); assert r['status']=='PASS_TEST_COHORT_ONLY'
    assert hashlib.sha256(out.read_bytes()).hexdigest()==r['submission_sha256']
    results[arm]=r
if len(arms)>1:
    subprocess.run([sys.executable,str(tool),'combine','--manifest',str(manifest),'--sha256',DIGEST,'--base',str(base),'--competition',str(competition[0]),'--out',str(final),'--started',repr(started)],check=True)
    summary=json.loads((base/'combine/result.json').read_text()); assert summary['status']=='PASS_RANK_MEAN_TEST_COHORT_ONLY'
else:
    summary=results[arms[0]]
assert hashlib.sha256(final.read_bytes()).hexdigest()==summary['submission_sha256']
print(json.dumps({'release':release['release'],'status':summary['status'],'studies':summary['studies'],'submission_sha256':summary['submission_sha256'],'rank_weights':release['rank_weights'],'arms':{a:{k:results[a].get(k) for k in ('submission_sha256','n_models','k_eval','rank_weight','model_seconds','elapsed_seconds','sampled_tree_rss_bytes')} for a in results},'elapsed_seconds':time.monotonic()-started}))
'''


def render_code(release, source, manifest_text, digest):
    need(hashlib.sha256(manifest_text.encode('utf-8')).hexdigest() == digest, 'Manifest text differs from digest')
    need(RELEASE_NAME.fullmatch(release), 'Release name must match [a-z0-9][a-z0-9-]*')
    # Kaggle notebooks with a source line of ~67K characters never start (ERROR, empty log; measured 2026-10-07,
    # while v3's 32.6K-character line ran), so long literals are split into implicitly concatenated chunks.
    encoded = base64.b85encode(source).decode()
    chunks = [encoded[i:i + SOURCE_CHUNK] for i in range(0, len(encoded), SOURCE_CHUNK)] or ['']
    # v5: the multi-arm manifest grew to ~12.7K characters on one line; split it like the source so the longest line
    # stays near the 4,002 characters of the v4 notebooks that ran on Kaggle (the digest check still binds the text).
    mchunks = [manifest_text[i:i + SOURCE_CHUNK] for i in range(0, len(manifest_text), SOURCE_CHUNK)] or ['']
    values = {'RELEASE': release, 'SOURCE_SHA': repr(hashlib.sha256(source).hexdigest()),
              'SOURCE': '(\n' + '\n'.join(repr(c) for c in chunks) + '\n)',
              'MANIFEST': '(\n' + '\n'.join(repr(c) for c in mchunks) + '\n)', 'DIGEST': repr(digest)}
    tokens = re.findall(r'@@([A-Z_]+)@@', CODE)
    need(sorted(set(tokens)) == sorted(values), 'Template placeholders differ')
    code = re.sub(r'@@([A-Z_]+)@@', lambda m: values[m.group(1)], CODE)   # one pass: inserted text is never rescanned
    compile(code, 'exp111-release-notebook', 'exec')
    need(max(len(line) for line in code.split('\n')) <= MAX_NOTEBOOK_LINE, 'Notebook line too long for Kaggle')
    return code


def kernel_metadata(release, docker, dataset=NEW_DATASET):
    need(RELEASE_NAME.fullmatch(release), 'Release name must match [a-z0-9][a-z0-9-]*')
    need(docker == DOCKER_IMAGE and re.fullmatch(r'gcr\.io/[^@\s]+@sha256:[0-9a-f]{64}', docker), 'Pinned image differs')
    check_dataset(dataset)
    slug = KERNEL_PREFIX + release + KERNEL_SUFFIX
    return {'id': OWNER + '/' + slug, 'title': slug, 'code_file': 'release.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': False, 'keywords': ['gpu'],
            'dataset_sources': [OLD_DATASET, dataset, OPENCV_DATASET], 'kernel_sources': [],
            'competition_sources': [COMPETITION], 'model_sources': [], 'docker_image': docker,
            'machine_shape': 'NvidiaTeslaT4'}


def parse_model_seconds(texts):
    out = {}
    for text in texts or []:
        name, sep, value = text.partition('=')
        need(sep == '=' and NAME.fullmatch(name) and name not in out, 'Model seconds must be NAME=S: ' + repr(text))
        seconds = float(value)
        need(math.isfinite(seconds) and seconds > 0, 'Model seconds must be positive: ' + repr(text))
        out[name] = seconds
    return out


def notebook(a):
    need(RELEASE_NAME.fullmatch(a.release), 'Release name must match [a-z0-9][a-z0-9-]*')
    out, package = Path(a.out), check_package(a.package, a.dataset)
    need(not out.exists(), 'Choose a fresh kernel directory')
    am_path = package / NEW_MANIFEST
    need(am_path.is_file(), 'Run stage-assets first')
    am = read(am_path)
    need(am.get('schema') == ASSETS_SCHEMA and am.get('dataset') == a.dataset, 'Assets manifest schema differs')
    need(package_files(package) == am['files'], 'Package files differ from the assets manifest')
    need(read(package / 'dataset-metadata.json') == dataset_metadata(a.dataset), 'dataset-metadata.json differs')
    verify_code(package, am)
    old = verify_old_assets(OLD_LOCAL)
    need(read(ACCEPTANCE)['status'] == 'ROOT_T4_V2_ENGINEERING_PASS', 'Completed T4 acceptance required')
    pulled = read(a.docker_from)
    need(pulled.get('id') == DOCKER_SOURCE_KERNEL, 'Docker pin must come from the B224 release v3 kernel')
    docker = pulled.get('docker_image')
    overrides = parse_model_seconds(a.model_seconds)
    parsed = [parse_arm_spec(text) for text in a.arm]
    given = [w is not None for _, _, _, w in parsed]
    need(all(given) or not any(given), 'Give a rank weight :W for every arm or for none (none = equal weights)')
    need(len(parsed) > 1 or not any(given), 'A single-arm release has no rank weight')
    need(set(overrides) <= {name for name, _, _, _ in parsed}, 'Model seconds given for an unknown arm')
    arms, specs, seconds_basis = [], [], {}
    for text, (name, members, k, w) in zip(a.arm, parsed):
        arch = am['checkpoints'][members[0]]['meta']['arch'] if members[0] in am['checkpoints'] else None
        need(arch is not None, 'Unknown staged checkpoint: ' + members[0])
        model_seconds, seconds_basis[name] = arm_model_seconds(name, arch, overrides)
        arm = build_arm(name, members, k, am, model_seconds, 1.0 if w is None else w)
        for ck in arm['checkpoints']:
            code = am['checkpoints'][ck['name']]['training_code_sha256']
            for f in TRAINING_CODE_BOUND:
                need(code.get(f) == old['files']['training/src/' + f],
                     'Checkpoint %s was trained with %s differing from the frozen serving copy' % (ck['name'], f))
            if arm['input_policy'] == 'center3':
                # v6: the shipped input code must be the code the member was trained and evaluated with
                trained = {**{f: code.get(f) for f in CODE_FILES if f != 'raptor_radimagenet.py'},
                           **am['checkpoints'][ck['name']].get('input_code_sha256', {})}
                need(trained == CODE_FILES, 'Checkpoint %s was trained with input code differing from the shipped '
                                            'raptor_data.py/raptor_radimagenet.py' % ck['name'])
        arms.append(arm)
        specs.append(text)
    validate_arms(arms)
    weights = rank_weights(arms)
    plan_hidden = operational_plan(a.expected_studies, arms)
    meta = kernel_metadata(a.release, docker, a.dataset)
    source = Path(__file__).read_bytes()
    equal = len({x['weight'] for x in arms}) == 1
    manifest = {'schema': SCHEMA, 'release': a.release, 'created_at': datetime.now(timezone.utc).isoformat(),
                'release_source_sha256': sha(__file__), 'old_asset_dataset': OLD_DATASET,
                'old_asset_protocol_sha256': ASSETS_PROTOCOL_SHA, 'T4_acceptance_sha256': sha(ACCEPTANCE),
                'new_asset_dataset': a.dataset, 'new_assets_package': str(package.resolve()),
                'new_assets_manifest_sha256': sha(am_path),
                'opencv_dataset': OPENCV_DATASET, 'docker_image': docker, 'time_policy': TIME, 'RSS_bytes': RSS,
                'output_bytes': OUTPUT, 'prob_clamp': PROB_EPS, 'retries': 0, 'arm_specs': specs, 'arms': arms,
                'rank_weights': weights, 'model_seconds_basis': seconds_basis,
                'combination': ('single arm: raw mean_prob probabilities' if len(arms) == 1 else
                                'per-target mean of average-tie percentile ranks, equal arm weights (EXP098)' if equal
                                else 'per-target weighted mean of average-tie percentile ranks, '
                                     'sum_i(W_i*r_i)/sum_i(W_i) (weights normalised to 1)'),
                'study_count': 'dynamic complete official test cohort', 'accuracy_claim': False}
    out.mkdir(parents=True)
    write(out / 'release-manifest.json', manifest)
    digest = sha(out / 'release-manifest.json')
    text = (out / 'release-manifest.json').read_bytes().decode('utf-8')
    code = render_code(a.release, source, text, digest)
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
          'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
          'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': [
          {'cell_type': 'code', 'id': 'exp111-release-v6', 'metadata': {},
           'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
    write(out / 'release.ipynb', nb)
    write(out / 'kernel-metadata.json', meta)
    py = str(ROOT / '.venv/Scripts/python.exe')
    plan = {'status': 'CPU_PREPARED_NOT_PUSHED', 'release': a.release, 'kernel': meta['id'], 'arms': [
                {'name': x['name'], 'checkpoints': [c['name'] + ':' + c['sha256'] for c in x['checkpoints']],
                 'k_eval_release': x['k_eval'], 'k_eval_checkpoints': [c['meta']['k_eval'] for c in x['checkpoints']],
                 'arch': x['arch'], 'input_policy': x['input_policy'], 'norm': x['norm'],
                 'model_seconds_per_study': x['model_seconds_per_study'], 'rank_weight': x['weight'],
                 'rank_weight_normalised': weights[x['name']]} for x in arms],
            'manifest_sha256': digest, 'notebook_sha256': sha(out / 'release.ipynb'),
            'metadata_sha256': sha(out / 'kernel-metadata.json'), 'source_sha256': sha(__file__),
            'new_asset_dataset': a.dataset, 'new_assets_manifest_sha256': sha(am_path), 'docker_image': docker,
            'docker_image_source': {'kernel': DOCKER_SOURCE_KERNEL, 'pulled_metadata': str(Path(a.docker_from)),
                                    'pulled_metadata_sha256': sha(a.docker_from)},
            'screen_at_expected_hidden_N': plan_hidden, 'max_studies_passing_screen': max_studies(arms),
            # absolute backslash path and UTF-8 console output (D-B-015 items 9, 10 and 16)
            'publish_assets_argv': [py, '-m', 'kaggle', 'datasets', 'create', '-p', str(package.resolve()),
                                    '--dir-mode', 'zip'],
            'publish_assets_env': {'PYTHONIOENCODING': 'utf-8'},
            # no --timeout: pushes with --timeout 32400 never started on 2026-10-07 (D-B-015 item 13)
            'push_argv': [py, '-m', 'kaggle', 'kernels', 'push', '-p', str(out.resolve()),
                          '--accelerator', 'NvidiaTeslaT4'],
            'submit_argv_template': [py, '-m', 'kaggle', 'competitions', 'submit', COMPETITION, '-k', meta['id'],
                                     '-v', '<VERSION>', '-f', 'submission.csv', '-m', '<message>'],
            'first_output': WORK_PREFIX + a.release + '/' + (arms[0]['name'] + '/work/result.json' if len(arms) == 1
                                                            else 'combine/result.json'),
            'private': True, 'model_loads': 0, 'GPU_remote_network': 0, 'dataset_upload': False, 'submission': False}
    write(out / 'execution-plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('status', 'kernel', 'manifest_sha256', 'notebook_sha256',
                                           'max_studies_passing_screen')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    st = sub.add_parser('stage-assets')
    st.add_argument('--checkpoint', action='append', required=True, help='NAME=PATH (repeatable)')
    st.add_argument('--package', default=str(PACKAGE))
    st.add_argument('--dataset', default=NEW_DATASET, help='owner/slug of the new private assets dataset')
    st.add_argument('--copy', action='store_true', help='copy instead of hard-linking')
    nb = sub.add_parser('notebook')
    nb.add_argument('--release', required=True)
    nb.add_argument('--arm', action='append', required=True,
                    help='NAME or NAME=CK1+CK2[@K][:W] (repeatable, ordered; W for every arm or none)')
    nb.add_argument('--out', required=True)
    nb.add_argument('--package', default=str(PACKAGE))
    nb.add_argument('--dataset', default=NEW_DATASET, help='owner/slug of the new private assets dataset')
    nb.add_argument('--model-seconds', action='append', default=[],
                    help='ARM=S seconds per model per study (default: by architecture, %r)' % MODEL_SECONDS)
    nb.add_argument('--expected-studies', type=int, default=1300)
    nb.add_argument('--docker-from', default=str(DOCKER_PULL))
    for mode in ['run', 'worker']:
        action = sub.add_parser(mode)
        for key in ['manifest', 'sha256', 'arm', 'assets', 'new-assets', 'work', 'competition', 'opencv']:
            action.add_argument('--' + key, required=True)
        if mode == 'run':
            action.add_argument('--out', required=True)
            action.add_argument('--started', type=float, required=True)
            action.add_argument('--bootstrap-max-seconds', type=float, default=600.0)
    cb = sub.add_parser('combine')
    for key in ['manifest', 'sha256', 'base', 'competition', 'out']:
        cb.add_argument('--' + key, required=True)
    cb.add_argument('--started', type=float, required=True)
    a = parser.parse_args()
    {'stage-assets': stage_assets, 'notebook': notebook, 'run': run, 'worker': worker, 'combine': combine}[a.action](a)


if __name__ == '__main__':
    main()
