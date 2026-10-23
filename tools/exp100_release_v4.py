"""EXP100 test-only release v4: serve ENSEMBLES of checkpoints with ONE DICOM decode per study per arm.

  # 1. stage checkpoints into the new private assets package (hard-link on the same volume, else copy)
  python tools/exp100_release_v4.py stage-assets --checkpoint student=artifacts/experiments/EXP095_STUDENT_B8/model_best.pt \
      --checkpoint control=artifacts/experiments/EXP090_COV_BF16_CTRL/model_best.pt
  # 2. generate a Kaggle kernel directory (never pushed here)
  python tools/exp100_release_v4.py notebook --release student --arm student --out .local/exp100-release-v4/student
  python tools/exp100_release_v4.py notebook --release pair --arm ens=student+control@24 --arm student --out DIR

Arms. A release manifest lists one or more arms; each arm is
  {name, arch, res, k_eval, contract, norm, window_policy, aggregator: 'mean_prob', model_seconds_per_study,
   checkpoints: [{name, path: 'checkpoints/<name>.pt', sha256, bytes, meta}]}.
`--arm NAME` serves the staged checkpoint NAME alone; `--arm NAME=CK1+CK2[@K]` serves the probability mean of the
staged checkpoints CK1, CK2 at k_eval K (default: the members' common training k_eval). In the worker an arm's models
are wrapped in one nn.Module returning logit(mean_i sigmoid(model_i(x))) with probabilities clamped to
[1e-6, 1-1e-6], so the unchanged raptor_runtime.predict decodes every study once and runs every member.
Several arms are combined exactly like EXP098: per target, average-tie percentile rank over the full supplied cohort,
equal-weight mean across arms (one decode per arm), atomic publication.

Safety (from release v3/v3.1 unless noted): frozen hashes of the EXP089 assets (wheels, serving code, supervisor) and,
new, of the v4 assets manifest and every served checkpoint; dependency versions; Linux CPython 3.12; test-contract
check; launch_budget with the deadline anchored at the first-cell clock (v3.1) where a later arm must still fit the
conservative projection of itself and every arm after it; resource validation fatal only on oom/oom_kill, sampled
RSS > 8 GiB, child/cleanup errors (v3); full ordered output; atomic publication; receipts and evidence. Checkpoint
metadata arch/res/lab/contract/norm/window_policy must equal the arm; the release k_eval may differ from the
checkpoint's training k_eval (both recorded).

Up-front screen: sum over arms of 1.3*(6.52503319258 + n_models*model_seconds_per_study)*N + 600 <= 30600, where
model_seconds_per_study is a manifest field (default 0.35 s per model per study on a T4, not yet measured).
stage-assets/notebook read local files only: they never push kernels, publish datasets or submit.
"""
from __future__ import annotations

import argparse
import base64
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
SCHEMA = 'exp100-test-release-v4'
OWNER = 'taeyangg4'
COMPETITION = 'rsna-knee-abnormality-detection'
OLD_DATASET = 'taeyangg4/knee-exp089-recipe-transfer-v1'
OLD_PROTOCOL = 'exp089-t4-protocol.json'
ASSETS_PROTOCOL_SHA = '3e0e5529191570955b3229458adf511073401dc354de04b33164988f830687a8'
OLD_LOCAL = ROOT / '.local/exp089-recipe-transfer/t4-v3/dataset'
ACCEPTANCE = ROOT / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-v2-root-acceptance.json'
NEW_DATASET = 'taeyangg4/knee-release-assets-v4'
NEW_MANIFEST = 'assets-manifest.json'
ASSETS_SCHEMA = 'knee-release-assets-v4'
PACKAGE = ROOT / 'kaggle/packages/KNEE_RELEASE_ASSETS_V4'
OPENCV_DATASET = 'mattiaangeli/opencv-python-headless-4120088-x86'
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
PROB_EPS = 1e-6
RSS = 8 * 2**30
OUTPUT = 1 * 2**30
VERSIONS = ('2.10.0+cu128', '1.0.19', '4.12.0', '3.0.2', '3.2.6')
SERVING = ['config/workspace.toml', 'b_rn3f_serving.py', 'process_resource_receipt.py',
           *['training/src/' + n for n in ['raptor_common.py', 'raptor_runtime.py', 'raptor_dicom.py', 'raptor_ensemble.py']]]
TRAINING_CODE_BOUND = ['raptor_common.py', 'raptor_runtime.py']
META_KEYS = ['arch', 'res', 'lab', 'contract', 'k_eval', 'norm', 'window_policy', 'epoch', 'tag', 'amp_train', 'amp_eval']
REQUIRED_META = ['arch', 'res', 'lab', 'contract', 'k_eval', 'norm', 'window_policy']
ARM_MATCH = ['arch', 'res', 'contract', 'norm', 'window_policy']
ARM_KEYS = {'name', 'arch', 'res', 'k_eval', 'contract', 'norm', 'window_policy', 'aggregator',
            'model_seconds_per_study', 'checkpoints'}
CKPT_KEYS = {'name', 'path', 'sha256', 'bytes', 'meta'}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}')
RELEASE_NAME = re.compile(r'[a-z0-9][a-z0-9-]{0,39}')
SOURCE_CHUNK = 4000          # characters per embedded base85 line
MAX_NOTEBOOK_LINE = 30000    # Kaggle failed to start a notebook with a 66,897-character line; 32,646 ran
SHA = re.compile(r'[0-9a-f]{64}')
ARM_SPEC = re.compile(r'(?P<name>[A-Za-z0-9][A-Za-z0-9_-]{0,47})'
                      r'(?:=(?P<members>[A-Za-z0-9][A-Za-z0-9_-]{0,47}(?:\+[A-Za-z0-9][A-Za-z0-9_-]{0,47})*))?'
                      r'(?:@(?P<k>[1-9][0-9]{0,3}))?')


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
    temporary = destination.with_name(destination.name + '.exp100-tmp')
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
    """NAME | NAME=CK1+CK2[@K] | NAME@K -> (arm name, member checkpoint names, k_eval or None)."""
    m = ARM_SPEC.fullmatch(text or '')
    need(m is not None, 'Arm spec must be NAME or NAME=CK1+CK2[@K]: ' + repr(text))
    members = m.group('members').split('+') if m.group('members') else [m.group('name')]
    need(len(set(members)) == len(members), 'Duplicate ensemble member in ' + repr(text))
    return m.group('name'), members, int(m.group('k')) if m.group('k') else None


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
        need(arm['norm'] in ('imagenet', 'none') and arm['window_policy'] in ('legacy', 'global-valid', 'within-slot-valid'),
             'Unsupported norm/window policy')
        m = arm['model_seconds_per_study']
        need(type(m) in (int, float) and math.isfinite(m) and m > 0, 'model_seconds_per_study must be positive')
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
            need(isinstance(meta, dict) and all(k in meta for k in REQUIRED_META) and meta['lab'] == TARGETS
                 and all(meta[k] == arm[k] for k in ARM_MATCH), 'Member metadata differs from arm ' + arm['name'])
    return arms


def build_arm(name, members, k_eval, assets_manifest, model_seconds):
    entries = []
    for c in members:
        need(c in assets_manifest['checkpoints'], 'Unknown staged checkpoint: ' + c)
        e = assets_manifest['checkpoints'][c]
        entries.append({'name': c, 'path': e['path'], 'sha256': e['sha256'], 'bytes': e['bytes'], 'meta': e['meta']})
    first = entries[0]['meta']
    for key in ARM_MATCH:
        need(all(e['meta'][key] == first[key] for e in entries), 'Ensemble members differ in ' + key)
    if k_eval is None:
        trained = {e['meta']['k_eval'] for e in entries}
        need(len(trained) == 1, 'Members were trained at different k_eval; give NAME=...@K')
        k_eval = trained.pop()
    arm = {'name': name, **{k: first[k] for k in ARM_MATCH}, 'k_eval': k_eval, 'aggregator': 'mean_prob',
           'model_seconds_per_study': model_seconds, 'checkpoints': entries}
    validate_arms([arm])
    return arm


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
    return p


def verify_old_assets(root, opencv=None):
    root = Path(root)
    need(sha(root / OLD_PROTOCOL) == ASSETS_PROTOCOL_SHA, 'Frozen assets protocol changed')
    old = read(root / OLD_PROTOCOL)
    for name in [*SERVING, *[r['path'] for r in old['wheels']]]:
        need(sha(root / name) == old['files'][name], 'Selected immutable asset changed: ' + name)
    if opencv is not None:
        need(sha(opencv) == old['opencv_sha256'], 'OpenCV wheel changed')
    return old


def verify_new_assets(release, root, arms):
    """Bind the served checkpoints: manifest digest, presence, and sha256 of every member file."""
    root = Path(root)
    need((root / NEW_MANIFEST).is_file(), 'Missing new assets manifest')
    need(sha(root / NEW_MANIFEST) == release['new_assets_manifest_sha256'], 'New assets manifest changed')
    am = read(root / NEW_MANIFEST)
    need(am.get('schema') == ASSETS_SCHEMA and am.get('dataset') == NEW_DATASET, 'New assets schema differs')
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
    expected = {'lab': TARGETS, **{k: arm[k] for k in ARM_MATCH}}
    bad = sorted(k for k, v in expected.items() if meta.get(k) != v)
    need(not bad, 'Checkpoint metadata differs from arm %s: %s' % (ck['name'], bad))
    drift = sorted(k for k, v in ck['meta'].items() if meta.get(k) != v)
    need(not drift, 'Checkpoint metadata differs from staged record %s: %s' % (ck['name'], drift))
    return {'name': ck['name'], 'path': ck['path'], 'sha256': ck['sha256'], 'tag': meta.get('tag'),
            'epoch': meta.get('epoch'), 'amp_train': meta.get('amp_train'),
            'k_eval_checkpoint': meta.get('k_eval'), 'k_eval_release': arm['k_eval']}


# ------------------------------------------------------------------ ensemble and rank combination
def mean_prob_module(models):
    """One module: forward = logit(mean_i sigmoid(model_i(x))), probabilities clamped to [1e-6, 1-1e-6]."""
    import torch
    need(len(models) > 0, 'Empty ensemble')

    class MeanProbEnsemble(torch.nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)

        def forward(self, x):
            logits = [m(x) for m in self.members]
            with torch.autocast(x.device.type, enabled=False):
                p = torch.stack([torch.sigmoid(z.float()) for z in logits]).mean(0)
                p = p.clamp(PROB_EPS, 1.0 - PROB_EPS)
                return torch.log(p) - torch.log1p(-p)

    return MeanProbEnsemble(models)


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


def rank_mean(tables):
    """Per target: equal-weight mean of each arm's average-tie percentile rank (EXP098 for two arms)."""
    need(len(tables) >= 2 and len({len(t) for t in tables}) == 1 and len(tables[0]) > 0, 'Two or more equal arm tables')
    need(all(len(row) == len(TARGETS) for t in tables for row in t), 'Arm table width differs')
    def mean(values):
        total = 0.0   # plain left-to-right float addition: (a+b)/2 for two arms exactly as EXP098;
        for v in values:   # builtin sum() uses compensated summation since Python 3.12
            total += v
        return total / len(values)

    cols = []
    for t in range(len(TARGETS)):
        ranks = [average_rank([row[t] for row in tab]) for tab in tables]
        cols.append([mean(v) for v in zip(*ranks)])
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
         'aggregator': arm['aggregator'], 'n_models': len(arm['checkpoints']), 'k_eval': arm['k_eval'],
         'cleanup_errors': [], 'training': 0, 'label_Gold_reads': 0, 'scoring': 0, 'retries': 0}
    source = model = None
    try:
        envdir = Path('/kaggle/temp') / ('exp100-release-env-' + arm['name'])
        need(not envdir.exists(), 'New dependency dir required')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target',
                        str(envdir), *[str(assets / x['path']) for x in old['wheels']], a.opencv], check=True)
        sys.path[:0] = [str(envdir), str(assets / 'training/src')]
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
        from raptor_common import LAB, load_checkpoint, write_predictions, environment
        from raptor_runtime import configure_runtime, predict
        from raptor_dicom import DirectoryDicomSource, DicomCorpus
        runtime = configure_runtime('fp16', 'cuda', 'strict')
        need(LAB == TARGETS, 'Target order changed')
        members, checks = [], []
        for ck in arm['checkpoints']:
            member, meta = load_checkpoint(new / ck['path'])
            checks.append(checkpoint_check(meta, arm, ck))
            members.append(member.eval().to('cuda'))   # move now: CPU RSS holds at most one checkpoint
        model = mean_prob_module(members)
        rows = pd.read_csv(Path(a.competition) / 'test_series.csv', dtype={UID: str, 'SeriesInstanceUID': str})
        source = DirectoryDicomSource(Path(a.competition) / 'test_series')
        corpus = DicomCorpus.from_records(contract['ids'], rows.to_dict('records'), source, arm['contract'], 'test')
        probs, timing = predict(model, corpus, contract['ids'], arm['k_eval'], arm['res'], 'fp16', 'cuda',
                                norm=arm['norm'], batch=1, window_policy=arm['window_policy'])
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
        for label, action in [('model', lambda: model.to('cpu') if model is not None else None),
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
             and w['n_models'] == len(arm['checkpoints']) and w['aggregator'] == 'mean_prob', 'Worker ensemble differs')
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
                      worker_sha256=sha(work / 'worker/receipt.json'),
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
            arms[arm['name']] = {k: r[k] for k in ('submission_sha256', 'studies', 'elapsed_seconds',
                                                   'sampled_tree_rss_bytes', 'n_models', 'k_eval')}
        cols = rank_mean(tables)
        raw = work / 'combined.csv'
        write_columns(raw, ids, cols)
        published_sha = csv_check(raw, ids)
        result.update(status='PASS_RANK_MEAN_TEST_COHORT_ONLY', studies=len(ids), submission_sha256=published_sha,
                      formula='per-target mean of average-tie percentile ranks over the full cohort, equal arm weights (EXP098)',
                      arm_order=[x['name'] for x in p['arms']], arms=arms,
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
    """Read non-weight checkpoint fields on CPU; the values must be plain JSON."""
    if 'torch' not in sys.modules:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    need(isinstance(ck, dict) and isinstance(ck.get('model'), dict), 'Not a raptor checkpoint: ' + str(path))
    missing = [k for k in REQUIRED_META if k not in ck]
    need(not missing, 'Checkpoint metadata missing %s: %s' % (missing, path))
    meta = jsonable({k: ck[k] for k in META_KEYS if k in ck})
    need(meta['lab'] == TARGETS, 'Target order differs: ' + str(path))
    code = jsonable(((ck.get('provenance') or {}).get('code_sha256')) or {})
    return meta, code


def package_files(package):
    """sha256 of every package file except the manifest itself and Kaggle's dataset-metadata.json."""
    package = Path(package)
    out = {}
    for path in sorted(package.rglob('*')):
        rel = path.relative_to(package).as_posix()
        if path.is_file() and rel not in (NEW_MANIFEST, 'dataset-metadata.json'):
            out[rel] = sha(path)
    return out


def dataset_metadata():
    return {'title': 'knee-release-assets-v4', 'id': NEW_DATASET, 'licenses': [{'name': 'other'}], 'isPrivate': True,
            'description': 'Own fixed checkpoints for the EXP100 v4 ensemble release; private research only.'}


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


def stage_assets(a):
    package = Path(a.package)
    specs = []
    for text in a.checkpoint:
        name, sep, path = text.partition('=')
        need(sep == '=' and NAME.fullmatch(name) and path, 'Checkpoint must be NAME=PATH: ' + repr(text))
        specs.append((name, Path(path) if Path(path).is_absolute() else ROOT / path))
    need(len({n for n, _ in specs}) == len(specs), 'Duplicate checkpoint name')
    (package / 'checkpoints').mkdir(parents=True, exist_ok=True)
    meta_path = package / 'dataset-metadata.json'
    if meta_path.exists():
        need(read(meta_path) == dataset_metadata(), 'dataset-metadata.json differs from the v4 dataset')
    else:
        write(meta_path, dataset_metadata())
    manifest_path = package / NEW_MANIFEST
    previous = read(manifest_path) if manifest_path.exists() else None
    checkpoints = dict(previous['checkpoints']) if previous else {}
    for name, source in specs:
        need(source.is_file(), 'Missing source checkpoint: ' + str(source))
        source_sha = sha(source)
        target = package / 'checkpoints' / (name + '.pt')
        if target.exists():
            need(sha(target) == source_sha, 'Refusing to replace staged checkpoint with different bytes: ' + name)
            mode = (checkpoints.get(name) or {}).get('link_mode', 'existing')
        else:
            mode = place(source, target, a.copy)
        need(sha(target) == source_sha == sha(source), 'Staged checkpoint hash differs: ' + name)
        meta, code = checkpoint_meta(target)
        try:
            rel = source.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(source)
        sidecars = {n: sha(source.parent / n) for n in ('summary.json', 'provenance.json', 'history.json')
                    if (source.parent / n).is_file()}
        checkpoints[name] = {'path': 'checkpoints/' + name + '.pt', 'sha256': source_sha, 'bytes': target.stat().st_size,
                             'meta': meta, 'training_code_sha256': code, 'source': rel,
                             'source_sidecars_sha256': sidecars, 'link_mode': mode}
    files = package_files(package)
    need(set(files) == {c['path'] for c in checkpoints.values()}, 'Unexpected or missing package files: %s'
         % sorted(set(files) ^ {c['path'] for c in checkpoints.values()}))
    need(all(files[c['path']] == c['sha256'] for c in checkpoints.values()), 'Package file hash differs')
    body = {'schema': ASSETS_SCHEMA, 'dataset': NEW_DATASET, 'checkpoints': dict(sorted(checkpoints.items())),
            'files': files, 'dataset_metadata_sha256': sha(meta_path)}
    if previous is not None and {k: v for k, v in previous.items() if k != 'staged_at'} == body:
        print(json.dumps({'status': 'UNCHANGED', 'manifest_sha256': sha(manifest_path), 'checkpoints': sorted(checkpoints)}))
        return
    temporary = manifest_path.with_name(NEW_MANIFEST + '.staging')
    need(not temporary.exists(), 'Stale staging manifest')
    write(temporary, {**body, 'staged_at': datetime.now(timezone.utc).isoformat()})
    os.replace(temporary, manifest_path)
    print(json.dumps({'status': 'STAGED', 'package': str(package), 'manifest_sha256': sha(manifest_path),
                      'checkpoints': {n: {'sha256': c['sha256'], 'link_mode': c['link_mode']} for n, c in checkpoints.items()}}))


CODE = r'''import time
started=time.monotonic()
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
final=Path('/kaggle/working/submission.csv')
base=Path('/kaggle/working/exp100-@@RELEASE@@')
assert not base.exists() and not final.exists()
base.mkdir()
source=base64.b85decode(@@SOURCE@@)
assert hashlib.sha256(source).hexdigest()==@@SOURCE_SHA@@
tool=base/'exp100_release_v4.py'; tool.write_bytes(source)
MANIFEST=@@MANIFEST@@
DIGEST=@@DIGEST@@
manifest=base/'release-manifest.json'; manifest.write_bytes(MANIFEST.encode('utf-8'))
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==DIGEST
def one(paths,marker):
    found=[p for p in paths if (p/marker).is_file()]
    assert len(found)==1,marker
    return found[0]
old=one([Path('/kaggle/input/knee-exp089-recipe-transfer-v1'),Path('/kaggle/input/datasets/taeyangg4/knee-exp089-recipe-transfer-v1')],'exp089-t4-protocol.json')
new=one([Path('/kaggle/input/knee-release-assets-v4'),Path('/kaggle/input/datasets/taeyangg4/knee-release-assets-v4')],'assets-manifest.json')
opencv_roots=[Path('/kaggle/input/opencv-python-headless-4120088-x86'),Path('/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86')]
opencv=[p for d in opencv_roots for p in d.glob('opencv_python_headless-4.12.0.88-*.whl')]
assert len(opencv)==1
comp_roots=[Path('/kaggle/input/rsna-knee-abnormality-detection'),Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
competition=[p for p in comp_roots if all((p/n).is_file() for n in ['test.csv','test_series.csv','sample_submission.csv']) and (p/'test_series').is_dir()]
assert len(competition)==1
n_test=sum(1 for _ in open(competition[0]/'sample_submission.csv',encoding='utf-8-sig'))-1
assert n_test>0
release=json.loads(MANIFEST); T=release['time_policy']; arms=[a['name'] for a in release['arms']]
screen=sum(T['inflation']*(T['decode_seconds_per_study']+len(a['checkpoints'])*a['model_seconds_per_study'])*n_test for a in release['arms'])+T['screen_fixed_seconds']
assert screen<=T['screen_seconds'],'v4 time screen failed before any GPU work'
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
print(json.dumps({'release':release['release'],'status':summary['status'],'studies':summary['studies'],'submission_sha256':summary['submission_sha256'],'arms':{a:{k:results[a].get(k) for k in ('submission_sha256','n_models','k_eval','model_seconds','elapsed_seconds','sampled_tree_rss_bytes')} for a in results},'elapsed_seconds':time.monotonic()-started}))
'''


def render_code(release, source, manifest_text, digest):
    need(hashlib.sha256(manifest_text.encode('utf-8')).hexdigest() == digest, 'Manifest text differs from digest')
    need(RELEASE_NAME.fullmatch(release), 'Release name must match [a-z0-9][a-z0-9-]*')
    # Kaggle notebooks with a source line of ~67K characters never start (ERROR, empty log; measured 2026-10-07,
    # while v3's 32.6K-character line ran), so long literals are split into implicitly concatenated chunks.
    encoded = base64.b85encode(source).decode()
    chunks = [encoded[i:i + SOURCE_CHUNK] for i in range(0, len(encoded), SOURCE_CHUNK)] or ['']
    values = {'RELEASE': release, 'SOURCE_SHA': repr(hashlib.sha256(source).hexdigest()),
              'SOURCE': '(\n' + '\n'.join(repr(c) for c in chunks) + '\n)', 'MANIFEST': repr(manifest_text), 'DIGEST': repr(digest)}
    tokens = re.findall(r'@@([A-Z_]+)@@', CODE)
    need(sorted(set(tokens)) == sorted(values), 'Template placeholders differ')
    code = re.sub(r'@@([A-Z_]+)@@', lambda m: values[m.group(1)], CODE)   # one pass: inserted text is never rescanned
    compile(code, 'exp100-release-notebook', 'exec')
    need(max(len(line) for line in code.split('\n')) <= MAX_NOTEBOOK_LINE, 'Notebook line too long for Kaggle')
    return code


def kernel_metadata(release, docker):
    need(RELEASE_NAME.fullmatch(release), 'Release name must match [a-z0-9][a-z0-9-]*')
    need(docker == DOCKER_IMAGE and re.fullmatch(r'gcr\.io/[^@\s]+@sha256:[0-9a-f]{64}', docker), 'Pinned image differs')
    slug = 'knee-exp100-' + release + '-v4'
    return {'id': OWNER + '/' + slug, 'title': slug, 'code_file': 'release.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': False, 'keywords': ['gpu'],
            'dataset_sources': [OLD_DATASET, NEW_DATASET, OPENCV_DATASET], 'kernel_sources': [],
            'competition_sources': [COMPETITION], 'model_sources': [], 'docker_image': docker,
            'machine_shape': 'NvidiaTeslaT4'}


def notebook(a):
    need(RELEASE_NAME.fullmatch(a.release), 'Release name must match [a-z0-9][a-z0-9-]*')
    out, package = Path(a.out), Path(a.package)
    need(not out.exists(), 'Choose a fresh kernel directory')
    am_path = package / NEW_MANIFEST
    need(am_path.is_file(), 'Run stage-assets first')
    am = read(am_path)
    need(am.get('schema') == ASSETS_SCHEMA and am.get('dataset') == NEW_DATASET, 'Assets manifest schema differs')
    need(package_files(package) == am['files'], 'Package files differ from the assets manifest')
    need(read(package / 'dataset-metadata.json') == dataset_metadata(), 'dataset-metadata.json differs')
    old = verify_old_assets(OLD_LOCAL)
    need(read(ACCEPTANCE)['status'] == 'ROOT_T4_V2_ENGINEERING_PASS', 'Completed T4 acceptance required')
    pulled = read(a.docker_from)
    need(pulled.get('id') == DOCKER_SOURCE_KERNEL, 'Docker pin must come from the B224 release v3 kernel')
    docker = pulled.get('docker_image')
    arms, specs = [], []
    for text in a.arm:
        name, members, k = parse_arm_spec(text)
        arm = build_arm(name, members, k, am, a.model_seconds_per_study)
        for ck in arm['checkpoints']:
            code = am['checkpoints'][ck['name']]['training_code_sha256']
            for f in TRAINING_CODE_BOUND:
                need(code.get(f) == old['files']['training/src/' + f],
                     'Checkpoint %s was trained with %s differing from the frozen serving copy' % (ck['name'], f))
        arms.append(arm)
        specs.append(text)
    validate_arms(arms)
    plan_hidden = operational_plan(a.expected_studies, arms)
    meta = kernel_metadata(a.release, docker)
    source = Path(__file__).read_bytes()
    manifest = {'schema': SCHEMA, 'release': a.release, 'created_at': datetime.now(timezone.utc).isoformat(),
                'release_source_sha256': sha(__file__), 'old_asset_dataset': OLD_DATASET,
                'old_asset_protocol_sha256': ASSETS_PROTOCOL_SHA, 'T4_acceptance_sha256': sha(ACCEPTANCE),
                'new_asset_dataset': NEW_DATASET, 'new_assets_manifest_sha256': sha(am_path),
                'opencv_dataset': OPENCV_DATASET, 'docker_image': docker, 'time_policy': TIME, 'RSS_bytes': RSS,
                'output_bytes': OUTPUT, 'prob_clamp': PROB_EPS, 'retries': 0, 'arm_specs': specs, 'arms': arms,
                'combination': ('single arm: raw mean_prob probabilities' if len(arms) == 1 else
                                'per-target mean of average-tie percentile ranks, equal arm weights (EXP098)'),
                'study_count': 'dynamic complete official test cohort', 'accuracy_claim': False}
    out.mkdir(parents=True)
    write(out / 'release-manifest.json', manifest)
    digest = sha(out / 'release-manifest.json')
    text = (out / 'release-manifest.json').read_bytes().decode('utf-8')
    code = render_code(a.release, source, text, digest)
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
          'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
          'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': [
          {'cell_type': 'code', 'id': 'exp100-release-v4', 'metadata': {},
           'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
    write(out / 'release.ipynb', nb)
    write(out / 'kernel-metadata.json', meta)
    py = str(ROOT / '.venv/Scripts/python.exe')
    plan = {'status': 'CPU_PREPARED_NOT_PUSHED', 'release': a.release, 'kernel': meta['id'], 'arms': [
                {'name': x['name'], 'checkpoints': [c['name'] + ':' + c['sha256'] for c in x['checkpoints']],
                 'k_eval_release': x['k_eval'], 'k_eval_checkpoints': [c['meta']['k_eval'] for c in x['checkpoints']],
                 'model_seconds_per_study': x['model_seconds_per_study']} for x in arms],
            'manifest_sha256': digest, 'notebook_sha256': sha(out / 'release.ipynb'),
            'metadata_sha256': sha(out / 'kernel-metadata.json'), 'source_sha256': sha(__file__),
            'new_assets_manifest_sha256': sha(am_path), 'docker_image': docker,
            'docker_image_source': {'kernel': DOCKER_SOURCE_KERNEL, 'pulled_metadata': str(Path(a.docker_from)),
                                    'pulled_metadata_sha256': sha(a.docker_from)},
            'screen_at_expected_hidden_N': plan_hidden, 'max_studies_passing_screen': max_studies(arms),
            'publish_assets_argv': [py, '-m', 'kaggle', 'datasets', 'create', '-p', str(package.resolve()),
                                    '--dir-mode', 'zip'],
            'push_argv': [py, '-m', 'kaggle', 'kernels', 'push', '-p', str(out.resolve()),
                          '--timeout', str(TIME['platform_full_seconds'])],
            'submit_argv_template': [py, '-m', 'kaggle', 'competitions', 'submit', COMPETITION, '-k', meta['id'],
                                     '-v', '<VERSION>', '-f', 'submission.csv', '-m', '<message>'],
            'first_output': 'exp100-' + a.release + '/' + (arms[0]['name'] + '/work/result.json' if len(arms) == 1
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
    st.add_argument('--copy', action='store_true', help='copy instead of hard-linking')
    nb = sub.add_parser('notebook')
    nb.add_argument('--release', required=True)
    nb.add_argument('--arm', action='append', required=True, help='NAME or NAME=CK1+CK2[@K] (repeatable, ordered)')
    nb.add_argument('--out', required=True)
    nb.add_argument('--package', default=str(PACKAGE))
    nb.add_argument('--model-seconds-per-study', type=float, default=MODEL_SECONDS_DEFAULT)
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
