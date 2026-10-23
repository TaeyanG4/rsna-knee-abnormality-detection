"""H217 v1.0.0 (B on Claude Code, 2026-10-19): H208 with the K2 v5 repaired RCO worker.

Same INF3 parent (byte-identical parent-run.py), same K1 v2 model/assets and the same
90/10 rank formula as submitted H208 v1. Only the RCO branch changes:
  1. worker l1_rco_k2_worker_v101.py (cfcba1a2, from the assets dataset) is replaced by
     the embedded l1_rco_k2_worker_v102.py (4c001941), the exact worker K2 v5 ran on hidden
     (0.944). v101 sends no progress during the whole-cohort header scan, so its 15-minute
     stall guard can abort on the large hidden set and H208 then silently publishes INF3 only.
  2. receipt version check follows the worker ('1.0.2-setup-progress').
  3. a wall-clock deadline: the RCO branch must finish within RCO_DEADLINE_SECONDS of the
     coordinator start, else it is stopped and the INF3 parent is published (never a 9 h timeout).
CPU packaging only; no model loads, GPU, network or submission command.
Changelog: v1.0.0 first version.
"""
from __future__ import annotations

import argparse
import base64
import copy
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from tools import c85r7_r6w958inf3rco_v1150 as h208
from tools import c85r7_rco_overlay_v1 as overlay

ROOT = h208.ROOT
VERSION = '1.1.52-r6w958inf3rcop'
SLUG = 'taeyangg4/knee-lb-r6w958inf3rcop-v1'
KEY = 'R6W958-INF3RCOP-V1'
H208_STAGE = h208.STAGE
H208_NB_SHA = '3258d455b14c9d5116473803a9a59e7048e4287c624d00d3c23cc3637f4bfd59'
K2V5_NB = ROOT / 'artifacts/experiments/L1_RCO_K2/v1.0.5-rcof0-progress/kernel/release.ipynb'
K2V5_NB_SHA = '141e2c09514e064ccd02907a4e9ae3f3c92c07ace21630b68b1c302bba7ac553'
OLD_WORKER_SHA = 'cfcba1a26dddda54ebfeadd874d58538c1ac358a81e8b45bbabb2563c8b45e1f'
NEW_WORKER_SHA = '4c0019419fc589759f0a92b5ccdd640506671cde2d9b3df1dab1a66fe5f7605e'
NEW_WORKER_NAME = 'l1_rco_k2_worker_v102.py'
NEW_WORKER_VERSION = '1.0.2-setup-progress'
DEADLINE_SECONDS = 30000  # 8h20m from coordinator start; Kaggle limit 9h
MIN_RCO_SECONDS = 1200    # skip RCO (parent only) if less than this remains
STAGE = H208_STAGE.parent / ('v' + VERSION)
need, sha, encoded, digest = overlay.need, overlay.sha, overlay.encoded, h208.digest
exact = h208.exact


def worker_bytes():
    need(sha(K2V5_NB) == K2V5_NB_SHA, 'K2 v5 notebook pin')
    src = ''.join(json.loads(K2V5_NB.read_bytes())['cells'][0]['source'])
    found = re.findall(r"_k2_worker_bytes = _k2_base64\.b64decode\('([^']+)'\)", src)
    need(len(found) == 1, 'K2 v5 embedded worker')
    data = base64.b64decode(found[0])
    need(digest(data) == NEW_WORKER_SHA, 'v102 worker identity')
    return data


# (old, new) exact single-occurrence replacements on the H208 overlay; identity first.
def edits(old):
    return [
        ("VERSION = '1.1.50-r6w958inf3rco'", "VERSION = " + repr(VERSION)),
        ("KERNEL = 'taeyangg4/knee-lb-r6w958inf3rco-v1'", "KERNEL = " + repr(SLUG)),
        ("CANDIDATE = 'R6W958-INF3RCO-V1'", "CANDIDATE = " + repr(KEY)),
        ("WORKER_SHA = " + repr(OLD_WORKER_SHA),
         "WORKER_SHA = " + repr(NEW_WORKER_SHA) + "\nWORKER_NAME = " + repr(NEW_WORKER_NAME) +
         "\nWORKER_VERSION = " + repr(NEW_WORKER_VERSION) +
         "\nRCO_DEADLINE_SECONDS = " + str(DEADLINE_SECONDS) + "\nMIN_RCO_SECONDS = " + str(MIN_RCO_SECONDS)),
        ("record.get('version') == '1.0.1-rcof0'", "record.get('version') == WORKER_VERSION"),
        ("def run_child(command, *, env=None, cwd=None):", "def run_child(command, *, env=None, cwd=None, timeout=None):"),
        ("        code = process.wait()\n", "        code = process.wait(timeout=timeout)\n"),
        ("        worker = assets / 'source/l1_rco_k2_worker_v101.py'\n",
         "        worker = runtime / WORKER_NAME  # H217: embedded repaired worker (K2 v5)\n"),
        ("        rco_start = time.monotonic()\n",
         "        rco_start = time.monotonic()\n"
         "        remaining = RCO_DEADLINE_SECONDS - (rco_start - started)\n"
         "        record.update(rco_deadline_seconds=RCO_DEADLINE_SECONDS, rco_remaining_at_start=remaining)\n"
         "        need(remaining >= MIN_RCO_SECONDS, 'RCO skipped: insufficient time before deadline')\n"),
        ("            '--scratch', str(job / 'scratch'), '--device', 'cuda:0'], env=env, cwd=output)\n",
         "            '--scratch', str(job / 'scratch'), '--device', 'cuda:0'], env=env, cwd=output,\n"
         "            timeout=max(1., RCO_DEADLINE_SECONDS - (time.monotonic() - started)))\n"),
    ]


def runtime_source(old):
    source = old.decode()
    for before, after in edits(old):
        source = exact(source, before, after)
    compile(source, '<H217 coordinator>', 'exec')
    return source.encode()


def package():
    need(sha(H208_STAGE / 'kernel/release.ipynb') == H208_NB_SHA, 'H208 notebook pin')
    base = json.loads((H208_STAGE / 'kernel/release.ipynb').read_bytes())
    code = ''.join(base['cells'][1]['source'])
    old_blob, old_pins, old_files = h208.archive(code)
    need(set(old_files) == {'parent-run.py', 'c85r7_rco_overlay_v1.py'}, 'H208 outer roster')
    files = {'parent-run.py': old_files['parent-run.py'],
             'c85r7_rco_overlay_v1.py': runtime_source(old_files['c85r7_rco_overlay_v1.py']),
             NEW_WORKER_NAME: worker_bytes()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as src, zipfile.ZipFile(buffer, 'w') as dst:
        for info in src.infolist():
            dst.writestr(info, files[info.filename])
        extra = zipfile.ZipInfo(NEW_WORKER_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        extra.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(extra, files[NEW_WORKER_NAME])
    blob = buffer.getvalue()
    pins = {n: digest(v) for n, v in files.items()}
    new_code = exact(code, repr(base64.b64encode(old_blob).decode()), repr(base64.b64encode(blob).decode()))
    new_code = exact(new_code, 'PINS = ' + repr(old_pins), 'PINS = ' + repr(pins))
    new_code = exact(new_code, repr(digest(old_blob)), repr(digest(blob)))
    compile(new_code, '<H217 notebook>', 'exec')
    nb = copy.deepcopy(base)
    nb['cells'][1]['source'] = new_code.splitlines(True)
    md = ''.join(base['cells'][0]['source'])
    md = exact(md, '# H208 fixed INF3 + K1 v2 RCO\n', '# H217 = H208 with repaired K1 RCO worker (K2 v5 v102) + RCO deadline\n')
    nb['cells'][0]['source'] = md.splitlines(True)
    meta = json.loads((H208_STAGE / 'kernel/kernel-metadata.json').read_bytes())
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    need(meta['is_private'] is True and meta['enable_internet'] is False, 'private offline required')
    proof = dict(parent_notebook='H208 ' + H208_NB_SHA, parent_run_byte_identical=True,
        outer_members=pins, worker_sha256=NEW_WORKER_SHA, replaced_worker_sha256=OLD_WORKER_SHA,
        worker_source='K2 v5 notebook ' + K2V5_NB_SHA, model_sha256=overlay.MODEL_SHA,
        asset_manifest_sha256=overlay.MANIFEST_SHA, formula=h208.FORMULA,
        rco_deadline_seconds=DEADLINE_SECONDS, min_rco_seconds=MIN_RCO_SECONDS,
        builder='B Claude Code solo', independent_new_review=False)
    return nb, meta, proof, old_files, files


def materialize(stage=STAGE):
    nb, meta, proof, old, new = package()
    stage = Path(stage)
    need(not stage.exists(), 'immutable stage already exists')
    registration = dict(status='PREPARED_LOCAL_ONLY', candidate_id=KEY, kernel_id=SLUG, version=VERSION,
        formula=h208.FORMULA, notebook_sha256=digest(encoded(nb)), metadata_sha256=digest(encoded(meta)),
        qualified_for_submission=False, source_sha256=sha(__file__),
        runtime='hidden UNMEASURED; RCO deadline guard publishes INF3 parent if exceeded')
    diffs = list(difflib.unified_diff(old['c85r7_rco_overlay_v1.py'].decode().splitlines(True),
        new['c85r7_rco_overlay_v1.py'].decode().splitlines(True), fromfile='H208/overlay', tofile='H217/overlay'))
    outputs = {'kernel/release.ipynb': encoded(nb), 'kernel/kernel-metadata.json': encoded(meta),
        'embedded-file-manifest.json': encoded(proof), 'registration.json': encoded(registration),
        'full-payload.diff': ''.join(diffs).encode(),
        'CHANGELOG.md': ('# H217 ' + VERSION + '\n\nH208 v1 with the K2 v5 repaired RCO worker v102 embedded, '
            'receipt version check updated and an RCO wall-clock deadline (' + str(DEADLINE_SECONDS) + ' s) that '
            'falls back to the INF3 parent. parent-run.py byte-identical; model/assets/formula unchanged.\n').encode()}
    for name, data in outputs.items():
        overlay.write_once(stage / name, data)
    return registration


def validate(stage=STAGE):
    nb, meta, proof, _, files = package()
    stage = Path(stage)
    for name, value in [('kernel/release.ipynb', nb), ('kernel/kernel-metadata.json', meta),
                        ('embedded-file-manifest.json', proof)]:
        need((stage / name).read_bytes() == encoded(value), 'delivered drift: ' + name)
    delivered = json.loads((stage / 'kernel/release.ipynb').read_bytes())
    _, _, actual = h208.archive(''.join(delivered['cells'][1]['source']))
    need(actual == files, 'actual outer archive differs')
    return dict(status='PASS_STATIC_B_DIRECT', members=sorted(actual), qualified_for_submission=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'validate'])
    action = p.parse_args().action
    print(json.dumps({'prepare': materialize, 'validate': validate}[action](), sort_keys=True))
