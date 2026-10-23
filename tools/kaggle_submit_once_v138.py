"""kaggle_submit_once v1.3.8 — the only allowed way to submit to the competition (B, 2026-10-08, after the 09:02 duplicate submissions).

Incident (D-B-055): an automation judged success by the word "Successfully" in the CLI output; the current Kaggle CLI prints only
"N submissions remaining today." on success, so successful submissions were retried (rad5 x3, coat5rad5 x2; 3 daily slots lost).
v1.1.0 closes the gaps found by slot C's review (.local/crosscheck-20261008/release-screen/SUBMIT_ONCE_REVIEW.md): replay after an
unknown outcome, concurrent callers, exceptions/timeouts without a ledger entry, and a malformed header-only list read as empty.

Changelog v1.3.8: separate version; replay generic v1.4.4 admissions and parent chains.
Backup: .local/backups/strict-variant-gate-v1.4.4/.
Changelog v1.3.7: separate version; add v1.4.3 only for knee-lb-r224flipq-v1.
Backup: .local/backups/strict-variant-gate-v1.4.3/.
Changelog v1.3.6: admit exact-diff R224 gate v1.4.2 only for knee-lb-r224q-v1;
retain all older gate branches. Backup: .local/backups/strict-variant-gate-v1.4.2/.
Changelog v1.3.5: accept separate R224 gate v1.4.0 for exactly knee-lb-r224q-v1;
require its PIN, helper hashes and pinned C85-R7 parent. Legacy gates unchanged.
Backup: .local/backups/strict-variant-gate-v1.4.0/.
Changelog v1.3.4: accept gate v1.3.0 only for the two pinned standalone families;
require both standalone helper source hashes. Backup: .local/backups/strict-variant-gate-v1.2.0/.
Changelog v1.3.3: accept gate v1.2.0 and refuse fixture-only admission.
Changelog v1.3.2: read gate v1.1.0; require hash-bound registered parent and
authenticate C85-R7 v1.1.0 or EXP143v2 standalone kernel bytes.
Backup: .local/backups/kaggle-submit-once-v1.3.1/.

Changelog v1.3.1: check prediction fallback counters only; ignore anchor methods;
support HB127 missing-member counters. FAST strict-gate interface unchanged.
Backup: .local/backups/kaggle-submit-once-v1.3.0/.

Changelog v1.3.0: mandatory hash-bound visible receipt and PASS review admission;
explicit registration/real-slug mapping; FAST requires fresh strict gate evidence.
Backup: .local/backups/kaggle-submit-once-v1.2.0/.

Rules enforced here:
  1. Every submission carries a unique ID; the description starts with "[<ID>]". The candidate identity is (ID) and (kernel, version):
     either one already submitted (or still unresolved) blocks a new attempt, so renaming the ID cannot resubmit the same version.
  2. One process at a time: an exclusive lock file; a leftover lock refuses (check by hand, then delete it).
  3. A durable IN_FLIGHT ledger record is written and flushed BEFORE the submit call. Any attempt whose last state is IN_FLIGHT or
     UNCONFIRMED blocks every later attempt for that identity until `--reconcile` resolves it (found in the list -> SUBMITTED;
     absent after a manual check -> `--reconcile --declare-absent` -> RESOLVED_ABSENT).
  4. The submission list must parse with the exact expected header and well-formed rows, else nothing is submitted.
  5. ONE ATTEMPT PER CANDIDATE (user rule, 2026-10-08 11:1x KST: "<owner remark removed>"): any earlier
     attempt for this ID or kernel/version (submitted, errored on Kaggle, unconfirmed, or declared absent) and any list row carrying
     "[<ID>]" (any status) refuses the submission; the failure is reported to the user. Only `--user-approved-retry "<the user's words,
     time>"` (recorded in the ledger) lets a candidate be tried again, and never while a non-ERROR row or an open attempt exists.
     Nothing is submitted when the UTC day already has --daily-limit non-ERROR rows.
  6. Exactly ONE submit call per invocation; never retried automatically; exceptions are recorded, never raised past the ledger.
  7. Success only when a new "[<ID>]" row appears in the list (polled), never from the CLI text.
Exit codes: 0 submitted and confirmed (or --check-only admitted); 2 refused before submitting; 3 submit call made but outcome unknown (run --reconcile later).
Any exit code other than 0 is reported to the user; nothing retries automatically (scheduled callers run this once and stop).
Usage: python tools/kaggle_submit_once.py --id H-B-31-rad5-v6 --kernel taeyangg4/knee-exp111-rad5-v6 --version 2 --message "..." --receipt <visible.json> --review <review.md>
       python tools/kaggle_submit_once.py --reconcile --id H-B-31-rad5-v6 --kernel taeyangg4/knee-exp111-rad5-v6 --version 2 [--declare-absent]
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

VERSION = '1.3.8'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMP = 'rsna-knee-abnormality-detection'
SCREEN_DIR = ROOT / 'artifacts' / 'experiments' / 'SUBMISSION_SCREEN'
LEDGER = SCREEN_DIR / 'submit_ledger.jsonl'
LOCK = SCREEN_DIR / 'submit.lock'
HEADER = ['ref', 'fileName', 'date', 'description', 'status', 'publicScore', 'privateScore']
ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$')
OPEN_STATES = ('IN_FLIGHT', 'UNCONFIRMED')
# Immutable hidden-proven parent kernel identities. A provenance-only copy may
# change directory layout, never these bytes. EXP143v2: ledger ref 57005140.
PARENT_KERNEL_PINS = {
    'taeyangg4/knee-exp169-c85r7codec-v1': {
        'release.ipynb': '71c3a8f02e42e8fc7b3b734a10a33c12b700fac3e48d540b4802a5aaceb26eba',
        'kernel-metadata.json': 'ee4920c4679f7c3ca5244cb62acb0139210a2563d69de8c19917dbacf49a1663',
    },
    'taeyangg4/knee-exp152-own0949-v2': {
        'release.ipynb': '1045c22cd6de8e28dfeb60fc47e62b0356d069487d520c63e67421586aadb08a',
        'kernel-metadata.json': 'cff7e3ea949caca229869d379c26b237696e7fe2c84175af38b47114b91daa58',
    },
}

# Additive v1.3.0 policy: these pins never extend legacy receipt admission.
STANDALONE_PARENT_PINS = {
    'taeyangg4/knee-hb157-rkd-standalone-v1': ('nartaa/rsna-knee-0949-anatomical-mirror', {
        'release.ipynb': '8f1f96530b08543bd6caeed48489eea7c01657d6994f22e21a8400047fd4e100',
        'kernel-metadata.json': 'b42f0cb371d4a970074eee5cd58ec75992c264c3e1ad25a5994b9e9e97757ef1',
    }),
    'taeyangg4/knee-hb156-cnxs-l2-v1': ('taeyangg4/knee-hb151-pl-coat-v1', {
        'release.ipynb': 'd1a557f282fd429b61d0f1b70998d5604741410511773da5cf52f3cd51e3d8ed',
        'kernel-metadata.json': 'a9f0296f00e9acdd35bb3b7016e5c9e8cc4eeb1dc732fa5c95cc5b9e012f1680',
    }),
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _local(path):
    """Admission only reads project evidence, never credential or report trees."""
    p = Path(path)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    _need(p.is_relative_to(ROOT.resolve()), 'evidence must be inside project')
    _need(not any(x.lower() in {'.ssh', '.kaggle', '.codex', 'api-keys', 'relabel'}
                  for x in p.parts), 'forbidden evidence path')
    return p


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def _read_evidence(path, hashes):
    p = _local(path)
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    _need(str(p) not in hashes or hashes[str(p)] == digest, 'evidence changed during admission')
    hashes[str(p)] = digest
    return data


def _json_evidence(path, hashes):
    result = json.loads(_read_evidence(path, hashes).decode('utf-8-sig'), object_pairs_hook=_json_pairs)
    _need(isinstance(result, dict), 'evidence JSON must be an object')
    return result


def _positive_version(value):
    _need(type(value) in (str, int) and re.fullmatch(r'[1-9][0-9]*', str(value)),
          'Kaggle version must be a positive integer')
    return str(value)


def _zero_counters(value, where):
    if isinstance(value, dict):
        _need(bool(value), 'empty fallback counters: ' + where)
        for key, child in value.items():
            _zero_counters(child, where + '.' + key)
    else:
        _need(type(value) in (int, float) and value == 0, 'nonzero/malformed fallback counter: ' + where)


def _fallbacks(record, prefix=''):
    """Check prediction failures, not successful use of a prior anatomy anchor."""
    if isinstance(record, list):
        for i, child in enumerate(record):
            if isinstance(child, (dict, list)):
                _fallbacks(child, prefix + '.' + str(i))
        return
    for key, value in record.items():
        name = (prefix + '.' + key).strip('.')
        prediction = (key in {'prediction_fallbacks', 'a172_prediction_fallbacks',
                             'r_prediction_fallbacks', 's_r7_prediction_fallbacks',
                             'r_fallback_count'}
                      or prefix.split('.')[-1] == 'fallbacks'
                      and key in {'prediction', 'a172_prediction', 'r_prediction', 's_r7_prediction'})
        if prediction:
            # Producer equality assertions, not numeric prediction counters.
            if type(value) is bool and name in {
                    'pr7.telemetry.r_fallback_count', 'r7_telemetry.telemetry.r_fallback_count'}:
                _need(value is True, 'failed fallback telemetry assertion: ' + name)
            else:
                _zero_counters(value, name)
        elif isinstance(value, (dict, list)):
            _fallbacks(value, name)


def _prediction_evidence(record):
    if record['status'] == 'COMPLETE_HB127':
        # HB127 finish() records nonfinite prediction rows by member, and carries
        # authenticated_pr7() prediction counters in r7_telemetry.
        missing = record.get('missing')
        _need(isinstance(missing, dict) and {'R2-s43', 'R2-s44', 'S_R7'} <= missing.keys(),
              'missing HB127 prediction counters')
        _zero_counters(missing, 'missing')
        telemetry = record.get('r7_telemetry')
        _need(isinstance(telemetry, dict) and
              {'r_prediction_fallbacks', 's_r7_prediction_fallbacks'} <= telemetry.keys(),
              'missing HB127 R7 prediction counters')
    else:
        counters = record.get('fallbacks')
        _need(isinstance(counters, dict) and bool(counters), 'missing fallback counters')
        _need(bool({'prediction', 'a172_prediction', 'r_prediction', 's_r7_prediction'} & counters.keys())
              or counters.keys() == {'placeholder_errors'} and (
                  'FAST' in record['status'] or record.get('fast_path') is True
                  or 'FAST' in str(record.get('mode', ''))), 'missing prediction fallback counters')
    _fallbacks(record)


def _strict_gate(kernel, binding, hashes):
    path = _local(Path('.local/state/gates') / (kernel.split('/')[1] + '.json'))
    gate = _json_evidence(path, hashes)
    _need(binding.get('gate_sha256') == hashes[str(path)], 'FAST/gate hash binding missing or stale')
    _need(gate.get('status') == 'PASS' and gate.get('errors') == []
          and gate.get('gate_version') in {'1.0.0', '1.1.0', '1.2.0', '1.3.0', '1.4.0', '1.4.2', '1.4.3', '1.4.4'}, 'strict gate is not a supported PASS')
    _need(not gate.get('fixture_only') and not (gate.get('stage_inventory') or {}).get('fixture_only'), 'fixture-only gate cannot authorize submission')
    _need(gate.get('kernel_id') == kernel, 'strict gate kernel mismatch')
    if gate['gate_version'] == '1.4.4':
        # Dynamic slugs are safe only after replaying the exact pin and independent
        # dispatcher review, including the admitted parent chain. No new slug list.
        from tools import strict_variant_gate as base_gate
        from tools import strict_variant_gate_r224_v144 as contract
        observed = base_gate.Inputs(ROOT)
        checked = contract.verify_receipt(path, observed)
        _need(checked == gate, 'generic admission changed during replay')
        observed.recheck()
        for name, digest in observed.hashes().items():
            _need(hashlib.sha256(_read_evidence(name, hashes)).hexdigest() == digest, 'generic admission input drift')
        return gate
    inputs = gate.get('input_sha256')
    _need(isinstance(inputs, dict) and bool(inputs), 'strict gate has no input hashes')
    for name, digest in inputs.items():
        _need(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest), 'bad gate input hash')
        data = _read_evidence(name, hashes)
        _need(hashlib.sha256(data).hexdigest() == digest, 'stale gate input: ' + name)
    stage = _local(gate['stage'])
    required = {stage / 'registration.json', stage / 'kernel/kernel-metadata.json',
                ROOT / 'tools/strict_variant_gate.py'}
    if gate['gate_version'] == '1.3.0':
        _need(kernel in STANDALONE_PARENT_PINS, 'unsupported standalone child')
        required.update({ROOT / 'tools/strict_variant_gate_standalone.py',
                         ROOT / 'tools/strict_variant_gate_hb151.py'})
    if gate['gate_version'] in ('1.4.0', '1.4.2'):
        _need(kernel == 'taeyangg4/knee-lb-r224q-v1', 'unsupported R224 child')
        helper = 'strict_variant_gate_r224_v142.py' if gate['gate_version'] == '1.4.2' else 'strict_variant_gate_r224.py'
        required.update({ROOT / 'tools' / helper,
                         ROOT / 'tools/strict_variant_gate_standalone.py',
                         stage / 'R224-PIN.json', stage / 'allowlist.json', stage / 'full-diff.txt'})
    if gate['gate_version'] == '1.4.3':
        _need(kernel == 'taeyangg4/knee-lb-r224flipq-v1', 'unsupported R224 flip child')
        required.update({ROOT / 'tools/strict_variant_gate_r224_v143.py',
                         ROOT / 'tools/strict_variant_gate_r224_v142.py',
                         ROOT / 'tools/strict_variant_gate_standalone.py',
                         ROOT / '.local/state/gates/knee-lb-r224q-v1.json',
                         stage / 'R224-PIN.json', stage / 'allowlist.json', stage / 'full-diff.txt'})
    recorded = {_local(p) for p in inputs}
    _need(required <= recorded, 'gate missing required stage/tool hashes')
    manifests = [p for p in recorded if p.parent == stage and p.name in ('SHA256SUMS', 'SHA256SUMS.txt')]
    _need(len(manifests) == 1, 'gate missing stage manifest hash')
    kernel_files = {p.resolve() for p in (stage / 'kernel').rglob('*') if p.is_file()}
    _need(kernel_files == {p for p in recorded if p.is_relative_to(stage / 'kernel')},
          'gate kernel file set changed')
    reg = _json_evidence(stage / 'registration.json', hashes)
    meta = _json_evidence(stage / 'kernel/kernel-metadata.json', hashes)
    _need(reg.get('kernel_id') == kernel == meta.get('id'), 'gate stage identity mismatch')
    if gate['gate_version'] in ('1.4.0', '1.4.2'):
        pin = _json_evidence(stage / 'R224-PIN.json', hashes)
        allow = _json_evidence(stage / 'allowlist.json', hashes)
        _need(pin.get('version') == gate['gate_version'] and pin.get('status') == 'REVIEW_REQUIRED'
              and pin.get('fixture_only') is False and pin.get('kernel_id') == kernel
              and reg.get('binding_version') == gate['gate_version'] and reg.get('version') == '1.1.23-r224q'
              and pin.get('parent') == 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0'
              and pin.get('parent_kernel_files') == PARENT_KERNEL_PINS['taeyangg4/knee-exp169-c85r7codec-v1']
              and allow.get('pin_sha256') == hashes[str(stage / 'R224-PIN.json')], 'R224 PIN binding differs')
    if gate['gate_version'] == '1.4.3':
        from tools import strict_variant_gate_r224_v143 as contract
        pin = _json_evidence(stage / 'R224-PIN.json', hashes)
        allow = _json_evidence(stage / 'allowlist.json', hashes)
        _need(pin.get('version') == '1.4.3' and pin.get('status') == 'REVIEW_REQUIRED'
              and pin.get('fixture_only') is False and pin.get('kernel_id') == kernel
              and reg.get('binding_version') == '1.4.3' and reg.get('version') == '1.1.24-r224flipq'
              and reg.get('source_stage') == contract.CANONICAL_STAGE
              and pin.get('source_stage') == contract.CANONICAL_STAGE
              and pin.get('parent') == contract.PARENT and pin.get('parent_kernel_files') == contract.PARENT_PINS
              and pin.get('kernel_files') == contract.CHILD_PINS
              and pin.get('parent_receipt') == contract.PARENT_RECEIPT
              and pin.get('parent_receipt_sha256') == contract.PARENT_RECEIPT_SHA
              and hashes[str(ROOT / contract.PARENT_RECEIPT)] == contract.PARENT_RECEIPT_SHA
              and pin.get('exact_member_diffs') == contract.EXACT_CHANGES
              and pin.get('identical_members') == contract.IDENTICAL
              and allow.get('pin_sha256') == hashes[str(stage / 'R224-PIN.json')], 'R224 flip PIN binding differs')
    _registered_parent(gate, reg, recorded, hashes)
    return gate


def _registered_parent(gate, reg, recorded, hashes):
    """Bind both ends of the comparison; a PASS for a different parent is useless."""
    _need(isinstance(reg.get('parent'), str) and bool(reg['parent']), 'registered parent missing')
    _need(isinstance(gate.get('parent'), str) and bool(gate['parent']), 'gate parent missing')
    parent = _local(reg['parent'])
    _need(parent == _local(gate['parent']), 'gate parent differs from registered parent')
    _need(parent != _local(gate['stage']), 'stage cannot be its own parent')
    required = {parent / 'registration.json', parent / 'kernel/kernel-metadata.json'}
    _need(required <= recorded, 'gate missing required parent hashes')
    manifests = [p for p in recorded if p.parent == parent and p.name in ('SHA256SUMS', 'SHA256SUMS.txt')]
    _need(len(manifests) == 1, 'gate missing parent manifest hash')
    _need(reg.get('parent_sha256sums') == hashes[str(manifests[0])], 'registered parent manifest mismatch')
    preg = _json_evidence(parent / 'registration.json', hashes)
    meta = _json_evidence(parent / 'kernel/kernel-metadata.json', hashes)
    identity = meta.get('id')
    pins = PARENT_KERNEL_PINS
    if gate.get('gate_version') == '1.3.0':
        _need(gate.get('kernel_id') in STANDALONE_PARENT_PINS, 'unsupported standalone child')
        parent_id, parent_pins = STANDALONE_PARENT_PINS[gate['kernel_id']]
        pins = {parent_id: parent_pins}
    if gate.get('gate_version') in ('1.4.0', '1.4.2'):
        _need(gate.get('kernel_id') == 'taeyangg4/knee-lb-r224q-v1', 'unsupported R224 child')
        pins = {'taeyangg4/knee-exp169-c85r7codec-v1': PARENT_KERNEL_PINS['taeyangg4/knee-exp169-c85r7codec-v1']}
    if gate.get('gate_version') == '1.4.3':
        from tools import strict_variant_gate_r224_v143 as contract
        _need(gate.get('kernel_id') == contract.SLUG, 'unsupported R224 flip child')
        pins = {'taeyangg4/knee-lb-r224q-v1': contract.PARENT_PINS}
    _need(isinstance(identity, str) and identity in pins
          and preg.get('kernel_id') == identity, 'unsupported parent identity')
    folder = parent / 'kernel'
    files = {p.resolve() for p in folder.rglob('*') if p.is_file()}
    _need(files == {p for p in recorded if p.is_relative_to(folder)}, 'gate parent kernel file set changed')
    actual = {p.relative_to(folder).as_posix(): hashes[str(p)] for p in files}
    _need(actual == pins[identity], 'parent kernel differs from hidden-proven pins')


def qualify(kernel, version, receipt, review):
    """Offline, hash-bound admission. See kaggle_submit_once.md for transport schema."""
    _need(receipt and review, '--receipt and --review are required')
    _need(isinstance(kernel, str) and re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', kernel), 'bad kernel slug')
    version = _positive_version(version)
    hashes = {}
    receipt = _local(receipt)
    record = _json_evidence(receipt, hashes)
    _need(isinstance(record.get('status'), str) and record['status'].startswith('COMPLETE'),
          'visible receipt status must start with COMPLETE')
    _prediction_evidence(record)
    if 'codecs' in record:
        codec = record['codecs']
        _need(codec == 'ACTIVE' or isinstance(codec, dict) and codec.get('status') == 'ACTIVE', 'codecs not ACTIVE')
        if isinstance(codec, dict) and 's_r7_gate' in codec:
            _need(codec['s_r7_gate'] == 'ACTIVE', 'S_R7 codecs not ACTIVE')

    # Runtime receipts predate transport provenance. Preserve them; a sibling
    # sidecar binds their exact bytes to the confirmed remote slug/version.
    sidecar = receipt.parent / 'submission-binding.json'
    binding = record
    if sidecar.exists():
        binding = _json_evidence(sidecar, hashes)
        _need(binding.get('schema_version') == '1.0.0', 'unsupported submission binding')
        _need(binding.get('receipt_sha256') == hashes[str(receipt)], 'visible receipt binding hash mismatch')
    _need(binding.get('kernel') == kernel, 'receipt kernel mismatch/missing transport binding')
    _need(_positive_version(binding.get('kernel_version')) == version, 'receipt Kaggle version mismatch')
    for key in ('kernel', 'kernel_id'):
        if key in record:
            _need(record[key] == kernel, 'conflicting receipt ' + key)
    if 'kernel_version' in record:
        _need(_positive_version(record['kernel_version']) == version, 'conflicting receipt kernel_version')
    runtime_version = record.get('version')
    _need(type(runtime_version) in (str, int), 'receipt version missing/malformed')
    if type(runtime_version) in (int, str) and re.fullmatch(r'[1-9][0-9]*', str(runtime_version)):
        _need(str(runtime_version) == version, 'receipt version mismatch')
        runtime_version = None
    else:
        _need(isinstance(runtime_version, str) and
              re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?', runtime_version),
              'receipt runtime version malformed')

    regpath = _local(binding.get('registration', ''))
    reg = _json_evidence(regpath, hashes)
    _need(binding.get('registration_sha256') == hashes[str(regpath)], 'registration hash missing/mismatch')
    registered = reg.get('kernel_id')
    _need(isinstance(registered, str) and re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', registered), 'invalid registration kernel')
    if runtime_version is not None:
        _need(runtime_version == reg.get('version'), 'runtime version differs from registration')

    if 'fast_path' in record:
        _need(type(record['fast_path']) is bool, 'fast_path must be boolean')
    fast = 'FAST' in record['status'].upper() or record.get('fast_path') is True
    mode = record.get('mode', '')
    _need(isinstance(mode, str), 'invalid visible mode')
    fast = fast or 'FAST' in mode.upper()
    fast = fast or (receipt.parent / 'FAST_PATH_VISIBLE').exists()
    gate = _strict_gate(kernel, binding, hashes) if fast or 'gate_sha256' in binding else None
    if gate:
        _need(regpath == _local(gate['stage']) / 'registration.json', 'visible registration differs from gated stage')
    owner, slug = kernel.split('/')
    directory_bound = receipt.parent.name in {
        slug + '-visible', kernel.replace('/', '-') + '-visible',
        slug + '-v' + version + '-visible', kernel.replace('/', '-') + '-v' + version + '-visible'}
    if registered != kernel:
        reg_owner, reg_slug = registered.split('/')
        _need(owner == reg_owner and slug == (reg_owner + '-' + reg_slug).lower(), 'unrecognised registration/real-slug mapping')
        _need(directory_bound or gate is not None, 'prefixed-title alias lacks directory/gate binding')
    if slug.lower().startswith(owner.lower() + '-'):
        _need(directory_bound or gate is not None, 'prefixed-title slug lacks directory/gate binding')

    review_text = _read_evidence(review, hashes).decode('utf-8-sig').replace('\r\n', '\n').rstrip()
    _need(review_text.endswith('VERDICT: PASS') and re.findall(r'^VERDICT:.*$', review_text, re.M) == ['VERDICT: PASS'],
          'pre-submission review must end with one VERDICT: PASS')
    _need(re.search(r'(?<![A-Za-z0-9_/-])' + re.escape(kernel) + r'(?![A-Za-z0-9_/-])', review_text),
          'review does not name exact real kernel slug')
    admission = dict(kernel=kernel, kernel_version=version, registered_kernel=registered,
                     fast_path=fast, input_sha256=hashes)
    _recheck(admission)
    return admission


def _recheck(admission):
    for name, digest in admission['input_sha256'].items():
        _need(hashlib.sha256(_local(name).read_bytes()).hexdigest() == digest, 'admission evidence changed: ' + name)


def kaggle_cli(*args):
    exe = ROOT / '.venv' / 'Scripts' / 'python.exe'
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    r = subprocess.run([str(exe), '-X', 'utf8', '-m', 'kaggle', *args], capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env, timeout=600)
    return r.returncode, (r.stdout + r.stderr).strip()


def read_submissions(cli):
    """Rows of the competition submission list, or None when it cannot be read or does not have the exact expected shape."""
    try:
        code, out = cli('competitions', 'submissions', COMP, '--csv')
    except Exception:
        return None
    start = out.find('ref,')
    if code != 0 or start < 0:
        return None
    try:
        parsed = list(csv.reader(io.StringIO(out[start:])))
    except csv.Error:
        return None
    if not parsed or parsed[0] != HEADER:
        return None
    rows = []
    for vals in parsed[1:]:
        if not vals:
            continue
        if len(vals) != len(HEADER) or not vals[0].isdigit():
            return None
        rows.append(dict(zip(HEADER, vals)))
    return rows


def carries(row, sub_id):
    return (row.get('description') or '').startswith(f'[{sub_id}]')


def is_error(row):
    return 'ERROR' in (row.get('status') or '').upper()


def used_today(rows, today):
    return sum(1 for r in rows if (r.get('date') or '').startswith(today) and not is_error(r))


def read_ledger(ledger):
    if not ledger.exists():
        return []
    recs = []
    for line in ledger.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def append_ledger(rec, ledger):
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def same_identity(rec, sub_id, kernel, version):
    return rec.get('id') == sub_id or (rec.get('kernel') == kernel and str(rec.get('version')) == str(version))


def attempt_states(recs, sub_id, kernel, version):
    """Last state of every attempt (keyed by attempt id) that belongs to this candidate identity."""
    states = {}
    for r in recs:
        if r.get('attempt') and same_identity(r, sub_id, kernel, version):
            states[r['attempt']] = r
    return states


class Lock:
    def __init__(self, path):
        self.path, self.fd = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        os.write(self.fd, f'pid {os.getpid()} at {time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n'.encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            _release_lock(self.path)


def _release_lock(path, sleep=time.sleep):
    """Bounded Windows sharing-violation retry; persistent failure leaves a safe lock."""
    for attempt in range(40):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as exc:
            if (isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32)) and attempt < 39:
                sleep(0.05)
            else:
                return


def _base(sub_id, kernel, version, attempt=None):
    return dict(tool=f'kaggle_submit_once v{VERSION}', id=sub_id, kernel=kernel, version=str(version), attempt=attempt,
                time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))


def submit_once(sub_id, kernel, version, message, cli=kaggle_cli, daily_limit=5, poll_seconds=30, poll_tries=10,
                sleep=time.sleep, now=None, ledger=LEDGER, lock=LOCK, user_approved_retry=None, receipt=None, review=None):
    if not ID_RE.match(sub_id):
        raise ValueError('bad --id (letters, digits, . _ -; 3-81 chars)')
    ledger, lock = Path(ledger), Path(lock)
    with Lock(lock) as held:
        if held is None:
            rec = {**_base(sub_id, kernel, version), 'outcome': 'REFUSED', 'reason': f'lock exists ({lock}); another submission is '
                   'running or a previous run crashed — check the list and ledger, then delete the lock by hand'}
            return 2, rec  # no ledger write without the lock
        return _submit_locked(sub_id, kernel, version, message, cli, daily_limit, poll_seconds, poll_tries, sleep, now, ledger,
                              user_approved_retry, receipt, review)


def _submit_locked(sub_id, kernel, version, message, cli, daily_limit, poll_seconds, poll_tries, sleep, now, ledger,
                   user_approved_retry=None, receipt=None, review=None):
    desc = f'[{sub_id}] {message}'
    today = (now or dt.datetime.now(dt.timezone.utc)).strftime('%Y-%m-%d')

    def refuse(reason, **extra):
        rec = {**_base(sub_id, kernel, version), 'outcome': 'REFUSED', 'reason': reason, **extra}
        append_ledger(rec, ledger)
        return 2, rec

    try:
        admission = qualify(kernel, version, receipt, review)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return refuse('admission: ' + str(exc))
    states = attempt_states(read_ledger(ledger), sub_id, kernel, version)
    unresolved = [a for a, r in states.items() if r.get('outcome') in OPEN_STATES]
    if unresolved:
        return refuse('unresolved earlier attempt for this ID or kernel/version; run --reconcile first', attempts=unresolved)
    rows = read_submissions(cli)
    if rows is None:
        return refuse('submission list unreadable or malformed')
    by_ref = {r['ref']: r for r in rows}
    submitted = [r for r in states.values() if r.get('outcome') == 'SUBMITTED']
    live = [r['ref'] for r in submitted if not (r.get('ref') in by_ref and is_error(by_ref[r['ref']]))]
    if live:
        return refuse('this ID or kernel/version was already submitted', existing=live)
    dup = [r['ref'] for r in rows if carries(r, sub_id) and not is_error(r)]
    if dup:
        return refuse('duplicate: a non-ERROR list row already carries this ID', existing=dup)
    earlier = sorted(states) + [r['ref'] for r in rows if carries(r, sub_id)]
    if earlier and not user_approved_retry:
        return refuse("already attempted once (one attempt per candidate); report to the user; a retry needs "
                      "--user-approved-retry with the user's words", earlier=earlier)
    if used_today(rows, today) >= daily_limit:
        return refuse(f'daily limit {daily_limit} reached')
    try:
        # Listing may take time; check again immediately before IN_FLIGHT.
        _need(qualify(kernel, version, receipt, review) == admission, 'admission inputs changed')
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return refuse('admission recheck: ' + str(exc))
    attempt = uuid.uuid4().hex
    before = set(by_ref)
    append_ledger({**_base(sub_id, kernel, version, attempt), 'outcome': 'IN_FLIGHT', 'description': desc,
                   'user_approved_retry': user_approved_retry, 'admission': admission}, ledger)
    rec = {**_base(sub_id, kernel, version, attempt), 'admission': admission}
    try:
        code, out = cli('competitions', 'submit', COMP, '-k', kernel, '-v', str(version), '-f', 'submission.csv', '-m', desc)
        rec['cli_exit'] = code
        rec['cli_last_line'] = out.splitlines()[-1][:300] if out else ''
    except Exception as e:  # e.g. subprocess.TimeoutExpired: the request may still have reached Kaggle
        rec['cli_exception'] = f'{type(e).__name__}: {str(e)[:200]}'
    for _ in range(poll_tries):
        try:
            sleep(poll_seconds)
            rows = read_submissions(cli)
        except Exception:
            rows = None
        if rows is None:
            continue
        new = [r for r in rows if r['ref'] not in before and carries(r, sub_id)]
        if new:
            rec.update(outcome='SUBMITTED', ref=new[0]['ref'], status=new[0]['status'], time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            append_ledger(rec, ledger)
            return 0, rec
    rec.update(outcome='UNCONFIRMED', reason='no new row with this ID after polling; blocked until --reconcile',
               time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    append_ledger(rec, ledger)
    return 3, rec


def reconcile(sub_id, kernel, version, declare_absent=False, cli=kaggle_cli, ledger=LEDGER, lock=LOCK):
    ledger, lock = Path(ledger), Path(lock)
    with Lock(lock) as held:
        if held is None:
            return 2, {**_base(sub_id, kernel, version), 'outcome': 'REFUSED', 'reason': f'lock exists ({lock})'}
        states = attempt_states(read_ledger(ledger), sub_id, kernel, version)
        open_attempts = [a for a, r in states.items() if r.get('outcome') in OPEN_STATES]
        if not open_attempts:
            return 0, {**_base(sub_id, kernel, version), 'outcome': 'NOTHING_OPEN'}
        rows = read_submissions(cli)
        if rows is None:
            return 2, {**_base(sub_id, kernel, version), 'outcome': 'REFUSED', 'reason': 'submission list unreadable or malformed'}
        found = [r for r in rows if carries(r, sub_id)]
        if found:
            last = None
            for a in open_attempts:
                last = {**_base(sub_id, kernel, version, a), 'outcome': 'SUBMITTED', 'ref': found[0]['ref'],
                        'status': found[0]['status'], 'via': 'reconcile'}
                append_ledger(last, ledger)
            return 0, last
        if not declare_absent:
            return 3, {**_base(sub_id, kernel, version), 'outcome': 'STILL_OPEN', 'reason': 'no list row carries this ID; if the '
                       'Kaggle website also shows none, rerun with --declare-absent', 'attempts': open_attempts}
        last = None
        for a in open_attempts:
            last = {**_base(sub_id, kernel, version, a), 'outcome': 'RESOLVED_ABSENT', 'via': 'reconcile --declare-absent'}
            append_ledger(last, ledger)
        return 0, last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--id', required=True)
    ap.add_argument('--kernel', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--message')
    ap.add_argument('--receipt', help='required to submit: visible-run receipt JSON')
    ap.add_argument('--review', help='required to submit: pre-submission PASS review')
    ap.add_argument('--check-only', action='store_true', help='offline admission only; no endpoints, lock or ledger')
    ap.add_argument('--daily-limit', type=int, default=5)
    ap.add_argument('--reconcile', action='store_true')
    ap.add_argument('--declare-absent', action='store_true')
    ap.add_argument('--user-approved-retry', default=None, help="the user's approval, quoted with its time, for retrying a candidate")
    a = ap.parse_args()
    print(f'kaggle_submit_once v{VERSION}', flush=True)
    if a.check_only:
        if a.reconcile or a.declare_absent:
            ap.error('--check-only cannot reconcile')
        try:
            rec = {'outcome': 'ADMITTED_OFFLINE', 'admission': qualify(a.kernel, a.version, a.receipt, a.review)}
            code = 0
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            code, rec = 2, {'outcome': 'REFUSED', 'reason': 'admission: ' + str(exc)}
    elif a.reconcile:
        code, rec = reconcile(a.id, a.kernel, a.version, a.declare_absent)
    else:
        if not a.message:
            ap.error('--message is required to submit')
        if not a.receipt or not a.review:
            ap.error('--receipt and --review are required to submit')
        code, rec = submit_once(a.id, a.kernel, a.version, a.message, daily_limit=a.daily_limit,
                                user_approved_retry=a.user_approved_retry, receipt=a.receipt, review=a.review)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
