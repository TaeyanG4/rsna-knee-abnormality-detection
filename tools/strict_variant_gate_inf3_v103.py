"""H197 INF3 admission-cycle repair v1.0.3. No launch or transport action.

Immutable subject E has no reviews; envelope V attaches completed reviews of E.
Preserved scientific source/registration/replay bytes are never refrozen.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import strict_variant_gate_inf3_v1 as old
from tools import native_codex_dispatch_record_v1 as native
from tools.strict_variant_gate import verdict

b = old.b
need, sha = b.need, b.sha
VERSION = '1.0.3'
PHASES = old.PHASES
FROZEN = '.local/scratch/slot-b/batch5-1018/inf3-release/fix1/FROZEN-EQUIVALENCE.json'
FROZEN_SHA = '72e91f232efcd76ee8ac518a55e9e5ba765284af5fdafb50491cd1bdeaee350d'
SOURCE_PATHS = (*b.SOURCE_FILES, 'tools/strict_variant_gate_inf3_v103.py',
                'tests/test_hb197_inf3_fix1.py', 'tools/strict_variant_gate.py')
ROLES = {'kernel', 'metadata', 'build', 'registration', 'cpu', 'legacy_tests',
         'bridge_registration', 'bridge_result', 'parent_gate'}
SUBJECT_FIELDS = {'schema', 'candidate', 'slug', 'version', 'kernel_sha256', 'phase',
                  'formula', 'fast', 'fixture_only', 'remote_version', 'frozen', 'sources', 'artifacts'}
ATTEST_FIELDS = {'schema', 'candidate', 'slug', 'version', 'kernel_sha256', 'phase',
                'builder', 'reviewer', 'verdict', 'evidence_sha256', 'checked_source_sha256',
                'remote_version', 'checks'}


def ref(path, root=ROOT):
    path = native.local(path, root)
    return {'path': path.relative_to(Path(root).resolve()).as_posix(), 'sha256': sha(path)}


class Inputs:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.consumed = {}

    def file(self, item):
        need(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'exact path/hash reference')
        need(isinstance(item['path'], str) and not Path(item['path']).is_absolute() and
             re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None, 'relative path and SHA256 required')
        path = native.local(item['path'], self.root); key = path.relative_to(self.root).as_posix()
        need(key == item['path'] and (key not in self.consumed or self.consumed[key] == item['sha256']), 'canonical consistent evidence path')
        need(sha(path) == item['sha256'], 'stale evidence: ' + key)
        self.consumed[key] = item['sha256']
        return path

    def json(self, item):
        return native.parse(self.file(item).read_bytes())

    def pins(self, values):
        need(isinstance(values, dict) and values, 'nonempty source/evidence closure')
        for path, digest in values.items():
            self.file({'path': path, 'sha256': digest})

    def recheck(self):
        self.pins(self.consumed.copy())


def native_chain(inputs, receipt_ref, report_ref):
    receipt = inputs.json(receipt_ref); report = inputs.file(report_ref)
    need(native.local(receipt['out'], inputs.root) == report and receipt['report_sha256'] == report_ref['sha256'], 'native report binding')
    completion_path = native.local(receipt['completion_source'], inputs.root)
    completion_ref = {'path': completion_path.relative_to(inputs.root).as_posix(), 'sha256': receipt['completion_evidence_sha256']}
    completion = inputs.json(completion_ref)
    prompt = native.local(completion['prompt'], inputs.root)
    inputs.file({'path': prompt.relative_to(inputs.root).as_posix(), 'sha256': receipt['prompt_sha256']})
    need(re.fullmatch('native-astra-[br][1-9][0-9]*', receipt['lane']) is not None and
         re.fullmatch('[0-9a-f]{32}', receipt['run_id']) is not None, 'native archive identity')
    prefix = '.local/astra-improve/lanes/runs/' + receipt['lane'] + '/' + receipt['run_id'] + '/'
    for name, digest in [('receipt.json', receipt_ref['sha256']), ('completion.json', completion_ref['sha256']),
                         ('prompt.txt', receipt['prompt_sha256']), ('report.md', report_ref['sha256']), ('run.log', receipt['log_sha256'])]:
        inputs.file({'path': prefix + name, 'sha256': digest})
    return native.verify(inputs.file(receipt_ref), inputs.root)


def review(inputs, item, kernel, phase, subject_sha, sources, remote):
    value = inputs.json(item)
    need(set(value) == ATTEST_FIELDS | {'report', 'native_receipt'}, 'exact complete review sidecar')
    actual = native_chain(inputs, value['native_receipt'], value['report'])
    text = verdict(inputs.file(value['report']).read_bytes())
    need(re.findall(r'^[ \t]*VERDICT[ \t]*:.*$', text, re.M) == ['VERDICT: PASS'], 'ambiguous review verdict')
    prefix = 'H197-REVIEW-ATTESTATION: '
    lines = [line for line in text.splitlines() if line.lstrip().startswith('H197-REVIEW-ATTESTATION')]
    need(len(lines) == 1 and lines[0].startswith(prefix), 'one exact actual report attestation required')
    bound = native.parse(lines[0][len(prefix):].encode())
    need(bound == {k: value[k] for k in ATTEST_FIELDS}, 'report/sidecar mismatch')
    need(value['schema'] == 'h197-independent-review/v2' and value['candidate'] == b.KEY and
         value['slug'] == b.SLUG and value['version'] == b.VERSION and value['kernel_sha256'] == kernel and
         value['phase'] == phase and value['builder'] == b.BUILDER and value['reviewer'] == actual['agent_id'] and
         value['reviewer'] != b.BUILDER and value['verdict'] == 'PASS' and value['evidence_sha256'] == subject_sha and
         value['checked_source_sha256'] == sources and value['remote_version'] == remote and
         isinstance(value['checks'], dict), 'exact independent review identity/phase/evidence/source/remote')
    inputs.pins(sources)
    return value


def base(inputs, subject_ref, fixture_frozen=None):
    e = inputs.json(subject_ref)
    need(set(e) == SUBJECT_FIELDS and isinstance(e['artifacts'], dict) and e['artifacts'] and
         not any('review' in key.lower() for key in e['artifacts']), 'subject excludes reviews; old cyclic contract refused')
    anchor = {'path': FROZEN, 'sha256': FROZEN_SHA} if fixture_frozen is None else fixture_frozen
    need(e['frozen'] == anchor, 'fixed unchanged scientific equivalence required')
    f = inputs.json(anchor)
    need(f.get('schema') == 'h197-fix1-equivalence/v1' and f.get('candidate') == b.KEY and
         f.get('fixture_only') is (fixture_frozen is not None) and set(f.get('roles', {})) == ROLES and
         f.get('original_registration_bindings') == 368, 'original scientific closure scope')
    inputs.pins(f['bindings'])
    for item in f['roles'].values():
        need(f['bindings'].get(item['path']) == item['sha256'], 'mandatory frozen result absent')
        inputs.file(item)
    records = {k: inputs.json(v) for k, v in f['roles'].items() if k not in ('kernel', 'metadata')}
    kernel = f['roles']['kernel']['sha256']; reg = records['registration']; build = records['build']; cpu = records['cpu']
    need(reg['candidate'] == b.KEY and len(reg['bindings']) == 368 and reg['source_sha256'] == f['original_source_sha256'] and
         all(f['bindings'].get(p) == h for p, h in reg['bindings'].items()), 'unchanged original registration/source closure')
    need(build['kernel_sha256'] == kernel and build['metadata_sha256'] == f['roles']['metadata']['sha256'] and
         build['registration_sha256'] == f['roles']['registration']['sha256'] and
         build['source_sha256'] == f['original_source_sha256'] and build['unchanged'] == 28 and
         len(build['all_embedded']) == 31, 'unchanged scientific construction')
    need(cpu['status'] == 'PASS_EXTRACTED_REAL58_REPLAY' and cpu['kernel_sha256'] == kernel and
         cpu['source_sha256'] == f['original_source_sha256'] and cpu['registration_sha256'] == f['roles']['registration']['sha256'] and
         cpu['bridge_result_sha256'] == f['roles']['bridge_result']['sha256'] and all(cpu.get(k) is True for k in
         ('exact_q3_bytes', 'exact_final_float64', 'parent_noop_float64_and_csv_exact')), 'unchanged actual scientific replay')
    need(records['bridge_result']['proxy_gate'] == 'PASS', 'original proxy decision')
    parent = records['parent_gate']
    need(parent['status'] == 'PASS' and parent['fixture_only'] is False and not parent['errors'] and
         all(f['bindings'].get(p) == h for p, h in parent['input_sha256'].items()), 'original authentic parent closure')
    sources = e['sources']; need(set(SOURCE_PATHS) <= set(sources), 'complete current source map')
    inputs.pins(sources)
    if fixture_frozen is None:
        need(sources['tools/strict_variant_gate_inf3_v103.py'] == sha(__file__), 'loaded gate source identity')
    for item in e['artifacts'].values():
        inputs.file(item)
    tests = inputs.json(e['artifacts']['tests']); log = inputs.file(tests['log'])
    need(tests.get('schema') == 'h197-fix1-tests/v1' and tests.get('actual_run') is True and
         tests.get('exit_code') == 0 and tests.get('failed') == 0 and type(tests.get('passed')) is int and
         tests['passed'] > 0 and tests.get('kernel_sha256') == kernel and
         tests.get('source_sha256') == {p: sources[p] for p in SOURCE_PATHS}, 'current actual scoped test receipt')
    need(re.search(r'\b' + str(tests['passed']) + r' passed\b', log.read_text(encoding='utf-8')) is not None, 'test log mismatch')
    return e, f, records, kernel, sources


def _evaluate(phase, subject_ref, envelope_ref, root, fixture_frozen=None):
    root = Path(root).resolve(); fixture = root != ROOT.resolve()
    need(phase in PHASES and ((fixture and fixture_frozen is not None and root.is_relative_to(ROOT / '.local/scratch')) or
         (not fixture and fixture_frozen is None)), 'invalid production/fixture boundary')
    inputs = Inputs(root); tracked = []; token = old.CHECKED.set(tracked)
    try:
        e, f, records, kernel, sources = base(inputs, subject_ref, fixture_frozen)
        need(e['schema'] == 'h197-subject/v2' and e['candidate'] == b.KEY and e['slug'] == b.SLUG and
             e['version'] == b.VERSION and e['kernel_sha256'] == kernel and e['phase'] == phase and
             e['formula'] == 'H189_F_Q3' and e['fast'] is False and e['fixture_only'] is fixture, 'exact phase subject identity')
        remote = e['remote_version']
        need((phase == 'BUILD_PASS' and remote is None) or
             (phase != 'BUILD_PASS' and type(remote) is int and remote > 0), 'phase remote-version identity')
        envelope = inputs.json(envelope_ref)
        required = {'schema', 'subject', 'primary'} | ({'minimum', 'integration'} if phase == 'SUBMISSION_PASS' else set())
        need(set(envelope) == required and envelope['schema'] == 'h197-review-envelope/v1' and
             envelope['subject'] == subject_ref, 'complete exact acyclic envelope')
        rv = review(inputs, envelope['primary'], kernel, phase, subject_ref['sha256'], sources, remote)
        result = dict(schema='h197-inf3-gate/v2', gate_version=VERSION, status=phase, phase=phase,
            candidate=b.KEY, slug=b.SLUG, version=b.VERSION, kernel_sha256=kernel,
            metadata_sha256=f['roles']['metadata']['sha256'], gate_sha256=sources['tools/strict_variant_gate_inf3_v103.py'],
            fixture_only=fixture, qualified_for_submission=False, fast=False, subject=subject_ref,
            envelope=envelope_ref, source_sha256=sources, remote_version=remote)
        refs = e['artifacts']
        if phase != 'BUILD_PASS':
            targets = records['bridge_registration']['targets']
            for key, kind in [('full_visible', 'FULL_VISIBLE'), ('n32', 'N32')]:
                execution = old.execution(refs[key], kind, kernel, result['metadata_sha256'], targets, root)
                need(execution['remote_version'] == remote, 'exact visible/N32 remote version')
            result['runtime'] = old.measured_runtime(refs['runtime'], refs['n32'], kernel, root)
            old.memory(refs['memory'], kernel, root)
            need(rv['checks'].get('joint_adverse_gold_reviewed') is True and rv['checks'].get('all_new_path_gates_checked') is True,
                 'actual independent result/new-path review')
        if phase == 'SUBMISSION_PASS':
            minimum = review(inputs, envelope['minimum'], kernel, 'SUBMISSION_MINIMUM', subject_ref['sha256'], sources, remote)
            need(minimum['checks'].get('duplicate_clear') is True and minimum['checks'].get('description_accurate') is True,
                 'minimum exact version/duplicates/description')
            integration = inputs.json(refs['canonical_integration']); sender = inputs.file(refs['sender'])
            need(refs['sender']['path'] == 'tools/kaggle_submit_once.py' and sources.get(refs['sender']['path']) == refs['sender']['sha256'],
                 'actual canonical sender in complete source map')
            need(integration.get('schema') == 'h197-canonical-integration/v1' and integration.get('status') == 'PASS' and
                 integration.get('candidate') == b.KEY and integration.get('canonical_sha256') == refs['sender']['sha256'] and
                 integration.get('gate_sha256') == result['gate_sha256'] and integration.get('phase') == phase and
                 all(integration.get(k) is True for k in ('mandatory_nonfast_gate', 'late_recheck', 'shared_sender')), 'actual current canonical integration')
            review(inputs, envelope['integration'], kernel, 'CANONICAL_INTEGRATION', refs['canonical_integration']['sha256'], sources, remote)
            it = inputs.json(integration['tests']); inputs.file(it['log'])
            need(it.get('exit_code') == 0 and it.get('failed') == 0 and it.get('actual_run') is True and
                 all(it.get(k) is True for k in ('nonfast_bypass_rejected', 'wrong_candidate_rejected', 'wrong_phase_rejected',
                 'stale_source_rejected', 'late_recheck_verified', 'shared_sender_verified')) and
                 it.get('source_sha256', {}).get('tools/kaggle_submit_once.py') == refs['sender']['sha256'] and
                 it.get('source_sha256', {}).get('tools/strict_variant_gate_inf3_v103.py') == result['gate_sha256'], 'actual complete canonical integration tests')
            scan = inputs.json(refs['secret_scan'])
            need(scan.get('version') == '1.0.2' and scan.get('status') == 'PASS' and scan.get('kernel_sha256') == kernel, 'broad current secret scan')
            result['canonical_sha256'] = sha(sender)
            result['qualified_for_submission'] = not fixture
        # Unchanged scientific parsers use old.file; merge their actual complete
        # reads into the same closure before the mandatory final hash pass.
        for path, digest in tracked:
            inputs.file({'path': path.resolve().relative_to(root).as_posix(), 'sha256': digest})
        inputs.recheck()
        result['consumed_sha256'] = dict(sorted(inputs.consumed.items()))
        return result
    finally:
        old.CHECKED.reset(token)


def evaluate(phase, subject_ref, envelope_ref):
    """Production offline evaluation, fixed project root, no fixture override."""
    return _evaluate(phase, subject_ref, envelope_ref, ROOT)


def evaluate_fixture(phase, subject_ref, envelope_ref, *, root, frozen):
    """Explicit TEST-ONLY full parser entrypoint; never returns admission."""
    need(Path(root).resolve() != ROOT.resolve(), 'fixture cannot use production root')
    return _evaluate(phase, subject_ref, envelope_ref, root, frozen)


def verify_receipt(receipt_ref, *, phase='SUBMISSION_PASS', slug=None, remote_version=None):
    """Canonical offline API: exact replay and complete late recheck, no send."""
    inputs = Inputs(); saved = inputs.json(receipt_ref)
    need(saved.get('schema') == 'h197-inf3-gate/v2' and saved.get('fixture_only') is False and
         saved.get('phase') == phase and saved.get('slug') == (slug or b.SLUG), 'exact nonfixture receipt required')
    if phase == 'SUBMISSION_PASS':
        need(type(remote_version) is int and saved.get('remote_version') == remote_version and
             saved.get('qualified_for_submission') is True, 'exact submission remote version required')
    fresh = evaluate(phase, saved['subject'], saved['envelope'])
    need(fresh == saved, 'receipt replay differs')
    inputs.pins(saved['consumed_sha256']); inputs.recheck()
    return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=PHASES); parser.add_argument('--subject', required=True)
    parser.add_argument('--envelope', required=True); parser.add_argument('--out')
    args = parser.parse_args()
    try:
        result = evaluate(args.phase, ref(args.subject), ref(args.envelope))
        if args.out:
            b.write(native.local(args.out, ROOT), b.encoded(result))
        print(json.dumps(result, sort_keys=True)); return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status='HOLD', reason=str(error), qualified_for_submission=False))); return 2


if __name__ == '__main__':
    raise SystemExit(main())
