"""INF3 diagnostic qualification adapter v1.0.0. Offline; never submits.

Distinct release and diagnostic slug/version identities are deliberate. This is
not a replacement for v103 SUBMISSION_PASS or canonical sender integration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from tools import hb197_inf3_qualification_v1 as b
from tools import hb197_inf3_collect_v1 as collect
from tools import strict_variant_gate_inf3_v103 as old
from tools.strict_variant_gate import verdict

ROOT = b.ROOT
VERSION = '1.0.0'
PHASES = ('LAUNCH_READY', 'QUALIFICATION_READY')
need = b.need
ATTEST = {'schema', 'candidate', 'phase', 'builder', 'reviewer', 'verdict',
          'subject_sha256', 'source_sha256', 'release', 'diagnostic', 'checks'}
CHECKS = {
    'LAUNCH_READY': {'minimal_root_delta', 'ordinary_root_parity', 'same32_all_branches',
                    'full_original_paths_tta', 'complete_wall_contract', 'watchdog_owned_children',
                    'cpu_tests_and_diff', 'separate_remote_identities'},
    'QUALIFICATION_READY': {'actual_all_children_logs_exits', 'actual_same_cohort_all_branches',
                    'all20_checkpoint_tta_cost_identities', 'actual_full_shape_cuda_device_convention',
                    'actual_codecs_and_original_fallback_policy', 'sampled_memory_limits_honest',
                    'actual_complete_wall_and_full_global_cost', 'separate_remote_identities'}}


def review(inputs, item, subject_ref, subject):
    value = inputs.json(item)
    need(set(value) == ATTEST | {'report', 'native_receipt'}, 'exact qualification review fields')
    actual = old.native_chain(inputs, value['native_receipt'], value['report'])
    text = verdict(inputs.file(value['report']).read_bytes())
    need(re.findall(r'^[ \t]*VERDICT[ \t]*:.*$', text, re.M) == ['VERDICT: PASS'], 'single PASS verdict')
    prefix = 'H197-QUALIFICATION-ATTESTATION: '
    lines = [x for x in text.splitlines() if x.lstrip().startswith('H197-QUALIFICATION-ATTESTATION')]
    need(len(lines) == 1 and lines[0].startswith(prefix), 'one exact qualification attestation')
    need(old.native.parse(lines[0][len(prefix):].encode()) == {k: value[k] for k in ATTEST}, 'actual report attestation mismatch')
    need(value['schema'] == 'h197-qualification-review/v1' and value['candidate'] == b.science.KEY and
         value['phase'] == subject['phase'] and value['builder'] == b.BUILDER and
         value['reviewer'] == actual['agent_id'] and value['reviewer'] != b.BUILDER and
         value['verdict'] == 'PASS' and value['subject_sha256'] == subject_ref['sha256'] and
         value['source_sha256'] == subject['sources'] and value['release'] == subject['release'] and
         value['diagnostic'] == subject['diagnostic'], 'actual different reviewer and exact scope')
    need(set(value['checks']) == CHECKS[subject['phase']] and all(v is True for v in value['checks'].values()),
         'all technical review checks required; no efficacy veto')
    return value


def _evaluate(phase, subject_ref, envelope_ref, root, *, fixture_frozen=None):
    root = Path(root).resolve(); fixture = root != ROOT.resolve()
    need(phase in PHASES and ((fixture and fixture_frozen is not None and root.is_relative_to(ROOT / '.local/scratch')) or
         (not fixture and fixture_frozen is None)), 'fixture boundary')
    inputs = old.Inputs(root); e = inputs.json(subject_ref)
    fields = {'schema', 'candidate', 'phase', 'fixture_only', 'sources', 'prior_build', 'build',
              'diff', 'tests', 'preservation', 'release', 'diagnostic', 'artifacts'}
    need(set(e) == fields and e['schema'] == 'h197-qualification-subject/v1' and
         e['candidate'] == b.science.KEY and e['phase'] == phase and e['fixture_only'] is fixture, 'exact acyclic qualification subject')
    prior_saved = inputs.json(e['prior_build'])
    if fixture:
        replay = old.evaluate_fixture('BUILD_PASS', prior_saved['subject'], prior_saved['envelope'], root=root, frozen=fixture_frozen)
    else:
        need(e['prior_build'] == old.ref(b.BUILD), 'fixed original scientific BUILD prerequisite')
        replay = old.verify_receipt(e['prior_build'], phase='BUILD_PASS')
    need(replay == prior_saved and replay['kernel_sha256'] == b.KERNEL, 'actual prior BUILD replay')
    inputs.pins(replay['consumed_sha256'])
    build = inputs.json(e['build']); expected = b.construction()
    need(build == expected and e['sources'] == b.sources(), 'exact built diagnostic and current sources')
    inputs.pins(e['sources'])
    nb, metadata, files, proof, diff, ndiff = b.payload()
    need(inputs.json(e['diff']) == proof, 'full exact reviewed diagnostic diff')
    constructed = {'diagnostic-kernel/release.ipynb': b.encoded(nb), 'diagnostic-kernel/kernel-metadata.json': b.encoded(metadata),
                   'FULL-EMBEDDED.diff': diff.encode(), 'FULL-NOTEBOOK.diff': ndiff.encode(),
                   **{'diagnostic-runtime/' + n: data for n, data in files.items()}}
    prefix = b.WORK.relative_to(ROOT).as_posix() + '/'
    for name, data in constructed.items():
        inputs.file({'path': prefix + name, 'sha256': b.science.digest(data)})
    preservation = inputs.json(e['preservation'])
    need(preservation['schema'] == 'h197-qualification-preservation/v1' and
         preservation['prior_build_sha256'] == e['prior_build']['sha256'] and
         preservation['bindings'] == replay['consumed_sha256'], 'complete prior closure preservation')
    inputs.pins(preservation['bindings'])
    tests = inputs.json(e['tests']); test_log = inputs.file(tests['log'])
    need(tests['schema'] == 'h197-qualification-tests/v1' and tests['fixture_only'] is fixture and
         tests['actual_run'] is True and tests['exit_code'] == tests['failed'] == 0 and
         type(tests['passed']) is int and tests['passed'] > 0 and tests['source_sha256'] == e['sources'] and
         re.search(r'\b' + str(tests['passed']) + r' passed\b', test_log.read_text()) is not None, 'actual current test evidence')
    for key, slug, kernel, meta in [('release', b.science.SLUG, b.KERNEL, b.METADATA),
         ('diagnostic', b.SLUG, build['diagnostic_kernel_sha256'], build['diagnostic_metadata_sha256'])]:
        identity = e[key]
        need(set(identity) == {'slug', 'version', 'kernel_sha256', 'metadata_sha256'} and
             identity['slug'] == slug and identity['kernel_sha256'] == kernel and identity['metadata_sha256'] == meta,
             'separate exact release/diagnostic identity')
        version = identity['version']
        need(version is None if phase == 'LAUNCH_READY' else type(version) is int and version > 0, 'phase remote version')
    env = inputs.json(envelope_ref)
    need(set(env) == {'schema', 'subject', 'review'} and env['schema'] == 'h197-qualification-envelope/v1' and
         env['subject'] == subject_ref, 'exact acyclic envelope')
    review(inputs, env['review'], subject_ref, e)
    result = dict(schema='h197-qualification-gate/v1', gate_version=VERSION, phase=phase,
        status=phase, fixture_only=fixture, candidate=b.science.KEY, subject=subject_ref, envelope=envelope_ref,
        release=e['release'], diagnostic=e['diagnostic'], source_sha256=e['sources'],
        qualified_for_submission=False, launch_authorized=False,
        next_required='B live quota/candidate/resource/scan decision' if phase == 'LAUNCH_READY' else
                      'independent minimum review plus separately versioned canonical adapter; no sender change here')
    if phase == 'LAUNCH_READY':
        need(e['artifacts'] == {}, 'no invented launch execution evidence')
    else:
        need(set(e['artifacts']) == {'full_visible', 'n32', 'population'}, 'exact completed qualification artifacts')
        visible = collect.execution(inputs, e['artifacts']['full_visible'], build, fixture=fixture)
        n32 = collect.execution(inputs, e['artifacts']['n32'], build, fixture=fixture)
        for key, run in [('release', visible), ('diagnostic', n32)]:
            need(run['slug'] == e[key]['slug'] and run['remote_version'] == e[key]['version'], 'separate actual remote binding')
        population = inputs.json(e['artifacts']['population']); inputs.file(population['source'])
        runtime = collect.projection(n32, population)
        need(runtime['status'] == 'PASS', 'measured hidden projection exceeds9h')
        result.update(full_visible=visible, n32=n32, runtime=runtime)
    inputs.recheck(); result['consumed_sha256'] = dict(sorted(inputs.consumed.items()))
    return result


def evaluate(phase, subject_ref, envelope_ref):
    return _evaluate(phase, subject_ref, envelope_ref, ROOT)


def evaluate_fixture(phase, subject_ref, envelope_ref, *, root, frozen):
    need(Path(root).resolve() != ROOT.resolve(), 'fixture cannot use production root')
    return _evaluate(phase, subject_ref, envelope_ref, root, fixture_frozen=frozen)


def verify_receipt(receipt_ref, *, phase, release_version=None, diagnostic_version=None):
    """Future canonical adapter must call this and late_recheck; no admission here."""
    inputs = old.Inputs(); saved = inputs.json(receipt_ref)
    need(saved['schema'] == 'h197-qualification-gate/v1' and saved['fixture_only'] is False and saved['phase'] == phase,
         'production exact phase only')
    need(saved['release']['version'] == release_version and saved['diagnostic']['version'] == diagnostic_version,
         'caller must bind both actual remote versions')
    current = evaluate(phase, saved['subject'], saved['envelope'])
    need(current == saved, 'qualification receipt replay differs')
    inputs.pins(saved['consumed_sha256']); inputs.recheck()
    return current


def late_recheck(result):
    need(result['fixture_only'] is False and result['schema'] == 'h197-qualification-gate/v1', 'production recheck only')
    inputs = old.Inputs(); inputs.pins(result['consumed_sha256']); inputs.recheck()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=PHASES); parser.add_argument('--subject', required=True)
    parser.add_argument('--envelope', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.phase, old.ref(args.subject), old.ref(args.envelope))
        b.write(old.native.local(args.out, ROOT), b.encoded(result))
        print(json.dumps(dict(status=result['status'], consumed=len(result['consumed_sha256']), qualified_for_submission=False)))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps(dict(status='HOLD', reason=str(error), qualified_for_submission=False))); return 2


if __name__ == '__main__': raise SystemExit(main())
