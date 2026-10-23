"""C4 admission repair v1.0.2: archived report approval and complete result seal.

Frozen scientific code/results are not regenerated. No launch or account action.
The coordinator-attested native archive is the existing trust boundary, not a
platform signature. An isolated fixture can never return production admission.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import strict_variant_gate_c4_v1 as old
from tools import native_codex_dispatch_record_v1 as native
from tools.strict_variant_gate import verdict

b = old.b
need, finite = b.need, old.finite
VERSION = '1.0.2'
PHASES = old.PHASES
REVIEW_PHASE = dict(zip(PHASES, ('BUILD', 'CACHE_LAUNCH', 'QUALIFICATION', 'SUBMISSION')))
REVIEW_PHASES = {*REVIEW_PHASE.values(), 'GOLD_RESULT', 'SUBMISSION_MINIMUM', 'CANONICAL_INTEGRATION'}
PREFIX = 'H198-REVIEW-ATTESTATION: '
FROZEN = '.local/scratch/slot-b/batch5-1018/c4-bridge/fix1/FROZEN-EQUIVALENCE-v102.json'
FROZEN_SHA = '73b05ccd27862d52e194a876676b4d1ce4e72a5422cc30d003d7345b560a68c8'
SOURCE_PATHS = ('tools/strict_variant_gate_c4_v102.py', 'tests/test_hb198_c4_fix1.py',
                'tools/native_codex_dispatch_record_v1.py', 'tools/strict_variant_gate.py',
                'tools/strict_variant_gate_c4_v1.py', 'tools/c85r7_r6w958c4_v1147.py',
                'tools/hb198_c4_bridge_v1.py', 'tools/hb198_c4_gold58_v1.py',
                'tools/hb145_output_io.py')
ROLES = {'kernel', 'metadata', 'build', 'pins', 'parity', 'strict', 'oof', 'assets',
         'parent_replay', 'legacy_tests', 'legacy_card', 'parent_gate'}
ATTEST_FIELDS = {'schema', 'candidate', 'slug', 'version', 'kernel_sha256', 'builder',
                'reviewer', 'phase', 'verdict', 'evidence_sha256', 'source_sha256',
                'remote_version', 'checks', 'auxiliary_reviews'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path, root=ROOT):
    path = native.local(path, root)
    return {'path': path.relative_to(Path(root).resolve()).as_posix(), 'sha256': sha(path)}


class Inputs:
    """Hash before parsing; reject inconsistent aliases and recheck at return."""
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.consumed = {}

    def file(self, item):
        need(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'exact path/hash reference required')
        need(isinstance(item['path'], str) and not Path(item['path']).is_absolute(), 'relative evidence path required')
        need(re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None, 'SHA256 required')
        path = native.local(item['path'], self.root)
        key = path.relative_to(self.root).as_posix()
        need(key == item['path'], 'canonical relative path required')
        need(key not in self.consumed or self.consumed[key] == item['sha256'], 'conflicting evidence reference')
        need(sha(path) == item['sha256'], 'stale evidence: ' + key)
        self.consumed[key] = item['sha256']
        return path

    def json(self, item):
        return native.parse(self.file(item).read_bytes())

    def pins(self, mapping):
        need(isinstance(mapping, dict) and bool(mapping), 'nonempty source/input closure required')
        for path, digest in mapping.items():
            self.file({'path': path, 'sha256': digest})

    def recheck(self):
        for path, digest in self.consumed.copy().items():
            self.file({'path': path, 'sha256': digest})


def archive(inputs, receipt_ref, report_ref):
    """Register every verifier read before invoking the full native verifier."""
    receipt = inputs.json(receipt_ref)
    report = inputs.file(report_ref)
    path = inputs.file(receipt_ref)
    need(native.local(receipt['out'], inputs.root) == report, 'native report path mismatch')
    need(receipt['report_sha256'] == report_ref['sha256'], 'native report digest mismatch')
    source = native.local(receipt['completion_source'], inputs.root)
    source_ref = {'path': source.relative_to(inputs.root).as_posix(), 'sha256': receipt['completion_evidence_sha256']}
    completion = inputs.json(source_ref)
    prompt = native.local(completion['prompt'], inputs.root)
    inputs.file({'path': prompt.relative_to(inputs.root).as_posix(), 'sha256': receipt['prompt_sha256']})
    need(re.fullmatch('native-astra-[br][1-9][0-9]*', receipt['lane']) is not None and
         re.fullmatch('[0-9a-f]{32}', receipt['run_id']) is not None, 'native archive identity')
    prefix = '.local/astra-improve/lanes/runs/' + receipt['lane'] + '/' + receipt['run_id'] + '/'
    for name, digest in [('receipt.json', receipt_ref['sha256']), ('completion.json', source_ref['sha256']),
                         ('prompt.txt', receipt['prompt_sha256']), ('report.md', report_ref['sha256']),
                         ('run.log', receipt['log_sha256'])]:
        inputs.file({'path': prefix + name, 'sha256': digest})
    return native.verify(path, inputs.root)


def review(item, kernel, phase, evidence_sha, sources, remote_version=None, *, inputs=None):
    inputs = inputs or Inputs()
    need(phase in REVIEW_PHASES, 'unknown review phase')
    wrapper = inputs.json(item)
    need(set(wrapper) == ATTEST_FIELDS | {'report', 'native_receipt'}, 'complete exact review schema')
    actual = archive(inputs, wrapper['native_receipt'], wrapper['report'])
    text = verdict(inputs.file(wrapper['report']).read_bytes())
    need(re.findall(r'^[ \t]*VERDICT[ \t]*:.*$', text, re.M) == ['VERDICT: PASS'],
         'ambiguous indented review verdict')
    lines = [line for line in text.splitlines() if line.lstrip().startswith('H198-REVIEW-ATTESTATION')]
    need(len(lines) == 1 and lines[0].startswith(PREFIX), 'exactly one unambiguous attestation required')
    bound = native.parse(lines[0][len(PREFIX):].encode('utf-8'))
    need(bound == {k: wrapper[k] for k in ATTEST_FIELDS}, 'report/sidecar mismatch')
    need(bound['schema'] == 'h198-independent-review/v2' and bound['candidate'] == b.KEY and
         bound['slug'] == b.SLUG and bound['version'] == b.VERSION and bound['kernel_sha256'] == kernel and
         bound['builder'] == b.BUILDER and bound['reviewer'] == actual['agent_id'] and
         bound['reviewer'] != b.BUILDER and bound['phase'] == phase and bound['verdict'] == 'PASS' and
         bound['evidence_sha256'] == evidence_sha and bound['source_sha256'] == sources and
         bound['remote_version'] == remote_version, 'exact independent candidate/phase/evidence/source identity')
    need(type(bound['checks']) is dict and type(bound['auxiliary_reviews']) is dict, 'review check/reference maps required')
    inputs.pins(sources)
    return bound


def identity(value, kernel, phase):
    need(value.get('schema') == 'h198-result-seal/v2' and value.get('candidate') == b.KEY and
         value.get('slug') == b.SLUG and value.get('version') == b.VERSION and
         value.get('kernel_sha256') == kernel and value.get('phase') == phase and
         value.get('formula') == 'H189_F_Q4' and value.get('fast') is False, 'exact sealed candidate identity')


def base(inputs, evidence_ref, fixture_frozen=None):
    evidence = inputs.json(evidence_ref)
    need(isinstance(evidence.get('artifacts'), dict) and bool(evidence['artifacts']), 'complete post-result artifact map required')
    for item in evidence['artifacts'].values():
        inputs.file(item)
    # The production anchor is fixed in reviewed code, never supplied by a sidecar.
    anchor = {'path': FROZEN, 'sha256': FROZEN_SHA} if fixture_frozen is None else fixture_frozen
    need(evidence.get('frozen') == anchor, 'fixed original equivalence anchor required')
    frozen = inputs.json(anchor)
    need(frozen.get('schema') == 'h198-frozen-equivalence/v1' and frozen.get('candidate') == b.KEY and
         set(frozen.get('roles', {})) == ROLES and frozen.get('prospective_bindings') == 334 and
         frozen.get('fixture_only') is (fixture_frozen is not None), 'frozen equivalence schema/scope')
    inputs.pins(frozen['bindings'])
    roles = frozen['roles']
    for item in roles.values():
        need(frozen['bindings'].get(item['path']) == item['sha256'], 'mandatory role missing from frozen closure')
        inputs.file(item)
    kernel = roles['kernel']['sha256']
    need(kernel == frozen['kernel_sha256'], 'kernel anchor binding')
    sources = evidence['sources']
    need(isinstance(sources, dict) and set(SOURCE_PATHS) <= set(sources), 'mandatory current source closure')
    inputs.pins(sources)
    if fixture_frozen is None:
        need(sources['tools/strict_variant_gate_c4_v102.py'] == sha(__file__), 'loaded gate source identity')
    records = {key: inputs.json(item) for key, item in roles.items() if key not in ('kernel', 'metadata')}
    pins = records['pins']
    need(len(pins['bindings']) == 334 and all(frozen['bindings'].get(p.replace('\\', '/')) == h
         for field in ('bindings', 'ruler_bindings') for p, h in pins[field].items()), 'complete prospective roster/ruler closure')
    parent = records['parent_gate']
    need(parent['status'] == 'PASS' and parent['fixture_only'] is False and not parent['errors'] and
         all(frozen['bindings'].get(p.replace('\\', '/')) == h for p, h in parent['input_sha256'].items()), 'unchanged authentic parent admission closure')
    build = records['build']
    need(build['candidate'] == b.KEY and build['kernel_id'] == b.SLUG and build['version'] == b.VERSION and
         build['kernel_sha256'] == kernel and build['metadata_sha256'] == roles['metadata']['sha256'] and
         build['parent_receipt_sha256'] == roles['parent_gate']['sha256'] and
         build['builder'] == b.BUILDER and build['unchanged'] == 22 and len(build['all_embedded']) == 31,
         'fixed scientific construction identity')
    p, st, o, a, replay = (records[k] for k in ('parity', 'strict', 'oof', 'assets', 'parent_replay'))
    need(all(x['kernel_sha256'] == kernel for x in (p, st, a, replay)), 'original result kernel binding')
    need(p['status'] == 'PASS_CPU_REAL_RENDERER' and len(p['cases']) == 5 and
         all(x['exact'] is True and x['windows'] == 36 for x in p['cases']), 'original renderer cases')
    need(st['status'] == 'PASS_CPU_STRICT_NO_FORWARD' and len(st['models']) == 5 and
         all(x['strict'] is True for x in st['models']), 'original strict checkpoint proof')
    need(a['status'] == 'TRAINING_POLICY_BOUND' and a['c4_archive_sha256'] == b.ARCHIVE_SHA and
         a['source_model_hashes'] == [x['sha256'] for x in st['models']], 'original asset/training proof')
    need(o['input_pins_sha256'] == roles['pins']['sha256'] and o['clean_oof'] is False and
         o['exposure'] == 'EXPOSED' and o['primary']['gate_pass'] is True and
         o['historical']['registered_gate']['status'] == 'ADVANCE', 'unchanged Q32/EXP183 result')
    need(replay['status'] == 'PASS_PARENT_EXACT_NOOP' and replay['noop_float64_and_csv_exact'] is True and
         replay['exact_F_reference'] is True and replay['exact_Q_cache'] is True and replay['members'] == 20 and
         replay['studies'] == 58 and replay['card_sha256'] == roles['legacy_card']['sha256'], 'exact parent replay/card')
    legacy = records['legacy_tests']
    need(legacy['exit_code'] == 0 and legacy['failed'] == 0 and legacy['passed'] > 0 and
         legacy['kernel_sha256'] == kernel, 'preserved scoped legacy tests')
    tests = inputs.json(evidence['artifacts']['tests'])
    need(tests.get('schema') == 'h198-fix1-tests/v1' and tests.get('actual_run') is True and
         tests.get('exit_code') == 0 and tests.get('failed') == 0 and type(tests.get('passed')) is int and
         tests['passed'] > 0 and tests.get('kernel_sha256') == kernel and tests.get('source_sha256') == sources,
         'current actual CPU test receipt')
    log = inputs.file(tests['log']).read_text(encoding='utf-8')
    need(re.search(r'\b' + str(tests['passed']) + r' passed\b', log) is not None, 'actual test summary mismatch')
    return evidence, frozen, records, kernel, sources


def launch(inputs, e, frozen, kernel, sources):
    refs = e['artifacts']
    card = inputs.json(refs['launch_card'])
    original = inputs.json(frozen['roles']['legacy_card'])
    # This is a new integration artifact, not a rewrite of the old science card.
    need(card.get('schema') == 'h198-cache-integration/v2' and card.get('candidate') == b.KEY and
         card.get('kernel_sha256') == kernel and card.get('legacy_card') == frozen['roles']['legacy_card'] and
         card.get('gate_sha256') == sources['tools/strict_variant_gate_c4_v102.py'] and
         card.get('gate_module') == 'tools.strict_variant_gate_c4_v102' and card.get('gate_phase') == 'CACHE_LAUNCH_PASS' and
         card.get('late_recheck') is True and card.get('science_unchanged') is True, 'reviewed versioned cache integration required')
    worker_ref = card['worker']; inputs.file(worker_ref)
    need(sources.get(worker_ref['path']) == worker_ref['sha256'] and
         worker_ref['path'] != 'tools/hb198_c4_gold58_v1.py', 'new worker source in independent review closure')
    queue = inputs.json(card['queue'])
    need(len(queue) == 1 and queue[0]['tag'] == original['tag'] and queue[0]['continue_on_failure'] is False and
         len(queue[0]['command']) == 4 and queue[0]['command'][1:] == ['-B', str(inputs.root / worker_ref['path']), 'run'], 'exact new worker queue')
    worker_source = inputs.file(worker_ref).read_text(encoding='utf-8')
    need('strict_variant_gate_c4_v102' in worker_source and 'verify_receipt' in worker_source, 'new worker must consume repaired admission')
    claim = inputs.json(refs['resource_claim'])
    need(claim.get('state') == 'CLAIMED' and claim.get('owner') == 'B' and claim.get('tag') == original['tag'] and
         claim.get('physical_gpu') == 0 and claim.get('logical_device') == 'cuda:0', 'actual B resource claim')
    mem = inputs.json(refs['memory'])
    need(mem.get('status') == 'PASS_MEASURED' and mem.get('kernel_sha256') == kernel and
         mem.get('renderer_sha256') == original['renderer_sha256'] and mem.get('K') == 36 and
         mem.get('tta') == 'plane-aware-coronal-axial-flip' and mem.get('model_sha256') == [x['sha256'] for x in original['models']] and
         mem.get('physical_gpu') == 0 and '4070 Ti SUPER' in mem.get('device_name', '') and
         mem.get('actual_full_shape_both_views') is True, 'actual measured full-shape memory')
    peaks = mem['peak_reserved_bytes']
    need(len(peaks) == 5 and all(finite(x) and 0 < x <= 12 * 2**30 for x in peaks) and
         finite(mem['desktop_bytes']) and mem['desktop_bytes'] >= 0 and finite(mem['total_bytes']) and
         mem['desktop_bytes'] + max(peaks) + 2 * 2**30 <= mem['total_bytes'], 'five measured 12GiB/2GiB memory gates')
    need(len(mem['raw_receipts']) == 5, 'five raw model memory receipts')
    for item in mem['raw_receipts']:
        inputs.file(item)
    return card, original, mem, claim


def gold(inputs, e, card, original, kernel):
    import numpy as np
    refs = e['artifacts']; worker = inputs.json(refs['gold_worker']); result = inputs.json(refs['gold_bridge'])
    monitor = inputs.json(refs['gold_monitor'])
    need(worker.get('state') == 'INFERENCE_COMPLETE' and worker.get('failures') == 0 and
         worker.get('shape') == [5, 58, 12] and worker.get('dtype') == 'float32' and
         worker.get('card_sha256') == refs['launch_card']['sha256'] and
         worker.get('source_sha256') == card['worker']['sha256'] and worker.get('npz_sha256') == refs['gold_npz']['sha256'], 'complete current cache worker')
    need(monitor.get('status') == 'PASS' and monitor.get('worker_sha256') == refs['gold_worker']['sha256'] and
         monitor.get('child_exit_code') == 0 and monitor.get('owned_cleanup_complete') is True, 'actual completed controller/cleanup evidence')
    need(result.get('status') == 'COMPLETE_EXPOSED_DESCRIPTIVE_BRIDGE' and result.get('kernel_sha256') == kernel and
         result.get('worker_sha256') == refs['gold_worker']['sha256'] and result.get('card_sha256') == refs['launch_card']['sha256'] and
         result.get('parent_noop_exact') is True and result.get('clean_oof') is False and result.get('exposure') == 'EXPOSED', 'actual exposed Gold bridge')
    with np.load(inputs.file(refs['gold_npz']), allow_pickle=False) as data:
        need(set(data.files) == {'ids', 'targets', 'folds', 'values'} and data['ids'].tolist() == original['corpus']['ids'] and
             data['targets'].tolist() == original['corpus']['targets'] and data['folds'].tolist() == list(range(5)), 'Gold exact roster/target/fold order')
        values = data['values']
        need(values.shape == (5, 58, 12) and values.dtype == np.float32 and np.isfinite(values).all() and
             ((values >= 0) & (values <= 1)).all(), 'Gold all finite FP32 probabilities')


def execution(inputs, item, kind, kernel, metadata, targets, remote_version):
    import numpy as np
    value = inputs.json(item); n = {'FULL_VISIBLE': 3, 'N32': 32}[kind]
    need(value.get('schema') == 'h198-qualified-execution/v1' and value.get('state') == 'PASS' and
         value.get('kind') == kind and value.get('studies') == n and value.get('slug') == b.SLUG and
         value.get('kernel_sha256') == kernel and value.get('metadata_sha256') == metadata and
         value.get('formula') == 'H189_F_Q4' and value.get('fast') is False and value.get('archive_sha256') == b.ARCHIVE_SHA and
         value.get('remote_version') == remote_version, 'exact complete visible/N32 identity')
    required = {'own', 'q', 'raptor', 'r224', 'r6'}
    need(set(value.get('child_logs', {})) == set(value.get('child_exit_codes', {})) == required and
         all(type(x) is int and x == 0 for x in value['child_exit_codes'].values()) and
         all(value.get(k) is True for k in ('cuda_all_entrypoints', 'linux_actual', 'codecs_pass', 'fallbacks_pass')), 'all actual CUDA entrypoints/logs')
    for item in value['child_logs'].values():
        inputs.file(item)
    ids = inputs.json(value['ids'])
    with inputs.file(value['output_csv']).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); rows = list(reader)
    need(len(ids) == len(set(ids)) == n and reader.fieldnames == ['StudyInstanceUID', *targets] and
         [row['StudyInstanceUID'] for row in rows] == ids, 'all IDs x12 output order')
    values = np.asarray([[float(row[t]) for t in targets] for row in rows])
    need(values.shape == (n, 12) and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'finite complete output')


def runtime(inputs, item, n32_ref):
    import numpy as np
    value = inputs.json(item); raw = value['per_study_seconds']
    need(set(raw) == set(old.REQUIRED_TERMS) and all(isinstance(x, list) and len(x) == 32 and
         all(finite(y) and y > 0 for y in x) for x in raw.values()), 'actual complete N32 timing arrays')
    need(value.get('execution_sha256') == n32_ref['sha256'] and all(value.get(k) is True for k in
         ('full_path_measured', 'startup_fixed_proven', 'all_sequential_phase_costs_included', 'cohort_source_verified')) and
         value.get('cohort_studies') == 1300, 'full-path runtime/cohort identity')
    cohort = inputs.json(value['cohort_source'])
    need(cohort.get('hidden_studies') == 1300 and cohort.get('verified') is True, 'actual cohort source')
    need(bool(value['raw_timing_receipts']), 'raw timing evidence')
    for item in value['raw_timing_receipts']:
        inputs.file(item)
    rates = {k: float(np.percentile(v, 95)) for k, v in raw.items()}
    startup = value['fixed_startup_seconds']
    need(finite(startup) and startup >= 0, 'fixed startup measurement')
    projected = startup + 1300 * sum(rates.values())
    need(finite(value.get('projected_seconds')) and abs(value['projected_seconds'] - projected) < 1e-6 and projected <= 32400, 'measured hidden runtime <=9h')
    return dict(per_study_p95=rates, fixed_startup_seconds=startup, cohort_studies=1300,
                cohort_source_verified=True, full_path_measured=True, startup_fixed_proven=True, projected_seconds=projected)


def evaluate(phase, evidence, review_ref, *, root=ROOT, fixture_frozen=None):
    """No writes. Non-ROOT evaluation is explicitly synthetic, never admission."""
    need(phase in PHASES, 'unknown admission phase')
    root = Path(root).resolve(); fixture = root != ROOT.resolve()
    need((fixture and fixture_frozen is not None and root.is_relative_to(ROOT / '.local/scratch')) or
         (not fixture and fixture_frozen is None), 'fixture cannot run in production or outside fixed scratch')
    inputs = Inputs(root)
    e, frozen, records, kernel, sources = base(inputs, evidence, fixture_frozen)
    identity(e, kernel, phase)
    need(e.get('fixture_only') is fixture, 'explicit fixture mode mismatch')
    remote = e.get('remote_version')
    need((phase == 'SUBMISSION_PASS' and type(remote) is int and remote > 0) or
         (phase != 'SUBMISSION_PASS' and remote is None), 'phase remote-version contract')
    rv = review(review_ref, kernel, REVIEW_PHASE[phase], evidence['sha256'], sources, remote, inputs=inputs)
    facts = dict(candidate=b.KEY, slug=b.SLUG, version=b.VERSION, kernel_sha256=kernel, parent_kernel_sha256=b.PARENT_SHA,
        fast=False, exact_diff=True, retained15=True, mixed_renderer=True, formula='H189_F_Q4', builder=b.BUILDER,
        reviewer=rv['reviewer'], source_pins_current=True, review_current=True, renderer_real_pass=True,
        strict5_pass=True, cpu_regressions_pass=True, oof_q32_pass=True, inherited_numeric_pass=True)
    refs = e['artifacts']; aux = rv['auxiliary_reviews']
    if phase != 'BUILD_PASS':
        prior = inputs.json(refs['build_admission'])
        need(prior['phase'] == 'BUILD_PASS' and prior['fixture_only'] is fixture, 'prior BUILD phase/fixture identity')
        replay = evaluate('BUILD_PASS', prior['evidence'], prior['review'], root=root, fixture_frozen=fixture_frozen)
        need(replay == prior, 'stale prior BUILD admission')
        inputs.pins(prior['consumed_sha256'])
        card, original, mem, claim = launch(inputs, e, frozen, kernel, sources)
        facts.update(build_admitted=True, launch_review_current=True, workload=290, model_count=5, studies=58,
            corpus_authenticated=True, resource_claim=claim, executor='B', logical_device='cuda:0', guarded_worker=True,
            memory=dict(desktop_bytes=mem['desktop_bytes'], peak_reserved_bytes=max(mem['peak_reserved_bytes']), total_bytes=mem['total_bytes']))
    if phase in ('QUALIFICATION_LAUNCH_PASS', 'SUBMISSION_PASS'):
        gold(inputs, e, card, original, kernel)
        gr = review(aux['gold_result'], kernel, 'GOLD_RESULT', evidence['sha256'], sources, remote, inputs=inputs)
        need(gr['checks'].get('joint_harm_reviewed') is True, 'independent joint-harm review')
        intent = inputs.json(refs['candidate_intent']); assets = inputs.json(refs['remote_assets']); quota = inputs.json(refs['quota'])
        need(intent.get('candidate') == b.KEY and intent.get('decision') == 'QUALIFY_TO_SUBMIT' and intent.get('owner') == 'B', 'B candidate intent')
        need(assets.get('dataset') == b.DATASET and assets.get('archive_sha256') == b.ARCHIVE_SHA and assets.get('accessible') is True, 'current remote asset proof')
        need(quota.get('source') == 'https://www.kaggle.com/settings' and quota.get('read_immediately_before_decision') is True and
             quota.get('observed_at') == intent.get('quota_observed_at') and quota.get('remaining_seconds', 0) > 0, 'live quota/intent reading')
        facts.update(gold_state='INFERENCE_COMPLETE', gold_shape=[5, 58, 12], gold_dtype='float32', gold_finite=True,
            gold_parent_noop=True, gold_same_roster=True, gold_failures=0, gold_review_pass=True, joint_harm_reviewed=True,
            submission_intent=True, remote_assets_verified=True, live_quota_verified=True)
    if phase == 'SUBMISSION_PASS':
        for key, kind in [('full_visible', 'FULL_VISIBLE'), ('n32', 'N32')]:
            execution(inputs, refs[key], kind, kernel, frozen['roles']['metadata']['sha256'], original['corpus']['targets'], remote)
        timing = runtime(inputs, refs['runtime'], refs['n32'])
        minimum = review(aux['minimum'], kernel, 'SUBMISSION_MINIMUM', evidence['sha256'], sources, remote, inputs=inputs)
        need(all(minimum['checks'].get(k) is True for k in ('duplicate_clear', 'description_accurate', 'all_ids_12_finite', 'runtime_le_9h')), 'minimum exact-version checks')
        integration = inputs.json(refs['canonical_integration'])
        ir = review(aux['integration'], kernel, 'CANONICAL_INTEGRATION', evidence['sha256'], sources, remote, inputs=inputs)
        sender = refs['sender']; sender_path = inputs.file(sender)
        need(sender['path'] == 'tools/kaggle_submit_once.py' and sources.get(sender['path']) == sender['sha256'] and
             integration.get('status') == 'PASS' and integration.get('candidate') == b.KEY and
             integration.get('entrypoint_sha256') == sender['sha256'] and integration.get('gate_sha256') == sources[SOURCE_PATHS[0]] and
             integration.get('mandatory_for_nonfast') is True and integration.get('late_recheck') is True and
             ir['checks'].get('canonical_one_attempt_and_late_recheck') is True, 'current canonical integration')
        sender_text = sender_path.read_text(encoding='utf-8')
        need(all(t in sender_text for t in ('strict_variant_gate_c4_v102', 'SUBMISSION_PASS', 'verify_receipt', b.KEY)), 'canonical dispatch path missing')
        integration_tests = inputs.json(integration['tests'])
        need(integration_tests.get('status') == 'PASS' and integration_tests.get('canonical_sha256') == sender['sha256'] and
             integration_tests.get('gate_sha256') == sources[SOURCE_PATHS[0]] and integration_tests.get('late_mutation_rejected') is True,
             'actual canonical bypass/late mutation tests')
        inputs.file(integration_tests['log'])
        scan = inputs.json(refs['secret_scan'])
        need(scan.get('status') == 'PASS' and scan.get('version') == '1.0.2' and scan.get('kernel_sha256') == kernel, 'current broad secret scan')
        inputs.file(scan['log'])
        facts.update(remote_version=remote, runtime=timing, minimum_review_pass=True, full_visible=True, n32=True,
            cuda_all_entrypoints=True, child_logs_pass=True, codecs_pass=True, fallbacks_pass=True, all_ids_12_finite=True,
            duplicate_clear=True, description_accurate=True, canonical_submit_integration=True, secret_scan_v102_pass=True)
    result = old.core(phase, facts, fixture_only=fixture)
    result.update(schema='h198-c4-gate/v2', gate_version=VERSION, gate_sha256=sources[SOURCE_PATHS[0]],
        slug=b.SLUG, version=b.VERSION, metadata_sha256=frozen['roles']['metadata']['sha256'],
        evidence=evidence, review=review_ref, evidence_sha256=evidence['sha256'], source_sha256=sources,
        remote_version=remote, fixture_only=fixture, qualified_for_submission=phase == 'SUBMISSION_PASS' and not fixture)
    inputs.recheck()
    result['consumed_sha256'] = dict(sorted(inputs.consumed.items()))
    return result


def verify_receipt(receipt, *, phase='SUBMISSION_PASS', slug=None, remote_version=None):
    """Canonical API: replay and compare, then rehash complete closure at transport."""
    inputs = Inputs(); saved = inputs.json(receipt)
    need(saved.get('schema') == 'h198-c4-gate/v2' and saved.get('fixture_only') is False and
         saved.get('phase') == phase and saved.get('slug') == (slug or b.SLUG), 'production exact receipt required')
    if phase == 'SUBMISSION_PASS':
        need(type(remote_version) is int and saved.get('remote_version') == remote_version and
             saved.get('qualified_for_submission') is True, 'canonical exact version required')
    fresh = evaluate(phase, saved['evidence'], saved['review'])
    need(fresh == saved, 'admission receipt does not replay exactly')
    inputs.pins(saved['consumed_sha256']); inputs.recheck()
    return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=PHASES); parser.add_argument('--evidence', required=True)
    parser.add_argument('--review', required=True); parser.add_argument('--out')
    args = parser.parse_args()
    try:
        result = evaluate(args.phase, ref(args.evidence), ref(args.review))
        if args.out:
            b.write(native.local(args.out, ROOT), b.encoded(result))
        print(json.dumps(result, sort_keys=True)); return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'HOLD', 'reason': str(exc)})); return 2


if __name__ == '__main__':
    raise SystemExit(main())
