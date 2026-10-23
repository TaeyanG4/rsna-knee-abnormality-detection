"""Complete TEST-ONLY INF3 evidence DAG; no model/notebook/GPU/real admission."""
import copy
import json
from pathlib import Path
import pytest
from tools import strict_variant_gate_inf3_v103 as g


def put(root, name, value):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else g.b.encoded(value))
    return g.ref(path, root)


class Chain:
    def __init__(self, root):
        self.root = root; self.count = 0
        self.sources = {p: put(root, p, (g.ROOT / p).read_bytes())['sha256'] for p in g.SOURCE_PATHS}
        self.sender = put(root, 'tools/kaggle_submit_once.py', b'# TEST ONLY CANONICAL SOURCE; NEVER EXECUTE\n')
        self.sources[self.sender['path']] = self.sender['sha256']
        production = json.loads((g.ROOT / g.FROZEN).read_text())
        records = {k: json.loads((g.ROOT / v['path']).read_text()) for k, v in production['roles'].items()
                   if k not in ('kernel', 'metadata')}
        bindings = {}
        for index in range(368):
            item = put(root, f'fixture/inputs/{index:03}.txt', f'TEST ONLY INPUT {index}\n'.encode())
            bindings[item['path']] = item['sha256']
        records['registration']['bindings'] = bindings.copy()
        records['parent_gate']['input_sha256'] = dict(list(bindings.items())[:2])
        roles = {}
        for key in ('kernel', 'metadata'):
            roles[key] = put(root, 'fixture/' + key, (g.ROOT / production['roles'][key]['path']).read_bytes())
        for key in ('registration', 'bridge_registration', 'bridge_result', 'parent_gate', 'legacy_tests'):
            roles[key] = put(root, 'fixture/' + key + '.json', records[key])
        for key in ('cpu', 'build'):
            records[key]['registration_sha256'] = roles['registration']['sha256']
            if key == 'cpu':
                records[key]['bridge_result_sha256'] = roles['bridge_result']['sha256']
            roles[key] = put(root, 'fixture/' + key + '.json', records[key])
        for item in roles.values():
            bindings[item['path']] = item['sha256']
        frozen = dict(schema='h197-fix1-equivalence/v1', candidate=g.b.KEY, fixture_only=True, roles=roles,
            original_source_sha256=production['original_source_sha256'], bindings=bindings, original_registration_bindings=368)
        self.frozen = put(root, 'fixture/FROZEN.json', frozen)
        self.roles = roles; self.kernel = roles['kernel']['sha256']; self.targets = records['bridge_registration']['targets']
        log = put(root, 'fixture/tests.log', b'100 passed in 0.01s -- TEST ONLY PARSER FIXTURE\n')
        tests = dict(schema='h197-fix1-tests/v1', actual_run=True, exit_code=0, failed=0, passed=100,
            kernel_sha256=self.kernel, source_sha256={p: self.sources[p] for p in g.SOURCE_PATHS}, log=log)
        self.artifacts = {'tests': put(root, 'fixture/tests.json', tests)}
        self.subjects = {}; self.envelopes = {}; self.receipts = {}
        self.phase('BUILD_PASS')
        self.prepare_execution()
        self.phase('QUALIFICATION_PASS')
        self.prepare_transport()
        self.phase('SUBMISSION_PASS')

    def review(self, subject, phase, remote, checks=None, change=None, report_end='VERDICT: PASS', duplicate=False, extra_verdict=None):
        self.count += 1; name = f'fixture/review-{self.count}'
        claim = dict(schema='h197-independent-review/v2', candidate=g.b.KEY, slug=g.b.SLUG, version=g.b.VERSION,
            kernel_sha256=self.kernel, phase=phase, builder=g.b.BUILDER, reviewer='/root/fixture_inf3_reviewer', verdict='PASS',
            evidence_sha256=subject['sha256'], checked_source_sha256=self.sources, remote_version=remote, checks=checks or {})
        if change:
            claim.update(change)
        line = 'H197-REVIEW-ATTESTATION: ' + json.dumps(claim, sort_keys=True)
        body = '# TEST ONLY native-shaped archive. No actual task completed.\n' + line + '\n'
        if duplicate:
            body += line + '\n'
        if extra_verdict:
            body += extra_verdict + '\n'
        report = put(self.root, name + '.md', (body + report_end + '\n').encode())
        prompt = put(self.root, name + '.prompt.txt', b'TEST ONLY synthetic native fixture, not actual completion\n')
        completion = dict(schema=g.native.SCHEMA, host=g.native.HOST, model=g.native.MODEL,
            agent_id=claim['reviewer'], lane='native-astra-r8', prompt=str(self.root / prompt['path']),
            report=str(self.root / report['path']), terminal_status='completed', observed_at='2026-10-18T00:00:00+09:00',
            prompt_sha256=prompt['sha256'], report_sha256=report['sha256'], completion_message='TEST ONLY; no real review or platform completion')
        completed = put(self.root, name + '.completion.json', completion)
        receipt = g.native.record(self.root / completed['path'], 'TEST ONLY fixture, not B attestation', self.root)
        return put(self.root, name + '.wrapper.json', dict(claim, report=report, native_receipt=g.ref(receipt, self.root)))

    def phase(self, phase):
        remote = None if phase == 'BUILD_PASS' else 9
        subject = dict(schema='h197-subject/v2', candidate=g.b.KEY, slug=g.b.SLUG, version=g.b.VERSION,
            kernel_sha256=self.kernel, phase=phase, formula='H189_F_Q3', fast=False, fixture_only=True,
            remote_version=remote, frozen=self.frozen, sources=self.sources, artifacts=copy.deepcopy(self.artifacts))
        e = put(self.root, 'fixture/' + phase + '.subject.json', subject)
        checks = {} if phase == 'BUILD_PASS' else dict(joint_adverse_gold_reviewed=True, all_new_path_gates_checked=True)
        envelope = dict(schema='h197-review-envelope/v1', subject=e, primary=self.review(e, phase, remote, checks))
        if phase == 'SUBMISSION_PASS':
            envelope['minimum'] = self.review(e, 'SUBMISSION_MINIMUM', remote, dict(duplicate_clear=True, description_accurate=True))
            envelope['integration'] = self.review(self.artifacts['canonical_integration'], 'CANONICAL_INTEGRATION', remote)
        v = put(self.root, 'fixture/' + phase + '.envelope.json', envelope)
        result = g.evaluate_fixture(phase, e, v, root=self.root, frozen=self.frozen)
        self.subjects[phase] = e; self.envelopes[phase] = v
        self.receipts[phase] = put(self.root, 'fixture/' + phase + '.receipt.json', result)

    def prepare_execution(self):
        for key, kind, n in [('full_visible', 'FULL_VISIBLE', 3), ('n32', 'N32', 32)]:
            ids = [f'TEST-ONLY-{i:03}' for i in range(n)]
            ids_ref = put(self.root, f'fixture/{key}-ids.json', ids)
            csv = 'StudyInstanceUID,' + ','.join(self.targets) + '\n'
            csv += ''.join(uid + ',' + ','.join(['0.5'] * 12) + '\n' for uid in ids)
            output = put(self.root, f'fixture/{key}.csv', csv.encode())
            logs = {role: put(self.root, f'fixture/{key}-{role}.log', b'TEST ONLY COMPLETE CHILD\n') for role in g.old.ENTRYPOINTS}
            execution = dict(schema='h197-qualified-execution/v1', state='PASS', kind=kind, candidate=g.b.KEY, slug=g.b.SLUG,
                version=g.b.VERSION, kernel_sha256=self.kernel, metadata_sha256=self.roles['metadata']['sha256'], fast=False,
                formula='H189_F_Q3', studies=n, remote_version=9, all_model_entrypoints=sorted(g.old.ENTRYPOINTS),
                cuda_all_entrypoints=True, linux_actual=True, codecs_pass=True, fallbacks_pass=True, all_20_cost_identities=True,
                child_logs=logs, child_exit_codes={k: 0 for k in logs}, ids=ids_ref, output_csv=output)
            self.artifacts[key] = put(self.root, f'fixture/{key}.json', execution)
        cohort = put(self.root, 'fixture/cohort.json', dict(hidden_studies=1300, verified=True))
        raw = put(self.root, 'fixture/timing.log', b'TEST ONLY TIMING\n')
        runtime = dict(schema='h197-runtime/v1', kernel_sha256=self.kernel, n32_sha256=self.artifacts['n32']['sha256'],
            hidden_studies=1300, cohort_source=cohort, per_study_seconds={k: [1.] * 32 for k in g.old.LINEAR_TERMS},
            global_seconds={k: 10. for k in g.old.GLOBAL_TERMS}, global_measured_studies=1300,
            all_sequential_phase_costs_included=True, all_inference_retries_tta_included=True, fixed_startup_seconds=30.,
            startup_fixed_proven=True, raw_receipts=[raw], projected_seconds=5250.)
        self.artifacts['runtime'] = put(self.root, 'fixture/runtime.json', runtime)
        memraw = put(self.root, 'fixture/memory.log', b'TEST ONLY MEMORY\n')
        memory = dict(schema='h197-memory/v1', kernel_sha256=self.kernel, actual_full_shape_tta=True,
            device_convention_verified=True, measurements=[dict(desktop_bytes=2 * 2**30, peak_reserved_bytes=10 * 2**30,
                total_bytes=16 * 2**30, raw_receipt=memraw)])
        self.artifacts['memory'] = put(self.root, 'fixture/memory.json', memory)

    def prepare_transport(self):
        source = {'tools/kaggle_submit_once.py': self.sender['sha256'],
                  'tools/strict_variant_gate_inf3_v103.py': self.sources['tools/strict_variant_gate_inf3_v103.py']}
        tests = dict(exit_code=0, failed=0, actual_run=True, source_sha256=source,
            log=put(self.root, 'fixture/transport-tests.log', b'TEST ONLY INTEGRATION TEST\n'),
            nonfast_bypass_rejected=True, wrong_candidate_rejected=True, wrong_phase_rejected=True,
            stale_source_rejected=True, late_recheck_verified=True, shared_sender_verified=True)
        integration = dict(schema='h197-canonical-integration/v1', status='PASS', candidate=g.b.KEY,
            canonical_sha256=self.sender['sha256'], gate_sha256=source['tools/strict_variant_gate_inf3_v103.py'],
            phase='SUBMISSION_PASS', mandatory_nonfast_gate=True, late_recheck=True, shared_sender=True,
            tests=put(self.root, 'fixture/transport-tests.json', tests))
        self.artifacts['canonical_integration'] = put(self.root, 'fixture/integration.json', integration)
        self.artifacts['sender'] = self.sender
        self.artifacts['secret_scan'] = put(self.root, 'fixture/scan.json', dict(version='1.0.2', status='PASS', kernel_sha256=self.kernel))

    def evaluate(self, phase='SUBMISSION_PASS', subject=None, envelope=None):
        return g.evaluate_fixture(phase, subject or self.subjects[phase], envelope or self.envelopes[phase], root=self.root, frozen=self.frozen)


@pytest.fixture(scope='module')
def chain(tmp_path_factory):
    return Chain(tmp_path_factory.mktemp('TEST-ONLY-inf3'))


@pytest.mark.parametrize('phase', g.PHASES)
def test_complete_production_schema_all_phases(chain, phase):
    result = chain.evaluate(phase)
    assert result['phase'] == phase and result['fixture_only'] is True and result['qualified_for_submission'] is False
    assert len(result['consumed_sha256']) > 390


def test_old_cycle_is_preserved_and_new_graph_is_acyclic(chain):
    source = Path(g.old.__file__).read_text()
    assert "minimum=review(e['minimum_review'],kernel,'SUBMISSION_MINIMUM',evidence_ref['sha256'])" in source
    minimum = put(chain.root, 'fixture/cycle-minimum.json', dict(evidence_sha256='0' * 64))
    evidence = put(chain.root, 'fixture/cycle-E.json', dict(minimum_review=minimum))
    new_minimum = put(chain.root, minimum['path'], dict(evidence_sha256=evidence['sha256']))
    with pytest.raises(ValueError, match='stale evidence'):
        g.Inputs(chain.root).file(minimum)
    second_evidence = put(chain.root, evidence['path'], dict(minimum_review=new_minimum))
    assert evidence['sha256'] != second_evidence['sha256']
    e = json.loads((chain.root / chain.subjects['SUBMISSION_PASS']['path']).read_text())
    v = json.loads((chain.root / chain.envelopes['SUBMISSION_PASS']['path']).read_text())
    assert 'minimum_review' not in e and not any('review' in k for k in e['artifacts'])
    assert json.loads((chain.root / v['minimum']['path']).read_text())['evidence_sha256'] == chain.subjects['SUBMISSION_PASS']['sha256']
    assert json.loads((chain.root / v['integration']['path']).read_text())['evidence_sha256'] == e['artifacts']['canonical_integration']['sha256']


@pytest.mark.parametrize('placement', ['top', 'artifacts'])
def test_old_cycle_layout_explicitly_refused(chain, placement):
    e = json.loads((chain.root / chain.subjects['SUBMISSION_PASS']['path']).read_text())
    target = e if placement == 'top' else e['artifacts']; target['minimum_review'] = chain.envelopes['SUBMISSION_PASS']
    item = put(chain.root, 'fixture/old-cycle-layout.json', e)
    with pytest.raises(ValueError, match='subject excludes reviews'):
        chain.evaluate(subject=item)


@pytest.mark.parametrize('field', sorted(g.ATTEST_FIELDS))
def test_missing_every_review_field(chain, field):
    v = json.loads((chain.root / chain.envelopes['BUILD_PASS']['path']).read_text())
    wrapper = json.loads((chain.root / v['primary']['path']).read_text()); del wrapper[field]
    item = put(chain.root, 'fixture/missing-review-field.json', wrapper)
    with pytest.raises(ValueError, match='complete review sidecar'):
        g.review(g.Inputs(chain.root), item, chain.kernel, 'BUILD_PASS', chain.subjects['BUILD_PASS']['sha256'], chain.sources, None)


@pytest.mark.parametrize('change', [dict(candidate='WRONG'), dict(slug='wrong/slug'), dict(version='wrong'),
    dict(kernel_sha256='0' * 64), dict(phase='SUBMISSION_PASS'), dict(evidence_sha256='0' * 64),
    dict(checked_source_sha256={}), dict(reviewer=g.b.BUILDER), dict(builder='/root/other'), dict(remote_version=99)])
def test_native_attestation_exact_scope_and_real_builder(chain, change):
    item = chain.review(chain.subjects['BUILD_PASS'], 'BUILD_PASS', None, change=change)
    with pytest.raises(ValueError):
        g.review(g.Inputs(chain.root), item, chain.kernel, 'BUILD_PASS', chain.subjects['BUILD_PASS']['sha256'], chain.sources, None)


@pytest.mark.parametrize('kwargs', [dict(report_end='VERDICT: FAIL'), dict(report_end='VERDICT: HOLD'),
    dict(duplicate=True), dict(extra_verdict='VERDICT: FAIL'), dict(extra_verdict='  VERDICT: FAIL')])
def test_report_verdict_and_attestation_boundary(chain, kwargs):
    item = chain.review(chain.subjects['BUILD_PASS'], 'BUILD_PASS', None, **kwargs)
    with pytest.raises(ValueError):
        g.review(g.Inputs(chain.root), item, chain.kernel, 'BUILD_PASS', chain.subjects['BUILD_PASS']['sha256'], chain.sources, None)


@pytest.mark.parametrize('name', ['.local/scratch/slot-b/k2-release/BUILD-REVIEW.md',
    '.local/scratch/slot-b/batch5-1018/C4-DESIGN-REVIEW.md',
    '.local/scratch/slot-b/batch5-1018/inf3-release/BUILD-REVIEW.md'])
def test_genuine_fail_design_and_original_fail_are_not_new_pass(tmp_path, name):
    report = g.ROOT / name; receipt = Path(str(report) + '.receipt.json'); actual = g.native.verify(receipt)
    sources = {p: g.sha(g.ROOT / p) for p in g.SOURCE_PATHS}
    wrapper = dict(schema='h197-independent-review/v2', candidate=g.b.KEY, slug=g.b.SLUG, version=g.b.VERSION,
        kernel_sha256='a' * 64, phase='BUILD_PASS', builder=g.b.BUILDER, reviewer=actual['agent_id'], verdict='PASS',
        evidence_sha256='b' * 64, checked_source_sha256=sources, remote_version=None, checks={}, report=g.ref(report), native_receipt=g.ref(receipt))
    path = tmp_path / 'GENUINE-NEGATIVE-ONLY.json'; path.write_bytes(g.b.encoded(wrapper))
    with pytest.raises(ValueError, match='review must end|attestation'):
        g.review(g.Inputs(), g.ref(path), 'a' * 64, 'BUILD_PASS', 'b' * 64, sources, None)


@pytest.mark.parametrize('which', ['prompt', 'report', 'completion', 'archive_prompt', 'archive_report', 'archive_completion', 'archive_receipt', 'archive_log'])
def test_each_native_archive_edge_changed(chain, which):
    v = json.loads((chain.root / chain.envelopes['BUILD_PASS']['path']).read_text())
    wrapper = json.loads((chain.root / v['primary']['path']).read_text())
    receipt = json.loads((chain.root / wrapper['native_receipt']['path']).read_text())
    c = json.loads(Path(receipt['completion_source']).read_text())
    archive = chain.root / '.local/astra-improve/lanes/runs' / receipt['lane'] / receipt['run_id']
    paths = dict(prompt=Path(c['prompt']), report=Path(c['report']), completion=Path(receipt['completion_source']),
        archive_prompt=archive / 'prompt.txt', archive_report=archive / 'report.md', archive_completion=archive / 'completion.json',
        archive_receipt=archive / 'receipt.json', archive_log=archive / 'run.log')
    p = paths[which]; data = p.read_bytes()
    try:
        p.write_bytes(data + b' ')
        with pytest.raises(ValueError, match='stale evidence'):
            chain.evaluate('BUILD_PASS')
    finally:
        p.write_bytes(data)


@pytest.mark.parametrize('name', ['primary', 'minimum', 'integration'])
def test_required_envelope_reviews_cannot_be_missing_or_replaced(chain, name):
    envelope = json.loads((chain.root / chain.envelopes['SUBMISSION_PASS']['path']).read_text())
    del envelope[name]
    with pytest.raises(ValueError, match='complete exact acyclic envelope'):
        chain.evaluate(envelope=put(chain.root, 'fixture/missing-envelope.json', envelope))


@pytest.mark.parametrize('key', ['full_visible', 'n32'])
def test_execution_version_cannot_disagree_with_review(chain, key):
    subject = json.loads((chain.root / chain.subjects['QUALIFICATION_PASS']['path']).read_text())
    value = json.loads((chain.root / chain.artifacts[key]['path']).read_text()); value['remote_version'] = 10
    subject['artifacts'][key] = put(chain.root, 'fixture/wrong-execution.json', value)
    e = put(chain.root, 'fixture/wrong-remote-E.json', subject)
    rv = chain.review(e, 'QUALIFICATION_PASS', 9, dict(joint_adverse_gold_reviewed=True, all_new_path_gates_checked=True))
    v = put(chain.root, 'fixture/wrong-remote-V.json', dict(schema='h197-review-envelope/v1', subject=e, primary=rv))
    with pytest.raises(ValueError, match='visible/N32 remote version'):
        chain.evaluate('QUALIFICATION_PASS', e, v)


@pytest.mark.parametrize('role', sorted(g.ROLES))
def test_every_frozen_scientific_result_late_mutation(chain, monkeypatch, role):
    late_mutation(chain, monkeypatch, chain.roles[role]['path'])


@pytest.mark.parametrize('source', g.SOURCE_PATHS)
def test_every_mandatory_source_late_mutation(chain, monkeypatch, source):
    late_mutation(chain, monkeypatch, source)


@pytest.mark.parametrize('artifact', ['tests', 'full_visible', 'n32', 'runtime', 'memory', 'canonical_integration', 'sender', 'secret_scan'])
def test_every_execution_transport_artifact_late_mutation(chain, monkeypatch, artifact):
    late_mutation(chain, monkeypatch, chain.artifacts[artifact]['path'])


def late_mutation(chain, monkeypatch, relative):
    p = chain.root / relative; data = p.read_bytes(); original = g.Inputs.recheck
    def injected(inputs):
        p.write_bytes(data + b' ')
        return original(inputs)
    try:
        monkeypatch.setattr(g.Inputs, 'recheck', injected)
        with pytest.raises(ValueError, match='stale evidence'):
            chain.evaluate()
    finally:
        p.write_bytes(data)


def test_all_returned_consumed_paths_include_and_reject_changed_raw_review_inputs(chain):
    value = chain.evaluate(); inputs = g.Inputs(chain.root); inputs.pins(value['consumed_sha256'])
    assert chain.sender['path'] in value['consumed_sha256']
    assert 'fixture/full_visible-q.log' in value['consumed_sha256'] and 'fixture/timing.log' in value['consumed_sha256']
    for relative, digest in value['consumed_sha256'].items():
        p = chain.root / relative; data = p.read_bytes()
        try:
            p.write_bytes(data + b' ')
            with pytest.raises(ValueError):
                inputs.file(dict(path=relative, sha256=digest))
        finally:
            p.write_bytes(data)
    inputs.recheck()


def test_fixture_has_no_production_or_cli_root_bypass(chain):
    with pytest.raises(ValueError, match='fixture cannot'):
        g.evaluate_fixture('BUILD_PASS', chain.subjects['BUILD_PASS'], chain.envelopes['BUILD_PASS'], root=g.ROOT, frozen=chain.frozen)
    with pytest.raises(TypeError):
        g.evaluate('BUILD_PASS', chain.subjects['BUILD_PASS'], chain.envelopes['BUILD_PASS'], root=chain.root)
    with pytest.raises(ValueError, match='nonfixture receipt'):
        g.verify_receipt(g.ref(chain.root / chain.receipts['SUBMISSION_PASS']['path']), remote_version=9)


def test_real_production_frozen_identity_complete():
    inputs = g.Inputs(); frozen = inputs.json(dict(path=g.FROZEN, sha256=g.FROZEN_SHA))
    assert len(frozen['bindings']) == 638 and frozen['fixture_only'] is False
    for item in frozen['roles'].values():
        assert frozen['bindings'][item['path']] == item['sha256']
    inputs.pins(frozen['bindings']); inputs.recheck()
