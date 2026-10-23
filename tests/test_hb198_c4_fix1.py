"""C4 FIX1 full-schema CPU fixtures. Synthetic receipts are NEVER admissions.

No model forward, notebook harness, GPU, network or production receipt writes.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import strict_variant_gate_c4_v102 as g


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else g.b.encoded(value))
    return g.ref(path, root)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.counter = 0
        self.sources = {}
        for name in g.SOURCE_PATHS:
            self.sources[name] = write(root, name, (g.ROOT / name).read_bytes())['sha256']
        self.worker = write(root, 'tools/c4_fixture_worker.py', b'# SYNTHETIC NEVER EXECUTE\n# strict_variant_gate_c4_v102 verify_receipt\n')
        self.sender = write(root, 'tools/kaggle_submit_once.py', b'# SYNTHETIC NEVER EXECUTE\n# strict_variant_gate_c4_v102 verify_receipt SUBMISSION_PASS R6W958-C4-V1\n')
        self.sources[self.worker['path']] = self.worker['sha256']
        self.sources[self.sender['path']] = self.sender['sha256']
        production = json.loads((g.ROOT / g.FROZEN).read_text())
        roles = {k: json.loads((g.ROOT / v['path']).read_text()) for k, v in production['roles'].items()
                 if k not in ('kernel', 'metadata')}
        bindings = {}
        for index in range(334):
            item = write(root, f'synthetic/prospective/{index:03}.txt', f'SYNTHETIC INPUT {index}\n'.encode())
            bindings[item['path']] = item['sha256']
        roles['pins']['bindings'] = bindings.copy()
        roles['pins']['ruler_bindings'] = dict(list(bindings.items())[:7])
        roles['parent_gate']['input_sha256'] = dict(list(bindings.items())[:2])
        card = roles['legacy_card']
        card['corpus']['ids'] = [f'fixture-study-{i:02}' for i in range(58)]
        card['corpus']['path'] = 'synthetic/corpus'
        card['corpus']['metadata'] = {}
        card['corpus']['volumes'] = []
        card['sources'] = {}; card['parent_files'] = {}
        for i, model in enumerate(card['models']):
            model_ref = write(root, f'synthetic/model-{i}.txt', f'SYNTHETIC MODEL {i}'.encode())
            model['source'] = model_ref['path']; model['source_sha256'] = model['sha256'] = model_ref['sha256']
            bindings[model_ref['path']] = model_ref['sha256']
            roles['strict']['models'][i]['sha256'] = model_ref['sha256']
        roles['assets']['source_model_hashes'] = [m['sha256'] for m in card['models']]
        self.kernel = production['roles']['kernel']['sha256']
        role_refs = {}
        for key in ('kernel', 'metadata'):
            role_refs[key] = write(root, 'synthetic/' + key, (g.ROOT / production['roles'][key]['path']).read_bytes())
        role_refs['pins'] = write(root, 'synthetic/pins.json', roles['pins'])
        role_refs['parent_gate'] = write(root, 'synthetic/parent_gate.json', roles['parent_gate'])
        role_refs['legacy_card'] = write(root, 'synthetic/legacy_card.json', card)
        roles['oof']['input_pins_sha256'] = role_refs['pins']['sha256']
        roles['parent_replay']['card_sha256'] = role_refs['legacy_card']['sha256']
        roles['build']['parent_receipt_sha256'] = role_refs['parent_gate']['sha256']
        for key, value in roles.items():
            if key not in role_refs:
                role_refs[key] = write(root, 'synthetic/' + key + '.json', value)
        for item in role_refs.values():
            bindings[item['path']] = item['sha256']
        frozen = dict(schema='h198-frozen-equivalence/v1', candidate=g.b.KEY, kernel_sha256=self.kernel,
                      roles=role_refs, bindings=bindings, prospective_bindings=334, fixture_only=True)
        self.frozen = write(root, 'synthetic/FROZEN.json', frozen)
        self.roles = role_refs
        self.original = card
        log = write(root, 'synthetic/tests.log', b'100 passed in 0.01s -- SYNTHETIC PARSER FIXTURE, NOT ACTUAL EVIDENCE\n')
        tests = dict(schema='h198-fix1-tests/v1', actual_run=True, exit_code=0, failed=0, passed=100,
                     kernel_sha256=self.kernel, source_sha256=self.sources, log=log)
        self.refs = {'tests': write(root, 'synthetic/tests.json', tests)}
        self.evidences = {}; self.reviews = {}; self.receipts = {}
        self.build('BUILD_PASS')
        self.prepare_launch()
        self.build('CACHE_LAUNCH_PASS')
        self.prepare_gold()
        self.build('QUALIFICATION_LAUNCH_PASS')
        self.prepare_submission()
        self.build('SUBMISSION_PASS')

    def review(self, evidence, phase, remote=None, checks=None, aux=None, change=None,
               verdict_line='VERDICT: PASS', duplicate=False, extra_verdict=None):
        self.counter += 1
        name = f'synthetic/reviews/{self.counter}'
        bound = dict(schema='h198-independent-review/v2', candidate=g.b.KEY, slug=g.b.SLUG, version=g.b.VERSION,
            kernel_sha256=self.kernel, builder=g.b.BUILDER, reviewer='/root/fixture_reviewer', phase=phase,
            verdict='PASS', evidence_sha256=evidence['sha256'], source_sha256=self.sources,
            remote_version=remote, checks=checks or {}, auxiliary_reviews=aux or {})
        if change:
            bound.update(change)
        line = g.PREFIX + json.dumps(bound, sort_keys=True)
        text = '# SYNTHETIC ARCHIVE FIXTURE. NO REAL TASK COMPLETION.\n' + line + '\n'
        if duplicate:
            text += line + '\n'
        if extra_verdict:
            text += extra_verdict + '\n'
        text += verdict_line + '\n'
        report = write(self.root, name + '.md', text.encode())
        prompt = write(self.root, name + '.prompt.txt', b'SYNTHETIC PARSER FIXTURE ONLY; never real platform evidence\n')
        completion = dict(schema=g.native.SCHEMA, host=g.native.HOST, model=g.native.MODEL,
            agent_id=bound['reviewer'], lane='native-astra-r1', prompt=str(self.root / prompt['path']),
            report=str(self.root / report['path']), terminal_status='completed', observed_at='2026-10-18T00:00:00+00:00',
            prompt_sha256=prompt['sha256'], report_sha256=report['sha256'],
            completion_message='SYNTHETIC TEST FIXTURE. No native task or production review occurred.')
        c = write(self.root, name + '.completion.json', completion)
        receipt = g.native.record(self.root / c['path'], 'SYNTHETIC TEST ARCHIVE ONLY, NOT B COMPLETION ATTESTATION', self.root)
        wrapper = dict(bound, report=report, native_receipt=g.ref(receipt, self.root))
        return write(self.root, name + '.wrapper.json', wrapper)

    def build(self, phase):
        remote = 7 if phase == 'SUBMISSION_PASS' else None
        evidence = dict(schema='h198-result-seal/v2', candidate=g.b.KEY, slug=g.b.SLUG, version=g.b.VERSION,
            kernel_sha256=self.kernel, formula='H189_F_Q4', fast=False, phase=phase, fixture_only=True,
            remote_version=remote, frozen=self.frozen, sources=self.sources, artifacts=copy.deepcopy(self.refs))
        item = write(self.root, f'synthetic/{phase}.evidence.json', evidence)
        aux = {}
        if phase in ('QUALIFICATION_LAUNCH_PASS', 'SUBMISSION_PASS'):
            aux['gold_result'] = self.review(item, 'GOLD_RESULT', remote, {'joint_harm_reviewed': True})
        if phase == 'SUBMISSION_PASS':
            aux['minimum'] = self.review(item, 'SUBMISSION_MINIMUM', remote,
                dict(duplicate_clear=True, description_accurate=True, all_ids_12_finite=True, runtime_le_9h=True))
            aux['integration'] = self.review(item, 'CANONICAL_INTEGRATION', remote,
                dict(canonical_one_attempt_and_late_recheck=True))
        review = self.review(item, g.REVIEW_PHASE[phase], remote, aux=aux)
        result = g.evaluate(phase, item, review, root=self.root, fixture_frozen=self.frozen)
        self.evidences[phase] = item; self.reviews[phase] = review
        self.receipts[phase] = write(self.root, f'synthetic/{phase}.receipt.json', result)
        if phase == 'BUILD_PASS':
            self.refs['build_admission'] = self.receipts[phase]

    def prepare_launch(self):
        queue = write(self.root, 'synthetic/queue.json', [dict(tag=self.original['tag'],
            command=['python', '-B', str(self.root / self.worker['path']), 'run'], continue_on_failure=False)])
        card = dict(schema='h198-cache-integration/v2', candidate=g.b.KEY, kernel_sha256=self.kernel,
            legacy_card=self.roles['legacy_card'], gate_sha256=self.sources[g.SOURCE_PATHS[0]],
            gate_module='tools.strict_variant_gate_c4_v102', gate_phase='CACHE_LAUNCH_PASS',
            late_recheck=True, science_unchanged=True, worker=self.worker, queue=queue)
        self.refs['launch_card'] = write(self.root, 'synthetic/card.json', card)
        self.refs['resource_claim'] = write(self.root, 'synthetic/claim.json', dict(state='CLAIMED', owner='B',
            tag=self.original['tag'], physical_gpu=0, logical_device='cuda:0'))
        raw = [write(self.root, f'synthetic/memory-{i}.log', b'SYNTHETIC MEMORY ONLY\n') for i in range(5)]
        mem = dict(status='PASS_MEASURED', kernel_sha256=self.kernel, renderer_sha256=self.original['renderer_sha256'],
            K=36, tta='plane-aware-coronal-axial-flip', model_sha256=[x['sha256'] for x in self.original['models']],
            physical_gpu=0, device_name='SYNTHETIC 4070 Ti SUPER', actual_full_shape_both_views=True,
            peak_reserved_bytes=[8 * 2**30] * 5, desktop_bytes=2**30, total_bytes=16 * 2**30, raw_receipts=raw)
        self.refs['memory'] = write(self.root, 'synthetic/memory.json', mem)

    def prepare_gold(self):
        path = self.root / 'synthetic/gold.npz'
        np.savez(path, ids=np.asarray(self.original['corpus']['ids']), targets=np.asarray(self.original['corpus']['targets']),
                 folds=np.arange(5), values=np.full((5, 58, 12), .5, dtype=np.float32))
        self.refs['gold_npz'] = g.ref(path, self.root)
        worker = dict(state='INFERENCE_COMPLETE', failures=0, shape=[5, 58, 12], dtype='float32',
            card_sha256=self.refs['launch_card']['sha256'], source_sha256=self.worker['sha256'], npz_sha256=self.refs['gold_npz']['sha256'])
        self.refs['gold_worker'] = write(self.root, 'synthetic/worker.json', worker)
        self.refs['gold_monitor'] = write(self.root, 'synthetic/monitor.json', dict(status='PASS',
            worker_sha256=self.refs['gold_worker']['sha256'], child_exit_code=0, owned_cleanup_complete=True))
        result = dict(status='COMPLETE_EXPOSED_DESCRIPTIVE_BRIDGE', kernel_sha256=self.kernel,
            worker_sha256=self.refs['gold_worker']['sha256'], card_sha256=self.refs['launch_card']['sha256'],
            parent_noop_exact=True, clean_oof=False, exposure='EXPOSED')
        self.refs['gold_bridge'] = write(self.root, 'synthetic/gold.json', result)
        self.refs['candidate_intent'] = write(self.root, 'synthetic/intent.json', dict(candidate=g.b.KEY,
            decision='QUALIFY_TO_SUBMIT', owner='B', quota_observed_at='2026-10-18T00:00:00Z'))
        self.refs['remote_assets'] = write(self.root, 'synthetic/remote-assets.json', dict(dataset=g.b.DATASET,
            archive_sha256=g.b.ARCHIVE_SHA, accessible=True))
        self.refs['quota'] = write(self.root, 'synthetic/quota.json', dict(source='https://www.kaggle.com/settings',
            read_immediately_before_decision=True, observed_at='2026-10-18T00:00:00Z', remaining_seconds=3600))

    def prepare_submission(self):
        for key, kind, n in [('full_visible', 'FULL_VISIBLE', 3), ('n32', 'N32', 32)]:
            names = {'own', 'q', 'raptor', 'r224', 'r6'}
            logs = {name: write(self.root, f'synthetic/{key}-{name}.log', b'SYNTHETIC CHILD EXIT 0\n') for name in names}
            ids = [f'fixture-visible-{i:02}' for i in range(n)]
            ids_ref = write(self.root, f'synthetic/{key}-ids.json', ids)
            csv = 'StudyInstanceUID,' + ','.join(self.original['corpus']['targets']) + '\n'
            csv += ''.join(uid + ',' + ','.join(['0.5'] * 12) + '\n' for uid in ids)
            csv_ref = write(self.root, f'synthetic/{key}.csv', csv.encode())
            value = dict(schema='h198-qualified-execution/v1', state='PASS', kind=kind, studies=n,
                slug=g.b.SLUG, kernel_sha256=self.kernel, metadata_sha256=self.roles['metadata']['sha256'],
                formula='H189_F_Q4', fast=False, archive_sha256=g.b.ARCHIVE_SHA, remote_version=7,
                child_logs=logs, child_exit_codes={name: 0 for name in names}, cuda_all_entrypoints=True,
                linux_actual=True, codecs_pass=True, fallbacks_pass=True, ids=ids_ref, output_csv=csv_ref)
            self.refs[key] = write(self.root, f'synthetic/{key}.json', value)
        cohort = write(self.root, 'synthetic/cohort.json', dict(hidden_studies=1300, verified=True))
        timing = dict(per_study_seconds={k: [1.] * 32 for k in g.old.REQUIRED_TERMS},
            execution_sha256=self.refs['n32']['sha256'], full_path_measured=True, startup_fixed_proven=True,
            all_sequential_phase_costs_included=True, cohort_source_verified=True, cohort_studies=1300,
            cohort_source=cohort, raw_timing_receipts=[write(self.root, 'synthetic/timing.log', b'SYNTHETIC TIMING\n')],
            fixed_startup_seconds=10., projected_seconds=5210.)
        self.refs['runtime'] = write(self.root, 'synthetic/runtime.json', timing)
        ilog = write(self.root, 'synthetic/integration-tests.log', b'SYNTHETIC INTEGRATION TESTS\n')
        it = write(self.root, 'synthetic/integration-tests.json', dict(status='PASS', canonical_sha256=self.sender['sha256'],
            gate_sha256=self.sources[g.SOURCE_PATHS[0]], late_mutation_rejected=True, log=ilog))
        self.refs['canonical_integration'] = write(self.root, 'synthetic/integration.json', dict(status='PASS',
            candidate=g.b.KEY, entrypoint_sha256=self.sender['sha256'], gate_sha256=self.sources[g.SOURCE_PATHS[0]],
            mandatory_for_nonfast=True, late_recheck=True, tests=it))
        self.refs['sender'] = self.sender
        self.refs['secret_scan'] = write(self.root, 'synthetic/scan.json', dict(status='PASS', version='1.0.2',
            kernel_sha256=self.kernel, log=write(self.root, 'synthetic/scan.log', b'SYNTHETIC SCAN\n')))

    def evaluate(self, phase='SUBMISSION_PASS'):
        return g.evaluate(phase, self.evidences[phase], self.reviews[phase], root=self.root, fixture_frozen=self.frozen)


@pytest.fixture(scope='module')
def chain(tmp_path_factory):
    return Fixture(tmp_path_factory.mktemp('synthetic-c4-fix1'))


@pytest.mark.parametrize('phase', g.PHASES)
def test_complete_real_schema_evaluate_path_is_fixture_only(chain, phase):
    result = chain.evaluate(phase)
    assert result['status'] == phase and result['fixture_only'] is True
    assert result['qualified_for_submission'] is False
    assert len(result['consumed_sha256']) > 370


def test_no_review_self_inclusion_or_source_seal_cycle(chain):
    for phase in g.PHASES:
        evidence = json.loads((chain.root / chain.evidences[phase]['path']).read_text())
        assert 'review' not in evidence and 'auxiliary_reviews' not in evidence
        review = json.loads((chain.root / chain.reviews[phase]['path']).read_text())
        assert review['evidence_sha256'] == chain.evidences[phase]['sha256']
        for item in review['auxiliary_reviews'].values():
            assert json.loads((chain.root / item['path']).read_text())['evidence_sha256'] == chain.evidences[phase]['sha256']


@pytest.mark.parametrize('change', [
    {'candidate': 'K2'}, {'kernel_sha256': '0' * 64}, {'slug': 'wrong/slug'}, {'phase': 'DESIGN'},
    {'phase': 'GOLD_RESULT'}, {'evidence_sha256': '0' * 64}, {'source_sha256': {}},
    {'reviewer': g.b.BUILDER}, {'builder': '/root/wrong_builder'}, {'remote_version': 6}, {'verdict': 'HOLD'}])
def test_exact_attestation_rejects_wrong_scope_or_identity(chain, change):
    item = chain.review(chain.evidences['BUILD_PASS'], 'BUILD', change=change)
    with pytest.raises((ValueError, KeyError)):
        g.review(item, chain.kernel, 'BUILD', chain.evidences['BUILD_PASS']['sha256'], chain.sources, inputs=g.Inputs(chain.root))


@pytest.mark.parametrize('kwargs', [dict(verdict_line='VERDICT: FAIL'), dict(verdict_line='VERDICT: HOLD'),
    dict(duplicate=True), dict(extra_verdict='VERDICT: FAIL'), dict(extra_verdict='VERDICT: PASS'),
    dict(extra_verdict='  VERDICT: FAIL')])
def test_actual_report_verdict_and_attestation_ambiguity(chain, kwargs):
    item = chain.review(chain.evidences['BUILD_PASS'], 'BUILD', **kwargs)
    with pytest.raises(ValueError):
        g.review(item, chain.kernel, 'BUILD', chain.evidences['BUILD_PASS']['sha256'], chain.sources, inputs=g.Inputs(chain.root))


@pytest.mark.parametrize('field', sorted(g.ATTEST_FIELDS))
def test_missing_required_attestation_field(chain, field):
    value = json.loads((chain.root / chain.reviews['BUILD_PASS']['path']).read_text()); del value[field]
    item = write(chain.root, 'synthetic/missing-wrapper.json', value)
    with pytest.raises(ValueError):
        g.review(item, chain.kernel, 'BUILD', chain.evidences['BUILD_PASS']['sha256'], chain.sources, inputs=g.Inputs(chain.root))


def test_sidecar_cannot_add_scope(chain):
    value = json.loads((chain.root / chain.reviews['BUILD_PASS']['path']).read_text()); value['checks']['invented'] = True
    item = write(chain.root, 'synthetic/invented-wrapper.json', value)
    with pytest.raises(ValueError, match='report/sidecar mismatch'):
        g.review(item, chain.kernel, 'BUILD', chain.evidences['BUILD_PASS']['sha256'], chain.sources, inputs=g.Inputs(chain.root))


@pytest.mark.parametrize('report_name,phase', [
    ('.local/scratch/slot-b/k2-release/BUILD-REVIEW.md', 'BUILD'),
    ('.local/scratch/slot-b/k2-release/BUILD-REVIEW.md', 'GOLD_RESULT'),
    ('.local/scratch/slot-b/batch5-1018/C4-DESIGN-REVIEW.md', 'BUILD'),
    ('.local/scratch/slot-b/batch5-1018/C4-DESIGN-REVIEW.md', 'SUBMISSION')])
def test_genuine_native_fail_and_design_reports_cannot_become_pass(tmp_path, report_name, phase):
    report = g.ROOT / report_name; receipt = Path(str(report) + '.receipt.json')
    actual = g.native.verify(receipt)
    sources = {p: g.sha(g.ROOT / p) for p in g.SOURCE_PATHS}
    wrapper = dict(schema='h198-independent-review/v2', candidate=g.b.KEY, slug=g.b.SLUG, version=g.b.VERSION,
        kernel_sha256='d86ca0e93e11773ad6119bb9c84ea632130723d899a5f4ef5d5faa4029fc28d9',
        builder=g.b.BUILDER, reviewer=actual['agent_id'], phase=phase, verdict='PASS', evidence_sha256='0' * 64,
        source_sha256=sources, remote_version=None, checks={}, auxiliary_reviews={}, report=g.ref(report), native_receipt=g.ref(receipt))
    path = tmp_path / 'GENUINE-NEGATIVE-TEST-ONLY.json'; path.write_bytes(g.b.encoded(wrapper))
    with pytest.raises(ValueError, match='review must end|attestation'):
        g.review(g.ref(path), wrapper['kernel_sha256'], phase, '0' * 64, sources)


@pytest.mark.parametrize('which', ['prompt', 'report', 'completion', 'archive_report', 'archive_prompt', 'archive_receipt', 'archive_log'])
def test_full_native_chain_mutations(chain, which):
    wrapper = json.loads((chain.root / chain.reviews['BUILD_PASS']['path']).read_text())
    receipt = json.loads((chain.root / wrapper['native_receipt']['path']).read_text())
    completion = json.loads(Path(receipt['completion_source']).read_text())
    archive = chain.root / '.local/astra-improve/lanes/runs' / receipt['lane'] / receipt['run_id']
    paths = {'prompt': Path(completion['prompt']), 'report': Path(completion['report']), 'completion': Path(receipt['completion_source']),
             'archive_report': archive / 'report.md', 'archive_prompt': archive / 'prompt.txt',
             'archive_receipt': archive / 'receipt.json', 'archive_log': archive / 'run.log'}
    path = paths[which]; original = path.read_bytes()
    try:
        path.write_bytes(original + b' ')
        with pytest.raises(ValueError, match='stale evidence'):
            chain.evaluate('BUILD_PASS')
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize('role', sorted(g.ROLES))
def test_every_mandatory_frozen_result_late_mutation(chain, monkeypatch, role):
    path = chain.root / chain.roles[role]['path']; original = path.read_bytes()
    recheck = g.Inputs.recheck
    def mutate_then_recheck(inputs):
        path.write_bytes(original + b' ')
        return recheck(inputs)
    try:
        monkeypatch.setattr(g.Inputs, 'recheck', mutate_then_recheck)
        with pytest.raises(ValueError, match='stale evidence'):
            chain.evaluate('BUILD_PASS')
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize('source', g.SOURCE_PATHS)
def test_every_mandatory_source_late_mutation(chain, monkeypatch, source):
    path = chain.root / source; original = path.read_bytes(); recheck = g.Inputs.recheck
    def mutate_then_recheck(inputs):
        path.write_bytes(original + b'\n# drift\n')
        return recheck(inputs)
    try:
        monkeypatch.setattr(g.Inputs, 'recheck', mutate_then_recheck)
        with pytest.raises(ValueError, match='stale evidence'):
            chain.evaluate('BUILD_PASS')
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize('key', ['tests', 'build_admission', 'launch_card', 'resource_claim', 'memory', 'gold_npz',
    'gold_worker', 'gold_monitor', 'gold_bridge', 'candidate_intent', 'remote_assets', 'quota', 'full_visible',
    'n32', 'runtime', 'canonical_integration', 'sender', 'secret_scan'])
def test_every_later_phase_artifact_late_mutation(chain, monkeypatch, key):
    path = chain.root / chain.refs[key]['path']; original = path.read_bytes(); recheck = g.Inputs.recheck
    def mutate_then_recheck(inputs):
        if chain.refs['runtime']['path'] in inputs.consumed:
            path.write_bytes(original + b' ')
        return recheck(inputs)
    try:
        monkeypatch.setattr(g.Inputs, 'recheck', mutate_then_recheck)
        with pytest.raises(ValueError, match='stale evidence'):
            chain.evaluate()
    finally:
        path.write_bytes(original)


def test_all_consumed_original_roster_ruler_and_raw_artifacts_rechecked(chain):
    saved = chain.evaluate(); inputs = g.Inputs(chain.root); inputs.pins(saved['consumed_sha256'])
    for relative in saved['consumed_sha256']:
        path = chain.root / relative; original = path.read_bytes()
        try:
            path.write_bytes(original + b' ')
            with pytest.raises(ValueError, match='stale evidence'):
                inputs.file({'path': relative, 'sha256': saved['consumed_sha256'][relative]})
        finally:
            path.write_bytes(original)
    inputs.recheck()


def test_fixture_cannot_admit_production(chain):
    with pytest.raises(ValueError, match='fixture cannot'):
        g.evaluate('BUILD_PASS', chain.evidences['BUILD_PASS'], chain.reviews['BUILD_PASS'], fixture_frozen=chain.frozen)
    with pytest.raises(ValueError, match='production exact receipt'):
        g.verify_receipt(g.ref(chain.root / chain.receipts['SUBMISSION_PASS']['path']), remote_version=7)


@pytest.mark.parametrize('bad', ['../escape.txt', '/absolute.txt'])
def test_reference_path_escape(chain, bad):
    with pytest.raises(ValueError):
        g.Inputs(chain.root).file({'path': bad, 'sha256': '0' * 64})


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match='duplicate JSON key'):
        g.native.parse(b'{"verdict":"FAIL","verdict":"PASS"}')


def test_actual_production_anchor_complete_original_role_and_input_closure():
    inputs = g.Inputs()
    frozen = inputs.json({'path': g.FROZEN, 'sha256': g.FROZEN_SHA})
    assert set(frozen['roles']) == g.ROLES and frozen['fixture_only'] is False
    assert len(frozen['bindings']) == 594
    for item in frozen['roles'].values():
        assert frozen['bindings'][item['path']] == item['sha256']
    inputs.pins(frozen['bindings'])
    inputs.recheck()


@pytest.mark.parametrize('change', [dict(peak_reserved_bytes=[0] * 5), dict(peak_reserved_bytes=[13 * 2**30] * 5),
    dict(desktop_bytes=10 * 2**30), dict(actual_full_shape_both_views=False), dict(raw_receipts=[]),
    dict(model_sha256=['0' * 64] * 5), dict(physical_gpu=1)])
def test_full_launch_parser_rejects_invalid_memory_even_with_updated_file_hash(chain, change):
    evidence = json.loads((chain.root / chain.evidences['CACHE_LAUNCH_PASS']['path']).read_text())
    memory = json.loads((chain.root / chain.refs['memory']['path']).read_text()); memory.update(change)
    evidence['artifacts']['memory'] = write(chain.root, 'synthetic/invalid-memory.json', memory)
    frozen = json.loads((chain.root / chain.frozen['path']).read_text())
    with pytest.raises(ValueError):
        g.launch(g.Inputs(chain.root), evidence, frozen, chain.kernel, chain.sources)


@pytest.mark.parametrize('change', [dict(fixed_startup_seconds=-1), dict(projected_seconds=40000),
    dict(all_sequential_phase_costs_included=False), dict(raw_timing_receipts=[]), dict(execution_sha256='0' * 64)])
def test_runtime_parser_rejects_invalid_measurements_even_with_updated_hash(chain, change):
    value = json.loads((chain.root / chain.refs['runtime']['path']).read_text()); value.update(change)
    item = write(chain.root, 'synthetic/invalid-runtime.json', value)
    with pytest.raises(ValueError):
        g.runtime(g.Inputs(chain.root), item, chain.refs['n32'])


def test_original_old_worker_card_is_not_silently_accepted(chain):
    evidence = json.loads((chain.root / chain.evidences['CACHE_LAUNCH_PASS']['path']).read_text())
    evidence['artifacts']['launch_card'] = chain.roles['legacy_card']
    frozen = json.loads((chain.root / chain.frozen['path']).read_text())
    with pytest.raises(ValueError, match='versioned cache integration'):
        g.launch(g.Inputs(chain.root), evidence, frozen, chain.kernel, chain.sources)


def test_additional_reviewed_artifact_is_not_optional_hash_protection(chain):
    evidence = json.loads((chain.root / chain.evidences['BUILD_PASS']['path']).read_text())
    extra = write(chain.root, 'synthetic/extra-proof.txt', b'exact reviewed extra proof\n')
    evidence['artifacts']['extra_proof'] = extra
    item = write(chain.root, 'synthetic/extra-proof-evidence.json', evidence)
    review = chain.review(item, 'BUILD')
    result = g.evaluate('BUILD_PASS', item, review, root=chain.root, fixture_frozen=chain.frozen)
    assert result['consumed_sha256'][extra['path']] == extra['sha256']
    (chain.root / extra['path']).write_bytes(b'changed proof\n')
    with pytest.raises(ValueError, match='stale evidence'):
        g.evaluate('BUILD_PASS', item, review, root=chain.root, fixture_frozen=chain.frozen)
