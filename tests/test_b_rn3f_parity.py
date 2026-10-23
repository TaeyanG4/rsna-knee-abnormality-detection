"""Synthetic CPU only: no real cohort, CUDA, model, remote or DICOM reads."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b_parity', ROOT / 'tools/b_rn3f_parity.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


@pytest.fixture(scope='module')
def frozen():
    return p.passive_sources(ROOT / '.local/b-rn3f-staging/20261004-v1/assets')


def test_synthetic_integer_center_and_analytical_reference(frozen):
    common, runtime = frozen
    import torch
    assert not torch.cuda.is_initialized()
    for name, vol, mask in p.synthetics():
        centers = common.select_centers(mask, 64, 24, {'slots': common.SLOTS64}, policy='within-slot-valid')
        assert centers == p.centers_reference(mask)
        if name in ['allzero', 'slot_boundary', 'sparse_isolated']: assert centers == [-1] * 24
        if name == 'sparse_valid': assert set(centers) == {3, 34}
        windows = common.windows_from(vol, centers, 224)
        normalized = runtime.normalize_input(windows, 'imagenet')
        row = p.input_case(name, vol, mask, centers, windows, normalized)
        assert row['pass'], row
        assert row['normalized_sha256'] == runtime.tensor_sha256(normalized)
        if centers == [-1] * 24:
            mean, std = p.constants()
            np.testing.assert_array_equal(normalized[0, :, 0, 0], (-mean / std)[0, :, 0, 0])
            assert np.all(normalized != 0)
    assert not torch.cuda.is_initialized()


def test_center_reference_random_masks_matches_passive_selection(frozen):
    common, _ = frozen; rng = np.random.default_rng(51)
    for _ in range(50):
        mask = rng.integers(0, 2, 64, dtype=np.uint8)
        assert p.centers_reference(mask) == common.select_centers(mask, 64, 24, {'slots': common.SLOTS64}, policy='within-slot-valid')


def test_analytical_reference_empty_and_constant():
    volume = np.full((64, 336, 336), 255, np.uint8)
    windows, normalized = p.analytical_reference(volume, [-1] * 24)
    assert not windows.any() and np.all(normalized < 0)
    windows, normalized = p.analytical_reference(volume, [1] * 24)
    assert np.all(windows == 1)
    with pytest.raises(ValueError): p.analytical_reference(volume, [0] * 24)
    with pytest.raises(ValueError): p.centers_reference(np.full(64, 2, np.uint8))


def test_error_bounds_fail_not_relaxed():
    ref = np.zeros((2, 3), np.float64)
    ok = np.zeros((2, 3), np.float32)
    assert p.errors(ok, ref, p.LIMITS['normalized'])['pass']
    ok[0, 0] = 1e-5
    assert not p.errors(ok, ref, p.LIMITS['normalized'])['pass']
    ok[0, 0] = np.nan
    with pytest.raises(ValueError): p.errors(ok, ref, p.LIMITS['normalized'])


def test_rank_independent_average_ties():
    x = np.asarray([[.2], [.1], [.2], [.9]], np.float32)
    np.testing.assert_array_equal(p.rank_pct(x)[:, 0], [.625, .25, .625, 1.])


def test_rank_bound_covers_near_tie_crossings_and_exact_constant():
    rng = np.random.default_rng(32)
    ref = rng.uniform(.4, .401, (100, 12)).astype(np.float32)
    actual = (ref + rng.uniform(-.00199, .00199, ref.shape)).astype(np.float32)
    delta = np.abs(p.rank_pct(actual) - p.rank_pct(ref))
    assert np.all(delta <= p.rank_bound(ref) + 1e-15)
    # Broad rank shifts can be numerically valid in a dense cluster; no ad hoc percentage cap.
    assert delta.max() > .3
    spaced = np.linspace(0, 1, 100, dtype=np.float32)[:, None]
    assert np.all(p.rank_bound(spaced) == 0)


def test_probability_fixed100_all72_targets_and_blend():
    rng = np.random.default_rng(6)
    ref = rng.uniform(.1, .9, (6, 100, 12)).astype(np.float32)
    actual = (ref + rng.uniform(-.0001, .0001, ref.shape)).astype(np.float32)
    report = p.probability_gate(actual, ref)
    assert report['pass'] and len(report['member_target']) == 72
    bad = ref.copy(); bad[4, 1, 7] += .0021
    assert not p.probability_gate(bad, ref)['pass']
    bad = ref.copy(); bad[3, :, 0] += .0006  # Mean fails although max/p99 pass.
    assert not p.probability_gate(bad, ref)['pass']


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'nan', 'range'])
def test_probability_rejects_contract_violation(bad):
    raw = np.full((6, 100, 12), .5, np.float32); ref = raw.copy()
    if bad == 'dtype': raw = raw.astype(np.float64)
    if bad == 'shape': raw = raw[:5]
    if bad == 'nan': raw[0, 0, 0] = np.nan
    if bad == 'range': raw[0, 0, 0] = 1.01
    with pytest.raises(ValueError): p.probability_gate(raw, ref)


def test_freeze_immutable_and_tamper_rejected(tmp_path):
    protocol = tmp_path / 'protocol.json'
    p.freeze(protocol); digest = p.sha(protocol); p.validate_protocol(protocol, digest)
    with pytest.raises(ValueError): p.freeze(protocol)
    changed = p.read(protocol); changed['limits']['normalized']['max_abs'] *= 10
    protocol.write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError): p.validate_protocol(protocol, digest)
    with pytest.raises(ValueError): p.validate_protocol(protocol, p.sha(protocol))


def input_fixture():
    ids = [str(i) for i in range(100)]
    def row(uid):
        return {'uid': uid, 'volume_sha256': '1' * 64, 'mask_sha256': '2' * 64,
                'selected_centers': [-1] * 24, 'empty_windows': 24, 'normalized_sha256': '3' * 64,
                'shape': [24, 3, 224, 224], 'dtype': 'float32', 'pass': True,
                **{name: {'pass': True, 'limits': p.LIMITS[name], 'max_abs': 0., 'p99_abs': 0., 'mean_abs': 0.}
                   for name in ['window', 'normalized']}}
    rows = [row(uid) for uid in ids]
    receipt = {'runtime': {'input_evidence': copy.deepcopy(rows)},
               'inference': {'input_tensor_hashes': [{'uid': uid, 'sha256': '3' * 64} for uid in ids]}}
    report = {'schema': p.SCHEMA, 'protocol_sha256': 'protocol', 'helper_sha256': p.sha(p.__file__),
              'serving_manifest_sha256': p.SERVING_SHA, 'target': 'local-reference', 'pass': True,
              'rows': rows, 'synthetic': [row(name) for name in p.SYNTHETIC_NAMES]}
    return report, receipt, ids


@pytest.mark.parametrize('bad', [None, 'uid', 'hash', 'centers', 'limit', 'partial', 'synthetic'])
def test_input_report_must_bind_actual_producer_tensors(bad):
    report, receipt, ids = input_fixture()
    if bad == 'uid': report['rows'].reverse()
    if bad == 'hash': report['rows'][0]['normalized_sha256'] = '4' * 64
    if bad == 'centers': report['rows'][0]['selected_centers'] = [1] * 24
    if bad == 'limit': report['rows'][0]['normalized']['max_abs'] = 1e-4
    if bad == 'partial': receipt['runtime']['input_evidence'].pop()
    if bad == 'synthetic': report['synthetic'].pop()
    if bad:
        with pytest.raises(ValueError): p.validate_input_report(report, receipt, ids, 'protocol', 'local-reference')
    else: p.validate_input_report(report, receipt, ids, 'protocol', 'local-reference')


def test_end_to_end_synthetic_producer_reports(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import shutil
    assets = tmp_path / 'assets'; assets.mkdir()
    stage = ROOT / '.local/b-rn3f-staging/20261004-v1/assets'
    shutil.copyfile(stage / 'b_rn3f_serving.py', assets / 'b_rn3f_serving.py')
    (assets / 'b-rn3f-assets.json').write_text('{}', encoding='utf-8')
    producer_spec = importlib.util.spec_from_file_location('synthetic_producer', assets / 'b_rn3f_serving.py')
    producer = importlib.util.module_from_spec(producer_spec); producer_spec.loader.exec_module(producer)
    input_report, input_receipt, ids = input_fixture()
    monkeypatch.setattr(p, 'SERVING_SHA', p.sha(assets / 'b-rn3f-assets.json'))
    monkeypatch.setattr(p, 'UID_SHA', producer.ids_sha(ids))
    (assets / 'engineering100.json').write_text(json.dumps({'ids': ids}), encoding='utf-8')
    protocol = tmp_path / 'protocol.json'; p.freeze(protocol); protocol_sha = p.sha(protocol)
    raw = np.random.default_rng(11).uniform(.1, .9, (6, 100, 12)).astype(np.float32)
    records = input_receipt['inference']['input_tensor_hashes']
    digest = hashlib.sha256(''.join(json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n' for r in records).encode()).hexdigest()
    inference = {'status': 'COMPLETE', 'all_finite': True, 'input_contract': producer.INPUT, 'studies': 100,
                 'member_count': 6, 'requested_amp': 'fp16', 'effective_amp': 'fp16', 'batch_size': 1,
                 'residency': 'resident', 'runtime': {'policy': 'strict'}, 'fp32_fallbacks': 0, 'neutral_fills': 0,
                 'failures': [], 'input_preparations': 100, 'completed_batches': 100,
                 'input_tensor_hashes': records, 'ordered_inputs_sha256': digest,
                 'member_results': [{'name': m['name'], 'checkpoint_sha256': m['sha256'], 'status': 'COMPLETE',
                                     'completed_studies': 100, 'ordered_inputs_sha256': digest,
                                     'probabilities_sha256': producer.tensor_sha256(raw[i])}
                                    for i, m in enumerate(producer.expected_members())]}
    paths = []
    for target in ['local-reference', 't4-engineering']:
        work = tmp_path / target; work.mkdir()
        producer.atomic_outputs(work, ids, raw, p.SERVING_SHA, inference,
                                {'target': target, 'cohort': 'engineering100', 'input_evidence': input_receipt['runtime']['input_evidence']})
        report = copy.deepcopy(input_report)
        report.update(target=target, protocol_sha256=protocol_sha, serving_manifest_sha256=p.SERVING_SHA)
        report_path = tmp_path / (target + '.json'); p.write_new(report_path, report)
        paths.append((work, report_path))
    args = SimpleNamespace(protocol=protocol, protocol_sha256=protocol_sha, assets=assets,
                           local_work=paths[0][0], local_input=paths[0][1], t4_work=paths[1][0], t4_input=paths[1][1],
                           out=tmp_path / 'paired.json')
    result = p.evaluate(args)
    assert result['status'] == 'PASS_NUMERICAL_ONLY' and result['submission_ready'] is False
    report = p.read(args.t4_input); report['rows'][0]['normalized_sha256'] = '4' * 64
    args.t4_input.write_text(json.dumps(report), encoding='utf-8'); args.out = tmp_path / 'mismatch.json'
    with pytest.raises(ValueError, match='producer tensors'): p.evaluate(args)
