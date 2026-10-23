"""CPU synthetic checks for the unreleased exact-six engineering producer."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b_rn3f_serving', ROOT / 'tools/b_rn3f_serving.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def raw_example():
    values = np.asarray([[.1, .2, .3, .4], [.9, .7, .5, .1],
                         [.3, .1, .4, .2], [.1, .2, .3, .4],
                         [.1, .1, .3, .2], [.5, .5, .2, .1]], dtype=np.float32)
    return np.repeat(values[:, :, None], 12, axis=2)


def inference_for(ids, raw):
    records = [{'uid': uid, 'sha256': hashlib.sha256(uid.encode()).hexdigest()} for uid in ids]
    digest = hashlib.sha256(''.join(json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n' for r in records).encode()).hexdigest()
    return {'status': 'COMPLETE', 'all_finite': True, 'input_contract': s.INPUT,
            'studies': len(ids), 'member_count': 6, 'requested_amp': 'fp16', 'effective_amp': 'fp16',
            'batch_size': 1, 'residency': 'resident', 'runtime': {'policy': 'strict'},
            'fp32_fallbacks': 0, 'neutral_fills': 0, 'failures': [],
            'input_preparations': len(ids), 'completed_batches': len(ids),
            'input_tensor_hashes': records, 'ordered_inputs_sha256': digest,
            'member_results': [{'name': member['name'], 'checkpoint_sha256': member['sha256'],
                                'status': 'COMPLETE', 'completed_studies': len(ids),
                                'ordered_inputs_sha256': digest, 'probabilities_sha256': s.tensor_sha256(raw[i])}
                               for i, member in enumerate(s.expected_members())]}


def test_passive_import_does_not_import_runtime_or_initialize_cuda():
    code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('subject',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert not any(x in sys.modules for x in ('torch','timm','numpy','cv2','pydicom')); print('passive')")
    proc = subprocess.run([sys.executable, '-c', code, str(ROOT / 'tools/b_rn3f_serving.py')],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == 'passive'


def test_average_tie_ranks_and_singleton():
    values = np.repeat(np.asarray([.2, .1, .2, .9])[:, None], 12, axis=1)
    np.testing.assert_array_equal(s.rank_pct(values)[:, 0], [.625, .25, .625, 1])
    assert s.rank_pct(np.zeros((1, 12), np.float32)).tolist() == [[1.] * 12]
    assert np.all(s.rank_pct(np.zeros((4, 12), np.float32)) == .625)


def test_fixed_fold_rank_blend_not_wrong_reductions():
    raw = raw_example(); ids = list('abcd')
    value = s.aggregate(raw, ids, ids, s.POLICY['member_order'])
    # Independently enumerated average ranks, divided by cohort N=4.
    rank = np.asarray([[1, 2, 3, 4], [4, 3, 2, 1], [3, 1, 4, 2],
                       [1, 2, 3, 4], [1.5, 1.5, 4, 3], [3.5, 3.5, 2, 1]]) / 4
    expected = ((.9 * rank[::2] + .1 * rank[1::2]).sum(axis=0) / 3)
    np.testing.assert_allclose(value[:, 0], expected, rtol=0, atol=1e-15)
    assert value.dtype == np.float64
    assert not np.allclose(value, s.rank_pct(value))  # No final rerank.
    assert not np.allclose(value, raw.mean(axis=0))
    assert not np.allclose(value, .9 * s.rank_pct(raw[::2].mean(0)) + .1 * s.rank_pct(raw[1::2].mean(0)))


@pytest.mark.parametrize('bad', ['nan', 'inf', 'negative', 'over_one', 'shape', 'dtype', 'members', 'uids', 'duplicate', 'empty'])
def test_aggregation_rejects_invalid_or_partial_contract(bad):
    raw = raw_example(); ids = list('abcd'); expected = list(ids); members = list(s.POLICY['member_order'])
    if bad in ('nan', 'inf', 'negative', 'over_one'):
        raw[0, 0, 0] = {'nan': np.nan, 'inf': np.inf, 'negative': -.1, 'over_one': 1.1}[bad]
    if bad == 'shape': raw = raw[:5]
    if bad == 'dtype': raw = raw.astype(np.float64)
    if bad == 'members': members = members[::-1]
    if bad == 'uids': ids = ids[::-1]
    if bad == 'duplicate': ids[-1] = ids[0]
    if bad == 'empty': ids = []
    with pytest.raises(ValueError): s.aggregate(raw, ids, expected, members)


def test_raw_shards_global_rank_and_overlap_missing_rejection():
    raw = raw_example(); ids = list('abcd'); members = s.POLICY['member_order']
    shards = [(['c', 'a'], members, raw[:, [2, 0]]), (['b', 'd'], members, raw[:, [1, 3]])]
    joined = s.merge_raw_shards(ids, shards)
    np.testing.assert_array_equal(joined, raw)
    np.testing.assert_array_equal(s.aggregate(joined, ids, ids, members), s.aggregate(raw, ids, ids, members))
    shard_rank = np.concatenate([s.aggregate(r, u, u, m) for u, m, r in shards])
    assert not np.allclose(shard_rank, s.aggregate(raw, ids, ids, members)[[2, 0, 1, 3]])
    with pytest.raises(ValueError, match='missing'): s.merge_raw_shards(ids, shards[:1])
    with pytest.raises(ValueError, match='duplicate'): s.merge_raw_shards(ids, [*shards, shards[0]])


def test_output_atomic_and_roundtrip(tmp_path):
    raw = raw_example(); ids = list('abcd'); digest = '1' * 64
    assets = tmp_path / 'assets'; assets.mkdir(); (assets / s.MARKER).write_text('{}', encoding='utf-8')
    result = s.atomic_outputs(tmp_path, ids, raw, s.sha(assets / s.MARKER), inference_for(ids, raw),
                              {'target': 'local-reference', 'cohort': 'engineering100'})
    assert result['submission_ready'] is False and result['serving_parity_gate'] == 'NOT_EVALUATED'
    assert not (tmp_path / 'result.pending').exists()
    assert set(p.name for p in (tmp_path / 'result').iterdir()) == {'receipt.json', 'members.npz', 'predictions.csv'}
    s.validate_result(tmp_path, assets, ids, 'local-reference', 'engineering100')
    with pytest.raises(ValueError, match='existing'): s.atomic_outputs(tmp_path, ids, raw, digest, inference_for(ids, raw), {})
    with pytest.raises(ValueError, match='runtime/cohort'): s.validate_result(tmp_path, assets, ids, 't4-engineering', 'engineering100')


@pytest.mark.parametrize('bad', ['missing_member', 'partial_member', 'wrong_weight', 'raw_hash', 'uid_order', 'fallback', 'cleanup', 'partial_inputs', 'failed'])
def test_partial_or_failed_receipt_never_publishes(tmp_path, bad):
    raw = raw_example(); ids = list('abcd'); receipt = inference_for(ids, raw)
    if bad == 'missing_member': receipt['member_results'].pop()
    if bad == 'partial_member': receipt['member_results'][0]['completed_studies'] = 3
    if bad == 'wrong_weight': receipt['member_results'][0]['checkpoint_sha256'] = '0' * 64
    if bad == 'raw_hash': raw[0, 0, 0] = .8
    if bad == 'uid_order': receipt['input_tensor_hashes'].reverse()
    if bad == 'fallback': receipt['fp32_fallbacks'] = 1
    if bad == 'cleanup': receipt['cleanup_failures'] = ['failure']
    if bad == 'partial_inputs': receipt['input_preparations'] = 3
    if bad == 'failed': receipt['status'] = 'FAILED'
    with pytest.raises(ValueError): s.atomic_outputs(tmp_path, ids, raw, '1' * 64, receipt, {})
    assert not (tmp_path / 'result').exists()
    assert not (tmp_path / 'result.pending').exists()


def test_write_failure_retains_pending_but_no_final(tmp_path, monkeypatch):
    raw = raw_example(); ids = list('abcd')
    def fail(*args, **kwargs): raise OSError('simulated disk failure')
    monkeypatch.setattr(s, 'write_new', fail)
    with pytest.raises(OSError): s.atomic_outputs(tmp_path, ids, raw, '1' * 64, inference_for(ids, raw), {})
    assert not (tmp_path / 'result').exists()
    assert (tmp_path / 'result.pending').exists()


def test_target_csv_order_and_nonfinite_rejected(tmp_path):
    path = tmp_path / 'bad.csv'
    for header, row in [([s.UID, *s.TARGETS[::-1]], ['a', *([.5] * 12)]),
                         ([s.UID, *s.TARGETS], ['a', *([float('nan')] * 12)]),
                         ([s.UID, *s.TARGETS], ['a'])]:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream); writer.writerow(header); writer.writerow(row)
        with pytest.raises(ValueError): s.validate_frame(path, ['a'])


@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    """Manifest logic with tiny synthetic bytes, never copy real weights in tests."""
    manifest = {'schema': s.SCHEMA, 'policy': copy.deepcopy(s.POLICY), 'policy_sha256': s.POLICY_SHA,
                'members': s.expected_members(), 'validation_status': 'CPU_STAGED_NOT_INFERENCE_VALIDATED',
                'reference_cache_manifest_sha256': s.CACHE_MANIFEST_SHA,
                'wheels': [{'path': 'wheels/' + name, 'sha256': h} for name, h in s.WHEEL_SHA.items()], 'files': {}}
    inventory = {'b_rn3f_serving.py': '1' * 64, 'config/workspace.toml': s.CONFIG_SHA,
                 'engineering100.json': '2' * 64}
    inventory.update({'training/src/' + n: h for n, h in s.CODE_SHA.items()})
    inventory.update({m['path']: m['sha256'] for m in s.expected_members()})
    inventory.update({'wheels/' + n: h for n, h in s.WHEEL_SHA.items()})
    for relative in inventory:
        p = tmp_path / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture', encoding='utf-8')
    ids = [str(i) for i in range(100)]
    monkeypatch.setattr(s, 'ENGINEERING100_UID_SHA', s.ids_sha(ids))
    (tmp_path / 'engineering100.json').write_text(json.dumps({'ids': ids, 'selection_source_sha256': s.SELECTION_SHA}), encoding='utf-8')
    manifest.update(files=inventory, engineering100_uid_sha256=s.ids_sha(ids), engineering_selection_source_sha256=s.SELECTION_SHA)
    (tmp_path / s.MARKER).write_text(json.dumps(manifest), encoding='utf-8')
    def fake_sha(path):
        if Path(path).resolve() == Path(s.__file__).resolve(): return '1' * 64
        return inventory[Path(path).relative_to(tmp_path).as_posix()]
    monkeypatch.setattr(s, 'sha', fake_sha)
    return tmp_path, manifest


@pytest.mark.parametrize('mutation', [None, 'member', 'policy', 'source', 'wheel', 'missing', 'traversal', 'cache'])
def test_exact_asset_contract(fake_assets, mutation):
    assets, manifest = fake_assets
    if mutation == 'member': manifest['members'].reverse()
    if mutation == 'policy': manifest['policy']['final_rerank'] = True
    if mutation == 'source': manifest['files']['training/src/raptor_dicom.py'] = '3' * 64
    if mutation == 'wheel': manifest['wheels'].pop()
    if mutation == 'missing': manifest['files'].pop('engineering100.json')
    if mutation == 'traversal': manifest['files']['../escape'] = '3' * 64
    if mutation == 'cache': manifest['reference_cache_manifest_sha256'] = '0' * 64
    (assets / s.MARKER).write_text(json.dumps(manifest), encoding='utf-8')
    if mutation is None: s.validate_assets(assets)
    else:
        with pytest.raises(ValueError): s.validate_assets(assets)


def test_public3_refuses_hidden_count(tmp_path):
    with (tmp_path / 'test.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow([s.UID]); writer.writerows([[str(i)] for i in range(4)])
    with pytest.raises(ValueError, match='hidden'): s.resolve_ids(tmp_path, 'public3', tmp_path)


def test_launch_failure_writes_failed_receipt_without_model_call(tmp_path, monkeypatch):
    assets = tmp_path / 'assets'; assets.mkdir(); (assets / s.MARKER).write_text('{}', encoding='utf-8')
    work = tmp_path / 'work'
    monkeypatch.setattr(s, 'validate_assets', lambda p: {})
    monkeypatch.setattr(s, 'resolve_ids', lambda *args: ['a'])
    def fail(*args, **kwargs): raise OSError('synthetic launch failure')
    monkeypatch.setattr(s.subprocess, 'Popen', fail)
    args = SimpleNamespace(assets=str(assets), work=str(work), max_seconds=1, target='local-reference',
                           cache_root='unused', cohort='engineering100', competition=None, opencv_wheel=None, cpu_threads=12,
                           manifest_sha256=s.sha(assets / s.MARKER))
    with pytest.raises(ValueError, match='launch failure'): s.run(args)
    receipt = s.read(work / 'run-receipt.json')
    assert receipt['status'] == 'FAILED' and not (work / 'result').exists()


def test_pinned_allzero_and_slot_boundary_sentinel_cpu(monkeypatch):
    legacy = ROOT / s.LEGACY_ASSETS
    for name, digest in s.CODE_SHA.items():
        assert s.sha(legacy / 'training/src' / name) == digest
    monkeypatch.syspath_prepend(str(legacy / 'training/src'))
    monkeypatch.setenv('KNEE_WORKSPACE_ROOT', str(legacy))
    import raptor_common as common
    import raptor_runtime as runtime
    import torch
    assert not torch.cuda.is_initialized()
    spec = {'slots': common.SLOTS64, 'depth': 64}
    mask = np.zeros(64, dtype=np.uint8)
    centers = common.select_centers(mask, 64, 24, spec, policy='within-slot-valid')
    assert centers == [-1] * 24
    corpus = SimpleNamespace(mask=lambda _: mask, volume=lambda _: np.zeros((64, 2, 2), np.uint8), depth=64, spec=spec)
    value = runtime.prepare_input(corpus, 'synthetic', 24, 224, 'imagenet', 'within-slot-valid')
    assert value.shape == (24, 3, 224, 224) and value.dtype == np.float32
    np.testing.assert_array_equal(value[0, :, 0, 0], -runtime.IMAGENET_MEAN[:, 0, 0] / runtime.IMAGENET_STD[:, 0, 0])
    assert not np.any(value == 0)  # Blank windows are normalized, not zeroed afterward.
    mask[16:20] = 1  # Two planes each side of a slot boundary; never a valid within-slot triplet.
    assert common.select_centers(mask, 64, 24, spec, policy='within-slot-valid') == [-1] * 24
    mask[0:3] = 1
    assert common.select_centers(mask, 64, 24, spec, policy='within-slot-valid') == [1] * 24
    assert s.tensor_sha256(value) == runtime.tensor_sha256(value)
    assert not torch.cuda.is_initialized()
    vol = np.zeros((64, 336, 336), np.uint8)
    observed_base = SimpleNamespace(ids=['synthetic'], contract=s.INPUT['contract'], spec=spec, depth=64,
                                    mask=lambda _: mask, volume=lambda _: vol)
    observed = s.ObservedCorpus(observed_base)
    observed.mask('synthetic'); observed.volume('synthetic')
    assert observed.records['synthetic']['selected_centers'] == [1] * 24
    assert observed.records['synthetic']['volume_sha256'] == hashlib.sha256(vol.tobytes()).hexdigest()
