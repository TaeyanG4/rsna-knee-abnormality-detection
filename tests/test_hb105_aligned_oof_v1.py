"""Bounded CPU scientific invariants; no checkpoints or production readout."""
import csv
import os

import numpy as np
import pytest
from scipy.stats import rankdata
from tools import hb105_aligned_oof_v1 as h


def raw():
    rng = np.random.default_rng(341)
    return {f: {k: rng.integers(0, 10, (9, 12)).astype(np.float32)/10
                for keys in h.FAMILIES.values() for k in keys} for f in range(5)}


def rank(a):
    return (rankdata(a, axis=0, method='average')-1)/(len(a)-1)


def test_probability_first_fp32_and_fixed_pooled_ranks():
    r = raw(); actual = h.variants(r)
    fam = {name: rank(np.concatenate([
        np.stack([r[f][k] for k in keys]).mean(0, dtype=np.float32)
        for f in range(5)])) for name, keys in h.FAMILIES.items()}
    for name, keys in h.BUNDLES.items():
        np.testing.assert_array_equal(actual[name], rank(sum(fam[k] for k in keys)/len(keys)))
    old = rank(np.concatenate([np.mean([rank(r[f][k]) for k in ('M42', 'M43')], axis=0)
                               for f in range(5)]))
    assert not np.array_equal(old, fam['T'])
    assert not np.array_equal(actual['Q7'][:9], rank(actual['Q7'][:9]))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -0.1, 1.1])
def test_nonfinite_or_out_of_range_rejected(bad):
    r = raw(); r[2]['S43'][0, 1] = bad
    with pytest.raises(ValueError, match='invalid probabilities'):
        h.variants(r)


def test_member_and_precision_drift():
    r = raw(); r[0]['M43'] = r[0]['M43'].astype(np.float64)
    with pytest.raises(ValueError, match='invalid probabilities'):
        h.variants(r)
    r = raw(); del r[0]['M43']
    with pytest.raises(ValueError, match='members differ'):
        h.variants(r)
    r = raw(); r[0]['C4'] = r[0]['C4'][:-1]
    with pytest.raises(ValueError, match='row shapes differ'):
        h.variants(r)


def test_uid_order_duplicate_and_missing_rejected(tmp_path):
    path = tmp_path/'predictions.csv'; ids = ['u0', 'u1', 'u2']
    def write(order):
        with path.open('w', newline='') as stream:
            out = csv.writer(stream); out.writerow([h.flip.UID, *h.ml.TARGETS])
            out.writerows([[u, *([0.2]*12)] for u in order])
    write(ids)
    assert h.load_csv(path, ids).dtype == np.float32
    for order in (ids[::-1], ['u0', 'u0', 'u2'], ids[:2]):
        write(order)
        with pytest.raises(ValueError, match='UID/order'):
            h.load_csv(path, ids)


def test_fold_exclusion_and_cross_fold_substitution():
    r = {'train': ['a', 'b'], 'val': ['c']}
    e = dict(train_uid_sha256=h.serving.uid_hash(r['train']),
             heldout_uid_sha256=h.serving.uid_hash(r['val']), train_count=2, heldout_count=1)
    p = dict(train_ids_ordered_sha256=e['train_uid_sha256'],
             eval_ids_ordered_sha256=e['heldout_uid_sha256'])
    h.check_member_ancestry(e, p, r)
    for bad in ({'train': ['a', 'c'], 'val': ['c']}, {'train': ['a', 'c'], 'val': ['b']}):
        with pytest.raises(ValueError, match='fold exclusion'):
            h.check_member_ancestry(e, p, bad)


def test_substitution_drift_rejected_before_load(tmp_path):
    path = tmp_path/'oof.csv'; path.write_bytes(b'fixed science')
    expected = h.pin(path)
    h.checked(tmp_path, 'oof.csv', expected, {})
    path.write_bytes(b'new science!!')
    with pytest.raises(ValueError, match='binding changed'):
        h.checked(tmp_path, 'oof.csv', expected, {})


def test_fivefold_partition_overlap_rejected():
    ids = [str(i) for i in range(4343)]
    folds = [dict(val=ids[f::5], train=[u for i, u in enumerate(ids) if i % 5 != f]) for f in range(5)]
    h.serving.validate_rosters(folds)
    folds[0]['train'][0] = folds[0]['val'][0]
    with pytest.raises(ValueError):
        h.serving.validate_rosters(folds)


def test_three_paired_contrasts_and_bootstrap_arithmetic():
    ids = [str(i) for i in range(24)]
    y = np.tile(np.arange(24) % 2, (12, 1)).T
    mask = np.ones(y.shape, bool); mask[0] = False
    refs = {r: (y, mask, 1) for r in h.ml.RULERS}
    components = [dict(component_id=str(i), members=ids[i:i+2]) for i in range(0, 24, 2)]
    ctx = h.ml.Context(ids, refs, components)
    rng = np.random.default_rng(73)
    vv = {k: rng.integers(0, 5, (24, 12))/4 for k in h.BUNDLES}
    result = h.comparisons(ctx, vv)
    assert len(result) == 3 and ctx.w.shape == (301, 24)
    np.testing.assert_allclose(ctx.delta(vv['add'], vv['Q7'])-ctx.delta(vv['replace'], vv['Q7']),
                               ctx.delta(vv['add'], vv['replace']))
    for a, b in h.CONTRASTS:
        d = ctx.delta(vv[a], vv[b]); out = result[a+'-minus-'+b]
        assert out['release'] == 'FULL_RELEASE_HOLD' and 'verdict' not in out
        assert out['lenses']['L2']['value'] == pytest.approx(d[0, :2].mean())
        assert out['rulers']['jev']['weak_four']['valid_draws'] == 300
        assert out['rulers']['jev']['weak_four']['value'] == pytest.approx(d[0, 0, h.ml.WEAK].mean())


def test_cuda_and_checkpoint_loading_tripwired():
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda._initialized
    assert torch.get_num_threads() <= 2 and torch.get_num_interop_threads() <= 2
    for call in (torch.cuda.is_available, torch.cuda.init, torch.cuda.device_count,
                 lambda: torch.zeros(1).cuda(), lambda: torch.zeros(1, device='cuda'),
                 lambda: torch.load('anything.pt')):
        with pytest.raises(RuntimeError, match='FORBIDDEN_CUDA'):
            call()
