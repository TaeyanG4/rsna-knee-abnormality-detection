"""Tests for identity, factorial pairing, and harm/uncertainty refusal."""
import numpy as np
import pytest
from tools import hb178_stack_analysis_v1 as h
from tools import multi_lens_readout as ml


def test_uid_alignment_and_refuse_duplicates_or_missing():
    p = np.arange(36).reshape(3,12)/36
    assert np.array_equal(h.align(['a','b','c'],['c','a','b'],p),p[[2,0,1]])
    for ids in (['a','b','b'],['a','b','d']):
        with pytest.raises(ValueError): h.align(['a','b','c'],ids,p)


@pytest.mark.parametrize('bad',[np.full((3,12),np.nan),np.full((3,12),1.1),np.ones((3,11))])
def test_invalid_predictions_refused(bad):
    with pytest.raises(ValueError): h.probs(bad,3)


def test_no_false_full_stack_support():
    assert h.common_folds(dict(native=[0,1],scale=[0],multiview=[2,3])) == []
    assert h.common_folds(dict(native=[0,1],scale=[0])) == [0]


def test_factorial_ranks_each_member_before_mixing():
    rng=np.random.default_rng(178)
    a=[rng.random((20,12)) for _ in range(4)]
    f=h.factorial(*a)
    assert np.array_equal(f['stack'],(ml.rank(a[1])+ml.rank(a[3]))/2)
    assert np.array_equal(f['baseline'],(ml.rank(a[0])+ml.rank(a[2]))/2)
    assert not np.array_equal(f['stack'],ml.rank((a[1]+a[3])/2))


def test_paired_contrasts_and_interaction_not_sum_of_ci():
    rng=np.random.default_rng(42)
    s={k:rng.random((301,4,12)) for k in h.ARMS}
    d=h.contrasts(s)
    np.testing.assert_allclose(d['interaction'],d['stack_minus_baseline']-
                              d['native_minus_baseline']-d['scale_minus_baseline'])
    np.testing.assert_allclose(d['stack_minus_native'],s['stack']-s['native'])


def scores():
    return {k:np.full((301,4,12),.8+i*.002) for i,k in enumerate(h.ARMS)}


def test_harm_or_missing_draw_refuses_flag():
    s=scores()
    assert h.summarize(s)['exploratory_consistency']
    s['stack'][:,3,10]=.79
    assert not h.summarize(s)['floor_pass']
    s=scores();s['stack'][1,0,0]=np.nan
    assert not h.summarize(s)['all_draws_valid']
    assert not h.summarize(s)['exploratory_consistency']


def test_oai_masked_targets_do_not_pollute_weak_or_mapped_nine():
    s={k:v[:,0,:].copy() for k,v in scores().items()}
    for v in s.values(): v[:,[1,10,11]]=np.nan
    result=h.summarize(s,panel=True)
    assert result['all_draws_valid'] and result['floor_pass']
    assert result['contrasts']['stack_minus_baseline']['primary']['valid_draws']==300


def test_bootstrap_auc_ties_match_expanded_weighted_rows():
    from sklearn.metrics import roc_auc_score
    p=np.tile([.2,.5,.5,.8],(12,1)).T
    y=np.tile([0.,1.,0.,1.],(12,1)).T
    w=np.array([[1,1,1,1],[2,1,3,2]])
    got=ml.auc_draws(p,y,np.ones_like(y,dtype=bool),w)
    for i in range(2):
        assert got[i,0]==pytest.approx(roc_auc_score(y[:,0],p[:,0],sample_weight=w[i]))
