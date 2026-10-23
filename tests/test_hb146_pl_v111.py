"""v1.1 train-only mapping, masks and explicit exposure contract."""
import numpy as np
import pytest
from tools import hb146_pl_labels_v111 as pl
from tools import hb146_screen_v111 as screen


def test_train_only_intercept_mean_and_rank():
    ids = ['a','b','v']
    l2 = {u: np.full(12,v,np.float32) for u,v in zip(ids,[.2,.6,.9])}
    t = {u: np.full(12,v,np.float32) for u,v in zip(ids,[.01,.2,.8])}
    scaled, meta = pl.scaled_teacher(l2,t,ids[:2],ids)
    np.testing.assert_allclose(scaled[:2].mean(0), .4, atol=1e-7)
    assert np.all(np.diff(scaled,axis=0)>0)
    t['v'][:]=.001
    l2['v'][:]=0
    changed, other=pl.scaled_teacher(l2,t,ids[:2],ids)
    assert meta==other and meta['fit_rows']==2
    np.testing.assert_array_equal(scaled[:2],changed[:2])


def test_silent_fill_and_shared_l2_weights():
    base=np.full((3,12),.2,np.float32)
    teacher=np.full((3,12),.8,np.float32)
    mask=np.zeros((3,12),bool); mask[0,0]=True
    arms=pl.arm_targets(base,teacher,mask)
    np.testing.assert_array_equal(arms['fill'][~mask],base[~mask])
    np.testing.assert_array_equal(arms['fill'][mask],teacher[mask])
    np.testing.assert_allclose(arms['global'],.5)
    l2={str(i):a for i,a in enumerate(base)}
    np.testing.assert_allclose(pl.frozen_weights(l2,['0','1']),4)


def test_exposed_flag_and_limitations_cannot_be_waived():
    p=dict(status=pl.STATUS, limitations=pl.LIMITATIONS, exposed_exploratory=True)
    with pytest.raises(ValueError,match='explicit'):
        pl.require_exposed(p)
    pl.require_exposed(p,True)
    with pytest.raises(ValueError,match='limitations'):
        pl.require_exposed({**p,'limitations':[]},True)
    with pytest.raises(ValueError,match='explicit'):
        pl.register('unused')
    with pytest.raises(ValueError,match='explicit'):
        screen.prepare('unused')


def test_recorded_exploratory_admission(tmp_path,monkeypatch):
    p=dict(status=pl.STATUS, limitations=pl.LIMITATIONS, exposed_exploratory=True)
    monkeypatch.setattr(screen,'prepared',lambda *a: {})
    monkeypatch.setattr(pl,'validate_labels',lambda *a: ({},{},p))
    pl.write(tmp_path/'prepare.json',p)
    with pytest.raises(ValueError,match='explicit'):
        screen.admit(tmp_path)
    admitted=screen.admit(tmp_path,exposed_exploratory=True)
    assert admitted['status']==pl.STATUS
    assert pl.read(tmp_path/'admission.json')==admitted


def test_silence_missing_excluded_and_strict_threshold(monkeypatch,tmp_path):
    marker={'a':np.full(12,.5),'b':np.full(12,.4)}
    guard={'a':np.array([.58,.59,np.nan]+[.2]*9),'b':np.zeros(12)}
    monkeypatch.setattr(pl,'check_pins',lambda *a: None)
    monkeypatch.setattr(pl,'pins',lambda *a: {})
    monkeypatch.setattr(pl,'table',lambda p,**k: marker if p==pl.MARKER else guard)
    mask,_=pl.silence_mask(['a','b','missing'])
    assert mask[0,0] and not mask[0,1] and not mask[0,2]
    assert not mask[1:].any()
