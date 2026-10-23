"""Metric invariants for CPU-only OOF diagnosis, including ties and group weights."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.oof_error_audit import weighted_auc,row_error_contributions,conditional_intervals


def test_weighted_auc_equals_explicit_group_resampling_with_ties():
    y=np.array([0,1,0,1,1,0])
    p=np.array([.2,.2,.5,.5,.9,.9])
    weights=np.array([2,2,0,0,3,3])
    idx=np.repeat(np.arange(len(y)),weights)
    assert np.isclose(weighted_auc(y,p,weights),roc_auc_score(y[idx],p[idx]))
    assert np.isnan(weighted_auc(y,p,(1-y).astype(float)))


def test_pair_endpoint_contributions_sum_to_macro_deficit():
    y=np.array([0,1,0,1,1,0])
    p=np.array([.2,.2,.5,.5,.9,.9])
    assert np.isclose(row_error_contributions(y,p).sum(),(1-roc_auc_score(y,p))/12)
    assert np.isclose(row_error_contributions(y,np.ones(6)*.5).sum(),.5/12)
    assert not row_error_contributions(y,y.astype(float)).any()


def test_group_bootstrap_keeps_identical_models_paired():
    y=np.tile((np.arange(40)%2)[:,None],(1,12))
    p=y*.8+.1
    result=conditional_intervals(y,{'R1':p,'identical':p.copy()},np.repeat(np.arange(20),2),10,42)
    for ci in result['models']['identical']['paired_delta_vs_R1'].values():
        assert ci['low']==ci['high']==0
