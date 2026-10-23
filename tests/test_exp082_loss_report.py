"""Synthetic paired-metric and completion checks; no actual predictions/Gold."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import exp082_loss_report as r


def sample():
    y=np.tile(np.array([0,1,0,1,0,1],dtype=np.float32)[:,None],(1,12))
    good=.1+.8*y; bad=1-good; b=np.full_like(y,.5)
    return y,b,bad,good


def test_known_perfect_reversal_improves_both_registered_contrasts():
    y,b,bad,good=sample();rows=r.target_rows(y,r.scores_for(b,bad,good))
    assert len(rows)==12 and all(x['P']==x['N']==3 for x in rows)
    for key in r.PAIRS:
        assert r.aggregate(rows)[key]['macro12']==pytest.approx(1.)
        assert sum(x['macro_contribution'][key] for x in rows)==pytest.approx(1.)


def test_identical_candidate_zero_in_metrics_and_paired_bootstrap():
    y,b,bad,good=sample();scores=r.scores_for(b,good,good)
    ci=r.intervals(y,scores,np.array(['a','a','b','b','c','c']),reps=12)
    for key in r.PAIRS:
        assert r.aggregate(r.target_rows(y,scores))[key]['macro12']==0
        assert ci['contrasts'][key]['macro12']==dict(requested_reps=12,valid_reps=12,low=0.,high=0.)


def test_rank_ties_and_no_final_rerank():
    y,b,bad,good=sample();scores=r.scores_for(b,bad,good)
    assert np.unique(scores['B']).tolist()==[7/12]
    # tied negative rank=(1+2+3)/3/6, positive=(4+5+6)/3/6
    assert scores['blend_ones'][0,0]==pytest.approx(.9*7/12+.1/3)
    assert scores['blend_ones'][1,0]==pytest.approx(.9*7/12+.1*5/6)


def test_subgroups_and_bootstrap_consume_frozen_scores_without_rerank(monkeypatch):
    y,b,bad,good=sample();scores=r.scores_for(b,bad,good)
    monkeypatch.setattr(r,'fixed_blend',lambda *a:pytest.fail('Unexpected rerank'))
    take=np.array([True,True,False,False,True,True])
    rows=r.target_rows(y[take],{k:v[take] for k,v in scores.items()})
    assert r.aggregate(rows)['primary_blend_ones_minus_clip']['macro12']==1.
    ci=r.intervals(y,scores,np.array(['a','a','b','b','c','c']),reps=5)
    assert ci['contrasts']['primary_blend_ones_minus_clip']['macro12']['low']==1.


def test_single_class_group_never_fabricates_macro12():
    y,b,bad,good=sample();scores=r.scores_for(b,bad,good)
    rows=r.target_rows(y[::2],{k:v[::2] for k,v in scores.items()})
    assert all(x['N']==3 and x['P']==0 and all(v is None for v in x['auc'].values()) for x in rows)
    assert all(x['macro12'] is None for x in r.aggregate(rows).values())


def test_missing_completion_stops_before_reading_any_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(r.guard,'read',lambda *a:pytest.fail('Read before complete'))
    with pytest.raises(ValueError,match='Completed SUCCESS'):
        r.require_complete(tmp_path,[{'tag':r.guard.TAG,'command':['python','train.py']}])


def test_failed_job_rejected_even_if_success_exists(tmp_path,monkeypatch):
    d=tmp_path/'artifacts/logs'/r.guard.TAG;d.mkdir(parents=True)
    (d/'FAILED').write_text('exit1');(d/'SUCCESS').write_text('exit0')
    monkeypatch.setattr(r.guard,'read',lambda *a:pytest.fail('Read failed artifact'))
    with pytest.raises(ValueError,match='Completed SUCCESS'):
        r.require_complete(tmp_path,[{'tag':r.guard.TAG}])


def test_prediction_input_invalid_rejected():
    y,b,bad,good=sample();bad[0,0]=np.nan
    with pytest.raises(ValueError,match='Invalid probabilities'):
        r.scores_for(b,bad,good)
