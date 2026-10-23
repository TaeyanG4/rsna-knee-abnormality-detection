"""Synthetic CPU-only evaluator checks: never open real predictions or labels."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import exp083_slot_bias_report as r


def sample():
    y = np.tile(np.array([0, 1, 0, 1], dtype=np.int8)[:,None], (1,12))
    good = (.1 + .8 * y).astype(np.float32)
    return y, {'control':1-good, 'treatment':good, 'frozen_b8':np.full_like(good,.5)}


def histories():
    receipt = dict(requires_skip_review=False, equal_realized_update_budget=True, paired_rng_trace_exact=True,
        arms={a:dict(attempts=1100, updates=1100, skips=0, last_batch_size=19,
                    any_skips_require_review=False, separate_optimizer_scheduler_scaler=True) for a in ('control','treatment')})
    history = [dict(epoch=i, attempts=(i+1)*55, updates=(i+1)*55, skips=0, loss=1., F2_metric=None,gold_metric=None) for i in range(20)]
    scale = [dict(attempt=i+1,before=1024.,after=1024.) for i in range(1100)]
    trace = [dict(epoch=i//55,batch=i%55,uids_sha256=str(i),before=str(i),after=str(i+1)) for i in range(1100)]
    return receipt, {a:deepcopy(history) for a in ('control','treatment')}, {a:deepcopy(scale) for a in ('control','treatment')}, {a:deepcopy(trace) for a in ('control','treatment')}


def test_primary_secondary_exact_and_full12_only():
    y,p = sample(); rows = r.target_rows(y,p); result=r.macro(rows)
    assert result['primary_delta'] == 1 and result['secondary_control_minus_frozen'] == -.5
    assert rows[-1]['target'] == 'Fracture' and rows[-1]['P'] == rows[-1]['N'] == 2
    y[:,0] = 0
    result = r.macro(r.target_rows(y,p))
    assert result['primary_delta'] is None and not result['complete12']


def test_paired_components_share_draws_across_all_targets_and_arms():
    y,p=sample()
    # Both components include positive and negative: every macro draw is valid.
    value=r.paired_bootstrap(y,p,['g1','g1','g2','g2'],reps=12,seed=4)
    assert (value['attempted'],value['valid'],value['invalid']) == (12,12,0)
    assert value['intervals']['primary_delta']['macro12'] == [1.,1.]
    p['treatment'] = p['control'].copy()
    repeated=r.paired_bootstrap(y,p,['g1','g1','g2','g2'],reps=12,seed=4)
    assert repeated['draws_sha256'] == value['draws_sha256']
    assert repeated['intervals']['primary_delta']['macro12'] == [0.,0.]


def test_single_class_any_target_discards_whole_macro_draw_without_retry():
    y,p=sample(); y[:,0]=0
    value=r.paired_bootstrap(y,p,['a','b','c','d'],reps=7,seed=1)
    assert value['attempted'] == value['invalid'] == 7 and value['valid'] == 0
    assert value['invalid_targets']['ACL'] == 7
    assert value['intervals']['primary_delta']['macro12'] is None
    assert all(x is None for x in value['intervals']['primary_delta']['targets'].values())


def test_component_multiplicity_keeps_study_weighting():
    from sklearn.metrics import roc_auc_score
    y,p=sample(); weights=np.array([2,2,1,1])
    p['treatment'][:,0]=np.array([.2,.1,.3,.9])
    rows=r.target_rows(y,p,weights=weights)
    expanded=np.repeat(np.arange(4),weights)
    assert rows[0]['auc']['treatment'] == roc_auc_score(y[expanded,0],p['treatment'][expanded,0])


def test_skip_or_trace_or_intermediate_metric_blocks_evaluation():
    values=histories(); r.validate_counts(*values)
    for changed in ('skip','trace','metric','scale','epoch'):
        receipt,h,s,t=histories()
        if changed=='skip': receipt['requires_skip_review']=True
        if changed=='trace': t['treatment'][10]['before']='different'
        if changed=='metric': h['control'][0]['F2_metric']=.9
        if changed=='scale': s['control'][12]['after']=512.
        if changed=='epoch': h['control'][-1]['attempts']=1099
        with pytest.raises(ValueError): r.validate_counts(receipt,h,s,t)


def test_exact_prediction_csv_order_columns_and_values(tmp_path):
    _,p=sample(); values=p['treatment']; ids=['a','b','c','d']; csv=tmp_path/'pred.csv'
    frame=pd.DataFrame(values,columns=r.TARGETS);frame.insert(0,r.UID,ids);frame.to_csv(csv,index=False)
    r.validate_prediction(values,ids,r.TARGETS,csv)
    frame.iloc[::-1].to_csv(csv,index=False)
    with pytest.raises(ValueError,match='order'):r.validate_prediction(values,ids,r.TARGETS,csv)
    frame.to_csv(csv,index=False); wrong=values.copy();wrong[0,0]=.7
    with pytest.raises(ValueError,match='differ'):r.validate_prediction(wrong,ids,r.TARGETS,csv)
    with pytest.raises(ValueError):r.validate_prediction(values.astype(np.float64),ids,r.TARGETS)


def test_incomplete_job_rejected_before_loading_checkpoint_or_predictions(tmp_path):
    reg={'run_dir':'run','job_dir':'job'}
    with pytest.raises(ValueError,match='completed'):r.validate_completed(tmp_path,reg,{})


def test_containment_and_frozen_hash_reject_changes(tmp_path):
    with pytest.raises(ValueError):r.contained(tmp_path,'../outside')
    f=tmp_path/'x';f.write_text('before');expected=r.sha(f); f.write_text('after')
    with pytest.raises(ValueError,match='Hash mismatch'):r.bind(tmp_path,'x',expected,{})


def test_changed_analysis_registration_rejected_before_other_reads(tmp_path):
    p=tmp_path/'registration.json';p.write_text('{}');expected=r.sha(p);p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='registration hash mismatch'):
        r.run(p,expected,tmp_path)


def test_passive_import_does_not_load_torch_or_read_scores():
    command="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from tools import exp083_slot_bias_report;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"
    p=subprocess.run([sys.executable,'-c',command,str(r.ROOT)],capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    assert p.returncode==0,p.stderr
