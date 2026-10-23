"""CPU checks for preregistration rejection and the primary A/B estimand."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.input_ab_report import apply_gate, mean_fold_bootstrap, validate_summary


def protocol_and_summary():
    args = dict(arch='arch', res=224, epochs=8, fixed_epoch=8, amp='fp16',
                eval_amp='fp16', runtime_policy='strict', wd=.02)
    manifest = dict(epochs=8, expected_args=args, label_hash='label', split_hash='split',
                    training_code_sha256={'train':'code'}, initial_state_sha256='init',
                    corpus_manifest_sha256='corpus', excluded_uids=['bad'],
                    expected_ids={'0':dict(train_ids_sha256='train',eval_ids_sha256='eval',train=80,val=20)})
    run = dict(tag='RUN', arm='A', fold=0, policy='global-valid')
    receipt = dict(requested_amp='fp16',effective_amp='fp16',policy='strict',precision_fallback_allowed=False)
    provenance = dict(checkpoint_selection='fixed-final', code_sha256={'train':'code'},
                      initial_state_sha256='init', corpus_manifest_sha256='corpus',
                      eval_label_hash='label', excluded_ids=['bad'], train_ids_sha256='train',
                      eval_ids_sha256='eval', runtime=receipt,eval_runtime=receipt.copy())
    summary = dict(no_oof_epoch_selection=True,no_gold_epoch_selection=True,selected_epoch_1based=8,
                   selected_checkpoint='best',results={'best':{}},provenance=provenance,
                   args={**args,'tag':'RUN','fold':0,'window_policy':'global-valid'},
                   label_hash='label',eval_label_hash='label',split_hash='split',max_epochs=8,
                   window_policy='global-valid',fold=0,backbone='arch',img=224,train=80,val=20,rows=20,
                   history=[dict(epoch=i,oof_auc=None,gold_auc=None) for i in range(8)],
                   optimizer_updates=80,skipped_updates=0)
    return manifest,run,summary


def test_fixed_run_matches_frozen_protocol():
    manifest,run,summary=protocol_and_summary()
    validate_summary(summary,manifest,run)


@pytest.mark.parametrize('path,value', [
    (('label_hash',),'changed'),
    (('args','wd'),.2),
    (('args','arch'),'other'),
    (('provenance','corpus_manifest_sha256'),'other'),
    (('provenance','initial_state_sha256'),'other'),
    (('provenance','train_ids_sha256'),'other'),
    (('provenance','eval_runtime','effective_amp'),'bf16'),
    (('selected_checkpoint',),'swa'),
    (('no_gold_epoch_selection',),False),
])
def test_identically_wrong_arms_cannot_pass_preregistration(path,value):
    manifest,run,summary=protocol_and_summary()
    cursor=summary
    for key in path[:-1]: cursor=cursor[key]
    cursor[path[-1]]=value
    with pytest.raises(ValueError): validate_summary(summary,manifest,run)


def test_fixed_epoch_rejects_interim_gold_exposure():
    manifest,run,summary=protocol_and_summary()
    summary['history'][2]['gold_auc']=.9
    with pytest.raises(ValueError,match='inspected'): validate_summary(summary,manifest,run)


def test_custom_manifest_gate_is_applied_without_hidden_constants():
    rows=[dict(delta=.003,per_target_delta={'ACL':-.02},skipped_update_fraction={'A':.005,'B':0})]*2
    gate=dict(both_fold_delta_positive=True,mean_fold_delta_at_least=.004,
              target_drop_review_below=-.03,max_skipped_update_fraction=.01)
    result=apply_gate(rows,gate)
    assert not result['development_screen'] and not result['target_review_required']
    gate.update(mean_fold_delta_at_least=.002,max_skipped_update_fraction=.001)
    assert not apply_gate(rows,gate)['numeric_gate']


def test_bootstrap_estimates_mean_fold_delta_not_pooled_ranking():
    y=np.tile(np.arange(40)[:,None]%2,(1,12)).astype(float)
    groups=np.repeat(np.arange(20),2)
    # Each report group contains both classes; opposing fold gains cancel exactly.
    whole=[(y,1-y,y,groups),(y,y,1-y,groups+20)]
    interval=mean_fold_bootstrap(whole,reps=10)
    assert interval['estimand']=='mean_fold_auc_delta'
    assert interval['valid_reps']==10
    assert interval['low']==interval['high']==0
