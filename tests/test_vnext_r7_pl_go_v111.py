import copy
from pathlib import Path
import pytest
from tools import vnext_r7_pl_go_v111 as go
from tools import vnext_r7_pl_train_v110 as train
from pl_readout_fixture_v111 import make_readout


@pytest.mark.parametrize('winner',['global','fill'])
def test_real_readout_selects_and_binds_evidence(tmp_path,winner):
    path=make_readout(tmp_path/'screen',winner)
    doc=train.p.read(path)
    assert doc['status']==go.source.pl.STATUS
    assert doc['registered_go']['selected_arm']==winner
    result=go.validate(train.asset(path),winner)
    assert result['registered_go']==doc['registered_go']
    assert result['source_sha256']==train.p.sha(path)
    assert result['evidence_sha256']
    assert go.source.choose_arm.__module__=='tools.hb146_readout_v111'


@pytest.mark.parametrize('kind',['no-go','engineering','wrong-arm','legacy','nested','version',
    'scope','decision','gate','macro','image','identity','missing-evidence','hash','mixed-checkpoint'])
def test_readout_fail_closed(tmp_path,kind):
    path=make_readout(tmp_path/'screen',winner=None if kind=='no-go' else 'global',
                      engineering=kind=='engineering')
    doc=train.p.read(path)
    arm='fill' if kind=='wrong-arm' else 'global'
    if kind=='legacy':doc=dict(status='GO',winning_arm='global',clean_oof=False,exposure='EXPOSED_EXPLORATORY')
    elif kind=='nested':doc=doc['registered_go']
    elif kind=='version':doc['version']='1.1.0'
    elif kind=='scope':doc['registered_go']['status_scope']='CLEAN'
    elif kind=='decision':doc['registered_go']['selected_arm']='fill'
    elif kind=='gate':doc['arms']['fill']['registered_go']['GO']=True
    elif kind=='macro':doc['arms']['global']['report_seed_mean_macros']['jev']['value']=0.
    elif kind=='image':doc['arms']['global']['lenses']['L7']['entries']['gold58']['value']=0.
    elif kind=='identity':doc['prediction_bindings']['val-global-42']['identity']['seed']=43
    elif kind=='missing-evidence':doc['prediction_bindings'].pop('val-control-42')
    elif kind=='hash':doc['ruler_input_hashes']={'synthetic':'bad'}
    elif kind=='mixed-checkpoint':doc['prediction_bindings']['val-global-42']['checkpoint_sha256']='0'*64
    train.p.dump(path,doc)
    with pytest.raises(ValueError,match='scientific GO'):go.validate(train.asset(path),arm)


@pytest.mark.parametrize('field',['registered_go','prediction_bindings','ruler_input_hashes','teacher_f0_binding'])
def test_post_admission_readout_mutations_are_rejected(tmp_path,field):
    path=make_readout(tmp_path/'screen')
    pin=train.asset(path)
    record=dict(status='ADMITTED',go=pin,spec_sha256='spec',decision=go.validate(pin,'global'))
    doc=train.p.read(path)
    doc[field]={}
    train.p.dump(path,doc)
    with pytest.raises(ValueError):go.admission(record,'global','spec')


def test_replaced_consistent_source_cannot_reuse_admission(tmp_path):
    path=make_readout(tmp_path/'screen')
    pin=train.asset(path)
    record=dict(status='ADMITTED',go=pin,spec_sha256='spec',decision=go.validate(pin,'global'))
    doc=train.p.read(path)
    doc['notes'].append('Different source readout')
    train.p.dump(path,doc)
    record['go']=train.asset(path)
    with pytest.raises(ValueError,match='evidence changed'):go.admission(record,'global','spec')
