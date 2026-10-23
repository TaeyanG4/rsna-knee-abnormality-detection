"""CPU delivery tests; fixture exports are never trained candidate evidence."""
import csv
import hashlib
import json
import shutil

import pytest
from tools import k1_failure_dataset_v1 as d
from tools import l1_rco_k1_failure_inputs_v1 as inputs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(d.assets.encoded(value))


def fixture_export(tmp_path):
    """Real seed0 predictions + synthetic completion/model, in pytest scratch only."""
    root=tmp_path/'fixture-export'
    p=d.producer
    reg=p.expected_registration()
    prep=p.k.read(p.WORK/'prepare.json')
    write(root/'registration.json',reg)
    write(root/'prepare.json',prep)
    (root/'fit/models').mkdir(parents=True)
    (root/'fit/models/E_prior_s0_f0.pt').write_bytes(b'TEST_ONLY_NOT_A_CHECKPOINT')
    shutil.copyfile(p.k.parent.OUT/'holdout.csv',root/'fit/holdout.csv')
    common=dict(status='COMPLETE',contract_id=reg['contract_id'],model_fold=0,
                model_sha256=d.sha(root/'fit/models/E_prior_s0_f0.pt'))
    write(root/'fit/result.json',dict(common,cpu_test=False,clean_oof=False,
        exposure='EXPOSED_EXPLORATORY',holdout_rows=977,holdout_sha256=d.sha(root/'fit/holdout.csv')))
    write(root/'fit/fit-complete.json',dict(common,epochs=10,optimizer_steps=15950,train_studies=3190))
    write(root/'fit/runtime.json',dict(test_only=True))
    files=['registration.json','prepare.json','fit/result.json','fit/fit-complete.json',
           'fit/runtime.json','fit/holdout.csv','fit/models/E_prior_s0_f0.pt']
    pins={n:d.sha(root/n) for n in files}
    write(root/'EXPORT.json',dict(version='1.0.0',fold=0,contract_id=reg['contract_id'],sha256=pins))
    (root/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(pins.items())))
    return root


def test_real_977_seed0_predictions_equal_gate_and_hold(tmp_path):
    root=fixture_export(tmp_path)
    result=d.gate.evaluate(root)
    assert result['tool_pass'] and result['mean_delta']==0
    assert len(result['wins'])==4 and result['holdout_rows']==977
    assert result['status']=='HOLD'
    with pytest.raises(ValueError,match='screen HOLD'):
        d.verify_export(root/'EXPORT.json')


def test_package_fill_and_tamper(tmp_path,monkeypatch):
    root=fixture_export(tmp_path)
    # Only the efficacy verdict is stubbed: no new fit or improvement is asserted.
    monkeypatch.setattr(d.gate,'evaluate',lambda _:dict(status='SCREEN_PASS_PENDING_REVIEW',test_only=True))
    output=tmp_path/'dataset'
    model=d.package(root/'EXPORT.json',output)
    assert model['id']=='k1_v3g_failure'
    filled=tmp_path/'filled.json'
    d.fill(output,filled)
    config=d.read(filled);template=d.read(d.TEMPLATE)
    assert [m['id'] for m in config['models']]==['k1_v2','k1_v3g_failure','k1_v3g_seed1']
    assert config['models'][0]==template['models'][0] and config['models'][2]==template['models'][2]
    assert config['alpha']==.25 and config['drop_o'] and config['k1_runtime_version']==10
    with pytest.raises(ValueError,match='existing output'):
        d.package(root/'EXPORT.json',output)
    (output/'weights/E_prior_s0_f0.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'):
        d.verify_dataset(output)
    (root/'fit/models/E_prior_s0_f0.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='export hash mismatch'):
        d.verify_export(root/'EXPORT.json')


def test_b3_projection_keeps_exact_values_and_rejects_holdout(tmp_path):
    p=d.producer;reg=p.k.read(p.k.PARENT)
    source=tmp_path/'source.csv';output=tmp_path/'canonical.csv'
    def emit(uid,fold):
        with source.open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['StudyInstanceUID','source_oof_fold','target','weight','agreement','primary_error'])
            writer.writerow([uid,fold,p.k.s.LABELS[0],'1.250000','0','0'])
    emit(reg['train'][0],1)
    receipt=inputs.convert(source,output)
    assert receipt['rows']==receipt['studies']==1 and receipt['holdout_rows']==0
    with output.open(newline='') as f:
        row=next(csv.DictReader(f));assert row['weight']=='1.250000'
    emit(reg['holdout'][0],1)
    with pytest.raises(ValueError,match='holdout UID'):
        inputs.convert(source,tmp_path/'leak.csv')
    emit(reg['train'][0],0)
    with pytest.raises(ValueError,match='training folds only'):
        inputs.convert(source,tmp_path/'fold0.csv')
