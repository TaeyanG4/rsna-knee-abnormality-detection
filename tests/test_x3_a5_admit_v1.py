"""Truthful nested X3 receipts, legacy refusal and offline wrapper acceptance."""
import json
import numpy as np
import pytest
from tools import x3_a5_admit_v1 as m
from tools import k1_admit_v1 as old
from tools import x3_a5_integration_v1 as x
from tools import k1n_cpu_replay_v1 as replay
from test_k1_admit_v1 import case, put, change, transport
from test_k1_admit_a5_v1 import a5_case, refresh


def test_ordinary_a5_admission_outputs_unchanged(a5_case, monkeypatch, capsys):
    monkeypatch.setattr(m, 'TEMPLATE', old.TEMPLATE)
    monkeypatch.setattr(m, 'LEDGER', old.LEDGER)
    out = m.prepare(**a5_case)
    before = {p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert json.loads(before['ordinary-visible-registration.json'])['tool_version'] == '1.0.5'
    old.prepare(**a5_case)
    assert before == {p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out


@pytest.fixture
def x3_case(a5_case, monkeypatch):
    c=a5_case
    monkeypatch.setattr(m,'TEMPLATE',old.TEMPLATE)
    monkeypatch.setattr(m,'LEDGER',old.LEDGER)
    stage=c['stage']
    change(stage/'config.json',lambda z:z.update(drop_o=False))
    cfg=json.loads((stage/'config.json').read_bytes()); cfg.pop('a5')
    digest=m.sha(m.encoded(cfg))
    reg=json.loads((stage/'registration.json').read_bytes())
    formula=reg['a5_contract']['formula']+'; inherited outer K1: '+x.FORMULA+'; test K1'
    change(stage/'registration.json',lambda z:z.update(inf3_a5=True,drop_o=False,
        formula=formula,config_sha256=digest,registered_config_sha256=digest))
    rng=np.random.default_rng(381)
    for key,n in [('visible_dir',3),('n32_dir',32)]:
        d=c[key]; diagnostic=n==32; ids=['a','b','c'] if n==3 else list(map(str,range(n)))
        public=replay.csv_bytes(ids,rng.random((n,12)))
        original=replay.csv_bytes(ids,rng.random((n,12)))
        prime=replay.csv_bytes(ids,rng.random((n,12)))
        own=rng.random((n,12))
        inf3=replay.csv_bytes(ids,.85*x.util.ranks(x.util.table(prime)[1])+.15*x.util.ranks(own))
        inputs=dict(schema='x3-a5-inputs/v1',ids=ids,own=own.tolist(),p_failed=[False]*n,o_failed=[False]*n,
            original_inf3_sha256=m.sha(original),original_p_sha256=m.sha(public))
        put(d/'x3-a5-inputs.json',inputs)
        for name,data in [('x3-a5-original-p.csv',public),('x3-a5-original-inf3.csv',original),
                          ('a5-pprime.csv',prime),('x3-a5-inf3.csv',inf3),('k1sr-parent.csv',inf3)]:
            (d/name).write_bytes(data)
        change(d/'a5-receipt.json',lambda z:z.update(original_p_sha256=m.sha(public),pprime_sha256=m.sha(prime)))
        comp=dict(schema='x3-a5-inf3/v1',status='COMPLETE_INF3_A5',formula=x.FORMULA,
            diagnostic=diagnostic,never_submit=diagnostic,studies=n,original_inf3_sha256=m.sha(original),
            original_p_sha256=m.sha(public),pprime_sha256=m.sha(prime),inf3_sha256=m.sha(inf3),
            inputs_sha256=m.sha(m.encoded(inputs)),a5_receipt_sha256=m.sha((d/'a5-receipt.json').read_bytes()))
        put(d/'x3-a5-inf3-receipt.json',comp)
        def parent(z):
            z.update(drop_o=False,submission_sha256=None if diagnostic else m.sha(original),
                x3_a5_inputs_sha256=m.sha(m.encoded(inputs)),a5_original_p_sha256=m.sha(public),
                skipped_members=[],intended_members=['R','R224','S_R7','EXP143v2','EXP155','Q_A5q'],
                executed_members=['R','R224','S_R7','EXP143v2','EXP155','Q_A5q'])
            z['fallbacks']['a172_prediction']=0
        change(d/'c85r7-receipt.json',parent)
        def k1(z):
            z.update(drop_o=False,formula=formula,config_sha256=digest,registered_config_sha256=digest,
                parent_mode='INF3_A5_PPRIME',parent_sha256=m.sha(inf3),
                inf3_a5_receipt_sha256=m.sha(m.encoded(comp)),submission_sha256=m.sha(inf3))
            if diagnostic:
                for row in z['per_study']: row['config_sha256']=digest
        change(d/'k1sr-receipt.json',k1)
        refresh(d)
        if not diagnostic:
            (d/'submission.csv').write_bytes(inf3)
            m.TEMPLATE.write_bytes(inf3)
            transport(d)
    return c


def test_offline_admitted_and_legacy_refused(x3_case,capsys):
    with pytest.raises(old.Refuse,match='exact A5 receipt'):
        old.validate(**x3_case)
    out=m.prepare(**x3_case)
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out
    result=json.loads((out/'ordinary-visible-registration.json').read_bytes())
    assert result['tool_version']=='1.0.8-x3' and result['projection'] is None
    assert any('nested INF3 replay' in v for v in m.validate(**x3_case)[0].checks)


def test_actual_candidate_contract_offline_acceptance(x3_case,capsys):
    from tools import x3_a5_stage_v1 as builder
    outputs,reg=builder.artifacts()
    stage=x3_case['stage'].parent/('v'+builder.RELEASE)
    for name,data in outputs.items():
        p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    reg['stage']=str(stage);put(stage/'registration.json',reg)
    x3_case.update(stage=stage,kernel=reg['kernel_id'])
    keys=['kernel_id','release_version','formula','contract','model_sha256','parent_weight',
          'k1_weight','r224_weight','drop_o','config_sha256','registered_config_sha256','a5_contract_sha256']
    for key in ('visible_dir','n32_dir'):
        d=x3_case[key]; diag=key=='n32_dir'
        def k1(z):
            z.update({k:reg[k] for k in keys})
            if diag:
                z['kernel_id']+='-n32';z['release_version']+='-n32'
                for row in z['per_study']:row['config_sha256']=reg['config_sha256']
        change(d/'k1sr-receipt.json',k1)
        change(d/'c85r7-receipt.json',lambda z:z.update(version=reg['release_version']+('-n32' if diag else '')))
        change(d/'a5-receipt.json',lambda z:z.update(formula=reg['a5_contract']['formula'],
            checkpoint_hashes=reg['a5_contract']['checkpoint_sha256']))
        change(d/'x3-a5-inf3-receipt.json',lambda z:z.update(a5_receipt_sha256=m.sha((d/'a5-receipt.json').read_bytes())))
        change(d/'k1sr-receipt.json',lambda z:z.update(inf3_a5_receipt_sha256=m.sha((d/'x3-a5-inf3-receipt.json').read_bytes())))
        refresh(d)
        if not diag:transport(d,kernel=reg['kernel_id'])
    with pytest.raises(old.Refuse):old.validate(**x3_case)
    m.prepare(**x3_case)
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out


@pytest.mark.parametrize('directory',['visible_dir','n32_dir'])
@pytest.mark.parametrize('name',['x3-a5-inputs.json','x3-a5-original-inf3.csv','x3-a5-original-p.csv',
    'a5-pprime.csv','x3-a5-inf3.csv','x3-a5-inf3-receipt.json','k1sr-parent.csv'])
def test_chain_tamper(x3_case,directory,name):
    p=x3_case[directory]/name
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(m.Refuse): m.validate(**x3_case)


def test_rebound_wrong_math_refused(x3_case):
    d=x3_case['visible_dir']; wrong=(d/'a5-pprime.csv').read_bytes()
    (d/'x3-a5-inf3.csv').write_bytes(wrong); (d/'k1sr-parent.csv').write_bytes(wrong)
    change(d/'x3-a5-inf3-receipt.json',lambda z:z.update(inf3_sha256=m.sha(wrong)))
    change(d/'k1sr-receipt.json',lambda z:z.update(parent_sha256=m.sha(wrong),
        inf3_a5_receipt_sha256=m.sha((d/'x3-a5-inf3-receipt.json').read_bytes())))
    transport(d)
    with pytest.raises(m.Refuse,match='nested INF3 replay'): m.validate(**x3_case)


@pytest.mark.parametrize('directory', ['visible_dir', 'n32_dir'])
@pytest.mark.parametrize('missing', ['EXP143v2', 'EXP155', 'Q_A5q'])
def test_retained_o_requires_every_executed_member(x3_case, directory, missing):
    d = x3_case[directory]
    change(d/'c85r7-receipt.json', lambda z: z.update(
        executed_members=[v for v in z['executed_members'] if v != missing]))
    refresh(d)
    if directory == 'visible_dir':
        transport(d)
    with pytest.raises(m.Refuse, match='requires executed O members'):
        m.validate(**x3_case)


@pytest.mark.parametrize('directory', ['visible_dir', 'n32_dir'])
@pytest.mark.parametrize('receipt', ['k1sr-receipt.json', 'c85r7-receipt.json'])
@pytest.mark.parametrize('value', [True, 0, None])
def test_retained_o_drop_o_consistency(x3_case, directory, receipt, value):
    d = x3_case[directory]
    change(d/receipt, lambda z: z.update(drop_o=value))
    refresh(d)
    if directory == 'visible_dir':
        transport(d)
    with pytest.raises(m.Refuse, match='drop_o'):
        m.validate(**x3_case)


@pytest.mark.parametrize('directory', ['visible_dir', 'n32_dir'])
@pytest.mark.parametrize('receipt', ['k1sr-receipt.json', 'c85r7-receipt.json'])
def test_retained_o_missing_drop_o_matches_ordinary(x3_case, directory, receipt):
    d = x3_case[directory]
    change(d/receipt, lambda z: z.pop('drop_o'))
    refresh(d)
    if directory == 'visible_dir':
        transport(d)
    if receipt == 'k1sr-receipt.json':
        with pytest.raises(m.Refuse, match='drop_o'):
            m.validate(**x3_case)
    else:
        # The ordinary contract treats an omitted parent flag as false.
        m.validate(**x3_case)
