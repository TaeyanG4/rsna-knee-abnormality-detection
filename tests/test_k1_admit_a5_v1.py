"""A5 offline admission: refusal coverage and byte-identical legacy outputs."""
import json
from types import ModuleType

import pytest

from tools import k1_admit_v1 as m
from tools import k1n_qualify_v1 as q
from test_k1_admit_v1 import case, put, change, transport, k1n, n32
from test_k1n_qualify_v1 import ratio_case


def attest(directory, name, kernel):
    put(directory / (name + '.download.json'), dict(
        status='DOWNLOADED', kernel=kernel, version=1, file=name,
        local_name=name, sha256=m.sha((directory / name).read_bytes())))


def refresh(directory):
    """Refresh transport/hash bindings so mutations test semantic validation."""
    a = directory / 'a5-receipt.json'
    change(directory / 'k1sr-receipt.json',
           lambda k: k.update(a5_receipt_sha256=m.sha(a.read_bytes())))
    k = json.loads((directory / 'k1sr-receipt.json').read_bytes())
    for name in ('k1sr-receipt.json', 'c85r7-receipt.json', 'a5-receipt.json'):
        attest(directory, name, k['kernel_id'])


@pytest.fixture
def a5_case(case):
    k1n(case)
    stage, d = case['stage'], case['visible_dir']
    contract = dict(schema='p25-a5-contract/v1', dataset='mattiaangeli/knee-mri-fold-weights',
                    formula='Pprime=rank(.85*rank(P)+.15*rank(A5))',
                    checkpoint_sha256=[str(i)*64 for i in range(5)])
    config = json.loads((stage / 'config.json').read_bytes())
    k1hash = m.sha(m.encoded(config))
    change(stage / 'config.json', lambda c: c.update(a5=contract))
    contract_hash = m.sha(m.encoded(contract))
    for p in (stage / 'registration.json', d / 'k1sr-receipt.json'):
        change(p, lambda o: o.update(
            formula=contract['formula'] + '; inherited outer K1: ' + o['formula'],
            a5_contract_sha256=contract_hash, registered_config_sha256=k1hash))
    change(stage / 'registration.json', lambda o: o.update(
        schema='p25-a5-registration/v1', a5_contract=contract))
    change(d / 'c85r7-receipt.json', lambda p: p.update(submission_sha256='f'*64))
    put(d / 'a5-receipt.json', dict(schema='p25-a5/v1', status='COMPLETE_A5',
        strict_load=True, studies=3, applicable=3, diagnostic=False, never_submit=False,
        dataset=contract['dataset'], formula=contract['formula'],
        checkpoint_hashes=contract['checkpoint_sha256'], original_p_sha256='f'*64,
        pprime_sha256=m.sha((d / 'k1sr-parent.csv').read_bytes()), seconds=27.3))
    change(d / 'k1sr-receipt.json', lambda k: k.update(parent_mode='P25_A5_PPRIME'))
    refresh(d)
    transport(d)
    nd = n32(case)
    change(nd / 'k1sr-receipt.json', lambda k: k.update(
        failed_study_count=0, failed_study_fraction=0., fail_loud_triggered=False,
        per_study=[dict(uid=str(i), ok=True, config_sha256=k1hash) for i in range(32)]))
    p = json.loads((d / 'c85r7-receipt.json').read_bytes())
    p.update(status='DIAGNOSTIC_C85R7', version='1.2.3-k1-n32', studies=32,
             never_submit=True, diagnostic=True, submission_sha256=None)
    put(nd / 'c85r7-receipt.json', p)
    a = json.loads((d / 'a5-receipt.json').read_bytes())
    a.update(studies=32, applicable=32, diagnostic=True, never_submit=True, seconds=56.8)
    put(nd / 'a5-receipt.json', a)
    refresh(nd)
    return case


def test_a5_real_wrapper_and_immutable_rerun(a5_case, capsys):
    out = m.prepare(**a5_case)
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out
    r = json.loads(first['ordinary-visible-registration.json'])
    assert r['tool_version'] == '1.0.5'
    assert r['projection'] is None  # execution evidence is not a runtime estimate
    for d in (a5_case['visible_dir'], a5_case['n32_dir']):
        assert str(d / 'a5-receipt.json') in r['input_sha256']
    m.prepare(**a5_case)
    assert first == {p.name: p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize('directory', ['visible_dir', 'n32_dir'])
@pytest.mark.parametrize('key,value', [
    ('schema', 'other'), ('status', 'FAILED_A5'), ('strict_load', False),
    ('strict_load', 1), ('applicable', 2), ('studies', 3.0),
    ('dataset', 'other/dataset'), ('formula', 'different formula'),
    ('checkpoint_hashes', ['a'*64]*5), ('original_p_sha256', 'bad'),
    ('pprime_sha256', 'b'*64), ('never_submit', None), ('diagnostic', None),
])
def test_a5_semantic_refusals(a5_case, directory, key, value):
    d = a5_case[directory]
    change(d / 'a5-receipt.json', lambda a: a.update({key: value}))
    refresh(d)
    with pytest.raises(m.Refuse):
        m.validate(**a5_case)


@pytest.mark.parametrize('directory', ['visible_dir', 'n32_dir'])
def test_a5_missing_receipt(a5_case, directory):
    # Rename only synthetic evidence; no deletion.
    p = a5_case[directory] / 'a5-receipt.json'
    p.rename(p.with_suffix('.preserved'))
    with pytest.raises(m.Refuse, match='missing input'):
        m.validate(**a5_case)


@pytest.mark.parametrize('directory', ['visible_dir', 'n32_dir'])
@pytest.mark.parametrize('key,value', [('kernel', 'other/kernel'), ('version', 2), ('sha256', 'f'*64)])
def test_a5_download_binding(a5_case, directory, key, value):
    change(a5_case[directory] / 'a5-receipt.json.download.json', lambda d: d.update({key: value}))
    with pytest.raises(m.Refuse):
        m.validate(**a5_case)


@pytest.mark.parametrize('key,value', [('drop_o', False), ('extra', 1), ('a5', {})])
def test_a5_config_tamper(a5_case, key, value):
    change(a5_case['stage'] / 'config.json', lambda c: c.update({key: value}))
    with pytest.raises(m.Refuse):
        m.validate(**a5_case)


@pytest.mark.parametrize('key,value', [('parent_mode', 'INF3'), ('a5_receipt_sha256', 'b'*64),
    ('config_sha256', 'b'*64), ('registered_config_sha256', 'b'*64),
    ('a5_contract_sha256', 'b'*64), ('submission_sha256', 'b'*64)])
def test_a5_k1_binding(a5_case, key, value):
    change(a5_case['visible_dir'] / 'k1sr-receipt.json', lambda k: k.update({key: value}))
    transport(a5_case['visible_dir'])
    with pytest.raises(m.Refuse):
        m.validate(**a5_case)


def test_a5_original_parent_binding(a5_case):
    change(a5_case['visible_dir'] / 'c85r7-receipt.json', lambda p: p.update(submission_sha256='a'*64))
    transport(a5_case['visible_dir'])
    with pytest.raises(m.Refuse, match='original P'):
        m.validate(**a5_case)


def test_a5_requires_n32(a5_case):
    a5_case.pop('n32_dir')
    with pytest.raises(m.Refuse, match='requires --n32-dir'):
        m.validate(**a5_case)


@pytest.mark.parametrize('key,value', [('k1_studies', 31), ('k1_executed', False),
    ('status', 'FAILED'), ('kernel_id', 'other/n32'), ('release_version', 'other-n32'),
    ('never_submit', False), ('failed_study_count', 1), ('fail_loud_triggered', True),
    ('formula', 'wrong'), ('model_sha256', {}), ('per_study', [])])
def test_a5_n32_binding(a5_case, key, value):
    change(a5_case['n32_dir'] / 'k1sr-receipt.json', lambda k: k.update({key: value}))
    refresh(a5_case['n32_dir'])
    with pytest.raises(m.Refuse):
        m.validate(**a5_case)


@pytest.mark.parametrize('ratio_case', [
    {'template': 'c5-dropo-p25-v9.template.json'},
    {'template': 'p25-dropo-v10.json'},
    {'template': 'c5-dropo-p25-v9.template.json', 'p_weight_mode': 'flat'},
], indirect=True)
def test_v9_v10_flat_admission_bytes_match_v104(ratio_case, monkeypatch):
    # The preserved pre-edit source is the oracle, not duplicated expectations.
    old = ModuleType('k1_admit_v104')
    backup = m.ROOT / '.local/backups/astra-b1/a5-admit-v1.0.4/k1_admit_v1.py'
    old.__file__ = str(m.ROOT / 'tools/k1_admit_v1.py')
    exec(compile(backup.read_bytes(), str(backup), 'exec'), old.__dict__)
    monkeypatch.setattr(old, 'TEMPLATE', m.TEMPLATE)
    monkeypatch.setattr(old, 'LEDGER', m.LEDGER)
    c = dict(ratio_case)
    config = json.loads((c['stage'] / 'config.json').read_bytes())
    if config.get('projection_basis') == 'n32_ratio':
        q.qualify(c['stage'], c['n32_dir'])
    else:
        c.pop('n32_dir')
    out = old.prepare(**c)
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    m.prepare(**c)
    assert first == {p.name: p.read_bytes() for p in out.iterdir()}
    assert old.validate(**c)[0].hashes == m.validate(**c)[0].hashes
    assert old.validate(**c)[0].checks == m.validate(**c)[0].checks
