"""H-B-156 L2 v2.0.0 identity/diff proof. Never print label rows."""
import copy
import json
from pathlib import Path
import pytest
from tools import hb156_package_v2 as pack
from tools import hb156_vast_v2 as guard


def test_exact_exp143_labels_and_parent_delta():
    exp=pack.registration_inputs()
    before=pack.read(pack.PARENT/pack.OLD_STAGE/'manifest.json')
    after=pack.read(pack.STAGE/'manifest.json')
    assert pack.sha(pack.STAGE/exp['label_file_name']) == pack.LABEL_SHA
    for key in ('train_ids','excluded_gold_ids','pos_weight','train_labels'):
        assert bool(after[key]==exp[key]), key
    for key in ('recipe','budgets','init_sha256','decision','train_ids','excluded_gold_ids','pos_weight'):
        assert bool(before[key]==after[key]), key
    assert bool(before['train_labels']!=after['train_labels'])
    # Real L2 is fractional. A thresholded/binarized reconstruction is forbidden.
    assert any(v not in (0.,1.) for row in after['train_labels'].values() for v in row)
    proof=pack.read(pack.WORK/'mechanical-diff.json')
    assert proof['status']=='PASS'
    for item in proof['unchanged']:
        dest=pack.PACKAGE/item['path'].replace(pack.OLD_STAGE,pack.STAGE_NAME)
        assert pack.sha(dest)==pack.sha(pack.PARENT/item['path'])==item['sha256'],item['path']
    assert (pack.PACKAGE/'training/src/hb156_vast_v1.py').read_bytes()==(pack.PARENT/'training/src/hb156_vast_v1.py').read_bytes()
    assert (pack.PACKAGE/'tools/hb156_admission_v1.py').read_bytes()==(pack.PARENT/'tools/hb156_admission_v1.py').read_bytes()


def test_soft_file_supplied_refused_before_build():
    with pytest.raises(ValueError,match='soft-label substitute refused'):
        pack.registration_inputs(pack.PARENT/pack.OLD_STAGE/'soft-labels.csv')


def test_runtime_rejects_soft_substitution(monkeypatch):
    monkeypatch.setattr(guard,'ROOT',pack.PACKAGE)
    monkeypatch.setattr(guard,'STAGE',pack.STAGE)
    monkeypatch.setattr(guard,'PROTOCOL',pack.STAGE/'manifest.json')
    original=guard.sha256
    soft=pack.PARENT/pack.OLD_STAGE/'soft-labels.csv'
    monkeypatch.setattr(guard,'sha256',lambda path:original(soft) if Path(path).name==pack.LABEL.name else original(path))
    with pytest.raises(ValueError,match='soft-label substitute refused'):
        guard.load_registration(guard.PROTOCOL)


@pytest.mark.parametrize('key',['train_ids','excluded_gold_ids','pos_weight','train_labels','budgets','init_sha256'])
def test_rehashed_manifest_mutation_refused(monkeypatch,key):
    monkeypatch.setattr(guard,'ROOT',pack.PACKAGE)
    monkeypatch.setattr(guard,'STAGE',pack.STAGE)
    monkeypatch.setattr(guard,'PROTOCOL',pack.STAGE/'manifest.json')
    original=guard.read_json
    p=original(guard.PROTOCOL)
    if key in ('train_ids','excluded_gold_ids'): p[key]=list(reversed(p[key]))
    elif key=='pos_weight': p[key][0]+=1
    elif key=='train_labels': p[key][p['train_ids'][0]][0]=-.01
    elif key=='budgets': p[key]['local']+=1
    else: p[key]='0'*64
    p.pop('content_sha256');p['content_sha256']=pack.json_hash(p)
    monkeypatch.setattr(guard,'read_json',lambda path:copy.deepcopy(p) if Path(path)==guard.PROTOCOL else original(path))
    with pytest.raises(ValueError,match='non-allowlisted'):
        guard.load_registration(guard.PROTOCOL)


def test_standalone_kill_registration():
    p=pack.read(pack.STAGE/'kill-test.json')
    assert p['baseline_auc']==.946 and p['alive_min']==.950 and p['dead_max']==.946
    assert p['labels_sha256']==pack.LABEL_SHA
    assert p['protocol_sha256']==pack.sha(pack.STAGE/'manifest.json')
    assert p['trigger']==dict(candidate='H-B-151',submission=57193756,auc=.944,dead_max=.947)


@pytest.mark.parametrize('mutation',['remove','replace'])
def test_rehashed_source_binding_mutation_refused(monkeypatch,mutation):
    monkeypatch.setattr(guard,'ROOT',pack.PACKAGE)
    monkeypatch.setattr(guard,'STAGE',pack.STAGE)
    monkeypatch.setattr(guard,'PROTOCOL',pack.STAGE/'manifest.json')
    original=guard.read_json
    p=original(guard.PROTOCOL)
    key='training/src/hb156_vast_v1.py'
    if mutation=='remove': del p['source_bindings'][key]
    else: p['source_bindings'][key]='0'*64
    p.pop('content_sha256');p['content_sha256']=pack.json_hash(p)
    monkeypatch.setattr(guard,'read_json',lambda path:copy.deepcopy(p) if Path(path)==guard.PROTOCOL else original(path))
    with pytest.raises(ValueError,match='source binding.*drift'):
        guard.load_registration(guard.PROTOCOL)


def test_rebuild_refuses_executed_stage(monkeypatch,tmp_path):
    monkeypatch.setattr(pack,'STAGE',tmp_path)
    (tmp_path/'budget.json').write_text('{}')
    with pytest.raises(ValueError,match='execution state'): pack.safe_build_target()
