"""Only path correction and immutable-contract regression for failedtrialv1."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import d01_integrated_trial_v2 as trial


def mount(path,raw=b'fixed train metadata'):
    path.mkdir(parents=True)
    for name in ('train_series.csv','test.csv','test_series.csv','sample_submission.csv'):(path/name).write_bytes(raw)
    for name in ('train_series','test_series'):(path/name).mkdir()
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('layout',['flat','nested'])
def test_exact_hash_flat_and_nested_root(tmp_path,layout):
    flat=tmp_path/'rsna-knee-abnormality-detection';nested=tmp_path/'competitions/rsna-knee-abnormality-detection'
    selected=flat if layout=='flat' else nested;digest=mount(selected)
    assert trial.resolve_competition(digest,[nested,flat])==selected
    assert trial.resolve_competition(digest,[selected,selected])==selected


@pytest.mark.parametrize('case',['absent','wrong_hash','two_complete','incomplete'])
def test_fail_closed_mount_selection(tmp_path,case):
    a,b=tmp_path/'a',tmp_path/'b';digest=hashlib.sha256(b'fixed train metadata').hexdigest()
    if case!='absent':mount(a)
    if case=='wrong_hash':digest='0'*64
    if case=='two_complete':mount(b)
    if case=='incomplete':(a/'train_series.csv').unlink()
    with pytest.raises(ValueError):trial.resolve_competition(digest,[a,b])


def original_one_direct():
    original=trial.read(ROOT/'kaggle/packages/KNEE_ANCHOR_0943/rsna-knee-fast-2xt4-inference.ipynb')
    tree=ast.parse(''.join(original['cells'][6]['source']))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_one_direct')
    namespace={'Path':Path};exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged_original_function>','exec'),namespace)
    return namespace['_one_direct']


def test_original_anchor_proxy_resolution_flat_mask_is_narrow_and_restored(tmp_path):
    flat,proxy=tmp_path/'flat',tmp_path/'proxy';mount(flat);mount(proxy)
    select=original_one_direct();required=['test.csv','test_series.csv','test_series']
    with pytest.raises(FileNotFoundError):select('competition',[proxy,flat],required)
    before=(flat/'test.csv').read_bytes();original_exists=Path.exists
    restore=trial.install_proxy_probe_mask(proxy,flat,flat)
    try:
        assert select('competition',[proxy,flat],required)==proxy
        assert not (flat/'test.csv').exists()
        assert (flat/'test.csv').is_file() and (flat/'test.csv').read_bytes()==before
        assert (flat/'train_series.csv').exists() and (flat/'test_series.csv').exists()
    finally:restore()
    assert Path.exists is original_exists and (flat/'test.csv').exists()


def test_nested_mount_does_not_change_exists(tmp_path):
    flat,nested,proxy=tmp_path/'flat',tmp_path/'nested',tmp_path/'proxy';mount(nested);mount(proxy)
    original=Path.exists;restore=trial.install_proxy_probe_mask(proxy,nested,flat)
    assert Path.exists is original;restore();assert Path.exists is original


def test_original_wrapper_and_nonpath_functions_remain_frozen():
    before=ROOT/'tools/d01_integrated_trial.py'
    assert trial.sha(before)=='599e7d08d208586ad2b306b25a9a4bcb6c946592377ec94ca619ac7915ba4135'
    left={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(before.read_text('utf-8')).body if isinstance(n,ast.FunctionDef)}
    right={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(trial.__file__).read_text('utf-8')).body if isinstance(n,ast.FunctionDef)}
    for name in ('projection','verify_r2','cgroup','snapshot','resource_gate','reap_owned','run_phase','worker'):
        assert left[name]==right[name],name


def test_original30cells_and_133_protocol_unchanged():
    previous=ROOT/'.local/d01-integrated-trial/20261003-v2/payload'
    protocol=trial.read(previous/'trial-protocol.json');original=trial.read(previous/'anchor-original.ipynb')
    notebook=trial.anchor_notebook(original,protocol)
    assert notebook['cells'][1:-1]==original['cells'] and len(notebook['cells'])==32
    prefix=''.join(notebook['cells'][0]['source'])
    assert 'Fat_Suppression' in prefix
    assert "_trial_meta = _trial_meta[[" not in prefix
    assert trial.sha(previous/'trial-protocol.json')=='5ef7bc3383ed8628c62b0ba9405b7d5e73cf4b30932e90d8013e6a06a70de0ea'
    assert trial.sha(previous/'helper/d01_release_v3.py')==trial.V3_SHA
    assert protocol['session_seconds']==5400 and protocol['internal_deadline_seconds']==5100
    assert len(protocol['ids'])==133
