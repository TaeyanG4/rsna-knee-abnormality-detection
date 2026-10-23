"""v1.0.0: three-member fullbag replacement, filler and generated CPU contracts.

Changelog v1.0.0: replace seed0 only; retain original four-member templates.
"""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from tools import k1n_stage_v1 as b
from test_fullbag_post_v1 import FILL, fullbag_assets, load
from test_k1n_members_v1 import filled, runtime
from test_k1n_v7 import records, replay

CONFIGS = b.ROOT/'configs/k1n'

def read(path):
    return json.loads(path.read_bytes())

def template(flat):
    return CONFIGS/('p25-fbr-dropo'+('-flatp' if flat else '')+'-v10.template.json')

@pytest.mark.parametrize('flat', [False, True])
def test_replaces_seed0_only_and_preserves_equal_parent_weights(flat):
    c = read(template(flat))
    parent = read(CONFIGS/('p25-dropo'+('-flatp' if flat else '')+'-v10.json'))
    assert [m['id'] for m in c['models']] == ['k1_v2','k1_v3g_fullbag','k1_v3g_seed1']
    assert 'member_weights' not in c and 'member_weights' not in parent
    assert c['models'][0] == parent['models'][0] and c['models'][2] == parent['models'][2]
    old = read(CONFIGS/('p25-fb-dropo'+('-flatp' if flat else '')+'-v10.template.json'))
    assert len(old['models']) == 4 and old['member_weights'] == [.25]*4
    assert c['models'][1] == old['models'][3]
    assert {k for k in c.keys() | parent.keys() if c.get(k) != parent.get(k)} == {
        'models','kernel_id','release_version'}
    assert c['k1_eval_stride'] == 2 and c['alpha'] == .25
    with pytest.raises(ValueError):
        b.artifacts(c)

def test_flat_twin_only_changes_p_mode_and_identity():
    a, z = read(template(False)), read(template(True))
    assert {k for k in a.keys() | z.keys() if a.get(k) != z.get(k)} == {
        'p_weight_mode','kernel_id','release_version'}
    assert z['p_weight_mode'] == 'flat'

@pytest.mark.parametrize('flat', [False, True])
def test_verified_synthetic_fullbag_fill_prepare_validate(tmp_path, monkeypatch, flat):
    _, dataset = fullbag_assets(tmp_path/'fixture')
    manifest = read(dataset/'asset-manifest.json')
    assert manifest['model_seed'] == 0 and manifest['model_fold'] == 0
    target = tmp_path/'filled.json'
    result = subprocess.run([sys.executable,'-B',str(FILL),str(template(flat)),str(target),
        'taeyangg4/cpu-p25fbr-fixture-v1','1.1.90-cpu-p25fbr-fixture',
        'k1_v3g_fullbag='+str(dataset)],cwd=b.ROOT,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    c = read(target)
    assert 'REPLACE' not in target.read_text() and 'replace-' not in target.read_text()
    assert c['models'][1]['id'] == 'k1_v3g_fullbag'
    assert c['models'][1]['checkpoint'] == 'weights/E_prior_s0_f0.pt'
    monkeypatch.setattr(b,'stage',lambda c,d=False:tmp_path/('stage-n32' if d else 'stage'))
    assert b.run('prepare',c)['status'] == 'PREPARED_LOCAL_ONLY'
    assert b.run('validate',c)['status'] == 'PASS_CPU_VALIDATE'
    parent = read(CONFIGS/('p25-dropo'+('-flatp' if flat else '')+'-v10.json'))
    expected_sources = set(b.package(parent)[1]['dataset_sources'])
    expected_sources.remove(parent['models'][1]['dataset_slug'])
    expected_sources.add(c['models'][1]['dataset_slug'])
    for diagnostic in (False,True):
        _, meta, _, _, _, _ = b.package(c,diagnostic)
        assert set(meta['dataset_sources']) == expected_sources
        reg = read(b.stage(c,diagnostic)/'registration.json')
        assert reg['models'] == c['models'] and not reg['qualified_for_submission']
        assert reg['never_submit'] is diagnostic
        assert 'member_weights' not in reg
        assert reg['capture_timeout_seconds'] == 900 and reg['call_timeout_seconds'] == 300

@pytest.mark.parametrize('flat', [False, True])
def test_replacement_runtime_is_exact_equal_three_member_blend(flat):
    c = filled(template(flat))
    parent = read(CONFIGS/('p25-dropo'+('-flatp' if flat else '')+'-v10.json'))
    old, new = runtime(parent), runtime(c)
    assert new.MODEL_IDS == ('k1_v2','k1_v3g_fullbag','k1_v3g_seed1')
    rng = np.random.default_rng(1021)
    p, *arrays = [rng.integers(0,9,(32,12)).astype(np.float32)/10 for _ in range(4)]
    payload = replay.csv_bytes(list(map(str,range(32))),p)
    assert new.blend(payload,records(new,arrays)) == old.blend(payload,records(old,arrays))

@pytest.mark.parametrize('fault', ['seed','fold','variant','id'])
def test_replacement_filler_rejects_wrong_identity(tmp_path, monkeypatch, fault):
    fill = load('fbr_filler', FILL)
    assert fill.FOLD['k1_v3g_fullbag'] == 0
    manifest = dict(model_seed=0,model_fold=0,variant='fullbag')
    if fault in ('seed','fold'):
        manifest['model_'+fault] = 1
    elif fault == 'variant':
        manifest['variant'] = 'standard'
    (tmp_path/'asset-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(fill,'verify',lambda _:dict(model=dict(
        id='wrong' if fault == 'id' else 'k1_v3g_fullbag')))
    monkeypatch.setattr(sys,'argv',['fill',str(template(False)),str(tmp_path/'out.json'),
        'owner/test','1.1.90-test','k1_v3g_fullbag='+str(tmp_path)])
    with pytest.raises(SystemExit):
        fill.main()
    assert not (tmp_path/'out.json').exists()
