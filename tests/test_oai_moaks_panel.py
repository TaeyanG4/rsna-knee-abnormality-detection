"""CPU-only panel contracts v1.0.3. Synthetic labels only; no report access.

Changelog 1.0.3: original vendor digests, private import isolation and tamper gates.
"""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import oai_moaks_panel as p


def test_vendor_matches_original_registration():
    p.validate_vendor()
    original = p.read(p.OUT / 'registration.json')
    assert original['version'] == '1.0.2'
    assert p.VERSION == '1.0.3'
    for path in p.SERVING_SOURCES:
        assert p.sha(p.OUT / 'vendor' / path) == original['pins'][path]


def test_private_imports_ignore_live_modules(monkeypatch):
    import sys
    poison = SimpleNamespace(build_model=lambda *a, **kw: pytest.fail('live source used'))
    monkeypatch.setitem(sys.modules, 'raptor_common', poison)
    monkeypatch.setitem(sys.modules, 'training.src.raptor_common', poison)
    isolated = p.Vendor(p.OUT / 'vendor')
    common = isolated.load('training.src.raptor_common')
    assert isolated.load('raptor_common') is common
    scales = isolated.load('training.src.raptor_exp179')
    mil = isolated.load('training.arms.mil')
    coatnet = isolated.load('training.arms.coatnet')
    assert scales.select_centers is common.select_centers
    assert mil.common is common
    assert coatnet.study_rng is common.study_rng
    assert 'vendor' in common.__file__
    assert sys.modules['raptor_common'] is poison
    assert sys.modules['training.src.raptor_common'] is poison
    with pytest.raises(ValueError, match='unvendored'):
        isolated.load('training.src.unregistered_dependency')


def test_vendor_tampering_fails_before_execution(tmp_path):
    path = tmp_path / 'tools/extlabels.py'
    path.parent.mkdir()
    path.write_text("raise AssertionError('tampered bytes executed')")
    p.write(tmp_path / 'manifest.json', p.vendor.manifest)
    with pytest.raises(ValueError, match='vendored source changed'):
        p.Vendor(tmp_path).load('tools.extlabels')


def test_validate_detects_archived_guard_tampering(tmp_path):
    import shutil
    for name in ('vendor',):
        shutil.copytree(p.OUT / name, tmp_path / name)
    for name in ('runtime-v1.0.3.json', 'registration.json'):
        shutil.copyfile(p.OUT / name, tmp_path / name)
    p.validate_vendor(tmp_path)
    with (tmp_path / 'vendor/tools/exp191_softprior.py').open('ab') as f:
        f.write(b'\n# changed\n')
    with pytest.raises(ValueError, match='vendored source changed'):
        p.validate_vendor(tmp_path)


def native():
    scores = dict(MACLTR=0, MEFFWK=0, MSYIC=0, MPOPCYS=0)
    scores.update({f'MMT{s}{r}': 0 for s in 'ML' for r in 'ABP'})
    scores.update({f'MMRT{s}': 0 for s in 'ML'})
    scores.update({'MOS'+r: 0 for rs in p.labels.OST_REGIONS.values() for r in rs})
    scores.update({'MCM'+r: 0 for rs in p.labels.CART_REGIONS.values() for r in rs})
    return scores


def test_complete_does_not_impute_missing_or_invalid():
    s = native()
    assert p.complete(s)
    for field in s:
        bad = dict(s); bad.pop(field)
        assert not p.complete(bad)
        bad[field] = 99
        assert not p.complete(bad)


@pytest.mark.parametrize('field,target,threshold', [
    ('MACLTR', 'ACL', 1), ('MMTLA', 'Lateral Meniscus', 2),
    ('MMTMP', 'Medial Meniscus', 2), ('MEFFWK', 'Effusion', 2),
    ('MSYIC', 'Synovitis', 2), ('MPOPCYS', "Baker's", 1)])
def test_mapping_thresholds(field, target, threshold):
    s = native(); s[field] = threshold-1
    assert p.labels.map_moaks(s)[target] == 0
    s[field] = threshold
    assert p.labels.map_moaks(s)[target] == 1
    assert all(p.labels.map_moaks(s)[t] is None for t in ('MCL', 'Contusion', 'Fracture'))


def test_root_tear_and_effusion_not_synovitis():
    s = native(); s.update(MMRTL=1, MEFFWK=3)
    r = p.labels.map_moaks(s)
    assert r['Lateral Meniscus'] == 1 and r['Effusion'] == 1 and r['Synovitis'] == 0


@pytest.mark.parametrize('target', ['Medial OA', 'Lateral OA', 'PF OA'])
def test_conservative_oa(target):
    s = native(); cart = 'MCM'+p.labels.CART_REGIONS[target][0]
    ost = 'MOS'+p.labels.OST_REGIONS[target][0]
    assert p.labels.map_moaks(s)[target] == 0
    s[cart] = 1.0
    assert p.labels.map_moaks(s)[target] is None
    s[cart] = 1.1
    assert p.labels.map_moaks(s)[target] is None
    s[ost] = 1
    assert p.labels.map_moaks(s)[target] == 1
    s[cart] = 1.2
    assert not p.complete(s)


def test_knee_selection_order_independent():
    ids = ['OAI_1_L_V00', 'OAI_1_R_V00', 'OAI_2_R_V00']
    expected = min(ids[:2], key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    assert p.select_knees(ids) == p.select_knees(ids[::-1]) == sorted([expected, ids[2]])


def test_cohort_requires_baseline_complete_and_all_slots(monkeypatch):
    ids = ['OAI_1_R_V00', 'OAI_2_R_V00', 'OAI_3_R_V00', 'OAI_4_R_V01']
    reads = {k: SimpleNamespace(scores=native()) for k in ids}
    reads[ids[1]].scores.pop('MACLTR')
    picks = {k: {'slots': [{'rendered': 1}]*5} for k in ids}
    picks[ids[2]]['slots'] = [None]*5
    monkeypatch.setattr(p, 'readings', lambda: reads)
    monkeypatch.setattr(p, 'cache_metadata', lambda: ({}, {'ids': ids}, picks))
    assert p.cohort()[0] == ids[:1]


def slots():
    result = []
    for plane, fluid, sequence, contrast in [
        ('Sagittal',1,'SAG_IW_TSE','IW_TSE_FS'), ('Sagittal',0,'SAG_IW_TSE','IW_TSE_FS'),
        ('Coronal',1,'COR_MPR','DESS_WE'), ('Coronal',0,'COR_IW_TSE','IW_TSE_NON_FS'),
        ('Axial',-1,'AX_MPR','DESS_WE')]:
        result.append(dict(slot=[plane,fluid],source_sequence=sequence,contrast_family=contrast,
                           group_key='group',series='series',rendered=12,fallback=None))
    result[1].update(fallback='copy_of_sagittal_fluid1',series='series#sag0_fallback_copy')
    return result


@pytest.mark.parametrize('mutation', ['plane', 'contrast', 'copy', 'fallback', 'missing'])
def test_routing_fails_closed(mutation):
    s = slots(); p.check_slots(s)
    if mutation == 'plane': s[2]['slot'] = ['Sagittal',1]
    if mutation == 'contrast': s[2]['contrast_family'] = 'IW_TSE_FS'
    if mutation == 'copy': s[1]['series'] = 'different#sag0_fallback_copy'
    if mutation == 'fallback': s[1]['fallback'] = None
    if mutation == 'missing': s[4] = None
    with pytest.raises(ValueError): p.check_slots(s)


@pytest.mark.parametrize('ancestry', [{'oai': True}, {'oai': None}, 'EXP155', 'EXP163',
                                    'EXP168', 'public 0.949 R', {'teacher': 'OAIZIB'}])
def test_reject_oai(ancestry):
    with pytest.raises(ValueError): p.deny_ancestry(ancestry)


def test_unknown_model_is_rejected_before_loading():
    with pytest.raises(ValueError, match='identity'): p.inspect_candidate('public_R')
    p.deny_ancestry({'oai': False, 'source': 'ImageNet'})


def test_mask_allows_background_but_rejects_missing_acquisition():
    spec={'depth':6,'slots':[['Sagittal',1,3],['Coronal',1,3]]}
    p.check_mask([1,1,1,0,1,1],spec)
    for mask in ([1,1,1,0,0,0], [1,1,1,2,1,1], [1,1,1]):
        with pytest.raises(ValueError): p.check_mask(mask,spec)


def test_auc_ties_missing_and_one_class():
    y = np.array([0,1,0,1,np.nan])
    assert p.auc(y, np.ones(5)) == .5
    assert p.auc(y, np.array([0,1,0,1,-100])) == 1
    assert np.isnan(p.auc(np.ones(5), np.arange(5)))


def test_paired_bootstrap_identical_seeds_and_missing_target():
    rng = np.random.default_rng(123)
    y = np.tile(np.arange(60)[:,None] % 2, (1,12)).astype(float)
    y[:,1] = np.nan
    a = rng.random((2,60,12))
    result = p.paired(y, a, a, draws=1000)
    assert result['weak_four']['ci95'] == [0.,0.]
    assert result['weak_four']['valid_draws'] == 1000
    assert result['per_target']['MCL']['ci95'] is None
    assert result['per_target']['MCL']['undefined_draws'] == 1000
    assert p.paired(y,a,a,draws=3) == p.paired(y,a,a,draws=3)


def test_seed_mean_is_not_auc_of_prediction_mean():
    y = np.tile(np.arange(60)[:,None] % 2, (1,12)).astype(float)
    control = np.full((2,60,12), .5)
    arms = np.stack([y, .49-.01*y])
    result = p.paired(y, arms, control, draws=8)
    assert result['weak_four']['delta'] == 0
    assert p.auc(y[:,0], arms.mean(0)[:,0]) == 1


def test_weak_primary_does_not_silently_drop_undefined_target():
    y = np.tile(np.arange(60)[:,None] % 2, (1,12)).astype(float)
    y[:,p.TARGETS.index('Synovitis')] = 0
    a = np.full((1,60,12), .5)
    assert p.paired(y,a,a,draws=8)['weak_four']['delta'] is None
    with pytest.raises(ValueError): p.paired(y,a,a[:,:,:11])


def test_pair_identity():
    assert p.valid_pair('EXP191_PA_S42','EXP185_S42') == ('EXP191_PA_S',42)
    with pytest.raises(ValueError): p.valid_pair('EXP191_PA_S42','EXP179_M42')
    with pytest.raises(ValueError): p.valid_pair('EXP191_PA_S42','EXP185_S43')


def test_prediction_hash_ids_and_partial_guard(tmp_path, monkeypatch):
    out = tmp_path
    for f in ('registration.json','PROTOCOL.md'): (out/f).write_text('{}')
    path = out/'pred.csv'; ids=['OAI_1_R_V00','OAI_2_R_V00']
    path.write_text('study_key,'+','.join(p.TARGETS)+'\n'+'\n'.join(k+','+','.join(['.5']*12) for k in ids)+'\n')
    receipt = dict(predictions_sha256=p.sha(path),registration_sha256=p.sha(out/'registration.json'),
                   protocol_sha256=p.sha(out/'PROTOCOL.md'),routing={'status':'PASS'},full_panel=False,key='test',recipe={})
    p.write(str(path)+'.json',receipt)
    monkeypatch.setattr(p,'inspect_candidate',lambda k: {})
    assert p.prediction(path,ids,out,smoke=True)[0].shape == (2,12)
    with pytest.raises(ValueError, match='partial'): p.prediction(path,ids,out)
    with pytest.raises(ValueError, match='IDs'): p.prediction(path,ids[::-1],out,smoke=True)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='changed'): p.prediction(path,ids,out,smoke=True)


@pytest.mark.parametrize('family',['global','mil','coatnet'])
def test_serving_matches_unchunked_original(family):
    import torch
    from torch import nn
    from training.arms import mil, coatnet
    from training.src import raptor_common as common
    from training.src.raptor_exp179 import scale_windows
    from training.src.raptor_radimagenet import normalise
    torch.set_num_threads(2); torch.manual_seed(123)
    class Backbone(nn.Module):
        num_features=4
        def __init__(self):
            super().__init__(); self.proj=nn.Conv2d(3,4,1)
            self.head=SimpleNamespace(norm=nn.Identity())
        def forward_features(self,x):
            return self.proj(torch.nn.functional.adaptive_avg_pool2d(x, (2,2)))
        def forward(self,x): return self.forward_features(x).mean((2,3))
    backbone=Backbone()
    model = (mil.FindingMIL(backbone) if family=='mil' else coatnet.RaptorClassifier(backbone)).eval()
    depth, size = (96,384) if family=='coatnet' else (64,336)
    vol = np.random.default_rng(1).integers(0,256,(depth,size,size),dtype=np.uint8)
    mask = np.ones(depth,np.uint8)
    cache=SimpleNamespace(get=lambda uid:(vol,mask), mask=lambda uid:mask, spec={'slots':common.SLOTS64})
    actual=p.predict_one(model,{'family':family},cache,'synthetic')
    with torch.inference_mode():
        if family=='coatnet':
            a,b=coatnet.infer_pair(model,coatnet.Config(),vol,mask,chunk=94)
            expected=(a+b)*np.float32(.5)
        else:
            centers=common.select_centers(mask,64,36,cache.spec,'within-slot-valid')
            x=torch.from_numpy(normalise(scale_windows(vol,centers,224,'mixed'),'imagenet'))[None]
            expected=(model(x).sigmoid()+model(mil.flip_inputs(x,['synthetic'],cache,36)).sigmoid())[0].numpy()*.5
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
