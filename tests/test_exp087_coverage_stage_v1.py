"""Synthetic adapter and saved pilot only; no new archive/header/model access."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import random
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'training/src'))
os.environ['CUDA_VISIBLE_DEVICES']=''
from tools import exp087_coverage_stage_v1 as stage
from tools import exp087_series_coverage as policy
from exp087_coverage_data import CoverageStudyWindows
from raptor_data import StudyWindows,collate


def synthetic(train,reason=None):
    rc,_=policy.helpers(); uid='synthetic'
    vol=np.arange(64*8*8,dtype=np.uint8).reshape(64,8,8); mask=np.ones(64,np.uint8); mask[20]=0
    alt=np.flip(vol[32:44],axis=1).copy()
    class Corpus:
        depth=64
        spec={'slots':rc.SLOTS64}
        def volume(self,u):return vol
        def mask(self,u):return mask
    b=StudyWindows(str(ROOT/'synthetic-not-opened'),[uid],{uid:np.arange(12,dtype=np.float32)},224,12 if train else 24,train,'imagenet',42,window_policy='within-slot-valid')
    b.corpus=Corpus()
    class Store:
        membership={'train':[uid],'val':[uid]}
        baseline_corpus=(ROOT/'synthetic-not-opened').resolve()
        def entry(self,u,split):return None if reason=='absent' else dict(uid=u,reason=reason,mask=[1]*12,native_indices=list(range(12)))
        def volume(self,row):return alt
    return b,Store(),vol,mask,alt


@pytest.mark.parametrize('train,epoch',[(False,0),(True,0),(True,3),(True,7)])
def test_wrapper_matches_frozen_policy_and_preserves_original_labels_rng(train,epoch):
    import torch
    b,store,vol,mask,alt=synthetic(train); b.set_epoch(epoch)
    adapter=CoverageStudyWindows(b,store)
    rng=(random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
    actual,label,uid=adapter[0]
    expected,trace=policy.windows(vol,mask,b.corpus.spec,uid,train=train,epoch=epoch,alternate=alt,alt_mask=[1]*12,native_indices=list(range(12)))
    assert np.array_equal(actual.numpy(),expected)
    assert np.array_equal(label.numpy(),b.labels[uid])
    assert random.getstate()==rng[0] and np.array_equal(np.random.get_state()[1],rng[1][1]) and torch.equal(torch.random.get_rng_state(),rng[2])
    reference=b[0][0]
    forced=CoverageStudyWindows(b,store,force_original=True)[0][0]
    assert torch.equal(reference,forced)
    assert torch.equal(reference,CoverageStudyWindows(b,store,enabled=False)[0][0])
    changed={int(i) for i in trace['replacements']}
    assert all(torch.equal(reference[i],actual[i]) for i in set(range(len(actual)))-changed)


@pytest.mark.parametrize('reason',['absent','geometry_ineligible','shared_SOP','sampled_render_duplicate','no_valid_alt_triplet'])
def test_expected_fallback_keeps_original_exactly(reason):
    import torch
    b,store,*_=synthetic(False,reason)
    assert torch.equal(b[0][0],CoverageStudyWindows(b,store)[0][0])


def test_wrong_shared_recipe_and_wrong_G1_order_rejected():
    b,store,*_=synthetic(True); b.seed=43
    with pytest.raises(ValueError,match='contract'):CoverageStudyWindows(b,store)
    b.seed=42; store.membership={'train':['different'],'val':['synthetic']}
    with pytest.raises(ValueError,match='ordering'):CoverageStudyWindows(b,store)
    store.membership={'train':['synthetic'],'val':['synthetic']};b.corpus_dir='another-corpus'
    with pytest.raises(ValueError,match='corpus path'):CoverageStudyWindows(b,store)


def test_eval_batch4_K24_original_collation_unchanged():
    b,store,*_=synthetic(False)
    wrapper=CoverageStudyWindows(b,store,force_original=True)
    first=collate([b[0] for _ in range(4)]); second=collate([wrapper[0] for _ in range(4)])
    assert first[0].shape==second[0].shape==(4,24,3,224,224)
    assert np.array_equal(first[0].numpy(),second[0].numpy()) and first[2]==second[2]


def test_scale_supervisor_is_exact_frozen_v3_body_and_caps_not_policy_changed():
    original=(ROOT/'tools/exp087_series_coverage.py').read_text(encoding='utf-8')
    actual=(ROOT/'tools/exp087_coverage_stage_v1.py').read_text(encoding='utf-8')
    def body(text):return text[text.index('def run(plan_path, expected_sha, execute_pixels):'):text.index('\ndef main():')].strip()
    assert body(original)==body(actual)
    assert policy.sha(ROOT/'tools/exp087_series_coverage.py')==stage.HELPER_SHA
    assert stage.CAPS['pixel_reads']==307*12 and stage.CAPS['rendered_bytes']==307*12*336*336
    assert stage.CAPS['wall_seconds']==900 and stage.CAPS['rss_bytes']==8*1024**3


def test_stage_reuses_exact_saved8_inputs_only():
    # Authorized stored artifact verification, not raw DICOM or cache pixel access.
    fixture=policy.read_json(ROOT/stage.PE/'freeze-v3/fixture.json')
    result=policy.read_json(ROOT/stage.PILOT/'receipt.json')
    assert [p['uid'] for p in fixture['pairs']]==[p['uid'] for p in result['pairs']]
    assert len(result['pairs'])==8
    for p in result['pairs']:
        assert p['original_replay_exact'] and len(p['checks'])==9
        for role in ('original','alternate'):
            arr=np.load(ROOT/stage.PILOT/(p['uid']+'-'+role+'.npy'),allow_pickle=False)
            assert arr.shape==(12,336,336) and arr.dtype==np.uint8
            assert policy.array_hash(arr)==p[role]['volume_hash']
        assert all(c['noop_exact'] and c['RNG_exact'] for c in p['checks'])


def test_new_budget_fails_closed():
    budget=policy.Budget(stage.CAPS)
    budget.add('pixel_reads',3684)
    with pytest.raises(RuntimeError,match='pixel_reads cap'):budget.add('pixel_reads',1)


def test_new_cross_G1_SOP_evidence_requires_ancestor_review():
    owners={}
    stage.register_sops(owners,'T1','train',['sop1'])
    stage.register_sops(owners,'T1','train',['sop1'])
    with pytest.raises(ValueError,match='ancestor review'):
        stage.register_sops(owners,'V1','val',['sop1'])


def test_store_pickling_has_no_module_or_parent_array():
    import pickle
    from exp087_coverage_data import CoverageStore
    store=CoverageStore.__new__(CoverageStore)
    store.entries={};store.membership={};store.baseline_corpus=Path('synthetic')
    store._cached_uid='synthetic';store._cached=np.ones((12,2,2),np.uint8)
    restored=pickle.loads(pickle.dumps(store))
    assert restored._cached_uid is None and restored._cached is None
    assert restored.entries=={} and restored.baseline_corpus==Path('synthetic')
