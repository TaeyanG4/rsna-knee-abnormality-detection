"""Exact stage scope and authenticated Pprime integration; no launch or admission."""
from functools import lru_cache
import ast
import json
from types import ModuleType

import numpy as np
import pytest

from tools import p25_a5_stage_v1 as b
from test_k1n_v7 import records,replay


@lru_cache(None)
def artifacts(diagnostic=False): return b.artifacts(diagnostic)


@pytest.mark.parametrize('diagnostic',[False,True])
def test_stage_scope(diagnostic):
    outputs,reg=artifacts(diagnostic)
    scope=json.loads(outputs['payload-scope.json'])
    fields=json.loads(outputs['field-level-diff.json'])
    assert {f['field'] for f in fields}=={'a5','kernel_id','release_version'}
    assert len(scope['untouched'])==37 and scope['parent_artifacts_byte_identical']
    assert scope['changed']==sorted([b.RUNTIME,b.CODEC]) and scope['added']==b.MODULES
    assert scope['k1_functions_identical_except_finish']
    meta=json.loads(outputs['kernel/kernel-metadata.json'])
    old=json.loads((b.PARENT/'kernel/kernel-metadata.json').read_bytes())
    assert meta['dataset_sources']==old['dataset_sources']+[b.a5.DATASET]
    for k in old.keys()-{'id','title','dataset_sources'}: assert meta[k]==old[k]
    assert reg['never_submit']==diagnostic and not reg['qualified_for_submission']
    assert reg['schema']=='p25-a5-registration/v1'
    assert reg['kernel_id']==b.SLUG+('-n32' if diagnostic else '')
    cfg=json.loads(outputs['config.json'])
    parent=json.loads((b.PARENT/'config.json').read_bytes())
    for k in parent.keys()-{'kernel_id','release_version'}: assert cfg[k]==parent[k]
    assert cfg['alpha']==.25 and cfg['drop_o'] and cfg['k1_eval_stride']==2
    assert reg['a5_contract']['weight']==.15 and reg['a5_contract']['folds']==list(range(5))


def runtime():
    outputs,reg=artifacts()
    nb=json.loads(outputs['kernel/release.ipynb'])
    files=b.v10.replay.load().before.parent.archive(''.join(nb['cells'][2]['source']))[2]
    module=ModuleType('p25_a5_baked_runtime')
    exec(compile(files[b.RUNTIME],'<baked A5>','exec'),module.__dict__)
    return module,reg


@pytest.mark.parametrize('diagnostic',[False,True])
@pytest.mark.parametrize('fault',['none','missing','original_hash','pprime_hash','roster'])
def test_finish_uses_pprime_and_keeps_outer_math(tmp_path,diagnostic,fault):
    r,_=runtime()
    outputs,_=artifacts(diagnostic)
    nb=json.loads(outputs['kernel/release.ipynb'])
    tree=ast.parse(''.join(nb['cells'][2]['source']))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=='finish'
           and isinstance(n.func.value,ast.Name) and n.func.value.id=='_k1sr']
    assert len(calls)==1
    cfg=ast.literal_eval(calls[0].args[2])
    rng=np.random.default_rng(337)
    n=23
    parent=replay.csv_bytes(list(map(str,range(n))),rng.integers(0,9,(n,12))/8)
    raw=rng.integers(0,8,(5,n,12))/7
    pprime=b.a5.blend(parent,raw,np.ones(n,bool))
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(parent)
    arrays=[rng.integers(0,6,(n,12))/5 for _ in r.MODEL_IDS]
    rows=records(r,arrays)
    worker=tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    if fault=='roster':
        pprime=pprime.replace(b'\n0,',b'\nother,')
    receipt=dict(status='COMPLETE_A5',original_p_sha256=b.digest(parent),pprime_sha256=b.digest(pprime))
    if fault=='original_hash': receipt['original_p_sha256']='wrong'
    if fault=='pprime_hash': receipt['pprime_sha256']='wrong'
    (tmp_path/'a5-receipt.json').write_text(json.dumps(receipt))
    if fault!='none':
        with pytest.raises(ValueError):
            r.finish(tmp_path,tmp_path/'run',cfg,12.,parent_override=None if fault=='missing' else pprime)
        assert not (tmp_path/'k1sr-parent.csv').exists()
        return
    result=r.finish(tmp_path,tmp_path/'run',cfg,12.,parent_override=pprime)
    assert result['status']=='COMPLETE_SHARED_K1N'
    assert result['parent_mode']=='P25_A5_PPRIME'
    assert (tmp_path/'k1sr-parent.csv').read_bytes()==pprime
    target=tmp_path/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')
    assert target.read_bytes()==r.blend(pprime,rows)
    if diagnostic: assert not (tmp_path/'submission.csv').exists()


def test_admit_dry_run_stays_hold():
    result=b.run('admit',True)
    assert result['status']=='HOLD_NOT_ADMITTED' and not result['qualified_for_submission']
