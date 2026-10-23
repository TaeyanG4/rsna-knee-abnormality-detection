"""Focused changed-path checks; no dataset, model loading or GPU."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import exp086_dinov2s_v2 as g
from training.adapters import dinov2_cpu_pos_v2 as adapter


def test_renderer_exact_and_v1_is_separate():
    original=(ROOT/'training/src/raptor_train.py').read_text(encoding='utf-8')
    rendered=g.render_training_source(original)
    assert rendered==(ROOT/g.RUNNER).read_text(encoding='utf-8')
    ast.parse(rendered)
    assert 'from tools import exp086_dinov2s_v2 as guard' in rendered
    assert 'position_policy=guard.POSITION_POLICY' in rendered
    assert g.PROTOCOL.endswith('_V2_PROTOCOL') and g.TAG.endswith('_V2')
    old=g.read(ROOT/g.V1_PROTOCOL)
    assert all(g.sha256(ROOT/n)==h for n,h in old['source_bindings'].items())


@pytest.mark.parametrize('bad',['threads','determinism','warn_only','dtype'])
def test_changed_operation_rejects_unregistered_policy_before_interpolation(bad):
    import torch
    threads=torch.get_num_threads();strict=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        p=torch.zeros(1,17,2,dtype=torch.float32)
        if bad=='threads':torch.set_num_threads(2)
        if bad=='determinism':torch.use_deterministic_algorithms(False)
        if bad=='warn_only':torch.use_deterministic_algorithms(True,warn_only=True)
        if bad=='dtype':p=p.double()
        with pytest.raises(ValueError):adapter.resample_cpu(p,(2,2),(4,4),1)
    finally:
        torch.set_num_threads(threads);torch.use_deterministic_algorithms(strict,warn_only=warn)


def test_install_preserves_parameter_rng_and_other_instances():
    import torch
    class Fake:
        def _pos_embed(self,x):return x
    one=Fake();other=Fake();old_class=Fake._pos_embed
    one.pos_embed=torch.nn.Parameter(torch.zeros(1,1370,384))
    one.dynamic_img_size=True;one.patch_embed=SimpleNamespace(grid_size=(37,37))
    one.num_prefix_tokens=1;one.no_embed_class=False
    param=one.pos_embed;rng=torch.get_rng_state().clone()
    adapter.install(one)
    assert one.pos_embed is param and param.requires_grad and torch.equal(rng,torch.get_rng_state())
    assert Fake._pos_embed is old_class and other._pos_embed.__func__ is old_class
    assert one._pos_embed.__func__ is adapter.position_embed_cpu
    with pytest.raises(ValueError):adapter.install(one)


def pilot(steady=1.0):
    times=[steady]*32;actual=sum(times[8:])/24
    return dict(schema=g.SCHEMA,status='PASS',phase='pilot',updates=32,skips=0,training_ids_only=True,
        validation_rows=0,position_policy=g.POSITION_POLICY,step_seconds=times,steady_update_seconds=actual,
        projected_full_seconds=1.3*(actual*3448)+300)


@pytest.mark.parametrize('bad',[None,'old_limit','false_times','position','missing_times'])
def test_v2_cost_recomputed_and_5300_cap(bad):
    p=pilot()
    if bad=='old_limit':p=pilot((5350-300)/(1.3*3448))
    if bad=='false_times':p['steady_update_seconds']=.99
    if bad=='position':p['position_policy']='gpu'
    if bad=='missing_times':p['step_seconds']=[]
    if bad:
        with pytest.raises(ValueError):g.pilot_cost_gate(p,100)
    else:g.pilot_cost_gate(p,100)


@pytest.mark.parametrize('bad',[None,'timeout','access'])
def test_v2_resource_additional_explicit_fields(bad):
    r=dict(status='COMPLETE',child_exit_code=0,wrapper_exit_code=0,child_exit_code_before_cleanup=0,failure=None,
        argv=['worker'],retries=0,supervisor_sha256=g.RESOURCE_SHA,child_identity={'pid':123},observed_identities=[{'pid':123}],
        sample_count=1,sampled_tree_rss_max_bytes=100,access_failure_count=0,failed_sample_attempts=0,
        cleanup=dict(actions=[],errors=[],survivors=[]),elapsed_seconds=100,timeout_seconds=180,access_failures=[])
    if bad=='timeout':r['timeout_seconds']=190
    if bad=='access':r['access_failures']=['error']
    if bad:
        with pytest.raises(ValueError):g.resource_gate(r,['worker'],180)
    else:g.resource_gate(r,['worker'],180)


def test_adapter_has_no_detach_or_global_class_write():
    source=(ROOT/g.ADAPTER).read_text(encoding='utf-8');tree=ast.parse(source)
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'detach','numpy','no_grad','requires_grad_'} for n in ast.walk(tree))
    assert 'dtype=posemb.dtype' in source and "device=posemb.device" in source
    assert 'VisionTransformer._pos_embed =' not in source
