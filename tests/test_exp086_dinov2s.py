"""Focused CPU contract tests. No real images, model construction or CUDA."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import exp086_dinov2s as g


def test_renderer_matches_explicit_runner_and_preserves_real_optimizer_math():
    original=(ROOT/'training/src/raptor_train.py').read_text(encoding='utf-8')
    rendered=g.render_training_source(original)
    assert rendered==(ROOT/g.RUNNER).read_text(encoding='utf-8')
    tree=ast.parse(rendered)
    assert not any(isinstance(n,ast.Name) and n.id in {'gold_labels','gold_ids','macro_auc'} for n in ast.walk(tree))
    assert 'strict=False' not in rendered and 'pretrained=True' not in rendered
    assert 'guard.build_offline_model' in rendered and 'guard.finish_pilot' in rendered
    assert 'same_training_schedule_in_pilot' not in rendered  # Schedule uses the actual full steps.
    for fragment in [
        'loss = lossf(model(x).float(), y) / group_size',
        'scaler.scale(loss).backward()',
        'scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)',
        'scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)',
        'torch.optim.AdamW', 'total_steps=steps, pct_start=0.15',
        'seed_all(a.seed)', 'torch.Generator().manual_seed(a.seed)',
        'tds.set_epoch(ep)', 'for ep in range(a.epochs)']:
        assert fragment in original and fragment in rendered
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='ep')
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='evaluate']
    assert len(calls)==1 and ast.unparse(calls[0].args[0])=='train_ids[:4]'
    assert rendered.index('guard.finish_pilot(')<rendered.index('torch.save(best,')
    assert 'if a.phase == "pilot" and optimizer_updates == 32:' in rendered


@pytest.mark.parametrize('which',['helper','runner'])
def test_passive_import_never_imports_torch_or_common(tmp_path,which):
    path=ROOT/('tools/exp086_dinov2s.py' if which=='helper' else g.RUNNER)
    script="import importlib.util,sys; s=importlib.util.spec_from_file_location('passive',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;assert 'raptor_common' not in sys.modules"
    p=subprocess.run([sys.executable,'-c',script,str(path)],cwd=tmp_path,capture_output=True,text=True,
                     env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    assert p.returncode==0,p.stderr
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('path',['../escape','workspace://../escape','C:/escape','/escape'])
def test_relative_path_escape_rejected(tmp_path,path):
    with pytest.raises(ValueError):g.contained(tmp_path,path)


def protocol(tmp_path):
    ids={'train':[f't{i}' for i in range(3455)],'val':[f'v{i}' for i in range(888)]}
    (tmp_path/'ids.json').write_text(json.dumps(ids))
    p={'schema':g.SCHEMA,'arm':{'expected_args':{'arch':g.ARCH}},'source_bindings':{},'training_code_sha256':{},
       'ordered_ids':{'path':'ids.json',**{k+'_sha256':g.uid_hash(v) for k,v in ids.items()}}}
    (tmp_path/'p.json').write_text(json.dumps(p));return p,ids,tmp_path/'p.json'


@pytest.mark.parametrize('change',[None,'schema','arch','order','uid_overlap','source','input'])
def test_frozen_protocol_uid_source_input_binding(tmp_path,change):
    p,ids,path=protocol(tmp_path)
    if change=='schema':p['schema']='old'
    if change=='arch':p['arm']['expected_args']['arch']='competition-checkpoint'
    if change=='order':ids['val'].reverse()
    if change=='uid_overlap':
        ids['val'][0]=ids['train'][0];p['ordered_ids']['val_sha256']=g.uid_hash(ids['val'])
    if change=='source':p['source_bindings']={'ids.json':'wrong'}
    if change=='input':p['input_bindings']={'ids.json':'wrong'}
    (tmp_path/'ids.json').write_text(json.dumps(ids));path.write_text(json.dumps(p))
    if change:
        with pytest.raises(ValueError):g.validate_protocol(path,tmp_path,change=='input')
    else:assert g.validate_protocol(path,tmp_path)==(p,ids)


def pilot():
    return {'status':'PASS','phase':'pilot','updates':32,'skips':0,'training_ids_only':True,'validation_rows':0,
            'steady_update_seconds':1.0,'projected_full_seconds':1.3*3448+300}


@pytest.mark.parametrize('bad',[None,'skip','updates','validation','late','slow','fake_projection','nan'])
def test_pilot_gate_uses_measured_steady_cost_and_actual_elapsed(bad):
    p=pilot();elapsed=80.
    if bad=='skip':p['skips']=1
    if bad=='updates':p['updates']=31
    if bad=='validation':p['validation_rows']=4
    if bad=='late':elapsed=181
    if bad=='slow':p['steady_update_seconds']=2.;p['projected_full_seconds']=1.3*2*3448+300
    if bad=='fake_projection':p['projected_full_seconds']=1.
    if bad=='nan':p['steady_update_seconds']=float('nan')
    if bad:
        with pytest.raises(ValueError):g.pilot_cost_gate(p,elapsed)
    else:g.pilot_cost_gate(p,elapsed)


def test_fractional_speed_uses_exact_registered_parentheses():
    p=pilot();p['steady_update_seconds']=0.987654321;p['projected_full_seconds']=1.3*(p['steady_update_seconds']*3448)+300
    g.pilot_cost_gate(p,90.)


def resource():
    return {'status':'COMPLETE','child_exit_code':0,'wrapper_exit_code':0,'child_exit_code_before_cleanup':0,'failure':None,
       'argv':['worker'],'retries':0,'supervisor_sha256':g.RESOURCE_SHA,'sample_count':3,'sampled_tree_rss_max_bytes':1000,
       'access_failure_count':0,'failed_sample_attempts':0,'cleanup':{'actions':[],'errors':[],'survivors':[]},
       'elapsed_seconds':80.,'child_identity':{'pid':123},'observed_identities':[{'pid':123}]}


@pytest.mark.parametrize('bad',[None,'helper','argv','pid','late','survivors','cleanup','access','exit0_after_cleanup'])
def test_resource_natural_completion_required(bad):
    r=resource()
    if bad=='helper':r['supervisor_sha256']='other'
    if bad=='argv':r['argv']=['other']
    if bad=='pid':r['observed_identities']=[]
    if bad=='late':r['elapsed_seconds']=180.
    if bad=='survivors':r['cleanup']['survivors']=[123]
    if bad=='cleanup':r['cleanup']['actions']=[{'kill':123}]
    if bad=='access':r['access_failure_count']=1
    if bad=='exit0_after_cleanup':r['child_exit_code_before_cleanup']=None
    if bad:
        with pytest.raises(ValueError):g.resource_gate(r,['worker'],180)
    else:g.resource_gate(r,['worker'],180)


def test_phase_launch_matches_one_candidate_and_same_full_schedule(tmp_path):
    p={'arm':{'expected_args':{'arch':g.ARCH,'epochs':8,'fixed_epoch':8,'bs':8,'accum':1,'workers':12,'grad_ckpt':True,'smoke':False}},'paths':{'corpus':'cache'}}
    pilot=g.phase_command(p,tmp_path/'manifest.json','pilot',tmp_path)
    full=g.phase_command(p,tmp_path/'manifest.json','full',tmp_path)
    for argv in (pilot,full):
        assert argv[argv.index('--epochs')+1]=='8' and argv[argv.index('--fixed-epoch')+1]=='8'
        assert '--grad-ckpt' in argv and '--smoke' not in argv
    assert pilot[pilot.index('--tag')+1]==g.TAG+'_PILOT' and full[full.index('--tag')+1]==g.TAG


def test_no_gpu_without_exact_root_decision(tmp_path,monkeypatch):
    monkeypatch.setattr(g,'validate_protocol',lambda *a:({},{}))
    (tmp_path/'p.json').write_text('{}');(tmp_path/'decision.json').write_text('{}')
    with pytest.raises(ValueError,match='root phase'):
        g.run(tmp_path/'p.json','pilot',tmp_path/'decision.json',root=tmp_path)


def test_offline_initial_rejects_asset_before_model_import(monkeypatch):
    monkeypatch.setattr(g,'sha256',lambda p:'wrong')
    with pytest.raises(ValueError,match='initial weight'):
        g.build_offline_model(g.ARCH,{'path':'absent','sha256':g.WEIGHT_SHA})


def test_existing_exp084_input_sampler_is_reused_verbatim():
    candidate=(ROOT/g.RUNNER).read_text(encoding='utf-8')
    parent=(ROOT/'training/runners/exp084_group_loss.py').read_text(encoding='utf-8')
    for line in parent.splitlines():
        if any(s in line for s in ['tds = StudyWindows(','generator=shuffle_generator,','return DataLoader(StudyWindows(','shuffle_generator = ','tds.set_epoch(ep)']):
            assert line in candidate
    assert 'from raptor_data import StudyWindows, collate' in candidate
    assert 'eval_train_studies' not in candidate  # Explicit train_ids[:4] above, no V pilot sample.


def test_gold_blocker_applies_to_new_runner(tmp_path):
    script="import sys;sys.path.insert(0,sys.argv[1]);from tools.exp086_dinov2s import install_gold_blocker;from pathlib import Path;root=Path(sys.argv[2]);install_gold_blocker(root);open(root/'storage/competition/train.csv')"
    result=subprocess.run([sys.executable,'-c',script,str(ROOT),str(tmp_path)],capture_output=True,text=True)
    assert result.returncode!=0 and 'Gold access forbidden' in result.stderr and 'FileNotFoundError' not in result.stderr


@pytest.mark.parametrize('training,epoch',[(False,0),(True,0),(True,1)])
def test_pixels_uid_and_shuffle_ignore_encoder_global_rng(monkeypatch,training,epoch):
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT/'training/src'))
    import raptor_data
    class Corpus:
        depth=64
        spec={'slots':[['a','x',18],['b','x',14],['c','x',12],['d','x',8],['e','x',12]]}
        def __init__(self,*a):pass
        def volume(self,uid):return np.arange(64*12*12,dtype=np.uint8).reshape(64,12,12)
        def mask(self,uid):return np.ones(64,np.uint8)
    monkeypatch.setattr(raptor_data,'Corpus',Corpus)
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        ds=raptor_data.StudyWindows('unused',['synthetic'],{'synthetic':np.zeros(12,np.float32)},224,12 if training else 24,training,'imagenet',42,window_policy='within-slot-valid')
        ds.set_epoch(epoch);torch.manual_seed(1);a=ds[0]
        torch.manual_seed(9);torch.rand(100003);b=ds[0]
        assert a[2]==b[2]=='synthetic' and torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
        first=torch.randperm(3455,generator=torch.Generator().manual_seed(42));torch.rand(20001)
        assert torch.equal(first,torch.randperm(3455,generator=torch.Generator().manual_seed(42)))
    finally:torch.set_num_threads(old_threads)
