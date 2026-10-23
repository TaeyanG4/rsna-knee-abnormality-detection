"""CoAtNet lifecycle v1.0.0: real CPU cards and scientific resume parity.

Synthetic environment fixtures are schema tests, never GPU qualification.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from training.arms import coatnet_lifecycle as c, coatnet_fit as fit
from training.arms import lifecycle as life
from training.runners.coatnet_lifecycle import combine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def declaration(tmp_path_factory):
    path = tmp_path_factory.mktemp('coatnet-target')/'cpu.json'
    fit.configure()
    c.write(path,fit.observation('cpu'))
    return path


def member(declaration, variant='control', population='fold0', seed=42):
    return c.register(ROOT,target=declaration.relative_to(ROOT).as_posix(),target_sha256=c.c.sha256(declaration),
        variant=variant,population_name=population,seed=seed,arm='cpu-fixture',attempt='v1')


@pytest.mark.parametrize('population,count,updates',[('fold0',3455,3456),('full',4347,4352)])
@pytest.mark.parametrize('variant',c.c.PRESETS)
@pytest.mark.parametrize('seed',[42,43])
def test_real_member_matrix(declaration,population,count,updates,variant,seed):
    spec=member(declaration,variant,population,seed)
    d=spec.document
    assert spec.run_kind=='member' and d['inputs']['controls']=={}
    assert d['config']['train_count']==count and d['config']['total_updates']==updates
    assert d['config']['recipe']['swa_1based']==[14,11,12]
    assert d['config']['bn']==dict(policy='all-training-once',count=count,epoch=16)
    assert not {'wall_budget','time_limit','projection'} & set(d['config'])
    assert c.Spec(spec.encoded).sha256==spec.sha256


@pytest.mark.parametrize('change',['fold','seed','data','metadata','source','kind','recipe','backend'])
def test_identity_drift_rejected(declaration,change):
    doc=member(declaration).document
    if change=='fold': doc['fold']=False
    if change=='seed': doc['seed']=44
    if change=='data': doc['inputs']['data']['extra']='0'*64
    if change=='metadata': doc['inputs']['data'][c.c.METADATA]='0'*64
    if change=='source': doc['inputs']['sources'].pop(next(iter(doc['inputs']['sources'])))
    if change=='kind': doc['run_kind']='experiment'
    if change=='recipe': doc['config']['recipe']['epochs']=15
    if change=='backend': doc['config']['backend_policy']['cudnn_benchmark']=False
    with pytest.raises(ValueError):
        c.validate_inputs(c.Spec.from_document(doc),ROOT)


def test_missing_target_and_historical_control_rejected(declaration):
    with pytest.raises(ValueError,match='target declaration required'):
        c.register(ROOT,target=None,target_sha256=None,variant='control',population_name='fold0',seed=42,arm='a',attempt='v1')
    controls={c.c.CONTROL+'CONTROL/'+name:c.c.PINS[c.c.CONTROL+'CONTROL/'+name] for name in ('summary.json','provenance.json')}
    with pytest.raises(ValueError,match='runtime incomplete'):
        c.register(ROOT,target=None,target_sha256=None,variant='crop',population_name='fold0',seed=42,arm='a',attempt='v1',controls=controls)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Sequential(nn.Linear(3,4),nn.Dropout(.2))
        self.head=nn.Linear(4,12)
    def forward(self,x): return self.head(self.backbone(x.reshape(x.shape[0],3)))


def stack():
    model=Tiny()
    recipe=dict(bb_lr=.003,head_lr=.006,wd=.02,pct_start=.3,clip=3.)
    return model, *fit.make_optimization(model,recipe,32,'none'),recipe


def batches():
    generator=torch.Generator().manual_seed(58)
    return [(torch.randn(1,1,3,1,1,generator=generator),torch.rand(1,12,generator=generator),['fixture']) for _ in range(17)]


def test_accumulation_and_tail_match_explicit_reference():
    torch.manual_seed(42)
    model,opt,sched,scaler,recipe=stack()
    torch.manual_seed(42)
    ref,ro,rs,_,_=stack()
    data=batches(); weights=torch.linspace(.5,2,12)
    torch.manual_seed(99)
    actual=fit.train_epoch(model,data,opt,sched,scaler,weights,recipe,'cpu','none')
    torch.manual_seed(99)
    for group in (data[:16],data[16:]):
        ro.zero_grad(set_to_none=True)
        for x,y,_ in group:
            (nn.functional.binary_cross_entropy_with_logits(ref(x),y,pos_weight=weights)/len(group)).backward()
        nn.utils.clip_grad_norm_(ref.parameters(),recipe['clip'],error_if_nonfinite=True)
        ro.step();rs.step()
    assert actual['updates']==2 and actual['batches']==17 and actual['skips']==0
    for key,value in model.state_dict().items(): assert torch.equal(value,ref.state_dict()[key])
    assert opt.state_dict()['param_groups']==ro.state_dict()['param_groups']
    assert sched.state_dict()==rs.state_dict()


def test_resume_restores_rng_optimizer_and_swa(tmp_path):
    torch.manual_seed(42)
    model,opt,sched,scaler,recipe=stack()
    identity=dict(device='cpu',swa_epochs=[1,2])
    data=batches();weights=torch.ones(12)
    record=fit.train_epoch(model,data,opt,sched,scaler,weights,recipe,'cpu','none')
    fit.save_torch(tmp_path/'swa-01.pt',dict(model=model.state_dict()))
    members={'swa-01.pt':c.c.sha256(tmp_path/'swa-01.pt')}
    fit.save_epoch(tmp_path/'epoch.pt',model,opt,sched,scaler,identity,1,[record],members)
    expected=fit.train_epoch(model,data,opt,sched,scaler,weights,recipe,'cpu','none')
    resumed,oo,ss,sc,_=stack()
    out=tmp_path/'resumed';out.mkdir()
    start,history,retained=fit.restore_epoch(tmp_path/'epoch.pt',resumed,oo,ss,sc,identity,out)
    actual=fit.train_epoch(resumed,data,oo,ss,sc,weights,recipe,'cpu','none')
    assert start==1 and history==[record] and retained==members
    assert c.c.sha256(out/'swa-01.pt')==members['swa-01.pt']
    assert actual==expected and sched.state_dict()==ss.state_dict()
    for key,value in model.state_dict().items(): assert torch.equal(value,resumed.state_dict()[key])
    (tmp_path/'swa-01.pt').write_bytes(b'tampered fixture')
    with pytest.raises(ValueError,match='hash differs|input bytes differ'):
        fit.restore_epoch(tmp_path/'epoch.pt',resumed,oo,ss,sc,identity,out)


def test_ordered_swa_and_integer_state():
    states=[dict(x=torch.tensor([v]),num_batches_tracked=torch.tensor(i)) for i,v in enumerate((1e20,-1e20,3.))]
    result=fit.swa_average(states)
    assert result['x'].item()==1 and result['num_batches_tracked'].item()==2


def test_combine_packet_is_hold_and_rejects_changed_artifact(declaration,tmp_path):
    spec=member(declaration)
    for kind in ('cpu','target'):
        dest=tmp_path/kind;(dest/'outputs').mkdir(parents=True)
        for name in ('model_best.pt','predictions-mean.csv','summary.json','provenance.json'):
            (dest/'outputs'/name).write_text('SYNTHETIC SCHEMA FIXTURE ONLY',encoding='utf-8')
        observation=fit.observation('cpu')
        # Explicit synthetic CUDA shape for packet-format tests; no real execution claim.
        summary=dict(observation=observation,cpu_cuda_initialized=False if kind=='cpu' else None,
            results=dict(train_batches=1,validation_batches=1,bn_batches=1,optimizer_steps=1,
                         checkpoint_reloaded=True,finite_outputs=True,real_data=True))
        fit.record_smoke(spec,dest,summary,['synthetic-fixture-never-qualification',kind])
    report=combine(spec,tmp_path/'cpu',tmp_path/'target',tmp_path/'combined')
    assert report['status']=='HOLD'
    note=c.read(tmp_path/'combined/verification.json')
    with pytest.raises(ValueError,match='PASS'):
        life.require_verification_pass(note,**spec.verification_args)
    (tmp_path/'combined/cpu/outputs/model_best.pt').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError): life.verify_evidence(spec,note,tmp_path/'combined')


@pytest.mark.parametrize('variant,population,seed',[(v,'fold0',42) for v in c.c.PRESETS]+[('crop','full',43)])
def test_real_lifecycle_cpu_card(variant,population,seed):
    location=os.environ.get('COATNET_LIFECYCLE_SMOKE_OUT')
    if not location: pytest.skip('Use run-step6b-tests.py for durable real CPU lifecycle cards')
    card=Path(location)/f'{variant}-{population}-s{seed}'
    card.mkdir(parents=True,exist_ok=False)
    commands=[]
    def run(args,code,status):
        argv=[sys.executable,'-B','-m','training.runners.coatnet_lifecycle',*map(str,args)]
        result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        log=card/f'command-{len(commands):02d}.log'
        log.write_text(result.stdout+result.stderr,encoding='utf-8')
        commands.append(dict(argv=argv,returncode=result.returncode,log=log.name,sha256=c.c.sha256(log)))
        c.write(card/f'command-{len(commands)-1:02d}.json',commands[-1])
        assert result.returncode==code,result.stderr[-2500:]+result.stdout[-2500:]
        output=json.loads(result.stdout)
        assert output['status']==status and output['launch_authorized'] is False
        return output
    target=card/'target-cpu.json';specpath=card/'spec.json';stage=card/'stage';smoke=card/'smoke'
    run(['observe','--device','cpu','--out',target],0,'OBSERVATION_RECORDED')
    registration=['register','--root',ROOT,'--out',specpath,'--variant',variant,'--population',population,
        '--seed',seed,'--arm','cpu-'+variant,'--attempt','v1','--target',target.relative_to(ROOT).as_posix(),
        '--target-sha256',c.c.sha256(target)]
    run(registration,0,'REGISTERED');run(registration,2,'REJECTED')
    s=['--spec',specpath]
    run(['validate',*s,'--root',ROOT],0,'VALIDATED')
    for _ in range(2): run(['prepare',*s,'--root',ROOT,'--out',stage],0,'STAGED_NO_EXECUTION')
    run(['validate',*s,'--root',stage,'--staged'],0,'VALIDATED')
    common=[*s,'--stage',stage,'--corpus',ROOT/c.c.CORPUS,'--device','cpu','--workers','0']
    run(['smoke',*common,'--out',smoke],0,'ENGINEERING_SMOKE_COMPLETE')
    summary=c.read(smoke/'outputs/summary.json')
    assert summary['updates']==1 and summary['bn']['studies']==2 and summary['bn']['bn_modules']==25
    assert summary['results']['train_batches']==2 and not summary['cpu_cuda_initialized']
    resumed=card/'resumed'
    run(['smoke',*common,'--out',resumed,'--resume',smoke/'epochs/epoch-01.pt'],0,'ENGINEERING_SMOKE_COMPLETE')
    assert c.read(resumed/'outputs/summary.json')['final_state_sha256']==summary['final_state_sha256']
    assert c.read(resumed/'outputs/summary.json')['results']['optimizer_steps']==0
    for name in ('plain','mirror','mean'):
        assert (smoke/f'outputs/predictions-{name}.csv').read_bytes()==(resumed/f'outputs/predictions-{name}.csv').read_bytes()
    run(['admit',*s,'--stage',stage,'--note',smoke/'verification.json','--evidence-root',smoke],0,'HOLD')
    run(['train',*common,'--out',card/'forbidden-full-cpu'],2,'REJECTED')
    receipt=card/'artifact-receipt.json'
    r=run(['receipt',*s,'--outputs',smoke/'outputs','--note',smoke/'verification.json','--evidence-root',smoke,'--out',receipt],0,'ARTIFACTS_VERIFIED')
    digest=life.content_sha256(c.read(receipt));archive=card/'artifacts.tar'
    run(['export',*s,'--outputs',smoke/'outputs','--receipt',receipt,'--archive',archive,'--receipt-sha256',digest],0,'EXPORTED')
    run(['collect',*s,'--archive',archive,'--receipt-sha256',digest,'--out',card/'collected'],0,'COLLECTED_ARTIFACTS_VERIFIED')
    for name,sha in r['outputs'].items(): assert c.c.sha256(card/'collected/outputs'/name)==sha
    c.write(card/'command-results.json',commands)
