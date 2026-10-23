"""CPU regression tests for the one-delta failure-weight producer."""
import ast
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pytest
import torch
import torch.nn.functional as F
from tools import l1_rco_k1_failure_weights_v1 as w
from tools import l1_rco_k1_failure_v1 as p
from tools import l1_rco_k1_failure_gate_v1 as gate


def csv(tmp_path, rows='a,x,2\n'):
    path=tmp_path/'weights.csv'
    path.write_text('study_uid,target,weight\n'+rows)
    return path


def bce(source):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='bce_soft')
    scope={'torch':torch,'F':F}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'bce','exec'),scope)
    return scope['bce_soft']


def test_exact_seed0_loss_and_gradient():
    raw=(p.ROOT/p.k.s.SOURCE/'compact_train.py').read_text(encoding='utf8')
    old=bce(p.k.parent.derive_trainer(raw)); new=bce(p.derive_trainer(raw))
    z=torch.tensor([[-1.5,.5,2.],[3.,-2.,.25]],requires_grad=True)
    target=torch.tensor([[0.,.25,1.],[.8,0.,.6]])
    one=torch.ones_like(target)
    a=old(z,target,one); ga=torch.autograd.grad(a,z)[0]
    b=new(z,target,one); gb=torch.autograd.grad(b,z)[0]
    assert torch.equal(a,b) and torch.equal(ga,gb)
    weight=torch.tensor([[2.,.5,1.],[.5,2.,1.]])
    norm=w.normalize(weight)
    assert torch.equal(norm.mean(0),torch.ones(3))
    expected=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*norm).mean()
    assert torch.allclose(new(z,target,weight),expected,atol=1e-7,rtol=0)
    assert not torch.equal(new(z,target,weight),a)
    assert torch.equal(new(z,target,weight*4),new(z,target,weight))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('rows', ['h,x,1\n','unknown,x,1\n','a,bad,1\n','a,x,1\na,x,2\n',
                                  'a,x,nan\n','a,x,inf\n','a,x,0\n','a,x,-1\n','a,x,1,extra\n','a,x\n'])
def test_reject_bad_weights(tmp_path,rows):
    with pytest.raises(ValueError): w.load_weights(csv(tmp_path,rows),['a'],['h'],['x'])


def test_defaults_and_train_only(tmp_path):
    path=csv(tmp_path)
    result=w.tensor_weights(path,['a','b'],['h'],['x','y'],['b','a'],torch.ones(2,2))
    assert torch.equal(result,torch.tensor([[1.,1.],[2.,1.]]))
    with pytest.raises(ValueError): w.tensor_weights(path,['a'],['h'],['x'],['h'],torch.ones(1,1))


def test_minimal_trainer_diff():
    raw=(p.ROOT/p.k.s.SOURCE/'compact_train.py').read_text(encoding='utf8')
    old=p.k.parent.derive_trainer(raw)
    new=p.derive_trainer(raw)
    restored=new.split('\n',1)[1].replace('tools.l1_rco_k1_failure_v1','tools.l1_rco_k1_v3')
    restored=restored.replace('    from tools.l1_rco_k1_failure_weights_v1 import normalize\n    weight = normalize(weight)\n','')
    restored=restored.replace('        from tools.l1_rco_k1_v3 import training_weights\n        return kind, t, training_weights(pool, t)',
                              '        return kind, t, torch.ones_like(t)')
    assert restored==old
    reg=p.k.read(p.k.PARENT)
    assert (len(reg['train']),len(reg['holdout']),reg['recipe']['total_steps'])==(3190,977,15950)


@pytest.mark.parametrize('deltas,passed', [([.003]*4,True),([.003,.003,.003,-.0001],True),
                                         ([.001]*4,False),([.01,.01,-.001,-.001],False),([0.]*4,False)])
def test_gate(deltas,passed):
    base={r:.8 for r in gate.gate.RULERS}
    out=gate.decide(base,{r:.8+d for r,d in zip(gate.gate.RULERS,deltas)})
    assert (out['status']=='SCREEN_PASS_PENDING_REVIEW') is passed
    assert out['independent_efficacy_evidence'] is False


def test_testonly_admission_blocked(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'TEST_ONLY',True);monkeypatch.setattr(p,'WORK',tmp_path)
    result=p.admit({'test_only':True},tmp_path,tmp_path/'missing')
    assert result['status']=='BLOCKED'
    assert 'TEST_ONLY' in result['blockers'][0]


def test_idle_files_accepts_failure_but_rejects_identity(tmp_path):
    root=tmp_path/'.local/scratch/astra-b2/h224-folds-v1/gpu0-owner/artifacts/logs'
    tags=('H224_K1_V3G_ALL','H224_K1_V3G_F3','DA052_K1_FULLBAG_QUALIFY','DA052_K1_V3G_FULLBAG_F0')
    for tag in tags:
        d=root/tag;d.mkdir(parents=True)
        p.k.write(d/'job.json',dict(run_id=tag,command_sha256='a'*64))
        p.k.write(d/'receipt.json',dict(run_id=tag,command_sha256='a'*64,state='FAILED'))
    assert p.idle_files(tmp_path)['status']=='KNOWN_JOBS_TERMINAL'
    p.k.write(root/tags[0]/'receipt.json',dict(run_id='wrong',command_sha256='a'*64,state='SUCCESS'))
    with pytest.raises(ValueError,match='identity'):p.idle_files(tmp_path)
