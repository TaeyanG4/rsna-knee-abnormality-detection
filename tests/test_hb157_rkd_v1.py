"""HB157 v1.0.1 CPU contract, resume, gate and exact serving-scope regressions."""
import copy
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from tools import hb157_rkd_v1 as b
from tools import hb157_rkd_killtest_v1 as k
from training.arms import hb157_rkd_fit_v1 as f


def test_registered_only_authorized_scientific_changes():
    old=b.read(b.ROOT/b.SPEC)['config'];new=b.scientific()
    for key in old.keys()-{'epochs','total_updates','scope','decision','adoption','recipe','continuation','bn','supervision','class_weight_recipe'}:
        assert new[key]==old[key],key
    assert new['recipe']==dict(old['recipe'],epochs=6,swa_1based=[4,5,6],labels='H-B-151 soft probabilities')
    assert new['continuation']==dict(old['continuation'],data_epochs=list(range(16,22)))
    assert new['bn']==dict(old['bn'],epoch=22)
    assert new['seed']==44 and new['total_updates']==1632 and new['clean_oof'] is False


def test_real_inputs_and_frozen_weights():
    assert b.sha(b.ROOT/b.LABELS)==b.LABEL_SHA
    ids,labels=b.population();p=b.read(b.ROOT/b.ROSTER)
    assert len(ids)==len(labels)==4347 and not set(ids)&set(p['excluded_gold_ids'])
    assert b.scientific()['pos_weight']==p['pos_weight']
    assert any(0<v<1 for row in labels.values() for v in row)


def test_pin_and_config_drift(tmp_path,monkeypatch):
    registered=b.registration();monkeypatch.setattr(b,'PROTOCOL',tmp_path/'protocol.json')
    b.atomic_json(b.PROTOCOL,registered);b.validate()
    registered['pins'][b.LABELS]='0'*64;b.atomic_json(b.PROTOCOL,registered)
    with pytest.raises(ValueError,match='drift'):b.validate()
    original=b.sha
    monkeypatch.setattr(b,'sha',lambda p:'0'*64 if Path(p)==b.ROOT/b.PARENT else original(p))
    with pytest.raises(ValueError,match='authenticated input'):b.registration()


def smoke():
    return dict(status='PASS',protocol_sha256=b.sha(b.PROTOCOL),device_name='NVIDIA GeForce RTX 4070 Ti SUPER',
        total_memory_mib=16376,desktop_before_cuda=dict(used_mib=1222,total_mib=16376),
        shape=dict(batch=1,k=12,res=384,accumulation=16,bn_batch=2,k_eval=94,crop=320,mirror=True,precision='bf16',checkpointing=False),
        elapsed_seconds=100,updates=1,bn_batches=1,
        measurements={p:dict(peak_reserved_bytes=9*1024**3,seconds_per_unit=.1) for p in ['train','bn','serving']})


@pytest.mark.parametrize('change',['train','bn','serving','desktop','desktop_nan','desktop_total','headroom','shape','mirror','device','partial','time','stale','rate'])
def test_measurements_fail_closed(change):
    s=smoke();b.check_smoke(s)
    if change in ['train','bn','serving']:s['measurements'][change]['peak_reserved_bytes']=12.5*1024**3
    elif change=='desktop':s['desktop_before_cuda']['used_mib']=-1
    elif change=='desktop_nan':s['desktop_before_cuda']['used_mib']=float('nan')
    elif change=='desktop_total':s['desktop_before_cuda']['total_mib']=float('inf')
    elif change=='headroom':s['desktop_before_cuda']['used_mib']=6000
    elif change=='shape':s['shape']['k_eval']=24
    elif change=='mirror':s['shape']['mirror']=False
    elif change=='device':s['device_name']='RTX 4090'
    elif change=='partial':s['updates']=0
    elif change=='time':s['elapsed_seconds']=166
    elif change=='stale':s['protocol_sha256']='0'*64
    else:s['measurements']['train']['seconds_per_unit']=0
    with pytest.raises(ValueError):b.check_smoke(s)


def test_memory_exact_headroom():
    s=smoke();s['desktop_before_cuda']['used_mib']=16376-11*1024;b.check_smoke(s)
    s['desktop_before_cuda']['used_mib']+=1
    with pytest.raises(ValueError):b.check_smoke(s)


@pytest.mark.parametrize('phase',['train','bn','serving'])
@pytest.mark.parametrize('peak_gib',[10.9,11,12])
def test_measured_like_peaks_pass(phase,peak_gib):
    s=smoke();s['measurements'][phase]['peak_reserved_bytes']=int(peak_gib*1024**3)
    b.check_smoke(s)


@pytest.mark.parametrize('phase',['train','bn','serving'])
@pytest.mark.parametrize('total_field',['desktop','cuda'])
def test_largest_peak_uses_smaller_measured_total(phase,total_field):
    s=smoke();s['measurements'][phase]['peak_reserved_bytes']=12*1024**3
    if total_field=='desktop':s['desktop_before_cuda']['total_mib']=16000
    else:s['total_memory_mib']=16000
    s['desktop_before_cuda']['used_mib']=16000-14*1024
    b.check_smoke(s)
    s['desktop_before_cuda']['used_mib']+=1
    with pytest.raises(ValueError,match='desktop \\+ peak'):b.check_smoke(s)


@pytest.mark.parametrize('phase',['train','bn','serving'])
@pytest.mark.parametrize('field',['peak_reserved_bytes','seconds_per_unit'])
@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_peak_and_rate_remain_finite_positive(phase,field,value):
    s=smoke();s['measurements'][phase][field]=value
    with pytest.raises(ValueError,match='peak/rate gate'):b.check_smoke(s)


@pytest.mark.parametrize('change',['missing','self','not_independent','hold','stale'])
def test_review_gate(change,tmp_path):
    receipt=tmp_path/'receipt';receipt.write_text('CPU fixture')
    r=dict(verdict='PASS',independent=True,reviewer='independent-fixture',receipt_sha256=b.sha(receipt))
    b.check_review(r,dict(receipt_sha256=receipt))
    if change=='missing':r.pop('receipt_sha256')
    if change=='self':r['reviewer']='hb157-rkd'
    if change=='not_independent':r['independent']=False
    if change=='hold':r['verdict']='HOLD'
    if change=='stale':receipt.write_text('changed')
    with pytest.raises(ValueError):b.check_review(r,dict(receipt_sha256=receipt))


def test_legacy_review_cannot_admit_new_registration():
    legacy=b.read(b.WORK.parent/'pre-gpu-review.json')
    with pytest.raises(ValueError,match='stale/missing review binding: protocol_sha256'):
        b.check_review(legacy,dict(protocol_sha256=b.PROTOCOL,cpu_smoke_sha256=b.WORK/'cpu-smoke.json'))


@pytest.mark.parametrize('mode',['smoke','fit'])
def test_no_direct_cuda_or_supervisor_bypass(mode,monkeypatch):
    def deny(*a,**kw):raise ValueError('gpu_job denied')
    monkeypatch.setattr(b,'require_gpu_job',deny)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:pytest.fail('CUDA before admission'))
    with pytest.raises(ValueError,match='gpu_job denied'):f.run('cuda',mode)
    with pytest.raises(ValueError,match='gpu_job denied'):b.supervise(mode)


def test_missing_review_stops_before_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'WORK',tmp_path);monkeypatch.setattr(b,'require_cpu',lambda:None)
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**kw:None)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:pytest.fail('CUDA before review'))
    for mode in ['smoke','fit']:
        with pytest.raises(FileNotFoundError):f.run('cuda',mode)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Linear(12,8);self.head=nn.Linear(8,12)
    def forward(self,x):return self.head(self.backbone(x.flatten(1)))


def test_soft_bce_accumulation_matches_parent_and_weights():
    from training.arms.rplus_fit import train_epoch
    torch.set_num_threads(2);torch.manual_seed(44);a=Tiny();other=copy.deepcopy(a)
    recipe=b.scientific()['recipe'];weights=torch.tensor(b.scientific()['pos_weight'])
    rows=[(torch.randn(1,1,3,2,2),torch.rand(1,12),torch.ones(1,12),['synthetic']) for _ in range(19)]
    opt1=f.make_optimization(a,recipe,1632,'none');opt2=f.make_optimization(other,recipe,1632,'none')
    r=f.train_epoch(a,rows,*opt1,weights,recipe,'cpu','none')
    assert r==train_epoch(other,rows,*opt2,weights,recipe,'cpu','none')
    for key,value in a.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
    assert [g['lr'] for g in opt1[0].param_groups]==[4.01e-6,.000195]


def tiny_journal(tmp_path):
    model=Tiny();opt,sch,scaler=f.make_optimization(model,b.scientific()['recipe'],1632,'none')
    model(torch.ones(1,12)).sum().backward();opt.step();sch.step();opt.zero_grad()
    identity=dict(epochs=6,swa_epochs=[4,5,6],updates_per_epoch=272)
    history=[dict(epoch=1,updates=272,skips=0)]
    r=f.journal(tmp_path,model,opt,sch,scaler,identity,1,history,{})
    return model,opt,sch,scaler,identity,r


def test_resume_restores_optimizer_scheduler_and_next_step(tmp_path):
    model,opt,sch,scaler,identity,_=tiny_journal(tmp_path)
    other=Tiny();opt2,sch2,scaler2=f.make_optimization(other,b.scientific()['recipe'],1632,'none')
    assert f.restore(tmp_path,other,opt2,sch2,scaler2,identity)[0]==1
    assert sch2.state_dict()==sch.state_dict() and len(opt2.state)==len(opt.state)>0
    for m,o in [(model,opt),(other,opt2)]:m(torch.ones(1,12)).sum().backward();o.step()
    for key,value in model.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)


@pytest.mark.parametrize('change',['identity','bytes','path','swa','epoch','history'])
def test_resume_rejects_corruption(tmp_path,change):
    model,opt,sch,scaler,identity,r=tiny_journal(tmp_path)
    if change=='identity':identity['epochs']=5
    elif change=='bytes':(tmp_path/r['path']).write_bytes(b'broken')
    elif change=='path':r['path']='../outside.pt'
    elif change=='swa':r['swa_members']={'swa-04.pt':'0'*64}
    elif change=='epoch':r['epoch']=7
    else:
        ck=torch.load(tmp_path/r['path'],weights_only=True);ck['history'][0]['skips']=1
        f.save(tmp_path/r['path'],ck);r['sha256']=b.sha(tmp_path/r['path'])
    b.atomic_json(tmp_path/'journal.json',r)
    with pytest.raises(ValueError):f.restore(tmp_path,model,opt,sch,scaler,identity)


def test_real_desktop_parser():
    from tools.hb151_build_v3 import admission_snapshot,python_blockers,desktop_memory_snapshot
    snapshot=admission_snapshot();blockers=python_blockers(snapshot)
    assert all(isinstance(pid,int) for pid in blockers)
    assert python_blockers('1, C:\\Windows\\explorer.exe\n2, C:\\Python\\python.exe\n')==[2]
    memory=desktop_memory_snapshot();assert 16000<=memory['total_mib']<=16400


@pytest.mark.parametrize('score,verdict',[(.949,'DEAD'),(.948,'DEAD'),(.950,'INCONCLUSIVE'),(.95199,'INCONCLUSIVE'),(.952,'ALIVE'),(.953,'ALIVE')])
def test_registered_public_decision(score,verdict):assert k.outcome(score)==verdict


@pytest.mark.parametrize('score',[float('nan'),float('inf'),-1,2])
def test_invalid_score(score):
    with pytest.raises(ValueError):k.outcome(score)


def test_serving_parent_embedded_and_guard():
    parent,donor=k.parents();candidate=k.payload('a'*64)
    assert candidate['cells'][0]==donor['cells'][0]
    assert k.embedded(parent)==k.embedded(candidate)
    assert candidate['metadata']==parent['metadata']
    assert k.metadata()['title']==k.SLUG.split('/')[1]
    for cell in candidate['cells'][1:]:
        if cell['cell_type']=='code':assert k.source(cell).startswith('if not FAST:\n')


@pytest.mark.parametrize('change',['launch','metadata','embedded','markdown','checkpoint'])
def test_prepared_stage_scope_rejects_extra_changes(tmp_path,monkeypatch,change):
    # Small stage fixture; equality of every cell and embedded byte is exercised.
    monkeypatch.setattr(k,'OUT',tmp_path);pin=b.sha(b.ROOT/b.PARENT)
    nb=k.payload(pin);b.atomic_json(tmp_path/'kernel/release.ipynb',nb)
    b.atomic_json(tmp_path/'kernel/kernel-metadata.json',k.metadata())
    b.atomic_json(tmp_path/'dataset/dataset-metadata.json',k.dataset_metadata())
    b.atomic_json(tmp_path/'build-receipt.json',dict(checkpoint_sha256=pin,placeholder=True))
    k.write(tmp_path/'unpacked/_fastread.py',k.embedded(nb)['_FASTREAD_B64'])
    original=k.sha;monkeypatch.setattr(k,'sha',lambda p:pin if Path(p)==tmp_path/'dataset'/k.NAME else original(p))
    assert k.scope()['status']=='PASS_CPU_SCOPE'
    if change=='launch':nb['cells'][-1]['source']+='\nprint("changed runtime")\n'
    elif change=='metadata':nb['metadata']['changed']=True
    elif change=='markdown':nb['cells'][1]['source']='unexpected edit'
    elif change=='embedded':k.write(tmp_path/'unpacked/_fastread.py',b'changed')
    else:b.atomic_json(tmp_path/'build-receipt.json',dict(checkpoint_sha256='0'*64,placeholder=True))
    b.atomic_json(tmp_path/'kernel/release.ipynb',nb)
    with pytest.raises(ValueError):k.scope()


def test_placeholder_cannot_release(tmp_path,monkeypatch):
    monkeypatch.setattr(k,'OUT',tmp_path);monkeypatch.setattr(k,'scope',lambda:None)
    b.atomic_json(tmp_path/'build-receipt.json',dict(placeholder=True))
    with pytest.raises(ValueError,match='placeholder'):k.admit()


@pytest.mark.parametrize('change',['none','cpu','partial','parent','labels','weights','bn','swa','exposure','supervisor','bytes'])
def test_final_candidate_lineage(tmp_path,monkeypatch,change):
    monkeypatch.setattr(b,'WORK',tmp_path);monkeypatch.setattr(b,'RUN',tmp_path/'fit')
    p=dict(mode='fit',device='cuda',epochs=6,run_dir=b.RUN.relative_to(b.ROOT).as_posix(),
        protocol_sha256=b.sha(b.PROTOCOL),soft_label_sha256=b.LABEL_SHA,
        public_parent_sha256=b.PARENT_SHA,pos_weight=b.scientific()['pos_weight'],bn=dict(unique_studies=4347))
    ck=dict(model={'fixture':torch.zeros(1)},provenance=p,seed=44,swa_epochs_1based=[4,5,6],clean_oof=False,exposure='EXPOSED')
    if change=='cpu':p['device']='cpu'
    elif change=='parent':p['public_parent_sha256']='0'*64
    elif change=='labels':p['soft_label_sha256']='0'*64
    elif change=='weights':p['pos_weight']=[1]*12
    elif change=='bn':p['bn']['unique_studies']=2
    elif change=='swa':ck['swa_epochs_1based']=[1,2,3]
    elif change=='exposure':ck['clean_oof']=True
    path=b.RUN/'model_best.pt';f.save(path,ck)
    b.atomic_json(tmp_path/'budget.json',dict(attempt=1,running=False))
    b.atomic_json(tmp_path/'fit-attempt-1/receipt.json',dict(exit_code=1 if change=='supervisor' else 0,reason='completed',survivors=[]))
    b.atomic_json(b.RUN/'summary.json',dict(status='COMPLETE_TRAINING_AND_READOUT',updates=272 if change=='partial' else 1632,
        protocol_sha256=b.sha(b.PROTOCOL),model_sha256='0'*64 if change=='bytes' else b.sha(path)))
    if change=='none':assert k.final_checkpoint()==path
    else:
        with pytest.raises(ValueError):k.final_checkpoint()


def test_real_cpu_candidate_exact_public_R_loader(tmp_path,monkeypatch):
    # A real newly trained smoke checkpoint, never eligible for scientific release.
    checkpoint=b.WORK/'cpu-run/model_best.pt'
    # A new source version must not rewrite the preserved reviewed protocol.
    monkeypatch.setattr(b,'PROTOCOL',tmp_path/'protocol.json')
    b.atomic_json(b.PROTOCOL,b.registration())
    monkeypatch.setattr(k,'OUT',tmp_path)
    assert k.stage(checkpoint,True)['status']=='PASS_CPU_SCOPE'
    assert k.loader_smoke()['strict_load'] is True
