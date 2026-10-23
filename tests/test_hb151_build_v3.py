"""HB151 v3.0.2 contracts; add measured memory caps/headroom regression tests."""
import copy
import json
from pathlib import Path
import pytest
from tools import hb151_build_v3 as b


def test_public_boundaries():
    assert b.public_decision(.950) == 'alive'
    assert b.public_decision(.947) == 'dead'
    assert b.public_decision(.948) == 'seed43'
    assert b.public_decision(.946) == 'dead'
    for value in (float('nan'), float('inf'), -1, 2):
        with pytest.raises(ValueError): b.public_decision(value)


def test_whole_row_fallback_and_gold():
    labels, counts = b.mix_labels(['a','b','c'], ['gold'], {'a':[.8]*12,'b':[.6]*12},
        {'a':[.2]*12}, {u:[.1]*12 for u in ['a','b','c']})
    assert counts == dict(r7_q7s=1,r7_l2=1,l2_only=1)
    assert labels['a'] == pytest.approx([.62]*12)
    assert labels['b'] == pytest.approx([.45]*12)
    assert labels['c'] == [.1]*12
    with pytest.raises(ValueError): b.mix_labels(['gold'],['gold'],{},{},{'gold':[.1]*12})
    with pytest.raises(ValueError): b.mix_labels(['a'],[],{'a':[float('nan')]*12},{},{'a':[.1]*12})
    with pytest.raises(ValueError): b.mix_labels(['a'],[],{'a':[.2]*11},{},{'a':[.1]*12})
    with pytest.raises(ValueError): b.mix_labels(['a'],[],{},{},{})


def test_reader_no_silent_cell_fill(tmp_path):
    p=tmp_path/'labels.csv'
    p.write_text(','.join([b.parent.UID,*b.TARGETS])+'\na,'+','.join(['.2']*11+[''])+'\n')
    with pytest.raises(ValueError): b.probability_rows(p)


def test_generated_science_is_allowlisted():
    from tools.hb151_trainer_codegen_v3 import verify
    diff=verify()
    assert 'guard.require_child' in diff
    assert 'model(x.to(device' not in diff


def test_exact_execution_rejects_recipe_mutation():
    p=b.read_json(b.ROOT/b.lineage.RECIPE)
    settings=b.execution_settings(p,1,'bf16',42,False)
    assert settings['total_steps']==4352 and settings['accumulation']==16
    for values in [(2,'fp16',43,False),(1,'bf16',42,True),(1,'fp16',42,False),(1,'bf16',43,False)]:
        with pytest.raises(ValueError): b.execution_settings(p,*values)


def test_python_only_admission():
    snapshot='1, C:\\Windows\\explorer.exe\n2, C:\\Apps\\chrome.exe\n3, C:\\Python\\python.exe\n4, pythonw.exe\n'
    assert b.python_blockers(snapshot)==[3,4]
    assert b.python_blockers(snapshot,[3])==[4]
    with pytest.raises(ValueError): b.python_blockers('bad output')


def smoke():
    return dict(schema='hb151-smoke/v3',protocol_sha256=b.sha256(b.PROTOCOL),
        device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_memory_mib=16376,
        desktop_before_cuda=dict(used_mib=1222, total_mib=16376),
        shape=dict(batch=1,accumulation=16,k=12,res=384,bn_batch=2,precision='bf16',grad_checkpointing=False),
        measurements={p:dict(peak_reserved_bytes=9*1024**3,seconds_per_unit=.1) for p in ('train','bn')},
        elapsed_seconds=100,train_updates=2,bn_batches=2)


@pytest.mark.parametrize('change',['vram','bn','device','shape','time','nan','partial','stale'])
def test_smoke_fail_closed(change):
    s=smoke(); b.check_smoke(s)
    if change=='vram': s['measurements']['train']['peak_reserved_bytes']=12.5*1024**3
    if change=='bn': s['measurements']['bn']['peak_reserved_bytes']=12.5*1024**3
    if change=='device': s['device_name']='RTX 3090'
    if change=='shape': s['shape']['grad_checkpointing']=True
    if change=='time': s['elapsed_seconds']=181
    if change=='nan': s['measurements']['train']['seconds_per_unit']=float('nan')
    if change=='partial': s['train_updates']=1
    if change=='stale': s['protocol_sha256']='0'*64
    with pytest.raises(ValueError): b.check_smoke(s)


def test_real_failed_smoke_measurements_now_pass(monkeypatch):
    # Recheck original evidence on CPU, without rewriting/rebinding its receipt.
    old_stage = b.ROOT/'.local/scratch/astra-b4/H-B-151/v3'
    monkeypatch.setattr(b, 'PROTOCOL', old_stage/'manifest.json')
    s = b.read_json(old_stage/'cuda-smoke.json')
    assert s['measurements']['train']['peak_reserved_bytes'] == 10267656192
    assert s['measurements']['bn']['peak_reserved_bytes'] == 10892607488
    assert s['measurements']['train']['seconds_per_unit'] == pytest.approx(4.235)
    b.check_smoke(s)


@pytest.mark.parametrize('phase', ['train', 'bn'])
def test_memory_cap_and_exact_headroom_boundary(phase):
    s = smoke()
    s['measurements'][phase]['peak_reserved_bytes'] = 12*1024**3
    s['desktop_before_cuda']['used_mib'] = 16376 - 14*1024
    b.check_smoke(s)
    s['desktop_before_cuda']['used_mib'] += 1
    with pytest.raises(ValueError, match='headroom'): b.check_smoke(s)


@pytest.mark.parametrize('change', ['missing', 'nan', 'inf', 'negative', 'small_total', 'zero_time'])
def test_memory_measurements_and_throughput_fail_closed(change):
    s = smoke()
    if change == 'missing': s.pop('desktop_before_cuda')
    if change == 'nan': s['desktop_before_cuda']['used_mib'] = float('nan')
    if change == 'inf': s['desktop_before_cuda']['total_mib'] = float('inf')
    if change == 'negative': s['desktop_before_cuda']['used_mib'] = -1
    if change == 'small_total': s['desktop_before_cuda']['total_mib'] = 12000
    if change == 'zero_time': s['measurements']['bn']['seconds_per_unit'] = 0
    with pytest.raises(ValueError): b.check_smoke(s)


def test_direct_fit_and_smoke_cannot_bypass_gpu_job(monkeypatch):
    def deny(*args,**kwargs): raise ValueError('not under gpu_job')
    monkeypatch.setattr(b,'require_gpu_job',deny)
    for mode in ('fit','smoke'):
        with pytest.raises(ValueError,match='not under gpu_job'): b.require_child(mode)


def test_review_missing_or_stale_blocks(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'load_registration',lambda *a,**k:{})
    monkeypatch.setattr(b,'require_validation',lambda *a:None)
    monkeypatch.setattr(b,'check_smoke',lambda *a:None)
    b.atomic_json(tmp_path/'cuda-smoke.json',{})
    b.atomic_json(tmp_path/'smoke-attempt/receipt.json',dict(exit_code=0,reason='completed',survivors=[]))
    with pytest.raises(FileNotFoundError): b.require_fit()
    b.atomic_json(tmp_path/'independent-review.json',dict(verdict='PASS',reviewer='other',independent=True))
    with pytest.raises(ValueError,match='stale review'): b.require_fit()


def test_real_prepared_labels_and_parent_weights():
    p=b.load_registration()
    base=b.read_json(b.ROOT/b.lineage.RECIPE)
    assert p['pos_weight']==base['pos_weight']
    assert p['train_ids']==base['train_ids']
    assert p['recipe']==base['recipe']
    assert len(p['train_labels'])==4347
    assert p['label_provenance']['counts']==dict(r7_q7s=4335,r7_l2=4,l2_only=8)
    assert p['label_provenance']['calibration_verified'] is False
    assert not set(p['train_labels'])&set(p['excluded_gold_ids'])
    assert p['clean_oof'] is False


def test_checkpoint_replay_atomic_and_export(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(b,'RUN',tmp_path/'run')
    monkeypatch.setattr(b,'EXPORT',tmp_path/'export'/b.EXPORT.name)
    path=tmp_path/'run'/'epoch_01.pt'
    b.save_torch_new(path,{'model':{'weight':torch.tensor([1.])},'epoch':1})
    b.save_torch_new(path,{'model':{'weight':torch.tensor([2.])},'epoch':1})
    assert torch.load(path,weights_only=True)['model']['weight'].item()==2
    b.export_checkpoint(path)
    assert b.sha256(path)==b.sha256(b.EXPORT)
    assert b.EXPORT.with_suffix('.sha256').read_text().startswith(b.sha256(path))
    b.atomic_json(b.RUN/'summary.json',{'complete':True})
    with pytest.raises(ValueError): b.save_torch_new(path,{})


def test_budget_bounded_status_only(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'STAGE',tmp_path)
    assert b.charged_seconds()==0
    b.atomic_json(tmp_path/'budget.json',{'charged_seconds':123})
    assert b.charged_seconds()==123


def test_startup_publication_race_and_timeout(tmp_path):
    import threading
    path = tmp_path/'identity.json'
    writer = threading.Timer(.03, lambda: path.write_text('{"pid":123}'))
    writer.start()
    try:
        assert b.startup_publication(lambda: b.read_json(path)) == {'pid':123}
    finally:
        writer.join()
    with pytest.raises(FileNotFoundError):
        b.startup_publication(lambda: b.read_json(tmp_path/'absent'), seconds=.01)
    with pytest.raises(ValueError):
        b.startup_publication(lambda: b.need(False, 'wrong identity'))


def test_review_bound_positive_and_semantics(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'STAGE',tmp_path)
    monkeypatch.setattr(b,'load_registration',lambda *a,**k:{})
    monkeypatch.setattr(b,'require_validation',lambda *a:None)
    s=smoke()
    b.atomic_json(tmp_path/'cuda-smoke.json',s)
    b.atomic_json(tmp_path/'smoke-attempt/receipt.json',dict(exit_code=0,reason='completed',survivors=[]))
    b.atomic_json(tmp_path/'cpu-validation.json',{'status':'PASS_CPU'})
    review=dict(verdict='PASS',reviewer='independent-fixture',independent=True,
        protocol_sha256=b.sha256(b.PROTOCOL),smoke_sha256=b.sha256(tmp_path/'cuda-smoke.json'),
        smoke_receipt_sha256=b.sha256(tmp_path/'smoke-attempt/receipt.json'),
        cpu_validation_sha256=b.sha256(tmp_path/'cpu-validation.json'),
        probability_semantics='native-probability-space accepted by owner')
    b.atomic_json(tmp_path/'independent-review.json',review)
    assert b.require_fit() == s
    review.pop('probability_semantics')
    b.atomic_json(tmp_path/'independent-review.json',review)
    with pytest.raises(ValueError,match='calibration evidence'): b.require_fit()


def test_export_failure_and_cpu_recovery(monkeypatch,tmp_path):
    import torch
    monkeypatch.setattr(b,'RUN',tmp_path/'run')
    monkeypatch.setattr(b,'EXPORT',tmp_path/'export'/b.EXPORT.name)
    monkeypatch.setattr(b,'load_registration',lambda *a,**k:{})
    source=b.RUN/'model_best.pt'
    b.save_torch_atomic(source,{'model':{'weight':torch.tensor([1.])}})
    b.atomic_json(b.RUN/'summary.json',dict(status='COMPLETE_TRAINING_NO_EVALUATION',model_sha256=b.sha256(source)))
    replace=b.safe_replace
    monkeypatch.setattr(b,'safe_replace',lambda *a,**k:False)
    assert b.export_checkpoint(source)['status']=='EXPORT_PENDING'
    with pytest.raises(ValueError,match='export failed'): b.recover_export()
    monkeypatch.setattr(b,'safe_replace',replace)
    assert b.main(['export'])==0
    assert b.main(['export'])==0
    assert b.EXPORT.with_suffix('.sha256').read_text().startswith(b.sha256(source))
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint drift'): b.recover_export()


def test_real_desktop_snapshot_parser():
    # Binds the admission parser to the captured real desktop table, no CUDA.
    snapshot=(b.STAGE/'nvidia-compute-apps.txt').read_text()
    assert isinstance(b.python_blockers(snapshot),list)
