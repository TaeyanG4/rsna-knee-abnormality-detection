"""Common cache contracts: no GPU execution or label output."""
import copy
from pathlib import Path
import numpy as np
import pytest
from tools import hb167_common_gold58_v1 as d


@pytest.fixture
def archive(tmp_path):
    ids=['a','b']
    targets=[str(i) for i in range(12)]
    arrays=dict(ids=np.array(ids),targets=np.array(targets),
                **{m:np.full((2,12),.4) for m in d.MEMBERS},
                **{'failed_'+m:np.zeros(2,bool) for m in d.MEMBERS})
    return tmp_path/'predictions.npz',ids,targets,arrays


@pytest.mark.parametrize('damage',['none','nan','bounds','shape','order','targets','mask-type','failed','neutral','extra','missing'])
def test_archive_fail_closed(archive,damage):
    path,ids,targets,arrays=archive
    if damage=='nan': arrays['R'][0,0]=np.nan
    if damage=='bounds': arrays['R'][0,0]=1.1
    if damage=='shape': arrays['R']=np.ones((1,12))
    if damage=='order': arrays['ids']=arrays['ids'][::-1]
    if damage=='targets': arrays['targets']=arrays['targets'][::-1]
    if damage=='mask-type': arrays['failed_R']=np.zeros(2,int)
    if damage=='failed': arrays['failed_R'][0]=True
    if damage=='neutral': arrays['R224'][0]=.5
    if damage=='extra': arrays['other']=np.zeros(1)
    if damage=='missing': del arrays['EXP155']
    d.atomic_npz(path,**arrays)
    if damage=='none': d.check_archive(path,ids,targets)
    else:
        with pytest.raises(ValueError): d.check_archive(path,ids,targets)


def test_duplicate_ids_rejected(archive):
    path,ids,targets,arrays=archive
    arrays['ids']=np.array(['a','a'])
    d.atomic_npz(path,**arrays)
    with pytest.raises(ValueError,match='unique'): d.check_archive(path,['a','a'],targets)


@pytest.mark.parametrize('damage',['none','self','verdict','binding','identity','smoke','memory','report'])
def test_independent_review_gate(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(d,'ROOT',tmp_path)
    report=tmp_path/'review.md'; report.write_text('Independent review evidence')
    binding={'registration_sha256':'a','smoke_sha256':'b','tool_sha256':'c','plan_sha256':'d'}
    value=dict(schema='HB167-common-review/v1',verdict='PASS',reviewer='independent-lane',builder=d.BUILDER,
        binding=binding.copy(),member_identity_checked=True,cpu_smoke_checked=True,
        serial_memory_gate_checked=True,report_path='review.md',report_sha256=d.sha(report))
    if damage=='self': value['reviewer']=d.BUILDER
    if damage=='verdict': value['verdict']='FAIL'
    if damage=='binding': value['binding']={}
    if damage=='identity': value['member_identity_checked']=False
    if damage=='smoke': value['cpu_smoke_checked']=False
    if damage=='memory': value['serial_memory_gate_checked']=False
    if damage=='report': value['report_sha256']='0'*64
    if damage=='none': d.check_review(value,binding)
    else:
        with pytest.raises(ValueError): d.check_review(value,binding)


def test_cpu_worker_requires_present_empty_cuda(monkeypatch):
    for val in (None,'0'):
        if val is None: monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
        else: monkeypatch.setenv('CUDA_VISIBLE_DEVICES',val)
        with pytest.raises(ValueError,match='present empty'): d.worker(True)


def test_full_consumer_rejects_smoke_receipt(tmp_path):
    from tools import c85r7_r224flip_diagnostics as consumer
    p=tmp_path/'receipt.json'
    d.io.atomic_json(p,d.stamp(schema='HB167-parent-common-cpu-smoke/v1',state='CPU_SMOKE_ONLY'))
    with pytest.raises(ValueError,match='identity or exposure'): consumer.read_common(p,['a','b'])


def test_actual_frozen_r224_extraction():
    ns=d.r224_namespace()
    from tools.c85r7_r224_timing import FUNCTIONS
    assert all(callable(ns[n]) for n in FUNCTIONS)
    assert ns['IMG']==384 and ns['MAXS']==96 and ns['K_EVAL']==94


def test_full_cohort_aggregation_and_no_batch_rank(monkeypatch):
    import sys
    import tools
    # Frozen runtime imports intentionally replace these names in a dedicated
    # worker. Restore them in this shared pytest interpreter after the test.
    for name in ('exp169_partner_bundle_release','exp152_own0949_release'):
        monkeypatch.setattr(tools,name,getattr(tools,name,None),raising=False)
        key='tools.'+name
        monkeypatch.setitem(sys.modules,key,sys.modules.get(key))
    monkeypatch.setitem(sys.modules,'hb167_frozen_v2',sys.modules.get('hb167_frozen_v2'))
    monkeypatch.setattr(sys,'path',sys.path.copy())
    q=d.q_runtime()
    rng=np.random.default_rng(167)
    scores={f:rng.random((10 if f=='l2' else 5,58,12)).astype('float32') for f in ('l2','dino','convnext')}
    result,_=q.aggregate(scores)
    paired=(scores['l2'][:5]+scores['l2'][5:])/2
    expected=q.rank(sum(q.rank(q.rank(a.mean(0))) for a in (paired,scores['dino'],scores['convnext']))/3)
    np.testing.assert_array_equal(result,expected)
    batched=np.concatenate([q.aggregate({f:a[:,i:i+2] for f,a in scores.items()})[0] for i in range(0,58,2)])
    assert not np.allclose(result,batched)
    scores['dino'][0,0,0]=np.nan
    with pytest.raises(ValueError): q.aggregate(scores)


@pytest.mark.parametrize('peak,used,ok',[(12,1000,True),(12.01,1000,False),(12,3000,False),(0,1000,False)])
def test_real_memory_gate_boundaries(peak,used,ok):
    from tools.c85r7_flipall_timing import memory_admission
    desktop=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_mib=16376,used_mib=used)
    if ok: memory_admission(desktop,[peak*2**30])
    else:
        with pytest.raises(ValueError): memory_admission(desktop,[peak*2**30])


def test_registered_single_serial_job():
    from tools.gpu_queue import validate_plan
    plan=d.read(d.STAGE/'gpu-plan.json')
    assert validate_plan(plan)==[dict(tag=d.TAG,command=d.launch_command(),continue_on_failure=False)]


def test_frozen_sources_exact():
    files=d.sources()
    assert len(files)==31
    for name,data in files.items(): assert (d.RUNTIME/name).read_bytes()==data


def test_stale_pin_fails(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'ROOT',tmp_path)
    p=tmp_path/'source.py';p.write_bytes(b'old')
    expected=d.sha(p);p.write_bytes(b'new')
    with pytest.raises(ValueError,match='identity mismatch'): d.pin(p,expected)


def test_atomic_npz_retries_sharing_violation(archive,monkeypatch):
    path,ids,targets,arrays=archive
    original=d.io.os.replace
    attempts=[]
    def busy_once(a,b):
        attempts.append(1)
        if len(attempts)==1: raise PermissionError('sharing')
        return original(a,b)
    monkeypatch.setattr(d.io.os,'replace',busy_once)
    d.atomic_npz(path,**arrays)
    d.check_archive(path,ids,targets)
    assert len(attempts)==2


def test_private_gpu_worker_cannot_bypass_review(monkeypatch):
    monkeypatch.setattr(d,'validate',lambda: {})
    def reject():
        raise ValueError('independent PASS required')
    monkeypatch.setattr(d,'gpu_review',reject)
    with pytest.raises(ValueError,match='independent PASS'):
        d.worker(False)


def test_full_consumer_accepts_authenticated_synthetic_cache(tmp_path):
    from tools import c85r7_r224flip_diagnostics as consumer
    ids=[f'synthetic-{i:02}' for i in range(58)]
    predictions=tmp_path/'synthetic.npz'
    d.atomic_npz(predictions,ids=np.asarray(ids),targets=np.asarray(consumer.gold.LABELS),
        **{m:np.full((58,12),.4) for m in d.MEMBERS},
        **{'failed_'+m:np.zeros(58,bool) for m in d.MEMBERS})
    evidence=tmp_path/'evidence.json'
    d.io.atomic_json(evidence,{'synthetic_test_only':True})
    files=d.sources()
    members={m:dict(serving_source_sha256=consumer.pack.digest(files[
        'raptor224_run.py' if m=='R224' else 'exp169_release_v2.py' if m=='Q_A5q' else 'public-serving.py']),
        checkpoint_sha256=consumer.r224.CHECKPOINT_SHA if m=='R224' else 'a'*64,
        receipt_path=d.rel(evidence),receipt_sha256=d.sha(evidence)) for m in d.MEMBERS}
    receipt=tmp_path/'receipt.json'
    d.io.atomic_json(receipt,d.stamp(schema=d.SCHEMA,state='COMPLETE',members=members,
        parent_kernel_sha256=d.sha(consumer.PARENT/'kernel/release.ipynb'),
        formula_sha256=d.sha(d.ROOT/'tools/exp169_r224.py'),
        predictions_path=d.rel(predictions),predictions_sha256=d.sha(predictions)))
    _,arrays,masks=consumer.read_common(receipt,ids)
    assert set(arrays)==set(d.MEMBERS) and not any(a.any() for a in masks.values())


@pytest.mark.parametrize('damage',['incomplete','stale'])
def test_finalize_rejects_incomplete_or_stale_worker(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(d,'WORK',tmp_path/'work')
    monkeypatch.setattr(d,'STAGE',tmp_path/'stage')
    d.io.atomic_json(d.STAGE/'registration.json',{})
    d.io.atomic_json(d.WORK/'smoke/worker.json',dict(state='RUNNING' if damage=='incomplete' else 'INFERENCE_COMPLETE',
        studies=2,registration_sha256='stale'))
    with pytest.raises(ValueError,match='incomplete worker|stale worker'):
        d.finalize(True)
    assert d.read(d.WORK/'smoke/receipt.json')['state']=='FAILED'


@pytest.fixture
def finalization(tmp_path,monkeypatch):
    """Synthetic full cache exercises actual final validators without GPU work."""
    from tools import c85r7_r224flip_diagnostics as consumer
    reg=d.read(d.STAGE/'registration.json')
    ids=[f'synthetic-{i:02}' for i in range(58)]
    reg['ids']=ids
    monkeypatch.setattr(d,'WORK',tmp_path/'work')
    monkeypatch.setattr(d,'STAGE',tmp_path/'stage')
    out=d.WORK/'gpu'
    d.io.atomic_json(d.STAGE/'registration.json',reg)
    d.io.atomic_json(d.STAGE/'independent-review.json',{'synthetic_test_only':True})
    d.atomic_npz(out/'predictions.npz',ids=np.array(ids),targets=np.array(reg['targets']),
        **{m:np.full((58,12),.4) for m in d.MEMBERS},
        **{'failed_'+m:np.zeros(58,bool) for m in d.MEMBERS})
    d.atomic_npz(out/'q-leaves.npz',synthetic_test_only=np.array([True]))
    d.io.atomic_json(out/'worker.json',dict(state='INFERENCE_COMPLETE',studies=58,
        registration_sha256=d.sha(d.STAGE/'registration.json'),seconds=1,
        predictions_sha256=d.sha(out/'predictions.npz'),q_leaves_sha256=d.sha(out/'q-leaves.npz'),
        memory_measurements=[dict(peak_reserved_bytes=2**30)]))
    d.io.atomic_json(out/'desktop.json',dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',
        total_mib=16376,used_mib=1000))
    return consumer,out,ids,dict(complete=True,peak_gpu_mib=2024)


@pytest.mark.parametrize('error',[None,ValueError,PermissionError,KeyboardInterrupt])
def test_complete_published_only_after_final_validation(finalization,monkeypatch,error):
    consumer,out,ids,telemetry=finalization
    original=consumer.validate_common
    calls=[]
    def final_validator(candidate,roster):
        if candidate['state']!='COMPLETE':
            return original(candidate,roster)
        calls.append(True)
        assert candidate['state']=='COMPLETE'
        assert d.read(out/'receipt.json')['state']=='FINALIZING'
        with pytest.raises(ValueError,match='identity or exposure'):
            consumer.read_common(out/'receipt.json',ids)
        validated=original(candidate,roster)
        if error: raise error('injected final validation error')
        return validated
    with monkeypatch.context() as scoped:
        scoped.setattr(consumer,'validate_common',final_validator)
        if error:
            with pytest.raises(error,match='injected final validation error'):
                d.finalize(False,telemetry)
        else:
            assert d.finalize(False,telemetry)['state']=='COMPLETE'
    assert calls==[True]
    if error:
        assert d.read(out/'receipt.json')['state']=='FAILED'
        assert d.read(out/'failure.json')['state']=='FAILED'
        with pytest.raises(ValueError,match='identity or exposure'):
            consumer.read_common(out/'receipt.json',ids)
    else:
        consumer.read_common(out/'receipt.json',ids)


def test_complete_write_failure_is_nonconsumable(finalization,monkeypatch):
    consumer,out,ids,telemetry=finalization
    original=d.io.atomic_json
    def fail_commit(path,value):
        if Path(path)==out/'receipt.json' and value['state']=='COMPLETE':
            raise PermissionError('injected commit error')
        return original(path,value)
    monkeypatch.setattr(d.io,'atomic_json',fail_commit)
    with pytest.raises(PermissionError,match='injected commit error'):
        d.finalize(False,telemetry)
    assert d.read(out/'receipt.json')['state']=='FAILED'
    with pytest.raises(ValueError,match='identity or exposure'):
        consumer.read_common(out/'receipt.json',ids)


@pytest.mark.parametrize('damage',['none','raise','returned-error','worker'])
def test_cleanup_must_succeed_before_finalization(tmp_path,monkeypatch,damage):
    from tools import hb149_monitor_v125 as monitor
    from tools import c85r7_r224flip_diagnostics as consumer
    monkeypatch.setattr(d,'WORK',tmp_path)
    monkeypatch.setattr(d,'validate',lambda: {})
    class Child:
        returncode=1 if damage=='worker' else 0
        def poll(self): return self.returncode
    monkeypatch.setattr(d.subprocess,'Popen',lambda *a,**kw: Child())
    monkeypatch.setattr(monitor,'_WindowsJob',lambda child: object())
    events=[]
    def cleanup(*args):
        events.append('cleanup')
        if damage=='raise': raise RuntimeError('injected cleanup error')
        return ['injected cleanup error'] if damage=='returned-error' else []
    def finish(*args):
        assert events==['cleanup']
        events.append('finalize')
        return {'state':'CPU_SMOKE_ONLY'}
    monkeypatch.setattr(monitor,'_cleanup',cleanup)
    monkeypatch.setattr(d,'finalize',finish)
    if damage=='none':
        assert d.run(True)['state']=='CPU_SMOKE_ONLY'
        assert events==['cleanup','finalize']
    else:
        with pytest.raises((RuntimeError,ValueError),match='cleanup|worker failed'):
            d.run(True)
        assert events==['cleanup']
        receipt=tmp_path/'smoke/receipt.json'
        assert d.read(receipt)['state']=='FAILED'
        with pytest.raises(ValueError,match='identity or exposure'):
            consumer.read_common(receipt,[])
