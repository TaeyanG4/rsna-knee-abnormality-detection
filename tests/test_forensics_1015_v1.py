"""Forensics tests v1.1.1: isolated retry routing and terminal predecessor queue."""
import numpy as np
import pytest
from tools import forensics_1015_v1 as f


def test_auc_ties_and_missing_classes():
    assert f.auc([0,1,0,1],[.5]*4)==.5
    assert f.auc([0,1],[.1,.9])==1
    assert f.auc([1,1],[.1,.9]) is None
    with pytest.raises(ValueError): f.auc([0,1],[float('nan'),.2])


def test_paired_bootstrap_and_undefined_targets():
    y=np.tile([0,1,0,1],(12,1)).T
    a=np.tile([.1,.9,.3,.8],(12,1)).T
    result=f.bootstrap(y,a,a,draws=50)
    assert all(r['delta']==0 and r['ci95']==[0,0] for r in result.values())
    b=1-a
    r=f.bootstrap(y,b,a,draws=50)
    assert all(v['delta']==-1 and v['ci95']==[-1,-1] for v in r.values())
    y[:,0]=1
    assert f.bootstrap(y,a,a,draws=20)['ACL']['valid_draws']==0


def test_leakage_and_duplicates():
    assert f.exclusion(['a','b'],['b','c'])['overlap']==1
    with pytest.raises(ValueError): f.exclusion(['a','a'],[])


def test_order_changes_ties_and_entropy():
    x=np.tile([0.,.5,1.],(12,1)).T
    r=f.target_structure(x,x[::-1],x)
    assert all(v['soft_l2_spearman']==-1 and v['pair_strict_reversal_fraction']==1 for v in r.values())
    assert all(v['rank_position_changed_fraction']==pytest.approx(2/3) for v in r.values())
    assert r['ACL']['r7_covariance_share']==pytest.approx(.7)


def test_memory_gate_measured_boundary():
    desktop=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_mib=16376,used_mib=1000)
    assert f.memory(desktop,[12*2**30])==12*2**30
    with pytest.raises(ValueError): f.memory(desktop,[12*2**30+1])
    desktop['used_mib']=3000
    with pytest.raises(ValueError): f.memory(desktop,[12*2**30])


def test_idle_only():
    # v1.1.0: committed progress only; GPU activity no longer resets the watchdog.
    assert f.idle_reason(900,0)
    assert f.idle_reason(9000,0,[1000,50,200],16376)
    assert f.idle_reason(9000,8999) is None
    assert f.idle_reason(899,0,[1000,50,200],16376) is None
    # WDDM stall signature (util>=95, power<90 W, VRAM near total) needs 600 s without progress.
    assert f.idle_reason(600,0,[14000,99,40],16376)
    assert f.idle_reason(599,0,[14000,99,40],16376) is None
    assert f.idle_reason(700,0,[14000,99,150],16376) is None


def test_review_and_controller_bypass(monkeypatch):
    with pytest.raises(ValueError): f.check_review({'verdict':'PASS','reviewer':'root'})
    monkeypatch.delenv('FORENSICS_CONTROLLER',raising=False)
    with pytest.raises(ValueError): f.require_controller('full')


def test_roster_and_probability_contract():
    with pytest.raises(ValueError): f.valid_probs(np.ones((2,11)),2)
    with pytest.raises(ValueError): f.valid_probs(np.full((2,12),1.1),2)
    with pytest.raises(ValueError): f.definitions('def load_model(): pass')


def test_registration_rejects_mismatch(monkeypatch,tmp_path):
    monkeypatch.setattr(f,'WORK',tmp_path)
    f.io.atomic_json(tmp_path/'registration.json',{'sources':{'a':'old'}})
    monkeypatch.setattr(f,'registration',lambda:{'sources':{'a':'new'}})
    with pytest.raises(ValueError,match='stale'): f.registered()


def test_hb156_scored_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError,match='checkpoint changed'):
        f.hb156_scored_identity(tmp_path,'0'*64)


def test_raw_input_pairing_rejected(tmp_path):
    for arm in f.ARMS:
        f.io.atomic_json(tmp_path/'journal'/f'{arm}-000.json',{'input':
            {'volume_sha256': 'different' if arm=='HB156_train_eval' else 'same', 'mask_sha256':'mask'}})
    with pytest.raises(ValueError,match='different inputs'): f.raw_identity_check(tmp_path,1)


def test_windows_pread_preserves_binary_bytes_and_offset(tmp_path):
    import os
    import types
    if hasattr(os,'pread'): pytest.skip('Windows compatibility adapter only')
    path=tmp_path/'binary.dat'; data=b'a\r\n\x1ab\x00\xff'*200
    path.write_bytes(data)
    module=types.SimpleNamespace(_O_FLAGS=os.O_RDONLY)
    assert f.windows_pread_adapter(module)
    fd=module.os.open(path,module._O_FLAGS)
    try:
        os.lseek(fd,17,os.SEEK_SET)
        assert module.os.pread(fd,110,3)==data[3:113]
        assert os.lseek(fd,0,os.SEEK_CUR)==17
    finally: os.close(fd)


def precision_rows(ids=('a','b'), delta=0., control_delta=0., bad=False):
    rows=[]
    for arm in f.PRECISION_ARMS:
        for mode in f.PRECISIONS:
            for uid in ids:
                value=.25+(delta if arm=='HB156' else control_delta) if mode=='fp16' else .25
                rows.append(dict(arm=arm,precision=mode,uid=uid,probabilities=[value]*12,error=None,
                    nonfinite_probabilities=0,saturated_probabilities=0,
                    input=dict(volume_sha256=uid,mask_sha256='m'),modules={'stage':dict(
                        max_abs_finite=1.,nonfinite=int(bad and arm=='HB156' and mode=='fp16'),
                        nan=0,inf=int(bad and arm=='HB156' and mode=='fp16'),at_or_above_fp16_limit=0)}))
    return rows


def test_overflow_registered_reading_strict_threshold_and_control():
    ids=[str(i) for i in range(16)]
    result=f.precision_summary(precision_rows(ids,delta=.019),ids,'cuda:0')
    assert result['registered_reading']=='FP16_SERVING_DEFECT_NOT_SUPPORTED'
    assert result['gpu_hypothesis_test']
    assert f.precision_summary(precision_rows(ids,delta=.021),ids,'cuda:0')['registered_reading']=='FP16_SERVING_DEFECT_SUPPORTED'
    assert f.precision_summary(precision_rows(ids,delta=.03,control_delta=.03),ids,'cuda:0')['registered_reading']=='FP16_SERVING_DEFECT_NOT_SUPPORTED'
    assert f.precision_summary(precision_rows(delta=.03),['a','b'],'cuda:0')['registered_reading']=='CALIBRATION_ONLY'
    assert f.precision_summary(precision_rows(delta=.03),['a','b'],'cpu')['registered_reading']=='CPU_DIAGNOSTIC_ONLY'
    assert f.precision_summary(precision_rows(bad=True),['a','b'],'cpu')['models']['HB156']['criterion']
    assert f.phase_count('precision')==96


def test_overflow_missing_or_unpaired_evidence_rejected():
    rows=precision_rows()
    with pytest.raises(ValueError): f.precision_summary(rows[:-1],['a','b'],'cpu')
    rows[0]['input']['volume_sha256']='changed'
    with pytest.raises(ValueError,match='inputs differ'): f.precision_summary(rows,['a','b'],'cpu')
    ids=[str(i) for i in range(16)]
    rows=precision_rows(ids); rows[32]['modules']['stage']['nonfinite']=1
    assert f.precision_summary(rows,ids,'cuda:0')['registered_reading']=='INCONCLUSIVE_INVALID_FP32'


def test_activation_hook_nonfinite_limit_counts_and_no_mutation():
    import torch
    model=torch.nn.Sequential(torch.nn.Identity())
    audit=f.ActivationAudit(model)
    x=torch.tensor([65504.,65505.,float('inf'),float('nan'),-2.])
    out=model(x)
    assert out is x
    assert audit.stats['0']['nonfinite']==2
    assert audit.stats['logits']['at_or_above_fp16_limit']==2
    assert audit.stats['logits']['max_abs_finite']==65505
    assert audit.first_nonfinite=='0'
    audit.close()
    assert not model[0]._forward_hooks


def test_precision_adapter_exact_fp16_and_cpu_real_policy():
    import torch
    # Read only authenticated source. Never instantiate weights or CUDA here.
    source=f.payload('HB156')[0]
    ns=dict(torch=torch)
    fn,proof=f.precision_function(ns,source,'fp16','cuda:0')
    assert proof['exact_scored_function']
    assert proof['requested_precision']==proof['effective_precision']=='fp16'
    class Probe(torch.nn.Module):
        def forward(self,x):
            assert not torch.is_autocast_enabled('cpu')
            dtype=torch.get_autocast_dtype('cpu') if torch.is_autocast_enabled('cpu') else torch.float32
            return torch.full((1,12),1.,dtype=dtype)
    for mode in f.PRECISIONS:
        fn,proof=f.precision_function(ns,source,mode,'cpu')
        assert not proof['exact_scored_function']
        assert proof['requested_precision']==mode and proof['effective_precision']=='fp32'
        p=fn(Probe(),torch.zeros(1),'cpu')
        assert p.shape==(12,) and np.isfinite(p).all()
    class Bad(torch.nn.Module):
        def forward(self,x): return torch.full((1,12),float('inf'))
    with pytest.raises(AssertionError,match='Nonfinite fp16 logits'): fn(Bad(),torch.zeros(1),'cpu')
    audit=f.serving_nonfinite_audit(source)
    assert audit['silent_successful_fallback'] is False


def test_precision_failure_is_retained_and_hooks_restored():
    import torch
    source=f.payload('HB156')[0]
    ns=dict(torch=torch)
    original,proof=f.precision_function(ns,source,'fp32','cpu')
    ns['infer_probs']=original
    exec(f.definitions(source)['infer_pair_probs'],ns)
    ns['gpu_windows']=lambda *args,**kwargs: torch.zeros(94,3,320,320).expand(94,3,320,320)
    class Bad(torch.nn.Module):
        def forward(self,x): return torch.full((1,12),float('nan'))
    model=Bad().eval()
    result=f.precision_infer(ns,source,model,None,None,'cpu','fp16')
    assert result['error']=='Nonfinite fp16 logits' and result['probabilities'] is None
    assert result['modules']['logits']['nonfinite']==12
    assert ns['infer_probs'] is original and not model._forward_hooks


def test_recovered_hb156_training_identity_and_curve():
    data=f.read(f.HB156_LOGS/'S42/summary.json')
    assert data['model_sha256']=='33d9ef8a284460f8f3d199a98edad911170cffcaccec562d68475214c1f613e1'
    assert data['status']=='COMPLETE_TRAINING_NO_EVALUATION'
    telemetry=f.training_telemetry(f.HB156_LOGS/'fit-attempt-1',data['history'])
    assert len(telemetry['child_epoch_metrics'])==16 and telemetry['child_epoch_summary_agree']


# ---- v1.1.0 additions (Claude builder, 2026-10-15) ----

def test_v111_output_and_cache_locations_and_tag():
    assert f.VERSION=='1.1.1' and f.TAG=='FORENSICS_LENSES_1015'
    assert f.WORK.name=='v111' and f.CACHE.name=='analysis' and f.WORK.parent==f.CACHE.parent
    assert f.GROUPS[0]==('precision-calibration','precision') and len(f.GROUPS)==4
    assert set(f.LENS_PHASES)=={p for g in f.GROUPS[1:3] for p in g}
    assert f.PREREQUISITE['full']=='calibration'  # paired group no longer waits for the precision group


def test_stage_view_selects_stages_infinity_and_largest():
    def row(m,inf=0,bad=0,lim=0): return dict(max_abs_finite=m,nonfinite=bad,nan=0,inf=inf,at_or_above_fp16_limit=lim)
    stages={'logits':row(9.),'backbone':row(100.),'backbone.stem':row(5.),'backbone.stages.0':row(65504.,lim=3),
            'backbone.stages.3':row(7.,inf=2,bad=2),'backbone.stages.0.blocks.1.mlp':row(70000.),'att':row(1.)}
    v=f.stage_view(stages)
    assert set(v['stages'])=={'logits','backbone','backbone.stem','backbone.stages.0','backbone.stages.3','att'}
    assert v['stages']['backbone.stages.3']['max_abs']=='Infinity' and v['stages']['backbone.stages.3']['fraction_of_fp16_max'] is None
    assert v['stages']['backbone.stages.0']['fraction_of_fp16_max']==1.0
    assert v['largest_module']=='backbone.stages.0.blocks.1.mlp' and v['largest_module_max_abs']==70000.


def test_precision_summary_reports_stage_view():
    ids=[str(i) for i in range(16)]
    result=f.precision_summary(precision_rows(ids,delta=.0),ids,'cuda:0')
    for arm in f.PRECISION_ARMS:
        for mode in f.PRECISIONS:
            assert 'stage_view' in result['models'][arm]['modes'][mode]


def test_cap_memory_is_noop_on_cpu():
    import torch
    f.cap_memory('cpu')
    assert not torch.cuda.is_initialized()


def test_completed_lens_phase_binds_plan_and_journals(monkeypatch,tmp_path):
    monkeypatch.setattr(f,'WORK',tmp_path)
    monkeypatch.setattr(f,'binding',lambda:{'b':'1'})
    seen=[]
    monkeypatch.setattr(f,'memory',lambda desktop,peaks: seen.append(peaks))
    folder=tmp_path/'gpu'/'mil-full'
    plan=dict(phase='mil-full',tasks=[dict(key='a'),dict(key='b')],hb162_unavailable_reason='mil phase')
    f.io.atomic_json(folder/'plan.json',plan)
    pins={}
    for name in ('a.json','b.json'):
        f.io.atomic_json(folder/'journal'/name,dict(peak_reserved_bytes=5))
        pins[name]=f.sha(folder/'journal'/name)
    f.io.atomic_json(folder/'controller.json',dict(exit_code=0,cleanup_errors=[],binding={'b':'1'},
        journals=pins,desktop={},plan=plan))
    assert f.completed_phase('mil-full')['exit_code']==0 and seen==[[5,5]]
    f.io.atomic_json(folder/'plan.json',dict(plan,tasks=plan['tasks'][:1]))
    with pytest.raises(ValueError,match='plan changed'): f.completed_phase('mil-full')
    f.io.atomic_json(folder/'plan.json',plan)
    f.io.atomic_json(folder/'journal'/'a.json',dict(peak_reserved_bytes=6))
    with pytest.raises(ValueError,match='journal changed'): f.completed_phase('mil-full')


def test_run_all_continues_after_failed_group(monkeypatch,tmp_path):
    monkeypatch.setattr(f,'WORK',tmp_path)
    monkeypatch.setattr(f,'admit',lambda dry=False:None)
    monkeypatch.setattr(f,'require_gpu_job',lambda:None)
    import tools.c85r7_flipall_timing as timing
    monkeypatch.setattr(timing,'desktop_snapshot',lambda:{})
    ran=[]
    def supervise(phase,desktop):
        ran.append(phase)
        if phase=='precision': raise ValueError('synthetic precision failure')
    monkeypatch.setattr(f,'supervise',supervise)
    monkeypatch.setattr(f,'completed_phase',lambda phase,count=None:{})
    monkeypatch.setattr(f,'evaluate',lambda:dict(status='RESULT_REVIEW_REQUIRED'))
    with pytest.raises(ValueError,match='phase groups failed'): f.run_all()
    assert ran==[p for g in f.GROUPS for p in g]
    summary=f.read(tmp_path/'gpu/run-summary.json')
    assert summary['groups']['precision'].startswith('FAILED') and summary['groups']['full']=='COMPLETE'
    assert summary['evaluation']['status']=='RESULT_REVIEW_REQUIRED'


def test_review_requires_scope_and_lens_bindings(monkeypatch,tmp_path):
    monkeypatch.setattr(f,'WORK',tmp_path)
    for name in ('registration.json','cpu-smoke.json','lenses/registration.json','lenses/cpu-smoke.json'):
        f.io.atomic_json(tmp_path/name,{'n':name})
    evidence=tmp_path/'REVIEW.md'; evidence.write_text('x')
    review=dict(verdict='PASS',independent=True,reviewer='fable',scope=f.TAG,
        registration_sha256=f.sha(tmp_path/'registration.json'),cpu_smoke_sha256=f.sha(tmp_path/'cpu-smoke.json'),
        lens_registration_sha256=f.sha(tmp_path/'lenses/registration.json'),
        lens_cpu_smoke_sha256=f.sha(tmp_path/'lenses/cpu-smoke.json'),
        evidence_path=str(evidence),evidence_sha256=f.sha(evidence))
    monkeypatch.setattr(f,'ROOT',tmp_path)
    f.check_review(review)
    for key,bad in (('scope','FORENSICS_1015'),('lens_cpu_smoke_sha256','0'*64),('reviewer','claude-forensics-lens-builder')):
        with pytest.raises(ValueError): f.check_review(dict(review,**{key:bad}))


def test_retry_routes_lens_registration_without_editing_shared_defaults(monkeypatch,tmp_path):
    from tools import lens_completion_1015_v1 as L
    monkeypatch.setattr(L,'OUT',L.OUT)
    monkeypatch.setattr(L,'LWORK',L.LWORK)
    monkeypatch.setattr(f,'WORK',tmp_path/'v111')
    module=f.lens()
    assert module is L and L.OUT==f.WORK and L.LWORK==f.WORK/'lenses'
    monkeypatch.setattr(L,'registration',lambda:dict(sources={'new':'pin'}))
    L.io.atomic_json(L.LWORK/'registration.json',dict(sources={'old':'pin'}))
    with pytest.raises(ValueError,match='stale'): L.registered()
    L.io.atomic_json(L.LWORK/'registration.json',dict(sources={'new':'pin'}))
    assert L.registered()['sources']=={'new':'pin'}


@pytest.mark.parametrize('command',['register','validate','cpu-smoke'])
def test_lens_cli_routes_into_retry_cpu_commands(monkeypatch,tmp_path,command):
    import types
    seen=[]
    monkeypatch.setattr(f,'WORK',tmp_path)
    monkeypatch.setattr(f,'lens',lambda:types.SimpleNamespace(main=lambda args:seen.append(args)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    f.main(['lens-'+command])
    assert seen==[[command]]
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'


def test_retry_queue_tolerates_failed_hb162_but_remains_one_serial_job(monkeypatch,tmp_path):
    monkeypatch.setattr(f,'WORK',tmp_path)
    monkeypatch.setattr(f,'registered',lambda:{'ids':list(range(58))})
    monkeypatch.setattr(f,'payload',lambda *args:None)
    assert f.prepare()['queue_started'] is False
    queue=f.read(tmp_path/'queue.json')
    assert len(queue)==1 and queue[0]['tag']==f.TAG
    assert queue[0]['after']=='HB162_SAM_F0' and queue[0]['continue_on_failure'] is True
    assert queue[0]['command'][-1]=='run'
