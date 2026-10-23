"""H202 source/identity/shape/failure/completion tests; no torch/model/GPU/data rows."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from tools import hb202_small58_v1 as m


def test_real_source_inventory_and_actual_embedded_functions():
    inv = m.inventory()
    entries = m.members(inv)
    ns, hashes = m.embedded(inv)
    assert len(hashes) == 31 and len(entries) == 10
    assert set(m.FUNCS).issubset(ns)
    assert 'torch' not in sys.modules
    assert all((e['seed'], e['fold']) == (s, f)
               for e, (s, f) in zip(entries, [(s, f) for s in (42, 43) for f in range(5)]))


def test_actual_release_metadata_rejects_wrong_spec():
    inv = m.inventory(); entries = m.members(inv); ns, _ = m.embedded(inv)
    e = entries[0]
    ck = dict(arch=e['arch'], res=224, lab=m.TARGETS, contract='raptor64_336_c140_s02',
              norm='imagenet', window_policy='within-slot-valid', fold=e['fold'], epoch=19,
              model_kind='finding_spatial_tokens_v1', input_policy='center3', k_eval=24,
              flip_tta='coronal_axial_horizontal_mean_probability',
              provenance={'spec_sha256':e['spec_sha256']})
    ns['inf2_small_meta'](ck, e)
    ck['provenance']['spec_sha256'] = '0'*64
    with pytest.raises(ValueError, match='spec binding'):
        ns['inf2_small_meta'](ck, e)


def test_static_preparation_never_imports_torch_or_reads_execution_assets(tmp_path, monkeypatch):
    # Audit real static preparation, including exact parent ID metadata, in isolated scratch.
    original = m.PREP
    for name in ('INVENTORY.json', 'LAUNCH-SPEC.md', 'WORKER-CONTRACT.md'):
        (tmp_path/name).write_bytes((original/name).read_bytes())
    monkeypatch.setattr(m, 'PREP', tmp_path)
    original_open = Path.open
    def safe_open(path, *args, **kwargs):
        name = str(path).replace('\\', '/')
        assert '/h202-small58/' not in name
        assert not name.endswith('.pt') and 'vols_shard' not in name
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', safe_open)
    result = m.prepare()
    assert result['status'] == 'PREPARED_NOT_REVIEWED_NOT_RUN'
    frozen = json.loads((tmp_path/'CONTRACT.json').read_text())
    assert frozen['recipe'] == m.recipe() and frozen['model_forwards'] == 0
    assert frozen['weight_hashes_measured'] is False
    assert 'torch' not in sys.modules


def test_source_pin_mutation_and_output_pin_rejected(tmp_path):
    path = tmp_path/'source.txt'; path.write_text('original')
    key = path.relative_to(m.ROOT).as_posix()
    pins = {key:m.pin(path)}
    m.verify_pins(pins)
    path.write_text('changed')
    with pytest.raises(ValueError, match='pin changed'):
        m.verify_pins(pins)
    with pytest.raises(ValueError, match='unsafe pin'):
        m.verify_pins({(m.RUN/'worker.json').relative_to(m.ROOT).as_posix():{}})


@pytest.mark.parametrize('change', ['same_builder', 'wrong_hash', 'fail', 'duplicate'])
def test_review_gate_fails_closed(change):
    text = 'Contract-SHA256: '+('a'*64)+'\nReviewer-Agent: /root/other_reviewer\nVerdict: PASS\n'
    assert m.parse_review(text, 'a'*64) == '/root/other_reviewer'
    if change == 'same_builder': text = text.replace('/root/other_reviewer',m.BUILDER)
    if change == 'wrong_hash': text = text.replace('a'*64,'b'*64)
    if change == 'fail': text = text.replace('Verdict: PASS','Verdict: FAIL')
    if change == 'duplicate': text += 'Verdict: PASS\n'
    with pytest.raises(ValueError): m.parse_review(text,'a'*64)


def job_fixture():
    own, parent = dict(pid=123,create_time=1.), dict(pid=122,create_time=.5)
    job = dict(schema='gpu-job/v2',phase='running',tag=m.TAG,cwd=str(m.ROOT),
        command=m.command(),command_sha256=m.jr.command_digest(m.command(),m.ROOT),
        child_identity=own,pid=parent['pid'],create_time=parent['create_time'],run_id='abc')
    args = [m.PYTHON,'-B',str(m.ROOT/'tools/gpu_job.py'),'--_run-job',m.TAG,'abc']
    return job,own,parent,args


@pytest.mark.parametrize('change', ['tag','command','pid','start','parent','dead'])
def test_job_identity_rejects_wrong_owner(change):
    job,own,parent,args = job_fixture()
    m.check_job(job,own,parent,m.command(),args,lambda x:'alive')
    alive = lambda x:'alive'
    if change == 'tag': job['tag']='OLD_CONSUMED_TAG'
    if change == 'command': job['command']=[m.PYTHON,'other.py']
    if change == 'pid': own=dict(own,pid=999)
    if change == 'start': own=dict(own,create_time=2.)
    if change == 'parent': args[-1]='another-run'
    if change == 'dead': alive=lambda x:'dead'
    with pytest.raises(ValueError): m.check_job(job,own,parent,m.command(),args,alive)


def test_gpu_app_filter_allows_desktop_but_not_other_ml():
    m.check_compute_apps('10, C:\\Windows\\explorer.exe\n11, chrome.exe\n123, python.exe', {123})
    with pytest.raises(ValueError, match='another Python'):
        m.check_compute_apps('222, C:\\Python312\\python.exe', {123})


@pytest.mark.parametrize('change', ['shape','dtype','nan','range'])
def test_invalid_probability_cache_rejected(change):
    p = np.full((10,58,12),.5,np.float32)
    m.validate_values(p)
    if change=='shape': p=p[:9]
    if change=='dtype': p=p.astype(np.float64)
    if change=='nan': p[0,0,0]=np.nan
    if change=='range': p[0,0,0]=1.1
    with pytest.raises(ValueError): m.validate_values(p)


def test_seed_pair_then_fold_is_exact_and_not_mean_of_ten():
    p = np.random.default_rng(72).random((10,58,12),dtype=np.float32)
    expected=((p[:5]+p[5:])/2).mean(0)
    assert np.array_equal(m.seed_pair_fold_mean(p),expected)
    assert not np.array_equal(expected,p.mean(0))
    assert expected.dtype==np.float32
    # The actual frozen candidate consumes exactly the same mean/rank order.
    inv=m.inventory(); ns,_=m.embedded(inv)
    scores=dict(small=p,l2=p.copy(),dino=p[:5].copy(),convnext=p[:5].copy())
    _,family=ns['aggregate'](scores)
    assert np.array_equal(family['dino'][:,:8],ns['rank'](expected)[:,:8])
    assert np.array_equal(family['dino'][:,8:],ns['rank'](p[:5].mean(0))[:,8:])


def test_output_npz_order_and_complete_dtype(tmp_path,monkeypatch):
    ids=[f'synthetic-{i:02}' for i in range(58)]
    monkeypatch.setattr(m,'ROSTER_SHA',m.uid_hash(ids))
    p=tmp_path/'p.npz'
    arrays=dict(ids=np.asarray(ids),targets=np.asarray(m.TARGETS),member_ids=np.asarray(m.MEMBERS),
                probabilities=np.full((10,58,12),.5,np.float32))
    np.savez(p,**arrays);m.check_npz(p,ids)
    arrays['member_ids']=arrays['member_ids'][::-1]
    np.savez(p,**arrays)
    with pytest.raises(ValueError,match='exact output'):m.check_npz(p,ids)


def completion_fixture():
    desktop=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_bytes=16376*2**20,desktop_bytes=2*2**30)
    env=dict(device='cuda:0',autocast_dtype='torch.float16',cudnn_tf32=False,matmul_tf32=False,
             cudnn_benchmark=False,cudnn_deterministic=True,deterministic=True,matmul_precision='highest')
    rows=[dict(member=n,studies=58,reference_index=0,reference_max_abs=0.,deleted=True,views=2,
               reference_peak_reserved_bytes=3*2**30,peak_reserved_bytes=3*2**30,
               peak_allocated_bytes=2*2**30,model_sha256='a'*64) for n in m.MEMBERS]
    w=dict(schema='h202-small58-worker/v1',version=m.VERSION,state='COMPLETE',recipe=m.recipe(),
           contract_sha256='contract',review={'sha256':'review'},output_sha256='npz',data_sha256='data',
           predictions=580,failures=0,fallback_count=0,pid=123,member_ids=m.MEMBERS,
           ordered_uid_sha256=m.ROSTER_SHA,targets=m.TARGETS,renderer_rows=58,model_measurements=rows,
           desktop=desktop,environment=env,exposure='EXPOSED_GOLD58',clean_oof=False,qualified_for_submission=False)
    p=dict(schema='h199-serving-process/v1',child_exit_code=0,owned_cleanup_complete=True,
           survivors=[],failure=None,child={'pid':123,'create_time':1.})
    return w,p


@pytest.mark.parametrize('change',['partial','fallback','survivor','exit','live','unknown','nan_parity',
    'bad_parity','memory','exposure','precision','hash','roster','incomplete_render','model_hash','unmeasured'])
def test_completion_rejects_failures(change):
    w,p=completion_fixture();alive=lambda x:'dead'
    entries=[{'sha256':'a'*64} for _ in range(10)]
    m.completed(w,p,'contract',{'sha256':'review'},'npz','data',alive,entries)
    if change=='partial':w['predictions']=579
    if change=='fallback':w['fallback_count']=1
    if change=='survivor':p['survivors']=[{'pid':22}]
    if change=='exit':p['child_exit_code']=1
    if change=='live':alive=lambda x:'alive'
    if change=='unknown':alive=lambda x:'unknown'
    if change=='nan_parity':w['model_measurements'][0]['reference_max_abs']=float('nan')
    if change=='bad_parity':w['model_measurements'][0]['reference_max_abs']=1e-3
    if change=='memory':w['model_measurements'][0]['peak_reserved_bytes']=13*2**30
    if change=='exposure':w['clean_oof']=True
    if change=='precision':w['environment']['matmul_tf32']=True
    if change=='hash':w['output_sha256']='other'
    if change=='roster':w['ordered_uid_sha256']='other'
    if change=='incomplete_render':w['renderer_rows']=57
    if change=='model_hash':w['model_measurements'][0]['model_sha256']='b'*64
    if change=='unmeasured':w['model_measurements'][0]['reference_peak_reserved_bytes']=0
    with pytest.raises(ValueError):m.completed(w,p,'contract',{'sha256':'review'},'npz','data',alive,entries)


def test_measured_memory_12g_and_headroom_are_both_required():
    desktop=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_bytes=16376*2**20,desktop_bytes=1*2**30)
    m.infra().memory_ok(desktop,[12*2**30])
    with pytest.raises(ValueError):m.infra().memory_ok(dict(desktop,desktop_bytes=3*2**30),[12*2**30])


def test_watchdog_required_thresholds_reused_without_total_cap():
    from tools.hb155_watchdog_v1 import Policy
    policy=Policy(0,total=None)
    assert policy.idle==900 and policy.stall==600 and policy.collapse==1800 and policy.fraction==.25
    assert policy.check(900)=='no_heartbeat_15m'
    policy=Policy(0,total=None,healthy_rates={'prediction':1.})
    beat=lambda t,u:dict(phase='prediction',sequence=u,completed_at=t,event='commit',
                         detail=dict(progress_kind='prediction',cumulative_units=u))
    policy.check(0,beat(0,1))
    for t in range(100,1901,100):
        reason=policy.check(t,beat(t,1+t//100))
    assert reason=='throughput_below_25pct_30m'


def test_queue_is_one_candidate_normal_gpu_queue():
    from tools.gpu_queue import validate_plan
    q=validate_plan(m.queue())
    assert len(q)==1 and q[0]['tag']==m.TAG and q[0]['command']==m.command()
    assert q[0]['continue_on_failure'] is False


def test_raw_row_shape_failure_without_any_real_volume():
    volume=np.zeros((64,336,336),np.uint8);mask=np.ones(64,np.uint8);centers=[1]*36
    row=m.bind_row(volume,mask,centers,0)
    assert row['bytes']==64*336*336
    with pytest.raises(ValueError):m.bind_row(volume[:63],mask,centers,0)
    mask[1]=2
    with pytest.raises(ValueError):m.bind_row(volume,mask,centers,0)
