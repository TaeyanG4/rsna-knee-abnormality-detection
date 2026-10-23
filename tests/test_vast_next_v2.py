"""v2.0.0 additive local seed1 / Vast seed2 CPU contracts; synthetic export only."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from tools.l1_rco_k1_next_v2 import backend, ROOT, MODES, LOCAL_CACHE
from tools import l1_rco_k1_all_v1 as standard
from tools import l1_rco_k1_nextlaunch_v2 as launch
from tools import k1_fold_dataset_v1 as assets
from tools import k1n_stage_v1 as stage
import test_vast_next_v1 as previous


@pytest.mark.parametrize('mode', MODES)
def test_exact_seed_only(mode):
    k=backend(mode); r=k.expected_registration(); parent=standard.expected_registration()
    raw=assets.SERVING.read_text(encoding='utf-8')
    restored=k.derive_trainer(raw).replace('tools.l1_rco_k1_'+mode+'_v1','tools.l1_rco_k1_all_v1').replace(f'seed_base == {MODES[mode][1]}','seed_base == 0')
    assert restored == standard.derive_trainer(raw)
    assert r['train']==parent['train'] and r['holdout']==[]
    assert r['recipe']['total_steps']==20830 and r['recipe']['warm_steps']==2083
    assert {n:v for n,v in r['recipe'].items() if n not in ('seed_base','unchanged')}=={n:v for n,v in parent['recipe'].items() if n!='unchanged'}
    argv=k.trainer_argv(k.CACHE,r)
    assert argv[argv.index('--seed-base')+1]==str(MODES[mode][1])
    assert 'gpu_job.py' in k.launch_command()[1] and k.TAG in k.launch_command()
    if mode=='local_all_seed1':
        assert k.CACHE==LOCAL_CACHE and k.CACHE.is_absolute()
        assert r['execution']['meta_eps'] is False


@pytest.mark.parametrize('mode', MODES)
def test_package_cli_and_fill(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(previous,'backend',backend)
    k,r=previous.fixture_export(tmp_path/'export',mode,monkeypatch)
    seed=MODES[mode][1]; dataset=tmp_path/'dataset'; export=k.WORK/'EXPORT.json'
    commands=[['verify-export','--export',str(export)],
              ['package','--export',str(export),'--output',str(dataset),'--dataset-id',f'taeyangg4/knee-k1-v3gall-s{seed}-v1-assets'],
              ['verify-dataset','--dataset',str(dataset)]]
    for cmd in commands:
        p=subprocess.run([sys.executable,'-B','-m','tools.k1_fold_dataset_v1',*cmd],cwd=ROOT,capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
    assert assets.verify_dataset(dataset)['id']==MODES[mode][0]
    template=ROOT/f'configs/k1n/c5-alls{seed}-dropo-p25-v10.template.json'
    p=subprocess.run([sys.executable,'-B',str(ROOT/'.local/scratch/slot-b/submit-1021/fill_template_v1.py'),str(template),str(tmp_path/'filled.json'),'test-owner/seed-fixture','1.0.0-fixture',MODES[mode][0]+'='+str(dataset)],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert k.read(tmp_path/'filled.json')['models'][0]['id']==MODES[mode][0]
    with pytest.raises(ValueError,match='producer seed'):
        assets.next_backend('standard',3,mode)


@pytest.mark.parametrize('name,dual',[('c5-alls2-dropo-p25-v10.template.json',False),('c5-alls12-dropo-p25-v10.template.json',True)])
def test_templates_generator(name,dual):
    c=assets.read(ROOT/'configs/k1n'/name); parent=assets.read(ROOT/'configs/k1n/c5-dropo-p25-v9.json')
    assert stage.VERSION=='1.8.2' and c['k1_runtime_version']==10
    n=2 if dual else 1
    assert c['models'][n:]==parent['models'][1:] and len(c['models'])==4+n
    assert all(m['id']!='k1_v2' for m in c['models'])
    if dual: assert c['member_weights']==[.1,.1,.2,.2,.2,.2]
    else: assert 'member_weights' not in c
    for m in c['models'][:n]:
        m['checkpoint_sha256']=hashlib.sha256(('fixture-'+m['id']).encode()).hexdigest()
        m['manifest_sha256']=hashlib.sha256(('manifest-'+m['id']).encode()).hexdigest()
    c['kernel_id']='test-owner/fixture'
    for diagnostic in (False,True):
        outputs,r=stage.artifacts(c,diagnostic)
        assert outputs and r['kernel_sha256']


def test_local_reference_without_shards_or_meta(tmp_path,monkeypatch):
    import tools.l1_rco_k1_next_v2 as factory
    k=backend('local_all_seed1'); cache=tmp_path/'cache'; cache.mkdir()
    receipt={'hashes':{'px.u8.npy':'x','meta.f32.npy':'m','index.json':'i'}}
    k.write(cache/'receipt.json',receipt)
    monkeypatch.setattr(factory,'LOCAL_CACHE',cache)
    monkeypatch.setattr(k.parent,'CACHE',cache)
    monkeypatch.setattr(k,'complete_cache',lambda reg,c: {'studies':4167})
    evidence=k.next_evidence(cache)
    assert evidence['local_reference_bindings']['meta_eps_required'] is False
    assert not (cache/'shards').exists() and not (cache/'H224-PARITY.json').exists()
    with pytest.raises(ValueError,match='F: cache'): k.next_evidence(tmp_path/'wrong')
    monkeypatch.setattr(k,'WORK',tmp_path/'work')
    assert k.admit(k.expected_registration(),cache,tmp_path/'absent-review')['status']=='BLOCKED'


def test_waiter_requires_fail_and_shared_ledger(tmp_path,monkeypatch):
    from tools import l1_rco_k1_nextlaunch_v1 as old
    assert old.LEDGER==launch.LEDGER
    monkeypatch.setattr(launch,'ROOT',tmp_path); monkeypatch.setattr(launch,'LEDGER',tmp_path/'selected-once.json')
    k=backend('all_seed2'); monkeypatch.setattr(launch,'backend',lambda _:k)
    report=tmp_path/'review.md';report.write_text('synthetic independent review')
    gate=tmp_path/'gate.json';k.write(gate,dict(holdout_rows=977,status='HOLD'))
    d=dict(schema='vast-next-decision/v1',selected='all_seed2',owner='B',review_verdict='PASS',builder='builder',reviewer='reviewer',gate='gate.json',gate_sha256=k.sha(gate),review_report='review.md',review_report_sha256=k.sha(report))
    path=tmp_path/'decision.json';k.write(path,d)
    assert launch.decision(k,path)==d
    k.write(gate,dict(holdout_rows=977,status='SCREEN_PASS_PENDING_REVIEW'));d['gate_sha256']=k.sha(gate);k.write(path,d)
    with pytest.raises(ValueError,match='gate-FAIL'): launch.decision(k,path)
    launch.LEDGER.write_text('{}')
    assert 'already selected' in launch.execute('all_seed2',path,True)['reason']
    with pytest.raises(ValueError,match='Vast seed2 only'):launch.execute('local_all_seed1',path,True)


def test_local_vram_gate():
    from tools.l1_rco_k1_next_v2 import local_memory_gate
    k=backend('local_all_seed1')
    safe=dict(desktop_bytes=2*2**30,total_bytes=16*2**30)
    local_memory_gate(k,safe)
    assert safe['f1_peak_reserved']==9554624512
    with pytest.raises(ValueError,match='VRAM'):
        local_memory_gate(k,dict(desktop_bytes=6*2**30,total_bytes=16*2**30))


def test_dual_seed_fill_cli(tmp_path,monkeypatch):
    monkeypatch.setattr(previous,'backend',backend)
    pairs=[]
    for mode in MODES:
        k,r=previous.fixture_export(tmp_path/mode,mode,monkeypatch)
        seed=MODES[mode][1]; dataset=tmp_path/f'dataset{seed}'
        p=subprocess.run([sys.executable,'-B','-m','tools.k1_fold_dataset_v1','package','--export',str(k.WORK/'EXPORT.json'),'--output',str(dataset),'--dataset-id',f'taeyangg4/knee-k1-v3gall-s{seed}-v1-assets'],cwd=ROOT,capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
        pairs.append(MODES[mode][0]+'='+str(dataset))
    p=subprocess.run([sys.executable,'-B',str(ROOT/'.local/scratch/slot-b/submit-1021/fill_template_v1.py'),str(ROOT/'configs/k1n/c5-alls12-dropo-p25-v10.template.json'),str(tmp_path/'filled.json'),'test-owner/dual-seed-fixture','1.0.0-fixture',*pairs],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    filled=json.loads((tmp_path/'filled.json').read_text())
    assert filled['member_weights']==[.1,.1,.2,.2,.2,.2]
    assert [m['id'] for m in filled['models'][:2]]==['k1_v3g_all_s1','k1_v3g_all_s2']
