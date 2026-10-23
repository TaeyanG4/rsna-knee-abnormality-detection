"""H204 metadata/synthetic CPU checks; no model, GPU or probability inputs."""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tools import hb204_c4_cache_v103 as m


def forbidden(*args, **kwargs):
    raise AssertionError('forbidden old admission replay or execution')


def test_historical_memory_is_result_only(monkeypatch):
    for name in ('memory_admission','authorize_memory','checked_gate','validate','review'):
        monkeypatch.setattr(m.old,name,forbidden)
    monkeypatch.setattr(m.old.gate,'verify_receipt',forbidden)
    inputs=m.old.gate.Inputs(m.ROOT)
    reg,card,mem=m.historical(inputs)
    assert len(card['models'])==len(mem['raw_receipts'])==5
    assert card['workload']==290 and mem['actual_full_shape_both_views']
    assert all(not p.endswith(('.pt','.bin','.npz','.npy','.csv')) for p in inputs.consumed)
    assert not any('consumed_sha256' in p for p in inputs.consumed)


@pytest.mark.parametrize('kind',[
    'model_order','roster','shape','views','peak','nan','cleanup','source','worker_count','cache_fake','ticket','job'])
def test_historical_bad_result_rejected(monkeypatch,kind):
    actual=m.old.gate.Inputs.json
    def changed(self,item):
        value=copy.deepcopy(actual(self,item)); p=item['path']
        if p.endswith('gold58-launch.json'):
            if kind=='model_order':value['models'].reverse()
            if kind=='roster':value['corpus']['ids'].reverse()
        if p.endswith('memory-run/model-0.json'):
            if kind=='shape':value['input_shape']=[24,3,224,224]
            if kind=='views':value['views']=1
            if kind=='peak':value['peak_reserved_bytes']+=1
            if kind=='nan':value['seconds']=float('nan')
            if kind=='source':value['source_sha256']='0'*64
        if p.endswith('memory-run/process.json') and kind=='cleanup':value['owned_cleanup_complete']=False
        if p.endswith('memory-run/worker.json'):
            if kind=='worker_count':value['workload']=4
            if kind=='cache_fake':value['npz_sha256']='0'*64
        if p.endswith('memory-run/ticket.json') and kind=='ticket':value['admission']['sha256']='0'*64
        if p.endswith('memory-run/bound-job.json') and kind=='job':value['command']=['incorrect']
        return value
    monkeypatch.setattr(m.old.gate.Inputs,'json',changed)
    with pytest.raises(ValueError):m.historical(m.old.gate.Inputs(m.ROOT))


def test_private_runtime_keeps_science_supervisor_and_old_globals():
    oldrel,oldself,oldtags=m.old.REL,m.old.SELF,copy.deepcopy(m.old.TAGS)
    rt=m.runtime_module()
    for name in ('worker','run','finish','supervise','memory_ok','check_job','finite_cache','atomic_npz'):
        assert inspect.getsource(getattr(rt,name))==inspect.getsource(getattr(m.old,name))
    assert inspect.getsource(rt.Progress.check)==inspect.getsource(m.old.Progress.check)
    assert inspect.getsource(rt.Progress.__init__)==inspect.getsource(m.old.Progress.__init__)
    assert rt.REL==m.REL and rt.SELF==m.SELF and rt.TAGS=={'cache':m.TAG}
    assert rt.command('cache')==m.command()
    assert (m.old.REL,m.old.SELF,m.old.TAGS)==(oldrel,oldself,oldtags)
    with pytest.raises(ValueError):rt.command('memory')


def test_static_registration_reads_no_models_or_arrays(monkeypatch):
    actual=m.old.gate.Inputs.file
    def checked(self,item):
        assert not item['path'].endswith(('.pt','.bin','.npy','.npz','.csv'))
        return actual(self,item)
    monkeypatch.setattr(m.old.gate.Inputs,'file',checked)
    reg,card,_=m.validate()
    assert reg['execution_authorized'] is False and reg['qualified_for_submission'] is False
    assert card['expected_shape']==[5,58,12]


@pytest.mark.parametrize('kind',['queue','source','kernel','tag','workload','fixture'])
def test_static_drift_rejected(monkeypatch,kind):
    actual=m.old.gate.Inputs.json
    def changed(self,item):
        value=copy.deepcopy(actual(self,item))
        if item['path']==(m.REL/'registration.json').as_posix():
            if kind=='queue':value['queue'][0]['command'][-1]='measure'
            if kind=='source':value['source_sha256'][m.SELF]='0'*64
            if kind=='kernel':value['kernel_sha256']='0'*64
            if kind=='tag':value['tag']=m.old.TAGS['cache']
            if kind=='workload':value['workload']=5
            if kind=='fixture':value['fixture_only']=True
        return value
    monkeypatch.setattr(m.old.gate.Inputs,'json',changed)
    with pytest.raises(ValueError):m.validate()


def test_execution_rechecks_weight_and_corpus_pins_without_reading_them(monkeypatch):
    actual=m.old.gate.Inputs.file; seen=[]
    def checked(self,item):
        if item['path'].endswith('.pt'):
            seen.append(item); raise ValueError('TEST stop before checkpoint bytes')
        return actual(self,item)
    monkeypatch.setattr(m.old.gate.Inputs,'file',checked)
    with pytest.raises(ValueError,match='TEST stop'):m.validate(execution=True)
    assert len(seen)==1 and seen[0]['sha256']!='0'*64


def test_missing_review_blocks_without_execution(monkeypatch):
    monkeypatch.setattr(m,'memory_review',lambda _: (_ for _ in ()).throw(ValueError('TEST missing actual review')))
    monkeypatch.setattr(m,'runtime_module',forbidden)
    with pytest.raises(ValueError,match='missing actual review'):m.authorize()


@pytest.mark.parametrize('verdict,agent,subjects',[
    ('VERDICT: FAIL','/root/distinct',True),('VERDICT: PASS',m.old.BUILDER,True),
    ('VERDICT: PASS','/root/distinct',False),('VERDICT: PASS\nVERDICT: FAIL','/root/distinct',True)])
def test_result_review_must_be_distinct_exact_pass(tmp_path,monkeypatch,verdict,agent,subjects):
    root=tmp_path; report=root/m.PREP/'H204-MEMORY-REVIEW.md'; report.parent.mkdir(parents=True)
    report.write_text(verdict+'\n'+(m.MEM_SHA+'\n'+m.REG_SHA if subjects else ''),encoding='utf-8')
    Path(str(report)+'.receipt.json').write_text('{}')
    monkeypatch.setattr(m.old.gate,'archive',lambda *a:dict(agent_id=agent))
    with pytest.raises(ValueError):m.memory_review(m.old.gate.Inputs(root))


def test_admission_cannot_replay_mutated_record(tmp_path,monkeypatch):
    path=tmp_path/m.REL/'CACHE-ADMISSION.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(state='TEST old')))
    monkeypatch.setattr(m,'authorize',lambda root:dict(state='TEST changed'))
    with pytest.raises(ValueError):m.admission(tmp_path)


def test_progress_keeps_completed_units_when_next_phase_overwrites():
    rt=m.runtime_module(); p=rt.Progress(0,dict(pid=1,create_time=1000),'token')
    for i in range(1,91):
        p.check(i*10,dict(pid=1,token='token',time=1000+i*10,sequence=4*i,units=2*i,phase='render'),None,1000+i*10)
    assert p.units==180 and p.last==900
    with pytest.raises(ValueError,match='900s'):p.check(1800,None,None,2800)


def test_no_memory_cli_and_no_gpu_or_torch_static_import():
    with pytest.raises(SystemExit):m.main(['measure'])
    cp=subprocess.run([sys.executable,'-B','-c',
        "from tools import hb204_c4_cache_v103 as m; import sys; assert 'torch' not in sys.modules; m.validate(); assert 'torch' not in sys.modules"],
        cwd=m.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),capture_output=True,text=True,timeout=15)
    assert cp.returncode==0,cp.stderr


def test_runtime_retains_no_op_on_wrong_environment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    with pytest.raises(ValueError,match='Windows physical0'):m.runtime_module().run('cache')


def test_fresh_registration_no_overwrite():
    with pytest.raises(ValueError,match='immutable output'):m.register()
