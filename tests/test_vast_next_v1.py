"""v1.0.0 CPU-only next-job contracts; synthetic outputs never qualify a fit."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
os.environ['CUDA_VISIBLE_DEVICES']=''
from tools.l1_rco_k1_next_v1 import backend, ROOT, MODES
from tools import l1_rco_k1_all_v1 as standard
from tools import l1_rco_k1_fullbag_v1 as fullbag
from tools import l1_rco_k1_nextlaunch_v1 as launch
from tools import l1_rco_k1_nexthost_v1 as host
from tools import k1_fold_dataset_v1 as assets
from tools import k1n_stage_v1 as stage
from test_fullbag_post_v1 import waiter_fixture

@pytest.mark.parametrize('mode',MODES)
def test_exact_source_and_recipe(mode):
 k=backend(mode); reg=k.expected_registration()
 raw=assets.SERVING.read_text(encoding='utf-8'); actual=k.derive_trainer(raw)
 assert len(reg['train'])==4167 and not reg['holdout']
 assert reg['recipe']['total_steps']==20830 and reg['recipe']['warm_steps']==2083
 cmd=k.trainer_argv(k.CACHE,reg)
 assert reg['model_seed']==MODES[mode][1]
 if mode=='all_seed1':
  restored=actual.replace('tools.l1_rco_k1_all_seed1_v1','tools.l1_rco_k1_all_v1').replace('seed_base == 1','seed_base == 0')
  assert restored==standard.derive_trainer(raw)
  assert cmd[cmd.index('--seed-base')+1]=='1'
  assert {n:v for n,v in reg['recipe'].items() if n not in ('seed_base','unchanged')}=={n:v for n,v in standard.expected_registration()['recipe'].items() if n!='unchanged'}
 else:
  restored=actual.replace('tools.l1_rco_k1_fullbag_all_v1','tools.l1_rco_k1_fullbag_v1')
  restored=restored.replace('assert not gold  # all labelled studies; no held-out fold','assert not gold and all(pool_fold != 0)')
  restored=restored.replace('tools.l1_rco_k1_allstate_v1','tools.l1_rco_k1_foldstate_v1')
  restored=restored.replace("guard_constant=a.l1_guard_constant, fold='all')",'guard_constant=a.l1_guard_constant, fold=0)')
  restored=restored.replace('            tr = np.arange(len(pool))  # all labelled studies','            tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)')
  assert restored==fullbag.derive_trainer(raw)
  assert cmd[cmd.index('--train-windows')+1]=='640'
  changes={'model_fold','only_folds','steps_per_epoch','total_steps','warm_steps','bias','unchanged'}
  parent=fullbag.expected_registration()
  assert {n:v for n,v in reg['recipe'].items() if n not in changes}=={n:v for n,v in parent['recipe'].items() if n not in changes}
 assert standard.WORK==standard.ALL_BASE/'all'
 assert fullbag.k.WORK==fullbag.WORK
 assert k.command()[2]=='run' and k.command(action='infer')[2]=='infer'


def fixture_export(root,mode,monkeypatch):
 k=backend(mode); reg=k.expected_registration(); original_argv=k.trainer_argv(k.CACHE,reg)
 hashes=host.prepared_hashes(reg,k)
 monkeypatch.setattr(k,'WORK',root); monkeypatch.setattr(k,'OUT',root/'fit')
 model=k.OUT/'models/E_prior_s0_fall.pt'; model.parent.mkdir(parents=True)
 model.write_bytes(b'SYNTHETIC TEST FIXTURE NOT A TRAINED CHECKPOINT')
 k.write(root/'registration.json',reg)
 k.write(root/'prepare.json',dict(contract_id=reg['contract_id'],**hashes,train_count=4167,holdout_training_rows=0,argv=original_argv,cache=str(k.CACHE.resolve())))
 k.write(k.OUT/'fit-complete.json',dict(status='COMPLETE',contract_id=reg['contract_id'],model_fold='all',model_sha256=k.sha(model),epochs=10,optimizer_steps=20830,train_studies=4167,selection='final EMA',own_holdout_evaluation=False))
 k.write(k.OUT/k.EPOCHS,dict(committed_epochs=10,log=[.01]*10))
 with (k.OUT/'train-guard.csv').open('w',newline='') as stream:
  w=csv.writer(stream); w.writerow([k.s.UID]+k.s.LABELS+['status','model_fold'])
  for i,uid in enumerate(k.guard_ids(reg)): w.writerow([uid]+[.1+i*.01]*12+['ok','all'])
 guard=k.verify_prediction_guard(reg,k.OUT)
 k.write(k.OUT/'result.json',dict(status='COMPLETE',contract_id=reg['contract_id'],model_fold='all',model_sha256=k.sha(model),holdout_rows=0,training_guard_rows=32,training_guard_sha256=k.sha(k.OUT/'train-guard.csv'),prediction_guard=guard,cpu_test=False,clean_oof=False,own_holdout_evaluation=False,selection='final EMA',exposure='EXPOSED_EXPLORATORY'))
 k.write(k.OUT/'runtime.json',dict(test_fixture=True))
 k.export(reg)
 return k,reg

@pytest.mark.parametrize('mode',MODES)
def test_package_host_roundtrip_and_cli(tmp_path,monkeypatch,mode):
 k,reg=fixture_export(tmp_path/'export',mode,monkeypatch)
 path=k.WORK/'EXPORT.json'; slug='taeyangg4/'+('knee-k1-v3gfball-v1-assets' if mode=='fullbag_all' else 'knee-k1-v3gall-s1-v1-assets')
 for cmd in ([sys.executable,'-B','-m','tools.k1_fold_dataset_v1','verify-export','--export',str(path)],
             [sys.executable,'-B','-m','tools.k1_fold_dataset_v1','package','--export',str(path),'--output',str(tmp_path/'dataset'),'--dataset-id',slug],
             [sys.executable,'-B','-m','tools.k1_fold_dataset_v1','verify-dataset','--dataset',str(tmp_path/'dataset')]):
  result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
  assert result.returncode==0,result.stderr
 assert assets.verify_dataset(tmp_path/'dataset')['id']==MODES[mode][0]
 manifest=k.read(tmp_path/'dataset/asset-manifest.json')
 assert manifest['model_seed']==MODES[mode][1]
 template='c5-fball-dropo-p25-v10.template.json' if mode=='fullbag_all' else 'c5-alls1-dropo-p25-v10.template.json'
 fill=subprocess.run([sys.executable,'-B',str(ROOT/'.local/scratch/slot-b/submit-1021/fill_template_v1.py'),
      str(ROOT/'configs/k1n'/template),str(tmp_path/'filled.json'),'test-owner/next-fixture','1.0.0-fixture',
      MODES[mode][0]+'='+str(tmp_path/'dataset')],cwd=ROOT,capture_output=True,text=True)
 assert fill.returncode==0,fill.stderr
 assert k.read(tmp_path/'filled.json')['models'][0]['id']==MODES[mode][0]
 metadata=k.read(tmp_path/'dataset/dataset-metadata.json'); saved=copy.deepcopy(metadata)
 metadata['id']='test-owner/wrong';metadata['title']='wrong';k.write(tmp_path/'dataset/dataset-metadata.json',metadata)
 with pytest.raises(ValueError,match='identity'):assets.verify_dataset(tmp_path/'dataset')
 k.write(tmp_path/'dataset/dataset-metadata.json',saved)
 # Tiny host arithmetic/path differences allowed; recipe drift rejected even if resealed.
 reg['collapse_guard']['constant_loss']+=1e-14; reg['collapse_guard']['threshold']+=1e-14
 reg['contract_id']=k.digest({n:v for n,v in reg.items() if n!='contract_id'})
 k.write(k.WORK/'registration.json',reg)
 for name in ('prepare.json','fit/result.json','fit/fit-complete.json'):
  r=k.read(k.WORK/name);r['contract_id']=reg['contract_id']
  if name=='prepare.json':
   r['argv'][0]='/usr/bin/python3'; r['argv'][1]='/root/work/trainer.py'
   i=r['argv'].index('--l1-guard-constant');r['argv'][i+1]=repr(reg['collapse_guard']['constant_loss'])
  k.write(k.WORK/name,r)
 k.export(reg); assert assets.verify_export(path)[0]['fold']=='all'
 reg['recipe']['lr']*=2;reg['contract_id']=k.digest({n:v for n,v in reg.items() if n!='contract_id'})
 k.write(k.WORK/'registration.json',reg)
 for name in ('prepare.json','fit/result.json','fit/fit-complete.json'):
  r=k.read(k.WORK/name);r['contract_id']=reg['contract_id'];k.write(k.WORK/name,r)
 k.export(reg)
 with pytest.raises(ValueError,match='scientific identity'): assets.verify_export(path)

@pytest.mark.parametrize('mode',MODES)
def test_templates_and_generator(mode):
 name='c5-fball-dropo-p25-v10.template.json' if mode=='fullbag_all' else 'c5-alls1-dropo-p25-v10.template.json'
 c=assets.read(ROOT/'configs/k1n'/name);parent=assets.read(ROOT/'configs/k1n/c5-dropo-p25-v9.json')
 assert stage.VERSION=='1.8.2' and c['k1_runtime_version']==10
 assert len(c['models'])==5 and c['models'][1:]==parent['models'][1:]
 assert c['models'][0]['id']==MODES[mode][0] and all('weight' not in m for m in c['models'])
 assert {n:v for n,v in c.items() if n not in ('models','kernel_id','release_version','k1_runtime_version')}=={n:v for n,v in parent.items() if n not in ('models','kernel_id','release_version','k1_runtime_version')}
 c['kernel_id']='test-owner/next-'+mode.replace('_','-');c['models'][0]['checkpoint_sha256']=hashlib.sha256(b'synthetic checkpoint').hexdigest();c['models'][0]['manifest_sha256']=hashlib.sha256(b'synthetic manifest').hexdigest()
 for diagnostic in (False,True):
  outputs,registration=stage.artifacts(c,diagnostic)
  assert outputs and registration['kernel_sha256']

@pytest.mark.parametrize('mode',MODES)
def test_waiter_selection_fail_closed(tmp_path,monkeypatch,mode):
 monkeypatch.setattr(launch,'ROOT',tmp_path);monkeypatch.setattr(launch,'LEDGER',tmp_path/'selected-once.json')
 k=backend(mode);monkeypatch.setattr(launch,'backend',lambda _:k)
 waiter_fixture(tmp_path,'STARTED')
 assert launch.execute(mode,tmp_path/'decision.json',True)['status']=='BLOCKED'
 waiter_fixture(tmp_path,'SUCCESS')
 gate=tmp_path/'gate.json';report=tmp_path/'review.md';report.write_text('synthetic reviewer fixture')
 k.write(gate,dict(holdout_rows=977,status='SCREEN_PASS_PENDING_REVIEW'))
 d=dict(schema='vast-next-decision/v1',selected=mode,owner='B',review_verdict='PASS',builder='builder',reviewer='verifier',gate='gate.json',gate_sha256=k.sha(gate),review_report='review.md',review_report_sha256=k.sha(report))
 path=tmp_path/'decision.json';k.write(path,d)
 assert launch.decision(k,path)==d
 k.write(gate,dict(holdout_rows=977,status='HOLD'));d['gate_sha256']=k.sha(gate);k.write(path,d)
 if mode=='fullbag_all':
  with pytest.raises(ValueError,match='did not pass'):launch.decision(k,path)
 else: assert launch.decision(k,path)==d
 launch.LEDGER.write_text('{}')
 assert 'already selected' in launch.execute(mode,path,True)['reason']
 d['reviewer']=d['builder'];k.write(path,d)
 with pytest.raises(ValueError):launch.decision(k,path)
