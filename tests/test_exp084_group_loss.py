"""CPU synthetic EXP084 contract checks. No images, Gold or training imports."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import exp084_group_loss as g


def synthetic_rows():
    return [{g.UID: str(i), 'component_id': f'{i // 2:064x}'} for i in range(40)]


def test_component_hash_rule_is_stable_and_grouped():
    import hashlib
    rows = synthetic_rows()
    ordered, assignment = g.frozen_split(rows)
    assert g.frozen_split(list(reversed(rows))) == (ordered, assignment)
    for r in rows:
        raw = hashlib.sha256(('RN_GROUP_SCREEN_V1:' + r['component_id']).encode('utf-8')).digest()
        assert assignment[r['component_id']] == ('val' if int.from_bytes(raw,'big')%5==2 else 'train')
    g.validate_membership([r[g.UID] for r in rows], ordered, rows)


def test_cross_component_duplicate_or_coverage_is_rejected():
    rows = synthetic_rows(); ordered, _ = g.frozen_split(rows)
    with pytest.raises(ValueError, match='duplicate'):
        g.frozen_split(rows + [rows[0]])
    corrupted = deepcopy(ordered)
    # Every synthetic component contains two UIDs; moving one crosses its boundary.
    moved = corrupted['train'].pop(); corrupted['val'] = sorted(corrupted['val']+[moved])
    with pytest.raises(ValueError, match='Component'):
        g.validate_membership([r[g.UID] for r in rows], corrupted, rows)
    with pytest.raises(ValueError, match='coverage'):
        g.validate_membership([], ordered, rows)


def test_weights_ignore_validation_and_preserve_float32():
    labels = {'train': np.tile(np.array([0,.5,1],np.float32),4), 'val':np.zeros(12,np.float32)}
    weight = g.class_weights(labels,['train'])
    assert weight.dtype == np.float32
    np.testing.assert_array_equal(weight,np.tile([10,1,1],4))
    p = {'loss':{'baseline_pos_weight':weight.tolist()}}
    for policy, actual in [('clipped_positive_term',weight),('ones_positive_term',np.ones(12,np.float32))]:
        a = {'loss_policy':policy,'actual_pos_weight':actual.tolist()}
        np.testing.assert_array_equal(g.loss_weights(weight,p,a),actual)
    labels['val'][:] = 1
    np.testing.assert_array_equal(g.class_weights(labels,['train']),weight)


def test_renderer_has_no_gold_scoring_swa_or_epoch_evaluation():
    original = (g.ROOT/'training/src/raptor_train.py').read_text(encoding='utf-8')
    candidate = (g.ROOT/g.ENTRYPOINT).read_text(encoding='utf-8')
    assert candidate == g.render_training_source(original)
    tree = ast.parse(candidate)
    assert not any(isinstance(n,ast.Name) and n.id in {'gold_labels','gold_ids'} for n in ast.walk(tree))
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert not any(n.func.id in {'macro_auc','exec','eval'} for n in calls)
    assert sum(n.func.id=='evaluate' for n in calls) == 1
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop = next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='ep')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='evaluate' for n in ast.walk(loop))
    old = ast.parse(original)
    old_main = next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    old_loop = next(n for n in old_main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='ep')
    old_steps = next(n for n in old_loop.body if isinstance(n,ast.For))
    new_steps = next(n for n in loop.body if isinstance(n,ast.For))
    assert ast.dump(old_steps) == ast.dump(new_steps)  # Actual optimization unchanged.


def test_import_is_passive_and_gold_blocker_denies_open(tmp_path):
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    script = "import importlib.util,sys; p=sys.argv[1]; s=importlib.util.spec_from_file_location('runner',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;assert 'raptor_common' not in sys.modules;from tools.exp084_group_loss import install_gold_blocker;from pathlib import Path;root=Path(sys.argv[2]);install_gold_blocker(root);open(root/'storage/competition/train.csv')"
    result = subprocess.run([sys.executable,'-c',script,str(g.ROOT/g.ENTRYPOINT),str(tmp_path)],cwd=g.ROOT,capture_output=True,text=True,env=env)
    assert result.returncode != 0 and 'Gold access forbidden' in result.stderr
    assert 'AssertionError' not in result.stderr and 'FileNotFoundError' not in result.stderr


def test_pretraining_rejects_initial_or_runtime_drift():
    p = {'training_code_sha256':{},'ordered_ids':{'train_sha256':'t','val_sha256':'v'},'corpus_manifest_sha256':'c',
        'expected_runtime':{'runtime':{},'eval_runtime':{}},'environment':{k:k for k in g.ENV_KEYS}}
    arm = {'initial_state_sha256':'init','loss_policy':'ones_positive_term'}
    prov = dict(code_sha256={},initial_state_sha256='init',train_ids_sha256='t',eval_ids_sha256='v',
        train_ids_ordered_sha256='t',eval_ids_ordered_sha256='v',corpus_manifest_sha256='c',checkpoint_selection='fixed-final',
        sampling='sha256(seed,epoch,uid,stream)-v1',gold_access='disabled',loss_policy='ones_positive_term',
        metric_access='deferred_until_all_three_complete',runtime={},eval_runtime={})
    g.validate_pretraining(prov,p['environment'],p,arm)
    bad = deepcopy(prov);bad['initial_state_sha256']='exposed-checkpoint'
    with pytest.raises(ValueError,match='provenance'):g.validate_pretraining(bad,p['environment'],p,arm)
    bad = deepcopy(prov);bad['runtime']={'fallback':True}
    with pytest.raises(ValueError,match='runtime'):g.validate_pretraining(bad,p['environment'],p,arm)


def test_completed_fit_fails_closed_on_skips_or_metric_exposure(tmp_path):
    p = {'expected_ids':{'train':8,'val':2}}
    arm = {'expected_args':{'epochs':2},'expected_attempts':2,'initial_state_sha256':'init'}
    summary = dict(selected_checkpoint='best',selected_epoch_1based=2,train=8,val=2,optimizer_updates=2,skipped_updates=0,
        optimizer_skip_fraction=0,weak_oof_macro_auc=None,metric_access='deferred_until_all_three_complete',provenance={'initial_state_sha256':'init'})
    for name in ('model_best.pt','oof_best.csv','provenance.json','history.json'):(tmp_path/name).write_text('test')
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    g.validate_completion(tmp_path,p,arm)
    summary['weak_oof_macro_auc']=.9
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Completion'):g.validate_completion(tmp_path,p,arm)


def supervisor_fixture(tmp_path, monkeypatch):
    p = {'schema':g.SCHEMA,'macro12_valid':True,'execution':{'total_time_limit_seconds':7200},'paths':{},
         'arms':{'RN_clip':{'tag':g.TAGS['RN_clip'],'time_limit_seconds':1800,'expected_args':{}}}}
    (tmp_path/'artifacts/logs').mkdir(parents=True)
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(p))
    monkeypatch.setattr(g,'validate_bindings',lambda *args:None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    return path


def test_supervisor_never_retries_existing_output(tmp_path, monkeypatch):
    path = supervisor_fixture(tmp_path,monkeypatch)
    (tmp_path/'artifacts/experiments'/g.TAGS['RN_clip']).mkdir(parents=True)
    monkeypatch.setattr(g.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='No automatic retry'):g.run_arm(path,'RN_clip',tmp_path)


@pytest.mark.skipif(os.name != 'nt',reason='Windows owned-process-tree timeout contract')
def test_supervisor_timeout_stops_only_owned_child_tree(tmp_path, monkeypatch):
    path = supervisor_fixture(tmp_path,monkeypatch)
    calls=[]
    class FakeProcess:
        pid=123456789
        waits=0
        def wait(self,timeout):
            self.waits+=1
            if self.waits==1:
                assert 0 < timeout <= 1800
                raise subprocess.TimeoutExpired('synthetic',timeout)
            return 0
    monkeypatch.setattr(g.subprocess,'Popen',lambda *args,**kwargs:FakeProcess())
    monkeypatch.setattr(g.subprocess,'run',lambda command,**kwargs:calls.append(command))
    with pytest.raises(RuntimeError,match='wall-time limit'):g.run_arm(path,'RN_clip',tmp_path)
    assert calls==[['taskkill','/PID','123456789','/T','/F']]
