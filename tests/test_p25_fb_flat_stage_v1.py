"""F4 v1.0.0: dual-parent scope, real baked arithmetic and fail-loud regression."""
from functools import lru_cache
import json
from types import ModuleType
import numpy as np
import pytest
from tools import p25_fb_flat_stage_v1 as b
from tools import k1n_cpu_replay_v1 as cpu
from tools import k1n_flat_p_check_v1 as mathcheck


@lru_cache(None)
def package(diagnostic=False): return b.package(diagnostic)


def runtime(files):
    r = ModuleType('f4_baked_runtime')
    exec(compile(files[b.RUNTIME], '<F4 embedded runtime>', 'exec'), r.__dict__)
    return r


@pytest.mark.parametrize('diagnostic',[False,True])
def test_dual_parent_scope(diagnostic):
    po,pr,pc,c,cfg,onb,nb,om,meta,old,new,ff,fnb,fp = package(diagnostic)
    outputs,reg = b.artifacts(diagnostic)
    scope = json.loads(outputs['payload-scope.json'])
    assert {f['field'] for f in json.loads(outputs['field-level-diff.json'])} == {
        'release_version','kernel_id','p_weight_mode'}
    assert set(new)==set(old)==set(ff)
    assert {n for n in old if old[n]!=new[n]} == {b.flat.HELPER,b.flat.SCHEMA,b.CODEC,b.RUNTIME}
    assert {n for n in ff if ff[n]!=new[n]} == {b.CODEC,b.RUNTIME}
    assert scope['k1_functions_identical'] and scope['parent_launch_pool_threads_timing_identical']
    assert scope['flat_schema_byte_identical'] and scope['flat_runtime_plus_exact_fb_patch']
    assert len(scope['untouched']) == len(old)-4
    assert c['models']==pc['models'] and len(c['models'])==4
    assert c['models'][-1]['checkpoint_sha256']==b.FB_SHA
    assert c['alpha']==.25 and c['drop_o'] and c['k1_eval_stride']==2
    assert c['k1_guard_hours']==8.8 and c['k1_max_failed_fraction']==.05
    assert cfg['p_weight_mode']=='flat' and cfg['p_raptor_weights']==[.5]*12
    assert cfg['k1_runtime_version']==10 and 'a5_contract' not in cfg
    assert reg['never_submit'] is diagnostic and not reg['qualified_for_submission']
    assert reg['capture_timeout_seconds']==pr['capture_timeout_seconds']==900
    assert reg['call_timeout_seconds']==pr['call_timeout_seconds']==300
    assert meta == dict(om,id=cfg['kernel_id'],title=cfg['kernel_id'].split('/')[1])
    assert b.DATASET_ID in meta['dataset_sources']
    assert runtime(new).CONFIG_SHA==cfg['config_sha256']


@pytest.mark.parametrize('diagnostic',[False,True])
def test_flat_four_member_full_math_and_finish(tmp_path,diagnostic):
    parts=package(diagnostic); cfg,old,new=parts[4],parts[9],parts[10]
    rng=np.random.default_rng(187); n=32; ids=list(map(str,range(n)))
    arrays=[rng.integers(1,31,(n,12))/32 for _ in range(3)]
    masks=[np.zeros(n,bool) for _ in range(3)]
    before,after,comparison=mathcheck.compare(arrays,masks,old,new)
    assert comparison['changed_columns']==[5,6,8,9]
    result,payload=cpu.replay(new,tmp_path/'parent',ids,arrays,diagnostic=diagnostic)
    np.testing.assert_array_equal(b.util.table(payload)[1],after.astype(np.float32))
    assert result['p_branch_exact'] and not result['o_executed']
    r=runtime(new); ka=[rng.integers(0,8,(n,12))/7 for _ in r.MODEL_IDS]
    rows=[dict(uid=uid,ok=True,config_sha256=r.CONFIG_SHA,k1_models=list(r.MODEL_IDS),
        finished_elapsed_seconds=1.,predictions={mid:ka[j][i].tolist() for j,mid in enumerate(r.MODEL_IDS)})
        for i,uid in enumerate(ids)]
    dest=tmp_path/'finish';dest.mkdir()
    (dest/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(payload)
    worker=dest/'run/pr7/pr7-run/s_r7/controller/worker0';worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    rec=r.finish(dest,dest/'run',cfg,12.)
    actual=b.util.table((dest/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')).read_bytes())[1]
    rank=b.util.ranks
    expected=rank(.75*rank(after.astype(np.float32))+.25*rank(np.mean([
        rank(a.astype(np.float32)) for a in ka],axis=0))).astype(np.float32)
    np.testing.assert_array_equal(actual,expected)
    assert rec['p_raptor_weights']==[.5]*12 and rec['parent_mode']=='INF3_P_R224'
    assert rec['k1_studies']==32 and rec['failed_study_count']==0
    if diagnostic: assert rec['never_submit'] and not (dest/'submission.csv').exists()


@pytest.mark.parametrize('diagnostic',[False,True])
def test_missing_fb_fails_loud(tmp_path,diagnostic):
    parts=package(diagnostic);r=runtime(parts[10]);cfg=parts[4]
    rows=[dict(uid=str(i),ok=True,config_sha256=r.CONFIG_SHA,k1_models=list(r.MODEL_IDS),
        finished_elapsed_seconds=1.,predictions={mid:[.2]*12 for mid in r.MODEL_IDS[:-1]}) for i in range(3)]
    worker=tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0';worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(
        cpu.csv_bytes(['0','1','2'],np.full((3,12),.5)))
    with pytest.raises(RuntimeError,match='fail-loud'):r.finish(tmp_path,tmp_path/'run',cfg,1.)
    assert json.loads((tmp_path/'k1sr-receipt.json').read_bytes())['fail_loud_triggered']


def test_isolated_check_precedes_any_stage_write(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'stage',lambda diagnostic=False:tmp_path/('n32' if diagnostic else 'base'))
    def refuse(*args,**kwargs): raise ValueError('sentinel embedded failure')
    monkeypatch.setattr(b.selfcheck,'check',refuse)
    with pytest.raises(ValueError,match='sentinel embedded failure'):b.run('prepare')
    assert not (tmp_path/'base').exists() and not (tmp_path/'n32').exists()


def test_unqualified_admit_is_hold():
    assert b.run('admit',True)['status']=='HOLD_NOT_ADMITTED'
    with pytest.raises(ValueError):b.run('admit')
