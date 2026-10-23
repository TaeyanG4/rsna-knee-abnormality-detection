"""CPU contracts for registered K1 N-model stages; no labels, network or GPU."""
import ast
import base64
import contextlib
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pytest
from scipy.stats import rankdata
import torch

from tools import k1n_config_v1 as schema
from tools import k1n_stage_v1 as b
from tools import k1n_cpu_replay_v1 as replay
from tools import k1sr_runtime_v6 as v6
from tools import k1sr_runtime_v7 as v7


def configuration(n=3, drop=False):
    c = schema.current()
    c.update(drop_o=drop, alpha=.25 if drop else .2)
    while len(c['models']) < n:
        i = len(c['models'])
        m = dict(c['models'][1], id='test_fold_'+str(i),
            dataset_slug='test-owner/knee-test-fold-'+str(i),
            checkpoint_sha256=hashlib.sha256(('checkpoint'+str(i)).encode()).hexdigest(),
            manifest_sha256=hashlib.sha256(('manifest'+str(i)).encode()).hexdigest())
        c['models'].append(m)
    c['models'] = c['models'][:n]
    return schema.validate(c)


def runtime(c):
    module = ModuleType('k1n_test_runtime')
    source = Path(v7.__file__).read_text(encoding='utf-8').replace(
        'REGISTERED = config_schema.current()', 'REGISTERED = '+repr(c))
    exec(compile(source, '<baked v7>', 'exec'), module.__dict__)
    return module


@lru_cache(None)
def package(drop=False, diagnostic=False):
    return b.package(configuration(drop=drop), diagnostic)


def records(r, arrays):
    return [dict(uid=str(i), ok=True, config_sha256=r.CONFIG_SHA,
        k1_models=list(r.MODEL_IDS), finished_elapsed_seconds=10.,
        predictions={m:values[i].tolist() for m,values in zip(r.MODEL_IDS, arrays)},
        per_model_seconds={m:dict(load=1.,classify=2.) for m in r.MODEL_IDS})
        for i in range(len(arrays[0]))]


def rank(x):
    return rankdata(x, method='average', axis=0)/len(x)


@pytest.mark.parametrize('n', range(1,7))
@pytest.mark.parametrize('drop', [False,True])
def test_n_model_math_ties_roster_and_x3_legacy(n, drop):
    c = configuration(n,drop)
    r = runtime(c)
    rng = np.random.default_rng(517)
    parent, *arrays = [rng.integers(0,10,(32,12)).astype(np.float32)/10 for _ in range(n+1)]
    payload = replay.csv_bytes(list(map(str,range(32))), parent)
    rows = list(reversed(records(r, arrays)))
    result = r.blend(payload, rows)
    actual = r.util.table(result)[1]
    expected = rank((1-c['alpha'])*rank(parent)+c['alpha']*rank(
        np.mean([rank(x) for x in arrays], axis=0))).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
    if n == 3 and not drop:
        assert result == v6.blend(payload, rows)


@pytest.mark.parametrize('fault', ['ok','missing','failed','late','partial','nonfinite','corrupt','disabled','config','finish_config'])
@pytest.mark.parametrize('diagnostic', [False,True])
def test_six_model_finish_fails_closed(tmp_path, fault, diagnostic):
    c = configuration(6, True)
    r = runtime(c)
    parent = replay.csv_bytes(['0','1','2'], np.full((3,12),.23456789))
    filename = 'diagnostic-c85r7.csv' if diagnostic else 'submission.csv'
    (tmp_path/filename).write_bytes(parent)
    worker = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    rows = records(r, [np.full((3,12),.3) for _ in r.MODEL_IDS])
    if fault == 'missing': rows.pop()
    if fault == 'failed': rows[0]['ok'] = False
    if fault == 'late': rows[0]['finished_elapsed_seconds'] = r.GUARD_SECONDS+1
    if fault == 'partial': rows[0]['predictions'].pop(r.MODEL_IDS[-1])
    if fault == 'nonfinite': rows[0]['predictions'][r.MODEL_IDS[-1]][0] = float('nan')
    if fault == 'config': rows[0]['config_sha256'] = 'wrong'
    if fault == 'disabled': (worker.parent/'k1-disabled.json').write_text('{}')
    (worker/'k1-studies.jsonl').write_text('{' if fault == 'corrupt' else '\n'.join(map(json.dumps,rows)))
    cfg = b.config(c,diagnostic)
    if fault == 'finish_config': cfg['config_sha256'] = 'wrong'
    receipt = r.finish(tmp_path,tmp_path/'run',cfg,12.)
    assert receipt['k1_executed'] == (fault == 'ok')
    assert receipt['status'] == ('COMPLETE_SHARED_K1N' if fault == 'ok' else 'DEGRADED_PARENT_ONLY')
    assert receipt['parent_mode'] == 'INF3_P_R224'
    assert receipt['model_identities'] == c['models']
    assert set(receipt['per_model_seconds']) == set(r.MODEL_IDS)
    assert (tmp_path/'k1sr-parent.csv').read_bytes() == parent
    result = tmp_path/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')
    if fault != 'ok': assert result.read_bytes() == parent
    if diagnostic:
        assert receipt['never_submit'] and receipt['submission_sha256'] is None
        assert not (tmp_path/'submission.csv').exists()


@pytest.mark.parametrize('fault', ['ok','load0','load1','load2','load3','load4','load5',
    'predict0','predict1','predict2','predict3','predict4','predict5','transfer','load_timeout','group_timeout','guard'])
@pytest.mark.parametrize('stride', [2,3])
def test_six_model_owner_single_transfer_budgets_and_any_failure(tmp_path, monkeypatch, fault, stride):
    c = configuration(6)
    c['k1_eval_stride'] = stride
    r = runtime(c)
    clock = [1000.]
    monkeypatch.setenv('K1SR_STARTED','1000')
    monkeypatch.setattr(r,'time',SimpleNamespace(time=lambda:clock[0],perf_counter=lambda:clock[0]))
    monkeypatch.setattr(r,'deadline',lambda seconds: contextlib.nullcontext() if seconds>0 else (_ for _ in ()).throw(TimeoutError('budget')))
    monkeypatch.setattr(r,'precision',contextlib.nullcontext)
    monkeypatch.setattr(r.w,'validate_item',lambda *a:(np.zeros(1),np.zeros(1)))
    calls, prepared = [], object()
    def load(device, mid):
        name = 'load'+str(r.MODEL_IDS.index(mid))
        calls.append(name)
        if name == fault: raise RuntimeError(name)
        clock[0] += 1501 if fault == 'load_timeout' else 200
        return mid
    def transfer(*args):
        calls.append('transfer')
        if fault == 'transfer': raise RuntimeError(fault)
        clock[0] += 2
        return prepared
    def predict(mid, values, *, chunk, fp16):
        assert values is prepared and chunk == 32 and fp16
        name = 'predict'+str(r.MODEL_IDS.index(mid))
        calls.append(name)
        if name == fault: raise RuntimeError(name)
        clock[0] += 25 if fault == 'group_timeout' else 1
        return np.full(12,.4)
    monkeypatch.setattr(r,'load_classifier',load)
    monkeypatch.setattr(r,'transfer_inputs',transfer)
    monkeypatch.setattr(r,'predict_prepared',predict)
    if fault == 'guard': clock[0] += r.GUARD_SECONDS+1
    worker = tmp_path/'worker0'; worker.mkdir()
    owner = r.Owner(worker,'cuda')
    owner.handle(dict(uid='0',_k1sr=dict(item={},detail={})),lambda item:calls.append('parent'))
    row = json.loads((worker/'k1-studies.jsonl').read_bytes())
    assert row['k1_eval_stride'] == stride
    assert calls[0] == 'parent' and calls.count('transfer') <= 1
    assert row['ok'] == (fault == 'ok')
    assert row['load_timeout_seconds'] == 1500 and row['call_timeout_seconds'] == 120
    if fault == 'ok':
        assert row['load_seconds'] == 1200 and row['classifier_seconds'] == 8
        assert calls.count('transfer') == 1 and len(row['predictions']) == 6
        assert all(row['per_model_seconds'][m] == dict(load=200,classify=1) for m in r.MODEL_IDS)
    else:
        assert owner.failed and not owner.models and 'predictions' not in row
        assert (tmp_path/'k1-disabled.json').exists()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('diagnostic', [False,True])
@pytest.mark.parametrize('failure', ['healthy','survivors','r224_abort'])
def test_actual_drop_parent_orchestration_and_original_p_math(tmp_path, diagnostic, failure):
    files = package(True,diagnostic)[3]
    ids = list(map(str,range(40)))
    rng = np.random.default_rng(45)
    arrays = [rng.integers(0,10,(40,12)).astype(np.float32)/10 for _ in range(3)]
    failures = {'R':{},'R224':{},'S_R7':{}}
    if failure == 'survivors':
        failures = {'R':{'0':'injected','1':'injected'},'R224':{'0':'injected'},'S_R7':{'0':'injected','2':'injected'}}
        arrays[0][3] = .5  # R sentinel also activates the unchanged survivor policy.
    if failure == 'r224_abort':
        failures['R224'] = {'0':'injected','1':'injected'}
        with pytest.raises(Exception, match='more than 3%'):
            replay.replay(files,tmp_path,ids,arrays,failures=failures,diagnostic=diagnostic)
        assert not (tmp_path/'submission.csv').exists()
        return
    result,payload = replay.replay(files,tmp_path,ids,arrays,failures=failures,diagnostic=diagnostic)
    assert result['p_branch_exact'] and result['o_executed'] is False
    if failure == 'healthy':
        weights = np.asarray((.5,.5,.5,.5,.5,.2,.6,.5,.8,.35,.5,.5))
        expected = weights*(.65*rank(arrays[0])+.35*rank(arrays[1]))+(1-weights)*rank(arrays[2])
        np.testing.assert_array_equal(replay.util.table(payload)[1], expected.astype(np.float32))
    else:
        assert np.all(replay.util.table(payload)[1][0] == .5)
        assert result['receipt']['fallbacks']['r224_prediction'] == 1


@pytest.mark.parametrize('drop', [False,True])
@pytest.mark.parametrize('diagnostic', [False,True])
def test_payload_scope(drop, diagnostic):
    nb, meta, old, new, old_nb, old_meta = package(drop,diagnostic)
    assert {n for n in new.keys() & old.keys() if new[n] != old[n]} == {
        'exp169_r7_port.py','exp169_release_c85r7.py','exp169_release_c85r7codec.py'}
    assert new.keys()-old.keys() == b.ADDED
    assert old.keys()-new.keys() == {'tools/k1sr_runtime_v6.py'}
    for member in ('exp169_r224.py','raptor224_run.py','exp169_r7_src/compact_infer.py',
                   'exp169_r7_src/compact_train.py','tools/k1_shared_render_v1.py','a172-cell.py.txt'):
        assert new[member] == old[member]
    codec = 'exp169_release_c85r7codec.py'
    assert b.parent_edit.function(new[codec].decode(),'run_public') == b.parent_edit.function(old[codec].decode(),'run_public')
    base = 'exp169_release_c85r7.py'
    assert b.parent_edit.function(new[base].decode(),'authenticated_pr7') == b.parent_edit.function(old[base].decode(),'authenticated_pr7')
    assert nb.keys() == old_nb.keys() and nb['cells'][0] == old_nb['cells'][0]
    assert nb['cells'][3:] == old_nb['cells'][3:]
    normalized = copy.deepcopy(nb)
    code = ''.join(nb['cells'][2]['source'])
    old_blob, old_pins, _ = b.before.parent.archive(''.join(old_nb['cells'][2]['source']))
    blob, pins, _ = b.before.parent.archive(code)
    code = b.exact(code,repr(base64.b64encode(blob).decode()),repr(base64.b64encode(old_blob).decode()))
    code = b.exact(code,'PINS = '+repr(pins),'PINS = '+repr(old_pins))
    code = b.exact(code,repr(b.digest(blob)),repr(b.digest(old_blob)))
    code = b.exact(code,'from tools import k1sr_runtime_v7 as _k1sr','from tools import k1sr_runtime_v6 as _k1sr')
    code = b.exact(code,repr(b.config(configuration(drop=drop),diagnostic)),repr(b.before.config('k1x3sr20',diagnostic)))
    normalized['cells'][2]['source'] = code.splitlines(True)
    normalized['cells'][1] = old_nb['cells'][1]
    assert normalized == old_nb
    ignored = {'id','title','dataset_sources'}
    assert {k:v for k,v in meta.items() if k not in ignored} == {k:v for k,v in old_meta.items() if k not in ignored}
    removed = set(old_meta['dataset_sources'])-set(meta['dataset_sources'])
    assert removed == ({'taeyangg4/knee-exp169-a172-v2-assets','taeyangg4/knee-exp155-own0949-oai-v1-2-assets'} if drop else set())
    if not drop:
        for func in ('run_release','collect','preflight'):
            member = 'exp169_release_c85r7.py' if func == 'preflight' else codec
            assert b.parent_edit.function(new[member].decode(),func) == b.parent_edit.function(old[member].decode(),func)


@pytest.mark.parametrize('fault', ['empty','seven','placeholder_slug','placeholder_hash','bad_path','duplicate_id',
    'duplicate_hash','duplicate_mount','unknown','bool_alpha','nan_alpha','chunk64','bool_chunk','non_drop_alpha','legacy_alias'])
def test_config_rejects_invalid_registrations(fault):
    c = configuration()
    if fault == 'empty': c['models'] = []
    if fault == 'seven': c['models'] *= 3
    if fault == 'placeholder_slug': c['models'][1]['dataset_slug'] = 'user/placeholder-fold'
    if fault == 'placeholder_hash': c['models'][1]['checkpoint_sha256'] = '0'*64
    if fault == 'bad_path': c['models'][1]['checkpoint'] = 'weights/../source/a.pt'
    if fault == 'duplicate_id': c['models'][1]['id'] = c['models'][0]['id']
    if fault == 'duplicate_hash': c['models'][1]['checkpoint_sha256'] = c['models'][0]['checkpoint_sha256']
    if fault == 'duplicate_mount': c['models'][1]['dataset_slug'] = 'different/'+c['models'][0]['dataset_slug'].split('/')[1]
    if fault == 'unknown': c['allow_unpinned'] = True
    if fault == 'bool_alpha': c['alpha'] = True
    if fault == 'nan_alpha': c['alpha'] = float('nan')
    if fault == 'chunk64': c['encoder_chunk'] = 64
    if fault == 'bool_chunk': c['encoder_chunk'] = True
    if fault == 'non_drop_alpha': c['alpha'] = .25
    if fault == 'legacy_alias': c['models'][0]['dataset_slug'] = 'test/legacy-alias'
    with pytest.raises(Exception): schema.validate(c)


def test_defaults_examples_and_placeholder_prepare_refusal(monkeypatch,tmp_path):
    for drop in (False,True):
        c = configuration(drop=drop); del c['alpha']; del c['encoder_chunk']
        valid = schema.validate(c)
        assert valid['alpha'] == (.25 if drop else .2) and valid['encoder_chunk'] == 32
    assert json.loads((b.ROOT/'configs/k1n/current-x3.json').read_bytes()) == schema.current()
    c = json.loads((b.ROOT/'configs/k1n/six-model-drop-o.placeholder.json').read_bytes())
    monkeypatch.setattr(b,'stage',lambda c,diagnostic=False:tmp_path/str(diagnostic))
    with pytest.raises(Exception): b.run('prepare',c)
    assert not list(tmp_path.iterdir())


def test_six_model_baked_payload_and_custom_registered_alpha():
    c = configuration(6,True)
    c['alpha'] = .37
    _,meta,_,files,_,_ = b.package(c)
    module = ModuleType('six_baked')
    exec(compile(files['tools/k1sr_runtime_v7.py'],'<six baked>','exec'),module.__dict__)
    assert module.REGISTERED == c and module.K1_WEIGHT == .37
    assert meta['dataset_sources'][-6:] == [m['dataset_slug'] for m in c['models']]
    rng = np.random.default_rng(32)
    values = [rng.random((32,12)).astype(np.float32) for _ in range(7)]
    payload = replay.csv_bytes(list(map(str,range(32))),values[0])
    output = module.blend(payload,records(module,values[1:]))
    expected = rank(.63*rank(values[0])+.37*rank(np.mean([rank(v) for v in values[1:]],axis=0)))
    np.testing.assert_array_equal(replay.util.table(output)[1],expected.astype(np.float32))


@pytest.mark.parametrize('drop', [False,True])
def test_paired_prepare_validate_immutable(monkeypatch,tmp_path,drop):
    # Scratch remains within the project's fixed pytest lane directory.
    monkeypatch.setattr(b,'stage',lambda c,diagnostic=False:tmp_path/('n32' if diagnostic else 'release'))
    c = configuration(drop=drop)
    result = b.run('prepare',c)
    assert result['status'] == 'PREPARED_LOCAL_ONLY'
    assert [v['never_submit'] for v in result['stages']] == [False,True]
    assert b.run('validate',c)['status'] == 'PASS_CPU_VALIDATE'
    with pytest.raises(Exception,match='immutable stage exists'): b.run('prepare',c)
    (tmp_path/'release/config.json').write_text('{}')
    with pytest.raises(Exception,match='artifact drift'): b.run('validate',c)


def test_loader_rejects_unpinned_assets_before_source_or_torch_load(monkeypatch,tmp_path):
    (tmp_path/'asset-manifest.json').write_text('{}')
    monkeypatch.setattr(v7.util,'mount',lambda *a:tmp_path)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('deserialization before pin'))
    monkeypatch.setattr(v7.importlib.util,'spec_from_file_location',lambda *a,**k:pytest.fail('source import before pin'))
    with pytest.raises(Exception,match='manifest pin'): v7.load_classifier('cpu',v7.MODEL_IDS[1])
    with pytest.raises(Exception,match='unknown K1'): v7.load_classifier('cpu','unknown')


def test_v6_guard_render_precision_and_transfer_functions_unchanged():
    def nodes(module):
        return {n.name:ast.dump(n) for n in ast.parse(Path(module.__file__).read_text(encoding='utf-8')).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,z = nodes(v6),nodes(v7)
    for name in ('remaining','deadline','phase_deadline','disabled','disable',
                 'precision','transfer_inputs','predict_prepared','patch_main','worker_main'):
        assert a[name] == z[name], name
    source = Path(v7.__file__).read_text(encoding='utf-8').replace(
        'shared.SharedR6(ds, ci, eval_stride=K1_EVAL_STRIDE)', 'shared.SharedR6(ds, ci)')
    loader = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LoaderDataset')
    assert a['LoaderDataset'] == ast.dump(loader)


def test_original_runtime_and_stages_unchanged():
    pins = json.loads((b.ROOT/'.local/scratch/astra-b1/k1n-v7/original-hashes.json').read_bytes())
    for name,pin in pins.items(): assert v7.w.sha(Path(name)) == pin
