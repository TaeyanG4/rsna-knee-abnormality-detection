"""CPU regression coverage for the in-pass K1 release integration."""
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pytest
import torch
from scipy.stats import rankdata
from tools import c85r7_r6w958k1sr_v1156 as b
from tools import k1sr_runtime_v1 as r
from tools import k1_shared_render_harness_v1 as h


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setenv('K1SR_STARTED', str(time.time()))


def csv_payload(values):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\r\n')
    writer.writerow([r.w.UID]+r.w.LABELS)
    writer.writerows([str(i), *row] for i,row in enumerate(values))
    return stream.getvalue().encode()


def records(values):
    return [dict(uid=str(i), ok=True, prediction=row.tolist(), finished_elapsed_seconds=10.)
            for i,row in enumerate(values)]


def test_blend_independent_reference_ties_and_order():
    rng = np.random.default_rng(98)
    a,bv = [rng.integers(0, 10, size=(32,12))/10 for _ in range(2)]
    out = r.blend(csv_payload(a), list(reversed(records(bv))))
    ids, actual = r.util.table(out)
    rank = lambda x: rankdata(x, method='average', axis=0)/len(x)
    expected = rank(.8*rank(a.astype(np.float32))+.2*rank(bv.astype(np.float32))).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert ids == [str(i) for i in range(32)]


@pytest.mark.parametrize('fault', ['missing','error','late','disabled','corrupt','ok'])
def test_finish_parent_bytes_and_status(tmp_path, fault):
    parent = csv_payload(np.full((3,12), .23456789))
    (tmp_path/'submission.csv').write_bytes(parent)
    worker = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    rows = records(np.full((3,12), .3))
    if fault == 'missing': rows.pop()
    if fault == 'error': rows[0]['ok'] = False
    if fault == 'late': rows[0]['finished_elapsed_seconds'] = r.GUARD_SECONDS+1
    if fault == 'disabled': (worker.parent/'k1-disabled.json').write_text('{}')
    (worker/'k1-studies.jsonl').write_text('\n'.join(json.dumps(x) for x in rows) if fault != 'corrupt' else '{')
    result = r.finish(tmp_path, tmp_path/'run', b.config('k1sr20'), 100.)
    assert (tmp_path/'k1sr-parent.csv').read_bytes() == parent
    assert result['k1_executed'] == (fault == 'ok')
    assert result['status'] == ('COMPLETE_SHARED_K1' if fault == 'ok' else 'DEGRADED_PARENT_ONLY')
    if fault != 'ok': assert (tmp_path/'submission.csv').read_bytes() == parent


def test_old_success_invalidated_before_read(tmp_path):
    receipt = tmp_path/'k1sr-receipt.json'
    receipt.write_text('{"status":"COMPLETE_SHARED_K1","k1_executed":true}')
    with pytest.raises(FileNotFoundError):
        r.finish(tmp_path, tmp_path/'run', {}, 10.)
    assert json.loads(receipt.read_bytes()) == dict(status='COMPARING',k1_executed=False)


class FakeDataset:
    def __len__(self): return 1
    def __getitem__(self,index): return dict(uid='0', error=None, seconds=[1,2,3])


@pytest.mark.parametrize('fault', ['capture','guard'])
def test_loader_failure_and_guard_keep_parent(monkeypatch, tmp_path, fault):
    if fault == 'guard': monkeypatch.setenv('K1SR_STARTED', str(time.time()-r.GUARD_SECONDS-1))
    monkeypatch.setitem(sys.modules, 'compact_infer', SimpleNamespace())
    def fail(*args): raise RuntimeError('capture fault')
    monkeypatch.setattr(r.shared, 'SharedR6', fail)
    data = r.LoaderDataset(FakeDataset(), tmp_path/'worker0')
    item = data[0]
    assert {k:v for k,v in item.items() if k != '_k1sr'} == FakeDataset()[0]
    assert (data[0] == FakeDataset()[0])  # disabled until owner exits


@pytest.mark.parametrize('fault', ['load','predict','guard','journal'])
def test_owner_fault_is_optional_and_parent_runs_first(monkeypatch, tmp_path, fault):
    owner = r.Owner(tmp_path, 'cuda')
    calls = []
    monkeypatch.setattr(r.w, 'validate_item', lambda *args: (torch.zeros(1),torch.zeros(1)))
    monkeypatch.setattr(r, 'load_classifier', lambda device: calls.append('load') or object())
    monkeypatch.setattr(r.shared, 'predict', lambda *args, **kwargs: calls.append('predict') or np.full(12,.2))
    def fail(*args, **kwargs): raise RuntimeError('injected')
    if fault == 'load': monkeypatch.setattr(r, 'load_classifier', fail)
    if fault == 'predict': monkeypatch.setattr(r.shared, 'predict', fail)
    if fault == 'guard': monkeypatch.setenv('K1SR_STARTED', str(time.time()-r.GUARD_SECONDS-1))
    if fault == 'journal': (tmp_path/'k1-studies.jsonl').mkdir()
    item = dict(uid='x', _k1sr=dict(item={},detail={}))
    owner.handle(item, lambda item: calls.append('parent'))
    assert calls[0] == 'parent' and owner.failed
    assert '_k1sr' not in item
    if fault != 'journal':
        assert json.loads((tmp_path/'k1-studies.jsonl').read_text())['ok'] is False
    if fault == 'guard': assert calls == ['parent']


def test_telemetry_failure_does_not_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(r.w, 'write', lambda *a: (_ for _ in ()).throw(OSError('disk')))
    r.disable(tmp_path, 'failure')


@pytest.mark.parametrize('variant', list(b.VARIANTS))
@pytest.mark.parametrize('diagnostic', [False,True])
def test_package_scope_registration_and_compilation(variant,diagnostic):
    nb,meta,old,new = b.package(variant,diagnostic)
    artifacts,reg = b.artifacts(variant,diagnostic)
    cfg = b.config(variant,diagnostic)
    assert cfg['parent_weight'] == .8 and cfg['k1_weight'] == .2
    assert cfg['r224_weight'] == (0. if variant.startswith('n224') else .35)
    assert reg['guard_seconds'] == 28800 and not reg['execution_verified']
    assert meta['id'] == cfg['kernel_id'] and meta['title'] == meta['id'].split('/')[1]
    assert meta['enable_internet'] is False and meta['is_private'] is True
    assert reg['never_submit'] is diagnostic
    scope = json.loads(artifacts['payload-scope.json'])
    assert all(new[n] == old[n] for n in scope['untouched'])
    for name in ('exp169_r7_src/compact_infer.py','exp169_r7_src/compact_train.py',
                 'exp189_s6_src/compact_infer.py', 'a172-cell.py.txt'):
        assert old[name] == new[name]
    # Normalize the exact three port hook lines; all other port bytes stay equal.
    port = new['exp169_r7_port.py'].decode().replace(
        '        from tools.k1sr_runtime_v1 import install\n        _k1_worker = install(runtime_source)\n','')
    assert port.replace('"--worker", _k1_worker','"--worker", "compact_infer.py"').encode() == old['exp169_r7_port.py']
    if variant.startswith('n224'):
        assert new['exp169_r224.py'].decode().replace(b.n224.R224_NEW,b.n224.R224_OLD).encode() == old['exp169_r224.py']
    else: assert new['exp169_r224.py'] == old['exp169_r224.py']
    if not diagnostic: assert new['raptor224_run.py'] == old['raptor224_run.py']
    assert b.parent.archive(''.join(nb['cells'][2]['source']))[2] == new
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']),'<notebook>','exec')


def test_exact_main_hooks_and_spawn_pickle(tmp_path):
    import pickle
    source = (h.runtime_source()/'compact_infer.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    main = ast.get_source_segment(source, next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))+'\n'
    patched = r.patch_main(main)
    compile(patched,'<hooked main>','exec')
    normalized = patched.replace('    _k1_owner = _k1sr.Owner(out, str(dev))\n','').replace(
        '        ds = _k1sr.LoaderDataset(ds, out)\n','').replace('_k1_owner.handle(item, handle)','handle(item)')
    assert normalized == main
    dataset = pickle.loads(pickle.dumps(r.LoaderDataset(FakeDataset(),tmp_path)))
    assert dataset.adapter is None and len(dataset)==1


def test_worker_hook_setup_failure_runs_original(monkeypatch, tmp_path):
    calls=[]
    ci=SimpleNamespace(__file__=__file__,main=lambda:calls.append('original'))
    monkeypatch.setitem(sys.modules,'compact_infer',ci)
    monkeypatch.setattr(sys,'argv',['worker','--output',str(tmp_path)])
    r.worker_main()
    assert calls==['original']


def test_deadline_expired():
    with pytest.raises(TimeoutError):
        with r.deadline(0): pass


def test_install_failure_keeps_original_worker(monkeypatch, tmp_path):
    source = tmp_path/'r7_runtime'
    source.mkdir()
    assert r.install(source) == 'k1sr_worker.py'
    monkeypatch.setattr(Path, 'write_text', lambda *a, **k: (_ for _ in ()).throw(OSError('injected')))
    assert r.install(source) == 'compact_infer.py'
    assert (tmp_path/'controller/k1-disabled.json').is_file()


def test_n32_preserves_all_series_metadata_and_largest8(tmp_path):
    from tools import k1sr_n32_v1 as n32
    source, dest = tmp_path/'raw', tmp_path/'n32'
    source.mkdir()
    with (source/'train_series.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(n32.COLS)
        for i in range(35):
            uid=str(i)
            writer.writerow([uid, 'series', 'Coronal', '1', '0'])
            folder=source/'train_series'/uid/'series'
            folder.mkdir(parents=True)
            for j in range(i+1): (folder/(str(j)+'.dcm')).touch()
    result=n32.prepare(source,dest)
    assert result['studies']==32 and n32.prepare(source,dest)==result
    selection=json.loads((dest/'selection.json').read_bytes())
    assert set(selection['largest8']) == {str(i) for i in range(27,35)}
    with (dest/'test_series.csv').open() as f:
        assert all(row['Fat_Suppression']=='1' for row in csv.DictReader(f))
    assert not (dest/'train.csv').exists()


def test_precision_restores_parent_flags():
    before = (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
              torch.backends.cudnn.benchmark,torch.get_float32_matmul_precision())
    with pytest.raises(RuntimeError):
        with r.precision():
            assert not torch.backends.cuda.matmul.allow_tf32
            raise RuntimeError('injected')
    assert before == (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                      torch.backends.cudnn.benchmark,torch.get_float32_matmul_precision())


def test_n32_all_parent_branches_bind_competition_override(monkeypatch, tmp_path):
    _, _, _, files = b.package('k1sr20', True)
    # Execute only the routing/source-generation functions, never public models.
    def function(member, name, scope):
        tree = ast.parse(files[member].decode())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), member, 'exec'), scope)
        return scope[name]
    monkeypatch.setenv('C85_COMPETITION', str(tmp_path))
    tree = ast.parse(files['exp169_release_v5q.py'].decode())
    pin = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'A172_CELL_SHA')
    v5_scope = dict(hashlib=hashlib, require=r.w.need, A172_CELL_SHA=pin)
    function('exp169_release_v5q.py', '_replace_once', v5_scope)
    transform = function('exp169_release_v5q.py', 'a172_runtime_source', v5_scope)
    scope = dict(os=os, Path=Path, need=r.w.need,
                 v5q=SimpleNamespace(a172_runtime_source=transform))
    root = function('exp169_release_c85.py', 'competition_root', scope)
    assert root() == tmp_path
    generated = function('exp169_release_c85.py', 'a172_source', scope)(
        files['a172-cell.py.txt'].decode(), tmp_path)
    assert '_comp = _c85.competition_root()' in generated
    assert 'rsna-knee-abnormality-detection/sample_submission.csv' not in generated
    assert function('raptor224_run.py', 'find_test_root', dict(os=os))() == str(tmp_path)
    assert b'C85_COMPETITION' in files['exp169_release_c85.py']
