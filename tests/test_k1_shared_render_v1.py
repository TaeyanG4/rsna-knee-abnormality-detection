"""CPU regressions for row identity, side isolation and fixed parity gates."""
import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pytest
import torch

from tools import k1_shared_render_v1 as s
from tools import k1_shared_render_harness_v1 as h


def windows(path):
    nodes = ast.parse(path.read_text(encoding='utf-8')).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and
                n.name in ('covering_centers', 'phys_step', 'window_rows')]
    ns = dict(math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)
    return ns


K1 = h.ASSETS / 'source/compact_train.py'
R6 = h.OUT / 'parent/exp169_r7_src/compact_train.py'


def test_pinned_window_functions_identical():
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name in ('covering_centers', 'phys_step', 'window_rows')}
    assert functions(K1) == functions(R6)
    for name, pin in s.R6_SOURCES.items():
        assert s.w.sha(R6.parent / (name + '.py')) == pin


def test_parent_runtime_patch_and_rendering_methods():
    runtime = h.runtime_source()
    assert s.w.sha(runtime / 'compact_infer.py') == s.R6_RUNTIME_SHA
    def methods(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CStudies')
        return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
    raw, patched = methods(R6.parent/'compact_infer.py'), methods(runtime/'compact_infer.py')
    # Only constructor/device setup and exception diagnostics change in CStudies.
    assert {k for k in raw if raw[k] != patched[k]} == {'__init__', '_setup', '__getitem__'}
    for name in ('_native_and_geometry', '_prepare', 'rows'):
        assert raw[name] == patched[name]


@pytest.mark.parametrize('n,spacing', [(1, 3.5), (2, 3.5), (4, 3.5), (10, 3.5), (13, .7)])
def test_rows_keep_block_phase_edges_and_all_metadata(n, spacing):
    # Each pixel equals its original row number, giving an independent oracle.
    px = torch.arange(2*n, dtype=torch.uint8)[:, None, None].expand(-1,384,384).clone()
    meta = torch.arange(2*n*12, dtype=torch.float32).reshape(2*n,12)
    meta[:,5] = math.log1p(spacing)
    rows = dict(px=px, meta=meta, blocks=[dict(n_slices=n, views=[(0,0),(1,n)])])
    fn = windows(K1)
    item = s.from_rows(rows, 'fixture', fn['window_rows'])
    ps = fn['phys_step'](math.expm1(float(meta[0,5])), 3.5)
    centers = fn['covering_centers'](n, 1, 3*ps)
    expected = [[offset+min(n-1,max(0,k+d*ps)) for d in (-1,0,1)] for offset in (0,n) for k in centers]
    assert item['images'][:,:,0,0].tolist() == expected
    indices = [offset+k for offset in (0,n) for k in centers]
    assert torch.equal(item['meta'], meta[indices][:,[1,3,4,5,6,7]])
    # Endpoint n-1 can be missing from the stride1 parent windows when ps>1.
    for view in (0, 1):
        assert len(item['images']) == 2*len(centers)


def test_stride3_not_universal_subset_of_stride1():
    fn = windows(K1)
    assert fn['covering_centers'](4,1,2) == [1,3]
    assert fn['covering_centers'](4,1,6) == [1,3]
    examples = [(n,ps) for n in range(1,60) for ps in range(2,12)
                if not set(fn['covering_centers'](n,1,3*ps)) <= set(fn['covering_centers'](n,1,ps))]
    assert examples  # Raw px row capture, not window-only selection, is necessary.


def test_empty_or_invalid_rows_fail():
    with pytest.raises(ValueError):
        s.from_rows(dict(px=torch.zeros(1,384,384),meta=torch.zeros(1,12),blocks=[]),'x',None)
    with pytest.raises(ValueError, match='no K1 windows'):
        s.from_rows(dict(px=torch.zeros(1,384,384,dtype=torch.uint8),meta=torch.zeros(1,12),blocks=[]),
                    'x',windows(K1)['window_rows'])


@pytest.mark.parametrize('fail',[False,True])
def test_original_side_clears_and_restores_forced_state(monkeypatch,fail):
    ci = SimpleNamespace(FORCED_SIDE={'uid':'R','other':'L'})
    sentinel = object()
    class DS:
        uids=['uid']; side_net=sentinel; eval_stride=1; extent_meta=True; fluid=True
        def __getitem__(self,i):
            assert self.side_net is None and self.eval_stride == 3
            assert not self.extent_meta and not self.fluid
            assert 'uid' not in ci.FORCED_SIDE
            if fail: raise RuntimeError('injected render failure')
            return {'sentinel':1}
    monkeypatch.setattr(s.w,'validate_item',lambda *args: None)
    ds = DS()
    if fail:
        with pytest.raises(RuntimeError): s.original_side(ds,ci,0)
    else:
        assert s.original_side(ds,ci,0) == {'sentinel':1}
    assert ds.side_net is sentinel and ds.eval_stride == 1 and ds.extent_meta and ds.fluid
    assert ci.FORCED_SIDE == {'uid':'R','other':'L'}


@pytest.mark.parametrize('flipped,fail',[(False,False),(True,False),(False,True)])
def test_capture_routes_and_restores_hooks(monkeypatch, flipped, fail):
    ci = SimpleNamespace(FORCED_SIDE={})
    sentinel = object()
    class DS:
        uids=['uid'];native=True;side_net=sentinel;eval_stride=1;extent_meta=True;fluid=True;window_rows=None
        def __init__(self): self.builder=SimpleNamespace(build=lambda uid: {})
        def _native_and_geometry(self,*args):
            if flipped: ci.FORCED_SIDE['uid']='R'
            return {'flipped':flipped}
        def __getitem__(self,i):
            if self.eval_stride == 3:
                assert 'uid' not in ci.FORCED_SIDE
                return {'uid':'uid','route':'original'}
            self._native_and_geometry('uid',None)
            self.builder.build('uid')
            return {'uid':'uid','error':'bad' if fail else None,'anchor':{'method':'test'}}
    monkeypatch.setattr(s,'from_rows',lambda *args: {'uid':'uid','route':'shared'})
    monkeypatch.setattr(s.w,'validate_item',lambda *args: None)
    ds=DS(); build=ds.builder.build; native=ds._native_and_geometry
    adapter=s.SharedR6.__new__(s.SharedR6)
    adapter.ds,adapter.module,adapter.active=ds,ci,False
    if fail:
        with pytest.raises(ValueError,match='parent rendering failed'):adapter[0]
    else:
        parent,item,info=adapter[0]
        assert item['route'] == ('original' if flipped else 'shared')
    assert ds.builder.build == build and ds._native_and_geometry == native
    assert not ci.FORCED_SIDE and not adapter.active


def test_precision_gates_reject_rank_and_absolute_failures():
    a=np.repeat(np.linspace(.1,.8,32)[:,None],12,axis=1)
    ok=s.compare_predictions(a,a,a+.00001)
    assert ok['fp32_pass'] and ok['fp16_status']=='PASS'
    assert not s.compare_predictions(a,a+.001)['fp32_pass']
    assert s.compare_predictions(a,a,a+.006)['fp16_status']=='FAIL'
    close=np.repeat(np.linspace(.5,.501,32)[:,None],12,axis=1)
    assert s.compare_predictions(close,close,close[::-1])['fp16_status']=='FAIL'
    with pytest.raises(ValueError):s.compare_predictions(a[:1],a[:1],a[:1])
    with pytest.raises(ValueError):s.compare_predictions(a,a*np.nan)


def test_cpu_fp16_forbidden():
    with pytest.raises(ValueError,match='requires CUDA'):
        s.predict(None,None,None,'cpu',True)


def test_roster_has_largest8_and_compressed_without_duplicates():
    counts={str(i):i+1 for i in range(100)}
    ids, largest=h.select_roster(counts,['1','2','3','99'])
    assert len(ids)==len(set(ids))==32 and set(largest)<=set(ids)
    assert {'1','2','3','99'} <= set(ids)
    assert largest == [str(i) for i in range(99,91,-1)]


def test_no_cuda_initialized():
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()


@pytest.fixture
def comparison_receipts(tmp_path, monkeypatch):
    """Synthetic predictions only; no CUDA execution or qualification evidence."""
    monkeypatch.setattr(h, 'OUT', tmp_path)
    s.w.write(tmp_path / 'selection.private.json', {})
    for arm, renderer in [('original', s.K1_INFER_SHA), ('shared', s.R6_RUNTIME_SHA)]:
        records = [dict(uid=str(i), windows=1, images_sha256='image', meta_sha256='meta',
                        render_seconds=1.0, route='shared-rows',
                        fp32=[.5 + i / 32000] * 12, fp16=[.5 + i / 32000] * 12)
                   for i in range(32)]
        receipt = dict(status='COMPLETE', adapter_sha256=s.w.sha(s.__file__),
                       harness_sha256=s.w.sha(h.__file__), worker_sha256=s.w.sha(s.w.__file__),
                       selection_sha256=s.w.sha(tmp_path / 'selection.private.json'),
                       device='cpu', environment={'test_fixture': True}, windows_driver_sha256=None,
                       renderer_sha256=renderer, records=records, fp16_requested=arm == 'shared')
        s.w.write(tmp_path / (arm + '-gpu') / 'receipt.private.json', receipt)
    aggregate = tmp_path / 'GPU-PARITY.json'
    s.w.write(aggregate, {'status': 'PASS', 'stale': True})
    return tmp_path, aggregate


@pytest.mark.parametrize('missing', [1, 32])
def test_compare_rejects_requested_missing_fp16(comparison_receipts, missing):
    out, aggregate = comparison_receipts
    path = out / 'shared-gpu/receipt.private.json'
    receipt = s.w.read(path)
    for row in receipt['records'][:missing]:
        del row['fp16']
    s.w.write(path, receipt)
    with pytest.raises(ValueError, match='requested FP16 needs results'):
        h.compare()
    assert not aggregate.exists()


@pytest.mark.parametrize('gate', ['pass', 'absolute', 'one_target_rank'])
def test_compare_enforces_both_requested_fp16_gates(comparison_receipts, gate):
    out, aggregate = comparison_receipts
    path = out / 'shared-gpu/receipt.private.json'
    receipt = s.w.read(path)
    for i, row in enumerate(receipt['records']):
        if gate == 'absolute':
            row['fp16'] = [v + .006 for v in row['fp32']]
        elif gate == 'one_target_rank':
            row['fp16'][7] = receipt['records'][31-i]['fp32'][7]
    s.w.write(path, receipt)
    if gate == 'pass':
        h.compare()
    else:
        with pytest.raises(ValueError, match='parity gate failed'):
            h.compare()
    result = s.w.read(aggregate)
    assert result['status'] == result['fp16_status'] == ('PASS' if gate == 'pass' else 'FAIL')
    assert 'stale' not in result


def test_compare_optional_fp16_stays_optional(comparison_receipts):
    out, aggregate = comparison_receipts
    path = out / 'shared-gpu/receipt.private.json'
    receipt = s.w.read(path)
    receipt['fp16_requested'] = False
    for row in receipt['records']:
        del row['fp16']
    s.w.write(path, receipt)
    h.compare()
    result = s.w.read(aggregate)
    assert result['status'] == 'PASS' and result['fp16_status'] == 'NOT_RUN'


@pytest.mark.parametrize('defect', ['under32', 'missing_fp32'])
def test_requested_fp16_cannot_bypass_gates_in_cpu_mode(comparison_receipts, defect):
    out, _ = comparison_receipts
    for arm in ['original', 'shared']:
        receipt = s.w.read(out / (arm + '-gpu') / 'receipt.private.json')
        if defect == 'under32':
            receipt['records'] = receipt['records'][:31]
        else:
            for row in receipt['records']:
                del row['fp32']
        s.w.write(out / (arm + '-cpu-smoke') / 'receipt.private.json', receipt)
    aggregate = out / 'CPU-PARITY.json'
    s.w.write(aggregate, {'status': 'PASS', 'stale': True})
    with pytest.raises(ValueError, match='requested FP16 needs results|parity gate failed'):
        h.compare(cpu_smoke=True)
    if defect == 'under32':
        assert not aggregate.exists()
    else:
        assert s.w.read(aggregate)['status'] == 'FAIL'


@pytest.mark.parametrize('cpu_smoke', [False, True])
def test_compare_invalidates_before_first_read_even_on_crash(tmp_path, monkeypatch, cpu_smoke):
    monkeypatch.setattr(h, 'OUT', tmp_path)
    aggregate = tmp_path / ('CPU-PARITY.json' if cpu_smoke else 'GPU-PARITY.json')
    s.w.write(aggregate, {'status': 'PASS'})
    def crash(path):
        assert not aggregate.exists()
        raise SystemExit('injected compare crash')
    monkeypatch.setattr(s.w, 'read', crash)
    with pytest.raises(SystemExit, match='injected compare crash'):
        h.compare(cpu_smoke)
    assert not aggregate.exists()


def test_compare_stale_source_cannot_leave_pass(comparison_receipts):
    out, aggregate = comparison_receipts
    path = out / 'original-gpu/receipt.private.json'
    receipt = s.w.read(path)
    receipt['harness_sha256'] = 'stale'
    s.w.write(path, receipt)
    with pytest.raises(ValueError, match='stale/incomplete result'):
        h.compare()
    assert not aggregate.exists()


def test_compare_real_cpu_receipts_replay(tmp_path, monkeypatch):
    """Replay historical real predictions as a test, not new inference evidence.

    Preserve all original receipts. Source binding is patched ONLY in this test
    to the backed-up v1.0.0 producer; production still rejects stale receipts.
    """
    import shutil
    recorded = h.OUT
    old_source = h.ROOT / '.local/backups/k1-shared-render-fix1/k1_shared_render_harness_v1.py'
    old_sha = s.w.sha(old_source)
    assert old_sha == '3ff831b71065df97baca2eea05bf76152d7729f1722b3ca733fbc7e3101d4675'
    for relative in ['selection.private.json', 'original-cpu-smoke/receipt.private.json',
                     'shared-cpu-smoke/receipt.private.json']:
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(recorded / relative, dest)
    sha = s.w.sha
    monkeypatch.setattr(h, 'OUT', tmp_path)
    monkeypatch.setattr(s.w, 'sha', lambda p: old_sha if Path(p).resolve() == Path(h.__file__).resolve() else sha(p))
    h.compare(cpu_smoke=True)
    result = s.w.read(tmp_path / 'CPU-PARITY.json')
    assert result['status'] == 'PASS' and result['tensors_bit_exact'] and result['fp32_max_abs'] == 0.0
