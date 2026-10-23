"""Failure-directed CPU regressions; no model, GPU, network or production data."""
import ast
import csv
import sys
import json
import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RLELossless

from tools import l1_rco_k2_worker_v101 as old
from tools import l1_rco_k2_worker_v102 as new
from tools import l1_rco_k2_failure_qualification_v1 as q


def make_dicom(root, study='1.2.3', index=0, compressed=False):
    path = root / 'test_series' / study / '1.2.4' / f'{index}.dcm'
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    fm.MediaStorageSOPInstanceUID = f'1.2.5.{index+1}'
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b'\0'*128)
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = '1.2.4'
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Rows = 8; ds.Columns = 8; ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
    ds.PixelData = np.arange(64, dtype=np.uint16).reshape(8,8).tobytes()
    if compressed: ds.compress(RLELossless)
    ds.save_as(path, enforce_file_format=True)
    return path


class Aborted(RuntimeError): pass


def clock_guard(module, monkeypatch, tmp_path):
    clock = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module, 'heartbeat', lambda *a, **k: None)
    wd = module.Watchdog(tmp_path, tmp_path/'receipt.json', False)
    def abort(reason): raise Aborted(reason)
    monkeypatch.setattr(wd, 'abort', abort)
    def check():
        # Execute one actual iteration of the production watchdog.
        calls = iter([False, True])
        wd.stop = SimpleNamespace(wait=lambda _: next(calls))
        wd.watch()
    return clock, wd, check


@pytest.mark.parametrize('worker', [old, new])
def test_large_cohort_live_scan_old_false_abort_new_progress(worker, monkeypatch, tmp_path):
    for n in range(6): make_dicom(tmp_path, index=n)
    clock, wd, check = clock_guard(worker, monkeypatch, tmp_path)
    original = pydicom.dcmread
    def slow(*a, **kw):
        clock[0] += 200
        check()  # watchdog can fire before the next actual IO completion
        return original(*a, **kw)
    monkeypatch.setattr(pydicom, 'dcmread', slow)
    if worker is old:
        with pytest.raises(Aborted, match='15 minutes'):
            worker.codec_evidence(tmp_path, 'test', ['1.2.3'])
    else:
        result = worker.codec_evidence(tmp_path, 'test', ['1.2.3'], wd.progress)
        assert clock[0] > 900 and result['files'] == 6
        assert wd.window_work == 0 and wd.phase == 'codec-header'


def test_blocked_single_read_still_aborts(monkeypatch, tmp_path):
    make_dicom(tmp_path)
    clock, wd, check = clock_guard(new, monkeypatch, tmp_path)
    def blocked(*a, **kw):
        clock[0] += 900
        check()
    monkeypatch.setattr(pydicom, 'dcmread', blocked)
    with pytest.raises(Aborted, match='15 minutes'):
        new.codec_evidence(tmp_path, 'test', ['1.2.3'], wd.progress)
    assert wd.window_work == 0 and wd.phase == 'setup'


def test_bounded_telemetry_and_silent_timeout(monkeypatch, tmp_path):
    clock, wd, check = clock_guard(new, monkeypatch, tmp_path)
    publications=[]
    monkeypatch.setattr(new,'heartbeat',lambda *a,**k: publications.append(k))
    for i in range(100_000):
        clock[0] = i * .02
        wd.progress('codec-header')
    assert len(wd.samples) <= 362 and len(publications) <= 401
    assert wd.window_work == 0 and not wd.healthy
    clock[0] += 900
    with pytest.raises(Aborted,match='15 minutes'): check()


def test_actual_uncompressed_and_rle_evidence_parity(tmp_path):
    make_dicom(tmp_path, compressed=False)
    make_dicom(tmp_path, study='1.2.6', compressed=True)
    events=[]
    a=old.codec_evidence(tmp_path,'test',['1.2.3','1.2.6'])
    b=new.codec_evidence(tmp_path,'test',['1.2.3','1.2.6'],events.append)
    assert a==b and b['files']==2 and len(b['syntaxes'])==2
    assert events==['codec-header','codec-decoded']*2
    assert all(v['decoded'] and v['pixel_shape']==[8,8] for v in b['syntaxes'].values())


def test_throughput_collapse_guard_survives_sparse_telemetry(monkeypatch,tmp_path):
    clock,wd,check=clock_guard(new,monkeypatch,tmp_path)
    for i in range(30):
        clock[0]=i*5.1;wd.progress('classifier-chunk',100);check()
    assert wd.healthy>0
    with pytest.raises(Aborted,match='throughput below'):
        for i in range(1,420):
            clock[0]+=5.1;wd.progress('classifier-chunk',1);check()


@pytest.mark.parametrize('fault',['empty','uid','truncated','decoder'])
def test_material_input_failures_remain_fail_closed(fault, tmp_path, monkeypatch):
    p=make_dicom(tmp_path)
    if fault=='empty': p.unlink()
    if fault=='uid':
        ds=pydicom.dcmread(p);ds.StudyInstanceUID='1.9.9';ds.save_as(p,enforce_file_format=True)
    if fault=='truncated': p.write_bytes(b'not a dicom')
    if fault=='decoder':
        monkeypatch.setattr(pydicom.pixels,'get_decoder',lambda _:SimpleNamespace(is_available=False))
    for worker in (old,new):
        with pytest.raises((ValueError,pydicom.errors.InvalidDicomError)):
            worker.codec_evidence(tmp_path,'test',['1.2.3'])


def test_exact_scientific_function_ast_preserved():
    def funcs(module):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=funcs(old),funcs(new)
    changed={k for k in a if a[k]!=b[k]}
    assert changed=={'Watchdog','assets_check','codec_evidence','install_label_free_source','run'}
    assert old.CONTRACT==new.CONTRACT and old.MODEL_SHA==new.MODEL_SHA
    assert old.VERSION==new.ASSET_VERSION
    for name in ('predict','validate_item','load_model','configure','read_template'):
        assert a[name]==b[name]


def test_renderer_reads_report_actual_progress_without_label_access(monkeypatch,tmp_path):
    clock,wd,check=clock_guard(new,monkeypatch,tmp_path)
    calls=[]
    class Source:
        current='1.2.3'
        split='test'
        def read(self,item):
            clock[0]+=200;check();calls.append(item);return b'unchanged bytes'
    monkeypatch.setitem(sys.modules,'probe_infer',SimpleNamespace(OneStudySource=Source))
    new.install_label_free_source(wd.progress)
    src=sys.modules['probe_infer'].OneStudySource()
    for i in range(6):
        assert src.read(f'test_series/1.2.3/1.2.4/{i}.dcm')==b'unchanged bytes'
    assert clock[0]>900 and len(calls)==6 and wd.window_work==0
    assert src.read('test.csv').decode().splitlines()==['StudyInstanceUID','1.2.3']
    assert len(calls)==6  # roster synthesized, target table never touched
    with pytest.raises(ValueError):src.read('reports.txt')


def test_watchdog_abort_exits_even_if_diagnostic_write_fails(monkeypatch,tmp_path):
    wd=new.Watchdog(tmp_path,tmp_path/'receipt.json',False)
    def failed(*a,**k):raise OSError('fixture')
    monkeypatch.setattr(new,'write',failed)
    monkeypatch.setattr(new.os,'_exit',lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as result:wd.abort('fixture')
    assert result.value.code==70


def test_qualification_selection_is_fixed_disjoint_and_covers_large_inputs():
    counts={f'1.2.{i}':i+1 for i in range(50)}
    ids=q.select_studies(counts)
    assert len(ids)==len(set(ids))==32
    assert ids[:8]==[f'1.2.{i}' for i in range(49,41,-1)]
    assert ids==q.select_studies(dict(reversed(list(counts.items()))))
    with pytest.raises(ValueError):q.select_studies(dict(list(counts.items())[:31]))
    with pytest.raises(ValueError):q.select_studies(dict(counts,bad=0))


def test_built_candidate_has_exact_worker_codec_and_qualification_guard():
    root=Path(__file__).resolve().parents[1]
    stage=root/'artifacts/experiments/L1_RCO_K2/v1.0.5-rcof0-progress'
    parent=root/'artifacts/experiments/L1_RCO_K2/v1.0.4-rcof0-codecfix'
    def payloads(source):
        found=[]
        for n in ast.walk(ast.parse(source)):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode':
                if isinstance(n.args[0],ast.Constant):found.append(base64.b64decode(n.args[0].value))
        return found
    def source(path):return ''.join(json.loads(path.read_bytes())['cells'][0]['source'])
    p=source(stage/'kernel/release.ipynb')
    v=source(stage/'qualification/release.ipynb')
    oldsource=source(parent/'kernel/release.ipynb')
    assert payloads(p)==payloads(oldsource)+[Path(new.__file__).read_bytes()]
    assert payloads(v)==payloads(p)+[Path(q.__file__).read_bytes()]
    assert 'QUALIFICATION ONLY: never submit this version' in v
    assert 'QUALIFICATION ONLY: never submit this version' not in p
    assert "str(_k2_worker)" in p and "str(assets/'source/l1_rco_k2_worker_v101.py')" not in p
    assert (stage/'kernel/kernel-metadata.json').read_bytes()==(parent/'kernel/kernel-metadata.json').read_bytes()
    assert json.loads((stage/'build-receipt.json').read_bytes())['qualified_for_submission'] is False
