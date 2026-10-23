"""Synthetic-only tests; no workspace pixels, labels, weights or models opened."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tools import exp087_series_coverage as cover


def metadata_plan():
    p = dict(uid="study", current="current", original_series=["s0","s1","current",None,"s4"], alternatives=["z","a"])
    rows = {(p["uid"],s):{cover.UID:"study",cover.SERIES:s,"Anatomical_Plane":"Coronal","Fluid_Sensitive":"1","Fat_Suppression":"1"} for s in ["current","z","a"]}
    return p,rows


def headers():
    return [dict(StudyInstanceUID="study",SeriesInstanceUID="series",ImageOrientationPatient=[1,0,0,0,0,1],
        ImagePositionPatient=[0,-i*3,0],PixelSpacing=[.5,.5],Rows=8,Columns=8,NumberOfFrames=1,
        ImageType=["ORIGINAL"],FrameOfReferenceUID="frame") for i in range(14)]


def test_lexical_choice_is_fixed_and_does_not_accept_used_or_contrast_mismatch():
    p,rows=metadata_plan()
    assert cover.lexical_alternate(p,rows)=="a"
    p["original_series"][3]="a"
    with pytest.raises(ValueError,match="strict alternate"):
        cover.lexical_alternate(p,rows)
    p,rows=metadata_plan(); rows[("study","a")]["Fat_Suppression"]="0"
    with pytest.raises(ValueError,match="strict alternate"):
        cover.lexical_alternate(p,rows)


def test_metadata_unknown_current_is_explicit_absence():
    p,rows=metadata_plan(); rows[("study","current")]["Fat_Suppression"]="unknown"
    assert cover.lexical_alternate(p,rows) is None


def test_only_odd_slot_occurrences_change_with_original_position_order():
    centers=[20,42,33,8,40,35,60,39]
    mapping,reason=cover.replacement_map(centers,np.ones(12,np.uint8),list(range(12)))
    assert mapping=={2:1,5:3}
    assert reason=="replaced"
    assert all(33<=centers[i]<=42 for i in mapping)


def test_holes_and_native_repeats_cannot_create_cross_series_or_invalid_triplets():
    mask=np.zeros(12,np.uint8); mask[1:4]=1; mask[8:11]=1
    # Two isolated triplets yield centers2/9; ordinal mapping uses these only.
    mapping,_=cover.replacement_map([33,33,42,42],mask,list(range(12)))
    assert mapping=={1:2,3:9}
    native=list(range(12)); native[2]=native[1]; native[9]=native[8]
    assert cover.replacement_map([33,42],mask,native)==({},"no_valid_alt_triplet")


@pytest.mark.parametrize("centers", [[],[-1]*12,[20,33,50]])
def test_zero_or_one_slot_window_is_noop(centers):
    assert cover.replacement_map(centers,np.ones(12,np.uint8),list(range(12)))[0]=={}


@pytest.mark.parametrize("train,epoch",[(False,0),(True,0),(True,3),(True,7)])
def test_full_input_matches_existing_StudyWindows_and_keeps_untouched_windows(train,epoch):
    rc,_=cover.helpers()
    from raptor_data import StudyWindows
    volume=np.arange(64*8*8,dtype=np.uint8).reshape(64,8,8)
    mask=np.ones(64,np.uint8); mask[20]=0
    spec={"slots":rc.SLOTS64}
    class Corpus:
        depth=64
        def volume(self,uid):return volume
        def mask(self,uid):return mask
    corpus=Corpus(); corpus.spec=spec
    baseline=StudyWindows("not-opened",["synthetic"],{"synthetic":np.zeros(12,np.float32)},4,12 if train else 24,train,"imagenet",42,window_policy="within-slot-valid")
    baseline.corpus=corpus; baseline.set_epoch(epoch)
    reference=baseline[0][0].numpy()
    original_mask=mask.copy()
    alt=np.flip(volume[32:44],axis=1).copy()
    disabled,dr=cover.windows(volume,mask,spec,"synthetic",train=train,epoch=epoch,enabled=False,res=4)
    forced,fr=cover.windows(volume,mask,spec,"synthetic",train=train,epoch=epoch,alternate=alt,alt_mask=np.ones(12,np.uint8),native_indices=list(range(12)),force_original=True,res=4)
    candidate,cr=cover.windows(volume,mask,spec,"synthetic",train=train,epoch=epoch,alternate=alt,alt_mask=np.ones(12,np.uint8),native_indices=list(range(12)),res=4)
    absent,ar=cover.windows(volume,mask,spec,"synthetic",train=train,epoch=epoch,res=4)
    assert np.array_equal(reference,disabled)
    assert np.array_equal(reference,forced)
    assert np.array_equal(reference,absent)
    assert np.array_equal(mask,original_mask)
    for key in ("centers","intensity","window_rng_after"):
        assert dr[key]==fr[key]==cr[key]==ar[key]
    replaced={int(i) for i in cr["replacements"]}
    assert replaced==set(i for j,i in enumerate([i for i,c in enumerate(dr["centers"]) if 33<=c<=42]) if j%2)
    for i in set(range(len(reference)))-replaced:
        assert np.array_equal(reference[i],candidate[i])


def test_geometry_is_internal_only_and_ownership_mismatch_is_hard_error():
    rows=headers(); before=deepcopy(rows)
    assert cover.geometry(rows,"study","series")["eligible"]
    assert rows==before
    other=deepcopy(rows)
    for h in other:h["FrameOfReferenceUID"]="different-cross-series-frame"
    assert cover.geometry(other,"study","series")["eligible"]
    rows[0]["StudyInstanceUID"]="wrong"
    with pytest.raises(ValueError,match="ownership"):
        cover.geometry(rows,"study","series")
    rows=headers(); rows[3]["ImagePositionPatient"]=rows[2]["ImagePositionPatient"]
    assert cover.geometry(rows,"study","series")["reason"]=="duplicate_projected_positions"


def test_native_selection_preserves_cached_not_native_adjacency():
    records=[dict(path=f"{i:02d}",header=h) for i,h in enumerate(headers())]
    ordered,indices=cover.native_selection(records[::-1])
    expected=np.linspace(0,12,12).round().astype(int).tolist()
    assert indices==expected
    assert [r["path"] for r in ordered]==[f"{i:02d}" for i in range(14)]
    assert any(b-a>1 for a,b in zip(indices,indices[1:]))


def test_budget_denies_before_overread_or_unbounded_deflated_header():
    caps=dict(cover.CAPS); caps["header_bytes"]=5; caps["pixel_reads"]=1
    budget=cover.Budget(caps); handle=io.BytesIO(b"0123456789")
    reader=cover.HeaderReader(handle,budget)
    assert reader.read(4)==b"0123"
    with pytest.raises(RuntimeError,match="total cap"):reader.read(2)
    assert handle.tell()==4
    with pytest.raises(RuntimeError,match="unbounded"):reader.read()
    with pytest.raises(RuntimeError,match="seek-to-end"):reader.seek(0,2)
    budget.add("pixel_reads",1)
    with pytest.raises(RuntimeError,match="pixel_reads cap"):budget.add("pixel_reads",1)
    assert budget.pixel_reads==1


def test_run_denies_pixels_without_explicit_decision(tmp_path):
    p=tmp_path/"plan.json"; p.write_text("{}")
    with pytest.raises(ValueError,match="explicit pixel"):
        cover.run(p,cover.sha(p),False)
    with pytest.raises(ValueError,match="exact plan"):
        cover.run(p,"wrong",True)
    with pytest.raises(ValueError,match="authorized supervisor"):
        cover.child_run(p,tmp_path,cover.sha(p),True)


def test_header_identity_keeps_optional_tags_and_selected_native_indices():
    rows=headers()
    class Header:
        def __getattr__(self,key):
            if key in rows[0]:return rows[0][key]
            raise AttributeError(key)
    h=cover.canonical_header(Header())
    assert h["Rows"]==8 and h["EchoTime"] is None
    assert h["ImageOrientationPatient"]==[1.,0.,0.,0.,0.,1.]


@pytest.mark.parametrize("photometric",["MONOCHROME2","MONOCHROME1"])
def test_synthetic_dicom_render_matches_passive_original_adapter(photometric):
    # All DICOM bytes are generated in memory; no real archive/cache is accessed.
    from pydicom.dataset import Dataset,FileDataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    from types import SimpleNamespace
    _,rd=cover.helpers()
    data={}; rows=[]; members=[]
    for i in range(14):
        meta=FileMetaDataset(); meta.TransferSyntaxUID=ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID="1.2.840.10008.5.1.4.1.1.4"; meta.MediaStorageSOPInstanceUID=f"1.2.3.4.{i+1}"
        ds=FileDataset(None,{},file_meta=meta,preamble=b"\0"*128)
        ds.StudyInstanceUID="1.2.3"; ds.SeriesInstanceUID="1.2.3.4"; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID=meta.MediaStorageSOPClassUID; ds.ImageOrientationPatient=[1,0,0,0,0,1]
        ds.ImagePositionPatient=[0,-3*i,0]; ds.PixelSpacing=[.5,.5]; ds.Rows=8; ds.Columns=8
        ds.SamplesPerPixel=1; ds.PhotometricInterpretation=photometric; ds.BitsAllocated=16; ds.BitsStored=16
        ds.HighBit=15; ds.PixelRepresentation=0; ds.RescaleSlope=2.; ds.RescaleIntercept=-10.
        ds.FrameOfReferenceUID="1.2.3.5"; ds.ImageType=["ORIGINAL"]
        ds.PixelData=(np.arange(64,dtype=np.uint16).reshape(8,8)+i*71).tobytes()
        f=io.BytesIO(); ds.save_as(f,enforce_file_format=True); raw=f.getvalue(); path=f"{i:02d}"
        data[path]=(raw,ds); members.append(SimpleNamespace(path=path,info=json.dumps({"file_size":len(raw)})))
        rows.append(dict(path=path,header=cover.canonical_header(ds)))
    class Source:
        def read(self,m):return data[m.path][0]
        def read_header(self,m):return data[m.path][1]
        def members(self,*args):return members
    ordered,indices=cover.native_selection(rows)
    row={cover.UID:"1.2.3",cover.SERIES:"1.2.3.4","Anatomical_Plane":"Coronal","Fluid_Sensitive":"1","Fat_Suppression":"1"}
    record=dict(metadata=row,series="1.2.3.4",selected_indices=indices,ordered_paths=[h["path"] for h in ordered],audited_headers=rows)
    budget=cover.Budget()
    actual,mask,receipt=cover.render(Source(),record,{m.path:m for m in members},budget)
    expected,expected_mask,_=rd.prepare_study("1.2.3",[row],Source(),dict(slots=[["Coronal",1,12]],depth=12,img=336,crop_mm=140.,span=[.02,.98]),split="train")
    assert np.array_equal(actual,expected) and np.array_equal(mask,expected_mask)
    assert budget.pixel_reads==len(set(indices)) and budget.rendered_bytes==1354752
    assert len(receipt["selected_decoded_plane_hashes"])==12


def test_invalid_pixel_payload_fails_instead_of_returning_original():
    from types import SimpleNamespace
    class Source:
        def read(self,m):return b"bad"
    record=dict(selected_indices=[0]*12,ordered_paths=["bad"],audited_headers=[dict(path="bad",header={})])
    with pytest.raises(Exception):
        cover.render(Source(),record,{"bad":SimpleNamespace(path="bad",info='{"file_size":3}')},cover.Budget())


def test_candidate_preprocessing_does_not_consume_global_rng():
    import random
    import torch
    rc,_=cover.helpers()
    volume=np.arange(64*4*4,dtype=np.uint8).reshape(64,4,4)
    before=(random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
    cover.windows(volume,np.ones(64,np.uint8),{"slots":rc.SLOTS64},"synthetic",train=True,epoch=3,
        alternate=volume[32:44],alt_mask=np.ones(12,np.uint8),native_indices=list(range(12)),res=4)
    assert random.getstate()==before[0]
    after=np.random.get_state()
    assert after[0]==before[1][0] and np.array_equal(after[1],before[1][1]) and after[2:]==before[1][2:]
    assert torch.equal(torch.random.get_rng_state(),before[2])


def fake_supervised_run(monkeypatch,tmp_path,*,elapsed=1.,failure=None):
    """Fake Popen/psutil only: no subprocess, source pixels, or model exists."""
    import psutil
    from types import SimpleNamespace
    output=tmp_path/"output"; fixture=tmp_path/"fixture.json"; plan=tmp_path/"plan.json"
    fixture.write_text(json.dumps({"pairs":[{}]*4}))
    plan.write_text(json.dumps(dict(fixture=str(fixture),fixture_sha256=cover.sha(fixture),
        output=str(output),environment={},interpreter="not-executed")))
    monkeypatch.setattr(cover,"local",lambda p:Path(p))
    monkeypatch.setattr(cover,"uri",lambda p:str(p))
    monkeypatch.setattr(cover,"verify_bindings",lambda *args:None)
    ticks=iter([0.,elapsed]); monkeypatch.setattr(cover.time,"monotonic",lambda:next(ticks))
    fake=SimpleNamespace(pid=777,returncode=None,running=True)
    children=[]
    class Handle:
        def __init__(self,pid):self.pid=pid; self.running=True
        def children(self,recursive=True):return children if self.pid==777 else []
        def is_running(self):return fake.running if self.pid==777 else self.running
        def kill(self):
            self.running=False
            if self.pid==777:fake.running=False; fake.returncode=-9
        def memory_info(self):
            if failure=="monitor":raise OSError("synthetic monitor exception")
            return SimpleNamespace(rss=1024)
    child=Handle(777); descendant=Handle(778)
    if failure:children.append(descendant)
    handles={777:child,778:descendant,os.getpid():Handle(os.getpid())}
    monkeypatch.setattr(psutil,"Process",lambda pid:handles[pid])
    monkeypatch.setattr(psutil,"wait_procs",lambda values,timeout:(list(values),[]))
    def poll():
        if failure:return fake.returncode
        fake.running=False; fake.returncode=0; return 0
    fake.poll=poll; fake.kill=child.kill; fake.wait=lambda timeout:fake.returncode
    def popen(*args,**kwargs):
        (output/"receipt.json").write_text('{"status":"CPU_ENGINEERING_COMPLETE"}')
        return fake
    monkeypatch.setattr(cover.subprocess,"Popen",popen)
    original_dump=cover.dump
    if failure=="job_write":
        def dump(path,value):
            if Path(path).name=="job.json":raise OSError("synthetic job receipt error")
            original_dump(path,value)
        monkeypatch.setattr(cover,"dump",dump)
    return plan,output,fake,descendant


@pytest.mark.parametrize("elapsed",[600.,600.001])
def test_supervisor_exited_between_polls_still_fails_at_final_deadline(monkeypatch,tmp_path,elapsed):
    plan,output,fake,_=fake_supervised_run(monkeypatch,tmp_path,elapsed=elapsed)
    with pytest.raises(RuntimeError,match="CPU pilot failed"):
        cover.run(plan,cover.sha(plan),True)
    receipt=json.loads((output/"supervisor.json").read_text())
    assert receipt["status"]=="FAIL" and receipt["killed"]=="wall_final"
    assert receipt["returncode"]==0 and receipt["wall_seconds"]>=600


def test_supervisor_under_deadline_normal_exit_passes(monkeypatch,tmp_path):
    plan,output,_,_=fake_supervised_run(monkeypatch,tmp_path,elapsed=599.9)
    cover.run(plan,cover.sha(plan),True)
    assert json.loads((output/"supervisor.json").read_text())["status"]=="PASS"


@pytest.mark.parametrize("failure",["job_write","monitor"])
def test_post_spawn_exceptions_cleanup_child_and_descendant_and_write_fail(monkeypatch,tmp_path,failure):
    plan,output,fake,descendant=fake_supervised_run(monkeypatch,tmp_path,failure=failure)
    with pytest.raises(RuntimeError,match="CPU pilot failed"):
        cover.run(plan,cover.sha(plan),True)
    receipt=json.loads((output/"supervisor.json").read_text())
    assert receipt["status"]=="FAIL" and "synthetic" in receipt["error"]
    assert not fake.running and not descendant.running
    assert set(receipt["cleanup_killed_pids"])=={777,778}
    assert receipt["cleanup_errors"]==[]
