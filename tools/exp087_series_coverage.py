"""Bounded, explicitly authorized CPU-only EXP087 input engineering pilot.

Import is passive. `freeze` reads existing metadata only; `run` requires an
explicit pixel flag and exact plan SHA. No training/model/label APIs are used.
Original source files and corpora are immutable. No automatic retries.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UID, SERIES = "StudyInstanceUID", "SeriesInstanceUID"
POLICY = "coronal-fluid-second-series-preserve-baseline-centers-v1"
EVIDENCE = "docs/handoffs/evidence/2026-10/2026-10-04-exp087-coverage-pilot"
PREVIOUS = "artifacts/experiments/EXP081_CORONAL_SPAN_FIXED_F2"
GROUP = "artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL"
CORPUS = "storage/cache/derived/RAPTOR64_336_C140_S02"
ARCHIVE = "storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip"
INDEX = ".local/kneeops/archive-index/9175e113e4a00fe679c4088b.sqlite"
EXPECTED_T = "a69580bd6a68dc802cce4f214ac7afffb9742a7f6ad7963c33245b9de72cace5"
EXPECTED_V = "56ae86a4c180cc46d28e2a192485d133d58f4d05907b67fa18076f199f507dc4"
BASELINE_METADATA_SHA = {
    f"workspace://{CORPUS}/ids.npy": "48cf9dcbd34276c898dabfd09d901d5288b272522328016076da0b9189a3246a",
    f"workspace://{CORPUS}/masks.npy": "011696644468b5a624c9e7862be5a853444e1c98640a77118ba996fd6b437ea2",
}
CAPS = dict(pairs_min=4, pairs_max=8, pixel_reads=192, headers=2048,
            header_bytes=64*1024**2, pixel_bytes=1024**3, rendered_bytes=21676032,
            rss_bytes=4*1024**3, wall_seconds=600)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    return digest(dict(shape=list(a.shape), dtype=str(a.dtype),
                       bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest()))


def uid_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def local(path):
    p = Path(str(path).removeprefix("workspace://"))
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("path outside workspace")
    return p


def uri(path):
    return "workspace://" + local(path).relative_to(ROOT).as_posix()


def helpers():
    # Explicit passive helpers, never import raptor_corpus or an experiment runner.
    directory = str(ROOT / "training/src")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import raptor_common as rc
    import raptor_dicom as rd
    return rc, rd


def flag(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number in (0, 1) else None
    except (TypeError, ValueError):
        return None


def lexical_alternate(plan, metadata):
    """Use the complete previously frozen alternate list, not max-span picks."""
    original = plan["original_series"]
    if len(original) != 5 or original[2] != plan["current"]:
        raise ValueError("original five-slot selection mismatch")
    uid, current = plan["uid"], plan["current"]
    row = metadata.get((uid, current))
    if row is None or row.get("Anatomical_Plane") != "Coronal" or flag(row.get("Fluid_Sensitive")) != 1 or flag(row.get("Fat_Suppression")) not in (0, 1):
        return None
    candidates = plan["alternatives"]
    if len(candidates) != len(set(candidates)):
        raise ValueError("duplicate alternate series metadata")
    for series in candidates:
        other = metadata.get((uid, series))
        if (series in original or other is None or other.get(UID) != uid or other.get(SERIES) != series or
                other.get("Anatomical_Plane") != "Coronal" or flag(other.get("Fluid_Sensitive")) != 1 or
                flag(other.get("Fat_Suppression")) != flag(row.get("Fat_Suppression"))):
            raise ValueError("invalid frozen strict alternate list")
    return min(candidates) if candidates else None


def geometry(headers, uid, series):
    """EXP081 geometry gate, copied as pure logic; no anatomy-overlap claim."""
    def reject(reason):
        return dict(eligible=False, reason=reason, headers=len(headers))
    def vector(value, n):
        try:
            a = np.asarray(value, dtype=float).reshape(-1)
            return a if len(a) == n and np.isfinite(a).all() else None
        except (ValueError, TypeError):
            return None
    def angle(a, b):
        return float(np.degrees(np.arccos(np.clip(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)), -1, 1))))
    normals, axes, positions, frames = [], [], [], []
    if not headers:
        return reject("no_headers")
    for h in headers:
        if h.get(UID) != uid or h.get(SERIES) != series:
            raise ValueError("DICOM ownership mismatch")
        iop, ipp, spacing = vector(h.get("ImageOrientationPatient"), 6), vector(h.get("ImagePositionPatient"), 3), vector(h.get("PixelSpacing"), 2)
        if iop is None or ipp is None or spacing is None:
            return reject("missing_or_nonfinite_geometry")
        try:
            if int(h.get("Rows", 0)) <= 0 or int(h.get("Columns", 0)) <= 0 or min(spacing) <= 0:
                return reject("invalid_matrix_or_spacing")
            if int(h.get("NumberOfFrames", 1)) != 1:
                return reject("multiframe")
        except (ValueError, TypeError, OverflowError):
            return reject("invalid_matrix_or_frames")
        kinds = h.get("ImageType", [])
        kinds = kinds.split("\\") if isinstance(kinds, str) else kinds
        if any("MOSAIC" in str(v).upper() for v in kinds):
            return reject("mosaic")
        a, b = iop[:3], iop[3:]
        if max(abs(np.linalg.norm(a)-1), abs(np.linalg.norm(b)-1), abs(np.dot(a,b))) > .001:
            return reject("invalid_iop")
        normal = np.cross(a,b); normal /= np.linalg.norm(normal)
        normals.append(normal); axes.append((a,b)); positions.append(ipp)
        frames.append(str(h.get("FrameOfReferenceUID") or ""))
    if not all(frames) or len(set(frames)) != 1:
        return reject("missing_or_inconsistent_frame")
    if int(np.argmax(np.abs(normals[0]))) != 1:
        return reject("not_coronal_dominant_normal")
    if max(angle(normals[0], n) for n in normals) > 1 or max(max(angle(axes[0][0],a), angle(axes[0][1],b)) for a,b in axes) > 1:
        return reject("orientation_drift")
    projections = np.sort(np.asarray(positions) @ normals[0])
    if len(projections) < 3:
        return reject("fewer_than_three_positions")
    gaps = np.diff(projections)
    if np.any(gaps <= .001):
        return reject("duplicate_projected_positions")
    if float(gaps.max()/np.median(gaps)) > 1.5:
        return reject("irregular_gaps")
    return dict(eligible=True, reason="eligible", headers=len(headers),
                extent_mm=float(projections[-1]-projections[0]), frame_uid=frames[0])


def native_selection(headers):
    records = []
    for h in sorted(headers, key=lambda r: r["path"]):
        d = h["header"]; iop = np.asarray(d["ImageOrientationPatient"], float)
        position = float(np.dot(np.asarray(d["ImagePositionPatient"], float), np.cross(iop[:3], iop[3:])))
        records.append((position, h))
    records.sort(key=lambda r: r[0])  # Stable ties preserve lexical paths.
    n = len(records); low = int(.02*n); high = max(low, int(.98*n)-1)
    indices = np.linspace(low, high, 12).round().astype(int).tolist()
    return [r[1] for r in records], indices


def source_stat(path):
    s = local(path).stat()
    return dict(path=uri(path), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def pick_subset(ids):
    found = {}
    with local(f"{CORPUS}/picks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["uid"] in ids:
                if row["uid"] in found:
                    raise ValueError("duplicate stored pick")
                found[row["uid"]] = row
    if set(found) != set(ids):
        raise ValueError("stored pick missing")
    return found


def freeze(directory):
    directory = local(directory)
    directory.mkdir(parents=True, exist_ok=False)
    membership = read_json(local(f"{GROUP}/ordered-ids.json"))
    train, val = membership["train"], membership["val"]
    if (len(train) != 3455 or len(val) != 888 or len(set(train)) != len(train) or
            set(train) & set(val) or uid_hash(train) != EXPECTED_T or uid_hash(val) != EXPECTED_V):
        raise ValueError("G1 exact membership mismatch")
    previous = read_json(local(f"{PREVIOUS}/metadata-map.json"))
    metadata = {}
    for r in previous["series"]:
        key = (r["uid"], r["series"])
        if key in metadata:
            raise ValueError("duplicate metadata key")
        metadata[key] = r["metadata"]
    audited = {}
    for r in read_json(local(f"{PREVIOUS}/geometry.json"))["records"]:
        key = (r["uid"], r["series"])
        if key in audited:
            raise ValueError("duplicate geometry key")
        audited[key] = r
    pairs, visited = [], 0
    for p in sorted(previous["plans"], key=lambda r: r["uid"]):
        if p["uid"] not in set(train):
            continue
        visited += 1
        alt = lexical_alternate(p, metadata)
        if alt is None or (p["uid"], p["current"]) not in audited or (p["uid"], alt) not in audited:
            continue
        records = []
        for role, series in (("original", p["current"]), ("alternate", alt)):
            r = audited[(p["uid"], series)]
            if r["metadata"] != metadata[(p["uid"], series)]:
                raise ValueError("header/metadata linkage mismatch")
            measured = geometry([h["header"] for h in r["headers"]], p["uid"], series)
            if measured["eligible"] != r["geometry"]["eligible"]:
                raise ValueError("stored header geometry gate mismatch")
            ordered, indices = native_selection(r["headers"])
            records.append(dict(role=role, series=series, metadata=r["metadata"],
                geometry=measured, audited_headers=r["headers"], ordered_paths=[h["path"] for h in ordered],
                selected_indices=indices, selected_paths=[ordered[i]["path"] for i in indices]))
        pairs.append(dict(uid=p["uid"], original_series=p["original_series"], series=records))
        if len(pairs) == CAPS["pairs_max"]:
            break
    if len(pairs) < CAPS["pairs_min"]:
        dump(directory / "HOLD.json", dict(status="HOLD_FEWER_THAN4", pairs=len(pairs), pixels=0))
        raise ValueError("fewer than4 exact T audited pairs; no expansion authorized")
    picks = pick_subset({p["uid"] for p in pairs})
    for p in pairs:
        pick = picks[p["uid"]]
        if [s.get("series") if s else None for s in pick["slots"]] != p["original_series"]:
            raise ValueError("current baseline picks disagree with frozen source")
        if pick["slots"][2]["slices"] != p["series"][0]["selected_indices"]:
            raise ValueError("original native indices mismatch")
    bindings = [f"{GROUP}/ordered-ids.json", f"{PREVIOUS}/metadata-map.json", f"{PREVIOUS}/geometry.json",
        f"{CORPUS}/manifest.json", "training/src/raptor_common.py", "training/src/raptor_dicom.py",
        "training/src/raptor_data.py", "tools/exp087_series_coverage.py", "tests/test_exp087_series_coverage.py",
        f"{EVIDENCE}/root-cpu-implementation-decision.json"]
    manifest = read_json(local(f"{CORPUS}/manifest.json"))
    fixture = dict(schema="exp087-coverage-fixture-v1", created_at=now(), policy=POLICY,
        train_uid_sha256=EXPECTED_T, validation_uid_sha256=EXPECTED_V,
        fixture_uid_sha256=uid_hash([p["uid"] for p in pairs]), pairs=pairs, pick_subset_sha256=digest(picks),
        prior_plan_rows_visited=visited, count_scope="bounded fixture selection only, not treatment prevalence",
        source_sha256={uri(p):sha(local(p)) for p in bindings},
        baseline_metadata_sha256=BASELINE_METADATA_SHA,
        baseline_metadata_hash_scope="existing G1 bindings reused here; actual small-file hashes checked before runtime pixel access",
        large_source_stat=[source_stat(ARCHIVE), source_stat(INDEX)],
        archive_directory_sha256_recorded=manifest["archive"]["directory_sha256"],
        cache_shards_recorded=manifest["shards"],
        original_cache_stat=[source_stat(f"{CORPUS}/{r['file']}") for r in manifest["shards"]],
        caps=CAPS, cpu_workers=dict(requested=12,effective=1,internal=1,
            reason="<=16 tiny series, deterministic serial archive IO, bound RSS and avoid ZIP contention; no pool"),
        access=dict(new_headers=0,pixels=0,models=0,labels_or_predictions_files=0,GPU=0))
    dump(directory / "fixture.json", fixture)
    output = ".local/exp087-coverage/pilot-v1"
    plan = dict(schema="exp087-coverage-execution-plan-v1", created_at=now(), status="READY_FOR_ROOT_REVIEW_NOT_EXECUTED",
        fixture=uri(directory / "fixture.json"), fixture_sha256=sha(directory / "fixture.json"),
        source_sha256=fixture["source_sha256"], output=uri(output), caps=CAPS,
        interpreter=str(Path(sys.executable).resolve()), policy=POLICY,
        exact_run_argv=[str(Path(sys.executable).resolve()),str(ROOT/"tools/exp087_series_coverage.py"),"run","--plan",str(directory/"execution-plan.json"),"--plan-sha256","<SHA256_OF_THIS_PLAN>","--execute-pixels"],
        environment=dict(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"),
        output_immutable=True, auto_retry=0, no_models_or_labels=True,
        goal="engineering replay/no-op/RNG/input difference only; no clinical information or efficacy claim",
        existing_overlay_policy="metadata receipts reusable; regenerate at most192 selected planes to measure this decoder and bind SOP/raw hashes; no prior overlays assumed equivalent")
    dump(directory / "execution-plan.json", plan)
    print(json.dumps(dict(status="METADATA_FROZEN_NO_PIXELS", pairs=len(pairs),
        fixture_sha256=plan["fixture_sha256"], plan_sha256=sha(directory/"execution-plan.json"))))


def replacement_map(centers, alt_mask, native_indices):
    if len(alt_mask) != 12 or len(native_indices) != 12:
        raise ValueError("alternate depth must be12")
    valid = [c for c in range(1,11) if np.all(np.asarray(alt_mask)[c-1:c+2] > 0)
             and len(set(native_indices[c-1:c+2])) == 3]
    positions = [i for i,c in enumerate(centers) if 33 <= c <= 42]
    if not valid:
        return {}, "no_valid_alt_triplet"
    mapping = {i: valid[round(((centers[i]-33)/9)*(len(valid)-1))] for i in positions[1::2]}
    return mapping, "replaced" if mapping else "fewer_than_two_original_slot_windows"


def windows(volume, mask, spec, uid, *, seed=42, epoch=0, train=False, alternate=None, alt_mask=None,
            native_indices=None, enabled=True, force_original=False, res=224):
    rc, _ = helpers()
    rng = rc.study_rng(seed,epoch,uid) if train else None
    centers = rc.select_centers(mask,64,12 if train else 24,spec,"within-slot-valid",rng)
    state_after = repr(rng.getstate()) if rng else None
    x = rc.windows_from(volume,centers,res)
    mapping, reason = {}, "disabled" if not enabled else "absent_alternate"
    if enabled and alternate is not None:
        mapping, reason = replacement_map(centers, alt_mask, native_indices)
        if force_original:
            x2 = rc.windows_from(volume,[centers[i] for i in mapping],res)
        else:
            x2 = rc.windows_from(alternate,list(mapping.values()),res)
        for j,i in enumerate(mapping):
            x[i] = x2[j]
    intensity = rc.study_rng(seed,epoch,uid,"intensity").random() if train else None
    if train:
        x = np.clip(x * (1+(intensity-.5)*.20),0,1)
    mean = np.array([.485,.456,.406],np.float32).reshape(3,1,1)
    std = np.array([.229,.224,.225],np.float32).reshape(3,1,1)
    x = (x-mean)/std
    if not np.isfinite(x).all():
        raise ValueError("nonfinite inputs")
    return x,dict(centers=centers,replacements={str(k):v for k,v in mapping.items()},reason=reason,
                  window_rng_after=state_after,intensity=intensity)


class Budget:
    def __init__(self, caps=None):
        self.caps = CAPS if caps is None else caps
        self.started=time.monotonic(); self.headers=self.header_bytes=self.pixel_reads=self.pixel_bytes=self.rendered_bytes=0
    def check(self):
        if time.monotonic()-self.started >= self.caps["wall_seconds"]:
            raise RuntimeError("wall cap exceeded")
    def add(self, key, value):
        self.check()
        if value < 0 or getattr(self,key)+value > self.caps[key]:
            raise RuntimeError(f"{key} cap exceeded")
        setattr(self,key,getattr(self,key)+value)
    def report(self):
        return dict(seconds=time.monotonic()-self.started, **{k:getattr(self,k) for k in ("headers","header_bytes","pixel_reads","pixel_bytes","rendered_bytes")})


class HeaderReader:
    def __init__(self, handle, budget):
        self.handle,self.budget,self.used=handle,budget,0
    def tell(self):
        return self.handle.tell()
    def seek(self, offset, whence=0):
        self.budget.check()
        if whence not in (0,1):
            raise RuntimeError("header seek-to-end denied")
        target=offset if whence==0 else self.tell()+offset
        if not 0 <= target <= 2*1024**2:
            raise RuntimeError("header seek cap")
        return self.handle.seek(offset,whence)
    def read(self,size=-1):
        if size is None or size < 0 or self.used+size > 2*1024**2 or self.tell()+size > 2*1024**2:
            raise RuntimeError("unbounded/oversized header read")
        self.budget.check()
        if self.budget.header_bytes+size > self.budget.caps["header_bytes"]:
            raise RuntimeError("header total cap")
        value=self.handle.read(size); self.used+=len(value); self.budget.add("header_bytes",len(value)); return value
    def readinto(self,buffer):
        value=self.read(len(buffer)); buffer[:len(value)]=value; return len(value)


def canonical_header(ds):
    result={UID:str(getattr(ds,UID,"")),SERIES:str(getattr(ds,SERIES,""))}
    for key in ("ImageOrientationPatient","ImagePositionPatient","PixelSpacing"):
        value=getattr(ds,key,None); result[key]=None if value is None else [float(v) for v in value]
    for key in ("Rows","Columns","NumberOfFrames"):
        result[key]=int(getattr(ds,key,1 if key=="NumberOfFrames" else 0))
    result["FrameOfReferenceUID"]=str(getattr(ds,"FrameOfReferenceUID","") or "")
    kind=getattr(ds,"ImageType",[]) or []
    result["ImageType"]=kind.split("\\") if isinstance(kind,str) else [str(x) for x in kind]
    for key in ("EchoTime","RepetitionTime","SliceThickness","SpacingBetweenSlices"):
        value=getattr(ds,key,None); result[key]=None if value is None else float(value)
    return result


def zip_entry(member):
    entry=zipfile.ZipInfo(member.path)
    for k,v in json.loads(member.info).items():
        setattr(entry,k,v)
    return entry


def verify_headers(source, pair, record, budget):
    import pydicom
    members=source.members("train",pair["uid"],record["series"])
    stored={h["path"]:h for h in record["audited_headers"]}
    if len(stored)!=len(record["audited_headers"]) or [m.path for m in members]!=sorted(stored):
        raise ValueError("archive member list mismatch")
    headers,sops,syntaxes=[],[],[]
    for m in members:
        budget.add("headers",1); entry=zip_entry(m); old=stored[m.path]
        for name,key in (("CRC","zip_crc32"),("header_offset","zip_offset"),("file_size","zip_member_bytes"),("compress_size","zip_compressed_bytes")):
            if getattr(entry,name)!=old[key]:
                raise ValueError("archive index/member provenance mismatch")
        with source.archive.open(entry,"r") as handle:
            ds=pydicom.dcmread(HeaderReader(handle,budget),stop_before_pixels=True)
        if "PixelData" in ds or canonical_header(ds)!=old["header"]:
            raise ValueError("actual header differs from frozen audit")
        sop=str(getattr(ds,"SOPInstanceUID","") or "")
        if not sop:
            raise ValueError("missing SOPInstanceUID")
        headers.append(canonical_header(ds)); sops.append(sop)
        syntaxes.append(str(ds.file_meta.TransferSyntaxUID))
    if len(sops)!=len(set(sops)):
        raise ValueError("duplicate SOP within series")
    return {m.path:m for m in members},dict(geometry=geometry(headers,pair["uid"],record["series"]),
        sops=sops,transfer_syntaxes=sorted(set(syntaxes)),members=len(members))


def render(source, record, members, budget):
    """Healthy-input math copied from passive raptor_dicom.prepare_study."""
    import cv2
    import pydicom
    from pydicom.pixels import apply_modality_lut
    arrays,raw,decoded={},[],[]
    indices=record["selected_indices"]
    ordered={h["path"]:h for h in record["audited_headers"]}
    for i in sorted(set(indices)):
        budget.check(); path=record["ordered_paths"][i]; m=members[path]
        size=json.loads(m.info)["file_size"]
        budget.add("pixel_reads",1); budget.add("pixel_bytes",size)
        payload=source.read(m)
        if len(payload)!=size:
            raise ValueError("pixel payload size mismatch")
        ds=pydicom.dcmread(io.BytesIO(payload))
        if str(ds.StudyInstanceUID)!=record["metadata"][UID] or str(ds.SeriesInstanceUID)!=record["series"]:
            raise ValueError("decoded ownership mismatch")
        a=apply_modality_lut(ds.pixel_array,ds).astype(np.float32)
        if a.ndim!=2 or not np.isfinite(a).all():
            raise ValueError("nonfinite/non2D pixels")
        if str(getattr(ds,"PhotometricInterpretation",""))=="MONOCHROME1":
            a=a.max()-a
        arrays[i]=a
        raw.append(dict(path=path,sha256=hashlib.sha256(payload).hexdigest(),bytes=size))
        decoded.append(dict(native_index=i,path=path,shape=list(a.shape),dtype=str(a.dtype),sha256=array_hash(a)))
    lo,hi=np.percentile(np.concatenate([arrays[i].ravel() for i in indices]),[2.,98.])
    vol=np.zeros((12,336,336),np.uint8)
    for j,i in enumerate(indices):
        budget.check(); a=arrays[i]; spacing=ordered[record["ordered_paths"][i]]["header"]["PixelSpacing"][0]
        normalized=np.clip((a-lo)/(hi-lo+1e-6),0,1)
        height,width=normalized.shape
        crop=min(int(round(140./max(spacing,1e-3))),min(height,width))
        if crop<=0:
            raise ValueError("empty crop")
        y,x=(height-crop)//2,(width-crop)//2
        vol[j]=(cv2.resize(normalized[y:y+crop,x:x+crop],(336,336),interpolation=cv2.INTER_AREA)*255).astype(np.uint8)
    mask=(vol.reshape(12,-1).sum(1)>0).astype(np.uint8)
    budget.add("rendered_bytes",vol.nbytes)
    return vol,mask,dict(normalization=[float(lo),float(hi)],raw_sources=raw,decoded=decoded,
        selected_decoded_plane_hashes=[next(r["sha256"] for r in decoded if r["native_index"]==i) for i in indices],
        volume_hash=array_hash(vol),mask_hash=array_hash(mask))


def verify_bindings(plan, fixture):
    if fixture["caps"]!=CAPS or plan["caps"]!=CAPS or fixture["policy"]!=POLICY:
        raise ValueError("policy/budget mismatch")
    for path,expected in plan["source_sha256"].items():
        if sha(local(path))!=expected:
            raise ValueError(f"source changed: {path}")
    if fixture["source_sha256"]!=plan["source_sha256"]:
        raise ValueError("plan/fixture source bindings differ")
    if fixture.get("baseline_metadata_sha256")!=BASELINE_METADATA_SHA:
        raise ValueError("original ids/mask binding absent or altered")
    for path,expected in fixture["baseline_metadata_sha256"].items():
        if sha(local(path))!=expected:
            raise ValueError("original ids/mask changed")
    for expected in fixture["large_source_stat"]+fixture["original_cache_stat"]:
        if source_stat(expected["path"])!=expected:
            raise ValueError("large source stat changed; no full hash retry")
    if digest(pick_subset({p["uid"] for p in fixture["pairs"]}))!=fixture["pick_subset_sha256"]:
        raise ValueError("current selected picks changed")


def child_run(plan_path, output, expected_sha, execute_pixels):
    if (not execute_pixels or sha(plan_path)!=expected_sha or
            os.environ.get("EXP087_AUTHORIZED_PLAN_SHA256")!=expected_sha):
        raise ValueError("child requires exact authorized supervisor plan")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import cv2
    import torch
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    plan=read_json(plan_path)
    if sha(local(plan["fixture"]))!=plan["fixture_sha256"] or local(plan["output"])!=local(output):
        raise ValueError("child fixture/output binding mismatch")
    fixture=read_json(local(plan["fixture"]))
    verify_bindings(plan,fixture)
    rc,rd=helpers(); budget=Budget(); result=[]
    corpus=rc.Corpus(local(CORPUS))
    try:
        with rd.IndexedZipDicomSource(local(ARCHIVE),local(INDEX)) as source:
            for pair in fixture["pairs"]:
                budget.check(); sr=[]; member_maps=[]
                for record in pair["series"]:
                    members,header=verify_headers(source,pair,record,budget)
                    sr.append(header); member_maps.append(members)
                reason=None
                if not all(r["geometry"]["eligible"] for r in sr):
                    reason="geometry_ineligible"
                elif set(sr[0]["sops"]) & set(sr[1]["sops"]):
                    reason="shared_SOP"
                original,original_mask,original_receipt=render(source,pair["series"][0],member_maps[0],budget)
                volume=corpus.volume(pair["uid"]); mask=corpus.mask(pair["uid"])
                if not np.array_equal(original,volume[32:44]) or not np.array_equal(original_mask,mask[32:44]):
                    raise ValueError("original12 replay differs from original immutable cache")
                alt=alt_mask=alt_receipt=None
                if reason is None:
                    alt,alt_mask,alt_receipt=render(source,pair["series"][1],member_maps[1],budget)
                    if np.array_equal(alt,original):
                        reason="sampled_render_duplicate"
                checks=[]
                for train,epoch in [(False,0)]+[(True,e) for e in range(8)]:
                    base,br=windows(volume,mask,corpus.spec,pair["uid"],train=train,epoch=epoch,enabled=False)
                    no_op,nr=windows(volume,mask,corpus.spec,pair["uid"],train=train,epoch=epoch,alternate=alt,
                        alt_mask=alt_mask,native_indices=pair["series"][1]["selected_indices"],force_original=True)
                    if not np.array_equal(base,no_op) or any(br[k]!=nr[k] for k in ("centers","window_rng_after","intensity")):
                        raise ValueError("forced-original no-op/RNG mismatch")
                    treatment,tr=windows(volume,mask,corpus.spec,pair["uid"],train=train,epoch=epoch,
                        alternate=alt if reason is None else None,alt_mask=alt_mask,
                        native_indices=pair["series"][1]["selected_indices"])
                    changed=[i for i in range(len(base)) if not np.array_equal(base[i],treatment[i])]
                    replaced={int(i) for i in tr["replacements"]}
                    if set(changed)-replaced or any(br[k]!=tr[k] for k in ("centers","window_rng_after","intensity")):
                        raise ValueError("untouched windows/RNG changed")
                    checks.append(dict(train=train,epoch=epoch,centers=br["centers"],intensity=br["intensity"],
                        replacement_map=tr["replacements"],changed_windows=changed,reason=reason or tr["reason"],
                        baseline_hash=array_hash(base),treatment_hash=array_hash(treatment),noop_exact=True,RNG_exact=True))
                stem=output/pair["uid"]
                np.save(str(stem)+"-original.npy",original)
                if alt is not None:
                    np.save(str(stem)+"-alternate.npy",alt)
                result.append(dict(uid=pair["uid"],header=sr,reason=reason,original=original_receipt,
                    alternate=alt_receipt,checks=checks,original_replay_exact=True,
                    decoded_selected_equal=None if alt_receipt is None else original_receipt["selected_decoded_plane_hashes"]==alt_receipt["selected_decoded_plane_hashes"]))
                dump(output/(pair["uid"]+"-receipt.json"),result[-1])
                print(json.dumps(dict(uid=pair["uid"],completed_pairs=len(result),budget=budget.report())),flush=True)
        dump(output/"receipt.json",dict(status="CPU_ENGINEERING_COMPLETE",completed_at=now(),policy=POLICY,
            pairs=result,budget=budget.report(),caps=CAPS,workers=fixture["cpu_workers"],
            measured_scope=dict(pixel_bytes="charged indexed member size before read; successful read length asserted equal",
                active_eval_studies=sum(any(not c["train"] and bool(c["changed_windows"]) for c in p["checks"]) for p in result),
                active_train_studies=sum(any(c["train"] and bool(c["changed_windows"]) for c in p["checks"]) for p in result),
                sampled_render_duplicates=sum(p["reason"]=="sampled_render_duplicate" for p in result),
                source_identity="index CRC/offset/size plus streamed header and selected full member CRC; original selected slot replay; no full ZIP/shard rehash"),
            environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,cv2=cv2.__version__,
                torch=torch.__version__,cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"]),
            clinical_or_performance_claim=False,labels_or_predictions_files=0,models=0,GPU=0,
            limits=["engineering fixture is not treatment prevalence","pixel difference is not clinical information","no patient independence proof","no GPU training authorization"]))
    except Exception as exc:
        dump(output/"FAIL.json",dict(status="FAIL",error=f"{type(exc).__name__}: {exc}",budget=budget.report(),completed_pairs=len(result)))
        raise


def run(plan_path, expected_sha, execute_pixels):
    if not execute_pixels or sha(plan_path)!=expected_sha:
        raise ValueError("explicit pixel decision and exact plan hash required")
    plan=read_json(plan_path); fixture_path=local(plan["fixture"])
    if sha(fixture_path)!=plan["fixture_sha256"]:
        raise ValueError("fixture hash mismatch")
    fixture=read_json(fixture_path); verify_bindings(plan,fixture)
    if not CAPS["pairs_min"]<=len(fixture["pairs"])<=CAPS["pairs_max"]:
        raise ValueError("pair count out of frozen range")
    output=local(plan["output"]); output.mkdir(parents=True,exist_ok=False)
    import psutil
    env=os.environ.copy(); env.update(plan["environment"])
    env["EXP087_AUTHORIZED_PLAN_SHA256"]=expected_sha
    command=[plan["interpreter"],str(Path(__file__).resolve()),"_child","--plan",str(plan_path),"--output",str(output),"--plan-sha256",expected_sha,"--execute-pixels"]
    peak=0; started=time.monotonic(); killed=None; process=None; failure=None
    tracked={}; cleanup_errors=[]; cleanup_killed=[]
    try:
        with (output/"stdout.log").open("x",encoding="utf-8") as stdout, (output/"stderr.log").open("x",encoding="utf-8") as stderr:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            # Track immediately: even writing job.json can fail after a child starts.
            try:
                child=psutil.Process(process.pid)
                tracked.update({q.pid:q for q in [child]+child.children(recursive=True)})
            except psutil.NoSuchProcess:
                pass
            dump(output/"job.json",dict(pid=process.pid,started_at=now(),argv=command,plan_sha256=expected_sha,
                fixture_sha256=plan["fixture_sha256"],caps=CAPS,automatic_retry=0))
            while process.poll() is None:
                try:
                    child=psutil.Process(process.pid); tree=[child]+child.children(recursive=True)
                    tracked.update({q.pid:q for q in tree})
                    rss=psutil.Process(os.getpid()).memory_info().rss+sum(q.memory_info().rss for q in tree if q.is_running())
                    peak=max(peak,rss)
                except psutil.NoSuchProcess:
                    rss=0
                if time.monotonic()-started>=CAPS["wall_seconds"] or rss>CAPS["rss_bytes"]:
                    killed="wall" if time.monotonic()-started>=CAPS["wall_seconds"] else "process_tree_RSS"
                    break
                time.sleep(.1)
    except BaseException as exc:
        failure=f"{type(exc).__name__}: {exc}"
    finally:
        # Always discover/stop descendants as well as the direct child. Keep
        # previously seen Process objects if their parent has already exited.
        if process is not None:
            try:
                child=psutil.Process(process.pid)
                tracked.update({q.pid:q for q in [child]+child.children(recursive=True)})
            except psutil.NoSuchProcess:
                pass
            except BaseException as exc:
                cleanup_errors.append(f"discover: {type(exc).__name__}: {exc}")
            for q in reversed(list(tracked.values())):
                try:
                    if q.is_running():
                        q.kill(); cleanup_killed.append(q.pid)
                except psutil.NoSuchProcess:
                    pass
                except BaseException as exc:
                    cleanup_errors.append(f"kill {q.pid}: {type(exc).__name__}: {exc}")
            try:
                if process.poll() is None:
                    process.kill()
                    if process.pid not in cleanup_killed:
                        cleanup_killed.append(process.pid)
                process.wait(timeout=10)
            except BaseException as exc:
                cleanup_errors.append(f"direct child wait: {type(exc).__name__}: {exc}")
            try:
                _,alive=psutil.wait_procs(list(tracked.values()),timeout=2)
                if alive:
                    cleanup_errors.append("surviving descendants: "+",".join(str(q.pid) for q in alive))
            except BaseException as exc:
                cleanup_errors.append(f"tree wait: {type(exc).__name__}: {exc}")
    elapsed=time.monotonic()-started
    if elapsed>=CAPS["wall_seconds"]:
        killed=killed or "wall_final"
    returncode=None if process is None else process.returncode
    status="PASS" if (returncode==0 and killed is None and failure is None and not cleanup_errors
        and not cleanup_killed and elapsed<CAPS["wall_seconds"] and (output/"receipt.json").is_file()) else "FAIL"
    supervisor=dict(status=status,returncode=returncode,killed=killed,error=failure,
        cleanup_killed_pids=cleanup_killed,cleanup_errors=cleanup_errors,
        peak_process_tree_rss=peak,wall_seconds=elapsed,poll_seconds=.1,
        note="Supervisor+child+descendant RSS sampled every100ms; final wall checked after cleanup",finished_at=now())
    try:
        dump(output/"supervisor.json",supervisor)
    except BaseException as exc:
        # Cleanup has already run. If the filesystem itself rejects the failure
        # receipt, preserve a machine-readable failure on the caller's stderr.
        supervisor.update(status="FAIL",receipt_write_error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(supervisor),file=sys.stderr,flush=True)
        raise RuntimeError("supervisor receipt write failed after cleanup") from exc
    if status!="PASS":
        raise RuntimeError("CPU pilot failed; preserve outputs and do not retry")
    print(json.dumps(dict(status=status,output=uri(output))))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze"); f.add_argument("--directory",required=True)
    r=sub.add_parser("run"); r.add_argument("--plan",required=True,type=Path); r.add_argument("--plan-sha256",required=True); r.add_argument("--execute-pixels",action="store_true")
    c=sub.add_parser("_child"); c.add_argument("--plan",required=True,type=Path); c.add_argument("--output",required=True,type=Path); c.add_argument("--plan-sha256",required=True); c.add_argument("--execute-pixels",action="store_true")
    args=p.parse_args()
    if args.command=="freeze":freeze(args.directory)
    elif args.command=="run":run(args.plan,args.plan_sha256,args.execute_pixels)
    else:child_run(args.plan,args.output,args.plan_sha256,args.execute_pixels)


if __name__=="__main__":
    main()
