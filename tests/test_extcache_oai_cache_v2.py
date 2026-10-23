"""CPU-only synthetic tests for tools/extcache/oai_cache_v2.py (H-B-89, v1.0.1)."""
from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from tools.extcache import common
from tools.extcache import oai_cache_v2 as v2
from tools import exp155_oai_data

GEOM = {"Sagittal": ([0, 1, 0, 0, 0, 1], lambda i: [i * 2, 0, 0]),
        "Coronal": ([1, 0, 0, 0, 0, 1], lambda i: [0, i * 2, 0]),
        "Axial": ([1, 0, 0, 0, 1, 0], lambda i: [0, 0, i * 2])}
VSHA = "f" * 64


def _dcm(sop, i, plane, base, value=None):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    ds = FileDataset("synthetic", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = sop
    ds.Rows = ds.Columns = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.5, 0.5]
    ds.InstanceNumber = i + 1
    iop, ipp = GEOM[plane]
    ds.ImageOrientationPatient = iop
    ds.ImagePositionPatient = ipp(i)
    px = (np.arange(256, dtype=np.uint16).reshape(16, 16) * (base % 7 + 1) + i * 11 + base * 3) % 3000
    ds.PixelData = (np.full((16, 16), value, np.uint16) if value is not None else px.astype(np.uint16)).tobytes()
    out = io.BytesIO()
    ds.save_as(out)
    return out.getvalue()


# study, archive id, plane, source_sequence, contrast_family, fluid, n frames, eligible, projection frame
ARCHIVES = [
    ("OAI_1_R_V00", "101", "Sagittal", "SAG_IW_TSE", "IW_TSE_FS", 1, 6, True, True),
    ("OAI_1_R_V00", "102", "Coronal", "COR_MPR", "DESS_WE", 1, 5, True, False),
    ("OAI_1_R_V00", "103", "Coronal", "COR_IW_TSE", "IW_TSE_NON_FS", 0, 5, True, False),
    ("OAI_1_R_V00", "104", "Axial", "AX_MPR", "DESS_WE", -1, 7, True, True),
    ("OAI_2_L_V00", "201", "Sagittal", "SAG_IW_TSE", "IW_TSE_FS", 1, 5, True, False),
    ("OAI_2_L_V00", "202", "Sagittal", "SAG_IW_TSE", "IW_TSE_FS", 1, 6, True, False),
    ("OAI_2_L_V00", "203", "Coronal", "COR_MPR", "DESS_WE", 1, 5, False, False),   # quarantined
    ("OAI_2_L_V00", "204", "Axial", "AX_MPR", "DESS_WE", -1, 5, True, False),
    ("OAI_3_R_V00", "301", "Axial", "AX_MPR", "DESS_WE", -1, 5, True, False),
    ("OAI_4_L_V00", "401", "Coronal", "COR_IW_TSE", "IW_TSE_NON_FS", 0, 5, False, False),  # no selection at all
]
SLOT_ORDER = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", -1)]
MARK = 4000  # projection-frame pixel value (other pixels are < 3000); must never reach GENERAL


def _sha(path):
    return common.sha256(path)


def make_audit(tmp: Path, select_override=None):
    raw, out = tmp / "raw", tmp / "audit"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    archives, routing, groups = [], [], {}
    for n, (study, aid, plane, seqn, fam, fluid, frames, eligible, proj) in enumerate(ARCHIVES):
        series_key = f"{study}_{aid}"
        path = raw / f"{aid}.tar.gz"
        members, refs, excluded = [], [], []
        if proj:
            sop = f"1.2.{aid}.0"
            members.append(("000", _dcm(sop, 0, "Coronal" if plane != "Coronal" else "Axial", n, MARK)))
            excluded.append(dict(member="000", frame_number=1, reason="projection_image", sop_instance_uid=sop))
        for i in (3, 0, 4, 1, 2, 5, 6)[:frames] if frames <= 7 else range(frames):
            sop = f"1.2.{aid}.{i + 1}"
            members.append((f"{i + 1:03d}", _dcm(sop, i, plane, n)))
            refs.append([f"{i + 1:03d}", 1, sop])
        refs.sort()
        with tarfile.open(path, "w:gz") as tar:
            for name, data in members:
                info = tarfile.TarInfo(f"./{name}" if n % 2 else name)  # real OAI tars use ./NNN
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        st = path.stat()
        stamp = dict(tool_version=v2.AUDIT_VERSION, verification_sha256=VSHA)
        archives.append(dict(series_key=series_key, study_key=study, relocated_archive=str(path),
                             raw_size=st.st_size, raw_mtime_ns=st.st_mtime_ns, status="READ_OK",
                             excluded_frames=excluded, **stamp))
        gkey = f"{series_key}::g{aid}"
        routing.append(dict(group_key=gkey, study_key=study, series_key=series_key, plane=plane,
                            proposed_fluid=fluid, source_sequence=seqn, contrast_family=fam,
                            contrast_certainty="HEADER_SUPPORTED_SOURCE_PROTOCOL", contrast_verified=True,
                            frame_count=len(refs), frame_refs=refs, route_eligible=eligible,
                            routing_issues=[] if eligible else ["geometry_vs_source_plane_conflict"],
                            issues=[] if eligible else ["missing_invalid_or_oblique_iop"],
                            relocated_archive=str(path), dicom_series_description=f"{seqn}_RIGHT", **stamp))
        groups[gkey] = routing[-1]
    sel = {"OAI_1_R_V00": {("Sagittal", 1): "OAI_1_R_V00_101::g101", ("Coronal", 1): "OAI_1_R_V00_102::g102",
                           ("Coronal", 0): "OAI_1_R_V00_103::g103", ("Axial", -1): "OAI_1_R_V00_104::g104"},
           "OAI_2_L_V00": {("Sagittal", 1): "OAI_2_L_V00_201::g201", ("Axial", -1): "OAI_2_L_V00_204::g204"},
           "OAI_3_R_V00": {("Axial", -1): "OAI_3_R_V00_301::g301"}, "OAI_4_L_V00": {}}
    for study, slot, g in (select_override or []):
        sel[study][slot] = g
    rows = []
    for contract in (v2.R64, v2.E143):
        for study in sorted(sel):
            for slot in SLOT_ORDER:
                g = sel[study].get(slot)
                rows.append(dict(contract=contract, study_key=study, plane=slot[0], fluid=slot[1],
                                 after_group_key=g, before_series_key=None,
                                 contrast_family=groups[g]["contrast_family"] if g else None,
                                 tool_version=v2.AUDIT_VERSION, verification_sha256=VSHA))
    for name, recs in (("archives.jsonl", archives), ("routing.jsonl", routing), ("slot-selections.jsonl", rows)):
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    summary = dict(tool_version=v2.AUDIT_VERSION, status="COMPLETE_HEADERS", execution_complete=True,
                   verification_sha256=VSHA, output_sha256={n: _sha(out / n) for n in
                                                            ("archives.jsonl", "routing.jsonl", "slot-selections.jsonl")})
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return out


def build_all(tmp: Path, audit_dir: Path, mode=v2.MODES[0], tag="a"):
    audit = v2.load_audit(audit_dir)
    gen_root = tmp / f"gen_{tag}"
    assert v2.build_general(audit, gen_root)["status"] == "COMPLETE"
    v2.build_r64(audit, gen_root, tmp / f"r64_{tag}", mode, shard_size=2)
    v2.build_e143(audit, gen_root, tmp / f"e143_{tag}", mode)
    return audit, gen_root, tmp / f"r64_{tag}", tmp / f"e143_{tag}"


def _slot_ranges(contract):
    out, off = {}, 0
    for plane, fluid, k in v2.contract_slots(contract):
        out[(plane, fluid)] = slice(off, off + k)
        off += k
    return out


def _e143(out, uid):
    ids = json.loads((out / "manifest.json").read_text())["ids"]
    i = ids.index(uid)
    return np.load(out / f"study_{i:05d}.npy"), json.loads((out / f"study_{i:05d}.json").read_text())


@pytest.fixture(scope="module", autouse=True)
def no_disk_reserve():
    with pytest.MonkeyPatch.context() as mp:  # synthetic outputs are tiny; the 100 GiB reserve is for real builds
        mp.setattr(common, "MIN_FREE", 0)
        yield


@pytest.fixture(scope="module")
def built(tmp_path_factory, no_disk_reserve):
    tmp = tmp_path_factory.mktemp("hb89")
    audit_dir = make_audit(tmp)
    return (tmp, audit_dir, *build_all(tmp, audit_dir))


def test_selection_exactly_follows_slot_selections(built):
    tmp, audit_dir, audit, gen_root, r64, e143 = built
    picks = {json.loads(l)["uid"]: json.loads(l) for l in (r64 / "picks.jsonl").read_text().splitlines()}
    for contract in (v2.R64, v2.E143):
        for uid in ("OAI_1_R_V00", "OAI_2_L_V00", "OAI_3_R_V00"):
            slots = picks[uid]["slots"] if contract == v2.R64 else _e143(e143, uid)[1]["slots"]
            for (plane, fluid, _), pick in zip(v2.contract_slots(contract), slots, strict=True):
                want = audit.selections[contract][uid][(plane, fluid)]
                if (plane, fluid) == v2.SAG0:
                    continue
                assert (pick or {}).get("group_key") == want
                assert pick is None or pick["fallback"] is None
    # A different selection changes the pick (selection is not re-derived by the tool).
    alt = make_audit(tmp / "alt", [("OAI_2_L_V00", ("Sagittal", 1), "OAI_2_L_V00_202::g202")])
    a2 = v2.load_audit(alt)
    plan = v2.plan_study(a2, v2.R64, "OAI_2_L_V00", "competition_then_copy")
    assert plan[0]["group_key"] == "OAI_2_L_V00_202::g202"
    assert plan[1]["group_key"] == "OAI_2_L_V00_201::g201" and plan[1]["fallback"] == v2.COMPETITION


def test_quarantined_and_projection_never_used(built, tmp_path, monkeypatch):
    tmp, audit_dir, audit, gen_root, r64, e143 = built
    index = v2.read_gen_index(gen_root)
    keys = {r["group_key"] for r in index}
    assert "OAI_2_L_V00_203::g203" not in keys and "OAI_4_L_V00_401::g401" not in keys
    for r in index:
        with np.load(v2.gen_dir(gen_root) / r["npz_file"]) as z:
            assert MARK not in z["vol"]
            assert z["vol"].shape[0] == int(r["frame_count"])
    for name in ("picks.jsonl",):
        text = (r64 / name).read_text()
        assert "g203" not in text and "g401" not in text
    assert json.loads((r64 / "manifest.json").read_text())["excluded_studies_no_selection"] == ["OAI_4_L_V00"]
    # The quarantined archives are never opened.
    opened = []
    real = tarfile.open
    monkeypatch.setattr(tarfile, "open", lambda p, *a, **k: (opened.append(Path(p).name), real(p, *a, **k))[1])
    v2.build_general(v2.load_audit(audit_dir), tmp_path / "g2")
    assert "203.tar.gz" not in opened and "401.tar.gz" not in opened and len(opened) == 8
    # A selection pointing at a quarantined group is refused.
    bad = make_audit(tmp_path / "bad", [("OAI_2_L_V00", ("Coronal", 1), "OAI_2_L_V00_203::g203")])
    with pytest.raises(ValueError, match="eligible"):
        v2.load_audit(bad)


def _render_alone(contract, gen_root, gen_rows, gkey, plane, fluid):
    row = dict(gen_rows[gkey], plane=plane, fluid_sensitive=str(fluid))
    if contract == v2.R64:
        return common.render_study([row], "oai", gen_root)[0]
    return exp155_oai_data.render_study([row], gen_root)[0]


def test_fallback_only_on_empty_sag0_and_flagged(built, tmp_path):
    tmp, audit_dir, audit, gen_root, r64, e143 = built
    gen_rows = {r["group_key"]: r for r in v2.read_gen_index(gen_root)}
    corpus = common.Corpus(r64)
    picks = {json.loads(l)["uid"]: json.loads(l) for l in (r64 / "picks.jsonl").read_text().splitlines()}
    assert picks["OAI_1_R_V00"]["sag0_fallback"] == v2.COPY            # single FS sagittal -> copy
    assert picks["OAI_2_L_V00"]["sag0_fallback"] == v2.COMPETITION     # unused 2nd sagittal (pick_series)
    assert picks["OAI_3_R_V00"]["sag0_fallback"] is None               # no sagittal at all -> stays empty
    for contract in (v2.R64, v2.E143):
        rng = _slot_ranges(contract)
        for uid, src in (("OAI_1_R_V00", "OAI_1_R_V00_101::g101"), ("OAI_2_L_V00", "OAI_2_L_V00_202::g202")):
            vol = corpus.volume(uid) if contract == v2.R64 else _e143(e143, uid)[0]
            alone = _render_alone(contract, gen_root, gen_rows, src, "Sagittal", 0)
            assert np.array_equal(vol[rng[v2.SAG0]], alone[rng[v2.SAG0]])
            slots = picks[uid]["slots"] if contract == v2.R64 else _e143(e143, uid)[1]["slots"]
            assert [s["fallback"] for s in slots if s] .count(None) == len([s for s in slots if s]) - 1
        vol3 = corpus.volume("OAI_3_R_V00") if contract == v2.R64 else _e143(e143, "OAI_3_R_V00")[0]
        assert not vol3[rng[v2.SAG0]].any()
    man = json.loads((e143 / "manifest.json").read_text())
    assert man["sag0_fallback_counts"] == {v2.COMPETITION: 1, v2.COPY: 1, "none": 1}
    # Other empty slots (OAI_2 coronal) are never filled.
    assert not corpus.volume("OAI_2_L_V00")[_slot_ranges(v2.R64)[("Coronal", 1)]].any()
    # Modes.
    plans = {m: v2.plan_study(audit, v2.R64, "OAI_1_R_V00", m)[1] for m in v2.MODES}
    assert plans["none"]["group_key"] is None and plans["competition_strict"]["group_key"] is None
    assert plans["copy"]["fallback"] == v2.COPY
    assert v2.plan_study(audit, v2.R64, "OAI_2_L_V00", "copy")[1]["group_key"] == "OAI_2_L_V00_201::g201"


def test_general_has_no_fallback(built):
    tmp, audit_dir, audit, gen_root, r64, e143 = built
    index = v2.read_gen_index(gen_root)
    eligible = sorted(g for g, r in audit.groups.items() if r["eligible"])
    assert sorted(r["group_key"] for r in index) == eligible
    assert all(r["series_key"] == r["group_key"] and "#" not in r["series_key"] for r in index)
    assert not [r for r in index if r["plane"] == "Sagittal" and r["fluid_sensitive"] == "0"]
    receipt = json.loads((v2.gen_dir(gen_root) / "receipt.json").read_text())
    assert receipt["fallback_rows"] == 0 and receipt["audit_summary_sha256"] == audit.summary_sha256


def test_other_slots_unchanged_vs_reference(built, tmp_path):
    """Reference = the v1 renderers' competition pick path (pick_series) on the same groups."""
    tmp, audit_dir, audit, gen_root, r64, e143 = built
    gen_rows = {r["group_key"]: r for r in v2.read_gen_index(gen_root)}
    corpus = common.Corpus(r64)
    uid = "OAI_1_R_V00"
    _, _, r, e = build_all(tmp_path, audit_dir, "none", tag="none")
    ref_rows = [dict(gen_rows[g], rule_hit=audit.groups[g]["source_sequence"])  # legacy, not "known contrast"
                for g in audit.selections[v2.R64][uid].values() if g]
    for contract in (v2.R64, v2.E143):
        if contract == v2.R64:
            ref, _, ref_picks = common.render_study(ref_rows, "oai", gen_root)
            new = corpus.volume(uid)
        else:
            ref, _, ref_picks = exp155_oai_data.render_study(ref_rows, gen_root)
            new = _e143(e143, uid)[0]
        assert ref_picks[1] is None  # legacy path: the only sagittal is used once -> Sag0 empty
        for slot, sl in _slot_ranges(contract).items():
            if slot != v2.SAG0:
                assert np.array_equal(ref[sl], new[sl]), slot
        # mode "none" equals the reference everywhere
        none_vol = common.Corpus(r).volume(uid) if contract == v2.R64 else _e143(e, uid)[0]
        assert np.array_equal(none_vol, ref)


def test_general_matches_v1_decoder_on_same_frames(built):
    """GENERAL v2 pixels/geometry = sources.dicom_series on the group's frames (v1 path)."""
    tmp, audit_dir, audit, gen_root, r64, e143 = built
    row = next(r for r in v2.read_gen_index(gen_root) if r["group_key"] == "OAI_1_R_V00_101::g101")
    ser = common.load_series(row, "oai", gen_root)
    with tarfile.open(audit.groups[row["group_key"]]["archive"], "r:gz") as tar:
        members = [(m.name, tar.extractfile(m).read()) for m in tar if m.name != "000"]
    ref = v2.sources.dicom_series(members, source="oai", study_key="x", series_key="x")
    assert np.array_equal(ser.vol, ref.vol) and np.allclose(ser.ipp, ref.ipp)
    assert np.allclose(ser.per_slice_spacing, ref.per_slice_spacing)
    assert ser.ipp[:, 0].tolist() == [0, 2, 4, 6, 8, 10]


def test_deterministic_bytes_and_binding(built, tmp_path):
    tmp, audit_dir, audit, gen_root, r64, e143 = built
    _, g2, r2, e2 = build_all(tmp_path, audit_dir, tag="b")
    for a, b in ((v2.gen_dir(gen_root), v2.gen_dir(g2)), (r64, r2), (e143, e2)):
        names = sorted(p.name for p in a.iterdir())
        assert names == sorted(p.name for p in b.iterdir())
        for n in names:
            assert _sha(a / n) == _sha(b / n), n
    for out in (r64, e143):
        man = json.loads((out / "manifest.json").read_text())
        reg = json.loads((out / "build-registration.json").read_text())
        assert man["audit_summary_sha256"] == _sha(audit_dir / "summary.json")
        assert man["tool_sha256"] == _sha(Path(v2.__file__))
        assert reg["code_sha256"]["tools/extcache/oai_cache_v2.py"] == man["tool_sha256"]
    assert v2.verify(audit, gen_root, r64, e143, sample=3)["errors"] == []
    from tools.exp143_v2 import check_corpus
    check_corpus(e143, full_hash=True)


def test_tampering_fails_closed(tmp_path):
    audit_dir = make_audit(tmp_path)
    with (audit_dir / "routing.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="differs from summary"):
        v2.load_audit(audit_dir)
    audit_dir = make_audit(tmp_path / "m")
    audit = v2.load_audit(audit_dir)
    path = Path(audit.groups["OAI_1_R_V00_102::g102"]["archive"])
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    res = v2.build_general(audit, tmp_path / "g")
    assert res["status"] == "INCOMPLETE" and res["failed"] == 1
    with pytest.raises(ValueError, match="differs"):  # changed audit/code -> new folder required
        v2._registration(v2.gen_dir(tmp_path / "g") / "registration.json", {"other": 1})


@pytest.mark.parametrize("builder", (v2.build_r64, v2.build_e143))
def test_production_refuses_general_with_stale_code(built, tmp_path, monkeypatch, builder):
    _, _, audit, gen_root, _, _ = built
    changed = dict(v2.code_hashes())
    changed["tools/extcache/oai_cache_v2.py"] = "0" * 64
    monkeypatch.setattr(v2, "code_hashes", lambda: changed)
    monkeypatch.setattr(v2, "_production_ids", lambda *args: pytest.fail("stale GENERAL was accepted"))
    with pytest.raises(ValueError, match="GENERAL v2 is unfinished or bound"):
        builder(audit, gen_root, tmp_path / "production")
    assert not (tmp_path / "production").exists()


def test_verify_hashes_every_recorded_production_file(built, tmp_path):
    _, _, audit, gen_root, r64, e143 = built
    rcopy, ecopy = tmp_path / "r64", tmp_path / "e143"
    shutil.copytree(r64, rcopy)
    shutil.copytree(e143, ecopy)
    rman = json.loads((rcopy / "manifest.json").read_text())
    eman = json.loads((ecopy / "manifest.json").read_text())
    names = [(r64, rcopy, s["file"]) for s in rman["shards"]]
    names += [(r64, rcopy, n) for n in ("ids.npy", "masks.npy", "picks.jsonl")]
    for file in eman["files"]:
        names.extend((e143, ecopy, f"study_{file['row']:05d}.{suffix}") for suffix in ("npy", "json"))
    for original, copied, name in names:
        path = copied / name
        with path.open("ab") as f:
            f.write(b"corrupt")
        try:
            errors = v2.verify(audit, gen_root, rcopy, ecopy, sample=0)["errors"]
            assert any(name in error for error in errors), (name, errors)
        finally:
            shutil.copyfile(original / name, path)


def test_populated_unregistered_destinations_are_refused(built, tmp_path, monkeypatch):
    _, _, audit, gen_root, _, _ = built
    monkeypatch.setattr(v2, "_registration", lambda *args: pytest.fail("registered an unsafe destination"))
    for builder in (v2.build_r64, v2.build_e143):
        out = tmp_path / builder.__name__
        out.mkdir()
        (out / "manifest.json").write_text("legacy", encoding="utf-8")
        with pytest.raises(ValueError, match="unregistered populated build destination"):
            builder(audit, gen_root, out)
        assert not (out / "build-registration.json").exists()
    legacy_gen = tmp_path / "legacy_gen"
    legacy_gen.mkdir()
    (legacy_gen / "manifest.json").write_text("legacy", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered populated build destination"):
        v2.build_general(audit, legacy_gen)
    assert not v2.gen_dir(legacy_gen).exists()
