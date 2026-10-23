"""H-B-89 OAI cache v2 builder (oai-cache-v2 v1.0.1). CPU only; no network, no GPU.

Rebuilds the OAI GENERAL cache and both OAI PRODUCTION caches as NEW versions whose
series selection comes ONLY from the verified H-B-88 header audit
(artifacts/experiments/HB88_OAI_HEADER_AUDIT_V2/run-v1: summary.json, routing.jsonl,
slot-selections.jsonl, archives.jsonl; AFTER routing).

Pixel/geometry/normalisation code paths are the existing ones, called unchanged:
  decode + slice order + spacing  tools.extcache.sources.dicom_series   (EXP135 GENERAL)
  native dtype                    tools.extcache.common._native_dtype
  RAPTOR64 336/c140/s02 render    tools.extcache.common.render_study    (EXT_OAI_RAPTOR64_336_C140)
  EXP143 96-slot 384 render       tools.exp155_oai_data.render_study    (EXP155_OAI_RAPTOR96_384_C140_S02)
What differs from the v1 builders (selection only):
  1. GENERAL unit = one audit group (archive frames in routing.jsonl frame_refs), not the whole
     archive: projection-image frames (archives.jsonl excluded_frames) are never decoded.
  2. Only route-eligible groups (route_eligible, no routing_issues/issues) enter GENERAL;
     quarantined groups are never opened.
  3. plane / fluid_sensitive / rule_hit come from routing.jsonl (not IOP + image03 description
     prefixes); rule_hit starts with "known contrast:" so both renderers use their strict
     per-slot pick (common.pick_strict) instead of the cross-contrast pick_series fallback.
  4. PRODUCTION rows per study are exactly the slot-selections after_group_key per slot; an empty
     selection stays empty, except the sagittal fluid-0 slot (OAI has no non-FS sagittal), which
     is filled by --sag0-fallback (default competition_then_copy: an unused route-eligible
     sagittal group as raptor_corpus.pick_series would take it, else the selected sagittal
     fluid-1 group re-rendered with the fluid-0 slot geometry) and flagged per study/slot.
  5. GENERAL .npz are written with fixed zip timestamps (np.savez_compressed embeds the wall
     clock) so GENERAL bytes are reproducible; array names/dtypes/layout are unchanged.
  6. Studies with no selected group at all are excluded from PRODUCTION and listed.
Every manifest binds the audit summary.json sha256, the audit output hashes, this file's
sha256 and the reused code hashes. Run: python -m tools.extcache.oai_cache_v2 --help

Changelog
  v1.0.1 - 2026-10-11 (Codex sol-2 for B, H-B-89): reject stale GENERAL code and
           unregistered populated destinations; verify every recorded production file hash.
  v1.0.0 - 2026-10-10 (Claude opus for B, H-B-89): initial GENERAL/RAPTOR64/EXP143 v2 builder,
           tests tests/test_extcache_oai_cache_v2.py, 20-study pilot under .local/hb89-pilot/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

from . import common, sources
from .common import INDEX_FIELDS, ROOT, sha256

VERSION = "oai-cache-v2-1.0.1"
AUDIT_VERSION = "oai-header-audit-v2.0.4"
SCHEMA_GEN = "hb89-oai-general-v2"
SCHEMA_R64 = "hb89-oai-raptor64-336-v2"
SCHEMA_E143 = "hb89-oai-exp143-96x384-v2"
AUDIT_DIR = ROOT / "artifacts/experiments/HB88_OAI_HEADER_AUDIT_V2/run-v1"
# GENERAL lives at <GEN_ROOT>/storage/cache/external/OAI/gen so the unchanged
# common.load_series/render_study and exp155_oai_data.render_study read it via their `root`.
GEN_ROOT = ROOT / "storage/cache/external/OAI_HB89_V2"
OUT_R64 = ROOT / "storage/cache/derived/EXT_OAI_RAPTOR64_336_C140_v2"
OUT_E143 = ROOT / "storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02_v2"
R64, E143 = "raptor64_336_c140_s02", "exp143_raptor96_384_c140_s02"
SAG0 = ("Sagittal", 0)
MODES = ("competition_then_copy", "copy", "competition_strict", "none")
COPY, COMPETITION = "copy_of_sagittal_fluid1", "competition_unused_sagittal"
CODE_FILES = ("tools/extcache/oai_cache_v2.py", "tools/extcache/common.py", "tools/extcache/sources.py",
              "tools/exp155_oai_data.py", "tools/exp143_v2.py", "training/src/exp143_v2.py",
              "training/src/raptor_corpus.py", "training/src/raptor_common.py")
EXTRA_FIELDS = ("group_key", "archive", "archive_size", "archive_mtime_ns", "frame_count",
                "source_sequence", "contrast_family", "contrast_certainty", "contrast_verified")
GEN_FIELDS = INDEX_FIELDS + EXTRA_FIELDS
STUDY_RE = re.compile(r"OAI_([0-9]+)_([RL])_(V00)")


def contract_slots(contract: str) -> list[tuple[str, int, int]]:
    if contract == R64:
        slots = common.CONTRACTS[R64]["slots"]
    elif contract == E143:
        from exp143_v2 import SPEC
        slots = SPEC["slots"]
    else:
        raise ValueError(contract)
    return [(str(p), int(f), int(k)) for p, f, k in slots]


def _jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _dump(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def _registration(path: Path, value: dict):
    """Create the registration, or require that an existing one is identical (resume)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(_dump(value)):
            raise ValueError(f"{path} differs (audit/code/selection changed): choose a new output folder")
    else:
        _atomic_text(path, _dump(value))
    return sha256(path)


def _build_destination(path: Path, registration: str):
    """Only an empty destination or one already registered by this builder is writable."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"build destination is not a directory: {path}")
        if any(path.iterdir()) and not (path / registration).is_file():
            raise ValueError(f"unregistered populated build destination: {path}")


def code_hashes(root: Path = ROOT) -> dict:
    return {p: sha256(Path(root) / p) for p in CODE_FILES}


def _require_space(path: Path, projected_bytes: int):
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < projected_bytes + common.MIN_FREE:
        raise RuntimeError(f"refusing write: {free / 2**30:.1f} GiB free < projected "
                           f"{projected_bytes / 2**30:.1f} GiB + 100 GiB reserve")


# ----------------------------------------------------------------------------- audit


@dataclass
class Audit:
    directory: Path
    summary_sha256: str
    output_sha256: dict
    verification_sha256: str
    groups: dict
    archives: dict
    selections: dict          # contract -> study -> {(plane, fluid): group_key | None}
    studies: list
    sagittal_by_study: dict = field(default_factory=dict)

    def binding(self) -> dict:
        return dict(audit_dir=self.directory.relative_to(ROOT).as_posix() if self.directory.is_relative_to(ROOT)
                    else self.directory.as_posix(), audit_version=AUDIT_VERSION,
                    summary_sha256=self.summary_sha256, output_sha256=self.output_sha256,
                    verification_sha256=self.verification_sha256)


def load_audit(audit_dir: Path = AUDIT_DIR) -> Audit:
    """Read the audit and fail closed on anything not bound to its summary.json."""
    audit_dir = Path(audit_dir).resolve()
    summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("tool_version") != AUDIT_VERSION or summary.get("status") != "COMPLETE_HEADERS"
            or summary.get("execution_complete") is not True or not summary.get("verification_sha256")):
        raise ValueError("audit summary is not a complete, verified v2.0.4 header audit")
    outputs = {}
    for name in ("archives.jsonl", "routing.jsonl", "slot-selections.jsonl"):
        digest = sha256(audit_dir / name)
        if digest != summary.get("output_sha256", {}).get(name):
            raise ValueError(f"audit output differs from summary.json: {name}")
        outputs[name] = digest
    vsha = summary["verification_sha256"]

    def stamped(rec, what):
        if rec.get("tool_version") != AUDIT_VERSION or rec.get("verification_sha256") != vsha:
            raise ValueError(f"{what} record not stamped by the summarised audit")
        return rec

    archives = {}
    for rec in _jsonl(audit_dir / "archives.jsonl"):
        stamped(rec, "archives")
        if rec["series_key"] in archives:
            raise ValueError("duplicate archive record")
        archives[rec["series_key"]] = dict(
            path=rec["relocated_archive"], size=int(rec["raw_size"]), mtime_ns=int(rec["raw_mtime_ns"]),
            status=rec["status"], excluded=sorted(str(f["member"]) for f in rec.get("excluded_frames", [])))
    groups, sagittal = {}, defaultdict(list)
    for rec in _jsonl(audit_dir / "routing.jsonl"):
        stamped(rec, "routing")
        key = rec["group_key"]
        if key in groups:
            raise ValueError("duplicate routing group")
        refs = rec["frame_refs"]
        if any(int(fr[1]) != 1 for fr in refs):
            raise ValueError("multi-frame group needs a dedicated decoder")
        members = [str(fr[0]) for fr in refs]
        if len(set(members)) != len(members) or int(rec["frame_count"]) != len(members):
            raise ValueError(f"inconsistent frame_refs: {key}")
        arch = archives.get(rec["series_key"])
        if (arch is None or arch["path"] != rec["relocated_archive"] or arch["status"] != "READ_OK"
                or set(members) & set(arch["excluded"])):
            raise ValueError(f"group/archive receipt mismatch: {key}")
        eligible = rec["route_eligible"] is True and not rec["routing_issues"] and not rec["issues"]
        groups[key] = dict(group_key=key, study_key=rec["study_key"], series_key=rec["series_key"],
                           plane=rec["plane"], fluid=int(rec["proposed_fluid"]),
                           source_sequence=rec["source_sequence"], contrast_family=rec["contrast_family"],
                           contrast_certainty=rec["contrast_certainty"],
                           contrast_verified=bool(rec["contrast_verified"]), frame_count=len(members),
                           members=members, sops=[str(fr[2]) for fr in refs], eligible=eligible,
                           archive=rec["relocated_archive"], description=rec["dicom_series_description"])
        if eligible and rec["plane"] == "Sagittal":
            sagittal[rec["study_key"]].append(key)
    slots = {c: contract_slots(c) for c in (R64, E143)}
    selections = {c: defaultdict(dict) for c in slots}
    for rec in _jsonl(audit_dir / "slot-selections.jsonl"):
        stamped(rec, "slot-selections")
        c, study, slot = rec["contract"], rec["study_key"], (rec["plane"], int(rec["fluid"]))
        if c not in selections or slot in selections[c][study]:
            raise ValueError(f"unknown contract or duplicate slot selection: {c} {study} {slot}")
        g = rec["after_group_key"]
        if g is not None:
            grp = groups.get(g)
            if (grp is None or not grp["eligible"] or grp["study_key"] != study or grp["plane"] != rec["plane"]
                    or grp["contrast_family"] != rec["contrast_family"]):
                raise ValueError(f"selection is not an eligible routed group of that study/plane: {g}")
        selections[c][study][slot] = g
    studies = sorted(selections[R64])
    for c, slot_list in slots.items():
        if sorted(selections[c]) != studies:
            raise ValueError("contracts cover different studies")
        want = sorted((p, f) for p, f, _ in slot_list)
        for study, chosen in selections[c].items():
            if not STUDY_RE.fullmatch(study) or sorted(chosen) != want:
                raise ValueError(f"slot set mismatch for {study} {c}")
            used = [g for g in chosen.values() if g]
            if len(used) != len(set(used)):
                raise ValueError(f"group selected twice in one study: {study}")
    return Audit(audit_dir, sha256(audit_dir / "summary.json"), outputs, vsha, groups, archives,
                 {c: dict(v) for c, v in selections.items()}, studies,
                 {k: sorted(v) for k, v in sagittal.items()})


# ----------------------------------------------------------------------------- GENERAL


def gen_dir(gen_root: Path = GEN_ROOT) -> Path:
    return common.gen_dir("oai", gen_root)


def _write_npz(path: Path, arrays: list[tuple[str, np.ndarray]]):
    """np.savez_compressed layout with fixed zip timestamps (byte-reproducible)."""
    tmp = path.with_name(path.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays:
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with zf.open(info, "w", force_zip64=True) as fid:
                np.lib.format.write_array(fid, np.asanyarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def npz_name(group_key: str) -> str:
    return hashlib.sha256(group_key.encode()).hexdigest()[:24] + ".npz"


def decode_group(grp: dict, arch: dict, out_dir: Path) -> dict:
    """Decode exactly the audit group's frames with the unchanged EXP135 decoder; write one NPZ."""
    import pydicom
    path = Path(arch["path"])
    st = path.stat()
    if st.st_size != arch["size"] or st.st_mtime_ns != arch["mtime_ns"]:
        raise ValueError("raw archive size/mtime differs from the audit receipt")
    wanted = dict(zip(grp["members"], grp["sops"]))
    found = {}
    with tarfile.open(path, "r|gz") as tar:
        for m in tar:
            name = PurePosixPath(m.name.replace("\\", "/")).as_posix()  # audit's member-name normalisation
            if m.isfile() and name in wanted:
                if name in found:
                    raise ValueError(f"duplicate archive member {name}")
                found[name] = sources._read_member(tar, m)
    if set(found) != set(wanted):
        raise ValueError("group frames missing from archive")
    members = []
    for name in grp["members"]:
        ds = pydicom.dcmread(io.BytesIO(found[name]), stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
        if str(ds.SOPInstanceUID) != wanted[name]:
            raise ValueError(f"SOPInstanceUID differs from audit frame_refs: {name}")
        members.append((name, found[name]))
    sid, side, visit = STUDY_RE.fullmatch(grp["study_key"]).groups()
    series = sources.dicom_series(members, source="oai", study_key=grp["study_key"], series_key=grp["group_key"],
                                  subject_id=sid, knee_side=side, visit=visit,
                                  description=grp["description"], archive=str(path))
    if series.meta["plane"] != grp["plane"] or len(series.vol) != grp["frame_count"]:
        raise ValueError("decoded plane/frame count differs from the audit group")
    vol = common._native_dtype(np.asarray(series.vol))
    ipp = np.asarray(series.ipp, np.float32)
    iop = np.asarray(series.iop, np.float32)
    ps = np.asarray(series.pixel_spacing, np.float32)
    pss = np.asarray(series.per_slice_spacing, np.float32)
    if (vol.ndim != 3 or ipp.shape != (vol.shape[0], 3) or iop.shape != (6,) or ps.shape != (2,)
            or pss.shape != (vol.shape[0],)):
        raise ValueError("invalid GENERAL geometry")
    name = npz_name(grp["group_key"])
    _write_npz(Path(out_dir) / name, [("vol", vol), ("ipp", ipp), ("iop", iop), ("pixel_spacing", ps),
                                      ("slice_spacing", np.float32(series.slice_spacing)),
                                      ("per_slice_spacing", pss)])
    meta = dict(series.meta, plane=grp["plane"], fluid_sensitive=grp["fluid"],
                rule_hit=f"known contrast: HB88 {grp['source_sequence']} {grp['contrast_family']}")
    row = {k: meta.get(k, "") for k in INDEX_FIELDS}
    row.update(source="oai", rows=vol.shape[1], cols=vol.shape[2], n_slices=vol.shape[0],
               pixel_spacing=json.dumps(ps.tolist()), slice_spacing=series.slice_spacing,
               dtype=str(vol.dtype), npz_sha256=sha256(Path(out_dir) / name), npz_file=name,
               group_key=grp["group_key"], archive=str(path), archive_size=arch["size"],
               archive_mtime_ns=arch["mtime_ns"], frame_count=grp["frame_count"],
               source_sequence=grp["source_sequence"], contrast_family=grp["contrast_family"],
               contrast_certainty=grp["contrast_certainty"], contrast_verified=grp["contrast_verified"])
    return {k: str(v) for k, v in row.items()}


def _decode_task(task):
    grp, arch, out_dir = task
    started = time.perf_counter()
    try:
        return grp["group_key"], decode_group(grp, arch, Path(out_dir)), None, time.perf_counter() - started
    except Exception as e:  # noqa: BLE001 - logged, never silently dropped
        return grp["group_key"], None, f"{type(e).__name__}: {e}", time.perf_counter() - started


def _write_index(d: Path, rows: list[dict]):
    rows = sorted(rows, key=lambda r: (r["study_key"], r["group_key"]))
    tmp = d / "index.csv.partial"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GEN_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, d / "index.csv")


def read_gen_index(gen_root: Path = GEN_ROOT) -> list[dict]:
    path = gen_dir(gen_root) / "index.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_general(audit: Audit, gen_root: Path = GEN_ROOT, studies=None, workers: int = 1) -> dict:
    d = gen_dir(gen_root)
    _build_destination(Path(gen_root), "storage/cache/external/OAI/gen/registration.json")
    _build_destination(d, "registration.json")
    want = set(studies) if studies else None
    targets = sorted(g for g, r in audit.groups.items()
                     if r["eligible"] and (want is None or r["study_key"] in want))
    if not targets:
        raise ValueError("no eligible audit groups for the requested studies")
    _require_space(d, int(70 * 2**30 * len(targets) / 9366))
    reg = dict(schema=SCHEMA_GEN, tool_version=VERSION, audit=audit.binding(), code_sha256=code_hashes(),
               studies=sorted(want) if want else "ALL", groups=len(targets))
    reg_sha = _registration(d / "registration.json", reg)
    done = {r["group_key"]: r for r in read_gen_index(gen_root)
            if r["group_key"] in targets and (d / r["npz_file"]).is_file()
            and sha256(d / r["npz_file"]) == r["npz_sha256"]}
    todo = [g for g in targets if g not in done]
    failures, seconds = {}, []
    tasks = [(audit.groups[g], audit.archives[audit.groups[g]["series_key"]], str(d)) for g in todo]
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        chunk = max(workers, 1) * 8
        for start in range(0, len(tasks), chunk):
            part = tasks[start:start + chunk]
            for key, row, err, sec in (pool.map(_decode_task, part) if pool else map(_decode_task, part)):
                seconds.append(sec)
                if err:
                    failures[key] = err
                else:
                    done[key] = row
            _write_index(d, list(done.values()))
    finally:
        if pool:
            pool.shutdown()
    _atomic_text(d / "failures.json", _dump(failures))
    if failures:
        return dict(status="INCOMPLETE", groups=len(done), failed=len(failures), output=str(d))
    _write_index(d, [done[g] for g in targets])
    receipt = dict(schema=SCHEMA_GEN, tool_version=VERSION, groups=len(targets),
                   index_sha256=sha256(d / "index.csv"), registration_sha256=reg_sha,
                   audit_summary_sha256=audit.summary_sha256, finished=True,
                   fallback_rows=0, raw_archives_deleted=False)
    _atomic_text(d / "receipt.json", _dump(receipt))
    return dict(status="COMPLETE", groups=len(targets), decoded_now=len(todo), output=str(d),
                decode_seconds_total=float(sum(seconds)),
                decode_seconds_per_group=float(np.mean(seconds)) if seconds else None)


def general_binding(audit: Audit, gen_root: Path) -> tuple[dict, dict]:
    """Verify a finished GENERAL is bound to this audit; return {group_key: row} and its hashes."""
    d = gen_dir(gen_root)
    receipt = json.loads((d / "receipt.json").read_text(encoding="utf-8"))
    reg = json.loads((d / "registration.json").read_text(encoding="utf-8"))
    if (not receipt.get("finished") or receipt["schema"] != SCHEMA_GEN
            or receipt["audit_summary_sha256"] != audit.summary_sha256
            or reg["audit"]["summary_sha256"] != audit.summary_sha256
            or reg.get("code_sha256") != code_hashes()
            or receipt["index_sha256"] != sha256(d / "index.csv")
            or receipt["registration_sha256"] != sha256(d / "registration.json")):
        raise ValueError("GENERAL v2 is unfinished or bound to a different audit")
    rows = {r["group_key"]: r for r in read_gen_index(gen_root)}
    for key, row in rows.items():
        grp = audit.groups.get(key)
        if grp is None or not grp["eligible"] or row["series_key"] != key or row["study_key"] != grp["study_key"]:
            raise ValueError(f"GENERAL row is not an eligible audit group: {key}")
    return rows, dict(index_sha256=receipt["index_sha256"], receipt_sha256=sha256(d / "receipt.json"),
                      registration_sha256=receipt["registration_sha256"], groups=len(rows))


# ----------------------------------------------------------------------------- PRODUCTION


def plan_study(audit: Audit, contract: str, study: str, mode: str) -> list[dict]:
    """Per slot: the audit-selected group, or for the empty sagittal fluid-0 slot the fallback."""
    if mode not in MODES:
        raise ValueError(mode)
    chosen = audit.selections[contract][study]
    used = {g for g in chosen.values() if g}
    plan = []
    for plane, fluid, count in contract_slots(contract):
        g, fb = chosen[(plane, fluid)], None
        if g is None and (plane, fluid) == SAG0 and mode != "none":
            if mode in ("competition_then_copy", "competition_strict"):
                # raptor_corpus.pick_series: the first unused series of the plane, any contrast.
                cands = [x for x in audit.sagittal_by_study.get(study, []) if x not in used]
                if cands:
                    g, fb = cands[0], COMPETITION
            if g is None and mode in ("competition_then_copy", "copy") and chosen[("Sagittal", 1)]:
                g, fb = chosen[("Sagittal", 1)], COPY
        plan.append(dict(plane=plane, fluid=fluid, count=count, group_key=g, fallback=fb))
    return plan


def plan_rows(plan: list[dict], gen_rows: dict) -> list[dict]:
    rows = []
    for p in plan:
        if p["group_key"] is None:
            continue
        base = gen_rows[p["group_key"]]
        row = dict(base, plane=p["plane"], fluid_sensitive=str(p["fluid"]))
        if p["fallback"] == COPY:
            row["series_key"] = base["series_key"] + "#sag0_fallback_copy"
        rows.append(row)
    if not all(str(r["rule_hit"]).startswith("known contrast") for r in rows):
        raise ValueError("strict per-slot pick needs known-contrast rows")
    return rows


def _check_picks(plan, rows, picks, groups):
    keys = iter(r["series_key"] for r in rows)
    for p, pick in zip(plan, picks, strict=True):
        if p["group_key"] is None:
            if pick is not None:
                raise RuntimeError("renderer filled a slot the audit left empty")
            continue
        if pick is None or pick["series"] != next(keys):
            raise RuntimeError("renderer pick differs from the audit selection")
        grp = groups[p["group_key"]]
        pick.update(group_key=p["group_key"], slot=[p["plane"], p["fluid"]], fallback=p["fallback"],
                    source_sequence=grp["source_sequence"], contrast_family=grp["contrast_family"])


def render(contract: str, plan: list[dict], gen_rows: dict, groups: dict, gen_root: Path):
    rows = plan_rows(plan, gen_rows)
    if contract == R64:
        vol, mask, picks = common.render_study(rows, "oai", gen_root)
    else:
        from tools.exp155_oai_data import render_study
        vol, mask, picks = render_study(rows, gen_root)
    _check_picks(plan, rows, picks, groups)
    return vol, mask, picks


def _sag0_flag(plan):
    return next((p["fallback"] for p in plan if (p["plane"], p["fluid"]) == SAG0), None)


def _production_ids(audit: Audit, studies, contract: str):
    ids = sorted(set(studies) if studies else audit.studies)
    excluded = [s for s in ids if not any(audit.selections[contract][s].values())]
    return [s for s in ids if s not in excluded], excluded


def _registration_value(audit, schema, contract, mode, ids, excluded, gen_hashes, extra=None):
    return dict(schema=schema, tool_version=VERSION, contract=contract, slots=contract_slots(contract),
                sag0_fallback_mode=mode, ids=ids, excluded_studies_no_selection=excluded,
                audit=audit.binding(), general=gen_hashes, code_sha256=code_hashes(), **(extra or {}))


def _r64_shard(task):
    shard_no, start, block, gen_root, out = task
    out, gen_root = Path(out), Path(gen_root)
    name = f"vols_shard{shard_no:04d}.npy"
    receipt_path = out / f"vols_shard{shard_no:04d}.receipt.json"
    if receipt_path.exists():
        rec = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (rec["start"] == start and [p["uid"] for p in rec["picks"]] == [u for u, *_ in block]
                and (out / name).is_file() and sha256(out / name) == rec["shard"]["sha256"]):
            return rec, 0.0
    started = time.perf_counter()
    tmp = out / (name + ".partial")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(block), 64, 336, 336))
    picks_out = []
    for j, (uid, plan, rows_by_group, groups) in enumerate(block):
        v, m, picks = render(R64, plan, rows_by_group, groups, gen_root)
        arr[j] = v
        picks_out.append(dict(uid=uid, shard=shard_no, row=j, slots=picks, mask=m.tolist(),
                              sag0_fallback=_sag0_flag(plan),
                              gen_sha256=[rows_by_group[p["group_key"]]["npz_sha256"] for p in plan if p["group_key"]]))
    arr.flush()
    del arr
    os.replace(tmp, out / name)
    shard = dict(shard=shard_no, start=start, stop=start + len(block), file=name, done=True,
                 bytes=(out / name).stat().st_size, sha256=sha256(out / name))
    rec = dict(start=start, shard=shard, picks=picks_out)
    _atomic_text(receipt_path, _dump(rec))
    return rec, time.perf_counter() - started


def _study_inputs(audit, contract, uid, mode, gen_rows):
    plan = plan_study(audit, contract, uid, mode)
    keys = {p["group_key"] for p in plan if p["group_key"]}
    missing = keys - set(gen_rows)
    if missing:
        raise ValueError(f"GENERAL v2 lacks selected groups for {uid}: {sorted(missing)}")
    return plan, {k: gen_rows[k] for k in keys}, {k: audit.groups[k] for k in keys}


def build_r64(audit: Audit, gen_root: Path = GEN_ROOT, out: Path = OUT_R64, mode: str = MODES[0],
              studies=None, shard_size: int = 16, workers: int = 1) -> dict:
    gen_rows, gen_hashes = general_binding(audit, gen_root)
    ids, excluded = _production_ids(audit, studies, R64)
    out = Path(out)
    _build_destination(out, "build-registration.json")
    _require_space(out, len(ids) * 64 * 336 * 336)
    reg = _registration_value(audit, SCHEMA_R64, R64, mode, ids, excluded, gen_hashes, dict(shard_size=shard_size))
    reg_sha = _registration(out / "build-registration.json", reg)
    tasks = []
    for shard_no, start in enumerate(range(0, len(ids), shard_size)):
        block = [(uid, *_study_inputs(audit, R64, uid, mode, gen_rows)) for uid in ids[start:start + shard_size]]
        tasks.append((shard_no, start, block, str(gen_root), str(out)))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = list(pool.map(_r64_shard, tasks) if pool else map(_r64_shard, tasks))
    finally:
        if pool:
            pool.shutdown()
    shards = [r["shard"] for r, _ in results]
    picks = [p for r, _ in results for p in r["picks"]]
    masks = np.asarray([p["mask"] for p in picks], np.uint8).reshape(len(ids), 64)
    np.save(out / "ids.npy", np.asarray(ids, dtype=f"<U{max(map(len, ids))}"))
    np.save(out / "masks.npy", masks)
    _atomic_text(out / "picks.jsonl", "".join(json.dumps(p, sort_keys=True) + "\n" for p in picks))
    spec = dict(common.CONTRACTS[R64])
    spec = dict(slots=spec["slots"], img=spec["img"], crop_mm=spec["crop_mm"], span=list(spec["span"]), depth=64,
                normalization="series percentile 2-98 over picked slices, clip, uint8",
                order="IOP/IPP normal projection, InstanceNumber fallback; stable by path",
                resize="cv2.INTER_AREA on [0,1] float",
                series_selection=f"HB88 audit run-v1 slot-selections (AFTER routing), strict per slot; "
                                 f"sagittal fluid-0 fallback: {mode}",
                mask="filled slots", spacing_unknown="full-field resize")
    flags = Counter(p["sag0_fallback"] or "none" for p in picks)
    manifest = dict(contract=R64, spec=spec, builder=VERSION, schema=SCHEMA_R64, studies=len(ids),
                    shard_size=shard_size, dtype="uint8", shape_total=[len(ids), 64, 336, 336], source="oai",
                    shards=shards, ids_sha256=sha256(out / "ids.npy"), masks_sha256=sha256(out / "masks.npy"),
                    picks_sha256=sha256(out / "picks.jsonl"), gen_index_sha256=gen_hashes["index_sha256"],
                    build_registration_sha256=reg_sha, audit_summary_sha256=audit.summary_sha256,
                    tool_sha256=sha256(Path(__file__)), sag0_fallback_mode=mode,
                    sag0_fallback_counts=dict(sorted(flags.items())), excluded_studies_no_selection=excluded,
                    finished=True)
    _atomic_text(out / "manifest.json", _dump(manifest))
    if len(common.Corpus(out)) != len(ids):
        raise RuntimeError("Corpus could not read the v2 production output")
    return dict(status="COMPLETE", studies=len(ids), excluded=len(excluded), output=str(out),
                sag0_fallback_counts=manifest["sag0_fallback_counts"],
                render_seconds_total=float(sum(s for _, s in results)))


def _e143_study(task):
    row, uid, plan, rows_by_group, groups, gen_root, out = task
    out = Path(out)
    npy, receipt = out / f"study_{row:05d}.npy", out / f"study_{row:05d}.json"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old.get("uid") == uid and npy.is_file() and sha256(npy) == old.get("file_sha256"):
            return dict(row=row, uid=uid, sha256=old["file_sha256"], receipt_sha256=sha256(receipt)), old["sag0_fallback"], 0.0
    started = time.perf_counter()
    volume, mask, picks = render(E143, plan, rows_by_group, groups, Path(gen_root))
    tmp = npy.with_name(npy.name + ".partial")
    with tmp.open("wb") as f:
        np.save(f, volume, allow_pickle=False)
    os.replace(tmp, npy)
    digest = sha256(npy)
    _atomic_text(receipt, _dump(dict(uid=uid, row=row, mask=mask.tolist(), positive_mask_slices=int(mask.sum()),
                                     slots=picks, file_sha256=digest, sag0_fallback=_sag0_flag(plan),
                                     gen_sha256=[rows_by_group[p["group_key"]]["npz_sha256"]
                                                 for p in plan if p["group_key"]])))
    return (dict(row=row, uid=uid, sha256=digest, receipt_sha256=sha256(receipt)), _sag0_flag(plan),
            time.perf_counter() - started)


def build_e143(audit: Audit, gen_root: Path = GEN_ROOT, out: Path = OUT_E143, mode: str = MODES[0],
               studies=None, workers: int = 1) -> dict:
    from exp143_v2 import CONTRACT, SPEC
    from tools.exp143_v2 import check_corpus
    if CONTRACT != E143:
        raise ValueError("EXP143 contract changed")
    gen_rows, gen_hashes = general_binding(audit, gen_root)
    ids, excluded = _production_ids(audit, studies, E143)
    out = Path(out)
    _build_destination(out, "build-registration.json")
    _require_space(out, len(ids) * SPEC["depth"] * SPEC["img"] ** 2)
    reg = _registration_value(audit, SCHEMA_E143, E143, mode, ids, excluded, gen_hashes, dict(spec=SPEC))
    reg_sha = _registration(out / "build-registration.json", reg)
    tasks = [(i, uid, *_study_inputs(audit, E143, uid, mode, gen_rows), str(gen_root), str(out))
             for i, uid in enumerate(ids)]
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = list(pool.map(_e143_study, tasks, chunksize=4) if pool else map(_e143_study, tasks))
    finally:
        if pool:
            pool.shutdown()
    flags = Counter(f or "none" for _, f, _ in results)
    manifest = dict(schema=SCHEMA_E143, contract=CONTRACT, spec=SPEC, ids=ids, files=[r for r, _, _ in results],
                    finished=True, build_registration_sha256=reg_sha, tool_version=VERSION,
                    tool_sha256=sha256(Path(__file__)), audit_summary_sha256=audit.summary_sha256,
                    gen_index_sha256=gen_hashes["index_sha256"], sag0_fallback_mode=mode,
                    sag0_fallback_counts=dict(sorted(flags.items())), excluded_studies_no_selection=excluded)
    _atomic_text(out / "manifest.json", _dump(manifest))
    check_corpus(out)
    return dict(status="COMPLETE", studies=len(ids), excluded=len(excluded), output=str(out),
                sag0_fallback_counts=manifest["sag0_fallback_counts"],
                render_seconds_total=float(sum(s for _, _, s in results)))


def verify(audit: Audit, gen_root: Path, out_r64: Path | None, out_e143: Path | None, sample: int = 0) -> dict:
    """Hash checks of every output plus a deterministic re-render of `sample` studies per cache."""
    errors = []
    gen_rows, _ = general_binding(audit, gen_root)
    for key, row in gen_rows.items():
        if sha256(gen_dir(gen_root) / row["npz_file"]) != row["npz_sha256"]:
            errors.append(f"GENERAL {key}")
    def check_file(path: Path, expected: str):
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"{contract} hash {path.name}")
    for contract, out in ((R64, out_r64), (E143, out_e143)):
        if out is None:
            continue
        out = Path(out)
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        reg = json.loads((out / "build-registration.json").read_text(encoding="utf-8"))
        if (man["audit_summary_sha256"] != audit.summary_sha256 or man["build_registration_sha256"] != sha256(out / "build-registration.json")
                or man["tool_sha256"] != sha256(Path(__file__)) or reg["code_sha256"] != code_hashes()):
            errors.append(f"{contract} binding")
        if contract == R64:
            for shard in man["shards"]:
                check_file(out / shard["file"], shard["sha256"])
            for name, field in (("ids.npy", "ids_sha256"), ("masks.npy", "masks_sha256"),
                                ("picks.jsonl", "picks_sha256")):
                check_file(out / name, man[field])
        else:
            for file in man["files"]:
                row = file["row"]
                check_file(out / f"study_{row:05d}.npy", file["sha256"])
                check_file(out / f"study_{row:05d}.json", file["receipt_sha256"])
        ids = reg["ids"]
        for uid in ids[:sample]:
            plan, rows, groups = _study_inputs(audit, contract, uid, reg["sag0_fallback_mode"], gen_rows)
            vol, _, _ = render(contract, plan, rows, groups, gen_root)
            stored = (common.Corpus(out).volume(uid) if contract == R64
                      else np.load(out / f"study_{ids.index(uid):05d}.npy"))
            if not np.array_equal(vol, stored):
                errors.append(f"{contract} re-render {uid}")
    return dict(errors=errors, general_groups=len(gen_rows))


# ----------------------------------------------------------------------------- pilot


def pilot_studies(audit: Audit, n: int = 20) -> list[str]:
    """Deterministic mix: old Sag0 from AX_MPR, old Sag1 from COR_MPR, two sagittal groups,
    old Cor1 from COR_IW_TSE, then unaffected studies."""
    seq = {g["series_key"]: g["source_sequence"] for g in audit.groups.values()}
    before = defaultdict(dict)
    for rec in _jsonl(audit.directory / "slot-selections.jsonl"):
        if rec["contract"] == R64:
            before[rec["study_key"]][(rec["plane"], int(rec["fluid"]))] = seq.get(rec["before_series_key"])
    cats = [lambda s: before[s][SAG0] == "AX_MPR", lambda s: before[s][("Sagittal", 1)] == "COR_MPR",
            lambda s: len(audit.sagittal_by_study.get(s, [])) == 2,
            lambda s: before[s][("Coronal", 1)] == "COR_IW_TSE",
            lambda s: not any(audit.selections[R64][s].values()), lambda s: True]
    quota = [6, 4, 3, 3, 1, n]
    chosen = []
    for cat, q in zip(cats, quota):
        for s in audit.studies:
            if len(chosen) >= n or q <= 0:
                break
            if s not in chosen and cat(s):
                chosen.append(s)
                q -= 1
    return sorted(chosen[:n])


def compare_old(audit: Audit, gen_root: Path, studies, mode, old_r64: Path, old_e143: Path, new_r64: Path,
                new_e143: Path) -> dict:
    """Aggregate per-slot comparison of v2 against the v1 caches (no per-study rows emitted)."""
    gen_rows, _ = general_binding(audit, gen_root)
    seq_of_archive = {g["series_key"]: g["source_sequence"] for g in audit.groups.values()}
    result = {}
    old_picks = {}
    want = set(studies)
    with (Path(old_r64) / "picks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if any(s in line[:80] for s in want):
                rec = json.loads(line)
                if rec["uid"] in want:
                    old_picks[rec["uid"]] = rec["slots"]
    old_r64c, new_r64c = common.Corpus(old_r64), common.Corpus(new_r64)
    old_e_man = json.loads((Path(old_e143) / "manifest.json").read_text(encoding="utf-8"))
    new_e_ids = json.loads((Path(new_e143) / "manifest.json").read_text(encoding="utf-8"))["ids"]
    for contract in (R64, E143):
        counts = Counter()
        for uid in studies:
            if contract == R64:
                old_slots, old_vol = old_picks[uid], old_r64c.volume(uid)
                new_vol = new_r64c.volume(uid) if uid in new_r64c else None
            else:
                i = old_e_man["ids"].index(uid)
                old_slots = json.loads((Path(old_e143) / f"study_{i:05d}.json").read_text(encoding="utf-8"))["slots"]
                old_vol = np.load(Path(old_e143) / f"study_{i:05d}.npy")
                new_vol = (np.load(Path(new_e143) / f"study_{new_e_ids.index(uid):05d}.npy")
                           if uid in new_e_ids else None)
            plan = plan_study(audit, contract, uid, mode)
            offset = 0
            for p, old in zip(plan, old_slots):
                sl = slice(offset, offset + p["count"])
                offset += p["count"]
                slot = f"{p['plane'][:3]}{p['fluid']}"
                old_key = old["series"] if old else None
                new_key = p["group_key"]
                if new_vol is None:
                    cat = "study_excluded_no_selection" + ("" if old_key else "_old_empty")
                elif new_key is None:
                    cat = "empty_both" if old_key is None else f"emptied(old={seq_of_archive.get(old_key)})"
                elif p["fallback"]:
                    cat = f"fallback:{p['fallback']}(old={seq_of_archive.get(old_key) if old_key else None})"
                elif old_key is None:
                    cat = f"newly_filled({audit.groups[new_key]['source_sequence']})"
                elif audit.groups[new_key]["series_key"] == old_key:
                    same = np.array_equal(old_vol[sl], new_vol[sl])
                    excl = bool(audit.archives[old_key]["excluded"])
                    cat = ("same_series_identical_pixels" if same else
                           "same_series_pixels_differ" + ("(projection_frames_excluded)" if excl else "(other)"))
                else:
                    cat = f"rerouted({seq_of_archive.get(old_key)}->{audit.groups[new_key]['source_sequence']})"
                counts[f"{slot}:{cat}"] += 1
        result[contract] = dict(sorted(counts.items()))
    return result


def run_pilot(audit: Audit, out: Path, n: int, mode: str, workers: int) -> dict:
    out = Path(out)
    studies = pilot_studies(audit, n)
    t0 = time.perf_counter()
    gen = build_general(audit, out / "gen_root", studies, workers)
    if gen["status"] != "COMPLETE":
        raise RuntimeError(f"pilot GENERAL incomplete: {gen}")
    t1 = time.perf_counter()
    r64 = build_r64(audit, out / "gen_root", out / "r64", mode, studies, 16, workers)
    t2 = time.perf_counter()
    e143 = build_e143(audit, out / "gen_root", out / "e143", mode, studies, workers)
    t3 = time.perf_counter()
    ver = verify(audit, out / "gen_root", out / "r64", out / "e143", sample=2)
    cmp_ = compare_old(audit, out / "gen_root", studies, mode, ROOT / "storage/cache/derived/EXT_OAI_RAPTOR64_336_C140",
                       ROOT / "storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02", out / "r64", out / "e143")
    gdir = gen_dir(out / "gen_root")
    gen_bytes = sum(p.stat().st_size for p in gdir.glob("*.npz"))
    report = dict(tool_version=VERSION, studies=len(studies), mode=mode, workers=workers, general=gen, r64=r64,
                  e143=e143, verify=ver, compare=cmp_, wall_seconds=dict(general=t1 - t0, r64=t2 - t1, e143=t3 - t2),
                  general_bytes=gen_bytes, general_groups=gen["groups"],
                  audit_summary_sha256=audit.summary_sha256, tool_sha256=sha256(Path(__file__)))
    _atomic_text(out / "pilot-report.json", _dump(report))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path, default=AUDIT_DIR)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("fingerprint")
    p = sub.add_parser("general")
    p.add_argument("--gen-root", type=Path, default=GEN_ROOT)
    p.add_argument("--workers", type=int, default=1)
    for name, default in (("r64", OUT_R64), ("e143", OUT_E143)):
        p = sub.add_parser(name)
        p.add_argument("--gen-root", type=Path, default=GEN_ROOT)
        p.add_argument("--out", type=Path, default=default)
        p.add_argument("--sag0-fallback", choices=MODES, default=MODES[0])
        p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("verify")
    p.add_argument("--gen-root", type=Path, default=GEN_ROOT)
    p.add_argument("--r64", type=Path, default=OUT_R64)
    p.add_argument("--e143", type=Path, default=OUT_E143)
    p.add_argument("--sample", type=int, default=8)
    p = sub.add_parser("pilot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--sag0-fallback", choices=MODES, default=MODES[0])
    p.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if getattr(a, "workers", 1) not in (1, 2, 3, 4):
        ap.error("workers must be 1..4")
    if a.command == "fingerprint":
        print(_dump(dict(tool_version=VERSION, code_sha256=code_hashes(),
                         audit_summary_sha256=sha256(a.audit / "summary.json"))))
        return
    audit = load_audit(a.audit)
    if a.command == "general":
        result = build_general(audit, a.gen_root, None, a.workers)
    elif a.command == "r64":
        result = build_r64(audit, a.gen_root, a.out, a.sag0_fallback, None, 16, a.workers)
    elif a.command == "e143":
        result = build_e143(audit, a.gen_root, a.out, a.sag0_fallback, None, a.workers)
    elif a.command == "verify":
        result = verify(audit, a.gen_root, a.r64, a.e143, a.sample)
    else:
        if not 1 <= a.n <= 20:
            ap.error("pilot is limited to 1..20 studies")
        result = run_pilot(audit, a.out, a.n, a.sag0_fallback, a.workers)
    result = {"tool_version": VERSION, **result}
    print(_dump(result))
    if result.get("status") == "INCOMPLETE" or result.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
