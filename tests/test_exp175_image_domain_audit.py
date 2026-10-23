"""Synthetic EXP175 tests: estimators, validity, sampling, metadata and local I/O."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from tools import exp175_image_domain_audit as a


def _verification_note(tmp_path):
    # Synthetic test receipt only, never a real-data execution approval.
    path = tmp_path / "synthetic-verification.json"
    path.write_text(json.dumps(dict(verdict="PASS", verified_tool_version=a.VERSION,
                                    source_hashes=a.source_hashes())), encoding="utf-8")
    return path


@pytest.mark.parametrize("actual", [-1, "-1", "", None, np.nan, np.inf, 2, .5, "unknown"])
@pytest.mark.parametrize("slot", [0, 1, -1])
def test_unknown_fluid_flags_are_unassessed_not_mismatches(actual, slot):
    assert a.fluid_slot_status(slot, actual) == dict(fluid_slot_mismatch=False, fluid_slot_check="UNASSESSED")


@pytest.mark.parametrize("actual", [0, 1, "0", "1"])
def test_known_fluid_flags_keep_match_mismatch_and_any_slot(actual):
    flag = int(actual)
    assert a.fluid_slot_status(flag, actual)["fluid_slot_check"] == "MATCH"
    assert a.fluid_slot_status(1 - flag, actual) == dict(fluid_slot_mismatch=True, fluid_slot_check="MISMATCH")
    assert a.fluid_slot_status(-1, actual)["fluid_slot_check"] == "ANY_CONTRAST_SLOT"


@pytest.mark.parametrize("unclipped_pixels", [0, 1, 63])
def test_saturated_two_level_images_do_not_license_noise_or_contrast(unclipped_pixels):
    x = np.zeros((128, 128), np.float32)
    x[16:112, 16:112] = 1
    x[16, 16:16 + unclipped_pixels] = .5
    fluid, fat = np.zeros_like(x, bool), np.zeros_like(x, bool)
    fluid[32:48, 32:48] = True
    fat[80:96, 80:96] = True
    result = a.image_metrics(x, fluid, fat)
    assert not result["valid_signal"]
    assert result["nonclipped_signal_support"] == unclipped_pixels
    assert result["clip0"] > 0 and result["clip255"] > 0
    assert np.isfinite(result["p50"]) and np.isfinite(result["foreground_fraction"])
    for name in ("noise_hh_flat", "snr_proxy", "cnr_proxy", "central_peripheral_ratio", "fluid_fat_ratio"):
        assert np.isnan(result[name]), name
    assert np.isnan(a.tissue_ratio(x, fluid, fat))


def test_contrast_regions_require_their_own_nonclipped_support():
    x = np.full((128, 128), .4, np.float32)
    yy, xx = np.indices(x.shape)
    rr = ((yy - 63.5) / 64) ** 2 + ((xx - 63.5) / 64) ** 2
    x[rr < .25] = 1
    fluid, fat = np.zeros_like(x, bool), np.zeros_like(x, bool)
    fluid[56:72, 56:72] = True
    fat[16:24, 16:24] = True
    result = a.image_metrics(x, fluid, fat)
    assert result["valid_signal"]
    assert np.isnan(result["central_peripheral_ratio"])
    assert np.isnan(result["fluid_fat_ratio"])
    x[fluid] = .8
    assert a.image_metrics(x, fluid, fat)["fluid_fat_ratio"] == pytest.approx(2)


def test_hh_noise_tracks_known_added_noise_without_claiming_physical_snr():
    rng = np.random.default_rng(175)
    white = rng.normal(size=(192, 192)).astype(np.float32)
    lo = a.image_metrics(.5 + .01 * white)
    hi = a.image_metrics(.5 + .03 * white)
    assert lo["noise_support"] >= 64
    assert .007 < lo["noise_hh_flat"] < .013
    assert 2.5 < hi["noise_hh_flat"] / lo["noise_hh_flat"] < 3.5
    assert lo["snr_proxy"] > hi["snr_proxy"]


def test_blur_reduces_laplacian_and_highpass_energy_on_fixed_texture():
    yy, xx = np.indices((128, 128))
    x = (.5 + .2 * np.sin(xx * 1.1) + .1 * np.sin(yy * .3)).astype(np.float32)
    sharp, blur = a.image_metrics(x), a.image_metrics(gaussian_filter(x, 1.5))
    assert sharp["laplacian_var"] > blur["laplacian_var"] * 5
    assert sharp["hf_energy_ratio"] > blur["hf_energy_ratio"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_skewness_matches_bias_corrected_binary_moments(dtype):
    x = np.full((8, 8), .25, dtype=dtype)
    x[-2:] = .75
    # A positive affine transform of Bernoulli(p=1/4): skew = 2/sqrt(3),
    # with the registered unbiased correction sqrt(n*(n-1))/(n-2).
    n = x.size
    expected = 2 / np.sqrt(3) * np.sqrt(n * (n - 1)) / (n - 2)
    assert a.image_metrics(x)["skewness"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("value", [0., .5, 1.])
def test_constant_images_have_no_valid_noise_or_contrast(value):
    r = a.image_metrics(np.full((48, 48), value, np.float32))
    assert not r["valid_signal"]
    assert np.isnan(r["noise_hh_flat"]) and np.isnan(r["snr_proxy"])
    assert np.isnan(r["skewness"])
    assert r["clip0"] == (value == 0)
    assert r["clip255"] == (value == 1)


def test_zero_padding_is_not_valid_air_noise():
    x = np.zeros((128, 128), np.float32)
    x[24:104, 24:104] = .5
    r = a.image_metrics(x)
    assert r["air_support"] > 64 and r["air_zero_fraction"] == 1
    assert r["air_raw_sd"] == 0 and np.isnan(r["air_sd"])


def test_border_connected_low_signal_keeps_air_statistics_descriptive():
    rng = np.random.default_rng(8)
    x = rng.uniform(.001, .04, (128, 128)).astype(np.float32)
    x[24:104, 24:104] = .5
    r = a.image_metrics(x)
    assert r["air_positive_fraction"] == 1
    assert r["air_sd"] > 0
    assert r["fs_consistency"] == "UNASSESSED_NO_VALIDATED_TISSUE_ROIS"
    assert np.isnan(r["fluid_fat_ratio"])


def test_tissue_ratio_requires_disjoint_validated_rois():
    x = np.full((48, 48), .2, np.float32)
    fluid, fat = np.zeros(x.shape, bool), np.zeros(x.shape, bool)
    fluid[4:10, 4:10] = True
    fat[30:36, 30:36] = True
    x[fluid] = .8
    assert a.tissue_ratio(x, fluid, fat) == pytest.approx(4)
    with pytest.raises(ValueError, match="disjoint"):
        a.tissue_ratio(x, fluid, fluid)
    assert np.isnan(a.tissue_ratio(x))


def test_volume_aggregation_preserves_invalid_slice_count():
    rng = np.random.default_rng(12)
    vol = np.stack([np.zeros((64, 64)), .5 + .02 * rng.normal(size=(64, 64))]).astype(np.float32)
    r = a.volume_metrics(vol)
    assert r["sampled_slices"] == 2 and r["valid_signal_slices"] == 1
    assert r["noise_hh_flat_valid_slices"] == 1
    assert np.isfinite(r["slice_difference"])


def test_sampling_is_order_invariant_and_never_duplicates():
    ids = [f"s{i}" for i in range(100)]
    first = a.stable_sample(ids, 8, 175, "MRNET")
    assert first == a.stable_sample(ids[::-1], 8, 175, "MRNET")
    assert first != a.stable_sample(ids, 8, 176, "MRNET")
    assert len(set(first)) == 8
    assert a.slice_indices([1, 3, 6, 8, 9]).tolist() == [1, 6, 9]
    assert a.slice_indices([5]).tolist() == [5]
    with pytest.raises(ValueError, match="duplicate"):
        a.stable_sample(["a", "a"], 2, 175, "x")


def test_spacing_unknown_never_becomes_placeholder_point5_mm():
    pick = {"median_spacing": .5, "spacing_unknown": "True", "files": 3, "slices": [0, 0, 1, 2]}
    r = a.geometry(pick, {"slice_spacing": "2"}, 4, 8)
    assert np.isnan(r["pixel_spacing_row"]) and np.isnan(r["physical_span_mm"])
    assert r["duplicate_fraction"] == .25 and r["index_span_fraction"] == 1
    r = a.geometry(dict(pick, spacing_unknown=False), {"pixel_spacing": "[0.4, 0.6]", "slice_spacing": "2"}, 4, 8)
    assert r["pixel_spacing_row"] == .4 and r["pixel_spacing_col"] == .6
    assert r["physical_span_mm"] == 4


@pytest.mark.parametrize("scalar", [.4, .5])
@pytest.mark.parametrize("flag", [None, False, "False"])
def test_legacy_rendering_scalar_never_certifies_native_axes(scalar, flag):
    # .4 could be the row of [.4, .6]; .5 could be measured OR a fallback.
    pick = {"median_spacing": scalar}
    if flag is not None:
        pick["spacing_unknown"] = flag
    r = a.geometry(pick, {}, 18, 18)
    assert np.isnan(r["pixel_spacing_row"]) and np.isnan(r["pixel_spacing_col"])
    assert r["spacing_unknown"] is True and r["pixel_spacing_source"] == "unavailable"
    assert r["legacy_rendering_spacing_scalar"] == scalar
    assert "legacy_rendering_spacing_scalar" not in a.METRICS


@pytest.mark.parametrize("spacing,row,col", [
    ("[0.4, 0.6]", .4, .6), (r"0.4\0.6", .4, .6), ([.4], .4, np.nan),
    ([.4, -1], .4, np.nan), ([None, .6], np.nan, .6),
    ("unreadable", np.nan, np.nan), ([], np.nan, np.nan),
])
def test_native_spacing_preserves_only_metadata_axes_and_independent_z(spacing, row, col):
    r = a.geometry({"median_spacing": .5, "files": 3, "slices": [0, 2]},
                   {"pixel_spacing": spacing, "slice_spacing": -2, "thickness": 1.5}, 2, 8)
    np.testing.assert_allclose([r["pixel_spacing_row"], r["pixel_spacing_col"]], [row, col], equal_nan=True)
    assert r["spacing_unknown"] == (not all(np.isfinite([row, col])))
    # Absent in-plane axes do not erase separately measured through-plane data.
    assert r["slice_spacing"] == 2 and r["physical_span_mm"] == 4 and r["thickness"] == 1.5
    assert r["legacy_rendering_spacing_scalar"] == .5


def test_explicit_unknown_metadata_suppresses_placeholder_axes_and_z():
    r = a.geometry({"median_spacing": .5, "files": 3, "slices": [0, 2]},
                   {"pixel_spacing": [.5, .5], "spacing_unknown": True,
                    "slice_spacing": 2, "thickness": 1.5}, 2, 8)
    assert r["spacing_unknown"] and r["spacing_declared_unknown"]
    assert all(np.isnan(r[k]) for k in ("pixel_spacing_row", "pixel_spacing_col", "slice_spacing",
                                      "thickness", "physical_span_mm"))


def test_shift_statistics_detect_shift_and_handle_degenerate_reference():
    x = np.linspace(0, 1, 1000)
    same, shifted = a.shift_statistics(x, x), a.shift_statistics(x, x + 3)
    assert same["ks"] == 0 and same["histogram_overlap"] == pytest.approx(1) and same["flag"] == "NO_FLAG"
    assert shifted["ks"] == 1 and shifted["outside_5_95"] == 1 and shifted["histogram_overlap"] == 0
    assert "MEDIAN_SHIFT" in shifted["flag"]
    degenerate = a.shift_statistics([1] * 30, [1] * 30)
    assert "ZERO_REFERENCE_IQR" in degenerate["flag"]
    assert degenerate["histogram_overlap"] == 1
    assert a.shift_statistics([], [2])["flag"] == "INSUFFICIENT"


@pytest.mark.parametrize("reference", [np.linspace(0, 1, 1000), np.ones(30)])
def test_histogram_overlap_keeps_reference_endpoints_out_of_exterior_tails(reference):
    for endpoint, direction in ((reference.min(), -np.inf), (reference.max(), np.inf)):
        outside = np.full(30, np.nextafter(endpoint, direction))
        assert a.shift_statistics(reference, outside)["histogram_overlap"] == 0
    assert a.shift_statistics(reference, reference)["histogram_overlap"] == 1
    at_maximum = a.shift_statistics(reference, np.full(30, reference.max()))
    assert at_maximum["histogram_overlap"] > 0


def test_paths_cannot_escape_allowed_root(tmp_path):
    assert a.contained(tmp_path, "safe.npz").parent == tmp_path
    with pytest.raises(ValueError, match="escapes"):
        a.contained(tmp_path, "../unsafe.npz")


def test_selected_series_join_not_first_series_or_fluid_equals_fs(tmp_path):
    picks = {"u": {"slots": [{"series": "selected", "files": 8}]}}
    series = {"selected": {"StudyInstanceUID": "u", "Fluid_Sensitive": "1", "Fat_Suppression": "0",
                           "Anatomical_Plane": "Sagittal"},
              "other": {"StudyInstanceUID": "u", "Fluid_Sensitive": "0", "Fat_Suppression": "1"}}
    headers = {"selected": {"vendor": "SIEMENS"}, "other": {"vendor": "GE"}}
    pick, meta = a.selected_metadata("COMPETITION", "u", 0, picks, {}, series, headers, tmp_path)
    assert meta["vendor"] == "SIEMENS" and meta["fat_suppression"] == "0" and meta["fluid_sensitive"] == "1"
    series["selected"]["StudyInstanceUID"] = "wrong"
    with pytest.raises(ValueError, match="another study"):
        a.selected_metadata("COMPETITION", "u", 0, picks, {}, series, headers, tmp_path)


def test_csv_header_adapter_discards_non_acquisition_fields(tmp_path):
    d = tmp_path / "storage/cache/dicom"
    d.mkdir(parents=True)
    (d / "headers.csv").write_text('SeriesInstanceUID,Manufacturer,Report,MagneticFieldStrength\ns,GE,never-used,3\n', encoding="utf-8")
    meta, hashes, errors = a.header_metadata(tmp_path)
    assert meta["s"]["vendor"] == "GE" and meta["s"]["field_strength"] == "3"
    assert "Report" not in meta["s"] and len(hashes) == 1 and not errors


def test_conflicting_scanner_metadata_is_sticky(tmp_path):
    d = tmp_path / "storage/cache/dicom"
    d.mkdir(parents=True)
    (d / "headers.csv").write_text('SeriesInstanceUID,Manufacturer\ns,GE\ns,SIEMENS\ns,GE\n', encoding="utf-8")
    meta, _, errors = a.header_metadata(tmp_path)
    assert meta["s"] == {"header_conflicted": True} and errors


def test_external_chosen_series_cannot_join_another_study(tmp_path):
    with pytest.raises(ValueError, match="another study"):
        a.selected_metadata("MRNET", "u", 0, {"u": {"slots": [{"series": "s"}]}},
                            {"s": {"study_key": "other"}}, {}, {}, tmp_path)


def test_expired_legacy_deadline_does_not_truncate_summarization(tmp_path):
    row = dict(source="COMPETITION", split="train", group="Axial", domain="production",
               vendor="", field_strength="", series_description="", fat_suppression="")
    row.update({key: 0. for key in a.METRICS})
    result, complete = a.summarize(pd.DataFrame([row]), tmp_path, time.monotonic() - 1)
    assert complete and result == []
    status = json.loads((tmp_path / "summary_status.json").read_text())
    assert status["complete"] is True
    assert status["distribution_rows"] > 0


def test_general_hash_required_and_native_scale_explicit(tmp_path):
    p = tmp_path / "series.npz"
    vol = np.arange(4 * 32 * 32, dtype=np.uint16).reshape(4, 32, 32)
    np.savez_compressed(p, vol=vol)
    meta = {"npz_file": p.name, "npz_sha256": a.sha256(p)}
    r = a.general_task(tmp_path, meta, {"source": "MRNET"})
    assert r["domain"] == "general" and r["native_scale"] == "sample-series-p02-p98"
    assert r["native_dtype"] == "uint16" and r["sampled_slices"] == 3
    with pytest.raises(ValueError, match="hash mismatch"):
        a.general_task(tmp_path, dict(meta, npz_sha256="wrong"), {})
    with pytest.raises(ValueError, match="256 MiB"):
        a.general_task(tmp_path, meta, {}, max_bytes=1)


def _tiny_corpus(root):
    directory = root / "storage/cache/derived/RAPTOR64_336_C140_S02"
    directory.mkdir(parents=True)
    ids = ["train0", "train1", "test0"]
    np.save(directory / "ids.npy", np.array(ids))
    mask = np.zeros((3, 64), np.uint8); mask[:, :18] = 1
    np.save(directory / "masks.npy", mask)
    vol = np.zeros((3, 64, 336, 336), np.uint8)
    for i in range(3):
        vol[i, :18, 30:306, 30:306] = np.tile(np.arange(276, dtype=np.uint16) % 180 + 40, (276, 1)).astype(np.uint8)
    np.save(directory / "vols.npy", vol)
    manifest = {"contract": a.CONTRACT, "spec": {"depth": 64, "img": 336,
                "slots": [["Sagittal", 1, 18], ["Sagittal", 0, 14], ["Coronal", 1, 12], ["Coronal", 0, 8], ["Axial", -1, 12]]},
                "shards": [{"start": 0, "stop": 3, "file": "vols.npy"}], "finished": True}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "picks.jsonl").write_text("\n".join(json.dumps({"uid": uid, "slots": [
        {"series": uid + "s", "files": 18, "slices": list(range(18)), "median_spacing": .5}, None, None, None, None]}) for uid in ids), encoding="utf-8")
    comp = root / "storage/competition"; comp.mkdir(parents=True)
    for split, selected in (("train", ids[:2]), ("test", ids[2:])):
        pd.DataFrame({"StudyInstanceUID": selected, "SeriesInstanceUID": [u + "s" for u in selected],
                      "Fluid_Sensitive": 1, "Fat_Suppression": 0, "Anatomical_Plane": "Sagittal"}).to_csv(comp / (split + "_series.csv"), index=False)
    return directory


def test_synthetic_end_to_end_missing_sources_test_isolation_and_outputs(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(a.time, "monotonic", lambda: clock[0])
    for name, seconds in (("production_task", 1), ("summarize", 2), ("montages", 3), ("report", 4)):
        original = getattr(a, name)
        def timed(*args, _original=original, _seconds=seconds, **kwargs):
            result = _original(*args, **kwargs)
            clock[0] += _seconds
            return result
        monkeypatch.setattr(a, name, timed)
    _tiny_corpus(tmp_path)
    base = tmp_path / a.BASE; base.mkdir(parents=True)
    (base / "PROTOCOL.md").write_text("Synthetic protocol; no real studies.", encoding="utf-8")
    out = base / "run"
    note = _verification_note(tmp_path)
    args = a.parser().parse_args(["--root", str(tmp_path), "--out", str(out), "--sample-general", "0",
                                  "--sample-competition", "3", "--max-seconds", "-1", "--workers", "1", "--skip-behavior", "--worker",
                                  "--verification-note", str(note)])
    assert a.run(args) == 0
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "COMPLETE_WITH_GAPS"
    assert m["verification"] == a.verification_provenance(note)
    assert m["metrics_sha256"] == a.sha256(out / "metrics.csv")
    assert set(m["phase_timings"]) == set(a.PhaseTimings.NAMES)
    for name, seconds in (("extraction", 3), ("summary", 2), ("montage", 3), ("input_validation", 0), ("report", 4)):
        assert m["phase_timings"][name] == dict(status="COMPLETE", elapsed_seconds=seconds, duration_complete=True)
    assert m["phase_timings"]["behavior"]["status"] == "SKIPPED_REQUEST"
    assert m["phase_timings"]["behavior"]["elapsed_seconds"] is None
    assert m["phase_timings"]["metrics_load"]["status"] == "SKIPPED_FRESH_EXTRACTION"
    assert m["elapsed_seconds"] == 12
    assert str(note) in (out / "REPORT.md").read_text(encoding="utf-8")
    assert any(r["source"] == "SKM_TEA" and r["status"] == "MISSING_PRODUCTION_CACHE" for r in m["inventory"])
    frame = pd.read_csv(out / "metrics.csv")
    assert set(frame.uid) == {"train0", "train1", "test0"}
    assert frame.loc[frame.uid == "test0", "split"].item() == "test"
    assert frame.fat_suppression.eq(0).all()
    assert frame.pixel_spacing_row.isna().all() and frame.pixel_spacing_col.isna().all()
    assert frame.spacing_unknown.all() and frame.legacy_rendering_spacing_scalar.eq(.5).all()
    assert not any("labels" in p or p.endswith("train.csv") for p in m["input_hashes"])
    assert (out / "REPORT.md").exists() and list((out / "montages").glob("*.png"))
    assert pd.read_csv(out / "coverage.csv").query("group == 'Axial'").filled_studies.eq(0).all()
    reused, provenance = a.load_previous_metrics(tmp_path, out / "metrics.csv")
    assert len(reused) == len(frame)
    assert provenance["manifest_sha256"] == a.sha256(out / "manifest.json")
    with pytest.raises(FileExistsError):
        a.run(args)


def test_corpus_refuses_object_ids_and_wrong_contract(tmp_path):
    directory = _tiny_corpus(tmp_path)
    m = json.loads((directory / "manifest.json").read_text())
    m["contract"] = "other"
    (directory / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="contract"):
        a.open_corpus(directory)
    np.save(directory / "ids.npy", np.array([{"not": "safe"}], dtype=object))
    with pytest.raises(ValueError):
        a.open_corpus(directory)


def test_cli_refuses_existing_output_without_writing_failure(tmp_path):
    out = tmp_path / a.BASE / "prior"
    out.mkdir(parents=True)
    marker = out / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh"):
        a.main(["--root", str(tmp_path), "--out", str(out), "--worker",
                "--verification-note", str(_verification_note(tmp_path))])
    assert marker.read_text() == "unchanged"
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def _prior_metrics(root):
    base = root / a.BASE
    prior = base / "prior"
    prior.mkdir(parents=True)
    (base / "PROTOCOL.md").write_text("Synthetic protocol", encoding="utf-8")
    path = prior / "metrics.csv"
    pd.DataFrame([dict(source="KNEEMRI", split="external", uid="s", group="Sagittal_fluid1",
                       domain="production", noise_hh_flat=.01)]).to_csv(path, index=False)
    note = _verification_note(root)
    manifest = dict(version=a.VERSION, status="COMPLETE_WITH_GAPS", metric_rows=1,
                    metrics_sha256=a.sha256(path), verification=a.verification_provenance(note),
                    protocol_sha256=a.sha256(base / "PROTOCOL.md"), source_hashes=a.source_hashes(),
                    input_hashes={"storage/cache/derived/EXT_KNEEMRI_RAPTOR64_336_C140/manifest.json": "a" * 64})
    a.dump(prior / "manifest.json", manifest)
    return path, manifest, note


def test_reuse_keeps_original_cache_even_if_cache_was_rebuilt_or_removed(tmp_path):
    path, original, _ = _prior_metrics(tmp_path)
    cache = tmp_path / next(iter(original["input_hashes"]))
    cache.parent.mkdir(parents=True)
    cache.write_text("Rebuilt cache", encoding="utf-8")
    for present in (True, False):
        if not present:
            cache.unlink()
        frame, provenance = a.load_previous_metrics(tmp_path, path)
        assert len(frame) == 1
        assert provenance["input_hashes"] == original["input_hashes"]
        assert provenance["sha256"] == a.sha256(path)


@pytest.mark.parametrize("defect", ["absent_manifest", "absent_metric_hash", "changed_metrics", "failed", "running",
                                   "failure_file", "missing_cache", "bad_cache_hash", "missing_verification",
                                   "changed_note", "row_count"])
def test_behavior_only_refuses_unbound_or_invalid_original_metrics(tmp_path, defect):
    path, manifest, note = _prior_metrics(tmp_path)
    if defect == "absent_manifest":
        (path.parent / "manifest.json").unlink()
    elif defect == "absent_metric_hash":
        manifest.pop("metrics_sha256")
    elif defect == "changed_metrics":
        path.write_text(path.read_text() + "\n", encoding="utf-8")
    elif defect in ("failed", "running"):
        manifest["status"] = defect.upper()
    elif defect == "failure_file":
        (path.parent / "FAILURE.json").write_text("{}", encoding="utf-8")
    elif defect == "missing_cache":
        manifest["input_hashes"] = {"unrelated": "a" * 64}
    elif defect == "bad_cache_hash":
        manifest["input_hashes"] = {next(iter(manifest["input_hashes"])): "bad"}
    elif defect == "missing_verification":
        manifest.pop("verification")
    elif defect == "changed_note":
        note.write_text(note.read_text() + "\n", encoding="utf-8")
    elif defect == "row_count":
        manifest["metric_rows"] = 2
    if defect != "absent_manifest":
        a.dump(path.parent / "manifest.json", manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        a.load_previous_metrics(tmp_path, path)


def test_behavior_only_passes_bound_provenance_to_behavior_and_records_it(tmp_path, monkeypatch):
    from tools import exp175_behavior as b
    clock = [100.]
    monkeypatch.setattr(a.time, "monotonic", lambda: clock[0])
    path, original, note = _prior_metrics(tmp_path)
    out = tmp_path / a.BASE / "behavior"
    called = []
    def capture(root, frame, output, **kwargs):
        called.append(kwargs)
        assert kwargs["metrics_provenance"]["input_hashes"] == original["input_hashes"]
        assert kwargs["verification"] == a.verification_provenance(note)
        clock[0] += 5
        return {"status": "COMPLETE"}
    monkeypatch.setattr(b, "run_behavior", capture)
    args = a.parser().parse_args(["--root", str(tmp_path), "--out", str(out), "--worker",
                                  "--behavior-only", str(path), "--verification-note", str(note)])
    assert a.run(args) == 0
    result = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(called) == 1
    assert result["metrics_provenance"]["manifest_sha256"] == a.sha256(path.parent / "manifest.json")
    assert result["verification"]["sha256"] == a.sha256(note)
    assert result["phase_timings"]["behavior"] == dict(status="COMPLETE", elapsed_seconds=5, duration_complete=True)
    assert result["phase_timings"]["metrics_load"]["status"] == "COMPLETE"
    assert result["phase_timings"]["report"]["status"] == "COMPLETE"
    for name in ("extraction", "summary", "montage", "input_validation"):
        assert result["phase_timings"][name]["status"] == "SKIPPED_REUSED_METRICS"
        assert result["phase_timings"][name]["elapsed_seconds"] is None
    assert result["elapsed_seconds"] == 5


def test_failed_behavior_keeps_phase_duration_and_unstarted_report(tmp_path, monkeypatch):
    from tools import exp175_behavior as b
    clock = [100.]
    monkeypatch.setattr(a.time, "monotonic", lambda: clock[0])
    path, _, note = _prior_metrics(tmp_path)
    out = tmp_path / a.BASE / "failure"
    def fail(*args, **kwargs):
        clock[0] += 7
        raise ValueError("synthetic behavior failure")
    monkeypatch.setattr(b, "run_behavior", fail)
    args = a.parser().parse_args(["--root", str(tmp_path), "--out", str(out), "--worker",
                                  "--behavior-only", str(path), "--verification-note", str(note)])
    with pytest.raises(ValueError, match="synthetic behavior failure"):
        a.run(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "FAILED" and manifest["elapsed_seconds"] == 7
    assert manifest["phase_timings"]["behavior"] == dict(status="FAILED", elapsed_seconds=7, duration_complete=True)
    assert manifest["phase_timings"]["report"]["status"] == "NOT_STARTED"


@pytest.mark.parametrize("child_code", [0, 7])
def test_parent_waits_without_timeout_and_propagates_child_exit(tmp_path, monkeypatch, child_code):
    out = tmp_path / a.BASE / "untimed"
    note = _verification_note(tmp_path)
    calls = []
    def child(command, **kwargs):
        assert "timeout" not in kwargs and kwargs["check"] is False
        assert command[-1] == "--worker"
        calls.append(command)
        return type("Completed", (), {"returncode": child_code})()
    monkeypatch.setattr(a.subprocess, "run", child)
    assert a.main(["--root", str(tmp_path), "--out", str(out), "--max-seconds", "-1",
                   "--verification-note", str(note)]) == child_code
    assert len(calls) == 1 and not out.exists()


def test_no_images_records_report_and_skips_unrunnable_phases(tmp_path):
    base = tmp_path / a.BASE
    base.mkdir(parents=True)
    (base / "PROTOCOL.md").write_text("Synthetic no-images protocol", encoding="utf-8")
    out = base / "empty"
    args = a.parser().parse_args(["--root", str(tmp_path), "--out", str(out), "--worker",
                                  "--verification-note", str(_verification_note(tmp_path))])
    assert a.run(args) == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "NO_VALID_IMAGES"
    for name in ("extraction", "report"):
        assert manifest["phase_timings"][name]["status"] == "COMPLETE"
        assert manifest["phase_timings"][name]["elapsed_seconds"] >= 0
        assert manifest["phase_timings"][name]["duration_complete"]
    for name in ("summary", "montage", "input_validation", "behavior"):
        assert manifest["phase_timings"][name]["status"] == "SKIPPED_NO_VALID_IMAGES"
        assert manifest["phase_timings"][name]["elapsed_seconds"] is None


def test_long_behavior_keeps_measured_duration_and_genuine_gap_status(tmp_path, monkeypatch):
    from tools import exp175_behavior as b
    clock = [100.]
    monkeypatch.setattr(a.time, "monotonic", lambda: clock[0])
    path, _, note = _prior_metrics(tmp_path)
    out = tmp_path / a.BASE / "partial"
    def partial(*args, **kwargs):
        assert kwargs["deadline"] is None
        clock[0] += 100000
        return {"status": "COMPLETE_WITH_GAPS", "missing": ["Unavailable numeric predictions"]}
    monkeypatch.setattr(b, "run_behavior", partial)
    args = a.parser().parse_args(["--root", str(tmp_path), "--out", str(out), "--worker",
                                  "--max-seconds", "60", "--behavior-only", str(path),
                                  "--verification-note", str(note)])
    assert a.run(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE_WITH_GAPS"
    assert manifest["phase_timings"]["behavior"] == dict(
        status="COMPLETE_WITH_GAPS", elapsed_seconds=100000, duration_complete=True)
    assert manifest["phase_timings"]["report"]["status"] == "COMPLETE"


def test_cli_requires_verification_note():
    with pytest.raises(SystemExit) as exc:
        a.parser().parse_args([])
    assert exc.value.code == 2


@pytest.mark.parametrize("change", ["verdict", "version", "source", "missing_source"])
def test_verification_must_pass_exact_version_and_sources(tmp_path, change):
    note = _verification_note(tmp_path)
    receipt = json.loads(note.read_text())
    if change == "verdict":
        receipt["verdict"] = "HOLD"
    elif change == "version":
        receipt["verified_tool_version"] = "1.0.1"
    elif change == "source":
        receipt["source_hashes"]["exp175_behavior.py"] = "0" * 64
    else:
        receipt["source_hashes"].pop("exp175_behavior.py")
    note.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="exact tool version"):
        a.verification_provenance(note)
