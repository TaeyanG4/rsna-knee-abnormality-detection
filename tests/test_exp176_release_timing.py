"""EXP176 CPU fakes: no torch/CUDA import, dataset/label reads or network calls."""
import csv
import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from tools import exp176_release_timing as bench
from tools import release_cohort_check as cohort


def rows(n=100):
    return [{cohort.UID: str(i), "SeriesInstanceUID": str(s), "Fluid_Sensitive": "1",
             "Fat_Suppression": "1", "Anatomical_Plane": "Sagittal"}
            for i in range(n) for s in range(1+i%11)]


def series_file(tmp_path, n=100):
    path = tmp_path/"train_series.csv"
    bench.write_csv(path, bench.SERIES_FIELDS, rows(n))
    return path


def intended(tmp_path, n=48):
    path = series_file(tmp_path)
    return bench.choose_cohort(image_inventory(rows(), bench.sha(path)), n)


def image_inventory(metadata=None, series_sha="a"*64):
    metadata = rows() if metadata is None else metadata
    return {"schema": bench.INVENTORY_VERSION, "train_series_sha256": series_sha,
            "series": [bench.filename_record(r[cohort.UID], r["SeriesInstanceUID"],
                       [f"{j}.dcm" for j in range(1+(int(r[cohort.UID])+int(r["SeriesInstanceUID"]))%3)])
                       for r in metadata]}


def make_images(root, metadata=None):
    value = image_inventory(metadata)
    for r in value["series"]:
        directory = root/r[cohort.UID]/r["SeriesInstanceUID"]
        directory.mkdir(parents=True)
        for j in range(r["image_count"]):
            (directory/f"{j}.dcm").write_bytes(b"NOT A DICOM: filename-only tests")
    return root


def test_cohort_hash_requires_exact_count_membership_order_and_unique_ids():
    frozen = cohort.identity(["a", "b"])
    assert cohort.check(["a", "b"], frozen)["full_hidden_qualification"] is False
    for ids in (["a"], ["b", "a"], ["a", "c"], ["a", "a"], ["../a", "b"]):
        with pytest.raises(ValueError):
            cohort.check(ids, frozen)
    assert frozen["ordered_sha256"] != cohort.identity(["b", "a"])["ordered_sha256"]
    assert frozen["membership_sha256"] == cohort.identity(["b", "a"])["membership_sha256"]


def test_filename_selection_is_deterministic_tail_covering_and_honest():
    value = image_inventory()
    a = bench.choose_cohort(value, 48)
    b = bench.choose_cohort(image_inventory(list(reversed(rows()))), 48)
    assert a == b and a["studies"] == 48
    assert a["selected_series_count_min_max"] == [1, 11]
    assert a["image_count_coverage"]["status"] == "PASS_FILENAME_IMAGE_COUNTS_AND_LONG_SERIES"
    assert a["inventory_sha256"] == bench.digest_json(bench.validate_inventory(value))
    with pytest.raises(ValueError, match=">=40"):
        bench.choose_cohort(value, 39)


@pytest.mark.parametrize("n", [40, 41, 48, 99, 100])
def test_independent_image_total_and_longest_series_tails_with_identical_series_counts(n):
    # Every study has two series: the old series-count sampler has no information.
    # Highest totals and longest single series belong to disjoint sets of studies.
    records = []
    for i in range(100):
        lengths = (40, 40) if i < 20 else ((70, 1) if i >= 80 else (i//4+1, 1))
        for sid, length in enumerate(lengths):
            records.append(bench.filename_record(str(i), str(sid), [f"{j}.dcm" for j in range(length)]))
    value = {"schema": bench.INVENTORY_VERSION, "train_series_sha256": "b"*64, "series": records}
    chosen = bench.choose_cohort(value, n)
    assert len(chosen["ids"]) == n
    proof = chosen["image_count_coverage"]
    assert proof["ranges"]["total_images"]["selected"] == proof["ranges"]["total_images"]["population"]
    assert proof["ranges"]["max_series_images"]["selected"] == proof["ranges"]["max_series_images"]["population"]
    for groups in proof["requirements"].values():
        assert len(groups["tail_ids"]) == n//4
        assert set(groups["tail_ids"]+groups["spread_ids"]) <= set(chosen["ids"])
    if n == 48:
        assert set(proof["requirements"]["total_images"]["tail_ids"]).isdisjoint(
            proof["requirements"]["max_series_images"]["tail_ids"])
        bad = list(chosen["ids"])
        bad.remove(proof["requirements"]["max_series_images"]["tail_ids"][0])
        bad.append(next(str(i) for i in range(20, 80) if str(i) not in bad))
        with pytest.raises(ValueError, match="Missing image coverage"):
            bench.coverage_check(bad, bench.study_sizes(value))


def test_one_series_long_tail_is_selected_despite_many_short_series_elsewhere():
    value = image_inventory()
    value["series"] = [r for r in value["series"] if r[cohort.UID] != "44"]
    value["series"].append(bench.filename_record("44", "0", [f"{i}.dcm" for i in range(1000)]))
    result = bench.choose_cohort(value)
    assert "44" in result["ids"]
    assert result["study_sizes"]["44"] == {"series_count": 1, "total_images": 1000, "max_series_images": 1000}


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "1", None])
def test_inventory_rejects_invalid_counts(count):
    value = image_inventory()
    value["series"][0]["image_count"] = count
    with pytest.raises(ValueError, match="image count"):
        bench.validate_inventory(value, rows())


@pytest.mark.parametrize("mutation,match", [
    ("missing", "membership"), ("extra", "membership"), ("duplicate", "Duplicate"),
    ("hash", "filename SHA256"), ("unsafe", "safe study IDs"), ("drive", "Unsafe"),
    ("version", "version"), ("extra_field", "schema")])
def test_inventory_rejects_incomplete_ambiguous_or_malformed_evidence(mutation, match):
    value = image_inventory()
    if mutation == "missing": value["series"].pop()
    elif mutation == "extra": value["series"].append(bench.filename_record("extra", "0", ["x.dcm"]))
    elif mutation == "duplicate": value["series"].append(value["series"][0])
    elif mutation == "hash": value["series"][0]["filenames_sha256"] = "invalid"
    elif mutation == "unsafe": value["series"][0][cohort.UID] = "../escape"
    elif mutation == "drive": value["series"][0][cohort.UID] = "C:escape"
    elif mutation == "version": value["schema"] = "unregistered"
    else: value["series"][0]["total_images"] = 0
    with pytest.raises(ValueError, match=match):
        bench.validate_inventory(value, rows())


def test_filename_directory_and_zip_inventory_match_without_opening_image_bytes(tmp_path, monkeypatch):
    metadata = rows(5)
    root = make_images(tmp_path/"train_images", metadata)
    archive = tmp_path/"images.zip"
    files = sorted(root.rglob("*.dcm"))
    with zipfile.ZipFile(archive, "x") as z:
        for p in reversed(files):
            z.writestr("train_series/"+p.relative_to(root).as_posix(), b"NOT A DICOM")
        z.writestr("train.csv", "Do not open this entry")
        z.writestr("test_series/a/b/c.dcm", "not training")
    def forbidden(*a, **k):
        raise AssertionError("Image/archive member bytes must never be read")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    assert bench.scan_inventory(metadata, root, "a"*64) == bench.scan_inventory(metadata, archive, "a"*64)
    assert bench.scan_inventory(metadata, root, "a"*64) == bench.validate_inventory(image_inventory(metadata))


@pytest.mark.parametrize("mutation,match", [
    ("missing", "membership"), ("duplicate", "duplicate"), ("extra", "membership"),
    ("unsafe", "safe study IDs"), ("roots", "unambiguous"), ("symlink", "Symlink")])
def test_zip_inventory_fails_closed(tmp_path, mutation, match):
    archive = tmp_path/"images.zip"
    names = ["train_series/0/0/x.dcm"]
    if mutation == "missing": names = []
    elif mutation == "duplicate": names *= 2
    elif mutation == "extra": names.append("train_series/1/0/x.dcm")
    elif mutation == "unsafe": names = ["train_series/0/../x.dcm"]
    elif mutation == "roots": names.append("train_images/0/0/y.dcm")
    with zipfile.ZipFile(archive, "x") as z:
        for name in names:
            if mutation == "symlink":
                info = zipfile.ZipInfo(name)
                info.external_attr = (0o120777 << 16)
                z.writestr(info, "outside")
            elif mutation == "duplicate" and z.namelist():
                with pytest.warns(UserWarning, match="Duplicate"):
                    z.writestr(name, "not read")
            else:
                z.writestr(name, "not read")
        if mutation == "missing": z.writestr("train_series/", "")
    with pytest.raises(ValueError, match=match):
        bench.scan_inventory(rows(1), archive, "a"*64)


def test_only_series_metadata_schema_allowed(tmp_path):
    with pytest.raises(ValueError, match="Only train_series"):
        bench.read_series(tmp_path/"train.csv")
    p = tmp_path/"train_series.csv"
    p.write_text("StudyInstanceUID,Report\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        bench.read_series(p)


def test_subset_view_uses_safe_whole_root_symlink_and_subset_metadata(tmp_path):
    comp = tmp_path/"competition"
    comp.mkdir()
    frozen = intended(comp)
    images = make_images(comp/"train_images")
    # Fail loudly if this environment cannot create symlinks; Kaggle/Linux is required.
    try:
        view = bench.build_view(comp, tmp_path/"view", frozen)
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows lacks symlink privilege; Linux/Kaggle view preflight still required")
        raise
    assert (view/"test_series").is_symlink()
    assert (view/"test_series").resolve() == images.resolve()
    assert cohort.read_ids(view/"test.csv") == frozen["ids"]
    assert bench.jread(view/"inventory-check.json") == bench.check_inventory(bench.jread(view/"image-inventory.json"), frozen)
    with (view/"test_series.csv").open(newline="") as f:
        assert {r[cohort.UID] for r in csv.DictReader(f)} == set(frozen["ids"])
    # DirectoryDicomSource's exact containment predicate remains true.
    row = next(r for r in rows() if r[cohort.UID] == frozen["ids"][0])
    assert (view/"test_series"/row[cohort.UID]/row["SeriesInstanceUID"]).resolve().is_relative_to((view/"test_series").resolve())
    with pytest.raises(ValueError, match="Fresh"):
        bench.build_view(comp, view, frozen)


def test_wrapper_rewrite_routes_both_stages_and_keeps_gate_and_backend_reset():
    original = bench.code_cell(bench.jread(bench.RELEASE))
    wrapped = bench.rewrite_cell(original, Path("/kaggle/working/fake"))
    assert "Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')" not in wrapped
    assert "Path('/kaggle/input/competitions/rsna-knee-abnormality-detection/sample_submission.csv')" not in wrapped
    assert "_torch.backends.cudnn.benchmark = False" in wrapped
    assert "_exp169_signal.alarm(24300)" in wrapped
    assert "if _r.get('status') != 'PASS_BUNDLE':" in wrapped
    compile(wrapped, "benchmark-cell", "exec")
    with pytest.raises(ValueError, match="entrypoint"):
        bench.rewrite_cell(original.replace("\nmain()\n", "\nother()\n"), Path("/x"))


def test_stage_is_self_contained_private_no_upload_and_detects_drift(tmp_path, monkeypatch):
    series = series_file(tmp_path)
    inventory_path = tmp_path/"image-inventory.json"
    bench.jwrite(inventory_path, image_inventory(series_sha=bench.sha(series)))
    def forbidden(*a, **k):
        raise AssertionError("External process/network forbidden during stage")
    monkeypatch.setattr(bench.subprocess, "check_output", forbidden)
    before = bench.sha(bench.ROOT/"tools/exp169_partner_bundle_release.py")
    dest = tmp_path/"stage"
    bench.stage(dest, inventory_path, series=series)
    metadata = bench.jread(dest/"kernel/kernel-metadata.json")
    assert metadata["dataset_sources"] == bench.DATASETS
    assert metadata["is_private"] and not metadata["enable_internet"]
    plan = bench.push_plan(dest)
    assert plan["status"] == "PLAN_ONLY_NOT_EXECUTED"
    assert "--version <VERSION>" in "\n".join(plan["commands"])
    assert "competitions submit" not in "\n".join(plan["commands"])
    notebook = bench.jread(dest/"kernel/benchmark.ipynb")
    compile(bench.code_cell(notebook), "generated", "exec")
    assert bench.jread(dest/"config.json")["intended"] == bench.choose_cohort(bench.jread(inventory_path))
    assert "image-inventory.json" in bench.jread(dest/"build-receipt.json")["files"]
    assert bench.sha(bench.ROOT/"tools/exp169_partner_bundle_release.py") == before
    (dest/"kernel/kernel-metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="drift"):
        bench.checked_stage(dest)


@pytest.mark.parametrize("mutation", ["rename", "remove", "add", "unselected"])
def test_runtime_inventory_drift_fails_before_view_or_inference(tmp_path, mutation):
    comp = tmp_path/"competition"
    comp.mkdir()
    frozen = intended(comp)
    images = make_images(comp/"train_images")
    uid = frozen["ids"][0] if mutation != "unselected" else next(str(i) for i in range(100) if str(i) not in frozen["ids"])
    directory = images/uid/"0"
    if mutation in ("rename", "unselected"):
        (directory/"0.dcm").rename(directory/"replacement.dcm")  # Same count, different filename.
    elif mutation == "add": (directory/"added.dcm").touch()
    else: (directory/"0.dcm").unlink()
    with pytest.raises(ValueError, match="inventory|binding"):
        bench.build_view(comp, tmp_path/"view", frozen)
    assert not (tmp_path/"view").exists()


@pytest.mark.parametrize("escape", ["directory", "file"])
def test_filename_inventory_refuses_containment_escapes_without_symlink_privilege(tmp_path, monkeypatch, escape):
    root = make_images(tmp_path/"images", rows(1)).resolve()
    target = root/"0"/"0"
    if escape == "file": target /= "0.dcm"
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        return tmp_path/"outside" if path == target else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="escaping"):
        bench.scan_inventory(rows(1), root, "a"*64)


def test_inventory_command_and_stage_bind_full_filename_inventory(tmp_path):
    series = series_file(tmp_path)
    images = make_images(tmp_path/"train_images")
    result = bench.inventory(images, tmp_path/"registered", series)
    path = tmp_path/"registered/image-inventory.json"
    assert result["inventory_sha256"] == bench.digest_json(bench.jread(path))
    staged = tmp_path/"stage"
    bench.stage(staged, path, series=series)
    bench.checked_stage(staged)
    value = bench.jread(staged/"image-inventory.json")
    value["series"][0]["image_count"] += 1
    bench.jwrite(staged/"image-inventory.json", value)
    with pytest.raises(ValueError, match="Stage drift"):
        bench.checked_stage(staged)


@pytest.mark.parametrize("mutation", ["series_hash", "missing_series"])
def test_bad_inventory_cannot_create_a_stage(tmp_path, mutation):
    series = series_file(tmp_path)
    value = image_inventory(series_sha=bench.sha(series))
    if mutation == "series_hash": value["train_series_sha256"] = "b"*64
    else: value["series"].pop()
    path = tmp_path/"inventory.json"
    bench.jwrite(path, value)
    with pytest.raises(ValueError, match="drift|membership"):
        bench.stage(tmp_path/"stage", path, series=series)
    assert not (tmp_path/"stage").exists()


def test_no_inventory_or_forged_coverage_cannot_qualify(tmp_path):
    with pytest.raises(TypeError):
        bench.stage(tmp_path/"stage")
    frozen = intended(tmp_path)
    value = image_inventory(series_sha=bench.sha(tmp_path/"train_series.csv"))
    bad = copy.deepcopy(frozen)
    bad["image_count_coverage"]["requirements"]["total_images"]["tail_ids"] = []
    with pytest.raises(ValueError, match="binding"):
        bench.check_inventory(value, bad)
    bad = copy.deepcopy(frozen)
    bad["ids"][0] = "not-frozen"
    with pytest.raises(ValueError, match="binding"):
        bench.check_inventory(value, bad)


def test_trio_wrappers_delegate_without_changing_values_or_policy(tmp_path, monkeypatch):
    observer = bench.Observer(tmp_path, Path("/fake/view"))
    monkeypatch.setattr(bench, "_BENCH", observer)
    adapter = SimpleNamespace(input_record=lambda *a: "record", output_values=lambda *a: "aggregate", save_raw=lambda *a: "saved")
    ns = {"_E152": adapter, "find_test_root": lambda: "/real/test",
          "_pre_one": lambda uid: (uid, "volume", "mask"), "load_model": lambda *a: "model",
          "infer_pair_probs": lambda *a: "probabilities", "gpu_windows": lambda *a: "windows"}
    observer.install_trio(ns, adapter)
    assert ns["find_test_root"]() == "/fake/view"
    assert ns["_pre_one"]("a") == ("a", "volume", "mask")
    def fake_consumer():
        sid = "a"
        assert adapter.input_record() == "record"
        return ns["infer_pair_probs"](1, 2, 3, "cuda:0")
    assert fake_consumer() == "probabilities"
    assert ns["load_model"](1, 2, 3, "cuda:1") == "model"


def test_partner_fake_records_uid_prepare_gpu_and_default_argument_path(tmp_path, monkeypatch):
    observer = bench.Observer(tmp_path, Path("/view"))
    class Corpus:
        def volume(self, uid): return "volume"
        def mask(self, uid): return "mask"
    dicom = ModuleType("raptor_dicom"); dicom.DicomCorpus = Corpus
    runtime = ModuleType("raptor_runtime"); runtime.normalize_input = lambda x, _: x
    monkeypatch.setitem(sys.modules, "raptor_dicom", dicom)
    monkeypatch.setitem(sys.modules, "raptor_runtime", runtime)
    module = SimpleNamespace(__file__=str(bench.ROOT/"tools/exp169_partner_bundle_release.py"),
                             _load_models=lambda *a: "models", _infer_one=lambda *a: "p",
                             aggregate=lambda x: x, final_values=lambda *a: a, write_csv=lambda *a: "csv")
    def partner(work, comp, ids, manifest, deadline):
        c = Corpus()
        for i, uid in enumerate(ids):
            c.volume(uid); c.mask(uid)
            runtime.normalize_input("image", "imagenet")
            runtime.normalize_input("triplet", "imagenet")
            assert module._infer_one("model", "x", "cuda:0") == "p"
        return "result", "metrics"
    module.partner_scores = partner
    module.pilot = lambda *a: module.partner_scores(None, None, ["a", "b"], None, 123)
    def serve(*a, partner=partner):
        return partner(None, None, ["a", "b", "c"], None, 123)
    module.serve = serve
    observer.install_partner(module)
    assert module.pilot() == ("result", "metrics")
    assert module.serve() == ("result", "metrics")
    events = [json.loads(l) for f in observer.events.glob("*.jsonl") for l in f.read_text().splitlines()]
    summary = bench.summarize_events(events)
    assert summary["partner_pilot_scores/partner_gpu"]["studies"] == 2
    assert summary["partner_full/partner_gpu"]["studies"] == 3
    assert summary["partner_full/partner_prepare"]["studies"] == 3
    assert all(observer.stages[k] >= 0 for k in ("pilot", "serve", "partner_full", "partner_pilot_scores"))


def receipt_fixture(n=48, hidden=1300):
    return {"config": {"intended": {"studies": n}, "assumed_hidden_studies": hidden},
            "stages": {"trio": 300., "partner_full": 200., "pilot": 190., "serve": 201., "partner_pilot_scores": 189.},
            "wall_before_result_packaging_seconds": 750., "result_packaging_seconds": 1.,
            "inventory_preflight_seconds": 2.}


def test_projection_counts_pilot_once_scales_actual_N_not_three_and_checks_both_limits():
    r = receipt_fixture()
    result = bench.projection(r, {"pilot": {"elapsed_seconds": 189., "studies": 40}})
    assert result["linear_total_seconds"] == pytest.approx(501*1300/48+190+59+2)
    assert result["conservative_total_seconds"] == pytest.approx(1.5*(501*1300/48+59)+190+2+600)
    assert result["projected_release_pilot_gate_partner_seconds"] == pytest.approx(1.5*189*1300/40+600)
    assert result["guard_seconds"] == 24300 and result["kaggle_limit_seconds"] == 32400
    bad = receipt_fixture(hidden=3000)
    assert bench.projection(bad, {"pilot": {"elapsed_seconds": 189., "studies": 40}})["guard_margin_seconds"] < 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., 0.])
def test_projection_rejects_invalid_measurements(value):
    receipt = receipt_fixture()
    receipt["stages"]["trio"] = value
    with pytest.raises(ValueError, match="durations"):
        bench.projection(receipt, {"pilot": {"elapsed_seconds": 189., "studies": 40}})


def test_collect_rejects_zip_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "checked_stage", lambda _: {})
    archive = tmp_path/"bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.json", "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        bench.collect(tmp_path, archive, tmp_path/"collected")
    assert not (tmp_path/"escape.json").exists()


@pytest.mark.parametrize("evidence", ["valid", "forged_coverage", "renamed_file", "missing_check", "false_check"])
def test_package_collect_and_report_full_fake_receipt(tmp_path, monkeypatch, evidence):
    frozen = intended(tmp_path)
    value = image_inventory(series_sha=bench.sha(tmp_path/"train_series.csv"))
    output = tmp_path/"remote"; output.mkdir()
    config = {"intended": frozen, "assumed_hidden_studies": 1300}
    source_sha = {"benchmark": "fake-benchmark", "cohort_helper": "fake-cohort"}
    receipt = receipt_fixture()
    receipt.update(config=config, status="COMPLETE_SAMPLE_BENCHMARK", source_sha256=source_sha,
                   memory={"peak_system_used_bytes": 5*2**30, "peak_device_used_bytes": [4*2**30]*2})
    checked = bench.check_inventory(value, frozen)
    if evidence == "forged_coverage": frozen["image_count_coverage"]["requirements"]["total_images"]["tail_ids"] = []
    elif evidence == "renamed_file": value["series"][0]["filenames_sha256"] = "c"*64
    elif evidence == "false_check": checked["status"] = "NOT_VERIFIED"
    bench.jwrite(output/"image-inventory.json", value)
    if evidence != "missing_check": bench.jwrite(output/"inventory-check.json", checked)
    executed = {"status": "PASS_BUNDLE", "candidate": "TRIO_PLUS_BUNDLE", "studies": 48,
                "stages": {"parity": {"status": "PASS"}}, "qualification": "FULL_COHORT",
                "executed_arms": ["EXP155_BLEND3", "L2_RADR50_10", "DINO_BASE_5", "EXP170_L2_CONVNEXT_T_5"],
                "weights": {"trio": .85, "bundle": .15},
                "pilot": {"studies": 40, "elapsed_seconds": 189.}}
    bench.jwrite(output/"benchmark-receipt.json", receipt)
    bench.write_csv(output/"benchmark-predictions.csv", [cohort.UID, *bench.TARGETS],
                    [{cohort.UID: u, **{t: .5 for t in bench.TARGETS}} for u in frozen["ids"]])
    executed["submission_sha256"] = bench.sha(output/"benchmark-predictions.csv")
    bench.jwrite(output/"executed-candidate.json", executed)
    events = []
    for phase, kinds, ids in [("trio", ["trio_decode", "trio_inference"], frozen["ids"]),
                              ("partner_full", ["partner_decode", "partner_prepare", "partner_gpu"], frozen["ids"]),
                              ("partner_pilot_scores", ["partner_decode", "partner_prepare", "partner_gpu"], frozen["ids"][:40])]:
        events += [{"phase": phase, "kind": kind, "uid": u, "seconds": .1} for kind in kinds for u in ids]
    (output/"events").mkdir()
    (output/"events/1.jsonl").write_text("\n".join(json.dumps(e) for e in events)+"\n")
    archive = tmp_path/"results.zip"
    bench.pack_results(output, archive)
    build = {"config": config, "source_sha256": source_sha}
    monkeypatch.setattr(bench, "checked_stage", lambda _: build)
    bench.jwrite(tmp_path/"build-receipt.json", build)
    bench.collect(tmp_path, archive, tmp_path/"collected")
    if evidence != "valid":
        with pytest.raises((ValueError, FileNotFoundError)):
            bench.report(tmp_path/"collected", tmp_path/"report")
        assert not (tmp_path/"report").exists()
        return
    result = bench.report(tmp_path/"collected", tmp_path/"report")
    assert result["qualification"] == "TRAIN_SAMPLE_ONLY_NOT_FULL_COHORT"
    report = bench.jread(tmp_path/"report/report.json")
    assert report["release_reported_qualification_ignored"] == "FULL_COHORT"
    assert report["full_rehearsal_required"] is True
    assert report["hb41_coverage_gate"] == "PASS_FILENAME_IMAGE_COUNTS_AND_LONG_SERIES"
    assert report["inventory_preflight_seconds"] == 2.
    (tmp_path/"collected/benchmark-predictions.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="drift"):
        bench.report(tmp_path/"collected", tmp_path/"report2")
