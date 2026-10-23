"""Small, local fixtures for the OAI selection and budget planner."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "oai_plan.py"
SPEC = importlib.util.spec_from_file_location("oai_plan", SCRIPT)
assert SPEC and SPEC.loader
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def table(path: Path, delimiter: str, fields: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    metadata = table(tmp_path / "image03.txt", "\t",
                     ["collection_id", "src_subject_id", "visit", "image_description",
                      "image_file"],
                     [["collection_id", "Subject ID", "Visit", "Description", "URL"],
                      ["2343", "101", "V00", "SAG_IW_TSE_RIGHT", "s3://bucket/a.tar.gz"],
                      ["2343", "101", "V00", "COR_MPR_RIGHT", "s3://bucket/b.tar.gz"],
                      ["2343", "102", "V00", "SAG_IW_TSE_LEFT", "s3://bucket/c.tar.gz"],
                      ["2343", "101", "V01", "SAG_IW_TSE_RIGHT", "s3://bucket/d.tar.gz"]])
    package = table(tmp_path / "package_file_metadata_1.txt", ",",
                    ["PACKAGE_FILE_ID", "NDA_S3_URL", "FILE_SIZE", "DOWNLOAD_ALIAS",
                     "SHORT_NAME"],
                    [["1", "s3://bucket/a.tar.gz", "100", "image03/a.tar.gz", "image03"],
                     ["2", "s3://bucket/b.tar.gz", "200", "image03/b.tar.gz", "image03"],
                     ["3", "s3://bucket/c.tar.gz", "300", "image03/c.tar.gz", "image03"],
                     ["4", "s3://bucket/d.tar.gz", "400", "image03/d.tar.gz", "image03"]])
    moaks = table(tmp_path / "kMRI_SQ_MOAKS_BICL00.txt", "|",
                  ["ID", "SIDE", "READPRJ", "VERSION", "V00MACLTR"],
                  [["101", "1: Right", "1", "0.2", "0"],
                   ["102", "2: Left", "1", "0.2", ""]])
    kl = table(tmp_path / "kxr_sq_bu00.txt", "|",
               ["ID", "SIDE", "READPRJ", "VERSION", "V00XRKL"],
               [["101", "1: Right", "1", "0.2", ""],
                ["102", "2: Left", "1", "0.2", "2"]])
    return {"metadata": metadata, "package": package, "moaks": moaks, "kl": kl}


def test_modes_and_study_keys(inputs: dict[str, Path]) -> None:
    packages = planner.package_sizes(inputs["package"])
    series = ("SAG_IW_TSE", "COR_MPR")
    moaks = planner.clinical_knees(inputs["moaks"], "V00", "moaks_knees")
    kl = planner.clinical_knees(inputs["kl"], "V00", "kl_knees")
    assert moaks == {("101", "R")}
    assert kl == {("102", "L")}
    selected = planner.select_images(inputs["metadata"], packages, "V00", series,
                                     moaks)
    assert [(r["study_key"], r["series"]) for r in selected] == [
        ("OAI_101_R_V00", "SAG_IW_TSE"), ("OAI_101_R_V00", "COR_MPR")]
    assert {r["study_key"] for r in planner.select_images(
        inputs["metadata"], packages, "V00", series, kl)} == {"OAI_102_L_V00"}
    assert len(planner.select_images(inputs["metadata"], packages, "V00",
                                     series, None)) == 3


def test_budget_counts_only_verified_complete_archives(inputs: dict[str, Path],
                                                       tmp_path: Path, monkeypatch) -> None:
    local = tmp_path / "raw"
    (local / "image03").mkdir(parents=True)
    (local / "image03/a.tar.gz").write_bytes(b"x" * 100)
    (local / "image03/b.tar.gz").write_bytes(b"x" * 99)  # partial: reserve all 200
    packages = planner.package_sizes(inputs["package"])
    selected = planner.select_images(inputs["metadata"], packages, "V00",
                                     ("SAG_IW_TSE", "COR_MPR"), {("101", "R")}, local)
    assert [r["incremental_bytes"] for r in selected] == [0, 200]
    free = planner.MIN_FREE_BYTES + 1000
    monkeypatch.setattr(planner.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    totals = planner.check_budget(selected, tmp_path, 300)
    assert totals["total_bytes"] == 300
    assert totals["projected_free_bytes"] == free - 500
    with pytest.raises(ValueError, match="disk budget rejected"):
        planner.check_budget(selected, tmp_path, 801)
    with pytest.raises(ValueError, match="nonnegative"):
        planner.check_budget(selected, tmp_path, -1)
    assert planner.check_budget(selected, tmp_path, 800)["projected_free_bytes"] == (
        planner.MIN_FREE_BYTES)


def test_unknown_size_and_duplicate_series_fail_closed(inputs: dict[str, Path],
                                                       tmp_path: Path) -> None:
    packages = planner.package_sizes(inputs["package"])
    packages.pop("s3://bucket/a.tar.gz")
    with pytest.raises(ValueError, match="missing package size"):
        planner.select_images(inputs["metadata"], packages, "V00",
                              ("SAG_IW_TSE",), None)
    packages = planner.package_sizes(inputs["package"])
    with inputs["metadata"].open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t").writerow(
            ["2343", "101", "V00", "SAG_IW_TSE_RIGHT", "s3://bucket/c.tar.gz"])
    with pytest.raises(ValueError, match="multiple archives"):
        planner.select_images(inputs["metadata"], packages, "V00",
                              ("SAG_IW_TSE",), None)


def test_clinical_missing_codes_and_malformed_rows(tmp_path: Path) -> None:
    kl = table(tmp_path / "kxr_sq_bu00.txt", "|",
               ["ID", "SIDE", "VERSION", "V00XRKL"],
               [["101", "1: Right", "0.9", ".A: Missing"],
                ["102", "2: Left", "0.9", "2: Definite"],
                ["103", "1: Right", "0.9", "99"],
                ["104", "1: Right", "0.9", "2.5"]])
    assert planner.clinical_knees(kl, "V00", "kl_knees") == {("102", "L")}
    moaks = table(tmp_path / "kMRI_SQ_MOAKS_BICL00.txt", "|",
                  ["ID", "SIDE", "VERSION", "V00MACLTR", "V00MCMNTS"],
                  [["101", "1: Right", "0.2", ".: Missing Form", "11"],
                   ["102", "2: Left", "0.2", "1: Partial", ""],
                   ["103", "1: Right", "0.2", "99", ""]])
    assert planner.clinical_knees(moaks, "V00", "moaks_knees") == {("102", "L")}
    with kl.open("a", encoding="utf-8") as stream:
        stream.write("105|1: Right|0.9\n")
    with pytest.raises(ValueError, match="malformed row width"):
        planner.clinical_knees(kl, "V00", "kl_knees")


def test_alias_is_safe_without_download_root(inputs: dict[str, Path]) -> None:
    packages = planner.package_sizes(inputs["package"])
    packages["s3://bucket/a.tar.gz"] = (100, "../elsewhere/a.tar.gz")
    with pytest.raises(ValueError, match="unsafe download alias"):
        planner.select_images(inputs["metadata"], packages, "V00",
                              ("SAG_IW_TSE",), None)


def test_cli_writes_two_files_and_never_overwrites(inputs: dict[str, Path],
                                                  tmp_path: Path, monkeypatch,
                                                  capsys) -> None:
    monkeypatch.setattr(planner.shutil, "disk_usage", lambda _: SimpleNamespace(
        free=planner.MIN_FREE_BYTES + 10_000))
    args = ["--visit", "V00", "--mode", "moaks_knees",
            "--metadata", str(inputs["metadata"]), "--package-file", str(inputs["package"]),
            "--moaks", str(inputs["moaks"]), "--inflight-bytes", "500",
            "--out-dir", str(tmp_path / "selection")]
    assert planner.main(args) == 0
    assert '"total_bytes": 300' in capsys.readouterr().out
    csv_path = tmp_path / "selection/V00_moaks_knees_4series.csv"
    txt_path = csv_path.with_suffix(".txt")
    assert csv_path.exists() and txt_path.read_text(encoding="utf-8") == (
        "s3://bucket/a.tar.gz\ns3://bucket/b.tar.gz\n")
    before = csv_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        planner.main(args)
    assert csv_path.read_bytes() == before

    kl_args = ["--visit", "V00", "--mode", "kl_knees",
               "--metadata", str(inputs["metadata"]),
               "--package-file", str(inputs["package"]),
               "--kl", str(inputs["kl"]), "--inflight-bytes", "0",
               "--out-dir", str(tmp_path / "selection")]
    assert planner.main(kl_args) == 0
    assert (tmp_path / "selection/V00_kl_knees_4series.txt").read_text(
        encoding="utf-8") == "s3://bucket/c.tar.gz\n"
