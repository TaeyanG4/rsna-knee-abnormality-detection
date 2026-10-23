"""Preserve the anchor and fixed cohort when staging a larger engineering sample."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "anchor_runtime_benchmark", Path(__file__).resolve().parents[1] / "tools/anchor_runtime_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def fixture_inputs(tmp_path):
    package = tmp_path / "kaggle/packages/KNEE_ANCHOR_0943"
    package.mkdir(parents=True)
    cells = [{"cell_type": "code", "metadata": {"original": True}, "source": ["x = 1\n"],
              "outputs": [{"output_type": "stream", "text": ["old output\n"]}], "execution_count": 9}]
    (package / "anchor.ipynb").write_text(json.dumps({"cells": cells, "metadata": {"original": True}}))
    (package / "kernel-metadata.json").write_text(json.dumps({
        "code_file": "anchor.ipynb", "dataset_sources": ["owner/model/2"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "enable_gpu": True}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"ids": [f"uid-{i}" for i in range(100)],
        "excluded_corrupt_uids": ["bad"], "train_series_sha256": "test-series-hash"}))
    return package, cells, selection


def test_nested_samples_preserve_original_cells_and_dependencies(tmp_path):
    package, cells, selection = fixture_inputs(tmp_path)
    original_hash = benchmark.sha(package / "anchor.ipynb")
    for count in (50, 100):
        out = tmp_path / f"stage{count}"
        got = benchmark.build(tmp_path, selection, out, count)
        notebook = json.loads((out / f"anchor-runtime{count}.ipynb").read_text())
        meta = json.loads((out / "kernel-metadata.json").read_text())
        contract = json.loads((out / "selection-contract.json").read_text())
        assert notebook["cells"][1:-1] == cells
        assert contract["anchor_notebook_sha256"] == original_hash
        assert contract["ids"] == [f"uid-{i}" for i in range(count)]
        assert got["studies"] == contract["count"] == count
        assert meta["dataset_sources"] == ["owner/model/2"] and meta["enable_gpu"] is True
        assert meta["id"] == f"taeyangg4/knee-anchor-runtime{count}"
        assert meta["is_private"] is True and meta["enable_internet"] is False
        # Wrapper cells must compile without executing the notebook or reading data.
        for cell in (notebook["cells"][0], notebook["cells"][-1]):
            compile("".join(cell["source"]), "benchmark-wrapper", "exec")
    assert benchmark.sha(package / "anchor.ipynb") == original_hash


@pytest.mark.parametrize("problem", ["duplicate", "corrupt", "too_short", "wrong_count"])
def test_invalid_selection_never_creates_staging(tmp_path, problem):
    _, _, selection = fixture_inputs(tmp_path)
    value = json.loads(selection.read_text())
    if problem == "duplicate":
        value["ids"][-1] = value["ids"][0]
    elif problem == "corrupt":
        value["ids"][-1] = "bad"
    elif problem == "too_short":
        value["ids"].pop()
    selection.write_text(json.dumps(value))
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        benchmark.build(tmp_path, selection, out, 3 if problem == "wrong_count" else 100)
    assert not out.exists()


def test_existing_stage_is_immutable(tmp_path):
    _, _, selection = fixture_inputs(tmp_path)
    out = tmp_path / "stage"
    benchmark.build(tmp_path, selection, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        benchmark.build(tmp_path, selection, out, 100)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
