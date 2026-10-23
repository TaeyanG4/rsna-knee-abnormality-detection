"""Engineering selection and failure preservation; no GPU or remote actions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("runtime_benchmark", Path(__file__).resolve().parents[1] / "tools/d01_runtime_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_fixed_cohort_preserves_order_excludes_corruption_and_fills_reproducibly():
    first, _ = benchmark.fixed_ids(["gold", "b", "broken", "a", "b"], ["a", "b", "c", "d"], ["broken"], 3)
    second, _ = benchmark.fixed_ids(["gold", "b", "broken", "a", "b"], ["d", "c", "b", "a"], ["broken"], 3)
    assert first == second
    assert first[:2] == ["b", "a"] and len(set(first)) == 3
    assert not {"gold", "broken"} & set(first)
    with pytest.raises(ValueError):
        benchmark.fixed_ids([], ["a"], [], 2)


def test_header_scan_adds_compressed_study_without_replacing_base(monkeypatch):
    base = ["base"]
    metadata = [{benchmark.UID: uid, benchmark.SERIES: "series", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": "1"}
                for uid in ("base", "extra", "never")]
    adapter = SimpleNamespace(pick_series=lambda rs, plane, fluid, used:
                              rs[0] if plane == "Sagittal" and fluid == 1 and "series" not in used else None)
    source = SimpleNamespace(members=lambda split, uid, series: [SimpleNamespace(path=f"{uid}/first"), SimpleNamespace(path=f"{uid}/middle")])
    def header(source, member, uid, series, slot, selected):
        return {"uid": uid, "path": member.path, "compressed": uid == "extra", "transfer_syntax": "JPEG" if uid == "extra" else "RAW", "known_selected_slice": selected}
    monkeypatch.setattr(benchmark, "header_record", header)
    prior = [{"uid": "base", "slots": [{"series": "series", "selected_sources": [{"path": "base/first"}]}]}]
    result = benchmark.scan_headers(adapter, source, base, metadata, prior, ["extra", "never"], extra_limit=1)
    assert result["scanned_uids"] == ["base", "extra"]
    assert result["additional_compressed_uids"] == ["extra"]
    assert result["records"][0]["known_selected_slice"] is True
    assert result["records"][1]["known_selected_slice"] is False
    assert result["records"][0]["path"] == "base/first"
    assert base == ["base"]


def test_failed_offline_activation_saves_structured_receipt_without_prediction(tmp_path, monkeypatch):
    assets = tmp_path / "assets"; assets.mkdir()
    marker = assets / "d01-assets-manifest.json"; marker.write_text("{}")
    sha = benchmark.digest(marker)
    monkeypatch.setattr(benchmark, "ASSET_SHA", sha)
    selection = tmp_path / "selection.json"
    benchmark.dump(selection, {"assets_manifest_sha256": sha})
    def fail(*args):
        error = RuntimeError("decoder environment unavailable")
        error.receipt = {"status": "FAILED", "member": "F2", "uids": ["case"]}
        raise error
    package = SimpleNamespace(_validate_assets=lambda path: {}, _activate_offline_environment=fail)
    monkeypatch.setattr(benchmark, "passive_module", lambda *args: package)
    work = tmp_path / "results"
    with pytest.raises(RuntimeError, match="decoder environment"):
        benchmark.run(assets, tmp_path / "competition", selection, work)
    report = json.loads((work / "benchmark-receipt.json").read_text())
    assert report["status"] == "FAILED"
    assert report["failures"][0]["stage"] == "offline_environment"
    assert report["failures"][0]["inference_receipt"]["member"] == "F2"
    assert report["timings"]["worker_total_seconds"] > 0
    assert not (work / "engineering-raw-mean.csv").exists()


def test_distribution_preserves_slow_tail():
    result = benchmark.distribution([1] * 99 + [100])
    assert result["count"] == 100 and result["max"] == 100 and result["mean"] == 1.99
    assert benchmark.distribution([]) == {"count": 0}
