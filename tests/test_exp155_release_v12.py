"""EXP155 v1.2 local release checks; synthetic checkpoints and no network/CUDA."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp155_own0949_oai_release_v12 as release
from tools import exp155_predict_gold58_v12 as gold


@pytest.fixture(autouse=True)
def no_cuda_or_network(monkeypatch):
    assert not torch.cuda.is_initialized()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU-only release test touched CUDA or network")
    for name in ("_lazy_init", "init", "manual_seed_all", "is_available", "set_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    import socket
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    yield
    assert not torch.cuda.is_initialized()


def test_pinned_base_and_passive_import():
    assert release.VERSION == "1.2.0"
    assert release.BASE_SHA256 == "ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2"
    assert release.sha(release.BASE_PATH) == release.BASE_SHA256
    assert release._shared is not release
    spec = importlib.util.spec_from_file_location("exp155_v12_independent", release.TOOL)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    assert other._shared is not release._shared
    assert other.BASE_SHA256 == release.BASE_SHA256
    assert release.TOOL.name == "exp155_own0949_oai_release_v12.py"


def test_three_equal_average_tie_ranks_and_old_variant():
    own = np.tile(np.asarray([[.1], [.3], [.3], [.9]]), (1, 12))
    public = np.tile(np.asarray([[.9], [.3], [.3], [.1]]), (1, 12))
    member = np.tile(np.asarray([[.2], [.2], [.8], [.8]]), (1, 12))
    actual = release.output_values([own, public, member], "blend3", None)
    expected = (np.asarray([0, .5, .5, 1]) + np.asarray([1, .5, .5, 0])
                + np.asarray([1/6, 1/6, 5/6, 5/6])) / 3
    np.testing.assert_allclose(actual[:, 0], expected)
    np.testing.assert_array_equal(release.output_values([own], "own", lambda value: value), own)
    with pytest.raises(ValueError, match="three-arm|Three rank"):
        release.output_values([own, public], "blend3", None)


def test_notebooks_include_member_and_authenticate_v12():
    for variant in release.KERNELS:
        notebook, _, description = release.notebook(
            variant, release.ASSET_DATASET, "a"*64, "b"*64, "c"*64, [])
        code = notebook["cells"][-1]["source"]
        compile(code, "exp155-v12-notebook", "exec")
        assert "import exp155_own0949_oai_release_v12 as _e152" in code
        assert release.MEMBER_NAME in code
        assert "exp155_own0949_oai_release_v11.py" not in code
        if variant == "blend3":
            assert "'blend3'" in code and "_files['" + release.MEMBER_NAME + "']" in code
            assert "1/3" in description
        metadata = release.kernel_metadata(variant)
        assert metadata["is_private"] and not metadata["enable_internet"]
        assert metadata["title"] == release.KERNELS[variant]


def fake_stage(tmp_path, monkeypatch):
    average = tmp_path / "average"
    average.mkdir()
    state = {"x": torch.tensor([1.])}
    ck = dict(model=state, arch=release.ARCH, res=384, inference_res=320,
              lab=release.TARGETS, contract=release.CONTRACT, k_eval=94,
              norm="imagenet", window_policy="legacy", oai=True,
              checkpoint_kind="fixed-swa-bn", swa_epochs_1based=release.EPOCHS,
              seed=42, provenance={"protocol_sha256": "a"*64, "run_dir": release.RUN})
    torch.save(ck, average / release.OWN_NAME)
    release.write_json(average / "average-receipt.json", dict(
        schema="exp155-average-v1.1", oai=True, status="VERIFIED_FIXED_SWA_BN",
        epochs_1based=release.EPOCHS, checkpoint_sha256=release.sha(average / release.OWN_NAME),
        protocol_sha256="a"*64, state_schema=release.state_schema(state),
        source_final_sha256="b"*64, source_bindings_sha256="c"*64,
        shared_serving_sha256=release.BASE_SHA256, private_test_ids=["FAKE_DO_NOT_EXPORT"]))
    member_dir = tmp_path / "member"
    member_dir.mkdir()
    member = member_dir / release.MEMBER_NAME
    member_protocol_sha = release.sha(ROOT / "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json")
    mck = dict(ck, oai=False, provenance={"protocol_sha256": member_protocol_sha,
                                           "train_ids": ["FAKE_DO_NOT_EXPORT"]})
    torch.save(mck, member)
    release.write_json(member_dir / "average-receipt.json", dict(
        status="VERIFIED_FIXED_SWA_BN", checkpoint_sha256=release.sha(member),
        protocol_sha256=member_protocol_sha, epochs_1based=release.EPOCHS))
    monkeypatch.setattr(release, "public_source", lambda: ("# fake public code\n", []))
    monkeypatch.setattr(release, "serving_source", lambda source: source)
    dest = tmp_path / "stage"
    release.stage(SimpleNamespace(average=average, exp143_checkpoint=member,
                                  out=dest, dataset=release.ASSET_DATASET))
    return dest


def test_stage_archive_and_push_plan_are_local(tmp_path, monkeypatch):
    stage = fake_stage(tmp_path, monkeypatch)
    _, report, manifest = release.checked_stage(stage)
    assert set(report["kernels"]) == {"own", "blend", "blend3"}
    assert manifest["ranks"]["blend3"].find("1/3 each") >= 0
    with zipfile.ZipFile(stage / "dataset/exp155-assets.zip.bin") as archive:
        assert release.TOOL.name in archive.namelist()
        assert release.MEMBER_NAME in archive.namelist()
        assert b"FAKE_DO_NOT_EXPORT" not in archive.read("release-manifest.json")
    spec = importlib.util.spec_from_file_location("exp155_v12_packaged", stage / "assets" / release.TOOL.name)
    packaged_runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packaged_runtime)
    packaged_runtime.verify_assets(stage / "assets", manifest)
    packaged = release.safe_load(stage / "assets" / release.MEMBER_NAME)
    assert set(packaged["provenance"]) == {"protocol_sha256"}
    plan = tmp_path / "push-plan.ps1"
    result = release.push_plan(SimpleNamespace(stage=stage, out=plan))
    assert result["status"] == "PLAN_ONLY_NOT_EXECUTED"
    text = plan.read_text(encoding="utf-8")
    assert release.KERNELS["blend3"] in text
    assert "three-arm raw.npz parity" in text
    assert all(line.startswith("#") for line in text.splitlines()
               if "kaggle_submit_once.py" in line)


def test_gold58_wrapper_uses_v12_guard():
    assert gold.check_checkpoint.__globals__["own_meta"] is release.own_meta
    assert gold.driver.check_checkpoint.__module__.endswith("exp155_own0949_oai_v11")
