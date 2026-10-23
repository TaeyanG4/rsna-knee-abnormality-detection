"""CPU-only release contracts; no labels, torch import, account or GPU calls."""
from types import SimpleNamespace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp174_release_tested", ROOT / "tools/exp174_own0949_jev_release.py")
release = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = release
spec.loader.exec_module(release)


def _checkpoint():
    return dict(model={"fake": 1}, arch=release.ARCH, res=384, inference_res=320,
                lab=release.TARGETS, contract=release.CONTRACT, k_eval=94,
                norm="imagenet", window_policy="legacy", oai=False,
                checkpoint_kind="fixed-swa-bn", swa_epochs_1based=[14, 11, 12], seed=42,
                provenance=dict(protocol_sha256="a" * 64, initial_state_sha256="b" * 64,
                                run_dir=release.RUN, train_n=4347, seed=42, world_size=1,
                                execution=dict(grad_checkpointing=False, batch=1, effective_batch=16)))


def test_shared_source_pinned_and_isolated():
    assert release.sha(release.BASE_PATH) == release.BASE_SHA256
    original_spec = importlib.util.spec_from_file_location("exp152_independent_release_tested", release.BASE_PATH)
    original = importlib.util.module_from_spec(original_spec)
    sys.modules[original_spec.name] = original
    original_spec.loader.exec_module(original)
    assert original.VERSION == "1.0.1"
    assert original.OWN_NAME != release.OWN_NAME
    assert original.RUN.endswith("S42v2")
    assert release._shared.VERSION == "1.0.0"
    assert release._shared.RUN == release.RUN


def test_parser_excludes_gpu_and_account_commands():
    parser = release.parser()
    commands = next(a for a in parser._actions if hasattr(a, "choices") and isinstance(a.choices, dict)).choices
    assert set(commands) == {"average", "stage", "parity-plan", "public3-plan", "compare-parity", "push-plan"}
    average = parser.parse_args(["average"])
    assert average.protocol == str(release.PROTOCOL)
    assert average.out == str(release.BASE / "average-v1")
    assert average.checkpoint is None
    stage = parser.parse_args(["stage", "--out", "fake-stage"])
    assert stage.average == str(release.BASE / "average-v1")


def test_average_receipt_identity_create_only(tmp_path):
    path = tmp_path / "average-receipt.json"
    release.write_json(path, {"schema": "exp152-average-v1", "oai": False})
    receipt = release.read_json(path)
    assert receipt == {"schema": "exp174-average-v1", "oai": False,
                       "shared_serving_sha256": release.BASE_SHA256}
    with pytest.raises(FileExistsError):
        release.write_json(path, receipt)


def test_private_payload_cannot_export_training_identity():
    ck = _checkpoint()
    ck["label_set"] = "training-lineage-only"
    payload = release.private_payload(ck)
    assert "label_set" not in payload
    assert set(payload["provenance"]) == {"protocol_sha256", "initial_state_sha256", "run_dir"}
    assert "train_n" not in payload["provenance"]
    release.own_meta(payload, "a" * 64)
    ck["provenance"]["study_ids"] = ["fake"]
    with pytest.raises(ValueError, match="Unreviewed provenance"):
        release.private_payload(ck)


@pytest.mark.parametrize("field,value", [("train_n", 4348), ("seed", 43), ("world_size", 2)])
def test_private_payload_rejects_changed_population_or_seed(field, value):
    ck = _checkpoint()
    ck["provenance"][field] = value
    with pytest.raises(ValueError, match="training identity"):
        release.private_payload(ck)


def test_private_payload_rejects_gradient_checkpointing():
    ck = _checkpoint()
    ck["provenance"]["execution"]["grad_checkpointing"] = True
    with pytest.raises(ValueError, match="execution"):
        release.private_payload(ck)


def test_notebook_archive_runtime_and_private_t4_contract():
    variant = "own"
    nb, _, _ = release.notebook(variant, release.ASSET_DATASET, "a" * 64, "b" * 64, "c" * 64, [])
    source = nb["cells"][-1]["source"]
    compile(source, "exp174-notebook-test", "exec")
    assert "import exp174_own0949_jev_release as _e152" in source
    assert "'exp174_own0949_jev_release.py', 'exp152_own0949_release.py', 'public-serving.py'" in source
    assert "exp174-assets.zip.bin" in source
    assert "exp152-assets.zip.bin" not in source
    assert "main()" in source
    metadata = release.kernel_metadata(variant)
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"][0] == release.ASSET_DATASET
    assert metadata["dataset_sources"] == [release.ASSET_DATASET]
    assert release.POLICY == release._shared.POLICY
    assert set(release.KERNELS) == {"own"}


def test_unregistered_blend_is_not_a_release_variant():
    with pytest.raises(ValueError, match="own only"):
        release.notebook("blend", release.ASSET_DATASET, "a" * 64, "b" * 64, "c" * 64, [])
    with pytest.raises(ValueError, match="internal qualification only"):
        release.bind({}, {}, {}, "blend", Path("fake"))


def test_average_requires_completion_before_reconstruction(monkeypatch):
    calls = []
    args = SimpleNamespace(protocol="fake-protocol", checkpoint=None, out="fake-average")
    guard = SimpleNamespace(require_completed=lambda protocol: calls.append(("completed", protocol)))
    monkeypatch.setattr(release, "load_module", lambda name, path: guard)
    monkeypatch.setattr(release._shared, "average", lambda arguments: calls.append(("average", arguments)) or {"status": "fake"})
    assert release.average(args) == {"status": "fake"}
    assert calls == [("completed", "fake-protocol"), ("average", args)]


def test_internal_qualification_bind_restores_own_only(monkeypatch):
    captured = []
    monkeypatch.setattr(release, "_original_bind", lambda *args: captured.append(set(release._shared.KERNELS)))
    monkeypatch.setattr(release, "_qualification_active", True)
    release.bind({}, {}, {}, "blend", Path("fake"))
    assert captured == [{"own", "blend"}]
    assert set(release._shared.KERNELS) == {"own"}


def test_public3_plan_writes_runner_but_does_not_call_gpu(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    (stage / "assets").mkdir(parents=True)
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    release.write_json(stage / "assets/release-manifest.json", {"protocol_sha256": release.sha(protocol)})
    monkeypatch.setattr(release, "checked_stage", lambda path: (stage, {}, {}))
    monkeypatch.setattr(release._shared, "verify_plan", lambda path, report: None)
    def fail_gpu(args):
        pytest.fail("Plan command attempted GPU execution")
    monkeypatch.setattr(release._shared, "public3", fail_gpu)
    args = SimpleNamespace(stage=stage, plan=tmp_path / "parity-plan", protocol=protocol,
                           public_checkpoint=tmp_path / "public.pt", competition=tmp_path / "competition",
                           dicom_root=None, device="cuda:0", out=tmp_path / "runner-plan",
                           result_out=tmp_path / "results", python="gpu-python.exe")
    result = release.public3_plan(args)
    assert result["status"] == "PLAN_ONLY_NOT_EXECUTED"
    runner = args.out / "run-public3-b-only.py"
    compile(runner.read_text(encoding="utf-8"), str(runner), "exec")
    assert result["runner_sha256"] == release.sha(runner)
    assert "module._qualification_public3(args)" in runner.read_text(encoding="utf-8")
    assert json.loads((args.out / "runner-arguments.json").read_text())["device"] == "cuda:0"


def test_stage_requires_verified_average_before_torch_import(tmp_path):
    average = tmp_path / "average"
    average.mkdir()
    release.write_json(average / "average-receipt.json", {"status": "INCOMPLETE"})
    args = SimpleNamespace(average=average, out=tmp_path / "stage", dataset=release.ASSET_DATASET)
    with pytest.raises(ValueError, match="average receipt invalid"):
        release.stage(args)
    assert not args.out.exists()


def test_push_plan_contains_only_private_own_qualification(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    monkeypatch.setattr(release, "checked_stage", lambda path: (stage, {"dataset": release.ASSET_DATASET}, {}))
    args = SimpleNamespace(stage=stage, out=tmp_path / "push-plan.ps1", python="gpu-python.exe",
                           parity_plan="registered-plan", local_public3="local-public3", parity_out="parity.json")
    result = release.push_plan(args)
    text = args.out.read_text(encoding="utf-8")
    assert result["status"] == "PLAN_ONLY_NOT_EXECUTED"
    assert "kaggle datasets create -p ." in text
    assert "kernels push" in text and "--accelerator NvidiaTeslaT4" in text
    assert "knee-exp174-own0949-jev-v1" in text
    assert "compare-parity" in text and "--own" in text
    assert "--blend" not in text
    assert "kaggle_submit_once" not in text
    assert "competitions submit" not in text
    assert "blend-t4-output" not in text
    assert "--public" not in text
