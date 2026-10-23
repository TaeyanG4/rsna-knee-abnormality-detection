"""Tiny CPU-only release tests; no real patient data, CUDA calls or network."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import exp152_own0949_release as original
import exp155_own0949_oai_release as m


@pytest.fixture(autouse=True)
def forbid_cuda_and_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU-only test attempted CUDA/network")
    for name in ("_lazy_init", "init", "is_available", "is_initialized", "device_count",
                 "get_device_name", "get_device_capability", "manual_seed", "manual_seed_all", "set_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def checkpoint(state=None):
    return dict(model=state if state is not None else {"x": torch.tensor([1.])},
                arch=m.ARCH, res=384, inference_res=320, lab=m.TARGETS, contract=m.CONTRACT,
                k_eval=94, norm="imagenet", window_policy="legacy", oai=True,
                checkpoint_kind="fixed-swa-bn", swa_epochs_1based=m.EPOCHS, seed=42,
                provenance={"protocol_sha256": "a" * 64, "run_dir": m.RUN})


def fake_stage(tmp_path, monkeypatch):
    avg = tmp_path / "average"
    avg.mkdir()
    ck = checkpoint()
    torch.save(ck, avg / m.OWN_NAME)
    m.write_json(avg / "average-receipt.json", dict(schema="exp155-average-v1", oai=True,
                 status="VERIFIED_FIXED_SWA_BN", epochs_1based=m.EPOCHS,
                 checkpoint_sha256=m.sha(avg / m.OWN_NAME), protocol_sha256="a" * 64,
                 state_schema=m.state_schema(ck["model"]), source_final_sha256="b" * 64,
                 source_bindings_sha256="c" * 64, shared_serving_sha256=m.BASE_SHA256,
                 # This local-only field must never leak into staged manifest.
                 private_test_ids=["OAI_FAKE_DO_NOT_EXPORT"]))
    monkeypatch.setattr(m, "public_source", lambda: ("# fake public code\n", []))
    monkeypatch.setattr(m, "serving_source", lambda value: value)
    dest = tmp_path / "stage"
    m.stage(SimpleNamespace(average=avg, out=dest, dataset=m.ASSET_DATASET))
    return dest


def test_import_is_passive_and_shared_module_is_isolated():
    baseline = (original.OWN_NAME, original.PROTOCOL, original.RUN, original.own_meta,
                original.checked_stage, original.load_module)
    spec = importlib.util.spec_from_file_location("exp155_second_passive_import", m.TOOL)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    assert other._shared is not m._shared and m._shared is not original
    assert baseline == (original.OWN_NAME, original.PROTOCOL, original.RUN, original.own_meta,
                        original.checked_stage, original.load_module)
    assert m._shared.serving_source.__code__.co_code == original.serving_source.__code__.co_code
    assert m._shared.public3.__code__.co_code == original.public3.__code__.co_code
    assert m._shared.compare_parity.__code__.co_code == original.compare_parity.__code__.co_code
    assert m.sha(m.BASE_PATH) == m.BASE_SHA256
    assert not torch.cuda.__dict__.get("_initialized", False)


def test_cli_defaults_point_only_to_exp155():
    args = m.parser().parse_args(["average", "--out", "unused"])
    assert Path(args.protocol) == m.PROTOCOL
    assert "EXP155" in args.protocol and "S42v1" in m.RUN
    stage = m.parser().parse_args(["stage", "--average", "unused", "--out", "unused"])
    assert stage.dataset == m.ASSET_DATASET


def test_guard_redirect_does_not_rewrite_other_imports(monkeypatch):
    seen = []
    monkeypatch.setattr(m, "_original_load_module", lambda name, path: seen.append((name, path)))
    m.load_module("guard", ROOT / "tools/exp143_v2.py")
    m.load_module("training", ROOT / "training/src/exp143_own0949.py")
    assert seen[0][1] == ROOT / "tools/exp155_own0949_oai.py"
    assert seen[1][1] == ROOT / "training/src/exp143_own0949.py"


def test_actual_guard_exposes_public3_series_interface(tmp_path):
    from tools import exp155_own0949_oai as guard
    assert guard.series_records is guard.base.series_records
    path = tmp_path / "fake_test_series.csv"
    path.write_text("StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive\n"
                    "fake_study,fake_series,Sagittal,1\n", encoding="utf-8")
    records = guard.series_records(path)
    assert list(records) == ["fake_study"]
    assert records["fake_study"][0]["SeriesInstanceUID"] == "fake_series"


@pytest.mark.parametrize("key,value", [("oai", False), ("arch", "wrong"), ("seed", 43),
                                      ("swa_epochs_1based", [11, 12, 14]), ("contract", "r64")])
def test_checkpoint_metadata_never_forges_oai_free(key, value):
    ck = checkpoint()
    m.own_meta(ck, "a" * 64)
    ck[key] = value
    with pytest.raises(ValueError, match="metadata"):
        m.own_meta(ck, "a" * 64)
    with pytest.raises(ValueError):
        original.own_meta(checkpoint(), "a" * 64, run_dir=m.RUN)


@pytest.mark.parametrize("field", ["train_ids", "oai_labels", "participant_id"])
def test_checkpoint_private_data_is_rejected(field):
    ck = checkpoint()
    ck["provenance"][field] = ["OAI_FAKE_ID"]
    with pytest.raises(ValueError, match="provenance"):
        m.assert_private_checkpoint(ck)
    ck = checkpoint()
    ck[field] = ["OAI_FAKE_ID"]
    with pytest.raises(ValueError, match="payload"):
        m.assert_private_checkpoint(ck)


def test_nonidentifying_training_receipt_allowed():
    ck = checkpoint()
    ck["provenance"].update(initial_state_sha256="d" * 64, seed=42, world_size=1,
        precision="bf16", contract=m.CONTRACT, train_n=7, torch="2.8.0",
        execution={"lane": "local", "batch": 1, "accumulation": 16, "effective_batch": 16,
                   "grad_checkpointing": False, "microbatches_per_epoch": 7, "updates_per_epoch": 1,
                   "total_steps": 16, "max_hours": 20., "reserve_seconds": 600})
    ck["provenance"].update(rsna_train_n=5, oai_train_n=2)
    ck.update(epoch=15, tag="S42v1", amp_train="bf16", amp_eval="fp16", fold=-1,
              label_set="l2-dread75-aquila25-plus-oai-three-target")
    m.assert_private_checkpoint(ck)
    ck["provenance"]["execution"]["ids"] = ["OAI_FAKE_ID"]
    with pytest.raises(ValueError, match="execution"):
        m.assert_private_checkpoint(ck)


def test_notebook_reuses_runner_with_two_authenticated_modules():
    for variant in m.KERNELS:
        nb, title, description = m.notebook(variant, m.ASSET_DATASET, "a"*64, "b"*64, "c"*64, [])
        code = nb["cells"][-1]["source"]
        compile(code, "exp155-fake-notebook", "exec")
        assert "import exp155_own0949_oai_release as _e152" in code
        assert "'exp155_own0949_oai_release.py', 'exp152_own0949_release.py', 'public-serving.py'" in code
        assert "exp155-assets.zip.bin" in code and "exp152-assets.zip.bin" not in code
        assert "_e152.bind(globals()" in code and "main()" in code
        assert "OAI-derived" in description and "OAI-free" not in title + description
        metadata = m.kernel_metadata(variant)
        assert metadata["is_private"] and not metadata["enable_internet"]
        assert metadata["title"] == m.KERNELS[variant]
        assert metadata["dataset_sources"][0] == m.ASSET_DATASET


def test_staged_archive_contains_no_private_inputs_and_detects_drift(tmp_path, monkeypatch):
    dest = fake_stage(tmp_path, monkeypatch)
    _, report, manifest = m.checked_stage(dest)
    assert report["private_required"] and manifest["checkpoints"][m.OWN_NAME]["oai"] is True
    assert "source_bindings" not in manifest and "bn" not in manifest
    with zipfile.ZipFile(dest / "dataset/exp155-assets.zip.bin") as archive:
        assert set(archive.namelist()) == {m.OWN_NAME, m.TOOL.name, m.BASE_NAME,
                                           "public-serving.py", "release-manifest.json"}
        assert b"OAI_FAKE_DO_NOT_EXPORT" not in archive.read("release-manifest.json")
    assert {p.name for p in (dest / "dataset").iterdir()} == {"exp155-assets.zip.bin", "dataset-metadata.json"}
    with pytest.raises(ValueError, match="fresh"):
        m.fresh_dir(dest)
    (dest / "assets" / m.BASE_NAME).write_text("# changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        m.checked_stage(dest)


def test_packaged_adapter_imports_without_repository(tmp_path, monkeypatch):
    dest = fake_stage(tmp_path, monkeypatch)
    path = dest / "assets" / m.TOOL.name
    spec = importlib.util.spec_from_file_location("exp155_packaged_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.verify_assets(dest / "assets", m.read_json(dest / "assets/release-manifest.json"))
    runtime.own_meta(checkpoint(), "a" * 64)
    assert runtime._shared.OWN_NAME == m.OWN_NAME
    assert runtime.BASE_PATH.parent == path.parent


def test_parity_and_upload_plans_remain_local_only(tmp_path, monkeypatch):
    dest = fake_stage(tmp_path, monkeypatch)
    test = tmp_path / "test.csv"
    test.write_text("StudyInstanceUID\na\nb\nc\n")
    series = tmp_path / "test_series.csv"
    series.write_text("StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive\n")
    parity = tmp_path / "parity"
    m.parity_plan(SimpleNamespace(stage=dest, out=parity, test_csv=test, series_csv=series))
    plan = m.read_json(parity / "parity-plan.json")
    assert plan["tolerance"] == original.TOL
    assert plan["checkpoint_sha256"][m.OWN_NAME]["oai"] is True
    push = tmp_path / "push-plan.ps1"
    result = m.push_plan(SimpleNamespace(stage=dest, out=push))
    assert result["status"] == "PLAN_ONLY_NOT_EXECUTED" and result["sha256"] == m.sha(push)
    text = push.read_text()
    assert "--public" not in text and "exp152-" not in text
    assert all(line.startswith("#") for line in text.splitlines() if "kaggle_submit_once.py" in line)
    for slug in m.KERNELS.values():
        assert slug in text


@pytest.mark.parametrize("wrong_bn_ids", [False, True])
def test_average_authenticates_mixed_bn_and_preserves_oai(tmp_path, monkeypatch, wrong_bn_ids):
    model = torch.nn.Sequential(OrderedDict(linear=torch.nn.Linear(2, 2), bn=torch.nn.BatchNorm1d(2)))
    run = tmp_path / "run"
    run.mkdir()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    states, epoch_hashes = [], {}
    for epoch in m.EPOCHS:
        state = OrderedDict((key, value.clone()) for key, value in model.state_dict().items())
        for value in state.values():
            value.fill_(epoch)
        states.append(state)
        path = run / f"epoch_{epoch:02d}.pt"
        torch.save({"model": state, "epoch": epoch}, path)
        epoch_hashes[str(epoch)] = m.sha(path)
    final = m.average_states(states)
    final["bn.running_mean"].fill_(3.)
    final["bn.running_var"].fill_(4.)
    final["bn.num_batches_tracked"].fill_(2)
    ck = checkpoint(final)
    ck["provenance"]["protocol_sha256"] = m.sha(protocol_path)
    torch.save(ck, run / "model_best.pt")
    ids = ["rsna_fake_1", "rsna_fake_2", "oai_fake_1"]
    bn_ids = ids[:2] if wrong_bn_ids else ids
    bn = dict(studies=3, unique_studies=3, batches=2, num_batches_tracked=[2], mode="train",
              gradients="none", bn_modules=1,
              studies_sha256=hashlib.sha256("\n".join(sorted(bn_ids)).encode()).hexdigest())
    m.write_json(run / "summary.json", dict(status="COMPLETE_TRAINING_NO_EVALUATION", epochs=16,
                 model_sha256=m.sha(run / "model_best.pt"), identity=ck["provenance"],
                 bn=bn, swa_member_sha256=epoch_hashes))
    protocol = dict(recipe={"swa_1based": m.EPOCHS, "bn_batch": 2}, run_dirs={"local": m.RUN},
                    train_ids=ids, source_bindings={"fake_source.py": "a"*64})
    guard = SimpleNamespace(load_registration=lambda path: protocol,
                            registered_run=lambda p, lane: run)
    train = SimpleNamespace(build_model=lambda res: model)
    monkeypatch.setattr(m, "load_module", lambda name, path: guard if "guard" in name else train)
    args = SimpleNamespace(protocol=protocol_path, checkpoint=None, out=tmp_path / "average")
    if wrong_bn_ids:
        with pytest.raises(ValueError, match="BN replay cohort"):
            m.average(args)
        assert not args.out.exists()
    else:
        assert m.average(args)["status"] == "VERIFIED_FIXED_SWA_BN"
        saved = m.safe_load(args.out / m.OWN_NAME)
        assert saved["oai"] is True and saved["provenance"] == ck["provenance"]
        assert all(torch.equal(saved["model"][key], value) for key, value in final.items())
        receipt = m.read_json(args.out / "average-receipt.json")
        assert receipt["oai"] is True and receipt["bn"]["studies"] == 3
        assert "source_bindings" not in receipt


def test_serving_cost_is_weight_independent_projection():
    projection = m.projection()
    assert projection["own_seconds"] == pytest.approx(2156.)
    assert projection["blend_seconds"] == pytest.approx(4192.)
    assert "unmeasured" in projection["caveat"]
