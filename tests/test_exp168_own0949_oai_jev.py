"""CPU-only contract checks for the EXP168 label substitution and adapter."""
import csv
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from tools import exp143_v2 as base
from tools import exp168_labels as labels
from tools import exp168_own0949_oai_jev as exp168
from tools import exp168_own0949_oai_jev_release as release
import exp168_own0949_oai_jev as fit


@pytest.fixture(autouse=True)
def cpu_offline(monkeypatch):
    assert not torch.cuda.is_initialized()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("EXP168 unit test touched CUDA or network")
    for name in ("_lazy_init", "init", "manual_seed_all", "is_available", "set_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    yield
    assert not torch.cuda.is_initialized()


def _table(path, rows, soft=False):
    columns = [t + "_soft" for t in base.TARGETS] if soft else base.TARGETS
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([base.UID, *columns])
        writer.writerows(rows)


def test_jev_source_is_exact_hybrid_evidence_source():
    assert labels.JEV.relative_to(labels.ROOT).as_posix() == (
        "storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv")
    assert exp168.RECIPE == dict(exp168.prior.RECIPE,
        labels="l2-dread75-jev25-plus-oai-three-target-b1")
    assert exp168.OAI_LABELS == exp168.prior.OAI_LABELS
    assert exp168.RUN_DIRS != exp168.prior.RUN_DIRS


def test_source_join_and_binary_gate(tmp_path):
    path = tmp_path / "source.csv"
    _table(path, [["u2", *([1] * 12)], ["u1", *([0] * 12)]], soft=True)
    actual = labels.read_source(path, ["u1", "u2"], binary=True)
    assert set(actual) == {"u1", "u2"}
    assert actual["u1"].sum() == 0
    with pytest.raises(ValueError, match="cover"):
        labels.read_source(path, ["u1", "u3"], binary=True)
    _table(path, [["u1", *([.5] * 12)]])
    with pytest.raises(ValueError, match="binary"):
        labels.read_source(path, ["u1"], binary=True)


def test_float32_soft_mix_and_invalid_jev():
    dread = np.full(12, .2, np.float32)
    jev = np.ones(12, np.float32)
    mixed = labels.mix(dread, jev)
    np.testing.assert_array_equal(mixed, (np.float32(.75) * dread + np.float32(.25) * jev))
    assert mixed.dtype == np.float32
    with pytest.raises(ValueError, match="invalid"):
        labels.mix(dread, np.full(12, .5))


def test_trainer_reuses_exp155_without_mutating_it():
    assert fit.train.__code__ is fit.prior.train.__code__
    assert fit.probe_mode.__code__ is fit.prior.probe_mode.__code__
    assert fit.train.__globals__["__builtins__"]["__import__"] is fit._import_guard
    assert fit.prior.train.__globals__["__builtins__"] is not fit.train.__globals__["__builtins__"]


def test_release_equal_trio_and_metadata():
    rng = np.random.default_rng(42)
    arms = [rng.uniform(size=(7, 12)) for _ in range(3)]
    output = release.output_values(arms, "blend3", lambda a: a)
    expected = np.mean([np.asarray([[release.percentile_average(a[:, j].tolist())[i]
                                      for j in range(12)] for i in range(7)]) for a in arms], axis=0)
    np.testing.assert_allclose(output, expected)
    assert release.OWN_NAME.startswith("exp168-")
    assert release.RUN == exp168.RUN_DIRS["local"]


def test_release_rejects_old_label_tag():
    ck = {"provenance": {"execution": {}}, "label_set": "l2-dread75-aquila25-plus-oai-three-target-b1"}
    with pytest.raises(ValueError):
        release.assert_private_checkpoint(ck)


def test_release_notebooks_are_private_and_bind_new_adapter():
    for variant in ("own", "blend3"):
        notebook, _, description = release.notebook(
            variant, release.ASSET_DATASET, "a" * 64, "b" * 64, "c" * 64, [])
        code = notebook["cells"][-1]["source"]
        compile(code, "exp168-release-notebook", "exec")
        assert "import exp168_own0949_oai_jev_release as _e152" in code
        assert release.MEMBER_NAME in code
        if variant == "blend3":
            assert "'blend3'" in code and "1/3" in description
        metadata = release.kernel_metadata(variant)
        assert metadata["is_private"] and not metadata["enable_internet"]


def test_release_stage_contains_only_inference_assets(tmp_path, monkeypatch):
    average = tmp_path / "average"
    average.mkdir()
    state = {"x": torch.tensor([1.])}
    ck = dict(model=state, arch=release.ARCH, res=384, inference_res=320,
              lab=release.TARGETS, contract=release.CONTRACT, k_eval=94,
              norm="imagenet", window_policy="legacy", oai=True,
              checkpoint_kind="fixed-swa-bn", swa_epochs_1based=release.EPOCHS,
              seed=42, provenance={"protocol_sha256": "a" * 64, "run_dir": release.RUN},
              label_set=exp168.RECIPE["labels"], tag="S42")
    torch.save(ck, average / release.OWN_NAME)
    release.write_json(average / "average-receipt.json", dict(
        schema="exp168-average-v1", oai=True, status="VERIFIED_FIXED_SWA_BN",
        epochs_1based=release.EPOCHS,
        checkpoint_sha256=release.sha(average / release.OWN_NAME),
        protocol_sha256="a" * 64, state_schema=release.state_schema(state),
        source_final_sha256="b" * 64, source_bindings_sha256="c" * 64,
        shared_serving_sha256=release.BASE_SHA256))
    member_dir = tmp_path / "member"
    member_dir.mkdir()
    member = member_dir / release.MEMBER_NAME
    member_protocol = release.sha(release.ROOT /
        "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json")
    member_ck = dict(ck, oai=False,
        provenance={"protocol_sha256": member_protocol, "train_ids": ["FAKE_NOT_FOR_EXPORT"]})
    torch.save(member_ck, member)
    release.write_json(member_dir / "average-receipt.json", dict(
        status="VERIFIED_FIXED_SWA_BN", checkpoint_sha256=release.sha(member),
        protocol_sha256=member_protocol, epochs_1based=release.EPOCHS))
    monkeypatch.setattr(release, "public_source", lambda: ("# fake public source\n", []))
    monkeypatch.setattr(release, "serving_source", lambda source: source)
    staged = tmp_path / "stage"
    release.stage(SimpleNamespace(average=average, exp143_checkpoint=member,
                                  out=staged, dataset=release.ASSET_DATASET))
    _, report, manifest = release.checked_stage(staged)
    assert report["status"] == "STAGED_NOT_UPLOADED"
    assert set(report["kernels"]) == {"own", "blend", "blend3"}
    assert manifest["schema"] == "exp168-release-v1"
    with zipfile.ZipFile(staged / "dataset/exp168-assets.zip.bin") as archive:
        assert release.TOOL.name in archive.namelist()
        assert release.MEMBER_NAME in archive.namelist()
        assert b"FAKE_NOT_FOR_EXPORT" not in archive.read("release-manifest.json")


def test_queue_artifacts_chain_after_probe_without_launch(tmp_path, monkeypatch):
    experiment = tmp_path / "EXP168"
    experiment.mkdir()
    monkeypatch.setattr(exp168, "BASE", experiment)
    monkeypatch.setattr(exp168, "ROOT", tmp_path)
    monkeypatch.setattr(exp168, "load_registration", lambda *_args: {"run_dirs": exp168.RUN_DIRS})
    monkeypatch.setattr(exp168, "require_validation", lambda *_args: None)
    settings = {"lane": "local", "batch": 1, "accumulation": 16}
    monkeypatch.setattr(exp168, "execution_settings", lambda *_args: settings)
    monkeypatch.setattr(exp168, "registered_run", lambda *_args: tmp_path / "unused-run")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    common = dict(protocol=protocol, corpus=tmp_path, oai_corpus=tmp_path,
                  init_weights=tmp_path, python="python", workers=2)
    probe_file = experiment / "probe.json"
    exp168.probe_queue_json(SimpleNamespace(**common, after="EXP163_OWN0949_OAI_EXT_S42",
                                            steps=20, out=probe_file))
    probe_job = json.loads(probe_file.read_text(encoding="utf-8"))[0]
    assert probe_job["after"] == "EXP163_OWN0949_OAI_EXT_S42"
    assert probe_job["tag"] == "EXP168_OAI_JEV_PROBE_S42"
    assert probe_job["command"][1:4] == ["-m", "tools.exp168_own0949_oai_jev", "probe"]
    report = tmp_path / "report.json"
    report.write_text(json.dumps(dict(protocol_sha256=exp168.sha256(protocol),
        results=[dict(mode="off", admissible=True, status="MEASURED_SCRATCH_ONLY",
                      precision="bf16", skipped_updates=0, execution=settings)])), encoding="utf-8")
    queue_file = experiment / "queue.json"
    exp168.queue_json(SimpleNamespace(**common, after=probe_job["tag"],
                                      probe_report=report, out=queue_file))
    train_job = json.loads(queue_file.read_text(encoding="utf-8"))[0]
    assert train_job["after"] == probe_job["tag"]
    assert train_job["command"][1:4] == ["-m", "tools.exp168_own0949_oai_jev", "train"]
    assert train_job["continue_on_failure"] is False
