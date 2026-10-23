"""CPU-only EXP174 contract tests. Synthetic labels; never print source rows.
Changelog v1.1.0 (2026-10-12): accept valid long timing; preserve failure/scientific checks.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp174_own0949_jev as g


@pytest.fixture(autouse=True)
def no_gpu_or_subprocess(monkeypatch):
    import torch
    assert not torch.cuda.is_initialized()
    def fail(*args, **kwargs):
        pytest.fail("CPU contract test attempted CUDA/subprocess")
    for name in ("_lazy_init", "init", "is_available", "get_device_name", "set_device",
                 "manual_seed_all", "is_bf16_supported", "mem_get_info", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, fail)
    monkeypatch.setattr(g.subprocess, "run", fail)
    yield
    assert not torch.cuda.is_initialized()


def fake_control():
    ids = [f"fake-{i}" for i in range(4347)]
    return dict(schema=g.base.SCHEMA, status="REGISTERED_NO_TRAINING", created="fake",
        recipe=dict(g.base.RECIPE, labels="registered-replacement-soft"),
        budgets=dict(local=14.0, kaggle=9.0, reserve_seconds=600),
        run_dirs=dict(g.base.RUN_DIRS), train_ids=ids, excluded_gold_ids=[f"gold-{i}" for i in range(58)],
        train_labels={u: [.4] * 12 for u in ids}, pos_weight=[1.5] * 12,
        label_sha256="old", label_provenance={"oai": False}, label_provenance_sha256="old",
        label_file_name="l2_dread75_aquila25_v1.csv", init_sha256=g.INIT_SHA,
        init_provenance=dict(oai=False, knee_finetuning=False),
        corpus_manifest_sha256="corpus", corpus_build_sha256="build", public_gold_sha256="gold",
        source_bindings={}, evaluation=dict(gold_n=58, target_n=12), content_sha256="fake")


def swapped(control):
    p = copy.deepcopy(control)
    p.update(schema=g.SCHEMA, version=g.VERSION, recipe=copy.deepcopy(g.RECIPE), run_dirs=dict(g.RUN_DIRS),
        base_protocol_sha256=g.BASE_SHA, base_provenance_sha256="provenance",
        expected_initial_state_sha256=g.INITIAL_STATE_SHA,
        probe_contract=dict(measured_updates=100, warmup_updates=2, grad_checkpointing=False, max_seconds=1800))
    return p


def test_recipe_and_execution_diff_against_actual_exp143_v2_protocol():
    # Read only existing structured protocol/provenance, no raw competition CSV.
    control = g.read_json(g.BASE_PROTOCOL)
    p = swapped(control)
    g.assert_label_only_diff(p, control)
    changed = [k for k in control["recipe"] if control["recipe"][k] != p["recipe"][k]]
    assert changed == ["labels"]
    assert g.execution_settings(p, 1, "bf16", 42, False) == g.read_json(g.BASE_PROVENANCE)["execution"]
    assert len(control["train_ids"]) == 4347 and len(control["excluded_gold_ids"]) == 58
    assert g.sha256(g.BASE_PROTOCOL) == g.BASE_SHA


@pytest.mark.parametrize("key,value", [("bb_lr", .01), ("epochs", 15), ("oai", True),
    ("swa_1based", [11, 12, 14]), ("batch", 2), ("train_k", 10), ("res", 320)])
def test_scientific_changes_are_rejected(key, value):
    control = fake_control()
    p = swapped(control)
    p["recipe"][key] = value
    with pytest.raises(ValueError, match="recipe"):
        g.assert_label_only_diff(p, control)


@pytest.mark.parametrize("key,value", [("train_ids", []), ("excluded_gold_ids", []),
    ("init_sha256", "other"), ("corpus_manifest_sha256", "other"),
    ("budgets", dict(local=7, kaggle=9, reserve_seconds=600))])
def test_nonlabel_input_drift_is_rejected(key, value):
    control = fake_control()
    p = swapped(control)
    p[key] = value
    with pytest.raises(ValueError, match="contract changed"):
        g.assert_label_only_diff(p, control)


def test_local_schedule_fixed_and_no_mode_search():
    p = swapped(fake_control())
    s = g.execution_settings(p, 1, "bf16", 42)
    assert s["total_steps"] == 4352 and s["updates_per_epoch"] == 272
    assert s["accumulation"] == 16 and s["grad_checkpointing"] is False
    assert s["max_hours"] == 14
    for args in ((1, "bf16", 42, True), (2, "fp16", 43, False),
                 (1, "fp16", 42, False), (1, "bf16", 43, False), (1, "bf16", 42, 0)):
        with pytest.raises(ValueError):
            g.execution_settings(p, *args)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "ROOT", tmp_path)
    directory = tmp_path / "artifacts/experiments/EXP174_OWN0949_JEV"
    monkeypatch.setattr(g, "BASE", directory)
    monkeypatch.setattr(g, "PROTOCOL", directory / "protocol-v1/manifest.json")
    monkeypatch.setattr(g, "PROBE_ROOT", tmp_path / ".local/exp174-probes")
    monkeypatch.setattr(g, "PROBE_OUT", g.PROBE_ROOT / "probe-s42")
    record_file = tmp_path / "label-provenance.json"
    record_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(g, "LABEL_PROVENANCE", record_file)
    monkeypatch.setattr(g, "BASE_PROVENANCE", record_file)
    control = fake_control()
    labels = {u: [.25] * 12 for u in control["train_ids"]}
    record = {"oai": False, "rows": 4347}
    monkeypatch.setattr(g, "baseline_registration", lambda *a: control)
    monkeypatch.setattr(g, "label_table", lambda p: (labels, record))
    monkeypatch.setattr(g, "source_bindings", lambda: {"fake.py": "fake-sha"})
    monkeypatch.setattr(g.base, "pos_weights", lambda labels, ids: [3.] * 12)
    args = g.parser().parse_args(["prepare"])
    g.prepare(args)
    v = dict(status="VALIDATED_NO_TRAINING", protocol_sha256=g.sha256(g.PROTOCOL),
        initial_state_sha256=g.INITIAL_STATE_SHA, actual_pin_memory_tuple_returns_list=True,
        full_corpus_hashes=True, real_backbone_keys_loaded_cpu=True, label_formula_verified=True,
        label_only_diff_verified=True)
    g.write_new_json(g.validation_path(g.PROTOCOL), v)
    return SimpleNamespace(control=control, args=args, labels=labels)


def valid_report(p, protocol):
    settings = g.execution_settings(p, 1, "bf16", 42)
    rate, setup = 3.2, 30.0
    r = dict(protocol_sha256=g.sha256(protocol), mode="off", status="MEASURED_SCRATCH_ONLY",
        admissible=True, precision="bf16", grad_checkpointing=False, execution=settings,
        initial_state_sha256=g.INITIAL_STATE_SHA, torch=g.TORCH_VERSION,
        device_name="NVIDIA GeForce RTX 4070 Ti SUPER", skipped_updates=0, timed_updates=100,
        steps_requested=100, warmup_updates=2, update_seconds=[rate] * 100, setup_seconds=setup,
        seconds_per_update=sum([rate] * 100) / 100)
    r.update(g.projection(settings, r["seconds_per_update"], setup))
    return dict(status="SCRATCH_PROBE_ONLY_NO_FIT", protocol_sha256=g.sha256(protocol),
                results=[r], steps_per_mode=100, warmup_updates_per_mode=2)


def test_prepare_roundtrip_and_create_only(prepared):
    p = g.load_registration(g.PROTOCOL)
    g.assert_label_only_diff(p, prepared.control)
    assert p["train_labels"] == prepared.labels
    assert p["pos_weight"] == [3.] * 12
    assert p["source_bindings"] == {"fake.py": "fake-sha"}
    with pytest.raises(FileExistsError):
        g.prepare(prepared.args)


def test_source_drift_and_content_drift_fail(prepared, monkeypatch):
    monkeypatch.setattr(g, "source_bindings", lambda: {"fake.py": "changed"})
    with pytest.raises(ValueError, match="binding changed"):
        g.load_registration(g.PROTOCOL)
    p = g.read_json(g.PROTOCOL)
    p["train_ids"] = []
    g.PROTOCOL.write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        g.load_registration(g.PROTOCOL)


def test_probe_and_full_queue_shape_and_order(prepared):
    g.queue_json(g.parser().parse_args(["probe-queue"]))
    probe = g.read_json(g.PROTOCOL.with_name("probe-queue-s42.json"))
    assert len(probe) == 1 and set(probe[0]) == {"tag", "after", "continue_on_failure", "command"}
    assert probe[0]["tag"] == g.PROBE_TAG and probe[0]["after"] == g.PREDECESSOR
    assert probe[0]["continue_on_failure"] is False
    assert probe[0]["command"][1:4] == ["-m", "tools.exp174_own0949_jev", "probe"]
    with pytest.raises(FileNotFoundError):
        g.queue_json(g.parser().parse_args(["full-queue"]))
    p = g.load_registration(g.PROTOCOL)
    g.write_new_json(g.PROBE_OUT / "report.json", valid_report(p, g.PROTOCOL))
    g.queue_json(g.parser().parse_args(["full-queue"]))
    job = g.read_json(g.PROTOCOL.with_name("queue-s42.json"))[0]
    assert job["tag"] == g.FULL_TAG and job["after"] == g.PROBE_TAG
    assert job["continue_on_failure"] is False
    assert "--no-grad-checkpointing" in job["command"]
    assert "--oai-corpus" not in job["command"] and "--max-hours" not in job["command"]
    assert not g.registered_run(p).exists()


@pytest.mark.parametrize("key,value", [("initial_state_sha256", "wrong"), ("timed_updates", 99),
    ("skipped_updates", 1), ("admissible", False), ("grad_checkpointing", True),
    ("mode", "on"), ("torch", "different"), ("device_name", "Tesla T4"),
    ("seconds_per_update", 999), ("setup_seconds", float("nan"))])
def test_probe_admission_fails_closed(prepared, key, value):
    p = g.load_registration(g.PROTOCOL)
    report = valid_report(p, g.PROTOCOL)
    report["results"][0][key] = value
    with pytest.raises(ValueError):
        g.checked_probe(report, p, g.PROTOCOL)


@pytest.mark.parametrize("admissible", [True, False])
def test_probe_accepts_long_valid_timing(prepared, admissible):
    p = g.load_registration(g.PROTOCOL)
    report = valid_report(p, g.PROTOCOL)
    r = report["results"][0]
    r.update(seconds_per_update=12.0, update_seconds=[12.0] * 100)
    r.update(g.projection(r["execution"], 12.0, r["setup_seconds"]))
    r["admissible"] = admissible
    assert r["fits_budget"] is False
    g.checked_probe(report, p, g.PROTOCOL)


def test_output_paths_protect_frozen_experiments(tmp_path):
    for path in (ROOT / "artifacts/experiments/EXP143_OWN0949/S42v2/new.json",
                 ROOT / "artifacts/experiments/EXP168_OWN0949_OAI_JEV/new.json", tmp_path / "bad.json"):
        with pytest.raises(ValueError):
            g.write_new_json(path, {})
    with pytest.raises(ValueError):
        g.scratch_directory(ROOT / ".local/exp143-v2-probes/other")


def test_guard_shim_is_scoped_and_restored_even_on_failure():
    import tools
    previous = tools.exp143_v2
    with pytest.raises(RuntimeError):
        with g.guard_shim():
            from tools import exp143_v2 as shim
            assert shim is g
            raise RuntimeError("fake failure")
    assert tools.exp143_v2 is previous
    assert sys.modules["tools.exp143_v2"] is previous


def test_exact_frozen_trainer_functions_are_reused():
    import exp143_v2 as frozen
    fit = g.isolated_fit()
    # Code identity includes bytecode/constants/names: no altered optimizer/fit.
    for name in ("train", "probe_mode", "make_optimization", "accumulation_group", "average_registered_epochs"):
        a, b = getattr(fit, name).__code__, getattr(frozen, name).__code__
        assert a.co_code == b.co_code and a.co_consts == b.co_consts and a.co_names == b.co_names
    assert fit.build_model is frozen.build_model and fit.StudyDataset is frozen.StudyDataset
    assert fit.scratch_directory is g.scratch_directory


def test_gpu_job_receipt_uses_supervisor_parent_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "ROOT", tmp_path)
    monkeypatch.setattr(g.os, "getppid", lambda: 123)
    path = tmp_path / "artifacts/logs" / g.PROBE_TAG / "job.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(pid=123, tag=g.PROBE_TAG)), encoding="utf-8")
    g.require_gpu_job(g.PROBE_TAG)
    monkeypatch.setattr(g.os, "getppid", lambda: 456)
    with pytest.raises(ValueError, match="gpu_queue/gpu_job"):
        g.require_gpu_job(g.PROBE_TAG)


def test_predecessor_waits_for_successful_final_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "ROOT", tmp_path)
    logs = tmp_path / "artifacts/logs" / g.PREDECESSOR
    logs.mkdir(parents=True)
    assert g.predecessor_status() is False
    (logs / "SUCCESS").write_text("ok")
    assert g.predecessor_status() is False
    local = tmp_path / ".local"
    local.mkdir()
    (local / "exp168-full-queue.log").write_text(g.PREDECESSOR + ": SUCCESS\nplan finished\n")
    exp = tmp_path / "artifacts/experiments/EXP168_OWN0949_OAI_JEV"
    (exp / "protocol-v1").mkdir(parents=True)
    (exp / "protocol-v1/queue-s42.json").write_text(json.dumps([{"tag": g.PREDECESSOR}]))
    (exp / "S42").mkdir()
    (exp / "S42/model_best.pt").write_bytes(b"fake")
    (exp / "S42/summary.json").write_text(json.dumps(dict(status="COMPLETE_TRAINING_NO_EVALUATION",
        epochs=16, model_sha256=g.sha256(exp / "S42/model_best.pt"))))
    assert g.predecessor_status() is True  # no nonexistent autochain log dependency
    (logs / "FAILED").write_text("failed")
    with pytest.raises(RuntimeError, match="chain failed"):
        g.predecessor_status()


def test_fakes_reuse_existing_cpu_validation():
    import exp143_v2 as fit
    assert fit.tiny_validation()["cuda_calls"] == 0


def test_label_reuse_checks_formula_and_source_pins(monkeypatch, tmp_path):
    from tools import exp168_labels as lb
    control = fake_control()
    ids = control["train_ids"]
    monkeypatch.setattr(g, "ROOT", tmp_path)
    monkeypatch.setattr(lb, "DREAD", tmp_path / "dread.csv")
    monkeypatch.setattr(lb, "JEV", tmp_path / "jev.csv")
    expected = [float(format(float(v), ".9g")) for v in lb.mix([.2] * 12, [1.] * 12)]
    record = dict(name="l2_dread75_jev25_v1", rows=4347, targets=g.TARGETS, oai=False,
        gold58_included=False, csv_sha256=g.LABEL_SHA, baseline_protocol_sha256=g.BASE_SHA,
        baseline_train_ids_sha256=g.hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        source_sha256={"dread.csv": lb.DREAD_SHA, "jev.csv": lb.JEV_SHA})
    monkeypatch.setattr(g, "read_json", lambda p: record)
    hashes = {g.LABELS: g.LABEL_SHA, lb.DREAD: lb.DREAD_SHA, lb.JEV: lb.JEV_SHA}
    monkeypatch.setattr(g, "sha256", lambda p: hashes[Path(p)])
    table = {uid: list(expected) for uid in ids}
    monkeypatch.setattr(g.base, "read_label_table", lambda p: table)
    monkeypatch.setattr(lb, "read_source", lambda p, ids, binary=False: {u: [1. if binary else .2] * 12 for u in ids})
    checked, _ = g.label_table(control)
    assert len(checked) == 4347
    table[ids[0]][0] = .5
    with pytest.raises(ValueError, match="formula"):
        g.label_table(control)
