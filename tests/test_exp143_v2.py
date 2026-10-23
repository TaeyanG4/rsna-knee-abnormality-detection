"""CPU-only v2 contracts: tiny synthetic tables/models; never query/initialize CUDA."""
from __future__ import annotations

import copy
import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/src"))
from tools import exp143_v2 as guard
import exp143_v2 as fit
import exp143_own0949 as old_fit


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()

    def fail(*args, **kwargs):
        raise AssertionError("CPU tests must not touch CUDA")

    for name in ("_lazy_init", "init", "manual_seed_all", "is_available", "set_device",
                 "is_bf16_supported", "synchronize", "mem_get_info", "get_device_name",
                 "max_memory_allocated", "max_memory_reserved", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, fail)
    yield
    assert not torch.cuda.is_initialized()


def bare_registration():
    return dict(recipe=copy.deepcopy(guard.RECIPE), train_ids=list(range(4347)),
                budgets=dict(local=9., kaggle=9., reserve_seconds=600),
                run_dirs=copy.deepcopy(guard.RUN_DIRS))


def test_local_and_two_t4_have_same_global_batch_and_update_count():
    p = bare_registration()
    for world, precision, seed, accum, batches in ((1, "bf16", 42, 16, 4347), (2, "fp16", 43, 8, 2173)):
        s = guard.execution_settings(p, world, precision, seed)
        assert s["batch"] * world * s["accumulation"] == 16
        assert s["accumulation"] == accum and s["microbatches_per_epoch"] == batches
        assert s["updates_per_epoch"] == 272 and s["total_steps"] == 4352
    assert guard.execution_settings(p, 1, "bf16", 42, False)["grad_checkpointing"] is False
    assert guard.execution_settings(p, 1, "bf16", 42)["grad_checkpointing"] is True
    for args in ((2, "bf16", 43), (2, "fp16", 42), (3, "fp16", 43), (1, "none", 42)):
        with pytest.raises(ValueError):
            guard.execution_settings(p, *args)


def test_wall_projection_uses_registered_budget():
    p = bare_registration()
    p["budgets"]["local"] = 8.5
    s = guard.execution_settings(p, 1, "bf16", 42)
    assert guard.projection(s, 6, 60)["projected_seconds"] == 4352 * 6 + 60 + 600
    assert guard.projection(s, 6)["fits_budget"]
    assert not guard.projection(s, 7.5475)["fits_budget"]
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            guard.projection(s, value)


def test_optimizer_receives_published_rates_and_preserved_schedule(monkeypatch):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2)
    model.head = torch.nn.Linear(2, 1)
    captured = {}
    def adam(groups, **kwargs):
        captured["groups"] = [{"lr": g["lr"], "params": list(g["params"])} for g in groups]
        captured.update(kwargs)
        return "fake-optimizer"
    def schedule(optimizer, **kwargs):
        assert optimizer == "fake-optimizer"
        captured.update(kwargs)
        return "fake-schedule"
    monkeypatch.setattr(torch.optim, "AdamW", adam)
    monkeypatch.setattr(torch.optim.lr_scheduler, "OneCycleLR", schedule)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **kw: captured.update(scaler=kw))
    fit.make_optimization(model, guard.RECIPE, 4352, "bf16")
    assert [g["lr"] for g in captured["groups"]] == [4.01e-5, 1.95e-3]
    assert captured["weight_decay"] == .0588 and captured["pct_start"] == .15
    assert captured["max_lr"] == [4.01e-5, 1.95e-3] and captured["total_steps"] == 4352
    assert captured["scaler"]["enabled"] is False
    assert {id(v) for g in captured["groups"] for v in g["params"]} == {id(v) for v in model.parameters()}


def test_microbatch_gradient_matches_full_and_partial_group():
    # Real tiny CPU autograd: final short groups must not be underweighted.
    for n in (16, 11, 8, 5):
        x = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2) / 20
        w = torch.tensor([.2, -.1], requires_grad=True)
        expected = torch.autograd.grad((x @ w).square().mean(), w)[0]
        w.grad = None
        for i in range(n):
            final, count = fit.accumulation_group(i, n, 16)
            ((x[i] @ w).square() / count).backward()
            assert final == (i == n - 1)
        torch.testing.assert_close(w.grad, expected)
    assert fit.accumulation_group(4336, 4347, 16) == (False, 11)
    assert fit.accumulation_group(4346, 4347, 16) == (True, 11)
    assert fit.accumulation_group(2172, 2173, 8) == (True, 5)


def test_epoch_average_reads_protocol_members_only(tmp_path):
    for e in (11, 12, 14):
        torch.save({"model": {"weight": torch.tensor([float(e)])}}, tmp_path / f"epoch_{e:02d}.pt")
    averaged = fit.average_registered_epochs(tmp_path, guard.RECIPE)
    torch.testing.assert_close(averaged["weight"], torch.tensor([37 / 3]))
    with pytest.raises(FileNotFoundError):
        fit.average_registered_epochs(tmp_path, dict(guard.RECIPE, swa_1based=[13, 15, 16]))


def test_inherited_geometry_head_inference_and_tiny_forward_are_unchanged():
    for name in ("windows", "mirror_triplets", "infer_volume", "build_model", "collate", "unpack_batch"):
        assert inspect.getsource(getattr(fit, name)) == inspect.getsource(getattr(old_fit, name))
    assert fit.SPEC == old_fit.SPEC
    assert fit.tiny_validation()["cuda_calls"] == 0


def write_table(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(columns)
        w.writerows(rows)


def test_label_mapping_coverage_and_bad_values_fail_closed(tmp_path):
    file = tmp_path / "labels.csv"
    columns = [guard.UID, *reversed(guard.TARGETS)]
    values = [i / 12 for i in range(12)]
    write_table(file, columns, [["a", *reversed(values)], ["b", *reversed(values)]])
    labels = guard.read_label_table(file)
    assert labels["a"] == values
    assert guard.select_training_ids(["a", "b", "gold"], ["gold"], list(labels)) == ["a", "b"]
    with pytest.raises(ValueError, match="missing training labels"):
        guard.select_training_ids(["a", "b", "gold"], ["gold"], ["a"])
    invalid_rows = (["a", *([float("nan")] * 12)], ["a", *([1.1] * 12)],
                    ["a", *([.1] * 11)], ["a", *([.1] * 13)], ["", *([.1] * 12)])
    for row in invalid_rows:
        write_table(file, [guard.UID, *guard.TARGETS], [row])
        with pytest.raises(ValueError):
            guard.read_label_table(file)
    write_table(file, [guard.UID, *guard.TARGETS], [["a", *([.1] * 12)]] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        guard.read_label_table(file)
    write_table(file, [guard.UID, *guard.TARGETS[:-1], guard.TARGETS[0]], [])
    with pytest.raises(ValueError, match="columns"):
        guard.read_label_table(file)


@pytest.fixture
def registration_fixture(tmp_path, monkeypatch):
    # Cohort cardinality is real; every image, label, input and source is synthetic.
    base = tmp_path / "artifacts/experiments/EXP143_OWN0949"
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "BASE", base)
    monkeypatch.setattr(guard, "CODE", [])
    protocol_doc = tmp_path / guard.PROTOCOL
    protocol_doc.parent.mkdir(parents=True)
    protocol_doc.write_text("Amendment A2 synthetic test", encoding="utf-8")
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text("synthetic labels", encoding="utf-8")
    monkeypatch.setattr(guard, "LABELS", labels_file)
    monkeypatch.setattr(guard, "LABEL_SHA", guard.sha256(labels_file))
    lp = tmp_path / "provenance.json"
    lp.write_text(json.dumps(dict(source="synthetic", licence="test", oai=False)), encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for f in ("manifest.json", "build-registration.json"):
        (corpus / f).write_text("{}", encoding="utf-8")
    ids = [f"fake-{i}" for i in range(4405)]
    gold = ids[:58]
    labels = {u: [.25] * 12 for u in ids}
    monkeypatch.setattr(guard, "official_population", lambda _: (ids, gold, np.zeros((58, 12))))
    monkeypatch.setattr(guard, "read_label_table", lambda _: labels)
    monkeypatch.setattr(guard, "check_corpus", lambda *a, **kw: {"ids": ids})
    monkeypatch.setattr(guard, "pos_weights", lambda labels, ids: [3.] * 12)
    init = tmp_path / "storage/weights/exp143-imagenet-init/toy.pt"
    init.parent.mkdir(parents=True)
    init.write_bytes(b"fake ImageNet file, not a model")
    ip = init.with_suffix(".json")
    ip.write_text(json.dumps(dict(training_data="ImageNet-12k -> ImageNet-1k", oai=False,
                                 knee_finetuning=False, sha256=guard.sha256(init), source="synthetic", licence="test")))
    public = tmp_path / "fake-public.csv"
    public.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(guard, "PUBLIC_GOLD", public)
    args = guard.parser().parse_args(["prepare", "--init-weights", str(init),
        "--init-sha256", guard.sha256(init), "--init-provenance", str(ip), "--corpus", str(corpus)])
    return SimpleNamespace(args=args, labels=labels, gold=gold, init=init, corpus=corpus,
                           label_provenance=lp, base=base)


def test_prepare_roundtrip_binds_budget_labels_sources_and_paths(registration_fixture):
    f = registration_fixture
    f.args.max_hours = 8.75
    guard.prepare(f.args)
    p = guard.load_registration(f.args.protocol, f.corpus, f.init)
    assert p["budgets"]["local"] == 8.75
    assert p["recipe"]["swa_1based"] == [14, 11, 12]
    assert not set(f.gold) & set(p["train_labels"])
    assert len(p["train_labels"]) == 4347
    assert guard.registered_run(p).name == "S42v2"
    assert guard.registered_run(p, "kaggle").name == "S43v2"
    with pytest.raises(FileExistsError):
        guard.prepare(f.args)
    changed = copy.deepcopy(p)
    changed["budgets"]["local"] = 10
    Path(f.args.protocol).write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="registration content"):
        guard.load_registration(f.args.protocol)


def test_replacement_labels_register_and_coverage_fail_before_manifest(registration_fixture):
    f = registration_fixture
    alternate = f.base / "winner.csv"
    alternate.write_text("fake alternative labels", encoding="utf-8")
    f.args.labels = str(alternate)
    f.args.label_provenance = str(f.label_provenance)
    missing = next(u for u in f.labels if u not in f.gold)
    saved = f.labels.pop(missing)
    with pytest.raises(ValueError, match="missing training labels"):
        guard.prepare(f.args)
    assert not Path(f.args.protocol).exists()
    f.labels[missing] = saved
    guard.prepare(f.args)
    p = guard.load_registration(f.args.protocol)
    assert p["recipe"]["labels"] == "registered-replacement-soft"
    assert p["label_sha256"] == guard.sha256(alternate)


def test_replacement_labels_reject_oai_and_missing_provenance(registration_fixture):
    f = registration_fixture
    alternate = f.base / "winner.csv"
    alternate.write_text("fake alternative", encoding="utf-8")
    f.args.labels = str(alternate)
    with pytest.raises(ValueError, match="require --label-provenance"):
        guard.prepare(f.args)
    f.args.label_provenance = str(f.label_provenance)
    f.label_provenance.write_text(json.dumps(dict(oai=True, source="fake", licence="test")))
    with pytest.raises(ValueError, match="oai:false"):
        guard.prepare(f.args)


def test_old_run_protocol_and_scratch_paths_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "BASE", tmp_path)
    with pytest.raises(ValueError):
        guard.versioned_path(tmp_path / "protocol-v1/manifest.json", "protocol-v2")
    with pytest.raises(ValueError):
        guard.write_new_json(tmp_path / "S42/new-file.json", {})
    monkeypatch.setattr(fit, "ROOT", tmp_path)
    for name in (tmp_path / "S42v2", tmp_path / ".local/exp143-v2-probes", tmp_path / ".local/other"):
        with pytest.raises(ValueError):
            fit.scratch_directory(name)
    assert fit.scratch_directory(tmp_path / ".local/exp143-v2-probes/unique").name == "unique"


def test_checkpoint_contract_uses_manifest_set_and_run(tmp_path):
    protocol = tmp_path / "manifest.json"
    protocol.write_text("synthetic", encoding="utf-8")
    p = bare_registration()
    ck = dict(arch=p["recipe"]["arch"], contract=fit.CONTRACT, oai=False,
              lab=guard.TARGETS, checkpoint_kind="fixed-swa-bn",
              swa_epochs_1based=[14, 11, 12], provenance=dict(protocol_sha256=guard.sha256(protocol),
                                                          run_dir=p["run_dirs"]["local"]))
    guard.check_checkpoint(ck, p, protocol, "local")
    with pytest.raises(ValueError):
        guard.check_checkpoint(ck, p, protocol, "kaggle")
    ck["swa_epochs_1based"] = [13, 15, 16]
    with pytest.raises(ValueError):
        guard.check_checkpoint(ck, p, protocol, "local")


def test_queue_binds_selected_checkpointing_and_no_budget_override(registration_fixture):
    f = registration_fixture
    guard.prepare(f.args)
    v = dict(status="VALIDATED_NO_TRAINING", protocol_sha256=guard.sha256(f.args.protocol),
             full_corpus_hashes=True, actual_pin_memory_tuple_returns_list=True,
             real_backbone_keys_loaded_cpu=True)
    guard.write_new_json(guard.validation_path(f.args.protocol), v)
    queue = Path(f.args.protocol).with_name("queue.json")
    a = guard.parser().parse_args(["queue-json", "--init-weights", str(f.init), "--corpus", str(f.corpus),
                                  "--after", "probe_tag", "--out", str(queue), "--no-grad-checkpointing"])
    guard.queue_json(a)
    job = guard.read_json(queue)[0]
    assert job["tag"] == "EXP143_OWN0949_S42v2" and job["after"] == "probe_tag"
    assert "--no-grad-checkpointing" in job["command"]
    assert "--max-hours" not in job["command"] and "--out" not in job["command"]
    assert not (f.base / "S42v2").exists()


def test_probe_parent_is_scratch_only_and_runs_both_modes(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(fit, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "load_registration", lambda *a: bare_registration())
    monkeypatch.setattr(guard, "require_validation", lambda *a: None)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    protocol = tmp_path / "manifest.json"
    protocol.write_text("fake registration", encoding="utf-8")
    calls = []
    def child(command, **kwargs):
        mode = command[command.index("--mode") + 1]
        calls.append(mode)
        out = Path(command[command.index("--out") + 1])
        guard.write_new_json(out / (mode + ".json"), dict(mode=mode, protocol_sha256=guard.sha256(protocol)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", child)
    a = SimpleNamespace(protocol=str(protocol), corpus="fake", init_weights="fake", steps=20,
                        workers=0, precision="bf16", out=str(tmp_path / ".local/exp143-v2-probes/probe"))
    fit.probe(a)
    assert calls == ["on", "off"]
    report = guard.read_json(Path(a.out) / "report.json")
    assert report["status"] == "SCRATCH_PROBE_ONLY_NO_FIT"
    assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("validation.json"))
    with pytest.raises(FileExistsError):
        fit.probe(a)


def test_resume_wall_budget_includes_failed_attempts(tmp_path):
    (tmp_path / "stop-1.json").write_text(json.dumps(dict(elapsed_seconds=900)))
    (tmp_path / "stop-2.json").write_text(json.dumps(dict(elapsed_seconds=1100)))
    assert fit.elapsed_before_resume(tmp_path, 30) == 1100
    assert fit.elapsed_before_resume(tmp_path, 1200) == 1200
