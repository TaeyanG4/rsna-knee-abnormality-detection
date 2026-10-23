"""CPU fake contracts for EXP155 registration; no real data, backbone, network or CUDA."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import exp155_own0949_oai as guard
from tools import exp155_oai_data as data


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*a, **kw):
        raise AssertionError("CPU fakes must not query or initialize CUDA")
    for name in ("_lazy_init", "init", "is_available", "set_device", "manual_seed_all",
                 "is_bf16_supported", "synchronize", "mem_get_info", "get_device_name"):
        monkeypatch.setattr(torch.cuda, name, fail)
    yield
    assert not torch.cuda.is_initialized()


def table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=["study_key", *guard.TARGETS,
                                             *("mask_" + t for t in guard.TARGETS)])
        w.writeheader()
        w.writerows(rows)


def row(uid="OAI_123_R_V00"):
    r = dict(study_key=uid, **{t: "" for t in guard.TARGETS},
             **{"mask_" + t: 0 for t in guard.TARGETS})
    r.update({"PF OA": 1, "mask_PF OA": 1, "Synovitis": 0, "mask_Synovitis": 1})
    return r


def test_unknown_is_masked_and_other_targets_cannot_be_supervised(tmp_path):
    f = tmp_path / "oai.csv"
    table(f, [row()])
    labels, masks = guard.read_oai_labels(f)
    uid = "OAI_123_R_V00"
    assert labels[uid][6] == 1 and masks[uid][6] == 1
    assert labels[uid][5] == 0 and masks[uid][5] == 0
    assert labels[uid][8] == 0 and masks[uid][8] == 1
    bad = row()
    bad.update(ACL=1, mask_ACL=1)
    table(f, [bad])
    with pytest.raises(ValueError, match="three registered"):
        guard.read_oai_labels(f)


@pytest.mark.parametrize("change", [
    {"study_key": "OAI_123_R_V01"}, {"mask_PF OA": 2}, {"PF OA": "NaN"},
    {"PF OA": .5}, {"PF OA": ""}, {"Lateral OA": 1},
])
def test_invalid_oai_inputs_fail_closed(tmp_path, change):
    r = row()
    r.update(change)
    f = tmp_path / "oai.csv"
    table(f, [r])
    with pytest.raises(ValueError):
        guard.read_oai_labels(f)


def test_duplicate_oai_identity_rejected(tmp_path):
    f = tmp_path / "oai.csv"
    table(f, [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        guard.read_oai_labels(f)


def test_one_copy_mixing_excludes_gold_and_all_unknown_studies():
    rsna = {"rsna": [.25] * 12}
    labels = {"oai": [0.] * 12, "unknown": [0.] * 12, "noimage": [0.] * 12}
    masks = {"oai": [0] * 5 + [1] + [0] * 6, "unknown": [0] * 12,
             "noimage": [1] * 12}
    ids, values, mask, external = guard.combine_cohorts(["rsna"], rsna, ["gold"],
                                                       ["oai", "unknown"], labels, masks)
    assert ids == ["rsna", "oai"] and external == ["oai"]
    assert mask["rsna"] == [1] * 12 and mask["oai"] == masks["oai"]
    assert "gold" not in values and "unknown" not in values
    with pytest.raises(ValueError, match="Gold leakage"):
        guard.combine_cohorts(["rsna"], rsna, ["gold"], ["gold"], labels, masks)


def test_fixed_recipe_update_count_and_projection():
    p = dict(recipe=copy.deepcopy(guard.RECIPE), train_ids=list(range(4347 + 2399)),
             budgets=dict(local=14., reserve_seconds=600))
    s = guard.execution_settings(p, 1, "bf16", 42)
    assert s["accumulation"] == 16 and s["batch"] == 1
    assert s["total_steps"] == 6752 and s["updates_per_epoch"] == 422
    assert s["grad_checkpointing"] is False
    assert guard.projection(s, 3.15)["projected_seconds"] == 6752 * 3.15 + 600
    assert p["recipe"]["swa_1based"] == [14, 11, 12]
    for args in ((2, "fp16", 43, False), (1, "bf16", 42, True), (1, "fp16", 42, False)):
        with pytest.raises(ValueError):
            guard.execution_settings(p, *args)


@pytest.fixture
def registered(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    experiment = tmp_path / "artifacts/experiments/EXP155_OWN0949_OAI"
    monkeypatch.setattr(guard, "BASE", experiment)
    # scoped_path's default is bound at definition, so call through a test-local root.
    original = guard.scoped_path
    monkeypatch.setattr(guard, "scoped_path", lambda path, root=None: original(path, root or experiment))
    code = tmp_path / "fake-source.py"
    code.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(guard, "CODE", ["fake-source.py"])
    rsna = [f"fake-{i}" for i in range(4347)]
    gold = [f"gold-{i}" for i in range(58)]
    labels = {u: [.3] * 12 for u in rsna}
    baseline = dict(train_ids=rsna, train_labels=labels, excluded_gold_ids=gold,
        recipe=dict(guard.base.RECIPE, labels="registered-replacement-soft"),
        run_dirs={"local": "artifacts/experiments/EXP143_OWN0949/S42v2"},
        init_sha256="fake-init", init_provenance={"source": "tiny fake"}, public_gold_sha256="public")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("{}", encoding="utf-8")
    label_path = tmp_path / "labels.csv"
    label_path.write_text("fake labels", encoding="utf-8")
    baseline["label_sha256"] = guard.sha256(label_path)
    monkeypatch.setattr(guard, "LABEL_SHA", baseline["label_sha256"])
    monkeypatch.setattr(guard.base, "load_registration", lambda *a, **kw: baseline)
    monkeypatch.setattr(guard.base, "pos_weights", lambda ys, ids: [3.] * 12 if ids == rsna else [999.] * 12)
    monkeypatch.setattr(guard.base, "check_corpus", lambda *a, **kw: {"ids": rsna + gold})
    monkeypatch.setattr(data, "check_cache", lambda *a, **kw: {"ids": ["OAI_123_R_V00"]})
    for directory in (tmp_path / "corpus", tmp_path / "oai"):
        directory.mkdir()
        for name in ("manifest.json", "build-registration.json"):
            (directory / name).write_text("{}", encoding="utf-8")
    oai = tmp_path / "oai.csv"
    table(oai, [row()])
    oai.with_suffix(".provenance.json").write_text(json.dumps({"csv_sha256": guard.sha256(oai),
        "cache_manifest_sha256": guard.sha256(tmp_path / "oai/manifest.json"),
        "targets": list(guard.OAI_TARGETS), "other_targets_masked": True}))
    args = SimpleNamespace(protocol=experiment / "protocol-v1/manifest.json", base_protocol=baseline_path,
        corpus=tmp_path / "corpus", oai_corpus=tmp_path / "oai", labels=label_path,
        oai_labels=oai, init_weights=tmp_path / "fake.pt", max_hours=14.)
    guard.prepare(args)
    return args, baseline, code


def test_registration_keeps_rsna_weights_and_binds_oai(registered):
    args, baseline, _ = registered
    p = guard.load_registration(args.protocol, args.corpus, oai_corpus=args.oai_corpus)
    assert len(p["train_ids"]) == 4348
    assert p["pos_weight"] == [3.] * 12 and p["recipe"]["oai"] is True
    assert p["rsna_train_ids"] == baseline["train_ids"]
    assert not set(p["excluded_gold_ids"]) & set(p["train_ids"])
    with pytest.raises(FileExistsError):
        guard.prepare(args)
    (args.oai_corpus / "manifest.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus binding"):
        guard.load_registration(args.protocol, oai_corpus=args.oai_corpus)


def test_registration_detects_source_and_manifest_edits(registered):
    args, _, code = registered
    code.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        guard.load_registration(args.protocol)


def test_rehashed_wrong_masks_weights_and_gold_fail(registered):
    args, _, _ = registered
    p = guard.read_json(args.protocol)
    for mutate in (
        lambda q: q["pos_weight"].__setitem__(0, 999.),
        lambda q: q["train_masks"]["OAI_123_R_V00"].__setitem__(0, 1),
        lambda q: q["excluded_gold_ids"].__setitem__(0, q["train_ids"][0]),
    ):
        bad = copy.deepcopy(p)
        mutate(bad)
        with pytest.raises(ValueError):
            guard.validate_membership(bad)


def test_queue_is_local_no_checkpointing_and_create_only(registered, monkeypatch):
    args, _, _ = registered
    monkeypatch.setattr(guard, "require_validation", lambda _: None)
    original = guard.load_registration
    monkeypatch.setattr(guard, "load_registration", lambda protocol, *a: original(protocol))
    args.workers, args.after, args.python = 2, "EXP155_PROBE", "fake-python"
    p = original(args.protocol)
    args.probe_report = Path(args.protocol).with_name("fake-probe.json")
    args.probe_report.write_text(json.dumps(dict(protocol_sha256=guard.sha256(args.protocol),
        results=[dict(mode="off", admissible=True, status="MEASURED_SCRATCH_ONLY", precision="bf16",
                      skipped_updates=0, execution=guard.execution_settings(p, 1, "bf16", 42, False))])))
    args.out = Path(args.protocol).with_name("queue.json")
    guard.queue_json(args)
    q = guard.read_json(args.out)
    assert len(q) == 1 and q[0]["after"] == "EXP155_PROBE"
    assert q[0]["continue_on_failure"] is False
    assert "--oai-corpus" in q[0]["command"] and "--no-grad-checkpointing" in q[0]["command"]
    with pytest.raises(FileExistsError):
        guard.queue_json(args)
    args.probe_report.write_text(json.dumps(dict(results=[])))
    with pytest.raises(ValueError, match="admissible"):
        guard.queue_json(args)
