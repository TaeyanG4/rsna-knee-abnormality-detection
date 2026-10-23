"""EXP182 v2.0.0: seed science, lifecycle dispatch and authenticated history."""
from __future__ import annotations

import copy
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp182_own0949_s43 as s  # noqa: E402


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    import torch
    assert not torch.cuda.is_initialized()
    def fail(*args, **kwargs):
        pytest.fail("CPU contract test attempted CUDA")
    for name in ("_lazy_init", "init", "is_available", "get_device_name", "set_device",
                 "manual_seed_all", "is_bf16_supported", "mem_get_info", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, fail)
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def reg():
    baseline = s.bridge.baseline_registration()
    return baseline, s.registration(baseline, "t")


def test_seed_is_the_only_scientific_change(reg):
    baseline, p = reg
    s.assert_seed_only_diff(p, baseline)
    assert [k for k in baseline["recipe"] if p["recipe"][k] != baseline["recipe"][k]] == ["seed_local"]
    assert p["recipe"]["seed_local"] == p["seed"] == 43 and p["recipe"]["seed_kaggle"] == 43
    for key in ("train_ids", "train_labels", "pos_weight", "label_sha256", "init_sha256",
                "corpus_manifest_sha256", "budgets", "excluded_gold_ids"):
        assert p[key] == baseline[key]
    s42 = s.read_json(s.BASE_PROVENANCE)["execution"]
    assert s.execution_settings(p, 1, "bf16", 43, False) == s42
    assert s.registered_run(p).name == "S43"


@pytest.mark.parametrize("path,value", [(("recipe", "epochs"), 15), (("recipe", "swa_1based"), [14, 15, 16]),
    (("recipe", "labels"), "jev-hybrid-soft"), (("train_ids",), []), (("budgets",), dict(local=7.0)),
    (("pos_weight",), [1.0] * 12), (("seed",), 42), (("expected_initial_state_sha256",), "x")])
def test_other_changes_rejected(reg, path, value):
    baseline, p = reg
    bad = copy.deepcopy(p)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        s.assert_seed_only_diff(bad, baseline)


@pytest.mark.parametrize("world,precision,seed,ckpt", [(1, "bf16", 42, False), (1, "fp16", 43, False),
                                                       (1, "bf16", 43, True), (2, "bf16", 43, False)])
def test_execution_rejects_other_lanes(reg, world, precision, seed, ckpt):
    with pytest.raises(ValueError):
        s.execution_settings(reg[1], world, precision, seed, ckpt)


def test_initial_state_pin_is_seed_specific():
    assert s.INITIAL_STATE_SHA != s.INITIAL_STATE_SHA_S42 and len(s.INITIAL_STATE_SHA) == 64


@pytest.mark.parametrize("name", ["prepare", "train", "validate", "queue_json", "build_queue"])
def test_legacy_producers_removed(name):
    assert not hasattr(s, name)


@pytest.mark.parametrize("name", ["register", "prepare", "validate", "queue-json", "train"])
def test_legacy_cli_requires_fresh_lifecycle(name, capsys):
    with pytest.raises(SystemExit) as exc:
        s.main([name])
    assert exc.value.code == 2
    assert "legacy producer retired" in capsys.readouterr().err


def test_lifecycle_dispatch_is_seed43(monkeypatch):
    from tools import coatnet_config
    calls = []
    monkeypatch.setattr(coatnet_config, "main", lambda argv, **kw: calls.append((argv, kw)))
    s.main(["lifecycle", "prepare", "--out", "fresh"])
    assert calls == [(["prepare", "--out", "fresh"], {"preset": "control-full-s43"})]


def test_check_summary_rejects_seed42_identity(reg):
    _, p = reg
    settings = s.execution_settings(p, 1, "bf16", 43, False)
    good = dict(status="COMPLETE_TRAINING_NO_EVALUATION", epochs=16, updates=4352, skips=0,
                history=[dict(epoch=e) for e in range(1, 17)], model_sha256="m",
                identity=dict(protocol_sha256="p", initial_state_sha256=s.INITIAL_STATE_SHA, seed=43,
                              run_dir=s.RUN_DIRS["local"], execution=settings))
    s.check_summary(good, p, "p", "m")
    bad = copy.deepcopy(good)
    bad["identity"].update(seed=42, initial_state_sha256=s.INITIAL_STATE_SHA_S42)
    with pytest.raises(ValueError):
        s.check_summary(bad, p, "p", "m")


def test_write_scope(tmp_path):
    with pytest.raises(ValueError):
        s.write_new_json(tmp_path / "x.json", {})


def test_original_manifest_and_completed_cpu_record():
    p = s.load_registration()
    assert p["version"] == "1.0.0" and p["seed"] == 43
    assert s.require_completed()["updates"] == 4352
    assert s.main(["status"]) == dict(status="COMPLETE_TRAINING_NO_EVALUATION",
                                     historical_only=True, launch_authorized=False)


def test_fresh_manifest_cannot_impersonate_history(reg, tmp_path, monkeypatch):
    import json
    path = tmp_path / "manifest.json"
    monkeypatch.setattr(s, "PROTOCOL", path)
    path.write_text(json.dumps(reg[1]), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol path changed"):
        s.load_registration(path)


def test_original_sources_use_archive_not_changed_live(monkeypatch):
    real_hash = s.sha256
    sources = {(ROOT / name).resolve() for name in s.CODE}
    def changed_live(path):
        assert Path(path).resolve() not in sources, "mutable historical source read"
        return real_hash(path)
    monkeypatch.setattr(s, "sha256", changed_live)
    assert s.load_registration()["seed"] == 43


def test_historical_archive_pin_rejects_rewrite(monkeypatch):
    monkeypatch.setattr(s, "ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="archive index changed"):
        s.load_registration()


def test_historical_artifacts_reject_changed_digest(monkeypatch):
    real_hash = s.sha256
    monkeypatch.setattr(s, "sha256", lambda p: "0" * 64 if Path(p).name == "summary.json" else real_hash(p))
    with pytest.raises(ValueError, match="historical artifact changed"):
        s.require_completed()
