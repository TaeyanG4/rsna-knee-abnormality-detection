"""Synthetic-only registered H-B-150 decision and backend tests."""
import numpy as np
import pytest
from copy import deepcopy

from tools import hb150_readout_v1 as readout


def test_gate_same_three_rulers_and_threshold():
    mean = np.full((4, 12), .001)
    seed = np.stack([mean, mean])
    assert readout.gate(seed, mean)["GO"]
    mean[3] = 0
    assert readout.gate(seed, mean)["GO"]
    mean[2] = 0
    assert not readout.gate(seed, mean)["GO"]


def test_gate_metrics_must_pass_same_rulers():
    mean = np.full((4, 12), .002)
    mean[0, readout.lens._base.WEAK] = 0
    other = readout.lens._base.OTHER
    mean[1, other] = -.001
    assert not readout.gate(np.stack([mean, mean]), mean)["GO"]


@pytest.mark.parametrize("location", ["seed", "mean"])
def test_any_cell_floor_blocks_even_with_good_macros(location):
    mean = np.full((4, 12), .01)
    seeds = np.stack([mean, mean])
    (seeds if location == "seed" else mean).flat[0] = -.00300001
    result = readout.gate(seeds, mean)
    assert not result["GO"] and not result["cell_floor_pass"]
    (seeds if location == "seed" else mean).flat[0] = -.003
    assert readout.gate(seeds, mean)["GO"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_point_blocks(bad):
    mean = np.full((4, 12), .01)
    seeds = np.stack([mean, mean])
    seeds[1, 3, 11] = bad
    assert not readout.gate(seeds, mean)["GO"]
    assert not readout.gate(np.stack([mean, mean]), np.full((4, 12), bad))["GO"]


@pytest.mark.parametrize("shape", [(1, 4, 12), (3, 4, 12), (2, 3, 12), (2, 4, 11)])
def test_exact_tensor_required(shape):
    assert not readout.gate(np.full(shape, .01), np.full((4, 12), .01))["GO"]


def test_engineering_never_go():
    result = readout.gate(np.full((2, 4, 12), .01), np.full((4, 12), .01), engineering=True)
    assert not result["GO"] and result["status"] == "ENGINEERING_ONLY"


class NonlinearContext:
    """Synthetic scorer distinguishes averaging scores from scoring mean preds."""
    def __init__(self):
        self.ids = ["synthetic-0", "synthetic-1"]
        y = np.tile([[0.], [1.]], (1, 12))
        self.refs = {r: (y, np.ones_like(y, dtype=bool), 1) for r in readout.lens.RULERS}

    def score(self, p):
        return np.full((2, 4, 12), float(np.mean(p)) ** 2)

    def delta(self, a, c):
        return self.score(a) - self.score(c)


def test_backend_uses_mean_predictions_and_exposure(monkeypatch):
    ctx = NonlinearContext()
    arms = {42: np.full((2, 12), .2), 43: np.full((2, 12), .8)}
    controls = {42: np.full((2, 12), .4), 43: np.full((2, 12), .4)}
    original = readout.lens.evaluate
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(readout.lens, "evaluate", capture)
    result = readout.evaluate(ctx, arms, controls)
    assert seen["label_change"] is True
    assert result["registered_go"]["report_macro_seed_mean_deltas"]["jev"] == pytest.approx(.09)
    assert not result["registered_go"]["GO"]  # seed42 harms every cell
    assert result["status_scope"] == "EXPOSED_EXPLORATORY" and result["clean_oof"] is False
    assert result["release_scope"] == "none"
    assert not result["lenses"]["L7"]["eligible"]


def test_real_auc_backend_synthetic_context_and_descriptive_images():
    ids = [f"synthetic-{i}" for i in range(8)]
    y = np.tile((np.arange(8) % 2)[:, None], (1, 12)).astype(float)
    components = [{"component_id": uid, "members": [uid]} for uid in ids]
    refs = {r: (y, np.ones_like(y, dtype=bool), 1) for r in readout.lens.RULERS}
    ctx = readout.lens._base.Context(ids, refs, components)
    good, flat = .2 + .6 * y, np.full(y.shape, .5)
    images = {name: {"ids": ids, "labels": y, "mask": np.isfinite(y),
                    "components": components, "arm": good, "control": flat, "eligible": True}
              for name in ("gold58", "oai_moaks")}
    result = readout.evaluate(ctx, {"42": good, "43": good}, {"42": flat, "43": flat}, images)
    assert result["registered_go"]["GO"]
    assert not result["lenses"]["L7"]["PASS"]
    assert all(not x["eligible"] for x in result["lenses"]["L7"]["entries"].values())
    assert images["oai_moaks"]["eligible"]  # caller input untouched


@pytest.mark.parametrize("keys", [(42,), (42, 44), (42, 43, 44)])
def test_reject_wrong_seeds(keys):
    values = {s: np.full((2, 12), .5) for s in keys}
    with pytest.raises(ValueError, match="seeds"):
        readout.evaluate(NonlinearContext(), values, values)


@pytest.mark.parametrize("bad", [np.nan, -1., 2.])
def test_reject_bad_probability(bad):
    arms = {s: np.full((2, 12), .5) for s in (42, 43)}
    controls = {s: p.copy() for s, p in arms.items()}
    arms[43][0, 0] = bad
    with pytest.raises(ValueError, match="probabilities"):
        readout.evaluate(NonlinearContext(), arms, controls)


def records():
    identity = {"arm": "global", "seed": 42, "epochs": 20, "registration_sha256": "registration",
                "prepare_sha256": "prepare", "labels_sha256": "PL-global", "model": {"arch": "rn50"},
                "weights": [1.] * 12, "smoke": False, "engineering_only": False,
                "exposed_exploratory": True, "status": "EXPOSED", "exposure": "EXPOSED", "clean_oof": False,
                "resolution": 320, "config_sha256": "config", "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
                "runtime": {"device": "cuda", "policy": "strict", "precision_fallback_allowed": False}}
    summary = {"identity": identity, "smoke": False, "device": "cuda", "steps": 7960,
               "counters": {"updates": 7960, "skips": 0}, "checkpoint_sha256": "checkpoint"}
    receipt = {"identity": identity, "engineering_only": False, "k": 36, "tta": True,
               "rows": 8, "uid_sha256": "uids", "csv_sha256": "csv", "checkpoint_sha256": "checkpoint"}
    expected = {**deepcopy(identity), "steps": 7960}
    checks = dict(csv_sha256="csv", checkpoint_sha256="checkpoint", uid_sha256="uids", rows=8)
    return summary, receipt, expected, checks


def test_authenticated_records_pass():
    summary, receipt, expected, checks = records()
    assert readout.validate_records(summary, receipt, expected, **checks)["seed"] == 42


@pytest.mark.parametrize("key,value", [("arm", "fill"), ("seed", 43), ("epochs", 19),
    ("registration_sha256", "older"), ("prepare_sha256", "older"), ("labels_sha256", "wrong-PL"),
    ("gpu_name", "NVIDIA T4"), ("smoke", True), ("engineering_only", True),
    ("config_sha256", "other"), ("resolution", 224), ("clean_oof", True)])
def test_authentication_rejects_older_wrong_or_partial_identity(key, value):
    summary, receipt, expected, checks = records()
    receipt["identity"][key] = value
    with pytest.raises(ValueError):
        readout.validate_records(summary, receipt, expected, **checks)


@pytest.mark.parametrize("fault", ["steps", "updates", "skips", "csv", "checkpoint", "uid", "tta", "k"])
def test_authentication_rejects_partial_or_unbound_prediction(fault):
    summary, receipt, expected, checks = records()
    if fault == "steps": summary["steps"] -= 1
    elif fault == "updates": summary["counters"][fault] += 1
    elif fault == "skips": summary["counters"][fault] = 80
    elif fault in ("csv", "checkpoint", "uid"): receipt[fault + "_sha256"] = "other"
    elif fault == "tta": receipt["tta"] = False
    else: receipt["k"] = 24
    with pytest.raises(ValueError):
        readout.validate_records(summary, receipt, expected, **checks)


@pytest.mark.parametrize("skips", [0, 1, 79])
def test_h146_allowed_skipped_updates_preserve_complete_attempt_count(skips):
    summary, receipt, expected, checks = records()
    summary["counters"]["skips"] = skips
    assert readout.validate_records(summary, receipt, expected, **checks)["seed"] == 42
    summary["counters"]["updates"] -= 1
    with pytest.raises(ValueError, match="update count"):
        readout.validate_records(summary, receipt, expected, **checks)


@pytest.mark.parametrize("skips", [-1, 80, 1.5, True, None])
def test_invalid_skipped_update_count_rejected(skips):
    summary, receipt, expected, checks = records()
    summary["counters"]["skips"] = skips
    with pytest.raises(ValueError, match="update count"):
        readout.validate_records(summary, receipt, expected, **checks)


def test_legacy_baseline_gpu_requires_reviewed_bound_attestation():
    summary, receipt, expected, checks = records()
    identity = receipt["identity"]
    del identity["gpu_name"]
    with pytest.raises(ValueError, match="attestation"):
        readout.validate_records(summary, receipt, expected, baseline=True, **checks)
    att = {"reviewed": True, "checkpoint_sha256": "checkpoint", "csv_sha256": "csv",
           "identity": deepcopy(identity), "gpu_name": expected["gpu_name"]}
    readout.validate_records(summary, receipt, expected, baseline=True, gpu_attestation=att, **checks)
    att["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="attestation"):
        readout.validate_records(summary, receipt, expected, baseline=True, gpu_attestation=att, **checks)


def test_running_outputs_are_never_opened(monkeypatch):
    from tools import hb146_pl_labels_v111 as pl
    opened = []

    def fake_read(path):
        opened.append(path.name)
        assert path.name == "heartbeat.json"
        return {"status": "running"}

    monkeypatch.setattr(pl, "read", fake_read)
    with pytest.raises(ValueError, match="output access forbidden"):
        readout.authenticated_prediction("synthetic-never-opened", ["synthetic"], {})
    assert opened == ["heartbeat.json"]


def test_metadata_only_authentication_never_parses_csv(monkeypatch):
    from tools import hb146_pl_labels_v111 as pl
    import hashlib
    summary, receipt, expected, checks = records()
    ids = [f"synthetic-{i}" for i in range(8)]
    receipt["uid_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()

    def read(path):
        return {"heartbeat.json": {"status": "complete"}, "summary.json": summary,
                "val.receipt.json": receipt}[path.name]

    monkeypatch.setattr(pl, "read", read)
    monkeypatch.setattr(pl, "sha", lambda path: {"val.csv": "csv", "final.pt": "checkpoint"}[path.name])
    monkeypatch.setattr(pl, "table", lambda *args, **kwargs: pytest.fail("admission parsed CSV"))
    actual = readout.authenticated_prediction("synthetic", ids, expected, metadata_only=True)
    assert actual == receipt


@pytest.mark.parametrize("older", [False, True])
def test_baseline_admission_binds_winning_readout(monkeypatch, older):
    from tools import hb150_build_v1 as build
    c = {"pl_work": "synthetic", "arm": "global", "val": ["synthetic-id"]}
    receipt = {"identity": {"runtime": {"device": "cuda"}}, "csv_sha256": "current"}
    decision = {"registered_go": {"GO": True, "selected_arm": "global"},
                "prediction_bindings": {f"val-global-{seed}": deepcopy(receipt) for seed in (42, 43)}}
    if older:
        decision["prediction_bindings"]["val-global-42"]["csv_sha256"] = "old"

    def read(path):
        if path.name == "heartbeat.json": return {"status": "complete"}
        if path.name == "attestation.json":
            return {"schema": "hb150-baseline-attestation/v1", "members": {"42": {}, "43": {}}}
        if path.name == "readout.json": return decision
        raise AssertionError("unexpected read")

    def authenticate(*args, **kwargs):
        assert kwargs["metadata_only"] is True and kwargs["baseline"] is True
        return receipt

    monkeypatch.setattr(build, "read", read)
    monkeypatch.setattr(readout, "expected_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(readout, "authenticated_prediction", authenticate)
    from pathlib import Path
    if older:
        with pytest.raises(ValueError, match="older prediction pair"):
            readout.validate_baselines(c, Path("attestation.json"))
    else:
        assert set(readout.validate_baselines(c, Path("attestation.json"))) == {"42", "43"}


def test_cli_requires_explicit_exposure():
    with pytest.raises(ValueError, match="exposed-exploratory"):
        readout.run("global", "synthetic-never-opened", False)
