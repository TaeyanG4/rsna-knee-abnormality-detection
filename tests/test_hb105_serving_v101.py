"""CPU-only serving arithmetic, identity, failure and timing accounting tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch
from scipy.stats import rankdata

from tools import hb105_serving_v101 as h
from tools import exp169_release_v6 as v
from training.arms import mil
from training.src.raptor_exp185 import collapsed_logits

torch.set_num_threads(2)
torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU suite touched CUDA")
    for name in ("_lazy_init", "init", "device_count", "is_available", "synchronize", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def arrays(n=17):
    rng = np.random.default_rng(307)
    return {k: rng.random((c, n, 12), dtype=np.float32) for k, c in (("R50", 10), ("C4", 5), ("T", 10), ("S", 10))}


def rank(a):
    return (rankdata(a, axis=0, method="average") - 1) / max(1, len(a) - 1)


@pytest.mark.parametrize("n", [1, 2, 17])
@pytest.mark.parametrize("ties", [False, True])
def test_registered_alternatives_match_independent_formula_and_v6(n, ties, monkeypatch):
    scores = arrays(n)
    if ties:
        scores = {k: np.round(a * 4) / 4 for k, a in scores.items()}
    r, c, t, s = [scores[k] for k in ("R50", "C4", "T", "S")]
    family = [rank(((r[:5] + r[5:]) / 2).mean(0)), rank(c.mean(0)), rank(t.mean(0)), rank(s.mean(0))]
    actual = h.alternatives(scores)
    for name, indices in (("Q7", [0, 1, 2]), ("Q7_add_S", [0, 1, 2, 3]), ("Q7_replace_T_with_S", [0, 1, 3])):
        expected = rank(sum(rank(family[i]) for i in indices) / len(indices))
        np.testing.assert_array_equal(actual[name], expected)
    monkeypatch.setattr(v, "maths", lambda: (SimpleNamespace(rank=rank), None))
    np.testing.assert_array_equal(actual["Q7"], v.q_values(dict(l2=r, convnext=c, mil=t))[0])
    np.testing.assert_array_equal(actual["Q7_replace_T_with_S"], v.q_values(dict(l2=r, convnext=c, mil=s))[0])


def test_rank_before_member_mean_is_a_different_estimator():
    a = arrays()["T"]
    assert not np.array_equal(rank(a.mean(0)), rank(np.mean([rank(x) for x in a], axis=0)))


def test_roster_overlap_and_duplicates_are_rejected():
    ids = [str(i) for i in range(4343)]
    folds = [dict(val=ids[f::5], train=[u for i, u in enumerate(ids) if i % 5 != f]) for f in range(5)]
    assert len(h.validate_rosters(folds)) == 4343
    bad = copy.deepcopy(folds); bad[0]["train"][0] = bad[0]["val"][0]
    with pytest.raises(ValueError, match="overlap"): h.validate_rosters(bad)
    bad = copy.deepcopy(folds); bad[0]["train"].append(bad[0]["train"][0])
    with pytest.raises(ValueError, match="duplicate"): h.validate_rosters(bad)
    bad = copy.deepcopy(folds); bad[0]["val"][0] = bad[1]["val"][0]
    with pytest.raises(ValueError, match="partition"): h.validate_rosters(bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -.01, 1.01])
def test_invalid_probabilities_fail(bad):
    a = arrays(); a["S"][2, 3, 4] = bad
    with pytest.raises(ValueError, match="invalid family"):
        h.alternatives(a)


def test_incomplete_and_wrong_precision_members_fail():
    for action in (lambda a: a.pop("S"), lambda a: a.update(S=a["S"][:9]),
                   lambda a: a.update(T=a["T"].astype(np.float64))):
        a = arrays(); action(a)
        with pytest.raises(ValueError): h.alternatives(a)


@pytest.mark.parametrize("weight", [.3, .5])
def test_own_menu_matches_real_v6_blend(weight):
    a = arrays(); own = a["C4"][:2]
    result = h.own_alternatives(*own, a, weight)
    name = "B6" if weight == .3 else "B6-w50"
    for key, q in h.alternatives(a).items():
        np.testing.assert_array_equal(result[key], v.blend(name, [str(i) for i in range(17)], *own, q))


@pytest.mark.parametrize("weight", [.15, .4, np.nan, True])
def test_no_new_outer_weights(weight):
    a = arrays()
    with pytest.raises(ValueError, match="unregistered"): h.own_alternatives(*a["C4"][:2], a, weight)


def metadata(family="S"):
    return dict(arch=h.ARCH[family], res=224, lab=list(v.LABELS), contract="raptor64_336_c140_s02",
        norm="imagenet", window_policy="within-slot-valid", fold=0, epoch=19,
        model_kind="finding_spatial_tokens_v1", input_policy="center3", k_eval=24,
        flip_tta="coronal_axial_horizontal_mean_probability", seed=42, provenance={"spec_sha256": "bound"})


@pytest.mark.parametrize("key,bad", [("arch", h.ARCH["T"]), ("model_kind", "global"), ("fold", 1),
    ("epoch", 18), ("input_policy", "neighbor3"), ("norm", "none"), ("flip_tta", "none"), ("k_eval", 36)])
def test_model_contract_rejects_drift(key, bad):
    entry = dict(id="S-s42-f0", arch=h.ARCH["S"], seed=42, fold=0, spec_sha256="bound")
    ck = metadata(); h.meta(ck, entry, v)
    ck[key] = bad
    with pytest.raises(ValueError): h.meta(ck, entry, v)
    ck = metadata(); ck["provenance"]["spec_sha256"] = "changed"
    with pytest.raises(ValueError, match="spec binding"): h.meta(ck, entry, v)


def fake_corpus():
    return SimpleNamespace(spec={"slots": mil.common.SLOTS64}, mask=lambda uid: np.ones(64, np.uint8))


def test_plane_flip_and_instrumented_cpu_parity():
    corpus = fake_corpus()
    x = np.arange(36 * 3 * 4 * 4, dtype=np.float32).reshape(36, 3, 4, 4) / 10000
    centers = mil.common.select_centers(corpus.mask("u"), 64, 36, corpus.spec, "within-slot-valid")
    mirror = mil.flip_inputs(torch.from_numpy(x)[None], ["u"], corpus, 36)[0].numpy()
    assert any(c < 32 for c in centers) and any(c >= 32 for c in centers)
    for i, center in enumerate(centers):
        np.testing.assert_array_equal(mirror[i], x[i, :, :, ::-1] if center >= 32 else x[i])
    class Model(torch.nn.Module):
        def forward(self, z): return z[..., 0].mean((1, 2, 3))[:, None].repeat(1, 12)
    p, times = h.prediction(Model(), x, "u", corpus, "cpu", (np, torch, mil, v))
    np.testing.assert_array_equal(p, v.mil_prediction(Model(), x, "u", corpus, "cpu"))
    assert all(x >= 0 for x in times.values())


def test_finding_spatial_tokens_not_global_pooling():
    class Backbone(torch.nn.Module):
        num_features = 4
        def __init__(self):
            super().__init__(); self.head = SimpleNamespace(norm=torch.nn.Identity())
        def forward_features(self, x): return torch.cat([x, x[:, :1] ** 2], dim=1)
        def forward_head(self, x): return x.mean((2, 3))
    model = mil.FindingMIL(Backbone()).eval()
    x = torch.arange(1, 1 + 2 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 2, 3, 4, 4) / 100
    with torch.inference_mode():
        tokens, mask = model.encode(x)
        assert tokens.shape == (1, 32, 4) and mask.all()
        assert not torch.allclose(model(x), collapsed_logits(model, x), atol=1e-6, rtol=0)


def test_wrong_hash_rejected_before_deserialization(tmp_path, monkeypatch):
    p = tmp_path / "model_best.pt"; p.write_bytes(b"untrusted")
    def fail(*a, **kw): pytest.fail("must hash before load")
    monkeypatch.setattr(torch, "load", fail)
    with pytest.raises(ValueError, match="checkpoint changed"):
        h.load_model(p, dict(bytes=1, sha256="wrong"), "cpu", (np, torch, mil, v))


def test_readonly_discovery_missing_and_exact_aliases(tmp_path):
    root = tmp_path / "resident"; root.mkdir()
    p = root / "model_best.pt"; p.write_bytes(b"model")
    alias = root / "mil-copy.pt"; alias.write_bytes(p.read_bytes())
    corpus = root / "RAPTOR64_336_C140_S02"; corpus.mkdir()
    h.write(corpus / "manifest.json", {"test": True})
    m = dict(members=[dict(id="T-s42-f0", family="T", **h.pin(p))],
        corpus={"metadata": {"manifest.json": h.pin(corpus / "manifest.json")}})
    before = sorted(str(p) for p in root.rglob("*"))
    result = h.discover(m, [root])
    assert result["status"] == "MATCHED_METADATA_ONLY" and len(result["identical_aliases"]["T-s42-f0"]) == 2
    assert sorted(str(p) for p in root.rglob("*")) == before
    m["members"][0]["sha256"] = "missing"
    assert h.discover(m, [root])["status"] == "HOLD_MISSING_INPUTS"
    with pytest.raises(ValueError, match="missing search root"): h.discover(m, [root / "absent"])


def test_sources_drift_and_create_only(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "ROOT", tmp_path)
    p = tmp_path / "source.py"; p.write_bytes(b"original")
    m = {"sources": {"source.py": h.pin(p)}}; h.verify_sources(m)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"): h.verify_sources(m)
    j = tmp_path / "receipt.json"; h.write(j, {"first": True})
    with pytest.raises(FileExistsError): h.write(j, {"first": False})
    assert h.read(j) == {"first": True}


def financial_snapshot(**changes):
    return dict(source="B-authenticated-Vast-account", instance_id=54941539, machine_id=140617,
                no_topup=True, actual_billing_monitor="B", observed_at_utc="2026-10-12T00:00:00Z",
                billing_reference="SYNTHETIC TEST ONLY", **{
                    **dict(balance_usd=5.01, incurred_unsettled_usd=0, all_in_hourly_usd=10), **changes})


@pytest.mark.parametrize("change", [dict(balance_usd=5), dict(balance_usd=4),
    dict(balance_usd=6, incurred_unsettled_usd=1), dict(balance_usd=True),
    dict(all_in_hourly_usd=float("inf")), dict(incurred_unsettled_usd=-1)])
def test_finance_rejects_actual_reserve_breach_or_invalid_inputs(tmp_path, change):
    path = tmp_path / "actual.json"
    # Deliberately permit nonfinite JSON in the negative fixture.
    path.write_text(json.dumps(financial_snapshot(**change)))
    with pytest.raises(ValueError): h.finance(path)


@pytest.mark.parametrize("field,bad", [("source", "forecast"), ("instance_id", 1),
    ("machine_id", 1), ("no_topup", False), ("actual_billing_monitor", "none"),
    ("billing_reference", ""), ("observed_at_utc", "")])
def test_finance_requires_b_actual_observation_and_monitor(tmp_path, field, bad):
    d = financial_snapshot(); d[field] = bad
    path = tmp_path / "actual.json"; h.write(path, d)
    with pytest.raises(ValueError): h.finance(path)


def test_finance_does_not_project_runtime(tmp_path):
    path = tmp_path / "actual.json"; h.write(path, financial_snapshot())
    result = h.finance(path)
    assert result["available_usd"] == 5.01
    assert "approved_hours" not in result and "estimated_max_cost_usd" not in result


def test_slow_elapsed_completes_all_timing_loops_without_cuda(tmp_path, monkeypatch):
    """Exercise the timing branch with a fake device boundary, never real CUDA.

    Each timer tick advances 3601 seconds: loads, cohorts and every forward
    exceed both old caps. A very large rate*elapsed estimate is telemetry only.
    """
    import itertools
    ticks = itertools.count(3601, 3601)
    monkeypatch.setattr(h, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    monkeypatch.setattr(h, "START", 0)
    cohort = [dict(uid="a"), dict(uid="b")]
    members = [dict(id=f"T-s{s}-f{f}", family="T") for s in (42, 43) for f in range(5)]
    manifest = tmp_path / "manifest.json"
    h.write(manifest, dict(sources={}, cohort=cohort, members=members))
    resolution = tmp_path / "resolution.json"
    h.write(resolution, dict(status="MATCHED_METADATA_ONLY", manifest_sha256=h.sha(manifest),
        corpus="synthetic", members={e["id"]: "synthetic" for e in members}))
    finance = tmp_path / "actual.json"; h.write(finance, financial_snapshot())
    noop = lambda *a, **kw: None
    fake_device = SimpleNamespace(device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(name="FAKE 3090 CPU TEST", total_memory=123),
        synchronize=noop, reset_peak_memory_stats=noop, memory_allocated=lambda: 10,
        max_memory_allocated=lambda: 20, max_memory_reserved=lambda: 30)
    fake_torch = SimpleNamespace(cuda=fake_device, __version__="CPU_TEST_DOUBLE")
    p = np.full(12, .5, np.float32)
    corpus = SimpleNamespace(volume=lambda _: np.zeros(1), mask=lambda _: np.ones(1), spec={})
    fake_mil = SimpleNamespace(common=SimpleNamespace(select_centers=lambda *a: [0]))
    fake_serving = SimpleNamespace(backend=noop, mil_windows=lambda *a: np.zeros(1),
                                   mil_prediction=lambda *a: p)
    monkeypatch.setattr(h, "imports", lambda: (np, fake_torch, fake_mil, fake_serving))
    monkeypatch.setattr(h, "verify_corpus", lambda *a: corpus)
    loaded, forwarded = [], []
    def load(path, entry, *args):
        loaded.append(entry["id"]); return entry["id"]
    def predict(model, *args):
        forwarded.append(model)
        return p, dict(flip_cpu_s=0., transfer_s=0., forward_s=3601., sigmoid_d2h_s=0.)
    monkeypatch.setattr(h, "load_model", load)
    monkeypatch.setattr(h, "prediction", predict)
    out = tmp_path / "timing"
    result = h.run(manifest, out, resolution=resolution, family="T", finance_path=finance)
    assert result["status"] == "COMPONENT_PARITY_COMPLETE"
    assert len(loaded) == 10 and len(forwarded) == 60
    r = h.read(out / "receipt.json")
    assert r["total_s"] > 900 and r["estimated_elapsed_cost_usd"] > 5.01
    assert r["failures"] == 0 and not (out / "failure.json").exists()
    assert len((out / "events.jsonl").read_text().splitlines()) == 60
    assert torch.cuda._initialized is False


def test_successor_science_and_non_timing_functions_unchanged():
    import ast
    parent = Path(h.ROOT / "tools/hb105_serving.py").read_text(encoding="utf-8")
    child = Path(h.__file__).read_text(encoding="utf-8")
    def functions(source):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef)}
    old, new = functions(parent), functions(child)
    for name in old.keys() - {"budget", "run", "package", "main"}:
        assert old[name] == new[name], name
    assert "hours * 3600" not in child and "approved_hours" not in child
    assert "def budget(" not in child


def test_run_failure_preserves_evidence_without_cuda(tmp_path, monkeypatch):
    path = tmp_path / "input.json"; h.write(path, {"sources": {}})
    def fail(): raise RuntimeError("injected import failure")
    monkeypatch.setattr(h, "imports", fail)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected"): h.run(path, out)
    assert h.read(out / "failure.json")["status"] == "FAILED_PRESERVED"
    with pytest.raises(FileExistsError): h.run(path, out)


def test_package_closure_create_only(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "ROOT", tmp_path); monkeypatch.setattr(h, "SOURCES", ["tools/a.py"])
    (tmp_path / "tools").mkdir(); (tmp_path / "tools/a.py").write_text("pass\n")
    folder = tmp_path / "artifacts/experiments/HB105_SERVING_V1_0_1"; folder.mkdir(parents=True)
    for name in ("LAUNCH.md", "REPORT.md"): (folder / name).write_text(name)
    path = tmp_path / "manifest.json"; h.write(path, {"sources": {"tools/a.py": h.pin(tmp_path / "tools/a.py")}})
    out = tmp_path / "bundle.zip"; h.package(path, out)
    with zipfile.ZipFile(out) as z:
        assert "tools/a.py" in z.namelist() and "input-manifest.json" in z.namelist()
        assert all(not n.endswith((".pt", ".npy", ".csv")) for n in z.namelist())
    with pytest.raises(FileExistsError): h.package(path, out)


def test_compare_rejects_incomplete_receipt(tmp_path):
    manifest = tmp_path / "manifest.json"; h.write(manifest, {"sources": {}})
    folder = tmp_path / "timing"; folder.mkdir()
    h.write(folder / "receipt.json", dict(manifest_sha256=h.sha(manifest), status="FAILED_PRESERVED"))
    with pytest.raises(ValueError, match="incomplete"): h.compare(manifest, folder, folder, tmp_path / "result.json")


def test_compare_complete_synthetic_timings_and_detect_tamper(tmp_path):
    m = dict(sources={}, cohort=[dict(uid="a"), dict(uid="b")], members=[
        dict(id=f"{k}-s{s}-f{f}", family=k) for k in ("T", "S") for s in (42, 43) for f in range(5)])
    manifest = tmp_path / "manifest.json"; h.write(manifest, m)
    for family, factor in (("T", 1), ("S", 2)):
        folder = tmp_path / family; folder.mkdir()
        members = [e["id"] for e in m["members"] if e["family"] == family]
        np.savez(folder / "probabilities.npz", probabilities=np.full((3, 10, 2, 12), .4, np.float32), member_ids=members, ids=["a", "b"])
        events = []
        for rep in range(3):
            for i in range(2):
                for j, member in enumerate(members):
                    events.append(dict(repeat=rep, cohort_index=i, member=member, first_model_forward=rep == 0 and i == 0,
                        cache_read_s=1 if j == 0 else 0, preprocess_s=1 if j == 0 else 0, flip_cpu_s=.1,
                        transfer_s=.1, forward_s=factor, sigmoid_d2h_s=.1, mil_tta_wall_s=factor + .3))
        (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        r = dict(manifest_sha256=h.sha(manifest), status="COMPONENT_PARITY_COMPLETE", device="cuda:0", resident_models=10,
            failures=0, missing=0, fallback_count=0, max_reference_abs_error=0, repeat_max_abs_error=0,
            gpu={"name": "synthetic3090"}, runtime={"test": "synthetic"}, cold_definition="synthetic",
            python_start_to_imports_s=1, cuda_startup_s=1, measurements=[dict(hash_deserialize_construct_strict_load_device_s=1)] * 10,
            resident_allocated_bytes=100 * factor, peak_allocated_bytes=200 * factor, peak_reserved_bytes=300 * factor,
            total_s=100, estimated_elapsed_cost_usd=.01,
            files={p.name: h.pin(p) for p in folder.iterdir()})
        h.write(folder / "receipt.json", r)
    result = h.compare(manifest, tmp_path / "T", tmp_path / "S", tmp_path / "comparison.json")
    assert result["status"] == "COMPONENT_COMPARISON_ONLY_FULL_RELEASE_HOLD"
    assert result["warm_replacement_mil_delta_s_per_study"] == pytest.approx(10)
    assert result["families"]["S"]["warm_per_study"]["preprocess_s"] == 1
    assert result["S_T_mean_abs_probability_difference"] == 0
    (tmp_path / "S/events.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        h.compare(manifest, tmp_path / "T", tmp_path / "S", tmp_path / "tampered.json")
