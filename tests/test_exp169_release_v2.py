"""Registered offline CPU checks; synthetic data only except notebook replay.

These tests exercise release engineering, not model parity or qualification.
No checkpoint is deserialized and no notebook, GPU, or account API is run.
"""
from __future__ import annotations

import ast
import builtins
import copy
import csv
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import exp152_own0949_release as own
from tools import exp169_release_v2 as release
from tools import exp169_v102_replay as replay


ROOT = Path(__file__).resolve().parents[1]


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _csv(path, ids, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([release.UID, *release.TARGETS])
        writer.writerows([[uid, *row] for uid, row in zip(ids, values)])


def _reference_rank(values):
    """Pairwise definition, independent of the implementation's sorting."""
    a = np.asarray(values, dtype=np.float64)
    lower = (a[:, None, :] > a[None, :, :]).sum(axis=1)
    equal = (a[:, None, :] == a[None, :, :]).sum(axis=1)
    return (lower + (equal - 1) / 2) / max(1, len(a) - 1)


def _scores(n=4):
    rng = np.random.default_rng(172)
    return {family: rng.random((members, n, 12), dtype=np.float64)
            for family, members in (("l2", 10), ("dino", 5), ("convnext", 5))}


def _save_anchor(folder, competition, candidate="B172"):
    folder.mkdir(parents=True)
    ids = ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"]
    raw_values = np.tile(np.asarray([.4, .4, .2, .9])[:, None], (1, 12))
    ordinal = np.tile(np.asarray([1 / 3, 2 / 3, 0., 1.])[:, None], (1, 12))
    arrays = {"ids": np.asarray(ids), "targets": np.asarray(release.TARGETS), "own": raw_values}
    if candidate == "A172":
        arrays.update(public=np.full((4, 12), .3), exp143=np.full((4, 12), .7))
    np.savez(folder / "raw.npz", **arrays)
    binding = {"variant": "own" if candidate == "B172" else "blend3",
               "own_sha256": "synthetic-own", "public_sha256": None if candidate == "B172" else "synthetic-public"}
    _json(folder / "binding.json", binding)
    _json(folder / "inputs.json", {"synthetic": True})
    _json(folder / "runtime.json", {"device": "synthetic-cpu-test"})
    _csv(folder / "submission.csv", ids, ordinal)
    _csv(competition / "sample_submission.csv", ids, np.zeros((4, 12)))
    receipt = {"status": "COMPLETE_INFERENCE", "seconds": 10., "files": {
        name: release.sha(folder / name) for name in
        ("raw.npz", "inputs.json", "runtime.json", "binding.json", "submission.csv")}}
    _json(folder / "receipt.json", receipt)
    return ids, raw_values, ordinal, binding


@pytest.fixture
def anchor(tmp_path):
    work, competition = tmp_path / "anchor", tmp_path / "competition"
    ids, raw, ordinal, binding = _save_anchor(work, competition)
    return SimpleNamespace(work=work, sample=competition / "sample_submission.csv",
                           ids=ids, raw=raw, ordinal=ordinal,
                           manifest={"candidate": "B172", "anchor_binding": binding})


@pytest.fixture
def serving(tmp_path):
    return _build_serving(tmp_path, "B172")


@pytest.fixture
def serving_a(tmp_path):
    return _build_serving(tmp_path, "A172")


def _build_serving(tmp_path, candidate):
    work, anchor_dir = tmp_path / "partner", tmp_path / "anchor"
    competition, output = tmp_path / "competition", tmp_path / "output"
    ids, raw, ordinal, binding = _save_anchor(anchor_dir, competition, candidate)
    work.mkdir()
    files, entries = {}, []

    def add_file(name, content):
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[name] = {"sha256": release.sha(path), "bytes": path.stat().st_size}

    add_file("config/workspace.toml", b"# synthetic serving fixture\n")
    add_file("code/exp169_partner_bundle_release.py", Path(release.base.__file__).read_bytes())
    add_file("code/exp169_release_v2.py", Path(release.__file__).read_bytes())
    for family, seeds in (("l2", (42, 43)), ("dino", (42,)), ("convnext", (42,))):
        for seed in seeds:
            for fold in range(5):
                name = f"models/{family}_s{seed}_f{fold}.pt"
                add_file(name, f"synthetic-only-{family}-{seed}-{fold}".encode())
                tag = (f"EXP172_DINO_BASE_L2_S42_F{fold}" if family == "dino" else
                       f"SYNTHETIC_{family}_{seed}_{fold}")
                entry = dict(family=family, seed=seed, fold=fold, tag=tag, path=name,
                             source_sha256=files[name]["sha256"], **files[name])
                if family == "dino":
                    entry["format"] = "authenticated-original"
                entries.append(entry)
    recipe = release.RECIPES[candidate]
    manifest = dict(schema=release.base.SCHEMA, generator_version=release.VERSION,
                    candidate=candidate, k_eval=36,
                    weights={"trio": recipe["anchor_weight"], "bundle": recipe["bundle_weight"]},
                    family_weights=dict(l2=1/3, dino=1/3, convnext=1/3),
                    entries=entries, files=files, anchor_binding=binding,
                    verification_note={"path": "synthetic-note", "sha256": "synthetic-note-hash"},
                    engineering_protocol_sha256="synthetic-protocol-hash")
    _json(work / "release-manifest.json", manifest)
    scores = _scores()
    np.savez(work / "pilot-scores.npz", ids=np.asarray(ids), **scores)
    pilot = dict(status="PASS", gpus=2, studies=len(ids), full_cohort=len(ids),
                 elapsed_seconds=1., projected_full_seconds=10., peak_vram_bytes=1024,
                 peak_ram_bytes=2048, scores_path=str(work / "pilot-scores.npz"),
                 manifest_sha256=release.sha(work / "release-manifest.json"),
                 scores_sha256=release.sha(work / "pilot-scores.npz"))
    _json(work / "partner-pilot.json", pilot)
    metrics = dict(gpu_peak_bytes=[1024, 1024], ram={"maxrss_kib": 1024})

    def partner(*args):
        return copy.deepcopy(scores), copy.deepcopy(metrics)

    return SimpleNamespace(work=work, anchor=anchor_dir, competition=competition,
                           output=output, manifest=manifest, scores=scores,
                           partner=partner, ids=ids, raw=raw, ordinal=ordinal,
                           candidate=candidate)


def _run(s, **kwargs):
    return release.serve(s.work, s.anchor, s.output, s.competition,
                         clock=lambda: 100., partner=kwargs.get("partner", s.partner))


def test_qualified_notebook_reproduces_exact_bytes():
    actual = replay.regenerate(ROOT)
    target = ROOT / replay.BASELINE_STAGE_PATH / "kernel/release.ipynb"
    assert actual == target.read_bytes()
    assert len(actual) == 13463
    assert hashlib.sha256(actual).hexdigest() == replay.QUALIFIED_NOTEBOOK_SHA256


def test_foreign_tools_namespace_uses_packaged_imports(monkeypatch):
    original = builtins.__import__
    def foreign_tools(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tools" and "exp169_partner_bundle_release" in fromlist:
            raise ImportError("cannot import name from foreign tools package")
        if name == "tools.exp152_own0949_release":
            raise ImportError("foreign tools package has no EXP152 helper")
        return original(name, globals, locals, fromlist, level)
    monkeypatch.setitem(sys.modules, "exp169_partner_bundle_release", release.base)
    monkeypatch.setitem(sys.modules, "exp152_own0949_release", own)
    monkeypatch.setattr(builtins, "__import__", foreign_tools)
    namespace = runpy.run_path(str(Path(release.__file__)))
    assert namespace["base"] is release.base
    np.testing.assert_array_equal(namespace["rank"](np.ones((2, 12))),
                                  np.full((2, 12), .5))


@pytest.mark.parametrize("candidate,weights", [("A172", (.85, .15)), ("B172", (.7, .3))])
def test_candidate_notebook_binds_recipe_and_mounts(candidate, weights):
    anchor_nb, _, _ = own.notebook("own", "synthetic/b172-own", "1"*64,
                                    "2"*64, "3"*64, [])
    notebook = release.candidate_notebook(candidate, "synthetic/partner-v2", "4"*64,
                                         "5"*64, anchor_nb)
    source = notebook["cells"][-1]["source"]
    ast.parse(source)
    assert "synthetic/partner-v2" in source and "/kaggle/input/partner-v2" in source
    assert "4"*64 in source and "5"*64 in source
    assert "import exp169_release_v2 as _e169" in source
    assert "_torch.backends.cudnn.benchmark = False" in source
    assert "_torch.backends.cudnn.deterministic = True" in source
    assert source.index("cudnn.benchmark = False") < source.index("_e169.pilot(")
    assert "unlink(missing_ok=True)" in source
    assert "_r.get('status') != 'PASS_BUNDLE'" in source
    assert release.RECIPES[candidate]["anchor_weight"] == weights[0]
    assert release.RECIPES[candidate]["bundle_weight"] == weights[1]
    if candidate == "B172":
        assert len(notebook["cells"]) == 3
        assert "Credits and required acknowledgement" in notebook["cells"][1]["source"]
        assert "synthetic/b172-own" in source
        assert "_public_mounts" not in source
        assert own.PUBLIC_NAME not in source
        assert "knee-exp155-own0949-oai" not in source
        assert "rsna-knee-publication-swa-weights" not in source
        assert "'own', _work" in source
    else:
        assert "'blend3', _work" in source
        assert "rsna-knee-publication-swa-weights" in source


def test_average_ties_and_single_study_rank():
    values = np.tile(np.asarray([.4, .4, .2, .9])[:, None], (1, 12))
    np.testing.assert_array_equal(release.rank(values), _reference_rank(values))
    np.testing.assert_array_equal(release.rank(values)[:, 0], [.5, .5, 0., 1.])
    np.testing.assert_array_equal(release.rank(np.ones((1, 12))), np.zeros((1, 12)))


@pytest.mark.parametrize("candidate,weights", [("A172", (.85, .15)), ("B172", (.7, .3))])
def test_family_means_and_outer_formula(candidate, weights):
    scores = _scores(6)
    expected_families = {
        "l2": _reference_rank(((scores["l2"][:5] + scores["l2"][5:]) / 2).mean(0)),
        "dino": _reference_rank(scores["dino"].mean(0)),
        "convnext": _reference_rank(scores["convnext"].mean(0)),
    }
    expected_q = _reference_rank(sum(_reference_rank(a) for a in expected_families.values()) / 3)
    q, families = release.aggregate(scores)
    for family in expected_families:
        np.testing.assert_array_equal(families[family], expected_families[family])
    np.testing.assert_array_equal(q, expected_q)
    anchor = np.tile(np.asarray([.4, .4, .2, .9, .2, .8])[:, None], (1, 12))
    expected = weights[0] * _reference_rank(anchor) + weights[1] * _reference_rank(q)
    np.testing.assert_array_equal(release.final_values(anchor, q, candidate), expected)


def test_global_rank_after_shard_reunion_is_reorder_invariant():
    scores = _scores(6)
    expected, _ = release.aggregate(scores)
    order = np.asarray([0, 3, 5, 1, 2, 4])
    reunited = {name: np.concatenate([array[:, order[:3]], array[:, order[3:]]], axis=1)
                for name, array in scores.items()}
    actual, _ = release.aggregate(reunited)
    np.testing.assert_array_equal(actual[np.argsort(order)], expected)
    assert not np.array_equal(np.concatenate([
        release.aggregate({name: array[:, order[:3]] for name, array in scores.items()})[0],
        release.aggregate({name: array[:, order[3:]] for name, array in scores.items()})[0]
    ])[np.argsort(order)], expected)


@pytest.mark.parametrize("bad", ["missing_family", "missing_seed", "missing_target", "nan", "infinity", "outside_probability", "cohort_mismatch"])
def test_invalid_partner_arrays_fail(bad):
    scores = _scores()
    if bad == "missing_family":
        del scores["dino"]
    elif bad == "missing_seed":
        scores["l2"] = scores["l2"][:5]
    elif bad == "missing_target":
        scores["dino"] = scores["dino"][:, :, :11]
    elif bad == "cohort_mismatch":
        scores["dino"] = scores["dino"][:, :3]
    else:
        scores["dino"][0, 0, 0] = {"nan": np.nan, "infinity": np.inf, "outside_probability": 1.01}[bad]
    with pytest.raises(ValueError):
        release.aggregate(scores)


@pytest.mark.parametrize("candidate", ["A172", "B172"])
def test_final_values_reject_nonfinite_missing_target_or_cohort(candidate):
    valid = np.full((4, 12), .5)
    for invalid in (np.full((4, 12), np.nan), np.full((4, 12), 1.1), np.ones((4, 11)), np.ones((3, 12))):
        with pytest.raises(ValueError):
            release.final_values(valid, invalid, candidate)


def test_exact_partner_matrix_accepts_authenticated_twenty_members(serving):
    assert release.verify_assets(serving.work, serving.manifest) is True


@pytest.mark.parametrize("bad", ["duplicate_fold", "wrong_seed", "reordered", "source_reused", "wrong_dino_tag", "wrong_dino_format", "entry_hash", "weight", "family_weight"])
def test_partner_matrix_rejects_identity_drift(serving, bad):
    manifest = copy.deepcopy(serving.manifest)
    if bad == "duplicate_fold":
        manifest["entries"][1]["fold"] = 0
    elif bad == "wrong_seed":
        manifest["entries"][5]["seed"] = 42
    elif bad == "reordered":
        manifest["entries"][0], manifest["entries"][1] = manifest["entries"][1], manifest["entries"][0]
    elif bad == "source_reused":
        manifest["entries"][1]["source_sha256"] = manifest["entries"][0]["source_sha256"]
    elif bad == "wrong_dino_tag":
        manifest["entries"][10]["tag"] = "EXP160_DINO_BASE"
    elif bad == "wrong_dino_format":
        manifest["entries"][10]["format"] = "unverified"
    elif bad == "entry_hash":
        manifest["entries"][0]["sha256"] = "0"*64
    elif bad == "weight":
        manifest["weights"]["trio"] = .85
    else:
        manifest["family_weights"]["dino"] = .5
    with pytest.raises(ValueError):
        release.verify_assets(serving.work, manifest)


def test_b_anchor_uses_authenticated_raw_probabilities_not_ordinal_csv(anchor):
    ids, values = release.anchor_values(anchor.work, anchor.sample, anchor.manifest)
    assert ids == anchor.ids
    np.testing.assert_array_equal(values, anchor.raw)
    assert not np.array_equal(values, anchor.ordinal)
    np.testing.assert_array_equal(release.rank(values)[:, 0], [.5, .5, 0., 1.])


@pytest.mark.parametrize("bad", ["extra_public", "ids_reordered", "targets_reordered", "nonfinite", "out_of_range", "missing_row"])
def test_b_anchor_rejects_authenticated_but_invalid_raw_roster(anchor, bad):
    with np.load(anchor.work / "raw.npz", allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    if bad == "extra_public":
        arrays["public"] = arrays["own"].copy()
    elif bad == "ids_reordered":
        arrays["ids"] = arrays["ids"][::-1]
    elif bad == "targets_reordered":
        arrays["targets"] = arrays["targets"][::-1]
    elif bad == "missing_row":
        arrays["own"] = arrays["own"][:-1]
    else:
        arrays["own"][0, 0] = np.nan if bad == "nonfinite" else 1.1
    np.savez(anchor.work / "raw.npz", **arrays)
    receipt = release.read_json(anchor.work / "receipt.json")
    receipt["files"]["raw.npz"] = release.sha(anchor.work / "raw.npz")
    _json(anchor.work / "receipt.json", receipt)
    with pytest.raises(ValueError):
        release.anchor_values(anchor.work, anchor.sample, anchor.manifest)


def test_anchor_binding_and_stale_raw_hash_are_rejected(anchor):
    wrong = copy.deepcopy(anchor.manifest)
    wrong["anchor_binding"]["own_sha256"] = "different-checkpoint"
    with pytest.raises(ValueError, match="binding"):
        release.anchor_values(anchor.work, anchor.sample, wrong)
    with (anchor.work / "raw.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Anchor output drift"):
        release.anchor_values(anchor.work, anchor.sample, anchor.manifest)


def test_strict_synthetic_serve_publishes_correct_b_recipe(serving):
    result = _run(serving)
    assert result["status"] == "PASS_BUNDLE"
    assert result["candidate"] == "B172"
    assert result["executed_arms"] == release.expected_arms("B172")
    assert result["qualification"] == "VISIBLE_ONLY"
    ids, actual = release.validate_csv(serving.output / "submission.csv", serving.competition / "sample_submission.csv")
    assert ids == serving.ids
    expected = release.final_values(serving.raw, release.aggregate(serving.scores)[0], "B172")
    np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=0)
    assert result["submission_sha256"] == release.sha(serving.output / "submission.csv")


@pytest.mark.parametrize("exception", [RuntimeError("injected partner failure"), KeyboardInterrupt("injected interrupt")])
def test_strict_partner_failure_removes_submission_and_raises(serving, exception):
    def failing_partner(*args):
        serving.output.mkdir(exist_ok=True)
        (serving.output / "submission.csv").write_text("partial", encoding="utf-8")
        raise exception
    with pytest.raises(type(exception), match="injected"):
        _run(serving, partner=failing_partner)
    assert not (serving.output / "submission.csv").exists()
    failure = release.read_json(serving.output / "exp169-executed-candidate.json")
    assert failure["status"] == "FAILED_BUNDLE" and failure["candidate"] == "B172"


@pytest.mark.parametrize("bad", ["manifest_missing", "wrong_matrix", "pilot_missing", "pilot_failure",
                                "pilot_provenance", "pilot_cohort", "pilot_ram", "stale_submission"])
def test_preflight_failure_emits_failure_and_never_runs_partner(serving, bad):
    if bad == "manifest_missing":
        (serving.work / "release-manifest.json").unlink()
    elif bad == "wrong_matrix":
        serving.manifest["entries"][0]["fold"] = 1
        _json(serving.work / "release-manifest.json", serving.manifest)
    elif bad == "pilot_missing":
        (serving.work / "partner-pilot.json").unlink()
    elif bad.startswith("pilot_"):
        pilot = release.read_json(serving.work / "partner-pilot.json")
        key, value = {"pilot_failure": ("status", "FAILED"),
                      "pilot_provenance": ("manifest_sha256", "stale"),
                      "pilot_cohort": ("full_cohort", 999),
                      "pilot_ram": ("peak_ram_bytes", 0)}[bad]
        pilot[key] = value
        _json(serving.work / "partner-pilot.json", pilot)
    else:
        serving.output.mkdir()
        (serving.output / "submission.csv").write_text("stale", encoding="utf-8")
    calls = []
    def forbidden_partner(*args):
        calls.append(True)
        raise AssertionError("partner ran during preflight failure")
    with pytest.raises((ValueError, FileNotFoundError)):
        _run(serving, partner=forbidden_partner)
    assert calls == []
    assert not (serving.output / "submission.csv").exists()
    assert release.read_json(serving.output / "exp169-executed-candidate.json")["status"] == "FAILED_BUNDLE"


def test_returned_fallback_is_rejected_and_cannot_leave_pass_receipt(serving, monkeypatch):
    original = release.inspect.getsource
    def getsource(fn):
        text = original(fn)
        if fn is release.base.serve:
            text = release.replace_once(text, "    return receipt\n",
                "    receipt.update(status='FAILED_BUNDLE', candidate='TRIO_ONLY_FALLBACK')\n"
                "    return receipt\n")
        return text
    monkeypatch.setattr(release.inspect, "getsource", getsource)
    with pytest.raises(ValueError, match="Returned fallback"):
        _run(serving)
    assert not (serving.output / "submission.csv").exists()
    failure = release.read_json(serving.output / "exp169-executed-candidate.json")
    assert failure["status"] == "FAILED_BUNDLE"
    assert failure["candidate"] == "B172"


def test_backend_reset_reverses_upstream_trio_globals_without_torch():
    backend = SimpleNamespace(benchmark=True, deterministic=False)
    release.reset_backend(SimpleNamespace(backends=SimpleNamespace(cudnn=backend)))
    assert backend.benchmark is False and backend.deterministic is True


def test_injected_pilot_failure_resets_backend_then_propagates(monkeypatch):
    calls = []
    monkeypatch.setattr(release, "reset_backend", lambda: calls.append("reset"))
    def failing_pilot(*args, **kwargs):
        raise RuntimeError("injected pilot failure")
    monkeypatch.setattr(release.base, "pilot", failing_pilot)
    with pytest.raises(RuntimeError, match="injected pilot failure"):
        release.pilot("synthetic-work", "synthetic-competition", "synthetic-out")
    assert calls == ["reset"]


def test_stale_candidate_receipt_prevents_run_and_removes_root_csv(serving):
    serving.output.mkdir()
    _json(serving.output / "exp169-executed-candidate.json", {"status": "PASS_BUNDLE", "candidate": "OLD"})
    (serving.output / "submission.csv").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale candidate receipt"):
        _run(serving)
    assert not (serving.output / "submission.csv").exists()


@pytest.fixture
def staged(tmp_path, monkeypatch):
    root, folder = tmp_path / "root", tmp_path / "stage"
    paths = {"artifact": folder / "kernel/release.ipynb", "source": root / "tools/source.py",
             "proof": root / "proof.json", "note": root / "verification.md"}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic-" + name, encoding="utf-8")
    _json(folder / "assets/release-manifest.json", {"candidate": "B172"})
    archive = folder / "dataset/exp169-assets.zip.bin"
    archive.parent.mkdir()
    archive.write_bytes(b"synthetic opaque archive")
    receipt = dict(version=release.VERSION, status="PLAN_ONLY_NOT_PUSHED", candidate="B172",
                   manifest_sha256=release.sha(folder / "assets/release-manifest.json"),
                   notebook_sha256=release.sha(paths["artifact"]), archive_sha256=release.sha(archive),
                   files={name: release.sha(folder / name) for name in
                          ("kernel/release.ipynb", "assets/release-manifest.json", "dataset/exp169-assets.zip.bin")},
                   sources={"tools/source.py": release.sha(paths["source"])},
                   exp172_proof_files={"proof.json": release.sha(paths["proof"])},
                   verification_note={"path": str(paths["note"]), "sha256": release.sha(paths["note"])})
    _json(folder / "build-receipt.json", receipt)
    monkeypatch.setattr(release, "ROOT", root)
    return SimpleNamespace(root=root, folder=folder, paths=paths, receipt=receipt)


@pytest.mark.parametrize("changed", ["artifact", "source", "proof", "note"])
def test_verify_stage_rejects_stale_artifact_source_proof_or_note(staged, changed):
    folder, paths = staged.folder, staged.paths
    assert release.verify_stage(folder)["candidate"] == "B172"
    paths[changed].write_text("stale synthetic bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale|drift|binding"):
        release.verify_stage(folder)


@pytest.mark.parametrize("field,value", [("candidate", "A172"), ("manifest_sha256", "0"*64),
                                       ("notebook_sha256", "1"*64), ("archive_sha256", "2"*64)])
def test_verify_stage_rejects_top_level_identity_drift(staged, field, value):
    assert release.verify_stage(staged.folder)["candidate"] == "B172"
    receipt = copy.deepcopy(staged.receipt)
    receipt[field] = value
    _json(staged.folder / "build-receipt.json", receipt)
    with pytest.raises(ValueError, match="Stage identity/digest binding"):
        release.verify_stage(staged.folder)


def _v2_checked_output(staged, serving):
    """Wire a synthetic executed B run to the v2 stage receipts."""
    staged_manifest = staged.folder / "assets/release-manifest.json"
    shutil.copyfile(serving.work / "release-manifest.json", staged_manifest)
    receipt = release.read_json(staged.folder / "build-receipt.json")
    receipt["candidate"] = serving.candidate
    receipt["manifest_sha256"] = release.sha(staged_manifest)
    receipt["files"]["assets/release-manifest.json"] = receipt["manifest_sha256"]
    _json(staged.folder / "build-receipt.json", receipt)
    common = dict(version=release.VERSION, candidate=serving.candidate,
                  manifest_sha256=receipt["manifest_sha256"],
                  notebook_sha256=receipt["notebook_sha256"],
                  expected_arms=release.expected_arms(serving.candidate))
    _json(staged.folder / "parity-plan.json", dict(**common,
          schema="exp169-v2-parity-plan-v1", anchor_reference=None,
          anchor_reference_files={}, anchor_tolerance=dict(max_abs=.002, mean_abs=.0005),
          partner_tolerance=2e-5))
    _json(staged.folder / "timing-plan.json", dict(**common,
          schema="exp169-v2-timing-plan-v1", guard_seconds=24300,
          ram_limit_bytes=27*2**30, per_gpu_limit_bytes=15*2**30))
    _run(serving)
    shutil.copytree(serving.anchor, serving.output / "exp169-trio")
    return serving.work / "pilot-scores.npz"


def test_v2_parity_and_timing_accept_own_anchor_schema(staged, serving, tmp_path):
    partner = _v2_checked_output(staged, serving)
    result_path = tmp_path / "parity-result.json"
    with pytest.raises(ValueError, match="exact own-only local anchor"):
        release.compare_parity(staged.folder, serving.output, partner, result_path)
    parity = release.compare_parity(staged.folder, serving.output, partner,
                                    result_path, serving.anchor)
    assert parity["status"] == "PASS_VISIBLE_PARITY_UNQUALIFIED"
    assert parity["anchor_raw"]["own"]["max_abs"] == 0
    assert set(parity["partner_max_abs"]) == {"l2", "dino", "convnext"}
    timing = release.timing_check(staged.folder, serving.output)
    assert timing["status"] == "PASS_TIMING_RECEIPT_UNQUALIFIED"
    assert timing["intended_cohort_proof"] == "PENDING"
    with pytest.raises(ValueError, match="Visible run"):
        release.timing_check(staged.folder, serving.output, require_full=True)


def test_v2_parity_detects_changed_own_probabilities(staged, serving, tmp_path):
    partner = _v2_checked_output(staged, serving)
    with np.load(serving.anchor / "raw.npz", allow_pickle=False) as raw:
        arrays = {name: raw[name].copy() for name in raw.files}
    arrays["own"][0, 0] += .01
    np.savez(serving.anchor / "raw.npz", **arrays)
    receipt = release.read_json(serving.anchor / "receipt.json")
    receipt["files"]["raw.npz"] = release.sha(serving.anchor / "raw.npz")
    _json(serving.anchor / "receipt.json", receipt)
    with pytest.raises(ValueError, match="visible parity failed"):
        release.compare_parity(staged.folder, serving.output, partner,
                               tmp_path / "hold.json", serving.anchor)
    assert release.read_json(tmp_path / "hold.json")["status"] == "HOLD_PARITY"


def _rewrite_anchor_csv(folder, ids, values):
    """Tamper an anchor CSV and refresh its receipt hash (authenticated-but-wrong)."""
    (folder / "submission.csv").unlink()
    _csv(folder / "submission.csv", ids, values)
    receipt = release.read_json(folder / "receipt.json")
    receipt["files"]["submission.csv"] = release.sha(folder / "submission.csv")
    _json(folder / "receipt.json", receipt)


def _recompute_final(serving):
    """Rebuild the kernel final CSV and receipt hash from the kernel anchor."""
    trio = serving.output / "exp169-trio"
    ids, anchor = release.anchor_values(trio, trio / "submission.csv", serving.manifest)
    with np.load(serving.output / "partner-raw.npz", allow_pickle=False) as raw:
        bundle, _ = release.aggregate({name: raw[name] for name in ("l2", "dino", "convnext")})
    (serving.output / "submission.csv").unlink()
    release.base.write_csv(serving.output / "submission.csv", ids,
                           release.final_values(anchor, bundle, serving.candidate), trio / "submission.csv")
    receipt = release.read_json(serving.output / "exp169-executed-candidate.json")
    receipt["submission_sha256"] = release.sha(serving.output / "submission.csv")
    _json(serving.output / "exp169-executed-candidate.json", receipt)


def test_a172_parity_accepts_matching_anchor_csv_by_id(staged, serving_a, tmp_path):
    partner = _v2_checked_output(staged, serving_a)
    parity = release.compare_parity(staged.folder, serving_a.output, partner,
                                    tmp_path / "parity.json", serving_a.anchor)
    assert parity["status"] == "PASS_VISIBLE_PARITY_UNQUALIFIED" and parity["candidate"] == "A172"
    assert set(parity["anchor_raw"]) == {"own", "public", "exp143"}
    assert parity["anchor_csv"] == {"max_abs": 0., "mean_abs": 0.}
    assert parity["anchor_reference_csv_sha256"] == release.sha(serving_a.anchor / "submission.csv")


@pytest.mark.parametrize("candidate", ["A172", "B172"])
@pytest.mark.parametrize("side,bad,message", [
    ("kernel", "values_reversed", "visible parity failed"),
    ("reference", "values_reversed", "visible parity failed"),
    ("kernel", "rows_reversed", "Anchor CSV rows permuted"),
    ("reference", "rows_reversed", "Anchor CSV rows permuted"),
    ("kernel", "id_renamed", "Anchor CSV study IDs differ"),
    ("kernel", "value_beyond_tolerance", "visible parity failed"),
])
def test_parity_rejects_misaligned_anchor_csv_with_preserved_raw_arms(
        staged, tmp_path, candidate, side, bad, message):
    serving = _build_serving(tmp_path / "fixture", candidate)
    partner = _v2_checked_output(staged, serving)
    folder = serving.output / "exp169-trio" if side == "kernel" else serving.anchor
    ids, values = list(serving.ids), serving.ordinal.copy()
    if bad == "values_reversed":
        values = values[::-1]
    elif bad == "rows_reversed":
        ids, values = ids[::-1], values[::-1]
    elif bad == "id_renamed":
        ids[1] = "synthetic-z"
    else:
        values[0, 0] += .003  # registered anchor max_abs tolerance is .002
    _rewrite_anchor_csv(folder, ids, values)
    if side == "kernel" and bad in ("values_reversed", "value_beyond_tolerance"):
        _recompute_final(serving)  # final CSV is self-consistent with the tampered anchor
        assert release.inspect_output(staged.folder, serving.output)["status"] == \
            "EXECUTION_CHECKED_NOT_QUALIFIED"
    with pytest.raises(ValueError, match=message):
        release.compare_parity(staged.folder, serving.output, partner,
                               tmp_path / "hold.json", serving.anchor)
    if message == "visible parity failed":
        hold = release.read_json(tmp_path / "hold.json")
        assert hold["status"] == "HOLD_PARITY"
        assert all(d["max_abs"] == 0 for d in hold["anchor_raw"].values())
        assert hold["anchor_csv"]["max_abs"] > .002
    else:
        assert not (tmp_path / "hold.json").exists()


def test_parity_accepts_anchor_csv_within_tolerance(staged, serving_a, tmp_path):
    partner = _v2_checked_output(staged, serving_a)
    values = serving_a.ordinal.copy()
    values[0, 0] += .001
    _rewrite_anchor_csv(serving_a.anchor, serving_a.ids, values)
    parity = release.compare_parity(staged.folder, serving_a.output, partner,
                                    tmp_path / "parity.json", serving_a.anchor)
    assert parity["status"] == "PASS_VISIBLE_PARITY_UNQUALIFIED"
    assert 0 < parity["anchor_csv"]["max_abs"] <= .002


def test_serving_family_rank_equals_frozen_serving_aggregate():
    scores = _scores(6)
    _, families = release.aggregate(scores)
    for family, members in scores.items():
        np.testing.assert_array_equal(release.serving_family_rank(members, family), families[family])


def test_serving_family_rank_is_probability_mean_then_rank_not_rank_mean():
    s42 = np.tile(np.asarray([.1, .2, .9])[:, None], (1, 12))
    s43 = np.tile(np.asarray([.7, .8, .1])[:, None], (1, 12))
    actual = release.serving_family_rank(np.stack([s42, s43]), "l2")
    np.testing.assert_array_equal(actual, _reference_rank((s42 + s43) / 2))
    np.testing.assert_array_equal(actual[:, 0], [0., .75, .75])
    rank_mean = _reference_rank((_reference_rank(s42) + _reference_rank(s43)) / 2)
    np.testing.assert_array_equal(rank_mean[:, 0], [0., 1., .5])
    assert not np.array_equal(actual, rank_mean)
    np.testing.assert_array_equal(release.serving_family_rank(s42[None], "dino"), _reference_rank(s42))
    # Pooled OOF: per-fold seed-probability means, ranked within fold, then
    # concatenated and globally reranked (no member-rank averaging anywhere).
    folds = [np.stack([s42, s43]), np.stack([s43, s42])[:, :2]]
    pooled = release.rank(np.concatenate([release.serving_family_rank(f, "l2") for f in folds]))
    expected = _reference_rank(np.concatenate([_reference_rank(f.mean(0)) for f in folds]))
    np.testing.assert_array_equal(pooled, expected)


@pytest.mark.parametrize("family,members", [
    ("l2", np.full((3, 2, 12), .5)), ("dino", np.full((1, 2, 11), .5)),
    ("convnext", np.full((1, 2, 12), np.nan)), ("convnext", np.full((1, 2, 12), 1.1)),
    ("radr50", np.full((2, 2, 12), .5)), ("dino", np.empty((0, 2, 12)))])
def test_serving_family_rank_rejects_invalid_members(family, members):
    with pytest.raises(ValueError, match="Invalid serving family members"):
        release.serving_family_rank(members, family)


@pytest.mark.parametrize("ending", ["Verdict: PASS", "Verdict: PASS\n\n",
                                    "Quoted previous Verdict: PASS\nVerdict: FAIL",
                                    "Verdict: PASSIVE", "Verdict: PASS-WITH-NOTES", "> Verdict: PASS"])
def test_verification_note_requires_exact_final_pass_line(tmp_path, monkeypatch, ending):
    source = tmp_path / "synthetic_source.py"
    source.write_text("synthetic source", encoding="utf-8")
    monkeypatch.setattr(release, "source_files", lambda: [source])
    note = tmp_path / "verification.md"
    note.write_text("Source SHA256: " + release.sha(source) + "\n" + ending, encoding="utf-8")
    if ending.rstrip() == "Verdict: PASS":
        assert release.verified_note(note)["sha256"] == release.sha(note)
    else:
        with pytest.raises(ValueError, match="Independent verification PASS"):
            release.verified_note(note)


def test_verification_note_rejects_source_changed_after_pass(tmp_path, monkeypatch):
    source = tmp_path / "synthetic_source.py"
    source.write_text("initial source", encoding="utf-8")
    monkeypatch.setattr(release, "source_files", lambda: [source])
    note = tmp_path / "verification.md"
    note.write_text(release.sha(source) + "\nVerdict: PASS", encoding="utf-8")
    source.write_text("different source", encoding="utf-8")
    with pytest.raises(ValueError, match="Verification did not bind current file"):
        release.verified_note(note)


@pytest.fixture
def training_receipts(tmp_path):
    root = tmp_path / "research"
    folder = root / release.EXP172_REL
    folder.mkdir(parents=True)
    (folder / "PROTOCOL.md").write_text("synthetic training protocol", encoding="utf-8")
    protocol_hash = release.sha(folder / "PROTOCOL.md")
    protocols = {}
    for fold in range(5):
        path = folder / f"outputs/protocols/fold{fold}.json"
        _json(path, {"synthetic": True, "fold": fold})
        protocols[f"protocols/fold{fold}.json"] = {"sha256": release.sha(path)}
    training = {"protocol_sha256": protocol_hash, "targets": release.TARGETS, "files": protocols}
    _json(folder / "outputs/manifest.json", training)
    manifest_hash = release.sha(folder / "outputs/manifest.json")
    _json(folder / "outputs/collection.json", {"status": "VERIFIED", "completed_folds": list(range(5)),
                                               "manifest_sha256": manifest_hash})
    inputs, input_hashes, receipts, validations = {}, {}, [], []
    for fold in range(5):
        run = folder / f"outputs/artifacts/experiments/EXP172_DINO_BASE_L2_S42_F{fold}"
        run.mkdir(parents=True)
        outputs = {}
        for name in ("model_best.pt", "oof_best.csv", "provenance.json", "summary.json"):
            # Opaque synthetic bytes, never label rows or actual checkpoint data.
            (run / name).write_bytes(f"synthetic output {fold} {name}".encode())
            outputs[name] = {"sha256": release.sha(run / name), "bytes": (run / name).stat().st_size}
        receipt = run / "receipt.json"
        fold_protocol = protocols[f"protocols/fold{fold}.json"]["sha256"]
        _json(receipt, {"status": "PASS", "phase": "full", "fold": fold,
                        "protocol_sha256": fold_protocol, "outputs": outputs})
        validated = folder / f"outputs/VALIDATED_SUCCESS-fold{fold}.json"
        _json(validated, {"status": "VALIDATED_SUCCESS", "fold": fold, "manifest_sha256": manifest_hash,
                          "protocol_sha256": fold_protocol,
                          "phases": [{"phase": "full", "receipt_sha256": release.sha(receipt)}]})
        key = f"dino_l2_f{fold}"
        inputs[key] = (run / "oof_best.csv").relative_to(root).as_posix()
        input_hashes[key] = outputs["oof_best.csv"]["sha256"]
        receipts.append(receipt)
        validations.append(validated)
    screen = {"status": {"status": "ADMIT"}, "studies": 4343, "folds": list(range(5)),
              "protocol_sha256": protocol_hash, "input_files": inputs, "input_sha256": input_hashes,
              "new_bundle_minus_d_b_098": {"primary_mean_delta": .01, "primary_mean_ci95": [.001, .02],
                                           "references": {"dread": {"delta": 0.}}}}
    _json(folder / "screen/result.json", screen)
    return SimpleNamespace(root=root, folder=folder, receipts=receipts, validations=validations)


def test_training_receipts_link_five_folds_to_frozen_protocols(training_receipts):
    entries, proofs = release.exp172_entries(training_receipts.root)
    assert [(e["family"], e["seed"], e["fold"]) for e in entries] == [("dino", 42, f) for f in range(5)]
    assert len({e["source_sha256"] for e in entries}) == 5
    assert len(proofs) == 14


def test_self_consistent_fold_receipts_cannot_substitute_another_protocol(training_receipts):
    receipt_path, validated_path = training_receipts.receipts[2], training_receipts.validations[2]
    receipt, validated = release.read_json(receipt_path), release.read_json(validated_path)
    receipt["protocol_sha256"] = validated["protocol_sha256"] = "9"*64
    _json(receipt_path, receipt)
    validated["phases"][0]["receipt_sha256"] = release.sha(receipt_path)
    _json(validated_path, validated)
    with pytest.raises(ValueError, match="EXP172 fold receipt drift"):
        release.exp172_entries(training_receipts.root)


@pytest.mark.parametrize("field,value", [("protocol_sha256", "0"*64), ("targets", release.TARGETS[::-1])])
def test_training_manifest_rejects_protocol_or_target_drift(training_receipts, field, value):
    path = training_receipts.folder / "outputs/manifest.json"
    manifest = release.read_json(path)
    manifest[field] = value
    _json(path, manifest)
    collection_path = training_receipts.folder / "outputs/collection.json"
    collection = release.read_json(collection_path)
    collection["manifest_sha256"] = release.sha(path)
    _json(collection_path, collection)
    with pytest.raises(ValueError, match="training manifest protocol/targets drift"):
        release.exp172_entries(training_receipts.root)


@pytest.mark.parametrize("candidate,existing", [("A172", False), ("A172", True), ("B172", False), ("B172", True)])
def test_notebook_outer_handler_emits_identity_for_injected_pilot_failure(tmp_path, candidate, existing):
    anchor_nb, _, _ = own.notebook("own", "synthetic/b172-own", "1"*64, "2"*64, "3"*64, [])
    verification = {"path": "synthetic-verification.md", "sha256": "6"*64}
    notebook = release.candidate_notebook(candidate, "synthetic/partner-v2", "4"*64, "5"*64,
                                         anchor_nb, verification_note=verification)
    parsed = ast.parse(notebook["cells"][-1]["source"])
    outer = next(node for node in parsed.body if isinstance(node, ast.Try))
    assert len(outer.handlers) == 1 and isinstance(outer.handlers[0].type, ast.Name)
    assert outer.handlers[0].type.id == "BaseException"
    # Execute only the generated failure envelope. Its protected GPU/notebook
    # body is replaced with a synthetic exception; paths are mapped to tmp_path.
    outer.body = ast.parse("pilot()\n").body
    executable = ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[]))
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text("partial synthetic output", encoding="utf-8")
    receipt_path = tmp_path / "exp169-executed-candidate.json"
    if existing:
        _json(receipt_path, {"status": "PASS_BUNDLE", "candidate": "OLD"})
    def mapped_path(value):
        assert value in ("/kaggle/working/submission.csv", "/kaggle/working/exp169-executed-candidate.json")
        return tmp_path / value.rsplit("/", 1)[1]
    def pilot():
        raise RuntimeError("injected pilot failure")
    with pytest.raises(RuntimeError, match="injected pilot failure"):
        exec(compile(executable, "synthetic-notebook-failure-envelope", "exec"), {"Path": mapped_path, "pilot": pilot})
    assert not csv_path.exists()
    failure = release.read_json(receipt_path)
    assert failure["status"] == "FAILED_BUNDLE" and failure["candidate"] == candidate
    assert failure["generator_version"] == release.VERSION
    assert failure["manifest_sha256"] == "5"*64
    assert failure["verification_note"] == verification
    assert failure["error"] == "RuntimeError: injected pilot failure"
