"""EXP169 v3 (B174) offline CPU checks; synthetic data except the read-only
B172 v2.0.2 notebook template and the v1.0.2 replay. No checkpoint is
deserialized and no notebook, GPU or account API is run.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from tools import exp169_release_v2 as release2
from tools import exp169_release_v3 as release3
from tools import exp169_v102_replay as replay
from tests import test_exp169_release_v2 as v2t

ROOT = Path(__file__).resolve().parents[1]
R = v2t._reference_rank


def _anchor174(folder, competition, own, binding):
    """A second B-style own anchor with its own raw probabilities and binding."""
    v2t._save_anchor(folder, competition, "B172")
    with np.load(folder / "raw.npz", allow_pickle=False) as raw:
        arrays = {name: raw[name].copy() for name in raw.files}
    arrays["own"] = own
    (folder / "raw.npz").unlink()
    np.savez(folder / "raw.npz", **arrays)
    (folder / "binding.json").unlink()
    v2t._json(folder / "binding.json", binding)
    receipt = release3.read_json(folder / "receipt.json")
    for name in ("raw.npz", "binding.json"):
        receipt["files"][name] = release3.sha(folder / name)
    v2t._json(folder / "receipt.json", receipt)


def _b174_dir(folder, s, binding174):
    (folder / "code").mkdir(parents=True)
    (folder / "protocols").mkdir()
    shutil.copyfile(release3.__file__, folder / "code/exp169_release_v3.py")
    (folder / "protocols/final-b-rule-v1.md").write_text("synthetic rule\n", encoding="utf-8")
    files = {name: {"sha256": release3.sha(folder / name), "bytes": (folder / name).stat().st_size}
             for name in ("code/exp169_release_v3.py", "protocols/final-b-rule-v1.md")}
    partner = release3.read_json(s.work / "release-manifest.json")
    manifest = dict(schema=release3.SCHEMA, generator_version=release3.VERSION, candidate="B174",
                    weights=dict(release3.RECIPE), anchor_order=list(release3.ANCHOR_ARMS),
                    expected_arms=release3.expected_arms(), verification_note=None,
                    partner=dict(manifest_sha256=release3.sha(s.work / "release-manifest.json"),
                                 v2_source_sha256=release3.sha(s.work / "code/exp169_release_v2.py")),
                    anchors={"EXP143V2": dict(binding=partner["anchor_binding"],
                                              checkpoint_sha256=partner["anchor_binding"]["own_sha256"]),
                             "EXP174": dict(binding=binding174, checkpoint_sha256=binding174["own_sha256"])},
                    files=files)
    v2t._json(folder / "release-manifest.json", manifest)
    return manifest


@pytest.fixture
def b174(tmp_path):
    s = v2t._build_serving(tmp_path, "B172")
    rng = np.random.default_rng(174)
    own174 = rng.random((4, 12))
    binding174 = {"variant": "own", "own_sha256": "synthetic-own-174", "public_sha256": None}
    _anchor174(tmp_path / "anchor174", s.competition, own174, binding174)
    manifest = _b174_dir(tmp_path / "b174", s, binding174)
    return SimpleNamespace(**vars(s), anchor174=tmp_path / "anchor174", b174=tmp_path / "b174",
                           own174=own174, b174_manifest=manifest, tmp=tmp_path)


def _run3(s, output=None, **kwargs):
    return release3.serve(s.work, s.b174, s.anchor, s.anchor174, output or s.output, s.competition,
                          clock=lambda: 100., partner_scores=kwargs.get("partner", s.partner))


# ---------------------------------------------------------------- recipe

def test_recipe_and_arms_are_fixed():
    assert release3.RECIPE == {"EXP143V2": .35, "EXP174": .35, "bundle": .30}
    assert list(release3.RECIPE) == ["EXP143V2", "EXP174", "bundle"]
    assert release3.expected_arms() == ["EXP143V2", "EXP174", "L2_RADR50_10",
                                        "EXP172_L2_DINO_BASE_5", "EXP170_L2_CONVNEXT_T_5"]
    assert release2.RECIPES["B172"] == {"anchor": "EXP143V2", "anchor_weight": .70, "bundle_weight": .30}


def test_final_values_are_fixed_weighted_ranks():
    rng = np.random.default_rng(1)
    a143, a174, q = (rng.random((7, 12)) for _ in range(3))
    a143[:2] = .5  # average ties
    expected = (.35 * R(a143) + .35 * R(a174)) + .30 * R(q)
    np.testing.assert_array_equal(release3.final_values(a143, a174, q), expected)


def test_b174_differs_from_b172_only_in_the_anchor_arm():
    rng = np.random.default_rng(2)
    a143, a174, q = (rng.random((9, 12)) for _ in range(3))
    b172 = release2.final_values(a143, q, "B172")
    # Identical anchors reproduce B172 bit-for-bit (.35 + .35 == .70 exactly in binary).
    np.testing.assert_array_equal(release3.final_values(a143, a143, q), b172)
    # Otherwise the difference is exactly the EXP174 half of the anchor weight.
    np.testing.assert_allclose(release3.final_values(a143, a174, q) - b172,
                               .35 * (R(a174) - R(a143)), atol=1e-15, rtol=0)


@pytest.mark.parametrize("bad", ["nan", "above_one", "eleven_targets", "row_mismatch", "empty"])
def test_final_values_reject_invalid_inputs(bad):
    good = np.full((4, 12), .5)
    arrays = [good.copy(), good.copy(), good.copy()]
    if bad == "nan":
        arrays[1][0, 0] = np.nan
    elif bad == "above_one":
        arrays[2][0, 0] = 1.01
    elif bad == "eleven_targets":
        arrays[1] = arrays[1][:, :11]
    elif bad == "row_mismatch":
        arrays[0] = arrays[0][:3]
    else:
        arrays = [np.empty((0, 12))] * 3
    with pytest.raises(ValueError):
        release3.final_values(*arrays)


# ---------------------------------------------------------------- serving

def test_strict_serve_publishes_b174_recipe(b174):
    result = _run3(b174)
    assert result["status"] == "PASS_BUNDLE" and result["candidate"] == "B174"
    assert result["executed_arms"] == release3.expected_arms()
    assert result["weights"] == release3.RECIPE and result["qualification"] == "VISIBLE_ONLY"
    assert result["trio_seconds"] == 20.  # both anchors' receipts (10 s each)
    assert set(result["anchors"]) == {"EXP143V2", "EXP174"}
    ids, actual = release3.validate_csv(b174.output / "submission.csv", b174.competition / "sample_submission.csv")
    assert ids == b174.ids
    expected = release3.final_values(b174.raw, b174.own174, release3.aggregate(b174.scores)[0])
    np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=0)


def test_b174_and_b172_share_the_partner_bundle_and_agree_for_equal_anchors(b174):
    b172 = release2.serve(b174.work, b174.anchor, b174.tmp / "out172", b174.competition,
                          clock=lambda: 100., partner=b174.partner)
    first = _run3(b174)
    for key in ("family_rank_sha256", "bundle_rank_sha256"):
        assert first[key] == b172[key]
    assert first["partner_raw_sha256"] == b172["partner_raw_sha256"]
    assert release3.sha(b174.output / "submission.csv") != release3.sha(b174.tmp / "out172/submission.csv")
    # Same raw values in the EXP174 arm (distinct checkpoint binding) -> byte-identical B172 CSV.
    shutil.rmtree(b174.anchor174)
    _anchor174(b174.anchor174, b174.competition, b174.raw.copy(),
               b174.b174_manifest["anchors"]["EXP174"]["binding"])
    _run3(b174, output=b174.tmp / "out174-equal")
    assert (b174.tmp / "out174-equal/submission.csv").read_bytes() == (b174.tmp / "out172/submission.csv").read_bytes()


@pytest.mark.parametrize("bad", ["anchor174_rows_reversed", "anchor174_binding", "same_checkpoint",
                                 "anchor143_binding", "anchor174_missing"])
def test_anchor_wiring_rejects_misbinding_and_order(b174, bad):
    manifest = copy.deepcopy(b174.b174_manifest)
    if bad == "anchor174_rows_reversed":
        with np.load(b174.anchor174 / "raw.npz", allow_pickle=False) as raw:
            arrays = {name: raw[name].copy() for name in raw.files}
        arrays["ids"], arrays["own"] = arrays["ids"][::-1], arrays["own"][::-1]
        (b174.anchor174 / "raw.npz").unlink()
        np.savez(b174.anchor174 / "raw.npz", **arrays)
        (b174.anchor174 / "submission.csv").unlink()
        v2t._csv(b174.anchor174 / "submission.csv", b174.ids[::-1], np.zeros((4, 12)))
        receipt = release3.read_json(b174.anchor174 / "receipt.json")
        for name in ("raw.npz", "submission.csv"):
            receipt["files"][name] = release3.sha(b174.anchor174 / name)
        v2t._json(b174.anchor174 / "receipt.json", receipt)
    elif bad == "anchor174_binding":
        manifest["anchors"]["EXP174"]["binding"]["own_sha256"] = "another-checkpoint"
    elif bad == "same_checkpoint":
        manifest["anchors"]["EXP174"]["binding"] = copy.deepcopy(manifest["anchors"]["EXP143V2"]["binding"])
    elif bad == "anchor143_binding":
        manifest["anchors"]["EXP143V2"]["binding"]["own_sha256"] = "not-exp143"
    else:
        (b174.anchor174 / "raw.npz").unlink()
    partner = release3.read_json(b174.work / "release-manifest.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        release3.anchor_values(b174.anchor, b174.anchor174, b174.competition / "sample_submission.csv",
                               manifest, partner)


@pytest.mark.parametrize("bad", ["weights", "weight_order", "candidate", "anchor_order", "version",
                                 "file_hash", "wrapper", "partner_manifest", "binding_public"])
def test_verify_assets_rejects_recipe_or_binding_drift(b174, bad):
    partner = release3.read_json(b174.work / "release-manifest.json")
    assert release3.verify_assets(b174.b174, b174.b174_manifest, b174.work, partner) is True
    manifest = copy.deepcopy(b174.b174_manifest)
    if bad == "weights":
        manifest["weights"]["EXP174"] = .30
    elif bad == "weight_order":
        manifest["weights"] = {"EXP174": .35, "EXP143V2": .35, "bundle": .30}
    elif bad == "candidate":
        manifest["candidate"] = "B172"
    elif bad == "anchor_order":
        manifest["anchor_order"] = ["EXP174", "EXP143V2"]
    elif bad == "version":
        manifest["generator_version"] = "2.0.2"
    elif bad == "file_hash":
        (b174.b174 / "protocols/final-b-rule-v1.md").write_text("changed", encoding="utf-8")
    elif bad == "wrapper":
        with (b174.b174 / "code/exp169_release_v3.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# drift\n")
        manifest["files"]["code/exp169_release_v3.py"] = {
            "sha256": release3.sha(b174.b174 / "code/exp169_release_v3.py"),
            "bytes": (b174.b174 / "code/exp169_release_v3.py").stat().st_size}
    elif bad == "partner_manifest":
        manifest["partner"]["manifest_sha256"] = "0" * 64
    else:
        manifest["anchors"]["EXP174"]["binding"]["public_sha256"] = "public"
    with pytest.raises(ValueError):
        release3.verify_assets(b174.b174, manifest, b174.work, partner)


def test_partner_failure_removes_submission_and_records_b174(b174):
    def failing(*args):
        b174.output.mkdir(exist_ok=True)
        (b174.output / "submission.csv").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected partner failure")
    with pytest.raises(RuntimeError, match="injected"):
        _run3(b174, partner=failing)
    assert not (b174.output / "submission.csv").exists()
    failure = release3.read_json(b174.output / "exp169-executed-candidate.json")
    assert failure["status"] == "FAILED_BUNDLE" and failure["candidate"] == "B174"
    assert failure["generator_version"] == release3.VERSION


# ---------------------------------------------------------------- downloaded-run checks

def _stage_from(b174, monkeypatch):
    """Minimal synthetic v3 stage around the synthetic serve output."""
    stage = b174.tmp / "stage"
    for rel, src in {"code/assets/release-manifest.json": b174.b174 / "release-manifest.json",
                     "reused/b172-release-manifest.json": b174.work / "release-manifest.json"}.items():
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, stage / rel)
    for rel, text in {"kernel/release.ipynb": "synthetic notebook",
                      "anchor174/dataset/" + release3.ANCHOR174_ARCHIVE: "synthetic anchor archive",
                      "code/dataset/" + release3.CODE_ARCHIVE: "synthetic code archive"}.items():
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        (stage / rel).write_text(text, encoding="utf-8")
    build = dict(version=release3.VERSION, candidate="B174", status="PLAN_ONLY_UNVERIFIED_NOT_PUSHABLE",
                 verification_note=None, sources={},
                 manifest_sha256=release3.sha(stage / "code/assets/release-manifest.json"),
                 notebook_sha256=release3.sha(stage / "kernel/release.ipynb"),
                 anchor174_archive_sha256=release3.sha(stage / "anchor174/dataset" / release3.ANCHOR174_ARCHIVE),
                 code_archive_sha256=release3.sha(stage / "code/dataset" / release3.CODE_ARCHIVE))
    common = dict(version=release3.VERSION, candidate="B174", manifest_sha256=build["manifest_sha256"],
                  notebook_sha256=build["notebook_sha256"], expected_arms=release3.expected_arms())
    v2t._json(stage / "parity-plan.json", dict(**common, schema="exp169-v3-parity-plan-v1",
              anchor_tolerance=dict(max_abs=.002, mean_abs=.0005), partner_tolerance=2e-5))
    v2t._json(stage / "timing-plan.json", dict(**common, schema="exp169-v3-timing-plan-v1", guard_seconds=24300,
              ram_limit_bytes=27 * 2**30, per_gpu_limit_bytes=15 * 2**30))
    build["files"] = {p.relative_to(stage).as_posix(): release3.sha(p) for p in stage.rglob("*") if p.is_file()}
    v2t._json(stage / "build-receipt.json", build)
    monkeypatch.setattr(release3, "ROOT", b174.tmp)
    _run3(b174)
    shutil.copytree(b174.anchor, b174.output / "exp169-trio")
    shutil.copytree(b174.anchor174, b174.output / "exp169-anchor174")
    return stage


def test_inspect_parity_and_timing_accept_matching_run(b174, monkeypatch):
    stage = _stage_from(b174, monkeypatch)
    assert release3.inspect_output(stage, b174.output)["status"] == "EXECUTION_CHECKED_NOT_QUALIFIED"
    parity = release3.compare_parity(stage, b174.output, b174.work / "pilot-scores.npz",
                                     b174.anchor, b174.anchor174, b174.tmp / "parity.json")
    assert parity["status"] == "PASS_VISIBLE_PARITY_UNQUALIFIED"
    assert all(a["raw"]["max_abs"] == 0 and a["csv"]["max_abs"] == 0 for a in parity["anchors"].values())
    timing = release3.timing_check(stage, b174.output)
    assert timing["status"] == "PASS_TIMING_RECEIPT_UNQUALIFIED" and timing["anchor_seconds"] == 20.


@pytest.mark.parametrize("bad,error", [("anchor174_csv_reversed", "visible parity failed"),
                                       ("anchor174_raw_changed", "visible parity failed"),
                                       ("anchors_swapped", "binding differs"),
                                       ("final_csv_from_b172", "Final blend does not match")])
def test_parity_rejects_tampered_or_swapped_anchors(b174, monkeypatch, bad, error):
    stage = _stage_from(b174, monkeypatch)
    anchor143, anchor174 = b174.anchor, b174.anchor174
    if bad == "anchor174_csv_reversed":
        v2t_values = np.tile(np.asarray([1 / 3, 2 / 3, 0., 1.])[:, None], (1, 12))[::-1]
        folder = b174.output / "exp169-anchor174"
        (folder / "submission.csv").unlink()
        v2t._csv(folder / "submission.csv", b174.ids, v2t_values)
        receipt = release3.read_json(folder / "receipt.json")
        receipt["files"]["submission.csv"] = release3.sha(folder / "submission.csv")
        v2t._json(folder / "receipt.json", receipt)
    elif bad == "anchor174_raw_changed":
        with np.load(anchor174 / "raw.npz", allow_pickle=False) as raw:
            arrays = {name: raw[name].copy() for name in raw.files}
        arrays["own"][0, 0] = min(1., arrays["own"][0, 0] + .01)
        (anchor174 / "raw.npz").unlink()
        np.savez(anchor174 / "raw.npz", **arrays)
        receipt = release3.read_json(anchor174 / "receipt.json")
        receipt["files"]["raw.npz"] = release3.sha(anchor174 / "raw.npz")
        v2t._json(anchor174 / "receipt.json", receipt)
    elif bad == "anchors_swapped":
        anchor143, anchor174 = anchor174, anchor143
    else:  # a B172 final CSV relabelled as the B174 output
        release2.serve(b174.work, b174.anchor, b174.tmp / "out172", b174.competition,
                       clock=lambda: 100., partner=b174.partner)
        shutil.copyfile(b174.tmp / "out172/submission.csv", b174.output / "submission.csv")
        receipt = release3.read_json(b174.output / "exp169-executed-candidate.json")
        receipt["submission_sha256"] = release3.sha(b174.output / "submission.csv")
        v2t._json(b174.output / "exp169-executed-candidate.json", receipt)
    with pytest.raises(ValueError, match=error):
        release3.compare_parity(stage, b174.output, b174.work / "pilot-scores.npz",
                                anchor143, anchor174, b174.tmp / "hold.json")


# ---------------------------------------------------------------- notebook

@pytest.fixture(scope="module")
def notebook():
    from tools import exp174_own0949_jev_release as e174r
    template_path = ROOT / release3.B172_STAGE_REL / "kernel/release.ipynb"
    assert release3.sha(template_path) == release3.B172_NOTEBOOK_SHA
    anchor_nb, _, _ = e174r.notebook("own", release3.ANCHOR174_DATASET, "1" * 64, "2" * 64, "3" * 64, [])
    members = ["code/exp169_release_v3.py", "protocols/final-b-rule-v1.md", "release-manifest.json"]
    verification = {"path": "synthetic-verification.md", "sha256": "7" * 64}
    nb = release3.b174_notebook(release3.read_json(template_path), "".join(anchor_nb["cells"][-1]["source"]),
                                "4" * 64, "5" * 64, members, "6" * 64, verification)
    return SimpleNamespace(nb=nb, source=nb["cells"][-1]["source"], verification=verification)


def test_notebook_runs_both_anchors_then_partner_then_b174_serve(notebook):
    source = notebook.source
    ast.parse(source)
    order = [source.index(release3.ANCHOR143_ARCHIVE_SHA), source.index("_work143 = _work"),
             source.index("import exp174_own0949_jev_release as _e152"),
             source.index("_work = Path('/kaggle/working/exp169-anchor174')"),
             source.index("_work174 = _work"), source.index("_work = _work143"),
             source.index(release3.PARTNER_ARCHIVE_SHA), source.index("import exp169_release_v3 as _e169v3"),
             source.index("_e169.pilot("), source.index("_e169v3.serve(_partner, _b174, _work, _work174,")]
    assert order == sorted(order)
    for digest in ("1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64, "6" * 64,
                   release3.PARTNER_MANIFEST_SHA, release3.ANCHOR143_MANIFEST_SHA):
        assert digest in source
    assert "_e169.serve(" not in source and source.count("main()") == 2
    assert "_work.mkdir(exist_ok=False)" in source.split("_work174 = _work")[0].split("_work143 = _work")[1]
    assert "+ _e169.read_json(_work174/'receipt.json')['seconds']" in source
    for forbidden in ("_public_mounts", "rsna-knee-publication-swa-weights", "knee-exp155-own0949-oai"):
        assert forbidden not in source
    assert "'code/exp169_release_v3.py', 'protocols/final-b-rule-v1.md', 'release-manifest.json'" in source
    nb = notebook.nb
    assert len(nb["cells"]) == 3 and "Credits and required acknowledgement" in "".join(nb["cells"][1]["source"])
    assert nb["cells"][0]["source"].startswith("# EXP169 B174 / generator 3.0.0")
    assert ".35 r(EXP143v2 raw) + .35 r(EXP174 raw) + .30 Q" in nb["cells"][0]["source"]


def test_notebook_failure_envelope_records_b174(tmp_path, notebook):
    parsed = ast.parse(notebook.source)
    outer = next(node for node in parsed.body if isinstance(node, ast.Try))
    assert len(outer.handlers) == 1 and outer.handlers[0].type.id == "BaseException"
    outer.body = ast.parse("pilot()\n").body
    executable = ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[]))
    (tmp_path / "submission.csv").write_text("partial", encoding="utf-8")

    def mapped_path(value):
        assert value in ("/kaggle/working/submission.csv", "/kaggle/working/exp169-executed-candidate.json")
        return tmp_path / value.rsplit("/", 1)[1]

    def pilot():
        raise RuntimeError("injected pilot failure")
    with pytest.raises(RuntimeError, match="injected pilot failure"):
        exec(compile(executable, "b174-failure-envelope", "exec"), {"Path": mapped_path, "pilot": pilot})
    assert not (tmp_path / "submission.csv").exists()
    failure = release3.read_json(tmp_path / "exp169-executed-candidate.json")
    assert failure["candidate"] == "B174" and failure["generator_version"] == release3.VERSION
    assert failure["manifest_sha256"] == "5" * 64 and failure["verification_note"] == notebook.verification


def test_unwrap_round_trips_the_b172_envelope():
    template = release3.read_json(ROOT / release3.B172_STAGE_REL / "kernel/release.ipynb")
    inner = release3._unwrap(template["cells"][-1]["source"])
    rewrapped = release3._wrap(inner, "x", None)
    original = template["cells"][-1]["source"]
    marker = "except BaseException as _failure_error:\n"
    assert rewrapped.split(marker)[0] == original.split(marker)[0]


def test_qualified_v102_notebook_still_reproduces_exact_bytes():
    actual = replay.regenerate(ROOT)
    target = ROOT / replay.BASELINE_STAGE_PATH / "kernel/release.ipynb"
    assert actual == target.read_bytes()
    assert hashlib.sha256(actual).hexdigest() == replay.QUALIFIED_NOTEBOOK_SHA256 == release2.QUALIFIED_SHA
