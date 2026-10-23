"""C85-R7 CPU gates: notebook-derived pins, R7 port fail-closed checks, C85 v1.0.4 equivalence.

The public R7 notebook is read as JSON/AST literals only; none of its cells is executed.
Real R7 weights are not local: bundle tests use synthetic stand-in files.
"""
from __future__ import annotations

import ast
import builtins
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import exp169_release_c85 as c85
from tools import exp169_release_c85r7 as m
from tools import exp169_r7_port as r7
from tools import exp189_s6_port as s6

NB_DIR = m.ROOT / ".local/external/heliosli-r7-20261013"
NB = NB_DIR / "0-957-latest-dataset-score-in-one-click.ipynb"
S6_MANIFEST = m.ROOT / ".local/idea-scout/20261011-1030/s6-small/manifest.json"


def cells(path):
    return ["".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def literals(cell, names=("ARM", "BUNDLE_ID", "CODE_SLUG", "C_SLUG", "CODE_SHA", "SOURCE_SHA", "SOURCES", "PROBE_CLASS")):
    return {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(cell).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- notebook (data only)

def test_vendored_sources_are_the_notebook_sources_byte_for_byte():
    assert m.sha(NB) == r7.NOTEBOOK["sha256"] and m.sha(NB_DIR / "kernel-metadata.json") == r7.NOTEBOOK["metadata_sha256"]
    nb = cells(NB)
    assert [digest(c) for c in nb[2:]] == [r7.NOTEBOOK[k] for k in (
        "r_cell_sha256", "r_stage_cell_sha256", "s_cell_sha256", "s_run_cell_sha256", "blend_cell_sha256")]
    values = literals(nb[4])
    assert values["BUNDLE_ID"] == r7.BUNDLE_ID and values["ARM"] == r7.ARM
    assert values["C_SLUG"] == r7.DATASET_SLUG.split("/")[1] and values["CODE_SLUG"] == r7.CODE_SLUG.split("/")[1]
    assert r7.CODE_SLUG == m.v5q.CODE_SLUG
    assert values["SOURCE_SHA"] == r7.SOURCE_HASHES
    assert values["CODE_SHA"] == s6.CODE_HASHES  # same mainline code dataset pins as S6
    assert values["PROBE_CLASS"] is None
    for name, text in values["SOURCES"].items():
        assert (r7.SOURCE / name).read_bytes() == text.encode("utf-8")
    assert sorted(values["SOURCES"]) == sorted(p.name for p in r7.SOURCE.glob("*.py"))
    r7.verify_source()
    meta = json.loads((NB_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == [r7.DATASET_SLUG, m.p.DATASETS[0], m.v5q.CODE_SLUG]
    assert meta["docker_image"] == m.p.IMAGE and meta["machine_shape"] == m.p.MACHINE


def test_r_branch_and_blend_cells_identical_to_p956_public_reference():
    nb, ref = cells(NB), cells(m.p.PUBLIC)
    m.p.check_pin(m.p.PUBLIC, m.p.PUBLIC_SHA)
    assert nb[2] == ref[3] and nb[3] == ref[4] and nb[5] == ref[6] and nb[6] == ref[7]
    assert digest(nb[6]) == m.p.CELL_SHA
    # The S stage code differs from P956's public reference only in the bundle identity.
    changed = [(a, b) for a, b in zip(ref[5].splitlines(), nb[4].splitlines()) if a != b]
    assert changed == [("BUNDLE_ID = '209a32c8b50a6b1c'", "BUNDLE_ID = '8c7b666e27dc0e8f'"),
                       ("C_SLUG = 'rsna-knee-compact-r6cnxfsss'", "C_SLUG = 'rsna-knee-compact-r7cnxfsss'")]


# ---------------------------------------------------------------- R7 port

def declared_flags(path):
    flags = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            flags.update(a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--"))
    return flags


def test_diagnostic_patches_apply_once_and_remove_deadlines(tmp_path):
    runtime = r7.diagnostic_source(r7.SOURCE, tmp_path)
    pins = json.loads((runtime / "runtime_sha256.json").read_text())
    assert pins["compact_infer.py"] == "7cc97837dfd7b41d9604b6f43a6340bb7c34fa9708878a930334cc7e0cdf8b0e"
    assert pins["probe_dual.py"] == "4d38193af74a8cb47595356f30a8646b4c7888369715196366acfc3360420359"
    for name in ("compact_train.py", "px_cache.py", "probe_infer.py", "knee_anchor_bank_v2.py", "side_common.py"):
        assert pins[name] == r7.SOURCE_HASHES[name]
    worker = (runtime / "compact_infer.py").read_text(encoding="utf-8")
    controller = (runtime / "probe_dual.py").read_text(encoding="utf-8")
    for token in ("deadline", "timed_out", "item_timeout", "item-timeout"):
        assert token not in worker
    assert "deadline" not in controller and "%.9g" in controller
    assert 'device=a.device)' in worker and "traceback.format_exc()" in worker
    assert "--device" in declared_flags(runtime / "compact_infer.py") and "--device" in declared_flags(runtime / "probe_dual.py")
    r7.verify_source()  # originals untouched
    with pytest.raises(ValueError, match="expected once"):
        r7.diagnostic_source(m.ROOT / "tools/exp189_s6_src", tmp_path / "s6")


def test_worker_args_on_s6_manifest_equal_exp189_fixed_args(tmp_path):
    m.p.check_pin(S6_MANIFEST, s6.MANIFEST_SHA256)
    manifest = json.loads(S6_MANIFEST.read_text(encoding="utf-8"))
    bundle, anatomy = tmp_path / "bundle", tmp_path / "anatomy.pt"
    # exp189_s6_port.infer_studies' fixed --worker-args for this manifest (lines 659-666).
    assert r7.worker_args(manifest, bundle, anatomy) == [
        "--backbone", "convnext_small", "--size", "384", "--crop", "352", "--eval-stride", "1", "--views", "v4",
        "--state-spec", str(bundle / "state_spec.json"), "--side-model", str(bundle / "side_cls.pt"),
        "--side-threshold", "2.0", "--side-position-mm", "20.0", "--side-max-x", "0.0",
        "--slices", "3", "--phys-mm", "3.5", "--anchor-weights", str(anatomy)]


def r7_manifest(digests, **extra):
    value = dict(bundle_id=r7.BUNDLE_ID, arm=r7.ARM, models=list(r7.MODELS), sha256=digests,
                 backbone="convnext_small", size=384, crop=352, eval_stride=1, views="v4", slices=3, phys_mm=3.5,
                 state_spec="state_spec.json", side_model="side_cls.pt", side_threshold=2.0, side_position_mm=20.0,
                 side_max_x=0.0, anchor_weights_sha256=s6.ANATOMY_SHA256, fluid_stacker="fluid_stacker.json",
                 state_stacker="state_stacker.json", flip_tta=True, models2=None)
    value.update(extra)
    return value


def make_bundle(directory, **extra):
    directory.mkdir(parents=True)
    digests = {}
    for i, name in enumerate(r7.ROSTER[1:]):
        (directory / name).write_bytes(f"synthetic stand-in {i} {name}".encode())
        digests[name] = m.sha(directory / name)
    manifest = r7_manifest(digests, **extra)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def test_worker_args_full_r7_feature_set_and_flags_are_declared(tmp_path):
    bundle, anatomy = tmp_path / "b", tmp_path / "anatomy.pt"
    manifest = r7_manifest({}, trainer_md5="0" * 32, compile=True)
    args = r7.worker_args(manifest, bundle, anatomy)
    assert args == ["--backbone", "convnext_small", "--size", "384", "--crop", "352", "--eval-stride", "1", "--views", "v4",
        "--state-spec", str(bundle / "state_spec.json"), "--side-model", str(bundle / "side_cls.pt"),
        "--side-threshold", "2.0", "--side-position-mm", "20.0", "--side-max-x", "0.0", "--slices", "3", "--phys-mm", "3.5",
        "--trainer-md5", "0" * 32, "--compile", "--fluid-stacker", str(bundle / "fluid_stacker.json"), "--flip-tta",
        "--state-stacker", str(bundle / "state_stacker.json"), "--anchor-weights", str(anatomy)]
    minimal = r7.worker_args(dict(backbone="convnext_small", size=384, crop=352), bundle, anatomy)
    assert minimal == ["--backbone", "convnext_small", "--size", "384", "--crop", "352", "--eval-stride", "3",
                       "--views", "v1", "--slices", "3", "--phys-mm", "0.0"]
    runtime = r7.diagnostic_source(r7.SOURCE, tmp_path)
    assert {a for a in args if a.startswith("--")} <= declared_flags(runtime / "compact_infer.py")
    for bad in (dict(models2=["x.pt"]), dict(backbone="ofl_l")):
        with pytest.raises(ValueError, match="unsupported"):
            r7.worker_args(dict(manifest, **bad), bundle, anatomy)


def test_synthetic_bundle_identity_and_copy(tmp_path):
    manifest = make_bundle(tmp_path / "bundle")
    (tmp_path / "bundle/E_lrprior_s1_f0.pt").write_bytes(b"undeclared extra model")
    loaded, identity = r7.verify_bundle(tmp_path / "bundle")
    assert loaded == manifest and identity["bundle_id"] == r7.BUNDLE_ID and not identity["manifest_pinned"]
    assert identity["manifest_sha256"] == m.sha(tmp_path / "bundle/manifest.json")
    assert set(identity["file_sha256"]) == set(r7.ROSTER)
    assert identity["features"]["flip_tta"] is True and identity["features"]["state_stacker"] == "state_stacker.json"
    pinned = r7.verify_bundle(tmp_path / "bundle", identity["manifest_sha256"])[1]
    assert pinned["manifest_pinned"] is True
    copy = r7.prepare_bundle(tmp_path / "bundle", tmp_path / "copy", manifest)
    assert sorted(p.name for p in copy.iterdir()) == sorted(r7.declared_files(manifest))
    assert sorted(copy.glob(r7.ARM + "_s*_f*.pt")) == [copy / name for name in r7.MODELS]
    with pytest.raises(FileExistsError):
        r7.prepare_bundle(tmp_path / "bundle", tmp_path / "copy", manifest)


@pytest.mark.parametrize("name", r7.ROSTER)
def test_missing_r7_file_fails_closed(tmp_path, name):
    make_bundle(tmp_path / "bundle")
    (tmp_path / "bundle" / name).unlink()
    with pytest.raises(ValueError, match="R7 bundle files missing: " + name.replace(".", r"\.")):
        r7.verify_bundle(tmp_path / "bundle")


@pytest.mark.parametrize("fault", ["bundle_id", "roster", "models2", "ofl_l", "hash", "no digest", "anchor",
                                   "trainer md5", "trainer sha", "manifest pin", "unsafe name", "no crop"])
def test_r7_identity_faults_fail_closed(tmp_path, fault):
    manifest = make_bundle(tmp_path / "bundle")
    pin = None
    md5 = hashlib.md5((r7.SOURCE / "compact_train.py").read_bytes()).hexdigest()
    if fault == "bundle_id": manifest["bundle_id"] = "209a32c8b50a6b1c"
    if fault == "roster": manifest["models"] = manifest["models"][:3]
    if fault == "models2": manifest["models2"] = ["E_lrprior_s1_f0.pt"]
    if fault == "ofl_l": manifest["backbone"] = "ofl_l"
    if fault == "hash": (tmp_path / "bundle/side_cls.pt").write_bytes(b"tampered")
    if fault == "no digest": manifest["sha256"].pop("state_stacker.json")
    if fault == "anchor": manifest["anchor_weights_sha256"] = "0" * 64
    if fault == "trainer md5": manifest.update(trainer_md5="0" * 32, src_sha256={"compact_train.py": r7.SOURCE_HASHES["compact_train.py"]})
    if fault == "trainer sha": manifest.update(trainer_md5=md5, src_sha256={"compact_train.py": s6.SOURCE_HASHES["compact_train.py"]})
    if fault == "manifest pin": pin = "0" * 64
    if fault == "unsafe name": manifest["state_spec"] = "../state_spec.json"
    if fault == "no crop": manifest.pop("crop")
    (tmp_path / "bundle/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        r7.verify_bundle(tmp_path / "bundle", pin)
    if fault == "trainer sha":  # the matching pair passes
        manifest["src_sha256"] = {"compact_train.py": r7.SOURCE_HASHES["compact_train.py"]}
        (tmp_path / "bundle/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        assert r7.verify_bundle(tmp_path / "bundle")[1]["features"]["trainer_md5"] == md5


def competition(root, ids):
    rows = []
    for uid in ids:
        series = root / "test_series" / uid / "ser1"
        series.mkdir(parents=True)
        (series / "1.dcm").write_bytes(b"not a real DICOM; staging only")
        rows.append([uid, "ser1", "Sagittal", "True"])
    with (root / "test_series.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"])
        writer.writerows(rows)
    return [root / "test_series" / uid for uid in ids]


def fake_controller(seen, fail_uid=None):
    def run(args, env):
        get = lambda flag: args[args.index(flag) + 1]
        data, out, bundle = Path(get("--data")), Path(get("--output")), Path(get("--bundle"))
        seen.update(args=list(args), env=dict(env), bundle_files=sorted(p.name for p in bundle.iterdir()),
                    worker_args=json.loads(get("--worker-args")))
        ids = [row["StudyInstanceUID"] for row in csv.DictReader((data / "test.csv").open(encoding="utf-8"))]
        rng = np.random.default_rng(7)
        values = {uid: rng.random(12) for uid in ids}
        workers = []
        for gi, part in enumerate([ids[0::2], ids[1::2]]):
            wdir = out / f"worker{gi}"
            wdir.mkdir(parents=True)
            (wdir / "ids.json").write_text(json.dumps(part))
            fallback = {u: "gpu: injected" for u in part if u == fail_uid}
            with (wdir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([s6.UID, *s6.LABELS])
                w.writerows([[u] + ["%.9g" % x for x in values[u]] for u in part if u not in fallback])
            receipt = dict(status="PASS", models=list(r7.MODELS), studies=len(part), scored=len(part) - len(fallback),
                           fallback=fallback, stats=dict(decode_s=1, render_s=1, anchor_s=1, gpu_s=1, rows=len(part) - len(fallback)),
                           anchor_methods={"skm_tea_joint": len(part) - len(fallback)})
            (wdir / "receipt.json").write_text(json.dumps(receipt))
            workers.append(receipt)
        scored = [u for u in ids if u != fail_uid]
        mean = np.mean([values[u] for u in scored], axis=0)
        with (out / "submission.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([s6.UID, *s6.LABELS])
            w.writerows([[u] + ["%.9g" % x for x in (values[u] if u in scored else mean)] for u in ids])
        (out / "submission_receipt.json").write_text(json.dumps(dict(
            status="PASS", rows=len(ids), scored=len(scored), fallback_studies=len(ids) - len(scored),
            fallback_reasons={u: "gpu: injected" for u in ids if u not in scored}, worker_exit_codes={"0": 0, "1": 0},
            gpus=["Tesla T4", "Tesla T4"], workers=workers)))
    return run


@pytest.mark.parametrize("fail_uid", [None, "s1"])
def test_infer_studies_orchestration_mocked_controller(tmp_path, monkeypatch, fail_uid):
    ids = ["s0", "s1", "s2"]
    paths = competition(tmp_path / "comp", ids)
    manifest = make_bundle(tmp_path / "bundle")
    seen = {}
    monkeypatch.setattr(r7, "code_record", lambda *a, **k: dict(tree_sha256="t", files=1, closure_sha256={}, r7_closure_sha256={}))
    monkeypatch.setattr(s6, "_run_controller", fake_controller(seen, fail_uid))
    expected_sha = m.sha(tmp_path / "bundle/manifest.json")
    rows = r7.infer_studies(paths, bundle_dir=tmp_path / "bundle", code_dir=tmp_path / "code",
        anatomy_weights=tmp_path / "anatomy.pt", output_dir=tmp_path / "out", expected_manifest_sha256=expected_sha)
    assert [row[s6.UID] for row in rows] == ids
    args = seen["args"]
    assert "--deadline" not in args and args[args.index("--device") + 1] == "cuda"
    assert args[args.index("--max-fallback-fraction") + 1] == "1.0" and args[args.index("--arm") + 1] == r7.ARM
    assert args[args.index("--amp") + 1] == "auto" and args[args.index("--loader-workers") + 1] == "2"
    assert seen["bundle_files"] == sorted(r7.declared_files(manifest))
    bundle_copy = Path(args[args.index("--bundle") + 1])
    assert seen["worker_args"] == r7.worker_args(manifest, bundle_copy, (tmp_path / "anatomy.pt").resolve())
    assert not bundle_copy.exists()  # per-run temporary copy removed
    prov = json.loads((tmp_path / "out/provenance.json").read_text())
    assert prov["bundle_id"] == r7.BUNDLE_ID and prov["manifest_sha256"] == expected_sha
    assert prov["source_sha256"] == r7.SOURCE_HASHES and prov["worker_args"] == seen["worker_args"]
    assert (tmp_path / "out/predictions.csv").read_bytes() == (tmp_path / "out/controller/submission.csv").read_bytes()
    assert seen["env"]["HF_HUB_OFFLINE"] == "1"
    with pytest.raises(FileExistsError):
        r7.infer_studies(paths, bundle_dir=tmp_path / "bundle", code_dir=tmp_path / "code",
                         anatomy_weights=tmp_path / "anatomy.pt", output_dir=tmp_path / "out")


def test_infer_studies_fails_closed_before_controller(tmp_path, monkeypatch):
    paths = competition(tmp_path / "comp", ["s0"])
    make_bundle(tmp_path / "bundle")
    (tmp_path / "bundle/fluid_stacker.json").unlink()
    called = []
    monkeypatch.setattr(s6, "_run_controller", lambda *a: called.append(a))
    monkeypatch.setattr(r7, "code_record", lambda *a, **k: called.append("code"))
    with pytest.raises(ValueError, match="missing"):
        r7.infer_studies(paths, bundle_dir=tmp_path / "bundle", code_dir=tmp_path, anatomy_weights=tmp_path / "a.pt",
                         output_dir=tmp_path / "out")
    make_bundle(tmp_path / "bundle2")
    with pytest.raises(ValueError, match="manifest SHA256"):
        r7.infer_studies(paths, bundle_dir=tmp_path / "bundle2", code_dir=tmp_path, anatomy_weights=tmp_path / "a.pt",
                         output_dir=tmp_path / "out", expected_manifest_sha256="0" * 64)
    assert called == [] and not (tmp_path / "out").exists()


# ---------------------------------------------------------------- release arithmetic

def pct(x):
    return pd.DataFrame(x).rank(method="average", pct=True).to_numpy()


@pytest.mark.parametrize("pattern", range(32))
def test_blend_is_c85_v104_with_renamed_s_member(pattern):
    n = 300
    ids = [f"s{i}" for i in range(n)]
    values = np.random.default_rng(pattern).random((5, n, 12))
    failed = {name: dict.fromkeys(ids[member::3], "injected") if pattern & (1 << member) else {}
              for member, name in enumerate(m.MEMBERS)}
    renamed = {("S6" if k == "S_R7" else k): v for k, v in failed.items()}
    np.testing.assert_array_equal(m.blend(ids, *values, failed), c85.blend(ids, *values, renamed))
    if pattern == 0:
        r, s, e143, e155, q = values
        expected = .85 * pct(.5 * pct(r) + .5 * pct(s)) + .15 * pct((17 * pct(e143) + 17 * pct(e155) + 12 * pct(q)) / 46)
        np.testing.assert_array_equal(m.blend(ids, *values), expected)
        np.testing.assert_array_equal(m.reference(*values), expected)


def test_cpu_replay_reproduces_c85_v104_and_p956_public_formula(tmp_path):
    result = m.cpu_replay(tmp_path)
    assert m.cpu_replay(tmp_path) == result
    assert result["status"] == "PASS_CPU_ARITHMETIC_REPLAY" and result["candidate_status"] == "DIAGNOSTIC_C85R7"
    assert result["diagnostic_sha256"] == m.C85_REPLAY_CSV_SHA
    assert result["c85_v104_csv_identical"] and result["p_branch_equals_p956_public_formula"] and result["p_branch_csv_bytes_equal"]
    assert result["fallbacks"] == dict(a172_prediction=0, r_prediction=0, s_r7_prediction=0)
    assert not (tmp_path / "submission.csv").exists()
    receipt = m.read_json(tmp_path / m.RECEIPT)
    assert receipt["r7_identity"]["stand_in"] == m.STAND_IN and receipt["qualified_for_submission"] is False


def test_compare_p_reports_exact_parity_and_differences(tmp_path):
    m.cpu_replay(tmp_path / "replay")
    ours = tmp_path / "replay" / m.P_BRANCH
    result = m.compare_p(ours, m.p.REPLAY / "submission.csv")
    assert result["status"] == "P_BRANCH_MATCH" and result["exact_float32"] and result["studies"] == 3
    frame = pd.read_csv(m.p.REPLAY / "submission.csv")
    frame = frame.iloc[::-1]
    frame.to_csv(tmp_path / "reordered.csv", index=False)
    assert m.compare_p(ours, tmp_path / "reordered.csv")["status"] == "P_BRANCH_MATCH"
    frame.iloc[0, 3] = .123
    frame.to_csv(tmp_path / "changed.csv", index=False)
    changed = m.compare_p(ours, tmp_path / "changed.csv")
    assert changed["status"] == "P_BRANCH_DIFFERS" and changed["cells_differing"] == 1
    frame.iloc[:2].to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError, match="cohorts"):
        m.compare_p(ours, tmp_path / "short.csv")


A_FILES = ["exp169-partner/release-manifest.json", "exp169-executed-candidate.json", "partner-raw.npz",
           "a172-submission.csv", *["exp169-trio/" + x for x in
           ("receipt.json", "raw.npz", "inputs.json", "runtime.json", "binding.json", "submission.csv")]]


def copy_files(source, destination, names):
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)


@pytest.fixture
def staged(tmp_path):
    return m.stage_replay_input(tmp_path / "pr7")


def update_json(path, change):
    data = m.read_json(path)
    change(data)
    path.write_bytes(m.json_bytes(data))


@pytest.mark.parametrize("fault", ["bundle_id", "manifest", "files", "source", "preflight bundle"])
def test_pr7_identity_is_hard(staged, fault):
    prov, ident = staged / "pr7-run/s_r7/provenance.json", staged / "pr7-run/r7-identity.json"
    if fault == "bundle_id": update_json(prov, lambda r: r.update(bundle_id="209a32c8b50a6b1c"))
    if fault == "manifest": update_json(prov, lambda r: r.update(manifest_sha256="other"))
    if fault == "files": update_json(ident, lambda r: r["file_sha256"].update({"side_cls.pt": "other"}))
    if fault == "source": update_json(prov, lambda r: r["source_sha256"].update({"compact_infer.py": s6.SOURCE_HASHES["compact_infer.py"]}))
    if fault == "preflight bundle":
        update_json(ident, lambda r: r.update(bundle_id="x"))
    with pytest.raises(ValueError, match="identity"):
        m.authenticated_pr7(staged)


def test_pr7_r_journal_equals_recorded_raw_and_partial_workers(staged):
    ids, r_raw, s_before, _ = m.authenticated_pr7(staged)
    # Real C85 path: R arrives through the append journal written by c85.ServingBinding.
    journal = c85.Journal(staged / "pr7-run/r", ids, {"checkpoint_sha256": m.p.R_SHA, "source_pins": m.p.PINS})
    for i, uid in enumerate(ids): journal.record(uid, "R", r_raw[i].astype(np.float32))
    ids2, r_journal, _, evidence = m.authenticated_pr7(staged)
    np.testing.assert_array_equal(r_journal[[ids2.index(u) for u in ids]], r_raw.astype(np.float32))
    worker = staged / "pr7-run/s_r7/controller/worker0"
    part = m.read_json(worker / "ids.json")
    lines = (worker / "predictions.csv").read_bytes().splitlines(keepends=True)
    (worker / "predictions.csv").write_bytes(b"".join(lines[:2]) + b"interrupted-uid,0.")
    (worker / "receipt.json").unlink()
    _, _, s_after, evidence = m.authenticated_pr7(staged)
    for uid in part[1:]: assert uid in evidence["failures"]["S_R7"]
    np.testing.assert_array_equal(s_after[ids.index(part[0])], s_before[ids.index(part[0])])


def test_p_branch_failure_is_telemetry_only(tmp_path, staged, monkeypatch):
    publish = c85.publish
    def failing(src, dst, payload):
        if dst.name == m.P_BRANCH: raise OSError("injected p-branch failure")
        return publish(src, dst, payload)
    monkeypatch.setattr(c85, "publish", failing)
    receipt, payload = m.collect(c85.A5Q, staged, tmp_path / "out")
    assert receipt["status"] == "COMPLETE_C85R7"
    assert "injected" in receipt["telemetry"]["p_branch_written"]
    assert hashlib.sha256(payload).hexdigest() == m.C85_REPLAY_CSV_SHA
    assert (tmp_path / "out/submission.csv").read_bytes() == payload and not (tmp_path / "out" / m.P_BRANCH).exists()


@pytest.mark.parametrize("fault", ["csv pre-write", "csv publish-after", "receipt publish-partial"])
def test_publication_failure_is_failed_execution(tmp_path, staged, monkeypatch, fault):
    output = tmp_path / "out"
    write, replace = m.write_once, m.os.replace
    def fault_write(path, data):
        if fault == "csv pre-write" and path.name == "submission.csv": raise OSError("injected")
        return write(path, data)
    def fault_replace(src, dst):
        if fault == "csv publish-after" and dst == output / "submission.csv":
            replace(src, dst); raise OSError("injected")
        if fault == "receipt publish-partial" and dst == output / m.RECEIPT and b"COMPLETE_C85R7" in src.read_bytes():
            dst.write_bytes(src.read_bytes()[:32]); raise OSError("injected")
        return replace(src, dst)
    monkeypatch.setattr(m, "write_once", fault_write)
    monkeypatch.setattr(m.os, "replace", fault_replace)
    with pytest.raises(OSError):
        m.collect(c85.A5Q, staged, output)
    assert not (output / "submission.csv").exists()
    receipt = m.read_json(output / m.RECEIPT)
    assert receipt["status"] == "FAILED_EXECUTION" and receipt["submission_sha256"] is None
    assert receipt["authenticated_completed_components"] == ["A172", "PR7"] and receipt["child_receipts"]
    with pytest.raises(ValueError, match="stale"):
        m.collect(c85.A5Q, staged, output)


@pytest.mark.parametrize("fault", [None, "A172 failure", "PR7 failure"])
def test_runtime_orchestration_with_process_boundaries_mocked(tmp_path, monkeypatch, fault):
    blob, pins = m.package()
    runtime, output = tmp_path / "runtime", tmp_path / "output"
    m.p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    events = []
    def executor(args, *, label, env):
        assert env["HF_HUB_OFFLINE"] == "1" and env["PYTHONPATH"] == str(runtime)
        events.append(label)
        if label == "C85R7 frozen A172":
            assert args[-1].endswith("a172.py")
            if fault == "A172 failure": raise RuntimeError("injected A172 failure")
            copy_files(c85.A5Q, output / m.RUN / "a172", A_FILES)
        else:
            assert args[2:4] == [str(runtime / "exp169_release_c85r7.py"), "public-worker"]
            if fault == "PR7 failure": raise RuntimeError("injected PR7 failure")
            m.stage_replay_input(output / m.RUN / "pr7")
    monkeypatch.setattr(m.v5q, "run_process_group", executor)
    monkeypatch.setattr(m, "preflight", lambda *a: c85.authenticated_a172(c85.A5Q)[0])
    result = m.run_release(runtime, output)
    assert events == ["C85R7 frozen A172", "C85R7 R + S_R7"]
    assert result["status"] == "COMPLETE_C85R7" and (output / "submission.csv").exists()
    if fault is None:
        assert result["submission_sha256"] == m.C85_REPLAY_CSV_SHA
        assert (output / m.P_BRANCH).read_bytes() == (m.p.REPLAY / "submission.csv").read_bytes()
    if fault == "A172 failure": assert result["fallbacks"]["a172_prediction"] == 3
    if fault == "PR7 failure":
        assert result["fallbacks"]["r_prediction"] == result["fallbacks"]["s_r7_prediction"] == 3
    with pytest.raises(ValueError, match="stale"):
        m.run_release(runtime, output)


@pytest.mark.parametrize("fault", [None, "R", "S_R7"])
def test_public_worker_phases_are_isolated(tmp_path, monkeypatch, fault):
    out = tmp_path / "pr7"
    work = out / "pr7-run"
    work.mkdir(parents=True)
    m.write_once(out / "ids.json", m.json_bytes(["a"]))
    m.write_once(work / "r7-identity.json", m.json_bytes(dict(manifest_sha256="f" * 64)))
    m.write_once(work / "code-paths.json", m.json_bytes(dict(code=str(work / "code"), anatomy=str(work / "anatomy.pt"))))
    monkeypatch.setattr(m, "runtime_pins", lambda p: None)
    monkeypatch.setattr(m, "competition_root", lambda: tmp_path / "competition")
    monkeypatch.setattr(m.v5q, "_mount", lambda *a: tmp_path / "mount")
    monkeypatch.setattr(m.p, "check_pin", lambda *a: None)
    monkeypatch.setattr(m.s6, "install_wheels", lambda *a: None)
    events = []
    def r(args, **kw):
        assert args[2:4] == [str(tmp_path / "runtime/exp169_release_c85.py"), "r-worker"]
        events.append("R")
        if fault == "R": raise RuntimeError("child exit 7")
    def s(dirs, **kw):
        assert dirs == [tmp_path / "competition/test_series/a"] and kw["max_fallback_fraction"] == 1.0
        assert kw["expected_manifest_sha256"] == "f" * 64 and kw["source_dir"] == tmp_path / "runtime/exp169_r7_src"
        assert kw["output_dir"] == work / "s_r7" and kw["device"] == "cuda"
        events.append("S_R7")
        if fault == "S_R7": raise ValueError("R7 bundle files missing: state_stacker.json")
    monkeypatch.setattr(m.v5q, "run_process_group", r)
    monkeypatch.setattr(m.r7, "infer_studies", s)
    m.run_public(tmp_path / "runtime", out)
    assert events == ["R", "S_R7"]
    assert m.read_json(out / "R-exit.json")["exit_code"] == int(fault == "R")
    assert m.read_json(out / "S_R7-exit.json")["exit_code"] == int(fault == "S_R7")


@pytest.mark.parametrize("missing", [None, "state_stacker.json", "E_lrprior_s0_f3.pt"])
def test_preflight_records_identity_and_fails_closed(tmp_path, monkeypatch, missing):
    ids = ["s0", "s1"]
    comp = tmp_path / "comp"
    comp.mkdir()
    (comp / "sample_submission.csv").write_bytes(m.p.csv_bytes(ids, np.full((2, 12), .5)))
    (comp / "test.csv").write_text("StudyInstanceUID\ns1\ns0\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    shutil.copytree(r7.SOURCE, runtime / "exp169_r7_src")
    make_bundle(tmp_path / "bundle")
    if missing: (tmp_path / "bundle" / missing).unlink()
    mounts = {r7.DATASET_SLUG: tmp_path / "bundle"}
    monkeypatch.setattr(m, "competition_root", lambda: comp)
    monkeypatch.setattr(m.v5q, "_mount", lambda slug, required: mounts.get(slug, tmp_path / "other"))
    monkeypatch.setattr(m.p, "check_pin", lambda *a: None)
    prepared = []
    monkeypatch.setattr(m.s6, "prepare_code", lambda src, dst, install_wheels: prepared.append(dst) or (dst / "code", src / "anatomy.pt"))
    output = tmp_path / "out"
    if missing:
        with pytest.raises(ValueError, match="missing"): m.preflight(runtime, output)
        assert prepared == []
        return
    assert m.preflight(runtime, output) == ids
    identity = m.read_json(output / m.RUN / "pr7/pr7-run/r7-identity.json")
    assert identity["manifest_sha256"] == m.sha(tmp_path / "bundle/manifest.json") and identity["bundle_id"] == r7.BUNDLE_ID
    assert prepared == [output / m.RUN / "pr7/pr7-run/s-code"]


# ---------------------------------------------------------------- packaging and stages

def test_package_is_deterministic_and_isolated_import_works(tmp_path):
    blob, pins = m.package()
    assert (blob, pins) == m.package() and len(pins) == 27
    assert set(c85.package()[1]) < set(pins)
    assert {name: digest for name, digest in pins.items() if name in c85.package()[1]} == c85.package()[1]
    assert pins["exp169_release_c85.py"] == m.C85_SHA and pins["exp169_r7_port.py"] == m.R7_PORT_SHA
    runtime = tmp_path / "runtime"
    m.p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    m.runtime_pins(runtime)
    result = subprocess.run([sys.executable, "-c",
        "import exp169_release_c85r7 as c; c.runtime_pins(c.Path('.')); print(c.VERSION, c.r7.VERSION, c.c85.VERSION)"],
        cwd=runtime, capture_output=True, text=True, check=True,
        env=dict(m.os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    assert result.stdout.strip() == "1.0.0 1.0.0 1.0.4"
    with (runtime / "exp169_r7_src/compact_infer.py").open("ab") as f: f.write(b"\n")
    with pytest.raises(ValueError, match="SHA256"): m.runtime_pins(runtime)


def test_notebook_bootstrap_dispatch(tmp_path, monkeypatch):
    blob, pins = m.package()
    source = "".join(m.notebook_payload(blob, pins)["cells"][-1]["source"])
    runtime = tmp_path / "runtime"
    source = source.replace("Path('/kaggle/working/c85r7-runtime')", f"Path({str(runtime)!r})")
    source = source.replace("Path('/kaggle/working')", f"Path({str(tmp_path / 'output')!r})")
    events = []
    original = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "exp169_release_c85r7":
            def boundary(rt, out):
                m.runtime_pins(rt)
                assert rt == runtime and out == tmp_path / "output"
                events.append("dispatched")
            return types.SimpleNamespace(run_release=boundary)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    old_path = sys.path[:]
    try: exec(compile(source, "<test-c85r7-bootstrap>", "exec"), {})
    finally: sys.path[:] = old_path
    assert events == ["dispatched"]
    assert {name: m.sha(runtime / name) for name in pins} == pins


def test_stress_bootstrap_dispatches_full_release(tmp_path, monkeypatch):
    blob, pins = m.package()
    code = "".join(m.stress_notebook(blob, pins)["cells"][-1]["source"]).replace("/kaggle/working", tmp_path.as_posix())
    events = []
    def prepare(comp, out, n):
        assert n == 32 and comp == tmp_path / "competition"
        events.append("select"); return list(range(32))
    def run(rt, out):
        m.runtime_pins(rt)
        assert m.os.environ["C85_COMPETITION"] == str(tmp_path / "c85-stress-data") and out == tmp_path
        events.append("release")
    importer = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "exp169_release_c85r7": return types.SimpleNamespace(run_release=run, p=m.p,
            v5q=types.SimpleNamespace(_mount=lambda *a: tmp_path / "competition"))
        if name == "exp169_c85_stress": return types.SimpleNamespace(prepare=prepare)
        return importer(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("C85_COMPETITION", "")
    old_path = sys.path[:]
    try: exec(compile(code, "<stress bootstrap>", "exec"), {})
    finally: sys.path[:] = old_path
    assert events == ["select", "release"]


def test_all_cpu_cards_and_metadata(tmp_path):
    m.prepare(tmp_path)
    m.prepare(tmp_path)
    assert m.validate(tmp_path)["status"] == "PASS_CPU_VALIDATE"
    assert m.register(tmp_path)["status"] == "REGISTERED_LOCAL_ONLY"
    m.cpu_replay(tmp_path / "cpu-replay")
    result = m.admit(tmp_path)
    assert result["status"] == "HOLD_NOT_QUALIFIED" and result["cpu_preparation"] == "PASS"
    meta, stress = (m.read_json(tmp_path / f"{d}/kernel-metadata.json") for d in ("kernel", "stress-kernel"))
    old = m.read_json(c85.OUT / "kernel/kernel-metadata.json")
    assert meta["id"] == "taeyangg4/knee-exp169-c85r7-v1" and stress["id"] == "taeyangg4/knee-exp169-c85r7-v1-stress-n32"
    assert meta["dataset_sources"] == [d for d in old["dataset_sources"] if d != s6.DATASET_SLUG][:3] + [
        r7.DATASET_SLUG, m.v5q.CODE_SLUG]
    assert set(meta["dataset_sources"]) == set(old["dataset_sources"]) - {s6.DATASET_SLUG} | {r7.DATASET_SLUG}
    for key in ("docker_image", "machine_shape", "is_private", "enable_internet", "enable_gpu", "competition_sources"):
        assert meta[key] == old[key] == stress[key]
    update_json(tmp_path / "kernel/kernel-metadata.json", lambda r: r.update(enable_internet=True))
    with pytest.raises(ValueError, match="SHA256"): m.validate(tmp_path)


def test_c85_v104_sources_and_stage_unchanged():
    assert m.sha(m.ROOT / "tools/exp169_release_c85.py") == m.C85_SHA
    assert c85.validate(c85.OUT)["status"] == "PASS_CPU_VALIDATE"
    for line in (c85.OUT / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest_, name = line.split(maxsplit=1)
            path = Path(name.strip().lstrip("*"))
            path = path if path.is_absolute() else m.ROOT / path
            if path.is_file(): assert m.sha(path) == digest_, name
