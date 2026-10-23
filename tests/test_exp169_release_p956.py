"""P956 v1.0.1 CPU gates. No GPU, model inference, network or account operations.

The integration smoke replays genuine A5 outputs at model-executor boundaries;
it executes the new packaging, R binding, orchestration, validation and commit.
"""
from __future__ import annotations

import ast
import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
import types
import queue
import threading
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import exp169_release_p956 as p


@pytest.fixture
def visible():
    p.check_pin(p.VISIBLE / "exp169-trio/raw.npz", p.RAW_SHA)
    p.check_pin(p.VISIBLE / "s6-port-predictions.csv", p.S6_VISIBLE_SHA)
    ids, ours = p.v5q._read_csv(p.VISIBLE / "s6-port-predictions.csv")
    with np.load(p.VISIBLE / "exp169-trio/raw.npz", allow_pickle=False) as raw:
        p.v5q.validate_ids(ids, raw["ids"].tolist())
        r = raw["public"].copy()
    receipt = p.read_json(p.VISIBLE / "s6-port/controller/submission_receipt.json")
    return ids, r, ours, receipt


@pytest.mark.parametrize("n", [1, 2, 3, 1300])
@pytest.mark.parametrize("kind", ["random", "ties", "constant"])
def test_all_12_average_pct_float32_against_independent_pandas(n, kind):
    rng = np.random.default_rng(956)
    r, ours = rng.random((2, n, 12))
    if kind == "ties":
        r, ours = np.floor(r * 8) / 8, np.floor(ours * 8) / 8
        r[:, 0] = .125  # never the public failed-study sentinel
    if kind == "constant":
        r[:, ::2], ours[:, 1::2] = .25, .75
    expected = (.5 * pd.DataFrame(r).rank(method="average", pct=True)
                + .5 * pd.DataFrame(ours).rank(method="average", pct=True)).values.astype(np.float32)
    actual = p.blend([f"study-{i}" for i in range(n)], r, ours)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert np.all(actual > 0)  # rank/N, never (rank-1)/(N-1)


def test_ties_and_oa_are_half_half_without_final_rerank():
    r = np.tile([.1, .1, .9], (12, 1)).T
    ours = np.tile([.8, .2, .4], (12, 1)).T
    expected = np.tile([.75, 5 / 12, 5 / 6], (12, 1)).T.astype(np.float32)
    np.testing.assert_array_equal(p.blend(["a", "b", "c"], r, ours), expected)


@pytest.mark.parametrize("fault", ["empty", "duplicate", "shape", "nan", "inf", "range", "R sentinel"])
def test_blend_fail_closed(visible, fault):
    ids, r, ours, _ = visible
    if fault == "empty": ids = []
    if fault == "duplicate": ids[1] = ids[0]
    if fault == "shape": ours = ours[:, :11]
    if fault == "nan": ours[1, 2] = np.nan
    if fault == "inf": r[1, 2] = np.inf
    if fault == "range": ours[1, 2] = 1.01
    if fault == "R sentinel": r[0, :] = .5
    with pytest.raises(ValueError):
        p.blend(ids, r, ours)


@pytest.mark.parametrize("fault", ["missing exit", "count", "fallback", "worker failure",
                                    "worker count", "models", "worker fallback", "roster"])
def test_receipt_failures_never_leave_submission(tmp_path, visible, fault):
    ids, r, ours, receipt = visible
    if fault == "missing exit": receipt["worker_exit_codes"].pop("1")
    if fault == "count": receipt["scored"] -= 1
    if fault == "fallback": receipt["fallback_studies"] = 1
    if fault == "worker failure": receipt["workers"][0]["status"] = "FAIL"
    if fault == "worker count": receipt["workers"][0]["scored"] -= 1
    if fault == "models": receipt["workers"][0]["models"].append("extra.pt")
    if fault == "worker fallback": receipt["workers"][0]["fallback"] = {"decode": 1}
    if fault == "roster": receipt["workers"] *= 2
    with pytest.raises(ValueError):
        p.commit(tmp_path, ids, r, ours, receipt, {})
    assert not (tmp_path / "submission.csv").exists()


def test_anatomy_fallback_is_retained_and_real_replay_is_byte_exact(tmp_path, visible):
    report = p.cpu_replay(tmp_path)
    assert report["rows"] == 3 and report["findings"] == 12
    assert report["max_abs"] == 0 and report["exact_csv_bytes"] is True
    assert report["submission_sha256"] == "5c4828369e39d80a6e67ece4e784206e99323120aff975216005755f7f7053dd"
    receipt = p.read_json(tmp_path / "p956-receipt.json")
    assert receipt["s6_anatomy_anchors"]["fallback"] == 1
    ids, r, ours, _ = visible
    np.testing.assert_array_equal(p.v5q._read_csv(tmp_path / "submission.csv")[1].astype(np.float32),
                                  p.public_formula(r, ours))
    assert (tmp_path / "submission.csv").read_bytes() == p.csv_bytes(ids, p.public_formula(r, ours))


def test_stale_output_never_overwritten_and_late_failure_is_closed(tmp_path, visible, monkeypatch):
    ids, r, ours, receipt = visible
    (tmp_path / "submission.csv").write_bytes(b"pre-existing user output")
    with pytest.raises(ValueError, match="stale"):
        p.commit(tmp_path, ids, r, ours, receipt, {})
    assert (tmp_path / "submission.csv").read_bytes() == b"pre-existing user output"
    fresh = tmp_path / "fresh"
    def fail_replace(*args):
        assert not (fresh / "submission.csv").exists()
        assert p.read_json(fresh / "p956-receipt.json")["status"] == "PASS_P956_OUTPUT"
        raise OSError("injected commit failure")
    monkeypatch.setattr(p.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        p.commit(fresh, ids, r, ours, receipt, {})
    assert not (fresh / "submission.csv").exists()
    assert not (fresh / "p956-submission.tmp").exists()


def test_source_pins_and_r_only_adapter_preserve_predictor_functions(tmp_path, visible):
    from tools import exp152_own0949_release as helper
    original = (p.VISIBLE / "exp155-assets/public-serving.py").read_text(encoding="utf-8")
    derivative = p.r_runtime_source(original)
    def functions(source):
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    before, after = functions(original), functions(derivative)
    assert set(before) == set(after)
    assert [name for name in before if before[name] != after[name]] == ["_pre_one", "main"]
    assert "ranks[~np.isfinite(ranks)] = 0.5" not in derivative
    assert "nproc = min(12, max(1," in derivative
    # Neither new R adapter nor unchanged R serving has an execution deadline.
    for forbidden in ("timeout=", "signal.alarm(", "deadline", "wait_for("):
        assert forbidden not in derivative
    with pytest.raises(ValueError, match="drift"):
        p.r_runtime_source(original + "\n")
    namespace = {}
    weights, comp = tmp_path / p.R_NAME, tmp_path / "competition"
    p.bind_r(namespace, helper, weights, comp, tmp_path)
    assert namespace["ARMS"] == [dict(file=p.R_NAME, arch=helper.ARCH, res=384, w=1.0)]
    assert set(namespace["_E152_FILES"]) == set(namespace["_E152_MANIFEST"]["checkpoints"]) == {p.R_NAME}
    assert namespace["find_test_root"]() == str(comp)
    ids, r, _, _ = visible
    bound = namespace["_E152"]
    assert bound.checked_model is helper.checked_model
    assert bound.input_record is helper.input_record
    bound.save_raw(tmp_path, ids, [r], dict.fromkeys(ids), {}, "R")
    with np.load(tmp_path / "raw.npz", allow_pickle=False) as raw:
        assert set(raw.files) == {"ids", "targets", "public"}
        np.testing.assert_array_equal(raw["public"], r)
    np.testing.assert_array_equal(bound.output_values([r], "R", None), r)
    bound.finish_receipt(tmp_path, 999999)
    assert p.read_json(tmp_path / "receipt.json")["failure_count"] == 0
    with pytest.raises(ValueError, match="R-only"):
        bound.output_values([r, r], "R", None)


def test_deterministic_archive_authentication_and_import(tmp_path):
    blob, pins = p.package()
    assert p.package() == (blob, pins)
    assert len(pins) == 12 and not any(name.endswith(".pt") for name in pins)
    runtime = tmp_path / "runtime"
    p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    p.runtime_pins(runtime)
    # An isolated interpreter imports only embedded modules; no model is loaded.
    result = subprocess.run([sys.executable, "-c",
        "import exp169_release_p956 as p; p.runtime_pins(p.Path('.')); print(p.VERSION)"],
        cwd=runtime, capture_output=True, text=True, check=True,
        env=dict(p.os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    assert result.stdout.strip() == "1.0.1"
    with pytest.raises(ValueError, match="fresh"):
        p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    with pytest.raises(ValueError, match="archive hash"):
        p.unpack(blob + b"x", hashlib.sha256(blob).hexdigest(), tmp_path / "bad", pins)
    with pytest.raises(ValueError, match="roster"):
        p.unpack(blob, hashlib.sha256(blob).hexdigest(), tmp_path / "roster", {})
    bad = dict(pins, **{"exp169_release_p956.py": "0" * 64})
    with pytest.raises(ValueError, match="member differs"):
        p.unpack(blob, hashlib.sha256(blob).hexdigest(), tmp_path / "member", bad)
    (runtime / "exp189_s6_port.py").write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        p.runtime_pins(runtime)


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "C:/absolute.py", "dir\\escape.py"])
def test_unpack_rejects_unsafe_paths(tmp_path, name):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        entry = zipfile.ZipInfo("payload")
        entry.filename = name  # Preserve backslashes even on Windows.
        archive.writestr(entry, b"x")
    blob = data.getvalue()
    # Windows normalizes backslashes while reading ZIP entries; that changes the
    # authenticated roster and must also fail before creating the destination.
    with pytest.raises(ValueError, match="unsafe|roster"):
        p.unpack(blob, hashlib.sha256(blob).hexdigest(), tmp_path / "out",
                 {name: hashlib.sha256(b"x").hexdigest()})
    assert not (tmp_path / "out").exists()


def test_kernelspec_title_dataset_roster_and_all_local_commands(tmp_path):
    p.prepare(tmp_path)
    assert p.validate(tmp_path)["status"] == "PASS_CPU_VALIDATE"
    p.prepare(tmp_path)  # deterministic idempotent write
    nb = p.read_json(tmp_path / "kernel/release.ipynb")
    meta = p.read_json(tmp_path / "kernel/kernel-metadata.json")
    assert nb["metadata"]["kernelspec"] == {"display_name": "Python 3", "language": "python", "name": "python3"}
    assert meta["id"] == "taeyangg4/knee-exp169-p956-v1"
    assert meta["title"] == "knee-exp169-p956-v1"
    assert meta["dataset_sources"] == ["nartaa/rsna-knee-publication-swa-weights-20261007",
        "heliosli/rsna-knee-compact-v1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
    assert meta["competition_sources"] == [p.COMPETITION]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["docker_image"] == p.IMAGE
    assert meta["enable_internet"] is False
    p.cpu_replay(tmp_path / "cpu-replay")
    assert p.register(tmp_path)["status"] == "REGISTERED_LOCAL_ONLY"
    assert p.admit(tmp_path)["status"] == "PASS_ADMIT_DRY_RUN"
    assert p.admit(tmp_path)["qualified_for_submission"] is False
    nb["metadata"]["kernelspec"]["name"] = "wrong"
    (tmp_path / "kernel/release.ipynb").write_bytes(p.json_bytes(nb))
    build = p.read_json(tmp_path / "build-receipt.json")
    build["notebook_sha256"] = p.sha(tmp_path / "kernel/release.ipynb")
    (tmp_path / "build-receipt.json").write_bytes(p.json_bytes(build))
    with pytest.raises(ValueError, match="kernelspec"):
        p.validate(tmp_path)


@pytest.mark.parametrize("fault", [None, "R failure", "S6 failure", "ID mismatch", "receipt failure", "GPU shape"])
def test_real_data_cpu_orchestration_smoke(tmp_path, visible, monkeypatch, fault):
    """Real A5 arrays at GPU boundaries, actual new control path and R callbacks.

    No synthetic claim of fresh model inference: only check_pin(weight), mounts,
    model executors and external environment are substituted here.
    """
    ids, r, ours, receipt = visible
    blob, pins = p.package()
    runtime = tmp_path / "runtime"
    p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    comp, output = tmp_path / "competition", tmp_path / "output"
    comp.mkdir()
    # Exercise differing valid sample/test order; the final CSV follows sample.
    order = [2, 0, 1]
    ordered_ids = [ids[i] for i in order]
    (comp / "sample_submission.csv").write_bytes(p.csv_bytes(ordered_ids, np.full((3, 12), .2)))
    with (comp / "test.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([p.UID])
        writer.writerows([[uid] for uid in ids])
    weights = tmp_path / "r-weights" / p.R_NAME
    mounts = {p.COMPETITION: comp, p.DATASETS[0]: weights.parent,
              p.s6.DATASET_SLUG: tmp_path / "bundle", p.v5q.CODE_SLUG: tmp_path / "source"}
    monkeypatch.setattr(p.v5q, "_mount", lambda slug, required: mounts[slug])
    original_pin = p.check_pin
    def pin(path, digest):
        if path == weights:
            assert digest == p.R_SHA
        else:
            original_pin(path, digest)
    monkeypatch.setattr(p, "check_pin", pin)
    events = []
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw:
                        "Tesla T4\nTesla T4\n" if fault != "GPU shape" else "Tesla P100\n")
    monkeypatch.setattr(p.s6, "verify_bundle", lambda bundle: events.append("preflight"))
    def prepare_code(source, destination, *, install_wheels):
        assert install_wheels is False
        destination.mkdir()
        events.append("prepare without install")
        return destination, destination / "anatomy.pt"
    monkeypatch.setattr(p.s6, "prepare_code", prepare_code)
    original_bind = p.bind_r
    def binding(namespace, helper, weight_path, competition, out):
        original_bind(namespace, helper, weight_path, competition, out)
        bound = namespace["_E152"]
        def replay_main():
            # Replaces existing GPU decoder/model execution, retaining new callbacks.
            if fault == "R failure":
                raise RuntimeError("injected R failure")
            assert len(namespace["ARMS"]) == 1
            bound.save_raw(out, ids, [r], dict.fromkeys(ids), namespace["_E152_MANIFEST"], "R")
            np.testing.assert_array_equal(bound.output_values([r], "R", None), r)
            bound.finish_receipt(out, 999999)
            events.append("R finished")
        namespace["main"] = replay_main
    monkeypatch.setattr(p, "bind_r", binding)
    # Execute the real r-worker Python function, with real module compilation/import
    # and binding, but no CUDA calls and no model weights loaded.
    def r_executor(args, *, label, env):
        assert env["HF_HUB_OFFLINE"] == "1"
        p.run_r(runtime, weights, comp, output / "p956-run/r")
    monkeypatch.setattr(p.v5q, "run_process_group", r_executor)
    def install(wheels, code):
        assert events[-1] == "R finished"
        events.append("S6 install")
    monkeypatch.setattr(p.s6, "install_wheels", install)
    def infer(studies, **kw):
        assert events[-1] == "S6 install"
        assert [s.name for s in studies] == ordered_ids
        assert kw["workers"] == 2 and kw["max_fallback_fraction"] == .5
        assert kw["amp"] == "auto" and kw["device"] == "cuda"
        if fault == "S6 failure": raise RuntimeError("injected S6 failure")
        out = kw["output_dir"]
        (out / "controller").mkdir(parents=True)
        if fault == "receipt failure": receipt["fallback_studies"] = 1
        (out / "controller/submission_receipt.json").write_bytes(p.json_bytes(receipt))
        for wi, indices in enumerate(([0, 1], [2])):
            worker = out / "controller" / f"worker{wi}"
            worker.mkdir()
            (worker / "ids.json").write_bytes(p.json_bytes([ids[i] for i in indices]))
            (worker / "predictions.csv").write_bytes(p.csv_bytes([ids[i] for i in indices], ours[indices]))
        (out / "provenance.json").write_bytes(p.json_bytes({"scope": "real A5 CPU output replay"}))
        with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([p.UID, *p.LABELS])
            writer.writerows([[ids[i], *ours[i]] for i in order])
        rows = [{p.UID: ids[i], **dict(zip(p.LABELS, ours[i]))} for i in order]
        if fault == "ID mismatch": rows.reverse()
        events.append("S6 finished")
        return rows
    monkeypatch.setattr(p.s6, "infer_studies", infer)
    # Import of the authenticated R source only initializes CPU tensors; visible
    # CUDA is disabled by the LAUNCH test command, and main was replaced above.
    try:
        if fault:
            with pytest.raises((ValueError, RuntimeError)):
                p.run_release(runtime, output)
            assert not (output / "submission.csv").exists()
        else:
            result = p.run_release(runtime, output)
            assert events == ["preflight", "prepare without install", "R finished", "S6 install", "S6 finished"]
            assert result["members"] == ["R", "S6"] and result["weights"] == [.5, .5]
            assert (output / "submission.csv").read_bytes() == p.csv_bytes(ordered_ids, p.public_formula(r[order], ours[order]))
    finally:
        sys.modules.pop("p956_r_serving", None)


def test_notebook_bootstrap_real_archive_cpu_smoke(tmp_path, monkeypatch, visible):
    p.prepare(tmp_path / "stage")
    notebook = p.read_json(tmp_path / "stage/kernel/release.ipynb")
    source = "".join(notebook["cells"][-1]["source"])
    # Literal path redirection and executor replacement only. Run actual bootstrap.
    runtime = tmp_path / "bootstrap-runtime"
    source = source.replace("Path('/kaggle/working/p956-runtime')", f"Path({str(runtime)!r})")
    source = source.replace("Path('/kaggle/working')", f"Path({str(tmp_path / 'output')!r})")
    import builtins
    original_import = builtins.__import__
    def import_module(name, *args, **kw):
        module = original_import(name, *args, **kw)
        if name == "exp169_release_p956":
            def replay(runtime_arg, output):
                assert runtime_arg == runtime
                module.runtime_pins(runtime)
                ids, r, ours, receipt = visible
                return module.commit(output, ids, r, ours, receipt, {"scope": "bootstrap CPU replay"})
            module.run_release = replay
        return module
    monkeypatch.setattr(builtins, "__import__", import_module)
    monkeypatch.syspath_prepend(str(runtime))
    try:
        exec(compile(source, "<local P956 bootstrap smoke>", "exec"), {})
        assert (tmp_path / "output/submission.csv").is_file()
    finally:
        # Bootstrap adds its own sys.path entry too.
        if str(runtime) in sys.path: sys.path.remove(str(runtime))
        for name in ("exp169_release_p956", "exp169_release_v5q", "exp189_s6_port"):
            sys.modules.pop(name, None)


def expanded_real_outputs(visible, n=100):
    """Replicate actual visible predictions to exercise the public 3% boundary."""
    source_ids, r, ours, _ = visible
    ids = [source_ids[i % 3] + f"-fault-test-{i}" for i in range(n)]
    return ids, r[np.arange(n) % 3].copy(), ours[np.arange(n) % 3].copy()


def write_s6_evidence(root, ids, ours, failures, *, dead=False, malformed=False):
    workers = []
    for wi in range(2):
        indices = list(range(wi, len(ids), 2))
        part = [ids[i] for i in indices]
        worker = root / f"worker{wi}"
        worker.mkdir(parents=True)
        (worker / "ids.json").write_bytes(p.json_bytes(part))
        good = [i for i in indices if ids[i] not in failures]
        if good:
            (worker / "predictions.csv").write_bytes(p.csv_bytes([ids[i] for i in good], ours[good]))
        if malformed and wi == 0:
            with (worker / "predictions.csv").open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([part[0], *(["nan"] * 12)])
        bad = {uid: failures[uid] for uid in part if uid in failures}
        if dead and wi == 0:
            (worker / "log.txt").write_text("Traceback: RuntimeError: injected worker startup failure", encoding="utf-8")
            workers.append(dict(status="NO_RECEIPT", worker=wi, fallback=bad))
        else:
            receipt = dict(status="PASS", models=[f"E_lrprior_s0_f{i}.pt" for i in range(4)],
                studies=len(part), scored=len(good), fallback=bad, stats={}, anchor_methods={})
            (worker / "receipt.json").write_bytes(p.json_bytes(receipt))
            workers.append(receipt)
    return dict(status="PASS", rows=len(ids), scored=len(ids) - len(failures),
        fallback_studies=len(failures), fallback_reasons=failures,
        worker_exit_codes={"0": 1 if dead else 0, "1": 0}, gpus=["Tesla T4", "Tesla T4"], workers=workers)


@pytest.mark.parametrize("path", ["healthy", "R decode", "R inference", "S6 decode", "S6 gpu",
                                  "S6 loader", "S6 dead worker", "both", "at thresholds"])
def test_real_output_failure_paths_exact_public_formula_and_receipt(tmp_path, visible, path):
    ids, r, ours = expanded_real_outputs(visible)
    rf, sf = {}, {}
    if path in {"R decode", "R inference", "both", "at thresholds"}:
        for i in range(3 if path == "at thresholds" else 1):
            rf[ids[i]] = "RuntimeError: injected " + path
            r[i] = .5
    if path.startswith("S6") or path in {"both", "at thresholds"}:
        indices = range(0, 100, 2) if path == "S6 dead worker" else range(50 if path == "at thresholds" else 1)
        sf = {ids[i]: "RuntimeError: injected " + path for i in indices}
    controller = write_s6_evidence(tmp_path / "workers", ids, ours, sf, dead=path == "S6 dead worker")
    observed = p.s6_failure_evidence(tmp_path / "workers", ids, controller)
    assert set(observed) == set(sf)
    if path == "S6 dead worker": assert all("startup failure" in why for why in observed.values())
    mask = np.array([uid in sf for uid in ids])
    # A controller mean fill must not enter either model's rank basis.
    ours[mask] = ours[~mask].mean(axis=0)
    receipt = p.commit(tmp_path / "output", ids, r, ours, controller, {"fault_injection": path},
                       r_failures=rf, s6_failures=observed)
    expected = p.public_formula(r, ours, mask)
    assert (tmp_path / "output/submission.csv").read_bytes() == p.csv_bytes(ids, expected)
    assert {entry["uid"] for entry in receipt["replacements"]} == set(rf) | set(sf)
    for entry in receipt["replacements"]:
        assert all(entry["exceptions"].values())
        uid = entry["uid"]
        assert entry["replacement"] == ("0.5" if uid in rf and uid in sf else "S6 rank" if uid in rf else "R rank")
    if path == "both": np.testing.assert_array_equal(expected[0], np.full(12, .5))
    if sf:
        changed = ours.copy()
        changed[mask] = .999
        np.testing.assert_array_equal(p.blend(ids, r, changed, observed), expected)


@pytest.mark.parametrize("model,count,passes", [("R", 3, True), ("R", 4, False),
                                               ("S6", 50, True), ("S6", 51, False)])
def test_exact_public_stop_boundaries(tmp_path, visible, model, count, passes):
    ids, r, ours = expanded_real_outputs(visible)
    rf = {uid: "ValueError: injected" for uid in ids[:count]} if model == "R" else {}
    sf = {uid: "ValueError: injected" for uid in ids[:count]} if model == "S6" else {}
    if rf: r[:count] = .5
    controller = write_s6_evidence(tmp_path / "workers", ids, ours, sf)
    if passes:
        p.commit(tmp_path / "output", ids, r, ours, controller, {}, r_failures=rf, s6_failures=sf)
    else:
        with pytest.raises(ValueError, match="stop threshold"):
            p.commit(tmp_path / "output", ids, r, ours, controller, {}, r_failures=rf, s6_failures=sf)
        assert not (tmp_path / "output/submission.csv").exists()


def test_s6_invalid_rows_and_failure_evidence_guards(tmp_path, visible):
    ids, r, ours = expanded_real_outputs(visible)
    sf = {ids[0]: "non-finite score"}
    controller = write_s6_evidence(tmp_path / "workers", ids, ours, sf, malformed=True)
    observed = p.s6_failure_evidence(tmp_path / "workers", ids, controller)
    assert observed == sf
    r[0] = .5
    with pytest.raises(ValueError, match="exception evidence"):
        p.commit(tmp_path / "missing-r-evidence", ids, r, ours, controller, {}, s6_failures=sf)
    controller["fallback_reasons"] = {}
    with pytest.raises(ValueError, match="scored-set"):
        p.s6_failure_evidence(tmp_path / "workers", ids, controller)


@pytest.mark.parametrize("partial_receipt", [False, True])
def test_s6_dead_worker_keeps_already_scored_rows(tmp_path, visible, partial_receipt):
    ids, _, ours = expanded_real_outputs(visible)
    # The dead worker wrote all but its last assigned study. Only that UID fails.
    failures = {ids[98]: "worker exited"}
    controller = write_s6_evidence(tmp_path, ids, ours, failures, dead=True)
    if partial_receipt:
        (tmp_path / "worker0/receipt.json").write_text('{"status":', encoding="utf-8")
    evidence = p.s6_failure_evidence(tmp_path, ids, controller)
    assert set(evidence) == {ids[98]}
    assert "startup failure" in evidence[ids[98]]
    # Exit after every prediction was flushed is not a failed study (cell 7).
    worker = tmp_path / "worker0"
    with (worker / "predictions.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([ids[98], *ours[98]])
    controller.update(scored=100, fallback_studies=0, fallback_reasons={})
    assert p.s6_failure_evidence(tmp_path, ids, controller) == {}


@pytest.mark.parametrize("path", ["R", "S6", "both", "dead S6"])
def test_release_orchestration_with_real_output_failures(tmp_path, visible, monkeypatch, path):
    ids, r, ours = expanded_real_outputs(visible)
    rf = {ids[0]: "OSError: original decode failure"} if path in {"R", "both"} else {}
    sf = {ids[i]: "RuntimeError: original S6 failure" for i in
          (range(0, 100, 2) if path == "dead S6" else [0])} if path != "R" else {}
    if rf: r[0] = .5
    comp = tmp_path / "comp"
    comp.mkdir()
    # Reverse sample order to verify that failures follow UID, never row position.
    ordered = ids[::-1]
    (comp / "sample_submission.csv").write_bytes(p.csv_bytes(ordered, ours[::-1]))
    (comp / "test.csv").write_text(p.UID + "\n" + "\n".join(ids) + "\n", encoding="utf-8")
    monkeypatch.setattr(p, "runtime_pins", lambda runtime: None)
    monkeypatch.setattr(p, "check_pin", lambda *args: None)
    monkeypatch.setattr(p.v5q, "_mount", lambda slug, required: comp if slug == p.COMPETITION else tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "Tesla T4\nTesla T4\n")
    monkeypatch.setattr(p.s6, "verify_bundle", lambda *a: None)
    monkeypatch.setattr(p.s6, "verify_source", lambda *a: None)
    monkeypatch.setattr(p.s6, "prepare_code", lambda *a, **kw: (tmp_path, tmp_path / "anatomy.pt"))
    monkeypatch.setattr(p.s6, "install_wheels", lambda *a: None)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "exp169_release_p956.py").write_bytes(Path(p.__file__).read_bytes())
    output = tmp_path / "output"
    def run_r(args, **kw):
        work = output / "p956-run/r"
        work.mkdir()
        helper = types.SimpleNamespace(checked_model=None, input_record=None)
        bound = p.RBinding(helper, work)
        bound.failures = rf
        bound.save_raw(work, ids, [r], dict.fromkeys(ids), {}, "R")
        bound.finish_receipt(work, 0)
    monkeypatch.setattr(p.v5q, "run_process_group", run_r)
    def infer(studies, **kw):
        assert [x.name for x in studies] == ordered and kw["max_fallback_fraction"] == .5
        out = kw["output_dir"]
        controller = write_s6_evidence(out / "controller", ids, ours, sf, dead=path == "dead S6")
        (out / "controller/submission_receipt.json").write_bytes(p.json_bytes(controller))
        (out / "provenance.json").write_bytes(p.json_bytes({"scope": "fault-injected real output replay"}))
        (out / "predictions.csv").write_bytes(p.csv_bytes(ordered, ours[::-1]))
        return p.s6.read_predictions(out / "predictions.csv", ordered)
    monkeypatch.setattr(p.s6, "infer_studies", infer)
    receipt = p.run_release(runtime, output)
    assert {entry["uid"] for entry in receipt["replacements"]} == set(rf) | set(sf)
    values = p.public_formula(r[::-1], ours[::-1], [uid in sf for uid in ordered])
    assert (output / "submission.csv").read_bytes() == p.csv_bytes(ordered, values)


@pytest.mark.parametrize("fault", [None, "decode", "empty", "inference", "nonfinite", "sentinel"])
def test_real_r_main_cpu_failure_transport(tmp_path, visible, monkeypatch, fault):
    """Execute adapted main/producer/consumer on CPU; replay model outputs only."""
    import multiprocessing as mp
    ids, r, _ = expanded_real_outputs(visible)
    comp, output = tmp_path / "comp", tmp_path / "r"
    comp.mkdir(); output.mkdir()
    (comp / "test.csv").write_text(p.UID + "\n" + "\n".join(ids) + "\n", encoding="utf-8")
    pd.DataFrame({p.UID: ids, "SeriesInstanceUID": ids}).to_csv(comp / "test_series.csv", index=False)
    (comp / "sample_submission.csv").write_bytes(p.csv_bytes(ids, r))
    source = p.r_runtime_source((p.VISIBLE / "exp155-assets/public-serving.py").read_text(encoding="utf-8"))
    selected = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and
                node.name in {"_pre_one", "_pre_init", "_exp152_bounded_preprocess", "main"}]
    helper = types.SimpleNamespace(checked_model=None, input_record=lambda *a: {})
    bound = p.RBinding(helper, output)
    def reader(uid, *args):
        i = ids.index(uid)
        if i == 0 and fault == "decode": raise OSError("injected original DICOM exception")
        return np.array([i]), np.array([0 if i == 0 and fault == "empty" else 1])
    def predict(model, xw, xm, device):
        i = int(xw[0])
        if i == 0 and fault == "inference": raise RuntimeError("injected model exception")
        if i == 0 and fault == "nonfinite": return np.full(12, np.nan)
        if i == 0 and fault == "sentinel": return np.full(12, .5)
        return r[i]
    class Pool:
        def __init__(self, n, initializer): initializer()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def imap_unordered(self, fn, batch, chunksize): return map(fn, reversed(batch))
    monkeypatch.setattr(mp, "get_context", lambda kind: types.SimpleNamespace(Pool=Pool))
    # XLA is explicitly unavailable for this CPU replay.
    monkeypatch.setitem(sys.modules, "torch_xla", None)
    namespace = dict(np=np, os=p.os, time=time, queue=queue, threading=threading, gc=gc,
        torch=types.SimpleNamespace(__version__="CPU replay", cuda=types.SimpleNamespace(
            is_available=lambda: False, device_count=lambda: 0)),
        LAB=list(p.LABELS), ARMS=[dict(file=p.R_NAME, arch="pinned", res=384, w=1)],
        find_test_root=lambda: str(comp), find_weight_file=lambda name: name,
        load_model=lambda *args: (None, 384), gpu_windows=lambda vol, *a, **kw: vol,
        infer_pair_probs=predict, K_EVAL=1, INPUT_RES=384, NORM=None, rankpct=None,
        QUEUE_MAX=4, FASTREAD_VARIANT="pinned", FR_HDR=1, FR_PX=1, IMG=1, MAXS=1, SLOTS=1,
        _load_fastread=lambda: types.SimpleNamespace(make_builder=lambda *a, **kw: reader),
        _E152=bound, _E152_WORK=output, _E152_VARIANT="R", _E152_MANIFEST={})
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<adapted R CPU main>", "exec"), namespace)
    namespace["main"]()
    receipt = p.read_json(output / "receipt.json")
    assert receipt["failure_count"] == int(fault is not None)
    if fault:
        assert set(receipt["failures"]) == {ids[0]}
        expected_message = {"decode": "OSError: injected original DICOM exception", "empty": "Empty decoded study",
            "inference": "RuntimeError: injected model exception", "nonfinite": "invalid R model prediction",
            "sentinel": "FailedStudySentinel"}[fault]
        assert expected_message in receipt["failures"][ids[0]]
    with np.load(output / "raw.npz", allow_pickle=False) as raw:
        expected = r.copy()
        if fault: expected[0] = .5
        np.testing.assert_array_equal(raw["public"], expected)
