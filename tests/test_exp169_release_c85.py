"""C85 CPU gates: genuine saved outputs; GPU/network boundaries never executed."""
from __future__ import annotations

import builtins
import copy
import hashlib
import io
import json
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import exp169_release_c85 as c


def pct(x):
    return pd.DataFrame(x).rank(method="average", pct=True).to_numpy()


@pytest.mark.parametrize("n", [1, 300, 1300])
@pytest.mark.parametrize("kind", ["random", "ties", "constant"])
def test_nested_formula_and_average_ties(n, kind):
    values = np.random.default_rng(85).random((5, n, 12))
    if kind == "ties": values = np.floor(values * 8) / 8
    if kind == "constant": values[:] = .25
    r, s, e143, e155, q = values
    expected = .85 * pct(.5 * pct(r) + .5 * pct(s)) + .15 * pct(
        (17 * pct(e143) + 17 * pct(e155) + 12 * pct(q)) / 46)
    actual = c.blend([f"study-{i}" for i in range(n)], *values)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64
    assert np.all(actual > 0)
    if n == 1: np.testing.assert_array_equal(actual, np.ones((1, 12)))


def test_nested_ranks_not_flat_weights_or_intermediate_float32():
    values = np.random.default_rng(85).random((5, 1300, 12))
    r, s, e143, e155, q = values
    actual = c.blend([f"study-{i}" for i in range(1300)], *values)
    public = .5 * pct(r) + .5 * pct(s)
    own = (17 * pct(e143) + 17 * pct(e155) + 12 * pct(q)) / 46
    assert not np.array_equal(actual, .85 * public + .15 * own)
    assert not np.array_equal(actual, .85 * pct(public.astype(np.float32)) + .15 * pct(own.astype(np.float32)))


@pytest.mark.parametrize("fault", ["empty", "duplicate", "shape", "nan", "inf", "range"])
def test_missing_nonfinite_invalid_predictions(fault):
    ids = ["a", "b", "c"]
    values = np.full((5, 3, 12), .2)
    args = list(values)
    if fault == "empty": ids = []
    if fault == "duplicate": ids[1] = ids[0]
    if fault == "shape": args[4] = args[4][:, :11]
    if fault == "nan": args[0][1, 1] = np.nan
    if fault == "inf": args[3][1, 1] = np.inf
    if fault == "range": args[2][1, 1] = 1.1
    with pytest.raises(ValueError): c.blend(ids, *args)


def copy_files(source, destination, names):
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)


A_FILES = ["exp169-partner/release-manifest.json", "exp169-executed-candidate.json", "partner-raw.npz",
           "a172-submission.csv", *["exp169-trio/" + x for x in
           ("receipt.json", "raw.npz", "inputs.json", "runtime.json", "binding.json", "submission.csv")]]
B_FILES = ["p956-receipt.json", "submission.csv", "p956-run/r/raw.npz", "p956-run/r/receipt.json",
           "p956-run/s6/predictions.csv", "p956-run/s6/provenance.json",
           "p956-run/s6/controller/submission_receipt.json", *[
           f"p956-run/s6/controller/worker{i}/{name}" for i in range(2)
           for name in ("ids.json", "predictions.csv", "receipt.json")]]


@pytest.fixture
def saved(tmp_path):
    a, b = tmp_path / "a172", tmp_path / "p956"
    copy_files(c.A5Q, a, A_FILES)
    copy_files(c.p.REPLAY, b, B_FILES)
    return a, b


def update_json(path, change):
    data = c.read_json(path)
    change(data)
    path.write_bytes(c.json_bytes(data))


@pytest.mark.parametrize("fault", ["Q7", "Q7+S", "checkpoint", "family"])
def test_q_a5q_identity_is_exact(saved, fault):
    a, _ = saved
    path = a / "exp169-partner/release-manifest.json"
    def change(m):
        if fault in {"Q7", "Q7+S"}: m["candidate"] = fault
        if fault == "checkpoint": m["entries"][0]["sha256"] = "0" * 64
        if fault == "family": m["entries"][-1]["family"] = "convnext_s"
    update_json(path, change)
    with pytest.raises(ValueError, match="SHA256"): c.authenticated_a172(a)


@pytest.mark.parametrize("fault", ["targets", "study", "missing", "nonfinite", "family shape"])
def test_raw_schema_and_cohort_reject_even_with_updated_file_hash(saved, fault):
    a, _ = saved
    partner = fault == "family shape"
    raw_path = a / ("partner-raw.npz" if partner else "exp169-trio/raw.npz")
    with np.load(raw_path, allow_pickle=False) as raw: arrays = {k: raw[k].copy() for k in raw.files}
    if fault == "targets": arrays["targets"] = arrays["targets"][::-1]
    if fault == "study": arrays["ids"] = arrays["ids"][::-1]
    if fault == "missing": arrays.pop("exp143")
    if fault == "nonfinite": arrays["own"][0, 1] = np.nan
    if partner: arrays["l2"] = arrays["l2"][:5]
    np.savez(raw_path, **arrays)
    if partner:
        update_json(a / "exp169-executed-candidate.json",
                    lambda r: r.update(partner_raw_sha256=c.sha(raw_path)))
    else:
        update_json(a / "exp169-trio/receipt.json",
                    lambda r: r["files"].update({"raw.npz": c.sha(raw_path)}))
    if fault in {'targets', 'family shape'}:
        with pytest.raises(ValueError): c.authenticated_a172(a)
    else:
        result = c.authenticated_a172(a)
        if fault in {'missing','nonfinite'}: assert result[-1]['prediction_fallbacks'] > 0
        else: assert result[0] == arrays['ids'].tolist()


def test_p956_target_identity_and_worker_evidence(saved):
    _, b = saved
    update_json(b / "p956-run/s6/controller/submission_receipt.json",
                lambda r: r["worker_exit_codes"].pop("1"))
    result = c.authenticated_p956(b)
    assert 'worker/environment' in result[-1]['telemetry']['controller_crosscheck']


def test_real_output_replay_and_deterministic_csv(tmp_path):
    result = c.cpu_replay(tmp_path)
    assert c.cpu_replay(tmp_path) == result
    assert result["status"] == "PASS_CPU_ARITHMETIC_REPLAY"
    assert result["candidate_status"] == "DIAGNOSTIC_C85"
    assert result["fallbacks"] == dict(a172_prediction=0, r_prediction=0, s6_prediction=0)
    assert result["studies"] == 3 and result["findings"] == 12
    assert result["diagnostic_sha256"] == "a59be5c6066f87b5309e242ecced132fab1894b8dd4b69169e8450a171ddee8a"
    assert result["qualified_for_submission"] is False
    assert not (tmp_path / "submission.csv").exists()
    telemetry = c.read_json(tmp_path / "c85-receipt.json")["s6_anatomy_anchors"]
    assert telemetry["fallback"] == 1 and telemetry["model_anchor"] == 2


def downloaded_observation_inputs(tmp_path):
    """Restore omitted CSVs only in scratch, using receipt-pinned historical bytes."""
    import csv
    frozen = c.ROOT / ".local/scratch/astra-1/c85-observation/remote/c85-observation/frozen-run"
    source = frozen / "c85-run/p956"
    b = tmp_path / "observation-p956"
    copy_files(source, b, [name for name in B_FILES if (source / name).is_file()])
    receipt = c.read_json(source / "p956-receipt.json")
    for name, digest in (("submission.csv", receipt["submission_sha256"]),
                         ("p956-run/s6/predictions.csv", receipt["provenance"]["s6_raw_sha256"])):
        c.p.check_pin(c.p.REPLAY / name, digest)
        copy_files(c.p.REPLAY, b, [name])
    # The observation retained worker IDs/receipts but omitted worker CSVs.
    # Partition the authenticated, complete S6 CSV by those recorded assignments.
    lines = (b / "p956-run/s6/predictions.csv").read_bytes().splitlines(keepends=True)
    rows = list(csv.DictReader(io.StringIO(b"".join(lines).decode())))
    by_id = {row[c.s6.UID]: line for row, line in zip(rows, lines[1:])}
    for i in range(2):
        worker = b / f"p956-run/s6/controller/worker{i}"
        ids = c.read_json(worker / "ids.json")
        (worker / "predictions.csv").write_bytes(lines[0] + b"".join(by_id[uid] for uid in ids))
    return frozen / "c85-run/a172", b


def test_downloaded_observation_completes_without_changing_csv(tmp_path):
    import importlib.util
    a, b = downloaded_observation_inputs(tmp_path)
    old_source = c.ROOT / ".local/backups/exp169-c85-v1.0.1-before-v1.0.2/tools/exp169_release_c85.py"
    spec = importlib.util.spec_from_file_location("c85_previous", old_source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    before, old_csv = old.collect(a, b, tmp_path / "old")
    after, new_csv = c.collect(a, b, tmp_path / "new")
    assert before["status"] == "FAILED_FALLBACK" and before["fallbacks"]["s6_anatomy"] == 1
    assert after["status"] == "COMPLETE_C85" and not any(after["fallbacks"].values())
    assert after["s6_anatomy_anchors"] == before["s6_anatomy_anchors"]
    assert after["s6_anatomy_anchors"]["methods"] == {
        "skm_tea_joint": 1, "skm_tea_joint_avg2": 1, "slab_prior": 1}
    assert old_csv == new_csv == (tmp_path / "new/submission.csv").read_bytes()
    assert after["submission_sha256"] == hashlib.sha256(old_csv).hexdigest()


@pytest.mark.parametrize("fault", [None, "anatomy", "R", "S6", "A172", "worker", "cohort", "roster"])
def test_receipt_complete_vs_fallback_no_root_csv(tmp_path, monkeypatch, fault):
    a = list(c.authenticated_a172(c.A5Q))
    b = list(c.authenticated_p956(c.p.REPLAY))
    # Exercise both model-only and slab-prior telemetry; neither is a prediction failure.
    b[-1]["s6_anatomy_anchors"] = dict(scored=3, model_anchor=3, fallback=0, methods={"test_only": 3})
    if fault == "anatomy": b[-1]["s6_anatomy_anchors"].update(model_anchor=2, fallback=1)
    if fault == "R": b[-1]["r_prediction_fallbacks"] = 1
    if fault == "S6":
        b[-1]["s6_prediction_fallbacks"] = 1
        b[-1]["s6_anatomy_anchors"].update(scored=2, model_anchor=2)
    if fault == "A172": a[-1]["prediction_fallbacks"] = 1
    if fault == "worker": b[-1]["worker_exit_codes"]["1"] = 1
    if fault == "cohort": b[0] = b[0][::-1]
    if fault == "roster": a[-1]["executed_arms"] = ["Q7"]
    monkeypatch.setattr(c, "authenticated_a172", lambda _: a)
    monkeypatch.setattr(c, "authenticated_p956", lambda _: b)
    result, payload = c.collect(Path("a"), Path("b"), tmp_path)
    assert result["status"] == "COMPLETE_C85"
    assert result["qualified_for_submission"] is False
    assert (tmp_path / "submission.csv").read_bytes() == payload
    assert result["submission_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result['telemetry']['uid_order_aligned'] == (fault == 'cohort')


def test_path_only_adapter_on_frozen_unlimited_a172(tmp_path):
    cell = "".join(c.read_json(c.v5q.STAGE / "kernel/release.ipynb")["cells"][4]["source"])
    unlimited = c.v5q.a172_runtime_source(cell)
    output = tmp_path / "a172"
    adapted = c.a172_source(cell, output)
    compile(adapted, '<own-only>', 'exec')
    assert '_c85.bind_own(' in adapted and '_c85.run_q(' in adapted
    assert 'pilot(' not in adapted and '.serve(' not in adapted
    assert f"_out = Path({output.as_posix()!r})" in adapted
    for token in ("timeout=", "signal.alarm(", "24300"):
        assert token not in adapted
    with pytest.raises(ValueError, match="drift"): c.a172_source(cell + "\n", output)
    with pytest.raises(ValueError, match="unsafe"): c.a172_source(cell, tmp_path / "bad'path")


def test_package_is_deterministic_and_isolated_import_works(tmp_path):
    blob, pins = c.package()
    assert (blob, pins) == c.package() and len(pins) == 18
    runtime = tmp_path / "runtime"
    c.p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    c.runtime_pins(runtime)
    result = subprocess.run([sys.executable, "-c",
        "import exp169_release_c85 as c; c.runtime_pins(c.Path('.')); print(c.VERSION)"],
        cwd=runtime, capture_output=True, text=True, check=True,
        env=dict(c.os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    assert result.stdout.strip() == "1.0.4"
    (runtime / "q-a5q-manifest.json").write_bytes(b"Q7")
    with pytest.raises(ValueError, match="SHA256"): c.runtime_pins(runtime)


@pytest.mark.parametrize("fault", [None, "A172 failure", "P956 failure", "P956 receipt", "complete test counters"])
def test_real_data_cpu_runtime_orchestration(tmp_path, monkeypatch, fault):
    blob, pins = c.package()
    runtime, output = tmp_path / "runtime", tmp_path / "output"
    c.p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    events = []
    def executor(args, *, label, env):
        assert env["HF_HUB_OFFLINE"] == "1" and env["PYTHONPATH"] == str(runtime)
        events.append(label)
        if label.endswith("A172"):
            if fault == "A172 failure": raise RuntimeError("injected A172 failure")
            copy_files(c.A5Q, output / "c85-run/a172", A_FILES)
        else:
            assert events == ["C85 frozen A172", "C85 unchanged P956"]
            if fault == "P956 failure": raise RuntimeError("injected P956 failure")
            destination = output / "c85-run/p956"
            copy_files(c.p.REPLAY, destination, B_FILES)
            if fault == "P956 receipt":
                update_json(destination / "p956-receipt.json", lambda r: r.update(status="FAIL"))
    monkeypatch.setattr(c.v5q, "run_process_group", executor)
    monkeypatch.setattr(c, 'preflight', lambda *args: c.authenticated_a172(c.A5Q)[0])
    if fault == "complete test counters":
        original = c.authenticated_p956
        def test_only(directory):
            result = original(directory)
            result[-1]["s6_anatomy_anchors"] = dict(scored=3, model_anchor=3, fallback=0,
                                                  methods={"TEST_ONLY_COUNTER_OVERRIDE": 3})
            return result
        monkeypatch.setattr(c, "authenticated_p956", test_only)
        assert c.run_release(runtime, output)["status"] == "COMPLETE_C85"
        assert (output / "submission.csv").exists()
    elif fault is None:
        result = c.run_release(runtime, output)
        assert result["status"] == "COMPLETE_C85"
        assert result["s6_anatomy_anchors"]["fallback"] == 1
        assert (output / "submission.csv").exists()
    else:
        result = c.run_release(runtime, output)
        assert result['status'] == 'COMPLETE_C85'
        assert (output / 'submission.csv').exists()
        assert events == ['C85 frozen A172', 'C85 unchanged P956']
        if fault == 'A172 failure': assert result['fallbacks']['a172_prediction'] == 3
        if fault == 'P956 failure': assert result['fallbacks']['r_prediction'] == result['fallbacks']['s6_prediction'] == 3
    with pytest.raises(ValueError, match="stale"): c.run_release(runtime, output)


def test_notebook_bootstrap_cpu_execution(tmp_path, monkeypatch):
    blob, pins = c.package()
    source = "".join(c.notebook_payload(blob, pins)["cells"][-1]["source"])
    runtime = tmp_path / "runtime"
    source = source.replace("Path('/kaggle/working/c85-runtime')", f"Path({str(runtime)!r})")
    source = source.replace("Path('/kaggle/working')", f"Path({str(tmp_path / 'output')!r})")
    events = []
    original_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "exp169_release_c85":
            def boundary(rt, out):
                c.runtime_pins(rt)
                assert rt == runtime and out == tmp_path / "output"
                events.append("authenticated runtime dispatched; no inference")
            return types.SimpleNamespace(run_release=boundary)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    old_path = sys.path[:]
    try: exec(compile(source, "<test-c85-bootstrap>", "exec"), {})
    finally: sys.path[:] = old_path
    assert events == ["authenticated runtime dispatched; no inference"]
    assert {name: c.sha(runtime / name) for name in pins} == pins


def test_all_cpu_cards_and_metadata(tmp_path):
    c.prepare(tmp_path)
    c.prepare(tmp_path)
    assert c.validate(tmp_path)["status"] == "PASS_CPU_VALIDATE"
    assert c.register(tmp_path)["status"] == "REGISTERED_LOCAL_ONLY"
    c.cpu_replay(tmp_path / "cpu-replay")
    result = c.admit(tmp_path)
    assert result["status"] == "HOLD_NOT_QUALIFIED" and result["cpu_preparation"] == "PASS"
    assert result["candidate_replay_status"] == "DIAGNOSTIC_C85"
    metadata = c.read_json(tmp_path / "kernel/kernel-metadata.json")
    assert metadata["is_private"] and not metadata["enable_internet"]
    assert metadata["dataset_sources"] == c.DATASETS and len(c.DATASETS) == 5
    assert metadata["id"] == c.KERNEL_ID
    update_json(tmp_path / "kernel/kernel-metadata.json", lambda r: r.update(is_private=False))
    with pytest.raises(ValueError, match="SHA256"): c.validate(tmp_path)


def test_all_historical_scored_stage_hashes_unchanged():
    historical = c.OUT.parent / "v1.0.0"
    baseline = c.read_json(historical / "scored-baseline.json")
    assert len(baseline) == 29
    assert {name: c.sha(c.ROOT / name) for name in baseline} == baseline
    backup = c.ROOT / ".local/backups/exp169-c85-v1.0.0-before-v1.0.1"
    preserved = c.read_json(backup / "preserved-baseline.json")
    assert {name: c.sha(c.ROOT / name) for name in preserved} == preserved
    for line in (historical / "SHA256SUMS.txt").read_text().splitlines():
        if not line.strip(): continue
        digest, name = line.split("  ", 1)
        path = backup / name if name.startswith(("tools/", "tests/")) else c.ROOT / name
        assert c.sha(path) == digest


@pytest.fixture
def healthy_cpu_runtime(tmp_path, monkeypatch):
    """Genuine saved arrays/identity checks; only anatomy counters and inference are mocked."""
    runtime, output = tmp_path / "runtime", tmp_path / "output"
    blob, pins = c.package()
    c.p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    def executor(args, *, label, env):
        if label.endswith("A172"):
            copy_files(c.A5Q, output / "c85-run/a172", A_FILES)
        else:
            copy_files(c.p.REPLAY, output / "c85-run/p956", B_FILES)
    monkeypatch.setattr(c.v5q, "run_process_group", executor)
    monkeypatch.setattr(c, 'preflight', lambda *args: c.authenticated_a172(c.A5Q)[0])
    original = c.authenticated_p956
    def test_only(directory):
        result = original(directory)
        result[-1]["s6_anatomy_anchors"] = dict(scored=3, model_anchor=3, fallback=0,
                                              methods={"TEST_ONLY_COUNTER_OVERRIDE": 3})
        return result
    monkeypatch.setattr(c, "authenticated_p956", test_only)
    return runtime, output


def output_snapshot(output):
    return {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}


@pytest.mark.parametrize("entry", ["collect", "run_release"])
@pytest.mark.parametrize("fault", [
    "csv pre-write", "csv partial-write", "csv short-write", "csv wrong-values",
    "csv publish-before", "csv publish-partial", "csv publish-after", "csv publish-corrupt",
    "receipt pre-write", "receipt partial-write", "receipt post-write", "receipt short-write",
    "receipt publish-before", "receipt publish-partial", "receipt publish-after",
])
def test_publication_failure_is_failed_execution_and_no_root(
        healthy_cpu_runtime, monkeypatch, entry, fault):
    runtime, output = healthy_cpu_runtime
    write, replace = c.write_once, c.os.replace
    triggered = []
    def fault_write(path, data):
        target = (fault.startswith("csv ") and path.name == "submission.csv") or (
            fault.startswith("receipt ") and b'"status": "COMPLETE_C85"' in data)
        if target and "publish-" not in fault:
            triggered.append(fault)
            if fault.endswith("pre-write"): raise OSError("injected " + fault)
            if fault.endswith(("partial-write", "short-write")):
                write(path, data[:32])
                if fault.endswith("partial-write"): raise OSError("injected " + fault)
                return
            if fault.endswith("wrong-values"):
                frame = pd.read_csv(io.BytesIO(data))
                frame.iloc[0, 1] = .123456
                return write(path, frame.to_csv(index=False, lineterminator="\n").encode())
            write(path, data)
            raise OSError("injected " + fault)
        return write(path, data)
    def fault_replace(src, dst):
        target = (fault.startswith("csv ") and dst == output / "submission.csv") or (
            fault.startswith("receipt ") and dst == output / "c85-receipt.json" and
            b'"status": "COMPLETE_C85"' in src.read_bytes())
        if target and "publish-" in fault:
            triggered.append(fault)
            if fault.endswith("publish-before"): raise OSError("injected " + fault)
            if fault.endswith("publish-partial"):
                # Adversarial non-atomic destination write before a publish error.
                dst.write_bytes(src.read_bytes()[:32])
                raise OSError("injected " + fault)
            replace(src, dst)
            if fault.endswith("publish-corrupt"):
                dst.write_bytes(b"corrupt final CSV")
                return
            raise OSError("injected " + fault)
        return replace(src, dst)
    monkeypatch.setattr(c, "write_once", fault_write)
    monkeypatch.setattr(c.os, "replace", fault_replace)
    with pytest.raises((OSError, ValueError)):
        if entry == "collect": c.collect(c.A5Q, c.p.REPLAY, output)
        else: c.run_release(runtime, output)
    assert triggered == [fault]
    assert not (output / "submission.csv").exists()
    receipt = c.read_json(output / "c85-receipt.json")
    assert receipt["status"] == "FAILED_EXECUTION"
    assert receipt["submission_sha256"] is None and receipt["qualified_for_submission"] is False
    assert receipt["authenticated_completed_components"] == ["A172", "P956"]
    assert receipt["child_receipts"] and receipt["error"]
    before = output_snapshot(output)
    with pytest.raises(ValueError, match="stale"):
        c.collect(c.A5Q, c.p.REPLAY, output)
    with pytest.raises(ValueError, match="stale"):
        c.run_release(runtime, output)
    assert output_snapshot(output) == before


def test_success_receipt_follows_verified_final_csv(healthy_cpu_runtime, monkeypatch):
    runtime, output = healthy_cpu_runtime
    write, replace = c.write_once, c.os.replace
    ids, e143, e155, q, _ = c.authenticated_a172(c.A5Q)
    _, r, s, _ = c.authenticated_p956(c.p.REPLAY)
    expected = .85 * pct(.5 * pct(r) + .5 * pct(s)) + .15 * pct(
        (17 * pct(e143) + 17 * pct(e155) + 12 * pct(q)) / 46)
    frame = pd.DataFrame(expected.astype(np.float32), columns=list(c.p.LABELS))
    frame.insert(0, c.p.UID, ids)
    expected_csv = frame.to_csv(index=False, lineterminator="\n").encode()
    success_writes = []
    def observe_write(path, data):
        if b'"status": "COMPLETE_C85"' in data:
            assert (output / "submission.csv").read_bytes() == expected_csv
            success_writes.append(path)
        return write(path, data)
    def observe_publish(src, dst):
        if dst == output / "submission.csv":
            assert not dst.exists() and not (output / "c85-receipt.json").exists()
            assert src.read_bytes() == expected_csv
        return replace(src, dst)
    monkeypatch.setattr(c, "write_once", observe_write)
    monkeypatch.setattr(c.os, "replace", observe_publish)
    result = c.run_release(runtime, output)
    assert len(success_writes) == 1 and result["status"] == "COMPLETE_C85"
    assert c.read_json(output / "c85-receipt.json") == result
    assert result["submission_sha256"] == hashlib.sha256(expected_csv).hexdigest()
    assert (output / "submission.csv").read_bytes() == expected_csv
    assert result["qualified_for_submission"] is False


@pytest.mark.parametrize("root_csv", [None, b"historical partial CSV", b"historical complete CSV"])
def test_existing_success_or_partial_record_is_immutable(healthy_cpu_runtime, root_csv):
    runtime, output = healthy_cpu_runtime
    output.mkdir()
    (output / "c85-receipt.json").write_bytes(b'{"status":"COMPLETE_C85","version":"1.0.0"}\n')
    if root_csv is not None: (output / "submission.csv").write_bytes(root_csv)
    before = output_snapshot(output)
    for diagnostic in (False, True):
        with pytest.raises(ValueError, match="stale"):
            c.collect(c.A5Q, c.p.REPLAY, output, diagnostic=diagnostic)
    with pytest.raises(ValueError, match="stale"): c.run_release(runtime, output)
    assert output_snapshot(output) == before


def test_existing_diagnostic_corruption_is_rejected_without_repair(tmp_path):
    c.cpu_replay(tmp_path)
    (tmp_path / "diagnostic-c85.csv").write_bytes(b"historical partial CSV")
    before = output_snapshot(tmp_path)
    with pytest.raises(ValueError, match="immutable"): c.cpu_replay(tmp_path)
    assert output_snapshot(tmp_path) == before


def test_healthy_diagnostic_is_not_a_success_receipt(healthy_cpu_runtime):
    _, output = healthy_cpu_runtime
    result, _ = c.collect(c.A5Q, c.p.REPLAY, output, diagnostic=True)
    assert result["status"] == "DIAGNOSTIC_C85" and result["submission_sha256"] is None
    assert not (output / "submission.csv").exists()


def test_unavailable_receipt_destination_retains_failure_evidence(healthy_cpu_runtime, monkeypatch):
    runtime, output = healthy_cpu_runtime
    replace = c.os.replace
    def unavailable(src, dst):
        if dst == output / "c85-receipt.json":
            raise OSError("receipt destination unavailable")
        return replace(src, dst)
    monkeypatch.setattr(c.os, "replace", unavailable)
    with pytest.raises(OSError, match="receipt destination unavailable"):
        c.run_release(runtime, output)
    assert not (output / "submission.csv").exists()
    assert not (output / "c85-receipt.json").exists()
    failures = [json.loads(p.read_bytes()) for p in output.glob(".c85-publication/*.json")]
    assert any(r["status"] == "FAILED_EXECUTION" and r["submission_sha256"] is None for r in failures)
    before = output_snapshot(output)
    with pytest.raises(ValueError, match="stale"): c.run_release(runtime, output)
    with pytest.raises(ValueError, match="stale"): c.collect(c.A5Q, c.p.REPLAY, output)
    assert output_snapshot(output) == before


@pytest.mark.parametrize("n", [1, 300, 1300])
@pytest.mark.parametrize("pattern", range(32))
def test_all_member_failure_combinations(n, pattern):
    ids = [f's{i}' for i in range(n)]
    values = np.random.default_rng(103).random((5,n,12))
    failed = np.zeros((5,n), bool)
    for member in range(5):
        if pattern & (1 << member): failed[member, ::3] = True
    failure_map = {name: dict.fromkeys(np.asarray(ids)[mask], 'injected') for name,mask in zip(c.MEMBERS,failed)}
    # Independent NaN/pandas transcription; failed rows never enter any rank cohort.
    inner = []
    for arr,mask in zip(values,failed):
        v=arr.copy(); v[mask]=np.nan
        inner.append(pd.DataFrame(v).rank(method='average',pct=True).to_numpy())
    r,s,e143,e155,q = inner
    public = .5*r + .5*s
    public[failed[0]] = s[failed[0]]
    public[failed[1]] = r[failed[1]]
    own = np.zeros((n,12)); denom=np.zeros(n)
    for weight,rank,mask in zip([17,17,12],[e143,e155,q],failed[2:]):
        own += weight*np.nan_to_num(rank); denom += weight*(~mask)
    own /= np.where(denom>0,denom,np.nan)[:,None]
    pr,oo = [pd.DataFrame(v).rank(method='average',pct=True).to_numpy() for v in [public,own]]
    expected=.85*pr+.15*oo
    pf,of=failed[:2].all(0),failed[2:].all(0)
    expected[pf]=oo[pf]; expected[of]=pr[of]; expected[pf&of]=.5
    np.testing.assert_allclose(c.blend(ids,*values,failure_map),expected,rtol=0,atol=1e-15)


def test_journal_partial_invalid_and_identity(tmp_path):
    ids=['a','b','c']; identity={'checkpoint':'test'}
    j=c.Journal(tmp_path,ids,identity)
    j.record('a','EXP155',np.arange(12)/20)
    j.record('b','EXP155',error='injected OOM')
    with (tmp_path/'predictions.jsonl').open('a') as f:
        f.write(json.dumps(dict(uid='c',member='EXP155',values=[2]*12))+'\n')
        f.write('{"uid":')
    values,fail=c.read_journal(tmp_path,ids,['EXP155'],identity)
    np.testing.assert_array_equal(values['EXP155'][0],np.arange(12)/20)
    assert set(fail['EXP155'])=={'b','c'}
    with pytest.raises(ValueError,match='identity'): c.read_journal(tmp_path,ids,['EXP155'],{})


def test_partial_s6_worker_recovers_flushed_rows(saved):
    _,b=saved
    ids,_,before,_=c.authenticated_p956(b)
    worker=b/'p956-run/s6/controller/worker0'
    part=c.read_json(worker/'ids.json')
    lines=(worker/'predictions.csv').read_bytes().splitlines(keepends=True)
    (worker/'predictions.csv').write_bytes(b''.join(lines[:2]))
    (worker/'receipt.json').unlink()
    (b/'p956-run/s6/controller/submission_receipt.json').unlink()
    _,_,after,receipt=c.authenticated_p956(b)
    for uid in part[1:]: assert uid in receipt['failures']['S6']
    np.testing.assert_array_equal(after[ids.index(part[0])],before[ids.index(part[0])])


def test_v102_stage_unchanged():
    baseline=c.read_json(c.ROOT/'.local/backups/exp169-c85-v1.0.2-before-v1.0.3/preserved-baseline.json')
    assert {name:c.sha(c.ROOT/name) for name in baseline}==baseline


@pytest.mark.parametrize("fault", [None, "decode", "empty", "inference", "nonfinite", "load"])
def test_own_main_cpu_recovery(tmp_path, monkeypatch, fault):
    import ast, time, queue, threading, gc
    p = c.p
    """Execute adapted main/producer/consumer on CPU; replay model outputs only."""
    import multiprocessing as mp
    ids, e143, e155, _, evidence = c.authenticated_a172(c.A5Q)
    r = e155
    comp, output = tmp_path / "comp", tmp_path / "r"
    comp.mkdir(); output.mkdir()
    (comp / "test.csv").write_text(p.UID + "\n" + "\n".join(ids) + "\n", encoding="utf-8")
    pd.DataFrame({p.UID: ids, "SeriesInstanceUID": ids}).to_csv(comp / "test_series.csv", index=False)
    (comp / "sample_submission.csv").write_bytes(p.csv_bytes(ids, r))
    source = c.serving_source((p.VISIBLE / "exp155-assets/public-serving.py").read_text(encoding="utf-8"))
    selected = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and
                node.name in {"_pre_one", "_pre_init", "_exp152_bounded_preprocess", "main"}]
    helper = types.SimpleNamespace(checked_model=None, input_record=lambda *a: {})
    bound = c.ServingBinding(helper, output, ids, {'test':'own'}, ['EXP155','EXP143v2'])
    def reader(uid, *args):
        i = ids.index(uid)
        if i == 0 and fault == "decode": raise OSError("injected original DICOM exception")
        return np.array([i]), np.array([0 if i == 0 and fault == "empty" else 1])
    def predict(model, xw, xm, device):
        i = int(xw[0])
        if model is None: raise RuntimeError("injected load failure")
        if model == 0 and i == 0 and fault == "inference": raise RuntimeError("injected model exception")
        if model == 0 and i == 0 and fault == "nonfinite": return np.full(12, np.nan)
        if i == 0 and fault == "sentinel": return np.full(12, .5)
        return (e155 if model == 0 else e143)[i]
    def loader(name, *args):
        if name == "0" and fault == "load": raise RuntimeError("injected load failure")
        return int(name), 320
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
        LAB=list(p.LABELS), ARMS=[dict(file=str(i), arch="pinned", res=384, w=1) for i in range(2)],
        find_test_root=lambda: str(comp), find_weight_file=lambda name: name,
        load_model=loader, gpu_windows=lambda vol, *a, **kw: vol,
        infer_pair_probs=predict, K_EVAL=1, INPUT_RES=384, NORM=None, rankpct=None,
        QUEUE_MAX=4, FASTREAD_VARIANT="pinned", FR_HDR=1, FR_PX=1, IMG=1, MAXS=1, SLOTS=1,
        _load_fastread=lambda: types.SimpleNamespace(make_builder=lambda *a, **kw: reader),
        _E152=bound, _E152_WORK=output, _E152_VARIANT="R", _E152_MANIFEST={})
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<adapted R CPU main>", "exec"), namespace)
    namespace["main"]()
    values, failures = c.read_journal(output,ids,['EXP155','EXP143v2'],{'test':'own'})
    affected = ids if fault == 'load' else ids[:1] if fault else []
    assert set(failures['EXP155']) == set(affected)
    assert set(failures['EXP143v2']) == (set(ids[:1]) if fault in {'decode','empty'} else set())
    for name,expected in [('EXP155',e155),('EXP143v2',e143)]:
        good=[i for i,uid in enumerate(ids) if uid not in failures[name]]
        np.testing.assert_array_equal(values[name][good],expected[good].astype(np.float32))

@pytest.mark.parametrize('fault', [None,'decode','inference','load'])
def test_q_generated_worker_cpu_partial_recovery(tmp_path,monkeypatch,fault):
    import inspect, threading, time, torch
    from tools import exp169_partner_bundle_release as base
    ids,_,_,_,_=c.authenticated_a172(c.A5Q)
    m=c.manifest(c.A5Q/'exp169-partner/release-manifest.json')
    with np.load(c.A5Q/'partner-raw.npz') as raw:
        real=[raw[e['family']][e['fold']+(5 if e['family']=='l2' and e['seed']==43 else 0)] for e in m['entries']]
    journal=c.Journal(tmp_path,ids,{'manifest_sha256':c.MANIFEST_SHA})
    class Corpus:
        def __init__(self,*args): self.ids=ids; self.depth=1; self.spec=None
        def volume(self,uid):
            i=ids.index(uid)
            if i==0 and fault=='decode': raise OSError('injected decode')
            return np.full((1,1,1,2),i)
        def mask(self,uid): return np.ones(1)
    for name,attrs in {
        'raptor_dicom':dict(DicomCorpus=Corpus,DirectoryDicomSource=lambda p:p),
        'raptor_common':dict(select_centers=lambda *a:None,windows_from=lambda v,*a:v),
        'raptor_radimagenet':dict(center3_windows_from=lambda v,*a:v,normalise=lambda v,*a:v),
        'raptor_runtime':dict(normalize_input=lambda v,*a:v),
    }.items(): monkeypatch.setitem(sys.modules,name,types.SimpleNamespace(**attrs))
    for name,fn in dict(device_count=lambda:2,get_device_name=lambda i:'T4 CPU mock',set_device=lambda i:None,
                        reset_peak_memory_stats=lambda i:None,max_memory_allocated=lambda i:0).items():
        monkeypatch.setattr(torch.cuda,name,fn)
    def load(*args):
        if fault=='load': raise RuntimeError('injected load')
        return list(range(20))
    def infer(model,x,*args):
        i=int(x.flat[0])
        if i==0 and model==3 and fault=='inference': raise RuntimeError('injected inference')
        return real[model][i]
    source=c.q_runtime_source(c.v5q.a172_function_source('partner_scores',inspect.getsource(base.partner_scores)))
    scope=dict(base.__dict__,journal=journal,_load_models=load,_infer_one=infer,
               plane_flags=lambda *a:[False],_memory=lambda:{})
    old_threads=torch.get_num_threads(); old_deterministic=torch.are_deterministic_algorithms_enabled()
    try:
        exec(compile(source,'<Q CPU execution>','exec'),scope)
        scores,timings=scope['partner_scores'](tmp_path,tmp_path,ids,m)
    finally:
        torch.set_num_threads(old_threads); torch.use_deterministic_algorithms(old_deterministic)
    values,failures=c.read_journal(tmp_path,ids,[str(i) for i in range(20)],{'manifest_sha256':c.MANIFEST_SHA})
    if fault=='load': assert all(len(f)==len(ids) for f in failures.values())
    else:
        for model in range(20):
            expected={ids[0]} if fault=='decode' or fault=='inference' and model>=3 else set()
            assert set(failures[str(model)])==expected
            good=[i for i,u in enumerate(ids) if u not in expected]
            np.testing.assert_array_equal(values[str(model)][good],real[model][good])
    if fault=='inference': assert np.isfinite(scores['l2'][:3,0]).all() and np.isnan(scores['l2'][3:,0]).all()


def test_stress_selector_extremes_and_no_label_reads(tmp_path):
    from tools import exp169_c85_stress as stress
    comp=tmp_path/'competition'; comp.mkdir()
    rows=[]
    for i in range(40):
        uid=f's{i:02d}'
        for j in range(i%5+1):
            path=comp/'train_series'/uid/f'ser{j}'
            path.mkdir(parents=True)
            for k in range(i+1): (path/f'{k}.dcm').touch()
            rows.append([uid,f'ser{j}','Sagittal','True'])
    stress.table(comp/'train_series.csv',stress.COLS,rows)
    ids=stress.prepare(comp,tmp_path/'standin')
    assert len(ids)==len(set(ids))==32
    assert ids[:4]==['s04','s00','s39','s01']
    selected=c.read_json(tmp_path/'standin/selection.json')
    assert selected['method']=='round-robin max/min series/max/min slices'
    with (tmp_path/'standin/sample_submission.csv').open() as f:
        assert pd.read_csv(f).iloc[:,1:].eq(.5).all().all()
    assert all((tmp_path/'standin/test_series'/uid).is_dir() for uid in ids)


def test_own_arms_exclude_r(tmp_path, monkeypatch):
    # Import the actual helpers: a SimpleNamespace exporting input_record hid
    # the v1.0.3 failure because EXP155's top-level module does not export it.
    from tools import exp155_own0949_oai_release_v12 as helper
    import torch
    ids,_,e155,_,evidence=c.authenticated_a172(c.A5Q)
    (tmp_path/'sample_submission.csv').write_bytes(c.p.csv_bytes(ids,e155))
    ns = {}
    files = {name: tmp_path/name for name in (helper.OWN_NAME, helper.PUBLIC_NAME, helper.MEMBER_NAME)}
    # Binding only; no model load or CUDA operation in this contract regression.
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'is_available', lambda: True)
        helper.bind(ns, {}, files, 'blend3', tmp_path/'own')
    frozen = ns['_E152']
    assert not hasattr(helper, 'input_record')
    c.bind_own(ns,helper,tmp_path/'own',tmp_path,{})
    assert [a['file'] for a in ns['ARMS']]==[helper.OWN_NAME,helper.MEMBER_NAME]
    assert ns['_E152'].members==['EXP155','EXP143v2']
    assert ns['_E152'].checked_model is frozen.checked_model is helper.checked_model
    assert ns['_E152'].input_record is frozen.input_record
    vol = np.ones((96, 2, 2), dtype=np.uint8)
    mask = np.ones(96, dtype=np.uint8)
    inputs = {'_eval_centers': lambda *args: [1, 2]}
    assert ns['_E152'].input_record(inputs, vol, mask, vol, vol) == frozen.input_record(inputs, vol, mask, vol, vol)


@pytest.mark.parametrize('fault', [None, 'own', 'Q', 'all-own'])
def test_saved_predictions_through_journals_and_collect(saved, tmp_path, fault):
    a,b=saved
    ids,e143,e155,q,_=c.authenticated_a172(a)
    _,r,s,_=c.authenticated_p956(b)
    m=c.manifest(a/'exp169-partner/release-manifest.json')
    own=c.Journal(a/'exp169-trio',ids,m['anchor_binding'])
    journal=c.Journal(a/'q-journal',ids,{'manifest_sha256':c.MANIFEST_SHA})
    failures={name:{} for name in c.MEMBERS}
    for name,arr in [('EXP143v2',e143),('EXP155',e155)]:
        for i,uid in enumerate(ids):
            if i==0 and (fault=='all-own' or fault=='own' and name=='EXP155'):
                failures[name][uid]='injected'
                own.record(uid,name,error='injected')
            else: own.record(uid,name,arr[i])
    with np.load(a/'partner-raw.npz') as raw:
        for j,e in enumerate(m['entries']):
            arr=raw[e['family']][e['fold']+(5 if e['family']=='l2' and e['seed']==43 else 0)]
            for i,uid in enumerate(ids):
                if i==0 and fault in {'Q','all-own'}:
                    failures['Q_A5q'][uid]='injected'
                    continue
                journal.record(uid,str(j),arr[i])
    # Simulate abrupt termination during final NPZ serialization.
    (a/'partner-raw.npz').write_bytes(b'PK\x03')
    with (a/'q-journal/predictions.jsonl').open('a') as f: f.write('{"uid":')
    _,e143j,e155j,qj,evidence=c.authenticated_a172(a)
    result,payload=c.collect(a,b,tmp_path/'recovered')
    assert result['status']=='COMPLETE_C85'
    assert {k:set(v) for k,v in evidence['failures'].items()}=={k:set(failures[k]) for k in c.MEMBERS[2:]}
    expected=c.blend(ids,r,s,e143j,e155j,qj,failures)
    assert payload==c.p.csv_bytes(ids,expected)
    if fault is None:
        assert hashlib.sha256(payload).hexdigest()=='a59be5c6066f87b5309e242ecced132fab1894b8dd4b69169e8450a171ddee8a'


def test_truncated_s6_append_retains_complete_studies(saved):
    _,b=saved
    before=c.authenticated_p956(b)
    worker=b/'p956-run/s6/controller/worker0'
    with (worker/'predictions.csv').open('a') as f: f.write('interrupted-uid,0.')
    after=c.authenticated_p956(b)
    np.testing.assert_array_equal(before[2],after[2])
    assert before[-1]['failures']==after[-1]['failures']


def test_r_child_failure_still_runs_s6(tmp_path,monkeypatch):
    out=tmp_path/'public'; work=out/'p956-run'; work.mkdir(parents=True)
    c.write_once(out/'ids.json',c.json_bytes(['a']))
    c.write_once(work/'code-paths.json',c.json_bytes(dict(code=str(work/'code'),anatomy=str(work/'anatomy.pt'))))
    monkeypatch.setattr(c,'runtime_pins',lambda p:None)
    monkeypatch.setattr(c,'competition_root',lambda:tmp_path/'competition')
    monkeypatch.setattr(c.v5q,'_mount',lambda *a:tmp_path)
    monkeypatch.setattr(c.p,'check_pin',lambda *a:None)
    monkeypatch.setattr(c.s6,'verify_bundle',lambda *a:None)
    monkeypatch.setattr(c.s6,'install_wheels',lambda *a:None)
    events=[]
    def r(*a,**kw):
        events.append('R'); raise RuntimeError('child exit 7')
    def s(*a,**kw): events.append('S6')
    monkeypatch.setattr(c.v5q,'run_process_group',r)
    monkeypatch.setattr(c.s6,'infer_studies',s)
    c.run_public(tmp_path/'runtime',out)
    assert events==['R','S6']
    assert c.read_json(out/'R-exit.json')['exit_code']==1
    assert c.read_json(out/'S6-exit.json')['exit_code']==0


def test_stress_bootstrap_dispatches_full_release(tmp_path,monkeypatch):
    blob,pins=c.package()
    code=''.join(c.stress_notebook(blob,pins)['cells'][-1]['source'])
    code=code.replace('/kaggle/working',tmp_path.as_posix())
    events=[]
    def prepare(comp,out,n):
        assert n==32 and comp==tmp_path/'competition'
        events.append('select extremes'); return list(range(32))
    def run(rt,out):
        c.runtime_pins(rt)
        assert c.os.environ['C85_COMPETITION']==str(tmp_path/'c85-stress-data')
        assert out==tmp_path
        events.append('full C85 release')
    importer=builtins.__import__
    def fake_import(name,*args,**kwargs):
        if name=='exp169_release_c85': return types.SimpleNamespace(run_release=run,p=c.p,
            v5q=types.SimpleNamespace(_mount=lambda *a:tmp_path/'competition'))
        if name=='exp169_c85_stress': return types.SimpleNamespace(prepare=prepare)
        return importer(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',fake_import)
    monkeypatch.setenv('C85_COMPETITION','')
    old_path=sys.path[:]
    try: exec(compile(code,'<stress bootstrap>','exec'),{})
    finally: sys.path[:]=old_path
    assert events==['select extremes','full C85 release']
