"""CPU tiny fakes only. No CUDA initialization/query, network, real checkpoint or data reads."""
from __future__ import annotations

import ast
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import exp152_own0949_release as m


@pytest.fixture(autouse=True)
def forbid_gpu_and_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU-only test attempted CUDA/network")
    for name in ("_lazy_init", "init", "is_available", "is_initialized", "device_count", "get_device_name",
                 "get_device_capability", "manual_seed", "manual_seed_all", "set_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def tiny_state(value, tracked):
    return OrderedDict(weight=torch.tensor([value, value+1], dtype=torch.float32),
                       **{"bn.running_mean": torch.tensor([value]),
                          "bn.running_var": torch.tensor([value+1]),
                          "bn.num_batches_tracked": torch.tensor(tracked, dtype=torch.long),
                          "fixed": torch.tensor([7], dtype=torch.long)})


def rows(values):
    return np.asarray([[v] * 12 for v in values], np.float32)


def test_import_and_cli_are_cpu_passive():
    # Reimport while every CUDA query is forbidden. Parser has no execution side effects.
    spec = importlib.util.spec_from_file_location("exp152_passive_test", m.TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VERSION == "1.0.1"
    assert module.parser().parse_args(["average", "--out", "unused"]).command == "average"
    assert not torch.cuda.__dict__.get("_initialized", False)


def test_safe_load_supports_frozen_trainer_torchversion_without_cuda(tmp_path):
    from torch.torch_version import TorchVersion
    checkpoint = tmp_path / "version.pt"
    torch.save({"model": {"x": torch.tensor([1.])}, "provenance": {"torch": TorchVersion("2.8.0")}}, checkpoint)
    actual = m.safe_load(checkpoint)
    assert actual["provenance"]["torch"] == "2.8.0"
    assert actual["model"]["x"].item() == 1


def test_average_matches_frozen_helper_with_float64_cancellation():
    states = [tiny_state(2., 14), tiny_state(8., 11), tiny_state(5., 12)]
    # Discriminates float64 accumulation from float32 accumulation.
    for s, value in zip(states, (1e8, 1, -1e8)):
        s["weight"] = torch.tensor([value], dtype=torch.float32)
    tree = ast.parse((ROOT / "tools/exp112_full_swa.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "swa_average")
    scope = {"OrderedDict": OrderedDict}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "frozen-swa-helper", "exec"), scope)
    actual, expected = m.average_states(states), scope["swa_average"](states)
    assert list(actual) == list(expected)
    assert all(torch.equal(actual[k], expected[k]) for k in actual)
    assert actual["weight"].item() == pytest.approx(1/3)
    assert actual["bn.num_batches_tracked"].item() == 12


@pytest.mark.parametrize("fault", ["keys", "shape", "dtype", "integer", "nan"])
def test_average_fails_on_drift(fault):
    states = [tiny_state(1., 14), tiny_state(2., 11), tiny_state(3., 12)]
    if fault == "keys":
        del states[1]["weight"]
    elif fault == "shape":
        states[1]["weight"] = torch.ones(3)
    elif fault == "dtype":
        states[1]["weight"] = states[1]["weight"].double()
    elif fault == "integer":
        states[1]["fixed"] += 1
    else:
        states[1]["weight"][0] = float("nan")
    with pytest.raises(ValueError):
        m.average_states(states)


def test_bn_replay_preserved_but_no_non_bn_drift_allowed():
    avg = m.average_states([tiny_state(1., 14), tiny_state(2., 11), tiny_state(3., 12)])
    final = OrderedDict((k, v.clone()) for k, v in avg.items())
    bn_keys = {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
    final["bn.running_mean"].fill_(99.)
    final["bn.running_var"].fill_(12.)
    final["bn.num_batches_tracked"].fill_(2174)
    rebuilt = m.merge_replayed_bn(avg, final, bn_keys, 2174)
    assert all(torch.equal(rebuilt[k], final[k]) for k in final)
    assert rebuilt["bn.running_mean"].item() == 99.
    final["weight"][0] += .001
    with pytest.raises(ValueError, match="non-BN"):
        m.merge_replayed_bn(avg, final, bn_keys, 2174)


@pytest.mark.parametrize("fault", ["tracked", "nan", "negative_var", "keys"])
def test_incomplete_bn_fails_closed(fault):
    avg = tiny_state(1., 12)
    final = OrderedDict((k, v.clone()) for k, v in avg.items())
    bn_keys = {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
    final["bn.num_batches_tracked"].fill_(2174)
    if fault == "tracked":
        final["bn.num_batches_tracked"].fill_(2173)
    elif fault == "nan":
        final["bn.running_mean"].fill_(float("nan"))
    elif fault == "negative_var":
        final["bn.running_var"].fill_(-1)
    else:
        del final["weight"]
    with pytest.raises(ValueError):
        m.merge_replayed_bn(avg, final, bn_keys, 2174)


def own_checkpoint(state=None):
    return dict(model=state or tiny_state(1., 2174), arch=m.ARCH, res=384, inference_res=320,
                lab=m.TARGETS, contract=m.CONTRACT, k_eval=94, norm="imagenet", window_policy="legacy",
                oai=False, checkpoint_kind="fixed-swa-bn", swa_epochs_1based=m.EPOCHS, seed=42,
                provenance={"protocol_sha256": "a"*64, "run_dir": m.RUN})


@pytest.mark.parametrize("key,value", [("arch", "other"), ("res", 320), ("k_eval", 24),
                                     ("oai", True), ("swa_epochs_1based", [11,12,14]),
                                     ("lab", list(reversed(m.TARGETS))), ("seed", 43)])
def test_own_metadata_fails_closed(key, value):
    ck = own_checkpoint()
    m.own_meta(ck, "a"*64)
    ck[key] = value
    with pytest.raises(ValueError):
        m.own_meta(ck, "a"*64)


def public_rank(a):
    return a.argsort(0).argsort(0).astype(np.float64) / max(1, len(a)-1)


def test_two_rank_contracts_are_explicit_and_per_target():
    a, b = rows([.4, .1, .1, .9]), rows([.1, .8, .4, .9])
    b[:, 1] = b[::-1, 1]
    assert np.array_equal(m.output_values([a], "own", public_rank), public_rank(a))
    blend = m.output_values([a, b], "blend", public_rank)
    assert blend[:, 0].tolist() == pytest.approx([1/3, 5/12, 1/4, 1])
    assert blend[:, 1].tolist() != blend[:, 0].tolist()
    assert not np.array_equal(blend, public_rank(blend))  # no second rank
    assert m.percentile_average([1]) == [0]
    assert m.percentile_average([1, 1, 1]) == [.5, .5, .5]
    assert m.EPOCHS == [14, 11, 12]


@pytest.mark.parametrize("arms", [[rows([float("nan")])], [rows([1.1])], [np.zeros((3, 11))]])
def test_invalid_output_never_becomes_submission(arms):
    with pytest.raises(ValueError):
        m.output_values(arms, "own", public_rank)


def test_generated_source_compiles_and_math_helpers_unchanged():
    source, _ = m.public_source()  # static JSON read only, never execute third-party code
    generated = m.serving_source(source)
    compile(generated, "public-serving.py", "exec")
    def functions(code):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    before, after = functions(source), functions(generated)
    assert set(before) <= set(after)  # B: v1.0.1 adds exactly one helper (checked below)
    assert {k for k in before if before[k] != after[k]} == {"load_model", "find_weight_file", "main"}
    assert set(after) - set(before) == {"_exp152_bounded_preprocess"}
    assert "del xw, xm" in generated
    assert "_exp152_errors.append" in generated and "not _exp152_errors" in generated
    assert "sub.head()" not in generated
    main = next(n for n in ast.parse(generated).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    # The shared GPU-window pair is outside the arm loop.
    consumer = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == "consumer")
    arm_loop = next(n for n in ast.walk(consumer) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "gpu_windows" for n in ast.walk(arm_loop))
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "infer_pair_probs" for n in ast.walk(arm_loop))


def test_preprocessing_batches_bound_decoded_results_and_preserve_order():
    source, _ = m.public_source()
    generated = m.serving_source(source)
    tree = ast.parse(generated)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_exp152_bounded_preprocess")
    scope = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "bounded-prep", "exec"), scope)
    assert generated.index("_load_fastread()  # Materialize/import once") < generated.index(
        'with mp.get_context("fork").Pool')
    assert "_exp152_bounded_preprocess(pool, test_ids, _pre_one, QUEUE_MAX)" in generated

    ids = [f"study-{i}" for i in range(11)]
    pos = {sid: i for i, sid in enumerate(ids)}
    expected = [(sid, sid.upper(), i % 2) for i, sid in enumerate(ids)]
    decoded = {item[0]: item for item in expected}
    class FakePool:
        batch_sizes = []
        def imap_unordered(self, decoder, batch, chunksize):
            assert chunksize == 2
            self.batch_sizes.append(len(batch))
            ready = [decoder(sid) for sid in batch]
            yield from reversed(ready)  # mimic completion order differing from input order
    pool = FakePool()
    ordered = [None] * len(ids)
    for sid, vol, mask in scope["_exp152_bounded_preprocess"](pool, ids, decoded.__getitem__, 3):
        ordered[pos[sid]] = (sid, vol, mask)
    assert ordered == expected
    assert pool.batch_sizes == [3, 3, 3, 2]
    assert max(pool.batch_sizes) <= 3


def test_source_drift_rejected():
    with pytest.raises(ValueError, match="marker"):
        m.replace_once("x x", "x", "y")
    with pytest.raises(ValueError):
        m.replace_function("def x(): pass\n", "load_model", "def load_model(): pass")


def test_notebooks_compile_slug_private_and_approval_marking():
    _, credits = m.public_source()
    for variant in m.KERNELS:
        nb, title, desc = m.notebook(variant, m.ASSET_DATASET, "a"*64, "b"*64, "c"*64, credits)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "release.ipynb", "exec")
                assert "zip.bin" in cell["source"] and "extractall" in cell["source"]
        meta = m.kernel_metadata(variant)
        assert meta["title"] == meta["id"].split("/")[1] == m.KERNELS[variant]
        assert meta["is_private"] and not meta["enable_internet"] and meta["enable_gpu"]
        assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["docker_image"] == m.DOCKER
        assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"]
        if variant == "blend":
            assert "OAI-DERIVED" in title and "TAEYANG APPROVAL" in title and "explicit approval" in desc
            assert m.PUBLIC_DATASET in meta["dataset_sources"]
        else:
            assert m.PUBLIC_DATASET not in meta["dataset_sources"]


def staged_fake(tmp_path):
    avg = tmp_path / "avg"
    avg.mkdir()
    ck = own_checkpoint()
    torch.save(ck, avg / m.OWN_NAME)
    m.write_json(avg / "average-receipt.json", dict(status="VERIFIED_FIXED_SWA_BN", epochs_1based=m.EPOCHS,
                  checkpoint_sha256=m.sha(avg / m.OWN_NAME), protocol_sha256="a"*64,
                  state_schema=m.state_schema(ck["model"]), source_final_sha256="b"*64, source_bindings={}))
    dest = tmp_path / "stage"
    result = m.stage(SimpleNamespace(average=avg, out=dest, dataset=m.ASSET_DATASET))
    assert result["status"] == "STAGED_NOT_UPLOADED"
    return dest


def plan_args(dest, out, tmp_path):
    test_csv, series_csv = tmp_path / "test.csv", tmp_path / "test_series.csv"
    test_csv.write_text("StudyInstanceUID\na\nb\nc\n")
    series_csv.write_text("StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive\n")
    return SimpleNamespace(stage=dest, out=out, test_csv=test_csv, series_csv=series_csv)


def test_stage_zipbin_bindings_no_training_inputs_and_create_only(tmp_path):
    dest = staged_fake(tmp_path)
    path, report, manifest = m.checked_stage(dest)
    assert report["private_required"] and manifest["epochs_1based"] == [14, 11, 12]
    with zipfile.ZipFile(dest / "dataset/exp152-assets.zip.bin") as z:
        assert set(z.namelist()) == {m.OWN_NAME, m.TOOL.name, "public-serving.py", "release-manifest.json"}
    assert {p.name for p in (dest / "dataset").iterdir()} == {"dataset-metadata.json", "exp152-assets.zip.bin"}
    with pytest.raises(ValueError, match="fresh"):
        m.fresh_dir(dest)
    with (dest / "assets/public-serving.py").open("a") as f:
        f.write("# tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        m.checked_stage(dest)


def test_prefrozen_parity_plan_and_plan_only_upload_commands(tmp_path):
    dest = staged_fake(tmp_path)
    frozen = tmp_path / "parity"
    result = m.parity_plan(plan_args(dest, frozen, tmp_path))
    assert result["status"] == "FROZEN_NOT_RUN"
    assert m.read_json(frozen / "tol.json")["max_abs"] == 2e-3
    assert m.read_json(frozen / "tol.json")["mean_abs"] == 5e-4
    _, report, _ = m.checked_stage(dest)
    m.verify_plan(frozen, report)
    planfile = tmp_path / "push-plan.ps1"
    m.push_plan(SimpleNamespace(stage=dest, out=planfile))
    text = planfile.read_text()
    assert "Push-Location -LiteralPath" in text and "datasets create -p ." in text
    assert " -m kaggle kernels push -p " in text
    assert "--public" not in text
    assert all(line.startswith("#") for line in text.splitlines() if "kaggle_submit_once.py" in line)
    assert not any(line.startswith("&") and "kaggle_push.py" in line for line in text.splitlines())
    (frozen / "tol.json").write_text(json.dumps(dict(m.TOL, max_abs=.01)))
    with pytest.raises(ValueError, match="tolerance"):
        m.verify_plan(frozen, report)


def test_checked_model_sha_arch_keys_before_device(tmp_path, monkeypatch):
    path = tmp_path / m.OWN_NAME
    ck = own_checkpoint()
    torch.save(ck, path)
    manifest = {"checkpoints": {m.OWN_NAME: {"sha256": m.sha(path)}},
                "protocol_sha256": "a"*64, "own_state_schema": m.state_schema(ck["model"])}
    called = []
    def construct(*a, **kw):
        called.append(True)
        raise RuntimeError("reached backbone")
    ns = {"_E152_MANIFEST": manifest, "build_backbone": construct}
    with pytest.raises(RuntimeError, match="reached backbone"):
        m.checked_model(ns, path, m.ARCH, 384, "cpu")
    called.clear()
    manifest["own_state_schema"] = {}
    with pytest.raises(ValueError, match="keys/shapes/dtypes"):
        m.checked_model(ns, path, m.ARCH, 384, "cpu")
    assert not called
    manifest["checkpoints"][m.OWN_NAME]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="SHA"):
        m.checked_model(ns, path, m.ARCH, 384, "cpu")


def test_csv_bytes_and_array_hashes(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    m.csv_write(a, ["u1", "u2", "u3"], rows([0, .5, 1]))
    m.csv_write(b, ["u1", "u2", "u3"], rows([0, .5, 1]))
    assert a.read_bytes() == b.read_bytes()
    assert m.array_sha(np.zeros((2,3), np.uint8)) != m.array_sha(np.zeros((3,2), np.uint8))
    assert m.array_sha(torch.zeros(2,3)) == m.array_sha(np.zeros((2,3), np.float32))
    with pytest.raises(ValueError, match="exists"):
        m.csv_write(a, ["u"], rows([.2]))


def test_projection_basis_and_no_claim_of_measured_exp152():
    p = m.projection()
    assert p["own_seconds"] == pytest.approx(2156)
    assert p["blend_seconds"] == pytest.approx(4192)
    assert p["conservative_blend_seconds"] == pytest.approx(6228)
    assert "unmeasured" in p["caveat"]


@pytest.mark.parametrize("fault", [None, "not_complete", "epoch_sha", "final_sha", "bn_cohort", "non_bn"])
def test_average_command_reconstructs_completed_trainer_only(tmp_path, monkeypatch, fault):
    import hashlib
    model = torch.nn.Sequential(OrderedDict(linear=torch.nn.Linear(2, 2), bn=torch.nn.BatchNorm1d(2)))
    # All RNG use above is CPU-only. No forward; exact fake state is enough.
    run = tmp_path / "run"
    run.mkdir()
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    states, epoch_sha = [], {}
    for epoch in m.EPOCHS:
        state = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        for k in state:
            if state[k].is_floating_point():
                state[k].fill_(epoch / 10.)
        state["bn.num_batches_tracked"].fill_(epoch)
        states.append(state)
        torch.save({"model": state, "epoch": epoch}, run / f"epoch_{epoch:02d}.pt")
        epoch_sha[str(epoch)] = m.sha(run / f"epoch_{epoch:02d}.pt")
    avg = m.average_states(states)
    final = OrderedDict((k, v.clone()) for k, v in avg.items())
    final["bn.running_mean"].fill_(20.)
    final["bn.running_var"].fill_(30.)
    final["bn.num_batches_tracked"].fill_(2)
    if fault == "non_bn":
        final["linear.weight"].add_(.25)
    ck = own_checkpoint(final)
    ck["provenance"]["protocol_sha256"] = m.sha(protocol)
    torch.save(ck, run / "model_best.pt")
    summary = dict(status="COMPLETE_TRAINING_NO_EVALUATION", epochs=16,
                   model_sha256=m.sha(run / "model_best.pt"), identity=ck["provenance"],
                   swa_member_sha256=epoch_sha, bn=dict(studies=3, unique_studies=3, batches=2,
                   num_batches_tracked=[2], mode="train", gradients="none", bn_modules=1,
                   studies_sha256=hashlib.sha256(b"a\nb\nc").hexdigest()))
    if fault == "not_complete":
        summary["status"] = "TRAINING"
    elif fault == "epoch_sha":
        summary["swa_member_sha256"]["11"] = "bad"
    elif fault == "final_sha":
        summary["model_sha256"] = "bad"
    elif fault == "bn_cohort":
        summary["bn"]["studies_sha256"] = "bad"
    m.write_json(run / "summary.json", summary)
    p = dict(recipe={"swa_1based": m.EPOCHS, "bn_batch": 2}, run_dirs={"local": m.RUN},
             train_ids=["a", "b", "c"], source_bindings={})
    guard = SimpleNamespace(load_registration=lambda path: p, registered_run=lambda p, lane: run)
    training = SimpleNamespace(build_model=lambda res: model)
    monkeypatch.setattr(m, "load_module", lambda name, path: guard if "guard" in name else training)
    args = SimpleNamespace(protocol=protocol, checkpoint=run / "model_best.pt", out=tmp_path / "averaged")
    if fault:
        with pytest.raises(ValueError):
            m.average(args)
        assert not args.out.exists()
    else:
        result = m.average(args)
        assert result["status"] == "VERIFIED_FIXED_SWA_BN"
        got = m.safe_load(args.out / m.OWN_NAME)
        assert all(torch.equal(got["model"][k], final[k]) for k in final)
        receipt = m.read_json(args.out / "average-receipt.json")
        assert receipt["epochs_1based"] == [14,11,12] and receipt["epoch_sha256"] == epoch_sha
        assert receipt["bn_buffer_keys"] == ["bn.num_batches_tracked", "bn.running_mean", "bn.running_var"]


@pytest.mark.parametrize("fault", [None, "input", "raw", "csv", "gpu", "binding", "tol"])
def test_full_parity_gate_accepts_only_bound_outputs(tmp_path, monkeypatch, fault):
    dest = staged_fake(tmp_path)
    _, report, manifest = m.checked_stage(dest)
    plan = tmp_path / "plan"
    m.parity_plan(plan_args(dest, plan, tmp_path))
    local = tmp_path / "local"
    local.mkdir()
    ids = ["a", "b", "c"]
    rt = {"python": "test", "packages": {}, "cuda": "test", "cudnn": 0, "actual": {},
          "devices": [{"name": "Tesla T4", "capability": [7,5]}] * 2}
    monkeypatch.setattr(m, "runtime_info", lambda: rt)
    inputs = {uid: {name: uid + name for name in ("volume", "mask", "centers", "plain", "mirror")} for uid in ids}
    for variant in m.KERNELS:
        for remote in (False, True):
            output = tmp_path / (variant + "-t4") if remote else local / variant
            output.mkdir()
            arms = [rows([.1,.3,.6])] + ([rows([.6,.3,.1])] if variant == "blend" else [])
            if remote:
                arms[0][0,0] += .003 if fault == "raw" else .0001
            m.csv_write(output / "submission.csv", ids, m.output_values(arms, variant, public_rank))
            rec_inputs = json.loads(json.dumps(inputs))
            if remote and fault == "input":
                rec_inputs["a"]["volume"] = "bad"
            m.save_raw(output, ids, arms, rec_inputs, manifest, variant)
            if remote and fault == "binding":
                binding_path = output / "binding.json"
                binding = m.read_json(binding_path)
                binding["own_sha256"] = "0"*64
                binding_path.write_text(json.dumps(binding))
            if remote and fault == "gpu":
                record = m.read_json(output / "runtime.json")
                record["devices"][0]["name"] = "RTX 3090"
                (output / "runtime.json").write_text(json.dumps(record))
            if remote and fault == "csv":
                with (output / "submission.csv").open("ab") as f:
                    f.write(b"\n")
            m.finish_receipt(output, 1.)
    m.write_json(local / "public3-check.json", dict(status="PASS_LOCAL_PUBLIC3", manifest_sha256=report["manifest_sha256"],
                 plan_sha256=m.sha(plan / "parity-plan.json")))
    if fault == "tol":
        (plan / "tol.json").write_text(json.dumps(dict(m.TOL, max_abs=.1)))
    args = SimpleNamespace(stage=dest, plan=plan, local=local, own=tmp_path / "own-t4",
                           blend=tmp_path / "blend-t4", out=tmp_path / "parity-result.json")
    if fault:
        with pytest.raises(ValueError):
            m.compare_parity(args)
    else:
        result = m.compare_parity(args)
        assert result["status"] == "PASS_FINAL_PUBLIC3"
        assert all(v["final_csv_exact"] and v["input_hashes_exact"] for v in result["variants"].values())
