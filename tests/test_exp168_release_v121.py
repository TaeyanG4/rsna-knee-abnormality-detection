"""Synthetic CPU regression of the real Public3 and packaged wrapper paths."""
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from tools import exp143_v2 as guard_base
from tools import exp168_own0949_oai_jev_release as frozen
from tools import exp168_own0949_oai_jev_release_v121 as release


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    assert not torch.cuda.is_initialized()
    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic release test attempted CUDA/network")
    for name in ("init", "_lazy_init", "is_available", "manual_seed_all", "set_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    yield
    assert not torch.cuda.is_initialized()


def test_guard_facade_uses_real_reader_without_mutation(tmp_path):
    guard = release.load_module("exp168_test_guard", release.ROOT / "tools/exp143_v2.py")
    from tools import exp168_own0949_oai_jev as training_guard
    assert guard.load_registration.__code__ is not guard_base.load_registration.__code__
    assert guard.load_registration.__code__.co_code == training_guard.load_registration.__code__.co_code
    assert guard.registered_run.__code__.co_code == training_guard.registered_run.__code__.co_code
    assert guard.series_records is guard_base.series_records
    assert not hasattr(training_guard, "series_records")
    series = tmp_path / "series.csv"
    series.write_text("StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive\nu1,s1,Sagittal,1\n", encoding="utf-8")
    assert guard.series_records(series)["u1"][0]["SeriesInstanceUID"] == "s1"
    with series.open("a", encoding="utf-8") as stream:
        stream.write("u1,s1,Coronal,0\n")
    with pytest.raises(ValueError, match="invalid series identities"):
        guard.series_records(series)


def test_only_guard_path_is_redirected(tmp_path, monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(release, "_original_load", lambda name, path: calls.append((name, path)) or sentinel)
    path = tmp_path / "exp143_v2.py"
    assert release.load_module("unrelated", path) is sentinel
    assert calls == [("unrelated", path)]


def test_frozen_bindings_and_isolated_version():
    manifest = release.read_json(release.PROTOCOL)
    for name in (release.ADAPTER_NAME, release.BASE_NAME, "exp168_own0949_oai_jev.py", "exp143_v2.py"):
        rel = "tools/" + name
        assert release.sha(release.ROOT / rel) == manifest["source_bindings"][rel]
    assert release.sha(release.ADAPTER_PATH) == release.ADAPTER_SHA256
    assert frozen.VERSION == "1.2.0" and release.VERSION == "1.2.1"
    assert frozen.TOOL != release.TOOL
    assert release._shared is not frozen._shared
    assert release._shared.load_module is release.load_module
    assert release._shared.output_values is release.output_values
    assert release._shared.checked_model is release.checked_model
    assert release._shared.save_raw is release.save_raw


SERVING = '''import numpy as np
from types import SimpleNamespace
class RaptorClassifier:
    def __init__(self, *args, **kwargs): pass
    def load_state_dict(self, state, strict): self.value = float(state['x'][0])
    def eval(self): return self
    def to(self, device):
        assert device == 'cpu'
        return self
def build_backbone(*args, **kwargs): return SimpleNamespace(num_features=1)
def _load_fastread():
    return SimpleNamespace(make_builder=lambda *a, **k: lambda uid, records, root: (
        np.asarray([len(records[uid])], dtype=np.uint8), np.ones(1, dtype=np.uint8)))
FASTREAD_VARIANT = 'synthetic'
SLOTS = 1
FR_HDR = FR_PX = 1
def gpu_windows(vol, mask, *args, **kwargs): return vol.astype(np.float32)
def eval_windows(vol, mask, *args, **kwargs): return vol.astype(np.float32)
def infer_pair_probs(model, plain, mirror, device): return np.full(12, model.value, dtype=np.float32)
def _eval_centers(*args): return [0]
def rankpct(values): return np.argsort(np.argsort(values, axis=0), axis=0) / max(1, len(values)-1)
'''


@pytest.fixture
def staged(tmp_path, monkeypatch):
    # Reuse the frozen EXP168 synthetic stage fixture; substitute only the new
    # private adapter namespace so the actual versioned stage runs.
    spec = importlib.util.spec_from_file_location("exp168_stage_test_helpers",
        release.ROOT / "tests/test_exp168_own0949_oai_jev.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.release = release._base
    helper.test_release_stage_contains_only_inference_assets(tmp_path, monkeypatch)
    # The reused helper intentionally stages a trivial serving source. Keep its
    # artifacts untouched; build a new synthetic stage with executable CPU fakes.
    monkeypatch.setattr(release._base, "public_source", lambda: (SERVING, []))
    monkeypatch.setattr(release._base, "serving_source", lambda source: source)
    member = tmp_path / "member" / release.MEMBER_NAME
    member_ck = release.safe_load(member)
    member_ck["model"]["x"] = torch.tensor([.6])
    torch.save(member_ck, member)
    member_receipt = member.parent / "average-receipt.json"
    receipt = release.read_json(member_receipt)
    receipt["checkpoint_sha256"] = release.sha(member)
    member_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    out = tmp_path / "stage-executable"
    release.stage(SimpleNamespace(average=tmp_path / "average", exp143_checkpoint=member,
                                  out=out, dataset=release.ASSET_DATASET))
    return out, member


def test_stage_archive_notebook_and_import_are_self_contained(staged, tmp_path, monkeypatch):
    stage, _ = staged
    _, report, manifest = release.checked_stage(stage)
    assert report["version"] == manifest["version"] == "1.2.1"
    with zipfile.ZipFile(stage / "dataset/exp168-assets.zip.bin") as archive:
        assert set(archive.namelist()) == set(manifest["asset_sha256"]) | {"release-manifest.json"}
        assert archive.read(release.ADAPTER_NAME) == release.ADAPTER_PATH.read_bytes()
        packaged = tmp_path / "isolated"
        archive.extractall(packaged)
    spec = importlib.util.spec_from_file_location("exp168_packaged_v121", packaged / release.TOOL.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.verify_assets(packaged, manifest)
    assert module.VERSION == "1.2.1"
    assert module._shared is not release._shared
    for variant in release.KERNELS:
        nb = release.read_json(stage / variant / "release.ipynb")
        code = nb["cells"][-1]["source"]
        compile(code, "notebook", "exec")
        assert "import exp168_own0949_oai_jev_release_v121 as _e152" in code
        assert "_hash(_assets/'" + release.TOOL.name + "')" in code
        # Execute only the archive allowlist assignment from the real notebook.
        line = next(line.strip() for line in code.splitlines() if line.strip().startswith("_allowed ="))
        scope = {}
        exec(line, scope)
        assert scope["_allowed"] == set(manifest["asset_sha256"]) | {"release-manifest.json"}
        meta = release.read_json(stage / variant / "kernel-metadata.json")
        assert meta["is_private"] and not meta["enable_internet"]
        assert meta["id"].endswith(release.KERNELS[variant])
        assert release.kernel_metadata(variant)["dataset_sources"][0] == release.ASSET_DATASET
    (packaged / release.ADAPTER_NAME).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Asset hash mismatch"):
        module.verify_assets(packaged, manifest)


@pytest.fixture
def public3_case(staged, tmp_path, monkeypatch):
    stage, member = staged
    comp = tmp_path / "competition"
    comp.mkdir()
    (comp / "test_series").mkdir()
    (comp / "test.csv").write_text("StudyInstanceUID\nu1\nu2\nu3\n", encoding="utf-8")
    series = comp / "test_series.csv"
    series.write_text("StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive\n"
                      "u1,s1,Sagittal,1\nu2,s2,Coronal,0\nu3,s3,Axial,1\n", encoding="utf-8")
    plan = tmp_path / "plan"
    release.parity_plan(SimpleNamespace(stage=stage, out=plan, test_csv=comp / "test.csv", series_csv=series))
    assert "all three final CSVs" in " ".join(release.read_json(plan / "parity-plan.json")["sequence"])
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    public = tmp_path / release.PUBLIC_NAME
    torch.save(dict(model={"x": torch.tensor([.3])}, arch=release.ARCH, res=384), public)
    run = tmp_path / "run"
    run.mkdir()
    shutil.copyfile(tmp_path / "average" / release.OWN_NAME, run / "model_best.pt")
    real_sha = release.sha
    # Only synthetic protocol/checkpoint identity constants are substituted.
    def fixture_sha(path):
        path = Path(path)
        if path == protocol: return "a" * 64
        if path == public: return release.PUBLIC_SHA
        if path == run / "model_best.pt": return "b" * 64
        return real_sha(path)
    monkeypatch.setattr(release._shared, "sha", fixture_sha)
    records_calls = []
    def series_records(path):
        records_calls.append(path)
        return guard_base.series_records(path)
    guard = SimpleNamespace(load_registration=lambda path: {}, registered_run=lambda p: run,
                            base=SimpleNamespace(series_records=series_records))
    ns = {}
    exec(SERVING, ns)
    mismatch = {"enabled": False}
    class Reference(ns["RaptorClassifier"]):
        pass
    references = []
    def build_reference(res):
        model = Reference()
        model.is_member = bool(references)
        references.append(model)
        return model
    def infer_volume(model, vol, mask, device, precision):
        value = model.value - (.01 if mismatch["enabled"] and model.is_member else 0)
        return np.full(12, value, dtype=np.float32)
    training = SimpleNamespace(build_model=build_reference, SPEC="fake", infer_volume=infer_volume,
                               windows=lambda vol, mask, **kw: vol.astype(np.float32))
    monkeypatch.setattr(release, "_original_load", lambda name, path:
        guard if Path(path).name == "exp168_own0949_oai_jev.py" else training)
    def bind(ns, manifest, files, variant, work):
        ns.update(_E152_FILES=files, _E152_MANIFEST=manifest)
    monkeypatch.setattr(release._shared, "bind", bind)
    fake_dicom = SimpleNamespace(
        DirectoryDicomSource=lambda *a, **k: SimpleNamespace(close=lambda: None),
        prepare_study=lambda uid, records, *args: (np.asarray([len(records)], dtype=np.uint8),
                                                   np.ones(1, dtype=np.uint8), None))
    monkeypatch.setitem(sys.modules, "raptor_dicom", fake_dicom)
    monkeypatch.setattr(release._shared, "runtime_info", lambda: dict(
        devices=[dict(name="synthetic T4"), dict(name="synthetic T4")],
        python="synthetic", packages={}, cuda=None, cudnn=None, actual={}))
    args = SimpleNamespace(stage=stage, plan=plan, protocol=protocol, public_checkpoint=public,
        exp143_checkpoint=member, competition=comp, dicom_root=None, device="cpu", out=tmp_path / "public3")
    return args, records_calls, mismatch


def test_public3_runs_all_variants_and_compare_checks_all_arms(public3_case, tmp_path):
    args, records_calls, _ = public3_case
    result = release.public3(args)
    assert result["status"] == "PASS_LOCAL_PUBLIC3"
    assert records_calls == [args.competition / "test_series.csv"]
    assert set(result["comparisons"]) == {"own", "blend", "blend3"}
    assert result["max_abs_exp143"] == 0
    with np.load(args.out / "blend3/raw.npz") as arrays:
        assert set(arrays.files) == {"ids", "targets", "own", "public", "exp143"}
        assert arrays["exp143"].shape == (3, 12)
    compare = SimpleNamespace(stage=args.stage, plan=args.plan, local=args.out,
        own=args.out / "own-predictor", blend=args.out / "blend-predictor",
        blend3=args.out / "blend3-predictor", out=tmp_path / "compare.json")
    passed = release.compare_parity(compare)
    assert passed["status"] == "PASS_FINAL_PUBLIC3"
    assert set(passed["variants"]["blend3"]["raw"]) == {"own", "public", "exp143"}
    # Simulate validly receipted third-arm drift while retaining the same CSV.
    raw = compare.blend3 / "raw.npz"
    with np.load(raw) as z:
        changed = {name: z[name].copy() for name in z.files}
    changed["exp143"] -= .01
    np.savez(raw, **changed)
    receipt = compare.blend3 / "receipt.json"
    payload = release.read_json(receipt)
    payload["files"]["raw.npz"] = release.sha(raw)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    compare.out = tmp_path / "compare-drift.json"
    with pytest.raises(ValueError, match="Parity gate failed"):
        release.compare_parity(compare)
    drift = release.read_json(compare.out)
    assert not drift["variants"]["blend3"]["raw"]["exp143"]["passed"]


def test_public3_rejects_changed_reference_before_reader(public3_case):
    args, calls, _ = public3_case
    args.exp143_checkpoint.write_bytes(b"wrong reference")
    with pytest.raises(ValueError, match="EXP143 reference checkpoint mismatch"):
        release.public3(args)
    assert not calls and not args.out.exists()


def test_public3_fails_on_predictor_difference(public3_case):
    args, _, mismatch = public3_case
    mismatch["enabled"] = True
    with pytest.raises(ValueError, match="Local byte parity failed"):
        release.public3(args)
    result = release.read_json(args.out / "public3-check.json")
    assert result["status"] == "FAIL_LOCAL_PUBLIC3"
    assert result["max_abs_exp143"] > 0


def test_cli_requires_third_reference_and_download_and_plan_is_updated(staged, tmp_path):
    stage, member = staged
    parser = release.parser()
    args = parser.parse_args(["public3", "--stage", str(stage), "--plan", "plan", "--out", "out",
                              "--public-checkpoint", "public.pt", "--exp143-checkpoint", str(member)])
    assert args.exp143_checkpoint == str(member)
    with pytest.raises(SystemExit):
        parser.parse_args(["compare-parity", "--stage", str(stage), "--plan", "plan", "--local", "local",
                           "--own", "own", "--blend", "blend", "--out", "out"])
    plan = tmp_path / "push-plan.ps1"
    release.push_plan(SimpleNamespace(stage=stage, out=plan))
    text = plan.read_text(encoding="utf-8")
    assert "unsupported" not in text
    assert "--blend3" in text and "EXP168-v1-2-1-" in text
