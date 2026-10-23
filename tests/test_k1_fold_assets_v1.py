"""Synthetic completion fixtures are NOT trained fold evidence."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools import k1_fold_dataset_v1 as a
from tools import k1_fold_configs_v1 as c
from tools import k1n_config_v1 as schema

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
TEMPLATES = a.ROOT / "configs/k1n"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(a.encoded(value))


def seal(root, info):
    for name in info["sha256"]:
        info["sha256"][name] = a.sha(root / name)
    write(root / "EXPORT.json", info)
    (root / "SHA256SUMS").write_text(
        "".join(f"{pin}  {name}\n" for name, pin in sorted(info["sha256"].items())), encoding="utf-8")


def synthetic_export(root, fold=1, real_checkpoint=None):
    root.mkdir(parents=True)
    name = f"fit/models/E_prior_s0_f{fold}.pt"
    checkpoint = root / name
    checkpoint.parent.mkdir(parents=True)
    if real_checkpoint:
        shutil.copyfile(real_checkpoint, checkpoint)
    else:
        checkpoint.write_bytes(f"SYNTHETIC NON-MODEL fold {fold}".encode())
    pin = a.sha(checkpoint)
    reg = dict(model_fold=fold, recipe=dict(model_fold=fold, epochs=10, total_steps=100),
               train=[f"synthetic-train-{i}" for i in range((0, 3073, 3100, 3138)[fold])],
               holdout=["synthetic-holdout"], test_fixture=True)
    reg["contract_id"] = a.digest(reg)
    contract = reg["contract_id"]
    write(root / "registration.json", reg)
    (root / "fit/holdout.csv").write_bytes(b"synthetic prediction bytes, no labels\n")
    write(root / "fit/result.json", dict(status="COMPLETE", contract_id=contract, model_fold=fold,
        model_sha256=pin, cpu_test=False, clean_oof=False, exposure="EXPOSED_EXPLORATORY",
        holdout_rows=1, holdout_sha256=a.sha(root / "fit/holdout.csv"), test_fixture=True))
    write(root / "fit/fit-complete.json", dict(status="COMPLETE", contract_id=contract,
        model_fold=fold, model_sha256=pin, epochs=10, optimizer_steps=100,
        train_studies=len(reg["train"]), test_fixture=True))
    write(root / "fit/runtime.json", dict(test_fixture=True))
    training = a.derive_training_source(a.SERVING.read_bytes(), fold)
    write(root / "prepare.json", dict(contract_id=contract,
        trainer_sha256=hashlib.sha256(training).hexdigest(), test_fixture=True))
    info = dict(version="1.0.0", fold=fold, contract_id=contract,
        sha256={n: "" for n in (name, "fit/holdout.csv", "fit/result.json",
                               "fit/fit-complete.json", "fit/runtime.json", "registration.json", "prepare.json")})
    seal(root, info)
    return root / "EXPORT.json"


def packaged(root, fold):
    export = synthetic_export(root / f"f{fold}", fold)
    dest = root / f"dataset{fold}"
    model = a.package(export, dest, f"test-owner/knee-h224-fold{fold}-test-assets")
    return dest, model


@pytest.mark.parametrize("fold", [1, 2, 3])
def test_export_package_roundtrip_inventory_and_immutable(tmp_path, fold):
    dest, model = packaged(tmp_path, fold)
    assert a.verify_dataset(dest) == model
    assert model["checkpoint"] == f"weights/E_prior_s0_f{fold}.pt"
    manifest = a.read(dest / "asset-manifest.json")
    assert manifest["clean_oof"] is False and manifest["model_fold"] == fold
    assert len(list(p for p in dest.rglob("*") if p.is_file())) == 4
    with pytest.raises(ValueError, match="output exists"):
        a.package(tmp_path / f"f{fold}/EXPORT.json", dest, model["dataset_slug"])


@pytest.mark.parametrize("fault", [
    "bytes", "sums", "missing", "extra", "traversal", "placeholder", "contract", "fold",
    "incomplete", "cpu", "epochs", "steps", "train_count", "holdout_hash", "holdout_count",
    "model_hash", "trainer_hash", "duplicate_sum", "duplicate_json", "public", "source",
])
def test_bad_exports_fail_before_destination(tmp_path, fault):
    export = synthetic_export(tmp_path / "export")
    root = export.parent
    info = a.read(export)
    if fault == "bytes":
        (root / "fit/models/E_prior_s0_f1.pt").write_bytes(b"corrupt")
    elif fault == "sums":
        (root / "SHA256SUMS").write_text("", encoding="utf-8")
    elif fault == "missing":
        # Missing source is represented by an expected member renamed in the manifest.
        info["sha256"]["missing.pt"] = info["sha256"].pop("fit/models/E_prior_s0_f1.pt")
        write(export, info)
    elif fault in ("extra", "traversal", "placeholder"):
        if fault == "extra":
            info["sha256"]["credentials.json"] = "a" * 64
        elif fault == "traversal":
            info["sha256"]["../escape"] = info["sha256"].pop("prepare.json")
        else:
            info["sha256"]["prepare.json"] = "REPLACE_WITH_REAL_SHA256"
        write(export, info)
    elif fault == "duplicate_sum":
        sums = root / "SHA256SUMS"
        sums.write_text(sums.read_text() + sums.read_text().splitlines(True)[0], encoding="utf-8")
    elif fault == "duplicate_json":
        export.write_text(export.read_text().replace('"fold": 1,', '"fold": 1, "fold": 2,'), encoding="utf-8")
    elif fault not in ("public", "source"):
        name = "fit/result.json"
        key, value = {
            "contract": ("contract_id", hashlib.sha256(b"bad").hexdigest()),
            "fold": ("model_fold", 2), "incomplete": ("status", "RUNNING"),
            "cpu": ("cpu_test", True), "epochs": ("epochs", 9), "steps": ("optimizer_steps", 99),
            "train_count": ("train_studies", 3072), "holdout_hash": ("holdout_sha256", "b" * 64),
            "holdout_count": ("holdout_rows", 2), "model_hash": ("model_sha256", "b" * 64),
            "trainer_hash": ("trainer_sha256", hashlib.sha256(b"wrong").hexdigest()),
        }[fault]
        if fault in ("epochs", "steps", "train_count"):
            name = "fit/fit-complete.json"
        if fault == "trainer_hash":
            name = "prepare.json"
        receipt = a.read(root / name)
        receipt[key] = value
        write(root / name, receipt)
        seal(root, info)
    dest = tmp_path / "dataset"
    if fault == "public":
        slug = "test-owner/placeholder-assets"
    else:
        slug = "test-owner/knee-fold1-test-assets"
    source = a.SERVING
    if fault == "source":
        source = tmp_path / "wrong-source.py"
        source.write_text("raise RuntimeError('never import me')", encoding="utf-8")
    with pytest.raises(ValueError):
        a.package(export, dest, slug, source)
    assert not dest.exists()


@pytest.mark.parametrize("fault", ["public", "extra", "bytes", "manifest", "exposure"])
def test_dataset_tampering_rejected(tmp_path, fault):
    dest, _ = packaged(tmp_path, 1)
    if fault == "extra":
        (dest / "unapproved.txt").write_text("no")
    elif fault == "bytes":
        (dest / "weights/E_prior_s0_f1.pt").write_bytes(b"no")
    else:
        path = dest / ("dataset-metadata.json" if fault == "public" else "asset-manifest.json")
        value = a.read(path)
        if fault == "public":
            value["isPrivate"] = False
        elif fault == "manifest":
            value["files"]["../escape"] = value["files"].pop("source/compact_train.py")
        else:
            value["clean_oof"] = True
        write(path, value)
    with pytest.raises(ValueError):
        a.verify_dataset(dest)


@pytest.mark.parametrize("name,n,expected", [
    ("five", 5, [8.64, 9.04]), ("six", 6, [9.288, 9.688]),
])
def test_runtime_conservative_hold_and_no_placeholder_config(tmp_path, name, n, expected):
    path = TEMPLATES / f"{name}-model-folds-stride2.template.json"
    value = c.template(path)
    assert value["model_count"] == n
    result = c.projection(value)
    assert result["projected_hours"] == expected and result["status"] == "HOLD"
    with pytest.raises(ValueError):
        schema.validate(value)
    with pytest.raises(ValueError, match="exceeds 8.0h"):
        c.run("prepare", path, [tmp_path / "nonexistent"] * (n - 3), tmp_path / "out",
              "test-owner/knee-folds-test", "1.1.999-test")
    assert not (tmp_path / "out").exists()


def scoped_template(tmp_path, count=5):
    name = "five" if count == 5 else "six"
    value = c.template(TEMPLATES / f"{name}-model-folds-stride2.template.json")
    value["runtime"]["parent_k1_models_stride3"] = 3
    value["runtime"]["basis"] = "SYNTHETIC TEST ONLY: assume parent contains three stride3 K1 classifiers."
    path = tmp_path / f"scope{count}.json"
    write(path, value)
    return path


def test_parent_scope_math_upper_not_lower(tmp_path):
    five = c.projection(c.template(scoped_template(tmp_path)))
    six = c.projection(c.template(scoped_template(tmp_path, 6)))
    assert five["projected_hours"] == [7.29, 7.69] and five["status"] == "PASS_ESTIMATE_ONLY"
    assert six["projected_hours"] == [7.938, 8.338] and six["status"] == "HOLD"
    # Inclusive exact boundary; no rounding of 8.000001 into acceptance.
    value = c.template(scoped_template(tmp_path))
    value["runtime"]["parent_hours"] = [7.21, 7.21]
    assert c.projection(value)["status"] == "PASS_ESTIMATE_ONLY"
    value["runtime"]["parent_hours"] = [7.21000001, 7.21000001]
    assert c.projection(value)["status"] == "HOLD"


@pytest.mark.parametrize("field,value", [
    ("parent_k1_models_stride3", True), ("parent_k1_models_stride3", 1),
    ("parent_hours", [6.0, 6.5]), ("parent_hours", [float("nan"), 6.9]),
    ("limit_hours", 9), ("stride2_multiplier", 1), ("basis", "TODO"),
])
def test_template_runtime_assumption_tampering(tmp_path, field, value):
    template = a.read(TEMPLATES / "five-model-folds-stride2.template.json")
    template["runtime"][field] = value
    path = tmp_path / "tampered.json"
    if field == "parent_hours" and str(value[0]) == "nan":
        path.write_text('{"schema": "k1-fold-template/v1", "runtime": {"parent_hours": [NaN, 6.9]}}')
    else:
        write(path, template)
    with pytest.raises(ValueError):
        c.template(path)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/absolute", "weights/../source/x.py",
                                 "weights//model.pt", r"weights\\model.pt", "."])
def test_paths_rejected_before_access(tmp_path, name):
    with pytest.raises(ValueError):
        a.member(tmp_path, name)


def test_real_base_pins_config_materialization_stage_compatibility_and_drift(tmp_path):
    dirs = [packaged(tmp_path, f)[0] for f in (1, 2)]
    template = scoped_template(tmp_path)
    out = tmp_path / "config"
    args = (template, dirs, out, "test-owner/knee-folds-cpu-test", "1.1.999-folds-cpu-test")
    assert c.run("prepare", *args)["runtime"]["projected_hours"] == [7.29, 7.69]
    assert c.run("validate", *args)["status"] == "PASS_LOCAL_CONFIG"
    config = schema.validate(a.read(out / "config.json"))
    assert [m["id"] for m in config["models"]] == [
        "k1_v2", "k1_v3g_seed0", "k1_v3g_seed1", "k1_v3g_fold1", "k1_v3g_fold2"]
    from tools import k1n_stage_v1 as stage
    # CPU-only generation in memory. Does not create a release stage.
    for diagnostic in (False, True):
        payload = stage.package(config, diagnostic)
        assert payload[1]["is_private"]
        assert dirs[0].name not in payload[1]["dataset_sources"]
        assert "test-owner/knee-h224-fold1-test-assets" in payload[1]["dataset_sources"]
    with pytest.raises(ValueError, match="output exists"):
        c.run("prepare", *args)
    (out / "config.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="artifact drift"):
        c.run("validate", *args)


def test_reversed_folds_rejected(tmp_path):
    dirs = [packaged(tmp_path, f)[0] for f in (2, 1)]
    with pytest.raises(ValueError, match="fold dataset identity"):
        c.run("prepare", scoped_template(tmp_path), dirs, tmp_path / "config",
              "test-owner/knee-folds-cpu-test", "1.1.999-folds-cpu-test")


def test_real_seed_checkpoint_roundtrip_cpu_smoke(tmp_path):
    """Real bytes, synthetic export receipts; does not certify a completed H224 fold."""
    seed = c.BASE_DIRS[2]
    assert a.verify_dataset(seed) == schema.current()["models"][2]
    export = synthetic_export(tmp_path / "real-bytes-synthetic-export", real_checkpoint=
                              seed / "weights/E_prior_s0_f0.pt")
    dest = tmp_path / "real-bytes-test-dataset"
    model = a.package(export, dest, "test-owner/knee-real-bytes-cpu-smoke")
    assert model["checkpoint_sha256"] == schema.current()["models"][2]["checkpoint_sha256"]
    assert a.sha(dest / model["checkpoint"]) == a.sha(seed / "weights/E_prior_s0_f0.pt")


def test_cli_cpu_commands(tmp_path):
    export = synthetic_export(tmp_path / "cli-export")
    output = tmp_path / "cli-dataset"
    def cli(module, *args, code=0):
        run = subprocess.run([sys.executable, "-B", "-m", module, *map(str, args)],
                             cwd=a.ROOT, capture_output=True, text=True, timeout=60)
        assert run.returncode == code, run.stderr
        return run.stdout
    cli("tools.k1_fold_dataset_v1", "verify-export", "--export", export)
    cli("tools.k1_fold_dataset_v1", "package", "--export", export, "--output", output,
        "--dataset-id", "test-owner/knee-fold-cli-assets")
    cli("tools.k1_fold_dataset_v1", "verify-dataset", "--dataset", output)
    cli("tools.k1_fold_configs_v1", "project", "--template",
        TEMPLATES / "five-model-folds-stride2.template.json", code=2)
    path = scoped_template(tmp_path)
    other, _ = packaged(tmp_path, 2)
    common = ("--template", path, "--fold-datasets", output, other, "--output", tmp_path / "config",
              "--kernel-id", "test-owner/knee-fold-cli-test", "--release-version", "1.1.999-fold-cli-test")
    cli("tools.k1_fold_configs_v1", "prepare", *common)
    cli("tools.k1_fold_configs_v1", "validate", *common)
