"""v1.3.0: CPU post-fit contracts; fixtures never qualify a release.

Changelog v1.3.0: cover the equal four-fold0 P25 v10 fullbag family.

Changelog v1.2.0: map timeout-only v10 templates to their versioned parents.

Changelog v1.1.0: enforce terminal-only remote execution and inherited projection basis.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools import k1n_stage_v1 as stage
from tools import l1_rco_k1_foldgate_v1 as gate
from tools import l1_rco_k1_fullbag_v1 as fullbag
from tools import k1_fold_dataset_v1 as assets
from test_k1_fold_assets_v1 import synthetic_export, seal, write

ROOT = stage.ROOT
POST = ROOT / ".local/scratch/astra-b2/fullbag-post-v1"
FILL = ROOT / ".local/scratch/slot-b/submit-1021/fill_template_v1.py"
TEMPLATES = sorted((ROOT / "configs/k1n").glob("*-fb-*.template.json"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


terminal = load("fullbag_terminal_test", POST / "fullbag_terminal_v1.py")


def waiter_fixture(root, state="STARTED"):
    owner = root / terminal.OWNER
    directory = owner / "artifacts/logs" / terminal.TAG
    command = ["/usr/bin/python", str(root / "tools/l1_rco_k1_fullbag_v1.py"),
               "run", "--mode", "fullbag", "--cache", "/cache", "--review", "/review"]
    digest = hashlib.sha256(json.dumps(dict(command=command, cwd=str(owner.resolve())),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    job = dict(schema="gpu-job/v2", tag=terminal.TAG, run_id="test-run", command=command,
               cwd=str(owner), command_sha256=digest, pid=100, create_time=1.)
    ledger = dict(state=state, jobs=[], spec=dict(jobs=[dict(tag=terminal.TAG,
        argv=["python", "gpu_job.py", "--", *command])]))
    receipt = dict(schema="gpu-job-receipt/v1", tag=terminal.TAG, run_id="test-run",
        command_sha256=digest, wrapper_identity=dict(pid=100, create_time=1.),
        state="SUCCESS", exit_code=0, ended="2026-10-21T20:00:00")
    write(directory / "job.json", job)
    write(root / terminal.TASK / "fit/status.json", dict(phase="training"))
    if state == "SUCCESS":
        write(directory / "receipt.json", receipt)
        (directory / "SUCCESS").write_text("exit 0 at " + receipt["ended"])
        ledger["jobs"] = [dict(run_id="test-run", receipt=receipt)]
    write(root / terminal.TASK / "after-f3-once.json", ledger)
    return directory, ledger, receipt


@pytest.mark.parametrize("state,expected", [("STARTED", "WAIT"), ("SUCCESS", "SUCCESS"),
                                            ("STOPPED", "FAIL_PRODUCTION")])
def test_terminal(tmp_path, state, expected):
    waiter_fixture(tmp_path, state)
    assert terminal.probe(tmp_path, terminal.binding(tmp_path)) == expected


@pytest.mark.parametrize("fault", ["failed", "stall", "changed_run", "receipt_run", "marker", "command", "failure_file"])
def test_terminal_fail_closed(tmp_path, fault):
    directory, _, receipt = waiter_fixture(tmp_path, "SUCCESS" if fault in ("receipt_run", "marker") else "STARTED")
    bound = terminal.binding(tmp_path)
    if fault == "failed":
        (directory / "FAILED").write_text("exit 1")
    elif fault == "failure_file":
        write(tmp_path / terminal.TASK / "fit/failure.json", dict(status="FAILED"))
    elif fault == "stall":
        path = tmp_path / terminal.TASK / "fit/status.json"
        os.utime(path, (time.time()-1801, time.time()-1801))
    elif fault in ("changed_run", "command"):
        job = terminal.read(directory / "job.json")
        if fault == "changed_run":
            job["run_id"] = "other"
        else:
            job["command"][2] = "infer"
        write(directory / "job.json", job)
    elif fault == "receipt_run":
        receipt["run_id"] = "other"
        write(directory / "receipt.json", receipt)
    else:
        (directory / "SUCCESS").write_text("exit 0 at other")
    if fault in ("changed_run", "receipt_run", "marker", "command"):
        with pytest.raises(AssertionError):
            terminal.probe(tmp_path, bound)
    else:
        assert terminal.probe(tmp_path, bound).startswith("FAIL_")


@pytest.mark.parametrize("fault", [None, "existing", "production", "stall"])
def test_waiter_shell_order_and_stop(tmp_path, fault):
    bash = Path("C:/Program Files/Git/usr/bin/bash.exe")
    if not bash.is_file():
        import shutil
        bash = shutil.which("bash")
    assert bash, "bash required for waiter CPU checks"
    project = tmp_path / "project"
    work = tmp_path / "waiter"
    work.mkdir()
    waiter_fixture(project, "STOPPED" if fault == "production" else "STARTED" if fault == "stall" else "SUCCESS")
    if fault == "stall":
        status = project / terminal.TASK / "fit/status.json"
        os.utime(status, (time.time()-1801, time.time()-1801))
    if fault == "existing":
        (work / "wait_fullbag_v1.claim").mkdir()
    before = {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()}
    (work / "fullbag_terminal_v1.py").write_bytes((POST / "fullbag_terminal_v1.py").read_bytes())
    mock = work / "python-mock.sh"
    mock.write_text('#!/usr/bin/env bash\nset -eu\n'
        'if [[ "$2" == *fullbag_terminal_v1.py ]]; then exec "'+Path(sys.executable).as_posix()+'" "$@"; fi\n'
        'echo "FORBIDDEN_REMOTE_STEP $*" >> "'+(work / 'commands.txt').as_posix()+'"\n'
        'exit 99\n', encoding="utf-8", newline="\n")
    if os.name != "nt":
        mock.chmod(0o700)
    env = dict(os.environ, FULLBAG_PROJECT=project.as_posix(), FULLBAG_WAITER_HOME=work.as_posix(),
               FULLBAG_PYTHON=mock.as_posix())
    result = subprocess.run([str(bash), str(POST / "wait_fullbag_then_export_v1.sh")],
                            env=env, capture_output=True, text=True, timeout=20)
    log = (work / "wait_fullbag.log").read_text(encoding='utf-8')
    if fault is None:
        assert result.returncode == 0 and "END_OK" in log
    else:
        assert result.returncode != 0 and "END_FAIL" in log and "END_OK" not in log
    assert not (work / "commands.txt").exists(), "remote Python may only execute the terminal helper"
    assert before == {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()}


def test_remote_helper_python311_api_contract():
    source = (POST / 'fullbag_terminal_v1.py').read_text()
    tree = ast.parse(source, feature_version=(3, 11))
    imports = {n.module if isinstance(n, ast.ImportFrom) else n.names[0].name
               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert imports == {'hashlib', 'json', 'pathlib', 'sys', 'time'}
    # Every attribute reached by this helper is enumerated: no project imports,
    # dynamic dispatch or Path.is_junction from the Python-3.12-only packager.
    assert {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} <= {
        'loads', 'dumps', 'dump', 'sha256', 'hexdigest', 'encode', 'index', 'time',
        'argv', 'write', 'strip', 'st_mtime', 'read_text', 'resolve', 'exists',
        'is_file', 'stat', 'open'}
    evidence = (POST / 'remote-stdlib-readonly.log').read_text()
    assert '/opt/conda/bin/python3.11' in evidence and 'PASS_READ_ONLY_PY311_PATHLIB' in evidence
    for method in ('read_text', 'resolve', 'exists', 'is_file', 'stat', 'open'):
        assert 'def ' + method + '(' in evidence
    assert 'newline=None' in evidence


def test_packager_only_runs_locally_in_postfit_card():
    waiter = (POST / 'wait_fullbag_then_export_v1.sh').read_text()
    assert 'tools.k1_fold_dataset_v1' not in waiter
    assert 'tools/l1_rco_k1_fullbag_v1.py' not in waiter
    blocks = re.findall(r'```bash\n(.*?)```', (POST / 'RUNBOOK.md').read_text(), re.S)
    packaging = [b for b in blocks if '-m tools.k1_fold_dataset_v1' in b]
    assert len(packaging) == 1
    block = packaging[0]
    assert "py='C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'" in block
    assert 'ssh ' not in block and 'scp ' not in block
    steps = [line for line in block.splitlines() if '-m tools.k1_fold_dataset_v1' in line]
    assert len(steps) == 3
    for line, command in zip(steps, ('verify-export', 'package', 'verify-dataset')):
        assert line.startswith('"$py" -B -m tools.k1_fold_dataset_v1 ' + command + ' ')
    assert '"$remote:$root/$task/dataset' not in '\n'.join(blocks)
    admission = next(b for b in blocks if '-m tools.k1_admit_v1 prepare' in b)
    assert '--n32-dir' not in admission


def fullbag_assets(root):
    export = synthetic_export(root / "export", 1)
    base = export.parent
    info = assets.read(export)
    old, new = "fit/models/E_prior_s0_f1.pt", "fit/models/E_prior_s0_f0.pt"
    (base / new).write_bytes((base / old).read_bytes())
    info["sha256"][new] = info["sha256"].pop(old)
    reg = assets.read(base / "registration.json")
    reg.pop("contract_id")
    reg.update(model_fold=0, variant="fullbag", train=[f"t{i}" for i in range(3190)],
               holdout=[f"h{i}" for i in range(977)])
    reg["recipe"].update(model_fold=0, windows=640, encoder_microbatch=32)
    reg["contract_id"] = assets.digest(reg)
    write(base / "registration.json", reg)
    info.update(fold=0, contract_id=reg["contract_id"])
    for name in ("prepare.json", "fit/result.json", "fit/fit-complete.json"):
        data = assets.read(base / name)
        data["contract_id"] = reg["contract_id"]
        if name == "prepare.json":
            data["trainer_sha256"] = hashlib.sha256(assets.derive_training_source(assets.SERVING.read_bytes(), 0)).hexdigest()
        else:
            data.update(model_fold=0, holdout_rows=977, train_studies=3190)
        write(base / name, data)
    seal(base, info)
    destination = root / "dataset"
    assets.package(export, destination, "taeyangg4/knee-k1-v3gfb-v1-assets")
    return export, destination


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_template_fill_prepare_validate(tmp_path, monkeypatch, path):
    _, dataset = fullbag_assets(tmp_path)
    target = tmp_path / "filled.json"
    subprocess.run([sys.executable, "-B", str(FILL), str(path), str(target),
        "taeyangg4/cpu-fullbag-fixture-v1", "1.1.90-cpu-fullbag-fixture",
        "k1_v3g_fullbag=" + str(dataset)], cwd=ROOT, check=True, capture_output=True)
    config = assets.read(target)
    legacy_name = {"c6-fb-dropo-p25-v10.template.json": "c6-fb-dropo-p25-v9.template.json",
                   "c6-fb-dropo-p25-flatp-v10.template.json": "c6-fb-dropo-p25-flatp.template.json",
                   "b6-fb-flat-r6s2-v10.template.json": "b6-fb-flat-r6s2-v9.template.json"}.get(path.name, path.name)
    p25_family = path.name.startswith('p25-fb-')
    if p25_family:
        parent_path = ROOT / 'configs/k1n' / (
            'p25-dropo-flatp-v10.json' if 'flatp' in path.name else 'p25-dropo-v10.json')
    else:
        lineage = next(v for v in assets.read(POST / "LINEAGE.json")["lineages"]
                       if Path(v["template"]).name == legacy_name)
        parent_path = ROOT / lineage["parent"]
        if config.get("k1_runtime_version") == 10:
            parent_path = parent_path.with_name({"b5-flat-r6s2-v9.json": "b5-flat-r6s2-v10.json",
                "c5-dropo-p25-v9.json": "c5-dropo-p25-v10.json",
                "c5-dropo-p25-flatp.json": "c5-dropo-p25-flatp-v10.json"}[parent_path.name])
    parent = assets.read(parent_path)
    assert config["models"][0] == parent["models"][0]
    if p25_family:
        assert config['models'][:3] == parent['models']
        assert config['member_weights'] == [.25]*4
        assert config['models'][3]['id'] == 'k1_v3g_fullbag'
        assert all(m['checkpoint'].endswith('_f0.pt') for m in config['models'])
    else:
        assert config["models"][2:] == parent["models"][1:]
        assert config["member_weights"] == [.1, .1, .2, .2, .2, .2]
        assert config["models"][1]["id"] == "k1_v3g_fullbag"
    assert {k for k in config.keys() | parent.keys() if config.get(k) != parent.get(k)} <= {
        "models", "member_weights", "kernel_id", "release_version"}
    assert assets.verify_dataset(dataset)["trainer_sha256"] == parent["models"][0]["trainer_sha256"]
    # Execute exact prepare/validate code with destinations isolated in pytest scratch.
    monkeypatch.setattr(stage, "stage", lambda c, d=False: tmp_path / ("stage-n32" if d else "stage"))
    assert stage.run("prepare", config)["status"] == "PREPARED_LOCAL_ONLY"
    assert stage.run("validate", config)["status"] == "PASS_CPU_VALIDATE"
    for diagnostic in (False, True):
        reg = assets.read(stage.stage(config, diagnostic) / "registration.json")
        assert reg["never_submit"] is diagnostic
        assert reg["models"] == config["models"]
        assert reg.get("member_weights", [.25]*4 if p25_family else None) == config["member_weights"]
        assert 'projection_basis' not in reg and 'projection_basis' not in config


@pytest.mark.parametrize('path', TEMPLATES, ids=lambda p: p.name)
def test_review_rejects_added_projection_basis(tmp_path, path):
    config = assets.read(path)
    helper = 'review_candidate_v10.py' if config.get('k1_runtime_version') == 10 else 'review_candidate_v1.py'
    review = load('fullbag_review_test', POST / helper)
    config['projection_basis'] = 'n32_ratio'
    target = tmp_path / 'drift.json'
    write(target, config)
    with pytest.raises(ValueError, match='registered lineage/weights'):
        review.review(target, tmp_path / 'review')
    assert not (tmp_path / 'review').exists()


@pytest.mark.parametrize("fault", ["seed", "fold", "variant", "id"])
def test_fill_rejects_wrong_fullbag_identity(tmp_path, monkeypatch, fault):
    fill = load("fill_fullbag_test", FILL)
    dataset = tmp_path / "dataset"
    manifest = dict(model_seed=0, model_fold=0, variant="fullbag")
    manifest[{"seed": "model_seed", "fold": "model_fold", "variant": "variant", "id": "variant"}[fault]] = (
        1 if fault in ("seed", "fold") else "fullbag" if fault == "id" else "seed0")
    write(dataset / "asset-manifest.json", manifest)
    monkeypatch.setattr(fill, "verify", lambda _: dict(model=dict(id="wrong" if fault == "id" else "k1_v3g_fullbag")))
    monkeypatch.setattr(sys, "argv", ["fill", str(TEMPLATES[0]), str(tmp_path / "out.json"),
        "owner/test", "1.1.90-test", "k1_v3g_fullbag="+str(dataset)])
    with pytest.raises(SystemExit):
        fill.main()


@pytest.mark.parametrize("wins,expected", [(0,"HOLD"), (2,"HOLD"), (3,"SCREEN_PASS_PENDING_REVIEW"), (4,"SCREEN_PASS_PENDING_REVIEW")])
def test_exact_fullbag_gate_rule(wins, expected):
    baseline = dict.fromkeys(gate.RULERS, .8)
    candidate = {r: .8 if i < wins else .799999 for i,r in enumerate(gate.RULERS)}
    assert fullbag.decide(baseline, candidate)["status"] == expected


def real_roster_fixture(base):
    """Use real baseline predictions/rulers, synthetic fullbag receipt (CPU fixture only)."""
    expected = fullbag.expected_registration()
    reg = copy.deepcopy(expected)
    reg["collapse_guard"]["constant_loss"] += 4e-15
    reg["collapse_guard"]["threshold"] += 4e-15
    reg.pop("contract_id")
    reg["contract_id"] = gate.k.digest(reg)
    (base / "fit").mkdir(parents=True, exist_ok=True)
    baseline = fullbag.k.parent.OUT
    (base / "fit/holdout.csv").write_bytes((baseline / "holdout.csv").read_bytes())
    result = assets.read(baseline / "result.json")
    result.update(contract_id=reg["contract_id"], holdout_sha256=assets.sha(base / "fit/holdout.csv"))
    argv = fullbag.k.trainer_argv(fullbag.k.CACHE, expected)
    for i in (0,1,*(argv.index(flag)+1 for flag in gate.HOST_PATH_FLAGS)):
        argv[i] = "/synthetic-vast/" + str(i)
    argv[argv.index("--l1-guard-constant")+1] = repr(reg["collapse_guard"]["constant_loss"])
    prep = dict(gate.prepared_hashes(expected, fullbag.k), contract_id=reg["contract_id"],
        train_count=3190, holdout_training_rows=0, argv=argv, cache=argv[argv.index("--px")+1])
    for name, value in (("registration.json", reg), ("prepare.json", prep), ("fit/result.json", result)):
        write(base / name, value)
    info = dict(fold=0, contract_id=reg["contract_id"], sha256={name:assets.sha(base / name) for name in (
        "registration.json", "prepare.json", "fit/result.json", "fit/holdout.csv")})
    write(base / "EXPORT.json", info)
    return dict(schema="k1-fullbag-gate-input/v1", models={
        "k1_v3g_seed0": dict(k1_csv=str(baseline / "holdout.csv"), k1_result=str(baseline / "result.json")),
        "k1_v3g_fullbag": dict(k1_csv=str(base / "fit/holdout.csv"), k1_result=str(base / "fit/result.json"))})


def test_real_roster_host_bound_gate(tmp_path):
    manifest = real_roster_fixture(tmp_path / "gate")
    result = gate.evaluate_fullbag(manifest)
    assert result["holdout_rows"] == 977 and result["status"] == "SCREEN_PASS_PENDING_REVIEW"
    assert result["metrics"]["k1_v3g_seed0"] == result["metrics"]["k1_v3g_fullbag"]
    assert result["evidence"]["k1_v3g_fullbag"]["binding"]["mode"] == "HOST_BOUND_EXPORT"
    path = tmp_path / "gate/prepare.json"
    prep = assets.read(path)
    prep["argv"][prep["argv"].index("--train-windows")+1] = "64"
    write(path, prep)
    export = assets.read(tmp_path / "gate/EXPORT.json")
    export["sha256"]["prepare.json"] = assets.sha(path)
    write(tmp_path / "gate/EXPORT.json", export)
    with pytest.raises(ValueError, match="recipe"):
        gate.evaluate_fullbag(manifest)
