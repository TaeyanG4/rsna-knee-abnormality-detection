"""CPU-only gate tests, including immutable real release inputs; no inference."""
import base64
import io
import json
from pathlib import Path
import re
import zipfile

import pytest

from tools import strict_variant_gate as g

REAL = g.ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7"


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else g.encoded(data))


def notebook(files, extra=""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    blob = buf.getvalue()
    code = ("import base64, hashlib\nBLOB = " + repr(base64.b64encode(blob).decode()) + "\nPINS = " + repr({k: g.sha(v) for k, v in files.items()}) + "\nblob = base64.b64decode(BLOB)\nassert hashlib.sha256(blob).hexdigest() == " + repr(g.sha(blob)) + "\n" + extra)
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": code.splitlines(True)}]}


def seal(stage):
    names = ["kernel/kernel-metadata.json", "kernel/release.ipynb", "registration.json"]
    lines = [f"{g.sha((stage / name).read_bytes())}  {name}\n" for name in names]
    put(stage / "SHA256SUMS", "".join(lines).encode())
    return g.sha((stage / "SHA256SUMS").read_bytes())


def rule(a, b):
    # Test fixture only: NOT a production allowlist generator.
    import difflib
    removed, added = [], []
    old, new = a.decode().splitlines(), b.decode().splitlines()
    for op, i, j, k, l in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if op != "equal":
            removed.extend(re.escape(x) for x in old[i:j])
            added.extend(re.escape(x) for x in new[k:l])
    return {"before_sha256": g.sha(a), "after_sha256": g.sha(b), "removed": removed, "added": added}


def dispatched(root, lane, path, data):
    put(path, data)
    record = dict(schema="dispatch-terminal/v1", lane=lane, run_id=g.sha((lane + str(path)).encode())[:32],
                  out=str(path), state="SUCCESS", exit_code=0, report_sha256=g.sha(data), prompt_sha256=g.sha(b"test prompt"))
    archive = root / ".local/astra-improve/lanes/runs" / lane / record["run_id"]
    put(archive / "report.md", data)
    put(archive / "prompt.txt", b"test prompt")
    put(archive / "receipt.json", record)
    receipt = Path(str(path) + ".receipt.json")
    put(receipt, record)
    return receipt


def endorse(config):
    root = config["stage"].parent
    allow = g.parse(config["allow_path"].read_bytes())
    folder = config["stage"] / "kernel"
    files = {p.relative_to(folder).as_posix(): g.sha(p.read_bytes()) for p in sorted(folder.rglob("*")) if p.is_file()}
    receipt = dispatched(root, "strict-gate-builder", root / "build.md", ("Kernel-SHA256: " + g.sha(g.encoded(files)) + "\n").encode())
    allow["review_binding"]["builder_receipt_sha256"] = g.sha(receipt.read_bytes())
    put(config["allow_path"], allow)
    dispatched(root, "strict-gate-reviewer", config["review"], (
        "Reviewer-Lane: strict-gate-reviewer\nSHA256SUMS: " + allow["stage_sha256sums"] +
        "\nAllowlist-SHA256: " + g.sha(config["allow_path"].read_bytes()) + "\nVERDICT: PASS\n").encode())


@pytest.fixture
def clean(tmp_path, monkeypatch):
    original_dispatch = g.dispatch
    def isolated_dispatch(path, inputs):
        isolated = g.Inputs(tmp_path)
        result = original_dispatch(g.local_path(path, inputs.root), isolated)
        inputs.data.update(isolated.data)
        return result
    monkeypatch.setattr(g, "dispatch", isolated_dispatch)
    parent, stage = tmp_path / "parent", tmp_path / "stage"
    files = {"runtime.py": b"WEIGHT = 0.85\n", "nested/helper.py": b"def run():\n    return 1\n"}
    meta = {"id": "test/strict-gate-synthetic-v1", "title": "strict-gate-synthetic-v1", "code_file": "release.ipynb", "docker_image": "image@sha256:fixed", "dataset_sources": ["test/data"], "machine_shape": "NvidiaTeslaT4", "enable_internet": False}
    for directory in (parent, stage):
        put(directory / "kernel/release.ipynb", notebook(files))
        put(directory / "kernel/kernel-metadata.json", meta)
        put(directory / "registration.json", {"kernel_id": meta["id"]})
        seal(directory)
    allow_path, review = tmp_path / "allow.json", tmp_path / "review.md"
    allow = {"version": g.VERSION, "review_binding": {"source_stage": str(stage), "review": str(review), "builder_receipt": str(tmp_path / "build.md.receipt.json"), "builder_receipt_sha256": ""}, "parent_sha256sums": g.sha((parent / "SHA256SUMS").read_bytes()), "stage_sha256sums": g.sha((stage / "SHA256SUMS").read_bytes()), "files": {}}
    put(allow_path, allow)
    config = dict(parent=parent, stage=stage, allow_path=allow_path, review=review)
    endorse(config)
    return config


def evaluate(config):
    return g.evaluate(**config)


def refresh(config, authorize=False):
    allow = g.parse(config["allow_path"].read_bytes())
    allow["stage_sha256sums"] = seal(config["stage"])
    if authorize:
        inputs = g.Inputs(g.ROOT)
        a, b = [g.kernel(config[k], inputs)[1] for k in ("parent", "stage")]
        allow["files"] = {name: rule(a[name], b[name]) for name in a.keys() & b.keys() if a[name] != b[name]}
    put(config["allow_path"], allow)
    endorse(config)


def errors(config):
    result = evaluate(config)
    assert result["status"] == "FAIL"
    return "\n".join(result["errors"])


def test_clean_stage_pass_and_all_hashes(clean):
    r = evaluate(clean)
    assert r["status"] == "PASS", r["errors"]
    assert len(r["stage_inventory"]["embedded"]) == 2
    assert len(r["input_sha256"]) == 20  # kernel/manifest/policy + both archived dispatches
    assert r["diff_sha256"] == g.sha(b"")


def test_explicit_arithmetic_allowlist_pass(clean):
    put(clean["stage"] / "kernel/release.ipynb", notebook({"runtime.py": b"WEIGHT = 0.75\n", "nested/helper.py": b"def run():\n    return 1\n"}))
    refresh(clean, authorize=True)
    r = evaluate(clean)
    assert r["status"] == "PASS", r["errors"]
    assert "WEIGHT = 0.75" in r["full_diff"] and "raw-stage" in r["full_diff"]


def test_renamed_runtime_cannot_be_allowed(clean):
    put(clean["stage"] / "kernel/release.ipynb", notebook({"renamed.py": b"WEIGHT = 0.85\n", "nested/helper.py": b"def run():\n    return 1\n"}))
    refresh(clean, authorize=True)
    assert "renamed" in errors(clean)


@pytest.mark.parametrize("version", ["v1.1.9-fluidR", "v1.1.10-own25"])
def test_real_prefixed_title_fails(clean, version):
    clean.update(stage=REAL / version, parent=REAL / "v1.1.0")
    assert "title must equal id slug" in errors(clean)


def test_real_own25q_has_no_independent_bound_review(clean):
    clean.update(stage=REAL / "v1.1.11-own25q", parent=REAL / "v1.1.0", review=g.ROOT / ".local/scratch/own25q-review/review/REPORT.md")
    assert any(word in errors(clean) for word in ("review", "Reviewer-Lane", "bound source"))


@pytest.mark.parametrize("field,value", [("docker_image", "other"), ("dataset_sources", ["other/data"]), ("machine_shape", "other"), ("enable_internet", True)])
def test_protected_metadata_even_with_allowlist(clean, field, value):
    path = clean["stage"] / "kernel/kernel-metadata.json"
    meta = g.parse(path.read_bytes())
    meta[field] = value
    put(path, meta)
    refresh(clean, authorize=True)
    assert "protected metadata differs: " + field in errors(clean)


def test_registration_identity(clean):
    put(clean["stage"] / "registration.json", {"kernel_id": "wrong/id"})
    refresh(clean)
    assert "registration kernel_id" in errors(clean)


@pytest.mark.parametrize("change", ["same_lane", "missing_hash", "stale_hash", "trailing", "earlier_fail", "no_lane"])
def test_review_fail_closed(clean, change):
    data = clean["review"].read_text()
    data = {"same_lane": data.replace("strict-gate-reviewer", "strict-gate-builder"), "missing_hash": "Reviewer-Lane: strict-gate-reviewer\nVERDICT: PASS\n", "stale_hash": re.sub(g.HEX, "0" * 64, data), "trailing": data + "More text\n", "earlier_fail": "VERDICT: FAIL\n" + data, "no_lane": "\n".join(data.splitlines()[1:])}[change]
    put(clean["review"], data.encode())
    errors(clean)


def test_builder_scratch_review_rejected_without_read(clean):
    clean["review"] = g.ROOT / ".local/scratch/strict-gate-builder/review.md"
    allow = g.parse(clean["allow_path"].read_bytes())
    allow["review_binding"]["review"] = str(clean["review"])
    put(clean["allow_path"], allow)
    assert "under builder scratch" in errors(clean)


def test_builder_identity_cannot_be_declared(clean):
    allow = g.parse(clean["allow_path"].read_bytes())
    allow["builder_lane"] = "someone-else"
    put(clean["allow_path"], allow)
    with pytest.raises(g.GateError, match="schema"):
        evaluate(clean)


def test_review_crlf_pass(clean):
    data = clean["review"].read_bytes().replace(b"\n", b"\r\n")
    dispatched(clean["stage"].parent, "strict-gate-reviewer", clean["review"], data)
    assert evaluate(clean)["status"] == "PASS"


def test_protected_bool_type_is_not_integer(clean):
    path = clean["stage"] / "kernel/kernel-metadata.json"
    meta = g.parse(path.read_bytes())
    meta["enable_internet"] = 0
    put(path, meta)
    refresh(clean, authorize=True)
    assert "protected metadata differs: enable_internet" in errors(clean)


def test_relabel_path_blocked_before_any_read():
    with pytest.raises(g.GateError, match="forbidden input"):
        g.Inputs(g.ROOT).read(g.ROOT / ".local/relabel/never-open.txt")


def test_manifest_stale_and_missing_coverage(clean):
    path = clean["stage"] / "kernel/release.ipynb"
    path.write_bytes(path.read_bytes() + b" ")
    assert "SHA256SUMS mismatch" in errors(clean)
    seal(clean["stage"])
    path = clean["stage"] / "SHA256SUMS"
    path.write_text("\n".join(x for x in path.read_text().splitlines() if "registration.json" not in x))
    assert "does not cover registration" in errors(clean)


def test_manifest_traversal(clean):
    put(clean["stage"] / "SHA256SUMS", ("0" * 64 + "  ../escape\n").encode())
    assert "unsafe relative path" in errors(clean)


@pytest.mark.parametrize("change", ["cell", "top", "outputs", "runtime", "extra_file"])
def test_unallowed_surfaces(clean, change):
    path = clean["stage"] / "kernel/release.ipynb"
    nb = g.parse(path.read_bytes())
    if change == "cell":
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["surprise"]})
    elif change == "top":
        nb["metadata"]["changed"] = 1
    elif change == "outputs":
        nb["cells"][0]["outputs"] = [{"output_type": "stream", "text": "extra"}]
    elif change == "runtime":
        nb = notebook({"runtime.py": b"WEIGHT = 0.86\n", "nested/helper.py": b"def run():\n    return 1\n"})
    else:
        put(clean["stage"] / "kernel/extra.py", b"print('extra')\n")
    put(path, nb)
    refresh(clean)
    errors(clean)


def test_line_pattern_and_hash_both_enforced(clean):
    put(clean["stage"] / "kernel/release.ipynb", notebook({"runtime.py": b"WEIGHT = 0.75\n", "nested/helper.py": b"def run():\n    return 1\n"}))
    refresh(clean, authorize=True)
    allow = g.parse(clean["allow_path"].read_bytes())
    allow["files"]["embedded/runtime.py"]["added"] = ["WEIGHT"]
    put(clean["allow_path"], allow)
    assert "unallowed added line" in errors(clean)
    allow["files"]["embedded/runtime.py"]["after_sha256"] = "0" * 64
    put(clean["allow_path"], allow)
    assert "allowlist hash mismatch" in errors(clean)


@pytest.mark.parametrize("kind", ["duplicate_binding", "unsafe_member", "pin_mismatch", "case_collision"])
def test_archive_rejects(clean, kind):
    files = {"runtime.py": b"x = 1\n"}
    if kind == "unsafe_member":
        files["../escape.py"] = b"x"
    if kind == "case_collision":
        files["Runtime.py"] = b"x"
    nb = notebook(files, "BLOB = 'again'\n" if kind == "duplicate_binding" else "")
    if kind == "pin_mismatch":
        nb["cells"][0]["source"] = [re.sub(g.sha(files["runtime.py"]), "0" * 64, line) for line in nb["cells"][0]["source"]]
    put(clean["stage"] / "kernel/release.ipynb", nb)
    refresh(clean)
    errors(clean)


def test_snapshot_change_detected(clean):
    inputs = g.Inputs(g.ROOT)
    inputs.read(clean["review"])
    put(clean["review"], b"different")
    with pytest.raises(g.GateError, match="changed during"):
        inputs.recheck()


def test_receipt_immutable_and_retry(clean, monkeypatch):
    result = evaluate(clean)
    receipt = clean["stage"].parent / "receipt.json"
    original_link = g.os.link
    attempts = []
    def transient(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("locked")
        return original_link(*args)
    monkeypatch.setattr(g.os, "link", transient)
    monkeypatch.setattr(g.time, "sleep", lambda _: None)
    g.write_receipt(receipt, result)
    g.write_receipt(receipt, result)
    assert len(attempts) == 2
    with pytest.raises(g.GateError, match="immutable"):
        g.write_receipt(receipt, {"different": True})
    assert g.parse(receipt.read_bytes())["status"] == "PASS"
    assert not receipt.with_suffix(".json.tmp").exists()


def test_cli_pass_failure_and_receipt(clean, monkeypatch, capsys):
    # Exercise actual CLI parser and writer without occupying a production slug.
    root = clean["stage"].parent
    monkeypatch.setattr(g, "ROOT", root)
    argv = sum((["--" + key, str(clean[value])] for key, value in (("stage", "stage"), ("parent", "parent"), ("allow", "allow_path"), ("review", "review"))), [])
    assert g.main(argv + ["--check-only"]) == 0
    assert not (root / ".local/state/gates").exists()
    assert g.main(argv) == 0
    assert (root / ".local/state/gates/strict-gate-synthetic-v1.json").is_file()
    put(clean["review"], b"VERDICT: FAIL\n")
    assert g.main(argv) == 1
    assert '"status": "FAIL"' in capsys.readouterr().out


def historical_config(tmp_path, version="v1.1.18-own25q3"):
    short = version.split("-", 1)[1]
    output = tmp_path / (version + "-gate-v110")
    review = g.ROOT / (g.PRESUB + short + "-review-out.md")
    g.bind(REAL / version, REAL / "v1.1.0", output,
           g.ROOT / (g.PRESUB + short + "-build-out.md.receipt.json"), review, True)
    return dict(stage=output, parent=REAL / "v1.1.0", allow_path=output / "allowlist.json", review=review)


@pytest.mark.parametrize("version", list(g.HISTORICAL))
def test_real_fast_stages_pass_with_original_reviews(tmp_path, version):
    config = historical_config(tmp_path, version)
    result = evaluate(config)
    assert result["status"] == "PASS", result["errors"]
    assert len(result["stage_inventory"]["embedded"]) == 29
    assert result["reviewer_lane"] == "slot-b"
    allow = g.parse(config["allow_path"].read_bytes())
    assert allow["fast_path"] == g.FAST_REFERENCE
    assert g.sha(config["review"].read_bytes()) == g.HISTORICAL[version]["review_sha256"]
    assert "raw-stage/release.ipynb" in result["full_diff"]
    assert "notebook/cells/2/source.txt" in result["comparison_sha256"]["stage"]
    assert g.parse((config["stage"] / "registration.json").read_bytes())["version"] == version.removeprefix("v")


@pytest.mark.parametrize("change,expected", [
    ("cell0", "cell 0 hash"), ("cell0_fields", "cell 0 hash"),
    ("guard", "guard must"), ("extra_run", "run cell has changes"),
    ("fields", "fields/top changed"), ("top", "fields/top changed"),
    ("markdown", "markdown changed"), ("extra_cell", "exactly parent two"),
    ("title", "title must"), ("rename", "renamed"),
])
def test_fast_negative_probes_even_with_refreshed_manifest(tmp_path, change, expected):
    config = historical_config(tmp_path)
    path = config["stage"] / "kernel/release.ipynb"
    nb = g.parse(path.read_bytes())
    if change == "cell0":
        nb["cells"][0]["source"].append("FAST = True\n")
    elif change == "cell0_fields":
        nb["cells"][0]["metadata"]["extra"] = True
    elif change == "guard":
        nb["cells"][2]["source"][0] = "if FAST:\n"
    elif change == "extra_run":
        nb["cells"][2]["source"].append("    print('changed bootstrap')\n")
    elif change == "fields":
        nb["cells"][2]["outputs"] = [{"output_type": "stream", "text": "unexpected"}]
    elif change == "top":
        nb["metadata"]["extra"] = True
    elif change == "markdown":
        nb["cells"][1]["source"].append("changed\n")
    elif change == "extra_cell":
        nb["cells"].append(dict(nb["cells"][1]))
    elif change == "rename":
        import ast
        source = "".join(nb["cells"][2]["source"])
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("BLOB", "PINS")}
        original = base64.b64decode(values["BLOB"])
        stream = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original)) as archive, zipfile.ZipFile(stream, "w") as out:
            for info in archive.infolist():
                name = "renamed.py" if info.filename == "exp169_release_c85.py" else info.filename
                out.writestr(name, archive.read(info))
        pins = dict(values["PINS"])
        pins["renamed.py"] = pins.pop("exp169_release_c85.py")
        source = source.replace(repr(values["BLOB"]), repr(base64.b64encode(stream.getvalue()).decode())).replace(repr(values["PINS"]), repr(pins)).replace(g.sha(original), g.sha(stream.getvalue()))
        nb["cells"][2]["source"] = source.splitlines(True)
    elif change == "title":
        meta_path = config["stage"] / "kernel/kernel-metadata.json"
        meta = g.parse(meta_path.read_bytes())
        meta["title"] = "taeyangg4-" + meta["title"]
        put(meta_path, meta)
    put(path, nb)
    allow = g.parse(config["allow_path"].read_bytes())
    allow["stage_sha256sums"] = seal(config["stage"])
    put(config["allow_path"], allow)
    assert expected in errors(config)


def test_generic_bind_requires_independent_hashed_review(clean):
    root = clean["stage"].parent
    output = root / "new-stage-v1"
    result = g.bind(clean["stage"], clean["parent"], output, root / "build.md.receipt.json", clean["review"])
    assert result["status"] == "REVIEW_REQUIRED"
    config = dict(stage=output, parent=clean["parent"], allow_path=output / "allowlist.json", review=clean["review"])
    assert "review does not name current" in errors(config)
    allow = g.parse(config["allow_path"].read_bytes())
    dispatched(root, "strict-gate-reviewer", config["review"], (
        "Reviewer-Lane: strict-gate-reviewer\nSHA256SUMS: " + allow["stage_sha256sums"] +
        "\nAllowlist-SHA256: " + g.sha(config["allow_path"].read_bytes()) + "\nVERDICT: PASS\n").encode())
    assert evaluate(config)["status"] == "PASS"
    with pytest.raises(g.GateError, match="already exists"):
        g.bind(clean["stage"], clean["parent"], output, root / "build.md.receipt.json", clean["review"])


def test_cannot_relabel_builder_dispatch(clean):
    root = clean["stage"].parent
    path = root / "build.md.receipt.json"
    record = g.parse(path.read_bytes())
    record["lane"] = "strict-gate-reviewer"
    put(path, record)
    allow = g.parse(clean["allow_path"].read_bytes())
    allow["review_binding"]["builder_receipt_sha256"] = g.sha(path.read_bytes())
    put(clean["allow_path"], allow)
    assert "receipt.json" in errors(clean)


def test_cannot_use_same_dispatched_lane(clean):
    root = clean["stage"].parent
    dispatched(root, "strict-gate-builder", clean["review"], clean["review"].read_bytes())
    assert "different-lane dispatched review required" in errors(clean)


def test_unbound_legacy_schema_rejected(clean):
    allow = g.parse(clean["allow_path"].read_bytes())
    allow.pop("review_binding")
    allow.update(builder_lane="claimed", builder_scratch=".local/scratch/claimed")
    put(clean["allow_path"], allow)
    with pytest.raises(g.GateError, match="schema"):
        evaluate(clean)


def test_fast_reference_cannot_be_substituted(tmp_path):
    config = historical_config(tmp_path)
    allow = g.parse(config["allow_path"].read_bytes())
    allow["fast_path"]["cell0_sha256"] = "0" * 64
    put(config["allow_path"], allow)
    assert "pinned reviewed" in errors(config)


def test_historical_review_cannot_be_replaced(tmp_path):
    config = historical_config(tmp_path)
    fake = tmp_path / "fake.md"
    put(fake, config["review"].read_bytes())
    allow = g.parse(config["allow_path"].read_bytes())
    allow["review_binding"]["review"] = str(fake)
    put(config["allow_path"], allow)
    config["review"] = fake
    assert "historical review hash/path mismatch" in errors(config)


def test_allowlist_change_needs_new_review_even_when_rules_still_pass(clean):
    allow = g.parse(clean["allow_path"].read_bytes())
    inputs = g.Inputs(g.ROOT)
    files = g.kernel(clean["stage"], inputs)[1]
    name = "embedded/runtime.py"
    allow["files"][name] = rule(files[name], files[name])
    put(clean["allow_path"], allow)
    assert "review does not name current allowlist hash" in errors(clean)


@pytest.mark.parametrize("part", ["prompt.txt", "report.md", "receipt.json"])
def test_dispatch_archive_tamper_fails(clean, part):
    root = clean["stage"].parent
    receipt = g.parse((root / "build.md.receipt.json").read_bytes())
    archive = root / ".local/astra-improve/lanes/runs" / receipt["lane"] / receipt["run_id"]
    put(archive / part, b"tampered")
    errors(clean)


def test_bind_cli_real_stage(tmp_path, capsys):
    output = tmp_path / "cli-bind-v110"
    review = g.ROOT / (g.PRESUB + "own25q3-review-out.md")
    args = ["bind", "--stage", str(REAL / "v1.1.18-own25q3"),
            "--parent", str(REAL / "v1.1.0"), "--output", str(output),
            "--builder-receipt", str(g.ROOT / (g.PRESUB + "own25q3-build-out.md.receipt.json")),
            "--review", str(review), "--fast"]
    assert g.main(args) == 0
    assert '"status": "BOUND"' in capsys.readouterr().out
    assert g.main(["--stage", str(output), "--parent", str(REAL / "v1.1.0"),
                   "--allow", str(output / "allowlist.json"), "--review", str(review), "--check-only"]) == 0
    assert g.main(args) == 1  # create-only, including when invoked through CLI
