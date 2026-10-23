"""Offline refusal coverage and real check-only integration for K1 admission."""
import json
import socket
import subprocess

import pytest

from tools import k1_admit_v1 as m


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.encoded(value))


def change(path, fn):
    value = json.loads(path.read_bytes())
    fn(value)
    put(path, value)


def transport(d, kernel="owner/k1", version=1):
    for name in ("k1sr-receipt.json", "c85r7-receipt.json", "submission.csv"):
        put(d / (name + ".download.json"), dict(
            status="DOWNLOADED", kernel=kernel, version=version, file=name,
            local_name=name, sha256=m.sha((d / name).read_bytes())))


@pytest.fixture
def case(tmp_path, monkeypatch):
    # pytest uses the required project-local --basetemp; no credentials/data.
    stage = tmp_path / "v1.2.3-k1"
    d = tmp_path / "visible"
    d.mkdir()
    notebook = b'{"cells":[]}'
    meta = {"id": "owner/k1", "code_file": "release.ipynb", "title": "k1"}
    put(stage / "kernel/kernel-metadata.json", meta)
    (stage / "kernel/release.ipynb").write_bytes(notebook)
    identity = dict(kernel_id="owner/k1", release_version="1.2.3-k1",
                    formula="rank(.8*rank(parent)+.2*rank(K1))",
                    contract={"precision": "float16", "models": 1},
                    model_sha256="a" * 64, parent_weight=.8, k1_weight=.2,
                    r224_weight=.35, diagnostic=False, never_submit=False)
    reg = dict(identity, stage=str(stage), kernel_sha256=m.sha(notebook),
               metadata_sha256=m.sha((stage / "kernel/kernel-metadata.json").read_bytes()))
    put(stage / "registration.json", reg)
    csv = ("id," + ",".join("target" + str(i) for i in range(12)) + "\n"
           + "".join(uid + "," + ",".join(["0.5"] * 12) + "\n" for uid in ("a", "b", "c"))).encode()
    (d / "submission.csv").write_bytes(csv)
    (d / "k1sr-parent.csv").write_bytes(csv)
    template = tmp_path / "sample_submission.csv"
    template.write_bytes(csv)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(b"")
    monkeypatch.setattr(m, "TEMPLATE", template)
    monkeypatch.setattr(m, "LEDGER", ledger)
    parent = dict(status="COMPLETE_C85R7", version="1.2.3-k1", studies=3, findings=12,
                  fallbacks=dict(a172_prediction=0, r_prediction=0, s_r7_prediction=0,
                                 r224_prediction=0),
                  codecs=dict(status="ACTIVE", s_r7_gate="ACTIVE"), submission_sha256=m.sha(csv))
    k = dict(identity, status="COMPLETE_SHARED_K1", k1_executed=True,
             k1_studies=3, submission_sha256=m.sha(csv), parent_sha256=m.sha(csv))
    put(d / "c85r7-receipt.json", parent)
    put(d / "k1sr-receipt.json", k)
    transport(d)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **kw: pytest.fail("network forbidden"))
    return dict(stage=stage, kernel="owner/k1", version=1, visible_dir=d)


def n32(case):
    d = case["visible_dir"].parent / "n32"
    k = json.loads((case["visible_dir"] / "k1sr-receipt.json").read_bytes())
    k.update(kernel_id="owner/k1-n32", release_version="1.2.3-k1-n32", diagnostic=True,
             never_submit=True, k1_studies=32, parent_seconds=1000,
             phases=dict(classifier_seconds=100, extra_render_seconds=2, load_seconds=60),
             per_study=[dict(uid=str(i), ok=True, classifier_seconds=100 / 32,
                             extra_render_seconds=2 / 32, load_seconds=60 / 32)
                        for i in range(32)])
    put(d / "k1sr-receipt.json", k)
    receipt_hash = m.sha((d / "k1sr-receipt.json").read_bytes())
    evidence = dict(
        startup=dict(parent_startup_seconds=300, receipt_sha256=receipt_hash,
                     method="synthetic timed startup measurement"),
        selection=dict(n=32, counts={str(i): 1000 for i in range(32)}),
        train=dict(studies=1000, dicom_files=500000, method="synthetic metadata aggregate"))
    refs = {}
    for name, value in evidence.items():
        put(d / (name + ".json"), value)
        refs[name] = dict(path=name + ".json", sha256=m.sha((d / (name + ".json")).read_bytes()))
    put(d / "projection-inputs.json", dict(
        schema="k1-admit-projection/v1", receipt_sha256=receipt_hash,
        compatibility_reason="same synthetic model and recipe", **refs))
    case["n32_dir"] = d
    return d


def test_prepare_runs_only_real_offline_wrapper(case, monkeypatch, capsys):
    run = subprocess.run
    calls = []

    def guarded(args, **kw):
        assert "--check-only" in args
        assert str(m.ROOT / "tools/kaggle_submit_once.py") in args
        assert kw.get("shell", False) is False
        calls.append(args)
        return run(args, **kw)

    monkeypatch.setattr(m.subprocess, "run", guarded)
    out = m.prepare(**case)
    assert len(calls) == 1
    assert (out / "READY-REVIEW.md").read_text().endswith("VERDICT: PASS\n")
    assert "independent reviewer verdict" in (out / "READY-REVIEW.md").read_text()
    assert json.loads((out / "ordinary-visible-registration.json").read_bytes())["formula"].startswith("rank(")
    assert "ADMITTED_OFFLINE" in capsys.readouterr().out
    # Deterministic rerun is allowed; original input files stay intact.
    before = {p.name: p.read_bytes() for p in case["visible_dir"].iterdir() if p.is_file()}
    m.prepare(**case)
    assert before == {p.name: p.read_bytes() for p in case["visible_dir"].iterdir() if p.is_file()}


def k1n(case, drop_o=True):
    """v7 K1N contract, including the hash-bound on-disk config."""
    stage, d = case["stage"], case["visible_dir"]
    put(stage / "config.json", {"schema": "k1n-config/v1", "drop_o": drop_o})
    digest = m.sha((stage / "config.json").read_bytes())
    for path in (stage / "registration.json", d / "k1sr-receipt.json"):
        change(path, lambda r: r.update(drop_o=drop_o, config_sha256=digest))
    change(d / "k1sr-receipt.json", lambda r: r.update(status="COMPLETE_SHARED_K1N"))
    def parent(r):
        r.update(drop_o=drop_o, intended_members=["R", "S_R7", "R224"],
                 executed_members=["R", "S_R7", "R224"],
                 skipped_members=["EXP143v2", "EXP155", "Q_A5q"] if drop_o else [])
        if drop_o:
            del r["fallbacks"]["a172_prediction"]
        else:
            for key in ("intended_members", "executed_members"):
                r[key] += ["EXP143v2", "EXP155", "Q_A5q"]
    change(d / "c85r7-receipt.json", parent)
    transport(d)


def test_v7_complete_shared_k1n_drop_o_real_check_only(case, capsys):
    k1n(case)
    out = m.prepare(**case)
    assert "ADMITTED_OFFLINE" in capsys.readouterr().out
    assert (out / "READY-REVIEW.md").read_text().endswith("VERDICT: PASS\n")
    registration = json.loads((out / "ordinary-visible-registration.json").read_bytes())
    assert str(case["stage"] / "config.json") in registration["input_sha256"]
    parent = json.loads((case["visible_dir"] / "c85r7-receipt.json").read_bytes())
    assert "a172_prediction" not in parent["fallbacks"]


def test_v7_drop_o_false_parent_flag_absent_real_check_only(case, capsys):
    k1n(case, drop_o=False)
    change(case["visible_dir"] / "c85r7-receipt.json", lambda r: r.pop("drop_o"))
    transport(case["visible_dir"])
    out = m.prepare(**case)
    assert "ADMITTED_OFFLINE" in capsys.readouterr().out
    assert (out / "READY-REVIEW.md").read_text().endswith("VERDICT: PASS\n")


@pytest.mark.parametrize("value", [True, None, 0, 1, "false"])
def test_drop_o_false_parent_flag_must_match_and_be_boolean(case, value):
    k1n(case, drop_o=False)
    change(case["visible_dir"] / "c85r7-receipt.json", lambda r: r.update(drop_o=value))
    with pytest.raises(m.Refuse, match="drop_o"):
        m.prepare(**case)
    assert not (case["visible_dir"] / m.OUTPUT).exists()


@pytest.mark.parametrize("member", ["EXP143v2", "EXP155", "Q_A5q"])
@pytest.mark.parametrize("parent_flag_absent", [False, True])
def test_drop_o_false_requires_each_executed_o_member(case, member, parent_flag_absent):
    k1n(case, drop_o=False)
    def mutate(r):
        if parent_flag_absent:
            del r["drop_o"]
        r["executed_members"].remove(member)
    change(case["visible_dir"] / "c85r7-receipt.json", mutate)
    with pytest.raises(m.Refuse, match="requires executed O members"):
        m.prepare(**case)
    assert not (case["visible_dir"] / m.OUTPUT).exists()


def test_drop_o_false_parent_flag_absent_still_requires_o_counter(case):
    k1n(case, drop_o=False)
    def mutate(r):
        del r["drop_o"]
        del r["fallbacks"]["a172_prediction"]
    change(case["visible_dir"] / "c85r7-receipt.json", mutate)
    with pytest.raises(m.Refuse, match="missing parent prediction fallback"):
        m.validate(**case)


@pytest.mark.parametrize("record", ["registration", "config", "k1sr-receipt", "c85r7-receipt"])
@pytest.mark.parametrize("value", [False, None, 0, 1, "true", "false", "MISSING"])
def test_drop_o_flags_must_be_explicit_consistent_booleans(case, record, value):
    k1n(case)
    base = case["stage"] if record in {"registration", "config"} else case["visible_dir"]
    def mutate(r):
        if value == "MISSING":
            del r["drop_o"]
        else:
            r["drop_o"] = value
    change(base / (record + ".json"), mutate)
    with pytest.raises(m.Refuse, match="drop_o"):
        m.prepare(**case)
    assert not (case["visible_dir"] / m.OUTPUT).exists()


@pytest.mark.parametrize("record", ["registration", "k1sr-receipt"])
def test_drop_o_config_hash_mismatch(case, record):
    k1n(case)
    base = case["stage"] if record == "registration" else case["visible_dir"]
    change(base / (record + ".json"), lambda r: r.update(config_sha256="0" * 64))
    with pytest.raises(m.Refuse, match="config SHA256"):
        m.validate(**case)


def test_drop_o_missing_config(case):
    k1n(case)
    (case["stage"] / "config.json").unlink()
    with pytest.raises(m.Refuse, match="explicit boolean"):
        m.validate(**case)


@pytest.mark.parametrize("key", ["skipped_members", "intended_members", "executed_members"])
@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "null", "string", "absent"])
def test_drop_o_member_contract_refusals(case, key, defect):
    k1n(case)
    def mutate(r):
        if defect == "missing":
            r[key].pop()
        elif defect == "extra":
            r[key].append("R" if key == "skipped_members" else "EXP143v2")
        elif defect == "duplicate":
            r[key][-1] = r[key][0]
        elif defect == "null":
            r[key] = None
        elif defect == "string":
            r[key] = ",".join(r[key])
        else:
            del r[key]
    change(case["visible_dir"] / "c85r7-receipt.json", mutate)
    with pytest.raises(m.Refuse, match="member contract"):
        m.validate(**case)


@pytest.mark.parametrize("key", ["r_prediction", "s_r7_prediction", "r224_prediction"])
@pytest.mark.parametrize("defect", ["missing", "nonzero"])
def test_drop_o_remaining_prediction_counters_required_and_zero(case, key, defect):
    k1n(case)
    def mutate(r):
        if defect == "missing":
            del r["fallbacks"][key]
        else:
            r["fallbacks"][key] = 1
    change(case["visible_dir"] / "c85r7-receipt.json", mutate)
    with pytest.raises(m.Refuse, match="missing .*prediction fallback|nonzero counter"):
        m.validate(**case)


@pytest.mark.parametrize("key", ["a172_prediction", "extra_prediction"])
@pytest.mark.parametrize("value", [0, 1])
def test_drop_o_every_present_prediction_counter_still_zero(case, key, value):
    k1n(case)
    change(case["visible_dir"] / "c85r7-receipt.json", lambda r: r["fallbacks"].update({key: value}))
    transport(case["visible_dir"])
    if value:
        with pytest.raises(m.Refuse, match="nonzero counter"):
            m.validate(**case)
    else:
        m.validate(**case)


@pytest.mark.parametrize("registered", [False, True])
def test_ordinary_variant_still_requires_o_counter(case, registered):
    if registered:
        k1n(case, drop_o=False)
        m.validate(**case)
    change(case["visible_dir"] / "c85r7-receipt.json", lambda r: r["fallbacks"].pop("a172_prediction"))
    with pytest.raises(m.Refuse, match="missing parent prediction fallback"):
        m.validate(**case)


@pytest.mark.parametrize("file,key,value,reason", [
    ("k1sr-receipt.json", "status", "DEGRADED_PARENT_ONLY", "no degraded"),
    ("k1sr-receipt.json", "status", None, "no degraded"),
    ("k1sr-receipt.json", "k1_executed", False, "no degraded"),
    ("k1sr-receipt.json", "k1_executed", 1, "no degraded"),
    ("k1sr-receipt.json", "k1_studies", 2, "studies"),
    ("k1sr-receipt.json", "kernel_id", "owner/wrong", "kernel identity"),
    ("k1sr-receipt.json", "release_version", "1.2.2-k1", "release version"),
    ("k1sr-receipt.json", "never_submit", True, "never_submit"),
    ("k1sr-receipt.json", "diagnostic", True, "diagnostic"),
    ("k1sr-receipt.json", "parent_sha256", "0" * 64, "Parent CSV"),
    ("k1sr-receipt.json", "submission_sha256", "0" * 64, "Final submission"),
    ("k1sr-receipt.json", "formula", "different", "formula"),
    ("k1sr-receipt.json", "model_sha256", "b" * 64, "model_sha256"),
    ("c85r7-receipt.json", "status", "COMPLETE_C85R7_OTHER", "Parent COMPLETE"),
    ("c85r7-receipt.json", "version", "1.2.2-k1", "Parent COMPLETE"),
    ("c85r7-receipt.json", "studies", 4, "3 studies"),
    ("c85r7-receipt.json", "findings", 11, "12 findings"),
    ("c85r7-receipt.json", "fallbacks", {}, "missing parent"),
    ("c85r7-receipt.json", "codecs", None, "codecs"),
    ("c85r7-receipt.json", "codecs", {"status": "ACTIVE", "s_r7_gate": "FAIL"}, "S_R7 codecs"),
    ("c85r7-receipt.json", "codecs", {"status": "ACTIVE", "selftest": {"jpeg": False}}, "selftest"),
    ("c85r7-receipt.json", "submission_sha256", "0" * 64, "Parent CSV"),
])
def test_receipt_refusals(case, file, key, value, reason):
    change(case["visible_dir"] / file, lambda r: r.update({key: value}))
    transport(case["visible_dir"])
    with pytest.raises(m.Refuse, match=reason):
        m.prepare(**case)
    assert not (case["visible_dir"] / m.OUTPUT / "READY-REVIEW.md").exists()


@pytest.mark.parametrize("key", ["a172_prediction", "r_prediction", "s_r7_prediction", "r224_prediction"])
def test_all_prediction_fallbacks(case, key):
    change(case["visible_dir"] / "c85r7-receipt.json", lambda r: r["fallbacks"].update({key: 1}))
    with pytest.raises(m.Refuse, match="nonzero counter"):
        m.validate(**case)


def test_anchor_fallback_is_not_prediction_failure(case):
    change(case["visible_dir"] / "c85r7-receipt.json",
           lambda r: r.update(s_r7_anatomy_anchors={"fallback": 2}))
    transport(case["visible_dir"])
    m.validate(**case)


@pytest.mark.parametrize("file", [
    "k1sr-receipt.json", "c85r7-receipt.json", "submission.csv", "k1sr-parent.csv",
    "k1sr-receipt.json.download.json", "c85r7-receipt.json.download.json", "submission.csv.download.json"])
def test_missing_input(case, file):
    (case["visible_dir"] / file).unlink()
    with pytest.raises(m.Refuse, match="missing input"):
        m.validate(**case)


@pytest.mark.parametrize("key,value,reason", [
    ("never_submit", True, "never_submit"), ("diagnostic", True, "diagnostic"),
    ("kernel_sha256", "0" * 64, "release.ipynb hash"),
    ("metadata_sha256", "0" * 64, "kernel-metadata.json hash"),
    ("stage", ".", "stage directory"),
    ("release_version", "1.2.2", "release version")])
def test_stage_refusals(case, key, value, reason):
    change(case["stage"] / "registration.json", lambda r: r.update({key: value}))
    with pytest.raises(m.Refuse, match=reason):
        m.validate(**case)


@pytest.mark.parametrize("key,value", [
    ("version", 2), ("version", True), ("kernel", "owner/other"), ("status", "FAILED"),
    ("sha256", "0" * 64), ("file", "other.csv")])
def test_download_provenance(case, key, value):
    change(case["visible_dir"] / "submission.csv.download.json", lambda r: r.update({key: value}))
    with pytest.raises(m.Refuse, match="download provenance"):
        m.validate(**case)


@pytest.mark.parametrize("defect", ["nan", "inf", "negative", "too_high", "columns", "duplicate", "wrong_id", "order", "extra"])
def test_csv_refusals_even_with_matching_hash(case, defect):
    p = case["visible_dir"] / "submission.csv"
    text = p.read_text()
    if defect in {"nan", "inf", "negative", "too_high"}:
        text = text.replace("0.5", {"nan": "NaN", "inf": "inf", "negative": "-0.1", "too_high": "1.1"}[defect], 1)
    elif defect == "columns":
        text = text.replace("target11", "wrong")
    elif defect == "duplicate":
        text = text.replace("\nb,", "\na,")
    elif defect == "wrong_id":
        text = text.replace("\nb,", "\nx,")
    elif defect == "order":
        lines = text.splitlines()
        text = "\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n"
    else:
        text += text.splitlines()[-1] + "\n"
    p.write_text(text)
    change(case["visible_dir"] / "k1sr-receipt.json", lambda r: r.update(submission_sha256=m.sha(p.read_bytes())))
    transport(case["visible_dir"])
    with pytest.raises(m.Refuse, match="CSV|template"):
        m.validate(**case)


def test_ledger_fail_closed(case):
    m.LEDGER.write_text(json.dumps(dict(kernel=case["kernel"], version="1", outcome="SUBMITTED")) + "\n")
    with pytest.raises(m.Refuse, match="already SUBMITTED"):
        m.validate(**case)
    m.LEDGER.write_text('{"broken":\n')
    with pytest.raises(ValueError):
        m.validate(**case)


def test_no_n32_candidate(case):
    case["kernel"] += "-n32"
    with pytest.raises(m.Refuse, match="N32 candidate"):
        m.validate(**case)


def test_n32_projection_and_real_check_only(case, capsys):
    n32(case)
    out = m.prepare(**case)
    p = json.loads((out / "ordinary-visible-registration.json").read_bytes())["projection"]
    assert p["parent_per_study_seconds"] == (1000 - 300) / 32 * .5
    assert p["projected_hours"] == pytest.approx((300 + 60 + 1300 * (10.9375 + 102 / 32)) / 3600)
    assert "PROJECTION" in capsys.readouterr().out


@pytest.mark.parametrize("key,value,reason", [
    ("status", "DEGRADED_PARENT_ONLY", "32/32"), ("k1_executed", False, "32/32"),
    ("k1_studies", 31, "32/32"), ("per_study", [], "per-study"),
    ("model_sha256", "b" * 64, "model_sha256"), ("formula", "other", "formula")])
def test_n32_refusals(case, key, value, reason):
    d = n32(case)
    change(d / "k1sr-receipt.json", lambda r: r.update({key: value}))
    with pytest.raises(m.Refuse, match=reason):
        m.validate(**case)


def test_n32_requires_documented_evidence(case):
    d = n32(case)
    (d / "projection-inputs.json").unlink()
    with pytest.raises(m.Refuse, match="projection-inputs.json"):
        m.validate(**case)


def test_n32_over_8_hours(case):
    d = n32(case)
    change(d / "train.json", lambda r: r.update(dicom_files=2000000))
    change(d / "projection-inputs.json",
           lambda r: r["train"].update(sha256=m.sha((d / "train.json").read_bytes())))
    with pytest.raises(m.Refuse, match="<= 8.0"):
        m.prepare(**case)
    assert not (case["visible_dir"] / m.OUTPUT / "READY-REVIEW.md").exists()


def test_stale_projection_hash(case):
    d = n32(case)
    change(d / "startup.json", lambda r: r.update(parent_startup_seconds=900))
    with pytest.raises(m.Refuse, match="projection evidence hash"):
        m.validate(**case)


def test_canonical_refusal_never_writes_ready_or_command(case, monkeypatch, capsys):
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(
        a, 2, '{"outcome":"REFUSED"}', ""))
    with pytest.raises(m.Refuse, match="check-only REFUSED"):
        m.prepare(**case)
    assert not (case["visible_dir"] / m.OUTPUT / "READY-REVIEW.md").exists()
    assert "SUBMIT COMMAND" not in capsys.readouterr().out


def test_inputs_changed_during_canonical_check_refused(case, monkeypatch):
    run = subprocess.run

    def changed(*args, **kwargs):
        result = run(*args, **kwargs)
        (case["visible_dir"] / "submission.csv").write_bytes(b"changed")
        return result

    monkeypatch.setattr(m.subprocess, "run", changed)
    with pytest.raises(m.Refuse, match="changed"):
        m.prepare(**case)
    assert not (case["visible_dir"] / m.OUTPUT / "READY-REVIEW.md").exists()


def test_existing_artifacts_not_overwritten(case):
    out = case["visible_dir"] / m.OUTPUT
    out.mkdir()
    p = out / "ordinary-visible-registration.json"
    p.write_bytes(b"prior immutable artifact")
    with pytest.raises(m.Refuse, match="existing admission differs"):
        m.prepare(**case)
    assert p.read_bytes() == b"prior immutable artifact"


def test_json_duplicates_rejected(case):
    (case["visible_dir"] / "k1sr-receipt.json").write_bytes(b'{"status":"x","status":"y"}')
    with pytest.raises(m.Refuse, match="duplicate JSON"):
        m.validate(**case)


def test_path_guard_and_command_quoting(case):
    with pytest.raises(m.Refuse, match="forbidden"):
        m.local(m.ROOT / ".local/relabel/receipt.json")
    with pytest.raises(m.Refuse, match="inside project"):
        m.local(m.ROOT.parent / "receipt.json")
    assert m.command_line(["python", "has'quote", "$literal"]) == "& 'python' 'has''quote' '$literal'"


def test_cli_refusal(case, capsys):
    code = m.main(["prepare", "--stage", str(case["stage"]), "--kernel", "owner/k1-n32",
                   "--version", "1", "--visible-dir", str(case["visible_dir"])])
    assert code == 2
    assert "REFUSE:" in capsys.readouterr().out


def test_submitted_outcome_takes_precedence(case):
    m.LEDGER.write_text("".join(json.dumps(dict(
        kernel=case["kernel"], version=1, outcome=outcome)) + "\n"
        for outcome in ("IN_FLIGHT", "SUBMITTED")))
    with pytest.raises(m.Refuse, match="already SUBMITTED"):
        m.validate(**case)


@pytest.mark.parametrize("outcome", ["IN_FLIGHT", "UNCONFIRMED", "RESOLVED_ABSENT"])
def test_prior_attempt_refused(case, outcome):
    m.LEDGER.write_text(json.dumps(dict(kernel=case["kernel"], version=1, outcome=outcome)))
    with pytest.raises(m.Refuse, match="prior or unresolved"):
        m.validate(**case)


@pytest.mark.parametrize("codec", [
    {"status": "ACTIVE", "selftest": []},
    {"status": "ACTIVE", "totals": []},
    {"status": "ACTIVE", "totals": {"plugin_failures": 1}},
    {"status": "ACTIVE", "totals": {"decode_failures": 1}},
    {"status": "ACTIVE", "hook_errors": ["failed"]},
])
def test_active_codec_with_errors_refused(case, codec):
    change(case["visible_dir"] / "c85r7-receipt.json", lambda r: r.update(codecs=codec))
    with pytest.raises(m.Refuse, match="codec"):
        m.validate(**case)


def test_r224_requires_counter(case):
    def remove_counter(r):
        r["intended_members"] = ["R224"]
        del r["fallbacks"]["r224_prediction"]
    change(case["visible_dir"] / "c85r7-receipt.json", remove_counter)
    with pytest.raises(m.Refuse, match="missing R224"):
        m.validate(**case)


@pytest.mark.parametrize("name,key,value,reason", [
    ("startup", "parent_startup_seconds", 1000, "smaller"),
    ("startup", "parent_startup_seconds", -1, "invalid"),
    ("startup", "method", "", "method"),
    ("train", "studies", 0, "invalid"),
    ("train", "method", "", "provenance"),
    ("selection", "counts", {"wrong": 1}, "roster/counts"),
    ("selection", "counts", {str(i): 0 for i in range(32)}, "invalid"),
    ("selection", "counts", {str(i): 1e308 for i in range(32)}, "invalid"),
])
def test_correctly_hashed_invalid_projection_inputs(case, name, key, value, reason):
    d = n32(case)
    change(d / (name + ".json"), lambda r: r.update({key: value}))
    change(d / "projection-inputs.json", lambda r: r[name].update(
        sha256=m.sha((d / (name + ".json")).read_bytes())))
    with pytest.raises(m.Refuse, match=reason):
        m.validate(**case)


def test_n32_inconsistent_phase_sum(case):
    d = n32(case)
    change(d / "k1sr-receipt.json", lambda r: r["phases"].update(classifier_seconds=101))
    digest = m.sha((d / "k1sr-receipt.json").read_bytes())
    change(d / "startup.json", lambda r: r.update(receipt_sha256=digest))
    def rebind(r):
        r["receipt_sha256"] = digest
        r["startup"]["sha256"] = m.sha((d / "startup.json").read_bytes())
    change(d / "projection-inputs.json", rebind)
    with pytest.raises(m.Refuse, match="phase/per-study"):
        m.validate(**case)


def test_generated_binding_changed_during_canonical_check(case, monkeypatch):
    run = subprocess.run
    def changed(*args, **kwargs):
        result = run(*args, **kwargs)
        (case["visible_dir"] / m.OUTPUT / "submission-binding.json").write_bytes(b"changed")
        return result
    monkeypatch.setattr(m.subprocess, "run", changed)
    with pytest.raises(m.Refuse, match="changed"):
        m.prepare(**case)
    assert not (case["visible_dir"] / m.OUTPUT / "READY-REVIEW.md").exists()
