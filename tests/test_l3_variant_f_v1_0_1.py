"""Synthetic-only safety/throughput contracts; no reports, label rows or reader calls."""
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import l3_variant_f_v1_0_1 as m


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def findings(p=.5):
    return {t: {"p": p, "severity": "uncertain"} for t in m.TARGETS}


def usage(n=100):
    return {"input_tokens": n, "output_tokens": 20, "cached_input_tokens": 10}


def stream(code="T00001", payload=None, tokens=None):
    return "\n".join([json.dumps({"type": "item.completed", "item": {
        "type": "agent_message", "text": json.dumps({code: payload or findings()})}}),
        json.dumps({"type": "turn.completed", "usage": tokens or usage()})])


@pytest.mark.parametrize("bad", [True, "0.5", -0.01, 1.01, None, float("nan"), float("inf")])
def test_strict_rejects_invalid_probability(bad):
    data = findings(); data[m.TARGETS[0]]["p"] = bad
    with pytest.raises(m.GuardError):
        m.strict_parse(json.dumps({"T": data}), "T")


def test_strict_exact_schema_no_clipping():
    data = findings(0); data[m.TARGETS[-1]]["p"] = 1
    assert m.strict_parse(json.dumps({"T": data}), "T") == data
    for mutant in ({"wrong": data}, {"T": {**data, "extra": {"p": .5}}},
                   {"T": {k: v for k, v in data.items() if k != m.TARGETS[0]}}):
        with pytest.raises(m.GuardError):
            m.strict_parse(json.dumps(mutant), "T")
    data[m.TARGETS[0]]["severity"] = "free prose"
    with pytest.raises(m.GuardError, match="SEVERITY_ENUM"):
        m.strict_parse(json.dumps({"T": data}), "T")
    with pytest.raises(m.GuardError, match="DUPLICATE_JSON_KEY"):
        m.strict_parse('{"T":{},"T":{}}', "T")


def test_variant_f_prefix_and_definitions_preserved():
    labels = {t: {"p": .5, "conf": .8} for t in m.TARGETS}
    original = m.e147._prompt_f("T", "SYNTHETIC ONLY", {"T": labels})
    terminal = 'Return ONLY the JSON object {"T": {...12 findings...}}.\n'
    new = m.prompt_f("T", "SYNTHETIC ONLY", labels)
    assert new.startswith(original.removesuffix(terminal))
    assert "not_assessable" in new and "Do not quote" in new
    assert m.prompt_f("T", "SYNTHETIC ONLY", labels, True).startswith(new)


def test_deterministic_sampling_and_rare_strata():
    rows = [{"uid": f"u{f}-{i}", "code": f"C{f}-{i:03}", "fold": f, "ineligible": []}
            for f in range(5) for i in range(45)]
    rows += [{"uid": "unassigned", "code": "U", "fold": -1, "ineligible": []},
             {"uid": "blank", "code": "B", "fold": 0, "ineligible": ["unavailable_images"]}]
    flight = pd.DataFrame({t: [(.5 if i % 45 == 0 else .1 if i % 2 else .9)
                             for i in range(len(rows))] for t in m.WEAK}, index=[r["uid"] for r in rows])
    chosen = m.pick_pilot(rows, flight)
    assert chosen == m.pick_pilot(list(reversed(rows)), flight)
    assert len({r["code"] for r in chosen}) == 200
    assert all(sum(r["fold"] == f for r in chosen) == 40 for f in range(5))
    assert not {"U", "B"} & {r["code"] for r in chosen}
    assert all(any(r["fold"] == f and r["bins"][m.WEAK[0]] == "uncertain" for r in chosen) for f in range(5))


def test_registration_gold_and_identity_guard(tmp_path, monkeypatch):
    reg = {"version": m.VERSION, "sources": {}, "cli_identity": {}, "inputs": {},
           "pilot": [{"code": str(i)} for i in range(200)], "rows": [{"uid": "gold"}],
           "gold_membership_sha256": m.digest(["gold"])}
    m.atomic(tmp_path / "registration.json", reg)
    monkeypatch.setattr(m, "source_hashes", lambda: {})
    monkeypatch.setattr(m, "cli_identity", lambda: {})
    monkeypatch.setattr(m.e109, "gold_uids", lambda _: {"gold"})
    with pytest.raises(m.GuardError, match="GOLD_EXCLUSION"):
        m.verify_registration(tmp_path)
    reg["rows"] = [{"uid": "nongold"}]
    m.atomic(tmp_path / "registration.json", reg)
    assert m.verify_registration(tmp_path) == reg
    monkeypatch.setattr(m, "source_hashes", lambda: {"changed": "hash"})
    with pytest.raises(m.GuardError, match="SOURCE_IDENTITY_CHANGED"):
        m.verify_registration(tmp_path)


def test_slot_selection_exact_exp145():
    mask = np.ones(64, dtype=bool)
    for plane in ("sag", "cor", "ax"):
        start, width = m.e145.GROUPS[plane][0]
        lo, hi = m.e145.SPAN[plane]
        expected = [start + round(k) for k in np.linspace(lo*(width-1), hi*(width-1), m.e145.COUNT[plane])]
        assert m.select_slots(mask, plane) == expected
        assert m.select_slots(np.zeros(64, dtype=bool), plane) == []


def test_ledger_concurrency_budget_duplicate_and_completion(tmp_path):
    ledger = m.Ledger(tmp_path, "hash", 100_000, "pilot")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(ledger.reserve, ["T"] * 8))
    assert sum(r[0] == "reserved" for r in results) == 1
    assert ledger.reserve("U") == ("reserved", 1)
    assert ledger.reserve("V")[0] == "budget"
    ledger.finish("T", 1, usage(), 3, findings())
    assert ledger.reserve("T")[0] == "complete"
    assert ledger.summary()["total_tokens"] == 120
    assert "payload" not in (tmp_path / "receipts/T.json").read_text()
    assert m.load(tmp_path / "labels/T.json")["findings"] == findings()


@pytest.mark.parametrize("kind", ["crash", "missing_usage", "overrun", "tool"])
def test_unknown_crash_overrun_and_tool_fail_closed(tmp_path, kind):
    ledger = m.Ledger(tmp_path, "hash", 200_000, "pilot")
    ledger.reserve("T")
    if kind != "crash":
        ledger.finish("T", 1, None if kind == "missing_usage" else usage(60_000 if kind == "overrun" else 100),
                      2, None, "FORBIDDEN_TOOL_EVENT" if kind == "tool" else "INVALID_JSON")
    with pytest.raises(m.GuardError, match="RECONCILIATION"):
        m.Ledger(tmp_path, "hash", 200_000, "pilot").startup_guard()
    assert ledger.reserve("T")[0] in ("halt", "blocked")


def test_max_attempts_and_export_failure_do_not_repeat(tmp_path, monkeypatch):
    ledger = m.Ledger(tmp_path, "hash", 200_000, "pilot")
    for attempt in (1, 2):
        assert ledger.reserve("bad") == ("reserved", attempt)
        ledger.finish("bad", attempt, usage(), 1, None, "INVALID_JSON")
    assert ledger.reserve("bad")[0] == "exhausted"
    ledger.reserve("good")
    monkeypatch.setattr(ledger, "export_receipt", lambda _: (_ for _ in ()).throw(PermissionError()))
    ledger.finish("good", 1, usage(), 1, findings())
    assert ledger.reserve("good")[0] == "complete"
    assert ledger.summary()["total_tokens"] == 360


def test_full_reuses_payload_receipt_attempts_and_wall_once(tmp_path):
    pilot = m.Ledger(tmp_path / "pilot", "hash", m.PILOT_CEILING, "pilot")
    session = pilot.start_session(2)
    pilot.reserve("T"); pilot.finish("T", 1, usage(), 3, findings())
    pilot.reserve("U"); pilot.finish("U", 1, usage(), 3, None, "INVALID_JSON")
    pilot.finish_session(session, 4)
    full = m.Ledger(tmp_path / "full", "hash", 10_000_000, "full")
    assert full.reuse_pilot(pilot, tmp_path) == 1
    assert full.reuse_pilot(pilot, tmp_path) == 1
    assert full.summary()["total_tokens"] == 240
    assert full.summary()["wall_seconds_all_sessions"] == 4
    assert full.reserve("T")[0] == "complete"
    assert full.reserve("U") == ("reserved", 2)
    assert m.load(tmp_path / "full/labels/T.json")["findings"] == findings()
    refs = m.load(tmp_path / "full/reused-pilot.json")
    assert len(refs["sources"]) == 2 and refs["budget_includes_pilot_spend"]


def test_event_usage_and_forbidden_tools():
    tokens, answer, err = m.parse_events(stream())
    assert tokens == usage() and not err and m.strict_parse(answer, "T00001") == findings()
    for raw in ("[]", "null", "invalid", stream(tokens={"input_tokens": True, "output_tokens": 2}),
                stream() + '\n{"type":"turn.completed","usage":{}}'):
        assert m.parse_events(raw)[2] == "EVENT_STREAM_INVALID"
    raw = stream() + '\n{"type":"item.started","item":{"type":"command_execution"}}'
    assert m.parse_events(raw)[2] == "FORBIDDEN_TOOL_EVENT"


def test_invoke_transport_uses_pipes_no_raw_logs(tmp_path):
    script = tmp_path / "fake_reader.py"
    script.write_text("import os,sys\nassert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
                      "assert sys.stdin.read()=='SYNTHETIC PROMPT'\n"
                      "print(" + repr(stream()) + ")\n", encoding="utf-8")
    tokens, answer, err = m.invoke_reader([sys.executable, "-B", str(script)], "SYNTHETIC PROMPT")
    assert tokens == usage() and not err
    assert m.strict_parse(answer, "T00001") == findings()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fake_reader.py"]


def test_reader_command_no_prompt_or_output_file():
    cmd = m.reader_command("T")
    assert "--ephemeral" in cmd and "--json" in cmd and "-o" not in cmd
    assert cmd[-1] == "-" and cmd.count("-i") == 3
    for feature in ("shell_tool", "multi_agent", "plugins", "view_image", "tool_search"):
        assert cmd[cmd.index(feature)-1] == "--disable"


def test_cli_preflight_effective_flags_and_fail_closed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cmd = m.reader_command("SCHEMA_CODE", tmp_path)
    names = [cmd[i+1] for i, arg in enumerate(cmd) if arg == "--disable"]
    monkeypatch.setattr(m, "cli_identity", lambda: {})
    effective = {name: ("true" if name == "unified_exec" else "false") for name in names}
    def fake(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="\n".join(f"{n} stable {v}" for n, v in effective.items()))
    monkeypatch.setattr(m.subprocess, "run", fake)
    rec = m.cli_preflight(tmp_path)
    assert rec["effective_features"]["shell_tool"] == "false"
    assert rec["forced_backend_features"] == ["unified_exec"]
    effective["shell_tool"] = "true"
    with pytest.raises(m.GuardError, match="CLI_PREFLIGHT_FLAGS"):
        m.cli_preflight(tmp_path)


def test_review_identity_scope_and_independence(tmp_path):
    for name in ("registration.json", "cli-preflight.json"):
        m.atomic(tmp_path / name, {})
    m.atomic(tmp_path / "cpu-smoke.json", {"registration_sha256": m.sha(tmp_path / "registration.json"), "passed": True})
    (tmp_path / "review.md").write_text("Independent test review", encoding="utf-8")
    rec = {"verdict": "PASS", "scope": "pilot", "version": m.VERSION,
           "code_sha256": m.sha(m.__file__), "registration_sha256": m.sha(tmp_path / "registration.json"),
           "cpu_smoke_sha256": m.sha(tmp_path / "cpu-smoke.json"),
           "cli_preflight_sha256": m.sha(tmp_path / "cli-preflight.json"),
           "reviewer": "independent", "report": "review.md", "report_sha256": m.sha(tmp_path / "review.md")}
    m.atomic(tmp_path / "review.json", rec)
    assert m.review_gate(tmp_path, tmp_path / "review.json", "pilot") == rec
    with pytest.raises(m.GuardError, match="REVIEW_IDENTITY"):
        m.review_gate(tmp_path, tmp_path / "review.json", "full")
    rec["reviewer"] = "l3_builder"; m.atomic(tmp_path / "review.json", rec)
    with pytest.raises(m.GuardError, match="INDEPENDENT_REVIEWER"):
        m.review_gate(tmp_path, tmp_path / "review.json", "pilot")


def test_exam_retries_only_known_usage_and_no_report_output(tmp_path, monkeypatch, capsys):
    row = {"code": "T", "uid": "u"}
    flight = pd.DataFrame({k: [.5] for t in m.TARGETS for k in (t, t+"__conf")}, index=["u"])
    monkeypatch.setattr(m, "private_report", lambda _: "PRIVATE SYNTHETIC SENTINEL")
    ledger = m.Ledger(tmp_path, "hash", 200_000, "pilot")
    calls = []
    def reader(args, prompt):
        assert "PRIVATE SYNTHETIC SENTINEL" in prompt
        calls.append(1)
        return usage(), "invalid" if len(calls) == 1 else json.dumps({"T": findings()}), None
    assert m.exam(tmp_path, row, flight, ledger, reader) == "complete"
    assert len(calls) == 2 and ledger.summary()["retry_attempts"] == 1
    assert m.exam(tmp_path, row, flight, ledger, reader) == "complete" and len(calls) == 2
    assert "PRIVATE SYNTHETIC SENTINEL" not in capsys.readouterr().out
    for artifact in tmp_path.rglob("*.json"):
        assert "PRIVATE SYNTHETIC SENTINEL" not in artifact.read_text()


def test_atomic_bounded_windows_retry_and_immutable(tmp_path, monkeypatch):
    counter = [0]
    replace = m.os.replace
    def flaky(a, b):
        counter[0] += 1
        if counter[0] < 3:
            raise PermissionError()
        replace(a, b)
    monkeypatch.setattr(m.os, "replace", flaky)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    m.immutable(tmp_path / "x.json", {"safe": True})
    assert counter[0] == 3
    m.immutable(tmp_path / "x.json", {"safe": True})
    with pytest.raises(m.GuardError, match="IMMUTABLE"):
        m.immutable(tmp_path / "x.json", {"safe": False})


def mock_run_inputs(tmp_path, monkeypatch):
    rows = [{"uid": f"u{i}", "code": f"C{i}", "ineligible": []} for i in range(3)]
    reg = {"pilot": [rows[2], rows[0], rows[1]], "rows": rows, "pilot_lanes": 2,
           "pilot_token_ceiling": m.PILOT_CEILING, "full_eligible": True,
           "counts": {"non_gold": 3}, "interpretation": "descriptive"}
    m.atomic(tmp_path / "registration.json", reg)
    monkeypatch.setattr(m, "verify_registration", lambda _: reg)
    monkeypatch.setattr(m, "review_gate", lambda *a: None)
    monkeypatch.setattr(m, "cli_preflight", lambda *a: None)
    monkeypatch.setattr(m, "validate", lambda *a: None)
    flight = pd.DataFrame({t: [.1, .5, .9] for t in m.TARGETS}, index=[r["uid"] for r in rows])
    monkeypatch.setattr(m, "inputs", lambda: (rows, flight, None, set()))
    calls = []
    def fake_exam(work, row, flight, ledger):
        state, attempt = ledger.reserve(row["code"])
        if state == "reserved":
            calls.append(row["code"])
            ledger.finish(row["code"], attempt, usage(), 2, findings(float(flight.loc[row["uid"], m.TARGETS[0]])))
            return "complete"
        return state
    monkeypatch.setattr(m, "exam", fake_exam)
    return calls


def test_first_two_pilot_resume_full_and_merged_report(tmp_path, monkeypatch):
    calls = mock_run_inputs(tmp_path, monkeypatch)
    first = m.run(tmp_path, limit=2)
    assert first["selected"] == 2 and set(calls) == {"C0", "C2"}
    m.run(tmp_path)
    assert sorted(calls) == ["C0", "C1", "C2"]
    full = m.run(tmp_path, "full", ceiling=10_000_000, confirm_full=True)
    assert full["reused_pilot"] == 3 and len(calls) == 3
    result = m.report(tmp_path, "full", 4, 1, 2)
    assert result["completed_exams"] == 3 and result["total_tokens"] == 360
    assert result["failure_attempt_rate"] == 0 and result["sessions"] == 3
    assert all(v["mae"] == 0 for v in result["per_target_flight_agreement"].values())
    assert result["full_projection"]["remaining_exams_after_pilot"] == 0
    with pytest.raises(m.GuardError, match="PILOT_FROZEN"):
        m.run(tmp_path)
    with pytest.raises(m.GuardError, match="ONLY_FIRST_TWO"):
        m.run(tmp_path, limit=3)


def test_report_refuses_live_run(tmp_path):
    with m.run_lock(tmp_path):
        with pytest.raises(m.GuardError, match="RUN_ALREADY_ACTIVE"):
            m.report(tmp_path)


def test_cli_never_prints_arbitrary_exception(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(m, "fixed_work", lambda _: tmp_path)
    monkeypatch.setattr(m, "register", lambda _: (_ for _ in ()).throw(ValueError("PRIVATE SENTINEL")))
    assert m.main(["register"]) == 2
    output = capsys.readouterr().out
    assert "PRIVATE SENTINEL" not in output and "VALUEERROR" in output


def warning(message=None, event_type="item.completed", item_type="error"):
    return json.dumps({"type": event_type, "item": {"type": item_type,
                      "message": m.BENIGN_DIAGNOSTIC if message is None else message}})


def test_exact_diagnostic_allows_complete_usage_only():
    tokens, answer, err = m.parse_events(warning() + "\n" + stream())
    assert tokens == usage() and err is None
    assert m.strict_parse(answer, "T00001") == findings()
    assert m.parse_events(warning())[0] is None  # warning is never proof of no charge


@pytest.mark.parametrize("event", [warning("UNKNOWN PRIVATE SENTINEL"),
    warning(m.BENIGN_DIAGNOSTIC + " "), warning(event_type="item.started"),
    json.dumps({"type": "turn.failed", "error": {"message": "PRIVATE SENTINEL"}}),
    json.dumps({"type": "error", "message": "PRIVATE SENTINEL"})])
def test_all_other_diagnostics_fatal_without_message_leak(event):
    tokens, _, err = m.parse_events(event + "\n" + stream())
    assert tokens == usage() and err == "READER_FATAL_EVENT"
    assert "PRIVATE" not in err


def test_known_warning_text_cannot_disguise_tool_event():
    assert m.parse_events(warning(item_type="command_execution") + "\n" + stream())[2] == "FORBIDDEN_TOOL_EVENT"


@pytest.mark.parametrize("prefix,expected", [(warning(), None), (warning("UNKNOWN PRIVATE SENTINEL"), "READER_FATAL_EVENT"),
    (warning(item_type="command_execution"), "FORBIDDEN_TOOL_EVENT")])
def test_live_stream_uses_same_diagnostic_classifier(tmp_path, prefix, expected, capsys):
    script = tmp_path / "fake_reader.py"
    script.write_text("import sys\nsys.stdin.read()\nprint(" + repr(prefix + "\n" + stream()) + ")\n", encoding="utf-8")
    tokens, answer, err = m.invoke_reader([sys.executable, "-B", str(script)], "SYNTHETIC ONLY")
    assert err == expected
    if expected is None:
        assert tokens == usage() and m.strict_parse(answer, "T00001") == findings()
    else:
        assert tokens is None
    assert "PRIVATE" not in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == [script]


def legacy_fixture(tmp_path, monkeypatch, codes=("T", "U")):
    prior = m.Ledger(tmp_path / "original", "legacy", 4_000_000, "pilot")
    session = prior.start_session(2)
    for code in codes:
        prior.reserve(code); prior.finish(code, 1, None, 1, None, "FORBIDDEN_TOOL_EVENT")
        # Simulate two calls reserved concurrently before either returned.
        if code != codes[-1]:
            with prior.connect() as con:
                con.execute("DELETE FROM meta WHERE key='halt'")
    prior.finish_session(session, 1)
    with prior.connect() as con:
        attempts = [dict(r) for r in con.execute("SELECT * FROM attempts ORDER BY code,attempt")]
        sessions = [dict(r) for r in con.execute("SELECT * FROM sessions ORDER BY id")]
    amendment = {"prior_attempts_sha256": m.digest(attempts), "prior_sessions_sha256": m.digest(sessions),
        "prior_unknown_provisional_hold_tokens": 100_000, "prior_unknown_codes": list(codes), "full_blocked": True}
    monkeypatch.setattr(m, "legacy_snapshot", lambda: (attempts, sessions))
    return prior, {"amendment": amendment}


def test_carry_unknown_idempotent_no_source_mutation_hold_and_no_replay(tmp_path, monkeypatch):
    prior, reg = legacy_fixture(tmp_path, monkeypatch)
    old_hash = m.sha(prior.path)
    ledger = m.Ledger(tmp_path / "new", "newreg", 150_000, "pilot")
    ledger.carry_unknown(reg); ledger.carry_unknown(reg); ledger.startup_guard()
    assert m.sha(prior.path) == old_hash and prior.summary()["halt"] == "UNKNOWN_USAGE"
    assert ledger.summary()["unknown_requires_manual_reconciliation"]
    assert ledger.summary()["prior_unknown_provisional_hold_tokens"] == 100_000
    assert ledger.summary()["unknown_actual_usage_tokens"] is None
    assert ledger.reserve("T")[0] == "blocked"
    assert ledger.reserve("new")[0] == "reserved"
    assert ledger.reserve("extra")[0] == "budget"
    ledger.finish("new", 1, usage(), 2, findings())
    ledger.startup_guard()
    assert ledger.summary()["attempted_exams"] == 3


def test_new_unknown_halts_amended_stage(tmp_path, monkeypatch):
    _, reg = legacy_fixture(tmp_path, monkeypatch)
    ledger = m.Ledger(tmp_path / "new", "newreg", 200_000, "pilot")
    ledger.carry_unknown(reg); ledger.reserve("new")
    ledger.finish("new", 1, None, 1, None, "READER_FATAL_EVENT")
    with pytest.raises(m.GuardError, match="RECONCILIATION"):
        ledger.startup_guard()
    assert ledger.reserve("other")[0] == "halt"


def test_changed_inherited_unknown_never_exempted(tmp_path, monkeypatch):
    _, reg = legacy_fixture(tmp_path, monkeypatch)
    ledger = m.Ledger(tmp_path / "new", "newreg", 200_000, "pilot")
    ledger.carry_unknown(reg)
    with ledger.connect() as con:
        con.execute("UPDATE attempts SET elapsed=9 WHERE code='T'")
    with pytest.raises(m.GuardError, match="RECONCILIATION"):
        ledger.startup_guard()
    with pytest.raises(m.GuardError, match="IMPORT_CONFLICT"):
        ledger.carry_unknown(reg)


def test_amended_run_skips_original_failures_and_full_is_blocked(tmp_path, monkeypatch):
    calls = mock_run_inputs(tmp_path, monkeypatch)
    _, amendment = legacy_fixture(tmp_path, monkeypatch, ("C2", "C0"))
    reg = m.load(tmp_path / "registration.json"); reg.update(amendment)
    m.atomic(tmp_path / "registration.json", reg)
    monkeypatch.setattr(m, "verify_registration", lambda _: reg)
    result = m.run(tmp_path, limit=2)
    assert calls == ["C1"] and result["prior_unknown_excluded_no_replay"] == 2
    assert result["completed_exams"] == 1 and result["attempted_exams"] == 3
    m.run(tmp_path)
    assert calls == ["C1"]
    report = m.report(tmp_path, "pilot", 2, 1, 2)
    assert report["unknown_attempts"] == 2 and report["failed_attempts"] == 2
    assert report["known_tokens_p50_per_completed_exam"] == 120
    assert report["known_tokens_p95_per_completed_exam"] == 120
    assert not report["full_projection"]["usage_accounting_complete"]
    assert set(report["known_usage_projection"]["ideal_hours_by_lanes"]) == {"2", "4", "6"}
    with pytest.raises(m.GuardError, match="FULL_BLOCKED"):
        m.run(tmp_path, "full", confirm_full=True)
    assert not (tmp_path / "full").exists()


def test_fatal_diagnostic_known_usage_still_halts(tmp_path):
    ledger = m.Ledger(tmp_path, "hash", 200_000, "pilot")
    ledger.reserve("T"); ledger.finish("T", 1, usage(), 1, None, "READER_FATAL_EVENT")
    assert ledger.reserve("U")[0] == "halt"
