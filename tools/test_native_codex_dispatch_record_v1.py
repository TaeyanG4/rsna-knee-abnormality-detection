"""Native archive boundary tests; all receipts are temporary fixtures only."""
import json
from pathlib import Path

import pytest

from tools import native_codex_dispatch_record_v1 as n
from tools import strict_variant_gate as gate


def fixture(root, lane='native-astra-b1', report_text='Kernel-SHA256: ' + 'a' * 64 + '\n'):
    home = root / lane
    home.mkdir(parents=True)
    prompt, report = home / 'prompt.md', home / 'report.md'
    prompt.write_text('TEST FIXTURE ONLY. No production native task.\n', encoding='utf-8')
    report.write_text(report_text, encoding='utf-8')
    value = dict(schema=n.SCHEMA, host=n.HOST, model=n.MODEL, agent_id='/root/test_' + lane.replace('-', '_'),
                 lane=lane, prompt=str(prompt), report=str(report), terminal_status='completed',
                 observed_at='2026-10-18T13:00:00+09:00', prompt_sha256=n.digest(prompt.read_bytes()),
                 report_sha256=n.digest(report.read_bytes()), completion_message='TEST FIXTURE ONLY completion')
    source = home / 'completion.json'
    source.write_bytes(n.wire(value))
    return source, value, report


def test_unchanged_gate_reads_native_archive_and_distinct_agents(tmp_path):
    bp, _, _ = fixture(tmp_path)
    rp, _, report = fixture(tmp_path, 'native-astra-r1', 'Reviewer-Lane: native-astra-r1\nVERDICT: PASS\n')
    br = n.record(bp, 'test fixture', tmp_path)
    rr = n.record(rp, 'test fixture', tmp_path)
    b = gate.dispatch(br, gate.Inputs(tmp_path))
    r = gate.dispatch(rr, gate.Inputs(tmp_path))
    assert b[0] != r[0]
    assert r[2] == report
    assert n.verify(rr, tmp_path)['host'] == n.HOST


@pytest.mark.parametrize('key,value', [
    ('terminal_status', 'failed'), ('terminal_status', 'running'), ('model', 'opus'),
    ('host', 'Claude Code subagent'), ('lane', 'astra-4'), ('agent_id', 'invented-cli-session'),
    ('observed_at', '2026-10-18T13:00:00'), ('report_sha256', '0' * 64),
    ('prompt_sha256', '0' * 64), ('completion_message', ''),
])
def test_reject_bad_completion_before_archive(tmp_path, key, value):
    source, evidence, report = fixture(tmp_path)
    evidence[key] = value
    source.write_bytes(n.wire(evidence))
    with pytest.raises(ValueError):
        n.record(source, 'test fixture', tmp_path)
    assert not Path(str(report) + '.receipt.json').exists()
    assert not (tmp_path / '.local').exists()


@pytest.mark.parametrize('which', ['prompt', 'report', 'completion', 'archive_completion', 'log'])
def test_tamper_detected(tmp_path, which):
    source, evidence, report = fixture(tmp_path)
    receipt = n.record(source, 'test fixture', tmp_path)
    record = n.parse(receipt.read_bytes())
    archive = tmp_path / '.local/astra-improve/lanes/runs' / record['lane'] / record['run_id']
    target = {'prompt': Path(evidence['prompt']), 'report': report, 'completion': source,
              'archive_completion': archive / 'completion.json', 'log': archive / 'run.log'}[which]
    target.write_bytes(target.read_bytes() + b'\nchanged')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        n.verify(receipt, tmp_path)
    if which in ('prompt', 'report'):
        # Gate binds its archived prompt; source-prompt freshness is the adapter's extra check.
        if which == 'report':
            with pytest.raises(ValueError):
                gate.dispatch(receipt, gate.Inputs(tmp_path))


def test_no_overwrite_and_partial_archive_never_published(tmp_path, monkeypatch):
    source, _, _ = fixture(tmp_path)
    receipt = n.record(source, 'test fixture', tmp_path)
    original = receipt.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        n.record(source, 'test fixture', tmp_path)
    assert receipt.read_bytes() == original

    source2, _, report2 = fixture(tmp_path, 'native-astra-r1')
    real = n.exclusive
    def fail_mid_archive(path, data):
        if path.name == 'run.log':
            raise OSError('simulated interruption')
        real(path, data)
    monkeypatch.setattr(n, 'exclusive', fail_mid_archive)
    with pytest.raises(OSError):
        n.record(source2, 'test fixture', tmp_path)
    assert not Path(str(report2) + '.receipt.json').exists()
    assert list((tmp_path / '.local/astra-improve/lanes/runs/native-astra-r1').glob('*/report.md'))


def test_paths_duplicate_json_and_empty_inputs(tmp_path):
    source, evidence, report = fixture(tmp_path)
    evidence['report'] = str(tmp_path.parent / 'outside.md')
    source.write_bytes(n.wire(evidence))
    with pytest.raises(ValueError, match='outside project'):
        n.record(source, 'test fixture', tmp_path)
    with pytest.raises(ValueError, match='duplicate'):
        n.parse(b'{"host":"a","host":"b"}')
    source.write_bytes(b'')
    with pytest.raises(ValueError, match='empty'):
        n.record(source, 'test fixture', tmp_path)


def test_review_fail_is_completed_task_but_gate_refuses_verdict(tmp_path):
    source, _, report = fixture(tmp_path, 'native-astra-r1', 'VERDICT: FAIL\n')
    receipt = n.record(source, 'test fixture', tmp_path)
    assert gate.dispatch(receipt, gate.Inputs(tmp_path))[0] == 'native-astra-r1'
    with pytest.raises(ValueError, match='VERDICT: PASS'):
        gate.verdict(report.read_bytes())


def test_archived_native_identity_cannot_be_swapped(tmp_path):
    source, _, _ = fixture(tmp_path)
    receipt = n.record(source, 'test fixture', tmp_path)
    value = n.parse(receipt.read_bytes())
    value['agent_id'] = '/root/another_agent'
    receipt.write_bytes(n.wire(value))
    with pytest.raises(ValueError, match='archive receipt'):
        n.verify(receipt, tmp_path)
    with pytest.raises(ValueError, match='archive receipt'):
        gate.dispatch(receipt, gate.Inputs(tmp_path))
