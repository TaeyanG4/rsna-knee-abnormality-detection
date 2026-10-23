"""Test release boundaries without importing training or loading real data/models."""
import csv
import importlib.util
from pathlib import Path
import time

import pytest

spec = importlib.util.spec_from_file_location('recipe_release', Path(__file__).parents[1] / 'tools/exp089_recipe_release_v31.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fixture_comp(tmp_path):
    (tmp_path / 'test_series').mkdir()
    write_csv(tmp_path / 'test.csv', [release.UID], [['1.1'], ['1.2']])
    write_csv(tmp_path / 'sample_submission.csv', [release.UID, *release.TARGETS],
              [['1.2', *([0] * 12)], ['1.1', *([0] * 12)]])
    write_csv(tmp_path / 'test_series.csv', [release.UID, 'SeriesInstanceUID'], [['1.1', '2.1'], ['1.2', '2.2']])
    return tmp_path


def test_sample_order_and_dynamic_membership(tmp_path):
    comp = fixture_comp(tmp_path)
    assert release.test_contract(comp)['ids'] == ['1.2', '1.1']
    write_csv(comp / 'test.csv', [release.UID], [['1.1']])
    with pytest.raises(ValueError, match='membership'):
        release.test_contract(comp)


@pytest.mark.parametrize('rows', [
    [['1.1', *([.5] * 12)]],
    [['1.2', *([.5] * 12)], ['1.1', *([.5] * 12)]],
    [['1.1', *(['nan'] * 12)], ['1.2', *([.5] * 12)]],
    [['1.1', *([1.1] * 12)], ['1.2', *([.5] * 12)]],
])
def test_reject_partial_reordered_nonfinite_outofrange(tmp_path, rows):
    path = tmp_path / 'raw.csv'
    write_csv(path, [release.UID, *release.TARGETS], rows)
    with pytest.raises(ValueError):
        release.csv_check(path, ['1.1', '1.2'])


def test_time_screen_and_public_full_budgets():
    assert release.operational_plan(3)['internal_seconds'] == 600
    assert release.operational_plan(1300)['internal_seconds'] == 32100
    assert release.operational_plan(1300)['projected_seconds'] < 30600
    with pytest.raises(ValueError):
        release.operational_plan(10000)


@pytest.mark.parametrize('failure_call', [1, 2])
def test_metadata_failure_before_and_after_rename_rolls_back(tmp_path, failure_call):
    raw, final = tmp_path / 'raw.csv', tmp_path / 'submission.csv'
    raw.write_bytes(b'opaque synthetic probabilities')
    calls = []
    def check():
        calls.append(1)
        if len(calls) == failure_call:
            raise ValueError('test metadata changed')
    with pytest.raises(ValueError):
        release.publish(raw, final, release.sha(raw), time.monotonic() + 10, check)
    assert not final.exists() and not final.with_name(final.name + '.exp089-tmp').exists()
    assert raw.exists()


def test_deadline_failure_preserves_existing_and_removes_own_temp(tmp_path):
    raw, final = tmp_path / 'raw.csv', tmp_path / 'submission.csv'
    raw.write_bytes(b'new')
    with pytest.raises(ValueError):
        release.publish(raw, final, release.sha(raw), time.monotonic() - 1, lambda: None)
    assert not final.exists() and not final.with_name(final.name + '.exp089-tmp').exists()
    final.write_bytes(b'old')
    with pytest.raises(ValueError):
        release.publish(raw, final, release.sha(raw), time.monotonic() + 10, lambda: None)
    assert final.read_bytes() == b'old'


def test_natural_child_exit_required(tmp_path):
    # Reuse completed metadata only, never execute the prior worker again.
    import json
    source = Path(__file__).parents[1] / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-results-v2/B224-supervisor.json'
    s = json.loads(source.read_text())
    release.validate_resource(s, s['argv'], s['child_identity']['pid'], 900)
    s['cleanup']['actions'] = [{'pid': 45, 'action': 'kill'}]
    with pytest.raises(ValueError, match='cleanup'):
        release.validate_resource(s, s['argv'], s['child_identity']['pid'], 900)


def test_v2_output_budget_covers_large_hidden_cohort():
    # Completed T4 v2 receipts: ~25.2KB DICOM receipt + ~4.7KB compressed copy per study.
    assert release.OUTPUT >= 5000 * 30_000
    assert release.SCHEMA == 'exp089-test-release-v3.1'


def test_v3_page_cache_max_events_pass_but_oom_fails():
    import json, copy
    source = Path(__file__).parents[1] / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-results-v2/B224-supervisor.json'
    base = json.loads(source.read_text())
    s = copy.deepcopy(base)
    for key in ('cgroup_start', 'cgroup_end'):
        s[key]['values']['memory.events']['max'] = 12345
        s[key]['values']['memory.events']['high'] = 7
    s['cgroup_telemetry_complete'] = False
    s['cgroup_end']['errors'] = ['memory.stat unreadable']
    t = release.validate_resource(s, s['argv'], s['child_identity']['pid'], 900)
    assert t['snapshots']['cgroup_end']['memory.events']['max'] == 12345 and t['cgroup_telemetry_complete'] is False
    for kind in ('oom', 'oom_kill'):
        bad = copy.deepcopy(base); bad['cgroup_end']['values']['memory.events'][kind] = 1
        with pytest.raises(ValueError, match='OOM'):
            release.validate_resource(bad, bad['argv'], bad['child_identity']['pid'], 900)
    rss = copy.deepcopy(base); rss['sampled_tree_rss_max_bytes'] = release.RSS + 1
    with pytest.raises(ValueError, match='resource observation'):
        release.validate_resource(rss, rss['argv'], rss['child_identity']['pid'], 900)


def test_v31_second_arm_late_launch_budget():
    # E1: B224 finished ~9,000 s after the first-cell clock; R384 must still launch for N=1,300.
    plan, deadline, timeout = release.launch_budget(1000.0, 1000.0 + 9000.0, 1300, 30000.0)
    assert deadline == 1000.0 + release.TIME['full_seconds'] and timeout >= plan['projected_seconds']
    # The single-arm default still rejects a stale clock.
    with pytest.raises(ValueError, match='bootstrap'):
        release.launch_budget(1000.0, 1000.0 + 9000.0, 1300)
    # A launch so late that the arm no longer fits its conservative projection is refused before any work.
    with pytest.raises(ValueError, match='projection'):
        release.launch_budget(1000.0, 1000.0 + 25000.0, 1300, 30000.0)
    # Public3: no projection gate, 600 s internal deadline.
    plan, deadline, timeout = release.launch_budget(5.0, 65.0, 3)
    assert deadline == 5.0 + 600 and timeout > 0
    with pytest.raises(ValueError):
        release.launch_budget(5.0, 5.0 + 700.0, 3, 30000.0)


def test_v31_e1_public3_and_hidden_bootstrap_values():
    # Exactly what the E1 cell passes: 600 for N<=3 (Public3), 30000 otherwise.
    for delay in (5.0, 60.0, 300.0):
        release.launch_budget(1000.0, 1000.0 + delay, 3, 600.0)
    with pytest.raises(ValueError, match='bootstrap'):
        release.launch_budget(1000.0, 1005.0, 3, 30000.0)   # the value the cell must not use for Public3
    release.launch_budget(1000.0, 1000.0 + 60.0, 1300, 30000.0)
    release.launch_budget(1000.0, 1000.0 + 12000.0, 1760, 30000.0)
