"""CPU-only checks of the diagnostic-only quarantine (tools/diagnostic_quarantine.py, EXP116 v1.1.1, D-B-024) and of the two consumers that
accept arbitrary prediction CSVs: tools/g1_compare.py (ruler) and tools/exp103_distill_labels.py (teacher). A file whose directory, or a parent
directory below the workspace root, carries DIAGNOSTIC_ONLY.json or a result.json / manifest.json with decision_eligible false is refused
before anything is read or any output directory is created. Synthetic files only (no competition rows are read or printed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnostic_quarantine as dq  # noqa: E402
from tools import exp103_distill_labels as distill  # noqa: E402
from tools import g1_compare as g1  # noqa: E402

T = list(g1.T)
UID = 'StudyInstanceUID'
B_REF = ROOT / g1.REF['B']


def write_preds(path, uids, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pd.DataFrame({UID: uids, **{t: rng.uniform(0.05, 0.95, len(uids)) for t in T}}).to_csv(path, index=False)
    return path


def marked(tmp_path, where='dir'):
    """A prediction CSV quarantined by: its own directory's marker ('dir'), a parent experiment directory's marker ('parent'), an
    assembly result.json ('result') or a registration manifest.json ('manifest') with decision_eligible false."""
    exp = tmp_path / 'EXP_LEAKY'
    csv = write_preds(exp / ('F0' if where == 'parent' else '.') / 'oof_k36.csv', ['u1', 'u2', 'u3'])
    if where in ('dir', 'parent'):
        dq.write_marker(exp, dq.marker_body('EXPTEST', 'leaky by construction', ['D-B-024'], tag='EXP_LEAKY'))
    elif where == 'result':
        (exp / 'result.json').write_text(json.dumps(dict(schema='x-oof-v1', decision_eligible=False, leak='leaky (D-B-024)')), encoding='utf-8')
    else:
        (exp / 'manifest.json').write_text(json.dumps(dict(schema='x', evaluation=dict(decision_eligible=False))), encoding='utf-8')
    return csv


# ------------------------------------------------------------------ the rule
@pytest.mark.parametrize('where', ['dir', 'parent', 'result', 'manifest'])
def test_quarantine_reason_finds_markers_and_ineligible_records(tmp_path, where):
    csv = marked(tmp_path, where)
    reason = dq.quarantine_reason(csv)
    assert reason and ('decision_eligible false' in reason)
    with pytest.raises(dq.DiagnosticOnlyInput, match='refusing') as e:
        dq.require_decision_eligible(csv, 'consumer-x')
    assert isinstance(e.value, SystemExit) and 'consumer-x' in str(e.value) and 'D-B-024' in str(e.value)


def test_clean_files_pass_and_the_walk_stops_at_the_workspace_root(tmp_path):
    clean = write_preds(tmp_path / 'EXP_OK' / 'oof_k36.csv', ['u1', 'u2'])
    (tmp_path / 'EXP_OK' / 'result.json').write_text(json.dumps(dict(decision_eligible=True)), encoding='utf-8')
    assert dq.quarantine_reason(clean) is None and dq.require_decision_eligible(clean, 'x') == clean.resolve()
    dq.write_marker(tmp_path, dq.marker_body('EXPTEST', 'above the root', ['D-B-024']))
    assert dq.quarantine_reason(clean, root=tmp_path) is None  # the root itself and its ancestors are not experiment directories
    assert 'above the root' in dq.quarantine_reason(clean)  # below the workspace root every ancestor of the file's directory is checked


def test_unreadable_record_fails_closed_and_markers_are_never_overwritten(tmp_path):
    csv = write_preds(tmp_path / 'EXP_BAD' / 'oof.csv', ['u1'])
    (tmp_path / 'EXP_BAD' / 'result.json').write_text('{not json', encoding='utf-8')
    assert 'unverifiable' in dq.quarantine_reason(csv)
    body = dq.marker_body('EXPTEST', 'r', ['D-B-024'])
    dq.write_marker(tmp_path / 'EXP_BAD', body)
    with pytest.raises(FileExistsError):
        dq.write_marker(tmp_path / 'EXP_BAD', body)
    with pytest.raises(ValueError):
        dq.write_marker(tmp_path, dict(body, decision_eligible=True))
    with pytest.raises(ValueError):
        dq.update_marker(tmp_path / 'EXP_BAD', decision_eligible=True)
    assert dq.update_marker(tmp_path / 'EXP_BAD', outputs={'oof.csv': 'abc'})['decision_eligible'] is False


def test_existing_prediction_files_are_not_quarantined():
    assert dq.quarantine_reason(B_REF) is None
    for d in ('EXP099_B8_DREAD_OOF', 'EXP107_RADR50_DREAD_OOF'):
        assert dq.quarantine_reason(ROOT / 'artifacts' / 'experiments' / d / 'oof_k36.csv') is None


def test_cli_check_exit_codes(tmp_path, capsys):
    csv = marked(tmp_path)
    assert dq.main(['check', str(csv)]) == 3 and '"decision_eligible": false' in capsys.readouterr().out
    assert dq.main(['check', str(write_preds(tmp_path / 'ok' / 'p.csv', ['u1']))]) == 0


# ------------------------------------------------------------------ tools/g1_compare.py
@pytest.mark.parametrize('where', ['dir', 'parent', 'result'])
def test_g1_compare_refuses_a_diagnostic_only_candidate(tmp_path, monkeypatch, where):
    csv = marked(tmp_path, where)
    out = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['g1_compare.py', '--out', str(out), '--candidate', f'leaky={csv}', '--contrast', 'leaky-B'])
    monkeypatch.setattr(g1.pd, 'read_csv', lambda *a, **k: pytest.fail('read before the quarantine check'))
    with pytest.raises(dq.DiagnosticOnlyInput, match='tools/g1_compare.py: refusing') as e:
        g1.main()
    assert str(csv) in str(e.value) and not out.exists()


def test_g1_compare_refuses_a_quarantined_reference(tmp_path, monkeypatch):
    csv = marked(tmp_path)
    monkeypatch.setitem(g1.REF, 'B', str(csv))
    monkeypatch.setattr(sys, 'argv', ['g1_compare.py', '--out', str(tmp_path / 'r')])
    with pytest.raises(dq.DiagnosticOnlyInput, match='refusing'):
        g1.main()
    assert not (tmp_path / 'r').exists()


@pytest.mark.skipif(not B_REF.exists() or not (ROOT / g1.LABELS).exists() or not (ROOT / g1.META).exists(), reason='G1 reference files absent')
def test_g1_compare_still_runs_on_an_eligible_candidate(tmp_path, monkeypatch):
    cand = tmp_path / 'EXP_OK' / 'oof_best.csv'
    cand.parent.mkdir()
    cand.write_bytes(B_REF.read_bytes())
    out = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['g1_compare.py', '--out', str(out), '--candidate', f'c={cand}', '--contrast', 'c-B', '--reps', '3'])
    g1.main()
    r = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    assert r['contrasts']['c-B']['macro12'] == 0.0 and set(r['files']) == {'B', 'c'}


# ------------------------------------------------------------------ tools/exp103_distill_labels.py
@pytest.mark.parametrize('where', ['dir', 'parent', 'result', 'manifest'])
def test_distill_refuses_a_diagnostic_only_teacher(tmp_path, monkeypatch, where):
    leaky = marked(tmp_path, where)
    clean = write_preds(tmp_path / 'EXP_OK' / 'oof_k36.csv', ['u1', 'u2', 'u3'])
    for name in ('DREAD', 'TRAIN', 'G1'):  # nothing may be read before the check
        monkeypatch.setattr(distill, name, tmp_path / 'absent' / name)
    out = tmp_path / 'distill_out'
    with pytest.raises(dq.DiagnosticOnlyInput, match='tools/exp103_distill_labels.py: refusing') as e:
        distill.build({'ok': str(clean), 'leaky': str(leaky)}, out)
    assert str(leaky) in str(e.value) and not out.exists()


def test_distill_still_builds_from_eligible_teachers(tmp_path, monkeypatch):
    uids = [f'u{i}' for i in range(6)]
    g1_ids = uids[:4]
    dread = tmp_path / 'dread.csv'
    pd.DataFrame({UID: uids, **{t: np.linspace(0.1, 0.9, 6) for t in distill.TARGETS}}).to_csv(dread, index=False)
    train = tmp_path / 'train.csv'
    pd.DataFrame({UID: ['g1', 'g2'], 'ACL': [np.nan, np.nan]}).to_csv(train, index=False)  # synthetic: no gold rows
    ordered = tmp_path / 'ordered-ids.json'
    ordered.write_text(json.dumps(dict(train=g1_ids[:3], val=g1_ids[3:])), encoding='utf-8')
    monkeypatch.setattr(distill, 'DREAD', dread); monkeypatch.setattr(distill, 'TRAIN', train); monkeypatch.setattr(distill, 'G1', ordered)
    teacher = write_preds(tmp_path / 'EXP_OK' / 'oof_k36.csv', g1_ids, seed=1)
    out = tmp_path / 'distill_out'
    distill.build({'ok': str(teacher)}, out)
    mixed = pd.read_csv(out / 'distill_v1.csv')
    assert len(mixed) == 6 and json.loads((out / 'receipt.json').read_text(encoding='utf-8'))['covered'] == 4
