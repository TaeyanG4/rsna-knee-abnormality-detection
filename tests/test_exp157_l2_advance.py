"""EXP157 CPU-only queue, registration and wrapper checks with tiny fakes."""
import hashlib
import json
from pathlib import Path

import pytest

from tools import exp157_l2_advance as m
from training.runners import exp157_l2 as runner


def test_ten_member_line_reuses_exactly_two_exp149_runs():
    assert len(m.NEW) == 8 and len(m.ALL) == 10
    assert [m.tag(*x) for x in m.NEW] == [
        'EXP157_L2_S42_F2', 'EXP157_L2_S42_F3', 'EXP157_L2_S42_F4',
        *[f'EXP157_L2_S43_F{i}' for i in range(5)]]
    assert [m.ensemble_tag(42, f) for f in range(2)] == [
        'EXP149_L2_S42_F0', 'EXP149_L2_S42_F1']
    assert m.control_tag(43, 0) == 'EXP127_AUG1_S43_F0'
    assert m.control_tag(43, 4) == 'EXP134_AUG1_S43_F4'


def test_saved_state_argument_is_exactly_once_and_is_removed():
    state, rest = runner.split_initial_state([
        '--run', m.tag(43, 2), '--initial-state', 'saved.pt', '--mode', 'full'])
    assert state == Path('saved.pt')
    assert rest == ['--run', m.tag(43, 2), '--mode', 'full']
    with pytest.raises(ValueError, match='Exactly one'):
        runner.split_initial_state(['--run', 'x'])
    with pytest.raises(ValueError, match='Exactly one'):
        runner.split_initial_state(['--initial-state', 'a', '--initial-state', 'b'])


def test_registration_detects_initial_state_tamper(tmp_path, monkeypatch):
    root = tmp_path
    for name, content in ((m.SOURCE, b'{}'), (m.PROTOCOL, b'protocol'),
                          (m.LABEL, b'labels'), (m.INITIAL, b'state'),
                          ('tools/new.py', b'new')):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
    sha = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    protocol = {'schema': 'exp157-l2-advance-v1', 'version': m.VERSION,
                'runs': {m.tag(s, f): {} for s, f in m.NEW},
                'exp157': {'source_manifest_sha256': sha(m.SOURCE),
                           'protocol_sha256': sha(m.PROTOCOL),
                           'initial_state_file_sha256': sha(m.INITIAL),
                           'initial_state_tensor_sha256': m.INITIAL_TENSOR_SHA,
                           'label_file_sha256': sha(m.LABEL)},
                'source_bindings': {'tools/new.py': sha('tools/new.py')}}
    file = root / m.MANIFEST
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(protocol), encoding='utf-8')
    assert m.registration(root)['version'] == m.VERSION
    (root / m.INITIAL).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='initial state differs'):
        m.registration(root)


def test_two_seed_lanes_chain_serially_and_smoke_does_not_launch(tmp_path, monkeypatch):
    p = {'runs': {m.tag(s, f): {} for s, f in m.NEW}}
    monkeypatch.setattr(m, 'registration', lambda root: p)
    monkeypatch.setattr(m, 'runner_argv', lambda p, run, *a, **k:
                        ['python', 'runner.py', '--run', run])
    a = tmp_path / 's42.json'
    b = tmp_path / 's43.json'
    first = m.local_queue(a, after='EXP155_OWN0949_OAI_S42v1_1',
                          lane='s42', root=tmp_path)
    second = m.local_queue(b, after=first['last_tag'], lane='s43', root=tmp_path)
    q1, q2 = m.read(a), m.read(b)
    assert len(q1) == 3 and len(q2) == 5
    assert q1[0]['after'] == 'EXP155_OWN0949_OAI_S42v1_1'
    assert q2[0]['after'] == q1[-1]['tag']
    assert all(not x['continue_on_failure'] for x in q1 + q2)
    assert all(q[i]['after'] == q[i - 1]['tag'] for q in (q1, q2) for i in range(1, len(q)))
    smoke = tmp_path / 'smoke.json'
    result = m.gpu_smoke(smoke, after='EXP155_OWN0949_OAI_S42v1_1', root=tmp_path)
    assert result['status'] == 'SMOKE_PLAN_NO_EXECUTION'
    assert m.read(smoke)[0]['after'] == 'EXP155_OWN0949_OAI_S42v1_1'
    assert not (tmp_path / '.local/exp157-l2-smoke').exists()
