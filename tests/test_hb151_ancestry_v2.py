import hashlib
import json

import pytest

from tools import hb151_ancestry_v2 as a


@pytest.mark.parametrize('replacement', [b'older baseline', b'partial'])
def test_rejects_changed_baseline(tmp_path, replacement):
    path = tmp_path/'baseline.json'
    path.write_bytes(b'authenticated baseline')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert a.verify_file(tmp_path, path.name, {'sha256': digest})[1] == digest
    path.write_bytes(replacement)
    with pytest.raises(ValueError, match='hash drift'):
        a.verify_file(tmp_path, path.name, digest)


def test_path_escape_rejected_before_read(tmp_path):
    with pytest.raises(ValueError, match='outside project'):
        a.verify_file(tmp_path, '../unread.json')


def positive():
    return [[[.004]*12 for _ in range(4)] for _ in range(2)]


def test_ancestry_fail_closed():
    assert a.ancestry_eligible(overlap=0, complete=True, calibrated=True)
    for patch in [dict(overlap=1), dict(complete=False), dict(complete=None),
                  dict(calibrated=False), dict(ranks=True)]:
        args = dict(overlap=0, complete=True, calibrated=True)
        args.update(patch)
        assert not a.ancestry_eligible(**args)


def test_no_ancestry_bypass():
    assert not a.decision(positive())['go']
    assert a.decision(positive(), ancestry_pass=True)['go']


@pytest.mark.parametrize('kind', ['macro', 'seed', 'weak', 'target'])
def test_violating_delta_blocks_go(kind):
    x = positive()
    if kind == 'macro':
        x = [[[.001]*12 for _ in range(4)] for _ in range(2)]
    elif kind == 'seed':
        x[0] = [[0.]*12 for _ in range(4)]
    elif kind == 'weak':
        for i in a.WEAK:
            x[0][0][i] = -.001
    else:
        x[0][0][0] = -.00301
    assert not a.decision(x, ancestry_pass=True)['go']


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_invalid_values(value):
    x = positive(); x[0][0][0] = value
    with pytest.raises(ValueError, match='nonfinite'):
        a.decision(x, ancestry_pass=True)


def test_bad_shape():
    with pytest.raises(ValueError, match='seed/ruler'):
        a.decision([])


def test_windows_retry(tmp_path, monkeypatch):
    original = a.Path.replace
    calls = []
    def replace(self, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError('sharing violation')
        return original(self, target)
    monkeypatch.setattr(a.Path, 'replace', replace)
    monkeypatch.setattr(a.time, 'sleep', lambda _: None)
    p = tmp_path/'report.json'
    a.write_json(p, {'status': 'STOP'})
    assert json.loads(p.read_text())['status'] == 'STOP'
    assert len(calls) == 3


def test_persistent_write_failure(tmp_path, monkeypatch):
    def denied(*args):
        raise PermissionError('denied')
    monkeypatch.setattr(a.Path, 'replace', denied)
    monkeypatch.setattr(a.time, 'sleep', lambda _: None)
    with pytest.raises(PermissionError):
        a.write_json(tmp_path/'report.json', {})
