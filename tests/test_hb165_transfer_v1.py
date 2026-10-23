"""CPU tests v1.0.0 for the H-B-165 small transfer / result return (no network, no remote)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import hb165_transfer_v1 as t
from tools import hb165_build_v1 as b

INVENTORY = b.WORK/'transfer-inventory.json'
REGISTERED = (b.WORK/'registration.json').is_file() and (b.WORK/'prepared.json').is_file()


@pytest.mark.skipif(not REGISTERED or not t.PARENT_INVENTORY.is_file(), reason='stage not registered/prepared')
def test_classify_ships_only_small_new_files():
    ship, expect, drift = t.classify()
    assert drift == [] and not set(ship) & set(expect)
    assert set(t.NEW_CODE) <= set(ship)
    assert {b.rel(b.ROOT/b.FOLD_SOURCE/n) for n in ('ordered-ids.json', 'group-metadata.csv')} <= set(ship)
    assert all(not p.startswith(t.FORBIDDEN) and v['bytes'] <= t.MAX_SHIP_BYTES for p, v in ship.items())
    assert sum(v['bytes'] for v in ship.values()) < 64 << 20
    for p in ('artifacts/experiments/EXP171_CONVNEXT_S_L2/initial-state.pt', 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv',
              '.local/scratch/slot-b/hb155/crop-cache-v220/manifest.json', '.local/scratch/astra-b3/H-B-147/native448/manifest.json',
              '.local/scratch/slot-b/hb155/pilot-v120/runtime-review.json', '.local/scratch/slot-b/hb155/pilot-v120/crop-review.json'):
        assert p in expect
    assert not any(p.endswith('.npy') and 'chunk' in p for p in ship)


@pytest.mark.skipif(not INVENTORY.is_file() or not (b.WORK/'transfer-files.txt').is_file(), reason='inventory not written')
def test_plan_is_strict_and_never_executes():
    v = t.plan('root@50.175.95.210', 51626)
    assert v['status'] == 'PLAN_ONLY'
    for cmd in (v['precheck'], v['receive'], v['verify']):
        assert cmd[:7] == ['ssh', '-p', '51626', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes']
    assert v['tar'][:5] == ['tar', '-cf', '-', '-T', str(b.WORK/'transfer-files.txt')]
    assert v['tar'][5:] == [b.rel(INVENTORY), b.rel(b.WORK/'transfer-files.txt'), b.rel(b.WORK/'SHA256SUMS')]
    m = b.read(INVENTORY)
    assert (b.WORK/'transfer-files.txt').read_text().split() == sorted(m['ship'])


@pytest.mark.parametrize('host,port', [('-oProxyCommand=x', 22), ('root@host;rm', 22), ('root@host', 0), ('root@host', 70000)])
def test_invalid_endpoint_rejected(host, port):
    with pytest.raises(ValueError):
        t.ssh_base(host, port)


def test_results_refuse_incomplete_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'WORK', tmp_path)
    for seed in b.SEEDS:
        for arm in b.ARMS:
            d = tmp_path/'fits'/b.member(arm, seed); d.mkdir(parents=True)
            (d/'heartbeat.json').write_text(json.dumps(dict(status='running')))
    with pytest.raises(ValueError, match='fit incomplete'):
        t.results()


def test_verify_results_detects_mismatch(tmp_path, monkeypatch):
    work = tmp_path  # --basetemp keeps tmp_path inside the project (rel() requires it)
    monkeypatch.setattr(b, 'WORK', work)
    f = work/'fits/a.csv'; f.parent.mkdir(); f.write_text('x')
    inv = dict(schema='hb165-results/v1', status='RESULTS_COMPLETE', shipped_unchanged={},
               files={b.rel(f): dict(bytes=1, sha256=hashlib.sha256(b'x').hexdigest())})
    (work/'result-inventory.json').write_text(json.dumps(inv))
    assert t.verify_results()['status'] == 'RESULTS_VERIFIED'
    f.write_text('y')
    with pytest.raises(ValueError, match='hash mismatch'):
        t.verify_results()


def test_receiver_script_checks_every_ship_and_expect_file(tmp_path):
    (tmp_path/'a').mkdir(); (tmp_path/'a/x.txt').write_bytes(b'abc'); (tmp_path/'y.bin').write_bytes(b'zz')
    m = dict(schema='hb165-transfer/v1', status='READY',
             ship={'a/x.txt': dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())},
             expect={'y.bin': dict(bytes=2, sha256=hashlib.sha256(b'zz').hexdigest())})
    (tmp_path/'inv.json').write_text(json.dumps(m))
    digest = hashlib.sha256((tmp_path/'inv.json').read_bytes()).hexdigest()
    ok = subprocess.run([sys.executable, '-c', t.VERIFY, str(tmp_path), 'inv.json', digest], capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(ok.stdout)['checked'] == 2
    (tmp_path/'y.bin').write_bytes(b'zZ')
    bad = subprocess.run([sys.executable, '-c', t.VERIFY, str(tmp_path), 'inv.json', digest], capture_output=True, text=True)
    assert bad.returncode == 1 and json.loads(bad.stdout)['bad'] == ['expect:y.bin']
    wrong = subprocess.run([sys.executable, '-c', t.VERIFY, str(tmp_path), 'inv.json', '0'*64], capture_output=True, text=True)
    assert wrong.returncode != 0


def test_cli_without_execute_is_plan_only(capsys):
    if not INVENTORY.is_file():
        pytest.skip('inventory not written')
    assert t.main(['send', '--host', 'root@50.175.95.210', '--port', '51626']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'PLAN_ONLY'
