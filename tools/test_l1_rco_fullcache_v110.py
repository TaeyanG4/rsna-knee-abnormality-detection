"""CPU tests for full-cache v1.1.0; no GPU or full-cache execution."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pytest
from tools import l1_rco_fullcache_v110 as c


def shard(out, reg, uid, offset=0):
    p = out / 'shards' / c.key(uid); p.mkdir(parents=True)
    px = np.full((2, 384, 384), offset, np.uint8)
    meta = np.arange(24, dtype=np.float32).reshape(2, 12)
    np.save(p / 'px.npy', px); np.save(p / 'meta.npy', meta)
    rec = {'contract_id': reg['contract_id'], 'uid': uid, 'rows': 2,
           'blocks': [{'views': [[0, 0], [1, 1]]}], 'cuda_initialized': False,
           'hashes': {name: c.sha(p / name) for name in c.FILES}}
    c.write(p / 'receipt.json', rec)
    return p


def test_no_labels_in_full_registration_inputs(monkeypatch):
    original = c.sha
    def guarded(path):
        assert Path(path).resolve() != (c.ROOT / c.s.TARGETS).resolve()
        return original(path)
    monkeypatch.setattr(c, 'sha', guarded)
    pins = c.pins()
    assert c.s.TARGETS not in pins
    c.verify_pins(pins)
    assert c.VERSION == '1.1.0'
    assert c.s.VERSION == '1.0.0'


def test_path_and_cleanup_guards(tmp_path):
    with pytest.raises(ValueError): c.safe_output(tmp_path)
    with pytest.raises(ValueError): c.clean_owned(c.WORK)
    with pytest.raises(ValueError): c.clean_owned(c.WORK / 'workers')
    assert c.safe_output(c.OUT) == c.OUT.resolve()


def test_disk_concurrent_and_assembly_margins(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'WORK', tmp_path)
    with patch.object(c.shutil, 'disk_usage', return_value=SimpleNamespace(free=47*c.GIB)):
        with pytest.raises(ValueError, match='combined'): c.disk_gate(tmp_path, 6)
    with patch.object(c.shutil, 'disk_usage', return_value=SimpleNamespace(free=48*c.GIB)):
        c.disk_gate(tmp_path, 6)
    with patch.object(c.shutil, 'disk_usage', return_value=SimpleNamespace(free=3*c.GIB)):
        with pytest.raises(ValueError): c.disk_gate(tmp_path, 0, 2*c.GIB)


def test_checksum_identity_uncommitted_resume(tmp_path):
    reg = {'contract_id': 'sealed'}
    assert c.validate_shard(reg, tmp_path, 'a') is None
    p = shard(tmp_path, reg, 'a')
    assert c.validate_shard(reg, tmp_path, 'a')['rows'] == 2
    with pytest.raises(ValueError): c.validate_shard({'contract_id': 'other'}, tmp_path, 'a')
    with (p / 'px.npy').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'): c.validate_shard(reg, tmp_path, 'a')


def test_shard_invalid_offset_and_nan(tmp_path):
    reg = {'contract_id': 'sealed'}; p = shard(tmp_path, reg, 'a')
    r = c.read(p / 'receipt.json'); r['blocks'][0]['views'][0][1] = 2
    c.write(p / 'receipt.json', r)
    with pytest.raises(ValueError, match='offset'): c.validate_shard(reg, tmp_path, 'a')
    r['blocks'][0]['views'][0][1] = 0
    np.save(p / 'meta.npy', np.full((2, 12), np.nan, np.float32))
    r['hashes']['meta.npy'] = c.sha(p / 'meta.npy'); c.write(p / 'receipt.json', r)
    with pytest.raises(ValueError, match='layout'): c.validate_shard(reg, tmp_path, 'a')


def test_assembly_order_offsets_restart(tmp_path):
    reg = {'contract_id': 'sealed'}
    shard(tmp_path, reg, 'a', 7); shard(tmp_path, reg, 'b', 9)
    with patch.object(c, 'disk_gate'):
        c.assemble(reg, tmp_path, ['b', 'a'])
        first = {name: c.sha(tmp_path / name) for name in c.LAYOUT}
        # A previous interrupted assembly can leave an uncommitted temp file.
        (tmp_path / 'px.u8.npy.tmp').write_bytes(b'interrupted')
        c.assemble(reg, tmp_path, ['b', 'a'])
    assert first == {name: c.sha(tmp_path / name) for name in c.LAYOUT}
    index = c.read(tmp_path / 'index.json')
    assert index['order'] == ['b', 'a'] and index['done']['a'] == [2, 2, 1]
    assert index['blocks']['a'][0]['views'] == [[0, 2], [1, 3]]
    px = np.load(tmp_path / 'px.u8.npy', mmap_mode='r')
    assert (px[:2] == 9).all() and (px[2:] == 7).all()


def test_assembly_missing_shard_blocks(tmp_path):
    with pytest.raises(ValueError, match='committed'): c.assemble({'contract_id': 'x'}, tmp_path, ['missing'])


def test_worker_limit_and_duplicate_roster(tmp_path):
    reg = {'train': ['a'], 'holdout': ['b'], 'contract_id': 'x'}
    with pytest.raises(ValueError, match='workers'): c.build(reg, tmp_path, ['a'], 7)
    with patch.object(c, 'validate'), patch.object(c, 'safe_output', return_value=tmp_path):
        with pytest.raises(ValueError, match='roster'): c.build(reg, tmp_path, ['a', 'a'], 1)


def test_ownership_lock(tmp_path):
    with c.file_lock(tmp_path / 'claim', blocking=False):
        with pytest.raises(BlockingIOError):
            with c.file_lock(tmp_path / 'claim', blocking=False): pass


def test_serial_reference_obeys_shared_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'WORK', tmp_path)
    with patch.object(c, '_serial_reference_locked') as render:
        with c.file_lock(tmp_path / 'fullcache.lock', blocking=False):
            with pytest.raises(BlockingIOError): c.serial_reference([], tmp_path)
        render.assert_not_called()
        c.serial_reference([], tmp_path)
        render.assert_called_once()


def test_full_admission_requires_k0_and_independent_exact_review(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c.s, 'WORK', tmp_path)
    monkeypatch.setattr(c, 'WORK', tmp_path)
    c.write(tmp_path / 'registration.json', {'contract_id': 'source'})
    c.write(tmp_path / 'proof.json', {'status': 'PASS', 'contract_id': 'full'})
    reg = {'contract_id': 'full'}
    k0, rev, report = tmp_path / 'k0.json', tmp_path / 'review.json', tmp_path / 'review.md'
    report.write_text('independent PASS')
    result = {'contract_id': 'source', 'K0_numeric_gate': 'PASS'}
    verdict = {'verdict': 'PASS', 'builder': 'l1-rco', 'reviewer': 'other', 'contract_id': 'full',
               'K0_result_sha256': '', 'report': 'review.md', 'report_sha256': c.sha(report),
               'proof_sha256': c.sha(tmp_path / 'proof.json')}
    c.write(k0, result); verdict['K0_result_sha256'] = c.sha(k0); c.write(rev, verdict)
    with pytest.raises(ValueError): c.authorize_full(reg, None, None)
    with patch('tools.l1_rco_runtime_v1.go_rule', return_value=True):
        c.authorize_full(reg, k0, rev)
        for field, value in [('verdict', 'FAIL'), ('reviewer', 'l1-rco'), ('contract_id', 'stale'),
                             ('K0_result_sha256', 'bad'), ('report_sha256', 'bad'), ('proof_sha256', 'bad')]:
            bad = copy.deepcopy(verdict); bad[field] = value; c.write(rev, bad)
            with pytest.raises(ValueError): c.authorize_full(reg, k0, rev)
        c.write(rev, verdict)
        c.write(tmp_path / 'proof.json', {'status': 'PASS', 'contract_id': 'full', 'edited': True})
        with pytest.raises(ValueError, match='proof hash'): c.authorize_full(reg, k0, rev)
        c.write(rev, verdict); result['contract_id'] = 'wrong'; c.write(k0, result)
        with pytest.raises(ValueError, match='source registration'): c.authorize_full(reg, k0, rev)


def test_progress_failure_is_nonfatal(tmp_path):
    with patch.object(c, 'write', side_effect=PermissionError('locked')):
        c.progress(tmp_path, 'render', 2, 6, c.time.perf_counter(), 2)
