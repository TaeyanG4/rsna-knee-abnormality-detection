"""EXP153 offline control tests. Tiny fake corpus and archive; no CUDA/network."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp153_h1_advance as m


def test_run_matrix_has_only_eight_missing_members():
    assert len(m.NEW) == 8
    assert len(m.ALL) == 10
    assert [m.tag(*x) for x in m.NEW] == [
        'EXP153_H1_S42_F2', 'EXP153_H1_S42_F3', 'EXP153_H1_S42_F4',
        *[f'EXP153_H1_S43_F{i}' for i in range(5)]]
    assert m.HEADS[-1] == 'aquila_v1'


def test_corpus_hash_gate_checks_every_member(tmp_path, monkeypatch):
    files = {'ids.npy': b'ids', 'masks.npy': b'masks', 'shard.npy': b'shard'}
    digest = lambda data: hashlib.sha256(data).hexdigest()
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    manifest = {'contract': 'raptor64_336_c140_s02', 'finished': True,
                'ids_sha256': digest(files['ids.npy']),
                'masks_sha256': digest(files['masks.npy']),
                'shards': [{'file': 'shard.npy', 'done': True,
                            'sha256': digest(files['shard.npy'])}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(m, 'CORPUS_SHA', m.sha256(tmp_path / 'manifest.json'))
    assert m.verify_corpus(tmp_path)['members_hashed'] == 3
    (tmp_path / 'shard.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='shard.npy'):
        m.verify_corpus(tmp_path)


def test_stage_archive_has_unique_entries_and_offline_queue(tmp_path, monkeypatch):
    root, out = tmp_path / 'root', tmp_path / 'stage'
    (root / m.MANIFEST).parent.mkdir(parents=True)
    (root / m.MANIFEST).write_text('{"tiny":true}', encoding='utf-8')
    (root / 'tiny.py').write_text('x = 1\n', encoding='utf-8')
    protocol = {'runs': {m.tag(s, f): {} for s, f in m.NEW}}
    monkeypatch.setattr(m, 'registration', lambda given=root: protocol)
    monkeypatch.setattr(m, 'package_files', lambda given=root: {
        'tiny.py': {'sha256': m.sha256(root / 'tiny.py'), 'bytes': (root / 'tiny.py').stat().st_size}})
    monkeypatch.setattr(m, 'runner_argv', lambda *a, **kw:
                        ['python', str(root / m.RUNNER), '--run', a[1]])
    result = m.stage(out, root)
    assert result['status'] == 'STAGED_NO_EXECUTION'
    with zipfile.ZipFile(out / 'exp153-code.zip.bin') as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert set(names) == {'tiny.py', 'config/workspace.toml', 'exp153-bundle.json'}
    queue = m.read(out / 'vast-queue.json')
    assert len(queue) == 8 and queue[0]['tag'] == 'EXP153_H1_S42_F2'
    assert all(not x['continue_on_failure'] for x in queue)
    assert all(queue[i]['after'] == queue[i - 1]['tag'] for i in range(1, 8))


def test_local_queue_is_only_json_and_chains_after(tmp_path, monkeypatch):
    protocol = {'runs': {m.tag(s, f): {} for s, f in m.NEW}}
    monkeypatch.setattr(m, 'registration', lambda root: protocol)
    monkeypatch.setattr(m, 'runner_argv', lambda *a, **kw: ['python', 'runner.py', a[1]])
    queue_path = tmp_path / 'queue.json'
    m.local_queue(queue_path, after='CURRENT_GPU_JOB', root=tmp_path)
    q = m.read(queue_path)
    assert len(q) == 8 and q[0]['after'] == 'CURRENT_GPU_JOB'
    assert q[1]['after'] == q[0]['tag']
    assert q[-1]['tag'] == 'EXP153_H1_S43_F4'
