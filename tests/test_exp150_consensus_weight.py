"""CPU-only EXP150 engineering tests. No corpus, network or CUDA context."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest

from tools import exp150_consensus_weight as e


def test_unanimity_disagreement_missing_and_threshold():
    a = [np.array([[.49, .50, np.nan, .8]], float),
         np.array([[.1, .51, .3, .8]], float),
         np.array([[.4, .4, .2, .8]], float),
         np.array([[.2, .7, .1, .8]], float)]
    np.testing.assert_allclose(e.weight_matrix(a), [[1., .4, .4, 1.]])


def test_load_weights_checks_hash_and_values(tmp_path):
    path = tmp_path / 'weights.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([e.UID, *e.T])
        w.writerow(['u1', *([1.] * 12)])
        w.writerow(['u2', *([.4] * 12)])
    protocol = {'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'roster': {'train': 1, 'val': 1}}
    got = e.load_weights(path, ['u2'], tmp_path, protocol)
    assert got['u2'].tolist() == pytest.approx([.4] * 12)
    with path.open('a', encoding='utf-8') as f:
        f.write('tamper\n')
    with pytest.raises(ValueError, match='hash'):
        e.load_weights(path, ['u2'], tmp_path, protocol)


def test_safe_package_paths():
    assert str(e.safe('training/src/raptor_common.py')) == 'training/src/raptor_common.py'
    for name in ('../secret', '/absolute', 'C:/secret', 'a\\b'):
        with pytest.raises(ValueError):
            e.safe(name)


def test_vast_corpus_reuse_requires_local_file_hashes_and_sizes(tmp_path):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    records = []
    for name in ('ids.npy', 'masks.npy', 'vols_shard00.npy'):
        path = corpus / name
        path.write_bytes(name.encode())
        records.append(dict(file=name, bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(contract='raptor64_336_c140_s02', finished=True,
                    ids_sha256=records[0]['sha256'], masks_sha256=records[1]['sha256'],
                    shards=[dict(file=records[2]['file'], bytes=records[2]['bytes'],
                                 sha256=records[2]['sha256'], done=True)])
    source = corpus / 'manifest.json'
    source.write_text(json.dumps(manifest), encoding='utf-8')
    local = tmp_path / 'registered-manifest.json'
    local.write_bytes(source.read_bytes())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert e.verify_corpus(corpus, local, records, digest)['files'] == 3
    (corpus / 'vols_shard00.npy').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        e.verify_corpus(corpus, local, records, digest)


def test_cpu_fake_validation():
    assert e.validate_fake()['status'] == 'VALIDATED_CPU_FAKES'


def test_one_cpu_optimizer_step():
    torch = pytest.importorskip('torch')
    assert not torch.cuda.is_initialized()
    assert e.cpu_smoke()['status'] == 'ONE_CPU_STEP'
    assert not torch.cuda.is_initialized()
