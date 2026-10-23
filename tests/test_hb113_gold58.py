"""CPU tests for the paired Gold diagnostic; no label rows are emitted."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from tools import hb113_gold58 as h


def test_weighted_tied_auc_matches_expanded_reference():
    p = np.array([.2, .2, .7, .9, .9])
    y = np.array([0, 1, 0, 1, 1])
    weights = np.array([[1, 1, 1, 1, 1], [2, 0, 3, 1, 2]])
    actual = h.lens.auc_draws(p[:, None], y[:, None], np.ones((5, 1), bool), weights)
    for i, w in enumerate(weights):
        ix = np.repeat(np.arange(5), w)
        assert actual[i, 0] == pytest.approx(roc_auc_score(y[ix], p[ix]))


def test_paired_fixed_targets_and_zero_delta():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, (58, 12))
    p = rng.random((58, 12))
    result = h.paired_stats([str(i) for i in range(58)], p, p, y)
    assert result['macro12']['delta_member_minus_parent']['observed'] == 0
    assert result['weak_four']['delta_member_minus_parent']['ci95'] == [0, 0]
    y[:, 0] = 0
    result = h.paired_stats([str(i) for i in range(58)], p, p, y)
    assert result['macro12']['parent']['observed'] is None
    assert result['macro12']['parent']['valid_draws'] == 0
    assert result['weak_four']['parent']['observed'] is not None


def test_paired_direction_is_member_minus_parent():
    y = np.tile(np.array([0, 1] * 29)[:, None], (1, 12))
    result = h.paired_stats([str(i) for i in range(58)], 1-y, y, y)
    assert result['macro12']['delta_member_minus_parent']['observed'] == 1


def test_probabilities_fail_closed():
    data = {n+s: np.full((2, 12), .5) for n in ('parent', 'member') for s in ('', '_plain', '_mirror')}
    h.check_predictions(data, 2)
    data['member'][0, 0] = .6
    with pytest.raises(ValueError, match='mean mismatch'):
        h.check_predictions(data, 2)
    data['member'][0, 0] = np.nan
    with pytest.raises(ValueError, match='invalid paired'):
        h.check_predictions(data, 2)


def test_source_and_output_drift_rejected(tmp_path):
    file = tmp_path / 'pin'; file.write_bytes(b'x')
    with pytest.raises(ValueError):
        h.p956.check_pin(file, '0'*64)
    with pytest.raises(ValueError, match='scratch'):
        h.scratch(h.ROOT / 'unrelated')
    with pytest.raises(ValueError, match='duplicate'):
        h.paired_stats(['a', 'a'], np.zeros((2, 12)), np.zeros((2, 12)), np.zeros((2, 12)))


def test_real_p956_windows_and_anatomical_mirror():
    import torch
    torch.set_num_threads(2)
    module = h.runtime()
    assert module.INPUT_RES == 320 and module.K_EVAL == 94 and module.IMG == 384
    vol = np.broadcast_to(np.arange(384, dtype=np.uint16)[None, None, :] % 256, (96, 384, 384)).astype('uint8').copy()
    vol = (vol + np.arange(96, dtype='uint8')[:, None, None]).astype('uint8')
    mask = np.ones(96, 'uint8')
    for mirror in [False, True]:
        assert torch.equal(module.gpu_windows(vol, mask, 94, 320, 'cpu', mirror=mirror),
                           module.eval_windows(vol, mask, 94, 320, mirror=mirror))
    tri = torch.arange(2*3*4*4).reshape(2, 3, 4, 4)
    actual = module._mirror_tri(tri, [10, 70], 96)
    assert torch.equal(actual[0], tri[0].flip(0))
    assert torch.equal(actual[1], tri[1].flip(-1))
    with pytest.raises(AssertionError, match='Qualified inference requires CUDA'):
        module.infer_probs(None, None, 'cpu')


def test_windows_binary_positional_reads(tmp_path):
    module = h.runtime()
    reader = h.fastread(module)
    data = bytes(range(256))*20
    path = tmp_path / 'bytes.bin'; path.write_bytes(data)
    fd = reader._fopen(str(path))
    try:
        offsets = [12, 26, 255, 502, 1900]*10
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda i: reader._pread(fd, 97, i), offsets))
        assert values == [data[i:i+97] for i in offsets]
    finally:
        os.close(fd)


def test_no_cpu_full_or_cuda_smoke(monkeypatch):
    monkeypatch.setattr(h, 'validate', lambda: {})
    for device, smoke in [('cpu', False), ('cuda', True)]:
        with pytest.raises(ValueError, match='full check requires CUDA'):
            h.infer(device, h.SCRATCH / 'cpu-smoke', h.ROOT, smoke)


def test_exposure_caveat():
    assert 'report labels' in h.CAVEAT
    assert 'Forgetting biases' in h.CAVEAT
    assert 'not honest OOF' in h.CAVEAT


def test_scoring_receipts_alignment_and_smoke_separation(tmp_path, monkeypatch):
    import json
    stage = tmp_path / 'stage'; stage.mkdir()
    (stage / 'registration.json').write_text('{}')
    output = tmp_path / 'scratch' / 'pair'; output.mkdir(parents=True)
    ids = [str(i) for i in range(58)]
    y = np.tile(np.array([0, 1] * 29)[:, None], (1, 12))
    monkeypatch.setattr(h, 'STAGE', stage)
    monkeypatch.setattr(h, 'SCRATCH', tmp_path / 'scratch')
    monkeypatch.setattr(h, 'gold', lambda: (ids, y))
    monkeypatch.setattr(h, 'validate', lambda: {'bootstrap': {}})
    arrays = {n+s: np.full((2, 12), .5) for n in ('parent', 'member') for s in ('', '_plain', '_mirror')}
    def save(order):
        np.savez_compressed(output / 'predictions.npz', ids=order, targets=h.LABELS, **arrays)
        h.write(output / 'receipt.json', dict(status='CPU_SMOKE_ONLY', caveat=h.CAVEAT,
                 binding_sha256=h.sha(stage / 'registration.json'), prediction_sha256=h.sha(output / 'predictions.npz'),
                 studies=2, inputs=[{}, {}], device='cpu'))
    save(ids[:2])
    assert h.score(output, smoke=True)['studies'] == 2
    result = json.loads((output / 'score.json').read_text())
    assert result['caveat'] == h.CAVEAT and len(result['targets']) == 12
    with pytest.raises(ValueError, match='smoke/full'):
        h.score(output)
    save(ids[:2][::-1])
    with pytest.raises(ValueError, match='roster/target'):
        h.score(output, smoke=True)
    save(ids[:2])
    with (output / 'predictions.npz').open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        h.score(output, smoke=True)


def test_archive_adapter_bytes_and_closed_handles(tmp_path):
    import zipfile
    path = tmp_path / 'dicom.zip'
    data = bytes(range(256))*400
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('train_series/gold/series/1.dcm', data)
        z.writestr('train_series/excluded/series/2.dcm', b'excluded')
    reader = h.fastread(h.runtime())
    archive = h.ArchiveReader(reader, path, ['gold'])
    names = reader.glob.glob(archive.root + '/gold/series/*.dcm')
    assert len(names) == 1 and len(archive.members) == 1
    assert reader._read_whole(names[0]) == data
    fd = reader._fopen(names[0])
    assert reader._pread(fd, 100, 95) == data[95:195]
    assert reader._read_rest(fd, data[:1024], 1024) == data
    reader.os.close(fd)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(x == data for x in pool.map(lambda _: reader._read_whole(names[0]), range(32)))
    assert archive.read_errors == 0
    archive.close()
    with pytest.raises(ValueError, match='missing requested'):
        h.ArchiveReader(h.fastread(h.runtime()), path, ['missing'])


def test_archive_read_error_is_counted(tmp_path, monkeypatch):
    import zipfile
    path = tmp_path / 'bad-read.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('train_series/gold/series/1.dcm', b'bytes')
    reader = h.fastread(h.runtime())
    archive = h.ArchiveReader(reader, path, ['gold'])
    def fail(_):
        raise zipfile.BadZipFile('CRC mismatch')
    monkeypatch.setattr(archive.archive, 'read', fail)
    with pytest.raises(zipfile.BadZipFile):
        reader._fopen(next(iter(archive.members)))
    assert archive.read_errors == 1 and not archive.handles
    archive.close()
