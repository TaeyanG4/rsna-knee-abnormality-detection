"""CPU-only contract checks for EXP158; no network or CUDA initialization."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys

import pytest

from tools import exp158_dino_strong as e


def test_runner_keeps_triplets_and_imports_aug1():
    runner = e.runner_source()
    assert 'tds = Aug1TripletWindows(' in runner
    assert 'DataLoader(StudyWindows(corpus.dir, ids, all_labels, a.res, a.k_eval, False' in runner
    assert 'from tools import exp158_dino_strong as guard' in runner
    assert 'provenance["aug1_sha256"]' in runner
    assert 'guard.check_prefix_trace(trace.hexdigest())' in runner
    assert '--fixed-epoch' in runner


def test_prepare_validate_and_reject_tamper(tmp_path):
    overlay = tmp_path / 'overlay'
    with contextlib.redirect_stdout(io.StringIO()):
        e.prepare(overlay)
    p = e.validate_manifest(overlay / 'manifest.json', overlay)
    assert p['expected_args']['epochs'] == p['expected_args']['fixed_epoch'] == 20
    assert p['expected_args']['k'] == 12 and p['expected_args']['k_eval'] == 36
    assert p['expected_ids']['train'] == 3455 and p['expected_ids']['val'] == 888
    assert p['arm']['initial_state_sha256'] == e.INITIAL_SHA
    assert e.sha(overlay / e.PROTOCOL_BIND) == p['protocol_sha256']
    assert not (overlay / 'artifacts/experiments/EXP158_DINO_PATCH_STRONG').exists()
    assert e.read(overlay / 'queue.json') == e.QUEUE
    (overlay / 'training/src/exp158_aug1_triplet.py').write_text('tamper', encoding='utf-8')
    with pytest.raises(ValueError, match='file differs'):
        e.validate_manifest(overlay / 'manifest.json', overlay)


def test_linux_paths_and_detached_queue():
    with pytest.raises(ValueError, match='Linux'):
        e.vast_plan('C:\\workspace\\exp158')
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        e.vast_plan('/workspace/rsna-exp158-dino-strong-v1')
    text = output.getvalue()
    assert 'setsid nohup python tools/gpu_queue.py queue.json --foreground' in text
    assert '\\' not in text
    assert 'max_gpu_seconds=10800' in text


def test_no_remote_launch_on_prepare(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('external process should not start')
    monkeypatch.setattr('subprocess.run', forbidden)
    with contextlib.redirect_stdout(io.StringIO()):
        e.prepare(tmp_path / 'overlay')
    assert not (tmp_path / 'overlay' / 'admission.json').exists()
    assert not (tmp_path / 'overlay' / 'execution.json').exists()


def test_posix_package_path_rejects_escape(tmp_path):
    for name in ('../escape', 'C:\\windows', '/workspace/absolute', 'x\\y'):
        with pytest.raises(ValueError):
            e.within(tmp_path, name)
    assert e.within(tmp_path, 'training/src/ok.py') == tmp_path / 'training/src/ok.py'


def test_aug1_training_retains_adjacent_slice_channels(monkeypatch):
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    src = e.ROOT / 'training/src'
    sys.path.insert(0, str(src))
    try:
        import exp158_aug1_triplet as adapter
        seen = {}

        class Corpus:
            depth = 64
            spec = {'slots': [('Sagittal', 1, 18), ('Sagittal', 0, 14),
                              ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)]}
            def __init__(self, path): pass
            def volume(self, uid): return None
            def mask(self, uid): return None

        def windows(*args):
            return np.stack([np.full((2, 2), x, np.float32) for x in (.1, .2, .3)])[None]

        def augment(x, centers, rng, spec):
            seen['x'] = x.copy()
            seen['centers'] = centers
            return x

        monkeypatch.setattr(adapter, 'Corpus', Corpus)
        monkeypatch.setattr(adapter, 'select_centers', lambda *args: [40])
        monkeypatch.setattr(adapter, 'study_rng', lambda *args: object())
        monkeypatch.setattr(adapter, 'windows_from', windows)
        monkeypatch.setattr(adapter, 'augment_windows', augment)
        ds = adapter.Aug1TripletWindows.__new__(adapter.Aug1TripletWindows)
        ds.corpus, ds.corpus_dir = None, 'fake'
        ds.ids, ds.labels = ['synthetic'], {'synthetic': np.zeros(12, np.float32)}
        ds.epoch, ds.seed, ds.k, ds.res = torch.zeros(1, dtype=torch.int64), 42, 1, 2
        ds.window_policy, ds.jitter, ds.norm, ds.train = 'within-slot-valid', 0, 'imagenet', True
        image, label, uid = ds[0]
        assert uid == 'synthetic' and image.shape == (1, 3, 2, 2) and label.shape == (12,)
        assert seen['centers'] == [40]
        assert np.allclose(seen['x'][0, :, 0, 0], [.1, .2, .3])
        assert not torch.cuda.is_initialized()
    finally:
        sys.path.remove(str(src))
