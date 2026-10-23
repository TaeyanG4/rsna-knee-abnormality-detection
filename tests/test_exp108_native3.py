"""CPU-only checks for H-B-25 / EXP108: native-adjacent triplet windows, the native3 cache builder/renderer and the protocol guard.

No GPU and no training (the runner's --validate-only and smoke paths are exercised through tools/exp108_native3.py).
Tests on the real cache / protocol are skipped when those do not exist yet.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'src'))
import raptor_native3 as n3  # noqa: E402
import raptor_common  # noqa: E402
from raptor_common import Corpus, windows_from  # noqa: E402
from tools import exp108_native3 as m  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402

N3 = ROOT / m.CORPUS
BASE = ROOT / m.BASE_CORPUS
REAL = ROOT / m.PROTOCOL / 'manifest.json'


def _finished(path):
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('finished') is True
    except (OSError, ValueError):
        return False


built = pytest.mark.skipif(not _finished(N3 / 'manifest.json'), reason='native3 cache not finished')
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP108 protocol not prepared')
SPEC64 = dict(slots=[list(s) for s in raptor_common.SLOTS64], depth=64)


def runner_module():
    spec = importlib.util.spec_from_file_location('exp108_runner', ROOT / m.RUNNER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ neighbour rule and windows
def test_neighbour_indices_replicate_at_series_edges():
    assert n3.neighbour_indices(0, 5) == (0, 1) and n3.neighbour_indices(4, 5) == (3, 4) and n3.neighbour_indices(2, 5) == (1, 3)
    assert n3.neighbour_indices(0, 1) == (0, 0)
    for bad in ((5, 5), (-1, 5), (0, 0)):
        with pytest.raises(ValueError):
            n3.neighbour_indices(*bad)


def test_spec_for_adds_only_the_plane_description():
    base = dict(slots=SPEC64['slots'], img=336, crop_mm=140.0, span=[0.02, 0.98], depth=64, normalization='n')
    s = n3.spec_for(base)
    assert {k: v for k, v in s.items() if k not in m.NATIVE3_SPEC_KEYS} == base and n3.is_native3_spec(s) and not n3.is_native3_spec(base)
    assert s['plane_order'] == ['native_prev', 'centre', 'native_next'] and s['planes'] == 3 and s['base_contract'] == m.BASE_CONTRACT
    assert n3.CONTRACT not in raptor_common.CONTRACTS  # deliberately not a DicomCorpus contract


def _native_from_cache(vol):
    """A native3 volume whose outer planes are exactly the cache neighbours (centre replicated at the volume ends)."""
    d = vol.shape[0]
    return np.stack([vol[np.maximum(np.arange(d) - 1, 0)], vol, vol[np.minimum(np.arange(d) + 1, d - 1)]], 1)


def test_windows_from_native3_equals_windows_from_when_neighbours_coincide():
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 256, size=(64, 12, 12), dtype=np.uint8)
    v3 = _native_from_cache(vol)
    centers = [1, 5, 17, 33, 62, -1]
    for res in (12, 8):
        assert np.array_equal(n3.windows_from_native3(v3, centers, res), windows_from(vol, centers, res))
    v3b = v3.copy(); v3b[5, 0] = 0; v3b[5, 2] = 255
    x = n3.windows_from_native3(v3b, [5], 12)[0]
    assert np.array_equal(x[1], vol[5] / np.float32(255)) and x[0].max() == 0 and x[2].min() == 1
    assert not n3.windows_from_native3(v3, [-1], 8).any()
    with pytest.raises(ValueError, match='native3 volume'):
        n3.windows_from_native3(vol, [5], 8)


def _tiny_pair(tmp_path, n=3, img=8, missing=None, perturb=False):
    rng = np.random.default_rng(0)
    vols = rng.integers(1, 255, size=(n, 64, img, img), dtype=np.uint8)
    masks = np.ones((n, 64), np.uint8)
    for r, idx in (missing or {}).items():
        vols[r, list(idx)] = 0; masks[r, list(idx)] = 0
    ids = [f'u{i}' for i in range(n)]
    v3 = np.stack([_native_from_cache(v) for v in vols])
    if perturb:
        v3[:, :, 0] = 255 - v3[:, :, 0]; v3[:, :, 2] = 7
    for name, arr, spec in (('base', vols, dict(slots=SPEC64['slots'], img=img, depth=64)),
                            ('n3', v3, n3.spec_for(dict(slots=SPEC64['slots'], img=img, depth=64)))):
        d = tmp_path / name; d.mkdir()
        np.save(d / 'vols_shard00.npy', arr); np.save(d / 'masks.npy', masks); np.save(d / 'ids.npy', np.array(ids))
        (d / 'manifest.json').write_text(json.dumps(dict(contract=name, spec=spec, shards=[dict(shard=0, start=0, stop=n, file='vols_shard00.npy')])),
                                         encoding='utf-8')
    return ids


@pytest.mark.parametrize('train', [True, False])
def test_native_dataset_equals_studywindows_when_neighbours_coincide(tmp_path, train):
    from raptor_data import StudyWindows
    from raptor_native3_data import NativeTripletWindows
    ids = _tiny_pair(tmp_path, missing={1: range(32, 44), 2: range(64)})
    labels = {u: np.zeros(12, np.float32) for u in ids}
    a = NativeTripletWindows(tmp_path / 'n3', ids, labels, 8, 12, train, 'imagenet', 42, window_policy='within-slot-valid')
    b = StudyWindows(tmp_path / 'base', ids, labels, 8, 12, train, 'imagenet', 42, window_policy='within-slot-valid')
    for ep in (0, 3):
        a.set_epoch(ep); b.set_epoch(ep)
        for i in range(len(ids)):
            xa, ya, ua = a[i]; xb, yb, ub = b[i]
            assert ua == ub and np.array_equal(xa.numpy(), xb.numpy()) and np.array_equal(ya.numpy(), yb.numpy())


def test_native_dataset_changes_only_the_outer_channels(tmp_path):
    from raptor_data import StudyWindows
    from raptor_native3_data import NativeTripletWindows
    ids = _tiny_pair(tmp_path, perturb=True)
    labels = {u: np.zeros(12, np.float32) for u in ids}
    a = NativeTripletWindows(tmp_path / 'n3', ids, labels, 8, 12, True, 'none', 42, window_policy='within-slot-valid')
    b = StudyWindows(tmp_path / 'base', ids, labels, 8, 12, True, 'none', 42, window_policy='within-slot-valid')
    for i in range(len(ids)):
        xa, xb = a[i][0].numpy(), b[i][0].numpy()
        assert np.array_equal(xa[:, 1], xb[:, 1]) and not np.array_equal(xa[:, 0], xb[:, 0]) and not np.array_equal(xa[:, 2], xb[:, 2])
    with pytest.raises(ValueError, match='not a native3 cache'):
        NativeTripletWindows(tmp_path / 'base', ids, labels, 8, 12, True, 'none', 42, window_policy='within-slot-valid')[0]


def test_runner_input_check_reports_unchanged_centres(tmp_path):
    from types import SimpleNamespace
    ids = _tiny_pair(tmp_path, perturb=True)
    labels = {u: np.zeros(12, np.float32) for u in ids}
    a = SimpleNamespace(res=8, k=12, norm='imagenet', seed=42, window_policy='within-slot-valid')
    rep = runner_module().input_check(tmp_path / 'n3', tmp_path / 'base', ids, labels, a, n=3)
    assert rep == dict(studies=3, windows=36, centre_channel_equal=36, outer_channels_changed=36)


# ------------------------------------------------------------------ DICOM renderer on synthetic series
def _write_series(directory, positions, values, spacing=0.5, size=40, name_order=None):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    directory.mkdir(parents=True, exist_ok=True)
    for i, (z, v) in enumerate(zip(positions, values)):
        meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'; meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = 'MR'; ds.Rows = ds.Columns = size; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        ds.PixelSpacing = [spacing, spacing]; ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.ImagePositionPatient = [0, 0, float(z)]
        ds.InstanceNumber = i + 1
        yy, xx = np.mgrid[0:size, 0:size]
        ds.PixelData = ((xx + yy) * 3 + v).astype(np.uint16).tobytes()
        fname = (name_order[i] if name_order else f'{i:03d}') + '.dcm'
        pydicom.dcmwrite(directory / fname, ds, enforce_file_format=True)


def test_prepare_study_native3_uses_physical_neighbours_and_matches_prepare_study(tmp_path):
    from raptor_dicom import DirectoryDicomSource, prepare_study
    # 6 slices; file names deliberately not in physical order; z decreasing with the file index
    z = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    values = [600, 500, 400, 300, 200, 100]  # slice at z=k has value 100*(k+1)
    _write_series(tmp_path / 'src' / 'S' / 'ser', z, values, name_order=['f', 'b', 'd', 'a', 'e', 'c'])
    spec = dict(slots=[['Sagittal', 1, 3]], depth=3, img=8, crop_mm=140.0, span=[0.02, 0.98])
    rows = [{'StudyInstanceUID': 'S', 'SeriesInstanceUID': 'ser', 'Anatomical_Plane': 'Sagittal', 'Fluid_Sensitive': '1'}]
    with DirectoryDicomSource(tmp_path / 'src') as source:
        v3, m3, rec = n3.prepare_study_native3('S', rows, source, n3.spec_for(spec), 'train')
        v64, m64, _ = prepare_study('S', rows, source, spec, 'train')
    assert v3.shape == (3, 3, 8, 8) and np.array_equal(v3[:, 1], v64) and np.array_equal(m3, m64)
    # sorted by z: index k has value 100*(k+1); picks linspace(0, 4, 3) = [0, 2, 4]
    assert rec['slots'][0]['slices'] == [0, 2, 4] and rec['edge_replications'] == 1 and rec['neighbour_decode_failures'] == 0
    means = v3.reshape(3, 3, -1).mean(-1)
    assert np.array_equal(v3[0, 0], v3[0, 1])                       # pick 0: no previous slice -> centre replicated
    assert means[1, 0] < means[1, 1] < means[1, 2]                  # pick 2: native 1 / 2 / 3 brighten with z
    assert means[2, 0] < means[2, 1] and means[2, 2] > means[2, 1]  # pick 4: native 3 / 4 / 5 (5 is not a pick, outside the span)
    assert np.array_equal(v3[0, 2], v3[1, 0])                       # pick 0's next (native 1) == pick 2's previous (native 1)
    DC3 = n3.dicom_corpus_class()
    with DirectoryDicomSource(tmp_path / 'src') as source:
        with pytest.raises(ValueError, match='serves only'):
            DC3.from_records(['S'], rows, source, m.BASE_CONTRACT, 'train')


# ------------------------------------------------------------------ builder helpers
def test_disk_estimate_and_worker_cap(tmp_path):
    import raptor_native3_corpus as b
    spec = n3.spec_for(b.base_spec())
    est = b.disk_estimate(4410, spec, tmp_path / 'not' / 'yet')
    assert est['bytes_per_study'] == 64 * 3 * 336 * 336 == 21676032 and est['estimate_bytes'] == 4410 * 21676032
    assert est['required_free_gb'] == round(2.5 * est['estimate_bytes'] / 1e9, 1)
    with pytest.raises(SystemExit, match='workers'):
        b.main(['--workers', '4', '--estimate'])


@pytest.mark.skipif(not (BASE / 'manifest.json').exists(), reason='base cache absent')
def test_builder_base_spec_is_the_base_cache_spec():
    import raptor_native3_corpus as b
    assert json.loads(json.dumps(b.base_spec())) == m.read(BASE / 'manifest.json')['spec']


# ------------------------------------------------------------------ recipe / contract / checks
def test_recipe_args_are_the_unchanged_b8_recipe():
    args = m.recipe_args(dict(k99.COMMON))
    assert args == {k: v for k, v in k99.COMMON.items() if k != 'workers'} and (args['k'], args['k_eval']) == (12, 24)
    with pytest.raises(ValueError, match='B8 recipe'):
        m.recipe_args(dict(k99.COMMON, k=16))


def test_run_contract_from_train_size():
    c = m.run_contract(3455, m.recipe_args(dict(k99.COMMON)))
    assert (c['len_train_loader'], c['steps'], c['scheduler_total_steps'], c['max_fp16_skips'], c['eval_batch']) == (431, 3448, 3448, 34, 4)


def _manifests():
    spec64 = dict(slots=SPEC64['slots'], img=336, crop_mm=140.0, span=[0.02, 0.98], depth=64, normalization='n', order='o', resize='r', series_selection='s')
    base = dict(contract=m.BASE_CONTRACT, spec=spec64, studies=2, shard_size=400, order='x', builder='b', archive=dict(directory_sha256='z'), finished=True,
                shards=[dict(done=True)])
    n3m = dict(copy.deepcopy(base), contract=m.CONTRACT, spec=n3.spec_for(spec64), base_builder='b', shape_total=[2, 64, 3, 336, 336],
               centre_mask_mismatches=0, masks_equal_base=True, builder='native3')
    return n3m, base


def test_corpus_checks_accept_only_the_plane_change():
    n3m, base = _manifests()
    ids, masks = np.array(['a', 'b']), np.ones((2, 64), np.uint8)
    assert m.corpus_checks(n3m, base, ids, ids, masks, masks) == []
    assert m.corpus_checks(n3m, base, ids, ids[::-1], masks, masks) == ['study ids differ from the base cache']
    m2 = masks.copy(); m2[0, 5] = 0
    assert m.corpus_checks(n3m, base, ids, ids, m2, masks) == ['masks differ from the base cache']
    bad = copy.deepcopy(n3m); bad['spec']['crop_mm'] = 120.0; bad['centre_mask_mismatches'] = 3; bad['finished'] = False
    assert set(m.corpus_checks(bad, base, ids, ids, masks, masks)) == {
        'spec differs from the base spec beyond the native3 keys', 'build reported centre/mask differences to the base cache', 'native3 cache not finished'}


def test_check_reports_pass_conditions():
    dicom = dict(rendered=14, cache_equal=14, centre_equal_dicom64=14, centre_equal_base_cache=14, mask_equal=14, render_failed=[dict(cache_all_zero=True)])
    cons = dict(failures=0, checks=200000, ids_equal_base=True, masks_equal_base=True)
    assert m.check_reports(dicom, cons) == []
    assert m.check_reports(dict(dicom, cache_equal=13), cons) and m.check_reports(dicom, dict(cons, failures=1))
    assert m.check_reports(dict(dicom, render_failed=[dict(cache_all_zero=False, base64_fails_too=False)]), cons)
    assert m.check_reports(dict(dicom, render_failed=[dict(cache_all_zero=False, base64_fails_too=True)]), cons) == []


# ------------------------------------------------------------------ the real native3 cache
@built
def test_real_cache_manifest_ids_masks_and_centre_planes():
    d, b = Corpus(N3), Corpus(BASE)
    assert d.contract == m.CONTRACT and d.depth == 64 and n3.is_native3_spec(d.spec) and d.manifest['finished'] is True
    assert np.array_equal(d.ids, b.ids) and np.array_equal(d.masks, b.masks)
    assert m.corpus_checks(d.manifest, b.manifest, d.ids, b.ids, d.masks, b.masks) == []
    assert d.manifest['code_registry']['module_sha256'] == m.sha256(ROOT / m.EXT_FILES[0])
    assert d.manifest['code_registry']['raptor_common_sha256'] == m.read(ROOT / m.V4)['training_code_sha256']['raptor_common.py']
    rng = np.random.default_rng(7)
    for r in rng.choice(len(d), 3, replace=False):
        u = d.ids[r]
        v = d.volume(u)
        assert v.shape == (64, 3, 336, 336) and np.array_equal(v[:, 1], b.volume(u))


# ------------------------------------------------------------------ the registered protocol
@pytest.fixture()
def protocol():
    return m.read(REAL)


@pytest.fixture()
def env(monkeypatch):
    for k, v in {**m.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)


def _args(p, mode='full', output_root=None, extra=()):
    argv = m.runner_argv(p, mode, output_root=output_root)[2:] + list(extra)
    return runner_module().build_parser().parse_args(argv)


@prepared
def test_protocol_registration(protocol):
    p = protocol
    v4 = m.read(ROOT / m.V4)
    assert p['schema'] == m.SCHEMA and p['plan_item'] == 'H-B-25' and p['status'] == 'REGISTERED_NO_EXECUTION'
    assert p['initial_state_sha256'] == v4['initial_state_sha256'] and p['initial_state_sha256'].startswith('e4ea3ae6')
    assert p['common_args'] == {k: v for k, v in v4['common_args'].items() if k != 'workers'}
    s = p['split']
    assert (s['n_train'], s['n_val'], s['len_train_loader'], s['steps'], s['max_fp16_skips'], s['eval_batch']) == (3455, 888, 431, 3448, 34, 4)
    assert s['pos_weight'] == v4['loss']['baseline_pos_weight'] and p['training_code_sha256'] == v4['training_code_sha256']
    assert p['corpus']['manifest_sha256'] == m.sha256(N3 / 'manifest.json') and p['corpus']['planes'] == 3
    assert p['base_corpus']['manifest_sha256'] == v4['corpus_manifest_sha256']
    for path, expected in p['source_bindings'].items():
        assert m.sha256(ROOT / path) == expected, path
    assert [e['file'] for e in p['extra_evals']] == ['oof_k36.csv'] and p['tags']['full'] == 'EXP108_B8_NATIVE3_F0'


@prepared
def test_queue_json_entry(protocol):
    q = m.read(ROOT / m.PROTOCOL / 'queue.json')
    assert len(q) == 1 and q[0]['tag'] == 'EXP108_B8_NATIVE3_F0' and q[0]['after'] == 'EXP106_B8_E16_F0' and q[0]['continue_on_failure'] is True
    assert q[0]['command'] == m.runner_argv(protocol, 'full')
    v8 = m.read(ROOT / 'artifacts/experiments/LOCAL_GPU_CHAIN_V8.json')
    assert set(q[0]) == set(v8[0]) and q[0]['command'][0] == v8[0]['command'][0]


@prepared
def test_runner_guard_accepts_registered_argv_and_rejects_changes(protocol, env, tmp_path):
    p = protocol
    got, ordered, out_root = m.validate_arguments(_args(p), ROOT)
    assert out_root == (ROOT / 'artifacts/experiments').resolve() and len(ordered['train']) == 3455
    m.validate_arguments(_args(p, 'smoke', output_root=tmp_path), ROOT)
    for extra, message in [(['--k', '16'], 'frozen argument changed: k'), (['--k-eval', '36'], 'frozen argument changed: k_eval'),
                           (['--tag', 'EXP108_X'], 'tag/mode'), (['--workers', '4'], 'registered worker count'), (['--seed', '43'], 'seed'),
                           (['--corpus', str(BASE)], 'corpus'), (['--output-root', str(tmp_path)], 'artifacts/experiments'),
                           (['--epochs', '16'], 'frozen argument changed: epochs')]:
        with pytest.raises(ValueError, match=message):
            m.validate_arguments(_args(p, extra=extra), ROOT)
    with pytest.raises(ValueError, match='outside artifacts'):
        m.validate_arguments(_args(p, 'smoke', output_root=ROOT / 'artifacts' / 'tmp-smoke'), ROOT)


@prepared
def test_runner_guard_requires_the_thread_environment(protocol, env, monkeypatch):
    monkeypatch.delenv('OMP_NUM_THREADS')
    with pytest.raises(ValueError, match='OMP_NUM_THREADS'):
        m.validate_arguments(_args(protocol), ROOT)
