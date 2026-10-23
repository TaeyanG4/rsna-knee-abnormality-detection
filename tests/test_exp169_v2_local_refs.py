"""CPU checks for tools/exp169_v2_local_refs.py (Claude opus for B, 2026-10-10).

Synthetic data only: no checkpoint is deserialized, no GPU, no labels. The staged
frozen sources are read (and the patched partner flow is executed with fake model
calls) when the v2 stage folders exist locally.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tools import exp169_v2_local_refs as refs

ROOT = Path(__file__).resolve().parents[1]
COMPARE = Path(r'C:/Users/Taeyang/.claude/skills/numerical-parity-skill/scripts/compare_arrays.py')
STAGES = [ROOT / f'artifacts/releases/EXP169_{c}_{v}_stage' for c in ('A172', 'B172')
          for v in ('v2_0_1', 'v2_0_2')]
STAGES = [s for s in STAGES if (s / 'assets/code/exp169_partner_bundle_release.py').is_file()]


def _scores(n=3, seed=169):
    rng = np.random.default_rng(seed)
    return {family: (0.1 + 0.8 * rng.random((rows, n, 12))).astype(np.float32)
            for family, rows in refs.FAMILIES}


def _raw(path, ids, scores):
    with Path(path).open('xb') as stream:
        np.savez(stream, ids=np.asarray(ids), **scores)
    return Path(path)


def _entries():
    out = []
    for family, seeds in (('l2', (42, 43)), ('dino', (42,)), ('convnext', (42,))):
        for seed in seeds:
            for fold in range(5):
                path = f'models/{family}_s{seed}_f{fold}.pt'
                out.append(dict(family=family, seed=seed, fold=fold, tag=f'T_{family}_{seed}_{fold}',
                                path=path, sha256=f'{family}{seed}{fold}'.ljust(64, '0'),
                                source_sha256='s' * 64, bytes=10 + fold))
    return out


def _manifest():
    entries = _entries()
    files = {e['path']: dict(sha256=e['sha256'], bytes=e['bytes']) for e in entries}
    files.update({'code/exp169_partner_bundle_release.py': dict(sha256='c' * 64, bytes=1),
                  'config/workspace.toml': dict(sha256='w' * 64, bytes=1),
                  'protocols/verification.md': dict(sha256='v' * 64, bytes=1)})
    return dict(candidate='B172', entries=entries, files=files, k_eval=36)


# ------------------------------------------------------------------ layout

def test_member_layout_matches_frozen_row_rule():
    layout = refs.member_layout()
    assert len(layout) == 20 and len({k for k, _, _ in layout}) == 20
    assert ('l2_s43_f0', 'l2', 5) in layout and ('l2_s42_f4', 'l2', 4) in layout
    assert ('dino_s42_f4', 'dino', 4) in layout and ('convnext_s42_f0', 'convnext', 0) in layout
    entries = _entries()
    assert sorted(refs.layout_from_manifest(entries[::-1])) == sorted(layout)
    with pytest.raises(ValueError, match='20-member layout'):
        refs.layout_from_manifest(entries[:-1])
    bad = copy.deepcopy(entries)
    bad[5]['seed'] = 42  # an S43 member relabelled as S42 collides with fold rows 0-4
    with pytest.raises(ValueError, match='20-member layout'):
        refs.layout_from_manifest(bad)


def test_split_members_roundtrip_dtype_and_no_overwrite(tmp_path):
    ids, scores = ['s-a', 's-b', 's-c'], _scores()
    raw = _raw(tmp_path / 'partner-raw.npz', ids, scores)
    info = refs.split_members(raw, tmp_path / 'members.npz')
    assert info['studies'] == 3 and len(info['members']) == 20
    with np.load(tmp_path / 'members.npz', allow_pickle=False) as z:
        assert set(z.files) == {'ids', *info['members']}
        assert z['ids'].tolist() == ids
        for key, family, row in refs.member_layout():
            assert z[key].dtype == np.float32 and z[key].shape == (3, 12)
            assert np.array_equal(z[key], scores[family][row])
    with pytest.raises(FileExistsError):
        refs.split_members(raw, tmp_path / 'members.npz')


@pytest.mark.parametrize('bad', ['missing', 'extra', 'rows', 'nonfinite', 'range', 'dup_ids', 'cohort'])
def test_partner_contract_rejects_invalid_arrays(tmp_path, bad):
    ids, scores = ['s-a', 's-b', 's-c'], _scores()
    arrays = {'ids': np.asarray(ids), **scores}
    if bad == 'missing':
        arrays.pop('dino')
    elif bad == 'extra':
        arrays['public'] = scores['dino']
    elif bad == 'rows':
        arrays['convnext'] = scores['convnext'][:4]
    elif bad == 'nonfinite':
        arrays['l2'] = scores['l2'].copy(); arrays['l2'][7, 1, 3] = np.nan
    elif bad == 'range':
        arrays['dino'] = scores['dino'].copy(); arrays['dino'][0, 0, 0] = 1.5
    elif bad == 'dup_ids':
        arrays['ids'] = np.asarray(['s-a', 's-a', 's-c'])
    else:
        arrays['ids'] = np.asarray(ids[:2])
    raw = tmp_path / 'bad.npz'
    np.savez(raw, **arrays)
    with pytest.raises(ValueError):
        refs.split_members(raw, tmp_path / 'members.npz')
    assert not (tmp_path / 'members.npz').exists()


@pytest.mark.skipif(not COMPARE.is_file(), reason='parity skill compare_arrays.py not present')
@pytest.mark.parametrize('change,key,expected', [
    (None, None, 0),
    ('one_cell', 'l2_s43_f2', 0),          # max 1.5e-3 <= 2e-3, mean tiny: cross-GPU fp16 noise passes
    ('one_cell_big', 'dino_s42_f1', 1),    # max 2.5e-3 > 2e-3
    ('member_shift', 'convnext_s42_f3', 1),  # mean 6e-4 > 5e-4 in one member only
])
def test_members_npz_is_loadable_by_compare_arrays_with_registered_tol(tmp_path, change, key, expected):
    ids, scores = ['s-a', 's-b', 's-c'], _scores()
    other = {k: v.copy() for k, v in scores.items()}
    if change:
        family, row = next((f, r) for k, f, r in refs.member_layout() if k == key)
        if change == 'member_shift':
            other[family][row] += np.float32(6e-4)
        else:
            other[family][row, 1, 4] += np.float32(1.5e-3 if change == 'one_cell' else 2.5e-3)
    a = refs.split_members(_raw(tmp_path / 'a.npz', ids, scores), tmp_path / 'a-members.npz')
    b = refs.split_members(_raw(tmp_path / 'b.npz', ids, other), tmp_path / 'b-members.npz')
    run = subprocess.run([sys.executable, str(COMPARE), a['out'], b['out'], '--tol', str(refs.TOL_JSON),
                          '--out', str(tmp_path / 'report.json')], capture_output=True, text=True)
    assert run.returncode == expected, run.stdout + run.stderr
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert report['tolerance_file_sha256'] == refs.sha(refs.TOL_JSON)
    assert report['keys']['ids']['exact_equal'] is True
    assert report['failed_keys'] == ([key] if expected else [])
    if change == 'member_shift':
        # The family-level 3-D array would average the shift over five members and pass.
        whole = subprocess.run([sys.executable, str(COMPARE), str(tmp_path / 'a.npz'), str(tmp_path / 'b.npz'),
                                '--tol', str(refs.TOL_JSON)], capture_output=True, text=True)
        assert whole.returncode == 0, whole.stdout


def test_registration_binds_tol_json():
    reg = refs.registration_binding()
    assert reg['tol_json_sha256'] == refs.sha(refs.TOL_JSON)
    tol = json.loads(refs.TOL_JSON.read_text(encoding='utf-8'))
    assert tol == {'default': {'max_abs': 0.002, 'mean_abs': 0.0005, 'min_spearman': 0.99}}
    doc = json.loads(refs.REGISTRATION.read_text(encoding='utf-8'))
    assert doc['unchanged']['in_kernel_pilot_vs_full_partner_gate'] == 2e-05


# ------------------------------------------------------------------ identity

def test_partner_identity_ignores_non_partner_files_but_not_members():
    a, b = _manifest(), _manifest()
    b['candidate'] = 'A172'
    b['files']['protocols/verification.md'] = dict(sha256='x' * 64, bytes=2)
    assert refs.partner_identity({'A': a, 'B': b})['digest']
    for mutate in (lambda m: m['entries'][3].update(tag='other'),
                   lambda m: m['files']['models/dino_s42_f0.pt'].update(sha256='y' * 64),
                   lambda m: m['files']['code/exp169_partner_bundle_release.py'].update(bytes=3),
                   lambda m: m.update(k_eval=34)):
        c = _manifest()
        mutate(c)
        with pytest.raises(ValueError, match='Partner members differ'):
            refs.partner_identity({'A': a, 'C': c})


def test_verify_checkpoints_fails_loudly_on_any_mismatch(tmp_path):
    manifest = _manifest()
    for e in manifest['entries']:
        path = tmp_path / e['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(e['tag'].encode())
        e['sha256'] = refs.sha(path)
        e['bytes'] = path.stat().st_size
        manifest['files'][e['path']] = dict(sha256=e['sha256'], bytes=e['bytes'])
    assert len(refs.verify_checkpoints(tmp_path, manifest)) == 20
    (tmp_path / manifest['entries'][12]['path']).write_bytes(b'tampered!!!')
    with pytest.raises(ValueError, match='SHA/bytes mismatch'):
        refs.verify_checkpoints(tmp_path, manifest)


def test_stage_identity_binds_receipt_to_manifest(tmp_path):
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets/release-manifest.json').write_text(json.dumps({'candidate': 'B172'}), encoding='utf-8')
    receipt = dict(status='PLAN_ONLY_NOT_PUSHED', candidate='B172', version='2.0.2',
                   manifest_sha256=refs.sha(tmp_path / 'assets/release-manifest.json'))
    (tmp_path / 'build-receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    assert refs.stage_identity(tmp_path)['candidate'] == 'B172'
    (tmp_path / 'assets/release-manifest.json').write_text(json.dumps({'candidate': 'B172', 'x': 1}),
                                                           encoding='utf-8')
    with pytest.raises(ValueError, match='Stage receipt/manifest binding'):
        refs.stage_identity(tmp_path)


# ------------------------------------------------------------------ inputs

def _competition(tmp_path, ids=('s-a', 's-b', 's-c')):
    comp, dicom = tmp_path / 'competition', tmp_path / 'dicom'
    comp.mkdir()
    with (comp / 'sample_submission.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow([refs.UID, *refs.TARGETS]); w.writerows([[u, *[.5] * 12] for u in ids])
    (comp / 'test.csv').write_text(refs.UID + '\n' + '\n'.join(ids) + '\n', encoding='utf-8')
    (comp / 'test_series.csv').write_text(refs.UID + ',SeriesInstanceUID\n' +
                                          ''.join(f'{u},{u}.1\n{u},{u}.2\n' for u in ids), encoding='utf-8')
    for u in ids:
        (dicom / u / f'{u}.1').mkdir(parents=True)
        (dicom / u / f'{u}.1' / 'x.dcm').write_bytes(u.encode())
    return comp, dicom


def test_visible_inputs_and_view(tmp_path):
    comp, dicom = _competition(tmp_path)
    ids, info = refs.visible_inputs(comp, dicom)
    assert ids == ['s-a', 's-b', 's-c'] and info['dicom_tree']['files'] == 3
    view = refs.competition_view(tmp_path / 'view', comp, dicom)
    assert refs.tree_digest(view / 'test_series') == info['dicom_tree']
    assert refs.sha(view / 'test.csv') == info['csv_sha256']['test.csv']
    (dicom / 's-d').mkdir()
    with pytest.raises(ValueError, match='DICOM study folders differ'):
        refs.visible_inputs(comp, dicom)


def test_visible_inputs_rejects_roster_disagreement(tmp_path):
    comp, dicom = _competition(tmp_path)
    (comp / 'test.csv').write_text(refs.UID + '\ns-b\ns-a\ns-c\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Visible roster differs'):
        refs.visible_inputs(comp, dicom)


# ------------------------------------------------------------------ outputs

def test_finalize_partner_writes_four_files_atomically(tmp_path):
    ids, scores = ['s-a', 's-b', 's-c'], _scores()
    out = tmp_path / 'local-refs'
    receipt = refs.finalize_partner(out, ids, scores, {'device': 'synthetic'}, {'note': 'synthetic'})
    assert sorted(p.name for p in out.iterdir()) == sorted(refs.PARTNER_FILES)
    assert receipt['status'] == 'COMPLETE_LOCAL_PARTNER_REFERENCE' and receipt['studies'] == 3
    stored = json.loads((out / 'partner-receipt.json').read_text(encoding='utf-8'))
    assert stored['files'] == {n: refs.sha(out / n) for n in refs.PARTNER_FILES[:3]}
    assert stored['summary']['l2']['shape'] == [10, 3, 12] and stored['summary']['dino']['finite'] is True
    with np.load(out / 'partner-raw.npz', allow_pickle=False) as z:
        assert set(z.files) == {'ids', 'l2', 'dino', 'convnext'} and z['ids'].tolist() == ids
        assert all(np.array_equal(z[f], scores[f]) for f, _ in refs.FAMILIES)
    with pytest.raises(ValueError, match='Fresh partner outputs'):
        refs.finalize_partner(out, ids, scores, {}, {})


def test_finalize_partner_refuses_invalid_scores_without_outputs(tmp_path):
    scores = _scores()
    scores['l2'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Partner family invalid'):
        refs.finalize_partner(tmp_path / 'out', ['a', 'b', 'c'], scores, {}, {})
    assert not any((tmp_path / 'out' / n).exists() for n in refs.PARTNER_FILES)


def _anchor(folder, ids, binding, extra=None):
    folder.mkdir(parents=True)
    arrays = {'ids': np.asarray(ids), 'targets': np.asarray(refs.TARGETS),
              'own': np.full((len(ids), 12), .25, np.float32)}
    arrays.update(extra or {})
    np.savez(folder / 'raw.npz', **arrays)
    (folder / 'inputs.json').write_text(json.dumps({u: {'volume': 'v'} for u in ids}), encoding='utf-8')
    (folder / 'runtime.json').write_text('{}', encoding='utf-8')
    (folder / 'binding.json').write_text(json.dumps(binding), encoding='utf-8')
    with (folder / 'submission.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow([refs.UID, *refs.TARGETS]); w.writerows([[u, *[.5] * 12] for u in ids])
    receipt = {'status': 'COMPLETE_INFERENCE', 'seconds': 1.,
               'files': {n: refs.sha(folder / n) for n in refs.ANCHOR_FILES}}
    (folder / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')


@pytest.mark.parametrize('bad', [None, 'binding', 'stale_raw', 'order', 'extra_arm', 'status'])
def test_check_anchor_folder_contract(tmp_path, bad):
    ids, binding = ['s-a', 's-b', 's-c'], {'variant': 'own', 'manifest_sha256': 'm' * 64}
    folder = tmp_path / 'anchor'
    _anchor(folder, ids, binding, {'public': np.zeros((3, 12))} if bad == 'extra_arm' else None)
    check_ids, check_binding = list(ids), dict(binding)
    if bad == 'binding':
        check_binding['manifest_sha256'] = 'n' * 64
    elif bad == 'stale_raw':
        np.savez(folder / 'raw.npz', ids=np.asarray(ids), targets=np.asarray(refs.TARGETS),
                 own=np.full((3, 12), .3))
    elif bad == 'order':
        check_ids = ids[::-1]
    elif bad == 'status':
        r = json.loads((folder / 'receipt.json').read_text(encoding='utf-8')); r['status'] = 'RUNNING'
        (folder / 'receipt.json').write_text(json.dumps(r), encoding='utf-8')
    if bad is None:
        summary = refs.check_anchor_folder(folder, check_binding, check_ids)
        assert summary['shape'] == [3, 12] and summary['finite']
    else:
        with pytest.raises(ValueError):
            refs.check_anchor_folder(folder, check_binding, check_ids)


def test_compare_inputs_reports_equal_and_differing_keys(tmp_path):
    local = {'s-a': {'volume': '1', 'plain': '2'}, 's-b': {'volume': '3', 'plain': '4'}}
    same, other = tmp_path / 'same.json', tmp_path / 'other.json'
    same.write_text(json.dumps(local), encoding='utf-8')
    changed = copy.deepcopy(local); changed['s-b']['plain'] = '9'
    other.write_text(json.dumps(changed), encoding='utf-8')
    out = refs.compare_inputs(local, [same, other, tmp_path / 'missing.json'])
    assert out[str(same)]['equal'] is True and out[str(other)]['differing_keys'] == {'s-b': ['plain']}
    assert out[str(tmp_path / 'missing.json')] == 'missing'


def test_cli_split_members(tmp_path, capsys):
    raw = _raw(tmp_path / 'r.npz', ['a', 'b', 'c'], _scores())
    result = refs.main(['split-members', '--raw', str(raw), '--out', str(tmp_path / 'm.npz')])
    assert result['sha256'] == refs.sha(tmp_path / 'm.npz')
    assert f'exp169_v2_local_refs v{refs.VERSION} split-members' in capsys.readouterr().out


def test_b174_publish_layout_binding_and_receipt(tmp_path):
    ids = ['s-a', 's-b', 's-c']
    binding = {'variant': 'own', 'manifest_sha256': 'm' * 64}
    work, out = tmp_path / 'b174-anchor174.incomplete', tmp_path / 'b174-anchor174'
    local_receipt = tmp_path / 'b174-anchor174.local-receipt.json'
    _anchor(work, ids, binding)
    (work / 'runtime.json').write_text(json.dumps({'device': 'cpu', 'precision': 'fp32'}), encoding='utf-8')
    receipt = json.loads((work / 'receipt.json').read_text(encoding='utf-8'))
    receipt['files']['runtime.json'] = refs.sha(work / 'runtime.json')
    (work / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    result = refs.publish_b174_anchor(work, out, local_receipt, binding, ids,
                                      {'binding_equal': True})
    assert result['status'] == 'COMPLETE_LOCAL_OWN_ANCHOR_REFERENCE'
    assert result['device'] == 'cpu' and result['precision'] == 'fp32'
    assert result['files']['raw.npz'] == refs.sha(out / 'raw.npz')
    assert result['files']['receipt.json'] == refs.sha(out / 'receipt.json')
    assert refs.check_anchor_folder(out, binding, ids)['shape'] == [3, 12]
    assert json.loads(local_receipt.read_text(encoding='utf-8')) == result
    with pytest.raises(ValueError, match='Fresh B174'):
        refs.publish_b174_anchor(out, out, local_receipt, binding, ids, {})


def test_b174_publish_rejects_bad_binding_before_move(tmp_path):
    ids = ['s-a', 's-b', 's-c']
    work = tmp_path / 'b174-anchor174.incomplete'
    _anchor(work, ids, {'manifest_sha256': 'm' * 64})
    with pytest.raises(ValueError):
        refs.publish_b174_anchor(work, tmp_path / 'b174-anchor174',
                                 tmp_path / 'local-receipt.json', {'manifest_sha256': 'n' * 64}, ids, {})
    assert work.exists() and not (tmp_path / 'b174-anchor174').exists()


def test_b174_cpu_patch_on_staged_sources():
    stage = refs.B174_STAGE
    source = (stage / 'anchor174/assets/public-serving.py').read_text(encoding='utf-8')
    patched, info = refs.patch_source(source, refs.SERVING_PATCHES + refs.B174_CPU_PATCHES)
    assert info['original_sha256'] == refs.sha(stage / 'anchor174/assets/public-serving.py')
    assert 'dev = "cpu"  # local reference only' in patched
    assert 'ngpu = 0  # never use the busy local GPU' in patched
    assert 'torch.autocast("cpu", enabled=False)' in patched
    assert 'assert str(device) == "cpu"' in patched
    compile(patched, '<b174-cpu-serving>', 'exec')
    source = (stage / 'anchor174/assets/exp152_own0949_release.py').read_text(encoding='utf-8')
    bind = source[source.index('def bind(ns, manifest, files, variant, work):'):]
    bind = bind[:bind.index('\ndef kernel_metadata(')]
    patched, _ = refs.patch_source(bind, refs.B174_BIND_PATCHES)
    assert '_EXP169_LOCAL_DEVICE == "cpu"' in patched
    compile(patched, '<b174-cpu-bind>', 'exec')


def test_b174_cpu_runtime_does_not_query_cuda():
    torch = SimpleNamespace(__version__='synthetic', get_num_threads=lambda: 2)
    runtime = refs.b174_cpu_runtime(torch)
    assert runtime['device'] == 'cpu' and runtime['precision'] == 'fp32'
    assert runtime['threads'] == 2


# ------------------------------------------------------------------ staged frozen sources

def _staged_base(stage, name):
    spec = importlib.util.spec_from_file_location(name, stage / 'assets/code/exp169_partner_bundle_release.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not STAGES, reason='no staged v2 folders')
def test_partner_patches_apply_once_and_keep_frozen_inference_lines():
    hashes = set()
    for k, stage in enumerate(STAGES):
        base = _staged_base(stage, f'_exp169_staged_base_{k}')
        source = refs.inspect.getsource(base.partner_scores)
        patched, info = refs.patch_source(source, refs.PARTNER_PATCHES)
        hashes.add((info['original_sha256'], info['patched_sha256']))
        compile(patched, 'patched', 'exec')
        assert 'range(2)' not in patched and "'T4'" not in patched and 'Barrier(1)' in patched
        for line in ("plain = _infer_one(own[m], x, device, family == 'dino')",
                     'mirror[indices] = mirror[indices, :, :, ::-1]',
                     'value = (plain + _infer_one(own[m], mirror, device)) / 2',
                     "row = entry['fold'] + (5 if slot else 0)",
                     "centers = select_centers(mask, corpus.depth, K, corpus.spec, 'within-slot-valid')",
                     'torch.use_deterministic_algorithms(True)', 'torch.set_num_threads(1)'):
            assert source.count(line) == patched.count(line) == 1, line
    assert len(hashes) == 1  # identical frozen partner source in every stage


@pytest.mark.skipif(not STAGES, reason='no staged v2 folders')
def test_serving_patch_applies_to_staged_public_serving():
    for stage in STAGES:
        serving = stage / 'own-anchor/assets/public-serving.py'
        if not serving.is_file():
            continue
        manifest = json.loads((stage / 'assets/release-manifest.json').read_text(encoding='utf-8'))
        assert refs.sha(serving) == manifest['anchor_binding']['serving_sha256']
        patched, info = refs.patch_source(serving.read_text(encoding='utf-8'), refs.SERVING_PATCHES)
        assert info['original_sha256'] == refs.sha(serving)
        assert 'get_context("fork")' not in patched and patched.count('_EXP169_LOCAL_POOL(') == 1
        compile(patched, 'public-serving.py', 'exec')


def _fake_modules(n_uids):
    """Minimal torch/raptor stand-ins: windows carry the study index, models their entry index."""
    torch = ModuleType('torch')
    cuda = SimpleNamespace(device_count=lambda: 1, set_device=lambda g: None,
                           reset_peak_memory_stats=lambda g: None, max_memory_allocated=lambda g: 123)
    torch.cuda = cuda
    torch.set_num_threads = lambda n: None
    torch.use_deterministic_algorithms = lambda flag: None
    corpus_ids = [f'u{i}' for i in range(n_uids)]

    class Corpus:
        def __init__(self, *args):
            self.ids, self.depth, self.spec = corpus_ids, 64, {'slots': []}

        def volume(self, uid):
            return np.full((2,), float(corpus_ids.index(uid)), np.float32)

        def mask(self, uid):
            return np.ones(64, np.uint8)

    dicom = ModuleType('raptor_dicom'); dicom.DicomCorpus = Corpus; dicom.DirectoryDicomSource = lambda p: p
    win = lambda vol, centers, res: np.full((36, 3, 2, 2), vol[0], np.float32)
    common = ModuleType('raptor_common'); common.select_centers = lambda *a: list(range(36)); common.windows_from = win
    rad = ModuleType('raptor_radimagenet'); rad.center3_windows_from = win; rad.normalise = lambda x, n: x
    runtime = ModuleType('raptor_runtime'); runtime.normalize_input = lambda x, n: x
    return {'torch': torch, 'raptor_dicom': dicom, 'raptor_common': common,
            'raptor_radimagenet': rad, 'raptor_runtime': runtime}


@pytest.mark.skipif(not STAGES, reason='no staged v2 folders')
def test_patched_partner_flow_runs_one_consumer_and_fills_frozen_layout(monkeypatch):
    base = _staged_base(STAGES[-1], '_exp169_staged_base_flow')
    for name, module in _fake_modules(4).items():
        monkeypatch.setitem(sys.modules, name, module)
    entries = _entries()

    def load_models(work, entries_, device):
        assert device == 'cuda:0'
        return list(range(len(entries_)))

    def infer_one(model, x, device, dino=False):
        return np.full(12, (model + 1) / 100 + float(x.flat[0]) / 1000, np.float32)

    fn, info = refs.local_partner_function(base, SimpleNamespace(_load_models=load_models),
                                           overrides={'_infer_one': infer_one,
                                                      'plane_flags': lambda centers, spec: [k % 2 == 0 for k in centers]})
    ids = ['u2', 'u0', 'u3']
    scores, metrics = fn(Path('.'), Path('.'), ids, {'entries': entries}, 1e18)
    assert metrics['completed'] == [3] and metrics['gpu_peak_bytes'] == [123]
    for m, e in enumerate(entries):
        row = e['fold'] + (5 if e['family'] == 'l2' and e['seed'] == 43 else 0)
        for i, uid in enumerate(ids):
            expected = (m + 1) / 100 + int(uid[1:]) / 1000
            assert np.allclose(scores[e['family']][row, i], expected, atol=1e-7)
    assert all(np.isfinite(a).all() for a in scores.values())
    assert info['patched_sha256'] != info['original_sha256']


@pytest.mark.skipif(not STAGES, reason='no staged v2 folders')
def test_fastread_patches_apply_to_frozen_module_and_shim_reads_binary(tmp_path):
    import ast
    import base64
    serving = next((s / 'own-anchor/assets/public-serving.py' for s in STAGES
                    if (s / 'own-anchor/assets/public-serving.py').is_file()), None)
    if serving is None:
        pytest.skip('no staged own-anchor serving source')
    tree = ast.parse(serving.read_text(encoding='utf-8'))
    blob = next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                any(getattr(t, 'id', '') == '_FASTREAD_B64' for t in n.targets))
    source = base64.b64decode(ast.literal_eval(blob)).decode('utf-8')
    patched, info = refs.patch_source(source, refs.FASTREAD_PATCHES)
    compile(patched, '_fastread.py', 'exec')
    assert patched.count('os.pread(') == source.count('os.pread(') and '_exp169_pread' in patched
    assert 'getattr(os, "O_BINARY", 0)' in patched
    # The shim alone: binary-safe positional reads (CR/LF/Ctrl-Z bytes are not translated).
    helper = refs.FASTREAD_PATCHES[0][1].split('\n\n\nCROP_MM')[0].replace('import numpy as np\n\n', '')
    scope = {}
    exec('import os, threading\n' + helper, scope)
    data = bytes(range(256)) * 8 + b'\r\n\x1a\r\r\n'
    path = tmp_path / 'blob.bin'
    path.write_bytes(data)
    import os
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
    try:
        assert scope['_exp169_pread'](fd, 7, 2048) == data[2048:2055]
        assert scope['_exp169_pread'](fd, 10_000, 0) == data
        assert scope['_exp169_pread'](fd, 4, 10) == data[10:14]
    finally:
        os.close(fd)
