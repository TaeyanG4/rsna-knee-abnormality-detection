"""H-B-173 exact two-parent scope, independent mutation probes and keep gate."""
import copy
import os
import pytest
from tools import c85r7_r224mirnfq_v1129 as b


def test_two_way_exact_bytes_and_flip_off():
    nb, meta, files, blob = b.payload()
    parent = b.authenticated(b.BASE)[1]
    reference = b.authenticated(b.REFERENCE)[1]
    assert len(files) == 31
    assert sorted(k for k in files if files[k] != parent[k]) == sorted([b.SOURCE, b.WRAPPER])
    assert sorted(k for k in files if files[k] != reference[k]) == sorted([b.PORT, b.CHAIN, b.WRAPPER])
    assert files[b.SOURCE] == reference[b.SOURCE]
    assert files[b.PORT] == parent[b.PORT] and b.old.FLAG_OLD.encode() in files[b.PORT]
    assert files == b.reverse_reference(reference)
    assert b.scope(nb, meta) == (files, blob)
    assert meta['id'] == b.SLUG and meta['title'] == b.SLUG.split('/')[1]
    wrapper = files[b.WRAPPER].decode()
    assert 'FLIP_ALL' not in wrapper
    assert 'v' + b.VERSION in wrapper and 'VERSION = "' + b.VERSION + '"' in wrapper


@pytest.mark.parametrize('fault', ['fast', 'bootstrap', 'metadata', 'cell_fields', 'top', 'extra_cell'])
def test_forbidden_notebook_and_metadata_rejected(fault):
    nb, meta, _, _ = b.payload()
    if fault == 'fast': nb['cells'][0]['source'].append('\n# drift\n')
    if fault == 'bootstrap': nb['cells'][-1]['source'].append('\nimport multiprocessing\n')
    if fault == 'metadata': meta['enable_internet'] = not meta.get('enable_internet', False)
    if fault == 'cell_fields': nb['cells'][-1]['metadata']['unexpected'] = 1
    if fault == 'top': nb['metadata']['unexpected'] = 1
    if fault == 'extra_cell': nb['cells'].append(copy.deepcopy(nb['cells'][0]))
    with pytest.raises(ValueError, match='forbidden'):
        b.scope(nb, meta)


@pytest.mark.parametrize('member', [b.PORT, b.SOURCE, 'public-serving.py'])
def test_repacked_and_repinned_semantic_drift_rejected(member):
    nb, _, _, _ = b.authenticated(b.BASE)
    candidate, meta, files, _ = b.payload()
    _, original_blob, assignments = b.old.unpack(b.flipq.restored(nb))
    files[member] += b'\n# repinned unauthorized drift\n'
    changed, _ = b.repack(nb, original_blob, assignments, files)
    with pytest.raises(ValueError, match='forbidden'):
        b.scope(changed, meta)


def test_admission_identity_drift_rejected(monkeypatch):
    original = b.old.read
    def damaged(path):
        value = original(path)
        if path.name == 'knee-lb-r224q-v1.json': value['stage_inventory']['archive_sha256'] = '0' * 64
        return value
    monkeypatch.setattr(b.old, 'read', damaged)
    with pytest.raises(ValueError, match='inventory drift'):
        b.authenticated(b.BASE)


def test_registered_keep_and_nine_hour_gate():
    assert b.decision(.957, .956, 19080, 1737.7479396114036)
    assert not b.decision(.957, .95599, 19080, 1737.7479396114036)
    assert b.decision(.957, .958, 30000, 2400)
    assert not b.decision(.957, .958, 30000, 2400.001)
    for bad in [float('nan'), float('inf'), -1, 0]:
        with pytest.raises(ValueError): b.decision(.957, .958, 19080, bad)
    rec = b.registration()
    assert rec['flip_tta'] is False and rec['qualified_for_submission'] is False
    assert rec['public_reading']['keep_min'] == .956
    assert rec['timing']['runtime_limit_hours'] == 9
    assert rec['timing']['measured_candidate_runtime'] is False
    assert rec['exposure'] == 'EXPOSED' and rec['clean_oof'] is False


def test_full_inventory_both_bases():
    values, files, _ = b.artifacts()
    for name in ('v1.1.23-r224q', 'v1.1.26-r224mirq'):
        manifest = b.json.loads(values['inventory-vs-' + name + '.json'])
        assert manifest['added'] == manifest['removed'] == []
        assert sum(k.startswith('embedded/') for k in manifest['inventory']) == 31
        assert 'notebook/top.json' in manifest['inventory']
        assert 'kernel/kernel-metadata.json' in manifest['inventory']
        assert all(k.encode() in values['full-diff-vs-' + name + '.txt'] for k in files)


def test_empty_cuda_visibility():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
