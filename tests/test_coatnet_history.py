"""Historical CoAtNet v1.0.0: immutable provenance and fail-closed completion."""
import copy
from pathlib import Path

import pytest

from training.arms import coatnet_history as h


def manifests():
    return h._index(h.ROOT)[1]['manifests']


@pytest.mark.parametrize('manifest', manifests(), ids=lambda m: m['path'].split('/')[2]+'-'+m['arm'])
def test_all_historical_registrations_and_cpu_validation(manifest):
    p = h.read_registration(manifest['path'])
    v = h.require_validation(manifest['path'])
    assert p['arm'] == manifest['arm']
    assert v['status'] == 'VALIDATED_NO_TRAINING'


@pytest.mark.parametrize('manifest', manifests(), ids=lambda m: m['path'].split('/')[2]+'-'+m['arm'])
def test_real_completed_artifacts_or_explicit_incomplete_block(manifest, monkeypatch):
    import torch
    def no_cuda(*args, **kwargs):
        raise AssertionError('historical verifier must remain CPU-only')
    monkeypatch.setattr(torch.cuda, '_lazy_init', no_cuda)
    p = h.read_registration(manifest['path'])
    if p['run_dirs']['local'] not in h.INVENTORIES:
        with pytest.raises(ValueError, match='no authenticated complete output inventory'):
            h.verified_arm(manifest['path'])
    else:
        record = h.verified_arm(manifest['path'])
        assert record['historical_only'] and not record['launch_authorized']
        assert record['protocol_sha256'] == manifest['sha256']


@pytest.mark.parametrize('part,message', [
    ('index.json', 'archive index changed'),
    ('manifest-CONTROL.json', 'manifest bytes changed'),
    ('blob', 'source archive blob changed'),
    ('validation.json', 'CPU validation changed'),
    ('output-inventory.json', 'output inventory changed'),
    ('summary.json', 'inventory member changed'),
])
def test_independent_hash_boundaries_fail_closed(part, message, monkeypatch):
    original = h.sha256
    def changed(path):
        path = Path(path)
        corrupt = ('blobs' in path.parts) if part == 'blob' else path.name == part
        return '0'*64 if corrupt else original(path)
    monkeypatch.setattr(h, 'sha256', changed)
    with pytest.raises(ValueError, match=message):
        h.verified_arm(h.CONTROL)


def test_source_archive_does_not_follow_live_wrapper_edits(monkeypatch):
    original = h.sha256
    protected = {h.ROOT/'tools/exp181_coatnet_f0_bridge.py', h.ROOT/'tools/exp188_coatnet_coverage_f0.py'}
    def no_live_source(path):
        assert Path(path).resolve() not in protected
        return original(path)
    monkeypatch.setattr(h, 'sha256', no_live_source)
    assert h.read_registration(h.CONTROL)['arm'] == 'CONTROL'


@pytest.mark.parametrize('artifact,field', [('summary.json', 'identity'), ('provenance.json', 'seed')])
def test_provenance_scientific_identity_is_checked(artifact, field, monkeypatch):
    original = h.read_json
    def changed(path):
        value = original(path)
        if Path(path).name == artifact:
            value = copy.deepcopy(value)
            if field == 'identity':
                value[field]['seed'] = 43
            else:
                value[field] = 43
        return value
    monkeypatch.setattr(h, 'read_json', changed)
    with pytest.raises(ValueError, match='summary differs|provenance differs'):
        h.verified_arm(h.CONTROL)


def test_checkpoint_identity_not_only_protocol_hash(monkeypatch):
    import torch
    original = torch.load
    def changed(*args, **kwargs):
        ck = original(*args, **kwargs)
        ck['provenance'] = dict(ck['provenance'], precision='fp16')
        return ck
    monkeypatch.setattr(torch, 'load', changed)
    with pytest.raises(ValueError, match='checkpoint provenance differs'):
        h.verified_arm(h.CONTROL)


def test_rejects_unregistered_path_even_for_identical_bytes(tmp_path):
    path = tmp_path/'manifest.json'
    path.write_bytes((h.ROOT/h.CONTROL).read_bytes())
    with pytest.raises(ValueError, match='root|unregistered historical manifest'):
        h.read_registration(path)


@pytest.mark.parametrize('bad', ['nan', 'inf', '-0.1', '1.1'])
def test_probability_range_and_finiteness(tmp_path, bad):
    path = tmp_path/'oof.csv'
    path.write_text(','.join(['StudyInstanceUID', *h.TARGETS])+'\n' +
                    ','.join(['fixture', bad, *(['0.5']*11)])+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='probabilities invalid'):
        h.load_oof(path, ['fixture'])
