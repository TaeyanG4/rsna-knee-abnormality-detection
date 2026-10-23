"""CPU-only release-contract checks. No torch import or CUDA initialization."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import exp140_aug1_release as m


def test_fold_seed_provenance_and_inventory(tmp_path):
    tag = m.tag(42, 1)
    run = tmp_path / 'artifacts/experiments' / tag
    run.mkdir(parents=True)
    model = run / 'model_best.pt'
    model.write_bytes(b'tiny checkpoint')
    proto = 'a' * 64
    m.save(run / 'summary.json', {
        'fold': 1, 'mode': 'full', 'selected_checkpoint': 'fixed-final',
        'worker_cleanup': {'clean': True}, 'total_seconds': 2,
        'provenance': {'seed': 42, 'fold': 1, 'protocol_sha256': proto}})
    provenance = {'seed': 42, 'fold': 1, 'exp127_variant': 'AUG1',
                  'protocol_sha256': proto, 'arch': 'resnet50',
                  'norm': 'radimagenet', 'input_policy': 'center3'}
    m.save(run / 'provenance.json', provenance)
    m.save(run / 'output-inventory.json', {name: {
        'sha256': m.digest(run / name), 'bytes': (run / name).stat().st_size}
        for name in ('model_best.pt', 'summary.json', 'provenance.json')})
    entry = m.checkpoint_entry(tmp_path, 42, 1)
    assert entry['path'] == 'checkpoints/rad42_f1.pt'
    assert entry['protocol_sha256'] == proto
    provenance['seed'] = 43
    m.save(run / 'provenance.json', provenance)
    with pytest.raises(ValueError, match='receipt inventory'):
        m.checkpoint_entry(tmp_path, 42, 1)


def test_staging_rejects_tampered_source(tmp_path):
    root = tmp_path / 'build'
    root.mkdir()
    source = tmp_path / 'model.pt'
    source.write_bytes(b'first')
    m.save(root / 'upload-manifest.json', {'files': [{
        'source': str(source), 'destination': 'checkpoints/rad42_f0.pt',
        'sha256': m.digest(source), 'bytes': source.stat().st_size}]})
    source.write_bytes(b'other')
    with pytest.raises(ValueError, match='Source hash'):
        m.stage(root)
    source.write_bytes(b'first')
    assert m.stage(root)['status'] == 'STAGED_HASH_VERIFIED'
    assert (root / 'upload-stage/checkpoints/rad42_f0.pt').read_bytes() == b'first'


def test_b_notebook_only_rad_binding_cell_changes():
    old_raw = m.OLD.read_text(encoding='utf-8')
    fake = {'schema': 'exp133-rad-assets-v1', 'dataset': m.DATASET,
            'checkpoints': [{'seed': 42, 'fold': 0, 'sha256': 'a' * 64}]}
    changed = m.b_notebook(old_raw, fake)
    before, after = json.loads(old_raw), json.loads(changed)
    assert after['cells'][:-1] == before['cells'][:-1]
    assert changed.replace(json.dumps(after['cells'][-1]['source'], ensure_ascii=False)[1:-1],
                           json.dumps(before['cells'][-1]['source'], ensure_ascii=False)[1:-1], 1) == old_raw
    tail = after['cells'][-1]['source']
    assert m.old_payload_literal(tail)[1]['rad_manifest'] == fake
    assert hashlib.sha256(old_raw.encode()).hexdigest() == m.BASE_SHA


def test_rank_reduction_is_exp133_rule():
    # Two seeds of fold-mean probabilities; rank each seed per target over the
    # entire cohort, then average their percentile ranks. Ties are averaged.
    a = [[0.1] * 12, [0.1] * 12, [0.9] * 12]
    b = [[0.9] * 12, [0.2] * 12, [0.2] * 12]
    ranked = m.e133.e128.rank_blend(a, b, (.5, .5))
    assert ranked[0][0] == pytest.approx((.5 + 1) / 2)
    assert ranked[1][0] == pytest.approx((.5 + .5) / 2)
    assert ranked[2][0] == pytest.approx((1 + .5) / 2)
    assert all(0 <= value <= 1 for row in ranked for value in row)


def test_standalone_public3_receipt_reconstructs_tiny_cohort(tmp_path):
    ids = ['study-a', 'study-b', 'study-c']
    sample = tmp_path / 'sample.csv'
    m.e133.write_csv(sample, ids, [[0.0] * 12 for _ in ids])
    output = tmp_path / 'output'
    (output / 'exp133_rad').mkdir(parents=True)
    a = [[0.1] * 12, [0.5] * 12, [0.9] * 12]
    b = [[0.9] * 12, [0.5] * 12, [0.1] * 12]
    for seed, values in ((42, a), (43, b)):
        m.e133.write_csv(output / 'exp133_rad' / ('seed%d.csv' % seed), ids, values)
    values = m.e133.e128.rank_blend(a, b, (.5, .5))
    m.e133.write_csv(output / 'exp133_rad.csv', ids, values)
    m.e133.write_csv(output / 'submission.csv', ids, values)
    m.save(output / 'exp140_stage_timing.json', {
        'schema': 'exp140-aug1-rad-standalone-v1', 'status': 'PASS',
        'submission_sha256': m.digest(output / 'submission.csv'),
        'worker': {'status': 'PASS', 'studies': 3, 'csv_sha256': {
            str(seed): m.digest(output / 'exp133_rad' / ('seed%d.csv' % seed))
            for seed in (42, 43)}}})
    assert m.validate_public3('A', output, sample)['status'] == 'PASS_A_PUBLIC3'
    (output / 'submission.csv').write_text('tampered', encoding='utf-8')
    with pytest.raises((ValueError, StopIteration)):
        m.validate_public3('A', output, sample)


def test_built_notebook_cells_compile_and_metadata():
    out = m.OUT
    for directory, slug, sources in [
        ('rad_standalone', 'knee-exp140-radr50-aug1', [m.DATASET]),
        ('blend_b3_aug1', 'knee-exp140-b3-aug1',
         [m.e133.e128.DATASET, m.e133.e128.READER_DATASET, m.DATASET]),
    ]:
        kernel = out / directory
        nb = json.loads((kernel / 'release.ipynb').read_text(encoding='utf-8'))
        for index, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '%s cell %d' % (directory, index), 'exec')
        meta = json.loads((kernel / 'kernel-metadata.json').read_text(encoding='utf-8'))
        assert meta['id'] == 'taeyangg4/' + slug
        assert meta['dataset_sources'] == sources
        assert meta['competition_sources'] == ['rsna-knee-abnormality-detection']
        assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
        assert meta['machine_shape'] == 'NvidiaTeslaT4'
        assert meta['docker_image'] == m.e133.e128.DOCKER
    assert m.runtime_projection()['margin_seconds'] > 0
    actual_manifest = json.loads((out / 'upload-stage' / m.ASSET_NAME).read_text(encoding='utf-8'))
    expected_b = m.b_notebook(m.OLD.read_text(encoding='utf-8'), actual_manifest)
    assert (out / 'blend_b3_aug1/release.ipynb').read_text(encoding='utf-8') == expected_b
    b_receipt = json.loads((out / 'blend_b3_aug1/build-receipt.json').read_text(encoding='utf-8'))
    assert b_receipt['registered_weights'] == m.e133.WEIGHTS
    assert b_receipt['rad_manifest_sha256'] == hashlib.sha256(
        json.dumps(actual_manifest, sort_keys=True).encode()).hexdigest()
