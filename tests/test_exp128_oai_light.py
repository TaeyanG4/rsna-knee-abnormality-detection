"""CPU-only contract tests; no public notebook code is executed."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import time


PATH = Path(__file__).resolve().parents[1] / 'tools/exp128_oai_light.py'
SPEC = importlib.util.spec_from_file_location('exp128_oai_light_test', PATH)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def write_csv(path, ids, values):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([mod.UID, *mod.TARGETS])
        for uid, value in zip(ids, values):
            w.writerow([uid, *([value] * len(mod.TARGETS))])


def test_public_code_cells_and_credit_are_preserved():
    source, code = mod.load_source()  # JSON parsing only; no third-party execution
    for variant in mod.KERNELS:
        nb = mod.notebook(variant, source, code, {} if variant == 'blend_b2' else None)
        assert nb['cells'][5:7] == code
        assert nb['cells'][2:5] == source['cells'][:3]
        compile(nb['cells'][-1]['source'], '<exp128-generated-tail>', 'exec')
        credit = ''.join(nb['cells'][1]['source'])
        assert all(name in credit for name in ('nartaa', 'dreaddevelopment', 'OAI',
                                               'goodpjw2008', 'Apache License', 'licences=[other]'))
        assert 'anchor-a' not in json.dumps(nb['metadata'])


def test_average_ties_are_percentiles_and_weights_are_uniform():
    assert mod.rank_average([3, 1, 1, 4]) == [0.75, 0.375, 0.375, 1.0]
    first = [[0.1] * 12, [0.5] * 12, [0.9] * 12]
    second = [[0.9] * 12, [0.5] * 12, [0.1] * 12]
    rows = mod.rank_blend(first, second, mod.WEIGHTS['blend_b1'])
    assert all(row == [row[0]] * 12 for row in rows)
    assert rows[0][0] == (0.85 * (1/3) + 0.15 * 1)
    assert mod.WEIGHTS == {'blend_b1': (0.85, 0.15), 'blend_b2': (0.9, 0.1)}


def test_private_metadata_and_sources():
    for variant in mod.KERNELS:
        meta = mod.metadata(variant)
        assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
        assert meta['machine_shape'] == 'NvidiaTeslaT4'
        assert meta['docker_image'] == mod.DOCKER
        assert meta['competition_sources'] == [mod.COMPETITION]
        assert mod.DATASET in meta['dataset_sources']
    assert mod.READER_DATASET in mod.metadata('blend_b1')['dataset_sources']
    assert mod.READER_DATASET not in mod.metadata('blend_b2')['dataset_sources']


def test_compare_three_study_csv(tmp_path):
    ids = ['a', 'b', 'c']
    ours, public = tmp_path / 'ours.csv', tmp_path / 'public.csv'
    write_csv(ours, ids, [0.1, 0.2, 0.3])
    write_csv(public, ids[::-1], [0.3, 0.2, 0.1])
    assert mod.compare_csv(ours, public)['exact'] is True
    write_csv(public, ids, [0.1, 0.2, 0.31])
    result = mod.compare_csv(ours, public)
    assert result['mismatch_count'] == 12
    assert result['studies'] == 3


def test_guard_falls_back_byte_exact_on_reader_failure(tmp_path, monkeypatch):
    final = tmp_path / 'submission.csv'
    write_csv(final, ['a', 'b', 'c'], [0.1, 0.2, 0.3])
    before = final.read_bytes()
    def fail(*_args):
        raise TimeoutError('synthetic cutoff')
    monkeypatch.setattr(mod, '_reader_stage', fail)
    result = mod.run_after_oai('blend_b1', start=time.monotonic(), working=tmp_path)
    assert result['path'] == 'pure_oai'
    assert result['reason'].startswith('TimeoutError:')
    assert final.read_bytes() == before
    assert (tmp_path / 'exp128_blend_receipt.json').is_file()


def test_guard_falls_back_byte_exact_on_rad_failure(tmp_path, monkeypatch):
    final = tmp_path / 'submission.csv'
    write_csv(final, ['a', 'b', 'c'], [0.1, 0.2, 0.3])
    before = final.read_bytes()
    monkeypatch.setattr(mod, '_rad_stage', lambda *_args: (_ for _ in ()).throw(RuntimeError('synthetic worker failure')))
    result = mod.run_after_oai('blend_b2', payload={}, start=time.monotonic(), working=tmp_path)
    assert result['path'] == 'pure_oai'
    assert final.read_bytes() == before


def test_guarded_success_and_runtime_projection(tmp_path, monkeypatch):
    final = tmp_path / 'submission.csv'
    write_csv(final, ['a', 'b', 'c'], [0.1, 0.2, 0.3])
    monkeypatch.setattr(mod, '_reader_stage', lambda *_args: [[0.3]*12, [0.2]*12, [0.1]*12])
    result = mod.run_after_oai('blend_b1', start=time.monotonic(), working=tmp_path)
    assert result['path'] == 'blended'
    assert mod.read_submission(final)[0] == ['a', 'b', 'c']
    assert mod.runtime_projection(1300)['oai_0949_seconds'] is None
    assert mod.runtime_projection(1300)['rad_b2']['five_model_plain_screen'] == 2112.5


def test_registered_cutoff_and_diagnostic_failure_keep_pure_csv(tmp_path, monkeypatch):
    final = tmp_path / 'submission.csv'
    write_csv(final, ['a', 'b', 'c'], [0.1, 0.2, 0.3])
    before = final.read_bytes()
    monkeypatch.setattr(mod, '_reader_stage', lambda *_args: (_ for _ in ()).throw(AssertionError('stage ran')))
    monkeypatch.setattr(mod, 'write_json', lambda *_args: (_ for _ in ()).throw(OSError('synthetic receipt failure')))
    result = mod.run_after_oai('blend_b1', start=time.monotonic() - mod.CUTOFF_S, working=tmp_path)
    assert result['path'] == 'pure_oai'
    assert 'cutoff' in result['reason']
    assert final.read_bytes() == before


def test_builder_stages_three_private_kernels_without_execution(tmp_path):
    out = tmp_path / 'kernels'
    report = mod.build(out)
    assert report['status'] == 'UNPUSHED'
    assert set(report['kernels']) == set(mod.KERNELS)
    for variant in mod.KERNELS:
        path = out / variant
        nb = json.loads((path / 'release.ipynb').read_text(encoding='utf-8'))
        meta = json.loads((path / 'kernel-metadata.json').read_text(encoding='utf-8'))
        spec = json.loads((path / 'submission_screen_spec.json').read_text(encoding='utf-8'))
        assert meta['is_private'] and meta['enable_internet'] is False
        assert nb['cells'][5:7] == mod.load_source()[1]
        assert spec['basis'] == 'PUBLIC_ONLY'
    assert report['kernels']['blend_b2']['public_code_cells_unchanged']
