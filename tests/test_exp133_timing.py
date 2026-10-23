"""CPU-only timing-builder tests; import and code generation never import torch."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import exp133_timing as timing


def test_original_release_immutable_and_all_cells_compile(tmp_path):
    release = timing.RELEASE / 'release.ipynb'
    before = release.read_bytes()
    out = tmp_path / 'timing'
    receipt = timing.build(out)
    assert release.read_bytes() == before
    assert receipt['released_notebook_sha256'] == hashlib.sha256(before).hexdigest()
    nb = json.loads((out / 'release.ipynb').read_text())
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), str(i), 'exec')
    old = json.loads((timing.RELEASE / 'kernel-metadata.json').read_text())
    new = json.loads((out / 'kernel-metadata.json').read_text())
    for key in ('dataset_sources', 'competition_sources', 'docker_image', 'machine_shape'):
        assert new[key] == old[key]
    assert new['is_private'] and not new['enable_internet']
    assert new['id'] == 'taeyangg4/knee-exp133-b3-timing60'
    with pytest.raises(ValueError, match='refusing overwrite'):
        timing.build(out)


def test_nested_reader_child_compiles():
    tree = ast.parse(timing.reader_source())
    child = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'child' for t in n.targets)]
    assert len(child) == 1
    compile(ast.literal_eval(child[0].value), '<reader child>', 'exec')


def test_rad_source_compiles_and_keeps_exact_serving_calls():
    source = timing.rad_source()
    compile(source, '<rad>', 'exec')
    assert "tta.predict(model, corpus, [uid], 36, 224, 'fp16', 'cuda'" in source
    assert "with v6.serving_input(tta, 'center3')" in source
    assert "window_policy='within-slot-valid'" in source
    assert "'submission.csv'" not in source


def test_measured_projection_and_invalid_attribution():
    stages = {name: dict(studies=[dict(uid=str(i), decode_seconds=2., forward_seconds=3.)
                                 for i in range(58)], steady_seconds=580.,
                        peak_gpu_allocated_bytes=[123]) for name in ('oai','rad','cnx')}
    result = timing.projection(stages)
    assert result['total_seconds'] == 42900 + 2 * timing.b3.FIXED_S
    assert not result['can_start_all']
    stages['cnx']['studies'][0]['forward_seconds'] = 0
    with pytest.raises(ValueError, match='Unattributed'):
        timing.projection(stages)


def test_shadow_only_reads_series_metadata():
    source = timing.runtime_setup(['u'])
    compile(source, '<shadow>', 'exec')
    assert "_comp/'train_series.csv'" in source
    assert "_comp/'train_series'" in source
    assert 'train.csv' not in source


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., 0.])
def test_projection_rejects_invalid_rates(bad):
    stages = {name: dict(studies=[dict(uid=str(i), decode_seconds=2., forward_seconds=3.)
                                 for i in range(58)], steady_seconds=580.,
                        peak_gpu_allocated_bytes=[123]) for name in ('oai', 'rad', 'cnx')}
    stages['rad']['steady_seconds'] = bad
    with pytest.raises(ValueError, match='Invalid measured rate'):
        timing.projection(stages)
    stages['rad']['steady_seconds'] = 580.
    stages['cnx']['studies'][0]['decode_seconds'] = bad
    with pytest.raises(ValueError, match='Unattributed'):
        timing.projection(stages)


def test_projection_rejects_cohort_mismatch():
    stages = {name: dict(studies=[dict(uid=str(i), decode_seconds=2., forward_seconds=3.)
                                 for i in range(58)], steady_seconds=580.,
                        peak_gpu_allocated_bytes=[123]) for name in ('oai', 'rad', 'cnx')}
    stages['cnx']['studies'][0]['uid'] = 'other'
    with pytest.raises(ValueError, match='cohort/order differs'):
        timing.projection(stages)


def test_oai_keeps_inference_implementation_and_never_publishes():
    nb = json.loads((timing.RELEASE / 'release.ipynb').read_text(encoding='utf-8'))
    original = next(''.join(c['source']) for c in nb['cells']
                    if c['cell_type'] == 'code' and 'def main():' in ''.join(c['source']))
    source = timing.oai_source(original)
    assert 'from __future__ import' not in original
    assert 'ROOT = str(_TIMING_SHADOW)' in source
    assert 'if os.path.exists(ssub):' in source
    assert 'test_ids = test["StudyInstanceUID"].tolist()' in source
    before = ast.parse(original)
    after = ast.parse(source)
    names = ['build_study', 'gpu_windows', 'infer_pair_probs', 'load_model', '_eval_centers', '_mirror_tri']
    for name in names:
        get = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(get(before)) == ast.dump(get(after))
    assert '/kaggle/working/submission.csv' not in source
    assert 'timing-oai.json' in source


def test_reader_hooks_with_distinct_cpu_cuda_fakes(tmp_path, monkeypatch):
    cuda = types.SimpleNamespace(
        synchronize=lambda *a: None,
        reset_peak_memory_stats=lambda *a: None,
        max_memory_allocated=lambda *a: 123,
        empty_cache=lambda: None,
        device_count=lambda: 2,
        get_device_name=lambda *a: 'fake T4')
    assert len({id(v) for v in vars(cuda).values()}) == len(vars(cuda))
    class Module:
        def _call_impl(self, value):
            return value + 1
    class Dataset:
        @property
        def pixel_array(self):
            return [1, 2, 3]
    torch = types.SimpleNamespace(cuda=cuda, nn=types.SimpleNamespace(Module=Module))
    pydicom = types.SimpleNamespace(dcmread=lambda path: Dataset(),
                                   dataset=types.SimpleNamespace(Dataset=Dataset))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'pydicom', pydicom)
    tree = ast.parse(timing.reader_source())
    child = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'child' for t in n.targets))
    source = ast.literal_eval(child.value)
    hooks = source[source.index('import functools'):source.index('spec.loader.exec_module(module)')]
    hooks = hooks.replace('/kaggle/working/timing-cnx-events.jsonl', (tmp_path/'events.jsonl').as_posix())
    namespace = dict(json=json, pathlib=__import__('pathlib'), ids=json.dumps(['u', 'v']))
    exec(compile(hooks, '<CPU fake hooks>', 'exec'), namespace)
    def run_one(uid):
        result = pydicom.dcmread('/train_series/'+uid+'/x.dcm')
        assert result.pixel_array == [1, 2, 3]
        assert Module()._call_impl(5) == 6
    run_one('u')
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert len(events) == 3
    assert all(event['uids'] == ['u'] for event in events)
    assert [e['kind'] for e in events] == ['decode_seconds', 'decode_seconds', 'forward_seconds']

