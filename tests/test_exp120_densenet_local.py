"""CPU-only EXP120 protocol and DenseNet loader checks."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp120_densenet_local as exp
from tools import exp113_radr50_local as base


def test_checkpoint_map_reports_every_unmapped_key():
    mapped, report = exp.map_densenet_state({'backbone.0.conv0.weight': 1,
                                              'backbone.0.norm0.weight': 2,
                                              'classifier.weight': 3},
                                             {'features.conv0.weight', 'features.norm0.weight', 'features.norm0.bias'})
    assert mapped == {'features.conv0.weight': 1, 'features.norm0.weight': 2}
    assert report['unmapped_keys'] == ['classifier.weight']
    assert report['missing_keys'] == ['features.norm0.bias']


def test_f0_screen_gate(monkeypatch, tmp_path):
    fake = {'folds': {str(k): {} for k in exp.FOLDS}, 'common_args': {}, 'paths': {}}
    monkeypatch.setattr(exp, 'read', lambda p: fake if Path(p).name == 'manifest.json' else json.loads(Path(p).read_text()))
    monkeypatch.setattr(exp, 'runner_argv', lambda p, fold, **kwargs: ['python', str(fold)])
    monkeypatch.setattr(exp, 'sha256', lambda p: 'protocol-hash')
    jobs = exp.queue_entries((0,), after='TAIL')
    assert len(jobs) == 1 and jobs[0]['tag'] == exp.tag_for(0)
    assert jobs[0]['after'] == 'TAIL' and jobs[0]['continue_on_failure'] is True
    with pytest.raises(ValueError, match='screen-pass'):
        exp.queue_entries((1,2,3,4), after='TAIL')
    record = tmp_path / 'pass.json'
    record.write_text(json.dumps(dict(decision='PASS', fold=0, protocol_sha256='protocol-hash',
                                      evidence='fold-0 paired screen artifact', recorded_at='2026-10-08T03:00:00+09:00')))
    jobs = exp.queue_entries((1,2,3,4), after='TAIL', screen_pass=record)
    assert len(jobs) == 4 and all(j['continue_on_failure'] for j in jobs)
    assert jobs[1]['after'] == jobs[0]['tag']
    record.write_text(record.read_text().replace('protocol-hash','wrong-hash'))
    with pytest.raises(ValueError, match='recorded F0 PASS'):
        exp.queue_entries((1,), screen_pass=record)


def test_runner_parser_accepts_only_imagenet_and_exp120_guard():
    spec = importlib.util.spec_from_file_location('exp120_runner_test', ROOT / exp.RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_parser()._option_string_actions['--norm'].choices == ['imagenet']
    assert mod.build_parser()._option_string_actions['--pretrained'].choices == ['radimagenet']
    source = (ROOT / exp.RUNNER).read_text(encoding='utf-8-sig')
    assert 'from tools import exp120_densenet_local as guard' in source
    assert 'guard.build_densenet_model' in source
    assert 'from raptor_radimagenet import CenterSliceWindows' in source


@pytest.mark.skipif(not (ROOT / exp.WEIGHTS_LOCAL).is_file(), reason='DenseNet121 weights absent')
def test_real_checkpoint_strict_map_and_cpu_only(monkeypatch):
    import torch
    monkeypatch.setenv('OMP_NUM_THREADS','2')
    model, report = exp.build_densenet_model(exp.ARCH, ROOT / exp.WEIGHTS_LOCAL)
    assert report['source_keys'] == 725
    assert report['mapped_keys'] == report['target_keys']
    assert not report['unmapped_keys'] and not report['missing_keys']
    assert report['byte_equal_after_load']
    assert not torch.cuda.is_initialized()
    assert model.backbone.num_features > 0


@pytest.mark.skipif(not exp.manifest_path().is_file(), reason='run EXP120 prepare first')
def test_prepared_manifest_keeps_exp113_recipe_and_binds_new_code():
    p = exp.read(exp.manifest_path())
    b = exp.read(ROOT / base.PROTOCOL / 'manifest.json')
    assert p['common_args'] == {**b['common_args'], 'arch':'densenet121', 'norm':'imagenet'}
    assert p['seed'] == 42 and p['label_set'] == b['label_set']
    assert p['paths'] == b['paths'] and p['fold_protocol'] == b['fold_protocol']
    assert p['init_weights']['sha256'] == exp.WEIGHTS_SHA256
    assert p['input']['norm'] == 'imagenet'
    assert p['initial_state_sha256'] == exp.read(ROOT / p['initial_state']['keys_file'])['combined_sha256']
    for fold in exp.FOLDS:
        now, old = p['folds'][str(fold)], b['folds'][str(fold)]
        assert {k:v for k,v in now.items() if k != 'tag'} == {k:v for k,v in old.items() if k != 'tag'}
        assert now['tag'] == exp.tag_for(fold)
    for path, digest in p['source_bindings'].items():
        assert exp.sha256(ROOT / path) == digest


@pytest.mark.skipif(not exp.manifest_path().is_file(), reason='run EXP120 prepare first')
def test_cpu_validate_fold_zero_no_cuda(monkeypatch):
    import torch
    for key,val in {**exp.THREAD_ENV, 'HF_HUB_OFFLINE':'1'}.items():
        monkeypatch.setenv(key,val)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV',raising=False)
    p = exp.read(exp.manifest_path())
    argv = exp.runner_argv(p,0)
    spec = importlib.util.spec_from_file_location('exp120_runner_validate', ROOT / exp.RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args = runner.build_parser().parse_args(argv[2:])
    got, fold, ordered, _ = exp.validate_arguments(args, ROOT)
    assert got['schema'] == exp.SCHEMA and fold['tag'] == exp.tag_for(0)
    assert len(ordered['val']) == fold['n_val']
    assert not torch.cuda.is_initialized()
