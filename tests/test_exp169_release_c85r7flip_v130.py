"""Sealed fold policy, native math, atomic integration, and unchanged closure."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import exp169_release_c85r7flip_v130 as b
from tools import exp169_r7_selective_flip_v130 as f
from tools import exp169_c85r7flip_cpu_v130 as smoke
from tools import exp169_release_c85r7codec_deadline_v122 as base
from tools import hb149_r7_flip_v120 as hb


@pytest.mark.parametrize('variant', f.VARIANTS)
@pytest.mark.parametrize('reversed_order', [False, True])
def test_same_input_native_hb149_logits_and_states(tmp_path, variant, reversed_order):
    import torch
    torch.set_num_threads(2)
    runtime = hb.runtime_source(tmp_path/'hb', dual=True)
    reference = (runtime/'compact_infer.py').read_text(encoding='utf-8')
    source = f.patch_worker(reference, variant)
    labels = hb.LABELS
    spec = dict(categories={k:[0,1] for k in labels}, prior={k:[.5,.5] for k in labels},
                binaries={'b': labels})
    models = [SimpleNamespace(fold=k, _c85r7_flip=k in f.VARIANTS[variant],
                              state_k=[2]*12, bin_names=[('b', k) for k in labels]) for k in range(4)]
    if reversed_order:
        models.reverse()
    image = torch.arange(1*3*3*4*5).reshape(1,3,3,4,5)
    meta = torch.zeros(1,3,9)
    meta[0,:,7] = torch.tensor([0,1,2])
    calls = []
    def one_pass(model, x, mt, mask):
        calls.append(model.fold)
        offset = x[0,0,0,0,0].float() + model.fold
        z = torch.arange(12).reshape(1,12).float()+offset
        sl = torch.arange(24).reshape(1,12,2).float()*.01+offset*.02
        return z, sl, z*.03
    result = smoke.compare(source, reference, models, image, meta, spec, one_pass)
    assert result['logits_exact'] and result['state_heads_exact'] and result['mcl_plain_exact']
    for k in range(4):
        assert calls.count(k) == 4 + int(k in f.VARIANTS[variant])


@pytest.mark.parametrize('variant', f.VARIANTS)
def test_atomic_worker_adapter_and_restore(tmp_path, variant):
    with base.metadata_io.writers(base):
        originals = base.r7.diagnostic_source, base.r7.worker_args, base.r7.write_provenance
        with f.installed(base.r7, variant, base.metadata_io):
            rt = base.r7.diagnostic_source(base.r7.SOURCE, tmp_path)
            src = (rt/'compact_infer.py').read_text(encoding='utf-8')
            assert '_c85r7_flip' in src and 'publication_binding' in src
            assert all(b.digest((rt/k).read_bytes()) == v for k,v in b.read(rt/'runtime_sha256.json').items())
        assert originals == (base.r7.diagnostic_source, base.r7.worker_args, base.r7.write_provenance)


@pytest.mark.parametrize('variant', f.VARIANTS)
def test_checkpoint_filename_bound_load_and_reject_roster(tmp_path, variant):
    import torch
    src = f.patch_worker((base.r7.SOURCE/'compact_infer.py').read_text(encoding='utf-8'), variant)
    for name in f.ROSTER:
        (tmp_path/name).touch()
    class Model:
        def __init__(self, *args, **kwargs): pass
        def load_state_dict(self, *args, **kwargs): pass
        def eval(self): return self
        def requires_grad_(self, *args): return self
        def to(self, *args, **kwargs): return self
    ns = smoke.extract(src, ['load_group'], dict(Path=Path, torch=SimpleNamespace(load=lambda *a,**kw:{},
        channels_last=torch.channels_last), CT=SimpleNamespace(Net=Model, OFL_BACKBONE='ofl'),
        a=SimpleNamespace(init=None, freeze_blocks=0, slices=3), state_spec={}, N_EXTENT=[0],
        N_SYN=[0], net_has_syn=True, dev='cpu'))
    paths, models, _ = ns['load_group'](tmp_path,'E_lrprior','convnext_small',352)
    assert [(p.name,m._c85r7_flip) for p,m in zip(paths,models)] == [(n,n in f.policy(variant)['flip_models']) for n in f.ROSTER]
    (tmp_path/'E_lrprior_s1_f0.pt').touch()
    with pytest.raises(ValueError, match='roster'):
        ns['load_group'](tmp_path,'E_lrprior','convnext_small',352)


def test_anchor_drift_rejected():
    source = (base.r7.SOURCE/'compact_infer.py').read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='anchor drift'):
        f.patch_worker(source.replace('def score_v4c(', 'def changed('), 'FLIP_ALL')


def test_only_native_score_condition_changes():
    source = (base.r7.SOURCE/'compact_infer.py').read_text(encoding='utf-8')
    patched = f.patch_worker(source, 'FLIP_F0F2')
    original = smoke.function(source,'score_v4c')
    modified = smoke.function(patched,'score_v4c')
    test = next(n for n in ast.walk(modified) if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp))
    test.test = ast.parse('FLIP is None', mode='eval').body
    assert ast.dump(original) == ast.dump(modified)
    for name in ('mirror_perm','mirror_windows','state_readout','score','apply_stacker','apply_state_stacker'):
        assert ast.dump(smoke.function(source,name)) == ast.dump(smoke.function(patched,name))


def test_sealed_packages_keep_all_base_files_and_settings(tmp_path):
    original = b.frozen_files()
    for variant in f.VARIANTS:
        artifacts, files = b.payloads(variant)
        assert all(files[k] == v for k,v in original.items())
        wrapper = files[b.MODULE+'.py'].decode()
        assert f"FLIP_VARIANT = '{variant}'" in wrapper
        assert "selective_flip.installed(r7, FLIP_VARIANT, metadata_io)" in wrapper
        assert "sealed runtime: use the v1.3.0 builder" in wrapper
        for kind in ('kernel','stress-kernel-clean','stress-kernel'):
            meta = json.loads(artifacts[kind+'/kernel-metadata.json'])
            prior = b.read(b.BASE/kind/'kernel-metadata.json')
            assert {k:v for k,v in meta.items() if k not in ('id','title')} == {k:v for k,v in prior.items() if k not in ('id','title')}
            assert meta['is_private'] and not meta['enable_internet']
    b.prepare(tmp_path)
    assert b.validate(tmp_path)['base_files_unchanged'] == 34
    b.register(tmp_path)
    damaged = tmp_path/'FLIP_ALL/kernel/release.ipynb'
    damaged.write_bytes(damaged.read_bytes()+b' ')
    with pytest.raises(ValueError, match='drift'):
        b.validate(tmp_path)


def test_projection_and_fixed_gate():
    half, full = (b.runtime_projection(v) for v in f.VARIANTS)
    assert half['projected_hours'] == pytest.approx(6.7931827444)
    assert full['projected_hours'] == pytest.approx(8.8863654887)
    assert half['within_7_5h'] and not full['within_7_5h']
    assert half['incremental_hours']*2 == full['incremental_hours']
    assert all(b.runtime_projection(v)['go_uses_c'] == 1 for v in f.VARIANTS)
    assert 'every target >=-.005' in b.registration()['rule']


@pytest.mark.parametrize('variant', f.VARIANTS)
def test_sealed_notebook_extraction_executes(tmp_path, variant, monkeypatch):
    artifacts, files = b.payloads(variant)
    import sys
    monkeypatch.setattr(sys, 'path', list(sys.path))
    for kind in ('kernel', 'stress-kernel-clean', 'stress-kernel'):
        code = ''.join(json.loads(artifacts[kind+'/release.ipynb'])['cells'][-1]['source'])
        preamble = code.split(f'import {b.MODULE} as release')[0]
        target = tmp_path/kind
        preamble = preamble.replace("Path('/kaggle/working/c85r7ins-runtime')", repr(target))
        exec(compile(preamble, '<extraction>', 'exec'), {'WindowsPath': type(target)})
        assert {name:(target/name).read_bytes() for name in files} == files


@pytest.mark.parametrize('variant', f.VARIANTS)
def test_runtime_flip_arguments_and_provenance(tmp_path, variant):
    manifest = dict(backbone='convnext_small', size=384, crop=352, flip_tta=False)
    identity = dict(manifest_sha256='test', manifest_pinned=True, file_sha256={},
                    declared={}, features={})
    with base.metadata_io.writers(base):
        with f.installed(base.r7, variant, base.metadata_io):
            args = base.r7.worker_args(manifest, tmp_path, tmp_path/'anatomy.pt')
            assert args.count('--flip-tta') == 1
            record = base.r7.write_provenance(tmp_path/'provenance.json',
                identity=identity, code=dict(tree_sha256='test', files={}, closure_sha256={},
                                            r7_closure_sha256={}), args=args)
            assert record['flip_policy'] == f.policy(variant)
            assert b.read(tmp_path/'provenance.json') == record
    assert manifest['flip_tta'] is False
