"""CPU-only real DICOM/window smoke for both sealed native flip policies."""
import ast
import csv
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp169_release_c85r7flip_v130 as build
from tools import exp169_r7_selective_flip_v130 as flip


def function(source, name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)


def extract(source, names, namespace):
    for name in names:
        exec(compile(ast.Module(body=[function(source, name)], type_ignores=[]), '<native-r7>', 'exec'), namespace)
    return namespace


def compare(source, reference, models, image, meta, spec, one_pass):
    import torch
    from tools.hb149_r7_flip_v120 import LABELS
    ns = extract(reference, ['mirror_perm', 'mirror_windows', 'state_layout', 'state_readout', 'score_v4c'],
        dict(torch=torch, MIRROR_PAIRS=(("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))))
    perm, bp = ns['mirror_perm'](LABELS, spec, list(models[0].bin_names))
    columns = ns['state_layout'](spec, LABELS)['columns']
    mclcols = []
    for c in columns:
        if c['class'] == 'MCL':
            mclcols.extend(range(c['start'], c['stop']) if 'start' in c else [c['column']])
    ns.update(FLIP=None, MCL_STATE_COLS=mclcols, EXTRA={}, one_pass=one_pass)
    mask = torch.ones(image.shape[:2], dtype=torch.bool)
    plain = torch.stack(ns['score_v4c'](models, image, meta, mask))
    plain_state = torch.stack(ns['EXTRA']['group1']['state'])
    ns['FLIP'] = dict(perm=perm, bin_perm=bp, mcl=LABELS.index('MCL'))
    native = torch.stack(ns['score_v4c'](models, image, meta, mask))
    native_state = torch.stack(ns['EXTRA']['group1']['state'])
    extract(source, ['score_v4c'], ns)
    selective = torch.stack(ns['score_v4c'](models, image, meta, mask))
    selective_state = torch.stack(ns['EXTRA']['group1']['state'])
    for i, model in enumerate(models):
        assert torch.equal(selective[i], (native if model._c85r7_flip else plain)[i])
        assert torch.equal(selective_state[i], (native_state if model._c85r7_flip else plain_state)[i])
    assert torch.equal(selective[:, LABELS.index('MCL')], plain[:, LABELS.index('MCL')])
    assert torch.equal(selective_state[:, mclcols], plain_state[:, mclcols])
    assert torch.isfinite(selective).all() and torch.isfinite(selective_state).all()
    return dict(logits_exact=True, state_heads_exact=True, mcl_plain_exact=True,
                logits_sha256=build.digest(selective.numpy().tobytes()))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from tools import exp169_release_c85r7codec_deadline_v122 as base
    from tools import hb149_r7_flip_v120 as hb
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    out = build.SCRATCH / 'real-smoke'
    out.mkdir(parents=True, exist_ok=True)
    invocation = build.read(ROOT / '.local/scratch/astra-b5/H-B-149/cpu-smoke-v121/invocation_dual.json')
    def arg(flag):
        return invocation[invocation.index(flag)+1]
    deps = ROOT / '.local/scratch/claude-hb128/runtime/deps'
    sys.path[:0] = [str(deps), arg('--code')]
    runtimes = {}
    for variant in flip.VARIANTS:
        stage = out / variant
        stage.mkdir(exist_ok=True)
        runtime = stage / 'r7_runtime'
        if not runtime.exists():
            with base.metadata_io.writers(base):
                with flip.installed(base.r7, variant, base.metadata_io):
                    runtime = base.r7.diagnostic_source(base.r7.SOURCE, stage)
        worker = runtime/'compact_infer.py'
        if '_c85r7_flip' not in worker.read_text(encoding='utf-8'):
            worker.write_text(flip.patch_worker(worker.read_text(encoding='utf-8'), variant), encoding='utf-8')
            (runtime/'runtime_sha256.json').write_bytes(build.encoded(
                {p.name: build.digest(p.read_bytes()) for p in sorted(runtime.glob('*.py'))}))
        pins = build.read(runtime / 'runtime_sha256.json')
        assert all(build.digest((runtime/k).read_bytes()) == v for k, v in pins.items())
        runtimes[variant] = runtime
    runtime = runtimes['FLIP_F0F2']
    sys.path.insert(0, str(runtime))
    specmod = importlib.util.spec_from_file_location('compact_infer', runtime / 'compact_infer.py')
    mod = importlib.util.module_from_spec(specmod)
    sys.modules['compact_infer'] = mod
    specmod.loader.exec_module(mod)
    uids = build.read(Path(arg('--assignment')))
    data = out/'data'
    data.mkdir(exist_ok=True)
    shutil.copyfile(Path(arg('--data'))/'test_series.csv', data/'test_series.csv')
    with (data/'test.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID'])
        writer.writerows([uid] for uid in uids)
    train = hb.TrainSource(ROOT/'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip')
    try:
        for member in train.members[uids[0]]:
            rel = Path(member)
            target = data/'test_series'/uids[0]/rel.parent.name/rel.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with train.zip.open(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
    finally:
        train.close()
    audit = hb.decode_audit(data/'test_series'/uids[0])
    assert audit['decode_failures'] == 0
    dataset = mod.CStudies(str(data), 'test', uids, str(out/'preprocess'), arg('--code'),
        384, 1, 'v4', 3, arg('--anchor-weights'), 3.5, arg('--side-model'), 2., 0., 20.,
        extent_meta=True, fluid=True, device='cpu')
    item = dataset[0]
    if item['error']:
        raise RuntimeError(item['error'])
    assert len(item['images']) > 0
    # Actual rendered input: one window per available acquisition plane. Full
    # study preprocessing, deliberately bounded model forwards (not qualification).
    planes = item['meta'][:, 7]
    indices = [int((planes == p).nonzero()[0]) for p in planes.unique()]
    image = item['images'][indices][None, :, :, 16:368, 16:368]
    meta = item['meta'][indices][None]
    import compact_train as CT
    state_spec = build.read(Path(arg('--state-spec')))
    source = (runtime/'compact_infer.py').read_text(encoding='utf-8')
    hb_stage = out/'hb149-reference'
    if not (hb_stage/'r7_runtime').exists():
        hb.runtime_source(hb_stage, dual=True)
    reference = (hb_stage/'r7_runtime/compact_infer.py').read_text(encoding='utf-8')
    ns = extract(source, ['load_group'], dict(torch=torch, Path=Path, CT=CT,
        a=SimpleNamespace(init=None, freeze_blocks=0, slices=3), state_spec=state_spec,
        N_EXTENT=[0], N_SYN=[0], net_has_syn=True, dev=torch.device('cpu')))
    _, models, _ = ns['load_group'](arg('--models'), 'E_lrprior', 'convnext_small', 352)
    assert [m._c85r7_flip for m in models] == [True, False, True, False]
    # Cache only actual real-checkpoint forwards; scoring/remapping runs
    # independently for native and sealed variants on exactly the same tensors.
    cache = {}
    with torch.inference_mode():
        mirrored, mirrored_meta = mod.mirror_windows(image, meta)
        mask = torch.ones(image.shape[:2], dtype=torch.bool)
        for i, model in enumerate(models):
            for mirrored_flag, x, mt in [(False, image, meta), (True, mirrored, mirrored_meta)]:
                cache[(id(model), mirrored_flag)] = tuple(t.float() for t in model(x, mt, mask, chunk=3, return_aux=True))
            print(json.dumps(dict(event='real_fold_forward_pass', fold=i, windows=len(indices))), flush=True)
        def one_pass(model, x, mt, mask):
            return cache[id(model), x.data_ptr() != image.data_ptr()]
        results = {}
        for variant, rt in runtimes.items():
            for i, model in enumerate(models):
                model._c85r7_flip = i in flip.VARIANTS[variant]
            results[variant] = compare((rt/'compact_infer.py').read_text(encoding='utf-8'),
                reference, models, image, meta, state_spec, one_pass)
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS_REAL_CPU_SMOKE', studies=1, rendered_windows=len(item['images']),
        scored_windows=len(indices), checkpoint_models=4, real_forwards=8, cuda_initialized=False, decode_audit=audit,
        variants=results, scope='full real DICOM preprocessing; truncated-window fp32 inference; not AUC/T4 qualification')
    (out/'result.json').write_bytes(build.encoded(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
