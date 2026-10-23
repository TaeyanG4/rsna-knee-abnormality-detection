"""Offline EXP157 single-head L2 release. Never uploads, runs CUDA or submits."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)) if str(HERE) not in sys.path else None
import exp140_aug1_release as base

OUT = ROOT / '.local/exp157-l2-release'
DATASET = 'taeyangg4/knee-exp157-l2-aug1-assets'
KERNEL = 'taeyangg4/knee-exp157-l2-aug1'

def checkpoint_entry(root, seed, fold):
    import exp157_l2_advance as advance
    name = advance.ensemble_tag(seed, fold)
    run = Path(root) / 'artifacts/experiments' / name
    inventory = json.loads((run / 'output-inventory.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    prov = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
    for file in ('summary.json', 'provenance.json', 'model_best.pt'):
        info = inventory[file]
        base.need((run / file).stat().st_size == info['bytes'] and base.digest(run / file) == info['sha256'], 'Inventory mismatch: ' + name + '/' + file)
    base.need(summary['selected_checkpoint'] == 'fixed-final' and summary['selected_epoch_1based'] == 20 and summary['worker_cleanup']['clean'] is True and summary['total_seconds'] > 0, 'Incomplete L2 run: ' + name)
    base.need(prov['seed'] == seed and prov['fold'] == fold and prov['label_set'] == 'l2' and prov['training_mode'] == 'full' and prov['initial_state_matches_registered'] is True, 'L2 provenance mismatch: ' + name)
    base.need(prov['arch'] == 'resnet50' and summary['args']['norm'] == 'radimagenet' and summary['args']['input_policy'] == 'center3', 'Serving contract mismatch: ' + name)
    base.need(prov['label_file_sha256'] == base.digest(Path(root) / 'artifacts/experiments/EXP149_LABEL_SOURCES/derived/l2.csv'), 'L2 labels mismatch: ' + name)
    base.need(summary['provenance']['protocol_sha256'] == prov['protocol_sha256'], 'Protocol receipt mismatch: ' + name)
    info = inventory['model_best.pt']
    return dict(seed=seed, fold=fold, tag=name, path=f'checkpoints/rad{seed}_f{fold}.pt', sha256=info['sha256'], bytes=info['bytes'], protocol_sha256=prov['protocol_sha256'])

def manifest(root=ROOT):
    entries = [checkpoint_entry(root, seed, fold) for seed in (42, 43) for fold in range(5)]
    base.need(len({x['sha256'] for x in entries}) == 10, 'Ten distinct checkpoint hashes required')
    return dict(schema='exp133-rad-assets-v1', dataset=DATASET, training='EXP157 L2: 0.75 Dread + 0.25 aquila_v1, AUG1, D-B-083', checkpoints=entries)

def notebook(data):
    # A separate module namespace changes only the dataset binding in the reused builder.
    spec = importlib.util.spec_from_file_location('_exp157_exp140', HERE / 'exp140_aug1_release.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.DATASET = DATASET
    nb = adapter.a_notebook(data)
    source = nb['cells'][-1]['source']
    if isinstance(source, list):
        source = ''.join(source)
    header = "NOTEBOOK_STARTED = __import__('time').monotonic()\nimport os\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'\nos.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\nimport pathlib, json\n"
    header += "_comp_paths = (pathlib.Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), pathlib.Path('/kaggle/input/rsna-knee-abnormality-detection'))\n_comp = [p for p in _comp_paths if all((p / n).exists() for n in ('test.csv', 'test_series.csv', 'test_series', 'sample_submission.csv'))]\nassert len(_comp) == 1, 'Expected one competition mount'\n"
    slug = DATASET.split('/')[-1]
    header += f"_asset_paths = (pathlib.Path('/kaggle/input/{slug}'), pathlib.Path('/kaggle/input/datasets/taeyangg4/{slug}'))\n_assets = [p for p in _asset_paths if (p / 'exp133-rad-assets.json').is_file()]\nassert len(_assets) == 1, 'Expected one asset mount'\n"
    header += "_mounted = json.loads((_assets[0] / 'exp133-rad-assets.json').read_text())\nassert _mounted == " + repr(data) + "\nfor _c in _mounted['checkpoints']:\n    _p = _assets[0] / _c['path']\n    assert _p.is_file() and _p.stat().st_size == _c['bytes'], _c['path']\n"
    shim = "\ndef _exp157_input_dir(dataset, marker, input_root=None):\n    p = _assets[0] if dataset == e.RAD_DATASET else _comp[0] if dataset == e.e128.COMPETITION else None\n    assert p is not None and (p / marker).exists(), (dataset, marker)\n    return p\ne.input_dir = _exp157_input_dir\n"
    base.need(source.count('import exp133_oai_b3 as e\n') == 1 and
              source.count('start = time.monotonic()') == 1,
              'Frozen EXP140 notebook insertion points differ')
    source = source.replace('import exp133_oai_b3 as e\n', 'import exp133_oai_b3 as e\n' + shim, 1)
    source = source.replace('start = time.monotonic()', 'start = NOTEBOOK_STARTED', 1).replace('EXP140', 'EXP157')
    source = header + source
    compile(source, '<EXP157 release>', 'exec')
    nb['cells'][-1]['source'] = source
    nb['cells'][0]['source'] = '# EXP157 L2 AUG1 single-head RadR50\nFive folds × two seeds; frozen EXP140 flip TTA and seed rank reduction.\n'
    return nb

def build(out=OUT, root=ROOT):
    import exp157_l2_advance as advance
    out, root = Path(out), Path(root)
    base.need(not out.exists(), 'Choose a fresh output directory')
    advance.verify(root)
    data = manifest(root)
    kernel, assets = out / 'kernel', out / 'upload-stage'
    kernel.mkdir(parents=True)
    assets.mkdir()
    base.save(assets / base.ASSET_NAME, data)
    base.save(assets / 'dataset-metadata.json', dict(title=DATASET.split('/')[-1], id=DATASET, isPrivate=True, licenses=[{'name': 'other'}]))
    files = [dict(source=str(root / 'artifacts/experiments' / c['tag'] / 'model_best.pt'), destination=c['path'], sha256=c['sha256'], bytes=c['bytes']) for c in data['checkpoints']]
    for name in (base.ASSET_NAME, 'dataset-metadata.json'):
        path = assets / name
        files.append(dict(source=str(path), destination=name, sha256=base.digest(path), bytes=path.stat().st_size))
    base.save(out / 'upload-manifest.json', dict(schema='exp157-stage-v1', dataset=DATASET, status='NOT_UPLOADED', files=files))
    base.save(kernel / 'release.ipynb', notebook(data))
    meta = base.e133.e128.metadata('fidelity')
    meta.update(id=KERNEL, title=KERNEL.split('/')[-1], dataset_sources=[DATASET])
    base.save(kernel / 'kernel-metadata.json', meta)
    base.save(kernel / 'runtime-projection-1300.json', base.runtime_projection())
    base.save(out / 'build-receipt.json', dict(schema='exp157-build-v1', status='UNPUSHED', notebook_sha256=base.digest(kernel / 'release.ipynb'), checkpoints=data['checkpoints'], network_calls=0, gpu_runs=0, submissions=0))
    return dict(status='BUILT', out=str(out))

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('build', 'stage', 'validate-public3'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--output', type=Path)
    p.add_argument('--sample', type=Path, default=ROOT / 'storage/competition/sample_submission.csv')
    a = p.parse_args(argv)
    if a.command == 'build':
        result = build(a.out)
    elif a.command == 'stage':
        result = base.stage(a.out)
    else:
        base.need(a.output is not None, '--output required')
        # The reused standalone notebook retains EXP140's receipt schema/name.
        result = base.validate_public3('A', a.output, a.sample, a.out / 'kernel')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
