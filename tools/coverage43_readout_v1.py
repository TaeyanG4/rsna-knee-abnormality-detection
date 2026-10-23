"""COVERAGE43 paired CPU adapter v1.0.0.

Changelog: reuse EXP190F's byte-preserving conversion and fixed multi-lens API
for the collected lifecycle coverage variant. No training or release admission.
Launch: python -B -m tools.coverage43_readout_v1
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import csv
import json
from dataclasses import asdict
from pathlib import Path

from training.arms import coatnet_lifecycle as contract, lifecycle as life
from tools.exp190f_readout import convert
from tools import multi_lens_readout as multi

ROOT = Path(__file__).resolve().parents[1]
BASE = '.local/refactor/qual-4070-coverage-fold0-s43-2bba6848551347eaa2d6e7809e1269cc'
CONTROL = '.local/refactor/qual-4070-control-fold0-s43-8594f460d5d24d09a9f0f3f6796ab082/train/outputs'
PAIR_MAP = '.local/refactor/coverage43-controls.json'
SCRATCH = '.local/scratch/sol-2/coverage43-screen'
VIEWS = ('mean', 'plain', 'mirror')
SPEC_SHA = '39f9382124d52a57bee6666b8d4158befd9ff0c4d872c987d6d4e0c6b3157ce5'
RECEIPT_SHA = 'e339131bc36099892eeb9fb3e8c16edc5f877d68642e5014a411ed0403b206d3'
SOURCES = ('tools/coverage43_readout_v1.py', 'tools/exp190f_readout.py',
           'training/arms/exp190f.py', 'training/arms/coatnet_lifecycle.py',
           'training/arms/coatnet.py', 'training/arms/lifecycle.py',
           'tools/multi_lens_readout.py', 'tools/exp156_k_sweep.py',
           'tools/exp125_radr50_k16_local.py', 'tools/exp115_vlm_eval.py')
need = life._need


def read(path):
    return life.parse_json(Path(path).read_bytes())


def cpu_only():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    def forbidden(*args, **kwargs):
        raise RuntimeError('CUDA forbidden in COVERAGE43 CPU readout')
    need(not torch.cuda.is_initialized(), 'CUDA already initialized')
    torch.cuda._lazy_init = forbidden
    torch.cuda.init = forbidden


def pin(root, name, bindings, expected=None):
    digest = life.file_sha256(root/name)
    need(expected is None or digest == expected, 'input hash differs: '+name)
    need(bindings.get(name, digest) == digest, 'input changed: '+name)
    bindings[name] = digest
    return digest


def validate_fit(summary, provenance, config, population_sha):
    need(summary['status'] == 'COMPLETE_TRAINING_AND_READOUT' and summary['mode'] == 'full'
         and summary['updates'] == 3456 and summary['skips'] == 0, 'full completed fit required')
    need(provenance['scientific'] == config and provenance['population_sha256'] == population_sha,
         'paired scientific configuration differs')
    identity = provenance['identity']
    need(identity['mode'] == 'full' and identity['spec_sha256'] == provenance['spec_sha256']
         and identity['train_sha256'] == config['train_sha256']
         and identity['readout_sha256'] == config['val_sha256']
         and provenance['readout'] == 'held-out-fold0', 'fit roster identity differs')
    life._hash(provenance['verification_sha256'], 'fit verification')


def validate_csv(path, ids):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = csv.DictReader(stream)
        need([row['study_id'] for row in rows] == ids, 'prediction UID order differs')
    values = life.load_prediction_csv(path, ids, multi.TARGETS)
    # The reusable loader already rejects duplicates, shape drift and nonfinite values.
    import numpy as np
    need(np.all((values >= 0) & (values <= 1)), 'probabilities outside [0,1]')
    return values


def view_spec(view, pair):
    need(view in VIEWS, 'unknown view')
    return dict(name='COVERAGE43-vs-CONTROL43-'+view, family='coatnet', label_change=False,
                seeds={'43': pair})


def run(root=ROOT):
    root = Path(root)
    out, scratch = root/BASE/'screen-full', root/SCRATCH
    need(not out.exists(), 'fresh screen-full required; existing results are immutable')
    need((multi.DRAWS, multi.SEED) == (300, 20261003), 'bootstrap contract differs')
    bindings = {}
    for name in SOURCES:
        pin(root, name, bindings)
    spec_path = BASE+'/spec.json'
    pin(root, spec_path, bindings)
    spec = contract.Spec((root/spec_path).read_bytes())
    need(spec.sha256 == SPEC_SHA, 'registered coverage spec differs')
    doc = spec.document
    need((doc['seed'], doc['fold'], doc['config']['variant'], doc['run_kind']) ==
         (43, 0, 'coverage', 'experiment'), 'coverage identity differs')
    pin(root, PAIR_MAP, bindings)
    need(read(root/PAIR_MAP) == doc['inputs']['controls'], 'control pairing differs')
    for name, digest in doc['inputs']['controls'].items():
        pin(root, name, bindings, digest)
    pop_name = doc['population_source']
    pop_sha = doc['inputs']['rosters'][pop_name]
    pin(root, pop_name, bindings, pop_sha)
    population = contract.population(spec, root)
    ids = population['val_ids']
    need(len(ids) == 888 and len(set(ids)) == 888, 'heldout roster differs')
    collection_name = BASE+'/collected-full/collection.json'
    receipt_name = BASE+'/collected-full/receipt.json'
    pin(root, collection_name, bindings)
    pin(root, receipt_name, bindings, RECEIPT_SHA)
    collection, receipt = read(root/collection_name), read(root/receipt_name)
    need(collection['schema'] == 'arm-collection/v1' and
         collection['status'] == 'COLLECTED_ARTIFACTS_VERIFIED' and
         collection['spec_sha256'] == spec.sha256 and collection['receipt_sha256'] == RECEIPT_SHA,
         'collection identity differs')
    need(receipt['schema'] == 'arm-artifact-receipt/v1' and receipt['status'] == 'ARTIFACTS_VERIFIED'
         and receipt['spec_sha256'] == spec.sha256 and receipt['scope'] == spec.scope
         and receipt['exit_code'] == 0 and receipt['metrics'] == dict(updates=3456, skips=0)
         and set(receipt['outputs']) == set(doc['outputs']), 'collected receipt differs')
    arm = BASE+'/collected-full/outputs'
    for name in ('summary.json', 'provenance.json', *('predictions-'+v+'.csv' for v in VIEWS)):
        pin(root, arm+'/'+name, bindings, receipt['outputs'][name])
    reference = dict(doc['config'], variant='control', factory=asdict(contract.c.PRESETS['control']))
    for directory, config in ((arm, doc['config']), (CONTROL, reference)):
        summary, provenance = [read(root/directory/name) for name in ('summary.json', 'provenance.json')]
        validate_fit(summary, provenance, config, pop_sha)
        life.check_target_observation(spec, summary['observation'])
        if directory == arm:
            need(provenance['spec_sha256'] == spec.sha256 and
                 provenance['verification_sha256'] == receipt['verification_sha256'], 'arm provenance differs')
    for name in (multi.COMPONENTS, multi.JEV, multi.AQUILA, multi.DREAD, *multi.CONSENSUS):
        pin(root, name, bindings)
    scratch.mkdir(parents=True, exist_ok=True)
    specs = {}
    for view in VIEWS:
        pair = {}
        for side, directory in (('arm', arm), ('control', CONTROL)):
            name = directory+'/predictions-'+view+'.csv'
            pin(root, name, bindings)
            validate_csv(root/name, ids)
            destination = scratch/(side+'-'+view+'.csv')
            # Fixed scratch may be reused only for exactly the same converted bytes.
            if destination.exists():
                raw = (root/name).read_bytes()
                need(destination.read_bytes() == b'StudyInstanceUID,'+raw[len(b'study_id,'):],
                     'existing scratch conversion differs')
                item = dict(path=destination.resolve().as_posix(), sha256=life.file_sha256(destination))
            else:
                item = convert(root/name, destination, ids)
            pair[side] = [item]
        specs[view] = view_spec(view, pair)
        multi.write(scratch/f'spec-{view}.json', specs[view])
    manifest = dict(schema='coverage43-screen-inputs/v1', input_sha256=bindings, specs=specs,
        spec_sha256=spec.sha256, receipt_sha256=RECEIPT_SHA, rows=len(ids),
        checkpoint_binding='Receipt-bound only; no checkpoint load or rehash; control CSVs directly hashed.',
        checkpoint_sha256=receipt['outputs']['model_best.pt'], draws=300, seed=20261003,
        scope='PAIRED_FOLD0_S43_SURROGATE_ONLY', launch_authorized=False)
    # Freeze bindings before results, then check them again before publication.
    multi.write(scratch/'input-manifest.json', manifest)
    cache, results = {}, {}
    for view in VIEWS:
        results[view] = multi.clean(multi.run_spec(specs[view], root=root, context_cache=cache))
        results[view].update(adapter_version='1.0.0', launch_authorized=False,
                             scope=manifest['scope'], input_manifest_sha256=life.content_sha256(manifest))
    life._verify_files(root, bindings)
    life.create_new(out/'input-manifest.json', life.canonical_bytes(manifest))
    for view, result in results.items():
        life.create_new(out/f'result-{view}.json', life.canonical_bytes(result))
    return {view: dict(verdict=r['verdict'], L1=r['lenses']['L1'], L2=r['lenses']['L2'],
                       harm_guard=r['harm_guard']) for view, r in results.items()}


if __name__ == '__main__':
    cpu_only()
    print(json.dumps(run(), allow_nan=False))
