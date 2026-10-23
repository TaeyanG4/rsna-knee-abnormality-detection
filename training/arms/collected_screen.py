"""Collected MIL paired readout v1.0.0 (2026-10-11).

Changelog: verify collected artifacts against archived specs and receipts, pin
screen inputs, and publish all seven lenses beside the registered gate. The
mechanical verdict is evidence for B's holistic decision, never an admission.
No training-source requalification or automatic model selection is implied.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from training.arms import lifecycle as life
from training.arms.mil_screen_replay import path, portable_path
from tools import multi_lens_readout as multi

VERSION = '1.0.0'
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'collected-mil-screen/v1'
REQUEST = 'collected-mil-screen-request/v1'
SOURCES = ('training/arms/collected_screen.py', 'tools/mil_screen.py',
           'training/arms/mil_screen_replay.py', 'training/arms/mil_screen.py',
           'training/arms/lifecycle.py', 'tools/multi_lens_readout.py',
           'tools/exp156_k_sweep.py', 'tools/exp125_radr50_k16_local.py',
           'tools/exp115_vlm_eval.py')
EVALUATION = (multi.COMPONENTS, multi.JEV, multi.AQUILA, multi.DREAD, *multi.CONSENSUS)
need = life._need


def read(file):
    return life.parse_json(Path(file).read_bytes())


def pin(root, value, bindings, *, suffix=None):
    row = {'path': str(value)} if isinstance(value, (str, Path)) else value
    need(type(row) is dict and set(row) <= {'path', 'sha256'} and 'path' in row,
         'file binding must contain path and optional sha256')
    name = portable_path(row['path'], root)
    need(not any(p.lower() in {'api-keys', '.kaggle', '.codex', '.ssh', 'relabel'}
                 for p in name.split('/')), 'restricted input')
    need(suffix is None or name.endswith(suffix), 'input type differs')
    digest = life.file_sha256(path(root, name))
    need(row.get('sha256', digest) == digest, 'input hash differs')
    need(bindings.get(name, digest) == digest, 'input changed while binding')
    bindings[name] = digest
    return dict(path=name, sha256=digest)


def predictions(root, expr, bindings):
    if isinstance(expr, list):
        need(bool(expr), 'empty prediction list')
        return [predictions(root, row, bindings) for row in expr]
    if type(expr) is dict and 'blend' in expr:
        need(set(expr) <= {'blend', 'rank'}, 'unknown blend field')
        parts = []
        for part in expr['blend']:
            need(set(part) == {'files', 'weight'}, 'unknown blend part')
            parts.append(dict(files=predictions(root, part['files'], bindings), weight=part['weight']))
        return dict(expr, blend=parts)
    if type(expr) is dict and 'rank_folds_like' in expr:
        need(set(expr) == {'files', 'rank_folds_like'}, 'unknown fold-rank field')
        return {k: predictions(root, v, bindings) for k, v in expr.items()}
    return pin(root, expr, bindings, suffix='.csv')


def bind_readout(spec, root):
    need(type(spec) is dict and set(spec) <= {'name', 'family', 'label_change', 'seeds',
         'fold_rank', 'partners', 'anchor', 'image_reads', 'old_verdict'}, 'unknown readout field')
    need(type(spec.get('seeds')) is dict and bool(spec['seeds']), 'paired seeds required')
    need(type(spec.get('label_change', False)) is bool and
         type(spec.get('fold_rank', False)) is bool, 'readout flags must be boolean')
    out, bindings = copy.deepcopy(spec), {}
    # Artifact family identifiers and the fixed L6 bundle use different names.
    # Normalize before replacement so a MIL candidate never keeps its own family.
    if out.get('family') == 'convnext-mil':
        out['family'] = 'convnext'
    for seed, pair in out['seeds'].items():
        need(type(seed) is str and seed.isdecimal() and str(int(seed)) == seed,
             'canonical integer seed required')
        need(type(pair) is dict and set(pair) == {'arm', 'control'}, 'paired arm/control required')
        out['seeds'][seed] = {side: predictions(root, expr, bindings) for side, expr in pair.items()}
    if 'partners' in out:
        out['partners'] = {k: predictions(root, v, bindings) for k, v in out['partners'].items()}
    if 'anchor' in out:
        out['anchor'] = predictions(root, out['anchor'], bindings)
    for name, item in out.get('image_reads', {}).items():
        need(set(item) <= {'arm', 'control', 'labels', 'components', 'eligible'} and
             {'arm', 'control', 'labels', 'components'} <= set(item), 'image-read fields differ')
        need(type(item.get('eligible', False)) is bool, 'image eligibility must be boolean')
        for side in ('arm', 'control'):
            item[side] = predictions(root, item[side], bindings)
        item['labels'] = pin(root, item['labels'], bindings, suffix='.csv')['path']
        if item['components'] == 'gold58_singletons':
            need(name == 'gold58', 'singleton policy is Gold58 diagnostic only')
            need(not item.get('eligible', False), 'Gold58 is diagnostic only')
        else:
            item['components'] = pin(root, item['components'], bindings, suffix='.json')['path']
    for name in (*SOURCES, *EVALUATION):
        pin(root, name, bindings)
    recheck(root, bindings)
    return out, bindings


def recheck(root, bindings):
    for name, digest in bindings.items():
        need(life.file_sha256(path(root, name)) == digest, 'screen input changed: ' + name)


def readout(spec, *, root=ROOT):
    """Hash-bind a legacy/new pair, execute multi_lens_readout, retain every lens."""
    bound, bindings = bind_readout(spec, root)
    result = multi.clean(multi.run_spec(bound, root=root))
    recheck(root, bindings)
    return dict(schema='holistic-paired-readout/v1', version=VERSION,
                holistic_decision='B_DECISION_PENDING', launch_authorized=False,
                mechanical_verdict=result.get('verdict'), multi_lens=result,
                input_sha256=bindings, readout_spec=bound)


def collection(root, entry, seed, family, bindings):
    need(type(entry) is dict and set(entry) == {'spec', 'collection', 'prediction'},
         'collection entry fields differ')
    # Caller must supply hashes: registration never silently trusts a receipt label.
    for key in ('spec', 'collection'):
        need(type(entry[key]) is dict and set(entry[key]) == {'path', 'sha256'},
             'explicit collected spec/manifest hashes required')
    spec_file = pin(root, entry['spec'], bindings, suffix='.json')
    spec = life.ArmSpec(path(root, spec_file['path']).read_bytes())
    doc = spec.document
    need(doc['family'] == family and doc['seed'] == int(seed), 'collection family/seed differs')
    need(doc['config']['targets'] == list(multi.TARGETS), 'collection target identity differs')
    need(spec.run_kind == 'experiment', 'member has no paired validation population')
    manifest_file = pin(root, entry['collection'], bindings, suffix='/collection.json')
    manifest = read(path(root, manifest_file['path']))
    need(set(manifest) == {'schema', 'status', 'scientific_status', 'spec_sha256',
         'receipt_sha256', 'archive_sha256'} and manifest['schema'] == 'arm-collection/v1' and
         manifest['status'] == 'COLLECTED_ARTIFACTS_VERIFIED' and
         manifest['scientific_status'] == 'UNASSESSED' and manifest['spec_sha256'] == spec.sha256,
         'collection manifest differs')
    life._hash(manifest['archive_sha256'], 'collection archive')
    base = manifest_file['path'].removesuffix('/collection.json')
    receipt_file = pin(root, base+'/receipt.json', bindings)
    receipt = read(path(root, receipt_file['path']))
    life.verify_artifact_receipt(spec, path(root, base+'/outputs'), receipt,
                                expected_receipt_sha256=manifest['receipt_sha256'])
    for name, digest in receipt['outputs'].items():
        pin(root, dict(path=base+'/outputs/'+name, sha256=digest), bindings)
    name = entry['prediction']
    need(name in doc['outputs'] and name.endswith('.csv'), 'prediction absent from receipt')
    prediction = pin(root, dict(path=base+'/outputs/'+name, sha256=receipt['outputs'][name]), bindings)
    # Validate canonical row order, not only a matching number of CSV rows.
    import pandas as pd
    ids = pd.read_csv(path(root, prediction['path']), usecols=[multi.UID],
                      dtype={multi.UID: str})[multi.UID].tolist()
    roots = doc['config']['roots']
    roster_name = roots['ordered']
    need(roster_name in doc['inputs']['rosters'], 'validation roster must be registered')
    pin(root, dict(path=roster_name, sha256=doc['inputs']['rosters'][roster_name]), bindings)
    ordered = read(path(root, roster_name))
    need(len(ids) == roots['n_val'] and len(ids) == len(set(ids)) and
         ids == ordered['val'] and not set(ids) & set(ordered['train']) and
         hashlib.sha256('\n'.join(ids).encode()).hexdigest() == roots['val_sha256'],
         'collection validation roster differs')
    controls = doc['inputs']['controls']
    summary_dirs = [p.removesuffix('/summary.json') for p in controls if p.endswith('/summary.json')]
    need(len(summary_dirs) == 1 and summary_dirs[0]+'/provenance.json' in controls,
         'registered corresponding control evidence required')
    for control_file in (summary_dirs[0]+'/summary.json', summary_dirs[0]+'/provenance.json'):
        pin(root, dict(path=control_file, sha256=controls[control_file]), bindings)
    return prediction, dict(spec_sha256=spec.sha256, seed=doc['seed'], fold=doc['fold'],
                            receipt_sha256=manifest['receipt_sha256'],
                            collection=manifest_file, prediction=prediction,
                            registered_control=summary_dirs[0])


def draft(root, request):
    need(type(request) is dict and request.get('schema') == REQUEST and
         set(request) <= {'schema', 'readout', 'collections', 'registered_gate'}, 'request fields differ')
    readout_spec = copy.deepcopy(request['readout'])
    need('seeds' not in readout_spec, 'seeds are resolved from collections')
    readout_spec['seeds'], records, bindings = {}, {}, {}
    need(type(request['collections']) is dict and bool(request['collections']), 'collections required')
    for seed, pair in request['collections'].items():
        need(type(seed) is str and seed.isdecimal() and str(int(seed)) == seed, 'invalid seed')
        need(type(pair) is dict and set(pair) == {'arm', 'control'}, 'collection pair required')
        need(type(pair['arm']) is list and bool(pair['arm']), 'arm collections required')
        arms, records[seed] = [], []
        for entry in pair['arm']:
            prediction, record = collection(root, entry, seed, readout_spec['family'], bindings)
            arms.append(prediction); records[seed].append(record)
        folds = [r['fold'] for r in records[seed]]
        need(len(folds) == len(set(folds)), 'duplicate collected fold')
        controls = pair['control'] if type(pair['control']) is list else [pair['control']]
        need(len(controls) == len(arms), 'one registered control prediction per fold required')
        for control, record in zip(controls, records[seed]):
            need(type(control) is dict and set(control) == {'path', 'sha256'},
                 'control prediction must be hash-pinned')
            control = pin(root, control, bindings, suffix='.csv')
            need(control['path'].rsplit('/', 1)[0] == record['registered_control'],
                 'prediction is not the registered corresponding control')
        readout_spec['seeds'][seed] = dict(arm=arms, control=pair['control'])
    bound, screen_bindings = bind_readout(readout_spec, root)
    bindings.update(screen_bindings)
    gate = None
    if request.get('registered_gate') is not None:
        row = request['registered_gate']
        need(type(row) is dict and set(row) == {'path', 'sha256'}, 'registered gate must be pinned')
        row = pin(root, row, bindings, suffix='.json')
        gate = dict(binding=row, result=read(path(root, row['path'])))
    recheck(root, bindings)
    return dict(schema=SCHEMA, version=VERSION, request=request, readout=bound,
                collections=records, input_sha256=bindings, registered_gate=gate,
                holistic_decision='B_DECISION_PENDING', launch_authorized=False)


def register(root, request, out):
    spec = draft(root, request)
    life.create_new(out, life.canonical_bytes(spec))
    return spec


def validate(root, spec):
    need(type(spec) is dict and spec.get('schema') == SCHEMA, 'invalid collected screen spec')
    expected = draft(root, spec['request'])
    need(life.canonical_bytes(spec) == life.canonical_bytes(expected), 'collected registration differs')
    return expected


def screen(root, spec, out):
    need(not Path(out).exists(), 'fresh versioned screen required')
    validate(root, spec)
    result = readout(spec['readout'], root=root)
    validate(root, spec)
    result.update(schema='collected-mil-screen-result/v1', spec_sha256=life.content_sha256(spec),
                  collections=spec['collections'], registered_gate=spec['registered_gate'],
                  scientific_status='SCREENED_B_DECISION_PENDING',
                  collection_input_sha256=spec['input_sha256'])
    life.create_new(out, life.canonical_bytes(result))
    return result
