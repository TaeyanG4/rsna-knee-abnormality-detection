"""H105 aligned heldout OOF surrogate, v1.0.0. Additive CPU reader only.

Changelog 2026-10-12: authenticated saved substitutions, probability-first
FP32 seed means, fixed pooled ranks, paired component readouts. No forwards.
"""
from __future__ import annotations

import os
import sys
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
sys.dont_write_bytecode = True

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = 'artifacts/experiments/HB105_ALIGNED_OOF_V1'
SCRATCH = '.local/scratch/sol-2/hb105-aligned-readout'
AUTH = 'artifacts/experiments/HB105_SERVING_V1/input-manifest.json'
AUTH_SHA = '47598e384892f11809a25ca05d5e119fd080ca0041ccf8e39f606a3e9561c84b'
LEGACY = '.local/hb105m-collected-v1/fivefold-q7-with-s-v1.json'
TINY_SHA = 'cf8419f6538a6bc531838f402bb1168645d2534da0e59934686a73fc4de357d9'
VERSION = '1.0.0'
FAMILIES = {'R50': ('R42', 'R43'), 'C4': ('C4',), 'T': ('M42', 'M43'), 'S': ('S42', 'S43')}
BUNDLES = {'Q7': ('R50', 'C4', 'T'), 'add': ('R50', 'C4', 'T', 'S'),
           'replace': ('R50', 'C4', 'S')}
CONTRASTS = (('add', 'Q7'), ('replace', 'Q7'), ('add', 'replace'))
POLICY = dict(scope='OOF_SURROGATE_ONLY', release='FULL_RELEASE_HOLD',
    arithmetic='CSV to FP32; heldout seed probability mean; concatenate fold0..4; '
               'pooled average-tie family rank; equal-family FP64 mean; pooled bundle rank',
    subsets='Fivefold readouts slice the fixed pooled predictions; no subset family reranking',
    draws=300, seed=20261003, unit='heldout connected component',
    contrasts=[a+'-minus-'+b for a, b in CONTRASTS],
    own_blends='MISSING_DEPENDENCY: immutable H105 manifest binds deployment anchor assets, '
               'not common-UID heldout own S42/S43 predictions; .30/.50 comparisons omitted',
    limitations=['Each row uses only its heldout fold; no all-fold model inference on training UIDs.',
                 'Conditional bootstrap excludes fit/selection uncertainty.',
                 'Report-label rulers only; no image truth, private gain, independent ensemble or GPU admission.'])


def need(ok, message):
    if not ok:
        raise ValueError(message)


def cpu_guard():
    import torch
    need(not torch.cuda._initialized, 'CUDA was already initialized')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    def tripwire(api):
        # Distinct identities preserve torch's function-to-rule registry.
        def forbidden(*args, **kwargs):
            raise RuntimeError('FORBIDDEN_CUDA_OR_CHECKPOINT_LOAD: '+api)
        return forbidden
    for name in ('_lazy_init', 'init', 'is_available', 'device_count', 'synchronize',
                 'get_device_properties', 'get_device_name', 'set_device', 'current_device'):
        setattr(torch.cuda, name, tripwire('cuda.'+name))
    torch.Tensor.cuda = tripwire('Tensor.cuda')
    torch.nn.Module.cuda = tripwire('Module.cuda')
    torch.load = tripwire('torch.load')


cpu_guard()
import numpy as np
from tools import hb105_members_readout as legacy
from tools import ensemble_ablation_v1 as ea
from tools import multi_lens_readout as ml
from tools import hb105_serving_v101 as serving
from tools import exp169_v2_l2_flip_oof as flip
from tools import exp115_vlm_eval  # references()'s lazy numeric label loader; source-bound
from tools import exp152_own0949_release  # serving.rank()'s lazy average-tie helper; source-bound


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(path):
    return serving.pin(path)


def checked(root, rel, expected, bindings):
    path = root / rel
    need(path.resolve().is_relative_to(root.resolve()), 'input outside project')
    actual = pin(path)
    need(actual == expected if isinstance(expected, dict) else actual['sha256'] == expected,
         'binding changed: '+rel)
    bindings[rel] = actual
    return path


def load_csv(path, ids):
    # The legacy API aligns by UID; additionally reject reorder/duplicates here.
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        need(rows.fieldnames == [flip.UID, *ml.TARGETS], 'prediction columns')
        actual = [r[flip.UID] for r in rows]
    need(actual == ids and len(set(actual)) == len(ids), 'prediction UID/order differs')
    return ea.hb.load_oof(path, ids).astype(np.float32)


def key_for(e):
    return ('C4' if e['family'] == 'C4' else
            {'R50': 'R', 'T': 'M', 'S': 'S'}[e['family']] + str(e['seed']))


def check_member_ancestry(e, prov, roster):
    need(e['train_uid_sha256'] == prov['train_ids_ordered_sha256']
         and e['heldout_uid_sha256'] == prov['eval_ids_ordered_sha256'], 'member provenance')
    need(e['heldout_uid_sha256'] == serving.uid_hash(roster['val'])
         and e['train_uid_sha256'] in (serving.uid_hash(roster['train']),
                                      serving.uid_hash(sorted(roster['train'])))
         and not set(roster['train']) & set(roster['val'])
         and (e['train_count'], e['heldout_count']) == (len(roster['train']), len(roster['val'])),
         'member fold exclusion differs')


def authenticate(root=ROOT):
    """Only audited paths; existing H105 inventory pins anchor checkpoint ancestry.

    No checkpoint content reads: previously authenticated digest + inventory,
    provenance, scientific receipts and current CSV bytes form the binding.
    """
    root = Path(root).resolve(); bindings = {}
    auth = read(checked(root, AUTH, AUTH_SHA, bindings))
    saved = read(checked(root, LEGACY, auth['bindings'][LEGACY], bindings))
    need(saved['status'] == 'COMPLETE_OOF_READOUT', 'legacy context status')
    for path, digest in saved['source_sha256'].items():
        checked(root, path, digest, bindings)
    def bound(rel):
        need(rel in auth['bindings'], 'not in immutable H105: '+rel)
        return read(checked(root, rel, auth['bindings'][rel], bindings))
    rosters = [bound(f'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{f}/ordered-ids.json')
               for f in range(5)]
    serving.validate_rosters(rosters)
    members = auth['members']
    expected = {(family, seed, f) for family in FAMILIES for seed in
                ((42,) if family == 'C4' else (42, 43)) for f in range(5)}
    need(len(members) == 35 and {(e['family'], e['seed'], e['fold']) for e in members} == expected,
         'member coverage')
    raw = {f: {} for f in range(5)}; records = []
    for e in members:
        directory = Path(e['path']).parent.as_posix()
        inv = bound(directory+'/output-inventory.json')
        need(inv['model_best.pt'] == {k: e[k] for k in ('bytes', 'sha256')}, 'checkpoint inventory')
        need((root/e['path']).stat().st_size == e['bytes'], 'checkpoint size changed')
        prov_path = directory+'/provenance.json'
        prov = bound(prov_path)
        checked(root, prov_path, inv['provenance.json'], bindings)
        f = e['fold']; ids = rosters[f]['val']
        check_member_ancestry(e, prov, rosters[f])
        # Reuse the existing summary/roster gate, including sorted-same-roster allowance.
        summary_path = directory+'/summary.json'
        summary = read(checked(root, summary_path, inv['summary.json'], bindings))
        roster_info = dict(train_sha256=serving.uid_hash(rosters[f]['train']),
                           sorted_train_sha256=serving.uid_hash(sorted(rosters[f]['train'])),
                           val_sha256=serving.uid_hash(ids))
        ea.validate_summary({**summary, 'provenance': prov}, roster_info, len(ids), f)
        if e.get('spec_sha256'):
            spec_paths = [p for p, v in auth['bindings'].items()
                          if p.endswith('.json') and v['sha256'] == e['spec_sha256']]
            need(len(spec_paths) == 1, 'spec identity ambiguous')
            spec = bound(spec_paths[0]); base = Path(directory).parent.as_posix()
            receipt = bound(base+'/receipt.json'); collection = bound(base+'/collection.json')
            need(receipt['status'] == 'ARTIFACTS_VERIFIED' and receipt['exit_code'] == 0
                 and receipt['metrics'].get('engineering_smoke', 0) == 0
                 and collection['status'] == 'COLLECTED_ARTIFACTS_VERIFIED'
                 and collection['receipt_sha256'] == bindings[base+'/receipt.json']['sha256']
                 and receipt['spec_sha256'] == collection['spec_sha256'] == prov['spec_sha256'] == e['spec_sha256']
                 and receipt['verification_sha256'] == prov['verification_sha256'], 'scientific receipt')
            for name in ('model_best.pt', 'provenance.json', 'summary.json', legacy.PRIMARY):
                need(receipt['outputs'][name] == inv[name]['sha256'], 'receipt output ancestry')
            need(receipt['outputs']['output-inventory.json'] == bindings[directory+'/output-inventory.json']['sha256'],
                 'receipt inventory ancestry')
            need(summary['status'] == 'FIXED_FINAL_COMPLETE' and summary['epochs'] == 20
                 and summary['attempted_updates'] == spec['config']['updates']['steps']
                 and summary['skipped_updates'] <= spec['config']['updates']['max_skips']
                 and prov['checkpoint_selection'] == 'fixed-final', 'incomplete fit')
        if e['family'] == 'R50':
            src = next(s for s in flip.SOURCES if (s.seed, s.fold) == (e['seed'], f))
            folder = flip.OUT_REL+'/'+src.tag
            rc = read(root/folder/flip.RECEIPT)
            # Existing CPU verifier; deliberately not verify_receipt(), which rehashes checkpoints.
            flip.verify_folder(root/folder, src, ids, e['sha256'])
            need(rc['source']['output_inventory_sha256'] == bindings[directory+'/output-inventory.json']['sha256']
                 and rc['source']['files']['provenance.json'] == inv['provenance.json']
                 and rc['source']['roster']['held_out_ids_sha256'] == e['heldout_uid_sha256'],
                 'R50 source ancestry')
            for node in (rc['source']['protocol'], rc['source']['roster']['exp099_ordered_ids']):
                checked(root, node['path'], node['sha256'], bindings)
            for name in (flip.RECEIPT, flip.INVENTORY):
                bindings[folder+'/'+name] = pin(root/folder/name)
            for name, expected_pin in rc['outputs']['files'].items():
                checked(root, folder+'/'+name, expected_pin, bindings)
            path = folder+'/'+flip.TTA
            substitution = 'audited R50 single-view -> saved plane-flip mean'
        else:
            path = directory+'/'+legacy.PRIMARY
            checked(root, path, inv[legacy.PRIMARY], bindings)
            if e['id'] == 'T-s43-f0':
                need(inv[legacy.PRIMARY]['sha256'] == TINY_SHA, 'Tiny v3 substitution drift')
                substitution = 'audited old Tiny F0 -> deployed v3 saved OOF'
            else:
                need(path in saved['bindings'] and bindings[path]['sha256'] == saved['bindings'][path],
                     'unchanged prediction drift')
                substitution = 'unchanged legacy prediction'
        raw[f][key_for(e)] = load_csv(root/path, ids)
        records.append(dict(member=e, prediction=path, prediction_pin=bindings[path],
                            substitution=substitution, checkpoint_binding='receipt/inventory; size checked, not rehashed'))
    ruler_bindings = saved['results']['pooled_5fold']['context_bindings']
    need(set(ruler_bindings) == {ml.COMPONENTS, ml.JEV, ml.AQUILA, ml.DREAD, *ml.CONSENSUS}, 'ruler set changed')
    for result in saved['results'].values():
        need(result['context_bindings'] == ruler_bindings, 'legacy ruler scopes differ')
    for path, digest in ruler_bindings.items():
        checked(root, path, digest, bindings)
    return raw, rosters, dict(bindings=bindings, members=records, ruler_bindings=ruler_bindings)


def variants(raw):
    need(set(raw) == set(range(5)), 'five heldout folds required')
    for fold in raw.values():
        need(len({p.shape for p in fold.values()}) == 1, 'member row shapes differ')
    families = {}
    for family, keys in FAMILIES.items():
        means = []
        for f in range(5):
            need(set(raw[f]) == set(sum((list(k) for k in FAMILIES.values()), [])), 'raw members differ')
            a = np.stack([raw[f][key] for key in keys])
            need(a.dtype == np.float32 and a.ndim == 3 and a.shape[2] == 12
                 and a.shape[1] > 0 and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), 'invalid probabilities')
            means.append(a.mean(axis=0, dtype=np.float32))
        families[family] = serving.rank(np.concatenate(means))
    return {name: serving.rank(sum(families[k] for k in keys)/len(keys))
            for name, keys in BUNDLES.items()}


def source_pins(root=ROOT):
    # Imported project dependency closure only; no workspace crawl.
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and str(filename).endswith('.py'):
            path = Path(filename).resolve()
            if path.is_relative_to(root) and path.relative_to(root).parts[0] in ('tools', 'training'):
                paths.add(path)
    return {p.relative_to(root).as_posix(): pin(p) for p in sorted(paths)}


def freeze():
    _, _, inputs = authenticate()
    manifest = dict(version=VERSION, policy=POLICY, **inputs, sources=source_pins(),
                    authority=dict(path=AUTH, sha256=AUTH_SHA),
                    direct_hashing='All listed bindings/sources including CSVs; no checkpoints.',
                    receipt_binding='35 checkpoint hashes from immutable H105; pinned inventories/provenance/receipts; current sizes checked.')
    serving.write(ROOT/EXP/'input-source-manifest.json', manifest)
    return dict(status='FROZEN_INPUTS_FULL_RELEASE_HOLD', members=len(inputs['members']))


def load_frozen():
    path = ROOT/EXP/'input-source-manifest.json'
    frozen = read(path)
    need(frozen['version'] == VERSION and frozen['policy'] == POLICY
         and frozen['authority'] == dict(path=AUTH, sha256=AUTH_SHA), 'frozen policy drift')
    for rel, value in {**frozen['bindings'], **frozen['sources']}.items():
        checked(ROOT, rel, value, {})
    raw, rosters, current = authenticate()
    need(all(current[k] == frozen[k] for k in current), 'frozen input substitution drift')
    return raw, rosters, frozen, pin(path)


def comparisons(ctx, values):
    result = {}
    for a, b in CONTRASTS:
        item = ea.comparison(ctx, values[a], values[b])
        # Original threshold diagnostics are retained but cannot imply release admission.
        item['diagnostic_lens_verdict'] = item.pop('verdict')
        item.pop('decision_label'); item.pop('release_scope')
        item.update(scope='OOF_SURROGATE_ONLY', release='FULL_RELEASE_HOLD')
        item['warnings'] = POLICY['limitations']
        delta = ctx.delta(values[a], values[b])
        for i, ruler in enumerate(ml.RULERS):
            item['rulers'][ruler]['weak_four'] = ml.stat(delta[:, i, ml.WEAK].mean(1))
        result[a+'-minus-'+b] = item
    return result


def run(smoke=False):
    raw, rosters, frozen, manifest_pin = load_frozen()
    values = variants(raw)
    ids = sum((r['val'] for r in rosters), [])
    by_fold = {}; offset = 0
    for f, r in enumerate(rosters):
        by_fold[f] = list(range(offset, offset+len(r['val'])))
        offset += len(r['val'])
    selected = {f: ix[:16] if smoke else ix for f, ix in by_fold.items()}
    scopes = [('pooled_5fold', sum(selected.values(), [])),
              *[(f'fold{f}', selected[f]) for f in range(5)]]
    results = {}
    for name, ix in scopes:
        ctx = ml.make_context([ids[i] for i in ix], ROOT)
        need(ctx.bindings == frozen['ruler_bindings'], 'ruler changed during context build')
        results[name] = dict(rows=len(ix), ordered_uid_sha256=serving.uid_hash(ctx.ids),
                             comparisons=comparisons(ctx, {k: p[ix] for k, p in values.items()}))
    # Detect mutation during evaluation, without re-reading model bytes.
    for rel, expected in {**frozen['bindings'], **frozen['sources']}.items():
        checked(ROOT, rel, expected, {})
    import torch
    need(not torch.cuda._initialized, 'CUDA initialized')
    result = ml.clean(dict(version=VERSION, policy=POLICY, input_manifest=manifest_pin,
        status='ENGINEERING_SMOKE_ONLY' if smoke else 'COMPLETE_OOF_SURROGATE_ONLY',
        results=results, legacy_context=dict(path=LEGACY, pin=frozen['bindings'][LEGACY],
                                          interpretation='old Tiny/plain R50/rank-first; not recomputed'),
        cuda_initialized=False, member_rows=sum(len(p) for fold in raw.values() for p in fold.values())))
    out = ROOT/SCRATCH/('smoke.json' if smoke else 'readout.json')
    if smoke:
        ml.write(out, result)  # fixed engineering scratch may be overwritten
    else:
        serving.write(out, result)  # scientific result create-only
    return dict(status=result['status'], scopes=len(results), comparisons=18,
                evaluated_rows=len(scopes[0][1]), member_rows=result['member_rows'], output=str(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('freeze', 'smoke', 'readout'))
    a = ap.parse_args(argv)
    print(json.dumps(freeze() if a.command == 'freeze' else run(a.command == 'smoke')))


if __name__ == '__main__':
    main()
