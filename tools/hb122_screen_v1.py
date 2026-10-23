"""HB122 seven-lens screen v1.0.0; strict collection/panel authentication.

Changelog 2026-10-13: shared fresh S controls, corrected L4, fixed pooled Q7S
fold0 replacement, descriptive mapped-nine OAI macro. No training/selection.
2026-10-13 claude-hb122: fail closed unless every fit records the registered
workers=2/device/batch identity (PROTOCOL-v1.1), and the four fits share one
runtime observation.
The historical OOF reader installs its CPU tripwires only in readout processes;
never import it from a runner or panel-inference process.
"""
import argparse
import csv
import os
from pathlib import Path
import time

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
from tools import hb122_build_v1 as build
from tools import multi_lens_readout as ml
from tools import multi_lens_readout_l4_v2 as corrected

mil, life = build.mil, build.life
ROOT, WORK = build.ROOT, build.WORK
VERSION = '1.0.0'
PRIMARY = 'oof_k36_flip_tta.csv'
PANEL_TARGETS = (0, 2, 3, 4, 5, 6, 7, 8, 9)
ANCESTRY = ('DESCRIPTIVE_ONLY: full trainable/teacher ancestor exclusion of OAI participants '
            'is not proven. No independent image-truth or RSNA macro12 claim.')
FINAL_PROXY = ('Unavailable: no authenticated common-UID heldout OWN anchors for final-P956; '
               'Q7S is an aligned OOF surrogate only.')


def source_pins():
    names = (*mil.SOURCE_FILES, 'tools/hb122_screen_v1.py', 'tools/hb122_build_v1.py',
             'tools/multi_lens_readout.py', 'tools/multi_lens_readout_l4_v2.py',
             'tools/hb105_aligned_oof_v1.py', 'tools/oai_moaks_panel.py')
    return {n: mil.sha(ROOT / n) for n in names}


def snapshot(kind, seed, smoke=False):
    """Scientific snapshots require complete fits, PASS evidence and collection."""
    spec = build.spec_for(kind, seed)
    mil.validate_recipe(spec)
    life.validate_inputs(spec, ROOT)
    run = WORK / build.key(kind, seed)
    base = run / ('cpu-collected' if smoke else 'collected')
    out = base / 'outputs'
    receipt, collected = mil.read(base / 'receipt.json'), mil.read(base / 'collection.json')
    export = mil.read(run / ('cpu-export' if smoke else 'export') / 'export.json')
    life.verify_artifact_receipt(spec, out, receipt, expected_receipt_sha256=export['receipt_sha256'])
    mil.need(collected['status'] == 'COLLECTED_ARTIFACTS_VERIFIED'
             and collected['spec_sha256'] == spec.sha256
             and collected['receipt_sha256'] == export['receipt_sha256'], 'collection binding differs')
    summary = build.old.check_outputs(spec, out, smoke=smoke)
    prov = mil.read(out / 'provenance.json')
    mil.need(receipt['metrics']['engineering_smoke'] == int(smoke), 'fit mode differs')
    identity = prov['identity']
    mil.need(identity['workers'] == build.WORKERS and identity['mode'] == ('smoke' if smoke else 'full')
             and identity['device'] == ('cpu' if smoke else 'cuda') and identity['batch'] == (2 if smoke else 8),
             'registered loader/device identity differs (PROTOCOL-v1.1)')
    if not smoke:
        mil.need(receipt['verification_sha256'] == prov['verification_sha256'], 'fit evidence differs')
    note_root = run / ('cpu' if smoke else 'pass')
    note = mil.read(note_root / ('verification.json' if smoke else 'verification-pass.json'))
    life.verify_evidence(spec, note, note_root)
    mil.need(life.content_sha256(note) == receipt['verification_sha256'], 'receipt evidence differs')
    if not smoke:
        life.require_verification_pass(note, **spec.verification_args)
        life.check_target_observation(spec, summary['observation'])
        r = spec.document['config']['roots']
        mil.need(prov['train_ids_ordered_sha256'] == r['train_sha256']
                 and prov['eval_ids_ordered_sha256'] == r['val_sha256']
                 and prov['checkpoint_selection'] == 'fixed-final', 'heldout ancestry differs')
    files = (base/'receipt.json', base/'collection.json', out/'model_best.pt', out/PRIMARY,
             out/'provenance.json', out/'summary.json', out/'output-inventory.json', build.spec_path(kind, seed))
    return dict(key=build.key(kind, seed), seed=seed, spec_sha256=spec.sha256,
                engineering_smoke=smoke,
                runtime_sha256=life.content_sha256(identity['observation']['runtime']),
                checkpoint=(out/'model_best.pt').relative_to(ROOT).as_posix(),
                prediction=(out/PRIMARY).relative_to(ROOT).as_posix(),
                bindings={p.relative_to(ROOT).as_posix(): mil.sha(p) for p in files})


def all_snapshots(smoke=False):
    snaps = {build.key(k, s): snapshot(k, s, smoke) for k, s in build.PAIRS}
    mil.need(len({v['runtime_sha256'] for v in snaps.values()}) == 1, 'fits ran on different runtimes')
    return snaps


def load_predictions(path, ids, column=mil.UID):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        mil.need(reader.fieldnames == [column, *mil.TARGETS], 'prediction columns differ')
        rows = list(reader)
    mil.need([r[column] for r in rows] == ids and len(ids) == len(set(ids)), 'prediction roster differs')
    values = np.asarray([[float(r[t]) for t in mil.TARGETS] for r in rows], dtype=np.float32)
    mil.need(values.shape == (len(ids), 12) and np.isfinite(values).all()
             and ((values >= 0) & (values <= 1)).all(), 'invalid predictions')
    return values


def panel_context():
    from tools import oai_moaks_panel as p
    ids = p.validate()
    route = p.routing()
    pins = {str(q.relative_to(ROOT)).replace('\\', '/'): mil.sha(q) for q in
            (p.OUT/'registration.json', p.OUT/'PROTOCOL.md', p.OUT/'runtime-v1.0.3.json',
             p.OUT/'vendor/manifest.json')}
    return p, ids, route, pins


def recipe_path(smoke=False):
    return WORK / ('panel-smoke-recipes-v1.json' if smoke else 'panel-recipes-v1.json')


def member_recipe(item):
    return dict(key=item['key'], checkpoint=item['checkpoint'],
        sha256=item['bindings'][item['checkpoint']], family='mil', seed=item['seed'],
        targets=mil.TARGETS, cache='64', k=36, tta='mean_probability',
        precision_cpu='fp32', precision_gpu='fp16', snapshot=item, eligible=False, ancestry=ANCESTRY)


def panel_prepare(smoke=False):
    p, ids, route, pins = panel_context()
    members = all_snapshots(smoke)
    recipes = {}
    for name, item in members.items():
        recipes[name] = member_recipe(item)
    doc = dict(version=VERSION, sources=source_pins(), panel_pins=pins, recipes=recipes,
               rows=len(ids), ordered_ids_sha256=mil.uid_hash(ids), routing=route, engineering_smoke=smoke)
    build.same_or_new(recipe_path(smoke), doc)
    return dict(status='PANEL_RECIPES_READY', members=len(recipes), rows=len(ids), engineering_smoke=smoke)


def admit_panel(kind, seed, smoke=False):
    p, ids, route, pins = panel_context()
    doc = mil.read(recipe_path(smoke))
    mil.need(doc['version'] == VERSION and doc['sources'] == source_pins() and doc['panel_pins'] == pins
             and doc['engineering_smoke'] is smoke and doc['ordered_ids_sha256'] == mil.uid_hash(ids)
             and doc['rows'] == len(ids) and doc['routing'] == route
             and set(doc['recipes']) == {build.key(k, s) for k, s in build.PAIRS}, 'panel recipe environment drift')
    rec = doc['recipes'][build.key(kind, seed)]
    mil.need(rec == member_recipe(snapshot(kind, seed, smoke)), 'panel member drift')
    return p, ids, doc, rec


def panel_file(kind, seed, smoke=False):
    return WORK / ('panel-smoke' if smoke else 'panel-predictions-v1') / (build.key(kind, seed) + '.csv')


def authenticate_panel_receipt(receipt, recipe, doc, ids, digest, smoke=False):
    """Pure binding gate, also exercised with synthetic prediction files."""
    mil.need(receipt['version'] == VERSION and receipt['recipe'] == recipe
             and receipt['sources'] == doc['sources'] and receipt['panel_pins'] == doc['panel_pins']
             and receipt['recipe_file_sha256'] == life.content_sha256(doc)
             and receipt['predictions_sha256'] == digest and receipt['rows'] == len(ids)
             and receipt['ids_sha256'] == mil.uid_hash(ids)
             and receipt['engineering_smoke'] is smoke
             and receipt['full_panel'] is (not smoke)
             and receipt['device'] == ('cpu' if smoke else 'cuda')
             and receipt['volume_files_verified'] > 0, 'panel prediction binding differs')


def panel_prediction(kind, seed, smoke=False, admitted=None):
    p, all_ids, doc, recipe = admitted or admit_panel(kind, seed, smoke)
    ids = all_ids[:2] if smoke else all_ids
    path = panel_file(kind, seed, smoke)
    receipt = mil.read(str(path) + '.json')
    authenticate_panel_receipt(receipt, recipe, doc, ids, mil.sha(path), smoke)
    if not smoke:
        life.check_target_observation(build.spec_for(kind, seed), receipt['observation'])
    return ids, load_predictions(path, ids, 'study_key'), receipt


def panel_infer(kind, seed, smoke=False):
    import torch
    import cv2
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    admitted = admit_panel(kind, seed, smoke)
    p, ids, doc, recipe = admitted
    destination = panel_file(kind, seed, smoke)
    if destination.exists():
        _, _, receipt = panel_prediction(kind, seed, smoke, admitted)
        return dict(status='AUTHENTICATED_EXISTING_PANEL', rows=receipt['rows'], key=recipe['key'])
    mil.need(not Path(str(destination)+'.json').exists(), 'orphan panel receipt')
    device = 'cpu' if smoke else 'cuda'
    mil.configure_runtime('none' if smoke else 'fp16', device, policy='strict')
    observed = mil.observation(device)
    if not smoke:
        life.check_target_observation(build.spec_for(kind, seed), observed)
    ck = p.load_checkpoint(ROOT / recipe['checkpoint'])
    spec = build.spec_for(kind, seed)
    mil.need(ck['lab'] == mil.TARGETS and ck['arch'] == mil.ARCH and ck['fold'] == 0
             and ck['model_kind'] == 'finding_spatial_tokens_v1'
             and ck['provenance']['spec_sha256'] == spec.sha256
             and ck['init_weights_sha256'] == spec.document['config']['model']['weights_sha256']
             and ck['epoch'] == (0 if smoke else 19), 'panel checkpoint metadata differs')
    del ck
    model, cache = p.model_for(recipe).to(device), p.Cache('64')
    use_ids = ids[:2] if smoke else ids
    start = time.monotonic()
    values = np.asarray([p.predict_one(model, recipe, cache, uid, device) for uid in use_ids])
    mil.need(values.shape == (len(use_ids), 12) and np.isfinite(values).all()
             and ((values >= 0) & (values <= 1)).all(), 'invalid panel forward')
    mil.need(smoke or len(ids) == 557, 'full panel size differs')
    mil.need(snapshot(kind, seed, smoke) == recipe['snapshot'], 'model changed during inference')
    life.create_new(destination, life.prediction_csv_bytes(use_ids, values, mil.TARGETS, id_column='study_key'))
    receipt = dict(version=VERSION, recipe=recipe, sources=source_pins(), panel_pins=doc['panel_pins'],
        recipe_file_sha256=mil.sha(recipe_path(smoke)), predictions_sha256=mil.sha(destination),
        rows=len(use_ids), ids_sha256=mil.uid_hash(use_ids), engineering_smoke=smoke,
        full_panel=not smoke, device=device, observation=observed,
        seconds=time.monotonic()-start, volume_files_verified=len(cache.verified))
    build.same_or_new(str(destination)+'.json', receipt)
    panel_prediction(kind, seed, smoke, admitted)
    mil.need(not smoke or not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    return dict(status='PANEL_INFERRED', key=recipe['key'], rows=len(use_ids), seconds=receipt['seconds'])


def fixed_bundle(raw, arms, controls, variants):
    """Only replace fold0 S42/S43; ranks fixed on all five folds before slicing."""
    output = []
    n = len(raw[0]['S42'])
    for pair in (arms, controls):
        value = {f: dict(v) for f, v in raw.items()}
        for seed in ('42', '43'):
            a = pair[seed]
            mil.need(a.shape == raw[0]['S'+seed].shape and a.dtype == np.float32, 'replacement shape/dtype differs')
            value[0]['S'+seed] = a
        output.append(variants(value)['add'][:n])
    return output


def image_item(ids, arms, controls, y):
    mil.need(set(arms) == set(controls) == {'42', '43'} and len(ids) == len(set(ids)), 'panel seeds/IDs differ')
    y = np.asarray(y)
    mil.need(y.shape == (len(ids), 12) and not np.isinf(y).any()
             and np.isin(y[np.isfinite(y)], (0, 1)).all(), 'invalid binary panel outcomes')
    for array in (*arms.values(), *controls.values()):
        mil.need(array.shape == y.shape and np.isfinite(array).all()
                 and ((array >= 0) & (array <= 1)).all(), 'invalid panel probabilities')
    return dict(ids=ids, arm=np.stack(list(arms.values())).mean(0, dtype=np.float32),
                control=np.stack(list(controls.values())).mean(0, dtype=np.float32),
                labels=np.nan_to_num(y), mask=np.isfinite(y), eligible=False,
                components=[dict(component_id=u, members=[u]) for u in ids],
                component_policy='one knee per participant; participant singleton bootstrap')


def evaluate(ctx, arms, controls, bundle, item):
    result = corrected.evaluate(ctx, arms, controls, bundle=bundle, image_reads={'oai_moaks': item})
    w = ml.weights(item['ids'], item['components'])
    a = ml.auc_draws(item['arm'], item['labels'], item['mask'], w)
    c = ml.auc_draws(item['control'], item['labels'], item['mask'], w)
    # Fixed nine, never nanmean or changing target support per bootstrap draw.
    ix = list(PANEL_TARGETS)
    overall = ml.stat((a[:, ix]-c[:, ix]).mean(1), eligible=False, reason=ANCESTRY)
    overall.update(targets=[mil.TARGETS[t] for t in ix], arm_auc=ml.clean(a[0, ix].mean()),
                   control_auc=ml.clean(c[0, ix].mean()), metric='mapped-nine macro AUC')
    entry = result['lenses']['L7']['entries']['oai_moaks']
    for statistic in entry['targets'].values():
        statistic.update(eligible=False, reason=ANCESTRY)
    entry.update(overall_macro=overall, ancestry_reason=ANCESTRY,
                 unavailable_targets={mil.TARGETS[t]: 'No registered MOAKS mapping' for t in (1, 10, 11)})
    result['lenses']['L6']['final_proxy'] = ml.unavailable(FINAL_PROXY)
    result.update(scope='FOLD0_SCREEN_ONLY', release_scope='none', not_release_authorization=True,
        bootstrap=dict(draws=ml.DRAWS, seed=ml.SEED, interval='paired percentile 95%',
                       limitations='Conditional on fitted models; excludes fit/selection/private-shift uncertainty'),
        warnings=[ANCESTRY, FINAL_PROXY, 'Holistic decision belongs to B; no automatic release.',
                  'Report-ruler supervision is exposed; L7 cannot be treated as independent.',
                  'Negative/flat two-seed evidence is retained; coefficient is never tuned.'])
    return ml.clean(result)


def readout(smoke=False):
    # Authenticate checkpoints/receipts first. The historical reader below bans
    # torch.load/CUDA globally for this CPU readout process only.
    snaps = all_snapshots(smoke)
    panels, panel_receipts, ids_panel = {}, {}, None
    for kind, seed in build.PAIRS:
        ids, values, receipt = panel_prediction(kind, seed, smoke)
        mil.need(ids_panel is None or ids == ids_panel, 'panel paired roster differs')
        ids_panel = ids
        panels.setdefault(kind, {})[str(seed)] = values
        panel_receipts[build.key(kind, seed)] = mil.sha(str(panel_file(kind, seed, smoke))+'.json')
    from tools import oai_moaks_panel as p
    y = p.outcomes(ids_panel)
    from tools import hb105_aligned_oof_v1 as aligned
    raw, rosters, auth = aligned.authenticate()
    ids = rosters[0]['val']
    arrays = {kind: {str(seed): (raw[0]['S'+str(seed)].copy() if smoke else
        load_predictions(ROOT / snaps[build.key(kind, seed)]['prediction'], ids)) for seed in (42, 43)}
              for kind in ('control', 'rank')}
    arm, control = fixed_bundle(raw, arrays['rank'], arrays['control'], aligned.variants)
    ctx = ml.make_context(ids, ROOT)
    mil.need(ctx.bindings == auth['ruler_bindings'], 'ruler binding changed')
    bundle = dict(context=ctx, arm=arm, control=control, recipe=dict(scope='ALIGNED_OOF_SURROGATE_ONLY',
        change='only fold0 S42/S43; folds1-4 fixed', arithmetic=aligned.POLICY['arithmetic'],
        scoring='fold0 slice of fixed pooled predictions; no fold0 reranking', final_proxy=FINAL_PROXY))
    result = evaluate(ctx, arrays['rank'], arrays['control'], bundle,
                      image_item(ids_panel, panels['rank'], panels['control'], y))
    result.update(version=VERSION, status='ENGINEERING_CONTROL_REPLAY' if smoke else 'COMPLETE_FOLD0_SCREEN',
        sources=source_pins(), fitted_members=snaps, panel_receipts=panel_receipts,
        aligned_authentication=auth, ruler_bindings=ctx.bindings, rows=len(ids), panel_rows=len(ids_panel),
        ordered_ids_sha256=mil.uid_hash(ids), engineering_smoke=smoke,
        replay_reason='CPU fits cannot supply scientific heldout OOF; authentic deployed S used on both sides' if smoke else None)
    # Detect mutation of all small source/receipt/OOF inputs during scoring.
    for item in snaps.values():
        for rel, digest in item['bindings'].items():
            mil.need(mil.sha(ROOT/rel) == digest, 'member changed during scoring')
    output = WORK / ('screen-cpu-smoke-v1.json' if smoke else 'screen-seven-lens-v1.json')
    build.same_or_new(output, result)
    return dict(status=result['status'], output=str(output.relative_to(ROOT)),
                rows=len(ids), panel_rows=len(ids_panel), lenses=list(result['lenses']))


def waiting():
    missing = []
    for kind, seed in build.PAIRS:
        run = WORK / build.key(kind, seed)
        for p in (run/'collected/receipt.json', panel_file(kind, seed), Path(str(panel_file(kind, seed))+'.json')):
            if not p.is_file():
                missing.append(p.relative_to(ROOT).as_posix())
    status = dict(status='WAITING_FOR_COMPLETE_FITS_AND_PANEL' if missing else 'READY_FOR_AUTHENTICATION',
                  missing=missing, scientific_readout_written=False,
                  lens_reasons={f'L{i}': ('Missing complete seed-paired fits' if i < 6 else
                      'L6 requires completed replacement fits; final-P956 remains unavailable' if i == 6 else
                      'L7 requires all four 557-knee panel predictions; ancestry remains descriptive') for i in range(1, 8)})
    ml.write(WORK/'screen-waiting-v1.json', status)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('panel-prepare', 'panel-infer', 'readout', 'waiting'))
    ap.add_argument('--kind', choices=('control', 'rank'))
    ap.add_argument('--seed', type=int, choices=(42, 43))
    ap.add_argument('--engineering-smoke', action='store_true')
    args = ap.parse_args()
    if args.command == 'waiting': result = waiting()
    elif args.command == 'panel-prepare': result = panel_prepare(args.engineering_smoke)
    elif args.command == 'panel-infer': result = panel_infer(args.kind, args.seed, args.engineering_smoke)
    else: result = readout(args.engineering_smoke)
    print(life.canonical_bytes(result).decode())


if __name__ == '__main__':
    main()
