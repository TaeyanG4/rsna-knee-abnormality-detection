"""HB123 seven-lens screen v1.0.0; frozen v2 fits versus shared HB122 controls.

Changelog 2026-10-13: imported corrected L4/two-seed L5/fixed Q7S replacement,
header-bound OAI557 canonical serving for the arm only. Full controls mandatory.
No changes to training, deployed readers, checkpoints or existing receipts.
"""
import argparse
import os
from pathlib import Path
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
from tools import hb122_screen_v1 as shared
from tools import hb123_build_v2 as build
from tools import hb123_oai_side_v1 as side

mil, life, lat = build.mil, build.life, build.mil.lat
ROOT, WORK = build.ROOT, side.WORK
VERSION = '1.0.0'
OAI_MAP_SHA256 = 'dc70a83d2d1f6393b6380ed06c555c0fba023ee9c5d2cdfa60e0735b43dbe3a3'
PAIRS = tuple((kind, seed) for kind in ('control', 'right') for seed in (42, 43))
# Import the corrected readout implementation; never patch shared globals.
evaluate, fixed_bundle, image_item = shared.evaluate, shared.fixed_bundle, shared.image_item


def key(kind, seed):
    mil.need((kind, seed) in PAIRS, 'unregistered member')
    return f'{kind}-s{seed}'


def spec_for(kind, seed):
    key(kind, seed)
    return shared.build.spec_for('control', seed) if kind == 'control' else build.spec_for(seed)


def run_for(kind, seed):
    key(kind, seed)
    return shared.WORK/key(kind, seed) if kind == 'control' else build.WORK/f's{seed}'


def source_pins():
    names = (*mil.EXTRA_SOURCES, 'tools/hb123_oai_side_v1.py', 'tools/hb123_screen_v1.py',
             '.local/run-scripts/hb123-screen-v1.ps1', '.local/scratch/sol-5/hb123/SCREEN-PROTOCOL-v1.md')
    return {**shared.source_pins(), **{p: mil.sha(ROOT/p) for p in names}}


def snapshot(kind, seed, smoke=False):
    if kind == 'control':
        return shared.snapshot('control', seed, smoke)
    spec, run = spec_for(kind, seed), run_for(kind, seed)
    mil.validate_recipe(spec)
    life.validate_inputs(spec, ROOT)
    base = run/('cpu-collected' if smoke else 'collected')
    out = base/'outputs'
    receipt, collected = mil.read(base/'receipt.json'), mil.read(base/'collection.json')
    export = mil.read(run/('cpu-export' if smoke else 'export')/'export.json')
    life.verify_artifact_receipt(spec, out, receipt, expected_receipt_sha256=export['receipt_sha256'])
    mil.need(collected['status'] == 'COLLECTED_ARTIFACTS_VERIFIED'
             and collected['spec_sha256'] == spec.sha256
             and collected['receipt_sha256'] == export['receipt_sha256'], 'arm collection binding differs')
    summary = build.old.check_outputs(spec, out, smoke=smoke)
    prov = mil.read(out/'provenance.json')
    identity = prov['identity']
    mil.need(receipt['metrics']['engineering_smoke'] == int(smoke)
             and prov['rendering'] == spec.document['config']['windows']['laterality_v1'], 'arm rendering/mode differs')
    mil.need(identity['workers'] == 2 and identity['mode'] == ('smoke' if smoke else 'full')
             and identity['device'] == ('cpu' if smoke else 'cuda') and identity['batch'] == (2 if smoke else 8),
             'arm registered loader/device differs')
    note_root = run/('cpu' if smoke else 'pass')
    note = mil.read(note_root/('verification.json' if smoke else 'verification-pass.json'))
    life.verify_evidence(spec, note, note_root)
    mil.need(life.content_sha256(note) == receipt['verification_sha256'], 'arm receipt evidence differs')
    if not smoke:
        mil.need(receipt['verification_sha256'] == prov['verification_sha256'], 'arm fit evidence differs')
        life.require_verification_pass(note, **spec.verification_args)
        life.check_target_observation(spec, summary['observation'])
        roots = spec.document['config']['roots']
        mil.need(prov['train_ids_ordered_sha256'] == roots['train_sha256']
                 and prov['eval_ids_ordered_sha256'] == roots['val_sha256']
                 and prov['checkpoint_selection'] == 'fixed-final', 'arm heldout ancestry differs')
    files = (base/'receipt.json', base/'collection.json', out/'model_best.pt', out/shared.PRIMARY,
             out/'provenance.json', out/'summary.json', out/'output-inventory.json', build.spec_path(seed))
    return dict(key=key(kind, seed), seed=seed, spec_sha256=spec.sha256, engineering_smoke=smoke,
        runtime_sha256=life.content_sha256(identity['observation']['runtime']),
        checkpoint=(out/'model_best.pt').relative_to(ROOT).as_posix(),
        prediction=(out/shared.PRIMARY).relative_to(ROOT).as_posix(),
        bindings={p.relative_to(ROOT).as_posix(): mil.sha(p) for p in files})


def all_snapshots(smoke=False):
    snaps = {key(k, s): snapshot(k, s, smoke) for k, s in PAIRS}
    mil.need(len({s['runtime_sha256'] for s in snaps.values()}) == 1, 'paired fits ran on different runtimes')
    return snaps


def rendering(kind):
    if kind == 'control':
        return dict(adapter='identity', policy='shared control pixels unchanged')
    mil.need(kind == 'right', 'unknown serving kind')
    result = side.validate()
    mil.need(result['sha256'] == OAI_MAP_SHA256, 'registered OAI side map digest differs')
    return dict(adapter='hb123-oai-canonical/v1', policy=lat.POLICY,
                side_map=result['path'], side_map_sha256=result['sha256'],
                transform='before window selection/resize/TTA; sagittal and slice order unchanged')


def member_recipe(kind, item):
    return {**shared.member_recipe(item), 'rendering': rendering(kind)}


class CanonicalCache:
    def __init__(self, base, sides, ids):
        sides.require(ids)
        self.base, self.sides = base, sides

    def __getattr__(self, name):
        if name in ('base', 'sides'):
            raise AttributeError(name)
        return getattr(self.base, name)

    def get(self, uid):
        volume, mask = self.base.get(uid)
        return lat.canonical_volume(volume, self.sides.side(uid), self.base.spec['slots']), mask


def serving_cache(kind, base, contract, ids):
    mil.need(contract == rendering(kind), 'unregistered panel rendering')
    if kind == 'control':
        return base
    return CanonicalCache(base, lat.SideMap(ROOT/contract['side_map'], contract['side_map_sha256']), ids)


def recipe_path(smoke=False):
    return WORK/('panel-smoke-recipes-v1.json' if smoke else 'panel-recipes-v1.json')


def panel_prepare(smoke=False):
    snaps = all_snapshots(smoke)  # Fail before any registration if a control is incomplete.
    p, ids, route, pins = shared.panel_context()
    recipes = {key(k, s): member_recipe(k, snaps[key(k, s)]) for k, s in PAIRS}
    doc = dict(version=VERSION, sources=source_pins(), panel_pins=pins, recipes=recipes,
        rows=len(ids), ordered_ids_sha256=mil.uid_hash(ids), routing=route, engineering_smoke=smoke)
    build.same_or_new(recipe_path(smoke), doc)
    return dict(status='PANEL_RECIPES_READY', members=4, rows=len(ids), engineering_smoke=smoke)


def admit_panel(kind, seed, smoke=False):
    p, ids, route, pins = shared.panel_context()
    doc = mil.read(recipe_path(smoke))
    mil.need(doc['version'] == VERSION and doc['sources'] == source_pins() and doc['panel_pins'] == pins
             and doc['engineering_smoke'] is smoke and doc['ordered_ids_sha256'] == mil.uid_hash(ids)
             and doc['rows'] == len(ids) and doc['routing'] == route
             and set(doc['recipes']) == {key(k, s) for k, s in PAIRS}, 'panel environment drift')
    # Even a single arm inference requires all paired controls to authenticate.
    snaps = all_snapshots(smoke)
    for k, s in PAIRS:
        mil.need(doc['recipes'][key(k, s)] == member_recipe(k, snaps[key(k, s)]), 'panel member drift')
    return p, ids, doc, doc['recipes'][key(kind, seed)]


def panel_file(kind, seed, smoke=False):
    return WORK/('panel-smoke' if smoke else 'panel-predictions-v1')/(key(kind, seed)+'.csv')


def panel_prediction(kind, seed, smoke=False, admitted=None):
    p, all_ids, doc, recipe = admitted or admit_panel(kind, seed, smoke)
    ids = all_ids[:2] if smoke else all_ids
    path = panel_file(kind, seed, smoke)
    receipt = mil.read(str(path)+'.json')
    shared.authenticate_panel_receipt(receipt, recipe, doc, ids, mil.sha(path), smoke)
    if not smoke:
        life.check_target_observation(spec_for(kind, seed), receipt['observation'])
    return ids, shared.load_predictions(path, ids, 'study_key'), receipt


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
    observed, spec = mil.observation(device), spec_for(kind, seed)
    if not smoke:
        life.check_target_observation(spec, observed)
    ck = p.load_checkpoint(ROOT/recipe['checkpoint'])
    mil.need(ck['lab'] == mil.TARGETS and ck['arch'] == mil.ARCH and ck['fold'] == 0
             and ck['model_kind'] == 'finding_spatial_tokens_v1'
             and ck['provenance']['spec_sha256'] == spec.sha256
             and ck['init_weights_sha256'] == spec.document['config']['model']['weights_sha256']
             and ck['epoch'] == (0 if smoke else 19), 'panel checkpoint metadata differs')
    if kind == 'right':
        mil.need(ck['provenance']['rendering'] == spec.document['config']['windows']['laterality_v1'],
                 'checkpoint canonical rendering differs')
    del ck
    model = p.model_for(recipe).to(device)
    cache = serving_cache(kind, p.Cache('64'), recipe['rendering'], ids)
    use_ids = ids[:2] if smoke else ids
    start = time.monotonic()
    values = np.asarray([p.predict_one(model, recipe, cache, uid, device) for uid in use_ids])
    mil.need(values.shape == (len(use_ids), 12) and np.isfinite(values).all()
             and ((values >= 0) & (values <= 1)).all(), 'invalid panel forward')
    mil.need(smoke or len(ids) == 557, 'full panel size differs')
    mil.need(snapshot(kind, seed, smoke) == recipe['snapshot'] and source_pins() == doc['sources']
             and rendering(kind) == recipe['rendering'], 'panel inputs changed during inference')
    life.create_new(destination, life.prediction_csv_bytes(use_ids, values, mil.TARGETS, id_column='study_key'))
    receipt = dict(version=VERSION, recipe=recipe, sources=doc['sources'], panel_pins=doc['panel_pins'],
        recipe_file_sha256=mil.sha(recipe_path(smoke)), predictions_sha256=mil.sha(destination),
        rows=len(use_ids), ids_sha256=mil.uid_hash(use_ids), engineering_smoke=smoke,
        full_panel=not smoke, device=device, observation=observed,
        seconds=time.monotonic()-start, volume_files_verified=len(cache.verified))
    build.same_or_new(str(destination)+'.json', receipt)
    panel_prediction(kind, seed, smoke, admitted)
    mil.need(not smoke or not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    return dict(status='PANEL_INFERRED', key=recipe['key'], rows=len(use_ids), seconds=receipt['seconds'])


def readout(smoke=False):
    snaps = all_snapshots(smoke)
    sources = source_pins()
    panels, receipts, ids_panel = {}, {}, None
    for kind, seed in PAIRS:
        ids, values, _ = panel_prediction(kind, seed, smoke)
        mil.need(ids_panel is None or ids == ids_panel, 'paired panel roster differs')
        ids_panel = ids
        panels.setdefault(kind, {})[str(seed)] = values
        receipts[key(kind, seed)] = mil.sha(str(panel_file(kind, seed, smoke))+'.json')
    from tools import oai_moaks_panel as p
    y = p.outcomes(ids_panel)
    # This historical module installs tripwires; import only after receipt auth.
    from tools import hb105_aligned_oof_v1 as aligned
    raw, rosters, auth = aligned.authenticate()
    ids = rosters[0]['val']
    arrays = {kind: {str(seed): (raw[0]['S'+str(seed)].copy() if smoke else
        shared.load_predictions(ROOT/snaps[key(kind, seed)]['prediction'], ids)) for seed in (42, 43)}
        for kind in ('control', 'right')}
    arm, control = fixed_bundle(raw, arrays['right'], arrays['control'], aligned.variants)
    ctx = shared.ml.make_context(ids, ROOT)
    mil.need(ctx.bindings == auth['ruler_bindings'], 'ruler binding changed')
    bundle = dict(context=ctx, arm=arm, control=control, recipe=dict(scope='ALIGNED_OOF_SURROGATE_ONLY',
        change='only fold0 S42/S43; folds1-4 fixed', arithmetic=aligned.POLICY['arithmetic'],
        scoring='fold0 slice of fixed pooled predictions; no fold0 reranking', final_proxy=shared.FINAL_PROXY))
    result = evaluate(ctx, arrays['right'], arrays['control'], bundle,
                      image_item(ids_panel, panels['right'], panels['control'], y))
    result.update(version=VERSION, status='ENGINEERING_CONTROL_REPLAY' if smoke else 'COMPLETE_FOLD0_SCREEN',
        sources=sources, fitted_members=snaps, panel_receipts=receipts, oai_side_map=side.validate(),
        aligned_authentication=auth, ruler_bindings=ctx.bindings, rows=len(ids), panel_rows=len(ids_panel),
        ordered_ids_sha256=mil.uid_hash(ids), engineering_smoke=smoke,
        replay_reason='CPU fits cannot supply scientific heldout OOF; authentic deployed S replayed on both sides' if smoke else None)
    for item in snaps.values():
        for rel, digest in item['bindings'].items():
            mil.need(mil.sha(ROOT/rel) == digest, 'member changed during scoring')
    mil.need(source_pins() == sources, 'readout source drift')
    for kind, seed in PAIRS:
        mil.need(mil.sha(str(panel_file(kind, seed, smoke))+'.json') == receipts[key(kind, seed)], 'panel receipt drift')
    output = WORK/('screen-cpu-smoke-v1.json' if smoke else 'screen-seven-lens-v1.json')
    build.same_or_new(output, result)
    return dict(status=result['status'], output=output.relative_to(ROOT).as_posix(),
                rows=len(ids), panel_rows=len(ids_panel), lenses=list(result['lenses']))


def waiting():
    missing = []
    for kind, seed in PAIRS:
        for path in (run_for(kind, seed)/'collected/receipt.json', panel_file(kind, seed),
                     Path(str(panel_file(kind, seed))+'.json')):
            if not path.is_file():
                missing.append(path.relative_to(ROOT).as_posix())
    result = dict(status='WAITING_FOR_COMPLETE_FITS_AND_PANEL' if missing else 'READY_FOR_AUTHENTICATION',
        missing=missing, scientific_readout_written=False, existence_check_only=True)
    shared.ml.write(WORK/'screen-waiting-v1.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['panel-prepare', 'panel-infer', 'admit', 'readout', 'waiting'])
    parser.add_argument('--kind', choices=['control', 'right'], default='right')
    parser.add_argument('--seed', type=int, choices=[42, 43], default=42)
    parser.add_argument('--engineering-smoke', action='store_true')
    a = parser.parse_args()
    if a.command == 'waiting': result = waiting()
    elif a.command == 'panel-prepare': result = panel_prepare(a.engineering_smoke)
    elif a.command == 'readout': result = readout(a.engineering_smoke)
    elif a.command == 'admit':
        admit_panel(a.kind, a.seed, a.engineering_smoke)
        result = dict(status='PANEL_ADMITTED', key=key(a.kind, a.seed), engineering_smoke=a.engineering_smoke)
    else: result = panel_infer(a.kind, a.seed, a.engineering_smoke)
    print(life.canonical_bytes(result).decode())


if __name__ == '__main__':
    main()
