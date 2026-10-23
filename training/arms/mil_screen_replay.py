"""Immutable collected-MIL screen replay v1.0.0 (2026-10-11).

Changelog: migrate EXP185/EXP186B historical readouts to the explicit screen
engine, normalizing legacy Windows paths at registration. Bind all predictions,
evaluation inputs and executable sources; never change published evidence.
This replay certifies numerical parity, not new training or model admission.
No model loading, credential/report-text access, timers or launch operations.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PureWindowsPath

import numpy as np
from training.arms import lifecycle as life
from training.arms import mil_screen as engine
from tools import exp115_vlm_eval as ev
from tools import exp149_label_sources as rosters
from tools import exp156_k_sweep as labels
from tools import exp179_screen as context_helper
from tools.exp162_dino_folds_kaggle import TARGETS

VERSION = '1.0.0'
SCHEMA = 'mil-screen-replay/v1'
HISTORIES = {
    'mean-seed-f0': 'artifacts/experiments/EXP185_FINDING_MIL/screen/result.json',
    'pooled-s42': 'artifacts/experiments/EXP186_FINDING_MIL_5FOLD/screen-b/result.json',
}
SOURCES = ('training/arms/mil_screen.py', 'training/arms/mil_screen_replay.py',
           'training/arms/lifecycle.py', 'tools/mil_screen.py',
           'tools/exp115_vlm_eval.py', 'tools/exp149_label_sources.py',
           'tools/exp156_k_sweep.py', 'tools/exp162_dino_folds_kaggle.py',
           'tools/exp169_bundle_screen.py', 'tools/exp170_convnext_t_l2.py',
           'tools/exp179_screen.py', 'tools/exp113_radr50_local.py',
           'tools/exp127_radr50_single.py', 'tools/exp134_aug1_kfold.py')
need = engine.need


def read(path):
    return life.parse_json(Path(path).read_bytes())


def portable_path(name, root):
    """Normalize legacy inputs once; an absolute path must be under this root."""
    need(type(name) is str, 'path must be text')
    value = name.replace('\\', '/')
    if PureWindowsPath(value).is_absolute() or value.startswith('/'):
        prefix = Path(root).resolve().as_posix().rstrip('/') + '/'
        need(value.casefold().startswith(prefix.casefold()), 'legacy path outside root')
        value = value[len(prefix):]
    life.safe_member(value)
    return value


def path(root, name):
    life.safe_member(name)
    base = Path(root).resolve()
    target = base
    for part in name.split('/'):
        target /= part
        need(not target.is_symlink() and not target.is_junction(), 'linked input forbidden')
    need(target.resolve().is_relative_to(base), 'input outside root')
    return target


def hashed(root, name):
    return life.file_sha256(path(root, name))


def context(root, kind):
    """Re-use frozen roster/label policies; only aggregates and hashes escape."""
    need(kind in HISTORIES, 'unknown replay kind')
    manifest = rosters._load_manifest('student', root)
    canonical, fold_of = rosters.fold_rosters(manifest, root)
    by_fold = {f: [u for u in canonical if fold_of[u] == f] for f in range(5)}
    ids = by_fold[0] if kind == 'mean-seed-f0' else [u for f in range(5) for u in by_fold[f]]
    need(len(ids) == len(set(ids)) == (888 if kind == 'mean-seed-f0' else 4343), 'roster differs')
    reference_paths = [('jev', labels.JEV), ('aquila', labels.AQUILA), ('dread', labels.DREAD),
                       *[(f'consensus_{i}', p) for i, p in enumerate(labels.CONSENSUS)]]
    context_inputs = {rosters.STUDENT_MANIFEST, manifest['paths']['base_ordered_ids'], ev.COMPONENTS,
                      rosters.aug.PROTOCOL + '/s42/manifest.json',
                      portable_path(str(rosters.e127.manifest_path('AUG1', 42, Path(root))), root),
                      *(manifest['folds'][str(f)]['ordered_ids']['path'] for f in range(5)),
                      *(p for _, p in reference_paths)}
    # All input paths are fixed metadata or numeric label CSVs, never report text.
    bindings = {p: hashed(root, p) for p in sorted(context_inputs)}
    raw = labels.references(ids, root)
    refs = {name: raw[key] for name, key in (('jev', 'jev_hybrid'), ('aquila', 'aquila_v1'),
                                            ('dread', 'dread'), ('consensus', 'consensus'))}
    refs = context_helper._references(refs, (len(ids), 12))
    draws = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    metadata = dict(ordered_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        components_sha256=hashed(root, ev.COMPONENTS), bootstrap_seed=20261003,
        reference_sha256={name: hashed(root, p) for name, p in reference_paths},
        mask_sha256={name: hashlib.sha256(np.ascontiguousarray(m, dtype=np.uint8).tobytes()).hexdigest()
                     for name, (_, m, _) in refs.items()},
        coverage={name: {t: int(m[:, i].sum()) for i, t in enumerate(TARGETS)}
                  for name, (_, m, _) in refs.items()})
    draw_hash = hashlib.sha256()
    for draw in draws:
        draw_hash.update(len(draw).to_bytes(8, 'little'))
        draw_hash.update(np.asarray(draw, dtype='<i8').tobytes())
    need(bindings == {p: hashed(root, p) for p in bindings}, 'evaluation inputs changed during read')
    return by_fold, ids, refs, draws, metadata, bindings, draw_hash.hexdigest()


def historical_inputs(root, kind, historical):
    """Only known completed screen layouts; preserve the original hash bindings."""
    need(historical['bootstrap'] == dict(unit='validation component', draws=300, seed=20261003),
         'bootstrap contract differs')
    if kind == 'mean-seed-f0':
        need(historical['schema'] == 'exp185-finding-mil-v1-screen' and
             historical['seeds'] == [42, 43] and historical['studies'] == 888, 'historical scope differs')
        pairs = {}
        for seed in (42, 43):
            pairs[f'candidate_S{seed}'] = (f'artifacts/experiments/EXP185_FINDING_MIL/outputs/S{seed}/'
                f'results/EXP185_FINDING_MIL_F0_S{seed}/oof_k36_flip_tta.csv')
            pairs[f'control_S{seed}'] = (f'artifacts/experiments/EXP179_MULTISCALE/outputs/M{seed}/'
                f'results/EXP179_M{seed}_F0/oof_k36_flip_tta.csv')
        pairs.update(R42='artifacts/experiments/EXP149_L2_S42_F0/oof_k36.csv',
            R43='artifacts/experiments/EXP157_L2_S43_F0/oof_k36.csv',
            D172='artifacts/experiments/EXP172_DINO_BASE_L2/outputs/artifacts/experiments/'
                 'EXP172_DINO_BASE_L2_S42_F0/oof_best.csv')
        need(set(historical['input_sha256']) == set(pairs), 'historical inputs differ')
        return {k: dict(path=p, sha256=historical['input_sha256'][k]) for k, p in pairs.items()}
    need(historical['schema'] == 'exp186b-finding-mil-5fold-v1-screen' and
         historical['folds'] == list(range(5)) and historical['seed'] == 42 and
         historical['studies'] == 4343, 'historical scope differs')
    inputs = historical['input_files']
    need(set(inputs) == {f'{name}_f{f}' for f in range(5)
                        for name in ('candidate', 'control', 'rad42', 'rad43', 'dino')}, 'five folds required')
    result = {}
    for key, row in inputs.items():
        p = portable_path(row['path'], root)
        need(p.startswith('artifacts/experiments/') and p.endswith('.csv'), 'unexpected prediction path')
        result[key] = dict(path=p, sha256=row['sha256'])
    return result


def draft(root, kind, historical_sha256):
    need(kind in HISTORIES, 'unknown replay kind')
    historical_path = HISTORIES[kind]
    need(hashed(root, historical_path) == historical_sha256, 'historical receipt differs')
    historical = read(path(root, historical_path))
    inputs = historical_inputs(root, kind, historical)
    for row in inputs.values():
        need(hashed(root, row['path']) == row['sha256'], 'prediction differs from historical receipt')
    _, ids, _, draws, metadata, bindings, draw_hash = context(root, kind)
    if kind == 'mean-seed-f0':
        need(metadata == historical['context'], 'historical evaluation context differs')
    else:
        for key in ('ordered_ids_sha256', 'components_sha256', 'reference_sha256'):
            need(metadata[key] == historical[key], 'historical evaluation context differs')
        need({name: list(row.values()) for name, row in metadata['coverage'].items()} ==
             historical['mask_coverage'], 'historical mask coverage differs')
    return dict(schema=SCHEMA, version=VERSION, kind=kind, studies=len(ids),
        historical=dict(path=historical_path, sha256=historical_sha256), predictions=inputs,
        context=metadata, context_inputs=bindings, draws_sha256=draw_hash,
        bootstrap=dict(draws=len(draws), seed=20261003, unit='validation component'),
        sources={p: hashed(root, p) for p in SOURCES},
        scientific_status='HISTORICAL_REPLAY_ONLY', launch_authorized=False)


def register(root, kind, historical_sha256, out):
    spec = draft(root, kind, historical_sha256)
    life.create_new(out, life.canonical_bytes(spec))
    return spec


def validate(root, spec):
    need(type(spec) is dict and spec.get('schema') == SCHEMA and spec.get('kind') in HISTORIES,
         'invalid replay spec')
    expected = draft(root, spec['kind'], spec['historical']['sha256'])
    need(life.canonical_bytes(spec) == life.canonical_bytes(expected), 'replay registration differs')
    return expected


def screen(root, spec, out):
    """Publish only a complete replay with exact historical numerical parity."""
    need(not Path(out).exists(), 'fresh versioned screen required')
    validate(root, spec)
    by_fold, _, refs, draws, _, _, _ = context(root, spec['kind'])
    def load(key, fold):
        row = spec['predictions'][key]
        need(hashed(root, row['path']) == row['sha256'], 'prediction changed before read')
        # Historical screens used csv + Python float, not pandas' parser.
        # Retain that conversion boundary: near ties can change rank readouts.
        value = engine.bundle.load_oof(path(root, row['path']), by_fold[fold])
        need(hashed(root, row['path']) == row['sha256'], 'prediction changed during read')
        return value
    if spec['kind'] == 'mean-seed-f0':
        numerical = engine.score_seeds({s: load(f'candidate_S{s}', 0) for s in (42, 43)},
            {s: load(f'control_S{s}', 0) for s in (42, 43)},
            {s: load(f'R{s}', 0) for s in (42, 43)}, load('D172', 0), refs, draws, ev)
    else:
        numerical = engine.score_folds({f: {k: load(f'{k}_f{f}', f)
            for k in ('candidate', 'control', 'rad42', 'rad43', 'dino')} for f in range(5)}, refs, draws, ev)
    historical = read(path(root, spec['historical']['path']))
    need(life.canonical_bytes(numerical) == life.canonical_bytes({k: historical[k] for k in numerical}),
         'historical numerical parity differs')
    # Recheck identities after scoring; never publish partial/stale-context results.
    validate(root, spec)
    result = dict(schema='mil-screen-result/v1', version=VERSION, kind=spec['kind'],
        spec_sha256=life.content_sha256(spec), scientific_status='HISTORICAL_REPLAY_ONLY',
        launch_authorized=False, status='PARITY_VERIFIED', numerical=numerical,
        studies=spec['studies'], bootstrap=spec['bootstrap'], draws_sha256=spec['draws_sha256'],
        historical=spec['historical'], sources=spec['sources'])
    life.create_new(out, life.canonical_bytes(result))
    return result
