"""Legacy artifact compatibility v2.0.0.

Changelog 2.0.0: retire source generation and fresh launches; retain historical
validation, collection and scientific readouts. Timing is measurement only.

EXP191B PA_PFOA, v1.0.0: frozen EXP191 recipe, PF OA-only label overlay.

Changelog 1.0.0: project authenticated PA labels onto CONTROL PF OA; mandatory
control reuse, uncapped serial Vast runs, paired seed-mean component readout.
Historical EXP191 verification uses explicit family configuration; no generated
runner or cloned function namespace remains available for fresh execution.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp191_softprior as parent

VERSION = '2.0.0'
HISTORICAL_VERSION = '1.0.0'
EXP = 'artifacts/experiments/EXP191B_PFOA_SOFTPRIOR'
TOOL = 'tools/exp191b_pfoa_softprior.py'
RUNNER = 'training/runners/exp191b_pfoa_softprior.py'
PROTOCOL = EXP + '/PROTOCOL.md'
LABEL = 'storage/labels/derived_constellation/l2_exp191b_pa_pfoa.csv'
SOURCE_OVERLAY = '.local/exp191-overlay-v101'
SOURCE_SHA = '76f9a9a78a9f0ffa91e2c09a254587f3395a398af9b4d4b04504c18e09d26407'
PA_SHA = '82dc31d8b09d19548ed907de60d869a3f045a98bfd0ca3e0f1ddcc667965d44d'
REMOTE_ROOT = '/workspace/exp191b'
REMOTE_RESULTS = '/workspace/exp191b-results'
ARM = 'PA_PFOA'
SEEDS = (42, 43)
CONTROL_PREDICTIONS = {
    42: '632413a0749fead9a102390aa2e037a0527d7e8cc2ee8f39271065626dc94c3b',
    43: '8235271d50ee1616c7a3a60acae51d4dffb797feda1098010eb7a370436b4478'}
CONTROL_PROTOCOLS = {
    42: 'bf1a1521c9fcb37d20617b0cdc3853ccb50935bf43e9cae81d28e049d50a8a0f',
    43: 'b163a14d87581bbc84957167502c62c71ea1724ef6d2776a0185043b1cd7f34b'}
TARGETS = parent.TARGETS
need, read, sha, write = parent.need, parent.read, parent.sha, parent.write




SCHEMA = 'exp191b-pfoa-v1'




def tag(arm, seed):
    need(arm == ARM and seed in SEEDS, 'only PA_PFOA seeds 42/43 are permitted')
    return f'EXP191B_PFOA_SOFTPRIOR_{arm}_S{seed}_F0'



def historical_config():
    return parent.SoftpriorHistory(SCHEMA, HISTORICAL_VERSION, {ARM: LABEL}, TOOL,
        RUNNER, REMOTE_ROOT, REMOTE_RESULTS, tag, validate)


def protocol_path(arm, seed):
    tag(arm, seed)
    return parent.protocol_path(arm, seed)


def _protocol(arm, seed, package, manifest, runner_sha, tool_sha):
    return parent._protocol(arm, seed, package, manifest, runner_sha, tool_sha,
                            history=historical_config())


def verify_run(out, arm, seed, package=ROOT):
    return parent.verify_run(out, arm, seed, package, history=historical_config())


def export(seed, out, root=ROOT):
    return parent.export(ARM, seed, out, root, history=historical_config())


def vast_stage(*args, **kwargs):
    raise ValueError('Legacy MIL vast_stage is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.')


def project_labels(control_bytes, pa_bytes):
    """Copy CSV cell strings; no numeric round-trip of unchanged cells."""
    import csv
    import io
    def rows(data):
        return list(csv.reader(io.StringIO(data.decode('utf-8'), newline='')))
    old, prior = rows(control_bytes), rows(pa_bytes)
    need(old[0] == prior[0] == [parent.fill.UID, *TARGETS], 'label header differs')
    need(len(old) == len(prior) and len({r[0] for r in old[1:]}) == len(old)-1,
         'label roster differs')
    col = old[0].index('PF OA')
    changed = 0
    result = [old[0]]
    for a, b in zip(old[1:], prior[1:]):
        need(len(a) == len(b) == 13 and a[0] == b[0], 'label row order differs')
        c = list(a)
        c[col] = b[col]
        changed += c[col] != a[col]
        result.append(c)
    need(parent._text(old[0], old[1:]) == control_bytes, 'CONTROL CSV dialect differs')
    return parent._text(result[0], result[1:]), {
        'rows': len(old)-1, 'changed_cells': changed,
        'changed_by_target': {t: changed if t == 'PF OA' else 0 for t in TARGETS},
        'untouched_cells_byte_identical': True}


def build_labels(root=ROOT, dry_run=False):
    root = Path(root)
    control, prior = root / parent.L2, root / parent.LABELS['PA']
    need(sha(control) == parent.fill.L2_SHA and sha(prior) == PA_SHA, 'pinned labels changed')
    data, qa = project_labels(control.read_bytes(), prior.read_bytes())
    need(qa['changed_cells'] == 717 and qa['rows'] == 4347, 'registered label counts differ')
    qa.update(schema='exp191b-labels-v1', sha256=parent.digest_bytes(data),
              inputs={parent.L2: sha(control), parent.LABELS['PA']: sha(prior)},
              protocol_sha256=sha(root / PROTOCOL), eligible_pf_oa_cells=749,
              formula='authenticated EXP191 PA PF OA projection; silent cells := float32(aquila)',
              forbidden_relabel_access=False)
    if not dry_run:
        dest = root / LABEL
        need(not dest.exists() and not dest.with_suffix('.qa.json').exists(), 'fresh labels required')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(data)
        write(dest.with_suffix('.qa.json'), qa)
    return qa


def controls(root=ROOT):
    root = Path(root)
    source = root / SOURCE_OVERLAY
    need(sha(source / 'manifest.json') == SOURCE_SHA, 'EXP191 source manifest changed')
    old = read(source / 'manifest.json')
    result = {}
    for seed in SEEDS:
        record = parent.control_reuse(seed, root)
        expected = dict(old['controls'][str(seed)])
        for r in (record, expected):
            r['path'] = r['path'].replace('\\', '/')
        need(record == expected and not record.get('retrain'), 'EXP191 CONTROL reuse failed')
        result[str(seed)] = record
    return result


def queue(tasks=None):
    tasks = [(ARM, s) for s in SEEDS] if tasks is None else tasks
    return [{'tag': tag(a, s), 'command': ['python', TOOL, 'run', '--seed', str(s)],
             'continue_on_failure': False} for a, s in tasks]


def readout_context():
    from tools.exp179_screen import load_context
    return load_context(ROOT)


def prepare(out, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL prepare is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def validate(package, full=False):
    package = Path(package)
    m = read(package / 'manifest.json')
    need(m['schema'] == SCHEMA and m['version'] == HISTORICAL_VERSION and
         m['source_root'] == parent.SOURCE_ROOT and m['remote_root'] == REMOTE_ROOT and
         m['exp191_manifest_sha256'] == SOURCE_SHA and m['wall_time_limit'] is None and
         m['tasks'] == [{'arm': ARM, 'seed': s} for s in SEEDS] and
         set(m['runs']) == {tag(ARM, s) for s in SEEDS} and
         read(package / 'queue.json') == queue(), 'manifest/task contract differs')
    required = {TOOL, RUNNER, PROTOCOL, LABEL, parent.L2, parent.TOOL, parent.FILL_TOOL,
                parent.MODEL, parent.INIT, 'queue.json'}
    need(set(m['files']) >= required, 'transitive files missing')
    need(m['files'][parent.INIT]['sha256'] == parent.digest_bytes(b''), 'tools package differs')
    for name, entry in m['files'].items():
        parent.safe(name)
        need(entry['origin'] in ('exp179', 'overlay'), 'unknown origin')
        if full or entry['origin'] == 'overlay':
            p = package / name
            need(p.is_file() and not p.is_symlink() and p.stat().st_size == entry['bytes'] and
                 sha(p) == entry['sha256'], 'package drift: ' + name)
    need(sha(package / PROTOCOL) == m['protocol_sha256'] and
         sha(package / LABEL) == m['label_diff']['sha256'] and
         m['label_diff']['changed_by_target'] == {t: 717 if t == 'PF OA' else 0 for t in TARGETS},
         'protocol/label identity differs')
    control_path = parent.mil.inherited_path(package, parent.L2)
    prior_path = parent.mil.inherited_path(package, parent.LABELS['PA'])
    need(sha(control_path) == parent.fill.L2_SHA and sha(prior_path) == PA_SHA,
         'parent label identity differs')
    projected, _ = project_labels(control_path.read_bytes(), prior_path.read_bytes())
    need((package / LABEL).read_bytes() == projected and
         read((package / LABEL).with_suffix('.qa.json')) == m['label_diff'], 'PF OA projection differs')
    for seed in SEEDS:
        run = m['runs'][tag(ARM, seed)]
        need(run['arm'] == ARM and run['seed'] == seed and
             run['protocol'] == protocol_path(ARM, seed), 'run identity differs')
        q = read(package / run['protocol'])
        expected = _protocol(ARM, seed, package, m, sha(package / RUNNER), sha(package / TOOL))
        expected['label_set']['name'] = 'l2_exp191b_pa_pfoa'
        expected['exp191']['exp191b'] = {'parent_manifest_sha256': SOURCE_SHA, 'target': 'PF OA'}
        need(q == expected and sha(package / run['protocol']) == run['sha256'], 'frozen recipe differs')
        need(m['controls'][str(seed)] == {
            'path': f'artifacts/experiments/EXP185_FINDING_MIL/outputs/S{seed}/results/EXP185_FINDING_MIL_F0_S{seed}',
            'prediction_sha256': CONTROL_PREDICTIONS[seed], 'protocol_sha256': CONTROL_PROTOCOLS[seed],
            'manifest_sha256': 'f121fc26001cd29831e438801b2ab7c5389c0004fcc3106a8976ba327988661e'},
            'CONTROL identity changed or retraining requested')
    if full:
        need(str(package.resolve()) == REMOTE_ROOT and
             sha(package / 'source-manifest.json') == m['source_manifest_sha256'], 'remote source differs')
    return m


def validate_arguments(a, root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL validate_arguments is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def admit(package=ROOT, dry_run=False):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL admit is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def run(seed):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL run is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def collect(archive, overlay, out, seed):
    """Authenticate the historical EXP191B family without cloned globals."""
    return parent.collect(archive, overlay, out, ARM, seed, history=historical_config())


def paired_readout(candidates, control, refs, draws, evaluator):
    import numpy as np
    from tools import exp179_screen as helper
    names = ('jev', 'consensus', 'dread', 'aquila')
    need(set(candidates) == set(control) == set(SEEDS) and len(draws) == 300, 'paired seeds/draws differ')
    shape = candidates[42].shape
    need(len(shape) == 2 and shape[1] == 12 and all(
        p.shape == shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
         for p in [*candidates.values(), *control.values()]), 'prediction arrays differ')
    refs = helper._references(refs, shape)
    draws = helper._draws(draws, shape[0])
    def score(seed, ix):
        values = {}
        for name in names:
            y, mask, minimum = refs[name]
            a = evaluator.auc_targets(y[ix], candidates[seed][ix], mask[ix], minimum)
            b = evaluator.auc_targets(y[ix], control[seed][ix], mask[ix], minimum)
            values[name] = {'control': b, 'candidate': a, 'delta': a-b}
        values['mean_jev_consensus'] = {k: (values['jev'][k]+values['consensus'][k])/2
                                       for k in ('control', 'candidate', 'delta')}
        return values
    full = np.arange(shape[0])
    per = {s: score(s, full) for s in SEEDS}
    avg = {r: {k: np.mean([per[s][r][k] for s in SEEDS], axis=0)
               for k in ('control', 'candidate', 'delta')} for r in (*names, 'mean_jev_consensus')}
    boot = {r: [] for r in avg}
    seed_boot = {s: {r: [] for r in avg} for s in SEEDS}
    pf = list(TARGETS).index('PF OA')
    for ix in draws:
        sampled = [score(s, ix) for s in SEEDS]
        for r in avg:
            boot[r].append(np.mean([v[r]['delta'][pf] for v in sampled]))
            for s, v in zip(SEEDS, sampled):
                seed_boot[s][r].append(v[r]['delta'][pf])
    def serial(values):
        return {r: {**{k: {t: float(a[i]) if np.isfinite(a[i]) else None for i, t in enumerate(TARGETS)}
                           for k, a in v.items()},
                    'macro_delta': float(np.mean(v['delta'])) if np.isfinite(v['delta']).all() else None}
                for r, v in values.items()}
    mean_delta = avg['mean_jev_consensus']['delta']
    harm = dict(macro_ge_minus_0_001=bool(np.isfinite(mean_delta).all() and mean_delta.mean() >= -.001),
        other_targets_ge_minus_0_010=bool(all(np.isfinite(mean_delta[i]) and mean_delta[i] >= -.010
                                            for i, t in enumerate(TARGETS) if t != 'PF OA')),
        dread_macro_ge_minus_0_003=bool(np.isfinite(avg['dread']['delta']).all() and avg['dread']['delta'].mean() >= -.003))
    success = bool(np.isfinite(mean_delta[pf]) and mean_delta[pf] >= .002 and
                   all(per[s]['mean_jev_consensus']['delta'][pf] > 0 for s in SEEDS))
    def intervals(values):
        return {r: {'valid_draws': int(np.isfinite(v).sum()), 'ci95': np.percentile(v, [2.5, 97.5]).tolist()
                    if np.isfinite(v).all() else None} for r, v in values.items()}
    return dict(status='SUCCESS' if success and all(harm.values()) else 'HOLD',
        primary_success=success, harm_guard=harm, per_seed={str(s): serial(per[s]) for s in SEEDS},
        seed_average=serial(avg), pf_oa_component_bootstrap=intervals(boot),
        per_seed_pf_oa_component_bootstrap={str(s): intervals(seed_boot[s]) for s in SEEDS},
        draws=300, bootstrap_seed=20261003,
        averaging='arithmetic mean of paired per-seed AUCs/deltas, not prediction ensemble')


def screen(overlay, outputs, out, replay=False):
    from tools import exp179_screen as helper, exp115_vlm_eval as ev, exp169_bundle_screen as bundle
    overlay, outputs, out = map(Path, (overlay, outputs, out))
    m = validate(overlay)
    need(not out.exists() and controls() == m['controls'], 'fresh screen/current controls required')
    ids, refs, draws, context = readout_context()
    need(len(ids) == 888 and len(draws) == 300 and context['bootstrap_seed'] == 20261003, 'readout context differs')
    need(context == m['readout_context'], 'preregistered readout inputs changed')
    controls_pred, candidates, hashes, pairs = {}, {}, {}, {}
    for seed in SEEDS:
        path = ROOT / m['controls'][str(seed)]['path'] / 'oof_k36_flip_tta.csv'
        controls_pred[seed] = bundle.load_oof(path, ids)
        control_file = {'path': str(path), 'sha256': sha(path)}
        if replay:
            candidates[seed] = controls_pred[seed].copy()
        else:
            collected = outputs / f'{ARM}-S{seed}'
            receipt = read(collected / 'collection.json')
            need(receipt['status'] == 'VERIFIED' and receipt['arm'] == ARM and receipt['seed'] == seed and
                 receipt['manifest_sha256'] == sha(overlay / 'manifest.json'), 'candidate not collected')
            run_dir = collected / 'results' / tag(ARM, seed)
            verify_run(run_dir, ARM, seed, overlay)
            path = run_dir / 'oof_k36_flip_tta.csv'
            candidates[seed] = bundle.load_oof(path, ids)
        hashes[str(seed)] = sha(path)
        pairs[str(seed)] = {'arm': {'path': str(path), 'sha256': sha(path)}, 'control': control_file}
    result = paired_readout(candidates, controls_pred, refs, draws, ev)
    result.update(schema='exp191b-screen-v1', context=context, manifest_sha256=sha(overlay / 'manifest.json'),
                  candidate_sha256=hashes, controls=m['controls'], engineering_replay=replay,
                  limitations=['exposed fold-0 weak/report references; no private LB estimate',
                               'aquila partially self-referenced', 'SUCCESS permits confirmation, not release'])
    if replay:
        result['status'] = 'ENGINEERING_REPLAY_CONTROL_VS_ITSELF'
    parent.mil.attach_holistic_readout(result, {
        'name': 'EXP191B historical paired screen', 'family': 'convnext',
        'label_change': True, 'seeds': pairs,
        'partners': parent.mil.historical_partners(ROOT)}, ROOT)
    out.mkdir(parents=True)
    write(out / 'result.json', result)
    return {'status': result['status'], 'path': str(out / 'result.json')}


def smoke(overlay, out, seed=42):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL smoke is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


# Historical family identity is supplied explicitly; no runtime namespace cloning.



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('build-labels'); a.add_argument('--dry-run', action='store_true')
    a = sub.add_parser('prepare'); a.add_argument('--out', required=True)
    a = sub.add_parser('validate'); a.add_argument('--package', required=True); a.add_argument('--full', action='store_true')
    a = sub.add_parser('vast-stage'); a.add_argument('--overlay', required=True)
    a = sub.add_parser('admit'); a.add_argument('--package', default=str(ROOT)); a.add_argument('--dry-run', action='store_true')
    a = sub.add_parser('run'); a.add_argument('--seed', type=int, choices=SEEDS, required=True)
    a = sub.add_parser('export'); a.add_argument('--seed', type=int, choices=SEEDS, required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('collect'); a.add_argument('--seed', type=int, choices=SEEDS, required=True)
    a.add_argument('--archive', required=True); a.add_argument('--overlay', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('smoke'); a.add_argument('--overlay', required=True); a.add_argument('--out', required=True)
    a.add_argument('--seed', type=int, choices=SEEDS, default=42)
    for name in ('screen', 'screen-replay'):
        a = sub.add_parser(name); a.add_argument('--overlay', required=True); a.add_argument('--out', required=True)
        a.add_argument('--outputs', default=EXP + '/outputs')
    a = p.parse_args(argv)
    if a.command == 'build-labels': result = build_labels(dry_run=a.dry_run)
    elif a.command == 'prepare': result = prepare(a.out)
    elif a.command == 'validate':
        validate(a.package, a.full); result = {'status': 'VERIFIED_PACKAGE'}
    elif a.command == 'vast-stage': result = vast_stage(a.overlay)
    elif a.command == 'admit': result = admit(a.package, a.dry_run)
    elif a.command == 'run': result = run(a.seed)
    elif a.command == 'export': result = export(a.seed, a.out)
    elif a.command == 'collect': result = collect(a.archive, a.overlay, a.out, a.seed)
    elif a.command == 'smoke': result = smoke(a.overlay, a.out, a.seed)
    else: result = screen(a.overlay, a.outputs, a.out, replay=a.command == 'screen-replay')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
