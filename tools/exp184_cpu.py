"""CPU collect and matched fold-0 screens for EXP184 and EXP182_SYNOVITIS_FILL.

No label rows, report text, GPU, network, Kaggle push or submission are used.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEAK = ('Lateral Meniscus', 'PF OA', 'Lateral OA', 'Synovitis')


def module(experiment):
    if experiment == 'exp184':
        from tools import exp184_aquila_share as tool
    elif experiment == 'exp182':
        from tools import exp182_synovitis_fill as tool
    else:
        raise ValueError('experiment must be exp184 or exp182')
    return tool


def harm_guard(pair, targets):
    j, c = pair['jev'], pair['consensus']
    per = {target: ((j['per_target_delta'][target] + c['per_target_delta'][target]) / 2
                    if j['per_target_delta'][target] is not None and
                    c['per_target_delta'][target] is not None else None)
           for target in targets}
    macro = ((j['delta'] + c['delta']) / 2 if j['delta'] is not None and
             c['delta'] is not None else None)
    dread = pair['dread']['delta']
    checks = {
        'macro12_at_least_minus_0_001': macro is not None and macro >= -.001,
        'other_targets_at_least_minus_0_010': all(per[t] is not None and per[t] >= -.010
                                               for t in targets if t not in WEAK),
        'dread_macro_at_least_minus_0_003': dread is not None and dread >= -.003,
    }
    return dict(status='PASS' if all(checks.values()) else 'HOLD_REVIEW',
                checks=checks, macro12_mean_jev_consensus_delta=macro,
                per_target_mean_jev_consensus_delta=per, dread_macro_delta=dread)


def _finite(value):
    import numpy as np
    return float(value) if np.isfinite(value) else None


def _interval(values, minimum_fraction=.90):
    """Never silently interpolate an interval from mostly undefined draws."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    good = values[np.isfinite(values)]
    required = max(1, int(np.ceil(minimum_fraction * len(values))))
    return (np.percentile(good, [2.5, 97.5]).tolist() if len(good) >= required
            else None), int(len(good))


def score_pair(candidate, control, refs, draws, evaluator, targets):
    """One common component draw roster for all references and both readouts."""
    import numpy as np
    selected = [targets.index(t) for t in WEAK]
    non_syn = [i for i, t in enumerate(targets) if t != 'Synovitis']
    result, cached = {}, {}
    for name, (y, mask, minimum) in refs.items():
        ca = evaluator.auc_targets(y, candidate, mask, minimum)
        co = evaluator.auc_targets(y, control, mask, minimum)
        d = ca - co
        draw_macro, draw_weak, draw_11 = [], [], []
        for ix in draws:
            a = evaluator.auc_targets(y[ix], candidate[ix], mask[ix], minimum)
            b = evaluator.auc_targets(y[ix], control[ix], mask[ix], minimum)
            delta = a - b
            draw_macro.append(evaluator.macro(a) - evaluator.macro(b))
            draw_weak.append(float(np.mean(delta[selected])) if np.isfinite(delta[selected]).all()
                             else float('nan'))
            draw_11.append(evaluator.macro(a[non_syn]) - evaluator.macro(b[non_syn]))
        cached[name] = dict(macro=np.asarray(draw_macro), weak=np.asarray(draw_weak),
                            non_syn=np.asarray(draw_11))
        result[name] = dict(candidate_macro=_finite(evaluator.macro(ca)),
                            control_macro=_finite(evaluator.macro(co)),
                            delta=_finite(evaluator.macro(ca)-evaluator.macro(co)),
                            candidate_per_target={t: _finite(v) for t, v in zip(targets, ca)},
                            control_per_target={t: _finite(v) for t, v in zip(targets, co)},
                            per_target_delta={t: _finite(v) for t, v in zip(targets, d)},
                            macro_ci95=None, macro_valid_draws=0)
        result[name]['macro_ci95'], result[name]['macro_valid_draws'] = _interval(cached[name]['macro'])
    def primary(indices, key):
        if key == 'non_syn':
            # EXP182 registered the 11-target macro AUC, not a mean of 11
            # independently rounded per-target deltas.
            point = [evaluator.macro(np.asarray([result[n]['candidate_per_target'][targets[i]]
                                                 for i in indices], dtype=float)) -
                     evaluator.macro(np.asarray([result[n]['control_per_target'][targets[i]]
                                                 for i in indices], dtype=float))
                     for n in ('jev', 'consensus')]
        else:
            point = [np.mean(np.asarray([result[n]['per_target_delta'][targets[i]]
                                         for i in indices], dtype=float))
                     for n in ('jev', 'consensus')]
        draws_mean = (cached['jev'][key] + cached['consensus'][key]) / 2
        ci95, valid = _interval(draws_mean)
        return dict(delta=_finite(np.mean(point)), ci95=ci95,
                    valid_draws=valid, total_draws=len(draws_mean))
    return dict(references=result, weak_four=primary(selected, 'weak'),
                non_synovitis_11=primary(non_syn, 'non_syn'))


def decide(experiment, scored, targets):
    guard = harm_guard(scored['references'], targets)
    if experiment == 'exp184':
        primary = scored['weak_four']
        status = 'REVIEW_WEAK_AND_IMAGE_EVIDENCE' if (guard['status'] == 'PASS' and
                  primary['delta'] is not None and primary['ci95'] is not None) else 'HOLD_REVIEW'
        return dict(status=status, weak_target_primary=primary, harm_guard=guard,
                    aquila='report only; A50 is self-referenced',
                    synovitis_gold58='report only; no Gold58 model serving in this CPU screen',
                    synovitis_image_read='required separately before claiming a Synovitis gain')
    primary = scored['non_synovitis_11']
    old_gate = primary['delta'] is not None and primary['delta'] >= -.003
    return dict(status='INCONCLUSIVE_SYNOVITIS' if old_gate else 'HOLD_NON_HARM',
                registered_exp182_non_synovitis_11=primary,
                registered_non_harm_pass=old_gate,
                weak_target_diagnostic=scored['weak_four'], diagnostic_harm_guard=guard,
                synovitis_image_read='required separately; Gold58 report only')


def collect(experiment, source, root=ROOT):
    """Validate every remote arm before copying any output into the experiment."""
    tool = module(experiment)
    root, source = Path(root), Path(source)
    status_name = f'{experiment}-status.json'
    status = tool.read(source / status_name)
    tool.need(status['state'] == 'COMPLETE' and status['version'] == tool.VERSION and
              all(status['arms'][a]['state'] == 'COMPLETE' for a in tool.ARMS),
              'kernel did not complete every arm')
    destination = root / tool.EXP / 'outputs' / 'kaggle'
    tool.need(not (destination / status_name).exists() and
              all(not (destination / tool.tag(a)).exists() for a in tool.ARMS),
              'collected output already exists')
    for arm in tool.ARMS:
        hashes = tool.verify_run(source / tool.tag(arm), arm, root)
        tool.need(hashes == status['arms'][arm]['sha256'], 'status hashes differ: ' + arm)
    copied = {}
    for arm in tool.ARMS:
        dest = destination / tool.tag(arm)
        dest.mkdir(parents=True)
        for name in (*tool.REQUIRED_OUTPUTS, 'output-inventory.json'):
            shutil.copyfile(source / tool.tag(arm) / name, dest / name)
        copied[arm] = dest.as_posix()
    shutil.copyfile(source / status_name, destination / status_name)
    return copied


def screen(experiment, root=ROOT, destination=None):
    from tools import exp115_vlm_eval as ev
    from tools import exp156_k_sweep as e156
    from tools import exp169_bundle_screen as bundle
    tool = module(experiment)
    root = Path(root)
    destination = Path(destination) if destination else root / tool.EXP / 'screen'
    tool.need(not destination.exists(), 'fresh screen directory required')
    output = root / tool.EXP / 'outputs' / 'kaggle'
    status_name = f'{experiment}-status.json'
    status = tool.read(output / status_name)
    tool.need(status['state'] == 'COMPLETE' and all(status['arms'][a]['state'] == 'COMPLETE'
              for a in tool.ARMS), 'collect a complete kernel first')
    q = tool.read(root / tool.protocol_path(tool.ARMS[0]))
    ids = tool.read(root / q['folds']['0']['ordered_ids']['path'])['val']
    tool.need(len(ids) == len(set(ids)) == 888, 'fold-0 roster differs')
    refs_raw = e156.references(ids, root)
    refs = {name: refs_raw[key] for name, key in
            (('jev', 'jev_hybrid'), ('consensus', 'consensus'),
             ('dread', 'dread'), ('aquila', 'aquila_v1'))}
    draws = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    predictions, inputs = {}, {}
    for arm in tool.ARMS:
        run = output / tool.tag(arm)
        tool.verify_run(run, arm, root)
        predictions[arm] = {}
        for mode, name in (('plain', 'oof_k36.csv'), ('flip_tta', 'oof_k36_flip_tta.csv')):
            path = run / name
            predictions[arm][mode] = bundle.load_oof(path, ids)
            inputs[f'{arm}_{mode}'] = dict(path=path.relative_to(root).as_posix(),
                                          sha256=tool.sha(path))
    control = 'A25' if experiment == 'exp184' else 'CONTROL'
    candidates = ('A0', 'A50') if experiment == 'exp184' else ('FILL',)
    results = {}
    for arm in candidates:
        results[arm] = {}
        for mode in ('plain', 'flip_tta'):
            scored = score_pair(predictions[arm][mode], predictions[control][mode], refs,
                                draws, ev, tool.TARGETS)
            results[arm][mode] = dict(scored=scored, decision=decide(experiment, scored, tool.TARGETS))
    report = dict(schema=f'{experiment}-cpu-screen-v1', experiment=experiment,
                  protocol_sha256=tool.sha(root / tool.PROTOCOL),
                  status_sha256=tool.sha(output / status_name), control=control,
                  candidates=list(candidates), studies=len(ids), seed=42, fold=0,
                  bootstrap=dict(unit='validation component', draws=300, seed=20261003,
                                 shared_across='arms, references and modes'),
                  limitations=['one exposed weak OOF fold; no private-score inference',
                               'report-derived rulers share ancestry and target definitions'],
                  input_files=inputs, results=results)
    destination.mkdir(parents=True)
    tool.write_json(destination / 'result.json', report)
    return dict(status='SCREENED', result=str(destination / 'result.json'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('collect', 'screen'))
    ap.add_argument('--experiment', required=True, choices=('exp182', 'exp184'))
    ap.add_argument('--source', type=Path, help='Kaggle output download directory for collect')
    ap.add_argument('--destination', type=Path, help='fresh screen output directory')
    a = ap.parse_args(argv)
    if a.command == 'collect':
        if a.source is None:
            ap.error('collect requires --source')
        result = collect(a.experiment, a.source)
    else:
        result = screen(a.experiment, destination=a.destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
