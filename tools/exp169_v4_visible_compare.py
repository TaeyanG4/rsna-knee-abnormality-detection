"""Read-only A4/B4 visible-run parity audit; local references use exp169_v4_visible_refs.

python -m tools.exp169_v4_visible_compare --candidate A4 --out report.json

The three-study visible cohort is too small for a meaningful Spearman gate
(the registered array comparator also requires ten rows). It is reported as
null; the frozen max/mean absolute bounds remain mandatory for every arm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tools import exp169_release_v4 as v4
from tools import exp169_v2_local_refs as local
from tools.parity_compare_arrays import compare as compare_array

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/releases/EXP169_V4_A4_B4_s43_stage_v411'
REFS = ROOT / 'artifacts/experiments/EXP169_RELEASE_V2/local-refs/v4-visible'
TOL_SHA = '9b3377bcb62d0235bea265e7b318d15e1f099b2652f2ebd115284ebaa9175752'
TARGETS = local.TARGETS
UID = local.UID


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_values(path: Path, ids: list[str]) -> np.ndarray:
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    if header != [UID, *TARGETS]:
        raise ValueError(f'{path}: target names/order differ')
    if [r[0] for r in rows] != ids or any(len(r) != 13 for r in rows):
        raise ValueError(f'{path}: ID/order/width differ')
    values = np.asarray([[float(x) for x in r[1:]] for r in rows], dtype=np.float64)
    if values.shape != (len(ids), 12) or not np.isfinite(values).all():
        raise ValueError(f'{path}: invalid CSV probabilities')
    return values


def csv_cells(path: Path, ids: list[str]) -> list[list[str]]:
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    if rows[0] != [UID, *TARGETS] or [r[0] for r in rows[1:]] != ids or any(len(r) != 13 for r in rows[1:]):
        raise ValueError(f'{path}: submission header/IDs/width differ')
    return [r[1:] for r in rows[1:]]


def npz(path: Path, keys: set[str], ids: list[str] | None = None,
        targets: bool = False) -> dict:
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != keys:
            raise ValueError(f'{path}: keys differ: {source.files}')
        out = {key: source[key].copy() for key in source.files}
    got = list(map(str, out['ids'].tolist()))
    if out['ids'].ndim != 1 or len(got) != 3 or len(set(got)) != 3:
        raise ValueError(f'{path}: expected three unique ordered IDs')
    if ids is not None and got != ids:
        raise ValueError(f'{path}: IDs/order differ')
    if targets and (out['targets'].ndim != 1 or out['targets'].tolist() != TARGETS):
        raise ValueError(f'{path}: targets/order differ')
    for key, value in out.items():
        if key in ('ids', 'targets'):
            continue
        shape = (10 if key == 'l2' else 5, 3, 12) if key in ('l2', 'dino', 'convnext') else (3, 12)
        if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
            raise ValueError(f'{path}: {key} shape/probabilities invalid')
    return out


def anchor(folder: Path, keys: set[str], ids: list[str], sample: Path) -> dict:
    receipt = read_json(folder / 'receipt.json')
    if receipt.get('status') != 'COMPLETE_INFERENCE' or set(receipt.get('files', {})) != {
            'raw.npz', 'inputs.json', 'runtime.json', 'binding.json', 'submission.csv'}:
        raise ValueError(f'{folder}: anchor receipt incomplete')
    for name in ('raw.npz', 'inputs.json', 'runtime.json', 'binding.json', 'submission.csv'):
        if sha(folder / name) != receipt['files'][name]:
            raise ValueError(f'{folder}: {name} receipt hash differs')
    raw = npz(folder / 'raw.npz', {'ids', 'targets', *keys}, ids, True)
    csv_values(folder / 'submission.csv', ids)
    inputs = read_json(folder / 'inputs.json')
    if set(inputs) != set(ids):
        raise ValueError(f'{folder}: input record roster differs')
    return raw


def metric(reference: np.ndarray, visible: np.ndarray, tolerance: dict) -> dict:
    result = compare_array(reference, visible, tolerance)
    if result.get('pass') is not True:
        result['pass'] = False
    return result


def own_anchor_csv_values(raw: np.ndarray) -> np.ndarray:
    """Reproduce frozen public-serving.py rankpct and its float32 CSV input."""
    ranks = raw.argsort(0).argsort(0).astype(np.float64)
    ranks /= max(1, raw.shape[0] - 1)
    return ranks.astype(np.float32).astype(np.float64)


def s42_folder(candidate: str) -> str:
    """B4 kept the original B172 trio folder name for its own-only S42 arm."""
    if candidate == 'A4':
        return 'exp169-own42'
    if candidate == 'B4':
        return 'exp169-trio'
    raise ValueError('candidate must be A4 or B4')


def local_reference(name: str, folder: Path, ids: list[str], input_sha256: str,
                    stage: Path = STAGE, root: Path = ROOT) -> np.ndarray:
    proof = folder / 'local-receipt.json'
    if not proof.is_file():
        raise FileNotFoundError(proof)
    evidence = read_json(proof)
    reference_path = folder / 'raw.npz'
    stage, root = Path(stage), Path(root)
    own_input = read_json(stage / ('visible-a4' if name == 'S43' else 'visible-b4') /
                          s42_folder('A4' if name == 'S43' else 'B4') / 'inputs.json')
    expected = (dict(stage_manifest_sha256=sha(stage / 's43/assets/release-manifest.json'),
                     checkpoint_sha256=read_json(stage / 's43/assets/release-manifest.json')['checkpoint_sha256'])
                if name == 'S43' else
                dict(b172_manifest_sha256=sha(root / v4.B172 / 'assets/release-manifest.json'),
                     c4_manifest_sha256=sha(stage / 'c4/release-manifest.json'),
                     c4_archive_sha256=sha(stage / 'c4/dataset' / v4.C4_ARCHIVE)))
    if (evidence.get('schema') != 'exp169-v4-local-visible-reference-v1' or
            evidence.get('status') != 'COMPLETE_LOCAL_REFERENCE' or
            evidence.get('source') != 'independent-serving-run' or
            evidence.get('candidate_arm') != name or
            evidence.get('device') != ('cpu-fp32' if name == 'S43' else 'local-cuda-fp16') or
            evidence.get('raw_sha256') != sha(reference_path) or
            evidence.get('visible_input_sha256') != input_sha256 or
            evidence.get('input_records_sha256') != local.text_sha(json.dumps(own_input, sort_keys=True)) or
            evidence.get('tool') != 'tools/exp169_v4_visible_refs.py' or
            evidence.get('tool_sha256') != sha(root / 'tools/exp169_v4_visible_refs.py') or
            evidence.get('ids') != ids or
            any(evidence.get(key) != value for key, value in expected.items())):
        raise ValueError('local reference receipt/provenance differs')
    if name == 'C4':
        source_code = read_json(root / v4.B172 / 'assets/release-manifest.json')['source_code']
        loaded = evidence.get('staged_modules', {})
        expected_modules = {'exp169_partner_bundle_release', 'exp169_release_v2', *local.RUNTIME_MODULES}
        if set(loaded) != expected_modules or any(
                member.get('sha256') != source_code.get(module + '.py') for module, member in loaded.items()):
            raise ValueError('C4 staged runtime receipt differs')
    reference = npz(reference_path,
                    {'ids', 'targets', 'own'} if name == 'S43' else {'ids', 'l2', 'dino', 'convnext'},
                    ids, name == 'S43')
    return reference['own' if name == 'S43' else 'convnext']


def new_report(candidate: str, stage: Path) -> dict:
    return dict(schema='exp169-v4-visible-parity-v1', candidate=candidate,
                status='HOLD', visible=str(Path(stage) / ('visible-' + candidate.lower())),
                comparisons={}, failed_comparisons=[], checks={}, failed_checks=[],
                files={}, missing=[], errors=[], timing={},
                spearman_note='Three visible studies per target: rank correlation is not gated by the registered comparator; max/mean absolute gates are enforced.')


def record_file(report: dict, name: str, path: Path) -> None:
    path = Path(path)
    if path.is_file():
        report['files'][name] = dict(path=str(path), sha256=sha(path))
    elif str(path) not in report['missing']:
        report['missing'].append(str(path))


def finish_report(report: dict) -> dict:
    report['failed_comparisons'] = [name for name, result in report['comparisons'].items()
                                    if result.get('pass') is not True]
    report['failed_checks'] = [name for name, passed in report['checks'].items() if passed is False]
    report['status'] = ('PASS_VISIBLE_PARITY' if not report['errors'] and not report['missing'] and
                        not report['failed_comparisons'] and not report['failed_checks'] else 'HOLD')
    return report


def _compare(candidate: str, stage: Path = STAGE, refs: Path = REFS,
            root: Path = ROOT, report: dict | None = None) -> dict:
    if candidate not in ('A4', 'B4'):
        raise ValueError('candidate must be A4 or B4')
    stage, refs, root = map(Path, (stage, refs, root))
    report = report if report is not None else new_report(candidate, stage)
    record_file(report, 'parity_plan', stage / candidate / 'parity-plan.json')
    plan = read_json(stage / candidate / 'parity-plan.json')
    tol_path = root / plan['partner']['cross_gpu_tolerance_file']
    record_file(report, 'tolerance', tol_path)
    if (plan.get('candidate') != candidate or plan.get('status') != 'FROZEN_NOT_RUN' or
            plan['partner']['cross_gpu_tolerance_sha256'] != TOL_SHA or
            sha(tol_path) != TOL_SHA):
        raise ValueError('frozen parity plan/tolerance binding differs')
    tol = read_json(tol_path)['default']
    if tol != plan['partner']['cross_gpu_tolerance'] or tol != dict(max_abs=.002, mean_abs=.0005, min_spearman=.99):
        raise ValueError('frozen tolerance values differ')
    visible = stage / ('visible-' + candidate.lower())
    report.update(plan_sha256=sha(stage / candidate / 'parity-plan.json'),
                  tolerance_sha256=TOL_SHA, tolerance=tol)

    def check(name, fn):
        try:
            report['checks'][name] = fn()
            if report['checks'][name] is False:
                report['errors'].append(name)
        except (OSError, KeyError, ValueError, IndexError, TypeError) as exc:
            report['checks'][name] = False
            report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')

    receipt_path = visible / v4.RECEIPT
    record_file(report, 'visible_receipt', receipt_path)
    if not receipt_path.is_file():
        return finish_report(report)
    receipt = read_json(receipt_path)
    report['timing'] = dict(wall_clock_seconds=receipt.get('timing_policy', {}).get('wall_clock_seconds'),
                            accounted_elapsed_seconds=receipt.get('elapsed_seconds'),
                            stages={k: v.get('seconds') for k, v in receipt.get('stages', {}).items()},
                            anchors={k: v.get('seconds') for k, v in receipt.get('anchors', {}).items()})
    check('pass_bundle_receipt', lambda: receipt.get('status') == 'PASS_BUNDLE' and
          receipt.get('candidate') == candidate and receipt.get('studies') == 3 and
          receipt.get('qualification') == 'VISIBLE_ONLY' and
          receipt.get('executed_arms') ==
          (['public', 'OWN_S42_S43', 'EXP155'] if candidate == 'A4' else
           ['OWN_S42_S43', 'EXP174']) +
          ['L2_RADR50_10', 'EXP172_L2_DINO_BASE_5', 'EXP179_183_L2_CONVNEXT_T_5'])
    check('release_manifest_receipt', lambda: receipt.get('manifest_sha256') ==
          sha(stage / 'code/assets/release-manifest.json'))
    sample = root / 'storage/competition/sample_submission.csv'
    with sample.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    if rows[0] != [UID, *TARGETS] or len(rows) != 4:
        raise ValueError('visible sample roster/targets differ')
    ids = [r[0] for r in rows[1:]]
    own42 = s42_folder(candidate)
    for label, folder in [('S42', own42), ('S43', 'exp169-s43-own'),
                          ('extra', 'exp169-trio' if candidate == 'A4' else 'exp169-anchor174')]:
        record_file(report, label + '_raw', visible / folder / 'raw.npz')
        record_file(report, label + '_receipt', visible / folder / 'receipt.json')
    for label, path in [('visible_inputs', visible / own42 / 'inputs.json'),
                        ('partner_raw', visible / 'partner-raw.npz'),
                        ('pilot_raw', visible / 'exp169-v4-pilot/pilot-scores.npz'),
                        ('submission', visible / 'submission.csv'),
                        ('B172_own_raw', root / v4.B172 / 'kernel-output/exp169-trio/raw.npz'),
                        ('B172_partner_raw', root / v4.B172 / 'kernel-output/partner-raw.npz')]:
        record_file(report, label, path)
    if candidate == 'A4':
        record_file(report, 'A172_raw', root / v4.A172 / 'kernel-output/exp169-trio/raw.npz')
    else:
        record_file(report, 'B174_raw', root / v4.B174 / 'kernel-output/exp169-anchor174/raw.npz')
        record_file(report, 'EXP174_cpu_fp32_raw', root / 'artifacts/experiments/EXP169_RELEASE_V2/local-refs/b174-anchor174/raw.npz')
    raw42 = anchor(visible / own42, {'own'}, ids, sample)
    raw43 = anchor(visible / 'exp169-s43-own', {'own'}, ids, sample)
    for arm, folder, raw in [('S42', own42, raw42), ('S43', 'exp169-s43-own', raw43)]:
        check(arm + '_anchor_csv_rank', lambda f=folder, r=raw: bool(np.array_equal(
            csv_values(visible / f / 'submission.csv', ids),
            own_anchor_csv_values(r['own']))))
    s43manifest = read_json(stage / 's43/assets/release-manifest.json')
    binding43 = read_json(visible / 'exp169-s43-own/binding.json')
    check('S43_binding', lambda: binding43['variant'] == 'own' and
          binding43['own_sha256'] == s43manifest['checkpoint_sha256'] and
          binding43['serving_sha256'] == s43manifest['serving_sha256'] and
          binding43['source_sha256'] == s43manifest['source_sha256'] and
          binding43['public_sha256'] is None)
    check('anchor_sha256', lambda: all(sha(visible / folder / 'raw.npz') == receipt['anchors'][name]['raw_sha256']
                                      for name, folder in [('S42', own42), ('S43', 'exp169-s43-own')]))
    if candidate == 'A4':
        extra = anchor(visible / 'exp169-trio', {'own', 'public', 'exp143'}, ids, sample)
        check('extra_anchor_sha256', lambda: sha(visible / 'exp169-trio/raw.npz') == receipt['anchors']['public_EXP155']['raw_sha256'])
        check('public_EXP155_anchor_csv_exact', lambda: sha(visible / 'exp169-trio/submission.csv') ==
              sha(root / v4.A172 / 'kernel-output/exp169-trio/submission.csv'))
    else:
        extra = anchor(visible / 'exp169-anchor174', {'own'}, ids, sample)
        check('extra_anchor_sha256', lambda: sha(visible / 'exp169-anchor174/raw.npz') == receipt['anchors']['EXP174']['raw_sha256'])
        check('EXP174_anchor_csv_exact', lambda: sha(visible / 'exp169-anchor174/submission.csv') ==
              sha(root / v4.B174 / 'kernel-output/exp169-anchor174/submission.csv'))
    check('all_anchor_inputs_equal', lambda: all(read_json(visible / folder / 'inputs.json') ==
          read_json(visible / own42 / 'inputs.json') for folder in
          ('exp169-s43-own', 'exp169-trio' if candidate == 'A4' else 'exp169-anchor174')))
    check('visible_inputs_match_B172', lambda: read_json(visible / own42 / 'inputs.json') ==
          read_json(root / v4.B172 / 'kernel-output/exp169-trio/inputs.json'))
    partner = npz(visible / 'partner-raw.npz', {'ids', 'l2', 'dino', 'convnext'}, ids)
    pilot = npz(visible / 'exp169-v4-pilot/pilot-scores.npz', {'ids', 'l2', 'dino', 'convnext'}, ids)
    check('partner_manifest_c4', lambda: read_json(visible / 'exp169-partner/release-manifest.json')
          ['c4_manifest_sha256'] == sha(stage / 'c4/release-manifest.json'))
    check('pilot_full_byte_hash', lambda: sha(visible / 'partner-raw.npz') ==
          sha(visible / 'exp169-v4-pilot/pilot-scores.npz') == receipt['partner_raw_sha256'] == receipt['pilot']['scores_sha256'])
    for family in ('l2', 'dino', 'convnext'):
        check('pilot_full_' + family, lambda f=family: bool(np.array_equal(pilot[f], partner[f])))
    b172own = npz(root / v4.B172 / 'kernel-output/exp169-trio/raw.npz',
                  {'ids', 'targets', 'own'}, ids, True)
    b172partner = npz(root / v4.B172 / 'kernel-output/partner-raw.npz',
                      {'ids', 'l2', 'dino', 'convnext'}, ids)
    comparisons = {'S42': (b172own['own'], raw42['own']),
                   'R50_s42': (b172partner['l2'][:5], partner['l2'][:5]),
                   'R50_s43': (b172partner['l2'][5:], partner['l2'][5:]),
                   'DINO': (b172partner['dino'], partner['dino'])}
    if candidate == 'A4':
        a172 = npz(root / v4.A172 / 'kernel-output/exp169-trio/raw.npz',
                   {'ids', 'targets', 'own', 'public', 'exp143'}, ids, True)
        comparisons.update(public=(a172['public'], extra['public']),
                           EXP155=(a172['own'], extra['own']))
    else:
        b174 = npz(root / v4.B174 / 'kernel-output/exp169-anchor174/raw.npz',
                   {'ids', 'targets', 'own'}, ids, True)
        comparisons['EXP174'] = (b174['own'], extra['own'])
        cpu174 = root / 'artifacts/experiments/EXP169_RELEASE_V2/local-refs/b174-anchor174/raw.npz'
        comparisons['EXP174_cpu_fp32'] = (npz(cpu174, {'ids', 'targets', 'own'}, ids, True)['own'], extra['own'])
    for name, (reference, actual) in comparisons.items():
        report['comparisons'][name] = metric(reference, actual, tol)
    for name, reference, actual in [('R50_s42', b172partner['l2'][:5], partner['l2'][:5]),
                                    ('R50_s43', b172partner['l2'][5:], partner['l2'][5:]),
                                    ('DINO', b172partner['dino'], partner['dino'])]:
        for fold in range(5):
            report['comparisons'][f'{name}_f{fold}'] = metric(reference[fold], actual[fold], tol)

    # Independent new references must carry a local receipt and matching input
    # records. A raw copy of the visible array alone cannot pass this gate.
    s43folder = refs / 's43-own'
    c4folder = refs / 'c4-partner'
    for name, folder, key, actual in [('S43', s43folder, 'own', raw43['own']),
                                      ('C4', c4folder, 'convnext', partner['convnext'])]:
        record_file(report, name + '_local_raw', folder / 'raw.npz')
        record_file(report, name + '_local_receipt', folder / 'local-receipt.json')
        try:
            reference = local_reference(name, folder, ids, sha(visible / own42 / 'inputs.json'), stage, root)
            report['comparisons'][name] = metric(reference, actual, tol)
            if name == 'C4':
                for fold in range(5):
                    report['comparisons'][f'C4_f{fold}'] = metric(reference[fold], actual[fold], tol)
            report['checks'][name + '_independent_reference'] = True
        except FileNotFoundError as exc:
            missing = exc.filename or str(exc)
            if missing not in report['missing']:
                report['missing'].append(missing)
            report['checks'][name + '_independent_reference'] = False
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report['errors'].append(f'{name} reference: {type(exc).__name__}: {exc}')
            report['checks'][name + '_independent_reference'] = False

    q4 = v4.q4_values({f: partner[f] for f in ('l2', 'dino', 'convnext')})
    kwargs = dict(public=extra['public'], exp155=extra['own']) if candidate == 'A4' else dict(exp174=extra['own'])
    expected = v4.final_values(candidate, raw42['own'], raw43['own'], q4, **kwargs)
    exact_cells = [[format(float(x), '.10g') for x in row] for row in expected]
    exact = np.asarray([[float(x) for x in row] for row in exact_cells])
    written = csv_values(visible / 'submission.csv', ids)
    check('final_csv_exact_by_id', lambda: csv_cells(visible / 'submission.csv', ids) == exact_cells)
    report['final_csv'] = dict(max_abs=float(np.max(np.abs(exact - written))),
                               unequal_cells=int(np.count_nonzero(exact != written)),
                               sha256=sha(visible / 'submission.csv'))
    check('final_csv_receipt_hash', lambda: report['final_csv']['sha256'] == receipt['submission_sha256'])
    return finish_report(report)


def compare(candidate: str, stage: Path = STAGE, refs: Path = REFS,
            root: Path = ROOT) -> dict:
    """Always return a written HOLD on missing/drifted visible inputs."""
    report = new_report(candidate, stage)
    try:
        return _compare(candidate, stage, refs, root, report)
    except (OSError, KeyError, ValueError, IndexError, TypeError, RuntimeError) as exc:
        if isinstance(exc, FileNotFoundError):
            missing = exc.filename or str(exc)
            if missing not in report['missing']:
                report['missing'].append(missing)
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        return finish_report(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('A4', 'B4', 'both'), default='both')
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--refs', type=Path, default=REFS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = {c: compare(c, args.stage, args.refs) for c in ('A4', 'B4')} if args.candidate == 'both' else compare(args.candidate, args.stage, args.refs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({c: r['status'] for c, r in result.items()} if args.candidate == 'both' else result['status']))
    return 0 if (all(r['status'] == 'PASS_VISIBLE_PARITY' for r in result.values()) if args.candidate == 'both' else result['status'] == 'PASS_VISIBLE_PARITY') else 1


if __name__ == '__main__':
    raise SystemExit(main())
