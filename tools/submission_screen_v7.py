"""CPU-only v7 receipt adapter; delegates all registry/admission operations to the frozen screen.
Same CLI as submission_screen.py. release.kernel_dir/release.manifest must supply the
actual release-manifest.json; copied receipts without arm CSVs or a manifest HOLD.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location('screen_v7_base', Path(__file__).with_name('submission_screen.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
_original_results = base.public3_results
_original_engineering = base.engineering_section


def nested_results(ctx, out_dir):
    final, arms, results = _original_results(ctx, out_dir)
    shared = [(p, d) for p, d in results if isinstance(d.get('arms'), dict) and 'shared' in p.parts]
    if not shared:
        return final, arms, results
    if len(shared) != 1:
        return None, {}, results
    p, d = shared[0]
    arms = {name: (p, dict(rec, arm=name, status=d.get('status'),
                          studies=d.get('studies'), release_manifest_sha256=d.get('release_manifest_sha256')))
            for name, rec in d['arms'].items() if isinstance(rec, dict)}
    if final is None and len(arms) == 1:
        final = (p, dict(d, **next(iter(d['arms'].values()))))
    return final, arms, results


def validate_nested(ctx, out_dir, release):
    """Check original nested receipts directly. No detached legacy receipts are created."""
    final, arms, results = nested_results(ctx, out_dir)
    shared = [(p, d) for p, d in results if isinstance(d.get('arms'), dict) and 'shared' in p.parts]
    if not shared:
        mp = ctx.path(release.get('manifest'))
        if mp is None and release.get('kernel_dir'):
            mp = ctx.path(release['kernel_dir']) / 'release-manifest.json'
        if mp is not None and not mp.is_file():
            return ['release manifest unavailable'], []
        if mp is not None and mp.is_file():
            try:
                if base.read_json(mp).get('schema') == 'exp119-test-release-v7':
                    return ['v7 shared nested receipt unavailable'], []
            except (OSError, ValueError, TypeError):
                return ['release manifest unreadable'], []
        return [], []
    missing, conflicting = [], []
    if len(shared) != 1 or final is None:
        return ['ambiguous shared/final receipt'], []
    sp, sd = shared[0]
    fp, fd = final
    mp = ctx.path(release.get('manifest'))
    if mp is None and release.get('kernel_dir'):
        mp = ctx.path(release['kernel_dir']) / 'release-manifest.json'
    if mp is None or not mp.is_file():
        return ['release manifest unavailable: every nested arm requires manifest binding'], []
    try:
        manifest = base.read_json(mp)
        ma = manifest['arms']
        if not isinstance(ma, list) or not ma or any(not isinstance(a, dict) for a in ma):
            raise ValueError('invalid arm list')
        names = [a['name'] for a in ma]
        expected_sha = ctx.sha(mp)
        weights = {a['name']: float(a['weight']) for a in ma}
        total = sum(weights.values())
        if len(set(names)) != len(names) or not names or total <= 0 or any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError('invalid arm names/weights')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return ['unreadable/invalid release manifest: ' + str(exc)], []
    for label, receipt in [('shared', sd), ('final', fd)]:
        for key, expected in [('release_manifest_sha256', expected_sha), ('schema', manifest.get('schema')),
                              ('release', manifest.get('release')), ('arm_order', names)]:
            if receipt.get(key) is None:
                missing.append(label + ': missing ' + key)
            elif receipt[key] != expected:
                conflicting.append(label + ': ' + key + ' differs from manifest')
        if not isinstance(receipt.get('arms'), dict):
            missing.append(label + ': missing arms')
        elif set(receipt['arms']) != set(names):
            conflicting.append(label + ': arm list differs from manifest')
        rw = receipt.get('rank_weights')
        if not isinstance(rw, dict):
            missing.append(label + ': missing rank_weights')
        elif set(rw) != set(names) or any(not isinstance(rw[n], (float, int)) or not math.isfinite(rw[n]) or abs(rw[n] - weights[n]/total) > 1e-9 for n in names):
            conflicting.append(label + ': weights differ from manifest')
    if sd.get('studies') is None or fd.get('studies') is None:
        missing.append('shared/final study count unavailable')
    elif sd['studies'] != fd['studies']:
        conflicting.append('shared/final study counts differ')
    if fp != sp:
        if sd.get('decode') is None or fd.get('shared_decode') is None:
            missing.append('shared decode binding unavailable')
        elif sd['decode'] != fd['shared_decode']:
            conflicting.append('shared/final decode receipts differ')
    if fd.get('status') == 'PASS_RANK_MEAN_TEST_COHORT_ONLY':
        if fd.get('formula') is None:
            missing.append('final rank formula unavailable')
        elif fd['formula'] != manifest.get('combination'):
            conflicting.append('final rank formula differs from manifest')
        if not isinstance(fd.get('arm_weights'), dict):
            missing.append('final arm weights unavailable')
        elif fd['arm_weights'] != weights:
            conflicting.append('final arm weights differ from manifest')
    shared_arms = sd.get('arms') if isinstance(sd.get('arms'), dict) else {}
    final_arms = fd.get('arms') if isinstance(fd.get('arms'), dict) else {}
    for arm in ma:
        name = arm['name']
        sr, fr = shared_arms.get(name, {}), final_arms.get(name, {})
        expected = arm.get('checkpoints')
        if not isinstance(sr, dict) or not isinstance(fr, dict):
            missing.append(name + ': missing nested arm receipt')
            continue
        checkpoints = sr.get('checkpoints')
        if (not isinstance(checkpoints, list) or not checkpoints or not isinstance(expected, list) or not expected
                or any(not isinstance(c, dict) or not c.get('sha256') or not c.get('name') or not c.get('path')
                       for c in checkpoints + expected)):
            missing.append(name + ': missing shared checkpoint binding')
        elif ([(c['name'], c['path'], c['sha256']) for c in checkpoints]
              != [(c['name'], c['path'], c['sha256']) for c in expected]
              or any(c.get('k_eval_release') != arm.get('k_eval') for c in checkpoints)):
            conflicting.append(name + ': shared checkpoints differ from manifest')
        for key, value in [('k_eval', arm.get('k_eval')), ('n_models', len(expected or [])), ('rank_weight', weights[name])]:
            if sr.get(key) is None or fr.get(key) is None:
                missing.append(name + ': missing ' + key)
            elif sr[key] != value or fr[key] != value:
                conflicting.append(name + ': ' + key + ' differs from manifest')
        if sr.get('rank_weight_normalised') is None:
            missing.append(name + ': missing normalised rank weight')
        elif (not isinstance(sr['rank_weight_normalised'], (int, float))
              or not math.isfinite(sr['rank_weight_normalised'])
              or abs(sr['rank_weight_normalised'] - weights[name]/total) > 1e-9):
            conflicting.append(name + ': normalised rank weight differs from manifest')
        raw = sp.parents[2] / name / 'arm.csv'  # shared/work/result.json -> release root
        if not raw.is_file():
            missing.append(name + ': arm.csv unavailable')
        else:
            digest = ctx.sha(raw)
            if not sr.get('submission_sha256') or not fr.get('submission_sha256'):
                missing.append(name + ': missing arm CSV binding')
            elif digest != sr['submission_sha256'] or digest != fr['submission_sha256']:
                conflicting.append(name + ': arm CSV hash differs')
            sample = ctx.path(ctx.cfg['sample_submission'])
            if sample is None or not sample.is_file():
                missing.append(name + ': CSV cohort validator unavailable')
            else:
                ok, n, err = base.RP.validate_csv(raw, sample)
                if not ok or n != sd.get('studies'):
                    conflicting.append(name + ': arm CSV invalid: ' + str(err))
    return missing, conflicting


def engineering_section(ctx, members, release):
    release = release or {}
    out = _original_engineering(ctx, members, release)
    directory = ctx.path(release.get('out_dir'))
    if directory is None or not directory.is_dir():
        return out
    missing, conflicting = validate_nested(ctx, directory, release)
    final, nested, results = nested_results(ctx, directory)
    if any('shared' in p.parts and isinstance(d.get('arms'), dict) for p, d in results):
        if set(nested) != {m['arm'] for m in members}:
            conflicting.append('served arm list differs from screened members')
        for member in members:
            if member.get('k_eval') is None:
                missing.append(member['arm'] + ': screened K binding unavailable')
    out['v7_adapter'] = dict(missing_bindings=missing, conflicting_bindings=conflicting)
    out['problems'].extend(missing + conflicting)
    out.setdefault('hard_failures', []).extend(conflicting)
    if conflicting:
        out['engineering_pass'] = False
    if missing or conflicting:
        out['checkpoint_fingerprint'] = None
    return out


base.public3_results = nested_results
base.engineering_section = engineering_section
# Export reusable CPU APIs; main retains the original fingerprint registry semantics.
Ctx = base.Ctx
main = base.main




def exp126_engineering(ctx, members, release):
    """Stage 2: validate original receipt, own manifest, shared CSVs and anchor gate."""
    outdir = ctx.path(release['out_dir'])
    receipt_path = outdir / 'exp126/receipt.json'
    missing, hard, arm_reports, fp_items = [], [], {}, []
    out = dict(scope=base.ENGINEERING_SCOPE, scored_prediction_or_score_recovered=False,
               out_dir=ctx.rel(outdir), final_result=ctx.rel(receipt_path), status='PASS_TEST_COHORT_ONLY')
    def check(ok, message):
        if not ok: hard.append(message)
    try:
        r = base.read_json(receipt_path)
        shared_path = outdir / 'exp126/shared/receipt.json'
        if not shared_path.is_file():
            raise FileNotFoundError('shared receipt unavailable')
        shared = base.read_json(shared_path)
        mp = ctx.path(release.get('manifest'))
        if mp is None and release.get('kernel_dir'):
            mp = ctx.path(release['kernel_dir']) / 'exp126-own-manifest.json'
        if mp is None or not mp.is_file():
            raise FileNotFoundError('EXP126 own manifest unavailable')
        manifest = base.read_json(mp)
        digest = ctx.sha(mp)
        check(manifest.get('schema') == 'exp119-test-release-v7', 'own manifest schema differs')
        check(r.get('schema') == 'exp126-anchor-blend-v8', 'EXP126 parent receipt schema differs')
        check(shared.get('schema') == 'exp126-anchor-blend-v8-worker', 'EXP126 worker receipt schema differs')
        check(r.get('path') == 'blended', 'anchor-only fallback: candidate blend was not served')
        check(r.get('final_weights') == [0.7, 0.3], 'registered 0.7/0.3 weights differ')
        anchor = r.get('anchor', {})
        check(anchor.get('ok') is True and anchor.get('members') == ['resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3']
              and anchor.get('audit_members') == 4 and anchor.get('audit_status') == 'COMPLETE'
              and anchor.get('family_reduction') == 'rank_of_member_probability_mean'
              and not anchor.get('failure_events'), 'four-anchor-child hard gate failed')
        hook = r.get('hook') or {}
        check(hook.get('thread_finished') is True and not hook.get('hook_errors')
              and not hook.get('anchor_raised') and not hook.get('dropped_ids'),
              'shared hook failed or dropped studies')
        anchor_csv = outdir / 'exp126/anchor_submission.csv'
        if not anchor_csv.is_file() or not r.get('anchor_sha256'):
            missing.append('anchor CSV/hash binding unavailable')
        else:
            check(ctx.sha(anchor_csv) == r['anchor_sha256'], 'anchor backup hash differs')
        check(r.get('release_manifest_sha256') == digest == shared.get('release_manifest_sha256'), 'own manifest identity differs')
        check(r.get('worker') == shared, 'embedded worker receipt differs from shared receipt')
        config_path = (ctx.path(release.get('kernel_dir')) / 'exp126-config.json') if release.get('kernel_dir') else None
        if config_path is None or not config_path.is_file():
            missing.append('EXP126 build config unavailable')
            config = {}
        else:
            config = base.read_json(config_path)
            check(config.get('manifest_sha256') == digest and config.get('final_weights') == [0.7, 0.3]
                  and config.get('own') in ('rad5', 'coat5rad5') and
                  config.get('own') == ('rad5' if len(manifest.get('arms', [])) == 1 else 'coat5rad5'),
                  'EXP126 build config differs')
            check(shared.get('plane_flip_tta') is config.get('plane_flip_tta'), 'served TTA policy differs from build config')
        rad = next((a for a in manifest['arms'] if a.get('name') == 'rad5'), {})
        check(config.get('rad_k') in (24, 36) and rad.get('k_eval') == config.get('rad_k'),
              'EXP126 RadR50 K differs from build config')
        check(shared.get('status') == 'PASS' and shared.get('store', {}).get('trusted') is True, 'own worker/store failed')
        sample = ctx.path(ctx.cfg['sample_submission'])
        ids = list(base.pd.read_csv(sample, dtype={'StudyInstanceUID': str})['StudyInstanceUID'])
        check(shared.get('ids_sha256') == base.uid_hash(ids) and shared.get('studies') == len(ids), 'own cohort binding differs')
        scored, unscored = shared.get('scored_ids', []), shared.get('unscored', {})
        check(len(scored) == len(set(scored)) and not set(scored) & set(unscored)
              and set(scored) | set(unscored) == set(ids) and len(unscored) <= math.floor(.05*len(ids)), 'own study coverage failed')
        names = [a['name'] for a in manifest['arms']]
        check(set(shared.get('arms', {})) == set(names) == {m['arm'] for m in members}, 'own arm set differs')
        weights = manifest.get('rank_weights')
        check(isinstance(weights, dict) and set(weights) == set(names) and
              all(weights[n] == 1/len(names) for n in names), 'own arms must use equal rank mean')
        check(r.get('own_weights') == weights, 'served own rank weights differ')
        for arm in manifest['arms']:
            name = arm['name']; item = shared.get('arms', {}).get(name, {})
            checks = item.get('checkpoints', [])
            expected = arm.get('checkpoints', [])
            if not checks or not expected or any(not c.get('sha256') for c in checks + expected):
                missing.append(name + ': checkpoint binding unavailable')
            else:
                check(len(checks) == len(expected) and all(c.get('name') == e.get('name') and c.get('sha256') == e.get('sha256')
                      and c.get('k_eval_release') == arm.get('k_eval') for c, e in zip(checks, expected)), name + ': checkpoint/K binding differs')
            check(item.get('arm') == arm, name + ': embedded arm manifest differs')
            csvpath = outdir / 'exp126/shared' / (name + '.csv')
            if not csvpath.is_file() or not item.get('csv_sha256'):
                missing.append(name + ': CSV binding unavailable')
            else:
                check(ctx.sha(csvpath) == item['csv_sha256'], name + ': CSV hash differs')
                frame = base.load_pred(csvpath)
                check(list(frame.columns) == list(base.TARGETS), name + ': CSV target order differs')
                check(list(frame.index) == scored, name + ': CSV scored cohort differs')
                check(((frame.to_numpy() >= 0) & (frame.to_numpy() <= 1)).all(), name + ': probabilities outside [0,1]')
            member = next((m for m in members if m['arm'] == name), {})
            trained = set(member.get('declared_checkpoints') or []) if member.get('oof_declared_null') else {x.get('checkpoint_sha256') for x in member.get('runs', {}).values() if x.get('checkpoint_sha256')}
            served = {c.get('sha256') for c in checks if c.get('sha256')}
            binding = 'unknown'
            if not trained: missing.append(name + ': trained checkpoint binding unavailable')
            else:
                check(served == trained, name + ': served checkpoints differ from screened runs')
                binding = 'declared_match' if member.get('oof_declared_null') else 'full_reported'
            if member.get('k_eval') is None: missing.append(name + ': screened K unavailable')
            else: check(member['k_eval'] == arm.get('k_eval'), name + ': screened K differs')
            arm_reports[name] = dict(member=member.get('name'), status='PASS_TEST_COHORT_ONLY', binding=binding,
                                     k_eval=arm.get('k_eval'), k_matches_oof=member.get('k_eval') == arm.get('k_eval'), n_models=len(checks))
            fp_items.append((served, arm.get('k_eval'), (weights or {}).get(name, 0)))
        sub = outdir / 'submission.csv'
        output_sha = ctx.sha(sub)
        check(output_sha == r.get('final_sha256') == r.get('blended_sha256'), 'final CSV receipt hash differs')
        ok, count, err = base.RP.validate_csv(sub, sample)
        check(ok, 'final CSV invalid: ' + str(err))
        rt = r.get('runtime', {})
        check(rt.get('rad_k') == config.get('rad_k'), 'Public3 runtime K differs from build config')
        total = rt.get('total_seconds')
        check(isinstance(total, (float, int)) and math.isfinite(total) and total <= 28800
              and rt.get('fits_one_hour_margin') is True and rt.get('margin_seconds', 0) >= 3600,
              'Public3 runtime gate lacks 60-minute margin')
        kernel_dir = ctx.path(release.get('kernel_dir'))
        plan_path = kernel_dir / 'execution-plan.json' if kernel_dir is not None else None
        if plan_path is None or not plan_path.is_file():
            missing.append('1300-study execution plan unavailable')
            hidden = {}
        else:
            hidden = base.read_json(plan_path).get('runtime') or {}
            check(hidden.get('rad_k') == config.get('rad_k'), '1300-study runtime K differs from build config')
            htotal = hidden.get('total_seconds')
            check(hidden.get('studies') == 1300 and isinstance(htotal, (float, int))
                  and math.isfinite(htotal) and htotal <= 28800
                  and hidden.get('fits_one_hour_margin') is True
                  and hidden.get('margin_seconds', 0) >= 3600,
                  '1300-study projected runtime lacks 60-minute margin')
        out.update(submission_sha256=output_sha, submission_csv_sha256=output_sha, csv_matches_receipt=True,
                   csv_valid=ok, csv_studies=count, csv_error=err, studies=count, release=manifest.get('release'),
                   release_manifest_sha256=digest, reduction_state='expected_reported', arms=arm_reports,
                   checkpoint_fingerprint=None, checkpoint_fingerprint_basis='Stage2 anchor checkpoint inventory is unavailable; own-only identity would falsely equate anchor blends with own-only submissions. Duplicate detection retains release-manifest and member fingerprints.',
                   runtime=dict(projected_seconds=hidden.get('total_seconds'), studies=hidden.get('studies'),
                                budget_seconds=28800, public3_projected_seconds=total,
                                within_budget=not any('runtime' in h for h in hard) and bool(hidden),
                                scope='1300-study generated execution plan plus Public3 launch gate'))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        missing.append('EXP126 binding unavailable: ' + str(exc))
    out['stage2_anchor_blend'] = True
    out.update(problems=hard + missing, hard_failures=hard, engineering_pass=not hard,
               v7_adapter=dict(missing_bindings=missing, conflicting_bindings=hard))
    if missing or hard: out['checkpoint_fingerprint'] = None
    return out


_v7_engineering = engineering_section
def engineering_section(ctx, members, release):
    directory = ctx.path((release or {}).get('out_dir'))
    if directory is not None and (directory / 'exp126/receipt.json').is_file():
        return exp126_engineering(ctx, members, release)
    return _v7_engineering(ctx, members, release)


_original_oof = base.oof_section
def guarded_oof(ctx, members, baseline, lineage, spec):
    result = _original_oof(ctx, members, baseline, lineage, spec)
    report = base.lineage_report(ctx, baseline, {})
    gates = list(report.get('gate_failures') or [])
    if report['status'] in ('FAIL_DIRECT', 'FAIL_LINEAGE'):
        gates.append('baseline ' + report['status'])
    result['baseline']['lineage'] = dict(report, gate_failures=gates)
    release = spec.get('release') or {}
    directory = ctx.path(release.get('out_dir'))
    if directory is not None and (directory / 'exp126/receipt.json').is_file():
        result['available'] = False
        result['eligibility'] = 'NOT_AVAILABLE (anchor plus own blend has no pooled OOF; displayed OOF is own-arm context only)'
        result['filter'] = 'NOT_AVAILABLE'
        result.setdefault('notes', []).append('EXP126 own-arm OOF is context only; the anchor/own 0.7/0.3 blend has no pooled OOF. Public-only decision basis.')
    return result

base.oof_section = guarded_oof
base.engineering_section = engineering_section

_original_fingerprints = base.fingerprints_of
def fingerprints_of(members, engineering):
    result = _original_fingerprints(members, engineering)
    if engineering.get('stage2_anchor_blend'):
        result['members'] = None  # Own-only OOF does not identify the composite anchor blend.
    return result
base.fingerprints_of = fingerprints_of

if __name__ == '__main__':
    raise SystemExit(main())



