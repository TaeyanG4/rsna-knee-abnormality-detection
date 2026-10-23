"""P4/F4 admission extension v1.0.3; compose the unmodified K1 offline gate.

Changelog 1.0.3: allow only the exact registered F4 flat stage in addition to
P4; pin flat weights and retain four-member N32 32/32 with no projection inputs.

Changelog 1.0.2: require N32 32/32 successful rows, zero failures and all four
finite member predictions on every study; repair P4-ADM-01.
Changelog 1.0.1: qualify N32 from actual notebook receipts without synthetic
projection inputs; retain all visible checks and bind N32 evidence into offline
preparation. Runtime is judged by B's 8-step gate; projection is null.
Changelog 1.0.0: pin public-weight P4 and require all four members in every successful
study, with authenticated identities, finite predictions and measured timings.
No network, GPU, submission or modification of existing receipts.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from tools import k1_admit_v1 as base
from tools import p25_fb_stage_v1 as build

VERSION = '1.0.3'


def contract(e, config, reg, k, parent, count, diagnostic=False, builder=None):
    expected = (builder or build).configuration()
    cfg = build.parent.legacy.config(expected, diagnostic)
    e.check(config == expected, 'P4/F4 exact registered config; no O/A5')
    for key in ('config_sha256', 'registered_config_sha256', 'formula', 'model_sha256',
                'contract', 'k1_weight', 'parent_weight', 'r224_weight', 'drop_o',
                'kernel_id', 'release_version', 'diagnostic', 'never_submit'):
        e.check(k.get(key) == cfg.get(key) and reg.get(key) == cfg.get(key), 'P4 pinned '+key)
    for obj in (config, reg, k, parent):
        if expected.get('p_weight_mode') == 'flat':
            # Parent receipts describe P via their formula, not necessarily a
            # p_weight_mode field. Any explicit recipe must nevertheless match.
            e.check(obj.get('p_weight_mode', 'flat') == 'flat' and
                    obj.get('p_raptor_weights', [.5]*12) == [.5]*12 and
                    not any('a5' in key.lower() for key in obj),
                    'F4 flat weights and no A5 contract')
        else:
            e.check(obj.get('p_weight_mode', 'public') == 'public' and
                    not any('a5' in key.lower() or key == 'p_raptor_weights' for key in obj),
                    'P4 public weights and no A5 contract')
    if expected.get('p_weight_mode') == 'flat':
        for obj in (reg,k):
            e.check(obj.get('p_weight_mode') == 'flat' and obj.get('p_raptor_weights') == [.5]*12,
                    'F4 explicit registered flat weights')
    e.check(k.get('parent_mode') == 'INF3_P_R224' and parent.get('drop_o') is True,
            'P4 no O parent mode')
    for key, value in [('intended_members', {'R','S_R7','R224'}),
                       ('executed_members', {'R','S_R7','R224'}),
                       ('skipped_members', {'EXP143v2','EXP155','Q_A5q'})]:
        e.check(isinstance(parent.get(key), list) and len(parent[key]) == len(value)
                and set(parent[key]) == value, 'P4 parent '+key)
    models = expected['models']; ids = [m['id'] for m in models]
    e.check(len(ids) == 4 and k.get('model_identities') == models
            and k.get('k1_models') == ids, 'P4 all FOUR pinned members')
    e.check(k.get('status') == 'COMPLETE_SHARED_K1N' and k.get('k1_executed') is True
            and k.get('fail_loud_triggered') is False, 'P4 completed fail-loud execution')
    e.check(k.get('k1_max_failed_fraction') == .05 and k.get('k1_eval_stride') == 2,
            'P4 stride2 / failure limit')
    rows = k.get('per_study')
    base.need(isinstance(rows,list) and len(rows) == count and all(isinstance(r,dict) for r in rows),
              'P4 complete study telemetry')
    base.need(all(isinstance(r.get('uid'),str) and r['uid'] for r in rows)
              and len({r['uid'] for r in rows}) == count, 'P4 unique study telemetry')
    failed = 0
    for row in rows:
        e.check(type(row.get('ok')) is bool and row.get('config_sha256') == cfg['config_sha256']
                and row.get('model_identities') == models and row.get('k1_models') == ids,
                'P4 per-study identity')
        failed += not row['ok']
        if not row['ok']: continue
        pred, times = row.get('predictions',{}), row.get('per_model_seconds',{})
        e.check(set(pred) == set(times) == set(ids), 'P4 successful study executed FOUR members')
        for mid in ids:
            p = pred[mid]
            e.check(isinstance(p,list) and len(p) == 12 and all(type(v) in (int,float)
                    and math.isfinite(v) and 0 <= v <= 1 for v in p), 'P4 finite member prediction')
            base.positive(times[mid].get('classify'), 'P4 member classify seconds')
            base.positive(times[mid].get('load'), 'P4 member load seconds', zero=True)
        for phase, field in (('classify','classifier_seconds'),('load','load_seconds')):
            measured = sum(times[mid][phase] for mid in ids)
            if phase == 'classify':
                measured += base.positive(row.get('transfer_seconds'), 'P4 transfer seconds',zero=True)
            e.check(math.isclose(base.positive(row.get(field), 'P4 '+field,zero=True),
                                measured,rel_tol=1e-6,abs_tol=1e-6), 'P4 measured row '+field)
    e.check(k.get('failed_study_count') == failed and k.get('failed_study_fraction') == failed/count
            and failed/count <= .05 and k.get('k1_studies') == count-failed,
            'P4 failed fraction <= .05 and exact count')
    totals = k.get('per_model_seconds',{})
    e.check(set(totals) == set(ids), 'P4 four aggregate timings')
    for mid in ids:
        base.positive(totals[mid].get('classify'), 'P4 aggregate classify')
        for phase in ('load','classify'):
            actual = base.positive(totals[mid].get(phase), 'P4 aggregate '+phase, zero=phase=='load')
            measured = sum(base.positive(r.get('per_model_seconds',{}).get(mid,{}).get(phase),
                'P4 row '+phase,zero=True) for r in rows)
            e.check(math.isclose(actual,measured,rel_tol=1e-6,abs_tol=1e-6), 'P4 timing aggregation')


def validate(stage, kernel, version, visible_dir, n32_dir=None):
    from tools import p25_fb_flat_stage_v1 as flat_build
    selected = flat_build if base.local(stage) == flat_build.stage().resolve() else build
    base.need(n32_dir is not None, 'P4 requires exact candidate N32 functional execution')
    base.need(base.local(stage) == selected.stage().resolve(), 'P4/F4 exact stage')
    selected.run('validate')
    e,reg,k,projection = base.validate(stage,kernel,version,visible_dir,None)
    c = e.obj(base.local(stage)/'config.json')
    builder = selected if selected is flat_build else None
    contract(e,c,reg,k,e.obj(base.local(visible_dir)/'c85r7-receipt.json'),3,builder=builder)
    _, roster = base.table(e.read(base.local(visible_dir)/'submission.csv'))
    e.check({r['uid'] for r in k['per_study']} == set(roster), 'P4 executed visible roster')
    n = base.local(n32_dir)
    nr = e.obj(selected.stage(True)/'registration.json')
    nk = e.obj(n/'k1sr-receipt.json')
    parent = e.obj(n/'c85r7-receipt.json')
    contract(e,c,nr,nk,parent,32,True,builder=builder)
    e.check(parent.get('status') == 'DIAGNOSTIC_C85R7' and parent.get('studies') == 32
            and parent.get('findings') == 12 and parent.get('version') == nr['release_version']
            and parent.get('diagnostic') is True and parent.get('never_submit') is True,
            'P4 N32 parent identity')
    base.prediction_fallbacks(parent); base.codecs_active(parent.get('codecs'))
    versions = set()
    for name in ('k1sr-receipt.json','c85r7-receipt.json'):
        d=e.obj(n/(name+'.download.json'))
        e.check(d.get('status') == 'DOWNLOADED' and d.get('kernel') == nr['kernel_id']
                and type(d.get('version')) is int and d['version'] > 0
                and d.get('file') == d.get('local_name') == name
                and d.get('sha256') == base.sha(e.read(n/name)), 'P4 N32 download '+name)
        versions.add(d['version'])
    e.check(len(versions)==1, 'P4 one N32 version')
    for key in ('model_sha256', 'model_identities', 'formula', 'contract', 'r224_weight'):
        e.check(nk.get(key) == k.get(key), 'P4 N32/visible same '+key)
    e.check(type(nk.get('k1_studies')) is int and nk['k1_studies'] == 32
            and nk.get('failed_study_count') == 0 and nk.get('failed_study_fraction') == 0
            and all(row.get('ok') is True for row in nk['per_study']),
            'P4 N32 executed all FOUR members on 32/32 successful studies; zero failures')
    e.checks.remove('N32: NOT PROVIDED (optional; no runtime qualification claimed)')
    e.checks.append("P4 N32 functional qualification: PASS; runtime is judged by B's "
                    "8-step gate; no hidden runtime projection claimed")
    projection = None
    e.recheck()
    return e,reg,k,projection


def prepare(stage, kernel, version, visible_dir, n32_dir=None):
    e, reg, k, projected = validate(stage, kernel, version, visible_dir, n32_dir)
    visible, stage = base.local(visible_dir), base.local(stage)
    out = base.local(visible / base.OUTPUT)
    out.mkdir(exist_ok=True)
    output_version = VERSION
    registration = dict(
        schema="ordinary-visible-submission-registration/v1", tool_version=output_version,
        kernel_id=kernel, version=k["release_version"], stage=str(stage),
        kernel_sha256=reg["kernel_sha256"], metadata_sha256=reg["metadata_sha256"],
        stage_registration_sha256=e.hashes[str(stage / "registration.json")],
        formula=k["formula"], execution_verified=True, input_sha256=dict(e.hashes),
        projection=projected, independent_review="NOT PERFORMED BY THIS TOOL",
    )
    rp = out / "ordinary-visible-registration.json"
    registration_bytes = base.encoded(registration)
    binding = dict(schema_version="1.0.0", kernel=kernel, kernel_version=version,
                   receipt_sha256=e.hashes[str(visible / "c85r7-receipt.json")],
                   registration=str(rp), registration_sha256=base.sha(registration_bytes),
                   k1_receipt_sha256=e.hashes[str(visible / "k1sr-receipt.json")],
                   submission_sha256=k["submission_sha256"])
    review = (f"# K1 automated admission v{output_version}\n\n"
              f"Kernel: {kernel}\nKaggle version: {version}\nStage: {stage}\n\n"
              f"Formula (verbatim receipt): {k['formula']}\n\n"
              "Machine checks only; this document is not an independent reviewer verdict.\n"
              "B must retain the independent review and refresh live allowance/duplicates before submission.\n\n"
              + "\n".join("- " + check for check in e.checks)
              + "\n\nProjection: " + json.dumps(projected, sort_keys=True)
              + "\n\nVERDICT: PASS\n").encode()
    base.write_once(rp, registration_bytes)
    base.write_once(out / "submission-binding.json", base.encoded(binding))
    # Canonical wrapper binds its sibling sidecar. Original receipts stay untouched.
    base.write_once(out / "c85r7-receipt.json", e.read(visible / "c85r7-receipt.json"))
    pending = out / "machine-check.pending.md"
    base.write_once(pending, review)
    for path in (rp, out / "submission-binding.json", out / "c85r7-receipt.json", pending):
        e.read(path)
    identity = "K1-" + base.sha(f"{kernel}:{version}".encode())[:20]
    command_args = [base.sys.executable, str(base.ROOT / "tools/kaggle_submit_once.py"),
            "--id", identity, "--kernel", kernel, "--version", str(version),
            "--message", k["formula"], "--receipt", str(out / "c85r7-receipt.json")]
    result = base.subprocess.run(command_args + ["--review", str(pending), "--check-only"],
                            cwd=base.ROOT, capture_output=True, text=True, check=False)
    print("CHECK-ONLY " + result.stdout.strip())
    base.need(result.returncode == 0, "kaggle_submit_once --check-only REFUSED")
    try:
        outcome = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise base.Refuse("check-only returned malformed outcome") from None
    base.need(isinstance(outcome, dict) and outcome.get("outcome") == "ADMITTED_OFFLINE",
         "check-only did not admit")
    e.recheck()
    base.ledger_check(e, kernel, version)
    final_review = out / "READY-REVIEW.md"
    base.write_once(final_review, review.replace(
        b"\n\nVERDICT: PASS", b"\n\nCanonical --check-only: ADMITTED_OFFLINE\n\nVERDICT: PASS"))
    print("PASS: " + str(out))
    command = base.command_line(command_args + ["--review", str(final_review)])
    print("SUBMIT COMMAND (NOT EXECUTED): " + command)
    family = 'f4' if reg.get('p_weight_mode') == 'flat' else 'p4'
    base.write_once(out/(family+'-four-member-admission.json'),base.encoded(dict(
        schema=family+'-four-member-admission/v1',version=VERSION,status='PASS_OFFLINE_'+family.upper(),
        input_sha256=e.hashes,checks=e.checks,projection=projected,
        extension_sha256=base.sha(Path(__file__).read_bytes()),independent_review=False)))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['validate','prepare'])
    p.add_argument('--stage',default=str(build.stage()))
    p.add_argument('--kernel',default=build.SLUG)
    p.add_argument('--version',type=int,required=True)
    p.add_argument('--visible-dir',required=True)
    p.add_argument('--n32-dir',required=True)
    args=vars(p.parse_args()); action=args.pop('action')
    try:
        result=prepare(**args) if action=='prepare' else validate(**args)
        from tools import p25_fb_flat_stage_v1 as flat_build
        family = 'F4' if base.local(args['stage']) == flat_build.stage().resolve() else 'P4'
        print(json.dumps(dict(status='PASS_OFFLINE_'+family,action=action)))
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print('REFUSE: '+str(exc)); return 2
    return 0


if __name__=='__main__': raise SystemExit(main())
