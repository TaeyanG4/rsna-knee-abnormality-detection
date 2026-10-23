"""Offline N32-ratio qualification v1.0.1; never push, submit or edit a stage.

Changelog v1.0.1: flat P is recipe identity; measured N32 limit is 9 h.

Changelog v1.0.0: candidate-bound P25 N32 timing gate, rechecked by admission.
The immutable external record qualifies runtime only; visible output and
independent review remain mandatory. N32 itself is always never-submit.
"""
from __future__ import annotations

import argparse
import math

from tools import k1_admit_v1 as admit
from tools import k1n_stage_v1 as builder

VERSION = '1.0.1'
RECORD = 'n32-ratio-qualification.json'


def measured_hours(parent_seconds, k1_seconds):
    parent = admit.positive(parent_seconds, 'N32 parent_seconds')
    k1 = admit.positive(k1_seconds, 'N32 k1_seconds')
    hours = 6.2 * (parent + k1) / 1785
    admit.need(math.isfinite(hours), 'nonfinite N32-ratio hours')
    return dict(formula='6.2 * (parent_seconds + k1_seconds) / 1785',
        anchor_hidden_hours=6.2, anchor_includes_queue=True,
        anchor_parent_seconds=1366, anchor_k1_seconds=419, anchor_total_seconds=1785,
        parent_seconds=parent, k1_seconds=k1, n32_total_seconds=parent+k1,
        hours=hours, limit_hours=builder.members.N32_QUALIFICATION_LIMIT_HOURS,
        scope='Runtime estimate, not a guarantee')


def stage_contract(e, stage, c, diagnostic):
    reg = e.obj(stage/'registration.json')
    cfg = builder.config(c, diagnostic)
    admit.need(all(reg.get(k) == v for k, v in cfg.items()),
               'N32-ratio stage/config contract mismatch')
    admit.need(admit.local(reg.get('stage', '')) == stage, 'N32-ratio stage directory mismatch')
    for name, key in (('release.ipynb', 'kernel_sha256'),
                      ('kernel-metadata.json', 'metadata_sha256')):
        admit.need(admit.sha(e.read(stage/'kernel'/name)) == reg.get(key),
                   'N32-ratio stage artifact hash mismatch: '+name)
    meta = e.obj(stage/'kernel/kernel-metadata.json')
    admit.need(meta.get('id') == cfg['kernel_id'], 'N32-ratio metadata identity mismatch')
    notebook = e.obj(stage/'kernel/release.ipynb')
    admit.need(builder.runtime9.notebook_config(notebook) == cfg,
               'N32-ratio notebook config mismatch')
    admit.need(e.obj(stage/'config.json') == c, 'N32-ratio canonical config mismatch')
    return reg


def evaluate(e, stage, directory):
    """Recompute every qualification input; never trust a saved PASS boolean."""
    stage, directory = admit.local(stage), admit.local(directory)
    c = builder.canonical(e.obj(stage/'config.json'))
    admit.need(c.get('projection_basis') == 'n32_ratio', 'registered n32_ratio required')
    reg = stage_contract(e, stage, c, False)
    nstage = stage.with_name(stage.name+'-n32')
    nreg = stage_contract(e, nstage, c, True)
    receipt = directory/'k1sr-receipt.json'
    n = e.obj(receipt)
    admit.need(n.get('status') == 'COMPLETE_SHARED_K1N' and n.get('k1_executed') is True
               and type(n.get('k1_studies')) is int and n['k1_studies'] == 32,
               'N32 ratio requires 32/32 executed K1 studies')
    admit.need(type(n.get('failed_study_count')) is int and n['failed_study_count'] == 0
               and type(n.get('failed_study_fraction')) in (int, float)
               and n['failed_study_fraction'] == 0 and n.get('fail_loud_triggered') is False,
               'N32 ratio requires zero failed studies')
    # Identity includes recipe, weights, assets, guard, runtime and both slugs.
    for key, value in builder.config(c, True).items():
        admit.need(key in n and n[key] == value, 'N32 receipt contract mismatch: '+key)
    rows = n.get('per_study')
    admit.need(isinstance(rows, list) and len(rows) == 32 and
               all(isinstance(r, dict) and r.get('ok') is True and
                   isinstance(r.get('uid'), str) and r['uid'] and
                   not r.get('systemic_failure') for r in rows) and
               len({r['uid'] for r in rows}) == 32,
               'N32 ratio requires 32 unique successful rows')
    ids = [m['id'] for m in c['models']]
    for row in rows:
        admit.need(row.get('config_sha256') == reg['config_sha256'] and
                   row.get('k1_models') == ids and row.get('model_identities') == c['models'] and
                   row.get('k1_eval_stride') == c['k1_eval_stride'],
                   'N32 per-study candidate mismatch')
        predictions = row.get('predictions')
        admit.need(isinstance(predictions, dict) and set(predictions) == set(ids),
                   'N32 per-study missing executed model')
        for values in predictions.values():
            admit.need(isinstance(values, list) and len(values) == 12 and
                       all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                           for v in values), 'invalid N32 model output')
    phases = n.get('phases')
    admit.need(isinstance(phases, dict), 'N32 phases required')
    timing_keys = ('k1_owner_seconds', 'extra_render_seconds', 'failed_capture_seconds')
    for key in timing_keys:
        aggregate = admit.positive(phases.get(key), 'N32 phase '+key, zero=True)
        total = sum(admit.positive(r.get(key), 'N32 row '+key, zero=True) for r in rows)
        admit.need(math.isclose(aggregate, total, rel_tol=1e-6, abs_tol=1e-6),
                   'N32 phase/per-study mismatch: '+key)
    k1_seconds = admit.positive(n.get('k1_seconds'), 'N32 k1_seconds')
    admit.need(math.isclose(k1_seconds, sum(phases[k] for k in timing_keys),
                           rel_tol=1e-6, abs_tol=1e-6), 'N32 K1 total/phase mismatch')
    download = e.obj(directory/'k1sr-receipt.json.download.json')
    receipt_sha = e.hashes[str(receipt)]
    admit.need(download.get('status') == 'DOWNLOADED' and
               download.get('kernel') == nreg['kernel_id'] and
               type(download.get('version')) is int and download['version'] > 0 and
               download.get('file') == download.get('local_name') == receipt.name and
               download.get('sha256') == receipt_sha, 'N32 exact-version download provenance')
    projection = measured_hours(n.get('parent_seconds'), n.get('k1_seconds'))
    admit.need(projection['hours'] <= projection['limit_hours'],
               f"N32 ratio exceeds {projection['limit_hours']:.1f} h")
    return dict(schema='k1n-n32-ratio-qualification/v1', tool_version=VERSION,
        stage=str(stage), kernel_id=reg['kernel_id'], release_version=reg['release_version'],
        config_sha256=reg['config_sha256'], n32_kernel_id=nreg['kernel_id'],
        n32_kernel_version=download['version'], receipt_sha256=receipt_sha,
        projection_basis='n32_ratio', projection=projection, k1_studies=32,
        failed_study_count=0, qualified_for_submission=True,
        qualification_scope='Runtime only; ordinary visible admission and independent review required',
        input_sha256=dict(e.hashes))


def qualify(stage, n32_dir):
    e = admit.Evidence()
    result = evaluate(e, stage, n32_dir)
    e.recheck()
    destination = admit.local(n32_dir)/RECORD
    admit.need(not destination.is_relative_to(admit.local(stage)) and
               not destination.is_relative_to(admit.local(stage).with_name(
                   admit.local(stage).name+'-n32')), 'qualification must be external to stage')
    admit.write_once(destination, admit.encoded(result))
    e.recheck()
    return result


def verify(e, stage, n32_dir):
    admit.need(n32_dir is not None, 'N32-ratio qualification record required: provide --n32-dir')
    # Use a separate evidence set so deterministic record bytes do not depend
    # on the caller's preceding visible-output checks.
    inputs = admit.Evidence()
    expected = evaluate(inputs, stage, n32_dir)
    recorded = e.obj(admit.local(n32_dir)/RECORD)
    admit.need(recorded == expected, 'N32-ratio qualification record mismatch')
    for name, digest in inputs.hashes.items():
        admit.need(admit.sha(e.read(name)) == digest, 'N32 qualification input changed')
    e.checks.append(f"Candidate-bound N32 ratio <= {expected['projection']['limit_hours']:.1f} h, "
                    '32/32 K1, zero failed: PASS')
    return expected['projection']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['qualify'])
    parser.add_argument('--stage', required=True)
    parser.add_argument('--n32-dir', required=True)
    args = parser.parse_args()
    try:
        result = qualify(args.stage, args.n32_dir)
        print(admit.encoded(result).decode(), end='')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print('REFUSE: '+str(exc))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
