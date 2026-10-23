"""Local CPU qualification for EXP169 v6; never reads labels or credentials.

1.2.2: registered-tolerance comparison of completed Kaggle v1.2.1 outputs.
1.2.0: parent-exact A4 R50/C4 identity and Q7 replay; unchanged v1.1.x MIL.
1.1.1: Amendment 2 six-candidate menu and registration-bound replays.
1.1.0: eight receipt-bound weight replays and complete loose-weight MIL smoke.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp169_release_v6 as v

VERSION = '1.2.2'  # Qualifier only; the scored serving release remains 1.2.1.


def comparison_delta(actual, expected, tolerance):
    """Fail closed on shape/nonfinite errors; apply both bounds inclusively."""
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    v.need(a.shape == b.shape and a.size > 0, 'comparison shape differs or empty')
    v.need(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite comparison')
    delta = np.abs(a-b)
    maximum, mean = float(delta.max()), float(delta.mean())
    return dict(status='PASS' if maximum <= tolerance['max_abs'] and
                mean <= tolerance['mean_abs'] else 'FAIL',
                max_abs_delta=maximum, mean_abs_delta=mean, tolerance=tolerance)


def column_health(values):
    """Last two axes are studies/targets; check every individual model."""
    import numpy as np
    a = np.asarray(values)
    v.need(a.ndim in (2, 3) and a.shape[-1] == 12 and a.shape[-2] >= 2,
           'health check needs >=2 studies and 12 targets')
    finite = bool(np.isfinite(a).all())
    constants = np.argwhere(np.ptp(a, axis=-2) == 0).tolist()
    bounded = bool(((a >= 0) & (a <= 1)).all())
    return dict(status='PASS' if finite and bounded and not constants else 'FAIL',
                finite=finite, bounded=bounded, constant_columns=constants)


def compare_kaggle(download, out, *, tolerances, sample=None, a5=None, a4=None, mil=None):
    """Compare saved real inference only; create a new audit, never rewrite inputs."""
    import numpy as np
    import zipfile
    from datetime import datetime, timezone
    out, download = Path(out), Path(download)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='FAIL', qualifier_version=VERSION, checks=[], inputs=[],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  qualifier_sha256=v.sha(__file__))
    checks = result['checks']

    def check(name, ok, **details):
        checks.append(dict(name=name, status='PASS' if ok else 'FAIL', **details))

    def pin(path):
        result['inputs'].append(v.file_pin(path))
        return Path(path)

    def read(path):
        return v.read(pin(path))

    def bound(path, digest):
        path = pin(path)
        v.need(v.sha(path) == digest, 'hash binding differs: ' + str(path))
        return path

    def cohort(actual, name):
        v.need(actual == ids, name + ': IDs/order differ from sample_submission')
        check(name + '/ids-order', True)

    def health(name, values):
        checks.append(dict(name=name + '/health', **column_health(values)))

    def compare(name, actual, expected, policy):
        checks.append(dict(name=name, **comparison_delta(actual, expected, policy)))

    def csv_values(path):
        cid, values = v5._read_csv(pin(path))
        cohort(cid, str(path))
        health(str(path), values)
        return values

    try:
        # Must exist before any prediction file is opened. Persist the exact policy.
        tol = read(tolerances)
        for key in ('formula', 't4', 'cpu_t4'):
            for metric in ('max_abs', 'mean_abs'):
                val = tol[key][metric]
                v.need(type(val) in (int, float) and np.isfinite(val) and val >= 0,
                       'invalid tolerance')
        v.need(tol['formula']['max_abs'] <= 1e-6, 'formula tolerance exceeds 1e-6')
        v.write_new(out / 'tolerances.json', tol)
        result['tolerances'] = tol
        sample = Path(sample or ROOT / 'storage/competition/sample_submission.csv')
        a5 = Path(a5 or ROOT / 'artifacts/experiments/EXP169_RELEASE_V5Q/v1.2.0/visible/a5_172')
        a4 = Path(a4 or ROOT / '.local/kout-a4-v2')
        mil = Path(mil or ROOT / '.local/relv6-smoke/members-v120')
        run = download / 'v6-run'
        rc = read(run / 'receipt.json')
        v.need(rc['status'] == 'PASS_COMPLETE_INFERENCE' and rc['candidate'] == 'A6'
               and rc['version'] == '1.2.1' and rc['no_fallback'] is True,
               'requires completed v1.2.1 A6 inference without fallback')
        regpath = bound(download / 'v6-code/registration.json', rc['registration_sha256'])
        reg = v.read(regpath)
        # Authenticate the portable registration against the existing built runtime.
        runtime = ROOT / v.EXP / 'v1.2.1/build-01/stage/runtime.zip'
        with zipfile.ZipFile(runtime) as archive:
            v.need(archive.read('registration.json') == regpath.read_bytes(),
                   'download registration differs from staged runtime')
        result['runtime_registration_match'] = str(runtime)
        specs = reg['candidates']
        v.need(specs == v.candidate_specs(), 'registered six-candidate menu differs')
        v.need(rc['candidates'] == {k: dict(**s, sha256=rc['candidates'][k]['sha256'])
                                   for k, s in specs.items()}, 'receipt candidate specs differ')
        v.need(reg['version'] == rc['version'], 'registration version differs')
        # Bind the arithmetic source before importing the frozen parents.
        bound(v.__file__, reg['sources']['tools/exp169_release_v6.py']['sha256'])
        v2, v5 = v.maths()
        v4 = v.a4_module()
        ids, _ = v5._read_csv(pin(sample))
        v.need(len(ids) == rc['studies'] and len(ids) >= 2, 'study count differs')
        result['studies'] = len(ids)
        member_path = bound(run / 'members.npz', rc['members_sha256'])
        with np.load(member_path, allow_pickle=False) as raw:
            keys = {'s42', 's43', 'exp155', 'public', 's6', 'l2', 'convnext', 'mil'}
            v.need(set(raw.files) == keys | {'ids'}, 'member roster differs (DINO forbidden)')
            cohort(raw['ids'].tolist(), 'members')
            scores = {k: raw[k].copy() for k in keys}
        for key, values in scores.items():
            shape = (({'l2':10, 'convnext':5, 'mil':10}[key], len(ids), 12)
                     if key in ('l2', 'convnext', 'mil') else (len(ids), 12))
            v.probability(values, shape, key)
            health(key, values)
        q7, _ = v.q_values({k:scores[k] for k in ('l2', 'convnext', 'mil')})
        # Independent explicit expression checks the production Q7 aggregation.
        r, c, m = (scores[k] for k in ('l2', 'convnext', 'mil'))
        explicit_q = v2.rank((v2.rank(((r[:5]+r[5:])/2).mean(0)) +
                              v2.rank(c.mean(0)) + v2.rank(m.mean(0))) / 3)
        compare('Q7/formula', q7, explicit_q, dict(max_abs=0., mean_abs=0.))
        own = (scores['s42'].astype(np.float64)+scores['s43'].astype(np.float64))/2
        written = {}
        for name, spec in specs.items():
            family, weight = spec['family'], spec['partner_weight']
            anchor = own if family == 'B6' else v5.trio_values(
                'A5-172' if family == 'A6' else 'A5q-172', ids,
                scores['public'], scores['s6'], own, scores['exp155'])
            expected = (1-weight)*v2.rank(anchor) + weight*v2.rank(explicit_q)
            path = bound(run / (name + '.csv'), rc['candidates'][name]['sha256'])
            written[name] = csv_values(path)
            compare('candidate/' + name, written[name], expected, tol['formula'])
        subpath = bound(download / 'submission.csv', rc['submission_sha256'])
        submission = csv_values(subpath)
        compare('submission=A6', submission, written['A6'], dict(max_abs=0., mean_abs=0.))
        check('submission=A6/bytes', v.sha(subpath) == v.sha(run / 'A6.csv'))

        # Earlier real T4 anchors authenticate all five files through their receipts.
        refs = {}
        for folder, expected_keys, mapping in (
            (a5 / 'exp169-trio', ['own','public','exp143'], {'own':'exp155','public':'public'}),
            (a4 / 'exp169-own42', ['own'], {'own':'s42'}),
            (a4 / 'exp169-s43-own', ['own'], {'own':'s43'})):
            for file in ('receipt.json','raw.npz','inputs.json','runtime.json','binding.json','submission.csv'):
                pin(folder / file)
            rid, arrays, _ = v4.authenticated_raw(folder, sample, expected_keys)
            cohort(rid, str(folder))
            refs.update({dest:arrays[src] for src, dest in mapping.items()})
        a5rc = read(a5 / 'v5q-receipt.json')
        refs['s6'] = csv_values(bound(a5 / 's6-port-predictions.csv', a5rc['s6_raw_sha256']))
        a4rc = read(a4 / 'exp169-executed-candidate.json')
        with np.load(bound(a4 / 'partner-raw.npz', a4rc['partner_raw_sha256']), allow_pickle=False) as raw:
            cohort(raw['ids'].tolist(), 'A4 partner')
            refs.update({k:raw[k].copy() for k in ('l2','convnext')})
        oldmanifest = v.read(bound(a4 / 'exp169-partner/release-manifest.json',
                                   a4rc['partner_manifest_sha256']))
        newmanifest = read(run / 'partner/release-manifest.json')
        entries = v.q7_entries(newmanifest)
        check('A4/R50-C4-roster-hashes', entries == v.q7_entries(oldmanifest))
        cpurc = read(mil / 'receipt.json')
        v.need(cpurc['status'] == 'PASS', 'CPU MIL reference incomplete')
        cpureg = v.read(bound(ROOT / v.EXP / 'v1.2.0/build-01/registration.json',
                             cpurc['registration_sha256']))
        check('MIL/roster-hashes', [(e['seed'],e['fold'],e['sha256']) for e in reg['mil']] ==
              [(e['seed'],e['fold'],e['sha256']) for e in cpureg['mil']])
        with np.load(bound(mil / 'members.npz', cpurc['output']['sha256']), allow_pickle=False) as raw:
            cohort(raw['ids'].tolist(), 'CPU MIL')
            refs['mil'] = raw['mil'].copy()
        for key, reference in refs.items():
            policy = tol['cpu_t4'] if key == 'mil' else tol['t4']
            health('reference/' + key, reference)
            compare('member/' + key, scores[key], reference, policy)
            if reference.ndim == 3:
                names = ([e['name'] for e in reg['mil']] if key == 'mil' else
                         [e['tag'] for e in entries if e['family'] == key])
                v.need(len(names) == len(reference), 'member names/count differ')
                for i, name in enumerate(names):
                    compare('member/' + key + '/' + name, scores[key][i], reference[i], policy)

        timing = dict(elapsed_seconds=rc['elapsed_seconds'], shards=[],
                      per_model_seconds=None,
                      note='Per-model times not recorded; shard times include multiple models. '
                           'Linear 1300-study projection includes startup and 3-study shard imbalance; '
                           'informational only, not a hidden-runtime qualification.')
        for phase in ('anchor', 'partner'):
            for shard in (0, 1):
                path = run / f'{phase}-{shard}.npz.json'
                item = read(path)
                bound(run / f'{phase}-{shard}.npz', item['sha256'])
                v.need(item['status'] == 'PASS', 'incomplete shard')
                timing['shards'].append({k:item[k] for k in ('phase','device','studies','seconds')})
        s6rc = read(run / 's6/controller/submission_receipt.json')
        v.need(s6rc['status'] == 'PASS' and s6rc['fallback_studies'] == 0, 'S6 fallback/incomplete')
        timing['s6_seconds'] = s6rc['seconds']
        timing['s6_worker_seconds'] = [w['seconds'] for w in s6rc['workers']]
        timing['projection_1300_seconds'] = rc['elapsed_seconds'] * 1300 / len(ids)
        timing['projection_1300_hours'] = timing['projection_1300_seconds'] / 3600
        result['timing'] = timing
    except (OSError, ValueError, KeyError, TypeError, AssertionError, zipfile.BadZipFile) as exc:
        # No raw rows or arrays in diagnostics, including on malformed inputs.
        check('structural-validation', False, error=str(exc), error_type=type(exc).__name__)
    check('inputs-unchanged', all(Path(p['path']).is_file() and v.sha(p['path']) == p['sha256']
                                 for p in result['inputs']))
    result['status'] = 'PASS' if all(c['status'] == 'PASS' for c in checks) else 'FAIL'
    v.write_new(out / 'receipt.json', result)
    return result


def replay(out, *, registration=None, members=None):
    import numpy as np
    v2, v5 = v.maths()
    v4 = v.a4_module()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for name, work in [('A4', ROOT / '.local/kout-a4-v2'),
                       ('A5', ROOT / 'artifacts/experiments/EXP169_RELEASE_V5Q/v1.2.0/visible/a5_172'),
                       ('A5q', ROOT / 'artifacts/experiments/EXP169_RELEASE_V5Q/v1.2.0/visible/a5q_172')]:
        rc = v.read(work / 'exp169-executed-candidate.json')
        raw_path = work / 'partner-raw.npz'
        v.need(v.sha(raw_path) == rc['partner_raw_sha256'], 'saved partner hash differs')
        with np.load(raw_path, allow_pickle=False) as raw:
            scores = {k: raw[k].copy() for k in ('l2', 'dino', 'convnext')}
            partner_ids = raw['ids'].tolist()
        q, _ = v2.aggregate(scores)
        sample = ROOT / 'storage/competition/sample_submission.csv'
        ids, a, _ = v4.authenticated_raw(work / 'exp169-trio',
                                        sample, ['own', 'public', 'exp143'])
        v.need(ids == partner_ids, 'replay cohort differs')
        if name == 'A4':
            _, s42, _ = v4.authenticated_raw(work / 'exp169-own42', sample, ['own'])
            _, s43, _ = v4.authenticated_raw(work / 'exp169-s43-own', sample, ['own'])
            # Neutralise S6 by R; its percentile blend has exactly R's ordering.
            actual = v.blend('A6', ids, s42['own'], s43['own'], q,
                             public=a['public'], exp155=a['own'], s6=a['public'])
            expected = v4.final_values('A4', s42['own'], s43['own'], q,
                                       public=a['public'], exp155=a['own'])
            np.testing.assert_array_equal(actual, expected)
            expected_hash = rc['submission_sha256']
            substitutions = 'Q7 := historical Q4; P954 := rank-equivalent R'
        else:
            r5 = v.read(work / 'v5q-receipt.json')
            v.need(v.sha(work / 's6-port-predictions.csv') == r5['s6_raw_sha256'], 'S6 replay hash differs')
            sid, s6 = v5._read_csv(work / 's6-port-predictions.csv')
            v.need(sid == ids, 'S6 replay cohort differs')
            actual = v.blend('A6' if name == 'A5' else 'A6q', ids, a['exp143'], a['exp143'], q,
                             public=a['public'], exp155=a['own'], s6=s6)
            expected_hash = r5['submission_sha256']
            substitutions = 'Q7 := Q_A172; OWN := historical EXP143 (both seeds neutralised)'
        target = out / (name + '.csv')
        v5._write_csv(target, ids, actual)
        v.need(v.sha(work / 'submission.csv') == expected_hash, 'saved final hash differs')
        sid, saved = v5._read_csv(work / 'submission.csv')
        _, written = v5._read_csv(target)
        v.need(sid == ids, 'saved final cohort differs')
        np.testing.assert_array_equal(written, saved)
        # Historical A4 uses a different CSV formatter; values must be exact.
        if name != 'A4':
            v.need(v.sha(target) == expected_hash, 'A5 CSV bytes differ')
        results.append(dict(name=name, status='PASS', studies=len(ids), substitutions=substitutions,
                            max_abs_delta=float(np.max(np.abs(written-saved))),
                            byte_identical=v.sha(target) == expected_hash,
                            output=v.file_pin(target), reference_sha256=expected_hash,
                            partner_sha256=v.sha(raw_path)))
    # Exercise all new final CSV paths using authenticated saved member arrays.
    # New Q7 uses authenticated A4 R50/C4 plus actual fresh ten-member MIL.
    work4 = ROOT / '.local/kout-a4-v2'
    _, s42, _ = v4.authenticated_raw(work4 / 'exp169-own42', sample, ['own'])
    _, s43, _ = v4.authenticated_raw(work4 / 'exp169-s43-own', sample, ['own'])
    if registration:
        v.need(not v.validate(registration)["blockers"], "replay registration blocked")
    specs = v.read(registration)["candidates"] if registration else v.candidate_specs()
    v.need(members is not None, 'Q7 replay requires authenticated member replay')
    member_root = Path(members)
    member_receipt = v.read(member_root / 'receipt.json')
    source = member_root / 'a4-q7-members.npz'
    v.need(member_receipt['status'] == 'PASS' and v.sha(source) == member_receipt['replay_output']['sha256'],
           'Q7 member replay binding differs')
    with np.load(source, allow_pickle=False) as raw:
        v.need(raw['ids'].tolist() == ids, 'Q7 member cohort differs')
        q7_scores = {k: raw[k].copy() for k in ('l2', 'convnext', 'mil')}
    q, _ = v.q_values(q7_scores)
    r, c, mil = (q7_scores[k] for k in ('l2', 'convnext', 'mil'))
    np.testing.assert_array_equal(q, v2.rank((v2.rank(((r[:5]+r[5:])/2).mean(0)) +
                                            v2.rank(c.mean(0)) + v2.rank(mil.mean(0)))/3))
    extra = dict(public=a['public'], exp155=a['own'], s6=s6)
    variants = v.candidate_values(specs, 'A6', ids, s42['own'], s43['own'], q, **extra)
    own = (s42['own'].astype(np.float64) + s43['own'].astype(np.float64)) / 2
    checks = []
    for name, values in variants.items():
        family, weight = specs[name]['family'], specs[name]['partner_weight']
        anchor = (own if family == 'B6' else v5.trio_values(
            'A5-172' if family == 'A6' else 'A5q-172', ids, a['public'], s6, own, a['own']))
        expected = (1-weight)*v2.rank(anchor) + weight*v2.rank(q)
        np.testing.assert_array_equal(values, expected)
        if name in ('A6', 'A6q'):
            np.testing.assert_array_equal(values, v5.blend_arrays(
                'A5-172' if name == 'A6' else 'A5q-172', ids, a['public'], s6, own, a['own'], q))
        elif name == 'B6':
            np.testing.assert_array_equal(values, v2.final_values(own, q, 'B172'))
        target = out / (name + '.csv')
        v5._write_csv(target, ids, values)
        checks.append(dict(candidate=name, **specs[name], status='PASS', studies=len(ids),
                           max_abs_delta=0., output=v.file_pin(target)))
    bonly = v.candidate_values(specs, 'B6-w50', ids, s42['own'], s43['own'], q)
    for name, values in bonly.items():
        np.testing.assert_array_equal(values, variants[name])
    result = dict(status='PASS', registration_sha256=v.sha(registration) if registration else None, cases=results, weighted_cases=checks,
                  weighted_substitution='None: saved authenticated A4 R50/C4 + fresh CPU MIL on all visible studies',
                  members_sha256=v.sha(source),
                  b_only_same_outputs=True, source_sha256=v.sha(v.__file__),
                  qualifier_sha256=v.sha(__file__))
    v.write_new(out / 'receipt.json', result)
    return result


def smoke(out, *, studies=2):
    import numpy as np
    import torch
    from training.arms import mil
    from torch.utils.data import DataLoader
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    v.backend(strict=True)
    result = dict(status='PARTIAL', checks=[], blockers=[], source_sha256=v.sha(v.__file__))
    # Dummy labels exist only to use the official evaluation interface.
    # No ground-truth or radiology report file is opened.
    corpus_path = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
    corpus = mil.common.Corpus(corpus_path)
    for seed, fold, checkpoint in [
        (42, 0, ROOT / 'artifacts/experiments/EXP185_FINDING_MIL/outputs/S42/results/EXP185_FINDING_MIL_F0_S42/model_best.pt'),
        (43, 0, ROOT / v.F0 / 'outputs/model_best.pt')]:
        with checkpoint.with_name('oof_k36_flip_tta.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))[:studies]
        ids = [r[v.UID] for r in rows]
        saved = np.asarray([[float(r[t]) for t in v.LABELS] for r in rows])
        entry = dict(seed=seed, fold=fold, **v.file_pin(checkpoint))
        if seed == 43:
            entry['spec_sha256'] = '43fd7a50bd45b6b4c2047975b313c7bf3557424abad30a072c14ec433906f435'
        model = v.load_mil(checkpoint, entry, 'cpu')
        ds = mil.MILWindows(corpus_path, ids, {uid:np.zeros(12, np.float32) for uid in ids}, 36, False, seed)
        loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=mil.collate)
        reference, order, batches = mil.evaluate(model, loader, 'cpu', 'none', corpus, tta=True)
        v.need(order == ids, 'MIL reference ordering differs')
        actual = []
        for i, uid in enumerate(ids):
            centers = mil.common.select_centers(corpus.mask(uid), 64, 36, corpus.spec, 'within-slot-valid')
            x = v.mil_windows(corpus.volume(uid), centers)
            np.testing.assert_array_equal(x, ds[i][0].numpy())
            actual.append(v.mil_prediction(model, x, uid, corpus, 'cpu'))
        actual = np.asarray(actual)
        np.testing.assert_array_equal(actual, reference)
        delta = np.abs(actual-saved)
        v.need(delta.max() <= .002 and delta.mean() <= .0005, 'CPU versus saved fp16 OOF exceeds tolerance')
        result['checks'].append(dict(name=f'MIL-S{seed}-F{fold}', status='PASS', studies=len(ids),
                  cpu_reference_max_delta=0., saved_fp16_oof_max_delta=float(delta.max()),
                  saved_fp16_oof_mean_delta=float(delta.mean()), checkpoint=v.file_pin(checkpoint),
                  oof=v.file_pin(checkpoint.with_name('oof_k36_flip_tta.csv'))))
        print(f'MIL S{seed} F{fold}: PASS', flush=True)
        del model, loader, ds
        gc.collect()
    competition = ROOT / '.local/relv6-smoke/competition'
    rows = list(csv.DictReader((competition / 'test_series.csv').open()))
    ids = list(dict.fromkeys(r[v.UID] for r in rows))[:1]
    paths = dict(s42=str(ROOT / v.B172 / 'own-anchor/assets'), s43=str(ROOT / v.V4 / 's43/assets'),
                 exp155=str(ROOT / v.E155 / 'assets'))
    ns, models = v.anchor_models(paths, 'A6', 'cpu', selected=['s42', 's43', 'exp155'])
    reader = v.anchor_reader(ns)
    records = {uid:[r for r in rows if r[v.UID] == uid] for uid in ids}
    for uid in ids:
        vol, mask = reader(uid, records, str(competition / 'test_series'))
        v.need(np.any(vol) and np.any(mask), 'empty DICOM input')
        x = ns['gpu_windows'](vol, mask, 94, 320, 'cpu')
        xf = ns['gpu_windows'](vol, mask, 94, 320, 'cpu', mirror=True)
        for key, model in models.items():
            p = (v.infer(model, x, 'cpu', anchor=True) + v.infer(model, xf, 'cpu', anchor=True)) * .5
            result['checks'].append(dict(name='DICOM-anchor-'+key, status='PASS', studies=1, finite=bool(np.isfinite(p).all())))
            print('DICOM anchor '+key+': PASS', flush=True)
    try:
        reader(ids[0], records, str(out / 'absent-series'))
    except ValueError:
        result['checks'].append(dict(name='missing-DICOM-fails-closed', status='PASS'))
    else:
        raise AssertionError('missing DICOM accepted')
    del models, model, x, xf, vol
    gc.collect()
    partner = ROOT / v.B172 / 'assets'
    sys.path.insert(0, str(partner / 'code'))
    v2, _ = v.maths()
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_common import select_centers, windows_from
    from raptor_radimagenet import center3_windows_from, normalise
    from raptor_runtime import normalize_input
    corpus = DicomCorpus.from_records(ids, rows, DirectoryDicomSource(competition / 'test_series'))
    uid = ids[0]
    vol, mask = corpus.volume(uid), corpus.mask(uid)
    centers = select_centers(mask, 64, 36, corpus.spec, 'within-slot-valid')
    flags = np.asarray(v2.base.plane_flags(centers, corpus.spec), bool)
    entries = v.read(partner / 'release-manifest.json')['entries']
    for family in ('l2',):
        entry = next(e for e in entries if e['family'] == family)
        model = v2._load_models(partner, [entry], 'cpu')[0]
        x = (normalise(center3_windows_from(vol, centers, 224), 'radimagenet') if family == 'l2'
             else normalize_input(windows_from(vol, centers, 224), 'imagenet'))
        p = v.infer(model, x, 'cpu')
        if family == 'l2':
            x[flags] = x[flags, :, :, ::-1]
            p = (p + v.infer(model, x, 'cpu')) / 2
        result['checks'].append(dict(name='DICOM-partner-'+family, status='PASS', studies=1, finite=bool(np.isfinite(p).all())))
        print('DICOM partner '+family+': PASS', flush=True)
        del model
        gc.collect()
    checkpoint = ROOT / v.F0 / 'outputs/model_best.pt'
    model = v.load_mil(checkpoint, dict(seed=43,fold=0,**v.file_pin(checkpoint),
                      spec_sha256='43fd7a50bd45b6b4c2047975b313c7bf3557424abad30a072c14ec433906f435'), 'cpu')
    p = v.mil_prediction(model, v.mil_windows(vol, centers), uid, corpus, 'cpu')
    result['checks'].append(dict(name='DICOM-partner-MIL',status='PASS',studies=1,finite=bool(np.isfinite(p).all())))
    result['blockers'] = ['Fresh public R inference: checkpoint not in named local stages',
                          'Fresh S6 inference: anatomy.pt not in named local stages',
                          'Complete ten-fold upload inference is qualified separately by mil-pack',
                          'Actual two-T4 visible/hidden inference not executed locally']
    result['seconds'] = time.monotonic()-started
    v.write_new(out / 'receipt.json', result)
    return result


def package(stage, out):
    stage, out = Path(stage).resolve(), Path(out).resolve()
    v.need(not out.exists(), 'package smoke needs fresh directory')
    v.safe_extract(stage / 'runtime.zip', out)
    # -I excludes both the working tree and inherited PYTHONPATH.
    code = ('import sys; sys.path.insert(0, '+repr(str(out))+'); '
            'from tools import exp169_release_v6 as v; '
            'from training.arms import mil; from tools import exp189_s6_port; '
            'v.maths(); v.a4_policy(); print("ISOLATED_RUNTIME_IMPORT_PASS")')
    proc = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=out,
                          capture_output=True, text=True, check=True)
    result = dict(status='PASS', runtime_sha256=v.sha(stage / 'runtime.zip'), stdout=proc.stdout.strip())
    for name in v.CANDIDATES:
        meta = v.read(stage / name / 'kernel-metadata.json')
        v.need(meta['title'] == meta['id'].split('/')[-1], 'kernel title differs from slug')
        v.need(v.a4_module().C4_DATASET in meta['dataset_sources'], 'missing C4 mount')
    result['slug_titles_and_c4_mounts'] = 6
    v.write_new(out / 'qualification.json', result)
    return result


def mil_pack(registration, stage, out):
    """Run all uploaded weights on one shared real DICOM volume, then new-fold OOF parity."""
    import numpy as np
    from training.arms import mil
    from torch.utils.data import DataLoader
    m = v.read(registration)
    v.need(not v.validate(m)['blockers'], 'MIL smoke requires complete registration')
    folder = v.check_mil_upload(m, Path(stage) / 'mil-dataset')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    v.backend(strict=True)
    sys.path.insert(0, str(ROOT / v.B172 / 'assets/code'))
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_common import select_centers
    competition = ROOT / '.local/relv6-smoke/competition'
    with (competition / 'test_series.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    uid = rows[0][v.UID]
    corpus = DicomCorpus.from_records([uid], rows, DirectoryDicomSource(competition / 'test_series'))
    vol, mask = corpus.volume(uid), corpus.mask(uid)
    v.need(np.any(vol) and np.count_nonzero(mask) >= 3, 'empty DICOM study')
    centers = select_centers(mask, 64, 36, corpus.spec, 'within-slot-valid')
    x = v.mil_windows(vol, centers)
    corpus_path = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
    oof_corpus = mil.common.Corpus(corpus_path)
    outputs, checks, parity = [], [], []
    for entry in m['mil']:
        model = v.load_mil(folder / entry['name'], entry, 'cpu')
        p = v.mil_prediction(model, x, uid, corpus, 'cpu')
        outputs.append(p)
        checks.append(dict(name=entry['name'], status='PASS', finite=bool(np.isfinite(p).all()),
                           checkpoint_sha256=entry['sha256']))
        print(entry['name'] + ': DICOM PASS', flush=True)
        if entry['seed'] == 43 and entry['fold'] > 0:
            oof = Path(entry['path']).with_name('oof_k36_flip_tta.csv')
            rc = v.read(entry['collection_receipt']['path'])
            v.need(v.sha(oof) == rc['outputs'][oof.name], 'OOF receipt hash differs')
            with oof.open(newline='') as stream:
                selected = [next(csv.DictReader(stream))]
            ids = [r[v.UID] for r in selected]
            saved = np.asarray([[float(r[t]) for t in v.LABELS] for r in selected])
            ds = mil.MILWindows(corpus_path, ids, {i:np.zeros(12, np.float32) for i in ids}, 36, False, 43)
            loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=mil.collate)
            reference, order, _ = mil.evaluate(model, loader, 'cpu', 'none', oof_corpus, tta=True)
            v.need(order == ids, 'OOF order differs')
            study = ids[0]
            ctr = select_centers(oof_corpus.mask(study), 64, 36, oof_corpus.spec, 'within-slot-valid')
            windows = v.mil_windows(oof_corpus.volume(study), ctr)
            np.testing.assert_array_equal(windows, ds[0][0].numpy())
            actual = v.mil_prediction(model, windows, study, oof_corpus, 'cpu')[None]
            np.testing.assert_array_equal(actual, reference)
            delta = np.abs(actual-saved)
            v.need(delta.max() <= .002 and delta.mean() <= .0005, 'saved OOF tolerance exceeded')
            parity.append(dict(name=entry['name'], status='PASS', studies=1, cpu_reference_max_delta=0.,
                 saved_fp16_oof_max_delta=float(delta.max()), saved_fp16_oof_mean_delta=float(delta.mean()),
                 oof=v.file_pin(oof)))
            print(entry['name'] + ': OOF PASS', flush=True)
            del ds, loader
        del model
        gc.collect()
    values = np.asarray(outputs, np.float32)[:, None, :]
    v.probability(values, (10, 1, 12), 'complete packaged MIL')
    with (out / 'mil.npz').open('xb') as stream:
        np.savez(stream, ids=[uid], mil=values, mean_probability=values.mean(0))
    prior = ROOT / '.local/relv6-smoke/mil-pack-v111'
    old = v.read(prior / 'receipt.json')
    v.need(v.sha(prior / 'mil.npz') == old['output']['sha256'], 'v1.1.1 MIL reference changed')
    with np.load(prior / 'mil.npz', allow_pickle=False) as reference:
        v.need(reference['ids'].tolist() == [uid], 'v1.1.1 MIL cohort differs')
        np.testing.assert_array_equal(values, reference['mil'])
    previous = ROOT / v.EXP / 'v1.1.1/build-01/stage/mil-dataset'
    identity = {p.name: v.sha(p) for p in folder.iterdir() if p.is_file()}
    v.need(identity == {p.name: v.sha(p) for p in previous.iterdir() if p.is_file()},
           'MIL dataset differs from v1.1.1')
    result = dict(status='PASS', studies=1, shared_dicom_volumes=1, shared_window_arrays=1,
                  weights=checks, new_fold_oof=parity, output=v.file_pin(out / 'mil.npz'),
                  registration_sha256=v.sha(registration), source_sha256=v.sha(v.__file__),
                  qualifier_sha256=v.sha(__file__), seconds=time.monotonic()-started,
                  v111_exact_member_delta=0., v111_reference=v.file_pin(prior / 'mil.npz'),
                  unchanged_dataset_files=identity)
    v.write_new(out / 'receipt.json', result)
    return result


def members_replay(registration, stage, out):
    """Actual Q7 CPU worker on every visible study, bound to A4 v2 member slots."""
    import numpy as np
    m = v.read(registration)
    v.need(not v.validate(m)['blockers'], 'member replay requires complete registration')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parent = ROOT / '.local/kout-a4-v2'
    receipt = v.read(parent / 'exp169-executed-candidate.json')
    refpath = parent / 'partner-raw.npz'
    v.need(v.sha(refpath) == receipt['partner_raw_sha256'], 'A4 raw reference changed')
    with np.load(refpath, allow_pickle=False) as raw:
        ids = raw['ids'].tolist()
        reference = {k: raw[k].copy() for k in ('l2', 'convnext')}
    paths = {}
    for key in ('partner', 'c4'):
        pin = m['assets'][key]
        target = out / key
        v.safe_extract(Path(pin['path']), target)
        manifests = list(target.rglob('release-manifest.json'))
        v.need(len(manifests) == 1 and v.sha(manifests[0]) == pin['manifest_sha256'], 'archive manifest differs')
        paths[key] = str(manifests[0].parent)
    manifest = v.a4_module().overlay_c4(paths['partner'], paths['c4'], m['assets']['c4']['manifest_sha256'])
    a4manifest = parent / 'exp169-partner/release-manifest.json'
    v.need(v.sha(a4manifest) == receipt['partner_manifest_sha256'], 'A4 member manifest changed')
    entries = v.q7_entries(manifest)
    v.need(entries == v.q7_entries(v.read(a4manifest)), 'R50/C4 A4 member identity differs')
    paths['mil'] = str(v.check_mil_upload(m, Path(stage) / 'mil-dataset'))
    competition = ROOT / '.local/relv6-smoke/competition'
    with (competition / 'test_series.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    job = dict(ids=ids, rows=rows, images=str(competition / 'test_series'), paths=paths, mil=m['mil'])
    v.write_new(out / 'job.json', job)
    # Use the real production worker, including the same mixed backend policy.
    v.backend()
    scores = v.partner_worker(job, 'cpu')
    v.need(set(scores) == {'l2', 'convnext', 'mil'}, 'DINO entered Q7 worker')
    v2, _ = v.maths()
    tolpath = ROOT / 'artifacts/experiments/EXP169_RELEASE_V2/pre-reading/partner-crossgpu-tol.json'
    tol = v.read(tolpath)['default']
    checks = []
    for key in reference:
        delta = np.abs(scores[key]-reference[key])
        v.need(delta.max() <= tol['max_abs'] and delta.mean() <= tol['mean_abs'],
               'CPU vs A4 fp16 tolerance exceeded: ' + key)
        def family(a):
            return v2.rank(((a[:5]+a[5:])/2).mean(0) if key == 'l2' else a.mean(0))
        np.testing.assert_array_equal(family(scores[key]), family(reference[key]))
        checks.append(dict(family=key, status='PASS', members=len(scores[key]), studies=len(ids),
                           max_abs_delta=float(delta.max()), mean_abs_delta=float(delta.mean()),
                           exact_family_ranks=True))
        print(key + ': A4 member identity / CPU parity PASS', flush=True)
    with (out / 'members.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **scores)
    # Replay the retained families directly from the authenticated A4 output.
    replay_scores = {**reference, 'mil': scores['mil']}
    with (out / 'a4-q7-members.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **replay_scores)
    for key in reference:
        np.testing.assert_array_equal(replay_scores[key], reference[key])
    result = dict(status='PASS', studies=len(ids), dino_models_loaded=0, dino_inference_calls=0,
                  a4_member_entries_exact=True, a4_output_replay_exact=True, checks=checks,
                  member_pins=entries, a4_manifest=v.file_pin(a4manifest), a4_raw=v.file_pin(refpath),
                  tolerance=v.file_pin(tolpath), output=v.file_pin(out / 'members.npz'),
                  replay_output=v.file_pin(out / 'a4-q7-members.npz'),
                  registration_sha256=v.sha(registration), source_sha256=v.sha(v.__file__),
                  qualifier_sha256=v.sha(__file__), seconds=time.monotonic()-started)
    v.write_new(out / 'receipt.json', result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['replay','smoke','package','mil-pack','members','compare-kaggle'])
    p.add_argument('--out', required=True)
    p.add_argument('--stage')
    p.add_argument('--registration')
    p.add_argument('--members')
    p.add_argument('--download')
    p.add_argument('--tolerances')
    a = p.parse_args()
    if a.command == 'compare-kaggle':
        if not a.download or not a.tolerances:
            p.error('compare-kaggle requires --download and pre-registered --tolerances')
        result = compare_kaggle(a.download, a.out, tolerances=a.tolerances)
    elif a.command == 'replay':
        result = replay(a.out, registration=a.registration, members=a.members)
    elif a.command == 'mil-pack':
        result = mil_pack(a.registration, a.stage, a.out)
    elif a.command == 'members':
        result = members_replay(a.registration, a.stage, a.out)
    else:
        result = package(a.stage, a.out) if a.command == 'package' else globals()[a.command](a.out)
    print(json.dumps(result, indent=2))
    if a.command == 'compare-kaggle' and result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
