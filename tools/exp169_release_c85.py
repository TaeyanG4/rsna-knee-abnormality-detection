"""C85 v1.0.4: preregistered nested ranks, frozen A5q and P956 serving.

2026-10-12: new wrapper only; no historical source/stage changes. CPU commands
never infer, install, submit, query scores, or operate on a Kaggle account.
Changelog v1.0.1: verify staged CSV, atomically publish before success receipt;
publication exceptions invalidate only this invocation's output and record failure.
Changelog v1.0.2: S6 anatomy localization paths are telemetry only, matching
the public S6 pipeline; prediction fallbacks and nonzero worker exits still fail.
Changelog v1.0.3: successful-row ranks and survivor substitution; recover child
outputs, own-only CoAtNet pass, and receipt comparisons as telemetry.
Changelog v1.0.4: retain EXP155's frozen shared serving binding, including its
input_record callback, when wrapping the own-only arms with prediction journals.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

try:
    from tools import exp169_release_p956 as p, exp169_release_v2 as a172
except ImportError:
    import exp169_release_p956 as p
    import exp169_release_v2 as a172

v5q, s6 = p.v5q, p.s6
need, sha, read_json, write_once, json_bytes = p.need, p.sha, p.read_json, p.write_once, p.json_bytes
ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.4"
OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85/v1.0.4"
A5Q = v5q.OUT / "visible" / v5q.CANDIDATES["A5q-172"]
KERNEL_ID = "taeyangg4/knee-exp169-c85-v104"
FORMULA = "P=.5*r(R)+.5*r(S6); O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46; C85=.85*r(P)+.15*r(O)"
MANIFEST_SHA = "39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc"
P956_SHA = "7d04817dee31c36a51babf63e1ae2752301875586935d363fd241c9f60e8d901"
P956_REPLAY_SHA = "dea7cc58532d641352671f878139d7c90f92a7c325b452c536c74a8e6ad85417"
A5Q_RECEIPT_SHA = "794276e2f842f2421985f89e4ca9f1c981a4c78573e90e06a64b7f280e8c2c1f"
DATASETS = ["taeyangg4/knee-exp169-a172-v2-assets",
            "taeyangg4/knee-exp155-own0949-oai-v1-2-assets", *p.DATASETS]
MEMBERS = ["R", "S6", "EXP143v2", "EXP155", "Q_A5q"]


def rank(values):
    """P956's registered average-rank/N helper; Q keeps its frozen internal ranks."""
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    need(a.ndim == 2 and a.shape[1] == 12 and len(a) > 0, "invalid rank shape")
    return np.asarray([s6._pct(a[:, j]) for j in range(12)]).T


def scored_rank(values, failed):
    import numpy as np
    result = np.zeros_like(values, dtype=np.float64)
    if (~failed).any(): result[~failed] = rank(values[~failed])
    return result


def blend(ids, r, ours, exp143, exp155, q, failures=None):
    """All inner math stays float64; serialize only the final C85 matrix."""
    ids = v5q.validate_ids(ids)
    r, ours, exp143, exp155, q = [v5q._probabilities(name, arr, len(ids))
        for name, arr in zip(MEMBERS, (r, ours, exp143, exp155, q))]
    import numpy as np
    failures = failures or {}
    masks = [np.asarray([uid in failures.get(name, {}) for uid in ids]) for name in MEMBERS]
    masks[0] |= np.all(r == .5, axis=1)  # public cell 7's R failure sentinel
    rr, ss, e143, e155, qq = [scored_rank(v, f) for v, f in
                            zip((r, ours, exp143, exp155, q), masks)]
    rf, sf, f143, f155, fq = masks
    public = .5 * rr + .5 * ss
    public[rf], public[sf], public[rf & sf] = ss[rf], rr[sf], .5
    denom = 17 * (~f143).astype(int) + 17 * (~f155).astype(int) + 12 * (~fq).astype(int)
    own = (17 * e143 + 17 * e155 + 12 * qq) / np.maximum(denom, 1)[:, None]
    pf, of = rf & sf, denom == 0
    pr, oo = scored_rank(public, pf), scored_rank(own, of)
    result = .85 * pr + .15 * oo
    result[of], result[pf], result[pf & of] = pr[of], oo[pf], .5
    return result


def reference(r, ours, exp143, exp155, q):
    """Independent pandas transcription, including average ties."""
    import pandas as pd
    def rank(x):
        return pd.DataFrame(x).rank(method="average", pct=True).to_numpy()
    return .85 * rank(.5 * rank(r) + .5 * rank(ours)) + .15 * rank(
        (17 * rank(exp143) + 17 * rank(exp155) + 12 * rank(q)) / 46)


def array_sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def manifest(path):
    p.check_pin(path, MANIFEST_SHA)
    value = read_json(path)
    need(value["candidate"] == "A172" and value["generator_version"] == "2.0.2", "Q_A5q identity")
    expected = [(family, seed, fold) for family, seeds in
        (("l2", (42, 43)), ("dino", (42,)), ("convnext", (42,))) for seed in seeds for fold in range(5)]
    need([(e["family"], e["seed"], e["fold"]) for e in value["entries"]] == expected,
         "Q_A5q checkpoint roster/order")
    need(value["anchor_binding"]["checkpoint_sha256"][p.R_NAME] == p.R_SHA, "R identity")
    return value


def optional_json(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def telemetry(checks, name, check):
    try:
        value = check()
        checks[name] = True if value is None else value
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        checks[name] = f'{type(exc).__name__}: {exc}'


def clean_member(ids, values, failures=None):
    import numpy as np
    values = np.asarray(values).copy()
    need(values.shape == (len(ids), 12), 'member shape')
    failures = dict(failures or {})
    valid = np.isfinite(values).all(1) & ((values >= 0) & (values <= 1)).all(1)
    failures.update({uid: 'InvalidPrediction: nonfinite or out of range' for uid, ok in zip(ids, valid) if not ok})
    values[[uid in failures for uid in ids]] = .5
    return values, failures


def read_journal(directory, ids, members, expected_identity):
    import numpy as np
    identity = read_json(directory / 'journal-identity.json')
    need(identity == expected_identity, 'journal checkpoint/source identity')
    v5q.validate_ids(read_json(directory / 'ids.json'))
    need(set(read_json(directory / 'ids.json')) == set(ids), 'journal cohort')
    values = {name: np.full((len(ids), 12), .5) for name in members}
    failures = {name: {uid: 'MissingPrediction: child did not finish study' for uid in ids} for name in members}
    positions = {uid: i for i, uid in enumerate(ids)}
    path = directory / 'predictions.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try: row = json.loads(line)
            except ValueError: continue  # interrupted final append
            uid, name = row['uid'], row['member']
            need(uid in positions and name in values, 'journal UID/member')
            if 'values' in row:
                try: v = p.RBinding.prediction(row['values'])
                except (ValueError, TypeError) as exc:
                    failures[name][uid] = f'InvalidPrediction: {exc}'
                    continue
                values[name][positions[uid]] = v
                failures[name].pop(uid, None)
            else:
                failures[name][uid] = row['error']
    return values, failures


def authenticated_a172(directory):
    """Read completed studies, preserving hard model identities and soft receipts."""
    import numpy as np
    directory = Path(directory)
    mpath = directory / 'exp169-partner/release-manifest.json'
    if not mpath.exists(): mpath = directory / 'q-a5q-manifest.json'
    m = manifest(mpath) if mpath.exists() else None
    trio, checks = directory / 'exp169-trio', {}
    receipt = optional_json(directory / 'exp169-executed-candidate.json')
    tr = optional_json(trio / 'receipt.json')
    failures = {}
    if (trio / 'journal-identity.json').exists():
        need(m is not None, 'own manifest identity missing')
        ids = v5q.validate_ids(read_json(trio / 'ids.json'))
        values, failures = read_journal(trio, ids, ['EXP143v2', 'EXP155'], m['anchor_binding'])
    elif (trio / 'raw.npz').exists():
        need(m is not None, 'own manifest identity missing')
        need(read_json(trio / 'binding.json') == m['anchor_binding'], 'trio checkpoint/source binding differs')
        with np.load(trio / 'raw.npz', allow_pickle=False) as raw:
            need(raw['targets'].tolist() == list(p.LABELS), 'trio target/schema mismatch')
            ids = v5q.validate_ids(raw['ids'].tolist())
            values = {}
            for key, name in [('exp143', 'EXP143v2'), ('own', 'EXP155')]:
                if key in raw:
                    values[name], failures[name] = clean_member(ids, raw[key])
                else:
                    values[name] = np.full((len(ids), 12), .5)
                    failures[name] = dict.fromkeys(ids, 'MissingPrediction: missing own member')
    else:
        ids = read_json(directory / 'ids.json') if (directory / 'ids.json').exists() else []
        values = {name: np.full((len(ids), 12), .5) for name in ['EXP143v2', 'EXP155']}
        failures = {name: dict.fromkeys(ids, 'MissingPrediction: A172 child aborted') for name in values}
    q = np.full((len(ids), 12), .5)
    qfail = dict.fromkeys(ids, 'MissingPrediction: Q child did not score study')
    rawpath = directory / 'partner-raw.npz'
    # The per-model journal is flushed first; an interrupted NPZ write must not
    # hide predictions the child already completed.
    if rawpath.exists() and not (directory / 'q-journal/journal-identity.json').exists():
        need(m is not None, 'Q manifest identity missing')
        with np.load(rawpath, allow_pickle=False) as raw:
            qids = v5q.validate_ids(raw['ids'].tolist())
            need(set(qids) == set(ids), 'Q cohort mismatch')
            arrays = {name: raw[name].copy() for name in ('l2', 'dino', 'convnext')}
    elif (directory / 'q-journal/journal-identity.json').exists():
        need(m is not None, 'Q manifest identity missing')
        qids = ids
        qnames = [str(i) for i in range(20)]
        qvalues, qfailures = read_journal(directory / 'q-journal', ids, qnames,
                                        {'manifest_sha256': MANIFEST_SHA})
        arrays = {name: np.stack([qvalues[str(i)] for i in indices]).astype(np.float32)
                  for name, indices in [('l2', range(10)), ('dino', range(10,15)), ('convnext', range(15,20))]}
        for i, name in enumerate(qnames):
            for uid in qfailures[name]:
                family, offset = ('l2', 0) if i < 10 else ('dino', 10) if i < 15 else ('convnext', 15)
                arrays[family][i-offset, ids.index(uid)] = np.nan
    else:
        arrays = None
    if arrays is not None:
        good = np.ones(len(ids), bool)
        for name, value in arrays.items():
            need(value.shape == (10 if name == 'l2' else 5, len(ids), 12), 'Q family shape')
            good &= np.isfinite(value).all((0,2)) & ((value >= 0) & (value <= 1)).all((0,2))
        if good.any():
            scored, families = a172.aggregate({name: value[:, good] for name, value in arrays.items()})
            for uid, row in zip(np.asarray(qids)[good], scored):
                q[ids.index(uid)] = row
                qfail.pop(uid, None)
            if good.all():
                checks['bundle_rank_digest'] = array_sha(scored) == receipt.get('bundle_rank_sha256')
                checks['family_rank_digests'] = {k: array_sha(v) for k,v in families.items()} == receipt.get('family_rank_sha256')
    failures['Q_A5q'] = qfail
    for name, digest in tr.get('files', {}).items():
        telemetry(checks, 'trio_digest:' + name, lambda n=name,d=digest: p.check_pin(trio/n,d))
    for name, key in [('partner-raw.npz','partner_raw_sha256'), ('a172-submission.csv','submission_sha256')]:
        if key in receipt: telemetry(checks, name, lambda n=name,k=key: p.check_pin(directory/n,receipt[k]))
    checks['receipt_complete'] = receipt.get('status') == 'PASS_BUNDLE'
    checks['receipt_cohort'] = receipt.get('studies') == len(ids)
    def replay():
        tids, tv = v5q._read_csv(trio/'submission.csv')
        saved = directory/'a172-submission.csv'
        if not saved.exists(): saved = directory/'submission.csv'
        sids, sv = v5q._read_csv(saved)
        need(set(tids) == set(sids) == set(ids), 'replay cohorts')
        expected = a172.final_values(tv[[tids.index(uid) for uid in ids]], q, 'A172')
        return bool(np.array_equal(sv[[sids.index(uid) for uid in ids]].astype(np.float32), expected.astype(np.float32)))
    telemetry(checks, 'saved_replay', replay)
    evidence = dict(studies=len(ids), executed_arms=['EXP155','EXP143v2','Q_A5q'],
                    failures=failures, prediction_fallbacks=len(set().union(*[set(x) for x in failures.values()])),
                    telemetry=checks, child=optional_json(directory/'child-exit.json'),
                    q_timings=optional_json(directory/'q-timings.json'),
                    phases={name: optional_json(directory/(name+'-exit.json')) for name in ('own','Q_A5q')})
    return ids, values['EXP143v2'], values['EXP155'], q, evidence


def authenticated_p956(directory):
    import csv
    import numpy as np
    directory = Path(directory)
    receipt = optional_json(directory/'p956-receipt.json')
    provenance = receipt.get('provenance', {})
    if provenance:
        need(provenance.get('source_pins') == p.PINS, 'P956 serving identity')
        if receipt.get('version') == '1.0.1': need(provenance.get('runtime_sha256') == P956_SHA, 'P956 runtime identity')
    work = directory/'p956-run'
    checks = {}
    rr = optional_json(work/'r/receipt.json')
    controller = optional_json(work/'s6/controller/submission_receipt.json')
    if (directory/'ids.json').exists(): ids = v5q.validate_ids(read_json(directory/'ids.json'))
    elif (work/'r/raw.npz').exists():
        with np.load(work/'r/raw.npz', allow_pickle=False) as raw: ids = v5q.validate_ids(raw['ids'].tolist())
    elif (work/'s6/predictions.csv').exists(): ids, _ = v5q._read_csv(work/'s6/predictions.csv')
    else: ids = []
    r, ours = np.full((len(ids),12), .5), np.full((len(ids),12), .5)
    rf, sf = dict.fromkeys(ids, 'MissingPrediction: R child aborted'), dict.fromkeys(ids, 'MissingPrediction: S6 child aborted')
    if (work/'r/journal-identity.json').exists():
        vals, fails = read_journal(work/'r', ids, ['R'], {'checkpoint_sha256': p.R_SHA, 'source_pins': p.PINS})
        r, rf = vals['R'], fails['R']
    elif (work/'r/raw.npz').exists():
        need(provenance.get('source_pins') == p.PINS, 'R serving identity')
        with np.load(work/'r/raw.npz', allow_pickle=False) as raw:
            need(set(raw.files) == {'ids','targets','public'} and raw['targets'].tolist() == list(p.LABELS), 'R target/schema mismatch')
            rids = v5q.validate_ids(raw['ids'].tolist())
            need(set(ids) == set(rids), 'R study mismatch')
            r, rf = clean_member(ids, raw['public'][[rids.index(uid) for uid in ids]], rr.get('failures',{}))
    rf.update({uid: 'FailedStudySentinel: public cell 7' for uid, failed in zip(ids, np.all(r==.5,axis=1)) if failed})
    prov_path = work/'s6/provenance.json'
    workers = sorted((work/'s6/controller').glob('worker*'))
    if prov_path.exists() or workers:
        prov = read_json(prov_path)
        for key, expected in [('bundle_id',s6.BUNDLE_ID), ('manifest_sha256',s6.MANIFEST_SHA256),
                              ('sha256',s6.HASHES), ('source_sha256',s6.SOURCE_HASHES), ('anchor_weights_sha256',s6.ANATOMY_SHA256)]:
            need(prov.get(key) == expected, 'S6 identity: ' + key)
        assigned, scored = set(), set()
        for worker in workers:
            if not worker.is_dir(): continue
            if not (worker/'ids.json').exists():
                checks[worker.name] = 'child stopped before assignment was written'
                continue
            part = read_json(worker/'ids.json')
            need(len(set(part)) == len(part) and not assigned.intersection(part) and set(part)<=set(ids), 'S6 assignment cohort')
            assigned.update(part)
            wr = optional_json(worker/'receipt.json')
            sf.update({uid: why for uid,why in wr.get('fallback',{}).items() if uid in ids})
            prediction = worker/'predictions.csv'
            if not prediction.exists(): continue
            with prediction.open(encoding='utf-8',newline='') as stream:
                reader = csv.DictReader(line for line in stream if line.endswith('\n'))
                need(reader.fieldnames == [p.UID,*p.LABELS], 'S6 prediction schema')
                for row in reader:
                    uid = row[p.UID]
                    need(uid in part and uid not in scored, 'S6 prediction UID differs')
                    try: value = p.RBinding.prediction([float(row[t]) for t in p.LABELS])
                    except (ValueError, TypeError) as exc:
                        sf[uid] = str(exc)
                        continue
                    ours[ids.index(uid)] = value
                    if uid not in wr.get('fallback', {}): sf.pop(uid,None)
                    scored.add(uid)
        checks['assigned_cohort'] = assigned == set(ids)
    telemetry(checks, 'controller_crosscheck', lambda: p.s6_failure_evidence(work/'s6/controller',ids,controller))
    anchors = dict(scored=len(ids)-len(sf), model_anchor=0, fallback=0, methods={})
    try: anchors = p.anatomy_anchors(controller)
    except (ValueError, KeyError, TypeError) as exc: checks['anatomy'] = str(exc)
    for name,key in [('r/raw.npz','r_raw_sha256'),('s6/predictions.csv','s6_raw_sha256'),('s6/provenance.json','s6_provenance_sha256')]:
        if key in provenance: telemetry(checks,name,lambda n=name,k=key:p.check_pin(work/n,provenance[k]))
    if 'submission_sha256' in receipt: telemetry(checks,'submission_digest',lambda:p.check_pin(directory/'submission.csv',receipt['submission_sha256']))
    telemetry(checks,'saved_replay',lambda:(directory/'submission.csv').read_bytes()==p.csv_bytes(ids,p.public_formula(r,ours,[uid in sf for uid in ids])))
    checks.update(receipt_cohort=receipt.get('studies')==len(ids), r_fallback_count=rr.get('failure_count',0)==len(rf))
    return ids, r, ours, dict(studies=len(ids),executed_arms=['R','S6'],source_version=receipt.get('version','C85 recovery'),
        worker_exit_codes=controller.get('worker_exit_codes',{}), r_prediction_fallbacks=len(rf),
        s6_prediction_fallbacks=len(sf),s6_anatomy_anchors=anchors,failures={'R':rf,'S6':sf},telemetry=checks,
        child=optional_json(directory/'child-exit.json'),
        phases={name: optional_json(directory/(name+'-exit.json')) for name in ('R','S6')})


def candidate_receipt(ids, a, b, payload):
    return dict(status='COMPLETE_C85', version=VERSION, formula=FORMULA,
        rank='average / successful rows, float64 nested ranks; final CSV float32',
        failure_policy='renormalise own members per study; surviving branch rank; both branches failed 0.5',
        intended_members=MEMBERS, executed_members=MEMBERS,
        execution_passes=[dict(serving='A172/v5q own-only',arms=['EXP155','EXP143v2','Q_A5q']),dict(serving='P956 recovery',arms=['R','S6'])],
        studies=len(ids), findings=12, fallbacks=dict(a172_prediction=a['prediction_fallbacks'],r_prediction=b['r_prediction_fallbacks'],s6_prediction=b['s6_prediction_fallbacks']),
        s6_anatomy_anchors=b['s6_anatomy_anchors'], a172=a, p956=b,
        submission_sha256=hashlib.sha256(payload).hexdigest(),qualified_for_submission=False)


def publish(staged, final, payload):
    """Same-filesystem atomic publication; staged bytes must already be verified."""
    digest = hashlib.sha256(payload).hexdigest()
    p.check_pin(staged, digest)
    need(not final.exists(), "immutable artifact exists: " + str(final))
    os.replace(staged, final)
    p.check_pin(final, digest)


def failed_execution(output, error, completed, a172_dir, p956_dir):
    """Only called after this invocation claimed a fresh output, never on stale runs."""
    # An exception may occur after rename takes effect or during receipt publication.
    # Remove only this invocation's uncommitted root files; retain staged/child evidence.
    for name in ("submission.csv", "diagnostic-c85.csv", "c85-receipt.json"):
        (output / name).unlink(missing_ok=True)
    children = {}
    for label, path in (("A172", a172_dir / "exp169-executed-candidate.json"),
                        ("P956", p956_dir / "p956-receipt.json"),
                        ("R", p956_dir / "p956-run/r/receipt.json"),
                        ("S6", p956_dir / "p956-run/s6/controller/submission_receipt.json")):
        if path.is_file():
            children[label] = dict(path=str(path), sha256=sha(path))
    payload = json_bytes(dict(status="FAILED_EXECUTION", version=VERSION,
        intended_members=MEMBERS, authenticated_completed_components=completed,
        child_receipts=children, error=f"{type(error).__name__}: {error}",
        submission_sha256=None, qualified_for_submission=False))
    staged = output / ".c85-publication/failed-receipt.json"
    write_once(staged, payload)
    publish(staged, output / "c85-receipt.json", payload)


def collect(a172_dir, p956_dir, output, *, diagnostic=False):
    import numpy as np
    record = output / "c85-receipt.json"
    stage = output / ".c85-publication"
    repeat = diagnostic and record.is_file() and (output / "diagnostic-c85.csv").is_file()
    # Stale/immutable records are rejected outside the failure handler. A diagnostic
    # repeat is read-only and must match both original artifacts exactly.
    need(not (output / "submission.csv").exists(), "stale C85 submission")
    need(repeat or not any(path.exists() for path in
         (record, stage, output / "diagnostic-c85.csv")), "stale C85 output")
    if not repeat:
        stage.mkdir(parents=True)  # exclusive claim; never reused after failure
    completed = []
    try:
        ids, exp143, exp155, q, a = authenticated_a172(a172_dir)
        completed.append("A172")
        pids, r, ours, b = authenticated_p956(p956_dir)
        completed.append("P956")
        if not ids:
            ids = pids
            exp143, exp155, q = [np.full((len(ids),12), .5) for _ in range(3)]
            a['failures'] = {name: dict.fromkeys(ids, 'A172 child aborted') for name in MEMBERS[2:]}
            a['prediction_fallbacks'] = len(ids)
        if not pids:
            pids = ids
            r, ours = [np.full((len(ids),12), .5) for _ in range(2)]
            b['failures'] = {name: dict.fromkeys(ids, 'P956 child aborted') for name in MEMBERS[:2]}
            b['r_prediction_fallbacks'] = b['s6_prediction_fallbacks'] = len(ids)
        need(set(ids) == set(pids), 'C85 cohort mismatch')
        order = [pids.index(uid) for uid in ids]
        r, ours = r[order], ours[order]
        values = blend(ids, r, ours, exp143, exp155, q, {**a['failures'], **b['failures']})
        expected = reference(r, ours, exp143, exp155, q)
        payload = p.csv_bytes(ids, values)
        receipt = candidate_receipt(ids, a, b, payload)
        receipt['telemetry'] = dict(healthy_reference_exact=bool(np.array_equal(values, expected)),
                                    uid_order_aligned=ids != pids, csv_float32_roundtrip=True)
        if diagnostic and receipt["status"] == "COMPLETE_C85":
            receipt.update(status="DIAGNOSTIC_C85", submission_sha256=None)
        receipt_payload = json_bytes(receipt)
        if repeat:
            need(record.read_bytes() == receipt_payload and
                 (output / "diagnostic-c85.csv").read_bytes() == payload,
                 "immutable diagnostic replay differs")
            return receipt, payload
        if diagnostic or receipt["status"] == "COMPLETE_C85":
            name = "diagnostic-c85.csv" if diagnostic else "submission.csv"
            staged = stage / name
            write_once(staged, payload)
            saved_ids, saved = v5q._read_csv(staged)
            v5q.validate_ids(ids, saved_ids)
            v5q._probabilities('final CSV', saved, len(ids))
            receipt['telemetry']['csv_float32_roundtrip'] = bool(np.array_equal(saved.astype(np.float32), values.astype(np.float32)))
            publish(staged, output / name, payload)
        # Success is materialized only after final CSV publication and verification.
        receipt_payload = json_bytes(receipt)
        staged = stage / "c85-receipt.json"
        write_once(staged, receipt_payload)
        publish(staged, record, receipt_payload)
        return receipt, payload
    except BaseException as error:
        if not repeat:
            failed_execution(output, error, completed, a172_dir, p956_dir)
        raise


class Journal:
    """Append each prediction before another model/study can fail."""
    def __init__(self, directory, ids, identity):
        import threading
        self.directory, self.lock = Path(directory), threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        write_once(self.directory/'ids.json', json_bytes(ids))
        write_once(self.directory/'journal-identity.json', json_bytes(identity))

    def record(self, uid, member, values=None, error=None, seconds=None):
        import numpy as np
        row = dict(uid=uid, member=member)
        if error is None:
            row['values'] = np.asarray(p.RBinding.prediction(values)).tolist()
        else: row['error'] = str(error)
        if seconds is not None: row['seconds'] = seconds
        with self.lock, (self.directory/'predictions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        return values


class ServingBinding(p.RBinding):
    def __init__(self, helper, work, ids, identity, members):
        super().__init__(helper, work)
        self.journal, self.members = Journal(work, ids, identity), members

    def infer(self, uid, arm, infer, *args):
        import time
        start = time.monotonic()
        try:
            value = self.prediction(infer(*args))
            return self.journal.record(uid, self.members[arm], value, seconds=time.monotonic()-start)
        except Exception as exc:
            import numpy as np
            self.journal.record(uid, self.members[arm], error=f'{type(exc).__name__}: {exc}', seconds=time.monotonic()-start)
            return np.full(12, .5, dtype=np.float32)

    def capture_failures(self, ids, errors):
        super().capture_failures(ids, errors)
        for uid, why in self.failures.items():
            self.study_error(uid, why)

    def study_error(self, uid, why):
        for member in self.members: self.journal.record(uid, member, error=why)

    def save_raw(self, *args): pass  # the append journal is the authoritative raw output
    def output_values(self, arms, *args): return arms[0]
    def finish_receipt(self, work, seconds):
        write_once(self.work/'receipt.json', json_bytes(dict(status='RECORDED', seconds=seconds, members=self.members,
                                                             failures=self.failures)))


def serving_source(source):
    source = p.r_runtime_source(source)
    source = source.replace('arm_probs[a][i] = _E152.prediction(infer_pair_probs(model, xw, xm, d))',
                            'arm_probs[a][i] = _E152.infer(sid, a, infer_pair_probs, model, xw, xm, d)')
    source = source.replace('                _exp152_errors.append((i, type(e).__name__, str(e)))',
        '                _exp152_errors.append((i, type(e).__name__, str(e)))\n'
        '                _E152.study_error(sid, f"{type(e).__name__}: {e}")')
    # Preserve already-loaded arms if one model cannot initialize.
    source = source.replace('            model, res = load_model(find_weight_file(arm["file"]), arm["arch"], arm["res"], d)',
        '            try:\n                model, res = load_model(find_weight_file(arm["file"]), arm["arch"], arm["res"], d)\n'
        '            except Exception as exc:\n                model, res = None, arm["res"]\n'
        '                print("MODEL_LOAD_EXCEPTION", arm["file"], repr(exc), flush=True)')
    compile(source, '<c85-serving>', 'exec')
    return source


def bind_own(namespace, helper, work, competition, identity):
    ids, _ = v5q._read_csv(competition/'sample_submission.csv')
    namespace['ARMS'] = [arm for arm in namespace['ARMS'] if arm['file'] != p.R_NAME]
    names = ['EXP155' if 'exp155-' in arm['file'] else 'EXP143v2' for arm in namespace['ARMS']]
    need(names == ['EXP155','EXP143v2'], 'own arm order')
    # EXP155.bind installs its adapted private EXP152 module here. The top-level
    # EXP155 module does not export input_record; preserve the frozen serving ABI.
    namespace.update(_E152=ServingBinding(namespace['_E152'], work, ids, identity, names),
                     find_test_root=lambda: str(competition))


def q_runtime_source(source):
    """Keep frozen window/model math; isolate decode/inference exceptions by study."""
    def wrap_block(source, start, end, indent, handler, after=0):
        a = source.index(start, after)
        b = source.index(end, a)
        body = source[a:b]
        return source[:a] + ' '*indent+'try:\n' + ''.join('    '+line if line.strip() else line for line in body.splitlines(True)) + handler + source[b:]
    source = wrap_block(source, '                started = time.monotonic()', '        except BaseException as exc:', 16,
        '                except Exception as exc:\n                    journal.record(uid, "0", error=f"decode: {type(exc).__name__}: {exc}")\n')
    source = source.replace('                    scores[family][row, i] = value',
        '                    scores[family][row, i] = value\n                    journal.record(ids[i], str(m), value)')
    source = wrap_block(source, '                started = time.monotonic()', '                work_queue.task_done()\n                active = False', 16,
        '                except Exception as exc:\n                    journal.record(ids[i], str(m) if "m" in locals() else "0", error=f"inference: {type(exc).__name__}: {exc}")\n', source.index('    def consumer('))
    a, b = source.index("    need(not errors,"), source.index("    timings['ram']")
    source = source[:a] + "    timings['exceptions'] = errors\n" + source[b:]
    source = source.replace("float(np.percentile(timings.pop(name), 95))", "float(np.percentile(timings[name], 95)) if timings[name] else None")
    compile(source, '<c85-Q>', 'exec')
    return source


def run_q(module, partner, competition, output):
    import inspect
    import numpy as np
    module.reset_backend()
    m = manifest(partner/'release-manifest.json')
    ids = read_json(output/'ids.json')
    journal = Journal(output/'q-journal', ids, {'manifest_sha256': MANIFEST_SHA})
    source = q_runtime_source(v5q.a172_function_source('partner_scores', inspect.getsource(module.base.partner_scores)))
    scope = dict(module.base.__dict__, _load_models=module._load_models, journal=journal)
    exec(compile(source, '<c85-Q>', 'exec'), scope)
    scores, timings = scope['partner_scores'](partner, competition, ids, m)
    np.savez(output/'partner-raw.npz', ids=np.asarray(ids), **scores)
    write_once(output/'q-timings.json', json_bytes(timings))


def a172_source(cell, output):
    """Authenticated extraction/binding, own arms only, same Q windows and models."""
    import textwrap
    source = v5q.a172_runtime_source(cell)
    directory = output.as_posix()
    need("'" not in directory and "\n" not in directory and "\\" not in directory, 'unsafe output path')
    # Reuse both authenticated asset setup blocks, without the trio-dependent pilot/serve veto.
    own = textwrap.dedent(source[source.index('    # No pip,'):source.index('    main()')])
    own = own.replace("exec(compile((_assets/'public-serving.py').read_text(),", "exec(compile(_c85.serving_source((_assets/'public-serving.py').read_text()),")
    own += "_c85.bind_own(globals(), _e152, _work, _comp, _c85.manifest(_out/'q-a5q-manifest.json')['anchor_binding'])\n"
    own += "_c85.phase(_out, 'own', main)\n"
    partner = textwrap.dedent(source[source.index('        import os as _os'):source.index('        import torch as _torch')])
    # install_a172_runtime is unnecessary: run_q applies the same unlimited function adapter.
    partner = partner[:partner.index('import exp169_release_v5q as _v5q')]
    source = "from pathlib import Path\nimport os, sys\nimport exp169_release_c85 as _c85\n"
    source += "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
    source += f"_out = Path({directory!r})\n_comp = _c85.competition_root()\n"
    source += (own + partner).replace('/kaggle/working', directory)
    source += "_c85.phase(_out, 'Q_A5q', lambda: _c85.run_q(_e169, _partner, _comp, _out))\n"
    compile(source, '<c85-a172>', 'exec')
    return source


def competition_root():
    return Path(os.environ['C85_COMPETITION']) if os.environ.get('C85_COMPETITION') else v5q._mount(p.COMPETITION, 'test.csv')


def phase(output, name, action):
    import time
    started = time.monotonic()
    record = dict(phase=name, exit_code=0)
    try: action()
    except Exception as exc:
        record.update(exit_code=1, error=f'{type(exc).__name__}: {exc}')
    record['seconds'] = time.monotonic()-started
    write_once(output/(name+'-exit.json'), json_bytes(record))
    print('C85_PHASE', json.dumps(record), flush=True)
    return record


def preflight(runtime, output):
    """Identity checks happen before operational child errors become recoverable."""
    comp = competition_root()
    ids, _ = v5q._read_csv(comp/'sample_submission.csv')
    import csv
    with (comp/'test.csv').open(newline='',encoding='utf-8') as stream:
        test_ids = v5q.validate_ids([row[p.UID] for row in csv.DictReader(stream)])
    need(set(ids) == set(test_ids), 'test/sample cohort')
    p.check_pin(v5q._mount(p.DATASETS[0], p.R_NAME)/p.R_NAME, p.R_SHA)
    for slug, filename, digest in [(DATASETS[0], 'exp169-assets.zip.bin', v5q.A172_ARCHIVE_SHA),
        (DATASETS[1], 'exp155-assets.zip.bin', '40b8d02847bf36938d82a79e88cb1dad4296db0cc95691b8cd0d2a2ce1a7e5fa')]:
        p.check_pin(v5q._mount(slug, filename)/filename, digest)
    s6.verify_bundle(v5q._mount(s6.DATASET_SLUG, 'manifest.json'))
    s6.verify_source(runtime/'exp189_s6_src')
    # Keep the verified extracted code for the public child; no redundant extraction.
    code, anatomy = s6.prepare_code(v5q._mount(v5q.CODE_SLUG, 'anatomy.pt'), output/'c85-run/p956/p956-run/s6-code', install_wheels=False)
    write_once(output/'c85-run/p956/p956-run/code-paths.json', json_bytes(dict(code=str(code), anatomy=str(anatomy))))
    return ids


def run_r(runtime, weights, competition, output):
    import importlib.util
    import types
    p.runtime_pins(runtime)
    p.check_pin(weights, p.R_SHA)
    ids, _ = v5q._read_csv(competition/'sample_submission.csv')
    spec = importlib.util.spec_from_file_location('p956_r_helper', runtime/'exp152_own0949_release.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    module = types.ModuleType('c85_r_serving')
    module.__file__ = str(runtime/'public-serving.py')
    sys.modules[module.__name__] = module
    exec(compile(serving_source((runtime/'public-serving.py').read_text(encoding='utf-8')), module.__file__, 'exec'), module.__dict__)
    p.bind_r(module.__dict__, helper, weights, competition, output)
    module._E152 = ServingBinding(helper, output, ids, {'checkpoint_sha256':p.R_SHA,'source_pins':p.PINS}, ['R'])
    module.main()


def run_public(runtime, output):
    runtime_pins(runtime)
    comp = competition_root()
    ids = read_json(output/'ids.json')
    work = output/'p956-run'
    work.mkdir(exist_ok=True)
    weights = v5q._mount(p.DATASETS[0], p.R_NAME)/p.R_NAME
    p.check_pin(weights, p.R_SHA)
    bundle = v5q._mount(s6.DATASET_SLUG, 'manifest.json')
    s6.verify_bundle(bundle)
    code, anatomy = work/'s6-code/code', work/'s6-code/anatomy.pt'
    # prepare_code returns the actual extracted code path; persist it at preflight.
    paths = read_json(work/'code-paths.json')
    code, anatomy = Path(paths['code']), Path(paths['anatomy'])
    env = dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE='1')
    phase(output, 'R', lambda: v5q.run_process_group([sys.executable,'-u',str(runtime/'exp169_release_c85.py'),
        'r-worker','--runtime',str(runtime),'--weights',str(weights),'--competition',str(comp),'--output',str(work/'r')], label='C85 R', env=env))
    def infer_s6():
        s6.install_wheels(sorted((work/'s6-code/wheels').glob('*.whl')), code)
        s6.infer_studies([comp/'test_series'/uid for uid in ids], bundle_dir=bundle, code_dir=code,
            anatomy_weights=anatomy, output_dir=work/'s6', workers=2, amp='auto', device='cuda',
            max_fallback_fraction=1.0, source_dir=runtime/'exp189_s6_src')
    phase(output, 'S6', infer_s6)


def runtime_pins(runtime):
    p.runtime_pins(runtime)
    for name, digest in (("exp169_release_p956.py", P956_SHA),
                         ("exp169_release_v2.py", v5q.A172_SOURCE_SHA),
                         ("exp169_partner_bundle_release.py", a172.BASE_SHA),
                         ("q-a5q-manifest.json", MANIFEST_SHA), ("a172-cell.py.txt", v5q.A172_CELL_SHA)):
        p.check_pin(runtime / name, digest)


def run_release(runtime, output):
    """Target-only orchestration; CPU tests replace process execution boundaries."""
    runtime_pins(runtime)
    need(not any((output / x).exists() for x in
         ("submission.csv", "c85-run", "c85-receipt.json", ".c85-publication", "diagnostic-c85.csv")),
         "stale C85 output")
    work = output / "c85-run"
    work.mkdir(parents=True)
    adir, pdir = work / "a172", work / "p956"
    adir.mkdir(); pdir.mkdir()
    env = dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    ids = preflight(runtime, output)
    for directory in (adir, pdir): write_once(directory/'ids.json', json_bytes(ids))
    write_once(adir/'q-a5q-manifest.json', (runtime/'q-a5q-manifest.json').read_bytes())
    script = work/'a172.py'
    write_once(script, a172_source((runtime/'a172-cell.py.txt').read_text(encoding='utf-8'), adir).encode())
    phase(adir, 'child', lambda: v5q.run_process_group([sys.executable,'-u',str(script)], label='C85 frozen A172',env=env))
    phase(pdir, 'child', lambda: v5q.run_process_group([sys.executable,'-u',str(runtime/'exp169_release_c85.py'),
        'public-worker','--runtime',str(runtime),'--output',str(pdir)], label='C85 unchanged P956',env=env))
    receipt, _ = collect(adir,pdir,output)
    return receipt


def source_payloads():
    files = {name: path.read_bytes() for name, path in p.source_files().items()}
    p.check_pin(Path(p.__file__), P956_SHA)
    p.check_pin(v5q.STAGE / "kernel/release.ipynb", v5q.A172_NOTEBOOK_SHA)
    cell = "".join(read_json(v5q.STAGE / "kernel/release.ipynb")["cells"][4]["source"])
    need(hashlib.sha256(cell.encode()).hexdigest() == v5q.A172_CELL_SHA, "A172 cell drift")
    for name, digest in (("exp169_release_v2.py", v5q.A172_SOURCE_SHA),
                         ("exp169_partner_bundle_release.py", a172.BASE_SHA)):
        path = ROOT / "tools" / name
        p.check_pin(path, digest)
        files[name] = path.read_bytes()
    mpath = A5Q / "exp169-partner/release-manifest.json"
    manifest(mpath)
    files.update({"exp169_release_c85.py": Path(__file__).read_bytes(),
                  "exp169_c85_stress.py": (ROOT/'tools/exp169_c85_stress.py').read_bytes(),
                  "a172-cell.py.txt": cell.encode(), "q-a5q-manifest.json": mpath.read_bytes()})
    return files


def package():
    files = source_payloads()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(info, data)
    return payload.getvalue(), {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}


def notebook_payload(blob, pins):
    digest = hashlib.sha256(blob).hexdigest()
    bootstrap = "\n".join([
        "import base64, hashlib, io, sys, zipfile", "from pathlib import Path",
        f"BLOB = {base64.b64encode(blob).decode()!r}", f"PINS = {pins!r}",
        "blob = base64.b64decode(BLOB)", f"assert hashlib.sha256(blob).hexdigest() == {digest!r}",
        "runtime = Path('/kaggle/working/c85-runtime')", "assert not runtime.exists(), 'stale runtime'",
        "with zipfile.ZipFile(io.BytesIO(blob)) as archive:",
        "    assert len(archive.namelist()) == len(PINS) and set(archive.namelist()) == set(PINS)",
        "    for name in archive.namelist():",
        "        assert not name.startswith(('/', '\\\\')) and '\\\\' not in name and ':' not in name and '..' not in Path(name).parts",
        "        assert hashlib.sha256(archive.read(name)).hexdigest() == PINS[name]",
        "    runtime.mkdir()", "    archive.extractall(runtime)", "sys.path.insert(0, str(runtime))",
        "import exp169_release_c85 as release",
        "release.run_release(runtime, Path('/kaggle/working'))", ""])
    compile(bootstrap, "<c85-bootstrap>", "exec")
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
        display_name="Python 3", language="python", name="python3")), cells=[
        dict(cell_type="markdown", id="c85-policy", metadata={}, source=[
            f"# C85 v{VERSION}\n" + FORMULA + "\nFrozen A5q bundle, P956 v1.0.1 R/S6. "
            "Credits: nartaa public R and heliosli S6; original licenses/provenance retained in assets. "
            "Per-study survivor substitution; child exits, anatomy and replay checks are telemetry. "
            "Auxiliary candidate; public veto <= own P956 -0.002. "
            "No weight search, no timeout; hidden runtime unknown. CPU-prepared, not qualified.\n"]),
        dict(cell_type="code", id="c85-run", metadata={}, execution_count=None, outputs=[], source=bootstrap.splitlines(True))])


def metadata():
    return dict(id=KERNEL_ID, title=KERNEL_ID.split("/")[1], code_file="release.ipynb",
        language="python", kernel_type="notebook", is_private=True, enable_gpu=True, enable_tpu=False,
        enable_internet=False, machine_shape=p.MACHINE, docker_image=p.IMAGE,
        dataset_sources=DATASETS, competition_sources=[p.COMPETITION], kernel_sources=[], model_sources=[])


def stress_notebook(blob, pins):
    notebook = notebook_payload(blob, pins)
    notebook['cells'][0]['source'] = [f'# C85 v{VERSION} N32 diagnostic only; never submit\n'
        'Full release path on 32 training studies selected by max/min series and slice counts. '
        'No labels. Phase exit files and prediction journals retain timings and exceptions.\n']
    code = ''.join(notebook['cells'][1]['source'])
    code = code.replace("release.run_release(runtime, Path('/kaggle/working'))", "\n".join([
        'import os, time', 'import exp169_c85_stress as stress',
        "competition = release.v5q._mount(release.p.COMPETITION, 'train_series.csv')",
        "data = Path('/kaggle/working/c85-stress-data')",
        'selected = stress.prepare(competition, data, n=32)',
        "os.environ['C85_COMPETITION'] = str(data)",
        "print('C85_STRESS_SELECTION', len(selected), flush=True)",
        'started = time.monotonic()',
        "release.run_release(runtime, Path('/kaggle/working'))",
        "print('C85_STRESS_TOTAL_SECONDS', time.monotonic()-started, flush=True)"]))
    compile(code, '<c85-stress>', 'exec')
    notebook['cells'][1]['source'] = code.splitlines(True)
    return notebook


def stress_metadata():
    value = metadata()
    value.update(id=KERNEL_ID+'-stress-n32', title=KERNEL_ID.split('/')[1]+'-stress-n32')
    return value


def prepare(out=OUT):
    blob, pins = package()
    write_once(out / "kernel/release.ipynb", json_bytes(notebook_payload(blob, pins)))
    write_once(out / "kernel/kernel-metadata.json", json_bytes(metadata()))
    write_once(out / "stress-kernel/release.ipynb", json_bytes(stress_notebook(blob, pins)))
    write_once(out / "stress-kernel/kernel-metadata.json", json_bytes(stress_metadata()))
    record = dict(status="PREPARED_LOCAL_ONLY", version=VERSION, formula=FORMULA,
        kernel_id=KERNEL_ID, files=pins, archive_sha256=hashlib.sha256(blob).hexdigest(),
        notebook_sha256=sha(out / "kernel/release.ipynb"), metadata_sha256=sha(out / "kernel/kernel-metadata.json"),
        qualification="PENDING independent review, fresh parity/visible run, representative timing")
    write_once(out / "build-receipt.json", json_bytes(record))
    m = manifest(A5Q / "exp169-partner/release-manifest.json")
    write_once(out / "input-identities.json", json_bytes(dict(q_manifest_sha256=MANIFEST_SHA,
        q_checkpoints=m["entries"], anchor_binding=m["anchor_binding"],
        s6=dict(bundle_id=s6.BUNDLE_ID, manifest_sha256=s6.MANIFEST_SHA256, checkpoint_sha256=s6.HASHES,
                source_sha256=s6.SOURCE_HASHES, anatomy_sha256=s6.ANATOMY_SHA256),
        a172_archive_sha256=v5q.A172_ARCHIVE_SHA, a172_notebook_sha256=v5q.A172_NOTEBOOK_SHA,
        a172_cell_sha256=v5q.A172_CELL_SHA, p956_source_sha256=P956_SHA,
        source_pins=pins, dataset_sources=DATASETS)))
    return record


def validate(out=OUT):
    b = read_json(out / "build-receipt.json")
    blob, pins = package()
    need(b["files"] == pins and b["archive_sha256"] == hashlib.sha256(blob).hexdigest(), "source archive drift")
    for name, data, key in (("kernel/release.ipynb", json_bytes(notebook_payload(blob, pins)), "notebook_sha256"),
                            ("kernel/kernel-metadata.json", json_bytes(metadata()), "metadata_sha256")):
        p.check_pin(out / name, b[key])
        need((out / name).read_bytes() == data, "stage differs: " + name)
    for name, data in [('release.ipynb', stress_notebook(blob,pins)), ('kernel-metadata.json', stress_metadata())]:
        need((out/'stress-kernel'/name).read_bytes() == json_bytes(data), 'stress stage differs: '+name)
    return dict(status="PASS_CPU_VALIDATE", files=len(pins), notebook_sha256=b["notebook_sha256"],
                archive_sha256=b["archive_sha256"], qualified_for_submission=False)


def register(out=OUT):
    record = dict(**validate(out), version=VERSION, formula=FORMULA, kernel_id=KERNEL_ID,
                  public_veto="C85 <= own P956 -0.002", role="auxiliary; not automatic Final B")
    record["status"] = "REGISTERED_LOCAL_ONLY"
    write_once(out / "registration.json", json_bytes(record))
    return record


def cpu_replay(output):
    p.check_pin(A5Q / "exp169-executed-candidate.json", A5Q_RECEIPT_SHA)
    p.check_pin(p.REPLAY / "p956-receipt.json", P956_REPLAY_SHA)
    receipt, payload = collect(A5Q, p.REPLAY, output, diagnostic=True)
    _, repeated = collect(A5Q, p.REPLAY, output, diagnostic=True)
    need(payload == repeated, "non-deterministic replay")
    result = dict(status="PASS_CPU_ARITHMETIC_REPLAY", candidate_status=receipt["status"],
        studies=receipt["studies"], findings=12, independent_formula_exact=True, deterministic_csv=True,
        diagnostic_sha256=hashlib.sha256(payload).hexdigest(), fallbacks=receipt["fallbacks"],
        qualified_for_submission=False, scope="recorded real outputs; no fresh inference or hidden-runtime qualification")
    write_once(output / "replay.json", json_bytes(result))
    return result


def admit(out=OUT, replay_dir=None):
    validation = validate(out)
    reg = read_json(out / "registration.json")
    need(reg["notebook_sha256"] == validation["notebook_sha256"], "registration drift")
    replay_dir = replay_dir or out / "cpu-replay"
    replay = read_json(replay_dir / "replay.json")
    need(replay["status"] == "PASS_CPU_ARITHMETIC_REPLAY" and replay["independent_formula_exact"]
         and replay["deterministic_csv"], "CPU arithmetic replay required")
    p.check_pin(replay_dir / "diagnostic-c85.csv", replay["diagnostic_sha256"])
    return dict(status="HOLD_NOT_QUALIFIED", cpu_preparation="PASS", qualified_for_submission=False,
        candidate_replay_status=replay["candidate_status"],
        remaining=["independent review", "fresh visible run and serving parity",
                   "representative timing demonstrating <=8h target; hidden runtime unknown", "B gain/duplicate review"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "register", "admit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--out", type=Path, default=OUT)
        if name == "admit":
            cmd.add_argument("--dry-run", required=True, action="store_true")
            cmd.add_argument("--replay-dir", type=Path)
    cmd = sub.add_parser("cpu-replay")
    cmd.add_argument("--output", type=Path, default=OUT / "cpu-replay")
    cmd = sub.add_parser("r-worker")
    for name in ("runtime", "weights", "competition", "output"):
        cmd.add_argument("--"+name, type=Path, required=True)
    cmd = sub.add_parser("public-worker")
    cmd.add_argument("--runtime", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd = sub.add_parser("run")
    cmd.add_argument("--runtime", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "r-worker": result = run_r(args.runtime,args.weights,args.competition,args.output)
    elif args.command == "public-worker": result = run_public(args.runtime,args.output)
    elif args.command == "admit": result = admit(args.out,args.replay_dir)
    elif args.command == "run": result = run_release(args.runtime, args.output)
    elif args.command == "cpu-replay": result = cpu_replay(args.output)
    else: result = globals()[args.command](args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
