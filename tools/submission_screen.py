"""H-B-26 submission screen: a CPU-only report generator over EXISTING artifacts.

For one candidate (a spec naming its members' OOF CSVs on the 4,343 G1 studies, its
release Public3 output directory and its pre-registered decision) it writes
artifacts/experiments/SUBMISSION_SCREEN/<candidate>/screen.json + screen.md:

1. Leakage lineage: for every member, each fold model's training UIDs (protocol ordered-ids,
   bound to the fold run's provenance hashes) against the validation UIDs of its OOF, plus
   the influence of teachers/ancestors (label-set teacher receipts, spec-named ancestors):
   a distilled student's validation studies that trained a teacher whose outputs became the
   student's targets are flagged (D-A-002, D-B-024). Gold58 is reported only as absent or
   present in training; it is never a criterion (plan H-B-26, D-A-049).
2. OOF: pooled and per-fold macro and per-target AUC against a named baseline (default
   CoAtNet5 Dread K36) with a 500-draw whole-component bootstrap (VALIDATION_GROUP_CONTRACT_V1,
   seed 20261003), ALWAYS against all three independent report-derived references:
   rint(Dread), flight_v4 (>= 0.5, Gold rows excluded) and relabel_v1 (>= 0.5), plus a
   "direction consistency" line (Dread CI above 0 and not worse than -0.002 on flight and
   relabel_v1). All three share the report-text ceiling: agreement guards against one
   labeller's quirks, not against report-vs-image gaps (only Gold58/public see those).
   Weak-label development evidence; it does not establish public transfer (D-B-016 proxy,
   D-B-020 as corrected, D-A-049).
3. Replication: declared or auto-detected (seed42_protocol_sha256 link) seed/split replicates.
4. Engineering: Public3 receipt status and sha (PASS_TEST_COHORT_ONLY ...), CSV validity via
   tools/anchor_receipt_policy.validate_csv, served checkpoints bound to the OOF fold runs,
   reduction recipe and runtime projection. Engineering path only, no score identity (D-C-004).
5. Decision: the pre-registered decision the public score would change, and whether the
   candidate duplicates an already-scored submission (cached submissions list, fingerprints).

Admission gates fail closed (H-B-35, D-B-033 review, astra code review 2026-10-08, astra re-review 2026-10-08 A1-A4):
- quarantine: tools/diagnostic_quarantine.py is checked before loading any member, baseline, fold or replicate OOF; every fold
  OOF file and fold run directory is checked before the pooled OOF is read. A quarantined member/baseline disables OOF and
  replication scoring (at most PUBLIC-ONLY with an explicit public basis); a quarantined teacher is FAIL_LINEAGE. Metadata may
  still be traced to explain ancestry. A replicate is resolved like a member (pooled OOF, fold OOF files, fold run directories,
  teachers), so a quarantined source anywhere in it skips the replicate with its marker/record reason;
- spec ingestion: decision.basis must be exactly "public" or "oof" (lowercase; anything else, e.g. "OOF" or "oof+public",
  is rejected); PUBLIC-ONLY is reachable only with an explicit "basis": "public";
- registration: rules (or a control role) count as pre-registered only with a registered_at that carries a time zone
  ("YYYY-MM-DD HH:MM KST" or ISO 8601 with an offset) and is earlier than the screen run and, for a candidate with a
  submission_ref, than the cached submission time of that ref (Kaggle CLI dates are UTC; a score is recorded after its
  submission). A submission_ref absent from the cache (or no cache) is REGISTRATION_NOT_VERIFIED (HOLD); a candidate not
  yet submitted (submission_ref null) is ordered against the screen run;
- lineage: a label file is admitted only with verified teacher ancestry (receipt.json beside it, output sha = label file
  sha, a non-empty teacher mapping whose every teacher OOF is bound by its recorded sha; EXP116 per-file outputs and
  oof/oof_sha256 teacher pairs are also accepted) or when it is on the approved
  non-distilled list (Dread / flight / relabel_v1, DEFAULTS['non_distilled_label_files']) without a teacher receipt; the
  receipt is checked independently of spec overrides (spec "teachers" only add teachers; "teachers": [] is accepted only on
  an approved file). Anything else, a missing label file, a missing OOF file or a missing fold-run OOF file is FAIL_LINEAGE
  (BLOCK). A member without OOF (e.g. an all-data model) must declare `"oof": null` with `"oof_reason"`; its OOF lines read
  NOT AVAILABLE and the best verdict is PUBLIC-ONLY (its `"checkpoint_sha256"`, a sha or a list, binds the served checkpoints);
- per-fold label files (A4): a protocol that registers one label file per fold (folds[k].label_file, EXP116) has EVERY fold's
  file resolved and checked: file sha and semantic sha against the registration, the fold run provenance, the assembled OOF
  result and the receipt beside the file; the teacher ancestry of all fold files is traced and combined. Any mismatch is
  FAIL_LINEAGE, and a spec "label_file" cannot replace the registered per-fold files;
- duplicates: checked against the persistent registry DEFAULTS['fingerprint_registry'] (seeded with --seed-registry, read by
  every screen, appended by every run that writes its reports) plus the candidates of the run. A scored (COMPLETE) submission
  is a duplicate on an equal member-OOF fingerprint, served-checkpoint fingerprint (per arm: checkpoint sha set, served K and
  normalised blend weight; arm names are cosmetic and excluded, A2) or release manifest sha (BLOCK); a match with a submitted
  candidate whose status the cache cannot confirm is HOLD; a missing or unreadable registry is HOLD ("duplicate coverage
  unavailable"); an equal three-study Public3 output sha alone is a warning;
- duplicate coverage (A1): a readable registry is not enough. It is reconciled against the submissions cache (--submissions or
  the bundle's submissions_cache): every cached submission that is not ERROR needs a registry entry with its submission_ref in
  the current fingerprint schema (FP_KEYS) - a screened candidate bound after submission (--bind-submission) or an acknowledged
  historical entry (--record-historical: fingerprints computed from local artifacts, every unavailable one with its reason).
  An uncovered submission, or no cache at all, is HOLD ("duplicate coverage incomplete"); a submission covered only by entries
  without any fingerprint is a caveat (a repeat of that model cannot be detected).

No training, GPU, network, Kaggle command or submission; thread pools limited to one thread.

  python tools/submission_screen.py --spec SPEC.json --submissions CACHE.csv   # without a cache duplicate coverage is HOLD
  python tools/submission_screen.py --specs BACKTEST.json      # several candidates, shared registry
  python tools/submission_screen.py --seed-registry A/screen.json B/screen.json   # create/extend the fingerprint registry
  python tools/submission_screen.py --record-historical HIST.json --submissions CACHE.csv   # acknowledge pre-screen submissions
  python tools/submission_screen.py --bind-submission CANDIDATE REF --submissions CACHE.csv   # after submitting a screened candidate
  python tools/submission_screen.py --candidate NAME --member coat5=PATH/oof_k36.csv:0.5 \
      --member rad5=PATH/oof_k36.csv:0.5 --release-out .local/x/out-y --plan-item H-B-31 \
      --decision-text "..." --rule "public>=0.920:transfer confirmed" --rule "public<=0.914:no transfer" \
      --registered-at "2026-10-07 16:17 KST"
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'submission-screen-v1'


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name + '_for_screen', Path(__file__).resolve().parent / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RP = _load_tool('anchor_receipt_policy')
DQ = _load_tool('diagnostic_quarantine')
TARGETS = list(RP.TARGETS)

DEFAULTS = {
    'baseline_name': 'CoAtNet5 Dread K36',
    'baseline_oof': 'artifacts/experiments/EXP099_B8_DREAD_OOF/oof_k36.csv',
    'folds_csv': 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/folds.csv',
    'components': 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json',
    'label_sets_dir': 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/label-sets',
    'train_csv': 'storage/competition/train.csv',
    'sample_submission': 'storage/competition/sample_submission.csv',
    'experiments_dir': 'artifacts/experiments',
    'out_root': 'artifacts/experiments/SUBMISSION_SCREEN',
    # persistent duplicate registry, read by every screen and appended by every writing run (astra review 2026-10-08)
    'fingerprint_registry': 'artifacts/experiments/SUBMISSION_SCREEN/fingerprints.json',
    'draws': 500,
    'seed': 20261003,
    # Three independent report-derived references, always scored (user rule 2026-10-07 19:50 KST).
    'references': {
        'dread': dict(path='storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv', rule='rint', exclude_gold=False),
        'flight_v4': dict(path='storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv', rule='ge0.5', exclude_gold=True),
        'relabel_v1': dict(path='storage/labels/derived_relabel/relabel_v1.csv', rule='ge0.5', exclude_gold=False),
    },
    'direction_tolerance': -0.002,
    # Report-derived label files without model teachers: an explicit `teachers: []` is accepted only on these (H-B-35).
    'non_distilled_label_files': {
        'dread': ['storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'],
        'flight_v4': ['storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv',
                      'storage/labels/derived_EXP002/flight_public_control_weak4349.csv'],
        'relabel_v1': ['storage/labels/derived_relabel/relabel_v1.csv'],
    },
}
EVIDENCE_CLASS = ('weak-label development evidence: report-derived labels on the 4,343 G1 studies; it can filter candidates '
                  'but does not establish public (official-label) transfer (D-B-016 is a weak-label proxy, D-B-020 as corrected, D-A-049)')
CEILING_NOTE = ('Dread, flight_v4 and relabel_v1 are all extracted from the same report text, so they share its ceiling: '
                'agreement across them guards against one labeller\'s quirks, not against report-vs-image gaps, which only '
                'Gold58 (diagnostic, exposed) or the public score can see')
ENGINEERING_SCOPE = ('Public3 runs the release on the 3-study supplied test cohort: it proves the engineering path only, '
                     'not hidden runtime, hidden completeness, accuracy or score identity (D-C-004, D-A-049)')
PASS_STATUSES = ('PASS_TEST_COHORT_ONLY', 'PASS_RANK_MEAN_TEST_COHORT_ONLY')
OPS = {'>=': lambda a, b: a >= b - 1e-12, '<=': lambda a, b: a <= b + 1e-12, '>': lambda a, b: a > b, '<': lambda a, b: a < b}
RULE_TEXT = {'rint': 'rint(float32 soft label)', 'ge0.5': 'soft label >= 0.5'}
TZ_NAMES = {'KST': 9, 'UTC': 0, 'GMT': 0, 'Z': 0}
_WHEN = re.compile(r'(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)\s*([A-Za-z]{1,4}|[+-]\d{2}:?\d{2})?')
LINEAGE_ORDER = ['FAIL_DIRECT', 'FAIL_LINEAGE', 'INCOMPLETE', 'FLAG_INDIRECT', 'FLAG_COMPONENT', 'NOT_AVAILABLE', 'PASS']


# ----------------------------------------------------------------------------- basics

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def uid_hash(ids):
    """Ordered UID hash used by the EXP084/EXP099 runners: sha256 of the newline-joined list."""
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None, sets to sorted lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, set):
        return sorted(clean(v) for v in o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, Path):
        return o.as_posix()
    return o


class Ctx:
    """Paths, configuration and cached shared inputs for one screen run."""

    def __init__(self, root=ROOT, **overrides):
        self.root = Path(root)
        self.cfg = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
        self._sha, self._protocols, self._pop, self._comp, self._gold, self._oof_results, self._labels = {}, None, None, None, None, None, {}
        self.oof_results_skipped = []

    def path(self, p):
        if p is None or p == '':
            return None
        q = Path(str(p))
        return q if q.is_absolute() else self.root / Path(str(p).replace('\\', '/'))

    def rel(self, p):
        if p is None:
            return None
        try:
            return Path(p).resolve().relative_to(self.root.resolve()).as_posix()
        except (ValueError, OSError):
            return Path(p).as_posix()

    def sha(self, p):
        key = str(Path(p).resolve())
        if key not in self._sha:
            self._sha[key] = sha256_file(p)
        return self._sha[key]

    def protocol_by_sha(self, sha):
        if self._protocols is None:
            self._protocols = {}
            exp = self.path(self.cfg['experiments_dir'])
            for m in sorted(exp.glob('*/manifest.json')) if exp is not None and exp.is_dir() else []:
                self._protocols.setdefault(self.sha(m), m)
        return self._protocols.get(sha)

    def oof_results(self):
        """Eligible OOF results for replicate auto-detection; retain quarantine skips for reporting."""
        if self._oof_results is None:
            self._oof_results = []
            exp = self.path(self.cfg['experiments_dir'])
            for r in sorted(exp.glob('*/result.json')) if exp is not None and exp.is_dir() else []:
                try:
                    d = read_json(r)
                except (OSError, ValueError):
                    continue
                if isinstance(d, dict) and 'oof' in str(d.get('schema', '')):
                    reason = DQ.quarantine_reason(r, root=self.root)
                    if reason:
                        self.oof_results_skipped.append((r, d, reason))
                    else:
                        self._oof_results.append((r, d))
        return self._oof_results

    def population(self):
        """UID -> fold of the screen cohort (folds.csv); per-fold tables use these folds."""
        if self._pop is None:
            df = pd.read_csv(self.path(self.cfg['folds_csv']), dtype={'StudyInstanceUID': str})
            if df['StudyInstanceUID'].duplicated().any():
                raise ValueError('folds.csv has duplicate UIDs')
            self._pop = dict(zip(df['StudyInstanceUID'], df['fold'].astype(int)))
        return self._pop

    def components(self):
        if self._comp is None:
            of, members = {}, {}
            p = self.path(self.cfg['components'])
            for c in read_json(p) if p is not None and p.is_file() else []:
                members[c['component_id']] = list(c['members'])
                for u in c['members']:
                    of[u] = c['component_id']
            self._comp = (of, members)
        return self._comp

    def gold(self):
        """The 58 fully labelled official rows of train.csv (report-only, never a criterion)."""
        if self._gold is None:
            p = self.path(self.cfg['train_csv'])
            if p is None or not p.is_file():
                self._gold = set()
            else:
                df = pd.read_csv(p, dtype={'StudyInstanceUID': str}, usecols=['StudyInstanceUID', *TARGETS])
                self._gold = set(df.loc[df[TARGETS].notna().all(axis=1), 'StudyInstanceUID'])
        return self._gold

    def labels(self, path, uids, rule, exclude_gold):
        key = (str(path), rule, bool(exclude_gold), len(uids))
        if key not in self._labels:
            self._labels[key] = load_labels(path, uids, rule, self.gold() if exclude_gold else None)
        return self._labels[key]


def load_pred(path):
    df = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    missing = [t for t in ['StudyInstanceUID', *TARGETS] if t not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    df = df.set_index('StudyInstanceUID')
    if df.index.duplicated().any():
        raise ValueError(f'{path}: duplicate UIDs')
    out = df[TARGETS].astype(np.float64)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError(f'{path}: non-finite predictions')
    return out


def load_labels(path, uids, rule='rint', gold=None):
    """Binary labels for the screen UIDs: 'rint' = numpy.rint(float32) (Dread contract),
    'ge0.5' = soft label >= 0.5. Gold rows are dropped first when gold is given."""
    df = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    dropped = 0
    if gold:
        keep = ~df.index.isin(list(gold))
        dropped = int((~keep).sum())
        df = df[keep]
    miss = [u for u in uids if u not in df.index]
    if miss:
        raise ValueError(f'{path}: {len(miss)} screen UIDs missing from the label file')
    soft = df.loc[list(uids), TARGETS].to_numpy(dtype=np.float32)
    if not np.isfinite(soft).all():
        raise ValueError(f'{path}: non-finite labels on screen UIDs')
    if rule == 'rint':
        y = np.rint(soft)
    elif rule == 'ge0.5':
        y = (soft >= 0.5).astype(np.float32)
    else:
        raise ValueError(f'unknown label rule {rule}')
    y = y.astype(np.int8)
    if not np.isin(y, (0, 1)).all():
        raise ValueError(f'{path}: binarised labels outside {{0,1}}')
    return y, dropped


def auc_cols(Y, S):
    """Per-column average-tie rank AUC (as tools/g1_compare.py); NaN where a class is absent."""
    try:
        from scipy.stats import rankdata
        R = rankdata(S, axis=0)
    except ImportError:  # pragma: no cover
        R = pd.DataFrame(S).rank(method='average').to_numpy()
    Y = Y.astype(np.float64)
    P = Y.sum(0)
    N = Y.shape[0] - P
    with np.errstate(invalid='ignore', divide='ignore'):
        a = ((R * Y).sum(0) - P * (P + 1) / 2) / (P * N)
    a[(P == 0) | (N == 0)] = np.nan
    return a


def nanmean(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float('nan')


def combine(preds, weights):
    """One member: its probabilities. Several: per-target weighted mean of average-tie percentile
    ranks over the full cohort (EXP098 / release v5-v6 formula); weights normalised to sum 1."""
    if len(preds) == 1:
        return preds[0].to_numpy(dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (w <= 0).any() or not np.isfinite(w).all():
        raise ValueError('blend weights must be positive')
    w = w / w.sum()
    out = np.zeros(preds[0].shape, dtype=np.float64)
    for p, wi in zip(preds, w):
        out += wi * p.rank(method='average', pct=True).to_numpy(dtype=np.float64)
    return out


def member_fingerprint(members):
    """None when a member has no OOF sha (declared without OOF or file missing): such a fingerprint never matches."""
    if not members or any(not M.get('oof_sha256') for M in members):
        return None
    tot = sum(x['weight'] for x in members) or 1.0
    return hashlib.sha256(json.dumps(sorted([M.get('oof_sha256'), round(M['weight'] / tot, 9)] for M in members)).encode()).hexdigest()


def parse_when(text, default_tz=None):
    """Aware datetime from 'YYYY-MM-DD HH:MM[:SS[.ffffff]]' ('T' or space) followed by KST/UTC/GMT/Z or +HH:MM.
    Without a zone it returns None unless default_tz is given (the Kaggle CLI writes submission dates in UTC)."""
    if text is None or str(text).strip() == '':
        return None
    m = _WHEN.fullmatch(str(text).strip())
    if not m:
        return None
    day, clock, zone = m.groups()
    try:
        when = dt.datetime.fromisoformat(f'{day}T{clock}')
    except ValueError:
        return None
    if zone is None:
        return when.replace(tzinfo=default_tz) if default_tz is not None else None
    if zone.upper() in TZ_NAMES:
        return when.replace(tzinfo=dt.timezone(dt.timedelta(hours=TZ_NAMES[zone.upper()])))
    if zone[0] in '+-':
        hh, mm = int(zone[1:3]), int(zone[-2:])
        if hh > 23 or mm > 59:
            return None
        off = dt.timedelta(hours=hh, minutes=mm)
        return when.replace(tzinfo=dt.timezone(off if zone[0] == '+' else -off))
    return None


def registration_check(dec, cached, ref, now, cache_given=True):
    """Fail-closed pre-registration gate (H-B-35): registered_at must parse with a time zone and be earlier than the
    screen run and, when the candidate has a submission_ref, than the cached submission time of that submission. The cache
    records only the submission time and a score is recorded after it, so a registration later than the submission cannot be
    shown to precede the score. A submission_ref that is absent from the cache (or no cache at all) cannot be ordered:
    REGISTRATION_NOT_VERIFIED (astra review 2026-10-08). A candidate not yet submitted (no submission_ref) is ordered
    against the screen run."""
    raw = dec.get('registered_at')
    out = dict(valid=False, status='REGISTRATION_NOT_VERIFIED', registered_at=raw, registered_at_utc=None, screen_run_at=now.isoformat(timespec='seconds'),
               submission_recorded_at_utc=None, score_order=None)
    when = parse_when(raw)
    if raw is None or str(raw).strip() == '':
        out['reason'] = 'registered_at missing'
        return out
    if when is None:
        out['reason'] = (f'registered_at {raw!r} has no time zone (write e.g. "2026-10-07 16:17 KST")' if parse_when(raw, dt.timezone.utc)
                         else f'registered_at {raw!r} is not a parseable time ("YYYY-MM-DD HH:MM KST" or ISO 8601 with an offset)')
        return out
    out['registered_at_utc'] = when.astimezone(dt.timezone.utc).isoformat(timespec='seconds')
    if not when < now:
        out['reason'] = f'registered_at {raw} is not earlier than the screen run ({out["screen_run_at"]})'
        return out
    if cached is None and ref:
        out['reason'] = (f'submission {ref} is ' + ('not in the cached submissions list' if cache_given else 'not checkable (no submissions cache given)')
                         + ': its submission time is unknown, so the registration cannot be ordered before its score')
        return out
    if cached is None:
        out['score_order'] = 'no submission yet: ordered against the screen run'
    else:
        sub_at = parse_when(cached.get('date'), dt.timezone.utc)
        if sub_at is None:
            out['reason'] = f'cached date of submission {ref} ({cached.get("date")!r}) is not parseable: registration cannot be ordered before its score'
            return out
        out['submission_recorded_at_utc'] = sub_at.astimezone(dt.timezone.utc).isoformat(timespec='seconds')
        if not when < sub_at:
            out['reason'] = (f'registered_at {raw} is not earlier than submission {ref} (cached {out["submission_recorded_at_utc"]}), '
                             'so its score may have been known')
            return out
        out['score_order'] = f'earlier than submission {ref} (cached {out["submission_recorded_at_utc"]}; its score is recorded after it)'
    out.update(valid=True, status='VERIFIED', reason='ok')
    return out


# ----------------------------------------------------------------------------- 1 lineage

def known_non_distilled(ctx, label_file):
    """Name of the known non-distilled label set whose file this is (resolved path match), else None."""
    if label_file is None:
        return None
    target = Path(label_file).resolve()
    for name, paths in (ctx.cfg.get('non_distilled_label_files') or {}).items():
        for p in [paths] if isinstance(paths, str) else paths:
            q = ctx.path(p)
            if q is not None and q.resolve() == target:
                return name
    return None


def gate_failures(M, prefix='', field='gate_failures'):
    """Fail-closed gate failures of a member and, recursively, of its teachers."""
    out = [prefix + g for g in M.get(field) or []]
    for t in M.get('teachers') or []:
        out += gate_failures(t, f'{prefix}teacher {t["name"]}: ', field=field)
    return out


def quarantine_oof(ctx, M, path, role):
    """Refuse prediction use, retaining metadata for lineage/engineering and the marker reason for the report."""
    reason = DQ.quarantine_reason(path, root=ctx.root) if path is not None else None
    if reason:
        failure = f'{role} {ctx.rel(path)} quarantined: {reason}'
        M['oof_gate_failures'].append(failure)
        if M['depth'] > 0:
            M['gate_failures'].append(failure)  # a diagnostic-only teacher is never an admitted target source
    return reason


def _k_of(path):
    m = re.search(r'k(\d+)', Path(path).stem)
    return int(m.group(1)) if m else None


def teacher_lineage(ctx, M, ms, label_file):
    """Teacher specs to resolve for member M, with fail-closed gate failures (H-B-35; astra review 2026-10-08).

    Receipt binding is validated independently of spec overrides. Admission needs one of
    - verified teacher ancestry: receipt.json beside the label file, readable, whose output sha256 equals the label file sha and whose
      teachers are a non-empty name -> {path, sha256} mapping (each teacher OOF is then bound by that sha when resolved), or
    - a label file on the approved non-distilled list (DEFAULTS['non_distilled_label_files']) with no teacher receipt beside it.
    Anything else is FAIL_LINEAGE. A spec "teachers" list only ADDS teachers (more influence, never less): it cannot stand in for a
    receipt, and an explicit "teachers": [] is accepted only on an approved non-distilled file.
    EXP116 receipts bind outputs[label_file.name].sha256 and teachers' {oof, oof_sha256} pairs instead."""
    teacher_specs = ms.get('teachers')
    gates = M['gate_failures']
    if teacher_specs is not None and not isinstance(teacher_specs, list):
        gates.append('spec "teachers" is not a list: teacher lineage unverifiable')
        M['label_provenance'] = f'spec "teachers" is not a list ({teacher_specs!r})'
        teacher_specs = []
    spec_teachers = [dict(t, source='spec') for t in teacher_specs or [] if isinstance(t, dict)]
    if teacher_specs and len(spec_teachers) != len(teacher_specs):
        gates.append('spec "teachers" entries must be objects with "name" and "oof"')
    if label_file is None or not label_file.is_file():
        M['label_provenance'] = 'label file not found: teacher lineage unverifiable'
        M['issues'].append('label file not found: teacher lineage unverifiable')
        gates.append('label file not found: neither a teacher receipt nor the approved non-distilled list can be checked')
        M['teacher_binding'] = 'UNVERIFIED'
        return spec_teachers
    lf = ctx.rel(label_file)
    known = known_non_distilled(ctx, label_file)
    rc = label_file.parent / 'receipt.json'
    rec, rec_error = None, None
    if rc.is_file():
        try:
            rec = read_json(rc)
        except (OSError, ValueError):
            rec_error = 'not readable'
        if rec_error is None and not isinstance(rec, dict):
            rec, rec_error = None, 'not a JSON object'
    rec_teachers = rec.get('teachers') if rec is not None else None
    out_sha = (rec.get('output') or {}).get('sha256') if rec is not None and isinstance(rec.get('output'), dict) else None
    if rec is not None and 'outputs' in rec:
        # Per-file receipts must bind this exact sibling, never any other entry with an equal hash.
        outputs = rec['outputs']
        output = outputs.get(label_file.name) if isinstance(outputs, dict) else None
        file_sha = output.get('sha256') if isinstance(output, dict) else None
        if 'output' in rec and (out_sha is None or out_sha != file_sha):
            gates.append(f'teacher receipt {ctx.rel(rc)} has conflicting output/outputs bindings for {lf}')
        out_sha = file_sha
    binds = rec is not None and out_sha is not None and out_sha == M['label_file_sha256']
    pv = label_file.parent / 'provenance.json'
    labeller = None
    if pv.is_file():
        try:
            d = read_json(pv)
            labeller = d.get('labeller', 'unrecorded') if isinstance(d, dict) else 'provenance.json is not an object'
        except (OSError, ValueError):
            labeller = 'provenance.json not readable'
    if rec_error is not None:
        gates.append(f'teacher receipt {ctx.rel(rc)} is {rec_error}: the teachers behind {lf} cannot be bound')
        M['label_provenance'] = f'teacher receipt {ctx.rel(rc)} {rec_error}'
        M['teacher_binding'] = 'UNVERIFIED'
        return spec_teachers
    if known is not None:
        if rec is not None and rec_teachers:
            # a teacher receipt beside an approved file: either it binds this file (distilled, so not approved) or it is ambiguous
            gates.append(f'{lf} is on the approved non-distilled list ({known}), but {ctx.rel(rc)} records teachers for this label file'
                         + ('' if binds else ' (its output sha does not bind the file: ambiguous)'))
            M['label_provenance'] = f'approved non-distilled label set {known}, contradicted by the teacher receipt {ctx.rel(rc)}'
            M['teacher_binding'] = 'CONTRADICTED'
            return spec_teachers
        M['teacher_binding'] = 'APPROVED_NON_DISTILLED'
        if teacher_specs == []:
            M['label_provenance'] = f'spec declares no teachers; {lf} is the known non-distilled label set {known}'
        else:
            M['label_provenance'] = (f'{lf} is the known non-distilled label set {known} (approved list)'
                                     + (f'; {ctx.rel(pv)}: labeller {labeller}' if labeller else '')
                                     + (f'; {len(spec_teachers)} spec-named teacher(s) added' if spec_teachers else ''))
        return spec_teachers
    # not on the approved list: only a binding receipt with a non-empty teacher mapping verifies the ancestry
    M['teacher_binding'] = 'UNVERIFIED'
    if teacher_specs == []:
        gates.append(f'teachers: [] declared for {lf}, which is not a known non-distilled label set '
                     f'({", ".join(sorted(ctx.cfg.get("non_distilled_label_files") or {}))}): its teachers must be named in a receipt')
    if rec is None:
        gates.append(f'{lf} is not a known non-distilled label set and has no teacher receipt beside it: teacher ancestry unverifiable'
                     + (' (spec-named teachers cannot replace the receipt)' if spec_teachers else ''))
        M['label_provenance'] = (f'no teacher receipt beside {lf} and not on the approved non-distilled list'
                                 + (f'; labeller {labeller}' if labeller else ''))
        return spec_teachers
    if out_sha is None:
        gates.append(f'teacher receipt {ctx.rel(rc)} records no output sha256: it cannot be bound to {lf}')
        M['label_provenance'] = f'teacher receipt {ctx.rel(rc)} does not bind this label file (no output sha256)'
        return spec_teachers
    if not binds:
        gates.append(f'teacher receipt {ctx.rel(rc)} output sha {str(out_sha)[:12]} differs from the label file sha '
                     f'{str(M["label_file_sha256"])[:12]}: the teachers behind these targets cannot be bound')
        M['label_provenance'] = f'teacher receipt {ctx.rel(rc)} does not bind this label file'
        return spec_teachers
    if not isinstance(rec_teachers, dict) or not rec_teachers:
        gates.append(f'teacher receipt {ctx.rel(rc)}: teachers record is not a non-empty name -> OOF mapping, so the ancestry is not verified')
        M['label_provenance'] = f'teacher receipt {ctx.rel(rc)}: no teacher mapping'
        return spec_teachers
    teachers = []
    for n, t in rec_teachers.items():
        # Choose a complete schema pair; partial or contradictory legacy fields cannot be bypassed with aliases.
        legacy = isinstance(t, dict) and ('path' in t or 'sha256' in t)
        path_key, sha_key = ('path', 'sha256') if legacy else ('oof', 'oof_sha256')
        if not isinstance(t, dict) or not t.get(path_key) or not t.get(sha_key):
            gates.append(f'teacher receipt {ctx.rel(rc)}: teacher {n} has no recorded OOF path and sha256, so its OOF is not bound')
            continue
        if legacy and ('oof' in t or 'oof_sha256' in t):
            alias = ctx.path(t.get('oof'))
            if alias is None or alias.resolve() != ctx.path(t['path']).resolve() or t.get('oof_sha256') != t['sha256']:
                gates.append(f'teacher receipt {ctx.rel(rc)}: teacher {n} has conflicting OOF path/sha256 aliases')
        teachers.append(dict(name=n, oof=t[path_key], expected_sha256=t[sha_key], source='receipt'))

    def key(t):
        p = ctx.path(t.get('oof'))
        return str(p.resolve()) if p is not None else None
    seen = {key(t) for t in teachers}
    extra = [t for t in spec_teachers if key(t) is None or key(t) not in seen]
    M['teacher_binding'] = 'RECEIPT_BOUND'
    M['label_provenance'] = (f'teacher receipt {ctx.rel(rc)}: {rec.get("teacher_rule", "teacher outputs")} mixed into the targets'
                             + (f'; {len(extra)} spec-named teacher(s) added' if extra else ''))
    return teachers + extra


def per_fold_label_files(manifest):
    """{fold: folds[k].label_file} of a protocol that registers one label file per fold (EXP116), else {}."""
    folds = (manifest or {}).get('folds')
    if not isinstance(folds, dict):
        return {}
    return {int(k): f.get('label_file') for k, f in folds.items() if isinstance(f, dict) and 'label_file' in f}


def _short(x):
    return str(x)[:12]


def fold_label_check(ctx, M, ms, plf, res):
    """Per-fold label files (astra re-review 2026-10-08 A4): resolve EVERY folds[k].label_file, verify its file and semantic sha
    against the registration, the fold run provenance, the assembled OOF result and the receipt beside it, and combine the teacher
    ancestry of all fold files (fail closed). Returns the combined teacher specs."""
    gates, files, teachers, seen, bindings = M['gate_failures'], {}, [], {}, {}
    sources = res.get('sources') or {}
    if ms.get('label_file'):
        gates.append(f'spec "label_file" {ms["label_file"]} cannot replace the per-fold label files the protocol registers (folds[k].label_file)')
    missing = sorted(set(M['folds']) - set(plf))
    if missing:
        gates.append(f'the protocol registers per-fold label files but none for fold(s) {missing}: what those folds trained on is not bound')
    for k in sorted(set(M['folds']) & set(plf)):
        lf, tag = plf[k], f'fold {k} label file'
        if not isinstance(lf, dict) or not all(lf.get(x) for x in ('path', 'sha256', 'semantic_sha256')):
            gates.append(f'{tag}: the registration records no path / sha256 / semantic_sha256, so the file this fold trained on is not bound')
            continue
        p = ctx.path(lf['path'])
        rec = files[k] = dict(path=ctx.rel(p), registered_sha256=lf['sha256'], semantic_sha256=lf['semantic_sha256'], sha256=None)
        if p is None or not p.is_file():
            gates.append(f'{tag} {lf["path"]} not found: the targets this fold trained on cannot be checked')
            continue
        rec['sha256'] = sha = ctx.sha(p)
        if sha != lf['sha256']:
            gates.append(f'{tag} {rec["path"]}: sha {_short(sha)} differs from the registered sha {_short(lf["sha256"])}')
        run = M['runs'].get(k) or {}
        if not run.get('provenance_found'):
            gates.append(f'{tag}: fold run provenance missing, so the file this fold trained on is not bound')
        else:
            got_sha, got_sem = run.get('label_file_sha256'), run.get('label_semantic_sha256') or run.get('label_hash')
            if got_sha is None and got_sem is None:
                gates.append(f'{tag}: fold run provenance records neither label_file_sha256 nor a label semantic hash')
            if got_sha is not None and got_sha != lf['sha256']:
                gates.append(f'{tag}: fold run provenance label_file_sha256 {_short(got_sha)} differs from the registered sha {_short(lf["sha256"])}')
            if got_sem is not None and got_sem != lf['semantic_sha256']:
                gates.append(f'{tag}: fold run provenance label semantic sha {_short(got_sem)} differs from the registered {_short(lf["semantic_sha256"])}')
            recorded = ctx.path(run.get('label_file_recorded'))
            if recorded is not None and recorded.resolve() != p.resolve():
                gates.append(f'{tag}: fold run provenance names {run["label_file_recorded"]}, not the registered {lf["path"]}')
        assembled = (sources.get(str(k)) or {}).get('label_file_sha256')
        if assembled is not None and assembled != lf['sha256']:
            gates.append(f'{tag}: the assembled OOF result records label_file_sha256 {_short(assembled)} for this fold, not the registered sha')
        rc = p.parent / 'receipt.json'
        try:
            outs = read_json(rc).get('outputs') if rc.is_file() else None
        except (OSError, ValueError, AttributeError):
            outs = None  # an unreadable receipt is reported by teacher_lineage
        out = outs.get(p.name) if isinstance(outs, dict) else None
        if isinstance(out, dict) and out.get('semantic_sha256') != lf['semantic_sha256']:
            gates.append(f'{tag}: receipt {ctx.rel(rc)} records semantic sha {_short(out.get("semantic_sha256"))} for {p.name}, '
                         f'not the registered {_short(lf["semantic_sha256"])}')
        S = dict(gate_failures=[], issues=[], label_file_sha256=sha)
        specs = teacher_lineage(ctx, S, ms, p)
        gates += [f'{tag}: {g}' for g in S['gate_failures']]
        M['issues'] += [f'{tag}: {i}' for i in S['issues']]
        rec.update(teacher_binding=S.get('teacher_binding'), label_provenance=S.get('label_provenance'))
        bindings[k] = S.get('teacher_binding')
        for t in specs:
            q = ctx.path(t.get('oof'))
            key = str(q.resolve()) if q is not None else f'unresolved:{t.get("name")}'
            if key in seen:
                if seen[key].get('expected_sha256') != t.get('expected_sha256'):
                    gates.append(f'{tag}: teacher {t.get("name")} is bound to sha {_short(t.get("expected_sha256"))}, another fold file binds '
                                 f'{_short(seen[key].get("expected_sha256"))}')
                continue
            seen[key] = t
            teachers.append(t)
    kinds = sorted({str(b) for b in bindings.values()})
    M['teacher_binding'] = kinds[0] if len(kinds) == 1 else ('MIXED' if kinds else 'UNVERIFIED')
    if len(kinds) > 1:
        gates.append('the fold label files have different teacher bindings (' + ', '.join(f'fold {k}: {b}' for k, b in sorted(bindings.items())) + ')')
    first = files[min(files)] if files else {}
    M['label_files'] = {k: files[k] for k in sorted(files)}
    M['label_file'] = first.get('path')
    M['label_file_sha256'] = first.get('sha256')
    M['label_provenance'] = (f'per-fold label files (folds[k].label_file): {len(files)} of {len(M["folds"])} fold files resolved and checked against '
                             f'the registration, run provenance, assembled result and receipt; {len(teachers)} teacher(s) combined across folds'
                             + (f'; fold {min(files)}: {first["label_provenance"]}' if first.get('label_provenance') else ''))
    return teachers


def resolve_member(ctx, ms, depth=0):
    """Bind one OOF source to its protocol fold lists, fold runs, labels and teachers."""
    oof_path = ctx.path(ms.get('oof'))
    M = dict(name=ms.get('name') or (oof_path.parent.name if oof_path else '?'), oof=ctx.rel(oof_path),
             weight=float(ms.get('weight', 1.0)), arm=ms.get('arm') or ms.get('name'), issues=[], notes=[],
             folds={}, fold_of={}, runs={}, teachers=[], ancestor_sets=[], depth=depth, k_eval=None, label_set=None,
             label_file=None, label_provenance=None, protocol=None, protocol_sha256=None, external_init=None, seed=None,
             gate_failures=[], oof_gate_failures=[], oof_declared_null=False, oof_reason=None)
    if 'oof' in ms and ms['oof'] is None:
        # declared without OOF (e.g. an all-data model): allowed only with a stated reason (H-B-35, fail closed)
        ck = ms.get('checkpoint_sha256')
        M.update(oof_declared_null=True, oof_reason=str(ms.get('oof_reason') or '').strip() or None, label_set=ms.get('label_set'),
                 declared_checkpoints=[ck] if isinstance(ck, str) else [str(c) for c in ck or []],
                 label_provenance='not checked: the member declares no OOF, so there is no validation set to expose')
        if not M['oof_reason']:
            M['gate_failures'].append('"oof": null without a stated "oof_reason" (e.g. "all-data model: no held-out folds")')
        return M
    if oof_path is None or not oof_path.is_file():
        M['issues'].append(f'OOF file missing: {ms.get("oof")}')
        M['gate_failures'].append(f'OOF file missing: {ms.get("oof")} (a member without OOF must declare "oof": null with "oof_reason")')
        return M
    quarantined = quarantine_oof(ctx, M, oof_path, 'OOF')
    M['oof_sha256'] = ctx.sha(oof_path)  # identity only, never prediction evidence when quarantined
    if ms.get('expected_sha256') and ms['expected_sha256'] != M['oof_sha256']:
        # a teacher whose OOF is not the file the receipt recorded is not the teacher behind the targets (fail closed)
        M['issues'].append('OOF sha256 differs from the sha recorded by the referencing receipt')
        M['gate_failures'].append(f'OOF sha {M["oof_sha256"][:12]} differs from the sha {str(ms["expected_sha256"])[:12]} recorded by the '
                                  'referencing teacher receipt: the teacher OOF is not bound')
    M['k_eval'] = _k_of(oof_path)
    res_path = oof_path.parent / 'result.json'
    try:
        res = read_json(res_path) if res_path.is_file() else {}
    except (OSError, ValueError):
        if not quarantined:
            raise
        res = {}  # the quarantine gate already records unreadable eligibility metadata (fail closed)
    M['oof_result'] = ctx.rel(res_path) if res else None
    if not res:
        M['issues'].append('OOF result.json missing: fold runs and protocol cannot be bound')
    out_sha = (res.get('outputs') or {}).get(oof_path.name)
    M['oof_sha_matches_result'] = (out_sha == M['oof_sha256']) if out_sha else None
    if out_sha and out_sha != M['oof_sha256']:
        M['issues'].append('OOF file sha differs from result.json outputs')
    M['label_set'] = ms.get('label_set') or res.get('label_set')
    M['seed'] = res.get('seed')
    M['seed42_protocol_sha256'] = res.get('seed42_protocol_sha256')
    # protocol
    proto = ctx.path(ms['protocol']) if ms.get('protocol') else None
    if proto is None and res.get('protocol'):
        p = ctx.path(res['protocol'])
        proto = p / 'manifest.json' if p.is_dir() else p
    if proto is None and res.get('protocol_sha256'):
        proto = ctx.protocol_by_sha(res['protocol_sha256'])
    manifest = read_json(proto) if proto is not None and Path(proto).is_file() else None
    if manifest is not None:
        M['protocol'] = ctx.rel(proto)
        M['protocol_sha256'] = ctx.sha(proto)
    M['protocol_sha_matches_result'] = (M['protocol_sha256'] == res.get('protocol_sha256')) if manifest and res.get('protocol_sha256') else None
    if manifest is None:
        M['issues'].append('training protocol manifest not found (by path or sha)')
    elif M['protocol_sha_matches_result'] is False:
        M['issues'].append('protocol manifest sha differs from the OOF result protocol_sha256')
    M['seed'] = M['seed'] if M['seed'] is not None else (manifest or {}).get('seed')
    M['protocol_folds_sha'] = ((manifest or {}).get('fold_protocol') or {}).get('folds_csv_sha256') or (manifest or {}).get('folds_csv_sha256')
    # label-set registration
    reg = None
    if M['label_set']:
        rp = ctx.path(ctx.cfg['label_sets_dir']) / f"{M['label_set']}.json"
        if rp.is_file():
            reg = read_json(rp)
        elif manifest and isinstance(manifest.get('label_sets'), dict) and M['label_set'] in manifest['label_sets']:
            reg = manifest['label_sets'][M['label_set']]
        elif manifest and isinstance(manifest.get('label_set'), dict) and manifest['label_set'].get('name') == M['label_set']:
            reg = manifest['label_set']
    M['label_registration'] = {k: (reg or {}).get(k) for k in ('path', 'sha256', 'semantic_sha256', 'gold_overlap', 'pos_weight_source')} if reg else None
    # folds (training and validation UID lists)
    folds_spec = (manifest or {}).get('folds')
    if isinstance(folds_spec, dict) and folds_spec:
        for k, f in sorted(folds_spec.items(), key=lambda kv: int(kv[0])):
            oi = (f or {}).get('ordered_ids') or {}
            p = ctx.path(oi.get('path'))
            if p is None or not p.is_file():
                M['issues'].append(f'fold {k}: ordered-ids file missing')
                continue
            d = read_json(p)
            tr, va = [str(u) for u in d.get('train', [])], [str(u) for u in d.get('val', [])]
            M['folds'][int(k)] = dict(source=ctx.rel(p), train=tr, val=va,
                                      file_sha_ok=(ctx.sha(p) == oi['sha256']) if oi.get('sha256') else None,
                                      train_hash_ok=(uid_hash(tr) == oi['train_sha256']) if oi.get('train_sha256') else None,
                                      val_hash_ok=(uid_hash(va) == oi['val_sha256']) if oi.get('val_sha256') else None)
            for key in ('file_sha_ok', 'train_hash_ok', 'val_hash_ok'):
                if M['folds'][int(k)][key] is False:
                    M['issues'].append(f'fold {k}: {key[:-3]} does not match the protocol manifest')
    elif manifest is not None:
        M['issues'].append('protocol has no per-fold ordered-ids: training UIDs cannot be bound')
    for k, f in M['folds'].items():
        for u in f['val']:
            if u in M['fold_of']:
                M['issues'].append(f'UID {u} is validation in folds {M["fold_of"][u]} and {k}')
            M['fold_of'][u] = k
    plf = per_fold_label_files(manifest)
    # fold sources: every fold OOF file and fold run directory passes the quarantine gate BEFORE any prediction is read
    # (astra re-review 2026-10-08 A3: a quarantined source fold disqualifies the pooled OOF built from it)
    sources = res.get('sources') or {}
    fold_src = {}
    for k in M['folds']:
        src = sources.get(str(k)) or {}
        rp = ctx.path((ms.get('fold_runs') or {}).get(str(k)) or src.get('run'))
        if M['k_eval'] == 24 and oof_path.name == 'oof_k24.csv' and rp is not None:
            fpath, fsha = rp / 'oof_best.csv', src.get('oof_k24_sha256')
        else:
            fpath, fsha = ctx.path(src.get(f'oof_k{M["k_eval"]}')), src.get(f'oof_k{M["k_eval"]}_sha256')
        reason = quarantine_oof(ctx, M, fpath, f'fold {k} OOF')
        if not reason and rp is not None and rp.is_dir():
            reason = quarantine_oof(ctx, M, rp, f'fold {k} run directory')
        fold_src[k] = (src, rp, fpath, fsha, reason)
    if not M['oof_gate_failures']:
        M['pred'] = load_pred(oof_path)
    if M['folds'] and 'pred' in M and set(M['pred'].index) != set(M['fold_of']):
        M['issues'].append(f'OOF UIDs ({len(M["pred"])}) differ from the union of fold validation lists ({len(M["fold_of"])})')
    # fold runs: provenance hashes, checkpoints, per-fold OOF binding
    label_file = None if plf else ctx.path(ms.get('label_file') or (reg or {}).get('path'))
    for k, f in M['folds'].items():
        src, rp, fpath, fsha, fold_quarantined = fold_src[k]
        r = dict(run=ctx.rel(rp) if rp is not None else None)
        prov = read_json(rp / 'provenance.json') if rp is not None and (rp / 'provenance.json').is_file() else None
        if prov is None:
            M['issues'].append(f'fold {k}: run provenance.json missing; training UIDs come from the protocol lists only')
        else:
            r['provenance_found'] = True
            r['train_hash_ok'] = prov.get('train_ids_ordered_sha256') == uid_hash(f['train'])
            r['eval_hash_ok'] = prov.get('eval_ids_ordered_sha256') == uid_hash(f['val'])
            r['label_hash'] = prov.get('label_hash')
            r.update(label_file_recorded=prov.get('label_file'), label_file_sha256=prov.get('label_file_sha256'),
                     label_semantic_sha256=prov.get('label_semantic_sha256'))
            if plf:
                sem = plf[k].get('semantic_sha256') if isinstance(plf.get(k), dict) else None
            else:
                sem = (reg or {}).get('semantic_sha256')
            r['label_hash_ok'] = (prov.get('label_hash') == sem) if sem and prov.get('label_hash') else None
            r['gold_access'] = prov.get('gold_access')
            r['seed'] = prov.get('seed')
            if not plf and label_file is None and prov.get('label_file'):
                label_file = ctx.path(prov['label_file'])
            for key in ('train_hash_ok', 'eval_hash_ok', 'label_hash_ok'):
                if r[key] is False:
                    M['issues'].append(f'fold {k}: run provenance {key[:-3]} conflicts with the protocol/label registration')
        inv = rp / 'output-inventory.json' if rp is not None else None
        ck = ((read_json(inv).get('model_best.pt') or {}).get('sha256')) if inv is not None and inv.is_file() else None
        r['checkpoint_sha256'] = ck or src.get('checkpoint_sha256')
        if fpath is None or not fpath.is_file():
            r['fold_oof'] = ctx.rel(fpath) if fpath is not None else None
            if fold_quarantined:
                r['quarantine_reason'] = fold_quarantined
            M['gate_failures'].append(f'fold {k}: fold-run OOF file not available, so the pooled OOF is not bound to this fold run')
        elif fold_quarantined:
            r.update(fold_oof=ctx.rel(fpath), quarantine_reason=fold_quarantined)
        elif 'pred' not in M:
            r['fold_oof'] = ctx.rel(fpath)  # metadata only: no comparison against quarantined pooled predictions
        else:
            fp = load_pred(fpath)
            r['fold_oof'] = ctx.rel(fpath)
            r['fold_oof_sha_ok'] = (ctx.sha(fpath) == fsha) if fsha else None
            r['fold_oof_uids_equal_val'] = set(fp.index) == set(f['val'])
            common = fp.index.intersection(M['pred'].index)
            r['fold_oof_max_abs_diff_vs_pooled'] = float(np.abs(fp.loc[common].to_numpy() - M['pred'].loc[common].to_numpy()).max()) if len(common) else None
            if r['fold_oof_uids_equal_val'] is False or r['fold_oof_sha_ok'] is False or (r['fold_oof_max_abs_diff_vs_pooled'] or 0) > 1e-6:
                M['issues'].append(f'fold {k}: fold-run OOF is not bound to the pooled OOF / validation list')
        M['runs'][k] = r
    M['teachers'], M['teacher_binding'] = [], None
    if plf:
        teacher_specs = fold_label_check(ctx, M, ms, plf, res)
    else:
        M['label_file'] = ctx.rel(label_file) if label_file is not None else None
        M['label_file_sha256'] = ctx.sha(label_file) if label_file is not None and label_file.is_file() else None
        if reg and reg.get('path') and ms.get('label_file'):
            regp = ctx.path(reg['path'])
            if regp is None or label_file is None or regp.resolve() != label_file.resolve():
                M['gate_failures'].append(f'spec "label_file" {ms["label_file"]} is not the registered file {reg["path"]} of label set {M["label_set"]}: '
                                          'a spec override cannot replace the label file the fold runs trained on')
        if reg and reg.get('sha256') and M['label_file_sha256'] and reg['sha256'] != M['label_file_sha256']:
            M['issues'].append('label file sha differs from its registration')
            M['gate_failures'].append('label file sha differs from its registration: the lineage check would read another file than the one trained on')
        teacher_specs = teacher_lineage(ctx, M, ms, label_file)
    if depth >= 4 and teacher_specs:
        M['issues'].append('teacher lineage deeper than 4 levels not followed')
        M['gate_failures'].append('teacher lineage deeper than 4 levels: ancestry not verified')
        teacher_specs = []
    for ts in teacher_specs:
        t = resolve_member(ctx, {k: v for k, v in ts.items() if k != 'source'}, depth + 1)
        t['teacher_source'] = ts.get('source', 'spec')
        if not t.get('oof_declared_null') and t.get('oof_sha256') and not t['folds']:
            M['gate_failures'].append(f'teacher {t["name"]}: fold lists not bound, so the studies behind its predictions cannot be traced')
        M['teachers'].append(t)
    for a in ms.get('ancestors') or []:
        if a.get('oof'):
            M['teachers'].append(resolve_member(ctx, {**a, 'name': a.get('name', 'ancestor')}, depth + 1))
        elif a.get('train_uids_file'):
            p = ctx.path(a['train_uids_file'])
            if p is None or not p.is_file():
                M['issues'].append(f'ancestor {a.get("name")}: train_uids_file missing')
                continue
            if p.suffix == '.json':
                d = read_json(p)
                uids = d.get('train', []) if isinstance(d, dict) else d
            else:
                uids = pd.read_csv(p, dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
            M['ancestor_sets'].append(dict(name=a.get('name', 'ancestor'), kind=a.get('kind', 'ancestor'), uids=set(map(str, uids)), source=ctx.rel(p)))
        else:
            M['notes'].append(f'ancestor {a.get("name")} ({a.get("kind", "?")}): {a.get("external", "external, no UID list")}')
    iw = (manifest or {}).get('init_weights')
    ca = (manifest or {}).get('common_args') or {}
    if isinstance(iw, dict):
        M['external_init'] = (f'{iw.get("name")} sha {str(iw.get("sha256"))[:12]} ({iw.get("kaggle_dataset") or iw.get("upstream") or "external"}); '
                              'external pretraining, not trained on competition studies per manifest; not checkable at UID level')
    elif ca.get('pretrained'):
        M['external_init'] = f'{ca.get("pretrained")} {ca.get("arch", "")} pretrained weights (external; not checkable at UID level)'
    if M['oof_gate_failures']:
        M.pop('pred', None)  # a quarantined fold also disqualifies its assembled OOF
    return M


def influence(M, k, memo):
    """UIDs whose labels or images shaped the fold-k model of M: its training UIDs plus, for every
    teacher whose predictions became its targets, the influence of the teacher fold model that
    produced each training study's target (recursively)."""
    key = (M.get('oof_sha256') or M['oof'], k)
    if key in memo:
        return memo[key]
    S = set(M['folds'][k]['train'])
    inf = set(S)
    for t in M['teachers']:
        if not t['folds']:
            continue
        for j in sorted({t['fold_of'][u] for u in S if u in t['fold_of']}):
            inf |= influence(t, j, memo)
    for a in M['ancestor_sets']:
        inf |= a['uids']
    memo[key] = inf
    return inf


def teacher_unbound(M):
    return any((not t['folds']) or t['issues'] or teacher_unbound(t) for t in M['teachers'])


def lineage_report(ctx, M, memo):
    comp_of, comp_members = ctx.components()
    folds, total_v = {}, 0
    agg = dict(direct=0, component=0, indirect=0)
    infl_all = set()
    for k, f in M['folds'].items():
        V, T = set(f['val']), set(f['train'])
        I = influence(M, k, memo)
        infl_all |= I
        direct = V & T
        indirect = (V & I) - direct
        comp = {u for u in V - I if any(w != u and w in I for w in comp_members.get(comp_of.get(u), []))}
        folds[k] = dict(n_train=len(T), n_val=len(V), influence_size=len(I), direct_overlap=len(direct),
                        indirect_overlap=len(indirect), indirect_share=len(indirect) / len(V) if V else None,
                        component_overlap=len(comp), examples=sorted(direct | indirect | comp)[:3])
        total_v += len(V)
        agg['direct'] += len(direct)
        agg['indirect'] += len(indirect)
        agg['component'] += len(comp)
    gates = gate_failures(M)
    if agg['direct']:
        status = 'FAIL_DIRECT'
    elif gates:
        status = 'FAIL_LINEAGE'
    elif M.get('oof_declared_null'):
        status = 'NOT_AVAILABLE'
    elif M['issues'] or not M['folds'] or teacher_unbound(M):
        status = 'INCOMPLETE'
    elif agg['indirect']:
        status = 'FLAG_INDIRECT'
    elif agg['component']:
        status = 'FLAG_COMPONENT'
    else:
        status = 'PASS'
    gold = ctx.gold()
    label_uids = set()
    if M.get('label_file') and ctx.path(M['label_file']).is_file():
        label_uids = set(pd.read_csv(ctx.path(M['label_file']), dtype={'StudyInstanceUID': str}, usecols=['StudyInstanceUID'])['StudyInstanceUID'])
    return dict(name=M['name'], oof=M['oof'], oof_sha256=M.get('oof_sha256'), k_eval=M.get('k_eval'), seed=M.get('seed'),
                label_set=M.get('label_set'), label_file=M.get('label_file'), label_files=M.get('label_files'), label_provenance=M.get('label_provenance'),
                teacher_binding=M.get('teacher_binding'), teacher_source=M.get('teacher_source'),
                protocol=M.get('protocol'), protocol_sha256=M.get('protocol_sha256'), external_init=M.get('external_init'),
                status=status, studies_validated=total_v, direct_overlap=agg['direct'], indirect_overlap=agg['indirect'],
                indirect_share=agg['indirect'] / total_v if total_v else None, component_overlap=agg['component'],
                folds=folds, runs={k: dict(r) for k, r in M['runs'].items()},
                gold58=dict(in_training_influence=len(gold & infl_all), in_label_file=len(gold & label_uids) if label_uids else None,
                            in_oof=len(gold & set(M['pred'].index)) if 'pred' in M else None),
                issues=list(M['issues']), notes=list(M['notes']), gate_failures=gates,
                oof_gate_failures=gate_failures(M, field='oof_gate_failures'),
                oof_declared_null=bool(M.get('oof_declared_null')), oof_reason=M.get('oof_reason'),
                teachers=[lineage_report(ctx, t, memo) for t in M['teachers']],
                ancestors=[dict(name=a['name'], kind=a['kind'], n_uids=len(a['uids']), source=a['source']) for a in M['ancestor_sets']])


def lineage_section(ctx, members):
    memo = {}
    reps = [lineage_report(ctx, M, memo) for M in members]
    status = min((r['status'] for r in reps), key=LINEAGE_ORDER.index) if reps else 'INCOMPLETE'
    gold_train = sum(r['gold58']['in_training_influence'] for r in reps)
    checked = not any(r.get('oof_declared_null') for r in reps)  # a member without OOF has no training UID lists to check
    return dict(status=status, members=reps,
                gold58=dict(policy='report only, never a pass/fail criterion (H-B-26; D-A-002, D-A-049): exposed through earlier selection, 9 MCL positives',
                            n_gold=len(ctx.gold()), in_any_training_influence=gold_train, absent_from_training=(gold_train == 0) if checked else None))


# ----------------------------------------------------------------------------- 2 OOF

def comp_index(ctx, uids):
    comp_of, _ = ctx.components()
    groups = {}
    for i, u in enumerate(uids):
        groups.setdefault(comp_of.get(u, 'singleton:' + u), []).append(i)
    return [np.asarray(groups[c], dtype=np.int64) for c in sorted(groups)]


def score_block(Y, A, B, fold_arr, cidx, draws, seed):
    pa, pb = auc_cols(Y, A), auc_cols(Y, B)
    pooled = dict(n=int(len(Y)), candidate=nanmean(pa), baseline=nanmean(pb))
    pooled['delta'] = pooled['candidate'] - pooled['baseline']
    tg = {t: dict(candidate=float(pa[i]), baseline=float(pb[i]), delta=float(pa[i] - pb[i]), positives=int(Y[:, i].sum())) for i, t in enumerate(TARGETS)}
    if np.array_equal(A, B):
        pooled['ci95'] = [0.0, 0.0]
        for t in TARGETS:
            tg[t]['ci95'] = [0.0, 0.0]
    else:
        rng = np.random.default_rng(seed)
        dm, dtg = [], []
        for _ in range(draws):
            pick = rng.integers(0, len(cidx), len(cidx))
            ii = np.concatenate([cidx[j] for j in pick])
            a, b = auc_cols(Y[ii], A[ii]), auc_cols(Y[ii], B[ii])
            dm.append(nanmean(a) - nanmean(b))
            dtg.append(a - b)
        pooled['ci95'] = [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))]
        dtg = np.asarray(dtg)
        for i, t in enumerate(TARGETS):
            col = dtg[:, i][np.isfinite(dtg[:, i])]
            tg[t]['ci95'] = [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))] if len(col) else None
    pooled['targets'] = tg
    folds = {}
    for k in sorted(set(fold_arr.tolist())):
        m = fold_arr == k
        fa, fb = auc_cols(Y[m], A[m]), auc_cols(Y[m], B[m])
        folds[int(k)] = dict(n=int(m.sum()), candidate=nanmean(fa), baseline=nanmean(fb), delta=nanmean(fa) - nanmean(fb),
                             targets={t: float(fa[i] - fb[i]) for i, t in enumerate(TARGETS)})
    return dict(pooled=pooled, folds=folds, n_folds=len(folds),
                folds_positive=sum(1 for f in folds.values() if f['delta'] > 0), folds_negative=sum(1 for f in folds.values() if f['delta'] < 0),
                targets_up=sum(1 for t in TARGETS if tg[t]['delta'] > 0), targets_down=sum(1 for t in TARGETS if tg[t]['delta'] < 0))


def reference_sets(ctx, members, spec):
    """The three fixed references plus any member label set that is neither one of them nor teacher-derived."""
    refs = {k: dict(v) for k, v in ctx.cfg['references'].items()}
    skipped = {}
    for M in members:
        ls = M.get('label_set')
        if not ls or ls in refs:
            continue
        if M['teachers']:
            skipped[ls] = 'targets contain teacher predictions: scoring against them would be circular'
        elif M.get('label_file') and ctx.path(M['label_file']).is_file():
            refs[ls] = dict(path=M['label_file'], rule='ge0.5', exclude_gold=True, member_label_set=True)
    for ls, v in (spec.get('extra_label_sets') or {}).items():
        refs.setdefault(ls, dict(path=v, rule='ge0.5', exclude_gold=True) if isinstance(v, str) else dict(v))
    return refs, skipped


def direction_consistency(by_label, tol):
    """Improved vs Dread with CI above 0 AND not worse than tol (-0.002) vs flight_v4 and relabel_v1."""
    d = by_label.get('dread', {}).get('pooled')
    f = by_label.get('flight_v4', {}).get('pooled')
    r = by_label.get('relabel_v1', {}).get('pooled')
    if not (d and f and r):
        return dict(consistent=None, line='direction consistency: not computable (a reference is missing)')
    c_d = d['ci95'][0] > 0
    c_f = f['delta'] >= tol - 1e-12
    c_r = r['delta'] >= tol - 1e-12
    ok = bool(c_d and c_f and c_r)
    line = (f'direction consistency: {"CONSISTENT" if ok else "NOT CONSISTENT"} - Dread {d["delta"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] '
            f'({"CI above 0" if c_d else "CI not above 0"}); flight_v4 {f["delta"]:+.4f} ({"ok" if c_f else f"worse than {tol:+.3f}"}); '
            f'relabel_v1 {r["delta"]:+.4f} ({"ok" if c_r else f"worse than {tol:+.3f}"})')
    return dict(consistent=ok, dread_ci_above_0=c_d, flight_not_worse=c_f, relabel_not_worse=c_r, tolerance=tol, line=line, ceiling_note=CEILING_NOTE)


def oof_not_available(members, baseline, missing, failures=None):
    """OOF section for a candidate with a member without OOF: every OOF line reads NOT AVAILABLE (H-B-35)."""
    why = '; '.join(f'{M["name"]}: ' + (f'no OOF by declaration ({M["oof_reason"]})' if M.get('oof_declared_null') and M.get('oof_reason')
                                        else 'oof null without a stated oof_reason' if M.get('oof_declared_null')
                                        else f'OOF file missing ({M.get("oof")})') for M in missing)
    if failures:
        why = 'OOF quarantine gate failed: ' + '; '.join(failures)
    base_set = baseline.get('label_set') or 'dread'
    tot = sum(x['weight'] for x in members) or 1.0
    formula = ('member probabilities' if len(members) == 1 else
               'per-target weighted mean of average-tie percentile ranks over the cohort, weights ' +
               ', '.join(f'{M["name"]} {M["weight"] / tot:.3f}' for M in members))
    return dict(evidence_class=EVIDENCE_CLASS, ceiling_note=CEILING_NOTE, available=False, eligibility=f'NOT AVAILABLE ({why})',
                filter='NOT_AVAILABLE', label_change=any((M.get('label_set') or 'dread') != base_set for M in members),
                direction=dict(consistent=None, line=f'direction consistency: NOT AVAILABLE ({why})'),
                baseline=dict(name=baseline['name'], oof=baseline['oof'], oof_sha256=baseline.get('oof_sha256'), label_set=base_set),
                candidate_formula=formula, n_studies=0, by_label_set={}, label_sets_not_scored={}, gate_failures=failures or [],
                notes=['OOF NOT AVAILABLE: no weak-label OOF evidence exists for this candidate, so it cannot be SUBMIT-ELIGIBLE on OOF evidence; '
                       'the best verdict is PUBLIC-ONLY (a public score answers only its pre-registered question)'])


def oof_section(ctx, members, baseline, lineage, spec):
    failures = [g for M in members + [baseline] for g in gate_failures(M, M['name'] + ': ', field='oof_gate_failures')]
    if failures:
        out = oof_not_available(members, baseline, [], failures=failures)
        if any(gate_failures(t, field='oof_gate_failures') for t in baseline.get('teachers') or []):
            out['baseline']['lineage'] = lineage_report(ctx, baseline, {})
        return out
    if 'pred' not in baseline:
        raise ValueError(f'{baseline["name"]}: OOF not loadable')
    missing = [M for M in members if 'pred' not in M]
    if missing:
        return oof_not_available(members, baseline, missing)
    pop = ctx.population()
    uids = sorted(pop)
    fold_arr = np.asarray([pop[u] for u in uids])
    for M in members + [baseline]:
        if 'pred' not in M:
            raise ValueError(f'{M["name"]}: OOF not loadable')
        if set(M['pred'].index) != set(uids):
            raise ValueError(f'{M["name"]}: OOF UIDs do not equal the {len(uids)}-study screen cohort')
    A = combine([M['pred'].loc[uids] for M in members], [M['weight'] for M in members])
    B = combine([baseline['pred'].loc[uids]], [1.0])
    cidx = comp_index(ctx, uids)
    draws, seed = int(ctx.cfg['draws']), int(ctx.cfg['seed'])
    refs, skipped = reference_sets(ctx, members, spec)
    by_label = {}
    for ls, ref in refs.items():
        p = ctx.path(ref['path'])
        Y, dropped = ctx.labels(p, uids, ref.get('rule', 'ge0.5'), ref.get('exclude_gold', False))
        blk = score_block(Y, A, B, fold_arr, cidx, draws, seed)
        blk.update(label_file=ctx.rel(p), label_file_sha256=ctx.sha(p), labels=RULE_TEXT[ref.get('rule', 'ge0.5')],
                   gold_rows_dropped=dropped if ref.get('exclude_gold') else None, member_label_set=bool(ref.get('member_label_set')))
        by_label[ls] = blk
    base_set = baseline.get('label_set') or 'dread'
    label_change = any((M.get('label_set') or 'dread') != base_set for M in members)
    elig = {'FAIL_DIRECT': 'INVALID (direct train/validation overlap)',
            'FAIL_LINEAGE': 'INVALID (lineage gate failed, fail closed)',
            'INCOMPLETE': 'UNVERIFIED (lineage incomplete)',
            'FLAG_INDIRECT': 'INELIGIBLE (teacher/ancestor lineage exposes validation studies; leaky by construction, D-B-024)',
            'FLAG_COMPONENT': 'INELIGIBLE (validation studies share a component with training studies)',
            'NOT_AVAILABLE': 'NOT AVAILABLE (no OOF)',
            'PASS': 'ELIGIBLE'}[lineage['status']]
    d = by_label['dread']['pooled']
    identical = bool(np.array_equal(A, B))
    if identical:
        filt = 'IDENTICAL_TO_BASELINE'
    elif not elig.startswith('ELIGIBLE') or label_change:
        filt = 'NOT_APPLICABLE'
    elif d['ci95'][1] < 0:
        filt = 'WORSE'
    elif d['ci95'][0] > 0:
        filt = 'BETTER_WEAK_LABEL'
    else:
        filt = 'NOT_SEPARATED'
    direction = direction_consistency(by_label, float(ctx.cfg['direction_tolerance']))
    if identical:
        direction['line'] = 'direction consistency: not applicable (candidate identical to the baseline)'
    elif not elig.startswith('ELIGIBLE'):
        direction['line'] += f' [computed on {elig.split(" ")[0]} OOF: not usable]'
    tot = sum(x['weight'] for x in members)
    formula = ('member probabilities' if len(members) == 1 else
               'per-target weighted mean of average-tie percentile ranks over the cohort, weights ' +
               ', '.join(f'{M["name"]} {M["weight"] / tot:.3f}' for M in members))
    notes = ['pooled OOF mixes five fold models per family; the release averages five models per family, so OOF is not a direct validation of the served ensemble (D-B-030 review)',
             f'per-fold tables use {ctx.cfg["folds_csv"]}; bootstrap: {draws} draws of whole components ({ctx.cfg["components"]}), seed {seed}, '
             'percentile 95% interval of macro(candidate) - macro(baseline), the same draws for every reference']
    if label_change:
        notes.append('label-set change: each reference favours models trained on its own labelling style, so cross-label deltas are not comparable effect sizes')
    if spec.get('weights_selected_on_oof'):
        notes.append('blend weight was selected on this OOF (selection exposure on the same 4,343 studies)')
    return dict(evidence_class=EVIDENCE_CLASS, ceiling_note=CEILING_NOTE, available=True, eligibility=elig, filter=filt, label_change=label_change,
                direction=direction,
                baseline=dict(name=baseline['name'], oof=baseline['oof'], oof_sha256=baseline.get('oof_sha256'), label_set=base_set),
                candidate_formula=formula, n_studies=len(uids), by_label_set=by_label, label_sets_not_scored=skipped, notes=notes)


# ----------------------------------------------------------------------------- 3 replication

def replicate_specs(ctx, M):
    found = []
    seen = {M.get('oof_sha256')}
    for r in M.get('declared_replicates') or []:
        found.append(dict(r, source='spec'))
    results = [(path, d, None) for path, d in ctx.oof_results()] + ctx.oof_results_skipped
    for path, d, reason in results:
        linked = M.get('protocol_sha256') and d.get('seed42_protocol_sha256') == M['protocol_sha256']
        reverse = M.get('seed42_protocol_sha256') and d.get('protocol_sha256') == M['seed42_protocol_sha256']
        if not (linked or reverse):
            continue
        f = path.parent / Path(M['oof']).name
        reason = reason or DQ.quarantine_reason(f, root=ctx.root)
        if reason:
            found.append(dict(oof=ctx.rel(f), kind='seed', label=f'seed {d.get("seed", "?")} ({path.parent.name})',
                              source='auto: seed42_protocol_sha256 link', skip_reason=reason))
            continue
        if f.is_file() and ctx.sha(f) not in seen:
            seen.add(ctx.sha(f))
            found.append(dict(oof=ctx.rel(f), kind='seed', label=f'seed {d.get("seed", "?")} ({path.parent.name})', source='auto: seed42_protocol_sha256 link'))
    return found


def replication_section(ctx, members, baseline, spec):
    failures = [g for M in members + [baseline] for g in gate_failures(M, M['name'] + ': ', field='oof_gate_failures')]
    if failures or any('pred' not in M for M in members + [baseline]):
        return dict(status='NOT_AVAILABLE', candidate_delta_dread=None, replicates=[], seed_replicates=0, split_replicates=0,
                    fold_partitions=sorted({M.get('protocol_folds_sha') for M in members if M.get('protocol_folds_sha')}),
                    note=('OOF quarantine gate failed; replication not scored: ' + '; '.join(failures) if failures else
                          'a member or baseline has no OOF, so no replicate can be scored like the candidate'))
    pop = ctx.population()
    uids = sorted(pop)
    fold_arr = np.asarray([pop[u] for u in uids])
    refs, _ = reference_sets(ctx, members, spec)
    Ys = {ls: ctx.labels(ctx.path(r['path']), uids, r.get('rule', 'ge0.5'), r.get('exclude_gold', False))[0] for ls, r in refs.items()}
    B = combine([baseline['pred'].loc[uids]], [1.0])
    cidx = comp_index(ctx, uids)
    cand = combine([M['pred'].loc[uids] for M in members], [M['weight'] for M in members])
    base_delta = nanmean(auc_cols(Ys['dread'], cand)) - nanmean(auc_cols(Ys['dread'], B))
    reps = []
    for i, M in enumerate(members):
        for r in replicate_specs(ctx, M):
            p = ctx.path(r['oof'])
            reason = r.get('skip_reason') or (DQ.quarantine_reason(p, root=ctx.root) if p is not None else None)
            if reason:
                reps.append(dict(member=M['name'], **r, status='SKIPPED_QUARANTINED', reason=reason))
                continue
            if p is None or not p.is_file():
                reps.append(dict(member=M['name'], **r, status='MISSING'))
                continue
            # resolved like a member, so quarantine follows its fold OOF files, fold run directories and teachers before any
            # prediction is scored (astra re-review 2026-10-08 A3); the rejection reason is kept
            try:
                R = resolve_member(ctx, dict(name=r.get('label') or 'replicate', oof=r['oof']))
            except (OSError, ValueError) as exc:
                reps.append(dict(member=M['name'], **r, status='UNRESOLVABLE', reason=f'{type(exc).__name__}: {exc}'))
                continue
            why = gate_failures(R, field='oof_gate_failures')
            if why:
                reps.append(dict(member=M['name'], **r, status='SKIPPED_QUARANTINED', reason='; '.join(why)))
                continue
            pr = R.get('pred')
            if pr is None:
                reps.append(dict(member=M['name'], **r, status='NOT_LOADED', reason='replicate predictions were not loaded'))
                continue
            if set(pr.index) != set(uids):
                reps.append(dict(member=M['name'], **r, status='COHORT_MISMATCH'))
                continue
            preds = [pr.loc[uids] if j == i else X['pred'].loc[uids] for j, X in enumerate(members)]
            A = combine(preds, [X['weight'] for X in members])
            blk = score_block(Ys['dread'], A, B, fold_arr, cidx, int(ctx.cfg['draws']), int(ctx.cfg['seed']))
            other = {ls: nanmean(auc_cols(Y, A)) - nanmean(auc_cols(Y, B)) for ls, Y in Ys.items() if ls != 'dread'}
            reps.append(dict(member=M['name'], **r, status='OK', oof_sha256=ctx.sha(p), delta=blk['pooled']['delta'],
                             ci95=blk['pooled']['ci95'], folds_positive=blk['folds_positive'], n_folds=blk['n_folds'],
                             delta_other_references=other,
                             sign_agrees=bool(np.sign(blk['pooled']['delta']) == np.sign(base_delta)) if base_delta != 0 else None))
    ok = [r for r in reps if r['status'] == 'OK']
    return dict(status='AVAILABLE' if ok else 'NOT_AVAILABLE', candidate_delta_dread=base_delta, replicates=reps,
                seed_replicates=len([r for r in ok if r.get('kind') == 'seed']), split_replicates=len([r for r in ok if r.get('kind') == 'split']),
                fold_partitions=sorted({M.get('protocol_folds_sha') for M in members if M.get('protocol_folds_sha')}),
                note=('a replicate swaps one member for its replicate and is scored like the candidate (Dread with bootstrap, other references point only); '
                      'the same-machine seed swing is about 0.006 (D-B-017); a member without a replicate is a single run'))


# ----------------------------------------------------------------------------- 4 engineering

CHECKPOINT_FP_BASIS = ('sha256 of the sorted per-arm [sorted served checkpoint sha256 set, served K, blend weight normalised to sum 1 and rounded '
                       'to 9 decimals]; arm names, tags and release names are cosmetic and excluded (astra re-review 2026-10-08 A2)')


def checkpoint_fingerprint(items):
    """Served-model identity from (checkpoint shas, served K, blend weight) per arm; None when any arm reports no checkpoint sha."""
    items = list(items)
    if not items or any(not s for s, _, _ in items):
        return None
    tot = sum(float(w) for _, _, w in items) or 1.0
    canon = sorted(([sorted(set(s)), k, round(float(w) / tot, 9)] for s, k, w in items), key=json.dumps)
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()


def public3_results(ctx, out_dir):
    """(final (path, result), {arm: (path, result)}, every readable result) of a Public3 output directory."""
    results = []
    for p in sorted(out_dir.rglob('result.json')):
        try:
            d = read_json(p)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            results.append((p, d))
    combined = [x for x in results if 'combine' in x[0].parts]
    arms = {d.get('arm'): (p, d) for p, d in results if d.get('arm')}
    final = combined[0] if combined else (next(iter(arms.values())) if len(arms) == 1 else None)
    return final, arms, results


def public3_identity(ctx, out_dir):
    """Served-checkpoint fingerprint, release manifest sha and output sha of a Public3 output directory from its result.json files
    alone (historical submissions): weights are the reported rank weights of a combine result, 1.0 for a single arm."""
    d = ctx.path(out_dir)
    if d is None or not d.is_dir():
        raise ValueError(f'Public3 output directory {out_dir} not found')
    final, arms, _ = public3_results(ctx, d)
    if final is None:
        raise ValueError(f'{out_dir}: no final Public3 result.json (combine result or a single arm)')
    fp, fd = final
    rw = fd.get('rank_weights') if len(arms) > 1 else {a: 1.0 for a in arms}
    problems = []
    if not isinstance(rw, dict) or set(rw) != set(arms):
        problems.append(f'reported weights {rw!r} do not name the arms {sorted(arms)}')
        items = []
    else:
        items = [({c.get('sha256') for c in r.get('checkpoints') or [] if isinstance(c, dict) and c.get('sha256')}, r.get('k_eval'), float(rw[a]))
                 for a, (_, r) in arms.items()]
    sub = d / 'submission.csv'
    return dict(out_dir=ctx.rel(d), final_result=ctx.rel(fp), final_result_sha256=ctx.sha(fp), status=fd.get('status'),
                checkpoint_fingerprint=checkpoint_fingerprint(items) if items else None, checkpoint_fingerprint_basis=CHECKPOINT_FP_BASIS,
                release_manifest_sha256=fd.get('release_manifest_sha256'), submission_sha256=fd.get('submission_sha256'),
                submission_csv_sha256=ctx.sha(sub) if sub.is_file() else None,
                arms={str(a): dict(result=ctx.rel(p), k_eval=r.get('k_eval'), n_checkpoints=len(r.get('checkpoints') or []),
                                   weight=(rw or {}).get(a) if isinstance(rw, dict) else None) for a, (p, r) in arms.items()},
                problems=problems)


def engineering_section(ctx, members, release):
    out = dict(scope=ENGINEERING_SCOPE, scored_prediction_or_score_recovered=False, problems=[])
    out_dir = ctx.path((release or {}).get('out_dir'))
    out['out_dir'] = ctx.rel(out_dir) if out_dir is not None else None
    if out_dir is None or not out_dir.is_dir():
        out.update(status='MISSING', engineering_pass=False, problems=['release Public3 output directory not found'])
        return out
    final, arms, results = public3_results(ctx, out_dir)
    problems = []
    if final is None:
        out.update(status='MISSING' if not results else 'AMBIGUOUS', engineering_pass=False,
                   problems=['no final Public3 result.json' if not results else 'several arm results without a combine result'])
        return out
    fp, fd = final
    out.update(final_result=ctx.rel(fp), status=fd.get('status'), release=fd.get('release'), release_manifest_sha256=fd.get('release_manifest_sha256'),
               submission_sha256=fd.get('submission_sha256'), studies=fd.get('studies'), accuracy_claim=fd.get('accuracy_claim'),
               public3_elapsed_seconds=fd.get('elapsed_seconds'))
    hard = []
    arm_status_ok = all(d.get('status') in PASS_STATUSES for _, d in arms.values())
    if fd.get('status') not in PASS_STATUSES:
        hard.append(f'final status {fd.get("status")} is not a PASS_*_TEST_COHORT_ONLY status')
    if not arm_status_ok:
        hard.append('an arm result is not PASS_TEST_COHORT_ONLY')
    sub = out_dir / 'submission.csv'
    out['submission_csv_sha256'] = ctx.sha(sub) if sub.is_file() else None
    out['csv_matches_receipt'] = (out['submission_csv_sha256'] == fd.get('submission_sha256')) if sub.is_file() else None
    if out['csv_matches_receipt'] is not True:
        hard.append('submission.csv missing or its sha differs from the receipt')
    sample = ctx.path(ctx.cfg['sample_submission'])
    if sub.is_file() and sample is not None and sample.is_file():
        ok, n, err = RP.validate_csv(sub, sample)
        out.update(csv_valid=ok, csv_studies=n, csv_error=err, csv_validator='tools/anchor_receipt_policy.validate_csv')
        if not ok:
            hard.append(f'submission.csv invalid: {err}')
    else:
        out.update(csv_valid=None, csv_error='sample submission or submission.csv unavailable')
        problems.append('CSV validity not checked')
    # arms bound to members (checkpoints, K, weights)
    total_w = sum(M['weight'] for M in members) or 1.0
    expected_w = {M['arm']: M['weight'] / total_w for M in members}
    arm_rep, fp_items = {}, []
    for M in members:
        a = arms.get(M['arm'])
        if a is None:
            arm_rep[M['arm']] = dict(member=M['name'], binding='unknown', note='no Public3 arm result with this name')
            problems.append(f'arm {M["arm"]} has no Public3 result')
            continue
        d = a[1]
        served = {c.get('sha256') for c in d.get('checkpoints') or [] if c.get('sha256')}
        if M.get('oof_declared_null'):  # no OOF fold runs: the spec's checkpoint_sha256 is the only binding (H-B-35)
            trained = set(M.get('declared_checkpoints') or [])
            n_expected, matched = len(trained), 'declared_match'
        else:
            trained = {r.get('checkpoint_sha256') for r in M['runs'].values() if r.get('checkpoint_sha256')}
            n_expected, matched = len(M['folds']), 'full_reported'
        if not served or not trained:
            binding = 'unknown'
            problems.append(f'arm {M["arm"]}: served/trained checkpoint hashes unavailable'
                            + (' (no OOF fold runs: declare "checkpoint_sha256" for this member)' if M.get('oof_declared_null') else ''))
        elif served == trained and len(trained) == n_expected:
            binding = matched
        elif served < trained:
            binding = 'partial_reported'
            hard.append(f'arm {M["arm"]}: serves only part of the OOF fold models (partial_reported)')
        else:
            binding = 'conflicting_evidence'
            hard.append(f'arm {M["arm"]}: served checkpoints are not the OOF fold models (conflicting_evidence)')
        k_ok = (d.get('k_eval') == M.get('k_eval')) if d.get('k_eval') is not None and M.get('k_eval') is not None else None
        if k_ok is False:
            hard.append(f'arm {M["arm"]}: served K{d.get("k_eval")} differs from OOF K{M.get("k_eval")}')
        arm_rep[M['arm']] = dict(member=M['name'], status=d.get('status'), n_models=d.get('n_models'), k_eval=d.get('k_eval'),
                                 k_matches_oof=k_ok, submission_sha256=d.get('submission_sha256'),
                                 checkpoints=[f'{c.get("tag")}:{str(c.get("sha256"))[:12]}' for c in d.get('checkpoints') or []],
                                 binding=binding, elapsed_seconds=d.get('elapsed_seconds'))
        fp_items.append((served, d.get('k_eval'), expected_w[M['arm']]))  # no arm name: cosmetic (A2)
    out['arms'] = arm_rep
    if len(members) > 1:
        rw = fd.get('rank_weights')
        if not isinstance(rw, dict):
            red = 'unknown'
        elif set(rw) == set(expected_w) and all(abs(float(rw[k]) - expected_w[k]) < 1e-9 for k in rw):
            red = 'expected_reported'
        else:
            red = 'changed_reported'
        out.update(rank_weights_reported=rw, rank_weights_expected=expected_w, formula_reported=fd.get('formula'))
    else:
        first = next(iter(arm_rep.values()), {})
        n_exp = len(members[0].get('declared_checkpoints') or []) if members[0].get('oof_declared_null') else len(members[0]['folds'])
        red = 'expected_reported' if first.get('n_models') == n_exp and n_exp else 'unknown'
    out['reduction_state'] = red
    if red == 'changed_reported':
        hard.append('served reduction/weights differ from the screened candidate')
    elif red == 'unknown':
        problems.append('served reduction not reported')
    # served-checkpoint fingerprint only when every member's arm reports its checkpoint shas (an empty one never matches);
    # identity = checkpoint sha set + served K + blend weight per arm, never the arm name (astra re-review 2026-10-08 A2)
    out['checkpoint_fingerprint'] = checkpoint_fingerprint(fp_items) if len(fp_items) == len(members) else None
    out['checkpoint_fingerprint_basis'] = CHECKPOINT_FP_BASIS
    # runtime projection (execution-plan.json of the kernel directory)
    kd = ctx.path((release or {}).get('kernel_dir')) if (release or {}).get('kernel_dir') else (out_dir.parent / out_dir.name[4:] if out_dir.name.startswith('out-') else None)
    plan = kd / 'execution-plan.json' if kd is not None else None
    rt = dict(kernel_dir=ctx.rel(kd) if kd is not None else None)
    if plan is not None and plan.is_file():
        e = read_json(plan)
        s = e.get('screen_at_expected_hidden_N') or {}
        within = (s['projected_seconds'] <= s['internal_seconds']) if s.get('projected_seconds') is not None and s.get('internal_seconds') is not None else None
        rt.update(kernel=e.get('kernel'), studies=s.get('studies'), projected_seconds=s.get('projected_seconds'), budget_seconds=s.get('internal_seconds'),
                  max_studies_passing_screen=e.get('max_studies_passing_screen'), within_budget=within, scope=s.get('scope'))
        if within is False:
            hard.append('runtime projection exceeds the internal budget')
    else:
        rt.update(within_budget=None, note='no execution-plan.json runtime projection found')
    out['runtime'] = rt
    if (release or {}).get('anchor_receipt') or (release or {}).get('anchor_submission'):
        rcp, aud, asub = (ctx.path(release.get(k)) for k in ('anchor_receipt', 'anchor_audit', 'anchor_submission'))
        ok, n, err = RP.validate_csv(asub, sample) if asub is not None and asub.is_file() and sample is not None else (False, 0, 'anchor submission unavailable')
        out['anchor_receipt_policy'] = RP.classify(read_json(rcp) if rcp is not None and rcp.is_file() else None,
                                                   read_json(aud) if aud is not None and aud.is_file() else None, csv_valid=ok, study_count=n)
    out['hard_failures'] = hard
    out['problems'] = hard + problems
    out['engineering_pass'] = not hard
    return out


# ----------------------------------------------------------------------------- 5 decision

def load_submissions(path):
    if path is None or not Path(path).is_file():
        return None
    p = Path(path)
    rows = read_json(p) if p.suffix == '.json' else list(csv.DictReader(p.open(encoding='utf-8', newline='')))
    out = {}
    for r in rows:
        ref = str(r.get('ref') or r.get('submission_id') or '').strip()
        if not ref:
            continue
        pub = r.get('public_score', r.get('publicScore'))
        try:
            pub = float(pub) if pub not in (None, '') else None
        except ValueError:
            pub = None
        out[ref] = dict(ref=ref, date=str(r.get('date', '')), status=str(r.get('status', '')).replace('SubmissionStatus.', ''),
                        public=pub, description=r.get('description', ''))
    return dict(path=p.as_posix(), sha256=sha256_file(p), modified=dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec='seconds'), rows=out)


def evaluate_rules(dec, public):
    ref_pub = (dec.get('reference') or {}).get('public')
    for r in dec.get('rules') or []:
        on = r.get('on', 'public')
        if on == 'delta' and ref_pub is None:
            continue
        val = public if on == 'public' else public - ref_pub
        if OPS[r['op']](val, float(r['value'])):
            return r.get('reading', f'{on} {r["op"]} {r["value"]}')
    return dec.get('otherwise', 'inconclusive')


def rule_text(dec):
    ref = (dec.get('reference') or {}).get('public')
    parts = []
    for r in dec.get('rules') or []:
        if r.get('on') == 'delta':
            parts.append(f'public - {ref} {r["op"]} {r["value"]:+.3f} -> {r.get("reading")}')
        else:
            parts.append(f'public {r["op"]} {r["value"]:.3f} -> {r.get("reading")}')
    if parts:
        parts.append(f'otherwise -> {dec.get("otherwise", "inconclusive")}')
    return '; '.join(parts)


# ----------------------------------------------------------------------------- persistent fingerprint registry (duplicate coverage)

REGISTRY_SCHEMA = 'submission-screen-fingerprints-v1'
HISTORICAL_SCHEMA = 'submission-screen-historical-v1'
# current fingerprint schema; 'served_checkpoints' replaced the arm-named 'release' fingerprint (astra re-review 2026-10-08 A2)
FP_KEYS = ('members', 'served_checkpoints', 'release_manifest', 'public3_submission')
LEGACY_FP_KEYS = ('release',)  # kept in old entries for the record; never matched; an entry without every FP_KEYS key covers nothing
ENTRY_KINDS = ('screen', 'historical')
BASES = ('public', 'oof')
REGISTRY_NOTE = ('persistent duplicate registry of tools/submission_screen.py (astra review 2026-10-08): one entry per screened candidate '
                 '(member-OOF fingerprint, served-checkpoint fingerprint, release manifest sha, Public3 output sha, submission_ref); '
                 'every screen run that writes its report reads it and appends its candidates (atomic replace); a missing or unreadable '
                 'registry makes every screen HOLD (duplicate coverage unavailable); create or extend it only with --seed-registry. '
                 'Coverage (astra re-review 2026-10-08 A1): every non-ERROR cached submission needs an entry with its submission_ref in the '
                 'current fingerprint schema - kind "screen" (bound with --bind-submission) or kind "historical" (--record-historical, '
                 'acknowledged, each unavailable fingerprint with its reason)')


def spec_sha256(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def validate_spec(spec):
    """Spec ingestion checks (fail closed): candidate name, decision object and the decision basis enum (exact lowercase)."""
    if not isinstance(spec, dict):
        raise ValueError('a candidate spec must be a JSON object')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', str(spec.get('candidate', ''))) or str(spec.get('candidate')) in ('.', '..'):
        raise ValueError('candidate name must be [A-Za-z0-9_.-]+')
    dec = spec.get('decision')
    if dec is not None and not isinstance(dec, dict):
        raise ValueError(f'{spec["candidate"]}: "decision" must be an object')
    if isinstance(dec, dict) and 'basis' in dec and (not isinstance(dec['basis'], str) or dec['basis'] not in BASES):
        raise ValueError(f'{spec["candidate"]}: decision.basis {dec["basis"]!r} is not supported; use exactly one of '
                         f'{" | ".join(repr(b) for b in BASES)} (lowercase). "oof+public" and other spellings are rejected')
    if not isinstance(spec.get('members'), list) or not spec['members']:
        raise ValueError(f'{spec["candidate"]}: "members" must be a non-empty list')
    return spec


def fingerprints_of(members, eng):
    return dict(members=member_fingerprint(members), served_checkpoints=eng.get('checkpoint_fingerprint'),
                release_manifest=eng.get('release_manifest_sha256'), public3_submission=eng.get('submission_sha256'))


def registry_path(ctx):
    return ctx.path(ctx.cfg['fingerprint_registry'])


def _registry_entry_ok(e):
    if not (isinstance(e, dict) and isinstance(e.get('candidate'), str) and bool(e['candidate']) and isinstance(e.get('fingerprints'), dict)
            and (e.get('submission_ref') is None or isinstance(e['submission_ref'], str))):
        return False
    kind = e.get('kind', 'screen')
    if kind == 'historical':
        return bool(e.get('submission_ref')) and isinstance(e.get('acknowledged'), bool) and isinstance(e.get('unavailable', {}), dict)
    return kind in ENTRY_KINDS


def entry_covers(e):
    """An entry covers its submission_ref when it is in the current fingerprint schema (every FP_KEYS key present, None allowed for a
    screened candidate) and, for a historical entry, acknowledged with a recorded reason for every unavailable fingerprint."""
    fp = e.get('fingerprints') or {}
    if not e.get('submission_ref') or not all(k in fp for k in FP_KEYS):
        return False
    if e.get('kind') == 'historical':
        why = e.get('unavailable') or {}
        return e.get('acknowledged') is True and all(fp[k] or str(why.get(k) or '').strip() for k in FP_KEYS)
    return e.get('kind', 'screen') == 'screen'


def read_registry(path):
    """The registry document; ValueError when the file is not a valid registry (fail closed)."""
    d = read_json(path)
    if not isinstance(d, dict) or d.get('schema') != REGISTRY_SCHEMA or not isinstance(d.get('entries'), list):
        raise ValueError(f'not a {REGISTRY_SCHEMA} document')
    bad = [i for i, e in enumerate(d['entries']) if not _registry_entry_ok(e)]
    if bad:
        raise ValueError(f'malformed entries {bad[:5]}')
    return d


def load_registry(ctx):
    """dict(path, sha256, available, reason, entries) of the persistent registry; missing or unreadable -> available False (HOLD)."""
    p = registry_path(ctx)
    out = dict(path=ctx.rel(p) if p is not None else None, sha256=None, available=False, reason=None, entries=[])
    if p is None or not p.is_file():
        out['reason'] = f'fingerprint registry {out["path"]} is missing'
        return out
    try:
        d = read_registry(p)
    except (OSError, ValueError) as exc:
        out['reason'] = f'fingerprint registry {out["path"]} is unreadable ({exc})'
        return out
    out.update(sha256=sha256_file(p), available=True, reason='ok', entries=list(d['entries']))
    return out


def registry_entry(spec, fps, source, recorded_at, spec_sha=None):
    """A screened-candidate entry; only the fingerprint keys the report actually carries are recorded (a legacy report without
    'served_checkpoints' gives a legacy entry that covers nothing)."""
    return dict(candidate=str(spec['candidate']), kind='screen', submission_ref=str(spec.get('submission_ref') or '') or None,
                fingerprints={k: fps[k] for k in FP_KEYS + LEGACY_FP_KEYS if k in fps}, recorded_at=recorded_at, source=source,
                spec_sha256=spec_sha)


_ABSENT = object()


def _same_entry(a, b):
    return (a['candidate'] == b['candidate'] and (a.get('submission_ref') or None) == (b.get('submission_ref') or None)
            and a.get('kind', 'screen') == b.get('kind', 'screen')
            and all(a['fingerprints'].get(k, _ABSENT) == b['fingerprints'].get(k, _ABSENT) for k in FP_KEYS + LEGACY_FP_KEYS))


def write_json_atomic(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(doc), indent=1, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    os.replace(tmp, path)


def append_registry(ctx, entries, create=False):
    """Append the entries that are not present yet (same candidate, submission_ref and fingerprints) by an atomic replace.
    A missing registry is created only by seeding (create=True); an unreadable one raises and is never replaced.
    Returns dict(path, appended, created, note)."""
    p = registry_path(ctx)
    created = False
    if p.is_file():
        doc = read_registry(p)
    elif create:
        doc, created = dict(schema=REGISTRY_SCHEMA, note=REGISTRY_NOTE, created_at=dt.datetime.now().astimezone().isoformat(timespec='seconds'), entries=[]), True
    else:
        return dict(path=ctx.rel(p), appended=0, created=False, note='registry missing: not created by a screen run (seed it with --seed-registry)')
    added = 0
    for e in entries:
        if not any(_same_entry(e, x) for x in doc['entries']):
            doc['entries'].append(clean(e))
            added += 1
    if added or created:
        doc['updated_at'] = dt.datetime.now().astimezone().isoformat(timespec='seconds')
        write_json_atomic(p, doc)
    return dict(path=ctx.rel(p), appended=added, created=created,
                note=f'ok: {added} appended, {len(entries) - added} already present (same candidate, submission_ref and fingerprints)')


def seed_registry(ctx, screen_paths):
    """Create or extend the registry from existing screen.json reports (their recorded fingerprints and submission_ref)."""
    entries = []
    for sp in screen_paths:
        sp = ctx.path(str(sp))
        r = read_json(sp)
        if not isinstance(r, dict) or not isinstance(r.get('decision'), dict) or not r.get('candidate'):
            raise ValueError(f'{sp}: not a submission screen report')
        fps = r['decision'].get('fingerprints') or {}
        entries.append(registry_entry(dict(candidate=r['candidate'], submission_ref=r['decision'].get('submission_ref')), fps,
                                      source=f'seed from {ctx.rel(sp)} (sha256 {sha256_file(sp)[:12]}, schema {r.get("schema")})',
                                      recorded_at=r.get('generated_at'), spec_sha=r.get('spec_sha256')))
    return append_registry(ctx, entries, create=True)


def registry_view(ctx, batch=()):
    """The registry a screen checks against: the persistent entries plus the candidates of this run."""
    reg = load_registry(ctx)
    return dict(reg, persistent=len(reg['entries']), batch=len(batch), entries=reg['entries'] + list(batch))


def coverage_section(registry, subs):
    """Duplicate coverage (astra re-review 2026-10-08 A1): a readable registry is reconciled against the submissions cache. Complete
    only when every cached submission that is not ERROR has a covering entry (entry_covers); no cache -> incomplete (fail closed)."""
    reg = registry or {}
    out = dict(available=bool(reg.get('available')), reason=reg.get('reason'), registry=reg.get('path'), registry_sha256=reg.get('sha256'),
               registry_entries=reg.get('persistent', 0), run_candidates=reg.get('batch', 0), cache_given=subs is not None, complete=False,
               incomplete_reason=None, submissions_requiring_coverage=None, covered=None, uncovered=[], uncovered_legacy_only=[],
               acknowledged_historical=[], without_fingerprints=[], cached_scored_submissions_without_registry_entry=None)
    if not out['available']:
        out['incomplete_reason'] = f'registry unavailable ({reg.get("reason")})'
        return out
    covering, legacy = {}, set()
    for e in reg.get('entries') or []:
        ref = e.get('submission_ref')
        if not ref:
            continue
        if entry_covers(e):
            covering.setdefault(ref, []).append(e)
        else:
            legacy.add(ref)
    if subs is None:
        out['incomplete_reason'] = ('no submissions cache given: the registry cannot be reconciled against the submitted models, so a repeat '
                                    'of a submission without a registry entry would go undetected (pass --submissions)')
        return out
    rows = subs.get('rows') or {}
    need = sorted(r for r, row in rows.items() if row.get('status') != 'ERROR')
    unc = [r for r in need if r not in covering]
    out.update(submissions_requiring_coverage=len(need), covered=len(need) - len(unc), uncovered=unc,
               uncovered_legacy_only=[r for r in unc if r in legacy],
               acknowledged_historical=[r for r in need if r in covering and all(e.get('kind') == 'historical' for e in covering[r])],
               without_fingerprints=[r for r in need if r in covering
                                     and not any(any((e.get('fingerprints') or {}).get(k) for k in FP_KEYS) for e in covering[r])],
               cached_scored_submissions_without_registry_entry=sum(1 for r in unc if rows[r].get('status') == 'COMPLETE'), complete=not unc)
    if unc:
        out['incomplete_reason'] = (f'{len(unc)} of {len(need)} cached non-ERROR submission(s) have no registry entry in the current fingerprint schema: '
                                    + ', '.join(f'{r} ({rows[r].get("status") or "?"}{"; legacy entry only" if r in legacy else ""})' for r in unc[:8])
                                    + (f' (+{len(unc) - 8} more)' if len(unc) > 8 else '')
                                    + '; record them with --record-historical or bind the screened candidate with --bind-submission')
    return out


def _inventory_checkpoint(ctx, path):
    p = ctx.path(path)
    d = read_json(p) if p is not None and p.is_file() else None
    sha = ((d.get('model_best.pt') or {}).get('sha256')) if isinstance(d, dict) else None
    if not sha:
        raise ValueError(f'{path}: no output inventory with a model_best.pt sha256')
    return sha


_SHA = re.compile(r'[0-9a-f]{64}')


def _json_value(ctx, ref):
    """Value at the dotted key of a JSON file ({"json": path, "key": "a.0.b"}): historical fingerprints are read from recorded
    artifacts (receipts, manifests, inventories), never typed in."""
    if not isinstance(ref, dict) or not ref.get('json') or not ref.get('key'):
        raise ValueError(f'a JSON field source needs "json" and "key": {ref!r}')
    p = ctx.path(ref['json'])
    if p is None or not p.is_file():
        raise ValueError(f'{ref["json"]} not found')
    v = read_json(p)
    for part in str(ref['key']).split('.'):
        if isinstance(v, list) and part.isdigit() and int(part) < len(v):
            v = v[int(part)]
        elif isinstance(v, dict) and part in v:
            v = v[part]
        else:
            raise ValueError(f'{ref["json"]}: key {ref["key"]} not found')
    return v


def ref_txt(ref):
    return json.dumps(ref) if isinstance(ref, dict) else str(ref)


def historical_fingerprints(ctx, h):
    """(fingerprints, unavailable reasons, sources) of one historical submission, computed only from local artifacts: "members"
    (pooled OOF CSVs + weights), "public3" (the Public3 output directory of the submitted kernel version), "served" (per arm:
    checkpoint shas from output inventories or JSON fields of receipts / asset manifests, K (a number or a JSON field) and blend
    weight), "values" (release_manifest / public3_submission read from JSON fields of receipts) and "files" (shas of the release
    manifest / Public3 submission files; must agree with any value already read). Every fingerprint that stays None needs a reason
    under "unavailable"; a computed one may not also be declared unavailable."""
    ref = h['submission_ref']
    fps, src = {}, {}
    if h.get('members'):
        mm, rows = [], []
        for m in h['members']:
            p = ctx.path(m.get('oof'))
            if p is None or not p.is_file():
                raise ValueError(f'{ref}: member OOF {m.get("oof")} not found')
            s = ctx.sha(p)
            if m.get('expected_sha256') and m['expected_sha256'] != s:
                raise ValueError(f'{ref}: member OOF {m["oof"]} sha {s[:12]} differs from the expected {str(m["expected_sha256"])[:12]}')
            mm.append(dict(oof_sha256=s, weight=float(m.get('weight', 1.0))))
            rows.append(dict(oof=ctx.rel(p), sha256=s, weight=float(m.get('weight', 1.0))))
        fps['members'] = member_fingerprint(mm)
        src['members'] = rows
    if h.get('public3'):
        ident = public3_identity(ctx, h['public3'].get('out_dir'))
        fps.update(served_checkpoints=ident['checkpoint_fingerprint'], release_manifest=ident['release_manifest_sha256'],
                   public3_submission=ident['submission_sha256'])
        src['public3'] = ident
    if h.get('served'):
        items, rows = [], []
        for a in h['served']:
            shas = list(a.get('checkpoints') or []) + [_inventory_checkpoint(ctx, x) for x in a.get('inventories') or []]
            for jref in a.get('json_checkpoints') or []:
                v = _json_value(ctx, jref)
                for x in v if isinstance(v, list) else [v]:
                    s = x.get('sha256') if isinstance(x, dict) else x
                    if not isinstance(s, str) or not _SHA.fullmatch(s):
                        raise ValueError(f'{ref_txt(jref)}: not a checkpoint sha256 ({str(s)[:20]!r})')
                    shas.append(s)
            if any(not isinstance(s, str) or not _SHA.fullmatch(s) for s in shas):
                raise ValueError(f'{ref}: served checkpoints must be sha256 values')
            k = _json_value(ctx, a['k_eval']) if isinstance(a.get('k_eval'), dict) else a.get('k_eval')
            items.append((shas, k, float(a.get('weight', 1.0))))
            rows.append(dict(checkpoints=sorted(shas), k_eval=k, weight=float(a.get('weight', 1.0)), source=a.get('source'),
                             json_checkpoints=a.get('json_checkpoints'), inventories=a.get('inventories'),
                             k_eval_source=a['k_eval'] if isinstance(a.get('k_eval'), dict) else 'declared'))
        served = checkpoint_fingerprint(items)
        if fps.get('served_checkpoints') and served != fps['served_checkpoints']:
            raise ValueError(f'{ref}: the "served" checkpoints differ from the Public3 receipt checkpoints')
        fps['served_checkpoints'] = served
        src['served'] = rows
    for key, vref in (h.get('values') or {}).items():
        if key not in ('release_manifest', 'public3_submission'):
            raise ValueError(f'{ref}: "values" may name release_manifest or public3_submission, not {key}')
        v = _json_value(ctx, vref)
        if not isinstance(v, str) or not _SHA.fullmatch(v):
            raise ValueError(f'{ref_txt(vref)}: {key} is not a sha256')
        if fps.get(key) and fps[key] != v:
            raise ValueError(f'{ref_txt(vref)}: {key} {v[:12]} differs from {fps[key][:12]} (Public3 receipt)')
        fps[key] = v
        src.setdefault('values', {})[key] = dict(vref, value=v)
    for key, path in (h.get('files') or {}).items():
        if key not in ('release_manifest', 'public3_submission'):
            raise ValueError(f'{ref}: "files" may name release_manifest or public3_submission, not {key}')
        p = ctx.path(path)
        if p is None or not p.is_file():
            raise ValueError(f'{ref}: {key} file {path} not found')
        if fps.get(key) and fps[key] != ctx.sha(p):
            raise ValueError(f'{ref}: {key} sha of {path} differs from the value already read ({fps[key][:12]})')
        fps[key] = ctx.sha(p)
        src.setdefault('files', {})[key] = dict(path=ctx.rel(p), sha256=fps[key])
    declared = h.get('unavailable') or {}
    if not isinstance(declared, dict) or set(declared) - set(FP_KEYS):
        raise ValueError(f'{ref}: "unavailable" must map fingerprint keys {FP_KEYS} to reasons')
    why = {}
    for k in FP_KEYS:
        fps.setdefault(k, None)
        reason = str(declared.get(k) or '').strip()
        if fps[k] is None and not reason:
            raise ValueError(f'{ref}: fingerprint {k} could not be computed from the given artifacts and no reason is recorded under "unavailable"')
        if fps[k] is not None and reason:
            raise ValueError(f'{ref}: fingerprint {k} was computed but is also declared unavailable')
        if reason:
            why[k] = reason
    return fps, why, src


def record_historical(ctx, spec_path, subs, now=None):
    """Append acknowledged historical entries (submissions scored before the screen existed) from a submission-screen-historical-v1
    document: every ref must be a non-ERROR submission of the cache and carry a "mapping" (description -> experiments). A ref already
    covered by a screened candidate, or by an identical historical entry, is skipped; a different historical entry for it is refused."""
    now = now or dt.datetime.now().astimezone().isoformat(timespec='seconds')
    sp = ctx.path(str(spec_path))
    doc = read_json(sp)
    if not isinstance(doc, dict) or doc.get('schema') != HISTORICAL_SCHEMA or not isinstance(doc.get('submissions'), list):
        raise ValueError(f'{spec_path}: not a {HISTORICAL_SCHEMA} document with a "submissions" list')
    p = registry_path(ctx)
    if p is None or not p.is_file():
        raise ValueError(f'fingerprint registry {ctx.rel(p) if p else None} is missing: seed it first (--seed-registry)')
    reg = read_registry(p)
    rows = (subs or {}).get('rows') or {}
    names = [h.get('candidate') if isinstance(h, dict) else None for h in doc['submissions']]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', n) for n in names):
        raise ValueError('historical candidates need unique names matching [A-Za-z0-9_.-]+')
    entries, skipped = [], []
    for h in doc['submissions']:
        ref = str(h.get('submission_ref') or '')
        row = rows.get(ref)
        if row is None:
            raise ValueError(f'submission {ref!r} is not in the submissions cache {(subs or {}).get("path")}')
        if row['status'] == 'ERROR':
            raise ValueError(f'submission {ref} is ERROR: nothing was scored, so there is nothing to cover')
        if not str(h.get('mapping') or '').strip():
            raise ValueError(f'{ref}: "mapping" (description -> experiments) is required')
        fps, why, src = historical_fingerprints(ctx, h)
        e = dict(candidate=h['candidate'], kind='historical', submission_ref=ref, acknowledged=True, fingerprints=fps, unavailable=why,
                 mapping=h['mapping'], experiments=list(h.get('experiments') or []), notes=list(h.get('notes') or []),
                 submission=dict(status=row['status'], date_utc=row['date'], public=row['public'], description=row['description']),
                 fingerprint_sources=src, recorded_at=now,
                 source=f'record-historical {ctx.rel(sp)} (sha256 {sha256_file(sp)[:12]}) against submissions cache {subs["path"]} '
                        f'(sha256 {subs["sha256"][:12]})', spec_sha256=spec_sha256(h))
        same_ref = [x for x in reg['entries'] if x.get('submission_ref') == ref]
        screened = [x for x in same_ref if x.get('kind', 'screen') == 'screen' and entry_covers(x)]
        hist = [x for x in same_ref if x.get('kind') == 'historical']
        if screened:
            skipped.append(dict(ref=ref, why=f'already covered by screened candidate {screened[-1]["candidate"]}'))
        elif any(_same_entry(clean(e), x) for x in hist):
            skipped.append(dict(ref=ref, why='identical historical entry already recorded'))
        elif hist:
            raise ValueError(f'{ref}: a different historical entry ({hist[-1]["candidate"]}) is already recorded; resolve it by hand')
        else:
            entries.append(e)
    rec = append_registry(ctx, entries) if entries else dict(path=ctx.rel(p), appended=0, created=False, note='nothing to append')
    cov = coverage_section(registry_view(ctx), subs)
    return dict(rec, recorded=[dict(ref=e['submission_ref'], candidate=e['candidate'], unavailable=sorted(e['unavailable']),
                                    fingerprints={k: (v[:12] if v else None) for k, v in e['fingerprints'].items()}) for e in entries],
                skipped=skipped, coverage=dict(complete=cov['complete'], uncovered=cov['uncovered'], incomplete_reason=cov['incomplete_reason'],
                                               acknowledged_historical=len(cov['acknowledged_historical']), without_fingerprints=cov['without_fingerprints']))


def bind_submission(ctx, candidate, ref, subs, index=None, now=None):
    """After submitting a screened candidate: append a copy of its newest unbound current-schema registry entry (or entry #index)
    with submission_ref REF, so the submission is covered and later screens see it as submitted. REF must be in the submissions
    cache; a ref bound to another candidate, or several different current entries without --bind-index, are refused."""
    now = now or dt.datetime.now().astimezone().isoformat(timespec='seconds')
    p = registry_path(ctx)
    if p is None or not p.is_file():
        raise ValueError(f'fingerprint registry {ctx.rel(p) if p else None} is missing: seed it first (--seed-registry)')
    entries = read_registry(p)['entries']
    ref = str(ref)
    row = ((subs or {}).get('rows') or {}).get(ref)
    if row is None:
        raise ValueError(f'submission {ref} is not in the submissions cache {(subs or {}).get("path")}: refresh the cache '
                         '(kaggle competitions submissions --csv) after submitting')
    other = sorted({x['candidate'] for x in entries if x.get('submission_ref') == ref and x['candidate'] != candidate})
    if other:
        raise ValueError(f'submission {ref} is already bound to {", ".join(other)}')
    pool = [(i, x) for i, x in enumerate(entries) if x['candidate'] == candidate and not x.get('submission_ref')
            and x.get('kind', 'screen') == 'screen' and all(k in (x.get('fingerprints') or {}) for k in FP_KEYS)]
    if index is not None:
        pool = [(i, x) for i, x in pool if i == index]
        if not pool:
            raise ValueError(f'registry entry #{index} is not an unbound current-schema screen entry of {candidate}')
    elif not pool:
        raise ValueError(f'no unbound registry entry of {candidate} in the current fingerprint schema: screen the candidate with a writing run first')
    distinct = {json.dumps({k: x['fingerprints'].get(k) for k in FP_KEYS}, sort_keys=True) for _, x in pool}
    if len(distinct) > 1:
        raise ValueError(f'{candidate} has {len(distinct)} different current fingerprint sets (entries {[i for i, _ in pool]}): '
                         'choose the submitted one with --bind-index')
    i, src = pool[-1]
    new = dict(candidate=candidate, kind='screen', submission_ref=ref, fingerprints=dict(src['fingerprints']), recorded_at=now,
               source=f'bind {now}: submission {ref} ({row["status"]}, {row["date"]} UTC) bound to registry entry #{i} ({src.get("source")})',
               spec_sha256=src.get('spec_sha256'),
               bound_from=dict(index=i, recorded_at=src.get('recorded_at'), source=src.get('source'), spec_sha256=src.get('spec_sha256')),
               submission=dict(status=row['status'], date_utc=row['date'], public=row['public'], description=row['description']),
               submissions_cache=dict(path=subs['path'], sha256=subs['sha256']))
    rec = append_registry(ctx, [new])
    return dict(rec, bound=dict(candidate=candidate, ref=ref, from_entry=i, fingerprints={k: (v[:12] if v else None) for k, v in new['fingerprints'].items()}))


def _merge_items(items):
    out = {}
    for it in items:
        k = (it['ref'], it['candidate'])
        if k in out:
            out[k]['matched'] = sorted(set(out[k]['matched']) | set(it['matched']), key=FP_KEYS.index)
        else:
            out[k] = dict(it)
    return list(out.values())


def duplicate_check(spec, fps, subs, registry, own_date):
    """Compare this candidate's fingerprints with every registry entry except itself (same submission_ref, or the same name
    without a submission). Scored (COMPLETE) match -> duplicate; submitted match whose status the cache cannot confirm -> unverified
    (HOLD); match with a screened candidate that has no recorded submission -> caveat; ERROR -> resubmission note; Public3 sha alone
    -> warning."""
    own_ref = str(spec.get('submission_ref') or '') or None
    rows = (subs or {}).get('rows', {})
    res = dict(duplicate_of=[], duplicates_scored_later=[], public3_hash_only_matches=[], duplicates_unverified=[],
               matches_unsubmitted=[], matches_errored=[])
    for e in (registry or {}).get('entries') or []:
        eref = e.get('submission_ref') or None
        if (eref and eref == own_ref) or (not eref and e['candidate'] == spec['candidate']):
            continue
        same = [k for k in FP_KEYS if fps.get(k) and fps[k] == (e.get('fingerprints') or {}).get(k)]
        if not same:
            continue
        row = rows.get(eref) if eref else None
        item = dict(ref=eref, candidate=e['candidate'], public=(row or {}).get('public'), status=(row or {}).get('status'), matched=same)
        if same == ['public3_submission']:  # three-study output hash alone: a warning, not a duplicate (H-B-35)
            if row is not None and row['status'] == 'COMPLETE':
                res['public3_hash_only_matches'].append(item)
            continue
        if not eref:
            res['matches_unsubmitted'].append(item)
        elif row is None:
            res['duplicates_unverified'].append(dict(item, why=f'submission {eref} is not in the cached list (pending or newer than the cache)'
                                                     if subs else f'no submissions cache given: the status of submission {eref} is unknown'))
        elif row['status'] == 'COMPLETE':
            res['duplicates_scored_later' if own_date and row['date'] >= own_date else 'duplicate_of'].append(item)
        elif row['status'] == 'ERROR':
            res['matches_errored'].append(item)
        else:
            res['duplicates_unverified'].append(dict(item, why=f'submission {eref} has status {row["status"] or "unknown"} (not final)'))
    return {k: _merge_items(v) for k, v in res.items()}


def decision_section(spec, eng, members, subs, registry, now=None):
    now = now or dt.datetime.now().astimezone()
    dec = dict(spec.get('decision') or {})
    role = dec.get('role', 'test' if dec.get('rules') else 'none')
    claimed = bool(dec.get('rules')) or (role == 'control' and bool(dec.get('controls_for')))
    ref = str(spec.get('submission_ref') or '') or None
    cached = (subs or {}).get('rows', {}).get(ref) if ref else None
    reg = registration_check(dec, cached, ref, now, cache_given=subs is not None) if claimed else None
    pre = bool(claimed and reg['valid'])
    out = dict(plan_item=dec.get('plan_item'), role=role, basis=dec.get('basis', 'public'), basis_declared='basis' in dec, text=dec.get('text'),
               pre_registered=pre, pre_registration_claimed=claimed, registration=reg,
               registration_status=(reg['status'] if reg else 'NOT_CLAIMED'), rules=rule_text(dec), controls_for=dec.get('controls_for'),
               reference=dec.get('reference'), registered_at=dec.get('registered_at'), submission_ref=ref,
               submissions_cache=(subs or {}).get('path'), submissions_cache_sha256=(subs or {}).get('sha256'),
               submissions_cache_modified=(subs or {}).get('modified'))
    if ref and cached:
        out['cached'] = cached
        if cached['status'] == 'COMPLETE' and cached['public'] is not None:
            out['state'] = 'SCORED'
            out['outcome'] = evaluate_rules(dec, cached['public']) if dec.get('rules') else ('control score recorded' if role == 'control' else 'no pre-registered reading')
        else:
            out['state'] = f'SUBMITTED ({cached["status"] or "unknown"})'
    elif ref:
        out['state'] = 'SUBMITTED (not in the cached list; pending or newer than the cache)' if subs else 'SUBMITTED (no submissions cache given)'
    else:
        out['state'] = 'OPEN' if pre else 'REGISTRATION_NOT_VERIFIED' if claimed else 'NONE'
    fps = fingerprints_of(members, eng)
    out['fingerprints'] = fps
    registry = registry if registry is not None else dict(available=False, reason='no registry given', entries=[], persistent=0, batch=0)
    out.update(duplicate_check(spec, fps, subs, registry, (cached or {}).get('date')))
    cov = out['duplicate_coverage'] = coverage_section(registry, subs)
    out['duplicate_check_basis'] = (
        'duplicate = equal member-OOF fingerprint (OOF shas + weights), served-checkpoint fingerprint (checkpoint sha set + served K + blend weight '
        'per arm; arm names excluded) or release manifest sha with a submission that is COMPLETE in the cached list, checked against the persistent '
        f'registry {registry.get("path")} ({registry.get("persistent", 0)} entries) plus the {registry.get("batch", 0)} candidate(s) of this run; a match '
        'with a submitted candidate whose status the cache cannot confirm is HOLD; a match with a screened candidate without a recorded submission_ref '
        'is a caveat; an equal three-study Public3 output sha alone is only a warning'
        + ((f'; coverage COMPLETE: all {cov["submissions_requiring_coverage"]} non-ERROR cached submissions have a current registry entry '
            f'({len(cov["acknowledged_historical"])} acknowledged historical, {len(cov["without_fingerprints"])} without any fingerprint)')
           if cov['complete'] else f'; coverage INCOMPLETE: {cov["incomplete_reason"]}')
        if registry.get('available') else f'UNAVAILABLE: {registry.get("reason")}')
    if not claimed:
        line = f'no pre-registered decision ({dec.get("text") or "none recorded"}): a public score would be measurement-only'
    elif not pre:
        what = rule_text(dec) if dec.get('rules') else 'control of ' + '; '.join(dec.get('controls_for') or [])
        line = (f'{dec.get("plan_item") or "?"}: registration NOT VERIFIED ({reg["reason"]}), so these rules are not admitted as pre-registered '
                f'and a public score would be measurement-only: {what}')
    elif role == 'control':
        line = f'{dec.get("plan_item") or "?"} control: its public score is the reference of {"; ".join(dec.get("controls_for") or [])}'
    else:
        line = f'{dec.get("plan_item") or "?"}: {rule_text(dec)}'
    if out.get('state') == 'SCORED':
        line += f' | scored {cached["public"]:.3f} as {ref}: {out["outcome"]}'
    elif ref:
        line += f' | {out["state"]} as {ref}'
    if out['duplicate_of']:
        line += ' | DUPLICATE of ' + ', '.join(f'{d["ref"]} ({d["candidate"]})' for d in out['duplicate_of'])
    if out['duplicates_unverified']:
        line += ' | POSSIBLE DUPLICATE (status unverified) of ' + ', '.join(f'{d["ref"]} ({d["candidate"]})' for d in out['duplicates_unverified'])
    if not registry.get('available'):
        line += f' | duplicate coverage UNAVAILABLE ({registry.get("reason")})'
    elif not cov['complete']:
        line += ' | duplicate coverage INCOMPLETE' + (f' ({len(cov["uncovered"])} uncovered submission(s))' if cov['uncovered'] else ' (no submissions cache)')
    if out['public3_hash_only_matches']:
        line += ' | WARNING: same Public3 output sha as ' + ', '.join(f'{d["ref"]} ({d["candidate"]})' for d in out['public3_hash_only_matches']) + ' (not a duplicate)'
    out['line'] = line
    return out


# ----------------------------------------------------------------------------- verdict, report

def verdict_section(spec, lin, oof, rep, eng, dec, is_baseline):
    blockers, holds, caveats = [], [], []
    baseline_lineage = oof['baseline'].get('lineage') or {}
    if baseline_lineage.get('gate_failures'):
        blockers.append('baseline lineage gate failed (' + baseline_lineage['status'] + '): '
                        + '; '.join(baseline_lineage['gate_failures']))
    st = lin['status']
    if st == 'FAIL_DIRECT':
        blockers.append('direct train/validation UID overlap: OOF invalid, fix the protocol first')
    elif st == 'FAIL_LINEAGE':
        fails = [g for m in lin['members'] for g in m.get('gate_failures') or []]
        blockers.append('lineage gate failed (fail closed): ' + '; '.join(fails[:3]) + (f' (+{len(fails) - 3} more)' if len(fails) > 3 else ''))
    elif st == 'INCOMPLETE':
        issues = [i for m in lin['members'] for i in m['issues']][:2]
        holds.append('leakage lineage incomplete' + (': ' + '; '.join(issues) if issues else ''))
    elif st in ('FLAG_INDIRECT', 'FLAG_COMPONENT'):
        if dec['basis'] == 'oof':
            blockers.append('decision rests on OOF, but its OOF is ineligible (lineage exposure)')
        else:
            caveats.append('OOF ineligible (teacher/ancestor lineage exposes its validation studies); decide on public only')
    oof_available = oof.get('available', True)
    public_basis = dec['basis'] == 'public' and bool(dec.get('basis_declared'))
    if oof.get('gate_failures') and not public_basis:
        blockers.append('OOF quarantine gate failed; decision requires explicit public basis: ' + '; '.join(oof['gate_failures']))
    if not oof_available:
        if dec['basis'] == 'oof':
            blockers.append('decision rests on OOF, but the OOF is NOT AVAILABLE')
        elif public_basis:
            caveats.append(f'OOF {oof["eligibility"]}: public-only evidence')
        else:
            holds.append(f'OOF {oof["eligibility"]} and the decision does not declare "basis": "public": PUBLIC-ONLY needs an explicit public basis')
    if oof['filter'] == 'WORSE':
        holds.append('weak-label OOF below the baseline on Dread (bootstrap CI < 0)')
    if oof['label_change']:
        caveats.append('label-set change: cross-label OOF deltas are weak-label evidence only')
    if not is_baseline and oof['eligibility'].startswith('ELIGIBLE') and oof['direction'].get('consistent') is False:
        caveats.append('direction not consistent across Dread / flight_v4 / relabel_v1')
    if rep['status'] == 'NOT_AVAILABLE' and not is_baseline:
        caveats.append('no seed/split replicate (single run per member)')
    if spec.get('weights_selected_on_oof'):
        caveats.append('blend weight selected on the same OOF')
    if eng.get('status') in ('MISSING', 'AMBIGUOUS', None):
        holds.append('no Public3 PASS receipt')
    elif not eng.get('engineering_pass'):
        blockers.append('engineering: ' + '; '.join(eng.get('hard_failures') or ['not PASS']))
    elif eng.get('problems'):
        holds.append('engineering evidence incomplete: ' + '; '.join(eng['problems']))
    if eng.get('engineering_pass') and (eng.get('runtime') or {}).get('within_budget') is None:
        caveats.append('no hidden-runtime projection')
    if not dec['pre_registered']:
        if dec.get('pre_registration_claimed'):
            holds.append(f'decision not admitted as pre-registered (fail closed, {dec.get("registration_status")}): '
                         f'{(dec.get("registration") or {}).get("reason")}')
        else:
            holds.append('no pre-registered decision: measurement-only submission')
    if dec['duplicate_of']:
        blockers.append('duplicates scored submission ' + ', '.join(d['ref'] for d in dec['duplicate_of']))
    cov = dec.get('duplicate_coverage') or {}
    if not cov.get('available'):
        holds.append(f'duplicate coverage unavailable: {cov.get("reason") or "no fingerprint registry"}')
    elif not cov.get('complete'):
        holds.append(f'duplicate coverage incomplete: {cov.get("incomplete_reason")}')
    elif cov.get('without_fingerprints'):
        caveats.append(f'{len(cov["without_fingerprints"])} submission(s) are covered only by acknowledged entries without any fingerprint '
                       f'({", ".join(cov["without_fingerprints"])}): a repeat of those models cannot be detected by fingerprint')
    if dec.get('duplicates_unverified'):
        holds.append('possible duplicate not verifiable: ' + '; '.join(f'{d["ref"]} ({d["candidate"]}, {"/".join(d["matched"])}): {d["why"]}'
                                                                   for d in dec['duplicates_unverified']))
    if dec.get('matches_unsubmitted'):
        caveats.append('same fingerprint as screened candidate(s) without a recorded submission_ref: '
                       + ', '.join(f'{d["candidate"]} ({"/".join(d["matched"])})' for d in dec['matches_unsubmitted'])
                       + '; if one was submitted, this is a duplicate (record its submission_ref by screening it with the ref)')
    if dec.get('matches_errored'):
        caveats.append('same fingerprint as failed (ERROR) submission(s) ' + ', '.join(f'{d["ref"]} ({d["candidate"]})' for d in dec['matches_errored'])
                       + ': a resubmission, not a duplicate of a score')
    if dec.get('public3_hash_only_matches'):
        caveats.append('WARNING: Public3 three-study output sha equals scored submission ' + ', '.join(f'{d["ref"]} ({d["candidate"]})' for d in dec['public3_hash_only_matches'])
                       + ', but member OOF, served checkpoints and release manifest differ: not a duplicate')
    v = 'BLOCK' if blockers else 'HOLD' if holds else 'SUBMIT-ELIGIBLE' if oof_available else 'PUBLIC-ONLY' if public_basis else 'HOLD'
    blk = oof['by_label_set'].get('dread')
    d = blk['pooled'] if blk else None
    elig = oof['eligibility'].split(' ')[0].lower()
    if not oof_available:
        oof_txt = f'OOF {oof["eligibility"]}'
    elif is_baseline:
        oof_txt = f'OOF = baseline ({d["candidate"]:.4f} Dread)'
    else:
        others = ', '.join(f'{ls} {oof["by_label_set"][ls]["pooled"]["delta"]:+.4f}' for ls in ('flight_v4', 'relabel_v1') if ls in oof['by_label_set'])
        if not oof['eligibility'].startswith('ELIGIBLE'):
            dir_txt = 'direction not usable (OOF ineligible)'
        else:
            dir_txt = 'direction ' + ('consistent' if oof['direction'].get('consistent') else 'not consistent') + (' (label change)' if oof['label_change'] else '')
        oof_txt = (f'OOF Dread {d["delta"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] vs {oof["baseline"]["name"]} '
                   f'({blk["folds_positive"]}/{blk["n_folds"]} folds up), {others}; {elig}{", label change" if oof["label_change"] else ""}; {dir_txt}')
    rep_txt = f'replication {rep["status"].lower().replace("_", " ")}' + (f' ({rep["seed_replicates"]} seed)' if rep['seed_replicates'] else '')
    eng_txt = f'Public3 {eng.get("status") or "missing"}' + (f' {str(eng.get("submission_sha256"))[:8]}' if eng.get('submission_sha256') else '')
    line = f'{spec["candidate"]}: {v} - leakage {st}; {oof_txt}; {rep_txt}; {eng_txt}; decision {dec["line"]}'
    if dec.get('state') == 'SCORED' or str(dec.get('state', '')).startswith('SUBMITTED'):
        line += f' [retrospective: the screen as it would have read before {dec["submission_ref"]}]'
    return dict(verdict=v, blockers=blockers, holds=holds, caveats=caveats, line=line,
                policy='the number of remaining submissions is never a reason to submit (H-B-26); OOF filters, public decides only pre-registered questions'
                + ('; PUBLIC-ONLY = no OOF evidence exists (declared "oof": null), so only the pre-registered public question can be answered' if v == 'PUBLIC-ONLY' else ''))


def f4(x, sign=True):
    return '-' if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f'{x:+.4f}' if sign else f'{x:.4f}')


def ci(c):
    return '-' if not c else f'[{c[0]:+.4f}, {c[1]:+.4f}]'


def render_md(r):
    L = [f'# Submission screen: {r["candidate"]}', '']
    if r.get('title'):
        L += [r['title'], '']
    L += [f'**Verdict: {r["verdict"]["verdict"]}**', '', r['verdict']['line'], '']
    for k in ('blockers', 'holds', 'caveats'):
        if r['verdict'][k]:
            L.append(f'- {k}: ' + '; '.join(r['verdict'][k]))
    L += ['', f'Generated {r["generated_at"]} by tools/submission_screen.py ({SCHEMA}); spec sha {r["spec_sha256"][:12]}. {r["verdict"]["policy"]}.', '']
    lin = r['lineage']
    L += ['## 1. Leakage lineage', '', f'Status **{lin["status"]}**. Gold58: {lin["gold58"]["in_any_training_influence"]} of {lin["gold58"]["n_gold"]} in any training influence '
          f'({"not checked: a member declares no OOF" if lin["gold58"]["absent_from_training"] is None else "absent" if lin["gold58"]["absent_from_training"] else "PRESENT"}); {lin["gold58"]["policy"]}.', '',
          '| member | OOF | label set | status | validated | direct | indirect (share) | component | gold in train |', '|---|---|---|---|---|---|---|---|---|']

    def rows(m, prefix=''):
        src = f'no OOF ({m.get("oof_reason") or "no oof_reason stated"})' if m.get('oof_declared_null') else f'{m["oof"]} K{m["k_eval"]}'
        out = [f'| {prefix}{m["name"]} | {src} | {m["label_set"]} | {m["status"]} | {m["studies_validated"]} | {m["direct_overlap"]} | '
               f'{m["indirect_overlap"]} ({(m["indirect_share"] or 0):.0%}) | {m["component_overlap"]} | {m["gold58"]["in_training_influence"]} |']
        for t in m['teachers']:
            out += rows(t, prefix + 'teacher of ' + m['name'] + ': ')
        return out
    for m in lin['members']:
        L += rows(m)
    L.append('')
    for m in lin['members']:
        runs_ok = bool(m['runs']) and all(x.get('train_hash_ok') and x.get('eval_hash_ok') for x in m['runs'].values())
        L.append(f'- {m["name"]}: protocol {m["protocol"]} ({str(m["protocol_sha256"])[:12]}); fold runs bound by provenance hashes: '
                 f'{"n/a (no OOF)" if m.get("oof_declared_null") else "yes" if runs_ok else "NO"}; '
                 f'labels: {m["label_provenance"]}; init: {m["external_init"] or "-"}' + (f'; issues: {"; ".join(m["issues"])}' if m['issues'] else '')
                 + (f'; GATE FAILURES (fail closed): {"; ".join(m["gate_failures"])}' if m.get('gate_failures') else '')
                 + (f'; OOF GATE FAILURES: {"; ".join(m["oof_gate_failures"])}' if m.get('oof_gate_failures') else ''))
        for t in m['teachers']:
            L.append(f'  - teacher {t["name"]}: {t["oof"]} K{t["k_eval"]} ({t["status"]} as its own OOF); through its fold models '
                     f'{m["indirect_overlap"]} of {m["studies_validated"]} student validation studies were in the training of a teacher whose outputs became student targets')
    oof = r['oof']
    L += ['', '## 2. OOF (weak-label, three report-derived references)', '', f'{oof["evidence_class"]}. {oof["ceiling_note"]}.', '',
          f'Eligibility: **{oof["eligibility"]}**; filter: {oof["filter"]}; candidate = {oof["candidate_formula"]}; baseline = {oof["baseline"]["name"]} ({oof["baseline"]["oof"]}).', '',
          f'**{oof["direction"]["line"]}**', '']
    if oof.get('available', True):
        L += ['| reference | labels | candidate | baseline | delta | 95% CI | folds up | targets up |', '|---|---|---|---|---|---|---|---|']
    else:
        L.append(f'**OOF {oof["eligibility"]}**: every OOF line is NOT AVAILABLE.')
    for ls, b in oof['by_label_set'].items():
        p = b['pooled']
        L.append(f'| {ls} | {b["labels"]}{" (Gold rows dropped: %s)" % b["gold_rows_dropped"] if b.get("gold_rows_dropped") is not None else ""} | {f4(p["candidate"], False)} | '
                 f'{f4(p["baseline"], False)} | {f4(p["delta"])} | {ci(p["ci95"])} | {b["folds_positive"]}/{b["n_folds"]} | {b["targets_up"]}/12 |')
    L.append('')
    for ls, b in oof['by_label_set'].items():
        p = b['pooled']
        L += [f'### {ls} ({b["label_file"]})', '', '| fold | n | candidate | baseline | delta |', '|---|---|---|---|---|']
        L += [f'| {k} | {f["n"]} | {f4(f["candidate"], False)} | {f4(f["baseline"], False)} | {f4(f["delta"])} |' for k, f in b['folds'].items()]
        L += ['', '| target | positives | candidate | baseline | delta | 95% CI |', '|---|---|---|---|---|---|']
        L += [f'| {t} | {x["positives"]} | {f4(x["candidate"], False)} | {f4(x["baseline"], False)} | {f4(x["delta"])} | {ci(x["ci95"])} |' for t, x in p['targets'].items()]
        L.append('')
    for k, v in oof['label_sets_not_scored'].items():
        L.append(f'- label set {k} not scored: {v}')
    L += [f'- {n}' for n in oof['notes']]
    rep = r['replication']
    L += ['', '## 3. Replication', '', f'Status **{rep["status"]}** ({rep["seed_replicates"]} seed, {rep["split_replicates"]} split; fold partitions {rep["fold_partitions"]}). {rep["note"]}.', '']
    for x in rep['replicates']:
        L.append(f'- {x["member"]} -> {x.get("label")} ({x["oof"]}, {x["source"]}): ' +
                 (f'Dread delta {f4(x["delta"])} {ci(x["ci95"])}, folds up {x["folds_positive"]}/{x["n_folds"]}, sign agrees with the candidate: {x["sign_agrees"]}; '
                  + ', '.join(f'{k} {f4(v)}' for k, v in (x.get('delta_other_references') or {}).items()) if x['status'] == 'OK' else
                  x['status'] + (': ' + x['reason'] if x.get('reason') else '')))
    if not rep['replicates']:
        L.append('- none declared or found')
    e = r['engineering']
    L += ['', '## 4. Engineering', '', f'{e["scope"]}.', '', f'- Public3: status **{e.get("status")}** ({e.get("final_result")}), output sha {e.get("submission_sha256")}, '
          f'submission.csv matches receipt: {e.get("csv_matches_receipt")}, CSV valid: {e.get("csv_valid")} ({e.get("csv_error") or "ok"}); engineering pass: **{e.get("engineering_pass")}**']
    for a, x in (e.get('arms') or {}).items():
        L.append(f'- arm {a}: {x.get("status")}, {x.get("n_models")} models, K{x.get("k_eval")} (OOF K match {x.get("k_matches_oof")}), served checkpoints bound to the OOF fold runs: {x.get("binding")}')
    L.append(f'- reduction: {e.get("reduction_state")}' + (f' (reported {e.get("rank_weights_reported")}, expected {e.get("rank_weights_expected")})' if e.get('rank_weights_reported') is not None else ''))
    rt = e.get('runtime') or {}
    L.append(f'- runtime projection: {rt.get("projected_seconds"):.0f} s for {rt.get("studies")} studies vs budget {rt.get("budget_seconds")} s, within budget {rt.get("within_budget")}, '
             f'max studies passing {rt.get("max_studies_passing_screen")}; Public3 elapsed {e.get("public3_elapsed_seconds")} s ({rt.get("scope")})'
             if rt.get('projected_seconds') is not None else f'- runtime projection: {rt.get("note", "-")}')
    if e.get('problems'):
        L.append('- problems: ' + '; '.join(e['problems']))
    d = r['decision']
    reg = d.get('registration') or {}
    L += ['', '## 5. Decision', '', f'- {d["line"]}', f'- plan item {d["plan_item"]}, role {d["role"]}, basis {d["basis"]}{"" if d.get("basis_declared") else " (default, not declared)"}, '
          f'pre-registered {d["pre_registered"]} ({d.get("registration_status")})'
          + (f' ({d["registered_at"]})' if d.get('registered_at') else '') + f'; state {d["state"]}'
          + (f'; registration gate: {"PASS" if reg.get("valid") else "FAIL"} ({reg.get("reason")}{"; " + reg["score_order"] if reg.get("score_order") else ""})' if reg else ''),
          f'- duplicate check: {d["duplicate_check_basis"]}; cache {d["submissions_cache"]} (modified {d["submissions_cache_modified"]}); duplicates: '
          + (', '.join(f'{x["ref"]} ({x["candidate"]}, {x["matched"]})' for x in d['duplicate_of']) or 'none')
          + ('; same fingerprint scored later: ' + ', '.join(f'{x["ref"]} ({x["candidate"]})' for x in d['duplicates_scored_later']) if d['duplicates_scored_later'] else '')
          + ('; WARNING only (Public3 output sha alone): ' + ', '.join(f'{x["ref"]} ({x["candidate"]})' for x in d['public3_hash_only_matches']) if d.get('public3_hash_only_matches') else '')]
    cov = d.get('duplicate_coverage') or {}
    state = ('UNAVAILABLE (' + str(cov.get('reason')) + ')' if not cov.get('available') else 'COMPLETE' if cov.get('complete')
             else 'INCOMPLETE (' + str(cov.get('incomplete_reason')) + ')')
    L.append(f'- duplicate coverage: {state}; registry {cov.get("registry")} (sha {str(cov.get("registry_sha256"))[:12]}, {cov.get("registry_entries")} entries) '
             f'+ {cov.get("run_candidates")} candidate(s) of this run; non-ERROR cached submissions {cov.get("submissions_requiring_coverage")}, covered '
             f'{cov.get("covered")}, acknowledged historical {len(cov.get("acknowledged_historical") or [])}, without any fingerprint '
             f'{len(cov.get("without_fingerprints") or [])}; COMPLETE cached submissions without a registry entry: '
             f'{cov.get("cached_scored_submissions_without_registry_entry")}')
    for key, label in (('duplicates_unverified', 'POSSIBLE DUPLICATE, status unverified (HOLD)'), ('matches_unsubmitted', 'same fingerprint as a screened candidate without a recorded submission (caveat)'),
                       ('matches_errored', 'same fingerprint as a failed submission (resubmission)')):
        if d.get(key):
            L.append(f'- {label}: ' + ', '.join(f'{x["ref"]} ({x["candidate"]}, {"/".join(x["matched"])}{"; " + x["why"] if x.get("why") else ""})' for x in d[key]))
    if r.get('registry_record'):
        rr = r['registry_record']
        L.append(f'- registry record: {rr.get("path")}: appended {rr.get("appended")} entr{"y" if rr.get("appended") == 1 else "ies"} ({rr.get("note")})')
    if d.get('text'):
        L.append(f'- decision text: {d["text"]}')
    return '\n'.join(L) + '\n'


# ----------------------------------------------------------------------------- driver

def baseline_member(ctx, spec):
    b = spec.get('baseline') or {}
    return resolve_member(ctx, dict(name=b.get('name', ctx.cfg['baseline_name']), oof=b.get('oof', ctx.cfg['baseline_oof'])))


def prepare(ctx, spec):
    members = []
    for ms in spec['members']:
        M = resolve_member(ctx, ms)
        M['declared_replicates'] = ms.get('replicates') or []
        members.append(M)
    return members


def screen(ctx, spec, subs=None, registry=None, prepared=None):
    """One candidate. registry: the view run_many builds (persistent entries + this run); None loads the persistent registry."""
    validate_spec(spec)
    if prepared is None:
        members = prepare(ctx, spec)
        prepared = (members, baseline_member(ctx, spec), engineering_section(ctx, members, spec.get('release')))
    members, base, eng = prepared
    if registry is None:
        registry = registry_view(ctx)
    now = dt.datetime.now().astimezone()
    lin = lineage_section(ctx, members)
    oof = oof_section(ctx, members, base, lin, spec)
    rep = replication_section(ctx, members, base, spec)
    dec = decision_section(spec, eng, members, subs, registry, now=now)
    is_base = len(members) == 1 and members[0].get('oof_sha256') == base.get('oof_sha256')
    ver = verdict_section(spec, lin, oof, rep, eng, dec, is_base)
    return clean(dict(schema=SCHEMA, candidate=spec['candidate'], title=spec.get('title'), generated_at=now.isoformat(timespec='seconds'),
                      spec_sha256=spec_sha256(spec), config={k: ctx.cfg[k] for k in sorted(ctx.cfg)},
                      verdict=ver, lineage=lin, oof=oof, replication=rep, engineering=eng, decision=dec,
                      constraints='CPU only; reads existing artifacts; no training, GPU, network, Kaggle command or submission'))


def write_outputs(ctx, report, spec, overwrite=False):
    out = ctx.path(ctx.cfg['out_root']) / report['candidate']
    if (out / 'screen.json').exists() and not overwrite:
        raise FileExistsError(f'{out / "screen.json"} exists (use --overwrite to replace this screen)')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'spec.json').write_text(json.dumps(spec, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    (out / 'screen.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    (out / 'screen.md').write_text(render_md(report), encoding='utf-8', newline='\n')
    return out


def run_many(ctx, specs, subs, overwrite=False, write=True):
    """Screen several candidates against the persistent fingerprint registry plus each other (shared duplicate check).
    A run that writes its reports appends its candidates to the registry (atomic replace); write=False is a dry run that only reads it."""
    for s in specs:
        validate_spec(s)
    names = [s['candidate'] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError('candidate names in one run must be unique')
    out_root = ctx.path(ctx.cfg['out_root'])
    if write and not overwrite:
        for s in specs:
            if (out_root / s['candidate'] / 'screen.json').exists():
                raise FileExistsError(f'screen for {s["candidate"]} exists (use --overwrite)')
    run_at = dt.datetime.now().astimezone().isoformat(timespec='seconds')
    prepared, batch = {}, []
    for s in specs:
        members = prepare(ctx, s)
        eng = engineering_section(ctx, members, s.get('release'))
        prepared[s['candidate']] = (members, baseline_member(ctx, s), eng)
        batch.append(registry_entry(s, fingerprints_of(members, eng),
                                    source=(f'screen run {run_at}: {ctx.rel(out_root / s["candidate"] / "screen.json")}' if write else f'dry run {run_at}'),
                                    recorded_at=run_at, spec_sha=spec_sha256(s)))
    registry = registry_view(ctx, batch)
    reports = [screen(ctx, s, subs, registry, prepared[s['candidate']]) for s in specs]
    if write:
        if registry['available']:
            record = append_registry(ctx, batch)
        else:
            record = dict(path=registry['path'], appended=0, created=False,
                          note=f'not appended: {registry["reason"]} (a screen run never creates or replaces the registry; use --seed-registry)')
        for r, s in zip(reports, specs):
            r['registry_record'] = record
            write_outputs(ctx, r, s, overwrite)
    return reports


def parse_member(text):
    name, _, rest = text.partition('=')
    if not name or not rest:
        raise argparse.ArgumentTypeError('--member NAME=PATH[:WEIGHT]')
    path, weight = rest, 1.0
    head, sep, tail = rest.rpartition(':')
    if sep and head:
        try:
            weight, path = float(tail), head
        except ValueError:
            pass
    return dict(name=name, oof=path, weight=weight, arm=name)


def parse_rule(text):
    m = re.fullmatch(r'\s*(public|delta)\s*(>=|<=|>|<)\s*([-+]?\d*\.?\d+)\s*:\s*(.+)', text)
    if not m:
        raise argparse.ArgumentTypeError('--rule "public>=0.920:reading" or "delta<=-0.005:reading"')
    return dict(on=m.group(1), op=m.group(2), value=float(m.group(3)), reading=m.group(4).strip())


def spec_from_cli(a):
    dec = dict(plan_item=a.plan_item, text=a.decision_text, rules=[parse_rule(r) for r in a.rule or []], basis=a.basis)
    if a.registered_at:
        dec['registered_at'] = a.registered_at
    if a.otherwise:
        dec['otherwise'] = a.otherwise
    if a.reference_public is not None:
        dec['reference'] = dict(public=a.reference_public, ref=a.reference_ref)
    if a.control_for:
        dec.update(role='control', controls_for=a.control_for)
    spec = dict(candidate=a.candidate, members=[parse_member(m) for m in a.member], decision=dec,
                release=dict(out_dir=a.release_out, kernel_dir=a.kernel_dir), submission_ref=a.submission_ref)
    if a.baseline_oof:
        spec['baseline'] = dict(name=a.baseline_name or 'baseline', oof=a.baseline_oof)
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', type=Path, help='candidate spec JSON')
    ap.add_argument('--specs', type=Path, help='JSON {"candidates": [spec, ...], "submissions_cache": path} screened with a shared registry')
    ap.add_argument('--candidate')
    ap.add_argument('--member', action='append', help='NAME=PATH_TO_OOF_CSV[:WEIGHT] (repeat)')
    ap.add_argument('--release-out')
    ap.add_argument('--kernel-dir')
    ap.add_argument('--plan-item')
    ap.add_argument('--decision-text')
    ap.add_argument('--rule', action='append', help='"public>=0.920:reading" or "delta<=-0.005:reading" (repeat, first match wins)')
    ap.add_argument('--otherwise')
    ap.add_argument('--registered-at', help='when the rules were registered, with a time zone ("2026-10-07 16:17 KST"); required for them to count as pre-registered')
    ap.add_argument('--basis', default='public', choices=list(BASES))
    ap.add_argument('--reference-public', type=float)
    ap.add_argument('--reference-ref')
    ap.add_argument('--control-for', action='append')
    ap.add_argument('--submission-ref')
    ap.add_argument('--baseline-oof')
    ap.add_argument('--baseline-name')
    ap.add_argument('--submissions', type=Path, help='cached Kaggle submissions list (CLI CSV or JSON); without it the registry cannot be '
                                                  'reconciled against the submitted models, so duplicate coverage is incomplete (HOLD)')
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out-root')
    ap.add_argument('--draws', type=int)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--registry', help='persistent fingerprint registry (default ' + DEFAULTS['fingerprint_registry'] + '); missing or unreadable -> HOLD')
    ap.add_argument('--seed-registry', nargs='+', metavar='SCREEN_JSON',
                    help='create or extend the fingerprint registry from existing screen.json reports, then exit')
    ap.add_argument('--record-historical', type=Path, metavar='HIST_JSON',
                    help=f'append acknowledged historical entries ({HISTORICAL_SCHEMA}: submissions scored before the screen existed) with every '
                         'fingerprint the local artifacts allow and a reason for each unavailable one; needs --submissions; then exit')
    ap.add_argument('--bind-submission', nargs=2, metavar=('CANDIDATE', 'REF'),
                    help='after submitting a screened candidate: append its newest current registry entry with submission_ref REF '
                         '(REF must be in --submissions); then exit')
    ap.add_argument('--bind-index', type=int, help='registry entry index for --bind-submission when the candidate has several different current entries')
    a = ap.parse_args(argv)
    ctx = Ctx(a.root, out_root=a.out_root, draws=a.draws, fingerprint_registry=a.registry)
    if a.seed_registry:
        rec = seed_registry(ctx, a.seed_registry)
        print(json.dumps(rec))
        return 0
    if a.record_historical or a.bind_submission:
        subs = load_submissions(ctx.path(str(a.submissions))) if a.submissions else None
        if subs is None:
            ap.error('--record-historical / --bind-submission need --submissions CACHE (the cached submissions list that names the refs)')
        rec = (record_historical(ctx, a.record_historical, subs) if a.record_historical
               else bind_submission(ctx, a.bind_submission[0], a.bind_submission[1], subs, index=a.bind_index))
        print(json.dumps(clean(rec), indent=1, ensure_ascii=False))
        return 0
    if a.specs:
        bundle = read_json(a.specs)
        subs = load_submissions(ctx.path(str(a.submissions)) if a.submissions else ctx.path(bundle.get('submissions_cache')))
        out = ctx.path(ctx.cfg['out_root'])
        name = bundle.get('summary_name', 'BACKTEST')
        if (out / f'{name}.json').exists() and not a.overwrite:
            raise FileExistsError(f'{out / (name + ".json")} exists (use --overwrite)')
        reports = run_many(ctx, bundle['candidates'], subs, a.overwrite)
        summary = dict(schema=SCHEMA + '-summary', generated_at=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
                       specs=ctx.rel(a.specs), submissions_cache=(subs or {}).get('path'), submissions_cache_sha256=(subs or {}).get('sha256'),
                       fingerprint_registry=reports[0].get('registry_record') if reports else None, ceiling_note=CEILING_NOTE,
                       duplicate_coverage={k: v for k, v in (reports[0]['decision'].get('duplicate_coverage') or {}).items()
                                           if k in ('complete', 'incomplete_reason', 'submissions_requiring_coverage', 'covered', 'uncovered',
                                                    'acknowledged_historical', 'without_fingerprints')} if reports else None,
                       verdicts={r['candidate']: dict(verdict=r['verdict']['verdict'], line=r['verdict']['line'], direction=r['oof']['direction']['line']) for r in reports})
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{name}.json').write_text(json.dumps(clean(summary), indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
        (out / f'{name}.md').write_text(f'# Submission screen {name}\n\nGenerated {summary["generated_at"]}; specs {summary["specs"]}; submissions cache {summary["submissions_cache"]}. '
                                        f'{CEILING_NOTE}.\n\n' + '\n'.join(f'- {v["line"]}\n  - {v["direction"]}' for v in summary['verdicts'].values()) + '\n',
                                        encoding='utf-8', newline='\n')
        for r in reports:
            print(r['verdict']['line'])
        return 0
    if a.spec:
        spec = read_json(a.spec)
    elif a.candidate and a.member:
        spec = spec_from_cli(a)
    else:
        ap.error('give --spec, --specs, or --candidate with --member')
    subs = load_submissions(a.submissions) if a.submissions else load_submissions(ctx.path(spec.get('submissions_cache')))
    rep = run_many(ctx, [spec], subs, a.overwrite)[0]
    print(rep['verdict']['line'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
