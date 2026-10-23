"""EXP169 v2 pre-reading check A v1.1.0: DINO marginal pooled-OOF screen (CPU only).

Claude opus for B, 2026-10-10 (stands in for the Codex sol implementer).
Spec: artifacts/experiments/EXP169_RELEASE_V2/PRE_READING_CHECKS.md section A (v1.0.1).
Run as ``python -m tools.exp169_v2_dino_marginal --r-mode plain|flip``; write the
flip registration once, before check-B outputs exist, with
``--r-mode flip --register-only``.

One frozen comparison, no weight sweep (r = global average-tie rank to [0, 1]):
    candidate Q_RCD = r((r(R) + r(C) + r(D)) / 3)   -> exp169_bundle_screen.bundle_of
    baseline  Q_RC  = r((r(R) + r(C)) / 2)           -> exp169_bundle_screen.global_rank
                                                       on the SAME returned ranked R/C
C = EXP170 L2 ConvNeXt-T oof_k36_flip_tta.csv, D = EXP172 L2 DINO-base oof_best.csv
(K36 plain). Within each EXP099 fold every family is ranked per target with average
ties (exp115_vlm_eval.pct_rank, rank / n_fold); folds are concatenated in the EXP172
fold/roster order over exactly the 4,343 pooled held-out studies, then each family is
globally ranked and merged as above.

r-modes:
  plain  registered-input continuity report (unchanged since v1.0.0). R = mean of the
         within-fold ranked plain oof_k36.csv S42/S43 tables (seed-rank mean);
         every R/C/D hash is authenticated against EXP172 screen/result.json;
         registration pre-reading/marginal-registration.json.
  flip   exact serving-policy reconciliation (v1.1.0 serving aggregation). R inputs
         are the two check-B K36 flip-TTA tables pre-reading/l2-flip-oof/<tag>/
         oof_k36_flip_tta.csv; per fold R = pct_rank((S42 + S43) / 2), i.e. the
         serving PROBABILITY mean then one rank (no seed-rank averaging). Each
         fold's R/C/D order is asserted identical to exp169_release_v2.
         serving_family_rank (exact equivalence; only the within-fold scale differs,
         rank / n vs (rank - 1) / (n - 1), and the EXP172 rank / n scale is kept for
         fold pooling so C/D stay bit-identical to EXP172). Requires the separate,
         pre-written registration pre-reading/marginal-registration-flip-v1.1.json.
         Refuses to run until all ten <tag> folders hold oof_k36_flip_tta.csv,
         oof_plain_k36.csv, output-inventory.json (exactly those two CSVs) and a
         check-B receipt.json (schema exp169-v2-l2-flip-oof-receipt-v1, status
         ADMITTED, own tag/seed/fold, inventory binding, roster hash, passed plain
         gate); the regenerated plain OOF must also match the frozen oof_k36.csv
         within the EXP124 reproduction ceiling 0.002 (max absolute difference).
         Seed-rank-mean on the flip inputs is reported as a point-only diagnostic so
         input (plain -> flip) and aggregation (rank mean -> probability mean)
         effects are separated; only the serving arm is gated.

Reused, never copied: exp172_dino_base_l2 (validate_manifest, _verify_fold, tag,
uid_hash, paired_jev_consensus, admission, read/sha/need/write_new),
exp169_bundle_screen (paths, load_oof, _verify_inventory, global_rank, bundle_of),
exp156_k_sweep.references (JEV/CONSENSUS/DREAD/AQUILA), exp149 fold rosters,
exp115 pct_rank/component_groups/bootstrap_index, exp162 _score. EXP172's own
screen() is monolithic, so its preamble is re-composed from those same helpers in
``exp172_inputs`` (wrapper; no file of those tools is modified).

Admission (frozen, unchanged after results): mean(JEV, consensus) delta > 0 AND
paired 95 % CI lower > 0 AND Dread delta >= -0.003, else HOLD. 300 component
bootstrap draws, seed 20261003, one draw list shared by both arms. Aquila is
self-referenced to L2 and report-only. Ruler-map tiers (evidence_ledger
ruler-map-v1) are diagnostic only: a HOLD whose per-target losses are all on
LOW-validity targets is flagged REVIEW (image-read follow-up), never re-gated.

Reference SHA256s, roster/component/draw digests and helper code hashes are frozen
in pre-reading/marginal-registration.json BEFORE any candidate metric is computed;
a later run must match it exactly. EXP172's registered result is only read.

Changelog:
1.0.0 - Initial check-A screen (plain and flip R modes), registration, ruler join.
1.1.0 - Claude opus for B, 2026-10-10 (verifier hb78-fix2 item 12): flip mode now uses
        the serving aggregation (per-fold probability mean of S42/S43, then one rank,
        equivalence-checked against exp169_release_v2.serving_family_rank) instead of
        the seed-rank mean; flip mode requires its own write-once registration
        marginal-registration-flip-v1.1.json (--register-only) and reads the check-B
        receipt contract (status ADMITTED, inventory/roster/plain-gate binding).
        Plain mode behaviour and its registration are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp169_bundle_screen as bundle_helper
from tools import exp172_dino_base_l2 as e172

SCHEMA = 'exp169-v2-dino-marginal-screen-v1'
REGISTRATION_SCHEMA = 'exp169-v2-dino-marginal-registration-v1'
FLIP_REGISTRATION_SCHEMA = 'exp169-v2-dino-marginal-registration-flip-v1.1'
VERSION = '1.1.0'
SELF = 'tools/exp169_v2_dino_marginal.py'
EXP = 'artifacts/experiments/EXP169_RELEASE_V2'
PRE = EXP + '/pre-reading'
REGISTRATION = PRE + '/marginal-registration.json'               # plain mode (v1.0.0)
FLIP_REGISTRATION = PRE + '/marginal-registration-flip-v1.1.json'  # flip mode (v1.1.0)
REGISTRATIONS = {'plain': REGISTRATION, 'flip': FLIP_REGISTRATION}
OUT = {'plain': PRE + '/marginal-plain-r', 'flip': PRE + '/marginal-flip-r'}
FLIP_ROOT = PRE + '/l2-flip-oof'
FLIP_FILE, FLIP_PLAIN = 'oof_k36_flip_tta.csv', 'oof_plain_k36.csv'
FLIP_INVENTORY, FLIP_RECEIPT = 'output-inventory.json', 'receipt.json'
# Receipt contract written by tools/exp169_v2_l2_flip_oof.py (check B) v1.0.0 produce().
FLIP_RECEIPT_CONTRACT = {'schema': 'exp169-v2-l2-flip-oof-receipt-v1',
                         'job_id': 'EXP169_V2_L2_K36_FLIP_OOF_v1', 'status': 'ADMITTED'}
PLAIN_CEILING = 0.002
EXP172_RESULT = e172.EXP + '/screen/result.json'
EXP172_OUTPUTS = e172.EXP + '/outputs'
CONVNEXT = ('artifacts/experiments/EXP170_CONVNEXT_T_L2/outputs/results/'
            'EXP170_CONVNEXT_T_L2_S42_F{fold}/oof_k36_flip_tta.csv')
RULER_MAP = 'artifacts/evidence_ledger/ruler-map-v1/map.json'
DOCUMENTS = (EXP + '/PRE_READING_CHECKS.md', EXP + '/PROTOCOL.md', e172.PROTOCOL)
HELPERS = ('tools/exp172_dino_base_l2.py', 'tools/exp169_bundle_screen.py',
           'tools/exp170_convnext_t_l2.py', 'tools/exp156_k_sweep.py',
           'tools/exp115_vlm_eval.py', 'tools/exp149_label_sources.py',
           'tools/exp162_dino_folds_kaggle.py', 'tools/exp158_dino_strong.py')
STUDIES, FOLDS, SEEDS = 4343, tuple(range(5)), (42, 43)
DRAWS, SEED, DREAD_FLOOR = 300, 20261003, -0.003
PRIMARY = ('jev', 'consensus')
REFERENCE_KEYS = (('jev', 'jev_hybrid'), ('aquila', 'aquila_v1'),
                  ('dread', 'dread'), ('consensus', 'consensus'))
FORMULAS = {'candidate': 'Q_RCD = r((r(R) + r(C) + r(D)) / 3)',
            'baseline': 'Q_RC = r((r(R) + r(C)) / 2)',
            'r': 'global average-tie rank per target, (rank - 1) / (n - 1)',
            'family_inputs': 'within-fold average-tie target ranks (rank / n_fold); '
                             'R = mean of ranked S42 and S43; folds concatenated'}
GATE = {'primary': 'mean(JEV, consensus) macro-12 AUC delta (Q_RCD - Q_RC) > 0',
        'ci': 'paired component-bootstrap 95 % CI lower bound > 0',
        'dread_floor': 'Dread delta >= -0.003', 'otherwise': 'HOLD',
        'aquila': 'self-referenced to L2; report only',
        'ruler_map': 'diagnostic only; HOLD with losses only on LOW-validity targets -> REVIEW'}
R_POLICY = {'plain': 'L2 RadR50 S42/S43 plain oof_k36.csv (EXP172-registered inputs)',
            'flip': 'L2 RadR50 S42/S43 K36 flip-TTA ' + FLIP_FILE + ' under ' + FLIP_ROOT}
# FORMULAS / R_POLICY above are the frozen v1.0.0 plain-mode registration values (the
# v1.0.0 'flip' R_POLICY entry is superseded by FLIP_R_POLICY and is kept only so the
# plain registration still matches). Flip mode registers the serving aggregation:
FLIP_FORMULAS = {
    **FORMULAS,
    'family_inputs': 'serving aggregation per fold: R = pct_rank((S42 + S43) / 2) of the '
                     'two K36 flip-TTA OOF probability tables (probability mean, then one '
                     'average-tie rank; no seed-rank averaging); C = pct_rank(flip-TTA '
                     'ConvNeXt-T); D = pct_rank(plain DINO); within-fold scale rank / n_fold '
                     '(EXP172); folds concatenated',
    'serving_equivalence': 'per fold, every family order (average ties) equals '
                           'tools/exp169_release_v2.py:serving_family_rank exactly',
    'diagnostic_only': 'seed-rank-mean R on the flip inputs, point estimates, never gated'}
FLIP_R_POLICY = ('L2 RadR50 S42/S43 K36 flip-TTA ' + FLIP_FILE + ' under ' + FLIP_ROOT +
                 ' (check B), serving probability mean of the two seeds per fold, then rank')
SERVING_CODE = (('tools/exp169_release_v2.py', 'serving_family_rank'),
                ('tools/exp169_release_v2.py', 'rank'),
                ('tools/exp152_own0949_release.py', 'percentile_average'))
REPORT_KIND = {'plain': 'registered-input continuity report',
               'flip': 'exact serving-policy reconciliation'}
need, sha, read, write_new = e172.need, e172.sha, e172.read, e172.write_new
TARGETS = e172.TARGETS


# ------------------------------------------------------------------ identities
def l2_tags(root=ROOT):
    """{(seed, fold): experiment tag} for the ten frozen L2 RadR50 checkpoints."""
    root = Path(root)
    tags = {}
    for fold in FOLDS:
        p = bundle_helper.paths(root, fold)
        tags[(42, fold)] = p['l2_s42'].parent.name
        tags[(43, fold)] = p['l2_s43'].parent.name
    return tags


def helper_hashes(root=ROOT):
    return {name: sha(Path(root) / name) for name in HELPERS}


def serving_code_hashes(root=ROOT):
    """SHA256 of the exact source of the serving rank functions flip mode relies on."""
    import ast
    out = {}
    for path, name in SERVING_CODE:
        source = (Path(root) / path).read_text(encoding='utf-8')
        nodes = [n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name == name]
        need(len(nodes) == 1, f'serving helper {path}:{name} not found exactly once')
        segment = ast.get_source_segment(source, nodes[0])
        out[f'{path}:{name}'] = hashlib.sha256(segment.encode('utf-8')).hexdigest()
    return out


def reference_hashes(root=ROOT):
    """Same names/files as the EXP172 screen's ``reference_sha256`` block."""
    from tools import exp156_k_sweep as e156
    pairs = (('jev', e156.JEV), ('aquila', e156.AQUILA), ('dread', e156.DREAD),
             *[(f'consensus_{i}', x) for i, x in enumerate(e156.CONSENSUS)])
    return {name: sha(Path(root) / path) for name, path in pairs}


def draw_digest(draws):
    import numpy as np
    h = hashlib.sha256()
    for ix in draws:
        a = np.asarray(ix, dtype='<i8')
        h.update(len(a).to_bytes(8, 'little'))
        h.update(a.tobytes())
    return h.hexdigest()


def make_draws(groups):
    from tools import exp115_vlm_eval as ev
    return ev.bootstrap_index(groups, DRAWS, SEED)


# ------------------------------------------------------------------ flip inputs
def check_flip_receipt(folder, tag, seed, fold):
    """Check-B receipt contract (tools/exp169_v2_l2_flip_oof.py produce/verify_receipt):
    identity, ADMITTED status, inventory of exactly the two CSVs bound to the receipt,
    file bytes/sha256, and a passed plain gate at the 0.002 ceiling. Returns the receipt."""
    folder = Path(folder)
    receipt = read(folder / FLIP_RECEIPT)
    need(all(receipt.get(k) == v for k, v in FLIP_RECEIPT_CONTRACT.items()) and
         (receipt.get('tag'), receipt.get('seed'), receipt.get('fold')) == (tag, seed, fold),
         'receipt identity/status differs from the check-B ADMITTED contract')
    inventory = read(folder / FLIP_INVENTORY)
    outputs = receipt['outputs']
    need(set(inventory) == {FLIP_FILE, FLIP_PLAIN} and outputs['files'] == inventory and
         outputs['inventory_sha256'] == sha(folder / FLIP_INVENTORY),
         'output inventory differs from receipt')
    for name in (FLIP_FILE, FLIP_PLAIN):
        bundle_helper._verify_inventory(folder / name)
    gate = receipt['plain_gate']
    need(gate['passed'] is True and gate['tolerance'] == PLAIN_CEILING and
         gate['max_abs_diff'] is not None and gate['max_abs_diff'] <= PLAIN_CEILING,
         'receipt plain gate not passed')
    return receipt


def flip_r_inputs(root=ROOT):
    """Fail loudly unless all ten flip-TTA L2 folders exist with verified receipts."""
    root = Path(root)
    tags = l2_tags(root)
    missing, invalid, found = [], [], {}
    for key, tag in sorted(tags.items()):
        folder = root / FLIP_ROOT / tag
        files = [folder / n for n in (FLIP_FILE, FLIP_PLAIN, FLIP_INVENTORY, FLIP_RECEIPT)]
        absent = [p.name for p in files if not p.is_file()]
        if absent:
            missing.append(f'{tag} ({", ".join(absent)})')
            continue
        try:
            receipt = check_flip_receipt(folder, tag, *key)
            roster = {'ids_sha256': receipt['outputs']['ids_sha256'],
                      'rows': receipt['outputs']['rows']}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            invalid.append(f'{tag}: {exc!r}')
            continue
        found[key] = {'tag': tag, 'flip': folder / FLIP_FILE, 'plain': folder / FLIP_PLAIN,
                      'receipt': folder / FLIP_RECEIPT, **roster}
    need(not missing and not invalid and len(found) == 10,
         f'flip R inputs not ready: {len(missing)} missing, {len(invalid)} invalid of 10 '
         f'under {FLIP_ROOT}: ' + '; '.join(missing + invalid))
    return found


# ------------------------------------------------------------------ EXP172 inputs
def exp172_inputs(root=ROOT, outputs=None):
    """EXP172 screen preamble re-composed from its own helpers: collection, overlay
    manifest, EXP099 rosters, per-fold receipt chain and input paths."""
    from tools import exp149_label_sources as e149
    root = Path(root)
    outputs = Path(outputs) if outputs else root / EXP172_OUTPUTS
    collection = read(outputs / 'collection.json')
    need(collection['schema'] == e172.SCHEMA and collection['status'] == 'VERIFIED' and
         collection['completed_folds'] == list(FOLDS) and
         collection['manifest_sha256'] == sha(outputs / 'manifest.json'),
         'EXP172 five-fold collection differs')
    overlay = Path(collection['overlay'])
    p = e172.validate_manifest(overlay / 'manifest.json', overlay)
    canonical, fold_of = e149.fold_rosters(e149._load_manifest('student', root), root)
    ids_by_fold = {f: [uid for uid in canonical if fold_of[uid] == f] for f in FOLDS}
    ids = [uid for f in FOLDS for uid in ids_by_fold[f]]
    need(len(ids) == len(set(ids)) == STUDIES, 'pooled EXP099 held-out coverage differs')
    paths = {}
    for f in FOLDS:
        q = read(overlay / p['fold_protocols'][str(f)]['path'])
        run = outputs / 'artifacts/experiments' / e172.tag(f)
        e172._verify_fold(run, f, q, p['fold_protocols'][str(f)]['sha256'], overlay)
        control = bundle_helper.paths(root, f)
        paths[f] = {'l2_s42': control['l2_s42'], 'l2_s43': control['l2_s43'],
                    'dino_l2': run / 'oof_best.csv',
                    'convnext': root / CONVNEXT.format(fold=f)}
    return {'ids': ids, 'ids_by_fold': ids_by_fold, 'paths': paths,
            'collection_sha256': sha(outputs / 'collection.json'),
            'manifest_sha256': collection['manifest_sha256']}


def authenticate(paths, root, registered):
    """Verify inventories and that every R/C/D input hash equals EXP172's."""
    root = Path(root)
    files, hashes = {}, {}
    for f, local in paths.items():
        for name, path in local.items():
            bundle_helper._verify_inventory(path)
            key = f'{name}_f{f}'
            files[key] = path.relative_to(root).as_posix()
            hashes[key] = sha(path)
            need(registered.get(key) == hashes[key],
                 'input differs from EXP172 screen/result.json: ' + key)
    return files, hashes


# ------------------------------------------------------------------ computation
def fold_family_ranks(s42, s43, c, d):
    """Within-fold average-tie target ranks: R = mean of ranked seeds.

    Plain-mode (v1.0.0 registered) rule; in flip mode only a point diagnostic."""
    from tools import exp115_vlm_eval as ev
    return {'R': (ev.pct_rank(s42) + ev.pct_rank(s43)) / 2.0,
            'C': ev.pct_rank(c), 'D': ev.pct_rank(d)}


def serving_fold_ranks(s42, s43, c, d):
    """Flip-mode within-fold family ranks with the serving aggregation.

    R = pct_rank((S42 + S43) / 2): probabilities are averaged over the two seeds,
    then ranked once (the frozen serving ``aggregate`` rule; never a seed-rank mean).
    C and D are single-member families. Every family's within-fold order, ties
    included, must equal tools/exp169_release_v2.serving_family_rank exactly; only
    the scale differs (rank / n here, the EXP172 fold-pooling scale, versus
    (rank - 1) / (n - 1)), and the global rerank after concatenation follows."""
    import numpy as np
    from scipy.stats import rankdata
    from tools import exp115_vlm_eval as ev
    from tools import exp169_release_v2 as v2
    s42, s43 = np.asarray(s42, np.float64), np.asarray(s43, np.float64)
    out = {'R': ev.pct_rank((s42 + s43) / 2.0), 'C': ev.pct_rank(c), 'D': ev.pct_rank(d)}
    serving = {'R': v2.serving_family_rank([s42, s43], 'l2'),
               'C': v2.serving_family_rank([c], 'convnext'),
               'D': v2.serving_family_rank([d], 'dino')}
    for name in out:
        need(np.array_equal(rankdata(out[name], axis=0), rankdata(serving[name], axis=0)),
             f'{name}: within-fold order differs from exp169_release_v2.serving_family_rank')
    return out


def arms(R, C, D):
    """Q_RCD via bundle_of; Q_RC via global_rank of the SAME ranked R/C arrays."""
    q_rcd, ranked = bundle_helper.bundle_of({'l2': R, 'dino': D, 'convnext': C})
    q_rc = bundle_helper.global_rank((ranked['l2'] + ranked['convnext']) / 2.0)
    return q_rcd, q_rc, ranked


def compare(candidate, baseline, refs, draws, evaluator):
    """EXP172 paired JEV/consensus bootstrap and its unchanged admission gate."""
    pair = e172.paired_jev_consensus(candidate, baseline, refs, draws, evaluator)
    return pair, e172.admission(pair)


def point_deltas(candidate, baseline, refs, evaluator):
    """Point macro-12 deltas only (no bootstrap, no gate) for diagnostics."""
    import numpy as np
    out = {}
    for name, (y, mask, minimum) in refs.items():
        c = evaluator.macro(evaluator.auc_targets(y, candidate, mask, minimum))
        b = evaluator.macro(evaluator.auc_targets(y, baseline, mask, minimum))
        out[name] = {'candidate': float(c) if np.isfinite(c) else None,
                     'baseline': float(b) if np.isfinite(b) else None,
                     'delta': float(c - b) if np.isfinite(c - b) else None}
    primary = [out[r]['delta'] for r in PRIMARY]
    return {'references': out,
            'primary_mean_delta': None if None in primary else sum(primary) / 2.0}


def target_counts(refs):
    import numpy as np
    out = {}
    for name, (y, mask, minimum) in refs.items():
        y, mask = np.asarray(y), np.asarray(mask, bool)
        out[name] = {'min_class': int(minimum), 'per_target': {
            t: {'n': int(mask[:, i].sum()), 'positives': int((y[mask[:, i], i] == 1).sum()),
                'negatives': int(mask[:, i].sum() - (y[mask[:, i], i] == 1).sum())}
            for i, t in enumerate(TARGETS)}}
    return out


def fold_deltas(candidate, baseline, refs, sizes, evaluator):
    from tools import exp162_dino_folds_kaggle as folds
    out, start = {}, 0
    for f, n in sizes.items():
        sl = slice(start, start + n)
        out[str(f)] = {}
        for name, (y, mask, minimum) in refs.items():
            ref = (y[sl], mask[sl], minimum)
            a = folds._score(candidate[sl], ref, evaluator)['macro']
            b = folds._score(baseline[sl], ref, evaluator)['macro']
            out[str(f)][name] = {'candidate': a, 'baseline': b,
                                 'delta': a - b if a is not None and b is not None else None}
        start = sl.stop
    return out


def ruler_diagnostics(pair, status, root=ROOT):
    """Join per-target deltas to ruler-map-v1 tiers; diagnostic only."""
    path = Path(root) / RULER_MAP
    if not path.is_file():
        return {'status': 'NOT-CHECKED', 'reason': 'ruler map missing: ' + RULER_MAP}
    try:
        rec = read(path)['recommendations']
        tiers = {t: rec[t]['validity'] for t in TARGETS}
        rulers = {t: list(rec[t].get('references', [])) for t in TARGETS}
    except (KeyError, TypeError) as exc:
        return {'status': 'NOT-CHECKED', 'reason': f'no simple per-target join: {exc!r}'}
    refs = ('jev', 'consensus', 'dread')
    deltas = {r: pair['references'][r]['per_target_delta'] for r in refs}
    negative = {r: [t for t in TARGETS if deltas[r][t] is not None and deltas[r][t] < 0]
                for r in refs}
    union = sorted({t for r in refs for t in negative[r]}, key=TARGETS.index)
    non_low = {}
    for r in refs:
        values = [deltas[r][t] for t in TARGETS if tiers[t] != 'LOW' and deltas[r][t] is not None]
        non_low[r] = sum(values) / len(values) if values else None
    review = status == 'HOLD' and bool(union) and all(tiers[t] == 'LOW' for t in union)
    return {'status': 'CHECKED', 'map': RULER_MAP, 'map_sha256': sha(path),
            'tiers': tiers, 'recommended_rulers': rulers,
            'negative_targets': {r: [{'target': t, 'tier': tiers[t], 'delta': deltas[r][t]}
                                     for t in negative[r]] for r in refs},
            'non_low_mean_target_delta': non_low,
            'flag': 'REVIEW' if review else 'NONE',
            'rule': GATE['ruler_map']}


# ------------------------------------------------------------------ registration
FROZEN = ('schema', 'formulas', 'gate', 'cohort', 'components', 'draws', 'reference_sha256',
          'registered_inputs_sha256', 'helper_code_sha256', 'flip_contract', 'r_policy')
FROZEN_FLIP = FROZEN + ('r_mode', 'serving_code_sha256')
FROZEN_BY_MODE = {'plain': FROZEN, 'flip': FROZEN_FLIP}


def mode_contract(r_mode, root=ROOT):
    """Mode-specific frozen registration fields. Plain returns the v1.0.0 values unchanged."""
    tags = sorted(l2_tags(root).values())
    if r_mode == 'plain':
        return {'schema': REGISTRATION_SCHEMA, 'formulas': FORMULAS,
                'flip_contract': {'root': FLIP_ROOT, 'file': FLIP_FILE, 'plain_file': FLIP_PLAIN,
                                  'tags': tags, 'plain_max_abs_diff_ceiling': PLAIN_CEILING,
                                  'requires': ['output-inventory.json', 'receipt.json status PASS']},
                'r_policy': R_POLICY}
    need(r_mode == 'flip', 'r-mode must be plain or flip')
    return {'schema': FLIP_REGISTRATION_SCHEMA, 'r_mode': 'flip', 'formulas': FLIP_FORMULAS,
            'flip_contract': {
                'root': FLIP_ROOT, 'file': FLIP_FILE, 'plain_file': FLIP_PLAIN,
                'inventory': FLIP_INVENTORY, 'receipt': FLIP_RECEIPT, 'tags': tags,
                'writer': 'tools/exp169_v2_l2_flip_oof.py (check B) produce()',
                'receipt_contract': {
                    **FLIP_RECEIPT_CONTRACT,
                    'identity': 'receipt tag/seed/fold equal the source checkpoint',
                    'inventory': 'exactly {oof_plain_k36.csv, oof_k36_flip_tta.csv}; equals '
                                 'receipt outputs.files; its sha256 equals '
                                 'outputs.inventory_sha256; file bytes/sha256 match',
                    'roster': 'outputs.ids_sha256 equals the EXP172 fold-order uid_hash and '
                              'outputs.rows the fold size',
                    'plain_gate': 'passed is true, tolerance 0.002, max_abs_diff <= 0.002'},
                'plain_max_abs_diff_ceiling': PLAIN_CEILING,
                'independent_plain_check': 'regenerated oof_plain_k36.csv versus the '
                                           'EXP172-authenticated frozen oof_k36.csv, '
                                           'max abs diff <= 0.002'},
            'r_policy': {'flip': FLIP_R_POLICY},
            'serving_code_sha256': serving_code_hashes(root)}


def registration_payload(root, ids, ids_by_fold, groups, draws, registered, exp172_result_sha,
                         r_mode='plain'):
    from tools import exp115_vlm_eval as ev
    root = Path(root)
    contract = mode_contract(r_mode, root)
    payload = {
        'schema': contract['schema'], 'tool': SELF, 'tool_version': VERSION,
        'tool_sha256': sha(root / SELF) if (root / SELF).is_file() else None,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'author': 'Claude opus for B (Codex sol stand-in)',
        'formulas': contract['formulas'], 'gate': GATE,
        'cohort': {'studies': len(ids), 'folds': list(FOLDS),
                   'fold_sizes': {str(f): len(ids_by_fold[f]) for f in FOLDS},
                   'pooled_order_sha256': e172.uid_hash(ids),
                   'fold_order_sha256': {str(f): e172.uid_hash(ids_by_fold[f]) for f in FOLDS}},
        'components': {'path': ev.COMPONENTS, 'sha256': sha(root / ev.COMPONENTS),
                       'groups': len(groups)},
        'draws': {'unit': 'validation component', 'draws': DRAWS, 'seed': SEED,
                  'generator': 'exp115_vlm_eval.bootstrap_index', 'shared_by_both_arms': True,
                  'indices_sha256': draw_digest(draws)},
        'reference_sha256': reference_hashes(root),
        'registered_inputs_sha256': registered,
        'helper_code_sha256': helper_hashes(root),
        'flip_contract': contract['flip_contract'],
        'r_policy': contract['r_policy'],
        'exp172_result_sha256': exp172_result_sha,
        'protocol_documents_sha256': {d: sha(root / d) for d in DOCUMENTS if (root / d).is_file()},
    }
    if r_mode == 'flip':
        plain_reg = root / REGISTRATION
        payload.update(
            r_mode='flip', serving_code_sha256=contract['serving_code_sha256'],
            supersedes_for_flip={
                'path': REGISTRATION,
                'sha256': sha(plain_reg) if plain_reg.is_file() else None,
                'reason': 'v1.0.0 flip formula averaged seed ranks (verifier hb78-fix2 item 12); '
                          'that file now governs plain mode only'})
    return payload


def register(path, payload, frozen_keys=FROZEN, create=True):
    """Write once; later runs must match every frozen field exactly.

    ``create=False`` (flip mode) requires a registration written beforehand."""
    path = Path(path)
    if not path.exists():
        need(create, 'registration missing; write it first with --r-mode flip '
                     '--register-only: ' + str(path))
        write_new(path, payload)
        return {'path': str(path), 'sha256': sha(path), 'created': True,
                'tool_sha256_matches': True}
    frozen = read(path)
    for key in frozen_keys:
        need(frozen.get(key) == payload.get(key),
             f'registration mismatch on {key}; use a new --registration file')
    return {'path': str(path), 'sha256': sha(path), 'created': False,
            'tool_sha256_matches': frozen.get('tool_sha256') == payload.get('tool_sha256')}


# ------------------------------------------------------------------ run
def preamble(root, outputs=None):
    """EXP172 authentication, cohort, references and draws; no candidate metric."""
    from tools import exp115_vlm_eval as ev
    root = Path(root)
    prior_path = root / EXP172_RESULT
    prior = read(prior_path)
    need(prior['schema'] == e172.SCHEMA and prior['status']['status'] == 'ADMIT' and
         prior['studies'] == STUDIES, 'EXP172 registered screen differs')
    inputs = exp172_inputs(root, outputs)
    files, hashes = authenticate(inputs['paths'], root, prior['input_sha256'])
    refs_sha = reference_hashes(root)
    need(refs_sha == prior['reference_sha256'], 'reference files differ from EXP172 screen')
    groups = ev.component_groups(inputs['ids'], root)
    draws = make_draws(groups)
    return {'prior': prior, 'prior_path': prior_path, 'inputs': inputs, 'files': files,
            'hashes': hashes, 'refs_sha': refs_sha, 'groups': groups, 'draws': draws}


def register_only(r_mode, root=ROOT, registration=None, outputs=None):
    """Write the flip-mode registration once, before any check-B output or candidate
    metric exists. Reads only the EXP172-registered inputs, references and rosters."""
    need(r_mode == 'flip', '--register-only writes the flip registration only')
    root = Path(root)
    registration = Path(registration) if registration else root / FLIP_REGISTRATION
    need(not registration.resolve().is_relative_to((root / e172.EXP).resolve()),
         'never write inside EXP172')
    need(not registration.exists(), 'registration is write-once and exists: ' + str(registration))
    pre = preamble(root, outputs)
    inputs = pre['inputs']
    return register(registration, registration_payload(
        root, inputs['ids'], inputs['ids_by_fold'], pre['groups'], pre['draws'], pre['hashes'],
        sha(pre['prior_path']), 'flip'), FROZEN_FLIP)


def run(r_mode, root=ROOT, registration=None, out_dir=None, outputs=None):
    import numpy as np
    import scipy
    from tools import exp115_vlm_eval as ev
    from tools import exp156_k_sweep as e156
    from tools import exp162_dino_folds_kaggle as folds

    started = time.monotonic()
    need(r_mode in OUT, 'r-mode must be plain or flip')
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / OUT[r_mode]
    registration = Path(registration) if registration else root / REGISTRATIONS[r_mode]
    exp172_dir = (root / e172.EXP).resolve()
    for target in (out_dir, registration):
        need(not target.resolve().is_relative_to(exp172_dir), 'never write inside EXP172')
    need(not (out_dir / 'result.json').exists(), 'fresh output required: ' + str(out_dir))
    flip = flip_r_inputs(root) if r_mode == 'flip' else None   # fail loudly first
    need(r_mode == 'plain' or registration.is_file(),
         'flip mode requires its pre-written registration (--r-mode flip --register-only): '
         + str(registration))

    pre = preamble(root, outputs)
    prior, prior_path, inputs = pre['prior'], pre['prior_path'], pre['inputs']
    ids, ids_by_fold = inputs['ids'], inputs['ids_by_fold']
    files, hashes, refs_sha = pre['files'], pre['hashes'], pre['refs_sha']
    groups, draws = pre['groups'], pre['draws']
    reg = register(registration, registration_payload(
        root, ids, ids_by_fold, groups, draws, hashes, sha(prior_path), r_mode),
        FROZEN_BY_MODE[r_mode], create=r_mode == 'plain')

    # ---- candidate metrics start only after the registration is frozen
    sections = {'R': [], 'C': [], 'D': []}
    rank_mean_r = []   # flip mode only: seed-rank-mean diagnostic on the flip inputs
    flip_files, flip_hashes, plain_check = {}, {}, {}
    for f in FOLDS:
        fid, local = ids_by_fold[f], inputs['paths'][f]
        s42 = bundle_helper.load_oof(local['l2_s42'], fid)
        s43 = bundle_helper.load_oof(local['l2_s43'], fid)
        if flip is not None:
            for seed, frozen in ((42, s42), (43, s43)):
                item = flip[(seed, f)]
                need(item['ids_sha256'] == e172.uid_hash(fid) and item['rows'] == len(fid),
                     f'{item["tag"]} receipt roster differs from the EXP172 fold order')
                regen = bundle_helper.load_oof(item['plain'], fid)
                diff = float(np.max(np.abs(regen - frozen)))
                plain_check[item['tag']] = diff
                need(diff <= PLAIN_CEILING, f'{item["tag"]} plain regeneration differs: {diff}')
                for name in ('flip', 'plain', 'receipt'):
                    key = f'{item["tag"]}_{name}'
                    flip_files[key] = item[name].relative_to(root).as_posix()
                    flip_hashes[key] = sha(item[name])
            s42 = bundle_helper.load_oof(flip[(42, f)]['flip'], fid)
            s43 = bundle_helper.load_oof(flip[(43, f)]['flip'], fid)
        c = bundle_helper.load_oof(local['convnext'], fid)
        d = bundle_helper.load_oof(local['dino_l2'], fid)
        if flip is None:
            ranks = fold_family_ranks(s42, s43, c, d)          # v1.0.0 plain rule
        else:
            ranks = serving_fold_ranks(s42, s43, c, d)         # serving aggregation
            rank_mean_r.append(fold_family_ranks(s42, s43, c, d)['R'])
        for name in sections:
            sections[name].append(ranks[name])
    pooled = {name: np.concatenate(parts) for name, parts in sections.items()}
    q_rcd, q_rc, ranked = arms(pooled['R'], pooled['C'], pooled['D'])
    refs_raw = e156.references(ids, root)
    refs = {name: refs_raw[key] for name, key in REFERENCE_KEYS}
    pair, status = compare(q_rcd, q_rc, refs, draws, ev)
    diagnostic = None
    if flip is not None:
        rm_rcd, rm_rc, rm_ranked = arms(np.concatenate(rank_mean_r), pooled['C'], pooled['D'])
        diagnostic = {
            'note': 'flip inputs with the v1.0.0 seed-rank-mean R; point estimates only, never '
                    'gated; separates the aggregation effect from the input effect',
            'q_rcd_minus_q_rc': point_deltas(rm_rcd, rm_rc, refs, ev),
            'standalone_R_macro': {r: folds._score(rm_ranked['l2'], ref, ev)['macro']
                                   for r, ref in refs.items()},
            'global_R_rank_max_abs_diff_vs_serving': float(
                np.max(np.abs(rm_ranked['l2'] - ranked['l2'])))}
    standalone = {name: {r: folds._score(value, ref, ev) for r, ref in refs.items()}
                  for name, value in (('R', ranked['l2']), ('C', ranked['convnext']),
                                      ('D', ranked['dino']), ('Q_RC', q_rc), ('Q_RCD', q_rcd))}
    continuity = {}
    # Identical arrays must give identical macros; R and Q_RCD only in plain mode.
    for ours, theirs in (('C', 'convnext'), ('D', 'dino_l2'),
                         *((('R', 'l2'), ('Q_RCD', 'new_bundle')) if r_mode == 'plain' else ())):
        diffs = {}
        for r in refs:
            a, b = standalone[ours][r]['macro'], prior['standalone'][theirs][r]['macro']
            need((a is None) == (b is None), f'{ours} macro definedness differs from EXP172')
            diffs[r] = 0.0 if a is None else abs(a - b)
        continuity[f'{ours}_vs_exp172_{theirs}'] = diffs
        need(max(diffs.values()) <= 1e-12, f'{ours} macro differs from EXP172 {theirs}')
    ruler = ruler_diagnostics(pair, status['status'], root)
    result = {
        'schema': SCHEMA, 'tool': SELF, 'tool_version': VERSION,
        'tool_sha256': sha(root / SELF) if (root / SELF).is_file() else None,
        'report_kind': REPORT_KIND[r_mode], 'r_mode': r_mode,
        'r_policy': R_POLICY[r_mode] if r_mode == 'plain' else FLIP_R_POLICY,
        'note': 'A plain-R pass alone does not qualify flip-R serving.',
        'formulas': FORMULAS if r_mode == 'plain' else FLIP_FORMULAS, 'gate': GATE,
        'status': {**status, 'review_flag': ruler.get('flag', 'NOT-CHECKED')},
        'primary': {'mean_delta': pair['primary_mean_delta'], 'ci95': pair['primary_mean_ci95'],
                    'dread_delta': pair['references']['dread']['delta']},
        'studies': len(ids), 'folds': list(FOLDS),
        'registration': reg,
        'roster_sha256': e172.uid_hash(ids),
        'components': {'path': ev.COMPONENTS, 'sha256': sha(root / ev.COMPONENTS),
                       'groups': len(groups)},
        'draws': {'unit': 'validation component', 'draws': DRAWS, 'seed': SEED,
                  'shared_by_both_arms': True, 'indices_sha256': draw_digest(draws)},
        'reference_sha256': refs_sha,
        'helper_code_sha256': helper_hashes(root),
        'protocol_sha256': {'exp172_protocol': sha(root / e172.PROTOCOL),
                            'exp172_result': sha(prior_path),
                            'exp172_collection': inputs['collection_sha256'],
                            'exp172_manifest': inputs['manifest_sha256'],
                            **{d: sha(root / d) for d in DOCUMENTS if (root / d).is_file()}},
        'input_files': {**files, **flip_files}, 'input_sha256': {**hashes, **flip_hashes},
        'r_inputs_used': ('EXP172-registered plain oof_k36.csv (l2_s42_f*/l2_s43_f*)'
                          if r_mode == 'plain' else 'flip-TTA files (*_flip keys)'),
        'flip_plain_regeneration_max_abs_diff': plain_check or None,
        'q_rcd_minus_q_rc': pair,
        'per_target_counts': target_counts(refs),
        'per_fold': fold_deltas(q_rcd, q_rc, refs,
                                {f: len(ids_by_fold[f]) for f in FOLDS}, ev),
        'standalone': standalone,
        'exp172_continuity_abs_macro_diff': continuity,
        'ruler_map': ruler,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'host': platform.node(),
                        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'omp_num_threads': os.environ.get('OMP_NUM_THREADS')},
        'finished_at': datetime.now(timezone.utc).isoformat(),
    }
    if r_mode == 'flip':
        result['serving_equivalence'] = {
            'helper': 'tools/exp169_release_v2.py:serving_family_rank',
            'checked': 'every fold, families R/C/D: identical average-tie order (rankdata equal)',
            'serving_code_sha256': serving_code_hashes(root)}
        result['seed_rank_mean_diagnostic'] = diagnostic
    if r_mode == 'flip' and (root / OUT['plain'] / 'result.json').is_file():
        plain = read(root / OUT['plain'] / 'result.json')
        rm = diagnostic['q_rcd_minus_q_rc']

        def effect(a, b):
            return None if a is None or b is None else a - b

        result['plain_vs_flip'] = {
            'plain_result_sha256': sha(root / OUT['plain'] / 'result.json'),
            'plain_registration_sha256': plain['registration']['sha256'],
            'flip_registration_sha256': reg['sha256'],
            'decomposition': 'input effect = flip_seed_rank_mean - plain; aggregation effect '
                             '= flip_serving - flip_seed_rank_mean (point deltas)',
            'references': {r: {'plain_delta': plain['q_rcd_minus_q_rc']['references'][r]['delta'],
                               'flip_seed_rank_mean_delta': rm['references'][r]['delta'],
                               'flip_delta': pair['references'][r]['delta'],
                               'input_effect': effect(
                                   rm['references'][r]['delta'],
                                   plain['q_rcd_minus_q_rc']['references'][r]['delta']),
                               'aggregation_effect': effect(pair['references'][r]['delta'],
                                                            rm['references'][r]['delta'])}
                           for r in refs},
            'primary_mean_delta': {'plain': plain['primary']['mean_delta'],
                                   'flip_seed_rank_mean': rm['primary_mean_delta'],
                                   'flip': pair['primary_mean_delta']},
            'status': {'plain': plain['status']['status'], 'flip': status['status']}}
    result['runtime_seconds'] = time.monotonic() - started
    write_new(out_dir / 'result.json', result)
    return {'status': status['status'], 'review_flag': result['status']['review_flag'],
            'result': str(out_dir / 'result.json'), 'mean_delta': pair['primary_mean_delta'],
            'ci95': pair['primary_mean_ci95'],
            'dread_delta': pair['references']['dread']['delta'],
            'runtime_seconds': result['runtime_seconds']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--r-mode', choices=sorted(OUT), required=True)
    ap.add_argument('--registration', type=Path, default=None,
                    help='default ' + REGISTRATION + ' (plain) or ' + FLIP_REGISTRATION + ' (flip)')
    ap.add_argument('--out-dir', type=Path, default=None,
                    help='default ' + OUT['plain'] + ' or ' + OUT['flip'])
    ap.add_argument('--register-only', action='store_true',
                    help='flip only: write the write-once flip registration and stop')
    a = ap.parse_args(argv)
    print(f'exp169_v2_dino_marginal v{VERSION} r-mode={a.r_mode}'
          + (' register-only' if a.register_only else ''), flush=True)
    if a.register_only:
        print(json.dumps(register_only(a.r_mode, ROOT, a.registration), indent=2))
    else:
        print(json.dumps(run(a.r_mode, ROOT, a.registration, a.out_dir), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
