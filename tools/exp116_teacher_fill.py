"""EXP116 (H-B-46) v1.1.2: targeted teacher fill. RadR50 trained exactly as EXP113 (local RTX 4070 Ti SUPER, EXP107 runner unchanged, seed 42,
EXP099 folds, RN_clip recipe, K12 / K24 / K36) on Dread labels in which only report-silent or disputed cells are replaced by
0.3 * Dread + 0.7 * calibrated teacher; the loss pos_weight stays the ORIGINAL Dread vector of each fold (the EXP113 manifest values).

  python tools/exp116_teacher_fill.py build-labels [--calibration soft|rint] [--dest DIR]   # CPU: per-fold label files + cell mask + receipt
  python tools/exp116_teacher_fill.py build-labels --reissue [--calibration soft|rint]      # new receipt for UNCHANGED labels (old kept as receipt_v<old>.json)
  python tools/exp116_teacher_fill.py prepare [--calibration soft|rint] [--dry-run DIR]     # register EXP116_TEACHER_FILL_PROTOCOL_V1
  python tools/exp116_teacher_fill.py prepare --supersede                                    # re-register before any run (old kept as manifest_v<old>.json)
  python tools/exp116_teacher_fill.py argv --fold K [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp116_teacher_fill.py validate --fold K               # runner --validate-only on CPU, no output
  python tools/exp116_teacher_fill.py smoke --fold K --out DIR [--train 16 --val 2 --epochs 1 --workers 2]   # CPU fit, DIR outside artifacts/
  python tools/exp116_teacher_fill.py queue-json --folds 0,1,2,3,4 --after TAG [--out FILE|-]   # tools/gpu_queue.py entries (not started)
  python tools/exp116_teacher_fill.py assemble [--runs-dir DIR]       # pooled OOF -> EXP116_TEACHER_FILL_OOF_LEAKY (diagnostic only)

Teacher: per target the equal mean of the percentile ranks (average rank / (n + 1)) of the TRUE OOF K36 predictions of CoAtNet5
(EXP099_B8_DREAD_OOF) and RadR50x5 (EXP107_RADR50_DREAD_OOF); every pooled row is re-checked against the fold file of the model that did
not train on it. Ranks remove each teacher's output scale, including the pos_weight shift of weighted BCE (D-A-047: q = w p / (1 - p + w p)),
and Platt scaling sigma(a * logit(rank mean) + b), fitted by Newton's method on the soft-target BCE, maps the score back to the Dread label
scale. --calibration soft (DEFAULT since v1.1.0): the fit target is the SOFT Dread value of the fitted rows, so the calibrated mean equals
the soft-Dread mean on the FITTED rows only (a rint fit instead moved the Synovitis calibrated mean 0.449 -> 0.312 while pos_weight stayed
Dread's 1.23). The fill then replaces only the masked cells, a selected subset whose calibrated-teacher mean differs from its Dread mean, so
the whole-target label level DOES shift on the filled targets (soft, global table: PF OA +0.019439, Effusion +0.009867, Synovitis +0.001990;
every table in the receipt's level_shift). That shift is registered as part of the intervention (the loss pos_weight stays the fixed Dread
vector); it is not a neutral rescaling. --calibration rint: the fit target is rint(float32 Dread), kept as an option under the '_rint'
suffix (label set teacher_fill_v1_rint, dir derived_teacher_fill_rint); it moves the level towards the rint prevalence.
Cross-fitted either way: the calibrator applied to the rows of fold j never saw a fold-j label.

Cells (the per-cell relabel_v1 states exist only in .local/relabel, which is not opened; the frozen conversion of tools/exp109_relabel_v1.py
"na -> 0.05; Synovitis na -> clip(0.6 x Effusion raw p)" is inverted from the stored CSV instead):
  PF OA      relabel_v1 == 0.05. Also catches 'neg' cells the labeller scored exactly 0.05 (stored HC-04 aggregate: 728 na + 172 neg = 900).
  Synovitis  S1 |Syn - clip(0.6 * Eff, .02, .98)| <= 1e-4 (Effusion mentioned: the CSV Effusion is the clipped raw p), or
             S2 Eff == 0.05 (Effusion itself 'na' or 'neg' at 0.05, raw p unknown) and 0.02 < Syn < 0.55 on the 0.006 grid (0.6 x a two-decimal
             raw p). On the 4,349 relabel rows: 3,731 cells vs 3,729 'na' and 714 vs 713 with Dread >= .5 in the stored aggregate (D-C-006).
  Effusion   Dread, Steven v4, Pilkwang v2 and relabel_v1 binarised at 0.5 do not all agree (all four present).
Baker's, Medial OA and every other target keep Dread. Rows outside the 4,343 G1 studies keep Dread.

Leak rule: see LEAK_RULE (one label file per student fold). The student OOF is LEAKY (teacher ancestry, D-B-024) and never decision
evidence; the experiment is judged on the public LB only. Quarantine (v1.1.1): training/runners/exp116_teacher_fill.py writes
DIAGNOSTIC_ONLY.json (tools/diagnostic_quarantine.py) into every fold run directory before the model is built, assemble writes one into
the OOF directory before any CSV, and tools/g1_compare.py / tools/exp103_distill_labels.py refuse inputs under such a marker (tools/
submission_screen.py checks it before loading any member / baseline / fold / teacher / replicate OOF: v1.1.2 records that). The inherited
EXP113 caveat "OOF is for ensembling / teacher pseudo-labels" is revoked for EXP116 (REVOKED_INHERITED_CAVEATS).
Nothing here trains on or touches the GPU.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp113_radr50_local as e113  # noqa: E402
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools import diagnostic_quarantine as dq  # noqa: E402
# names the shared EXP107 runner reads from its guard module (`guard.<name>`), re-exported unchanged -- except pos_weight (below)
from tools.exp113_radr50_local import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                       CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, contract, UID, T,
                                       DREAD, KEYS_FILE, FOLDS, EXTRA_EVAL, RUN_FILES, fold_contract, state_hashes, mismatched_keys, dump,
                                       train_python, within, display_path, _set_flag)

VERSION = '1.1.2'  # 1.1.2: assembled sources name each fold file (oof_k24 / oof_k36 beside their sha256), consumer status of
#                   tools/submission_screen.py corrected (it refuses quarantined inputs), reissue keeps labels_built_by; label bytes unchanged
#                   1.1.1: level-shift claims corrected + registered, DIAGNOSTIC_ONLY quarantine, reissue/supersede; label bytes unchanged
#                   1.1.0: --calibration soft (default) | rint; 1.0.0 was rint only
SCHEMA = 'exp116-teacher-fill-v1'
RECEIPT_SCHEMA = 'exp116-teacher-fill-receipt-v1'
CALIBRATIONS = ('soft', 'rint')
DEFAULT_CALIBRATION = 'soft'
LABEL_SET = 'teacher_fill_v1'  # bound to the registered label set by validate_arguments (the shared runner records guard.LABEL_SET)
BASE_PROTOCOL = e113.PROTOCOL
BASE_MANIFEST_SHA256 = 'a134327a10bfe051a0d58f3bb83969b2bdb8b9fcc4f54a309893504c960bd31d'
SHARED_RUNNER = e113.SHARED_RUNNER
PROTOCOL = 'artifacts/experiments/EXP116_TEACHER_FILL_PROTOCOL_V1'
QUEUE_FILE = PROTOCOL + '/queue-entries.json'
RUNNER = 'training/runners/exp116_teacher_fill.py'
TOOL = 'tools/exp116_teacher_fill.py'
QUARANTINE = 'tools/diagnostic_quarantine.py'  # marker writer used by the runner entry at run time (bound by sha256 in the registration)
SEED = e113.SEED
TAG_PREFIX = 'EXP116_TEACHER_FILL'
OOF_DIR = 'artifacts/experiments/EXP116_TEACHER_FILL_OOF_LEAKY'
LABEL_DIR = 'storage/labels/derived_teacher_fill'  # + '_rint' for the rint variant
MASK_NAME = 'cell_mask_v1.csv'
RECEIPT_NAME = 'receipt.json'
LOCAL_WORKERS = e113.LOCAL_WORKERS
DEFAULT_AFTER = 'EXP113_RADR50_LOCAL_F4'


def _suffix(calibration):
    if calibration not in CALIBRATIONS:
        raise ValueError(f'calibration must be one of {CALIBRATIONS}')
    return '' if calibration == DEFAULT_CALIBRATION else '_' + calibration


def label_name_for(calibration=DEFAULT_CALIBRATION):
    return 'teacher_fill_v1' + _suffix(calibration)


def label_dir_for(calibration=DEFAULT_CALIBRATION):
    return LABEL_DIR + _suffix(calibration)


def global_name(calibration=DEFAULT_CALIBRATION):
    return label_name_for(calibration) + '.csv'


def fold_name(k, calibration=DEFAULT_CALIBRATION):
    return f'{label_name_for(calibration)}_f{int(k)}.csv'


SOURCES = {'dread': DREAD,
           'steven_v4': 'storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
           'pilkwang_v2': 'storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv',
           'relabel_v1': 'storage/labels/derived_relabel/relabel_v1.csv'}
RELABEL_PROVENANCE = 'storage/labels/derived_relabel/provenance.json'
RELABEL_CONVERSION = 'p clipped to [0.02, 0.98]; na -> 0.05; Synovitis na -> 0.6 x Effusion p (clipped)'
TEACHERS = {'coatnet5': ('artifacts/experiments/EXP099_B8_DREAD_OOF', 'exp099-oof-v1'),
            'radr50x5': ('artifacts/experiments/EXP107_RADR50_DREAD_OOF', 'exp107-oof-v1')}
TEACHER_FILE = 'oof_k36.csv'
FILL_TARGETS = ('PF OA', 'Synovitis', 'Effusion')
DREAD_WEIGHT, TEACHER_WEIGHT = 0.3, 0.7
NA_VALUE = 0.05
SYN_FACTOR, CLIP_LO, CLIP_HI = 0.6, 0.02, 0.98
S1_TOL, GRID, GRID_TOL, S2_MAX = 1e-4, 0.006, 1e-6, 0.55
# stored aggregate of the raw relabel states over the 4,349 non-Gold rows (.local/hc04-label-semantics-v1/result.json, D-C-006); reference only
HC04_REFERENCE = {'PF OA': dict(na=728, neg_at_0_05=172, na_dread_ge_0_5=6),
                  'Synovitis': dict(na=3729, neg=55, unc=41, pos=524, na_dread_ge_0_5=713),
                  'Effusion': dict(na=422, neg_at_0_05=303)}
CSV_FLOAT = '%.8g'

LEAK_RULE = (
    'D-C-005 / D-B-024. Row rule: the fill value of a study comes only from the teacher fold models that did not train on it (true OOF, '
    're-checked per fold file) and from a Platt calibrator fitted without its label (cross-fitted), so a fill never encodes the study\'s own '
    'label. Training rule: the labels student fold k trains on contain fills for studies of folds j != k made by teacher fold-j models that '
    'trained on fold k\'s validation studies. D-C-005 allows existing outer-fold teacher targets as a training recipe, but the student\'s OOF '
    'on the same five folds is then invalid as evidence (D-B-024 VERIFIED: every student validation set lies inside the other four teachers\' '
    'training sets). Removing that path needs nested teachers (the D-C-005 pilot, about 3 GPU-hours for one outer fold), outside this CPU task. '
    'Conservative choice implemented: (1) one label file per student fold k in which fold-k rows (and the 6 non-G1 rows) stay Dread and both '
    'the teacher rank normalisation and every calibrator use only studies outside fold k (calibrator for fold-j rows: folds not in {j, k}), so '
    'no value in student k\'s labels depends on a fold-k label (only the teacher weights still do); (2) the student OOF is LEAKY (teacher '
    'ancestry, D-B-024) in every fold run directory and in ' + OOF_DIR + ' (assembled with decision_eligible false); each of those '
    'directories carries ' + dq.MARKER + ' from before its first prediction is written: never a ruler, comparison, gate, ensemble-weight fit, '
    'ensemble member, stacking input or teacher / pseudo-label source; (3) the experiment is judged on the public LB only (data no teacher '
    'saw), as pre-registered in H-B-46. teacher_fill_v1.csv (all rows cross-fitted over all folds) is the reference table of the receipt and is '
    'not used to train any fold.')
CHANGED_FIELDS = ('schema', 'created_at', 'plan_item', 'source_bindings', 'folds', 'tags', 'local', 'evaluation', 'caveats', 'derived_from',
                  'label_sets', 'label_set', 'paths', 'version', 'teacher_fill', 'leak_rule', 'quarantine', 'supersedes')
FOLD_CHANGED_FIELDS = ('tag', 'label_file', 'pos_weight_source', 'pos_weight_if_recomputed_from_fill')
CAVEATS = [
    'label intervention: Dread / consensus OOF cannot judge it (consensus excludes exactly these cells, D-B-035); the student OOF is also leaky '
    'by construction (D-B-024) -> judged on data no teacher saw (public), as pre-registered in H-B-46',
    'PF OA cells = relabel_v1 == 0.05, which also contains report-negated cells the labeller scored exactly 0.05 (172 of 900 by the stored '
    'HC-04 aggregate); the per-cell states are not used (.local/relabel not opened)',
    'Synovitis cells are inferred from the frozen conversion (S1/S2 rule); count 3,731 vs 3,729 stored na cells, so a handful of cells may be '
    'misclassified in either direction',
    'teacher = two Dread-trained image models (CoAtNet5 K36, RadR50x5 K36 from Kaggle T4); they inherit Dread\'s definitions, and the RadR50 '
    'teacher is the same family as the student',
    'per-fold label files: the five fold students train on fills calibrated on slightly different study sets (by design, see leak_rule)',
]
# EXP113 caveats that must NOT be inherited: EXP116 OOF is LEAKY (D-B-024), so it is never for ensembling or teacher pseudo-labels
REVOKED_INHERITED_CAVEATS = ('OOF is for ensembling / teacher pseudo-labels; not private evidence',)
QUARANTINE_CAVEAT = (
    'EXP116 OOF (every fold run directory and the assembled ' + OOF_DIR + ') is LEAKY by teacher ancestry (D-B-024) and diagnostic only: NOT '
    'for ensembling, teacher pseudo-labels, stacking, ensemble-weight fits, rulers, comparisons or gates (the inherited EXP113 caveat "OOF is '
    'for ensembling / teacher pseudo-labels" is revoked); each of those directories carries ' + dq.MARKER + ' written before training / before '
    'any CSV, tools/g1_compare.py and tools/exp103_distill_labels.py refuse such inputs, and the experiment is judged on the public LB only')
LEVEL_TEXT = {
    'soft': 'the soft Platt fit matches the calibrated mean to the soft-Dread mean on the FITTED rows only; only the masked cells are filled and '
            'their calibrated-teacher mean differs from their Dread mean, so the whole-target level shifts',
    'rint': 'the rint Platt fit matches the calibrated mean to the rint(Dread) prevalence on the fitted rows; only the masked cells are filled, '
            'so the whole-target level shifts towards the rint scale',
}
LEVEL_RULE = ('whole-target level shift of each filled target = mean(filled table) - mean(Dread) over the study set of the table = (cells / '
              'rows) x cell_mean_change. The Platt calibrator is calibrated in the large on its FITTED rows only; the masked cells are a selected '
              'subset, so the shift is not zero. It is part of the registered intervention (the loss pos_weight stays the fixed Dread vector), '
              'not a neutral rescaling.')


def level_shift(summaries):
    """Registered whole-target level shift per table and filled target (from the summaries of compute())."""
    per = {}
    for key, s in summaries.items():
        n = s['universe']
        per[key] = {t: dict(mean_dread=s['targets'][t]['mean_dread'], mean_filled=s['targets'][t]['mean_filled_table'],
                            shift=s['targets'][t]['mean_filled_table'] - s['targets'][t]['mean_dread'], cells=s['targets'][t]['cells'], rows=n,
                            cell_mean_change=s['targets'][t]['cell_mean_change']) for t in FILL_TARGETS}
    return dict(rule=LEVEL_RULE, per_table=per)


def level_shift_text(shift):
    """'PF OA +0.019439, Effusion +0.009867, Synovitis +0.001990': the global table's shifts, largest first."""
    g = shift['per_table']['global']
    return ', '.join(f"{t} {g[t]['shift']:+.6f}" for t in sorted(FILL_TARGETS, key=lambda t: -abs(g[t]['shift'])))


def caveats_for(shift, calibration):
    """EXP116 caveats appended to the filtered EXP113 list; the level caveat carries the registered global-table shifts."""
    return CAVEATS + [
        f'calibration scale and label level ({calibration}): {LEVEL_TEXT[calibration]} (global table: {level_shift_text(shift)}; every '
        'table in the receipt level_shift); this shift is part of the registered intervention (pos_weight stays the fixed Dread vector), not a '
        'neutral rescaling' + ('; the rint variant (label set teacher_fill_v1_rint) moves the level further, towards the rint prevalence'
                              if calibration == 'soft' else ''),
        QUARANTINE_CAVEAT]


def inherited_caveats(base_caveats):
    """The EXP113 caveats minus REVOKED_INHERITED_CAVEATS (each must be present: a changed base list is refused)."""
    missing = [c for c in REVOKED_INHERITED_CAVEATS if c not in base_caveats]
    if missing:
        raise ValueError(f'EXP113 caveat to revoke not found: {missing}')
    return [c for c in base_caveats if c not in REVOKED_INHERITED_CAVEATS]


# ------------------------------------------------------------------ names
def tag_for(fold, mode='full'):
    return f'{TAG_PREFIX}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def label_dir(root=ROOT, dest=None, calibration=DEFAULT_CALIBRATION):
    return Path(dest) if dest else Path(root) / label_dir_for(calibration)


def _manifest_path(root=ROOT):
    return Path(root) / PROTOCOL / 'manifest.json'


def _protocol(root=ROOT):
    return read(_manifest_path(root))


# ------------------------------------------------------------------ pure rules (unit-tested)
def pf_oa_silent(relabel_pf):
    """relabel_v1 PF OA cells written by the frozen 'na -> 0.05' conversion (includes 'neg' cells scored exactly 0.05)."""
    import numpy as np
    return np.abs(np.asarray(relabel_pf, np.float64) - NA_VALUE) < 5e-7


def synovitis_silent(syn, eff):
    """(mask, S1, S2) of relabel_v1 Synovitis cells written by 'Synovitis na -> clip(0.6 x Effusion raw p)' (see the module docstring).
    Not covered: Effusion 'na' with raw p <= 0.03 (Synovitis 0.02, same as a 'neg' scored 0.02), raw Effusion p > 0.98 (CSV Effusion 0.98,
    Synovitis 0.588-0.6) and an 'na' Effusion with raw p > 0.91 (Synovitis >= 0.55); compute() counts the stored cells of the last two
    patterns (uncovered_band_cells: 3 on relabel_v1, all Synovitis 0.6 with Effusion 0.05, i.e. more likely a 'pos' / 'unc'
    scored 0.6 than an 'na' Effusion with raw p 1.0)."""
    import numpy as np
    syn, eff = np.asarray(syn, np.float64), np.asarray(eff, np.float64)
    s1 = np.abs(syn - np.clip(SYN_FACTOR * eff, CLIP_LO, CLIP_HI)) <= S1_TOL
    grid = np.abs(syn / GRID - np.rint(syn / GRID)) < GRID_TOL
    s2 = (np.abs(eff - NA_VALUE) < 5e-7) & (syn > CLIP_LO + 1e-6) & (syn < S2_MAX) & grid & ~s1
    return s1 | s2, s1, s2


def effusion_disagree(stack):
    """stack (n_sources, n) of Effusion probabilities; True where all sources are present and their 0.5-binarisations are not unanimous."""
    import numpy as np
    stack = np.asarray(stack, np.float64)
    present = np.isfinite(stack).all(0)
    b = stack >= 0.5
    return present & ~(b.all(0) | (~b).all(0))


def percentile_rank(x):
    import numpy as np
    x = np.asarray(x, np.float64)
    return k99.avg_rank(x) / (len(x) + 1)


def teacher_score(*predictions):
    """Equal mean of the per-teacher percentile ranks (one target, one study set)."""
    import numpy as np
    return np.mean([percentile_rank(p) for p in predictions], axis=0)


def logit(p):
    import numpy as np
    p = np.asarray(p, np.float64)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    import numpy as np
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, np.float64)))  # 1 / (1 + exp(-z)) without overflow


def platt_fit(x, y, iters=100, ridge=1e-9):
    """(a, b) maximising the Bernoulli likelihood of y under sigmoid(a x + b) (Newton / IRLS, deterministic)."""
    import numpy as np
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) != len(y) or len(x) < 2 or not 0 < y.sum() < len(y):
        raise ValueError('Platt fit needs both classes')
    X = np.column_stack([x, np.ones_like(x)])
    w = np.zeros(2)
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-12:
            break
    return float(w[0]), float(w[1])


def cross_fitted_teacher(c, r, y, folds, scale=DEFAULT_CALIBRATION):
    """Calibrated teacher over one study set: rank mean of the two teachers' predictions on this set, then for every fold j a Platt
    calibrator fitted on the other folds' rows (soft-target BCE against y: the soft Dread value for scale 'soft', rint(float32 Dread)
    for 'rint') applied to the fold-j rows. Returns (calibrated, per-fold params)."""
    import numpy as np
    c, r, y = (np.asarray(v, np.float64) for v in (c, r, y))
    folds = np.asarray(folds)
    y_rint = np.rint(y.astype(np.float32)).astype(np.float64)
    x = logit(teacher_score(c, r))
    raw = (np.asarray(c, np.float64) + np.asarray(r, np.float64)) / 2
    cal = np.full(len(x), np.nan)
    params = {}
    for j in sorted(set(folds.tolist())):
        held, fit = folds == j, folds != j
        a, b = platt_fit(x[fit], y[fit])
        cal[held] = sigmoid(a * x[held] + b)
        pf, ph = sigmoid(a * x[fit] + b), cal[held]
        eps = 1e-12
        params[str(int(j))] = dict(
            scale=scale, a=a, b=b, n_fit=int(fit.sum()), target_mean_fit=float(y[fit].mean()), rint_prevalence_fit=float(y_rint[fit].mean()),
            mean_calibrated_fit=float(pf.mean()), n_heldout=int(held.sum()), target_mean_heldout=float(y[held].mean()),
            rint_prevalence_heldout=float(y_rint[held].mean()), mean_calibrated_heldout=float(ph.mean()),
            mean_raw_teacher_heldout=float(raw[held].mean()),
            logloss_heldout=float(-np.mean(y[held] * np.log(ph + eps) + (1 - y[held]) * np.log(1 - ph + eps))),
            brier_heldout=float(np.mean((ph - y[held]) ** 2)))
    if np.isnan(cal).any():
        raise AssertionError('uncalibrated rows')
    return cal, params


# ------------------------------------------------------------------ inputs
def _frame(path):
    """UID-indexed float64 frame of the 12 targets (plain or *_soft columns), original row order."""
    import pandas as pd
    df = pd.read_csv(path, dtype={UID: str})
    if df[UID].isna().any() or df[UID].duplicated().any():
        raise ValueError(f'missing or duplicate UID in {path}')
    cols = [t if t in df.columns else t + '_soft' for t in T]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    out = df.set_index(UID)[cols].astype('float64')
    out.columns = list(T)
    return out


def _source_path(root, path):
    p = Path(str(path).replace('\\', '/'))
    return p if p.is_absolute() else Path(root) / p


def load_teacher(root, directory, schema, val_ids):
    """Pooled K36 OOF of one teacher, verified as TRUE OOF: result.json hashes, and every fold's rows equal that fold's own file (the model
    that did not train on them) with exactly the fold's validation ids."""
    import numpy as np
    import pandas as pd
    root, d = Path(root), Path(root) / directory
    r = read(d / 'result.json')
    if r.get('schema') != schema or r.get('label_set') != 'dread' or r.get('k36_missing_folds'):
        raise ValueError(f'{directory}: expected a complete {schema} dread OOF')
    if sha256(d / TEACHER_FILE) != r['outputs'][TEACHER_FILE]:
        raise ValueError(f'{directory}/{TEACHER_FILE} differs from its result.json')
    pooled = pd.read_csv(d / TEACHER_FILE, dtype={UID: str})
    if list(pooled.columns) != [UID] + list(T) or pooled[UID].duplicated().any():
        raise ValueError(f'{directory}: unexpected OOF table')
    pooled = pooled.set_index(UID)
    if sorted(pooled.index) != sorted(u for v in val_ids.values() for u in v):
        raise ValueError(f'{directory}: OOF does not cover the fold studies exactly')
    per_fold = {}
    for k, ids in val_ids.items():
        src = r['sources'][str(k)]
        f = _source_path(root, src['oof_k36'])
        if sha256(f) != src['oof_k36_sha256']:
            raise ValueError(f'{directory} fold {k}: {f} differs from result.json')
        part = pd.read_csv(f, dtype={UID: str})
        if part[UID].tolist() != list(ids):
            raise ValueError(f'{directory} fold {k}: fold file rows are not the fold-{k} validation ids')
        if not np.array_equal(part[list(T)].to_numpy(np.float64), pooled.loc[list(ids), list(T)].to_numpy(np.float64)):
            raise ValueError(f'{directory} fold {k}: pooled rows differ from the fold file')
        per_fold[str(k)] = dict(file=display_path(f, root), sha256=src['oof_k36_sha256'], rows=len(ids))
    if not np.isfinite(pooled.to_numpy(np.float64)).all():
        raise ValueError(f'{directory}: non-finite teacher predictions')
    return pooled, dict(dir=directory, result_sha256=sha256(d / 'result.json'), oof=directory + '/' + TEACHER_FILE,
                        oof_sha256=r['outputs'][TEACHER_FILE], protocol_sha256=r.get('protocol_sha256'), per_fold=per_fold)


def load_inputs(root=ROOT):
    """Everything the fill reads, each bound by sha256 (EXP113 manifest, fold files, four label sets, relabel provenance, two teachers)."""
    import pandas as pd
    root = Path(root)
    base = root / BASE_PROTOCOL / 'manifest.json'
    if sha256(base) != BASE_MANIFEST_SHA256:
        raise ValueError('EXP113 manifest changed')
    m = read(base)
    val_ids, train_ids = {}, {}
    for k in FOLDS:
        f = m['folds'][str(k)]
        if sha256(root / f['ordered_ids']['path']) != f['ordered_ids']['sha256']:
            raise ValueError(f'fold {k} ordered ids changed')
        o = read(root / f['ordered_ids']['path'])
        if uid_hash(o['val']) != f['ordered_ids']['val_sha256'] or uid_hash(o['train']) != f['ordered_ids']['train_sha256']:
            raise ValueError(f'fold {k} ordered ids changed')
        val_ids[k], train_ids[k] = list(o['val']), list(o['train'])
    folds_csv = root / m['paths']['folds'] if 'folds' in m['paths'] else root / 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/folds.csv'
    if sha256(folds_csv) != m['fold_protocol']['folds_csv_sha256']:
        raise ValueError('EXP099 folds.csv changed')
    fc = pd.read_csv(folds_csv, dtype={UID: str})
    fold_of = dict(zip(fc[UID], fc['fold'].astype(int)))
    for k, ids in val_ids.items():
        if any(fold_of.get(u) != k for u in ids):
            raise ValueError(f'folds.csv disagrees with the fold-{k} validation ids')
    g1 = [u for k in FOLDS for u in val_ids[k]]
    if len(g1) != len(set(g1)) or set(g1) != set(fold_of):
        raise ValueError('fold validation sets do not partition the folds.csv studies')
    sources, hashes = {}, {}
    for name, path in SOURCES.items():
        sources[name] = _frame(root / path)
        hashes[path] = sha256(root / path)
        missing = [u for u in g1 if u not in sources[name].index]
        if missing:
            raise ValueError(f'{name} misses {len(missing)} G1 studies')
    if sources['dread'].isna().any().any():
        raise ValueError('Dread has missing values')
    prov = read(root / RELABEL_PROVENANCE)
    if prov.get('csv_sha256') != hashes[SOURCES['relabel_v1']] or prov.get('conversion') != RELABEL_CONVERSION:
        raise ValueError('relabel_v1 provenance does not bind the CSV or the frozen conversion changed')
    hashes[RELABEL_PROVENANCE] = sha256(root / RELABEL_PROVENANCE)
    teachers, teacher_meta = {}, {}
    for name, (directory, schema) in TEACHERS.items():
        teachers[name], teacher_meta[name] = load_teacher(root, directory, schema, val_ids)
        hashes[directory + '/' + TEACHER_FILE] = teacher_meta[name]['oof_sha256']
        hashes[directory + '/result.json'] = teacher_meta[name]['result_sha256']
    hashes[BASE_PROTOCOL + '/manifest.json'] = BASE_MANIFEST_SHA256
    hashes[display_path(folds_csv, root)] = sha256(folds_csv)
    return dict(manifest=m, val_ids=val_ids, train_ids=train_ids, fold_of={u: fold_of[u] for u in g1}, g1=g1, sources=sources, teachers=teachers,
                teacher_meta=teacher_meta, hashes=hashes, relabel_state_counts=prov.get('state_counts_all_reports'))


# ------------------------------------------------------------------ fill
def cell_masks(sources, ids):
    """{target: bool array over ids} plus rule detail counts."""
    import numpy as np
    R = sources['relabel_v1'].loc[ids]
    pf = pf_oa_silent(R['PF OA'].to_numpy())
    syn, s1, s2 = synovitis_silent(R['Synovitis'].to_numpy(), R['Effusion'].to_numpy())
    stack = np.stack([sources[n].loc[ids, 'Effusion'].to_numpy(np.float64) for n in ('dread', 'steven_v4', 'pilkwang_v2', 'relabel_v1')])
    eff = effusion_disagree(stack)
    detail = dict(synovitis_s1=int(s1.sum()), synovitis_s2=int(s2.sum()),
                  effusion_sources_positive={str(n): int(((stack >= 0.5).sum(0) == n)[eff].sum()) for n in (1, 2, 3)},
                  effusion_missing_source=int((~np.isfinite(stack).all(0)).sum()))
    return {'PF OA': pf, 'Synovitis': syn, 'Effusion': eff}, detail


def fill_table(inputs, masks, universe, calibration=DEFAULT_CALIBRATION):
    """Dread copy in which the masked cells of `universe` (a list of G1 ids) become 0.3 Dread + 0.7 calibrated teacher, calibrated and
    rank-normalised over `universe` only; the calibration target is the soft Dread value ('soft') or rint(float32 Dread) ('rint').
    Returns (table, calibration params, calibrated teacher per target)."""
    import numpy as np
    D = inputs['sources']['dread']
    out = D.copy()
    index = {u: i for i, u in enumerate(inputs['g1'])}
    rows = np.array([index[u] for u in universe])
    folds = np.array([inputs['fold_of'][u] for u in universe])
    if calibration not in CALIBRATIONS:
        raise ValueError(f'calibration must be one of {CALIBRATIONS}')
    soft = D.loc[universe, list(T)].to_numpy(np.float64)
    Y = soft if calibration == 'soft' else np.rint(soft.astype(np.float32)).astype(np.float64)
    params, calibrated = {}, {}
    for t in FILL_TARGETS:
        j = T.index(t)
        c = inputs['teachers']['coatnet5'].loc[universe, t].to_numpy(np.float64)
        r = inputs['teachers']['radr50x5'].loc[universe, t].to_numpy(np.float64)
        cal, params[t] = cross_fitted_teacher(c, r, Y[:, j], folds, calibration)
        calibrated[t] = dict(zip(universe, cal))
        sel = masks[t][rows]
        cells = [u for u, s in zip(universe, sel) if s]
        out.loc[cells, t] = DREAD_WEIGHT * D.loc[cells, t].to_numpy(np.float64) + TEACHER_WEIGHT * cal[sel]
    return out, params, calibrated


def summarize(table, inputs, masks, universe, train_ids=None):
    import numpy as np
    D = inputs['sources']['dread']
    index = {u: i for i, u in enumerate(inputs['g1'])}
    rows = np.array([index[u] for u in universe])
    per = {}
    for t in T:
        old_u, new_u = D.loc[universe, t].to_numpy(np.float64), table.loc[universe, t].to_numpy(np.float64)
        rec = dict(mean_dread=float(old_u.mean()), mean_filled_table=float(new_u.mean()))
        if t in FILL_TARGETS:
            sel = masks[t][rows]
            o, n = old_u[sel], new_u[sel]
            ro, rn = np.rint(o.astype(np.float32)), np.rint(n.astype(np.float32))
            rec.update(cells=int(sel.sum()), cell_mean_dread=float(o.mean()) if sel.any() else None, cell_mean_filled=float(n.mean()) if sel.any() else None,
                       cell_mean_change=float((n - o).mean()) if sel.any() else None, cell_mean_abs_change=float(np.abs(n - o).mean()) if sel.any() else None,
                       rint_flips_0_to_1=int(((ro == 0) & (rn == 1)).sum()), rint_flips_1_to_0=int(((ro == 1) & (rn == 0)).sum()))
            if not np.array_equal(old_u[~sel], new_u[~sel]):
                raise AssertionError(f'{t}: an unmasked cell changed')
        elif not np.array_equal(old_u, new_u):
            raise AssertionError(f'{t}: untouched target changed')
        per[t] = rec
    body = dict(universe=len(universe), targets=per)
    if train_ids is not None:
        filled = {u: table.loc[u, list(T)].to_numpy(np.float32) for u in train_ids}
        dread = {u: D.loc[u, list(T)].to_numpy(np.float32) for u in train_ids}
        body['pos_weight_fixed_dread'] = e107.pos_weight(dread, train_ids).tolist()
        body['pos_weight_if_recomputed_from_fill_not_used'] = e107.pos_weight(filled, train_ids).tolist()
    return body


def compute(inputs, calibration=DEFAULT_CALIBRATION):
    """All tables in memory: {'global': table, 0..4: per-student-fold tables}, the cell mask frame and the receipt body (no file hashes).
    The receipt also carries both scales' filled-cell summaries of the global table (scale_comparison) for the record."""
    import numpy as np
    import pandas as pd
    if calibration not in CALIBRATIONS:
        raise ValueError(f'calibration must be one of {CALIBRATIONS}')
    g1 = inputs['g1']
    masks, detail = cell_masks(inputs['sources'], g1)
    D = inputs['sources']['dread']
    R = inputs['sources']['relabel_v1']
    tables, params, summaries = {}, {}, {}
    tables['global'], params['global'], _ = fill_table(inputs, masks, g1, calibration)
    summaries['global'] = summarize(tables['global'], inputs, masks, g1)
    comparison = {}
    for other in CALIBRATIONS:
        table = tables['global'] if other == calibration else fill_table(inputs, masks, g1, other)[0]
        comparison[other] = {t: {k: v for k, v in s.items() if k.startswith('cell_') or k.startswith('rint_')}
                             for t, s in summarize(table, inputs, masks, g1)['targets'].items() if t in FILL_TARGETS}
    for k in FOLDS:
        universe = [u for u in g1 if inputs['fold_of'][u] != k]
        tables[k], params[f'fold{k}'], _ = fill_table(inputs, masks, universe, calibration)
        if not tables[k].loc[inputs['val_ids'][k]].equals(D.loc[inputs['val_ids'][k]]):
            raise AssertionError(f'fold {k} validation rows changed')
        summaries[f'fold{k}'] = summarize(tables[k], inputs, masks, universe, inputs['train_ids'][k])
    extra = [u for u in D.index if u not in set(g1)]
    for key, tab in tables.items():
        if not tab.loc[extra].equals(D.loc[extra]):
            raise AssertionError(f'{key}: a non-G1 row changed')
    mask_frame = pd.DataFrame({UID: g1, **{t: masks[t].astype(int) for t in FILL_TARGETS}})
    Rg = R.loc[g1]
    Dg = D.loc[g1]
    allrows_syn, _, _ = synovitis_silent(R['Synovitis'].to_numpy(), R['Effusion'].to_numpy())
    allrows_pf = pf_oa_silent(R['PF OA'].to_numpy())
    Dall = D.reindex(R.index)
    checks = dict(
        relabel_rows=len(R),
        synovitis_cells_all_relabel_rows=int(allrows_syn.sum()), synovitis_cells_dread_ge_0_5_all_relabel_rows=int((Dall['Synovitis'].to_numpy()[allrows_syn] >= 0.5).sum()),
        pf_oa_cells_all_relabel_rows=int(allrows_pf.sum()), pf_oa_cells_dread_ge_0_5_all_relabel_rows=int((Dall['PF OA'].to_numpy()[allrows_pf] >= 0.5).sum()),
        uncovered_band_cells=int((~allrows_syn & (((np.abs(R['Effusion'].to_numpy() - CLIP_HI) < 5e-7) & (R['Synovitis'].to_numpy() >= 0.588 - 1e-6)
                                                    & (R['Synovitis'].to_numpy() <= 0.6 + 1e-6))
                                                   | ((np.abs(R['Effusion'].to_numpy() - NA_VALUE) < 5e-7) & (R['Synovitis'].to_numpy() >= S2_MAX)
                                                      & (R['Synovitis'].to_numpy() <= 0.6 + 1e-6)
                                                      & (np.abs(R['Synovitis'].to_numpy() / GRID - np.rint(R['Synovitis'].to_numpy() / GRID)) < GRID_TOL)))).sum()),
        hc04_reference=HC04_REFERENCE,
        reading='HC-04 counts come from the raw states (D-C-006 audit, read there, not here): PF OA na + neg-at-0.05 = 900 cells; Synovitis na 3,729 '
                'and 713 with Dread >= .5. The rule counts on the same 4,349 rows are listed beside them.')
    cells = {t: dict(count=int(masks[t].sum()), of=len(g1), dread_ge_0_5=int((Dg[t].to_numpy()[masks[t]] >= 0.5).sum()),
                     relabel_ge_0_5=int((Rg[t].to_numpy()[masks[t]] >= 0.5).sum())) for t in FILL_TARGETS}
    shift = level_shift(summaries)
    receipt = dict(
        schema=RECEIPT_SCHEMA, version=VERSION, plan_item='H-B-46', label_name=label_name_for(calibration), calibration=calibration,
        fill=dict(rule=f'{DREAD_WEIGHT} * Dread + {TEACHER_WEIGHT} * calibrated teacher on the masked cells; every other cell = Dread',
                  targets=list(FILL_TARGETS), untouched=[t for t in T if t not in FILL_TARGETS],
                  teacher='equal mean of percentile ranks (avg rank / (n + 1)) of CoAtNet5 (EXP099 K36) and RadR50x5 (EXP107 K36) true OOF, per '
                          'target, over the study set of the table',
                  calibration='Platt: sigmoid(a * logit(rank mean) + b), Newton fit of the soft-target BCE against '
                              + ('the SOFT Dread value' if calibration == 'soft' else 'rint(float32 Dread)')
                              + ' of the fitted rows, per target, cross-fitted by fold (fold-j rows use a calibrator fitted on the other folds '
                                'of the table\'s study set)',
                  scale_rule=(('soft: mean calibrated = mean soft Dread on the FITTED rows only (calibration in the large of the '
                               'calibrator); the filled cells are a selected subset whose calibrated-teacher mean differs from their Dread '
                               'mean, so the whole-target level shifts (global table: ' if calibration == 'soft' else
                               'rint: mean calibrated = rint(Dread) prevalence on the fitted rows; only the masked cells are filled, so the '
                               'whole-target level moves towards the rint scale (global table: ')
                              + level_shift_text(shift) + '; level_shift) and this shift is part of the registered intervention'),
                  pos_weight_scale='D-A-047: ranks remove each teacher\'s scale (incl. the weighted-BCE shift q = wp/(1-p+wp)); Platt returns '
                                   'to the Dread label scale (mean_raw_teacher_heldout shows the raw distortion). Loss pos_weight = the '
                                   'original Dread vector (fixed)'),
        cells=dict(rules={'PF OA': 'relabel_v1 == 0.05', 'Synovitis': 'S1 |Syn - clip(0.6 Eff, .02, .98)| <= 1e-4, or S2 Eff == 0.05 and '
                          '0.02 < Syn < 0.55 and Syn on the 0.006 grid', 'Effusion': 'Dread, Steven v4, Pilkwang v2, relabel_v1 at 0.5 not unanimous'},
                   counts_g1=cells, detail=detail, checks=checks),
        tables={'global': dict(file=global_name(calibration), use='reference (all G1 rows cross-fitted over all folds); not used to train any fold'),
                **{f'fold{k}': dict(file=fold_name(k, calibration), use=f'training labels of student fold {k} (fold-{k} rows = Dread; teacher '
                                                                        f'ranks and calibrators use only studies outside fold {k})') for k in FOLDS}},
        summaries=summaries, level_shift=shift, calibration_params=params, scale_comparison=dict(table='global', per_scale=comparison),
        leak_rule=LEAK_RULE, student_oof=dict(decision_eligible=False, status='LEAKY (teacher ancestry, D-B-024): diagnostics only',
                                              judge='public LB only', marker=dq.MARKER),
        inputs={k: v for k, v in sorted(inputs['hashes'].items())}, teachers=inputs['teacher_meta'])
    return tables, mask_frame, receipt


def table_csv(table):
    out = table.reset_index()
    out.columns = [UID] + list(T)
    return out.to_csv(index=False, float_format=CSV_FLOAT, lineterminator='\n')


def label_texts(tables, mask_frame, calibration):
    """{file name: exact CSV text} of every label output (reference table, five fold files, cell mask)."""
    return {global_name(calibration): table_csv(tables['global']), **{fold_name(k, calibration): table_csv(tables[k]) for k in FOLDS},
            MASK_NAME: mask_frame.to_csv(index=False, lineterminator='\n')}


def dump_lf(path, value, replace=False):
    """JSON as the EXP099 dump (indent 2, UTF-8, no NaN) with LF newlines; exclusive create, or an atomic replace when replace=True."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    target = path.with_name(f'{path.name}.tmp-{os.getpid()}') if replace else path
    with open(target, 'x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    if replace:
        os.replace(target, path)
    return path


def superseded_note(old_sha, new_path, reason, **extra):
    return dict(status='SUPERSEDED', superseded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), superseded_by=dict(path=new_path, version=VERSION),
                reason=reason, original_sha256=old_sha, note='this file is the previous version kept for the record (original bytes: '
                'original_sha256); it is not used by any run', **extra)


SUPERSEDE_REASONS = {
    '1.1.1': ('v1.1.1 correction (astra re-review 2026-10-08, .local/astra-20261007/astra-coderereview-a-out.md section 3): the v1.1.0 '
              'text claimed the soft fill preserves the label level ("not the target\'s level"); the calibrator preserves the mean on '
              'its FITTED rows only and the whole-target level shifts on the filled targets (registered as level_shift, part of the '
              'intervention); and the inherited EXP113 caveat "OOF is for ensembling / teacher pseudo-labels" contradicted the LEAKY '
              '(D-B-024) diagnostic-only status of the student OOF, now quarantined with DIAGNOSTIC_ONLY.json markers.'),
    '1.1.2': ('v1.1.2 correction (astra batch re-review 2026-10-08, .local/astra-20261007/astra-batch-rereview-out.md section C): the '
              'assembled OOF result.json recorded oof_k36_sha256 but not the fold file path oof_k36, so tools/submission_screen.py could '
              'not bind the fold runs (FAIL_LINEAGE even for a correctly quarantined EXP116 with an explicit public basis) - assemble now '
              'writes oof_k24 / oof_k36 beside their sha256 (C1); the registration still listed tools/submission_screen.py as a pending '
              'consumer with "no marker check yet" although it checks DIAGNOSTIC_ONLY.json before loading any OOF - now listed under '
              'consumers_refusing (C2); a second reissue keeps the labels_built_by record of the version that built the labels. No '
              'label byte, fold, recipe, seed, weight or pos_weight changes; no EXP116 run exists.'),
}
SUPERSEDE_REASON = SUPERSEDE_REASONS[VERSION]


def build_labels(root=ROOT, dest=None, calibration=DEFAULT_CALIBRATION):
    """Write the per-fold label files, the reference table, the cell mask and the receipt (refuses an existing destination)."""
    import hashlib
    import numpy as np
    root = Path(root)
    out = label_dir(root, dest, calibration)
    if out.exists():
        raise SystemExit('refusing to overwrite ' + str(out))
    inputs = load_inputs(root)
    tables, mask_frame, receipt = compute(inputs, calibration)
    texts = label_texts(tables, mask_frame, calibration)
    out.mkdir(parents=True)
    for name, text in texts.items():
        with open(out / name, 'x', encoding='utf-8', newline='') as f:
            f.write(text)
    D, _ = k99._load_labels(root / DREAD)
    outputs = {}
    for name, key in [(global_name(calibration), 'global')] + [(fold_name(k, calibration), k) for k in FOLDS]:
        labels, semantic = k99._load_labels(out / name)
        expect = tables[key]
        if set(labels) != set(D):
            raise AssertionError(f'{name}: row set differs from Dread')
        worst = max(float(np.abs(labels[u].astype(np.float64) - expect.loc[u, list(T)].to_numpy(np.float64)).max()) for u in labels)
        if worst > 1e-6:
            raise AssertionError(f'{name}: written values differ from the computed table by {worst}')
        same = [u for u in labels if not (expect.loc[u, list(T)].to_numpy(np.float64) != inputs['sources']['dread'].loc[u, list(T)].to_numpy(np.float64)).any()]
        if any(not np.array_equal(labels[u], D[u]) for u in same):
            raise AssertionError(f'{name}: an unfilled row does not equal Dread in float32')
        outputs[name] = dict(sha256=hashlib.sha256(texts[name].encode('utf-8')).hexdigest(), semantic_sha256=semantic, rows=len(labels))
        if sha256(out / name) != outputs[name]['sha256']:
            raise AssertionError('written bytes differ')
    outputs[MASK_NAME] = dict(sha256=sha256(out / MASK_NAME), rows=len(mask_frame))
    receipt.update(built_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), label_dir=display_path(out, root), outputs=outputs,
                   tool=TOOL, tool_sha256=sha256(root / TOOL))
    dump_lf(out / RECEIPT_NAME, receipt)
    brief = dict(label_dir=display_path(out, root), calibration=calibration, cells=receipt['cells']['counts_g1'], detail=receipt['cells']['detail'],
                 checks={k: v for k, v in receipt['cells']['checks'].items() if k not in ('hc04_reference', 'reading')},
                 scale_comparison={c: {t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for t, s in per.items()}
                                   for c, per in receipt['scale_comparison']['per_scale'].items()},
                 level_shift_global=level_shift_text(receipt['level_shift']),
                 outputs={k: v['sha256'] for k, v in outputs.items()}, receipt_sha256=sha256(out / RECEIPT_NAME))
    print(json.dumps(brief, indent=1))
    return receipt


def reissue_receipt(root=ROOT, dest=None, calibration=DEFAULT_CALIBRATION):
    """New receipt (current VERSION text, level_shift, quarantine status) for an EXISTING label directory whose label files stay
    byte-identical: every stored file must equal both the old receipt's sha256 and a rebuild from the bound inputs, and the inputs must be
    those the old receipt recorded. The old receipt is kept as receipt_v<old version>.json with a SUPERSEDED note (written first); the new
    receipt.json binds it by sha256 under `supersedes`. Nothing else in the directory is written."""
    import hashlib
    root = Path(root)
    out = label_dir(root, dest, calibration)
    current = out / RECEIPT_NAME
    if not current.is_file():
        raise SystemExit('no receipt to reissue: ' + str(current))
    old, old_sha = read(current), sha256(current)
    if old.get('schema') != RECEIPT_SCHEMA or old.get('calibration') != calibration or old.get('label_name') != label_name_for(calibration):
        raise ValueError('unexpected receipt (schema / calibration / label name)')
    if old.get('version') == VERSION:
        raise SystemExit(f'{current} is already v{VERSION}')
    archive = out / f"receipt_v{old['version']}.json"
    if archive.exists():
        raise SystemExit('refusing to overwrite ' + str(archive))
    inputs = load_inputs(root)
    tables, mask_frame, receipt = compute(inputs, calibration)
    if receipt['inputs'] != old.get('inputs'):
        raise ValueError('the bound inputs differ from those the labels were built from')
    texts = label_texts(tables, mask_frame, calibration)
    if set(texts) != set(old['outputs']):
        raise ValueError('the output set differs from the old receipt')
    for name, text in texts.items():
        rebuilt, stored, recorded = hashlib.sha256(text.encode('utf-8')).hexdigest(), sha256(out / name), old['outputs'][name]['sha256']
        if not rebuilt == stored == recorded:
            raise ValueError(f'{name}: rebuild {rebuilt[:12]}, stored {stored[:12]} and old receipt {recorded[:12]} differ (labels must stay '
                             'byte-identical)')
        if name != MASK_NAME and k99._load_labels(out / name)[1] != old['outputs'][name]['semantic_sha256']:
            raise ValueError(f'{name}: semantic hash differs from the old receipt')
    note = superseded_note(old_sha, display_path(current, root), SUPERSEDE_REASON,
                           label_files='unchanged: byte-identical, re-verified against a rebuild from the same bound inputs')
    dump_lf(archive, {'SUPERSEDED': note, **old})
    # the version that BUILT the labels is carried forward through every reissue (a reissued receipt is not a build)
    built_by = copy.deepcopy(old['labels_built_by']) if isinstance(old.get('labels_built_by'), dict) else dict(
        version=old['version'], tool_sha256=old.get('tool_sha256'), built_at=old['built_at'],
        note='label bytes unchanged since; this receipt only corrects the text and adds level_shift / student_oof')
    receipt.update(built_at=old['built_at'], reissued_at=note['superseded_at'], label_dir=display_path(out, root), outputs=copy.deepcopy(old['outputs']),
                   tool=TOOL, tool_sha256=sha256(root / TOOL), labels_built_by=built_by,
                   supersedes=dict(path=display_path(archive, root), version=old['version'], sha256=sha256(archive), original_sha256=old_sha,
                                   reason=SUPERSEDE_REASON))
    dump_lf(current, receipt, replace=True)
    brief = dict(label_dir=display_path(out, root), calibration=calibration, version=VERSION, receipt_sha256=sha256(current),
                 superseded=dict(path=display_path(archive, root), sha256=sha256(archive), original_sha256=old_sha),
                 level_shift_global=level_shift_text(receipt['level_shift']), outputs_unchanged={k: v['sha256'] for k, v in old['outputs'].items()})
    print(json.dumps(brief, indent=1))
    return receipt


# ------------------------------------------------------------------ prepare (protocol registration)
def started_runs(root=ROOT):
    """EXP116 outputs that bind a registration by sha256 (full fold run directories, the assembled OOF directory)."""
    root = Path(root)
    return [display_path(d, root) for d in [root / 'artifacts' / 'experiments' / tag_for(k) for k in FOLDS] + [root / OOF_DIR] if d.exists()]


def prepare(root=ROOT, dest=None, labels_dest=None, calibration=DEFAULT_CALIBRATION, supersede=False):
    """Register the protocol (manifest.json). supersede=True replaces an existing registration of an older VERSION before any EXP116 run
    exists: the old manifest is kept as manifest_v<old version>.json with a SUPERSEDED note and the new manifest binds it by sha256."""
    import hashlib
    import numpy as np
    import pandas as pd
    root = Path(root)
    destination = Path(dest) if dest else root / PROTOCOL
    previous = None
    if destination.exists() and not supersede:
        raise SystemExit('Refusing to overwrite existing registration: ' + str(destination))
    if supersede:
        current = destination / 'manifest.json'
        if not current.is_file():
            raise SystemExit('nothing to supersede: ' + str(current))
        old = read(current)
        if old.get('schema') != SCHEMA:
            raise ValueError('the registration to supersede is not an EXP116 manifest')
        if old.get('version') == VERSION:
            raise SystemExit(f'{current} is already v{VERSION}')
        runs = started_runs(root)
        if runs:
            raise SystemExit('refusing to supersede: EXP116 outputs already bind the registration: ' + ', '.join(runs))
        archive = destination / f"manifest_v{old.get('version', 'unversioned')}.json"
        if archive.exists():
            raise SystemExit('refusing to overwrite ' + str(archive))
        previous = dict(path=current, archive=archive, body=old, sha=sha256(current))
    base = root / BASE_PROTOCOL / 'manifest.json'
    if sha256(base) != BASE_MANIFEST_SHA256:
        raise ValueError('EXP113 manifest changed')
    bp = read(base)
    if bp['schema'] != e113.SCHEMA or bp['seed'] != SEED or bp['tags']['prefix'] != e113.TAG_PREFIX or bp['label_set']['name'] != 'dread':
        raise ValueError('unexpected EXP113 registration')
    for path, expected in bp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('EXP113 binding changed: ' + path)
    for name, expected in bp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    if calibration not in CALIBRATIONS:
        raise ValueError(f'calibration must be one of {CALIBRATIONS}')
    ldir = label_dir(root, labels_dest, calibration)
    lrel = display_path(ldir, root)
    receipt_path = ldir / RECEIPT_NAME
    if not receipt_path.is_file():
        raise SystemExit(f'build-labels --calibration {calibration} first: ' + str(receipt_path))
    receipt = read(receipt_path)
    if receipt.get('schema') != RECEIPT_SCHEMA:
        raise ValueError('unexpected receipt')
    if receipt.get('version') != VERSION or 'level_shift' not in receipt:
        raise SystemExit(f"receipt is v{receipt.get('version')}, the tool is v{VERSION}: build-labels --reissue --calibration {calibration} first")
    if receipt.get('calibration') != calibration or receipt.get('label_name') != label_name_for(calibration):
        raise ValueError(f"receipt calibration {receipt.get('calibration')!r} / label {receipt.get('label_name')!r} is not the requested {calibration!r}")
    label_name = label_name_for(calibration)
    for name, rec in receipt['outputs'].items():
        if sha256(ldir / name) != rec['sha256']:
            raise ValueError(f'{name} differs from the receipt')
    inputs = load_inputs(root)  # re-verifies every input hash, then rebuilds in memory: the stored files must be reproducible byte for byte
    tables, mask_frame, _ = compute(inputs, calibration)
    rebuilt = label_texts(tables, mask_frame, calibration)
    for name, text in rebuilt.items():
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != receipt['outputs'][name]['sha256']:
            raise ValueError(f'{name}: rebuilding from the bound inputs gives different bytes')
    gold = k99.gold_uids(root)
    dread_labels, _ = k99._load_labels(root / DREAD)
    folds = {}
    for k in FOLDS:
        f = copy.deepcopy(bp['folds'][str(k)])
        train = read(root / f['ordered_ids']['path'])['train']
        path = (Path(lrel) / fold_name(k, calibration)).as_posix()
        labels, semantic = k99._load_labels(relpath(root, path))
        raw =pd.read_csv(relpath(root, path), dtype={UID: str}, usecols=[UID])[UID].astype(str).tolist()
        k99.check_label_set(labels, raw, gold, {k: train})
        if semantic != receipt['outputs'][fold_name(k, calibration)]['semantic_sha256']:
            raise ValueError(f'fold {k}: semantic hash differs from the receipt')
        if not np.array_equal(e107.pos_weight(dread_labels, train), np.asarray(f['pos_weight'], np.float32)):
            raise ValueError(f'fold {k}: Dread pos_weight differs from the EXP113 registration')
        f.update(tag=tag_for(k), label_file=dict(name=label_name, calibration=calibration, path=path, sha256=sha256(relpath(root, path)),
                                                semantic_sha256=semantic, fold=k),
                 pos_weight_source='dread: the EXP113 fold vector, fixed (the guard\'s pos_weight applies the EXP099 rule to the Dread labels)',
                 pos_weight_if_recomputed_from_fill=e107.pos_weight(labels, train).tolist())
        folds[str(k)] = f
    extra = [BASE_PROTOCOL + '/manifest.json', RUNNER, TOOL, QUARANTINE, *[(Path(lrel) / n).as_posix() for n in receipt['outputs']],
             (Path(lrel) / RECEIPT_NAME).as_posix(), *sorted(k for k in inputs['hashes'] if k not in bp['source_bindings'])]
    bindings = {**bp['source_bindings'], **{s: sha256(relpath(root, s)) for s in extra}}
    manifest = copy.deepcopy(bp)
    manifest.update(
        schema=SCHEMA, version=VERSION, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-46',
        source_bindings=bindings, folds=folds,
        label_set=dict(name=label_name, calibration=calibration, per_fold=True, dir=lrel, receipt=(Path(lrel) / RECEIPT_NAME).as_posix(),
                       receipt_sha256=sha256(receipt_path), reference_table=(Path(lrel) / global_name(calibration)).as_posix(),
                       rule='student fold K trains on folds[K].label_file (eval-labels = the same file; the shared runner only hash-checks it); '
                            'the reference table is never a training file'),
        paths={**bp['paths'], 'labels': None, 'eval_labels': None, 'labels_rule': 'per fold: folds[K].label_file.path (view_for(K))'},
        label_sets=None,
        teacher_fill=dict(receipt=(Path(lrel) / RECEIPT_NAME).as_posix(), calibration=calibration, weights=dict(dread=DREAD_WEIGHT, teacher=TEACHER_WEIGHT),
                          scale_comparison=receipt['scale_comparison'], level_shift=receipt['level_shift'],
                          targets=list(FILL_TARGETS), untouched=[t for t in T if t not in FILL_TARGETS], cells=receipt['cells']['counts_g1'],
                          teachers={n: v[0] + '/' + TEACHER_FILE for n, v in TEACHERS.items()}, fill=receipt['fill']),
        leak_rule=LEAK_RULE,
        tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=OOF_DIR, exp113_prefix=bp['tags']['prefix']),
        local={**bp['local'], 'runner': RUNNER, 'guard': TOOL, 'shared_runner': SHARED_RUNNER,
               'runner_rule': 'training/runners/exp116_teacher_fill.py points the shared runner\'s guard import (tools.exp107_kfold_radr50) at '
                              'tools/exp116_teacher_fill.py for its own process and calls the unchanged EXP107 runner main(); OMP/MKL/OPENBLAS/'
                              'NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 are supplied when absent; the guard\'s pos_weight returns the Dread '
                              'vector, so the runner\'s equality check binds the fixed Dread weighting',
               'queue': 'tools/exp116_teacher_fill.py queue-json -> entries for tools/gpu_queue.py (after = the current chain tail; '
                        'continue_on_failure true)'},
        evaluation={**bp['evaluation'], 'oof_dir': OOF_DIR, 'decision_eligible': False,
                    'label_sets': 'scored with Dread (np.rint) as EXP099, for diagnostics only: the OOF is leaky (leak_rule)',
                    'judge': 'public LB only; the student OOF is LEAKY (teacher ancestry, D-B-024) and never decision evidence'},
        caveats=inherited_caveats(bp['caveats']) + caveats_for(receipt['level_shift'], calibration),
        quarantine=dict(
            decision_eligible=False, status='LEAKY (teacher ancestry, D-B-024): diagnostics only; judged on the public LB only', marker=dq.MARKER,
            marker_module=QUARANTINE, revoked_inherited_caveats=list(REVOKED_INHERITED_CAVEATS),
            fold_runs=RUNNER + ' writes ' + dq.MARKER + ' into each fold run directory the moment the shared runner opens train.log there (the '
                      'directory was just created empty; the model is not built yet) and, after a completed run, adds the sha256 of every '
                      'output (oof_best.csv, ' + EXTRA_EVAL['file'] + ') to it; verify_run refuses a fold directory without it',
            assembled=OOF_DIR + ' gets ' + dq.MARKER + ' before any CSV; result.json records decision_eligible false',
            shared_runner_caveat='the shared runner writes summary.json caveat "weak-label fold OOF for ensembling / teacher pseudo-labels" (an '
                                 'unchanged, sha256-bound EXP107 file); for EXP116 that inherited text is overridden by ' + dq.MARKER,
            consumers_refusing=['tools/g1_compare.py', 'tools/exp103_distill_labels.py', 'tools/submission_screen.py'],
            consumers_pending=[],
            consumer_rules={'tools/submission_screen.py': 'checks the marker / decision_eligible records before loading any member, baseline, fold, '
                                                          'teacher or replicate OOF: a quarantined member or baseline disables OOF and replication '
                                                          'scoring (PUBLIC-ONLY at best, with an explicit public basis), a quarantined teacher is '
                                                          'FAIL_LINEAGE, a replicate with a quarantined pooled / fold / run-directory / teacher '
                                                          'source is skipped with the reason; metadata stays traceable for lineage'},
            forbidden_uses=list(dq.FORBIDDEN_USES)),
        derived_from=dict(path=BASE_PROTOCOL + '/manifest.json', sha256=BASE_MANIFEST_SHA256, tag_prefix=bp['tags']['prefix'],
                          changed_fields=list(CHANGED_FIELDS), fold_changed_fields=list(FOLD_CHANGED_FIELDS), changed_args={},
                          unchanged='seed, folds, recipe (common_args), initial state (EXP113 keys file), environment, weights, K, windows, '
                                    'contract and the Dread pos_weight vectors'))
    if any(c in manifest['caveats'] for c in REVOKED_INHERITED_CAVEATS):
        raise AssertionError('a revoked caveat survived')
    if previous:
        note = superseded_note(previous['sha'], display_path(previous['path'], root), SUPERSEDE_REASON,
                               runs_started=False, label_files='unchanged (the new registration binds the same label files)')
        dump_lf(previous['archive'], {'SUPERSEDED': note, **previous['body']})
        manifest['supersedes'] = dict(path=display_path(previous['archive'], root), version=previous['body'].get('version'),
                                      sha256=sha256(previous['archive']), original_sha256=previous['sha'],
                                      created_at=previous['body'].get('created_at'), reason=SUPERSEDE_REASON)
        dump_lf(previous['path'], manifest, replace=True)
    else:
        destination.mkdir(parents=True)
        dump_lf(destination / 'manifest.json', manifest)
    print(json.dumps(dict(status='REGISTERED_NO_EXECUTION', version=VERSION, calibration=calibration, label_set=label_name,
                          protocol=display_path(destination, root), supersedes=manifest.get('supersedes', {}).get('path'),
                          manifest_sha256=sha256(destination / 'manifest.json'), tags=[f['tag'] for f in folds.values()],
                          label_files={k: f['label_file']['sha256'] for k, f in folds.items()}), indent=1))
    return manifest


# ------------------------------------------------------------------ runner guard (called by the shared EXP107 runner as `guard`)
_DREAD_CACHE = {}


def pos_weight(labels, train_ids):
    """EXP116: the loss pos_weight is fixed to the ORIGINAL Dread values -> the EXP099 rule applied to the Dread labels of train_ids (the
    filled `labels` must still cover every id). The shared runner compares this with the manifest fold vector (= EXP113's)."""
    missing = [u for u in train_ids if u not in labels]
    if missing:
        raise ValueError(f'label file misses {len(missing)} training ids')
    if 'dread' not in _DREAD_CACHE:
        _DREAD_CACHE['dread'] = k99._load_labels(ROOT / DREAD)[0]
    return e107.pos_weight(_DREAD_CACHE['dread'], train_ids)


def protocol_view(p, fold):
    """The protocol as the shared runner must see it for one fold: label_set and paths.labels / eval_labels = that fold's label file."""
    view = copy.deepcopy(p)
    lf = p['folds'][str(fold)]['label_file']
    view['label_set'] = dict(name=p['label_set']['name'], calibration=p['label_set'].get('calibration'), path=lf['path'], sha256=lf['sha256'],
                             semantic_sha256=lf['semantic_sha256'], fold=int(fold))
    view['paths'] = {**p['paths'], 'labels': lf['path'], 'eval_labels': lf['path']}
    return view


def validate_arguments(args, root):
    """Called by the runner before any training import. Returns (protocol view for this fold, fold spec, ordered ids, output root)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p.get('schema') != SCHEMA or manifest_path.resolve() != _manifest_path(root).resolve():
        raise ValueError('wrong protocol')
    if p.get('derived_from', {}).get('sha256') != BASE_MANIFEST_SHA256 or p['tags']['prefix'] != TAG_PREFIX:
        raise ValueError('not the EXP116 registration')
    if p.get('version') != VERSION or (p.get('quarantine') or {}).get('decision_eligible') is not False:
        raise ValueError(f"registration v{p.get('version')} is not the v{VERSION} quarantined registration (prepare --supersede)")
    calibration = p['label_set'].get('calibration')
    if calibration not in CALIBRATIONS or p['label_set']['name'] != label_name_for(calibration):
        raise ValueError('label set / calibration of the registration are inconsistent')
    global LABEL_SET
    LABEL_SET = p['label_set']['name']  # recorded by the shared runner as guard.LABEL_SET
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol: KNEE_REMOTE_RELAX_ENV must not be set (environment and initial state are enforced)')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
    smoke = args.mode == 'smoke'
    for k, v in p['common_args'].items():
        got = getattr(args, k, None)
        if got != v and not (smoke and k in p['smoke_allowed'] and got in p['smoke_allowed'][k]):
            raise ValueError('frozen argument changed: ' + k)
    if args.seed != p['seed']:
        raise ValueError('seed changed')
    w = p['resource_args']['workers']
    if not w['min'] <= args.workers <= w['max']:
        raise ValueError('workers outside the registered range')
    if str(args.fold) not in p['folds']:
        raise ValueError('unregistered fold')
    view = protocol_view(p, args.fold)
    fold = view['folds'][str(args.fold)]
    if tag_for(args.fold, args.mode) != args.tag or (not smoke and fold['tag'] != args.tag):
        raise ValueError('tag/fold/mode mismatch')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, view['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    if sha256(relpath(root, view['label_set']['path'])) != view['label_set']['sha256']:
        raise ValueError('fold label file changed')
    weights = Path(args.init_weights)
    if weights.resolve() != e113.weights_path(root).resolve():
        raise ValueError('init weights must be the local RadImageNet file ' + e113.WEIGHTS_LOCAL)
    if not weights.is_file() or sha256(weights) != p['init_weights']['sha256'] or p['init_weights']['sha256'] != e113.RADIMAGENET_SHA256:
        raise ValueError('init weights differ from the registered RadImageNet file')
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    for key in ('ordered_ids', 'groups'):
        if sha256(relpath(root, fold[key]['path'])) != fold[key]['sha256']:
            raise ValueError(f'fold {args.fold} {key} file changed')
    ordered = read(relpath(root, fold['ordered_ids']['path']))
    for k in ('train', 'val'):
        if uid_hash(ordered[k]) != fold['ordered_ids'][k + '_sha256']:
            raise ValueError('ordered fold ids changed')
    default_root = (root / 'artifacts' / 'experiments').resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else default_root
    if smoke:
        if not args.output_root or within(out_root, root / 'artifacts'):
            raise ValueError('smoke runs need --output-root outside artifacts/')
        if args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke needs >= one batch of training studies, >= 1 validation study and >= 1 epoch')
    elif out_root != default_root:
        raise ValueError('local full runs write to artifacts/experiments only (no --output-root)')
    return view, fold, ordered, out_root


def view_for(fold, root=ROOT):
    return protocol_view(_protocol(root), fold)


def runner_argv(view, fold, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None):
    """Runner entry argv for one fold; `view` = view_for(fold) (its paths are that fold's label file)."""
    root = Path(root)
    argv = [python or train_python(), str(root / RUNNER), '--protocol', str(root / PROTOCOL / 'manifest.json'),
            '--fold', str(int(fold)), '--mode', mode, '--tag', tag_for(fold, mode), '--seed', str(view['seed']),
            '--workers', str(LOCAL_WORKERS if workers is None else int(workers))]
    for k, v in view['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(relpath(root, view['paths']['corpus'])), '--labels', str(relpath(root, view['paths']['labels'])),
             '--eval-labels', str(relpath(root, view['paths']['eval_labels'])), '--init-weights', str(e113.weights_path(root))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if mode == 'smoke':
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=16, val=2, epochs=1), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def queue_entries(folds, after, workers=LOCAL_WORKERS, root=ROOT):
    """tools/gpu_queue.py job list: one full fit per fold, chained, continue_on_failure (independent folds)."""
    p = _protocol(root)
    entries, previous = [], after
    for k in folds:
        if str(k) not in p['folds']:
            raise SystemExit(f'unregistered fold {k}')
        e = dict(tag=tag_for(k), command=runner_argv(protocol_view(p, k), k, 'full', root=root, workers=workers))
        if previous:
            e['after'] = previous
        e['continue_on_failure'] = True
        entries.append(e); previous = e['tag']
    return entries


def validate_cli(fold):
    argv = runner_argv(view_for(fold), fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=e113.local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(fold, out, train=16, val=2, epochs=1, workers=2):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(view_for(fold), fold, 'smoke', workers=workers, output_root=out, smoke=dict(train=train, val=val, epochs=epochs))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=e113.local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for(fold, 'smoke')
    report = dict(version=VERSION, returncode=r.returncode, seconds=round(time.time() - t0), run=str(run),
                  files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json'); prov = s['provenance']
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], val=s['val'], seed=prov['seed'],
                      label_set=prov['label_set'], label_file_sha256=prov['label_file_sha256'], actual_pos_weight=prov['actual_pos_weight'],
                      initial_state_matches_registered=prov['initial_state_matches_registered'], cuda_initialized=prov.get('cuda_initialized'),
                      history=s['history'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ quarantine markers (DIAGNOSTIC_ONLY.json, tools/diagnostic_quarantine.py)
OOF_FILES = ('oof_best.csv', EXTRA_EVAL['file'])
MARKER_REASON = ('EXP116 student OOF: trained on teacher-fill labels whose fills for the other folds came from teacher fold models that trained '
                 'on this fold\'s validation studies, so the OOF is LEAKY by teacher ancestry (D-B-024, leak_rule). Diagnostics only; the '
                 'experiment is judged on the public LB only.')
SHARED_RUNNER_CAVEAT = ('summary.json "caveat" ("weak-label fold OOF for ensembling / teacher pseudo-labels") is written by the unchanged, '
                        'sha256-bound shared runner ' + SHARED_RUNNER + '; for EXP116 it does not apply and this marker overrides it')


def run_marker(tag, fold, mode, protocol_path, root=ROOT):
    """Marker body for one fold run directory (written by the runner entry before the model is built)."""
    protocol_path = Path(protocol_path)
    return dq.marker_body('EXP116', MARKER_REASON, ['D-B-024', 'D-C-005'], plan_item='H-B-46', tag=tag, fold=int(fold), mode=mode,
                          stage='written before training: the shared runner had just created the empty run directory and opened train.log; '
                                'the model was not built yet',
                          judge='public LB only', oof_files=list(OOF_FILES), overrides=dict(summary_caveat=SHARED_RUNNER_CAVEAT),
                          protocol=display_path(protocol_path, root), protocol_sha256=sha256(protocol_path) if protocol_path.is_file() else None,
                          tool_version=VERSION, written_by=RUNNER)


def annotate_run_marker(run):
    """After a completed run: bind every OOF CSV of the run (and its inventory) by sha256 in the marker."""
    run = Path(run)
    outputs = {n: sha256(run / n) for n in OOF_FILES if (run / n).is_file()}
    inv = run / 'output-inventory.json'
    return dq.update_marker(run, stage='complete: OOF CSVs written under this marker (bound below)', outputs=outputs,
                            output_inventory_sha256=sha256(inv) if inv.is_file() else None,
                            completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))


def marker_problems(run, k, tag):
    """verify_run part: the fold directory must carry its own diagnostic-only marker, and the OOF CSVs it binds must be the files present."""
    path = Path(run) / dq.MARKER
    if not path.is_file():
        return [f'fold {k}: {dq.MARKER} missing (EXP116 fold runs are written under a diagnostic-only marker; run via {RUNNER})']
    try:
        m = read(path)
    except (OSError, ValueError):
        return [f'fold {k}: {dq.MARKER} is not readable']
    problems = []
    if m.get('decision_eligible') is not False or m.get('schema') != dq.SCHEMA or m.get('tag') != tag or m.get('experiment') != 'EXP116':
        problems.append(f'fold {k}: {dq.MARKER} does not record decision_eligible false for {tag}')
    for name, digest in (m.get('outputs') or {}).items():
        if not (Path(run) / name).is_file() or sha256(Path(run) / name) != digest:
            problems.append(f'fold {k}: {name} differs from the sha256 bound in {dq.MARKER}')
    return problems


# ------------------------------------------------------------------ verification and assembly
def verify_run(run, p, k, manifest_sha):
    """Problems (empty list = complete and consistent) for one local EXP116 fold directory."""
    import numpy as np
    run = Path(run); view = protocol_view(p, k); fold = view['folds'][str(k)]; tag = tag_for(k)
    missing = [n for n in RUN_FILES if not (run / n).is_file()]
    if missing:
        return [f'fold {k}: missing ' + ', '.join(missing)]
    problems = marker_problems(run, k, tag)
    inv = read(run / 'output-inventory.json')
    for name, rec in inv.items():
        if not (run / name).is_file() or sha256(run / name) != rec['sha256']:
            problems.append(f'fold {k}: {name} differs from output-inventory.json')
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expect = dict(fold=k, label_set=p['label_set']['name'], label_semantic_sha256=view['label_set']['semantic_sha256'], label_file_sha256=view['label_set']['sha256'],
                  protocol_sha256=manifest_sha, eval_ids_ordered_sha256=fold['ordered_ids']['val_sha256'],
                  train_ids_ordered_sha256=fold['ordered_ids']['train_sha256'], training_mode='full', checkpoint_selection='fixed-final', seed=SEED,
                  arch=view['common_args']['arch'], input_policy='center3', norm='radimagenet', remote_relaxed=False,
                  initial_state_sha256=view['initial_state_sha256'], initial_state_matches_registered=True, environment_matches_baseline=True,
                  runner_sha256=view['source_bindings'][SHARED_RUNNER], guard_sha256=view['source_bindings'][TOOL])
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'fold {k}: provenance {key} = {prov.get(key)!r}, expected {value!r}')
    if (prov.get('init_weights') or {}).get('weights_sha256') != e113.RADIMAGENET_SHA256:
        problems.append(f'fold {k}: init weights are not the registered RadImageNet file')
    if not np.array_equal(np.asarray(prov.get('actual_pos_weight') or [], np.float32), np.asarray(fold['pos_weight'], np.float32)):
        problems.append(f'fold {k}: pos_weight differs from the fixed Dread vector')
    if s.get('optimizer_updates') != fold['steps'] or s.get('skipped_updates', 10 ** 9) > fold['max_fp16_skips']:
        problems.append(f"fold {k}: updates {s.get('optimizer_updates')} (expected {fold['steps']}), skipped {s.get('skipped_updates')}")
    if s.get('selected_epoch_1based') != view['common_args']['epochs'] or s.get('train') != fold['n_train'] or s.get('val') != fold['n_val']:
        problems.append(f'fold {k}: epoch/count mismatch')
    if run.name != tag or (s.get('args') or {}).get('tag') != tag:
        problems.append(f'fold {k}: run directory / tag is not {tag}')
    return problems


def assemble(runs_dir=None, root=ROOT):
    import pandas as pd
    root = Path(root)
    manifest_path = _manifest_path(root)
    p = read(manifest_path); manifest_sha = sha256(manifest_path)
    runs = Path(runs_dir).resolve() if runs_dir else root / 'artifacts' / 'experiments'
    out = root / OOF_DIR
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    frames24, frames36, sources, problems = {}, {}, {}, []
    for k in sorted(val_ids):
        run = runs / tag_for(k)
        bad = verify_run(run, p, k, manifest_sha)
        if bad:
            problems += bad; continue
        prov, s = read(run / 'provenance.json'), read(run / 'summary.json')
        frames24[k] = pd.read_csv(run / 'oof_best.csv', dtype={UID: str})
        frames36[k] = pd.read_csv(run / EXTRA_EVAL['file'], dtype={UID: str})
        # each fold file is named beside its digest (v1.1.2, C1): consumers bind the pooled OOF to these fold runs by path and sha256
        sources[k] = dict(run=display_path(run, root), oof_k24=display_path(run / 'oof_best.csv', root), oof_k24_sha256=sha256(run / 'oof_best.csv'),
                          oof_k36=display_path(run / EXTRA_EVAL['file'], root), oof_k36_sha256=sha256(run / EXTRA_EVAL['file']),
                          checkpoint_sha256=read(run / 'output-inventory.json')['model_best.pt']['sha256'], label_file_sha256=prov.get('label_file_sha256'),
                          seed=prov.get('seed'), initial_state_sha256=prov.get('initial_state_sha256'), environment=prov.get('environment'),
                          fp16_skipped_updates=s.get('skipped_updates'), total_seconds=s.get('total_seconds'))
    if problems:
        print(json.dumps(dict(status='INCOMPLETE', problems=problems), indent=1))
        raise SystemExit(2)
    result = k99.assemble_and_score(frames24, frames36, val_ids, root / DREAD)
    out.mkdir(parents=True)
    dq.write_marker(out, dq.marker_body('EXP116', MARKER_REASON, ['D-B-024', 'D-C-005'], plan_item='H-B-46', tag=OOF_DIR.rsplit('/', 1)[-1],
                                        stage='written before any CSV of the assembled OOF', judge='public LB only',
                                        oof_files=['oof_k24.csv', 'oof_k36.csv'], protocol=PROTOCOL + '/manifest.json',
                                        protocol_sha256=manifest_sha, tool_version=VERSION, written_by=TOOL))
    result['oof_k24'].to_csv(out / 'oof_k24.csv', index=False)
    result['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp116-oof-v1', version=VERSION, label_set=p['label_set']['name'], calibration=p['label_set'].get('calibration'),
                observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), protocol=PROTOCOL,
                protocol_sha256=manifest_sha, seed=SEED, exp113_protocol_sha256=BASE_MANIFEST_SHA256, runs_dir=display_path(runs, root),
                rows=int(len(result['oof_k24'])), sources={str(k): v for k, v in sources.items()},
                outputs={f: sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv')},
                decision_eligible=False, leak='LEAKY by teacher ancestry (leak_rule, D-B-024): diagnostics only; never a ruler, comparison, gate, '
                                              'ensemble-weight fit, ensemble member, stacking input or teacher / pseudo-label source; the '
                                              'experiment is judged on the public LB only',
                quarantine_marker=dict(path=display_path(out / dq.MARKER, root), sha256=sha256(out / dq.MARKER)),
                scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                leaky_diagnostic_metrics=result['metrics'], k36_missing_folds=result['k36_missing'])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    md = k99.result_markdown(body['label_set'], {**body, 'metrics': result['metrics']}).replace(
        f"# EXP099 B8 5-fold OOF ({body['label_set']})", '# EXP116 teacher-fill RadR50 5-fold OOF (LEAKY by construction, diagnostics only)', 1)
    (out / 'result.md').write_text(md, encoding='utf-8')
    print(json.dumps(dict(rows=body['rows'], decision_eligible=False,
                          leaky_metrics={k: {kk: (round(vv['macro12'], 6) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
                                         for k, v in result['metrics'].items()}), indent=1))
    return body


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('build-labels'); s.add_argument('--dest', default=None, help=f'new directory (default {LABEL_DIR}[_rint])')
    s.add_argument('--calibration', choices=CALIBRATIONS, default=DEFAULT_CALIBRATION, help='Platt target: soft Dread value (default) or rint(Dread)')
    s.add_argument('--reissue', action='store_true', help='rewrite the receipt of the EXISTING directory (labels must rebuild byte-identically; '
                                                          'the old receipt is kept as receipt_v<old>.json)')
    s = sub.add_parser('prepare'); s.add_argument('--dry-run', default=None, help='write the registration to this new directory instead (outside artifacts/)')
    s.add_argument('--calibration', choices=CALIBRATIONS, default=DEFAULT_CALIBRATION, help='which build-labels output to register (default soft)')
    s.add_argument('--supersede', action='store_true', help='replace an older-version registration before any EXP116 run (old kept as '
                                                            'manifest_v<old>.json)')
    s = sub.add_parser('argv'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s.add_argument('--mode', choices=['full', 'smoke'], default='full'); s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    s = sub.add_parser('validate'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s = sub.add_parser('smoke'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--out', required=True)
    s.add_argument('--train', type=int, default=16); s.add_argument('--val', type=int, default=2)
    s.add_argument('--epochs', type=int, default=1); s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('queue-json'); s.add_argument('--folds', default='0,1,2,3,4')
    s.add_argument('--after', required=True, help=f'tag the first entry waits for (the current GPU chain tail, e.g. {DEFAULT_AFTER}); "none" only for an idle GPU')
    s.add_argument('--workers', type=int, default=LOCAL_WORKERS)
    s.add_argument('--out', default=None, help=f'new JSON file for the entries (default {QUEUE_FILE}; "-" = print only)')
    s = sub.add_parser('assemble'); s.add_argument('--runs-dir', default=None)
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    print(f'exp116_teacher_fill v{VERSION}', file=sys.stderr)
    if a.command == 'build-labels':
        (reissue_receipt if a.reissue else build_labels)(dest=a.dest, calibration=a.calibration)
    elif a.command == 'prepare':
        if a.dry_run and within(a.dry_run, ROOT / 'artifacts'):
            raise SystemExit('--dry-run must be outside artifacts/')
        prepare(dest=a.dry_run, calibration=a.calibration, supersede=a.supersede)
    elif a.command == 'argv':
        print(json.dumps(runner_argv(view_for(a.fold), a.fold, a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli(a.fold))
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.fold, a.out, a.train, a.val, a.epochs, a.workers))
    elif a.command == 'queue-json':
        entries = queue_entries([int(x) for x in a.folds.split(',')], None if a.after.lower() == 'none' else a.after, a.workers)
        out = a.out or str(ROOT / QUEUE_FILE)
        if out != '-':
            dump_lf(out, entries)
            print('written ' + out, file=sys.stderr)
        print(json.dumps(entries, indent=1))
    else:
        assemble(a.runs_dir)


if __name__ == '__main__':
    main()
