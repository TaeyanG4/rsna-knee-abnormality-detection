"""CPU-only checks for EXP116 (H-B-46 targeted teacher fill; tools/exp116_teacher_fill.py, training/runners/exp116_teacher_fill.py).

Rules: cell masks invert the frozen relabel_v1 conversion; the teacher is a scale-free rank mean, Platt-calibrated and cross-fitted (a row's
calibrated value never depends on its own label); the default soft calibration keeps the calibrated mean equal to the soft-Dread mean of
the FITTED rows only, so the whole-target level of a filled target shifts and that shift is registered (level_shift; rint stays an option
under the _rint suffix); a student-fold table never depends on that fold's labels or teacher outputs; untouched targets and cells equal
Dread; the loss pos_weight is the fixed Dread vector; the protocol equals EXP113 except the documented fields and revokes the inherited
"OOF is for ensembling / teacher pseudo-labels" caveat; the v1.1.2 receipts / registration supersede v1.1.1 (which superseded v1.1.0)
with byte-identical labels, a reissue keeps labels_built_by, the registration lists tools/submission_screen.py as a refusing consumer;
the EXP107 runner runs unchanged with the EXP116 guard; every fold run directory gets DIAGNOSTIC_ONLY.json before training (runner entry
hook) and assembly requires it; queue entries are valid gpu_queue jobs; the student OOF assembles into a LEAKY, decision-ineligible,
marked directory whose result names every fold file (oof_k24 / oof_k36 beside their sha256). One CPU smoke (2 optimizer updates, --device cpu, subprocess). No GPU, no CUDA context.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp116_teacher_fill as g  # noqa: E402
from tools import exp113_radr50_local as e113  # noqa: E402
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools import exp109_relabel_v1 as r109  # noqa: E402
from tools import diagnostic_quarantine as dq  # noqa: E402

UID = g.UID
T = list(g.T)
REAL = ROOT / g.PROTOCOL / 'manifest.json'
BASE = ROOT / g.BASE_PROTOCOL / 'manifest.json'
LABELS = ROOT / g.label_dir_for('soft')
LABELS_RINT = ROOT / g.label_dir_for('rint')
WEIGHTS = ROOT / e113.WEIGHTS_LOCAL
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
QUEUE = ROOT / g.QUEUE_FILE
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP116 protocol not prepared')
built = pytest.mark.skipif(not (LABELS / g.RECEIPT_NAME).exists(), reason='EXP116 labels not built')
weights = pytest.mark.skipif(not WEIGHTS.exists(), reason='RadImageNet weights not present locally')
corpus = pytest.mark.skipif(not (CORPUS / 'manifest.json').exists(), reason='64-slot cache not present')
real_inputs = pytest.mark.skipif(not BASE.exists() or g.sha256(BASE) != g.BASE_MANIFEST_SHA256, reason='EXP113 registration absent or changed')


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def entry():
    import tools
    saved_module, saved_attr = sys.modules.get('tools.exp107_kfold_radr50'), getattr(tools, 'exp107_kfold_radr50', None)
    yield load_path('exp116_runner_entry', ROOT / g.RUNNER)
    sys.modules['tools.exp107_kfold_radr50'] = saved_module
    tools.exp107_kfold_radr50 = saved_attr


@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**g.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    monkeypatch.delenv(e107.WEIGHTS_ENV, raising=False)
    return monkeypatch


def parsed(entry, argv):
    return entry.load_runner().build_parser().parse_args(argv[2:])


def _mutate(argv, flag, value=None):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag] + ([str(value)] if value is not None else [])
    return argv


def without_thread_env():
    return {k: v for k, v in e113.local_env().items() if k not in g.THREAD_ENV}


# ------------------------------------------------------------------ names
def test_names_are_new_and_never_collide():
    assert g.PROTOCOL not in (e113.PROTOCOL, e107.PROTOCOL) and g.OOF_DIR.endswith('_LEAKY') and g.LABEL_DIR.startswith('storage/labels/derived_')
    for k in g.FOLDS:
        for mode in ('full', 'smoke'):
            new = g.tag_for(k, mode)
            for old in (e113.tag_for(k, mode), e113.tag_for(k, mode, 'relabel_v1'), e107.tag_for(k, mode)):
                assert new != old and old not in new and new not in old
    assert g.tag_for(2) == 'EXP116_TEACHER_FILL_F2' and g.tag_for(0, 'smoke') == 'EXP116_TEACHER_FILL_F0_SMOKE'
    assert g.fold_name(3) == 'teacher_fill_v1_f3.csv' and g.global_name() == 'teacher_fill_v1.csv' and g.VERSION == '1.1.2'
    assert (g.DEFAULT_CALIBRATION, g.CALIBRATIONS) == ('soft', ('soft', 'rint')) and g.label_name_for() == 'teacher_fill_v1' == g.LABEL_SET
    assert g.fold_name(3, 'rint') == 'teacher_fill_v1_rint_f3.csv' and g.global_name('rint') == 'teacher_fill_v1_rint.csv'
    assert g.label_name_for('rint') == 'teacher_fill_v1_rint' and g.label_dir_for('rint') == 'storage/labels/derived_teacher_fill_rint'
    assert g.label_dir_for('soft') == g.LABEL_DIR and g.label_dir(ROOT, None, 'rint') == LABELS_RINT
    for bad in ('hard', 'Soft', ''):
        with pytest.raises(ValueError):
            g.fold_name(0, bad)
    assert set(g.FILL_TARGETS) == {'PF OA', 'Synovitis', 'Effusion'} and "Baker's" not in g.FILL_TARGETS and 'Medial OA' not in g.FILL_TARGETS
    assert (g.DREAD_WEIGHT, g.TEACHER_WEIGHT) == (0.3, 0.7)


# ------------------------------------------------------------------ cell rules
def test_pf_oa_rule_is_relabel_value_0_05():
    x = np.array([0.05, 0.0500, 0.05000001, 0.02, 0.1, 0.55, 0.049])
    assert g.pf_oa_silent(x).tolist() == [True, True, True, False, False, False, False]


@pytest.mark.parametrize('syn, eff, expect, part', [
    (0.27, 0.45, True, 's1'), (0.02, 0.03, True, 's1'), (0.552, 0.92, True, 's1'), (0.588, 0.98, True, 's1'),
    (0.036, 0.05, True, 's2'), (0.12, 0.05, True, 's2'), (0.3, 0.05, True, 's2'), (0.03, 0.05, True, 's1'),
    (0.02, 0.05, False, None),  # ambiguous with a 'neg' scored 0.02 (documented exclusion)
    (0.85, 0.9, False, None), (0.6, 0.05, False, None), (0.25, 0.05, False, None), (0.08, 0.05, False, None), (0.7, 0.3, False, None)])
def test_synovitis_rule_cases(syn, eff, expect, part):
    mask, s1, s2 = g.synovitis_silent([syn], [eff])
    assert bool(mask[0]) is expect
    if part:
        assert bool({'s1': s1, 's2': s2}[part][0]) and not bool((s2 if part == 's1' else s1)[0])


def test_synovitis_rule_inverts_the_frozen_conversion():
    """Random labeller records through tools/exp109_relabel_v1.convert (4-decimal CSV rounding): every Synovitis 'na' is found, except the
    documented case (Effusion itself 'na' with raw p <= 0.03 -> Synovitis 0.02, indistinguishable from a 'neg' scored 0.02). Raw Effusion p
    is drawn on the labeller's two-decimal grid up to 0.97 (stored maximum, HC-04), and up to 0.91 when Effusion itself is 'na' (an 'na'
    Effusion with raw p > 0.91 would give Synovitis >= 0.55, which the rule leaves to 'pos'; no such stored value is on the 0.006 grid)."""
    rng = np.random.default_rng(7)
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    found = missed_documented = 0
    for _ in range(4000):
        f = {t: dict(s='neg', p=0.05) for t in T}
        eff_state = str(rng.choice(['pos', 'neg', 'unc', 'na']))
        f['Effusion'] = dict(s=eff_state, p=float(rng.choice(grid[grid <= (0.91 if eff_state == 'na' else 0.97)])))
        f['Synovitis'] = dict(s=str(rng.choice(['na', 'neg', 'unc', 'pos'], p=[.7, .1, .1, .1])), p=float(rng.choice(grid)))
        row = {t: round(v, 4) for t, v in r109.convert(f).items()}
        mask, _, _ = g.synovitis_silent([row['Synovitis']], [row['Effusion']])
        if f['Synovitis']['s'] == 'na':
            if mask[0]:
                found += 1
            else:
                assert f['Effusion']['s'] == 'na' and f['Effusion']['p'] <= 0.03 + 1e-9, (f['Effusion'], row)
                missed_documented += 1
        pf = g.pf_oa_silent([row['PF OA']])
        assert pf[0]  # 'neg' at p 0.05 -> 0.05: also caught (documented contamination)
    assert found > 2000 and missed_documented < found / 20


def test_effusion_disagreement():
    stack = np.array([[0.9, 0.9, 0.1, 0.5, 0.49, np.nan],
                      [0.8, 0.2, 0.1, 0.5, 0.2, 0.9],
                      [0.7, 0.9, 0.3, 0.6, 0.1, 0.9],
                      [0.6, 0.9, 0.4, 0.5, 0.3, 0.9]])
    assert g.effusion_disagree(stack).tolist() == [False, True, False, False, False, False]


# ------------------------------------------------------------------ teacher and calibration
def test_rank_mean_removes_the_weighted_bce_scale():
    """D-A-047: a weighted-BCE teacher outputs q = w p / (1 - p + w p); the percentile rank (and the rank mean) is unchanged by it."""
    rng = np.random.default_rng(0)
    p, r = rng.random(500), rng.random(500)
    for w in (1.5, 4.0, 9.0):
        q = w * p / (1 - p + w * p)
        assert np.array_equal(g.teacher_score(q, r), g.teacher_score(p, r))
    s = g.teacher_score(p, r)
    assert s.min() > 0 and s.max() < 1 and np.isfinite(g.logit(s)).all()


def test_platt_fit_recovers_parameters_and_calibration_in_the_large():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40000)
    y = (rng.random(40000) < g.sigmoid(1.7 * x - 1.2)).astype(float)
    a, b = g.platt_fit(x, y)
    assert abs(a - 1.7) < 0.06 and abs(b + 1.2) < 0.05
    assert abs(g.sigmoid(a * x + b).mean() - y.mean()) < 1e-6
    soft = np.clip(g.sigmoid(1.7 * x - 1.2) + rng.normal(scale=0.15, size=len(x)), 0.02, 0.98)  # soft targets in (0, 1)
    a2, b2 = g.platt_fit(x, soft)
    assert abs(g.sigmoid(a2 * x + b2).mean() - soft.mean()) < 1e-3  # soft-target BCE: calibrated in the large on the fitted rows
    with pytest.raises(ValueError):
        g.platt_fit(x[:10], np.zeros(10))


@pytest.mark.parametrize('scale', ['soft', 'rint'])
def test_cross_fitted_value_never_uses_its_own_label(scale):
    rng = np.random.default_rng(2)
    n = 600
    folds = np.repeat(np.arange(5), n // 5)
    c, r = rng.random(n), rng.random(n)
    y = (rng.random(n) < 0.3).astype(float) if scale == 'rint' else np.where(rng.random(n) < 0.3, 0.85, 0.1) + rng.uniform(-0.05, 0.05, n)
    cal, params = g.cross_fitted_teacher(c, r, y, folds, scale)
    y2 = y.copy(); y2[folds == 2] = 1 - y2[folds == 2]  # flip every fold-2 label
    cal2, _ = g.cross_fitted_teacher(c, r, y2, folds, scale)
    assert np.array_equal(cal[folds == 2], cal2[folds == 2])  # fold-2 values ignore fold-2 labels
    assert not np.allclose(cal[folds != 2], cal2[folds != 2])  # the other folds' calibrators use them
    for j, p in params.items():
        assert p['scale'] == scale and abs(p['mean_calibrated_fit'] - p['target_mean_fit']) < 1e-3 and p['n_fit'] + p['n_heldout'] == n
        assert abs(p['rint_prevalence_fit'] - np.rint(y[folds != int(j)].astype(np.float32)).mean()) < 1e-9


# ------------------------------------------------------------------ fill tables on a synthetic workspace
def _synthetic_inputs(seed=3, n_per_fold=24):
    rng = np.random.default_rng(seed)
    uids = [f'u{i:03d}' for i in range(5 * n_per_fold)]
    extra = ['x1', 'x2']
    fold_of = {u: i // n_per_fold for i, u in enumerate(uids)}
    truth = rng.random((len(uids), 12)) < 0.35
    D = pd.DataFrame(np.where(truth, 0.8, 0.1) + rng.uniform(-0.05, 0.05, truth.shape), index=uids, columns=T).round(3)
    D = pd.concat([D, pd.DataFrame(0.2, index=extra, columns=T)])
    R = pd.DataFrame(np.where(truth, 0.7, 0.03), index=uids, columns=T)
    R['PF OA'] = np.where(rng.random(len(uids)) < 0.3, 0.05, R['PF OA'])
    R['Synovitis'] = np.where(rng.random(len(uids)) < 0.6, np.clip(0.6 * R['Effusion'], 0.02, 0.98), 0.85)
    S = pd.DataFrame(np.where(rng.random(truth.shape) < 0.8, truth, ~truth) * 0.9 + 0.05, index=uids, columns=T)
    P = pd.DataFrame(np.where(rng.random(truth.shape) < 0.8, truth, ~truth) * 0.9 + 0.05, index=uids, columns=T)
    C = pd.DataFrame(np.clip(truth * 0.5 + rng.random(truth.shape) * 0.6, 0, 1), index=uids, columns=T)
    Q = pd.DataFrame(np.clip(truth * 0.4 + rng.random(truth.shape) * 0.7, 0, 1), index=uids, columns=T)
    val = {k: [u for u in uids if fold_of[u] == k] for k in range(5)}
    return dict(sources={'dread': D, 'relabel_v1': R, 'steven_v4': S, 'pilkwang_v2': P}, teachers={'coatnet5': C, 'radr50x5': Q},
                g1=uids, fold_of=fold_of, val_ids=val, train_ids={k: [u for u in uids if fold_of[u] != k] for k in range(5)},
                hashes={'x': 'y'}, teacher_meta={}, relabel_state_counts=None)


@pytest.mark.parametrize('cal', ['soft', 'rint'])
def test_fill_formula_untouched_cells_and_fold_rows(cal):
    inputs = _synthetic_inputs()
    tables, mask_frame, receipt = g.compute(inputs, cal)
    assert receipt['calibration'] == cal and receipt['label_name'] == g.label_name_for(cal) and set(receipt['scale_comparison']['per_scale']) == set(g.CALIBRATIONS)
    assert receipt['tables']['fold2']['file'] == g.fold_name(2, cal) and receipt['tables']['global']['file'] == g.global_name(cal)
    D = inputs['sources']['dread']
    masks, _ = g.cell_masks(inputs['sources'], inputs['g1'])
    gt = tables['global']
    _, params, calibrated = g.fill_table(inputs, masks, inputs['g1'], cal)
    for t, per in params.items():
        for j, p in per.items():
            assert p['scale'] == cal and abs(p['mean_calibrated_fit'] - p['target_mean_fit']) < 1e-3
    for t in T:
        if t not in g.FILL_TARGETS:
            assert gt[t].equals(D[t]) and all(tables[k][t].equals(D[t]) for k in range(5))
    for t in g.FILL_TARGETS:
        m = masks[t]
        cells = [u for u, s in zip(inputs['g1'], m) if s]
        other = [u for u, s in zip(inputs['g1'], m) if not s] + ['x1', 'x2']
        assert np.allclose(gt.loc[cells, t], 0.3 * D.loc[cells, t] + 0.7 * np.array([calibrated[t][u] for u in cells]))
        assert gt.loc[other, t].equals(D.loc[other, t])
        assert int(mask_frame[t].sum()) == int(m.sum()) == receipt['cells']['counts_g1'][t]['count']
    for k in range(5):
        assert tables[k].loc[inputs['val_ids'][k]].equals(D.loc[inputs['val_ids'][k]])
        assert tables[k].loc[['x1', 'x2']].equals(D.loc[['x1', 'x2']])
        s = receipt['summaries'][f'fold{k}']
        assert s['pos_weight_fixed_dread'] == e107.pos_weight({u: D.loc[u, T].to_numpy(np.float32) for u in inputs['train_ids'][k]},
                                                               inputs['train_ids'][k]).tolist()
    assert receipt['leak_rule'] == g.LEAK_RULE and receipt['schema'] == g.RECEIPT_SCHEMA


def test_soft_and_rint_fills_differ_only_in_scale_not_in_cells():
    inputs = _synthetic_inputs(seed=11)
    soft, mask_s, rs = g.compute(inputs, 'soft')
    rint, mask_r, rr = g.compute(inputs, 'rint')
    assert mask_s.equals(mask_r) and rs['cells'] == rr['cells']
    D = inputs['sources']['dread']
    for t in g.FILL_TARGETS:
        changed_s = soft['global'][t] != D[t]
        changed_r = rint['global'][t] != D[t]
        assert changed_s.equals(changed_r)  # same cells
        assert not np.allclose(soft['global'].loc[changed_s, t], rint['global'].loc[changed_r, t])  # different values
    for t in T:
        if t not in g.FILL_TARGETS:
            assert soft['global'][t].equals(rint['global'][t])


@pytest.mark.parametrize('cal', ['soft', 'rint'])
def test_student_fold_table_ignores_that_folds_labels_and_teacher_outputs(cal):
    """The conservative leak rule: student k's label file is unchanged when fold-k Dread labels or fold-k teacher outputs change."""
    a = _synthetic_inputs(seed=5)
    b = _synthetic_inputs(seed=5)
    k = 3
    rows = a['val_ids'][k]
    rng = np.random.default_rng(9)
    b['sources']['dread'].loc[rows, T] = rng.random((len(rows), 12)).round(3)
    for name in ('coatnet5', 'radr50x5'):
        b['teachers'][name].loc[rows, T] = rng.random((len(rows), 12))
    masks_a, _ = g.cell_masks(a['sources'], a['g1'])
    masks_b, _ = g.cell_masks(b['sources'], b['g1'])  # masks depend on Dread only through the Effusion vote of fold-k rows
    universe = [u for u in a['g1'] if a['fold_of'][u] != k]
    ta, _, _ = g.fill_table(a, masks_a, universe, cal)
    tb, _, _ = g.fill_table(b, masks_b, universe, cal)
    keep = [u for u in a['g1'] if a['fold_of'][u] != k]
    assert ta.loc[keep].equals(tb.loc[keep])
    tga, _, _ = g.fill_table(a, masks_a, a['g1'], cal)  # the global reference table does depend on them (why it is not a training file)
    tgb, _, _ = g.fill_table(b, masks_b, b['g1'], cal)
    assert not tga.loc[keep].equals(tgb.loc[keep])


def test_table_csv_round_trips_through_the_runner_loader(tmp_path):
    inputs = _synthetic_inputs()
    tables, _, _ = g.compute(inputs)
    path = tmp_path / 'x.csv'
    path.write_text(g.table_csv(tables[1]), encoding='utf-8', newline='')
    labels, _ = k99._load_labels(path)
    D = inputs['sources']['dread']
    assert set(labels) == set(D.index)
    for u in labels:
        assert np.abs(labels[u] - tables[1].loc[u, T].to_numpy(np.float64)).max() < 1e-6
        if (tables[1].loc[u, T] == D.loc[u, T]).all():
            assert np.array_equal(labels[u], D.loc[u, T].to_numpy(np.float32))


def test_load_teacher_verifies_true_oof(tmp_path):
    val = {0: ['a', 'b'], 1: ['c', 'd']}
    d = tmp_path / 'T'
    d.mkdir()
    pooled = pd.DataFrame({UID: ['a', 'b', 'c', 'd'], **{t: [0.1, 0.2, 0.3, 0.4] for t in T}})
    sources = {}
    for k, ids in val.items():
        part = pooled[pooled[UID].isin(ids)]
        part.to_csv(d / f'f{k}.csv', index=False)
        sources[str(k)] = dict(oof_k36=f'T/f{k}.csv', oof_k36_sha256=g.sha256(d / f'f{k}.csv'))
    pooled.to_csv(d / g.TEACHER_FILE, index=False)
    result = dict(schema='s', label_set='dread', k36_missing_folds=[], outputs={g.TEACHER_FILE: g.sha256(d / g.TEACHER_FILE)}, sources=sources)
    (d / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    frame, meta = g.load_teacher(tmp_path, 'T', 's', val)
    assert frame.loc['c', 'ACL'] == 0.3 and meta['per_fold']['1']['rows'] == 2
    with pytest.raises(ValueError, match='validation ids'):
        g.load_teacher(tmp_path, 'T', 's', {0: ['a', 'c'], 1: ['b', 'd']})
    pd.DataFrame({UID: ['c', 'd'], **{t: [0.3, 0.5] for t in T}}).to_csv(d / 'f1.csv', index=False)
    sources['1']['oof_k36_sha256'] = g.sha256(d / 'f1.csv')
    (d / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(ValueError, match='pooled rows differ'):
        g.load_teacher(tmp_path, 'T', 's', val)


# ------------------------------------------------------------------ real inputs
@real_inputs
@pytest.mark.parametrize('cal', ['soft', 'rint'])
def test_real_masks_teacher_and_tables(cal):
    inputs = g.load_inputs(ROOT)
    tables, mask_frame, receipt = g.compute(inputs, cal)
    cells = receipt['cells']
    assert {t: v['count'] for t, v in cells['counts_g1'].items()} == {'PF OA': 899, 'Synovitis': 3725, 'Effusion': 1737}
    ch = cells['checks']
    assert (ch['pf_oa_cells_all_relabel_rows'], ch['synovitis_cells_all_relabel_rows'], ch['synovitis_cells_dread_ge_0_5_all_relabel_rows']) == (900, 3731, 714)
    ref = g.HC04_REFERENCE
    assert ch['pf_oa_cells_all_relabel_rows'] == ref['PF OA']['na'] + ref['PF OA']['neg_at_0_05']
    assert abs(ch['synovitis_cells_all_relabel_rows'] - ref['Synovitis']['na']) <= 5
    D = inputs['sources']['dread']
    for k in g.FOLDS:
        assert tables[k].loc[inputs['val_ids'][k]].equals(D.loc[inputs['val_ids'][k]])
        s = receipt['summaries'][f'fold{k}']
        assert s['pos_weight_fixed_dread'] == inputs['manifest']['folds'][str(k)]['pos_weight']
    for key, params in receipt['calibration_params'].items():
        for t, per in params.items():
            for j, p in per.items():
                assert p['scale'] == cal and abs(p['mean_calibrated_fit'] - p['target_mean_fit']) < 1e-3  # calibrated in the large
                assert p['mean_raw_teacher_heldout'] > p['rint_prevalence_heldout'] + 0.05  # the D-A-047 distortion the ranks remove
                if key.startswith('fold'):
                    assert j != key[4:]
    gs = receipt['summaries']['global']['targets']
    shift = receipt['level_shift']
    assert set(shift['per_table']) == {'global', *[f'fold{k}' for k in g.FOLDS]} and 'FITTED rows only' in shift['rule']
    for key, per in shift['per_table'].items():
        s = receipt['summaries'][key]
        for t in g.FILL_TARGETS:  # whole-target shift = (cells / rows) x cell_mean_change, registered for every table
            assert per[t]['shift'] == s['targets'][t]['mean_filled_table'] - s['targets'][t]['mean_dread'] and per[t]['rows'] == s['universe']
            assert abs(per[t]['shift'] - per[t]['cells'] / per[t]['rows'] * per[t]['cell_mean_change']) < 1e-9
    sg = shift['per_table']['global']
    if cal == 'soft':  # calibrated in the large on the fitted rows only: the filled cells move the whole-target level (part of the intervention)
        assert [round(sg[t]['shift'], 6) for t in ('PF OA', 'Effusion', 'Synovitis')] == [0.019439, 0.009867, 0.00199]
        assert all(abs(gs[t]['mean_filled_table'] - gs[t]['mean_dread']) < 0.02 for t in g.FILL_TARGETS)
    else:  # the rint scale moves Synovitis from the soft mean towards the rint prevalence
        assert gs['Synovitis']['mean_filled_table'] < gs['Synovitis']['mean_dread'] - 0.1
    rule = receipt['fill']['scale_rule']
    assert rule.startswith(cal + ':') and g.level_shift_text(shift) in rule and 'part of the registered intervention' in rule
    assert "not the target's level" not in rule and 'not the label level' not in rule
    assert receipt['student_oof']['decision_eligible'] is False and receipt['student_oof']['judge'] == 'public LB only'
    comp = receipt['scale_comparison']['per_scale']
    assert set(comp) == {'soft', 'rint'} and comp['rint']['Synovitis']['rint_flips_1_to_0'] > comp['soft']['Synovitis']['rint_flips_1_to_0']


@built
@pytest.mark.parametrize('cal', ['soft', 'rint'])
def test_built_files_match_the_receipt_and_rebuild_identically(cal):
    import hashlib
    labels = LABELS if cal == 'soft' else LABELS_RINT
    if not (labels / g.RECEIPT_NAME).exists():
        pytest.skip(f'{cal} labels not built')
    receipt = g.read(labels / g.RECEIPT_NAME)
    assert receipt['calibration'] == cal and receipt['label_name'] == g.label_name_for(cal)
    assert set(receipt['outputs']) == {g.global_name(cal), *[g.fold_name(k, cal) for k in g.FOLDS], g.MASK_NAME}
    for name, rec in receipt['outputs'].items():
        assert g.sha256(labels / name) == rec['sha256'], name
    tables, mask_frame, _ = g.compute(g.load_inputs(ROOT), cal)
    assert hashlib.sha256(g.table_csv(tables[0]).encode('utf-8')).hexdigest() == receipt['outputs'][g.fold_name(0, cal)]['sha256']
    assert hashlib.sha256(mask_frame.to_csv(index=False, lineterminator='\n').encode('utf-8')).hexdigest() == receipt['outputs'][g.MASK_NAME]['sha256']
    assert receipt['version'] == g.VERSION and set(receipt['inputs']) >= set(g.SOURCES.values())
    assert b'\r\n' not in (labels / g.RECEIPT_NAME).read_bytes()  # LF
    # reissue chain: v1.1.2 supersedes v1.1.1, which superseded v1.1.0; every archive keeps a SUPERSEDED note, is bound by sha256 and records
    # the same label outputs; labels_built_by stays the version that built the label bytes (1.1.0)
    sup = receipt['supersedes']
    archive = ROOT / sup['path']
    assert archive == labels / f"receipt_v{sup['version']}.json" and g.sha256(archive) == sup['sha256'] and sup['version'] == '1.1.1'
    old = g.read(archive)
    note = old['SUPERSEDED']
    assert note['status'] == 'SUPERSEDED' and note['superseded_by'] == dict(path=g.display_path(labels / g.RECEIPT_NAME, ROOT), version=g.VERSION)
    assert note['original_sha256'] == sup['original_sha256'] and old['version'] == '1.1.1' and old['outputs'] == receipt['outputs']
    assert sup['reason'] == note['reason'] == g.SUPERSEDE_REASONS['1.1.2']
    first = g.read(ROOT / old['supersedes']['path'])
    assert old['supersedes']['version'] == '1.1.0' and g.sha256(ROOT / old['supersedes']['path']) == old['supersedes']['sha256']
    assert first['version'] == '1.1.0' and first['outputs'] == receipt['outputs'] and old['supersedes']['reason'] == g.SUPERSEDE_REASONS['1.1.1']
    assert receipt['labels_built_by'] == old['labels_built_by'] and receipt['labels_built_by']['version'] == '1.1.0'
    assert receipt['built_at'] == old['built_at'] == first['built_at']


@real_inputs
@built
def test_prepare_refuses_a_receipt_of_the_other_calibration(tmp_path):
    labels = tmp_path / 'labels'; labels.mkdir()
    receipt = g.read(LABELS / g.RECEIPT_NAME)
    receipt['calibration'] = 'rint'
    (labels / g.RECEIPT_NAME).write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError, match='calibration'):
        g.prepare(ROOT, dest=tmp_path / 'proto', labels_dest=labels, calibration='soft')
    with pytest.raises(ValueError, match='calibration'):
        g.prepare(ROOT, dest=tmp_path / 'proto', labels_dest=labels, calibration='hard')
    assert not (tmp_path / 'proto').exists()


@real_inputs
@built
def test_prepare_refuses_a_receipt_older_than_the_tool(tmp_path):
    labels = tmp_path / 'labels'; labels.mkdir()
    receipt = g.read(LABELS / g.RECEIPT_NAME)
    receipt['version'] = '1.1.0'
    (labels / g.RECEIPT_NAME).write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(SystemExit, match='reissue'):
        g.prepare(ROOT, dest=tmp_path / 'proto', labels_dest=labels, calibration='soft')
    assert not (tmp_path / 'proto').exists()


def _v110_copy(src, dest, version='1.1.0'):
    """A label directory as v<version> left it: the label files plus the archived receipt (without its SUPERSEDED note) as receipt.json."""
    dest.mkdir()
    old = g.read(src / f'receipt_v{version}.json')
    old.pop('SUPERSEDED')
    for name in old['outputs']:
        shutil.copy2(src / name, dest / name)
    (dest / g.RECEIPT_NAME).write_text(json.dumps(old, indent=2), encoding='utf-8')
    return old


@real_inputs
@built
def test_reissue_keeps_label_bytes_and_refuses_changed_labels(tmp_path):
    if not (LABELS / 'receipt_v1.1.0.json').exists():
        pytest.skip('v1.1.0 receipt archive absent')
    old = _v110_copy(LABELS, tmp_path / 'a')
    before = {n: g.sha256(tmp_path / 'a' / n) for n in old['outputs']}
    new = g.reissue_receipt(ROOT, dest=tmp_path / 'a', calibration='soft')
    assert new['version'] == g.VERSION and new['outputs'] == old['outputs'] and before == {n: v['sha256'] for n, v in old['outputs'].items()}
    assert {n: g.sha256(tmp_path / 'a' / n) for n in old['outputs']} == before
    archived = g.read(tmp_path / 'a' / 'receipt_v1.1.0.json')
    assert archived['SUPERSEDED']['status'] == 'SUPERSEDED' and {k: v for k, v in archived.items() if k != 'SUPERSEDED'} == old
    assert new['supersedes']['sha256'] == g.sha256(tmp_path / 'a' / 'receipt_v1.1.0.json')
    with pytest.raises(SystemExit, match='already'):
        g.reissue_receipt(ROOT, dest=tmp_path / 'a', calibration='soft')
    _v110_copy(LABELS, tmp_path / 'b')
    with open(tmp_path / 'b' / g.fold_name(2), 'ab') as f:
        f.write(b'\n')
    with pytest.raises(ValueError, match='byte-identical'):
        g.reissue_receipt(ROOT, dest=tmp_path / 'b', calibration='soft')
    assert not (tmp_path / 'b' / 'receipt_v1.1.0.json').exists() and g.read(tmp_path / 'b' / g.RECEIPT_NAME)['version'] == '1.1.0'
    # a second reissue (v1.1.1 -> current) keeps the record of the version that built the labels
    if (LABELS / 'receipt_v1.1.1.json').exists():
        mid = _v110_copy(LABELS, tmp_path / 'c', '1.1.1')
        again = g.reissue_receipt(ROOT, dest=tmp_path / 'c', calibration='soft')
        assert again['labels_built_by'] == mid['labels_built_by'] and again['labels_built_by']['version'] == '1.1.0'
        assert again['supersedes']['version'] == '1.1.1' and again['outputs'] == mid['outputs'] == old['outputs']


@real_inputs
@built
def test_supersede_is_refused_once_runs_bind_the_registration(tmp_path, monkeypatch):
    proto = tmp_path / 'proto'; proto.mkdir()
    (proto / 'manifest.json').write_text(json.dumps(dict(schema=g.SCHEMA, version='1.1.0')), encoding='utf-8')
    monkeypatch.setattr(g, 'started_runs', lambda root=ROOT: ['artifacts/experiments/EXP116_TEACHER_FILL_F0'])
    with pytest.raises(SystemExit, match='already bind'):
        g.prepare(ROOT, dest=proto, calibration='soft', supersede=True)
    monkeypatch.undo()
    (proto / 'manifest.json').write_text(json.dumps(dict(schema=g.SCHEMA, version=g.VERSION)), encoding='utf-8')
    with pytest.raises(SystemExit, match='already'):
        g.prepare(ROOT, dest=proto, calibration='soft', supersede=True)
    with pytest.raises(SystemExit, match='Refusing to overwrite'):
        g.prepare(ROOT, dest=proto, calibration='soft')
    assert sorted(x.name for x in proto.iterdir()) == ['manifest.json']


def test_no_stale_level_or_leak_claims_in_generated_text():
    stale = ("not the label level", "not the target's level", 'leak-free', 'leak free')
    for text in (g.__doc__, g.LEAK_RULE, g.QUARANTINE_CAVEAT, *g.CAVEATS, *g.LEVEL_TEXT.values(), g.LEVEL_RULE):
        assert not any(x in text for x in stale), text[:120]
    assert 'LEAKY' in g.LEAK_RULE and 'public LB only' in g.LEAK_RULE and 'FITTED rows only' in g.__doc__ and '+0.019439' in g.__doc__
    assert g.inherited_caveats(['a', *g.REVOKED_INHERITED_CAVEATS, 'b']) == ['a', 'b']
    with pytest.raises(ValueError, match='revoke'):
        g.inherited_caveats(['a'])


# ------------------------------------------------------------------ the registration
@prepared
def test_manifest_equals_exp113_except_documented_fields():
    p, p113 = g.read(REAL), g.read(BASE)
    differing = {k for k in set(p) | set(p113) if p.get(k) != p113.get(k)}
    assert differing <= set(g.CHANGED_FIELDS), differing - set(g.CHANGED_FIELDS)
    for key in ('common_args', 'seed', 'recipe', 'resource_args', 'smoke_allowed', 'training_code_sha256', 'environment', 'corpus_manifest_sha256',
                'excluded_uids', 'fold_protocol', 'total_steps', 'extra_eval', 'init_weights', 'input', 'initial_state_sha256', 'initial_state'):
        assert p[key] == p113[key], key
    for k in g.FOLDS:
        new, old = p['folds'][str(k)], p113['folds'][str(k)]
        assert {x for x in set(new) | set(old) if new.get(x) != old.get(x)} <= set(g.FOLD_CHANGED_FIELDS)
        assert new['pos_weight'] == old['pos_weight'] and new['tag'] == g.tag_for(k)
        cal = p['label_set']['calibration']
        assert new['label_file']['path'] == f'{g.label_dir_for(cal)}/{g.fold_name(k, cal)}' and new['label_file']['sha256'] == g.sha256(ROOT / new['label_file']['path'])
        assert new['label_file']['calibration'] == cal and new['label_file']['name'] == p['label_set']['name']
    assert p['label_set']['calibration'] == 'soft' == p['teacher_fill']['calibration'] and p['label_set']['name'] == 'teacher_fill_v1'  # the registered default
    assert set(p['teacher_fill']['scale_comparison']['per_scale']) == {'soft', 'rint'}
    assert p['schema'] == g.SCHEMA and p['plan_item'] == 'H-B-46' and p['derived_from']['sha256'] == g.BASE_MANIFEST_SHA256
    assert p['evaluation']['decision_eligible'] is False and p['evaluation']['oof_dir'] == g.OOF_DIR and p['leak_rule'] == g.LEAK_RULE
    assert p['local']['runner'] == g.RUNNER and p['local']['shared_runner'] == e113.SHARED_RUNNER and p['paths']['labels'] is None
    for path, expected in p['source_bindings'].items():
        assert g.sha256(ROOT / path) == expected, path
    assert {g.RUNNER, g.TOOL, e113.TOOL, e113.RUNNER, e107.RUNNER, g.BASE_PROTOCOL + '/manifest.json'} <= set(p['source_bindings'])
    # v1.1.1: corrected level text, revoked inherited caveat, quarantine; v1.1.2: consumer status, superseded v1.1.1 registration, LF
    assert p['version'] == g.VERSION and g.QUARANTINE in p['source_bindings'] and b'\r\n' not in REAL.read_bytes()
    assert p['source_bindings'][g.label_dir_for('soft') + '/' + g.RECEIPT_NAME] == p['label_set']['receipt_sha256']
    assert not set(g.REVOKED_INHERITED_CAVEATS) & set(p['caveats']) and set(g.REVOKED_INHERITED_CAVEATS) <= set(p113['caveats'])
    assert not any(re.match(r'OOF is for', c) for c in p['caveats']) and p['caveats'][-1] == g.QUARANTINE_CAVEAT
    assert p['caveats'] == g.inherited_caveats(p113['caveats']) + g.caveats_for(g.read(ROOT / p['label_set']['receipt'])['level_shift'], 'soft')
    level = [c for c in p['caveats'] if c.startswith('calibration scale')]
    assert len(level) == 1 and 'PF OA +0.019439, Effusion +0.009867, Synovitis +0.001990' in level[0] and 'not the label level' not in level[0]
    assert "not the target's level" not in json.dumps(p['teacher_fill']['fill']) and p['teacher_fill']['level_shift']['per_table']['global']
    q = p['quarantine']
    assert q['decision_eligible'] is False and q['marker'] == dq.MARKER and q['revoked_inherited_caveats'] == list(g.REVOKED_INHERITED_CAVEATS)
    assert q['consumers_refusing'] == ['tools/g1_compare.py', 'tools/exp103_distill_labels.py', 'tools/submission_screen.py'] and q['consumers_pending'] == []
    assert 'no marker check' not in json.dumps(q) and 'before loading any member' in q['consumer_rules']['tools/submission_screen.py']  # (the supersede reason quotes it)
    assert p['evaluation']['judge'].startswith('public LB only')
    sup = p['supersedes']
    archive = ROOT / sup['path']
    assert archive == REAL.parent / 'manifest_v1.1.1.json' and g.sha256(archive) == sup['sha256'] and sup['version'] == '1.1.1'
    assert sup['reason'] == g.SUPERSEDE_REASONS['1.1.2']
    old = g.read(archive)
    assert old['SUPERSEDED']['superseded_by'] == dict(path=g.PROTOCOL + '/manifest.json', version=g.VERSION) and old['version'] == '1.1.1'
    assert old['SUPERSEDED']['original_sha256'] == sup['original_sha256'] and old['SUPERSEDED']['runs_started'] is False
    first = ROOT / old['supersedes']['path']
    assert first == REAL.parent / 'manifest_v1.1.0.json' and g.sha256(first) == old['supersedes']['sha256'] and old['supersedes']['version'] == '1.1.0'
    assert g.read(first)['version'] == '1.1.0' and old['supersedes']['reason'] == g.SUPERSEDE_REASONS['1.1.1']
    assert {k: old['folds'][k]['label_file'] for k in old['folds']} == {k: p['folds'][k]['label_file'] for k in p['folds']}  # same label files


@prepared
def test_guard_pos_weight_is_the_fixed_dread_vector():
    p = g.read(REAL)
    for k in g.FOLDS:
        train = g.read(ROOT / p['folds'][str(k)]['ordered_ids']['path'])['train']
        filled, _ = k99._load_labels(ROOT / p['folds'][str(k)]['label_file']['path'])
        got = g.pos_weight(filled, train)
        assert np.array_equal(got, np.asarray(p['folds'][str(k)]['pos_weight'], np.float32))
        assert not np.array_equal(e107.pos_weight(filled, train), got)  # what the filled labels would have given (not used)
    with pytest.raises(ValueError, match='misses'):
        g.pos_weight({}, train[:3])


def test_entry_runs_the_unchanged_exp107_runner_with_the_exp116_guard(entry):
    import tools
    runner = entry.load_runner()
    assert Path(runner.__file__).resolve() == (ROOT / e107.RUNNER).resolve()
    assert sys.modules['tools.exp107_kfold_radr50'] is g and tools.exp107_kfold_radr50 is g
    names = set(re.findall(r'guard\.([A-Za-z_]+)', (ROOT / e107.RUNNER).read_text(encoding='utf-8')))
    assert len(names) > 10 and all(hasattr(g, n) for n in names), sorted(n for n in names if not hasattr(g, n))
    for name in ('worker_init', 'read_rows', 'install_gold_blocker', 'uid_hash', 'sha256', 'read', 'canonical_order', 'validate_fold_membership',
                 'fold_contract', 'state_hashes', 'mismatched_keys', 'CODE_NAMES', 'ENV_KEYS'):
        assert getattr(g, name) is getattr(e107, name), name
    assert g.pos_weight is not e107.pos_weight and g.LABEL_SET == 'teacher_fill_v1'
    shared = load_path('exp107_shared_parser_116', ROOT / e107.RUNNER).build_parser()
    assert [a.dest for a in runner.build_parser()._actions] == [a.dest for a in shared._actions]


FAKE_RUNNER = """
import argparse
from pathlib import Path
EVENTS = []
LOG_NAME = 'train.log'


def build_parser():
    ap = argparse.ArgumentParser()
    for name in ('--protocol', '--tag', '--output-root', '--mode'):
        ap.add_argument(name)
    ap.add_argument('--fold', type=int)
    ap.add_argument('--validate-only', action='store_true')
    return ap


def main(argv):
    a = build_parser().parse_args(argv)
    if a.validate_only:
        EVENTS.append('validated')
        return
    out = Path(a.output_root) / a.tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Refusing to overwrite existing run')
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / LOG_NAME, 'a', encoding='utf-8')
    EVENTS.append(('log opened', sorted(p.name for p in out.iterdir())))
    log.write('training\\n'); log.close()
    for name in ('oof_best.csv', 'oof_k36.csv'):
        (out / name).write_text('StudyInstanceUID\\nu1\\n', encoding='utf-8')
    (out / 'output-inventory.json').write_text('{}', encoding='utf-8')
"""


def _fake_runner(log_name='train.log'):
    mod = types.ModuleType('fake_exp107_runner')
    exec(FAKE_RUNNER, mod.__dict__)
    mod.LOG_NAME = log_name
    return mod


def test_entry_writes_the_diagnostic_marker_before_training(guard_env, entry, tmp_path):
    fake = _fake_runner()
    guard_env.setattr(entry, 'load_runner', lambda: fake)
    tag = g.tag_for(1, 'smoke')
    argv = ['--protocol', str(tmp_path / 'manifest.json'), '--fold', '1', '--mode', 'smoke', '--tag', tag, '--output-root', str(tmp_path)]
    entry.main(argv)
    run = tmp_path / tag
    assert fake.EVENTS == [('log opened', [dq.MARKER, 'train.log'])]  # the marker existed when the runner opened its log (before training)
    m = g.read(run / dq.MARKER)
    assert m['decision_eligible'] is False and m['schema'] == dq.SCHEMA and (m['experiment'], m['tag'], m['fold']) == ('EXP116', tag, 1)
    assert 'D-B-024' in m['discoveries'] and 'LEAKY' in m['reason'] and m['judge'] == 'public LB only' and m['written_by'] == g.RUNNER
    assert 'ensembling / teacher pseudo-labels' in m['overrides']['summary_caveat'] and m['stage'].startswith('complete')
    assert m['outputs'] == {n: g.sha256(run / n) for n in ('oof_best.csv', 'oof_k36.csv')} and b'\r\n' not in (run / dq.MARKER).read_bytes()
    assert g.marker_problems(run, 1, tag) == [] and dq.quarantine_reason(run / 'oof_best.csv')
    with pytest.raises(dq.DiagnosticOnlyInput, match='DIAGNOSTIC_ONLY'):
        dq.require_decision_eligible(run / 'oof_k36.csv', 'test')
    (run / 'oof_k36.csv').write_text('changed', encoding='utf-8')
    assert any('oof_k36.csv differs' in x for x in g.marker_problems(run, 1, tag))
    with pytest.raises(SystemExit, match='Refusing to overwrite'):  # a second attempt never reuses the marked directory
        entry.main(argv)


def test_entry_fails_closed_without_a_marker_and_skips_it_for_validate_only(guard_env, entry, tmp_path):
    fake = _fake_runner('run.log')  # a runner that never opens <run>/train.log: the hook cannot fire
    guard_env.setattr(entry, 'load_runner', lambda: fake)
    tag = g.tag_for(2, 'smoke')
    argv = ['--protocol', str(tmp_path / 'm.json'), '--fold', '2', '--mode', 'smoke', '--tag', tag, '--output-root', str(tmp_path)]
    with pytest.raises(SystemExit, match='without DIAGNOSTIC_ONLY.json'):
        entry.main(argv)
    assert not (tmp_path / tag / dq.MARKER).exists()
    ok = _fake_runner()
    guard_env.setattr(entry, 'load_runner', lambda: ok)
    entry.main(argv[:-4] + ['--tag', g.tag_for(3, 'smoke'), '--output-root', str(tmp_path / 'v'), '--validate-only'])
    assert ok.EVENTS == ['validated'] and not (tmp_path / 'v').exists()


def test_shared_runner_is_unchanged_and_opens_train_log_right_after_creating_the_empty_run_dir():
    bound = g.read(BASE)['source_bindings']
    for path in (e107.RUNNER, e107.TOOL, e113.TOOL, e113.RUNNER):
        assert g.sha256(ROOT / path) == bound[path], path  # EXP113 bindings intact (nothing it binds was edited)
    src = (ROOT / e107.RUNNER).read_text(encoding='utf-8')
    order = [src.index('if out.exists() and any(out.iterdir()):'), src.index('out.mkdir(parents=True, exist_ok=True)'),
             src.index('open(out / "train.log", "a"'), src.index('model = built.to(dev)'), src.index('write_predictions(out / "oof_best.csv"')]
    assert order == sorted(order) and src.count('"train.log"') == 1
    assert 'open(' not in src[order[1]:order[2]]  # nothing else is opened between creating the directory and the marker


def test_entry_supplies_only_missing_thread_env():
    entry = load_path('exp116_entry_plain', ROOT / g.RUNNER)
    env = {'OMP_NUM_THREADS': '4'}
    assert entry.supply_registered_env(env) == ['MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'HF_HUB_OFFLINE']
    assert env['OMP_NUM_THREADS'] == '4' and env['MKL_NUM_THREADS'] == '1'


@prepared
@weights
@pytest.mark.parametrize('fold', [0, 3])
def test_runner_argv_round_trip_validates(guard_env, entry, fold):
    view = g.view_for(fold)
    argv = g.runner_argv(view, fold)
    assert argv[1] == str(ROOT / g.RUNNER) and '--label-set' not in argv and '--output-root' not in argv
    assert argv[argv.index('--labels') + 1] == argv[argv.index('--eval-labels') + 1] == str(ROOT / view['label_set']['path'])
    assert view['label_set']['path'].endswith(g.fold_name(fold, view['label_set']['calibration']))
    args = parsed(entry, argv)
    for k, v in view['common_args'].items():
        assert getattr(args, k) == v and type(getattr(args, k)) is type(v), k
    got, spec, ordered, out_root = g.validate_arguments(args, ROOT)
    assert spec['tag'] == g.tag_for(fold) and got['label_set']['sha256'] == spec['label_file']['sha256'] and got['label_set']['fold'] == fold
    assert g.LABEL_SET == got['label_set']['name'] == spec['label_file']['name']  # what the shared runner records as guard.LABEL_SET
    assert out_root == (ROOT / 'artifacts' / 'experiments').resolve() and len(ordered['train']) == spec['n_train']


@prepared
@weights
@pytest.mark.parametrize('flag, value, message', [
    ('--seed', '43', 'seed changed'), ('--tag', 'EXP113_RADR50_LOCAL_F0', 'tag'), ('--tag', 'EXP116_TEACHER_FILL_F1', 'tag'),
    ('--protocol', str(BASE), 'wrong protocol'), ('--bb-lr', '0.0003', 'frozen argument changed: bb_lr'),
    ('--device', 'cpu', 'frozen argument changed: device'), ('--labels', str(ROOT / g.DREAD), 'input path differs: labels'),
    ('--labels', str(ROOT / g.LABEL_DIR / g.fold_name(1)), 'input path differs: labels'),
    ('--labels', str(ROOT / g.label_dir_for('soft') / g.global_name('soft')), 'input path differs: labels'),
    ('--labels', str(ROOT / g.label_dir_for('rint') / g.fold_name(0, 'rint')), 'input path differs: labels'),
    ('--eval-labels', str(ROOT / g.DREAD), 'input path differs: eval_labels'), ('--workers', '99', 'workers'),
    ('--output-root', str(ROOT / 'artifacts' / 'other'), 'artifacts/experiments only')])
def test_validate_arguments_rejects(guard_env, entry, flag, value, message):
    argv = _mutate(g.runner_argv(g.view_for(0), 0), flag, value)
    with pytest.raises(ValueError, match=message):
        g.validate_arguments(parsed(entry, argv), ROOT)


@prepared
@weights
def test_relax_env_smoke_rules_and_exp113_guard_refusal(guard_env, entry, tmp_path):
    args = parsed(entry, g.runner_argv(g.view_for(0), 0))
    guard_env.setenv('KNEE_REMOTE_RELAX_ENV', '1')
    with pytest.raises(ValueError, match='KNEE_REMOTE_RELAX_ENV'):
        g.validate_arguments(args, ROOT)
    guard_env.delenv('KNEE_REMOTE_RELAX_ENV')
    with pytest.raises(ValueError, match='wrong protocol'):
        e113.validate_arguments(args, ROOT)
    smoke = parsed(entry, g.runner_argv(g.view_for(0), 0, 'smoke', workers=0, output_root=tmp_path))
    assert (smoke.device, smoke.amp, smoke.tag) == ('cpu', 'none', 'EXP116_TEACHER_FILL_F0_SMOKE')
    assert g.validate_arguments(smoke, ROOT)[3] == tmp_path.resolve()
    inside = parsed(entry, g.runner_argv(g.view_for(0), 0, 'smoke', output_root=ROOT / 'artifacts' / 'experiments'))
    with pytest.raises(ValueError, match='outside artifacts'):
        g.validate_arguments(inside, ROOT)


# ------------------------------------------------------------------ queue entries
@prepared
def test_queue_entries_are_well_formed_gpu_queue_jobs():
    entries = g.queue_entries([0, 1, 2, 3, 4], 'PREV')
    assert [e['tag'] for e in entries] == [f'EXP116_TEACHER_FILL_F{k}' for k in range(5)]
    assert [e['after'] for e in entries] == ['PREV'] + [f'EXP116_TEACHER_FILL_F{k}' for k in range(4)]
    for k, e in enumerate(entries):
        assert set(e) == {'tag', 'command', 'after', 'continue_on_failure'} and e['continue_on_failure'] is True
        cmd = e['command']
        assert all(isinstance(c, str) for c in cmd) and Path(cmd[1]) == ROOT / g.RUNNER and cmd[cmd.index('--fold') + 1] == str(k)
        assert cmd[cmd.index('--device') + 1] == 'cuda' and cmd[cmd.index('--workers') + 1] == '12' and cmd[cmd.index('--mode') + 1] == 'full'
        assert cmd[cmd.index('--labels') + 1].endswith(g.fold_name(k)) and not {'--output-root', '--validate-only', '--smoke-train'} & set(cmd)
    with pytest.raises(SystemExit):
        g.queue_entries([7], 'PREV')


@prepared
def test_saved_queue_entries_match_the_protocol():
    if not QUEUE.exists():
        pytest.skip('queue-entries.json not written')
    saved = json.loads(QUEUE.read_text(encoding='utf-8'))
    assert saved == g.queue_entries([0, 1, 2, 3, 4], saved[0].get('after')) and saved[0]['after'] == g.DEFAULT_AFTER


# ------------------------------------------------------------------ verification and assembly (synthetic workspace)
@pytest.fixture()
def mini(tmp_path):
    uids = [f'u{i:02d}' for i in range(15)]
    y = np.array([i % 2 for i in range(15)])
    (tmp_path / Path(g.DREAD).parent).mkdir(parents=True)
    pd.DataFrame({UID: uids, **{t: np.where(y == 1, 0.95, 0.05) for t in T}}).to_csv(tmp_path / g.DREAD, index=False)
    folds = {k: dict(train=[u for u in uids if u not in uids[3 * k:3 * k + 3]], val=uids[3 * k:3 * k + 3]) for k in range(5)}
    proto = tmp_path / g.PROTOCOL; spec = {}
    for k, v in folds.items():
        (proto / f'fold{k}').mkdir(parents=True)
        (proto / f'fold{k}' / 'ordered-ids.json').write_text(json.dumps(v), encoding='utf-8')
        spec[str(k)] = dict(ordered_ids=dict(path=f'{g.PROTOCOL}/fold{k}/ordered-ids.json', train_sha256=g.uid_hash(v['train']), val_sha256=g.uid_hash(v['val'])),
                            n_train=12, n_val=3, steps=20, max_fp16_skips=0, tag=g.tag_for(k), pos_weight=[1.0] * 12,
                            label_file=dict(path=f'storage/labels/x/f{k}.csv', sha256=f'lab{k}', semantic_sha256=f'sem{k}'))
    manifest = dict(schema=g.SCHEMA, seed=42, common_args=e107.EXPECTED_RECIPE, folds=spec, initial_state_sha256='init',
                    label_set=dict(name='teacher_fill_v1', calibration='soft'),
                    source_bindings={g.SHARED_RUNNER: 'runner-sha', g.TOOL: 'guard-sha'}, tags=dict(prefix=g.TAG_PREFIX), paths=dict(corpus='c'))
    (proto / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path, folds


def _fake_run(root, runs, k, folds, marker=True, **overrides):
    p = g.read(root / g.PROTOCOL / 'manifest.json'); view = g.protocol_view(p, k)
    tag = g.tag_for(k); run = runs / tag; run.mkdir(parents=True)
    if marker:
        dq.write_marker(run, g.run_marker(tag, k, 'full', root / g.PROTOCOL / 'manifest.json', root))
    val = folds[k]['val']
    frame = pd.DataFrame({UID: val, **{t: [0.9 if int(u[1:]) % 2 else 0.1 for u in val] for t in T}})
    frame.to_csv(run / 'oof_best.csv', index=False); frame.to_csv(run / 'oof_k36.csv', index=False)
    prov = dict(fold=k, seed=42, label_set=p['label_set']['name'], label_semantic_sha256=view['label_set']['semantic_sha256'], label_file_sha256=view['label_set']['sha256'],
                protocol_sha256=g.sha256(root / g.PROTOCOL / 'manifest.json'), eval_ids_ordered_sha256=g.uid_hash(val),
                train_ids_ordered_sha256=g.uid_hash(folds[k]['train']), training_mode='full', checkpoint_selection='fixed-final', arch='resnet50',
                input_policy='center3', norm='radimagenet', remote_relaxed=False, initial_state_sha256='init', initial_state_matches_registered=True,
                environment_matches_baseline=True, runner_sha256='runner-sha', guard_sha256='guard-sha',
                init_weights=dict(weights_sha256=e113.RADIMAGENET_SHA256), actual_pos_weight=[1.0] * 12)
    prov.update(overrides)
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    (run / 'summary.json').write_text(json.dumps(dict(provenance=prov, optimizer_updates=20, skipped_updates=0, selected_epoch_1based=20, train=12, val=3,
                                                      args=dict(tag=tag), total_seconds=1)), encoding='utf-8')
    (run / 'history.json').write_text('[]', encoding='utf-8'); (run / 'model_best.pt').write_bytes(b'w')
    names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv']
    (run / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=g.sha256(run / n)) for n in names}), encoding='utf-8')
    if marker:
        g.annotate_run_marker(run)


def test_assemble_writes_a_leaky_decision_ineligible_oof(mini):
    root, folds = mini
    runs = root / 'artifacts' / 'experiments'
    for k in folds:
        _fake_run(root, runs, k, folds)
    body = g.assemble(root=root)
    assert body['decision_eligible'] is False and 'metrics' not in body and body['leaky_diagnostic_metrics']['k24']['pooled']['macro12'] == 1.0
    assert (body['label_set'], body['calibration']) == ('teacher_fill_v1', 'soft')
    assert (root / g.OOF_DIR / 'result.md').read_text(encoding='utf-8').startswith('# EXP116 teacher-fill RadR50 5-fold OOF (LEAKY')
    out = root / g.OOF_DIR
    m = g.read(out / dq.MARKER)
    assert m['decision_eligible'] is False and m['stage'] == 'written before any CSV of the assembled OOF' and 'D-B-024' in m['discoveries']
    assert body['quarantine_marker']['sha256'] == g.sha256(out / dq.MARKER) and 'LEAKY' in body['leak'] and 'public LB only' in body['leak']
    for name in ('oof_k24.csv', 'oof_k36.csv'):
        assert dq.quarantine_reason(out / name, root) and dq.quarantine_reason(root / 'artifacts' / 'experiments' / g.tag_for(0) / 'oof_best.csv', root)
    # v1.1.2 (C1): every fold file is named beside its digest, as the screen's fold-run binding reads it
    for k, src in body['sources'].items():
        run = root / 'artifacts' / 'experiments' / g.tag_for(int(k))
        assert src['oof_k36'] == g.display_path(run / 'oof_k36.csv', root) == f'artifacts/experiments/{g.tag_for(int(k))}/oof_k36.csv'
        assert src['oof_k36_sha256'] == g.sha256(run / 'oof_k36.csv') and src['oof_k24_sha256'] == g.sha256(run / 'oof_best.csv')
        assert src['oof_k24'] == g.display_path(run / 'oof_best.csv', root)
    assert g.read(out / 'result.json')['sources'] == body['sources']
    with pytest.raises(SystemExit):
        g.assemble(root=root)


def test_assemble_refuses_fold_runs_without_the_marker(mini):
    root, folds = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, marker=k != 3)
    problems = g.verify_run(runs / g.tag_for(3), g.read(root / g.PROTOCOL / 'manifest.json'), 3, g.sha256(root / g.PROTOCOL / 'manifest.json'))
    assert problems == [f'fold 3: {dq.MARKER} missing (EXP116 fold runs are written under a diagnostic-only marker; run via {g.RUNNER})']
    with pytest.raises(SystemExit):
        g.assemble(runs_dir=runs, root=root)
    assert not (root / g.OOF_DIR).exists()


@pytest.mark.parametrize('damage, message', [(dict(label_file_sha256='lab0'), 'label_file_sha256'), (dict(actual_pos_weight=[2.0] * 12), 'fixed Dread'),
                                             (dict(guard_sha256='exp113'), 'guard_sha256'), (dict(label_set='dread'), 'label_set')])
def test_verify_run_refuses_wrong_fold_label_or_weighting(mini, damage, message):
    root, folds = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, **(damage if k == 2 else {}))
    problems = g.verify_run(runs / g.tag_for(2), g.read(root / g.PROTOCOL / 'manifest.json'), 2, g.sha256(root / g.PROTOCOL / 'manifest.json'))
    assert problems and any(message in x for x in problems), problems
    with pytest.raises(SystemExit):
        g.assemble(runs_dir=runs, root=root)


# ------------------------------------------------------------------ the shared runner on CPU (subprocess; no CUDA context)
@prepared
@weights
@corpus
def test_validate_only_on_cpu():
    fold = 2
    view = g.view_for(fold)
    argv = g.runner_argv(view, fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=without_thread_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    out = json.loads(r.stdout)
    spec = view['folds'][str(fold)]
    assert out['status'] == 'VALIDATED_NO_TRAINING' and out['tag'] == g.tag_for(fold) and out['output_exists'] is False
    assert out['initial_state_matches_registered'] is True and out['label_semantic_sha256'] == spec['label_file']['semantic_sha256']
    assert out['pos_weight'] == [round(float(x), 6) for x in spec['pos_weight']] and out['steps'] == spec['steps']


@prepared
@weights
@corpus
def test_cpu_smoke_two_optimizer_updates(tmp_path):
    fold = 1
    view = g.view_for(fold)
    argv = g.runner_argv(view, fold, 'smoke', workers=2, output_root=tmp_path, smoke=dict(train=16, val=2, epochs=1))
    r = subprocess.run(argv, cwd=ROOT, env=without_thread_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    run = tmp_path / g.tag_for(fold, 'smoke')
    s = g.read(run / 'summary.json'); prov = s['provenance']
    assert s['optimizer_updates'] == 2 and s['skipped_updates'] == 0 and (s['train'], s['val'], s['mode']) == (16, 2, 'smoke')
    assert np.isfinite(s['history'][0]['loss']) and prov['cuda_initialized'] is False and prov['initial_state_matches_registered'] is True
    assert prov['label_set'] == 'teacher_fill_v1' and prov['label_file_sha256'] == view['label_set']['sha256']
    assert prov['actual_pos_weight'] == view['folds'][str(fold)]['pos_weight'] == g.read(BASE)['folds'][str(fold)]['pos_weight']
    assert prov['guard_sha256'] == g.sha256(ROOT / g.TOOL) and prov['runner_sha256'] == g.sha256(ROOT / e107.RUNNER)
    for name in ('oof_best.csv', 'oof_k36.csv'):
        df = pd.read_csv(run / name, dtype={UID: str})
        assert list(df.columns) == [UID] + T and len(df) == 2 and np.isfinite(df[T].to_numpy()).all()
    m = g.read(run / dq.MARKER)  # written by the entry hook inside the real shared runner, annotated after the run
    assert m['decision_eligible'] is False and m['tag'] == g.tag_for(fold, 'smoke') and m['protocol_sha256'] == g.sha256(REAL)
    assert m['outputs'] == {n: g.sha256(run / n) for n in ('oof_best.csv', 'oof_k36.csv')} and m['stage'].startswith('complete')
    assert dq.MARKER not in g.read(run / 'output-inventory.json') and dq.quarantine_reason(run / 'oof_best.csv')
