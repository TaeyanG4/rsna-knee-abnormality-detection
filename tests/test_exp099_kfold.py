"""CPU-only checks for tools/exp099_kfold.py (fold construction, contracts, label registration, OOF assembly). No GPU, no training."""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp099_kfold as m  # noqa: E402

UID = m.UID
BASE_DIR = ROOT / 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL'


@pytest.fixture(scope='module')
def real():
    base = m.read(BASE_DIR / 'ordered-ids.json')
    rows = m.read_rows(BASE_DIR / 'group-metadata.csv')
    labels, _ = m._load_labels(ROOT / m.DREAD)
    return base, rows, labels, m.build_folds(base, rows, labels)


# ------------------------------------------------------------------ fold construction on the real G1 population
def test_fold0_is_exactly_g1_val(real):
    base, _, _, fold = real
    assert {u for u, k in fold.items() if k == 0} == set(base['val'])
    split = m.fold_split(fold, m.canonical_order(base), 0)
    assert split == {'train': base['train'], 'val': base['val']}


def test_every_uid_in_exactly_one_fold_and_no_component_split(real):
    base, rows, _, fold = real
    assert set(fold) == set(base['train']) | set(base['val']) and len(fold) == 4343
    assert set(fold.values()) == {0, 1, 2, 3, 4}
    canonical = m.canonical_order(base)
    vals = [m.fold_split(fold, canonical, k)['val'] for k in m.FOLDS]
    assert sorted(u for v in vals for u in v) == sorted(canonical)
    folds_of_comp = {}
    for r in rows:
        folds_of_comp.setdefault(r['component_id'], set()).add(fold[r[UID]])
    assert all(len(v) == 1 for v in folds_of_comp.values())


def test_split_fold_counts_balanced(real):
    _, _, labels, fold = real
    counts = [sum(1 for k in fold.values() if k == f) for f in (1, 2, 3, 4)]
    mean = sum(counts) / 4
    assert sum(counts) == 3455 and all(abs(c - mean) <= 0.03 * mean for c in counts)
    pos = np.array([np.rint(np.stack([labels[u] for u, k in fold.items() if k == f]).astype(np.float32)).sum(0) for f in (1, 2, 3, 4)])
    assert (pos.max(0) - pos.min(0) <= np.maximum(3, 0.05 * pos.mean(0))).all()  # per-target positives also balanced


def test_build_folds_deterministic_and_row_order_free(real):
    base, rows, labels, fold = real
    shuffled = list(rows); random.Random(7).shuffle(shuffled)
    assert m.build_folds(base, shuffled, labels) == fold
    assert m.build_folds(base, rows, labels, seed=m.FOLD_SEED + 1) != fold  # the seed actually matters (tiebreaks)


def test_fold_lists_follow_canonical_order(real):
    base, rows, _, fold = real
    canonical = m.canonical_order(base)
    for k in m.FOLDS:
        split = m.fold_split(fold, canonical, k)
        val = set(split['val'])
        g = [{**r, 'split': 'val' if r[UID] in val else 'train'} for r in rows]
        assert m.validate_fold_membership(canonical, split, g, canonical) == (split['train'], split['val'])


# ------------------------------------------------------------------ contracts
def test_contract_steps_and_skip_gate():
    assert m.contract(3455) == dict(len_train_loader=431, steps=3448, scheduler_total_steps=3448, max_fp16_skips=34, eval_batch=4)
    c = m.contract(3479)
    assert (c['len_train_loader'], c['steps'], c['max_fp16_skips']) == (434, 3472, 34)
    assert m.contract(3480)['len_train_loader'] == 435 and m.contract(3487)['len_train_loader'] == 435  # drop_last


def test_pos_weight_rule_reproduces_g1_baseline(real):
    base, _, labels, _ = real
    expected = np.asarray(m.read(BASE_DIR / 'manifest.json')['loss']['baseline_pos_weight'], np.float32)
    got = m.pos_weight(labels, base['train'])
    assert got.dtype == np.float32 and np.array_equal(got, expected)
    assert np.array_equal(np.asarray(got.tolist(), np.float32), got)  # JSON round trip is exact


def test_pos_weight_clipping():
    labels = {'a': np.array([1.0] * 6 + [0.0] * 6, np.float32), 'b': np.array([1.0] * 6 + [0.0] * 6, np.float32)}
    w = m.pos_weight(labels, ['a', 'b'])
    assert np.allclose(w[:6], 1.0) and np.allclose(w[6:], 10.0)  # prev 1 -> .7 -> .43 -> clip 1; prev 0 -> .03 -> 32 -> clip 10


# ------------------------------------------------------------------ membership validation (synthetic)
def _rows(spec):
    return [{UID: u, 'component_id': c, 'split': s} for u, c, s in spec]


def test_membership_rejects_component_crossing_and_order():
    canonical = ['u1', 'u2', 'u3', 'u4']
    rows = _rows([('u1', 'c1', 'train'), ('u2', 'c2', 'train'), ('u3', 'c3', 'val'), ('u4', 'c4', 'val')])
    ok = {'train': ['u1', 'u2'], 'val': ['u3', 'u4']}
    assert m.validate_fold_membership(canonical, ok, rows, canonical)
    with pytest.raises(ValueError, match='canonical'):
        m.validate_fold_membership(canonical, {'train': ['u2', 'u1'], 'val': ['u3', 'u4']}, rows, canonical)
    crossing = _rows([('u1', 'c1', 'train'), ('u2', 'c2', 'train'), ('u3', 'c2', 'val'), ('u4', 'c4', 'val')])
    with pytest.raises(ValueError, match='Component'):
        m.validate_fold_membership(canonical, ok, crossing, canonical)
    with pytest.raises(ValueError, match='coverage'):
        m.validate_fold_membership(canonical, {'train': ['u1'], 'val': ['u3', 'u4']}, rows, canonical)
    with pytest.raises(ValueError, match='coverage'):
        m.validate_fold_membership(['u1', 'u2', 'u3'], ok, rows, canonical)  # study missing from corpus


# ------------------------------------------------------------------ label registration (synthetic workspace)
def _write_labels(path, uids, value=0.2):
    df = pd.DataFrame({UID: uids, **{t: [value] * len(uids) for t in m.T}})
    df.loc[df.index[::3], 'ACL'] = 0.9
    df.to_csv(path, index=False)


@pytest.fixture()
def workspace(tmp_path):
    proto = tmp_path / m.PROTOCOL
    folds = {0: {'train': ['a', 'b', 'c'], 'val': ['d', 'e']}, 1: {'train': ['c', 'd', 'e'], 'val': ['a', 'b']}}
    manifest = dict(schema=m.SCHEMA, paths=dict(label_sets_dir=m.PROTOCOL + '/label-sets'), label_sets={},
                    folds={str(k): dict(ordered_ids=dict(path=f'{m.PROTOCOL}/fold{k}/ordered-ids.json'), pos_weight=[1.0] * 12) for k in folds})
    for k, v in folds.items():
        (proto / f'fold{k}').mkdir(parents=True)
        (proto / f'fold{k}' / 'ordered-ids.json').write_text(json.dumps(v), encoding='utf-8')
    (proto / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    comp = tmp_path / 'storage/competition'; comp.mkdir(parents=True)
    with (comp / 'train.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow([UID, 'Report', 'ACL'] + m.T[1:])
        w.writerow(['g1', 'gold report, with comma', 1] + [0] * 11)
        w.writerow(['x9', 'unlabelled report', ''] + [''] * 11)
    (tmp_path / 'labels').mkdir()
    return tmp_path


def test_register_labels_accepts_clean_file(workspace):
    _write_labels(workspace / 'labels/ok.csv', ['a', 'b', 'c', 'd', 'e', 'x9'])
    rec = m.register_labels('teacher_v1', 'labels/ok.csv', root=workspace)
    assert rec['eligible_covered'] == 5 and rec['gold_overlap'] == 0 and rec['extra_rows_outside_g1'] == 1
    assert set(rec['pos_weight']) == {'0', '1'} and len(rec['pos_weight']['0']) == 12
    reg, _ = m.load_registration(workspace, m.read(workspace / m.PROTOCOL / 'manifest.json'), 'teacher_v1', workspace / m.PROTOCOL / 'manifest.json')
    assert reg['sha256'] == m.sha256(workspace / 'labels/ok.csv')
    with pytest.raises(SystemExit, match='already registered'):
        m.register_labels('teacher_v1', 'labels/ok.csv', root=workspace)


def test_register_labels_rejects_gold_rows(workspace):
    _write_labels(workspace / 'labels/gold.csv', ['a', 'b', 'c', 'd', 'e', 'g1'])
    with pytest.raises(ValueError, match='Gold'):
        m.register_labels('with_gold', 'labels/gold.csv', root=workspace)
    assert not (workspace / m.PROTOCOL / 'label-sets/with_gold.json').exists()


def test_register_labels_rejects_missing_train_uid(workspace):
    _write_labels(workspace / 'labels/short.csv', ['a', 'b', 'c', 'd'])  # 'e' is in fold-1 train
    with pytest.raises(ValueError, match='fold-1 train'):
        m.register_labels('short', 'labels/short.csv', root=workspace)


def test_registration_detects_changed_file(workspace):
    _write_labels(workspace / 'labels/ok.csv', ['a', 'b', 'c', 'd', 'e'])
    m.register_labels('ok', 'labels/ok.csv', root=workspace)
    _write_labels(workspace / 'labels/ok.csv', ['a', 'b', 'c', 'd', 'e'], value=0.3)
    mp = workspace / m.PROTOCOL / 'manifest.json'
    with pytest.raises(ValueError, match='changed since registration'):
        m.load_registration(workspace, m.read(mp), 'ok', mp)


def test_register_labels_rejects_bad_name_and_outside_path(workspace, tmp_path_factory):
    _write_labels(workspace / 'labels/ok.csv', ['a', 'b', 'c', 'd', 'e'])
    with pytest.raises(SystemExit):
        m.register_labels('Bad-Name', 'labels/ok.csv', root=workspace)
    outside = tmp_path_factory.mktemp('outside') / 'x.csv'
    _write_labels(outside, ['a', 'b', 'c', 'd', 'e'])
    with pytest.raises(ValueError, match='inside the workspace'):
        m.register_labels('outside', str(outside), root=workspace)


# ------------------------------------------------------------------ assemble (synthetic folds)
def _frame(uids, scores):
    return pd.DataFrame({UID: uids, **{t: scores for t in m.T}})


def test_assemble_and_score_synthetic(tmp_path):
    uids = [f'u{i:02d}' for i in range(12)]
    y = np.array([i % 2 for i in range(12)], np.float32)
    lab = pd.DataFrame({UID: uids, **{t: np.where(y == 1, 0.95, 0.05) for t in m.T}})
    lab.to_csv(tmp_path / 'labels.csv', index=False)
    val_ids = {0: uids[0:4], 1: uids[4:8], 2: uids[8:12]}
    perfect = {k: _frame(v, [0.9 if int(u[1:]) % 2 else 0.1 for u in v]) for k, v in val_ids.items()}
    inverted = {k: _frame(v, [0.1 if int(u[1:]) % 2 else 0.9 for u in v]) for k, v in val_ids.items()}
    r = m.assemble_and_score(perfect, inverted, val_ids, tmp_path / 'labels.csv')
    assert r['oof_k24'][UID].tolist() == sorted(uids) and r['k36_missing'] == []
    assert r['metrics']['k24']['pooled']['macro12'] == 1.0 and r['metrics']['k36']['pooled']['macro12'] == 0.0
    assert all(r['metrics']['k24'][f'fold{k}']['macro12'] == 1.0 for k in val_ids)
    partial = m.assemble_and_score(perfect, {0: inverted[0]}, val_ids, tmp_path / 'labels.csv')
    assert partial['oof_k36'] is None and partial['k36_missing'] == [1, 2] and 'k36' not in partial['metrics']
    bad = dict(perfect); bad[1] = perfect[1].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match='fold 1'):
        m.assemble_and_score(bad, {}, val_ids, tmp_path / 'labels.csv')
    with pytest.raises(ValueError, match='overlap'):
        m.assemble_and_score(perfect, {}, {0: uids[0:4], 1: uids[3:8], 2: uids[8:12]}, tmp_path / 'labels.csv')


def test_auc_matches_sklearn_convention():
    yy = np.array([0, 0, 1, 1, 0, 1]); s = np.array([.1, .4, .35, .8, .4, .4])
    from sklearn.metrics import roc_auc_score
    assert abs(m.auc(yy, s) - roc_auc_score(yy, s)) < 1e-12


# ------------------------------------------------------------------ the real registration (skipped until prepare has run)
REAL = ROOT / m.PROTOCOL / 'manifest.json'


@pytest.mark.skipif(not REAL.exists(), reason='protocol not prepared')
def test_real_protocol_files(real):
    base, rows, labels, fold = real
    p = m.read(REAL)
    assert p['schema'] == m.SCHEMA and p['common_args'] == m.COMMON == m.read(ROOT / m.V4)['common_args'] and p['seed'] == 42
    assert p['folds_csv_sha256'] == m.sha256(ROOT / p['paths']['folds'])
    folds_csv = pd.read_csv(ROOT / p['paths']['folds'], dtype={UID: str})
    assert dict(zip(folds_csv[UID], folds_csv['fold'])) == fold
    canonical = m.canonical_order(base)
    for k, spec in p['folds'].items():
        k = int(k)
        ordered = m.read(ROOT / spec['ordered_ids']['path'])
        assert ordered == m.fold_split(fold, canonical, k)
        assert m.sha256(ROOT / spec['ordered_ids']['path']) == spec['ordered_ids']['sha256']
        assert m.uid_hash(ordered['train']) == spec['ordered_ids']['train_sha256'] and m.uid_hash(ordered['val']) == spec['ordered_ids']['val_sha256']
        g = m.read_rows(ROOT / spec['groups']['path'])
        assert m.sha256(ROOT / spec['groups']['path']) == spec['groups']['sha256'] and [r[UID] for r in g] == [r[UID] for r in rows]
        m.validate_fold_membership(canonical, ordered, g, canonical)
        c = m.contract(len(ordered['train']))
        assert all(spec[key] == c[key] for key in c) and spec['n_train'] + spec['n_val'] == 4343
        assert np.array_equal(np.asarray(spec['pos_weight'], np.float32), m.pos_weight(labels, ordered['train']))
    assert p['folds']['0']['groups']['sha256'] == m.read(BASE_DIR / 'manifest.json')['groups']['sha256']


@pytest.mark.skipif(not (ROOT / m.PROTOCOL / 'label-sets/dread.json').exists(), reason='dread not registered')
def test_real_queue_entries():
    entries = m.queue_entries('dread', [1, 2, 3, 4], after='PREV')
    assert [e['tag'] for e in entries] == [f'EXP099_B8_DREAD_F{k}' for k in (1, 2, 3, 4)]
    assert [e['after'] for e in entries] == ['PREV', 'EXP099_B8_DREAD_F1', 'EXP099_B8_DREAD_F2', 'EXP099_B8_DREAD_F3']
    for k, e in zip((1, 2, 3, 4), entries):
        cmd = e['command']
        assert cmd[1].endswith('exp099_kfold.py') and cmd[cmd.index('--fold') + 1] == str(k) and cmd[cmd.index('--tag') + 1] == e['tag']
        assert '--grad-ckpt' in cmd and cmd[cmd.index('--amp') + 1] == 'fp16' and cmd[cmd.index('--seed') + 1] == '42'


# ------------------------------------------------------------------ end-to-end assemble / fold-0 reuse in a synthetic workspace
@pytest.fixture()
def mini(tmp_path):
    """3 folds x 4 studies, a Dread-like label file, registered 'dread', no Gold rows."""
    uids = [f'u{i:02d}' for i in range(12)]
    y = np.array([i % 2 for i in range(12)])
    (tmp_path / Path(m.DREAD).parent).mkdir(parents=True)
    pd.DataFrame({UID: uids, **{t: np.where(y == 1, 0.95, 0.05) for t in m.T}}).to_csv(tmp_path / m.DREAD, index=False)
    labels, semantic = m._load_labels(tmp_path / m.DREAD)
    folds = {k: dict(train=[u for u in uids if u not in uids[4 * k:4 * k + 4]], val=uids[4 * k:4 * k + 4]) for k in range(3)}
    proto = tmp_path / m.PROTOCOL
    spec = {}
    for k, v in folds.items():
        (proto / f'fold{k}').mkdir(parents=True)
        (proto / f'fold{k}' / 'ordered-ids.json').write_text(json.dumps(v), encoding='utf-8')
        spec[str(k)] = dict(ordered_ids=dict(path=f'{m.PROTOCOL}/fold{k}/ordered-ids.json', train_sha256=m.uid_hash(v['train']), val_sha256=m.uid_hash(v['val'])),
                            pos_weight=m.pos_weight(labels, v['train']).tolist(), n_train=8, n_val=4, steps=8, max_fp16_skips=0)
    manifest = dict(schema=m.SCHEMA, paths=dict(label_sets_dir=m.PROTOCOL + '/label-sets', corpus='corpus'), common_args=m.COMMON, initial_state_sha256='init',
                    label_sets={'dread': dict(path=m.DREAD, sha256=m.sha256(tmp_path / m.DREAD), semantic_sha256=semantic)}, folds=spec)
    (proto / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (tmp_path / 'storage/competition').mkdir(parents=True)
    pd.DataFrame({UID: ['g1'], 'Report': ['r'], 'ACL': [1]}).to_csv(tmp_path / m.GOLD_SOURCE, index=False)
    m.register_labels('dread', m.DREAD, root=tmp_path)
    return tmp_path, folds, semantic


def _fake_run(root, rel, k, folds, semantic, k36=True, reused=False):
    run = root / rel
    run.mkdir(parents=True)
    val = folds[k]['val']
    score = [0.9 if int(u[1:]) % 2 else 0.1 for u in val]
    _frame(val, score).to_csv(run / 'oof_best.csv', index=False)
    if k36:
        _frame(val, [1 - s for s in score]).to_csv(run / 'oof_k36.csv', index=False)
    prov = dict(eval_ids_ordered_sha256=m.uid_hash(val), train_ids_ordered_sha256=m.uid_hash(folds[k]['train']), label_hash=semantic, seed=42,
                actual_pos_weight=m.read(root / m.PROTOCOL / 'manifest.json')['folds'][str(k)]['pos_weight'], initial_state_sha256='init',
                checkpoint_selection='fixed-final')
    if not reused:
        prov.update(fold=k, label_set='dread', label_semantic_sha256=semantic)
    args = {**m.COMMON, 'seed': 42, 'variant': 'seed', 'slot_dropout': 0.0, 'labels': str(root / m.DREAD), 'eval_labels': str(root / m.DREAD),
            'corpus': str(root / 'corpus')}
    summary = dict(provenance=prov, args=args, optimizer_updates=8, skipped_updates=0, selected_epoch_1based=8, train=8, val=4)
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    for n in ('history.json', 'model_best.pt'):
        (run / n).write_text('x', encoding='utf-8')
    names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv']
    (run / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=m.sha256(run / n)) for n in names}), encoding='utf-8')
    return run


def _mark_success(root, tag):
    d = root / 'artifacts/logs' / tag
    d.mkdir(parents=True)
    (d / 'SUCCESS').write_text('exit 0', encoding='utf-8')


def test_assemble_end_to_end(mini):
    root, folds, semantic = mini
    for k in folds:
        _fake_run(root, f'artifacts/experiments/{m.tag_for("dread", k, "full")}', k, folds, semantic)
    body = m.assemble('dread', root=root)
    out = root / m.oof_dir_for('dread')
    k24 = pd.read_csv(out / 'oof_k24.csv', dtype={UID: str})
    assert k24[UID].tolist() == sorted(u for v in folds.values() for u in v['val']) and list(k24.columns) == [UID] + m.T
    assert body['metrics']['k24']['pooled']['macro12'] == 1.0 and body['metrics']['k36']['pooled']['macro12'] == 0.0
    assert (out / 'oof_k36.csv').exists() and (out / 'result.md').exists() and body['k36_missing_folds'] == []
    with pytest.raises(SystemExit):
        m.assemble('dread', root=root)  # refuses to overwrite


def test_assemble_incomplete_and_wrong_provenance(mini):
    root, folds, semantic = mini
    run = _fake_run(root, f'artifacts/experiments/{m.tag_for("dread", 0, "full")}', 0, folds, semantic)
    with pytest.raises(SystemExit):
        m.assemble('dread', root=root)  # folds 1, 2 missing
    _fake_run(root, f'artifacts/experiments/{m.tag_for("dread", 1, "full")}', 1, folds, semantic)
    _fake_run(root, f'artifacts/experiments/{m.tag_for("dread", 2, "full")}', 2, folds, semantic)
    s = m.read(run / 'summary.json')
    s['provenance']['label_semantic_sha256'] = 'other'
    (run / 'summary.json').write_text(json.dumps(s), encoding='utf-8')
    with pytest.raises(SystemExit):
        m.assemble('dread', root=root)
    assert not (root / m.oof_dir_for('dread')).exists()


def test_reuse_fold0_then_assemble_with_k36_override(mini, monkeypatch):
    root, folds, semantic = mini
    rel = 'artifacts/experiments/FAKE_EXP094L'
    monkeypatch.setattr(m, 'REUSE_RUN', rel)
    with pytest.raises(SystemExit) as e:
        m.reuse_fold0(root=root)  # run absent -> not finished, nothing written
    assert e.value.code == 2 and not (root / m.PROTOCOL / 'fold0-reuse.json').exists()
    _fake_run(root, rel, 0, folds, semantic, k36=False, reused=True)
    with pytest.raises(SystemExit) as e:
        m.reuse_fold0(root=root)  # files present but no SUCCESS marker yet
    assert e.value.code == 2
    _mark_success(root, 'FAKE_EXP094L')
    rec = m.reuse_fold0(root=root)
    assert rec['status'] == 'REUSED_AS_FOLD0' and all(rec['checks'].values()) and rec['k36_available'] is False
    for k in (1, 2):
        _fake_run(root, f'artifacts/experiments/{m.tag_for("dread", k, "full")}', k, folds, semantic)
    override = root / 'k36_f0.csv'
    _frame(folds[0]['val'], [0.5] * 4).to_csv(override, index=False)
    body = m.assemble('dread', {0: str(override)}, root=root)
    assert body['sources']['0']['reused'] and body['k36_missing_folds'] == [] and body['metrics']['k36']['fold0']['macro12'] == 0.5


def test_reuse_fold0_rejects_mismatch(mini, monkeypatch):
    root, folds, semantic = mini
    rel = 'artifacts/experiments/FAKE_EXP094L'
    monkeypatch.setattr(m, 'REUSE_RUN', rel)
    run = _fake_run(root, rel, 0, folds, semantic, k36=False, reused=True)
    _mark_success(root, 'FAKE_EXP094L')
    s = m.read(run / 'summary.json')
    s['args']['bb_lr'] = 5e-5
    (run / 'summary.json').write_text(json.dumps(s), encoding='utf-8')
    with pytest.raises(SystemExit) as e:
        m.reuse_fold0(root=root)
    assert e.value.code == 3 and not (root / m.PROTOCOL / 'fold0-reuse.json').exists()
