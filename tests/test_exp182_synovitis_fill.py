"""v1.1.0: private fresh stages and untimed contracts; historical QA stays immutable."""
from __future__ import annotations

import argparse
import ast
import shutil
import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from tools import exp182_synovitis_fill as t

ROOT = Path(__file__).resolve().parents[1]
HAS_EXP170 = (ROOT / t.EXP170_PROTOCOL).is_file()
HAS_BASE = (ROOT / t.BASE_ARCHIVE).is_file()


def synthetic(n=6):
    header = [t.UID, *t.TARGETS]
    rows = [[f'1.2.{i}', *[repr(0.1 * (j + 1) / 2) for j in range(12)]] for i in range(n)]
    return header, rows


def test_fill_rule_only_touches_silent_unguarded_synovitis():
    header, rows = synthetic()
    s, e = header.index('Synovitis'), header.index('Effusion')
    marker = {'1.2.0': 0.5, '1.2.1': 0.94, '1.2.2': 0.5, '1.2.3': 0.5, '1.2.5': 0.5}   # 1.2.4 missing
    relabel = {'1.2.0': 0.036, '1.2.1': 0.85, '1.2.2': 0.59, '1.2.4': 0.02, '1.2.5': 0.5899}  # 1.2.3 missing
    out, counts = t.fill_rows(header, rows, marker, relabel)
    filled = {'1.2.0', '1.2.5'}
    for old, new in zip(rows, out):
        assert [c for j, c in enumerate(old) if j != s] == [c for j, c in enumerate(new) if j != s]
        if old[0] in filled:
            expected = np.float32(0.22 + 0.41 * float(old[e]))
            assert new[s] == repr(float(expected))
        else:
            assert new[s] == old[s]
    assert counts['silent_filled'] == 2 and counts['explicit_marker'] == 1
    assert counts['explicit_relabel_guard'] == 1 and counts['missing_marker'] == 1
    assert counts['missing_relabel'] == 1 and counts['rows'] == 6


def test_fill_rule_rejects_wrong_header_and_bad_effusion():
    header, rows = synthetic(1)
    with pytest.raises(ValueError, match='header'):
        t.fill_rows(header[::-1], rows, {}, {})
    rows[0][header.index('Effusion')] = '1.5'
    with pytest.raises(ValueError, match='Effusion'):
        t.fill_rows(header, rows, {'1.2.0': 0.5}, {'1.2.0': 0.02})


def test_build_labels_reproduces_registered_file_and_counts():
    result = t.build_labels(ROOT, dry_run=True)
    prov, qa = t.read(ROOT / t.FILL_PROVENANCE), t.read(ROOT / t.FILL_QA)
    assert result['csv_sha256'] == prov['csv_sha256'] == t.sha(ROOT / t.FILL) == qa['csv_sha256']
    assert prov['protocol_sha256'] == t.sha(ROOT / t.PROTOCOL)
    assert qa['version'] == '1.0.0' and result['version'] == t.VERSION
    assert {k: qa[k] for k in result if k != 'version'} == {k: v for k, v in result.items() if k != 'version'}
    assert result['rows'] == 4347 and result['missing_marker'] == result['missing_relabel'] == 0
    assert 0.80 < result['silent_filled'] / result['rows'] < 0.88   # forum: 83.7 % not addressed
    assert qa['report_text_opened'] is False and qa['label_rows_printed'] is False


def test_filled_file_differs_from_l2_only_in_synovitis():
    h1, a = t._rows(ROOT / t.L2)
    h2, b = t._rows(ROOT / t.FILL)
    s = h1.index('Synovitis')
    assert h1 == h2 and [r[0] for r in a] == [r[0] for r in b]
    assert all(x[:s] + x[s+1:] == y[:s] + y[s+1:] for x, y in zip(a, b))
    changed = sum(x[s] != y[s] for x, y in zip(a, b))
    assert 0 < changed <= t.read(ROOT / t.FILL_QA)['silent_filled']


def test_build_labels_requires_protocol_and_pinned_inputs(tmp_path):
    with pytest.raises(ValueError, match='PROTOCOL'):
        t.build_labels(tmp_path)
    (tmp_path / t.PROTOCOL).parent.mkdir(parents=True)
    (tmp_path / t.PROTOCOL).write_text('registered', encoding='utf-8')
    for name in (t.L2, t.MARKER, t.RELABEL):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text('x', encoding='utf-8')
    with pytest.raises(ValueError, match='pinned label input changed'):
        t.build_labels(tmp_path)


def test_build_labels_never_overwrites():
    with pytest.raises(ValueError, match='already exists'):
        t.build_labels(ROOT)


def test_protocols_control_is_exp170_and_fill_changes_only_labels():
    c, f = t.build_protocol('CONTROL', ROOT), t.build_protocol('FILL', ROOT)
    if HAS_EXP170:
        assert t.control_matches_exp170(c, ROOT)
    from tools import exp166_convnext_t as source
    for q, arm in ((c, 'CONTROL'), (f, 'FILL')):
        _, semantic = source.old.load_label_csv(ROOT / t.LABELS[arm][0])
        assert q['label_set']['semantic_sha256'] == semantic and q['arm'] == arm
        assert q['paths']['labels'] == q['paths']['eval_labels'] == t.LABELS[arm][0]
    pc, pf = c['folds']['0']['pos_weight'], f['folds']['0']['pos_weight']
    syn = t.TARGETS.index('Synovitis')
    assert [i for i in range(12) if pc[i] != pf[i]] in ([], [syn])
    a, b = json.loads(json.dumps(c)), json.loads(json.dumps(f))
    for q in (a, b):
        for key in ('arm', 'label_set', 'paths', 'tags', 'recipe'):
            q.pop(key)
        q['folds']['0'].pop('tag'), q['folds']['0'].pop('pos_weight'), q['folds']['0'].pop('pos_weight_equals_exp099')
    assert a == b


@pytest.mark.skipif(not HAS_EXP170, reason='EXP170 overlay not present')
def test_control_mismatch_with_exp170_fails(monkeypatch):
    c = t.build_protocol('CONTROL', ROOT)
    c['common_args'] = dict(c['common_args'], epochs=19)
    with pytest.raises(ValueError, match='common_args'):
        t.control_matches_exp170(c, ROOT)


def test_runner_guard_and_arm_label_set(monkeypatch):
    runner = t.runner_source(ROOT)
    assert 'from tools import exp182_synovitis_fill as guard' in runner
    assert 'LABEL_SET' in t.guard_attributes(runner)
    monkeypatch.delenv('EXP182_ARM', raising=False)
    with pytest.raises(ValueError, match='EXP182_ARM'):
        t.LABEL_SET
    for arm in t.ARMS:
        monkeypatch.setenv('EXP182_ARM', arm)
        assert t.LABEL_SET == t.LABELS[arm][1]


def namespace(arm):
    q = t.build_protocol(arm, ROOT)
    args = t.runner_args(arm, ROOT)
    values = {}
    i = 2
    while i < len(args):
        key = args[i][2:].replace('-', '_')
        if i + 1 < len(args) and not args[i + 1].startswith('--'):
            values[key] = args[i + 1]; i += 2
        else:
            values[key] = True; i += 1
    absent = {k: v for k, v in q['common_args'].items() if k not in values}
    assert all(v is False for v in absent.values()), absent   # only store_true flags may be absent
    typed = dict(absent)
    for key, value in values.items():
        ref = q['common_args'].get(key)
        typed[key] = type(ref)(value) if ref is not None and not isinstance(ref, bool) else value
    typed.update(fold=int(values['fold']), seed=int(values['seed']), workers=int(values['workers']),
                 validate_only=False)
    return argparse.Namespace(**typed)


@pytest.mark.parametrize('arm', t.ARMS)
def test_validate_arguments_accepts_registered_arm_only(arm, monkeypatch):
    monkeypatch.setenv('EXP182_ARM', arm)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    from tools import exp166_convnext_t as source
    for key in source.THREADS:
        monkeypatch.setenv(key, '1')
    a = namespace(arm)
    q, info, ordered, out = t.validate_arguments(a, ROOT)
    assert q['arm'] == arm and info['tag'] == t.tag(arm) and len(ordered['train']) == 3455
    other = 'FILL' if arm == 'CONTROL' else 'CONTROL'
    for change, message in ((dict(tag=t.tag(other)), 'tag'), (dict(labels=str(ROOT / t.LABELS[other][0])), 'labels'),
                            (dict(epochs=19), 'epochs'), (dict(device='cpu'), 'device'),
                            (dict(output_root=str(ROOT)), 'output')):
        with pytest.raises(ValueError, match=message):
            t.validate_arguments(argparse.Namespace(**dict(vars(a), **change)), ROOT)
    monkeypatch.delenv('HF_HUB_OFFLINE')
    with pytest.raises(ValueError, match='environment'):
        t.validate_arguments(a, ROOT)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('EXP182_ARM', other)
    with pytest.raises(ValueError):
        t.validate_arguments(a, ROOT)


@pytest.mark.skipif(not (HAS_BASE and HAS_EXP170), reason='EXP166 base archive or EXP170 overlay absent')
def test_stage_is_private_no_internet_one_t4_kernel_and_tamper_evident(tmp_path):
    out = tmp_path / 'stage'
    root = private_stage_root(tmp_path / 'workspace', t)
    result = t.stage(out, root)
    assert result['status'] == 'STAGED_NO_PUSH'
    meta = t.read(out / 'kernel/kernel-metadata.json')
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['dataset_sources'] == [t.CORPUS_DATASET, t.DATASET]
    source = ''.join(t.read(out / 'kernel/run.ipynb')['cells'][0]['source'])
    compile(source, 'nb', 'exec')
    assert 'submit' not in source and 'remote-run' in source
    assert not any(k.arg == 'timeout' for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) for k in n.keywords)
    receipt = t.read(out / 'stage-receipt.json')
    assert receipt['pushed'] is False and set(receipt['files']) >= {t.FILL, t.L2, t.RUNNER, t.TOOL}
    with zipfile.ZipFile(out / 'code-dataset' / t.BASE_NAME) as z:
        assert not set(receipt['files']) & set(z.namelist())
    plan = t.push_plan(out, root)
    assert 'kernels push' in plan and 'submit' not in plan
    nb = out / 'kernel/run.ipynb'
    nb.write_text(nb.read_text(encoding='utf-8').replace('remote-run', 'remote-runx'), encoding='utf-8')
    with pytest.raises(ValueError, match='notebook'):
        t.verify_stage(out, root)


def test_remote_run_refuses_outside_kaggle_workspace():
    with pytest.raises(ValueError, match='Kaggle workspace'):
        t.remote_run(0.0, ROOT)


def fake_run(tmp_path, arm, gpu='Tesla T4'):
    q = t.read(ROOT / t.protocol_path(arm))
    info = q['folds']['0']
    run = tmp_path / t.tag(arm)
    run.mkdir()
    ids = t.read(ROOT / info['ordered_ids']['path'])['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        with (run / name).open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow([t.UID, *t.TARGETS])
            w.writerows([[u, *['0.5'] * 12] for u in ids])
    (run / 'model_best.pt').write_bytes(b'synthetic')
    env = {'gpu': gpu}
    args = dict(q['common_args'], seed=42, fold=0, tag=t.tag(arm), mode='full',
                labels=f'{t.WORKSPACE}/{t.LABELS[arm][0]}', eval_labels=f'{t.WORKSPACE}/{t.LABELS[arm][0]}')
    summary = dict(fold=0, train=3455, val=888, rows=888, selected_epoch_1based=20,
                   optimizer_updates=info['steps'], skipped_updates=0, label_set=t.LABELS[arm][1],
                   gold_access='disabled', environment=env, args=args)
    prov = dict(label_set=t.LABELS[arm][1], environment=env, initial_state_sha256=q['initial_state_sha256'],
                init_weights={'weights_sha256': q['init_weights']['sha256']}, runner_sha256=t.sha(ROOT / t.RUNNER),
                guard_sha256=t.sha(ROOT / t.TOOL), protocol_sha256=t.sha(ROOT / t.protocol_path(arm)),
                train_ids_ordered_sha256=info['ordered_ids']['train_sha256'],
                eval_ids_ordered_sha256=info['ordered_ids']['val_sha256'], actual_pos_weight=info['pos_weight'],
                flip_tta='coronal_axial_horizontal_mean_probability', label_file_sha256=q['label_set']['sha256'],
                label_semantic_sha256=q['label_set']['semantic_sha256'])
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    (run / 'history.json').write_text(json.dumps([{'epoch': i} for i in range(20)]), encoding='utf-8')
    inv = {n: {'bytes': (run / n).stat().st_size, 'sha256': t.sha(run / n)} for n in t.REQUIRED_OUTPUTS}
    (run / 'output-inventory.json').write_text(json.dumps(inv), encoding='utf-8')
    return run


@pytest.mark.parametrize('arm', t.ARMS)
def test_verify_run_accepts_complete_t4_arm_and_rejects_drift(tmp_path, arm):
    run = fake_run(tmp_path, arm)
    assert set(t.verify_run(run, arm, ROOT)) == set(t.REQUIRED_OUTPUTS)
    other = 'FILL' if arm == 'CONTROL' else 'CONTROL'
    with pytest.raises(ValueError):
        t.verify_run(run, other, ROOT)
    bad = fake_run(tmp_path / 'x', arm, gpu='NVIDIA GeForce RTX 3090') if (tmp_path / 'x').mkdir() is None else None
    with pytest.raises(ValueError, match='incomplete fixed-final'):
        t.verify_run(bad, arm, ROOT)
    (run / 'oof_k36.csv').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        t.verify_run(run, arm, ROOT)


def private_stage_root(out, module):
    """Copy explicit stage inputs; generated runner/protocol writes stay private."""
    from tools import exp166_convnext_t as base, exp170_convnext_t_l2 as e170
    from tools import exp182_synovitis_fill as e182
    names={base.MANIFEST,base.BASE,base.PROTOCOL,base.TOOL,base.WEIGHT,base.INITIAL,
           base.BASE_RUNNER,module.TOOL,module.PROTOCOL,module.BASE_ARCHIVE,
           module.EXP170_PROTOCOL,module.EXP170_MANIFEST,e170.FOLD0_ROSTER,
           module.L2,module.L2_PROVENANCE,'config/workspace.toml'}
    if module is e182:
        names.update([module.FILL,module.FILL_PROVENANCE,module.FILL_QA])
    else:
        names.update(module.DERIVED.values());names.update(module.DERIVED_PROVENANCE.values())
        names.add(e182.protocol_path('CONTROL'))
    before={name:module.sha(ROOT/name) for name in names}
    for name in names:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name in (base.WEIGHT,base.INITIAL,module.BASE_ARCHIVE):os.link(ROOT/name,target)
        else:shutil.copyfile(ROOT/name,target)
    (out/module.RUNNER).parent.mkdir(parents=True,exist_ok=True)
    assert before == {name:module.sha(out/name) for name in names}
    return out
