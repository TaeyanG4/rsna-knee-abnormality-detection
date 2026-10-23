"""EXP173 A1 CPU tests; no CUDA, Kaggle account, network, or corpus access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tools import exp173_gold_crossfit as e


def test_registered_arm_contract_and_split_audit(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'repo'
    (root / e.PROTOCOL).parent.mkdir(parents=True)
    (root / e.PROTOCOL).write_text('A1 registered before execution\n', encoding='utf-8')
    gold_path = root / e.e165.GOLD
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text('fake target-only CSV\n', encoding='utf-8')
    gold = {f'fake{i:02d}': np.asarray([(i + t) % 2 for t in range(12)], np.float32)
            for i in range(58)}
    audit = dict(report_groups=58, gold_series_studies=58, gold_series_uids=336,
                 shared_series_uids=0, gold_graph_components=0,
                 patient_image_near_duplicate_coverage='unavailable', limitation='fake')
    monkeypatch.setattr(e.e165, 'registration', lambda _: ({}, {}))
    monkeypatch.setattr(e.e165, 'load_gold', lambda *_: gold)
    monkeypatch.setattr(e, 'audit_gold_duplicates', lambda _: audit)
    receipt = e.split(root)
    assert receipt['half_sizes'] == [29, 29]
    first = (root / e.SPLIT).read_bytes()
    assert e.load_split(root)['duplicate_audit'] == audit
    assert e.ARMS == ('W4-S42', 'G4-H1', 'J4-H1', 'G4-H2', 'J4-H2')
    assert e.arm_parts('G4-H1') == ('G', 1, 2)
    assert e.arm_parts('J4-H1') == ('J', 1, 2)
    with pytest.raises(ValueError, match='Unknown cross-fit arm'):
        e.arm_parts('G1-H1')
    with pytest.raises(ValueError, match='immutable'):
        e.split(root)
    (root / e.SPLIT).unlink()
    e.split(root)
    assert (root / e.SPLIT).read_bytes() == first
    assert capsys.readouterr().out == ''


def test_matched_runner_adapter_changes_only_registered_label_contract(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    labels = root / e.EXP / 'arm-labels.csv'
    labels.parent.mkdir(parents=True)
    labels.write_text('fake\n', encoding='utf-8')
    monkeypatch.setattr(e, 'arm_labels', lambda *_: labels)
    monkeypatch.setattr(e.e165, 'runner_args', lambda *_: ['--tag', 'old', '--gold-labels', 'old'])
    seen = []

    def original_validate(args, path):
        assert args.tag == 'EXP165_G4_S42_F0'
        assert Path(args.gold_labels) == root / e.e165.GOLD
        return {'gold_copies': 4, 'seed': 42}, {'n_train': 3455}, {'train': []}, tmp_path

    monkeypatch.setattr(e.e165, 'validate_runner_args', original_validate)
    monkeypatch.setattr(e.e165, 'load_gold', lambda *_: None)
    from training.runners import exp165_gold_train as runner

    def fake_main(argv):
        opts = dict(zip(argv[::2], argv[1::2]))
        a = argparse.Namespace(tag=opts['--tag'], gold_labels=opts['--gold-labels'], bs=8, epochs=20)
        contract, _, _, _ = e.e165.validate_runner_args(a, root)
        contract['labels'] = e.e165.load_gold(labels, e.sha(labels))
        seen.append(contract)

    monkeypatch.setattr(runner, 'main', fake_main)
    monkeypatch.setattr(e.csv, 'DictReader', lambda _: [
        {e.e150.UID: f'fake{i}', **{t: str(i % 2) for t in e.e150.T}} for i in range(29)])
    e.train_arm(root, 'G4-H1', tmp_path, 'kaggle')
    e.train_arm(root, 'J4-H1', tmp_path, 'kaggle')
    assert len(seen) == 2
    assert all(x['gold_copies'] == 4 and x['train_loader'] == 446 and
               x['scheduler_steps'] == 8920 and x['max_fp16_skips'] == 89 and
               len(x['labels']) == 29 for x in seen)
    assert e.e165.validate_runner_args is original_validate


def test_half_stratified_paired_screen_and_cell_readout():
    halves = []
    for shift in (0, 1):
        y = np.asarray([[(i + t + shift) % 2 for t in range(12)] for i in range(29)], float)
        g = y * .8 + .1
        j = 1 - g
        halves.append(dict(y=y, jev=1-y, G4=g, J4=j, C=j.copy(), C_J=j.copy()))
    result = e.half_bootstrap(halves, draws=40)
    assert result['comparisons']['G4-J4']['delta'] == 1
    assert result['comparisons']['G4-J4']['ci95'][0] > 0
    assert result['comparisons']['G4-C']['delta'] == 1
    assert result['comparisons']['J4-C']['delta'] == 0
    assert result['bootstrap']['unit'] == 'study within scored half'
    cell = e.cell_readout(halves)
    assert cell['G4-J4']['cells']['JEV_disagrees_with_Gold']['n'] == 58 * 12
    assert cell['G4-J4']['cells']['JEV_disagrees_with_Gold']['mean_toward_Gold'] > 0


def test_private_stage_and_push_plan_do_not_push(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / e.SPLIT).parent.mkdir(parents=True)
    (root / e.SPLIT).write_text('{}\n', encoding='utf-8')
    source = root / 'fake.txt'
    source.write_text('fake private bundle\n', encoding='utf-8')
    control = root / e.CONTROL
    control.parent.mkdir(parents=True)
    control.write_bytes(b'fake checkpoint')
    monkeypatch.setattr(e, 'package_files', lambda _: {
        'fake.txt': {'sha256': e.sha(source), 'bytes': source.stat().st_size},
        e.CONTROL: {'sha256': e.sha(control), 'bytes': control.stat().st_size}})
    monkeypatch.setattr(e.e165, 'verify_run', lambda *_: {})
    out = tmp_path / 'stage'
    assert e.stage(out, root)['status'] == 'STAGED_NO_PUSH'
    assert all(json.loads((out / arm / 'kernel-metadata.json').read_text())['is_private']
               for arm in e.ARMS)
    plan = e.push_plan(out, root=root)
    assert plan.count('kaggle kernels push') == 5
    assert 'W4-S42 and G4-H1 first' in plan
    assert 'kaggle datasets create' in plan
    assert 'submit' not in plan
    assert '# & $kagglePython -X utf8 -m kaggle kernels push' in plan
    one = e.push_plan(out, 'W4-S42', root=root)
    assert one.count('kaggle kernels push') == 1
    assert '\n& $kagglePython -X utf8 -m kaggle kernels push' in one


def test_collect_rejects_truncated_output_before_copy(tmp_path):
    root = tmp_path / 'repo'
    split = root / e.SPLIT
    split.parent.mkdir(parents=True)
    split.write_text('{}\n', encoding='utf-8')
    control = root / e.CONTROL
    control.parent.mkdir(parents=True)
    control.write_bytes(b'fake checkpoint')
    source = tmp_path / 'download'
    source.mkdir()
    (source / 'exp173-status.json').write_text(json.dumps({
        'schema': 'exp173-remote-v1.1', 'state': 'COMPLETE', 'arm': 'W4-S43',
        'surface': 'vast', 'gpu': ['0, RTX 3090, fake driver'],
        'split_sha256': e.sha(split), 'control_checkpoint_sha256': e.sha(control)}), encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        e.collect(source, 'W4-S43', root)
    assert not (root / e.EXP / 'outputs/W4-S43').exists()


def test_registered_split_has_all_gold_studies_and_classes():
    if not (e.ROOT / e.SPLIT).is_file():
        pytest.skip('Project split absent in isolated copy')
    split = e.load_split()
    assert e.audit_gold_duplicates() == split['duplicate_audit']
    assert sorted(split['halves']['1'] + split['halves']['2']) == sorted(
        e.e165.load_gold(e.ROOT / e.e165.GOLD, e.sha(e.ROOT / e.e165.GOLD)))
