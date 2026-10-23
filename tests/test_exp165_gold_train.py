"""EXP165 CPU fakes: no CUDA context, corpus, Kaggle account or network."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile

import numpy as np
import pytest

from tools import exp165_gold_train as e
from training.runners.exp165_gold_train import GoldIndex


def test_official_reader_uses_only_complete_binary_targets(tmp_path, capsys):
    path = tmp_path / 'train.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([e.e150.UID, 'Report', *e.e150.T])
        for i in range(58):
            writer.writerow([f'private-{i}', 'private report text', *([i % 2] * 12)])
        writer.writerow(['unlabelled', 'private report text', *([''] * 12)])
    rows = e.labels_from_official(path)
    assert len(rows) == 58
    assert len(next(iter(rows.values()))) == 12
    assert capsys.readouterr().out == ''
    with path.open('a', encoding='utf-8') as stream:
        stream.write('partial,private report text,1\n')
    with pytest.raises(ValueError, match='complete binary'):
        e.labels_from_official(path)
    assert capsys.readouterr().out == ''


def test_gold_uid_and_component_exclusion():
    gold = [f'g{i}' for i in range(58)]
    roster = dict(train=['t0'], val=[f'v{i}' for i in range(888)])
    components = [dict(component_id='train', members=['t0']),
                  dict(component_id='val', members=roster['val']),
                  dict(component_id='gold', members=gold)]
    audit = e.disjoint_audit(gold, [roster] * 5, components, gold + ['t0', *roster['val']])
    assert audit['uid_overlap'] == audit['component_overlap'] == 0
    with pytest.raises(ValueError, match='component'):
        changed = [*components[:-1], dict(component_id='train', members=gold)]
        e.disjoint_audit(gold, [roster] * 5, changed, gold + ['t0', *roster['val']])
    with pytest.raises(ValueError, match='UID'):
        e.disjoint_audit(gold, [dict(train=['g0'], val=roster['val'])] * 5,
                         components, gold + ['t0', *roster['val']])
    groups = {u: 'a' * 64 for u in gold}
    groups.update({u: 'b' * 64 for u in ['t0', *roster['val']]})
    assert e.disjoint_audit(gold, [roster] * 5, components,
                            gold + ['t0', *roster['val']], groups)['report_component_overlap'] == 0
    groups['g0'] = 'b' * 64
    with pytest.raises(ValueError, match='report component'):
        e.disjoint_audit(gold, [roster] * 5, components,
                         gold + ['t0', *roster['val']], groups)


def test_gold_csv_hash_and_values(tmp_path):
    path = tmp_path / 'gold.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([e.e150.UID, *e.e150.T])
        for i in range(58):
            writer.writerow([f'g{i}', *([i % 2] * 12)])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(e.load_gold(path, digest)) == 58
    with path.open('a', encoding='utf-8') as stream:
        stream.write('tampered\n')
    with pytest.raises(ValueError, match='hash'):
        e.load_gold(path, digest)


def test_runner_guard_api_and_notebook_policy():
    assert e.guard_interface() >= 8
    source = e.notebook_source('a' * 64, 'b' * 64)
    compile(source, '<fake-notebook>', 'exec')
    assert "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'" in source
    assert 'remote-run' in source and 'EXP165_REMOTE' in source
    assert 'submit' not in source


def test_stage_archive_unique_entry_check(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'test.py').write_text('pass\n', encoding='utf-8')
    monkeypatch.setattr(e, 'guard_interface', lambda: 12)
    monkeypatch.setattr(e, 'package_files', lambda _: {'test.py': dict(
        sha256=e.sha(root / 'test.py'), bytes=(root / 'test.py').stat().st_size)})
    monkeypatch.setattr(e, 'PROTOCOL', 'test.py')
    monkeypatch.setattr(e, 'MANIFEST', 'test.py')
    out = tmp_path / 'stage'
    e.stage(out, root)
    archive = out / 'code-dataset/exp165-code.zip.bin'
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
    assert e.verify_stage(out, root)['schema'] == 'exp165-stage-v1'
    vast = tmp_path / 'vast'
    e.stage(vast, root, surface='vast')
    with zipfile.ZipFile(vast / 'package/exp165-code.zip.bin') as z:
        assert len(z.namelist()) == len(set(z.namelist()))
    assert e.verify_stage(vast, root)['surface'] == 'vast'


def test_gold_index_uses_four_distinct_augmentation_streams():
    class Fake:
        def __init__(self, ids, seed):
            self.ids, self.seed, self.epoch = ids, seed, 0

        def __getitem__(self, index):
            return self.ids[index], self.seed, self.epoch

        def __len__(self):
            return len(self.ids)

        def set_epoch(self, epoch):
            self.epoch = epoch

    index = GoldIndex(Fake(['base'], 42), lambda ids, seed: Fake(ids, seed), 42, ['gold'])
    assert len(index) == 5
    assert index.ids == ['base', 'gold', 'gold', 'gold', 'gold']
    assert len({index[i][1] for i in range(1, 5)}) == 4
    index.set_epoch(3)
    assert all(index[i][2] == 3 for i in range(5))


def test_registered_harm_boundary_and_seed_count():
    def body(primary, dread):
        return {'primary_mean_delta': primary, 'metrics': {'dread': {'delta': dread}}}

    assert e.harm_decision({'kaggle': body(-.003, 0), 'vast': body(0, -.003)}) == 'ADOPT_GOLD_X4'
    assert e.harm_decision({'kaggle': body(-.00301, 0), 'vast': body(0, 0)}) == 'USE_GOLD_X1_NO_RESCREEN'
    assert e.harm_decision({'kaggle': body(-.00301, 0), 'vast': body(0, -.00301)}) == 'EXCLUDE_GOLD'
    with pytest.raises(ValueError, match='Both registered seeds'):
        e.harm_decision({'kaggle': body(0, 0)})


def test_paired_mean_ci_identical_predictions():
    y = np.tile(np.array([[0.] * 12, [1.] * 12]), (40, 1))
    p = np.tile(np.array([[.2] * 12, [.8] * 12]), (40, 1))
    ref = (y, np.ones(y.shape, bool), 1)
    draws = [np.arange(80)] * 4
    assert e.paired_mean_ci(p, p, (ref, ref), draws) == [0.0, 0.0]
