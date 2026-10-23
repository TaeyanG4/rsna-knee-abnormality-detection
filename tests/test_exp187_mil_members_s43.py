# Compatibility tests v2.0.0: retired launch guards, archived scientific parity, collected readouts.
"""CPU checks for EXP187 private staging and fail-closed collection."""
import json
from pathlib import Path
import zipfile

import pytest

from tools import exp187_mil_members_s43 as exp


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _source(tmp_path, monkeypatch, group='01'):
    from tools import exp150_consensus_weight as corpus
    root = tmp_path / 'repo'
    source = tmp_path / 'download'
    folds = exp.KERNELS[group]
    completed = {}
    for fold in folds:
        tag = exp.tag(fold)
        run = source / f'fold{fold}' / 'results' / tag
        run.mkdir(parents=True)
        (run / 'model_best.pt').write_bytes(f'model-{fold}'.encode())
        _json(run / 'output-inventory.json', {'model_best.pt': {
            'sha256': exp.sha(run / 'model_best.pt'),
            'bytes': (run / 'model_best.pt').stat().st_size}})
        protocol = root / exp.protocol_path(fold)
        roster = root / exp.roster_path(fold)
        _json(roster, {'val': [f'uid-{fold}']})
        _json(protocol, {'fold': fold, 'exp185': {'model_sha256': 'model'},
                         'label_set': {'sha256': 'label'},
                         'folds': {str(fold): {'ordered_ids': {'sha256': exp.sha(roster)}}}})
        receipt = dict(schema=exp.SCHEMA, status='VALIDATED_SUCCESS',
                       version=exp.HISTORICAL_VERSION, group=group, fold=fold, tag=tag,
                       seed=43, protocol_sha256=exp.sha(protocol),
                       runner_sha256='runner', tool_sha256='tool',
                       model_sha256='model', label_sha256='label',
                       roster_sha256=exp.sha(roster),
                       corpus_manifest_sha256=corpus.CORPUS_HASH,
                       output_sha256={'model_best.pt': exp.sha(run / 'model_best.pt')},
                       inventory_sha256=exp.sha(run / 'output-inventory.json'),
                       elapsed_seconds=1)
        _json(source / f'fold{fold}' / f'validated-fold{fold}.json', receipt)
        completed[str(fold)] = receipt
    _json(source / 'exp187-status.json', dict(schema=exp.SCHEMA,
          version=exp.HISTORICAL_VERSION, state='COMPLETE', group=group,
          folds=list(folds), completed=completed, total_seconds=3))
    (root / exp.RUNNER).parent.mkdir(parents=True, exist_ok=True)
    (root / exp.RUNNER).write_bytes(b'runner')
    monkeypatch.setattr(exp, 'TOOL', 'tool.py')
    (root / 'tool.py').write_bytes(b'tool')
    monkeypatch.setattr(exp, 'MODEL', 'model.py')
    (root / 'model.py').write_bytes(b'model')
    for fold in folds:
        protocol = root / exp.protocol_path(fold)
        q = exp.read(protocol)
        q['exp185']['model_sha256'] = exp.sha(root / 'model.py')
        _json(protocol, q)
        receipt_path = source / f'fold{fold}' / f'validated-fold{fold}.json'
        receipt = exp.read(receipt_path)
        receipt['protocol_sha256'] = exp.sha(protocol)
        receipt['runner_sha256'] = exp.sha(root / exp.RUNNER)
        receipt['tool_sha256'] = exp.sha(root / exp.TOOL)
        receipt['model_sha256'] = exp.sha(root / 'model.py')
        _json(receipt_path, receipt)
        completed[str(fold)] = receipt
    status = exp.read(source / 'exp187-status.json')
    status['completed'] = completed
    _json(source / 'exp187-status.json', status)
    monkeypatch.setattr(exp, 'verify_run', lambda run, fold, root: {
        'model_best.pt': exp.sha(Path(run) / 'model_best.pt')})
    return root, source


@pytest.mark.parametrize('entry,args', [('notebook_source', ('01', '0'*64)),
    ('stage', (None,)), ('runner_source', ()), ('runner_args', (0,)),
    ('remote_run', (None, None, None, '01')), ('push_plan', (None,))])
def test_private_launch_paths_retired(tmp_path, entry, args):
    with pytest.raises(ValueError, match='retired'):
        getattr(exp, entry)(*args)
    assert not list(tmp_path.iterdir())
    assert '__getattr__' not in vars(exp)


def test_staged_kernel_metadata_has_no_competition_source(tmp_path):
    for group in exp.KERNELS:
        exp.write(tmp_path / f'kernel-{group}' / 'kernel-metadata.json',
                  exp.kernel_metadata(group))
    for group in exp.KERNELS:
        metadata = exp.read(tmp_path / f'kernel-{group}' / 'kernel-metadata.json')
        assert metadata['competition_sources'] == []


def test_reused_base_file_requires_identical_bytes(tmp_path, monkeypatch):
    name = exp.REUSED_BASE[0]
    root = tmp_path / 'repo'
    path = root / '.local/exp179-overlay' / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b'pinned roster')
    monkeypatch.setattr(exp, 'SOURCE_MANIFEST', 'source.json')
    _json(root / 'source.json', {'files': {name: {
        'origin': 'overlay', 'sha256': exp.sha(path)}}})
    archive_path = tmp_path / 'base.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr(name, path.read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        exp._verify_reused_base(archive, root)
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr(name, b'different roster')
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match='reused input differs'):
        exp._verify_reused_base(archive, root)


def test_collect_requires_whole_verified_kernel(tmp_path, monkeypatch):
    root, source = _source(tmp_path, monkeypatch)
    result = exp.collect(source, '01', root)
    assert result['status'] == 'VERIFIED'
    for fold in exp.KERNELS['01']:
        dest = root / exp.EXP / 'outputs' / f'fold{fold}'
        assert (dest / 'results' / exp.tag(fold) / 'model_best.pt').is_file()
        assert exp.read(dest / 'collection.json')['status'] == 'VERIFIED'


def test_collect_rejects_incomplete_status_before_copy(tmp_path, monkeypatch):
    root, source = _source(tmp_path, monkeypatch)
    status = exp.read(source / 'exp187-status.json')
    status['completed'].pop('1')
    _json(source / 'exp187-status.json', status)
    with pytest.raises(ValueError, match='did not complete'):
        exp.collect(source, '01', root)
    assert not (root / exp.EXP / 'outputs').exists()


def test_collect_rejects_false_runner_receipt_before_copy(tmp_path, monkeypatch):
    root, source = _source(tmp_path, monkeypatch)
    receipt_path = source / 'fold1' / 'validated-fold1.json'
    receipt = exp.read(receipt_path)
    receipt['runner_sha256'] = 'wrong'
    _json(receipt_path, receipt)
    status = exp.read(source / 'exp187-status.json')
    status['completed']['1'] = receipt
    _json(source / 'exp187-status.json', status)
    with pytest.raises(ValueError, match='receipt or output differs'):
        exp.collect(source, '01', root)
    assert not (root / exp.EXP / 'outputs').exists()


def test_collect_rejects_changed_inventory_before_copy(tmp_path, monkeypatch):
    root, source = _source(tmp_path, monkeypatch)
    inventory = source / 'fold1' / 'results' / exp.tag(1) / 'output-inventory.json'
    inventory.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='receipt or output differs'):
        exp.collect(source, '01', root)
    assert not (root / exp.EXP / 'outputs').exists()


def test_prediction_rejects_wrong_roster_and_nonfinite(tmp_path):
    path = tmp_path / 'oof.csv'
    header = ','.join((exp.UID, *exp.TARGETS))
    values = ','.join('0.5' for _ in exp.TARGETS)
    path.write_text(f'{header}\nuid-a,{values}\n', encoding='utf-8')
    exp._prediction_values(path, ['uid-a'])
    with pytest.raises(ValueError, match='study IDs'):
        exp._prediction_values(path, ['uid-b'])
    path.write_text(f'{header}\nuid-a,nan,{",".join("0.5" for _ in exp.TARGETS[1:])}\n',
                    encoding='utf-8')
    with pytest.raises(ValueError, match='probability invalid'):
        exp._prediction_values(path, ['uid-a'])


def test_collection_uses_historical_source_and_accepts_measured_long_runtime(tmp_path, monkeypatch):
    stage, source = _source(tmp_path, monkeypatch)
    status = exp.read(source / 'exp187-status.json')
    status['total_seconds'] = exp.KERNEL_CAP * 3
    for fold in exp.KERNELS['01']:
        receipt = status['completed'][str(fold)]
        receipt['elapsed_seconds'] = exp.FOLD_CAP * 3
        _json(source / f'fold{fold}' / f'validated-fold{fold}.json', receipt)
    _json(source / 'exp187-status.json', status)
    destination = tmp_path / 'new-root'
    result = exp.collect(source, '01', destination, source_root=stage)
    assert result['status'] == 'VERIFIED'
    receipt = exp.read(destination / exp.EXP / 'outputs/fold0/collection.json')
    assert receipt['historical_source_root'] == str(stage.resolve())
