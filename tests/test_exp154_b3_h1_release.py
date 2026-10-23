"""EXP154 release contracts, exercised with small CPU-only artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import exp154_b3_h1_release as m


def _fake_checkpoint_header(path, head_count=5):
    path = Path(path)
    p = m.read(path.parent / 'provenance.json')
    return {'arch': 'resnet50', 'res': 224, 'lab': m.e133.TARGETS,
            'contract': 'raptor64_336_c140_s02', 'norm': 'radimagenet',
            'window_policy': 'within-slot-valid', 'input_policy': 'center3',
            'tag': p['run'], 'fold': p['fold'], 'heads': list(m.HEADS),
            'label_set': 'h1', 'epoch': 19,
            'provenance': {'protocol_sha256': p['protocol_sha256'], 'seed': p['seed'],
                           'full_model_initial_state_sha256': m.INITIAL_SHA,
                           'heads': list(m.HEADS)},
            'model': {f'heads.{i}.norm.weight': b'tiny' for i in range(head_count)}}


@pytest.fixture(autouse=True)
def _no_torch_checkpoint_load(monkeypatch):
    monkeypatch.setattr(m, 'load_checkpoint_header', _fake_checkpoint_header)


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def _list(path):
    value = {'schema': 'exp154-checkpoint-list-v1', 'heads': list(m.HEADS),
             'entries': [{'seed': s, 'fold': f, 'tag': m.tag(s, f)}
                         for s in (42, 43) for f in range(5)]}
    _write(path, value)
    return value


def _fake_roster(root, *, duplicate_weights=False):
    """Construct ten complete receipts with two distinct registered protocols."""
    entries = _list(root / 'checkpoint-list.json')['entries']
    specs = {}
    for item in entries:
        name, fold = item['tag'], item['fold']
        specs[name] = {'steps': 17, 'heads': list(m.HEADS),
                       'initial_state_sha256': m.INITIAL_SHA,
                       'train_ids_sha256': f'train-{name}', 'eval_ids_sha256': f'eval-{name}',
                       'head_labels': [{'name': h, 'sha256': f'label-{h}'} for h in m.HEADS]}
    common = {'common_args': {'batch_size': 8}}
    source = {**common, 'runs': {k: v for k, v in specs.items() if k.startswith('EXP149')}}
    _write(root / m.PROTOCOL149, source)
    continuation = {**common, 'runs': {k: v for k, v in specs.items() if k.startswith('EXP153')},
                    'exp153': {'source_manifest_sha256': m.sha(root / m.PROTOCOL149),
                               'initial_state_file_sha256': 'initial-file-sha'}}
    _write(root / m.PROTOCOL153, continuation)
    for item in entries:
        name, seed, fold = item['tag'], item['seed'], item['fold']
        spec = specs[name]
        proto = m.PROTOCOL149 if name.startswith('EXP149') else m.PROTOCOL153
        p = {'run': name, 'seed': seed, 'fold': fold, 'heads': list(m.HEADS),
             'label_set': 'h1', 'arch': 'resnet50', 'norm': 'radimagenet',
             'input_policy': 'center3', 'k_eval_extra': 36, 'training_mode': 'full',
             'epochs': 20, 'checkpoint_selection': 'fixed-final', 'gold_access': 'disabled',
             'full_model_initial_state_sha256': m.INITIAL_SHA, 'fp16_skipped_updates': 0,
             'protocol_sha256': m.sha(root / proto), 'train_folds': [f for f in range(5) if f != fold],
             'eval_folds': [fold], 'steps': 17,
             'train_ids_ordered_sha256': spec['train_ids_sha256'],
             'eval_ids_ordered_sha256': spec['eval_ids_sha256'],
             'head_label_files': {h['name']: h['sha256'] for h in spec['head_labels']}}
        if name.startswith('EXP153'):
            p.update(protocol_path=m.PROTOCOL153, initial_state_file_sha256='initial-file-sha',
                     exp153_surface='local')
        history = [{'epoch': i, 'fp16_skips': 0, 'optimizer_updates': 17} for i in range(20)]
        s = {'run': name, 'fold': fold, 'mode': 'full', 'label_set': 'h1',
             'heads': list(m.HEADS), 'selected_checkpoint': 'fixed-final',
             'selected_epoch_1based': 20, 'optimizer_updates': 17, 'skipped_updates': 0,
             'gold': 0, 'gold_access': 'disabled', 'worker_cleanup': {'clean': True},
             'total_seconds': 1, 'provenance': p, 'history': history,
             'args': {'variant': 'AUG1', 'mode': 'full', 'seed': seed, 'fold': fold,
                      'tag': name, 'run': name, 'batch_size': 8}}
        run = root / 'artifacts/experiments' / name
        for filename, obj in (('summary.json', s), ('provenance.json', p), ('history.json', history)):
            _write(run / filename, obj)
        for filename in ('oof_best.csv', 'oof_k36.csv'):
            (run / filename).write_bytes(b'tiny')
        for head in m.HEADS:
            for suffix in ('', '_k36'):
                (run / f'oof_head_{head}{suffix}.csv').write_bytes(b'tiny')
        (run / 'model_best.pt').write_bytes(b'same' if duplicate_weights else name.encode())
        _write(run / 'output-inventory.json', {
            p.name: m.record(p) for p in run.iterdir() if p.name != 'output-inventory.json'})
    return entries


def _refresh(run, filename):
    inv = m.read(run / 'output-inventory.json')
    inv[filename] = m.record(run / filename)
    (run / 'output-inventory.json').write_text(json.dumps(inv), encoding='utf-8')


def test_registered_list_and_missing_checkpoint_fail_closed(tmp_path):
    assert m.checkpoint_list() == [
        {'seed': s, 'fold': f, 'tag': m.tag(s, f)} for s in (42, 43) for f in range(5)]
    listing = tmp_path / 'checkpoint-list.json'
    data = _list(listing)
    assert m.checkpoint_list(listing) == data['entries']
    data['heads'].reverse()
    _write(listing, data)
    with pytest.raises(ValueError, match='list/order/heads'):
        m.checkpoint_list(listing)
    _list(listing)
    out = tmp_path / 'registered.json'
    with pytest.raises(ValueError, match='Incomplete checkpoints'):
        m.register_checkpoints(out, tmp_path, listing)
    assert not out.exists()


def test_registration_binds_ten_distinct_receipts_and_detects_tampering(tmp_path):
    entries = _fake_roster(tmp_path)
    listing, output = tmp_path / 'checkpoint-list.json', tmp_path / 'registered.json'
    result = m.register_checkpoints(output, tmp_path, listing)
    assert result['checkpoints'] == 10
    assert [(e['seed'], e['fold'], e['tag']) for e in m.registered(output, tmp_path, listing)['entries']] == [
        (i['seed'], i['fold'], i['tag']) for i in entries]
    run = tmp_path / 'artifacts/experiments' / entries[0]['tag']
    (run / 'model_best.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='inventory SHA/bytes'):
        m.registered(output, tmp_path, listing)


def test_registration_rejects_changed_heads_protocol_and_missing_member(tmp_path):
    items = _fake_roster(tmp_path)
    run = tmp_path / 'artifacts/experiments' / items[0]['tag']
    p = m.read(run / 'provenance.json')
    p['heads'] = p['heads'][:-1]
    _write(run / 'provenance.json', p)
    _refresh(run, 'provenance.json')
    with pytest.raises(ValueError, match='provenance/history'):
        m.checkpoint_entry(tmp_path, items[0])
    p['heads'] = list(m.HEADS)
    _write(run / 'provenance.json', p)
    _refresh(run, 'provenance.json')
    protocol = m.read(tmp_path / m.PROTOCOL149)
    protocol['runs'][items[0]['tag']]['heads'] = ['wrong']
    _write(tmp_path / m.PROTOCOL149, protocol)
    with pytest.raises(ValueError, match='provenance|protocol'):
        m.checkpoint_entry(tmp_path, items[0])
    (run / 'oof_head_aquila_v1_k36.csv').unlink()
    with pytest.raises(FileNotFoundError):
        m.checkpoint_entry(tmp_path, items[0])


def test_registration_rejects_duplicate_checkpoints(tmp_path):
    _fake_roster(tmp_path, duplicate_weights=True)
    with pytest.raises(ValueError, match='Ten distinct'):
        m.register_checkpoints(tmp_path / 'registered.json', tmp_path,
                               tmp_path / 'checkpoint-list.json')


def test_registration_inspects_actual_checkpoint_head_indices(tmp_path, monkeypatch):
    items = _fake_roster(tmp_path)
    monkeypatch.setattr(m, 'load_checkpoint_header', lambda path: _fake_checkpoint_header(path, 4))
    with pytest.raises(ValueError, match='exactly five heads'):
        m.checkpoint_entry(tmp_path, items[0])


def test_five_head_reduction_matches_exact_cohort_rank_rule():
    import exp151_bundle_release as base
    scope = {}
    exec(base.components_source(), scope)
    a = np.zeros((10, 3, 5, 12), np.float32)
    b = np.zeros_like(a)
    a[:, :, 0, :] = np.array([0.1, 0.2, 0.9])[None, :, None]
    a[:, :, 1:, :] = np.array([0.9, 0.2, 0.1])[None, :, None, None]
    b[:] = a
    got = m.reduce_heads(a, b, scope['rank_mean'], scope['fold_ranks'])
    assert got.shape == (3, 12)
    assert got[:, 0] == pytest.approx([1, 2 / 3, 1 / 3])
    reversed_view = m.reduce_heads(a, b[:, ::-1].copy(), scope['rank_mean'], scope['fold_ranks'])
    assert reversed_view[:, 0] == pytest.approx([2 / 3, 2 / 3, 2 / 3])
    with pytest.raises(ValueError, match='shape'):
        m.reduce_heads(a[:, :, :4], b[:, :, :4], scope['rank_mean'], scope['fold_ranks'])
    a[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='probabilities'):
        m.reduce_heads(a, b, scope['rank_mean'], scope['fold_ranks'])


def test_blend_fixed_weights_average_ties_and_no_fallback():
    oai = [[0.1] * 12, [0.1] * 12, [0.9] * 12]
    cnx = [[0.9] * 12, [0.2] * 12, [0.2] * 12]
    h1 = [[0.3] * 12, [0.8] * 12, [0.1] * 12]
    result = m.blend(oai, cnx, h1)
    assert [r[0] for r in result] == pytest.approx([
        .8 * .5 + .1 * 1 + .1 * (2 / 3),
        .8 * .5 + .1 * .5 + .1 * 1,
        .8 * 1 + .1 * .5 + .1 * (1 / 3)])
    with pytest.raises(ValueError, match='All three'):
        m.blend(oai, cnx, None)
    with pytest.raises(ValueError, match='All three'):
        m.blend(oai, None, h1)


def test_archive_rejects_traversal_duplicates_and_plain_zip(tmp_path):
    path = tmp_path / 'code.zip.bin'
    m.pack_archive(path, {'code/a.py': b'print(1)\n'})
    m.verify_archive(path, {'code/a.py': {'sha256': hashlib.sha256(b'print(1)\n').hexdigest(), 'bytes': 9}})
    with pytest.raises(ValueError, match='Unsafe'):
        m.pack_archive(tmp_path / 'bad.zip.bin', {'../escape': b'x'})
    with pytest.raises(ValueError, match='Fresh'):
        m.pack_archive(tmp_path / 'plain.zip', {'a': b'x'})
    duplicate = tmp_path / 'dup.zip.bin'
    with zipfile.ZipFile(duplicate, 'w') as z:
        z.writestr('a', b'one')
        z.writestr('a', b'two')
    with pytest.raises(ValueError, match='duplicate'):
        m.verify_archive(duplicate, {'a': {'sha256': '0' * 64, 'bytes': 3}})


def _fake_base():
    cell = m.e133.e128.cell
    return {'nbformat': 4, 'cells': [cell('import time as _exp133_time\n_EXP133_START = _exp133_time.monotonic()\n'),
                                    cell('Old credit', 'markdown'),
                                    cell('OAI_FROZEN_SENTINEL = 1\n'),
                                    cell('OLD_PARTNER = True\n')]}


def test_notebook_preserves_public_oai_cell_and_compiles():
    base = _fake_base()
    nb = m.notebook(base, 'a' * 64)
    assert nb['cells'][3] == base['cells'][2]
    assert nb['cells'][3] is not base['cells'][2]
    assert "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'" in nb['cells'][1]['source']
    assert 'CUDA_VISIBLE_DEVICES' not in nb['cells'][1]['source']
    assert 'OAI-derived' in nb['cells'][2]['source']
    assert 'goodpjw2008' in nb['cells'][2]['source']
    assert 'knee-exp154-b3-h1-assets' in nb['cells'][1]['source']
    assert 'OLD_PARTNER' not in json.dumps(nb)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'notebook cell {i}', 'exec')


def test_worker_source_has_five_head_shapes_and_compiles():
    source = (ROOT / 'tools/exp151_bundle_release.py').read_text(encoding='utf-8')
    worker = m.worker_source(source)
    assert "build_multihead(build_model('resnet50', pretrained=False), 5)" in worker
    assert 'scores = np.empty((10, 2, len(ids), 5, 12)' in worker
    assert 'tuple(logits.shape) == (1, 5, 12)' in worker
    assert "exp154_h1_stage_timing.json" in worker
    assert "head-probabilities.npz" in worker
    assert 'check_checkpoint(ck, entry)' in worker
    compile(worker, 'exp154_h1_worker.py', 'exec')


def test_exp151_binding_and_child_cuda_environment(monkeypatch):
    assert m.VERSION == '1.0.1'
    assert m.SOURCE_BINDINGS['tools/exp151_bundle_release.py'] == m.sha(
        ROOT / 'tools/exp151_bundle_release.py')
    env = m.e133.child_env(ROOT / 'tools')
    assert env['CUDA_VISIBLE_DEVICES'] == '0'
    assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    monkeypatch.setattr(m, 'sha', lambda _: '0' * 64)
    with pytest.raises(ValueError, match='Frozen source SHA differs'):
        m.build_sources()


def test_run_after_oai_cpu_success_and_failed_partner_leave_no_submission(tmp_path):
    ids = ['a', 'b', 'c']
    sample = tmp_path / 'sample.csv'
    m.e133.write_csv(sample, ids, [[0] * 12] * 3)
    def start_case(name):
        work = tmp_path / name
        work.mkdir()
        m.e133.write_csv(work / 'submission.csv', ids, [[0.1] * 12, [0.2] * 12, [0.9] * 12])
        return work
    def h1(_manifest, _start, got_ids, rec, _working, _work):
        assert got_ids == ids
        rec['csv_sha256'] = 'fake'
        return [[0.3] * 12, [0.8] * 12, [0.1] * 12]
    def cnx(_payload, _start, got_ids, rec, _working):
        assert got_ids == ids
        rec['csv_sha256'] = 'fake'
        return [[0.9] * 12, [0.2] * 12, [0.2] * 12]
    good = start_case('good')
    receipt = m.run_after_oai(tmp_path, {}, 0, good, sample, 'a' * 64,
                              stages={'h1': h1, 'cnx': cnx}, clock=lambda: 1)
    assert receipt['status'] == 'PASS' and receipt['weights'] == m.WEIGHTS
    assert m.e133.validate_csv(good / 'submission.csv', sample)[1] == m.blend(
        m.e133.validate_csv(good / 'exp154_oai.csv', sample)[1], cnx(None, 0, ids, {}, good),
        h1({}, 0, ids, {}, good, tmp_path))
    bad = start_case('bad')
    def fail(*_args):
        raise RuntimeError('partner failed')
    with pytest.raises(RuntimeError, match='partner failed'):
        m.run_after_oai(tmp_path, {}, 0, bad, sample, 'a' * 64,
                        stages={'h1': fail, 'cnx': cnx}, clock=lambda: 1)
    assert not (bad / 'submission.csv').exists()
    assert m.read(bad / 'exp154_stage_timing.json')['status'] == 'FAILED'


def test_projection_is_bound_and_explicit_about_measurement():
    p = m.projection(1300)
    assert p['baseline_seconds'] == pytest.approx(3.2 * 3600)
    assert p['projected_seconds'] == pytest.approx(3.68 * 3600)
    assert p['budget_seconds'] == pytest.approx(3.68 * 3600 * 1.3 + 600)
    assert p['margin_seconds'] > 0
    assert 'Unmeasured' in p['uncertainty']
    with pytest.raises(ValueError, match='Positive'):
        m.projection(True)


def test_tiny_stage_and_public3_recompute_without_gpu_or_network(tmp_path, monkeypatch):
    import exp151_bundle_release as base
    _fake_roster(tmp_path)
    listing = tmp_path / 'checkpoint-list.json'
    registration = tmp_path / 'registered.json'
    m.register_checkpoints(registration, tmp_path, listing)
    source = base.components_source().encode()
    monkeypatch.setattr(m, 'build_sources', lambda root: ({'exp151_components.py': source}, _fake_base()))
    staged = tmp_path / 'stage'
    assert m.stage(registration, staged, tmp_path, listing)['status'] == 'STAGED_HASH_VERIFIED'
    manifest, stage_receipt = m.validate_stage(staged)
    assert manifest['weights'] == {'oai': .8, 'cnx': .1, 'h1': .1}
    assert manifest['oai_derived'] and not manifest['submission_authorized']
    assert 'OAI-derived' in m.read(staged / 'dataset/dataset-metadata.json')['description']
    metadata = m.read(staged / 'kernel/kernel-metadata.json')
    assert metadata['id'] == m.KERNEL and metadata['title'] == 'knee-exp154-b3-h1'
    assert metadata['dataset_sources'] == [m.e133.e128.DATASET, m.e133.e128.READER_DATASET, m.DATASET]
    push = m.push_plan(staged)
    assert push['status'] == 'PLAN_ONLY_NO_NETWORK'
    assert str(staged / 'dataset') in push['commands'][0]
    assert 'datasets create -p .' in push['commands'][1]
    assert any('kernels push -p .' in c for c in push['commands'])
    with zipfile.ZipFile(staged / 'dataset/checkpoints.zip.bin') as z:
        assert len(z.namelist()) == len(set(z.namelist())) == 10

    ids = ['a', 'b', 'c']
    sample = tmp_path / 'sample.csv'
    m.e133.write_csv(sample, ids, [[0] * 12] * 3)
    output = tmp_path / 'public3'
    (output / 'exp154_h1').mkdir(parents=True)
    scores = np.random.default_rng(4).uniform(.05, .95, (10, 2, 3, 5, 12)).astype(np.float32)
    scope = {}
    exec(compile(source, 'exp151_components.py', 'exec'), scope)
    h1 = m.reduce_heads(scores[:, 0], scores[:, 1], scope['rank_mean'], scope['fold_ranks']).tolist()
    pure = [[.1] * 12, [.2] * 12, [.9] * 12]
    cnx = [[.9] * 12, [.2] * 12, [.3] * 12]
    for name, values in (('exp154_oai.csv', pure), ('exp154_h1.csv', h1),
                         ('exp133_cnx.csv', cnx), ('exp154_h1/submission.csv', h1),
                         ('submission.csv', m.blend(pure, cnx, h1))):
        m.e133.write_csv(output / name, ids, values)
    np.savez(output / 'exp154_h1/head-probabilities.npz', scores=scores)
    worker = {'schema': 'exp154-h1-worker-v1', 'status': 'PASS', 'studies': 3,
              'models': 10, 'heads': list(m.HEADS), 'entries': manifest['entries'],
              'k_eval': 36, 'rule': m.RULE, 'training': 0, 'label_reads': 0,
              'manifest_sha256': m.object_sha(manifest),
              'submission_sha256': m.sha(output / 'exp154_h1/submission.csv'),
              'raw_scores_sha256': m.sha(output / 'exp154_h1/head-probabilities.npz')}
    _write(output / 'exp154_h1/exp154_h1_stage_timing.json', worker)
    receipt = {'schema': 'exp154-b3-h1-stages-v1', 'status': 'PASS', 'path': 'B3_H1',
               'weights': m.WEIGHTS, 'heads': list(m.HEADS), 'studies': 3,
               'manifest_sha256': stage_receipt['manifest_sha256'],
               'public_source_sha256': m.e133.e128.SOURCE_SHA,
               'oai_weight_sha256': m.e133.e128.WEIGHT_SHA, 'elapsed_seconds': 100,
               'final_sha256': m.sha(output / 'submission.csv'),
               'oai_sha256': m.sha(output / 'exp154_oai.csv'),
               'stages': {'h1': {'status': 'PASS', 'csv_sha256': m.sha(output / 'exp154_h1.csv'),
                                'worker': worker},
                          'cnx': {'status': 'PASS', 'csv_sha256': m.sha(output / 'exp133_cnx.csv'),
                                  'checkpoint_sha256': m.CNX_CHECKPOINTS, 'source_sha256': m.CNX_SOURCES}}}
    _write(output / 'exp154_stage_timing.json', receipt)
    assert m.validate_public3(output, sample, staged)['status'] == 'PASS_B3_H1_PUBLIC3'
    worker['raw_scores_sha256'] = '0' * 64
    receipt['stages']['h1']['worker'] = worker
    _write(output / 'exp154_h1/exp154_h1_stage_timing.json', worker)
    (output / 'exp154_stage_timing.json').write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError, match='Raw H1 scores SHA'):
        m.validate_public3(output, sample, staged)
