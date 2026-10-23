"""Synthetic-only EXP080 checks; never open current run results or Gold files."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import exp080_f2_report as report

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL'
METADATA = ROOT/'docs/handoffs/evidence/2026-10/2026-10-03-exp080-f2/metadata-preflight.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return report.primary.sha256_file(path)


@pytest.fixture
def manifest():
    # Registration is data, not a run result. No summary/OOF is opened here.
    return report.read(PROTOCOL/'manifest.json')


def summary_for(manifest, run):
    protocol = report.protocol_for_run(manifest, run)
    epochs = run['fixed_epoch']
    runtime = dict(requested_amp='fp16', effective_amp='fp16', device='cuda', policy='strict',
        precision_fallback_allowed=False, backends=dict(cudnn_benchmark=False,
        cudnn_deterministic=True, cudnn_allow_tf32=False, matmul_allow_tf32=False,
        float32_matmul_precision='highest', deterministic_algorithms=True,
        cublas_workspace_config=':4096:8'))
    p = dict(checkpoint_selection='fixed-final', code_sha256=deepcopy(manifest['training_code_sha256']),
        initial_state_sha256=run['initial_state_sha256'],
        corpus_manifest_sha256=manifest['corpus_manifest_sha256'],
        eval_label_hash=manifest['label_hash'], excluded_ids=sorted(manifest['excluded_uids']),
        sampling='sha256(seed,epoch,uid,stream)-v1', runtime=deepcopy(runtime), eval_runtime=deepcopy(runtime))
    p.update({k: v for k, v in manifest['expected_ids']['2'].items() if k.endswith('_sha256')})
    return dict(args={**protocol['expected_args'], 'tag': run['tag'], 'fold': 2}, provenance=p,
        no_oof_epoch_selection=True, no_gold_epoch_selection=True,
        selected_epoch_1based=epochs, selected_checkpoint='best', best_epoch=epochs-1,
        results={'best': {'gold_diagnostic': 'IGNORED'}}, label_hash=manifest['label_hash'],
        eval_label_hash=manifest['label_hash'], split_hash=manifest['split_hash'], max_epochs=epochs,
        window_policy=run['policy'], fold=2, backbone=protocol['expected_args']['arch'], img=224,
        train=3475, val=868, rows=868, optimizer_updates=100*epochs, skipped_updates=0,
        history=[dict(epoch=i, loss=.2, oof_auc=None, gold_auc=None,
                      optimizer_updates=(i+1)*100, skipped_updates=0) for i in range(epochs)],
        environment={k: 'synthetic-'+k for k in report.ENV_KEYS}, total_seconds=100.,
        gold_auc='DO NOT SELECT')


def test_registration_and_each_arm_adapter(manifest):
    report.validate_protocol(manifest)
    before = deepcopy(manifest)
    for run in manifest['runs']:
        protocol = report.protocol_for_run(manifest, run)
        assert report.validate_run(summary_for(manifest, run), protocol, run, .01) == 0
        assert protocol['epochs'] == (20 if run['arm'] == 'RN_B' else 8)
    assert manifest == before


@pytest.mark.parametrize('path,value', [
    (('folds',), [0, 1]), (('targets',), ['ACL']), (('eval_labels',), 'other.csv'),
    (('expected_ids', '2', 'val'), 869), (('expected_ids', '2', 'train'), 3474),
    (('runs', 2, 'fixed_epoch'), 8), (('runs', 1, 'arm'), 'A'),
    (('runs', 0, 'initial_state_sha256'), 'changed'),
    (('expected_args_by_arm', 'A', 'amp'), 'bf16'),
    (('expected_args_by_arm', 'RN_B', 'oof_subsample'), 100),
    (('evaluation', 'requires_all_three_completed_fits'), False),
    (('evaluation', 'fixed_global_rank_weights', 'candidate_RN_B'), .2),
    (('evaluation', 'ratio_epoch_target_search'), True),
    (('evaluation', 'major_drop_review'), -.002),
    (('evaluation', 'conditional_ci', 'reps'), 50),
    (('stop_review', 'optimizer_skip_fraction'), .02),
])
def test_changed_registration_rejected(manifest, path, value):
    cursor = manifest
    for k in path[:-1]: cursor = cursor[k]
    cursor[path[-1]] = value
    with pytest.raises(ValueError): report.validate_protocol(manifest)


@pytest.mark.parametrize('arm', range(3))
@pytest.mark.parametrize('path,value', [
    (('selected_epoch_1based',), 7), (('best_epoch',), 2),
    (('args', 'oof_subsample'), 200), (('provenance', 'initial_state_sha256'), 'changed'),
    (('provenance', 'code_sha256', 'raptor_data.py'), 'changed'),
    (('provenance', 'eval_ids_sha256'), 'changed'),
    (('provenance', 'sampling'), 'stateful'),
    (('provenance', 'runtime', 'device'), 'cpu'),
    (('provenance', 'runtime', 'backends', 'matmul_allow_tf32'), True),
    (('provenance', 'eval_runtime', 'effective_amp'), 'bf16'),
    (('provenance', 'eval_runtime', 'precision_fallback_allowed'), True),
    (('history', 0, 'oof_auc'), .9), (('history', 0, 'gold_auc'), .9),
    (('history', 0, 'loss'), float('nan')), (('optimizer_updates',), 0),
    (('history', -1, 'optimizer_updates'), 999999), (('val',), 869),
])
def test_bad_summary_rejected(manifest, arm, path, value):
    run = manifest['runs'][arm]; summary = summary_for(manifest, run)
    cursor = summary
    for k in path[:-1]: cursor = cursor[k]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        report.validate_run(summary, report.protocol_for_run(manifest, run), run, .01)


def test_uid_newline_serializations_are_distinct_and_order_independent():
    assert report.uid_digest(['b', 'a']) == hashlib.sha256(b'a\nb').hexdigest()
    assert report.uid_digest(['b', 'a'], True) == hashlib.sha256(b'a\nb\n').hexdigest()
    assert report.uid_digest(['a', 'b']) != report.uid_digest(['a', 'b'], True)


def columns(values):
    return np.tile(np.asarray(values, dtype=np.float32)[:, None], (1, 12))


def test_hand_calculated_ties_same_blend_three_contrasts_and_decomposition():
    y = columns([0, 0, 1, 1]); a = y.copy(); b = np.full(y.shape, .5)
    rn = columns([0, .5, .5, 1])
    rows, scores = report.paired_rows(y, a, b, rn)
    np.testing.assert_array_equal(scores['B'][:, 0], [.625]*4)
    np.testing.assert_allclose(scores['blend'][:, 0], [.5875, .625, .625, .6625], rtol=0, atol=1e-15)
    for r in rows:
        assert r['P'] == r['N'] == 2
        assert r['auc'] == {'A': 1, 'B': .5, 'RN_B': .875, 'blend': .875}
        assert r['delta'] == {'input_B_minus_A': -.5, 'complement_blend_minus_B': .375,
                              'descriptive_blend_minus_A': -.125}
        d = r['delta']
        assert d['descriptive_blend_minus_A'] == d['input_B_minus_A']+d['complement_blend_minus_B']
        assert r['macro_contribution']['complement_blend_minus_B'] == .375/12
    assert report.aggregate(rows)['complement_blend_minus_B'] == {'macro12': .375, 'PFOA': .375, 'other11': .375}


def test_float32_reference_rounding_and_one_class_macro_null():
    y = columns([0., .5, .500000000001, 1.]); a = columns([0, .2, .4, 1])
    y[:, 0] = 0
    rows, _ = report.paired_rows(y, a, a, a)
    assert rows[1]['P'] == 1 and rows[1]['N'] == 3
    assert rows[0]['auc']['A'] is None
    assert all(x['macro12'] is None for x in report.aggregate(rows).values())


def test_group_and_bootstrap_never_rerank(monkeypatch):
    y = columns([0, 1, 0, 1, 0, 1]); a = columns([.1, .7, .2, .3, .5, .6])
    b = columns([.2, .3, .4, .1, .6, .5]); rn = columns([.1, .8, .2, .7, .3, .6])
    _, scores = report.paired_rows(y, a, b, rn)
    def forbidden(*args, **kwargs): raise AssertionError('rerank')
    monkeypatch.setattr(pd.DataFrame, 'rank', forbidden)
    rows = report.rows_from_scores(y[[0, 1, 4, 5]], {k: v[[0, 1, 4, 5]] for k, v in scores.items()})
    assert rows[0]['P'] == rows[0]['N'] == 2
    groups = np.array(['x', 'x', 'y', 'y', 'z', 'z'])
    result = report.conditional_intervals(y, scores, groups, reps=31, seed=71)
    assert result == report.conditional_intervals(y, scores, groups, reps=31, seed=71)
    # Each group contains both classes, so every paired draw is valid.
    assert all(x['macro12']['valid_reps'] == 31 for x in result['contrasts'].values())


def test_one_class_bootstrap_returns_null_instead_of_partial_macro():
    y = columns([0, 0, 0]); a = columns([.2, .5, .8])
    _, scores = report.paired_rows(y, a, a, a)
    result = report.conditional_intervals(y, scores, np.array(['a', 'b', 'c']), reps=9)
    assert all(x['macro12'] == {'requested_reps': 9, 'valid_reps': 0, 'low': None, 'high': None}
               for x in result['contrasts'].values())


def test_paired_cluster_intervals_match_independent_repeated_row_auc():
    y = columns([0, 1, 0, 1, 0, 1])
    a = columns([.2, .4, .3, .7, .8, .6]); b = columns([.2, .6, .5, .4, .3, .8])
    rn = columns([.1, .7, .2, .9, .3, .7]); groups = np.array(['x', 'x', 'y', 'y', 'z', 'z'])
    _, scores = report.paired_rows(y, a, b, rn)
    actual = report.conditional_intervals(y, scores, groups, reps=50, seed=91)
    # Independent pair counting, with explicitly repeated rows rather than weights.
    def brute(labels, values):
        pos, neg = values[labels == 1], values[labels == 0]
        return float(np.mean((pos[:, None] > neg) + .5*(pos[:, None] == neg)))
    members = [np.flatnonzero(groups == name) for name in np.unique(groups)]
    rng = np.random.default_rng(91); expected = {k: [] for k in report.CONTRASTS}
    for _ in range(50):
        take = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        auc = {k: brute(y[take, 0], p[take, 0]) for k, p in scores.items()}
        for k, (left, right) in report.CONTRASTS.items(): expected[k].append(auc[right]-auc[left])
    for k, values in expected.items():
        found = actual['contrasts'][k]['macro12']
        assert found['valid_reps'] == 50
        assert found['low'] == pytest.approx(np.quantile(values, .025), abs=1e-14)
        assert found['high'] == pytest.approx(np.quantile(values, .975), abs=1e-14)


@pytest.mark.parametrize('fault', ['uid', 'duplicate', 'order', 'nan', 'inf', 'range', 'extra'])
def test_bad_predictions_rejected(tmp_path, fault):
    frame = pd.DataFrame(columns([.2, .8]), columns=report.LAB)
    frame.insert(0, report.UID, ['a', 'b'])
    if fault == 'uid': frame.loc[1, report.UID] = 'c'
    elif fault == 'duplicate': frame.loc[1, report.UID] = 'a'
    elif fault == 'order': frame = frame[[report.UID, *reversed(report.LAB)]]
    elif fault == 'nan': frame.loc[0, 'ACL'] = np.nan
    elif fault == 'inf': frame.loc[0, 'ACL'] = np.inf
    elif fault == 'range': frame.loc[0, 'ACL'] = 1.1
    else: frame['extra'] = 0
    path = tmp_path/'p.csv'; frame.to_csv(path, index=False)
    with pytest.raises(ValueError): report.predictions(path, ['a', 'b'])


@pytest.fixture
def synthetic(tmp_path, manifest):
    """Complete local dataset/receipts; model/training source files contain text only."""
    root = tmp_path; m = deepcopy(manifest)
    train = [f't{i:04d}' for i in range(3475)]; val = [f'v{i:04d}' for i in range(868)]
    ids = train+val; rng = np.random.default_rng(80)
    y = rng.integers(0, 2, size=(len(ids), 12)).astype(float)
    label_path = root/m['labels']; label_path.parent.mkdir(parents=True)
    frame = pd.DataFrame(y, columns=report.LAB); frame.insert(0, report.UID, ids)
    frame.to_csv(label_path, index=False)
    train_groups = [f'tg{i%3357:04d}' for i in range(len(train))]
    val_groups = [f'vg{i%840:04d}' for i in range(len(val))]
    split = pd.DataFrame({report.UID: ids, 'fold': [0]*len(train)+[2]*len(val),
        'report_group_sha256': train_groups+val_groups, 'is_gold': False,
        'quarantined_gold_report_duplicate': False})
    split_path = root/m['fold_file']; split_path.parent.mkdir(parents=True)
    split.to_csv(split_path, index=False)
    m['label_hash'] = report.primary.load_label_csv(label_path)[1]
    m['eval_label_hash'] = m['label_hash']; m['split_hash'] = report.primary.load_folds(split_path)[1]
    m['source_file_sha256'] = {m['labels']: report.primary.sha256_file(label_path),
                             m['fold_file']: report.primary.sha256_file(split_path)}
    corpus = root/m['corpus']; corpus.mkdir(parents=True)
    m['corpus_manifest_sha256'] = write(corpus/'manifest.json', {'synthetic': True})
    np.save(corpus/'ids.npy', np.array(ids)); np.save(corpus/'masks.npy', np.ones((len(ids), 3), np.uint8))
    meta_path = root/'storage/labels/derived_meta/study_meta.csv'; meta_path.parent.mkdir(parents=True)
    pd.DataFrame({report.UID: ids, 'scanner_group': ['g'+str(i%2) for i in range(len(ids))],
                  'field': '3T', 'manufacturer': 'synthetic'}).to_csv(meta_path, index=False)
    m['metadata_source_hashes'] = {report.uri(p, root): report.primary.sha256_file(p)
                                 for p in (meta_path, corpus/'ids.npy', corpus/'masks.npy')}
    for name in m['training_code_sha256']:
        m['training_code_sha256'][name] = write(root/'training/src'/name, {'never_import': name})
    m['expected_ids']['2'].update(train_ids_sha256=report.uid_digest(train), eval_ids_sha256=report.uid_digest(val))
    for ref in m['parents'].values():
        ref['sha256'] = write(report.unuri(ref['path'], root), {'synthetic_parent': ref['path']})
    for path in m['frozen_parent_result_sources']:
        m['frozen_parent_result_sources'][path] = write(report.unuri(path, root), {'synthetic_prior': path})
    parent_commands = {}
    for run in m['runs']:
        ref = run['copied_from']; parent_commands.setdefault(ref['path'], []).append(
            {'tag': ref['tag'], 'command': ['python', 'synthetic.py', '--tag', ref['tag'], '--fold', '0']})
    for path, commands in parent_commands.items():
        digest = write(report.unuri(path, root), commands)
        for run in m['runs']:
            if run['copied_from']['path'] == path: run['copied_from']['sha256'] = digest
    queue = []
    for i, run in enumerate(m['runs']):
        q = {'tag': run['tag'], 'command': ['python', 'synthetic.py', '--tag', run['tag'], '--fold', '2']}
        if i: q['after'] = m['runs'][i-1]['tag']
        queue.append(q)
        log = root/'artifacts/logs'/run['tag']; log.mkdir(parents=True)
        (log/'SUCCESS').write_text('exit 0 at 2026-10-03T20:00:00')
        write(log/'job.json', {'tag': run['tag'], 'command': q['command']})
        exp = root/'artifacts/experiments'/run['tag']; exp.mkdir(parents=True)
        write(exp/'summary.json', summary_for(m, run))
        pred = pd.DataFrame(rng.random((len(val), 12)), columns=report.LAB)
        pred.insert(0, report.UID, val); pred.to_csv(exp/'oof_best.csv', index=False)
    queue_path = report.unuri(m['queue']['path'], root); m['queue']['sha256'] = write(queue_path, queue)
    metadata = report.read(METADATA)
    metadata['sources'] = {'gold': {'path': 'workspace://storage/gold/MUST_NOT_OPEN.csv', 'sha256': 'unavailable'}}
    metadata['corpus_and_split'].update(f2_train_uid_hash_sorted_newline=report.uid_digest(train, True),
                                       f2_validation_uid_hash_sorted_newline=report.uid_digest(val, True))
    decision = {'manifest_sha256': m['parents']['exp079_manifest']['sha256'],
                'report_sha256': m['parents']['exp079_report']['sha256'],
                'report': m['parents']['exp079_report']['path'],
                'status': 'PROMISING_DEVELOPMENT_COMPLEMENT_RETAINED_NOT_DEPLOYED'}
    manifest_path = root/'manifest.json'; manifest_hash = write(manifest_path, m)
    metadata_path = root/'metadata.json'; metadata_hash = write(metadata_path, metadata)
    decision_path = root/'decision.json'; decision_hash = write(decision_path, decision)
    launch = report.read(PROTOCOL/'launch-receipt.json')
    launch.update(manifest_sha256=manifest_hash, queue_sha256=m['queue']['sha256'],
        exp079_decision_sha256=decision_hash, metadata_preflight_sha256=metadata_hash,
        training_source_sha256=m['training_code_sha256'], expected_ids=m['expected_ids']['2'],
        command=['python', 'tools/gpu_queue.py', str(queue_path)])
    launch_path = root/'launch.json'; launch_hash = write(launch_path, launch)
    args = [manifest_path, manifest_hash, launch_path, launch_hash, decision_path, metadata_path]
    return dict(root=root, manifest=m, queue=queue, args=args, ids=val, train=train, metadata=metadata)


def test_complete_synthetic_build_no_gold_and_no_overwrite(synthetic):
    s = synthetic; out = s['root']/'out'
    result = report.build(*s['args'], out, root=s['root'])
    assert result['studies'] == 868
    saved = report.read(out/'report.json')
    assert saved['conditional_intervals']['groups'] == 840
    assert len(saved['targets']) == 12 and len(saved['execution']) == 3
    assert all(k in saved['fracture']['delta'] for k in report.CONTRASTS)
    for row in saved['targets']:
        d = row['delta']
        assert d['descriptive_blend_minus_A'] == pytest.approx(d['input_B_minus_A']+d['complement_blend_minus_B'], abs=1e-15)
    assert (out/'analysis_plan.json').exists() and (out/'targets.csv').exists()
    with pytest.raises(FileExistsError): report.build(*s['args'], out, root=s['root'])


@pytest.mark.parametrize('fault', ['third_success', 'third_summary', 'third_oof', 'failed', 'wrong_job', 'bad_marker'])
def test_incomplete_or_changed_job_rejects_before_opening_any_summary(synthetic, monkeypatch, fault):
    s = synthetic; root = s['root']; tag = report.TAGS[2]
    paths = {'third_success': root/'artifacts/logs'/tag/'SUCCESS',
             'third_summary': root/'artifacts/experiments'/tag/'summary.json',
             'third_oof': root/'artifacts/experiments'/tag/'oof_best.csv'}
    if fault in paths: paths[fault].unlink()
    elif fault == 'failed': (root/'artifacts/logs'/tag/'FAILED').write_text('exit1')
    elif fault == 'wrong_job': write(root/'artifacts/logs'/tag/'job.json', {'tag': tag, 'command': ['changed']})
    else: (root/'artifacts/logs'/tag/'SUCCESS').write_text('exit 1')
    original = report.read
    def guarded(path):
        if Path(path).name == 'summary.json': raise AssertionError('Opened summary before completion gate')
        return original(path)
    monkeypatch.setattr(report, 'read', guarded)
    with pytest.raises(ValueError): report.load_inputs(*s['args'], root=root)


@pytest.mark.parametrize('index', [0, 2, 4, 5])
def test_changed_manifest_launch_decision_metadata_file_rejected(synthetic, index):
    s = synthetic; path = s['args'][index]
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='Frozen source changed'):
        report.load_inputs(*s['args'], root=s['root'])


@pytest.mark.parametrize('fault', ['source', 'ids', 'metadata_newline', 'cross_group', 'environment', 'third_uid'])
def test_changed_data_runtime_and_population_rejected(synthetic, fault):
    s = synthetic; root = s['root']; m = s['manifest']
    if fault == 'source': (root/'training/src/raptor_runtime.py').write_text('changed')
    elif fault == 'ids': np.save(root/m['corpus']/'ids.npy', np.array(['changed']))
    elif fault == 'metadata_newline':
        # Direct cohort validation isolates serialization from receipt file SHA.
        meta = deepcopy(s['metadata'])
        meta['corpus_and_split']['f2_validation_uid_hash_sorted_newline'] = report.uid_digest(s['ids'])
        split = pd.read_csv(root/m['fold_file'], dtype={report.UID: str}).set_index(report.UID)
        with pytest.raises(ValueError, match='final-newline'):
            report.validate_cohort(m, meta, split, {u: None for u in s['train']+s['ids']}, s['train']+s['ids'])
        return
    elif fault == 'cross_group':
        split = pd.read_csv(root/m['fold_file'], dtype={report.UID: str}).set_index(report.UID)
        split.loc[s['ids'][0], 'report_group_sha256'] = split.loc[s['train'][0], 'report_group_sha256']
        with pytest.raises(ValueError, match='cross-fold'):
            report.validate_cohort(m, s['metadata'], split, {u: None for u in s['train']+s['ids']}, s['train']+s['ids'])
        return
    elif fault == 'environment':
        path = root/'artifacts/experiments'/report.TAGS[2]/'summary.json'; summary = report.read(path)
        summary['environment']['cuda'] = 'changed'; write(path, summary)
    else:
        path = root/'artifacts/experiments'/report.TAGS[2]/'oof_best.csv'
        f = pd.read_csv(path); f.loc[0, report.UID] = 'wrong'; f.to_csv(path, index=False)
    with pytest.raises(ValueError): report.load_inputs(*s['args'], root=root)


def test_passive_import_does_not_import_torch_or_training_entrypoint():
    code = "import sys; from tools import exp080_f2_report; assert 'torch' not in sys.modules; assert 'raptor_train' not in sys.modules; print('PASSIVE')"
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'PASSIVE'
