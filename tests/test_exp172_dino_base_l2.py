"""CPU contracts for the registered EXP172 L2 DINOv2-BASE run.
Changelog v1.2.0 (2026-10-12): isolate exact historical helper dependencies
from live refactored job helpers; retain original hashes and scientific checks.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path, PureWindowsPath
import shutil

import numpy as np
import pytest

from tools import exp172_dino_base_l2 as e


def test_five_fold_queue_and_registered_inputs():
    assert e.SCHEMA == 'exp172-dino-base-l2-v1'
    assert e.VERSION == '1.1.0'
    assert e.REMOTE_SOURCE == '/workspace/rsna-exp160-dino-partner-v1_1b-BASE'
    assert e.LABEL.endswith('/l2_dread75_aquila25_v1.csv')
    assert e.roster_path(0) == e.FOLD0_ROSTER
    assert all('EXP162_DINO_FOLDS/rosters/' in e.roster_path(f) for f in range(1, 5))
    assert [e.tag(f) for f in range(5)] == [
        f'EXP172_DINO_BASE_L2_S42_F{f}' for f in range(5)]
    jobs = e.queue()
    assert [job['tag'] for job in jobs] == [e.tag(f) for f in range(5)]
    assert [job['command'][-1] for job in jobs] == [str(f) for f in range(5)]
    assert all('exp172_dino_base_l2' in ' '.join(job['command']) for job in jobs)
    assert e.BUDGET['total_seconds'] >= 1.3 * 4261
    with pytest.raises(ValueError):
        e.tag(5)


def test_runner_guard_has_every_referenced_attribute():
    runner = e.runner_source()
    assert 'from tools import exp172_dino_base_l2 as guard' in runner
    names = {node.attr for node in ast.walk(ast.parse(runner))
             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
             and node.value.id == 'guard'}
    assert names
    assert not [name for name in names if not hasattr(e, name)]
    assert set(e.guard_attributes(runner)) == names
    with pytest.raises(ValueError, match='runner guard attributes missing'):
        e.guard_attributes('from tools import exp172_dino_base_l2 as guard\n'
                           'guard.this_attribute_must_not_exist\n')


@pytest.fixture
def historical_root(tmp_path):
    """Sparse real-data fixture; frozen registrations and live sources stay intact.

    Resolve historical Python bytes by the registered digest, including the
    pre-refactor backup. Never rebind a manifest to the new queue implementation.
    """
    root = tmp_path / 'historical-inputs'
    source = e.read(e.ROOT / e.SOURCE)
    protocol_path = 'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1/s42/manifest.json'
    parent = e.read(e.ROOT / e.folds.PARENT)
    protocol = e.read(e.ROOT / protocol_path)
    names = {e.SOURCE, e.SOURCE_RECEIPT, e.PROTOCOL, e.SELF, *e.EXTRA,
             e.FOLD0_ROSTER, e.folds.PARENT, parent['ordered_ids']['path'], protocol_path,
             '.local/exp160-collected-v1/BASE/provenance.json',
             '.local/exp160-collected-v1/BASE/summary.json',
             '.local/exp160-overlay-v1_1/BASE/' + e.base.RUNNER,
             'tools/exp160_dino_partner.py',
             *(e.roster_path(f) for f in range(1, 5)),
             *(v['ordered_ids']['path'] for v in protocol['folds'].values())}

    def copy(src, name):
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        assert e.sha(dest) == e.sha(src)

    for name in names:
        copy(e.ROOT / name, name)
    for name, entry in source['files'].items():
        if name.startswith('tools/') and name.endswith('.py'):
            candidates = [e.ROOT / prefix / name for prefix in (
                '.local/exp160-overlay-v1_1/BASE', '',
                '.local/backups/refactor-invocation11')]
            candidates.append(e.ROOT / '.local/hc06-illinois-pair-v2' / Path(name).name)
            src = next((p for p in candidates if p.is_file() and
                        e.sha(p) == entry['sha256']), None)
            assert src is not None, f'No authentic historical source for {name}'
            copy(src, '.local/exp160-overlay-v1_1/BASE/' + name)
    return root


@pytest.fixture
def prepared(tmp_path, historical_root):
    # Preparation and validation are CPU only; real frozen source metadata is
    # useful here because a fake could miss a transitive module or guard import.
    overlay = tmp_path / 'overlay'
    result = e.prepare(overlay, root=historical_root)
    return overlay, result


def test_prepare_and_validate_five_fold_overlay(prepared):
    overlay, result = prepared
    manifest = e.validate_manifest(overlay / 'manifest.json', overlay)
    assert manifest['version'] == e.VERSION
    assert result['status'] == 'PREPARED_NOT_LAUNCHED'
    assert manifest['serial_folds'] == list(range(5))
    assert manifest['files'][e.LABEL]['sha256'] == e.LABEL_SHA
    assert {'config/workspace.toml', 'tools/exp160_dino_partner.py',
            'training/src/raptor_radimagenet.py'} <= set(manifest['files'])
    assert (overlay / 'config/workspace.toml').is_file()
    for fold in range(5):
        protocol = json.loads((overlay / f'protocols/fold{fold}.json').read_text())
        assert protocol['fold'] == fold
        assert protocol['expected_args']['fold'] == (2 if fold == 0 else fold)
        assert protocol['expected_args']['seed'] == 42
        assert protocol['expected_args']['epochs'] == 20
        assert protocol['paths']['labels'] == e.LABEL
        assert protocol['paths']['eval_labels'] == e.LABEL
        assert protocol['ordered_ids']['path'] == e.roster_path(fold)
        assert protocol['arm']['actual_pos_weight'] is None
        ids = json.loads((overlay / e.roster_path(fold)).read_text())
        assert len(ids['train']) + len(ids['val']) == 4343
        if fold:
            assert ids['train'] == sorted(ids['train'])
            assert ids['val'] == sorted(ids['val'])


def test_local_overlay_hashes_detect_tampering(prepared):
    overlay, _ = prepared
    for name in ('config/workspace.toml', e.LABEL, e.roster_path(2)):
        target = overlay / name
        content = target.read_bytes()
        target.write_bytes(content + b'changed')
        with pytest.raises(ValueError, match='package file differs'):
            e.validate_manifest(overlay / 'manifest.json', overlay)
        target.write_bytes(content)


def test_running_v100_overlay_remains_valid():
    # Verify the actual frozen package, including its original tool hashes.
    # The collector hotfix must not require rebuilding the running Vast job.
    overlay = e.ROOT / '.local/exp172-overlay'
    before = e.sha(overlay / 'manifest.json')
    manifest = e.validate_manifest(overlay / 'manifest.json', overlay)
    assert manifest['version'] == '1.0.0'
    assert manifest['files'][e.SELF]['sha256'] == e.sha(overlay / e.SELF)
    assert e.sha(overlay / 'manifest.json') == before


def test_unknown_manifest_version_still_rejected(prepared):
    overlay, _ = prepared
    path = overlay / 'manifest.json'
    manifest = e.read(path)
    manifest['version'] = '99.0.0'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='EXP172 manifest contract differs'):
        e.validate_manifest(path, overlay)


def test_v101_manifest_remains_valid_without_rebinding(tmp_path, monkeypatch, historical_root):
    overlay = tmp_path / 'v101-overlay'
    with monkeypatch.context() as patch:
        patch.setattr(e, 'VERSION', '1.0.1')
        e.prepare(overlay, root=historical_root)
    before = e.sha(overlay / 'manifest.json')
    manifest = e.validate_manifest(overlay / 'manifest.json', overlay)
    assert manifest['version'] == '1.0.1'
    assert manifest['files'][e.SELF]['sha256'] == e.sha(overlay / e.SELF)
    assert e.sha(overlay / 'manifest.json') == before


@pytest.fixture
def windows_fold_receipts(monkeypatch):
    # Model the Windows collector's local filesystem on every test host. The
    # synthetic receipts retain the Linux paths actually emitted by Vast.
    run = PureWindowsPath(r'H:\exp172\collected\fold0')
    package = PureWindowsPath(r'H:\exp172\overlay')
    protocol_sha = 'a' * 64
    updates = 8620
    protocol = {
        'expected_updates': updates,
        'files': {e.SELF: {'sha256': 'b' * 64}},
        'initial_asset': {'sha256': 'c' * 64, 'revision': 'frozen-base'},
        'label_hash': 'd' * 64,
        'ordered_ids': {'path': 'rosters/fold0.json',
                        'train_sha256': 'e' * 64, 'val_sha256': 'f' * 64},
    }
    provenance = {
        'gpu_family': 'RTX3090', 'contract_helper_sha256': 'b' * 64,
        'initial_asset_file_sha256': 'c' * 64,
        'initial_asset_revision': 'frozen-base',
        'aggregation_policy': e.AGGREGATION_POLICY,
        'label_file_sha256': e.LABEL_SHA, 'label_semantic_sha256': 'd' * 64,
        'runtime': {'effective_amp': 'fp16'},
        'train_ids_ordered_sha256': 'e' * 64, 'eval_ids_ordered_sha256': 'f' * 64,
    }
    linux_label = e.REMOTE_ROOT + '/' + e.LABEL
    summary = {
        'fold': 2, 'args': {'seed': 42, 'tag': e.tag(0),
                           'labels': linux_label, 'eval_labels': linux_label},
        'selected_epoch_1based': 20, 'optimizer_updates': updates,
        'skipped_updates': 0, 'gold_access': 'disabled', 'weak_oof_macro_auc': None,
        'train': 4343-e.COUNTS[0], 'val': e.COUNTS[0], 'provenance': provenance,
    }
    receipt = {
        'schema': e.SCHEMA, 'status': 'PASS', 'phase': 'full', 'fold': 0,
        'protocol_sha256': protocol_sha, 'updates': updates, 'skips': 0,
        'validation_rows': e.COUNTS[0],
        'outputs': {n: {'sha256': '0' * 64, 'bytes': 1, 'payload': True}
                    for n in e.old.OUTPUTS},
    }
    history = [{'epoch': i, 'optimizer_updates': (i+1)*(updates//20),
                'skipped_updates': 0, 'oof_auc': None, 'gold_auc': None}
               for i in range(20)]
    documents = {run / name: value for name, value in (
        ('receipt.json', receipt), ('summary.json', summary),
        ('provenance.json', provenance), ('history.json', history))}
    documents[package / protocol['ordered_ids']['path']] = {'val': ['synthetic-study']}
    entries = {run / n: entry for n, entry in receipt['outputs'].items()}
    predictions_checked = []
    monkeypatch.setattr(e, 'Path', PureWindowsPath)
    monkeypatch.setattr(e, 'read', documents.__getitem__)
    monkeypatch.setattr(e, '_entry', entries.__getitem__)
    monkeypatch.setattr(e.folds, '_prediction_values',
                        lambda path, ids: predictions_checked.append((path, ids)))
    return (str(run), 0, protocol, protocol_sha, str(package)), summary, receipt, predictions_checked


def test_verify_linux_receipts_from_windows_host_path(windows_fold_receipts):
    args, summary, receipt, checked = windows_fold_receipts
    assert '\\' in args[0] and '\\' in args[4]
    assert str(PureWindowsPath(e.REMOTE_ROOT) / e.LABEL) != summary['args']['labels']
    assert e._verify_fold(*args) is receipt
    assert checked == [(PureWindowsPath(args[0]) / 'oof_best.csv', ['synthetic-study'])]


@pytest.mark.parametrize('field', ['labels', 'eval_labels'])
@pytest.mark.parametrize('bad_path', [
    '/workspace/wrong-run/' + e.LABEL,
    str(PureWindowsPath(e.REMOTE_ROOT) / e.LABEL),
])
def test_verify_rejects_wrong_remote_label_paths(windows_fold_receipts, field, bad_path):
    args, summary, _, checked = windows_fold_receipts
    summary['args'][field] = bad_path
    with pytest.raises(ValueError, match='fold completion differs'):
        e._verify_fold(*args)
    assert not checked


def test_vast_plan_is_one_detached_serial_queue():
    plan = e.vast_plan('/workspace/exp172-overlay')
    assert 'setsid nohup python tools/gpu_queue.py queue.json --foreground' in plan
    assert 'serial_folds=[0,1,2,3,4]' in plan
    assert e.REMOTE_ROOT in plan
    assert 'output-inventory.json' not in plan
    assert '\\' not in plan
    with pytest.raises(ValueError):
        e.vast_plan(r'H:\bad')


def test_missing_fold_exports_stay_incomplete(prepared, tmp_path):
    overlay, _ = prepared
    exports = tmp_path / 'exports'
    exports.mkdir()
    output = tmp_path / 'collected'
    result = e.collect(exports, overlay, output)
    assert result['status'] == 'INCOMPLETE'
    assert result['completed_folds'] == []
    assert set(result['incomplete_folds']) == {str(f) for f in range(5)}
    screened = e.screen(output, tmp_path / 'screen')
    assert screened['verdict'] == 'INCOMPLETE'


def test_successful_screen_reports_every_standalone_family(tmp_path, monkeypatch):
    """Synthetic five-fold OOF; real CSV/hash/rank/AUC/bootstrap/report paths.

    Only collection admission, roster and reference acquisition are replaced;
    this is a reporting regression, not certification of real training receipts.
    """
    from tools import exp115_vlm_eval as ev
    from tools import exp149_label_sources as e149
    from tools import exp156_k_sweep as e156
    from tools import exp169_bundle_screen as bundle

    root, outputs, overlay = tmp_path, tmp_path / 'outputs', tmp_path / 'overlay'
    ids_by_fold = {f: [f'synthetic-{f}-{i}' for i in range(e.COUNTS[f])]
                   for f in range(5)}
    ids = [uid for group in ids_by_fold.values() for uid in group]
    fold_of = {uid: f for f, group in ids_by_fold.items() for uid in group}
    manifest = {'fold_protocols': {str(f): {'path': f'fold{f}.json',
                                         'sha256': str(f) * 64} for f in range(5)}}
    e.write_new(outputs / 'manifest.json', manifest)
    e.write_new(outputs / 'collection.json', {
        'schema': e.SCHEMA, 'status': 'VERIFIED', 'completed_folds': list(range(5)),
        'manifest_sha256': e.sha(outputs / 'manifest.json'), 'overlay': str(overlay)})
    for f in range(5):
        e.write_new(overlay / f'fold{f}.json', {'fold': f})
    monkeypatch.setattr(e, 'validate_manifest', lambda *a: manifest)
    checked = []
    monkeypatch.setattr(e, '_verify_fold', lambda run, f, *a: checked.append(f))
    monkeypatch.setattr(e149, '_load_manifest', lambda *a: {})
    monkeypatch.setattr(e149, 'fold_rosters', lambda *a: (ids, fold_of))

    rng = np.random.default_rng(172)
    refs = {}
    for name in ('jev_hybrid', 'aquila_v1', 'dread', 'consensus'):
        y = rng.integers(0, 2, size=(len(ids), 12))
        mask = rng.random(y.shape) > 0.15
        if name == 'aquila_v1':
            mask[:, -1] = False  # Undefined report-only target must stay null.
        refs[name] = (y, mask, 3)
    monkeypatch.setattr(e156, 'references', lambda actual, r: refs if actual == ids else {})
    for name in ('JEV', 'AQUILA', 'DREAD'):
        monkeypatch.setattr(e156, name, f'references/{name}.json')
    monkeypatch.setattr(e156, 'CONSENSUS', ['references/consensus.json'])
    for name in (e.PROTOCOL, e156.JEV, e156.AQUILA, e156.DREAD, *e156.CONSENSUS):
        e.write_new(root / name, {'synthetic': True})
    # Real component resampling, including non-singleton groups.
    monkeypatch.setattr(ev, 'COMPONENTS', 'components.json')
    e.write_new(root / ev.COMPONENTS, [
        {'component_id': str(i), 'members': ids[i:i+2]} for i in range(0, len(ids), 2)])

    source = {name: [] for name in ('l2_s42', 'l2_s43', 'dino_l2', 'dino', 'convnext')}
    hashes = {}
    for f, fid in ids_by_fold.items():
        paths = bundle.paths(root, f)
        paths['dino_l2'] = outputs / 'artifacts/experiments' / e.tag(f) / 'oof_best.csv'
        paths['convnext'] = (root / 'artifacts/experiments/EXP170_CONVNEXT_T_L2/outputs/results'
                            / f'EXP170_CONVNEXT_T_L2_S42_F{f}' / 'oof_k36_flip_tta.csv')
        for name in source:
            # Quantized, distinct families exercise ties and wrong-family errors.
            values = rng.integers(0, 33, size=(len(fid), 12)) / 32.0
            source[name].append(values)
            path = paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow([e.UID, *e.TARGETS])
                for i in rng.permutation(len(fid)):
                    writer.writerow([fid[i], *values[i]])
            digest = e.sha(path)
            hashes[f'{name}_f{f}'] = digest
            e.write_new(path.parent / 'output-inventory.json', {
                path.name: {'sha256': digest, 'bytes': path.stat().st_size}})
    e.write_new(root / 'artifacts/experiments/EXP170_CONVNEXT_T_L2/screen/result.json',
                {'status': {'status': 'ADMIT'}, 'input_sha256': hashes})

    # Observe the actual arrays scored as well as the serialized metrics.
    score = e.folds._score
    scored = []
    def record_score(values, ref, evaluator):
        scored.append((values.copy(), ref))
        return score(values, ref, evaluator)
    monkeypatch.setattr(e.folds, '_score', record_score)
    destination = root / 'screen'
    response = e.screen(outputs, destination, root)
    result = e.read(destination / 'result.json')
    assert checked == list(range(5))
    assert response['status'] == result['status']['status']
    assert result['bootstrap'] == {'unit': 'validation component', 'draws': 300,
                                    'seed': 20261003}
    assert result['input_sha256']['convnext_f4'] == hashes['convnext_f4']
    standalone = result['standalone']
    assert set(standalone) == {'dino_l2', 'dino_dread', 'l2', 'convnext',
                               'old_bundle', 'new_bundle'}

    # Independent average-tie rank oracle (not the production rank helpers).
    def ranks(values):
        columns = []
        for column in values.T:
            _, inv, counts = np.unique(column, return_inverse=True, return_counts=True)
            columns.append((np.cumsum(counts) - (counts-1)/2)[inv])
        return np.column_stack(columns)
    def global_ranks(values):
        return (ranks(values) - 1) / (len(values) - 1)
    pooled = {name: np.concatenate([ranks(x)/len(x) for x in parts])
              for name, parts in source.items()}
    expected = {name: global_ranks(pooled[key]) for name, key in
                (('dino_l2', 'dino_l2'), ('dino_dread', 'dino'), ('convnext', 'convnext'))}
    expected['l2'] = global_ranks((pooled['l2_s42'] + pooled['l2_s43']) / 2)
    expected['old_bundle'] = global_ranks(
        (expected['l2'] + expected['dino_dread'] + expected['convnext']) / 3)
    expected['new_bundle'] = global_ranks(
        (expected['l2'] + expected['dino_l2'] + expected['convnext']) / 3)
    references = dict(zip(('jev', 'aquila', 'dread', 'consensus'), refs.values()))
    for (family, reports), calls in zip(standalone.items(),
                                       np.arange(len(scored)-24, len(scored)).reshape(6, 4)):
        assert set(reports) == set(references)
        for (name, ref), call in zip(references.items(), calls):
            np.testing.assert_array_equal(scored[call][0], expected[family])
            assert scored[call][1] is ref
            # Independently calculate masked AUC using the tie-rank oracle.
            y, mask, _ = ref
            aucs = []
            for t, target in enumerate(e.TARGETS):
                yy, pp = y[mask[:, t], t], expected[family][mask[:, t], t]
                positives, negatives = yy.sum(), len(yy)-yy.sum()
                if positives < 3 or negatives < 3:
                    assert reports[name]['per_target'][target] is None
                    continue
                rank = ranks(pp[:, None])[:, 0]
                auc = (rank[yy == 1].sum() - positives*(positives+1)/2) / (positives*negatives)
                assert reports[name]['per_target'][target] == pytest.approx(auc)
                aucs.append(auc)
            assert reports[name]['macro'] == pytest.approx(np.mean(aucs))
    assert standalone['convnext']['jev']['macro'] != standalone['l2']['jev']['macro']


def test_paired_primary_uses_jev_and_consensus_and_aquila_is_report_only():
    class Evaluator:
        @staticmethod
        def auc_targets(y, pred, mask, minimum):
            multiplier = {0: 1.0, 1: -5.0, 2: -0.02, 3: 2.0}[int(y[0])]
            return np.mean(pred, axis=0) * multiplier

        @staticmethod
        def macro(values):
            return float(np.mean(values))

    candidate = np.full((4, 12), 0.6)
    baseline = np.full((4, 12), 0.5)
    refs = {name: (np.full(4, marker), np.ones(4), 1) for name, marker in
            (('jev', 0), ('aquila', 1), ('dread', 2), ('consensus', 3))}
    draws = [np.arange(4)] * 300
    pair = e.paired_jev_consensus(candidate, baseline, refs, draws, Evaluator)
    assert pair['primary_mean_delta'] == pytest.approx(0.15)
    assert pair['references']['aquila']['delta'] == pytest.approx(-0.5)
    assert e.admission(pair)['status'] == 'ADMIT'
    pair['primary_mean_ci95'][0] = 0
    assert e.admission(pair)['status'] == 'HOLD'
    pair['primary_mean_ci95'][0] = 0.01
    pair['references']['dread']['delta'] = -0.003
    assert e.admission(pair)['status'] == 'ADMIT'
    pair['references']['dread']['delta'] = -0.003001
    assert e.admission(pair)['status'] == 'HOLD'
