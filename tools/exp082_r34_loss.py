"""EXP082 CPU registration and contract helpers. Importing this module does no I/O.

The frozen trainer is read as text only. A separate static entrypoint is rendered
with a bounded, reviewable diff; neither preparation nor tests import that trainer.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/manifest.json'
PROTOCOL = 'artifacts/experiments/EXP082_R34_LOSS_F2_PROTOCOL'
ENTRYPOINT = 'training/runners/exp082_r34_loss_train.py'
TAG = 'EXP082_R34_ONES_F2'
SCHEMA = 'exp082-r34-loss-f2-v1'
CODE_NAMES = ('raptor_train.py', 'raptor_common.py', 'raptor_data.py', 'raptor_runtime.py')
ENV_KEYS = ('torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'pydicom', 'cv2')
UID = 'StudyInstanceUID'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def uid_hash(ids, sort=False):
    return hashlib.sha256('\n'.join(sorted(ids) if sort else ids).encode()).hexdigest()


def contained(root, relative):
    p = Path(relative)
    if p.is_absolute() or p.drive or '..' in p.parts:
        raise ValueError('Expected contained relative path')
    return Path(root) / p


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def reconstruct_ids(corpus_ids, folds, excluded, expected):
    """Gold-free membership from frozen nonlabel folds; preserve corpus order."""
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError('Duplicate corpus UID')
    pool = [u for u in corpus_ids if folds.get(u, -1) >= 0 and u not in excluded]
    train = [u for u in pool if folds[u] != 2]
    evaluation = [u for u in pool if folds[u] == 2]
    actual = dict(train=len(train), val=len(evaluation), train_ids_sha256=uid_hash(train, True),
                  eval_ids_sha256=uid_hash(evaluation, True))
    if actual != expected:
        raise ValueError('Gold-free cohort differs from frozen baseline membership')
    return train, evaluation


def class_weights(labels, train_ids):
    import numpy as np
    # Exact original float32/order/reduction/clipping sequence.
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    baseline = np.clip((1 - prev) / prev, 1, 10).astype(np.float32)
    if baseline.shape != (12,) or not np.isfinite(baseline).all():
        raise ValueError('Invalid baseline class weights')
    return baseline, np.ones(12, dtype=np.float32)


def read_training_labels(path, train_ids, targets):
    """Read only weak train-label values; no official Gold file or scoring."""
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(path, dtype={UID: str})
    # Match the frozen reader's dropna and float32 conversion, then keep train.
    frame = frame.dropna(subset=targets)
    frame = frame.loc[frame[UID].isin(set(train_ids)), [UID, *targets]]
    if frame[UID].duplicated().any() or set(frame[UID]) != set(train_ids):
        raise ValueError('Incomplete or duplicate training labels')
    values = frame[targets].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError('Invalid training labels')
    return dict(zip(frame[UID], values))


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Frozen source transformation no longer unique: ' + old[:80])
    return source.replace(old, new, 1)


def render_training_source(source):
    """Return a static no-Gold trainer. Never eval/exec/import frozen source."""
    start = source.index('"""')
    end = source.index('"""', start + 3) + 3
    source = source[:start] + '"""EXP082 fixed RN20/ones entrypoint; generated explicit diff, no Gold access."""' + source[end:]
    old = '''sys.path.insert(0, str(Path(__file__).resolve().parent))
from raptor_common import (CACHE, EXPERIMENTS, LAB, ROOT, Corpus, build_model, environment, eval_centers,  # noqa: E402
                           gold_labels, load_folds, load_label_csv, macro_auc, seed_all, sha256_text,
                           train_centers, windows_from, write_predictions, sha256_file)
from raptor_data import StudyWindows, collate
from raptor_runtime import configure_runtime'''
    source = replace_once(source, old, '''ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "storage" / "cache" / "derived"
EXPERIMENTS = ROOT / "artifacts" / "experiments"''')
    source = replace_once(source, '    ap.add_argument("--tag", required=True)',
        '    ap.add_argument("--protocol", required=True)\n    ap.add_argument("--tag", required=True)')
    source = replace_once(source, '    a = ap.parse_args()', '''    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools import exp082_r34_loss as guard
    protocol, ordered = guard.validate_arguments(a, ROOT)
    guard.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from raptor_common import (LAB, Corpus, build_model, environment, load_folds,
                               load_label_csv, macro_auc, seed_all, sha256_text,
                               write_predictions, sha256_file)
    from raptor_data import StudyWindows, collate
    from raptor_runtime import configure_runtime''')
    source = replace_once(source, '    seed_all(a.seed)',
        '    run_environment = environment()  # Audit imports happen before the original seed boundary.\n    seed_all(a.seed)')
    begin = source.index('    gold = gold_labels()')
    finish = source.index('    log(f"arch ', begin)
    source = source[:begin] + '''    folds, split_hash = load_folds(Path(a.fold_file))
    excluded = set(protocol["excluded_uids"])
    train_ids, oof_ids = guard.reconstruct_ids(corpus.ids, folds, excluded, protocol["expected_ids"])
    guard.validate_ordered_ids(train_ids, oof_ids, ordered)
    if any(u not in labels for u in train_ids) or any(u not in eval_labels for u in oof_ids):
        raise SystemExit("Frozen train/eval label coverage changed")
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: eval_labels[u] for u in oof_ids}}
    guard.validate_label_contract(label_hash, eval_label_hash, split_hash, protocol)
    log(f"{a.tag} | corpus {corpus.contract} ({len(corpus)}) | train {len(train_ids)} | oof {len(oof_ids)} (fold {a.fold}/{a.folds}) | Gold disabled")
''' + source[finish:]
    source = replace_once(source,
        '    pos_weight = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)',
        '''    baseline_pos_weight = np.clip((1 - prev) / prev, 1, 10).astype(np.float32)
    guard.validate_loss_vectors(baseline_pos_weight, protocol)
    pos_weight = torch.ones(len(LAB), dtype=torch.float32, device=dev)''')
    source = replace_once(source,
        '    source_files = [Path(__file__), Path(__file__).with_name("raptor_common.py"), Path(__file__).with_name("raptor_data.py"), Path(__file__).with_name("raptor_runtime.py")]',
        '    source_files = [ROOT / "training" / "src" / name for name in guard.CODE_NAMES]')
    source = replace_once(source,
        '    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")',
        '''    provenance.update(adapter_sha256=sha256_file(Path(__file__)),
                      contract_helper_sha256=sha256_file(Path(guard.__file__)),
                      protocol_sha256=sha256_file(Path(a.protocol)),
                      train_ids_ordered_sha256=guard.uid_hash(train_ids),
                      eval_ids_ordered_sha256=guard.uid_hash(oof_ids),
                      gold_access="disabled", loss_policy="ones_positive_term",
                      baseline_pos_weight=baseline_pos_weight.tolist(),
                      actual_pos_weight=pos_weight.detach().cpu().tolist())
    guard.validate_pretraining(provenance, run_environment, protocol)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")''')
    source = replace_once(source, '        gold_auc = None  # Gold58 is evaluated only after the checkpoint is fixed.',
        '        gold_auc = None  # Compatibility placeholder; Gold is never loaded or evaluated.')
    source = replace_once(source, '''        if gold_ids:
            m, per, P, U = evaluate(gold_ids); write_predictions(out / f"gold_{name}.csv", U, P)
            r.update(gold_auc=m, gold_per_target=per)
''', '')
    source = replace_once(source,
        '''        log(f"{name}: OOF {r.get('oof_auc', float('nan')):.4f} | gold {r.get('gold_auc', float('nan')):.4f}")''',
        '''        log(f"{name}: OOF {r.get('oof_auc', float('nan')):.4f} | Gold disabled")''')
    source = replace_once(source, '"gold": len(gold_ids)', '"gold": 0')
    source = replace_once(source,
        '               "gold_once_after_weak_selected_epoch": results[chosen].get("gold_auc"), "gold_per_target": results[chosen].get("gold_per_target"),',
        '               "gold_access": "disabled", "loss_policy": "ones_positive_term",')
    source = replace_once(source,
        '"caveat": "weak-label OOF; Gold58 is an exposed diagnostic set"',
        '"caveat": "exposed F2 weak development; known F0/F1 duplicate both in train; not independent/private evidence"')
    return source


def validate_ordered_ids(train, evaluation, ordered):
    if train != ordered['train'] or evaluation != ordered['eval']:
        raise ValueError('Frozen ordered UID sequence changed')


def validate_label_contract(label_hash, eval_hash, split_hash, protocol):
    if (label_hash != protocol['label_hash'] or eval_hash != protocol['eval_label_hash']
            or split_hash != protocol['split_hash']):
        raise ValueError('Canonical label/split hash changed')


def validate_loss_vectors(baseline, protocol):
    import numpy as np
    if (np.asarray(baseline).dtype != np.float32 or np.asarray(baseline).shape != (12,)
            or not np.array_equal(baseline, np.asarray(protocol['loss']['baseline_pos_weight'], dtype=np.float32))
            or protocol['loss']['actual_pos_weight'] != [1.] * 12):
        raise ValueError('Loss vector differs from registration')


def validate_pretraining(provenance, env, protocol):
    expected = dict(code_sha256=protocol['training_code_sha256'],
                    initial_state_sha256=protocol['initial_state_sha256'],
                    train_ids_sha256=protocol['expected_ids']['train_ids_sha256'],
                    eval_ids_sha256=protocol['expected_ids']['eval_ids_sha256'],
                    train_ids_ordered_sha256=protocol['ordered_ids']['train_sha256'],
                    eval_ids_ordered_sha256=protocol['ordered_ids']['eval_sha256'],
                    corpus_manifest_sha256=protocol['corpus_manifest_sha256'],
                    checkpoint_selection='fixed-final', sampling='sha256(seed,epoch,uid,stream)-v1',
                    gold_access='disabled', loss_policy='ones_positive_term')
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError('Pretraining provenance mismatch; no optimizer step allowed')
    for name in ('runtime', 'eval_runtime'):
        if provenance[name] != protocol['expected_provenance'][name]:
            raise ValueError('Strict runtime differs from baseline: ' + name)
    if any(env.get(k) != protocol['environment'][k] for k in ENV_KEYS):
        raise ValueError('Local environment differs from baseline')


def install_gold_blocker(root):
    """Main-process defense; worker source has only the unchanged corpus adapter."""
    root = Path(root)
    forbidden_file = os.path.normcase(str((root / 'storage/competition/train.csv').resolve()))
    forbidden_dir = os.path.normcase(str((root / 'storage/gold').resolve())) + os.sep
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.normcase(str(Path(os.fsdecode(args[0])).resolve()))
        if path == forbidden_file or path.startswith(forbidden_dir):
            raise PermissionError('Gold access forbidden by EXP082 contract')
    sys.addaudithook(hook)


def validate_arguments(args, root):
    """Read-only launch gate; validates all bindings before torch/model/input use."""
    root = Path(root)
    protocol = read(args.protocol)
    if protocol.get('schema') != SCHEMA or protocol.get('tag') != TAG:
        raise ValueError('Wrong EXP082 protocol')
    for key, value in protocol['expected_args'].items():
        if getattr(args, key, None) != value:
            raise ValueError('Frozen argument changed: ' + key)
    for key in ('corpus', 'labels', 'eval_labels', 'fold_file', 'exclude_ids'):
        if Path(getattr(args, key)).resolve() != contained(root, protocol['paths'][key]).resolve():
            raise ValueError('Frozen path changed: ' + key)
    if args.tag != TAG or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('Require frozen run tag and HF_HUB_OFFLINE=1')
    for relative, expected in protocol['source_bindings'].items():
        if sha256(contained(root, relative)) != expected:
            raise ValueError('Frozen file hash changed: ' + relative)
    for name, expected in protocol['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('Frozen training source changed: ' + name)
    ordered = read(contained(root, protocol['ordered_ids']['path']))
    if (uid_hash(ordered['train']) != protocol['ordered_ids']['train_sha256']
            or uid_hash(ordered['eval']) != protocol['ordered_ids']['eval_sha256']):
        raise ValueError('Ordered IDs hash mismatch')
    return protocol, ordered


def prepare(root=ROOT):
    import numpy as np
    root = Path(root)
    destination = root / PROTOCOL
    if destination.exists():
        raise ValueError('Refusing to overwrite an existing EXP082 protocol')
    parent = read(root / PARENT)
    for name, expected in parent['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('Frozen source mismatch')
    source = (root / 'training/src/raptor_train.py').read_text(encoding='utf-8')
    adapter = (root / ENTRYPOINT).read_text(encoding='utf-8')
    if adapter != render_training_source(source):
        raise ValueError('Static entrypoint is not the reviewed bounded transformation')
    for relative, expected in {**parent['source_file_sha256'], **{k.removeprefix('workspace://'): v for k, v in parent['metadata_source_hashes'].items()}}.items():
        if sha256(root / relative) != expected:
            raise ValueError('Frozen metadata/labels changed: ' + relative)
    corpus_ids = np.load(root / parent['corpus'] / 'ids.npy', allow_pickle=False).astype(str).tolist()
    with (root / parent['fold_file']).open(newline='', encoding='utf-8-sig') as stream:
        folds = {r[UID]: int(r['fold']) for r in csv.DictReader(stream)}
    train, evaluation = reconstruct_ids(corpus_ids, folds, set(parent['excluded_uids']), parent['expected_ids']['2'])
    labels = read_training_labels(root / parent['labels'], train, parent['targets'])
    baseline, actual = class_weights(labels, train)
    prov = read(root / 'artifacts/experiments/EXP080_R34_B_F2/provenance.json')
    report = read(root / 'artifacts/predictions/EXP080_F2_CONFIRM_REPORT/report.json')
    job = read(root / 'artifacts/logs/EXP080_R34_B_F2/job.json')
    destination.mkdir(parents=True)
    ordered_path = PROTOCOL + '/ordered-ids.json'
    new_json(root / ordered_path, dict(train=train, eval=evaluation))
    relative_bindings = [PARENT, 'artifacts/experiments/EXP080_R34_B_F2/provenance.json',
        'artifacts/logs/EXP080_R34_B_F2/job.json',
        'artifacts/predictions/EXP080_F2_CONFIRM_REPORT/report.json',
        *parent['source_file_sha256'], parent['corpus'] + '/ids.npy',
        parent['corpus'] + '/masks.npy', parent['corpus'] + '/manifest.json',
        'tools/exp082_r34_loss.py', ENTRYPOINT, ordered_path]
    bindings = {p: sha256(root / p) for p in relative_bindings}
    expected_args = {**parent['expected_args_by_arm']['RN_B'], 'fold': 2, 'topk': 3}
    protocol = dict(schema=SCHEMA, observed_at=datetime.now(timezone.utc).isoformat(),
        status='CPU_PREPARED_PENDING_ROOT_REVIEW_NO_TRAINING', tag=TAG,
        hypothesis='Change only RN positive BCE weight to ones; compare frozen B+RN90/10 development blends.',
        expected_args=expected_args, paths={k: parent[k] for k in ('corpus', 'labels', 'eval_labels', 'fold_file')},
        source_bindings=bindings, training_code_sha256=parent['training_code_sha256'],
        expected_ids=parent['expected_ids']['2'], excluded_uids=parent['excluded_uids'],
        ordered_ids=dict(path=ordered_path, train_sha256=uid_hash(train), eval_sha256=uid_hash(evaluation),
                         basis='Original corpus.ids order filtered by frozen fold>=0/common quarantine; membership equals EXP080.'),
        label_hash=parent['label_hash'], eval_label_hash=parent['eval_label_hash'], split_hash=parent['split_hash'],
        corpus_manifest_sha256=parent['corpus_manifest_sha256'], targets=parent['targets'],
        initial_state_sha256=parent['pretrained']['ResNet34']['initial_full_model_sha256'],
        pretrained=parent['pretrained']['ResNet34'], expected_provenance=prov,
        environment={k: report['local_environment'][k] for k in ENV_KEYS},
        loss=dict(baseline_rule='train float32 soft mean clipped[.03,.7], then (1-prev)/prev clipped[1,10]',
                  baseline_pos_weight=baseline.tolist(), actual_pos_weight=actual.tolist(),
                  reduction='mean', weight=None, candidate='ones positive term only'),
        baselines={k.removeprefix('workspace://'): v for k, v in report['source_hashes'].items()
                   if any(t in k for t in ('EXP080_R34_B_F2/', 'EXP080_INPUT_B_F2/'))},
        evaluation=dict(primary='macro12 AUC(.9rank(B8)+.1rank(RNones20)) - macro12 AUC(.9rank(B8)+.1rank(RNclip20))',
            rank='average tie rank/n per full F2 target; never rerank subgroups/bootstrap',
            fixed_weights=[.9,.1], conditional_ci=dict(reps=500, seed=20261003, unit='report_group'),
            secondary=['RN alone', 'all12 P/N and macro contributions', 'PFOA/other11/Fracture/MCL', 'existing scanner/mask groups'],
            no_ratio_epoch_target_search=True, no_all_targets_positive_gate=True, no_minimum_auc_gain_cutoff=True,
            decision='Retain small/uncertain effects with limits. Negative primary: do not directly replace; not broad strategy rejection.'),
        execution=dict(new_fits=1, expected_batches_per_epoch=434, expected_attempts=8680,
            strict_fp16=True, eval_batch=4, epoch_metric_early_stopping=False,
            optimizer_skip_review_fraction=.01, consecutive_skip_stop=10,
            estimate_gpu_minutes=[13,18], estimate_scope='Prior F2 RN772s; not measured for new loss',
            automatic_followup=False, launch_authority='Root CPU review before one locally authorized fit; this prep does not launch.'),
        limits=['Exposed weak F2/one seed, not independent/private evidence',
                'Known F0/F1 exact cached input and seven source pixel pairs duplicate; both train in F2',
                'No exhaustive duplicate prerequisite or case removal in this screen',
                'No anchor0.943/D01 transfer claim; future grouped split needs both arms retrained'],
        cpu_verified_scope='UID membership/order, label-derived weight vector, static source and synthetic CPU contracts only',
        deferred_runtime_checks=['Actual GPU initial state', 'strict runtime/environment before first optimizer step', 'full training/OOF outputs'])
    protocol['paths']['exclude_ids'] = 'artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/excluded.csv'
    manifest_path = destination / 'manifest.json'
    new_json(manifest_path, protocol)
    cmd = list(job['command'])
    cmd[1] = ENTRYPOINT
    cmd[cmd.index('--tag') + 1] = TAG
    cmd += ['--protocol', str(manifest_path)]
    new_json(destination / 'queue.json', [dict(tag=TAG, command=cmd)])
    (destination / 'trainer.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), adapter.splitlines(True),
        fromfile='frozen/raptor_train.py', tofile=ENTRYPOINT)), encoding='utf-8')
    return dict(status=protocol['status'], protocol=str(manifest_path), manifest_sha256=sha256(manifest_path),
                train=len(train), val=len(evaluation), baseline_pos_weight=baseline.tolist(), no_training=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if not args.prepare:
        parser.error('--prepare is the only supported operation; it never launches training')
    print(json.dumps(prepare(), indent=2))


if __name__ == '__main__':
    main()
