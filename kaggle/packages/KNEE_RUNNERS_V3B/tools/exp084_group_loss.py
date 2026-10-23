"""EXP084 explicit grouped, no-Gold fixed-final contract. Import has no I/O.

prepare registers the single prechosen split on CPU. run is an explicit bounded
supervisor used only by the parent GPU queue; it never retries a failed arm.
The original trainer is transformed as text, never imported or executed here.
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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL'
ENTRYPOINT = 'training/runners/exp084_group_loss.py'
PARENT = 'artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/manifest.json'
GROUP = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1'
SCHEMA = 'exp084-group-r34-loss-v1'
CODE_NAMES = ('raptor_train.py', 'raptor_common.py', 'raptor_data.py', 'raptor_runtime.py')
ENV_KEYS = ('torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'pydicom', 'cv2')
ARM_ORDER = ('RN_clip', 'RN_ones', 'B')
TAGS = dict(RN_clip='EXP084_RN_CLIP_G1', RN_ones='EXP084_RN_ONES_G1', B='EXP084_B_G1')
UID = 'StudyInstanceUID'
SALT = 'RN_GROUP_SCREEN_V1'
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def relpath(root, value):
    value = value.removeprefix('workspace://')
    path = Path(value)
    if path.is_absolute() or path.drive or '..' in path.parts:
        raise ValueError('Expected contained workspace-relative path')
    return Path(root) / path


def frozen_split(rows):
    """One deterministic component assignment. Only prepare calls this on real data."""
    if not rows or len({r[UID] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate eligible UID')
    assignments = {}
    for row in rows:
        comp = row['component_id']
        if len(comp) != 64 or any(c not in '0123456789abcdef' for c in comp):
            raise ValueError('Malformed component ID')
        if comp not in assignments:
            h = hashlib.sha256((SALT + ':' + comp).encode('utf-8')).digest()
            assignments[comp] = 'val' if int.from_bytes(h, 'big') % 5 == 2 else 'train'
    ordered = {s: sorted(r[UID] for r in rows if assignments[r['component_id']] == s) for s in ('train', 'val')}
    if not ordered['train'] or not ordered['val']:
        raise ValueError('Empty fixed split; do not redraw salt')
    return ordered, assignments


def validate_membership(corpus_ids, ordered, rows):
    train, val = ordered['train'], ordered['val']
    eligible = {r[UID] for r in rows}
    if (train != sorted(set(train)) or val != sorted(set(val)) or set(train) & set(val)
            or set(train) | set(val) != eligible or not eligible <= set(corpus_ids)
            or len(corpus_ids) != len(set(corpus_ids))):
        raise ValueError('Ordered cohort coverage/disjointness mismatch')
    tcomp = {r['component_id'] for r in rows if r[UID] in set(train)}
    vcomp = {r['component_id'] for r in rows if r[UID] in set(val)}
    if tcomp & vcomp:
        raise ValueError('Component crosses train/validation')
    return train, val


def class_weights(labels, train):
    import numpy as np
    values = np.stack([labels[u] for u in train]).astype(np.float32)
    if values.shape != (len(train), 12) or not np.isfinite(values).all():
        raise ValueError('Invalid training labels')
    prev = np.clip(values.mean(0), .03, .7)
    return np.clip((1 - prev) / prev, 1, 10).astype(np.float32)


def arm_for(protocol, tag):
    matches = [a for a in protocol['arms'].values() if a['tag'] == tag]
    if len(matches) != 1:
        raise ValueError('Unregistered arm tag')
    return matches[0]


def worker_init(_worker_id):
    """Bound worker-internal CPU pools; sampling itself is UID/epoch deterministic."""
    import torch
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def validate_label_contract(label_hash, eval_hash, split_hash, protocol):
    if (label_hash != protocol['label_hash'] or eval_hash != protocol['eval_label_hash']
            or split_hash != protocol['split_hash']):
        raise ValueError('Label/reference/split semantic hash mismatch')


def loss_weights(baseline, protocol, arm):
    import numpy as np
    expected = np.asarray(protocol['loss']['baseline_pos_weight'], dtype=np.float32)
    if baseline.dtype != np.float32 or baseline.shape != (12,) or not np.array_equal(baseline, expected):
        raise ValueError('T-only loss vector mismatch')
    actual = np.ones(12, dtype=np.float32) if arm['loss_policy'] == 'ones_positive_term' else baseline
    if not np.array_equal(actual, np.asarray(arm['actual_pos_weight'], dtype=np.float32)):
        raise ValueError('Registered arm loss mismatch')
    return actual


def install_gold_blocker(root):
    root = Path(root)
    blocked_file = os.path.normcase(str((root / 'storage/competition/train.csv').resolve()))
    blocked_dir = os.path.normcase(str((root / 'storage/gold').resolve())) + os.sep
    def hook(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.normcase(str(Path(os.fsdecode(args[0])).resolve()))
            if path == blocked_file or path.startswith(blocked_dir):
                raise PermissionError('Gold access forbidden by EXP084 contract')
    sys.addaudithook(hook)


def validate_bindings(protocol, root):
    if protocol.get('schema') != SCHEMA or tuple(protocol['arm_order']) != ARM_ORDER:
        raise ValueError('Wrong EXP084 protocol')
    for path, expected in protocol['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('Frozen binding changed: ' + path)
    for name, expected in protocol['training_code_sha256'].items():
        if sha256(Path(root) / 'training/src' / name) != expected:
            raise ValueError('Original training source changed: ' + name)


def validate_arguments(args, root):
    protocol = read(args.protocol)
    validate_bindings(protocol, root)
    arm = arm_for(protocol, args.tag)
    for key, expected in arm['expected_args'].items():
        if getattr(args, key, None) != expected:
            raise ValueError('Frozen argument changed: ' + key)
    for key, expected in protocol['paths'].items():
        if Path(getattr(args, key)).resolve() != relpath(root, expected).resolve():
            raise ValueError('Frozen input path changed: ' + key)
    for key, expected in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(key) != expected:
            raise ValueError('Require registered environment: ' + key)
    ordered = read(relpath(root, protocol['ordered_ids']['path']))
    if any(uid_hash(ordered[k]) != protocol['ordered_ids'][k + '_sha256'] for k in ('train', 'val')):
        raise ValueError('Ordered UID hash mismatch')
    return protocol, ordered


def validate_pretraining(provenance, env, protocol, arm):
    expected = dict(code_sha256=protocol['training_code_sha256'], initial_state_sha256=arm['initial_state_sha256'],
        train_ids_sha256=protocol['ordered_ids']['train_sha256'], eval_ids_sha256=protocol['ordered_ids']['val_sha256'],
        train_ids_ordered_sha256=protocol['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=protocol['ordered_ids']['val_sha256'],
        corpus_manifest_sha256=protocol['corpus_manifest_sha256'], checkpoint_selection='fixed-final',
        sampling='sha256(seed,epoch,uid,stream)-v1', gold_access='disabled', loss_policy=arm['loss_policy'],
        metric_access='deferred_until_all_three_complete')
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError('Pretraining provenance mismatch; optimizer forbidden')
    for key in ('runtime', 'eval_runtime'):
        if provenance[key] != protocol['expected_runtime'][key]:
            raise ValueError('Strict runtime mismatch: ' + key)
    if any(env.get(k) != protocol['environment'][k] for k in ENV_KEYS):
        raise ValueError('Training environment differs from registration')


def render_training_source(source):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.exp082_r34_loss import render_training_source as prior_render, replace_once
    source = prior_render(source)
    source = source.replace('EXP082 fixed RN20/ones', 'EXP084 grouped fixed-final')
    source = source.replace('build_model, environment, load_folds,', 'build_model, environment,')
    source = source.replace('load_label_csv, macro_auc, seed_all,', 'load_label_csv, seed_all,')
    source = replace_once(source, 'from tools import exp082_r34_loss as guard', 'from tools import exp084_group_loss as guard')
    source = replace_once(source, '    protocol, ordered = guard.validate_arguments(a, ROOT)',
        '    protocol, ordered = guard.validate_arguments(a, ROOT)\n    arm = guard.arm_for(protocol, a.tag)')
    source = replace_once(source, '    run_environment = environment()  # Audit imports happen before the original seed boundary.',
        '    guard.worker_init(0)\n    run_environment = environment()  # Imports happen before the original seed boundary.')
    source = replace_once(source, 'persistent_workers=a.workers > 0, generator=shuffle_generator)',
        'persistent_workers=a.workers > 0, generator=shuffle_generator, worker_init_fn=guard.worker_init)')
    source = replace_once(source, 'num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False)',
        'num_workers=min(a.workers, 2), collate_fn=collate, persistent_workers=False, worker_init_fn=guard.worker_init)')
    begin = source.index('    folds, split_hash = load_folds(Path(a.fold_file))')
    finish = source.index('    log(f"arch ', begin)
    source = source[:begin] + '''    split_hash = guard.sha256(Path(a.fold_file))
    excluded = set(protocol["excluded_uids"])
    group_rows = guard.read_rows(ROOT / protocol["groups"]["path"])
    train_ids, oof_ids = guard.validate_membership(corpus.ids.astype(str).tolist(), ordered, group_rows)
    if any(u not in labels for u in train_ids) or any(u not in eval_labels for u in oof_ids):
        raise SystemExit("Frozen label coverage changed")
    # V labels are not used by training/inference; scoring belongs to the later evaluator.
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}
    guard.validate_label_contract(label_hash, eval_label_hash, split_hash, protocol)
    log(f"{a.tag} | train {len(train_ids)} | val {len(oof_ids)} | Gold and metrics disabled")
''' + source[finish:]
    source = replace_once(source, '    guard.validate_loss_vectors(baseline_pos_weight, protocol)\n    pos_weight = torch.ones(len(LAB), dtype=torch.float32, device=dev)',
        '    pos_weight = torch.tensor(guard.loss_weights(baseline_pos_weight, protocol, arm), dtype=torch.float32, device=dev)')
    source = source.replace('loss_policy="ones_positive_term"', 'loss_policy=arm["loss_policy"]')
    source = replace_once(source, '    guard.validate_pretraining(provenance, run_environment, protocol)',
        '    provenance.update(metric_access="deferred_until_all_three_complete", requested_workers=a.workers, effective_workers=a.workers, internal_cpu_threads=1)\n    guard.validate_pretraining(provenance, run_environment, protocol, arm)')
    source = replace_once(source, '        macro, per = macro_auc(Y, P)\n        return macro, per, P, U',
        '        return None, {}, P, U  # No validation metric is computed here.')
    source = replace_once(source, '        oof_auc, oof_per, _, _ = evaluate(oof_epoch_ids) if oof_epoch_ids and a.fixed_epoch is None else (None, {}, None, None)',
        '        oof_auc, oof_per = None, {}  # Fixed-final: no epoch evaluation.')
    source = replace_once(source, '        score = float(ep) if a.fixed_epoch is not None else oof_auc',
        '        score = float(ep)  # Epoch index only, never validation selection.')
    source = replace_once(source, '        if best is None or score > best["selection"]:', '        if ep == a.epochs - 1:')
    source = replace_once(source, '''        if a.fixed_epoch is None:
            topk.append(state(ep, {"selection": score})); topk.sort(key=lambda s: -s["selection"]); del topk[max(1, a.topk):]
''', '')
    begin = source.index('    results = {}')
    finish = source.index('    summary = {', begin)
    source = source[:begin] + '''    model.load_state_dict(best["model"])
    _, _, P, U = evaluate(oof_ids)
    if U != oof_ids:
        raise RuntimeError("Final prediction UID order changed")
    write_predictions(out / "oof_best.csv", U, P)
    chosen = "best"
    results = {chosen: {"oof_rows": len(U), "metric_access": "deferred_until_all_three_complete"}}
    log("Final predictions saved; validation metrics deferred until all three fits complete")
''' + source[finish:]
    source = source.replace('"loss_policy": "ones_positive_term"', '"loss_policy": arm["loss_policy"]')
    source = source.replace('exposed F2 weak development; known F0/F1 duplicate both in train; not independent/private evidence',
        'prospective grouped image-model exclusion; weak labels/researcher exposure/unknown patient links remain; not D01/private evidence')
    source = replace_once(source, '    json.dump(summary, open(out / "summary.json", "w"), indent=1)',
        '''    summary.update(metric_access="deferred_until_all_three_complete", arm=arm["id"],
                   expected_attempts=arm["expected_attempts"],
                   optimizer_skip_fraction=skipped_updates / max(1, optimizer_updates + skipped_updates))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    guard.validate_completion(out, protocol, arm)''')
    return source


def validate_completion(out, protocol, arm):
    s = read(Path(out) / 'summary.json')
    p = s['provenance']
    if (s['selected_checkpoint'] != 'best' or s['selected_epoch_1based'] != arm['expected_args']['epochs']
            or s['train'] != protocol['expected_ids']['train'] or s['val'] != protocol['expected_ids']['val']
            or s['optimizer_updates'] + s['skipped_updates'] != arm['expected_attempts']
            or s['optimizer_skip_fraction'] >= .01 or s['weak_oof_macro_auc'] is not None
            or s['metric_access'] != 'deferred_until_all_three_complete'
            or p['initial_state_sha256'] != arm['initial_state_sha256']):
        raise ValueError('Completion contract/skip review failed; stop subsequent arms')
    for name in ('model_best.pt', 'oof_best.csv', 'provenance.json', 'history.json'):
        if not (Path(out) / name).is_file():
            raise ValueError('Missing final artifact: ' + name)


def prepare(asset_receipt, root=ROOT):
    import numpy as np
    import pandas as pd
    root = Path(root)
    install_gold_blocker(root)
    destination = root / PROTOCOL
    if destination.exists():
        raise ValueError('Refusing to redraw/overwrite existing EXP084 registration')
    parent = read(root / PARENT)
    prior = read(root / 'artifacts/experiments/EXP082_R34_LOSS_F2_PROTOCOL/manifest.json')
    group_contract = read(root / GROUP / 'contract.json')
    for name, expected in parent['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('Frozen original source mismatch')
    for name, expected in group_contract['outputs'].items():
        if sha256(root / GROUP / name) != expected['sha256']:
            raise ValueError('Existing group contract output mismatch')
    for path, expected in {**parent['source_file_sha256'], **parent['metadata_source_hashes']}.items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('Frozen label/metadata mismatch: ' + path)
    original = (root / 'training/src/raptor_train.py').read_text(encoding='utf-8')
    rendered = render_training_source(original)
    if (root / ENTRYPOINT).read_text(encoding='utf-8') != rendered:
        raise ValueError('Runner differs from reviewed renderer')
    sys.path.insert(0, str(root / 'training/src'))
    from raptor_common import load_label_csv
    labels, label_hash = load_label_csv(root / parent['labels'])
    if label_hash != parent['label_hash']:
        raise ValueError('Label semantic hash changed')
    rows = read_rows(root / GROUP / 'groups.csv')
    if len(rows) != 4343 or len({r['component_id'] for r in rows}) != 4196:
        raise ValueError('Eligible group count drift')
    ordered, assignments = frozen_split(rows)
    corpus_ids = np.load(root / parent['corpus'] / 'ids.npy', allow_pickle=False).astype(str).tolist()
    train, val = validate_membership(corpus_ids, ordered, rows)
    if any(u not in labels for u in train + val):
        raise ValueError('Eligible label coverage mismatch')
    weights = class_weights(labels, train)
    y = np.rint(np.stack([labels[u] for u in val]).astype(np.float32)).astype(int)
    counts = [{'target': target, 'P': int(y[:, j].sum()), 'N': int(len(val)-y[:, j].sum())} for j, target in enumerate(parent['targets'])]
    valid_macro = all(r['P'] and r['N'] for r in counts)
    assets = read(asset_receipt)
    for key, parent_key in (('RN', 'ResNet34'), ('B', 'CoAtNet')):
        asset = assets['assets'][key]
        if (asset['fresh_full_model_state_sha256'] != parent['pretrained'][parent_key]['initial_full_model_sha256']
                or asset['seed'] != 42 or asset['device'] != 'cpu'
                or sha256(asset['local_cache_path']) != asset['sha256']):
            raise ValueError('Offline initial asset/CPU state receipt mismatch')
    meta_path = 'storage/labels/derived_meta/study_meta.csv'
    meta = pd.read_csv(root / meta_path, dtype={UID: str}).set_index(UID)
    if not meta.index.is_unique or not set(train + val) <= set(meta.index):
        raise ValueError('Metadata coverage mismatch')
    masks = np.load(root / parent['corpus'] / 'masks.npy', allow_pickle=False)
    mask_group = dict(zip(corpus_ids, np.where((masks == 0).any(1), 'some_zero_mask', 'all_nonzero')))
    destination.mkdir(parents=True)
    ordered_path = PROTOCOL + '/ordered-ids.json'
    new_json(root / ordered_path, ordered)
    group_path = PROTOCOL + '/group-metadata.csv'
    cols = [UID, 'component_id', 'report_group_sha256', 'split', 'scanner_group', 'field', 'manufacturer', 'cached_mask']
    with (root / group_path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, lineterminator='\n'); writer.writeheader()
        for r in sorted(rows, key=lambda r: r[UID]):
            u = r[UID]
            out = {k: r[k] for k in cols[:3]}
            out.update(split=assignments[r['component_id']], cached_mask=mask_group[u])
            out.update({k: 'unknown' if pd.isna(meta.loc[u, k]) else str(meta.loc[u, k]) for k in ('scanner_group', 'field', 'manufacturer')})
            writer.writerow(out)
    new_json(destination / 'initial-assets.json', assets)
    arms = {}
    for name in ARM_ORDER:
        base = 'B' if name == 'B' else 'RN_B'
        args = {**parent['expected_args_by_arm'][base], 'workers': 12, 'device': 'cuda', 'fold': 2, 'topk': 3}
        tag = TAGS[name]
        init = parent['pretrained']['CoAtNet' if name == 'B' else 'ResNet34']['initial_full_model_sha256']
        run = 'artifacts/experiments/' + tag
        policy = 'ones_positive_term' if name == 'RN_ones' else 'clipped_positive_term'
        arms[name] = dict(id=name, tag=tag, expected_args=args, loss_policy=policy,
            actual_pos_weight=(np.ones(12, np.float32) if name == 'RN_ones' else weights).tolist(),
            initial_state_sha256=init, expected_attempts=args['epochs'] * (len(train)//8),
            time_limit_seconds=3600 if name == 'B' else 1800,
            outputs={k: run + '/' + filename for k, filename in dict(summary='summary.json', provenance='provenance.json', predictions='oof_best.csv', checkpoint='model_best.pt', history='history.json').items()},
            logs='artifacts/logs/' + tag)
    bindings = [PARENT, 'artifacts/experiments/EXP082_R34_LOSS_F2_PROTOCOL/manifest.json',
        GROUP + '/contract.json', *[GROUP + '/' + name for name in group_contract['outputs']],
        parent['labels'], 'artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/excluded.csv', meta_path, parent['corpus'] + '/ids.npy', parent['corpus'] + '/masks.npy', parent['corpus'] + '/manifest.json',
        'tools/exp082_r34_loss.py', 'tools/exp084_group_loss.py', ENTRYPOINT, ordered_path, group_path, PROTOCOL + '/initial-assets.json',
        'tools/exp084_group_loss_report.py']
    protocol = dict(schema=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(), status='CPU_REGISTERED_PENDING_ROOT_LAUNCH_REVIEW',
        arm_order=list(ARM_ORDER), arms=arms, source_bindings={p: sha256(root / p) for p in bindings},
        training_code_sha256=parent['training_code_sha256'], targets=parent['targets'],
        paths=dict(corpus=parent['corpus'], labels=parent['labels'], eval_labels=parent['eval_labels'], fold_file=group_path, exclude_ids='artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/excluded.csv'),
        labels=dict(path=parent['labels'], sha256=sha256(root / parent['labels']), semantic_sha256=label_hash, reference='float32 then numpy.rint; fixed weak Dread labels'),
        label_hash=label_hash, eval_label_hash=label_hash, split_hash=sha256(root / group_path),
        ordered_ids=dict(path=ordered_path, sha256=sha256(root / ordered_path), train_sha256=uid_hash(train), val_sha256=uid_hash(val)),
        groups=dict(path=group_path, sha256=sha256(root / group_path), component_column='component_id', subgroup_columns=['scanner_group','field','manufacturer','cached_mask']),
        expected_ids=dict(train=len(train), val=len(val), train_components=sum(s=='train' for s in assignments.values()), val_components=sum(s=='val' for s in assignments.values())),
        split=dict(salt=SALT, exact_payload="UTF8('RN_GROUP_SCREEN_V1:' + lowercase_component_id); no BOM/newline", rule='unsigned SHA256 big-endian integer modulo5 equals2 => val; remainder=>train', fold_argument_semantics='New grouped-hash residue2 of5; never historical F2 membership', untouched=False, patient_independence_proven=False, no_redraw=True),
        excluded_uids=parent['excluded_uids'], eligibility='Existing eligible4343 includes prior Gold/report quarantine and excludes corrupt2; no Gold reread',
        loss=dict(baseline_pos_weight=weights.tolist(), rule='New T float32 soft mean clipped[.03,.7], then (1-prev)/prev clipped[1,10]'),
        reference_counts=counts, macro12_valid=bool(valid_macro),
        corpus_manifest_sha256=parent['corpus_manifest_sha256'],
        expected_runtime={k: prior['expected_provenance'][k] for k in ('runtime','eval_runtime')}, environment=prior['environment'],
        initial_assets=dict(path=PROTOCOL + '/initial-assets.json', sha256=sha256(destination / 'initial-assets.json')),
        execution=dict(new_fits=3, total_time_limit_seconds=7200, retries=0, estimate_gpu_minutes=[66,82], eval_batch=4, workers=12,
            internal_thread_env=THREAD_ENV, optimizer_skip_review_fraction=.01, consecutive_skip_stop=10, all_three_before_metrics=True),
        evaluation=dict(primary='macro12(blend_ones)-macro12(blend_clip)', weights=[.9,.1], rank='average tie rank/N on full V; no final/subgroup/bootstrap rerank', bootstrap_reps=500, bootstrap_seed=20261003, bootstrap_unit='component_id', no_ratio_epoch_target_search=True),
        limitations=['Past researcher exposure, weak labels and unknown patient links remain', 'External Dread labeler full ancestry unknown; no new image teacher/calibrator', 'Not exact D01/public0.943/private validation', 'No A arm: historical B-minus-A Fracture loss not directly resolved'])
    new_json(destination / 'manifest.json', protocol)
    queue = []
    for index, name in enumerate(ARM_ORDER):
        item = dict(tag=TAGS[name], command=[sys.executable, 'tools/exp084_group_loss.py', 'run', '--protocol', PROTOCOL + '/manifest.json', '--arm', name])
        if index: item['after'] = TAGS[ARM_ORDER[index-1]]
        queue.append(item)
    new_json(destination / 'queue.json', queue)
    (destination / 'trainer.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), rendered.splitlines(True), fromfile='frozen/raptor_train.py', tofile=ENTRYPOINT)), encoding='utf-8')
    return dict(status=protocol['status'], manifest_sha256=sha256(destination / 'manifest.json'), expected_ids=protocol['expected_ids'], macro12_valid=valid_macro, no_training=True)


def run_arm(protocol_path, name, root=ROOT):
    """Enforce bounded arm/whole-experiment wall time and kill only owned child tree."""
    root = Path(root); protocol = read(protocol_path); validate_bindings(protocol, root)
    if not protocol['macro12_valid']:
        raise ValueError('Frozen split macro12 invalid; root review required, no redraw')
    arm = protocol['arms'][name]; index = ARM_ORDER.index(name)
    for previous in ARM_ORDER[:index]:
        prior = protocol['arms'][previous]
        if not (root / prior['logs'] / 'SUCCESS').is_file() or (root / prior['logs'] / 'FAILED').exists():
            raise ValueError('Predecessor is not successful: ' + previous)
        validate_completion(root / 'artifacts/experiments' / prior['tag'], protocol, prior)
    run_path = root / 'artifacts/experiments' / arm['tag']
    if run_path.exists():
        raise ValueError('No automatic retry/overwrite of existing arm')
    budget_path = root / 'artifacts/logs/EXP084_GROUP_R34_LOSS_BUDGET.json'
    if name == ARM_ORDER[0]:
        new_json(budget_path, dict(started_epoch=time.time(), protocol_sha256=sha256(protocol_path)))
    budget = read(budget_path)
    if budget['protocol_sha256'] != sha256(protocol_path):
        raise ValueError('Experiment budget protocol mismatch')
    remaining = protocol['execution']['total_time_limit_seconds'] - (time.time()-budget['started_epoch'])
    timeout = min(remaining, arm['time_limit_seconds'])
    if timeout <= 0:
        raise ValueError('Overall120-minute budget exhausted')
    command = [sys.executable, str(root / ENTRYPOINT), '--protocol', str(Path(protocol_path).resolve()), '--tag', arm['tag']]
    for key, value in {**arm['expected_args'], **protocol['paths']}.items():
        option = '--' + key.replace('_','-')
        if isinstance(value, bool):
            if value: command.append(option)
        else:
            command.extend([option, str(relpath(root,value)) if key in protocol['paths'] else str(value)])
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'PYTHONDONTWRITEBYTECODE':'1'}
    if env.get('CUDA_VISIBLE_DEVICES') == '':
        raise ValueError('GPU hidden in inherited environment; parent must choose correct launch environment')
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    proc = subprocess.Popen(command, cwd=root, env=env, creationflags=flags, start_new_session=os.name!='nt')
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'], check=False)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
        raise RuntimeError('Registered wall-time limit exceeded; no retry')
    if code != 0:
        raise RuntimeError('Registered arm failed with exit ' + str(code))
    validate_completion(run_path, protocol, arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--asset-receipt', required=True)
    p = sub.add_parser('run'); p.add_argument('--protocol', required=True); p.add_argument('--arm', choices=ARM_ORDER, required=True)
    args = parser.parse_args()
    if args.command == 'prepare': print(json.dumps(prepare(args.asset_receipt), indent=2))
    else: run_arm(args.protocol, args.arm)


if __name__ == '__main__':
    main()
