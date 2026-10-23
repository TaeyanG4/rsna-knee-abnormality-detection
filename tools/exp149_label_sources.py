"""EXP149 label-source experiment registration and CPU control surface.

No command in this module launches training. Queue commands only write JSON.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp134_aug1_kfold as aug
from tools import exp127_radr50_single as e127
from tools import exp117_multihead as mh

PROTOCOL = 'artifacts/experiments/EXP149_LABEL_SOURCES'
TEACHER_MANIFEST = PROTOCOL + '/teachers/manifest.json'
STUDENT_MANIFEST = PROTOCOL + '/students/manifest.json'
TOOL = 'tools/exp149_label_sources.py'
RUNNER = 'training/runners/exp149_label_sources.py'
TEST = 'tests/test_exp149_label_sources.py'
AQUILA = 'storage/labels/derived_constellation/aquila_v1.csv'
JEV = 'storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv'
COMPONENTS = e127.COMPONENTS
T = e127.T
UID = e127.UID
EFFUSION = T.index('Effusion')
ARMS = ('L1', 'L2', 'L3', 'H1')
TEACHERS = {
    'T234': ((2, 3, 4), (0, 1)),
    'T134': ((1, 3, 4), (2,)),
    'T124': ((1, 2, 4), (3,)),
    'T123': ((1, 2, 3), (4,)),
    'T034': ((0, 3, 4), (2,)),
    'T024': ((0, 2, 4), (3,)),
    'T023': ((0, 2, 3), (4,)),
}
STUDENT_TAGS = tuple(f'EXP149_{arm}_S42_F{fold}' for arm in ARMS for fold in (0, 1))
THREAD_ENV = aug.THREAD_ENV
CODE_NAMES, ENV_KEYS, NEW_CODE = aug.CODE_NAMES, aug.ENV_KEYS, e127.NEW_CODE
worker_init, install_gold_blocker = aug.worker_init, aug.install_gold_blocker
read, sha256, uid_hash = aug.read, aug.sha256, aug.uid_hash
pos_weight, fold_contract = aug.pos_weight, aug.fold_contract
state_hashes, mismatched_keys = aug.state_hashes, aug.mismatched_keys
load_label_csv, forbid_cuda_init = aug.load_label_csv, aug.forbid_cuda_init
head_pos_weight, stack_sources = mh.head_pos_weight, mh.stack_sources
build_multihead, masked_multihead_bce, rank_mean = mh.build_multihead, mh.masked_multihead_bce, mh.rank_mean


def path(root, relative):
    return Path(root) / relative


def exclusive_json(dest, value):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def base_protocol(root=ROOT):
    root = Path(root)
    p = read(root / aug.PROTOCOL / 's42/manifest.json')
    f0 = read(e127.manifest_path('AUG1', 42, root))
    if p['seed'] != 42 or p['common_args'] != f0['common_args'] or p['folds']['0'] != f0['folds']['0']:
        raise ValueError('EXP134/EXP127 matched control contract changed')
    if p['common_args']['epochs'] != 20 or p['common_args']['k'] != 12 or p['common_args']['k_eval'] != 24:
        raise ValueError('EXP149 fixed recipe changed')
    for rel, expected in p['source_bindings'].items():
        if sha256(root / rel) != expected:
            raise ValueError('frozen AUG1 source changed: ' + rel)
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('frozen training source changed: ' + name)
    for k in range(5):
        f = p['folds'][str(k)]
        if sha256(root / f['ordered_ids']['path']) != f['ordered_ids']['sha256']:
            raise ValueError('EXP099 ordered roster changed')
    return p


def fold_rosters(p, root=ROOT):
    """Canonical EXP099 roster and map; no label or report is read."""
    root = Path(root)
    canonical = aug.canonical_order(read(root / p['paths']['base_ordered_ids']))
    if len(canonical) != len(set(canonical)):
        raise ValueError('canonical roster duplicated')
    val = {k: read(root / p['folds'][str(k)]['ordered_ids']['path'])['val'] for k in range(5)}
    fold_of = {}
    for k, ids in val.items():
        if len(ids) != p['folds'][str(k)]['n_val']:
            raise ValueError('EXP099 fold size changed')
        for uid in ids:
            if uid in fold_of:
                raise ValueError('EXP099 fold overlap')
            fold_of[uid] = k
    if set(fold_of) != set(canonical):
        raise ValueError('EXP099 folds do not cover canonical roster')
    for k in (0, 1):
        original = read(root / p['folds'][str(k)]['ordered_ids']['path'])
        if [u for u in canonical if fold_of[u] != k] != original['train']:
            raise ValueError('student train order differs from EXP099 contract')
    return canonical, fold_of


def ordered_for(canonical, fold_of, train_folds, eval_folds):
    train_folds, eval_folds = tuple(train_folds), tuple(eval_folds)
    if set(train_folds) & set(eval_folds) or not train_folds or not eval_folds:
        raise ValueError('teacher/student fold intersection')
    return {'train': [u for u in canonical if fold_of[u] in train_folds],
            'val': [u for u in canonical if fold_of[u] in eval_folds]}


def source_record(relative, root=ROOT):
    values, semantic = load_label_csv(path(root, relative))
    return values, {'path': relative, 'sha256': sha256(path(root, relative)), 'semantic_sha256': semantic}


def _spec(tag, kind, arm, train_folds, eval_folds, ordered, labels, label_rec, base, *, heads=(), head_labels=(), root=ROOT):
    pw = pos_weight(labels, ordered['train'])
    fc = fold_contract(len(ordered['train']), base['common_args'])
    spec = {'tag': tag, 'kind': kind, 'arm': arm, 'fold': eval_folds[0] if kind == 'student' else -1,
            'train_folds': list(train_folds), 'eval_folds': list(eval_folds), 'ordered_ids': ordered,
            'train_ids_sha256': uid_hash(ordered['train']), 'eval_ids_sha256': uid_hash(ordered['val']),
            'n_train': len(ordered['train']), 'n_val': len(ordered['val']), **fc,
            'pos_weight': pw.tolist(), 'labels': label_rec['path'],
            'label_sha256': label_rec['sha256'], 'label_semantic_sha256': label_rec['semantic_sha256'],
            'label_set': arm.lower() if kind == 'student' else 'dread',
            'initial_state_sha256': base['initial_state_sha256'],
            'heads': list(heads), 'head_labels': list(head_labels)}
    if heads:
        by_name = {h['name']: source_record(h['path'], root)[0] for h in head_labels}
        spec['head_pos_weight'] = {n: head_pos_weight(by_name[n], ordered['train']).tolist() for n in heads}
    return spec


def _manifest(base, kind, runs, root=ROOT):
    p = copy.deepcopy(base)
    p.update(schema='exp149-label-sources-v1', status='REGISTERED_NO_EXECUTION',
             experiment='EXP149_LABEL_SOURCES', run_kind=kind, runs=runs,
             source_bindings={**base['source_bindings'], 'tools/exp117_multihead.py': sha256(path(root, 'tools/exp117_multihead.py')),
                              TOOL: sha256(path(root, TOOL)), RUNNER: sha256(path(root, RUNNER)),
                              TEST: sha256(path(root, TEST)), PROTOCOL + '/PROTOCOL.md': sha256(path(root, PROTOCOL + '/PROTOCOL.md'))})
    return p


def prepare_teachers(root=ROOT):
    from tools import exp130_gold58_compare as g130
    root = Path(root)
    dest = root / TEACHER_MANIFEST
    if dest.exists():
        raise FileExistsError(dest)
    base = base_protocol(root)
    canonical, fold_of = fold_rosters(base, root)
    gold_ids, _ = g130.official_gold(root)
    if set(canonical) & set(gold_ids):
        raise ValueError('Gold58 overlaps teacher training/evaluation roster')
    dread, rec = source_record(base['paths']['labels'], root)
    runs = {}
    for teacher, (train_folds, eval_folds) in TEACHERS.items():
        tag = f'EXP149_{teacher}_S42'
        ordered = ordered_for(canonical, fold_of, train_folds, eval_folds)
        runs[tag] = _spec(tag, 'teacher', teacher, train_folds, eval_folds, ordered, dread, rec, base)
    p = _manifest(base, 'teacher', runs, root)
    exclusive_json(dest, p)
    print(json.dumps({'status': p['status'], 'manifest': str(dest), 'teachers':
                      {tag: {'train': s['n_train'], 'val': s['n_val'], 'steps': s['steps']} for tag, s in runs.items()}}, indent=2))
    return p


def _load_manifest(kind, root=ROOT):
    file = path(root, TEACHER_MANIFEST if kind == 'teacher' else STUDENT_MANIFEST)
    p = read(file)
    if p['schema'] != 'exp149-label-sources-v1' or p['run_kind'] != kind or p['status'] != 'REGISTERED_NO_EXECUTION':
        raise ValueError('EXP149 manifest invalid')
    base = base_protocol(root)
    if p['common_args'] != base['common_args'] or p['initial_state_sha256'] != base['initial_state_sha256']:
        raise ValueError('EXP149 recipe/initial state differs')
    for rel, digest in p['source_bindings'].items():
        if sha256(path(root, rel)) != digest:
            raise ValueError('EXP149 source changed: ' + rel)
    return p


def validate_arguments(args, root=ROOT):
    root = Path(root)
    if args.run.startswith('EXP149_T'):
        kind = 'teacher'
    elif args.run in STUDENT_TAGS:
        kind = 'student'
    else:
        raise ValueError('unregistered EXP149 run')
    p = _load_manifest(kind, root)
    if args.run not in p['runs'] or Path(args.protocol).resolve() != path(root, TEACHER_MANIFEST if kind == 'teacher' else STUDENT_MANIFEST).resolve():
        raise ValueError('run/protocol mismatch')
    spec = p['runs'][args.run]
    if args.fold != spec['fold'] or args.variant != 'AUG1' or args.tag not in (None, args.run):
        raise ValueError('run fold, tag, or variant changed')
    if args.mode == 'full' and args.validate_only is False and os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('remote relaxation refused')
    for key, expected in p['common_args'].items():
        got = getattr(args, key)
        if got != expected and not (args.validate_only and key == 'device' and got == 'cpu'):
            raise ValueError('registered recipe changed: ' + key)
    if not 0 <= args.workers <= 12:
        raise ValueError('workers must be 0..12')
    if Path(args.corpus).resolve() != path(root, p['paths']['corpus']).resolve() or Path(args.labels).resolve() != path(root, spec['labels']).resolve() or Path(args.eval_labels).resolve() != path(root, p['paths']['eval_labels']).resolve():
        raise ValueError('registered input path changed')
    if Path(args.init_weights).resolve() != aug.weights_path(root).resolve() or sha256(args.init_weights) != p['init_weights']['sha256']:
        raise ValueError('initial weights changed')
    if sha256(path(root, spec['labels'])) != spec['label_sha256']:
        raise ValueError('training labels changed')
    for h in spec['head_labels']:
        if sha256(path(root, h['path'])) != h['sha256']:
            raise ValueError('head labels changed')
    canonical, fold_of = fold_rosters(p, root)
    ordered = ordered_for(canonical, fold_of, spec['train_folds'], spec['eval_folds'])
    if ordered != spec['ordered_ids'] or uid_hash(ordered['train']) != spec['train_ids_sha256'] or uid_hash(ordered['val']) != spec['eval_ids_sha256']:
        raise ValueError('registered ordered roster changed')
    if kind == 'teacher' and (set(spec['train_folds']) & set(spec['eval_folds'])):
        raise ValueError('teacher leakage')
    default = path(root, 'artifacts/experiments').resolve()
    out = Path(args.output_root).resolve() if args.output_root else default
    if args.mode == 'full' and out != default:
        raise ValueError('full output root changed')
    if args.mode == 'smoke' and (out == default or not str(out).startswith(str(path(root, '.local').resolve()))):
        raise ValueError('smoke output must be under .local')
    return p, spec, ordered, out


def runner_argv(p, tag, root=ROOT, workers=12, *, mode='full', output_root=None):
    root = Path(root)
    s = p['runs'][tag]
    argv = aug.e113.runner_argv(p, max(0, s['fold']), 'full', root=root, workers=workers, label_set='dread')
    argv[1] = str(root / RUNNER)
    i = argv.index('--label-set')
    del argv[i:i + 2]
    argv = aug._set_flag(argv, '--protocol', str(root / (TEACHER_MANIFEST if p['run_kind'] == 'teacher' else STUDENT_MANIFEST)))
    argv = aug._set_flag(argv, '--tag', tag)
    argv = aug._set_flag(argv, '--fold', str(s['fold']))
    argv = aug._set_flag(argv, '--labels', str(root / s['labels']))
    argv = aug._set_flag(argv, '--mode', mode)
    argv += ['--run', tag]
    if output_root:
        argv += ['--output-root', str(output_root)]
    return argv


def validate_cli(kind, root=ROOT):
    p = _load_manifest(kind, root)
    for tag in p['runs']:
        argv = aug._set_flag(runner_argv(p, tag, root, workers=0), '--device', 'cpu') + ['--validate-only']
        result = subprocess.run(argv, cwd=root, env=aug.e113.local_env(), check=False)
        if result.returncode:
            raise RuntimeError(f'validation failed for {tag} ({result.returncode})')
    return len(p['runs'])


def queue_entries(kind, after, root=ROOT, workers=12, tags=None):
    if not 0 <= workers <= 12:
        raise ValueError('workers must be 0..12')
    p = _load_manifest(kind, root)
    chosen = list(p['runs']) if tags is None else [f'EXP149_{s}_S42' if kind == 'teacher' and s in TEACHERS else s for s in tags]
    if not chosen or len(chosen) != len(set(chosen)) or any(tag not in p['runs'] for tag in chosen):
        raise ValueError('queue tag subset invalid or duplicated')
    previous = None if after.lower() == 'none' else after
    entries = []
    for tag in chosen:
        entry = {'tag': tag, 'command': runner_argv(p, tag, root, workers), 'continue_on_failure': False}
        if previous:
            entry['after'] = previous
        entries.append(entry)
        previous = tag
    return entries


def strict_aquila(root, expected_ids, gold_ids):
    """Read only target columns and reject partial, duplicate, or Gold rows."""
    import numpy as np
    import pandas as pd
    file = path(root, AQUILA)
    if not file.is_file():
        raise FileNotFoundError('Aquila image-checked labels not yet available: ' + str(file))
    header = pd.read_csv(file, nrows=0).columns.tolist()
    if header != [UID, *T]:
        raise ValueError('aquila_v1 needs exactly the canonical UID + 12 target columns')
    d = pd.read_csv(file, dtype={UID: str}, usecols=[UID, *T])
    ids = d[UID].tolist()
    if d[UID].isna().any() or len(set(ids)) != len(ids) or not set(expected_ids).issubset(set(ids)):
        raise ValueError('aquila_v1 must cover every EXP099 training study once')
    if set(ids) & set(gold_ids):
        raise ValueError('aquila_v1 includes Gold58 studies')
    values = d[T].to_numpy(np.float32)
    if not np.isfinite(values).all() or ((values < .02 - 1e-7) | (values > .98 + 1e-7)).any():
        raise ValueError('aquila_v1 values must be finite and clipped 0.02..0.98')
    labels, semantic = load_label_csv(file)
    if not set(expected_ids).issubset(labels):
        raise ValueError('Aquila target loading lost studies')
    return labels, {'path': AQUILA, 'sha256': sha256(file), 'semantic_sha256': semantic}


def five_head_initial_state(base, root=ROOT):
    """Hash the precise seed-42 initial multi-head state on CPU at registration."""
    import torch
    sys.path.insert(0, str(Path(root) / 'training/src'))
    from raptor_common import seed_all
    from raptor_radimagenet import build_radimagenet_model
    worker_init(0)
    seed_all(42)
    with torch.device('cpu'):
        model, _ = build_radimagenet_model(base['common_args']['arch'], aug.weights_path(root), base['init_weights']['sha256'])
        multi = build_multihead(model, 5)
    result, _ = state_hashes(multi)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU registration initialized CUDA')
    return result


def derived_csv(root, relative, ids, labels):
    import numpy as np
    dest = path(root, relative)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([UID, *T])
        for uid in ids:
            v = np.asarray(labels[uid], np.float32)
            if v.shape != (12,) or not np.isfinite(v).all() or ((v < 0) | (v > 1)).any():
                raise ValueError('derived label vector invalid')
            writer.writerow([uid, *[format(float(x), '.9g') for x in v]])
    loaded, rec = source_record(relative, root)
    if set(loaded) != set(ids):
        raise ValueError('derived label serialization changed UID coverage')
    return loaded, rec


def teacher_for(outer_fold, study_fold):
    if outer_fold not in (0, 1) or study_fold == outer_fold:
        raise ValueError('invalid outer/training fold')
    table = {0: {1: 'T234', 2: 'T134', 3: 'T124', 4: 'T123'},
             1: {0: 'T234', 2: 'T034', 3: 'T024', 4: 'T023'}}
    return table[outer_fold][study_fold]


def verify_run(run, spec, manifest_path, root=ROOT):
    """Require exact inventory, recipe, roster, and final-epoch provenance."""
    run, root = Path(run), Path(root)
    if run.name != spec['tag']:
        raise ValueError('run tag mismatch')
    inv = read(run / 'output-inventory.json')
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv'}
    if spec['heads']:
        required |= {f'oof_head_{h}{suffix}.csv' for h in spec['heads'] for suffix in ('', '_k36')}
    if set(inv) != required:
        raise ValueError('incomplete EXP149 run inventory')
    for name, item in inv.items():
        if Path(name).name != name or item['sha256'] != sha256(run / name):
            raise ValueError('run inventory changed: ' + name)
    summary, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    if (summary.get('mode') != 'full' or summary.get('optimizer_updates') != spec['steps']
            or summary.get('selected_epoch_1based') != 20 or summary.get('train') != spec['n_train']
            or summary.get('val') != spec['n_val'] or not 0 <= summary.get('skipped_updates', -1) <= spec['max_fp16_skips']):
        raise ValueError('run count/epoch/skip contract differs')
    expected = {'run': spec['tag'], 'train_folds': spec['train_folds'], 'eval_folds': spec['eval_folds'],
                'train_ids_ordered_sha256': spec['train_ids_sha256'], 'eval_ids_ordered_sha256': spec['eval_ids_sha256'],
                'label_file_sha256': spec['label_sha256'], 'label_semantic_sha256': spec['label_semantic_sha256'],
                'initial_state_sha256': spec['initial_state_sha256'], 'heads': spec['heads'],
                'protocol_sha256': sha256(manifest_path), 'gold_access': 'disabled', 'steps': spec['steps']}
    if any(prov.get(k) != v for k, v in expected.items()):
        raise ValueError('run provenance differs')
    if len(read(run / 'history.json')) != 20:
        raise ValueError('full run history incomplete')
    return {str(run / name): inv[name]['sha256'] for name in required}


def teacher_values(root=ROOT):
    """Read K36 teacher Effusion only after all seven runs pass inventory gates."""
    root = Path(root)
    p = _load_manifest('teacher', root)
    value = {}
    for tag, spec in p['runs'].items():
        verify_run(path(root, 'artifacts/experiments') / tag, spec, path(root, TEACHER_MANIFEST), root)
        ids = spec['ordered_ids']['val']
        pred = aug.e125.load_table(path(root, 'artifacts/experiments') / tag / 'oof_k36.csv', ids, prediction=True)
        value[spec['arm']] = {uid: float(pred[i, EFFUSION]) for i, uid in enumerate(ids)}
    return value


def l3_rows(canonical, fold_of, dread, aquila, teachers, outer):
    """Blend all targets and substitute leakage-safe K36 Effusion in training rows."""
    import numpy as np
    values = {}
    for uid in canonical:
        v = (.75 * dread[uid] + .25 * aquila[uid]).astype(np.float32)
        if fold_of[uid] != outer:
            teacher = teacher_for(outer, fold_of[uid])
            if (uid not in teachers.get(teacher, {}) or outer in TEACHERS[teacher][0]
                    or fold_of[uid] in TEACHERS[teacher][0]):
                raise ValueError('missing or leaking nested teacher value')
            value = teachers[teacher][uid]
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid nested teacher Effusion')
            v[EFFUSION] = value
        values[uid] = v
    return values


def build_l3_labels(root=ROOT):
    import numpy as np
    from tools import exp130_gold58_compare as g130
    root = Path(root)
    base = base_protocol(root)
    canonical, fold_of = fold_rosters(base, root)
    gold, _ = g130.official_gold(root)
    aquila, _ = strict_aquila(root, canonical, gold)
    dread, _ = source_record(base['paths']['labels'], root)
    teachers = teacher_values(root)
    all_values = {outer: l3_rows(canonical, fold_of, dread, aquila, teachers, outer) for outer in (0, 1)}
    outputs = {}
    for outer, values in all_values.items():
        rel = f'{PROTOCOL}/derived/l3_outer{outer}.csv'
        _, rec = derived_csv(root, rel, canonical, values)
        outputs[str(outer)] = rec
    exclusive_json(path(root, PROTOCOL + '/derived/l3-provenance.json'),
                   {'schema': 'exp149-l3-v1', 'teacher_manifest_sha256': sha256(path(root, TEACHER_MANIFEST)),
                    'teachers': {tag: sha256(path(root, 'artifacts/experiments') / tag / 'output-inventory.json') for tag in _load_manifest('teacher', root)['runs']},
                    'outer_labels': outputs})
    print(json.dumps({'status': 'L3_LABELS_BUILT', 'outer': outputs}, indent=2))
    return outputs


def prepare_students(root=ROOT):
    import numpy as np
    from tools import exp130_gold58_compare as g130
    root = Path(root)
    dest = path(root, STUDENT_MANIFEST)
    if dest.exists():
        raise FileExistsError(dest)
    base = base_protocol(root)
    canonical, fold_of = fold_rosters(base, root)
    gold, _ = g130.official_gold(root)
    aquila, aquila_rec = strict_aquila(root, canonical, gold)
    dread, dread_rec = source_record(base['paths']['labels'], root)
    l3_receipt = path(root, PROTOCOL + '/derived/l3-provenance.json')
    l3_records = read(l3_receipt) if l3_receipt.exists() else None
    if l3_records is None or l3_records['teacher_manifest_sha256'] != sha256(path(root, TEACHER_MANIFEST)):
        raise ValueError('L3 labels require validated nested teachers')
    for outer in (0, 1):
        l3rec = l3_records['outer_labels'][str(outer)]
        if sha256(path(root, l3rec['path'])) != l3rec['sha256']:
            raise ValueError('L3 label file changed')
    head_records = []
    for name, rel in (*mh.HEADS, ('aquila_v1', AQUILA)):
        _, rec = source_record(rel, root)
        head_records.append({'name': name, **rec})
    multi_hash = five_head_initial_state(base, root)
    blended = {uid: (.75 * dread[uid] + .25 * aquila[uid]).astype(np.float32) for uid in canonical}
    l2, l2_rec = derived_csv(root, PROTOCOL + '/derived/l2.csv', canonical, blended)
    runs = {}
    for outer in (0, 1):
        ordered = ordered_for(canonical, fold_of, [k for k in range(5) if k != outer], [outer])
        l3rec = l3_records['outer_labels'][str(outer)]
        if sha256(path(root, l3rec['path'])) != l3rec['sha256']:
            raise ValueError('L3 label file changed')
        l3, semantic = load_label_csv(path(root, l3rec['path']))
        if semantic != l3rec['semantic_sha256']:
            raise ValueError('L3 label semantics changed')
        for arm, labels, rec in (('L1', aquila, aquila_rec), ('L2', l2, l2_rec), ('L3', l3, l3rec), ('H1', dread, dread_rec)):
            tag = f'EXP149_{arm}_S42_F{outer}'
            heads = [h['name'] for h in head_records] if arm == 'H1' else []
            runs[tag] = _spec(tag, 'student', arm, [k for k in range(5) if k != outer], [outer], ordered,
                              labels, rec, base, heads=heads, head_labels=head_records if heads else [], root=root)
            if arm == 'H1':
                runs[tag]['initial_state_sha256'] = multi_hash
    p = _manifest(base, 'student', runs, root)
    p['multihead'] = {'loss': 'sum of five equally weighted masked per-head BCE terms',
                      'serving': 'per-target equal mean of five head average-ranks / roster size'}
    p['aquila'] = aquila_rec
    p['l3_provenance_sha256'] = sha256(l3_receipt)
    exclusive_json(dest, p)
    print(json.dumps({'status': p['status'], 'manifest': str(dest), 'students': list(runs)}, indent=2))
    return p


def paired_metric(candidate, control, reference, draws):
    """Macro AUC and paired component-bootstrap delta with finite-draw gate."""
    import numpy as np
    from tools import exp115_vlm_eval as v115
    y, mask, min_class = reference
    a = v115.auc_targets(y, candidate, mask, min_class)
    b = v115.auc_targets(y, control, mask, min_class)
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        raise ValueError('reference has no scorable targets')
    delta_draw = []
    for ii in draws:
        av = v115.auc_targets(y[ii], candidate[ii], mask[ii], min_class)
        bv = v115.auc_targets(y[ii], control[ii], mask[ii], min_class)
        if not np.isfinite(av).any() or not np.isfinite(bv).any():
            raise ValueError('bootstrap draw has no scorable targets')
        delta_draw.append(float(np.nanmean(av) - np.nanmean(bv)))
    ds = np.asarray(delta_draw, np.float64)
    if not np.isfinite(ds).all() or len(ds) != 300:
        raise ValueError('incomplete/nonfinite bootstrap draws')
    return {'candidate': float(np.nanmean(a)), 'control': float(np.nanmean(b)),
            'delta': float(np.nanmean(a) - np.nanmean(b)),
            'ci95': np.percentile(ds, [2.5, 97.5]).tolist(),
            'per_target_delta': {name: float(x) if np.isfinite(x) else None for name, x in zip(T, a - b)},
            'scored_targets': int(np.isfinite(a).sum()), 'bootstrap_draws': len(ds)}


def registered_references(ids, root=ROOT):
    import numpy as np
    from tools import exp125_radr50_k16_local as e125
    root = Path(root)
    jev = e125.load_table(root / JEV, ids)
    if not np.isfinite(jev).all() or not np.isin(jev, [0.0, 1.0]).all():
        raise ValueError('JEV HYBRID must be complete binary labels')
    # Its file may include other studies; the registered pooled cohort must all be covered.
    refs = {'jev_hybrid': (jev, np.ones(jev.shape, bool), 1)}
    raw = {name: e125.load_table(root / rel, ids) for name, rel in
           [('dread', mh.HEADS[0][1]), ('relabel_v1', mh.HEADS[1][1]),
            ('steven_v4', mh.HEADS[2][1]), ('pilkwang_v2', mh.HEADS[3][1]),
            ('flight_v4', e127.LABEL_PATHS['flight_v4'])]}
    for name in ('dread', 'flight_v4', 'relabel_v1'):
        x = raw[name]
        if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
            raise ValueError('incomplete report-derived reference: ' + name)
        y = np.rint(x.astype(np.float32)) if name == 'dread' else (x >= .5).astype(float)
        refs[name] = (y, np.ones(x.shape, bool), 1)
    stack = np.stack([raw[n] for n in ('dread', 'steven_v4', 'pilkwang_v2', 'relabel_v1')])
    finite = np.isfinite(stack).all(0)
    binary = stack >= .5
    agree = finite & (binary.all(0) | (~binary).all(0))
    refs['consensus'] = (binary[0].astype(float), agree, 2)
    aquila = e125.load_table(root / AQUILA, ids)
    if not np.isfinite(aquila).all():
        raise ValueError('aquila reference incomplete')
    refs['aquila_v1'] = ((aquila >= .5).astype(float), np.ones(aquila.shape, bool), 1)
    return refs


def control_tag(arm, fold):
    if arm == 'H1':
        return f'EXP138_BUNDLE_AUG1_MH_S42_F{fold}'
    return 'EXP127_AUG1_S42_F0' if fold == 0 else 'EXP134_AUG1_S42_F1'


def _verify_control(arm, fold, root=ROOT):
    from tools import exp138_bundle_aug1_mh as e138
    root = Path(root)
    tag = control_tag(arm, fold)
    run = root / 'artifacts/experiments' / tag
    if arm == 'H1':
        p = e138.load_protocol(42, root)
        e138.verify_run(run, p, fold, root)
    elif fold == 0:
        p = e127.load_protocol('AUG1', 42, root)
        e127.verify_run(run, p, sha256(e127.manifest_path('AUG1', 42, root)), root)
    else:
        p = aug.load_protocol(42, root)
        aug.verify_run(run, p, sha256(aug.manifest_path(42, root)), root)
    return run


def gold58_inference(tags, root=ROOT, device='cuda', workers=2):
    """Fresh plain-K36 predictions from each exact checkpoint; no TTA."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tools import exp130_gold58_compare as g130
    root = Path(root)
    sys.path.insert(0, str(root / 'training/src'))
    from raptor_common import Corpus, build_model
    from raptor_data import collate
    from raptor_exp127 import loader_factory
    from raptor_runtime import configure_runtime
    ids, binary = g130.official_gold(root)
    base = base_protocol(root)
    canonical, _ = fold_rosters(base, root)
    if set(ids) & set(canonical):
        raise ValueError('Gold58 overlaps EXP099 development roster')
    corpus = Corpus(root / base['paths']['corpus'])
    if not set(ids).issubset(set(corpus.ids.astype(str))):
        raise ValueError('Gold58 study absent from fixed corpus')
    configure_runtime('fp16' if device == 'cuda' else 'none', device, policy='strict')
    labels = {uid: np.zeros(12, np.float32) for uid in ids}
    windows = loader_factory('AUG1')
    ds = windows(corpus.dir, ids, labels, 224, 36, False, 'radimagenet', 42,
                 window_policy='within-slot-valid', input_policy='center3')
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=workers, collate_fn=collate)
    results, hashes = {}, {}
    for tag in tags:
        file = root / 'artifacts/experiments' / tag / 'model_best.pt'
        ckpt = torch.load(file, map_location='cpu', weights_only=False)
        if ckpt.get('tag') != tag or ckpt.get('k_eval') != 24 or ckpt.get('input_policy') != 'center3':
            raise ValueError('Gold checkpoint recipe/tag mismatch: ' + tag)
        model = build_model(ckpt['arch'], pretrained=False)
        heads = ckpt.get('heads', [])
        if heads:
            if len(heads) not in (4, 5):
                raise ValueError('unexpected Gold checkpoint head count')
            model = build_multihead(model, len(heads))
        model.load_state_dict(ckpt['model'], strict=True)
        model = model.to(device).eval()
        chunks, seen = [], []
        with torch.no_grad():
            for batch in dl:
                if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                    raise TypeError('invalid Gold batch')
                x, _, uids = batch
                with torch.autocast(device, dtype=torch.float16) if device == 'cuda' else __import__('contextlib').nullcontext():
                    logits = model(x.to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite Gold logits')
                chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
                seen.extend(uids)
        if seen != ids:
            raise ValueError('Gold inference UID order changed')
        prob = np.concatenate(chunks)
        if heads:
            prob = rank_mean(prob)
        if prob.shape != (58, 12) or not np.isfinite(prob).all():
            raise ValueError('Gold inference shape/values invalid')
        results[tag] = prob
        hashes[tag] = sha256(file)
        del ckpt, model
        if device == 'cuda':
            torch.cuda.empty_cache()
    return ids, np.asarray(binary, float), results, hashes


def screen_markdown(metrics):
    focus = ('Effusion', 'MCL', 'PF OA', 'Synovitis')
    lines = ['# EXP149 two-fold label-source screen', '',
             'All OOF predictions use plain K36 and per-fold target ranks. Gold58 is exposed and report-only.', '',
             '| Arm | Verdict | JEV delta | JEV CI95 | F0 JEV delta | F1 JEV delta | Consensus delta | Gold58 delta |',
             '|---|---|---:|---|---:|---:|---:|---:|']
    for arm in ARMS:
        x = metrics[arm]
        j = x['pooled']['jev_hybrid']
        lines.append(f"| {arm} | {x['status']} | {j['delta']:+.5f} | {j['ci95']} | "
                     f"{x['per_fold']['0']['jev_hybrid']['delta']:+.5f} | "
                     f"{x['per_fold']['1']['jev_hybrid']['delta']:+.5f} | "
                     f"{x['pooled']['consensus']['delta']:+.5f} | "
                     f"{x['gold58_exposed_report_only']['two_fold_mean_delta']:+.5f} |")
    lines += ['', '## Focus targets', '',
              '| Arm | Reference | ' + ' | '.join(focus) + ' |',
              '|---|---|' + '|'.join(['---:'] * len(focus)) + '|']
    for arm in ARMS:
        x = metrics[arm]
        for name, values in [('JEV', x['pooled']['jev_hybrid']['per_target_delta']),
                             ('Consensus', x['pooled']['consensus']['per_target_delta']),
                             ('Gold58 exposed', x['gold58_exposed_report_only']['per_target_delta'])]:
            lines.append('| ' + arm + ' | ' + name + ' | ' + ' | '.join(
                'NA' if values[t] is None else f'{values[t]:+.5f}' for t in focus) + ' |')
    lines += ['', 'Full per-target deltas, component-bootstrap intervals, biased references, and per-fold Gold58 deltas are in `result.json`.', '']
    return '\n'.join(lines)


def screen(root=ROOT, device='cuda', workers=2):
    import numpy as np
    from tools import exp125_radr50_k16_local as e125
    from tools import exp138_bundle_aug1_mh as e138
    from tools import exp130_gold58_compare as g130
    root = Path(root)
    out = root / PROTOCOL / 'screen'
    if out.exists():
        raise FileExistsError('screen already exists: ' + str(out))
    p = _load_manifest('student', root)
    canonical, fold_of = fold_rosters(p, root)
    fold_ids = {f: [u for u in canonical if fold_of[u] == f] for f in (0, 1)}
    ids = fold_ids[0] + fold_ids[1]
    refs = registered_references(ids, root)
    draws = e125.component_draws(ids, read(root / COMPONENTS), draws=300, seed=20261003)
    # Bind all full student and matched-control outputs before costly fresh inference.
    for tag, spec in p['runs'].items():
        verify_run(root / 'artifacts/experiments' / tag, spec, root / STUDENT_MANIFEST, root)
    for arm in ARMS:
        for fold in (0, 1):
            _verify_control(arm, fold, root)
    unique_tags = list(p['runs']) + list(dict.fromkeys(control_tag(arm, f) for arm in ARMS for f in (0, 1)))
    gold_ids, gold_y, gold_pred, gold_hashes = gold58_inference(unique_tags, root, device, workers)
    all_metrics = {}
    for arm in ARMS:
        cand_folds, ctrl_folds = [], []
        fold_metrics = {}
        for fold in (0, 1):
            tag = f'EXP149_{arm}_S42_F{fold}'
            ctag = control_tag(arm, fold)
            fid = fold_ids[fold]
            a = e125.load_table(root / 'artifacts/experiments' / tag / 'oof_k36.csv', fid, prediction=True)
            b = e125.load_table(root / 'artifacts/experiments' / ctag / 'oof_k36.csv', fid, prediction=True)
            cand_folds.append(e138.fold_ranks(a))
            ctrl_folds.append(e138.fold_ranks(b))
            fold_ref = registered_references(fid, root)
            fold_metric = {}
            for name, ref in fold_ref.items():
                from tools import exp115_vlm_eval as v115
                av = v115.auc_targets(ref[0], cand_folds[-1], ref[1], ref[2])
                bv = v115.auc_targets(ref[0], ctrl_folds[-1], ref[1], ref[2])
                fold_metric[name] = {'delta': float(np.nanmean(av) - np.nanmean(bv)),
                                     'per_target_delta': {t: float(d) if np.isfinite(d) else None for t, d in zip(T, av - bv)}}
            fold_metrics[str(fold)] = fold_metric
        pooled_a, pooled_b = np.concatenate(cand_folds), np.concatenate(ctrl_folds)
        metrics = {name: paired_metric(pooled_a, pooled_b, ref, draws) for name, ref in refs.items()}
        gold_a = np.mean([gold_pred[f'EXP149_{arm}_S42_F{f}'] for f in (0, 1)], axis=0)
        gold_b = np.mean([gold_pred[control_tag(arm, f)] for f in (0, 1)], axis=0)
        from tools import exp115_vlm_eval as v115
        gold_ref = (gold_y, np.ones(gold_y.shape, bool), 1)
        ga = v115.auc_targets(gold_y, gold_a, gold_ref[1], gold_ref[2])
        gb = v115.auc_targets(gold_y, gold_b, gold_ref[1], gold_ref[2])
        gold_delta = float(np.nanmean(ga) - np.nanmean(gb))
        gold_fold = {}
        for f in (0, 1):
            aa = v115.auc_targets(gold_y, gold_pred[f'EXP149_{arm}_S42_F{f}'], gold_ref[1], 1)
            bb = v115.auc_targets(gold_y, gold_pred[control_tag(arm, f)], gold_ref[1], 1)
            gold_fold[str(f)] = {'delta': float(np.nanmean(aa) - np.nanmean(bb)),
                                  'per_target_delta': {t: float(d) if np.isfinite(d) else None for t, d in zip(T, aa - bb)}}
        gates = {'jev_pooled_positive': metrics['jev_hybrid']['delta'] > 0,
                 'jev_both_folds_nonnegative': all(fold_metrics[str(f)]['jev_hybrid']['delta'] >= 0 for f in (0, 1)),
                 'consensus_floor': metrics['consensus']['delta'] >= -.003,
                 'gold58_floor': gold_delta >= -.005}
        all_metrics[arm] = {'status': 'ADVANCES' if all(gates.values()) else 'HOLD', 'gates': gates,
                            'pooled': metrics, 'per_fold': fold_metrics,
                            'gold58_exposed_report_only': {'two_fold_mean_delta': gold_delta,
                                'per_target_delta': {t: float(d) if np.isfinite(d) else None for t, d in zip(T, ga - gb)},
                                'per_fold': gold_fold}}
    out.mkdir(parents=True)
    gold_dir = out / 'gold58'
    gold_dir.mkdir()
    for tag in unique_tags:
        g130.write_predictions(gold_dir / f'{tag}.csv', gold_ids, gold_pred[tag])
    exclusive_json(out / 'result.json', {'schema': 'exp149-label-sources-screen-v1',
                    'student_manifest_sha256': sha256(root / STUDENT_MANIFEST),
                    'teacher_manifest_sha256': sha256(root / TEACHER_MANIFEST),
                    'pooled_studies': len(ids), 'folds': [0, 1], 'k_eval': 36,
                    'bootstrap': {'component_draws': 300, 'seed': 20261003},
                    'reference_sha256': {rel: sha256(root / rel) for rel in
                        (JEV, AQUILA, COMPONENTS, e127.LABEL_PATHS['dread'],
                         e127.LABEL_PATHS['flight_v4'], e127.LABEL_PATHS['relabel_v1'],
                         e127.LABEL_PATHS['steven_v4'], e127.LABEL_PATHS['pilkwang_v2'])},
                    'oof_k36_sha256': {tag: sha256(root / 'artifacts/experiments' / tag / 'oof_k36.csv') for tag in unique_tags},
                    'gold58': {'studies': len(gold_ids), 'checkpoint_sha256': gold_hashes,
                               'plain_k36': True, 'exposed_report_only': True},
                    'metrics': all_metrics})
    with (out / 'result.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write(screen_markdown(all_metrics))
    print(json.dumps({arm: all_metrics[arm]['status'] for arm in ARMS}, indent=2))
    return all_metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare-teachers')
    sub.add_parser('validate-teachers')
    sub.add_parser('build-l3-labels')
    sub.add_parser('prepare-students')
    sub.add_parser('validate-students')
    for command in ('queue-teachers', 'queue-students'):
        q = sub.add_parser(command)
        q.add_argument('--after', required=True)
        q.add_argument('--out', required=True)
        q.add_argument('--workers', type=int, default=12)
        q.add_argument('--tags', help='comma-separated registered tags; teacher shorthand T234 etc. accepted for lane splitting')
    s = sub.add_parser('screen')
    s.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    s.add_argument('--workers', type=int, default=2)
    smoke = sub.add_parser('gpu-smoke')
    smoke.add_argument('--run', required=True)
    a = ap.parse_args(argv)
    if a.command == 'prepare-teachers':
        prepare_teachers()
    elif a.command == 'validate-teachers':
        validate_cli('teacher')
    elif a.command == 'build-l3-labels':
        build_l3_labels()
    elif a.command == 'prepare-students':
        prepare_students()
    elif a.command == 'validate-students':
        validate_cli('student')
    elif a.command.startswith('queue-'):
        kind = 'teacher' if a.command == 'queue-teachers' else 'student'
        entries = queue_entries(kind, a.after, workers=a.workers,
                                tags=a.tags.split(',') if a.tags else None)
        exclusive_json(Path(a.out), entries)
        print(json.dumps({'queue': a.out, 'tags': [e['tag'] for e in entries]}, indent=2))
    elif a.command == 'screen':
        screen(device=a.device, workers=a.workers)
    elif a.command == 'gpu-smoke':
        kind = 'teacher' if a.run.startswith('EXP149_T') else 'student'
        p = _load_manifest(kind)
        if a.run not in p['runs']:
            raise ValueError('unregistered smoke run')
        out = path(ROOT, f'.local/exp149-smoke/{a.run}')
        if out.exists():
            raise FileExistsError('smoke scratch exists; choose a clean scratch directory')
        argv = runner_argv(p, a.run, workers=0, mode='smoke', output_root=out)
        result = subprocess.run(argv, cwd=ROOT, env=aug.e113.local_env(), check=False)
        if result.returncode:
            raise RuntimeError('GPU smoke failed; see scratch output')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
