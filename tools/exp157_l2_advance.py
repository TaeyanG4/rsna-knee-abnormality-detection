"""Offline EXP157 L2 registration, CPU checks, local queue, and run verification.

This module never starts a GPU job. The frozen EXP149 runner is executed by a
new wrapper which substitutes only the registration and the saved initial state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = '1.0.0'
EXP = 'artifacts/experiments/EXP157_L2_ADVANCE'
PROTOCOL = EXP + '/PROTOCOL.md'
MANIFEST = EXP + '/manifest.json'
INITIAL = EXP + '/initial-state.pt'
SOURCE = 'artifacts/experiments/EXP149_LABEL_SOURCES/students/manifest.json'
LABEL = 'artifacts/experiments/EXP149_LABEL_SOURCES/derived/l2.csv'
RUNNER = 'training/runners/exp157_l2.py'
FROZEN_RUNNER = 'training/runners/exp149_label_sources.py'
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
NEW = tuple((s, f) for s in (42, 43) for f in range(5) if s == 43 or f >= 2)
ALL = tuple((s, f) for s in (42, 43) for f in range(5))
INITIAL_TENSOR_SHA = 'ce44133b637ef8260578c3faa6ec9ffb76b578bf60f730eb45f7c1db5e8949c1'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def tag(seed, fold):
    return f'EXP157_L2_S{seed}_F{fold}'


def source_tag(fold):
    return f'EXP149_L2_S42_F{fold}'


def ensemble_tag(seed, fold):
    return source_tag(fold) if seed == 42 and fold in (0, 1) else tag(seed, fold)


def control_tag(seed, fold):
    return (f'EXP127_AUG1_S{seed}_F0' if fold == 0 else
            f'EXP134_AUG1_S{seed}_F{fold}')


def source_registration(root=ROOT):
    from tools import exp149_label_sources as e149
    root = Path(root)
    p = e149._load_manifest('student', root)
    first = p['runs'][source_tag(0)]
    need(p['initial_state_sha256'] == INITIAL_TENSOR_SHA and
         first['initial_state_sha256'] == INITIAL_TENSOR_SHA and
         first['labels'] == LABEL and not first['heads'] and
         first['label_set'] == 'l2', 'EXP149 L2 registration differs')
    need(sha256(root / LABEL) == first['label_sha256'], 'EXP149 L2 label bytes differ')
    for fold in (0, 1):
        s = p['runs'][source_tag(fold)]
        need(s['labels'] == LABEL and s['label_sha256'] == first['label_sha256'] and
             s['initial_state_sha256'] == INITIAL_TENSOR_SHA and not s['heads'],
             'Existing L2 fold registration differs')
    return p


def build_initial_file(root, source):
    """Seed-42 CPU construction reproduces EXP149's one-head initial tensors."""
    import torch
    from tools import exp134_aug1_kfold as aug
    sys.path.insert(0, str(Path(root) / 'training/src'))
    from raptor_common import seed_all
    from raptor_radimagenet import build_radimagenet_model
    root = Path(root)
    path = root / INITIAL
    need(not path.exists(), 'EXP157 initial-state file exists')
    aug.forbid_cuda_init()
    aug.worker_init(0)
    seed_all(42)
    with torch.device('cpu'):
        model, report = build_radimagenet_model(source['common_args']['arch'],
                                                aug.weights_path(root),
                                                source['init_weights']['sha256'])
    state_sha, _ = aug.state_hashes(model)
    need(state_sha == INITIAL_TENSOR_SHA and
         report['weights_sha256'] == source['init_weights']['sha256'],
         'EXP149 initial tensor or weight source differs')
    need(not torch.cuda.is_initialized(), 'CPU prepare initialized CUDA')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)
    return sha256(path)


def prepare(root=ROOT):
    from tools import exp149_label_sources as e149
    from tools import exp130_gold58_compare as gold
    root = Path(root)
    need(not (root / MANIFEST).exists() and not (root / INITIAL).exists(),
         'EXP157 already prepared; no overwrite')
    source = source_registration(root)
    canonical, fold_of = e149.fold_rosters(source, root)
    gold_ids, _ = gold.official_gold(root)
    need(not set(canonical) & set(gold_ids), 'Gold58 intersects EXP099 training roster')
    labels, label_record = e149.source_record(LABEL, root)
    need(label_record['sha256'] == source['runs'][source_tag(0)]['label_sha256'] and
         label_record['semantic_sha256'] == source['runs'][source_tag(0)]['label_semantic_sha256'],
         'L2 file content or semantics changed')
    runs = {}
    for seed, fold in NEW:
        train_folds = [f for f in range(5) if f != fold]
        ordered = e149.ordered_for(canonical, fold_of, train_folds, [fold])
        spec = e149._spec(tag(seed, fold), 'student', 'L2', train_folds,
                          [fold], ordered, labels, label_record, source, root=root)
        spec['seed'] = seed
        need(spec['initial_state_sha256'] == INITIAL_TENSOR_SHA and
             spec['labels'] == LABEL and spec['n_train'] + spec['n_val'] == 4343,
             'New L2 fold contract differs')
        runs[spec['tag']] = spec
    for fold in (0, 1):
        old = source['runs'][source_tag(fold)]
        ordered = e149.ordered_for(canonical, fold_of,
                                   [f for f in range(5) if f != fold], [fold])
        need(old['ordered_ids'] == ordered, 'Existing L2 roster differs')
        e149.verify_run(root / 'artifacts/experiments' / source_tag(fold),
                        old, root / SOURCE, root)
    initial_sha = build_initial_file(root, source)
    p = copy.deepcopy(source)
    p.update(schema='exp157-l2-advance-v1', version=VERSION,
             status='REGISTERED_NO_EXECUTION', experiment='EXP157_L2_ADVANCE',
             seed=42, run_kind='student', runs=runs)
    p['exp157'] = {'source_manifest_sha256': sha256(root / SOURCE),
                   'protocol_sha256': sha256(root / PROTOCOL),
                   'initial_state_file': INITIAL,
                   'initial_state_file_sha256': initial_sha,
                   'initial_state_tensor_sha256': INITIAL_TENSOR_SHA,
                   'label_file': LABEL, 'label_file_sha256': label_record['sha256'],
                   'prior_tags': [source_tag(0), source_tag(1)],
                   'new_tags': list(runs), 'controls':
                   [control_tag(seed, fold) for seed, fold in ALL],
                   'all_seeds': [42, 43], 'fixed_final_epochs': 20,
                   'independent_trajectories': True}
    p['source_bindings'][SOURCE] = sha256(root / SOURCE)
    for name in (PROTOCOL, RUNNER, 'tools/exp157_l2_advance.py'):
        p['source_bindings'][name] = sha256(root / name)
    write_new(root / MANIFEST, p)
    return {'status': p['status'], 'manifest': MANIFEST,
            'new_runs': list(runs), 'initial_state_file_sha256': initial_sha,
            'initial_state_tensor_sha256': INITIAL_TENSOR_SHA}


def registration(root=ROOT):
    root = Path(root)
    p = read(root / MANIFEST)
    need(p['schema'] == 'exp157-l2-advance-v1' and p['version'] == VERSION and
         tuple(p['runs']) == tuple(tag(s, f) for s, f in NEW),
         'EXP157 manifest differs')
    x = p['exp157']
    need(x['source_manifest_sha256'] == sha256(root / SOURCE) and
         x['protocol_sha256'] == sha256(root / PROTOCOL) and
         x['initial_state_file_sha256'] == sha256(root / INITIAL) and
         x['initial_state_tensor_sha256'] == INITIAL_TENSOR_SHA and
         x['label_file_sha256'] == sha256(root / LABEL),
         'EXP157 source, label or initial state differs')
    for name, digest in p['source_bindings'].items():
        need(sha256(root / name) == digest, 'Bound EXP157 source differs: ' + name)
    return p


def validate_arguments(a, root=ROOT):
    """Guard used only inside the frozen runner's process-local import shim."""
    from tools import exp149_label_sources as e149
    root = Path(root)
    p = registration(root)
    need(a.run in p['runs'] and a.tag == a.run and
         Path(a.protocol).resolve() == (root / MANIFEST).resolve(),
         'EXP157 run tag or manifest differs')
    s = p['runs'][a.run]
    need(a.run == tag(a.seed, a.fold) and (a.seed, a.fold) in NEW and
         a.variant == 'AUG1' and a.mode in ('full', 'smoke'),
         'EXP157 seed, fold or mode differs')
    need(a.device == ('cpu' if a.validate_only else 'cuda') and
         not os.environ.get('KNEE_REMOTE_RELAX_ENV') and 0 <= a.workers <= 12,
         'EXP157 device, workers or environment differs')
    for key, value in p['common_args'].items():
        if key == 'device' and a.validate_only:
            continue
        need(getattr(a, key) == value, 'EXP157 recipe differs: ' + key)
    need(Path(a.corpus).resolve() == (root / CORPUS).resolve() and
         Path(a.labels).resolve() == (root / LABEL).resolve() and
         Path(a.eval_labels).resolve() == (root / p['paths']['eval_labels']).resolve() and
         Path(a.init_weights).resolve() == e149.aug.weights_path(root).resolve(),
         'EXP157 input path differs')
    need(sha256(a.init_weights) == p['init_weights']['sha256'],
         'RadImageNet initial weight bytes differ')
    canonical, fold_of = e149.fold_rosters(p, root)
    ordered = e149.ordered_for(canonical, fold_of, s['train_folds'], s['eval_folds'])
    need(ordered == s['ordered_ids'] and
         e149.uid_hash(ordered['train']) == s['train_ids_sha256'] and
         e149.uid_hash(ordered['val']) == s['eval_ids_sha256'] and
         s['labels'] == LABEL and s['label_sha256'] == sha256(root / LABEL) and
         s['initial_state_sha256'] == INITIAL_TENSOR_SHA and not s['heads'],
         'EXP157 roster, label or initial-state registration differs')
    out = root / 'artifacts/experiments'
    if a.mode == 'full':
        need(a.output_root is None or Path(a.output_root).resolve() == out.resolve(),
             'Full run output root differs')
        need(not (out / a.run).exists(), 'Full run output already exists')
    else:
        need(a.output_root is not None and
             Path(a.output_root).resolve().is_relative_to((root / '.local').resolve()) and
             Path(a.output_root).resolve() != out.resolve() and
             not (Path(a.output_root) / a.run).exists(),
             'Smoke requires a fresh .local output')
        out = Path(a.output_root)
    return p, s, ordered, out


def runner_argv(p, run, root=ROOT, workers=12, *, validate_only=False,
                smoke_output=None):
    from tools import exp149_label_sources as e149
    root = Path(root)
    s = p['runs'][run]
    source = source_registration(root)
    argv = e149.runner_argv(source, source_tag(0), root, workers)
    argv[1] = str(root / RUNNER)
    for flag, value in (('--protocol', root / MANIFEST), ('--run', run),
                        ('--tag', run), ('--fold', s['fold']), ('--seed', s['seed'])):
        argv = e149.aug._set_flag(argv, flag, str(value))
    argv += ['--initial-state', str(root / INITIAL)]
    if smoke_output is not None:
        argv = e149.aug._set_flag(argv, '--mode', 'smoke')
        argv += ['--output-root', str(smoke_output), '--smoke-train', '8',
                 '--smoke-val', '4', '--smoke-epochs', '1']
    if validate_only:
        argv = e149.aug._set_flag(argv, '--device', 'cpu') + ['--validate-only']
    return argv


def validate(root=ROOT):
    root = Path(root)
    p = registration(root)
    source_registration(root)
    for run, spec in p['runs'].items():
        need(spec['seed'] in (42, 43) and spec['fold'] in range(5) and
             spec['n_train'] + spec['n_val'] == 4343 and
             spec['steps'] == spec['scheduler_total_steps'],
             'EXP157 fold/step contract differs: ' + run)
        argv = runner_argv(p, run, root, workers=0, validate_only=True)
        argv[0] = sys.executable
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUTF8='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
        result = subprocess.run(argv, cwd=root, env=env, check=False)
        need(result.returncode == 0, 'CPU runner validation failed: ' + run)
    return {'status': 'VALIDATED_NO_TRAINING', 'runs': len(p['runs'])}


def gpu_smoke(out, after='none', run=None, root=ROOT):
    """Write a one-batch queue plan; B alone may launch it on the GPU."""
    root, out = Path(root), Path(out)
    p = registration(root)
    run = run or tag(*NEW[0])
    need(run in p['runs'] and not out.exists(), 'Smoke tag or queue output invalid')
    smoke_root = root / '.local/exp157-l2-smoke'
    need(not (smoke_root / run).exists(), 'Smoke output exists; use a fresh run')
    command = runner_argv(p, run, root, workers=0, smoke_output=smoke_root)
    entry = {'tag': 'EXP157_L2_GPU_SMOKE', 'command': command,
             'continue_on_failure': False}
    if after.lower() != 'none':
        entry['after'] = after
    write_new(out, [entry])
    return {'status': 'SMOKE_PLAN_NO_EXECUTION', 'path': str(out), 'after': after,
            'output_root': str(smoke_root)}


def local_queue(out, after='none', workers=12, lane='all', root=ROOT):
    """Serial one-GPU queue; lane selects one seed group for controlled chaining."""
    root, out = Path(root), Path(out)
    p = registration(root)
    need(not out.exists() and 0 <= workers <= 12 and lane in ('all', 's42', 's43'),
         'Queue output, workers or lane invalid')
    selected = [(s, f) for s, f in NEW if lane == 'all' or lane == f's{s}']
    need(selected, 'Empty EXP157 queue lane')
    previous = None if after.lower() == 'none' else after
    entries = []
    for seed, fold in selected:
        run = tag(seed, fold)
        entry = {'tag': run, 'command': runner_argv(p, run, root, workers),
                 'continue_on_failure': False}
        if previous:
            entry['after'] = previous
        entries.append(entry)
        previous = run
    write_new(out, entries)
    return {'status': 'QUEUED_NO_EXECUTION', 'path': str(out), 'lane': lane,
            'runs': len(entries), 'last_tag': previous}


def verify_run(run_dir, seed, fold, root=ROOT):
    """Verify inventory, full completion, recipe, source and OOF roster on CPU."""
    import csv
    from tools import exp149_label_sources as e149
    root, run_dir = Path(root), Path(run_dir)
    prior = seed == 42 and fold in (0, 1)
    p = source_registration(root) if prior else registration(root)
    name = source_tag(fold) if prior else tag(seed, fold)
    spec = p['runs'][name]
    manifest = root / (SOURCE if prior else MANIFEST)
    e149.verify_run(run_dir, spec, manifest, root)
    inv = read(run_dir / 'output-inventory.json')
    summary, prov, history = (read(run_dir / name) for name in
                              ('summary.json', 'provenance.json', 'history.json'))
    need(summary['provenance'] == prov and summary['history'] == history and
         summary['run'] == name and summary['fold'] == fold and
         summary['gold'] == 0 and summary['gold_access'] == 'disabled' and
         summary['selected_checkpoint'] == 'fixed-final' and
         summary['skipped_updates'] == 0 and
         summary['worker_cleanup']['clean'] is True and
         summary['optimizer_updates'] == spec['steps'] and
         [h['epoch'] for h in history] == list(range(20)) and
         all(h['fp16_skips'] == 0 for h in history),
         'EXP157 full completion/zero-skip gate differs: ' + name)
    need(prov['seed'] == seed and prov['fold'] == fold and
         prov['full_model_initial_state_sha256'] == INITIAL_TENSOR_SHA and
         prov['initial_state_matches_registered'] is True and
         prov['environment_matches_baseline'] is True and
         prov['label_file'] == LABEL and prov['head_label_files'] == {},
         'EXP157 provenance differs: ' + name)
    args = summary['args']
    need(args['seed'] == seed and args['fold'] == fold and args['run'] == name and
         args['tag'] == name and args['mode'] == 'full' and args['variant'] == 'AUG1' and
         all(args.get(k) == v for k, v in p['common_args'].items()),
         'EXP157 recorded recipe differs: ' + name)
    if not prior:
        need(prov['init_weights']['initial_state_file_sha256'] ==
             p['exp157']['initial_state_file_sha256'] and
             prov['init_weights']['full_model_initial_state_sha256'] == INITIAL_TENSOR_SHA and
             prov['runner_sha256'] == p['source_bindings'][FROZEN_RUNNER],
             'EXP157 wrapper initial-file provenance differs: ' + name)
    for filename in ('oof_best.csv', 'oof_k36.csv'):
        with (run_dir / filename).open(newline='', encoding='utf-8') as stream:
            rows = csv.DictReader(stream)
            need(rows.fieldnames == [e149.UID, *e149.T], 'OOF column order differs')
            ids = [row[e149.UID] for row in rows]
        need(ids == spec['ordered_ids']['val'], 'OOF study order differs: ' + name)
    return {'tag': name, 'checkpoint_sha256': inv['model_best.pt']['sha256'],
            'zero_skips': True}


def verify(root=ROOT):
    root = Path(root)
    registration(root)
    checks = [verify_run(root / 'artifacts/experiments' / ensemble_tag(s, f),
                         s, f, root) for s, f in ALL]
    need(len({x['checkpoint_sha256'] for x in checks}) == 10,
         'Ten checkpoints are not distinct')
    return {'status': 'VERIFIED_SUBMISSION_INPUTS', 'runs': 10, 'checks': checks}


def collect(out, root=ROOT):
    """Record checked local outputs after all ten runs exist; no moving files."""
    root, out = Path(root), Path(out)
    need(not out.exists(), 'Use a fresh collection receipt')
    checks = verify(root)
    write_new(out, {'schema': 'exp157-l2-collection-v1',
                    'manifest_sha256': sha256(root / MANIFEST), **checks})
    return {'status': 'COLLECTED_VERIFIED', 'receipt': str(out), 'runs': 10}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'validate', 'verify'):
        sub.add_parser(name)
    smoke = sub.add_parser('gpu-smoke')
    smoke.add_argument('--out', type=Path, required=True)
    smoke.add_argument('--after', default='none')
    smoke.add_argument('--run')
    queue = sub.add_parser('local-queue')
    queue.add_argument('--out', type=Path, required=True)
    queue.add_argument('--after', default='none')
    queue.add_argument('--workers', type=int, default=12)
    queue.add_argument('--lane', choices=('all', 's42', 's43'), default='all')
    receipt = sub.add_parser('collect')
    receipt.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if a.command == 'prepare':
        result = prepare()
    elif a.command == 'validate':
        result = validate()
    elif a.command == 'gpu-smoke':
        result = gpu_smoke(a.out, a.after, a.run)
    elif a.command == 'local-queue':
        result = local_queue(a.out, a.after, a.workers, a.lane)
    elif a.command == 'collect':
        result = collect(a.out)
    else:
        result = verify()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
