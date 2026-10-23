"""Legacy artifact compatibility v2.0.0.

Changelog 2.0.0: retire source generation and fresh launches; retain historical
validation, collection and scientific readouts. Timing is measurement only.

EXP187 seed-43 finding-MIL production folds; CPU stage/collect, T4 train only."""
from __future__ import annotations

import argparse
import copy
import csv
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = '2.0.0'
HISTORICAL_VERSION = '1.0.0'
SCHEMA = 'exp187-mil-members-s43-v1'
EXP = 'artifacts/experiments/EXP187_MIL_MEMBERS_S43'
PROTOCOL = EXP + '/PROTOCOL.md'
TOOL = 'tools/exp187_mil_members_s43.py'
RUNNER = 'training/runners/exp187_mil_members_s43.py'
MODEL = 'training/src/raptor_exp185.py'
SOURCE_PROTOCOL = '.local/exp179-overlay/protocols/M43.json'
SOURCE_MANIFEST = '.local/exp179-overlay/manifest.json'
FOLD_SOURCE = 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/manifest.json'
BASE_ARCHIVE = '.local/exp166-vast-stage-v1/package/exp166-code.zip.bin'
BASE_SHA = 'ecac612702c5ac72ca50a7af6a96ff32375cdc025f82f2b2d094a18a0884ac50'
BASE_NAME = 'exp166-code.zip.bin'
OVERLAY_NAME = 'exp187-overlay.zip.bin'
BASE_DATASET = 'taeyangg4/knee-exp184-aquila-share-private'
DATASET = 'taeyangg4/knee-exp187-mil-s43-private'
CORPUS_DATASET = 'taeyangg4/knee-raptor-corpus-336'
KERNELS = {'01': (0, 1), '23': (2, 3), '4': (4,)}
WORKSPACE = '/tmp/exp187-workspace'
RESULTS = '/tmp/exp187-results'
FOLD_CAP = 9900
KERNEL_CAP = 30600
WORKERS = 4
OUTPUTS = ('summary.json', 'provenance.json', 'history.json', 'model_best.pt',
           'oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv')
UID = 'StudyInstanceUID'
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
REUSED_OVERLAY = ('tools/exp179_multiscale.py', 'tools/exp170_convnext_t_l2.py',
                  'training/src/raptor_exp179.py', 'config/workspace.toml',
                  'storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv',
                  'storage/labels/derived_constellation/l2_dread75_aquila25_v1.provenance.json')
REUSED_BASE = ('artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold0/ordered-ids.json',)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def safe(name):
    p = PurePosixPath(name)
    need(name and p.as_posix() == name and not p.is_absolute() and
         '..' not in p.parts and ':' not in name and '\\' not in name, 'unsafe archive path')
    return name


def tag(fold):
    need(fold in range(5), 'unknown EXP187 fold')
    return f'EXP187_MIL_S43_F{fold}'


def protocol_path(fold):
    return f'{EXP}/folds/fold{fold}.json'


def roster_path(fold):
    return f'{EXP}/rosters/fold{fold}.json' if fold else \
        'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold0/ordered-ids.json'




def dataset_factory(value):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL dataset_factory is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def evaluate_flip_tta(model, corpus, ids, labels, a, dev, eval_dtype, autocast):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL evaluate_flip_tta is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def stamp_outputs(out, protocol):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL stamp_outputs is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def runner_source(root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL runner_source is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def build_protocol(fold, root=ROOT, runner_hash=None):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL build_protocol is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def _payload(root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL _payload is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def _verify_reused_base(archive, root):
    """Reuse only pinned overlay inputs whose bytes are already in EXP166."""
    root = Path(root)
    source = read(root / SOURCE_MANIFEST)['files']
    for name in REUSED_BASE:
        entry = source[name]
        path = root / '.local/exp179-overlay' / name
        need(entry['origin'] == 'overlay' and sha(path) == entry['sha256'],
             'EXP179 overlay input differs: ' + name)
        need(name in archive.namelist() and
             hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256'],
             'EXP166 reused input differs: ' + name)


def kernel_id(group):
    need(group in KERNELS, 'unknown kernel group')
    return f'taeyangg4/exp187-mil-s43-f{group}-t4'


def kernel_metadata(group):
    from tools import exp150_consensus_weight as corpus
    return dict(id=kernel_id(group), title=f'EXP187 MIL seed 43 folds {group}',
                code_file='run.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=True, enable_tpu=False,
                enable_internet=False, machine_shape='NvidiaTeslaT4',
                docker_image=corpus.DOCKER,
                dataset_sources=[CORPUS_DATASET, BASE_DATASET, DATASET],
                competition_sources=[],
                kernel_sources=[], model_sources=[])


def notebook_source(group, overlay_sha):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL notebook_source is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def stage(out, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL stage is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def verify_stage(out, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL verify_stage is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def validate_arguments(a, root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL validate_arguments is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def runner_args(fold, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL runner_args is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def verify_workspace(root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL verify_workspace is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def link_corpus(root, input_root):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL link_corpus is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def _prediction_values(path, ids):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = csv.DictReader(stream)
        need(rows.fieldnames == [UID, *TARGETS], 'OOF columns differ')
        values = list(rows)
    need([row[UID] for row in values] == list(ids) and len(set(ids)) == len(ids),
         'OOF study IDs differ')
    for row in values:
        for target in TARGETS:
            value = float(row[target])
            need(math.isfinite(value) and 0 <= value <= 1, 'OOF probability invalid')


def verify_run(run, fold, root=ROOT, gpu='T4'):
    root, run = Path(root), Path(run)
    q = read(root / protocol_path(fold))
    info = q['folds'][str(fold)]
    need(run.name == tag(fold) and q['exp187']['fold'] == fold,
         'EXP187 run identity differs')
    inv = read(run / 'output-inventory.json')
    need(set(inv) == set(OUTPUTS), 'incomplete EXP187 output inventory')
    for name, entry in inv.items():
        need(safe(name) == name and (run / name).is_file() and
             (run / name).stat().st_size == entry['bytes'] and
             sha(run / name) == entry['sha256'], 'EXP187 output differs: ' + name)
    summary, prov, history = (read(run / name) for name in
                              ('summary.json', 'provenance.json', 'history.json'))
    need(summary['fold'] == fold and summary['train'] == info['n_train'] and
         summary['val'] == summary['rows'] == info['n_val'] and
         summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == info['steps'] and
         summary['skipped_updates'] <= info['max_fp16_skips'] and
         [entry['epoch'] for entry in history] == list(range(20)) and
         summary['gold_access'] == 'disabled' and
         summary['weak_oof_macro_auc'] is None and
         summary['exp179'] == prov['exp179'] == q['exp179'] and
         summary['exp185'] == prov['exp185'] == q['exp185'] and
         summary['exp187'] == prov['exp187'] == q['exp187'] and
         gpu in summary['environment']['gpu'] and gpu in prov['environment']['gpu'],
         'EXP187 fixed-final training evidence differs')
    args = summary['args']
    need(all(args.get(key) == value for key, value in q['common_args'].items()) and
         args['seed'] == 43 and args['fold'] == fold and args['tag'] == tag(fold) and
         args['mode'] == 'full' and
         args['labels'] == args['eval_labels'] == f"{WORKSPACE}/{q['paths']['labels']}" and
         prov['initial_state_sha256'] == q['initial_state_sha256'] and
         prov['init_weights']['weights_sha256'] == q['init_weights']['sha256'] and
         prov['runner_sha256'] == sha(root / RUNNER) == q['runner_sha256'] and
         prov['guard_sha256'] == sha(root / TOOL) == q['tool_sha256'] and
         prov['protocol_sha256'] == sha(root / protocol_path(fold)) and
         prov['train_ids_ordered_sha256'] == info['ordered_ids']['train_sha256'] and
         prov['eval_ids_ordered_sha256'] == info['ordered_ids']['val_sha256'] and
         prov['actual_pos_weight'] == info['pos_weight'] and
         prov['flip_tta'] == 'coronal_axial_horizontal_mean_probability' and
         prov['label_file_sha256'] == q['label_set']['sha256'] and
         prov['label_semantic_sha256'] == q['label_set']['semantic_sha256'],
         'EXP187 runner recipe/provenance differs')
    ids = read(root / info['ordered_ids']['path'])['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        _prediction_values(run / name, ids)
    import torch
    checkpoint = torch.load(run / 'model_best.pt', map_location='cpu', weights_only=False)
    keys = read(root / q['initial_state']['keys_file'])['keys']
    need(checkpoint.get('model_kind') == 'finding_spatial_tokens_v1' and
         checkpoint['tag'] == tag(fold) and set(checkpoint['model']) == set(keys),
         'EXP187 checkpoint kind, identity or parameter set differs')
    return {name: inv[name]['sha256'] for name in OUTPUTS}


def _status(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    write(tmp, value)
    tmp.replace(path)


def remote_run(started, input_root, output_root, group, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL remote_run is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def collect(source_dir, group, root=ROOT, *, source_root=None):
    """Collect an entire complete kernel into EXP186-compatible per-fold paths."""
    from tools import exp150_consensus_weight as corpus
    root, source_dir = Path(root), Path(source_dir)
    # Authenticate against the immutable historical stage, not the retired live tool.
    historical = Path(source_root) if source_root is not None else root
    need(group in KERNELS, 'unknown kernel group')
    status = read(source_dir / 'exp187-status.json')
    need(status['schema'] == SCHEMA and status['version'] == HISTORICAL_VERSION and
         status['state'] == 'COMPLETE' and status['group'] == group and
         status['folds'] == list(KERNELS[group]) and
         set(status['completed']) == {str(f) for f in KERNELS[group]} and
         0 < status['total_seconds'] < float('inf'), 'kernel did not complete its folds')
    validated = {}
    for fold in KERNELS[group]:
        folder = source_dir / f'fold{fold}'
        run = folder / 'results' / tag(fold)
        hashes = verify_run(run, fold, historical)
        receipt = read(folder / f'validated-fold{fold}.json')
        protocol = read(historical / protocol_path(fold))
        need(receipt == status['completed'][str(fold)] and
             receipt['schema'] == SCHEMA and receipt['status'] == 'VALIDATED_SUCCESS' and
             receipt['version'] == HISTORICAL_VERSION and receipt['group'] == group and
             receipt['fold'] == fold and receipt['tag'] == tag(fold) and receipt['seed'] == 43 and
             receipt['protocol_sha256'] == sha(historical / protocol_path(fold)) and
             receipt['runner_sha256'] == sha(historical / RUNNER) and
             receipt['tool_sha256'] == sha(historical / TOOL) and
             receipt['model_sha256'] == sha(historical / MODEL) == protocol['exp185']['model_sha256'] and
             receipt['label_sha256'] == protocol['label_set']['sha256'] and
             receipt['roster_sha256'] == sha(historical / roster_path(fold)) ==
                 protocol['folds'][str(fold)]['ordered_ids']['sha256'] and
             receipt['corpus_manifest_sha256'] == corpus.CORPUS_HASH and
             receipt['output_sha256'] == hashes and
             receipt['inventory_sha256'] == sha(run / 'output-inventory.json') and
             0 < receipt['elapsed_seconds'] < float('inf'),
             'fold receipt or output differs')
        dest = root / EXP / 'outputs' / f'fold{fold}'
        need(not dest.exists(), 'fold already collected')
        validated[fold] = (folder, dest, receipt)
    result = {}
    for fold, (folder, dest, receipt) in validated.items():
        dest.mkdir(parents=True)
        shutil.copytree(folder / 'results', dest / 'results')
        shutil.copyfile(folder / f'validated-fold{fold}.json', dest / f'validated-fold{fold}.json')
        write(dest / 'collection.json', dict(schema=SCHEMA, status='VERIFIED',
             version=VERSION, fold=fold, group=group,
             historical_source_root=str(historical.resolve()),
             protocol_sha256=receipt['protocol_sha256'],
             output_sha256=receipt['output_sha256'],
             inventory_sha256=receipt['inventory_sha256'],
             receipt_sha256=sha(dest / f'validated-fold{fold}.json'),
             status_sha256=sha(source_dir / 'exp187-status.json')))
        result[str(fold)] = str(dest)
    return {'status': 'VERIFIED', 'group': group, 'folds': result}


def push_plan(out, root=ROOT):
    """Retired execution entry point; historical artifacts remain readable."""
    raise ValueError("Legacy MIL push_plan is retired; use tools/mil_register.py and tools/arm_lifecycle.py. Softprior training requires a distinct registered family.")


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('stage', 'verify-stage', 'push-plan'):
        sub.add_parser(command).add_argument('--out', required=True)
    p = sub.add_parser('remote-run')
    p.add_argument('--started', type=float, required=True)
    p.add_argument('--input-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--group', choices=tuple(KERNELS), required=True)
    p = sub.add_parser('collect')
    p.add_argument('--source', required=True)
    p.add_argument('--source-root', help='Immutable historical stage used to authenticate source bytes')
    p.add_argument('--group', choices=tuple(KERNELS), required=True)
    args = parser.parse_args(argv)
    if args.command == 'stage':
        result = stage(args.out)
    elif args.command == 'verify-stage':
        result = verify_stage(args.out)
    elif args.command == 'push-plan':
        print(push_plan(args.out))
        return 0
    elif args.command == 'remote-run':
        result = remote_run(args.started, args.input_root, args.output_root, args.group)
    else:
        result = collect(args.source, args.group, source_root=args.source_root)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
