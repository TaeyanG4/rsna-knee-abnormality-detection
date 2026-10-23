"""EXP085 paired relative-position adaptation; import is passive.

prepare reads metadata/checkpoint bytes only. run is an explicit launch requiring
a phase-specific root decision; neither importing this module nor prepare trains.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'exp085-relative-position-v2-fp32-reduction'
PARENT = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
PROTOCOL = 'artifacts/experiments/EXP085_RELATIVE_POSITION_G1_PROTOCOL_V2'
RUNNER = 'training/runners/exp085_relative_position.py'
EVIDENCE = 'docs/handoffs/evidence/2026-10/2026-10-04-exp085-relative-position'
ARMS = ('control', 'treatment')
SIZES = (18, 14, 12, 8, 12)
UID = 'StudyInstanceUID'
THREAD_ENV = {key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def contained(root, relative):
    path = Path(str(relative).removeprefix('workspace://'))
    if path.is_absolute() or path.drive or '..' in path.parts:
        raise ValueError('Expected contained workspace relative path')
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes workspace')
    return result


def ids_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def position_metadata(centers, mask):
    """Cache-index position (not millimetres/anatomical direction), shared in arms."""
    import numpy as np
    mask = np.asarray(mask)
    if mask.shape != (64,) or not np.isfinite(mask).all() or not len(centers):
        raise ValueError('Expected finite 64-plane mask and nonempty centers')
    slots = np.full(len(centers), -1, np.int64)
    positions = np.zeros(len(centers), np.float32)
    all_valid = []
    start = 0
    for slot, size in enumerate(SIZES):
        for c in range(start + 1, start + size - 1):
            if np.all(mask[c - 1:c + 2] > 0):
                all_valid.append(c)
        for index, center in enumerate(centers):
            if int(center) != center:
                raise ValueError('Noninteger center')
            if start + 1 <= center < start + size - 1:
                if not np.all(mask[center - 1:center + 2] > 0):
                    raise ValueError('Center triplet is not mask-valid')
                slots[index] = slot
                positions[index] = np.float32((center - start) / (size - 1))
        start += size
    if not all_valid:
        if list(centers) != [-1] * len(centers):
            raise ValueError('Empty mask requires all -1 centers')
    elif np.any(slots < 0):
        raise ValueError('Invalid/boundary/empty center in nonempty study')
    return slots, positions


class PositionWindows:
    """Pickleable adapter: original StudyWindows supplies every pixel unchanged."""
    def __init__(self, *args, **kwargs):
        from raptor_data import StudyWindows
        self.base = StudyWindows(*args, **kwargs)
        if self.base.window_policy != 'within-slot-valid':
            raise ValueError('EXP085 requires unchanged within-slot-valid sampling')

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __getitem__(self, index):
        import torch
        from raptor_common import select_centers, study_rng
        b = self.base
        epoch = int(b.epoch.item())
        x, y, uid = b[index]
        if int(b.epoch.item()) != epoch:
            raise ValueError('Dataset epoch changed during pixel/center replay')
        if b.corpus.depth != 64 or tuple(int(slot[2]) for slot in b.corpus.spec['slots']) != SIZES:
            raise ValueError('Fixed64 cache slot contract changed')
        mask = b.corpus.mask(uid)
        rng = study_rng(b.seed, epoch, uid) if b.train else None
        centers = select_centers(mask, b.corpus.depth, b.k, b.corpus.spec, b.window_policy, rng)
        slots, positions = position_metadata(centers, mask)
        return x, torch.from_numpy(slots), torch.from_numpy(positions), y, uid, torch.tensor(centers, dtype=torch.int64)


def collate(batch):
    import torch
    return tuple(torch.stack([item[k] for item in batch]) if k != 4 else [item[k] for item in batch]
                 for k in range(6))


def make_model(base, treatment, dimension=64):
    """Keep original B backbone/head; a zero-initialized residual starts at B."""
    import torch
    from torch import nn
    if type(treatment) is not bool or dimension != 64:
        raise ValueError('Fixed boolean treatment and 64-dimensional attention required')
    features = int(base.clsW.shape[1])

    class PositionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.treatment = treatment
            self.qkv = nn.Linear(features, 3 * dimension, bias=False)
            self.projection = nn.Linear(dimension, features, bias=False)
            self.relative = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.relative[-1].weight)

        def contextualize(self, features, slots, positions):
            if slots.dtype != torch.int64 or slots.shape != features.shape[:2] or positions.shape != slots.shape:
                raise ValueError('Position/mask shape or slot dtype mismatch')
            if not positions.dtype.is_floating_point or not bool(torch.isfinite(positions).all()):
                raise ValueError('Nonfinite/nonfloat position')
            if bool(((slots < -1) | (slots > 4)).any()) or bool(((positions < 0) | (positions > 1)).any()):
                raise ValueError('Slot/position range mismatch')
            # No cross-slot mixing; invalid queries attend themselves then get zero residual.
            valid = slots >= 0
            same = (slots[:, :, None] == slots[:, None, :]) & valid[:, :, None] & valid[:, None, :]
            eye = torch.eye(slots.shape[1], dtype=torch.bool, device=slots.device)[None]
            allowed = same | ((~valid)[:, :, None] & eye)
            h = self.base.norm(features)
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            p = positions if self.treatment else torch.zeros_like(positions)
            delta = p[:, None, :] - p[:, :, None]
            coordinates = torch.stack((delta, delta.abs()), dim=-1)
            bias = self.relative(coordinates).squeeze(-1).float()
            # .float() alone does not stop the surrounding autocast from
            # choosing reduced-precision matmul. Fence the full reduction.
            with torch.autocast(device_type=features.device.type, enabled=False):
                logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (dimension ** .5) + bias
                attention = torch.softmax(logits.masked_fill(~allowed, float('-inf')), dim=-1)
                context = torch.matmul(attention, v.float())
            projected = self.projection(context.to(v.dtype))
            residual = projected.to(features.dtype)
            residual = residual * valid.unsqueeze(-1).to(residual.dtype)
            self.last_attention_dtypes = {name: str(value.dtype) for name, value in
                [('q', q), ('k', k), ('v', v), ('bias', bias), ('logits', logits),
                 ('attention', attention), ('context', context), ('projection', projected), ('residual', residual)]}
            return features + residual

        def head(self, features, slots, positions):
            return self.base.head(self.contextualize(features, slots, positions))

        def forward(self, x, slots, positions):
            return self.head(self.base.encode(x), slots, positions)

    return PositionModel()


def verify_initial_forward(model, pixels, slots, positions):
    """Eval/train exact B gate, preserving RNG and all buffers around both probes."""
    import torch
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(pixels.device).clone() if pixels.device.type == 'cuda' else None
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = {name: module.training for name, module in model.named_modules()}

    def restore_rng_buffers():
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, pixels.device)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])

    def rng_state():
        cpu = torch.random.get_rng_state().clone()
        gpu = torch.cuda.get_rng_state(pixels.device).clone() if cuda_state is not None else None
        return cpu, gpu

    result = {}
    try:
        with torch.no_grad():
            for training in (False, True):
                model.train(training)
                restore_rng_buffers()
                expected = model.base(pixels)
                expected_rng = rng_state()
                expected_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
                restore_rng_buffers()
                actual = model(pixels, slots, positions)
                actual_rng = rng_state()
                rng_exact = torch.equal(expected_rng[0], actual_rng[0]) and (
                    cuda_state is None or torch.equal(expected_rng[1], actual_rng[1]))
                buffers_exact = all(torch.equal(value, expected_buffers[name]) for name, value in model.named_buffers())
                exact = torch.equal(expected, actual)
                if not exact or not rng_exact or not buffers_exact or not bool(torch.isfinite(actual).all()):
                    raise ValueError('Initial B eval/train forward, RNG or buffer parity failed')
                result['train' if training else 'eval'] = {'logits_exact': exact, 'rng_consumption_exact': rng_exact,
                                                         'buffer_updates_exact': buffers_exact}
    finally:
        restore_rng_buffers()
        for name, module in model.named_modules():
            module.training = modes[name]
    result['probe_rng_and_buffers_restored'] = True
    return result


def optimizer_groups(model, bb_lr, head_lr):
    backbone = list(model.base.backbone.parameters())
    ids = set(map(id, backbone))
    head = [p for p in model.parameters() if id(p) not in ids]
    if not backbone or not head or not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Both arms must train the full backbone and all attention parameters')
    return [{'params': backbone, 'lr': bb_lr}, {'params': head, 'lr': head_lr}]


def read_training_labels(path, train_ids, targets):
    """V rows are skipped before parsing target values; no metric labels in runner."""
    import numpy as np
    wanted = set(train_ids)
    result = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = [target if target in reader.fieldnames else target + '_soft' for target in targets]
        if any(c not in reader.fieldnames for c in columns):
            raise ValueError('Training label schema changed')
        for row in reader:
            uid = row[UID]
            if uid not in wanted:
                continue
            if uid in result:
                raise ValueError('Duplicate training UID')
            value = np.asarray([float(row[c]) for c in columns], dtype=np.float32)
            if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                raise ValueError('Invalid T supervision')
            result[uid] = value
    if set(result) != wanted:
        raise ValueError('T supervision coverage changed')
    return result


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Registered EXP085 budget exhausted; retries0')


def prepare(root=ROOT, out=None):
    """Metadata-only draft; no labels/predictions/cache arrays/model execution."""
    root = Path(root)
    out = Path(out) if out else root / PROTOCOL
    old = read(root / PARENT)
    b = old['arms']['B']
    summary_path = root / b['outputs']['summary']
    summary = read(summary_path)
    if not (root / b['logs'] / 'SUCCESS').is_file() or (root / b['logs'] / 'FAILED').exists():
        raise ValueError('EXP084 B parent job must have completed successfully')
    for name in ('tools/exp084_group_loss.py', 'training/runners/exp084_group_loss.py'):
        if sha(root / name) != old['source_bindings'][name]:
            raise ValueError('EXP084 source differs from completed parent contract')
    if (summary['selected_epoch_1based'] != 8 or summary['skipped_updates'] != 0
            or summary['optimizer_updates'] != 3448 or summary['weak_oof_macro_auc'] is not None
            or summary['provenance']['protocol_sha256'] != sha(root / PARENT)
            or summary['provenance']['train_ids_ordered_sha256'] != old['ordered_ids']['train_sha256']
            or summary['provenance']['eval_ids_ordered_sha256'] != old['ordered_ids']['val_sha256']):
        raise ValueError('EXP084 B completion/ancestor mismatch')
    p = {key: copy.deepcopy(old[key]) for key in ('targets', 'paths', 'ordered_ids', 'groups', 'expected_ids',
        'split', 'excluded_uids', 'loss', 'reference_counts', 'macro12_valid', 'corpus_manifest_sha256',
        'label_hash', 'eval_label_hash', 'split_hash', 'training_code_sha256', 'expected_runtime', 'environment')}
    p.update(schema=SCHEMA, status='CPU_DRAFT_REQUIRES_ROOT_PHASE_DECISION', prepared_at=datetime.now(timezone.utc).isoformat(),
             supersedes_cpu_draft={'path': 'artifacts/experiments/EXP085_RELATIVE_POSITION_G1_PROTOCOL/manifest.json',
                 'sha256': sha(root / 'artifacts/experiments/EXP085_RELATIVE_POSITION_G1_PROTOCOL/manifest.json'),
                 'snapshot': EVIDENCE + '/v1-snapshot/snapshot-manifest.json',
                 'reason': 'Explicit autocast-off FP32 attention reduction; V1 was CPU draft only, no pilot/failure run'},
             applicable_decisions=['M02', 'M03', 'M04', 'M05', 'M10'], arm_order=list(ARMS),
             parent_protocol={'path': PARENT, 'sha256': sha(root / PARENT)},
             initial_checkpoint={'path': b['outputs']['checkpoint'], 'sha256': sha(root / b['outputs']['checkpoint']),
                 'ancestor': 'EXP084_B_G1 T-only fixed8, same group V', 'mode': 'weights-only adaptation + fresh AdamW/OneCycleLR/GradScaler; not optimizer continuation'},
             training=copy.deepcopy(b['expected_args']),
             architecture={'feature_dim': 1024, 'attention_dim': 64, 'heads': 1, 'extra_parameters': 262208,
                 'block': 'original B features + zero-initialized projected within-slot qkv attention, then original B.head verbatim',
                 'position': '(center-slot_start)/(slot_size-1), cache-relative index only; not mm or anatomy direction',
                 'bias': 'Linear(2,16)-Tanh-Linear(16,1) on signed(delta=p_key-p_query),abs(delta); final layer zero',
                 'control': 'position input allzero; same architecture/parameters/trainability/mask',
                 'treatment': 'actual fixed slot-relative position', 'invalid': 'self-attention fallback then zero residual; original B blank pooling unchanged',
                 'permutation': 'joint token/slot/position permutation equivariant, pooled output invariant; no slot-cross attention',
                 'attention_precision': 'autocast disabled for FP32 logits/softmax/value reduction; qkv/positionMLP/projection retain ambient native autocast dtype',
                 'first_batch_gate': 'Original B exact eval+train outputs, RNG consumption and buffer updates; restore initial RNG/buffers/modes after probes',
                 'zero_init': 'Only output projection and final relative-bias layer zero; no multiplicative zero gate'},
             execution={'total_time_limit_seconds': 7200, 'per_fit_seconds': 3600, 'pilot_seconds': 600,
                 'retries': 0, 'pilot_steps_per_arm': 32, 'pilot_eval_train_studies': 16, 'projection_safety_factor': 1.25,
                 'pilot_warmup_steps': 8, 'projection_rule': 'Count first8 train steps and first eval batch startup once; extrapolate remaining steady samples, then multiply total by1.25',
                 'seed': 42, 'epoch_offset': 8, 'full_epochs': 8, 'attempts_per_fit': 3448,
                 'budget_scope': 'sum of supervised pilot/full elapsed time including startup/evaluation/cleanup; human review gap excluded',
                 'fixed_arm_order': list(ARMS), 'all_skips_require_failure': True},
             evaluation={'primary': 'macro12(treatment)-macro12(control)',
                 'mandatory_reference_contrasts': ['control-B_original', 'treatment-B_original'],
                 'secondary_fixed_blends': '0.9*rank(each arm or original B)+0.1*rank(EXP084_RN_CLIP_G1)',
                 'weights': [.9,.1], 'ranks': 'average ties / full V N once before subgroups/bootstrap; no final rerank',
                 'bootstrap_reps': 500, 'bootstrap_seed': 20261004, 'bootstrap_unit': 'component_id',
                 'targets': 'all12 P/N/delta/conditionalCI; explicit Fracture and group summaries',
                 'metrics': 'only after both final fits; no pilot, epoch, Gold, LB, ratio, routing or best-epoch search'},
             outputs={arm: {name: f'artifacts/experiments/EXP085_RELATIVE_POSITION_G1/{arm}/{filename}'
                           for name, filename in [('summary','summary.json'),('provenance','provenance.json'),
                           ('checkpoint','model_final.pt'),('predictions','oof_final.csv'),('history','history.json'),('trace','trace.json')]}
                      for arm in ARMS},
             limitations=copy.deepcopy(old['limitations']) + ['Same development V reused after prior results; not untouched confirmation',
                'Effect is position information in this attention/adaptation recipe; each versus B also changes architecture/adaptation/restart',
                'Position can proxy acquisition protocol; lack of benefit does not refute broader sequence/3D/pretraining approaches'])
    p['training'].update(pretrained=b['outputs']['checkpoint'], tag=None)
    # Copy expected byte hashes without opening any supervision/prediction/cache.
    p['input_bindings'] = {key: value for key, value in old['source_bindings'].items()
                           if key in {old['paths']['labels'], old['paths']['exclude_ids'], old['ordered_ids']['path'], old['groups']['path'],
                                      old['paths']['corpus'] + '/manifest.json', old['paths']['corpus'] + '/ids.npy', old['paths']['corpus'] + '/masks.npy'}}
    sources = ['tools/exp085_relative_position.py', RUNNER, 'tools/exp084_group_loss.py', 'tools/process_resource_receipt.py', PARENT,
               b['outputs']['summary'], b['outputs']['provenance']]
    p['source_bindings'] = {path: sha(root / path) for path in sources}
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'manifest.json', p)
    return p


def validate_protocol(path, root=ROOT, verify_inputs=False):
    p = read(path)
    if p['schema'] != SCHEMA or p['arm_order'] != list(ARMS) or p['execution']['total_time_limit_seconds'] != 7200:
        raise ValueError('Wrong EXP085 protocol')
    for name, digest in p['source_bindings'].items():
        if sha(contained(root, name)) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for name, digest in p['training_code_sha256'].items():
        if sha(Path(root) / 'training/src' / name) != digest:
            raise ValueError('Original training source changed: ' + name)
    if verify_inputs:
        for name, digest in p['input_bindings'].items():
            if sha(contained(root, name)) != digest:
                raise ValueError('Frozen input changed: ' + name)
        if sha(contained(root, p['initial_checkpoint']['path'])) != p['initial_checkpoint']['sha256']:
            raise ValueError('Common B checkpoint changed')
    ordered = read(contained(root, p['ordered_ids']['path']))
    if any(ids_hash(ordered[key]) != p['ordered_ids'][key + '_sha256'] for key in ('train', 'val')):
        raise ValueError('Ordered IDs changed')
    return p, ordered


def pilot_cost_gate(pilot, p):
    if pilot.get('status') != 'PASS' or pilot.get('phase') != 'pilot' or pilot.get('paired_initialization_exact') is not True:
        raise ValueError('Completed paired pilot required')
    projected = pilot['conservative_full_seconds_by_arm']
    elapsed = pilot.get('supervised_elapsed_seconds')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 < elapsed <= p['execution']['pilot_seconds']:
        raise ValueError('Invalid supervised pilot elapsed time')
    if set(projected) != set(ARMS) or any(type(value) not in (int,float) or not math.isfinite(value) or not 0 < value <= p['execution']['per_fit_seconds'] for value in projected.values()):
        raise ValueError('Projected full fit exceeds per-fit cap')
    if sum(projected.values()) + pilot['supervised_elapsed_seconds'] > p['execution']['total_time_limit_seconds']:
        raise ValueError('Projected pilot+full exceeds total cap')


def run(protocol, phase, decision, pilot_dir=None, root=ROOT):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.process_resource_receipt import supervise
    p, _ = validate_protocol(protocol, root, verify_inputs=True)
    d = read(decision)
    if d.get('protocol_sha256') != sha(protocol) or d.get('allow_phase') != phase or d.get('retries') != 0:
        raise ValueError('Explicit matching root phase decision required')
    limit = p['execution']['pilot_seconds']
    run_dir = Path(root) / ('artifacts/experiments/EXP085_RELATIVE_POSITION_G1_PILOT' if phase == 'pilot' else 'artifacts/experiments/EXP085_RELATIVE_POSITION_G1')
    pilot = None
    if phase == 'full':
        if not pilot_dir:
            raise ValueError('Full requires paired pilot directory')
        pilot_path = Path(pilot_dir) / 'receipt.json'
        pilot = read(pilot_path)
        resource = read(Path(pilot_dir).with_name(Path(pilot_dir).name + '_RESOURCE') / 'receipt.json')
        if resource['status'] != 'COMPLETE' or resource['child_exit_code'] != 0 or resource['cleanup']['survivors'] or resource['cleanup']['errors']:
            raise ValueError('Pilot supervisor failed')
        pilot['supervised_elapsed_seconds'] = resource['elapsed_seconds']
        if pilot['protocol_sha256'] != sha(protocol) or d.get('pilot_receipt_sha256') != sha(pilot_path):
            raise ValueError('Pilot/decision binding differs')
        pilot_cost_gate(pilot, p)
        limit = p['execution']['total_time_limit_seconds'] - resource['elapsed_seconds']
    if run_dir.exists():
        raise ValueError('Fresh phase output required; retries0')
    command = [sys.executable, str(Path(root) / RUNNER), '--protocol', str(Path(protocol).resolve()), '--phase', phase,
               '--out', str(run_dir), '--time-limit', str(limit)]
    if pilot_dir:
        command += ['--pilot-dir', str(Path(pilot_dir).resolve())]
    before = dict(os.environ)
    try:
        os.environ.update({**THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                           'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONDONTWRITEBYTECODE': '1'})
        receipt = supervise(command, run_dir.with_name(run_dir.name + '_RESOURCE'), limit, root)
    finally:
        os.environ.clear(); os.environ.update(before)
    if receipt['status'] != 'COMPLETE' or read(run_dir / 'receipt.json').get('status') != 'PASS':
        raise RuntimeError('EXP085 phase failed; inspect resource/phase receipt; retries0')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare_parser = sub.add_parser('prepare'); prepare_parser.add_argument('--out')
    execute = sub.add_parser('run')
    execute.add_argument('--protocol', required=True); execute.add_argument('--phase', choices=['pilot','full'], required=True)
    execute.add_argument('--root-decision', required=True); execute.add_argument('--pilot-dir')
    a = parser.parse_args()
    result = prepare(out=a.out) if a.action == 'prepare' else run(a.protocol, a.phase, a.root_decision, a.pilot_dir)
    print(json.dumps({'status': result['status']}))


if __name__ == '__main__':
    main()
