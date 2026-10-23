"""EXP083 registration and explicit adapters; importing performs no I/O or GPU work."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'artifacts/experiments/EXP083_SLOT_BIAS_F2_PROTOCOL'
RUNNER = 'training/runners/exp083_slot_bias.py'
TAG = 'EXP083_SLOT_BIAS_F2'
SIZES = (18, 14, 12, 8, 12)
UID = 'StudyInstanceUID'
HEAD_KEYS = ('norm.weight', 'norm.bias', 'att.0.weight', 'att.0.bias',
             'att.3.weight', 'att.3.bias', 'clsW', 'clsb')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def centers_from_mask(mask, k=24):
    """Same deterministic within-slot-valid formula; no image/label access."""
    import numpy as np
    mask = np.asarray(mask)
    if mask.shape != (64,) or k != 24:
        raise ValueError('Expected64-mask/K24 contract')
    candidates, start = [], 0
    for size in SIZES:
        candidates.extend(c for c in range(start + 1, start + size - 1)
                          if np.all(mask[c - 1:c + 2] > 0))
        start += size
    if not candidates:
        return [-1] * k
    positions = np.linspace(0, len(candidates) - 1, k).round().astype(int)
    return [candidates[i] for i in positions]


def slots_for_centers(centers, mask):
    import numpy as np
    mask = np.asarray(mask)
    if mask.shape != (64,) or len(centers) != 24:
        raise ValueError('Wrong center/mask shape')
    if any(int(c) != c for c in centers):
        raise ValueError('Noninteger center')
    if -1 in centers:
        if list(centers) != [-1] * 24 or centers_from_mask(mask) != [-1] * 24:
            raise ValueError('Empty center must mean no valid triplet')
        return [-1] * 24
    result = []
    for c in centers:
        start = 0
        found = False
        for slot, size in enumerate(SIZES):
            if start + 1 <= c < start + size - 1:
                if not np.all(mask[c - 1:c + 2] > 0):
                    raise ValueError('Triplet has a zero cache mask')
                result.append(slot); found = True; break
            start += size
        if not found:
            raise ValueError('Triplet crosses a slot boundary')
    return result


def make_head(shared_state, treatment):
    import torch
    from torch import nn
    if set(shared_state) != set(HEAD_KEYS):
        raise ValueError('Unexpected B8 head state')
    features = int(shared_state['clsW'].shape[1])
    if tuple(shared_state['clsW'].shape) != (12, features):
        raise ValueError('Expected12 targets')

    class SlotBiasHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(features)
            self.att = nn.Sequential(nn.Linear(features, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, features))
            self.clsb = nn.Parameter(torch.zeros(12))
            self.slot_bias = nn.Parameter(torch.zeros(5, 12), requires_grad=bool(treatment))
            expected = {**shared_state, 'slot_bias': self.slot_bias.detach().clone()}
            self.load_state_dict(expected, strict=True)

        def biased_attention(self, logits, slots):
            if slots.shape != logits.shape[:2] or slots.dtype != torch.int64:
                raise ValueError('Invalid slot tensor')
            if bool(((slots < -1) | (slots > 4)).any()):
                raise ValueError('Unknown slot ID')
            # Explicit cast prevents FP32 parameter promotion in either arm.
            bias = self.slot_bias.to(dtype=logits.dtype)[slots.clamp(min=0)]
            bias = bias * (slots >= 0).unsqueeze(-1).to(dtype=logits.dtype)
            return logits + bias

        def forward(self, features, slots):
            h = self.norm(features)
            attention_logits = self.att(h)
            biased = self.biased_attention(attention_logits, slots)
            a = torch.softmax(biased, dim=1)
            pooled = torch.einsum('bkn,bkf->bnf', a, h)
            output = (pooled * self.clsW).sum(-1) + self.clsb
            self.last_dtypes = {name: str(value.dtype) for name, value in
                [('feature', features), ('normalized_feature', h), ('attention_logits', attention_logits),
                 ('biased_attention_logits', biased), ('pooled', pooled), ('output', output)]}
            return output

    return SlotBiasHead()


def shared_head_hash(head):
    return state_hash({k: v for k, v in head.state_dict().items() if k != 'slot_bias'})


def native_feature_array(encoded, expected_shape):
    import torch
    if encoded.dtype != torch.float16:
        raise TypeError('Native feature dtype is not FP16; no quantizing cast is allowed')
    if tuple(encoded.shape) != tuple(expected_shape) or not bool(torch.isfinite(encoded).all()):
        raise ValueError('Feature shape/finite contract failed')
    return encoded.detach().cpu().numpy()


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('Registered EXP083 wall-time budget exceeded')


def prepare(root=ROOT):
    import numpy as np
    import torch
    from tools import exp082_r34_loss as old
    root = Path(root); out = root / PROTOCOL
    if out.exists() and any(not p.name.startswith('cpu-tests') for p in out.iterdir()):
        raise ValueError('Refusing to overwrite existing EXP083 registration')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation requires an uninitialized CUDA context')
    parent = read(root / 'artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/manifest.json')
    loss = read(root / 'artifacts/experiments/EXP082_R34_LOSS_F2_PROTOCOL/manifest.json')
    ordered = read(root / loss['ordered_ids']['path'])
    for name, expected in parent['training_code_sha256'].items():
        if digest(root / 'training/src' / name) != expected:
            raise ValueError('Frozen training source changed')
    corpus = root / parent['corpus']
    for uri, expected in parent['metadata_source_hashes'].items():
        if uri.endswith(('/ids.npy', '/masks.npy')) and digest(root / uri.removeprefix('workspace://')) != expected:
            raise ValueError('Frozen corpus metadata changed')
    if digest(corpus / 'manifest.json') != parent['corpus_manifest_sha256']:
        raise ValueError('Corpus manifest changed')
    ids = np.load(corpus / 'ids.npy', allow_pickle=False).astype(str).tolist()
    with (root / parent['fold_file']).open(newline='', encoding='utf-8-sig') as stream:
        folds = {r[UID]: int(r['fold']) for r in csv.DictReader(stream)}
    train, evaluation = old.reconstruct_ids(ids, folds, set(parent['excluded_uids']), parent['expected_ids']['2'])
    old.validate_ordered_ids(train, evaluation, ordered)
    all_ids = train + evaluation
    masks = np.load(corpus / 'masks.npy', allow_pickle=False)
    index = {u: i for i, u in enumerate(ids)}
    centers = np.asarray([centers_from_mask(masks[index[u]]) for u in all_ids], dtype=np.int16)
    slots = np.asarray([slots_for_centers(c, masks[index[u]]) for u, c in zip(all_ids, centers)], dtype=np.int8)
    checkpoint = 'artifacts/experiments/EXP080_INPUT_B_F2/model_best.pt'
    expected_checkpoint = '5e3399bb3f44bb43f81b58c4048cabad1c4657f78a4952d30b15a957f022992f'
    if digest(root / checkpoint) != expected_checkpoint:
        raise ValueError('B8 checkpoint changed')
    ck = torch.load(root / checkpoint, map_location='cpu', weights_only=False)
    state = ck['model']  # Historical diagnostic metadata is neither selected nor used.
    head = {k: state[k].detach().cpu().clone() for k in HEAD_KEYS}
    encoder_sha = state_hash({k: v for k, v in state.items() if k.startswith('backbone.')})
    head_sha = state_hash(head); features = int(head['clsW'].shape[1])
    if ck['arch'] != parent['expected_args_by_arm']['B']['arch'] or ck['epoch'] != 7:
        raise ValueError('Expected EXP080 B final8 architecture')
    if any(v.dtype != torch.float32 for v in head.values()):
        raise ValueError('Expected FP32 head parameters')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head, out / 'head-initial.pt')
    np.save(out / 'centers.npy', centers, allow_pickle=False)
    np.save(out / 'slots.npy', slots, allow_pickle=False)
    write_new(out / 'ordered-ids.json', ordered)
    pilot_ids = train[:8] + evaluation[:8]
    write_new(out / 'pilot-ids.json', pilot_ids)
    group_path = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/contract.json'
    group = read(root / group_path)
    if group['cross_old_fold_component']['old_folds'] != [0, 1] or group['new_split_created']:
        raise ValueError('Known grouped development contract changed')
    bindings = [
        'tools/exp083_slot_bias.py', RUNNER, 'tools/exp082_r34_loss.py',
        'artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/manifest.json',
        'artifacts/experiments/EXP080_INPUT_B_F2/provenance.json',
        'artifacts/experiments/EXP082_R34_LOSS_F2_PROTOCOL/manifest.json', group_path,
        *parent['source_file_sha256'], parent['corpus'] + '/manifest.json',
        parent['corpus'] + '/ids.npy', parent['corpus'] + '/masks.npy', checkpoint,
        *[PROTOCOL + '/' + n for n in ('head-initial.pt', 'centers.npy', 'slots.npy', 'ordered-ids.json', 'pilot-ids.json')]]
    for relative, expected in parent['source_file_sha256'].items():
        if digest(root / relative) != expected:
            raise ValueError('Frozen label/split input changed')
    shape = [len(all_ids), 24, features]
    manifest = dict(schema='exp083-slot-bias-f2-v1', created_at=datetime.now(timezone.utc).isoformat(),
        status='CPU_PREPARED_NO_GPU_ROOT_REVIEW_REQUIRED', tag=TAG,
        source_bindings={p: digest(root / p) for p in bindings}, training_code_sha256=parent['training_code_sha256'],
        checkpoint=checkpoint, checkpoint_sha256=expected_checkpoint, architecture=ck['arch'],
        saved_head_sha256=head_sha, encoder_state_sha256=encoder_sha,
        corpus=parent['corpus'], corpus_manifest_sha256=parent['corpus_manifest_sha256'],
        labels=parent['labels'], label_hash=parent['label_hash'], eval_label_hash=parent['eval_label_hash'],
        targets=parent['targets'], reference_binary_rule='float32 then numpy.rint; only future fixed evaluator may score',
        ordered_ids=dict(train=len(train), eval=len(evaluation), train_sha256=ids_hash(train),
                         eval_sha256=ids_hash(evaluation), combined_sha256=ids_hash(all_ids)),
        cache=dict(shape=shape, dtype='float16', data_bytes=len(all_ids) * 24 * features * 2,
                   format='NPY memmap with native FP16 values; header bytes extra', res=224, k=24,
                   window_policy='within-slot-valid', norm='imagenet', extraction_batch=4,
                   extraction_workers=2, slots=list(SIZES), empty_slot=-1, empty_bias=0,
                   extraction_split_loaders=True, roundtrip='Exact native values and original-head logits per batch',
                   rule='Actual triplet center membership, no index binning/quota change/blank-feature overwrite'),
        head=dict(trainable_slot_bias_shape=[5, 12], bias_initial=0., empty_bias_trainable=False,
                  shared_init='saved B8 head in both arms', shared_head_sha256=head_sha,
                  intervention='attention-logit bias only, cast to logits dtype in BOTH arms',
                  dropout=.2, epochs=20, batch=64, drop_last=False, workers=0, seed=42,
                  optimizer='AdamW', lr=.001, weight_decay=.02, scheduler='OneCycleLR', pct_start=.15,
                  optimizer_defaults=dict(betas=[.9, .999], eps=1e-8, amsgrad=False),
                  scheduler_defaults=dict(anneal_strategy='cos', cycle_momentum=True, base_momentum=.85,
                                          max_momentum=.95, div_factor=25., final_div_factor=10000., three_phase=False),
                  gradient_clip=3., amp='fp16', amp_init_scale=1024., runtime_policy='strict',
                  loss='BCE positive-term clipped weight, mean reduction in both arms',
                  pos_weight=loss['loss']['baseline_pos_weight'], expected_batches=55, last_batch=19,
                  expected_attempts=1100, any_skips_require_review=True),
        expected_provenance=read(root / 'artifacts/experiments/EXP080_INPUT_B_F2/provenance.json'),
        environment=loss['environment'],
        phases=dict(pilot_ids=pilot_ids, pilot_max_seconds=300, total_max_seconds=2700,
                    full_requires_root_decision=True, automatic_retries=0,
                    deadline_scope='Whole phase wall time checked between batches/epochs; not a hard OS hang watchdog'),
        evaluation=dict(primary='macro12 AUC(treatment_head) - macro12 AUC(retrained_control_head)',
            reference='same fixed F2 labels/UIDs; untouched/independent claim prohibited',
            bootstrap=dict(unit='VALIDATION_GROUP_CONTRACT_V1 component', reps=500, seed=20261003),
            no_interim_F2_metrics=True, no_ratio_or_target_tuning=True, no_all_targets_positive_gate=True,
            no_minimum_positive_auc_cutoff=True, no_automatic_expansion=True),
        limits=['Frozen features cannot recover absent image information',
                'Additional60parameters and slot information are not fully disentangled',
                'Exposed F2 weak development, known F0/F1 pair both train; unknown patient duplicates remain',
                'Inherits EXP080 B8 and prior researcher/Gold exposure; no Gold file read does not erase this lineage',
                'No anchor0.943/D01/private transfer claim'],
        pending=['GPU pilot feature dtype/parity/throughput', 'Root one full-phase decision', 'GPU feature cache and head fits', 'Separate paired evaluation'])
    write_new(out / 'manifest.json', manifest)
    system_python = read(root / 'artifacts/logs/EXP080_R34_B_F2/job.json')['command'][0]
    prefix = [system_python, RUNNER, '--protocol', str(out / 'manifest.json')]
    write_new(out / 'commands.json', dict(
        environment={'HF_HUB_OFFLINE': '1'},
        pilot=prefix + ['--phase', 'pilot', '--out', str(out / 'pilot-v1')],
        full=prefix + ['--phase', 'full', '--out', str(out / 'run-v1'), '--pilot-receipt', str(out / 'pilot-v1/receipt.json'),
                       '--root-decision', str(out / 'root-full-decision.json')],
        root_full_decision_required_fields=dict(decision='RUN_ONE_FULL_PHASE', protocol_sha256=digest(out / 'manifest.json'),
            pilot_receipt_sha256='FILL_AFTER_ACTUAL_PILOT', max_total_seconds=2700, automatic_retries=0),
        launch='Use existing gpu_job.py with separate phase tags and --wait omitted. These argv are not executed by preparation.'))
    return dict(status=manifest['status'], feature_dim=features, cache=manifest['cache'],
                manifest_sha256=digest(out / 'manifest.json'), saved_head_sha256=head_sha,
                cuda_initialized=torch.cuda.is_initialized(), feature_cache_created=False)


def validate_registration(path, root=ROOT):
    root = Path(root); m = read(path)
    if m.get('schema') != 'exp083-slot-bias-f2-v1' or m.get('tag') != TAG:
        raise ValueError('Wrong EXP083 registration')
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('HF_HUB_OFFLINE=1 required')
    for p, expected in m['source_bindings'].items():
        relative = Path(p)
        if relative.is_absolute() or relative.drive or '..' in relative.parts or digest(root / relative) != expected:
            raise ValueError('Source/input hash mismatch: ' + p)
    for name, expected in m['training_code_sha256'].items():
        if digest(root / 'training/src' / name) != expected:
            raise ValueError('Frozen source changed')
    return m


def validate_phase_authority(phase, protocol_path, m, pilot_path=None, decision_path=None):
    if phase == 'pilot':
        if pilot_path or decision_path:
            raise ValueError('Pilot cannot consume a full decision')
        return m['phases']['pilot_max_seconds']
    if phase != 'full' or not pilot_path or not decision_path:
        raise ValueError('Full requires exact successful pilot and root decision')
    pilot, decision = read(pilot_path), read(decision_path)
    if (pilot.get('status') != 'PASS' or pilot.get('phase') != 'pilot'
            or pilot.get('protocol_sha256') != digest(protocol_path)
            or pilot.get('pilot_ids') != m['phases']['pilot_ids']
            or pilot.get('feature_dtype') != 'float16' or pilot.get('initial_logits_exact') is not True
            or pilot.get('initial_train_logits_exact') is not True
            or pilot.get('native_cache_roundtrip_exact') is not True
            or pilot.get('native_cache_logits_exact') is not True
            or pilot.get('extraction_split_loaders') is not True
            or pilot.get('disjoint_initial_head_storage') is not True
            or pilot.get('encoder_before_sha256') != m['encoder_state_sha256']
            or pilot.get('encoder_after_sha256') != m['encoder_state_sha256']):
        raise ValueError('Pilot receipt does not match this registration')
    required = dict(decision='RUN_ONE_FULL_PHASE', protocol_sha256=digest(protocol_path),
                    pilot_receipt_sha256=digest(pilot_path), max_total_seconds=2700, automatic_retries=0)
    if any(decision.get(k) != v for k, v in required.items()):
        raise ValueError('Explicit root full-phase decision mismatch')
    remaining = 2700 - float(pilot['elapsed_seconds'])
    if not 0 < remaining <= 2700:
        raise ValueError('Invalid remaining shared budget')
    return remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if not args.prepare:
        parser.error('Only --prepare is supported here; GPU phases use the separate runner')
    print(json.dumps(prepare(), indent=2))


if __name__ == '__main__':
    main()
