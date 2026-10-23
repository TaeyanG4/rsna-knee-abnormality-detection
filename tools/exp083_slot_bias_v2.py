"""EXP083 v2 native-FP32 cache correction; CPU preparation only until root launch."""
from __future__ import annotations

import argparse
import csv
import copy
import math
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD_PROTOCOL = 'artifacts/experiments/EXP083_SLOT_BIAS_F2_PROTOCOL'
PROTOCOL = 'artifacts/experiments/EXP083_SLOT_BIAS_F2_PROTOCOL_V2'
RUNNER = 'training/runners/exp083_slot_bias_v2.py'
TAG = 'EXP083_SLOT_BIAS_F2_V2'
V1_MANIFEST_SHA = '575b011556ff74a00239cf566a3e7043caa76fb6543127cf813fd5eae9edeaa4'
V1_FAILURE_SHA = 'eed76d4058efa3db5a5feb5ffa7ca89960c4f977eb75c0c296d7b76d6af33550'
V1_HELPER_SHA = 'aa4f03942c5ce159b62b33efe5be2feb925259d56e48426f82942611be794b06'
V1_RUNNER_SHA = 'b31b63edaf96d80729c5ad453ee5284e67aa7bc77ccf57f3b8af8b55c50026ff'
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
    if encoded.dtype != torch.float32:
        raise TypeError(f'Native feature dtype is {encoded.dtype}, expected torch.float32; no quantizing cast is allowed')
    if tuple(encoded.shape) != tuple(expected_shape) or not bool(torch.isfinite(encoded).all()):
        raise ValueError('Feature shape/finite contract failed')
    return encoded.detach().cpu().numpy()


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('Registered EXP083 wall-time budget exceeded')


def prepare(root=ROOT):
    """Copy only frozen small payloads; no torch, checkpoint load, labels or cache reads."""
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise RuntimeError('CPU preparation requires CUDA to remain uninitialized')
    root = Path(root); out = root / PROTOCOL; previous = root / OLD_PROTOCOL
    if out.exists() and any(not p.name.startswith('cpu-tests') for p in out.iterdir()):
        raise ValueError('Refusing to overwrite existing EXP083v2 registration')
    frozen = {'tools/exp083_slot_bias.py': V1_HELPER_SHA,
              'training/runners/exp083_slot_bias.py': V1_RUNNER_SHA,
              OLD_PROTOCOL + '/manifest.json': V1_MANIFEST_SHA,
              OLD_PROTOCOL + '/pilot-v1/failure.json': V1_FAILURE_SHA}
    for p, expected in frozen.items():
        if digest(root / p) != expected:
            raise ValueError('Preserved v1 source/failure changed: ' + p)
    old = read(previous / 'manifest.json'); failure = read(previous / 'pilot-v1/failure.json')
    if (failure.get('status') != 'FAILED' or failure.get('phase') != 'pilot'
            or failure.get('protocol_sha256') != V1_MANIFEST_SHA
            or failure.get('error') != 'Native feature dtype is not FP16; no quantizing cast is allowed'
            or failure.get('no_F2_score_evaluated') is not True):
        raise ValueError('Expected v1 native dtype failure')
    spent = float(failure['elapsed_seconds'])
    if not math.isfinite(spent) or not 0 < spent < 2700:
        raise ValueError('Invalid prior spent budget')
    for name, expected in old['training_code_sha256'].items():
        if digest(root / 'training/src' / name) != expected:
            raise ValueError('Frozen training source changed')
    if old['cache']['shape'] != [4343, 24, 1024] or old['cache']['dtype'] != 'float16':
        raise ValueError('Expected frozen v1 shape/dtype')
    m = copy.deepcopy(old)
    m.update(schema='exp083-slot-bias-f2-v2-native-fp32', tag=TAG,
             created_at=datetime.now(timezone.utc).isoformat(), status='CPU_PREPARED_NATIVE_FP32_EXPECTED_NOT_YET_OBSERVED')
    m['cache'].update(dtype='float32', data_bytes=4343 * 24 * 1024 * 4,
                      format='NPY memmap with native FP32 values; no cast; header bytes extra',
                      native_dtype_observation='UNVERIFIED: prior failure only established not-FP16; next pilot must record actual dtype and require FP32')
    m['phases'].update(spent_previous_seconds=spent, prior_failure_sha256=V1_FAILURE_SHA,
                      remaining_before_v2_seconds=2700-spent, automatic_retries=0,
                      budget_scope='2700s total includes preserved v1 failed pilot + v2pilot + any separately authorized full phase')
    m['correction'] = dict(v1_manifest_sha256=V1_MANIFEST_SHA, v1_failure_sha256=V1_FAILURE_SHA,
        v1_helper_sha256=V1_HELPER_SHA, v1_runner_sha256=V1_RUNNER_SHA,
        scope='Expected native dtype and cache FP32 only; preserve source/head/inputs/loss/epochs/RNG and all exact parity gates',
        historical_failure='FAILED_UNCHANGED', actual_native_dtype='UNVERIFIED', additional_retries=0)
    new_bindings = {}
    for p, expected in old['source_bindings'].items():
        if p == 'tools/exp083_slot_bias.py': new_bindings['tools/exp083_slot_bias_v2.py'] = digest(root / 'tools/exp083_slot_bias_v2.py')
        elif p == 'training/runners/exp083_slot_bias.py': new_bindings[RUNNER] = digest(root / RUNNER)
        elif p.startswith(OLD_PROTOCOL + '/'):
            new_bindings[PROTOCOL + p[len(OLD_PROTOCOL):]] = expected
        else: new_bindings[p] = expected
    new_bindings.update(frozen)
    payload_names = ('head-initial.pt', 'centers.npy', 'slots.npy', 'ordered-ids.json', 'pilot-ids.json')
    payloads = {}
    for name in payload_names:
        p = previous / name; expected = old['source_bindings'][OLD_PROTOCOL + '/' + name]
        if digest(p) != expected: raise ValueError('Frozen payload changed: ' + name)
        payloads[name] = p.read_bytes()
    out.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        with (out / name).open('xb') as f: f.write(value)
        if digest(out / name) != old['source_bindings'][OLD_PROTOCOL + '/' + name]:
            raise ValueError('Copied payload bytes changed')
    m['source_bindings'] = new_bindings
    write_new(out / 'manifest.json', m)
    prior_commands = read(previous / 'commands.json')
    prefix = [prior_commands['pilot'][0], RUNNER, '--protocol', str(out / 'manifest.json')]
    write_new(out / 'commands.json', dict(environment={'HF_HUB_OFFLINE': '1'},
        pilot=prefix + ['--phase', 'pilot', '--out', str(out / 'pilot-v1')],
        full=prefix + ['--phase', 'full', '--out', str(out / 'run-v1'), '--pilot-receipt', str(out / 'pilot-v1/receipt.json'),
                       '--root-decision', str(out / 'root-full-decision.json')],
        root_full_decision_required_fields=dict(decision='RUN_ONE_FULL_PHASE', protocol_sha256=digest(out / 'manifest.json'),
            pilot_receipt_sha256='FILL_AFTER_ACTUAL_V2_PILOT', prior_failure_sha256=V1_FAILURE_SHA,
            spent_previous_seconds=spent, max_total_seconds=2700, automatic_retries=0),
        budget='Full remainder=2700-v1failedpilot-v2successfulpilot; no automatic retry',
        launch='Use existing gpu_job.py; CPU preparation does not run phases.'))
    return dict(status=m['status'], manifest_sha256=digest(out / 'manifest.json'),
                cache=m['cache'], phases=m['phases'], copied_payload_sha256={n:digest(out/n) for n in payload_names},
                torch_imported='torch' in sys.modules,
                cuda_initialized=('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized()),
                feature_cache_created=False, actual_native_dtype='UNVERIFIED')


def validate_registration(path, root=ROOT):
    root = Path(root); m = read(path)
    if m.get('schema') != 'exp083-slot-bias-f2-v2-native-fp32' or m.get('tag') != TAG:
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
    spent = float(m['phases']['spent_previous_seconds'])
    if (not math.isfinite(spent) or not 0 < spent < 2700
            or m['phases']['prior_failure_sha256'] != V1_FAILURE_SHA):
        raise ValueError('Prior failed-pilot budget binding missing')
    if phase == 'pilot':
        if pilot_path or decision_path:
            raise ValueError('Pilot cannot consume a full decision')
        return min(m['phases']['pilot_max_seconds'], 2700-spent)
    if phase != 'full' or not pilot_path or not decision_path:
        raise ValueError('Full requires exact successful pilot and root decision')
    pilot, decision = read(pilot_path), read(decision_path)
    if (pilot.get('status') != 'PASS' or pilot.get('phase') != 'pilot'
            or pilot.get('protocol_sha256') != digest(protocol_path)
            or pilot.get('prior_failure_sha256') != V1_FAILURE_SHA
            or pilot.get('spent_previous_seconds') != spent
            or pilot.get('pilot_ids') != m['phases']['pilot_ids']
            or pilot.get('feature_dtype') != 'float32'
            or pilot.get('observed_native_feature_dtypes') != ['torch.float32'] or pilot.get('initial_logits_exact') is not True
            or pilot.get('initial_train_logits_exact') is not True
            or pilot.get('native_cache_roundtrip_exact') is not True
            or pilot.get('native_cache_logits_exact') is not True
            or pilot.get('extraction_split_loaders') is not True
            or pilot.get('disjoint_initial_head_storage') is not True
            or pilot.get('encoder_before_sha256') != m['encoder_state_sha256']
            or pilot.get('encoder_after_sha256') != m['encoder_state_sha256']):
        raise ValueError('Pilot receipt does not match this registration')
    required = dict(decision='RUN_ONE_FULL_PHASE', protocol_sha256=digest(protocol_path),
                    pilot_receipt_sha256=digest(pilot_path), prior_failure_sha256=V1_FAILURE_SHA,
                    spent_previous_seconds=spent, max_total_seconds=2700, automatic_retries=0)
    if any(decision.get(k) != v for k, v in required.items()):
        raise ValueError('Explicit root full-phase decision mismatch')
    elapsed = float(pilot['elapsed_seconds'])
    if not math.isfinite(elapsed) or not 0 < elapsed <= m['phases']['pilot_max_seconds']:
        raise ValueError('Invalid v2 pilot elapsed budget')
    remaining = 2700 - spent - elapsed
    if not 0 < remaining < 2700:
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
