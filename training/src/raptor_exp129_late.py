"""EXP129 LATE: observe one fixed training trajectory without RNG or model mutation.

SWA means arithmetic FP32 averaging of the final three complete state dicts,
including floating BatchNorm running statistics; integer buffers use the final
epoch. No BatchNorm recalibration. EMA starts from the model after the caller's
registered warmup, smooths floating parameters after successful optimizer steps,
and copies all buffers (including BatchNorm statistics) from that same step.
EMA shadows stay on the model device; checkpoint/finalization alone copy to CPU.
Engineering smoke uses epochs=3; it is not a registered 20-epoch experiment.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch


def _clone_state(state, *, cpu=False):
    return {k: (v.detach().cpu() if cpu else v.detach()).clone() for k, v in state.items()}


def _validate_state(state, reference=None, *, finite=False):
    if not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('LATE requires a nonempty tensor-only state dict')
    if reference is not None:
        if state.keys() != reference.keys():
            raise ValueError('LATE state keys changed')
        for key in state:
            if state[key].shape != reference[key].shape or state[key].dtype != reference[key].dtype:
                raise ValueError(f'LATE state shape/dtype changed: {key}')
    for key, value in state.items():
        if value.is_complex():
            raise ValueError(f'LATE does not support complex state: {key}')
        if finite and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'nonfinite LATE state: {key}')


def state_sha256(state):
    """EXP099-style sorted key+raw-byte hash, including bfloat16 safely."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(key.encode())
        digest.update(raw)
    return digest.hexdigest()


def average_states(states):
    """FP32 arithmetic mean; nonfloating state is copied from the last epoch."""
    if len(states) != 3:
        raise ValueError('LATE requires exactly three epoch states')
    reference = states[-1]
    for state in states:
        _validate_state(state, reference, finite=True)
    result = {}
    for key, final in reference.items():
        if final.is_floating_point():
            total = states[0][key].detach().cpu().to(torch.float32).clone()
            total.add_(states[1][key].detach().cpu().to(torch.float32))
            total.add_(final.detach().cpu().to(torch.float32))
            result[key] = total.div_(3).to(final.dtype)
        else:
            result[key] = final.detach().cpu().clone()
    _validate_state(result, reference, finite=True)
    return result


def _save_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: a rerun may never overwrite a frozen checkpoint.
    with path.open('xb') as stream:
        torch.save(payload, stream)


class LateTracker:
    """Call update only after the optimizer success/skip decision, then end_epoch.

    Counter `attempted_updates` includes skipped optimizer attempts; the EMA's
    `successful_updates` must equal scheduled updates minus fp16 skips. Tracking
    never changes training mode, RNG state, parameters, buffers, gradients, or
    optimizer/scheduler state. Instantiate after the original warmup forward.
    """

    def __init__(self, model, epochs=20, decay=.998):
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 3:
            raise ValueError('LATE epochs must be an integer >= 3')
        if not 0 < decay < 1:
            raise ValueError('LATE decay must be between zero and one')
        state = model.state_dict()
        _validate_state(state, finite=True)
        self.epochs, self.decay = epochs, float(decay)
        self.selected_epochs = tuple(range(epochs - 2, epochs + 1))
        self.parameter_keys = frozenset(k for k, _ in model.named_parameters())
        self.ema = _clone_state(state)
        self.initial_state_sha256 = state_sha256(state)
        self.epoch_states = {}
        self.last_epoch = 0
        self.attempted_updates = self.successful_updates = self.skipped_updates = 0

    @torch.no_grad()
    def update(self, model, successful=True):
        if self.last_epoch == self.epochs:
            raise ValueError('LATE cannot update after final epoch')
        if not isinstance(successful, bool):
            raise ValueError('successful must be an explicit bool')
        state = model.state_dict()
        _validate_state(state, self.ema)
        if any(state[key].device != self.ema[key].device for key in state):
            raise ValueError('LATE model device changed')
        self.attempted_updates += 1
        if not successful:
            self.skipped_updates += 1
            return
        for key, value in state.items():
            if key in self.parameter_keys and value.is_floating_point():
                self.ema[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.ema[key].copy_(value.detach())
        self.successful_updates += 1

    def end_epoch(self, model, epoch_1based, out_dir, checkpoint_meta):
        if (isinstance(epoch_1based, bool) or not isinstance(epoch_1based, int)
                or epoch_1based != self.last_epoch + 1 or epoch_1based > self.epochs):
            raise ValueError('LATE epochs must be completed exactly once in consecutive order')
        if epoch_1based in self.selected_epochs:
            state = _clone_state(model.state_dict(), cpu=True)
            _validate_state(state, self.ema, finite=True)
            meta = copy.deepcopy(checkpoint_meta)
            if 'model' in meta:
                raise ValueError('checkpoint metadata must not supply model')
            meta.update(model=state, epoch=epoch_1based - 1,
                        late={'arm': 'final_epoch_snapshot', 'epoch_1based': epoch_1based,
                              'state_sha256': state_sha256(state),
                              'successful_updates': self.successful_updates})
            _save_exclusive(Path(out_dir) / f'epoch_{epoch_1based}.pt', meta)
            self.epoch_states[epoch_1based] = state
        self.last_epoch = epoch_1based

    def final_states(self):
        if self.last_epoch != self.epochs or tuple(self.epoch_states) != self.selected_epochs:
            raise ValueError('LATE finalization requires all final three epochs')
        _validate_state(self.ema, finite=True)
        return {'swa': average_states(list(self.epoch_states.values())),
                'ema': _clone_state(self.ema, cpu=True)}

    def metadata(self, expected_updates=None):
        if expected_updates is not None and self.successful_updates != expected_updates:
            raise ValueError('LATE EMA successful update count differs from expected count')
        states = self.final_states()
        return {'epochs': self.epochs, 'selected_epochs_1based': list(self.selected_epochs),
                'scope': 'registered_20_epoch' if self.epochs == 20 else 'engineering_smoke_only',
                'swa_policy': 'arithmetic_fp32_floating_parameters_and_buffers; nonfloat_final_epoch',
                'batchnorm_policy': 'SWA averaged running statistics; no recalibration; EMA latest successful-step buffers',
                'ema_policy': 'floating_parameters_only; buffers_latest_successful_step; initialized_after_registered_warmup',
                'ema_decay': self.decay, 'initial_state_sha256': self.initial_state_sha256,
                'attempted_updates': self.attempted_updates,
                'successful_updates': self.successful_updates, 'skipped_updates': self.skipped_updates,
                'expected_successful_updates': expected_updates,
                'epoch_state_sha256': {str(ep): state_sha256(s) for ep, s in self.epoch_states.items()},
                'final_state_sha256': {arm: state_sha256(s) for arm, s in states.items()}}

    def save_final(self, out_dir, checkpoint_meta, expected_updates=None):
        states = self.final_states()
        metadata = self.metadata(expected_updates)
        if 'model' in checkpoint_meta:
            raise ValueError('checkpoint metadata must not supply model')
        for arm in states:
            path = Path(out_dir) / f'model_{arm}.pt'
            if path.exists():
                raise FileExistsError(f'LATE checkpoint already exists: {path}')
        for arm, state in states.items():
            payload = copy.deepcopy(checkpoint_meta)
            payload.update(model=state, epoch=self.epochs - 1,
                           late={**metadata, 'arm': arm, 'state_sha256': metadata['final_state_sha256'][arm]})
            _save_exclusive(Path(out_dir) / f'model_{arm}.pt', payload)
        return metadata
