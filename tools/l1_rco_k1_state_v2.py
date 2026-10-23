"""L1 RCO K1 v2.0.0: v1 resumable trainer state + best-epoch copy + collapse stop.

Changelog v2.0.0 (K1 v2 rerun after the v1 epoch-3 collapse, D-B-214):
- best-loss epoch checkpoint: after each committed epoch-end checkpoint, a
  byte-identical copy is kept when that epoch has the strictly lowest epoch loss
  so far (first minimum), so a pre-collapse state survives the rolling file;
- ckpt/epochs.json sidecar with the committed per-epoch loss log for the
  launcher's collapse backstop (small file; no checkpoint read by the launcher);
- registered collapse guard (rule in tools/l1_rco_k1_v2.collapse_reason): the
  trainer stops itself with exit code 4 AFTER the committed epoch checkpoint and
  best copy, writes collapse.json and never reaches the final model export. A
  resumed run whose committed log already collapsed stops before any new step.
Everything else is the reviewed v1.0.0 state (subclass; Monitor, ThroughputGuard
and atomic_torch_save are re-exported unchanged). No GPU launch entrypoint.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path

from tools.l1_rco_stage0_v1 import need, read, write, replace
from tools.l1_rco_k1_state_v1 import Monitor, ThroughputGuard, atomic_torch_save  # noqa: F401  (re-exported)
from tools.l1_rco_k1_state_v1 import TrainerState as TrainerStateV1

VERSION = '2.0.0'
BEST = 'ckpt/best_E_prior_s0_f0.pt'
BEST_META = 'ckpt/best.json'
EPOCHS = 'ckpt/epochs.json'
COLLAPSE = 'collapse.json'
COLLAPSE_EXIT = 4


def best_epoch(log):
    """First index of the minimum committed epoch loss (strict improvement)."""
    return min(range(len(log)), key=lambda e: (log[e], e)) if log else None


def copy_file_atomic(src, dst):
    """Byte copy with fsync, then bounded-retry atomic replace; returns SHA256."""
    src, dst = Path(src), Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp'); h = hashlib.sha256()
    with src.open('rb') as a, tmp.open('wb') as b:
        for chunk in iter(lambda: a.read(8 << 20), b''):
            h.update(chunk); b.write(chunk)
        b.flush(); os.fsync(b.fileno())
    replace(tmp, dst)
    return h.hexdigest()


class TrainerState(TrainerStateV1):
    def __init__(self, *args, guard_constant=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_constant = guard_constant
        # Completed epochs held by the rolling checkpoint when it is at a boundary.
        self.committed = self.start_ep if not self.batch else None
        self.reconcile_best()
        self.write_epochs()
        self.check_guard()  # resumed collapsed log: stop before any optimizer step

    def reconcile_best(self):
        """Copy the rolling checkpoint when it is the boundary of the best epoch.

        Idempotent: a crash between the rolling save and this copy is repaired at
        resume, because no optimizer step can happen before this call.
        """
        if not self.log or self.committed is None or self.committed != len(self.log):
            return
        epoch = best_epoch(self.log)
        if epoch != len(self.log) - 1:
            return
        meta_path = self.out / BEST_META
        if meta_path.exists() and (self.out / BEST).exists():
            meta = read(meta_path)
            if meta.get('epoch') == epoch and meta.get('step_total') == self.step_total:
                return
        self.monitor.phase('checkpoint')
        digest = copy_file_atomic(self.path, self.out / BEST)
        write(meta_path, dict(version=VERSION, epoch=epoch, loss=self.log[epoch], log=list(self.log),
                              step_total=self.step_total, committed_epochs=self.committed, sha256=digest,
                              source='byte copy of the committed epoch-end rolling checkpoint (full training state incl. EMA)'))
        self.monitor.phase('train')

    def write_epochs(self):
        epoch = best_epoch(self.log)
        write(self.out / EPOCHS, dict(version=VERSION, committed_epochs=len(self.log), log=list(self.log),
                                      step_total=self.step_total, best_epoch=epoch,
                                      best_loss=None if epoch is None else self.log[epoch],
                                      guard_constant=self.guard_constant))

    def check_guard(self):
        if self.guard_constant is None:
            return  # CPU resume fixtures only; production argv always carries the registered constant
        from tools.l1_rco_k1_v2 import collapse_reason
        reason = collapse_reason(self.log, self.guard_constant)
        if not reason:
            return
        receipt = dict(reason, version=VERSION, status='FAILED', source='trainer epoch-end guard',
                       step_total=self.step_total, committed_epochs=len(self.log), log=list(self.log),
                       checkpoint=str(self.path), checkpoint_kept=self.path.exists(),
                       best_checkpoint=str(self.out / BEST) if (self.out / BEST).exists() else None,
                       best=read(self.out / BEST_META) if (self.out / BEST_META).exists() else None)
        write(self.out / COLLAPSE, receipt)
        print('COLLAPSE GUARD: epoch %d loss %.5f; %s; checkpoint kept, stopping with exit %d' % (
            reason['epoch'], reason['loss'], ' | '.join(reason['criteria']), COLLAPSE_EXIT), flush=True)
        self.monitor.phase('collapse-stop')
        raise SystemExit(COLLAPSE_EXIT)

    def epoch_end(self, epoch, log):
        super().epoch_end(epoch, log)  # rolling checkpoint committed at (epoch + 1, batch 0)
        self.committed = epoch + 1
        need(len(self.log) == self.committed, 'epoch log/checkpoint boundary mismatch')
        self.reconcile_best()
        self.write_epochs()
        self.check_guard()
