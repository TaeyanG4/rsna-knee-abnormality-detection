"""H-B-147 progress v1.0.0 (2026-10-13).

Changelog: phase-aware completed-work heartbeats; no timer thread or CUDA polling.
"""
import json
import os
from pathlib import Path
import time


def atomic_json(path, value):
    """Bounded publication, including Windows readers lacking delete sharing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    deadline = time.monotonic() + 3
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)


class Progress:
    def __init__(self, path=None, token=None):
        self.path = Path(path) if path else None
        self.token = token
        self.sequence = 0
        self.phase = "startup"
        self.completed_at = None
        self.last_write = 0.

    def phase_start(self, phase):
        self.phase = phase
        self._write("enter", None)

    def completed(self, phase, detail=None, *, throttle=False):
        now = time.monotonic()
        if throttle and now - self.last_write < .5:
            return
        self.phase = phase
        self.sequence += 1
        self.completed_at = now
        self._write("completed", detail)

    def _write(self, event, detail):
        if self.path is None:
            return
        record = dict(schema="hb147-progress/v1", token=self.token, pid=os.getpid(),
                      sequence=self.sequence, phase=self.phase, event=event,
                      completed_at=self.completed_at, detail=detail)
        atomic_json(self.path, record)
        self.last_write = time.monotonic()


PROGRESS = Progress(os.environ.get("HB147_HEARTBEAT"), os.environ.get("HB147_TOKEN"))


class ProgressWriter:
    """torch.save file adapter: emit only after real writes return; 4 MiB chunks.

    No heartbeat while a write/flush, CPU copy or CUDA call is blocked. This
    preserves the zip serialization and does not spawn another writer thread.
    """
    def __init__(self, stream, progress=PROGRESS):
        self.stream, self.progress, self.written = stream, progress, 0

    def write(self, data):
        view = memoryview(data).cast("B")
        total = 0
        while total < len(view):
            size = self.stream.write(view[total:total + 4 * 1024 * 1024])
            if not size:
                raise OSError("checkpoint write made no progress")
            total += size
            self.written += size
            self.progress.completed("checkpoint_write", {"bytes": self.written}, throttle=True)
        return total

    def flush(self):
        self.stream.flush()
        self.progress.completed("checkpoint_write", {"bytes": self.written, "flushed": True})

    def __getattr__(self, name):
        return getattr(self.stream, name)
