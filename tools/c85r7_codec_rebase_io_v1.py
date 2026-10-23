"""v1.0.0: bounded atomic writes for the new codec-base packaging/flip adapter."""
from pathlib import Path
import os
import time

def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.new')
    for attempt in range(40):
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
            return
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32)) or attempt == 39:
                raise
            time.sleep(.05)

def atomic_text(path, text):
    atomic_write(path, text.encode('utf-8'))
