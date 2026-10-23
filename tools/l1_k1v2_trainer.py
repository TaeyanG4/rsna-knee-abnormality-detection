"""l1_k1v2_trainer shim v1.0.1 (B, 2026-10-18) for H-B-182 K1 v2 holdout inference.
tools/l1_rco_k1_v2.py import_trainer() loads .local/scratch/l1-rco/k1v2/compact_train_l1.py from a file path under the
module name 'l1_k1v2_trainer' without registering it in sys.modules. Windows spawn DataLoader workers (num_workers=3)
pickle Studies/collate by that name, which failed on 10-18 11:53 (PicklingError). This shim changes no registered source,
recipe or value: in a process that already holds the file-loaded trainer (the parent), importing this name returns THAT
module object (pickle needs the identical class); in a fresh process (a spawn worker) it executes the same trainer file.
Changelog: v1.0.1 reuse the parent's already-loaded module (v1.0.0 re-executed it -> "not the same object");
v1.0.0 first version."""
import gc as _gc
import sys as _sys
import types as _types
from pathlib import Path as _Path

_trainer = (_Path(__file__).resolve().parent.parent / '.local' / 'scratch' / 'l1-rco' / 'k1v2' / 'compact_train_l1.py').resolve()
_loaded = [m for m in _gc.get_objects() if isinstance(m, _types.ModuleType) and m.__name__ == __name__
           and m is not _sys.modules.get(__name__) and getattr(m, '__file__', None)
           and _Path(m.__file__).resolve() == _trainer]
if _loaded:
    _sys.modules[__name__] = _loaded[0]
else:
    exec(compile(_trainer.read_bytes(), str(_trainer), 'exec'))
