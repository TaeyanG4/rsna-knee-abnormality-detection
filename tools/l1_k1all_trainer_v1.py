"""v1.0.0 all-data trainer shim for spawn workers; never imports fold outputs."""
from pathlib import Path
_path = Path(__file__).resolve().parents[1] / '.local/scratch/astra-b2/h224-all-v1/all/compact_train_l1.py'
exec(compile(_path.read_bytes(), str(_path), 'exec'))
