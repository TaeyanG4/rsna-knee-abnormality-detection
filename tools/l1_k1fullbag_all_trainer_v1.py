"""v1.0.0 spawn-safe fullbag_all trainer shim."""
from pathlib import Path
_path = Path(__file__).resolve().parents[1] / '.local/scratch/astra-b2/vast-next-v1/fullbag_all/compact_train_l1.py'
exec(compile(_path.read_bytes(), str(_path), 'exec'))
