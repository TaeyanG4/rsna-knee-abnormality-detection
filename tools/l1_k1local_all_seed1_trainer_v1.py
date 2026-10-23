"""v1.0.0 spawn-safe local_all_seed1 trainer shim."""
from pathlib import Path
_path = Path(__file__).resolve().parents[1] / '.local/scratch/astra-b2/vast-next-v1/local_all_seed1/compact_train_l1.py'
exec(compile(_path.read_bytes(), str(_path), 'exec'))
