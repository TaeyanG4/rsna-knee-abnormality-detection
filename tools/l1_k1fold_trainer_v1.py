"""v1.0.0 importable fold trainer for spawn DataLoader workers."""
import os
from pathlib import Path
_path = Path(__file__).resolve().parents[1] / '.local/scratch/astra-b2/h224-folds-v1' / ('f'+os.environ.get('K1_FOLD','1')) / 'compact_train_l1.py'
exec(compile(_path.read_bytes(), str(_path), 'exec'))
