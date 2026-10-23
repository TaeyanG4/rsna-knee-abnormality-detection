"""v1.0.0: importable isolated failure-weight trainer for spawn workers."""
from tools.l1_rco_k1_failure_v1 import WORK
_path = WORK / "compact_train_l1.py"
exec(compile(_path.read_bytes(), str(_path), "exec"))
