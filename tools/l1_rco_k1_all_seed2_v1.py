"""v1.0.0 additive all_seed2 entrypoint; B owns GPU launches."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.l1_rco_k1_next_v2 import backend, main
k = backend('all_seed2')
check_trainer_args = k.check_trainer_args
check_training_ids = k.check_training_ids
if __name__ == '__main__':
    main(k)
