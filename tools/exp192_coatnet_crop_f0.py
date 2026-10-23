"""v2.0.0: explicit CoAtNet preset and immutable historical access.

Changelog: retire patched producers, pilots, queue generation and timing gates;
scientific helpers remain, new execution uses the shared registered lifecycle.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "training/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tools import exp181_coatnet_f0_bridge as bridge
from exp192_crop import CROP_SPEC, CropStudyDataset, infer_pair

VERSION, ARM = "2.0.0", "CROP"
EXP = ROOT / "artifacts/experiments/EXP192B_COATNET_CROP_F0"
PROTOCOL = EXP / "PROTOCOL.md"
REG = EXP / "protocol-v1"
MANIFEST, VALIDATION = REG / "manifest-CROP.json", REG / "validation.json"
QUEUE, CONTROL = REG / "queue-exp192b.json", REG / "control-binding.json"
ATTEMPT, OOF_ATTEMPT = EXP / "attempt.json", EXP / "oof-attempt.json"
FIT_TIME = EXP / "fit-wall.json"
COLLECTION = EXP / "collection.json"
SCREEN = EXP / "screen/result-CROP-vs-CONTROL.json"
TRAIN_TAG, OOF_TAG = "EXP192B_CROP_F0", "EXP192B_CROP_F0_OOF"
SOURCE = ["tools/exp192_coatnet_crop_f0.py", "training/src/exp192_crop.py",
          "tools/exp192_cpu_smoke.py", "tools/exp188_coatnet_coverage_f0.py",
          "artifacts/experiments/EXP192B_COATNET_CROP_F0/PROTOCOL.md",
          "artifacts/experiments/EXP192B_COATNET_CROP_F0/protocol-v1/control-binding.json"]
BASE_CODE = tuple(bridge.CODE)
BASE_REGISTRATION = bridge.registration
CONTROL_BUDGETS = dict(bridge.BUDGETS)
CHECKPOINT_POLICY = "exp192-alternating-c140-c100"
BASE_SAVE = bridge.save_torch_new
BASE_CHECK = bridge.check_checkpoint


def save_crop_checkpoint(path, payload):
    if Path(path).name == "model_best.pt":
        payload = dict(payload, window_policy=CHECKPOINT_POLICY, crop_policy=copy.deepcopy(CROP_SPEC))
    return BASE_SAVE(path, payload)


def check_crop_checkpoint(ck, p, protocol_sha):
    BASE_CHECK(ck, p, protocol_sha)
    if ck.get("window_policy") != CHECKPOINT_POLICY or ck.get("crop_policy") != CROP_SPEC:
        raise ValueError("checkpoint lacks the registered mixed-crop serving policy")






def spec(arm):
    if arm != ARM:
        raise ValueError("EXP192 permits only CROP")
    return dict(dataset="exp192_crop.CropStudyDataset", augmentation=copy.deepcopy(CROP_SPEC))


def run_rel(arm):
    spec(arm)
    return "artifacts/experiments/EXP192B_COATNET_CROP_F0/CROP"

















































def registration(*_args, **_kwargs):
    from training.arms import coatnet_history
    return coatnet_history.read_registration(MANIFEST)


def verified_candidate():
    from training.arms import coatnet_history
    return coatnet_history.verified_arm(MANIFEST)


def main(argv=None):
    from tools import coatnet_history_cli
    return coatnet_history_cli.main(argv, preset='crop-f0-s42', protocol=MANIFEST, candidate=ARM)


if __name__ == '__main__':
    sys.exit(main())
