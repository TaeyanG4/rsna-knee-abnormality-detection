"""v1.0.0: strict training-only study/target weights and batch-target normalization.

Changelog: missing cells default to one; reject leakage and malformed input.
No labels are printed, no GPU initialization, no training entry point.
"""
import csv
import math
from pathlib import Path

VERSION = "1.0.0"


def load_weights(path, train, holdout, targets):
    train, holdout = set(train), set(holdout)
    if train & holdout:
        raise ValueError("overlapping training/holdout roster")
    values = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["study_uid", "target", "weight"]:
            raise ValueError("weights require exact study_uid,target,weight columns")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("malformed weight row")
            uid, target = row["study_uid"], row["target"]
            if uid in holdout:
                raise ValueError("holdout UID in weights; fail closed")
            if uid not in train or target not in targets:
                raise ValueError("unknown training UID or target")
            key = (uid, target)
            if key in values:
                raise ValueError("duplicate study/target weight")
            value = float(row["weight"])
            # Positive float32 range also prevents underflow/overflow in normalization.
            if not math.isfinite(value) or not 1e-12 <= value <= 1e12:
                raise ValueError("weight must be finite and positive in [1e-12,1e12]")
            values[key] = value
    return values


def tensor_weights(path, train, holdout, targets, pool, like):
    import torch
    values = load_weights(path, train, holdout, targets)
    if len(pool) != len(set(pool)) or not set(pool) <= set(train):
        raise ValueError("weight lookup pool must be unique training studies")
    result = torch.tensor([[values.get((uid, target), 1.) for target in targets]
                           for uid in pool], dtype=like.dtype, device=like.device)
    if result.shape != like.shape:
        raise ValueError("target/weight shape mismatch")
    return result


def normalize(weight):
    """Each target's batch mean is one; all-one tensors remain bit identical."""
    return weight / weight.mean(dim=0, keepdim=True)
