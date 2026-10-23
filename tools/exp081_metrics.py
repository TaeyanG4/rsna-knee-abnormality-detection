"""Pure CPU metrics for the frozen EXP081 paired input comparison.

Callers own file access, cohort/provenance gates and permission to open labels.
This module consumes arrays only: no training, inference, ranking or selection.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# Passive metric helper; support package imports and the tools/ CLI entry point.
if __package__:
    from .oof_error_audit import LAB, weighted_auc
else:
    from oof_error_audit import LAB, weighted_auc

DIAGNOSTIC_FIELDS = ("scanner_group", "field", "manufacturer", "cached_mask")
PFOA = LAB.index("PF OA")
FRACTURE = LAB.index("Fracture")


def _probabilities(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or raw.ndim != 2 or raw.shape[1] != len(LAB) or not len(raw):
        raise ValueError(f"{name} must be a nonempty numeric [N,12] array")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must contain finite values in [0,1]")
    return array


def _groups(value, rows, name):
    array = np.asarray(value, dtype=object)
    if array.shape != (rows,):
        raise ValueError(f"{name} must have one scalar group per row")
    values = array.tolist()
    if all(isinstance(v, str) for v in values):
        if any(not v.strip() for v in values):
            raise ValueError(f"{name} contains a missing/empty group")
        return np.asarray(values, dtype=str)
    if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in values):
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(f"{name} contains a missing/nonfinite group")
        # Preserve integer group identity; casting large integer IDs to float can merge clusters.
        return np.asarray(values, dtype=object)
    raise ValueError(f"{name} must contain only nonempty strings or only finite numbers")


def _mean_complete(values):
    return None if any(v is None for v in values) else float(np.mean(values))


def _aggregates(rows):
    result = {}
    for arm in ("baseline", "candidate", "delta"):
        values = [row["delta"] if arm == "delta" else row["auc"][arm] for row in rows]
        result[arm] = dict(macro12=_mean_complete(values), PFOA=values[PFOA],
                          other11=_mean_complete([v for j, v in enumerate(values) if j != PFOA]),
                          Fracture=values[FRACTURE])
    return result


def _summary(binary, baseline, candidate):
    targets = []
    weights = np.ones(len(binary), dtype=float)
    for index, target in enumerate(LAB):
        positive = int(binary[:, index].sum())
        negative = len(binary) - positive
        if positive and negative:
            first = weighted_auc(binary[:, index], baseline[:, index], weights)
            second = weighted_auc(binary[:, index], candidate[:, index], weights)
            delta = second - first
        else:
            first = second = delta = None
        targets.append(dict(target=target, P=positive, N=negative,
                            auc=dict(baseline=first, candidate=second), delta=delta,
                            macro_contribution=None if delta is None else delta / len(LAB)))
    return dict(studies=len(binary), targets=targets, aggregates=_aggregates(targets))


def _conditional_intervals(binary, baseline, candidate, groups, reps, seed):
    unique, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, len(LAB)), dtype=float)
    for draw in range(reps):
        # Same complete report-group draw supplies both arms and every target.
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[codes]
        for index in range(len(LAB)):
            first = weighted_auc(binary[:, index], baseline[:, index], weights)
            second = weighted_auc(binary[:, index], candidate[:, index], weights)
            draws[draw, index] = second - first

    def interval(values):
        finite = values[np.isfinite(values)]
        return dict(requested_reps=reps, valid_reps=int(len(finite)),
                    low=float(np.quantile(finite, .025)) if len(finite) else None,
                    high=float(np.quantile(finite, .975)) if len(finite) else None)

    return dict(unit="report_group_within_fold", groups=int(len(unique)), seed=seed,
                contrast="candidate_minus_baseline", requested_reps=reps,
                macro12=interval(draws.mean(axis=1)), PFOA=interval(draws[:, PFOA]),
                other11=interval(np.delete(draws, PFOA, axis=1).mean(axis=1)),
                Fracture=interval(draws[:, FRACTURE]),
                targets={target: interval(draws[:, j]) for j, target in enumerate(LAB)},
                scope="Conditional on fixed raw probabilities and exposed report labels. No reranking; excludes label, seed, prior-selection, patient-independence and private-distribution uncertainty; no multiplicity correction.")


def evaluate(y, baseline, candidate, report_groups, changed_mask, diagnostic_groups=None,
             reps=500, seed=20261003):
    """Return JSON-safe paired metrics without reading or modifying any input.

    The primary estimand is the all-row, complete-12 mean AUC difference.
    Changed/unchanged and existing metadata groups are descriptive only.
    ``diagnostic_groups`` is an optional mapping of allowed field names to vectors.
    """
    labels = _probabilities(y, "labels")
    baseline = _probabilities(baseline, "baseline")
    candidate = _probabilities(candidate, "candidate")
    if labels.shape != baseline.shape or labels.shape != candidate.shape:
        raise ValueError("labels and both predictions must have the same shape")
    rows = len(labels)
    groups = _groups(report_groups, rows, "report_groups")
    mask = np.asarray(changed_mask)
    if (mask.shape != (rows,) or mask.dtype.kind not in "biuf" or not np.isfinite(mask).all()
            or not np.isin(mask, [0, 1]).all()):
        raise ValueError("changed_mask must be a finite binary vector with one entry per row")
    mask = mask.astype(bool)
    if isinstance(reps, (bool, np.bool_)) or not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError("reps must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    reps, seed = int(reps), int(seed)
    if diagnostic_groups is None:
        diagnostic_groups = {}
    if not isinstance(diagnostic_groups, Mapping) or not set(diagnostic_groups) <= set(DIAGNOSTIC_FIELDS):
        raise ValueError("diagnostic_groups supports only scanner_group, field, manufacturer, cached_mask")
    diagnostics = {field: _groups(diagnostic_groups[field], rows, field)
                   for field in DIAGNOSTIC_FIELDS if field in diagnostic_groups}
    # Preserve the parent's exact rounding contract, including float32 half ties.
    binary = np.rint(labels.astype(np.float32)).astype(np.uint8)
    full = _summary(binary, baseline, candidate)
    subsets = {}
    for name, take in (("changed", mask), ("unchanged", ~mask)):
        subsets[name] = _summary(binary[take], baseline[take], candidate[take])
    group_rows = []
    for field, values in diagnostics.items():
        for group in np.unique(values):
            take = values == group
            summary = _summary(binary[take], baseline[take], candidate[take])
            group_rows.append(dict(field=field, group=group.item() if isinstance(group, np.generic) else group,
                                   **summary))
    return dict(schema="exp081-paired-raw-probability-metrics-v1", studies=rows, targets_order=list(LAB),
                label_rule="float32 then numpy.rint; 0.5 rounds to 0", prediction_scale="raw probabilities; no reranking",
                full=full, conditional_intervals=_conditional_intervals(binary, baseline, candidate, groups, reps, seed),
                subsets=subsets, diagnostic_groups=group_rows,
                decision="Descriptive paired comparison only; no tuned gate, routing, selection or automatic deployment.")
