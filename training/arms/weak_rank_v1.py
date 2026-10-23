"""H-B-122 loss v1.0.0: fixed BCE + .10 weak-target ranking (2026-10-13).

Enumerate at most 28 ordered high/low pairs per target in a batch of eight.
The expectation of bounded uniform pair sampling is computed exactly: 1/2
unrestricted, 1/4 nuisance-low, 1/4 nuisance-high. Each empty stratum falls
back to unrestricted. Gap weights multiply losses (not normalized away).
Targets without any eligible pairs contribute zero to the fixed four-way mean.
No random draws, hard mining, validation input, or changes to batch exposure.
"""
import torch
from torch.nn import functional as F

VERSION = "1.0.0"
MAPPING = ((3, 2), (5, 4), (6, 4), (8, 7))
CONTRACT = dict(version=VERSION, coefficient=.10, gap=.5,
                targets=[t for t, _ in MAPPING], nuisance=[n for _, n in MAPPING],
                threshold=.5, different_components=True, max_batch=8,
                mixture=[.5, .25, .25], empty_stratum="unrestricted",
                empty_target="zero_in_fixed_four_mean", gap_weight="multiply_then_mean",
                sampling="exact_expectation_all_batch_pairs_no_rng")


def pair_sets(labels, components):
    """CPU metadata only. Returns index pairs and aggregate coverage, no rows."""
    y = torch.as_tensor(labels).detach().cpu().float()
    if (y.ndim != 2 or y.shape[1] != 12 or not 1 <= len(y) <= 8
            or len(components) != len(y) or any(not isinstance(c, str) or not c for c in components)
            or not bool(torch.isfinite(y).all()) or not bool(((y >= 0) & (y <= 1)).all())):
        raise ValueError("invalid batch labels/components")
    different = torch.tensor([[a != b for b in components] for a in components])
    sets, coverage = {}, {}
    for target, nuisance in MAPPING:
        gap = y[:, target, None] - y[None, :, target]
        eligible = (gap >= .5) & different
        strata = y[:, nuisance] >= .5
        unrestricted = eligible.nonzero(as_tuple=True)
        low = (eligible & ~strata[:, None] & ~strata[None, :]).nonzero(as_tuple=True)
        high = (eligible & strata[:, None] & strata[None, :]).nonzero(as_tuple=True)
        sets[target] = (unrestricted, low, high, gap)
        coverage[str(target)] = dict(eligible=len(unrestricted[0]), low=len(low[0]),
                                    high=len(high[0]), fallback_strata=int(not len(low[0])) + int(not len(high[0])))
    return sets, coverage


def ranking_loss(logits, labels, components):
    if logits.ndim != 2 or logits.shape != labels.shape or not bool(torch.isfinite(logits).all()):
        raise ValueError("invalid logits")
    sets, coverage = pair_sets(labels, components)
    terms = []
    for target, _ in MAPPING:
        unrestricted, low, high, gap = sets[target]
        if not len(unrestricted[0]):
            terms.append(logits[:, target].float().sum() * 0)
            continue
        means = []
        for pairs in (unrestricted, low, high):
            i, j = pairs if len(pairs[0]) else unrestricted
            weights = gap[i, j].to(logits.device)
            i, j = i.to(logits.device), j.to(logits.device)
            means.append((weights * F.softplus(-(logits[i, target].float() - logits[j, target].float()))).mean())
        terms.append(.5 * means[0] + .25 * means[1] + .25 * means[2])
    return torch.stack(terms).mean(), coverage


def objective(logits, labels, pos_weight, components, *, enabled):
    bce = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight)
    if not enabled:
        return bce, {}  # bit-identical BCE, no added zero or RNG use
    rank, coverage = ranking_loss(logits, labels, components)
    return bce + CONTRACT["coefficient"] * rank, coverage
