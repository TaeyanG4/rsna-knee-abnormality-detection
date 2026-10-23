"""CPU-only synthetic checks for the registered EXP188 training-bag sampler."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "src"))

from exp188_coverage import coverage_centres, eligible_centres, independent_groups
from raptor_common import study_rng, train_centers


SIZES = (26, 22, 18, 12, 18)
NAMES = ("sagittal_fluid", "sagittal_other", "coronal_fluid", "coronal_other", "axial")


def slots(fallback=()):
    out, start = [], 0
    for name, size in zip(NAMES, SIZES):
        out.append(dict(group=name, start=start, stop=start + size,
                        series="synthetic", fallback=name in fallback))
        start += size
    return out


def assert_unique(centres):
    real = [c for c in centres if c >= 0]
    assert len(real) == len(set(real))


def test_all_available_groups_covered_with_fixed_k():
    mask = np.ones(96, dtype=np.uint8)
    groups = independent_groups(mask, slots())
    assert len(groups) == 5
    for epoch in range(100):
        centres = coverage_centres(mask, slots(), 12, study_rng(42, epoch, "synthetic"))
        assert len(centres) == 12
        assert_unique(centres)
        for _, eligible in groups:
            assert set(centres) & set(eligible)


def test_control_candidate_population_and_determinism():
    mask = np.ones(96, dtype=np.uint8)
    mask[13:17] = 0
    mask[50:53] = 0
    expected = set(eligible_centres(mask))
    control = train_centers(mask, 96, 12, study_rng(42, 3, "same"))
    a = coverage_centres(mask, slots(), 12, study_rng(42, 3, "same"))
    b = coverage_centres(mask, slots(), 12, study_rng(42, 3, "same"))
    assert a == b
    assert set(control) <= expected
    assert set(a) <= expected
    assert_unique(a)


def test_zero_eligible_group_and_fallback_never_credit_coverage():
    mask = np.ones(96, dtype=np.uint8)
    mask[48:66] = 0  # coronal fluid has no eligible centre
    groups = independent_groups(mask, slots(fallback={"sagittal_other"}))
    names = {name for name, _ in groups}
    assert "coronal_fluid" not in names
    assert "sagittal_other" not in names
    assert len(groups) == 3
    for epoch in range(20):
        centres = coverage_centres(mask, slots(fallback={"sagittal_other"}), 12,
                                   study_rng(42, epoch, "fallback"))
        assert_unique(centres)
        for _, eligible in groups:
            assert set(centres) & set(eligible)


def test_more_available_groups_than_k():
    mask = np.ones(96, dtype=np.uint8)
    centres = coverage_centres(mask, slots(), 3, study_rng(42, 0, "small-k"))
    assert len(centres) == 3
    assert_unique(centres)
    assert sum(bool(set(centres) & set(eligible))
               for _, eligible in independent_groups(mask, slots())) == 3


def test_fewer_eligible_centres_than_k_empty_padding():
    mask = np.zeros(96, dtype=np.uint8)
    mask[5:8] = 1
    centres = coverage_centres(mask, slots(), 12, study_rng(42, 0, "underfilled"))
    assert len(centres) == 12
    assert set(c for c in centres if c >= 0) == {5, 6, 7}
    assert centres.count(-1) == 9
    assert_unique(centres)


def test_no_eligible_centres():
    mask = np.zeros(96, dtype=np.uint8)
    assert coverage_centres(mask, slots(), 12, study_rng(42, 0, "empty")) == [-1] * 12
