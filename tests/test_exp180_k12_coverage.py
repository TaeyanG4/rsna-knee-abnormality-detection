"""Synthetic metadata checks for EXP180; no production images or labels."""
import numpy as np
import pytest

from tools import exp180_k12_coverage as x


def fixture_slots(mask, fallback=False):
    spec = {"slots": [["Sagittal", 1, 5], ["Sagittal", 0, 5], ["Coronal", 1, 5],
                      ["Coronal", 0, 5], ["Axial", -1, 5]]}
    selected = [{"series": f"series{i}", "protocol_flag": (1 if i == 1 and fallback else
                 int(slot[1]) if int(slot[1]) >= 0 else 0)} for i, slot in enumerate(spec["slots"])]
    return spec, x.provenance(spec, selected, mask, "test", {})


@pytest.mark.parametrize("depth,sizes", [(64, (18, 14, 12, 8, 12)),
                                        (96, (26, 22, 18, 12, 18))])
@pytest.mark.parametrize("alteration", ["swap_contrasts", "swap_planes", "redistribute_depth"])
def test_registered_cache_layout_rejects_misidentified_groups(depth, sizes, alteration):
    spec = {"depth": depth, "slots": [[plane, fluid, size] for (plane, fluid), size in zip(
        (("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", -1)),
        sizes)]}
    mask = np.ones(depth, np.uint8)
    selected = [{"series": f"series{i}", "protocol_flag": max(0, slot[1])}
                for i, slot in enumerate(spec["slots"])]
    assert [slot["group"] for slot in x.provenance(spec, selected, mask, "test", {})] == list(x.GROUPS)

    if alteration == "swap_contrasts":
        spec["slots"][:2] = spec["slots"][1], spec["slots"][0]
        selected[:2] = selected[1], selected[0]
    elif alteration == "swap_planes":
        spec["slots"][0][0], spec["slots"][2][0] = spec["slots"][2][0], spec["slots"][0][0]
    else:
        spec["slots"][0][2] -= 1
        spec["slots"][1][2] += 1
    with pytest.raises(ValueError, match="cache slot layout mismatch"):
        x.provenance(spec, selected, mask, "test", {})


def test_fallback_copy_is_cache_loss_and_never_independent_coverage():
    mask = np.ones(25, np.uint8)
    _, slots = fixture_slots(mask, fallback=True)
    state = x.availability(mask, slots)
    assert state["sagittal_fluid"] == "available"
    assert state["sagittal_other"] == "fallback_only"
    event, covered, omitted = x.audit_bag([1, 1, 6] + [-1] * 9, mask, slots, state)
    assert covered == {"sagittal_fluid"}
    assert "sagittal_other" not in omitted
    assert event["fallback_windows"] == 1
    assert event["duplicate_centres"] == 1
    assert event["empty_windows"] == 9


def test_absent_pick_slot_is_missing_series():
    mask = np.ones(25, np.uint8)
    spec, _ = fixture_slots(mask)
    selected = [{"series": f"series{i}", "protocol_flag": int(slot[1]) if slot[1] >= 0 else 0}
                for i, slot in enumerate(spec["slots"])]
    selected[1] = None
    state = x.availability(mask, x.provenance(spec, selected, mask, "test", {}))
    assert state["sagittal_other"] == "missing_series"


def test_available_group_omission_and_cache_mask_loss_are_separate():
    mask = np.ones(25, np.uint8)
    mask[11:14] = 0
    spec, slots = fixture_slots(mask)
    state = x.availability(mask, slots)
    assert state["coronal_fluid"] == "no_valid_triplet"
    event, covered, omitted = x.audit_bag([1] * 12, mask, slots, state)
    assert "coronal_fluid" not in omitted
    assert "sagittal_other" in omitted
    assert event["bags_omitting_any"] == 1
    assert event["duplicate_centres"] == 11
    assert covered == {"sagittal_fluid"}
    sampled = x.select_centers(mask, 25, 12, spec, "within-slot-valid", x.study_rng(42, 0, "test"))
    assert len(sampled) == 12
    assert all(len({x.slot_index(j, slots) for j in (c - 1, c, c + 1)}) == 1 and
               np.all(mask[c - 1:c + 2] > 0) for c in sampled)


def test_legacy_cross_boundary_and_gap_do_not_credit_a_group():
    mask = np.ones(25, np.uint8)
    _, slots = fixture_slots(mask)
    state = x.availability(mask, slots)
    event, covered, omitted = x.audit_bag([4, 7], mask, slots, state)
    assert event["boundary_windows"] == event["mixed_series_windows"] == 1
    assert covered == {"sagittal_other"}
    assert "sagittal_fluid" in omitted
    mask[5] = 0
    event, covered, _ = x.audit_bag([4], mask, slots, x.availability(mask, slots))
    assert event["boundary_windows"] == 1
    assert event["mixed_series_windows"] == 0
    assert not covered


def test_replay_counts_and_strata_join_use_only_id_membership():
    mask = np.ones(25, np.uint8)
    spec, slots = fixture_slots(mask)
    data = {"a": (mask, slots), "b": (mask, slots)}
    folds = {0: ["a"], 1: ["a", "b"]}
    result, per_uid = x.replay_recipe("synthetic", spec, data, folds, (42,), 2, "within-slot-valid")
    assert result["summary"]["studies"] == 2
    assert result["summary"]["bags"] == result["expected_bags"] == 6
    assert result["per_fold"]["0"]["bags"] == 2
    joined = x.join_strata(per_uid, "synthetic", {"synthetic": {"ACL": {"FN": ["a", "missing"]}}},
                           "native_model")
    bucket = joined["targets"]["ACL"]["FN"]
    assert bucket["source_studies"] == 2 and bucket["matched_studies"] == 1
    assert bucket["bags"] == 4
