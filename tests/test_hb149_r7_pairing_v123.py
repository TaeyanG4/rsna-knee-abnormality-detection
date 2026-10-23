"""Independent entry-point rejection regressions for the v1.2.3 fix.

The complete v112 fixture deliberately has no migration and no contract patches.
"""
import pytest
from test_hb149_r7_flip_v120 import paired_artifacts
from test_hb149_r7_pairing_v122 import migrated
from tools import hb149_r7_pairing as pair


def invoke(entry, artifacts):
    selection, baseline, *_ = artifacts
    if entry == "validate":
        return pair.main(["validate", "--selection", str(selection), "--baseline", str(baseline)])
    if entry == "readout":
        return pair.readout(*artifacts, draws=100)
    return pair.selected_uids(selection, baseline,
                              pair.read_json(baseline / "identity.json")["uids"], require_complete=True)


@pytest.mark.parametrize("entry", ["validate", "readout", "inference"])
def test_complete_v112_without_lineage_rejected(paired_artifacts, entry):
    _, baseline, *_ = paired_artifacts
    assert pair.read_json(baseline / "identity.json")["version"] == "1.1.2"
    receipt = pair.read_json(baseline / "run_receipt.json")
    assert receipt["completed"] == receipt["selected"] == 100
    assert not (baseline / "carry_over_manifest.json").exists()
    with pytest.raises(ValueError, match="unregistered v113 baseline identity"):
        invoke(entry, paired_artifacts)


@pytest.mark.parametrize("entry", ["validate", "readout", "inference"])
@pytest.mark.parametrize("mutation", ["partial", "carry_count", "missing_manifest"])
def test_v113_requires_completion_and_lineage(migrated, entry, mutation):
    _, baseline, *_ = migrated
    if mutation == "partial":
        path = baseline / "run_receipt.json"
        value = pair.read_json(path)
        value["completed"] -= 1
    else:
        path = baseline / "carry_over_manifest.json"
        value = pair.read_json(path) if mutation == "carry_count" else {}
        if mutation == "carry_count":
            value["committed_carried"] -= 1
    pair.base.atomic_json(path, value)
    with pytest.raises(ValueError):
        invoke(entry, migrated)
