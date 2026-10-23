"""v1.2.2: unchanged cohort with authenticated v113 identity transition."""
import pytest
from test_hb149_r7_flip_v120 import paired_artifacts
from hb145_v113_fixture import migrate_pair
from tools import hb149_r7_pairing as pair


@pytest.fixture
def migrated(paired_artifacts, monkeypatch):
    return migrate_pair(paired_artifacts, monkeypatch)


def test_migrated_validate_and_readout(migrated):
    selection, baseline, _, _, _ = migrated
    assert pair.main(["validate", "--selection", str(selection), "--baseline", str(baseline)]) == 0
    result = pair.readout(*migrated, draws=100)
    assert result["baseline_identity"] == pair.contract.IDENTITY
    assert result["selection_baseline_identity"] == pair.contract.SOURCE_IDENTITY
    assert result["carry_over"]["committed_carried"] == 16
    assert result["macro"]["delta_auc"]["ci95"] == [0, 0]


@pytest.mark.parametrize("mutation", ["source", "identity", "missing", "duplicate", "path", "hash", "flag", "intent", "journal", "lineage", "selection"])
def test_migration_tampering_refused(migrated, mutation):
    selection, baseline, _, _, _ = migrated
    path = baseline / "carry_over_manifest.json"
    value = pair.read_json(path)
    if mutation == "source": value["source_identity"] = "0" * 64
    elif mutation == "identity": value["identity"] = "0" * 64
    elif mutation == "missing": value["entries"].pop()
    elif mutation == "duplicate": value["entries"][1] = value["entries"][0]
    elif mutation == "path": value["entries"][0]["journal"] = "../identity.json"
    elif mutation == "hash": value["entries"][0]["destination_sha256"] = "0" * 64
    elif mutation == "flag": value["prediction_and_all_original_metadata_equal"] = False
    elif mutation == "intent":
        pair.base.atomic_json(baseline / "carry_over_intent.json", {})
    elif mutation in ("journal", "lineage"):
        entry = value["entries"][0]
        jp = baseline / "journal" / entry["journal"]
        record = pair.read_json(jp)
        record["carry_over"]["source_identity"] = "bad"
        pair.base.atomic_json(jp, record)
        if mutation == "lineage": entry["destination_sha256"] = pair.base.r7.s6.sha256(jp)
    else:
        record = pair.read_json(selection)
        record["baseline_identity"] = "unrelated"
        pair.base.atomic_json(selection, record)
    pair.base.atomic_json(path, value)
    with pytest.raises(ValueError):
        pair.baseline_ready(selection, baseline)


def test_v113_missing_lineage_cannot_use_same_identity(paired_artifacts, monkeypatch):
    selection, baseline, _, _, _ = paired_artifacts
    identity = pair.read_json(baseline / "identity.json")
    identity["version"] = "1.1.3"
    pair.base.atomic_json(baseline / "identity.json", identity)
    pair.base.atomic_json(selection, dict(pair.read_json(selection), baseline_identity=pair.base.fingerprint(identity)))
    with pytest.raises(ValueError, match="unregistered"):
        pair.baseline_ready(selection, baseline)
