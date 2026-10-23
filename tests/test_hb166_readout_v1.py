"""CPU tests v1.0.0 for the H-B-166 registered gate, L6 bundle and seven-lens readout (aggregates only)."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # empty, never -1 (Windows cu121 backward crash)

import numpy as np
import pytest

from tools import hb166_build_v1 as b
from tools import hb166_readout_v1 as r

REGISTERED = (b.WORK / "registration.json").exists()


def tensors(macro=.004, spread=.001, target_shift=None, seed_offsets=(0., 0.), draws=300):
    rng = np.random.default_rng(0)
    mean = np.full((draws + 1, 4, 12), macro) + np.r_[0, rng.normal(0, spread, draws)][:, None, None]
    if target_shift is not None:
        mean[:, :, target_shift[0]] += target_shift[1]
    seeds = {s: mean + off for s, off in zip(r.SEEDS, seed_offsets)}
    return seeds, mean


def test_gate_passes_registered_signal():
    seeds, mean = tensors()
    g = r.registered_gate(seeds, mean)
    assert g["PASS"] and g["status"] == "PASS_FOLD3_CONFIRMATION_REQUIRED" and g["ci_positive_rulers"] == 4
    assert g["not_release_authorization"] is True and "fold-3" in g["consequence"]


def test_gate_macro_threshold():
    seeds, mean = tensors(macro=.0019, spread=.0001)
    g = r.registered_gate(seeds, mean)
    assert not g["PASS"] and not g["checks"]["four_ruler_macro"] and g["checks"]["ci_lower_positive_rulers"]


def test_gate_requires_two_rulers_with_positive_lower_bound():
    seeds, mean = tensors(macro=.003, spread=.0001)
    mean[1:, 1:, :] = mean[1:, 1:, :] - .01   # three rulers' draws cross zero; points unchanged
    seeds = {s: mean.copy() for s in r.SEEDS}
    g = r.registered_gate(seeds, mean)
    assert g["ci_positive_rulers"] == 1 and not g["PASS"] and g["checks"]["four_ruler_macro"]
    mean[1:, 1, :] += .01
    g = r.registered_gate({s: mean.copy() for s in r.SEEDS}, mean)
    assert g["ci_positive_rulers"] == 2 and g["PASS"]


def test_gate_target_floor_blocks():
    seeds, mean = tensors(macro=.004, target_shift=(6, -.0101))   # PF OA 4-ruler mean = -0.0061
    g = r.registered_gate(seeds, mean)
    assert not g["PASS"] and not g["checks"]["target_floor"] and g["worst_target"] == "PF OA"
    seeds, mean = tensors(macro=.004, target_shift=(6, -.0089))   # -0.0049 stays above the floor
    assert r.registered_gate(seeds, mean)["checks"]["target_floor"]


def test_gate_both_seeds_positive():
    seeds, mean = tensors(macro=.004, seed_offsets=(.002, -.0045))
    g = r.registered_gate(seeds, mean)
    assert not g["PASS"] and not g["checks"]["both_seeds_positive"] and g["seed_four_ruler_macro"]["43"] < 0


@pytest.mark.parametrize("mutate", ["nan", "shape", "seed"])
def test_gate_invalid_evidence(mutate):
    seeds, mean = tensors()
    if mutate == "nan":
        mean[0, 0, 0] = np.nan
    elif mutate == "shape":
        mean = mean[:, :3]
    else:
        seeds = {"42": seeds["42"]}
    assert r.registered_gate(seeds, mean)["status"] == "INVALID_EVIDENCE"


def test_engineering_never_passes():
    seeds, mean = tensors()
    g = r.registered_gate(seeds, mean, engineering=True)
    assert g["status"] == "ENGINEERING_ONLY" and not g["PASS"]


def test_prediction_pin_and_roster_enforced(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("StudyInstanceUID," + ",".join(r.lens.TARGETS) + "\na," + ",".join(["0.5"] * 12) + "\n")
    assert r.load_csv(path, ["a"], b.sha(path)).shape == (1, 12)
    with pytest.raises(ValueError, match="pin"):
        r.load_csv(path, ["a"], "0" * 64)
    with pytest.raises(ValueError, match="roster"):
        r.load_csv(path, ["b"])


def test_partner_pins_and_bundle_on_same_rows():
    val = b.read(b.b155.config_path("control", 42))["val"]
    partners = r.partner_predictions(val)
    assert sorted(partners) == ["C4", "R50", "T"] and all(p.shape == (864, 12) for p in partners.values())
    rng = np.random.default_rng(1)
    a, c = rng.uniform(size=(864, 12)), rng.uniform(size=(864, 12))
    bundle = r.l6_bundle(object(), a, c, partners)
    assert bundle["arm"].shape == (864, 12) and not np.allclose(bundle["arm"], bundle["control"])
    same = r.l6_bundle(object(), a, a, partners)
    assert np.array_equal(same["arm"], same["control"])


def test_lens_table_reports_all_seven_with_reasons():
    lenses = {k: {"value": .001, "ci95": [0, .002], "PASS": False, "eligible": True} for k in ("L1", "L2", "L3", "L5", "L6")}
    lenses["L4"] = {"value": .004, "PASS": True, "eligible": True, "selected_targets": ["Lateral OA"]}
    lenses["L7"] = {"value": {"gold58": .01}, "PASS": False, "eligible": False}
    table = r.lens_table({"lenses": lenses})
    assert list(table) == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    assert table["L7"]["eligible"] is False and "OAI" in table["L7"]["reason"] and "Gold58" in table["L7"]["reason"]
    lenses["L6"] = {"value": None, "PASS": False, "eligible": False, "reason": "partner missing"}
    assert r.lens_table({"lenses": lenses})["L6"]["reason"]


@pytest.mark.skipif(not REGISTERED, reason="stage not registered yet")
def test_engineering_readout_end_to_end(tmp_path):
    out = tmp_path / "readout-engineering.json"
    result = r.run(None, engineering=True, out=out)
    assert result["status"] == "ENGINEERING_ONLY"
    data = b.read(out)
    assert set(data["lens_table"]) == {"L1", "L2", "L3", "L4", "L5", "L6", "L7"}
    assert data["lenses"]["L6"]["eligible"] is True and data["lenses"]["L6"]["rows"] == 864
    assert data["lenses"]["L7"]["eligible"] is False and "OAI" in data["lenses"]["L7"]["reason"]
    assert data["lenses"]["L5"]["eligible"] is True and data["exposure"] == "EXPOSED" and data["clean_oof"] is False
    assert data["registered_gate"]["PASS"] is False and data["engineering_only"] is True
