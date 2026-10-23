"""CPU contracts for frozen baseline, anatomical mirror and paired readout."""
import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import hb145_r7_heldout_v1 as old
from tools import hb145_r7_pipeline_v111 as oldpipe
from tools import hb149_r7_flip_v120 as h
from tools import hb149_r7_pipeline_v120 as pipe
from tools import hb149_r7_pairing as pair


def node(source, name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_frozen_sources_and_new_version():
    pins = pair.read_json(pair.SCRATCH / "baseline-source-pins-v122.json")
    assert all(old.r7.s6.sha256(old.ROOT / path) == digest for path, digest in pins.items())
    assert old.VERSION == "1.1.3" and h.VERSION == "1.2.0"
    assert old.SCRATCH != h.SCRATCH
    assert pipe.monitored_run is oldpipe.monitored_run


def test_seeded_cohort_and_pinning():
    identity = dict(version="1.1.2", models="heldout,f0", device="cuda", **old.EXPOSURE,
                    uids=[str(i) for i in range(2000)])
    selected = pair.cohort(identity)
    assert len(selected) == len(set(selected)) == 1000
    assert all(old.fold_k(u) in (1, 2, 3) for u in selected)
    assert selected == pair.cohort(dict(identity, uids=list(reversed(identity["uids"]))))
    with pytest.raises(ValueError):
        pair.cohort(dict(identity, version="1.1.1"))
    with pytest.raises(ValueError):
        pair.cohort(dict(identity, uids=selected[:20]))


def test_runtime_keeps_serving_math(tmp_path):
    runtime = h.runtime_source(tmp_path, dual=True)
    code = (runtime / "compact_infer.py").read_text(encoding="utf-8")
    original = (h.r7.SOURCE / "compact_infer.py").read_text(encoding="utf-8")
    for name in ("mirror_windows", "mirror_perm", "score_v4c", "score", "_native_and_geometry",
                 "_prepare", "_forward", "apply_stacker", "apply_state_stacker"):
        assert ast.dump(node(code, name)) == ast.dump(node(original, name)), name
    assert "hb149_r7_flip_v120 as HB" in code and "hb149_r7_pipeline_v120 as PIPE" in code
    assert "dev = self.device" in code and "sdev = self.device" in code
    frozen = old.ROOT / ".local/backups/astra-b5-H-B-145/v1.1.2/hb145_r7_pipeline_v111.py"
    original_pins = pair.read_json(pair.SCRATCH / "frozen-source-pins.json")
    assert old.r7.s6.sha256(frozen) == original_pins["tools/hb145_r7_pipeline_v111.py"]
    for name in ("install_dataset", "worker_init", "cpu_worker_init", "capture", "commit"):
        assert ast.dump(node(Path(pipe.__file__).read_text(), name)) == ast.dump(node(frozen.read_text(), name))


def mirror_namespace():
    import torch
    source = (h.r7.SOURCE / "compact_infer.py").read_text(encoding="utf-8")
    ns = {"torch": torch, "MIRROR_PAIRS": (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))}
    for name in ("mirror_perm", "mirror_windows", "score_v4c"):
        exec(compile(ast.Module(body=[node(source, name)], type_ignores=[]), "actual-r7-source", "exec"), ns)
    return torch, ns


def test_actual_plane_mirror_and_inverse():
    torch, ns = mirror_namespace()
    image = torch.arange(1 * 3 * 3 * 4 * 5).reshape(1, 3, 3, 4, 5)
    meta = torch.zeros((1, 3, 9))
    meta[0, :, 7] = torch.tensor([0, 1, 2])
    meta[0, :, 1] = torch.tensor([0, 2, 3])
    meta[0, :, 2] = .75
    out, mout = ns["mirror_windows"](image, meta)
    assert torch.equal(out[:, 0], image[:, 0].flip(1))
    assert torch.equal(out[:, 1:], image[:, 1:].flip(-1))
    assert mout[0, :, 2].tolist() == [-.75, .75, .75]
    assert mout[0, :, 1].tolist() == [0, 3, 2]
    back, bmeta = ns["mirror_windows"](out, mout)
    assert torch.equal(back, image) and torch.equal(bmeta, meta)
    assert meta[0, 0, 2] == .75


def test_actual_logit_permutation_mcl_and_state_heads():
    torch, ns = mirror_namespace()
    labels = h.LABELS
    spec = {"categories": {label: [0, 1] for label in labels}}
    bins = [("b", label) for label in labels]
    perm, binperm = ns["mirror_perm"](labels, spec, bins)
    assert [perm[p] for p in perm] == list(range(12))
    z = torch.arange(12).reshape(1, 12).float()
    zm = z * 2 + 10
    sl = torch.arange(24).reshape(1, 12, 2).float()
    bl = z + 100
    calls = iter([(z, sl, bl), (zm, sl + 50, bl + 50)])
    def state_readout(model, states, binaries):
        return torch.cat((states.flatten(1), binaries), 1)
    mcl = labels.index("MCL")
    mclcols = [2*mcl, 2*mcl+1, 24+mcl]
    ns.update(FLIP={"perm": perm, "bin_perm": binperm, "mcl": mcl},
              one_pass=lambda *args: next(calls), state_readout=state_readout,
              MCL_STATE_COLS=mclcols, EXTRA={})
    images, meta, mask = torch.zeros(1, 1, 3, 4, 4), torch.zeros(1, 1, 9), torch.ones(1, 1)
    result = ns["score_v4c"]([object()], images, meta, mask)[0]
    expected = .5 * (z[0] + zm[0, perm])
    expected[mcl] = z[0, mcl]
    assert torch.equal(result, expected)
    extra = ns["EXTRA"]["group1"]
    plain = state_readout(None, sl, bl)[0]
    mirrored = state_readout(None, (sl + 50)[:, perm], (bl + 50)[:, binperm])[0]
    state_expected = .5 * (plain + mirrored)
    state_expected[mclcols] = plain[mclcols]
    assert torch.equal(extra["state"][0], state_expected)
    spec["categories"]["Medial OA"] = [0, 1, 2]
    with pytest.raises(SystemExit):
        ns["mirror_perm"](labels, spec, bins)


@pytest.mark.parametrize("fold", [1, 2, 3])
def test_single_fold_journal_export_resume(tmp_path, monkeypatch, fold):
    uid = next(str(i) for i in range(100) if h.fold_k(str(i)) == fold)
    cfg = {"out": str(tmp_path), "identity": "test", "models": "heldout_flip", "flags": {}, "device": "cpu"}
    monkeypatch.setattr(h, "_HOOK", cfg)
    h.atomic_json(tmp_path / "hook.json", cfg)
    item = {"uid": uid, "hb145_active_fold": fold}
    pipe.capture(uid, item, [.3] * 12, [.4] * 12)
    pipe.commit(item)
    records = h.export_csv(tmp_path, [uid], "test")
    assert set(records[uid]["predictions"]) == {str(fold)}
    assert records[uid]["role_folds"] == {"heldout_flip": fold}
    assert records[uid]["clean_oof"] is False and records[uid]["flip_tta"] is True
    rows = pair.predictions(tmp_path / "heldout_flip_raw.csv", [uid], "heldout_flip")
    assert rows[uid] == [.3] * 12
    assert not (tmp_path / "f0_raw.csv").exists()
    records[uid]["predictions"]["0"] = dict(raw=[.3]*12, post=[.4]*12)
    with pytest.raises(ValueError):
        pipe.validate_record(records[uid])


def test_fold_zero_rejected():
    uid = next(str(i) for i in range(100) if h.fold_k(str(i)) == 0)
    with pytest.raises(ValueError):
        pipe.required_folds(uid)


def test_stats_identical_and_perfect_improvement():
    y = np.tile(np.arange(80)[:, None] % 2, (1, 12)).astype(float)
    p = 1 - y
    result = pair.metrics(y, p, y, draws=100)
    assert result["macro"]["delta_auc"]["value"] == 1
    assert result["macro"]["delta_auc"]["ci95"] == [1, 1]
    assert result["weak_four"]["spearman_mean"]["value"] == pytest.approx(-1)
    assert result["screen_gate"] == "ADVANCE_TO_BLEND_AND_T4_MEASUREMENT"
    identical = pair.metrics(y, y, y, draws=100)
    assert identical["macro"]["delta_auc"]["ci95"] == [0, 0]
    assert identical["screen_gate"] == "HOLD"


def test_stats_missing_ties_soft_labels_and_undefined_targets():
    y = np.tile([.1, .8, .2, .9] * 8, (12, 1)).T
    y[0, 0] = np.nan
    p = np.full(y.shape, .5)
    result = pair.metrics(y, p, p, draws=100)
    assert result["per_target"][0]["labelled"] == len(y) - 1
    assert result["macro"]["no_flip_auc"]["value"] == .5
    assert result["macro"]["spearman_mean"]["value"] is None
    y[:, 1] = 0
    result = pair.metrics(y, p, p, draws=100)
    assert result["macro"]["delta_auc"]["value"] is None
    assert result["macro"]["delta_auc"]["valid_bootstraps"] == 0
    assert result["weak_four"]["delta_auc"]["value"] == 0
    y[0, 0] = -1
    with pytest.raises(ValueError):
        pair.metrics(y, p, p, draws=100)


def test_runtime_bound():
    result = pair.runtime_estimate(1100)
    assert result["total_hours_unscaled"] == pytest.approx(8.2423092597)
    assert result["headroom_hours_unscaled"] > 0
    assert pair.runtime_estimate(1400)["headroom_hours_unscaled"] < 0
    full = pair.runtime_estimate()
    half = pair.runtime_estimate(variant="r7_flip_f0_f2")
    assert full["total_hours_unscaled"] == pytest.approx(8.8863654887)
    assert full["within_go_budget"] is False
    assert half["total_hours_unscaled"] == pytest.approx(6.7931827444)
    assert half["within_go_budget"] is True
    assert half["total_hours_calibrated_range"] == pytest.approx([6.2698870583, 6.4582735053])
    assert half["flip_folds"] == [0, 2]
    assert half["release_base"] == pair.RELEASE_BASE
    with pytest.raises(ValueError):
        pair.runtime_estimate(0)


def test_catastrophic_target_blocks_go_despite_positive_macro_ci():
    y = np.tile(np.arange(80)[:, None] % 2, (1, 12)).astype(float)
    plain, flip = 1 - y, y.copy()
    plain[:, 0], flip[:, 0] = y[:, 0], 1 - y[:, 0]
    result = pair.metrics(y, plain, flip, draws=100)
    assert result["macro"]["delta_auc"]["value"] > .001
    assert result["macro"]["delta_auc"]["ci95"][0] > 0
    assert result["target_floor"]["worst_delta_auc"] == -1
    assert result["target_floor"]["passed"] is False
    assert result["screen_gate"] == "HOLD"
    variants = pair.variant_readout(y, plain, flip, np.full(len(y), 2), draws=100)
    assert variants["go_variants"] == []


def test_registered_half_policy_and_runtime_gate():
    y = np.tile(np.arange(80)[:, None] % 2, (1, 12)).astype(float)
    plain, flip = 1 - y, y.copy()
    # Independent of label alternation; each fold has both classes.
    folds = np.repeat(np.arange(4), 20)
    result = pair.variant_readout(y, plain, flip, folds, draws=100)
    half = result["variants"]["r7_flip_f0_f2"]
    assert half["studies_using_flip"] == 40
    expected = np.where(np.isin(folds, [0, 2])[:, None], flip, plain)
    assert half["macro"] == pair.metrics(y, plain, expected, draws=100)["macro"]
    assert result["variants"]["full_r7_flip"]["screen_gate"] != "HOLD"
    assert result["variants"]["full_r7_flip"]["go_gate"] == "HOLD"
    assert result["go_variants"] == ["r7_flip_f0_f2"]
    assert result["release_authorized"] is False
    assert pair.variant_readout(y, plain, flip, folds, draws=100, studies=2000)["go_variants"] == []
    with pytest.raises(ValueError):
        pair.variant_readout(y, plain, flip, [-1] * 80, draws=100)


def test_production_limit_cpu_smoke_and_output_guards():
    with pytest.raises(SystemExit):
        h.main(["prepare", "--limit", "1"])
    with pytest.raises(SystemExit):
        h.main(["prepare", "--cpu-smoke", "--limit", "1", "--device", "cuda"])
    with pytest.raises(ValueError):
        pair.owned_output(old.SCRATCH / "bad.json")


@pytest.fixture
def paired_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(pair, "SIZE", 48)
    monkeypatch.setattr(pair, "SCRATCH", tmp_path)
    baseline, flip = tmp_path / "baseline", tmp_path / "flip"
    identity = {"version": "1.1.2", "models": "heldout,f0", "device": "cuda", **old.EXPOSURE,
                "uids": [str(i) for i in range(100)]}
    old.atomic_json(baseline / "identity.json", identity)
    selection = tmp_path / "selection.json"
    pair.select(baseline, selection)
    uids = pair.read_json(selection)["uids"]
    ident = old.fingerprint(identity)
    old.atomic_json(baseline / "run_receipt.json", dict(identity=ident, completed=100, selected=100,
                                                     failures=0, decode_failures=0, exit_code=0))
    cfg = dict(out=str(baseline), identity=ident, flags={}, models="heldout,f0", device="cuda")
    monkeypatch.setattr(old, "_HOOK", cfg)
    old.atomic_json(baseline / "hook.json", cfg)
    fid = {"version": h.VERSION, "models": "heldout_flip", "uids": uids, "flip_tta": True,
           "cpu_smoke": False, "device": "cuda", **old.EXPOSURE,
           "baseline_identity": ident, "selection_sha256": old.r7.s6.sha256(selection)}
    old.atomic_json(flip / "identity.json", fid)
    fc = dict(cfg, out=str(flip), models="heldout_flip", identity=old.fingerprint(fid))
    monkeypatch.setattr(h, "_HOOK", fc)
    old.atomic_json(flip / "hook.json", fc)
    for i, uid in enumerate(uids):
        value = [.1 + .8 * (i % 2)] * 12
        item = {"uid": uid}
        for k in oldpipe.required_folds(uid):
            item["hb145_active_fold"] = k
            oldpipe.capture(uid, item, value, value)
        oldpipe.commit(item)
        item = {"uid": uid, "hb145_active_fold": h.fold_k(uid)}
        pipe.capture(uid, item, value, value)
        pipe.commit(item)
    old.export_csv(baseline, uids, ident)
    h.export_csv(flip, uids, fc["identity"])
    old.atomic_json(flip / "run_receipt.json", dict(identity=fc["identity"], completed=48, selected=48,
                                                 failures=0, decode_failures=0, exit_code=0))
    labels = tmp_path / "numeric-label-fixture.csv"
    with labels.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([old.UID, *old.LABELS])
        writer.writerows([uid, *([i % 2] * 12)] for i, uid in enumerate(uids))
    return selection, baseline, flip, labels, tmp_path / "readout.json"


@pytest.fixture
def ready_artifacts(paired_artifacts, monkeypatch):
    from hb145_v113_fixture import migrate_pair
    return migrate_pair(paired_artifacts, monkeypatch)


def test_paired_readout_end_to_end(ready_artifacts):
    paired_artifacts = ready_artifacts
    result = pair.readout(*paired_artifacts, draws=100)
    assert result["macro"]["no_flip_auc"]["value"] == 1
    assert result["macro"]["delta_auc"]["ci95"] == [0, 0]
    assert result["clean_oof"] is False
    assert "uids" not in result
    with paired_artifacts[-1].with_suffix(".csv").open() as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 24
        assert {row["variant"] for row in rows} == set(pair.VARIANTS)
    assert result["go_variants"] == []
    assert result["variants"]["r7_flip_f0_f2"]["studies_using_flip"] == sum(
        old.fold_k(u) == 2 for u in pair.read_json(paired_artifacts[0])["uids"])


def test_completion_and_identity_gate(ready_artifacts):
    paired_artifacts = ready_artifacts
    selection, baseline, flip, labels, output = paired_artifacts
    receipt = pair.read_json(baseline / "run_receipt.json")
    receipt["completed"] -= 1
    old.atomic_json(baseline / "run_receipt.json", receipt)
    with pytest.raises(ValueError, match="full must complete"):
        pair.baseline_ready(selection, baseline)
    record = pair.read_json(selection)
    record["uids"] = record["uids"][::-1]
    old.atomic_json(selection, record)
    with pytest.raises(ValueError, match="selection/baseline"):
        pair.verify_selection(selection, baseline)


def test_csv_tampering_refused(ready_artifacts):
    paired_artifacts = ready_artifacts
    selection, baseline, flip, labels, output = paired_artifacts
    path = flip / "heldout_flip_raw.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    rows[0][h.LABELS[0]] = '.123'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="committed journal"):
        pair.readout(*paired_artifacts, draws=100)


def test_cpu_smoke_cannot_be_scored_as_production(ready_artifacts):
    paired_artifacts = ready_artifacts
    selection, baseline, flip, labels, output = paired_artifacts
    identity = pair.read_json(flip / "identity.json")
    identity["cpu_smoke"] = True
    old.atomic_json(flip / "identity.json", identity)
    with pytest.raises(ValueError, match="completion/identity"):
        pair.readout(*paired_artifacts, draws=100)


def test_bad_numeric_label_never_echoed(ready_artifacts):
    paired_artifacts = ready_artifacts
    selection, baseline, flip, labels, output = paired_artifacts
    content = labels.read_text()
    labels.write_text(content.replace(',0,', ',DO_NOT_ECHO,', 1))
    with pytest.raises(ValueError, match="malformed numeric label") as exc:
        pair.readout(*paired_artifacts, draws=100)
    assert "DO_NOT_ECHO" not in str(exc.value)
