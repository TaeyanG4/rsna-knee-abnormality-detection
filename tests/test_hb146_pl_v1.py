"""H-B-146 contract tests; synthetic label values only, no study rows printed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tools import hb146_pl_labels_v1 as pl
from tools import hb146_readout_v1 as readout
from training.arms.hb146_pl_v1 import Arm, SoftSource
from training.arms import mil_screen_hooks_v1 as h

torch.set_num_threads(2)


def values(v):
    return np.full(12, v, np.float32)


def test_fixed_soft_blend_joins_by_uid():
    l2 = {"a": values(.2), "b": values(.8)}
    teacher = {"b": values(.3), "a": values(.4)}
    got = pl.blend(l2, teacher, ["b", "a"])
    np.testing.assert_allclose(got[:, 0], [.55, .3])
    assert not np.isin(got, [0, 1]).any()
    assert pl.frozen_weights(l2, ["a"])[0] == 4
    assert pl.frozen_weights(dict(zip(["b", "a"], got)), ["a"])[0] != 4


@pytest.mark.parametrize("ids,teacher", [(["a", "a"], {"a": values(.5)}),
                                         (["a", "b"], {"a": values(.5)})])
def test_bad_joins_rejected(ids, teacher):
    with pytest.raises(ValueError):
        pl.blend({"a": values(.1), "b": values(.2)}, teacher, ids)


def test_duplicate_and_invalid_csv_rejected(tmp_path):
    p = tmp_path / "labels.csv"
    frame = pd.DataFrame([values(.2), values(.3)], columns=pl.TARGETS)
    frame.insert(0, pl.UID, ["x", "x"])
    frame.to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        pl.table(p)
    frame[pl.UID] = ["x", "y"]
    frame.loc[0, "ACL"] = np.nan
    frame.to_csv(p, index=False)
    with pytest.raises(ValueError, match="invalid"):
        pl.table(p)
    frame.loc[0, "ACL"] = 1.1
    frame.to_csv(p, index=False)
    with pytest.raises(ValueError, match="invalid"):
        pl.table(p)


def test_split_excludes_whole_cross_group_and_gold():
    by_fold = {k: [str(i) for i in range(100) if pl.hash_fold(str(i)) == k] for k in range(4)}
    cross = [by_fold[0][0], by_fold[1][0], by_fold[2][0]]
    gold = by_fold[0][1]
    ids = cross + [gold, by_fold[0][2], by_fold[1][2], by_fold[1][3]]
    rows = pd.DataFrame({pl.UID: ids, "component_id": ["cross"] * 3 + ["gold", "val", "same", "same"],
                         "fold": [4, 4, 4, 1, 2, 3, 3]})
    s = pl.split(rows, [gold])
    assert set(s["excluded_cross_hash"]) == set(cross)
    assert s["val"] == [by_fold[0][2]]
    assert set(s["train"]) == {by_fold[1][2], by_fold[1][3]}
    assert gold not in s["train"] + s["val"]
    assert not {s["components"][u] for u in s["train"]} & {s["components"][u] for u in s["val"]}


def test_hash_formula_exact():
    import hashlib
    for uid in ("1.2.3.4", "abc", "99"):
        assert pl.hash_fold(uid) == int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4


def test_soft_bce_shared_hook_matches_loss_and_gradient():
    # A real optimizer update proves the shared hook does not threshold y.
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.linspace(-.5, .5, 12))

        def forward(self, x):
            return self.logits.unsqueeze(0).expand(len(x), -1)

    m = Model()
    before = m.logits.detach().clone()
    y = torch.linspace(.05, .95, 12).unsqueeze(0)
    w = torch.linspace(1, 4, 12)
    expected_loss = -(w * y * torch.nn.functional.logsigmoid(before) +
                      (1 - y) * torch.nn.functional.logsigmoid(-before)).mean()
    grad = ((1 - y) * before.sigmoid() - w * y * (1 - before.sigmoid())).squeeze(0) / 12
    opt = torch.optim.SGD(m.parameters(), lr=.2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    counts = {"updates": 0, "skips": 0}
    result = h.train_epoch_with(Arm(), m, [(torch.zeros(1, 1), y, ["uid"])], opt, sched, scaler,
                               w, "cpu", "none", 0, counts)
    assert result["loss"] == pytest.approx(expected_loss.item(), abs=1e-7)
    torch.testing.assert_close(m.logits, before - .2 * grad)
    assert counts == {"updates": 1, "skips": 0}


def test_source_hook_only_changes_training_targets():
    class Dataset:
        k, ids = 12, ["a"]
        epoch = -1

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.ones(12, 3, 2, 2), torch.zeros(12), "a"

        def set_epoch(self, epoch):
            self.epoch = epoch

    base = Dataset()
    arm = Arm({"a": values(.37)})
    assert arm.dataset(base, train=False) is base
    assert Arm().dataset(base, train=True) is base
    ds = arm.dataset(base, train=True)
    assert ds.k == 12 and ds.ids == ["a"]
    ds.set_epoch(3)
    assert base.epoch == 3
    torch.testing.assert_close(ds[0][0], base[0][0])
    np.testing.assert_allclose(ds[0][1].numpy(), .37)


def test_teacher_mode_and_fold_provenance(tmp_path):
    ids = [str(i) for i in range(20)]
    csv = tmp_path / "raw.csv"
    frame = pd.DataFrame(np.full((len(ids), 12), .3), columns=pl.TARGETS)
    frame.insert(0, pl.UID, ids)
    frame["fold_k"] = 0
    frame["status"] = "ok"
    frame["output_kind"] = "pre_stacker"
    frame.to_csv(csv, index=False)
    receipt = tmp_path / "teacher.json"
    r = dict(csv_sha256=pl.sha(csv), mode="screen_f0", output_kind="pre_stacker", pins={}, complete=True,
             failures=0, source_pool_membership="manifest_contract_unverified_membership", model_selection="f0_only",
             validation_label_exposure="pooled_prior_not_clean", auxiliary_label_exposure="unknown")
    pl.write(receipt, r)
    assert len(pl.validate_teacher(csv, receipt, ids, "screen_f0")[0]) == len(ids)
    with pytest.raises(ValueError):
        pl.validate_teacher(csv, receipt, ids, "production_heldout")
    frame.loc[0, "fold_k"] = 1
    frame.to_csv(csv, index=False)
    r["csv_sha256"] = pl.sha(csv)
    pl.write(receipt, r)
    with pytest.raises(ValueError, match="wrong teacher fold"):
        pl.validate_teacher(csv, receipt, ids, "screen_f0")
    frame["fold_k"] = [pl.hash_fold(u) for u in ids]
    frame.to_csv(csv, index=False)
    r.update(csv_sha256=pl.sha(csv), mode="production_heldout", model_selection="sha1_first8_mod4")
    pl.write(receipt, r)
    assert len(pl.validate_teacher(csv, receipt, ids, "production_heldout")[0]) == len(ids)


def test_hash_tampering_fails(tmp_path):
    p = tmp_path / "input"
    p.write_text("old")
    pins = pl.pins([p])
    p.write_text("new")
    with pytest.raises(ValueError, match="hash changed"):
        pl.check_pins(pins)


def test_go_gate_boundary_missing_ancestry_and_engineering():
    assert readout.gate([-.003] * 4, 0., 0., True)["GO"]
    assert not readout.gate([-.00301, 0, 0, 0], .1, .1, True)["GO"]
    assert not readout.gate([0] * 4, -.0001, .1, True)["GO"]
    assert not readout.gate([0] * 4, .1, -.0001, True)["GO"]
    assert readout.gate([0] * 4, .1, .1, False)["status"] == "HOLD"
    assert not readout.gate([np.nan] * 4, .1, .1, True)["GO"]
    assert not readout.gate([0] * 4, .1, .1, True, engineering=True)["GO"]


def test_f0_wrapper_changes_only_generated_worker_import(tmp_path, monkeypatch):
    from tools import hb146_teacher_v1 as wrapper
    path = tmp_path / "compact_infer.py"
    path.write_text("from tools import hb145_r7_heldout_v1 as HB\nx = 1\n", encoding="utf-8")
    monkeypatch.setattr(wrapper, "_original_runtime", lambda _: tmp_path)
    wrapper.runtime_source(tmp_path)
    code = path.read_text()
    assert "from tools import hb146_teacher_v1 as HB" in code
    assert "x = 1" in code
    assert pl.sha(wrapper.__file__) in code
    assert {wrapper.hb.fold_k(str(i)) for i in range(30)} == {0}


def test_extra_and_nonfinite_targets_fail():
    with pytest.raises(ValueError, match="unexpected UID"):
        pl.aligned({"a": values(.1), "b": values(.2)}, ["a"], exact=True)
    with pytest.raises(ValueError, match="invalid soft"):
        pl.blend({"a": values(.1)}, {"a": values(np.nan)}, ["a"])


def test_scientific_review_cannot_waive_pooled_prior(tmp_path, monkeypatch):
    from tools import hb146_screen_v1 as screen
    monkeypatch.setattr(screen, "prepared", lambda *a: {})
    p = {"teacher": {"validation_label_exposure": "pooled_prior_not_clean", "auxiliary_label_exposure": "unknown"}}
    monkeypatch.setattr(pl, "validate_labels", lambda *a: ({}, {}, p))
    review = tmp_path / "review.json"
    pl.write(review, {"approved": True, "teacher_exposure_contract_accepted": True})
    with pytest.raises(ValueError, match="HOLD"):
        screen.admit(tmp_path, review)
    assert screen.admit(tmp_path, None, engineering=True)["engineering_only"]
    for teacher in ({}, {"validation_label_exposure": "none", "auxiliary_label_exposure": "unknown"}):
        with pytest.raises(ValueError, match="HOLD"):
            pl.require_clean_teacher({"teacher": teacher})


@pytest.mark.parametrize("mode", ["screen_f0", "production_heldout"])
def test_seal_binds_journals_runtime_and_safe_receipt(tmp_path, monkeypatch, mode):
    from tools import hb146_teacher_v1 as wrapper
    out, owned = tmp_path / "source", tmp_path / "owned"
    out.mkdir()
    owned.mkdir()
    monkeypatch.setattr(pl, "WORK", owned)
    uid = "synthetic"
    runtime = out / "stage/r7_runtime/compact_infer.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("# HB146 wrapper sha256: " + pl.sha(wrapper.__file__) + "\n")
    identity = {"uids": [uid], "code": {"test": True}, "runtime_sha256": pl.sha(runtime),
                "runner_sha256": pl.sha(wrapper.hb.__file__)}
    pl.write(out / "identity.json", identity)
    pl.write(out / "run_receipt.json", {"completed": 1, "selected": 1, "failures": 0})
    folds = [0] if mode == "screen_f0" else range(4)
    for k in folds:
        pl.write(out / f"receipt_f{k}.json", {"models": [wrapper.hb.r7.MODELS[k]]})
    fold = 0 if mode == "screen_f0" else pl.hash_fold(uid)
    frame = pd.DataFrame([values(.3)], columns=pl.TARGETS)
    frame.insert(0, pl.UID, [uid])
    frame["fold_k"], frame["status"], frame["output_kind"] = fold, "ok", "pre_stacker"
    frame.to_csv(out / "raw.csv", index=False)
    record = {pl.UID: uid, "identity": wrapper.hb.fingerprint(identity), "fold_k": fold,
              "status": "ok", "pooled_prior_bias": True, "raw": values(.3).tolist()}
    journal = wrapper.hb.record_path(out, uid)
    pl.write(journal, record)
    dest = owned / "teacher.json"
    assert wrapper.seal(out, mode, dest)["rows"] == 1
    r = pl.read(dest)
    assert r["validation_label_exposure"] == "pooled_prior_not_clean"
    assert str(journal.resolve()) in r["pins"]
    if mode == "production_heldout":
        wrapper.seal(out, mode)
        assert (owned / "teacher-heldout.json").exists()
        assert not (out / "teacher.json").exists()
    with pytest.raises(ValueError, match="scratch"):
        wrapper.seal(out, mode, out / "teacher.json")
    record["raw"][0] = .7
    pl.write(journal, record)
    with pytest.raises(ValueError, match="probabilities differ"):
        wrapper.seal(out, mode, dest)
    record["raw"][0] = float(values(.3)[0])
    pl.write(journal, record)
    runtime.write_text("tamper")
    with pytest.raises(ValueError, match="runtime identity"):
        wrapper.seal(out, mode, dest)


def test_panel_rejects_stale_preparation_and_wrong_arm(tmp_path):
    from tools import hb146_screen_v1 as screen
    pl.write(tmp_path / "prepare-engineering.json", {"version": 1})
    pl.write(tmp_path / "registration.json", {"version": 1})
    i = {"arm": "pl", "seed": 42, "smoke": True, "engineering_only": True,
         "prepare_sha256": pl.sha(tmp_path / "prepare-engineering.json"),
         "registration_sha256": pl.sha(tmp_path / "registration.json")}
    screen.check_panel_identity(tmp_path, i, "pl", 42, True, True)
    with pytest.raises(ValueError, match="mismatched"):
        screen.check_panel_identity(tmp_path, i, "control", 42, True, True)
    pl.write(tmp_path / "prepare-engineering.json", {"version": 2})
    with pytest.raises(ValueError, match="stale"):
        screen.check_panel_identity(tmp_path, i, "pl", 42, True, True)
