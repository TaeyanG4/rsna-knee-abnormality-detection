"""H-B-130 per-target aquila-share label arm: label build, single-difference binding,
runner label identity, two-persistent-worker loader parity and readout math.
Aggregate-only; never prints label rows or report text. CPU only."""
import hashlib
import json

import numpy as np
import pytest
import torch

from training.arms import mil_screen_hooks_v1 as hooks
from training.arms import hb130_label_share_arm_v1 as armmod
from training.arms.hb130_label_share_arm_v1 import ARM
from tools import hb130_label_build_v1 as labels_build
from tools import hb130_screen_v1 as hb130
from tools import mil_screen_arm_build_v1 as build
from tools import hb122_screen_v1 as screen

torch.set_num_threads(2)
ROOT = hooks.ROOT
CORPUS = build.CORPUS
REFERENCE = "training.arms.hb130_label_share_arm_v1:ARM"
FIVE = [hooks.TARGETS.index(t) for t in armmod.WEAK_FIVE]
OTHER = [i for i in range(12) if i not in FIVE]


def documents(seed, recipe_dir):
    """In-memory replica of mil_screen_arm_build_v1.configure (recipe written under recipe_dir)."""
    control = hooks.control_spec(seed)
    doc = control.document
    declaration = doc["target_environment"]["evidence"]["declaration.json"]
    target = next(p for p, h in doc["inputs"]["evidence"].items() if h == declaration)
    recipe_control = next(p for p in doc["inputs"]["evidence"] if p != target)
    recipe = hooks.life.ArmSpec((ROOT / recipe_control).read_bytes()).document
    recipe["config"][ARM.section][ARM.key] = ARM.contract(seed)
    recipe["arm"] = f"{ARM.name}_RECIPE_S{seed}_F0"
    recipe = hooks.life.ArmSpec.from_document(recipe)
    recipe_path = recipe_dir / f"hb130-aquila-pertarget-recipe-s{seed}-f0-v1.json"
    if not recipe_path.exists():
        hooks.life.create_new(recipe_path, hooks.life.canonical_bytes(recipe.document))
    doc["config"][ARM.section][ARM.key] = ARM.contract(seed)
    doc["arm"] = f"{ARM.name}_S{seed}_F0"
    doc["inputs"]["sources"] = {p: hooks.sha(ROOT / p) for p in sorted(hooks.expected_sources(ARM, control))}
    doc["inputs"]["evidence"] = {build.rel(recipe_path): recipe.sha256, target: declaration}
    return hooks.life.ArmSpec.from_document(doc)


@pytest.fixture(scope="module")
def specs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hb130-configs")
    return {seed: documents(seed, folder) for seed in (42, 43)}


@pytest.fixture(scope="module")
def runner_labels(specs):
    out = {}
    for seed, spec in specs.items():
        _, train, val, arm_labels, arm_weights = hooks.validate_data_with(ARM, spec, ROOT, CORPUS)
        control = hooks.control_spec(seed)
        _, ctrain, cval, control_labels, control_weights = hooks.validate_data_with(hooks.IDENTITY, control, ROOT, CORPUS)
        assert (train, val) == (ctrain, cval)
        out[seed] = dict(train=train, arm=arm_labels, control=control_labels, arm_w=arm_weights, control_w=control_weights)
    return out


def test_label_file_reproduces_construction_and_pins_match_arm():
    result = labels_build.check()
    assert result["status"] == "CHECKED" and result["label_rows_printed"] is False
    assert result["csv_sha256"] == armmod.LABEL_SHA256 == hooks.sha(ROOT / armmod.LABEL_PATH)
    assert result["semantic_sha256"] == armmod.LABEL_SEMANTIC
    assert tuple(result["pos_weight"]) == armmod.POS_WEIGHT
    a = result["aggregates"]
    assert a["rows"] == a["control_rows"] == 4347 and a["header_equal"] and a["uid_order_equal"]
    assert a["gold58_in_labels"] == 0 and a["gold58_in_fold0_roster"] == 0
    for t in hooks.TARGETS:
        if t in armmod.WEAK_FIVE:
            assert a["aquila_share_percent"][t] == 60 and a["cells_changed"][t] > 0
        else:
            assert a["aquila_share_percent"][t] == 25 and a["cells_changed"][t] == 0
            assert a["mean_shift_all"][t] == 0.0 and a["max_abs_cell_change"][t] == 0.0
    prov = json.loads((ROOT / armmod.LABEL_PROVENANCE).read_text(encoding="utf-8"))
    assert prov["csv_sha256"] == armmod.LABEL_SHA256 and prov["gold58_included"] is False
    assert prov["label_rows_printed"] is False and prov["builder"] == armmod.BUILDER


def test_share_mixer_equals_exp184_for_registered_global_share():
    rng = np.random.default_rng(0)
    ids = [f"u{i}" for i in range(50)]
    d = {u: rng.random(12) for u in ids}
    a = {u: rng.random(12) for u in ids}
    assert labels_build.mix_rows_pertarget(ids, d, a, (25,) * 12) == labels_build.e184.mix_rows(ids, d, a, 25)
    rows = labels_build.mix_rows_pertarget(ids, d, a)
    base = labels_build.e184.mix_rows(ids, d, a, 25)
    for r, b in zip(rows, base):
        assert [r[j + 1] for j in OTHER] == [b[j + 1] for j in OTHER]
        expected = (0.4 * d[r[0]][FIVE] + 0.6 * a[r[0]][FIVE]).astype(np.float32)
        assert [r[j + 1] for j in FIVE] == [repr(float(x)) for x in expected]


@pytest.mark.parametrize("seed", [42, 43])
def test_single_difference_effective_labels_and_bindings(specs, seed):
    spec = specs[seed]
    doc = spec.document
    control = hooks.control_spec(seed).document
    raw = hooks.single_difference(ARM, spec)
    assert raw[ARM.section].pop(ARM.key) == ARM.contract(seed)
    assert raw == control["config"]
    effective = ARM.validate_recipe(spec)
    swap = ARM.contract(seed)
    assert effective["labels"] == {k: swap[k] for k in armmod.LABEL_FIELDS}
    effective.pop("labels")
    other = dict(control["config"])
    other.pop("labels")
    assert effective == other
    for key in ("family", "fold", "seed", "attempt", "target_environment", "outputs"):
        assert doc[key] == control[key]
    for key in ("data", "rosters", "controls"):
        assert doc["inputs"][key] == control["inputs"][key]
    assert doc["inputs"]["sources"][armmod.LABEL_PATH] == armmod.LABEL_SHA256
    assert set(doc["inputs"]["sources"]) == hooks.expected_sources(ARM, hooks.control_spec(seed))
    assert hooks.load_arm(REFERENCE, spec) is ARM
    assert ARM.provenance(spec)["label_set"] == armmod.LABEL_NAME


@pytest.mark.parametrize("change", ["second", "contract", "label_source", "provenance", "control_label", "weights"])
def test_drift_fails_closed(specs, change):
    doc = specs[42].document
    if change == "second": doc["config"]["updates"]["head_lr"] *= 2
    if change == "contract": doc["config"]["labels"][ARM.key]["aquila_share_percent"][7] = 25
    if change == "label_source": doc["inputs"]["sources"][armmod.LABEL_PATH] = "0" * 64
    if change == "provenance": doc["inputs"]["sources"].pop(armmod.LABEL_PROVENANCE)
    if change == "control_label": doc["config"]["labels"]["path"] = armmod.LABEL_PATH
    if change == "weights": doc["config"]["labels"][ARM.key]["pos_weight"][4] = 1.0
    with pytest.raises(hooks.life.ContractError):
        ARM.validate_recipe(hooks.life.ArmSpec.from_document(doc))


def test_configure_serialization_parity(monkeypatch):
    """Generic configure writes exactly the in-memory documents (captured, nothing written)."""
    captured, checked = {}, []
    real_sha = build.sha

    def capture(path, value):
        captured[path.resolve()] = hooks.life.canonical_bytes(value)

    def digest(path):
        return hashlib.sha256(captured[path.resolve()]).hexdigest() if path.resolve() in captured else real_sha(path)

    def recipe_only(arm, spec, root, corpus):
        assert arm is ARM and root == ROOT
        ARM.validate_recipe(spec)
        checked.append(spec.document["seed"])

    work = ROOT / ".local/scratch/claude-hb130/configure-parity"
    config_dir = work / "configs"
    monkeypatch.setattr(build, "same_or_new", capture)
    monkeypatch.setattr(build, "sha", digest)
    monkeypatch.setattr(hooks, "validate_data_with", recipe_only)
    result = build.configure(REFERENCE, work, config_dir)
    assert checked == [42, 43]
    for seed in (42, 43):
        path = config_dir / f"hb130-aquila-pertarget-s{seed}-f0-v1.json"
        spec = hooks.life.ArmSpec(captured[path.resolve()])
        assert ARM.validate_recipe(spec)["labels"]["name"] == armmod.LABEL_NAME
        assert result["specs"][str(seed)]["sha256"] == spec.sha256
    record = json.loads(captured[(work / "arm-build.json").resolve()])
    assert record["arm"] == REFERENCE and record["workers"] == 2
    assert not work.exists()


@pytest.mark.parametrize("seed", [42, 43])
def test_runner_label_identity_only_five_targets_change(runner_labels, seed):
    r = runner_labels[seed]
    assert np.array_equal(r["arm_w"], np.asarray(armmod.POS_WEIGHT, np.float32))
    assert np.array_equal(r["arm_w"][OTHER], r["control_w"][OTHER])
    assert set(r["arm"]) == set(r["control"])
    arm = np.stack([r["arm"][u] for u in r["train"]])
    control = np.stack([r["control"][u] for u in r["train"]])
    assert np.array_equal(arm[:, OTHER], control[:, OTHER])
    assert all((arm[:, t] != control[:, t]).any() for t in FIVE)


def _close(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


@pytest.mark.parametrize("seed", [42, 43])
def test_real_images_two_persistent_workers_identical_draws(runner_labels, seed):
    """Runner loader construction; only targets of the five columns may differ."""
    r = runner_labels[seed]
    ids = r["train"][:17]
    streams = {}
    for side in ("control", "arm"):
        chosen = {u: r[side][u] for u in ids}
        arm = ARM if side == "arm" else hooks.IDENTITY
        ds = hooks.MILWindows(CORPUS, ids, chosen, 12, True, seed)
        ds = arm.dataset(ds, None, ROOT, train=True)
        loader = hooks.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True, num_workers=2,
                                  persistent_workers=True, worker_init_fn=hooks.worker_init,
                                  generator=torch.Generator().manual_seed(seed), collate_fn=arm.collate_fn)
        batches = []
        try:
            for epoch in range(3):
                ds.set_epoch(epoch)
                for x, y, ordered, *extra in loader:
                    assert not extra
                    batches.append((epoch, hashlib.sha256(x.numpy().tobytes()).hexdigest(), list(ordered), y.numpy()))
        finally:
            _close(loader)
        streams[side] = batches
    assert len(streams["arm"]) == len(streams["control"]) == 6
    assert len({b[1] for b in streams["arm"]}) == 6
    for (ea, xa, ia, ya), (ec, xc, ic, yc) in zip(streams["arm"], streams["control"]):
        assert (ea, xa, ia) == (ec, xc, ic)
        assert np.array_equal(ya[:, OTHER], yc[:, OTHER])
        assert np.array_equal(ya, np.stack([r["arm"][u] for u in ia]))


def _context(n, seed=0, shared=False):
    rng = np.random.default_rng(seed)
    ids = [f"s{i:03d}" for i in range(n)]
    refs, y = {}, None
    for ruler in screen.ml.RULERS:
        if y is None or not shared:
            y = (rng.random((n, 12)) < .4).astype(float)
            y[0], y[1] = 0, 1
        refs[ruler] = (y, np.ones_like(y, bool), 1)
    components = [dict(component_id=f"c{i // 2}", members=[u for u in ids[i:i + 2]]) for i in range(0, n, 2)]
    ctx = screen.ml.Context(ids, refs, components)
    ctx.y = y
    return ctx, rng


def _oai(n, rng, arms=None, controls=None):
    ids = [f"k{i:03d}" for i in range(n)]
    y = (rng.random((n, 12)) < .4).astype(float)
    y[0], y[1] = 0, 1
    y[:, [1, 10, 11]] = np.nan
    arms = arms or {s: rng.random((n, 12)).astype(np.float32) for s in ("42", "43")}
    controls = controls or {s: rng.random((n, 12)).astype(np.float32) for s in ("42", "43")}
    return screen.image_item(ids, arms, controls, y), arms, controls


def test_evaluate_equals_hb122_screen_without_label_change_or_gold():
    ctx, rng = _context(60)
    arms = {s: rng.random((60, 12)).astype(np.float32) for s in ("42", "43")}
    controls = {s: rng.random((60, 12)).astype(np.float32) for s in ("42", "43")}
    bundle = dict(context=ctx, arm=arms["42"], control=controls["42"], recipe=dict(scope="test"))
    oai, _, _ = _oai(40, rng)
    expected = screen.evaluate(ctx, arms, controls, bundle, oai)
    assert hb130.evaluate(ctx, arms, controls, bundle, oai, None, label_change=False) == expected
    changed = hb130.evaluate(ctx, arms, controls, bundle, oai, None, label_change=True)
    assert changed["lenses"]["L3"]["rulers"]["aquila"]["self_referential"] is True
    assert expected["lenses"]["L3"]["rulers"]["aquila"]["self_referential"] is False
    gy = (rng.random((58, 12)) < .4).astype(float)
    gy[0], gy[1] = 0, 1
    gids = [f"g{i:02d}" for i in range(58)]
    gold = hb130.gold_item(gids, {s: rng.random((58, 12)).astype(np.float32) for s in ("42", "43")},
                           {s: rng.random((58, 12)).astype(np.float32) for s in ("42", "43")}, gy)
    full = hb130.evaluate(ctx, arms, controls, bundle, oai, gold, label_change=True)
    entry = full["lenses"]["L7"]["entries"]["gold58"]
    assert entry["eligible"] is False and len(entry["fixed_targets"]) == 12
    assert full["lenses"]["L7"]["entries"]["oai_moaks"] == changed["lenses"]["L7"]["entries"]["oai_moaks"]
    assert full["lenses"]["L7"]["PASS"] is False


def _rank_auc(p, y):
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def test_weak_five_rule_flags_and_values():
    ctx, rng = _context(60, 1, shared=True)
    same = {s: rng.random((60, 12)).astype(np.float32) for s in ("42", "43")}
    n = 40
    oids = [f"k{i:03d}" for i in range(n)]
    oy = (rng.random((n, 12)) < .4).astype(float)
    oy[0], oy[1] = 0, 1
    oy[:, [1, 10, 11]] = np.nan
    octl = {s: rng.random((n, 12)).astype(np.float32) for s in ("42", "43")}
    oarm = {s: octl[s].copy() for s in octl}
    gy = (rng.random((58, 12)) < .4).astype(float)
    gy[0], gy[1] = 0, 1
    gids = [f"g{i:02d}" for i in range(58)]
    gctl = {s: rng.random((58, 12)).astype(np.float32) for s in ("42", "43")}
    garm = {s: gctl[s].copy() for s in gctl}
    for s in ("42", "43"):  # perfect separation on the five targets only
        garm[s][:, FIVE] = gy[:, FIVE] * .5 + .25
        oarm[s][:, FIVE] = oy[:, FIVE] * .5 + .25
    oai = screen.image_item(oids, oarm, octl, oy)
    gold = hb130.gold_item(gids, garm, gctl, gy)
    oai_seeds, gold_seeds = {"arm": oarm, "control": octl}, {"arm": garm, "control": gctl}
    rule = hb130.weak_five_rule(ctx, same, same, oai, oai_seeds, gold, gold_seeds)
    for s in ("42", "43"):
        g = rule["per_seed"][s]["gold58_weak_five"]["value"]
        o = rule["per_seed"][s]["oai_weak_five"]["value"]
        assert g == pytest.approx(np.mean([1 - _rank_auc(gctl[s][:, t], gy[:, t]) for t in FIVE]), abs=1e-12)
        assert o == pytest.approx(np.mean([1 - _rank_auc(octl[s][:, t], oy[:, t]) for t in FIVE]), abs=1e-12)
        assert rule["per_seed"][s]["combined"]["value"] == pytest.approx((g + o) / 2, abs=1e-12)
        assert rule["per_seed"][s]["other_seven_macro_jev_consensus"]["value"] == 0.0
    assert rule["combined_positive_both_seeds"] and rule["harm_guard_other_seven"]
    assert rule["registered_go_flag"] == "GO_RULE_MET"
    flat = hb130.weak_five_rule(ctx, same, same, oai, {"arm": octl, "control": octl}, gold,
                                {"arm": gctl, "control": gctl})
    assert flat["registered_go_flag"] == "GO_RULE_NOT_MET" and not flat["combined_positive_both_seeds"]
    worse = {s: same[s].copy() for s in same}
    for s in worse:
        worse[s][:, OTHER] = ((1 - ctx.y[:, OTHER]) * .5 + .25).astype(np.float32)
    harmed = hb130.weak_five_rule(ctx, worse, same, oai, oai_seeds, gold, gold_seeds)
    assert harmed["combined_positive_both_seeds"] is True
    assert harmed["seed_mean_of_deltas"]["other_seven_macro_jev_consensus"]["value"] < -.1
    assert harmed["harm_guard_other_seven"] is False and harmed["registered_go_flag"] == "GO_RULE_NOT_MET"


def test_gold58_roster_is_outside_training_and_arm_labels():
    ids, _ = hb130.gold_roster()
    assert ids == sorted(ids) and len(ids) == 58
    y = hb130.gold_outcomes(ids)
    assert y.shape == (58, 12) and ((y == 0) | (y == 1)).all() and (y.sum(0) > 0).all() and (y.sum(0) < 58).all()
    corpus_ids = set(np.load(CORPUS / "ids.npy", allow_pickle=True).tolist())
    assert set(ids) <= corpus_ids
    roster = hooks.read(ROOT / hooks.control_spec(42).document["config"]["roots"]["ordered"])
    assert not set(ids) & set(roster["train"] + roster["val"])
    header = (ROOT / armmod.LABEL_PATH).read_text(encoding="utf-8").splitlines()
    assert not set(ids) & {line.split(",", 1)[0] for line in header[1:]}


def test_gold58_commands_and_cpu_never_cuda():
    assert hb130.ARM_REFERENCE == REFERENCE and hb130.FIVE_IX == tuple(FIVE)
    assert not torch.cuda.is_initialized()
