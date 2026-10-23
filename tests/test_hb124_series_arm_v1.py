"""H-B-124 adapter tests; real acquisition metadata, synthetic targets only."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.arms import mil_screen_hooks_v1 as hooks
from training.arms.hb124_series_arm_v1 import ARM, TRAIN, PANEL, CORPORA, BoundProvenance
from tools.hb124_hook_build_v1 import documents
from tools.hb124_hook_smoke_v1 import loader_parity
from tools import mil_screen_arm_build_v1 as build

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def spec():
    return documents(42)[3]


@pytest.mark.parametrize("seed", [42, 43])
def test_only_one_control_config_key_and_preserved_source_hashes(seed):
    _, recipe, _, spec = documents(seed)
    control = hooks.control_spec(seed).document
    doc = spec.document
    config = ARM.validate_recipe(spec)
    assert config[ARM.section].pop(ARM.key) == ARM.contract(seed)
    assert config == control["config"]
    assert recipe.document["config"] == doc["config"]
    for key in ("family", "fold", "seed", "attempt", "target_environment", "outputs"):
        assert doc[key] == control[key]
    for key in ("data", "rosters", "controls"):
        assert doc["inputs"][key] == control["inputs"][key]
    for path, digest in control["inputs"]["sources"].items():
        assert doc["inputs"]["sources"][path] == digest == hooks.sha(hooks.ROOT / path)
    assert set(doc["inputs"]["sources"]) == hooks.expected_sources(ARM, hooks.control_spec(seed))
    assert hooks.load_arm("training.arms.hb124_series_arm_v1:ARM", spec) is ARM


def test_label_free_documents_match_shared_configure(monkeypatch):
    """Compare serialization only; deliberately do not validate/read labels."""
    work = hooks.ROOT / ".local/scratch/sol-3/hb124/configure-parity"
    config_dir = work / "configs"
    captured, checked = {}, []
    real_sha = build.sha

    def capture(path, value):
        captured[path.resolve()] = hooks.life.canonical_bytes(value)

    def digest(path):
        return (hashlib.sha256(captured[path.resolve()]).hexdigest()
                if path.resolve() in captured else real_sha(path))

    def recipe_only(arm, spec, root, corpus):
        assert arm is ARM and root == hooks.ROOT
        ARM.validate_recipe(spec)
        checked.append(spec.document["seed"])

    monkeypatch.setattr(build, "same_or_new", capture)
    monkeypatch.setattr(build, "sha", digest)
    monkeypatch.setattr(hooks, "validate_data_with", recipe_only)
    result = build.configure("training.arms.hb124_series_arm_v1:ARM", work, config_dir)
    assert checked == [42, 43]
    for seed in (42, 43):
        recipe_path, recipe, path, spec = documents(seed, build.rel(config_dir))
        assert captured[(hooks.ROOT / recipe_path).resolve()] == hooks.life.canonical_bytes(recipe.document)
        assert captured[(hooks.ROOT / path).resolve()] == hooks.life.canonical_bytes(spec.document)
        assert result["specs"][str(seed)] == dict(path=path, sha256=spec.sha256)


@pytest.mark.parametrize("change", ["second", "contract", "source", "control", "data", "target"])
def test_drift_fails_closed(spec, change):
    doc = spec.document
    if change == "second": doc["config"]["updates"]["head_lr"] *= 2
    if change == "contract": doc["config"][ARM.section][ARM.key]["hidden"] = 16
    if change == "source": doc["inputs"]["sources"].pop("training/src/hb124_series_context_v1.py")
    if change == "control": doc["inputs"]["sources"][hooks.control_path(42)] = "0" * 64
    if change == "data": doc["inputs"]["data"][doc["config"]["roots"]["manifest"]] = "0" * 64
    if change == "target": doc["target_environment"]["hostname"] += "-different"
    with pytest.raises(hooks.life.ContractError):
        ARM.validate_recipe(hooks.life.ArmSpec.from_document(doc))


class ToyMIL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(8, 8)
        self.backbone.num_features = 8
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(8, 12))

    def encode(self, x):
        return self.backbone(x).flatten(1, 2), torch.ones(x.shape[:3], dtype=torch.bool).flatten(1, 2)

    def head(self, encoded):
        return self.classifier(encoded[0].mean(1))

    def forward(self, x):
        return self.head(self.encode(x))


def metadata():
    return dict(series=torch.tensor([[0, 1, 0, 1]]), position=torch.tensor([[0., 0., 1., 1.]]),
                valid=torch.ones(1, 4, dtype=torch.bool))


def test_zero_identity_control_state_rng_schedule_and_finite_grads(spec, tmp_path):
    torch.manual_seed(42)
    model = ToyMIL()
    control = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    ARM.attach(model, spec, hooks.ROOT)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p, model.state_dict()[n]) for n, p in control.state_dict().items())
    x = torch.randn(1, 4, 3, 8)
    meta = ARM.move([metadata()], "cpu")
    for train in (False, True):
        control.train(train)
        model.train(train)
        rng = torch.get_rng_state().clone()
        expected = control(x)
        torch.set_rng_state(rng)
        actual = ARM.forward(model, x, meta)
        assert torch.equal(expected, actual)
    opt0, sched0, _ = hooks.optimizer_stack(control, 10, "cpu", "none")
    opt, sched, _ = hooks.optimizer_stack(model, 10, "cpu", "none")
    assert sched0.state_dict() == sched.state_dict()
    for a, b in zip(opt0.param_groups, opt.param_groups):
        assert {k: v for k, v in a.items() if k != "params"} == {k: v for k, v in b.items() if k != "params"}
    assert {id(p) for p in model.series_context.parameters()} <= {id(p) for p in opt.param_groups[1]["params"]}
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.series_context.out.weight.grad.abs().sum() > 0
    opt.step()
    opt.zero_grad()
    ARM.forward(model, x, meta).square().mean().backward()
    assert model.series_context.conv.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    state_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), state_path)
    clone = ARM.attach(ToyMIL(), spec, hooks.ROOT)
    hooks.strict_load(clone, torch.load(state_path, weights_only=True))
    assert torch.equal(ARM.forward(clone.eval(), x, meta), ARM.forward(model.eval(), x, meta))


def test_hook_context_masks_series(spec):
    model = ARM.attach(ToyMIL(), spec, hooks.ROOT).eval()
    with torch.no_grad(): model.series_context.out.weight.fill_(.1)
    x = torch.randn(1, 4, 3, 8)
    meta = metadata()
    baseline = model.series_context.encoded(model.encode(x), meta)[0].reshape(1, 4, 3, 8)
    x[:, [1, 3]] += 100
    changed = model.series_context.encoded(model.encode(x), meta)[0].reshape(1, 4, 3, 8)
    assert torch.equal(baseline[:, [0, 2]], changed[:, [0, 2]])


@pytest.mark.parametrize("seed", [42, 43])
def test_real_images_multi_epoch_two_persistent_worker_parity(seed):
    assert loader_parity(seed)["images_targets_ids_exact"]


@pytest.mark.parametrize("path", [TRAIN, PANEL])
def test_real_metadata_binding_and_panel_copy_mapping(spec, path):
    corpus = hooks.common.Corpus(hooks.ROOT / path)
    uid = str(corpus.ids[0])
    inner = hooks.MILWindows(corpus.dir, [uid], {uid: np.zeros(12, np.float32)}, 36, False, 42)
    wrapped = ARM.dataset(inner, spec, hooks.ROOT, train=False)
    x, y, got_uid, meta = wrapped[0]
    assert got_uid == uid and meta["series"].shape == (36,)
    assert wrapped.k == inner.k and wrapped.ids == inner.ids
    assert torch.equal(x, inner[0][0])
    batch = ARM.collate_fn([(x, y, uid, meta)])
    assert all(v.shape == (1, 36) for v in ARM.move([batch[3]], "cpu").values())
    if path == PANEL:
        provenance = wrapped.provenance
        sizes = [int(s[2]) for s in corpus.spec["slots"]]
        copies = 0
        with (corpus.dir / "picks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["slots"][1] and row["slots"][1].get("fallback") == "copy_of_sagittal_fluid1":
                    series, _ = provenance.by_uid[row["uid"]]
                    assert np.all(series[sizes[0]:sum(sizes[:2])] == series[0])
                    copies += 1
        assert copies == provenance.copies == 2177


def test_metadata_rejects_wrong_binding_and_shape(spec):
    corpus = hooks.common.Corpus(hooks.ROOT / TRAIN)
    ds = hooks.MILWindows(corpus.dir, [], {}, 12, True, 42)
    doc = spec.document
    doc["inputs"]["sources"][TRAIN + "/picks.jsonl"] = "0" * 64
    with pytest.raises(hooks.life.ContractError, match="sidecar"):
        ARM.dataset(ds, hooks.life.ArmSpec.from_document(doc), hooks.ROOT, train=True)
    with pytest.raises(hooks.life.ContractError, match="metadata"):
        ARM.move([], "cpu")
    model = ARM.attach(ToyMIL(), spec, hooks.ROOT)
    with pytest.raises(hooks.life.ContractError, match="shape"):
        ARM.forward(model, torch.zeros(2, 4, 3, 8), metadata())
    with pytest.raises(hooks.life.ContractError, match="already"):
        ARM.attach(model, spec, hooks.ROOT)


def test_probe_and_provenance(spec):
    model = ARM.attach(ToyMIL(), spec, hooks.ROOT).eval()
    x = torch.zeros(1, 1, 3, 8)
    assert torch.equal(model(x), ARM.forward(model, x, ARM.probe("cpu")))
    assert ARM.provenance(spec)["series_context"] == ARM.contract(42)
    assert not torch.cuda.is_initialized()
