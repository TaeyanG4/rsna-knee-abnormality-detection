"""CPU-only EXP132 converter, frozen queue and auxiliary path tests."""
from __future__ import annotations

import ast
import contextlib
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "src"))
from tools import exp132_external_aux as exp  # noqa: E402
import raptor_exp132 as runner  # noqa: E402
from raptor_radimagenet import CenterSliceWindows  # noqa: E402
from raptor_data import collate  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail_lazy(*args, **kwargs):
        raise AssertionError("EXP132 CPU tests must not call CUDA _lazy_init")
    def fail_init(*args, **kwargs):
        raise AssertionError("EXP132 CPU tests must not call CUDA init")
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail_lazy)
    monkeypatch.setattr(torch.cuda, "init", fail_init)
    with torch.device("cpu"):
        yield
    assert not torch.cuda.is_initialized()


def small_volume(offset=0):
    a = np.arange(5 * 12 * 13, dtype=np.uint16).reshape(5, 12, 13)
    return (a + offset).astype(np.uint16)


def test_restricted_pickle_and_kneemri_contract(tmp_path):
    src = exp.source_dir("KMRI", tmp_path)
    (src / "vol01").mkdir(parents=True)
    (src / "provenance.json").write_text('{"source":"synthetic"}', encoding="utf-8")
    with (src / "metadata.csv").open("w", encoding="utf-8") as f:
        f.write("examId,seriesNo,aclDiagnosis,kneeLR,roiX,roiY,roiZ,roiHeight,roiWidth,roiDepth,volumeFilename\n")
        f.write("1,1,2,L,0,0,0,1,1,1,vol01/a.pck\n")
        f.write("2,1,0,R,0,0,0,1,1,1,vol01/missing.pck\n")
    with (src / "vol01/a.pck").open("wb") as f:
        pickle.dump(small_volume(), f, protocol=4)
    receipt = exp.convert("kneemri", tmp_path, shard_size=1)
    assert receipt["studies"] == 1 and receipt["missing_files_skipped"] == 1
    out = exp.corpus_dir("KMRI", tmp_path)
    corpus = exp.Corpus(out)
    uid = str(corpus.ids[0])
    assert corpus.volume(uid).shape == (64, 336, 336)
    # v2: KneeMRI (sagittal PD-FS) fills only the sagittal fluid-sensitive group (slots 0-17)
    assert corpus.mask(uid)[:18].sum() == 18 and corpus.mask(uid)[18:].sum() == 0
    assert (out / "EXT132_KNEEMRI_labels.csv").read_text(encoding="utf-8").splitlines()[1].endswith(",1,1")
    assert corpus.spec["slots"] == [list(x) for x in exp.SLOTS64]
    for name, sha in receipt["output_sha256"].items():
        assert exp.digest(out / name) == sha
    with pytest.raises(FileExistsError):
        exp.convert("kneemri", tmp_path)


def test_malicious_pickle_rejected(tmp_path):
    class Attack:
        def __reduce__(self):
            return (eval, ("1+1",))
    path = tmp_path / "bad.pck"
    path.write_bytes(pickle.dumps(Attack(), protocol=4))
    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        exp.load_kneemri_pickle(path)


def test_mrnet_absent_and_tiny_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="MRNet directory absent"):
        exp.convert("mrnet", tmp_path)
    src = exp.source_dir("MRN", tmp_path)
    src.mkdir(parents=True)
    (src / "provenance.json").write_text('{"source":"synthetic"}', encoding="utf-8")
    for split, case in (("train", 1), ("valid", 2)):
        for name, value in (("abnormal", 1), ("acl", 1), ("meniscus", 0)):
            (src / f"{split}-{name}.csv").write_text(f"{case},{value}\n", encoding="utf-8")
        for plane in ("sagittal", "coronal", "axial"):
            folder = src / split / plane
            folder.mkdir(parents=True)
            fake = np.broadcast_to((np.arange(5, dtype=np.uint8) + case)[:, None, None],
                                   (5, 256, 256)).copy()
            np.save(folder / f"{case:04d}.npy", fake)
    receipt = exp.convert("mrnet", tmp_path, shard_size=1)
    assert receipt["studies"] == 2 and receipt["missing_files_skipped"] == 0
    corpus = exp.Corpus(exp.corpus_dir("MRN", tmp_path))
    for uid in corpus.ids:
        m = corpus.mask(str(uid))  # v2: sag T2FS -> 0-17, cor T1 -> 44-51 (non-FS group), axial -> 52-63
        assert m[:18].sum() == 18 and m[18:44].sum() == 0 and m[44:52].sum() == 8 and m[52:].sum() == 12
    text = (corpus.dir / "EXT132_MRNET_labels.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "StudyInstanceUID,mrn_abnormal,mrn_acl,mrn_meniscus"
    assert len(text.splitlines()) == 3


def test_missing_mrnet_plane_fails_loudly(tmp_path):
    src = exp.source_dir("MRN", tmp_path)
    src.mkdir(parents=True)
    for split in ("train", "valid"):
        for name in ("abnormal", "acl", "meniscus"):
            (src / f"{split}-{name}.csv").write_text("1,0\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="MRNet plane absent"):
        exp.mrnet_records(src)


def test_aux_schedule_rng_and_serving_state_bitwise(tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 4)
            self.norm = torch.nn.LayerNorm(4)
            self.att = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                                           torch.nn.Dropout(.2), torch.nn.Linear(8, 12))
            self.clsW = torch.nn.Parameter(torch.randn(12, 4))
            self.clsb = torch.nn.Parameter(torch.zeros(12))
        def encode(self, x):
            return self.backbone(x)
        def forward(self, x):
            h = self.norm(self.encode(x))
            a = torch.softmax(self.att(h), dim=1)
            return (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1) + self.clsb
    torch.manual_seed(42)
    model_a = Tiny()
    state = copy.deepcopy(model_a.state_dict())
    rng_before = torch.get_rng_state().clone()
    heads = runner.make_aux_head(model_a, {"KMRI": 2, "MRN": 3}, 42)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model_a.state_dict().keys() == state.keys()
    assert all(torch.equal(model_a.state_dict()[k], state[k]) for k in state)
    x = torch.randn(2, 3, 3)
    assert runner.aux_logits(model_a, heads["KMRI"], x).shape == (2, 2)
    aux_loss = runner.auxiliary_backward(model_a, heads["KMRI"], x, torch.ones(2, 2),
                                         torch.amp.GradScaler("cpu", enabled=False))
    assert aux_loss > 0 and model_a.backbone.weight.grad is not None
    assert heads["KMRI"].weight.grad is not None and model_a.clsW.grad is None
    model_a.zero_grad(set_to_none=True)
    assert [i for i in range(1, 17) if runner.aux_due(i)] == [4, 8, 12, 16]
    assert not any(runner.aux_due(i, False) for i in range(1, 17))
    # Build a tiny real Corpus and compare shuffled CenterSliceWindows loaders.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    ids = [f"TEST_{i}" for i in range(8)]
    one = np.tile(np.arange(64, dtype=np.uint8)[:, None, None], (1, 8, 8))
    vol = np.stack([one] * 8)
    np.save(corpus_dir / "ids.npy", np.asarray(ids))
    np.save(corpus_dir / "masks.npy", np.ones((8, 64), dtype=np.uint8))
    np.save(corpus_dir / "vols_shard00.npy", vol)
    spec = {"slots": exp.SLOTS64, "depth": 64, "img": 8}
    (corpus_dir / "manifest.json").write_text(json.dumps({"contract": "tiny", "spec": spec,
        "shards": [{"start": 0, "stop": 8, "file": "vols_shard00.npy"}]}), encoding="utf-8")
    labels = {u: np.zeros(12, np.float32) for u in ids}
    loaders = []
    for _ in range(2):
        ds = CenterSliceWindows(corpus_dir, ids, labels, 8, 3, True, "radimagenet", 42,
                                window_policy="within-slot-valid", input_policy="center3")
        loaders.append(torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
            drop_last=True, collate_fn=collate, generator=torch.Generator(device="cpu").manual_seed(42)))
    # Disabled EXP132 competition helper vs frozen EXP127 inline arithmetic,
    # with real dropout and identical per-step RNG state.
    model_b = Tiny(); model_b.load_state_dict(state)
    model_a.load_state_dict(state)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    lossf = torch.nn.BCEWithLogitsLoss()
    for epoch in range(2):
        for ds in (loaders[0].dataset, loaders[1].dataset):
            ds.set_epoch(epoch)
        for a, b in zip(*loaders, strict=True):
            assert a[2] == b[2] and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            # Adapt the actual loader images to Tiny while preserving data/order.
            batch = a[0].mean(dim=(-1, -2))
            y = a[1]
            before = torch.get_rng_state().clone()
            loss_a = runner.competition_backward(model_a, batch, y, lossf, scaler, len(a[2]), contextlib.nullcontext())
            after_a = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            logits_b = model_b(batch)
            loss_b = lossf(logits_b.float(), y)
            scaler.scale(loss_b).backward()
            assert torch.equal(loss_a, loss_b)
            assert torch.equal(after_a, torch.get_rng_state())
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 3.)
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), 3.)
            opt_a.step(); opt_b.step()
            opt_a.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
        assert all(torch.equal(model_a.state_dict()[k], model_b.state_dict()[k]) for k in state)
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert opt_a.state[pa].keys() == opt_b.state[pb].keys()
            assert all(torch.equal(opt_a.state[pa][key], opt_b.state[pb][key])
                       for key in opt_a.state[pa])


def test_frozen_parser_and_queue_plan_never_launches(tmp_path, monkeypatch):
    assert runner.build_parser()._option_string_actions["--variant"].choices == ["KMRI", "MRN", "BOTH"]
    assert exp.tag_for("KMRI", 42) == "EXP132_KMRI_S42_F0"
    with pytest.raises(ValueError):
        exp.tag_for("KMRI", 44)
    monkeypatch.setattr(exp, "register", lambda variant, root: None)
    monkeypatch.setattr(exp, "runner_argv", lambda variant, seed, root, workers: ["python", "runner", str(seed)])
    out = tmp_path / "plan.json"
    entries = exp.queue("KMRI", "PREVIOUS", str(out), root=tmp_path)
    assert entries[0]["after"] == "PREVIOUS" and entries[1]["after"] == entries[0]["tag"]
    assert json.loads(out.read_text(encoding="utf-8")) == entries
    with pytest.raises(FileExistsError):
        exp.queue("KMRI", "PREVIOUS", str(out), root=tmp_path)
    for path in ("tools/exp132_external_aux.py", "training/src/raptor_exp132.py"):
        ast.parse((ROOT / path).read_text(encoding="utf-8"))


def test_competition_update_core_retains_control_order():
    source = (ROOT / "training/src/raptor_exp132.py").read_text(encoding="utf-8")
    frozen = (ROOT / "training/runners/exp127_radr50_single.py").read_text(encoding="utf-8")
    for snippet in (
        "shuffle_generator = torch.Generator().manual_seed(a.seed)",
        "tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=dev == \"cuda\",",
        "steps = max(math.ceil(len(tl) / a.accum) * epochs, 1)",
        "loss = lossf(logits.float(), y)",
        "scaler.scale(loss).backward()",
        "scale_before = scaler.get_scale()\n            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)",
        "sched.step()",
    ):
        assert snippet in frozen and snippet in source
    assert source.index("loss = competition_backward(model, x, y, lossf, scaler, len(batch_uids), autocast(amp_dtype))") < source.index("if aux_due(i + 1, aux_enabled):")
    assert source.index("if aux_due(i + 1, aux_enabled):") < source.index("scaler.unscale_(opt)")
    assert "if aux_enabled:\n        head += list(aux_heads.parameters())" in source
    assert "if a.disable_aux and not a.validate_only:" in source


def test_kneemri_bare_filenames_resolve_in_vol_subfolders(tmp_path):
    # Real distribution: metadata.csv names bare files that live under vol01..volNN.
    src = exp.source_dir("KMRI", tmp_path)
    (src / "vol02").mkdir(parents=True)
    (src / "provenance.json").write_text('{"source":"synthetic"}', encoding="utf-8")
    with (src / "metadata.csv").open("w", encoding="utf-8") as f:
        f.write("examId,seriesNo,aclDiagnosis,kneeLR,roiX,roiY,roiZ,roiHeight,roiWidth,roiDepth,volumeFilename\n")
        f.write("7,5,1,0,0,0,0,1,1,1,7-5.pck\n")
        f.write("7,8,0,0,0,0,0,1,1,1,7-8.pck\n")
    with (src / "vol02/7-5.pck").open("wb") as f:
        pickle.dump(small_volume(), f, protocol=2)
    receipt = exp.convert("kneemri", tmp_path, shard_size=4)
    assert receipt["studies"] == 1 and receipt["missing_files_skipped"] == 1
    assert [str(u) for u in exp.Corpus(exp.corpus_dir("KMRI", tmp_path)).ids] == ["KMRI_7_5"]
