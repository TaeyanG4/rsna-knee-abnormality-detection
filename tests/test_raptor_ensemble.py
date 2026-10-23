"""CPU-only shared-input ensemble checks, with no training/checkpoint execution."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
from raptor_common import LAB
import raptor_ensemble as ensemble
from raptor_runtime import predict as predict_single


class TinyModel(torch.nn.Module):
    def __init__(self, offset=0.):
        super().__init__()
        self.register_buffer("bias", torch.linspace(-1, 1, len(LAB)) + offset)
        self.moves = []

    def to(self, device, *args, **kwargs):
        self.moves.append(str(device))
        return super().to(device, *args, **kwargs)

    def forward(self, x):
        return x.mean(dim=(1, 2, 3, 4))[:, None] + self.bias[None]


class ArrayCorpus:
    def __init__(self):
        self.ids = ["unseen-a", "unseen-b", "unseen-c"]
        self.contract = "test-contract"
        self.spec = {"slots": [["Sagittal", 1, 4], ["Coronal", 0, 3]], "depth": 7}
        self.depth = 7
        self.loads, self.current = 0, None
        self.arrays = {uid: (np.arange(7*6*6).reshape(7, 6, 6) + i*8).astype(np.uint8)
                       for i, uid in enumerate(self.ids)}
        for value in self.arrays.values():
            value.setflags(write=False)

    def __contains__(self, uid):
        return uid in self.arrays

    def _touch(self, uid):
        if self.current != uid:
            self.loads += 1; self.current = uid

    def volume(self, uid):
        self._touch(uid)
        return self.arrays[uid]

    def mask(self, uid):
        self._touch(uid)
        return np.ones(self.depth, np.uint8)


def _member(name, offset=0., **overrides):
    metadata = {"contract": "test-contract", "lab": list(LAB), "res": 6, "k_eval": 3,
                "norm": "imagenet", "window_policy": "legacy"}
    metadata.update(overrides)
    return ensemble.EnsembleMember(name, TinyModel(offset), metadata, str(len(name))*64)


@pytest.mark.parametrize("residency", ["resident", "sequential"])
def test_exact_member_predictions_and_mean_with_one_preparation_per_study(residency):
    corpus = ArrayCorpus()
    before = {uid: a.copy() for uid, a in corpus.arrays.items()}
    members = [_member("F0", -.2), _member("F1", .3), _member("F2", .9)]
    individual, reference_receipts = [], []
    for member in members:
        p, r = predict_single(copy.deepcopy(member.model), ArrayCorpus(), corpus.ids,
                              3, 6, "none", "cpu", norm="imagenet", batch=2, log=lambda _: None)
        individual.append(p); reference_receipts.append(r)
    mean, raw, receipt = ensemble.predict_ensemble(members, corpus, corpus.ids, amp="none", device="cpu",
                                                  batch=2, residency=residency, log=None)
    expected = np.stack(individual)
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(mean, expected.mean(axis=0, dtype=np.float32))
    assert corpus.loads == len(corpus.ids) == receipt["input_preparations"]
    assert receipt["input_tensor_hashes"] == reference_receipts[0]["input_tensor_hashes"]
    assert all(r["ordered_inputs_sha256"] == receipt["ordered_inputs_sha256"] for r in reference_receipts)
    assert all(r["status"] == "COMPLETE" and r["completed_studies"] == 3 for r in receipt["member_results"])
    assert receipt["all_finite"] and receipt["neutral_fills"] == receipt["fp32_fallbacks"] == 0
    for uid in corpus.ids:
        np.testing.assert_array_equal(corpus.arrays[uid], before[uid])
    assert all(member.model.moves[-1] == "cpu" for member in members)


@pytest.mark.parametrize("override", [{"contract": "wrong"}, {"res": 7}, {"k_eval": 4},
                                    {"norm": "none"}, {"window_policy": "global-valid"},
                                    {"lab": list(reversed(LAB))}])
def test_contract_mismatch_rejected_before_input_or_device_changes(override):
    corpus = ArrayCorpus()
    members = [_member("F0"), _member("F1", **override)]
    with pytest.raises(ValueError):
        ensemble.predict_ensemble(members, corpus, corpus.ids, amp="none", device="cpu")
    assert corpus.loads == 0 and all(not m.model.moves for m in members)


def test_nonfinite_member_fails_without_omission_or_neutral_fill():
    class Broken(TinyModel):
        def forward(self, x):
            return super().forward(x) * float("nan")
    corpus = ArrayCorpus(); bad = _member("bad"); bad.model = Broken()
    with pytest.raises(ensemble.EnsembleInferenceError) as failed:
        ensemble.predict_ensemble([_member("good"), bad], corpus, corpus.ids, amp="none", device="cpu", batch=2)
    receipt = failed.value.receipt
    assert receipt["status"] == "FAILED" and not receipt["all_finite"]
    assert receipt["failures"][0]["member"] == "bad"
    assert receipt["failures"][0]["uids"] == corpus.ids[:2]
    assert receipt["member_results"][0]["completed_studies"] == 2
    assert receipt["member_results"][1]["completed_studies"] == 0
    assert receipt["neutral_fills"] == receipt["fp32_fallbacks"] == 0


def test_member_cannot_silently_mutate_shared_input():
    class Mutating(TinyModel):
        def forward(self, x):
            x.add_(1.)
            return super().forward(x)
    corpus = ArrayCorpus(); before = corpus.arrays[corpus.ids[0]].copy()
    bad = _member("mutation"); bad.model = Mutating()
    with pytest.raises(ensemble.EnsembleInferenceError, match="mutated its shared input"):
        ensemble.predict_ensemble([bad, _member("later")], corpus, corpus.ids, amp="none", device="cpu")
    np.testing.assert_array_equal(corpus.arrays[corpus.ids[0]], before)


def test_manifest_hash_failures_precede_checkpoint_loading(tmp_path, monkeypatch):
    import raptor_common
    called = []
    monkeypatch.setattr(raptor_common, "load_checkpoint", lambda path: called.append(path))
    weight = tmp_path / "weights.pt"; weight.write_bytes(b"not actually loaded")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"members": [{"name": "F0", "path": "weights.pt", "sha256": "0"*64}]}))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ensemble.load_members(manifest)
    assert not called
    manifest.write_text(json.dumps({"members": [{"name": "F0", "path": "weights.pt", "sha256": ensemble.file_sha256(weight)}]}))
    resolved = ensemble.read_manifest(manifest)
    assert resolved["members"][0]["path"] == str(weight.resolve())


def test_manifest_rejects_duplicate_member_names_or_paths(tmp_path):
    weight = tmp_path / "weights.pt"; weight.write_bytes(b"placeholder")
    record = {"name": "F0", "path": "weights.pt", "sha256": ensemble.file_sha256(weight)}
    manifest = tmp_path / "manifest.json"
    for second in [dict(record), {**record, "name": "F1"}]:
        manifest.write_text(json.dumps({"members": [record, second]}))
        with pytest.raises(ValueError, match="duplicate|repeated"):
            ensemble.read_manifest(manifest)


def test_strict_dicom_failure_receipt_survives_without_model_prediction(tmp_path, monkeypatch):
    import raptor_dicom
    monkeypatch.setattr(raptor_dicom, "_contract_spec", lambda _: {"slots": [["Sagittal", 1, 7]],
                                                                "depth": 7, "img": 6, "crop_mm": 6., "span": [.02, .98]})
    ids = ["acquired-but-missing"]
    records = [{"StudyInstanceUID": ids[0], "SeriesInstanceUID": "series", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1}]
    source = raptor_dicom.DirectoryDicomSource(tmp_path)
    corpus = raptor_dicom.DicomCorpus.from_records(ids, records, source, contract="test-contract")
    with pytest.raises(ensemble.EnsembleInferenceError) as failed:
        ensemble.predict_ensemble([_member("F0")], corpus, ids, amp="none", device="cpu")
    failure = failed.value.receipt["failures"][0]
    assert failure["stage"] == "input_preparation"
    assert failure["source_receipt"]["slots"][0]["status"] == "acquired_files_missing"
    assert failed.value.receipt["member_results"][0]["completed_studies"] == 0


def test_cli_writes_raw_members_only_after_success(tmp_path, monkeypatch):
    import raptor_common
    import raptor_dicom
    import raptor_ensemble_serve
    corpus = ArrayCorpus()
    corpus.receipts = {u: {"uid": u, "status": "rendered"} for u in corpus.ids}
    corpus.manifest = {"fixture": True}
    members = [_member("F0", -.1), _member("F1", .1)]
    monkeypatch.setattr(ensemble, "load_members", lambda path: (members, {"manifest_sha256": "0"*64}))
    monkeypatch.setattr(raptor_dicom, "DicomCorpus", lambda *args: corpus)
    monkeypatch.setattr(raptor_common, "environment", lambda: {"fixture": "cpu"})
    output = tmp_path / "raw.csv"
    args = ["--manifest", "fixture.json", "--test-csv", "fixture.csv", "--series-csv", "fixture.csv",
            "--dicom-root", str(tmp_path), "--out", str(output), "--amp", "none", "--device", "cpu",
            "--residency", "resident", "--batch", "2"]
    receipt = raptor_ensemble_serve.main(args)
    assert receipt["status"] == "COMPLETE" and not receipt["ranking_applied"]
    with np.load(output.with_suffix(".members.npz"), allow_pickle=False) as archive:
        assert archive["member_probabilities"].shape == (2, 3, 12)
        np.testing.assert_array_equal(archive["mean_probabilities"], archive["member_probabilities"].mean(axis=0))
        assert archive["member_names"].tolist() == ["F0", "F1"]
    assert receipt["inference"]["input_preparations"] == 3
    with pytest.raises(FileExistsError):
        raptor_ensemble_serve.main(args)


def test_cli_failed_member_produces_failure_receipt_without_prediction_files(tmp_path, monkeypatch):
    import raptor_common
    import raptor_dicom
    import raptor_ensemble_serve
    corpus = ArrayCorpus(); corpus.receipts = {}; corpus.manifest = {"fixture": True}
    class Bad(TinyModel):
        def forward(self, x):
            raise RuntimeError("explicit failure")
    member = _member("F0"); member.model = Bad()
    monkeypatch.setattr(ensemble, "load_members", lambda _: ([member], {"manifest_sha256": "0"*64}))
    monkeypatch.setattr(raptor_dicom, "DicomCorpus", lambda *args: corpus)
    monkeypatch.setattr(raptor_common, "environment", lambda: {})
    output = tmp_path / "failed.csv"
    with pytest.raises(ensemble.EnsembleInferenceError):
        raptor_ensemble_serve.main(["--manifest", "fixture.json", "--test-csv", "fixture.csv", "--series-csv", "fixture.csv",
                                  "--dicom-root", str(tmp_path), "--out", str(output), "--amp", "none", "--device", "cpu"])
    assert not output.exists() and not output.with_suffix(".members.npz").exists()
    receipt = json.loads(output.with_suffix(".json").read_text())
    assert receipt["status"] == "FAILED" and receipt["inference"]["failures"][0]["member"] == "F0"


def test_real_dicom_decode_shared_across_members_and_default_ids(tmp_path, monkeypatch):
    pydicom = pytest.importorskip("pydicom")
    import raptor_dicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    spec = {"slots": [["Sagittal", 1, 7]], "depth": 7, "img": 6, "crop_mm": 6., "span": [.02, .98]}
    monkeypatch.setattr(raptor_dicom, "_contract_spec", lambda _: spec)
    ids = ["new-a", "new-b"]
    records = []
    for uid in ids:
        path = tmp_path / uid / "series" / "one.dcm"; path.parent.mkdir(parents=True)
        meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage; meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
        ds.SOPClassUID = MRImageStorage; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Rows = ds.Columns = 6; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        ds.InstanceNumber = 1; ds.PixelSpacing = [1., 1.]
        ds.PixelData = np.arange(36, dtype="<u2").tobytes(); ds.save_as(path, enforce_file_format=True)
        records.append({"StudyInstanceUID": uid, "SeriesInstanceUID": "series", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1})
    source = raptor_dicom.DirectoryDicomSource(tmp_path)
    reads = []; original_read = source.read
    def tracked_read(member):
        reads.append(member.path)
        return original_read(member)
    monkeypatch.setattr(source, "read", tracked_read)
    corpus = raptor_dicom.DicomCorpus.from_records(ids, records, source, contract="test-contract")
    mean, raw, receipt = ensemble.predict_ensemble([_member("F0"), _member("F1", .3), _member("F2", -.3)],
                                                  corpus, amp="none", device="cpu", batch=2,
                                                  residency="resident", runtime_policy="strict", log=None)
    # One header read and one selected-pixel read per study, independent of M=3.
    assert len(reads) == 4 and receipt["input_preparations"] == 2
    assert len(corpus.receipts) == 2 and all(r["status"] == "rendered" for r in corpus.receipts.values())
    np.testing.assert_array_equal(mean, raw.mean(axis=0, dtype=np.float32))
    assert receipt["runtime"]["policy"] == "strict" and receipt["window_policy"] == "legacy"
