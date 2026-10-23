"""CPU checks for the pinned EXP189 Gold58 measurement kernel."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import exp189_make_kernels as kernels


class Gold58KernelTests(unittest.TestCase):
    def test_registered_rosters_agree_and_kernel_uses_embedded_ids(self):
        ids, digest = kernels.registered_gold58()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 58)
        self.assertEqual(hashlib.sha256(("\n".join(ids) + "\n").encode("ascii")).hexdigest(), digest)
        notebook = json.loads((kernels.OUT / "gold58/exp189_gold58.ipynb").read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][2]["source"])
        self.assertIn(repr(ids), source)
        self.assertIn(repr(digest), source)
        self.assertNotIn("pd.read_csv(DATA / 'train.csv'", source)

    def test_registration_requires_local_train_series_membership(self):
        ids, _ = kernels.registered_gold58()
        with tempfile.TemporaryDirectory() as tmp:
            local_series = Path(tmp) / "train_series.csv"
            local_series.write_text("StudyInstanceUID,SeriesInstanceUID\n" +
                                    "".join(f"{uid},series\n" for uid in ids[:-1]), encoding="utf-8")
            original = kernels.TRAIN_SERIES
            try:
                kernels.TRAIN_SERIES = local_series
                with self.assertRaisesRegex(ValueError, "absent from local train_series.csv"):
                    kernels.registered_gold58()
            finally:
                kernels.TRAIN_SERIES = original

    def test_gold58_cell_checks_series_and_exports_timing_without_labels(self):
        ids, digest = kernels.registered_gold58()
        notebook = json.loads((kernels.OUT / "gold58/exp189_gold58.ipynb").read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][2]["source"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, work = root / "data", root / "work"
            data.mkdir()
            work.mkdir()
            (data / "train_series.csv").write_text(
                "StudyInstanceUID,SeriesInstanceUID\n" + "".join(f"{uid},series\n" for uid in ids),
                encoding="utf-8",
            )
            for uid in ids:
                series = data / "train_series" / uid / "series"
                series.mkdir(parents=True)
                (series / "slice.dcm").write_bytes(b"synthetic")
            calls = []

            def infer_studies(dirs, **kwargs):
                calls.append([path.name for path in dirs])
                output = kwargs["output_dir"]
                controller = output / "controller"
                controller.mkdir(parents=True)
                (output / "predictions.csv").write_text(
                    "StudyInstanceUID,ACL\n" + "".join(f"{uid},0.500000\n" for uid in ids), encoding="utf-8")
                (controller / "submission_receipt.json").write_text(json.dumps({
                    "rows": 58, "scored": 58, "fallback_studies": 0, "fallback_reasons": {},
                    "workers": [{"status": "PASS", "per_study": [
                        [uid, 3, 1.0, 2.0, 3.0, 4.0] for uid in ids], "fallback": {},
                        "stats": {"rows": 58}, "anchor_methods": {"skm_tea_joint": 58}}],
                }), encoding="utf-8")
                return [{"StudyInstanceUID": uid, "ACL": 0.5} for uid in ids]

            stub = SimpleNamespace(infer_studies=infer_studies, UID="StudyInstanceUID", LABELS=("ACL",),
                                   anatomy_fallbacks=lambda _: {"scored": 58, "fallback": 0})
            scope = {"DATA": data, "WORK": work, "BUNDLE": root / "bundle", "CODE": root / "code",
                     "ANATOMY": root / "anatomy.pt", "OBSERVED": {"tree_sha256": "test"}, "s6": stub}
            exec(compile(source, "<exp189-gold58-test>", "exec"), scope)
            self.assertEqual(calls, [ids])
            report = json.loads((work / "exp189_gold58_timing.json").read_text(encoding="utf-8"))
            self.assertEqual(report["gold58_ids_sha256"], digest)
            self.assertEqual(len(report["per_study"]), 58)
            self.assertEqual(report["per_study"][0]["anchor_s"], 4.0)
            self.assertNotIn("labels", report)
            self.assertTrue((work / "exp189_gold58_predictions.csv").is_file())

            (data / "train_series.csv").write_text(
                "StudyInstanceUID,SeriesInstanceUID\n" + "".join(f"{uid},series\n" for uid in ids[:-1]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssertionError, "absent from train_series.csv"):
                exec(compile(source, "<exp189-gold58-test>", "exec"), scope)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
