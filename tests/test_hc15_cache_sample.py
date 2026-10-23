"""New sample CLI gates using metadata/SQLite only; never native decode/GPU."""
import copy
from contextlib import closing
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import hc15_cache_sample as sample
from training.adapters.hc15_cache_store import CacheError, GEOMETRY, digest_file


class EnvironmentMetadataTests(unittest.TestCase):
    def test_own_record_ignores_vendored_metadata_records(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory).resolve()
            executable = prefix / "python.exe"
            executable.write_bytes(b"synthetic interpreter, never executed")
            (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            site = prefix / "Lib/site-packages"
            distributions = []
            for name in ("torch", "numpy", "pydicom", "psutil", "opencv-python", "setuptools"):
                own = PurePosixPath(f"{name}-1.0.dist-info/RECORD")
                paths = [own]
                if name == "setuptools":
                    paths += [PurePosixPath("setuptools/_vendor/packaging-24.2.dist-info/RECORD"),
                              PurePosixPath("setuptools/_vendor/wheel-0.45.dist-info/RECORD")]
                for relative in paths:
                    path = site / str(relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(str(relative))
                distributions.append(SimpleNamespace(metadata={"Name": name},
                                                      version="1.0+cpu" if name == "torch" else "1.0",
                                                      files=paths, locate_file=lambda p: site / str(p)))
            with patch.object(sample.sys, "prefix", str(prefix)), patch.object(sample.sys, "base_prefix", str(prefix.parent)), \
                    patch.object(sample.sys, "executable", str(executable)), patch.object(sample.sys, "_base_executable", str(executable)), \
                    patch.object(sample.sysconfig, "get_path", return_value=str(site)), \
                    patch.object(sample.importlib.metadata, "distributions", return_value=distributions):
                identity = sample.environment_identity()
                record = site / "setuptools-1.0.dist-info/RECORD"
                self.assertEqual(identity["packages"]["setuptools"]["record"], str(record))
                self.assertEqual(identity["packages"]["setuptools"]["record_sha256"], digest_file(record))
                # Ambiguous top-level metadata must still fail closed.
                distributions[-1].files.append(PurePosixPath("other-1.0.dist-info/RECORD"))
                with self.assertRaisesRegex(CacheError, "exact RECORD binding"):
                    sample.environment_identity()

    def test_project_metadata_excluded_by_explicit_venv_package_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory).resolve() / "venv"
            prefix.mkdir()
            executable = prefix / "python.exe"
            executable.write_bytes(b"synthetic interpreter")
            (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            purelib, platlib = prefix / "purelib", prefix / "platlib"
            purelib.mkdir(); platlib.mkdir()
            distributions = []
            for name in ("torch", "numpy", "pydicom", "psutil", "opencv-python"):
                own = PurePosixPath(f"{name}-1.0.dist-info/RECORD")
                record = purelib / str(own)
                record.parent.mkdir(); record.write_text(name)
                distributions.append(SimpleNamespace(metadata={"Name": name},
                                                      version="1.0+cpu" if name == "torch" else "1.0",
                                                      files=[own], locate_file=lambda p: purelib / str(p)))
            project_metadata = SimpleNamespace(metadata={"Name": "kneeops-local"}, version="1.0", files=[])

            def enumerate_metadata(**kwargs):
                # This is the observed real failure: unrestricted discovery
                # sees the project's egg-info with no installed-wheel RECORD.
                if not kwargs:
                    return distributions + [project_metadata]
                self.assertEqual(kwargs, {"path": [str(purelib), str(platlib)]})
                return distributions

            with patch.object(sample.sys, "prefix", str(prefix)), patch.object(sample.sys, "base_prefix", str(prefix.parent)), \
                    patch.object(sample.sys, "executable", str(executable)), patch.object(sample.sys, "_base_executable", str(executable)), \
                    patch.object(sample.sysconfig, "get_path", side_effect=lambda key, **kwargs: str(purelib if key == "purelib" else platlib)), \
                    patch.object(sample.importlib.metadata, "distributions", side_effect=enumerate_metadata) as discover:
                identity = sample.environment_identity()
                self.assertEqual(identity["package_directories"], [str(purelib), str(platlib)])
                self.assertNotIn("kneeops-local", identity["packages"])
                discover.assert_called_once_with(path=[str(purelib), str(platlib)])
                with patch.object(sample.sysconfig, "get_path", return_value=str(prefix.parent)):
                    with self.assertRaisesRegex(CacheError, "outside isolated"):
                        sample.environment_identity()


class SampleCLITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.out = self.root / "allocated"
        self.archive = self.root / "source.zip"
        self.archive.write_bytes(b"not decoded by these tests")
        self.index = self.root / "index.sqlite"
        self.rows = [{"StudyInstanceUID": "10", "SeriesInstanceUID": f"1.{i}",
                      "Anatomical_Plane": "Sagittal"} for i in range(1, 5)]
        with closing(sqlite3.connect(self.index)) as db, db:
            db.execute("CREATE TABLE files(path TEXT,info TEXT,split TEXT,uid TEXT,series TEXT)")
            for row in self.rows:
                for k in range(3):
                    db.execute("INSERT INTO files VALUES (?,?,?,?,?)",
                               (f"train_series/10/{row['SeriesInstanceUID']}/{k}.dcm",
                                json.dumps({"CRC": k}), "train", "10", row["SeriesInstanceUID"]))
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"gpu_dispatch_allowed": False,
                                            "archive": {"path": str(self.archive), "index": str(self.index),
                                                        "bytes": self.archive.stat().st_size}}))
        for relative in sample.SOURCE_PATHS.values():
            dest = self.root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((sample.ROOT / relative).read_bytes())
        self.environment = {"python": "dummy-never-launched", "worker_environment": sample.CPU_ENV,
                            "packages": {"torch": {"version": "2.5.1+cpu"}}}
        for target, value in (("ROOT", self.root), ("environment_identity", lambda: copy.deepcopy(self.environment)),
                              ("frozen_contract", self.frozen)):
            patcher = patch.object(sample, target, value)
            patcher.start(); self.addCleanup(patcher.stop)

    def frozen(self, root, manifest, expected, adapter, decoder, *, index_sha256):
        return {"source_manifest_sha256": expected, "adapter_sha256": digest_file(adapter),
                "decoder_sha256": digest_file(decoder),
                "producer_sha256": digest_file(self.root / sample.SOURCE_PATHS["store"]),
                "archive_index_sha256": index_sha256, "geometry": GEOMETRY,
                "source_archive": json.loads(self.manifest.read_text())["archive"],
                "study_uids": ["10"], "series": copy.deepcopy(self.rows)}

    def allocate(self, **budget):
        return sample.allocate(self.root, self.manifest, digest_file(self.manifest), self.out, **budget)

    def ready(self, allocation=None):
        allocation = allocation or self.out / "allocation.json"
        content = json.loads(allocation.read_text())
        review = self.root / "review.json"
        review.write_text(json.dumps({"schema": "hc15-cache-sample-review-v1", "verdict": "READY",
                                      "reviewer": "independent synthetic reviewer", "read_only": True,
                                      "allocation_sha256": digest_file(allocation),
                                      "source_sha256": content["source_sha256"]}))
        return review

    def test_metadata_allocation_binds_first_three_and_full_contract(self):
        path = self.allocate()
        allocation = sample.validate(path, self.ready(path))
        self.assertEqual([s["series"] for s in allocation["sample"]], ["1.1", "1.2", "1.3"])
        self.assertEqual(sum(s["raw_frames"] for s in allocation["sample"]), 9)
        self.assertEqual(len(allocation["contract"]["series"]), 4)
        self.assertEqual(allocation["contract"]["archive_index_sha256"], digest_file(self.index))
        self.assertFalse(allocation["training_ready"])
        self.assertFalse(allocation["gpu_dispatch_allowed"])
        self.assertEqual(set(self.out.iterdir()), {path})
        with self.assertRaisesRegex(CacheError, "already exists"):
            self.allocate()

    def test_invalid_limits_fail_before_any_allocation(self):
        for budget in ({"max_series": 0}, {"max_series": 4}, {"max_series": True},
                       {"max_raw_frames": 193}, {"max_raw_frames": 0},
                       {"max_bytes": sample.MAX_BYTES + 1}, {"max_bytes": 1},
                       {"seconds": 0}, {"seconds": 121}, {"seconds": float("nan")},
                       {"seconds": float("inf")}, {"reserve_bytes": -1}):
            with self.subTest(budget=budget), self.assertRaises(CacheError):
                self.allocate(**budget)
            self.assertFalse(self.out.exists())

    def test_raw_frame_cap_is_not_retained_frame_count_or_adaptive_selection(self):
        with closing(sqlite3.connect(self.index)) as db, db:
            for k in range(3, 193):
                db.execute("INSERT INTO files VALUES (?,?,?,?,?)",
                           (f"train_series/10/1.1/{k}.dcm", "{}", "train", "10", "1.1"))
        with self.assertRaisesRegex(CacheError, "Raw-frame cap"):
            self.allocate()
        self.assertFalse(self.out.exists())

    def test_ready_exact_hash_and_readonly_review_required_before_dispatch(self):
        path = self.allocate(); review_path = self.ready(path)
        original = json.loads(review_path.read_text())
        changes = [{"verdict": "HOLD"}, {"read_only": False}, {"reviewer": ""},
                   {"allocation_sha256": "f" * 64}, {"source_sha256": {}}]
        with patch.object(sample, "_supervise") as dispatch:
            for change in changes:
                review_path.write_text(json.dumps(original | change))
                with self.subTest(change=change), self.assertRaises(CacheError):
                    sample.run(path, review_path)
                self.assertFalse((self.out / "attempt.json").exists())
            dispatch.assert_not_called()

    def test_bad_allocation_even_with_matching_review_fails_before_dispatch(self):
        path = self.allocate(); original = json.loads(path.read_text())
        candidates = []
        for key, value in [("max_series", 4), ("seconds", 121), ("max_bytes", 1)]:
            changed = copy.deepcopy(original); changed["budget"][key] = value; candidates.append(changed)
        changed = copy.deepcopy(original); changed["sample"][0]["raw_frames"] = 1; candidates.append(changed)
        changed = copy.deepcopy(original); changed["contract"]["series"].pop(); candidates.append(changed)
        changed = copy.deepcopy(original); changed["gpu_dispatch_allowed"] = True; candidates.append(changed)
        changed = copy.deepcopy(original); changed["training_ready"] = True; candidates.append(changed)
        with patch.object(sample, "_supervise") as dispatch:
            for changed in candidates:
                path.write_bytes(sample.canonical(changed))
                with self.subTest(changed=changed["budget"]), self.assertRaises(CacheError):
                    sample.run(path, self.ready(path))
                self.assertFalse((self.out / "attempt.json").exists())
            dispatch.assert_not_called()

    def test_source_and_environment_drift_block_before_dispatch(self):
        path = self.allocate(); review = self.ready(path)
        targets = [self.index, self.archive] + [self.root / p for p in sample.SOURCE_PATHS.values()]
        with patch.object(sample, "_supervise") as dispatch:
            for target in targets:
                old = target.read_bytes(); stat = target.stat()
                # A same-size edit catches byte bindings (or archive mtime) too.
                target.write_bytes(bytes([old[0] ^ 1]) + old[1:])
                with self.subTest(target=target), self.assertRaises((CacheError, sqlite3.DatabaseError)):
                    sample.run(path, review)
                target.write_bytes(old)
                os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.environment["packages"]["torch"]["version"] = "new+cpu"
            with self.assertRaisesRegex(CacheError, "environment drift"):
                sample.run(path, review)
            dispatch.assert_not_called()

    def test_failed_timeout_and_crashed_dispatch_each_block_replay(self):
        path = self.allocate(); review = self.ready(path)
        with patch.object(sample, "_supervise", return_value={"status": "FAILED", "child_exit_code": 1}) as dispatch:
            result = sample.run(path, review)
            self.assertEqual(result["status"], "FAILED")
            attempt = json.loads((self.out / "attempt.json").read_text())
            self.assertEqual(attempt["budget"]["seconds"], 120)
            self.assertEqual(attempt["status"], "ATTEMPTED_NO_RETRY")
            with self.assertRaisesRegex(CacheError, "Earlier attempt"):
                sample.run(path, review)
            self.assertEqual(dispatch.call_count, 1)

    def test_exception_and_timeout_are_not_success_or_automatic_retry(self):
        path = self.allocate(); review = self.ready(path)
        with patch.object(sample, "_supervise", side_effect=TimeoutError("test timeout")) as dispatch:
            result = sample.run(path, review)
            self.assertEqual(result["failure"]["type"], "TimeoutError")
            self.assertFalse(result["training_ready"])
            with self.assertRaisesRegex(CacheError, "Earlier attempt"):
                sample.run(path, review)
            self.assertEqual(dispatch.call_count, 1)

    def test_exclusive_attempt_marker_blocks_a_competing_dispatch(self):
        path = self.allocate(); review = self.ready(path)
        original_validate = sample.validate

        def competing(*args):
            result = original_validate(*args)
            sample.write_new(self.out / "attempt.json", {"other_attempt": True})
            return result

        with patch.object(sample, "validate", side_effect=competing), patch.object(sample, "_supervise") as dispatch:
            with self.assertRaises(FileExistsError):
                sample.run(path, review)
            dispatch.assert_not_called()

    def test_supervisor_zero_without_worker_validation_is_failure(self):
        path = self.allocate(); review = self.ready(path)
        with patch.object(sample, "_supervise", return_value={"status": "COMPLETE", "child_exit_code": 0}):
            result = sample.run(path, review)
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse((self.out / "cache").exists())

    def test_aggregate_bytes_veto_otherwise_valid_sample(self):
        path = self.allocate(); review = self.ready(path)
        allocation = json.loads(path.read_text())

        def dispatch(*args):
            sample.write_new(self.out / "worker.json", {"status": "SAMPLE_VALIDATED", "complete": False,
                                                        "allocation_sha256": digest_file(path),
                                                        "series": allocation["sample"]})
            return {"status": "COMPLETE", "child_exit_code": 0}

        with patch.object(sample, "_supervise", side_effect=dispatch), patch.object(sample, "directory_bytes", return_value=sample.MAX_BYTES):
            result = sample.run(path, review)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["failure"]["type"], "AggregateBudgetExceeded")

    def test_aggregate_elapsed_time_veto_otherwise_valid_sample(self):
        path = self.allocate(); review = self.ready(path)
        allocation = json.loads(path.read_text())

        def dispatch(*args):
            sample.write_new(self.out / "worker.json", {"status": "SAMPLE_VALIDATED", "complete": False,
                                                        "allocation_sha256": digest_file(path),
                                                        "series": allocation["sample"]})
            return {"status": "COMPLETE", "child_exit_code": 0}

        with patch.object(sample, "_supervise", side_effect=dispatch), patch.object(sample.time, "monotonic", side_effect=[0, 0, 121]):
            result = sample.run(path, review)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["failure"]["type"], "AggregateBudgetExceeded")

    def test_valid_partial_sample_uses_cpu_env_then_restores_it(self):
        path = self.allocate(); review = self.ready(path)
        allocation = json.loads(path.read_text())
        before = {key: os.environ.get(key) for key in sample.CPU_ENV}

        def dispatch(argv, out, timeout, cwd):
            self.assertEqual({key: os.environ.get(key) for key in sample.CPU_ENV}, sample.CPU_ENV)
            self.assertEqual(argv[:2], [self.environment["python"], "-I"])
            self.assertLessEqual(timeout, 116)
            sample.write_new(self.out / "worker.json", {"status": "SAMPLE_VALIDATED", "complete": False,
                                                        "allocation_sha256": digest_file(path),
                                                        "series": allocation["sample"]})
            return {"status": "COMPLETE", "child_exit_code": 0}

        with patch.object(sample, "_supervise", side_effect=dispatch):
            result = sample.run(path, review)
        self.assertEqual(result["status"], "SAMPLE_VALIDATED")
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["gpu_dispatch_allowed"])
        self.assertEqual({key: os.environ.get(key) for key in sample.CPU_ENV}, before)

    def test_worker_cannot_start_directly_without_attempt(self):
        path = self.allocate(); review = self.ready(path)
        with self.assertRaises(FileNotFoundError):
            sample.worker(path, review, "not-a-parent-token")
        self.assertFalse((self.out / "worker.claim").exists())


if __name__ == "__main__":
    unittest.main()
