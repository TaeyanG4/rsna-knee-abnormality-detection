"""Launcher identity regression only: no DICOM/native backend/sample execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from tools import hc15_cache_sample_v2 as sample
from training.adapters.hc15_cache_store import CacheError, digest_file

ARTIFACTS = sample.ROOT / ".local/hc15-launcher-repair-v1"


class LauncherRepairTests(unittest.TestCase):
    def setUp(self):
        self.policy = sample.launcher_policy()
        self.argv = [self.policy["launcher"]["path"], "-I", "metadata-only-never-executed"]
        parent = {"pid": 11, "create_time": 1.0, "ppid": 0,
                  "exe": self.policy["worker"]["path"], "argv": ["parent"]}
        worker = {"pid": 33, "create_time": 3.0, "ppid": 22 if sys.platform == "win32" else 11,
                  "exe": self.policy["worker"]["path"], "argv": self.argv}
        redirector = {"pid": 22, "create_time": 2.0, "ppid": 11,
                      "exe": self.policy["launcher"]["path"], "argv": self.argv}
        self.chain = [worker, redirector, parent] if sys.platform == "win32" else [worker, parent]
        self.attempt = {"launcher_policy": self.policy, "parent_identity": parent,
                        "worker_argv": self.argv, "started_unix": 1.5}

    def test_bound_chain_accepts_only_exact_parent_pid_and_creation_time(self):
        sample.verify_chain(self.attempt, self.chain)
        for key, value in (("pid", 999), ("create_time", 1.1)):
            changed = copy.deepcopy(self.attempt)
            changed["parent_identity"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(CacheError, "identity changed|unrelated"):
                sample.verify_chain(changed, self.chain)

    def test_redirector_identity_or_argv_and_changed_traversal_are_rejected(self):
        changed = copy.deepcopy(self.chain)
        changed[0]["argv"] = self.argv + ["unexpected"]
        with self.assertRaisesRegex(CacheError, "argv differs"):
            sample.verify_chain(self.attempt, changed)
        changed = copy.deepcopy(self.chain)
        changed[0]["ppid"] = 999
        with self.assertRaisesRegex(CacheError, "Broken launcher"):
            sample.verify_chain(self.attempt, changed)
        if sys.platform == "win32":
            changed = copy.deepcopy(self.chain)
            changed[1]["exe"] = self.policy["worker"]["path"]
            with self.assertRaisesRegex(CacheError, "Unapproved redirector"):
                sample.verify_chain(self.attempt, changed)
        nodes = {v["pid"]: v for v in self.chain}
        reads = []

        def reused(pid):
            value = copy.deepcopy(nodes[pid])
            if len(reads) >= len(self.chain):
                value["create_time"] += 1
            reads.append(pid)
            return value

        with self.assertRaisesRegex(CacheError, "Ancestry changed"):
            sample.validate_parent(self.attempt, reader=reused, worker_pid=33)

    @unittest.skipUnless(sys.platform == "win32", "Observed Windows venv launcher regression")
    def test_actual_windows_venv_subprocess_preserves_guards_and_blocks_replay(self):
        self.assertNotEqual(Path(sys.executable).resolve(), Path(sys._base_executable).resolve(),
                            "Run this regression with the isolated CPU venv Python")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="guard-probe-", dir=ARTIFACTS) as directory:
            root = Path(directory)
            allocation, review = root / "allocation.json", root / "review.json"
            sample.write_new(allocation, {"scope": "LAUNCHER_PROBE_ONLY_NO_IMAGES"})
            sample.write_new(review, {"scope": "LAUNCHER_PROBE_ONLY_NO_IMAGES"})
            token = "one-use-metadata-probe"
            child = (
                f"import sys,json; sys.path.insert(0,{str(sample.ROOT)!r}); "
                "from tools import hc15_cache_sample_v2 as m; "
                f"proof=m.claim_worker({str(allocation)!r},{str(review)!r},{token!r}); "
                "blocked=False\n"
                "try:\n"
                f" m.claim_worker({str(allocation)!r},{str(review)!r},{token!r})\n"
                "except FileExistsError:\n blocked=True\n"
                "print(json.dumps({'proof':proof,'replay_blocked':blocked,'native_backend_imported':"
                "any(v in sys.modules for v in ('torch','numpy','pydicom','cv2'))}))"
            )
            argv = [sys.executable, "-I", "-c", child]
            attempt = {"schema": "hc15-cache-sample-attempt-v2", "token": token,
                       "allocation_sha256": digest_file(allocation), "review_sha256": digest_file(review),
                       "parent_identity": sample.process_identity(os.getpid()),
                       "launcher_policy": self.policy, "worker_argv": argv,
                       "started_unix": time.time(), "deadline_unix": time.time() + 15}
            sample.write_new(root / "attempt.json", attempt)
            result = subprocess.run(argv, capture_output=True, text=True, timeout=15, cwd=sample.ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            proof = json.loads(result.stdout)
            self.assertEqual(len(proof["proof"]["chain"]), 3)
            self.assertTrue(proof["replay_blocked"])
            self.assertFalse(proof["native_backend_imported"])
            self.assertTrue((root / "worker.claim").exists())
            # Keep the process metadata as evidence, no sample pixels/artifacts.
            proof_path = ARTIFACTS / ("guard-proof-" + str(time.time_ns()) + ".json")
            sample.write_new(proof_path, proof)

    def test_v2_allocate_run_and_worker_remain_disabled_before_io(self):
        self.assertFalse(sample.SAMPLE_EXECUTION_ENABLED)
        calls = [lambda: sample.allocate(None, None, None, None),
                 lambda: sample.run(None, None), lambda: sample.worker(None, None, None)]
        for call in calls:
            with self.assertRaisesRegex(CacheError, "disabled pending"):
                call()


if __name__ == "__main__":
    unittest.main()
