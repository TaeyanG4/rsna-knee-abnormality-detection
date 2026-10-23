"""Small HC15 admission and epoch-resume contract tests; never open real pixels.

Default invocation is stdlib-only. Numerical tests are opt-in and CPU-only:
  HC15_RUN_CPU_STATE_TESTS=1 CUDA_VISIBLE_DEVICES=-1 python -m unittest ...
The orchestration owner must allocate the numerical CPU lane before that command.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from training.runners import hc15_convnext as runner


class AdmissionTests(unittest.TestCase):
    def test_generic_ready_review_cannot_bypass_unimplemented_evidence_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "admission.json"
            path.write_text(json.dumps({"schema": "hc15-execution-admission-v1", "status": "READY",
                                       "review": {"status": "READY", "independent": True,
                                                  "author": "builder", "reviewer": "reviewer"}}))
            with self.assertRaisesRegex(runner.ContractError, "not implemented for pilot/CPU/cost/decision"):
                runner.execution_admission(argparse.Namespace(admission=str(path)))

    def test_prepare_split_guards_survive_optimized_python(self):
        result = subprocess.run(
            [sys.executable, "-O", "-c",
             "from tools.open_recipe_prepare import validate_unique_split; "
             "validate_unique_split(['same'], ['same'], 1, 1)"],
            cwd=runner.ROOT, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Training/validation UID overlap", result.stderr)

    def test_missing_admission_blocks_before_torch_import(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from training.runners import hc15_convnext as r; "
             "status=r.main(['train','--manifest','missing.json','--manifest-sha256','x'*64]); "
             "assert status == 2; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            cwd=runner.ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("admission JSON is absent", result.stderr)

    def test_lower_level_gpu_entrypoints_require_admission_capability(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from training.runners import hc15_convnext as r; "
             "exec(\"for call in (r.configure_cuda, lambda: r.predict_final(None,None,[],None,0)):\\n"
             " try: call()\\n except r.ContractError: pass\\n else: raise AssertionError('unblocked')\"); "
             "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            cwd=runner.ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_recipe_and_scheduler_match_registered_update_budget(self):
        self.assertEqual((runner.UPDATES_PER_EPOCH, runner.TOTAL_UPDATES), (431, 6034))
        self.assertEqual(runner.RECIPE["microbatch"] * runner.RECIPE["accumulation"], 8)
        self.assertEqual(runner.RECIPE["num_workers"], 0)
        for step in (0, 1, 430, 431, 6033, 6034):
            expected = min(1, (step + 1) / 431) * .5 * (1 + math.cos(math.pi * min(1, step / 6034)))
            self.assertEqual(runner.scheduler_multiplier(step), expected)
        self.assertEqual(runner.scheduler_multiplier(6034), 0.)

    def test_path_escape_and_invalid_npy_magic_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(runner.ContractError):
                runner.path_in(temporary, "../outside.npy")
            path = Path(temporary) / "bad.npy"
            path.write_bytes(b"not-an-npy")
            with self.assertRaises(runner.ContractError):
                runner.npy_header(path)

    def test_cache_contract_rejects_old_geometry_and_missing_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            series_csv = root / "series.csv"
            series_csv.write_text("StudyInstanceUID,SeriesInstanceUID,Fat_Suppression,Anatomical_Plane\n"
                                  "study,series,1,Sagittal\n", encoding="utf-8")
            # A valid uint8 NPY assembled using only the stdlib; no numerical import.
            header = repr({"descr": "|u1", "fortran_order": False, "shape": (3, 384, 384)}).encode()
            header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
            pixels = root / "series.npy"
            pixels.write_bytes(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + b"\0" * (3 * 384 * 384))
            cache = {"schema": "hc15-cache-v1", "complete": True, "source_manifest_sha256": "manifest",
                     "adapter_sha256": "adapter", "contract": {"size": 384, "mm_per_pixel": .4,
                        "max_slices": 64, "dtype": "uint8", "slots": 6},
                     "series": {"series": {"n": 3, "decoded_count": 3, "path": "series.npy",
                         "bytes": pixels.stat().st_size, "sha256": runner.sha256(pixels)}}}
            cache_path = root / "cache.json"
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            manifest = {"generated": [{"path": "series.csv", "sha256": runner.sha256(series_csv)}]}
            cohort = {"train_uids": ["study"], "val_uids": []}
            rows, counts, files = runner.validate_cache_metadata(cache_path, root, manifest, cohort,
                                                                  "manifest", "adapter", root)
            self.assertEqual((len(rows), counts["series"], len(files)), (1, 3, 1))
            with self.assertRaisesRegex(runner.ContractError, "no valid decoded"):
                runner.validate_cache_metadata(cache_path, root, manifest,
                    {"train_uids": ["study"], "val_uids": ["orphan"]}, "manifest", "adapter", root)
            cache["contract"]["slots"] = 5
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            with self.assertRaisesRegex(runner.ContractError, "geometry differs"):
                runner.validate_cache_metadata(cache_path, root, manifest, cohort,
                                                "manifest", "adapter", root)

    def test_inline_nonindependent_review_cannot_admit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "admission.json"
            path.write_text(json.dumps({"schema": "hc15-execution-admission-v1", "status": "READY",
                                       "review": {"status": "READY", "independent": False}}))
            args = argparse.Namespace(admission=str(path))
            with self.assertRaisesRegex(runner.ContractError, "Independent readiness"):
                runner.execution_admission(args)


@unittest.skipUnless(os.environ.get("HC15_RUN_CPU_STATE_TESTS") == "1", "CPU lane must be explicitly allocated")
class NumericalStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise RuntimeError("Numerical tests require CUDA_VISIBLE_DEVICES=-1")
        global np, torch
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def initialize(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.02)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: runner.scheduler_multiplier(step, warm=2, total=6))
        generator = torch.Generator().manual_seed(seed)
        return model, ema, optimizer, scheduler, generator

    def run_epoch(self, objects, step):
        model, ema, optimizer, scheduler, generator = objects
        model.train()
        order, batches = runner.epoch_order(17, generator)
        traces = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                # Exercise all three CPU RNGs alongside the order generator.
                x = torch.tensor([[index / 17, random.random(), np.random.random()]], dtype=torch.float32)
                x += torch.rand_like(x) * .01
                y = torch.tensor([[index % 2, (index + 1) % 2]], dtype=torch.float32)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y) / 8
                loss.backward()
                traces.append((index, x.clone(), float(loss)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            runner.ema_update(ema, model, step)
        return step, order, traces

    def assert_tree_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_checkpoint_epoch_resume_exact_cpu_state_and_rng(self):
        first = self.initialize(42)
        step, order, _ = self.run_epoch(first, 0)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "state.pt"
            digest = runner.save_boundary(checkpoint, *first, epoch_next=1, step=step,
                                          last_order=order, bindings={"synthetic": "only"}, count=17)
            expected_step, expected_order, expected_trace = self.run_epoch(first, step)
            expected_states = [runner.cpu_copy(value.state_dict()) for value in first[:4]]
            expected_rng = runner.capture_rng()
            resumed = self.initialize(999)  # The restore must undo this unrelated RNG consumption.
            state = runner.load_boundary(checkpoint, digest, {"synthetic": "only"}, count=17)
            epoch, restored_step = runner.restore_boundary(state, *resumed)
            self.assertEqual((epoch, restored_step), (1, 2))
            actual_step, actual_order, actual_trace = self.run_epoch(resumed, restored_step)
            self.assertEqual(actual_step, expected_step)
            self.assertEqual(actual_order, expected_order)
            self.assert_tree_equal(actual_trace, expected_trace)
            for value, expected in zip(resumed[:4], expected_states):
                self.assert_tree_equal(value.state_dict(), expected)
            self.assert_tree_equal(runner.capture_rng(), expected_rng)
            self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_rejects_partial_epoch_changed_binding_and_overwrite(self):
        objects = self.initialize(0)
        step, order, _ = self.run_epoch(objects, 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            with self.assertRaisesRegex(runner.ContractError, "epoch boundary"):
                runner.save_boundary(path, *objects, epoch_next=1, step=1,
                                     last_order=order, bindings={"x": "a"}, count=17)
            digest = runner.save_boundary(path, *objects, epoch_next=1, step=step,
                                          last_order=order, bindings={"x": "a"}, count=17)
            with self.assertRaisesRegex(runner.ContractError, "lineage"):
                runner.load_boundary(path, digest, {"x": "b"}, count=17)
            with self.assertRaisesRegex(runner.ContractError, "overwrite"):
                runner.save_boundary(path, *objects, epoch_next=1, step=step,
                                     last_order=order, bindings={"x": "a"}, count=17)

    def test_drop_last_is_at_effective_batch_not_microbatch(self):
        generator = torch.Generator().manual_seed(42)
        order, batches = runner.epoch_order(3455, generator)
        self.assertEqual(len(order), 3455)
        self.assertEqual(len(batches), 431)
        self.assertTrue(all(len(batch) == 8 for batch in batches))
        self.assertEqual([index for batch in batches for index in batch], order[:3448])
        self.assertEqual(len(set(order)), 3455)

    def test_pinned_generic_initialization_loads_strictly_on_cpu_without_download(self):
        manifest = runner.read_json(runner.ROOT / "artifacts/experiments/HC15_OPEN_RECIPE_PREP_V1/manifest.json")
        # Local independent generic weights only; no competition checkpoint/pixels.
        model = runner.build_offline_model(manifest["initialization"]["asset"])
        self.assertEqual(model.enc.num_features, 768)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_cache_loader_checks_hash_before_pixels_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.npy"
            np.save(path, np.zeros((3, 384, 384), dtype=np.uint8), allow_pickle=False)
            record = {"absolute_path": str(path), "n": 3, "bytes": path.stat().st_size,
                      "sha256": runner.sha256(path)}
            loader = runner.VerifiedSeriesLoader({"synthetic": record})
            self.assertEqual(loader("synthetic").shape, (3, 384, 384))
            # Explicit timestamp mutation avoids relying on filesystem resolution.
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            with self.assertRaisesRegex(runner.ContractError, "changed during execution"):
                loader("synthetic")
            bad = runner.VerifiedSeriesLoader({"synthetic": {**record, "sha256": "0" * 64}})
            with self.assertRaisesRegex(runner.ContractError, "SHA256"):
                bad("synthetic")


if __name__ == "__main__":
    unittest.main()
